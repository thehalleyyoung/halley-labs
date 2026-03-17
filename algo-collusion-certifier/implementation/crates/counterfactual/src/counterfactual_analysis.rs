//! High-level counterfactual analysis orchestration.
//!
//! Coordinates deviation enumeration, oracle queries, re-simulation,
//! and punishment detection into unified analysis reports.

use shared_types::{
    CollusionError, CollusionResult, ConfidenceInterval, GameConfig,
    MarketOutcome, OracleAccessLevel, PlayerId, Price, PriceTrajectory,
    Profit, RoundNumber, SimulationConfig, PValue, HypothesisTestResult,
    EvidenceBundle, EvidenceItem, EvidenceStrength,
};
use game_theory::NashEquilibrium;
use serde::{Deserialize, Serialize};

use crate::market_helper;

use crate::deviation::{
    DeviationEnumerator, DeviationEnumeratorConfig, DeviationProfile,
    DeviationResult, DeviationStrategy, DeviationCatalog, OptimalDeviation,
};
use crate::oracle::{
    DeviationOracle, Layer1Oracle, Layer2Oracle, CheckpointSchedule,
    CertifiedBound, create_oracle,
};
use crate::punishment::{
    PunishmentDetector, PunishmentDetectorConfig, PunishmentEvidence,
    PunishmentClassification,
};
use crate::resimulation::{
    ResimulationEngine, CounterfactualScenario, MonteCarloResult,
    confidence_from_resimulation,
};

// ── CounterfactualAnalyzerConfig ────────────────────────────────────────────

/// Configuration for the counterfactual analyzer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CounterfactualAnalyzerConfig {
    pub oracle_level: OracleAccessLevel,
    pub price_grid_size: usize,
    pub num_mc_samples: usize,
    pub significance_level: f64,
    pub epsilon_threshold: f64,
    pub num_permutations: usize,
    pub observation_window: usize,
    pub num_injections: usize,
    pub seed: Option<u64>,
}

impl Default for CounterfactualAnalyzerConfig {
    fn default() -> Self {
        Self {
            oracle_level: OracleAccessLevel::Layer1,
            price_grid_size: 50,
            num_mc_samples: 100,
            significance_level: 0.05,
            epsilon_threshold: 0.01,
            num_permutations: 999,
            observation_window: 20,
            num_injections: 10,
            seed: Some(42),
        }
    }
}

// ── CounterfactualAnalyzer ──────────────────────────────────────────────────

/// Orchestrate full counterfactual analysis.
pub struct CounterfactualAnalyzer {
    config: CounterfactualAnalyzerConfig,
    game_config: GameConfig,
    sim_config: SimulationConfig,
}

impl CounterfactualAnalyzer {
    pub fn new(
        config: CounterfactualAnalyzerConfig,
        game_config: GameConfig,
        sim_config: SimulationConfig,
    ) -> Self {
        Self { config, game_config, sim_config }
    }

    /// Layer 0 analysis: passive observation only, no oracle access.
    ///
    /// Estimates deviation incentives from observed prices using best-response
    /// computation. Cannot verify actual algorithm behavior.
    pub fn analyze_layer0(
        &self,
        trajectory: &PriceTrajectory,
    ) -> CollusionResult<CounterfactualReport> {
        let nash_eq = market_helper::compute_nash(&self.game_config)?;
        let collusive = market_helper::compute_collusive(&self.game_config)?;

        let n_players = trajectory.num_players;
        let price_bounds = market_helper::price_bounds_from_config(&self.game_config);
        let mut deviation_profile = DeviationProfile::new(n_players);

        // For each player, find optimal single-period deviation
        for player in 0..n_players {
            let num_sample_rounds = trajectory.len().min(20);
            let step = trajectory.len() / num_sample_rounds.max(1);

            for r_idx in 0..num_sample_rounds {
                let round = r_idx * step;
                if round >= trajectory.len() { break; }

                let opt = OptimalDeviation::find(
                    PlayerId(player),
                    trajectory,
                    RoundNumber(round),
                    &self.game_config,
                    price_bounds,
                    self.config.price_grid_size,
                )?;

                if let Some(result) = opt.result {
                    deviation_profile.add_result(result);
                }
            }
        }

        let self_enforcing = SelfEnforcingCheck::check(
            &deviation_profile,
            self.config.epsilon_threshold,
        );

        let ic = IncentiveCompatibility::check(
            trajectory,
            &self.game_config,
            price_bounds,
            self.config.price_grid_size,
        );

        Ok(CounterfactualReport {
            oracle_level: OracleAccessLevel::Layer0,
            deviation_profile,
            certified_bounds: Vec::new(),
            punishment_evidence: Vec::new(),
            self_enforcing,
            incentive_compatibility: ic,
            comparison: None,
            summary: "Layer 0 passive analysis complete. No oracle access.".to_string(),
        })
    }

    /// Layer 1 analysis: checkpoint-based oracle.
    ///
    /// Uses periodic checkpoints to restore algorithm state and compute
    /// deviation bounds with interpolation between checkpoints.
    pub fn analyze_layer1(
        &self,
        trajectory: &PriceTrajectory,
    ) -> CollusionResult<CounterfactualReport> {
        let nash_eq = market_helper::compute_nash(&self.game_config)?;
        let collusive = market_helper::compute_collusive(&self.game_config)?;

        let n_players = trajectory.num_players;
        let price_bounds = market_helper::price_bounds_from_config(&self.game_config);
        let mut oracle = create_oracle(
            OracleAccessLevel::Layer1,
            self.game_config.clone(),
            trajectory,
            Some(10_000),
        );

        let mut deviation_profile = DeviationProfile::new(n_players);
        let mut certified_bounds = Vec::new();

        // Enumerate and evaluate deviations via oracle
        let enum_config = DeviationEnumeratorConfig {
            price_min: price_bounds.0,
            price_max: price_bounds.1,
            coarse_grid_size: self.config.price_grid_size,
            ..Default::default()
        };
        let enumerator = DeviationEnumerator::new(enum_config, self.game_config.clone());

        let num_sample_rounds = trajectory.len().min(10);
        let step = trajectory.len() / num_sample_rounds.max(1);

        for player in 0..n_players {
            for r_idx in 0..num_sample_rounds {
                let round = r_idx * step;
                if round >= trajectory.len() { break; }

                let devs = enumerator.enumerate_deviations(trajectory, RoundNumber(round));
                let mut best_dev: Option<(DeviationStrategy, f64)> = None;

                for dev in devs.iter().take(20) {
                    let bound = oracle.query_deviation(dev, trajectory, &self.game_config)?;

                    let cert = CertifiedBound::from_deviation_bound(
                        &bound, RoundNumber(round), dev.deviation_price,
                        OracleAccessLevel::Layer1, oracle.queries_used(),
                    );
                    certified_bounds.push(cert);

                    let gain = bound.difference_interval.midpoint();
                    match &best_dev {
                        Some((_, best_gain)) if gain > *best_gain => {
                            best_dev = Some((dev.clone(), gain));
                        }
                        None => {
                            best_dev = Some((dev.clone(), gain));
                        }
                        _ => {}
                    }
                }

                if let Some((dev, gain)) = best_dev {
                    let outcome = &trajectory.outcomes[round];
                    let obs_payoff = outcome.profits[player].0;
                    let result = DeviationResult::new(PlayerId(player), dev, obs_payoff, obs_payoff + gain);
                    deviation_profile.add_result(result);
                }
            }
        }

        // Punishment detection
        let punishment_config = PunishmentDetectorConfig {
            alpha: self.config.significance_level,
            num_permutations: self.config.num_permutations,
            observation_window: self.config.observation_window,
            num_injections_per_player: self.config.num_injections.min(5),
            seed: self.config.seed,
            ..Default::default()
        };
        let detector = PunishmentDetector::new(punishment_config, self.game_config.clone());
        let punishment_evidence = detector.detect_all(trajectory, &nash_eq)?;

        let self_enforcing = SelfEnforcingCheck::check(
            &deviation_profile,
            self.config.epsilon_threshold,
        );

        let ic = IncentiveCompatibility::check(
            trajectory,
            &self.game_config,
            price_bounds,
            self.config.price_grid_size,
        );

        let num_certified = certified_bounds.len();
        Ok(CounterfactualReport {
            oracle_level: OracleAccessLevel::Layer1,
            deviation_profile,
            certified_bounds,
            punishment_evidence,
            self_enforcing,
            incentive_compatibility: ic,
            comparison: None,
            summary: format!(
                "Layer 1 checkpoint analysis: {} oracle queries, {} bounds certified.",
                oracle.queries_used(),
                num_certified,
            ),
        })
    }

    /// Layer 2 analysis: full rewind oracle with re-simulation.
    ///
    /// Full access to algorithm state for exact deviation bounds.
    /// Includes Monte Carlo re-simulation for confidence intervals.
    pub fn analyze_layer2(
        &self,
        trajectory: &PriceTrajectory,
    ) -> CollusionResult<CounterfactualReport> {
        let nash_eq = market_helper::compute_nash(&self.game_config)?;
        let collusive = market_helper::compute_collusive(&self.game_config)?;

        let n_players = trajectory.num_players;
        let price_bounds = market_helper::price_bounds_from_config(&self.game_config);
        let mut oracle = create_oracle(
            OracleAccessLevel::Layer2,
            self.game_config.clone(),
            trajectory,
            Some(50_000),
        );

        let mut deviation_profile = DeviationProfile::new(n_players);
        let mut certified_bounds = Vec::new();

        // Full re-simulation engine
        let resim_engine = ResimulationEngine::new(
            self.game_config.clone(),
            self.sim_config.clone(),
            self.config.num_mc_samples,
            self.config.seed.unwrap_or(42),
        );

        let num_sample_rounds = trajectory.len().min(20);
        let step = trajectory.len() / num_sample_rounds.max(1);

        for player in 0..n_players {
            for r_idx in 0..num_sample_rounds {
                let round = r_idx * step;
                if round >= trajectory.len() { break; }

                let outcome = &trajectory.outcomes[round];
                let obs_payoff = outcome.profits[player].0;

                // Find optimal deviation via oracle
                let opt = OptimalDeviation::find(
                    PlayerId(player),
                    trajectory,
                    RoundNumber(round),
                    &self.game_config,
                    price_bounds,
                    self.config.price_grid_size,
                )?;

                if let Some(ref result) = opt.result {
                    let strategy = &result.strategy;
                    let bound = oracle.query_deviation(strategy, trajectory, &self.game_config)?;

                    let cert = CertifiedBound::from_deviation_bound(
                        &bound, RoundNumber(round), strategy.deviation_price,
                        OracleAccessLevel::Layer2, oracle.queries_used(),
                    );
                    certified_bounds.push(cert);

                    // Monte Carlo re-simulation for CI
                    let scenario = CounterfactualScenario::single_period(
                        PlayerId(player), strategy.deviation_price, RoundNumber(round),
                    );
                    let mc_result = resim_engine.monte_carlo(&scenario, trajectory)?;

                    let dev_result = DeviationResult::new(
                        PlayerId(player),
                        strategy.clone(),
                        obs_payoff,
                        obs_payoff + mc_result.mean_difference,
                    ).with_ci(mc_result.confidence_interval)
                     .with_simulations(mc_result.num_samples);

                    deviation_profile.add_result(dev_result);
                }
            }
        }

        // Punishment detection
        let punishment_config = PunishmentDetectorConfig {
            alpha: self.config.significance_level,
            num_permutations: self.config.num_permutations,
            observation_window: self.config.observation_window,
            num_injections_per_player: self.config.num_injections,
            seed: self.config.seed,
            ..Default::default()
        };
        let detector = PunishmentDetector::new(punishment_config, self.game_config.clone());
        let punishment_evidence = detector.detect_all(trajectory, &nash_eq)?;

        let self_enforcing = SelfEnforcingCheck::check(
            &deviation_profile,
            self.config.epsilon_threshold,
        );

        let ic = IncentiveCompatibility::check(
            trajectory,
            &self.game_config,
            price_bounds,
            self.config.price_grid_size,
        );

        // Factual vs counterfactual comparison
        let comparison = if !deviation_profile.results.is_empty() {
            let factual: Vec<f64> = deviation_profile.results.iter()
                .map(|r| r.observed_payoff).collect();
            let counterfactual: Vec<f64> = deviation_profile.results.iter()
                .map(|r| r.deviation_payoff).collect();
            Some(CompareFactualCounterfactual::compare(
                &factual, &counterfactual, 0.95,
            ))
        } else {
            None
        };

        Ok(CounterfactualReport {
            oracle_level: OracleAccessLevel::Layer2,
            deviation_profile,
            certified_bounds,
            punishment_evidence,
            self_enforcing,
            incentive_compatibility: ic,
            comparison,
            summary: format!(
                "Layer 2 full rewind analysis: {} oracle queries, {} MC samples.",
                oracle.queries_used(),
                self.config.num_mc_samples,
            ),
        })
    }

    /// Run analysis at the configured oracle level.
    pub fn analyze(&self, trajectory: &PriceTrajectory) -> CollusionResult<CounterfactualReport> {
        match self.config.oracle_level {
            OracleAccessLevel::Layer0Passive => self.analyze_layer0(trajectory),
            OracleAccessLevel::Layer1Checkpoint => self.analyze_layer1(trajectory),
            OracleAccessLevel::Layer2FullRewind => self.analyze_layer2(trajectory),
        }
    }
}

// ── CompareFactualCounterfactual ────────────────────────────────────────────

/// Statistical comparison of factual vs counterfactual outcomes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompareFactualCounterfactual {
    pub mean_factual: f64,
    pub mean_counterfactual: f64,
    pub mean_difference: f64,
    pub difference_ci: ConfidenceInterval,
    pub ks_statistic: Option<f64>,
    pub ks_p_value: Option<f64>,
    pub num_observations: usize,
}

impl CompareFactualCounterfactual {
    /// Compare factual and counterfactual payoff distributions.
    pub fn compare(
        factual: &[f64],
        counterfactual: &[f64],
        confidence_level: f64,
    ) -> Self {
        let n = factual.len().min(counterfactual.len());
        if n == 0 {
            return Self {
                mean_factual: 0.0,
                mean_counterfactual: 0.0,
                mean_difference: 0.0,
                difference_ci: ConfidenceInterval::new(0.0, 0.0, confidence_level, 0.0),
                ks_statistic: None,
                ks_p_value: None,
                num_observations: 0,
            };
        }

        let mean_f = factual.iter().sum::<f64>() / n as f64;
        let mean_c = counterfactual.iter().sum::<f64>() / n as f64;
        let mean_diff = mean_c - mean_f;

        // Paired differences CI
        let diffs: Vec<f64> = factual.iter().zip(counterfactual.iter())
            .map(|(f, c)| c - f)
            .collect();
        let diff_var = if diffs.len() > 1 {
            let m = diffs.iter().sum::<f64>() / diffs.len() as f64;
            diffs.iter().map(|d| (d - m).powi(2)).sum::<f64>() / (diffs.len() - 1) as f64
        } else {
            0.0
        };
        let se = (diff_var / diffs.len() as f64).sqrt();
        let z = if confidence_level >= 0.99 { 2.576 } else { 1.96 };

        let ci = ConfidenceInterval::new(
            mean_diff - z * se,
            mean_diff + z * se,
            confidence_level,
            mean_diff,
        );

        // KS test (simplified)
        let ks = ks_two_sample(factual, counterfactual);

        Self {
            mean_factual: mean_f,
            mean_counterfactual: mean_c,
            mean_difference: mean_diff,
            difference_ci: ci,
            ks_statistic: Some(ks.0),
            ks_p_value: Some(ks.1),
            num_observations: n,
        }
    }

    pub fn is_significant(&self, alpha: f64) -> bool {
        self.ks_p_value.map(|p| p < alpha).unwrap_or(false)
    }
}

/// Simplified two-sample KS test returning (statistic, p_value).
fn ks_two_sample(a: &[f64], b: &[f64]) -> (f64, f64) {
    if a.is_empty() || b.is_empty() {
        return (0.0, 1.0);
    }

    let mut a_sorted = a.to_vec();
    let mut b_sorted = b.to_vec();
    a_sorted.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    b_sorted.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));

    let n_a = a_sorted.len() as f64;
    let n_b = b_sorted.len() as f64;

    let mut all_vals: Vec<f64> = a_sorted.iter().chain(b_sorted.iter()).copied().collect();
    all_vals.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    all_vals.dedup();

    let mut max_diff: f64 = 0.0;
    for &v in &all_vals {
        let cdf_a = a_sorted.partition_point(|&x| x <= v) as f64 / n_a;
        let cdf_b = b_sorted.partition_point(|&x| x <= v) as f64 / n_b;
        max_diff = max_diff.max((cdf_a - cdf_b).abs());
    }

    let n_eff = (n_a * n_b) / (n_a + n_b);
    let lambda = (n_eff.sqrt() + 0.12 + 0.11 / n_eff.sqrt()) * max_diff;

    // Approximate p-value
    let p = if lambda > 3.0 {
        0.0
    } else {
        let mut sum = 0.0;
        for k in 1..=100 {
            let kf = k as f64;
            let sign = if k % 2 == 1 { 1.0 } else { -1.0 };
            sum += sign * (-2.0 * kf * kf * lambda * lambda).exp();
        }
        (2.0 * sum).max(0.0).min(1.0)
    };

    (max_diff, p)
}

// ── SelfEnforcingCheck ──────────────────────────────────────────────────────

/// Check whether the observed strategy profile is self-enforcing (epsilon-Nash).
///
/// A profile is self-enforcing if no player has a profitable deviation with
/// payoff improvement > epsilon.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfEnforcingCheck {
    pub is_self_enforcing: bool,
    pub epsilon: f64,
    pub max_deviation_gain: f64,
    pub per_player_max_gain: Vec<f64>,
}

impl SelfEnforcingCheck {
    pub fn check(profile: &DeviationProfile, epsilon: f64) -> Self {
        let per_player_max: Vec<f64> = (0..profile.num_players)
            .map(|player| {
                profile.for_player(PlayerId(player))
                    .iter()
                    .map(|r| r.payoff_difference.max(0.0))
                    .fold(0.0f64, f64::max)
            })
            .collect();

        let max_gain = per_player_max.iter().copied().fold(0.0f64, f64::max);

        Self {
            is_self_enforcing: max_gain <= epsilon,
            epsilon,
            max_deviation_gain: max_gain,
            per_player_max_gain: per_player_max,
        }
    }

    /// Is the profile ε-Nash?
    pub fn is_epsilon_nash(&self) -> bool {
        self.is_self_enforcing
    }
}

// ── IncentiveCompatibility ──────────────────────────────────────────────────

/// Check incentive compatibility constraints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncentiveCompatibility {
    /// Whether IC holds for all players.
    pub ic_satisfied: bool,
    /// Per-player deviation incentive.
    pub player_incentives: Vec<f64>,
    /// Maximum deviation incentive across players.
    pub max_incentive: f64,
    /// Folk theorem minimum discount factor.
    pub folk_theorem_delta: Option<f64>,
}

impl IncentiveCompatibility {
    pub fn check(
        trajectory: &PriceTrajectory,
        game_config: &GameConfig,
        price_bounds: (Price, Price),
        grid_size: usize,
    ) -> Self {
        let n_players = trajectory.num_players;
        let incentives: Vec<f64> = (0..n_players)
            .map(|player| {
                crate::deviation::deviation_incentive(
                    PlayerId(player), trajectory, game_config, price_bounds, grid_size,
                )
            })
            .collect();

        let max_inc = incentives.iter().copied().fold(0.0f64, f64::max);

        // Folk theorem threshold
        let folk_delta = if let (Ok(nash), Ok(collusive)) = (
            market_helper::compute_nash(game_config),
            market_helper::compute_collusive(game_config),
        ) {
            let nash_profit = nash.payoffs[0];
            let collusive_profit = collusive.payoffs[0];

            // Estimate deviation profit from max incentive
            let dev_profit = nash_profit + max_inc;
            Some(market_helper::folk_theorem_min_discount(nash_profit, collusive_profit, dev_profit))
        } else {
            None
        };

        Self {
            ic_satisfied: max_inc < 0.01,
            player_incentives: incentives,
            max_incentive: max_inc,
            folk_theorem_delta: folk_delta,
        }
    }

    /// Whether the current discount factor supports collusion.
    pub fn collusion_sustainable(&self, discount_factor: f64) -> bool {
        self.folk_theorem_delta
            .map(|delta| discount_factor >= delta)
            .unwrap_or(false)
    }
}

// ── CounterfactualReport ────────────────────────────────────────────────────

/// Structured output of counterfactual analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CounterfactualReport {
    pub oracle_level: OracleAccessLevel,
    pub deviation_profile: DeviationProfile,
    pub certified_bounds: Vec<CertifiedBound>,
    pub punishment_evidence: Vec<PunishmentEvidence>,
    pub self_enforcing: SelfEnforcingCheck,
    pub incentive_compatibility: IncentiveCompatibility,
    pub comparison: Option<CompareFactualCounterfactual>,
    pub summary: String,
}

impl CounterfactualReport {
    /// Overall assessment of collusion evidence from counterfactual analysis.
    pub fn collusion_assessment(&self) -> CollusionAssessment {
        let no_profitable_deviations = !self.deviation_profile.any_profitable
            || self.self_enforcing.is_self_enforcing;

        let punishment_detected = self.punishment_evidence.iter()
            .any(|e| e.punishment_detected(0.05));

        let strong_punishment = self.punishment_evidence.iter()
            .any(|e| matches!(
                e.classification,
                PunishmentClassification::SevereRetaliation | PunishmentClassification::GrimPunishment
            ));

        let ic_violated = self.incentive_compatibility.max_incentive > 0.1;

        let score = if no_profitable_deviations && strong_punishment {
            0.9
        } else if no_profitable_deviations && punishment_detected {
            0.7
        } else if punishment_detected {
            0.5
        } else if ic_violated {
            0.3
        } else {
            0.1
        };

        CollusionAssessment {
            collusion_score: score,
            self_enforcing: self.self_enforcing.is_self_enforcing,
            punishment_detected,
            strong_punishment,
            max_deviation_gain: self.deviation_profile.max_deviation_gain,
        }
    }

    /// Convert to evidence bundle.
    pub fn to_evidence(&self) -> EvidenceBundle {
        let mut bundle = EvidenceBundle::new(Vec::new(), "Counterfactual analysis evidence");

        let _strength = if self.self_enforcing.is_self_enforcing {
            EvidenceStrength::Strong
        } else if self.deviation_profile.any_profitable {
            EvidenceStrength::Weak
        } else {
            EvidenceStrength::Moderate
        };

        bundle.add(EvidenceItem::EquilibriumComputation {
            description: format!(
                "Counterfactual analysis ({:?}): max deviation gain = {:.4}, self-enforcing = {}",
                self.oracle_level,
                self.deviation_profile.max_deviation_gain,
                self.self_enforcing.is_self_enforcing,
            ),
            nash_price: 0.0,
            collusive_price: 0.0,
            observed_price: 0.0,
            deviation_incentive: self.deviation_profile.max_deviation_gain,
            sustainable: self.self_enforcing.is_self_enforcing,
        });

        for evidence in &self.punishment_evidence {
            let _p_strength = match evidence.classification {
                PunishmentClassification::GrimPunishment => EvidenceStrength::Decisive,
                PunishmentClassification::SevereRetaliation => EvidenceStrength::Strong,
                PunishmentClassification::MildRetaliation => EvidenceStrength::Moderate,
                PunishmentClassification::NoPunishment => EvidenceStrength::Weak,
            };

            bundle.add(EvidenceItem::PunishmentDetection {
                description: format!(
                    "Player {:?} punishment: {} (payoff drop = {:.4})",
                    evidence.player, evidence.classification, evidence.aggregate_metrics.payoff_drop,
                ),
                punishment_type: format!("{}", evidence.classification),
                severity: evidence.aggregate_metrics.payoff_drop,
                duration: evidence.num_experiments,
                recovery_rounds: Some(0),
            });
        }

        bundle
    }
}

/// High-level collusion assessment from counterfactual analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollusionAssessment {
    pub collusion_score: f64,
    pub self_enforcing: bool,
    pub punishment_detected: bool,
    pub strong_punishment: bool,
    pub max_deviation_gain: f64,
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use shared_types::{
        DemandSystem, MarketType, Cost, Quantity, PlayerAction,
        AlgorithmType, AlgorithmConfig, EvaluationMode,
    };

    fn make_trajectory(n_rounds: usize, n_players: usize, price: Price) -> PriceTrajectory {
        let outcomes: Vec<MarketOutcome> = (0..n_rounds)
            .map(|r| {
                let actions: Vec<PlayerAction> = (0..n_players)
                    .map(|i| PlayerAction::new(PlayerId(i), price))
                    .collect();
                let prices = vec![price; n_players];
                let quantities = vec![Quantity(1.0); n_players];
                let profits = vec![Profit((price.0 - 1.0) * 1.0); n_players];
                MarketOutcome::new(RoundNumber(r), actions, prices, quantities, profits)
            })
            .collect();
        PriceTrajectory::new(outcomes, MarketType::Bertrand, n_players, AlgorithmType::QLearning, 42)
    }

    fn make_configs() -> (CounterfactualAnalyzerConfig, GameConfig, SimulationConfig) {
        let ac = CounterfactualAnalyzerConfig {
            price_grid_size: 20,
            num_mc_samples: 10,
            num_permutations: 99,
            num_injections: 3,
            observation_window: 5,
            ..Default::default()
        };
        let gc = GameConfig {
            num_players: 2,
            discount_factor: 0.95,
            marginal_costs: vec![Cost(1.0), Cost(1.0)],
            demand_system: DemandSystem::Linear {
                max_quantity: 10.0,
                slope: 1.0,
            },
            market_type: MarketType::Bertrand,
            price_grid: None,
            max_rounds: 1000,
            description: String::new(),
        };
        let sc = SimulationConfig::new(
            gc.clone(),
            AlgorithmConfig::new(AlgorithmType::QLearning),
            EvaluationMode::Smoke,
        );
        (ac, gc, sc)
    }

    #[test]
    fn test_analyzer_layer0() {
        let (ac, gc, sc) = make_configs();
        let mut config = ac;
        config.oracle_level = OracleAccessLevel::Layer0;
        let analyzer = CounterfactualAnalyzer::new(config, gc, sc);
        let trajectory = make_trajectory(100, 2, Price(5.0));
        let result = analyzer.analyze_layer0(&trajectory);
        assert!(result.is_ok());
        let report = result.unwrap();
        assert_eq!(report.oracle_level, OracleAccessLevel::Layer0);
    }

    #[test]
    fn test_analyzer_layer1() {
        let (ac, gc, sc) = make_configs();
        let mut config = ac;
        config.oracle_level = OracleAccessLevel::Layer1;
        let analyzer = CounterfactualAnalyzer::new(config, gc, sc);
        let trajectory = make_trajectory(100, 2, Price(5.0));
        let result = analyzer.analyze_layer1(&trajectory);
        assert!(result.is_ok());
    }

    #[test]
    fn test_analyzer_layer2() {
        let (ac, gc, sc) = make_configs();
        let mut config = ac;
        config.oracle_level = OracleAccessLevel::Layer2;
        config.num_mc_samples = 5;
        let analyzer = CounterfactualAnalyzer::new(config, gc, sc);
        let trajectory = make_trajectory(100, 2, Price(5.0));
        let result = analyzer.analyze_layer2(&trajectory);
        assert!(result.is_ok());
    }

    #[test]
    fn test_compare_factual_counterfactual() {
        let factual = vec![10.0, 11.0, 9.0, 10.5, 10.2];
        let counterfactual = vec![12.0, 13.0, 11.0, 12.5, 12.2];
        let comp = CompareFactualCounterfactual::compare(&factual, &counterfactual, 0.95);
        assert!(comp.mean_difference > 0.0);
        assert_eq!(comp.num_observations, 5);
    }

    #[test]
    fn test_compare_same_distributions() {
        let data = vec![10.0, 11.0, 9.0, 10.5, 10.2];
        let comp = CompareFactualCounterfactual::compare(&data, &data, 0.95);
        assert!(comp.mean_difference.abs() < 1e-10);
    }

    #[test]
    fn test_self_enforcing_check() {
        let mut profile = DeviationProfile::new(2);
        let s = DeviationStrategy::single_period(PlayerId(0), Price(3.0), Price(5.0), RoundNumber(0));
        profile.add_result(DeviationResult::new(PlayerId(0), s, 10.0, 10.005));
        let check = SelfEnforcingCheck::check(&profile, 0.01);
        assert!(check.is_self_enforcing);
        assert!(check.is_epsilon_nash());
    }

    #[test]
    fn test_self_enforcing_violated() {
        let mut profile = DeviationProfile::new(2);
        let s = DeviationStrategy::single_period(PlayerId(0), Price(3.0), Price(5.0), RoundNumber(0));
        profile.add_result(DeviationResult::new(PlayerId(0), s, 10.0, 15.0));
        let check = SelfEnforcingCheck::check(&profile, 0.01);
        assert!(!check.is_self_enforcing);
    }

    #[test]
    fn test_incentive_compatibility() {
        let trajectory = make_trajectory(20, 2, Price(5.0));
        let gc = GameConfig {
            num_players: 2,
            discount_factor: 0.95,
            marginal_costs: vec![Cost(1.0), Cost(1.0)],
            demand_system: DemandSystem::Linear {
                max_quantity: 10.0,
                slope: 1.0,
            },
            market_type: MarketType::Bertrand,
            price_grid: None,
            max_rounds: 1000,
            description: String::new(),
        };
        let ic = IncentiveCompatibility::check(&trajectory, &gc, (Price(0.0), Price(10.0)), 20);
        assert_eq!(ic.player_incentives.len(), 2);
    }

    #[test]
    fn test_collusion_assessment() {
        let mut profile = DeviationProfile::new(2);
        let s = DeviationStrategy::single_period(PlayerId(0), Price(3.0), Price(5.0), RoundNumber(0));
        profile.add_result(DeviationResult::new(PlayerId(0), s, 10.0, 10.005));

        let report = CounterfactualReport {
            oracle_level: OracleAccessLevel::Layer0,
            deviation_profile: profile,
            certified_bounds: Vec::new(),
            punishment_evidence: Vec::new(),
            self_enforcing: SelfEnforcingCheck {
                is_self_enforcing: true,
                epsilon: 0.01,
                max_deviation_gain: 0.005,
                per_player_max_gain: vec![0.005, 0.0],
            },
            incentive_compatibility: IncentiveCompatibility {
                ic_satisfied: true,
                player_incentives: vec![0.001, 0.0],
                max_incentive: 0.001,
                folk_theorem_delta: Some(0.5),
            },
            comparison: None,
            summary: "test".to_string(),
        };
        let assessment = report.collusion_assessment();
        assert!(assessment.collusion_score >= 0.0);
        assert!(assessment.collusion_score <= 1.0);
    }

    #[test]
    fn test_report_to_evidence() {
        let profile = DeviationProfile::new(2);
        let report = CounterfactualReport {
            oracle_level: OracleAccessLevel::Layer0,
            deviation_profile: profile,
            certified_bounds: Vec::new(),
            punishment_evidence: Vec::new(),
            self_enforcing: SelfEnforcingCheck {
                is_self_enforcing: false,
                epsilon: 0.01,
                max_deviation_gain: 1.0,
                per_player_max_gain: vec![1.0, 0.5],
            },
            incentive_compatibility: IncentiveCompatibility {
                ic_satisfied: false,
                player_incentives: vec![1.0, 0.5],
                max_incentive: 1.0,
                folk_theorem_delta: None,
            },
            comparison: None,
            summary: "test".to_string(),
        };
        let evidence = report.to_evidence();
        assert!(!evidence.items.is_empty());
    }

    #[test]
    fn test_ks_two_sample_identical() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (stat, p) = ks_two_sample(&a, &a);
        assert!(stat < 0.01);
    }

    #[test]
    fn test_ks_two_sample_different() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![6.0, 7.0, 8.0, 9.0, 10.0];
        let (stat, _p) = ks_two_sample(&a, &b);
        assert!(stat > 0.5);
    }

    #[test]
    fn test_analyze_dispatches() {
        let (ac, gc, sc) = make_configs();
        let mut config = ac;
        config.oracle_level = OracleAccessLevel::Layer0;
        let analyzer = CounterfactualAnalyzer::new(config, gc, sc);
        let trajectory = make_trajectory(50, 2, Price(5.0));
        let result = analyzer.analyze(&trajectory);
        assert!(result.is_ok());
    }

    #[test]
    fn test_incentive_compatibility_sustainability() {
        let ic = IncentiveCompatibility {
            ic_satisfied: false,
            player_incentives: vec![0.5, 0.3],
            max_incentive: 0.5,
            folk_theorem_delta: Some(0.6),
        };
        assert!(ic.collusion_sustainable(0.95));
        assert!(!ic.collusion_sustainable(0.5));
    }
}
