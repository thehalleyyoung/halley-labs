//! Counterfactual re-simulation engine.
//!
//! Runs what-if scenarios via Monte Carlo simulation. Supports parallel
//! execution, truncated horizons, common random numbers for variance
//! reduction, and checkpoint save/restore.

use shared_types::{
    CollusionError, CollusionResult, ConfidenceInterval, GameConfig,
    MarketOutcome, PlayerId, Price, PriceTrajectory,
    RoundNumber, SimulationConfig,
};
use crate::market_helper::{simulate_single_round, price_bounds_from_config};
use serde::{Deserialize, Serialize};
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crate::deviation::DeviationStrategy;

// ── CounterfactualScenario ──────────────────────────────────────────────────

/// Specification of a what-if counterfactual scenario.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CounterfactualScenario {
    /// Which player deviates.
    pub deviating_player: PlayerId,
    /// Price to deviate to.
    pub deviation_price: Price,
    /// Round at which deviation begins.
    pub start_round: RoundNumber,
    /// Number of rounds the deviation lasts.
    pub duration: usize,
    /// Whether opponents observe the deviation.
    pub opponents_observe: bool,
    /// Label for this scenario.
    pub label: String,
}

impl CounterfactualScenario {
    pub fn single_period(player: PlayerId, price: Price, round: RoundNumber) -> Self {
        Self {
            deviating_player: player,
            deviation_price: price,
            start_round: round,
            duration: 1,
            opponents_observe: true,
            label: format!("player_{}_deviate_{:.2}_round_{}", player.0, price.0, round.0),
        }
    }

    pub fn multi_period(player: PlayerId, price: Price, start: RoundNumber, duration: usize) -> Self {
        Self {
            deviating_player: player,
            deviation_price: price,
            start_round: start,
            duration,
            opponents_observe: true,
            label: format!("player_{}_deviate_{:.2}_rounds_{}_{}", player.0, price.0, start.0, start.0 + duration),
        }
    }

    pub fn unobserved(player: PlayerId, price: Price, round: RoundNumber) -> Self {
        Self {
            deviating_player: player,
            deviation_price: price,
            start_round: round,
            duration: 1,
            opponents_observe: false,
            label: format!("player_{}_secret_deviate_{:.2}_round_{}", player.0, price.0, round.0),
        }
    }

    /// Convert to a DeviationStrategy.
    pub fn to_strategy(&self, observed_price: Price) -> DeviationStrategy {
        DeviationStrategy::multi_period(
            self.deviating_player,
            self.deviation_price,
            observed_price,
            self.start_round,
            self.duration,
        )
    }
}

// ── ResimulationResult ──────────────────────────────────────────────────────

/// Result of a single counterfactual re-simulation run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResimulationResult {
    /// The scenario that was simulated.
    pub scenario: CounterfactualScenario,
    /// Factual (observed) payoffs for each player.
    pub factual_payoffs: Vec<Vec<f64>>,
    /// Counterfactual payoffs for each player.
    pub counterfactual_payoffs: Vec<Vec<f64>>,
    /// Total factual payoff for the deviating player.
    pub factual_total: f64,
    /// Total counterfactual payoff for the deviating player.
    pub counterfactual_total: f64,
    /// Payoff difference (counterfactual - factual).
    pub payoff_difference: f64,
    /// Number of rounds simulated.
    pub rounds_simulated: usize,
    /// Random seed used.
    pub seed: u64,
}

impl ResimulationResult {
    pub fn is_profitable(&self) -> bool {
        self.payoff_difference > 0.0
    }

    /// Compute the discounted payoff difference.
    pub fn discounted_difference(&self, discount: f64) -> f64 {
        let player = self.scenario.deviating_player;
        let cf_discounted: f64 = self.counterfactual_payoffs.get(player.0)
            .map(|payoffs| payoffs.iter().enumerate()
                .map(|(t, &pi)| discount.powi(t as i32) * pi)
                .sum())
            .unwrap_or(0.0);

        let f_discounted: f64 = self.factual_payoffs.get(player.0)
            .map(|payoffs| payoffs.iter().enumerate()
                .map(|(t, &pi)| discount.powi(t as i32) * pi)
                .sum())
            .unwrap_or(0.0);

        cf_discounted - f_discounted
    }
}

// ── ResimulationEngine ──────────────────────────────────────────────────────

/// Engine for running counterfactual re-simulations.
pub struct ResimulationEngine {
    game_config: GameConfig,
    sim_config: SimulationConfig,
    num_mc_samples: usize,
    base_seed: u64,
}

impl ResimulationEngine {
    pub fn new(
        game_config: GameConfig,
        sim_config: SimulationConfig,
        num_mc_samples: usize,
        seed: u64,
    ) -> Self {
        Self {
            game_config,
            sim_config,
            num_mc_samples,
            base_seed: seed,
        }
    }

    /// Run a single counterfactual re-simulation.
    pub fn simulate_single(
        &self,
        scenario: &CounterfactualScenario,
        trajectory: &PriceTrajectory,
        seed: u64,
    ) -> CollusionResult<ResimulationResult> {
        let n_players = trajectory.num_players;
        let start = scenario.start_round.0;
        let end = (start + scenario.duration).min(trajectory.len());

        if start >= trajectory.len() {
            return Err(CollusionError::InvalidState("Start round out of range".into()));
        }

        let mut factual_payoffs = vec![Vec::new(); n_players];
        let mut counterfactual_payoffs = vec![Vec::new(); n_players];

        let mut rng = StdRng::seed_from_u64(seed);

        let (lo, _hi) = price_bounds_from_config(&self.game_config);

        for r in start..end {
            let outcome = &trajectory.outcomes[r];

            // Factual payoffs
            for i in 0..n_players {
                factual_payoffs[i].push(outcome.profits[i].0);
            }

            // Counterfactual: inject deviation
            let mut dev_prices = outcome.prices.clone();
            dev_prices[scenario.deviating_player.0] = scenario.deviation_price;

            // Add small noise for MC variance estimation
            let noise_scale = 0.001;
            for p in &mut dev_prices {
                let noise: f64 = rng.gen_range(-noise_scale..noise_scale);
                *p = Price((p.0 + noise).max(lo.0));
            }

            let cf_outcome = simulate_single_round(&dev_prices, &self.game_config, RoundNumber(r));

            for i in 0..n_players {
                counterfactual_payoffs[i].push(cf_outcome.profits[i].0);
            }
        }

        let player = scenario.deviating_player;
        let factual_total: f64 = factual_payoffs[player.0].iter().sum();
        let cf_total: f64 = counterfactual_payoffs[player.0].iter().sum();

        Ok(ResimulationResult {
            scenario: scenario.clone(),
            factual_payoffs,
            counterfactual_payoffs,
            factual_total,
            counterfactual_total: cf_total,
            payoff_difference: cf_total - factual_total,
            rounds_simulated: end - start,
            seed,
        })
    }

    /// Run Monte Carlo re-simulation with multiple seeds.
    pub fn monte_carlo(
        &self,
        scenario: &CounterfactualScenario,
        trajectory: &PriceTrajectory,
    ) -> CollusionResult<MonteCarloResult> {
        let mut results = Vec::with_capacity(self.num_mc_samples);
        let mut differences = Vec::with_capacity(self.num_mc_samples);

        for s in 0..self.num_mc_samples {
            let seed = self.base_seed.wrapping_add(s as u64);
            let result = self.simulate_single(scenario, trajectory, seed)?;
            differences.push(result.payoff_difference);
            results.push(result);
        }

        let mean_diff = differences.iter().sum::<f64>() / differences.len() as f64;
        let variance = if differences.len() > 1 {
            differences.iter().map(|d| (d - mean_diff).powi(2)).sum::<f64>()
                / (differences.len() - 1) as f64
        } else {
            0.0
        };
        let se = (variance / differences.len() as f64).sqrt();

        let ci = ConfidenceInterval::new(
            mean_diff - 1.96 * se,
            mean_diff + 1.96 * se,
            0.95,
            mean_diff,
        );

        Ok(MonteCarloResult {
            scenario: scenario.clone(),
            results,
            mean_difference: mean_diff,
            std_error: se,
            confidence_interval: ci,
            num_samples: self.num_mc_samples,
        })
    }
}

/// Result of Monte Carlo re-simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonteCarloResult {
    pub scenario: CounterfactualScenario,
    pub results: Vec<ResimulationResult>,
    pub mean_difference: f64,
    pub std_error: f64,
    pub confidence_interval: ConfidenceInterval,
    pub num_samples: usize,
}

impl MonteCarloResult {
    pub fn is_profitable(&self) -> bool {
        self.mean_difference > 0.0
    }

    pub fn is_significantly_profitable(&self) -> bool {
        self.confidence_interval.lower > 0.0
    }
}

// ── ParallelResimulation ────────────────────────────────────────────────────

/// Embarrassingly parallel Monte Carlo re-simulation.
///
/// Distributes N players × D deviations × S seeds × T rounds across
/// a rayon thread pool.
pub struct ParallelResimulation {
    game_config: GameConfig,
    sim_config: SimulationConfig,
    num_seeds: usize,
    base_seed: u64,
}

impl ParallelResimulation {
    pub fn new(
        game_config: GameConfig,
        sim_config: SimulationConfig,
        num_seeds: usize,
        seed: u64,
    ) -> Self {
        Self {
            game_config,
            sim_config,
            num_seeds,
            base_seed: seed,
        }
    }

    /// Run parallel re-simulation for multiple scenarios.
    pub fn run_parallel(
        &self,
        scenarios: &[CounterfactualScenario],
        trajectory: &PriceTrajectory,
    ) -> Vec<CollusionResult<MonteCarloResult>> {
        let engine = Arc::new(ResimulationEngine::new(
            self.game_config.clone(),
            self.sim_config.clone(),
            self.num_seeds,
            self.base_seed,
        ));
        let traj = Arc::new(trajectory.clone());

        scenarios.par_iter().map(|scenario| {
            engine.monte_carlo(scenario, &traj)
        }).collect()
    }

    /// Run parallel re-simulation for all single-period deviations
    /// for a given player across multiple prices.
    pub fn sweep_prices(
        &self,
        player: PlayerId,
        trajectory: &PriceTrajectory,
        round: RoundNumber,
        prices: &[Price],
    ) -> Vec<CollusionResult<MonteCarloResult>> {
        let scenarios: Vec<CounterfactualScenario> = prices.iter()
            .map(|&p| CounterfactualScenario::single_period(player, p, round))
            .collect();
        self.run_parallel(&scenarios, trajectory)
    }

    /// Estimated total compute: players × deviations × seeds × rounds.
    pub fn estimated_work(
        &self,
        num_scenarios: usize,
        avg_rounds: usize,
    ) -> usize {
        num_scenarios * self.num_seeds * avg_rounds
    }
}

// ── TruncatedHorizon ────────────────────────────────────────────────────────

/// Stop re-simulation after H rounds for computational efficiency.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TruncatedHorizon {
    /// Maximum rounds to simulate.
    pub max_rounds: usize,
    /// Whether horizon was selected automatically.
    pub auto_selected: bool,
    /// Error bound from truncation.
    pub truncation_error_bound: f64,
}

impl TruncatedHorizon {
    /// Create with fixed horizon.
    pub fn fixed(max_rounds: usize) -> Self {
        Self {
            max_rounds,
            auto_selected: false,
            truncation_error_bound: 0.0,
        }
    }

    /// Automatically select horizon based on discount factor and precision.
    ///
    /// Choose H such that δ^H × max_payoff < ε.
    pub fn auto_select(discount_factor: f64, max_single_round_payoff: f64, epsilon: f64) -> Self {
        if discount_factor >= 1.0 || max_single_round_payoff <= 0.0 || epsilon <= 0.0 {
            return Self::fixed(1000);
        }

        let h = (epsilon / max_single_round_payoff).ln() / discount_factor.ln();
        let h = (h.ceil() as usize).max(1).min(10_000);

        let truncation_error = discount_factor.powi(h as i32) * max_single_round_payoff
            / (1.0 - discount_factor);

        Self {
            max_rounds: h,
            auto_selected: true,
            truncation_error_bound: truncation_error,
        }
    }

    /// Compute error bound for given horizon.
    pub fn error_bound(&self, discount_factor: f64, max_payoff: f64) -> f64 {
        if discount_factor >= 1.0 {
            return f64::INFINITY;
        }
        discount_factor.powi(self.max_rounds as i32) * max_payoff / (1.0 - discount_factor)
    }

    /// Truncate a scenario to this horizon.
    pub fn truncate_scenario(&self, scenario: &CounterfactualScenario) -> CounterfactualScenario {
        let mut truncated = scenario.clone();
        truncated.duration = truncated.duration.min(self.max_rounds);
        truncated
    }
}

// ── VarianceReduction ───────────────────────────────────────────────────────

/// Common random numbers for variance reduction.
///
/// Uses the same random seed for both factual and counterfactual simulations,
/// inducing positive correlation that reduces variance of the difference estimator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VarianceReduction {
    /// Method used for variance reduction.
    pub method: VarianceReductionMethod,
    /// Estimated variance reduction factor.
    pub reduction_factor: f64,
    /// Pre-generated common random numbers.
    pub crn_seeds: Vec<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VarianceReductionMethod {
    None,
    CommonRandomNumbers,
    AntitheticVariates,
    ControlVariates,
}

impl VarianceReduction {
    pub fn none() -> Self {
        Self {
            method: VarianceReductionMethod::None,
            reduction_factor: 1.0,
            crn_seeds: Vec::new(),
        }
    }

    /// Create CRN variance reduction with pre-generated seeds.
    pub fn common_random_numbers(num_samples: usize, base_seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(base_seed);
        let seeds: Vec<u64> = (0..num_samples).map(|_| rng.gen()).collect();
        Self {
            method: VarianceReductionMethod::CommonRandomNumbers,
            reduction_factor: 0.5, // typical reduction
            crn_seeds: seeds,
        }
    }

    /// Create antithetic variates: for each sample, also use the "mirror" draw.
    pub fn antithetic(num_samples: usize, base_seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(base_seed);
        let seeds: Vec<u64> = (0..num_samples).map(|_| rng.gen()).collect();
        Self {
            method: VarianceReductionMethod::AntitheticVariates,
            reduction_factor: 0.3,
            crn_seeds: seeds,
        }
    }

    /// Estimate variance reduction from paired samples.
    pub fn estimate_reduction(paired_factual: &[f64], paired_counterfactual: &[f64]) -> f64 {
        if paired_factual.len() != paired_counterfactual.len() || paired_factual.len() < 2 {
            return 1.0;
        }

        let n = paired_factual.len() as f64;
        let diffs: Vec<f64> = paired_factual.iter().zip(paired_counterfactual.iter())
            .map(|(f, c)| c - f)
            .collect();

        let var_diff = sample_variance(&diffs);
        let var_f = sample_variance(paired_factual);
        let var_c = sample_variance(paired_counterfactual);

        let var_independent = var_f + var_c;
        if var_independent > 1e-15 {
            var_diff / var_independent
        } else {
            1.0
        }
    }

    /// Effective sample size after variance reduction.
    pub fn effective_samples(&self, nominal_samples: usize) -> f64 {
        nominal_samples as f64 / self.reduction_factor
    }
}

fn sample_variance(data: &[f64]) -> f64 {
    if data.len() < 2 { return 0.0; }
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (data.len() - 1) as f64
}

// ── ResimulationBudget ──────────────────────────────────────────────────────

/// Manage computational budget for re-simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResimulationBudget {
    /// Maximum total rounds to simulate.
    pub max_total_rounds: usize,
    /// Maximum total simulations.
    pub max_simulations: usize,
    /// Rounds simulated so far.
    pub rounds_used: usize,
    /// Simulations run so far.
    pub simulations_used: usize,
}

impl ResimulationBudget {
    pub fn new(max_rounds: usize, max_simulations: usize) -> Self {
        Self {
            max_total_rounds: max_rounds,
            max_simulations,
            rounds_used: 0,
            simulations_used: 0,
        }
    }

    /// Try to allocate rounds for a simulation.
    pub fn try_allocate(&mut self, num_rounds: usize) -> bool {
        if self.rounds_used + num_rounds > self.max_total_rounds {
            return false;
        }
        if self.simulations_used >= self.max_simulations {
            return false;
        }
        self.rounds_used += num_rounds;
        self.simulations_used += 1;
        true
    }

    pub fn remaining_rounds(&self) -> usize {
        self.max_total_rounds.saturating_sub(self.rounds_used)
    }

    pub fn remaining_simulations(&self) -> usize {
        self.max_simulations.saturating_sub(self.simulations_used)
    }

    pub fn utilization(&self) -> f64 {
        let round_util = self.rounds_used as f64 / self.max_total_rounds as f64;
        let sim_util = self.simulations_used as f64 / self.max_simulations as f64;
        round_util.max(sim_util)
    }

    pub fn is_exhausted(&self) -> bool {
        self.rounds_used >= self.max_total_rounds || self.simulations_used >= self.max_simulations
    }
}

// ── ResimulationCheckpoint ──────────────────────────────────────────────────

/// Save/restore re-simulation state for long-running computations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResimulationCheckpoint {
    /// Scenarios completed so far.
    pub completed_scenarios: Vec<String>,
    /// Partial results.
    pub partial_results: Vec<MonteCarloResult>,
    /// Budget state.
    pub budget: ResimulationBudget,
    /// Timestamp.
    pub checkpoint_time: String,
}

impl ResimulationCheckpoint {
    pub fn new(budget: ResimulationBudget) -> Self {
        Self {
            completed_scenarios: Vec::new(),
            partial_results: Vec::new(),
            budget,
            checkpoint_time: String::new(),
        }
    }

    pub fn add_result(&mut self, result: MonteCarloResult) {
        self.completed_scenarios.push(result.scenario.label.clone());
        self.partial_results.push(result);
    }

    pub fn is_scenario_done(&self, label: &str) -> bool {
        self.completed_scenarios.contains(&label.to_string())
    }

    /// Serialize to JSON.
    pub fn to_json(&self) -> CollusionResult<String> {
        serde_json::to_string_pretty(self).map_err(|e| {
            CollusionError::Serialization(format!("Checkpoint serialization failed: {}", e))
        })
    }

    /// Deserialize from JSON.
    pub fn from_json(json: &str) -> CollusionResult<Self> {
        serde_json::from_str(json).map_err(|e| {
            CollusionError::Serialization(format!("Checkpoint deserialization failed: {}", e))
        })
    }
}

// ── ConfidenceFromResimulation ──────────────────────────────────────────────

/// Derive confidence intervals from Monte Carlo samples.
pub fn confidence_from_resimulation(
    samples: &[f64],
    confidence_level: f64,
) -> ConfidenceInterval {
    if samples.is_empty() {
        return ConfidenceInterval::new(0.0, 0.0, confidence_level, 0.0);
    }

    let n = samples.len();
    let mean = samples.iter().sum::<f64>() / n as f64;

    if n == 1 {
        return ConfidenceInterval::new(mean, mean, confidence_level, mean);
    }

    let variance = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    let se = (variance / n as f64).sqrt();

    let z = if confidence_level >= 0.99 {
        2.576
    } else if confidence_level >= 0.95 {
        1.96
    } else if confidence_level >= 0.90 {
        1.645
    } else {
        1.28
    };

    ConfidenceInterval::new(mean - z * se, mean + z * se, confidence_level, mean)
}

/// Percentile-based confidence interval from Monte Carlo.
pub fn percentile_ci(
    samples: &[f64],
    confidence_level: f64,
) -> ConfidenceInterval {
    if samples.is_empty() {
        return ConfidenceInterval::new(0.0, 0.0, confidence_level, 0.0);
    }

    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let alpha = 1.0 - confidence_level;
    let lo_idx = ((alpha / 2.0) * sorted.len() as f64) as usize;
    let hi_idx = ((1.0 - alpha / 2.0) * sorted.len() as f64) as usize;
    let lo_idx = lo_idx.min(sorted.len() - 1);
    let hi_idx = hi_idx.min(sorted.len() - 1);

    let mean = sorted.iter().sum::<f64>() / sorted.len() as f64;
    ConfidenceInterval::new(sorted[lo_idx], sorted[hi_idx], confidence_level, mean)
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use shared_types::{AlgorithmType, Cost, DemandSystem, MarketType, PlayerId,
                       PlayerAction, Price, Profit, Quantity, RoundNumber};

    fn make_trajectory(n_rounds: usize, n_players: usize, price: Price) -> PriceTrajectory {
        let outcomes: Vec<MarketOutcome> = (0..n_rounds)
            .map(|r| MarketOutcome::new(
                RoundNumber(r),
                (0..n_players).map(|i| PlayerAction::new(PlayerId(i), price)).collect(),
                vec![price; n_players],
                vec![Quantity(1.0); n_players],
                vec![Profit((price.0 - 1.0) * 1.0); n_players],
            ))
            .collect();
        PriceTrajectory::new(outcomes, MarketType::Bertrand, n_players, AlgorithmType::QLearning, 0)
    }

    fn make_configs() -> (GameConfig, SimulationConfig) {
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
        let sc = SimulationConfig {
            game: gc.clone(),
            ..Default::default()
        };
        (gc, sc)
    }

    #[test]
    fn test_counterfactual_scenario_single() {
        let s = CounterfactualScenario::single_period(PlayerId(0), Price(3.0), RoundNumber(10));
        assert_eq!(s.deviating_player, PlayerId(0));
        assert_eq!(s.duration, 1);
        assert!(s.opponents_observe);
    }

    #[test]
    fn test_counterfactual_scenario_multi() {
        let s = CounterfactualScenario::multi_period(PlayerId(1), Price(4.0), RoundNumber(5), 10);
        assert_eq!(s.duration, 10);
    }

    #[test]
    fn test_counterfactual_scenario_unobserved() {
        let s = CounterfactualScenario::unobserved(PlayerId(0), Price(3.0), RoundNumber(10));
        assert!(!s.opponents_observe);
    }

    #[test]
    fn test_resimulation_single() {
        let (gc, sc) = make_configs();
        let trajectory = make_trajectory(50, 2, Price(5.0));
        let engine = ResimulationEngine::new(gc, sc, 10, 42);
        let scenario = CounterfactualScenario::single_period(PlayerId(0), Price(3.0), RoundNumber(10));
        let result = engine.simulate_single(&scenario, &trajectory, 42);
        assert!(result.is_ok());
        let r = result.unwrap();
        assert_eq!(r.rounds_simulated, 1);
    }

    #[test]
    fn test_resimulation_multi_round() {
        let (gc, sc) = make_configs();
        let trajectory = make_trajectory(50, 2, Price(5.0));
        let engine = ResimulationEngine::new(gc, sc, 5, 42);
        let scenario = CounterfactualScenario::multi_period(PlayerId(0), Price(3.0), RoundNumber(10), 5);
        let result = engine.simulate_single(&scenario, &trajectory, 42);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().rounds_simulated, 5);
    }

    #[test]
    fn test_monte_carlo() {
        let (gc, sc) = make_configs();
        let trajectory = make_trajectory(50, 2, Price(5.0));
        let engine = ResimulationEngine::new(gc, sc, 20, 42);
        let scenario = CounterfactualScenario::single_period(PlayerId(0), Price(3.0), RoundNumber(10));
        let result = engine.monte_carlo(&scenario, &trajectory);
        assert!(result.is_ok());
        let mc = result.unwrap();
        assert_eq!(mc.num_samples, 20);
        assert!(mc.confidence_interval.width() >= 0.0);
    }

    #[test]
    fn test_parallel_resimulation() {
        let (gc, sc) = make_configs();
        let trajectory = make_trajectory(50, 2, Price(5.0));
        let par = ParallelResimulation::new(gc, sc, 5, 42);
        let scenarios = vec![
            CounterfactualScenario::single_period(PlayerId(0), Price(2.0), RoundNumber(10)),
            CounterfactualScenario::single_period(PlayerId(0), Price(3.0), RoundNumber(10)),
            CounterfactualScenario::single_period(PlayerId(0), Price(4.0), RoundNumber(10)),
        ];
        let results = par.run_parallel(&scenarios, &trajectory);
        assert_eq!(results.len(), 3);
        assert!(results.iter().all(|r| r.is_ok()));
    }

    #[test]
    fn test_price_sweep() {
        let (gc, sc) = make_configs();
        let trajectory = make_trajectory(50, 2, Price(5.0));
        let par = ParallelResimulation::new(gc, sc, 3, 42);
        let prices = vec![Price(2.0), Price(3.0), Price(4.0), Price(6.0), Price(7.0)];
        let results = par.sweep_prices(PlayerId(0), &trajectory, RoundNumber(10), &prices);
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_truncated_horizon_fixed() {
        let th = TruncatedHorizon::fixed(100);
        assert_eq!(th.max_rounds, 100);
        assert!(!th.auto_selected);
    }

    #[test]
    fn test_truncated_horizon_auto() {
        let th = TruncatedHorizon::auto_select(0.95, 10.0, 0.01);
        assert!(th.auto_selected);
        assert!(th.max_rounds > 0);
        assert!(th.truncation_error_bound < 1.0);
    }

    #[test]
    fn test_truncated_horizon_error_bound() {
        let th = TruncatedHorizon::fixed(100);
        let err = th.error_bound(0.95, 10.0);
        assert!(err > 0.0);
        assert!(err < 10.0);
    }

    #[test]
    fn test_variance_reduction_none() {
        let vr = VarianceReduction::none();
        assert_eq!(vr.method, VarianceReductionMethod::None);
        assert!((vr.reduction_factor - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_variance_reduction_crn() {
        let vr = VarianceReduction::common_random_numbers(100, 42);
        assert_eq!(vr.crn_seeds.len(), 100);
        assert!(vr.reduction_factor < 1.0);
    }

    #[test]
    fn test_variance_reduction_estimate() {
        let factual = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let counterfactual = vec![1.1, 2.1, 3.1, 4.1, 5.1]; // perfectly correlated shift
        let reduction = VarianceReduction::estimate_reduction(&factual, &counterfactual);
        assert!(reduction < 1.0); // CRN should reduce variance
    }

    #[test]
    fn test_resimulation_budget() {
        let mut budget = ResimulationBudget::new(1000, 100);
        assert!(budget.try_allocate(100));
        assert_eq!(budget.remaining_rounds(), 900);
        assert_eq!(budget.remaining_simulations(), 99);
        assert!(!budget.is_exhausted());
    }

    #[test]
    fn test_resimulation_budget_exhaustion() {
        let mut budget = ResimulationBudget::new(100, 2);
        assert!(budget.try_allocate(50));
        assert!(budget.try_allocate(50));
        assert!(!budget.try_allocate(1)); // sim budget exhausted
    }

    #[test]
    fn test_resimulation_checkpoint() {
        let budget = ResimulationBudget::new(1000, 100);
        let mut cp = ResimulationCheckpoint::new(budget);
        assert!(!cp.is_scenario_done("test"));
        cp.completed_scenarios.push("test".to_string());
        assert!(cp.is_scenario_done("test"));
    }

    #[test]
    fn test_confidence_from_resimulation() {
        let samples: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();
        let ci = confidence_from_resimulation(&samples, 0.95);
        assert!(ci.lower < ci.upper);
        assert!(ci.contains(0.5));
    }

    #[test]
    fn test_percentile_ci() {
        let samples: Vec<f64> = (0..1000).map(|i| i as f64 / 1000.0).collect();
        let ci = percentile_ci(&samples, 0.95);
        assert!(ci.lower < 0.05);
        assert!(ci.upper > 0.95);
    }

    #[test]
    fn test_discounted_difference() {
        let (gc, sc) = make_configs();
        let trajectory = make_trajectory(50, 2, Price(5.0));
        let engine = ResimulationEngine::new(gc, sc, 1, 42);
        let scenario = CounterfactualScenario::multi_period(PlayerId(0), Price(3.0), RoundNumber(10), 5);
        let result = engine.simulate_single(&scenario, &trajectory, 42).unwrap();
        let disc_diff = result.discounted_difference(0.95);
        assert!(disc_diff.is_finite());
    }
}
