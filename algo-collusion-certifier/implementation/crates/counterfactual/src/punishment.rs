//! M3 Punishment detection via controlled perturbation.
//!
//! Injects controlled deviations and statistically tests whether opponents
//! respond with punishment. Distribution-free permutation-based tests.

use shared_types::{
    CollusionError, CollusionResult, ConfidenceInterval, GameConfig,
    PlayerId, Price, PriceTrajectory,
    RoundNumber, PValue,
};
use game_theory::NashEquilibrium;
use serde::{Deserialize, Serialize};
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::market_helper::simulate_single_round;

// ── PunishmentClassification ────────────────────────────────────────────────

/// Classification of punishment response type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PunishmentClassification {
    NoPunishment,
    MildRetaliation,
    SevereRetaliation,
    GrimPunishment,
}

impl PunishmentClassification {
    pub fn from_metrics(metrics: &PunishmentMetrics) -> Self {
        if metrics.payoff_drop < 0.01 {
            PunishmentClassification::NoPunishment
        } else if metrics.punishment_severity < 0.2 {
            PunishmentClassification::MildRetaliation
        } else if metrics.punishment_duration < 10 || metrics.punishment_severity < 0.6 {
            PunishmentClassification::SevereRetaliation
        } else {
            PunishmentClassification::GrimPunishment
        }
    }

    pub fn severity_score(&self) -> f64 {
        match self {
            Self::NoPunishment => 0.0,
            Self::MildRetaliation => 0.3,
            Self::SevereRetaliation => 0.7,
            Self::GrimPunishment => 1.0,
        }
    }
}

impl std::fmt::Display for PunishmentClassification {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoPunishment => write!(f, "No Punishment"),
            Self::MildRetaliation => write!(f, "Mild Retaliation"),
            Self::SevereRetaliation => write!(f, "Severe Retaliation"),
            Self::GrimPunishment => write!(f, "Grim Punishment"),
        }
    }
}

// ── PunishmentMetrics ───────────────────────────────────────────────────────

/// Metrics quantifying punishment response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PunishmentMetrics {
    /// Average payoff reduction after deviation.
    pub payoff_drop: f64,
    /// Number of rounds until punishment begins.
    pub punishment_speed: usize,
    /// How many rounds punishment lasts.
    pub punishment_duration: usize,
    /// Magnitude of payoff reduction (fraction of on-path payoff).
    pub punishment_severity: f64,
    /// Cumulative discounted punishment cost.
    pub cumulative_cost: f64,
    /// Standard error of payoff drop estimate.
    pub payoff_drop_se: f64,
}

impl PunishmentMetrics {
    pub fn no_punishment() -> Self {
        Self {
            payoff_drop: 0.0,
            punishment_speed: 0,
            punishment_duration: 0,
            punishment_severity: 0.0,
            cumulative_cost: 0.0,
            payoff_drop_se: 0.0,
        }
    }

    /// Compute metrics from pre- and post-deviation payoff sequences.
    pub fn compute(
        on_path_payoffs: &[f64],
        post_deviation_payoffs: &[f64],
        discount_factor: f64,
    ) -> Self {
        if on_path_payoffs.is_empty() || post_deviation_payoffs.is_empty() {
            return Self::no_punishment();
        }

        let on_path_mean = on_path_payoffs.iter().sum::<f64>() / on_path_payoffs.len() as f64;
        let post_mean = post_deviation_payoffs.iter().sum::<f64>() / post_deviation_payoffs.len() as f64;
        let payoff_drop = (on_path_mean - post_mean).max(0.0);

        // Detect when punishment starts
        let mut punishment_speed = 0usize;
        let threshold = on_path_mean - 0.5 * payoff_drop.max(0.01);
        for (i, &payoff) in post_deviation_payoffs.iter().enumerate() {
            if payoff < threshold {
                punishment_speed = i;
                break;
            }
            punishment_speed = post_deviation_payoffs.len();
        }

        // Detect punishment duration
        let mut punishment_rounds = 0usize;
        for &payoff in post_deviation_payoffs {
            if payoff < threshold {
                punishment_rounds += 1;
            }
        }

        let severity = if on_path_mean.abs() > 1e-12 {
            payoff_drop / on_path_mean.abs()
        } else {
            0.0
        };

        // Cumulative discounted cost
        let cumulative_cost: f64 = post_deviation_payoffs.iter().enumerate()
            .map(|(t, &pi)| discount_factor.powi(t as i32) * (on_path_mean - pi).max(0.0))
            .sum();

        // Standard error via bootstrap-style estimate
        let diffs: Vec<f64> = post_deviation_payoffs.iter()
            .map(|&p| on_path_mean - p)
            .collect();
        let diff_mean = diffs.iter().sum::<f64>() / diffs.len() as f64;
        let variance = if diffs.len() > 1 {
            diffs.iter().map(|d| (d - diff_mean).powi(2)).sum::<f64>() / (diffs.len() - 1) as f64
        } else {
            0.0
        };
        let se = (variance / diffs.len() as f64).sqrt();

        Self {
            payoff_drop,
            punishment_speed,
            punishment_duration: punishment_rounds,
            punishment_severity: severity.min(1.0),
            cumulative_cost,
            payoff_drop_se: se,
        }
    }
}

// ── PunishmentTest ──────────────────────────────────────────────────────────

/// Statistical test for punishment response.
///
/// H0: post-deviation trajectory is same as on-path (no punishment).
/// H1: post-deviation shows payoff reduction (punishment).
/// Uses distribution-free permutation-based p-value.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PunishmentTest {
    pub player: PlayerId,
    pub p_value: PValue,
    pub test_statistic: f64,
    pub reject_null: bool,
    pub alpha: f64,
    pub num_permutations: usize,
    pub metrics: PunishmentMetrics,
    pub classification: PunishmentClassification,
}

impl PunishmentTest {
    /// Run permutation-based punishment test.
    pub fn run(
        player: PlayerId,
        on_path_payoffs: &[f64],
        post_deviation_payoffs: &[f64],
        alpha: f64,
        num_permutations: usize,
        seed: Option<u64>,
        discount_factor: f64,
    ) -> Self {
        let metrics = PunishmentMetrics::compute(on_path_payoffs, post_deviation_payoffs, discount_factor);

        // Observed test statistic: mean difference
        let obs_mean_on = if on_path_payoffs.is_empty() { 0.0 }
            else { on_path_payoffs.iter().sum::<f64>() / on_path_payoffs.len() as f64 };
        let obs_mean_post = if post_deviation_payoffs.is_empty() { 0.0 }
            else { post_deviation_payoffs.iter().sum::<f64>() / post_deviation_payoffs.len() as f64 };
        let observed_stat = obs_mean_on - obs_mean_post;

        // Permutation test
        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        let mut combined: Vec<f64> = on_path_payoffs.iter()
            .chain(post_deviation_payoffs.iter())
            .copied()
            .collect();
        let n_on = on_path_payoffs.len();
        let mut count_extreme = 0usize;

        for _ in 0..num_permutations {
            combined.shuffle(&mut rng);
            let perm_on = &combined[..n_on];
            let perm_post = &combined[n_on..];
            let perm_mean_on = if perm_on.is_empty() { 0.0 }
                else { perm_on.iter().sum::<f64>() / perm_on.len() as f64 };
            let perm_mean_post = if perm_post.is_empty() { 0.0 }
                else { perm_post.iter().sum::<f64>() / perm_post.len() as f64 };
            let perm_stat = perm_mean_on - perm_mean_post;

            if perm_stat >= observed_stat {
                count_extreme += 1;
            }
        }

        let p_val = (count_extreme as f64 + 1.0) / (num_permutations as f64 + 1.0);
        let pv = PValue::new_unchecked(p_val);
        let classification = PunishmentClassification::from_metrics(&metrics);

        Self {
            player,
            p_value: pv,
            test_statistic: observed_stat,
            reject_null: pv.is_significant(alpha),
            alpha,
            num_permutations,
            metrics,
            classification,
        }
    }

    /// Whether punishment was detected.
    pub fn punishment_detected(&self) -> bool {
        self.reject_null && self.metrics.payoff_drop > 0.0
    }
}

// ── ControlledPerturbation ──────────────────────────────────────────────────

/// Controlled perturbation: inject a deviation and observe the response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlledPerturbation {
    /// Player who deviates.
    pub deviating_player: PlayerId,
    /// The price deviation is to.
    pub deviation_price: Price,
    /// Round at which deviation is injected.
    pub injection_round: RoundNumber,
    /// Number of rounds to observe after deviation.
    pub observation_window: usize,
    /// Observed payoffs on the on-path trajectory.
    pub on_path_payoffs: Vec<f64>,
    /// Observed payoffs after the deviation.
    pub post_deviation_payoffs: Vec<f64>,
}

impl ControlledPerturbation {
    /// Create and execute a controlled perturbation.
    pub fn execute(
        deviating_player: PlayerId,
        deviation_price: Price,
        trajectory: &PriceTrajectory,
        injection_round: RoundNumber,
        observation_window: usize,
        game_config: &GameConfig,
    ) -> CollusionResult<Self> {
        if injection_round.0 >= trajectory.len() {
            return Err(CollusionError::InvalidState(
                format!("Injection round {} out of range", injection_round.0),
            ));
        }

        let obs_end = (injection_round.0 + 1 + observation_window).min(trajectory.len());

        // On-path payoffs for deviating player (before and after injection round)
        let on_path_start = injection_round.saturating_sub(observation_window).0;
        let on_path_payoffs: Vec<f64> = (on_path_start..injection_round.0)
            .map(|r| trajectory.outcomes[r].profits[deviating_player].0)
            .collect();

        let mut post_payoffs = Vec::with_capacity(observation_window);

        // Inject deviation at injection_round
        {
            let outcome = &trajectory.outcomes[injection_round];
            let mut dev_prices = outcome.prices.clone();
            dev_prices[deviating_player.0] = deviation_price;

            let dev_outcome = simulate_single_round(&dev_prices, game_config, injection_round);
            post_payoffs.push(dev_outcome.profits[deviating_player].0);
        }

        // Observe subsequent rounds — opponents may adjust
        for r in (injection_round.0 + 1)..obs_end {
            if r < trajectory.len() {
                // In a full implementation, opponents would re-optimize.
                // Here we use observed prices as proxy for opponent response.
                let outcome = &trajectory.outcomes[r];

                let response_outcome = simulate_single_round(&outcome.prices, game_config, RoundNumber(r));
                post_payoffs.push(response_outcome.profits[deviating_player].0);
            }
        }

        Ok(Self {
            deviating_player,
            deviation_price,
            injection_round,
            observation_window,
            on_path_payoffs,
            post_deviation_payoffs: post_payoffs,
        })
    }

    /// Compute payoff drop after perturbation.
    pub fn payoff_drop(&self) -> f64 {
        let pre_mean = if self.on_path_payoffs.is_empty() { 0.0 }
            else { self.on_path_payoffs.iter().sum::<f64>() / self.on_path_payoffs.len() as f64 };
        let post_mean = if self.post_deviation_payoffs.is_empty() { 0.0 }
            else { self.post_deviation_payoffs.iter().sum::<f64>() / self.post_deviation_payoffs.len() as f64 };
        (pre_mean - post_mean).max(0.0)
    }
}

// ── InjectionSchedule ───────────────────────────────────────────────────────

/// Schedule for when and how to inject deviations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InjectionSchedule {
    /// Planned injection rounds.
    pub injection_rounds: Vec<RoundNumber>,
    /// Minimum gap between injections (to ensure independence).
    pub min_gap: usize,
    /// Observation window per injection.
    pub observation_window: usize,
    /// Players to test.
    pub players: Vec<PlayerId>,
}

impl InjectionSchedule {
    /// Create a schedule with evenly spaced injections.
    pub fn uniform(
        trajectory_length: usize,
        num_injections: usize,
        observation_window: usize,
        players: Vec<PlayerId>,
    ) -> Self {
        let min_gap = observation_window * 2 + 1;
        let effective_length = trajectory_length.saturating_sub(observation_window);

        let spacing = if num_injections > 0 {
            effective_length / num_injections
        } else {
            effective_length
        };
        let spacing = spacing.max(min_gap);

        let mut rounds = Vec::new();
        let mut r = observation_window;
        for _ in 0..num_injections {
            if r + observation_window < trajectory_length {
                rounds.push(RoundNumber(r));
                r += spacing;
            }
        }

        Self {
            injection_rounds: rounds,
            min_gap,
            observation_window,
            players,
        }
    }

    /// Create a schedule with randomized injection rounds.
    pub fn randomized(
        trajectory_length: usize,
        num_injections: usize,
        observation_window: usize,
        players: Vec<PlayerId>,
        seed: u64,
    ) -> Self {
        let min_gap = observation_window * 2 + 1;
        let mut rng = StdRng::seed_from_u64(seed);
        let effective_start = observation_window;
        let effective_end = trajectory_length.saturating_sub(observation_window);

        let mut rounds: Vec<RoundNumber> = Vec::new();

        for _ in 0..num_injections * 10 {
            if rounds.len() >= num_injections {
                break;
            }
            if effective_start >= effective_end {
                break;
            }
            let candidate = rng.gen_range(effective_start..effective_end);
            let conflict = rounds.iter().any(|&r| {
                (candidate as isize - r.0 as isize).unsigned_abs() < min_gap
            });
            if !conflict {
                rounds.push(RoundNumber(candidate));
            }
        }

        rounds.sort_unstable();

        Self {
            injection_rounds: rounds,
            min_gap,
            observation_window,
            players,
        }
    }

    pub fn num_injections(&self) -> usize {
        self.injection_rounds.len()
    }

    /// Validate that injections are sufficiently separated.
    pub fn validate(&self) -> bool {
        for i in 1..self.injection_rounds.len() {
            if self.injection_rounds[i].0 - self.injection_rounds[i - 1].0 < self.min_gap {
                return false;
            }
        }
        true
    }

    /// Total number of perturbation experiments (injections × players).
    pub fn total_experiments(&self) -> usize {
        self.injection_rounds.len() * self.players.len()
    }
}

// ── RequiredInjections ──────────────────────────────────────────────────────

/// Compute required number of injection experiments.
///
/// J = O(σ² × log(1/β) / Δ_P²)
/// where σ² is payoff variance, β is failure probability, Δ_P is detection threshold.
pub fn required_injections(
    variance_estimate: f64,
    detection_threshold: f64,
    failure_probability: f64,
) -> usize {
    if detection_threshold <= 0.0 || failure_probability <= 0.0 {
        return 1;
    }
    let j = variance_estimate * (1.0 / failure_probability).ln() / detection_threshold.powi(2);
    (j.ceil() as usize).max(1)
}

// ── PunishmentEvidence ──────────────────────────────────────────────────────

/// Evidence bundle for punishment detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PunishmentEvidence {
    pub player: PlayerId,
    pub test_results: Vec<PunishmentTest>,
    pub perturbations: Vec<ControlledPerturbation>,
    pub aggregate_metrics: PunishmentMetrics,
    pub classification: PunishmentClassification,
    pub confidence_interval: Option<ConfidenceInterval>,
    pub num_experiments: usize,
}

impl PunishmentEvidence {
    /// Aggregate evidence from multiple perturbation experiments.
    pub fn aggregate(
        player: PlayerId,
        tests: Vec<PunishmentTest>,
        perturbations: Vec<ControlledPerturbation>,
        discount_factor: f64,
    ) -> Self {
        let num_experiments = tests.len();

        // Aggregate payoff drops
        let all_on_path: Vec<f64> = perturbations.iter()
            .flat_map(|p| p.on_path_payoffs.iter().copied())
            .collect();
        let all_post: Vec<f64> = perturbations.iter()
            .flat_map(|p| p.post_deviation_payoffs.iter().copied())
            .collect();

        let aggregate_metrics = PunishmentMetrics::compute(&all_on_path, &all_post, discount_factor);
        let classification = PunishmentClassification::from_metrics(&aggregate_metrics);

        // Combine p-values via Fisher's method
        let combined_chi2: f64 = tests.iter()
            .map(|t| -2.0 * t.p_value.value().max(1e-300).ln())
            .sum();
        let df = 2 * tests.len();
        // Approximate chi-squared p-value for aggregate
        let combined_p = chi2_survival(combined_chi2, df);

        let ci = if !all_on_path.is_empty() && !all_post.is_empty() {
            let mean_drop = aggregate_metrics.payoff_drop;
            let se = aggregate_metrics.payoff_drop_se;
            Some(ConfidenceInterval::new(
                mean_drop - 1.96 * se,
                mean_drop + 1.96 * se,
                0.95,
                mean_drop,
            ))
        } else {
            None
        };

        Self {
            player,
            test_results: tests,
            perturbations,
            aggregate_metrics,
            classification,
            confidence_interval: ci,
            num_experiments,
        }
    }

    /// Whether punishment is detected at given alpha.
    pub fn punishment_detected(&self, alpha: f64) -> bool {
        // Majority of tests should reject
        let reject_count = self.test_results.iter().filter(|t| t.reject_null).count();
        let reject_fraction = if self.test_results.is_empty() { 0.0 }
            else { reject_count as f64 / self.test_results.len() as f64 };
        reject_fraction > 0.5 && self.aggregate_metrics.payoff_drop > 0.0
    }
}

// ── PunishmentDetector ──────────────────────────────────────────────────────

/// Configuration for punishment detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PunishmentDetectorConfig {
    /// Significance level for individual tests.
    pub alpha: f64,
    /// Number of permutations per test.
    pub num_permutations: usize,
    /// Observation window after deviation.
    pub observation_window: usize,
    /// Number of injection experiments per player.
    pub num_injections_per_player: usize,
    /// Minimum gap between injections.
    pub min_injection_gap: usize,
    /// Random seed.
    pub seed: Option<u64>,
}

impl Default for PunishmentDetectorConfig {
    fn default() -> Self {
        Self {
            alpha: 0.05,
            num_permutations: 999,
            observation_window: 20,
            num_injections_per_player: 10,
            min_injection_gap: 50,
            seed: None,
        }
    }
}

/// Main punishment detection engine.
pub struct PunishmentDetector {
    config: PunishmentDetectorConfig,
    game_config: GameConfig,
}

impl PunishmentDetector {
    pub fn new(config: PunishmentDetectorConfig, game_config: GameConfig) -> Self {
        Self { config, game_config }
    }

    /// Run full punishment detection for all players.
    pub fn detect_all(
        &self,
        trajectory: &PriceTrajectory,
        nash_eq: &NashEquilibrium,
    ) -> CollusionResult<Vec<PunishmentEvidence>> {
        let mut all_evidence = Vec::new();

        for player in 0..trajectory.num_players {
            let evidence = self.detect_for_player(PlayerId(player), trajectory, nash_eq)?;
            all_evidence.push(evidence);
        }

        Ok(all_evidence)
    }

    /// Run punishment detection for a specific player.
    pub fn detect_for_player(
        &self,
        player: PlayerId,
        trajectory: &PriceTrajectory,
        nash_eq: &NashEquilibrium,
    ) -> CollusionResult<PunishmentEvidence> {
        let nash_price = self.game_config.competitive_price();

        // Create injection schedule
        let schedule = InjectionSchedule::uniform(
            trajectory.len(),
            self.config.num_injections_per_player,
            self.config.observation_window,
            vec![player],
        );

        let mut tests = Vec::new();
        let mut perturbations = Vec::new();
        let seed_base = self.config.seed.unwrap_or(42);

        for (idx, &inj_round) in schedule.injection_rounds.iter().enumerate() {
            // Execute controlled perturbation
            let perturbation = ControlledPerturbation::execute(
                player,
                nash_price,
                trajectory,
                inj_round,
                self.config.observation_window,
                &self.game_config,
            )?;

            // Run permutation test
            let test = PunishmentTest::run(
                player,
                &perturbation.on_path_payoffs,
                &perturbation.post_deviation_payoffs,
                self.config.alpha,
                self.config.num_permutations,
                Some(seed_base + idx as u64),
                self.game_config.discount_factor,
            );

            tests.push(test);
            perturbations.push(perturbation);
        }

        Ok(PunishmentEvidence::aggregate(
            player,
            tests,
            perturbations,
            self.game_config.discount_factor,
        ))
    }

    /// Quick punishment check: single injection, single test.
    pub fn quick_check(
        &self,
        player: PlayerId,
        trajectory: &PriceTrajectory,
        nash_price: Price,
        injection_round: RoundNumber,
    ) -> CollusionResult<PunishmentTest> {
        let perturbation = ControlledPerturbation::execute(
            player,
            nash_price,
            trajectory,
            injection_round,
            self.config.observation_window,
            &self.game_config,
        )?;

        Ok(PunishmentTest::run(
            player,
            &perturbation.on_path_payoffs,
            &perturbation.post_deviation_payoffs,
            self.config.alpha,
            self.config.num_permutations,
            self.config.seed,
            self.game_config.discount_factor,
        ))
    }

    /// Estimate required injections for given precision.
    pub fn required_injections_estimate(
        &self,
        trajectory: &PriceTrajectory,
        player: PlayerId,
        detection_threshold: f64,
    ) -> usize {
        let payoffs = (0..trajectory.len())
            .map(|r| trajectory.outcomes[r].profits[player].0)
            .collect::<Vec<f64>>();

        if payoffs.len() < 2 {
            return 1;
        }

        let mean = payoffs.iter().sum::<f64>() / payoffs.len() as f64;
        let variance = payoffs.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / (payoffs.len() - 1) as f64;

        required_injections(variance, detection_threshold, self.config.alpha)
    }
}

// ── Helper: approximate chi-squared survival function ───────────────────────

fn chi2_survival(x: f64, df: usize) -> f64 {
    if x <= 0.0 { return 1.0; }
    if df == 0 { return 0.0; }

    // Wilson-Hilferty normal approximation for chi-squared
    let k = df as f64;
    let z = ((x / k).powf(1.0 / 3.0) - (1.0 - 2.0 / (9.0 * k))) / (2.0 / (9.0 * k)).sqrt();
    let p = 0.5 * erfc(z / std::f64::consts::SQRT_2);
    p.clamp(0.0, 1.0)
}

fn erfc(x: f64) -> f64 {
    1.0 - erf(x)
}

fn erf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use shared_types::{Cost, DemandSystem, MarketType, Quantity, AlgorithmType};
    use crate::market_helper::{make_test_trajectory, make_test_game_config};

    fn make_trajectory(n_rounds: usize, n_players: usize, price: Price) -> PriceTrajectory {
        make_test_trajectory(n_rounds, n_players, price)
    }

    fn make_game_config() -> GameConfig {
        make_test_game_config(2)
    }

    #[test]
    fn test_punishment_classification_no_punishment() {
        let metrics = PunishmentMetrics::no_punishment();
        let class = PunishmentClassification::from_metrics(&metrics);
        assert_eq!(class, PunishmentClassification::NoPunishment);
        assert!((class.severity_score() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_punishment_classification_mild() {
        let metrics = PunishmentMetrics {
            payoff_drop: 0.5,
            punishment_speed: 1,
            punishment_duration: 3,
            punishment_severity: 0.1,
            cumulative_cost: 1.5,
            payoff_drop_se: 0.1,
        };
        let class = PunishmentClassification::from_metrics(&metrics);
        assert_eq!(class, PunishmentClassification::MildRetaliation);
    }

    #[test]
    fn test_punishment_classification_severe() {
        let metrics = PunishmentMetrics {
            payoff_drop: 2.0,
            punishment_speed: 1,
            punishment_duration: 5,
            punishment_severity: 0.5,
            cumulative_cost: 10.0,
            payoff_drop_se: 0.2,
        };
        let class = PunishmentClassification::from_metrics(&metrics);
        assert_eq!(class, PunishmentClassification::SevereRetaliation);
    }

    #[test]
    fn test_punishment_classification_grim() {
        let metrics = PunishmentMetrics {
            payoff_drop: 5.0,
            punishment_speed: 0,
            punishment_duration: 100,
            punishment_severity: 0.9,
            cumulative_cost: 50.0,
            payoff_drop_se: 0.3,
        };
        let class = PunishmentClassification::from_metrics(&metrics);
        assert_eq!(class, PunishmentClassification::GrimPunishment);
    }

    #[test]
    fn test_punishment_metrics_compute() {
        let on_path = vec![10.0, 10.0, 10.0, 10.0, 10.0];
        let post_dev = vec![8.0, 7.0, 6.0, 7.0, 9.0];
        let metrics = PunishmentMetrics::compute(&on_path, &post_dev, 0.95);
        assert!(metrics.payoff_drop > 0.0);
        assert!(metrics.punishment_severity > 0.0);
        assert!(metrics.cumulative_cost > 0.0);
    }

    #[test]
    fn test_punishment_metrics_no_drop() {
        let on_path = vec![5.0, 5.0, 5.0];
        let post_dev = vec![5.0, 5.0, 5.0];
        let metrics = PunishmentMetrics::compute(&on_path, &post_dev, 0.95);
        assert!(metrics.payoff_drop < 0.01);
    }

    #[test]
    fn test_punishment_test_significant() {
        let on_path = vec![10.0; 20];
        let post_dev = vec![5.0; 20];
        let test = PunishmentTest::run(PlayerId(0), &on_path, &post_dev, 0.05, 999, Some(42), 0.95);
        assert!(test.punishment_detected());
        assert!(test.p_value.value() < 0.05);
    }

    #[test]
    fn test_punishment_test_not_significant() {
        let on_path = vec![10.0, 10.1, 9.9, 10.0, 10.2];
        let post_dev = vec![10.0, 9.9, 10.1, 10.0, 9.8];
        let test = PunishmentTest::run(PlayerId(0), &on_path, &post_dev, 0.05, 999, Some(42), 0.95);
        assert!(!test.punishment_detected());
    }

    #[test]
    fn test_controlled_perturbation() {
        let trajectory = make_trajectory(100, 2, Price(5.0));
        let game_config = make_game_config();
        let perturbation = ControlledPerturbation::execute(PlayerId(0), Price(3.0), &trajectory, RoundNumber(50), 10, &game_config);
        assert!(perturbation.is_ok());
        let p = perturbation.unwrap();
        assert_eq!(p.deviating_player, PlayerId(0));
        assert!(!p.post_deviation_payoffs.is_empty());
    }

    #[test]
    fn test_injection_schedule_uniform() {
        let schedule = InjectionSchedule::uniform(200, 5, 10, vec![PlayerId(0), PlayerId(1)]);
        assert!(schedule.validate());
        assert!(!schedule.injection_rounds.is_empty());
        assert_eq!(schedule.players, vec![PlayerId(0), PlayerId(1)]);
    }

    #[test]
    fn test_injection_schedule_randomized() {
        let schedule = InjectionSchedule::randomized(200, 5, 10, vec![PlayerId(0)], 42);
        assert!(schedule.validate());
    }

    #[test]
    fn test_required_injections() {
        let j = required_injections(1.0, 0.1, 0.05);
        assert!(j >= 1);
        let j2 = required_injections(1.0, 0.01, 0.05);
        assert!(j2 > j); // tighter threshold needs more samples
    }

    #[test]
    fn test_punishment_evidence_aggregate() {
        let on_path = vec![10.0; 10];
        let post_dev = vec![7.0; 10];
        let test = PunishmentTest::run(PlayerId(0), &on_path, &post_dev, 0.05, 999, Some(42), 0.95);
        let perturbation = ControlledPerturbation {
            deviating_player: PlayerId(0),
            deviation_price: Price(3.0),
            injection_round: RoundNumber(50),
            observation_window: 10,
            on_path_payoffs: on_path,
            post_deviation_payoffs: post_dev,
        };
        let evidence = PunishmentEvidence::aggregate(PlayerId(0), vec![test], vec![perturbation], 0.95);
        assert!(evidence.punishment_detected(0.05));
        assert_eq!(evidence.num_experiments, 1);
    }

    #[test]
    fn test_punishment_detector_creation() {
        let config = PunishmentDetectorConfig::default();
        let game_config = make_game_config();
        let detector = PunishmentDetector::new(config, game_config);
        let trajectory = make_trajectory(200, 2, Price(5.0));
        let nash = NashEquilibrium::pure(vec![0, 0], vec![6.0, 6.0]);
        let result = detector.detect_for_player(PlayerId(0), &trajectory, &nash);
        assert!(result.is_ok());
    }

    #[test]
    fn test_punishment_detector_quick_check() {
        let config = PunishmentDetectorConfig {
            observation_window: 10,
            num_permutations: 99,
            ..Default::default()
        };
        let game_config = make_game_config();
        let detector = PunishmentDetector::new(config, game_config);
        let trajectory = make_trajectory(100, 2, Price(5.0));
        let result = detector.quick_check(PlayerId(0), &trajectory, Price(3.0), RoundNumber(50));
        assert!(result.is_ok());
    }

    #[test]
    fn test_required_injections_estimate() {
        let config = PunishmentDetectorConfig::default();
        let game_config = make_game_config();
        let detector = PunishmentDetector::new(config, game_config);
        let trajectory = make_trajectory(100, 2, Price(5.0));
        let n = detector.required_injections_estimate(&trajectory, PlayerId(0), 0.5);
        assert!(n >= 1);
    }

    #[test]
    fn test_chi2_survival() {
        let p = chi2_survival(0.0, 4);
        assert!((p - 1.0).abs() < 0.1);
        let p2 = chi2_survival(100.0, 4);
        assert!(p2 < 0.001);
    }
}
