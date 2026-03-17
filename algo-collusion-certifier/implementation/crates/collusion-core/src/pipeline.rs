//! End-to-end certification pipeline for the CollusionProof system.
//!
//! Orchestrates simulation, analysis, testing, and certification stages
//! with checkpoint support and parallel execution.

use crate::algorithm::{AlgorithmFactory, PricingAlgorithm, SandboxConfig};
use crate::detector::{CollusionDetector, CollusionReport, DetectionConfig, DetectionResult, Verdict};
use crate::scenario::{ExpectedVerdict, Scenario};
use serde::{Deserialize, Serialize};
use shared_types::{
    CollusionError, CollusionResult, ConfidenceInterval, DemandSystem,
    MarketOutcome, PlayerId, PlayerAction, Price, PriceTrajectory, Profit,
    Quantity, RoundNumber,
};
use std::collections::HashMap;
use std::time::{Duration, Instant};

// ═══════════════════════════════════════════════════════════════════════════
// Pipeline stages
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub enum PipelineStage {
    Simulate,
    Analyze,
    Test,
    Certify,
    Verify,
}

impl PipelineStage {
    pub fn all() -> &'static [PipelineStage] {
        &[
            PipelineStage::Simulate,
            PipelineStage::Analyze,
            PipelineStage::Test,
            PipelineStage::Certify,
            PipelineStage::Verify,
        ]
    }

    pub fn name(&self) -> &'static str {
        match self {
            PipelineStage::Simulate => "Simulate",
            PipelineStage::Analyze => "Analyze",
            PipelineStage::Test => "Test",
            PipelineStage::Certify => "Certify",
            PipelineStage::Verify => "Verify",
        }
    }

    pub fn next(&self) -> Option<PipelineStage> {
        match self {
            PipelineStage::Simulate => Some(PipelineStage::Analyze),
            PipelineStage::Analyze => Some(PipelineStage::Test),
            PipelineStage::Test => Some(PipelineStage::Certify),
            PipelineStage::Certify => Some(PipelineStage::Verify),
            PipelineStage::Verify => None,
        }
    }
}

impl std::fmt::Display for PipelineStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Pipeline configuration
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    pub num_simulations: usize,
    pub rounds_per_simulation: usize,
    pub num_price_levels: usize,
    pub price_min: Price,
    pub price_max: Price,
    pub detection_config: DetectionConfig,
    pub sandbox_config: SandboxConfig,
    pub enable_checkpoints: bool,
    pub checkpoint_interval: usize,
    pub parallel_scenarios: bool,
    pub seed: Option<u64>,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            num_simulations: 1,
            rounds_per_simulation: 1000,
            num_price_levels: 15,
            price_min: Price(0.0),
            price_max: Price(10.0),
            detection_config: DetectionConfig {
                min_trajectory_length: 50,
                ..DetectionConfig::default()
            },
            sandbox_config: SandboxConfig::default(),
            enable_checkpoints: true,
            checkpoint_interval: 100,
            parallel_scenarios: false,
            seed: Some(42),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Pipeline checkpoint
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineCheckpoint {
    pub stage: PipelineStage,
    pub trajectory: Option<PriceTrajectory>,
    pub detection_result: Option<DetectionResult>,
    pub metrics: PipelineMetrics,
    pub timestamp_ms: u128,
}

impl PipelineCheckpoint {
    pub fn new(stage: PipelineStage) -> Self {
        Self {
            stage,
            trajectory: None,
            detection_result: None,
            metrics: PipelineMetrics::new(),
            timestamp_ms: 0,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Pipeline metrics
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineMetrics {
    pub stage_durations: HashMap<String, Duration>,
    pub total_duration: Duration,
    pub trajectory_length: usize,
    pub num_players: usize,
    pub peak_memory_estimate: usize,
    pub stages_completed: usize,
}

impl PipelineMetrics {
    pub fn new() -> Self {
        Self {
            stage_durations: HashMap::new(),
            total_duration: Duration::ZERO,
            trajectory_length: 0,
            num_players: 0,
            peak_memory_estimate: 0,
            stages_completed: 0,
        }
    }

    pub fn record_stage(&mut self, stage: &PipelineStage, duration: Duration) {
        self.stage_durations.insert(stage.name().to_string(), duration);
        self.total_duration += duration;
        self.stages_completed += 1;
    }
}

impl Default for PipelineMetrics {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Certification result
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertificationResult {
    pub scenario_name: String,
    pub verdict: Verdict,
    pub confidence: f64,
    pub collusion_premium: f64,
    pub cp_confidence_interval: Option<ConfidenceInterval>,
    pub report: CollusionReport,
    pub metrics: PipelineMetrics,
    pub checkpoints: Vec<PipelineCheckpoint>,
    pub passed: bool,
}

impl CertificationResult {
    /// Check if the result matches the expected verdict.
    pub fn matches_expected(&self, expected: &ExpectedVerdict) -> bool {
        expected.matches(&self.verdict)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Progress callback
// ═══════════════════════════════════════════════════════════════════════════

pub type ProgressCallback = Box<dyn Fn(PipelineStage, f64) + Send + Sync>;

// ═══════════════════════════════════════════════════════════════════════════
// CertificationPipeline
// ═══════════════════════════════════════════════════════════════════════════

pub struct CertificationPipeline {
    config: PipelineConfig,
    progress_callback: Option<ProgressCallback>,
}

impl CertificationPipeline {
    pub fn new(config: PipelineConfig) -> Self {
        Self {
            config,
            progress_callback: None,
        }
    }

    pub fn with_progress(mut self, callback: ProgressCallback) -> Self {
        self.progress_callback = Some(callback);
        self
    }

    fn report_progress(&self, stage: PipelineStage, progress: f64) {
        if let Some(ref cb) = self.progress_callback {
            cb(stage, progress);
        }
    }

    /// Run the full certification pipeline for a scenario.
    pub fn run(&self, scenario: &Scenario) -> CollusionResult<CertificationResult> {
        let start = Instant::now();
        let mut metrics = PipelineMetrics::new();
        let mut checkpoints = Vec::new();

        // ── Stage 1: Simulate ──────────────────────────────────────────
        self.report_progress(PipelineStage::Simulate, 0.0);
        let sim_start = Instant::now();
        let trajectory = self.simulate(scenario)?;
        metrics.record_stage(&PipelineStage::Simulate, sim_start.elapsed());
        metrics.trajectory_length = trajectory.len();
        metrics.num_players = trajectory.num_players;
        self.report_progress(PipelineStage::Simulate, 1.0);

        if self.config.enable_checkpoints {
            let mut cp = PipelineCheckpoint::new(PipelineStage::Simulate);
            cp.trajectory = Some(trajectory.clone());
            cp.timestamp_ms = start.elapsed().as_millis();
            checkpoints.push(cp);
        }

        // ── Stage 2: Analyze ───────────────────────────────────────────
        self.report_progress(PipelineStage::Analyze, 0.0);
        let analyze_start = Instant::now();
        let _analysis = self.analyze(&trajectory)?;
        metrics.record_stage(&PipelineStage::Analyze, analyze_start.elapsed());
        self.report_progress(PipelineStage::Analyze, 1.0);

        // ── Stage 3: Test ──────────────────────────────────────────────
        self.report_progress(PipelineStage::Test, 0.0);
        let test_start = Instant::now();
        let report = self.test(&trajectory)?;
        metrics.record_stage(&PipelineStage::Test, test_start.elapsed());
        self.report_progress(PipelineStage::Test, 1.0);

        if self.config.enable_checkpoints {
            let mut cp = PipelineCheckpoint::new(PipelineStage::Test);
            cp.detection_result = Some(report.result.clone());
            cp.timestamp_ms = start.elapsed().as_millis();
            checkpoints.push(cp);
        }

        // ── Stage 4: Certify ───────────────────────────────────────────
        self.report_progress(PipelineStage::Certify, 0.0);
        let certify_start = Instant::now();
        let verdict = report.result.verdict;
        let confidence = report.result.confidence;
        let cp = report.result.collusion_premium_estimate;
        let cp_ci = report.result.cp_confidence_interval.clone();
        metrics.record_stage(&PipelineStage::Certify, certify_start.elapsed());
        self.report_progress(PipelineStage::Certify, 1.0);

        // ── Stage 5: Verify ────────────────────────────────────────────
        self.report_progress(PipelineStage::Verify, 0.0);
        let verify_start = Instant::now();
        let passed = scenario.expected_verdict.matches(&verdict);
        metrics.record_stage(&PipelineStage::Verify, verify_start.elapsed());
        self.report_progress(PipelineStage::Verify, 1.0);

        Ok(CertificationResult {
            scenario_name: scenario.name.clone(),
            verdict,
            confidence,
            collusion_premium: cp,
            cp_confidence_interval: cp_ci,
            report,
            metrics,
            checkpoints,
            passed,
        })
    }

    /// Stage 1: Simulate a trajectory from the scenario's algorithms.
    fn simulate(&self, scenario: &Scenario) -> CollusionResult<PriceTrajectory> {
        let sim_config = &scenario.simulation_config;
        let num_players = sim_config.num_players();
        let num_rounds = self.config.rounds_per_simulation;

        // Create algorithms
        let mut algorithms: Vec<Box<dyn PricingAlgorithm>> = Vec::new();
        for (i, algo_config) in scenario.algorithm_configs.iter().enumerate() {
            let algo = AlgorithmFactory::create(
                algo_config,
                PlayerId(i),
                self.config.num_price_levels,
                self.config.price_min,
                self.config.price_max,
            )?;
            algorithms.push(algo);
        }

        // Ensure we have enough algorithms
        while algorithms.len() < num_players {
            // Duplicate the last algorithm for remaining players
            if let Some(last_config) = scenario.algorithm_configs.last() {
                let algo = AlgorithmFactory::create(
                    last_config,
                    PlayerId(algorithms.len()),
                    self.config.num_price_levels,
                    self.config.price_min,
                    self.config.price_max,
                )?;
                algorithms.push(algo);
            } else {
                break;
            }
        }

        // Build market_sim GameConfig from shared_types game config
        let ms_config = build_ms_game_config(
            &sim_config.game,
            self.config.price_min,
            self.config.price_max,
            self.config.num_price_levels,
            num_rounds,
        )?;
        let market = market_sim::MarketFactory::create(&ms_config)
            .map_err(|e| CollusionError::Internal(format!("Market creation error: {}", e)))?;
        let mut outcomes = Vec::with_capacity(num_rounds);

        // Initial prices at midpoint
        let mid_price = Price((self.config.price_min.0 + self.config.price_max.0) / 2.0);
        let initial_actions: Vec<PlayerAction> = (0..num_players)
            .map(|i| PlayerAction::new(PlayerId(i), mid_price))
            .collect();
        let first_outcome = run_market_round(&*market, &initial_actions, 0)?;

        // Observe initial outcome
        for algo in &mut algorithms {
            algo.observe(&first_outcome);
        }
        outcomes.push(first_outcome);

        // Main simulation loop
        for round in 1..num_rounds {
            let actions: Vec<PlayerAction> = algorithms
                .iter_mut()
                .map(|algo| algo.act(RoundNumber(round)))
                .collect();

            let outcome = run_market_round(&*market, &actions, round)?;

            for algo in &mut algorithms {
                algo.observe(&outcome);
            }
            outcomes.push(outcome);
        }

        let market_type = sim_config.game.market_type.clone();
        let algorithm_type = sim_config.algorithm.algorithm_type.clone();
        let seed = self.config.seed.unwrap_or(0);
        Ok(PriceTrajectory::new(outcomes, market_type, num_players, algorithm_type, seed))
    }

    /// Stage 2: Analyze trajectory (basic statistics).
    fn analyze(&self, trajectory: &PriceTrajectory) -> CollusionResult<TrajectoryAnalysis> {
        let mean_price = trajectory.mean_price().0;
        let player_means: Vec<f64> = (0..trajectory.num_players)
            .map(|p| {
                let prices = trajectory.prices_for_player(PlayerId(p));
                let sum: f64 = prices.iter().map(|pr| pr.0).sum();
                sum / prices.len() as f64
            })
            .collect();

        let total_profits: Vec<f64> = (0..trajectory.num_players)
            .map(|p| {
                trajectory.outcomes.iter().map(|o| o.profits[p].0).sum()
            })
            .collect();

        // Price variance
        let overall_var = {
            let all_prices: Vec<f64> = trajectory.outcomes.iter()
                .flat_map(|o| o.prices.iter().map(|p| p.0))
                .collect();
            let n = all_prices.len() as f64;
            let mean = all_prices.iter().sum::<f64>() / n;
            all_prices.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0)
        };

        Ok(TrajectoryAnalysis {
            mean_price,
            player_means,
            total_profits,
            price_variance: overall_var,
            trajectory_length: trajectory.len(),
        })
    }

    /// Stage 3: Run detection tests.
    fn test(&self, trajectory: &PriceTrajectory) -> CollusionResult<CollusionReport> {
        let detector = CollusionDetector::new(self.config.detection_config.clone());
        detector.detect_full(trajectory)
    }

    pub fn config(&self) -> &PipelineConfig {
        &self.config
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Helpers: bridge between shared_types and market_sim type systems
// ═══════════════════════════════════════════════════════════════════════════

/// Run one round through a market_sim Market, converting between type systems.
fn run_market_round(
    market: &dyn market_sim::Market,
    actions: &[PlayerAction],
    round: usize,
) -> CollusionResult<MarketOutcome> {
    let ms_actions: Vec<market_sim::PlayerAction> = actions
        .iter()
        .map(|a| {
            market_sim::PlayerAction::new(
                a.player_id.0,
                a.price.map(|p| p.0).unwrap_or(0.0),
            )
        })
        .collect();

    let ms_outcome = market
        .simulate_round(&ms_actions, round as u64)
        .map_err(|e| CollusionError::Internal(format!("Simulation step error: {}", e)))?;

    Ok(MarketOutcome {
        round: RoundNumber(round),
        actions: actions.to_vec(),
        prices: ms_outcome.prices.iter().map(|&p| Price(p)).collect(),
        quantities: ms_outcome.quantities.iter().map(|&q| Quantity(q)).collect(),
        profits: ms_outcome.profits.iter().map(|&p| Profit(p)).collect(),
        total_surplus: Profit(ms_outcome.profits.iter().sum()),
        consumer_surplus: Profit(0.0),
    })
}

/// Build a market_sim::GameConfig from shared_types::GameConfig.
fn build_ms_game_config(
    game: &shared_types::GameConfig,
    price_min: Price,
    price_max: Price,
    grid_size: usize,
    num_rounds: usize,
) -> CollusionResult<market_sim::GameConfig> {
    let market_type = match game.market_type {
        shared_types::MarketType::Bertrand => market_sim::MarketType::Bertrand,
        shared_types::MarketType::Cournot => market_sim::MarketType::Cournot,
    };

    let (demand_system, intercept, slope, cross_slope, subst_elast, price_sens, outside_opt, mkt_size) =
        match &game.demand_system {
            DemandSystem::Linear { max_quantity, slope } => (
                market_sim::DemandSystemType::Linear,
                *max_quantity, *slope, slope * 0.5, 2.0, 1.0, 1.0, 100.0,
            ),
            DemandSystem::CES { elasticity_of_substitution, market_size, .. } => (
                market_sim::DemandSystemType::CES,
                10.0, 1.0, 0.5, *elasticity_of_substitution, 1.0, 1.0, *market_size,
            ),
            DemandSystem::Logit { temperature, outside_option_value, market_size } => (
                market_sim::DemandSystemType::Logit,
                10.0, 1.0, 0.5, 2.0, 1.0 / temperature, *outside_option_value, *market_size,
            ),
        };

    let marginal_costs: Vec<f64> = game.marginal_costs.iter().map(|c| c.0).collect();

    Ok(market_sim::GameConfig {
        market_type,
        demand_system,
        num_players: game.num_players,
        num_rounds: num_rounds as u64,
        demand_intercept: intercept,
        demand_slope: slope,
        demand_cross_slope: cross_slope,
        substitution_elasticity: subst_elast,
        price_sensitivity: price_sens,
        outside_option_value: outside_opt,
        market_size: mkt_size,
        marginal_costs,
        price_min: price_min.0,
        price_max: price_max.0,
        price_grid_size: grid_size,
        quantity_min: 0.0,
        quantity_max: price_max.0,
        quantity_grid_size: grid_size,
    })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TrajectoryAnalysis {
    mean_price: f64,
    player_means: Vec<f64>,
    total_profits: Vec<f64>,
    price_variance: f64,
    trajectory_length: usize,
}

// ═══════════════════════════════════════════════════════════════════════════
// Run full pipeline convenience function
// ═══════════════════════════════════════════════════════════════════════════

pub fn run_full_pipeline(
    scenario: &Scenario,
    config: PipelineConfig,
) -> CollusionResult<CertificationResult> {
    let pipeline = CertificationPipeline::new(config);
    pipeline.run(scenario)
}

// ═══════════════════════════════════════════════════════════════════════════
// ParallelPipelineRunner
// ═══════════════════════════════════════════════════════════════════════════

pub struct ParallelPipelineRunner {
    config: PipelineConfig,
}

impl ParallelPipelineRunner {
    pub fn new(config: PipelineConfig) -> Self {
        Self { config }
    }

    /// Run multiple scenarios sequentially (parallel with rayon would need Send bounds).
    pub fn run_all(&self, scenarios: &[Scenario]) -> Vec<CollusionResult<CertificationResult>> {
        scenarios
            .iter()
            .map(|scenario| {
                let pipeline = CertificationPipeline::new(self.config.clone());
                pipeline.run(scenario)
            })
            .collect()
    }

    /// Run all scenarios and return a summary.
    pub fn run_summary(&self, scenarios: &[Scenario]) -> PipelineRunSummary {
        let results = self.run_all(scenarios);

        let mut passed = 0;
        let mut failed = 0;
        let mut errors = 0;
        let mut total_duration = Duration::ZERO;

        for result in &results {
            match result {
                Ok(cert) => {
                    if cert.passed {
                        passed += 1;
                    } else {
                        failed += 1;
                    }
                    total_duration += cert.metrics.total_duration;
                }
                Err(_) => {
                    errors += 1;
                }
            }
        }

        PipelineRunSummary {
            total_scenarios: scenarios.len(),
            passed,
            failed,
            errors,
            total_duration,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineRunSummary {
    pub total_scenarios: usize,
    pub passed: usize,
    pub failed: usize,
    pub errors: usize,
    pub total_duration: Duration,
}

impl PipelineRunSummary {
    pub fn pass_rate(&self) -> f64 {
        if self.total_scenarios == 0 {
            return 0.0;
        }
        self.passed as f64 / self.total_scenarios as f64
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scenario::ScenarioLibrary;

    fn test_pipeline_config() -> PipelineConfig {
        PipelineConfig {
            rounds_per_simulation: 200,
            detection_config: DetectionConfig {
                min_trajectory_length: 50,
                ..DetectionConfig::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn test_pipeline_stage_all() {
        let stages = PipelineStage::all();
        assert_eq!(stages.len(), 5);
        assert_eq!(stages[0], PipelineStage::Simulate);
        assert_eq!(stages[4], PipelineStage::Verify);
    }

    #[test]
    fn test_pipeline_stage_next() {
        assert_eq!(PipelineStage::Simulate.next(), Some(PipelineStage::Analyze));
        assert_eq!(PipelineStage::Verify.next(), None);
    }

    #[test]
    fn test_pipeline_stage_display() {
        assert_eq!(format!("{}", PipelineStage::Simulate), "Simulate");
        assert_eq!(format!("{}", PipelineStage::Certify), "Certify");
    }

    #[test]
    fn test_pipeline_config_default() {
        let config = PipelineConfig::default();
        assert_eq!(config.num_simulations, 1);
        assert_eq!(config.rounds_per_simulation, 1000);
        assert!(config.enable_checkpoints);
    }

    #[test]
    fn test_pipeline_metrics() {
        let mut metrics = PipelineMetrics::new();
        assert_eq!(metrics.stages_completed, 0);

        metrics.record_stage(&PipelineStage::Simulate, Duration::from_millis(100));
        assert_eq!(metrics.stages_completed, 1);
        assert!(metrics.stage_durations.contains_key("Simulate"));
    }

    #[test]
    fn test_pipeline_checkpoint() {
        let cp = PipelineCheckpoint::new(PipelineStage::Test);
        assert_eq!(cp.stage, PipelineStage::Test);
        assert!(cp.trajectory.is_none());
        assert!(cp.detection_result.is_none());
    }

    #[test]
    fn test_run_grim_trigger_pair() {
        let lib = ScenarioLibrary::new();
        let scenario = lib.by_name("grim_trigger_pair").unwrap();
        let config = test_pipeline_config();
        let pipeline = CertificationPipeline::new(config);
        let result = pipeline.run(scenario).unwrap();
        assert!(!result.scenario_name.is_empty());
        assert!(result.metrics.stages_completed >= 3);
    }

    #[test]
    fn test_run_tft_pair() {
        let lib = ScenarioLibrary::new();
        let scenario = lib.by_name("tft_pair").unwrap();
        let config = test_pipeline_config();
        let result = run_full_pipeline(scenario, config).unwrap();
        assert!(!result.scenario_name.is_empty());
    }

    #[test]
    fn test_run_nash_equilibrium() {
        let lib = ScenarioLibrary::new();
        let scenario = lib.by_name("nash_equilibrium").unwrap();
        let config = test_pipeline_config();
        let result = run_full_pipeline(scenario, config).unwrap();
        assert!(!result.scenario_name.is_empty());
    }

    #[test]
    fn test_run_independent_bandits() {
        let lib = ScenarioLibrary::new();
        let scenario = lib.by_name("independent_bandits").unwrap();
        let config = test_pipeline_config();
        let result = run_full_pipeline(scenario, config).unwrap();
        assert!(!result.scenario_name.is_empty());
    }

    #[test]
    fn test_run_with_progress() {
        let lib = ScenarioLibrary::new();
        let scenario = lib.by_name("grim_trigger_pair").unwrap();
        let config = test_pipeline_config();

        let progress_log = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let log_clone = progress_log.clone();

        let pipeline = CertificationPipeline::new(config)
            .with_progress(Box::new(move |stage, pct| {
                log_clone.lock().unwrap().push((stage, pct));
            }));

        let result = pipeline.run(scenario).unwrap();
        assert!(!result.scenario_name.is_empty());

        let log = progress_log.lock().unwrap();
        assert!(log.len() >= 5); // At least 5 stages
    }

    #[test]
    fn test_parallel_runner_summary() {
        let lib = ScenarioLibrary::new();
        let scenarios: Vec<Scenario> = lib
            .by_category(crate::scenario::ScenarioCategory::KnownCollusive)
            .iter()
            .take(2)
            .map(|s| (*s).clone())
            .collect();

        let config = test_pipeline_config();
        let runner = ParallelPipelineRunner::new(config);
        let summary = runner.run_summary(&scenarios);

        assert_eq!(summary.total_scenarios, 2);
        assert!(summary.passed + summary.failed + summary.errors == 2);
    }

    #[test]
    fn test_pass_rate() {
        let summary = PipelineRunSummary {
            total_scenarios: 10,
            passed: 8,
            failed: 1,
            errors: 1,
            total_duration: Duration::from_secs(1),
        };
        assert!((summary.pass_rate() - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_pass_rate_empty() {
        let summary = PipelineRunSummary {
            total_scenarios: 0,
            passed: 0,
            failed: 0,
            errors: 0,
            total_duration: Duration::ZERO,
        };
        assert!((summary.pass_rate() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_certification_result_matches_expected() {
        let result = CertificationResult {
            scenario_name: "test".into(),
            verdict: Verdict::Collusive,
            confidence: 0.95,
            collusion_premium: 0.8,
            cp_confidence_interval: None,
            report: CollusionReport {
                result: DetectionResult::collusive(0.95, 0.8),
                layer0_results: None,
                layer1_results: None,
                layer2_results: None,
                duration_ms: 0,
                trajectory_length: 100,
                num_players: 2,
            },
            metrics: PipelineMetrics::new(),
            checkpoints: Vec::new(),
            passed: true,
        };
        assert!(result.matches_expected(&ExpectedVerdict::Collusive));
        assert!(!result.matches_expected(&ExpectedVerdict::Competitive));
    }

    #[test]
    fn test_pipeline_config_serialization() {
        let config = PipelineConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let restored: PipelineConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.rounds_per_simulation, 1000);
    }

    #[test]
    fn test_certification_result_serialization() {
        let result = CertificationResult {
            scenario_name: "test".into(),
            verdict: Verdict::Competitive,
            confidence: 0.95,
            collusion_premium: 0.0,
            cp_confidence_interval: None,
            report: CollusionReport {
                result: DetectionResult::competitive(),
                layer0_results: None,
                layer1_results: None,
                layer2_results: None,
                duration_ms: 0,
                trajectory_length: 100,
                num_players: 2,
            },
            metrics: PipelineMetrics::new(),
            checkpoints: Vec::new(),
            passed: true,
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("Competitive"));
    }
}
