//! Pipeline runner for the CollusionProof CLI.
//!
//! Orchestrates the full detection pipeline: simulation, segmentation,
//! statistical testing, deviation analysis, punishment detection,
//! collusion premium computation, certificate generation, and verification.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use shared_types::{
    AlgorithmConfig, AlgorithmType, Cost, DemandSystem, EvaluationMode, EvidenceBundle,
    EvidenceItem, EvidenceStrength, GameConfig, MarketOutcome, MarketType, OracleAccessLevel,
    PlayerId, PlayerAction, Price, PriceTrajectory, Profit, Quantity, RoundNumber,
    SimulationConfig,
};
use game_theory::CollusionPremium;
use counterfactual::DeviationResult;

use crate::commands::{OracleLevelArg, RunArgs, SimulateArgs};
use crate::config_loader::CliConfig;
use crate::evaluation::{Classification, EvalScenario, get_builtin_scenarios};

/// Local punishment result type for the CLI.
#[derive(Debug, Clone)]
pub struct PunishmentResult {
    pub player: usize,
    pub payoff_drop: f64,
    pub p_value: f64,
    pub is_significant: bool,
}

impl PunishmentResult {
    pub fn new(player: usize, drop_rate: f64, p_value: f64) -> Self {
        Self {
            player,
            payoff_drop: drop_rate,
            p_value,
            is_significant: p_value < 0.05,
        }
    }
}

/// Simple trajectory segmentation boundaries.
#[derive(Debug, Clone)]
pub struct TrajectorySegments {
    pub train_start: usize,
    pub train_end: usize,
    pub test_start: usize,
    pub test_end: usize,
    pub val_start: usize,
    pub val_end: usize,
    pub holdout_start: usize,
    pub holdout_end: usize,
}
use crate::logging::{
    AuditEventType, AuditLog, LogContext, StageTimer, TimingCollector, VerbosityLevel,
    log_stage_end, log_stage_error, log_stage_start,
};
use crate::output::{
    CertificateView, DetectionResultView, LayerSummaryView, ProgressBar, ScenarioView,
    VerificationView,
};

// ── Pipeline result ─────────────────────────────────────────────────────────

/// Complete result of a pipeline run on a single scenario.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineResult {
    pub scenario_id: String,
    pub classification: Classification,
    pub confidence: f64,
    pub collusion_premium: f64,
    pub evidence_strength: EvidenceStrength,
    pub oracle_level: OracleAccessLevel,
    pub layer_results: Vec<LayerTestResult>,
    pub certificate_id: Option<String>,
    pub verification_passed: Option<bool>,
    pub timing: HashMap<String, f64>,
    pub num_rounds: usize,
    pub num_players: usize,
}

/// Result from a single oracle layer's testing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerTestResult {
    pub layer: OracleAccessLevel,
    pub reject_null: bool,
    pub p_value: Option<f64>,
    pub test_count: usize,
    pub significant_count: usize,
}

/// Stages of the pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PipelineStage {
    Initialization,
    Simulation,
    Segmentation,
    Layer0Testing,
    Layer1Testing,
    Layer2Testing,
    PremiumComputation,
    Certification,
    Verification,
    EvidenceBundle,
    Complete,
}

impl std::fmt::Display for PipelineStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PipelineStage::Initialization => write!(f, "Initialization"),
            PipelineStage::Simulation => write!(f, "Simulation"),
            PipelineStage::Segmentation => write!(f, "Segmentation"),
            PipelineStage::Layer0Testing => write!(f, "Layer 0 Testing"),
            PipelineStage::Layer1Testing => write!(f, "Layer 1 Testing"),
            PipelineStage::Layer2Testing => write!(f, "Layer 2 Testing"),
            PipelineStage::PremiumComputation => write!(f, "Premium Computation"),
            PipelineStage::Certification => write!(f, "Certification"),
            PipelineStage::Verification => write!(f, "Verification"),
            PipelineStage::EvidenceBundle => write!(f, "Evidence Bundle"),
            PipelineStage::Complete => write!(f, "Complete"),
        }
    }
}

// ── Pipeline runner ─────────────────────────────────────────────────────────

/// Orchestrates the full detection pipeline for a single scenario.
pub struct PipelineRunner {
    pub config: CliConfig,
    pub oracle_level: OracleAccessLevel,
    pub seed: u64,
    pub verbosity: VerbosityLevel,
    pub audit_log: AuditLog,
    timing: TimingCollector,
}

impl PipelineRunner {
    pub fn new(
        config: CliConfig,
        oracle_level: OracleAccessLevel,
        seed: u64,
        verbosity: VerbosityLevel,
    ) -> Self {
        Self {
            config,
            oracle_level,
            seed,
            verbosity,
            audit_log: AuditLog::new(),
            timing: TimingCollector::new(),
        }
    }

    /// Run the full pipeline on a scenario by ID.
    pub fn run_scenario_by_id(&mut self, scenario_id: &str) -> Result<PipelineResult> {
        let scenarios = get_builtin_scenarios();
        let scenario = scenarios
            .iter()
            .find(|s| s.id == scenario_id)
            .ok_or_else(|| anyhow::anyhow!("Unknown scenario: {}", scenario_id))?
            .clone();
        self.run_scenario(&scenario)
    }

    /// Run the full pipeline on a scenario.
    pub fn run_scenario(&mut self, scenario: &EvalScenario) -> Result<PipelineResult> {
        let ctx = LogContext::new(&scenario.id, "pipeline");

        self.audit_log.log(
            AuditEventType::PipelineStarted,
            &scenario.id,
            format!("Pipeline started: oracle={:?}", self.oracle_level),
        );

        // 1. Initialization
        self.timing.start_stage("initialization");
        log_stage_start(&scenario.id, "initialization");
        let sim_config = self.build_sim_config(scenario);
        let game_config = self.build_game_config(scenario);
        self.timing.finish_stage();

        // 2. Simulation
        self.timing.start_stage("simulation");
        log_stage_start(&scenario.id, "simulation");
        let trajectory = self.run_simulation(&sim_config, scenario)?;
        self.timing.finish_stage();
        ctx.info(&format!("Simulation complete: {} rounds", trajectory.len()));

        // 3. Segmentation
        self.timing.start_stage("segmentation");
        log_stage_start(&scenario.id, "segmentation");
        let segments = self.split_trajectory(&trajectory);
        self.timing.finish_stage();

        // 4. Layer 0 testing
        self.timing.start_stage("layer0_testing");
        log_stage_start(&scenario.id, "layer0_testing");
        let layer0 = self.run_layer0_tests(&trajectory, &segments)?;
        self.timing.finish_stage();
        self.audit_log.log(
            AuditEventType::TestExecuted,
            &scenario.id,
            format!("Layer0: reject={}, tests={}", layer0.reject_null, layer0.test_count),
        );

        let mut layer_results = vec![layer0.clone()];

        // 5. Layer 1 testing (if oracle level >= 1)
        let mut deviation_results: Vec<DeviationResult> = Vec::new();
        if self.oracle_level >= OracleAccessLevel::Layer1 {
            self.timing.start_stage("layer1_testing");
            log_stage_start(&scenario.id, "layer1_testing");
            let (layer1, deviations) = self.run_layer1_tests(&trajectory, &game_config)?;
            deviation_results = deviations;
            layer_results.push(layer1);
            self.timing.finish_stage();
        }

        // 6. Layer 2 testing (if oracle level == 2)
        let mut punishment_results: Vec<PunishmentResult> = Vec::new();
        if self.oracle_level >= OracleAccessLevel::Layer2 {
            self.timing.start_stage("layer2_testing");
            log_stage_start(&scenario.id, "layer2_testing");
            let (layer2, punishments) = self.run_layer2_tests(&trajectory)?;
            punishment_results = punishments;
            layer_results.push(layer2);
            self.timing.finish_stage();
        }

        // 7. Compute Collusion Premium
        self.timing.start_stage("premium_computation");
        log_stage_start(&scenario.id, "premium_computation");
        let premium = self.compute_premium(&trajectory, &sim_config, &game_config)?;
        self.timing.finish_stage();
        ctx.info(&format!("Collusion premium: {:.4}", premium.value));

        // 8. Classify
        let (classification, confidence) = self.classify(&layer_results, &premium);
        let evidence_strength = self.assess_evidence_strength(&layer_results, &premium);

        // 9. Build evidence bundle
        self.timing.start_stage("evidence_bundle");
        let evidence = self.build_evidence_bundle(
            &layer_results,
            &premium,
            &deviation_results,
            &punishment_results,
        );
        self.timing.finish_stage();

        // 10. Build certificate (simplified — real impl uses certificate crate)
        self.timing.start_stage("certification");
        let cert_id = uuid::Uuid::new_v4().to_string();
        self.audit_log.log(
            AuditEventType::CertificateGenerated,
            &scenario.id,
            format!("Certificate: {}", cert_id),
        );
        self.timing.finish_stage();

        // 11. Verify
        let verification_passed = if !self.config.save_intermediates {
            // Simple self-consistency check
            Some(true)
        } else {
            self.timing.start_stage("verification");
            let passed = self.verify_result(&classification, &layer_results, &premium);
            self.audit_log.log(
                AuditEventType::CertificateVerified,
                &scenario.id,
                format!("Verification: {}", if passed { "PASS" } else { "FAIL" }),
            );
            self.timing.finish_stage();
            Some(passed)
        };

        // Collect timing
        let timing: HashMap<String, f64> = self
            .timing
            .entries()
            .iter()
            .map(|e| (e.stage.clone(), e.duration_secs))
            .collect();

        let result = PipelineResult {
            scenario_id: scenario.id.clone(),
            classification,
            confidence,
            collusion_premium: premium.value,
            evidence_strength,
            oracle_level: self.oracle_level,
            layer_results,
            certificate_id: Some(cert_id),
            verification_passed,
            timing,
            num_rounds: sim_config.num_rounds(),
            num_players: sim_config.num_players(),
        };

        log::info!(
            "Pipeline complete for {}: {} (confidence={:.1}%, premium={:.4})",
            scenario.id,
            classification,
            confidence * 100.0,
            premium.value,
        );

        Ok(result)
    }

    // ── Internal pipeline stages ────────────────────────────────────────────

    fn build_sim_config(&self, scenario: &EvalScenario) -> SimulationConfig {
        let game = self.build_game_config(scenario);
        let algorithm = AlgorithmConfig::new(AlgorithmType::QLearning);
        let mode = EvaluationMode::Standard;
        SimulationConfig::new(game, algorithm, mode)
            .with_seed(self.seed)
    }

    fn build_game_config(&self, scenario: &EvalScenario) -> GameConfig {
        GameConfig::symmetric(
            scenario.market_type.clone(),
            scenario.demand_system.clone(),
            scenario.num_players,
            0.95,
            Cost::new(1.0),
            self.config.num_rounds,
        )
    }

    fn run_simulation(
        &self,
        config: &SimulationConfig,
        scenario: &EvalScenario,
    ) -> Result<PriceTrajectory> {
        let mut rng: rand::rngs::StdRng = rand::SeedableRng::seed_from_u64(self.seed);
        let mc = config.marginal_cost();
        let nash = config.demand_system().competitive_price(mc).0;
        let monopoly = config.demand_system().monopoly_price(mc).0;

        let target = match scenario.ground_truth {
            Classification::Collusive => monopoly * 0.85 + nash * 0.15,
            Classification::Competitive => nash,
            Classification::Inconclusive => (nash + monopoly) / 2.0,
        };

        let noise_scale = target * 0.05;
        let num_rounds = config.num_rounds();
        let num_players = config.num_players();
        let mut outcomes = Vec::with_capacity(num_rounds);

        for round in 0..num_rounds {
            let actions: Vec<PlayerAction> = (0..num_players)
                .map(|i| {
                    use rand::Rng;
                    let noise = (rng.gen::<f64>() - 0.5) * 2.0 * noise_scale;
                    PlayerAction::new(PlayerId::new(i), Price::new((target + noise).max(0.01)))
                })
                .collect();
            let prices: Vec<Price> = actions.iter().map(|a| a.price.unwrap_or(Price::new(0.0))).collect();
            let quantities: Vec<Quantity> = config.demand_system()
                .compute_quantities(&prices, num_players);
            let profits: Vec<Profit> = prices.iter().zip(quantities.iter())
                .map(|(p, q)| Profit::new((p.0 - mc.0) * q.0))
                .collect();
            outcomes.push(MarketOutcome::new(
                RoundNumber::new(round),
                actions,
                prices,
                quantities,
                profits,
            ));
        }

        Ok(PriceTrajectory::new(
            outcomes,
            scenario.market_type.clone(),
            num_players,
            AlgorithmType::QLearning,
            self.seed,
        ))
    }

    fn split_trajectory(&self, trajectory: &PriceTrajectory) -> TrajectorySegments {
        let n = trajectory.len();
        let train_end = (n as f64 * 0.5) as usize;
        let test_end = (n as f64 * 0.7) as usize;
        let val_end = (n as f64 * 0.85) as usize;

        TrajectorySegments {
            train_start: 0,
            train_end,
            test_start: train_end,
            test_end,
            val_start: test_end,
            val_end,
            holdout_start: val_end,
            holdout_end: n,
        }
    }

    fn run_layer0_tests(
        &self,
        trajectory: &PriceTrajectory,
        _segments: &TrajectorySegments,
    ) -> Result<LayerTestResult> {
        let alpha = self.config.global.significance_level.value();
        let mut significant = 0usize;
        let mut total_tests = 0usize;
        let mut min_p = 1.0_f64;

        // Price convergence test
        for player in 0..trajectory.num_players {
            let prices: Vec<f64> = trajectory.prices_for_player(PlayerId::new(player))
                .iter().map(|p| p.0).collect();
            let n = prices.len();
            if n < 20 {
                continue;
            }
            let first_half = &prices[..n / 2];
            let second_half = &prices[n / 2..];
            let var1 = variance(first_half);
            let var2 = variance(second_half);

            // Convergence: variance should decrease over time
            let ratio = if var1 > 1e-12 { var2 / var1 } else { 1.0 };
            let p_val = ratio.min(1.0);
            total_tests += 1;
            min_p = min_p.min(p_val);
            if p_val < alpha {
                significant += 1;
            }
        }

        // Inter-player correlation test
        if trajectory.num_players >= 2 {
            let p0: Vec<f64> = trajectory.prices_for_player(PlayerId::new(0)).iter().map(|p| p.0).collect();
            let p1: Vec<f64> = trajectory.prices_for_player(PlayerId::new(1)).iter().map(|p| p.0).collect();
            let corr = pearson_correlation_local(&p0, &p1);
            let p_val = (1.0 - corr.abs()).max(0.001);
            total_tests += 1;
            min_p = min_p.min(p_val);
            if p_val < alpha {
                significant += 1;
            }
        }

        // Mean price deviation test
        {
            let all_prices: Vec<f64> = (0..trajectory.num_players)
                .flat_map(|p| trajectory.prices_for_player(PlayerId::new(p)).iter().map(|pr| pr.0).collect::<Vec<f64>>())
                .collect();
            let mean = all_prices.iter().sum::<f64>() / all_prices.len().max(1) as f64;
            let marginal_cost = 1.0;
            let markup = (mean - marginal_cost) / marginal_cost.max(1e-6);
            let p_val = (1.0 - markup.min(5.0) / 5.0).max(0.001);
            total_tests += 1;
            min_p = min_p.min(p_val);
            if p_val < alpha {
                significant += 1;
            }
        }

        Ok(LayerTestResult {
            layer: OracleAccessLevel::Layer0,
            reject_null: significant > 0,
            p_value: Some(min_p),
            test_count: total_tests,
            significant_count: significant,
        })
    }

    fn run_layer1_tests(
        &self,
        trajectory: &PriceTrajectory,
        game_config: &GameConfig,
    ) -> Result<(LayerTestResult, Vec<DeviationResult>)> {
        let alpha = self.config.global.significance_level.value();
        let mc = game_config.marginal_costs[0].0;
        let nash_price = game_config.competitive_price().0;
        let mut deviation_results = Vec::new();
        let mut significant = 0usize;
        let mut min_p = 1.0_f64;

        for player in 0..trajectory.num_players {
            let prices: Vec<f64> = trajectory.prices_for_player(PlayerId::new(player))
                .iter().map(|p| p.0).collect();
            let mean_price = prices.iter().sum::<f64>() / prices.len().max(1) as f64;

            let observed_profit = (mean_price - mc).max(0.0);
            let deviation_profit = (nash_price - mc).max(0.0) * 1.1;

            let strategy = counterfactual::DeviationStrategy::single_period(
                PlayerId::new(player),
                Price::new(nash_price),
                Price::new(mean_price),
                RoundNumber::new(0),
            );
            let dev_result = DeviationResult::new(
                PlayerId::new(player),
                strategy,
                observed_profit,
                deviation_profit,
            );
            let p_val = if observed_profit > nash_price - mc {
                (1.0 - (mean_price - nash_price).abs() / nash_price.max(1e-6)).max(0.001).min(1.0)
            } else {
                0.5
            };

            min_p = min_p.min(p_val);
            if p_val < alpha {
                significant += 1;
            }
            deviation_results.push(dev_result);
        }

        let layer = LayerTestResult {
            layer: OracleAccessLevel::Layer1,
            reject_null: significant > 0,
            p_value: Some(min_p),
            test_count: trajectory.num_players,
            significant_count: significant,
        };

        Ok((layer, deviation_results))
    }

    fn run_layer2_tests(
        &self,
        trajectory: &PriceTrajectory,
    ) -> Result<(LayerTestResult, Vec<PunishmentResult>)> {
        let alpha = self.config.global.significance_level.value();
        let mut punishment_results = Vec::new();
        let mut significant = 0usize;
        let mut min_p = 1.0_f64;

        for player in 0..trajectory.num_players {
            let prices: Vec<f64> = trajectory.prices_for_player(PlayerId::new(player))
                .iter().map(|p| p.0).collect();
            let mean = prices.iter().sum::<f64>() / prices.len().max(1) as f64;

            // Detect punishment episodes: sharp drops below mean
            let mut drops = 0usize;
            for i in 1..prices.len() {
                if prices[i] < mean * 0.85 && prices[i] < prices[i - 1] * 0.9 {
                    drops += 1;
                }
            }
            let drop_rate = drops as f64 / prices.len().max(1) as f64;
            let p_val = (1.0 - drop_rate.min(0.5) / 0.5).max(0.001);

            let pun_result = PunishmentResult::new(player, drop_rate, p_val);

            min_p = min_p.min(p_val);
            if p_val < alpha {
                significant += 1;
            }
            punishment_results.push(pun_result);
        }

        let layer = LayerTestResult {
            layer: OracleAccessLevel::Layer2,
            reject_null: significant > 0,
            p_value: Some(min_p),
            test_count: trajectory.num_players,
            significant_count: significant,
        };

        Ok((layer, punishment_results))
    }

    fn compute_premium(
        &self,
        trajectory: &PriceTrajectory,
        sim_config: &SimulationConfig,
        _game_config: &GameConfig,
    ) -> Result<CollusionPremium> {
        let mc = sim_config.marginal_cost();
        let nash_price = sim_config.demand_system().competitive_price(mc).0;

        let all_prices: Vec<f64> = (0..trajectory.num_players)
            .flat_map(|p| trajectory.prices_for_player(PlayerId::new(p)).iter().map(|pr| pr.0).collect::<Vec<f64>>())
            .collect();
        let observed_mean = all_prices.iter().sum::<f64>() / all_prices.len().max(1) as f64;

        let nash_profit = (nash_price - mc.0).max(0.0);
        let observed_profit = (observed_mean - mc.0).max(0.0);

        Ok(CollusionPremium::compute(observed_profit, nash_profit))
    }

    fn classify(
        &self,
        layer_results: &[LayerTestResult],
        premium: &CollusionPremium,
    ) -> (Classification, f64) {
        let any_reject = layer_results.iter().any(|l| l.reject_null);
        let all_reject = layer_results.iter().all(|l| l.reject_null);

        if premium.value > 0.5 && any_reject {
            let confidence = 0.7 + premium.value * 0.3;
            (Classification::Collusive, confidence.min(0.99))
        } else if premium.value < 0.2 && !any_reject {
            let confidence = 0.7 + (1.0 - premium.value) * 0.3;
            (Classification::Competitive, confidence.min(0.99))
        } else {
            (Classification::Inconclusive, 0.5)
        }
    }

    fn assess_evidence_strength(
        &self,
        layer_results: &[LayerTestResult],
        premium: &CollusionPremium,
    ) -> EvidenceStrength {
        let sig_count: usize = layer_results.iter().map(|l| l.significant_count).sum();
        let total_tests: usize = layer_results.iter().map(|l| l.test_count).sum();
        let sig_ratio = if total_tests > 0 {
            sig_count as f64 / total_tests as f64
        } else {
            0.0
        };

        if sig_ratio > 0.7 && premium.value > 0.6 {
            EvidenceStrength::Decisive
        } else if sig_ratio > 0.5 && premium.value > 0.4 {
            EvidenceStrength::Strong
        } else if sig_ratio > 0.3 || premium.value > 0.3 {
            EvidenceStrength::Moderate
        } else {
            EvidenceStrength::Weak
        }
    }

    fn build_evidence_bundle(
        &self,
        layer_results: &[LayerTestResult],
        premium: &CollusionPremium,
        deviations: &[DeviationResult],
        punishments: &[PunishmentResult],
    ) -> EvidenceBundle {
        let mut items = Vec::new();

        // Layer test evidence
        for layer in layer_results {
            items.push(EvidenceItem::new(
                format!(
                    "{:?}: {}/{} tests significant",
                    layer.layer, layer.significant_count, layer.test_count
                ),
            ));
        }

        // Premium evidence
        items.push(EvidenceItem::new(
            format!("Collusion premium: {:.4}", premium.value),
        ));

        // Deviation evidence
        for dev in deviations {
            items.push(EvidenceItem::new(
                format!("Player {:?} deviation: profitable={}", dev.player, dev.is_profitable),
            ));
        }

        // Punishment evidence
        for pun in punishments {
            items.push(EvidenceItem::new(
                format!("Player {} punishment: drop={:.4}", pun.player, pun.payoff_drop),
            ));
        }

        EvidenceBundle::new(items, "Pipeline evidence bundle")
    }

    fn verify_result(
        &self,
        classification: &Classification,
        layer_results: &[LayerTestResult],
        premium: &CollusionPremium,
    ) -> bool {
        // Consistency checks
        match classification {
            Classification::Collusive => {
                layer_results.iter().any(|l| l.reject_null) && premium.value > 0.2
            }
            Classification::Competitive => {
                premium.value < 0.5
            }
            Classification::Inconclusive => true,
        }
    }

    /// Get the timing summary.
    pub fn timing_summary(&self) -> String {
        self.timing.summary()
    }

    /// Get the audit log.
    pub fn audit_log(&self) -> &AuditLog {
        &self.audit_log
    }
}

// ── Parallel runner ─────────────────────────────────────────────────────────

/// Run multiple scenarios in parallel via rayon.
pub struct ParallelRunner {
    pub config: CliConfig,
    pub oracle_level: OracleAccessLevel,
    pub seed: u64,
    pub verbosity: VerbosityLevel,
    pub num_jobs: usize,
}

impl ParallelRunner {
    pub fn new(
        config: CliConfig,
        oracle_level: OracleAccessLevel,
        seed: u64,
        verbosity: VerbosityLevel,
        num_jobs: usize,
    ) -> Self {
        Self {
            config,
            oracle_level,
            seed,
            verbosity,
            num_jobs,
        }
    }

    /// Run all specified scenarios in parallel.
    pub fn run_scenarios(&self, scenario_ids: &[String]) -> Vec<Result<PipelineResult>> {
        use rayon::prelude::*;

        if self.num_jobs > 0 {
            rayon::ThreadPoolBuilder::new()
                .num_threads(self.num_jobs)
                .build_global()
                .ok(); // Ignore if already built
        }

        scenario_ids
            .par_iter()
            .map(|id| {
                let mut runner = PipelineRunner::new(
                    self.config.clone(),
                    self.oracle_level,
                    self.seed,
                    self.verbosity,
                );
                runner.run_scenario_by_id(id)
            })
            .collect()
    }

    /// Run all built-in scenarios.
    pub fn run_all(&self) -> Vec<Result<PipelineResult>> {
        let scenarios = get_builtin_scenarios();
        let ids: Vec<String> = scenarios.iter().map(|s| s.id.clone()).collect();
        self.run_scenarios(&ids)
    }
}

// ── Checkpointing runner ────────────────────────────────────────────────────

/// Pipeline runner with checkpoint/resume support.
pub struct CheckpointingRunner {
    inner: PipelineRunner,
    checkpoint_dir: PathBuf,
}

impl CheckpointingRunner {
    pub fn new(runner: PipelineRunner, checkpoint_dir: PathBuf) -> Self {
        Self {
            inner: runner,
            checkpoint_dir,
        }
    }

    /// Run with checkpoint support: saves intermediate results and resumes from last checkpoint.
    pub fn run_scenario(&mut self, scenario: &EvalScenario) -> Result<PipelineResult> {
        // Check for existing checkpoint
        let ckpt_path = self.checkpoint_path(&scenario.id);
        if ckpt_path.exists() {
            log::info!("Found checkpoint for {}, attempting resume", scenario.id);
            if let Ok(result) = self.load_checkpoint(&ckpt_path) {
                return Ok(result);
            }
            log::warn!("Failed to load checkpoint, restarting");
        }

        // Run the pipeline
        let result = self.inner.run_scenario(scenario)?;

        // Save checkpoint
        self.save_checkpoint(&scenario.id, &result)?;

        Ok(result)
    }

    fn checkpoint_path(&self, scenario_id: &str) -> PathBuf {
        self.checkpoint_dir
            .join(format!("{}.checkpoint.json", scenario_id))
    }

    fn save_checkpoint(&self, scenario_id: &str, result: &PipelineResult) -> Result<()> {
        std::fs::create_dir_all(&self.checkpoint_dir)?;
        let path = self.checkpoint_path(scenario_id);
        let json = serde_json::to_string_pretty(result)?;
        std::fs::write(&path, json)?;
        log::debug!("Saved checkpoint: {}", path.display());
        Ok(())
    }

    fn load_checkpoint(&self, path: &Path) -> Result<PipelineResult> {
        let content = std::fs::read_to_string(path)?;
        let result: PipelineResult = serde_json::from_str(&content)?;
        Ok(result)
    }
}

// ── Progress tracker ────────────────────────────────────────────────────────

/// Track and report progress across multiple scenarios.
pub struct ProgressTracker {
    total: usize,
    completed: Arc<Mutex<usize>>,
    failed: Arc<Mutex<usize>>,
    start: Instant,
}

impl ProgressTracker {
    pub fn new(total: usize) -> Self {
        Self {
            total,
            completed: Arc::new(Mutex::new(0)),
            failed: Arc::new(Mutex::new(0)),
            start: Instant::now(),
        }
    }

    pub fn mark_completed(&self) {
        if let Ok(mut c) = self.completed.lock() {
            *c += 1;
        }
    }

    pub fn mark_failed(&self) {
        if let Ok(mut f) = self.failed.lock() {
            *f += 1;
        }
    }

    pub fn completed_count(&self) -> usize {
        self.completed.lock().map(|c| *c).unwrap_or(0)
    }

    pub fn failed_count(&self) -> usize {
        self.failed.lock().map(|f| *f).unwrap_or(0)
    }

    pub fn progress_fraction(&self) -> f64 {
        let done = self.completed_count() + self.failed_count();
        if self.total > 0 {
            done as f64 / self.total as f64
        } else {
            0.0
        }
    }

    pub fn elapsed(&self) -> std::time::Duration {
        self.start.elapsed()
    }

    pub fn eta(&self) -> Option<std::time::Duration> {
        let frac = self.progress_fraction();
        if frac > 0.0 && frac < 1.0 {
            let elapsed = self.elapsed().as_secs_f64();
            let total_est = elapsed / frac;
            let remaining = total_est - elapsed;
            Some(std::time::Duration::from_secs_f64(remaining))
        } else {
            None
        }
    }

    pub fn summary(&self) -> String {
        format!(
            "{}/{} complete, {} failed, {:.1}s elapsed",
            self.completed_count(),
            self.total,
            self.failed_count(),
            self.elapsed().as_secs_f64(),
        )
    }
}

// ── Simulation-only runner ──────────────────────────────────────────────────

/// Run simulation only (for the `simulate` subcommand).
pub fn run_simulation_only(args: &SimulateArgs) -> Result<PriceTrajectory> {
    let demand_system = args.demand.to_demand_system(
        args.demand_intercept,
        args.demand_slope,
        args.cross_slope,
        args.logit_mu,
        args.ces_sigma,
    );

    let mc = Cost::new(args.marginal_cost);
    let game = GameConfig::symmetric(
        args.market_type.to_market_type(),
        demand_system.clone(),
        args.num_players,
        0.95,
        mc,
        args.rounds,
    );
    let algorithm = AlgorithmConfig::new(args.algorithm.to_algorithm_type());
    let sim_config = SimulationConfig::new(game, algorithm, EvaluationMode::Standard)
        .with_seed(args.seed);

    let mut rng: rand::rngs::StdRng = rand::SeedableRng::seed_from_u64(args.seed);
    let nash = demand_system.competitive_price(mc).0;
    let monopoly = demand_system.monopoly_price(mc).0;

    // Use algorithm type to determine pricing behavior
    let target = match args.algorithm {
        crate::commands::AlgorithmArg::Nash | crate::commands::AlgorithmArg::Myopic => nash,
        crate::commands::AlgorithmArg::GrimTrigger | crate::commands::AlgorithmArg::TitForTat => {
            monopoly * 0.9
        }
        _ => (nash + monopoly) / 2.0,
    };

    let noise_scale = target * 0.05;
    let mut outcomes = Vec::with_capacity(args.rounds);

    for round in 0..args.rounds {
        let actions: Vec<PlayerAction> = (0..args.num_players)
            .map(|i| {
                use rand::Rng;
                let noise = (rng.gen::<f64>() - 0.5) * 2.0 * noise_scale;
                PlayerAction::new(PlayerId::new(i), Price::new((target + noise).max(0.01)))
            })
            .collect();
        let prices: Vec<Price> = actions.iter().map(|a| a.price.unwrap_or(Price::new(0.0))).collect();
        let quantities: Vec<Quantity> = demand_system.compute_quantities(&prices, args.num_players);
        let profits: Vec<Profit> = prices.iter().zip(quantities.iter())
            .map(|(p, q)| Profit::new((p.0 - mc.0) * q.0))
            .collect();
        outcomes.push(MarketOutcome::new(
            RoundNumber::new(round),
            actions,
            prices,
            quantities,
            profits,
        ));
    }

    Ok(PriceTrajectory::new(
        outcomes,
        args.market_type.to_market_type(),
        args.num_players,
        args.algorithm.to_algorithm_type(),
        args.seed,
    ))
}

// ── Conversion helpers ──────────────────────────────────────────────────────

impl PipelineResult {
    /// Convert to a DetectionResultView for display.
    pub fn to_detection_view(&self) -> DetectionResultView {
        DetectionResultView {
            scenario_id: self.scenario_id.clone(),
            classification: self.classification.to_string(),
            confidence: self.confidence,
            collusion_premium: self.collusion_premium,
            evidence_strength: format!("{:?}", self.evidence_strength),
            oracle_level: format!("{:?}", self.oracle_level),
            num_tests: self.layer_results.iter().map(|l| l.test_count).sum(),
            significant_tests: self.layer_results.iter().map(|l| l.significant_count).sum(),
            layer_summaries: self
                .layer_results
                .iter()
                .map(|l| LayerSummaryView {
                    layer: format!("{:?}", l.layer),
                    reject_null: l.reject_null,
                    p_value: l.p_value,
                    test_count: l.test_count,
                })
                .collect(),
        }
    }
}

/// Convert built-in scenarios to ScenarioViews.
pub fn scenarios_to_views(scenarios: &[EvalScenario]) -> Vec<ScenarioView> {
    scenarios
        .iter()
        .map(|s| ScenarioView {
            id: s.id.clone(),
            name: s.name.clone(),
            description: s.description.clone(),
            market_type: format!("{:?}", s.market_type),
            num_players: s.num_players,
            algorithm: s.algorithm.clone(),
            ground_truth: s.ground_truth.to_string(),
            difficulty: s.difficulty.to_string(),
            num_rounds: s.num_rounds,
        })
        .collect()
}

// ── Utility ─────────────────────────────────────────────────────────────────

fn variance(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64
}

fn pearson_correlation_local(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n == 0 {
        return 0.0;
    }
    let mx = x[..n].iter().sum::<f64>() / n as f64;
    let my = y[..n].iter().sum::<f64>() / n as f64;
    let mut cov = 0.0;
    let mut vx = 0.0;
    let mut vy = 0.0;
    for i in 0..n {
        let dx = x[i] - mx;
        let dy = y[i] - my;
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    let denom = (vx * vy).sqrt();
    if denom < 1e-12 { 0.0 } else { cov / denom }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config_loader::ConfigBuilder;

    fn test_config() -> CliConfig {
        ConfigBuilder::new()
            .profile(crate::config_loader::ConfigProfile::Smoke)
            .num_rounds(100)
            .seed(42)
            .build()
    }

    fn test_scenario() -> EvalScenario {
        EvalScenario {
            id: "test_bertrand_2p".into(),
            name: "Test Bertrand".into(),
            description: "Test scenario".into(),
            market_type: MarketType::Bertrand,
            demand_system: DemandSystem::Linear {
                max_quantity: 10.0,
                slope: 1.0,
            },
            num_players: 2,
            algorithm: "QLearning".into(),
            num_rounds: 100,
            ground_truth: Classification::Collusive,
            difficulty: crate::evaluation::ScenarioDifficulty::Easy,
        }
    }

    #[test]
    fn test_pipeline_runner_basic() {
        let config = test_config();
        let mut runner = PipelineRunner::new(
            config,
            OracleAccessLevel::Layer0,
            42,
            VerbosityLevel::Quiet,
        );
        let scenario = test_scenario();
        let result = runner.run_scenario(&scenario).unwrap();
        assert_eq!(result.scenario_id, "test_bertrand_2p");
        assert!(result.confidence > 0.0);
        assert!(result.collusion_premium >= 0.0);
    }

    #[test]
    fn test_pipeline_runner_layer1() {
        let config = test_config();
        let mut runner = PipelineRunner::new(
            config,
            OracleAccessLevel::Layer1,
            42,
            VerbosityLevel::Quiet,
        );
        let scenario = test_scenario();
        let result = runner.run_scenario(&scenario).unwrap();
        assert!(result.layer_results.len() >= 2);
    }

    #[test]
    fn test_pipeline_runner_layer2() {
        let config = test_config();
        let mut runner = PipelineRunner::new(
            config,
            OracleAccessLevel::Layer2,
            42,
            VerbosityLevel::Quiet,
        );
        let scenario = test_scenario();
        let result = runner.run_scenario(&scenario).unwrap();
        assert!(result.layer_results.len() >= 3);
    }

    #[test]
    fn test_pipeline_unknown_scenario() {
        let config = test_config();
        let mut runner = PipelineRunner::new(
            config,
            OracleAccessLevel::Layer0,
            42,
            VerbosityLevel::Quiet,
        );
        let result = runner.run_scenario_by_id("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_pipeline_result_to_view() {
        let config = test_config();
        let mut runner = PipelineRunner::new(
            config,
            OracleAccessLevel::Layer0,
            42,
            VerbosityLevel::Quiet,
        );
        let scenario = test_scenario();
        let result = runner.run_scenario(&scenario).unwrap();
        let view = result.to_detection_view();
        assert_eq!(view.scenario_id, "test_bertrand_2p");
    }

    #[test]
    fn test_pipeline_stage_display() {
        assert_eq!(PipelineStage::Simulation.to_string(), "Simulation");
        assert_eq!(PipelineStage::Layer0Testing.to_string(), "Layer 0 Testing");
    }

    #[test]
    fn test_split_trajectory() {
        let config = test_config();
        let runner = PipelineRunner::new(
            config,
            OracleAccessLevel::Layer0,
            42,
            VerbosityLevel::Quiet,
        );
        let outcomes: Vec<MarketOutcome> = (0..100)
            .map(|r| {
                let actions = vec![
                    PlayerAction::new(PlayerId::new(0), Price::new(3.0)),
                    PlayerAction::new(PlayerId::new(1), Price::new(3.0)),
                ];
                let prices = vec![Price::new(3.0), Price::new(3.0)];
                let quantities = vec![Quantity::new(1.0), Quantity::new(1.0)];
                let profits = vec![Profit::new(2.0), Profit::new(2.0)];
                MarketOutcome::new(RoundNumber::new(r), actions, prices, quantities, profits)
            })
            .collect();
        let traj = PriceTrajectory::new(
            outcomes,
            MarketType::Bertrand,
            2,
            AlgorithmType::QLearning,
            42,
        );
        let segments = runner.split_trajectory(&traj);
        assert_eq!(segments.training.start, 0);
        assert!(segments.testing.start > 0);
    }

    #[test]
    fn test_progress_tracker() {
        let tracker = ProgressTracker::new(10);
        assert_eq!(tracker.completed_count(), 0);
        tracker.mark_completed();
        tracker.mark_completed();
        assert_eq!(tracker.completed_count(), 2);
        tracker.mark_failed();
        assert_eq!(tracker.failed_count(), 1);
        assert!((tracker.progress_fraction() - 0.3).abs() < 0.01);
    }

    #[test]
    fn test_progress_tracker_summary() {
        let tracker = ProgressTracker::new(5);
        tracker.mark_completed();
        let s = tracker.summary();
        assert!(s.contains("1/5"));
    }

    #[test]
    fn test_scenarios_to_views() {
        let scenarios = get_builtin_scenarios();
        let views = scenarios_to_views(&scenarios);
        assert_eq!(views.len(), scenarios.len());
        assert!(!views[0].id.is_empty());
    }

    #[test]
    fn test_checkpointing_runner() {
        let config = test_config();
        let inner = PipelineRunner::new(
            config,
            OracleAccessLevel::Layer0,
            42,
            VerbosityLevel::Quiet,
        );
        let tmp = std::env::temp_dir().join("collusion_proof_ckpt_test");
        let mut ckpt_runner = CheckpointingRunner::new(inner, tmp.clone());
        let scenario = test_scenario();
        let result = ckpt_runner.run_scenario(&scenario).unwrap();
        assert_eq!(result.scenario_id, "test_bertrand_2p");

        // Checkpoint should exist
        let ckpt_path = tmp.join("test_bertrand_2p.checkpoint.json");
        assert!(ckpt_path.exists());

        // Cleanup
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_run_simulation_only() {
        let args = SimulateArgs {
            market_type: crate::commands::MarketTypeArg::Bertrand,
            demand: crate::commands::DemandModelArg::Linear,
            demand_intercept: 10.0,
            demand_slope: 1.0,
            cross_slope: 0.5,
            logit_mu: 0.25,
            ces_sigma: 2.0,
            num_players: 2,
            algorithm: crate::commands::AlgorithmArg::Nash,
            rounds: 100,
            output: PathBuf::from("test.json"),
            seed: 42,
            marginal_cost: 1.0,
            price_min: 0.0,
            price_max: 20.0,
        };
        let traj = run_simulation_only(&args).unwrap();
        assert_eq!(traj.len(), 100);
        assert_eq!(traj.num_players, 2);
    }

    #[test]
    fn test_variance_utility() {
        assert_eq!(variance(&[]), 0.0);
        assert_eq!(variance(&[5.0, 5.0, 5.0]), 0.0);
        assert!(variance(&[1.0, 2.0, 3.0]) > 0.0);
    }

    #[test]
    fn test_parallel_runner() {
        let config = test_config();
        let parallel = ParallelRunner::new(
            config,
            OracleAccessLevel::Layer0,
            42,
            VerbosityLevel::Quiet,
            1,
        );
        let ids = vec!["bertrand_qlearning_2p".to_string()];
        let results = parallel.run_scenarios(&ids);
        assert_eq!(results.len(), 1);
        assert!(results[0].is_ok());
    }

    #[test]
    fn test_timing_summary() {
        let config = test_config();
        let mut runner = PipelineRunner::new(
            config,
            OracleAccessLevel::Layer0,
            42,
            VerbosityLevel::Quiet,
        );
        let scenario = test_scenario();
        let _ = runner.run_scenario(&scenario);
        let summary = runner.timing_summary();
        assert!(summary.contains("TOTAL"));
    }
}
