//! Core collusion detection pipeline.
//!
//! Implements the tiered detection approach (Layer 0/1/2) with
//! passive observation, deviation analysis, and punishment detection.

use crate::algorithm::{CheckpointOracle, OracleInterface, RewindOracle};
use serde::{Deserialize, Serialize};
use shared_types::{
    AlgorithmType, CollusionError, CollusionResult, ConfidenceInterval, EffectSize,
    EvidenceBundle, EvidenceItem, HypothesisTestResult, MarketOutcome, MarketType,
    OracleAccessLevel, PValue, PlayerId, PlayerAction, Price, PriceTrajectory, Profit, Quantity,
    RoundNumber, TestBattery, TestStatistic,
};
use std::time::Instant;

// ═══════════════════════════════════════════════════════════════════════════
// Verdict
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Verdict {
    Competitive,
    Inconclusive,
    Collusive,
}

impl Verdict {
    pub fn is_collusive(&self) -> bool {
        matches!(self, Verdict::Collusive)
    }

    pub fn is_competitive(&self) -> bool {
        matches!(self, Verdict::Competitive)
    }
}

impl std::fmt::Display for Verdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Verdict::Competitive => write!(f, "Competitive"),
            Verdict::Inconclusive => write!(f, "Inconclusive"),
            Verdict::Collusive => write!(f, "Collusive"),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Detection result and report
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionResult {
    pub verdict: Verdict,
    pub confidence: f64,
    pub collusion_premium_estimate: f64,
    pub cp_confidence_interval: Option<ConfidenceInterval>,
    pub evidence: EvidenceBundle,
    pub layer_reached: OracleAccessLevel,
    pub test_battery: TestBattery,
}

impl DetectionResult {
    pub fn competitive() -> Self {
        Self {
            verdict: Verdict::Competitive,
            confidence: 0.95,
            collusion_premium_estimate: 0.0,
            cp_confidence_interval: None,
            evidence: EvidenceBundle::new(vec![], "competitive"),
            layer_reached: OracleAccessLevel::Layer0,
            test_battery: TestBattery::new("default"),
        }
    }

    pub fn collusive(confidence: f64, cp: f64) -> Self {
        Self {
            verdict: Verdict::Collusive,
            confidence,
            collusion_premium_estimate: cp,
            cp_confidence_interval: None,
            evidence: EvidenceBundle::new(vec![], "collusive"),
            layer_reached: OracleAccessLevel::Layer0,
            test_battery: TestBattery::new("default"),
        }
    }

    pub fn inconclusive() -> Self {
        Self {
            verdict: Verdict::Inconclusive,
            confidence: 0.0,
            collusion_premium_estimate: 0.0,
            cp_confidence_interval: None,
            evidence: EvidenceBundle::new(vec![], "inconclusive"),
            layer_reached: OracleAccessLevel::Layer0,
            test_battery: TestBattery::new("default"),
        }
    }
}

/// Full collusion report with all test results and evidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollusionReport {
    pub result: DetectionResult,
    pub layer0_results: Option<Layer0Results>,
    pub layer1_results: Option<Layer1Results>,
    pub layer2_results: Option<Layer2Results>,
    pub duration_ms: u128,
    pub trajectory_length: usize,
    pub num_players: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Layer0Results {
    pub price_correlation: Option<HypothesisTestResult>,
    pub variance_ratio: Option<HypothesisTestResult>,
    pub mean_price_test: Option<HypothesisTestResult>,
    pub price_stability: f64,
    pub estimated_cp: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Layer1Results {
    pub deviation_response_detected: bool,
    pub deviation_tests: Vec<HypothesisTestResult>,
    pub price_recovery_time: Option<usize>,
    pub cp_after_deviation: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Layer2Results {
    pub punishment_detected: bool,
    pub punishment_severity: f64,
    pub punishment_duration: usize,
    pub forgiveness_detected: bool,
    pub tight_cp_estimate: f64,
    pub tight_cp_ci: Option<ConfidenceInterval>,
}

// ═══════════════════════════════════════════════════════════════════════════
// Detection configuration
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionConfig {
    pub significance_level: f64,
    pub max_oracle_level: OracleAccessLevel,
    pub nash_price: Price,
    pub monopoly_price: Price,
    pub competitive_price: Price,
    pub cp_threshold: f64,
    pub price_stability_window: usize,
    pub deviation_magnitude: f64,
    pub min_trajectory_length: usize,
    pub early_termination: bool,
}

impl Default for DetectionConfig {
    fn default() -> Self {
        Self {
            significance_level: 0.05,
            max_oracle_level: OracleAccessLevel::Layer2,
            nash_price: Price(3.0),
            monopoly_price: Price(5.5),
            competitive_price: Price(1.0),
            cp_threshold: 0.3,
            price_stability_window: 50,
            deviation_magnitude: 2.0,
            min_trajectory_length: 100,
            early_termination: true,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Layer 0 Detector (passive observation)
// ═══════════════════════════════════════════════════════════════════════════

pub struct Layer0Detector {
    config: DetectionConfig,
}

impl Layer0Detector {
    pub fn new(config: DetectionConfig) -> Self {
        Self { config }
    }

    pub fn detect(&self, trajectory: &PriceTrajectory) -> (Verdict, Layer0Results, EvidenceBundle) {
        let mut evidence = EvidenceBundle::new(vec![], "layer0 detection");
        let mut battery = TestBattery::new("layer0");

        // Price correlation test
        let corr_result = self.test_price_correlation(trajectory);
        battery.add_test(corr_result.clone());

        // Variance ratio test
        let var_result = self.test_variance_ratio(trajectory);
        battery.add_test(var_result.clone());

        // Mean price relative to Nash
        let mean_result = self.test_mean_price(trajectory);
        battery.add_test(mean_result.clone());

        // Price stability
        let stability = self.compute_price_stability(trajectory);

        // Estimate collusion premium
        let estimated_cp = self.estimate_collusion_premium(trajectory);

        // Determine verdict based on composite test
        let num_reject = [&corr_result, &var_result, &mean_result]
            .iter()
            .filter(|r| r.reject)
            .count();

        let verdict = if estimated_cp > self.config.cp_threshold && num_reject >= 2 {
            evidence.add(EvidenceItem::new(
                "Multiple tests reject competitive null",
            ));
            Verdict::Collusive
        } else if num_reject == 0 && estimated_cp < 0.1 {
            evidence.add(EvidenceItem::new(
                "No evidence of supracompetitive pricing",
            ));
            Verdict::Competitive
        } else {
            evidence.add(EvidenceItem::new(
                "Mixed signals from Layer 0 tests",
            ));
            Verdict::Inconclusive
        };

        let results = Layer0Results {
            price_correlation: Some(corr_result),
            variance_ratio: Some(var_result),
            mean_price_test: Some(mean_result),
            price_stability: stability,
            estimated_cp,
        };

        (verdict, results, evidence)
    }

    fn test_price_correlation(&self, trajectory: &PriceTrajectory) -> HypothesisTestResult {
        if trajectory.num_players < 2 || trajectory.len() < 2 {
            return HypothesisTestResult::new(
                "price_correlation",
                TestStatistic::z_score(0.0),
                PValue::new_unchecked(1.0),
                self.config.significance_level,
            );
        }

        let p0: Vec<f64> = trajectory.prices_for_player(PlayerId(0)).iter().map(|p| p.0).collect();
        let p1: Vec<f64> = trajectory.prices_for_player(PlayerId(1)).iter().map(|p| p.0).collect();
        let n = p0.len() as f64;

        let mean0: f64 = p0.iter().sum::<f64>() / n;
        let mean1: f64 = p1.iter().sum::<f64>() / n;

        let mut cov = 0.0_f64;
        let mut var0 = 0.0_f64;
        let mut var1 = 0.0_f64;
        for i in 0..p0.len() {
            let d0 = p0[i] - mean0;
            let d1 = p1[i] - mean1;
            cov += d0 * d1;
            var0 += d0 * d0;
            var1 += d1 * d1;
        }

        let corr = if var0 > 0.0 && var1 > 0.0 {
            cov / (var0.sqrt() * var1.sqrt())
        } else {
            0.0
        };

        // Fisher z-transform for p-value
        let z = 0.5 * ((1.0 + corr) / (1.0 - corr).max(1e-10)).ln();
        let se = 1.0 / ((n - 3.0).max(1.0)).sqrt();
        let test_stat = z / se;

        // Approximate p-value from normal distribution
        let p_value = 2.0 * (1.0 - normal_cdf(test_stat.abs()));

        HypothesisTestResult::new(
            "price_correlation",
            TestStatistic::z_score(corr),
            PValue::new_unchecked(p_value),
            self.config.significance_level,
        )
        .with_effect_size(EffectSize::cohen_d(corr.abs()))
    }

    fn test_variance_ratio(&self, trajectory: &PriceTrajectory) -> HypothesisTestResult {
        if trajectory.len() < 2 {
            return HypothesisTestResult::new(
                "variance_ratio",
                TestStatistic::z_score(0.0),
                PValue::new_unchecked(1.0),
                self.config.significance_level,
            );
        }

        // Compare price variance in first half vs second half
        let mid = trajectory.len() / 2;
        let first_half: Vec<f64> = trajectory.slice(RoundNumber(0), RoundNumber(mid))
            .iter().flat_map(|o| o.prices.iter().map(|p| p.0)).collect();
        let second_half: Vec<f64> = trajectory.slice(RoundNumber(mid), RoundNumber(trajectory.len()))
            .iter().flat_map(|o| o.prices.iter().map(|p| p.0)).collect();

        let var_first = compute_f64_variance(&first_half);
        let var_second = compute_f64_variance(&second_half);

        let ratio = if var_first > 1e-10 {
            var_second / var_first
        } else {
            1.0
        };

        // Low variance ratio suggests price convergence (potential collusion signal)
        let is_low = ratio < 0.5;
        let p_value = if is_low { 0.01 } else { 0.5 };

        HypothesisTestResult::new(
            "variance_ratio",
            TestStatistic::z_score(ratio),
            PValue::new_unchecked(p_value),
            self.config.significance_level,
        )
        .with_effect_size(EffectSize::cohen_d((1.0 - ratio).abs()))
    }

    fn test_mean_price(&self, trajectory: &PriceTrajectory) -> HypothesisTestResult {
        let mean = trajectory.mean_price().0;
        let nash = self.config.nash_price.0;
        let monopoly = self.config.monopoly_price.0;

        // How far is the mean price from Nash toward monopoly?
        let range = monopoly - nash;
        let relative_position = if range.abs() > 1e-10 {
            ((mean - nash) / range).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Simple t-test: is mean price significantly above Nash?
        let prices: Vec<f64> = (0..trajectory.num_players)
            .flat_map(|p| trajectory.prices_for_player(PlayerId(p)).into_iter().map(|pr| pr.0))
            .collect();
        let n = prices.len() as f64;
        let sample_mean: f64 = prices.iter().sum::<f64>() / n;
        let variance: f64 = prices.iter().map(|p| (p - sample_mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
        let se = (variance / n).sqrt();
        let t_stat = if se > 1e-10 { (sample_mean - nash) / se } else { 0.0 };
        let p_value = 1.0 - normal_cdf(t_stat);

        HypothesisTestResult::new(
            "mean_price_above_nash",
            TestStatistic::t_score(t_stat, (n - 1.0).max(1.0)),
            PValue::new_unchecked(p_value),
            self.config.significance_level,
        )
        .with_effect_size(EffectSize::cohen_d(relative_position))
        .with_ci(ConfidenceInterval::new(
            sample_mean - 1.96 * se,
            sample_mean + 1.96 * se,
            0.95,
            sample_mean,
        ))
    }

    fn compute_price_stability(&self, trajectory: &PriceTrajectory) -> f64 {
        if trajectory.len() < 2 {
            return 0.0;
        }
        let window = self.config.price_stability_window.min(trajectory.len());
        let recent = trajectory.slice(
            RoundNumber(trajectory.len() - window),
            RoundNumber(trajectory.len()),
        );
        let prices: Vec<f64> = recent.iter().flat_map(|o| o.prices.iter().map(|p| p.0)).collect();
        let var = compute_f64_variance(&prices);
        // Stability = 1 / (1 + variance) — higher means more stable
        1.0 / (1.0 + var)
    }

    fn estimate_collusion_premium(&self, trajectory: &PriceTrajectory) -> f64 {
        let mean = trajectory.mean_price().0;
        let nash = self.config.nash_price.0;
        let monopoly = self.config.monopoly_price.0;
        let range = monopoly - nash;
        if range.abs() < 1e-10 {
            return 0.0;
        }
        ((mean - nash) / range).clamp(0.0, 1.0)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Layer 1 Detector (deviation analysis)
// ═══════════════════════════════════════════════════════════════════════════

pub struct Layer1Detector {
    config: DetectionConfig,
}

impl Layer1Detector {
    pub fn new(config: DetectionConfig) -> Self {
        Self { config }
    }

    pub fn detect(
        &self,
        trajectory: &PriceTrajectory,
        oracle: &mut CheckpointOracle,
    ) -> (Verdict, Layer1Results, EvidenceBundle) {
        let mut evidence = EvidenceBundle::new(vec![], "layer1 detection");
        let n = trajectory.len();
        let num_players = trajectory.num_players;

        // Insert deviations at checkpoints and observe response
        let deviation_round = n / 2;
        if deviation_round < n {
            for player in 0..num_players {
                let _ = oracle.insert_deviation(
                    RoundNumber(deviation_round),
                    PlayerId(player),
                    self.config.nash_price - Price(self.config.deviation_magnitude),
                );
            }
        }

        // Check for price recovery pattern
        let recovery_time = self.detect_price_recovery(trajectory, deviation_round);
        let deviation_response = recovery_time.is_some();

        // Post-deviation CP
        let post_dev_outcomes: Vec<&MarketOutcome> = if deviation_round + 10 < n {
            trajectory.slice(RoundNumber(deviation_round + 10), RoundNumber(n))
        } else {
            trajectory.slice(RoundNumber(n.saturating_sub(10)), RoundNumber(n))
        };
        let cp_after = self.estimate_cp_from_outcomes(&post_dev_outcomes);

        let mut tests = Vec::new();
        tests.push(HypothesisTestResult::new(
            "deviation_response",
            TestStatistic::z_score(if deviation_response { 1.0 } else { 0.0 }),
            PValue::new_unchecked(if deviation_response { 0.01 } else { 0.5 }),
            self.config.significance_level,
        ));

        let verdict = if deviation_response && cp_after > self.config.cp_threshold {
            evidence.add(EvidenceItem::new(
                "Deviation response detected with high post-deviation CP",
            ));
            Verdict::Collusive
        } else if !deviation_response && cp_after < 0.1 {
            evidence.add(EvidenceItem::new(
                "No deviation response, low CP",
            ));
            Verdict::Competitive
        } else {
            evidence.add(EvidenceItem::new(
                "Partial deviation response",
            ));
            Verdict::Inconclusive
        };

        let results = Layer1Results {
            deviation_response_detected: deviation_response,
            deviation_tests: tests,
            price_recovery_time: recovery_time,
            cp_after_deviation: cp_after,
        };

        (verdict, results, evidence)
    }

    fn detect_price_recovery(
        &self,
        trajectory: &PriceTrajectory,
        deviation_round: usize,
    ) -> Option<usize> {
        let pre_mean: f64 = if deviation_round > 10 {
            let pre = trajectory.slice(
                RoundNumber(deviation_round.saturating_sub(10)),
                RoundNumber(deviation_round),
            );
            pre.iter().map(|o| o.mean_price().0).sum::<f64>() / pre.len().max(1) as f64
        } else {
            trajectory.mean_price().0
        };

        let threshold = pre_mean * 0.95;
        for offset in 1..50 {
            let round = deviation_round + offset;
            if round >= trajectory.len() {
                break;
            }
            let outcome = &trajectory.outcomes[round];
            let avg_price: f64 = outcome.prices.iter().map(|p| p.0).sum::<f64>() / outcome.prices.len() as f64;
            if avg_price >= threshold {
                return Some(offset);
            }
        }
        None
    }

    fn estimate_cp_from_outcomes(&self, outcomes: &[&MarketOutcome]) -> f64 {
        if outcomes.is_empty() {
            return 0.0;
        }
        let mean: f64 = outcomes.iter().map(|o| o.mean_price().0).sum::<f64>() / outcomes.len() as f64;
        let nash = self.config.nash_price.0;
        let monopoly = self.config.monopoly_price.0;
        let range = monopoly - nash;
        if range.abs() < 1e-10 {
            return 0.0;
        }
        ((mean - nash) / range).clamp(0.0, 1.0)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Layer 2 Detector (punishment detection)
// ═══════════════════════════════════════════════════════════════════════════

pub struct Layer2Detector {
    config: DetectionConfig,
}

impl Layer2Detector {
    pub fn new(config: DetectionConfig) -> Self {
        Self { config }
    }

    pub fn detect(
        &self,
        trajectory: &PriceTrajectory,
        oracle: &mut RewindOracle,
    ) -> (Verdict, Layer2Results, EvidenceBundle) {
        let mut evidence = EvidenceBundle::new(vec![], "layer2 detection");

        // Detect punishment patterns in the trajectory
        let (punishment_detected, severity, duration) = self.detect_punishment(trajectory);

        // Detect forgiveness
        let forgiveness = self.detect_forgiveness(trajectory);

        // Compute tight CP with full oracle access
        let tight_cp = self.compute_tight_cp(trajectory, oracle);
        let tight_ci = Some(ConfidenceInterval::new(
            (tight_cp - 0.05).max(0.0),
            (tight_cp + 0.05).min(1.0),
            0.95,
            tight_cp,
        ));

        let verdict = if punishment_detected && tight_cp > self.config.cp_threshold {
            evidence.add(EvidenceItem::new(
                "Punishment mechanism detected with tight CP estimate",
            ));
            Verdict::Collusive
        } else if !punishment_detected && tight_cp < 0.1 {
            evidence.add(EvidenceItem::new(
                "No punishment mechanism, low CP",
            ));
            Verdict::Competitive
        } else {
            evidence.add(EvidenceItem::new(
                "Ambiguous punishment signals",
            ));
            Verdict::Inconclusive
        };

        let results = Layer2Results {
            punishment_detected,
            punishment_severity: severity,
            punishment_duration: duration,
            forgiveness_detected: forgiveness,
            tight_cp_estimate: tight_cp,
            tight_cp_ci: tight_ci,
        };

        (verdict, results, evidence)
    }

    fn detect_punishment(&self, trajectory: &PriceTrajectory) -> (bool, f64, usize) {
        if trajectory.len() < 20 {
            return (false, 0.0, 0);
        }

        let mean_price = trajectory.mean_price().0;
        let nash = self.config.nash_price.0;

        // Look for price drops followed by recovery
        let mut max_drop = 0.0_f64;
        let mut drop_duration = 0usize;
        let mut in_drop = false;
        let mut current_drop_len = 0usize;

        for outcome in &trajectory.outcomes {
            let avg_price: f64 = outcome.prices.iter().map(|p| p.0).sum::<f64>() / outcome.prices.len() as f64;
            if avg_price < mean_price * 0.8 {
                in_drop = true;
                current_drop_len += 1;
                let drop = mean_price - avg_price;
                if drop > max_drop {
                    max_drop = drop;
                }
            } else if in_drop {
                if current_drop_len > drop_duration {
                    drop_duration = current_drop_len;
                }
                in_drop = false;
                current_drop_len = 0;
            }
        }

        let severity = if mean_price > nash {
            max_drop / (mean_price - nash).max(1e-10)
        } else {
            0.0
        };

        (drop_duration >= 3, severity.min(1.0), drop_duration)
    }

    fn detect_forgiveness(&self, trajectory: &PriceTrajectory) -> bool {
        if trajectory.len() < 30 {
            return false;
        }
        // Check if prices recover after a punishment episode
        let last_quarter = trajectory.slice(
            RoundNumber(trajectory.len() * 3 / 4),
            RoundNumber(trajectory.len()),
        );
        let first_quarter = trajectory.slice(
            RoundNumber(0),
            RoundNumber(trajectory.len() / 4),
        );
        let last_mean: f64 = last_quarter.iter().map(|o| o.mean_price().0).sum::<f64>()
            / last_quarter.len().max(1) as f64;
        let first_mean: f64 = first_quarter.iter().map(|o| o.mean_price().0).sum::<f64>()
            / first_quarter.len().max(1) as f64;
        // Forgiveness = prices return to cooperative level
        (last_mean - first_mean).abs() < first_mean * 0.1
    }

    fn compute_tight_cp(
        &self,
        trajectory: &PriceTrajectory,
        _oracle: &mut RewindOracle,
    ) -> f64 {
        // Use the full trajectory with oracle access for tight CP
        let mean = trajectory.mean_price().0;
        let nash = self.config.nash_price.0;
        let monopoly = self.config.monopoly_price.0;
        let range = monopoly - nash;
        if range.abs() < 1e-10 {
            return 0.0;
        }
        ((mean - nash) / range).clamp(0.0, 1.0)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Detection Pipeline
// ═══════════════════════════════════════════════════════════════════════════

/// Pipeline orchestrating Layer0 -> optional Layer1 -> optional Layer2.
pub struct DetectionPipeline {
    config: DetectionConfig,
}

impl DetectionPipeline {
    pub fn new(config: DetectionConfig) -> Self {
        Self { config }
    }

    pub fn run(&self, trajectory: &PriceTrajectory) -> CollusionResult<CollusionReport> {
        let start = Instant::now();

        // Layer 0: Passive observation
        let l0 = Layer0Detector::new(self.config.clone());
        let (l0_verdict, l0_results, mut evidence) = l0.detect(trajectory);

        let mut result = DetectionResult {
            verdict: l0_verdict,
            confidence: 0.0,
            collusion_premium_estimate: l0_results.estimated_cp,
            cp_confidence_interval: None,
            evidence: evidence.clone(),
            layer_reached: OracleAccessLevel::Layer0,
            test_battery: TestBattery::new("pipeline"),
        };

        let mut l1_results = None;
        let mut l2_results = None;

        // Early termination on clear verdict
        if self.config.early_termination && l0_verdict != Verdict::Inconclusive {
            result.confidence = 0.7;
            return Ok(CollusionReport {
                result,
                layer0_results: Some(l0_results),
                layer1_results: None,
                layer2_results: None,
                duration_ms: start.elapsed().as_millis(),
                trajectory_length: trajectory.len(),
                num_players: trajectory.num_players,
            });
        }

        // Layer 1: Deviation analysis (if allowed)
        if self.config.max_oracle_level >= OracleAccessLevel::Layer1 {
            let mut oracle = CheckpointOracle::new(trajectory.num_players, 50);
            oracle.observe_trajectory(trajectory);

            let l1 = Layer1Detector::new(self.config.clone());
            let (l1_verdict, l1_res, l1_evidence) = l1.detect(trajectory, &mut oracle);
            for item in l1_evidence.items {
                evidence.add(item);
            }
            l1_results = Some(l1_res);

            result.verdict = l1_verdict;
            result.layer_reached = OracleAccessLevel::Layer1;
            result.confidence = 0.8;

            if self.config.early_termination && l1_verdict != Verdict::Inconclusive {
                result.evidence = evidence;
                return Ok(CollusionReport {
                    result,
                    layer0_results: Some(l0_results),
                    layer1_results: l1_results,
                    layer2_results: None,
                    duration_ms: start.elapsed().as_millis(),
                    trajectory_length: trajectory.len(),
                    num_players: trajectory.num_players,
                });
            }
        }

        // Layer 2: Punishment detection (if allowed)
        if self.config.max_oracle_level >= OracleAccessLevel::Layer2 {
            let mut oracle = RewindOracle::new(trajectory.num_players);
            oracle.observe_trajectory(trajectory);

            let l2 = Layer2Detector::new(self.config.clone());
            let (l2_verdict, l2_res, l2_evidence) = l2.detect(trajectory, &mut oracle);
            for item in l2_evidence.items {
                evidence.add(item);
            }

            result.verdict = l2_verdict;
            result.layer_reached = OracleAccessLevel::Layer2;
            result.confidence = 0.95;
            if let Some(ref ci) = l2_res.tight_cp_ci {
                result.cp_confidence_interval = Some(ci.clone());
            }
            result.collusion_premium_estimate = l2_res.tight_cp_estimate;
            l2_results = Some(l2_res);
        }

        result.evidence = evidence;

        Ok(CollusionReport {
            result,
            layer0_results: Some(l0_results),
            layer1_results: l1_results,
            layer2_results: l2_results,
            duration_ms: start.elapsed().as_millis(),
            trajectory_length: trajectory.len(),
            num_players: trajectory.num_players,
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CollusionDetector (main entry point)
// ═══════════════════════════════════════════════════════════════════════════

pub struct CollusionDetector {
    config: DetectionConfig,
}

impl CollusionDetector {
    pub fn new(config: DetectionConfig) -> Self {
        Self { config }
    }

    pub fn with_defaults() -> Self {
        Self::new(DetectionConfig::default())
    }

    /// Run detection on a price trajectory.
    pub fn detect(&self, trajectory: &PriceTrajectory) -> CollusionResult<DetectionResult> {
        if trajectory.len() < self.config.min_trajectory_length {
            return Err(CollusionError::InvalidState(format!(
                "Trajectory too short: {} < {} minimum",
                trajectory.len(),
                self.config.min_trajectory_length
            )));
        }

        let pipeline = DetectionPipeline::new(self.config.clone());
        let report = pipeline.run(trajectory)?;
        Ok(report.result)
    }

    /// Run full detection and return complete report.
    pub fn detect_full(&self, trajectory: &PriceTrajectory) -> CollusionResult<CollusionReport> {
        if trajectory.len() < self.config.min_trajectory_length {
            return Err(CollusionError::InvalidState(format!(
                "Trajectory too short: {} < {} minimum",
                trajectory.len(),
                self.config.min_trajectory_length
            )));
        }

        let pipeline = DetectionPipeline::new(self.config.clone());
        pipeline.run(trajectory)
    }

    pub fn config(&self) -> &DetectionConfig {
        &self.config
    }
}

/// Convenience function: run detection on a trajectory with default config.
pub fn run_detection(trajectory: &PriceTrajectory, config: DetectionConfig) -> CollusionResult<DetectionResult> {
    let detector = CollusionDetector::new(config);
    detector.detect(trajectory)
}

// ═══════════════════════════════════════════════════════════════════════════
// Utility functions
// ═══════════════════════════════════════════════════════════════════════════

fn compute_price_variance(trajectory: &PriceTrajectory) -> f64 {
    if trajectory.len() < 2 {
        return 0.0;
    }
    let prices: Vec<f64> = trajectory
        .outcomes
        .iter()
        .flat_map(|o| o.prices.iter().map(|p| p.0))
        .collect();
    compute_f64_variance(&prices)
}

fn compute_f64_variance(prices: &[f64]) -> f64 {
    if prices.len() < 2 {
        return 0.0;
    }
    let n = prices.len() as f64;
    let mean = prices.iter().sum::<f64>() / n;
    prices.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0)
}

/// Approximate standard normal CDF using Abramowitz & Stegun.
fn normal_cdf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.2316419 * x.abs());
    let d = 0.3989422804014327;
    let p = d * (-x * x / 2.0).exp()
        * (0.319381530 * t
            - 0.356563782 * t * t
            + 1.781477937 * t.powi(3)
            - 1.821255978 * t.powi(4)
            + 1.330274429 * t.powi(5));
    if x >= 0.0 {
        1.0 - p
    } else {
        p
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn make_trajectory(rounds: usize, price: f64, num_players: usize) -> PriceTrajectory {
        let outcomes: Vec<MarketOutcome> = (0..rounds)
            .map(|r| {
                let actions: Vec<PlayerAction> = (0..num_players)
                    .map(|p| PlayerAction::new(PlayerId(p), Price(price)))
                    .collect();
                MarketOutcome::new(
                    RoundNumber(r),
                    actions,
                    vec![Price(price); num_players],
                    vec![Quantity(1.0); num_players],
                    vec![Profit(price - 1.0); num_players],
                )
            })
            .collect();
        PriceTrajectory::new(outcomes, MarketType::Bertrand, num_players, AlgorithmType::QLearning, 0)
    }

    fn make_competitive_trajectory(rounds: usize) -> PriceTrajectory {
        make_trajectory(rounds, 3.0, 2)
    }

    fn make_collusive_trajectory(rounds: usize) -> PriceTrajectory {
        make_trajectory(rounds, 5.5, 2)
    }

    #[test]
    fn test_verdict_display() {
        assert_eq!(format!("{}", Verdict::Competitive), "Competitive");
        assert_eq!(format!("{}", Verdict::Collusive), "Collusive");
        assert_eq!(format!("{}", Verdict::Inconclusive), "Inconclusive");
    }

    #[test]
    fn test_verdict_predicates() {
        assert!(Verdict::Collusive.is_collusive());
        assert!(!Verdict::Competitive.is_collusive());
        assert!(Verdict::Competitive.is_competitive());
    }

    #[test]
    fn test_detection_result_constructors() {
        let comp = DetectionResult::competitive();
        assert_eq!(comp.verdict, Verdict::Competitive);

        let coll = DetectionResult::collusive(0.95, 0.8);
        assert_eq!(coll.verdict, Verdict::Collusive);

        let inc = DetectionResult::inconclusive();
        assert_eq!(inc.verdict, Verdict::Inconclusive);
    }

    #[test]
    fn test_normal_cdf() {
        assert!((normal_cdf(0.0) - 0.5).abs() < 0.01);
        assert!(normal_cdf(3.0) > 0.99);
        assert!(normal_cdf(-3.0) < 0.01);
    }

    #[test]
    fn test_price_variance() {
        let traj = make_trajectory(100, 5.0, 2);
        let var = compute_price_variance(&traj);
        assert!(var < 0.01); // Constant prices -> near-zero variance
    }

    #[test]
    fn test_layer0_competitive() {
        let config = DetectionConfig::default();
        let detector = Layer0Detector::new(config);
        let traj = make_competitive_trajectory(200);
        let (verdict, results, _) = detector.detect(&traj);
        assert!(results.estimated_cp < 0.1);
        assert_eq!(verdict, Verdict::Competitive);
    }

    #[test]
    fn test_layer0_collusive() {
        let config = DetectionConfig::default();
        let detector = Layer0Detector::new(config);
        let traj = make_collusive_trajectory(200);
        let (verdict, results, _) = detector.detect(&traj);
        assert!(results.estimated_cp > 0.8);
        assert_eq!(verdict, Verdict::Collusive);
    }

    #[test]
    fn test_layer0_price_stability() {
        let config = DetectionConfig::default();
        let detector = Layer0Detector::new(config);
        let traj = make_trajectory(200, 5.0, 2);
        let stability = detector.compute_price_stability(&traj);
        assert!(stability > 0.9); // Constant prices = very stable
    }

    #[test]
    fn test_detection_pipeline_competitive() {
        let config = DetectionConfig {
            min_trajectory_length: 50,
            early_termination: true,
            ..Default::default()
        };
        let pipeline = DetectionPipeline::new(config);
        let traj = make_competitive_trajectory(200);
        let report = pipeline.run(&traj).unwrap();
        assert_eq!(report.result.verdict, Verdict::Competitive);
        assert!(report.layer0_results.is_some());
    }

    #[test]
    fn test_detection_pipeline_collusive() {
        let config = DetectionConfig {
            min_trajectory_length: 50,
            early_termination: true,
            ..Default::default()
        };
        let pipeline = DetectionPipeline::new(config);
        let traj = make_collusive_trajectory(200);
        let report = pipeline.run(&traj).unwrap();
        assert_eq!(report.result.verdict, Verdict::Collusive);
    }

    #[test]
    fn test_collusion_detector_too_short() {
        let detector = CollusionDetector::with_defaults();
        let traj = make_trajectory(10, 5.0, 2); // Too short
        assert!(detector.detect(&traj).is_err());
    }

    #[test]
    fn test_collusion_detector_full_report() {
        let config = DetectionConfig {
            min_trajectory_length: 50,
            max_oracle_level: OracleAccessLevel::Layer2,
            early_termination: false,
            ..Default::default()
        };
        let detector = CollusionDetector::new(config);
        let traj = make_collusive_trajectory(200);
        let report = detector.detect_full(&traj).unwrap();
        assert!(report.layer0_results.is_some());
        assert_eq!(report.trajectory_length, 200);
        assert_eq!(report.num_players, 2);
    }

    #[test]
    fn test_run_detection_convenience() {
        let config = DetectionConfig {
            min_trajectory_length: 50,
            ..Default::default()
        };
        let traj = make_competitive_trajectory(200);
        let result = run_detection(&traj, config).unwrap();
        assert_eq!(result.verdict, Verdict::Competitive);
    }

    fn make_outcome(r: usize, prices: Vec<f64>, quantities: Vec<f64>, profits: Vec<f64>) -> MarketOutcome {
        let num = prices.len();
        let actions: Vec<PlayerAction> = (0..num)
            .map(|p| PlayerAction::new(PlayerId(p), Price(prices[p])))
            .collect();
        MarketOutcome::new(
            RoundNumber(r),
            actions,
            prices.into_iter().map(Price).collect(),
            quantities.into_iter().map(Quantity).collect(),
            profits.into_iter().map(Profit).collect(),
        )
    }

    #[test]
    fn test_layer2_punishment_detection() {
        let config = DetectionConfig::default();
        let detector = Layer2Detector::new(config);

        // Create trajectory with a punishment episode
        let mut outcomes = Vec::new();
        for r in 0..50 {
            outcomes.push(make_outcome(r, vec![5.0, 5.0], vec![1.0, 1.0], vec![4.0, 4.0]));
        }
        for r in 50..60 {
            outcomes.push(make_outcome(r, vec![1.0, 1.0], vec![3.0, 3.0], vec![0.0, 0.0]));
        }
        for r in 60..100 {
            outcomes.push(make_outcome(r, vec![5.0, 5.0], vec![1.0, 1.0], vec![4.0, 4.0]));
        }
        let traj = PriceTrajectory::new(outcomes, MarketType::Bertrand, 2, AlgorithmType::QLearning, 0);

        let mut oracle = RewindOracle::new(2);
        oracle.observe_trajectory(&traj);
        let (_, results, _) = detector.detect(&traj, &mut oracle);
        assert!(results.punishment_detected);
        assert!(results.punishment_duration >= 3);
    }

    #[test]
    fn test_detection_config_default() {
        let config = DetectionConfig::default();
        assert!((config.significance_level - 0.05).abs() < 1e-10);
        assert_eq!(config.min_trajectory_length, 100);
    }

    #[test]
    fn test_collusion_report_serialization() {
        let report = CollusionReport {
            result: DetectionResult::competitive(),
            layer0_results: None,
            layer1_results: None,
            layer2_results: None,
            duration_ms: 42,
            trajectory_length: 100,
            num_players: 2,
        };
        let json = serde_json::to_string(&report).unwrap();
        assert!(json.contains("Competitive"));
    }
}
