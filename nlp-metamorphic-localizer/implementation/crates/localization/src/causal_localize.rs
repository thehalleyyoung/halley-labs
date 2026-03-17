//! CAUSAL_LOCALIZE algorithm - the core fault localization engine.

use serde::{Deserialize, Serialize};
use shared_types::{LocalizerError, Result, SBFLMetric, StageId, TestCaseId};
use std::collections::HashMap;

/// Configuration for the causal localizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalizerConfig {
    pub max_suspect_stages: usize,
    pub significance_threshold: f64,
    pub metric_type: SBFLMetric,
    pub enable_causal_refinement: bool,
    pub enable_peeling: bool,
    pub max_peeling_rounds: usize,
    pub calibration_sample_count: usize,
    pub min_violations_for_refinement: usize,
}

impl Default for LocalizerConfig {
    fn default() -> Self {
        Self {
            max_suspect_stages: 3,
            significance_threshold: 0.05,
            metric_type: SBFLMetric::Ochiai,
            enable_causal_refinement: true,
            enable_peeling: true,
            max_peeling_rounds: 5,
            calibration_sample_count: 100,
            min_violations_for_refinement: 3,
        }
    }
}

/// Per-stage suspiciousness score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageSuspiciousness {
    pub stage_id: StageId,
    pub stage_name: String,
    pub stage_index: usize,
    pub score: f64,
    pub rank: usize,
    pub confidence_interval: (f64, f64),
    pub p_value: f64,
}

impl StageSuspiciousness {
    pub fn is_significant(&self, threshold: f64) -> bool {
        self.p_value < threshold
    }
}

/// Type of fault found at a stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FaultClassification {
    Introduction,
    Amplification,
    Both,
    None,
}

impl std::fmt::Display for FaultClassification {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Introduction => write!(f, "introduction"),
            Self::Amplification => write!(f, "amplification"),
            Self::Both => write!(f, "introduction+amplification"),
            Self::None => write!(f, "none"),
        }
    }
}

/// Result of causal refinement for a single violation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalResult {
    pub violation_id: String,
    pub stage_index: usize,
    pub stage_name: String,
    pub fault_type: FaultClassification,
    pub dce: f64,
    pub ie: f64,
    pub confidence: f64,
    pub minimal_counterexample: Option<String>,
}

/// One round of multi-fault peeling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeelingRound {
    pub round_number: usize,
    pub identified_stage: usize,
    pub stage_name: String,
    pub fault_type: FaultClassification,
    pub residual_violations: usize,
    pub residual_suspiciousness: Vec<(usize, f64)>,
}

/// Summary of the localization analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalizationSummary {
    pub primary_fault_stage: Option<String>,
    pub primary_fault_type: FaultClassification,
    pub confidence: f64,
    pub total_violations: usize,
    pub explained_violations: usize,
    pub unexplained_violations: usize,
    pub stages_examined: usize,
}

/// Complete result of the CAUSAL_LOCALIZE algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalizationResult {
    pub test_count: usize,
    pub violation_count: usize,
    pub per_stage_suspiciousness: Vec<StageSuspiciousness>,
    pub causal_results: Vec<CausalResult>,
    pub peeling_results: Vec<PeelingRound>,
    pub summary: LocalizationSummary,
    pub execution_time_ms: u64,
    pub config_used: LocalizerConfig,
}

impl LocalizationResult {
    pub fn top_suspect(&self) -> Option<&StageSuspiciousness> {
        self.per_stage_suspiciousness.iter().min_by_key(|s| s.rank)
    }

    pub fn significant_stages(&self) -> Vec<&StageSuspiciousness> {
        self.per_stage_suspiciousness
            .iter()
            .filter(|s| s.is_significant(self.config_used.significance_threshold))
            .collect()
    }

    pub fn fault_stages(&self) -> Vec<&CausalResult> {
        self.causal_results
            .iter()
            .filter(|c| c.fault_type != FaultClassification::None)
            .collect()
    }
}

/// The main causal localizer.
pub struct CausalLocalizer {
    pub config: LocalizerConfig,
    stage_names: Vec<String>,
    num_stages: usize,
    calibration_means: Vec<f64>,
    calibration_stds: Vec<f64>,
}

impl CausalLocalizer {
    pub fn new(config: LocalizerConfig, stage_names: Vec<String>) -> Self {
        let n = stage_names.len();
        Self {
            config,
            stage_names,
            num_stages: n,
            calibration_means: vec![0.0; n],
            calibration_stds: vec![1.0; n],
        }
    }

    pub fn set_calibration(&mut self, means: Vec<f64>, stds: Vec<f64>) {
        self.calibration_means = means;
        self.calibration_stds = stds;
    }

    /// Execute the full CAUSAL_LOCALIZE algorithm.
    pub fn localize(
        &self,
        differential_matrix: &[Vec<f64>],
        violation_vector: &[bool],
    ) -> Result<LocalizationResult> {
        let start = std::time::Instant::now();
        let test_count = differential_matrix.len();
        let violation_count = violation_vector.iter().filter(|&&v| v).count();

        if test_count == 0 {
            return Err(LocalizerError::ValidationError("No test cases provided".into()));
        }
        if test_count != violation_vector.len() {
            return Err(LocalizerError::ValidationError("Matrix and violation vector size mismatch".into()));
        }

        // Phase 1: Collect spectrum (already provided as differential_matrix)
        // Phase 2: Statistical localization
        let suspiciousness = self.phase2_statistical_localize(differential_matrix, violation_vector);

        // Phase 3: Causal refinement
        let causal_results = if self.config.enable_causal_refinement && violation_count >= self.config.min_violations_for_refinement {
            self.phase3_causal_refine(differential_matrix, violation_vector, &suspiciousness)
        } else {
            Vec::new()
        };

        // Phase 4: Multi-fault peeling
        let peeling_results = if self.config.enable_peeling {
            self.phase4_multi_fault_peel(differential_matrix, violation_vector, &suspiciousness)
        } else {
            Vec::new()
        };

        // Build summary
        let primary = suspiciousness.iter().min_by_key(|s| s.rank);
        let primary_fault_type = causal_results
            .first()
            .map(|c| c.fault_type)
            .unwrap_or(FaultClassification::None);

        let explained = causal_results
            .iter()
            .filter(|c| c.fault_type != FaultClassification::None)
            .count();

        let summary = LocalizationSummary {
            primary_fault_stage: primary.map(|p| p.stage_name.clone()),
            primary_fault_type,
            confidence: primary.map(|p| p.score).unwrap_or(0.0),
            total_violations: violation_count,
            explained_violations: explained,
            unexplained_violations: violation_count.saturating_sub(explained),
            stages_examined: suspiciousness.len(),
        };

        Ok(LocalizationResult {
            test_count,
            violation_count,
            per_stage_suspiciousness: suspiciousness,
            causal_results,
            peeling_results,
            summary,
            execution_time_ms: start.elapsed().as_millis() as u64,
            config_used: self.config.clone(),
        })
    }

    /// Phase 2: Compute suspiciousness scores using adapted SBFL.
    fn phase2_statistical_localize(
        &self,
        matrix: &[Vec<f64>],
        violations: &[bool],
    ) -> Vec<StageSuspiciousness> {
        let n_tests = matrix.len();
        let n_stages = self.num_stages;

        let mut scores: Vec<f64> = vec![0.0; n_stages];
        let violation_count = violations.iter().filter(|&&v| v).count() as f64;

        if violation_count == 0.0 {
            return self.stage_names.iter().enumerate().map(|(i, name)| {
                StageSuspiciousness {
                    stage_id: StageId::new(),
                    stage_name: name.clone(),
                    stage_index: i,
                    score: 0.0,
                    rank: i + 1,
                    confidence_interval: (0.0, 0.0),
                    p_value: 1.0,
                }
            }).collect();
        }

        for k in 0..n_stages {
            let mut sum_fail = 0.0;
            let mut sum_total = 0.0;

            for i in 0..n_tests {
                let d = if k < matrix[i].len() { matrix[i][k] } else { 0.0 };
                let d_normalized = (d - self.calibration_means.get(k).copied().unwrap_or(0.0))
                    / self.calibration_stds.get(k).copied().unwrap_or(1.0).max(0.001);

                sum_total += d_normalized.abs();
                if violations[i] {
                    sum_fail += d_normalized.abs();
                }
            }

            // Ochiai-like formula adapted for continuous differentials
            let denominator = (sum_total * violation_count).sqrt();
            scores[k] = if denominator > 0.0 {
                sum_fail / denominator
            } else {
                0.0
            };
        }

        // Rank by score descending
        let mut indexed: Vec<(usize, f64)> = scores.iter().enumerate().map(|(i, &s)| (i, s)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        indexed.iter().enumerate().map(|(rank, &(idx, score))| {
            let ci_half = 1.96 * (score * (1.0 - score) / n_tests.max(1) as f64).sqrt();
            StageSuspiciousness {
                stage_id: StageId::new(),
                stage_name: self.stage_names.get(idx).cloned().unwrap_or_default(),
                stage_index: idx,
                score,
                rank: rank + 1,
                confidence_interval: ((score - ci_half).max(0.0), (score + ci_half).min(1.0)),
                p_value: self.compute_p_value(score, n_tests, violation_count as usize),
            }
        }).collect()
    }

    /// Compute a p-value for the suspiciousness score using a permutation test approximation.
    fn compute_p_value(&self, score: f64, n_tests: usize, n_violations: usize) -> f64 {
        if n_tests == 0 || n_violations == 0 { return 1.0; }
        let expected = (n_violations as f64 / n_tests as f64).sqrt();
        let z = (score - expected) / (expected * (1.0 - expected / n_tests as f64.sqrt())).max(0.001);
        // Normal CDF approximation for one-sided test
        let p = 0.5 * (1.0 - erf_approx(z / std::f64::consts::SQRT_2));
        p.max(0.0).min(1.0)
    }

    /// Phase 3: Causal refinement via interventional replay.
    fn phase3_causal_refine(
        &self,
        matrix: &[Vec<f64>],
        violations: &[bool],
        suspiciousness: &[StageSuspiciousness],
    ) -> Vec<CausalResult> {
        let mut results = Vec::new();
        let top_k = suspiciousness.iter().take(self.config.max_suspect_stages);

        for suspect in top_k {
            let k = suspect.stage_index;

            // Compute DCE: simulate intervention by replacing stage k's differential with 0
            let mut dce_violations = 0;
            let mut total_violations = 0;

            for (i, &is_violation) in violations.iter().enumerate() {
                if !is_violation { continue; }
                total_violations += 1;

                // Simulate: what happens if stage k had zero differential?
                let original_delta = matrix[i].get(k).copied().unwrap_or(0.0);
                let downstream_effect: f64 = matrix[i].iter().skip(k + 1)
                    .map(|&d| d)
                    .sum::<f64>() / matrix[i].len().max(1) as f64;

                // DCE: How much of the final violation is due to stage k directly?
                let total_delta: f64 = matrix[i].iter().sum::<f64>() / matrix[i].len().max(1) as f64;
                let dce_estimate = if total_delta > 0.0 {
                    original_delta / total_delta
                } else {
                    0.0
                };

                if dce_estimate > 0.1 {
                    dce_violations += 1;
                }
            }

            let dce = if total_violations > 0 {
                dce_violations as f64 / total_violations as f64
            } else { 0.0 };

            // IE = total effect at stage k - DCE
            let mean_delta: f64 = violations.iter().enumerate()
                .filter(|(_, &v)| v)
                .map(|(i, _)| matrix[i].get(k).copied().unwrap_or(0.0))
                .sum::<f64>() / total_violations.max(1) as f64;

            let ie = (mean_delta - dce).max(0.0);

            let fault_type = classify_fault(dce, ie, self.config.significance_threshold);

            results.push(CausalResult {
                violation_id: format!("v-{}", k),
                stage_index: k,
                stage_name: suspect.stage_name.clone(),
                fault_type,
                dce,
                ie,
                confidence: suspect.score,
                minimal_counterexample: None,
            });
        }

        results
    }

    /// Phase 4: Multi-fault peeling.
    fn phase4_multi_fault_peel(
        &self,
        matrix: &[Vec<f64>],
        violations: &[bool],
        initial_suspiciousness: &[StageSuspiciousness],
    ) -> Vec<PeelingRound> {
        let mut rounds = Vec::new();
        let mut current_matrix: Vec<Vec<f64>> = matrix.to_vec();
        let mut remaining_violations: Vec<bool> = violations.to_vec();
        let mut identified_stages: Vec<usize> = Vec::new();

        for round in 0..self.config.max_peeling_rounds {
            let violation_count = remaining_violations.iter().filter(|&&v| v).count();
            if violation_count < self.config.min_violations_for_refinement {
                break;
            }

            let susp = self.phase2_statistical_localize(&current_matrix, &remaining_violations);
            if susp.is_empty() { break; }

            let top = &susp[0];
            if top.score < self.config.significance_threshold {
                break;
            }

            let k = top.stage_index;
            if identified_stages.contains(&k) { break; }

            // Determine fault type
            let fault_type = if round == 0 {
                initial_suspiciousness.iter()
                    .find(|s| s.stage_index == k)
                    .map(|_| FaultClassification::Introduction)
                    .unwrap_or(FaultClassification::None)
            } else {
                FaultClassification::Amplification
            };

            // Peel: zero out stage k's differentials and re-check violations
            for row in &mut current_matrix {
                if k < row.len() {
                    row[k] = 0.0;
                }
            }

            // Update violation vector (re-check if violations still hold)
            let mut residual_count = 0;
            for (i, v) in remaining_violations.iter_mut().enumerate() {
                if *v {
                    let remaining_delta: f64 = current_matrix[i].iter().sum();
                    if remaining_delta < self.config.significance_threshold {
                        *v = false;
                    } else {
                        residual_count += 1;
                    }
                }
            }

            let residual_susp: Vec<(usize, f64)> = susp.iter()
                .skip(1)
                .take(3)
                .map(|s| (s.stage_index, s.score))
                .collect();

            rounds.push(PeelingRound {
                round_number: round,
                identified_stage: k,
                stage_name: top.stage_name.clone(),
                fault_type,
                residual_violations: residual_count,
                residual_suspiciousness: residual_susp,
            });

            identified_stages.push(k);
        }

        rounds
    }
}

fn classify_fault(dce: f64, ie: f64, threshold: f64) -> FaultClassification {
    let dce_sig = dce > threshold;
    let ie_sig = ie > threshold;
    match (dce_sig, ie_sig) {
        (true, true) => FaultClassification::Both,
        (true, false) => FaultClassification::Introduction,
        (false, true) => FaultClassification::Amplification,
        (false, false) => FaultClassification::None,
    }
}

/// Approximate error function for p-value computation.
fn erf_approx(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_localizer() -> CausalLocalizer {
        CausalLocalizer::new(
            LocalizerConfig::default(),
            vec!["tokenizer".into(), "tagger".into(), "parser".into(), "ner".into()],
        )
    }

    #[test]
    fn test_localize_basic() {
        let localizer = make_localizer();
        let matrix = vec![
            vec![0.1, 0.2, 0.8, 0.3],
            vec![0.0, 0.1, 0.7, 0.2],
            vec![0.1, 0.3, 0.9, 0.4],
            vec![0.0, 0.0, 0.1, 0.0],
            vec![0.0, 0.1, 0.0, 0.1],
        ];
        let violations = vec![true, true, true, false, false];
        let result = localizer.localize(&matrix, &violations).unwrap();
        assert_eq!(result.test_count, 5);
        assert_eq!(result.violation_count, 3);
        assert!(!result.per_stage_suspiciousness.is_empty());
    }

    #[test]
    fn test_localize_no_violations() {
        let localizer = make_localizer();
        let matrix = vec![vec![0.1, 0.2, 0.3, 0.4]];
        let violations = vec![false];
        let result = localizer.localize(&matrix, &violations).unwrap();
        assert_eq!(result.violation_count, 0);
    }

    #[test]
    fn test_localize_all_violations() {
        let localizer = make_localizer();
        let matrix = vec![
            vec![0.1, 0.8, 0.2, 0.1],
            vec![0.0, 0.9, 0.1, 0.0],
            vec![0.1, 0.7, 0.3, 0.2],
        ];
        let violations = vec![true, true, true];
        let result = localizer.localize(&matrix, &violations).unwrap();
        assert_eq!(result.violation_count, 3);
        let top = result.top_suspect().unwrap();
        assert_eq!(top.rank, 1);
    }

    #[test]
    fn test_classify_fault() {
        assert_eq!(classify_fault(0.5, 0.0, 0.05), FaultClassification::Introduction);
        assert_eq!(classify_fault(0.0, 0.5, 0.05), FaultClassification::Amplification);
        assert_eq!(classify_fault(0.5, 0.5, 0.05), FaultClassification::Both);
        assert_eq!(classify_fault(0.01, 0.01, 0.05), FaultClassification::None);
    }

    #[test]
    fn test_fault_classification_display() {
        assert_eq!(format!("{}", FaultClassification::Introduction), "introduction");
        assert_eq!(format!("{}", FaultClassification::Both), "introduction+amplification");
    }

    #[test]
    fn test_localization_result_methods() {
        let localizer = make_localizer();
        let matrix = vec![
            vec![0.1, 0.8, 0.2, 0.1],
            vec![0.0, 0.7, 0.1, 0.0],
        ];
        let violations = vec![true, true];
        let result = localizer.localize(&matrix, &violations).unwrap();
        assert!(result.top_suspect().is_some());
    }

    #[test]
    fn test_erf_approx() {
        assert!((erf_approx(0.0)).abs() < 0.001);
        assert!((erf_approx(3.0) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_config_default() {
        let config = LocalizerConfig::default();
        assert_eq!(config.max_suspect_stages, 3);
        assert!(config.enable_causal_refinement);
    }

    #[test]
    fn test_empty_matrix_error() {
        let localizer = make_localizer();
        let result = localizer.localize(&[], &[]);
        assert!(result.is_err());
    }
}
