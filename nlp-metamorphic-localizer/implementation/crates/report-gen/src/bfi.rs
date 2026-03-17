//! Behavioral Fragility Index (M7) computation.
//!
//! BFI_k = E[d_k(output)] / E[d_{k-1}(input)] measures how each stage
//! amplifies or absorbs divergence introduced by metamorphic transformations.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Core types ──────────────────────────────────────────────────────────────

/// Interpretation of a BFI value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BFIInterpretation {
    /// BFI >> 1: stage amplifies divergence.
    Amplifying,
    /// BFI ≈ 1: stage propagates divergence unchanged.
    Propagating,
    /// BFI << 1: stage absorbs / dampens divergence.
    Absorbing,
    /// Insufficient data to compute BFI.
    Undefined,
}

impl std::fmt::Display for BFIInterpretation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Amplifying => write!(f, "Amplifying"),
            Self::Propagating => write!(f, "Propagating"),
            Self::Absorbing => write!(f, "Absorbing"),
            Self::Undefined => write!(f, "Undefined"),
        }
    }
}

/// BFI result for a single stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BFIResult {
    pub stage_name: String,
    pub bfi_value: f64,
    pub interpretation: BFIInterpretation,
    pub confidence_interval: (f64, f64),
    pub sample_count: usize,
}

/// Per-transformation BFI profile for a stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BFIProfile {
    pub stage_name: String,
    pub per_transformation: HashMap<String, f64>,
    pub aggregate_bfi: f64,
}

/// Trend of BFI across pipeline stages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BFITrend {
    Increasing,
    Decreasing,
    Stable,
    Oscillating,
}

impl std::fmt::Display for BFITrend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Increasing => write!(f, "Increasing"),
            Self::Decreasing => write!(f, "Decreasing"),
            Self::Stable => write!(f, "Stable"),
            Self::Oscillating => write!(f, "Oscillating"),
        }
    }
}

// ── BFIComputer ─────────────────────────────────────────────────────────────

/// Computes the Behavioral Fragility Index for pipeline stages.
#[derive(Debug, Clone)]
pub struct BFIComputer {
    /// Small constant added to denominator to avoid division by zero.
    pub epsilon: f64,
    /// Optional calibration baselines per stage (mean, std_dev).
    pub calibration_data: HashMap<String, (f64, f64)>,
}

impl BFIComputer {
    pub fn new(epsilon: f64) -> Self {
        Self {
            epsilon,
            calibration_data: HashMap::new(),
        }
    }

    pub fn with_calibration(mut self, data: HashMap<String, (f64, f64)>) -> Self {
        self.calibration_data = data;
        self
    }

    /// Compute BFI for a single stage.
    ///
    /// `output_diffs` – d_k(output) values for this stage across test cases.
    /// `input_diffs`  – d_{k-1}(input) values (previous stage's output diffs).
    pub fn compute_bfi_for_stage(
        &self,
        stage_name: &str,
        output_diffs: &[f64],
        input_diffs: &[f64],
    ) -> BFIResult {
        let n = output_diffs.len().min(input_diffs.len());
        if n == 0 {
            return BFIResult {
                stage_name: stage_name.to_string(),
                bfi_value: f64::NAN,
                interpretation: BFIInterpretation::Undefined,
                confidence_interval: (f64::NAN, f64::NAN),
                sample_count: 0,
            };
        }

        let mean_out = output_diffs.iter().take(n).sum::<f64>() / n as f64;
        let mean_in = input_diffs.iter().take(n).sum::<f64>() / n as f64;
        let denominator = mean_in.abs().max(self.epsilon);
        let bfi = mean_out / denominator;

        let interpretation = interpret_bfi(bfi);

        // Bootstrap-style CI approximation: BFI ± 1.96 * SE(BFI)
        let ratios: Vec<f64> = output_diffs
            .iter()
            .take(n)
            .zip(input_diffs.iter().take(n))
            .map(|(&o, &i)| o / i.abs().max(self.epsilon))
            .collect();
        let ratio_mean = ratios.iter().sum::<f64>() / n as f64;
        let ratio_var = if n > 1 {
            ratios.iter().map(|r| (r - ratio_mean).powi(2)).sum::<f64>() / (n - 1) as f64
        } else {
            0.0
        };
        let se = (ratio_var / n as f64).sqrt();
        let ci = (bfi - 1.96 * se, bfi + 1.96 * se);

        BFIResult {
            stage_name: stage_name.to_string(),
            bfi_value: bfi,
            interpretation,
            confidence_interval: ci,
            sample_count: n,
        }
    }

    /// Compute BFI for all stages given per-stage differential vectors.
    ///
    /// `stage_diffs` is ordered: stage_diffs[k] = differentials at stage k.
    pub fn compute_all_bfi(
        &self,
        stage_names: &[String],
        stage_diffs: &[Vec<f64>],
    ) -> Vec<BFIResult> {
        let mut results = Vec::with_capacity(stage_names.len());
        for (k, name) in stage_names.iter().enumerate() {
            if k == 0 {
                // First stage: input is zero (original), so BFI uses epsilon floor.
                let zeros = vec![0.0; stage_diffs[k].len()];
                results.push(self.compute_bfi_for_stage(name, &stage_diffs[k], &zeros));
            } else {
                results.push(self.compute_bfi_for_stage(
                    name,
                    &stage_diffs[k],
                    &stage_diffs[k - 1],
                ));
            }
        }
        results
    }

    /// Find stages that amplify divergence (BFI >> 1).
    pub fn detect_amplifiers<'a>(&self, results: &'a [BFIResult], threshold: f64) -> Vec<&'a BFIResult> {
        results
            .iter()
            .filter(|r| r.bfi_value > threshold && r.interpretation == BFIInterpretation::Amplifying)
            .collect()
    }

    /// Find stages that absorb divergence (BFI << 1).
    pub fn detect_absorbers<'a>(&self, results: &'a [BFIResult], threshold: f64) -> Vec<&'a BFIResult> {
        results
            .iter()
            .filter(|r| r.bfi_value < threshold && r.interpretation == BFIInterpretation::Absorbing)
            .collect()
    }

    /// Compute per-transformation BFI profiles for each stage.
    pub fn compute_profiles(
        &self,
        stage_names: &[String],
        per_transform_diffs: &HashMap<String, Vec<Vec<f64>>>,
    ) -> Vec<BFIProfile> {
        stage_names
            .iter()
            .enumerate()
            .map(|(k, name)| {
                let mut per_transformation = HashMap::new();
                for (transform_name, diffs) in per_transform_diffs {
                    if k < diffs.len() && k > 0 && k - 1 < diffs.len() {
                        let mean_out = if diffs[k].is_empty() {
                            0.0
                        } else {
                            diffs[k].iter().sum::<f64>() / diffs[k].len() as f64
                        };
                        let mean_in = if diffs[k - 1].is_empty() {
                            0.0
                        } else {
                            diffs[k - 1].iter().sum::<f64>() / diffs[k - 1].len() as f64
                        };
                        let bfi = mean_out / mean_in.abs().max(self.epsilon);
                        per_transformation.insert(transform_name.clone(), bfi);
                    }
                }
                let aggregate = if per_transformation.is_empty() {
                    f64::NAN
                } else {
                    per_transformation.values().sum::<f64>() / per_transformation.len() as f64
                };
                BFIProfile {
                    stage_name: name.clone(),
                    per_transformation,
                    aggregate_bfi: aggregate,
                }
            })
            .collect()
    }

    /// Analyze BFI trend across pipeline stages.
    pub fn trend_analysis(&self, results: &[BFIResult]) -> BFITrend {
        let valid: Vec<f64> = results
            .iter()
            .filter(|r| !r.bfi_value.is_nan())
            .map(|r| r.bfi_value)
            .collect();
        if valid.len() < 2 {
            return BFITrend::Stable;
        }

        let mut increases = 0usize;
        let mut decreases = 0usize;
        let mut direction_changes = 0usize;
        let mut last_dir: Option<bool> = None;

        for w in valid.windows(2) {
            let diff = w[1] - w[0];
            let going_up = diff > 0.0;
            if diff.abs() > 0.05 {
                if going_up {
                    increases += 1;
                } else {
                    decreases += 1;
                }
                if let Some(was_up) = last_dir {
                    if was_up != going_up {
                        direction_changes += 1;
                    }
                }
                last_dir = Some(going_up);
            }
        }

        let total = increases + decreases;
        if total == 0 {
            BFITrend::Stable
        } else if direction_changes as f64 / total as f64 > 0.4 {
            BFITrend::Oscillating
        } else if increases > decreases * 2 {
            BFITrend::Increasing
        } else if decreases > increases * 2 {
            BFITrend::Decreasing
        } else {
            BFITrend::Stable
        }
    }
}

impl Default for BFIComputer {
    fn default() -> Self {
        Self::new(1e-6)
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn interpret_bfi(bfi: f64) -> BFIInterpretation {
    if bfi.is_nan() || bfi.is_infinite() {
        BFIInterpretation::Undefined
    } else if bfi > 1.2 {
        BFIInterpretation::Amplifying
    } else if bfi < 0.8 {
        BFIInterpretation::Absorbing
    } else {
        BFIInterpretation::Propagating
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_computer() -> BFIComputer {
        BFIComputer::new(1e-6)
    }

    #[test]
    fn test_bfi_amplifying() {
        let c = make_computer();
        let result = c.compute_bfi_for_stage("tok", &[2.0, 3.0, 2.5], &[0.5, 0.5, 0.5]);
        assert!(result.bfi_value > 1.2);
        assert_eq!(result.interpretation, BFIInterpretation::Amplifying);
        assert_eq!(result.sample_count, 3);
    }

    #[test]
    fn test_bfi_absorbing() {
        let c = make_computer();
        let result = c.compute_bfi_for_stage("ner", &[0.1, 0.15, 0.12], &[1.0, 1.2, 1.1]);
        assert!(result.bfi_value < 0.8);
        assert_eq!(result.interpretation, BFIInterpretation::Absorbing);
    }

    #[test]
    fn test_bfi_propagating() {
        let c = make_computer();
        let result = c.compute_bfi_for_stage("pos", &[1.0, 1.0, 1.0], &[1.0, 1.0, 1.0]);
        assert!((result.bfi_value - 1.0).abs() < 0.3);
        assert_eq!(result.interpretation, BFIInterpretation::Propagating);
    }

    #[test]
    fn test_bfi_undefined_empty() {
        let c = make_computer();
        let result = c.compute_bfi_for_stage("empty", &[], &[]);
        assert!(result.bfi_value.is_nan());
        assert_eq!(result.interpretation, BFIInterpretation::Undefined);
    }

    #[test]
    fn test_compute_all_bfi() {
        let c = make_computer();
        let names = vec!["s0".to_string(), "s1".to_string(), "s2".to_string()];
        let diffs = vec![
            vec![0.1, 0.2, 0.15],
            vec![0.5, 0.6, 0.55],
            vec![0.3, 0.35, 0.32],
        ];
        let results = c.compute_all_bfi(&names, &diffs);
        assert_eq!(results.len(), 3);
        assert_eq!(results[1].interpretation, BFIInterpretation::Amplifying);
        assert_eq!(results[2].interpretation, BFIInterpretation::Absorbing);
    }

    #[test]
    fn test_detect_amplifiers() {
        let c = make_computer();
        let results = vec![
            BFIResult { stage_name: "a".into(), bfi_value: 3.0, interpretation: BFIInterpretation::Amplifying, confidence_interval: (2.0, 4.0), sample_count: 10 },
            BFIResult { stage_name: "b".into(), bfi_value: 1.0, interpretation: BFIInterpretation::Propagating, confidence_interval: (0.8, 1.2), sample_count: 10 },
            BFIResult { stage_name: "c".into(), bfi_value: 0.3, interpretation: BFIInterpretation::Absorbing, confidence_interval: (0.1, 0.5), sample_count: 10 },
        ];
        let amps = c.detect_amplifiers(&results, 1.5);
        assert_eq!(amps.len(), 1);
        assert_eq!(amps[0].stage_name, "a");
    }

    #[test]
    fn test_detect_absorbers() {
        let c = make_computer();
        let results = vec![
            BFIResult { stage_name: "a".into(), bfi_value: 3.0, interpretation: BFIInterpretation::Amplifying, confidence_interval: (2.0, 4.0), sample_count: 10 },
            BFIResult { stage_name: "c".into(), bfi_value: 0.3, interpretation: BFIInterpretation::Absorbing, confidence_interval: (0.1, 0.5), sample_count: 10 },
        ];
        let abs = c.detect_absorbers(&results, 0.5);
        assert_eq!(abs.len(), 1);
        assert_eq!(abs[0].stage_name, "c");
    }

    #[test]
    fn test_trend_increasing() {
        let c = make_computer();
        let results: Vec<BFIResult> = (0..5)
            .map(|i| BFIResult {
                stage_name: format!("s{i}"),
                bfi_value: 1.0 + i as f64 * 0.5,
                interpretation: BFIInterpretation::Amplifying,
                confidence_interval: (0.0, 5.0),
                sample_count: 10,
            })
            .collect();
        assert_eq!(c.trend_analysis(&results), BFITrend::Increasing);
    }

    #[test]
    fn test_trend_stable() {
        let c = make_computer();
        let results: Vec<BFIResult> = (0..5)
            .map(|i| BFIResult {
                stage_name: format!("s{i}"),
                bfi_value: 1.0,
                interpretation: BFIInterpretation::Propagating,
                confidence_interval: (0.9, 1.1),
                sample_count: 10,
            })
            .collect();
        assert_eq!(c.trend_analysis(&results), BFITrend::Stable);
    }

    #[test]
    fn test_profiles() {
        let c = make_computer();
        let names = vec!["s0".into(), "s1".into()];
        let mut per_transform = HashMap::new();
        per_transform.insert("passive".to_string(), vec![vec![0.1, 0.2], vec![0.5, 0.6]]);
        let profiles = c.compute_profiles(&names, &per_transform);
        assert_eq!(profiles.len(), 2);
        assert!(profiles[1].per_transformation.contains_key("passive"));
    }

    #[test]
    fn test_epsilon_regularization() {
        let c = BFIComputer::new(0.01);
        let result = c.compute_bfi_for_stage("test", &[1.0], &[0.0]);
        assert!(result.bfi_value.is_finite());
        assert!(result.bfi_value > 0.0);
    }
}
