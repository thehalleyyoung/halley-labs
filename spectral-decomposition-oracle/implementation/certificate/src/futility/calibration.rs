//! Threshold calibration for futility predictor.
//!
//! Provides methods for calibrating the futility prediction threshold
//! using training data, including cross-validation, isotonic regression,
//! reliability diagrams, Brier score computation, and temperature scaling.

use crate::error::{CertificateError, CertificateResult};
use serde::{Deserialize, Serialize};

/// A single training instance: predicted probability and true outcome.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationSample {
    pub predicted_probability: f64,
    pub true_label: bool,
    pub instance_id: Option<String>,
}

/// Result of threshold calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationResult {
    pub optimal_threshold: f64,
    pub brier_score: f64,
    pub log_loss: f64,
    pub accuracy_at_threshold: f64,
    pub precision_at_threshold: f64,
    pub recall_at_threshold: f64,
    pub f1_at_threshold: f64,
    pub num_samples: usize,
    pub num_positive: usize,
    pub calibration_curve: Vec<(f64, f64)>,
    pub reliability_score: f64,
    pub auc_roc: f64,
}

impl CalibrationResult {
    /// Calibrate threshold from training data.
    ///
    /// Searches for the threshold that maximizes the F1 score.
    pub fn calibrate_threshold(samples: &[CalibrationSample]) -> CertificateResult<Self> {
        if samples.is_empty() {
            return Err(CertificateError::incomplete_data(
                "calibration samples",
                "no training data provided",
            ));
        }
        if samples.len() < 10 {
            return Err(CertificateError::incomplete_data(
                "calibration samples",
                format!("need at least 10 samples, got {}", samples.len()),
            ));
        }

        let num_positive = samples.iter().filter(|s| s.true_label).count();
        if num_positive == 0 || num_positive == samples.len() {
            return Err(CertificateError::incomplete_data(
                "calibration samples",
                "need both positive and negative samples",
            ));
        }

        // Compute Brier score
        let brier_score = Self::compute_brier_score(samples);
        let log_loss = Self::compute_log_loss(samples);

        // Search for optimal threshold
        let thresholds: Vec<f64> = (1..100).map(|i| i as f64 / 100.0).collect();
        let mut best_f1 = 0.0;
        let mut best_threshold = 0.5;
        let mut best_metrics = (0.0, 0.0, 0.0, 0.0); // acc, prec, rec, f1

        for &t in &thresholds {
            let (acc, prec, rec, f1) = Self::metrics_at_threshold(samples, t);
            if f1 > best_f1 {
                best_f1 = f1;
                best_threshold = t;
                best_metrics = (acc, prec, rec, f1);
            }
        }

        // Compute calibration curve (reliability diagram data)
        let calibration_curve = Self::compute_calibration_curve(samples, 10);
        let reliability_score = Self::compute_reliability_score(&calibration_curve);

        // Compute AUC-ROC
        let auc_roc = Self::compute_auc_roc(samples);

        Ok(CalibrationResult {
            optimal_threshold: best_threshold,
            brier_score,
            log_loss,
            accuracy_at_threshold: best_metrics.0,
            precision_at_threshold: best_metrics.1,
            recall_at_threshold: best_metrics.2,
            f1_at_threshold: best_metrics.3,
            num_samples: samples.len(),
            num_positive,
            calibration_curve,
            reliability_score,
            auc_roc,
        })
    }

    fn compute_brier_score(samples: &[CalibrationSample]) -> f64 {
        let n = samples.len() as f64;
        samples
            .iter()
            .map(|s| {
                let y = if s.true_label { 1.0 } else { 0.0 };
                (s.predicted_probability - y).powi(2)
            })
            .sum::<f64>()
            / n
    }

    fn compute_log_loss(samples: &[CalibrationSample]) -> f64 {
        let eps = 1e-15;
        let n = samples.len() as f64;
        -samples
            .iter()
            .map(|s| {
                let p = s.predicted_probability.clamp(eps, 1.0 - eps);
                if s.true_label {
                    p.ln()
                } else {
                    (1.0 - p).ln()
                }
            })
            .sum::<f64>()
            / n
    }

    fn metrics_at_threshold(samples: &[CalibrationSample], threshold: f64) -> (f64, f64, f64, f64) {
        let mut tp = 0usize;
        let mut fp = 0usize;
        let mut tn = 0usize;
        let mut fn_ = 0usize;

        for s in samples {
            let predicted_positive = s.predicted_probability >= threshold;
            match (predicted_positive, s.true_label) {
                (true, true) => tp += 1,
                (true, false) => fp += 1,
                (false, true) => fn_ += 1,
                (false, false) => tn += 1,
            }
        }

        let accuracy = (tp + tn) as f64 / samples.len() as f64;
        let precision = if tp + fp > 0 {
            tp as f64 / (tp + fp) as f64
        } else {
            0.0
        };
        let recall = if tp + fn_ > 0 {
            tp as f64 / (tp + fn_) as f64
        } else {
            0.0
        };
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        (accuracy, precision, recall, f1)
    }

    fn compute_calibration_curve(
        samples: &[CalibrationSample],
        num_bins: usize,
    ) -> Vec<(f64, f64)> {
        let mut bins: Vec<(f64, f64, usize)> = vec![(0.0, 0.0, 0); num_bins];

        for s in samples {
            let bin_idx = ((s.predicted_probability * num_bins as f64) as usize).min(num_bins - 1);
            bins[bin_idx].0 += s.predicted_probability;
            bins[bin_idx].1 += if s.true_label { 1.0 } else { 0.0 };
            bins[bin_idx].2 += 1;
        }

        bins.iter()
            .filter(|(_, _, count)| *count > 0)
            .map(|(pred_sum, true_sum, count)| {
                let mean_predicted = pred_sum / *count as f64;
                let fraction_positive = true_sum / *count as f64;
                (mean_predicted, fraction_positive)
            })
            .collect()
    }

    fn compute_reliability_score(curve: &[(f64, f64)]) -> f64 {
        if curve.is_empty() {
            return 0.0;
        }
        let total_error: f64 = curve.iter().map(|(pred, obs)| (pred - obs).abs()).sum();
        let avg_error = total_error / curve.len() as f64;
        (1.0 - avg_error).max(0.0)
    }

    fn compute_auc_roc(samples: &[CalibrationSample]) -> f64 {
        let mut sorted: Vec<&CalibrationSample> = samples.iter().collect();
        sorted.sort_by(|a, b| {
            b.predicted_probability
                .partial_cmp(&a.predicted_probability)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let total_pos = samples.iter().filter(|s| s.true_label).count() as f64;
        let total_neg = samples.iter().filter(|s| !s.true_label).count() as f64;

        if total_pos < 1.0 || total_neg < 1.0 {
            return 0.5;
        }

        let mut auc = 0.0;
        let mut tpr_prev = 0.0;
        let mut fpr_prev = 0.0;
        let mut tp = 0.0;
        let mut fp = 0.0;

        for s in &sorted {
            if s.true_label {
                tp += 1.0;
            } else {
                fp += 1.0;
            }
            let tpr = tp / total_pos;
            let fpr = fp / total_neg;
            auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2.0;
            tpr_prev = tpr;
            fpr_prev = fpr;
        }

        auc
    }

    /// Cross-validate the threshold using k-fold.
    pub fn cross_validate_threshold(
        samples: &[CalibrationSample],
        k_folds: usize,
    ) -> CertificateResult<CrossValidationResult> {
        if samples.len() < k_folds * 5 {
            return Err(CertificateError::incomplete_data(
                "calibration samples",
                format!(
                    "need at least {} samples for {}-fold CV",
                    k_folds * 5,
                    k_folds
                ),
            ));
        }

        let fold_size = samples.len() / k_folds;
        let mut fold_thresholds = Vec::with_capacity(k_folds);
        let mut fold_brier_scores = Vec::with_capacity(k_folds);
        let mut fold_f1_scores = Vec::with_capacity(k_folds);

        for fold in 0..k_folds {
            let test_start = fold * fold_size;
            let test_end = if fold == k_folds - 1 {
                samples.len()
            } else {
                (fold + 1) * fold_size
            };

            let train: Vec<CalibrationSample> = samples
                .iter()
                .enumerate()
                .filter(|(i, _)| *i < test_start || *i >= test_end)
                .map(|(_, s)| s.clone())
                .collect();

            if let Ok(result) = Self::calibrate_threshold(&train) {
                fold_thresholds.push(result.optimal_threshold);
                fold_brier_scores.push(result.brier_score);
                fold_f1_scores.push(result.f1_at_threshold);
            }
        }

        if fold_thresholds.is_empty() {
            return Err(CertificateError::validation_failed(
                "all CV folds failed",
                0,
                k_folds,
                vec!["insufficient data in folds".to_string()],
            ));
        }

        let mean_threshold =
            fold_thresholds.iter().sum::<f64>() / fold_thresholds.len() as f64;
        let std_threshold = {
            let var = fold_thresholds
                .iter()
                .map(|t| (t - mean_threshold).powi(2))
                .sum::<f64>()
                / fold_thresholds.len() as f64;
            var.sqrt()
        };
        let mean_brier =
            fold_brier_scores.iter().sum::<f64>() / fold_brier_scores.len() as f64;
        let mean_f1 = fold_f1_scores.iter().sum::<f64>() / fold_f1_scores.len() as f64;

        Ok(CrossValidationResult {
            k_folds,
            mean_threshold,
            std_threshold,
            fold_thresholds,
            mean_brier_score: mean_brier,
            fold_brier_scores,
            mean_f1_score: mean_f1,
            fold_f1_scores,
            threshold_stable: std_threshold < 0.1,
        })
    }
}

/// Cross-validation result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossValidationResult {
    pub k_folds: usize,
    pub mean_threshold: f64,
    pub std_threshold: f64,
    pub fold_thresholds: Vec<f64>,
    pub mean_brier_score: f64,
    pub fold_brier_scores: Vec<f64>,
    pub mean_f1_score: f64,
    pub fold_f1_scores: Vec<f64>,
    pub threshold_stable: bool,
}

/// Temperature scaling for probability calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemperatureScaling {
    pub temperature: f64,
    pub original_brier: f64,
    pub calibrated_brier: f64,
    pub improvement: f64,
}

impl TemperatureScaling {
    /// Find the optimal temperature by minimizing Brier score.
    pub fn optimize(samples: &[CalibrationSample]) -> CertificateResult<Self> {
        if samples.is_empty() {
            return Err(CertificateError::incomplete_data(
                "samples",
                "no samples for temperature scaling",
            ));
        }

        let original_brier = CalibrationResult::compute_brier_score(samples);

        // Grid search over temperature values
        let mut best_temp = 1.0;
        let mut best_brier = original_brier;

        for t_idx in 1..200 {
            let temp = t_idx as f64 * 0.05;
            let scaled_samples: Vec<CalibrationSample> = samples
                .iter()
                .map(|s| {
                    let logit = (s.predicted_probability / (1.0 - s.predicted_probability + 1e-15)).ln();
                    let scaled_logit = logit / temp;
                    let scaled_prob = 1.0 / (1.0 + (-scaled_logit).exp());
                    CalibrationSample {
                        predicted_probability: scaled_prob,
                        true_label: s.true_label,
                        instance_id: s.instance_id.clone(),
                    }
                })
                .collect();

            let brier = CalibrationResult::compute_brier_score(&scaled_samples);
            if brier < best_brier {
                best_brier = brier;
                best_temp = temp;
            }
        }

        Ok(Self {
            temperature: best_temp,
            original_brier,
            calibrated_brier: best_brier,
            improvement: original_brier - best_brier,
        })
    }

    /// Apply temperature scaling to a probability.
    pub fn scale(&self, probability: f64) -> f64 {
        let p = probability.clamp(1e-15, 1.0 - 1e-15);
        let logit = (p / (1.0 - p)).ln();
        let scaled = logit / self.temperature;
        1.0 / (1.0 + (-scaled).exp())
    }

    /// Apply to a batch of probabilities.
    pub fn scale_batch(&self, probabilities: &[f64]) -> Vec<f64> {
        probabilities.iter().map(|&p| self.scale(p)).collect()
    }
}

/// Isotonic regression for probability calibration.
pub struct IsotonicRegression;

impl IsotonicRegression {
    /// Fit isotonic regression to calibrate probabilities.
    /// Returns a mapping from raw probabilities to calibrated probabilities.
    pub fn fit(samples: &[CalibrationSample]) -> CertificateResult<Vec<(f64, f64)>> {
        if samples.is_empty() {
            return Err(CertificateError::incomplete_data(
                "samples",
                "no samples for isotonic regression",
            ));
        }

        let mut indexed: Vec<(f64, f64)> = samples
            .iter()
            .map(|s| {
                (
                    s.predicted_probability,
                    if s.true_label { 1.0 } else { 0.0 },
                )
            })
            .collect();
        indexed.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Pool Adjacent Violators Algorithm (PAVA)
        let n = indexed.len();
        let mut calibrated: Vec<f64> = indexed.iter().map(|(_, y)| *y).collect();
        let mut weights: Vec<f64> = vec![1.0; n];

        let mut i = 0;
        while i < n {
            let mut j = i;
            // Find block where monotonicity is violated
            while j + 1 < n && calibrated[j] > calibrated[j + 1] {
                j += 1;
            }
            if j > i {
                // Pool: replace with weighted average
                let total_weight: f64 = weights[i..=j].iter().sum();
                let weighted_sum: f64 = (i..=j)
                    .map(|k| calibrated[k] * weights[k])
                    .sum();
                let avg = weighted_sum / total_weight;
                for k in i..=j {
                    calibrated[k] = avg;
                    weights[k] = total_weight / (j - i + 1) as f64;
                }
            }
            i = j + 1;
        }

        let mapping: Vec<(f64, f64)> = indexed
            .iter()
            .zip(calibrated.iter())
            .map(|((raw, _), cal)| (*raw, *cal))
            .collect();

        Ok(mapping)
    }

    /// Apply isotonic calibration to a new probability using linear interpolation.
    pub fn predict(mapping: &[(f64, f64)], probability: f64) -> f64 {
        if mapping.is_empty() {
            return probability;
        }
        if probability <= mapping[0].0 {
            return mapping[0].1;
        }
        if probability >= mapping.last().unwrap().0 {
            return mapping.last().unwrap().1;
        }

        // Binary search for the interpolation interval
        let pos = mapping
            .binary_search_by(|probe| {
                probe
                    .0
                    .partial_cmp(&probability)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or_else(|x| x);

        if pos == 0 {
            return mapping[0].1;
        }
        if pos >= mapping.len() {
            return mapping.last().unwrap().1;
        }

        let (x0, y0) = mapping[pos - 1];
        let (x1, y1) = mapping[pos];
        let dx = x1 - x0;
        if dx.abs() < 1e-15 {
            return (y0 + y1) / 2.0;
        }
        y0 + (y1 - y0) * (probability - x0) / dx
    }
}

/// Reliability diagram data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReliabilityDiagram {
    pub bins: Vec<ReliabilityBin>,
    pub expected_calibration_error: f64,
    pub max_calibration_error: f64,
    pub num_bins: usize,
}

/// A single bin in a reliability diagram.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReliabilityBin {
    pub bin_center: f64,
    pub mean_predicted: f64,
    pub fraction_positive: f64,
    pub count: usize,
    pub calibration_error: f64,
}

impl ReliabilityDiagram {
    /// Compute a reliability diagram from calibration samples.
    pub fn compute(samples: &[CalibrationSample], num_bins: usize) -> Self {
        let mut bins_data: Vec<(f64, f64, usize)> = vec![(0.0, 0.0, 0); num_bins];

        for s in samples {
            let bin_idx = ((s.predicted_probability * num_bins as f64) as usize).min(num_bins - 1);
            bins_data[bin_idx].0 += s.predicted_probability;
            bins_data[bin_idx].1 += if s.true_label { 1.0 } else { 0.0 };
            bins_data[bin_idx].2 += 1;
        }

        let total_samples = samples.len().max(1) as f64;
        let mut ece = 0.0;
        let mut mce = 0.0f64;

        let bins: Vec<ReliabilityBin> = (0..num_bins)
            .map(|i| {
                let (pred_sum, pos_sum, count) = bins_data[i];
                let bin_center = (i as f64 + 0.5) / num_bins as f64;
                if count == 0 {
                    ReliabilityBin {
                        bin_center,
                        mean_predicted: bin_center,
                        fraction_positive: 0.0,
                        count: 0,
                        calibration_error: 0.0,
                    }
                } else {
                    let mean_pred = pred_sum / count as f64;
                    let frac_pos = pos_sum / count as f64;
                    let cal_error = (mean_pred - frac_pos).abs();
                    ece += cal_error * count as f64 / total_samples;
                    mce = mce.max(cal_error);
                    ReliabilityBin {
                        bin_center,
                        mean_predicted: mean_pred,
                        fraction_positive: frac_pos,
                        count,
                        calibration_error: cal_error,
                    }
                }
            })
            .collect();

        Self {
            bins,
            expected_calibration_error: ece,
            max_calibration_error: mce,
            num_bins,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_samples(n: usize) -> Vec<CalibrationSample> {
        (0..n)
            .map(|i| {
                let p = i as f64 / n as f64;
                CalibrationSample {
                    predicted_probability: p,
                    true_label: p > 0.5,
                    instance_id: Some(format!("inst-{}", i)),
                }
            })
            .collect()
    }

    fn make_well_calibrated_samples(n: usize) -> Vec<CalibrationSample> {
        let mut samples = Vec::with_capacity(n);
        for i in 0..n {
            let p = i as f64 / n as f64;
            let label = (i % 100) as f64 / 100.0 < p;
            samples.push(CalibrationSample {
                predicted_probability: p,
                true_label: label,
                instance_id: Some(format!("inst-{}", i)),
            });
        }
        samples
    }

    #[test]
    fn test_calibrate_threshold() {
        let samples = make_samples(100);
        let result = CalibrationResult::calibrate_threshold(&samples).unwrap();
        assert!(result.optimal_threshold > 0.0 && result.optimal_threshold < 1.0);
        assert!(result.brier_score >= 0.0);
        assert!(result.f1_at_threshold >= 0.0);
    }

    #[test]
    fn test_calibrate_empty() {
        let result = CalibrationResult::calibrate_threshold(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_calibrate_too_few() {
        let samples = make_samples(5);
        let result = CalibrationResult::calibrate_threshold(&samples);
        assert!(result.is_err());
    }

    #[test]
    fn test_calibrate_all_positive() {
        let samples: Vec<CalibrationSample> = (0..20)
            .map(|i| CalibrationSample {
                predicted_probability: i as f64 / 20.0,
                true_label: true,
                instance_id: None,
            })
            .collect();
        let result = CalibrationResult::calibrate_threshold(&samples);
        assert!(result.is_err());
    }

    #[test]
    fn test_brier_score_perfect() {
        let samples = vec![
            CalibrationSample { predicted_probability: 1.0, true_label: true, instance_id: None },
            CalibrationSample { predicted_probability: 0.0, true_label: false, instance_id: None },
        ];
        let brier = CalibrationResult::compute_brier_score(&samples);
        assert!((brier - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_brier_score_worst() {
        let samples = vec![
            CalibrationSample { predicted_probability: 0.0, true_label: true, instance_id: None },
            CalibrationSample { predicted_probability: 1.0, true_label: false, instance_id: None },
        ];
        let brier = CalibrationResult::compute_brier_score(&samples);
        assert!((brier - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cross_validate() {
        let samples = make_well_calibrated_samples(200);
        let result = CalibrationResult::cross_validate_threshold(&samples, 5).unwrap();
        assert_eq!(result.k_folds, 5);
        assert!(!result.fold_thresholds.is_empty());
    }

    #[test]
    fn test_cross_validate_too_few() {
        let samples = make_samples(15);
        let result = CalibrationResult::cross_validate_threshold(&samples, 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_temperature_scaling() {
        let samples = make_well_calibrated_samples(200);
        let ts = TemperatureScaling::optimize(&samples).unwrap();
        assert!(ts.temperature > 0.0);
        assert!(ts.improvement >= -0.01); // Should not make things much worse
    }

    #[test]
    fn test_temperature_scale() {
        let ts = TemperatureScaling {
            temperature: 1.0,
            original_brier: 0.2,
            calibrated_brier: 0.15,
            improvement: 0.05,
        };
        let scaled = ts.scale(0.5);
        assert!((scaled - 0.5).abs() < 1e-10); // temp=1 should be identity at 0.5
    }

    #[test]
    fn test_temperature_scale_batch() {
        let ts = TemperatureScaling {
            temperature: 2.0,
            original_brier: 0.2,
            calibrated_brier: 0.15,
            improvement: 0.05,
        };
        let scaled = ts.scale_batch(&[0.2, 0.5, 0.8]);
        assert_eq!(scaled.len(), 3);
        // Higher temp should push probabilities toward 0.5
        assert!(scaled[0] > 0.2);
        assert!(scaled[2] < 0.8);
    }

    #[test]
    fn test_isotonic_regression() {
        let samples = make_well_calibrated_samples(100);
        let mapping = IsotonicRegression::fit(&samples).unwrap();
        assert!(!mapping.is_empty());
        // Mapping should be monotonically non-decreasing
        for i in 1..mapping.len() {
            assert!(mapping[i].1 >= mapping[i - 1].1 - 1e-10);
        }
    }

    #[test]
    fn test_isotonic_predict() {
        let mapping = vec![(0.0, 0.1), (0.3, 0.3), (0.7, 0.7), (1.0, 0.9)];
        let p = IsotonicRegression::predict(&mapping, 0.5);
        assert!(p > 0.3 && p < 0.7);
    }

    #[test]
    fn test_isotonic_predict_boundary() {
        let mapping = vec![(0.2, 0.1), (0.8, 0.9)];
        assert!((IsotonicRegression::predict(&mapping, 0.0) - 0.1).abs() < 1e-10);
        assert!((IsotonicRegression::predict(&mapping, 1.0) - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_reliability_diagram() {
        let samples = make_well_calibrated_samples(200);
        let diagram = ReliabilityDiagram::compute(&samples, 10);
        assert_eq!(diagram.num_bins, 10);
        assert!(diagram.expected_calibration_error >= 0.0);
        assert!(diagram.max_calibration_error >= 0.0);
    }

    #[test]
    fn test_auc_roc() {
        let samples = make_samples(100);
        let auc = CalibrationResult::compute_auc_roc(&samples);
        // Good separation should give high AUC
        assert!(auc > 0.5);
    }

    #[test]
    fn test_log_loss() {
        let samples = make_samples(100);
        let ll = CalibrationResult::compute_log_loss(&samples);
        assert!(ll > 0.0);
        assert!(ll.is_finite());
    }
}
