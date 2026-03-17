// Threshold calibration for futility prediction: ROC curves, precision-recall
// curves, optimal threshold selection, and stability analysis.

use serde::{Deserialize, Serialize};

/// A point on the ROC curve.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ROCPoint {
    pub threshold: f64,
    pub fpr: f64,
    pub tpr: f64,
}

/// A point on the Precision-Recall curve.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PRPoint {
    pub threshold: f64,
    pub precision: f64,
    pub recall: f64,
}

/// Strategy for selecting the decision threshold.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThresholdStrategy {
    Fixed { threshold: f64 },
    TargetPrecision { target: f64 },
    TargetRecall { target: f64 },
    MaxF1,
    Youden,
    Bayesian { prior_positive: f64 },
}

/// Threshold calibration engine.
#[derive(Debug, Clone)]
pub struct ThresholdCalibrator {
    pub roc_curve: Vec<ROCPoint>,
    pub pr_curve: Vec<PRPoint>,
}

impl ThresholdCalibrator {
    pub fn new() -> Self {
        Self {
            roc_curve: Vec::new(),
            pr_curve: Vec::new(),
        }
    }

    /// Compute ROC curve from scores and binary labels.
    pub fn compute_roc(scores: &[f64], labels: &[bool]) -> Vec<ROCPoint> {
        let n = scores.len();
        if n == 0 {
            return vec![];
        }

        let total_pos = labels.iter().filter(|&&l| l).count() as f64;
        let total_neg = labels.iter().filter(|&&l| !l).count() as f64;

        if total_pos == 0.0 || total_neg == 0.0 {
            return vec![ROCPoint {
                threshold: 0.5,
                fpr: 0.0,
                tpr: 0.0,
            }];
        }

        // Sort by score descending
        let mut indexed: Vec<(f64, bool)> =
            scores.iter().copied().zip(labels.iter().copied()).collect();
        indexed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut roc = Vec::new();
        roc.push(ROCPoint {
            threshold: f64::INFINITY,
            fpr: 0.0,
            tpr: 0.0,
        });

        let mut tp = 0.0_f64;
        let mut fp = 0.0_f64;

        for (score, label) in &indexed {
            if *label {
                tp += 1.0;
            } else {
                fp += 1.0;
            }
            roc.push(ROCPoint {
                threshold: *score,
                fpr: fp / total_neg,
                tpr: tp / total_pos,
            });
        }

        roc
    }

    /// Compute Precision-Recall curve from scores and binary labels.
    pub fn compute_pr(scores: &[f64], labels: &[bool]) -> Vec<PRPoint> {
        let n = scores.len();
        if n == 0 {
            return vec![];
        }

        let total_pos = labels.iter().filter(|&&l| l).count() as f64;
        if total_pos == 0.0 {
            return vec![PRPoint {
                threshold: 0.5,
                precision: 0.0,
                recall: 0.0,
            }];
        }

        let mut indexed: Vec<(f64, bool)> =
            scores.iter().copied().zip(labels.iter().copied()).collect();
        indexed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut pr = Vec::new();
        let mut tp = 0.0_f64;
        let mut fp = 0.0_f64;

        for (score, label) in &indexed {
            if *label {
                tp += 1.0;
            } else {
                fp += 1.0;
            }
            let precision = tp / (tp + fp);
            let recall = tp / total_pos;
            pr.push(PRPoint {
                threshold: *score,
                precision,
                recall,
            });
        }

        pr
    }

    /// Compute Area Under ROC Curve (AUC-ROC) using the trapezoidal rule.
    pub fn auc_roc(roc: &[ROCPoint]) -> f64 {
        if roc.len() < 2 {
            return 0.0;
        }
        let mut auc = 0.0;
        for i in 1..roc.len() {
            let dx = roc[i].fpr - roc[i - 1].fpr;
            let avg_y = (roc[i].tpr + roc[i - 1].tpr) / 2.0;
            auc += dx * avg_y;
        }
        auc.abs()
    }

    /// Compute Area Under Precision-Recall Curve (AUC-PR).
    pub fn auc_pr(pr: &[PRPoint]) -> f64 {
        if pr.len() < 2 {
            return 0.0;
        }
        let mut auc = 0.0;
        for i in 1..pr.len() {
            let dx = pr[i].recall - pr[i - 1].recall;
            let avg_y = (pr[i].precision + pr[i - 1].precision) / 2.0;
            auc += dx * avg_y;
        }
        auc.abs()
    }

    /// Calibrate: select optimal threshold based on strategy.
    pub fn calibrate(
        &self,
        scores: &[f64],
        labels: &[bool],
        strategy: &ThresholdStrategy,
    ) -> f64 {
        match strategy {
            ThresholdStrategy::Fixed { threshold } => *threshold,
            ThresholdStrategy::TargetPrecision { target } => {
                Self::threshold_for_target_precision(scores, labels, *target)
            }
            ThresholdStrategy::TargetRecall { target } => {
                Self::threshold_for_target_recall(scores, labels, *target)
            }
            ThresholdStrategy::MaxF1 => Self::threshold_for_max_f1(scores, labels),
            ThresholdStrategy::Youden => Self::threshold_youden(scores, labels),
            ThresholdStrategy::Bayesian { prior_positive } => {
                Self::threshold_bayesian(scores, labels, *prior_positive)
            }
        }
    }

    /// Find threshold achieving target precision (minimum threshold meeting target).
    pub fn threshold_for_target_precision(
        scores: &[f64],
        labels: &[bool],
        target: f64,
    ) -> f64 {
        let pr = Self::compute_pr(scores, labels);
        // Find the lowest threshold that achieves the target precision
        let mut best_threshold = 0.9; // conservative default
        let mut best_recall = 0.0;

        for point in &pr {
            if point.precision >= target && point.recall > best_recall {
                best_recall = point.recall;
                best_threshold = point.threshold;
            }
        }

        best_threshold
    }

    /// Find threshold achieving target recall.
    pub fn threshold_for_target_recall(
        scores: &[f64],
        labels: &[bool],
        target: f64,
    ) -> f64 {
        let pr = Self::compute_pr(scores, labels);
        let mut best_threshold = 0.5;
        let mut best_precision = 0.0;

        for point in &pr {
            if point.recall >= target && point.precision > best_precision {
                best_precision = point.precision;
                best_threshold = point.threshold;
            }
        }

        best_threshold
    }

    /// Find threshold maximizing F1 score.
    pub fn threshold_for_max_f1(scores: &[f64], labels: &[bool]) -> f64 {
        let pr = Self::compute_pr(scores, labels);
        let mut best_f1 = 0.0;
        let mut best_threshold = 0.5;

        for point in &pr {
            let f1 = if point.precision + point.recall > 0.0 {
                2.0 * point.precision * point.recall / (point.precision + point.recall)
            } else {
                0.0
            };
            if f1 > best_f1 {
                best_f1 = f1;
                best_threshold = point.threshold;
            }
        }

        best_threshold
    }

    /// Find threshold using Youden's J statistic (maximize TPR - FPR).
    pub fn threshold_youden(scores: &[f64], labels: &[bool]) -> f64 {
        let roc = Self::compute_roc(scores, labels);
        let mut best_j = f64::NEG_INFINITY;
        let mut best_threshold = 0.5;

        for point in &roc {
            let j = point.tpr - point.fpr;
            if j > best_j {
                best_j = j;
                best_threshold = point.threshold;
            }
        }

        best_threshold
    }

    /// Bayesian threshold estimation incorporating class prior.
    pub fn threshold_bayesian(
        scores: &[f64],
        labels: &[bool],
        prior_positive: f64,
    ) -> f64 {
        // Adjust threshold based on prior: lower threshold if positives are rarer
        let base_threshold = Self::threshold_for_max_f1(scores, labels);
        let prior_ratio = prior_positive / (1.0 - prior_positive).max(1e-10);
        let adjusted = base_threshold - 0.1 * (1.0 - prior_ratio).max(-0.5).min(0.5);
        adjusted.max(0.1).min(0.95)
    }

    /// Analyze threshold stability across cross-validation folds.
    pub fn stability_analysis(fold_thresholds: &[f64]) -> ThresholdStability {
        let n = fold_thresholds.len() as f64;
        if n == 0.0 {
            return ThresholdStability {
                mean: 0.5,
                std: 0.0,
                range: 0.0,
                cv: 0.0,
                ci_lower: 0.5,
                ci_upper: 0.5,
            };
        }

        let mean = fold_thresholds.iter().sum::<f64>() / n;
        let variance = fold_thresholds
            .iter()
            .map(|&t| (t - mean).powi(2))
            .sum::<f64>()
            / (n - 1.0).max(1.0);
        let std = variance.sqrt();
        let range = fold_thresholds
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
            - fold_thresholds
                .iter()
                .cloned()
                .fold(f64::INFINITY, f64::min);
        let cv = if mean.abs() > 1e-10 { std / mean } else { 0.0 };

        // 95% CI using t-distribution approximation (z ≈ 1.96 for large n)
        let t_val = 1.96;
        let se = std / n.sqrt();
        let ci_lower = mean - t_val * se;
        let ci_upper = mean + t_val * se;

        ThresholdStability {
            mean,
            std,
            range,
            cv,
            ci_lower,
            ci_upper,
        }
    }
}

/// Threshold stability metrics across CV folds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdStability {
    pub mean: f64,
    pub std: f64,
    pub range: f64,
    pub cv: f64, // coefficient of variation
    pub ci_lower: f64,
    pub ci_upper: f64,
}

impl ThresholdStability {
    /// Return true if the threshold is considered stable (CV < 0.1).
    pub fn is_stable(&self) -> bool {
        self.cv < 0.1
    }

    pub fn summary(&self) -> String {
        format!(
            "Threshold: {:.4} ± {:.4} (95% CI: [{:.4}, {:.4}]), range={:.4}, CV={:.4}",
            self.mean, self.std, self.ci_lower, self.ci_upper, self.range, self.cv
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_scores_labels(n: usize) -> (Vec<f64>, Vec<bool>) {
        let mut scores = Vec::new();
        let mut labels = Vec::new();
        for i in 0..n {
            let score = i as f64 / n as f64;
            let label = score > 0.5;
            scores.push(score);
            labels.push(label);
        }
        (scores, labels)
    }

    #[test]
    fn test_roc_curve_basic() {
        let (scores, labels) = make_scores_labels(100);
        let roc = ThresholdCalibrator::compute_roc(&scores, &labels);
        assert!(!roc.is_empty());
        // First point should be (0, 0)
        assert_eq!(roc[0].fpr, 0.0);
        assert_eq!(roc[0].tpr, 0.0);
    }

    #[test]
    fn test_pr_curve_basic() {
        let (scores, labels) = make_scores_labels(100);
        let pr = ThresholdCalibrator::compute_pr(&scores, &labels);
        assert!(!pr.is_empty());
    }

    #[test]
    fn test_auc_roc_perfect() {
        // Perfect classifier: all positives scored higher than all negatives
        let scores = vec![0.9, 0.8, 0.2, 0.1];
        let labels = vec![true, true, false, false];
        let roc = ThresholdCalibrator::compute_roc(&scores, &labels);
        let auc = ThresholdCalibrator::auc_roc(&roc);
        assert!((auc - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_auc_roc_random() {
        // Random classifier should have AUC ≈ 0.5
        let scores = vec![0.1, 0.9, 0.3, 0.7, 0.5, 0.6, 0.2, 0.8];
        let labels = vec![false, false, true, true, false, true, true, false];
        let roc = ThresholdCalibrator::compute_roc(&scores, &labels);
        let auc = ThresholdCalibrator::auc_roc(&roc);
        assert!(auc >= 0.0 && auc <= 1.0);
    }

    #[test]
    fn test_threshold_for_target_precision() {
        let (scores, labels) = make_scores_labels(100);
        let threshold =
            ThresholdCalibrator::threshold_for_target_precision(&scores, &labels, 0.8);
        assert!(threshold >= 0.0 && threshold <= 1.0);
    }

    #[test]
    fn test_threshold_max_f1() {
        let (scores, labels) = make_scores_labels(100);
        let threshold = ThresholdCalibrator::threshold_for_max_f1(&scores, &labels);
        assert!(threshold >= 0.0 && threshold <= 1.0);
    }

    #[test]
    fn test_threshold_youden() {
        let (scores, labels) = make_scores_labels(100);
        let threshold = ThresholdCalibrator::threshold_youden(&scores, &labels);
        assert!(threshold.is_finite());
    }

    #[test]
    fn test_threshold_bayesian() {
        let (scores, labels) = make_scores_labels(100);
        let threshold =
            ThresholdCalibrator::threshold_bayesian(&scores, &labels, 0.3);
        assert!(threshold >= 0.1 && threshold <= 0.95);
    }

    #[test]
    fn test_calibrate_fixed() {
        let calibrator = ThresholdCalibrator::new();
        let threshold = calibrator.calibrate(
            &[0.5],
            &[true],
            &ThresholdStrategy::Fixed { threshold: 0.42 },
        );
        assert_eq!(threshold, 0.42);
    }

    #[test]
    fn test_stability_analysis() {
        let thresholds = vec![0.50, 0.52, 0.48, 0.51, 0.49];
        let stability = ThresholdCalibrator::stability_analysis(&thresholds);
        assert!((stability.mean - 0.5).abs() < 0.02);
        assert!(stability.is_stable());
    }

    #[test]
    fn test_stability_unstable() {
        let thresholds = vec![0.1, 0.9, 0.3, 0.7, 0.5];
        let stability = ThresholdCalibrator::stability_analysis(&thresholds);
        assert!(!stability.is_stable());
    }

    #[test]
    fn test_stability_empty() {
        let stability = ThresholdCalibrator::stability_analysis(&[]);
        assert_eq!(stability.mean, 0.5);
    }

    #[test]
    fn test_auc_pr() {
        let scores = vec![0.9, 0.8, 0.2, 0.1];
        let labels = vec![true, true, false, false];
        let pr = ThresholdCalibrator::compute_pr(&scores, &labels);
        let auc = ThresholdCalibrator::auc_pr(&pr);
        assert!(auc > 0.0);
    }

    #[test]
    fn test_roc_empty() {
        let roc = ThresholdCalibrator::compute_roc(&[], &[]);
        assert!(roc.is_empty());
    }

    #[test]
    fn test_threshold_stability_summary() {
        let stability = ThresholdStability {
            mean: 0.5,
            std: 0.02,
            range: 0.04,
            cv: 0.04,
            ci_lower: 0.48,
            ci_upper: 0.52,
        };
        let summary = stability.summary();
        assert!(summary.contains("0.5000"));
    }
}
