// Hypothesis testing harness for the spectral decomposition oracle.
// Tests H1-H7 from the research hypotheses.

use crate::evaluation::metrics::{bootstrap_ci, spearman_correlation, pearson_correlation};
use serde::{Deserialize, Serialize};

/// Outcome of a hypothesis test.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HypothesisOutcome {
    Pass,
    Fail,
    Inconclusive,
}

impl std::fmt::Display for HypothesisOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HypothesisOutcome::Pass => write!(f, "PASS"),
            HypothesisOutcome::Fail => write!(f, "FAIL"),
            HypothesisOutcome::Inconclusive => write!(f, "INCONCLUSIVE"),
        }
    }
}

/// Result of a single hypothesis test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HypothesisResult {
    pub hypothesis_id: String,
    pub description: String,
    pub outcome: HypothesisOutcome,
    pub statistic: f64,
    pub threshold: f64,
    pub ci_lower: Option<f64>,
    pub ci_upper: Option<f64>,
    pub details: String,
}

impl HypothesisResult {
    pub fn summary(&self) -> String {
        let ci_str = match (self.ci_lower, self.ci_upper) {
            (Some(lo), Some(hi)) => format!(" 95% CI: [{:.4}, {:.4}]", lo, hi),
            _ => String::new(),
        };
        format!(
            "[{}] {}: {} (stat={:.4}, threshold={:.4}{})\n  {}",
            self.outcome, self.hypothesis_id, self.description,
            self.statistic, self.threshold, ci_str, self.details
        )
    }
}

/// H1: Spectral gap ratio correlates with degradation.
/// ρ(δ²/γ², degradation) ≥ 0.4
pub struct H1SpectralGapCorrelation;

impl H1SpectralGapCorrelation {
    pub fn test(spectral_ratios: &[f64], degradation: &[f64]) -> HypothesisResult {
        let threshold = 0.4;
        let rho = spearman_correlation(spectral_ratios, degradation);
        let ci = bootstrap_ci(
            &spectral_ratios
                .iter()
                .zip(degradation.iter())
                .map(|(&x, &y)| x * y)
                .collect::<Vec<f64>>(),
            1000,
            0.95,
            42,
        );

        let outcome = if rho >= threshold {
            HypothesisOutcome::Pass
        } else if rho >= threshold - 0.1 {
            HypothesisOutcome::Inconclusive
        } else {
            HypothesisOutcome::Fail
        };

        HypothesisResult {
            hypothesis_id: "H1".to_string(),
            description: "Spectral gap ratio correlates with degradation".to_string(),
            outcome,
            statistic: rho,
            threshold,
            ci_lower: Some(ci.ci_lower),
            ci_upper: Some(ci.ci_upper),
            details: format!("Spearman ρ = {:.4}", rho),
        }
    }
}

/// H2: Spectral features outperform syntactic by ≥5 percentage points.
pub struct H2SpectralVsSyntactic;

impl H2SpectralVsSyntactic {
    pub fn test(spectral_accuracy: f64, syntactic_accuracy: f64, p_value: f64) -> HypothesisResult {
        let threshold = 0.05; // 5 percentage points
        let diff = spectral_accuracy - syntactic_accuracy;
        let significant = p_value < 0.05;

        let outcome = if diff >= threshold && significant {
            HypothesisOutcome::Pass
        } else if diff >= 0.0 && !significant {
            HypothesisOutcome::Inconclusive
        } else {
            HypothesisOutcome::Fail
        };

        HypothesisResult {
            hypothesis_id: "H2".to_string(),
            description: "Spectral ≥ syntactic + 5pp".to_string(),
            outcome,
            statistic: diff,
            threshold,
            ci_lower: None,
            ci_upper: None,
            details: format!(
                "spectral={:.4}, syntactic={:.4}, diff={:.4}, p={:.4}",
                spectral_accuracy, syntactic_accuracy, diff, p_value
            ),
        }
    }
}

/// H3: L3 (label quality) correlation ≥ 0.4.
pub struct H3LabelCorrelation;

impl H3LabelCorrelation {
    pub fn test(predicted_scores: &[f64], actual_improvements: &[f64]) -> HypothesisResult {
        let threshold = 0.4;
        let rho = spearman_correlation(predicted_scores, actual_improvements);

        let outcome = if rho >= threshold {
            HypothesisOutcome::Pass
        } else if rho >= threshold - 0.1 {
            HypothesisOutcome::Inconclusive
        } else {
            HypothesisOutcome::Fail
        };

        HypothesisResult {
            hypothesis_id: "H3".to_string(),
            description: "L3 correlation ≥ 0.4".to_string(),
            outcome,
            statistic: rho,
            threshold,
            ci_lower: None,
            ci_upper: None,
            details: format!("Spearman ρ = {:.4}", rho),
        }
    }
}

/// H4: Futility predictor precision ≥ 80%.
pub struct H4FutilityPrecision;

impl H4FutilityPrecision {
    pub fn test(
        precisions: &[f64], // per-fold precisions
    ) -> HypothesisResult {
        let threshold = 0.8;
        let mean_precision = if precisions.is_empty() {
            0.0
        } else {
            precisions.iter().sum::<f64>() / precisions.len() as f64
        };

        let ci = bootstrap_ci(precisions, 1000, 0.95, 42);

        let outcome = if mean_precision >= threshold {
            HypothesisOutcome::Pass
        } else if ci.ci_upper >= threshold {
            HypothesisOutcome::Inconclusive
        } else {
            HypothesisOutcome::Fail
        };

        HypothesisResult {
            hypothesis_id: "H4".to_string(),
            description: "Futility precision ≥ 80%".to_string(),
            outcome,
            statistic: mean_precision,
            threshold,
            ci_lower: Some(ci.ci_lower),
            ci_upper: Some(ci.ci_upper),
            details: format!("Mean precision = {:.4}", mean_precision),
        }
    }
}

/// H5: Cross-scaling correlation ρ > 0.9.
pub struct H5CrossScaling;

impl H5CrossScaling {
    pub fn test(small_scores: &[f64], large_scores: &[f64]) -> HypothesisResult {
        let threshold = 0.9;
        let rho = spearman_correlation(small_scores, large_scores);

        let outcome = if rho > threshold {
            HypothesisOutcome::Pass
        } else if rho > threshold - 0.1 {
            HypothesisOutcome::Inconclusive
        } else {
            HypothesisOutcome::Fail
        };

        HypothesisResult {
            hypothesis_id: "H5".to_string(),
            description: "Cross-scaling ρ > 0.9".to_string(),
            outcome,
            statistic: rho,
            threshold,
            ci_lower: None,
            ci_upper: None,
            details: format!("Spearman ρ = {:.4}", rho),
        }
    }
}

/// H6: R²(γ₂ ~ syntactic) < 0.70 (spectral features not redundant with syntactic).
pub struct H6SpectralNonRedundancy;

impl H6SpectralNonRedundancy {
    pub fn test(spectral_features: &[f64], syntactic_features: &[f64]) -> HypothesisResult {
        let threshold = 0.70;
        let r = pearson_correlation(spectral_features, syntactic_features);
        let r_squared = r * r;

        let outcome = if r_squared < threshold {
            HypothesisOutcome::Pass
        } else if r_squared < threshold + 0.1 {
            HypothesisOutcome::Inconclusive
        } else {
            HypothesisOutcome::Fail
        };

        HypothesisResult {
            hypothesis_id: "H6".to_string(),
            description: "R²(γ₂~syntactic) < 0.70".to_string(),
            outcome,
            statistic: r_squared,
            threshold,
            ci_lower: None,
            ci_upper: None,
            details: format!("R² = {:.4} (r = {:.4})", r_squared, r),
        }
    }
}

/// H7: Improvement on non-trivial classes.
pub struct H7NonTrivialImprovement;

impl H7NonTrivialImprovement {
    pub fn test(
        per_class_accuracies: &[f64],      // accuracy for each class
        baseline_accuracies: &[f64],        // baseline accuracy for each class
        trivial_class_idx: Option<usize>,   // index of "None" class
    ) -> HypothesisResult {
        let threshold = 0.0; // any improvement

        let mut improvements = Vec::new();
        for (i, (&acc, &baseline)) in per_class_accuracies
            .iter()
            .zip(baseline_accuracies.iter())
            .enumerate()
        {
            if Some(i) != trivial_class_idx {
                improvements.push(acc - baseline);
            }
        }

        let mean_improvement = if improvements.is_empty() {
            0.0
        } else {
            improvements.iter().sum::<f64>() / improvements.len() as f64
        };

        let all_improved = improvements.iter().all(|&d| d > 0.0);

        let outcome = if all_improved && mean_improvement > threshold {
            HypothesisOutcome::Pass
        } else if mean_improvement > threshold {
            HypothesisOutcome::Inconclusive
        } else {
            HypothesisOutcome::Fail
        };

        HypothesisResult {
            hypothesis_id: "H7".to_string(),
            description: "Improvement on non-trivial classes".to_string(),
            outcome,
            statistic: mean_improvement,
            threshold,
            ci_lower: None,
            ci_upper: None,
            details: format!(
                "Mean improvement = {:.4}, all improved = {}",
                mean_improvement, all_improved
            ),
        }
    }
}

/// Hypothesis testing harness that runs all hypotheses.
pub struct HypothesisHarness;

impl HypothesisHarness {
    /// Run a specific hypothesis test by ID.
    pub fn run_hypothesis(id: &str, data: &HypothesisData) -> Option<HypothesisResult> {
        match id {
            "H1" => Some(H1SpectralGapCorrelation::test(
                &data.spectral_ratios,
                &data.degradation_values,
            )),
            "H2" => Some(H2SpectralVsSyntactic::test(
                data.spectral_accuracy,
                data.syntactic_accuracy,
                data.h2_p_value,
            )),
            "H3" => Some(H3LabelCorrelation::test(
                &data.predicted_scores,
                &data.actual_improvements,
            )),
            "H4" => Some(H4FutilityPrecision::test(&data.futility_precisions)),
            "H5" => Some(H5CrossScaling::test(
                &data.small_scale_scores,
                &data.large_scale_scores,
            )),
            "H6" => Some(H6SpectralNonRedundancy::test(
                &data.spectral_feature_values,
                &data.syntactic_feature_values,
            )),
            "H7" => Some(H7NonTrivialImprovement::test(
                &data.per_class_accuracies,
                &data.baseline_accuracies,
                data.trivial_class_idx,
            )),
            _ => None,
        }
    }

    /// Run all hypotheses and return results.
    pub fn run_all(data: &HypothesisData) -> Vec<HypothesisResult> {
        let ids = ["H1", "H2", "H3", "H4", "H5", "H6", "H7"];
        ids.iter()
            .filter_map(|id| Self::run_hypothesis(id, data))
            .collect()
    }

    /// Generate a summary report of all hypothesis tests.
    pub fn report(results: &[HypothesisResult]) -> String {
        let mut report = String::from("=== Hypothesis Testing Report ===\n\n");

        let passed = results.iter().filter(|r| r.outcome == HypothesisOutcome::Pass).count();
        let failed = results.iter().filter(|r| r.outcome == HypothesisOutcome::Fail).count();
        let inconclusive = results
            .iter()
            .filter(|r| r.outcome == HypothesisOutcome::Inconclusive)
            .count();

        report.push_str(&format!(
            "Summary: {} passed, {} failed, {} inconclusive out of {} tests\n\n",
            passed,
            failed,
            inconclusive,
            results.len()
        ));

        for result in results {
            report.push_str(&result.summary());
            report.push('\n');
        }

        report
    }
}

/// Data bundle for hypothesis testing.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HypothesisData {
    pub spectral_ratios: Vec<f64>,
    pub degradation_values: Vec<f64>,
    pub spectral_accuracy: f64,
    pub syntactic_accuracy: f64,
    pub h2_p_value: f64,
    pub predicted_scores: Vec<f64>,
    pub actual_improvements: Vec<f64>,
    pub futility_precisions: Vec<f64>,
    pub small_scale_scores: Vec<f64>,
    pub large_scale_scores: Vec<f64>,
    pub spectral_feature_values: Vec<f64>,
    pub syntactic_feature_values: Vec<f64>,
    pub per_class_accuracies: Vec<f64>,
    pub baseline_accuracies: Vec<f64>,
    pub trivial_class_idx: Option<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_correlated_data(n: usize, correlation: f64) -> (Vec<f64>, Vec<f64>) {
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let y: Vec<f64> = x
            .iter()
            .enumerate()
            .map(|(i, &xi)| xi * correlation + (1.0 - correlation.abs()) * (i as f64 % 3.0))
            .collect();
        (x, y)
    }

    #[test]
    fn test_h1_pass() {
        let (x, y) = make_correlated_data(50, 0.9);
        let result = H1SpectralGapCorrelation::test(&x, &y);
        assert_eq!(result.hypothesis_id, "H1");
        assert!(result.statistic > 0.3);
    }

    #[test]
    fn test_h1_fail() {
        let x = vec![1.0, 3.0, 2.0, 5.0, 4.0];
        let y = vec![5.0, 2.0, 4.0, 1.0, 3.0];
        let result = H1SpectralGapCorrelation::test(&x, &y);
        assert_eq!(result.outcome, HypothesisOutcome::Fail);
    }

    #[test]
    fn test_h2_pass() {
        let result = H2SpectralVsSyntactic::test(0.85, 0.78, 0.01);
        assert_eq!(result.outcome, HypothesisOutcome::Pass);
    }

    #[test]
    fn test_h2_fail() {
        let result = H2SpectralVsSyntactic::test(0.80, 0.82, 0.01);
        assert_eq!(result.outcome, HypothesisOutcome::Fail);
    }

    #[test]
    fn test_h3_pass() {
        let (x, y) = make_correlated_data(50, 0.9);
        let result = H3LabelCorrelation::test(&x, &y);
        assert!(result.statistic > 0.3);
    }

    #[test]
    fn test_h4_pass() {
        let precisions = vec![0.85, 0.82, 0.88, 0.81, 0.84];
        let result = H4FutilityPrecision::test(&precisions);
        assert_eq!(result.outcome, HypothesisOutcome::Pass);
    }

    #[test]
    fn test_h4_fail() {
        let precisions = vec![0.5, 0.6, 0.55, 0.45, 0.5];
        let result = H4FutilityPrecision::test(&precisions);
        assert_eq!(result.outcome, HypothesisOutcome::Fail);
    }

    #[test]
    fn test_h5_pass() {
        let small = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let large = vec![1.1, 2.1, 3.1, 4.1, 5.1];
        let result = H5CrossScaling::test(&small, &large);
        assert_eq!(result.outcome, HypothesisOutcome::Pass);
    }

    #[test]
    fn test_h6_pass() {
        // Low correlation between spectral and syntactic
        let spectral = vec![1.0, 3.0, 2.0, 5.0, 4.0];
        let syntactic = vec![3.0, 2.0, 5.0, 4.0, 1.0];
        let result = H6SpectralNonRedundancy::test(&spectral, &syntactic);
        assert_eq!(result.outcome, HypothesisOutcome::Pass);
    }

    #[test]
    fn test_h6_fail() {
        // High correlation
        let spectral = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let syntactic = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = H6SpectralNonRedundancy::test(&spectral, &syntactic);
        assert_eq!(result.outcome, HypothesisOutcome::Fail);
    }

    #[test]
    fn test_h7_pass() {
        let per_class = vec![0.8, 0.7, 0.75, 0.9];
        let baseline = vec![0.6, 0.5, 0.55, 0.85];
        let result = H7NonTrivialImprovement::test(&per_class, &baseline, Some(3));
        assert_eq!(result.outcome, HypothesisOutcome::Pass);
    }

    #[test]
    fn test_h7_fail() {
        let per_class = vec![0.5, 0.4, 0.3, 0.9];
        let baseline = vec![0.6, 0.5, 0.55, 0.85];
        let result = H7NonTrivialImprovement::test(&per_class, &baseline, Some(3));
        assert_eq!(result.outcome, HypothesisOutcome::Fail);
    }

    #[test]
    fn test_harness_run_all() {
        let data = HypothesisData {
            spectral_ratios: (0..20).map(|i| i as f64).collect(),
            degradation_values: (0..20).map(|i| i as f64 * 0.5).collect(),
            spectral_accuracy: 0.85,
            syntactic_accuracy: 0.78,
            h2_p_value: 0.01,
            predicted_scores: (0..20).map(|i| i as f64).collect(),
            actual_improvements: (0..20).map(|i| i as f64 * 0.3).collect(),
            futility_precisions: vec![0.85, 0.82, 0.88],
            small_scale_scores: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            large_scale_scores: vec![1.1, 2.0, 3.1, 4.0, 5.1],
            spectral_feature_values: vec![1.0, 3.0, 2.0, 5.0, 4.0],
            syntactic_feature_values: vec![5.0, 2.0, 4.0, 1.0, 3.0],
            per_class_accuracies: vec![0.8, 0.7, 0.75, 0.9],
            baseline_accuracies: vec![0.6, 0.5, 0.55, 0.85],
            trivial_class_idx: Some(3),
        };

        let results = HypothesisHarness::run_all(&data);
        assert_eq!(results.len(), 7);
    }

    #[test]
    fn test_harness_report() {
        let results = vec![HypothesisResult {
            hypothesis_id: "H1".to_string(),
            description: "test hypothesis".to_string(),
            outcome: HypothesisOutcome::Pass,
            statistic: 0.5,
            threshold: 0.4,
            ci_lower: Some(0.3),
            ci_upper: Some(0.7),
            details: "test".to_string(),
        }];
        let report = HypothesisHarness::report(&results);
        assert!(report.contains("PASS"));
        assert!(report.contains("H1"));
    }

    #[test]
    fn test_hypothesis_outcome_display() {
        assert_eq!(HypothesisOutcome::Pass.to_string(), "PASS");
        assert_eq!(HypothesisOutcome::Fail.to_string(), "FAIL");
        assert_eq!(HypothesisOutcome::Inconclusive.to_string(), "INCONCLUSIVE");
    }

    #[test]
    fn test_run_single_hypothesis() {
        let data = HypothesisData {
            spectral_accuracy: 0.85,
            syntactic_accuracy: 0.78,
            h2_p_value: 0.01,
            ..Default::default()
        };
        let result = HypothesisHarness::run_hypothesis("H2", &data).unwrap();
        assert_eq!(result.hypothesis_id, "H2");
    }

    #[test]
    fn test_invalid_hypothesis_id() {
        let data = HypothesisData::default();
        assert!(HypothesisHarness::run_hypothesis("H99", &data).is_none());
    }
}
