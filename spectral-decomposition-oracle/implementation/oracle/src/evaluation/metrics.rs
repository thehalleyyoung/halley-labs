// Evaluation metrics: McNemar test, Holm-Bonferroni correction, bootstrap CI,
// Spearman correlation, and related statistical utilities.

use serde::{Deserialize, Serialize};

/// McNemar's test for comparing two classifiers on paired data.
/// Takes two boolean vectors: correct_a[i] and correct_b[i].
/// Returns (chi2_statistic, p_value).
pub fn compute_mcnemar(correct_a: &[bool], correct_b: &[bool]) -> (f64, f64) {
    assert_eq!(correct_a.len(), correct_b.len());

    let mut b_count = 0.0_f64; // a wrong, b correct
    let mut c_count = 0.0_f64; // a correct, b wrong

    for (&a, &b) in correct_a.iter().zip(correct_b.iter()) {
        match (a, b) {
            (false, true) => b_count += 1.0,
            (true, false) => c_count += 1.0,
            _ => {}
        }
    }

    if b_count + c_count == 0.0 {
        return (0.0, 1.0);
    }

    // McNemar with continuity correction
    let statistic = (b_count - c_count).abs() - 1.0;
    let statistic = if statistic > 0.0 {
        statistic * statistic / (b_count + c_count)
    } else {
        0.0
    };

    // Approximate p-value using chi-squared distribution with 1 df
    let p_value = chi2_survival(statistic, 1.0);

    (statistic, p_value)
}

/// Approximate survival function (1 - CDF) for chi-squared distribution.
/// Uses a simple approximation for 1 degree of freedom.
fn chi2_survival(x: f64, _df: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    // For 1 df: P(X > x) = 2 * (1 - Phi(sqrt(x)))
    // Using standard normal survival approximation
    let z = x.sqrt();
    2.0 * normal_survival(z)
}

/// Standard normal survival function approximation.
fn normal_survival(z: f64) -> f64 {
    if z < 0.0 {
        return 1.0 - normal_survival(-z);
    }
    // Abramowitz & Stegun approximation
    let t = 1.0 / (1.0 + 0.2316419 * z);
    let poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
    let pdf = (-z * z / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt();
    (pdf * poly).max(0.0).min(1.0)
}

/// Holm-Bonferroni correction for multiple comparisons.
/// Returns vector of booleans indicating which tests are significant after correction.
pub fn holm_bonferroni(p_values: &[f64], alpha: f64) -> Vec<bool> {
    let m = p_values.len();
    if m == 0 {
        return vec![];
    }

    // Sort p-values with original indices
    let mut indexed: Vec<(usize, f64)> = p_values.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut significant = vec![false; m];
    for (rank, &(orig_idx, p)) in indexed.iter().enumerate() {
        let adjusted_alpha = alpha / (m - rank) as f64;
        if p <= adjusted_alpha {
            significant[orig_idx] = true;
        } else {
            break; // All remaining are also non-significant
        }
    }

    significant
}

/// Bootstrap confidence interval for a metric.
pub fn bootstrap_ci(
    values: &[f64],
    n_bootstrap: usize,
    confidence: f64,
    seed: u64,
) -> BootstrapCI {
    use rand::prelude::*;
    use rand_chacha::ChaCha8Rng;

    let n = values.len();
    if n == 0 {
        return BootstrapCI {
            point_estimate: 0.0,
            ci_lower: 0.0,
            ci_upper: 0.0,
            std_error: 0.0,
        };
    }

    let point_estimate = values.iter().sum::<f64>() / n as f64;

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut bootstrap_means = Vec::with_capacity(n_bootstrap);

    for _ in 0..n_bootstrap {
        let mut sum = 0.0;
        for _ in 0..n {
            let idx = rng.gen_range(0..n);
            sum += values[idx];
        }
        bootstrap_means.push(sum / n as f64);
    }

    bootstrap_means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let alpha = 1.0 - confidence;
    let lower_idx = ((alpha / 2.0) * n_bootstrap as f64).floor() as usize;
    let upper_idx = ((1.0 - alpha / 2.0) * n_bootstrap as f64).ceil() as usize;

    let ci_lower = bootstrap_means[lower_idx.min(bootstrap_means.len() - 1)];
    let ci_upper = bootstrap_means[upper_idx.min(bootstrap_means.len() - 1)];

    let mean_boot = bootstrap_means.iter().sum::<f64>() / bootstrap_means.len() as f64;
    let std_error = (bootstrap_means
        .iter()
        .map(|&m| (m - mean_boot).powi(2))
        .sum::<f64>()
        / (bootstrap_means.len() - 1).max(1) as f64)
        .sqrt();

    BootstrapCI {
        point_estimate,
        ci_lower,
        ci_upper,
        std_error,
    }
}

/// Bootstrap confidence interval result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrapCI {
    pub point_estimate: f64,
    pub ci_lower: f64,
    pub ci_upper: f64,
    pub std_error: f64,
}

impl BootstrapCI {
    pub fn contains(&self, value: f64) -> bool {
        value >= self.ci_lower && value <= self.ci_upper
    }

    pub fn width(&self) -> f64 {
        self.ci_upper - self.ci_lower
    }

    pub fn summary(&self) -> String {
        format!(
            "{:.4} [{:.4}, {:.4}] (SE={:.4})",
            self.point_estimate, self.ci_lower, self.ci_upper, self.std_error
        )
    }
}

/// Spearman rank correlation coefficient.
pub fn spearman_correlation(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    let n = x.len();
    if n < 2 {
        return 0.0;
    }

    let rank_x = rank_values(x);
    let rank_y = rank_values(y);

    pearson_correlation(&rank_x, &rank_y)
}

/// Pearson correlation coefficient.
pub fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    if n < 2.0 {
        return 0.0;
    }

    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    let mut cov = 0.0_f64;
    let mut var_x = 0.0_f64;
    let mut var_y = 0.0_f64;

    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let dx = xi - mean_x;
        let dy = yi - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    if var_x < 1e-15 || var_y < 1e-15 {
        return 0.0;
    }

    cov / (var_x * var_y).sqrt()
}

/// Assign ranks to values (average rank for ties).
fn rank_values(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut indexed: Vec<(usize, f64)> = values.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = vec![0.0_f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n - 1
            && (indexed[j + 1].1 - indexed[j].1).abs() < 1e-15
        {
            j += 1;
        }
        // Average rank for tied values
        let avg_rank = (i + j) as f64 / 2.0 + 1.0;
        for k in i..=j {
            ranks[indexed[k].0] = avg_rank;
        }
        i = j + 1;
    }

    ranks
}

/// Cohen's kappa coefficient.
pub fn cohen_kappa(predictions: &[usize], labels: &[usize], n_classes: usize) -> f64 {
    let n = predictions.len() as f64;
    if n == 0.0 {
        return 0.0;
    }

    let mut matrix = vec![vec![0usize; n_classes]; n_classes];
    for (&pred, &label) in predictions.iter().zip(labels.iter()) {
        if pred < n_classes && label < n_classes {
            matrix[label][pred] += 1;
        }
    }

    let p_o = (0..n_classes).map(|i| matrix[i][i] as f64).sum::<f64>() / n;

    let mut p_e = 0.0;
    for c in 0..n_classes {
        let row_sum = matrix[c].iter().sum::<usize>() as f64;
        let col_sum = (0..n_classes).map(|r| matrix[r][c]).sum::<usize>() as f64;
        p_e += (row_sum / n) * (col_sum / n);
    }

    if (1.0 - p_e).abs() < 1e-12 {
        return if (p_o - 1.0).abs() < 1e-12 { 1.0 } else { 0.0 };
    }

    (p_o - p_e) / (1.0 - p_e)
}

/// Weighted accuracy: weight each class by inverse frequency.
pub fn weighted_accuracy(predictions: &[usize], labels: &[usize], n_classes: usize) -> f64 {
    let n = labels.len();
    if n == 0 {
        return 0.0;
    }

    let mut class_counts = vec![0usize; n_classes];
    for &l in labels {
        if l < n_classes {
            class_counts[l] += 1;
        }
    }

    let mut weighted_correct = 0.0;
    let mut total_weight = 0.0;

    for (&pred, &label) in predictions.iter().zip(labels.iter()) {
        if label < n_classes && class_counts[label] > 0 {
            let weight = 1.0 / class_counts[label] as f64;
            if pred == label {
                weighted_correct += weight;
            }
            total_weight += weight;
        }
    }

    if total_weight > 0.0 {
        weighted_correct / total_weight
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mcnemar_identical() {
        let a = vec![true, false, true, false, true];
        let b = a.clone();
        let (stat, p) = compute_mcnemar(&a, &b);
        assert_eq!(stat, 0.0);
        assert_eq!(p, 1.0);
    }

    #[test]
    fn test_mcnemar_different() {
        let a = vec![true, true, false, false, true, true, false, false, true, true];
        let b = vec![false, false, true, true, false, false, true, true, false, false];
        let (stat, _p) = compute_mcnemar(&a, &b);
        assert!(stat > 0.0);
    }

    #[test]
    fn test_holm_bonferroni_all_significant() {
        let p_values = vec![0.001, 0.002, 0.003];
        let result = holm_bonferroni(&p_values, 0.05);
        assert!(result.iter().all(|&s| s));
    }

    #[test]
    fn test_holm_bonferroni_none_significant() {
        let p_values = vec![0.5, 0.6, 0.7];
        let result = holm_bonferroni(&p_values, 0.05);
        assert!(result.iter().all(|&s| !s));
    }

    #[test]
    fn test_holm_bonferroni_mixed() {
        let p_values = vec![0.001, 0.5, 0.003];
        let result = holm_bonferroni(&p_values, 0.05);
        assert!(result[0]); // 0.001 < 0.05/3
        assert!(!result[1]); // 0.5 > 0.05/1
    }

    #[test]
    fn test_bootstrap_ci_basic() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ci = bootstrap_ci(&values, 1000, 0.95, 42);
        assert!((ci.point_estimate - 3.0).abs() < 1e-10);
        assert!(ci.ci_lower < ci.point_estimate);
        assert!(ci.ci_upper > ci.point_estimate);
    }

    #[test]
    fn test_bootstrap_ci_empty() {
        let ci = bootstrap_ci(&[], 100, 0.95, 42);
        assert_eq!(ci.point_estimate, 0.0);
    }

    #[test]
    fn test_bootstrap_ci_contains() {
        let ci = BootstrapCI {
            point_estimate: 0.5,
            ci_lower: 0.4,
            ci_upper: 0.6,
            std_error: 0.05,
        };
        assert!(ci.contains(0.5));
        assert!(!ci.contains(0.3));
        assert!((ci.width() - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_spearman_perfect() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let rho = spearman_correlation(&x, &y);
        assert!((rho - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_spearman_inverse() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let rho = spearman_correlation(&x, &y);
        assert!((rho - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_pearson_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rank_values() {
        let values = vec![3.0, 1.0, 2.0];
        let ranks = rank_values(&values);
        assert_eq!(ranks[0], 3.0); // 3.0 is rank 3
        assert_eq!(ranks[1], 1.0); // 1.0 is rank 1
        assert_eq!(ranks[2], 2.0); // 2.0 is rank 2
    }

    #[test]
    fn test_rank_values_ties() {
        let values = vec![1.0, 2.0, 2.0, 3.0];
        let ranks = rank_values(&values);
        assert_eq!(ranks[0], 1.0);
        assert_eq!(ranks[1], 2.5); // tied
        assert_eq!(ranks[2], 2.5); // tied
        assert_eq!(ranks[3], 4.0);
    }

    #[test]
    fn test_cohen_kappa_perfect() {
        let preds = vec![0, 1, 2, 3, 0, 1];
        let labels = vec![0, 1, 2, 3, 0, 1];
        let kappa = cohen_kappa(&preds, &labels, 4);
        assert!((kappa - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_weighted_accuracy() {
        let preds = vec![0, 0, 1, 1];
        let labels = vec![0, 0, 1, 1];
        let wa = weighted_accuracy(&preds, &labels, 2);
        assert!((wa - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_normal_survival() {
        let p = normal_survival(0.0);
        assert!((p - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_bootstrap_summary() {
        let ci = BootstrapCI {
            point_estimate: 0.8,
            ci_lower: 0.75,
            ci_upper: 0.85,
            std_error: 0.025,
        };
        let summary = ci.summary();
        assert!(summary.contains("0.8000"));
    }
}
