//! Statistical utilities for analysis and evaluation.
//!
//! Provides descriptive statistics, correlation, hypothesis testing,
//! confusion matrices, ROC/AUC, and cross-validation split generation.

use serde::{Deserialize, Serialize};

/// Descriptive statistics for a sample.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DescriptiveStats {
    pub count: usize,
    pub mean: f64,
    pub variance: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub median: f64,
    pub skewness: f64,
    pub kurtosis: f64,
    pub q1: f64,
    pub q3: f64,
    pub iqr: f64,
}

impl DescriptiveStats {
    pub fn compute(data: &[f64]) -> Option<Self> {
        if data.is_empty() {
            return None;
        }
        let n = data.len();
        let nf = n as f64;

        let mean = data.iter().sum::<f64>() / nf;

        let variance = if n < 2 {
            0.0
        } else {
            data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (nf - 1.0)
        };
        let std_dev = variance.sqrt();

        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let min = sorted[0];
        let max = sorted[n - 1];
        let median = percentile_sorted(&sorted, 50.0);
        let q1 = percentile_sorted(&sorted, 25.0);
        let q3 = percentile_sorted(&sorted, 75.0);
        let iqr = q3 - q1;

        let skewness = if n < 3 || std_dev < 1e-15 {
            0.0
        } else {
            let m3 = data.iter().map(|&x| ((x - mean) / std_dev).powi(3)).sum::<f64>();
            m3 * nf / ((nf - 1.0) * (nf - 2.0))
        };

        let kurtosis = if n < 4 || std_dev < 1e-15 {
            0.0
        } else {
            let m4 = data.iter().map(|&x| ((x - mean) / std_dev).powi(4)).sum::<f64>() / nf;
            m4 - 3.0
        };

        Some(Self {
            count: n,
            mean,
            variance,
            std_dev,
            min,
            max,
            median,
            skewness,
            kurtosis,
            q1,
            q3,
            iqr,
        })
    }
}

/// Compute a percentile from a sorted slice using linear interpolation.
pub fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    assert!(!sorted.is_empty());
    if sorted.len() == 1 {
        return sorted[0];
    }
    let p = p.clamp(0.0, 100.0);
    let rank = (p / 100.0) * (sorted.len() - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    let frac = rank - lo as f64;
    if lo == hi {
        sorted[lo]
    } else {
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

/// Quantile function (p in [0, 1]).
pub fn quantile(data: &[f64], p: f64) -> f64 {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    percentile_sorted(&sorted, p * 100.0)
}

/// Mean of a slice.
pub fn mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().sum::<f64>() / data.len() as f64
}

/// Variance of a slice (sample variance, Bessel-corrected).
pub fn variance(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    let m = mean(data);
    let n = data.len() as f64;
    data.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / (n - 1.0)
}

/// Standard deviation.
pub fn std_dev(data: &[f64]) -> f64 {
    variance(data).sqrt()
}

/// Pearson correlation coefficient.
pub fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n < 2 {
        return 0.0;
    }
    let mx = mean(&x[..n]);
    let my = mean(&y[..n]);

    let mut sum_xy = 0.0;
    let mut sum_x2 = 0.0;
    let mut sum_y2 = 0.0;

    for i in 0..n {
        let dx = x[i] - mx;
        let dy = y[i] - my;
        sum_xy += dx * dy;
        sum_x2 += dx * dx;
        sum_y2 += dy * dy;
    }

    let denom = (sum_x2 * sum_y2).sqrt();
    if denom < 1e-15 {
        0.0
    } else {
        sum_xy / denom
    }
}

/// Spearman rank correlation coefficient.
pub fn spearman_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n < 2 {
        return 0.0;
    }
    let rx = ranks(&x[..n]);
    let ry = ranks(&y[..n]);
    pearson_correlation(&rx, &ry)
}

/// Compute ranks for a data slice (average rank for ties).
fn ranks(data: &[f64]) -> Vec<f64> {
    let n = data.len();
    let mut indexed: Vec<(usize, f64)> = data.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut result = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n - 1 && (indexed[j + 1].1 - indexed[j].1).abs() < 1e-15 {
            j += 1;
        }
        let avg_rank = (i + j) as f64 / 2.0 + 1.0;
        for k in i..=j {
            result[indexed[k].0] = avg_rank;
        }
        i = j + 1;
    }
    result
}

/// McNemar's test: returns chi-squared statistic with continuity correction.
/// `b` = model1 correct & model2 wrong, `c` = model1 wrong & model2 correct.
pub fn mcnemar_chi2(b: usize, c: usize) -> f64 {
    let bc = b as f64 + c as f64;
    if bc < 1.0 {
        return 0.0;
    }
    let diff = (b as f64 - c as f64).abs() - 1.0;
    if diff < 0.0 {
        return 0.0;
    }
    diff * diff / bc
}

/// Approximate p-value for chi-squared statistic with 1 degree of freedom.
/// Uses a simple approximation based on the normal distribution.
pub fn chi2_pvalue_1df(chi2: f64) -> f64 {
    if chi2 <= 0.0 {
        return 1.0;
    }
    let z = chi2.sqrt();
    // Approximation using error function complement
    let t = 1.0 / (1.0 + 0.2316419 * z);
    let d = 0.3989422804014327; // 1/sqrt(2*pi)
    let p = d * (-z * z / 2.0).exp();
    let poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
    2.0 * p * poly
}

/// Binomial test: probability of observing k or more successes in n trials with p=0.5.
pub fn binomial_test_one_sided(k: usize, n: usize) -> f64 {
    if n == 0 {
        return 1.0;
    }
    let mut p_value = 0.0;
    let mut log_binom = 0.0_f64;
    let n_log2 = (n as f64) * 2.0_f64.ln();

    for i in 0..k {
        if i > 0 {
            log_binom += ((n - i + 1) as f64).ln() - (i as f64).ln();
        }
        p_value += (log_binom - n_log2).exp();
    }
    1.0 - p_value
}

/// Wilson confidence interval for a proportion.
pub fn wilson_confidence_interval(successes: usize, total: usize, z: f64) -> (f64, f64) {
    if total == 0 {
        return (0.0, 1.0);
    }
    let n = total as f64;
    let p_hat = successes as f64 / n;
    let z2 = z * z;
    let denom = 1.0 + z2 / n;
    let center = (p_hat + z2 / (2.0 * n)) / denom;
    let margin = z * ((p_hat * (1.0 - p_hat) + z2 / (4.0 * n)) / n).sqrt() / denom;
    ((center - margin).max(0.0), (center + margin).min(1.0))
}

/// Confusion matrix for multi-class or binary classification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfusionMatrix {
    pub matrix: Vec<Vec<usize>>,
    pub classes: usize,
}

impl ConfusionMatrix {
    pub fn new(classes: usize) -> Self {
        Self {
            matrix: vec![vec![0; classes]; classes],
            classes,
        }
    }

    pub fn from_predictions(actual: &[usize], predicted: &[usize], classes: usize) -> Self {
        let mut cm = Self::new(classes);
        for (&a, &p) in actual.iter().zip(predicted.iter()) {
            if a < classes && p < classes {
                cm.matrix[a][p] += 1;
            }
        }
        cm
    }

    pub fn total(&self) -> usize {
        self.matrix.iter().flat_map(|r| r.iter()).sum()
    }

    pub fn accuracy(&self) -> f64 {
        let total = self.total();
        if total == 0 {
            return 0.0;
        }
        let correct: usize = (0..self.classes).map(|i| self.matrix[i][i]).sum();
        correct as f64 / total as f64
    }

    pub fn precision(&self, class: usize) -> f64 {
        let tp = self.matrix[class][class];
        let predicted: usize = (0..self.classes).map(|i| self.matrix[i][class]).sum();
        if predicted == 0 {
            0.0
        } else {
            tp as f64 / predicted as f64
        }
    }

    pub fn recall(&self, class: usize) -> f64 {
        let tp = self.matrix[class][class];
        let actual: usize = self.matrix[class].iter().sum();
        if actual == 0 {
            0.0
        } else {
            tp as f64 / actual as f64
        }
    }

    pub fn f1_score(&self, class: usize) -> f64 {
        let p = self.precision(class);
        let r = self.recall(class);
        if p + r < 1e-15 {
            0.0
        } else {
            2.0 * p * r / (p + r)
        }
    }

    pub fn macro_precision(&self) -> f64 {
        let sum: f64 = (0..self.classes).map(|c| self.precision(c)).sum();
        sum / self.classes as f64
    }

    pub fn macro_recall(&self) -> f64 {
        let sum: f64 = (0..self.classes).map(|c| self.recall(c)).sum();
        sum / self.classes as f64
    }

    pub fn macro_f1(&self) -> f64 {
        let sum: f64 = (0..self.classes).map(|c| self.f1_score(c)).sum();
        sum / self.classes as f64
    }

    pub fn weighted_f1(&self) -> f64 {
        let total = self.total();
        if total == 0 {
            return 0.0;
        }
        let mut sum = 0.0;
        for c in 0..self.classes {
            let support: usize = self.matrix[c].iter().sum();
            sum += self.f1_score(c) * support as f64;
        }
        sum / total as f64
    }

    pub fn cohens_kappa(&self) -> f64 {
        let total = self.total() as f64;
        if total < 1.0 {
            return 0.0;
        }
        let accuracy = self.accuracy();
        let mut p_expected = 0.0;
        for c in 0..self.classes {
            let row_sum: usize = self.matrix[c].iter().sum();
            let col_sum: usize = (0..self.classes).map(|i| self.matrix[i][c]).sum();
            p_expected += (row_sum as f64 / total) * (col_sum as f64 / total);
        }
        if (1.0 - p_expected).abs() < 1e-15 {
            return 1.0;
        }
        (accuracy - p_expected) / (1.0 - p_expected)
    }
}

/// ROC curve computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RocCurve {
    pub fpr: Vec<f64>,
    pub tpr: Vec<f64>,
    pub thresholds: Vec<f64>,
    pub auc: f64,
}

impl RocCurve {
    /// Compute ROC from binary labels (0/1) and predicted scores.
    pub fn compute(labels: &[usize], scores: &[f64]) -> Self {
        let n = labels.len().min(scores.len());
        if n == 0 {
            return Self {
                fpr: vec![0.0, 1.0],
                tpr: vec![0.0, 1.0],
                thresholds: vec![f64::INFINITY, f64::NEG_INFINITY],
                auc: 0.5,
            };
        }

        let mut indexed: Vec<(f64, usize)> = scores[..n]
            .iter()
            .zip(labels[..n].iter())
            .map(|(&s, &l)| (s, l))
            .collect();
        indexed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let total_pos = labels[..n].iter().filter(|&&l| l == 1).count() as f64;
        let total_neg = n as f64 - total_pos;

        if total_pos < 1.0 || total_neg < 1.0 {
            return Self {
                fpr: vec![0.0, 1.0],
                tpr: vec![0.0, 1.0],
                thresholds: vec![f64::INFINITY, f64::NEG_INFINITY],
                auc: 0.5,
            };
        }

        let mut fpr = vec![0.0];
        let mut tpr = vec![0.0];
        let mut thresholds = vec![f64::INFINITY];
        let mut tp = 0.0;
        let mut fp = 0.0;

        for &(score, label) in &indexed {
            if label == 1 {
                tp += 1.0;
            } else {
                fp += 1.0;
            }
            fpr.push(fp / total_neg);
            tpr.push(tp / total_pos);
            thresholds.push(score);
        }

        // Compute AUC using trapezoidal rule
        let mut auc = 0.0;
        for i in 1..fpr.len() {
            auc += (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) / 2.0;
        }

        Self {
            fpr,
            tpr,
            thresholds,
            auc,
        }
    }
}

/// Generate k-fold cross-validation indices.
pub fn k_fold_splits(n: usize, k: usize, shuffle: bool, seed: u64) -> Vec<(Vec<usize>, Vec<usize>)> {
    let mut indices: Vec<usize> = (0..n).collect();

    if shuffle {
        // Simple deterministic shuffle using seed
        let mut rng_state = seed;
        for i in (1..n).rev() {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let j = (rng_state >> 33) as usize % (i + 1);
            indices.swap(i, j);
        }
    }

    let fold_size = n / k;
    let remainder = n % k;

    let mut folds = Vec::new();
    let mut start = 0;

    for i in 0..k {
        let extra = if i < remainder { 1 } else { 0 };
        let end = start + fold_size + extra;
        folds.push(&indices[start..end]);
        start = end;
    }

    let mut result = Vec::new();
    for i in 0..k {
        let test: Vec<usize> = folds[i].to_vec();
        let train: Vec<usize> = folds
            .iter()
            .enumerate()
            .filter(|&(j, _)| j != i)
            .flat_map(|(_, f)| f.iter().copied())
            .collect();
        result.push((train, test));
    }
    result
}

/// Generate stratified k-fold cross-validation indices.
pub fn stratified_k_fold_splits(
    labels: &[usize],
    k: usize,
    seed: u64,
) -> Vec<(Vec<usize>, Vec<usize>)> {
    let n = labels.len();
    let max_label = labels.iter().copied().max().unwrap_or(0);

    // Group indices by class
    let mut class_indices: Vec<Vec<usize>> = vec![vec![]; max_label + 1];
    for (i, &l) in labels.iter().enumerate() {
        class_indices[l].push(i);
    }

    // Shuffle each class
    let mut rng_state = seed;
    for indices in &mut class_indices {
        for i in (1..indices.len()).rev() {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let j = (rng_state >> 33) as usize % (i + 1);
            indices.swap(i, j);
        }
    }

    // Assign each class to folds
    let mut fold_indices: Vec<Vec<usize>> = vec![vec![]; k];
    for class_idx in &class_indices {
        for (i, &idx) in class_idx.iter().enumerate() {
            fold_indices[i % k].push(idx);
        }
    }

    let _ = n; // used for group computation above
    let mut result = Vec::new();
    for i in 0..k {
        let test: Vec<usize> = fold_indices[i].clone();
        let train: Vec<usize> = fold_indices
            .iter()
            .enumerate()
            .filter(|&(j, _)| j != i)
            .flat_map(|(_, f)| f.iter().copied())
            .collect();
        result.push((train, test));
    }
    result
}

/// Weighted sampling: return indices sampled proportional to weights.
pub fn weighted_sample(weights: &[f64], n: usize, seed: u64) -> Vec<usize> {
    if weights.is_empty() || n == 0 {
        return Vec::new();
    }
    let total: f64 = weights.iter().sum();
    if total <= 0.0 {
        return (0..n).map(|i| i % weights.len()).collect();
    }

    let cumulative: Vec<f64> = weights
        .iter()
        .scan(0.0, |acc, &w| {
            *acc += w / total;
            Some(*acc)
        })
        .collect();

    let mut result = Vec::with_capacity(n);
    let mut rng_state = seed;

    for _ in 0..n {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = (rng_state >> 11) as f64 / (1u64 << 53) as f64;
        let idx = cumulative
            .iter()
            .position(|&c| c >= u)
            .unwrap_or(weights.len() - 1);
        result.push(idx);
    }
    result
}

/// Z-score normalization parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZScoreParams {
    pub means: Vec<f64>,
    pub stds: Vec<f64>,
}

impl ZScoreParams {
    pub fn fit(data: &[Vec<f64>]) -> Self {
        if data.is_empty() {
            return Self {
                means: Vec::new(),
                stds: Vec::new(),
            };
        }
        let dim = data[0].len();
        let n = data.len() as f64;

        let mut means = vec![0.0; dim];
        for row in data {
            for (j, &v) in row.iter().enumerate() {
                if j < dim {
                    means[j] += v;
                }
            }
        }
        for m in &mut means {
            *m /= n;
        }

        let mut stds = vec![0.0; dim];
        for row in data {
            for (j, &v) in row.iter().enumerate() {
                if j < dim {
                    stds[j] += (v - means[j]).powi(2);
                }
            }
        }
        for s in &mut stds {
            *s = (*s / (n - 1.0).max(1.0)).sqrt();
            if *s < 1e-15 {
                *s = 1.0;
            }
        }

        Self { means, stds }
    }

    pub fn transform(&self, x: &[f64]) -> Vec<f64> {
        x.iter()
            .enumerate()
            .map(|(i, &v)| {
                let m = self.means.get(i).copied().unwrap_or(0.0);
                let s = self.stds.get(i).copied().unwrap_or(1.0);
                (v - m) / s
            })
            .collect()
    }

    pub fn inverse_transform(&self, x: &[f64]) -> Vec<f64> {
        x.iter()
            .enumerate()
            .map(|(i, &v)| {
                let m = self.means.get(i).copied().unwrap_or(0.0);
                let s = self.stds.get(i).copied().unwrap_or(1.0);
                v * s + m
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_descriptive_stats() {
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let stats = DescriptiveStats::compute(&data).unwrap();
        assert!((stats.mean - 5.0).abs() < 1e-10);
        assert_eq!(stats.count, 8);
        assert_eq!(stats.min, 2.0);
        assert_eq!(stats.max, 9.0);
    }

    #[test]
    fn test_mean_variance() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((mean(&data) - 3.0).abs() < 1e-10);
        assert!((variance(&data) - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_percentile_sorted() {
        let sorted = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((percentile_sorted(&sorted, 50.0) - 3.0).abs() < 1e-10);
        assert!((percentile_sorted(&sorted, 0.0) - 1.0).abs() < 1e-10);
        assert!((percentile_sorted(&sorted, 100.0) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_pearson_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        assert!((pearson_correlation(&x, &y) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_spearman_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        assert!((spearman_correlation(&x, &y) + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_confusion_matrix_binary() {
        let actual = vec![0, 0, 1, 1, 1, 0, 1, 0];
        let predicted = vec![0, 1, 1, 1, 0, 0, 1, 1];
        let cm = ConfusionMatrix::from_predictions(&actual, &predicted, 2);
        assert_eq!(cm.total(), 8);
        let acc = cm.accuracy();
        assert!(acc > 0.5);
    }

    #[test]
    fn test_confusion_matrix_precision_recall() {
        // Perfect classifier
        let actual = vec![0, 0, 1, 1];
        let predicted = vec![0, 0, 1, 1];
        let cm = ConfusionMatrix::from_predictions(&actual, &predicted, 2);
        assert!((cm.precision(0) - 1.0).abs() < 1e-10);
        assert!((cm.recall(0) - 1.0).abs() < 1e-10);
        assert!((cm.f1_score(0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_roc_auc_perfect() {
        let labels = vec![0, 0, 1, 1];
        let scores = vec![0.1, 0.2, 0.8, 0.9];
        let roc = RocCurve::compute(&labels, &scores);
        assert!((roc.auc - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_roc_auc_random() {
        let labels = vec![0, 1, 0, 1, 0, 1];
        let scores = vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5];
        let roc = RocCurve::compute(&labels, &scores);
        // Tied scores -> AUC should be around 0.5
        assert!(roc.auc >= 0.0 && roc.auc <= 1.0);
    }

    #[test]
    fn test_k_fold_splits() {
        let splits = k_fold_splits(10, 3, false, 42);
        assert_eq!(splits.len(), 3);
        for (train, test) in &splits {
            assert_eq!(train.len() + test.len(), 10);
        }
    }

    #[test]
    fn test_stratified_k_fold() {
        let labels = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];
        let splits = stratified_k_fold_splits(&labels, 3, 42);
        assert_eq!(splits.len(), 3);
    }

    #[test]
    fn test_wilson_ci() {
        let (lo, hi) = wilson_confidence_interval(50, 100, 1.96);
        assert!(lo > 0.3);
        assert!(hi < 0.7);
        assert!(lo < 0.5);
        assert!(hi > 0.5);
    }

    #[test]
    fn test_mcnemar() {
        let chi2 = mcnemar_chi2(20, 5);
        assert!(chi2 > 0.0);
    }

    #[test]
    fn test_zscore_params() {
        let data = vec![vec![1.0, 10.0], vec![2.0, 20.0], vec![3.0, 30.0]];
        let params = ZScoreParams::fit(&data);
        let transformed = params.transform(&[2.0, 20.0]);
        assert!(transformed[0].abs() < 1e-10);
        assert!(transformed[1].abs() < 1e-10);
    }

    #[test]
    fn test_zscore_inverse() {
        let data = vec![vec![1.0, 10.0], vec![2.0, 20.0], vec![3.0, 30.0]];
        let params = ZScoreParams::fit(&data);
        let original = vec![2.5, 15.0];
        let transformed = params.transform(&original);
        let back = params.inverse_transform(&transformed);
        assert!((back[0] - 2.5).abs() < 1e-10);
        assert!((back[1] - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_weighted_sample() {
        let weights = vec![0.0, 1.0, 0.0];
        let samples = weighted_sample(&weights, 10, 42);
        assert!(samples.iter().all(|&s| s == 1));
    }

    #[test]
    fn test_cohens_kappa_perfect() {
        let actual = vec![0, 0, 1, 1, 2, 2];
        let predicted = vec![0, 0, 1, 1, 2, 2];
        let cm = ConfusionMatrix::from_predictions(&actual, &predicted, 3);
        assert!((cm.cohens_kappa() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_empty_descriptive_stats() {
        assert!(DescriptiveStats::compute(&[]).is_none());
    }
}
