//! Statistical and clinical evaluation metrics.
//!
//! Provides confusion-matrix derived measures (sensitivity, specificity, F1,
//! MCC), ROC / PR curves with AUC, performance & scalability metrics, clinical
//! aggregate metrics, bootstrap confidence intervals, the Wilcoxon signed-rank
//! test, and descriptive statistics.

use serde::{Deserialize, Serialize};
use std::fmt;

// ═══════════════════════════════════════════════════════════════════════════
// Confusion Matrix
// ═══════════════════════════════════════════════════════════════════════════

/// Standard binary-classification confusion matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConfusionMatrix {
    /// True positives.
    pub tp: u64,
    /// False positives.
    pub fp: u64,
    /// False negatives.
    pub fn_: u64,
    /// True negatives.
    pub tn: u64,
}

impl ConfusionMatrix {
    /// Create a new confusion matrix.
    pub fn new(tp: u64, fp: u64, fn_: u64, tn: u64) -> Self {
        Self { tp, fp, fn_: fn_, tn }
    }

    /// Build from a list of (predicted_positive, actual_positive) pairs.
    pub fn from_predictions(pairs: &[(bool, bool)]) -> Self {
        let mut tp = 0u64;
        let mut fp = 0u64;
        let mut fn_ = 0u64;
        let mut tn = 0u64;
        for &(predicted, actual) in pairs {
            match (predicted, actual) {
                (true, true) => tp += 1,
                (true, false) => fp += 1,
                (false, true) => fn_ += 1,
                (false, false) => tn += 1,
            }
        }
        Self { tp, fp, fn_: fn_, tn }
    }

    /// Total number of samples.
    pub fn total(&self) -> u64 {
        self.tp + self.fp + self.fn_ + self.tn
    }

    /// Sensitivity (recall, true-positive rate).
    pub fn sensitivity(&self) -> f64 {
        let denom = self.tp + self.fn_;
        if denom == 0 { 0.0 } else { self.tp as f64 / denom as f64 }
    }

    /// Specificity (true-negative rate).
    pub fn specificity(&self) -> f64 {
        let denom = self.tn + self.fp;
        if denom == 0 { 0.0 } else { self.tn as f64 / denom as f64 }
    }

    /// Positive predictive value (precision).
    pub fn ppv(&self) -> f64 {
        let denom = self.tp + self.fp;
        if denom == 0 { 0.0 } else { self.tp as f64 / denom as f64 }
    }

    /// Negative predictive value.
    pub fn npv(&self) -> f64 {
        let denom = self.tn + self.fn_;
        if denom == 0 { 0.0 } else { self.tn as f64 / denom as f64 }
    }

    /// Accuracy.
    pub fn accuracy(&self) -> f64 {
        let t = self.total();
        if t == 0 { 0.0 } else { (self.tp + self.tn) as f64 / t as f64 }
    }

    /// F1 score (harmonic mean of precision and recall).
    pub fn f1_score(&self) -> f64 {
        let p = self.ppv();
        let r = self.sensitivity();
        if p + r == 0.0 { 0.0 } else { 2.0 * p * r / (p + r) }
    }

    /// F-beta score with configurable beta.
    pub fn fbeta_score(&self, beta: f64) -> f64 {
        let p = self.ppv();
        let r = self.sensitivity();
        let b2 = beta * beta;
        let denom = b2 * p + r;
        if denom == 0.0 { 0.0 } else { (1.0 + b2) * p * r / denom }
    }

    /// Matthews Correlation Coefficient.
    pub fn mcc(&self) -> f64 {
        let tp = self.tp as f64;
        let fp = self.fp as f64;
        let fn_ = self.fn_ as f64;
        let tn = self.tn as f64;
        let numerator = tp * tn - fp * fn_;
        let denominator = ((tp + fp) * (tp + fn_) * (tn + fp) * (tn + fn_)).sqrt();
        if denominator == 0.0 { 0.0 } else { numerator / denominator }
    }

    /// False positive rate.
    pub fn fpr(&self) -> f64 {
        1.0 - self.specificity()
    }

    /// False negative rate.
    pub fn fnr(&self) -> f64 {
        1.0 - self.sensitivity()
    }

    /// Balanced accuracy (average of sensitivity and specificity).
    pub fn balanced_accuracy(&self) -> f64 {
        (self.sensitivity() + self.specificity()) / 2.0
    }

    /// Prevalence (fraction of actual positives).
    pub fn prevalence(&self) -> f64 {
        let t = self.total();
        if t == 0 { 0.0 } else { (self.tp + self.fn_) as f64 / t as f64 }
    }

    /// Positive likelihood ratio.
    pub fn positive_likelihood_ratio(&self) -> f64 {
        let fpr = self.fpr();
        if fpr == 0.0 { f64::INFINITY } else { self.sensitivity() / fpr }
    }

    /// Negative likelihood ratio.
    pub fn negative_likelihood_ratio(&self) -> f64 {
        let spec = self.specificity();
        if spec == 0.0 { f64::INFINITY } else { self.fnr() / spec }
    }

    /// Diagnostic odds ratio.
    pub fn diagnostic_odds_ratio(&self) -> f64 {
        let nlr = self.negative_likelihood_ratio();
        if nlr == 0.0 { f64::INFINITY } else { self.positive_likelihood_ratio() / nlr }
    }

    /// Youden's J statistic (informedness).
    pub fn youdens_j(&self) -> f64 {
        self.sensitivity() + self.specificity() - 1.0
    }

    /// Merge two confusion matrices (element-wise sum).
    pub fn merge(&self, other: &ConfusionMatrix) -> ConfusionMatrix {
        ConfusionMatrix {
            tp: self.tp + other.tp,
            fp: self.fp + other.fp,
            fn_: self.fn_ + other.fn_,
            tn: self.tn + other.tn,
        }
    }
}

impl Default for ConfusionMatrix {
    fn default() -> Self {
        Self { tp: 0, fp: 0, fn_: 0, tn: 0 }
    }
}

impl fmt::Display for ConfusionMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CM(TP={}, FP={}, FN={}, TN={} | Acc={:.3}, F1={:.3}, MCC={:.3})",
            self.tp, self.fp, self.fn_, self.tn,
            self.accuracy(), self.f1_score(), self.mcc()
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ROC Curve
// ═══════════════════════════════════════════════════════════════════════════

/// A single point on an ROC curve.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct RocPoint {
    /// False positive rate.
    pub fpr: f64,
    /// True positive rate (sensitivity).
    pub tpr: f64,
    /// Threshold used.
    pub threshold: f64,
}

/// Receiver Operating Characteristic curve.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RocCurve {
    /// Ordered (by decreasing threshold) ROC points.
    pub points: Vec<RocPoint>,
}

impl RocCurve {
    /// Number of points.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Whether the curve is empty.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Area Under the ROC Curve (trapezoidal).
    pub fn auc(&self) -> f64 {
        compute_auc(self)
    }

    /// Find the threshold that maximises Youden's J.
    pub fn optimal_threshold_youden(&self) -> f64 {
        self.points
            .iter()
            .max_by(|a, b| {
                let ja = a.tpr - a.fpr;
                let jb = b.tpr - b.fpr;
                ja.partial_cmp(&jb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|p| p.threshold)
            .unwrap_or(0.5)
    }

    /// Find the threshold nearest a target sensitivity.
    pub fn threshold_at_sensitivity(&self, target_sens: f64) -> f64 {
        self.points
            .iter()
            .filter(|p| p.tpr >= target_sens)
            .min_by(|a, b| {
                (a.tpr - target_sens)
                    .abs()
                    .partial_cmp(&(b.tpr - target_sens).abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|p| p.threshold)
            .unwrap_or(0.5)
    }
}

/// Compute an ROC curve from (score, actual_positive) pairs.
///
/// Scores are assumed to be such that higher means "more likely positive".
pub fn compute_roc(predictions: &[(f64, bool)]) -> RocCurve {
    if predictions.is_empty() {
        return RocCurve { points: vec![] };
    }

    let mut sorted: Vec<(f64, bool)> = predictions.to_vec();
    sorted.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let total_pos = sorted.iter().filter(|(_, b)| *b).count() as f64;
    let total_neg = sorted.iter().filter(|(_, b)| !*b).count() as f64;

    if total_pos == 0.0 || total_neg == 0.0 {
        return RocCurve {
            points: vec![
                RocPoint { fpr: 0.0, tpr: 0.0, threshold: f64::INFINITY },
                RocPoint { fpr: 1.0, tpr: 1.0, threshold: f64::NEG_INFINITY },
            ],
        };
    }

    let mut points = Vec::with_capacity(sorted.len() + 2);
    points.push(RocPoint { fpr: 0.0, tpr: 0.0, threshold: f64::INFINITY });

    let mut tp = 0.0_f64;
    let mut fp = 0.0_f64;

    let mut prev_score = f64::INFINITY;
    for &(score, label) in &sorted {
        if (score - prev_score).abs() > f64::EPSILON {
            points.push(RocPoint {
                fpr: fp / total_neg,
                tpr: tp / total_pos,
                threshold: prev_score,
            });
            prev_score = score;
        }
        if label {
            tp += 1.0;
        } else {
            fp += 1.0;
        }
    }
    points.push(RocPoint {
        fpr: fp / total_neg,
        tpr: tp / total_pos,
        threshold: prev_score,
    });

    // Ensure endpoint (1, 1).
    if let Some(last) = points.last() {
        if (last.fpr - 1.0).abs() > f64::EPSILON || (last.tpr - 1.0).abs() > f64::EPSILON {
            points.push(RocPoint { fpr: 1.0, tpr: 1.0, threshold: f64::NEG_INFINITY });
        }
    }

    RocCurve { points }
}

/// Compute area under an ROC curve via the trapezoidal rule.
pub fn compute_auc(roc: &RocCurve) -> f64 {
    if roc.points.len() < 2 {
        return 0.0;
    }
    let mut area = 0.0;
    for w in roc.points.windows(2) {
        let dx = w[1].fpr - w[0].fpr;
        let avg_y = (w[0].tpr + w[1].tpr) / 2.0;
        area += dx * avg_y;
    }
    area
}

// ═══════════════════════════════════════════════════════════════════════════
// Precision–Recall Curve
// ═══════════════════════════════════════════════════════════════════════════

/// A single point on a precision–recall curve.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PrPoint {
    pub recall: f64,
    pub precision: f64,
    pub threshold: f64,
}

/// Precision–Recall curve.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecisionRecallCurve {
    pub points: Vec<PrPoint>,
}

impl PrecisionRecallCurve {
    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Average precision (AP) — area under the PR curve using interpolation.
    pub fn average_precision(&self) -> f64 {
        if self.points.len() < 2 {
            return 0.0;
        }
        let mut ap = 0.0;
        for w in self.points.windows(2) {
            let d_recall = w[1].recall - w[0].recall;
            ap += d_recall * w[1].precision;
        }
        ap
    }

    /// Maximum F1 score on the curve.
    pub fn max_f1(&self) -> f64 {
        self.points
            .iter()
            .map(|p| {
                let denom = p.precision + p.recall;
                if denom == 0.0 { 0.0 } else { 2.0 * p.precision * p.recall / denom }
            })
            .fold(0.0_f64, f64::max)
    }
}

/// Compute a precision–recall curve from (score, actual_positive) pairs.
pub fn compute_pr_curve(predictions: &[(f64, bool)]) -> PrecisionRecallCurve {
    if predictions.is_empty() {
        return PrecisionRecallCurve { points: vec![] };
    }

    let mut sorted: Vec<(f64, bool)> = predictions.to_vec();
    sorted.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let total_pos = sorted.iter().filter(|(_, b)| *b).count() as f64;
    if total_pos == 0.0 {
        return PrecisionRecallCurve { points: vec![] };
    }

    let mut points = Vec::with_capacity(sorted.len() + 1);
    points.push(PrPoint { recall: 0.0, precision: 1.0, threshold: f64::INFINITY });

    let mut tp = 0.0_f64;
    let mut fp = 0.0_f64;

    for &(score, label) in &sorted {
        if label {
            tp += 1.0;
        } else {
            fp += 1.0;
        }
        let precision = tp / (tp + fp);
        let recall = tp / total_pos;
        points.push(PrPoint { recall, precision, threshold: score });
    }

    PrecisionRecallCurve { points }
}

// ═══════════════════════════════════════════════════════════════════════════
// Descriptive Statistics
// ═══════════════════════════════════════════════════════════════════════════

/// Summary descriptive statistics for a numeric sample.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DescriptiveStats {
    pub count: usize,
    pub mean: f64,
    pub median: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub q1: f64,
    pub q3: f64,
    pub iqr: f64,
    pub variance: f64,
    pub skewness: f64,
    pub kurtosis: f64,
}

impl Default for DescriptiveStats {
    fn default() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            median: 0.0,
            std_dev: 0.0,
            min: 0.0,
            max: 0.0,
            q1: 0.0,
            q3: 0.0,
            iqr: 0.0,
            variance: 0.0,
            skewness: 0.0,
            kurtosis: 0.0,
        }
    }
}

impl fmt::Display for DescriptiveStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "n={}, mean={:.4}, median={:.4}, sd={:.4}, [{:.4}, {:.4}]",
            self.count, self.mean, self.median, self.std_dev, self.min, self.max,
        )
    }
}

/// Compute the percentile of a sorted slice using linear interpolation.
fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let idx = p / 100.0 * (sorted.len() - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    let frac = idx - lo as f64;
    if hi >= sorted.len() {
        sorted[sorted.len() - 1]
    } else {
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

/// Compute descriptive statistics for the given data.
pub fn compute_descriptive(data: &[f64]) -> DescriptiveStats {
    if data.is_empty() {
        return DescriptiveStats::default();
    }

    let n = data.len();
    let sum: f64 = data.iter().sum();
    let mean = sum / n as f64;

    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let min = sorted[0];
    let max = sorted[n - 1];
    let median = percentile_sorted(&sorted, 50.0);
    let q1 = percentile_sorted(&sorted, 25.0);
    let q3 = percentile_sorted(&sorted, 75.0);
    let iqr = q3 - q1;

    let variance = if n > 1 {
        let ss: f64 = data.iter().map(|x| (x - mean).powi(2)).sum();
        ss / (n - 1) as f64
    } else {
        0.0
    };
    let std_dev = variance.sqrt();

    let skewness = if n > 2 && std_dev > 0.0 {
        let m3: f64 = data.iter().map(|x| ((x - mean) / std_dev).powi(3)).sum();
        m3 / n as f64
    } else {
        0.0
    };

    let kurtosis = if n > 3 && std_dev > 0.0 {
        let m4: f64 = data.iter().map(|x| ((x - mean) / std_dev).powi(4)).sum();
        m4 / n as f64 - 3.0
    } else {
        0.0
    };

    DescriptiveStats {
        count: n,
        mean,
        median,
        std_dev,
        min,
        max,
        q1,
        q3,
        iqr,
        variance,
        skewness,
        kurtosis,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Performance Metrics
// ═══════════════════════════════════════════════════════════════════════════

/// Verification performance metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Average wall-clock time per guideline (ms).
    pub time_per_guideline_ms: f64,
    /// Average wall-clock time per drug pair (ms).
    pub time_per_drug_pair_ms: f64,
    /// Average memory consumption per guideline (bytes).
    pub memory_per_guideline_bytes: f64,
    /// Peak memory usage (bytes).
    pub peak_memory_bytes: u64,
    /// Total verification time (ms).
    pub total_time_ms: f64,
    /// Tier-1 time breakdown (ms).
    pub tier1_time_ms: f64,
    /// Tier-2 time breakdown (ms).
    pub tier2_time_ms: f64,
    /// Fraction of cases escalated from Tier 1 to Tier 2.
    pub escalation_rate: f64,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            time_per_guideline_ms: 0.0,
            time_per_drug_pair_ms: 0.0,
            memory_per_guideline_bytes: 0.0,
            peak_memory_bytes: 0,
            total_time_ms: 0.0,
            tier1_time_ms: 0.0,
            tier2_time_ms: 0.0,
            escalation_rate: 0.0,
        }
    }
}

impl PerformanceMetrics {
    /// Compute performance metrics from a sequence of per-benchmark timings.
    pub fn from_timings(
        timings: &[(f64, f64, u64)],
        n_guidelines: usize,
        n_drug_pairs: usize,
    ) -> Self {
        if timings.is_empty() {
            return Self::default();
        }
        let total_tier1: f64 = timings.iter().map(|t| t.0).sum();
        let total_tier2: f64 = timings.iter().map(|t| t.1).sum();
        let peak_mem = timings.iter().map(|t| t.2).max().unwrap_or(0);
        let total = total_tier1 + total_tier2;
        let avg_mem: f64 = timings.iter().map(|t| t.2 as f64).sum::<f64>() / timings.len() as f64;
        let escalation_count = timings.iter().filter(|t| t.1 > 0.0).count();

        Self {
            time_per_guideline_ms: if n_guidelines > 0 { total / n_guidelines as f64 } else { 0.0 },
            time_per_drug_pair_ms: if n_drug_pairs > 0 { total / n_drug_pairs as f64 } else { 0.0 },
            memory_per_guideline_bytes: if n_guidelines > 0 { avg_mem / n_guidelines as f64 } else { 0.0 },
            peak_memory_bytes: peak_mem,
            total_time_ms: total,
            tier1_time_ms: total_tier1,
            tier2_time_ms: total_tier2,
            escalation_rate: escalation_count as f64 / timings.len() as f64,
        }
    }
}

impl fmt::Display for PerformanceMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "total={:.1}ms (T1={:.1}ms, T2={:.1}ms), esc={:.1}%",
            self.total_time_ms, self.tier1_time_ms, self.tier2_time_ms,
            self.escalation_rate * 100.0,
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Scalability Metrics
// ═══════════════════════════════════════════════════════════════════════════

/// Empirical complexity classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComplexityClass {
    Constant,
    Logarithmic,
    Linear,
    Linearithmic,
    Quadratic,
    Cubic,
    Exponential,
}

impl fmt::Display for ComplexityClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Constant => "O(1)",
            Self::Logarithmic => "O(log n)",
            Self::Linear => "O(n)",
            Self::Linearithmic => "O(n log n)",
            Self::Quadratic => "O(n²)",
            Self::Cubic => "O(n³)",
            Self::Exponential => "O(2ⁿ)",
        };
        write!(f, "{}", s)
    }
}

/// Scalability measurement: (input_size, time_ms) pairs with fitted complexity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalabilityMetrics {
    /// Data points: (input size, elapsed time in ms).
    pub times: Vec<(usize, f64)>,
    /// R² of the best fit.
    pub r_squared: f64,
    /// Fitted complexity class.
    pub complexity: ComplexityClass,
    /// Slope coefficient of the best linear fit (on transformed data).
    pub slope: f64,
    /// Intercept of the best linear fit.
    pub intercept: f64,
}

impl ScalabilityMetrics {
    /// Create from raw (size, time) measurements and fit the complexity.
    pub fn from_measurements(times: Vec<(usize, f64)>) -> Self {
        let (complexity, r_squared, slope, intercept) = fit_complexity(&times);
        Self { times, r_squared, complexity, slope, intercept }
    }

    /// Predict time for a given input size using the fitted model.
    pub fn predict(&self, n: usize) -> f64 {
        let x = n as f64;
        match self.complexity {
            ComplexityClass::Constant => self.intercept,
            ComplexityClass::Logarithmic => self.slope * x.ln().max(1.0) + self.intercept,
            ComplexityClass::Linear => self.slope * x + self.intercept,
            ComplexityClass::Linearithmic => self.slope * x * x.ln().max(1.0) + self.intercept,
            ComplexityClass::Quadratic => self.slope * x * x + self.intercept,
            ComplexityClass::Cubic => self.slope * x * x * x + self.intercept,
            ComplexityClass::Exponential => self.intercept * (self.slope).powf(x),
        }
    }

    /// Whether the growth is considered acceptable (at most quadratic).
    pub fn is_acceptable(&self) -> bool {
        matches!(
            self.complexity,
            ComplexityClass::Constant
                | ComplexityClass::Logarithmic
                | ComplexityClass::Linear
                | ComplexityClass::Linearithmic
                | ComplexityClass::Quadratic
        )
    }

    /// Speedup ratio between first and last measurement normalised by size.
    pub fn per_unit_speedup(&self) -> f64 {
        if self.times.len() < 2 {
            return 1.0;
        }
        let first = &self.times[0];
        let last = &self.times[self.times.len() - 1];
        if last.0 == 0 || first.0 == 0 {
            return 1.0;
        }
        let per_unit_first = first.1 / first.0 as f64;
        let per_unit_last = last.1 / last.0 as f64;
        if per_unit_last == 0.0 { f64::INFINITY } else { per_unit_first / per_unit_last }
    }
}

/// Simple OLS on x,y data, returning (slope, intercept, r²).
fn simple_ols(xs: &[f64], ys: &[f64]) -> (f64, f64, f64) {
    let n = xs.len() as f64;
    if n < 2.0 {
        return (0.0, ys.first().copied().unwrap_or(0.0), 0.0);
    }
    let sx: f64 = xs.iter().sum();
    let sy: f64 = ys.iter().sum();
    let sxx: f64 = xs.iter().map(|x| x * x).sum();
    let sxy: f64 = xs.iter().zip(ys.iter()).map(|(x, y)| x * y).sum();

    let denom = n * sxx - sx * sx;
    if denom.abs() < f64::EPSILON {
        return (0.0, sy / n, 0.0);
    }
    let slope = (n * sxy - sx * sy) / denom;
    let intercept = (sy - slope * sx) / n;

    let y_mean = sy / n;
    let ss_tot: f64 = ys.iter().map(|y| (y - y_mean).powi(2)).sum();
    let ss_res: f64 = xs
        .iter()
        .zip(ys.iter())
        .map(|(x, y)| {
            let pred = slope * x + intercept;
            (y - pred).powi(2)
        })
        .sum();
    let r2 = if ss_tot > 0.0 { 1.0 - ss_res / ss_tot } else { 0.0 };
    (slope, intercept, r2)
}

/// Fit the best complexity class to (size, time) data.
fn fit_complexity(data: &[(usize, f64)]) -> (ComplexityClass, f64, f64, f64) {
    if data.len() < 2 {
        return (ComplexityClass::Constant, 1.0, 0.0, data.first().map(|d| d.1).unwrap_or(0.0));
    }

    let ns: Vec<f64> = data.iter().map(|d| d.0 as f64).collect();
    let ts: Vec<f64> = data.iter().map(|d| d.1).collect();

    // Try several transformations and pick best R².
    struct Candidate {
        class: ComplexityClass,
        xs: Vec<f64>,
    }

    let candidates = vec![
        Candidate {
            class: ComplexityClass::Linear,
            xs: ns.clone(),
        },
        Candidate {
            class: ComplexityClass::Logarithmic,
            xs: ns.iter().map(|n| n.max(1.0).ln()).collect(),
        },
        Candidate {
            class: ComplexityClass::Linearithmic,
            xs: ns.iter().map(|n| n * n.max(1.0).ln()).collect(),
        },
        Candidate {
            class: ComplexityClass::Quadratic,
            xs: ns.iter().map(|n| n * n).collect(),
        },
        Candidate {
            class: ComplexityClass::Cubic,
            xs: ns.iter().map(|n| n * n * n).collect(),
        },
    ];

    let mut best_class = ComplexityClass::Linear;
    let mut best_r2 = f64::NEG_INFINITY;
    let mut best_slope = 0.0;
    let mut best_intercept = 0.0;

    for c in &candidates {
        let (slope, intercept, r2) = simple_ols(&c.xs, &ts);
        if r2 > best_r2 {
            best_r2 = r2;
            best_class = c.class;
            best_slope = slope;
            best_intercept = intercept;
        }
    }

    // Check for constant.
    let t_mean = ts.iter().sum::<f64>() / ts.len() as f64;
    let max_dev = ts.iter().map(|t| (t - t_mean).abs()).fold(0.0_f64, f64::max);
    if max_dev < t_mean * 0.1 + 1.0 {
        return (ComplexityClass::Constant, 1.0, 0.0, t_mean);
    }

    (best_class, best_r2, best_slope, best_intercept)
}

// ═══════════════════════════════════════════════════════════════════════════
// Clinical Metrics
// ═══════════════════════════════════════════════════════════════════════════

/// Aggregate clinical outcome metrics.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ClinicalMetrics {
    /// Average number of conflicts per patient.
    pub conflicts_per_patient: f64,
    /// Average number of critical conflicts per patient.
    pub critical_per_patient: f64,
    /// Mean severity score (1=Minor, 2=Moderate, 3=Major, 4=Contraindicated).
    pub avg_severity: f64,
    /// Fraction of patients with at least one conflict.
    pub patients_with_conflicts: f64,
    /// Fraction of patients with at least one critical conflict.
    pub patients_with_critical: f64,
    /// Total number of unique drug pairs with interactions.
    pub unique_interacting_pairs: usize,
    /// Average number of medications per patient.
    pub avg_medications: f64,
}

impl Default for ClinicalMetrics {
    fn default() -> Self {
        Self {
            conflicts_per_patient: 0.0,
            critical_per_patient: 0.0,
            avg_severity: 0.0,
            patients_with_conflicts: 0.0,
            patients_with_critical: 0.0,
            unique_interacting_pairs: 0,
            avg_medications: 0.0,
        }
    }
}

impl ClinicalMetrics {
    /// Build from per-patient conflict counts.
    ///
    /// Each entry is (total_conflicts, critical_conflicts, max_severity, n_medications).
    pub fn from_patient_data(data: &[(usize, usize, f64, usize)]) -> Self {
        if data.is_empty() {
            return Self::default();
        }
        let n = data.len() as f64;
        let total_conflicts: usize = data.iter().map(|d| d.0).sum();
        let total_critical: usize = data.iter().map(|d| d.1).sum();
        let total_sev: f64 = data.iter().map(|d| d.2).sum();
        let with_conf = data.iter().filter(|d| d.0 > 0).count();
        let with_crit = data.iter().filter(|d| d.1 > 0).count();
        let total_meds: usize = data.iter().map(|d| d.3).sum();

        let count_with_severity = data.iter().filter(|d| d.0 > 0).count();

        Self {
            conflicts_per_patient: total_conflicts as f64 / n,
            critical_per_patient: total_critical as f64 / n,
            avg_severity: if count_with_severity > 0 {
                total_sev / count_with_severity as f64
            } else {
                0.0
            },
            patients_with_conflicts: with_conf as f64 / n,
            patients_with_critical: with_crit as f64 / n,
            unique_interacting_pairs: total_conflicts,
            avg_medications: total_meds as f64 / n,
        }
    }
}

impl fmt::Display for ClinicalMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "conflicts/pt={:.2}, critical/pt={:.2}, avg_sev={:.2}, meds/pt={:.1}",
            self.conflicts_per_patient,
            self.critical_per_patient,
            self.avg_severity,
            self.avg_medications,
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Bootstrap Confidence Intervals
// ═══════════════════════════════════════════════════════════════════════════

/// Bootstrap confidence interval calculator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrapCI {
    /// Lower bound of the CI.
    pub lower: f64,
    /// Upper bound of the CI.
    pub upper: f64,
    /// Point estimate (original statistic).
    pub point_estimate: f64,
    /// Confidence level (e.g. 0.95).
    pub confidence: f64,
    /// Number of bootstrap samples used.
    pub n_bootstrap: usize,
    /// Standard error of bootstrap distribution.
    pub bootstrap_se: f64,
}

impl fmt::Display for BootstrapCI {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:.4} [{:.4}, {:.4}] ({:.0}% CI, n_boot={})",
            self.point_estimate,
            self.lower,
            self.upper,
            self.confidence * 100.0,
            self.n_bootstrap,
        )
    }
}

/// Simple LCG-based deterministic PRNG for bootstrapping.
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed.wrapping_add(1) }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state
    }

    fn next_usize(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }
}

/// Compute a bootstrap confidence interval for the **mean**.
pub fn bootstrap_mean_ci(data: &[f64], confidence: f64, n_bootstrap: usize) -> BootstrapCI {
    bootstrap_ci(data, confidence, n_bootstrap, |sample| {
        sample.iter().sum::<f64>() / sample.len() as f64
    })
}

/// Compute a bootstrap confidence interval for the **median**.
pub fn bootstrap_median_ci(data: &[f64], confidence: f64, n_bootstrap: usize) -> BootstrapCI {
    bootstrap_ci(data, confidence, n_bootstrap, |sample| {
        let mut s = sample.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        percentile_sorted(&s, 50.0)
    })
}

/// Generic bootstrap CI computation.
pub fn bootstrap_ci<F>(data: &[f64], confidence: f64, n_bootstrap: usize, statistic: F) -> BootstrapCI
where
    F: Fn(&[f64]) -> f64,
{
    if data.is_empty() {
        return BootstrapCI {
            lower: 0.0,
            upper: 0.0,
            point_estimate: 0.0,
            confidence,
            n_bootstrap,
            bootstrap_se: 0.0,
        };
    }

    let point_estimate = statistic(data);
    let mut rng = SimpleRng::new(42);
    let n = data.len();

    let mut estimates = Vec::with_capacity(n_bootstrap);
    let mut resample = vec![0.0; n];

    for _ in 0..n_bootstrap {
        for slot in resample.iter_mut() {
            *slot = data[rng.next_usize(n)];
        }
        estimates.push(statistic(&resample));
    }

    estimates.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let alpha = 1.0 - confidence;
    let lower_idx = ((alpha / 2.0) * n_bootstrap as f64).floor() as usize;
    let upper_idx = ((1.0 - alpha / 2.0) * n_bootstrap as f64).ceil() as usize;
    let lower = estimates[lower_idx.min(estimates.len() - 1)];
    let upper = estimates[upper_idx.min(estimates.len() - 1)];

    let boot_mean: f64 = estimates.iter().sum::<f64>() / estimates.len() as f64;
    let bootstrap_se = (estimates
        .iter()
        .map(|e| (e - boot_mean).powi(2))
        .sum::<f64>()
        / (estimates.len() - 1).max(1) as f64)
        .sqrt();

    BootstrapCI {
        lower,
        upper,
        point_estimate,
        confidence,
        n_bootstrap,
        bootstrap_se,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Wilcoxon Signed-Rank Test
// ═══════════════════════════════════════════════════════════════════════════

/// Result of a Wilcoxon signed-rank test.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct WilcoxonResult {
    /// The W+ statistic (sum of positive ranks).
    pub w_plus: f64,
    /// The W- statistic (sum of negative ranks).
    pub w_minus: f64,
    /// The test statistic T = min(W+, W-).
    pub test_statistic: f64,
    /// Approximate z-score (normal approximation).
    pub z_score: f64,
    /// Approximate two-tailed p-value.
    pub p_value: f64,
    /// Number of non-zero differences.
    pub n_effective: usize,
    /// Whether the null hypothesis is rejected at α = 0.05.
    pub significant_005: bool,
}

/// Wilcoxon signed-rank test for paired observations.
pub struct WilcoxonSignedRank;

impl WilcoxonSignedRank {
    /// Run the Wilcoxon signed-rank test on paired data.
    ///
    /// Tests whether the median difference between pairs is zero.
    pub fn test(a: &[f64], b: &[f64]) -> WilcoxonResult {
        assert_eq!(a.len(), b.len(), "Paired data must have equal length");

        let diffs: Vec<f64> = a.iter().zip(b.iter()).map(|(x, y)| x - y).collect();
        Self::test_differences(&diffs)
    }

    /// Run on pre-computed differences.
    pub fn test_differences(diffs: &[f64]) -> WilcoxonResult {
        // Remove zeros.
        let nonzero: Vec<f64> = diffs.iter().copied().filter(|d| d.abs() > f64::EPSILON).collect();
        let n = nonzero.len();

        if n == 0 {
            return WilcoxonResult {
                w_plus: 0.0,
                w_minus: 0.0,
                test_statistic: 0.0,
                z_score: 0.0,
                p_value: 1.0,
                n_effective: 0,
                significant_005: false,
            };
        }

        // Rank by absolute value.
        let mut indexed: Vec<(usize, f64)> = nonzero.iter().enumerate().map(|(i, &d)| (i, d.abs())).collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Assign ranks with tie averaging.
        let mut ranks = vec![0.0_f64; n];
        let mut i = 0;
        while i < n {
            let mut j = i;
            while j < n && (indexed[j].1 - indexed[i].1).abs() < f64::EPSILON {
                j += 1;
            }
            let avg_rank = (i + j + 1) as f64 / 2.0; // 1-based average
            for k in i..j {
                ranks[indexed[k].0] = avg_rank;
            }
            i = j;
        }

        let mut w_plus = 0.0_f64;
        let mut w_minus = 0.0_f64;
        for (idx, &d) in nonzero.iter().enumerate() {
            if d > 0.0 {
                w_plus += ranks[idx];
            } else {
                w_minus += ranks[idx];
            }
        }

        let test_statistic = w_plus.min(w_minus);

        // Normal approximation.
        let nf = n as f64;
        let mean_w = nf * (nf + 1.0) / 4.0;
        let var_w = nf * (nf + 1.0) * (2.0 * nf + 1.0) / 24.0;
        let std_w = var_w.sqrt();

        let z_score = if std_w > 0.0 {
            (test_statistic - mean_w).abs() / std_w
        } else {
            0.0
        };

        // Approximate p-value using Φ(-z) * 2 with Abramowitz-Stegun.
        let p_value = 2.0 * normal_cdf_complement(z_score);

        WilcoxonResult {
            w_plus,
            w_minus,
            test_statistic,
            z_score,
            p_value,
            n_effective: n,
            significant_005: p_value < 0.05,
        }
    }
}

/// Complement of the standard normal CDF: P(Z > z).
fn normal_cdf_complement(z: f64) -> f64 {
    // Abramowitz & Stegun approximation 26.2.17 (max error 7.5e-8).
    let z = z.abs();
    let p = 0.2316419;
    let b1 = 0.319381530;
    let b2 = -0.356563782;
    let b3 = 1.781477937;
    let b4 = -1.821255978;
    let b5 = 1.330274429;

    let t = 1.0 / (1.0 + p * z);
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t3 * t;
    let t5 = t4 * t;

    let phi = (-z * z / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt();
    phi * (b1 * t + b2 * t2 + b3 * t3 + b4 * t4 + b5 * t5)
}

/// Cohen's d effect size for two independent samples.
pub fn cohens_d(a: &[f64], b: &[f64]) -> f64 {
    let stats_a = compute_descriptive(a);
    let stats_b = compute_descriptive(b);
    let pooled_var = ((stats_a.count as f64 - 1.0) * stats_a.variance
        + (stats_b.count as f64 - 1.0) * stats_b.variance)
        / (stats_a.count as f64 + stats_b.count as f64 - 2.0);
    let pooled_sd = pooled_var.sqrt();
    if pooled_sd == 0.0 {
        0.0
    } else {
        (stats_a.mean - stats_b.mean).abs() / pooled_sd
    }
}

/// Pearson correlation coefficient.
pub fn pearson_r(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    let n = x.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let x_mean = x.iter().sum::<f64>() / n;
    let y_mean = y.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let dx = xi - x_mean;
        let dy = yi - y_mean;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    let denom = (var_x * var_y).sqrt();
    if denom == 0.0 { 0.0 } else { cov / denom }
}

/// Spearman rank correlation coefficient.
pub fn spearman_rho(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    let rank_x = rank_data(x);
    let rank_y = rank_data(y);
    pearson_r(&rank_x, &rank_y)
}

/// Assign ranks to data (average rank for ties).
fn rank_data(data: &[f64]) -> Vec<f64> {
    let n = data.len();
    let mut indexed: Vec<(usize, f64)> = data.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && (indexed[j].1 - indexed[i].1).abs() < f64::EPSILON {
            j += 1;
        }
        let avg = (i + j + 1) as f64 / 2.0;
        for k in i..j {
            ranks[indexed[k].0] = avg;
        }
        i = j;
    }
    ranks
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_confusion_matrix_perfect() {
        let cm = ConfusionMatrix::new(50, 0, 0, 50);
        assert!((cm.accuracy() - 1.0).abs() < 1e-10);
        assert!((cm.sensitivity() - 1.0).abs() < 1e-10);
        assert!((cm.specificity() - 1.0).abs() < 1e-10);
        assert!((cm.f1_score() - 1.0).abs() < 1e-10);
        assert!((cm.mcc() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_confusion_matrix_from_predictions() {
        let pairs = vec![
            (true, true),
            (true, false),
            (false, true),
            (false, false),
        ];
        let cm = ConfusionMatrix::from_predictions(&pairs);
        assert_eq!(cm.tp, 1);
        assert_eq!(cm.fp, 1);
        assert_eq!(cm.fn_, 1);
        assert_eq!(cm.tn, 1);
        assert!((cm.accuracy() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_confusion_matrix_merge() {
        let a = ConfusionMatrix::new(10, 2, 3, 85);
        let b = ConfusionMatrix::new(5, 1, 2, 42);
        let c = a.merge(&b);
        assert_eq!(c.tp, 15);
        assert_eq!(c.fp, 3);
        assert_eq!(c.fn_, 5);
        assert_eq!(c.tn, 127);
    }

    #[test]
    fn test_confusion_matrix_youdens_j() {
        let cm = ConfusionMatrix::new(90, 10, 5, 95);
        let j = cm.youdens_j();
        assert!(j > 0.8);
    }

    #[test]
    fn test_roc_curve_perfect() {
        let preds: Vec<(f64, bool)> = (0..100)
            .map(|i| {
                if i < 50 { (0.2, false) } else { (0.8, true) }
            })
            .collect();
        let roc = compute_roc(&preds);
        let auc = roc.auc();
        assert!(auc > 0.99, "Perfect classifier should have AUC ~1.0, got {}", auc);
    }

    #[test]
    fn test_roc_curve_random() {
        let preds: Vec<(f64, bool)> = (0..200)
            .map(|i| ((i as f64 * 0.37).sin().abs(), i % 2 == 0))
            .collect();
        let roc = compute_roc(&preds);
        let auc = roc.auc();
        // Random should be ~0.5 ± some variance.
        assert!(auc > 0.1 && auc < 0.9, "AUC {} out of range for random", auc);
    }

    #[test]
    fn test_roc_empty() {
        let roc = compute_roc(&[]);
        assert!(roc.is_empty());
    }

    #[test]
    fn test_pr_curve_basic() {
        let preds: Vec<(f64, bool)> = vec![
            (0.9, true), (0.8, true), (0.7, false),
            (0.6, true), (0.4, false), (0.3, false),
        ];
        let pr = compute_pr_curve(&preds);
        assert!(!pr.is_empty());
        let ap = pr.average_precision();
        assert!(ap > 0.0 && ap <= 1.0);
    }

    #[test]
    fn test_descriptive_stats_basic() {
        let data = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let s = compute_descriptive(&data);
        assert_eq!(s.count, 5);
        assert!((s.mean - 6.0).abs() < 1e-10);
        assert!((s.median - 6.0).abs() < 1e-10);
        assert!((s.min - 2.0).abs() < 1e-10);
        assert!((s.max - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_descriptive_stats_single() {
        let s = compute_descriptive(&[42.0]);
        assert_eq!(s.count, 1);
        assert!((s.mean - 42.0).abs() < 1e-10);
        assert!((s.std_dev - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_descriptive_stats_empty() {
        let s = compute_descriptive(&[]);
        assert_eq!(s.count, 0);
    }

    #[test]
    fn test_descriptive_stats_quartiles() {
        let data: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let s = compute_descriptive(&data);
        assert!((s.q1 - 25.75).abs() < 1.0);
        assert!((s.q3 - 75.25).abs() < 1.0);
        assert!(s.iqr > 48.0 && s.iqr < 51.0);
    }

    #[test]
    fn test_scalability_linear() {
        let data: Vec<(usize, f64)> = (1..=10).map(|n| (n, n as f64 * 10.0 + 5.0)).collect();
        let sm = ScalabilityMetrics::from_measurements(data);
        assert_eq!(sm.complexity, ComplexityClass::Linear);
        assert!(sm.r_squared > 0.95);
    }

    #[test]
    fn test_scalability_quadratic() {
        let data: Vec<(usize, f64)> = (1..=10).map(|n| (n, (n * n) as f64 * 2.0)).collect();
        let sm = ScalabilityMetrics::from_measurements(data);
        assert!(
            matches!(sm.complexity, ComplexityClass::Quadratic),
            "Expected Quadratic, got {:?}",
            sm.complexity
        );
    }

    #[test]
    fn test_performance_metrics() {
        let timings = vec![
            (10.0, 20.0, 1000u64),
            (15.0, 25.0, 1500),
            (12.0, 0.0, 1200),
        ];
        let pm = PerformanceMetrics::from_timings(&timings, 3, 6);
        assert!((pm.tier1_time_ms - 37.0).abs() < 1e-10);
        assert!((pm.tier2_time_ms - 45.0).abs() < 1e-10);
        assert_eq!(pm.peak_memory_bytes, 1500);
        assert!((pm.escalation_rate - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_clinical_metrics_from_data() {
        let data = vec![
            (3, 1, 3.0, 8),
            (0, 0, 0.0, 5),
            (2, 2, 4.0, 10),
        ];
        let cm = ClinicalMetrics::from_patient_data(&data);
        assert!((cm.conflicts_per_patient - 5.0 / 3.0).abs() < 1e-10);
        assert!((cm.patients_with_conflicts - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_bootstrap_mean_ci() {
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let ci = bootstrap_mean_ci(&data, 0.95, 1000);
        assert!(ci.lower < ci.point_estimate);
        assert!(ci.upper > ci.point_estimate);
        assert!((ci.point_estimate - 49.5).abs() < 0.1);
    }

    #[test]
    fn test_bootstrap_median_ci() {
        let data: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let ci = bootstrap_median_ci(&data, 0.95, 500);
        assert!(ci.lower <= ci.point_estimate);
        assert!(ci.upper >= ci.point_estimate);
    }

    #[test]
    fn test_wilcoxon_identical() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = WilcoxonSignedRank::test(&a, &b);
        assert_eq!(result.n_effective, 0);
        assert!((result.p_value - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_wilcoxon_different() {
        let a = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let result = WilcoxonSignedRank::test(&a, &b);
        assert!(result.z_score > 2.0);
        assert!(result.significant_005);
    }

    #[test]
    fn test_cohens_d() {
        let a = vec![10.0, 12.0, 14.0, 16.0, 18.0];
        let b = vec![20.0, 22.0, 24.0, 26.0, 28.0];
        let d = cohens_d(&a, &b);
        // Large effect size.
        assert!(d > 2.0);
    }

    #[test]
    fn test_pearson_r_perfect() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let r = pearson_r(&x, &y);
        assert!((r - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_spearman_rho_perfect() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let rho = spearman_rho(&x, &y);
        assert!((rho - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_confusion_matrix_display() {
        let cm = ConfusionMatrix::new(10, 2, 3, 85);
        let s = format!("{}", cm);
        assert!(s.contains("TP=10"));
        assert!(s.contains("F1="));
    }

    #[test]
    fn test_fbeta_score() {
        let cm = ConfusionMatrix::new(80, 10, 20, 90);
        let f1 = cm.f1_score();
        let f1_via_beta = cm.fbeta_score(1.0);
        assert!((f1 - f1_via_beta).abs() < 1e-10);
        // F2 should be higher recall-weighted.
        let f2 = cm.fbeta_score(2.0);
        assert!(f2 > 0.0);
    }

    #[test]
    fn test_roc_optimal_threshold() {
        let preds: Vec<(f64, bool)> = (0..100)
            .map(|i| {
                let score = i as f64 / 100.0;
                let label = i >= 50;
                (score, label)
            })
            .collect();
        let roc = compute_roc(&preds);
        let thresh = roc.optimal_threshold_youden();
        assert!(thresh > 0.3 && thresh < 0.7);
    }

    #[test]
    fn test_scalability_predict() {
        let data: Vec<(usize, f64)> = (1..=10).map(|n| (n, n as f64 * 5.0)).collect();
        let sm = ScalabilityMetrics::from_measurements(data);
        let predicted = sm.predict(20);
        assert!(predicted > 80.0 && predicted < 120.0);
    }
}
