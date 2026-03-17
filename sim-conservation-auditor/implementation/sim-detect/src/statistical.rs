//! Statistical hypothesis tests for conservation law violation detection.
//!
//! Implements chi-squared, Kolmogorov–Smirnov, Grubbs, F-test, Wald,
//! and t-test with proper numerical CDF approximations.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Common result type
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    pub statistic: f64,
    pub p_value: f64,
    pub reject: bool,
    pub test_name: String,
    pub degrees_of_freedom: Option<f64>,
}

impl TestResult {
    pub fn new(name: &str, statistic: f64, p_value: f64, alpha: f64) -> Self {
        Self {
            statistic,
            p_value,
            reject: p_value < alpha,
            test_name: name.to_string(),
            degrees_of_freedom: None,
        }
    }

    pub fn with_df(mut self, df: f64) -> Self {
        self.degrees_of_freedom = Some(df);
        self
    }
}

/// Trait for all statistical tests.
pub trait StatisticalTest {
    fn test(&self, data: &[f64]) -> TestResult;
    fn name(&self) -> &str;
}

// ===========================================================================
// Mathematical helper functions – CDF approximations
// ===========================================================================

/// Standard normal CDF using the Abramowitz & Stegun approximation (7.1.26).
/// Maximum absolute error < 1.5 × 10^-7.
pub fn normal_cdf(x: f64) -> f64 {
    if x < -8.0 {
        return 0.0;
    }
    if x > 8.0 {
        return 1.0;
    }
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let t = x.abs();

    // Constants for the rational approximation
    let b1 = 0.319381530;
    let b2 = -0.356563782;
    let b3 = 1.781477937;
    let b4 = -1.821255978;
    let b5 = 1.330274429;
    let p = 0.2316419;

    let u = 1.0 / (1.0 + p * t);
    let pdf = (-0.5 * t * t).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let poly = u * (b1 + u * (b2 + u * (b3 + u * (b4 + u * b5))));
    let cdf_positive = 1.0 - pdf * poly;

    0.5 * (1.0 + sign * (2.0 * cdf_positive - 1.0))
}

/// Inverse normal CDF using the rational approximation by Peter Acklam.
/// Accurate to ~1.15e-9 in the central region.
pub fn normal_quantile(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if (p - 0.5).abs() < 1e-15 {
        return 0.0;
    }

    // Coefficients for the rational approximation
    let a = [
        -3.969683028665376e+01,
         2.209460984245205e+02,
        -2.759285104469687e+02,
         1.383577518672690e+02,
        -3.066479806614716e+01,
         2.506628277459239e+00,
    ];
    let b = [
        -5.447609879822406e+01,
         1.615858368580409e+02,
        -1.556989798598866e+02,
         6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    let c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
         4.374664141464968e+00,
         2.938163982698783e+00,
    ];
    let d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];

    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        let num = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q) + c[5];
        let den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q) + 1.0;
        num / den
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        let num = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r) + a[5];
        let den = (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r) + 1.0;
        q * num / den
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        let num = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q) + c[5];
        let den = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q) + 1.0;
        -(num / den)
    }
}

/// Chi-squared CDF via the Wilson-Hilferty normal approximation.
/// Accurate for df >= 2.
pub fn chi_squared_cdf(x: f64, df: f64) -> f64 {
    if x <= 0.0 || df <= 0.0 {
        return 0.0;
    }
    if df < 1.0 {
        return 0.0;
    }
    // Wilson-Hilferty approximation: transform chi-squared to approx normal
    let z = ((x / df).powf(1.0 / 3.0) - (1.0 - 2.0 / (9.0 * df)))
        / (2.0 / (9.0 * df)).sqrt();
    normal_cdf(z)
}

/// Upper-tail probability of chi-squared: P(X > x).
pub fn chi_squared_sf(x: f64, df: f64) -> f64 {
    1.0 - chi_squared_cdf(x, df)
}

/// Student's t CDF approximation.
/// Uses the regularized incomplete beta function relationship:
///   P(T <= t) = 1 - 0.5 * I_x(df/2, 0.5) where x = df / (df + t^2)
/// We use a simple approximation for large df and a series for small df.
pub fn t_cdf(t: f64, df: f64) -> f64 {
    if df <= 0.0 {
        return 0.5;
    }
    // For large df, t-distribution -> normal
    if df > 1000.0 {
        return normal_cdf(t);
    }

    let x = df / (df + t * t);
    let half_df = df / 2.0;
    let ibeta = regularized_incomplete_beta(x, half_df, 0.5);
    if t >= 0.0 {
        1.0 - 0.5 * ibeta
    } else {
        0.5 * ibeta
    }
}

/// Upper-tail probability of t-distribution: P(T > t).
pub fn t_sf(t: f64, df: f64) -> f64 {
    1.0 - t_cdf(t, df)
}

/// F-distribution CDF using the regularized incomplete beta function.
/// P(F <= f) = I_x(d1/2, d2/2) where x = d1*f / (d1*f + d2)
pub fn f_cdf(f: f64, d1: f64, d2: f64) -> f64 {
    if f <= 0.0 || d1 <= 0.0 || d2 <= 0.0 {
        return 0.0;
    }
    let x = d1 * f / (d1 * f + d2);
    regularized_incomplete_beta(x, d1 / 2.0, d2 / 2.0)
}

/// Upper-tail probability of F-distribution.
pub fn f_sf(f: f64, d1: f64, d2: f64) -> f64 {
    1.0 - f_cdf(f, d1, d2)
}

/// Regularized incomplete beta function I_x(a, b) using a continued fraction
/// expansion (Lentz's method). This is the standard numerical approach.
pub fn regularized_incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    // Use the symmetry relation if needed for convergence
    if x > (a + 1.0) / (a + b + 2.0) {
        return 1.0 - regularized_incomplete_beta(1.0 - x, b, a);
    }

    let ln_prefix = a * x.ln() + b * (1.0 - x).ln() - ln_beta(a, b);
    if ln_prefix < -500.0 {
        return 0.0;
    }
    let prefix = ln_prefix.exp();

    // Continued fraction using the modified Lentz method
    let max_iter = 200;
    let epsilon = 1e-14;
    let tiny = 1e-30;

    let mut c = 1.0;
    let mut d = 1.0 - (a + b) * x / (a + 1.0);
    if d.abs() < tiny {
        d = tiny;
    }
    d = 1.0 / d;
    let mut result = d;

    for m in 1..=max_iter {
        let m_f = m as f64;

        // Even step
        let num_even = m_f * (b - m_f) * x / ((a + 2.0 * m_f - 1.0) * (a + 2.0 * m_f));
        d = 1.0 + num_even * d;
        if d.abs() < tiny { d = tiny; }
        c = 1.0 + num_even / c;
        if c.abs() < tiny { c = tiny; }
        d = 1.0 / d;
        result *= d * c;

        // Odd step
        let num_odd = -((a + m_f) * (a + b + m_f) * x)
            / ((a + 2.0 * m_f) * (a + 2.0 * m_f + 1.0));
        d = 1.0 + num_odd * d;
        if d.abs() < tiny { d = tiny; }
        c = 1.0 + num_odd / c;
        if c.abs() < tiny { c = tiny; }
        d = 1.0 / d;
        let delta = d * c;
        result *= delta;

        if (delta - 1.0).abs() < epsilon {
            break;
        }
    }

    prefix * result / a
}

/// Log of the Beta function: ln B(a,b) = ln Γ(a) + ln Γ(b) - ln Γ(a+b)
pub fn ln_beta(a: f64, b: f64) -> f64 {
    ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
}

/// Stirling-series-based log-gamma approximation (Lanczos).
pub fn ln_gamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::INFINITY;
    }
    // Lanczos approximation with g = 7
    let coefficients = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];

    if x < 0.5 {
        // Reflection formula
        let pi = std::f64::consts::PI;
        return (pi / (pi * x).sin()).ln() - ln_gamma(1.0 - x);
    }

    let x = x - 1.0;
    let mut sum = coefficients[0];
    for (i, &c) in coefficients[1..].iter().enumerate() {
        sum += c / (x + i as f64 + 1.0);
    }

    let t = x + 7.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + sum.ln()
}

/// Gamma function via exp(ln_gamma(x)).
pub fn gamma_fn(x: f64) -> f64 {
    ln_gamma(x).exp()
}

// ===========================================================================
// Chi-Squared Test
// ===========================================================================

/// Chi-squared goodness-of-fit test.
/// Tests whether observed frequencies match expected frequencies.
pub struct ChiSquaredTest {
    pub significance: f64,
}

impl ChiSquaredTest {
    pub fn new(significance: f64) -> Self {
        Self { significance }
    }

    /// Perform chi-squared test on binned data.
    /// `observed` and `expected` are bin counts.
    pub fn test_binned(&self, observed: &[f64], expected: &[f64]) -> TestResult {
        assert_eq!(observed.len(), expected.len());
        let df = (observed.len() as f64 - 1.0).max(1.0);
        let mut chi2 = 0.0;
        for (o, e) in observed.iter().zip(expected.iter()) {
            if *e > 0.0 {
                chi2 += (o - e).powi(2) / e;
            }
        }
        let p = chi_squared_sf(chi2, df);
        TestResult::new("Chi-Squared", chi2, p, self.significance).with_df(df)
    }

    /// Auto-bin continuous data and test against a normal distribution.
    pub fn test_normality(&self, data: &[f64]) -> TestResult {
        let n = data.len();
        if n < 10 {
            return TestResult::new("Chi-Squared", 0.0, 1.0, self.significance);
        }
        let mean = data.iter().sum::<f64>() / n as f64;
        let var = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        let std = var.sqrt();
        if std < 1e-30 {
            return TestResult::new("Chi-Squared", 0.0, 1.0, self.significance);
        }

        let n_bins = ((2.0 * (n as f64).powf(2.0 / 5.0)).round() as usize).max(5);
        let min_val = data.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let bin_width = (max_val - min_val) / n_bins as f64;

        if bin_width <= 0.0 {
            return TestResult::new("Chi-Squared", 0.0, 1.0, self.significance);
        }

        let mut observed = vec![0.0_f64; n_bins];
        for &x in data {
            let idx = ((x - min_val) / bin_width).floor() as usize;
            let idx = idx.min(n_bins - 1);
            observed[idx] += 1.0;
        }

        let mut expected = vec![0.0_f64; n_bins];
        for i in 0..n_bins {
            let lo = min_val + i as f64 * bin_width;
            let hi = lo + bin_width;
            let p_lo = normal_cdf((lo - mean) / std);
            let p_hi = normal_cdf((hi - mean) / std);
            expected[i] = (p_hi - p_lo) * n as f64;
            if expected[i] < 0.5 {
                expected[i] = 0.5; // avoid zero expected counts
            }
        }

        self.test_binned(&observed, &expected)
    }
}

impl StatisticalTest for ChiSquaredTest {
    fn test(&self, data: &[f64]) -> TestResult {
        self.test_normality(data)
    }

    fn name(&self) -> &str {
        "ChiSquaredTest"
    }
}

// ===========================================================================
// Kolmogorov-Smirnov Test
// ===========================================================================

/// One-sample Kolmogorov-Smirnov test against a given CDF.
/// Default: tests against normal(0, σ_estimated).
pub struct KolmogorovSmirnovTest {
    pub significance: f64,
}

impl KolmogorovSmirnovTest {
    pub fn new(significance: f64) -> Self {
        Self { significance }
    }

    /// Two-sample KS test: compares the distributions of two samples.
    pub fn test_two_sample(&self, sample1: &[f64], sample2: &[f64]) -> TestResult {
        let mut s1 = sample1.to_vec();
        let mut s2 = sample2.to_vec();
        s1.sort_by(|a, b| a.partial_cmp(b).unwrap());
        s2.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n1 = s1.len() as f64;
        let n2 = s2.len() as f64;
        let mut i = 0usize;
        let mut j = 0usize;
        let mut d_max = 0.0_f64;

        while i < s1.len() && j < s2.len() {
            let ecdf1 = (i + 1) as f64 / n1;
            let ecdf2 = (j + 1) as f64 / n2;
            if s1[i] <= s2[j] {
                d_max = d_max.max((ecdf1 - j as f64 / n2).abs());
                i += 1;
            } else {
                d_max = d_max.max((i as f64 / n1 - ecdf2).abs());
                j += 1;
            }
        }

        let n_eff = (n1 * n2 / (n1 + n2)).sqrt();
        let p = ks_p_value(d_max, n_eff);
        TestResult::new("KS-TwoSample", d_max, p, self.significance)
    }

    /// One-sample KS test against the normal distribution with estimated parameters.
    pub fn test_normal(&self, data: &[f64]) -> TestResult {
        let n = data.len();
        if n < 5 {
            return TestResult::new("KS-Normal", 0.0, 1.0, self.significance);
        }
        let mean = data.iter().sum::<f64>() / n as f64;
        let var = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        let std = var.sqrt();
        if std < 1e-30 {
            return TestResult::new("KS-Normal", 0.0, 1.0, self.significance);
        }

        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mut d_max = 0.0_f64;
        for (i, &x) in sorted.iter().enumerate() {
            let ecdf = (i + 1) as f64 / n as f64;
            let cdf = normal_cdf((x - mean) / std);
            let d = (ecdf - cdf).abs();
            d_max = d_max.max(d);
            // Also check ecdf - 1/n
            let ecdf_below = i as f64 / n as f64;
            d_max = d_max.max((ecdf_below - cdf).abs());
        }

        // Lilliefors correction: reduce critical value because parameters are estimated
        let p = ks_p_value(d_max, (n as f64).sqrt()) * 0.8;
        let p = p.min(1.0);
        TestResult::new("KS-Normal", d_max, p, self.significance)
    }

    /// Test whether two halves of a time series have the same distribution (stationarity check).
    pub fn test_stationarity(&self, data: &[f64]) -> TestResult {
        if data.len() < 10 {
            return TestResult::new("KS-Stationarity", 0.0, 1.0, self.significance);
        }
        let mid = data.len() / 2;
        let first_half = &data[..mid];
        let second_half = &data[mid..];
        self.test_two_sample(first_half, second_half)
    }
}

/// Approximate p-value for the KS statistic using the Kolmogorov distribution.
/// Marsaglia et al. (2003) series approximation.
fn ks_p_value(d: f64, sqrt_n: f64) -> f64 {
    let z = d * sqrt_n;
    if z < 0.01 {
        return 1.0;
    }
    if z > 3.0 {
        return 0.0;
    }
    // Asymptotic formula: P(D > d) ≈ 2 * Σ_{k=1}^{∞} (-1)^{k+1} * exp(-2k²z²)
    let mut sum = 0.0;
    for k in 1..=100 {
        let sign = if k % 2 == 1 { 1.0 } else { -1.0 };
        let term = sign * (-2.0 * (k as f64).powi(2) * z * z).exp();
        sum += term;
        if term.abs() < 1e-15 {
            break;
        }
    }
    (2.0 * sum).max(0.0).min(1.0)
}

impl StatisticalTest for KolmogorovSmirnovTest {
    fn test(&self, data: &[f64]) -> TestResult {
        self.test_normal(data)
    }

    fn name(&self) -> &str {
        "KolmogorovSmirnovTest"
    }
}

// ===========================================================================
// Grubbs Test (Outlier Detection)
// ===========================================================================

/// Grubbs test for a single outlier in a univariate sample.
pub struct GrubbsTest {
    pub significance: f64,
}

impl GrubbsTest {
    pub fn new(significance: f64) -> Self {
        Self { significance }
    }

    /// Find and test the most extreme value (max |x - mean| / std).
    pub fn test_single_outlier(&self, data: &[f64]) -> TestResult {
        let n = data.len();
        if n < 3 {
            return TestResult::new("Grubbs", 0.0, 1.0, self.significance);
        }
        let mean = data.iter().sum::<f64>() / n as f64;
        let var = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        let std = var.sqrt();
        if std < 1e-30 {
            return TestResult::new("Grubbs", 0.0, 1.0, self.significance);
        }

        let g = data
            .iter()
            .map(|x| (x - mean).abs() / std)
            .fold(0.0_f64, f64::max);

        // P-value approximation: convert to t-distribution
        let n_f = n as f64;
        let t_sq = n_f * (n_f - 2.0) * g * g / (n_f * (n_f - 1.0) - n_f * g * g);
        let t_sq = t_sq.max(0.0);
        let t_val = t_sq.sqrt();
        let df = n_f - 2.0;
        let p_one_sided = t_sf(t_val, df);
        let p = (n_f * p_one_sided * 2.0).min(1.0); // Bonferroni-like correction

        TestResult::new("Grubbs", g, p, self.significance).with_df(df)
    }

    /// Iterative Grubbs: remove outliers one at a time until no more found.
    pub fn find_all_outliers(&self, data: &[f64]) -> Vec<usize> {
        let mut remaining: Vec<(usize, f64)> = data.iter().cloned().enumerate().collect();
        let mut outlier_indices = Vec::new();

        loop {
            if remaining.len() < 3 {
                break;
            }
            let values: Vec<f64> = remaining.iter().map(|(_, v)| *v).collect();
            let result = self.test_single_outlier(&values);
            if !result.reject {
                break;
            }
            // Find the most extreme point
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let (idx_in_remaining, _) = remaining
                .iter()
                .enumerate()
                .max_by(|(_, (_, a)), (_, (_, b))| {
                    (a - mean)
                        .abs()
                        .partial_cmp(&(b - mean).abs())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap();
            let (original_idx, _) = remaining.remove(idx_in_remaining);
            outlier_indices.push(original_idx);
        }
        outlier_indices
    }
}

impl StatisticalTest for GrubbsTest {
    fn test(&self, data: &[f64]) -> TestResult {
        self.test_single_outlier(data)
    }

    fn name(&self) -> &str {
        "GrubbsTest"
    }
}

// ===========================================================================
// F-Test (Variance Ratio)
// ===========================================================================

/// F-test to compare variances of two samples.
/// Useful for detecting if error variance has increased (structural break).
pub struct FTest {
    pub significance: f64,
}

impl FTest {
    pub fn new(significance: f64) -> Self {
        Self { significance }
    }

    /// Two-sample F-test: test H0: σ1² = σ2² against H1: σ1² ≠ σ2².
    pub fn test_two_sample(&self, sample1: &[f64], sample2: &[f64]) -> TestResult {
        let n1 = sample1.len();
        let n2 = sample2.len();
        if n1 < 2 || n2 < 2 {
            return TestResult::new("F-Test", 1.0, 1.0, self.significance);
        }
        let mean1 = sample1.iter().sum::<f64>() / n1 as f64;
        let mean2 = sample2.iter().sum::<f64>() / n2 as f64;
        let var1 = sample1.iter().map(|x| (x - mean1).powi(2)).sum::<f64>() / (n1 - 1) as f64;
        let var2 = sample2.iter().map(|x| (x - mean2).powi(2)).sum::<f64>() / (n2 - 1) as f64;

        if var2 < 1e-30 {
            return TestResult::new("F-Test", f64::INFINITY, 0.0, self.significance);
        }

        let f_stat = var1 / var2;
        let d1 = (n1 - 1) as f64;
        let d2 = (n2 - 1) as f64;

        // Two-sided test
        let p = if f_stat > 1.0 {
            2.0 * f_sf(f_stat, d1, d2)
        } else {
            2.0 * f_cdf(f_stat, d1, d2)
        };
        let p = p.min(1.0);

        TestResult::new("F-Test", f_stat, p, self.significance)
    }

    /// Test if variance has changed between first and second half of time series.
    pub fn test_variance_break(&self, data: &[f64]) -> TestResult {
        if data.len() < 4 {
            return TestResult::new("F-Test-Break", 1.0, 1.0, self.significance);
        }
        let mid = data.len() / 2;
        self.test_two_sample(&data[..mid], &data[mid..])
    }
}

impl StatisticalTest for FTest {
    fn test(&self, data: &[f64]) -> TestResult {
        self.test_variance_break(data)
    }

    fn name(&self) -> &str {
        "FTest"
    }
}

// ===========================================================================
// Wald Test (Mean Drift)
// ===========================================================================

/// Wald test for whether the mean drift of a series is significantly nonzero.
pub struct WaldTest {
    pub significance: f64,
    pub null_value: f64,
}

impl WaldTest {
    pub fn new(null_value: f64, significance: f64) -> Self {
        Self {
            significance,
            null_value,
        }
    }

    /// Test H0: slope = null_value using regression.
    pub fn test_drift(&self, data: &[f64]) -> TestResult {
        let n = data.len();
        if n < 3 {
            return TestResult::new("Wald", 0.0, 1.0, self.significance);
        }

        // Fit simple linear regression: y = a + b*x
        let n_f = n as f64;
        let x_mean = (n_f - 1.0) / 2.0;
        let y_mean = data.iter().sum::<f64>() / n_f;

        let mut sxy = 0.0;
        let mut sxx = 0.0;
        for (i, &y) in data.iter().enumerate() {
            let x = i as f64;
            sxy += (x - x_mean) * (y - y_mean);
            sxx += (x - x_mean).powi(2);
        }

        if sxx.abs() < 1e-30 {
            return TestResult::new("Wald", 0.0, 1.0, self.significance);
        }

        let slope = sxy / sxx;
        // Standard error of slope
        let y_pred: Vec<f64> = (0..n).map(|i| y_mean + slope * (i as f64 - x_mean)).collect();
        let sse: f64 = data
            .iter()
            .zip(y_pred.iter())
            .map(|(y, yp)| (y - yp).powi(2))
            .sum();
        let se_slope = (sse / ((n - 2) as f64 * sxx)).sqrt();

        if se_slope < 1e-30 {
            if (slope - self.null_value).abs() < 1e-15 {
                return TestResult::new("Wald", 0.0, 1.0, self.significance);
            } else {
                return TestResult::new("Wald", f64::INFINITY, 0.0, self.significance);
            }
        }

        let w = (slope - self.null_value) / se_slope;
        let w_sq = w * w;
        let p = chi_squared_sf(w_sq, 1.0);

        TestResult::new("Wald", w_sq, p, self.significance).with_df(1.0)
    }
}

impl StatisticalTest for WaldTest {
    fn test(&self, data: &[f64]) -> TestResult {
        self.test_drift(data)
    }

    fn name(&self) -> &str {
        "WaldTest"
    }
}

// ===========================================================================
// T-Test
// ===========================================================================

/// One-sample t-test: H0: μ = μ0.
pub struct TTest {
    pub null_mean: f64,
    pub significance: f64,
}

impl TTest {
    pub fn new(null_mean: f64) -> Self {
        Self {
            null_mean,
            significance: 0.05,
        }
    }

    pub fn with_significance(mut self, alpha: f64) -> Self {
        self.significance = alpha;
        self
    }

    /// One-sample t-test.
    pub fn test_one_sample(&self, data: &[f64]) -> TestResult {
        let n = data.len();
        if n < 2 {
            return TestResult::new("TTest", 0.0, 1.0, self.significance);
        }
        let mean = data.iter().sum::<f64>() / n as f64;
        let var = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        let se = (var / n as f64).sqrt();

        if se < 1e-30 {
            if (mean - self.null_mean).abs() < 1e-15 {
                return TestResult::new("TTest", 0.0, 1.0, self.significance);
            } else {
                return TestResult::new("TTest", f64::INFINITY, 0.0, self.significance);
            }
        }

        let t = (mean - self.null_mean) / se;
        let df = (n - 1) as f64;
        // Two-sided p-value
        let p = 2.0 * t_sf(t.abs(), df);
        let p = p.min(1.0);

        TestResult::new("TTest", t, p, self.significance).with_df(df)
    }

    /// Convenience alias for [`test_one_sample`](TTest::test_one_sample).
    pub fn test(&self, data: &[f64]) -> TestResult {
        self.test_one_sample(data)
    }

    /// Two-sample independent t-test (Welch's).
    pub fn test_two_sample(data1: &[f64], data2: &[f64], alpha: f64) -> TestResult {
        let n1 = data1.len();
        let n2 = data2.len();
        if n1 < 2 || n2 < 2 {
            return TestResult::new("TTest-Two", 0.0, 1.0, alpha);
        }
        let m1 = data1.iter().sum::<f64>() / n1 as f64;
        let m2 = data2.iter().sum::<f64>() / n2 as f64;
        let v1 = data1.iter().map(|x| (x - m1).powi(2)).sum::<f64>() / (n1 - 1) as f64;
        let v2 = data2.iter().map(|x| (x - m2).powi(2)).sum::<f64>() / (n2 - 1) as f64;

        let se = (v1 / n1 as f64 + v2 / n2 as f64).sqrt();
        if se < 1e-30 {
            return TestResult::new("TTest-Two", 0.0, 1.0, alpha);
        }

        let t = (m1 - m2) / se;

        // Welch-Satterthwaite degrees of freedom
        let num = (v1 / n1 as f64 + v2 / n2 as f64).powi(2);
        let denom = (v1 / n1 as f64).powi(2) / (n1 as f64 - 1.0)
            + (v2 / n2 as f64).powi(2) / (n2 as f64 - 1.0);
        let df = if denom > 0.0 { num / denom } else { 1.0 };

        let p = 2.0 * t_sf(t.abs(), df);
        TestResult::new("TTest-Two", t, p.min(1.0), alpha).with_df(df)
    }

    /// Paired t-test on the differences between two paired samples.
    pub fn test_paired(data1: &[f64], data2: &[f64], alpha: f64) -> TestResult {
        assert_eq!(data1.len(), data2.len());
        let diffs: Vec<f64> = data1.iter().zip(data2.iter()).map(|(a, b)| a - b).collect();
        let test = TTest::new(0.0).with_significance(alpha);
        let mut result = test.test_one_sample(&diffs);
        result.test_name = "TTest-Paired".to_string();
        result
    }
}

impl StatisticalTest for TTest {
    fn test(&self, data: &[f64]) -> TestResult {
        self.test_one_sample(data)
    }

    fn name(&self) -> &str {
        "TTest"
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normal_cdf_symmetry() {
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-6);
        let p = normal_cdf(1.96);
        assert!((p - 0.975).abs() < 0.005, "Φ(1.96) ≈ 0.975, got {}", p);
        assert!((normal_cdf(-1.96) - 0.025).abs() < 0.005);
    }

    #[test]
    fn test_normal_cdf_tails() {
        assert!(normal_cdf(-10.0) < 1e-10);
        assert!(normal_cdf(10.0) > 1.0 - 1e-10);
    }

    #[test]
    fn test_normal_quantile_roundtrip() {
        for &p in &[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99] {
            let z = normal_quantile(p);
            let p_back = normal_cdf(z);
            assert!(
                (p_back - p).abs() < 1e-5,
                "Round-trip failed for p={}: got z={}, back to p={}",
                p,
                z,
                p_back
            );
        }
    }

    #[test]
    fn test_chi_squared_cdf_basic() {
        // χ²(1) at x=3.841 should give p≈0.95
        let cdf = chi_squared_cdf(3.841, 1.0);
        assert!((cdf - 0.95).abs() < 0.02, "Got {}", cdf);
    }

    #[test]
    fn test_ln_gamma_known_values() {
        // Γ(1) = 1 -> ln Γ(1) = 0
        assert!(ln_gamma(1.0).abs() < 1e-10);
        // Γ(2) = 1 -> ln Γ(2) = 0
        assert!(ln_gamma(2.0).abs() < 1e-10);
        // Γ(5) = 24 -> ln Γ(5) = ln(24)
        assert!((ln_gamma(5.0) - (24.0_f64).ln()).abs() < 1e-8);
    }

    #[test]
    fn test_t_cdf_symmetry() {
        let df = 10.0;
        assert!((t_cdf(0.0, df) - 0.5).abs() < 1e-6);
        let upper = t_cdf(2.228, df);
        assert!((upper - 0.975).abs() < 0.02, "t_cdf(2.228, 10) ≈ 0.975, got {}", upper);
    }

    #[test]
    fn test_ttest_zero_mean() {
        // Data with mean very close to zero should not reject H0: μ=0
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let data: Vec<f64> = (0..500).map(|_| rng.gen::<f64>() - 0.5).collect();
        let test = TTest::new(0.0);
        let result = test.test(&data);
        // With 500 samples of uniform[-0.5, 0.5], mean is close to 0
        // May or may not reject; at least verify it runs
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    #[test]
    fn test_ttest_shifted_mean() {
        // Data with clear mean ≠ 0 should reject
        let data: Vec<f64> = (0..200).map(|_| 5.0 + 0.01).collect();
        let test = TTest::new(0.0);
        let result = test.test(&data);
        assert!(result.reject, "Should reject H0 when mean = 5.01");
        assert!(result.p_value < 0.001);
    }

    #[test]
    fn test_ftest_equal_variance() {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let s1: Vec<f64> = (0..200).map(|_| rng.gen::<f64>()).collect();
        let s2: Vec<f64> = (0..200).map(|_| rng.gen::<f64>()).collect();
        let ftest = FTest::new(0.05);
        let result = ftest.test_two_sample(&s1, &s2);
        // Equal variances should usually not reject
        assert!(result.statistic > 0.0);
    }

    #[test]
    fn test_ftest_different_variance() {
        let s1: Vec<f64> = (0..200).map(|i| (i as f64) * 0.001).collect();
        let s2: Vec<f64> = (0..200).map(|i| (i as f64) * 1.0).collect();
        let ftest = FTest::new(0.05);
        let result = ftest.test_two_sample(&s1, &s2);
        assert!(result.reject, "Should detect different variances");
    }

    #[test]
    fn test_grubbs_no_outlier() {
        let data: Vec<f64> = (0..50).map(|i| (i as f64) * 0.1).collect();
        let grubbs = GrubbsTest::new(0.05);
        let result = grubbs.test(&data);
        // A uniform-ish spread might or might not flag; just verify it runs
        assert!(result.p_value >= 0.0);
    }

    #[test]
    fn test_grubbs_with_outlier() {
        let mut data: Vec<f64> = vec![0.0; 50];
        use rand::Rng;
        let mut rng = rand::thread_rng();
        for d in data.iter_mut() {
            *d = rng.gen::<f64>() * 0.1;
        }
        data.push(100.0); // extreme outlier
        let grubbs = GrubbsTest::new(0.05);
        let result = grubbs.test(&data);
        assert!(result.reject, "Should detect the outlier (p={})", result.p_value);
    }

    #[test]
    fn test_ks_normal_data() {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        // Approximate normal via Box-Muller
        let data: Vec<f64> = (0..500)
            .map(|_| {
                let u1: f64 = rng.gen::<f64>().max(1e-10);
                let u2: f64 = rng.gen();
                (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
            })
            .collect();
        let ks = KolmogorovSmirnovTest::new(0.05);
        let result = ks.test(&data);
        // Normal data should usually not reject normality
        assert!(result.p_value > 0.0);
    }

    #[test]
    fn test_wald_no_drift() {
        let data: Vec<f64> = vec![1.0; 100];
        let wald = WaldTest::new(0.0, 0.05);
        let result = wald.test(&data);
        assert!(!result.reject, "Constant data has zero slope");
    }

    #[test]
    fn test_wald_with_drift() {
        let data: Vec<f64> = (0..200).map(|i| i as f64 * 0.5).collect();
        let wald = WaldTest::new(0.0, 0.05);
        let result = wald.test(&data);
        assert!(result.reject, "Linear data has nonzero slope");
    }

    #[test]
    fn test_chi_squared_uniform_bins() {
        let observed = vec![50.0, 50.0, 50.0, 50.0];
        let expected = vec![50.0, 50.0, 50.0, 50.0];
        let chi2 = ChiSquaredTest::new(0.05);
        let result = chi2.test_binned(&observed, &expected);
        assert!((result.statistic - 0.0).abs() < 1e-10);
        assert!(!result.reject);
    }

    #[test]
    fn test_chi_squared_bad_fit() {
        let observed = vec![100.0, 0.0, 100.0, 0.0];
        let expected = vec![50.0, 50.0, 50.0, 50.0];
        let chi2 = ChiSquaredTest::new(0.05);
        let result = chi2.test_binned(&observed, &expected);
        assert!(result.reject, "Clearly non-uniform bins should reject");
    }

    #[test]
    fn test_incomplete_beta_bounds() {
        assert!(regularized_incomplete_beta(0.0, 1.0, 1.0).abs() < 1e-12);
        assert!((regularized_incomplete_beta(1.0, 1.0, 1.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_incomplete_beta_uniform() {
        // For Beta(1,1), the regularized incomplete beta is just x
        for &x in &[0.1, 0.3, 0.5, 0.7, 0.9] {
            let val = regularized_incomplete_beta(x, 1.0, 1.0);
            assert!((val - x).abs() < 1e-6, "I_{}(1,1) should be {}, got {}", x, x, val);
        }
    }

    #[test]
    fn test_ks_stationarity() {
        let ks = KolmogorovSmirnovTest::new(0.05);
        // Stationary data: same distribution
        let data: Vec<f64> = (0..200).map(|i| (i % 10) as f64).collect();
        let result = ks.test_stationarity(&data);
        assert!(!result.reject, "Periodic data should be stationary");

        // Non-stationary: different means in two halves
        let mut data2: Vec<f64> = vec![0.0; 100];
        data2.extend(vec![10.0; 100]);
        let result2 = ks.test_stationarity(&data2);
        assert!(result2.reject, "Step function should be non-stationary");
    }

    #[test]
    fn test_ttest_paired() {
        let before = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let after = vec![1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5];
        let result = TTest::test_paired(&before, &after, 0.05);
        assert!(result.reject, "Consistent +0.5 shift should be significant");
    }
}
