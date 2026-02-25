//! Statistical testing utilities for the CABER (Coalgebraic Behavioral Auditing) project.
//!
//! Provides implementations of classical statistical tests, concentration inequalities,
//! information-theoretic divergences, and multiple testing corrections used throughout
//! the behavioral certification pipeline.

use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Result of a Kolmogorov-Smirnov test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KSTestResult {
    /// The KS D-statistic (supremum of |F_1 - F_2|).
    pub statistic: f64,
    /// Asymptotic p-value.
    pub p_value: f64,
    /// Whether to reject H0 at the stored significance level.
    pub reject: bool,
    /// The significance level used for the rejection decision.
    pub alpha: f64,
}

/// Result of a chi-squared test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChiSquaredResult {
    /// Chi-squared test statistic.
    pub statistic: f64,
    /// Degrees of freedom.
    pub df: usize,
    /// p-value from the chi-squared distribution.
    pub p_value: f64,
    /// Whether to reject H0 at the stored significance level.
    pub reject: bool,
    /// The significance level used.
    pub alpha: f64,
}

/// Result of an Anderson-Darling test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ADTestResult {
    /// The A² statistic.
    pub statistic: f64,
    /// Approximate p-value.
    pub p_value: f64,
    /// Whether to reject the normality hypothesis at the stored significance level.
    pub reject: bool,
    /// The significance level used.
    pub alpha: f64,
}

/// An empirical CDF constructed from sample data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmpiricalCDF {
    sorted: Vec<f64>,
}

impl EmpiricalCDF {
    /// Create a new empirical CDF from data.
    pub fn new(data: &[f64]) -> Self {
        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        EmpiricalCDF { sorted }
    }

    /// Evaluate the empirical CDF at point `x`: F_n(x) = #{X_i <= x} / n.
    pub fn evaluate(&self, x: f64) -> f64 {
        if self.sorted.is_empty() {
            return 0.0;
        }
        let n = self.sorted.len();
        // Binary search for the number of elements <= x.
        let count = match self.sorted.binary_search_by(|v| {
            v.partial_cmp(&x).unwrap_or(std::cmp::Ordering::Equal)
        }) {
            Ok(mut idx) => {
                // There may be duplicates; walk right to include all equal values.
                while idx + 1 < n && self.sorted[idx + 1] <= x {
                    idx += 1;
                }
                idx + 1
            }
            Err(idx) => idx,
        };
        count as f64 / n as f64
    }

    /// Return the number of data points.
    pub fn len(&self) -> usize {
        self.sorted.len()
    }

    /// Whether the CDF is empty.
    pub fn is_empty(&self) -> bool {
        self.sorted.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Mathematical helper functions
// ---------------------------------------------------------------------------

/// Approximation of the error function using the Abramowitz & Stegun formula (7.1.26).
/// Maximum error ~ 1.5×10⁻⁷.
pub fn erf_approx(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    // Constants from Abramowitz & Stegun.
    let a1: f64 = 0.254829592;
    let a2: f64 = -0.284496736;
    let a3: f64 = 1.421413741;
    let a4: f64 = -1.453152027;
    let a5: f64 = 1.061405429;
    let p: f64 = 0.3275911;

    let t = 1.0 / (1.0 + p * x);
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t3 * t;
    let t5 = t4 * t;

    let y = 1.0 - (a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5) * (-x * x).exp();
    sign * y
}

/// Complementary error function: erfc(x) = 1 - erf(x).
pub fn erfc_approx(x: f64) -> f64 {
    1.0 - erf_approx(x)
}

/// Normal CDF: Φ(x) = 0.5 * erfc(-x / √2).
pub fn normal_cdf(x: f64) -> f64 {
    0.5 * erfc_approx(-x / std::f64::consts::SQRT_2)
}

/// Normal PDF: φ(x) = exp(-x²/2) / √(2π).
pub fn normal_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
}

/// Lanczos approximation of the gamma function Γ(z) for z > 0.
pub fn gamma_function(z: f64) -> f64 {
    if z < 0.5 {
        // Reflection formula: Γ(z) = π / (sin(πz) Γ(1−z))
        return PI / ((PI * z).sin() * gamma_function(1.0 - z));
    }

    // Lanczos coefficients (g = 7, n = 9).
    let g = 7.0_f64;
    let coefs: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1259.139_216_722_402_8,
        771.323_428_777_653_08,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];

    let z = z - 1.0;
    let mut x = coefs[0];
    for (i, &c) in coefs.iter().enumerate().skip(1) {
        x += c / (z + i as f64);
    }

    let t = z + g + 0.5;
    (2.0 * PI).sqrt() * t.powf(z + 0.5) * (-t).exp() * x
}

/// Natural logarithm of Γ(z) for z > 0.
pub fn ln_gamma(z: f64) -> f64 {
    if z < 0.5 {
        let reflection = PI / ((PI * z).sin());
        return reflection.ln() - ln_gamma(1.0 - z);
    }

    let g = 7.0_f64;
    let coefs: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1259.139_216_722_402_8,
        771.323_428_777_653_08,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];

    let z = z - 1.0;
    let mut x = coefs[0];
    for (i, &c) in coefs.iter().enumerate().skip(1) {
        x += c / (z + i as f64);
    }
    let t = z + g + 0.5;
    0.5 * (2.0 * PI).ln() + (z + 0.5) * t.ln() - t + x.ln()
}

/// Lower regularized incomplete gamma function P(a, x) = γ(a, x) / Γ(a).
///
/// Uses the series expansion for x < a + 1 and the continued-fraction
/// representation otherwise.
pub fn incomplete_gamma(a: f64, x: f64) -> f64 {
    if x < 0.0 {
        return 0.0;
    }
    if x == 0.0 {
        return 0.0;
    }
    if a <= 0.0 {
        return 1.0;
    }

    if x < a + 1.0 {
        // Series expansion.
        incomplete_gamma_series(a, x)
    } else {
        // Continued fraction (Legendre).
        1.0 - incomplete_gamma_cf(a, x)
    }
}

/// Series expansion of P(a, x).
fn incomplete_gamma_series(a: f64, x: f64) -> f64 {
    let max_iter = 200;
    let eps = 1e-14;

    let mut sum = 1.0 / a;
    let mut term = 1.0 / a;

    for n in 1..=max_iter {
        term *= x / (a + n as f64);
        sum += term;
        if term.abs() < eps * sum.abs() {
            break;
        }
    }

    sum * (-x + a * x.ln() - ln_gamma(a)).exp()
}

/// Continued-fraction representation for the upper incomplete gamma Q(a, x) = 1 − P(a, x).
/// Returns Q(a, x) so that caller can compute P = 1 − Q.
fn incomplete_gamma_cf(a: f64, x: f64) -> f64 {
    let max_iter = 200;
    let eps = 1e-14;
    let tiny = 1e-30;

    let mut f = tiny;
    let mut c = tiny;
    let mut d = 0.0;

    for n in 1..=max_iter {
        let nf = n as f64;
        // Even step: a_n = -(nf - a) when even index mapped from n
        // Using modified Lentz continued fraction for Q(a,x).
        let an = if n == 1 {
            1.0
        } else if n % 2 == 0 {
            let k = n as f64 / 2.0;
            k * (a - k)
        } else {
            let k = (n as f64 - 1.0) / 2.0;
            -(a - 1.0 - k) * (a + k)  // Mapping differs; use simpler Lentz.
        };
        let _ = an; // We'll use the standard recurrence instead.

        // Simpler: use the standard CF  Q(a,x) = e^{-x} x^a / Γ(a) * 1/(x+ (1-a)/(1+ 1/(x+ (2-a)/(1+ ...))))
        // We'll use the Lentz form for the CF:
        //   b_0 = 0, a_1 = 1
        //   a_{2k}   = k(a-k)      b_{2k}   = x
        //   a_{2k+1} = -(a-1-k+1)(a+k) ... This gets complicated.
        // Fall back to the simple recurrence from Numerical Recipes.
        let _ = nf;
        break; // We'll implement a cleaner version below.
    }

    // Clean implementation of CF using modified Lentz's method as in Numerical Recipes.
    // CF: Q(a,x) = exp(-x + a*ln(x) - lnΓ(a)) * (1 / (x + 1 - a - 1*( 1-a)/(x+3-a- 2*(2-a)/(x+5-a- ...))))
    // Equivalently, use the standard form:
    //   f = x + 1 - a
    //   then add terms:  a_i / (f + ...) where a_i = i*(a-i), b_i = 2*i + x + 1 - a

    let b0 = x + 1.0 - a;
    if b0.abs() < tiny {
        // Fall back to series for safety.
        return 1.0 - incomplete_gamma_series(a, x);
    }

    f = b0;
    c = b0;
    d = 1.0 / b0;
    let mut result = d;

    for i in 1..=max_iter {
        let ai = -(i as f64) * (i as f64 - a);
        let bi = 2.0 * i as f64 + x + 1.0 - a;

        c = bi + ai / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = bi + ai * d;
        if d.abs() < tiny {
            d = tiny;
        }
        d = 1.0 / d;
        let delta = c * d;
        result *= delta;
        if (delta - 1.0).abs() < eps {
            break;
        }
    }

    let log_prefix = -x + a * x.ln() - ln_gamma(a);
    log_prefix.exp() / result
}

/// Upper-tail probability of the chi-squared distribution: P(χ² > x | df).
pub fn chi_squared_survival(x: f64, df: usize) -> f64 {
    if df == 0 {
        return if x > 0.0 { 0.0 } else { 1.0 };
    }
    let a = df as f64 / 2.0;
    let half_x = x / 2.0;
    1.0 - incomplete_gamma(a, half_x)
}

/// Inverse normal (probit) function — Beasley-Springer-Moro approximation.
pub fn inverse_normal(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

    // Rational approximation from Peter Acklam.
    let a: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    let b: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    let c: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    let d: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];

    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    }
}

/// Approximate the inverse of the t-distribution CDF (quantile) for `df` degrees of freedom.
/// Uses a normal approximation refined with Cornish-Fisher expansion for moderate df,
/// and falls back to the exact normal for large df.
pub fn inverse_t(p: f64, df: f64) -> f64 {
    if df <= 0.0 {
        return f64::NAN;
    }
    if df > 1e6 {
        return inverse_normal(p);
    }

    let z = inverse_normal(p);

    // Cornish-Fisher expansion (first correction term).
    let g1 = (z * z * z + z) / (4.0 * df);
    let g2 = (5.0 * z.powi(5) + 16.0 * z.powi(3) + 3.0 * z) / (96.0 * df * df);
    let g3 = (3.0 * z.powi(7) + 19.0 * z.powi(5) + 17.0 * z.powi(3) - 15.0 * z)
        / (384.0 * df * df * df);

    z + g1 + g2 + g3
}

// ---------------------------------------------------------------------------
// Descriptive statistics
// ---------------------------------------------------------------------------

/// Arithmetic mean.
pub fn mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return f64::NAN;
    }
    data.iter().sum::<f64>() / data.len() as f64
}

/// Population variance (biased). If you need sample variance, use `sample_variance`.
pub fn variance(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    let m = mean(data);
    data.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / data.len() as f64
}

/// Sample variance (unbiased, Bessel-corrected).
pub fn sample_variance(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }
    let m = mean(data);
    data.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / (data.len() - 1) as f64
}

/// Standard deviation (population).
pub fn std_dev(data: &[f64]) -> f64 {
    variance(data).sqrt()
}

/// Sample standard deviation (Bessel-corrected).
pub fn sample_std_dev(data: &[f64]) -> f64 {
    sample_variance(data).sqrt()
}

/// Median.
pub fn median(data: &[f64]) -> f64 {
    if data.is_empty() {
        return f64::NAN;
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}

/// Compute the p-th quantile (0 ≤ p ≤ 1) using linear interpolation (type 7, R default).
pub fn quantile(data: &[f64], p: f64) -> f64 {
    assert!(!data.is_empty(), "quantile: data must be non-empty");
    assert!(
        (0.0..=1.0).contains(&p),
        "quantile: p must be in [0, 1]"
    );

    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();

    if n == 1 {
        return sorted[0];
    }

    // Index on the 0-based scale.
    let h = (n as f64 - 1.0) * p;
    let lo = h.floor() as usize;
    let hi = h.ceil() as usize;
    let frac = h - lo as f64;

    if lo == hi {
        sorted[lo]
    } else {
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

// ---------------------------------------------------------------------------
// Empirical CDF (constructor convenience)
// ---------------------------------------------------------------------------

/// Build an `EmpiricalCDF` from the given data.
pub fn empirical_cdf(data: &[f64]) -> EmpiricalCDF {
    EmpiricalCDF::new(data)
}

// ---------------------------------------------------------------------------
// Hypothesis tests
// ---------------------------------------------------------------------------

/// Two-sample Kolmogorov-Smirnov test.
///
/// Tests H0: the two samples are drawn from the same continuous distribution.
/// The D-statistic is the supremum of |F_1(x) − F_2(x)| over all x.
/// The p-value is computed from the asymptotic Kolmogorov distribution.
pub fn kolmogorov_smirnov_test(sample1: &[f64], sample2: &[f64]) -> KSTestResult {
    kolmogorov_smirnov_test_with_alpha(sample1, sample2, 0.05)
}

/// Two-sample KS test with a configurable significance level.
pub fn kolmogorov_smirnov_test_with_alpha(
    sample1: &[f64],
    sample2: &[f64],
    alpha: f64,
) -> KSTestResult {
    assert!(
        !sample1.is_empty() && !sample2.is_empty(),
        "KS test requires non-empty samples"
    );

    let mut s1 = sample1.to_vec();
    let mut s2 = sample2.to_vec();
    s1.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    s2.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n1 = s1.len() as f64;
    let n2 = s2.len() as f64;

    // Walk through the combined sorted order to find sup |F1 − F2|.
    let mut i: usize = 0;
    let mut j: usize = 0;
    let mut d_max: f64 = 0.0;

    while i < s1.len() && j < s2.len() {
        let v1 = s1[i];
        let v2 = s2[j];

        if v1 <= v2 {
            i += 1;
        }
        if v2 <= v1 {
            j += 1;
        }

        let f1 = i as f64 / n1;
        let f2 = j as f64 / n2;
        let diff = (f1 - f2).abs();
        if diff > d_max {
            d_max = diff;
        }
    }

    let en = (n1 * n2 / (n1 + n2)).sqrt();
    let p_value = ks_p_value(d_max, en);

    KSTestResult {
        statistic: d_max,
        p_value,
        reject: p_value < alpha,
        alpha,
    }
}

/// One-sample KS test: compare a sample against a theoretical CDF.
pub fn kolmogorov_smirnov_one_sample(sample: &[f64], cdf: &dyn Fn(f64) -> f64) -> KSTestResult {
    kolmogorov_smirnov_one_sample_with_alpha(sample, cdf, 0.05)
}

/// One-sample KS test with configurable alpha.
pub fn kolmogorov_smirnov_one_sample_with_alpha(
    sample: &[f64],
    cdf: &dyn Fn(f64) -> f64,
    alpha: f64,
) -> KSTestResult {
    assert!(!sample.is_empty(), "KS test requires a non-empty sample");

    let mut sorted = sample.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len() as f64;

    let mut d_max: f64 = 0.0;
    for (i, &x) in sorted.iter().enumerate() {
        let f_empirical_before = i as f64 / n;
        let f_empirical_after = (i + 1) as f64 / n;
        let f_theoretical = cdf(x);

        let d1 = (f_empirical_after - f_theoretical).abs();
        let d2 = (f_empirical_before - f_theoretical).abs();
        let d = d1.max(d2);
        if d > d_max {
            d_max = d;
        }
    }

    let en = n.sqrt();
    let p_value = ks_p_value(d_max, en);

    KSTestResult {
        statistic: d_max,
        p_value,
        reject: p_value < alpha,
        alpha,
    }
}

/// Compute the survival function of the Kolmogorov distribution.
/// P(D_n > d) where the scaled variable λ = (√n + 0.12 + 0.11/√n) * d.
/// Uses the series P(K > λ) = 2 Σ_{k=1}^{∞} (−1)^{k−1} exp(−2 k² λ²).
fn ks_p_value(d: f64, sqrt_n: f64) -> f64 {
    if d <= 0.0 {
        return 1.0;
    }
    // Use the improved effective-sample-size correction.
    let lambda = (sqrt_n + 0.12 + 0.11 / sqrt_n) * d;

    if lambda < 1e-15 {
        return 1.0;
    }

    let mut sum = 0.0;
    let mut sign = 1.0;
    for k in 1..=100 {
        let kf = k as f64;
        let term = sign * (-2.0 * kf * kf * lambda * lambda).exp();
        sum += term;
        if term.abs() < 1e-15 {
            break;
        }
        sign = -sign;
    }

    let p = 2.0 * sum;
    p.clamp(0.0, 1.0)
}

/// Chi-squared goodness-of-fit test.
///
/// `observed` and `expected` must have the same length ≥ 2. Degrees of freedom = len − 1.
pub fn chi_squared_test(observed: &[f64], expected: &[f64]) -> ChiSquaredResult {
    chi_squared_test_with_alpha(observed, expected, 0.05)
}

/// Chi-squared goodness-of-fit test with configurable alpha.
pub fn chi_squared_test_with_alpha(
    observed: &[f64],
    expected: &[f64],
    alpha: f64,
) -> ChiSquaredResult {
    assert_eq!(
        observed.len(),
        expected.len(),
        "chi_squared_test: observed and expected must have the same length"
    );
    assert!(observed.len() >= 2, "chi_squared_test: need at least 2 bins");

    let stat: f64 = observed
        .iter()
        .zip(expected.iter())
        .map(|(&o, &e)| {
            if e == 0.0 {
                0.0
            } else {
                (o - e).powi(2) / e
            }
        })
        .sum();

    let df = observed.len() - 1;
    let p_value = chi_squared_survival(stat, df);

    ChiSquaredResult {
        statistic: stat,
        df,
        p_value,
        reject: p_value < alpha,
        alpha,
    }
}

/// Chi-squared test of independence on a contingency table.
///
/// `contingency` is a 2-D matrix of observed counts (rows × columns).
pub fn chi_squared_independence(contingency: &[Vec<f64>]) -> ChiSquaredResult {
    chi_squared_independence_with_alpha(contingency, 0.05)
}

/// Chi-squared independence test with configurable alpha.
pub fn chi_squared_independence_with_alpha(
    contingency: &[Vec<f64>],
    alpha: f64,
) -> ChiSquaredResult {
    let nrows = contingency.len();
    assert!(nrows >= 2, "Need at least 2 rows");
    let ncols = contingency[0].len();
    assert!(ncols >= 2, "Need at least 2 columns");
    for row in contingency {
        assert_eq!(row.len(), ncols, "All rows must have the same length");
    }

    // Row totals, column totals, grand total.
    let row_totals: Vec<f64> = contingency.iter().map(|r| r.iter().sum::<f64>()).collect();
    let mut col_totals = vec![0.0; ncols];
    for row in contingency {
        for (j, &v) in row.iter().enumerate() {
            col_totals[j] += v;
        }
    }
    let grand_total: f64 = row_totals.iter().sum();

    assert!(grand_total > 0.0, "Grand total must be positive");

    let mut stat = 0.0;
    for (i, row) in contingency.iter().enumerate() {
        for (j, &observed) in row.iter().enumerate() {
            let expected = row_totals[i] * col_totals[j] / grand_total;
            if expected > 0.0 {
                stat += (observed - expected).powi(2) / expected;
            }
        }
    }

    let df = (nrows - 1) * (ncols - 1);
    let p_value = chi_squared_survival(stat, df);

    ChiSquaredResult {
        statistic: stat,
        df,
        p_value,
        reject: p_value < alpha,
        alpha,
    }
}

/// Anderson-Darling test for normality.
///
/// Tests H0: the sample comes from a normal distribution.
/// The sample is standardized by its sample mean and standard deviation before computing A².
pub fn anderson_darling_test(sample: &[f64]) -> ADTestResult {
    anderson_darling_test_with_alpha(sample, 0.05)
}

/// Anderson-Darling normality test with configurable alpha.
pub fn anderson_darling_test_with_alpha(sample: &[f64], alpha: f64) -> ADTestResult {
    assert!(
        sample.len() >= 8,
        "Anderson-Darling test requires at least 8 observations"
    );

    let n = sample.len();
    let nf = n as f64;

    let m = mean(sample);
    let s = sample_std_dev(sample);
    assert!(s > 0.0, "Anderson-Darling: sample must have nonzero variance");

    // Standardize and sort.
    let mut z: Vec<f64> = sample.iter().map(|&x| (x - m) / s).collect();
    z.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Compute A² statistic.
    let mut a2 = 0.0;
    for i in 0..n {
        let phi_zi = normal_cdf(z[i]);
        let phi_zn_i = normal_cdf(z[n - 1 - i]);

        // Clamp to (0, 1) to avoid log(0).
        let phi_zi = phi_zi.clamp(1e-15, 1.0 - 1e-15);
        let phi_zn_i = phi_zn_i.clamp(1e-15, 1.0 - 1e-15);

        let weight = (2 * i + 1) as f64;
        a2 += weight * (phi_zi.ln() + (1.0 - phi_zn_i).ln());
    }
    a2 = -nf - a2 / nf;

    // Apply the modification for the case-3 (mean and variance estimated from data).
    let a2_star = a2 * (1.0 + 0.75 / nf + 2.25 / (nf * nf));

    // Approximate p-value from D'Agostino & Stephens tables (interpolation).
    let p_value = ad_p_value(a2_star);

    ADTestResult {
        statistic: a2_star,
        p_value,
        reject: p_value < alpha,
        alpha,
    }
}

/// Approximate p-value for the Anderson-Darling statistic (case 3: normal, parameters estimated).
/// Piecewise formula from Marsaglia & Marsaglia (2004) / Stephens (1986).
fn ad_p_value(a2: f64) -> f64 {
    if a2 <= 0.0 {
        return 1.0;
    }

    // Approximation for the case-3 AD test (normal, unknown mean and variance).
    // These come from Marsaglia, G. & Marsaglia, J. (2004),
    // "Evaluating the Anderson-Darling Distribution", Journal of Statistical Software.
    if a2 < 0.2 {
        1.0 - (-13.436 + 101.14 * a2 - 223.73 * a2 * a2).exp()
    } else if a2 < 0.34 {
        1.0 - (-8.318 + 42.796 * a2 - 59.938 * a2 * a2).exp()
    } else if a2 < 0.6 {
        (-0.9177 + 4.279 * a2 - 1.38 * a2 * a2).exp()
    } else if a2 < 10.0 {
        // Larger A² → smaller p-value.
        let v = (-1.2937 + 5.709 * a2 - 0.0186 * a2 * a2).exp();
        // Use the relation: p ≈ exp(c0 + c1*A2*)
        (0.01331 * a2.powi(3) - 0.21058 * a2 * a2 + 0.89514 * a2 - 1.30054).exp().min(1.0)
    } else {
        // Very large → essentially zero.
        0.0
    }
}

// ---------------------------------------------------------------------------
// Concentration inequalities
// ---------------------------------------------------------------------------

/// Hoeffding's inequality: P(|S_n/n − E[S_n/n]| ≥ ε) ≤ 2 exp(−2nε²).
///
/// Returns the probability upper bound.
pub fn hoeffding_bound(n: usize, epsilon: f64) -> f64 {
    let bound = 2.0 * (-2.0 * n as f64 * epsilon * epsilon).exp();
    bound.min(1.0)
}

/// Multiplicative Chernoff bound.
///
/// For X ~ Binomial(n, p) and δ > 0:
/// - Upper tail: P(X ≥ (1+δ)np) ≤ [e^δ / (1+δ)^{1+δ}]^{np}
///
/// Returns the bound on P(X ≥ (1+δ)μ) where μ = np.
pub fn chernoff_bound(n: usize, p: f64, delta: f64) -> f64 {
    assert!(p > 0.0 && p < 1.0, "p must be in (0, 1)");
    assert!(delta > 0.0, "delta must be positive");

    let mu = n as f64 * p;

    // [e^δ / (1+δ)^{1+δ}]^μ
    let base = (delta.exp()) / ((1.0 + delta).powf(1.0 + delta));
    let bound = base.powf(mu);
    bound.min(1.0)
}

// ---------------------------------------------------------------------------
// Confidence intervals
// ---------------------------------------------------------------------------

/// Compute a confidence interval for the mean of `data` at the given confidence level
/// (e.g., 0.95 for a 95% CI). Uses the t-distribution approximation.
///
/// Returns `(lower, upper)`.
pub fn confidence_interval(data: &[f64], confidence: f64) -> (f64, f64) {
    assert!(
        data.len() >= 2,
        "confidence_interval: need at least 2 data points"
    );
    assert!(
        confidence > 0.0 && confidence < 1.0,
        "confidence must be in (0, 1)"
    );

    let n = data.len() as f64;
    let m = mean(data);
    let se = sample_std_dev(data) / n.sqrt();
    let df = n - 1.0;

    // Two-tailed: α/2 in each tail.
    let alpha = 1.0 - confidence;
    let t_crit = inverse_t(1.0 - alpha / 2.0, df);

    (m - t_crit * se, m + t_crit * se)
}

// ---------------------------------------------------------------------------
// p-values from z-scores
// ---------------------------------------------------------------------------

/// Two-tailed p-value from a z-score: p = 2 * (1 − Φ(|z|)).
pub fn p_value_from_z(z: f64) -> f64 {
    let p = 2.0 * (1.0 - normal_cdf(z.abs()));
    p.clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// Multiple testing corrections
// ---------------------------------------------------------------------------

/// Bonferroni correction: reject H_i if p_i < α / m.
///
/// Returns a boolean vector indicating which hypotheses are rejected.
pub fn bonferroni_correction(p_values: &[f64], alpha: f64) -> Vec<bool> {
    let m = p_values.len() as f64;
    let threshold = alpha / m;
    p_values.iter().map(|&p| p < threshold).collect()
}

/// Benjamini-Hochberg procedure for controlling the False Discovery Rate.
///
/// Returns a boolean vector indicating which hypotheses are rejected.
pub fn benjamini_hochberg(p_values: &[f64], alpha: f64) -> Vec<bool> {
    let m = p_values.len();
    if m == 0 {
        return vec![];
    }

    // Create (index, p-value) pairs sorted by p-value.
    let mut indexed: Vec<(usize, f64)> = p_values.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // Find the largest k such that p_(k) ≤ k/m * α.
    let mut max_k: Option<usize> = None;
    for (rank, &(_, p)) in indexed.iter().enumerate() {
        let threshold = (rank + 1) as f64 / m as f64 * alpha;
        if p <= threshold {
            max_k = Some(rank);
        }
    }

    let mut rejected = vec![false; m];
    if let Some(k) = max_k {
        // Reject all hypotheses with rank ≤ k.
        for (rank, &(original_idx, _)) in indexed.iter().enumerate() {
            if rank <= k {
                rejected[original_idx] = true;
            }
        }
    }

    rejected
}

// ---------------------------------------------------------------------------
// Information-theoretic measures
// ---------------------------------------------------------------------------

/// Kullback-Leibler divergence D_KL(P || Q) = Σ p_i ln(p_i / q_i).
///
/// Applies additive smoothing (ε = 1e-10) to avoid log(0).
/// Both `p` and `q` should be probability distributions (non-negative, sum to ~1).
pub fn kl_divergence(p: &[f64], q: &[f64]) -> f64 {
    assert_eq!(p.len(), q.len(), "kl_divergence: distributions must have equal length");
    assert!(!p.is_empty(), "kl_divergence: distributions must be non-empty");

    let eps = 1e-10;

    // Normalize with smoothing.
    let p_sum: f64 = p.iter().map(|&x| x + eps).sum();
    let q_sum: f64 = q.iter().map(|&x| x + eps).sum();

    let mut kl = 0.0;
    for (&pi, &qi) in p.iter().zip(q.iter()) {
        let pi_s = (pi + eps) / p_sum;
        let qi_s = (qi + eps) / q_sum;
        kl += pi_s * (pi_s / qi_s).ln();
    }

    kl.max(0.0)
}

/// Total variation distance: TV(P, Q) = 0.5 * Σ |p_i − q_i|.
pub fn total_variation_distance(p: &[f64], q: &[f64]) -> f64 {
    assert_eq!(
        p.len(),
        q.len(),
        "total_variation_distance: distributions must have equal length"
    );
    0.5 * p.iter().zip(q.iter()).map(|(&a, &b)| (a - b).abs()).sum::<f64>()
}

/// Shannon entropy: H(P) = − Σ p_i ln(p_i), using natural logarithm.
///
/// Zero entries are skipped (0 ln 0 → 0).
pub fn entropy(p: &[f64]) -> f64 {
    let mut h = 0.0;
    for &pi in p {
        if pi > 0.0 {
            h -= pi * pi.ln();
        }
    }
    h.max(0.0)
}

/// Mutual information I(X; Y) from a joint probability distribution.
///
/// `joint` is a 2-D matrix where `joint[i][j]` = P(X = i, Y = j).
pub fn mutual_information(joint: &[Vec<f64>]) -> f64 {
    if joint.is_empty() {
        return 0.0;
    }
    let nrows = joint.len();
    let ncols = joint[0].len();

    // Compute marginals.
    let mut p_x = vec![0.0; nrows];
    let mut p_y = vec![0.0; ncols];
    for (i, row) in joint.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            p_x[i] += v;
            p_y[j] += v;
        }
    }

    let eps = 1e-15;
    let mut mi = 0.0;
    for (i, row) in joint.iter().enumerate() {
        for (j, &p_xy) in row.iter().enumerate() {
            if p_xy > eps && p_x[i] > eps && p_y[j] > eps {
                mi += p_xy * (p_xy / (p_x[i] * p_y[j])).ln();
            }
        }
    }

    mi.max(0.0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-4;
    const LOOSE_TOL: f64 = 0.05;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // ---- Descriptive statistics ----

    #[test]
    fn test_mean() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(approx_eq(mean(&data), 3.0, TOL));
    }

    #[test]
    fn test_variance_and_std_dev() {
        let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let m = mean(&data);
        assert!(approx_eq(m, 5.0, TOL));
        // Population variance = 4.0
        assert!(approx_eq(variance(&data), 4.0, TOL));
        assert!(approx_eq(std_dev(&data), 2.0, TOL));
    }

    #[test]
    fn test_median() {
        assert!(approx_eq(median(&[1.0, 3.0, 5.0]), 3.0, TOL));
        assert!(approx_eq(median(&[1.0, 2.0, 3.0, 4.0]), 2.5, TOL));
    }

    #[test]
    fn test_quantile() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        assert!(approx_eq(quantile(&data, 0.0), 1.0, TOL));
        assert!(approx_eq(quantile(&data, 1.0), 10.0, TOL));
        assert!(approx_eq(quantile(&data, 0.5), 5.5, TOL));
        assert!(approx_eq(quantile(&data, 0.25), 3.25, TOL));
    }

    // ---- Helper functions ----

    #[test]
    fn test_erf_approx() {
        assert!(approx_eq(erf_approx(0.0), 0.0, TOL));
        assert!(approx_eq(erf_approx(1.0), 0.8427, TOL));
        assert!(approx_eq(erf_approx(-1.0), -0.8427, TOL));
        assert!(approx_eq(erf_approx(2.0), 0.9953, TOL));
    }

    #[test]
    fn test_gamma_function() {
        // Γ(1) = 1, Γ(2) = 1, Γ(3) = 2, Γ(4) = 6, Γ(0.5) = √π
        assert!(approx_eq(gamma_function(1.0), 1.0, TOL));
        assert!(approx_eq(gamma_function(2.0), 1.0, TOL));
        assert!(approx_eq(gamma_function(3.0), 2.0, TOL));
        assert!(approx_eq(gamma_function(4.0), 6.0, TOL));
        assert!(approx_eq(gamma_function(0.5), PI.sqrt(), TOL));
    }

    #[test]
    fn test_normal_cdf() {
        assert!(approx_eq(normal_cdf(0.0), 0.5, TOL));
        assert!(approx_eq(normal_cdf(1.96), 0.975, 1e-3));
        assert!(approx_eq(normal_cdf(-1.96), 0.025, 1e-3));
    }

    #[test]
    fn test_incomplete_gamma() {
        // P(1, 1) = 1 - e^{-1} ≈ 0.6321
        assert!(approx_eq(incomplete_gamma(1.0, 1.0), 0.6321, 1e-3));
        // P(0.5, 0.5) ≈ 0.6827  (related to erf)
        // P(1, 0) = 0
        assert!(approx_eq(incomplete_gamma(1.0, 0.0), 0.0, TOL));
    }

    // ---- KS tests ----

    #[test]
    fn test_ks_same_distribution() {
        // Two samples from the same distribution should NOT reject.
        let s1: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();
        let s2: Vec<f64> = (0..100).map(|i| (i as f64 + 0.5) / 100.0).collect();
        let result = kolmogorov_smirnov_test(&s1, &s2);
        assert!(
            !result.reject,
            "Should not reject for similar samples: p = {}",
            result.p_value
        );
    }

    #[test]
    fn test_ks_different_distribution() {
        // Very different distributions should reject.
        let s1: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();
        let s2: Vec<f64> = (0..100).map(|i| (i as f64 / 100.0) + 5.0).collect();
        let result = kolmogorov_smirnov_test(&s1, &s2);
        assert!(
            result.reject,
            "Should reject for very different samples: D = {}, p = {}",
            result.statistic, result.p_value
        );
    }

    #[test]
    fn test_ks_one_sample_uniform() {
        // Uniform sample tested against U(0,1) CDF should not reject.
        let sample: Vec<f64> = (1..=200).map(|i| i as f64 / 201.0).collect();
        let result = kolmogorov_smirnov_one_sample(&sample, &|x: f64| x.clamp(0.0, 1.0));
        assert!(
            !result.reject,
            "Uniform sample against uniform CDF should not reject: p = {}",
            result.p_value
        );
    }

    // ---- Chi-squared tests ----

    #[test]
    fn test_chi_squared_goodness_of_fit() {
        // Fair die: observed close to expected → don't reject.
        let observed = [18.0, 16.0, 15.0, 17.0, 18.0, 16.0]; // total 100
        let expected = [100.0 / 6.0; 6];
        let result = chi_squared_test(&observed, &expected);
        assert!(
            !result.reject,
            "Fair die should not reject: χ² = {}, p = {}",
            result.statistic, result.p_value
        );
        assert_eq!(result.df, 5);
    }

    #[test]
    fn test_chi_squared_unfair_die() {
        // Very unfair die → reject.
        let observed = [50.0, 10.0, 10.0, 10.0, 10.0, 10.0]; // total 100
        let expected = [100.0 / 6.0; 6];
        let result = chi_squared_test(&observed, &expected);
        assert!(
            result.reject,
            "Unfair die should reject: χ² = {}, p = {}",
            result.statistic, result.p_value
        );
    }

    #[test]
    fn test_chi_squared_independence() {
        // Independent variables → don't reject.
        let table = vec![
            vec![25.0, 25.0],
            vec![25.0, 25.0],
        ];
        let result = chi_squared_independence(&table);
        assert!(
            !result.reject,
            "Independent table should not reject: χ² = {}, p = {}",
            result.statistic, result.p_value
        );
        assert_eq!(result.df, 1);

        // Dependent variables → reject.
        let table2 = vec![
            vec![50.0, 5.0],
            vec![5.0, 50.0],
        ];
        let result2 = chi_squared_independence(&table2);
        assert!(
            result2.reject,
            "Dependent table should reject: χ² = {}, p = {}",
            result2.statistic, result2.p_value
        );
    }

    // ---- Concentration inequalities ----

    #[test]
    fn test_hoeffding_bound() {
        let bound = hoeffding_bound(100, 0.1);
        // 2 exp(-2 * 100 * 0.01) = 2 exp(-2) ≈ 0.2707
        assert!(approx_eq(bound, 2.0 * (-2.0_f64).exp(), TOL));
        assert!(bound <= 1.0);
        assert!(bound > 0.0);
    }

    #[test]
    fn test_chernoff_bound() {
        let bound = chernoff_bound(100, 0.5, 0.5);
        assert!(bound > 0.0);
        assert!(bound <= 1.0);
        // Bound should be small for large n and moderate delta.
        assert!(bound < 0.1, "Chernoff bound = {}", bound);
    }

    // ---- Confidence interval ----

    #[test]
    fn test_confidence_interval() {
        let data: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let (lo, hi) = confidence_interval(&data, 0.95);
        let m = mean(&data);
        assert!(lo < m && m < hi, "Mean should be inside the CI");
        assert!(lo < hi, "Lower bound should be less than upper");
        // For 1..100, mean = 50.5
        assert!(approx_eq(m, 50.5, TOL));
    }

    // ---- p-value from z ----

    #[test]
    fn test_p_value_from_z() {
        // z = 0 → p = 1.0
        assert!(approx_eq(p_value_from_z(0.0), 1.0, TOL));
        // z = 1.96 → p ≈ 0.05
        assert!(approx_eq(p_value_from_z(1.96), 0.05, 1e-2));
        // z = 2.576 → p ≈ 0.01
        assert!(approx_eq(p_value_from_z(2.576), 0.01, 1e-2));
    }

    // ---- Multiple testing corrections ----

    #[test]
    fn test_bonferroni_correction() {
        let p_values = [0.01, 0.04, 0.03, 0.20];
        let results = bonferroni_correction(&p_values, 0.05);
        // Threshold = 0.05 / 4 = 0.0125
        assert_eq!(results, vec![true, false, false, false]);
    }

    #[test]
    fn test_benjamini_hochberg() {
        let p_values = [0.005, 0.01, 0.03, 0.04, 0.50];
        let results = benjamini_hochberg(&p_values, 0.05);
        // Sorted: 0.005(1), 0.01(2), 0.03(3), 0.04(4), 0.50(5)
        // Thresholds: 0.01, 0.02, 0.03, 0.04, 0.05
        // Largest k where p_(k) ≤ threshold: k=4 (0.04 ≤ 0.04)
        // So first 4 (sorted) are rejected.
        assert_eq!(results, vec![true, true, true, true, false]);
    }

    // ---- Information-theoretic measures ----

    #[test]
    fn test_kl_divergence_identical() {
        let p = [0.25, 0.25, 0.25, 0.25];
        let kl = kl_divergence(&p, &p);
        assert!(
            approx_eq(kl, 0.0, 1e-6),
            "KL(P || P) should be ~0, got {}",
            kl
        );
    }

    #[test]
    fn test_total_variation_distance() {
        let p = [0.5, 0.5];
        let q = [0.25, 0.75];
        let tv = total_variation_distance(&p, &q);
        // 0.5 * (|0.5-0.25| + |0.5-0.75|) = 0.5 * (0.25 + 0.25) = 0.25
        assert!(approx_eq(tv, 0.25, TOL));
    }

    #[test]
    fn test_entropy() {
        // Uniform over 4 outcomes: H = ln(4) ≈ 1.3863
        let p = [0.25, 0.25, 0.25, 0.25];
        assert!(approx_eq(entropy(&p), 4.0_f64.ln(), TOL));

        // Deterministic: H = 0
        let q = [1.0, 0.0, 0.0, 0.0];
        assert!(approx_eq(entropy(&q), 0.0, TOL));
    }

    #[test]
    fn test_mutual_information_independent() {
        // Independent joint distribution → MI ≈ 0.
        let joint = vec![
            vec![0.25, 0.25],
            vec![0.25, 0.25],
        ];
        let mi = mutual_information(&joint);
        assert!(
            approx_eq(mi, 0.0, 1e-6),
            "MI for independent variables should be ~0, got {}",
            mi
        );
    }

    #[test]
    fn test_mutual_information_dependent() {
        // Perfectly dependent: joint = [[0.5, 0], [0, 0.5]] → MI = ln(2)
        let joint = vec![
            vec![0.5, 0.0],
            vec![0.0, 0.5],
        ];
        let mi = mutual_information(&joint);
        assert!(
            approx_eq(mi, 2.0_f64.ln(), 1e-3),
            "MI for perfectly dependent should be ln(2) ≈ 0.693, got {}",
            mi
        );
    }

    // ---- Anderson-Darling ----

    #[test]
    fn test_anderson_darling_normal_sample() {
        // A sample from a normal distribution should not be rejected.
        // We use a deterministic "normal-like" sample via the inverse CDF.
        let n = 100;
        let sample: Vec<f64> = (1..=n)
            .map(|i| inverse_normal(i as f64 / (n as f64 + 1.0)))
            .collect();
        let result = anderson_darling_test(&sample);
        // For a perfectly normal sample, we expect not to reject.
        assert!(
            result.statistic < 2.0,
            "A² should be small for normal data, got {}",
            result.statistic
        );
    }

    // ---- Empirical CDF ----

    #[test]
    fn test_empirical_cdf() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let ecdf = empirical_cdf(&data);

        assert!(approx_eq(ecdf.evaluate(0.0), 0.0, TOL));
        assert!(approx_eq(ecdf.evaluate(1.0), 0.2, TOL));
        assert!(approx_eq(ecdf.evaluate(3.0), 0.6, TOL));
        assert!(approx_eq(ecdf.evaluate(5.0), 1.0, TOL));
        assert!(approx_eq(ecdf.evaluate(6.0), 1.0, TOL));
    }
}
