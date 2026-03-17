//! Effect size measures for collusion detection.
//!
//! Provides standardized effect sizes (Cohen's d, Hedges' g, Glass's Δ),
//! domain-specific collusion effect sizes, power analysis, and confidence
//! intervals for effect sizes.

use serde::{Deserialize, Serialize};
use shared_types::{CollusionError, CollusionResult, ConfidenceInterval};

// ── Effect interpretation ───────────────────────────────────────────────────

/// Qualitative interpretation of an effect size magnitude.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EffectInterpretation {
    Negligible,
    Small,
    Medium,
    Large,
}

impl std::fmt::Display for EffectInterpretation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Negligible => write!(f, "Negligible"),
            Self::Small => write!(f, "Small"),
            Self::Medium => write!(f, "Medium"),
            Self::Large => write!(f, "Large"),
        }
    }
}

/// Interpret a standardized effect size (|d|) according to Cohen's conventions.
pub fn effect_size_interpretation(d: f64) -> EffectInterpretation {
    let abs_d = d.abs();
    if abs_d < 0.2 {
        EffectInterpretation::Negligible
    } else if abs_d < 0.5 {
        EffectInterpretation::Small
    } else if abs_d < 0.8 {
        EffectInterpretation::Medium
    } else {
        EffectInterpretation::Large
    }
}

// ── Helper statistics ───────────────────────────────────────────────────────

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn variance(xs: &[f64], ddof: usize) -> f64 {
    let n = xs.len();
    if n <= ddof {
        return 0.0;
    }
    let m = mean(xs);
    let ss: f64 = xs.iter().map(|x| (x - m).powi(2)).sum();
    ss / (n - ddof) as f64
}

fn std_dev(xs: &[f64], ddof: usize) -> f64 {
    variance(xs, ddof).sqrt()
}

/// Pooled standard deviation for two independent samples.
fn pooled_sd(xs: &[f64], ys: &[f64]) -> f64 {
    let n1 = xs.len() as f64;
    let n2 = ys.len() as f64;
    if n1 + n2 <= 2.0 {
        return 0.0;
    }
    let var1 = variance(xs, 1);
    let var2 = variance(ys, 1);
    (((n1 - 1.0) * var1 + (n2 - 1.0) * var2) / (n1 + n2 - 2.0)).sqrt()
}

// ── Cohen's d ───────────────────────────────────────────────────────────────

/// Cohen's d: standardized mean difference using pooled SD.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CohenD {
    pub value: f64,
    pub interpretation: EffectInterpretation,
    pub group1_mean: f64,
    pub group2_mean: f64,
    pub pooled_sd: f64,
}

impl CohenD {
    /// Compute Cohen's d from two independent samples.
    pub fn compute(group1: &[f64], group2: &[f64]) -> CollusionResult<Self> {
        if group1.is_empty() || group2.is_empty() {
            return Err(CollusionError::StatisticalTest(
                "Cohen's d requires non-empty groups".into(),
            ));
        }
        let m1 = mean(group1);
        let m2 = mean(group2);
        let sp = pooled_sd(group1, group2);
        let d = if sp.abs() < 1e-15 { 0.0 } else { (m1 - m2) / sp };
        Ok(Self {
            value: d,
            interpretation: effect_size_interpretation(d),
            group1_mean: m1,
            group2_mean: m2,
            pooled_sd: sp,
        })
    }

    /// Cohen's d for a one-sample test against a known mean.
    pub fn one_sample(data: &[f64], mu0: f64) -> CollusionResult<Self> {
        if data.is_empty() {
            return Err(CollusionError::StatisticalTest(
                "Cohen's d requires non-empty data".into(),
            ));
        }
        let m = mean(data);
        let s = std_dev(data, 1);
        let d = if s.abs() < 1e-15 { 0.0 } else { (m - mu0) / s };
        Ok(Self {
            value: d,
            interpretation: effect_size_interpretation(d),
            group1_mean: m,
            group2_mean: mu0,
            pooled_sd: s,
        })
    }

    /// Non-centrality parameter for power analysis.
    pub fn noncentrality(&self, n1: usize, n2: usize) -> f64 {
        let nh = 2.0 * (n1 as f64) * (n2 as f64) / (n1 + n2) as f64;
        self.value * nh.sqrt()
    }

    /// Approximate variance of Cohen's d (large-sample formula).
    pub fn variance_estimate(&self, n1: usize, n2: usize) -> f64 {
        let n = n1 as f64 + n2 as f64;
        let d2 = self.value * self.value;
        (n1 as f64 + n2 as f64) / (n1 as f64 * n2 as f64) + d2 / (2.0 * n)
    }
}

// ── Hedges' g ───────────────────────────────────────────────────────────────

/// Hedges' g: bias-corrected Cohen's d for small samples.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HedgesG {
    pub value: f64,
    pub interpretation: EffectInterpretation,
    pub cohens_d: f64,
    pub correction_factor: f64,
}

impl HedgesG {
    /// Compute Hedges' g from two independent samples.
    pub fn compute(group1: &[f64], group2: &[f64]) -> CollusionResult<Self> {
        let cd = CohenD::compute(group1, group2)?;
        let df = (group1.len() + group2.len() - 2) as f64;
        // J correction factor: J ≈ 1 - 3/(4*df - 1)
        let j = if df > 0.0 { 1.0 - 3.0 / (4.0 * df - 1.0) } else { 1.0 };
        let g = cd.value * j;
        Ok(Self {
            value: g,
            interpretation: effect_size_interpretation(g),
            cohens_d: cd.value,
            correction_factor: j,
        })
    }

    /// Approximate variance of Hedges' g.
    pub fn variance_estimate(&self, n1: usize, n2: usize) -> f64 {
        let n = n1 as f64 + n2 as f64;
        let base = (n1 as f64 + n2 as f64) / (n1 as f64 * n2 as f64);
        let g2 = self.value * self.value;
        (base + g2 / (2.0 * n)) * self.correction_factor * self.correction_factor
    }
}

// ── Glass's Δ ───────────────────────────────────────────────────────────────

/// Glass's Δ: standardized mean difference using the control group SD.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlassDelta {
    pub value: f64,
    pub interpretation: EffectInterpretation,
    pub treatment_mean: f64,
    pub control_mean: f64,
    pub control_sd: f64,
}

impl GlassDelta {
    /// Compute Glass's Δ. `control` is the reference group.
    pub fn compute(treatment: &[f64], control: &[f64]) -> CollusionResult<Self> {
        if treatment.is_empty() || control.is_empty() {
            return Err(CollusionError::StatisticalTest(
                "Glass's delta requires non-empty groups".into(),
            ));
        }
        let mt = mean(treatment);
        let mc = mean(control);
        let sc = std_dev(control, 1);
        let delta = if sc.abs() < 1e-15 { 0.0 } else { (mt - mc) / sc };
        Ok(Self {
            value: delta,
            interpretation: effect_size_interpretation(delta),
            treatment_mean: mt,
            control_mean: mc,
            control_sd: sc,
        })
    }
}

// ── Point-biserial r ────────────────────────────────────────────────────────

/// Point-biserial correlation coefficient (effect size for binary predictor).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PointBiserialR {
    pub value: f64,
    pub interpretation: EffectInterpretation,
}

impl PointBiserialR {
    /// Compute from two groups (binary predictor → continuous outcome).
    pub fn compute(group0: &[f64], group1: &[f64]) -> CollusionResult<Self> {
        if group0.is_empty() || group1.is_empty() {
            return Err(CollusionError::StatisticalTest(
                "Point-biserial r requires non-empty groups".into(),
            ));
        }
        let n0 = group0.len() as f64;
        let n1 = group1.len() as f64;
        let n = n0 + n1;
        let m0 = mean(group0);
        let m1 = mean(group1);
        // Total variance
        let all: Vec<f64> = group0.iter().chain(group1.iter()).copied().collect();
        let sy = std_dev(&all, 1);
        let r = if sy.abs() < 1e-15 {
            0.0
        } else {
            (m1 - m0) / sy * (n0 * n1 / (n * n)).sqrt()
        };
        let interp = effect_size_interpretation(r);
        Ok(Self { value: r, interpretation: interp })
    }

    /// Convert to Cohen's d.
    pub fn to_cohens_d(&self) -> f64 {
        let r2 = self.value * self.value;
        if (1.0 - r2).abs() < 1e-15 {
            return f64::INFINITY * self.value.signum();
        }
        2.0 * self.value / (1.0 - r2).sqrt()
    }
}

// ── Odds ratio ──────────────────────────────────────────────────────────────

/// Odds ratio for 2×2 contingency tables.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OddsRatio {
    pub value: f64,
    pub log_or: f64,
    pub se_log_or: f64,
    /// Cells: a (treatment+success), b (treatment+failure),
    /// c (control+success), d (control+failure)
    pub a: usize,
    pub b: usize,
    pub c: usize,
    pub d: usize,
}

impl OddsRatio {
    /// Compute from a 2×2 table: [[a, b], [c, d]].
    pub fn compute(a: usize, b: usize, c: usize, d: usize) -> CollusionResult<Self> {
        if a == 0 || b == 0 || c == 0 || d == 0 {
            // Apply Haldane-Anscombe correction
            let af = a as f64 + 0.5;
            let bf = b as f64 + 0.5;
            let cf = c as f64 + 0.5;
            let df = d as f64 + 0.5;
            let or = (af * df) / (bf * cf);
            let log_or = or.ln();
            let se = (1.0 / af + 1.0 / bf + 1.0 / cf + 1.0 / df).sqrt();
            return Ok(Self { value: or, log_or, se_log_or: se, a, b, c, d });
        }
        let af = a as f64;
        let bf = b as f64;
        let cf = c as f64;
        let df = d as f64;
        let or = (af * df) / (bf * cf);
        let log_or = or.ln();
        let se = (1.0 / af + 1.0 / bf + 1.0 / cf + 1.0 / df).sqrt();
        Ok(Self { value: or, log_or, se_log_or: se, a, b, c, d })
    }

    /// 95% CI for the odds ratio (Woolf method).
    pub fn confidence_interval(&self, alpha: f64) -> ConfidenceInterval {
        // Normal quantile approximation
        let z = normal_quantile(1.0 - alpha / 2.0);
        let lo = (self.log_or - z * self.se_log_or).exp();
        let hi = (self.log_or + z * self.se_log_or).exp();
        ConfidenceInterval::new(lo, hi, 1.0 - alpha, self.value)
    }
}

// ── Collusion effect size ───────────────────────────────────────────────────

/// Domain-specific effect size: collusion premium / SE(collusion premium).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollusionEffectSize {
    pub value: f64,
    pub interpretation: EffectInterpretation,
    pub collusion_premium: f64,
    pub standard_error: f64,
    pub nash_baseline: f64,
    pub observed_mean: f64,
}

impl CollusionEffectSize {
    /// Compute the collusion effect size from observed prices, Nash price, and monopoly price.
    pub fn compute(
        observed_prices: &[f64],
        nash_price: f64,
        monopoly_price: f64,
    ) -> CollusionResult<Self> {
        if observed_prices.is_empty() {
            return Err(CollusionError::StatisticalTest(
                "Collusion effect size requires non-empty data".into(),
            ));
        }
        let obs_mean = mean(observed_prices);
        let obs_se = std_dev(observed_prices, 1) / (observed_prices.len() as f64).sqrt();

        let denom = monopoly_price - nash_price;
        let cp = if denom.abs() < 1e-12 {
            0.0
        } else {
            ((obs_mean - nash_price) / denom).clamp(0.0, 1.0)
        };

        // SE of the collusion premium via delta method
        let se_cp = if denom.abs() < 1e-12 { 0.0 } else { obs_se / denom.abs() };
        let es = if se_cp.abs() < 1e-15 { 0.0 } else { cp / se_cp };

        Ok(Self {
            value: es,
            interpretation: effect_size_interpretation(es),
            collusion_premium: cp,
            standard_error: se_cp,
            nash_baseline: nash_price,
            observed_mean: obs_mean,
        })
    }

    /// Whether the collusion premium is economically meaningful.
    pub fn is_economically_significant(&self, threshold: f64) -> bool {
        self.collusion_premium > threshold
    }
}

// ── Power analysis ──────────────────────────────────────────────────────────

/// Power analysis for hypothesis tests.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerAnalysis {
    pub effect_size: f64,
    pub alpha: f64,
    pub power: f64,
    pub required_n: Option<usize>,
}

impl PowerAnalysis {
    /// Compute power for a two-sample t-test given effect size, alpha, and sample sizes.
    pub fn two_sample_t_power(d: f64, alpha: f64, n1: usize, n2: usize) -> Self {
        let z_alpha = normal_quantile(1.0 - alpha / 2.0);
        let nh = 2.0 * (n1 as f64) * (n2 as f64) / (n1 + n2) as f64;
        let lambda = d.abs() * nh.sqrt();
        // Power ≈ Φ(λ - z_α)
        let power = normal_cdf(lambda - z_alpha);
        Self {
            effect_size: d,
            alpha,
            power,
            required_n: None,
        }
    }

    /// Compute required sample size per group for target power (balanced design).
    pub fn required_sample_size(d: f64, alpha: f64, target_power: f64) -> Self {
        if d.abs() < 1e-15 {
            return Self {
                effect_size: d,
                alpha,
                power: target_power,
                required_n: None,
            };
        }
        let z_alpha = normal_quantile(1.0 - alpha / 2.0);
        let z_beta = normal_quantile(target_power);
        let n = ((z_alpha + z_beta) / d).powi(2);
        let n_per_group = n.ceil() as usize;
        Self {
            effect_size: d,
            alpha,
            power: target_power,
            required_n: Some(n_per_group.max(2)),
        }
    }

    /// Compute power for a one-sample t-test.
    pub fn one_sample_t_power(d: f64, alpha: f64, n: usize) -> Self {
        let z_alpha = normal_quantile(1.0 - alpha / 2.0);
        let lambda = d.abs() * (n as f64).sqrt();
        let power = normal_cdf(lambda - z_alpha);
        Self {
            effect_size: d,
            alpha,
            power,
            required_n: Some(n),
        }
    }

    /// Sensitivity analysis: compute detectable effect size for given power and N.
    pub fn detectable_effect(alpha: f64, power: f64, n: usize) -> f64 {
        let z_alpha = normal_quantile(1.0 - alpha / 2.0);
        let z_beta = normal_quantile(power);
        (z_alpha + z_beta) / (n as f64).sqrt()
    }
}

// ── Effect size confidence interval ─────────────────────────────────────────

/// Confidence interval for an effect size.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectSizeCI {
    pub point_estimate: f64,
    pub lower: f64,
    pub upper: f64,
    pub level: f64,
    pub method: String,
}

impl EffectSizeCI {
    /// Approximate CI for Cohen's d using the large-sample normal approximation.
    pub fn for_cohens_d(d: &CohenD, n1: usize, n2: usize, alpha: f64) -> Self {
        let var_d = d.variance_estimate(n1, n2);
        let se = var_d.sqrt();
        let z = normal_quantile(1.0 - alpha / 2.0);
        Self {
            point_estimate: d.value,
            lower: d.value - z * se,
            upper: d.value + z * se,
            level: 1.0 - alpha,
            method: "Normal approximation".into(),
        }
    }

    /// Approximate CI for Hedges' g.
    pub fn for_hedges_g(g: &HedgesG, n1: usize, n2: usize, alpha: f64) -> Self {
        let var_g = g.variance_estimate(n1, n2);
        let se = var_g.sqrt();
        let z = normal_quantile(1.0 - alpha / 2.0);
        Self {
            point_estimate: g.value,
            lower: g.value - z * se,
            upper: g.value + z * se,
            level: 1.0 - alpha,
            method: "Normal approximation (Hedges)".into(),
        }
    }

    /// CI for a correlation coefficient using Fisher z-transform.
    pub fn for_correlation(r: f64, n: usize, alpha: f64) -> Self {
        let z_r = fisher_z(r);
        let se = 1.0 / ((n as f64 - 3.0).max(1.0)).sqrt();
        let z_crit = normal_quantile(1.0 - alpha / 2.0);
        let lo = inverse_fisher_z(z_r - z_crit * se);
        let hi = inverse_fisher_z(z_r + z_crit * se);
        Self {
            point_estimate: r,
            lower: lo,
            upper: hi,
            level: 1.0 - alpha,
            method: "Fisher z-transform".into(),
        }
    }

    /// Convert to shared_types ConfidenceInterval.
    pub fn to_confidence_interval(&self) -> ConfidenceInterval {
        ConfidenceInterval::new(self.lower, self.upper, self.level, self.point_estimate)
    }
}

// ── Utility functions ───────────────────────────────────────────────────────

/// Approximate inverse normal CDF (Beasley-Springer-Moro algorithm).
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

    // Rational approximation
    let t = if p < 0.5 {
        (-2.0 * p.ln()).sqrt()
    } else {
        (-2.0 * (1.0 - p).ln()).sqrt()
    };

    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;

    let val = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);

    if p < 0.5 { -val } else { val }
}

/// Approximate normal CDF using the error function approximation.
pub fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Error function approximation (Abramowitz & Stegun 7.1.26).
fn erf(x: f64) -> f64 {
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t
        * (0.254829592
            + t * (-0.284496736
                + t * (1.421413741
                    + t * (-1.453152027 + t * 1.061405429))));
    sign * (1.0 - poly * (-x * x).exp())
}

/// Fisher z-transform: z = 0.5 * ln((1+r)/(1-r)).
pub fn fisher_z(r: f64) -> f64 {
    let r_clamped = r.clamp(-0.9999, 0.9999);
    0.5 * ((1.0 + r_clamped) / (1.0 - r_clamped)).ln()
}

/// Inverse Fisher z-transform.
pub fn inverse_fisher_z(z: f64) -> f64 {
    let e2z = (2.0 * z).exp();
    (e2z - 1.0) / (e2z + 1.0)
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_effect_size_interpretation_negligible() {
        assert_eq!(effect_size_interpretation(0.1), EffectInterpretation::Negligible);
        assert_eq!(effect_size_interpretation(-0.05), EffectInterpretation::Negligible);
    }

    #[test]
    fn test_effect_size_interpretation_small() {
        assert_eq!(effect_size_interpretation(0.3), EffectInterpretation::Small);
        assert_eq!(effect_size_interpretation(-0.4), EffectInterpretation::Small);
    }

    #[test]
    fn test_effect_size_interpretation_medium() {
        assert_eq!(effect_size_interpretation(0.6), EffectInterpretation::Medium);
    }

    #[test]
    fn test_effect_size_interpretation_large() {
        assert_eq!(effect_size_interpretation(1.0), EffectInterpretation::Large);
        assert_eq!(effect_size_interpretation(-2.0), EffectInterpretation::Large);
    }

    #[test]
    fn test_cohens_d_equal_groups() {
        let g1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let g2 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let d = CohenD::compute(&g1, &g2).unwrap();
        assert!(approx_eq(d.value, 0.0, 1e-10));
    }

    #[test]
    fn test_cohens_d_different_groups() {
        let g1 = vec![10.0, 11.0, 12.0, 13.0, 14.0];
        let g2 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let d = CohenD::compute(&g1, &g2).unwrap();
        assert!(d.value > 0.0);
        // Mean diff = 9, pooled SD = sqrt(2.5) ≈ 1.581, d ≈ 5.69
        assert!(d.value > 5.0);
        assert_eq!(d.interpretation, EffectInterpretation::Large);
    }

    #[test]
    fn test_cohens_d_one_sample() {
        let data = vec![5.0, 6.0, 7.0, 8.0, 9.0];
        let d = CohenD::one_sample(&data, 0.0).unwrap();
        // mean=7, sd=√2.5≈1.58, d≈4.43
        assert!(d.value > 4.0);
    }

    #[test]
    fn test_cohens_d_empty_error() {
        assert!(CohenD::compute(&[], &[1.0]).is_err());
        assert!(CohenD::one_sample(&[], 0.0).is_err());
    }

    #[test]
    fn test_hedges_g_correction() {
        let g1 = vec![10.0, 11.0, 12.0, 13.0, 14.0];
        let g2 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let d = CohenD::compute(&g1, &g2).unwrap();
        let g = HedgesG::compute(&g1, &g2).unwrap();
        // Hedges' g should be slightly smaller in magnitude due to correction
        assert!(g.value.abs() <= d.value.abs());
        assert!(g.correction_factor < 1.0);
        assert!(g.correction_factor > 0.8); // Not too much correction for n=10
    }

    #[test]
    fn test_glass_delta() {
        let treatment = vec![10.0, 11.0, 12.0, 13.0, 14.0];
        let control = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let delta = GlassDelta::compute(&treatment, &control).unwrap();
        // mean_diff = 9, control sd = sqrt(2.5) ≈ 1.581, delta ≈ 5.69
        assert!(delta.value > 5.0);
    }

    #[test]
    fn test_glass_delta_empty_error() {
        assert!(GlassDelta::compute(&[], &[1.0]).is_err());
    }

    #[test]
    fn test_point_biserial_r() {
        let g0 = vec![1.0, 2.0, 3.0];
        let g1 = vec![10.0, 11.0, 12.0];
        let r = PointBiserialR::compute(&g0, &g1).unwrap();
        assert!(r.value > 0.0);
    }

    #[test]
    fn test_point_biserial_to_d() {
        let g0 = vec![0.0; 50];
        let g1 = vec![1.0; 50];
        let r = PointBiserialR::compute(&g0, &g1).unwrap();
        let d = r.to_cohens_d();
        assert!(d > 0.0);
    }

    #[test]
    fn test_odds_ratio_basic() {
        let or = OddsRatio::compute(10, 5, 3, 12).unwrap();
        // OR = (10*12)/(5*3) = 8.0
        assert!(approx_eq(or.value, 8.0, 1e-10));
    }

    #[test]
    fn test_odds_ratio_with_zero_cell() {
        let or = OddsRatio::compute(0, 5, 3, 12).unwrap();
        // Should apply Haldane-Anscombe correction
        assert!(or.value > 0.0);
    }

    #[test]
    fn test_odds_ratio_ci() {
        let or = OddsRatio::compute(20, 10, 5, 25).unwrap();
        let ci = or.confidence_interval(0.05);
        assert!(ci.lower < or.value);
        assert!(ci.upper > or.value);
    }

    #[test]
    fn test_collusion_effect_size() {
        let prices = vec![3.5, 3.6, 3.4, 3.5, 3.7, 3.5, 3.6, 3.4, 3.5, 3.6];
        let nash = 2.0;
        let monopoly = 5.0;
        let es = CollusionEffectSize::compute(&prices, nash, monopoly).unwrap();
        assert!(es.collusion_premium > 0.0);
        assert!(es.collusion_premium < 1.0);
        assert!(es.value > 0.0);
    }

    #[test]
    fn test_collusion_effect_size_empty_error() {
        assert!(CollusionEffectSize::compute(&[], 2.0, 5.0).is_err());
    }

    #[test]
    fn test_power_analysis_two_sample() {
        let pa = PowerAnalysis::two_sample_t_power(0.8, 0.05, 30, 30);
        assert!(pa.power > 0.5);
        assert!(pa.power < 1.0);
    }

    #[test]
    fn test_power_analysis_required_n() {
        let pa = PowerAnalysis::required_sample_size(0.5, 0.05, 0.8);
        assert!(pa.required_n.is_some());
        assert!(pa.required_n.unwrap() > 10);
    }

    #[test]
    fn test_power_analysis_one_sample() {
        let pa = PowerAnalysis::one_sample_t_power(1.0, 0.05, 50);
        assert!(pa.power > 0.9);
    }

    #[test]
    fn test_detectable_effect() {
        let d = PowerAnalysis::detectable_effect(0.05, 0.8, 100);
        assert!(d > 0.0);
        assert!(d < 1.0);
    }

    #[test]
    fn test_effect_size_ci_cohens_d() {
        let g1 = vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let g2 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let d = CohenD::compute(&g1, &g2).unwrap();
        let ci = EffectSizeCI::for_cohens_d(&d, 6, 6, 0.05);
        assert!(ci.lower < d.value);
        assert!(ci.upper > d.value);
    }

    #[test]
    fn test_effect_size_ci_correlation() {
        let ci = EffectSizeCI::for_correlation(0.5, 50, 0.05);
        assert!(ci.lower < 0.5);
        assert!(ci.upper > 0.5);
        assert!(ci.lower > -1.0);
        assert!(ci.upper < 1.0);
    }

    #[test]
    fn test_fisher_z_roundtrip() {
        for r in [-0.9, -0.5, 0.0, 0.3, 0.7, 0.95] {
            let z = fisher_z(r);
            let r2 = inverse_fisher_z(z);
            assert!(approx_eq(r, r2, 1e-10), "r={r}, r2={r2}");
        }
    }

    #[test]
    fn test_normal_quantile_symmetry() {
        let q975 = normal_quantile(0.975);
        let q025 = normal_quantile(0.025);
        assert!(approx_eq(q975, -q025, 0.01));
        assert!(approx_eq(q975, 1.96, 0.02));
    }

    #[test]
    fn test_normal_cdf_basic() {
        assert!(approx_eq(normal_cdf(0.0), 0.5, 1e-10));
        assert!(normal_cdf(3.0) > 0.99);
        assert!(normal_cdf(-3.0) < 0.01);
    }
}
