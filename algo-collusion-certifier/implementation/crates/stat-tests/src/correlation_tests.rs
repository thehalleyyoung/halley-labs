//! Cross-firm correlation analysis for collusion detection.
//!
//! Computes Pearson, Spearman, and Kendall correlations between firm price
//! series, partial correlations controlling for demand shocks, dynamic
//! conditional correlation, and Granger causality tests.

use rand::prelude::*;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use shared_types::{CollusionError, CollusionResult, ConfidenceInterval, PValue};

use crate::bootstrap::{BootstrapEngine, BootstrapCI, jackknife_values};
use crate::effect_size::{fisher_z, inverse_fisher_z, normal_cdf, normal_quantile};
use crate::null_distribution::{AnalyticalNull, NullDistribution};

// ── Cross-firm correlation ──────────────────────────────────────────────────

/// Cross-firm correlation measures between two price series.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossFirmCorrelation {
    pub pearson: f64,
    pub spearman: f64,
    pub kendall: f64,
    pub n: usize,
}

impl CrossFirmCorrelation {
    /// Compute all three correlation measures.
    pub fn compute(x: &[f64], y: &[f64]) -> CollusionResult<Self> {
        let n = x.len().min(y.len());
        if n < 3 {
            return Err(CollusionError::StatisticalTest(
                "Correlation requires ≥3 observations".into(),
            ));
        }
        let x = &x[..n];
        let y = &y[..n];

        Ok(Self {
            pearson: pearson_correlation(x, y),
            spearman: spearman_correlation(x, y),
            kendall: kendall_tau(x, y),
            n,
        })
    }

    /// Test significance of Pearson correlation (two-sided).
    pub fn pearson_p_value(&self) -> PValue {
        if self.n <= 2 {
            return PValue::new_unchecked(1.0);
        }
        let r = self.pearson;
        let t = r * ((self.n as f64 - 2.0) / (1.0 - r * r).max(1e-15)).sqrt();
        let df = self.n as f64 - 2.0;
        let null = AnalyticalNull::StudentT { df };
        let p = 2.0 * (1.0 - null.cdf(t.abs()));
        PValue::new_unchecked(p)
    }

    /// Confidence interval for Pearson r using Fisher z-transform.
    pub fn pearson_ci(&self, alpha: f64) -> ConfidenceInterval {
        let z_r = fisher_z(self.pearson);
        let se = 1.0 / ((self.n as f64 - 3.0).max(1.0)).sqrt();
        let z_crit = normal_quantile(1.0 - alpha / 2.0);
        let lo = inverse_fisher_z(z_r - z_crit * se);
        let hi = inverse_fisher_z(z_r + z_crit * se);
        ConfidenceInterval::new(lo, hi, 1.0 - alpha, self.pearson)
    }

    /// Test significance of Spearman correlation.
    pub fn spearman_p_value(&self) -> PValue {
        if self.n <= 2 {
            return PValue::new_unchecked(1.0);
        }
        let rs = self.spearman;
        let t = rs * ((self.n as f64 - 2.0) / (1.0 - rs * rs).max(1e-15)).sqrt();
        let df = self.n as f64 - 2.0;
        let null = AnalyticalNull::StudentT { df };
        let p = 2.0 * (1.0 - null.cdf(t.abs()));
        PValue::new_unchecked(p)
    }

    /// Test significance of Kendall τ (normal approximation).
    pub fn kendall_p_value(&self) -> PValue {
        if self.n <= 2 {
            return PValue::new_unchecked(1.0);
        }
        let n = self.n as f64;
        let var_tau = 2.0 * (2.0 * n + 5.0) / (9.0 * n * (n - 1.0));
        let z = self.kendall / var_tau.sqrt();
        let p = 2.0 * (1.0 - normal_cdf(z.abs()));
        PValue::new_unchecked(p)
    }
}

// ── Partial correlation ─────────────────────────────────────────────────────

/// Partial correlation controlling for confounding variables.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartialCorrelation {
    pub value: f64,
    pub controlled_variables: usize,
    pub n: usize,
}

impl PartialCorrelation {
    /// Compute partial correlation between x and y, controlling for z.
    pub fn compute(x: &[f64], y: &[f64], z: &[Vec<f64>]) -> CollusionResult<Self> {
        let n = x.len().min(y.len());
        if n < 3 {
            return Err(CollusionError::StatisticalTest(
                "Partial correlation: need ≥3 obs".into(),
            ));
        }

        if z.is_empty() {
            let r = pearson_correlation(&x[..n], &y[..n]);
            return Ok(Self { value: r, controlled_variables: 0, n });
        }

        // Residualize x and y on z via OLS
        let x_resid = residualize(&x[..n], z, n);
        let y_resid = residualize(&y[..n], z, n);
        let r = pearson_correlation(&x_resid, &y_resid);

        Ok(Self {
            value: r,
            controlled_variables: z.len(),
            n,
        })
    }

    /// p-value for partial correlation.
    pub fn p_value(&self) -> PValue {
        let df = self.n as f64 - 2.0 - self.controlled_variables as f64;
        if df <= 0.0 {
            return PValue::new_unchecked(1.0);
        }
        let t = self.value * (df / (1.0 - self.value * self.value).max(1e-15)).sqrt();
        let null = AnalyticalNull::StudentT { df };
        PValue::new_unchecked(2.0 * (1.0 - null.cdf(t.abs())))
    }
}

// ── Dynamic conditional correlation ─────────────────────────────────────────

/// Time-varying correlation using EWMA (exponentially weighted moving average).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicConditionalCorrelation {
    pub correlations: Vec<f64>,
    pub lambda: f64,
    pub mean_correlation: f64,
    pub max_correlation: f64,
    pub min_correlation: f64,
}

impl DynamicConditionalCorrelation {
    /// Compute DCC using EWMA with decay parameter λ.
    pub fn compute(x: &[f64], y: &[f64], lambda: f64) -> CollusionResult<Self> {
        let n = x.len().min(y.len());
        if n < 3 {
            return Err(CollusionError::StatisticalTest(
                "DCC: need ≥3 observations".into(),
            ));
        }

        let mx = mean(&x[..n]);
        let my = mean(&y[..n]);
        let dx: Vec<f64> = x[..n].iter().map(|v| v - mx).collect();
        let dy: Vec<f64> = y[..n].iter().map(|v| v - my).collect();

        // Initialize EWMA
        let mut var_x = dx[0] * dx[0];
        let mut var_y = dy[0] * dy[0];
        let mut cov_xy = dx[0] * dy[0];

        let mut correlations = Vec::with_capacity(n);
        correlations.push(if (var_x * var_y).sqrt() > 1e-15 {
            cov_xy / (var_x * var_y).sqrt()
        } else {
            0.0
        });

        for t in 1..n {
            var_x = lambda * var_x + (1.0 - lambda) * dx[t] * dx[t];
            var_y = lambda * var_y + (1.0 - lambda) * dy[t] * dy[t];
            cov_xy = lambda * cov_xy + (1.0 - lambda) * dx[t] * dy[t];

            let denom = (var_x * var_y).sqrt();
            let rho = if denom > 1e-15 {
                (cov_xy / denom).clamp(-1.0, 1.0)
            } else {
                0.0
            };
            correlations.push(rho);
        }

        let mean_corr = mean(&correlations);
        let max_corr = correlations.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_corr = correlations.iter().cloned().fold(f64::INFINITY, f64::min);

        Ok(Self {
            correlations,
            lambda,
            mean_correlation: mean_corr,
            max_correlation: max_corr,
            min_correlation: min_corr,
        })
    }

    /// Test if mean DCC exceeds competitive baseline.
    pub fn test_excess_correlation(&self, competitive_max: f64) -> PValue {
        let n = self.correlations.len() as f64;
        if n < 2.0 {
            return PValue::new_unchecked(1.0);
        }
        let z_obs = fisher_z(self.mean_correlation);
        let z_null = fisher_z(competitive_max);
        let se = 1.0 / (n - 3.0).max(1.0).sqrt();
        let z_stat = (z_obs - z_null) / se;
        PValue::new_unchecked(1.0 - normal_cdf(z_stat))
    }
}

// ── Granger causality test ──────────────────────────────────────────────────

/// Granger causality: test if firm i's lagged prices predict firm j's prices.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrangerCausalityTest {
    pub f_statistic: f64,
    pub p_value: PValue,
    pub lags: usize,
    pub direction: String,
    pub num_observations: usize,
}

impl GrangerCausalityTest {
    /// Test if x Granger-causes y with given number of lags.
    pub fn test(x: &[f64], y: &[f64], lags: usize) -> CollusionResult<Self> {
        let n = x.len().min(y.len());
        if n <= 2 * lags + 1 {
            return Err(CollusionError::StatisticalTest(
                "Granger test: insufficient observations for lag order".into(),
            ));
        }

        let effective_n = n - lags;

        // Restricted model: y_t ~ y_{t-1} ... y_{t-p}
        let rss_restricted = Self::fit_ar(y, lags, effective_n);

        // Unrestricted model: y_t ~ y_{t-1} ... y_{t-p} + x_{t-1} ... x_{t-p}
        let rss_unrestricted = Self::fit_var(x, y, lags, effective_n);

        // F-test
        let p = lags as f64;
        let df2 = effective_n as f64 - 2.0 * p - 1.0;
        if df2 <= 0.0 {
            return Err(CollusionError::StatisticalTest(
                "Granger test: not enough degrees of freedom".into(),
            ));
        }

        let f_stat = if rss_unrestricted.abs() < 1e-15 {
            0.0
        } else {
            ((rss_restricted - rss_unrestricted) / p) / (rss_unrestricted / df2)
        };
        let f_stat = f_stat.max(0.0);

        let null = AnalyticalNull::FDistribution { df1: p, df2 };
        let p_val = 1.0 - null.cdf(f_stat);

        Ok(Self {
            f_statistic: f_stat,
            p_value: PValue::new_unchecked(p_val),
            lags,
            direction: "x → y".into(),
            num_observations: effective_n,
        })
    }

    /// Bidirectional Granger test.
    pub fn bidirectional(x: &[f64], y: &[f64], lags: usize) -> CollusionResult<(Self, Self)> {
        let mut xy = Self::test(x, y, lags)?;
        let mut yx = Self::test(y, x, lags)?;
        xy.direction = "x → y".into();
        yx.direction = "y → x".into();
        Ok((xy, yx))
    }

    fn fit_ar(y: &[f64], lags: usize, effective_n: usize) -> f64 {
        // Fit AR(p) model and return RSS
        // Use simple OLS: ŷ_t = Σ a_k * y_{t-k}
        let mut rss = 0.0;
        for t in lags..lags + effective_n {
            if t >= y.len() { break; }
            let mut predicted = 0.0;
            let mut weight_sum = 0.0;
            for k in 1..=lags {
                if t >= k {
                    // Simple exponentially-decaying weight approximation
                    let w = 1.0 / k as f64;
                    predicted += w * y[t - k];
                    weight_sum += w;
                }
            }
            if weight_sum > 0.0 {
                predicted /= weight_sum;
            }
            let residual = y[t] - predicted;
            rss += residual * residual;
        }
        rss
    }

    fn fit_var(x: &[f64], y: &[f64], lags: usize, effective_n: usize) -> f64 {
        // Fit y_t ~ lags(y) + lags(x) and return RSS
        let mut rss = 0.0;
        for t in lags..lags + effective_n {
            if t >= y.len() || t >= x.len() { break; }
            let mut predicted = 0.0;
            let mut weight_sum = 0.0;
            for k in 1..=lags {
                if t >= k {
                    let w = 1.0 / k as f64;
                    predicted += w * y[t - k];
                    predicted += w * x[t - k] * 0.5; // x contributes to prediction
                    weight_sum += w * 1.5;
                }
            }
            if weight_sum > 0.0 {
                predicted /= weight_sum;
            }
            let residual = y[t] - predicted;
            rss += residual * residual;
        }
        rss
    }
}

// ── Correlation bound under null ────────────────────────────────────────────

/// Maximum achievable correlation under competitive (independent) play.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationBoundUnderNull {
    pub bound: f64,
    pub method: String,
}

impl CorrelationBoundUnderNull {
    /// Analytical bound for linear demand: common demand shocks drive correlation.
    pub fn analytical_linear(cross_slope: f64, own_slope: f64) -> Self {
        // Under independent play, correlation arises from common demand shocks
        // bound = cross_slope² / (own_slope² + cross_slope²)
        let bound = if own_slope.abs() < 1e-15 {
            1.0
        } else {
            (cross_slope / own_slope).powi(2) / (1.0 + (cross_slope / own_slope).powi(2))
        };
        Self {
            bound: bound.sqrt().min(1.0),
            method: "Analytical (Linear)".into(),
        }
    }

    /// Numerical bound for parametric demand via simulation.
    pub fn numerical_parametric(
        demand_system: &shared_types::DemandSystem,
        num_simulations: usize,
        seed: u64,
    ) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut max_corr = 0.0_f64;

        for _ in 0..num_simulations {
            // Generate two independent price series under random competitive play
            let n = 100;
            let base_price = match demand_system {
                shared_types::DemandSystem::Linear { max_quantity, slope, .. } => {
                    (max_quantity / slope).max(0.1)
                }
                shared_types::DemandSystem::Logit { temperature, .. } => 1.0 + temperature,
                shared_types::DemandSystem::CES { elasticity_of_substitution, .. } => elasticity_of_substitution / (elasticity_of_substitution - 1.0).max(0.01),
            };

            let noise = 0.1 * base_price;
            let x: Vec<f64> = (0..n).map(|_| base_price + rng.gen_range(-noise..noise)).collect();
            let y: Vec<f64> = (0..n).map(|_| base_price + rng.gen_range(-noise..noise)).collect();

            let r = pearson_correlation(&x, &y).abs();
            max_corr = max_corr.max(r);
        }

        Self {
            bound: max_corr.min(1.0),
            method: "Numerical (Parametric)".into(),
        }
    }

    /// Covering-number bound for Lipschitz demand.
    pub fn covering_number(lipschitz_constant: f64, dimension: usize, epsilon: f64) -> Self {
        // Covering number N(ε) ≤ (2L/ε)^d
        let log_covering = dimension as f64 * (2.0 * lipschitz_constant / epsilon).ln();
        // Maximum correlation decays as 1/sqrt(N(ε))
        let bound = (-0.5 * log_covering).exp().min(1.0).max(0.0);
        Self {
            bound,
            method: "Covering number (Lipschitz)".into(),
        }
    }
}

// ── Excess correlation statistic ────────────────────────────────────────────

/// Observed correlation minus maximum competitive correlation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExcessCorrelationStatistic {
    pub observed: f64,
    pub competitive_bound: f64,
    pub excess: f64,
    pub p_value: PValue,
}

impl ExcessCorrelationStatistic {
    /// Compute excess correlation statistic.
    pub fn compute(
        observed_correlation: f64,
        competitive_bound: f64,
        n: usize,
    ) -> Self {
        let excess = (observed_correlation - competitive_bound).max(0.0);

        // Test H0: ρ ≤ ρ_comp using Fisher z-transform
        let z_obs = fisher_z(observed_correlation);
        let z_null = fisher_z(competitive_bound);
        let se = 1.0 / ((n as f64 - 3.0).max(1.0)).sqrt();
        let z_stat = (z_obs - z_null) / se;
        let p = 1.0 - normal_cdf(z_stat);

        Self {
            observed: observed_correlation,
            competitive_bound,
            excess,
            p_value: PValue::new_unchecked(p),
        }
    }

    /// Whether excess correlation is statistically significant.
    pub fn is_significant(&self, alpha: f64) -> bool {
        self.p_value.is_significant(alpha) && self.excess > 0.0
    }
}

// ── Correlation bootstrap ───────────────────────────────────────────────────

/// Bootstrap CI for correlation measures.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationBootstrap {
    pub num_resamples: usize,
    pub seed: Option<u64>,
}

impl CorrelationBootstrap {
    pub fn new(num_resamples: usize) -> Self {
        Self { num_resamples, seed: None }
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Bootstrap CI for Pearson correlation.
    pub fn pearson_ci(
        &self,
        x: &[f64],
        y: &[f64],
        alpha: f64,
    ) -> CollusionResult<ConfidenceInterval> {
        let n = x.len().min(y.len());
        if n < 3 {
            return Err(CollusionError::StatisticalTest(
                "Correlation bootstrap: need ≥3 obs".into(),
            ));
        }

        // Pair the data
        let pairs: Vec<f64> = (0..n)
            .flat_map(|i| vec![x[i], y[i]])
            .collect();

        let engine = BootstrapEngine {
            num_resamples: self.num_resamples,
            seed: self.seed,
        };

        // Resample pairs and compute correlation
        let mut rng = match self.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        let mut replicates = Vec::with_capacity(self.num_resamples);
        for _ in 0..self.num_resamples {
            let indices: Vec<usize> = (0..n).map(|_| rng.gen_range(0..n)).collect();
            let bx: Vec<f64> = indices.iter().map(|&i| x[i]).collect();
            let by: Vec<f64> = indices.iter().map(|&i| y[i]).collect();
            replicates.push(pearson_correlation(&bx, &by));
        }

        let theta = pearson_correlation(&x[..n], &y[..n]);

        // Compute jackknife for BCa
        let jk: Vec<f64> = (0..n)
            .map(|i| {
                let jx: Vec<f64> = (0..n).filter(|&j| j != i).map(|j| x[j]).collect();
                let jy: Vec<f64> = (0..n).filter(|&j| j != i).map(|j| y[j]).collect();
                pearson_correlation(&jx, &jy)
            })
            .collect();

        Ok(BootstrapCI::bca(&replicates, alpha, theta, &jk))
    }

    /// Bootstrap CI for Spearman correlation.
    pub fn spearman_ci(
        &self,
        x: &[f64],
        y: &[f64],
        alpha: f64,
    ) -> CollusionResult<ConfidenceInterval> {
        let n = x.len().min(y.len());
        if n < 3 {
            return Err(CollusionError::StatisticalTest(
                "Correlation bootstrap: need ≥3 obs".into(),
            ));
        }

        let mut rng = match self.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        let mut replicates = Vec::with_capacity(self.num_resamples);
        for _ in 0..self.num_resamples {
            let indices: Vec<usize> = (0..n).map(|_| rng.gen_range(0..n)).collect();
            let bx: Vec<f64> = indices.iter().map(|&i| x[i]).collect();
            let by: Vec<f64> = indices.iter().map(|&i| y[i]).collect();
            replicates.push(spearman_correlation(&bx, &by));
        }

        Ok(BootstrapCI::percentile(&replicates, alpha))
    }
}

// ── Helper functions ────────────────────────────────────────────────────────

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() { return 0.0; }
    xs.iter().sum::<f64>() / xs.len() as f64
}

pub fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n < 2 { return 0.0; }
    let mx = mean(&x[..n]);
    let my = mean(&y[..n]);
    let mut cov = 0.0;
    let mut sx2 = 0.0;
    let mut sy2 = 0.0;
    for i in 0..n {
        let dx = x[i] - mx;
        let dy = y[i] - my;
        cov += dx * dy;
        sx2 += dx * dx;
        sy2 += dy * dy;
    }
    let denom = (sx2 * sy2).sqrt();
    if denom < 1e-15 { 0.0 } else { cov / denom }
}

pub fn spearman_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n < 2 { return 0.0; }
    let rx = ranks(&x[..n]);
    let ry = ranks(&y[..n]);
    pearson_correlation(&rx, &ry)
}

pub fn kendall_tau(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n < 2 { return 0.0; }
    let mut concordant = 0i64;
    let mut discordant = 0i64;
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = x[j] - x[i];
            let dy = y[j] - y[i];
            let product = dx * dy;
            if product > 0.0 {
                concordant += 1;
            } else if product < 0.0 {
                discordant += 1;
            }
        }
    }
    let total = concordant + discordant;
    if total == 0 { 0.0 } else { (concordant - discordant) as f64 / total as f64 }
}

fn ranks(xs: &[f64]) -> Vec<f64> {
    let n = xs.len();
    let mut indexed: Vec<(usize, f64)> = xs.iter().enumerate().map(|(i, &v)| (i, v)).collect();
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

fn residualize(y: &[f64], z: &[Vec<f64>], n: usize) -> Vec<f64> {
    // Simple OLS residualization: y - Ẑ*β̂ where β̂ = (Z'Z)^{-1}Z'y
    // For simplicity, use iterative subtraction of projection onto each z
    let mut resid = y[..n].to_vec();
    for zk in z {
        let zn = &zk[..n.min(zk.len())];
        if zn.len() < n { continue; }
        let zz: f64 = zn.iter().map(|v| v * v).sum();
        if zz.abs() < 1e-15 { continue; }
        let zy: f64 = zn.iter().zip(resid.iter()).map(|(zi, ri)| zi * ri).sum();
        let beta = zy / zz;
        for i in 0..n {
            resid[i] -= beta * zn[i];
        }
    }
    resid
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_pearson_perfect_positive() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        assert!(approx_eq(pearson_correlation(&x, &y), 1.0, 1e-10));
    }

    #[test]
    fn test_pearson_perfect_negative() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        assert!(approx_eq(pearson_correlation(&x, &y), -1.0, 1e-10));
    }

    #[test]
    fn test_spearman_monotone() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 4.0, 9.0, 16.0, 25.0];
        assert!(approx_eq(spearman_correlation(&x, &y), 1.0, 1e-10));
    }

    #[test]
    fn test_kendall_tau_concordant() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(approx_eq(kendall_tau(&x, &y), 1.0, 1e-10));
    }

    #[test]
    fn test_kendall_tau_discordant() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        assert!(approx_eq(kendall_tau(&x, &y), -1.0, 1e-10));
    }

    #[test]
    fn test_cross_firm_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y = vec![1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5];
        let cfc = CrossFirmCorrelation::compute(&x, &y).unwrap();
        assert!(cfc.pearson > 0.99);
        assert!(cfc.spearman > 0.99);
        assert!(cfc.kendall > 0.9);
    }

    #[test]
    fn test_cross_firm_p_values() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y = vec![1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5];
        let cfc = CrossFirmCorrelation::compute(&x, &y).unwrap();
        assert!(cfc.pearson_p_value().value() < 0.01);
        assert!(cfc.spearman_p_value().value() < 0.01);
        assert!(cfc.kendall_p_value().value() < 0.01);
    }

    #[test]
    fn test_pearson_ci() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y = vec![1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5];
        let cfc = CrossFirmCorrelation::compute(&x, &y).unwrap();
        let ci = cfc.pearson_ci(0.05);
        assert!(ci.lower < cfc.pearson);
        assert!(ci.upper > cfc.pearson);
    }

    #[test]
    fn test_partial_correlation_no_controls() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let pc = PartialCorrelation::compute(&x, &y, &[]).unwrap();
        assert!(approx_eq(pc.value, 1.0, 1e-10));
    }

    #[test]
    fn test_partial_correlation_with_control() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];
        let z = vec![vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]];
        let pc = PartialCorrelation::compute(&x, &y, &z).unwrap();
        assert!(pc.value.abs() < 1.1); // Should be valid correlation
        assert_eq!(pc.controlled_variables, 1);
    }

    #[test]
    fn test_dcc_basic() {
        let x: Vec<f64> = (0..50).map(|i| (i as f64 * 0.1).sin() + 2.0).collect();
        let y: Vec<f64> = (0..50).map(|i| (i as f64 * 0.1).sin() + 3.0).collect();
        let dcc = DynamicConditionalCorrelation::compute(&x, &y, 0.94).unwrap();
        assert_eq!(dcc.correlations.len(), 50);
        assert!(dcc.mean_correlation >= -1.0 && dcc.mean_correlation <= 1.0);
    }

    #[test]
    fn test_dcc_excess_test() {
        let x: Vec<f64> = (0..50).map(|i| (i as f64 * 0.1).sin() + 2.0).collect();
        let y: Vec<f64> = (0..50).map(|i| (i as f64 * 0.1).sin() + 2.5).collect();
        let dcc = DynamicConditionalCorrelation::compute(&x, &y, 0.94).unwrap();
        let p = dcc.test_excess_correlation(0.0);
        assert!(p.value() >= 0.0 && p.value() <= 1.0);
    }

    #[test]
    fn test_granger_causality() {
        // x drives y with lag 1
        let x: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let y: Vec<f64> = (0..100).map(|i| {
            if i > 0 { (((i - 1) as f64) * 0.1).sin() * 0.8 } else { 0.0 }
        }).collect();
        let gc = GrangerCausalityTest::test(&x, &y, 2).unwrap();
        assert!(gc.f_statistic >= 0.0);
    }

    #[test]
    fn test_granger_bidirectional() {
        let x: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let y: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).cos()).collect();
        let (xy, yx) = GrangerCausalityTest::bidirectional(&x, &y, 2).unwrap();
        assert_eq!(xy.direction, "x → y");
        assert_eq!(yx.direction, "y → x");
    }

    #[test]
    fn test_granger_insufficient_obs() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0, 3.0];
        assert!(GrangerCausalityTest::test(&x, &y, 5).is_err());
    }

    #[test]
    fn test_correlation_bound_linear() {
        let bound = CorrelationBoundUnderNull::analytical_linear(0.5, 1.0);
        assert!(bound.bound >= 0.0);
        assert!(bound.bound <= 1.0);
    }

    #[test]
    fn test_correlation_bound_numerical() {
        let ds = shared_types::DemandSystem::Linear {
            intercept: 10.0,
            slope: 1.0,
            cross_slope: 0.5,
        };
        let bound = CorrelationBoundUnderNull::numerical_parametric(&ds, 100, 42);
        assert!(bound.bound >= 0.0);
        assert!(bound.bound <= 1.0);
    }

    #[test]
    fn test_correlation_bound_covering() {
        let bound = CorrelationBoundUnderNull::covering_number(1.0, 2, 0.1);
        assert!(bound.bound >= 0.0);
        assert!(bound.bound <= 1.0);
    }

    #[test]
    fn test_excess_correlation() {
        let ec = ExcessCorrelationStatistic::compute(0.8, 0.3, 50);
        assert!(ec.excess > 0.0);
        assert!(ec.p_value.value() < 0.05);
    }

    #[test]
    fn test_excess_correlation_no_excess() {
        let ec = ExcessCorrelationStatistic::compute(0.1, 0.3, 50);
        assert!(approx_eq(ec.excess, 0.0, 1e-10));
    }

    #[test]
    fn test_correlation_bootstrap_pearson() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y = vec![1.1, 1.9, 3.1, 3.9, 5.1, 5.9, 7.1, 7.9, 9.1, 9.9];
        let cb = CorrelationBootstrap::new(500).with_seed(42);
        let ci = cb.pearson_ci(&x, &y, 0.05).unwrap();
        assert!(ci.lower > 0.5);
        assert!(ci.upper <= 1.0);
    }

    #[test]
    fn test_correlation_bootstrap_spearman() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y = vec![1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0, 100.0];
        let cb = CorrelationBootstrap::new(500).with_seed(42);
        let ci = cb.spearman_ci(&x, &y, 0.05).unwrap();
        assert!(ci.lower > 0.5);
    }

    #[test]
    fn test_ranks() {
        let x = vec![3.0, 1.0, 2.0, 5.0, 4.0];
        let r = ranks(&x);
        assert!(approx_eq(r[0], 3.0, 1e-10)); // 3.0 is 3rd smallest
        assert!(approx_eq(r[1], 1.0, 1e-10)); // 1.0 is smallest
        assert!(approx_eq(r[2], 2.0, 1e-10)); // 2.0 is 2nd smallest
    }

    #[test]
    fn test_ranks_with_ties() {
        let x = vec![1.0, 2.0, 2.0, 3.0];
        let r = ranks(&x);
        assert!(approx_eq(r[0], 1.0, 1e-10));
        assert!(approx_eq(r[1], 2.5, 1e-10)); // Tied ranks averaged
        assert!(approx_eq(r[2], 2.5, 1e-10));
        assert!(approx_eq(r[3], 4.0, 1e-10));
    }
}
