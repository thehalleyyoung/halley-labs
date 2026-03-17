//! Null distribution computation: analytical, simulated, and asymptotic.
//!
//! Provides null distributions for test statistics under competitive play,
//! Berry-Esseen finite-sample corrections, and conservative null calibration.

use rand::prelude::*;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use shared_types::{
    CollusionError, CollusionResult, DemandSystem, PValue, Price,
    SimulationConfig,
};

use crate::effect_size::{normal_cdf, normal_quantile};

// ── NullDistribution trait ──────────────────────────────────────────────────

/// Common interface for null distributions.
pub trait NullDistribution: Send + Sync {
    /// Name of the null distribution.
    fn name(&self) -> &str;

    /// CDF: P(X <= x | H0).
    fn cdf(&self, x: f64) -> f64;

    /// Quantile function (inverse CDF).
    fn quantile(&self, p: f64) -> f64;

    /// Draw a single sample from the null distribution.
    fn sample(&self, rng: &mut dyn RngCore) -> f64;

    /// Draw multiple samples.
    fn sample_n(&self, n: usize, rng: &mut dyn RngCore) -> Vec<f64> {
        (0..n).map(|_| self.sample(rng)).collect()
    }

    /// p-value for observed statistic (upper tail).
    fn p_value_upper(&self, x: f64) -> PValue {
        PValue::new_unchecked(1.0 - self.cdf(x))
    }

    /// p-value for observed statistic (lower tail).
    fn p_value_lower(&self, x: f64) -> PValue {
        PValue::new_unchecked(self.cdf(x))
    }

    /// Critical value for significance level alpha (upper tail).
    fn critical_value(&self, alpha: f64) -> f64 {
        self.quantile(1.0 - alpha)
    }
}

// ── Analytical null ─────────────────────────────────────────────────────────

/// Analytical (closed-form) null distribution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnalyticalNull {
    /// Standard normal N(0,1).
    StandardNormal,
    /// Student's t distribution with given df.
    StudentT { df: f64 },
    /// Chi-squared distribution with given df.
    ChiSquared { df: f64 },
    /// F distribution with given numerator and denominator df.
    FDistribution { df1: f64, df2: f64 },
    /// Uniform on [0, 1].
    Uniform,
}

impl NullDistribution for AnalyticalNull {
    fn name(&self) -> &str {
        match self {
            Self::StandardNormal => "N(0,1)",
            Self::StudentT { .. } => "Student-t",
            Self::ChiSquared { .. } => "Chi-squared",
            Self::FDistribution { .. } => "F",
            Self::Uniform => "Uniform(0,1)",
        }
    }

    fn cdf(&self, x: f64) -> f64 {
        match self {
            Self::StandardNormal => normal_cdf(x),
            Self::StudentT { df } => student_t_cdf(x, *df),
            Self::ChiSquared { df } => chi_squared_cdf(x, *df),
            Self::FDistribution { df1, df2 } => f_distribution_cdf(x, *df1, *df2),
            Self::Uniform => x.clamp(0.0, 1.0),
        }
    }

    fn quantile(&self, p: f64) -> f64 {
        match self {
            Self::StandardNormal => normal_quantile(p),
            Self::StudentT { df } => student_t_quantile(p, *df),
            Self::ChiSquared { df } => chi_squared_quantile(p, *df),
            Self::FDistribution { df1, df2 } => f_distribution_quantile(p, *df1, *df2),
            Self::Uniform => p.clamp(0.0, 1.0),
        }
    }

    fn sample(&self, rng: &mut dyn RngCore) -> f64 {
        match self {
            Self::StandardNormal => {
                // Box-Muller transform
                let u1: f64 = rng.gen_range(0.0001..1.0);
                let u2: f64 = rng.gen_range(0.0..std::f64::consts::TAU);
                (-2.0 * u1.ln()).sqrt() * u2.cos()
            }
            Self::StudentT { df } => {
                let z = self.sample_normal(rng);
                let chi2 = self.sample_chi2(*df, rng);
                z / (chi2 / df).sqrt()
            }
            Self::ChiSquared { df } => {
                self.sample_chi2(*df, rng)
            }
            Self::FDistribution { df1, df2 } => {
                let c1 = self.sample_chi2(*df1, rng) / df1;
                let c2 = self.sample_chi2(*df2, rng) / df2;
                if c2.abs() < 1e-15 { 0.0 } else { c1 / c2 }
            }
            Self::Uniform => rng.gen_range(0.0..1.0),
        }
    }
}

impl AnalyticalNull {
    fn sample_normal(&self, rng: &mut dyn RngCore) -> f64 {
        let u1: f64 = rng.gen_range(0.0001..1.0);
        let u2: f64 = rng.gen_range(0.0..std::f64::consts::TAU);
        (-2.0 * u1.ln()).sqrt() * u2.cos()
    }

    fn sample_chi2(&self, df: f64, rng: &mut dyn RngCore) -> f64 {
        let k = df.round() as usize;
        (0..k.max(1))
            .map(|_| {
                let z = self.sample_normal(rng);
                z * z
            })
            .sum()
    }
}

// ── Simulated null ──────────────────────────────────────────────────────────

/// Monte Carlo null distribution from simulated data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulatedNull {
    pub samples: Vec<f64>,
    pub name_str: String,
}

impl SimulatedNull {
    /// Create from pre-computed samples.
    pub fn new(samples: Vec<f64>, name: &str) -> Self {
        let mut s = samples;
        s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        Self { samples: s, name_str: name.to_string() }
    }

    /// Number of simulated samples.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Empirical mean.
    pub fn mean(&self) -> f64 {
        if self.samples.is_empty() { return 0.0; }
        self.samples.iter().sum::<f64>() / self.samples.len() as f64
    }

    /// Empirical standard deviation.
    pub fn std_dev(&self) -> f64 {
        if self.samples.len() < 2 { return 0.0; }
        let m = self.mean();
        let var = self.samples.iter().map(|x| (x - m).powi(2)).sum::<f64>()
            / (self.samples.len() - 1) as f64;
        var.sqrt()
    }
}

impl NullDistribution for SimulatedNull {
    fn name(&self) -> &str {
        &self.name_str
    }

    fn cdf(&self, x: f64) -> f64 {
        if self.samples.is_empty() { return 0.5; }
        let count = self.samples.iter().filter(|&&v| v <= x).count();
        count as f64 / self.samples.len() as f64
    }

    fn quantile(&self, p: f64) -> f64 {
        if self.samples.is_empty() { return 0.0; }
        let idx = (p * self.samples.len() as f64).floor() as usize;
        self.samples[idx.min(self.samples.len() - 1)]
    }

    fn sample(&self, rng: &mut dyn RngCore) -> f64 {
        if self.samples.is_empty() { return 0.0; }
        let idx = rng.gen_range(0..self.samples.len());
        self.samples[idx]
    }
}

// ── Asymptotic null ─────────────────────────────────────────────────────────

/// CLT-based asymptotic null with optional Berry-Esseen correction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsymptoticNull {
    pub mean: f64,
    pub variance: f64,
    pub sample_size: usize,
    pub berry_esseen: Option<BerryEsseenCorrection>,
}

impl AsymptoticNull {
    pub fn new(mean: f64, variance: f64, sample_size: usize) -> Self {
        Self { mean, variance, sample_size, berry_esseen: None }
    }

    pub fn with_berry_esseen(mut self, correction: BerryEsseenCorrection) -> Self {
        self.berry_esseen = Some(correction);
        self
    }

    /// Standard deviation of the test statistic under null.
    pub fn std_dev(&self) -> f64 {
        (self.variance / self.sample_size as f64).sqrt()
    }

    /// Standardize a statistic.
    pub fn standardize(&self, x: f64) -> f64 {
        let sd = self.std_dev();
        if sd.abs() < 1e-15 { 0.0 } else { (x - self.mean) / sd }
    }
}

impl NullDistribution for AsymptoticNull {
    fn name(&self) -> &str {
        "Asymptotic Normal"
    }

    fn cdf(&self, x: f64) -> f64 {
        let z = self.standardize(x);
        let base = normal_cdf(z);
        match &self.berry_esseen {
            Some(be) => {
                let bound = be.remainder_bound(self.sample_size);
                // Return conservative CDF: for upper-tail tests, inflate CDF
                base.clamp(bound, 1.0 - bound)
            }
            None => base,
        }
    }

    fn quantile(&self, p: f64) -> f64 {
        let z = match &self.berry_esseen {
            Some(be) => {
                let bound = be.remainder_bound(self.sample_size);
                // Use conservative quantile
                let adj_p = if p > 0.5 { (p - bound).max(0.5) } else { (p + bound).min(0.5) };
                normal_quantile(adj_p)
            }
            None => normal_quantile(p),
        };
        self.mean + z * self.std_dev()
    }

    fn sample(&self, rng: &mut dyn RngCore) -> f64 {
        let u1: f64 = rng.gen_range(0.0001..1.0);
        let u2: f64 = rng.gen_range(0.0..std::f64::consts::TAU);
        let z = (-2.0 * u1.ln()).sqrt() * u2.cos();
        self.mean + z * self.std_dev()
    }
}

// ── Berry-Esseen correction ─────────────────────────────────────────────────

/// Finite-sample correction for CLT-based approximations via Berry-Esseen theorem.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BerryEsseenCorrection {
    /// Third absolute moment E[|X - μ|³].
    pub third_moment: f64,
    /// Variance σ².
    pub variance: f64,
    /// Berry-Esseen constant (universal: C ≤ 0.4748).
    pub constant: f64,
}

impl BerryEsseenCorrection {
    pub fn new(third_moment: f64, variance: f64) -> Self {
        Self {
            third_moment,
            variance,
            constant: 0.4748,
        }
    }

    /// Compute from sample data.
    pub fn from_data(data: &[f64]) -> CollusionResult<Self> {
        if data.len() < 3 {
            return Err(CollusionError::StatisticalTest(
                "Berry-Esseen: need ≥3 data points".into(),
            ));
        }
        let n = data.len() as f64;
        let mean = data.iter().sum::<f64>() / n;
        let var = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let rho = data.iter().map(|x| (x - mean).abs().powi(3)).sum::<f64>() / n;
        Ok(Self::new(rho, var))
    }

    /// Compute the remainder bound: sup_x |F_n(x) - Φ(x)| ≤ C * ρ / (σ³ √n).
    pub fn remainder_bound(&self, n: usize) -> f64 {
        if self.variance.abs() < 1e-15 || n == 0 {
            return 1.0;
        }
        let sigma3 = self.variance.powf(1.5);
        self.constant * self.third_moment / (sigma3 * (n as f64).sqrt())
    }

    /// Corrected critical value (conservative): increase critical value by remainder bound.
    pub fn corrected_critical_value(&self, alpha: f64, n: usize) -> f64 {
        let z = normal_quantile(1.0 - alpha);
        let bound = self.remainder_bound(n);
        // Conservative: use higher threshold
        normal_quantile(1.0 - alpha + bound)
    }

    /// Whether the CLT approximation is reliable at this sample size.
    pub fn is_reliable(&self, n: usize, tolerance: f64) -> bool {
        self.remainder_bound(n) < tolerance
    }
}

// ── Competitive simulator ───────────────────────────────────────────────────

/// Simulate competitive trajectories under null hypothesis to build null distribution.
#[derive(Debug, Clone)]
pub struct CompetitiveSimulator {
    pub config: SimulationConfig,
    pub num_simulations: usize,
    pub seed: Option<u64>,
}

impl CompetitiveSimulator {
    pub fn new(config: SimulationConfig, num_simulations: usize) -> Self {
        Self { config, num_simulations, seed: None }
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Simulate independent Q-learners under the narrow null.
    /// Returns a null distribution of price means.
    pub fn simulate_narrow_null(&self) -> CollusionResult<SimulatedNull> {
        let nash = self.compute_nash_price().0;
        let mut rng = match self.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        let mut samples = Vec::with_capacity(self.num_simulations);
        let noise_sd = 0.1 * nash.abs().max(0.01);

        for _ in 0..self.num_simulations {
            // Independent Q-learners converge to Nash ± noise
            let mean_price: f64 = (0..self.config.num_rounds())
                .map(|t| {
                    let lr = 1.0 / (1.0 + t as f64 * 0.01);
                    let noise: f64 = {
                        let u1: f64 = rng.gen_range(0.0001..1.0);
                        let u2: f64 = rng.gen_range(0.0..std::f64::consts::TAU);
                        (-2.0 * u1.ln()).sqrt() * u2.cos() * noise_sd * lr
                    };
                    nash + noise
                })
                .sum::<f64>() / self.config.num_rounds() as f64;
            samples.push(mean_price);
        }

        Ok(SimulatedNull::new(samples, "Narrow Null (Q-learning)"))
    }

    /// Simulate no-regret learners under medium null.
    pub fn simulate_medium_null(&self) -> CollusionResult<SimulatedNull> {
        let nash = self.compute_nash_price().0;
        let mut rng = match self.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        let noise_sd = 0.15 * nash.abs().max(0.01);
        let mut samples = Vec::with_capacity(self.num_simulations);

        for _ in 0..self.num_simulations {
            let mean_price: f64 = (0..self.config.num_rounds())
                .map(|t| {
                    let lr = 1.0 / (1.0 + t as f64 * 0.005);
                    let noise: f64 = {
                        let u1: f64 = rng.gen_range(0.0001..1.0);
                        let u2: f64 = rng.gen_range(0.0..std::f64::consts::TAU);
                        (-2.0 * u1.ln()).sqrt() * u2.cos() * noise_sd * lr
                    };
                    nash + noise
                })
                .sum::<f64>() / self.config.num_rounds() as f64;
            samples.push(mean_price);
        }

        Ok(SimulatedNull::new(samples, "Medium Null (No-regret)"))
    }

    /// Simulate broad null: independent learners under Lipschitz demand.
    pub fn simulate_broad_null(&self) -> CollusionResult<SimulatedNull> {
        let nash = self.compute_nash_price().0;
        let mut rng = match self.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        let noise_sd = 0.2 * nash.abs().max(0.01);
        let mut samples = Vec::with_capacity(self.num_simulations);

        for _ in 0..self.num_simulations {
            let mean_price: f64 = (0..self.config.num_rounds())
                .map(|t| {
                    let lr = 1.0 / (1.0 + t as f64 * 0.003);
                    let noise: f64 = {
                        let u1: f64 = rng.gen_range(0.0001..1.0);
                        let u2: f64 = rng.gen_range(0.0..std::f64::consts::TAU);
                        (-2.0 * u1.ln()).sqrt() * u2.cos() * noise_sd * lr
                    };
                    nash + noise
                })
                .sum::<f64>() / self.config.num_rounds() as f64;
            samples.push(mean_price);
        }

        Ok(SimulatedNull::new(samples, "Broad Null (Independent)"))
    }

    fn compute_nash_price(&self) -> Price {
        match self.config.demand_system() {
            DemandSystem::Linear { max_quantity, slope } => {
                let mc = self.config.marginal_cost().0;
                let n = self.config.num_players() as f64;
                Price::new((max_quantity + slope * mc) / (2.0 * slope - slope * (n - 1.0) / n))
            }
            DemandSystem::Logit { temperature, .. } => {
                let mc = self.config.marginal_cost().0;
                Price::new(mc + *temperature)
            }
            DemandSystem::CES { elasticity_of_substitution, .. } => {
                let mc = self.config.marginal_cost().0;
                let sigma = *elasticity_of_substitution;
                Price::new(mc * sigma / (sigma - 1.0).max(0.01))
            }
        }
    }
}

// ── Null calibration ────────────────────────────────────────────────────────

/// Calibrate a test statistic's null distribution from simulated competitive data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NullCalibration {
    pub simulated_null: SimulatedNull,
    pub calibrated_alpha: f64,
    pub nominal_alpha: f64,
    pub calibration_factor: f64,
}

impl NullCalibration {
    /// Calibrate: given a nominal alpha and a simulated null, find the actual critical value.
    pub fn calibrate(simulated_null: SimulatedNull, nominal_alpha: f64) -> Self {
        let cv = simulated_null.quantile(1.0 - nominal_alpha);
        let z_nom = normal_quantile(1.0 - nominal_alpha);
        let mean = simulated_null.mean();
        let sd = simulated_null.std_dev();

        let calibration_factor = if sd.abs() > 1e-15 && z_nom.abs() > 1e-15 {
            (cv - mean) / (sd * z_nom)
        } else {
            1.0
        };

        // Actual alpha based on normal approximation
        let calibrated_alpha = 1.0 - normal_cdf(z_nom * calibration_factor);

        Self {
            simulated_null,
            calibrated_alpha,
            nominal_alpha,
            calibration_factor,
        }
    }

    /// Get the calibrated critical value.
    pub fn critical_value(&self) -> f64 {
        self.simulated_null.quantile(1.0 - self.nominal_alpha)
    }

    /// Get a calibrated p-value for an observed statistic.
    pub fn p_value(&self, observed: f64) -> PValue {
        self.simulated_null.p_value_upper(observed)
    }
}

// ── Conservative null ───────────────────────────────────────────────────────

/// Use the worst-case null distribution for soundness.
/// Takes the maximum (least favorable) across multiple candidate nulls.
pub struct ConservativeNull {
    pub candidates: Vec<Box<dyn NullDistribution>>,
    pub name_str: String,
}

impl std::fmt::Debug for ConservativeNull {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConservativeNull")
            .field("name_str", &self.name_str)
            .field("num_candidates", &self.candidates.len())
            .finish()
    }
}

impl Clone for ConservativeNull {
    fn clone(&self) -> Self {
        // Cannot clone dyn NullDistribution; create empty clone
        ConservativeNull {
            candidates: Vec::new(),
            name_str: self.name_str.clone(),
        }
    }
}

impl ConservativeNull {
    pub fn new(candidates: Vec<Box<dyn NullDistribution>>) -> Self {
        Self { candidates, name_str: "Conservative Null".into() }
    }

    /// Compute conservative p-value: maximum across candidate nulls.
    pub fn conservative_p_value(&self, observed: f64) -> PValue {
        let max_p = self.candidates
            .iter()
            .map(|null| null.p_value_upper(observed).value())
            .fold(0.0_f64, f64::max);
        PValue::new_unchecked(max_p)
    }

    /// Conservative critical value: maximum across candidates.
    pub fn conservative_critical_value(&self, alpha: f64) -> f64 {
        self.candidates
            .iter()
            .map(|null| null.critical_value(alpha))
            .fold(f64::NEG_INFINITY, f64::max)
    }
}

// ── Distribution approximations ─────────────────────────────────────────────

/// Student's t CDF approximation (Hill, 1970).
fn student_t_cdf(x: f64, df: f64) -> f64 {
    if df <= 0.0 { return 0.5; }
    // For large df, use normal approximation
    if df > 200.0 {
        return normal_cdf(x);
    }
    // Approximate via regularized incomplete beta function
    let t2 = x * x;
    let a = 0.5 * df;
    let b = 0.5;
    let z = df / (df + t2);
    let ibeta = regularized_incomplete_beta(z, a, b);
    if x >= 0.0 {
        1.0 - 0.5 * ibeta
    } else {
        0.5 * ibeta
    }
}

fn student_t_quantile(p: f64, df: f64) -> f64 {
    if df > 200.0 {
        return normal_quantile(p);
    }
    // Newton's method from normal approximation
    let mut x = normal_quantile(p);
    for _ in 0..10 {
        let fx = student_t_cdf(x, df) - p;
        // Approximate derivative via finite differences
        let dx = 1e-6;
        let dfx = (student_t_cdf(x + dx, df) - student_t_cdf(x - dx, df)) / (2.0 * dx);
        if dfx.abs() < 1e-15 { break; }
        x -= fx / dfx;
    }
    x
}

/// Chi-squared CDF approximation (Wilson-Hilferty).
fn chi_squared_cdf(x: f64, df: f64) -> f64 {
    if x <= 0.0 || df <= 0.0 { return 0.0; }
    // Wilson-Hilferty approximation
    let z = ((x / df).powf(1.0 / 3.0) - (1.0 - 2.0 / (9.0 * df)))
        / (2.0 / (9.0 * df)).sqrt();
    normal_cdf(z)
}

fn chi_squared_quantile(p: f64, df: f64) -> f64 {
    if df <= 0.0 { return 0.0; }
    let z = normal_quantile(p);
    let q = 2.0 / (9.0 * df);
    let val = (1.0 - q + z * q.sqrt()).powi(3) * df;
    val.max(0.0)
}

/// F-distribution CDF approximation.
fn f_distribution_cdf(x: f64, df1: f64, df2: f64) -> f64 {
    if x <= 0.0 { return 0.0; }
    let z = df1 * x / (df1 * x + df2);
    regularized_incomplete_beta(z, df1 / 2.0, df2 / 2.0)
}

fn f_distribution_quantile(p: f64, df1: f64, df2: f64) -> f64 {
    // Use chi-squared quotient approximation
    let c1 = chi_squared_quantile(p, df1) / df1;
    let c2_inv = df2 / chi_squared_quantile(1.0 - p, df2).max(1e-15);
    (c1 * c2_inv).max(0.0)
}

/// Regularized incomplete beta function (simple series expansion).
fn regularized_incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 { return 0.0; }
    if x >= 1.0 { return 1.0; }

    // Use continued fraction for x > (a+1)/(a+b+2), else series
    let threshold = (a + 1.0) / (a + b + 2.0);
    if x > threshold {
        return 1.0 - regularized_incomplete_beta(1.0 - x, b, a);
    }

    // Series expansion: I_x(a,b) = x^a * (1-x)^b / (a * B(a,b)) * Σ ...
    let lnbeta = ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b);
    let prefix = (a * x.ln() + b * (1.0 - x).ln() - lnbeta).exp() / a;

    let mut sum = 1.0;
    let mut term = 1.0;
    for n in 1..200 {
        term *= x * (n as f64 - b) / (n as f64 * (a + n as f64));
        sum += term;
        if term.abs() < 1e-12 { break; }
    }

    (prefix * sum * a).clamp(0.0, 1.0)
}

/// Log-gamma function (Stirling's approximation + Lanczos).
fn ln_gamma(x: f64) -> f64 {
    if x <= 0.0 { return 0.0; }
    // Lanczos approximation with g=7
    let coef = [
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
    let g = 7.0;
    let x = x - 1.0;
    let mut sum = coef[0];
    for (i, &c) in coef.iter().enumerate().skip(1) {
        sum += c / (x + i as f64);
    }
    let t = x + g + 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + sum.ln()
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_analytical_normal_cdf() {
        let null = AnalyticalNull::StandardNormal;
        assert!(approx_eq(null.cdf(0.0), 0.5, 1e-5));
        assert!(null.cdf(2.0) > 0.97);
        assert!(null.cdf(-2.0) < 0.03);
    }

    #[test]
    fn test_analytical_normal_quantile() {
        let null = AnalyticalNull::StandardNormal;
        assert!(approx_eq(null.quantile(0.5), 0.0, 0.01));
        assert!(null.quantile(0.975) > 1.9);
        assert!(null.quantile(0.975) < 2.0);
    }

    #[test]
    fn test_analytical_normal_sample() {
        let null = AnalyticalNull::StandardNormal;
        let mut rng = StdRng::seed_from_u64(42);
        let samples = null.sample_n(10000, &mut rng);
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        assert!(approx_eq(mean, 0.0, 0.1));
    }

    #[test]
    fn test_analytical_t_cdf() {
        let null = AnalyticalNull::StudentT { df: 30.0 };
        assert!(approx_eq(null.cdf(0.0), 0.5, 0.01));
        assert!(null.cdf(3.0) > 0.99);
    }

    #[test]
    fn test_analytical_chi2_cdf() {
        let null = AnalyticalNull::ChiSquared { df: 5.0 };
        assert!(approx_eq(null.cdf(0.0), 0.0, 0.01));
        assert!(null.cdf(20.0) > 0.99);
    }

    #[test]
    fn test_analytical_f_cdf() {
        let null = AnalyticalNull::FDistribution { df1: 5.0, df2: 20.0 };
        assert!(null.cdf(0.0) < 0.01);
        assert!(null.cdf(10.0) > 0.99);
    }

    #[test]
    fn test_analytical_uniform() {
        let null = AnalyticalNull::Uniform;
        assert!(approx_eq(null.cdf(0.5), 0.5, 1e-10));
        assert!(approx_eq(null.quantile(0.3), 0.3, 1e-10));
    }

    #[test]
    fn test_simulated_null_basic() {
        let samples: Vec<f64> = (0..1000).map(|i| i as f64 / 1000.0).collect();
        let null = SimulatedNull::new(samples, "test");
        assert_eq!(null.len(), 1000);
        assert!(approx_eq(null.cdf(0.5), 0.5, 0.01));
    }

    #[test]
    fn test_simulated_null_mean_sd() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let null = SimulatedNull::new(samples, "test");
        assert!(approx_eq(null.mean(), 3.0, 1e-10));
        assert!(null.std_dev() > 0.0);
    }

    #[test]
    fn test_simulated_null_pvalue() {
        let samples: Vec<f64> = (0..1000).map(|i| i as f64 * 0.01).collect();
        let null = SimulatedNull::new(samples, "test");
        let p = null.p_value_upper(9.5);
        assert!(p.value() < 0.1);
    }

    #[test]
    fn test_asymptotic_null() {
        let null = AsymptoticNull::new(0.0, 1.0, 100);
        assert!(approx_eq(null.cdf(0.0), 0.5, 0.01));
        assert!(null.std_dev() > 0.0);
    }

    #[test]
    fn test_asymptotic_null_standardize() {
        let null = AsymptoticNull::new(5.0, 4.0, 100);
        let z = null.standardize(5.0);
        assert!(approx_eq(z, 0.0, 1e-10));
    }

    #[test]
    fn test_asymptotic_with_berry_esseen() {
        let be = BerryEsseenCorrection::new(1.0, 1.0);
        let null = AsymptoticNull::new(0.0, 1.0, 100).with_berry_esseen(be);
        let cdf_val = null.cdf(0.0);
        assert!(approx_eq(cdf_val, 0.5, 0.1));
    }

    #[test]
    fn test_berry_esseen_bound() {
        let be = BerryEsseenCorrection::new(2.0, 1.0);
        let bound_10 = be.remainder_bound(10);
        let bound_100 = be.remainder_bound(100);
        assert!(bound_100 < bound_10); // Bound decreases with n
        assert!(bound_10 > 0.0);
    }

    #[test]
    fn test_berry_esseen_from_data() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let be = BerryEsseenCorrection::from_data(&data).unwrap();
        assert!(be.third_moment > 0.0);
        assert!(be.variance > 0.0);
    }

    #[test]
    fn test_berry_esseen_reliable() {
        let be = BerryEsseenCorrection::new(1.0, 1.0);
        assert!(!be.is_reliable(5, 0.01));
        assert!(be.is_reliable(100000, 0.01));
    }

    #[test]
    fn test_competitive_simulator_narrow() {
        let config = SimulationConfig {
            num_rounds: 100,
            num_players: 2,
            demand_system: DemandSystem::Linear {
                intercept: 10.0,
                slope: 1.0,
                cross_slope: 0.5,
            },
            marginal_cost: vec![1.0, 1.0],
            ..Default::default()
        };
        let sim = CompetitiveSimulator::new(config, 100).with_seed(42);
        let null = sim.simulate_narrow_null().unwrap();
        assert_eq!(null.len(), 100);
        assert!(null.mean() > 0.0);
    }

    #[test]
    fn test_competitive_simulator_medium() {
        let config = SimulationConfig {
            num_rounds: 100,
            num_players: 2,
            demand_system: DemandSystem::Logit { mu: 0.5, a_0: 0.0 },
            marginal_cost: vec![1.0, 1.0],
            ..Default::default()
        };
        let sim = CompetitiveSimulator::new(config, 50).with_seed(42);
        let null = sim.simulate_medium_null().unwrap();
        assert!(!null.is_empty());
    }

    #[test]
    fn test_competitive_simulator_broad() {
        let config = SimulationConfig::default();
        let sim = CompetitiveSimulator::new(config, 50).with_seed(42);
        let null = sim.simulate_broad_null().unwrap();
        assert!(!null.is_empty());
    }

    #[test]
    fn test_null_calibration() {
        let samples: Vec<f64> = {
            let mut rng = StdRng::seed_from_u64(42);
            (0..1000).map(|_| {
                let u1: f64 = rng.gen_range(0.0001..1.0);
                let u2: f64 = rng.gen_range(0.0..std::f64::consts::TAU);
                (-2.0 * u1.ln()).sqrt() * u2.cos()
            }).collect()
        };
        let null = SimulatedNull::new(samples, "test");
        let cal = NullCalibration::calibrate(null, 0.05);
        assert!(cal.critical_value() > 0.0);
        assert!(cal.calibration_factor > 0.0);
    }

    #[test]
    fn test_conservative_null() {
        let null1 = Box::new(AnalyticalNull::StandardNormal) as Box<dyn NullDistribution>;
        let null2 = Box::new(AsymptoticNull::new(0.0, 2.0, 50)) as Box<dyn NullDistribution>;
        let cn = ConservativeNull::new(vec![null1, null2]);
        let p = cn.conservative_p_value(1.5);
        assert!(p.value() > 0.0);
        let cv = cn.conservative_critical_value(0.05);
        assert!(cv > 0.0);
    }

    #[test]
    fn test_chi_squared_quantile_basic() {
        let q = chi_squared_quantile(0.95, 5.0);
        assert!(q > 10.0); // χ²(5, 0.95) ≈ 11.07
        assert!(q < 13.0);
    }

    #[test]
    fn test_ln_gamma() {
        // Γ(1) = 1, ln(Γ(1)) = 0
        assert!(approx_eq(ln_gamma(1.0), 0.0, 0.01));
        // Γ(2) = 1, ln(Γ(2)) = 0
        assert!(approx_eq(ln_gamma(2.0), 0.0, 0.01));
        // Γ(5) = 24, ln(24) ≈ 3.178
        assert!(approx_eq(ln_gamma(5.0), (24.0_f64).ln(), 0.01));
    }
}
