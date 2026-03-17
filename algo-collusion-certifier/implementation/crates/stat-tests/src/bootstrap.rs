//! Bootstrap methods for confidence intervals and p-values.
//!
//! Implements nonparametric, block (circular, moving, stationary), and
//! parametric bootstrap, plus BCa intervals, studentized intervals,
//! and the double bootstrap.

use rand::prelude::*;
use rand::SeedableRng;
use rand_distr::{Geometric, Distribution};
use serde::{Deserialize, Serialize};
use shared_types::{CollusionError, CollusionResult, ConfidenceInterval, PValue};

use crate::effect_size::normal_quantile;

// ── Bootstrap engine ────────────────────────────────────────────────────────

/// Generic bootstrap framework that delegates to specific strategies.
#[derive(Debug, Clone)]
pub struct BootstrapEngine {
    pub num_resamples: usize,
    pub seed: Option<u64>,
}

impl BootstrapEngine {
    pub fn new(num_resamples: usize) -> Self {
        Self { num_resamples, seed: None }
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    fn make_rng(&self) -> StdRng {
        match self.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        }
    }

    /// Run bootstrap with the given statistic function.
    pub fn run<F>(&self, data: &[f64], statistic: F) -> CollusionResult<Vec<f64>>
    where
        F: Fn(&[f64]) -> f64,
    {
        if data.is_empty() {
            return Err(CollusionError::StatisticalTest("Bootstrap: empty data".into()));
        }
        let mut rng = self.make_rng();
        let n = data.len();
        let mut replicates = Vec::with_capacity(self.num_resamples);

        for _ in 0..self.num_resamples {
            let sample: Vec<f64> = (0..n)
                .map(|_| data[rng.gen_range(0..n)])
                .collect();
            replicates.push(statistic(&sample));
        }
        Ok(replicates)
    }
}

// ── Nonparametric bootstrap ─────────────────────────────────────────────────

/// Standard nonparametric bootstrap: resample with replacement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NonparametricBootstrap {
    pub num_resamples: usize,
    pub seed: Option<u64>,
}

impl NonparametricBootstrap {
    pub fn new(num_resamples: usize) -> Self {
        Self { num_resamples, seed: None }
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Generate bootstrap replicates of a statistic.
    pub fn resample<F>(&self, data: &[f64], statistic: F) -> CollusionResult<Vec<f64>>
    where
        F: Fn(&[f64]) -> f64,
    {
        let engine = BootstrapEngine {
            num_resamples: self.num_resamples,
            seed: self.seed,
        };
        engine.run(data, statistic)
    }

    /// Bootstrap standard error.
    pub fn standard_error<F>(&self, data: &[f64], statistic: F) -> CollusionResult<f64>
    where
        F: Fn(&[f64]) -> f64,
    {
        let reps = self.resample(data, statistic)?;
        let mean = reps.iter().sum::<f64>() / reps.len() as f64;
        let var = reps.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (reps.len() - 1) as f64;
        Ok(var.sqrt())
    }
}

// ── Block bootstrap ─────────────────────────────────────────────────────────

/// Block bootstrap variant.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum BlockBootstrapMethod {
    /// Circular block bootstrap (Politis & Romano, 1992).
    Circular,
    /// Moving block bootstrap (Kunsch, 1989).
    Moving,
    /// Stationary bootstrap with random block length (Politis & Romano, 1994).
    Stationary,
}

/// Block bootstrap for time-series data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockBootstrap {
    pub num_resamples: usize,
    pub block_size: usize,
    pub method: BlockBootstrapMethod,
    pub seed: Option<u64>,
}

impl BlockBootstrap {
    pub fn new(num_resamples: usize, block_size: usize, method: BlockBootstrapMethod) -> Self {
        Self { num_resamples, block_size, method, seed: None }
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Optimal block size by rule-of-thumb: n^(1/3).
    pub fn optimal_block_size(n: usize) -> usize {
        ((n as f64).powf(1.0 / 3.0)).ceil() as usize
    }

    /// Generate bootstrap replicates preserving temporal dependence.
    pub fn resample<F>(&self, data: &[f64], statistic: F) -> CollusionResult<Vec<f64>>
    where
        F: Fn(&[f64]) -> f64,
    {
        if data.is_empty() {
            return Err(CollusionError::StatisticalTest("Block bootstrap: empty data".into()));
        }
        let n = data.len();
        let mut rng = match self.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        let mut replicates = Vec::with_capacity(self.num_resamples);

        for _ in 0..self.num_resamples {
            let sample = match self.method {
                BlockBootstrapMethod::Circular => {
                    self.circular_sample(data, n, &mut rng)
                }
                BlockBootstrapMethod::Moving => {
                    self.moving_sample(data, n, &mut rng)
                }
                BlockBootstrapMethod::Stationary => {
                    self.stationary_sample(data, n, &mut rng)
                }
            };
            replicates.push(statistic(&sample));
        }
        Ok(replicates)
    }

    fn circular_sample(&self, data: &[f64], n: usize, rng: &mut StdRng) -> Vec<f64> {
        let mut sample = Vec::with_capacity(n);
        while sample.len() < n {
            let start = rng.gen_range(0..n);
            for j in 0..self.block_size {
                if sample.len() >= n {
                    break;
                }
                sample.push(data[(start + j) % n]);
            }
        }
        sample.truncate(n);
        sample
    }

    fn moving_sample(&self, data: &[f64], n: usize, rng: &mut StdRng) -> Vec<f64> {
        let max_start = if n > self.block_size { n - self.block_size } else { 0 };
        let mut sample = Vec::with_capacity(n);
        while sample.len() < n {
            let start = rng.gen_range(0..=max_start);
            let end = (start + self.block_size).min(n);
            for j in start..end {
                if sample.len() >= n {
                    break;
                }
                sample.push(data[j]);
            }
        }
        sample.truncate(n);
        sample
    }

    fn stationary_sample(&self, data: &[f64], n: usize, rng: &mut StdRng) -> Vec<f64> {
        let p = 1.0 / self.block_size as f64;
        let geom = Geometric::new(p).unwrap_or_else(|_| Geometric::new(0.5).unwrap());
        let mut sample = Vec::with_capacity(n);
        let mut pos = rng.gen_range(0..n);

        while sample.len() < n {
            let block_len = (geom.sample(rng) as usize).max(1);
            for _ in 0..block_len {
                if sample.len() >= n {
                    break;
                }
                sample.push(data[pos % n]);
                pos += 1;
            }
            pos = rng.gen_range(0..n);
        }
        sample.truncate(n);
        sample
    }
}

// ── Parametric bootstrap ────────────────────────────────────────────────────

/// Parametric bootstrap: resample from a fitted parametric model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParametricBootstrap {
    pub num_resamples: usize,
    pub seed: Option<u64>,
    /// Fitted mean
    pub mu: f64,
    /// Fitted standard deviation
    pub sigma: f64,
}

impl ParametricBootstrap {
    pub fn new(num_resamples: usize, mu: f64, sigma: f64) -> Self {
        Self { num_resamples, seed: None, mu, sigma }
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Fit from data (MLE for normal distribution).
    pub fn fit(data: &[f64], num_resamples: usize) -> CollusionResult<Self> {
        if data.is_empty() {
            return Err(CollusionError::StatisticalTest(
                "Parametric bootstrap: empty data".into(),
            ));
        }
        let n = data.len() as f64;
        let mu = data.iter().sum::<f64>() / n;
        let sigma = (data.iter().map(|x| (x - mu).powi(2)).sum::<f64>() / n).sqrt();
        Ok(Self { num_resamples, seed: None, mu, sigma })
    }

    /// Generate bootstrap samples from the fitted normal.
    pub fn resample<F>(&self, n: usize, statistic: F) -> CollusionResult<Vec<f64>>
    where
        F: Fn(&[f64]) -> f64,
    {
        let mut rng = match self.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        let normal = rand_distr::Normal::new(self.mu, self.sigma.max(1e-15))
            .map_err(|e| CollusionError::StatisticalTest(format!("Normal distribution error: {e}")))?;

        let mut replicates = Vec::with_capacity(self.num_resamples);
        for _ in 0..self.num_resamples {
            let sample: Vec<f64> = (0..n).map(|_| normal.sample(&mut rng)).collect();
            replicates.push(statistic(&sample));
        }
        Ok(replicates)
    }
}

// ── Bootstrap CI ────────────────────────────────────────────────────────────

/// Bootstrap confidence interval methods.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrapCI;

impl BootstrapCI {
    /// Percentile bootstrap CI.
    pub fn percentile(replicates: &[f64], alpha: f64) -> ConfidenceInterval {
        let mut sorted = replicates.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = sorted.len();
        let lo_idx = ((alpha / 2.0) * n as f64).floor() as usize;
        let hi_idx = ((1.0 - alpha / 2.0) * n as f64).ceil() as usize;
        let lo = sorted[lo_idx.min(n - 1)];
        let hi = sorted[hi_idx.min(n - 1)];
        let point = sorted[n / 2];
        ConfidenceInterval::new(lo, hi, 1.0 - alpha, point)
    }

    /// BCa (Bias-Corrected and accelerated) bootstrap CI.
    pub fn bca(
        replicates: &[f64],
        alpha: f64,
        theta_hat: f64,
        jackknife_values: &[f64],
    ) -> ConfidenceInterval {
        let n = replicates.len() as f64;
        if replicates.is_empty() {
            return ConfidenceInterval::new(0.0, 0.0, 1.0 - alpha, 0.0);
        }

        // Bias correction: z0 = Φ^{-1}(proportion of replicates < theta_hat)
        let count_below = replicates.iter().filter(|&&x| x < theta_hat).count() as f64;
        let z0 = normal_quantile((count_below / n).clamp(0.001, 0.999));

        // Acceleration factor from jackknife
        let a = Self::compute_acceleration(jackknife_values);

        // Adjusted percentiles
        let z_lo = normal_quantile(alpha / 2.0);
        let z_hi = normal_quantile(1.0 - alpha / 2.0);

        let alpha1 = crate::effect_size::normal_cdf(
            z0 + (z0 + z_lo) / (1.0 - a * (z0 + z_lo)),
        );
        let alpha2 = crate::effect_size::normal_cdf(
            z0 + (z0 + z_hi) / (1.0 - a * (z0 + z_hi)),
        );

        let mut sorted = replicates.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let m = sorted.len();

        let lo_idx = (alpha1 * m as f64).floor() as usize;
        let hi_idx = (alpha2 * m as f64).ceil() as usize;
        let lo = sorted[lo_idx.min(m - 1)];
        let hi = sorted[hi_idx.min(m - 1)];

        ConfidenceInterval::new(lo, hi, 1.0 - alpha, theta_hat)
    }

    fn compute_acceleration(jackknife_values: &[f64]) -> f64 {
        if jackknife_values.is_empty() {
            return 0.0;
        }
        let n = jackknife_values.len() as f64;
        let mean = jackknife_values.iter().sum::<f64>() / n;
        let diffs: Vec<f64> = jackknife_values.iter().map(|x| mean - x).collect();
        let sum_cubed: f64 = diffs.iter().map(|d| d.powi(3)).sum();
        let sum_sq: f64 = diffs.iter().map(|d| d.powi(2)).sum();
        if sum_sq.abs() < 1e-15 {
            return 0.0;
        }
        sum_cubed / (6.0 * sum_sq.powf(1.5))
    }

    /// Studentized (bootstrap-t) CI.
    pub fn studentized(
        replicates: &[f64],
        se_replicates: &[f64],
        theta_hat: f64,
        se_hat: f64,
        alpha: f64,
    ) -> ConfidenceInterval {
        if replicates.is_empty() || se_replicates.is_empty() {
            return ConfidenceInterval::new(0.0, 0.0, 1.0 - alpha, theta_hat);
        }

        // t* = (θ* - θ_hat) / SE*
        let t_stars: Vec<f64> = replicates
            .iter()
            .zip(se_replicates.iter())
            .map(|(rep, se)| {
                if se.abs() < 1e-15 { 0.0 } else { (rep - theta_hat) / se }
            })
            .collect();

        let mut sorted_t = t_stars;
        sorted_t.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = sorted_t.len();

        let t_lo = sorted_t[((alpha / 2.0) * n as f64).floor() as usize % n];
        let t_hi = sorted_t[((1.0 - alpha / 2.0) * n as f64).ceil() as usize % n];

        // CI: theta_hat - t_hi * SE, theta_hat - t_lo * SE
        let lo = theta_hat - t_hi * se_hat;
        let hi = theta_hat - t_lo * se_hat;
        ConfidenceInterval::new(lo, hi, 1.0 - alpha, theta_hat)
    }
}

/// Compute percentile CI from bootstrap samples (convenience function).
pub fn compute_percentile_ci(samples: &[f64], alpha: f64) -> ConfidenceInterval {
    BootstrapCI::percentile(samples, alpha)
}

/// Compute BCa CI from bootstrap samples (convenience function).
pub fn compute_bca_ci(
    samples: &[f64],
    alpha: f64,
    theta_hat: f64,
    jackknife_values: &[f64],
) -> ConfidenceInterval {
    BootstrapCI::bca(samples, alpha, theta_hat, jackknife_values)
}

// ── Bootstrap p-value ───────────────────────────────────────────────────────

/// Compute a bootstrap-based p-value.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrapPValue;

impl BootstrapPValue {
    /// One-sided p-value: P(T* >= T_obs | H0).
    pub fn one_sided_upper(null_replicates: &[f64], observed: f64) -> PValue {
        if null_replicates.is_empty() {
            return PValue::new_unchecked(1.0);
        }
        let count = null_replicates.iter().filter(|&&x| x >= observed).count();
        PValue::new_unchecked((count as f64 + 1.0) / (null_replicates.len() as f64 + 1.0))
    }

    /// One-sided p-value: P(T* <= T_obs | H0).
    pub fn one_sided_lower(null_replicates: &[f64], observed: f64) -> PValue {
        if null_replicates.is_empty() {
            return PValue::new_unchecked(1.0);
        }
        let count = null_replicates.iter().filter(|&&x| x <= observed).count();
        PValue::new_unchecked((count as f64 + 1.0) / (null_replicates.len() as f64 + 1.0))
    }

    /// Two-sided p-value using absolute deviation.
    pub fn two_sided(null_replicates: &[f64], observed: f64) -> PValue {
        if null_replicates.is_empty() {
            return PValue::new_unchecked(1.0);
        }
        let null_mean = null_replicates.iter().sum::<f64>() / null_replicates.len() as f64;
        let obs_dev = (observed - null_mean).abs();
        let count = null_replicates
            .iter()
            .filter(|&&x| (x - null_mean).abs() >= obs_dev)
            .count();
        PValue::new_unchecked((count as f64 + 1.0) / (null_replicates.len() as f64 + 1.0))
    }
}

// ── Collusion premium bootstrap ─────────────────────────────────────────────

/// Bootstrap CI specifically for the collusion premium CP = (π̄ - π_N)/(π_M - π_N).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollusionPremiumBootstrap {
    pub num_resamples: usize,
    pub seed: Option<u64>,
}

impl CollusionPremiumBootstrap {
    pub fn new(num_resamples: usize) -> Self {
        Self { num_resamples, seed: None }
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Compute bootstrap CI for the collusion premium.
    pub fn compute(
        &self,
        observed_profits: &[f64],
        nash_profit: f64,
        monopoly_profit: f64,
        alpha: f64,
    ) -> CollusionResult<ConfidenceInterval> {
        if observed_profits.is_empty() {
            return Err(CollusionError::StatisticalTest(
                "CP bootstrap: empty profits".into(),
            ));
        }
        let denom = monopoly_profit - nash_profit;
        if denom.abs() < 1e-12 {
            return Err(CollusionError::StatisticalTest(
                "CP bootstrap: monopoly profit equals Nash profit".into(),
            ));
        }

        let cp_stat = |profits: &[f64]| -> f64 {
            let mean = profits.iter().sum::<f64>() / profits.len() as f64;
            ((mean - nash_profit) / denom).clamp(0.0, 1.0)
        };

        let engine = BootstrapEngine {
            num_resamples: self.num_resamples,
            seed: self.seed,
        };
        let reps = engine.run(observed_profits, cp_stat)?;

        // Compute jackknife values for BCa
        let n = observed_profits.len();
        let jackknife: Vec<f64> = (0..n)
            .map(|i| {
                let subset: Vec<f64> = observed_profits
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)
                    .map(|(_, v)| *v)
                    .collect();
                cp_stat(&subset)
            })
            .collect();

        let theta_hat = cp_stat(observed_profits);
        Ok(BootstrapCI::bca(&reps, alpha, theta_hat, &jackknife))
    }
}

// ── Double bootstrap ────────────────────────────────────────────────────────

/// Double bootstrap for improved coverage of studentized intervals.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoubleBootstrap {
    pub outer_resamples: usize,
    pub inner_resamples: usize,
    pub seed: Option<u64>,
}

impl DoubleBootstrap {
    pub fn new(outer_resamples: usize, inner_resamples: usize) -> Self {
        Self { outer_resamples, inner_resamples, seed: None }
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Compute calibrated CI via double bootstrap.
    pub fn compute<F>(
        &self,
        data: &[f64],
        statistic: F,
        alpha: f64,
    ) -> CollusionResult<ConfidenceInterval>
    where
        F: Fn(&[f64]) -> f64 + Copy,
    {
        if data.is_empty() {
            return Err(CollusionError::StatisticalTest(
                "Double bootstrap: empty data".into(),
            ));
        }
        let n = data.len();
        let theta_hat = statistic(data);

        let mut rng = match self.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        // Outer bootstrap
        let mut outer_pivots = Vec::with_capacity(self.outer_resamples);

        for _ in 0..self.outer_resamples {
            let outer_sample: Vec<f64> = (0..n).map(|_| data[rng.gen_range(0..n)]).collect();
            let theta_star = statistic(&outer_sample);

            // Inner bootstrap for SE estimation
            let mut inner_reps = Vec::with_capacity(self.inner_resamples);
            for _ in 0..self.inner_resamples {
                let inner_sample: Vec<f64> = (0..n)
                    .map(|_| outer_sample[rng.gen_range(0..n)])
                    .collect();
                inner_reps.push(statistic(&inner_sample));
            }
            let inner_mean = inner_reps.iter().sum::<f64>() / inner_reps.len() as f64;
            let inner_var = inner_reps
                .iter()
                .map(|x| (x - inner_mean).powi(2))
                .sum::<f64>()
                / (inner_reps.len() - 1).max(1) as f64;
            let se_star = inner_var.sqrt();

            if se_star > 1e-15 {
                outer_pivots.push((theta_star - theta_hat) / se_star);
            }
        }

        if outer_pivots.is_empty() {
            return Ok(ConfidenceInterval::new(theta_hat, theta_hat, 1.0 - alpha, theta_hat));
        }

        outer_pivots.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let m = outer_pivots.len();

        // Bootstrap SE of original estimator
        let boot_reps: Vec<f64> = (0..self.outer_resamples)
            .map(|_| {
                let s: Vec<f64> = (0..n).map(|_| data[rng.gen_range(0..n)]).collect();
                statistic(&s)
            })
            .collect();
        let boot_mean = boot_reps.iter().sum::<f64>() / boot_reps.len() as f64;
        let se_hat = (boot_reps
            .iter()
            .map(|x| (x - boot_mean).powi(2))
            .sum::<f64>()
            / (boot_reps.len() - 1).max(1) as f64)
            .sqrt();

        let lo_idx = ((alpha / 2.0) * m as f64).floor() as usize;
        let hi_idx = ((1.0 - alpha / 2.0) * m as f64).ceil() as usize;
        let t_lo = outer_pivots[lo_idx.min(m - 1)];
        let t_hi = outer_pivots[hi_idx.min(m - 1)];

        Ok(ConfidenceInterval::new(
            theta_hat - t_hi * se_hat,
            theta_hat - t_lo * se_hat,
            1.0 - alpha,
            theta_hat,
        ))
    }
}

// ── Jackknife helper ────────────────────────────────────────────────────────

/// Compute leave-one-out jackknife values.
pub fn jackknife_values<F>(data: &[f64], statistic: F) -> Vec<f64>
where
    F: Fn(&[f64]) -> f64,
{
    (0..data.len())
        .map(|i| {
            let subset: Vec<f64> = data
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, v)| *v)
                .collect();
            statistic(&subset)
        })
        .collect()
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn sample_data() -> Vec<f64> {
        vec![
            2.1, 2.3, 2.5, 2.7, 2.2, 2.4, 2.6, 2.8, 2.3, 2.5,
            2.4, 2.6, 2.5, 2.7, 2.3, 2.4, 2.6, 2.5, 2.7, 2.8,
        ]
    }

    fn mean_stat(data: &[f64]) -> f64 {
        data.iter().sum::<f64>() / data.len() as f64
    }

    #[test]
    fn test_bootstrap_engine_basic() {
        let data = sample_data();
        let engine = BootstrapEngine::new(1000).with_seed(42);
        let reps = engine.run(&data, mean_stat).unwrap();
        assert_eq!(reps.len(), 1000);
        let boot_mean = mean_stat(&reps);
        let true_mean = mean_stat(&data);
        assert!(approx_eq(boot_mean, true_mean, 0.1));
    }

    #[test]
    fn test_bootstrap_engine_empty() {
        let engine = BootstrapEngine::new(100);
        assert!(engine.run(&[], mean_stat).is_err());
    }

    #[test]
    fn test_nonparametric_bootstrap() {
        let data = sample_data();
        let boot = NonparametricBootstrap::new(500).with_seed(123);
        let reps = boot.resample(&data, mean_stat).unwrap();
        assert_eq!(reps.len(), 500);
    }

    #[test]
    fn test_nonparametric_se() {
        let data = sample_data();
        let boot = NonparametricBootstrap::new(1000).with_seed(42);
        let se = boot.standard_error(&data, mean_stat).unwrap();
        assert!(se > 0.0);
        assert!(se < 0.5); // SE should be small for this data
    }

    #[test]
    fn test_block_bootstrap_circular() {
        let data: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin() + 2.0).collect();
        let bb = BlockBootstrap::new(200, 5, BlockBootstrapMethod::Circular).with_seed(42);
        let reps = bb.resample(&data, mean_stat).unwrap();
        assert_eq!(reps.len(), 200);
    }

    #[test]
    fn test_block_bootstrap_moving() {
        let data: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).cos() + 3.0).collect();
        let bb = BlockBootstrap::new(200, 5, BlockBootstrapMethod::Moving).with_seed(42);
        let reps = bb.resample(&data, mean_stat).unwrap();
        assert_eq!(reps.len(), 200);
    }

    #[test]
    fn test_block_bootstrap_stationary() {
        let data: Vec<f64> = (0..100).map(|i| (i as f64 * 0.05).sin() + 4.0).collect();
        let bb = BlockBootstrap::new(200, 5, BlockBootstrapMethod::Stationary).with_seed(42);
        let reps = bb.resample(&data, mean_stat).unwrap();
        assert_eq!(reps.len(), 200);
    }

    #[test]
    fn test_block_bootstrap_optimal_size() {
        assert_eq!(BlockBootstrap::optimal_block_size(1000), 10);
        assert_eq!(BlockBootstrap::optimal_block_size(27), 3);
    }

    #[test]
    fn test_parametric_bootstrap() {
        let data = sample_data();
        let pb = ParametricBootstrap::fit(&data, 500).unwrap().with_seed(42);
        let reps = pb.resample(data.len(), mean_stat).unwrap();
        assert_eq!(reps.len(), 500);
        let boot_mean = mean_stat(&reps);
        assert!(approx_eq(boot_mean, pb.mu, 0.2));
    }

    #[test]
    fn test_parametric_bootstrap_empty() {
        assert!(ParametricBootstrap::fit(&[], 100).is_err());
    }

    #[test]
    fn test_percentile_ci() {
        let data = sample_data();
        let engine = BootstrapEngine::new(2000).with_seed(42);
        let reps = engine.run(&data, mean_stat).unwrap();
        let ci = BootstrapCI::percentile(&reps, 0.05);
        assert!(ci.lower < ci.upper);
        assert!(ci.contains(mean_stat(&data)));
    }

    #[test]
    fn test_bca_ci() {
        let data = sample_data();
        let engine = BootstrapEngine::new(2000).with_seed(42);
        let reps = engine.run(&data, mean_stat).unwrap();
        let jk = jackknife_values(&data, mean_stat);
        let theta = mean_stat(&data);
        let ci = BootstrapCI::bca(&reps, 0.05, theta, &jk);
        assert!(ci.lower < ci.upper);
    }

    #[test]
    fn test_studentized_ci() {
        let reps = vec![1.0, 1.1, 0.9, 1.05, 0.95, 1.02, 0.98, 1.03, 0.97, 1.01];
        let ses = vec![0.1; 10];
        let ci = BootstrapCI::studentized(&reps, &ses, 1.0, 0.1, 0.05);
        assert!(ci.lower < ci.upper);
    }

    #[test]
    fn test_bootstrap_pvalue_upper() {
        let null = vec![0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5];
        let p = BootstrapPValue::one_sided_upper(&null, 1.5);
        assert!(p.value() > 0.0);
        assert!(p.value() < 1.0);
    }

    #[test]
    fn test_bootstrap_pvalue_lower() {
        let null = vec![0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5];
        let p = BootstrapPValue::one_sided_lower(&null, 0.5);
        assert!(p.value() > 0.0);
    }

    #[test]
    fn test_bootstrap_pvalue_two_sided() {
        let null = vec![0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5];
        let p = BootstrapPValue::two_sided(&null, 3.0);
        // 3.0 is far from null mean, should be small p
        assert!(p.value() < 0.5);
    }

    #[test]
    fn test_collusion_premium_bootstrap() {
        let profits = vec![5.0, 5.1, 4.9, 5.2, 4.8, 5.0, 5.1, 4.9, 5.0, 5.1];
        let cpb = CollusionPremiumBootstrap::new(1000).with_seed(42);
        let ci = cpb.compute(&profits, 3.0, 7.0, 0.05).unwrap();
        assert!(ci.lower >= 0.0);
        assert!(ci.upper <= 1.0);
        assert!(ci.lower < ci.upper);
    }

    #[test]
    fn test_collusion_premium_bootstrap_equal_profits() {
        let profits = vec![5.0; 10];
        let cpb = CollusionPremiumBootstrap::new(100);
        assert!(cpb.compute(&profits, 5.0, 5.0, 0.05).is_err());
    }

    #[test]
    fn test_double_bootstrap() {
        let data = sample_data();
        let db = DoubleBootstrap::new(100, 50).with_seed(42);
        let ci = db.compute(&data, mean_stat, 0.05).unwrap();
        assert!(ci.lower < ci.upper);
    }

    #[test]
    fn test_jackknife_values() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let jk = jackknife_values(&data, mean_stat);
        assert_eq!(jk.len(), 5);
        // Leave-one-out means
        assert!(approx_eq(jk[0], 3.5, 1e-10)); // mean(2,3,4,5) = 3.5
        assert!(approx_eq(jk[4], 2.5, 1e-10)); // mean(1,2,3,4) = 2.5
    }

    #[test]
    fn test_compute_percentile_ci_convenience() {
        let reps = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let ci = compute_percentile_ci(&reps, 0.1);
        assert!(ci.lower < ci.upper);
    }

    #[test]
    fn test_compute_bca_ci_convenience() {
        let reps: Vec<f64> = (1..=100).map(|i| i as f64 * 0.1).collect();
        let jk = vec![5.0, 5.1, 4.9, 5.05, 4.95];
        let ci = compute_bca_ci(&reps, 0.05, 5.0, &jk);
        assert!(ci.lower < ci.upper);
    }
}
