//! Individual price-based statistical tests for collusion detection.
//!
//! Tests for supra-competitive pricing, cross-firm parallelism, price
//! persistence, convergence patterns, low dispersion, and Edgeworth cycles.

use rand::prelude::*;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use shared_types::{
    CollusionError, CollusionResult, ConfidenceInterval, EffectSize,
    HypothesisTestResult, PValue, PlayerId, Price, PriceTrajectory,
    TestId, TestStatistic,
};

use crate::bootstrap::{BootstrapEngine, BootstrapCI, BlockBootstrap, BlockBootstrapMethod};
use crate::correlation_tests::pearson_correlation;
use crate::effect_size::{normal_cdf, normal_quantile, CohenD};
use crate::null_distribution::{AnalyticalNull, NullDistribution};

// ── Helper statistics ───────────────────────────────────────────────────────

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() { return 0.0; }
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn variance(xs: &[f64]) -> f64 {
    let n = xs.len();
    if n < 2 { return 0.0; }
    let m = mean(xs);
    xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1) as f64
}

fn std_dev(xs: &[f64]) -> f64 {
    variance(xs).sqrt()
}

// ── Supra-competitive price test ────────────────────────────────────────────

/// Test if average prices exceed the competitive (Nash) level.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupraCompetitivePriceTest {
    pub competitive_price: Price,
    pub bootstrap_resamples: usize,
    pub seed: Option<u64>,
}

impl SupraCompetitivePriceTest {
    pub fn new(competitive_price: Price) -> Self {
        Self {
            competitive_price,
            bootstrap_resamples: 2000,
            seed: None,
        }
    }

    pub fn with_bootstrap(mut self, resamples: usize) -> Self {
        self.bootstrap_resamples = resamples;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Run the one-sample t-test with bootstrap calibration.
    pub fn test(&self, prices: &[f64]) -> CollusionResult<SupraCompetitiveResult> {
        if prices.is_empty() {
            return Err(CollusionError::StatisticalTest(
                "Supra-competitive test: empty data".into(),
            ));
        }

        let n = prices.len();
        let price_mean = mean(prices);
        let price_sd = std_dev(prices);
        let se = price_sd / (n as f64).sqrt();

        let price_diff = price_mean - self.competitive_price.0;

        // t-statistic
        let t_stat = if se.abs() < 1e-15 { 0.0 } else { price_diff / se };

        // Parametric p-value
        let null = AnalyticalNull::StudentT { df: (n - 1) as f64 };
        let parametric_p = 1.0 - null.cdf(t_stat);

        // Bootstrap calibration
        let centered: Vec<f64> = prices.iter().map(|p| p - price_diff).collect();

        let engine = BootstrapEngine {
            num_resamples: self.bootstrap_resamples,
            seed: self.seed,
        };
        let boot_stats = engine.run(&centered, |s| {
            let m = mean(s);
            let sd = std_dev(s);
            let se = sd / (s.len() as f64).sqrt();
            if se.abs() < 1e-15 { 0.0 } else { (m - self.competitive_price.0) / se }
        })?;

        let boot_count = boot_stats.iter().filter(|&&t| t >= t_stat).count();
        let bootstrap_p = (boot_count as f64 + 1.0) / (boot_stats.len() as f64 + 1.0);

        // Effect size
        let d = CohenD::one_sample(prices, self.competitive_price.0)?;

        // CI for price difference
        let ci = ConfidenceInterval::new(
            price_diff - 1.96 * se,
            price_diff + 1.96 * se,
            0.95,
            price_diff,
        );

        Ok(SupraCompetitiveResult {
            price_mean,
            competitive_price: self.competitive_price.0,
            price_difference: price_diff,
            t_statistic: t_stat,
            parametric_p_value: PValue::new_unchecked(parametric_p),
            bootstrap_p_value: PValue::new_unchecked(bootstrap_p),
            effect_size: d.value,
            confidence_interval: ci,
            sample_size: n,
            is_supra_competitive: bootstrap_p < 0.05,
        })
    }

    /// Test from a PriceTrajectory for a specific player.
    pub fn test_trajectory(
        &self,
        trajectory: &PriceTrajectory,
        player_id: usize,
    ) -> CollusionResult<SupraCompetitiveResult> {
        let prices: Vec<f64> = trajectory.prices_for_player(shared_types::PlayerId(player_id)).iter().map(|p| p.0).collect();
        self.test(&prices)
    }
}

/// Result of the supra-competitive price test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SupraCompetitiveResult {
    pub price_mean: f64,
    pub competitive_price: f64,
    pub price_difference: f64,
    pub t_statistic: f64,
    pub parametric_p_value: PValue,
    pub bootstrap_p_value: PValue,
    pub effect_size: f64,
    pub confidence_interval: ConfidenceInterval,
    pub sample_size: usize,
    pub is_supra_competitive: bool,
}

impl SupraCompetitiveResult {
    pub fn to_hypothesis_test_result(&self) -> HypothesisTestResult {
        HypothesisTestResult {
            test_id: TestId::new(),
            test_name: "Supra-Competitive Price Test".into(),
            test_statistic: TestStatistic::t_score(self.t_statistic, (self.sample_size.max(1) - 1) as f64),
            p_value: self.bootstrap_p_value,
            effect_size: Some(EffectSize::cohen_d(self.effect_size)),
            reject: self.is_supra_competitive,
            alpha: 0.05,
            confidence_interval: Some(self.confidence_interval.clone()),
            sample_size: self.sample_size,
            description: String::new(),
        }
    }
}

// ── Price parallelism test ──────────────────────────────────────────────────

/// Test for excess cross-firm price correlation (parallelism).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceParallelismTest {
    pub max_lag: usize,
    pub seed: Option<u64>,
}

impl PriceParallelismTest {
    pub fn new(max_lag: usize) -> Self {
        Self { max_lag, seed: None }
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Test parallelism between two firm's price series.
    pub fn test(&self, firm1_prices: &[f64], firm2_prices: &[f64]) -> CollusionResult<ParallelismResult> {
        let n = firm1_prices.len().min(firm2_prices.len());
        if n < 5 {
            return Err(CollusionError::StatisticalTest(
                "Parallelism test: need ≥5 observations".into(),
            ));
        }

        let p1 = &firm1_prices[..n];
        let p2 = &firm2_prices[..n];

        // Pearson and Spearman on levels
        let pearson = pearson_correlation(p1, p2);

        // Correlation of price changes
        let dp1: Vec<f64> = (1..n).map(|t| p1[t] - p1[t - 1]).collect();
        let dp2: Vec<f64> = (1..n).map(|t| p2[t] - p2[t - 1]).collect();
        let change_corr = pearson_correlation(&dp1, &dp2);

        // Cross-correlation at multiple lags
        let mut lag_correlations = Vec::with_capacity(self.max_lag * 2 + 1);
        for lag in -(self.max_lag as i64)..=(self.max_lag as i64) {
            let r = lagged_correlation(p1, p2, lag);
            lag_correlations.push((lag, r));
        }

        // Null distribution: under independent learning, level correlation
        // can be high due to common convergence. Focus on change correlation.
        let se = 1.0 / ((n as f64 - 3.0).max(1.0)).sqrt();
        let z_change = crate::effect_size::fisher_z(change_corr) / se;
        let p_value = 1.0 - normal_cdf(z_change);

        // Maximum lag correlation (excluding lag 0)
        let max_lag_corr = lag_correlations
            .iter()
            .filter(|(lag, _)| *lag != 0)
            .map(|(_, r)| r.abs())
            .fold(0.0_f64, f64::max);

        Ok(ParallelismResult {
            level_correlation: pearson,
            change_correlation: change_corr,
            lag_correlations,
            max_lag_correlation: max_lag_corr,
            p_value: PValue::new_unchecked(p_value),
            sample_size: n,
            is_parallel: p_value < 0.05,
        })
    }

    /// Test parallelism across all firm pairs.
    pub fn test_all_pairs(
        &self,
        firm_prices: &[Vec<f64>],
    ) -> CollusionResult<Vec<ParallelismResult>> {
        let mut results = Vec::new();
        for i in 0..firm_prices.len() {
            for j in (i + 1)..firm_prices.len() {
                results.push(self.test(&firm_prices[i], &firm_prices[j])?);
            }
        }
        Ok(results)
    }
}

/// Result of the price parallelism test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelismResult {
    pub level_correlation: f64,
    pub change_correlation: f64,
    pub lag_correlations: Vec<(i64, f64)>,
    pub max_lag_correlation: f64,
    pub p_value: PValue,
    pub sample_size: usize,
    pub is_parallel: bool,
}

fn lagged_correlation(x: &[f64], y: &[f64], lag: i64) -> f64 {
    let n = x.len().min(y.len());
    let abs_lag = lag.unsigned_abs() as usize;
    if abs_lag >= n { return 0.0; }

    let (a, b) = if lag >= 0 {
        (&x[..n - abs_lag], &y[abs_lag..n])
    } else {
        (&x[abs_lag..n], &y[..n - abs_lag])
    };
    pearson_correlation(a, b)
}

// ── Price persistence test ──────────────────────────────────────────────────

/// Test for supra-competitive price persistence via run-length analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PricePersistenceTest {
    pub competitive_price: Price,
}

impl PricePersistenceTest {
    pub fn new(competitive_price: Price) -> Self {
        Self { competitive_price }
    }

    /// Run the persistence test.
    pub fn test(&self, prices: &[f64]) -> CollusionResult<PersistenceResult> {
        if prices.is_empty() {
            return Err(CollusionError::StatisticalTest(
                "Persistence test: empty data".into(),
            ));
        }

        // Compute above-competitive indicators
        let above: Vec<bool> = prices.iter().map(|p| *p > self.competitive_price.0).collect();
        let n_above = above.iter().filter(|&&a| a).count();
        let fraction_above = n_above as f64 / prices.len() as f64;

        // Run lengths of above-competitive pricing
        let runs = compute_run_lengths(&above, true);
        let max_run = runs.iter().copied().max().unwrap_or(0);
        let mean_run = if runs.is_empty() {
            0.0
        } else {
            runs.iter().sum::<usize>() as f64 / runs.len() as f64
        };

        // Geometric distribution test under null:
        // If prices are independently above competitive with probability p,
        // runs follow Geometric(1-p). Test if observed runs are too long.
        let null_p: f64 = 0.5; // Under competitive null, about half should be above
        let expected_mean_run = 1.0 / (1.0 - null_p);

        // Test statistic: max run length compared to geometric null
        let n = prices.len();
        let expected_max = (n as f64).ln() / (1.0 / null_p).ln();
        let p_value = if max_run as f64 > expected_max {
            // Approximate p-value for maximum run
            let lambda = n as f64 * (1.0 - null_p).powi(max_run as i32);
            (-lambda).exp()
        } else {
            1.0
        };

        Ok(PersistenceResult {
            fraction_above_competitive: fraction_above,
            num_runs: runs.len(),
            max_run_length: max_run,
            mean_run_length: mean_run,
            expected_mean_run: expected_mean_run,
            p_value: PValue::new_unchecked(p_value),
            is_persistent: p_value < 0.05 && fraction_above > 0.5,
        })
    }
}

/// Result of the price persistence test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistenceResult {
    pub fraction_above_competitive: f64,
    pub num_runs: usize,
    pub max_run_length: usize,
    pub mean_run_length: f64,
    pub expected_mean_run: f64,
    pub p_value: PValue,
    pub is_persistent: bool,
}

fn compute_run_lengths(indicators: &[bool], target: bool) -> Vec<usize> {
    let mut runs = Vec::new();
    let mut current_run = 0;
    for &val in indicators {
        if val == target {
            current_run += 1;
        } else {
            if current_run > 0 {
                runs.push(current_run);
            }
            current_run = 0;
        }
    }
    if current_run > 0 {
        runs.push(current_run);
    }
    runs
}

// ── Convergence pattern test ────────────────────────────────────────────────

/// Test for suspicious convergence patterns (too fast, too synchronized).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergencePatternTest {
    pub competitive_price: Price,
    pub window_size: usize,
}

impl ConvergencePatternTest {
    pub fn new(competitive_price: Price, window_size: usize) -> Self {
        Self { competitive_price, window_size }
    }

    /// Test convergence speed and synchronization.
    pub fn test(&self, firm_prices: &[Vec<f64>]) -> CollusionResult<ConvergenceResult> {
        if firm_prices.is_empty() {
            return Err(CollusionError::StatisticalTest(
                "Convergence test: no firms".into(),
            ));
        }
        let n_firms = firm_prices.len();
        let min_len = firm_prices.iter().map(|v| v.len()).min().unwrap_or(0);
        if min_len < self.window_size * 2 {
            return Err(CollusionError::StatisticalTest(
                "Convergence test: series too short for window".into(),
            ));
        }

        // Compute convergence speed for each firm
        let convergence_speeds: Vec<f64> = firm_prices
            .iter()
            .map(|prices| self.estimate_convergence_speed(prices))
            .collect();

        // Simultaneous convergence: variance of convergence times across firms
        let convergence_times: Vec<usize> = firm_prices
            .iter()
            .map(|prices| self.estimate_convergence_time(prices))
            .collect();

        let time_spread = if convergence_times.len() < 2 {
            0.0
        } else {
            let times_f: Vec<f64> = convergence_times.iter().map(|&t| t as f64).collect();
            std_dev(&times_f)
        };

        // Under competitive play, convergence should be more spread out
        let mean_speed = mean(&convergence_speeds);
        let speed_sd = std_dev(&convergence_speeds);

        // Simultaneous convergence test: low spread → suspicious
        let expected_spread = (min_len as f64 * 0.1).max(5.0);
        let sync_statistic = if expected_spread > 0.0 {
            time_spread / expected_spread
        } else {
            1.0
        };

        // p-value: probability of seeing this much synchronization under H0
        let p_value = sync_statistic.min(1.0);

        Ok(ConvergenceResult {
            convergence_speeds,
            convergence_times,
            time_spread,
            mean_convergence_speed: mean_speed,
            synchronization_statistic: sync_statistic,
            p_value: PValue::new_unchecked(p_value),
            is_suspicious: p_value < 0.05,
        })
    }

    fn estimate_convergence_speed(&self, prices: &[f64]) -> f64 {
        // Speed = rate of variance reduction over windows
        let n = prices.len();
        if n < self.window_size * 2 { return 0.0; }

        let early_var = variance(&prices[..self.window_size]);
        let late_var = variance(&prices[n - self.window_size..]);

        if early_var.abs() < 1e-15 { return 0.0; }
        (early_var - late_var) / early_var
    }

    fn estimate_convergence_time(&self, prices: &[f64]) -> usize {
        let n = prices.len();
        if n < self.window_size { return n; }

        let final_mean = mean(&prices[n - self.window_size..]);
        let threshold = 0.1 * std_dev(prices);

        for t in 0..n {
            let end = (t + self.window_size).min(n);
            if end - t < self.window_size { break; }
            let window_mean = mean(&prices[t..end]);
            if (window_mean - final_mean).abs() < threshold.max(1e-10) {
                return t;
            }
        }
        n
    }
}

/// Result of the convergence pattern test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceResult {
    pub convergence_speeds: Vec<f64>,
    pub convergence_times: Vec<usize>,
    pub time_spread: f64,
    pub mean_convergence_speed: f64,
    pub synchronization_statistic: f64,
    pub p_value: PValue,
    pub is_suspicious: bool,
}

// ── Price dispersion test ───────────────────────────────────────────────────

/// Test for low price dispersion (a collusion signature).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceDispersionTest {
    pub seed: Option<u64>,
}

impl PriceDispersionTest {
    pub fn new() -> Self {
        Self { seed: None }
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Test if cross-firm price dispersion is abnormally low.
    /// `firm_prices`: one Vec per firm, all same length.
    pub fn test(&self, firm_prices: &[Vec<f64>]) -> CollusionResult<DispersionResult> {
        if firm_prices.len() < 2 {
            return Err(CollusionError::StatisticalTest(
                "Dispersion test: need ≥2 firms".into(),
            ));
        }
        let min_len = firm_prices.iter().map(|v| v.len()).min().unwrap_or(0);
        if min_len == 0 {
            return Err(CollusionError::StatisticalTest(
                "Dispersion test: empty price data".into(),
            ));
        }

        // Cross-sectional coefficient of variation at each time step
        let n_firms = firm_prices.len();
        let cvs: Vec<f64> = (0..min_len)
            .map(|t| {
                let prices_t: Vec<f64> = (0..n_firms).map(|i| firm_prices[i][t]).collect();
                let m = mean(&prices_t);
                let s = std_dev(&prices_t);
                if m.abs() < 1e-15 { 0.0 } else { s / m }
            })
            .collect();

        let mean_cv = mean(&cvs);
        let entropy = self.price_entropy(firm_prices, min_len);

        // Under competitive play with heterogeneous costs, CV should be moderate
        // Under collusion, CV should be low
        // One-sided test: H0: CV >= cv0 vs H1: CV < cv0
        let cv0 = 0.1; // Expected competitive dispersion
        let cv_se = std_dev(&cvs) / (cvs.len() as f64).sqrt();
        let z_stat = if cv_se.abs() < 1e-15 { 0.0 } else { (mean_cv - cv0) / cv_se };
        let p_value = normal_cdf(z_stat); // Lower tail

        Ok(DispersionResult {
            mean_cv,
            cv_time_series: cvs,
            entropy,
            z_statistic: z_stat,
            p_value: PValue::new_unchecked(p_value),
            is_low_dispersion: p_value < 0.05 && mean_cv < cv0,
        })
    }

    fn price_entropy(&self, firm_prices: &[Vec<f64>], min_len: usize) -> f64 {
        let n_firms = firm_prices.len();
        if n_firms == 0 || min_len == 0 { return 0.0; }

        // Discretize prices and compute entropy
        let all_prices: Vec<f64> = (0..n_firms)
            .flat_map(|i| firm_prices[i][..min_len].to_vec())
            .collect();

        let min_p = all_prices.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_p = all_prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max_p - min_p;
        if range < 1e-15 { return 0.0; }

        let n_bins = 20usize;
        let mut bins = vec![0usize; n_bins];
        for &p in &all_prices {
            let bin = ((p - min_p) / range * (n_bins - 1) as f64).round() as usize;
            bins[bin.min(n_bins - 1)] += 1;
        }

        let total = all_prices.len() as f64;
        bins.iter()
            .filter(|&&c| c > 0)
            .map(|&c| {
                let p = c as f64 / total;
                -p * p.ln()
            })
            .sum()
    }
}

/// Result of the price dispersion test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DispersionResult {
    pub mean_cv: f64,
    pub cv_time_series: Vec<f64>,
    pub entropy: f64,
    pub z_statistic: f64,
    pub p_value: PValue,
    pub is_low_dispersion: bool,
}

// ── Edgeworth cycle test ────────────────────────────────────────────────────

/// Detect Edgeworth cycles vs collusive stability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeworthCycleTest {
    pub max_period: usize,
}

impl EdgeworthCycleTest {
    pub fn new(max_period: usize) -> Self {
        Self { max_period }
    }

    /// Test for periodicity in a price series.
    pub fn test(&self, prices: &[f64]) -> CollusionResult<EdgeworthResult> {
        if prices.len() < 2 * self.max_period {
            return Err(CollusionError::StatisticalTest(
                "Edgeworth test: series too short for period detection".into(),
            ));
        }

        // Autocorrelation function
        let acf: Vec<f64> = (1..=self.max_period)
            .map(|lag| autocorrelation(prices, lag))
            .collect();

        // Find dominant period: lag with highest positive autocorrelation
        let dominant_period = acf
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i + 1)
            .unwrap_or(1);

        let dominant_acf = acf.get(dominant_period - 1).copied().unwrap_or(0.0);

        // Cycling strength: variance of price changes
        let changes: Vec<f64> = (1..prices.len())
            .map(|t| prices[t] - prices[t - 1])
            .collect();
        let change_var = variance(&changes);
        let price_var = variance(prices);

        let cycling_ratio = if price_var.abs() < 1e-15 {
            0.0
        } else {
            change_var / price_var
        };

        // Stability: low cycling_ratio and low autocorrelation → stable (possibly collusive)
        // High cycling_ratio with periodic ACF → Edgeworth cycles
        let is_cycling = dominant_acf > 0.3 && cycling_ratio > 0.5;
        let is_stable = dominant_acf < 0.2 && cycling_ratio < 0.3;

        // Test significance of the dominant autocorrelation
        let n = prices.len() as f64;
        let se_acf = 1.0 / n.sqrt();
        let z_stat = dominant_acf / se_acf;
        let p_value = 2.0 * (1.0 - normal_cdf(z_stat.abs()));

        Ok(EdgeworthResult {
            autocorrelation_function: acf,
            dominant_period,
            dominant_autocorrelation: dominant_acf,
            cycling_ratio,
            is_cycling,
            is_stable,
            z_statistic: z_stat,
            p_value: PValue::new_unchecked(p_value),
        })
    }

    /// Distinguish competitive cycles from collusive stability across firms.
    pub fn test_multi_firm(&self, firm_prices: &[Vec<f64>]) -> CollusionResult<Vec<EdgeworthResult>> {
        firm_prices.iter().map(|prices| self.test(prices)).collect()
    }
}

fn autocorrelation(xs: &[f64], lag: usize) -> f64 {
    let n = xs.len();
    if n <= lag + 1 { return 0.0; }
    let m = mean(xs);
    let var: f64 = xs.iter().map(|x| (x - m).powi(2)).sum::<f64>();
    if var.abs() < 1e-15 { return 0.0; }
    let cov: f64 = (0..n - lag)
        .map(|i| (xs[i] - m) * (xs[i + lag] - m))
        .sum();
    cov / var
}

/// Result of the Edgeworth cycle test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeworthResult {
    pub autocorrelation_function: Vec<f64>,
    pub dominant_period: usize,
    pub dominant_autocorrelation: f64,
    pub cycling_ratio: f64,
    pub is_cycling: bool,
    pub is_stable: bool,
    pub z_statistic: f64,
    pub p_value: PValue,
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_supra_competitive_above() {
        // Prices clearly above competitive level
        let prices = vec![5.0, 5.1, 4.9, 5.2, 5.0, 5.1, 4.8, 5.3, 5.0, 5.1,
                         5.2, 5.0, 4.9, 5.1, 5.0, 5.2, 5.1, 5.0, 4.9, 5.1];
        let test = SupraCompetitivePriceTest::new(3.0).with_bootstrap(500).with_seed(42);
        let result = test.test(&prices).unwrap();
        assert!(result.price_difference > 0.0);
        assert!(result.is_supra_competitive);
    }

    #[test]
    fn test_supra_competitive_at_level() {
        let prices = vec![3.0, 3.01, 2.99, 3.0, 3.01, 2.99, 3.0, 3.01, 2.99, 3.0];
        let test = SupraCompetitivePriceTest::new(3.0).with_bootstrap(500).with_seed(42);
        let result = test.test(&prices).unwrap();
        assert!(result.price_difference.abs() < 0.1);
    }

    #[test]
    fn test_supra_competitive_empty() {
        let test = SupraCompetitivePriceTest::new(3.0);
        assert!(test.test(&[]).is_err());
    }

    #[test]
    fn test_supra_competitive_to_hypothesis() {
        let prices = vec![5.0; 20];
        let test = SupraCompetitivePriceTest::new(3.0).with_bootstrap(200).with_seed(42);
        let result = test.test(&prices).unwrap();
        let htr = result.to_hypothesis_test_result();
        assert_eq!(htr.test_name, "Supra-Competitive Price Test");
    }

    #[test]
    fn test_parallelism_correlated() {
        let f1: Vec<f64> = (0..50).map(|i| 2.0 + 0.02 * i as f64).collect();
        let f2: Vec<f64> = (0..50).map(|i| 2.1 + 0.02 * i as f64).collect();
        let test = PriceParallelismTest::new(5);
        let result = test.test(&f1, &f2).unwrap();
        assert!(result.level_correlation > 0.99);
    }

    #[test]
    fn test_parallelism_uncorrelated() {
        let mut rng = StdRng::seed_from_u64(42);
        let f1: Vec<f64> = (0..100).map(|_| rng.gen_range(1.0..5.0)).collect();
        let f2: Vec<f64> = (0..100).map(|_| rng.gen_range(1.0..5.0)).collect();
        let test = PriceParallelismTest::new(3);
        let result = test.test(&f1, &f2).unwrap();
        assert!(result.level_correlation.abs() < 0.3);
    }

    #[test]
    fn test_parallelism_all_pairs() {
        let f1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let f2 = vec![1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1];
        let f3 = vec![1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2, 8.2, 9.2, 10.2];
        let test = PriceParallelismTest::new(2);
        let results = test.test_all_pairs(&[f1, f2, f3]).unwrap();
        assert_eq!(results.len(), 3); // C(3,2) = 3
    }

    #[test]
    fn test_parallelism_short_error() {
        let test = PriceParallelismTest::new(2);
        assert!(test.test(&[1.0, 2.0], &[1.0, 2.0]).is_err());
    }

    #[test]
    fn test_persistence_all_above() {
        let prices = vec![5.0; 100];
        let test = PricePersistenceTest::new(3.0);
        let result = test.test(&prices).unwrap();
        assert!(approx_eq(result.fraction_above_competitive, 1.0, 1e-10));
        assert!(result.max_run_length == 100);
        assert!(result.is_persistent);
    }

    #[test]
    fn test_persistence_alternating() {
        let prices: Vec<f64> = (0..100).map(|i| if i % 2 == 0 { 4.0 } else { 2.0 }).collect();
        let test = PricePersistenceTest::new(3.0);
        let result = test.test(&prices).unwrap();
        assert!(result.max_run_length == 1);
        assert!(!result.is_persistent);
    }

    #[test]
    fn test_persistence_empty() {
        let test = PricePersistenceTest::new(3.0);
        assert!(test.test(&[]).is_err());
    }

    #[test]
    fn test_convergence_pattern() {
        // Two firms converging to the same price at the same time
        let f1: Vec<f64> = (0..100).map(|i| 5.0 - 2.0 * (-0.05 * i as f64).exp()).collect();
        let f2: Vec<f64> = (0..100).map(|i| 5.0 - 2.0 * (-0.05 * i as f64).exp()).collect();
        let test = ConvergencePatternTest::new(3.0, 10);
        let result = test.test(&[f1, f2]).unwrap();
        assert_eq!(result.convergence_speeds.len(), 2);
    }

    #[test]
    fn test_convergence_pattern_error() {
        let test = ConvergencePatternTest::new(3.0, 10);
        assert!(test.test(&[]).is_err());
    }

    #[test]
    fn test_dispersion_low() {
        // All firms price nearly identically → low dispersion
        let f1 = vec![5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0];
        let f2 = vec![5.01, 5.01, 5.01, 5.01, 5.01, 5.01, 5.01, 5.01, 5.01, 5.01];
        let test = PriceDispersionTest::new();
        let result = test.test(&[f1, f2]).unwrap();
        assert!(result.mean_cv < 0.01);
    }

    #[test]
    fn test_dispersion_high() {
        let f1 = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0];
        let f2 = vec![8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0];
        let test = PriceDispersionTest::new();
        let result = test.test(&[f1, f2]).unwrap();
        assert!(result.mean_cv > 0.3);
    }

    #[test]
    fn test_dispersion_empty() {
        let test = PriceDispersionTest::new();
        assert!(test.test(&[vec![]]).is_err());
    }

    #[test]
    fn test_edgeworth_cycling() {
        // Sinusoidal prices (cycling)
        let prices: Vec<f64> = (0..200).map(|i| 3.0 + (i as f64 * 0.3).sin()).collect();
        let test = EdgeworthCycleTest::new(30);
        let result = test.test(&prices).unwrap();
        assert!(result.dominant_autocorrelation > 0.0);
        assert!(result.dominant_period > 1);
    }

    #[test]
    fn test_edgeworth_stable() {
        // Constant prices (stable)
        let prices = vec![5.0; 200];
        let test = EdgeworthCycleTest::new(20);
        let result = test.test(&prices).unwrap();
        assert!(result.is_stable);
        assert!(!result.is_cycling);
    }

    #[test]
    fn test_edgeworth_short_error() {
        let test = EdgeworthCycleTest::new(50);
        let prices = vec![1.0; 10];
        assert!(test.test(&prices).is_err());
    }

    #[test]
    fn test_edgeworth_multi_firm() {
        let f1: Vec<f64> = (0..200).map(|i| 3.0 + (i as f64 * 0.3).sin()).collect();
        let f2: Vec<f64> = (0..200).map(|i| 3.0 + (i as f64 * 0.3).cos()).collect();
        let test = EdgeworthCycleTest::new(20);
        let results = test.test_multi_firm(&[f1, f2]).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_run_lengths() {
        let indicators = vec![true, true, true, false, true, true, false, false, true];
        let runs = compute_run_lengths(&indicators, true);
        assert_eq!(runs, vec![3, 2, 1]);
    }

    #[test]
    fn test_run_lengths_all_true() {
        let indicators = vec![true; 10];
        let runs = compute_run_lengths(&indicators, true);
        assert_eq!(runs, vec![10]);
    }

    #[test]
    fn test_autocorrelation_zero_lag() {
        let xs: Vec<f64> = (0..100).map(|i| i as f64).collect();
        // lag=0 is always 1 (but our function starts at lag=1)
        let ac1 = autocorrelation(&xs, 1);
        assert!(ac1 > 0.9); // Strong positive autocorrelation for linear trend
    }

    #[test]
    fn test_lagged_correlation() {
        let x: Vec<f64> = (0..50).map(|i| (i as f64 * 0.2).sin()).collect();
        let y = x.clone();
        let r = lagged_correlation(&x, &y, 0);
        assert!(approx_eq(r, 1.0, 1e-10));
    }
}
