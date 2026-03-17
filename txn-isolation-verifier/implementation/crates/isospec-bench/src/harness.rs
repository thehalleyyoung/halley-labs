//! Benchmark harness for running experiments with statistical analysis.

use std::collections::HashMap;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for a single benchmark experiment.
#[derive(Debug, Clone)]
pub struct ExperimentConfig {
    /// Human-readable experiment name.
    pub name: String,
    /// Number of warm-up iterations (not included in measurements).
    pub warmup_iterations: usize,
    /// Number of measured iterations.
    pub measurement_iterations: usize,
    /// Optional timeout per iteration.
    pub timeout: Option<Duration>,
    /// Key-value parameters carried along for reporting.
    pub parameters: HashMap<String, String>,
}

impl ExperimentConfig {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            warmup_iterations: 3,
            measurement_iterations: 10,
            timeout: None,
            parameters: HashMap::new(),
        }
    }

    pub fn with_warmup(mut self, n: usize) -> Self {
        self.warmup_iterations = n;
        self
    }

    pub fn with_iterations(mut self, n: usize) -> Self {
        self.measurement_iterations = n;
        self
    }

    pub fn with_timeout(mut self, d: Duration) -> Self {
        self.timeout = Some(d);
        self
    }

    pub fn with_param(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.parameters.insert(key.into(), value.into());
        self
    }
}

impl Default for ExperimentConfig {
    fn default() -> Self {
        Self::new("default")
    }
}

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------

/// Outcome of a single iteration.
#[derive(Debug, Clone)]
pub struct IterationResult {
    pub duration: Duration,
    pub timed_out: bool,
    pub extra: HashMap<String, f64>,
}

impl IterationResult {
    pub fn from_duration(d: Duration) -> Self {
        Self { duration: d, timed_out: false, extra: HashMap::new() }
    }

    pub fn with_extra(mut self, key: impl Into<String>, value: f64) -> Self {
        self.extra.insert(key.into(), value);
        self
    }
}

/// Aggregated result of a benchmark experiment.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub config: ExperimentConfig,
    pub iterations: Vec<IterationResult>,
    pub stats: Statistics,
    pub warmup_duration: Duration,
    pub total_duration: Duration,
    pub timestamp: u64,
}

impl BenchmarkResult {
    pub fn iteration_count(&self) -> usize {
        self.iterations.len()
    }

    pub fn timed_out_count(&self) -> usize {
        self.iterations.iter().filter(|i| i.timed_out).count()
    }

    pub fn success_rate(&self) -> f64 {
        let total = self.iterations.len();
        if total == 0 {
            return 0.0;
        }
        let ok = total - self.timed_out_count();
        ok as f64 / total as f64
    }

    pub fn throughput_per_second(&self) -> f64 {
        let secs = self.total_duration.as_secs_f64();
        if secs < 1e-15 {
            return 0.0;
        }
        self.iterations.len() as f64 / secs
    }

    pub fn extra_stat(&self, key: &str) -> Option<Statistics> {
        let values: Vec<f64> = self.iterations.iter()
            .filter_map(|it| it.extra.get(key).copied())
            .collect();
        if values.is_empty() {
            None
        } else {
            Some(Statistics::from_samples(&values))
        }
    }
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Descriptive statistics over a set of duration samples.
#[derive(Debug, Clone)]
pub struct Statistics {
    pub count: usize,
    pub mean: f64,
    pub median: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub p25: f64,
    pub p75: f64,
    pub p90: f64,
    pub p95: f64,
    pub p99: f64,
}

impl Statistics {
    /// Build statistics from a slice of `f64` samples.
    pub fn from_samples(samples: &[f64]) -> Self {
        if samples.is_empty() {
            return Self::zero();
        }
        let mut sorted: Vec<f64> = samples.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = sorted.len();
        let sum: f64 = sorted.iter().sum();
        let mean = sum / n as f64;

        let median = percentile_sorted(&sorted, 50.0);

        let variance = if n > 1 {
            sorted.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64
        } else {
            0.0
        };
        let std_dev = variance.sqrt();

        Self {
            count: n,
            mean,
            median,
            std_dev,
            min: sorted[0],
            max: sorted[n - 1],
            p25: percentile_sorted(&sorted, 25.0),
            p75: percentile_sorted(&sorted, 75.0),
            p90: percentile_sorted(&sorted, 90.0),
            p95: percentile_sorted(&sorted, 95.0),
            p99: percentile_sorted(&sorted, 99.0),
        }
    }

    /// Build statistics from durations (converted to seconds).
    pub fn from_durations(durations: &[Duration]) -> Self {
        let secs: Vec<f64> = durations.iter().map(|d| d.as_secs_f64()).collect();
        Self::from_samples(&secs)
    }

    /// All-zero sentinel for empty sample sets.
    pub fn zero() -> Self {
        Self {
            count: 0, mean: 0.0, median: 0.0, std_dev: 0.0,
            min: 0.0, max: 0.0, p25: 0.0, p75: 0.0,
            p90: 0.0, p95: 0.0, p99: 0.0,
        }
    }

    /// Coefficient of variation (relative std-dev).
    pub fn coefficient_of_variation(&self) -> f64 {
        if self.mean.abs() < 1e-15 {
            0.0
        } else {
            self.std_dev / self.mean
        }
    }

    /// Return `true` when measurements look stable (CV < threshold).
    pub fn is_stable(&self, cv_threshold: f64) -> bool {
        self.count >= 3 && self.coefficient_of_variation() < cv_threshold
    }

    /// Inter-quartile range.
    pub fn iqr(&self) -> f64 {
        self.p75 - self.p25
    }

    /// Identify outlier bounds using the IQR method.
    pub fn outlier_bounds(&self) -> (f64, f64) {
        let iqr = self.iqr();
        (self.p25 - 1.5 * iqr, self.p75 + 1.5 * iqr)
    }
}

/// Linear-interpolation percentile on a *sorted* slice.
fn percentile_sorted(sorted: &[f64], pct: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let rank = pct / 100.0 * (sorted.len() - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    let frac = rank - lower as f64;
    if lower == upper || upper >= sorted.len() {
        sorted[lower]
    } else {
        sorted[lower] * (1.0 - frac) + sorted[upper] * frac
    }
}

// ---------------------------------------------------------------------------
// Comparison
// ---------------------------------------------------------------------------

/// Side-by-side comparison of two benchmark results.
#[derive(Debug, Clone)]
pub struct Comparison {
    pub baseline_name: String,
    pub candidate_name: String,
    pub speedup: f64,
    pub mean_diff_seconds: f64,
    pub median_diff_seconds: f64,
    pub baseline_stats: Statistics,
    pub candidate_stats: Statistics,
}

impl Comparison {
    pub fn compare(baseline: &BenchmarkResult, candidate: &BenchmarkResult) -> Self {
        let b = &baseline.stats;
        let c = &candidate.stats;
        let speedup = if c.mean.abs() < 1e-15 { 0.0 } else { b.mean / c.mean };
        Self {
            baseline_name: baseline.config.name.clone(),
            candidate_name: candidate.config.name.clone(),
            speedup,
            mean_diff_seconds: c.mean - b.mean,
            median_diff_seconds: c.median - b.median,
            baseline_stats: b.clone(),
            candidate_stats: c.clone(),
        }
    }

    pub fn is_faster(&self) -> bool {
        self.speedup > 1.0
    }

    pub fn is_slower(&self) -> bool {
        self.speedup < 1.0
    }

    pub fn percent_change(&self) -> f64 {
        if self.baseline_stats.mean.abs() < 1e-15 {
            0.0
        } else {
            (self.candidate_stats.mean - self.baseline_stats.mean) / self.baseline_stats.mean * 100.0
        }
    }
}

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

/// The main benchmark harness that drives experiment execution.
pub struct BenchmarkHarness {
    results: Vec<BenchmarkResult>,
    global_start: Instant,
    default_config: ExperimentConfig,
}

impl BenchmarkHarness {
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
            global_start: Instant::now(),
            default_config: ExperimentConfig::default(),
        }
    }

    pub fn with_default_config(mut self, config: ExperimentConfig) -> Self {
        self.default_config = config;
        self
    }

    /// Run a benchmark using the supplied closure.
    ///
    /// The closure receives the iteration index and returns an optional
    /// `HashMap<String,f64>` of extra metrics for that iteration.
    pub fn run<F>(&mut self, config: &ExperimentConfig, mut work: F) -> BenchmarkResult
    where
        F: FnMut(usize) -> Option<HashMap<String, f64>>,
    {
        // Warm-up phase
        let warmup_start = Instant::now();
        for i in 0..config.warmup_iterations {
            let _ = work(i);
        }
        let warmup_duration = warmup_start.elapsed();

        // Measurement phase
        let measure_start = Instant::now();
        let mut iterations = Vec::with_capacity(config.measurement_iterations);

        for i in 0..config.measurement_iterations {
            let iter_start = Instant::now();
            let extra = work(i);
            let elapsed = iter_start.elapsed();

            let timed_out = config.timeout.map_or(false, |t| elapsed > t);
            let mut result = IterationResult::from_duration(elapsed);
            result.timed_out = timed_out;
            if let Some(map) = extra {
                result.extra = map;
            }
            iterations.push(result);
        }

        let total_duration = measure_start.elapsed();

        let durations: Vec<Duration> = iterations.iter().map(|i| i.duration).collect();
        let stats = Statistics::from_durations(&durations);

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let result = BenchmarkResult {
            config: config.clone(),
            iterations,
            stats,
            warmup_duration,
            total_duration,
            timestamp: now,
        };

        self.results.push(result.clone());
        result
    }

    /// Run a simple benchmark with the default config.
    pub fn run_simple<F>(&mut self, name: &str, mut work: F) -> BenchmarkResult
    where
        F: FnMut(usize),
    {
        let cfg = ExperimentConfig::new(name)
            .with_warmup(self.default_config.warmup_iterations)
            .with_iterations(self.default_config.measurement_iterations);
        self.run(&cfg, |i| { work(i); None })
    }

    /// Return all recorded results.
    pub fn results(&self) -> &[BenchmarkResult] {
        &self.results
    }

    /// Elapsed wall-clock time since harness creation.
    pub fn elapsed(&self) -> Duration {
        self.global_start.elapsed()
    }

    /// Compare the last two results.
    pub fn compare_last_two(&self) -> Option<Comparison> {
        if self.results.len() < 2 {
            return None;
        }
        let n = self.results.len();
        Some(Comparison::compare(&self.results[n - 2], &self.results[n - 1]))
    }

    /// Compare two named results.
    pub fn compare_by_name(&self, baseline: &str, candidate: &str) -> Option<Comparison> {
        let b = self.results.iter().find(|r| r.config.name == baseline)?;
        let c = self.results.iter().find(|r| r.config.name == candidate)?;
        Some(Comparison::compare(b, c))
    }

    /// Clear all accumulated results.
    pub fn reset(&mut self) {
        self.results.clear();
        self.global_start = Instant::now();
    }

    /// Compute a comparison matrix for all recorded results.
    pub fn comparison_matrix(&self) -> Vec<Vec<Option<Comparison>>> {
        let n = self.results.len();
        let mut matrix = Vec::with_capacity(n);
        for i in 0..n {
            let mut row = Vec::with_capacity(n);
            for j in 0..n {
                if i == j {
                    row.push(None);
                } else {
                    row.push(Some(Comparison::compare(&self.results[i], &self.results[j])));
                }
            }
            matrix.push(row);
        }
        matrix
    }
}

impl Default for BenchmarkHarness {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Suite
// ---------------------------------------------------------------------------

/// A named collection of experiments to run together.
pub struct BenchmarkSuite {
    pub name: String,
    pub experiments: Vec<ExperimentConfig>,
    pub tags: Vec<String>,
}

impl BenchmarkSuite {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            experiments: Vec::new(),
            tags: Vec::new(),
        }
    }

    pub fn add_experiment(&mut self, config: ExperimentConfig) {
        self.experiments.push(config);
    }

    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    pub fn len(&self) -> usize {
        self.experiments.len()
    }

    pub fn is_empty(&self) -> bool {
        self.experiments.is_empty()
    }

    /// Run every experiment in the suite.
    pub fn run_all<F>(&self, harness: &mut BenchmarkHarness, mut factory: F) -> Vec<BenchmarkResult>
    where
        F: FnMut(&ExperimentConfig) -> Box<dyn FnMut(usize) -> Option<HashMap<String, f64>>>,
    {
        let mut results = Vec::with_capacity(self.experiments.len());
        for exp in &self.experiments {
            let mut work = factory(exp);
            let result = harness.run(exp, |i| work(i));
            results.push(result);
        }
        results
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_statistics_basic() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = Statistics::from_samples(&samples);
        assert_eq!(stats.count, 5);
        assert!((stats.mean - 3.0).abs() < 1e-9);
        assert!((stats.median - 3.0).abs() < 1e-9);
        assert!((stats.min - 1.0).abs() < 1e-9);
        assert!((stats.max - 5.0).abs() < 1e-9);
        assert!(stats.std_dev > 0.0);
    }

    #[test]
    fn test_statistics_single_sample() {
        let stats = Statistics::from_samples(&[42.0]);
        assert_eq!(stats.count, 1);
        assert!((stats.mean - 42.0).abs() < 1e-9);
        assert!((stats.median - 42.0).abs() < 1e-9);
        assert!((stats.std_dev - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_statistics_empty() {
        let stats = Statistics::from_samples(&[]);
        assert_eq!(stats.count, 0);
        assert!((stats.mean - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile_interpolation() {
        let sorted = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let p50 = percentile_sorted(&sorted, 50.0);
        assert!((p50 - 30.0).abs() < 1e-9);
        let p0 = percentile_sorted(&sorted, 0.0);
        assert!((p0 - 10.0).abs() < 1e-9);
        let p100 = percentile_sorted(&sorted, 100.0);
        assert!((p100 - 50.0).abs() < 1e-9);
    }

    #[test]
    fn test_harness_run_simple() {
        let mut harness = BenchmarkHarness::new();
        let config = ExperimentConfig::new("test_bench")
            .with_warmup(1)
            .with_iterations(5);
        let mut counter = 0u64;
        let result = harness.run(&config, |_i| {
            counter += 1;
            let mut extra = HashMap::new();
            extra.insert("counter".to_string(), counter as f64);
            Some(extra)
        });
        assert_eq!(result.iteration_count(), 5);
        assert_eq!(result.timed_out_count(), 0);
        assert!((result.success_rate() - 1.0).abs() < 1e-9);
        assert!(result.throughput_per_second() > 0.0);
    }

    #[test]
    fn test_harness_comparison() {
        let mut harness = BenchmarkHarness::new();

        let cfg1 = ExperimentConfig::new("fast").with_warmup(0).with_iterations(3);
        harness.run(&cfg1, |_| { std::thread::sleep(Duration::from_micros(10)); None });

        let cfg2 = ExperimentConfig::new("slow").with_warmup(0).with_iterations(3);
        harness.run(&cfg2, |_| { std::thread::sleep(Duration::from_micros(100)); None });

        let cmp = harness.compare_last_two().unwrap();
        assert_eq!(cmp.baseline_name, "fast");
        assert_eq!(cmp.candidate_name, "slow");
        // slow should be slower
        assert!(cmp.is_slower());
    }

    #[test]
    fn test_comparison_percent_change() {
        let stats_a = Statistics::from_samples(&[1.0, 1.0, 1.0]);
        let stats_b = Statistics::from_samples(&[2.0, 2.0, 2.0]);
        let cmp = Comparison {
            baseline_name: "a".to_string(),
            candidate_name: "b".to_string(),
            speedup: 0.5,
            mean_diff_seconds: 1.0,
            median_diff_seconds: 1.0,
            baseline_stats: stats_a,
            candidate_stats: stats_b,
        };
        assert!((cmp.percent_change() - 100.0).abs() < 1e-9);
    }

    #[test]
    fn test_coefficient_of_variation() {
        let stats = Statistics::from_samples(&[10.0, 10.0, 10.0, 10.0]);
        assert!((stats.coefficient_of_variation() - 0.0).abs() < 1e-9);
        assert!(stats.is_stable(0.1));
    }

    #[test]
    fn test_outlier_bounds() {
        let stats = Statistics::from_samples(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let (lo, hi) = stats.outlier_bounds();
        assert!(lo < stats.min);
        assert!(hi > stats.max);
    }

    #[test]
    fn test_benchmark_suite() {
        let mut suite = BenchmarkSuite::new("micro");
        suite.add_experiment(ExperimentConfig::new("a").with_warmup(0).with_iterations(2));
        suite.add_experiment(ExperimentConfig::new("b").with_warmup(0).with_iterations(2));
        assert_eq!(suite.len(), 2);
        assert!(!suite.is_empty());

        let mut harness = BenchmarkHarness::new();
        let results = suite.run_all(&mut harness, |_cfg| {
            Box::new(|_i| None)
        });
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_statistics_from_durations() {
        let durations = vec![
            Duration::from_millis(10),
            Duration::from_millis(20),
            Duration::from_millis(30),
        ];
        let stats = Statistics::from_durations(&durations);
        assert_eq!(stats.count, 3);
        assert!((stats.mean - 0.02).abs() < 0.001);
    }

    #[test]
    fn test_extra_stat() {
        let mut harness = BenchmarkHarness::new();
        let cfg = ExperimentConfig::new("extras").with_warmup(0).with_iterations(4);
        let result = harness.run(&cfg, |i| {
            let mut m = HashMap::new();
            m.insert("ops".to_string(), (i + 1) as f64 * 10.0);
            Some(m)
        });
        let ops_stat = result.extra_stat("ops").unwrap();
        assert_eq!(ops_stat.count, 4);
        assert!((ops_stat.mean - 25.0).abs() < 1e-9);
    }

    #[test]
    fn test_comparison_matrix() {
        let mut harness = BenchmarkHarness::new();
        for name in &["a", "b", "c"] {
            let cfg = ExperimentConfig::new(*name).with_warmup(0).with_iterations(2);
            harness.run(&cfg, |_| None);
        }
        let matrix = harness.comparison_matrix();
        assert_eq!(matrix.len(), 3);
        assert!(matrix[0][0].is_none()); // diagonal
        assert!(matrix[0][1].is_some());
    }
}
