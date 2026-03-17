//! Benchmark definitions and execution for performance evaluation.
//!
//! Measures wall-clock time, throughput, and resource usage for the localizer.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Configuration for a benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    pub name: String,
    pub warmup_iterations: usize,
    pub measurement_iterations: usize,
    pub pipeline_stages: usize,
    pub test_case_counts: Vec<usize>,
    pub transformation_counts: Vec<usize>,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            name: "default_benchmark".to_string(),
            warmup_iterations: 3,
            measurement_iterations: 10,
            pipeline_stages: 4,
            test_case_counts: vec![100, 500, 1000, 5000],
            transformation_counts: vec![5, 10, 15],
        }
    }
}

/// A single benchmark definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Benchmark {
    pub name: String,
    pub description: String,
    pub category: BenchmarkCategory,
    pub parameters: HashMap<String, String>,
}

/// Categories of benchmarks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BenchmarkCategory {
    /// Measures localization algorithm throughput.
    Localization,
    /// Measures shrinking performance.
    Shrinking,
    /// Measures grammar checking throughput.
    GrammarChecking,
    /// Measures differential computation.
    Differential,
    /// Measures end-to-end pipeline.
    EndToEnd,
    /// Measures discriminability matrix computation.
    Discriminability,
}

/// Result from executing a single benchmark.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub benchmark_name: String,
    pub category: String,
    pub iterations: usize,
    pub total_time: Duration,
    pub mean_time: Duration,
    pub median_time: Duration,
    pub min_time: Duration,
    pub max_time: Duration,
    pub std_dev: Duration,
    pub throughput: f64,
    pub samples: Vec<Duration>,
    pub metadata: HashMap<String, String>,
}

impl BenchmarkResult {
    /// Compute a result from a vector of duration samples.
    pub fn from_samples(
        name: impl Into<String>,
        category: impl Into<String>,
        samples: Vec<Duration>,
    ) -> Self {
        let n = samples.len();
        if n == 0 {
            return Self {
                benchmark_name: name.into(),
                category: category.into(),
                iterations: 0,
                total_time: Duration::ZERO,
                mean_time: Duration::ZERO,
                median_time: Duration::ZERO,
                min_time: Duration::ZERO,
                max_time: Duration::ZERO,
                std_dev: Duration::ZERO,
                throughput: 0.0,
                samples: Vec::new(),
                metadata: HashMap::new(),
            };
        }

        let total: Duration = samples.iter().sum();
        let mean = total / n as u32;

        let mut sorted = samples.clone();
        sorted.sort();
        let median = if n % 2 == 0 {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2
        } else {
            sorted[n / 2]
        };

        let mean_nanos = mean.as_nanos() as f64;
        let variance: f64 = samples
            .iter()
            .map(|s| {
                let diff = s.as_nanos() as f64 - mean_nanos;
                diff * diff
            })
            .sum::<f64>()
            / n as f64;
        let std_dev = Duration::from_nanos(variance.sqrt() as u64);

        let throughput = if mean.as_secs_f64() > 0.0 {
            1.0 / mean.as_secs_f64()
        } else {
            f64::INFINITY
        };

        Self {
            benchmark_name: name.into(),
            category: category.into(),
            iterations: n,
            total_time: total,
            mean_time: mean,
            median_time: median,
            min_time: sorted[0],
            max_time: *sorted.last().unwrap(),
            std_dev,
            throughput,
            samples,
            metadata: HashMap::new(),
        }
    }

    /// Format the result as a human-readable string.
    pub fn format_summary(&self) -> String {
        format!(
            "{} ({}) — mean: {:.2}ms, median: {:.2}ms, min: {:.2}ms, max: {:.2}ms, σ: {:.2}ms, throughput: {:.1}/s (n={})",
            self.benchmark_name,
            self.category,
            self.mean_time.as_secs_f64() * 1000.0,
            self.median_time.as_secs_f64() * 1000.0,
            self.min_time.as_secs_f64() * 1000.0,
            self.max_time.as_secs_f64() * 1000.0,
            self.std_dev.as_secs_f64() * 1000.0,
            self.throughput,
            self.iterations,
        )
    }
}

/// A suite of benchmarks to run together.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkSuite {
    pub name: String,
    pub description: String,
    pub benchmarks: Vec<Benchmark>,
    pub results: Vec<BenchmarkResult>,
    pub config: BenchmarkConfig,
}

impl BenchmarkSuite {
    pub fn new(name: impl Into<String>, config: BenchmarkConfig) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            benchmarks: Vec::new(),
            results: Vec::new(),
            config,
        }
    }

    pub fn add_benchmark(&mut self, benchmark: Benchmark) {
        self.benchmarks.push(benchmark);
    }

    pub fn record_result(&mut self, result: BenchmarkResult) {
        self.results.push(result);
    }

    /// Run a closure `iterations` times and record the timing.
    pub fn measure<F: FnMut()>(
        &mut self,
        name: impl Into<String>,
        category: impl Into<String>,
        mut f: F,
    ) -> BenchmarkResult {
        let name = name.into();
        let category = category.into();

        // Warmup.
        for _ in 0..self.config.warmup_iterations {
            f();
        }

        // Measurement.
        let mut samples = Vec::with_capacity(self.config.measurement_iterations);
        for _ in 0..self.config.measurement_iterations {
            let start = Instant::now();
            f();
            samples.push(start.elapsed());
        }

        let result = BenchmarkResult::from_samples(&name, &category, samples);
        self.results.push(result.clone());
        result
    }

    /// Generate a markdown report of all results.
    pub fn generate_report(&self) -> String {
        let mut report = format!("# Benchmark Report: {}\n\n", self.name);
        if !self.description.is_empty() {
            report.push_str(&format!("{}\n\n", self.description));
        }

        report.push_str("| Benchmark | Category | Mean (ms) | Median (ms) | Min (ms) | Max (ms) | σ (ms) | Throughput (/s) | N |\n");
        report.push_str("|-----------|----------|-----------|-------------|----------|----------|--------|-----------------|---|\n");

        for result in &self.results {
            report.push_str(&format!(
                "| {} | {} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.1} | {} |\n",
                result.benchmark_name,
                result.category,
                result.mean_time.as_secs_f64() * 1000.0,
                result.median_time.as_secs_f64() * 1000.0,
                result.min_time.as_secs_f64() * 1000.0,
                result.max_time.as_secs_f64() * 1000.0,
                result.std_dev.as_secs_f64() * 1000.0,
                result.throughput,
                result.iterations,
            ));
        }

        report
    }

    /// Compute speedup ratios between pairs of benchmarks.
    pub fn compare_results(&self, baseline: &str, candidate: &str) -> Option<SpeedupComparison> {
        let base = self.results.iter().find(|r| r.benchmark_name == baseline)?;
        let cand = self.results.iter().find(|r| r.benchmark_name == candidate)?;

        let speedup = base.mean_time.as_secs_f64() / cand.mean_time.as_secs_f64();

        Some(SpeedupComparison {
            baseline_name: baseline.to_string(),
            candidate_name: candidate.to_string(),
            baseline_mean: base.mean_time,
            candidate_mean: cand.mean_time,
            speedup,
            is_faster: speedup > 1.0,
        })
    }
}

/// Comparison between two benchmark results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeedupComparison {
    pub baseline_name: String,
    pub candidate_name: String,
    pub baseline_mean: Duration,
    pub candidate_mean: Duration,
    pub speedup: f64,
    pub is_faster: bool,
}

impl SpeedupComparison {
    pub fn format_summary(&self) -> String {
        if self.is_faster {
            format!(
                "{} is {:.2}× faster than {}",
                self.candidate_name, self.speedup, self.baseline_name
            )
        } else {
            format!(
                "{} is {:.2}× slower than {}",
                self.candidate_name,
                1.0 / self.speedup,
                self.baseline_name
            )
        }
    }
}

/// Scalability analysis: measure how performance scales with input size.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalabilityAnalysis {
    pub name: String,
    pub data_points: Vec<ScalabilityPoint>,
    pub complexity_estimate: String,
}

/// A single data point in a scalability analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalabilityPoint {
    pub input_size: usize,
    pub mean_time: Duration,
    pub throughput: f64,
}

impl ScalabilityAnalysis {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            data_points: Vec::new(),
            complexity_estimate: String::new(),
        }
    }

    pub fn add_point(&mut self, input_size: usize, mean_time: Duration, throughput: f64) {
        self.data_points.push(ScalabilityPoint {
            input_size,
            mean_time,
            throughput,
        });
    }

    /// Estimate the empirical complexity from the data points.
    pub fn estimate_complexity(&mut self) {
        if self.data_points.len() < 2 {
            self.complexity_estimate = "insufficient_data".to_string();
            return;
        }

        // Simple log-log regression to estimate O(n^k) complexity.
        let n = self.data_points.len() as f64;
        let log_sizes: Vec<f64> = self
            .data_points
            .iter()
            .map(|p| (p.input_size as f64).ln())
            .collect();
        let log_times: Vec<f64> = self
            .data_points
            .iter()
            .map(|p| p.mean_time.as_secs_f64().ln())
            .collect();

        let mean_x = log_sizes.iter().sum::<f64>() / n;
        let mean_y = log_times.iter().sum::<f64>() / n;

        let numerator: f64 = log_sizes
            .iter()
            .zip(&log_times)
            .map(|(x, y)| (x - mean_x) * (y - mean_y))
            .sum();
        let denominator: f64 = log_sizes.iter().map(|x| (x - mean_x).powi(2)).sum();

        if denominator.abs() < f64::EPSILON {
            self.complexity_estimate = "constant".to_string();
            return;
        }

        let slope = numerator / denominator;

        self.complexity_estimate = if slope < 0.2 {
            "O(1)".to_string()
        } else if slope < 0.8 {
            format!("O(n^{:.2}) ≈ O(√n)", slope)
        } else if slope < 1.3 {
            format!("O(n^{:.2}) ≈ O(n)", slope)
        } else if slope < 1.7 {
            format!("O(n^{:.2}) ≈ O(n·log(n))", slope)
        } else if slope < 2.3 {
            format!("O(n^{:.2}) ≈ O(n²)", slope)
        } else {
            format!("O(n^{:.2})", slope)
        };
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_result_from_samples() {
        let samples = vec![
            Duration::from_millis(10),
            Duration::from_millis(12),
            Duration::from_millis(11),
            Duration::from_millis(9),
            Duration::from_millis(13),
        ];
        let result = BenchmarkResult::from_samples("test", "localization", samples);

        assert_eq!(result.iterations, 5);
        assert_eq!(result.min_time, Duration::from_millis(9));
        assert_eq!(result.max_time, Duration::from_millis(13));
        assert!(result.throughput > 0.0);
    }

    #[test]
    fn test_benchmark_suite_measure() {
        let config = BenchmarkConfig {
            warmup_iterations: 1,
            measurement_iterations: 5,
            ..Default::default()
        };
        let mut suite = BenchmarkSuite::new("test_suite", config);

        let mut counter = 0u64;
        let result = suite.measure("counting", "unit_test", || {
            counter += 1;
        });

        assert_eq!(result.iterations, 5);
        assert!(result.mean_time < Duration::from_millis(1));
    }

    #[test]
    fn test_benchmark_comparison() {
        let mut suite = BenchmarkSuite::new("compare_test", BenchmarkConfig::default());
        suite.record_result(BenchmarkResult::from_samples(
            "slow",
            "test",
            vec![Duration::from_millis(100); 5],
        ));
        suite.record_result(BenchmarkResult::from_samples(
            "fast",
            "test",
            vec![Duration::from_millis(10); 5],
        ));

        let comparison = suite.compare_results("slow", "fast").unwrap();
        assert!(comparison.is_faster);
        assert!((comparison.speedup - 10.0).abs() < 1.0);
    }

    #[test]
    fn test_scalability_analysis() {
        let mut analysis = ScalabilityAnalysis::new("linear_test");
        // Simulate O(n) behavior.
        for &size in &[100, 200, 400, 800, 1600] {
            let time = Duration::from_micros(size as u64 * 10);
            let throughput = 1.0 / time.as_secs_f64();
            analysis.add_point(size, time, throughput);
        }
        analysis.estimate_complexity();
        assert!(analysis.complexity_estimate.contains("O(n"));
    }

    #[test]
    fn test_report_generation() {
        let mut suite = BenchmarkSuite::new("report_test", BenchmarkConfig::default());
        suite.record_result(BenchmarkResult::from_samples(
            "benchmark_1",
            "localization",
            vec![Duration::from_millis(50); 3],
        ));
        let report = suite.generate_report();
        assert!(report.contains("benchmark_1"));
        assert!(report.contains("localization"));
    }

    #[test]
    fn test_empty_benchmark_result() {
        let result = BenchmarkResult::from_samples("empty", "test", Vec::new());
        assert_eq!(result.iterations, 0);
        assert_eq!(result.total_time, Duration::ZERO);
    }
}
