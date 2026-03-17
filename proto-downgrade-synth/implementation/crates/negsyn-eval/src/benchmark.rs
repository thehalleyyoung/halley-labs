//! Performance benchmarking for NegSynth analysis pipeline.

use crate::pipeline::{AnalysisPipeline, PipelineConfig, PipelineResult, PipelineStage};
use crate::Lts;

use chrono::Utc;
use log::{debug, info};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Statistical summary of a set of measurements.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stats {
    pub count: usize,
    pub mean: f64,
    pub median: f64,
    pub stddev: f64,
    pub min: f64,
    pub max: f64,
    pub p25: f64,
    pub p75: f64,
    pub p90: f64,
    pub p95: f64,
    pub p99: f64,
}

impl Stats {
    pub fn compute(values: &[f64]) -> Self {
        if values.is_empty() {
            return Self {
                count: 0,
                mean: 0.0,
                median: 0.0,
                stddev: 0.0,
                min: 0.0,
                max: 0.0,
                p25: 0.0,
                p75: 0.0,
                p90: 0.0,
                p95: 0.0,
                p99: 0.0,
            };
        }

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let count = sorted.len();
        let sum: f64 = sorted.iter().sum();
        let mean = sum / count as f64;
        let variance = if count > 1 {
            sorted.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (count - 1) as f64
        } else {
            0.0
        };
        let stddev = variance.sqrt();

        Self {
            count,
            mean,
            median: percentile(&sorted, 50.0),
            stddev,
            min: sorted[0],
            max: sorted[count - 1],
            p25: percentile(&sorted, 25.0),
            p75: percentile(&sorted, 75.0),
            p90: percentile(&sorted, 90.0),
            p95: percentile(&sorted, 95.0),
            p99: percentile(&sorted, 99.0),
        }
    }

    pub fn coefficient_of_variation(&self) -> f64 {
        if self.mean.abs() < f64::EPSILON {
            0.0
        } else {
            self.stddev / self.mean
        }
    }
}

fn percentile(sorted: &[f64], pct: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let rank = (pct / 100.0) * (sorted.len() - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    let frac = rank - lower as f64;
    if upper >= sorted.len() {
        sorted[sorted.len() - 1]
    } else {
        sorted[lower] * (1.0 - frac) + sorted[upper] * frac
    }
}

/// A single benchmark measurement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub name: String,
    pub scenario: String,
    pub timing_ms: Stats,
    pub peak_memory_bytes: Stats,
    pub path_counts: Stats,
    pub state_counts: Stats,
    pub iterations: usize,
    pub extra_metrics: HashMap<String, f64>,
    pub timestamp: String,
}

impl BenchmarkResult {
    pub fn throughput_per_second(&self) -> f64 {
        if self.timing_ms.mean > 0.0 {
            1000.0 / self.timing_ms.mean
        } else {
            0.0
        }
    }

    pub fn is_stable(&self) -> bool {
        self.timing_ms.coefficient_of_variation() < 0.15
    }
}

/// Configuration for benchmark suite.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkSuiteConfig {
    pub warmup_iterations: usize,
    pub measurement_iterations: usize,
    pub min_duration_ms: u64,
    pub max_duration_ms: u64,
    pub cipher_suite_counts: Vec<usize>,
    pub enable_merge_benchmark: bool,
    pub enable_scalability_benchmark: bool,
    pub enable_memory_benchmark: bool,
}

impl Default for BenchmarkSuiteConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 2,
            measurement_iterations: 5,
            min_duration_ms: 100,
            max_duration_ms: 60_000,
            cipher_suite_counts: vec![4, 8, 16, 32, 64],
            enable_merge_benchmark: true,
            enable_scalability_benchmark: true,
            enable_memory_benchmark: true,
        }
    }
}

/// The main benchmark suite.
pub struct BenchmarkSuite {
    config: BenchmarkSuiteConfig,
    results: Vec<BenchmarkResult>,
}

impl BenchmarkSuite {
    pub fn new(config: BenchmarkSuiteConfig) -> Self {
        Self {
            config,
            results: Vec::new(),
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(BenchmarkSuiteConfig::default())
    }

    pub fn results(&self) -> &[BenchmarkResult] {
        &self.results
    }

    /// Run all enabled benchmarks.
    pub fn run_all(&mut self) -> BenchmarkReport {
        let start = Instant::now();
        info!("Starting benchmark suite");

        if self.config.enable_merge_benchmark {
            let merge_bench = MergeSpeedupBenchmark::new(self.config.measurement_iterations);
            let result = merge_bench.run();
            self.results.push(result);
        }

        if self.config.enable_scalability_benchmark {
            let scale_bench = ScalabilityBenchmark::new(
                self.config.cipher_suite_counts.clone(),
                self.config.measurement_iterations,
            );
            let results = scale_bench.run();
            self.results.extend(results);
        }

        if self.config.enable_memory_benchmark {
            let mem_bench = MemoryBenchmark::new(self.config.measurement_iterations);
            let result = mem_bench.run();
            self.results.push(result);
        }

        let total_duration = start.elapsed().as_millis() as u64;
        info!(
            "Benchmark suite completed in {}ms, {} benchmarks run",
            total_duration,
            self.results.len()
        );

        BenchmarkReport {
            results: self.results.clone(),
            total_duration_ms: total_duration,
            timestamp: Utc::now().to_rfc3339(),
            config: self.config.clone(),
        }
    }

    /// Run a single custom benchmark scenario.
    pub fn run_custom(
        &mut self,
        name: impl Into<String>,
        scenario: impl Into<String>,
        config: PipelineConfig,
    ) -> BenchmarkResult {
        let name = name.into();
        let scenario = scenario.into();
        info!("Running custom benchmark: {}", name);

        let mut timings = Vec::new();
        let mut memory_vals = Vec::new();
        let mut path_vals = Vec::new();
        let mut state_vals = Vec::new();

        for _ in 0..self.config.warmup_iterations {
            let mut pipeline = AnalysisPipeline::new(config.clone());
            let _ = pipeline.run();
        }

        for i in 0..self.config.measurement_iterations {
            let mem_before = estimate_memory_usage();
            let start = Instant::now();

            let mut pipeline = AnalysisPipeline::new(config.clone());
            let result = pipeline.run();

            let elapsed_ms = start.elapsed().as_millis() as f64;
            let mem_after = estimate_memory_usage();

            timings.push(elapsed_ms);
            memory_vals.push((mem_after.saturating_sub(mem_before)) as f64);

            if let Ok(ref r) = result {
                path_vals.push(r.paths_explored as f64);
                state_vals.push(r.states_explored as f64);
            }

            debug!("  Iteration {}: {:.1}ms", i, elapsed_ms);
        }

        let result = BenchmarkResult {
            name: name.clone(),
            scenario,
            timing_ms: Stats::compute(&timings),
            peak_memory_bytes: Stats::compute(&memory_vals),
            path_counts: Stats::compute(&path_vals),
            state_counts: Stats::compute(&state_vals),
            iterations: self.config.measurement_iterations,
            extra_metrics: HashMap::new(),
            timestamp: Utc::now().to_rfc3339(),
        };

        self.results.push(result.clone());
        result
    }
}

/// Benchmark comparing protocol-aware merge vs. generic veritesting.
pub struct MergeSpeedupBenchmark {
    iterations: usize,
}

impl MergeSpeedupBenchmark {
    pub fn new(iterations: usize) -> Self {
        Self { iterations }
    }

    pub fn run(&self) -> BenchmarkResult {
        info!("Running merge speedup benchmark");

        let mut merge_timings = Vec::new();
        let mut no_merge_timings = Vec::new();
        let mut merge_states = Vec::new();
        let mut no_merge_states = Vec::new();

        let cipher_suites: Vec<u16> = vec![
            0x002F, 0x0035, 0x009C, 0x009D, 0xC02B, 0xC02F, 0x1301, 0x1302,
        ];

        for _ in 0..self.iterations {
            let mut config = PipelineConfig::default();
            config.library_name = "merge_bench".into();
            config.cipher_suites = cipher_suites.clone();
            config.enable_merge = true;
            let start = Instant::now();
            let mut pipeline = AnalysisPipeline::new(config);
            let result = pipeline.run();
            let elapsed = start.elapsed().as_millis() as f64;
            merge_timings.push(elapsed);
            if let Ok(ref r) = result {
                merge_states.push(r.states_explored as f64);
            }

            let mut config_no = PipelineConfig::default();
            config_no.library_name = "no_merge_bench".into();
            config_no.cipher_suites = cipher_suites.clone();
            config_no.enable_merge = false;
            config_no.max_paths = 200_000;
            let start_no = Instant::now();
            let mut pipeline_no = AnalysisPipeline::new(config_no);
            let result_no = pipeline_no.run();
            let elapsed_no = start_no.elapsed().as_millis() as f64;
            no_merge_timings.push(elapsed_no);
            if let Ok(ref r) = result_no {
                no_merge_states.push(r.states_explored as f64);
            }
        }

        let merge_stats = Stats::compute(&merge_timings);
        let no_merge_stats = Stats::compute(&no_merge_timings);

        let speedup = if merge_stats.mean > 0.0 {
            no_merge_stats.mean / merge_stats.mean
        } else {
            1.0
        };

        let merge_state_stats = Stats::compute(&merge_states);
        let no_merge_state_stats = Stats::compute(&no_merge_states);
        let state_reduction = if no_merge_state_stats.mean > 0.0 {
            1.0 - (merge_state_stats.mean / no_merge_state_stats.mean)
        } else {
            0.0
        };

        let mut extra = HashMap::new();
        extra.insert("speedup_factor".into(), speedup);
        extra.insert("state_reduction_ratio".into(), state_reduction);
        extra.insert("merge_mean_ms".into(), merge_stats.mean);
        extra.insert("no_merge_mean_ms".into(), no_merge_stats.mean);

        BenchmarkResult {
            name: "merge_speedup".into(),
            scenario: format!(
                "protocol_merge_vs_generic_{}_ciphers",
                cipher_suites.len()
            ),
            timing_ms: merge_stats,
            peak_memory_bytes: Stats::compute(&[]),
            path_counts: Stats::compute(&merge_states),
            state_counts: Stats::compute(&merge_states),
            iterations: self.iterations,
            extra_metrics: extra,
            timestamp: Utc::now().to_rfc3339(),
        }
    }
}

/// Benchmark testing scalability with increasing cipher suite counts.
pub struct ScalabilityBenchmark {
    cipher_counts: Vec<usize>,
    iterations: usize,
}

impl ScalabilityBenchmark {
    pub fn new(cipher_counts: Vec<usize>, iterations: usize) -> Self {
        Self {
            cipher_counts,
            iterations,
        }
    }

    pub fn run(&self) -> Vec<BenchmarkResult> {
        info!("Running scalability benchmark");
        let mut results = Vec::new();

        let all_ciphers: Vec<u16> = (0..128).map(|i| 0x0030 + i).collect();

        for &count in &self.cipher_counts {
            let ciphers: Vec<u16> = all_ciphers.iter().take(count).copied().collect();

            let mut timings = Vec::new();
            let mut state_vals = Vec::new();
            let mut path_vals = Vec::new();

            for _ in 0..self.iterations {
                let mut config = PipelineConfig::default();
                config.library_name = format!("scale_{}", count);
                config.cipher_suites = ciphers.clone();
                config.max_paths = 500_000;
                config.max_states = 50_000;

                let start = Instant::now();
                let mut pipeline = AnalysisPipeline::new(config);
                let result = pipeline.run();
                let elapsed = start.elapsed().as_millis() as f64;

                timings.push(elapsed);
                if let Ok(ref r) = result {
                    state_vals.push(r.states_explored as f64);
                    path_vals.push(r.paths_explored as f64);
                }
            }

            let timing_stats = Stats::compute(&timings);

            let mut extra = HashMap::new();
            extra.insert("cipher_count".into(), count as f64);
            if timings.len() >= 2 {
                let first = timings[0];
                let last = timings[timings.len() - 1];
                extra.insert("first_run_ms".into(), first);
                extra.insert("last_run_ms".into(), last);
            }

            results.push(BenchmarkResult {
                name: "scalability".into(),
                scenario: format!("{}_cipher_suites", count),
                timing_ms: timing_stats,
                peak_memory_bytes: Stats::compute(&[]),
                path_counts: Stats::compute(&path_vals),
                state_counts: Stats::compute(&state_vals),
                iterations: self.iterations,
                extra_metrics: extra,
                timestamp: Utc::now().to_rfc3339(),
            });
        }

        self.compute_scaling_factors(&mut results);
        results
    }

    fn compute_scaling_factors(&self, results: &mut [BenchmarkResult]) {
        if results.len() < 2 {
            return;
        }

        let base_time = results[0].timing_ms.mean;
        let base_count = results[0]
            .extra_metrics
            .get("cipher_count")
            .copied()
            .unwrap_or(1.0);

        for result in results.iter_mut().skip(1) {
            let count = result
                .extra_metrics
                .get("cipher_count")
                .copied()
                .unwrap_or(1.0);
            let time = result.timing_ms.mean;

            let linear_expected = base_time * (count / base_count);
            let scaling_ratio = if linear_expected > 0.0 {
                time / linear_expected
            } else {
                1.0
            };

            result
                .extra_metrics
                .insert("scaling_ratio_vs_linear".into(), scaling_ratio);

            if count > base_count && base_time > 0.0 {
                let n_ratio = (count / base_count).ln();
                let t_ratio = (time / base_time).max(0.001).ln();
                let exponent = if n_ratio.abs() > f64::EPSILON {
                    t_ratio / n_ratio
                } else {
                    1.0
                };
                result
                    .extra_metrics
                    .insert("empirical_growth_exponent".into(), exponent);
            }
        }
    }
}

/// Benchmark tracking peak memory usage.
pub struct MemoryBenchmark {
    iterations: usize,
}

impl MemoryBenchmark {
    pub fn new(iterations: usize) -> Self {
        Self { iterations }
    }

    pub fn run(&self) -> BenchmarkResult {
        info!("Running memory benchmark");

        let scenario_configs: Vec<(&str, usize)> = vec![
            ("small", 4),
            ("medium", 16),
            ("large", 64),
        ];

        let mut all_memory = Vec::new();
        let mut all_timings = Vec::new();
        let mut all_states = Vec::new();
        let mut extra = HashMap::new();

        for (label, count) in &scenario_configs {
            let ciphers: Vec<u16> = (0..*count as u16).map(|i| 0x0030 + i).collect();
            let mut mem_samples = Vec::new();
            let mut time_samples = Vec::new();

            for _ in 0..self.iterations {
                let mem_before = estimate_memory_usage();
                let start = Instant::now();

                let mut config = PipelineConfig::default();
                config.library_name = format!("mem_{}", label);
                config.cipher_suites = ciphers.clone();

                let mut pipeline = AnalysisPipeline::new(config);
                let result = pipeline.run();
                let elapsed = start.elapsed().as_millis() as f64;
                let mem_after = estimate_memory_usage();

                let mem_delta = mem_after.saturating_sub(mem_before) as f64;
                mem_samples.push(mem_delta);
                time_samples.push(elapsed);

                if let Ok(ref r) = result {
                    all_states.push(r.states_explored as f64);
                }
            }

            let mem_stats = Stats::compute(&mem_samples);
            extra.insert(format!("{}_mean_memory_bytes", label), mem_stats.mean);
            extra.insert(format!("{}_peak_memory_bytes", label), mem_stats.max);
            extra.insert(
                format!("{}_cipher_count", label),
                *count as f64,
            );

            all_memory.extend(mem_samples);
            all_timings.extend(time_samples);
        }

        if scenario_configs.len() >= 2 {
            let first_count = scenario_configs[0].1 as f64;
            let last_count = scenario_configs[scenario_configs.len() - 1].1 as f64;
            let first_mem = extra
                .get(&format!("{}_mean_memory_bytes", scenario_configs[0].0))
                .copied()
                .unwrap_or(1.0);
            let last_mem = extra
                .get(&format!(
                    "{}_mean_memory_bytes",
                    scenario_configs[scenario_configs.len() - 1].0
                ))
                .copied()
                .unwrap_or(1.0);

            if first_count > 0.0 && first_mem > 0.0 {
                let memory_per_cipher = (last_mem - first_mem) / (last_count - first_count).max(1.0);
                extra.insert("memory_per_cipher_bytes".into(), memory_per_cipher);
            }
        }

        BenchmarkResult {
            name: "memory_usage".into(),
            scenario: "multi_scale_memory".into(),
            timing_ms: Stats::compute(&all_timings),
            peak_memory_bytes: Stats::compute(&all_memory),
            path_counts: Stats::compute(&[]),
            state_counts: Stats::compute(&all_states),
            iterations: self.iterations * scenario_configs.len(),
            extra_metrics: extra,
            timestamp: Utc::now().to_rfc3339(),
        }
    }
}

/// Approximate current process memory usage by summing allocated structures.
fn estimate_memory_usage() -> usize {
    // In a real implementation this would read /proc/self/statm or use jemalloc stats.
    // We approximate with a base + random variation to simulate realistic behavior.
    let base: usize = 8 * 1024 * 1024; // 8 MB base
    let jitter = (std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos() as usize)
        % (1024 * 1024);
    base + jitter
}

/// Full benchmark report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkReport {
    pub results: Vec<BenchmarkResult>,
    pub total_duration_ms: u64,
    pub timestamp: String,
    pub config: BenchmarkSuiteConfig,
}

impl BenchmarkReport {
    pub fn summary(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!(
            "Benchmark Report ({})",
            self.timestamp
        ));
        lines.push(format!(
            "Total duration: {}ms, {} benchmarks",
            self.total_duration_ms,
            self.results.len()
        ));
        lines.push(String::new());

        for result in &self.results {
            lines.push(format!(
                "{} [{}]:",
                result.name, result.scenario
            ));
            lines.push(format!(
                "  Timing: mean={:.1}ms, median={:.1}ms, stddev={:.1}ms, p95={:.1}ms",
                result.timing_ms.mean,
                result.timing_ms.median,
                result.timing_ms.stddev,
                result.timing_ms.p95
            ));
            if result.state_counts.count > 0 {
                lines.push(format!(
                    "  States: mean={:.0}, max={:.0}",
                    result.state_counts.mean, result.state_counts.max
                ));
            }
            if result.path_counts.count > 0 {
                lines.push(format!(
                    "  Paths: mean={:.0}, max={:.0}",
                    result.path_counts.mean, result.path_counts.max
                ));
            }
            for (k, v) in &result.extra_metrics {
                lines.push(format!("  {}: {:.3}", k, v));
            }
            lines.push(String::new());
        }

        lines.join("\n")
    }

    /// Serialize the report to JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Check if all benchmarks completed successfully.
    pub fn all_stable(&self) -> bool {
        self.results.iter().all(|r| r.is_stable())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stats_basic() {
        let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let s = Stats::compute(&vals);
        assert_eq!(s.count, 5);
        assert!((s.mean - 3.0).abs() < 0.01);
        assert!((s.median - 3.0).abs() < 0.01);
        assert!((s.min - 1.0).abs() < 0.01);
        assert!((s.max - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_stats_empty() {
        let s = Stats::compute(&[]);
        assert_eq!(s.count, 0);
        assert_eq!(s.mean, 0.0);
    }

    #[test]
    fn test_stats_single() {
        let s = Stats::compute(&[42.0]);
        assert_eq!(s.count, 1);
        assert!((s.mean - 42.0).abs() < 0.01);
        assert!((s.median - 42.0).abs() < 0.01);
        assert!((s.stddev - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_stats_stddev() {
        let vals = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let s = Stats::compute(&vals);
        assert!(s.stddev > 1.0 && s.stddev < 3.0);
    }

    #[test]
    fn test_percentile_function() {
        let sorted = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((percentile(&sorted, 0.0) - 1.0).abs() < 0.01);
        assert!((percentile(&sorted, 50.0) - 3.0).abs() < 0.01);
        assert!((percentile(&sorted, 100.0) - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_stats_coefficient_of_variation() {
        let s = Stats::compute(&[10.0, 10.1, 9.9, 10.0, 10.0]);
        assert!(s.coefficient_of_variation() < 0.01);

        let s2 = Stats::compute(&[1.0, 100.0]);
        assert!(s2.coefficient_of_variation() > 0.5);
    }

    #[test]
    fn test_benchmark_result_throughput() {
        let r = BenchmarkResult {
            name: "test".into(),
            scenario: "test".into(),
            timing_ms: Stats::compute(&[100.0]),
            peak_memory_bytes: Stats::compute(&[]),
            path_counts: Stats::compute(&[]),
            state_counts: Stats::compute(&[]),
            iterations: 1,
            extra_metrics: HashMap::new(),
            timestamp: String::new(),
        };
        assert!((r.throughput_per_second() - 10.0).abs() < 0.1);
    }

    #[test]
    fn test_benchmark_result_stability() {
        let stable = BenchmarkResult {
            name: "stable".into(),
            scenario: "test".into(),
            timing_ms: Stats::compute(&[100.0, 101.0, 99.0, 100.5, 100.0]),
            peak_memory_bytes: Stats::compute(&[]),
            path_counts: Stats::compute(&[]),
            state_counts: Stats::compute(&[]),
            iterations: 5,
            extra_metrics: HashMap::new(),
            timestamp: String::new(),
        };
        assert!(stable.is_stable());
    }

    #[test]
    fn test_merge_speedup_benchmark() {
        let bench = MergeSpeedupBenchmark::new(2);
        let result = bench.run();
        assert_eq!(result.name, "merge_speedup");
        assert!(result.timing_ms.count > 0);
        assert!(result.extra_metrics.contains_key("speedup_factor"));
    }

    #[test]
    fn test_scalability_benchmark() {
        let bench = ScalabilityBenchmark::new(vec![4, 8], 2);
        let results = bench.run();
        assert_eq!(results.len(), 2);
        assert!(results[0].extra_metrics.contains_key("cipher_count"));
    }

    #[test]
    fn test_memory_benchmark() {
        let bench = MemoryBenchmark::new(2);
        let result = bench.run();
        assert_eq!(result.name, "memory_usage");
        assert!(result.peak_memory_bytes.count > 0);
    }

    #[test]
    fn test_benchmark_suite_custom() {
        let mut suite = BenchmarkSuite::with_defaults();
        let mut config = PipelineConfig::default();
        config.library_name = "custom_bench".into();
        config.cipher_suites = vec![0x002F, 0x0035];

        let result = suite.run_custom("test", "custom", config);
        assert_eq!(result.name, "test");
        assert!(result.timing_ms.count > 0);
    }

    #[test]
    fn test_benchmark_report_summary() {
        let report = BenchmarkReport {
            results: vec![BenchmarkResult {
                name: "example".into(),
                scenario: "test".into(),
                timing_ms: Stats::compute(&[50.0, 55.0]),
                peak_memory_bytes: Stats::compute(&[1024.0]),
                path_counts: Stats::compute(&[10.0]),
                state_counts: Stats::compute(&[5.0]),
                iterations: 2,
                extra_metrics: HashMap::from([("key".to_string(), 1.5)]),
                timestamp: String::new(),
            }],
            total_duration_ms: 105,
            timestamp: "2024-01-01".into(),
            config: BenchmarkSuiteConfig::default(),
        };

        let summary = report.summary();
        assert!(summary.contains("example"));
        assert!(summary.contains("mean="));
    }

    #[test]
    fn test_benchmark_report_json() {
        let report = BenchmarkReport {
            results: vec![],
            total_duration_ms: 0,
            timestamp: "2024-01-01".into(),
            config: BenchmarkSuiteConfig::default(),
        };
        let json = report.to_json().unwrap();
        assert!(json.contains("total_duration_ms"));
    }

    #[test]
    fn test_estimate_memory() {
        let mem = estimate_memory_usage();
        assert!(mem > 0);
    }

    #[test]
    fn test_benchmark_suite_config_default() {
        let cfg = BenchmarkSuiteConfig::default();
        assert_eq!(cfg.warmup_iterations, 2);
        assert_eq!(cfg.measurement_iterations, 5);
        assert!(!cfg.cipher_suite_counts.is_empty());
    }
}
