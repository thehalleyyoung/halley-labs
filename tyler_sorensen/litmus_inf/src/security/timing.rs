//! Timing side-channel analysis for GPU kernels.
//!
//! Detects timing variations that depend on secret data, enabling
//! covert channel bandwidth estimation and vulnerability assessment.

use std::collections::HashMap;
use std::fmt;
use serde::{Serialize, Deserialize};

// ---------------------------------------------------------------------------
// ExecutionTimeModel
// ---------------------------------------------------------------------------

/// Models GPU kernel execution timing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionTimeModel {
    pub name: String,
    pub base_latency_ns: f64,
    pub per_thread_ns: f64,
    pub memory_latency_ns: f64,
    pub branch_penalty_ns: f64,
    pub cache_miss_penalty_ns: f64,
    pub instruction_count: usize,
    pub memory_accesses: usize,
    pub branches: usize,
}

impl ExecutionTimeModel {
    pub fn new(name: &str) -> Self {
        ExecutionTimeModel {
            name: name.to_string(),
            base_latency_ns: 1000.0,
            per_thread_ns: 10.0,
            memory_latency_ns: 100.0,
            branch_penalty_ns: 50.0,
            cache_miss_penalty_ns: 200.0,
            instruction_count: 0,
            memory_accesses: 0,
            branches: 0,
        }
    }

    /// Estimate execution time for given parameters.
    pub fn estimate(&self, secret_dependent_branches: usize, cache_misses: usize) -> f64 {
        self.base_latency_ns
            + self.instruction_count as f64 * self.per_thread_ns
            + self.memory_accesses as f64 * self.memory_latency_ns
            + secret_dependent_branches as f64 * self.branch_penalty_ns
            + cache_misses as f64 * self.cache_miss_penalty_ns
    }

    /// Maximum timing variation due to secret.
    pub fn max_variation(&self, max_branches: usize, max_cache_misses: usize) -> f64 {
        max_branches as f64 * self.branch_penalty_ns
            + max_cache_misses as f64 * self.cache_miss_penalty_ns
    }
}

// ---------------------------------------------------------------------------
// TimingVariation
// ---------------------------------------------------------------------------

/// Measurements of timing variation across different secret values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingVariation {
    pub secret_values: Vec<u64>,
    pub measurements: Vec<Vec<f64>>,
    pub means: Vec<f64>,
    pub variances: Vec<f64>,
    pub min_times: Vec<f64>,
    pub max_times: Vec<f64>,
}

impl TimingVariation {
    /// Create from raw measurements.
    /// `measurements[i]` is a vector of timing samples for secret value `i`.
    pub fn from_measurements(secret_values: Vec<u64>, measurements: Vec<Vec<f64>>) -> Self {
        assert_eq!(secret_values.len(), measurements.len());
        let means: Vec<f64> = measurements.iter()
            .map(|m| if m.is_empty() { 0.0 } else { m.iter().sum::<f64>() / m.len() as f64 })
            .collect();
        let variances: Vec<f64> = measurements.iter().zip(means.iter())
            .map(|(m, &mean)| {
                if m.len() < 2 { return 0.0; }
                m.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (m.len() - 1) as f64
            })
            .collect();
        let min_times: Vec<f64> = measurements.iter()
            .map(|m| m.iter().cloned().fold(f64::INFINITY, f64::min))
            .collect();
        let max_times: Vec<f64> = measurements.iter()
            .map(|m| m.iter().cloned().fold(f64::NEG_INFINITY, f64::max))
            .collect();

        TimingVariation { secret_values, measurements, means, variances, min_times, max_times }
    }

    /// Global mean across all secrets.
    pub fn global_mean(&self) -> f64 {
        if self.means.is_empty() { return 0.0; }
        self.means.iter().sum::<f64>() / self.means.len() as f64
    }

    /// Maximum difference between means (indicates timing channel).
    pub fn max_mean_difference(&self) -> f64 {
        if self.means.len() < 2 { return 0.0; }
        let min = self.means.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = self.means.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        max - min
    }

    /// Coefficient of variation for timing across secrets.
    pub fn cross_secret_cv(&self) -> f64 {
        let mean = self.global_mean();
        if mean <= 0.0 { return 0.0; }
        let variance = self.means.iter()
            .map(|&m| (m - mean).powi(2))
            .sum::<f64>() / self.means.len() as f64;
        variance.sqrt() / mean
    }

    /// Whether timing variation is statistically significant.
    pub fn is_significant(&self, threshold: f64) -> bool {
        self.max_mean_difference() > threshold
    }

    /// Number of distinct timing classes.
    pub fn num_timing_classes(&self, epsilon: f64) -> usize {
        let mut classes: Vec<f64> = Vec::new();
        for &mean in &self.means {
            let found = classes.iter().any(|&c| (c - mean).abs() < epsilon);
            if !found { classes.push(mean); }
        }
        classes.len()
    }
}

// ---------------------------------------------------------------------------
// DifferentialTiming
// ---------------------------------------------------------------------------

/// Differential timing analysis: compare timing distributions for
/// different secret values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DifferentialTiming {
    pub comparisons: Vec<TimingComparison>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingComparison {
    pub secret_a: u64,
    pub secret_b: u64,
    pub mean_diff: f64,
    pub t_statistic: f64,
    pub distinguishable: bool,
}

impl DifferentialTiming {
    /// Perform pairwise differential timing analysis.
    pub fn analyze(variation: &TimingVariation) -> Self {
        let mut comparisons = Vec::new();
        let n = variation.secret_values.len();
        for i in 0..n {
            for j in (i + 1)..n {
                let mean_diff = (variation.means[i] - variation.means[j]).abs();

                // Welch's t-test (simplified)
                let n_a = variation.measurements[i].len() as f64;
                let n_b = variation.measurements[j].len() as f64;
                let var_a = variation.variances[i];
                let var_b = variation.variances[j];

                let se = if n_a > 0.0 && n_b > 0.0 {
                    (var_a / n_a + var_b / n_b).sqrt()
                } else {
                    f64::INFINITY
                };

                let t_stat = if se > 0.0 { mean_diff / se } else { 0.0 };

                // Significant at ~95% (t > 1.96 for large samples)
                let distinguishable = t_stat > 1.96;

                comparisons.push(TimingComparison {
                    secret_a: variation.secret_values[i],
                    secret_b: variation.secret_values[j],
                    mean_diff,
                    t_statistic: t_stat,
                    distinguishable,
                });
            }
        }

        DifferentialTiming { comparisons }
    }

    /// Number of distinguishable pairs.
    pub fn distinguishable_pairs(&self) -> usize {
        self.comparisons.iter().filter(|c| c.distinguishable).count()
    }

    /// Total number of pairs.
    pub fn total_pairs(&self) -> usize {
        self.comparisons.len()
    }

    /// Maximum t-statistic (strength of the strongest difference).
    pub fn max_t_statistic(&self) -> f64 {
        self.comparisons.iter()
            .map(|c| c.t_statistic)
            .fold(0.0f64, f64::max)
    }

    /// Fraction of distinguishable pairs.
    pub fn distinguishability_ratio(&self) -> f64 {
        if self.comparisons.is_empty() { return 0.0; }
        self.distinguishable_pairs() as f64 / self.comparisons.len() as f64
    }
}

// ---------------------------------------------------------------------------
// CovertChannelBandwidth
// ---------------------------------------------------------------------------

/// Estimates the bandwidth of a covert timing channel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CovertChannelBandwidth {
    /// Estimated bandwidth in bits per second.
    pub bits_per_second: f64,
    /// Estimated bandwidth in bits per kernel invocation.
    pub bits_per_invocation: f64,
    /// Error rate of the channel.
    pub error_rate: f64,
    /// Effective bandwidth (accounting for error correction).
    pub effective_bps: f64,
}

impl CovertChannelBandwidth {
    /// Estimate bandwidth from timing variation data.
    pub fn estimate(
        variation: &TimingVariation,
        kernel_invocation_time_ns: f64,
    ) -> Self {
        let num_classes = variation.num_timing_classes(1.0);
        let bits_per_invocation = if num_classes > 1 {
            (num_classes as f64).log2()
        } else {
            0.0
        };

        let cv = variation.cross_secret_cv();
        // Higher CV means more noise, higher error rate
        let error_rate = (1.0 - cv).max(0.0).min(1.0);
        let error_rate = 1.0 - error_rate;

        let invocations_per_second = if kernel_invocation_time_ns > 0.0 {
            1e9 / kernel_invocation_time_ns
        } else {
            0.0
        };

        let bits_per_second = bits_per_invocation * invocations_per_second;

        // Shannon capacity: C = 1 - H(error)
        let h_err = if error_rate > 0.0 && error_rate < 1.0 {
            -error_rate * error_rate.log2() - (1.0 - error_rate) * (1.0 - error_rate).log2()
        } else {
            0.0
        };
        let effective_bps = bits_per_second * (1.0 - h_err).max(0.0);

        CovertChannelBandwidth {
            bits_per_second,
            bits_per_invocation,
            error_rate,
            effective_bps,
        }
    }

    /// Whether this constitutes a viable covert channel.
    pub fn is_viable(&self) -> bool {
        self.effective_bps > 1.0 // at least 1 bit/s
    }
}

impl fmt::Display for CovertChannelBandwidth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CovertChannel: {:.1} bps effective ({:.2} bits/invocation, {:.1}% error)",
            self.effective_bps, self.bits_per_invocation, self.error_rate * 100.0)
    }
}

// ---------------------------------------------------------------------------
// TimingChannelDetector
// ---------------------------------------------------------------------------

/// Top-level detector for timing side channels.
#[derive(Debug, Clone)]
pub struct TimingChannelDetector {
    pub models: Vec<ExecutionTimeModel>,
    pub variations: Vec<TimingVariation>,
}

impl TimingChannelDetector {
    pub fn new() -> Self {
        TimingChannelDetector { models: Vec::new(), variations: Vec::new() }
    }

    /// Add a kernel timing model.
    pub fn add_model(&mut self, model: ExecutionTimeModel) {
        self.models.push(model);
    }

    /// Add timing measurements.
    pub fn add_variation(&mut self, variation: TimingVariation) {
        self.variations.push(variation);
    }

    /// Simulate timing for a model with different secret inputs.
    pub fn simulate(
        &self,
        model: &ExecutionTimeModel,
        secret_configs: &[(usize, usize)], // (branches, cache_misses) per secret
    ) -> TimingVariation {
        let secrets: Vec<u64> = (0..secret_configs.len() as u64).collect();
        let measurements: Vec<Vec<f64>> = secret_configs.iter()
            .map(|&(branches, misses)| {
                let base = model.estimate(branches, misses);
                // Add some noise
                (0..100).map(|i| {
                    base + (i as f64 * 0.1).sin() * 5.0
                }).collect()
            })
            .collect();
        TimingVariation::from_measurements(secrets, measurements)
    }

    /// Generate a comprehensive report.
    pub fn generate_report(&self) -> TimingReport {
        let mut findings = Vec::new();

        for (i, variation) in self.variations.iter().enumerate() {
            let diff = DifferentialTiming::analyze(variation);
            let bw = CovertChannelBandwidth::estimate(variation, variation.global_mean());

            findings.push(TimingFinding {
                name: format!("variation_{}", i),
                max_mean_difference: variation.max_mean_difference(),
                distinguishable_pairs: diff.distinguishable_pairs(),
                total_pairs: diff.total_pairs(),
                bandwidth: bw,
                is_vulnerable: diff.distinguishable_pairs() > 0,
            });
        }

        let max_bw = findings.iter()
            .map(|f| f.bandwidth.effective_bps)
            .fold(0.0f64, f64::max);
        let any_vulnerable = findings.iter().any(|f| f.is_vulnerable);

        TimingReport {
            findings,
            max_bandwidth_bps: max_bw,
            any_vulnerable,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingFinding {
    pub name: String,
    pub max_mean_difference: f64,
    pub distinguishable_pairs: usize,
    pub total_pairs: usize,
    pub bandwidth: CovertChannelBandwidth,
    pub is_vulnerable: bool,
}

// ---------------------------------------------------------------------------
// TimingReport
// ---------------------------------------------------------------------------

/// Report of timing channel analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingReport {
    pub findings: Vec<TimingFinding>,
    pub max_bandwidth_bps: f64,
    pub any_vulnerable: bool,
}

impl TimingReport {
    pub fn severity(&self) -> &'static str {
        if self.max_bandwidth_bps > 1000.0 { "CRITICAL" }
        else if self.max_bandwidth_bps > 100.0 { "HIGH" }
        else if self.max_bandwidth_bps > 10.0 { "MEDIUM" }
        else if self.any_vulnerable { "LOW" }
        else { "NONE" }
    }

    pub fn to_text(&self) -> String {
        let mut s = String::new();
        s.push_str("=== Timing Channel Analysis ===\n");
        s.push_str(&format!("Findings: {}\n", self.findings.len()));
        s.push_str(&format!("Max bandwidth: {:.1} bps\n", self.max_bandwidth_bps));
        s.push_str(&format!("Severity: {}\n\n", self.severity()));
        for f in &self.findings {
            s.push_str(&format!("  [{}] diff={:.1}ns, {}/{} distinguishable, bw={:.1} bps\n",
                f.name, f.max_mean_difference, f.distinguishable_pairs,
                f.total_pairs, f.bandwidth.effective_bps));
        }
        s
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }
}

impl fmt::Display for TimingReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TimingReport({} findings, max_bw={:.1} bps, {})",
            self.findings.len(), self.max_bandwidth_bps, self.severity())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execution_time_model() {
        let model = ExecutionTimeModel::new("test_kernel");
        let t1 = model.estimate(0, 0);
        let t2 = model.estimate(5, 10);
        assert!(t2 > t1);
    }

    #[test]
    fn test_timing_variation() {
        let secrets = vec![0, 1];
        let measurements = vec![
            vec![100.0, 101.0, 99.0, 100.5],
            vec![200.0, 201.0, 199.0, 200.5],
        ];
        let var = TimingVariation::from_measurements(secrets, measurements);
        assert!(var.max_mean_difference() > 90.0);
        assert!(var.is_significant(10.0));
    }

    #[test]
    fn test_timing_variation_no_difference() {
        let secrets = vec![0, 1];
        let measurements = vec![
            vec![100.0, 100.0, 100.0],
            vec![100.0, 100.0, 100.0],
        ];
        let var = TimingVariation::from_measurements(secrets, measurements);
        assert!(var.max_mean_difference() < 1.0);
        assert!(!var.is_significant(1.0));
    }

    #[test]
    fn test_differential_timing() {
        let secrets = vec![0, 1, 2];
        let measurements = vec![
            vec![100.0; 100],
            vec![200.0; 100],
            vec![100.0; 100],
        ];
        let var = TimingVariation::from_measurements(secrets, measurements);
        let diff = DifferentialTiming::analyze(&var);
        assert!(diff.distinguishable_pairs() > 0);
        assert_eq!(diff.total_pairs(), 3);
    }

    #[test]
    fn test_covert_channel_bandwidth() {
        let secrets = vec![0, 1];
        let measurements = vec![
            vec![100.0; 50],
            vec![200.0; 50],
        ];
        let var = TimingVariation::from_measurements(secrets, measurements);
        let bw = CovertChannelBandwidth::estimate(&var, 1000.0);
        assert!(bw.bits_per_invocation > 0.0);
        assert!(bw.bits_per_second > 0.0);
    }

    #[test]
    fn test_covert_channel_not_viable() {
        let secrets = vec![0, 1];
        let measurements = vec![
            vec![100.0; 50],
            vec![100.0; 50],
        ];
        let var = TimingVariation::from_measurements(secrets, measurements);
        let bw = CovertChannelBandwidth::estimate(&var, 1000.0);
        assert_eq!(bw.bits_per_invocation, 0.0);
    }

    #[test]
    fn test_timing_classes() {
        let secrets = vec![0, 1, 2, 3];
        let measurements = vec![
            vec![100.0; 10],
            vec![100.0; 10],
            vec![200.0; 10],
            vec![200.0; 10],
        ];
        let var = TimingVariation::from_measurements(secrets, measurements);
        assert_eq!(var.num_timing_classes(1.0), 2);
    }

    #[test]
    fn test_timing_detector() {
        let mut detector = TimingChannelDetector::new();
        let secrets = vec![0, 1];
        let measurements = vec![
            vec![100.0; 50],
            vec![200.0; 50],
        ];
        detector.add_variation(TimingVariation::from_measurements(secrets, measurements));
        let report = detector.generate_report();
        assert!(report.any_vulnerable);
    }

    #[test]
    fn test_timing_report_severity() {
        let report = TimingReport {
            findings: vec![],
            max_bandwidth_bps: 0.0,
            any_vulnerable: false,
        };
        assert_eq!(report.severity(), "NONE");
    }

    #[test]
    fn test_simulate_timing() {
        let detector = TimingChannelDetector::new();
        let model = ExecutionTimeModel::new("test");
        let configs = vec![(0, 0), (5, 10)];
        let var = detector.simulate(&model, &configs);
        assert!(var.max_mean_difference() > 0.0);
    }

    #[test]
    fn test_timing_report_text() {
        let report = TimingReport {
            findings: vec![TimingFinding {
                name: "test".to_string(),
                max_mean_difference: 100.0,
                distinguishable_pairs: 1,
                total_pairs: 1,
                bandwidth: CovertChannelBandwidth {
                    bits_per_second: 100.0,
                    bits_per_invocation: 1.0,
                    error_rate: 0.1,
                    effective_bps: 50.0,
                },
                is_vulnerable: true,
            }],
            max_bandwidth_bps: 50.0,
            any_vulnerable: true,
        };
        let text = report.to_text();
        assert!(text.contains("Timing Channel"));
        assert!(text.contains("test"));
    }

    #[test]
    fn test_cross_secret_cv() {
        let secrets = vec![0, 1];
        let measurements = vec![
            vec![100.0; 10],
            vec![200.0; 10],
        ];
        let var = TimingVariation::from_measurements(secrets, measurements);
        assert!(var.cross_secret_cv() > 0.0);
    }

    #[test]
    fn test_max_variation() {
        let model = ExecutionTimeModel::new("test");
        let variation = model.max_variation(5, 10);
        assert!(variation > 0.0);
    }
}
