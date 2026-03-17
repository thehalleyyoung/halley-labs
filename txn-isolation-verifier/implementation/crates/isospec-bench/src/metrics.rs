//! Benchmark metrics collection and analysis.
//!
//! Provides types for recording per-run analysis metrics (time, memory,
//! SMT solver calls, anomaly counts, constraint sizes) and for comparing
//! metric sets across engine pairs.

use std::collections::HashMap;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Core metrics
// ---------------------------------------------------------------------------

/// Metrics captured during a single analysis run.
#[derive(Debug, Clone)]
pub struct AnalysisMetrics {
    /// Wall-clock time of the analysis.
    pub analysis_time: Duration,
    /// Peak memory usage in bytes (estimated).
    pub peak_memory_bytes: u64,
    /// Number of SMT solver invocations.
    pub smt_calls: u64,
    /// Total time spent inside the SMT solver.
    pub smt_time: Duration,
    /// Number of anomalies detected.
    pub anomalies_found: u64,
    /// Total number of constraints generated.
    pub constraint_count: u64,
    /// Number of variables in the SMT encoding.
    pub variable_count: u64,
    /// Number of dependency edges in the DSG.
    pub dependency_count: u64,
    /// Number of transactions in the workload.
    pub transaction_count: u64,
    /// Number of operations across all transactions.
    pub operation_count: u64,
    /// Additional user-defined metrics.
    pub extra: HashMap<String, f64>,
}

impl AnalysisMetrics {
    pub fn new() -> Self {
        Self {
            analysis_time: Duration::ZERO,
            peak_memory_bytes: 0,
            smt_calls: 0,
            smt_time: Duration::ZERO,
            anomalies_found: 0,
            constraint_count: 0,
            variable_count: 0,
            dependency_count: 0,
            transaction_count: 0,
            operation_count: 0,
            extra: HashMap::new(),
        }
    }

    pub fn with_time(mut self, d: Duration) -> Self {
        self.analysis_time = d;
        self
    }

    pub fn with_memory(mut self, bytes: u64) -> Self {
        self.peak_memory_bytes = bytes;
        self
    }

    pub fn with_smt(mut self, calls: u64, time: Duration) -> Self {
        self.smt_calls = calls;
        self.smt_time = time;
        self
    }

    pub fn with_anomalies(mut self, n: u64) -> Self {
        self.anomalies_found = n;
        self
    }

    pub fn with_constraints(mut self, constraints: u64, variables: u64) -> Self {
        self.constraint_count = constraints;
        self.variable_count = variables;
        self
    }

    pub fn with_workload_size(mut self, txns: u64, ops: u64) -> Self {
        self.transaction_count = txns;
        self.operation_count = ops;
        self
    }

    pub fn with_dependencies(mut self, n: u64) -> Self {
        self.dependency_count = n;
        self
    }

    pub fn set_extra(&mut self, key: impl Into<String>, value: f64) {
        self.extra.insert(key.into(), value);
    }

    /// SMT time as a fraction of total analysis time.
    pub fn smt_time_fraction(&self) -> f64 {
        let total = self.analysis_time.as_secs_f64();
        if total < 1e-15 {
            return 0.0;
        }
        self.smt_time.as_secs_f64() / total
    }

    /// Average time per SMT call.
    pub fn avg_smt_call_time(&self) -> Duration {
        if self.smt_calls == 0 {
            return Duration::ZERO;
        }
        self.smt_time / self.smt_calls as u32
    }

    /// Constraints per transaction.
    pub fn constraints_per_txn(&self) -> f64 {
        if self.transaction_count == 0 {
            return 0.0;
        }
        self.constraint_count as f64 / self.transaction_count as f64
    }

    /// Constraints per operation.
    pub fn constraints_per_op(&self) -> f64 {
        if self.operation_count == 0 {
            return 0.0;
        }
        self.constraint_count as f64 / self.operation_count as f64
    }
}

impl Default for AnalysisMetrics {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Metrics collector
// ---------------------------------------------------------------------------

/// Collects multiple metric snapshots and computes aggregates.
pub struct MetricsCollector {
    snapshots: Vec<AnalysisMetrics>,
    labels: Vec<String>,
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self { snapshots: Vec::new(), labels: Vec::new() }
    }

    pub fn record(&mut self, label: impl Into<String>, metrics: AnalysisMetrics) {
        self.labels.push(label.into());
        self.snapshots.push(metrics);
    }

    pub fn snapshot_count(&self) -> usize {
        self.snapshots.len()
    }

    pub fn get(&self, index: usize) -> Option<&AnalysisMetrics> {
        self.snapshots.get(index)
    }

    pub fn find_by_label(&self, label: &str) -> Option<&AnalysisMetrics> {
        self.labels.iter()
            .position(|l| l == label)
            .and_then(|idx| self.snapshots.get(idx))
    }

    /// Aggregate timing statistics across all snapshots.
    pub fn timing_summary(&self) -> TimingSummary {
        let times: Vec<f64> = self.snapshots.iter()
            .map(|m| m.analysis_time.as_secs_f64())
            .collect();

        if times.is_empty() {
            return TimingSummary::zero();
        }

        let n = times.len();
        let sum: f64 = times.iter().sum();
        let mean = sum / n as f64;
        let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let variance = if n > 1 {
            times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / (n - 1) as f64
        } else {
            0.0
        };

        TimingSummary {
            count: n,
            mean_seconds: mean,
            min_seconds: min,
            max_seconds: max,
            std_dev_seconds: variance.sqrt(),
            total_seconds: sum,
        }
    }

    /// Total anomalies found across all snapshots.
    pub fn total_anomalies(&self) -> u64 {
        self.snapshots.iter().map(|m| m.anomalies_found).sum()
    }

    /// Total SMT calls across all snapshots.
    pub fn total_smt_calls(&self) -> u64 {
        self.snapshots.iter().map(|m| m.smt_calls).sum()
    }

    /// Average constraint count per snapshot.
    pub fn avg_constraint_count(&self) -> f64 {
        if self.snapshots.is_empty() {
            return 0.0;
        }
        let total: u64 = self.snapshots.iter().map(|m| m.constraint_count).sum();
        total as f64 / self.snapshots.len() as f64
    }

    /// Average peak memory.
    pub fn avg_peak_memory(&self) -> f64 {
        if self.snapshots.is_empty() {
            return 0.0;
        }
        let total: u64 = self.snapshots.iter().map(|m| m.peak_memory_bytes).sum();
        total as f64 / self.snapshots.len() as f64
    }

    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    pub fn snapshots(&self) -> &[AnalysisMetrics] {
        &self.snapshots
    }

    pub fn clear(&mut self) {
        self.snapshots.clear();
        self.labels.clear();
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Timing summary
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct TimingSummary {
    pub count: usize,
    pub mean_seconds: f64,
    pub min_seconds: f64,
    pub max_seconds: f64,
    pub std_dev_seconds: f64,
    pub total_seconds: f64,
}

impl TimingSummary {
    pub fn zero() -> Self {
        Self {
            count: 0,
            mean_seconds: 0.0,
            min_seconds: 0.0,
            max_seconds: 0.0,
            std_dev_seconds: 0.0,
            total_seconds: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Comparison metrics
// ---------------------------------------------------------------------------

/// Comparison of analysis metrics between two engine configurations.
#[derive(Debug, Clone)]
pub struct ComparisonMetrics {
    pub engine_a: String,
    pub engine_b: String,
    pub time_ratio: f64,
    pub memory_ratio: f64,
    pub smt_call_ratio: f64,
    pub constraint_ratio: f64,
    pub anomaly_diff: i64,
    pub common_anomalies: u64,
    pub unique_to_a: u64,
    pub unique_to_b: u64,
}

impl ComparisonMetrics {
    /// Compare two metric snapshots.
    pub fn compare(
        engine_a: &str,
        metrics_a: &AnalysisMetrics,
        engine_b: &str,
        metrics_b: &AnalysisMetrics,
    ) -> Self {
        let safe_ratio = |a: f64, b: f64| -> f64 {
            if b.abs() < 1e-15 { 0.0 } else { a / b }
        };

        let time_a = metrics_a.analysis_time.as_secs_f64();
        let time_b = metrics_b.analysis_time.as_secs_f64();

        // Anomaly overlap estimation: min of the two is an upper bound on common
        let min_anomalies = metrics_a.anomalies_found.min(metrics_b.anomalies_found);
        let max_anomalies = metrics_a.anomalies_found.max(metrics_b.anomalies_found);

        Self {
            engine_a: engine_a.to_string(),
            engine_b: engine_b.to_string(),
            time_ratio: safe_ratio(time_a, time_b),
            memory_ratio: safe_ratio(
                metrics_a.peak_memory_bytes as f64,
                metrics_b.peak_memory_bytes as f64,
            ),
            smt_call_ratio: safe_ratio(
                metrics_a.smt_calls as f64,
                metrics_b.smt_calls as f64,
            ),
            constraint_ratio: safe_ratio(
                metrics_a.constraint_count as f64,
                metrics_b.constraint_count as f64,
            ),
            anomaly_diff: metrics_a.anomalies_found as i64 - metrics_b.anomalies_found as i64,
            common_anomalies: min_anomalies,
            unique_to_a: metrics_a.anomalies_found.saturating_sub(min_anomalies),
            unique_to_b: metrics_b.anomalies_found.saturating_sub(min_anomalies),
        }
    }

    /// True if engine_a is faster than engine_b.
    pub fn a_is_faster(&self) -> bool {
        self.time_ratio < 1.0
    }

    /// True if engine_a uses less memory.
    pub fn a_uses_less_memory(&self) -> bool {
        self.memory_ratio < 1.0
    }

    /// Speedup of B over A (>1 means B is faster).
    pub fn speedup_b_over_a(&self) -> f64 {
        if self.time_ratio.abs() < 1e-15 { 0.0 } else { 1.0 / self.time_ratio }
    }
}

// ---------------------------------------------------------------------------
// Metric aggregation helpers
// ---------------------------------------------------------------------------

/// Aggregate a slice of metric snapshots into a single summary snapshot.
pub fn aggregate_metrics(snapshots: &[AnalysisMetrics]) -> AnalysisMetrics {
    if snapshots.is_empty() {
        return AnalysisMetrics::new();
    }

    let n = snapshots.len() as u64;

    let total_time: Duration = snapshots.iter().map(|m| m.analysis_time).sum();
    let total_smt_time: Duration = snapshots.iter().map(|m| m.smt_time).sum();
    let total_memory: u64 = snapshots.iter().map(|m| m.peak_memory_bytes).sum();
    let total_smt_calls: u64 = snapshots.iter().map(|m| m.smt_calls).sum();
    let total_anomalies: u64 = snapshots.iter().map(|m| m.anomalies_found).sum();
    let total_constraints: u64 = snapshots.iter().map(|m| m.constraint_count).sum();
    let total_variables: u64 = snapshots.iter().map(|m| m.variable_count).sum();
    let total_deps: u64 = snapshots.iter().map(|m| m.dependency_count).sum();
    let total_txns: u64 = snapshots.iter().map(|m| m.transaction_count).sum();
    let total_ops: u64 = snapshots.iter().map(|m| m.operation_count).sum();

    // Merge extra maps by averaging
    let mut merged_extra: HashMap<String, f64> = HashMap::new();
    for snap in snapshots {
        for (k, v) in &snap.extra {
            *merged_extra.entry(k.clone()).or_default() += v / n as f64;
        }
    }

    AnalysisMetrics {
        analysis_time: total_time / n as u32,
        peak_memory_bytes: total_memory / n,
        smt_calls: total_smt_calls / n,
        smt_time: total_smt_time / n as u32,
        anomalies_found: total_anomalies,
        constraint_count: total_constraints / n,
        variable_count: total_variables / n,
        dependency_count: total_deps / n,
        transaction_count: total_txns / n,
        operation_count: total_ops / n,
        extra: merged_extra,
    }
}

/// Find the metric snapshot with the longest analysis time.
pub fn slowest_snapshot(snapshots: &[AnalysisMetrics]) -> Option<usize> {
    snapshots.iter()
        .enumerate()
        .max_by_key(|(_, m)| m.analysis_time)
        .map(|(i, _)| i)
}

/// Find the metric snapshot with the most anomalies.
pub fn most_anomalies_snapshot(snapshots: &[AnalysisMetrics]) -> Option<usize> {
    snapshots.iter()
        .enumerate()
        .max_by_key(|(_, m)| m.anomalies_found)
        .map(|(i, _)| i)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_metrics(time_ms: u64, anomalies: u64, smt_calls: u64) -> AnalysisMetrics {
        AnalysisMetrics::new()
            .with_time(Duration::from_millis(time_ms))
            .with_anomalies(anomalies)
            .with_smt(smt_calls, Duration::from_millis(time_ms / 2))
            .with_constraints(100 * smt_calls, 50 * smt_calls)
            .with_memory(1024 * 1024)
            .with_workload_size(3, 12)
            .with_dependencies(8)
    }

    #[test]
    fn test_analysis_metrics_builder() {
        let m = sample_metrics(500, 3, 10);
        assert_eq!(m.analysis_time, Duration::from_millis(500));
        assert_eq!(m.anomalies_found, 3);
        assert_eq!(m.smt_calls, 10);
        assert_eq!(m.constraint_count, 1000);
        assert_eq!(m.variable_count, 500);
    }

    #[test]
    fn test_smt_time_fraction() {
        let m = sample_metrics(1000, 0, 5);
        let frac = m.smt_time_fraction();
        assert!((frac - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_avg_smt_call_time() {
        let m = AnalysisMetrics::new()
            .with_smt(4, Duration::from_millis(200));
        let avg = m.avg_smt_call_time();
        assert_eq!(avg, Duration::from_millis(50));
    }

    #[test]
    fn test_constraints_per_txn() {
        let m = AnalysisMetrics::new()
            .with_constraints(300, 100)
            .with_workload_size(3, 12);
        assert!((m.constraints_per_txn() - 100.0).abs() < 1e-9);
        assert!((m.constraints_per_op() - 25.0).abs() < 1e-9);
    }

    #[test]
    fn test_metrics_collector() {
        let mut collector = MetricsCollector::new();
        collector.record("run1", sample_metrics(100, 2, 5));
        collector.record("run2", sample_metrics(200, 3, 10));
        collector.record("run3", sample_metrics(150, 1, 7));

        assert_eq!(collector.snapshot_count(), 3);
        assert_eq!(collector.total_anomalies(), 6);
        assert_eq!(collector.total_smt_calls(), 22);

        let ts = collector.timing_summary();
        assert_eq!(ts.count, 3);
        assert!((ts.mean_seconds - 0.15).abs() < 0.001);
        assert!((ts.min_seconds - 0.1).abs() < 0.001);
        assert!((ts.max_seconds - 0.2).abs() < 0.001);
    }

    #[test]
    fn test_collector_find_by_label() {
        let mut collector = MetricsCollector::new();
        collector.record("alpha", sample_metrics(100, 2, 5));
        collector.record("beta", sample_metrics(200, 3, 10));

        let m = collector.find_by_label("beta").unwrap();
        assert_eq!(m.anomalies_found, 3);
        assert!(collector.find_by_label("gamma").is_none());
    }

    #[test]
    fn test_comparison_metrics() {
        let a = sample_metrics(100, 3, 10);
        let b = sample_metrics(200, 5, 20);
        let cmp = ComparisonMetrics::compare("pg", &a, "mysql", &b);

        assert!(cmp.a_is_faster());
        assert!((cmp.time_ratio - 0.5).abs() < 0.01);
        assert_eq!(cmp.anomaly_diff, -2);
        assert!((cmp.speedup_b_over_a() - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_aggregate_metrics() {
        let snapshots = vec![
            sample_metrics(100, 2, 5),
            sample_metrics(200, 4, 15),
        ];
        let agg = aggregate_metrics(&snapshots);
        assert_eq!(agg.anomalies_found, 6); // sum
        assert_eq!(agg.smt_calls, 10); // avg
    }

    #[test]
    fn test_slowest_snapshot() {
        let snapshots = vec![
            sample_metrics(100, 0, 1),
            sample_metrics(500, 0, 1),
            sample_metrics(200, 0, 1),
        ];
        assert_eq!(slowest_snapshot(&snapshots), Some(1));
    }

    #[test]
    fn test_most_anomalies() {
        let snapshots = vec![
            sample_metrics(100, 1, 1),
            sample_metrics(100, 5, 1),
            sample_metrics(100, 3, 1),
        ];
        assert_eq!(most_anomalies_snapshot(&snapshots), Some(1));
    }

    #[test]
    fn test_collector_clear() {
        let mut collector = MetricsCollector::new();
        collector.record("x", sample_metrics(100, 1, 1));
        assert_eq!(collector.snapshot_count(), 1);
        collector.clear();
        assert_eq!(collector.snapshot_count(), 0);
    }

    #[test]
    fn test_extra_metrics() {
        let mut m = AnalysisMetrics::new();
        m.set_extra("cache_hits", 42.0);
        m.set_extra("cache_misses", 8.0);
        assert!((m.extra["cache_hits"] - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_aggregate() {
        let agg = aggregate_metrics(&[]);
        assert_eq!(agg.anomalies_found, 0);
        assert_eq!(agg.smt_calls, 0);
    }
}
