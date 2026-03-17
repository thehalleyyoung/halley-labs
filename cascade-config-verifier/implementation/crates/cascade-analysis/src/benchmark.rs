//! Benchmark evaluation harness for cascade analysis.
//!
//! Generates synthetic service topologies of increasing complexity and measures
//! the performance characteristics of the Tier 1 and Tier 2 analysis pipelines.
//! Results are collected as structured [`BenchmarkReport`] data suitable for
//! regression testing and performance dashboards.

use cascade_graph::rtig::{build_chain, build_diamond, RtigGraph};
use cascade_types::policy::{ResiliencePolicy, RetryPolicy, TimeoutPolicy};
use cascade_types::service::ServiceId;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Benchmark configuration
// ---------------------------------------------------------------------------

/// Configuration for a benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// Service counts to test at (e.g. [10, 50, 100, 500]).
    pub service_counts: Vec<usize>,
    /// Number of iterations per configuration for statistical stability.
    pub iterations: usize,
    /// Maximum wall-clock time per individual benchmark in milliseconds.
    pub timeout_ms: u64,
    /// Whether to include Tier 2 (deep) analysis benchmarks.
    pub include_tier2: bool,
    /// Retry counts to test with.
    pub retry_counts: Vec<u32>,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            service_counts: vec![5, 10, 25, 50, 100],
            iterations: 3,
            timeout_ms: 30_000,
            include_tier2: true,
            retry_counts: vec![1, 3, 5],
        }
    }
}

impl BenchmarkConfig {
    pub fn quick() -> Self {
        Self {
            service_counts: vec![5, 10, 25],
            iterations: 1,
            timeout_ms: 5_000,
            include_tier2: false,
            retry_counts: vec![2],
        }
    }

    pub fn full() -> Self {
        Self {
            service_counts: vec![5, 10, 25, 50, 100, 250, 500],
            iterations: 5,
            timeout_ms: 60_000,
            include_tier2: true,
            retry_counts: vec![1, 2, 3, 5, 10],
        }
    }

    pub fn total_runs(&self) -> usize {
        self.service_counts.len() * self.retry_counts.len() * self.iterations
    }
}

// ---------------------------------------------------------------------------
// Topology generators
// ---------------------------------------------------------------------------

/// Describes the shape of a synthetic topology.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TopologyShape {
    Chain,
    Diamond,
    Star,
    BinaryTree,
    FullMesh,
}

impl TopologyShape {
    pub fn all() -> Vec<TopologyShape> {
        vec![
            TopologyShape::Chain,
            TopologyShape::Diamond,
            TopologyShape::Star,
            TopologyShape::BinaryTree,
            TopologyShape::FullMesh,
        ]
    }

    pub fn name(&self) -> &'static str {
        match self {
            TopologyShape::Chain => "chain",
            TopologyShape::Diamond => "diamond",
            TopologyShape::Star => "star",
            TopologyShape::BinaryTree => "binary-tree",
            TopologyShape::FullMesh => "full-mesh",
        }
    }
}

/// Generates synthetic topologies for benchmarking.
pub struct TopologyGenerator;

impl TopologyGenerator {
    /// Generate a chain topology: S0 → S1 → ... → S(n-1).
    pub fn chain(n: usize, retries: u32) -> RtigGraph {
        let names: Vec<String> = (0..n).map(|i| format!("svc-{i}")).collect();
        let name_refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
        build_chain(&name_refs, retries)
    }

    /// Generate a star topology: hub → {spoke_0, spoke_1, ..., spoke_(n-2)}.
    pub fn star(n: usize, retries: u32) -> RtigGraph {
        let mut g = RtigGraph::new();
        let hub = ServiceId::new("hub");
        g.add_service(hub.clone());

        let policy = ResiliencePolicy::empty()
            .with_retry(RetryPolicy::new(retries))
            .with_timeout(TimeoutPolicy::new(5000));

        for i in 0..(n.saturating_sub(1)) {
            let spoke = ServiceId::new(format!("spoke-{i}"));
            g.add_service(spoke.clone());
            g.add_dependency(&hub, &spoke, policy.clone());
        }
        g
    }

    /// Generate a binary tree topology with `depth` levels.
    pub fn binary_tree(depth: usize, retries: u32) -> RtigGraph {
        let mut g = RtigGraph::new();
        let policy = ResiliencePolicy::empty()
            .with_retry(RetryPolicy::new(retries))
            .with_timeout(TimeoutPolicy::new(5000));

        let total_nodes = (1 << depth) - 1;
        let ids: Vec<ServiceId> = (0..total_nodes)
            .map(|i| ServiceId::new(format!("node-{i}")))
            .collect();

        for id in &ids {
            g.add_service(id.clone());
        }

        for i in 0..total_nodes {
            let left = 2 * i + 1;
            let right = 2 * i + 2;
            if left < total_nodes {
                g.add_dependency(&ids[i], &ids[left], policy.clone());
            }
            if right < total_nodes {
                g.add_dependency(&ids[i], &ids[right], policy.clone());
            }
        }
        g
    }

    /// Generate a full mesh topology (every service connects to every other).
    pub fn full_mesh(n: usize, retries: u32) -> RtigGraph {
        let mut g = RtigGraph::new();
        let policy = ResiliencePolicy::empty()
            .with_retry(RetryPolicy::new(retries));

        let ids: Vec<ServiceId> = (0..n)
            .map(|i| ServiceId::new(format!("mesh-{i}")))
            .collect();

        for id in &ids {
            g.add_service(id.clone());
        }

        for i in 0..n {
            for j in 0..n {
                if i != j {
                    g.add_dependency(&ids[i], &ids[j], policy.clone());
                }
            }
        }
        g
    }

    /// Generate a diamond topology scaled to `n` services.
    pub fn diamond_scaled(n: usize, retries: u32) -> RtigGraph {
        if n <= 4 {
            return build_diamond(retries);
        }
        let mut g = RtigGraph::new();
        let policy = ResiliencePolicy::empty()
            .with_retry(RetryPolicy::new(retries));

        let source = ServiceId::new("source");
        let sink = ServiceId::new("sink");
        g.add_service(source.clone());
        g.add_service(sink.clone());

        let middle_count = n.saturating_sub(2);
        for i in 0..middle_count {
            let mid = ServiceId::new(format!("mid-{i}"));
            g.add_service(mid.clone());
            g.add_dependency(&source, &mid, policy.clone());
            g.add_dependency(&mid, &sink, policy.clone());
        }
        g
    }

    /// Generate a topology of the given shape with approximately `n` services.
    pub fn generate(shape: TopologyShape, n: usize, retries: u32) -> RtigGraph {
        match shape {
            TopologyShape::Chain => Self::chain(n, retries),
            TopologyShape::Star => Self::star(n, retries),
            TopologyShape::BinaryTree => {
                let depth = (n as f64).log2().ceil() as usize;
                Self::binary_tree(depth.max(2), retries)
            }
            TopologyShape::FullMesh => Self::full_mesh(n.min(30), retries),
            TopologyShape::Diamond => Self::diamond_scaled(n, retries),
        }
    }
}

// ---------------------------------------------------------------------------
// Benchmark measurement
// ---------------------------------------------------------------------------

/// A single benchmark measurement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkMeasurement {
    pub topology: TopologyShape,
    pub service_count: usize,
    pub edge_count: usize,
    pub retry_count: u32,
    pub tier: String,
    pub elapsed: Duration,
    pub timed_out: bool,
}

impl BenchmarkMeasurement {
    pub fn throughput_services_per_sec(&self) -> f64 {
        let secs = self.elapsed.as_secs_f64();
        if secs == 0.0 {
            return f64::INFINITY;
        }
        self.service_count as f64 / secs
    }

    pub fn microseconds(&self) -> u128 {
        self.elapsed.as_micros()
    }
}

// ---------------------------------------------------------------------------
// Benchmark results
// ---------------------------------------------------------------------------

/// Summary statistics for a set of measurements.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkStats {
    pub min: Duration,
    pub max: Duration,
    pub mean: Duration,
    pub median: Duration,
    pub p95: Duration,
    pub count: usize,
}

impl BenchmarkStats {
    pub fn from_durations(mut durations: Vec<Duration>) -> Self {
        durations.sort();
        let count = durations.len();
        if count == 0 {
            return Self {
                min: Duration::ZERO,
                max: Duration::ZERO,
                mean: Duration::ZERO,
                median: Duration::ZERO,
                p95: Duration::ZERO,
                count: 0,
            };
        }

        let sum: Duration = durations.iter().sum();
        let mean = sum / count as u32;
        let median = durations[count / 2];
        let p95_idx = ((count as f64) * 0.95) as usize;
        let p95 = durations[p95_idx.min(count - 1)];

        Self {
            min: durations[0],
            max: durations[count - 1],
            mean,
            median,
            p95,
            count,
        }
    }

    pub fn mean_ms(&self) -> f64 {
        self.mean.as_secs_f64() * 1000.0
    }

    pub fn p95_ms(&self) -> f64 {
        self.p95.as_secs_f64() * 1000.0
    }
}

/// Complete benchmark report.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BenchmarkReport {
    pub measurements: Vec<BenchmarkMeasurement>,
    pub total_elapsed: Duration,
    pub config: Option<BenchmarkConfig>,
}

impl BenchmarkReport {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_measurement(&mut self, m: BenchmarkMeasurement) {
        self.measurements.push(m);
    }

    pub fn measurement_count(&self) -> usize {
        self.measurements.len()
    }

    pub fn filter_by_topology(&self, shape: TopologyShape) -> Vec<&BenchmarkMeasurement> {
        self.measurements
            .iter()
            .filter(|m| m.topology == shape)
            .collect()
    }

    pub fn filter_by_tier(&self, tier: &str) -> Vec<&BenchmarkMeasurement> {
        self.measurements
            .iter()
            .filter(|m| m.tier == tier)
            .collect()
    }

    pub fn stats_for(&self, shape: TopologyShape, tier: &str) -> BenchmarkStats {
        let durations: Vec<Duration> = self
            .measurements
            .iter()
            .filter(|m| m.topology == shape && m.tier == tier)
            .map(|m| m.elapsed)
            .collect();
        BenchmarkStats::from_durations(durations)
    }

    pub fn timed_out_count(&self) -> usize {
        self.measurements.iter().filter(|m| m.timed_out).count()
    }
}

// ---------------------------------------------------------------------------
// Benchmark runner
// ---------------------------------------------------------------------------

/// Runs the benchmark evaluation harness.
pub struct BenchmarkRunner {
    config: BenchmarkConfig,
}

impl BenchmarkRunner {
    pub fn new(config: BenchmarkConfig) -> Self {
        Self { config }
    }

    /// Run Tier 1 analysis benchmarks across all configured topologies.
    pub fn run_tier1_benchmarks(&self) -> BenchmarkReport {
        let mut report = BenchmarkReport::new();
        let start = Instant::now();

        for &n in &self.config.service_counts {
            for &retries in &self.config.retry_counts {
                for shape in TopologyShape::all() {
                    for _ in 0..self.config.iterations {
                        let graph = TopologyGenerator::generate(shape, n, retries);
                        let m = self.measure_tier1(&graph, shape, retries);
                        report.add_measurement(m);
                    }
                }
            }
        }

        report.total_elapsed = start.elapsed();
        report.config = Some(self.config.clone());
        report
    }

    fn measure_tier1(
        &self,
        graph: &RtigGraph,
        shape: TopologyShape,
        retries: u32,
    ) -> BenchmarkMeasurement {
        let start = Instant::now();

        // Tier 1 analysis operations: graph stats, path analysis, fan-in/out
        let _stats = graph.graph_stats();
        let _fan_in = graph.get_fan_in_services(2);
        let _fan_out = graph.get_fan_out_services(2);
        let _is_dag = graph.is_dag();
        let _cycles = graph.detect_cycles();
        let _diameter = graph.compute_diameter();
        let _treewidth = graph.compute_treewidth_estimate();

        if let Some(topo) = graph.topological_sort() {
            if topo.len() >= 2 {
                let _paths = graph.get_all_paths(&topo[0], &topo[topo.len() - 1]);
            }
        }

        let elapsed = start.elapsed();
        let timed_out = elapsed.as_millis() as u64 > self.config.timeout_ms;

        BenchmarkMeasurement {
            topology: shape,
            service_count: graph.service_count(),
            edge_count: graph.dependency_count(),
            retry_count: retries,
            tier: "tier1".into(),
            elapsed,
            timed_out,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- BenchmarkConfig ----------------------------------------------------

    #[test]
    fn test_config_default() {
        let cfg = BenchmarkConfig::default();
        assert!(!cfg.service_counts.is_empty());
        assert!(cfg.iterations >= 1);
        assert!(cfg.timeout_ms > 0);
    }

    #[test]
    fn test_config_quick() {
        let cfg = BenchmarkConfig::quick();
        assert!(!cfg.include_tier2);
        assert!(cfg.iterations <= 2);
    }

    #[test]
    fn test_config_full() {
        let cfg = BenchmarkConfig::full();
        assert!(cfg.include_tier2);
        assert!(cfg.service_counts.len() > 5);
    }

    #[test]
    fn test_config_total_runs() {
        let cfg = BenchmarkConfig {
            service_counts: vec![10, 20],
            iterations: 3,
            retry_counts: vec![1, 2],
            ..Default::default()
        };
        assert_eq!(cfg.total_runs(), 2 * 2 * 3);
    }

    #[test]
    fn test_config_serde() {
        let cfg = BenchmarkConfig::default();
        let json = serde_json::to_string(&cfg).unwrap();
        let deser: BenchmarkConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.service_counts, cfg.service_counts);
    }

    // -- TopologyShape ------------------------------------------------------

    #[test]
    fn test_topology_shape_all() {
        let all = TopologyShape::all();
        assert_eq!(all.len(), 5);
    }

    #[test]
    fn test_topology_shape_names() {
        assert_eq!(TopologyShape::Chain.name(), "chain");
        assert_eq!(TopologyShape::Diamond.name(), "diamond");
        assert_eq!(TopologyShape::Star.name(), "star");
        assert_eq!(TopologyShape::BinaryTree.name(), "binary-tree");
        assert_eq!(TopologyShape::FullMesh.name(), "full-mesh");
    }

    #[test]
    fn test_topology_shape_serde() {
        let shape = TopologyShape::Star;
        let json = serde_json::to_string(&shape).unwrap();
        let deser: TopologyShape = serde_json::from_str(&json).unwrap();
        assert_eq!(deser, TopologyShape::Star);
    }

    // -- TopologyGenerator --------------------------------------------------

    #[test]
    fn test_generate_chain() {
        let g = TopologyGenerator::chain(5, 2);
        assert_eq!(g.service_count(), 5);
        assert_eq!(g.dependency_count(), 4);
        assert!(g.is_dag());
    }

    #[test]
    fn test_generate_star() {
        let g = TopologyGenerator::star(6, 3);
        assert_eq!(g.service_count(), 6);
        assert_eq!(g.dependency_count(), 5);
        assert!(g.is_dag());
    }

    #[test]
    fn test_generate_star_single() {
        let g = TopologyGenerator::star(1, 1);
        assert_eq!(g.service_count(), 1);
        assert_eq!(g.dependency_count(), 0);
    }

    #[test]
    fn test_generate_binary_tree() {
        let g = TopologyGenerator::binary_tree(3, 1);
        assert_eq!(g.service_count(), 7);
        assert_eq!(g.dependency_count(), 6);
        assert!(g.is_dag());
    }

    #[test]
    fn test_generate_binary_tree_depth_2() {
        let g = TopologyGenerator::binary_tree(2, 2);
        assert_eq!(g.service_count(), 3);
        assert_eq!(g.dependency_count(), 2);
    }

    #[test]
    fn test_generate_full_mesh() {
        let g = TopologyGenerator::full_mesh(4, 1);
        assert_eq!(g.service_count(), 4);
        assert_eq!(g.dependency_count(), 12); // 4 * 3
        assert!(!g.is_dag()); // Full mesh has cycles
    }

    #[test]
    fn test_generate_diamond_scaled() {
        let g = TopologyGenerator::diamond_scaled(10, 2);
        assert_eq!(g.service_count(), 10);
        // source + sink + 8 middle nodes, edges = 8 + 8 = 16
        assert_eq!(g.dependency_count(), 16);
    }

    #[test]
    fn test_generate_diamond_small() {
        let g = TopologyGenerator::diamond_scaled(4, 1);
        assert_eq!(g.service_count(), 4);
    }

    #[test]
    fn test_generate_dispatch() {
        for shape in TopologyShape::all() {
            let g = TopologyGenerator::generate(shape, 10, 2);
            assert!(g.service_count() > 0);
        }
    }

    // -- BenchmarkMeasurement -----------------------------------------------

    #[test]
    fn test_measurement_throughput() {
        let m = BenchmarkMeasurement {
            topology: TopologyShape::Chain,
            service_count: 100,
            edge_count: 99,
            retry_count: 2,
            tier: "tier1".into(),
            elapsed: Duration::from_millis(50),
            timed_out: false,
        };
        assert!(m.throughput_services_per_sec() > 1000.0);
        assert_eq!(m.microseconds(), 50_000);
    }

    #[test]
    fn test_measurement_zero_duration() {
        let m = BenchmarkMeasurement {
            topology: TopologyShape::Star,
            service_count: 10,
            edge_count: 9,
            retry_count: 1,
            tier: "tier1".into(),
            elapsed: Duration::ZERO,
            timed_out: false,
        };
        assert!(m.throughput_services_per_sec().is_infinite());
    }

    // -- BenchmarkStats -----------------------------------------------------

    #[test]
    fn test_stats_from_durations() {
        let durations = vec![
            Duration::from_millis(10),
            Duration::from_millis(20),
            Duration::from_millis(30),
            Duration::from_millis(40),
            Duration::from_millis(50),
        ];
        let stats = BenchmarkStats::from_durations(durations);
        assert_eq!(stats.min, Duration::from_millis(10));
        assert_eq!(stats.max, Duration::from_millis(50));
        assert_eq!(stats.mean, Duration::from_millis(30));
        assert_eq!(stats.median, Duration::from_millis(30));
        assert_eq!(stats.count, 5);
    }

    #[test]
    fn test_stats_empty() {
        let stats = BenchmarkStats::from_durations(vec![]);
        assert_eq!(stats.count, 0);
        assert_eq!(stats.min, Duration::ZERO);
    }

    #[test]
    fn test_stats_single() {
        let stats = BenchmarkStats::from_durations(vec![Duration::from_millis(42)]);
        assert_eq!(stats.min, Duration::from_millis(42));
        assert_eq!(stats.max, Duration::from_millis(42));
        assert_eq!(stats.count, 1);
    }

    #[test]
    fn test_stats_mean_ms() {
        let stats = BenchmarkStats::from_durations(vec![
            Duration::from_millis(100),
            Duration::from_millis(200),
        ]);
        assert!((stats.mean_ms() - 150.0).abs() < 1.0);
    }

    #[test]
    fn test_stats_serde() {
        let stats = BenchmarkStats::from_durations(vec![Duration::from_millis(10)]);
        let json = serde_json::to_string(&stats).unwrap();
        let deser: BenchmarkStats = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.count, 1);
    }

    // -- BenchmarkReport ----------------------------------------------------

    #[test]
    fn test_report_empty() {
        let report = BenchmarkReport::new();
        assert_eq!(report.measurement_count(), 0);
        assert_eq!(report.timed_out_count(), 0);
    }

    #[test]
    fn test_report_add_measurement() {
        let mut report = BenchmarkReport::new();
        report.add_measurement(BenchmarkMeasurement {
            topology: TopologyShape::Chain,
            service_count: 10,
            edge_count: 9,
            retry_count: 2,
            tier: "tier1".into(),
            elapsed: Duration::from_millis(5),
            timed_out: false,
        });
        assert_eq!(report.measurement_count(), 1);
    }

    #[test]
    fn test_report_filter_by_topology() {
        let mut report = BenchmarkReport::new();
        for shape in [TopologyShape::Chain, TopologyShape::Star, TopologyShape::Chain] {
            report.add_measurement(BenchmarkMeasurement {
                topology: shape,
                service_count: 10,
                edge_count: 9,
                retry_count: 1,
                tier: "tier1".into(),
                elapsed: Duration::from_millis(1),
                timed_out: false,
            });
        }
        assert_eq!(report.filter_by_topology(TopologyShape::Chain).len(), 2);
        assert_eq!(report.filter_by_topology(TopologyShape::Star).len(), 1);
    }

    #[test]
    fn test_report_filter_by_tier() {
        let mut report = BenchmarkReport::new();
        for tier in ["tier1", "tier2", "tier1"] {
            report.add_measurement(BenchmarkMeasurement {
                topology: TopologyShape::Chain,
                service_count: 10,
                edge_count: 9,
                retry_count: 1,
                tier: tier.into(),
                elapsed: Duration::from_millis(1),
                timed_out: false,
            });
        }
        assert_eq!(report.filter_by_tier("tier1").len(), 2);
        assert_eq!(report.filter_by_tier("tier2").len(), 1);
    }

    #[test]
    fn test_report_stats_for() {
        let mut report = BenchmarkReport::new();
        for ms in [10, 20, 30] {
            report.add_measurement(BenchmarkMeasurement {
                topology: TopologyShape::Star,
                service_count: 10,
                edge_count: 9,
                retry_count: 1,
                tier: "tier1".into(),
                elapsed: Duration::from_millis(ms),
                timed_out: false,
            });
        }
        let stats = report.stats_for(TopologyShape::Star, "tier1");
        assert_eq!(stats.count, 3);
        assert_eq!(stats.min, Duration::from_millis(10));
    }

    #[test]
    fn test_report_timed_out_count() {
        let mut report = BenchmarkReport::new();
        report.add_measurement(BenchmarkMeasurement {
            topology: TopologyShape::FullMesh,
            service_count: 30,
            edge_count: 870,
            retry_count: 5,
            tier: "tier1".into(),
            elapsed: Duration::from_secs(31),
            timed_out: true,
        });
        report.add_measurement(BenchmarkMeasurement {
            topology: TopologyShape::Chain,
            service_count: 10,
            edge_count: 9,
            retry_count: 1,
            tier: "tier1".into(),
            elapsed: Duration::from_millis(5),
            timed_out: false,
        });
        assert_eq!(report.timed_out_count(), 1);
    }

    #[test]
    fn test_report_serde() {
        let mut report = BenchmarkReport::new();
        report.add_measurement(BenchmarkMeasurement {
            topology: TopologyShape::Chain,
            service_count: 5,
            edge_count: 4,
            retry_count: 1,
            tier: "tier1".into(),
            elapsed: Duration::from_millis(1),
            timed_out: false,
        });
        let json = serde_json::to_string(&report).unwrap();
        let deser: BenchmarkReport = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.measurement_count(), 1);
    }

    // -- BenchmarkRunner ----------------------------------------------------

    #[test]
    fn test_runner_quick_benchmark() {
        let config = BenchmarkConfig {
            service_counts: vec![5],
            iterations: 1,
            timeout_ms: 5_000,
            include_tier2: false,
            retry_counts: vec![1],
        };
        let runner = BenchmarkRunner::new(config);
        let report = runner.run_tier1_benchmarks();
        // 1 service_count * 1 retry * 5 shapes * 1 iteration = 5
        assert_eq!(report.measurement_count(), 5);
        assert!(report.total_elapsed.as_millis() > 0);
    }

    #[test]
    fn test_runner_all_shapes() {
        let config = BenchmarkConfig {
            service_counts: vec![10],
            iterations: 1,
            timeout_ms: 10_000,
            include_tier2: false,
            retry_counts: vec![2],
        };
        let runner = BenchmarkRunner::new(config);
        let report = runner.run_tier1_benchmarks();

        for shape in TopologyShape::all() {
            let filtered = report.filter_by_topology(shape);
            assert_eq!(
                filtered.len(),
                1,
                "Expected 1 measurement for {:?}",
                shape
            );
        }
    }

    #[test]
    fn test_runner_multiple_configs() {
        let config = BenchmarkConfig {
            service_counts: vec![5, 10],
            iterations: 2,
            timeout_ms: 10_000,
            include_tier2: false,
            retry_counts: vec![1, 3],
        };
        let runner = BenchmarkRunner::new(config.clone());
        let report = runner.run_tier1_benchmarks();
        // 2 counts * 2 retries * 5 shapes * 2 iters = 40
        assert_eq!(report.measurement_count(), 40);
        assert!(report.config.is_some());
    }
}
