//! Capacity modelling – estimation, inference, load modelling,
//! headroom analysis, and bottleneck detection.

use cascade_graph::rtig::{RtigGraph, ServiceNode};
use serde::{Deserialize, Serialize};

use crate::mesh::ServiceMesh;

// ── CapacityEstimate ────────────────────────────────────────────────

/// Capacity characterisation of a service.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapacityEstimate {
    pub baseline_rps: u64,
    pub peak_rps: u64,
    pub burst_capacity: u64,
    pub degradation_threshold: u64,
    pub source: CapacitySource,
}

/// Where the capacity estimate came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CapacitySource {
    Explicit,
    InferredFromResources,
    InferredFromLimits,
    InferredFromReplicas,
    Default,
}

/// Failure scenario for burst-load estimation.
#[derive(Debug, Clone)]
pub struct FailureSet {
    pub failed_services: Vec<String>,
}

/// A capacity bottleneck detected in the mesh.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapacityBottleneck {
    pub service: String,
    pub capacity: u64,
    pub peak_load: u64,
    pub headroom_pct: f64,
}

// ── CapacityModel ───────────────────────────────────────────────────

/// Core capacity-estimation logic.
pub struct CapacityModel;

impl CapacityModel {
    /// Estimate capacity from the explicit capacity field on a [`ServiceNode`].
    pub fn estimate_capacity(node: &ServiceNode) -> CapacityEstimate {
        let cap = node.capacity;
        if cap == 0 {
            return Self::default_estimate();
        }
        CapacityEstimate {
            baseline_rps: cap as u64,
            peak_rps: (cap as f64 * 1.5) as u64,
            burst_capacity: cap as u64 * 2,
            degradation_threshold: (cap as f64 * 0.8) as u64,
            source: CapacitySource::Explicit,
        }
    }

    /// Infer capacity from CPU and memory resource limits.
    ///
    /// Heuristic:
    ///   rps ≈ cpu_cores × 500  (typical for I/O-bound microservices)
    ///   Adjusted down if memory < 512 MiB.
    pub fn infer_from_resources(cpu_limit: f64, memory_limit_mb: u64) -> CapacityEstimate {
        let base_rps = (cpu_limit * 500.0) as u64;
        let mem_factor: f64 = if memory_limit_mb >= 512 {
            1.0
        } else if memory_limit_mb >= 256 {
            0.75
        } else {
            0.5
        };
        let adjusted = (base_rps as f64 * mem_factor) as u64;

        CapacityEstimate {
            baseline_rps: adjusted,
            peak_rps: (adjusted as f64 * 1.5) as u64,
            burst_capacity: adjusted * 2,
            degradation_threshold: (adjusted as f64 * 0.8) as u64,
            source: CapacitySource::InferredFromResources,
        }
    }

    /// Infer capacity from replica count and per-replica capacity.
    pub fn infer_from_replicas(replicas: u32, per_replica_capacity: u64) -> CapacityEstimate {
        let total = replicas as u64 * per_replica_capacity;
        CapacityEstimate {
            baseline_rps: total,
            peak_rps: (total as f64 * 1.3) as u64,
            burst_capacity: total + per_replica_capacity, // one extra replica worth
            degradation_threshold: (total as f64 * 0.75) as u64,
            source: CapacitySource::InferredFromReplicas,
        }
    }

    /// Infer from rate-limit configuration.
    pub fn infer_from_rate_limit(rps: f64, burst: u32) -> CapacityEstimate {
        let base = rps as u64;
        CapacityEstimate {
            baseline_rps: base,
            peak_rps: base + burst as u64,
            burst_capacity: base + burst as u64,
            degradation_threshold: (base as f64 * 0.9) as u64,
            source: CapacitySource::InferredFromLimits,
        }
    }

    fn default_estimate() -> CapacityEstimate {
        CapacityEstimate {
            baseline_rps: 100,
            peak_rps: 150,
            burst_capacity: 200,
            degradation_threshold: 80,
            source: CapacitySource::Default,
        }
    }
}

// ── LoadModel ───────────────────────────────────────────────────────

/// Load estimation for individual services within a mesh.
pub struct LoadModel;

impl LoadModel {
    /// Peak load = maximum possible incoming load accounting for all
    /// retry amplification paths.
    pub fn compute_peak_load(mesh: &ServiceMesh, service: &str) -> u64 {
        let graph = mesh.graph();
        let roots: Vec<String> = graph.roots().iter().map(|s| s.to_string()).collect();
        let mut max_load = 0u64;

        for root in &roots {
            let root_cap = graph.service(root).map(|n| n.capacity as u64).unwrap_or(100);
            let paths = Self::enumerate_paths(&graph, root, service);
            for path in &paths {
                let amp = Self::path_amplification(&graph, path);
                let load = (root_cap as f64 * amp) as u64;
                max_load = max_load.max(load);
            }
        }
        // Add baseline load
        let baseline = graph.service(service).map(|n| n.baseline_load as u64).unwrap_or(0);
        max_load.max(baseline)
    }

    /// Sustained load = sum of baseline loads from all predecessors,
    /// weighted by amplification factors.
    pub fn compute_sustained_load(mesh: &ServiceMesh, service: &str) -> u64 {
        let graph = mesh.graph();
        let incoming = graph.incoming_edges(service);
        if incoming.is_empty() {
            return graph.service(service).map(|n| n.baseline_load as u64).unwrap_or(0);
        }

        let mut total = 0u64;
        for edge in &incoming {
            let src_load = graph
                .service(edge.source.as_str())
                .map(|n| n.baseline_load as u64)
                .unwrap_or(0);
            let factor = edge.amplification_factor_f64().max(1.0);
            total += (src_load as f64 * factor) as u64;
        }
        total
    }

    /// Burst load = sustained load + extra load caused by failures in
    /// the given failure set, accounting for retry storms.
    pub fn compute_burst_load(
        mesh: &ServiceMesh,
        service: &str,
        failure_set: &FailureSet,
    ) -> u64 {
        let sustained = Self::compute_sustained_load(mesh, service);
        let graph = mesh.graph();
        let failed_set: std::collections::HashSet<&str> =
            failure_set.failed_services.iter().map(|s| s.as_str()).collect();

        // Estimate additional load from retries to failed services
        let mut extra = 0u64;
        let outgoing = graph.outgoing_edges(service);
        for edge in &outgoing {
            if failed_set.contains(edge.target.as_str()) {
                let retries = edge.retry_count as u64;
                let base = graph
                    .service(service)
                    .map(|n| n.baseline_load as u64)
                    .unwrap_or(0);
                extra += base * retries;
            }
        }

        // Also account for redirected load from predecessors of failed services
        for failed in &failure_set.failed_services {
            let preds = graph.predecessors(failed);
            for pred in &preds {
                if *pred != service {
                    // load from pred may be redirected to healthy alternatives
                    let pred_edges = graph.outgoing_edges(pred);
                    let total_targets = pred_edges.len().max(1) as u64;
                    let pred_load = graph.service(pred).map(|n| n.baseline_load as u64).unwrap_or(0);
                    extra += pred_load / total_targets;
                }
            }
        }

        sustained + extra
    }

    fn enumerate_paths(graph: &RtigGraph, from: &str, to: &str) -> Vec<Vec<String>> {
        let mut results = Vec::new();
        let mut path = vec![from.to_string()];
        let mut visited = std::collections::HashSet::new();
        visited.insert(from.to_string());
        Self::dfs(graph, from, to, &mut visited, &mut path, &mut results);
        results
    }

    fn dfs(
        graph: &RtigGraph,
        cur: &str,
        end: &str,
        visited: &mut std::collections::HashSet<String>,
        path: &mut Vec<String>,
        results: &mut Vec<Vec<String>>,
    ) {
        if cur == end && path.len() > 1 {
            results.push(path.clone());
            return;
        }
        for next in graph.successors(cur) {
            if !visited.contains(next) {
                visited.insert(next.to_string());
                path.push(next.to_string());
                Self::dfs(graph, next, end, visited, path, results);
                path.pop();
                visited.remove(next);
            }
        }
    }

    fn path_amplification(graph: &RtigGraph, path: &[String]) -> f64 {
        let mut amp = 1.0f64;
        for w in path.windows(2) {
            let edges = graph.outgoing_edges(&w[0]);
            let hop_amp = edges
                .iter()
                .filter(|e| e.target.as_str() == w[1])
                .map(|e| e.amplification_factor_f64())
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(1.0);
            amp *= hop_amp;
        }
        amp
    }
}

// ── CapacityPlanner ─────────────────────────────────────────────────

/// Planning tools for capacity provisioning.
pub struct CapacityPlanner;

impl CapacityPlanner {
    /// Required capacity = peak_load × (1 + safety_margin).
    pub fn compute_required_capacity(
        mesh: &ServiceMesh,
        service: &str,
        safety_margin: f64,
    ) -> u64 {
        let peak = LoadModel::compute_peak_load(mesh, service);
        (peak as f64 * (1.0 + safety_margin)) as u64
    }

    /// Headroom percentage = (capacity - current_load) / capacity × 100.
    pub fn compute_headroom(node: &ServiceNode, current_load: u64) -> f64 {
        if node.capacity == 0 {
            return 0.0;
        }
        let cap = node.capacity as u64;
        let remaining = cap.saturating_sub(current_load);
        remaining as f64 / cap as f64 * 100.0
    }

    /// Find services that are closest to their capacity limits.
    pub fn find_capacity_bottlenecks(mesh: &ServiceMesh) -> Vec<CapacityBottleneck> {
        let graph = mesh.graph();
        let ids: Vec<String> = graph.service_ids().iter().map(|s| s.to_string()).collect();
        let mut bottlenecks = Vec::new();

        for id in &ids {
            let node = match graph.service(id) {
                Some(n) => n,
                None => continue,
            };
            let peak = LoadModel::compute_peak_load(mesh, id);
            let headroom = Self::compute_headroom(node, peak);

            if headroom < 50.0 {
                bottlenecks.push(CapacityBottleneck {
                    service: id.clone(),
                    capacity: node.capacity as u64,
                    peak_load: peak,
                    headroom_pct: headroom,
                });
            }
        }

        bottlenecks.sort_by(|a, b| {
            a.headroom_pct
                .partial_cmp(&b.headroom_pct)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        bottlenecks
    }

    /// Recommend replica scaling for a service to meet a target headroom.
    pub fn recommend_replicas(
        node: &ServiceNode,
        target_load: u64,
        target_headroom_pct: f64,
    ) -> u32 {
        if node.capacity == 0 {
            return 1;
        }
        let cap = node.capacity as u64;
        let required_total = (target_load as f64 / (1.0 - target_headroom_pct / 100.0)) as u64;
        let replicas = (required_total + cap - 1) / cap;
        replicas.max(1) as u32
    }

    /// Summary of capacity across the entire mesh.
    pub fn capacity_summary(mesh: &ServiceMesh) -> CapacitySummary {
        let graph = mesh.graph();
        let ids: Vec<String> = graph.service_ids().iter().map(|s| s.to_string()).collect();

        let mut total_capacity = 0u64;
        let mut total_peak_load = 0u64;
        let mut min_headroom = f64::MAX;
        let mut min_headroom_svc = String::new();

        for id in &ids {
            let node = match graph.service(id) {
                Some(n) => n,
                None => continue,
            };
            total_capacity += node.capacity as u64;
            let peak = LoadModel::compute_peak_load(mesh, id);
            total_peak_load += peak;
            let h = Self::compute_headroom(node, peak);
            if h < min_headroom {
                min_headroom = h;
                min_headroom_svc = id.clone();
            }
        }

        let overall_headroom = if total_capacity > 0 {
            (total_capacity.saturating_sub(total_peak_load)) as f64 / total_capacity as f64 * 100.0
        } else {
            0.0
        };

        CapacitySummary {
            total_capacity,
            total_peak_load,
            overall_headroom_pct: overall_headroom,
            tightest_service: min_headroom_svc,
            tightest_headroom_pct: if min_headroom == f64::MAX {
                0.0
            } else {
                min_headroom
            },
        }
    }
}

/// High-level capacity summary for a mesh.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapacitySummary {
    pub total_capacity: u64,
    pub total_peak_load: u64,
    pub overall_headroom_pct: f64,
    pub tightest_service: String,
    pub tightest_headroom_pct: f64,
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::MeshBuilder;
    use cascade_graph::rtig::{DependencyEdgeInfo, ServiceNode};
    use cascade_types::topology::DependencyType;

    fn simple_mesh() -> ServiceMesh {
        MeshBuilder::new()
            .add_node(ServiceNode::new("gw", 1000).with_baseline_load(200))
            .unwrap()
            .add_node(ServiceNode::new("api", 500).with_baseline_load(100))
            .unwrap()
            .add_node(ServiceNode::new("db", 200).with_baseline_load(50))
            .unwrap()
            .add_dependency_with_retries("gw", "api", DependencyType::Synchronous, 3, 2000)
            .unwrap()
            .add_dependency_with_retries("api", "db", DependencyType::Synchronous, 2, 1000)
            .unwrap()
            .build()
            .unwrap()
    }

    fn overloaded_mesh() -> ServiceMesh {
        MeshBuilder::new()
            .add_node(ServiceNode::new("entry", 100).with_baseline_load(90))
            .unwrap()
            .add_node(ServiceNode::new("backend", 50).with_baseline_load(40))
            .unwrap()
            .add_dependency_with_retries("entry", "backend", DependencyType::Synchronous, 5, 1000)
            .unwrap()
            .build()
            .unwrap()
    }

    // ── CapacityModel ──────

    #[test]
    fn estimate_explicit_capacity() {
        let node = ServiceNode::new("svc", 500);
        let est = CapacityModel::estimate_capacity(&node);
        assert_eq!(est.baseline_rps, 500);
        assert!(est.peak_rps > est.baseline_rps);
        assert!(est.degradation_threshold < est.baseline_rps);
        assert_eq!(est.source, CapacitySource::Explicit);
    }

    #[test]
    fn estimate_zero_capacity_defaults() {
        let node = ServiceNode::new("svc", 0);
        let est = CapacityModel::estimate_capacity(&node);
        assert_eq!(est.source, CapacitySource::Default);
        assert!(est.baseline_rps > 0);
    }

    #[test]
    fn infer_from_resources() {
        let est = CapacityModel::infer_from_resources(2.0, 1024);
        assert_eq!(est.baseline_rps, 1000); // 2 cores × 500
        assert_eq!(est.source, CapacitySource::InferredFromResources);
    }

    #[test]
    fn infer_from_resources_low_memory() {
        let est = CapacityModel::infer_from_resources(2.0, 256);
        assert_eq!(est.baseline_rps, 750); // 1000 * 0.75
    }

    #[test]
    fn infer_from_replicas() {
        let est = CapacityModel::infer_from_replicas(3, 200);
        assert_eq!(est.baseline_rps, 600);
        assert_eq!(est.source, CapacitySource::InferredFromReplicas);
    }

    #[test]
    fn infer_from_rate_limit() {
        let est = CapacityModel::infer_from_rate_limit(500.0, 100);
        assert_eq!(est.baseline_rps, 500);
        assert_eq!(est.peak_rps, 600);
    }

    // ── LoadModel ──────────

    #[test]
    fn peak_load_simple() {
        let mesh = simple_mesh();
        let peak = LoadModel::compute_peak_load(&mesh, "db");
        assert!(peak > 0);
    }

    #[test]
    fn sustained_load_leaf() {
        let mesh = simple_mesh();
        let load = LoadModel::compute_sustained_load(&mesh, "db");
        assert!(load > 0);
    }

    #[test]
    fn sustained_load_root() {
        let mesh = simple_mesh();
        let load = LoadModel::compute_sustained_load(&mesh, "gw");
        assert_eq!(load, 200); // baseline
    }

    #[test]
    fn burst_load_with_failure() {
        let mesh = simple_mesh();
        let failure = FailureSet {
            failed_services: vec!["db".to_string()],
        };
        let burst = LoadModel::compute_burst_load(&mesh, "api", &failure);
        let sustained = LoadModel::compute_sustained_load(&mesh, "api");
        assert!(burst >= sustained);
    }

    #[test]
    fn burst_load_no_failure() {
        let mesh = simple_mesh();
        let failure = FailureSet {
            failed_services: vec![],
        };
        let burst = LoadModel::compute_burst_load(&mesh, "api", &failure);
        let sustained = LoadModel::compute_sustained_load(&mesh, "api");
        assert_eq!(burst, sustained);
    }

    // ── CapacityPlanner ────

    #[test]
    fn required_capacity() {
        let mesh = simple_mesh();
        let req = CapacityPlanner::compute_required_capacity(&mesh, "db", 0.2);
        let peak = LoadModel::compute_peak_load(&mesh, "db");
        assert!(req >= peak);
    }

    #[test]
    fn headroom_calculation() {
        let node = ServiceNode::new("svc", 100);
        let h = CapacityPlanner::compute_headroom(&node, 70);
        assert!((h - 30.0).abs() < 0.01);
    }

    #[test]
    fn headroom_zero_capacity() {
        let node = ServiceNode::new("svc", 0);
        assert_eq!(CapacityPlanner::compute_headroom(&node, 50), 0.0);
    }

    #[test]
    fn find_bottlenecks() {
        let mesh = overloaded_mesh();
        let bottlenecks = CapacityPlanner::find_capacity_bottlenecks(&mesh);
        assert!(!bottlenecks.is_empty());
        // backend should be the tightest (50 capacity, high load)
        assert!(bottlenecks.iter().any(|b| b.service == "backend"));
    }

    #[test]
    fn recommend_replicas() {
        let node = ServiceNode::new("svc", 100);
        let r = CapacityPlanner::recommend_replicas(&node, 250, 20.0);
        // Need 250 / 0.8 = 312.5, ceil(312.5 / 100) = 4
        assert!(r >= 3);
    }

    #[test]
    fn capacity_summary() {
        let mesh = simple_mesh();
        let summary = CapacityPlanner::capacity_summary(&mesh);
        assert!(summary.total_capacity > 0);
        assert!(!summary.tightest_service.is_empty());
    }

    #[test]
    fn capacity_summary_headroom_range() {
        let mesh = simple_mesh();
        let summary = CapacityPlanner::capacity_summary(&mesh);
        assert!(summary.overall_headroom_pct >= 0.0);
        assert!(summary.overall_headroom_pct <= 100.0);
    }
}
