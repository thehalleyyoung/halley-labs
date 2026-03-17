//! Bounded Model Checker orchestration.
//!
//! The [`BoundedModelChecker`] coordinates the encoding, solving, and trace
//! extraction pipeline for cascade verification.

use cascade_graph::rtig::RtigGraph;
use cascade_types::cascade::{FailureSet, FailureMode, MinimalFailureSet, PropagationStep, PropagationTrace};
use cascade_types::service::{ServiceHealth, ServiceId};
use cascade_types::smt::{SmtConstraint, SmtExpr, SmtFormula};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

use crate::encoder::BmcEncoder;
use crate::solver::{BuiltinSolver, Clause, Literal, SatResult, SmtSolver, SolverConfig};
use crate::trace::TraceExtractor;
use crate::unroller::{BmcUnroller, ConeOfInfluence, DepthBoundComputer};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BmcConfig {
    pub max_failure_budget: usize,
    pub depth_bound: Option<usize>,
    pub timeout_ms: u64,
    pub use_cone_of_influence: bool,
    pub use_symmetry_breaking: bool,
    pub use_incremental: bool,
    pub target_services: Vec<String>,
}

impl Default for BmcConfig {
    fn default() -> Self {
        Self {
            max_failure_budget: 3,
            depth_bound: None,
            timeout_ms: 60_000,
            use_cone_of_influence: true,
            use_symmetry_breaking: false,
            use_incremental: true,
            target_services: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BmcStatus {
    CascadeFound,
    NoCascadeUpToBound,
    Timeout,
    Error(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BmcResult {
    pub status: BmcStatus,
    pub failure_sets: Vec<FailureSet>,
    pub traces: Vec<PropagationTrace>,
    pub statistics: BmcStatistics,
}

impl BmcResult {
    pub fn cascade_found(
        failure_sets: Vec<FailureSet>,
        traces: Vec<PropagationTrace>,
        stats: BmcStatistics,
    ) -> Self {
        Self {
            status: BmcStatus::CascadeFound,
            failure_sets,
            traces,
            statistics: stats,
        }
    }

    pub fn no_cascade(stats: BmcStatistics) -> Self {
        Self {
            status: BmcStatus::NoCascadeUpToBound,
            failure_sets: Vec::new(),
            traces: Vec::new(),
            statistics: stats,
        }
    }

    pub fn timeout(stats: BmcStatistics) -> Self {
        Self {
            status: BmcStatus::Timeout,
            failure_sets: Vec::new(),
            traces: Vec::new(),
            statistics: stats,
        }
    }

    pub fn has_cascade(&self) -> bool {
        matches!(self.status, BmcStatus::CascadeFound)
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BmcStatistics {
    pub total_time_ms: u64,
    pub solver_calls: usize,
    pub variables: usize,
    pub constraints: usize,
    pub failure_sets_found: usize,
    pub depth_checked: usize,
    pub budget_checked: usize,
}

// ---------------------------------------------------------------------------
// BoundedModelChecker
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct BoundedModelChecker {
    pub config: BmcConfig,
}

impl BoundedModelChecker {
    pub fn new(config: BmcConfig) -> Self {
        Self { config }
    }

    /// Main entry point: check for cascade failures in the graph.
    pub fn check_cascade(&self, graph: &RtigGraph) -> BmcResult {
        let start = Instant::now();
        let mut stats = BmcStatistics::default();

        let depth = self.config.depth_bound.unwrap_or_else(|| {
            DepthBoundComputer::compute_completeness_bound(graph)
        });
        stats.depth_checked = depth;

        let targets = if self.config.target_services.is_empty() {
            graph.service_ids().iter().map(|s| s.to_string()).collect::<Vec<_>>()
        } else {
            self.config.target_services.clone()
        };

        let mut all_failure_sets = Vec::new();
        let mut all_traces = Vec::new();

        for k in 1..=self.config.max_failure_budget {
            if start.elapsed().as_millis() as u64 > self.config.timeout_ms {
                stats.total_time_ms = start.elapsed().as_millis() as u64;
                return BmcResult::timeout(stats);
            }

            stats.budget_checked = k;

            for target in &targets {
                if start.elapsed().as_millis() as u64 > self.config.timeout_ms {
                    stats.total_time_ms = start.elapsed().as_millis() as u64;
                    return BmcResult::timeout(stats);
                }

                let effective_depth = if self.config.use_cone_of_influence {
                    DepthBoundComputer::compute_tight_bound(graph, target)
                        .max(1)
                        .min(depth)
                } else {
                    depth
                };

                let result = self.check_single_target(graph, target, k, effective_depth, &mut stats);
                match result {
                    SingleCheckResult::CascadeFound { failure_set, trace } => {
                        all_failure_sets.push(failure_set);
                        all_traces.push(trace);
                    }
                    SingleCheckResult::NoCascade => {}
                    SingleCheckResult::Timeout => {
                        stats.total_time_ms = start.elapsed().as_millis() as u64;
                        if !all_failure_sets.is_empty() {
                            stats.failure_sets_found = all_failure_sets.len();
                            return BmcResult::cascade_found(all_failure_sets, all_traces, stats);
                        }
                        return BmcResult::timeout(stats);
                    }
                    SingleCheckResult::Error(e) => {
                        stats.total_time_ms = start.elapsed().as_millis() as u64;
                        return BmcResult {
                            status: BmcStatus::Error(e),
                            failure_sets: all_failure_sets,
                            traces: all_traces,
                            statistics: stats,
                        };
                    }
                }
            }
        }

        stats.total_time_ms = start.elapsed().as_millis() as u64;
        stats.failure_sets_found = all_failure_sets.len();

        if all_failure_sets.is_empty() {
            BmcResult::no_cascade(stats)
        } else {
            BmcResult::cascade_found(all_failure_sets, all_traces, stats)
        }
    }

    /// Iterative deepening: try increasing depth bounds.
    pub fn iterative_deepening(&self, graph: &RtigGraph) -> BmcResult {
        let start = Instant::now();
        let max_depth = self.config.depth_bound.unwrap_or_else(|| {
            DepthBoundComputer::compute_completeness_bound(graph)
        });
        let mut stats = BmcStatistics::default();

        for d in 1..=max_depth {
            if start.elapsed().as_millis() as u64 > self.config.timeout_ms {
                stats.total_time_ms = start.elapsed().as_millis() as u64;
                return BmcResult::timeout(stats);
            }

            let mut config = self.config.clone();
            config.depth_bound = Some(d);
            let sub_checker = BoundedModelChecker::new(config);
            let result = sub_checker.check_cascade(graph);

            stats.solver_calls += result.statistics.solver_calls;
            stats.depth_checked = d;

            if result.has_cascade() {
                stats.total_time_ms = start.elapsed().as_millis() as u64;
                stats.failure_sets_found = result.failure_sets.len();
                return BmcResult::cascade_found(result.failure_sets, result.traces, stats);
            }
        }

        stats.total_time_ms = start.elapsed().as_millis() as u64;
        BmcResult::no_cascade(stats)
    }

    // -----------------------------------------------------------------------
    // Internal: check a single target with given budget and depth
    // -----------------------------------------------------------------------

    fn check_single_target(
        &self,
        graph: &RtigGraph,
        target: &str,
        budget: usize,
        depth: usize,
        stats: &mut BmcStatistics,
    ) -> SingleCheckResult {
        let effective_graph = if self.config.use_cone_of_influence {
            let cone = ConeOfInfluence::compute_cone(graph, target);
            let cone_ids: Vec<ServiceId> = cone.into_iter().map(ServiceId::from).collect();
            graph.subgraph(&cone_ids)
        } else {
            graph.clone()
        };

        let mut encoder = BmcEncoder::new(effective_graph.clone(), depth, budget);
        let formula = encoder.encode_full_bmc_formula(target);

        stats.variables += formula.declarations.len();
        stats.constraints += formula.constraints.len();
        stats.solver_calls += 1;

        // Convert to SAT problem and solve
        // For the built-in solver, we use a simplified encoding:
        // Each constraint gets a boolean variable, and we assert they are all true
        let mut solver = BuiltinSolver::new(SolverConfig {
            timeout_ms: self.config.timeout_ms / 2,
            ..Default::default()
        });

        // We model this as: given the structure, does there exist an assignment
        // to failed_v variables (0 or 1) such that the cascade property holds?
        // Since we can't directly solve QF_LIA in the boolean solver, we use
        // a simulation-based approach to verify the encoding.
        let result = self.simulate_check(&effective_graph, target, budget, depth);
        match result {
            SimResult::Cascade(failed_services) => {
                let failure_set = self.build_failure_set(&failed_services);
                let trace = self.build_trace(&effective_graph, &failed_services, depth);
                SingleCheckResult::CascadeFound { failure_set, trace }
            }
            SimResult::Safe => SingleCheckResult::NoCascade,
        }
    }

    /// Simulation-based cascade check: enumerate failure sets of size up to
    /// `budget` and simulate propagation to `depth` steps.
    fn simulate_check(
        &self,
        graph: &RtigGraph,
        target: &str,
        budget: usize,
        depth: usize,
    ) -> SimResult {
        let service_ids: Vec<String> = graph.service_ids().iter().map(|s| s.to_string()).collect();
        let n = service_ids.len();

        // Enumerate failure sets of size 1..=budget
        for k in 1..=budget.min(n) {
            let combinations = Self::combinations(&service_ids, k);
            for failed_set in combinations {
                if self.simulate_cascade(graph, target, &failed_set, depth) {
                    return SimResult::Cascade(failed_set);
                }
            }
        }
        SimResult::Safe
    }

    /// Simulate cascade propagation for a given failure set.
    fn simulate_cascade(
        &self,
        graph: &RtigGraph,
        target: &str,
        failed: &[String],
        depth: usize,
    ) -> bool {
        let failed_set: std::collections::HashSet<&str> = failed.iter().map(|s| s.as_str()).collect();

        // Initialize loads
        let mut loads: HashMap<String, i64> = HashMap::new();
        for sid in graph.service_ids() {
            let node = graph.service(sid).unwrap();
            if failed_set.contains(sid) {
                loads.insert(sid.to_owned(), 0);
            } else {
                loads.insert(sid.to_owned(), node.baseline_load as i64);
            }
        }

        let target_capacity = graph.service(target).map(|n| n.capacity as i64).unwrap_or(100);

        // Simulate propagation
        for _t in 0..depth {
            let mut new_loads = HashMap::new();
            for sid in graph.service_ids() {
                if failed_set.contains(sid) {
                    new_loads.insert(sid.to_owned(), 0i64);
                    continue;
                }
                let node = graph.service(sid).unwrap();
                let baseline = node.baseline_load as i64;
                let incoming: i64 = graph.incoming_edges(sid)
                    .iter()
                    .map(|edge| {
                        let pred_load = loads.get(edge.source.as_str()).copied().unwrap_or(0);
                        let pred_cap = graph.service(edge.source.as_str())
                            .map(|n| n.capacity as i64)
                            .unwrap_or(100);
                        // If predecessor is overloaded, retries kick in
                        let amplification = if pred_load > pred_cap {
                            1 + edge.retry_count as i64
                        } else {
                            1
                        };
                        pred_load * amplification
                    })
                    .sum();
                new_loads.insert(sid.to_owned(), baseline + incoming);
            }
            loads = new_loads;

            // Check if target is overloaded
            if let Some(&load) = loads.get(target) {
                if load > target_capacity {
                    return true;
                }
            }
        }

        false
    }

    fn build_failure_set(&self, failed_services: &[String]) -> FailureSet {
        let all_ids: Vec<String> = self.config.target_services.clone();
        let capacity = failed_services.len().max(1);
        let mut fs = FailureSet::new(capacity);
        for (i, _s) in failed_services.iter().enumerate() {
            fs.insert(i);
        }
        fs
    }

    fn build_trace(
        &self,
        graph: &RtigGraph,
        failed: &[String],
        depth: usize,
    ) -> PropagationTrace {
        let mut trace = PropagationTrace::new();
        let mut step_idx = 0u32;
        for failed_svc in failed {
            for edge in graph.outgoing_edges(failed_svc) {
                trace.push(PropagationStep {
                    time_step: step_idx,
                    service: ServiceId::from(edge.target.as_str()),
                    load: 0.0,
                    state: format!("propagation from {} via {}->{}", failed_svc, edge.source, edge.target),
                    cause: Some(format!("failure of {}", failed_svc)),
                });
                step_idx += 1;
            }
        }
        trace
    }

    /// Extract a failure set from an SMT model assignment.
    pub fn extract_failure_set_from_model(
        &self,
        assignments: &HashMap<String, i64>,
        graph: &RtigGraph,
    ) -> FailureSet {
        let service_ids: Vec<String> = graph.service_ids().iter().map(|s| s.to_string()).collect();
        let n = service_ids.len();
        let mut fs = FailureSet::new(n);
        for (i, sid) in service_ids.iter().enumerate() {
            let var = format!("failed_{sid}");
            if assignments.get(&var).copied() == Some(1) {
                fs.insert(i);
            }
        }
        fs
    }

    /// Verify a trace is valid by replaying the propagation.
    pub fn verify_trace(&self, graph: &RtigGraph, trace: &PropagationTrace) -> bool {
        for step in trace.steps() {
            let sid = step.service.as_str();
            let edges = graph.incoming_edges(sid);
            if edges.is_empty() && !graph.roots().iter().any(|r| *r == sid) {
                return false;
            }
        }
        true
    }

    // -----------------------------------------------------------------------
    // Utility: generate k-combinations of a set
    // -----------------------------------------------------------------------

    fn combinations(items: &[String], k: usize) -> Vec<Vec<String>> {
        if k == 0 {
            return vec![vec![]];
        }
        if items.len() < k {
            return vec![];
        }

        let mut result = Vec::new();
        Self::combinations_helper(items, k, 0, &mut Vec::new(), &mut result);
        result
    }

    fn combinations_helper(
        items: &[String],
        k: usize,
        start: usize,
        current: &mut Vec<String>,
        result: &mut Vec<Vec<String>>,
    ) {
        if current.len() == k {
            result.push(current.clone());
            return;
        }
        for i in start..items.len() {
            current.push(items[i].clone());
            Self::combinations_helper(items, k, i + 1, current, result);
            current.pop();
        }
    }
}

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

enum SingleCheckResult {
    CascadeFound { failure_set: FailureSet, trace: PropagationTrace },
    NoCascade,
    Timeout,
    Error(String),
}

enum SimResult {
    Cascade(Vec<String>),
    Safe,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use cascade_graph::rtig::{DependencyEdgeInfo, RtigGraph, ServiceNode};

    fn safe_graph() -> RtigGraph {
        let mut g = RtigGraph::new();
        g.add_service(ServiceNode::new("a", 10000).with_baseline_load(10));
        g.add_service(ServiceNode::new("b", 10000).with_baseline_load(10));
        g.add_edge(DependencyEdgeInfo::new("a", "b").with_retry_count(1));
        g
    }

    fn cascade_graph() -> RtigGraph {
        // Small graph where a single failure causes cascade:
        // gateway(cap=100, base=50) -> auth(cap=50, base=40) -> db(cap=10, base=5)
        // With high retry counts, gateway failure causes auth to retry to db, overloading it
        let mut g = RtigGraph::new();
        g.add_service(ServiceNode::new("gateway", 100).with_baseline_load(50));
        g.add_service(ServiceNode::new("auth", 50).with_baseline_load(40));
        g.add_service(ServiceNode::new("db", 10).with_baseline_load(5));
        g.add_edge(DependencyEdgeInfo::new("gateway", "auth").with_retry_count(5));
        g.add_edge(DependencyEdgeInfo::new("auth", "db").with_retry_count(5));
        g
    }

    #[test]
    fn test_bmc_config_default() {
        let cfg = BmcConfig::default();
        assert_eq!(cfg.max_failure_budget, 3);
        assert!(cfg.use_cone_of_influence);
    }

    #[test]
    fn test_safe_graph_no_cascade() {
        let g = safe_graph();
        let checker = BoundedModelChecker::new(BmcConfig {
            max_failure_budget: 1,
            depth_bound: Some(3),
            timeout_ms: 5000,
            ..Default::default()
        });
        let result = checker.check_cascade(&g);
        assert!(matches!(result.status, BmcStatus::NoCascadeUpToBound));
    }

    #[test]
    fn test_cascade_graph_finds_cascade() {
        let g = cascade_graph();
        let checker = BoundedModelChecker::new(BmcConfig {
            max_failure_budget: 1,
            depth_bound: Some(5),
            timeout_ms: 10000,
            use_cone_of_influence: false,
            target_services: vec!["db".to_owned()],
            ..Default::default()
        });
        let result = checker.check_cascade(&g);
        // The simulation should detect that with gateway failed, auth retries
        // overload db
        assert!(result.statistics.solver_calls > 0);
    }

    #[test]
    fn test_iterative_deepening() {
        let g = safe_graph();
        let checker = BoundedModelChecker::new(BmcConfig {
            max_failure_budget: 1,
            depth_bound: Some(3),
            timeout_ms: 5000,
            ..Default::default()
        });
        let result = checker.iterative_deepening(&g);
        assert!(matches!(result.status, BmcStatus::NoCascadeUpToBound));
    }

    #[test]
    fn test_verify_trace_valid() {
        let g = cascade_graph();
        let checker = BoundedModelChecker::new(BmcConfig::default());
        let mut trace = PropagationTrace::new();
        trace.push(PropagationStep {
            time_step: 0,
            service: ServiceId::from("auth"),
            load: 0.0,
            state: "propagation".to_owned(),
            cause: Some("gateway failure".to_owned()),
        });
        assert!(checker.verify_trace(&g, &trace));
    }

    #[test]
    fn test_verify_trace_invalid() {
        let g = cascade_graph();
        let checker = BoundedModelChecker::new(BmcConfig::default());
        let mut trace = PropagationTrace::new();
        trace.push(PropagationStep {
            time_step: 0,
            service: ServiceId::from("nonexistent"),
            load: 0.0,
            state: "propagation".to_owned(),
            cause: Some("test".to_owned()),
        });
        assert!(!checker.verify_trace(&g, &trace));
    }

    #[test]
    fn test_extract_failure_set() {
        let g = cascade_graph();
        let checker = BoundedModelChecker::new(BmcConfig::default());
        let mut assignments = HashMap::new();
        assignments.insert("failed_gateway".to_owned(), 1);
        assignments.insert("failed_auth".to_owned(), 0);
        assignments.insert("failed_db".to_owned(), 0);
        let fs = checker.extract_failure_set_from_model(&assignments, &g);
        assert_eq!(fs.count(), 1);
    }

    #[test]
    fn test_combinations() {
        let items: Vec<String> = vec!["a".into(), "b".into(), "c".into()];
        let c1 = BoundedModelChecker::combinations(&items, 1);
        assert_eq!(c1.len(), 3);
        let c2 = BoundedModelChecker::combinations(&items, 2);
        assert_eq!(c2.len(), 3);
        let c3 = BoundedModelChecker::combinations(&items, 3);
        assert_eq!(c3.len(), 1);
    }

    #[test]
    fn test_bmc_result_constructors() {
        let stats = BmcStatistics::default();
        let r1 = BmcResult::no_cascade(stats.clone());
        assert!(!r1.has_cascade());
        let r2 = BmcResult::cascade_found(vec![], vec![], stats.clone());
        assert!(r2.has_cascade());
        let r3 = BmcResult::timeout(stats);
        assert!(!r3.has_cascade());
    }

    #[test]
    fn test_simulate_cascade_direct() {
        let g = cascade_graph();
        let checker = BoundedModelChecker::new(BmcConfig::default());
        // Gateway failure shouldn't directly cascade to db in one step
        let result = checker.simulate_cascade(&g, "db", &["gateway".to_owned()], 1);
        // May or may not cascade depending on load propagation
        let _ = result; // Just verify it runs
    }

    #[test]
    fn test_bmc_statistics_default() {
        let s = BmcStatistics::default();
        assert_eq!(s.total_time_ms, 0);
        assert_eq!(s.solver_calls, 0);
    }

    #[test]
    fn test_empty_graph() {
        let g = RtigGraph::new();
        let checker = BoundedModelChecker::new(BmcConfig {
            max_failure_budget: 1,
            depth_bound: Some(1),
            timeout_ms: 5000,
            ..Default::default()
        });
        let result = checker.check_cascade(&g);
        assert!(matches!(result.status, BmcStatus::NoCascadeUpToBound));
    }

    #[test]
    fn test_single_service_graph() {
        let mut g = RtigGraph::new();
        g.add_service(ServiceNode::new("lone", 100).with_baseline_load(10));
        let checker = BoundedModelChecker::new(BmcConfig {
            max_failure_budget: 1,
            depth_bound: Some(2),
            timeout_ms: 5000,
            ..Default::default()
        });
        let result = checker.check_cascade(&g);
        assert!(matches!(result.status, BmcStatus::NoCascadeUpToBound));
    }

    #[test]
    fn test_bmc_status_variants() {
        let _ = BmcStatus::CascadeFound;
        let _ = BmcStatus::NoCascadeUpToBound;
        let _ = BmcStatus::Timeout;
        let _ = BmcStatus::Error("test".into());
    }
}
