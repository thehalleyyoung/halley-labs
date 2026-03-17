//! Path analysis for RTIG graphs: enumeration, amplification, timeout chains, fan-in.

use crate::rtig::RtigGraph;
use cascade_types::service::ServiceId;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// A path in the graph that exceeds an amplification threshold.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskyPath {
    pub services: Vec<ServiceId>,
    pub total_amplification: f64,
    pub worst_case_latency_ms: u64,
    pub depth: usize,
}

/// Result of analysing a timeout chain along a path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeoutChainResult {
    pub chain: Vec<ServiceId>,
    pub total_timeout_ms: u64,
    pub chain_exceeds_budget: bool,
}

/// Fan-in statistics for a single service.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FanInAnalysis {
    pub service: ServiceId,
    pub fan_in: usize,
    pub total_retry_load: f64,
}

// ---------------------------------------------------------------------------
// PathEnumerator – DFS-based enumeration
// ---------------------------------------------------------------------------

pub struct PathEnumerator;

impl PathEnumerator {
    /// Enumerate all simple paths from `from` to `to`.
    pub fn enumerate_paths(graph: &RtigGraph, from: &ServiceId, to: &ServiceId) -> Vec<Vec<ServiceId>> {
        let mut results = Vec::new();
        let mut path = vec![from.clone()];
        let mut visited = HashSet::new();
        visited.insert(from.clone());
        Self::dfs(graph, from, to, &mut path, &mut visited, &mut results, 100);
        results
    }

    /// Enumerate paths up to `max_depth` hops.
    pub fn enumerate_bounded(
        graph: &RtigGraph,
        from: &ServiceId,
        to: &ServiceId,
        max_depth: usize,
    ) -> Vec<Vec<ServiceId>> {
        let mut results = Vec::new();
        let mut path = vec![from.clone()];
        let mut visited = HashSet::new();
        visited.insert(from.clone());
        Self::dfs(graph, from, to, &mut path, &mut visited, &mut results, max_depth);
        results
    }

    fn dfs(
        graph: &RtigGraph,
        current: &ServiceId,
        target: &ServiceId,
        path: &mut Vec<ServiceId>,
        visited: &mut HashSet<ServiceId>,
        results: &mut Vec<Vec<ServiceId>>,
        max_depth: usize,
    ) {
        if current == target && path.len() > 1 {
            results.push(path.clone());
            return;
        }
        if path.len() > max_depth {
            return;
        }
        for succ in graph.get_successors(current) {
            if !visited.contains(&succ) {
                visited.insert(succ.clone());
                path.push(succ.clone());
                Self::dfs(graph, &succ, target, path, visited, results, max_depth);
                path.pop();
                visited.remove(&succ);
            }
        }
    }

    /// Find roots (services with no predecessors).
    pub fn find_roots(graph: &RtigGraph) -> Vec<ServiceId> {
        graph
            .services()
            .into_iter()
            .filter(|s| graph.get_predecessors(s).is_empty())
            .collect()
    }

    /// Find leaves (services with no successors).
    pub fn find_leaves(graph: &RtigGraph) -> Vec<ServiceId> {
        graph
            .services()
            .into_iter()
            .filter(|s| graph.get_successors(s).is_empty())
            .collect()
    }
}

// ---------------------------------------------------------------------------
// PathComposer – amplification & timeout budget
// ---------------------------------------------------------------------------

pub struct PathComposer;

impl PathComposer {
    /// Compute the multiplicative amplification factor along a path.
    pub fn amplification_factor(graph: &RtigGraph, path: &[ServiceId]) -> f64 {
        let mut amp = 1.0f64;
        for window in path.windows(2) {
            if let Some(policy) = graph.get_edge_policy(&window[0], &window[1]) {
                amp *= policy.amplification_factor() as f64;
            }
        }
        amp
    }

    /// Compute the worst-case end-to-end timeout budget along a path.
    pub fn timeout_budget(graph: &RtigGraph, path: &[ServiceId]) -> u64 {
        let mut total = 0u64;
        for window in path.windows(2) {
            if let Some(policy) = graph.get_edge_policy(&window[0], &window[1]) {
                total = total.saturating_add(policy.worst_case_latency_ms());
            }
        }
        total
    }

    /// Return per-hop amplification factors.
    pub fn per_hop_amplification(graph: &RtigGraph, path: &[ServiceId]) -> Vec<f64> {
        path.windows(2)
            .map(|w| {
                graph
                    .get_edge_policy(&w[0], &w[1])
                    .map(|p| p.amplification_factor() as f64)
                    .unwrap_or(1.0)
            })
            .collect()
    }

    /// Return per-hop worst-case latencies.
    pub fn per_hop_latency(graph: &RtigGraph, path: &[ServiceId]) -> Vec<u64> {
        path.windows(2)
            .map(|w| {
                graph
                    .get_edge_policy(&w[0], &w[1])
                    .map(|p| p.worst_case_latency_ms())
                    .unwrap_or(0)
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// CascadePathAnalyzer – high-level analysis
// ---------------------------------------------------------------------------

pub struct CascadePathAnalyzer;

impl CascadePathAnalyzer {
    /// Find all root-to-leaf paths whose amplification exceeds `threshold`.
    pub fn find_all_risky_paths(graph: &RtigGraph, threshold: f64) -> Vec<RiskyPath> {
        let roots = PathEnumerator::find_roots(graph);
        let leaves = PathEnumerator::find_leaves(graph);
        let mut risky = Vec::new();

        for root in &roots {
            for leaf in &leaves {
                let paths = PathEnumerator::enumerate_paths(graph, root, leaf);
                for path in paths {
                    let amp = PathComposer::amplification_factor(graph, &path);
                    if amp >= threshold {
                        let latency = PathComposer::timeout_budget(graph, &path);
                        risky.push(RiskyPath {
                            depth: path.len() - 1,
                            total_amplification: amp,
                            worst_case_latency_ms: latency,
                            services: path,
                        });
                    }
                }
            }
        }
        risky.sort_by(|a, b| b.total_amplification.partial_cmp(&a.total_amplification).unwrap_or(std::cmp::Ordering::Equal));
        risky
    }

    /// Compute worst-case amplification across all root-to-leaf paths.
    pub fn compute_worst_case_amplification(graph: &RtigGraph) -> f64 {
        let roots = PathEnumerator::find_roots(graph);
        let leaves = PathEnumerator::find_leaves(graph);
        let mut worst = 1.0f64;
        for root in &roots {
            for leaf in &leaves {
                for path in PathEnumerator::enumerate_paths(graph, root, leaf) {
                    let amp = PathComposer::amplification_factor(graph, &path);
                    if amp > worst {
                        worst = amp;
                    }
                }
            }
        }
        worst
    }

    /// Compute timeout chains for all root-to-leaf paths, flagging those that
    /// exceed `budget_ms`.
    pub fn compute_timeout_chain(graph: &RtigGraph, budget_ms: u64) -> Vec<TimeoutChainResult> {
        let roots = PathEnumerator::find_roots(graph);
        let leaves = PathEnumerator::find_leaves(graph);
        let mut results = Vec::new();
        for root in &roots {
            for leaf in &leaves {
                for path in PathEnumerator::enumerate_paths(graph, root, leaf) {
                    let total = PathComposer::timeout_budget(graph, &path);
                    results.push(TimeoutChainResult {
                        chain: path,
                        total_timeout_ms: total,
                        chain_exceeds_budget: total > budget_ms,
                    });
                }
            }
        }
        results
    }

    /// Compute fan-in analysis for every service.
    pub fn fan_in_analysis(graph: &RtigGraph) -> Vec<FanInAnalysis> {
        graph
            .services()
            .iter()
            .map(|svc| {
                let preds = graph.get_predecessors(svc);
                let fan_in = preds.len();
                let total_retry_load: f64 = preds
                    .iter()
                    .map(|pred| {
                        graph
                            .get_edge_policy(pred, svc)
                            .map(|p| p.amplification_factor() as f64)
                            .unwrap_or(1.0)
                    })
                    .sum();
                FanInAnalysis {
                    service: svc.clone(),
                    fan_in,
                    total_retry_load,
                }
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rtig::{build_chain, build_diamond, RtigGraph};
    use cascade_types::policy::{ResiliencePolicy, RetryPolicy};
    use cascade_types::service::ServiceId;

    fn sid(s: &str) -> ServiceId {
        ServiceId::new(s)
    }

    fn chain_with_retries() -> RtigGraph {
        // A -> B (2 retries) -> C (3 retries)
        let mut g = RtigGraph::new();
        for n in &["A", "B", "C"] {
            g.add_service(sid(n));
        }
        let p_ab = ResiliencePolicy::empty().with_retry(RetryPolicy::new(2));
        let p_bc = ResiliencePolicy::empty().with_retry(RetryPolicy::new(3));
        g.add_dependency(&sid("A"), &sid("B"), p_ab);
        g.add_dependency(&sid("B"), &sid("C"), p_bc);
        g
    }

    #[test]
    fn test_enumerate_simple_chain() {
        let g = build_chain(&["A", "B", "C"], 1);
        let paths = PathEnumerator::enumerate_paths(&g, &sid("A"), &sid("C"));
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0].len(), 3);
    }

    #[test]
    fn test_enumerate_diamond() {
        let g = build_diamond(1);
        let paths = PathEnumerator::enumerate_paths(&g, &sid("A"), &sid("D"));
        assert_eq!(paths.len(), 2);
    }

    #[test]
    fn test_enumerate_bounded() {
        let g = build_chain(&["A", "B", "C", "D", "E"], 1);
        let paths = PathEnumerator::enumerate_bounded(&g, &sid("A"), &sid("E"), 2);
        assert!(paths.is_empty()); // 4 hops needed, max_depth=2
    }

    #[test]
    fn test_find_roots_and_leaves() {
        let g = build_chain(&["A", "B", "C"], 1);
        let roots = PathEnumerator::find_roots(&g);
        let leaves = PathEnumerator::find_leaves(&g);
        assert_eq!(roots.len(), 1);
        assert_eq!(leaves.len(), 1);
        assert_eq!(roots[0], sid("A"));
        assert_eq!(leaves[0], sid("C"));
    }

    #[test]
    fn test_amplification_factor() {
        let g = chain_with_retries();
        let path = vec![sid("A"), sid("B"), sid("C")];
        let amp = PathComposer::amplification_factor(&g, &path);
        // (1+2) * (1+3) = 3 * 4 = 12
        assert!((amp - 12.0).abs() < 1e-9);
    }

    #[test]
    fn test_per_hop_amplification() {
        let g = chain_with_retries();
        let path = vec![sid("A"), sid("B"), sid("C")];
        let hops = PathComposer::per_hop_amplification(&g, &path);
        assert_eq!(hops.len(), 2);
        assert!((hops[0] - 3.0).abs() < 1e-9);
        assert!((hops[1] - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_find_risky_paths() {
        let g = chain_with_retries();
        let risky = CascadePathAnalyzer::find_all_risky_paths(&g, 5.0);
        assert_eq!(risky.len(), 1);
        assert!((risky[0].total_amplification - 12.0).abs() < 1e-9);
    }

    #[test]
    fn test_no_risky_paths_below_threshold() {
        let g = build_chain(&["A", "B", "C"], 0); // 0 retries -> amp factor 1
        let risky = CascadePathAnalyzer::find_all_risky_paths(&g, 2.0);
        assert!(risky.is_empty());
    }

    #[test]
    fn test_worst_case_amplification() {
        let g = chain_with_retries();
        let worst = CascadePathAnalyzer::compute_worst_case_amplification(&g);
        assert!((worst - 12.0).abs() < 1e-9);
    }

    #[test]
    fn test_timeout_chain() {
        let g = chain_with_retries();
        let chains = CascadePathAnalyzer::compute_timeout_chain(&g, 1);
        assert_eq!(chains.len(), 1);
        assert!(chains[0].chain_exceeds_budget || chains[0].total_timeout_ms <= 1);
    }

    #[test]
    fn test_fan_in_analysis() {
        let g = chain_with_retries();
        let analysis = CascadePathAnalyzer::fan_in_analysis(&g);
        assert_eq!(analysis.len(), 3);
        let a = analysis.iter().find(|f| f.service == sid("A")).unwrap();
        assert_eq!(a.fan_in, 0);
        let c = analysis.iter().find(|f| f.service == sid("C")).unwrap();
        assert_eq!(c.fan_in, 1);
    }

    #[test]
    fn test_diamond_fan_in() {
        let g = build_diamond(1);
        let analysis = CascadePathAnalyzer::fan_in_analysis(&g);
        let exit_fi = analysis.iter().find(|f| f.service == sid("D")).unwrap();
        assert_eq!(exit_fi.fan_in, 2);
    }

    #[test]
    fn test_empty_graph() {
        let g = RtigGraph::new();
        let risky = CascadePathAnalyzer::find_all_risky_paths(&g, 1.0);
        assert!(risky.is_empty());
    }
}
