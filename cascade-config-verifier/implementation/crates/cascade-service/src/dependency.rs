//! Service dependency modelling – wraps [`cascade_graph::RtigGraph`] with
//! higher-level analysis helpers for criticality, impact, cycles, and
//! software-engineering quality metrics.

use std::collections::{HashSet, VecDeque};

use cascade_graph::rtig::RtigGraph;
use serde::{Deserialize, Serialize};

// ── DependencyGraph wrapper ─────────────────────────────────────────

/// A higher-level wrapper around [`RtigGraph`] that exposes
/// dependency-centric queries.
#[derive(Debug, Clone)]
pub struct DependencyGraph {
    inner: RtigGraph,
}

impl DependencyGraph {
    pub fn new(graph: RtigGraph) -> Self {
        Self { inner: graph }
    }

    pub fn inner(&self) -> &RtigGraph {
        &self.inner
    }

    pub fn service_count(&self) -> usize {
        self.inner.service_count()
    }

    pub fn edge_count(&self) -> usize {
        self.inner.edge_count()
    }

    pub fn service_ids(&self) -> Vec<String> {
        self.inner.service_ids().iter().map(|s| s.to_string()).collect()
    }

    pub fn direct_dependencies(&self, service: &str) -> Vec<String> {
        self.inner.successors(service).iter().map(|s| s.to_string()).collect()
    }

    pub fn direct_dependents(&self, service: &str) -> Vec<String> {
        self.inner.predecessors(service).iter().map(|s| s.to_string()).collect()
    }

    pub fn transitive_dependencies(&self, service: &str) -> HashSet<String> {
        let mut result = self.inner.forward_reachable(service);
        result.remove(service);
        result
    }

    pub fn transitive_dependents(&self, service: &str) -> HashSet<String> {
        let mut result = self.inner.reverse_reachable(service);
        result.remove(service);
        result
    }

    pub fn dependency_depth(&self, service: &str) -> usize {
        self.inner.longest_path_to(service)
    }

    pub fn is_dag(&self) -> bool {
        self.inner.is_dag()
    }

    pub fn roots(&self) -> Vec<String> {
        self.inner.roots().iter().map(|s| s.to_string()).collect()
    }

    pub fn leaves(&self) -> Vec<String> {
        self.inner.leaves().iter().map(|s| s.to_string()).collect()
    }
}

// ── CriticalDependency ──────────────────────────────────────────────

/// A service that many others depend on, representing high risk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriticalDependency {
    pub service: String,
    pub dependents_count: usize,
    pub max_amplification: f64,
    pub risk_score: f64,
}

// ── DependencyCycle ─────────────────────────────────────────────────

/// A circular dependency path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyCycle {
    pub services: Vec<String>,
    pub edges: Vec<(String, String)>,
}

// ── DependencyAnalyzer ──────────────────────────────────────────────

/// Static analysis of dependency structure.
pub struct DependencyAnalyzer;

impl DependencyAnalyzer {
    /// Identify services that are critical because many others depend on them.
    pub fn find_critical_dependencies(graph: &DependencyGraph) -> Vec<CriticalDependency> {
        let ids = graph.service_ids();
        let mut results: Vec<CriticalDependency> = Vec::new();

        for id in &ids {
            let dependents = graph.transitive_dependents(id);
            let dep_count = dependents.len();
            if dep_count == 0 {
                continue;
            }

            let max_amp = Self::compute_max_amplification(graph.inner(), id);
            let svc_count = graph.service_count().max(1) as f64;
            let risk_score = (dep_count as f64 / svc_count) * max_amp.ln_1p();

            results.push(CriticalDependency {
                service: id.clone(),
                dependents_count: dep_count,
                max_amplification: max_amp,
                risk_score,
            });
        }

        results.sort_by(|a, b| {
            b.risk_score
                .partial_cmp(&a.risk_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results
    }

    fn compute_max_amplification(graph: &RtigGraph, service: &str) -> f64 {
        let incoming = graph.incoming_edges(service);
        if incoming.is_empty() {
            return 1.0;
        }
        incoming
            .iter()
            .map(|e| e.amplification_factor_f64())
            .fold(1.0_f64, f64::max)
    }

    /// Services that, if removed, would disconnect the graph.
    pub fn find_single_points_of_failure(graph: &DependencyGraph) -> Vec<String> {
        let ids = graph.service_ids();
        let total = graph.service_count();
        if total <= 1 {
            return vec![];
        }

        let mut spofs = Vec::new();
        for id in &ids {
            let keep: Vec<cascade_types::service::ServiceId> = ids
                .iter()
                .filter(|s| s.as_str() != id.as_str())
                .map(|s| cascade_types::service::ServiceId::new(s.clone()))
                .collect();
            let sub = graph.inner().subgraph(&keep);
            // If removing the node increases the number of weakly-connected
            // components (approximated by checking root count).
            let orig_components = count_weak_components(graph.inner());
            let new_components = count_weak_components(&sub);
            if new_components > orig_components {
                spofs.push(id.clone());
            }
        }
        spofs
    }

    /// Maximum dependency chain depth from any root to `service`.
    pub fn compute_dependency_depth(graph: &DependencyGraph, service: &str) -> usize {
        graph.dependency_depth(service)
    }

    /// Detect all circular dependency paths.
    pub fn find_circular_dependencies(graph: &DependencyGraph) -> Vec<DependencyCycle> {
        let g = graph.inner();
        if g.is_dag() {
            return vec![];
        }

        let ids: Vec<String> = g.service_ids().iter().map(|s| s.to_string()).collect();
        let mut all_cycles: Vec<DependencyCycle> = Vec::new();
        let _global_visited: HashSet<String> = HashSet::new();

        for start in &ids {
            let mut visited: HashSet<String> = HashSet::new();
            let mut path: Vec<String> = vec![start.clone()];
            visited.insert(start.clone());
            Self::cycle_dfs(g, start, start, &mut visited, &mut path, &mut all_cycles);
        }

        // Canonicalise and deduplicate
        let mut seen: HashSet<Vec<String>> = HashSet::new();
        let mut unique = Vec::new();
        for cycle in all_cycles {
            let canon = canonical_rotation(&cycle.services);
            if seen.insert(canon) {
                unique.push(cycle);
            }
        }
        unique
    }

    fn cycle_dfs(
        graph: &RtigGraph,
        start: &str,
        cur: &str,
        visited: &mut HashSet<String>,
        path: &mut Vec<String>,
        results: &mut Vec<DependencyCycle>,
    ) {
        for next in graph.successors(cur) {
            if next == start && path.len() > 1 {
                let services = path.clone();
                let edges: Vec<(String, String)> = services
                    .windows(2)
                    .map(|w| (w[0].clone(), w[1].clone()))
                    .chain(std::iter::once((
                        services.last().unwrap().clone(),
                        services.first().unwrap().clone(),
                    )))
                    .collect();
                results.push(DependencyCycle { services, edges });
            } else if !visited.contains(next) {
                visited.insert(next.to_string());
                path.push(next.to_string());
                Self::cycle_dfs(graph, start, next, visited, path, results);
                path.pop();
                visited.remove(next);
            }
        }
    }

    /// Build a condensation (DAG of strongly-connected components).
    /// Each SCC is collapsed into a single representative node.
    pub fn condensation(graph: &DependencyGraph) -> DependencyGraph {
        let g = graph.inner();
        let scc_groups = cascade_graph::tarjan_scc(g);
        let mut condensed = RtigGraph::new();
        let mut svc_to_scc: std::collections::HashMap<String, String> = std::collections::HashMap::new();

        for (i, group) in scc_groups.iter().enumerate() {
            let rep = format!("scc-{}", i);
            let cap: u32 = group.iter().filter_map(|s| g.service(s.as_str())).map(|n| n.capacity).sum();
            condensed.add_service_node(&cascade_graph::rtig::ServiceNode::new(&rep, cap));
            for s in group {
                svc_to_scc.insert(s.to_string(), rep.clone());
            }
        }

        let mut seen_edges: std::collections::HashSet<(String, String)> = std::collections::HashSet::new();
        for edge in g.edges() {
            let src_scc = svc_to_scc.get(edge.source.as_str()).cloned().unwrap_or_default();
            let tgt_scc = svc_to_scc.get(edge.target.as_str()).cloned().unwrap_or_default();
            if src_scc != tgt_scc && seen_edges.insert((src_scc.clone(), tgt_scc.clone())) {
                condensed.add_edge(
                    cascade_graph::rtig::DependencyEdgeInfo::new(&src_scc, &tgt_scc)
                        .with_retry_count(edge.retry_count),
                );
            }
        }

        DependencyGraph::new(condensed)
    }
}

// ── ImpactAssessment ────────────────────────────────────────────────

/// Result of a failure-impact analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpactAssessment {
    pub affected_services: Vec<String>,
    pub total_load_increase: f64,
    pub cascade_probability: f64,
}

/// Impact analysis for service failures.
pub struct DependencyImpact;

impl DependencyImpact {
    /// Compute the impact of a set of services failing simultaneously.
    pub fn compute_failure_impact(
        graph: &DependencyGraph,
        failed: &[String],
    ) -> ImpactAssessment {
        let g = graph.inner();
        let mut affected: HashSet<String> = HashSet::new();

        for f in failed {
            let reach = g.reverse_reachable(f);
            affected.extend(reach);
        }
        // Remove the failed set from affected
        for f in failed {
            affected.remove(f.as_str());
        }

        let total_load_increase = Self::estimate_load_increase(g, failed, &affected);
        let cascade_prob = Self::estimate_cascade_probability(g, failed, &affected);

        ImpactAssessment {
            affected_services: affected.into_iter().collect(),
            total_load_increase,
            cascade_probability: cascade_prob,
        }
    }

    fn estimate_load_increase(
        graph: &RtigGraph,
        failed: &[String],
        affected: &HashSet<String>,
    ) -> f64 {
        let failed_set: HashSet<&str> = failed.iter().map(|s| s.as_str()).collect();
        let mut total = 0.0_f64;

        for svc in affected {
            let incoming = graph.incoming_edges(svc);
            for edge in &incoming {
                if failed_set.contains(edge.source.as_str()) {
                    // If a dependency is failed, retries from that edge add load
                    total += edge.retry_count as f64 * edge.amplification_factor_f64();
                }
            }
        }
        total
    }

    fn estimate_cascade_probability(
        graph: &RtigGraph,
        failed: &[String],
        affected: &HashSet<String>,
    ) -> f64 {
        if affected.is_empty() {
            return 0.0;
        }

        let mut overloaded_count = 0usize;
        for svc in affected {
            if let Some(node) = graph.service(svc) {
                let incoming = graph.incoming_edges(svc);
                let extra_load: f64 = incoming
                    .iter()
                    .filter(|e| {
                        failed.iter().any(|f| f.as_str() == e.source.as_str())
                    })
                    .map(|e| e.retry_count as f64 * e.amplification_factor_f64())
                    .sum();
                let total_load = node.baseline_load as f64 + extra_load;
                if node.capacity > 0 && total_load > node.capacity as f64 {
                    overloaded_count += 1;
                }
            }
        }
        if affected.is_empty() {
            0.0
        } else {
            overloaded_count as f64 / affected.len() as f64
        }
    }

    /// Compute the N services whose failure would have the largest impact.
    pub fn top_impact_services(graph: &DependencyGraph, n: usize) -> Vec<(String, f64)> {
        let ids = graph.service_ids();
        let mut scores: Vec<(String, f64)> = ids
            .iter()
            .map(|id| {
                let impact = Self::compute_failure_impact(graph, &[id.clone()]);
                let score = impact.affected_services.len() as f64
                    * (1.0 + impact.cascade_probability);
                (id.clone(), score)
            })
            .collect();
        scores.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        scores.truncate(n);
        scores
    }
}

// ── DependencyMetrics ───────────────────────────────────────────────

/// Software-engineering-style metrics for dependency graphs.
pub struct DependencyMetrics;

impl DependencyMetrics {
    /// Coupling score – ratio of actual edges to maximum possible edges.
    /// Higher means tighter coupling.
    pub fn coupling_score(graph: &DependencyGraph) -> f64 {
        let n = graph.service_count() as f64;
        if n <= 1.0 {
            return 0.0;
        }
        let max_edges = n * (n - 1.0);
        graph.edge_count() as f64 / max_edges
    }

    /// Cohesion score – fraction of services reachable from the primary root.
    /// Higher means better cohesion.
    pub fn cohesion_score(graph: &DependencyGraph) -> f64 {
        let n = graph.service_count();
        if n == 0 {
            return 0.0;
        }
        let roots = graph.roots();
        if roots.is_empty() {
            return 0.0;
        }
        let reachable = graph.inner().forward_reachable(&roots[0]);
        reachable.len() as f64 / n as f64
    }

    /// Robert C. Martin's instability index:
    ///   I = Ce / (Ca + Ce)
    /// where Ca = afferent coupling (incoming), Ce = efferent coupling (outgoing).
    pub fn instability_index(graph: &DependencyGraph, service: &str) -> f64 {
        let ca = graph.inner().fan_in(service) as f64;
        let ce = graph.inner().fan_out(service) as f64;
        if ca + ce == 0.0 {
            return 0.5; // undefined → neutral
        }
        ce / (ca + ce)
    }

    /// Abstractness score – ratio of leaf services (no outgoing deps) to total.
    pub fn abstractness(graph: &DependencyGraph) -> f64 {
        let n = graph.service_count();
        if n == 0 {
            return 0.0;
        }
        let leaves = graph.leaves().len();
        leaves as f64 / n as f64
    }

    /// Distance from the main sequence: |A + I - 1| where A=abstractness, I=avg instability.
    pub fn distance_from_main_sequence(graph: &DependencyGraph) -> f64 {
        let a = Self::abstractness(graph);
        let ids = graph.service_ids();
        if ids.is_empty() {
            return 0.0;
        }
        let avg_i: f64 = ids
            .iter()
            .map(|id| Self::instability_index(graph, id))
            .sum::<f64>()
            / ids.len() as f64;
        (a + avg_i - 1.0).abs()
    }

    /// Depth of the dependency tree (longest chain from root to leaf).
    pub fn max_depth(graph: &DependencyGraph) -> usize {
        let leaves = graph.leaves();
        leaves
            .iter()
            .map(|l| graph.dependency_depth(l))
            .max()
            .unwrap_or(0)
    }

    /// Average fan-in across all services.
    pub fn avg_fan_in(graph: &DependencyGraph) -> f64 {
        let n = graph.service_count();
        if n == 0 {
            return 0.0;
        }
        let total: usize = graph
            .service_ids()
            .iter()
            .map(|id| graph.inner().fan_in(id))
            .sum();
        total as f64 / n as f64
    }

    /// Average fan-out across all services.
    pub fn avg_fan_out(graph: &DependencyGraph) -> f64 {
        let n = graph.service_count();
        if n == 0 {
            return 0.0;
        }
        let total: usize = graph
            .service_ids()
            .iter()
            .map(|id| graph.inner().fan_out(id))
            .sum();
        total as f64 / n as f64
    }
}

// ── helpers ─────────────────────────────────────────────────────────

fn count_weak_components(graph: &RtigGraph) -> usize {
    let ids: Vec<String> = graph.service_ids().iter().map(|s| s.to_string()).collect();
    let mut visited: HashSet<String> = HashSet::new();
    let mut count = 0usize;

    for id in &ids {
        if visited.contains(id.as_str()) {
            continue;
        }
        count += 1;
        let mut queue = VecDeque::new();
        queue.push_back(id.clone());
        visited.insert(id.clone());
        while let Some(cur) = queue.pop_front() {
            for next in graph
                .successors(&cur)
                .iter()
                .chain(graph.predecessors(&cur).iter())
            {
                if visited.insert(next.to_string()) {
                    queue.push_back(next.to_string());
                }
            }
        }
    }
    count
}

fn canonical_rotation(ids: &[String]) -> Vec<String> {
    if ids.is_empty() {
        return vec![];
    }
    let n = ids.len();
    let mut best = ids.to_vec();
    for i in 1..n {
        let rotated: Vec<String> = ids[i..].iter().chain(ids[..i].iter()).cloned().collect();
        if rotated < best {
            best = rotated;
        }
    }
    best
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use cascade_graph::rtig::{RtigGraphBuilder, ServiceNode, DependencyEdgeInfo};
    use cascade_types::topology::DependencyType;

    fn chain_graph() -> DependencyGraph {
        let g = RtigGraphBuilder::new()
            .add_service(ServiceNode::new("a", 100))
            .add_service(ServiceNode::new("b", 100))
            .add_service(ServiceNode::new("c", 100))
            .add_edge(DependencyEdgeInfo::new("a", "b").with_retry_count(2))
            .add_edge(DependencyEdgeInfo::new("b", "c").with_retry_count(3))
            .build();
        DependencyGraph::new(g)
    }

    fn diamond_graph() -> DependencyGraph {
        let g = RtigGraphBuilder::new()
            .add_service(ServiceNode::new("root", 200))
            .add_service(ServiceNode::new("left", 100))
            .add_service(ServiceNode::new("right", 100))
            .add_service(ServiceNode::new("sink", 50))
            .add_edge(DependencyEdgeInfo::new("root", "left"))
            .add_edge(DependencyEdgeInfo::new("root", "right"))
            .add_edge(DependencyEdgeInfo::new("left", "sink"))
            .add_edge(DependencyEdgeInfo::new("right", "sink"))
            .build();
        DependencyGraph::new(g)
    }

    fn cyclic_graph() -> DependencyGraph {
        let g = RtigGraphBuilder::new()
            .add_service(ServiceNode::new("x", 100))
            .add_service(ServiceNode::new("y", 100))
            .add_service(ServiceNode::new("z", 100))
            .add_edge(DependencyEdgeInfo::new("x", "y"))
            .add_edge(DependencyEdgeInfo::new("y", "z"))
            .add_edge(DependencyEdgeInfo::new("z", "x"))
            .build();
        DependencyGraph::new(g)
    }

    // ── DependencyGraph ────

    #[test]
    fn basic_queries() {
        let dg = chain_graph();
        assert_eq!(dg.service_count(), 3);
        assert_eq!(dg.edge_count(), 2);
        assert_eq!(dg.direct_dependencies("a"), vec!["b"]);
        assert_eq!(dg.direct_dependents("c"), vec!["b"]);
    }

    #[test]
    fn transitive_deps() {
        let dg = chain_graph();
        let td = dg.transitive_dependencies("a");
        assert!(td.contains("b"));
        assert!(td.contains("c"));
        assert!(!td.contains("a"));
    }

    #[test]
    fn roots_and_leaves() {
        let dg = chain_graph();
        assert_eq!(dg.roots(), vec!["a"]);
        assert_eq!(dg.leaves(), vec!["c"]);
    }

    #[test]
    fn is_dag_true() {
        assert!(chain_graph().is_dag());
    }

    #[test]
    fn is_dag_false() {
        assert!(!cyclic_graph().is_dag());
    }

    // ── DependencyAnalyzer ─

    #[test]
    fn critical_deps_chain() {
        let dg = chain_graph();
        let crit = DependencyAnalyzer::find_critical_dependencies(&dg);
        assert!(!crit.is_empty());
        // "b" has one transitive dependent ("a"), "c" has two ("a","b")
        assert!(crit.iter().any(|c| c.service == "b" || c.service == "c"));
    }

    #[test]
    fn spof_chain() {
        let dg = chain_graph();
        let spofs = DependencyAnalyzer::find_single_points_of_failure(&dg);
        // Removing "b" disconnects a from c
        assert!(spofs.contains(&"b".to_string()));
    }

    #[test]
    fn dependency_depth() {
        let dg = chain_graph();
        assert_eq!(DependencyAnalyzer::compute_dependency_depth(&dg, "a"), 0);
        assert_eq!(DependencyAnalyzer::compute_dependency_depth(&dg, "b"), 1);
        assert_eq!(DependencyAnalyzer::compute_dependency_depth(&dg, "c"), 2);
    }

    #[test]
    fn circular_deps_none() {
        let dg = chain_graph();
        let cycles = DependencyAnalyzer::find_circular_dependencies(&dg);
        assert!(cycles.is_empty());
    }

    #[test]
    fn circular_deps_found() {
        let dg = cyclic_graph();
        let cycles = DependencyAnalyzer::find_circular_dependencies(&dg);
        assert_eq!(cycles.len(), 1);
        assert_eq!(cycles[0].services.len(), 3);
    }

    // ── DependencyImpact ───

    #[test]
    fn failure_impact_leaf() {
        let dg = chain_graph();
        let impact = DependencyImpact::compute_failure_impact(&dg, &["c".to_string()]);
        // a and b should be affected
        assert!(!impact.affected_services.is_empty());
    }

    #[test]
    fn failure_impact_root() {
        let dg = chain_graph();
        let impact = DependencyImpact::compute_failure_impact(&dg, &["a".to_string()]);
        // No predecessors, so no other services affected
        assert!(impact.affected_services.is_empty());
    }

    #[test]
    fn top_impact() {
        let dg = diamond_graph();
        let top = DependencyImpact::top_impact_services(&dg, 2);
        assert!(!top.is_empty());
        // sink should have highest impact (both left and right depend on it transitively)
        assert_eq!(top[0].0, "sink");
    }

    // ── DependencyMetrics ──

    #[test]
    fn coupling_score() {
        let dg = chain_graph();
        let cs = DependencyMetrics::coupling_score(&dg);
        assert!(cs > 0.0 && cs < 1.0);
    }

    #[test]
    fn cohesion_score() {
        let dg = chain_graph();
        let ch = DependencyMetrics::cohesion_score(&dg);
        assert!((ch - 1.0).abs() < 0.01);
    }

    #[test]
    fn instability_index() {
        let dg = chain_graph();
        let root_i = DependencyMetrics::instability_index(&dg, "a");
        let leaf_i = DependencyMetrics::instability_index(&dg, "c");
        assert!(root_i > leaf_i); // root is more unstable (all efferent)
    }

    #[test]
    fn max_depth() {
        let dg = chain_graph();
        assert_eq!(DependencyMetrics::max_depth(&dg), 2);
    }

    #[test]
    fn avg_fan() {
        let dg = diamond_graph();
        let fi = DependencyMetrics::avg_fan_in(&dg);
        let fo = DependencyMetrics::avg_fan_out(&dg);
        assert!(fi > 0.0);
        assert!(fo > 0.0);
    }
}
