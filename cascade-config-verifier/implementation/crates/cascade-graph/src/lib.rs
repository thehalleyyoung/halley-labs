//! # cascade-graph
//!
//! Graph algorithms for the Retry-Timeout Interaction Graph (RTIG) in the
//! CascadeVerify project. Provides topology construction, path analysis,
//! cascade composition, structural algorithms, treewidth estimation, and
//! symmetry detection for SMT pruning.

pub mod algorithms;
pub mod cascade_paths;
pub mod path_analysis;
pub mod rtig;
pub mod symmetry;
pub mod topology_builder;
pub mod treewidth;

pub use algorithms::{
    DominatorTree, MinCut, TransitiveClosure, articulation_points, betweenness_centrality,
    bridges, condensation_graph, dominator_tree, graph_diameter, graph_radius,
    k_core_decomposition, minimum_cut, shortest_path_dag, strongly_connected_components,
    tarjan_scc, transitive_closure,
};
pub use cascade_paths::{
    BottleneckFinder, CascadePathComposition, ConvergenceChecker, CriticalPathFinder,
    LoadPropagator, PropagationSimulator, SimulationState,
};
pub use path_analysis::{
    CascadePathAnalyzer, FanInAnalysis, PathComposer, PathEnumerator, RiskyPath,
    TimeoutChainResult,
};
pub use rtig::{GraphStats, RtigGraph, RtigGraphBuilder};
pub use symmetry::{
    AutomorphismDetector, Coloring, NodeSignature, Orbit, SymmetryBreakingConstraints,
    SymmetryClass,
};
pub use topology_builder::{
    PolicyResolver, ServiceResolver, TopologyBuilder, TopologyDiff, TopologySnapshot,
    TopologyWarning,
};
pub use treewidth::{Bag, EstimationMethod, TreeDecomposition, TreewidthEstimator};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use cascade_types::policy::{ResiliencePolicy, RetryPolicy};
    use cascade_types::service::ServiceId;

    fn sid(s: &str) -> ServiceId {
        ServiceId::new(s)
    }

    fn policy_with_retries(n: u32) -> ResiliencePolicy {
        ResiliencePolicy::empty().with_retry(RetryPolicy::new(n))
    }

    // -- RtigGraph basics ---------------------------------------------------

    #[test]
    fn test_reexport_rtig_graph() {
        let mut g = RtigGraph::new();
        g.add_service(sid("A"));
        g.add_service(sid("B"));
        g.add_dependency(&sid("A"), &sid("B"), policy_with_retries(2));
        assert_eq!(g.service_count(), 2);
        assert_eq!(g.dependency_count(), 1);
    }

    #[test]
    fn test_reexport_graph_builder() {
        let g = RtigGraphBuilder::new()
            .add_service(sid("X"))
            .add_service(sid("Y"))
            .add_dependency(sid("X"), sid("Y"), ResiliencePolicy::empty())
            .build()
            .unwrap();
        assert!(g.contains_service(&sid("X")));
        assert!(g.contains_service(&sid("Y")));
    }

    #[test]
    fn test_reexport_graph_stats() {
        let g = rtig::build_chain(&["A", "B", "C"], 2);
        let stats: GraphStats = g.graph_stats();
        assert_eq!(stats.service_count, 3);
        assert_eq!(stats.dependency_count, 2);
        assert!(stats.is_dag);
        assert_eq!(stats.cycle_count, 0);
    }

    // -- ServiceNode --------------------------------------------------------

    #[test]
    fn test_service_node_creation() {
        let node = ServiceNode::new("api-server", 1000);
        assert_eq!(node.id, "api-server");
        assert_eq!(node.capacity, 1000);
        assert_eq!(node.baseline_load, 0);
        assert_eq!(node.tier, 0);
    }

    #[test]
    fn test_service_node_builder() {
        let node = ServiceNode::new("db", 500)
            .with_baseline_load(200)
            .with_tier(3);
        assert_eq!(node.baseline_load, 200);
        assert_eq!(node.tier, 3);
    }

    #[test]
    fn test_service_node_headroom() {
        let node = ServiceNode::new("svc", 100).with_baseline_load(30);
        let headroom = node.headroom();
        assert!((headroom - 0.7).abs() < f64::EPSILON);
    }

    #[test]
    fn test_service_node_headroom_zero_capacity() {
        let node = ServiceNode::new("svc", 0);
        assert_eq!(node.headroom(), 0.0);
    }

    #[test]
    fn test_service_node_overloaded() {
        let ok = ServiceNode::new("svc", 100).with_baseline_load(80);
        assert!(!ok.is_overloaded());

        let bad = ServiceNode::new("svc", 100).with_baseline_load(150);
        assert!(bad.is_overloaded());
    }

    #[test]
    fn test_service_node_to_service_id() {
        let node = ServiceNode::new("my-svc", 100);
        let id: ServiceId = node.into();
        assert_eq!(id.as_str(), "my-svc");
    }

    #[test]
    fn test_service_node_serde() {
        let node = ServiceNode::new("test", 500).with_baseline_load(100);
        let json = serde_json::to_string(&node).unwrap();
        let deser: ServiceNode = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.id, "test");
        assert_eq!(deser.capacity, 500);
        assert_eq!(deser.baseline_load, 100);
    }

    // -- DependencyEdgeInfo -------------------------------------------------

    #[test]
    fn test_dependency_edge_info_creation() {
        let e = DependencyEdgeInfo::new("frontend", "backend");
        assert_eq!(e.source, "frontend");
        assert_eq!(e.target, "backend");
        assert_eq!(e.retry_count, 0);
        assert_eq!(e.timeout_ms, 30_000);
        assert_eq!(e.weight, 1.0);
    }

    #[test]
    fn test_dependency_edge_info_builder() {
        let e = DependencyEdgeInfo::new("a", "b")
            .with_retry_count(3)
            .with_timeout_ms(5000)
            .with_weight(2.5);
        assert_eq!(e.retry_count, 3);
        assert_eq!(e.timeout_ms, 5000);
        assert_eq!(e.weight, 2.5);
    }

    #[test]
    fn test_dependency_edge_info_amplification() {
        let e = DependencyEdgeInfo::new("a", "b").with_retry_count(4);
        assert_eq!(e.amplification_factor(), 5);

        let e0 = DependencyEdgeInfo::new("a", "b");
        assert_eq!(e0.amplification_factor(), 1);
    }

    #[test]
    fn test_dependency_edge_info_to_policy_with_retries() {
        let e = DependencyEdgeInfo::new("a", "b")
            .with_retry_count(3)
            .with_timeout_ms(5000);
        let policy = e.to_resilience_policy();
        assert!(policy.retry.is_some());
        assert!(policy.timeout.is_some());
        assert_eq!(policy.amplification_factor(), 4);
    }

    #[test]
    fn test_dependency_edge_info_to_policy_no_retries() {
        let e = DependencyEdgeInfo::new("a", "b").with_timeout_ms(10000);
        let policy = e.to_resilience_policy();
        assert!(policy.retry.is_none());
        assert!(policy.timeout.is_some());
        assert_eq!(policy.amplification_factor(), 1);
    }

    #[test]
    fn test_dependency_edge_info_serde() {
        let e = DependencyEdgeInfo::new("x", "y").with_retry_count(2);
        let json = serde_json::to_string(&e).unwrap();
        let deser: DependencyEdgeInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.source, "x");
        assert_eq!(deser.target, "y");
        assert_eq!(deser.retry_count, 2);
    }

    // -- RtigGraph convenience methods --------------------------------------

    #[test]
    fn test_add_edge_info() {
        let mut g = RtigGraph::new();
        g.add_service(sid("A"));
        g.add_service(sid("B"));
        let result = g.add_edge(DependencyEdgeInfo::new("A", "B").with_retry_count(3));
        assert!(result.is_some());
        assert_eq!(g.dependency_count(), 1);
        let pol = g.get_edge_policy(&sid("A"), &sid("B")).unwrap();
        assert_eq!(pol.amplification_factor(), 4);
    }

    #[test]
    fn test_add_edge_missing_node() {
        let mut g = RtigGraph::new();
        g.add_service(sid("A"));
        let result = g.add_edge(DependencyEdgeInfo::new("A", "B"));
        assert!(result.is_none());
    }

    #[test]
    fn test_incoming_edges() {
        let mut g = RtigGraph::new();
        for s in ["A", "B", "C"] {
            g.add_service(sid(s));
        }
        g.add_dependency(
            &sid("A"),
            &sid("C"),
            policy_with_retries(2),
        );
        g.add_dependency(
            &sid("B"),
            &sid("C"),
            policy_with_retries(3),
        );

        let incoming = g.incoming_edges("C");
        assert_eq!(incoming.len(), 2);
        let sources: Vec<&str> = incoming.iter().map(|e| e.target.as_str()).collect();
        assert!(sources.iter().all(|s| *s == "C"));
    }

    #[test]
    fn test_incoming_edges_empty() {
        let g = rtig::build_chain(&["A", "B", "C"], 1);
        let incoming = g.incoming_edges("A");
        assert!(incoming.is_empty());
    }

    #[test]
    fn test_incoming_edges_missing_service() {
        let g = RtigGraph::new();
        let incoming = g.incoming_edges("nonexistent");
        assert!(incoming.is_empty());
    }

    #[test]
    fn test_outgoing_edges() {
        let g = rtig::build_diamond(3);
        let outgoing = g.outgoing_edges("A");
        assert_eq!(outgoing.len(), 2);
    }

    #[test]
    fn test_outgoing_edges_leaf() {
        let g = rtig::build_chain(&["A", "B", "C"], 1);
        let outgoing = g.outgoing_edges("C");
        assert!(outgoing.is_empty());
    }

    #[test]
    fn test_path_amplification_chain() {
        let g = rtig::build_chain(&["A", "B", "C"], 2);
        let amp = g.path_amplification("A", "C");
        // Each edge has amp factor 3, path amp = 3 * 3 = 9
        assert_eq!(amp, 9);
    }

    #[test]
    fn test_path_amplification_diamond() {
        let g = rtig::build_diamond(2);
        let amp = g.path_amplification("A", "D");
        // Two paths, each with amp 3 * 3 = 9
        assert_eq!(amp, 9);
    }

    #[test]
    fn test_path_amplification_no_path() {
        let g = rtig::build_chain(&["A", "B", "C"], 1);
        let amp = g.path_amplification("C", "A");
        assert_eq!(amp, 0);
    }

    #[test]
    fn test_path_amplification_same_node() {
        let g = rtig::build_chain(&["A", "B"], 1);
        let amp = g.path_amplification("A", "A");
        // No path from A to A in a DAG (empty path)
        assert_eq!(amp, 0);
    }

    // -- CascadePathComposition re-export -----------------------------------

    #[test]
    fn test_cascade_path_composition_reexport() {
        // Ensure the re-export works
        let _: fn() -> CascadePathComposition = CascadePathComposition::new;
    }

    // -- Symmetry re-export -------------------------------------------------

    #[test]
    fn test_automorphism_detector_reexport() {
        let _ = AutomorphismDetector::new();
    }

    // -- Treewidth re-export ------------------------------------------------

    #[test]
    fn test_treewidth_estimator_reexport() {
        let _est = TreewidthEstimator::new(EstimationMethod::MinDegree);
    }
}
