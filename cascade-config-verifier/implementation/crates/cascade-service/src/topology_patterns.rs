//! Common topology pattern detection, generation, and pattern-specific
//! risk analysis.

use std::collections::HashMap;

use cascade_graph::rtig::{DependencyEdgeInfo, RtigGraph, RtigGraphSimpleBuilder, ServiceNode};
use serde::{Deserialize, Serialize};

// ── TopologyPattern ─────────────────────────────────────────────────

/// Recognised topology archetypes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TopologyPattern {
    Chain,
    Star,
    Tree,
    FullMesh,
    HubAndSpoke,
    Layered,
    Microkernel,
    Unknown,
}

impl std::fmt::Display for TopologyPattern {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Chain => write!(f, "Chain"),
            Self::Star => write!(f, "Star"),
            Self::Tree => write!(f, "Tree"),
            Self::FullMesh => write!(f, "Full Mesh"),
            Self::HubAndSpoke => write!(f, "Hub-and-Spoke"),
            Self::Layered => write!(f, "Layered"),
            Self::Microkernel => write!(f, "Microkernel"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

/// A sub-region of the graph matched to a pattern.
#[derive(Debug, Clone)]
pub struct SubGraphMatch {
    pub services: Vec<String>,
    pub pattern: TopologyPattern,
    pub confidence: f64,
}

// ── PatternDetector ─────────────────────────────────────────────────

/// Heuristic pattern detection on an [`RtigGraph`].
pub struct PatternDetector;

impl PatternDetector {
    /// Detect the dominant topology pattern of the graph.
    pub fn detect_pattern(graph: &RtigGraph) -> TopologyPattern {
        let n = graph.service_count();
        let m = graph.edge_count();

        if n == 0 {
            return TopologyPattern::Unknown;
        }
        if n == 1 {
            return TopologyPattern::Star; // degenerate single-node
        }

        // Chain: n-1 edges, max fan-in and fan-out both 1
        if Self::is_chain(graph, n, m) {
            return TopologyPattern::Chain;
        }

        // Star: one center with high fan-out, all others are leaves
        if Self::is_star(graph, n, m) {
            return TopologyPattern::Star;
        }

        // Full mesh: edge count close to n*(n-1)
        if Self::is_full_mesh(graph, n, m) {
            return TopologyPattern::FullMesh;
        }

        // Hub-and-spoke: a few hubs with high fan-out
        if Self::is_hub_and_spoke(graph, n) {
            return TopologyPattern::HubAndSpoke;
        }

        // Tree: DAG with every node except root having fan-in == 1
        if Self::is_tree(graph, n, m) {
            return TopologyPattern::Tree;
        }

        // Layered: DAG where nodes can be partitioned into tiers
        // and all edges go from tier i to tier i+1
        if Self::is_layered(graph) {
            return TopologyPattern::Layered;
        }

        // Microkernel: one core with plugins (leaves connected to core only)
        if Self::is_microkernel(graph, n) {
            return TopologyPattern::Microkernel;
        }

        TopologyPattern::Unknown
    }

    fn is_chain(graph: &RtigGraph, n: usize, m: usize) -> bool {
        if m != n - 1 {
            return false;
        }
        let ids = graph.service_ids();
        let max_fi = ids.iter().map(|id| graph.fan_in(id)).max().unwrap_or(0);
        let max_fo = ids.iter().map(|id| graph.fan_out(id)).max().unwrap_or(0);
        max_fi <= 1 && max_fo <= 1
    }

    fn is_star(graph: &RtigGraph, n: usize, m: usize) -> bool {
        if m != n - 1 {
            return false;
        }
        let ids = graph.service_ids();
        let has_center = ids.iter().any(|id| graph.fan_out(id) == n - 1);
        has_center
    }

    fn is_full_mesh(_graph: &RtigGraph, n: usize, m: usize) -> bool {
        let max_edges = n * (n - 1);
        m as f64 >= max_edges as f64 * 0.8
    }

    fn is_hub_and_spoke(graph: &RtigGraph, n: usize) -> bool {
        if n < 4 {
            return false;
        }
        let ids = graph.service_ids();
        let threshold = (n as f64 * 0.3).ceil() as usize;
        let hub_count = ids.iter().filter(|id| graph.fan_out(id) >= threshold).count();
        hub_count >= 1 && hub_count <= n / 3
    }

    fn is_tree(graph: &RtigGraph, n: usize, m: usize) -> bool {
        if m != n - 1 || !graph.is_dag() {
            return false;
        }
        let ids = graph.service_ids();
        let roots: Vec<_> = ids.iter().filter(|id| graph.fan_in(id) == 0).collect();
        if roots.len() != 1 {
            return false;
        }
        // Every non-root has exactly one incoming edge
        ids.iter().all(|id| graph.fan_in(id) <= 1)
    }

    fn is_layered(graph: &RtigGraph) -> bool {
        if !graph.is_dag() {
            return false;
        }
        let sorted: Vec<String> = match graph.topological_sort() {
            Some(s) => s.into_iter().map(|id| id.to_string()).collect(),
            None => return false,
        };
        let mut depth: HashMap<String, usize> = HashMap::new();
        for svc in &sorted {
            let d = graph
                .predecessors(svc)
                .iter()
                .filter_map(|p| depth.get(*p).copied())
                .max()
                .map(|m| m + 1)
                .unwrap_or(0);
            depth.insert(svc.to_string(), d);
        }
        // Check that all edges go from tier i to tier i+1
        for edge in graph.edges() {
            let src_d = depth.get(edge.source.as_str()).copied().unwrap_or(0);
            let tgt_d = depth.get(edge.target.as_str()).copied().unwrap_or(0);
            if tgt_d != src_d + 1 {
                return false;
            }
        }
        true
    }

    fn is_microkernel(graph: &RtigGraph, n: usize) -> bool {
        if n < 3 {
            return false;
        }
        let ids = graph.service_ids();
        // The "kernel" is the service with the highest combined fan-in + fan-out
        let kernel = ids
            .iter()
            .max_by_key(|id| graph.fan_in(id) + graph.fan_out(id));
        if let Some(k) = kernel {
            let k_degree = graph.fan_in(k) + graph.fan_out(k);
            // Kernel connects to most services
            k_degree as f64 >= (n as f64 - 1.0) * 0.7
                && ids
                    .iter()
                    .filter(|id| *id != k)
                    .all(|id| graph.fan_in(id) + graph.fan_out(id) <= 2)
        } else {
            false
        }
    }

    /// Detect sub-patterns within the graph by examining connected subgraphs.
    pub fn detect_sub_patterns(graph: &RtigGraph) -> Vec<SubGraphMatch> {
        let mut results = Vec::new();

        // Find chains
        let chains = Self::find_chains(graph);
        for chain in chains {
            if chain.len() >= 3 {
                results.push(SubGraphMatch {
                    services: chain,
                    pattern: TopologyPattern::Chain,
                    confidence: 1.0,
                });
            }
        }

        // Find stars
        let ids = graph.service_ids();
        for id in &ids {
            let fo = graph.fan_out(id);
            if fo >= 3 {
                let succs: Vec<String> = graph.successors(id).iter().map(|s| s.to_string()).collect();
                let all_leaves = succs.iter().all(|s| graph.fan_out(s) == 0);
                if all_leaves {
                    let mut svcs = vec![id.to_string()];
                    svcs.extend(succs);
                    results.push(SubGraphMatch {
                        services: svcs,
                        pattern: TopologyPattern::Star,
                        confidence: 0.9,
                    });
                }
            }
        }

        results
    }

    fn find_chains(graph: &RtigGraph) -> Vec<Vec<String>> {
        let ids = graph.service_ids();
        let chain_starts: Vec<&str> = ids
            .iter()
            .filter(|id| graph.fan_in(id) == 0 && graph.fan_out(id) == 1)
            .map(|s| *s)
            .collect();

        let mut chains = Vec::new();
        for start in chain_starts {
            let mut chain = vec![start.to_string()];
            let mut cur = start;
            loop {
                let succs = graph.successors(cur);
                if succs.len() == 1 && graph.fan_in(succs[0]) == 1 {
                    chain.push(succs[0].to_string());
                    cur = succs[0];
                } else {
                    if succs.len() == 1 {
                        chain.push(succs[0].to_string());
                    }
                    break;
                }
            }
            chains.push(chain);
        }
        chains
    }
}

// ── PatternGenerator ────────────────────────────────────────────────

/// Generate synthetic topologies for testing and benchmarking.
pub struct PatternGenerator;

impl PatternGenerator {
    /// Generate a linear chain: s0 -> s1 -> ... -> s_{n-1}.
    pub fn generate_chain(length: usize, retries: u32, timeout_ms: u64) -> RtigGraph {
        let mut builder = RtigGraphSimpleBuilder::new();
        for i in 0..length {
            let name = format!("s{}", i);
            builder = builder.add_service(ServiceNode::new(&name, 100));
        }
        for i in 0..length.saturating_sub(1) {
            let src = format!("s{}", i);
            let tgt = format!("s{}", i + 1);
            builder = builder.add_edge(
                DependencyEdgeInfo::new(&src, &tgt)
                    .with_retry_count(retries)
                    .with_timeout_ms(timeout_ms),
            );
        }
        builder.build()
    }

    /// Generate a star: center -> spoke_0, center -> spoke_1, ...
    pub fn generate_star(center: &str, spokes: usize, retries: u32, timeout_ms: u64) -> RtigGraph {
        let mut builder = RtigGraphSimpleBuilder::new()
            .add_service(ServiceNode::new(center, 1000));
        for i in 0..spokes {
            let name = format!("spoke-{}", i);
            builder = builder
                .add_service(ServiceNode::new(name.as_str(), 100))
                .add_edge(
                    DependencyEdgeInfo::new(center, name.as_str())
                        .with_retry_count(retries)
                        .with_timeout_ms(timeout_ms),
                );
        }
        builder.build()
    }

    /// Generate a balanced tree with given depth and branching factor.
    pub fn generate_tree(
        depth: usize,
        branching: usize,
        retries: u32,
        timeout_ms: u64,
    ) -> RtigGraph {
        let mut builder = RtigGraphSimpleBuilder::new();
        let mut counter = 0usize;
        let root = format!("n{}", counter);
        builder = builder.add_service(ServiceNode::new(root.as_str(), 500));
        counter += 1;

        let mut frontier = vec![root];
        for _level in 0..depth {
            let mut new_frontier = Vec::new();
            for parent in &frontier {
                for _b in 0..branching {
                    let child = format!("n{}", counter);
                    builder = builder
                        .add_service(ServiceNode::new(child.as_str(), 100))
                        .add_edge(
                            DependencyEdgeInfo::new(parent.as_str(), child.as_str())
                                .with_retry_count(retries)
                                .with_timeout_ms(timeout_ms),
                        );
                    counter += 1;
                    new_frontier.push(child);
                }
            }
            frontier = new_frontier;
        }
        builder.build()
    }

    /// Generate a grid mesh: rows × cols with edges going right and down.
    pub fn generate_mesh(
        rows: usize,
        cols: usize,
        retries: u32,
        timeout_ms: u64,
    ) -> RtigGraph {
        let mut builder = RtigGraphSimpleBuilder::new();
        let name = |r: usize, c: usize| -> String { format!("r{}c{}", r, c) };

        for r in 0..rows {
            for c in 0..cols {
                builder = builder.add_service(ServiceNode::new(&name(r, c), 100));
            }
        }
        for r in 0..rows {
            for c in 0..cols {
                if c + 1 < cols {
                    builder = builder.add_edge(
                        DependencyEdgeInfo::new(&name(r, c), &name(r, c + 1))
                            .with_retry_count(retries)
                            .with_timeout_ms(timeout_ms),
                    );
                }
                if r + 1 < rows {
                    builder = builder.add_edge(
                        DependencyEdgeInfo::new(&name(r, c), &name(r + 1, c))
                            .with_retry_count(retries)
                            .with_timeout_ms(timeout_ms),
                    );
                }
            }
        }
        builder.build()
    }

    /// Generate a hub-and-spoke topology with multiple hubs.
    pub fn generate_hub_and_spoke(
        hubs: usize,
        spokes_per_hub: usize,
        retries: u32,
        timeout_ms: u64,
    ) -> RtigGraph {
        let mut builder = RtigGraphSimpleBuilder::new();
        let entry = "entry";
        builder = builder.add_service(ServiceNode::new(entry, 2000));

        for h in 0..hubs {
            let hub_name = format!("hub-{}", h);
            builder = builder
                .add_service(ServiceNode::new(hub_name.as_str(), 500))
                .add_edge(
                    DependencyEdgeInfo::new(entry, hub_name.as_str())
                        .with_retry_count(retries)
                        .with_timeout_ms(timeout_ms),
                );
            for s in 0..spokes_per_hub {
                let spoke = format!("hub{}-spoke{}", h, s);
                builder = builder
                    .add_service(ServiceNode::new(spoke.as_str(), 100))
                    .add_edge(
                        DependencyEdgeInfo::new(hub_name.as_str(), spoke.as_str())
                            .with_retry_count(retries)
                            .with_timeout_ms(timeout_ms),
                    );
            }
        }
        builder.build()
    }

    /// Generate a pseudo-random graph with given edge probability.
    /// Uses a deterministic seed derived from the service count for
    /// reproducibility in tests.
    pub fn generate_random(services: usize, edge_probability: f64) -> RtigGraph {
        let mut builder = RtigGraphSimpleBuilder::new();
        for i in 0..services {
            builder = builder.add_service(ServiceNode::new(&format!("r{}", i), 100));
        }
        // Simple deterministic pseudo-random using a linear congruential generator
        let mut seed: u64 = (services as u64).wrapping_mul(2654435761);
        for i in 0..services {
            for j in 0..services {
                if i == j {
                    continue;
                }
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let val = (seed >> 33) as f64 / (u32::MAX as f64);
                if val < edge_probability {
                    builder = builder.add_edge(
                        DependencyEdgeInfo::new(&format!("r{}", i), &format!("r{}", j))
                            .with_retry_count(2),
                    );
                }
            }
        }
        builder.build()
    }
}

// ── PatternAnalyzer ─────────────────────────────────────────────────

/// Risk assessment for a chain topology.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainRisk {
    pub length: usize,
    pub total_amplification: f64,
    pub worst_case_latency_ms: u64,
    pub weakest_link: String,
    pub weakest_link_capacity: u64,
    pub risk_level: RiskLevel,
}

/// Risk assessment for a star topology.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StarRisk {
    pub center: String,
    pub spoke_count: usize,
    pub center_load_factor: f64,
    pub center_capacity: u64,
    pub risk_level: RiskLevel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Pattern-specific risk analysis.
pub struct PatternAnalyzer;

impl PatternAnalyzer {
    /// Analyse the cascade risk of a chain topology.
    pub fn analyze_chain_risk(chain: &RtigGraph) -> ChainRisk {
        let sorted: Vec<String> = chain.topological_sort()
            .map(|v| v.into_iter().map(|id| id.to_string()).collect())
            .unwrap_or_else(|| {
                chain.service_ids().iter().map(|s| s.to_string()).collect()
            });
        let length = sorted.len();

        let mut total_amp = 1.0_f64;
        let mut total_latency = 0u64;
        let mut weakest = sorted.first().cloned().unwrap_or_default();
        let mut weakest_cap = u64::MAX;

        for w in sorted.windows(2) {
            let edges = chain.outgoing_edges(&w[0]);
            for edge in &edges {
                if edge.target.as_str() == w[1] {
                    total_amp *= edge.amplification_factor_f64();
                    let retries = edge.retry_count.max(1) as u64;
                    total_latency += edge.timeout_ms * retries;
                }
            }
            if let Some(node) = chain.service(&w[1]) {
                if (node.capacity as u64) < weakest_cap {
                    weakest_cap = node.capacity as u64;
                    weakest = w[1].clone();
                }
            }
        }
        if let Some(first) = sorted.first() {
            if let Some(node) = chain.service(first) {
                if (node.capacity as u64) < weakest_cap {
                    weakest_cap = node.capacity as u64;
                    weakest = first.clone();
                }
            }
        }

        let risk_level = if total_amp > 100.0 || length > 8 {
            RiskLevel::Critical
        } else if total_amp > 20.0 || length > 5 {
            RiskLevel::High
        } else if total_amp > 5.0 || length > 3 {
            RiskLevel::Medium
        } else {
            RiskLevel::Low
        };

        ChainRisk {
            length,
            total_amplification: total_amp,
            worst_case_latency_ms: total_latency,
            weakest_link: weakest,
            weakest_link_capacity: weakest_cap,
            risk_level,
        }
    }

    /// Analyse the risk of a star topology.
    pub fn analyze_star_risk(star: &RtigGraph) -> StarRisk {
        let ids = star.service_ids();
        let center = ids
            .iter()
            .max_by_key(|id| star.fan_out(id))
            .map(|s| s.to_string())
            .unwrap_or_default();

        let spoke_count = star.fan_out(&center);
        let center_cap = star.service(&center).map(|n| n.capacity as u64).unwrap_or(0);

        let total_retry_load: f64 = star
            .outgoing_edges(&center)
            .iter()
            .map(|e| e.amplification_factor_f64())
            .sum();

        let load_factor = if center_cap > 0 {
            total_retry_load / center_cap as f64
        } else {
            f64::MAX
        };

        let risk_level = if spoke_count > 20 || load_factor > 1.0 {
            RiskLevel::Critical
        } else if spoke_count > 10 || load_factor > 0.5 {
            RiskLevel::High
        } else if spoke_count > 5 || load_factor > 0.2 {
            RiskLevel::Medium
        } else {
            RiskLevel::Low
        };

        StarRisk {
            center,
            spoke_count,
            center_load_factor: load_factor,
            center_capacity: center_cap,
            risk_level,
        }
    }

    /// Analyse a mesh-topology (grid) for overall risk.
    pub fn analyze_mesh_risk(graph: &RtigGraph) -> MeshRiskSummary {
        let n = graph.service_count();
        let m = graph.edge_count();
        let density = if n > 1 {
            m as f64 / (n * (n - 1)) as f64
        } else {
            0.0
        };

        let ids = graph.service_ids();
        let max_fi = ids.iter().map(|id| graph.fan_in(id)).max().unwrap_or(0);
        let max_fo = ids.iter().map(|id| graph.fan_out(id)).max().unwrap_or(0);

        let risk_level = if density > 0.5 {
            RiskLevel::Critical
        } else if max_fi > 5 || max_fo > 5 {
            RiskLevel::High
        } else if density > 0.2 {
            RiskLevel::Medium
        } else {
            RiskLevel::Low
        };

        MeshRiskSummary {
            service_count: n,
            edge_count: m,
            density,
            max_fan_in: max_fi,
            max_fan_out: max_fo,
            risk_level,
        }
    }
}

/// Summary risk for a mesh/grid topology.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshRiskSummary {
    pub service_count: usize,
    pub edge_count: usize,
    pub density: f64,
    pub max_fan_in: usize,
    pub max_fan_out: usize,
    pub risk_level: RiskLevel,
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── PatternDetector ────

    #[test]
    fn detect_chain() {
        let g = PatternGenerator::generate_chain(5, 2, 1000);
        assert_eq!(PatternDetector::detect_pattern(&g), TopologyPattern::Chain);
    }

    #[test]
    fn detect_star() {
        let g = PatternGenerator::generate_star("hub", 5, 2, 1000);
        assert_eq!(PatternDetector::detect_pattern(&g), TopologyPattern::Star);
    }

    #[test]
    fn detect_tree() {
        let g = PatternGenerator::generate_tree(2, 2, 1, 500);
        let p = PatternDetector::detect_pattern(&g);
        assert!(p == TopologyPattern::Tree || p == TopologyPattern::Layered);
    }

    #[test]
    fn detect_mesh_grid() {
        let g = PatternGenerator::generate_mesh(3, 3, 1, 500);
        let p = PatternDetector::detect_pattern(&g);
        // Grid is layered or unknown, not chain/star
        assert_ne!(p, TopologyPattern::Chain);
        assert_ne!(p, TopologyPattern::Star);
    }

    #[test]
    fn detect_empty() {
        let g = RtigGraphSimpleBuilder::new().build();
        assert_eq!(
            PatternDetector::detect_pattern(&g),
            TopologyPattern::Unknown
        );
    }

    #[test]
    fn detect_single_node() {
        let g = RtigGraphSimpleBuilder::new()
            .add_service(ServiceNode::new("only", 100))
            .build();
        assert_eq!(PatternDetector::detect_pattern(&g), TopologyPattern::Star);
    }

    #[test]
    fn sub_patterns_contain_chain() {
        // Build a mixed graph with a chain sub-pattern
        let g = RtigGraphSimpleBuilder::new()
            .add_service(ServiceNode::new("a", 100))
            .add_service(ServiceNode::new("b", 100))
            .add_service(ServiceNode::new("c", 100))
            .add_service(ServiceNode::new("x", 100))
            .add_service(ServiceNode::new("y", 100))
            .add_edge(DependencyEdgeInfo::new("a", "b"))
            .add_edge(DependencyEdgeInfo::new("b", "c"))
            .add_edge(DependencyEdgeInfo::new("a", "x"))
            .add_edge(DependencyEdgeInfo::new("a", "y"))
            .build();
        let subs = PatternDetector::detect_sub_patterns(&g);
        assert!(!subs.is_empty());
    }

    // ── PatternGenerator ───

    #[test]
    fn gen_chain_size() {
        let g = PatternGenerator::generate_chain(4, 1, 500);
        assert_eq!(g.service_count(), 4);
        assert_eq!(g.edge_count(), 3);
    }

    #[test]
    fn gen_star_size() {
        let g = PatternGenerator::generate_star("hub", 6, 1, 500);
        assert_eq!(g.service_count(), 7); // 1 hub + 6 spokes
        assert_eq!(g.edge_count(), 6);
    }

    #[test]
    fn gen_tree_size() {
        let g = PatternGenerator::generate_tree(2, 3, 1, 500);
        // Level 0: 1, Level 1: 3, Level 2: 9  => 13 nodes
        assert_eq!(g.service_count(), 13);
        assert_eq!(g.edge_count(), 12);
    }

    #[test]
    fn gen_mesh_size() {
        let g = PatternGenerator::generate_mesh(3, 4, 1, 500);
        assert_eq!(g.service_count(), 12);
        // Right edges: 3*3=9, Down edges: 2*4=8
        assert_eq!(g.edge_count(), 17);
    }

    #[test]
    fn gen_hub_spoke_size() {
        let g = PatternGenerator::generate_hub_and_spoke(2, 3, 1, 500);
        // 1 entry + 2 hubs + 6 spokes = 9
        assert_eq!(g.service_count(), 9);
    }

    #[test]
    fn gen_random_deterministic() {
        let g1 = PatternGenerator::generate_random(5, 0.3);
        let g2 = PatternGenerator::generate_random(5, 0.3);
        assert_eq!(g1.service_count(), g2.service_count());
        assert_eq!(g1.edge_count(), g2.edge_count());
    }

    // ── PatternAnalyzer ────

    #[test]
    fn chain_risk_low() {
        let g = PatternGenerator::generate_chain(3, 1, 500);
        let risk = PatternAnalyzer::analyze_chain_risk(&g);
        assert_eq!(risk.length, 3);
        assert!(risk.total_amplification >= 1.0);
        assert!(risk.risk_level == RiskLevel::Low || risk.risk_level == RiskLevel::Medium);
    }

    #[test]
    fn chain_risk_high_retries() {
        let g = PatternGenerator::generate_chain(6, 5, 2000);
        let risk = PatternAnalyzer::analyze_chain_risk(&g);
        assert!(risk.total_amplification > 10.0);
        assert!(risk.risk_level == RiskLevel::High || risk.risk_level == RiskLevel::Critical);
    }

    #[test]
    fn star_risk_small() {
        let g = PatternGenerator::generate_star("hub", 3, 1, 500);
        let risk = PatternAnalyzer::analyze_star_risk(&g);
        assert_eq!(risk.spoke_count, 3);
        assert_eq!(risk.center, "hub");
    }

    #[test]
    fn star_risk_large() {
        let g = PatternGenerator::generate_star("hub", 25, 3, 1000);
        let risk = PatternAnalyzer::analyze_star_risk(&g);
        assert_eq!(risk.risk_level, RiskLevel::Critical);
    }

    #[test]
    fn mesh_risk_summary() {
        let g = PatternGenerator::generate_mesh(3, 3, 2, 1000);
        let risk = PatternAnalyzer::analyze_mesh_risk(&g);
        assert_eq!(risk.service_count, 9);
        assert!(risk.density > 0.0);
    }
}
