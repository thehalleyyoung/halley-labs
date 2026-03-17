//! Clique expansion of a hypergraph into an ordinary weighted graph.
//!
//! Each hyperedge {v₁, …, v_d} with weight w is expanded into d(d−1)/2
//! graph edges, each with weight w/(d−1).  Duplicate graph edges arising
//! from different hyperedges are merged by summing their weights.

use std::collections::HashMap;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use spectral_types::graph::{Graph, Hypergraph};
use spectral_types::sparse::{CooMatrix, CsrMatrix};

use crate::error::{Result, SpectralCoreError};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for clique expansion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CliqueExpansionConfig {
    /// Maximum number of graph edges allowed; abort if exceeded.
    pub max_edges: usize,
    /// Merge duplicate edges by summing weights (recommended).
    pub merge_duplicates: bool,
    /// Approximate memory budget in MiB.
    pub memory_limit_mb: usize,
}

impl Default for CliqueExpansionConfig {
    fn default() -> Self {
        Self {
            max_edges: 50_000_000,
            merge_duplicates: true,
            memory_limit_mb: 4096,
        }
    }
}

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

/// Outcome of a clique expansion.
#[derive(Debug, Clone)]
pub struct CliqueExpansionResult {
    /// Symmetric adjacency matrix of the expanded graph.
    pub adjacency: CsrMatrix<f64>,
    /// The expanded Graph struct.
    pub graph: Graph,
    /// Number of distinct undirected edges in the graph.
    pub num_graph_edges: usize,
    /// Ratio of graph edges to hyperedges.
    pub expansion_ratio: f64,
    /// Wall-clock time in milliseconds.
    pub time_ms: f64,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Perform the clique expansion H → G.
///
/// # Errors
/// - [`SpectralCoreError::EmptyInput`] if the hypergraph has no vertices.
/// - [`SpectralCoreError::InvalidParameter`] if the estimated expansion
///   exceeds `config.max_edges` or the memory limit.
pub fn clique_expand(
    hypergraph: &Hypergraph,
    config: &CliqueExpansionConfig,
) -> Result<CliqueExpansionResult> {
    let start = Instant::now();
    let n = hypergraph.num_vertices;

    if n == 0 {
        return Err(SpectralCoreError::empty_input(
            "hypergraph has no vertices for clique expansion",
        ));
    }

    // --- estimate size ---
    let (est_edges, est_mem_bytes) = estimate_expansion_size(hypergraph);

    if est_edges > config.max_edges {
        return Err(SpectralCoreError::invalid_parameter(
            "estimated_edges",
            &est_edges.to_string(),
            &format!("exceeds max_edges limit of {}", config.max_edges),
        ));
    }

    let est_mem_mb = est_mem_bytes / (1024 * 1024);
    if est_mem_mb > config.memory_limit_mb {
        return Err(SpectralCoreError::invalid_parameter(
            "estimated_memory_mb",
            &est_mem_mb.to_string(),
            &format!("exceeds memory limit of {} MiB", config.memory_limit_mb),
        ));
    }

    log::info!(
        "Clique expansion: {} hyperedges, est ~{est_edges} graph edges, ~{est_mem_mb} MiB",
        hypergraph.num_edges(),
    );

    // --- expand ---
    let mut edge_map: HashMap<(usize, usize), f64> = HashMap::with_capacity(est_edges);

    for (ei, edge) in hypergraph.edges.iter().enumerate() {
        let d = edge.len();
        if d < 2 {
            continue;
        }
        let w = hypergraph.edge_weights[ei];
        let pair_weight = w / (d - 1) as f64;

        for a in 0..d {
            let u = edge[a];
            for b in (a + 1)..d {
                let v = edge[b];
                let key = if u < v { (u, v) } else { (v, u) };
                if config.merge_duplicates {
                    *edge_map.entry(key).or_insert(0.0) += pair_weight;
                } else {
                    edge_map.entry(key).or_insert(pair_weight);
                }
            }
        }
    }

    // Collect into sorted edge list.
    let mut edge_list: Vec<(usize, usize, f64)> = edge_map
        .into_iter()
        .map(|((u, v), w)| (u, v, w))
        .collect();
    edge_list.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

    let num_graph_edges = edge_list.len();

    // Build adjacency CSR.
    let adjacency = build_adjacency_from_edges(n, &edge_list);

    // Build Graph.
    let mut graph = Graph::new(n);
    for &(u, v, w) in &edge_list {
        graph.add_edge(u, v, w);
    }

    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    let expansion_ratio = if hypergraph.num_edges() > 0 {
        num_graph_edges as f64 / hypergraph.num_edges() as f64
    } else {
        0.0
    };

    log::info!(
        "Clique expansion done: {num_graph_edges} graph edges, ratio={expansion_ratio:.2}, time={elapsed:.1}ms"
    );

    Ok(CliqueExpansionResult {
        adjacency,
        graph,
        num_graph_edges,
        expansion_ratio,
        time_ms: elapsed,
    })
}

/// Estimate the number of graph edges and memory (bytes) that the clique
/// expansion will produce.
pub fn estimate_expansion_size(hypergraph: &Hypergraph) -> (usize, usize) {
    let mut total_pairs: usize = 0;
    for edge in &hypergraph.edges {
        let d = edge.len();
        if d >= 2 {
            total_pairs += d * (d - 1) / 2;
        }
    }
    // Each unique edge costs ≈24 bytes in the HashMap + 24 bytes in CSR.
    let est_mem = total_pairs * 48;
    (total_pairs, est_mem)
}

/// Build a symmetric CSR adjacency matrix from an undirected edge list.
///
/// Each `(u, v, w)` is stored in both the u-th and v-th row.
pub fn build_adjacency_from_edges(n: usize, edges: &[(usize, usize, f64)]) -> CsrMatrix<f64> {
    if n == 0 {
        return CsrMatrix::<f64>::zeros(0, 0);
    }

    let mut coo = CooMatrix::with_capacity(n, n, 2 * edges.len());
    for &(u, v, w) in edges {
        if u < n && v < n {
            coo.push(u, v, w);
            coo.push(v, u, w);
        }
    }
    coo.to_csr()
}

/// Compute weighted vertex degrees from a symmetric adjacency matrix.
pub fn weighted_degree_from_adjacency(adj: &CsrMatrix<f64>) -> Vec<f64> {
    let (n, _) = adj.shape();
    let mut deg = Vec::with_capacity(n);
    for i in 0..n {
        let vals = adj.row_values(i);
        let d: f64 = vals.iter().sum();
        deg.push(d);
    }
    deg
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn triangle_hypergraph() -> Hypergraph {
        // Single hyperedge {0,1,2} weight 6.0
        let mut hg = Hypergraph::new(3);
        hg.add_edge(vec![0, 1, 2], 6.0);
        hg
    }

    fn two_edge_hypergraph() -> Hypergraph {
        // 4 vertices, edges: {0,1,2} w=3, {1,2,3} w=6
        let mut hg = Hypergraph::new(4);
        hg.add_edge(vec![0, 1, 2], 3.0);
        hg.add_edge(vec![1, 2, 3], 6.0);
        hg
    }

    #[test]
    fn test_clique_expand_triangle() {
        let hg = triangle_hypergraph();
        let cfg = CliqueExpansionConfig::default();
        let res = clique_expand(&hg, &cfg).unwrap();
        // 1 edge with d=3 → 3 pairs, weight per pair = 6/(3-1) = 3.
        assert_eq!(res.num_graph_edges, 3);
        // Check adjacency is symmetric.
        assert!((res.adjacency.get(0, 1) - 3.0).abs() < 1e-12);
        assert!((res.adjacency.get(1, 0) - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_clique_expand_merges_duplicates() {
        let hg = two_edge_hypergraph();
        let cfg = CliqueExpansionConfig::default();
        let res = clique_expand(&hg, &cfg).unwrap();
        // Edge (1,2) appears in both hyperedges:
        //   from edge 0: 3/(3-1) = 1.5
        //   from edge 1: 6/(3-1) = 3.0
        //   merged: 4.5
        assert!((res.adjacency.get(1, 2) - 4.5).abs() < 1e-12);
    }

    #[test]
    fn test_clique_expand_no_merge() {
        let hg = two_edge_hypergraph();
        let cfg = CliqueExpansionConfig {
            merge_duplicates: false,
            ..Default::default()
        };
        let res = clique_expand(&hg, &cfg).unwrap();
        // Without merging, the first insertion wins; (1,2) gets 1.5 from edge 0.
        assert!((res.adjacency.get(1, 2) - 1.5).abs() < 1e-12);
    }

    #[test]
    fn test_clique_expand_empty_error() {
        let hg = Hypergraph::new(0);
        let cfg = CliqueExpansionConfig::default();
        assert!(clique_expand(&hg, &cfg).is_err());
    }

    #[test]
    fn test_clique_expand_max_edges_exceeded() {
        let mut hg = Hypergraph::new(100);
        hg.add_edge((0..100).collect(), 1.0);
        let cfg = CliqueExpansionConfig {
            max_edges: 10,
            ..Default::default()
        };
        let res = clique_expand(&hg, &cfg);
        assert!(res.is_err());
    }

    #[test]
    fn test_estimate_expansion_size() {
        let hg = triangle_hypergraph();
        let (edges, _) = estimate_expansion_size(&hg);
        assert_eq!(edges, 3);
    }

    #[test]
    fn test_estimate_expansion_singleton() {
        let mut hg = Hypergraph::new(2);
        hg.add_edge(vec![0], 1.0);
        let (edges, _) = estimate_expansion_size(&hg);
        assert_eq!(edges, 0);
    }

    #[test]
    fn test_build_adjacency_from_edges_basic() {
        let edges = vec![(0, 1, 1.0), (1, 2, 2.0)];
        let adj = build_adjacency_from_edges(3, &edges);
        assert!((adj.get(0, 1) - 1.0).abs() < 1e-12);
        assert!((adj.get(1, 0) - 1.0).abs() < 1e-12);
        assert!((adj.get(1, 2) - 2.0).abs() < 1e-12);
        assert!((adj.get(0, 2)).abs() < 1e-12);
    }

    #[test]
    fn test_weighted_degree() {
        let edges = vec![(0, 1, 3.0), (0, 2, 5.0)];
        let adj = build_adjacency_from_edges(3, &edges);
        let deg = weighted_degree_from_adjacency(&adj);
        assert!((deg[0] - 8.0).abs() < 1e-12);
        assert!((deg[1] - 3.0).abs() < 1e-12);
        assert!((deg[2] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_expansion_ratio() {
        let hg = two_edge_hypergraph();
        let cfg = CliqueExpansionConfig::default();
        let res = clique_expand(&hg, &cfg).unwrap();
        // 2 hyperedges → 5 unique graph edges → ratio = 2.5
        assert!(res.expansion_ratio > 0.0);
    }

    #[test]
    fn test_graph_struct_populated() {
        let hg = triangle_hypergraph();
        let cfg = CliqueExpansionConfig::default();
        let res = clique_expand(&hg, &cfg).unwrap();
        assert_eq!(res.graph.num_vertices, 3);
        assert!(res.graph.num_edges() > 0);
    }

    #[test]
    fn test_clique_expand_no_edges() {
        let hg = Hypergraph::new(5);
        let cfg = CliqueExpansionConfig::default();
        let res = clique_expand(&hg, &cfg).unwrap();
        assert_eq!(res.num_graph_edges, 0);
        assert_eq!(res.expansion_ratio, 0.0);
    }
}
