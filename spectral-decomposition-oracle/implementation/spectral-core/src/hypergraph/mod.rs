//! Hypergraph construction and Laplacian computation from MIP constraint matrices.
//!
//! This module provides the core spectral decomposition pipeline:
//! 1. **Construction** — build a constraint hypergraph from a MIP's sparse matrix
//! 2. **Preprocessing** — equilibrate (scale) the matrix for numerical stability
//! 3. **Laplacian** — assemble the hypergraph Laplacian (clique-expansion or incidence)
//! 4. **Incidence** — vertex–hyperedge incidence matrices and degree matrices
//! 5. **Clique expansion** — expand a hypergraph into an ordinary weighted graph

pub mod clique_expansion;
pub mod construction;
pub mod incidence;
pub mod laplacian;
pub mod preprocessing;

// Re-export key types and functions.
pub use clique_expansion::{clique_expand, CliqueExpansionConfig, CliqueExpansionResult};
pub use construction::{
    build_constraint_hypergraph, build_from_matrix, HypergraphConfig,
};
pub use incidence::{
    build_degree_matrices, build_incidence_matrix, build_weighted_incidence,
    compute_incidence_laplacian, DegreeMatrices,
};
pub use laplacian::{build_laplacian, build_normalized_laplacian, LaplacianConfig, LaplacianMethod};
pub use preprocessing::{scale_matrix, ScalingMethod, ScalingResult};

use serde::{Deserialize, Serialize};
use spectral_types::graph::Hypergraph;

// ---------------------------------------------------------------------------
// Shared types
// ---------------------------------------------------------------------------

/// Result of hypergraph construction with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HypergraphResult {
    pub hypergraph: Hypergraph,
    pub scaling_factors: Option<ScalingFactors>,
    pub construction_time_ms: f64,
    pub stats: HypergraphStats,
}

/// Aggregate statistics for a hypergraph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HypergraphStats {
    pub num_vertices: usize,
    pub num_edges: usize,
    pub max_edge_degree: usize,
    pub avg_edge_degree: f64,
    pub num_isolated_vertices: usize,
    pub num_singleton_edges: usize,
    pub num_empty_edges: usize,
    pub total_weight: f64,
}

/// Row/column scaling factors produced during matrix equilibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingFactors {
    pub row_scaling: Vec<f64>,
    pub col_scaling: Vec<f64>,
    pub method: String,
    pub iterations: usize,
    pub condition_before: f64,
    pub condition_after: f64,
}

// ---------------------------------------------------------------------------
// Stat computation helpers
// ---------------------------------------------------------------------------

/// Compute [`HypergraphStats`] from a [`Hypergraph`].
pub fn compute_stats(hg: &Hypergraph) -> HypergraphStats {
    let num_vertices = hg.num_vertices;
    let num_edges = hg.edges.len();

    let mut max_edge_degree: usize = 0;
    let mut sum_edge_degree: usize = 0;
    let mut num_singleton_edges: usize = 0;
    let mut num_empty_edges: usize = 0;
    let mut vertex_seen = vec![false; num_vertices];

    for edge in &hg.edges {
        let d = edge.len();
        if d == 0 {
            num_empty_edges += 1;
        } else if d == 1 {
            num_singleton_edges += 1;
        }
        if d > max_edge_degree {
            max_edge_degree = d;
        }
        sum_edge_degree += d;
        for &v in edge {
            if v < num_vertices {
                vertex_seen[v] = true;
            }
        }
    }

    let num_isolated_vertices = vertex_seen.iter().filter(|&&s| !s).count();
    let avg_edge_degree = if num_edges > 0 {
        sum_edge_degree as f64 / num_edges as f64
    } else {
        0.0
    };
    let total_weight: f64 = hg.edge_weights.iter().sum();

    HypergraphStats {
        num_vertices,
        num_edges,
        max_edge_degree,
        avg_edge_degree,
        num_isolated_vertices,
        num_singleton_edges,
        num_empty_edges,
        total_weight,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_hypergraph() -> Hypergraph {
        let mut hg = Hypergraph::new(5);
        hg.add_edge(vec![0, 1, 2], 1.0);
        hg.add_edge(vec![2, 3], 2.0);
        hg.add_edge(vec![3, 4], 0.5);
        hg
    }

    #[test]
    fn test_compute_stats_basic() {
        let hg = sample_hypergraph();
        let stats = compute_stats(&hg);
        assert_eq!(stats.num_vertices, 5);
        assert_eq!(stats.num_edges, 3);
        assert_eq!(stats.max_edge_degree, 3);
        assert!((stats.avg_edge_degree - 7.0 / 3.0).abs() < 1e-12);
        assert_eq!(stats.num_isolated_vertices, 0);
        assert_eq!(stats.num_singleton_edges, 0);
        assert_eq!(stats.num_empty_edges, 0);
        assert!((stats.total_weight - 3.5).abs() < 1e-12);
    }

    #[test]
    fn test_compute_stats_with_empty_and_singleton() {
        let mut hg = Hypergraph::new(6);
        hg.add_edge(vec![0, 1], 1.0);
        hg.add_edge(vec![2], 0.5);
        hg.add_edge(vec![], 0.1);
        let stats = compute_stats(&hg);
        assert_eq!(stats.num_edges, 3);
        assert_eq!(stats.num_singleton_edges, 1);
        assert_eq!(stats.num_empty_edges, 1);
        // vertices 3, 4, 5 are isolated
        assert_eq!(stats.num_isolated_vertices, 3);
    }

    #[test]
    fn test_compute_stats_empty_hypergraph() {
        let hg = Hypergraph::new(0);
        let stats = compute_stats(&hg);
        assert_eq!(stats.num_vertices, 0);
        assert_eq!(stats.num_edges, 0);
        assert_eq!(stats.max_edge_degree, 0);
        assert_eq!(stats.avg_edge_degree, 0.0);
    }

    #[test]
    fn test_hypergraph_result_fields() {
        let hg = sample_hypergraph();
        let stats = compute_stats(&hg);
        let result = HypergraphResult {
            hypergraph: hg,
            scaling_factors: None,
            construction_time_ms: 1.23,
            stats,
        };
        assert_eq!(result.construction_time_ms, 1.23);
        assert!(result.scaling_factors.is_none());
    }

    #[test]
    fn test_scaling_factors_struct() {
        let sf = ScalingFactors {
            row_scaling: vec![1.0, 2.0],
            col_scaling: vec![0.5, 1.5],
            method: "Ruiz".to_string(),
            iterations: 10,
            condition_before: 1e6,
            condition_after: 1.5,
        };
        assert_eq!(sf.method, "Ruiz");
        assert_eq!(sf.iterations, 10);
    }

    #[test]
    fn test_stats_max_edge_degree_single_large() {
        let mut hg = Hypergraph::new(10);
        hg.add_edge((0..10).collect(), 1.0);
        let stats = compute_stats(&hg);
        assert_eq!(stats.max_edge_degree, 10);
        assert_eq!(stats.num_isolated_vertices, 0);
    }
}
