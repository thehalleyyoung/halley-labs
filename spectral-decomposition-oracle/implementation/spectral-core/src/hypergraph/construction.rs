//! Constraint hypergraph construction from MIP sparse matrices.
//!
//! Given a sparse matrix A ∈ ℝ^{m×n}, the constraint hypergraph H(A) = (V, E, w)
//! is defined by:
//! - V = {0, …, n−1} (variables / columns)
//! - E = {e₀, …, e_{m−1}} where eᵢ = { j : A_{ij} ≠ 0 }
//! - wᵢ = ‖Ã_{i,:}‖₂² / dᵢ  (squared row norm of the scaled matrix, normalised
//!   by edge degree)

use std::time::Instant;

use serde::{Deserialize, Serialize};
use spectral_types::graph::Hypergraph;
use spectral_types::mip::MipInstance;
use spectral_types::sparse::CsrMatrix;

use crate::error::{Result, SpectralCoreError};
use crate::hypergraph::preprocessing::{scale_matrix, ScalingMethod};
use crate::hypergraph::{compute_stats, HypergraphResult, ScalingFactors};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration knobs for hypergraph construction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HypergraphConfig {
    /// Absolute values below this threshold are treated as structural zeros.
    pub zero_threshold: f64,
    /// Whether to equilibrate (scale) the matrix before construction.
    pub apply_scaling: bool,
    /// Which scaling method to use when `apply_scaling` is true.
    pub scaling_method: ScalingMethod,
    /// Drop edges that contain no vertices (empty rows).
    pub remove_empty_edges: bool,
    /// Drop edges that contain exactly one vertex.
    pub remove_singleton_edges: bool,
    /// Minimum edge degree to keep (0 means keep everything).
    pub min_edge_degree: usize,
}

impl Default for HypergraphConfig {
    fn default() -> Self {
        Self {
            zero_threshold: 1e-10,
            apply_scaling: true,
            scaling_method: ScalingMethod::Ruiz,
            remove_empty_edges: true,
            remove_singleton_edges: false,
            min_edge_degree: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Degree statistics
// ---------------------------------------------------------------------------

/// Vertex-degree and edge-degree statistics.
#[derive(Debug, Clone)]
pub struct DegreeStats {
    pub vertex_degrees: Vec<usize>,
    pub max_vertex_degree: usize,
    pub min_vertex_degree: usize,
    pub avg_vertex_degree: f64,
    pub edge_degrees: Vec<usize>,
    pub max_edge_degree: usize,
    pub min_edge_degree: usize,
    pub avg_edge_degree: f64,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Build the constraint hypergraph directly from a [`MipInstance`], using
/// default configuration.
pub fn build_constraint_hypergraph(instance: &MipInstance) -> Result<HypergraphResult> {
    let config = HypergraphConfig::default();
    build_from_matrix(&instance.constraint_matrix, &config)
}

/// Build a constraint hypergraph from a sparse matrix with explicit config.
///
/// # Errors
/// Returns [`SpectralCoreError::EmptyInput`] if the matrix has zero rows and
/// zero columns.  Returns [`SpectralCoreError::HypergraphConstruction`] if
/// any other invariant is violated.
pub fn build_from_matrix(
    matrix: &CsrMatrix<f64>,
    config: &HypergraphConfig,
) -> Result<HypergraphResult> {
    let start = Instant::now();
    let (m, n) = matrix.shape();

    if m == 0 && n == 0 {
        return Err(SpectralCoreError::empty_input("constraint matrix is 0×0"));
    }

    log::info!("Building constraint hypergraph from {m}×{n} matrix (nnz={})", matrix.nnz());

    // --- Step 1: optional scaling ---
    let (work_matrix, scaling_factors) = if config.apply_scaling && matrix.nnz() > 0 {
        let sr = scale_matrix(matrix, config.scaling_method);
        let sf = ScalingFactors {
            row_scaling: sr.row_scaling,
            col_scaling: sr.col_scaling,
            method: format!("{:?}", config.scaling_method),
            iterations: sr.iterations,
            condition_before: sr.condition_before,
            condition_after: sr.condition_after,
        };
        (sr.scaled_matrix, Some(sf))
    } else {
        (matrix.clone(), None)
    };

    // --- Step 2: extract hyperedges ---
    let mut edges: Vec<Vec<usize>> = Vec::with_capacity(m);
    let mut weights: Vec<f64> = Vec::with_capacity(m);

    for i in 0..m {
        let col_indices = work_matrix.row_indices(i);
        let row_vals = work_matrix.row_values(i);

        // Collect nonzero column indices above the threshold.
        let mut edge: Vec<usize> = Vec::new();
        let mut sum_sq = 0.0f64;

        for (k, &j) in col_indices.iter().enumerate() {
            let v = row_vals[k];
            if v.abs() > config.zero_threshold {
                edge.push(j);
                sum_sq += v * v;
            }
        }

        let degree = edge.len();
        let weight = compute_edge_weight_from_parts(sum_sq, degree);

        edges.push(edge);
        weights.push(weight);
    }

    // --- Step 3: filter ---
    let effective_min = if config.remove_empty_edges {
        config.min_edge_degree.max(1)
    } else {
        config.min_edge_degree
    };

    let (mut filt_edges, mut filt_weights) = if effective_min > 0
        || config.remove_singleton_edges
    {
        filter_edges(
            edges,
            weights,
            effective_min,
            config.remove_singleton_edges,
        )
    } else {
        (edges, weights)
    };

    // --- Step 4: build Hypergraph ---
    let mut hg = Hypergraph::new(n);
    for (edge, w) in filt_edges.drain(..).zip(filt_weights.drain(..)) {
        hg.add_edge(edge, w);
    }

    validate_hypergraph(&hg)?;

    let stats = compute_stats(&hg);
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;

    log::info!(
        "Hypergraph constructed: {} vertices, {} edges, max_deg={}, time={elapsed:.1}ms",
        stats.num_vertices,
        stats.num_edges,
        stats.max_edge_degree,
    );

    Ok(HypergraphResult {
        hypergraph: hg,
        scaling_factors,
        construction_time_ms: elapsed,
        stats,
    })
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Compute the degree sequence and associated statistics for a hypergraph.
pub fn compute_degree_sequence(hypergraph: &Hypergraph) -> DegreeStats {
    let n = hypergraph.num_vertices;

    // Vertex degrees: number of edges each vertex appears in.
    let mut vertex_degrees = vec![0usize; n];
    for edge in &hypergraph.edges {
        for &v in edge {
            if v < n {
                vertex_degrees[v] += 1;
            }
        }
    }

    let max_vertex_degree = vertex_degrees.iter().copied().max().unwrap_or(0);
    let min_vertex_degree = vertex_degrees.iter().copied().min().unwrap_or(0);
    let avg_vertex_degree = if n > 0 {
        vertex_degrees.iter().sum::<usize>() as f64 / n as f64
    } else {
        0.0
    };

    // Edge degrees: number of vertices in each edge.
    let edge_degrees: Vec<usize> = hypergraph.edges.iter().map(|e| e.len()).collect();
    let max_edge_degree = edge_degrees.iter().copied().max().unwrap_or(0);
    let min_edge_degree = edge_degrees.iter().copied().min().unwrap_or(0);
    let avg_edge_degree = if !edge_degrees.is_empty() {
        edge_degrees.iter().sum::<usize>() as f64 / edge_degrees.len() as f64
    } else {
        0.0
    };

    DegreeStats {
        vertex_degrees,
        max_vertex_degree,
        min_vertex_degree,
        avg_vertex_degree,
        edge_degrees,
        max_edge_degree,
        min_edge_degree,
        avg_edge_degree,
    }
}

/// Compute edge weight from squared row-value sum and edge degree.
///
/// w = ‖row‖₂² / d, with the convention that empty edges get weight 0.
pub fn compute_edge_weight(row_values: &[f64], degree: usize) -> f64 {
    if degree == 0 {
        return 0.0;
    }
    let sum_sq: f64 = row_values.iter().map(|v| v * v).sum();
    sum_sq / degree as f64
}

/// Filter edges that do not meet a minimum degree requirement.
pub fn filter_edges(
    edges: Vec<Vec<usize>>,
    weights: Vec<f64>,
    min_degree: usize,
    remove_singletons: bool,
) -> (Vec<Vec<usize>>, Vec<f64>) {
    let mut out_edges = Vec::with_capacity(edges.len());
    let mut out_weights = Vec::with_capacity(weights.len());
    for (edge, w) in edges.into_iter().zip(weights.into_iter()) {
        let d = edge.len();
        if d < min_degree {
            continue;
        }
        if remove_singletons && d == 1 {
            continue;
        }
        out_edges.push(edge);
        out_weights.push(w);
    }
    (out_edges, out_weights)
}

/// Validate basic invariants of a [`Hypergraph`].
pub fn validate_hypergraph(hg: &Hypergraph) -> Result<()> {
    if hg.edges.len() != hg.edge_weights.len() {
        return Err(SpectralCoreError::hypergraph(format!(
            "edges.len()={} != edge_weights.len()={}",
            hg.edges.len(),
            hg.edge_weights.len(),
        )));
    }
    for (i, edge) in hg.edges.iter().enumerate() {
        for &v in edge {
            if v >= hg.num_vertices {
                return Err(SpectralCoreError::hypergraph(format!(
                    "edge {i} contains vertex {v} >= num_vertices={}",
                    hg.num_vertices,
                )));
            }
        }
    }
    for (i, &w) in hg.edge_weights.iter().enumerate() {
        if w.is_nan() {
            return Err(SpectralCoreError::numerical_instability(format!(
                "edge {i} has NaN weight"
            )));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn compute_edge_weight_from_parts(sum_sq: f64, degree: usize) -> f64 {
    if degree == 0 {
        0.0
    } else {
        sum_sq / degree as f64
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use spectral_types::sparse::CooMatrix;

    /// Build a CSR matrix from dense row-major data.
    fn csr_from_dense(rows: usize, cols: usize, data: &[f64]) -> CsrMatrix<f64> {
        let mut coo = CooMatrix::new(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                let v = data[i * cols + j];
                if v.abs() > 1e-15 {
                    coo.push(i, j, v);
                }
            }
        }
        coo.to_csr()
    }

    fn make_simple_config() -> HypergraphConfig {
        HypergraphConfig {
            apply_scaling: false,
            ..Default::default()
        }
    }

    #[test]
    fn test_build_from_identity() {
        let eye = CsrMatrix::<f64>::identity(3);
        let cfg = make_simple_config();
        let res = build_from_matrix(&eye, &cfg).unwrap();
        assert_eq!(res.hypergraph.num_vertices, 3);
        assert_eq!(res.hypergraph.num_edges(), 3);
        // Each edge has degree 1, weight = 1^2 / 1 = 1.0.
        for w in &res.hypergraph.edge_weights {
            assert!((*w - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_build_from_dense_matrix() {
        // 2×3 matrix:  [1, 2, 0]
        //              [0, 3, 4]
        let mat = csr_from_dense(2, 3, &[1.0, 2.0, 0.0, 0.0, 3.0, 4.0]);
        let cfg = make_simple_config();
        let res = build_from_matrix(&mat, &cfg).unwrap();
        let hg = &res.hypergraph;
        assert_eq!(hg.num_vertices, 3);
        assert_eq!(hg.num_edges(), 2);
        // edge 0: columns {0,1}, weight = (1+4)/2 = 2.5
        assert_eq!(hg.edges[0], vec![0, 1]);
        assert!((hg.edge_weights[0] - 2.5).abs() < 1e-12);
        // edge 1: columns {1,2}, weight = (9+16)/2 = 12.5
        assert_eq!(hg.edges[1], vec![1, 2]);
        assert!((hg.edge_weights[1] - 12.5).abs() < 1e-12);
    }

    #[test]
    fn test_build_removes_empty_edges() {
        // Row 1 is all-zero → empty edge → removed by default.
        let mat = csr_from_dense(3, 2, &[1.0, 0.0, 0.0, 0.0, 0.0, 2.0]);
        let cfg = make_simple_config();
        let res = build_from_matrix(&mat, &cfg).unwrap();
        assert_eq!(res.hypergraph.num_edges(), 2);
    }

    #[test]
    fn test_build_keeps_empty_edges_if_configured() {
        let mat = csr_from_dense(3, 2, &[1.0, 0.0, 0.0, 0.0, 0.0, 2.0]);
        let cfg = HypergraphConfig {
            apply_scaling: false,
            remove_empty_edges: false,
            ..Default::default()
        };
        let res = build_from_matrix(&mat, &cfg).unwrap();
        // Empty edges with weight 0 are kept.
        assert_eq!(res.hypergraph.num_edges(), 3);
    }

    #[test]
    fn test_build_removes_singletons() {
        let mat = csr_from_dense(2, 3, &[1.0, 0.0, 0.0, 0.0, 2.0, 3.0]);
        let cfg = HypergraphConfig {
            apply_scaling: false,
            remove_singleton_edges: true,
            ..Default::default()
        };
        let res = build_from_matrix(&mat, &cfg).unwrap();
        // edge 0 has degree 1 → removed.
        assert_eq!(res.hypergraph.num_edges(), 1);
    }

    #[test]
    fn test_build_with_scaling() {
        let mat = csr_from_dense(2, 2, &[1e6, 0.0, 0.0, 1.0]);
        let cfg = HypergraphConfig {
            apply_scaling: true,
            scaling_method: ScalingMethod::Ruiz,
            ..Default::default()
        };
        let res = build_from_matrix(&mat, &cfg).unwrap();
        assert!(res.scaling_factors.is_some());
    }

    #[test]
    fn test_build_error_on_zero_matrix() {
        let mat = CsrMatrix::<f64>::zeros(0, 0);
        let cfg = make_simple_config();
        let res = build_from_matrix(&mat, &cfg);
        assert!(res.is_err());
    }

    #[test]
    fn test_compute_edge_weight_basic() {
        let vals = vec![3.0, 4.0];
        // sum_sq = 9 + 16 = 25, degree = 2, weight = 12.5
        assert!((compute_edge_weight(&vals, 2) - 12.5).abs() < 1e-12);
    }

    #[test]
    fn test_compute_edge_weight_empty() {
        let vals: Vec<f64> = vec![];
        assert_eq!(compute_edge_weight(&vals, 0), 0.0);
    }

    #[test]
    fn test_filter_edges_min_degree() {
        let edges = vec![vec![0], vec![0, 1], vec![0, 1, 2]];
        let weights = vec![1.0, 2.0, 3.0];
        let (fe, fw) = filter_edges(edges, weights, 2, false);
        assert_eq!(fe.len(), 2);
        assert_eq!(fw, vec![2.0, 3.0]);
    }

    #[test]
    fn test_validate_hypergraph_ok() {
        let mut hg = Hypergraph::new(3);
        hg.add_edge(vec![0, 1], 1.0);
        assert!(validate_hypergraph(&hg).is_ok());
    }

    #[test]
    fn test_validate_hypergraph_bad_vertex() {
        let mut hg = Hypergraph::new(3);
        hg.add_edge(vec![0, 5], 1.0);
        assert!(validate_hypergraph(&hg).is_err());
    }

    #[test]
    fn test_compute_degree_sequence() {
        let mut hg = Hypergraph::new(4);
        hg.add_edge(vec![0, 1, 2], 1.0);
        hg.add_edge(vec![1, 2, 3], 1.0);
        let ds = compute_degree_sequence(&hg);
        assert_eq!(ds.vertex_degrees, vec![1, 2, 2, 1]);
        assert_eq!(ds.max_vertex_degree, 2);
        assert_eq!(ds.min_vertex_degree, 1);
        assert_eq!(ds.edge_degrees, vec![3, 3]);
    }

    #[test]
    fn test_build_from_mip_instance() {
        let mut inst = MipInstance::new("test", 3, 2);
        let mat = csr_from_dense(2, 3, &[1.0, 2.0, 0.0, 0.0, 0.0, 3.0]);
        inst.constraint_matrix = mat;
        let res = build_constraint_hypergraph(&inst).unwrap();
        assert_eq!(res.hypergraph.num_vertices, 3);
        assert!(res.hypergraph.num_edges() > 0);
    }

    #[test]
    fn test_build_zero_threshold() {
        // Values near zero should be ignored.
        let mat = csr_from_dense(1, 3, &[1.0, 1e-15, 2.0]);
        let cfg = HypergraphConfig {
            apply_scaling: false,
            zero_threshold: 1e-10,
            ..Default::default()
        };
        let res = build_from_matrix(&mat, &cfg).unwrap();
        // The 1e-15 entry is below threshold, so edge has degree 2.
        assert_eq!(res.hypergraph.edges[0].len(), 2);
    }

    #[test]
    fn test_construction_time_recorded() {
        let mat = CsrMatrix::<f64>::identity(5);
        let cfg = make_simple_config();
        let res = build_from_matrix(&mat, &cfg).unwrap();
        assert!(res.construction_time_ms >= 0.0);
    }

    #[test]
    fn test_stats_computed_correctly() {
        let mat = csr_from_dense(3, 4, &[
            1.0, 2.0, 0.0, 0.0,
            0.0, 3.0, 4.0, 5.0,
            6.0, 0.0, 0.0, 7.0,
        ]);
        let cfg = make_simple_config();
        let res = build_from_matrix(&mat, &cfg).unwrap();
        assert_eq!(res.stats.num_vertices, 4);
        assert_eq!(res.stats.num_edges, 3);
        assert_eq!(res.stats.max_edge_degree, 3);
    }
}
