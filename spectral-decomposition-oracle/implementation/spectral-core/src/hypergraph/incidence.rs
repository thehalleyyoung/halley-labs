//! Incidence matrix construction for the hypergraph Laplacian.
//!
//! Provides binary and weighted incidence matrices, degree matrices, and the
//! incidence-based Laplacian L = D_v − H W_e D_e⁻¹ Hᵀ (Bolla 1993).

use spectral_types::graph::Hypergraph;
use spectral_types::sparse::{CooMatrix, CsrMatrix};

use crate::error::{Result, SpectralCoreError};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Precomputed diagonal matrices for the incidence Laplacian.
#[derive(Debug, Clone)]
pub struct DegreeMatrices {
    /// D_v diagonal: vertex degree (number of incident edges weighted by w/d).
    pub vertex_degrees: Vec<f64>,
    /// D_e diagonal: edge degree (|eᵢ|).
    pub edge_degrees: Vec<f64>,
    /// W_e diagonal: edge weights (wᵢ).
    pub edge_weights: Vec<f64>,
    /// D_v^{−1/2} (with zero-degree guard).
    pub inv_sqrt_vertex_degrees: Vec<f64>,
    /// D_e^{−1} (with zero-degree guard).
    pub inv_edge_degrees: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Build the binary vertex–hyperedge incidence matrix H (n × m).
///
/// H[v][e] = 1 if vertex v ∈ hyperedge e.
pub fn build_incidence_matrix(hypergraph: &Hypergraph) -> CsrMatrix<f64> {
    let n = hypergraph.num_vertices;
    let m = hypergraph.num_edges();

    if m == 0 || n == 0 {
        return CsrMatrix::<f64>::zeros(n, m);
    }

    let nnz_est: usize = hypergraph.edges.iter().map(|e| e.len()).sum();
    let mut coo = CooMatrix::with_capacity(n, m, nnz_est);

    for (edge_idx, edge) in hypergraph.edges.iter().enumerate() {
        for &v in edge {
            if v < n {
                coo.push(v, edge_idx, 1.0);
            }
        }
    }

    coo.to_csr()
}

/// Build the weighted incidence matrix H_w (n × m).
///
/// H_w[v][e] = sqrt(w_e) if vertex v ∈ hyperedge e, else 0.
pub fn build_weighted_incidence(hypergraph: &Hypergraph) -> CsrMatrix<f64> {
    let n = hypergraph.num_vertices;
    let m = hypergraph.num_edges();

    if m == 0 || n == 0 {
        return CsrMatrix::<f64>::zeros(n, m);
    }

    let nnz_est: usize = hypergraph.edges.iter().map(|e| e.len()).sum();
    let mut coo = CooMatrix::with_capacity(n, m, nnz_est);

    for (edge_idx, edge) in hypergraph.edges.iter().enumerate() {
        let w_sqrt = hypergraph.edge_weights[edge_idx].abs().sqrt();
        for &v in edge {
            if v < n {
                coo.push(v, edge_idx, w_sqrt);
            }
        }
    }

    coo.to_csr()
}

/// Compute the degree matrices D_v, D_e, W_e, and their inverses.
pub fn build_degree_matrices(hypergraph: &Hypergraph) -> DegreeMatrices {
    let n = hypergraph.num_vertices;
    let _m = hypergraph.num_edges();

    let edge_degrees: Vec<f64> = hypergraph.edges.iter().map(|e| e.len() as f64).collect();
    let edge_weights: Vec<f64> = hypergraph.edge_weights.clone();

    // Vertex degrees: d_v = Σ_{e ∋ v} w_e / d_e
    let mut vertex_degrees = vec![0.0f64; n];
    for (ei, edge) in hypergraph.edges.iter().enumerate() {
        let de = edge.len() as f64;
        if de < 1.0 {
            continue;
        }
        let contrib = edge_weights[ei] / de;
        for &v in edge {
            if v < n {
                vertex_degrees[v] += contrib;
            }
        }
    }

    let inv_sqrt_vertex_degrees: Vec<f64> = vertex_degrees
        .iter()
        .map(|&d| if d > 1e-300 { 1.0 / d.sqrt() } else { 0.0 })
        .collect();

    let inv_edge_degrees: Vec<f64> = edge_degrees
        .iter()
        .map(|&d| if d > 0.5 { 1.0 / d } else { 0.0 })
        .collect();

    DegreeMatrices {
        vertex_degrees,
        edge_degrees,
        edge_weights,
        inv_sqrt_vertex_degrees,
        inv_edge_degrees,
    }
}

/// Compute the incidence-based Laplacian:
///
/// L = D_v − H · diag(w_e) · diag(1/d_e) · Hᵀ
///
/// This is O(nnz) in the incidence representation and is preferred when
/// the maximum edge degree is large.
pub fn compute_incidence_laplacian(hypergraph: &Hypergraph) -> Result<CsrMatrix<f64>> {
    let n = hypergraph.num_vertices;
    let m = hypergraph.num_edges();

    if n == 0 {
        return Err(SpectralCoreError::empty_input(
            "hypergraph has no vertices",
        ));
    }
    if m == 0 {
        log::warn!("Hypergraph has no edges; Laplacian is the zero matrix");
        return Ok(CsrMatrix::<f64>::zeros(n, n));
    }

    let dm = build_degree_matrices(hypergraph);

    // Accumulate L = D_v − H W_e D_e^{-1} H^T  using per-edge contributions.
    // For each edge e with weight w_e and degree d_e, the outer product
    //   (w_e / d_e) * 1_e 1_e^T
    // contributes w_e/d_e to every pair (u, v) ∈ e × e.
    // L = D_v − Σ_e (w_e / d_e) 1_e 1_e^T
    //
    // We build L as a COO matrix.
    // Diagonal entries: L[v][v] = d_v  (from D_v)
    // Off-diagonal: L[u][v] = −Σ_{e ∋ u,v} w_e / d_e  for u ≠ v

    // Use a map to accumulate off-diagonal values.
    let mut off_diag: std::collections::HashMap<(usize, usize), f64> =
        std::collections::HashMap::new();

    for (ei, edge) in hypergraph.edges.iter().enumerate() {
        let de = edge.len();
        if de < 2 {
            continue;
        }
        let coeff = dm.edge_weights[ei] * dm.inv_edge_degrees[ei];
        for (a, &u) in edge.iter().enumerate() {
            if u >= n {
                continue;
            }
            for &v in &edge[a + 1..] {
                if v >= n {
                    continue;
                }
                *off_diag.entry((u, v)).or_insert(0.0) += coeff;
                *off_diag.entry((v, u)).or_insert(0.0) += coeff;
            }
        }
    }

    // Build COO.
    let est_nnz = n + 2 * off_diag.len();
    let mut coo = CooMatrix::with_capacity(n, n, est_nnz);

    // Diagonal entries.
    for v in 0..n {
        coo.push(v, v, dm.vertex_degrees[v]);
    }

    // Off-diagonal entries (negated).
    for (&(u, v), &val) in &off_diag {
        coo.push(u, v, -val);
    }

    let laplacian = coo.to_csr();

    log::debug!(
        "Incidence Laplacian: {}×{}, nnz={}",
        n,
        n,
        laplacian.nnz()
    );

    Ok(laplacian)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_hypergraph() -> Hypergraph {
        // 4 vertices, 2 edges: {0,1,2} weight=1, {1,2,3} weight=2
        let mut hg = Hypergraph::new(4);
        hg.add_edge(vec![0, 1, 2], 1.0);
        hg.add_edge(vec![1, 2, 3], 2.0);
        hg
    }

    #[test]
    fn test_incidence_matrix_shape() {
        let hg = simple_hypergraph();
        let h = build_incidence_matrix(&hg);
        assert_eq!(h.shape(), (4, 2));
    }

    #[test]
    fn test_incidence_matrix_values() {
        let hg = simple_hypergraph();
        let h = build_incidence_matrix(&hg);
        // vertex 0 in edge 0 only
        assert!((h.get(0, 0) - 1.0).abs() < 1e-12);
        assert!((h.get(0, 1)).abs() < 1e-12);
        // vertex 1 in both edges
        assert!((h.get(1, 0) - 1.0).abs() < 1e-12);
        assert!((h.get(1, 1) - 1.0).abs() < 1e-12);
        // vertex 3 in edge 1 only
        assert!((h.get(3, 0)).abs() < 1e-12);
        assert!((h.get(3, 1) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_weighted_incidence_values() {
        let hg = simple_hypergraph();
        let hw = build_weighted_incidence(&hg);
        // edge 0 weight 1.0 → sqrt = 1.0
        assert!((hw.get(0, 0) - 1.0).abs() < 1e-12);
        // edge 1 weight 2.0 → sqrt ≈ 1.4142
        assert!((hw.get(1, 1) - 2.0f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn test_degree_matrices() {
        let hg = simple_hypergraph();
        let dm = build_degree_matrices(&hg);
        // Edge degrees: both edges have 3 vertices.
        assert_eq!(dm.edge_degrees, vec![3.0, 3.0]);
        // Vertex 0: only in edge 0, contrib = 1/3 ≈ 0.333
        assert!((dm.vertex_degrees[0] - 1.0 / 3.0).abs() < 1e-12);
        // Vertex 1: in both edges, contrib = 1/3 + 2/3 = 1.0
        assert!((dm.vertex_degrees[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_incidence_laplacian_shape() {
        let hg = simple_hypergraph();
        let l = compute_incidence_laplacian(&hg).unwrap();
        assert_eq!(l.shape(), (4, 4));
    }

    #[test]
    fn test_incidence_laplacian_row_sums_zero() {
        let hg = simple_hypergraph();
        let l = compute_incidence_laplacian(&hg).unwrap();
        let n = l.shape().0;
        for i in 0..n {
            let row_sum = l.row_sum(i);
            assert!(
                row_sum.abs() < 1e-10,
                "row {i} sum = {row_sum} (expected ~0)"
            );
        }
    }

    #[test]
    fn test_incidence_laplacian_symmetric() {
        let hg = simple_hypergraph();
        let l = compute_incidence_laplacian(&hg).unwrap();
        let n = l.shape().0;
        for i in 0..n {
            for j in 0..n {
                let diff = (l.get(i, j) - l.get(j, i)).abs();
                assert!(diff < 1e-12, "L[{i},{j}] != L[{j},{i}]");
            }
        }
    }

    #[test]
    fn test_incidence_laplacian_positive_diagonal() {
        let hg = simple_hypergraph();
        let l = compute_incidence_laplacian(&hg).unwrap();
        let diag = l.diagonal();
        for (i, &d) in diag.iter().enumerate() {
            assert!(d >= -1e-14, "L[{i},{i}] = {d} is negative");
        }
    }

    #[test]
    fn test_empty_hypergraph() {
        let hg = Hypergraph::new(0);
        let result = compute_incidence_laplacian(&hg);
        assert!(result.is_err());
    }

    #[test]
    fn test_no_edges() {
        let hg = Hypergraph::new(3);
        let l = compute_incidence_laplacian(&hg).unwrap();
        // Zero matrix.
        assert_eq!(l.nnz(), 0);
    }

    #[test]
    fn test_single_edge() {
        let mut hg = Hypergraph::new(3);
        hg.add_edge(vec![0, 1, 2], 3.0);
        let l = compute_incidence_laplacian(&hg).unwrap();
        // Symmetric Laplacian with row sums = 0.
        for i in 0..3 {
            assert!(l.row_sum(i).abs() < 1e-12);
        }
    }

    #[test]
    fn test_build_incidence_empty() {
        let hg = Hypergraph::new(0);
        let h = build_incidence_matrix(&hg);
        assert_eq!(h.shape(), (0, 0));
    }

    #[test]
    fn test_inv_edge_degrees() {
        let mut hg = Hypergraph::new(4);
        hg.add_edge(vec![0, 1], 1.0);
        hg.add_edge(vec![0, 1, 2, 3], 2.0);
        let dm = build_degree_matrices(&hg);
        assert!((dm.inv_edge_degrees[0] - 0.5).abs() < 1e-12);
        assert!((dm.inv_edge_degrees[1] - 0.25).abs() < 1e-12);
    }
}
