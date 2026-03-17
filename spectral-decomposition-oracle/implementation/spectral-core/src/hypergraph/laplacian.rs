//! Hypergraph Laplacian construction with automatic method dispatch.
//!
//! Two Laplacian variants are supported:
//!
//! * **Clique-expansion Laplacian (L_CE)** — expands each hyperedge into a
//!   clique and forms L = D − W.  Preferred when the maximum edge degree is
//!   small (≤ threshold).
//!
//! * **Incidence-based Laplacian (L_I)** — uses the vertex–hyperedge incidence
//!   matrix: L = D_v − H W_e D_e⁻¹ Hᵀ  (Bolla 1993).  Preferred when edges
//!   are large, because it avoids the quadratic blowup of clique expansion.
//!
//! An *auto-dispatch* mode inspects the maximum edge degree and selects the
//! cheaper method.

use serde::{Deserialize, Serialize};
use spectral_types::graph::Hypergraph;
use spectral_types::sparse::{CooMatrix, CsrMatrix};

use crate::error::{Result, SpectralCoreError};
use crate::hypergraph::clique_expansion::{clique_expand, CliqueExpansionConfig};
use crate::hypergraph::incidence::compute_incidence_laplacian;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Which Laplacian construction method to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LaplacianMethod {
    /// Clique-expansion based.
    CliqueExpansion,
    /// Incidence-matrix based (Bolla 1993).
    Incidence,
    /// Automatically select based on max edge degree.
    Auto,
}

impl Default for LaplacianMethod {
    fn default() -> Self {
        Self::Auto
    }
}

/// Configuration for Laplacian construction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaplacianConfig {
    /// Maximum edge degree threshold for using clique expansion.
    /// If max_edge_degree > clique_threshold, the incidence method is used.
    pub clique_threshold: usize,
    /// Regularisation added to zero-degree vertices (avoids division by zero
    /// in the normalised Laplacian).
    pub regularization: f64,
    /// Override automatic method selection.
    pub force_method: Option<LaplacianMethod>,
}

impl Default for LaplacianConfig {
    fn default() -> Self {
        Self {
            clique_threshold: 200,
            regularization: 1e-10,
            force_method: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Build the combinatorial Laplacian of a hypergraph.
///
/// Dispatches to the clique-expansion or incidence method depending on the
/// maximum edge degree and configuration.
pub fn build_laplacian(
    hypergraph: &Hypergraph,
    config: &LaplacianConfig,
) -> Result<CsrMatrix<f64>> {
    let n = hypergraph.num_vertices;
    if n == 0 {
        return Err(SpectralCoreError::empty_input(
            "hypergraph has no vertices for Laplacian",
        ));
    }
    if hypergraph.num_edges() == 0 {
        log::warn!("Hypergraph has no edges; returning zero Laplacian");
        return Ok(CsrMatrix::<f64>::zeros(n, n));
    }

    let method = select_method(hypergraph, config);
    log::info!("Laplacian method: {:?}", method);

    match method {
        LaplacianMethod::CliqueExpansion => build_clique_laplacian(hypergraph),
        LaplacianMethod::Incidence => compute_incidence_laplacian(hypergraph),
        LaplacianMethod::Auto => {
            // Should not reach here after select_method, but handle gracefully.
            compute_incidence_laplacian(hypergraph)
        }
    }
}

/// Build the normalised Laplacian  L_norm = D^{−1/2} L D^{−1/2}.
///
/// Zero-degree vertices receive a small regularisation so that D^{−1/2}
/// is well-defined.  For a connected graph the diagonal entries are 1.
pub fn build_normalized_laplacian(
    hypergraph: &Hypergraph,
    config: &LaplacianConfig,
) -> Result<CsrMatrix<f64>> {
    let l = build_laplacian(hypergraph, config)?;
    let n = l.shape().0;

    let diag = l.diagonal();
    let mut inv_sqrt_d = Vec::with_capacity(n);
    for &d in &diag {
        let d_reg = if d.abs() < config.regularization {
            config.regularization
        } else {
            d
        };
        inv_sqrt_d.push(1.0 / d_reg.sqrt());
    }

    // L_norm[i][j] = inv_sqrt_d[i] * L[i][j] * inv_sqrt_d[j]
    let mut coo = CooMatrix::with_capacity(n, n, l.nnz());
    for i in 0..n {
        let cols = l.row_indices(i);
        let vals = l.row_values(i);
        for (k, &j) in cols.iter().enumerate() {
            let v = vals[k] * inv_sqrt_d[i] * inv_sqrt_d[j];
            coo.push(i, j, v);
        }
    }

    let lnorm = coo.to_csr();

    log::debug!(
        "Normalised Laplacian: {}×{}, nnz={}",
        n,
        n,
        lnorm.nnz()
    );

    Ok(lnorm)
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Extract the degree vector from the Laplacian diagonal.
pub fn compute_vertex_degrees(laplacian: &CsrMatrix<f64>) -> Vec<f64> {
    laplacian.diagonal()
}

/// Heuristic validity check: a Laplacian should be symmetric and have
/// non-negative diagonal entries with approximately zero row sums.
pub fn is_laplacian_valid(laplacian: &CsrMatrix<f64>) -> bool {
    let (n, m) = laplacian.shape();
    if n != m {
        return false;
    }
    let tol = 1e-8;

    for i in 0..n {
        // Positive (or zero) diagonal.
        let d = laplacian.get(i, i);
        if d < -tol {
            return false;
        }
        // Row sum ≈ 0.
        let rs = laplacian.row_sum(i);
        if rs.abs() > tol * (1.0 + d.abs()) {
            return false;
        }
        // Symmetry.
        let cols = laplacian.row_indices(i);
        let vals = laplacian.row_values(i);
        for (k, &j) in cols.iter().enumerate() {
            let lij = vals[k];
            let lji = laplacian.get(j, i);
            if (lij - lji).abs() > tol * (1.0 + lij.abs()) {
                return false;
            }
        }
    }
    true
}

/// Gershgorin bounds on the spectrum of a Laplacian.
///
/// Returns (λ_min_bound, λ_max_bound).
pub fn laplacian_spectrum_bounds(laplacian: &CsrMatrix<f64>) -> (f64, f64) {
    let n = laplacian.shape().0;
    if n == 0 {
        return (0.0, 0.0);
    }

    let mut global_min = f64::MAX;
    let mut global_max = f64::MIN;

    for i in 0..n {
        let d = laplacian.get(i, i);
        let cols = laplacian.row_indices(i);
        let vals = laplacian.row_values(i);
        let mut off_diag_sum = 0.0f64;
        for (k, &j) in cols.iter().enumerate() {
            if j != i {
                off_diag_sum += vals[k].abs();
            }
        }
        let lo = d - off_diag_sum;
        let hi = d + off_diag_sum;
        if lo < global_min {
            global_min = lo;
        }
        if hi > global_max {
            global_max = hi;
        }
    }

    (global_min, global_max)
}

// ---------------------------------------------------------------------------
// Internal: method selection
// ---------------------------------------------------------------------------

fn select_method(hypergraph: &Hypergraph, config: &LaplacianConfig) -> LaplacianMethod {
    if let Some(forced) = config.force_method {
        if forced != LaplacianMethod::Auto {
            return forced;
        }
    }

    let max_deg = hypergraph
        .edges
        .iter()
        .map(|e| e.len())
        .max()
        .unwrap_or(0);

    if max_deg <= config.clique_threshold {
        log::debug!(
            "Auto-dispatch: max_edge_degree={max_deg} <= threshold={}, using CliqueExpansion",
            config.clique_threshold
        );
        LaplacianMethod::CliqueExpansion
    } else {
        log::debug!(
            "Auto-dispatch: max_edge_degree={max_deg} > threshold={}, using Incidence",
            config.clique_threshold
        );
        LaplacianMethod::Incidence
    }
}

// ---------------------------------------------------------------------------
// Internal: clique-expansion Laplacian
// ---------------------------------------------------------------------------

/// Build L_CE = D − W via clique expansion.
fn build_clique_laplacian(hypergraph: &Hypergraph) -> Result<CsrMatrix<f64>> {
    let n = hypergraph.num_vertices;

    // Use the clique expansion module to get the adjacency matrix.
    let ce_config = CliqueExpansionConfig::default();
    let ce = clique_expand(hypergraph, &ce_config)?;
    let w = &ce.adjacency; // symmetric weight matrix

    // D = diag(W * 1)  →  weighted degree.
    let degrees: Vec<f64> = (0..n).map(|i| {
        let vals = w.row_values(i);
        vals.iter().sum::<f64>()
    }).collect();

    // L = D − W  (build via COO).
    let mut coo = CooMatrix::with_capacity(n, n, w.nnz() + n);

    for i in 0..n {
        // Diagonal: degree.
        coo.push(i, i, degrees[i]);
        // Off-diagonal: −W[i][j].
        let cols = w.row_indices(i);
        let vals = w.row_values(i);
        for (k, &j) in cols.iter().enumerate() {
            coo.push(i, j, -vals[k]);
        }
    }

    let laplacian = coo.to_csr();

    log::debug!(
        "Clique-expansion Laplacian: {}×{}, nnz={}",
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

    fn simple_hg() -> Hypergraph {
        let mut hg = Hypergraph::new(4);
        hg.add_edge(vec![0, 1, 2], 3.0);
        hg.add_edge(vec![2, 3], 2.0);
        hg
    }

    fn small_hg() -> Hypergraph {
        let mut hg = Hypergraph::new(3);
        hg.add_edge(vec![0, 1, 2], 6.0);
        hg
    }

    #[test]
    fn test_build_laplacian_auto() {
        let hg = simple_hg();
        let cfg = LaplacianConfig::default();
        let l = build_laplacian(&hg, &cfg).unwrap();
        assert_eq!(l.shape(), (4, 4));
    }

    #[test]
    fn test_build_laplacian_force_clique() {
        let hg = simple_hg();
        let cfg = LaplacianConfig {
            force_method: Some(LaplacianMethod::CliqueExpansion),
            ..Default::default()
        };
        let l = build_laplacian(&hg, &cfg).unwrap();
        assert_eq!(l.shape(), (4, 4));
    }

    #[test]
    fn test_build_laplacian_force_incidence() {
        let hg = simple_hg();
        let cfg = LaplacianConfig {
            force_method: Some(LaplacianMethod::Incidence),
            ..Default::default()
        };
        let l = build_laplacian(&hg, &cfg).unwrap();
        assert_eq!(l.shape(), (4, 4));
    }

    #[test]
    fn test_laplacian_row_sums_zero_clique() {
        let hg = small_hg();
        let cfg = LaplacianConfig {
            force_method: Some(LaplacianMethod::CliqueExpansion),
            ..Default::default()
        };
        let l = build_laplacian(&hg, &cfg).unwrap();
        for i in 0..3 {
            assert!(
                l.row_sum(i).abs() < 1e-10,
                "row {} sum = {}",
                i,
                l.row_sum(i)
            );
        }
    }

    #[test]
    fn test_laplacian_row_sums_zero_incidence() {
        let hg = small_hg();
        let cfg = LaplacianConfig {
            force_method: Some(LaplacianMethod::Incidence),
            ..Default::default()
        };
        let l = build_laplacian(&hg, &cfg).unwrap();
        for i in 0..3 {
            assert!(
                l.row_sum(i).abs() < 1e-10,
                "row {} sum = {}",
                i,
                l.row_sum(i)
            );
        }
    }

    #[test]
    fn test_laplacian_symmetric_clique() {
        let hg = small_hg();
        let cfg = LaplacianConfig {
            force_method: Some(LaplacianMethod::CliqueExpansion),
            ..Default::default()
        };
        let l = build_laplacian(&hg, &cfg).unwrap();
        let n = l.shape().0;
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (l.get(i, j) - l.get(j, i)).abs() < 1e-12,
                    "L[{i},{j}] != L[{j},{i}]"
                );
            }
        }
    }

    #[test]
    fn test_laplacian_symmetric_incidence() {
        let hg = small_hg();
        let cfg = LaplacianConfig {
            force_method: Some(LaplacianMethod::Incidence),
            ..Default::default()
        };
        let l = build_laplacian(&hg, &cfg).unwrap();
        let n = l.shape().0;
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (l.get(i, j) - l.get(j, i)).abs() < 1e-12,
                    "L[{i},{j}] != L[{j},{i}]"
                );
            }
        }
    }

    #[test]
    fn test_normalized_laplacian() {
        let hg = small_hg();
        let cfg = LaplacianConfig {
            force_method: Some(LaplacianMethod::CliqueExpansion),
            ..Default::default()
        };
        let ln = build_normalized_laplacian(&hg, &cfg).unwrap();
        let diag = ln.diagonal();
        // For a connected graph, diagonal entries of the normalised Laplacian
        // should be 1 (or very close).
        for (i, &d) in diag.iter().enumerate() {
            assert!(
                (d - 1.0).abs() < 1e-6,
                "L_norm[{i},{i}] = {d}, expected ~1"
            );
        }
    }

    #[test]
    fn test_normalized_laplacian_zero_degree_regularization() {
        // Vertex 3 is isolated → zero degree.
        let mut hg = Hypergraph::new(4);
        hg.add_edge(vec![0, 1, 2], 3.0);
        let cfg = LaplacianConfig {
            regularization: 1e-8,
            force_method: Some(LaplacianMethod::CliqueExpansion),
            ..Default::default()
        };
        let ln = build_normalized_laplacian(&hg, &cfg).unwrap();
        // Should not produce NaN or Inf.
        for i in 0..4 {
            for j in 0..4 {
                let v = ln.get(i, j);
                assert!(v.is_finite(), "L_norm[{i},{j}] is not finite: {v}");
            }
        }
    }

    #[test]
    fn test_is_laplacian_valid_true() {
        let hg = small_hg();
        let cfg = LaplacianConfig::default();
        let l = build_laplacian(&hg, &cfg).unwrap();
        assert!(is_laplacian_valid(&l));
    }

    #[test]
    fn test_is_laplacian_valid_false_asymmetric() {
        // Construct a non-symmetric matrix.
        let mut coo = CooMatrix::new(2, 2);
        coo.push(0, 0, 1.0);
        coo.push(0, 1, -1.0);
        coo.push(1, 1, 1.0);
        // Missing (1,0) entry → asymmetric.
        let mat = coo.to_csr();
        assert!(!is_laplacian_valid(&mat));
    }

    #[test]
    fn test_gershgorin_bounds() {
        let hg = small_hg();
        let cfg = LaplacianConfig::default();
        let l = build_laplacian(&hg, &cfg).unwrap();
        let (lo, hi) = laplacian_spectrum_bounds(&l);
        // Laplacian is PSD so smallest eigenvalue ≥ 0, Gershgorin lower bound
        // can be negative but upper bound should be positive.
        assert!(hi > 0.0);
        // For a connected graph, smallest eigenvalue = 0 so Gershgorin lower
        // bound should be ≤ 0.
        assert!(lo <= 1e-10);
    }

    #[test]
    fn test_compute_vertex_degrees() {
        let hg = small_hg();
        let cfg = LaplacianConfig::default();
        let l = build_laplacian(&hg, &cfg).unwrap();
        let deg = compute_vertex_degrees(&l);
        assert_eq!(deg.len(), 3);
        for &d in &deg {
            assert!(d >= 0.0);
        }
    }

    #[test]
    fn test_empty_hypergraph_error() {
        let hg = Hypergraph::new(0);
        let cfg = LaplacianConfig::default();
        assert!(build_laplacian(&hg, &cfg).is_err());
    }

    #[test]
    fn test_no_edges_zero_laplacian() {
        let hg = Hypergraph::new(5);
        let cfg = LaplacianConfig::default();
        let l = build_laplacian(&hg, &cfg).unwrap();
        assert_eq!(l.nnz(), 0);
    }

    #[test]
    fn test_auto_dispatch_small_edges() {
        let hg = small_hg();
        let cfg = LaplacianConfig {
            clique_threshold: 200,
            ..Default::default()
        };
        let method = select_method(&hg, &cfg);
        assert_eq!(method, LaplacianMethod::CliqueExpansion);
    }

    #[test]
    fn test_auto_dispatch_large_edges() {
        let mut hg = Hypergraph::new(300);
        hg.add_edge((0..300).collect(), 1.0);
        let cfg = LaplacianConfig {
            clique_threshold: 200,
            ..Default::default()
        };
        let method = select_method(&hg, &cfg);
        assert_eq!(method, LaplacianMethod::Incidence);
    }

    #[test]
    fn test_spectrum_bounds_single_vertex() {
        let mut hg = Hypergraph::new(1);
        hg.add_edge(vec![0], 1.0);
        let cfg = LaplacianConfig {
            force_method: Some(LaplacianMethod::Incidence),
            ..Default::default()
        };
        let l = build_laplacian(&hg, &cfg).unwrap();
        let (lo, hi) = laplacian_spectrum_bounds(&l);
        // 1×1 Laplacian, single vertex with self-loop degree.
        assert!(lo.is_finite());
        assert!(hi.is_finite());
    }
}
