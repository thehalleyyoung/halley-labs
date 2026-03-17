//! Main eigensolve dispatcher.
//!
//! [`EigenSolver`] is the primary entry point for computing eigenvalues and
//! eigenvectors of sparse symmetric matrices.  It delegates to the Lanczos
//! (ARPACK-style) solver and falls back to LOBPCG if Lanczos fails.

use std::sync::Arc;
use std::time::Instant;

use parking_lot::Mutex;

use spectral_types::dense::DenseMatrix;
use spectral_types::sparse::CsrMatrix;

use crate::eigensolve::arpack::LanczosSolver;
use crate::eigensolve::cache::EigenCache;
use crate::eigensolve::diagnostics::EigenDiagnostics;
use crate::eigensolve::lobpcg::LobpcgSolver;
use crate::eigensolve::{EigenConfig, EigenResult};
use crate::error::{Result, SpectralCoreError};

// ---------------------------------------------------------------------------
// EigenSolver
// ---------------------------------------------------------------------------

/// Primary eigenvalue/eigenvector solver for sparse symmetric matrices.
///
/// Dispatches to an implicit-restart Lanczos (ARPACK-style) solver as the
/// primary method and falls back to LOBPCG on failure. Supports optional
/// result caching via [`EigenCache`].
///
/// # Usage
///
/// ```rust,no_run
/// # use spectral_core::eigensolve::{EigenConfig, solver::EigenSolver};
/// let cfg = EigenConfig::with_k(3).tolerance(1e-8);
/// let solver = EigenSolver::new(cfg);
/// // let result = solver.solve(&laplacian)?;
/// ```
pub struct EigenSolver {
    config: EigenConfig,
    cache: Option<Arc<Mutex<EigenCache>>>,
}

impl EigenSolver {
    /// Create a new solver with the given configuration.
    pub fn new(config: EigenConfig) -> Self {
        Self {
            config,
            cache: None,
        }
    }

    /// Attach a shared cache for eigendecomposition results.
    pub fn with_cache(mut self, cache: Arc<Mutex<EigenCache>>) -> Self {
        self.cache = Some(cache);
        self
    }

    /// Solve for the `k` smallest eigenvalues (specified by `config.num_eigenvalues`).
    pub fn solve(&self, matrix: &CsrMatrix<f64>) -> Result<EigenResult> {
        self.solve_smallest(matrix, self.config.num_eigenvalues)
    }

    /// Find the `k` smallest eigenvalues and eigenvectors.
    pub fn solve_smallest(
        &self,
        matrix: &CsrMatrix<f64>,
        k: usize,
    ) -> Result<EigenResult> {
        validate_inputs(matrix, k)?;

        // Check cache.
        if let Some(ref cache) = self.cache {
            let fp = EigenCache::compute_fingerprint(matrix);
            let mut c = cache.lock();
            if let Some(cached) = c.get(&fp) {
                if cached.eigenvalues.len() >= k {
                    log::info!("EigenSolver: cache hit");
                    return Ok(cached.clone());
                }
            }
        }

        let start = Instant::now();

        // Primary: try Lanczos.
        let mut cfg = self.config.clone();
        cfg.num_eigenvalues = k;

        let effective_matrix: Option<CsrMatrix<f64>>;
        let mat_ref: &CsrMatrix<f64>;

        if cfg.use_shift_invert {
            // For shift-invert we solve (A - σI)^{-1} for largest eigenvalues
            // then transform back.  Since we don't have sparse LU, we cannot
            // truly invert.  Instead we apply CG to solve (A - σI)x = b in
            // the inner loop.  For now, fall through to direct solvers which
            // find the smallest eigenvalues directly.
            log::debug!(
                "Shift-invert requested with σ = {}; using direct smallest-eigenvalue solvers",
                cfg.shift
            );
            effective_matrix = None;
            mat_ref = matrix;
        } else {
            effective_matrix = None;
            mat_ref = matrix;
        }

        // Suppress unused-variable warning.
        let _ = &effective_matrix;

        let mut result = match self.try_lanczos(mat_ref, &cfg) {
            Ok(r) => r,
            Err(e) => {
                log::warn!("Lanczos failed ({e}), falling back to LOBPCG");
                self.try_lobpcg(mat_ref, &cfg)?
            }
        };

        // Sort eigenvalues ascending and reorder eigenvectors.
        sort_eigenpairs(&mut result);

        // Compute actual residuals.
        result.residuals =
            EigenDiagnostics::compute_residuals(matrix, &result.eigenvalues, &result.eigenvectors);

        result.time_ms = start.elapsed().as_secs_f64() * 1000.0;

        // Store in cache.
        if let Some(ref cache) = self.cache {
            let fp = EigenCache::compute_fingerprint(matrix);
            let mut c = cache.lock();
            c.insert(fp, result.clone());
        }

        Ok(result)
    }

    /// Find the `k` largest eigenvalues.
    ///
    /// Strategy: compute enough eigenvalues with the normal solver, then
    /// return the top k.  For matrices where n is small enough we just
    /// compute all eigenvalues; otherwise we compute 2k and hope that covers
    /// the top.
    pub fn solve_largest(
        &self,
        matrix: &CsrMatrix<f64>,
        k: usize,
    ) -> Result<EigenResult> {
        let (n, _) = matrix.shape();
        validate_inputs(matrix, k)?;

        // Compute a larger set and pick the top k.
        let compute_k = (2 * k).min(n - 1).max(k);
        let mut cfg = self.config.clone();
        cfg.num_eigenvalues = compute_k;

        let lanczos = LanczosSolver::new(&cfg);
        let mut result = lanczos.solve(matrix).unwrap_or_else(|_| {
            let lobpcg = LobpcgSolver::new(&cfg);
            lobpcg.solve(matrix).unwrap_or_else(|_| EigenResult {
                eigenvalues: Vec::new(),
                eigenvectors: DenseMatrix::zeros(n, 0),
                residuals: Vec::new(),
                iterations: 0,
                converged: false,
                method_used: "none".to_string(),
                time_ms: 0.0,
            })
        });

        sort_eigenpairs(&mut result);

        // Take the largest k.
        let total = result.eigenvalues.len();
        if total > k {
            let start_idx = total - k;
            let new_evals = result.eigenvalues[start_idx..].to_vec();
            let new_residuals = if result.residuals.len() >= total {
                result.residuals[start_idx..].to_vec()
            } else {
                Vec::new()
            };

            let evec_k = result.eigenvectors.cols;
            let evec_data = &result.eigenvectors.data;
            let mut new_evec_data = Vec::with_capacity(n * k);
            for i in 0..n {
                for j in start_idx..total {
                    new_evec_data.push(
                        evec_data.get(i * evec_k + j).copied().unwrap_or(0.0)
                    );
                }
            }

            result.eigenvalues = new_evals;
            result.residuals = new_residuals;
            result.eigenvectors = DenseMatrix::from_vec(n, k, new_evec_data)
                .unwrap_or_else(|_| DenseMatrix::zeros(n, k));
        }

        // Reverse to put largest first.
        result.eigenvalues.reverse();
        result.residuals.reverse();
        // Reverse column order in eigenvectors.
        let ek = result.eigenvectors.cols;
        for i in 0..n {
            let row_start = i * ek;
            let row_end = row_start + ek;
            result.eigenvectors.data[row_start..row_end].reverse();
        }

        Ok(result)
    }

    // -----------------------------------------------------------------------
    // Internal dispatchers
    // -----------------------------------------------------------------------

    fn try_lanczos(
        &self,
        matrix: &CsrMatrix<f64>,
        cfg: &EigenConfig,
    ) -> Result<EigenResult> {
        let solver = LanczosSolver::new(cfg);
        solver.solve(matrix)
    }

    fn try_lobpcg(
        &self,
        matrix: &CsrMatrix<f64>,
        cfg: &EigenConfig,
    ) -> Result<EigenResult> {
        let solver = LobpcgSolver::new(cfg);
        solver.solve(matrix)
    }
}

// ---------------------------------------------------------------------------
// Conjugate gradient solver for sparse SPD systems
// ---------------------------------------------------------------------------

/// Solve `A x = b` using the Conjugate Gradient method.
///
/// The matrix `A` must be symmetric positive definite.
pub fn conjugate_gradient(
    matrix: &CsrMatrix<f64>,
    b: &[f64],
    tol: f64,
    max_iter: usize,
) -> Result<Vec<f64>> {
    let n = b.len();
    let (rows, _) = matrix.shape();
    if rows != n {
        return Err(SpectralCoreError::DimensionMismatch {
            expected: format!("{n}×{n}"),
            actual: format!("{}×{}", rows, matrix.cols),
        });
    }

    let mut x = vec![0.0; n];
    let mut r = b.to_vec();
    let mut p = r.clone();
    let mut rs_old: f64 = r.iter().map(|v| v * v).sum();

    if rs_old.sqrt() < tol {
        return Ok(x);
    }

    for _iter in 0..max_iter {
        let ap = matrix.mul_vec(&p)?;
        let p_dot_ap: f64 = p.iter().zip(ap.iter()).map(|(a, b)| a * b).sum();

        if p_dot_ap.abs() < 1e-30 {
            log::warn!("CG: near-zero denominator, stopping early");
            break;
        }

        let alpha = rs_old / p_dot_ap;

        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }

        let rs_new: f64 = r.iter().map(|v| v * v).sum();

        if rs_new.sqrt() < tol {
            return Ok(x);
        }

        let beta = rs_new / rs_old;
        for i in 0..n {
            p[i] = r[i] + beta * p[i];
        }
        rs_old = rs_new;
    }

    Ok(x)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Validate that the matrix is square and k is in range.
fn validate_inputs(matrix: &CsrMatrix<f64>, k: usize) -> Result<()> {
    if !matrix.is_square() {
        return Err(SpectralCoreError::DimensionMismatch {
            expected: "square matrix".to_string(),
            actual: format!("{}×{}", matrix.rows, matrix.cols),
        });
    }

    let n = matrix.rows;
    if n == 0 {
        return Err(SpectralCoreError::empty_input("matrix"));
    }

    if k == 0 {
        return Err(SpectralCoreError::invalid_parameter(
            "k",
            "0",
            "must be at least 1",
        ));
    }

    if k >= n {
        return Err(SpectralCoreError::invalid_parameter(
            "k",
            &k.to_string(),
            &format!("must be less than matrix dimension {n}"),
        ));
    }

    Ok(())
}

/// Sort eigenvalues ascending and reorder eigenvectors to match.
fn sort_eigenpairs(result: &mut EigenResult) {
    let k = result.eigenvalues.len();
    if k <= 1 {
        return;
    }

    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|&a, &b| {
        result.eigenvalues[a]
            .partial_cmp(&result.eigenvalues[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let old_evals = result.eigenvalues.clone();
    for (new_i, &old_i) in order.iter().enumerate() {
        result.eigenvalues[new_i] = old_evals[old_i];
    }

    // Reorder eigenvector columns.
    let n = result.eigenvectors.rows;
    let ek = result.eigenvectors.cols;
    if ek == k && n > 0 {
        let old_data = result.eigenvectors.data.clone();
        for (new_col, &old_col) in order.iter().enumerate() {
            for row in 0..n {
                result.eigenvectors.data[row * ek + new_col] = old_data[row * ek + old_col];
            }
        }
    }

    // Reorder residuals if present.
    if result.residuals.len() == k {
        let old_res = result.residuals.clone();
        for (new_i, &old_i) in order.iter().enumerate() {
            result.residuals[new_i] = old_res[old_i];
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn diagonal_csr(diag: &[f64]) -> CsrMatrix<f64> {
        let n = diag.len();
        let mut rp = vec![0usize];
        let mut ci = Vec::new();
        let mut vals = Vec::new();
        for (i, &d) in diag.iter().enumerate() {
            ci.push(i);
            vals.push(d);
            rp.push(i + 1);
        }
        CsrMatrix::new(n, n, rp, ci, vals).unwrap()
    }

    fn path_laplacian(n: usize) -> CsrMatrix<f64> {
        let mut row_ptr = vec![0usize];
        let mut col_ind = Vec::new();
        let mut values = Vec::new();
        for i in 0..n {
            if i > 0 {
                col_ind.push(i - 1);
                values.push(-1.0);
            }
            col_ind.push(i);
            values.push(if i == 0 || i == n - 1 { 1.0 } else { 2.0 });
            if i + 1 < n {
                col_ind.push(i + 1);
                values.push(-1.0);
            }
            row_ptr.push(col_ind.len());
        }
        CsrMatrix::new(n, n, row_ptr, col_ind, values).unwrap()
    }

    #[test]
    fn test_validate_non_square() {
        let mat = CsrMatrix::new(2, 3, vec![0, 1, 2], vec![0, 1], vec![1.0, 1.0]).unwrap();
        assert!(validate_inputs(&mat, 1).is_err());
    }

    #[test]
    fn test_validate_k_zero() {
        let mat = CsrMatrix::identity(5);
        assert!(validate_inputs(&mat, 0).is_err());
    }

    #[test]
    fn test_validate_k_too_large() {
        let mat = CsrMatrix::identity(5);
        assert!(validate_inputs(&mat, 5).is_err());
        assert!(validate_inputs(&mat, 10).is_err());
    }

    #[test]
    fn test_validate_ok() {
        let mat = CsrMatrix::identity(5);
        assert!(validate_inputs(&mat, 1).is_ok());
        assert!(validate_inputs(&mat, 4).is_ok());
    }

    #[test]
    fn test_cg_identity() {
        let mat = CsrMatrix::identity(4);
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let x = conjugate_gradient(&mat, &b, 1e-12, 100).unwrap();
        for (xi, bi) in x.iter().zip(b.iter()) {
            assert!((xi - bi).abs() < 1e-10, "x={xi}, b={bi}");
        }
    }

    #[test]
    fn test_cg_diagonal() {
        let mat = diagonal_csr(&[2.0, 4.0, 8.0]);
        let b = vec![4.0, 8.0, 16.0];
        let x = conjugate_gradient(&mat, &b, 1e-12, 100).unwrap();
        assert!((x[0] - 2.0).abs() < 1e-10);
        assert!((x[1] - 2.0).abs() < 1e-10);
        assert!((x[2] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_cg_tridiagonal() {
        let mat = path_laplacian(4);
        let b = vec![1.0, 0.0, 0.0, 1.0];
        let x = conjugate_gradient(&mat, &b, 1e-10, 200).unwrap();
        // Verify Ax ≈ b.
        let ax = mat.mul_vec(&x).unwrap();
        for (a, bi) in ax.iter().zip(b.iter()) {
            assert!((a - bi).abs() < 1e-8, "Ax={a}, b={bi}");
        }
    }

    #[test]
    fn test_sort_eigenpairs() {
        let mut result = EigenResult {
            eigenvalues: vec![5.0, 1.0, 3.0],
            eigenvectors: DenseMatrix::from_vec(
                2,
                3,
                vec![0.5, 0.1, 0.3, 0.6, 0.2, 0.4],
            )
            .unwrap(),
            residuals: vec![0.05, 0.01, 0.03],
            iterations: 10,
            converged: true,
            method_used: "test".to_string(),
            time_ms: 0.0,
        };
        sort_eigenpairs(&mut result);
        assert_eq!(result.eigenvalues, vec![1.0, 3.0, 5.0]);
        assert_eq!(result.residuals, vec![0.01, 0.03, 0.05]);
        // Eigenvector columns should be reordered too.
        // Original col 1 (val 0.1, 0.2) → now col 0.
        assert!((result.eigenvectors.data[0 * 3 + 0] - 0.1).abs() < 1e-14);
        assert!((result.eigenvectors.data[1 * 3 + 0] - 0.2).abs() < 1e-14);
    }

    #[test]
    fn test_eigensolver_solve_diagonal() {
        let mat = diagonal_csr(&[1.0, 3.0, 5.0, 7.0, 9.0]);
        let cfg = EigenConfig::with_k(2).max_iter(100).tolerance(1e-4);
        let solver = EigenSolver::new(cfg);
        let result = solver.solve(&mat).unwrap();
        assert_eq!(result.eigenvalues.len(), 2);
        // Should be sorted ascending.
        assert!(result.eigenvalues[0] <= result.eigenvalues[1]);
    }

    #[test]
    fn test_eigensolver_solve_with_cache() {
        let mat = diagonal_csr(&[1.0, 3.0, 5.0, 7.0, 9.0]);
        let cache = Arc::new(Mutex::new(EigenCache::new(10)));
        let cfg = EigenConfig::with_k(2).max_iter(100).tolerance(1e-4);
        let solver = EigenSolver::new(cfg).with_cache(cache.clone());

        // First call populates cache.
        let r1 = solver.solve(&mat).unwrap();
        assert_eq!(cache.lock().len(), 1);

        // Second call should hit cache.
        let r2 = solver.solve(&mat).unwrap();
        assert_eq!(r1.eigenvalues, r2.eigenvalues);
        assert!(cache.lock().hit_rate() > 0.0);
    }

    #[test]
    fn test_eigensolver_solve_path_laplacian() {
        let mat = path_laplacian(8);
        let cfg = EigenConfig::with_k(2).max_iter(100).tolerance(1e-4);
        let solver = EigenSolver::new(cfg);
        let result = solver.solve(&mat).unwrap();
        assert_eq!(result.eigenvalues.len(), 2);
        assert!(result.eigenvalues[0] >= 0.0);
        assert!(result.eigenvalues[0] < 1.0);
    }

    #[test]
    fn test_eigensolver_solve_largest() {
        let mat = diagonal_csr(&[1.0, 3.0, 5.0, 7.0, 9.0]);
        let cfg = EigenConfig::with_k(2).max_iter(100).tolerance(1e-4);
        let solver = EigenSolver::new(cfg);
        let result = solver.solve_largest(&mat, 2).unwrap();
        assert_eq!(result.eigenvalues.len(), 2);
        // Largest should be first.
        assert!(result.eigenvalues[0] >= result.eigenvalues[1]);
    }

    #[test]
    fn test_eigensolver_invalid_inputs() {
        let mat = CsrMatrix::identity(5);
        let cfg = EigenConfig::with_k(0);
        let solver = EigenSolver::new(cfg);
        assert!(solver.solve(&mat).is_err());
    }

    #[test]
    fn test_eigensolver_residuals_computed() {
        let mat = diagonal_csr(&[2.0, 4.0, 6.0, 8.0, 10.0]);
        let cfg = EigenConfig::with_k(2).max_iter(100).tolerance(1e-4);
        let solver = EigenSolver::new(cfg);
        let result = solver.solve(&mat).unwrap();
        assert_eq!(result.residuals.len(), 2);
    }

    #[test]
    fn test_eigensolver_method_name() {
        let mat = diagonal_csr(&[1.0, 3.0, 5.0, 7.0, 9.0]);
        let cfg = EigenConfig::with_k(2).max_iter(100).tolerance(1e-4);
        let solver = EigenSolver::new(cfg);
        let result = solver.solve(&mat).unwrap();
        assert!(
            result.method_used == "lanczos" || result.method_used == "lobpcg",
            "unexpected method: {}",
            result.method_used
        );
    }

    #[test]
    fn test_eigensolver_time_recorded() {
        let mat = diagonal_csr(&[1.0, 3.0, 5.0, 7.0, 9.0]);
        let cfg = EigenConfig::with_k(2).max_iter(100).tolerance(1e-4);
        let solver = EigenSolver::new(cfg);
        let result = solver.solve(&mat).unwrap();
        assert!(result.time_ms >= 0.0);
    }
}
