//! LOBPCG — Locally Optimal Block Preconditioned Conjugate Gradient.
//!
//! Iterative eigensolver for the *smallest* eigenvalues of large, sparse,
//! symmetric positive semi-definite matrices.  Works well as a fallback when
//! the Lanczos method has convergence difficulties.

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use spectral_types::dense::DenseMatrix;
use spectral_types::sparse::CsrMatrix;

use crate::eigensolve::{EigenConfig, EigenResult};
use crate::error::{Result, SpectralCoreError};

// ---------------------------------------------------------------------------
// LobpcgSolver
// ---------------------------------------------------------------------------

/// LOBPCG eigensolver.
pub struct LobpcgSolver {
    config: EigenConfig,
}

impl LobpcgSolver {
    pub fn new(config: &EigenConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }

    /// Solve for the `k` smallest eigenvalues.
    pub fn solve(&self, matrix: &CsrMatrix<f64>) -> Result<EigenResult> {
        let (n, _) = matrix.shape();
        let k = self.config.num_eigenvalues;

        if k == 0 || k >= n {
            return Err(SpectralCoreError::invalid_parameter(
                "num_eigenvalues",
                &k.to_string(),
                &format!("must be in 1..{n}"),
            ));
        }

        let max_iter = self.config.max_iter;
        let tol = self.config.tolerance;

        // Diagonal preconditioner: T = 1 / diag(A).
        let precond = diagonal_preconditioner(matrix);

        // Initialize X (n × k) with random columns, then orthonormalize.
        let mut x = random_matrix(n, k, 123);
        orthonormalize(&mut x);

        // Compute AX.
        let mut ax = mat_mul_sparse(matrix, &x)?;

        // Initial Rayleigh quotient: eigenvalue estimates from X^T A X.
        let mut lambda = rayleigh_quotient_diag(&x, &ax, k);

        // Previous search direction (n × k); initialised to zeros.
        let mut p = DenseMatrix::zeros(n, k);
        let mut _ap = DenseMatrix::zeros(n, k);
        let mut has_p = false;

        let mut converged = false;

        for iter in 0..max_iter {
            // Residuals: R = AX - X * diag(lambda)
            let r = compute_residual_block(&ax, &x, &lambda);

            // Check convergence.
            let res_norms = column_norms(&r);
            let max_res = res_norms
                .iter()
                .zip(lambda.iter())
                .map(|(&rn, &lam)| rn / lam.abs().max(1.0))
                .fold(0.0_f64, f64::max);

            if self.config.verbose {
                log::debug!("LOBPCG iter {iter}: max_relative_residual = {max_res:.2e}");
            }

            if max_res < tol {
                converged = true;
                log::info!("LOBPCG converged at iteration {iter}");
                break;
            }

            // Apply preconditioner: W = T^{-1} R (element-wise).
            let mut w = apply_preconditioner(&r, &precond);

            // Build search basis S.
            // iter > 0 and has_p: S = [X, W, P] (n × 3k)
            // else:                S = [X, W]    (n × 2k)
            let (s, as_mat) = if has_p {
                let mut s = concat_columns(&x, &w);
                s = concat_columns(&s, &p);
                orthonormalize(&mut s);
                let a_s = mat_mul_sparse(matrix, &s)?;
                (s, a_s)
            } else {
                orthonormalize(&mut w);
                let mut s = concat_columns(&x, &w);
                orthonormalize(&mut s);
                let a_s = mat_mul_sparse(matrix, &s)?;
                (s, a_s)
            };

            let s_cols = s.cols;

            // Projected eigenproblem: S^T A S * c = lambda * c
            let a_proj = project_symmetric(&s, &as_mat);

            // Solve the small dense eigenproblem.
            let (small_evals, small_evecs) =
                jacobi_eigensolve_sorted(&a_proj, 500, 1e-14)?;

            // Select the k smallest.
            let k_actual = k.min(small_evals.len());
            lambda = small_evals[..k_actual].to_vec();

            // Reconstruct X = S * C_k and P = S * C_p.
            let mut new_x = DenseMatrix::zeros(n, k_actual);
            let mut new_p = DenseMatrix::zeros(n, k_actual);

            for col in 0..k_actual {
                for i in 0..n {
                    let mut vx = 0.0;
                    let mut vp = 0.0;
                    for j in 0..s_cols {
                        let sij = s.data[i * s_cols + j];
                        let cjc = small_evecs
                            .data
                            .get(j * small_evecs.cols + col)
                            .copied()
                            .unwrap_or(0.0);
                        vx += sij * cjc;
                        // P direction: contribution from W and P parts only
                        // (indices >= k in the basis).
                        if j >= k {
                            vp += sij * cjc;
                        }
                    }
                    new_x.data[i * k_actual + col] = vx;
                    new_p.data[i * k_actual + col] = vp;
                }
            }

            x = new_x;
            p = new_p;
            ax = mat_mul_sparse(matrix, &x)?;
            _ap = mat_mul_sparse(matrix, &p)?;
            has_p = true;
        }

        // Build final result.
        let residuals = {
            let r = compute_residual_block(&ax, &x, &lambda);
            column_norms(&r)
        };

        Ok(EigenResult {
            eigenvalues: lambda,
            eigenvectors: x,
            residuals,
            iterations: max_iter,
            converged,
            method_used: "lobpcg".to_string(),
            time_ms: 0.0,
        })
    }
}

// ---------------------------------------------------------------------------
// Dense helpers
// ---------------------------------------------------------------------------

/// Modified Gram-Schmidt orthonormalization (in-place, column-wise).
pub fn orthonormalize(v: &mut DenseMatrix<f64>) {
    let (n, k) = v.shape();
    for j in 0..k {
        // Subtract projections onto previous columns.
        for prev in 0..j {
            let dot: f64 = (0..n)
                .map(|i| v.data[i * k + j] * v.data[i * k + prev])
                .sum();
            for i in 0..n {
                v.data[i * k + j] -= dot * v.data[i * k + prev];
            }
        }
        // Normalize.
        let norm: f64 = (0..n).map(|i| v.data[i * k + j].powi(2)).sum::<f64>().sqrt();
        if norm > 1e-14 {
            for i in 0..n {
                v.data[i * k + j] /= norm;
            }
        }
    }
}

/// Rayleigh quotient diagonal: λ_j = (x_j^T A x_j) / (x_j^T x_j).
fn rayleigh_quotient_diag(
    x: &DenseMatrix<f64>,
    ax: &DenseMatrix<f64>,
    k: usize,
) -> Vec<f64> {
    let n = x.rows;
    (0..k)
        .map(|j| {
            let dot: f64 = (0..n).map(|i| x.data[i * k + j] * ax.data[i * k + j]).sum();
            let norm_sq: f64 = (0..n).map(|i| x.data[i * k + j].powi(2)).sum();
            if norm_sq > 1e-30 {
                dot / norm_sq
            } else {
                0.0
            }
        })
        .collect()
}

/// R = AX - X * diag(lambda).
fn compute_residual_block(
    ax: &DenseMatrix<f64>,
    x: &DenseMatrix<f64>,
    lambda: &[f64],
) -> DenseMatrix<f64> {
    let (n, k) = x.shape();
    let mut r = DenseMatrix::zeros(n, k);
    for j in 0..k {
        let lam = if j < lambda.len() { lambda[j] } else { 0.0 };
        for i in 0..n {
            r.data[i * k + j] = ax.data[i * k + j] - lam * x.data[i * k + j];
        }
    }
    r
}

/// Column-wise L2 norms.
fn column_norms(m: &DenseMatrix<f64>) -> Vec<f64> {
    let (n, k) = m.shape();
    (0..k)
        .map(|j| {
            (0..n)
                .map(|i| m.data[i * k + j].powi(2))
                .sum::<f64>()
                .sqrt()
        })
        .collect()
}

/// Diagonal preconditioner: T_i = 1 / max(|A_{ii}|, eps).
pub fn diagonal_preconditioner(a: &CsrMatrix<f64>) -> Vec<f64> {
    let diag = a.diagonal();
    diag.iter()
        .map(|&d| {
            let abs_d = d.abs();
            if abs_d > 1e-14 {
                1.0 / abs_d
            } else {
                1.0
            }
        })
        .collect()
}

/// Apply diagonal preconditioner element-wise to each column.
fn apply_preconditioner(r: &DenseMatrix<f64>, precond: &[f64]) -> DenseMatrix<f64> {
    let (n, k) = r.shape();
    let mut w = DenseMatrix::zeros(n, k);
    for i in 0..n {
        let pi = if i < precond.len() { precond[i] } else { 1.0 };
        for j in 0..k {
            w.data[i * k + j] = r.data[i * k + j] * pi;
        }
    }
    w
}

/// Horizontal concatenation: [A | B].
fn concat_columns(a: &DenseMatrix<f64>, b: &DenseMatrix<f64>) -> DenseMatrix<f64> {
    let n = a.rows;
    let ka = a.cols;
    let kb = b.cols;
    let k = ka + kb;
    let mut data = vec![0.0; n * k];
    for i in 0..n {
        for j in 0..ka {
            data[i * k + j] = a.data[i * ka + j];
        }
        for j in 0..kb {
            data[i * k + ka + j] = b.data[i * kb + j];
        }
    }
    DenseMatrix::from_vec(n, k, data).unwrap_or_else(|_| DenseMatrix::zeros(n, k))
}

/// Compute S^T * AS for symmetric projected matrix.
fn project_symmetric(s: &DenseMatrix<f64>, as_mat: &DenseMatrix<f64>) -> DenseMatrix<f64> {
    let (n, k) = s.shape();
    let mut proj = DenseMatrix::zeros(k, k);
    for i in 0..k {
        for j in i..k {
            let dot: f64 = (0..n)
                .map(|r| s.data[r * k + i] * as_mat.data[r * k + j])
                .sum();
            proj.data[i * k + j] = dot;
            proj.data[j * k + i] = dot;
        }
    }
    proj
}

/// Sparse matrix × dense column-block multiply.
fn mat_mul_sparse(
    a: &CsrMatrix<f64>,
    x: &DenseMatrix<f64>,
) -> Result<DenseMatrix<f64>> {
    let (n, _) = a.shape();
    let k = x.cols;
    let mut result = DenseMatrix::zeros(n, k);
    for col in 0..k {
        let v: Vec<f64> = (0..n).map(|i| x.data[i * k + col]).collect();
        let av = a.mul_vec(&v)?;
        for i in 0..n {
            result.data[i * k + col] = av[i];
        }
    }
    Ok(result)
}

/// Random n×k DenseMatrix with entries in [-0.5, 0.5].
fn random_matrix(n: usize, k: usize, seed: u64) -> DenseMatrix<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let data: Vec<f64> = (0..n * k).map(|_| rng.gen::<f64>() - 0.5).collect();
    DenseMatrix::from_vec(n, k, data).unwrap_or_else(|_| DenseMatrix::zeros(n, k))
}

// ---------------------------------------------------------------------------
// Small dense Jacobi eigensolver
// ---------------------------------------------------------------------------

/// Jacobi eigenvalue algorithm for a small symmetric dense matrix.
///
/// Returns eigenvalues sorted ascending and corresponding column eigenvectors.
pub fn jacobi_eigensolve_sorted(
    a: &DenseMatrix<f64>,
    max_iter: usize,
    tol: f64,
) -> Result<(Vec<f64>, DenseMatrix<f64>)> {
    let (m, mc) = a.shape();
    if m != mc {
        return Err(SpectralCoreError::eigensolve(
            "Jacobi requires a square matrix",
        ));
    }
    if m == 0 {
        return Ok((Vec::new(), DenseMatrix::zeros(0, 0)));
    }
    if m == 1 {
        return Ok((
            vec![a.data[0]],
            DenseMatrix::from_vec(1, 1, vec![1.0])
                .map_err(|e| SpectralCoreError::SpectralTypes(e.into()))?,
        ));
    }

    let mut d = a.data.clone(); // working copy
    let mut v = vec![0.0; m * m]; // eigenvector accumulator = I
    for i in 0..m {
        v[i * m + i] = 1.0;
    }

    for _iter in 0..max_iter {
        // Find largest off-diagonal element.
        let mut max_off = 0.0_f64;
        let mut p = 0;
        let mut q = 1;
        for i in 0..m {
            for j in (i + 1)..m {
                let val = d[i * m + j].abs();
                if val > max_off {
                    max_off = val;
                    p = i;
                    q = j;
                }
            }
        }

        if max_off < tol {
            break;
        }

        // Compute Jacobi rotation angle.
        let app = d[p * m + p];
        let aqq = d[q * m + q];
        let apq = d[p * m + q];

        let theta = if (app - aqq).abs() < 1e-30 {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * apq / (app - aqq)).atan()
        };

        let c = theta.cos();
        let s = theta.sin();

        // Apply rotation to matrix D: D' = G^T D G.
        // Update rows/cols p and q.
        let mut new_row_p = vec![0.0; m];
        let mut new_row_q = vec![0.0; m];
        for j in 0..m {
            new_row_p[j] = c * d[p * m + j] - s * d[q * m + j];
            new_row_q[j] = s * d[p * m + j] + c * d[q * m + j];
        }
        for j in 0..m {
            d[p * m + j] = new_row_p[j];
            d[q * m + j] = new_row_q[j];
        }
        let mut new_col_p = vec![0.0; m];
        let mut new_col_q = vec![0.0; m];
        for i in 0..m {
            new_col_p[i] = c * d[i * m + p] - s * d[i * m + q];
            new_col_q[i] = s * d[i * m + p] + c * d[i * m + q];
        }
        for i in 0..m {
            d[i * m + p] = new_col_p[i];
            d[i * m + q] = new_col_q[i];
        }

        // Accumulate eigenvectors: V = V * G.
        for i in 0..m {
            let vp = v[i * m + p];
            let vq = v[i * m + q];
            v[i * m + p] = c * vp - s * vq;
            v[i * m + q] = s * vp + c * vq;
        }
    }

    // Extract eigenvalues from diagonal.
    let eigenvalues: Vec<f64> = (0..m).map(|i| d[i * m + i]).collect();

    // Sort ascending and reorder eigenvectors.
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| eigenvalues[a].partial_cmp(&eigenvalues[b]).unwrap());

    let sorted_evals: Vec<f64> = order.iter().map(|&i| eigenvalues[i]).collect();
    let mut sorted_evecs_data = vec![0.0; m * m];
    for (new_col, &old_col) in order.iter().enumerate() {
        for row in 0..m {
            sorted_evecs_data[row * m + new_col] = v[row * m + old_col];
        }
    }

    let evecs = DenseMatrix::from_vec(m, m, sorted_evecs_data)
        .map_err(|e| SpectralCoreError::SpectralTypes(e.into()))?;

    Ok((sorted_evals, evecs))
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
    fn test_jacobi_2x2() {
        let a = DenseMatrix::from_vec(2, 2, vec![2.0, -1.0, -1.0, 2.0]).unwrap();
        let (evals, _) = jacobi_eigensolve_sorted(&a, 100, 1e-14).unwrap();
        assert!((evals[0] - 1.0).abs() < 1e-10, "got {}", evals[0]);
        assert!((evals[1] - 3.0).abs() < 1e-10, "got {}", evals[1]);
    }

    #[test]
    fn test_jacobi_diagonal() {
        let a = DenseMatrix::from_vec(3, 3, vec![5.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 8.0])
            .unwrap();
        let (evals, _) = jacobi_eigensolve_sorted(&a, 100, 1e-14).unwrap();
        assert!((evals[0] - 2.0).abs() < 1e-10);
        assert!((evals[1] - 5.0).abs() < 1e-10);
        assert!((evals[2] - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_jacobi_1x1() {
        let a = DenseMatrix::from_vec(1, 1, vec![3.5]).unwrap();
        let (evals, evecs) = jacobi_eigensolve_sorted(&a, 10, 1e-14).unwrap();
        assert_eq!(evals, vec![3.5]);
        assert!((evecs.data[0] - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_jacobi_empty() {
        let a = DenseMatrix::zeros(0, 0);
        let (evals, _) = jacobi_eigensolve_sorted(&a, 10, 1e-14).unwrap();
        assert!(evals.is_empty());
    }

    #[test]
    fn test_orthonormalize() {
        let mut m = DenseMatrix::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        orthonormalize(&mut m);
        // Check orthonormality.
        let n = m.rows;
        let k = m.cols;
        for i in 0..k {
            for j in 0..k {
                let dot: f64 = (0..n)
                    .map(|r| m.data[r * k + i] * m.data[r * k + j])
                    .sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-10,
                    "[{i},{j}] = {dot}, want {expected}"
                );
            }
        }
    }

    #[test]
    fn test_diagonal_preconditioner() {
        let mat = diagonal_csr(&[2.0, 4.0, 8.0]);
        let p = diagonal_preconditioner(&mat);
        assert!((p[0] - 0.5).abs() < 1e-14);
        assert!((p[1] - 0.25).abs() < 1e-14);
        assert!((p[2] - 0.125).abs() < 1e-14);
    }

    #[test]
    fn test_concat_columns() {
        let a = DenseMatrix::from_vec(2, 1, vec![1.0, 2.0]).unwrap();
        let b = DenseMatrix::from_vec(2, 1, vec![3.0, 4.0]).unwrap();
        let c = concat_columns(&a, &b);
        assert_eq!(c.shape(), (2, 2));
        assert!((c.data[0] - 1.0).abs() < 1e-14);
        assert!((c.data[1] - 3.0).abs() < 1e-14);
        assert!((c.data[2] - 2.0).abs() < 1e-14);
        assert!((c.data[3] - 4.0).abs() < 1e-14);
    }

    #[test]
    fn test_lobpcg_solve_diagonal() {
        let mat = diagonal_csr(&[1.0, 3.0, 5.0, 7.0, 9.0]);
        let cfg = EigenConfig::with_k(2).max_iter(200).tolerance(1e-4);
        let solver = LobpcgSolver::new(&cfg);
        let result = solver.solve(&mat).unwrap();
        assert_eq!(result.eigenvalues.len(), 2);
        // Should find the two smallest.
        assert!(
            (result.eigenvalues[0] - 1.0).abs() < 0.5,
            "got {}",
            result.eigenvalues[0]
        );
        assert!(
            (result.eigenvalues[1] - 3.0).abs() < 0.5,
            "got {}",
            result.eigenvalues[1]
        );
    }

    #[test]
    fn test_lobpcg_solve_path_laplacian() {
        let mat = path_laplacian(8);
        let cfg = EigenConfig::with_k(2).max_iter(300).tolerance(1e-4);
        let solver = LobpcgSolver::new(&cfg);
        let result = solver.solve(&mat).unwrap();
        assert_eq!(result.eigenvalues.len(), 2);
        // Smallest eigenvalue of path Laplacian is positive and < 1.
        assert!(result.eigenvalues[0] < 1.0);
    }

    #[test]
    fn test_lobpcg_invalid_k() {
        let mat = CsrMatrix::identity(3);
        let cfg = EigenConfig::with_k(0);
        let solver = LobpcgSolver::new(&cfg);
        assert!(solver.solve(&mat).is_err());
    }

    #[test]
    fn test_lobpcg_invalid_k_too_large() {
        let mat = CsrMatrix::identity(3);
        let cfg = EigenConfig::with_k(3);
        let solver = LobpcgSolver::new(&cfg);
        assert!(solver.solve(&mat).is_err());
    }

    #[test]
    fn test_column_norms() {
        let m = DenseMatrix::from_vec(2, 2, vec![3.0, 0.0, 4.0, 0.0]).unwrap();
        let norms = column_norms(&m);
        assert!((norms[0] - 5.0).abs() < 1e-14);
        assert!(norms[1].abs() < 1e-14);
    }

    #[test]
    fn test_jacobi_eigenvectors_orthogonal() {
        let a = DenseMatrix::from_vec(
            3,
            3,
            vec![4.0, -1.0, 0.0, -1.0, 4.0, -1.0, 0.0, -1.0, 4.0],
        )
        .unwrap();
        let (_, evecs) = jacobi_eigensolve_sorted(&a, 200, 1e-14).unwrap();
        let m = evecs.rows;
        for i in 0..m {
            for j in 0..m {
                let dot: f64 = (0..m)
                    .map(|r| evecs.data[r * m + i] * evecs.data[r * m + j])
                    .sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-6,
                    "[{i},{j}] = {dot}, expected {expected}"
                );
            }
        }
    }
}
