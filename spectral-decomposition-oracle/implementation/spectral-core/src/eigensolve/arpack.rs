//! Implicitly Restarted Lanczos Method (IRLM) — an ARPACK-style eigensolver
//! for real symmetric sparse matrices.
//!
//! Computes the *smallest* eigenvalues and corresponding eigenvectors of a
//! symmetric matrix via the three-term Lanczos recurrence, with optional full
//! reorthogonalisation and QR-based implicit restarts.

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use spectral_types::dense::DenseMatrix;
use spectral_types::sparse::CsrMatrix;

use crate::eigensolve::{EigenConfig, EigenResult};
use crate::error::{Result, SpectralCoreError};

// ---------------------------------------------------------------------------
// LanczosSolver
// ---------------------------------------------------------------------------

/// Implicitly restarted Lanczos eigensolver.
pub struct LanczosSolver {
    config: EigenConfig,
}

impl LanczosSolver {
    pub fn new(config: &EigenConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }

    /// Solve for the `k` smallest eigenvalues of a symmetric matrix.
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

        // Subspace dimension m: at least 2k, at most n.
        let m = (2 * k + 10).min(n);
        let max_restarts = self.config.max_iter;
        let tol = self.config.tolerance;

        // Random starting vector.
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut v0: Vec<f64> = (0..n).map(|_| rng.gen::<f64>() - 0.5).collect();
        let nrm = vec_norm(&v0);
        if nrm < 1e-15 {
            return Err(SpectralCoreError::numerical_instability(
                "random starting vector has zero norm",
            ));
        }
        vec_scale(&mut v0, 1.0 / nrm);

        let mut converged_eigenvalues: Option<Vec<f64>> = None;
        let mut converged_ritz_vecs: Option<DenseMatrix<f64>> = None;
        let mut final_converged = false;

        for restart in 0..max_restarts {
            // Build Krylov subspace of dimension m.
            let (alphas, betas, v_basis) = self.lanczos_iteration(matrix, &v0, m)?;

            let actual_m = alphas.len();

            // Eigenvalues of the tridiagonal matrix.
            let (tri_evals, tri_evecs) =
                qr_tridiagonal_full(&alphas, &betas, 1000, 1e-14)?;

            // Sort eigenvalues ascending and pick the k smallest.
            let mut order: Vec<usize> = (0..tri_evals.len()).collect();
            order.sort_by(|&a, &b| tri_evals[a].partial_cmp(&tri_evals[b]).unwrap());

            let wanted: Vec<usize> = order.iter().copied().take(k).collect();

            // Compute Ritz vectors in the original space: v_i = V * s_i.
            let ritz_vecs =
                self.compute_ritz_vectors(&v_basis, &tri_evecs, &wanted, n, actual_m);
            let ritz_vals: Vec<f64> = wanted.iter().map(|&i| tri_evals[i]).collect();

            // Check convergence via residual bounds.
            // For Lanczos: residual bound ≈ |β_m * s_i(m)| for each Ritz pair.
            let beta_m = if actual_m > 0 && actual_m <= betas.len() {
                betas[actual_m - 1].abs()
            } else if !betas.is_empty() {
                betas[betas.len() - 1].abs()
            } else {
                0.0
            };

            let mut all_converged = true;
            for (idx, &wi) in wanted.iter().enumerate() {
                let s_last = if actual_m > 0 {
                    tri_evecs
                        .data
                        .get((actual_m - 1) * tri_evecs.cols + wi)
                        .copied()
                        .unwrap_or(0.0)
                } else {
                    0.0
                };
                let bound = beta_m * s_last.abs();
                let threshold = tol * ritz_vals[idx].abs().max(1.0);
                if bound > threshold {
                    all_converged = false;
                    break;
                }
            }

            if all_converged || restart + 1 == max_restarts {
                converged_eigenvalues = Some(ritz_vals);
                converged_ritz_vecs = Some(ritz_vecs);
                final_converged = all_converged;
                if all_converged {
                    log::info!(
                        "Lanczos converged after {} restart(s), m={}",
                        restart + 1,
                        actual_m
                    );
                } else {
                    log::warn!(
                        "Lanczos did NOT converge after {} restart(s)",
                        restart + 1
                    );
                }
                break;
            }

            // Implicit restart: use the first wanted Ritz vector as the new
            // starting vector for the next Lanczos factorisation.
            let first_ritz: Vec<f64> = (0..n)
                .map(|i| ritz_vecs.data[i * k + 0])
                .collect();
            let nrm = vec_norm(&first_ritz);
            v0 = first_ritz;
            if nrm > 1e-15 {
                vec_scale(&mut v0, 1.0 / nrm);
            }
        }

        let eigenvalues = converged_eigenvalues.ok_or_else(|| {
            SpectralCoreError::eigensolve("Lanczos produced no results")
        })?;
        let eigenvectors = converged_ritz_vecs.ok_or_else(|| {
            SpectralCoreError::eigensolve("Lanczos produced no eigenvectors")
        })?;

        Ok(EigenResult {
            eigenvalues,
            eigenvectors,
            residuals: Vec::new(), // caller computes residuals
            iterations: max_restarts,
            converged: final_converged,
            method_used: "lanczos".to_string(),
            time_ms: 0.0,
        })
    }

    // -----------------------------------------------------------------------
    // Lanczos three-term recurrence
    // -----------------------------------------------------------------------

    /// Run `m` steps of the Lanczos recurrence starting from `v0`.
    ///
    /// Returns `(alphas, betas, V)` where V is an `n × m` matrix whose
    /// columns are the orthonormal Lanczos vectors, alphas has length m,
    /// and betas has length m (betas\[m-1\] is the residual norm after the
    /// last step).
    pub fn lanczos_iteration(
        &self,
        matrix: &CsrMatrix<f64>,
        v0: &[f64],
        m: usize,
    ) -> Result<(Vec<f64>, Vec<f64>, DenseMatrix<f64>)> {
        let n = v0.len();
        let m = m.min(n);

        let mut alphas = Vec::with_capacity(m);
        let mut betas = Vec::with_capacity(m);

        // Store Lanczos vectors column-by-column; convert to row-major at end.
        let mut v_cols: Vec<Vec<f64>> = Vec::with_capacity(m + 1);

        let mut v_curr = v0.to_vec();
        let nrm = vec_norm(&v_curr);
        if nrm < 1e-15 {
            return Err(SpectralCoreError::numerical_instability(
                "starting vector for Lanczos has zero norm",
            ));
        }
        vec_scale(&mut v_curr, 1.0 / nrm);
        v_cols.push(v_curr);

        let mut beta_prev: f64 = 0.0;
        let mut v_prev: Vec<f64> = vec![0.0; n];

        for j in 0..m {
            // w = A * v_j
            let mut w = matrix.mul_vec(&v_cols[j])?;

            // w = w - beta_{j-1} * v_{j-1}
            if j > 0 {
                for i in 0..n {
                    w[i] -= beta_prev * v_prev[i];
                }
            }

            // alpha_j = w · v_j
            let alpha_j = vec_dot(&w, &v_cols[j]);
            alphas.push(alpha_j);

            // w = w - alpha_j * v_j
            for i in 0..n {
                w[i] -= alpha_j * v_cols[j][i];
            }

            // Full reorthogonalisation against all previous Lanczos vectors.
            if self.config.reorthogonalize {
                for prev in &v_cols {
                    let coeff = vec_dot(&w, prev);
                    for i in 0..n {
                        w[i] -= coeff * prev[i];
                    }
                }
            }

            let beta_j = vec_norm(&w);
            betas.push(beta_j);

            // Check for invariant subspace.
            if beta_j < 1e-14 {
                log::debug!("Lanczos: invariant subspace found at step {}", j + 1);
                break;
            }

            // v_{j+1} = w / beta_j
            vec_scale(&mut w, 1.0 / beta_j);
            v_prev = v_cols[j].clone();
            beta_prev = beta_j;
            v_cols.push(w);
        }

        let actual_m = alphas.len();
        // Build DenseMatrix (n × actual_m, row-major).
        let mut v_data = vec![0.0; n * actual_m];
        for j in 0..actual_m {
            for i in 0..n {
                v_data[i * actual_m + j] = v_cols[j][i];
            }
        }
        let v_matrix = DenseMatrix::from_vec(n, actual_m, v_data)
            .map_err(|e| SpectralCoreError::SpectralTypes(e.into()))?;

        Ok((alphas, betas, v_matrix))
    }

    // -----------------------------------------------------------------------
    // Ritz vector computation
    // -----------------------------------------------------------------------

    /// Compute Ritz vectors: for each wanted index, v = V * s_i.
    fn compute_ritz_vectors(
        &self,
        v_basis: &DenseMatrix<f64>,
        tri_evecs: &DenseMatrix<f64>,
        wanted: &[usize],
        n: usize,
        m: usize,
    ) -> DenseMatrix<f64> {
        let k = wanted.len();
        let mut data = vec![0.0; n * k];

        for (col, &wi) in wanted.iter().enumerate() {
            for i in 0..n {
                let mut val = 0.0;
                for j in 0..m {
                    let v_ij = v_basis.data[i * m + j];
                    let s_jw = tri_evecs
                        .data
                        .get(j * tri_evecs.cols + wi)
                        .copied()
                        .unwrap_or(0.0);
                    val += v_ij * s_jw;
                }
                data[i * k + col] = val;
            }
        }

        DenseMatrix::from_vec(n, k, data).unwrap_or_else(|_| DenseMatrix::zeros(n, k))
    }
}

// ---------------------------------------------------------------------------
// Tridiagonal QR algorithm (Wilkinson shift) with eigenvector accumulation
// ---------------------------------------------------------------------------

/// Compute eigenvalues AND eigenvectors of a symmetric tridiagonal matrix.
///
/// The tridiagonal matrix T has `alpha[i]` on the diagonal and `beta[i]` on
/// the sub/super-diagonal.
///
/// Returns eigenvalues (unsorted) and an `m × m` eigenvector matrix.
pub fn qr_tridiagonal_full(
    alpha: &[f64],
    beta: &[f64],
    max_iter: usize,
    tol: f64,
) -> Result<(Vec<f64>, DenseMatrix<f64>)> {
    let m = alpha.len();
    if m == 0 {
        return Ok((Vec::new(), DenseMatrix::zeros(0, 0)));
    }
    if m == 1 {
        let evecs = DenseMatrix::from_vec(1, 1, vec![1.0])
            .map_err(|e| SpectralCoreError::SpectralTypes(e.into()))?;
        return Ok((vec![alpha[0]], evecs));
    }

    let mut diag = alpha.to_vec();
    let mut offdiag: Vec<f64> = beta.iter().copied().take(m - 1).collect();
    while offdiag.len() < m - 1 {
        offdiag.push(0.0);
    }

    // Eigenvector accumulator Q (m×m identity).
    let mut q_data = vec![0.0; m * m];
    for i in 0..m {
        q_data[i * m + i] = 1.0;
    }

    let mut p = m - 1;

    for _iter in 0..max_iter {
        // Deflation: find active sub-problem.
        while p > 0
            && offdiag[p - 1].abs()
                <= tol * (diag[p - 1].abs() + diag[p].abs()).max(tol)
        {
            offdiag[p - 1] = 0.0;
            p -= 1;
        }
        if p == 0 {
            break;
        }

        // Find the start of the active block.
        let mut q = p;
        while q > 0
            && offdiag[q - 1].abs()
                > tol * (diag[q - 1].abs() + diag[q].abs()).max(tol)
        {
            q -= 1;
        }

        // Wilkinson shift from the bottom 2×2 block.
        let d = (diag[p - 1] - diag[p]) / 2.0;
        let mu = if d.abs() < 1e-30 {
            diag[p] - offdiag[p - 1].abs()
        } else {
            let b2 = offdiag[p - 1] * offdiag[p - 1];
            diag[p] - b2 / (d + d.signum() * (d * d + b2).sqrt())
        };

        // Implicit QR step with Givens rotations on [q..=p].
        let mut x = diag[q] - mu;
        let mut z = offdiag[q];

        for i in q..p {
            let (c, s) = givens(x, z);

            // Apply rotation to tridiagonal entries (rows/cols i and i+1).
            if i > q {
                offdiag[i - 1] = (x * x + z * z).sqrt();
            }

            let d_i = diag[i];
            let d_i1 = diag[i + 1];
            let e_i = offdiag[i];

            diag[i] = c * c * d_i + 2.0 * c * s * e_i + s * s * d_i1;
            diag[i + 1] = s * s * d_i - 2.0 * c * s * e_i + c * c * d_i1;
            offdiag[i] = c * s * (d_i1 - d_i) + (c * c - s * s) * e_i;

            if i + 1 < p {
                z = -s * offdiag[i + 1];
                offdiag[i + 1] *= c;
                x = offdiag[i];
            }

            // Accumulate eigenvectors: Q = Q * G_i.
            for row in 0..m {
                let qi = q_data[row * m + i];
                let qi1 = q_data[row * m + i + 1];
                q_data[row * m + i] = c * qi - s * qi1;
                q_data[row * m + i + 1] = s * qi + c * qi1;
            }
        }
    }

    let q_matrix = DenseMatrix::from_vec(m, m, q_data)
        .map_err(|e| SpectralCoreError::SpectralTypes(e.into()))?;

    Ok((diag, q_matrix))
}

/// Compute eigenvalues only (no eigenvectors) — lighter weight.
pub fn qr_tridiagonal(
    alpha: &mut Vec<f64>,
    beta: &mut Vec<f64>,
    max_iter: usize,
    tol: f64,
) -> Vec<f64> {
    let m = alpha.len();
    if m == 0 {
        return Vec::new();
    }
    if m == 1 {
        return vec![alpha[0]];
    }

    while beta.len() < m - 1 {
        beta.push(0.0);
    }

    let mut p = m - 1;

    for _iter in 0..max_iter {
        while p > 0
            && beta[p - 1].abs()
                <= tol * (alpha[p - 1].abs() + alpha[p].abs()).max(tol)
        {
            beta[p - 1] = 0.0;
            p -= 1;
        }
        if p == 0 {
            break;
        }

        let mut q = p;
        while q > 0
            && beta[q - 1].abs() > tol * (alpha[q - 1].abs() + alpha[q].abs()).max(tol)
        {
            q -= 1;
        }

        let d = (alpha[p - 1] - alpha[p]) / 2.0;
        let mu = if d.abs() < 1e-30 {
            alpha[p] - beta[p - 1].abs()
        } else {
            let b2 = beta[p - 1] * beta[p - 1];
            alpha[p] - b2 / (d + d.signum() * (d * d + b2).sqrt())
        };

        let mut x = alpha[q] - mu;
        let mut z = beta[q];

        for i in q..p {
            let (c, s) = givens(x, z);

            if i > q {
                beta[i - 1] = (x * x + z * z).sqrt();
            }

            let d_i = alpha[i];
            let d_i1 = alpha[i + 1];
            let e_i = beta[i];

            alpha[i] = c * c * d_i + 2.0 * c * s * e_i + s * s * d_i1;
            alpha[i + 1] = s * s * d_i - 2.0 * c * s * e_i + c * c * d_i1;
            beta[i] = c * s * (d_i1 - d_i) + (c * c - s * s) * e_i;

            if i + 1 < p {
                z = -s * beta[i + 1];
                beta[i + 1] *= c;
                x = beta[i];
            }
        }
    }

    alpha.clone()
}

// ---------------------------------------------------------------------------
// Givens rotation
// ---------------------------------------------------------------------------

/// Compute Givens rotation coefficients (c, s) such that
/// `[c -s; s c]^T [a; b] = [r; 0]`.
fn givens(a: f64, b: f64) -> (f64, f64) {
    if b.abs() < 1e-30 {
        (1.0, 0.0)
    } else if b.abs() > a.abs() {
        let tau = -a / b;
        let s = 1.0 / (1.0 + tau * tau).sqrt();
        (s * tau, s)
    } else {
        let tau = -b / a;
        let c = 1.0 / (1.0 + tau * tau).sqrt();
        (c, c * tau)
    }
}

// ---------------------------------------------------------------------------
// Tiny vector helpers
// ---------------------------------------------------------------------------

fn vec_dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn vec_norm(a: &[f64]) -> f64 {
    vec_dot(a, a).sqrt()
}

fn vec_scale(a: &mut [f64], s: f64) {
    for v in a.iter_mut() {
        *v *= s;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a symmetric tridiagonal CSR matrix (path-graph Laplacian).
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
            let deg = if i == 0 || i == n - 1 { 1.0 } else { 2.0 };
            values.push(deg);
            if i + 1 < n {
                col_ind.push(i + 1);
                values.push(-1.0);
            }
            row_ptr.push(col_ind.len());
        }
        CsrMatrix::new(n, n, row_ptr, col_ind, values).unwrap()
    }

    fn diagonal_csr(diag: &[f64]) -> CsrMatrix<f64> {
        let n = diag.len();
        let mut rp = Vec::with_capacity(n + 1);
        let mut ci = Vec::with_capacity(n);
        let mut vals = Vec::with_capacity(n);
        rp.push(0);
        for (i, &d) in diag.iter().enumerate() {
            ci.push(i);
            vals.push(d);
            rp.push(i + 1);
        }
        CsrMatrix::new(n, n, rp, ci, vals).unwrap()
    }

    #[test]
    fn test_lanczos_iteration_identity() {
        let mat = CsrMatrix::identity(5);
        let cfg = EigenConfig::with_k(2);
        let solver = LanczosSolver::new(&cfg);
        let v0 = vec![1.0, 0.0, 0.0, 0.0, 0.0];
        let (alphas, _betas, v) = solver.lanczos_iteration(&mat, &v0, 5).unwrap();
        for &a in &alphas {
            assert!((a - 1.0).abs() < 1e-10, "alpha = {a}");
        }
        assert!(v.rows > 0);
    }

    #[test]
    fn test_lanczos_iteration_diagonal() {
        let mat = diagonal_csr(&[1.0, 3.0, 5.0, 7.0]);
        let cfg = EigenConfig::with_k(2);
        let solver = LanczosSolver::new(&cfg);
        let v0 = vec![0.5, 0.5, 0.5, 0.5];
        let (alphas, _betas, _v) = solver.lanczos_iteration(&mat, &v0, 4).unwrap();
        assert!(!alphas.is_empty());
    }

    #[test]
    fn test_qr_tridiagonal_2x2() {
        let alpha = vec![2.0, 2.0];
        let beta = vec![-1.0];
        let (evals, _) = qr_tridiagonal_full(&alpha, &beta, 100, 1e-14).unwrap();
        let mut sorted = evals.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((sorted[0] - 1.0).abs() < 1e-10, "got {}", sorted[0]);
        assert!((sorted[1] - 3.0).abs() < 1e-10, "got {}", sorted[1]);
    }

    #[test]
    fn test_qr_tridiagonal_3x3() {
        let alpha = vec![2.0, 2.0, 2.0];
        let beta = vec![-1.0, -1.0];
        let (evals, _) = qr_tridiagonal_full(&alpha, &beta, 200, 1e-14).unwrap();
        let mut sorted = evals.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let expected = [2.0 - 2.0_f64.sqrt(), 2.0, 2.0 + 2.0_f64.sqrt()];
        for (got, want) in sorted.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-8, "got {got}, want {want}");
        }
    }

    #[test]
    fn test_qr_tridiagonal_eigenvalues_only() {
        let mut alpha = vec![2.0, 2.0];
        let mut beta = vec![-1.0];
        let evals = qr_tridiagonal(&mut alpha, &mut beta, 100, 1e-14);
        let mut sorted = evals;
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((sorted[0] - 1.0).abs() < 1e-10);
        assert!((sorted[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_qr_tridiagonal_single() {
        let (evals, evecs) = qr_tridiagonal_full(&[5.0], &[], 10, 1e-14).unwrap();
        assert_eq!(evals.len(), 1);
        assert!((evals[0] - 5.0).abs() < 1e-14);
        assert!((evecs.data[0] - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_qr_tridiagonal_empty() {
        let (evals, evecs) = qr_tridiagonal_full(&[], &[], 10, 1e-14).unwrap();
        assert!(evals.is_empty());
        assert_eq!(evecs.rows, 0);
    }

    #[test]
    fn test_solve_diagonal_matrix() {
        let mat = diagonal_csr(&[1.0, 3.0, 5.0, 7.0, 9.0]);
        let cfg = EigenConfig::with_k(2).max_iter(100).tolerance(1e-6);
        let solver = LanczosSolver::new(&cfg);
        let result = solver.solve(&mat).unwrap();
        assert_eq!(result.eigenvalues.len(), 2);
        assert!(
            (result.eigenvalues[0] - 1.0).abs() < 0.5,
            "got {}",
            result.eigenvalues[0]
        );
    }

    #[test]
    fn test_solve_path_laplacian() {
        let mat = path_laplacian(8);
        let cfg = EigenConfig::with_k(2).max_iter(50).tolerance(1e-4);
        let solver = LanczosSolver::new(&cfg);
        let result = solver.solve(&mat).unwrap();
        assert_eq!(result.eigenvalues.len(), 2);
        // Smallest eigenvalue of path Laplacian is ≈ 2 - 2cos(π/n).
        assert!(result.eigenvalues[0] < 1.0);
    }

    #[test]
    fn test_solve_invalid_k_zero() {
        let mat = CsrMatrix::identity(5);
        let cfg = EigenConfig::with_k(0);
        let solver = LanczosSolver::new(&cfg);
        assert!(solver.solve(&mat).is_err());
    }

    #[test]
    fn test_solve_invalid_k_too_large() {
        let mat = CsrMatrix::identity(3);
        let cfg = EigenConfig::with_k(3);
        let solver = LanczosSolver::new(&cfg);
        assert!(solver.solve(&mat).is_err());
    }

    #[test]
    fn test_givens_basic() {
        let (c, s) = givens(3.0, 4.0);
        // c^2 + s^2 should equal 1
        assert!((c * c + s * s - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_givens_zero_b() {
        let (c, s) = givens(5.0, 0.0);
        assert!((c - 1.0).abs() < 1e-14);
        assert!(s.abs() < 1e-14);
    }

    #[test]
    fn test_vec_helpers() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert!((vec_dot(&a, &b) - 32.0).abs() < 1e-14);
        assert!((vec_norm(&a) - 14.0_f64.sqrt()).abs() < 1e-14);
    }

    #[test]
    fn test_eigenvectors_orthogonal_for_tridiag() {
        let alpha = vec![2.0, 2.0, 2.0];
        let beta = vec![-1.0, -1.0];
        let (_, evecs) = qr_tridiagonal_full(&alpha, &beta, 200, 1e-14).unwrap();
        let m = evecs.rows;
        for i in 0..m {
            for j in 0..m {
                let dot: f64 = (0..m)
                    .map(|r| evecs.data[r * m + i] * evecs.data[r * m + j])
                    .sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-6,
                    "Q^T Q [{i},{j}] = {dot}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn test_lanczos_iteration_produces_orthonormal_basis() {
        let mat = path_laplacian(6);
        let cfg = EigenConfig::with_k(2);
        let solver = LanczosSolver::new(&cfg);
        let v0 = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let (_a, _b, v) = solver.lanczos_iteration(&mat, &v0, 4).unwrap();
        let m = v.cols;
        let n = v.rows;
        for i in 0..m {
            for j in 0..m {
                let dot: f64 = (0..n)
                    .map(|r| v.data[r * m + i] * v.data[r * m + j])
                    .sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-10,
                    "V^T V [{i},{j}] = {dot}, expected {expected}"
                );
            }
        }
    }
}
