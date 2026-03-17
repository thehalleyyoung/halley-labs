//! Locally Optimal Block Preconditioned Conjugate Gradient (LOBPCG) eigensolver.
//!
//! Computes the smallest eigenvalues and corresponding eigenvectors of a
//! symmetric (or Hermitian) matrix using the LOBPCG algorithm.  Supports
//! both dense and sparse (CSR) matrices as well as user-supplied matrix-vector
//! products.
//!
//! # References
//! - A. V. Knyazev, "Toward the Optimal Preconditioned Eigensolver: Locally
//!   Optimal Block Preconditioned Conjugate Gradient Method", SIAM J. Sci.
//!   Comput. 23(2), 2001.

use crate::{DenseMatrix, CsrMatrix, DecompError, DecompResult, dot, norm2, normalize};
use crate::preconditioner::Preconditioner;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand::Rng;

// ═══════════════════════════════════════════════════════════════════════════
// Configuration
// ═══════════════════════════════════════════════════════════════════════════

/// Configuration for the LOBPCG solver.
#[derive(Debug, Clone)]
pub struct LobpcgConfig {
    /// Number of eigenvalues to compute (block size).
    pub block_size: usize,
    /// Maximum number of outer iterations.
    pub max_iter: usize,
    /// Convergence tolerance on residual norms.
    pub tol: f64,
    /// Verbosity level (0 = silent, 1 = summary, 2 = per-iteration).
    pub verbosity: u8,
}

impl Default for LobpcgConfig {
    fn default() -> Self {
        Self {
            block_size: 1,
            max_iter: 500,
            tol: 1e-8,
            verbosity: 0,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Block vectors  (column-major n × k storage)
// ═══════════════════════════════════════════════════════════════════════════

/// Column-major block of vectors of size `n × k`.
#[derive(Debug, Clone)]
pub struct BlockVectors {
    /// Column-major data: element (i, j) lives at `data[j * n + i]`.
    pub data: Vec<f64>,
    /// Number of rows (vector length).
    pub n: usize,
    /// Number of columns (block width).
    pub k: usize,
}

impl BlockVectors {
    /// Create a zero block of `n × k` vectors.
    pub fn new(n: usize, k: usize) -> Self {
        Self {
            data: vec![0.0; n * k],
            n,
            k,
        }
    }

    /// View column `j` as a slice.
    #[inline]
    pub fn col(&self, j: usize) -> &[f64] {
        let start = j * self.n;
        &self.data[start..start + self.n]
    }

    /// Mutable view of column `j`.
    #[inline]
    pub fn col_mut(&mut self, j: usize) -> &mut [f64] {
        let start = j * self.n;
        &mut self.data[start..start + self.n]
    }

    /// Copy `v` into column `j`.
    pub fn set_col(&mut self, j: usize, v: &[f64]) {
        assert_eq!(v.len(), self.n);
        self.col_mut(j).copy_from_slice(v);
    }

    /// In-place modified Gram-Schmidt orthogonalisation.
    /// Returns the number of linearly independent columns retained.
    pub fn orthogonalize(&mut self) -> usize {
        let mut good = 0;
        for j in 0..self.k {
            // Subtract projections of earlier columns
            for p in 0..good {
                let d = dot(self.col(p), self.col(j));
                // self.col_mut(j) -= d * self.col(p)
                // We need to avoid aliasing, so copy p-col first.
                let pcol: Vec<f64> = self.col(p).to_vec();
                let n = self.n;
                let cj = self.col_mut(j);
                for i in 0..n {
                    cj[i] -= d * pcol[i];
                }
            }
            let nrm = normalize(self.col_mut(j));
            if nrm < 1e-14 {
                // Linearly dependent — zero out
                for v in self.col_mut(j).iter_mut() {
                    *v = 0.0;
                }
            } else {
                // If j != good, swap columns
                if j != good {
                    // Copy col j into position good
                    let tmp: Vec<f64> = self.col(j).to_vec();
                    self.set_col(good, &tmp);
                    // Zero out j
                    for v in self.col_mut(j).iter_mut() {
                        *v = 0.0;
                    }
                }
                good += 1;
            }
        }
        good
    }

    /// Convert to a `DenseMatrix` (row-major).
    pub fn to_dense(&self) -> DenseMatrix {
        let mut m = DenseMatrix::zeros(self.n, self.k);
        for i in 0..self.n {
            for j in 0..self.k {
                m.set(i, j, self.data[j * self.n + i]);
            }
        }
        m
    }

    /// Create from a row-major `DenseMatrix`.
    pub fn from_dense(m: &DenseMatrix) -> Self {
        let n = m.rows;
        let k = m.cols;
        let mut bv = Self::new(n, k);
        for i in 0..n {
            for j in 0..k {
                bv.data[j * n + i] = m.get(i, j);
            }
        }
        bv
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Result type
// ═══════════════════════════════════════════════════════════════════════════

/// Result of the LOBPCG solver.
#[derive(Debug, Clone)]
pub struct LobpcgResult {
    /// Computed eigenvalues (smallest `k`).
    pub eigenvalues: Vec<f64>,
    /// Eigenvectors as columns of a `DenseMatrix` (n × k).
    pub eigenvectors: DenseMatrix,
    /// Final residual norms for each eigenpair.
    pub residual_norms: Vec<f64>,
    /// Number of outer iterations performed.
    pub iterations: usize,
    /// Per-eigenpair convergence flag.
    pub converged: Vec<bool>,
}

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Orthogonalise a single vector `w` against an ordered set of orthonormal
/// basis vectors using modified Gram-Schmidt (two passes for stability).
pub fn orthogonalize_against(w: &mut [f64], basis: &[&[f64]]) {
    for _pass in 0..2 {
        for b in basis {
            let d = dot(w, b);
            for (wi, &bi) in w.iter_mut().zip(b.iter()) {
                *wi -= d * bi;
            }
        }
    }
}

/// Jacobi eigenvalue algorithm for a small real symmetric matrix.
///
/// Returns `(eigenvalues, eigenvectors)` where eigenvalues are sorted in
/// ascending order and eigenvectors are the columns of the returned matrix.
pub fn small_symmetric_eigen(a: &DenseMatrix) -> DecompResult<(Vec<f64>, DenseMatrix)> {
    let n = a.rows;
    if n == 0 {
        return Err(DecompError::empty("small_symmetric_eigen: empty matrix"));
    }
    if n != a.cols {
        return Err(DecompError::require_square(a.rows, a.cols));
    }

    // Work on a mutable copy
    let mut s = a.clone();
    let mut v = DenseMatrix::eye(n);

    let max_sweeps = 100;
    let eps = 1e-15;

    for _sweep in 0..max_sweeps {
        // Compute off-diagonal Frobenius norm
        let mut off = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                off += 2.0 * s.get(i, j) * s.get(i, j);
            }
        }
        off = off.sqrt();
        if off < eps * s.frobenius_norm().max(eps) {
            break;
        }

        for p in 0..n {
            for q in (p + 1)..n {
                let apq = s.get(p, q);
                if apq.abs() < eps {
                    continue;
                }
                let tau = (s.get(q, q) - s.get(p, p)) / (2.0 * apq);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    -1.0 / (-tau + (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let ss = t * c;

                // Update S ← J^T S J  (Jacobi rotation on rows/cols p,q)
                // Update columns p, q of S
                for i in 0..n {
                    let sip = s.get(i, p);
                    let siq = s.get(i, q);
                    s.set(i, p, c * sip - ss * siq);
                    s.set(i, q, ss * sip + c * siq);
                }
                // Update rows p, q of S
                for j in 0..n {
                    let spj = s.get(p, j);
                    let sqj = s.get(q, j);
                    s.set(p, j, c * spj - ss * sqj);
                    s.set(q, j, ss * spj + c * sqj);
                }
                // Fix diagonal — rotation makes S(p,q) and S(q,p) zero
                // (they might have small residual from the row/col updates)
                s.set(p, q, 0.0);
                s.set(q, p, 0.0);

                // Accumulate eigenvectors: V ← V * J
                for i in 0..n {
                    let vip = v.get(i, p);
                    let viq = v.get(i, q);
                    v.set(i, p, c * vip - ss * viq);
                    v.set(i, q, ss * vip + c * viq);
                }
            }
        }
    }

    // Collect eigenvalues = diagonal of S
    let mut eigs: Vec<(f64, usize)> = (0..n).map(|i| (s.get(i, i), i)).collect();
    eigs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let eigenvalues: Vec<f64> = eigs.iter().map(|&(v, _)| v).collect();
    let mut eigvecs = DenseMatrix::zeros(n, n);
    for (new_j, &(_, old_j)) in eigs.iter().enumerate() {
        for i in 0..n {
            eigvecs.set(i, new_j, v.get(i, old_j));
        }
    }

    Ok((eigenvalues, eigvecs))
}

/// Solve the generalized eigenvalue problem `G z = λ B z` for a symmetric
/// positive-definite `B` and symmetric `G`.
///
/// Returns the `k` smallest eigenvalues and corresponding eigenvectors.
/// Uses Cholesky factorisation of B to reduce to a standard problem.
pub fn solve_generalized_eigen(
    g: &DenseMatrix,
    b: &DenseMatrix,
    k: usize,
) -> DecompResult<(Vec<f64>, DenseMatrix)> {
    let n = g.rows;
    if n == 0 {
        return Err(DecompError::empty("solve_generalized_eigen: empty"));
    }

    // Cholesky factorisation B = L L^T
    let l = cholesky_lower(b)?;
    let l_inv = lower_triangular_inverse(&l)?;
    let l_inv_t = l_inv.transpose();

    // Transform: C = L^{-1} G L^{-T}
    let tmp = l_inv.mul_mat(g)?;
    let c = tmp.mul_mat(&l_inv_t)?;

    // Symmetrise C to remove rounding asymmetry
    let mut c_sym = DenseMatrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let v = 0.5 * (c.get(i, j) + c.get(j, i));
            c_sym.set(i, j, v);
            c_sym.set(j, i, v);
        }
    }

    // Standard eigenproblem on C
    let (evals, evecs) = small_symmetric_eigen(&c_sym)?;

    // Back-transform: x = L^{-T} z
    let k_actual = k.min(n);
    let z_sub = evecs.submatrix(0, n, 0, k_actual);
    let x = l_inv_t.mul_mat(&z_sub)?;

    Ok((evals[..k_actual].to_vec(), x))
}

/// Cholesky factorisation: returns lower-triangular `L` such that `A = L L^T`.
fn cholesky_lower(a: &DenseMatrix) -> DecompResult<DenseMatrix> {
    let n = a.rows;
    let mut l = DenseMatrix::zeros(n, n);

    for j in 0..n {
        let mut sum = 0.0;
        for k in 0..j {
            sum += l.get(j, k) * l.get(j, k);
        }
        let diag = a.get(j, j) - sum;
        if diag <= 0.0 {
            // Try a small regularisation
            let reg = 1e-12 * a.frobenius_norm().max(1.0);
            let diag_reg = diag + reg;
            if diag_reg <= 0.0 {
                return Err(DecompError::NotPositiveDefinite {
                    context: format!("Cholesky failed at column {j}: diag = {diag:.2e}"),
                });
            }
            l.set(j, j, diag_reg.sqrt());
        } else {
            l.set(j, j, diag.sqrt());
        }

        let ljj = l.get(j, j);
        for i in (j + 1)..n {
            let mut s = 0.0;
            for k in 0..j {
                s += l.get(i, k) * l.get(j, k);
            }
            l.set(i, j, (a.get(i, j) - s) / ljj);
        }
    }

    Ok(l)
}

/// Invert a lower-triangular matrix.
fn lower_triangular_inverse(l: &DenseMatrix) -> DecompResult<DenseMatrix> {
    let n = l.rows;
    let mut inv = DenseMatrix::zeros(n, n);

    for j in 0..n {
        let ljj = l.get(j, j);
        if ljj.abs() < 1e-300 {
            return Err(DecompError::singular("lower_triangular_inverse: zero diagonal"));
        }
        inv.set(j, j, 1.0 / ljj);
        for i in (j + 1)..n {
            let mut s = 0.0;
            for k in j..i {
                s += l.get(i, k) * inv.get(k, j);
            }
            inv.set(i, j, -s / l.get(i, i));
        }
    }

    Ok(inv)
}

// ═══════════════════════════════════════════════════════════════════════════
// Rayleigh–Ritz
// ═══════════════════════════════════════════════════════════════════════════

/// Perform a Rayleigh–Ritz projection.
///
/// Given a basis `S` (n × m) and `AS = A * S` (n × m), extract the best
/// `k` approximate eigenpairs by solving the projected eigenproblem
/// `(S^T A S) z = λ (S^T S) z`.
///
/// Returns `(eigenvalues, eigenvectors_in_S_coords, rotated_S, rotated_AS)`.
pub fn rayleigh_ritz(
    s: &BlockVectors,
    as_mat: &BlockVectors,
    k: usize,
) -> DecompResult<(Vec<f64>, DenseMatrix, BlockVectors, BlockVectors)> {
    let n = s.n;
    let m = s.k;
    let k_actual = k.min(m);

    // Build G = S^T A S  and  B = S^T S
    let mut g = DenseMatrix::zeros(m, m);
    let mut b = DenseMatrix::zeros(m, m);
    for i in 0..m {
        for j in 0..m {
            g.set(i, j, dot(s.col(i), as_mat.col(j)));
            b.set(i, j, dot(s.col(i), s.col(j)));
        }
    }

    // Symmetrise
    for i in 0..m {
        for j in (i + 1)..m {
            let gv = 0.5 * (g.get(i, j) + g.get(j, i));
            g.set(i, j, gv);
            g.set(j, i, gv);
            let bv = 0.5 * (b.get(i, j) + b.get(j, i));
            b.set(i, j, bv);
            b.set(j, i, bv);
        }
    }

    let (evals, evecs) = solve_generalized_eigen(&g, &b, k_actual)?;

    // Rotate: X_new = S * Z,  AX_new = AS * Z
    let mut x_new = BlockVectors::new(n, k_actual);
    let mut ax_new = BlockVectors::new(n, k_actual);
    for j in 0..k_actual {
        for p in 0..m {
            let z_pj = evecs.get(p, j);
            let xn = x_new.col_mut(j);
            let sc = s.col(p);
            for i in 0..n {
                xn[i] += z_pj * sc[i];
            }
        }
        for p in 0..m {
            let z_pj = evecs.get(p, j);
            let axn = ax_new.col_mut(j);
            let asc = as_mat.col(p);
            for i in 0..n {
                axn[i] += z_pj * asc[i];
            }
        }
    }

    Ok((evals, evecs, x_new, ax_new))
}

// ═══════════════════════════════════════════════════════════════════════════
// Main LOBPCG
// ═══════════════════════════════════════════════════════════════════════════

/// Core LOBPCG algorithm.
///
/// # Arguments
/// * `matvec` – closure computing `y = A * x` for a single vector.
/// * `n` – dimension of the matrix.
/// * `config` – solver parameters.
/// * `preconditioner` – optional preconditioner implementing [`Preconditioner`].
/// * `constraints` – optional set of vectors that the solution must be
///   orthogonal to (columns of a `DenseMatrix`).
pub fn lobpcg<F>(
    matvec: F,
    n: usize,
    config: &LobpcgConfig,
    preconditioner: Option<&dyn Preconditioner>,
    constraints: Option<&DenseMatrix>,
) -> DecompResult<LobpcgResult>
where
    F: Fn(&[f64]) -> DecompResult<Vec<f64>>,
{
    let k = config.block_size;
    if k == 0 {
        return Err(DecompError::InvalidParameter {
            name: "block_size".into(),
            value: "0".into(),
            reason: "block_size must be >= 1".into(),
        });
    }
    if k > n {
        return Err(DecompError::BlockSizeTooLarge {
            block_size: k,
            dim: n,
        });
    }

    // Collect constraint vectors (already assumed orthonormal)
    let constraint_vecs: Vec<Vec<f64>> = if let Some(c) = constraints {
        (0..c.cols).map(|j| c.col(j)).collect()
    } else {
        Vec::new()
    };

    // ── Initialise X with random vectors ──────────────────────────────
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut x = BlockVectors::new(n, k);
    for j in 0..k {
        let col = x.col_mut(j);
        for i in 0..n {
            col[i] = rng.gen::<f64>() - 0.5;
        }
    }

    // Orthogonalise X against constraints
    orthogonalize_block_against_constraints(&mut x, &constraint_vecs);
    x.orthogonalize();

    // ── AX = A * X ────────────────────────────────────────────────────
    let mut ax = BlockVectors::new(n, k);
    for j in 0..k {
        let aj = matvec(x.col(j))?;
        ax.set_col(j, &aj);
    }

    // ── Initial Rayleigh quotient ─────────────────────────────────────
    let mut lambda = vec![0.0; k];
    {
        let mut m_mat = DenseMatrix::zeros(k, k);
        for i in 0..k {
            for j in 0..k {
                m_mat.set(i, j, dot(x.col(i), ax.col(j)));
            }
        }
        // Symmetrise
        for i in 0..k {
            for j in (i + 1)..k {
                let v = 0.5 * (m_mat.get(i, j) + m_mat.get(j, i));
                m_mat.set(i, j, v);
                m_mat.set(j, i, v);
            }
        }
        let (evals, evecs) = small_symmetric_eigen(&m_mat)?;
        lambda.copy_from_slice(&evals[..k]);

        // Rotate X and AX by eigenvectors
        let x_old = x.clone();
        let ax_old = ax.clone();
        for j in 0..k {
            let xj = x.col_mut(j);
            for i in 0..n {
                xj[i] = 0.0;
            }
            for p in 0..k {
                let z = evecs.get(p, j);
                let xj = x.col_mut(j);
                let old = x_old.col(p);
                for i in 0..n {
                    xj[i] += z * old[i];
                }
            }
        }
        for j in 0..k {
            let axj = ax.col_mut(j);
            for i in 0..n {
                axj[i] = 0.0;
            }
            for p in 0..k {
                let z = evecs.get(p, j);
                let axj = ax.col_mut(j);
                let old = ax_old.col(p);
                for i in 0..n {
                    axj[i] += z * old[i];
                }
            }
        }
    }

    // ── P, AP (search directions — empty at iter 0) ──────────────────
    let mut p = BlockVectors::new(n, k);
    #[allow(unused_assignments)]
    let mut ap = BlockVectors::new(n, k);
    let mut has_p = false;

    let mut residual_norms = vec![0.0; k];
    let mut converged_flags = vec![false; k];
    let mut final_iter = 0;

    // ── Main loop ─────────────────────────────────────────────────────
    for iter in 0..config.max_iter {
        final_iter = iter + 1;

        // Residual R = AX - X * diag(Lambda)
        let mut r = BlockVectors::new(n, k);
        for j in 0..k {
            let rj = r.col_mut(j);
            let axj = ax.col(j);
            let xj = x.col(j);
            let lj = lambda[j];
            for i in 0..n {
                rj[i] = axj[i] - lj * xj[i];
            }
            residual_norms[j] = norm2(rj);
        }

        // Check convergence
        let mut all_converged = true;
        for j in 0..k {
            converged_flags[j] = residual_norms[j] < config.tol;
            if !converged_flags[j] {
                all_converged = false;
            }
        }

        if config.verbosity >= 2 {
            let max_res = residual_norms.iter().cloned().fold(0.0_f64, f64::max);
            eprintln!("LOBPCG iter {iter}: max_residual = {max_res:.2e}");
        }

        if all_converged {
            if config.verbosity >= 1 {
                eprintln!("LOBPCG converged in {final_iter} iterations");
            }
            return Ok(LobpcgResult {
                eigenvalues: lambda,
                eigenvectors: x.to_dense(),
                residual_norms,
                iterations: final_iter,
                converged: converged_flags,
            });
        }

        // ── Precondition: W = M^{-1} R  (or W = R) ──────────────────
        let mut w = BlockVectors::new(n, k);
        match preconditioner {
            Some(pc) => {
                for j in 0..k {
                    if converged_flags[j] {
                        // Already converged – skip
                        w.set_col(j, r.col(j));
                    } else {
                        pc.apply(r.col(j), w.col_mut(j))?;
                    }
                }
            }
            None => {
                w.data.copy_from_slice(&r.data);
            }
        }

        // Orthogonalise W against constraints and X
        orthogonalize_block_against_constraints(&mut w, &constraint_vecs);
        orthogonalize_block_against_block(&mut w, &x);
        // Internal orthogonalisation of W
        let w_rank = w.orthogonalize();
        if w_rank == 0 {
            // W is entirely in span(X) — we're done or stuck
            break;
        }
        // Trim W to w_rank columns
        let w = trim_block(&w, w_rank);

        // ── Build trial basis S = [X, W, P]  (or [X, W] at iter 0) ──
        let (s, a_s) = if has_p {
            // Also orthogonalise P against X and W
            let mut p2 = p.clone();
            orthogonalize_block_against_block(&mut p2, &x);
            orthogonalize_block_against_block(&mut p2, &w);
            let p_rank = p2.orthogonalize();
            let p2 = trim_block(&p2, p_rank.max(1).min(p2.k));

            let mut ap2 = BlockVectors::new(n, p2.k);
            for j in 0..p2.k {
                let av = matvec(p2.col(j))?;
                ap2.set_col(j, &av);
            }

            let m = k + w.k + p2.k;
            let mut s = BlockVectors::new(n, m);
            let mut a_s = BlockVectors::new(n, m);
            let mut col = 0;
            for j in 0..k {
                s.set_col(col, x.col(j));
                a_s.set_col(col, ax.col(j));
                col += 1;
            }
            for j in 0..w.k {
                s.set_col(col, w.col(j));
                // Compute AW on the fly
                let aw_j = matvec(w.col(j))?;
                a_s.set_col(col, &aw_j);
                col += 1;
            }
            for j in 0..p2.k {
                s.set_col(col, p2.col(j));
                a_s.set_col(col, ap2.col(j));
                col += 1;
            }
            (s, a_s)
        } else {
            let m = k + w.k;
            let mut s = BlockVectors::new(n, m);
            let mut a_s = BlockVectors::new(n, m);
            let mut col = 0;
            for j in 0..k {
                s.set_col(col, x.col(j));
                a_s.set_col(col, ax.col(j));
                col += 1;
            }
            for j in 0..w.k {
                s.set_col(col, w.col(j));
                let aw_j = matvec(w.col(j))?;
                a_s.set_col(col, &aw_j);
                col += 1;
            }
            (s, a_s)
        };

        // ── Rayleigh–Ritz on [X, W, P] ──────────────────────────────
        let rr = rayleigh_ritz(&s, &a_s, k);
        let (new_lambda, _evecs, x_new, ax_new) = match rr {
            Ok(v) => v,
            Err(_) => {
                // Fallback: just do Rayleigh-Ritz on [X, W]
                let m2 = k + w.k;
                let mut s2 = BlockVectors::new(n, m2);
                let mut as2 = BlockVectors::new(n, m2);
                for j in 0..k {
                    s2.set_col(j, x.col(j));
                    as2.set_col(j, ax.col(j));
                }
                for j in 0..w.k {
                    s2.set_col(k + j, w.col(j));
                    let awj = matvec(w.col(j))?;
                    as2.set_col(k + j, &awj);
                }
                rayleigh_ritz(&s2, &as2, k)?
            }
        };

        // ── Update P = X_new - X_old,  AP = AX_new - AX_old ─────────
        p = BlockVectors::new(n, k);
        ap = BlockVectors::new(n, k);
        for j in 0..k {
            let pj = p.col_mut(j);
            let xn = x_new.col(j);
            let xo = x.col(j);
            for i in 0..n {
                pj[i] = xn[i] - xo[i];
            }
            let apj = ap.col_mut(j);
            let axn = ax_new.col(j);
            let axo = ax.col(j);
            for i in 0..n {
                apj[i] = axn[i] - axo[i];
            }
        }
        has_p = true;

        // ── Accept new iterates ──────────────────────────────────────
        x = x_new;
        ax = ax_new;
        lambda = new_lambda;
    }

    // Did not fully converge — report partial results
    let num_converged = converged_flags.iter().filter(|&&c| c).count();
    if num_converged == 0 {
        return Err(DecompError::EigenConvergenceFailure {
            converged: num_converged,
            requested: k,
            iterations: final_iter,
        });
    }

    Ok(LobpcgResult {
        eigenvalues: lambda,
        eigenvectors: x.to_dense(),
        residual_norms,
        iterations: final_iter,
        converged: converged_flags,
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// Convenience wrappers
// ═══════════════════════════════════════════════════════════════════════════

/// LOBPCG for a sparse CSR matrix.
///
/// Finds the `k` smallest eigenvalues of a symmetric positive-definite
/// sparse matrix.
pub fn lobpcg_sparse(
    a: &CsrMatrix,
    k: usize,
    tol: f64,
    max_iter: usize,
    preconditioner: Option<&dyn Preconditioner>,
) -> DecompResult<LobpcgResult> {
    if a.rows != a.cols {
        return Err(DecompError::require_square(a.rows, a.cols));
    }
    let n = a.rows;
    if n == 0 {
        return Err(DecompError::empty("lobpcg_sparse: empty matrix"));
    }
    if k > n {
        return Err(DecompError::TooManyEigenvalues {
            requested: k,
            size: n,
        });
    }

    let config = LobpcgConfig {
        block_size: k,
        max_iter,
        tol,
        verbosity: 0,
    };

    lobpcg(
        |x| a.mul_vec(x),
        n,
        &config,
        preconditioner,
        None,
    )
}

/// LOBPCG for a dense matrix.
///
/// Finds the `k` smallest eigenvalues of a symmetric dense matrix.
pub fn lobpcg_dense(
    a: &DenseMatrix,
    k: usize,
    tol: f64,
    max_iter: usize,
) -> DecompResult<LobpcgResult> {
    if a.rows != a.cols {
        return Err(DecompError::require_square(a.rows, a.cols));
    }
    let n = a.rows;
    if n == 0 {
        return Err(DecompError::empty("lobpcg_dense: empty matrix"));
    }
    if k > n {
        return Err(DecompError::TooManyEigenvalues {
            requested: k,
            size: n,
        });
    }

    let config = LobpcgConfig {
        block_size: k,
        max_iter,
        tol,
        verbosity: 0,
    };

    lobpcg(
        |x| a.mul_vec(x),
        n,
        &config,
        None,
        None,
    )
}

// ═══════════════════════════════════════════════════════════════════════════
// Internal helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Orthogonalise each column of `bv` against a set of constraint vectors.
fn orthogonalize_block_against_constraints(bv: &mut BlockVectors, constraints: &[Vec<f64>]) {
    if constraints.is_empty() {
        return;
    }
    let refs: Vec<&[f64]> = constraints.iter().map(|v| v.as_slice()).collect();
    for j in 0..bv.k {
        orthogonalize_against(bv.col_mut(j), &refs);
    }
}

/// Orthogonalise each column of `w` against all columns of `basis`.
fn orthogonalize_block_against_block(w: &mut BlockVectors, basis: &BlockVectors) {
    let wn = w.n;
    for j in 0..w.k {
        for _pass in 0..2 {
            for p in 0..basis.k {
                let d = dot(w.col(j), basis.col(p));
                let bp: Vec<f64> = basis.col(p).to_vec();
                let wj = w.col_mut(j);
                for i in 0..wn {
                    wj[i] -= d * bp[i];
                }
            }
        }
    }
}

/// Trim a `BlockVectors` to its first `m` columns.
fn trim_block(bv: &BlockVectors, m: usize) -> BlockVectors {
    if m >= bv.k {
        return bv.clone();
    }
    let mut out = BlockVectors::new(bv.n, m);
    for j in 0..m {
        out.set_col(j, bv.col(j));
    }
    out
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preconditioner::{IdentityPreconditioner, JacobiPreconditioner};

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // ── Test 1: diagonal matrix ──────────────────────────────────────
    #[test]
    fn test_lobpcg_diagonal_matrix() {
        // diag(1, 2, 3, 4, 5) — smallest eigenvalue is 1.0
        let a = DenseMatrix::from_diag(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = lobpcg_dense(&a, 1, 1e-8, 200).unwrap();
        assert!(approx_eq(result.eigenvalues[0], 1.0, 1e-6),
            "expected 1.0, got {}", result.eigenvalues[0]);
        assert!(result.converged[0]);
    }

    // ── Test 2: known SPD matrix ─────────────────────────────────────
    #[test]
    fn test_lobpcg_known_spd() {
        // A = [[2, -1, 0], [-1, 2, -1], [0, -1, 2]]
        // Eigenvalues: 2 - 2cos(k*pi/4) for k=1,2,3
        //   = 2 - sqrt(2), 2, 2 + sqrt(2)  ≈  0.5858, 2.0, 3.4142
        let a = DenseMatrix::from_row_major(3, 3, vec![
            2.0, -1.0,  0.0,
           -1.0,  2.0, -1.0,
            0.0, -1.0,  2.0,
        ]);
        let result = lobpcg_dense(&a, 1, 1e-8, 300).unwrap();
        let expected = 2.0 - (2.0_f64).sqrt();
        assert!(approx_eq(result.eigenvalues[0], expected, 1e-5),
            "expected {expected}, got {}", result.eigenvalues[0]);
    }

    // ── Test 3: block vectors orthogonalisation ──────────────────────
    #[test]
    fn test_block_vectors_orthogonalize() {
        let n = 5;
        let k = 3;
        let mut bv = BlockVectors::new(n, k);
        // Set some non-orthogonal vectors
        bv.col_mut(0).copy_from_slice(&[1.0, 1.0, 0.0, 0.0, 0.0]);
        bv.col_mut(1).copy_from_slice(&[1.0, 0.0, 1.0, 0.0, 0.0]);
        bv.col_mut(2).copy_from_slice(&[0.0, 1.0, 1.0, 1.0, 0.0]);

        let rank = bv.orthogonalize();
        assert_eq!(rank, 3);

        // Check orthonormality
        for i in 0..k {
            let ni = norm2(bv.col(i));
            assert!(approx_eq(ni, 1.0, 1e-12), "norm of col {i} = {ni}");
        }
        for i in 0..k {
            for j in (i + 1)..k {
                let d = dot(bv.col(i), bv.col(j));
                assert!(d.abs() < 1e-12, "dot({i},{j}) = {d}");
            }
        }
    }

    // ── Test 4: BlockVectors to/from DenseMatrix ─────────────────────
    #[test]
    fn test_block_vectors_dense_roundtrip() {
        let m = DenseMatrix::from_row_major(3, 2, vec![
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0,
        ]);
        let bv = BlockVectors::from_dense(&m);
        assert_eq!(bv.n, 3);
        assert_eq!(bv.k, 2);
        assert!(approx_eq(bv.col(0)[0], 1.0, 1e-15));
        assert!(approx_eq(bv.col(0)[1], 3.0, 1e-15));
        assert!(approx_eq(bv.col(1)[2], 6.0, 1e-15));

        let m2 = bv.to_dense();
        for i in 0..3 {
            for j in 0..2 {
                assert!(approx_eq(m.get(i, j), m2.get(i, j), 1e-15));
            }
        }
    }

    // ── Test 5: Rayleigh-Ritz ────────────────────────────────────────
    #[test]
    fn test_rayleigh_ritz_identity() {
        // If S is orthonormal columns of identity and AS = A*S for diagonal A,
        // Rayleigh-Ritz should recover the eigenvalues.
        let n = 4;
        let k = 2;
        let a = DenseMatrix::from_diag(&[1.0, 3.0, 5.0, 7.0]);
        let mut s = BlockVectors::new(n, k);
        s.col_mut(0).copy_from_slice(&[1.0, 0.0, 0.0, 0.0]);
        s.col_mut(1).copy_from_slice(&[0.0, 1.0, 0.0, 0.0]);

        let mut as_mat = BlockVectors::new(n, k);
        for j in 0..k {
            let v = a.mul_vec(s.col(j)).unwrap();
            as_mat.set_col(j, &v);
        }

        let (evals, _evecs, _x_new, _ax_new) = rayleigh_ritz(&s, &as_mat, k).unwrap();
        assert!(approx_eq(evals[0], 1.0, 1e-12));
        assert!(approx_eq(evals[1], 3.0, 1e-12));
    }

    // ── Test 6: generalized eigen solve ──────────────────────────────
    #[test]
    fn test_solve_generalized_eigen_identity_b() {
        // G z = λ I z  ⟹  standard eigenproblem
        let g = DenseMatrix::from_row_major(3, 3, vec![
            2.0, -1.0,  0.0,
           -1.0,  2.0, -1.0,
            0.0, -1.0,  2.0,
        ]);
        let b = DenseMatrix::eye(3);
        let (evals, evecs) = solve_generalized_eigen(&g, &b, 3).unwrap();

        let expected_min = 2.0 - (2.0_f64).sqrt();
        assert!(approx_eq(evals[0], expected_min, 1e-8),
            "expected {expected_min}, got {}", evals[0]);

        // Check Gz ≈ λz for first eigenpair
        let z0 = evecs.col(0);
        let gz0 = g.mul_vec(&z0).unwrap();
        for i in 0..3 {
            assert!(approx_eq(gz0[i], evals[0] * z0[i], 1e-8));
        }
    }

    // ── Test 7: convergence on larger problem ────────────────────────
    #[test]
    fn test_lobpcg_convergence_10x10() {
        let n = 10;
        let a = DenseMatrix::from_diag(
            &(1..=n).map(|i| i as f64).collect::<Vec<_>>()
        );
        let result = lobpcg_dense(&a, 2, 1e-6, 300).unwrap();
        // Smallest two eigenvalues: 1.0, 2.0
        assert!(approx_eq(result.eigenvalues[0], 1.0, 1e-4),
            "got {}", result.eigenvalues[0]);
        assert!(approx_eq(result.eigenvalues[1], 2.0, 1e-4),
            "got {}", result.eigenvalues[1]);
    }

    // ── Test 8: sparse LOBPCG ────────────────────────────────────────
    #[test]
    fn test_lobpcg_sparse_diagonal() {
        let a_dense = DenseMatrix::from_diag(&[1.0, 3.0, 6.0, 10.0]);
        let a = a_dense.to_csr();
        let result = lobpcg_sparse(&a, 1, 1e-8, 200, None).unwrap();
        assert!(approx_eq(result.eigenvalues[0], 1.0, 1e-5),
            "got {}", result.eigenvalues[0]);
    }

    // ── Test 9: small_symmetric_eigen ────────────────────────────────
    #[test]
    fn test_small_symmetric_eigen() {
        let a = DenseMatrix::from_row_major(3, 3, vec![
            4.0, 1.0, 0.0,
            1.0, 3.0, 1.0,
            0.0, 1.0, 2.0,
        ]);
        let (evals, evecs) = small_symmetric_eigen(&a).unwrap();
        // Check A*v ≈ λ*v for each eigenpair
        for j in 0..3 {
            let v = evecs.col(j);
            let av = a.mul_vec(&v).unwrap();
            for i in 0..3 {
                assert!(approx_eq(av[i], evals[j] * v[i], 1e-8),
                    "eigenpair {j}: av[{i}]={} != λ*v[{i}]={}", av[i], evals[j] * v[i]);
            }
        }
        // Eigenvalues should be sorted ascending
        for j in 1..3 {
            assert!(evals[j] >= evals[j - 1] - 1e-12);
        }
    }

    // ── Test 10: constraints ─────────────────────────────────────────
    #[test]
    fn test_lobpcg_with_constraints() {
        // diag(1,2,3,4,5), constrain against eigenvector of λ=1 (e1)
        // So the solver should find λ=2 as the smallest.
        let a = DenseMatrix::from_diag(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let n = 5;
        // Constraint: first standard basis vector
        let constraints = DenseMatrix::from_row_major(n, 1, vec![1.0, 0.0, 0.0, 0.0, 0.0]);

        let config = LobpcgConfig {
            block_size: 1,
            max_iter: 300,
            tol: 1e-6,
            verbosity: 0,
        };

        let result = lobpcg(
            |x| a.mul_vec(x),
            n,
            &config,
            None,
            Some(&constraints),
        ).unwrap();

        assert!(approx_eq(result.eigenvalues[0], 2.0, 1e-4),
            "expected ~2.0, got {}", result.eigenvalues[0]);
    }

    // ── Test 11: dense LOBPCG with preconditioner ────────────────────
    #[test]
    fn test_lobpcg_sparse_with_preconditioner() {
        let a_dense = DenseMatrix::from_row_major(4, 4, vec![
            4.0, -1.0,  0.0,  0.0,
           -1.0,  4.0, -1.0,  0.0,
            0.0, -1.0,  4.0, -1.0,
            0.0,  0.0, -1.0,  4.0,
        ]);
        let a = a_dense.to_csr();
        let pc = JacobiPreconditioner::from_csr(&a);
        let result = lobpcg_sparse(&a, 1, 1e-8, 300, Some(&pc)).unwrap();
        // Smallest eigenvalue of tridiag(4,-1,-1,4) of size 4
        // = 4 - 2*cos(pi/5) ≈ 2.382
        assert!(result.eigenvalues[0] > 2.0 && result.eigenvalues[0] < 4.0,
            "got {}", result.eigenvalues[0]);
        assert!(result.converged[0]);
    }

    // ── Test 12: orthogonalize_against ───────────────────────────────
    #[test]
    fn test_orthogonalize_against_fn() {
        let e1 = vec![1.0, 0.0, 0.0];
        let e2 = vec![0.0, 1.0, 0.0];
        let mut w = vec![1.0, 1.0, 1.0];
        orthogonalize_against(&mut w, &[e1.as_slice(), e2.as_slice()]);
        // w should now be ~[0, 0, 1]
        assert!(w[0].abs() < 1e-12);
        assert!(w[1].abs() < 1e-12);
        assert!(approx_eq(w[2], 1.0, 1e-12));
    }

    // ── Test 13: block size too large ────────────────────────────────
    #[test]
    fn test_lobpcg_block_size_too_large() {
        let a = DenseMatrix::eye(3);
        let result = lobpcg_dense(&a, 5, 1e-8, 100);
        assert!(result.is_err());
    }

    // ── Test 14: small_symmetric_eigen 1×1 ───────────────────────────
    #[test]
    fn test_small_symmetric_eigen_1x1() {
        let a = DenseMatrix::from_row_major(1, 1, vec![42.0]);
        let (evals, evecs) = small_symmetric_eigen(&a).unwrap();
        assert!(approx_eq(evals[0], 42.0, 1e-12));
        assert!(approx_eq(evecs.get(0, 0).abs(), 1.0, 1e-12));
    }

    // ── Test 15: multiple eigenvalues with dense LOBPCG ──────────────
    #[test]
    fn test_lobpcg_dense_multiple() {
        let n = 8;
        let diag: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let a = DenseMatrix::from_diag(&diag);
        let result = lobpcg_dense(&a, 3, 1e-6, 400).unwrap();
        // Should get 1, 2, 3 (approximately)
        for (j, expected) in [1.0, 2.0, 3.0].iter().enumerate() {
            assert!(approx_eq(result.eigenvalues[j], *expected, 0.1),
                "eig[{j}]: expected {expected}, got {}", result.eigenvalues[j]);
        }
    }

    // ── Test 16: Cholesky factorisation ──────────────────────────────
    #[test]
    fn test_cholesky_lower_basic() {
        let a = DenseMatrix::from_row_major(2, 2, vec![
            4.0, 2.0,
            2.0, 5.0,
        ]);
        let l = cholesky_lower(&a).unwrap();
        // L * L^T should equal A
        let lt = l.transpose();
        let prod = l.mul_mat(&lt).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                assert!(approx_eq(prod.get(i, j), a.get(i, j), 1e-12),
                    "({i},{j}): {} vs {}", prod.get(i, j), a.get(i, j));
            }
        }
    }
}
