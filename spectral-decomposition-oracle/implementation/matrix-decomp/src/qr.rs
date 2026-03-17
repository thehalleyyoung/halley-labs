//! QR decomposition via Householder reflections and Givens rotations.
//!
//! This module provides several complementary QR factorization algorithms:
//!
//! # Algorithms
//!
//! - **Householder QR** ([`QrDecomposition`]): The workhorse algorithm for dense
//!   matrices. Stores the factorization in compact (packed) form so that Q is
//!   never explicitly formed unless requested. Supports:
//!   - Extracting Q and R explicitly ([`get_q`](QrDecomposition::get_q),
//!     [`get_r`](QrDecomposition::get_r))
//!   - Thin (economy-size) Q ([`get_q_thin`](QrDecomposition::get_q_thin))
//!   - Implicit Q^T and Q application without forming Q
//!     ([`apply_qt`](QrDecomposition::apply_qt), [`apply_q`](QrDecomposition::apply_q))
//!   - Direct solve for square/overdetermined systems
//!     ([`solve`](QrDecomposition::solve))
//!   - Least-squares solve ([`solve_least_squares`](QrDecomposition::solve_least_squares))
//!   - Numerical rank estimation ([`rank`](QrDecomposition::rank))
//!
//! - **Column-pivoting QR** ([`QrPivoted`]): Rank-revealing factorization where
//!   columns are permuted so that |R[i,i]| ≥ |R[i+1,i+1]|. Essential for
//!   detecting numerical rank deficiency.
//!
//! - **Givens QR** ([`qr_givens`]): Builds Q and R using plane rotations
//!   instead of Householder reflectors. More expensive for dense matrices but
//!   advantageous for banded or sparse structures.
//!
//! - **Sparse QR** ([`factorize_sparse`]): Convenience wrapper that converts a
//!   [`CsrMatrix`] to dense and delegates to Householder QR.
//!
//! # Helpers
//!
//! - [`back_solve_upper_triangular`]: Solve Rx = b for upper-triangular R.
//!
//! # Numerical notes
//!
//! Householder reflections are backward-stable; the computed Q and R satisfy
//! `Q*R = A + E` where `‖E‖ = O(ε‖A‖)` in exact arithmetic. Column pivoting
//! further improves reliability when A is (near-)rank-deficient.

use crate::{DenseMatrix, CsrMatrix, DecompError, DecompResult, norm2};
use crate::givens::GivensRotation;

// ═══════════════════════════════════════════════════════════════════════════
// Inline Householder helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Generate a Householder reflector for vector `x` such that
/// `(I - tau * v * v^T) * x = ±‖x‖ * e_1`.
///
/// The sign is chosen to avoid catastrophic cancellation: we set
/// `v[0] = x[0] - mu` when `x[0] ≤ 0` and `v[0] = -sigma/(x[0]+mu)`
/// otherwise, where `mu = ‖x‖` and `sigma = ‖x[1:]‖²`.
///
/// Returns `(v, tau)` where `v` is normalized so that `v[0] = 1`.
/// When the sub-diagonal part of `x` is already zero, `tau = 0` and
/// no reflection is needed.
fn generate_householder(x: &[f64]) -> (Vec<f64>, f64) {
    let n = x.len();
    if n == 0 {
        return (vec![], 0.0);
    }
    if n == 1 {
        // Scalar case: no reflection needed (already a 1-element vector).
        return (vec![1.0], 0.0);
    }

    let mut v = x.to_vec();

    // Compute norm of x[1..] (the tail).
    let sigma: f64 = v[1..].iter().map(|&xi| xi * xi).sum();
    v[0] = 1.0;

    if sigma < 1e-300 {
        // x is effectively already proportional to e_1.
        return (v, 0.0);
    }

    let x0 = x[0];
    let mu = (x0 * x0 + sigma).sqrt();

    if x0 <= 0.0 {
        v[0] = x0 - mu;
    } else {
        v[0] = -sigma / (x0 + mu);
    }

    let tau = 2.0 * v[0] * v[0] / (sigma + v[0] * v[0]);

    // Normalize v so that v[0] = 1.
    let v0 = v[0];
    for vi in v.iter_mut() {
        *vi /= v0;
    }

    (v, tau)
}

/// Apply Householder reflector (I - tau * v * v^T) to columns col_start..col_start+ncols
/// of mat, for rows row_start..row_start+nrows, from the left.
///
/// `v` has length `nrows`; `v[0] = 1` implicitly.
fn apply_householder_left(
    v: &[f64],
    tau: f64,
    mat: &mut DenseMatrix,
    row_start: usize,
    col_start: usize,
    nrows: usize,
    ncols: usize,
) {
    if tau == 0.0 || nrows == 0 || ncols == 0 {
        return;
    }
    // For each column j, compute w_j = v^T * mat[row_start:row_start+nrows, j]
    // then mat[row_start + i, j] -= tau * v[i] * w_j
    for j in col_start..col_start + ncols {
        let mut w = 0.0;
        for i in 0..nrows {
            w += v[i] * mat.get(row_start + i, j);
        }
        w *= tau;
        for i in 0..nrows {
            let old = mat.get(row_start + i, j);
            mat.set(row_start + i, j, old - v[i] * w);
        }
    }
}

/// Apply Householder reflector (I - tau * v * v^T) from the right:
/// mat[:, col_start:col_start+ncols] = mat[:, col_start:col_start+ncols] * (I - tau * v * v^T)
///
/// Operates on rows row_start..row_start+nrows of the matrix.
#[allow(dead_code)]
fn apply_householder_right(
    v: &[f64],
    tau: f64,
    mat: &mut DenseMatrix,
    row_start: usize,
    col_start: usize,
    nrows: usize,
    ncols: usize,
) {
    if tau == 0.0 || nrows == 0 || ncols == 0 {
        return;
    }
    // For each row i, compute w_i = mat[i, col_start:] . v
    // then mat[i, col_start + j] -= tau * w_i * v[j]
    for i in row_start..row_start + nrows {
        let mut w = 0.0;
        for j in 0..ncols {
            w += mat.get(i, col_start + j) * v[j];
        }
        w *= tau;
        for j in 0..ncols {
            let old = mat.get(i, col_start + j);
            mat.set(i, col_start + j, old - w * v[j]);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// QrDecomposition — Householder QR
// ═══════════════════════════════════════════════════════════════════════════

/// QR decomposition of an m×n matrix via Householder reflections.
///
/// The factorization is stored in compact form: the upper triangle of `qr`
/// holds R, and the entries below the diagonal hold the Householder vectors
/// (with implicit leading 1).
#[derive(Clone, Debug)]
pub struct QrDecomposition {
    /// Combined storage: R on/above diagonal, Householder vectors below.
    pub qr: DenseMatrix,
    /// Householder scalars (one per reflection).
    pub tau: Vec<f64>,
    /// Number of rows of original matrix.
    pub m: usize,
    /// Number of columns of original matrix.
    pub n: usize,
}

impl QrDecomposition {
    /// Compute the QR factorization of `a` (m×n) via Householder reflections.
    ///
    /// Returns a compact representation; use [`get_q`], [`get_r`], etc. to
    /// extract the factors.
    pub fn factorize(a: &DenseMatrix) -> DecompResult<Self> {
        let m = a.rows;
        let n = a.cols;
        DecompError::check_non_empty(m, n, "QR factorization")?;

        let mut work = a.clone();
        let k = m.min(n);
        let mut tau_vec = vec![0.0; k];

        for step in 0..k {
            // Extract column vector a[step:m, step].
            let col_len = m - step;
            let mut col = vec![0.0; col_len];
            for i in 0..col_len {
                col[i] = work.get(step + i, step);
            }

            // Generate Householder reflector.
            let (v, tau) = generate_householder(&col);
            tau_vec[step] = tau;

            // Store the Householder vector below the diagonal.
            // v[0] = 1 (implicit), so we store v[1..] in work[step+1:m, step].
            // But first apply the reflector to the trailing submatrix.

            // Apply reflector to trailing submatrix work[step:m, step:n].
            apply_householder_left(&v, tau, &mut work, step, step, col_len, n - step);

            // Store Householder vector below diagonal (v[1..]).
            for i in 1..col_len {
                work.set(step + i, step, v[i]);
            }
        }

        Ok(QrDecomposition {
            qr: work,
            tau: tau_vec,
            m,
            n,
        })
    }

    /// Extract the upper-triangular matrix R (min(m,n) × n).
    pub fn get_r(&self) -> DenseMatrix {
        let k = self.m.min(self.n);
        let mut r = DenseMatrix::zeros(k, self.n);
        for i in 0..k {
            for j in i..self.n {
                r.set(i, j, self.qr.get(i, j));
            }
        }
        r
    }

    /// Form the full orthogonal matrix Q (m × m) by back-accumulating
    /// Householder reflectors applied to the identity.
    pub fn get_q(&self) -> DenseMatrix {
        let m = self.m;
        let k = m.min(self.n);
        let mut q = DenseMatrix::eye(m);

        // Apply reflectors in reverse order.
        for step in (0..k).rev() {
            let col_len = m - step;
            let v = self.extract_v(step, col_len);
            let tau = self.tau[step];
            apply_householder_left(&v, tau, &mut q, step, step, col_len, m - step);
        }
        q
    }

    /// Form the thin Q (m × min(m,n)).
    pub fn get_q_thin(&self) -> DenseMatrix {
        let m = self.m;
        let k = m.min(self.n);
        // Start with the first k columns of the identity.
        let mut qt = DenseMatrix::zeros(m, k);
        for i in 0..k {
            qt.set(i, i, 1.0);
        }

        // Apply reflectors in reverse order.
        for step in (0..k).rev() {
            let col_len = m - step;
            let v = self.extract_v(step, col_len);
            let tau = self.tau[step];
            apply_householder_left(&v, tau, &mut qt, step, step, col_len, k - step);
        }
        qt
    }

    /// Apply Q^T to B from the left, in-place, without forming Q.
    ///
    /// B must have m rows.
    pub fn apply_qt(&self, b: &mut DenseMatrix) {
        let m = self.m;
        let k = m.min(self.n);
        let b_cols = b.cols;

        for step in 0..k {
            let col_len = m - step;
            let v = self.extract_v(step, col_len);
            let tau = self.tau[step];
            apply_householder_left(&v, tau, b, step, 0, col_len, b_cols);
        }
    }

    /// Apply Q to B from the left, in-place, without forming Q.
    ///
    /// B must have m rows. Reflectors are applied in reverse order.
    pub fn apply_q(&self, b: &mut DenseMatrix) {
        let m = self.m;
        let k = m.min(self.n);
        let b_cols = b.cols;

        for step in (0..k).rev() {
            let col_len = m - step;
            let v = self.extract_v(step, col_len);
            let tau = self.tau[step];
            apply_householder_left(&v, tau, b, step, 0, col_len, b_cols);
        }
    }

    /// Solve Ax = b for square or overdetermined systems via QR.
    ///
    /// For an m×n matrix with m ≥ n, computes x = R⁻¹ (Q^T b).
    /// Returns an error if R is singular (zero diagonal element).
    pub fn solve(&self, b: &[f64]) -> DecompResult<Vec<f64>> {
        let m = self.m;
        let n = self.n;
        if b.len() != m {
            return Err(DecompError::VectorLengthMismatch {
                expected: m,
                actual: b.len(),
            });
        }
        if m < n {
            return Err(DecompError::Internal(
                "solve requires m >= n (overdetermined or square system)".into(),
            ));
        }

        // Compute Q^T * b using the compact form.
        let mut qtb = DenseMatrix::from_row_major(m, 1, b.to_vec());
        self.apply_qt(&mut qtb);

        // Extract the first n elements.
        let rhs: Vec<f64> = (0..n).map(|i| qtb.get(i, 0)).collect();

        // Extract R (n×n upper triangle).
        let r = self.get_r();

        // Back-substitute.
        back_solve_upper_triangular(&r, &rhs)
    }

    /// Solve the least-squares problem min ‖Ax − b‖₂.
    ///
    /// For m×n with m ≥ n, returns x = R⁻¹ (Q^T b)[0..n].
    pub fn solve_least_squares(&self, b: &[f64]) -> DecompResult<Vec<f64>> {
        let m = self.m;
        let n = self.n;
        if b.len() != m {
            return Err(DecompError::VectorLengthMismatch {
                expected: m,
                actual: b.len(),
            });
        }

        // Compute Q^T * b.
        let mut qtb = DenseMatrix::from_row_major(m, 1, b.to_vec());
        self.apply_qt(&mut qtb);

        // Take the first n entries and back-solve with R.
        let neff = n.min(m);
        let rhs: Vec<f64> = (0..neff).map(|i| qtb.get(i, 0)).collect();

        let r = self.get_r();
        // Use the neff × neff leading sub-block of R.
        let r_sub = if r.rows == neff && r.cols >= neff {
            r
        } else {
            let nr = neff.min(r.rows);
            let nc = neff.min(r.cols);
            let mut rs = DenseMatrix::zeros(nr, nc);
            for i in 0..nr {
                for j in i..nc {
                    rs.set(i, j, r.get(i, j));
                }
            }
            rs
        };

        back_solve_upper_triangular(&r_sub, &rhs[..r_sub.rows])
    }

    /// Estimate the numerical rank by counting diagonal entries of R
    /// with absolute value exceeding `tol`.
    pub fn rank(&self, tol: f64) -> usize {
        let k = self.m.min(self.n);
        let mut r = 0;
        for i in 0..k {
            if self.qr.get(i, i).abs() > tol {
                r += 1;
            }
        }
        r
    }

    // ── private helpers ────────────────────────────────────────────────

    /// Reconstruct the Householder vector for step `step`.
    /// v[0] = 1, v[1..] are stored below the diagonal.
    fn extract_v(&self, step: usize, col_len: usize) -> Vec<f64> {
        let mut v = vec![0.0; col_len];
        v[0] = 1.0;
        for i in 1..col_len {
            v[i] = self.qr.get(step + i, step);
        }
        v
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Column-pivoting (rank-revealing) QR
// ═══════════════════════════════════════════════════════════════════════════

/// QR decomposition with column pivoting: A * P = Q * R.
///
/// The column permutation is chosen so that |R[i,i]| ≥ |R[j,j]| for i < j,
/// making this a rank-revealing factorization.
#[derive(Clone, Debug)]
pub struct QrPivoted {
    /// Combined storage (same layout as [`QrDecomposition`]).
    pub qr: DenseMatrix,
    /// Householder scalars.
    pub tau: Vec<f64>,
    /// Column permutation: column `j` of the original matrix is column `perm[j]`.
    pub perm: Vec<usize>,
    pub m: usize,
    pub n: usize,
}

impl QrPivoted {
    /// Compute a column-pivoted QR factorization of `a`.
    ///
    /// At each step, the column with the largest remaining 2-norm is chosen
    /// as the next pivot. This provides a rank-revealing decomposition.
    pub fn factorize_pivoted(a: &DenseMatrix) -> DecompResult<Self> {
        let m = a.rows;
        let n = a.cols;
        DecompError::check_non_empty(m, n, "Pivoted QR factorization")?;

        let mut work = a.clone();
        let k = m.min(n);
        let mut tau_vec = vec![0.0; k];

        // Column permutation (initially identity).
        let mut perm: Vec<usize> = (0..n).collect();

        // Precompute column norms.
        let mut col_norms: Vec<f64> = (0..n)
            .map(|j| {
                let c = work.col(j);
                norm2(&c)
            })
            .collect();

        for step in 0..k {
            // Find the column with the largest remaining norm among step..n.
            let mut best_col = step;
            let mut best_norm = col_norms[step];
            for j in (step + 1)..n {
                if col_norms[j] > best_norm {
                    best_norm = col_norms[j];
                    best_col = j;
                }
            }

            // Swap columns `step` and `best_col` in work, perm, and col_norms.
            if best_col != step {
                for i in 0..m {
                    let a_val = work.get(i, step);
                    let b_val = work.get(i, best_col);
                    work.set(i, step, b_val);
                    work.set(i, best_col, a_val);
                }
                perm.swap(step, best_col);
                col_norms.swap(step, best_col);
            }

            // Extract column work[step:m, step].
            let col_len = m - step;
            let mut col = vec![0.0; col_len];
            for i in 0..col_len {
                col[i] = work.get(step + i, step);
            }

            let (v, tau) = generate_householder(&col);
            tau_vec[step] = tau;

            // Apply reflector to trailing submatrix.
            apply_householder_left(&v, tau, &mut work, step, step, col_len, n - step);

            // Store Householder vector below diagonal.
            for i in 1..col_len {
                work.set(step + i, step, v[i]);
            }

            // Update column norms for remaining columns.
            // Instead of recomputing, we use a cheap downdate.
            for j in (step + 1)..n {
                if col_norms[j] > 1e-300 {
                    let r_kj = work.get(step, j);
                    let temp = r_kj / col_norms[j];
                    let temp = (1.0 - temp * temp).max(0.0);
                    if temp < 1e-8 {
                        // Recompute from scratch to avoid cancellation.
                        let mut s = 0.0;
                        for i in (step + 1)..m {
                            let v = work.get(i, j);
                            s += v * v;
                        }
                        col_norms[j] = s.sqrt();
                    } else {
                        col_norms[j] *= temp.sqrt();
                    }
                }
            }
        }

        Ok(QrPivoted {
            qr: work,
            tau: tau_vec,
            perm,
            m,
            n,
        })
    }

    /// Estimate the numerical rank by counting |R[i,i]| > tol.
    pub fn rank(&self, tol: f64) -> usize {
        let k = self.m.min(self.n);
        let mut r = 0;
        for i in 0..k {
            if self.qr.get(i, i).abs() > tol {
                r += 1;
            }
        }
        r
    }

    /// Extract the upper-triangular R (min(m,n) × n).
    pub fn get_r(&self) -> DenseMatrix {
        let k = self.m.min(self.n);
        let mut r = DenseMatrix::zeros(k, self.n);
        for i in 0..k {
            for j in i..self.n {
                r.set(i, j, self.qr.get(i, j));
            }
        }
        r
    }

    /// Form the full orthogonal matrix Q (m × m).
    pub fn get_q(&self) -> DenseMatrix {
        let m = self.m;
        let k = m.min(self.n);
        let mut q = DenseMatrix::eye(m);

        for step in (0..k).rev() {
            let col_len = m - step;
            let mut v = vec![0.0; col_len];
            v[0] = 1.0;
            for i in 1..col_len {
                v[i] = self.qr.get(step + i, step);
            }
            let tau = self.tau[step];
            apply_householder_left(&v, tau, &mut q, step, step, col_len, m - step);
        }
        q
    }

    /// Return the column permutation vector.
    ///
    /// `perm[j]` is the original column index that ended up in position `j`.
    pub fn get_perm(&self) -> Vec<usize> {
        self.perm.clone()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Givens-based QR
// ═══════════════════════════════════════════════════════════════════════════

/// Compute QR decomposition using Givens rotations.
///
/// Returns `(Q, R)` where Q is m×m orthogonal and R is m×n upper triangular.
/// This approach is more expensive than Householder for dense matrices but
/// can be advantageous for sparse/banded matrices.
pub fn qr_givens(a: &DenseMatrix) -> DecompResult<(DenseMatrix, DenseMatrix)> {
    let m = a.rows;
    let n = a.cols;
    DecompError::check_non_empty(m, n, "Givens QR")?;

    let mut r = a.clone();
    let mut q = DenseMatrix::eye(m);

    for j in 0..n.min(m) {
        // Zero out entries below the diagonal in column j.
        for i in (j + 1..m).rev() {
            let a_val = r.get(i - 1, j);
            let b_val = r.get(i, j);
            if b_val.abs() < 1e-300 {
                continue;
            }

            let rot = GivensRotation::from_values(a_val, b_val);

            // Apply rotation to rows (i-1, i) of R for columns j..n.
            for col in j..n {
                let ri = r.get(i - 1, col);
                let rj = r.get(i, col);
                let (new_a, new_b) = rot.apply(ri, rj);
                r.set(i - 1, col, new_a);
                r.set(i, col, new_b);
            }

            // Accumulate Q: apply rotation to columns (i-1, i) of Q^T,
            // which is the same as applying to rows of Q.
            // Q = Q * G^T, so for each row of Q: update cols (i-1, i).
            for row in 0..m {
                let qi = q.get(row, i - 1);
                let qj = q.get(row, i);
                let (new_a, new_b) = rot.apply(qi, qj);
                q.set(row, i - 1, new_a);
                q.set(row, i, new_b);
            }
        }
    }

    Ok((q, r))
}

// ═══════════════════════════════════════════════════════════════════════════
// Sparse QR (via conversion)
// ═══════════════════════════════════════════════════════════════════════════

/// Compute the QR factorization of a sparse CSR matrix by converting to dense.
///
/// This is a convenience wrapper; for large sparse matrices a dedicated
/// sparse QR (e.g., SuiteSparseQR) should be used instead.
pub fn factorize_sparse(a: &CsrMatrix) -> DecompResult<QrDecomposition> {
    DecompError::check_non_empty(a.rows, a.cols, "Sparse QR factorization")?;
    let dense = a.to_dense();
    QrDecomposition::factorize(&dense)
}

// ═══════════════════════════════════════════════════════════════════════════
// Back-substitution
// ═══════════════════════════════════════════════════════════════════════════

/// Solve `R x = b` where R is upper triangular.
///
/// R must have at least as many columns as rows (i.e., R is k×n with k ≤ n)
/// and the k×k leading sub-block must be non-singular.
///
/// Returns a vector of length `n`. For a square system (k = n) this is
/// the unique solution. For an underdetermined system (k < n), the extra
/// unknowns are set to zero (this corresponds to the minimum-norm solution
/// only if combined with a suitable column permutation).
pub fn back_solve_upper_triangular(r: &DenseMatrix, b: &[f64]) -> DecompResult<Vec<f64>> {
    let k = r.rows; // number of equations
    let n = r.cols; // number of unknowns
    if b.len() != k {
        return Err(DecompError::VectorLengthMismatch {
            expected: k,
            actual: b.len(),
        });
    }
    if k == 0 {
        return Ok(vec![0.0; n]);
    }

    let mut x = vec![0.0; n];

    // Back substitution: solve from row k-1 up to row 0.
    for i in (0..k).rev() {
        let diag = r.get(i, i);
        if diag.abs() < 1e-15 {
            return Err(DecompError::SingularMatrix {
                context: format!("zero diagonal in R at index {}", i),
            });
        }
        let mut sum = b[i];
        for j in (i + 1)..n {
            sum -= r.get(i, j) * x[j];
        }
        x[i] = sum / diag;
    }

    Ok(x)
}

/// Forward-solve `L x = b` where L is lower triangular.
///
/// L must be square (n×n). Returns a vector of length n.
/// This is the dual of [`back_solve_upper_triangular`] and is provided for
/// completeness (useful internally for LQ-type operations).
pub fn forward_solve_lower_triangular(l: &DenseMatrix, b: &[f64]) -> DecompResult<Vec<f64>> {
    let n = l.rows;
    if l.cols != n {
        return Err(DecompError::NotSquare {
            rows: l.rows,
            cols: l.cols,
        });
    }
    if b.len() != n {
        return Err(DecompError::VectorLengthMismatch {
            expected: n,
            actual: b.len(),
        });
    }
    if n == 0 {
        return Ok(vec![]);
    }

    let mut x = vec![0.0; n];
    for i in 0..n {
        let diag = l.get(i, i);
        if diag.abs() < 1e-15 {
            return Err(DecompError::SingularMatrix {
                context: format!("zero diagonal in L at index {}", i),
            });
        }
        let mut sum = b[i];
        for j in 0..i {
            sum -= l.get(i, j) * x[j];
        }
        x[i] = sum / diag;
    }
    Ok(x)
}

/// Compute the condition number estimate of R from a QR factorization.
///
/// Uses the simple heuristic `|R[0,0]| / |R[k-1,k-1]|` where k = min(m,n).
/// This is not a rigorous condition number but is useful for quick checks.
pub fn condition_estimate_from_r(r: &DenseMatrix) -> f64 {
    let k = r.rows.min(r.cols);
    if k == 0 {
        return f64::INFINITY;
    }
    let max_diag = r.get(0, 0).abs();
    let min_diag = r.get(k - 1, k - 1).abs();
    if min_diag < 1e-300 {
        f64::INFINITY
    } else {
        max_diag / min_diag
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-10;

    /// Check that two matrices are approximately equal.
    fn assert_mat_approx(a: &DenseMatrix, b: &DenseMatrix, tol: f64) {
        assert_eq!(a.rows, b.rows, "row mismatch: {} vs {}", a.rows, b.rows);
        assert_eq!(a.cols, b.cols, "col mismatch: {} vs {}", a.cols, b.cols);
        for i in 0..a.rows {
            for j in 0..a.cols {
                let diff = (a.get(i, j) - b.get(i, j)).abs();
                assert!(
                    diff < tol,
                    "mismatch at ({},{}): {} vs {} (diff={})",
                    i, j, a.get(i, j), b.get(i, j), diff
                );
            }
        }
    }

    /// Check that a matrix is upper triangular (below-diagonal entries ≈ 0).
    fn assert_upper_triangular(r: &DenseMatrix, tol: f64) {
        for i in 0..r.rows {
            for j in 0..i.min(r.cols) {
                assert!(
                    r.get(i, j).abs() < tol,
                    "R[{},{}] = {} is not zero (upper triangular check)",
                    i, j, r.get(i, j)
                );
            }
        }
    }

    /// Check that Q^T * Q ≈ I.
    fn assert_orthogonal(q: &DenseMatrix, tol: f64) {
        let qtq = q.transpose().mul_mat(q).unwrap();
        let eye = DenseMatrix::eye(q.cols);
        assert_mat_approx(&qtq, &eye, tol);
    }

    // ── Test 1: QR of identity ──────────────────────────────────────────

    #[test]
    fn test_qr_identity() {
        let a = DenseMatrix::eye(4);
        let qr = QrDecomposition::factorize(&a).unwrap();
        let q = qr.get_q();
        let r = qr.get_r();

        // Q should be identity (or differ only by sign flips on columns).
        assert_orthogonal(&q, EPS);
        assert_upper_triangular(&r, EPS);

        // Q * R should reconstruct A.
        let qr_prod = q.mul_mat(&r).unwrap();
        assert_mat_approx(&qr_prod, &a, EPS);
    }

    // ── Test 2: Known 3×3 matrix ────────────────────────────────────────

    #[test]
    fn test_qr_3x3() {
        let a = DenseMatrix::from_row_major(3, 3, vec![
            12.0, -51.0,   4.0,
             6.0, 167.0, -68.0,
            -4.0,  24.0, -41.0,
        ]);
        let qr = QrDecomposition::factorize(&a).unwrap();
        let q = qr.get_q();
        let r = qr.get_r();

        // Check Q^T Q = I.
        assert_orthogonal(&q, EPS);

        // Check R is upper triangular.
        assert_upper_triangular(&r, EPS);

        // Check Q*R ≈ A.
        let qr_prod = q.mul_mat(&r).unwrap();
        assert_mat_approx(&qr_prod, &a, EPS);
    }

    // ── Test 3: Solve overdetermined system (least squares) ─────────────

    #[test]
    fn test_solve_least_squares() {
        // Overdetermined: 4×2 system, fit y = a + b*x to data.
        // x = [0, 1, 2, 3], y = [1, 2, 3, 4]  →  exact fit y = 1 + x.
        let a = DenseMatrix::from_row_major(4, 2, vec![
            1.0, 0.0,
            1.0, 1.0,
            1.0, 2.0,
            1.0, 3.0,
        ]);
        let b = vec![1.0, 2.0, 3.0, 4.0];

        let qr = QrDecomposition::factorize(&a).unwrap();
        let x = qr.solve_least_squares(&b).unwrap();

        assert!((x[0] - 1.0).abs() < 1e-10, "intercept: {}", x[0]);
        assert!((x[1] - 1.0).abs() < 1e-10, "slope: {}", x[1]);
    }

    // ── Test 4: Rank estimation ─────────────────────────────────────────

    #[test]
    fn test_rank_estimation() {
        // Rank-2 matrix (3×3 with linearly dependent rows).
        let a = DenseMatrix::from_row_major(3, 3, vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            2.0, 4.0, 6.0, // = 2 * row 0
        ]);
        let qr = QrDecomposition::factorize(&a).unwrap();
        let r = qr.rank(1e-10);
        assert_eq!(r, 2, "expected rank 2, got {}", r);
    }

    // ── Test 5: Column-pivoted QR rank-revealing ────────────────────────

    #[test]
    fn test_pivoted_qr_rank_revealing() {
        // 4×3 matrix of rank 2.
        let a = DenseMatrix::from_row_major(4, 3, vec![
            1.0, 0.0, 1.0,
            0.0, 1.0, 1.0,
            1.0, 1.0, 2.0, // row0 + row1
            2.0, 1.0, 3.0, // 2*row0 + row1
        ]);
        let piv = QrPivoted::factorize_pivoted(&a).unwrap();
        let rk = piv.rank(1e-10);
        assert_eq!(rk, 2, "expected rank 2, got {}", rk);

        // Q * R reconstructs A * P.
        let q = piv.get_q();
        let r = piv.get_r();
        assert_orthogonal(&q, EPS);
        assert_upper_triangular(&r, EPS);

        // Reconstruct A_permuted = Q * R, then un-permute columns.
        let ap = q.mul_mat(&r).unwrap();
        let perm = piv.get_perm();
        // Build the un-permuted matrix: original col perm[j] = ap col j.
        let mut reconstructed = DenseMatrix::zeros(a.rows, a.cols);
        for j in 0..a.cols {
            for i in 0..a.rows {
                reconstructed.set(i, perm[j], ap.get(i, j));
            }
        }
        assert_mat_approx(&reconstructed, &a, 1e-9);
    }

    // ── Test 6: Q^T application without forming Q ───────────────────────

    #[test]
    fn test_apply_qt_no_form() {
        let a = DenseMatrix::from_row_major(3, 3, vec![
            1.0, 1.0, 0.0,
            1.0, 0.0, 1.0,
            0.0, 1.0, 1.0,
        ]);
        let qr = QrDecomposition::factorize(&a).unwrap();

        // Build b = [1, 2, 3]^T as a column matrix.
        let mut b_mat = DenseMatrix::from_row_major(3, 1, vec![1.0, 2.0, 3.0]);
        let b_mat_orig = b_mat.clone();

        // Apply Q^T in-place.
        qr.apply_qt(&mut b_mat);

        // Compare with explicitly forming Q.
        let q = qr.get_q();
        let qt_b = q.transpose().mul_mat(&b_mat_orig).unwrap();
        assert_mat_approx(&b_mat, &qt_b, EPS);
    }

    // ── Test 7: Givens QR matches Householder QR ────────────────────────

    #[test]
    fn test_givens_vs_householder() {
        let a = DenseMatrix::from_row_major(4, 3, vec![
             1.0, -1.0,  4.0,
             1.0,  4.0, -2.0,
             1.0,  4.0,  2.0,
             1.0, -1.0,  0.0,
        ]);

        let (q_g, r_g) = qr_givens(&a).unwrap();
        let qr_h = QrDecomposition::factorize(&a).unwrap();
        let q_h = qr_h.get_q();
        let r_h = qr_h.get_r();

        // Both should reconstruct A.
        let prod_g = q_g.mul_mat(&r_g).unwrap();
        // For Givens, R is m×n, but upper-triangular only in the first min(m,n) rows.
        assert_mat_approx(&prod_g, &a, 1e-9);

        let prod_h = q_h.mul_mat(&r_h).unwrap();
        assert_mat_approx(&prod_h, &a, 1e-9);

        // Both should yield orthogonal Q.
        assert_orthogonal(&q_g, 1e-9);
        assert_orthogonal(&q_h, 1e-9);

        // R from Givens should be upper triangular.
        assert_upper_triangular(&r_g, 1e-9);
    }

    // ── Test 8: Sparse QR ───────────────────────────────────────────────

    #[test]
    fn test_sparse_qr() {
        // Build a sparse identity-ish matrix.
        let triplets = vec![
            (0, 0, 2.0), (0, 1, 1.0),
            (1, 0, 1.0), (1, 1, 3.0),
            (2, 0, 0.5), (2, 1, 0.5), (2, 2, 4.0),
        ];
        let csr = CsrMatrix::from_triplets(3, 3, &triplets);
        let qr = factorize_sparse(&csr).unwrap();
        let q = qr.get_q();
        let r = qr.get_r();

        assert_orthogonal(&q, 1e-9);
        assert_upper_triangular(&r, 1e-9);

        let prod = q.mul_mat(&r).unwrap();
        let dense = csr.to_dense();
        assert_mat_approx(&prod, &dense, 1e-9);
    }

    // ── Test 9: 1×1 case ────────────────────────────────────────────────

    #[test]
    fn test_qr_1x1() {
        let a = DenseMatrix::from_row_major(1, 1, vec![5.0]);
        let qr = QrDecomposition::factorize(&a).unwrap();
        let q = qr.get_q();
        let r = qr.get_r();
        let prod = q.mul_mat(&r).unwrap();
        assert_mat_approx(&prod, &a, EPS);

        // Solve 5x = 10 → x = 2.
        let x = qr.solve(&[10.0]).unwrap();
        assert!((x[0] - 2.0).abs() < EPS);
    }

    // ── Test 10: Tall thin matrix (m >> n) ──────────────────────────────

    #[test]
    fn test_qr_tall_thin() {
        // 6×2 matrix.
        let a = DenseMatrix::from_row_major(6, 2, vec![
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0,
            7.0, 8.0,
            9.0, 10.0,
            11.0, 12.0,
        ]);
        let qr = QrDecomposition::factorize(&a).unwrap();
        let q = qr.get_q();
        let r = qr.get_r();

        assert_eq!(q.rows, 6);
        assert_eq!(q.cols, 6);
        assert_eq!(r.rows, 2);
        assert_eq!(r.cols, 2);

        assert_orthogonal(&q, 1e-9);
        assert_upper_triangular(&r, 1e-9);

        // Q * R should give a 6×2 matrix ≈ A.
        let prod = q.mul_mat(&r).unwrap();
        assert_mat_approx(&prod, &a, 1e-9);

        // Thin Q test.
        let qt = qr.get_q_thin();
        assert_eq!(qt.rows, 6);
        assert_eq!(qt.cols, 2);

        // Qt^T * Qt should be 2×2 identity.
        let qtq = qt.transpose().mul_mat(&qt).unwrap();
        assert_mat_approx(&qtq, &DenseMatrix::eye(2), 1e-9);

        // Qt * R ≈ A.
        let prod2 = qt.mul_mat(&r).unwrap();
        assert_mat_approx(&prod2, &a, 1e-9);
    }

    // ── Test 11: Wide short matrix (m < n) ──────────────────────────────

    #[test]
    fn test_qr_wide_short() {
        // 2×4 matrix.
        let a = DenseMatrix::from_row_major(2, 4, vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
        ]);
        let qr = QrDecomposition::factorize(&a).unwrap();
        let q = qr.get_q();
        let r = qr.get_r();

        assert_eq!(q.rows, 2);
        assert_eq!(q.cols, 2);
        assert_eq!(r.rows, 2);
        assert_eq!(r.cols, 4);

        assert_orthogonal(&q, 1e-9);
        assert_upper_triangular(&r, 1e-9);

        let prod = q.mul_mat(&r).unwrap();
        assert_mat_approx(&prod, &a, 1e-9);
    }

    // ── Test 12: Back substitution ──────────────────────────────────────

    #[test]
    fn test_back_substitution() {
        // Upper triangular R:
        // [2  1  3]
        // [0  4  2]
        // [0  0  5]
        let r = DenseMatrix::from_row_major(3, 3, vec![
            2.0, 1.0, 3.0,
            0.0, 4.0, 2.0,
            0.0, 0.0, 5.0,
        ]);
        // b = R * [1, 2, 3] = [2+2+9, 8+6, 15] = [13, 14, 15]
        let b = vec![13.0, 14.0, 15.0];
        let x = back_solve_upper_triangular(&r, &b).unwrap();
        assert!((x[0] - 1.0).abs() < EPS, "x[0] = {}", x[0]);
        assert!((x[1] - 2.0).abs() < EPS, "x[1] = {}", x[1]);
        assert!((x[2] - 3.0).abs() < EPS, "x[2] = {}", x[2]);
    }

    // ── Test 13: Solve square system via QR ─────────────────────────────

    #[test]
    fn test_solve_square() {
        let a = DenseMatrix::from_row_major(3, 3, vec![
            1.0, 2.0, 3.0,
            0.0, 1.0, 4.0,
            5.0, 6.0, 0.0,
        ]);
        // True x = [1, 2, 3].
        // b = A * x.
        let b_vec = a.mul_vec(&[1.0, 2.0, 3.0]).unwrap();

        let qr = QrDecomposition::factorize(&a).unwrap();
        let x = qr.solve(&b_vec).unwrap();

        assert!((x[0] - 1.0).abs() < 1e-9, "x[0] = {}", x[0]);
        assert!((x[1] - 2.0).abs() < 1e-9, "x[1] = {}", x[1]);
        assert!((x[2] - 3.0).abs() < 1e-9, "x[2] = {}", x[2]);
    }

    // ── Test 14: apply_q round-trips with apply_qt ──────────────────────

    #[test]
    fn test_apply_q_roundtrip() {
        let a = DenseMatrix::from_row_major(4, 3, vec![
            2.0, -1.0,  0.0,
            1.0,  3.0, -1.0,
            0.0,  1.0,  2.0,
            1.0,  0.0,  1.0,
        ]);
        let qr = QrDecomposition::factorize(&a).unwrap();

        let mut b = DenseMatrix::from_row_major(4, 1, vec![1.0, 2.0, 3.0, 4.0]);
        let b_orig = b.clone();

        // Q^T * b, then Q * (Q^T * b) should give back b.
        qr.apply_qt(&mut b);
        qr.apply_q(&mut b);
        assert_mat_approx(&b, &b_orig, 1e-9);
    }

    // ── Test 15: Pivoted QR on full-rank matrix ─────────────────────────

    #[test]
    fn test_pivoted_qr_full_rank() {
        let a = DenseMatrix::from_row_major(3, 3, vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 10.0, // not 9 → full rank
        ]);
        let piv = QrPivoted::factorize_pivoted(&a).unwrap();
        assert_eq!(piv.rank(1e-10), 3);

        let q = piv.get_q();
        let r = piv.get_r();
        assert_orthogonal(&q, 1e-9);
        assert_upper_triangular(&r, 1e-9);

        // Reconstruct.
        let ap = q.mul_mat(&r).unwrap();
        let perm = piv.get_perm();
        let mut reconstructed = DenseMatrix::zeros(3, 3);
        for j in 0..3 {
            for i in 0..3 {
                reconstructed.set(i, perm[j], ap.get(i, j));
            }
        }
        assert_mat_approx(&reconstructed, &a, 1e-9);
    }

    // ── Test 16: Back-solve detects singular R ──────────────────────────

    #[test]
    fn test_back_solve_singular() {
        let r = DenseMatrix::from_row_major(2, 2, vec![
            1.0, 2.0,
            0.0, 0.0,
        ]);
        let result = back_solve_upper_triangular(&r, &[1.0, 2.0]);
        assert!(result.is_err());
    }

    // ── Test 17: Forward substitution ───────────────────────────────────

    #[test]
    fn test_forward_substitution() {
        // L = [2 0 0; 1 3 0; 4 2 5]
        let l = DenseMatrix::from_row_major(3, 3, vec![
            2.0, 0.0, 0.0,
            1.0, 3.0, 0.0,
            4.0, 2.0, 5.0,
        ]);
        // x = [1, 2, 3], b = L*x = [2, 7, 23]
        let b = vec![2.0, 7.0, 23.0];
        let x = forward_solve_lower_triangular(&l, &b).unwrap();
        assert!((x[0] - 1.0).abs() < EPS, "x[0] = {}", x[0]);
        assert!((x[1] - 2.0).abs() < EPS, "x[1] = {}", x[1]);
        assert!((x[2] - 3.0).abs() < EPS, "x[2] = {}", x[2]);
    }

    // ── Test 18: Condition estimate from R ──────────────────────────────

    #[test]
    fn test_condition_estimate() {
        // Well-conditioned diagonal matrix.
        let r = DenseMatrix::from_row_major(3, 3, vec![
            10.0, 0.0, 0.0,
             0.0, 5.0, 0.0,
             0.0, 0.0, 2.0,
        ]);
        let cond = condition_estimate_from_r(&r);
        assert!((cond - 5.0).abs() < EPS, "cond = {}", cond);

        // Near-singular diagonal.
        let r2 = DenseMatrix::from_row_major(2, 2, vec![
            100.0, 0.0,
              0.0, 1e-14,
        ]);
        let cond2 = condition_estimate_from_r(&r2);
        assert!(cond2 > 1e15);
    }

    // ── Test 19: Solve with non-trivial RHS matrix ──────────────────────

    #[test]
    fn test_qr_solve_multiple_rhs() {
        let a = DenseMatrix::from_row_major(3, 3, vec![
            2.0, 1.0, 0.0,
            1.0, 3.0, 1.0,
            0.0, 1.0, 2.0,
        ]);
        let qr = QrDecomposition::factorize(&a).unwrap();

        // Solve for two different right-hand sides.
        let x1_true = vec![1.0, 0.0, 0.0];
        let b1 = a.mul_vec(&x1_true).unwrap();
        let x1 = qr.solve(&b1).unwrap();
        for i in 0..3 {
            assert!(
                (x1[i] - x1_true[i]).abs() < 1e-9,
                "x1[{}] = {} vs {}",
                i, x1[i], x1_true[i]
            );
        }

        let x2_true = vec![0.0, 1.0, -1.0];
        let b2 = a.mul_vec(&x2_true).unwrap();
        let x2 = qr.solve(&b2).unwrap();
        for i in 0..3 {
            assert!(
                (x2[i] - x2_true[i]).abs() < 1e-9,
                "x2[{}] = {} vs {}",
                i, x2[i], x2_true[i]
            );
        }
    }

    // ── Test 20: Least squares with noisy data ──────────────────────────

    #[test]
    fn test_least_squares_noisy() {
        // 5 data points, fit y = a + b*x + c*x^2.
        // True: y = 1 + 2x + 0.5x^2.
        let xs = [0.0, 1.0, 2.0, 3.0, 4.0];
        let mut a_data = Vec::new();
        let mut b_data = Vec::new();
        for &x in &xs {
            a_data.push(1.0);
            a_data.push(x);
            a_data.push(x * x);
            b_data.push(1.0 + 2.0 * x + 0.5 * x * x);
        }
        let a = DenseMatrix::from_row_major(5, 3, a_data);
        let qr = QrDecomposition::factorize(&a).unwrap();
        let x = qr.solve_least_squares(&b_data).unwrap();

        assert!((x[0] - 1.0).abs() < 1e-9, "a = {}", x[0]);
        assert!((x[1] - 2.0).abs() < 1e-9, "b = {}", x[1]);
        assert!((x[2] - 0.5).abs() < 1e-9, "c = {}", x[2]);
    }

    // ── Test 21: Givens QR on small 2×2 ─────────────────────────────────

    #[test]
    fn test_givens_2x2() {
        let a = DenseMatrix::from_row_major(2, 2, vec![
            3.0, 1.0,
            4.0, 2.0,
        ]);
        let (q, r) = qr_givens(&a).unwrap();
        assert_orthogonal(&q, 1e-9);
        assert_upper_triangular(&r, 1e-9);
        let prod = q.mul_mat(&r).unwrap();
        assert_mat_approx(&prod, &a, 1e-9);
    }

    // ── Test 22: Thin Q columns are orthonormal ─────────────────────────

    #[test]
    fn test_thin_q_orthonormal_columns() {
        let a = DenseMatrix::from_row_major(5, 3, vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
            1.0, 1.0, 0.0,
            0.0, 1.0, 1.0,
        ]);
        let qr = QrDecomposition::factorize(&a).unwrap();
        let qt = qr.get_q_thin();

        assert_eq!(qt.rows, 5);
        assert_eq!(qt.cols, 3);

        // Check pairwise column orthogonality.
        for i in 0..3 {
            let ci = qt.col(i);
            let ni = norm2(&ci);
            assert!((ni - 1.0).abs() < 1e-9, "col {} norm = {}", i, ni);
            for j in (i + 1)..3 {
                let cj = qt.col(j);
                let d = dot(&ci, &cj);
                assert!(d.abs() < 1e-9, "dot(col{}, col{}) = {}", i, j, d);
            }
        }
    }

    // ── Test 23: QR of zero matrix ──────────────────────────────────────

    #[test]
    fn test_qr_zero_matrix() {
        let a = DenseMatrix::zeros(3, 3);
        let qr = QrDecomposition::factorize(&a).unwrap();
        let r = qr.get_r();

        // R should be all zeros.
        for i in 0..3 {
            for j in 0..3 {
                assert!(r.get(i, j).abs() < EPS, "R[{},{}] = {}", i, j, r.get(i, j));
            }
        }

        // Rank should be 0.
        assert_eq!(qr.rank(1e-10), 0);
    }

    // ── Test 24: Large-ish random-ish matrix factorization ──────────────

    #[test]
    fn test_qr_larger_matrix() {
        // Build a 10×7 matrix with a predictable pattern.
        let m = 10;
        let n = 7;
        let mut data = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                data[i * n + j] = ((i + 1) as f64) * ((j + 1) as f64)
                    + if i == j { 10.0 } else { 0.0 };
            }
        }
        let a = DenseMatrix::from_row_major(m, n, data);
        let qr = QrDecomposition::factorize(&a).unwrap();
        let q = qr.get_q();
        let r = qr.get_r();

        assert_orthogonal(&q, 1e-8);
        assert_upper_triangular(&r, 1e-8);

        let prod = q.mul_mat(&r).unwrap();
        assert_mat_approx(&prod, &a, 1e-8);
    }
}
