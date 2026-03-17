//! LU decomposition with partial pivoting (Doolittle algorithm).
//!
//! Provides both dense and sparse LU factorization, along with solvers,
//! determinant computation, matrix inversion, and condition number estimation.
//!
//! # Example
//! ```ignore
//! use matrix_decomp::{DenseMatrix, lu::LuDecomposition};
//! let a = DenseMatrix::from_row_major(3, 3, vec![
//!     2.0, 1.0, 1.0,
//!     4.0, 3.0, 3.0,
//!     8.0, 7.0, 9.0,
//! ]);
//! let lu = LuDecomposition::factorize(&a).unwrap();
//! let x = lu.solve(&[1.0, 1.0, 1.0]).unwrap();
//! ```

use crate::{CsrMatrix, DecompError, DecompResult, DenseMatrix};

// ═══════════════════════════════════════════════════════════════════════════
// Helper: forward substitution  (L y = b, unit lower triangular)
// ═══════════════════════════════════════════════════════════════════════════

/// Solve `L y = b` where `L` is a unit lower-triangular `n×n` dense matrix
/// stored in row-major order. Returns `y`.
pub fn forward_solve(l: &DenseMatrix, b: &[f64]) -> DecompResult<Vec<f64>> {
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
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut s = b[i];
        for j in 0..i {
            s -= l.get(i, j) * y[j];
        }
        // Unit diagonal ⇒ no division needed
        y[i] = s;
    }
    Ok(y)
}

// ═══════════════════════════════════════════════════════════════════════════
// Helper: back substitution  (U x = y, upper triangular)
// ═══════════════════════════════════════════════════════════════════════════

/// Solve `U x = y` where `U` is upper-triangular `n×n`. Returns `x`.
pub fn back_solve(u: &DenseMatrix, y: &[f64]) -> DecompResult<Vec<f64>> {
    let n = u.rows;
    if u.cols != n {
        return Err(DecompError::NotSquare {
            rows: u.rows,
            cols: u.cols,
        });
    }
    if y.len() != n {
        return Err(DecompError::VectorLengthMismatch {
            expected: n,
            actual: y.len(),
        });
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let diag = u.get(i, i);
        if diag.abs() < 1e-300 {
            return Err(DecompError::ZeroPivot {
                index: i,
                value: diag,
            });
        }
        let mut s = y[i];
        for j in (i + 1)..n {
            s -= u.get(i, j) * x[j];
        }
        x[i] = s / diag;
    }
    Ok(x)
}

// ═══════════════════════════════════════════════════════════════════════════
// Dense LU decomposition
// ═══════════════════════════════════════════════════════════════════════════

/// LU factorization of a square matrix with partial (row) pivoting.
///
/// Stores the combined L and U factors in a single matrix: the strict lower
/// triangle holds the multipliers (L has unit diagonal), and the upper
/// triangle (including diagonal) holds U.
#[derive(Clone, Debug)]
pub struct LuDecomposition {
    /// Combined L\U storage (n × n, row-major).
    lu: DenseMatrix,
    /// Row-permutation vector: `pivot[k]` is the row index that was moved
    /// into position `k` during the k-th elimination step.
    pivot: Vec<usize>,
    /// Matrix dimension.
    n: usize,
    /// Sign of the permutation (+1 or −1).
    sign: i32,
}

impl LuDecomposition {
    // ── Construction ────────────────────────────────────────────────────

    /// Doolittle LU factorization with partial pivoting.
    ///
    /// Given an n×n matrix A, computes P, L, U such that P·A = L·U where
    /// L is unit lower-triangular and U is upper-triangular.
    pub fn factorize(a: &DenseMatrix) -> DecompResult<Self> {
        if a.is_empty() {
            return Err(DecompError::empty("LU factorize: empty matrix"));
        }
        if !a.is_square() {
            return Err(DecompError::NotSquare {
                rows: a.rows,
                cols: a.cols,
            });
        }
        let n = a.rows;
        let mut lu = a.clone();
        let mut pivot: Vec<usize> = (0..n).collect();
        let mut sign: i32 = 1;

        // Relative tolerance for detecting singular matrices
        let max_abs = a.data.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
        let _singular_tol = (n as f64) * f64::EPSILON * max_abs;

        for k in 0..n {
            // --- partial pivoting: find row with max |a[i,k]| for i >= k ---
            let mut max_val = lu.get(k, k).abs();
            let mut max_row = k;
            for i in (k + 1)..n {
                let v = lu.get(i, k).abs();
                if v > max_val {
                    max_val = v;
                    max_row = i;
                }
            }

            if max_val < 1e-300 {
                return Err(DecompError::ZeroPivot {
                    index: k,
                    value: lu.get(max_row, k),
                });
            }

            // swap rows k and max_row in LU and pivot vector
            if max_row != k {
                lu.swap_rows(k, max_row);
                pivot.swap(k, max_row);
                sign = -sign;
            }

            let diag = lu.get(k, k);

            // --- compute multipliers and update trailing sub-matrix ---
            for i in (k + 1)..n {
                let mult = lu.get(i, k) / diag;
                lu.set(i, k, mult); // store multiplier in L part
                for j in (k + 1)..n {
                    let val = lu.get(i, j) - mult * lu.get(k, j);
                    lu.set(i, j, val);
                }
            }
        }

        Ok(Self { lu, pivot, n, sign })
    }

    /// Factorize a CSR sparse matrix by converting to dense first.
    pub fn factorize_from_csr(a: &CsrMatrix) -> DecompResult<Self> {
        let dense = a.to_dense();
        Self::factorize(&dense)
    }

    /// Re-factorize in place with a new matrix of the same size.
    pub fn refactorize(&mut self, a: &DenseMatrix) -> DecompResult<()> {
        if a.is_empty() {
            return Err(DecompError::empty("LU refactorize: empty matrix"));
        }
        if !a.is_square() {
            return Err(DecompError::NotSquare {
                rows: a.rows,
                cols: a.cols,
            });
        }
        if a.rows != self.n {
            return Err(DecompError::DimensionMismatch {
                expected_rows: self.n,
                expected_cols: self.n,
                actual_rows: a.rows,
                actual_cols: a.cols,
            });
        }
        let fresh = Self::factorize(a)?;
        self.lu = fresh.lu;
        self.pivot = fresh.pivot;
        self.sign = fresh.sign;
        Ok(())
    }

    // ── Solve ──────────────────────────────────────────────────────────

    /// Solve A x = b.
    ///
    /// Steps: apply permutation to b, forward-substitute (unit L),
    /// back-substitute (U).
    pub fn solve(&self, b: &[f64]) -> DecompResult<Vec<f64>> {
        if b.len() != self.n {
            return Err(DecompError::VectorLengthMismatch {
                expected: self.n,
                actual: b.len(),
            });
        }
        let n = self.n;

        // apply permutation: pb[i] = b[pivot[i]]
        let mut pb = vec![0.0; n];
        for i in 0..n {
            pb[i] = b[self.pivot[i]];
        }

        // forward substitution: L y = pb  (unit lower triangular)
        let mut y = vec![0.0; n];
        for i in 0..n {
            let mut s = pb[i];
            for j in 0..i {
                s -= self.lu.get(i, j) * y[j];
            }
            y[i] = s;
        }

        // back substitution: U x = y
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            let diag = self.lu.get(i, i);
            if diag.abs() < 1e-300 {
                return Err(DecompError::ZeroPivot {
                    index: i,
                    value: diag,
                });
            }
            let mut s = y[i];
            for j in (i + 1)..n {
                s -= self.lu.get(i, j) * x[j];
            }
            x[i] = s / diag;
        }

        Ok(x)
    }

    /// Solve A X = B for multiple right-hand sides (columns of B).
    pub fn solve_multiple(&self, b: &DenseMatrix) -> DecompResult<DenseMatrix> {
        if b.rows != self.n {
            return Err(DecompError::VectorLengthMismatch {
                expected: self.n,
                actual: b.rows,
            });
        }
        let mut result = DenseMatrix::zeros(self.n, b.cols);
        for j in 0..b.cols {
            let col: Vec<f64> = b.col(j);
            let x = self.solve(&col)?;
            for i in 0..self.n {
                result.set(i, j, x[i]);
            }
        }
        Ok(result)
    }

    // ── Determinant ────────────────────────────────────────────────────

    /// Compute det(A) = sign · ∏ U[i,i].
    pub fn determinant(&self) -> f64 {
        let mut det = self.sign as f64;
        for i in 0..self.n {
            det *= self.lu.get(i, i);
        }
        det
    }

    /// Compute ln|det(A)| = Σ ln|U[i,i]|.
    ///
    /// Returns `(log_abs_det, sign)` where sign is +1 or −1 giving the
    /// overall sign of the determinant.
    pub fn log_abs_determinant(&self) -> (f64, i32) {
        let mut log_abs = 0.0;
        let mut det_sign = self.sign;
        for i in 0..self.n {
            let d = self.lu.get(i, i);
            if d < 0.0 {
                det_sign = -det_sign;
            }
            log_abs += d.abs().ln();
        }
        (log_abs, det_sign)
    }

    // ── Inverse ────────────────────────────────────────────────────────

    /// Compute A⁻¹ by solving A X = I column-by-column.
    pub fn inverse(&self) -> DecompResult<DenseMatrix> {
        let n = self.n;
        let mut inv = DenseMatrix::zeros(n, n);
        for j in 0..n {
            let mut e = vec![0.0; n];
            e[j] = 1.0;
            let col = self.solve(&e)?;
            for i in 0..n {
                inv.set(i, j, col[i]);
            }
        }
        Ok(inv)
    }

    // ── Condition number estimate ──────────────────────────────────────

    /// Estimate κ₁(A) = ‖A‖₁ · ‖A⁻¹‖₁ using Hager's algorithm.
    ///
    /// This avoids forming A⁻¹ explicitly by using the LU factors to
    /// solve linear systems.
    pub fn condition_number_estimate(&self) -> DecompResult<f64> {
        let n = self.n;
        if n == 0 {
            return Err(DecompError::empty("condition number: empty matrix"));
        }

        // Reconstruct A from PA = LU ⇒ A = P^T L U
        // Compute ‖A‖₁ from the LU factors.
        let a_norm = self.reconstruct_a_one_norm();
        if a_norm == 0.0 {
            return Err(DecompError::singular("condition number: zero norm"));
        }

        // Hager's algorithm to estimate ‖A⁻¹‖₁
        let ainv_norm = self.hager_estimate()?;

        Ok(a_norm * ainv_norm)
    }

    /// Reconstruct ‖A‖₁ from the factors.
    fn reconstruct_a_one_norm(&self) -> f64 {
        let n = self.n;
        // Build P^{-1} so that A = P^{-1} L U
        let mut inv_perm = vec![0usize; n];
        for i in 0..n {
            inv_perm[self.pivot[i]] = i;
        }

        let mut max_col_sum = 0.0f64;
        for j in 0..n {
            // Compute column j of L*U
            let mut lu_col = vec![0.0; n];
            for i in 0..n {
                let mut s = 0.0;
                // Sum L[i,k] * U[k,j] for k < i and k <= j
                for k in 0..i.min(j + 1) {
                    s += self.lu.get(i, k) * self.lu.get(k, j);
                }
                // Diagonal contribution: L[i,i] = 1, U[i,j]
                if i <= j {
                    s += self.lu.get(i, j); // 1.0 * U[i,j]
                }
                lu_col[i] = s;
            }
            // Apply P^{-1}: A[original_row, j] = lu_col[permuted_row]
            let mut col_sum = 0.0;
            for i in 0..n {
                col_sum += lu_col[inv_perm[i]].abs();
            }
            max_col_sum = max_col_sum.max(col_sum);
        }
        max_col_sum
    }

    /// Hager's (1984) algorithm to estimate ‖A⁻¹‖₁ without forming A⁻¹.
    fn hager_estimate(&self) -> DecompResult<f64> {
        let n = self.n;
        if n == 1 {
            let d = self.lu.get(0, 0);
            if d.abs() < 1e-300 {
                return Err(DecompError::ZeroPivot { index: 0, value: d });
            }
            return Ok(1.0 / d.abs());
        }

        // x = [1/n, ..., 1/n]
        let mut x: Vec<f64> = vec![1.0 / n as f64; n];
        let mut gamma = 0.0f64;
        let max_iters = 5;

        for _ in 0..max_iters {
            // w = A⁻¹ x  (solve A w = x)
            let w = self.solve(&x)?;

            // gamma_new = ‖w‖₁
            let gamma_new: f64 = w.iter().map(|v| v.abs()).sum();

            // ξ = sign(w)
            let xi: Vec<f64> = w.iter().map(|&v| if v >= 0.0 { 1.0 } else { -1.0 }).collect();

            // z = A⁻ᵀ ξ  (solve Aᵀ z = ξ)
            let z = self.solve_transpose(&xi)?;

            // Check convergence
            let z_inf: f64 = z.iter().map(|v| v.abs()).fold(0.0f64, f64::max);

            if z_inf <= gamma_new || (gamma_new - gamma).abs() < 1e-14 * gamma_new {
                return Ok(gamma_new);
            }
            gamma = gamma_new;

            // x = e_j where j = argmax |z_j|
            let j_max = z
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
                .map(|(i, _)| i)
                .unwrap();

            x = vec![0.0; n];
            x[j_max] = 1.0;
        }

        // Final evaluation
        let w = self.solve(&x)?;
        let gamma_final: f64 = w.iter().map(|v| v.abs()).sum();
        Ok(gamma_final.max(gamma))
    }

    /// Solve Aᵀ x = b using the LU factors.
    ///
    /// Since PA = LU, we have Aᵀ = Uᵀ Lᵀ Pᵀ⁻¹ = Uᵀ Lᵀ P.
    /// So Aᵀ x = b  ⟹  Uᵀ Lᵀ P x = b
    /// Let w = P x, then Uᵀ (Lᵀ w) = b.
    /// Step 1: solve Uᵀ v = b  (forward sub with U transposed)
    /// Step 2: solve Lᵀ w = v  (back sub with L transposed, unit diag)
    /// Step 3: x = P⁻¹ w  (reverse permutation)
    fn solve_transpose(&self, b: &[f64]) -> DecompResult<Vec<f64>> {
        let n = self.n;

        // Step 1: forward sub with Uᵀ (lower triangular with diag = U diag)
        let mut v = vec![0.0; n];
        for i in 0..n {
            let mut s = b[i];
            for j in 0..i {
                s -= self.lu.get(j, i) * v[j]; // U^T[i,j] = U[j,i]
            }
            let diag = self.lu.get(i, i);
            if diag.abs() < 1e-300 {
                return Err(DecompError::ZeroPivot {
                    index: i,
                    value: diag,
                });
            }
            v[i] = s / diag;
        }

        // Step 2: back sub with Lᵀ (upper triangular with unit diag)
        let mut w = vec![0.0; n];
        for i in (0..n).rev() {
            let mut s = v[i];
            for j in (i + 1)..n {
                s -= self.lu.get(j, i) * w[j]; // L^T[i,j] = L[j,i]
            }
            w[i] = s; // unit diagonal
        }

        // Step 3: reverse permutation x[pivot[i]] = w[i]
        let mut x = vec![0.0; n];
        for i in 0..n {
            x[self.pivot[i]] = w[i];
        }

        Ok(x)
    }

    // ── Factor extraction ──────────────────────────────────────────────

    /// Extract the unit lower-triangular factor L.
    pub fn get_l(&self) -> DenseMatrix {
        let n = self.n;
        let mut l = DenseMatrix::zeros(n, n);
        for i in 0..n {
            l.set(i, i, 1.0);
            for j in 0..i {
                l.set(i, j, self.lu.get(i, j));
            }
        }
        l
    }

    /// Extract the upper-triangular factor U.
    pub fn get_u(&self) -> DenseMatrix {
        let n = self.n;
        let mut u = DenseMatrix::zeros(n, n);
        for i in 0..n {
            for j in i..n {
                u.set(i, j, self.lu.get(i, j));
            }
        }
        u
    }

    /// Return the permutation vector (clone).
    pub fn get_p(&self) -> Vec<usize> {
        self.pivot.clone()
    }

    /// Build the full n×n permutation matrix P such that P·A = L·U.
    pub fn get_p_matrix(&self) -> DenseMatrix {
        let n = self.n;
        let mut p = DenseMatrix::zeros(n, n);
        for i in 0..n {
            p.set(i, self.pivot[i], 1.0);
        }
        p
    }

    /// Matrix dimension.
    pub fn dim(&self) -> usize {
        self.n
    }

    /// Permutation sign (+1 or −1).
    pub fn permutation_sign(&self) -> i32 {
        self.sign
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Sparse LU decomposition (threshold pivoting)
// ═══════════════════════════════════════════════════════════════════════════

/// Sparse LU decomposition using CSR storage and threshold pivoting.
///
/// Stores L (unit lower triangular) and U (upper triangular) in separate
/// CSR-like compressed storage. The threshold parameter controls the
/// trade-off between sparsity and numerical stability.
#[derive(Clone, Debug)]
pub struct SparseLuDecomposition {
    /// Dimension of the square matrix.
    n: usize,
    /// Row pointers for L (unit lower triangular, diagonal implicit).
    l_row_ptr: Vec<usize>,
    /// Column indices for L values.
    l_col_idx: Vec<usize>,
    /// Values in L (excluding the unit diagonal).
    l_values: Vec<f64>,
    /// Row pointers for U (upper triangular, diagonal stored).
    u_row_ptr: Vec<usize>,
    /// Column indices for U values.
    u_col_idx: Vec<usize>,
    /// Values in U.
    u_values: Vec<f64>,
    /// Diagonal of U (quick access).
    u_diag: Vec<f64>,
    /// Row permutation.
    pivot: Vec<usize>,
    /// Permutation sign.
    sign: i32,
}

impl SparseLuDecomposition {
    /// Sparse LU factorization with threshold pivoting.
    ///
    /// `threshold` ∈ (0, 1] controls how aggressively we pivot.  A value
    /// of 1.0 gives full partial pivoting; smaller values allow weaker
    /// pivots if they preserve sparsity.
    ///
    /// For a candidate pivot at column k, we accept row i if
    /// |a[i,k]| >= threshold * max_{j>=k} |a[j,k]|.
    /// Among all acceptable candidates, we pick the one with the fewest
    /// non-zeros in its row (Markowitz-style heuristic).
    pub fn factorize_sparse_threshold(a: &CsrMatrix, threshold: f64) -> DecompResult<Self> {
        if a.rows == 0 || a.cols == 0 {
            return Err(DecompError::empty("sparse LU: empty matrix"));
        }
        if !a.is_square() {
            return Err(DecompError::NotSquare {
                rows: a.rows,
                cols: a.cols,
            });
        }
        let threshold = threshold.clamp(1e-3, 1.0);
        let n = a.rows;

        // Work with dense intermediate for simplicity on small-to-mid matrices.
        // A full sparse implementation with fill-in tracking would be needed
        // for very large systems.
        let mut work = a.to_dense();
        let mut pivot: Vec<usize> = (0..n).collect();
        let mut sign: i32 = 1;

        // We will accumulate L and U entries during elimination.
        let mut l_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n]; // l_entries[i] = list of (j, L[i,j]) for j < i
        let mut u_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n]; // u_entries[i] = list of (j, U[i,j]) for j >= i
        let mut u_diag = vec![0.0; n];

        for k in 0..n {
            // Find max |a[i,k]| for i >= k
            let mut max_val = 0.0f64;
            for i in k..n {
                let v = work.get(i, k).abs();
                if v > max_val {
                    max_val = v;
                }
            }

            if max_val < 1e-300 {
                return Err(DecompError::ZeroPivot {
                    index: k,
                    value: work.get(k, k),
                });
            }

            let accept_threshold = threshold * max_val;

            // Among rows i >= k with |a[i,k]| >= accept_threshold, pick
            // the one with the fewest non-zeros in columns k..n
            // (Markowitz heuristic for sparsity).
            let mut best_row = k;
            let mut best_nnz = usize::MAX;
            for i in k..n {
                if work.get(i, k).abs() >= accept_threshold {
                    let nnz: usize = (k..n)
                        .filter(|&j| work.get(i, j).abs() > 1e-300)
                        .count();
                    if nnz < best_nnz {
                        best_nnz = nnz;
                        best_row = i;
                    }
                }
            }

            // Swap rows
            if best_row != k {
                work.swap_rows(k, best_row);
                pivot.swap(k, best_row);
                sign = -sign;
                // Also swap accumulated L entries
                l_entries.swap(k, best_row);
            }

            // Record U entries for row k
            u_diag[k] = work.get(k, k);
            for j in k..n {
                let v = work.get(k, j);
                if v.abs() > 1e-300 {
                    u_entries[k].push((j, v));
                }
            }

            // Elimination
            let diag = work.get(k, k);
            for i in (k + 1)..n {
                let aik = work.get(i, k);
                if aik.abs() < 1e-300 {
                    continue;
                }
                let mult = aik / diag;
                work.set(i, k, 0.0); // zeroed out

                // Store L entry
                l_entries[i].push((k, mult));

                // Update trailing matrix with drop tolerance
                for j in (k + 1)..n {
                    let val = work.get(i, j) - mult * work.get(k, j);
                    work.set(i, j, val);
                }
            }
        }

        // Pack into CSR-like format
        let mut l_row_ptr = vec![0usize; n + 1];
        let mut l_col_idx = Vec::new();
        let mut l_values = Vec::new();
        for i in 0..n {
            // sort l_entries[i] by column
            l_entries[i].sort_by_key(|&(c, _)| c);
            for &(c, v) in &l_entries[i] {
                l_col_idx.push(c);
                l_values.push(v);
            }
            l_row_ptr[i + 1] = l_col_idx.len();
        }

        let mut u_row_ptr = vec![0usize; n + 1];
        let mut u_col_idx_flat = Vec::new();
        let mut u_values_flat = Vec::new();
        for i in 0..n {
            u_entries[i].sort_by_key(|&(c, _)| c);
            for &(c, v) in &u_entries[i] {
                u_col_idx_flat.push(c);
                u_values_flat.push(v);
            }
            u_row_ptr[i + 1] = u_col_idx_flat.len();
        }

        Ok(Self {
            n,
            l_row_ptr,
            l_col_idx,
            l_values,
            u_row_ptr,
            u_col_idx: u_col_idx_flat,
            u_values: u_values_flat,
            u_diag,
            pivot,
            sign,
        })
    }

    /// Solve A x = b using the sparse LU factors.
    pub fn solve(&self, b: &[f64]) -> DecompResult<Vec<f64>> {
        let n = self.n;
        if b.len() != n {
            return Err(DecompError::VectorLengthMismatch {
                expected: n,
                actual: b.len(),
            });
        }

        // Apply permutation
        let mut pb = vec![0.0; n];
        for i in 0..n {
            pb[i] = b[self.pivot[i]];
        }

        // Forward substitution: L y = pb (unit lower triangular)
        let mut y = pb;
        for i in 0..n {
            let start = self.l_row_ptr[i];
            let end = self.l_row_ptr[i + 1];
            for idx in start..end {
                let j = self.l_col_idx[idx];
                let l_ij = self.l_values[idx];
                y[i] -= l_ij * y[j];
            }
        }

        // Back substitution: U x = y
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            let mut s = y[i];
            let start = self.u_row_ptr[i];
            let end = self.u_row_ptr[i + 1];
            for idx in start..end {
                let j = self.u_col_idx[idx];
                if j > i {
                    s -= self.u_values[idx] * x[j];
                }
            }
            let diag = self.u_diag[i];
            if diag.abs() < 1e-300 {
                return Err(DecompError::ZeroPivot {
                    index: i,
                    value: diag,
                });
            }
            x[i] = s / diag;
        }

        Ok(x)
    }

    /// Determinant from the sparse factorization.
    pub fn determinant(&self) -> f64 {
        let mut det = self.sign as f64;
        for i in 0..self.n {
            det *= self.u_diag[i];
        }
        det
    }

    /// Matrix dimension.
    pub fn dim(&self) -> usize {
        self.n
    }

    /// Number of non-zeros in L (excluding unit diagonal).
    pub fn l_nnz(&self) -> usize {
        self.l_values.len()
    }

    /// Number of non-zeros in U.
    pub fn u_nnz(&self) -> usize {
        self.u_values.len()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn mat_approx_eq(a: &DenseMatrix, b: &DenseMatrix, tol: f64) -> bool {
        if a.rows != b.rows || a.cols != b.cols {
            return false;
        }
        a.data
            .iter()
            .zip(b.data.iter())
            .all(|(&x, &y)| (x - y).abs() < tol)
    }

    // ── 1. Identity matrix ─────────────────────────────────────────────

    #[test]
    fn test_identity_lu() {
        let eye = DenseMatrix::eye(4);
        let lu = LuDecomposition::factorize(&eye).unwrap();
        let l = lu.get_l();
        let u = lu.get_u();
        assert!(mat_approx_eq(&l, &DenseMatrix::eye(4), 1e-14));
        assert!(mat_approx_eq(&u, &DenseMatrix::eye(4), 1e-14));
        assert!(approx_eq(lu.determinant(), 1.0, 1e-14));
    }

    // ── 2. Known 3×3 PA = LU ──────────────────────────────────────────

    #[test]
    fn test_known_3x3_pa_eq_lu() {
        let a = DenseMatrix::from_row_major(
            3,
            3,
            vec![
                2.0, 1.0, 1.0,
                4.0, 3.0, 3.0,
                8.0, 7.0, 9.0,
            ],
        );
        let lu = LuDecomposition::factorize(&a).unwrap();
        let l = lu.get_l();
        let u = lu.get_u();
        let p = lu.get_p_matrix();

        // Verify P * A == L * U
        let pa = p.mul_mat(&a).unwrap();
        let lu_prod = l.mul_mat(&u).unwrap();
        assert!(mat_approx_eq(&pa, &lu_prod, 1e-12));
    }

    // ── 3. Solve Ax = b ───────────────────────────────────────────────

    #[test]
    fn test_solve_ax_eq_b() {
        let a = DenseMatrix::from_row_major(
            3,
            3,
            vec![
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0,
                7.0, 8.0, 10.0,
            ],
        );
        let b = vec![14.0, 32.0, 53.0];
        let lu = LuDecomposition::factorize(&a).unwrap();
        let x = lu.solve(&b).unwrap();

        // Verify A*x ≈ b
        let ax = a.mul_vec(&x).unwrap();
        for i in 0..3 {
            assert!(
                approx_eq(ax[i], b[i], 1e-10),
                "ax[{}] = {}, b[{}] = {}",
                i,
                ax[i],
                i,
                b[i]
            );
        }
    }

    // ── 4. Determinant ────────────────────────────────────────────────

    #[test]
    fn test_determinant() {
        // det([[1,2],[3,4]]) = 1*4 - 2*3 = -2
        let a = DenseMatrix::from_row_major(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let lu = LuDecomposition::factorize(&a).unwrap();
        assert!(approx_eq(lu.determinant(), -2.0, 1e-12));
    }

    // ── 5. Inverse A * A⁻¹ = I ────────────────────────────────────────

    #[test]
    fn test_inverse() {
        let a = DenseMatrix::from_row_major(
            3,
            3,
            vec![
                2.0, 1.0, 1.0,
                4.0, 3.0, 3.0,
                8.0, 7.0, 9.0,
            ],
        );
        let lu = LuDecomposition::factorize(&a).unwrap();
        let inv = lu.inverse().unwrap();
        let product = a.mul_mat(&inv).unwrap();
        let eye = DenseMatrix::eye(3);
        assert!(mat_approx_eq(&product, &eye, 1e-10));
    }

    // ── 6. Condition number ───────────────────────────────────────────

    #[test]
    fn test_condition_number() {
        // Well-conditioned matrix: identity has κ = 1
        let eye = DenseMatrix::eye(4);
        let lu = LuDecomposition::factorize(&eye).unwrap();
        let kappa = lu.condition_number_estimate().unwrap();
        assert!(
            approx_eq(kappa, 1.0, 0.1),
            "kappa(I) = {}, expected ~1.0",
            kappa
        );

        // Ill-conditioned matrix
        let a = DenseMatrix::from_row_major(
            2,
            2,
            vec![1.0, 0.0, 0.0, 1e-10],
        );
        let lu2 = LuDecomposition::factorize(&a).unwrap();
        let kappa2 = lu2.condition_number_estimate().unwrap();
        assert!(kappa2 > 1e9, "expected large condition number, got {}", kappa2);
    }

    // ── 7. Multiple right-hand sides ──────────────────────────────────

    #[test]
    fn test_solve_multiple_rhs() {
        let a = DenseMatrix::from_row_major(
            2,
            2,
            vec![2.0, 1.0, 1.0, 3.0],
        );
        // B = [[5, 3], [7, 4]]
        let b = DenseMatrix::from_row_major(2, 2, vec![5.0, 3.0, 7.0, 4.0]);
        let lu = LuDecomposition::factorize(&a).unwrap();
        let x = lu.solve_multiple(&b).unwrap();

        // Verify A * X ≈ B
        let ax = a.mul_mat(&x).unwrap();
        assert!(mat_approx_eq(&ax, &b, 1e-12));
    }

    // ── 8. Singular matrix detection ──────────────────────────────────

    #[test]
    fn test_singular_detection() {
        let a = DenseMatrix::from_row_major(
            3,
            3,
            vec![
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0,
                7.0, 8.0, 9.0, // row3 = row1 + row2 (linearly dependent, but pivoting may still work)
            ],
        );
        // This matrix is singular; factorize should detect it
        let result = LuDecomposition::factorize(&a);
        assert!(result.is_err(), "should detect singular matrix");
    }

    // ── 9. Sparse LU ─────────────────────────────────────────────────

    #[test]
    fn test_sparse_lu() {
        let triplets = vec![
            (0, 0, 4.0),
            (0, 1, 3.0),
            (1, 0, 6.0),
            (1, 1, 3.0),
        ];
        let a = CsrMatrix::from_triplets(2, 2, &triplets);
        let slu = SparseLuDecomposition::factorize_sparse_threshold(&a, 1.0).unwrap();

        let b = vec![10.0, 12.0];
        let x = slu.solve(&b).unwrap();

        // Verify: A*x ≈ b
        let ax = a.mul_vec(&x).unwrap();
        for i in 0..2 {
            assert!(
                approx_eq(ax[i], b[i], 1e-10),
                "sparse: ax[{}] = {}, b[{}] = {}",
                i,
                ax[i],
                i,
                b[i]
            );
        }

        // det([[4,3],[6,3]]) = 12 - 18 = -6
        assert!(approx_eq(slu.determinant(), -6.0, 1e-10));
    }

    // ── 10. Log-determinant ───────────────────────────────────────────

    #[test]
    fn test_log_abs_determinant() {
        let a = DenseMatrix::from_row_major(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let lu = LuDecomposition::factorize(&a).unwrap();
        let (log_abs, det_sign) = lu.log_abs_determinant();
        let det = lu.determinant();

        assert!(approx_eq(log_abs, det.abs().ln(), 1e-12));
        assert_eq!(det_sign, if det >= 0.0 { 1 } else { -1 });
    }

    // ── 11. Refactorize ──────────────────────────────────────────────

    #[test]
    fn test_refactorize() {
        let a1 = DenseMatrix::from_row_major(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let mut lu = LuDecomposition::factorize(&a1).unwrap();
        assert!(approx_eq(lu.determinant(), -2.0, 1e-12));

        let a2 = DenseMatrix::from_row_major(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        lu.refactorize(&a2).unwrap();
        // det([[5,6],[7,8]]) = 40 - 42 = -2
        assert!(approx_eq(lu.determinant(), -2.0, 1e-12));

        // Verify solve still works with new matrix
        let b = vec![11.0, 15.0];
        let x = lu.solve(&b).unwrap();
        let ax = a2.mul_vec(&x).unwrap();
        for i in 0..2 {
            assert!(approx_eq(ax[i], b[i], 1e-10));
        }
    }

    // ── 12. 1×1 matrix ───────────────────────────────────────────────

    #[test]
    fn test_1x1() {
        let a = DenseMatrix::from_row_major(1, 1, vec![7.0]);
        let lu = LuDecomposition::factorize(&a).unwrap();
        assert!(approx_eq(lu.determinant(), 7.0, 1e-14));

        let x = lu.solve(&[21.0]).unwrap();
        assert!(approx_eq(x[0], 3.0, 1e-14));

        let inv = lu.inverse().unwrap();
        assert!(approx_eq(inv.get(0, 0), 1.0 / 7.0, 1e-14));
    }

    // ── 13. Forward / back substitution helpers ───────────────────────

    #[test]
    fn test_forward_back_solve() {
        // L = [[1,0,0],[2,1,0],[3,4,1]]
        let l = DenseMatrix::from_row_major(
            3,
            3,
            vec![
                1.0, 0.0, 0.0,
                2.0, 1.0, 0.0,
                3.0, 4.0, 1.0,
            ],
        );
        let b = vec![1.0, 4.0, 15.0];
        let y = forward_solve(&l, &b).unwrap();
        // y[0] = 1, y[1] = 4 - 2*1 = 2, y[2] = 15 - 3*1 - 4*2 = 4
        assert!(approx_eq(y[0], 1.0, 1e-14));
        assert!(approx_eq(y[1], 2.0, 1e-14));
        assert!(approx_eq(y[2], 4.0, 1e-14));

        // U = [[2,1,1],[0,3,2],[0,0,4]]
        let u = DenseMatrix::from_row_major(
            3,
            3,
            vec![
                2.0, 1.0, 1.0,
                0.0, 3.0, 2.0,
                0.0, 0.0, 4.0,
            ],
        );
        let x = back_solve(&u, &y).unwrap();
        // x[2] = 4/4 = 1, x[1] = (2 - 2*1)/3 = 0, x[0] = (1 - 0 - 1)/2 = 0
        assert!(approx_eq(x[0], 0.0, 1e-14));
        assert!(approx_eq(x[1], 0.0, 1e-14));
        assert!(approx_eq(x[2], 1.0, 1e-14));
    }

    // ── 14. L/U extraction roundtrip ──────────────────────────────────

    #[test]
    fn test_lu_extraction_roundtrip() {
        let a = DenseMatrix::from_row_major(
            4,
            4,
            vec![
                2.0, 1.0, 4.0, 1.0,
                3.0, 4.0, -1.0, 2.0,
                1.0, 2.0, 1.0, 3.0,
                5.0, 1.0, 3.0, -2.0,
            ],
        );
        let lu = LuDecomposition::factorize(&a).unwrap();
        let l = lu.get_l();
        let u = lu.get_u();
        let p = lu.get_p_matrix();

        // L must be unit lower triangular
        for i in 0..4 {
            assert!(approx_eq(l.get(i, i), 1.0, 1e-14));
            for j in (i + 1)..4 {
                assert!(approx_eq(l.get(i, j), 0.0, 1e-14));
            }
        }

        // U must be upper triangular
        for i in 0..4 {
            for j in 0..i {
                assert!(approx_eq(u.get(i, j), 0.0, 1e-14));
            }
        }

        // P*A = L*U
        let pa = p.mul_mat(&a).unwrap();
        let lu_prod = l.mul_mat(&u).unwrap();
        assert!(mat_approx_eq(&pa, &lu_prod, 1e-10));
    }

    // ── 15. CSR factorize ─────────────────────────────────────────────

    #[test]
    fn test_factorize_from_csr() {
        let a_dense = DenseMatrix::from_row_major(
            3,
            3,
            vec![
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0,
                7.0, 8.0, 10.0,
            ],
        );
        let a_csr = a_dense.to_csr();
        let lu = LuDecomposition::factorize_from_csr(&a_csr).unwrap();

        let b = vec![14.0, 32.0, 53.0];
        let x = lu.solve(&b).unwrap();
        let ax = a_dense.mul_vec(&x).unwrap();
        for i in 0..3 {
            assert!(approx_eq(ax[i], b[i], 1e-10));
        }
    }

    // ── 16. Non-square rejection ──────────────────────────────────────

    #[test]
    fn test_non_square_rejected() {
        let a = DenseMatrix::zeros(3, 4);
        let result = LuDecomposition::factorize(&a);
        assert!(result.is_err());
    }

    // ── 17. Empty matrix rejection ────────────────────────────────────

    #[test]
    fn test_empty_matrix_rejected() {
        let a = DenseMatrix::zeros(0, 0);
        let result = LuDecomposition::factorize(&a);
        assert!(result.is_err());
    }

    // ── 18. Larger system accuracy ────────────────────────────────────

    #[test]
    fn test_larger_system() {
        let n = 10;
        // Build a diagonally dominant matrix
        let mut a = DenseMatrix::zeros(n, n);
        for i in 0..n {
            a.set(i, i, (n as f64) * 2.0);
            if i > 0 {
                a.set(i, i - 1, 1.0);
            }
            if i + 1 < n {
                a.set(i, i + 1, 1.0);
            }
        }

        // known solution x = [1, 2, ..., n]
        let x_true: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let b = a.mul_vec(&x_true).unwrap();

        let lu = LuDecomposition::factorize(&a).unwrap();
        let x = lu.solve(&b).unwrap();

        for i in 0..n {
            assert!(
                approx_eq(x[i], x_true[i], 1e-8),
                "x[{}] = {}, expected {}",
                i,
                x[i],
                x_true[i]
            );
        }
    }
}
