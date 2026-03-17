//! Cholesky decomposition for symmetric positive definite matrices.
//!
//! Provides three flavours of Cholesky factorisation:
//!
//! * **[`CholeskyDecomposition`]** – classic A = LLᵀ for dense SPD matrices.
//! * **[`ModifiedCholeskyDecomposition`]** – Gill-Murray-Wright modified Cholesky
//!   that adds a diagonal perturbation E so that A + E = LLᵀ is always valid,
//!   even when A is indefinite.
//! * **[`IncompleteCholesky`]** – IC(0) for sparse SPD matrices stored in CSR
//!   format, useful as a preconditioner for iterative solvers.
//!
//! Additionally, a simple **approximate minimum-degree** ordering and a
//! **sparse Cholesky with fill-reducing ordering** helper are provided.

use crate::{CsrMatrix, DecompError, DecompResult, DenseMatrix};

// Re-export dot for potential external use by callers.
#[allow(unused_imports)]
use crate::dot;

// ═══════════════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════════════

/// Default symmetry tolerance.
const SYMMETRY_TOL: f64 = 1e-10;

/// Minimum acceptable diagonal pivot before declaring non-positive-definite.
const MIN_PIVOT: f64 = 1e-14;

/// Machine epsilon for perturbation sizing.
const MACHINE_EPS: f64 = f64::EPSILON;

// ═══════════════════════════════════════════════════════════════════════════
// 1. Standard Cholesky (A = LLᵀ)
// ═══════════════════════════════════════════════════════════════════════════

/// Standard Cholesky decomposition: A = LLᵀ.
///
/// Works on dense symmetric positive-definite matrices.  The lower-triangular
/// factor L is stored internally and exposed via [`get_l`].
#[derive(Clone, Debug)]
pub struct CholeskyDecomposition {
    /// Lower-triangular factor such that A = L Lᵀ.
    l: DenseMatrix,
    /// Matrix dimension.
    n: usize,
}

impl CholeskyDecomposition {
    // ── helpers ──────────────────────────────────────────────────────────

    /// Check that a dense matrix is square and symmetric.
    fn check_symmetric(a: &DenseMatrix) -> DecompResult<()> {
        if a.is_empty() {
            return Err(DecompError::EmptyMatrix {
                context: "Cholesky factorization requires a non-empty matrix".into(),
            });
        }
        if !a.is_square() {
            return Err(DecompError::NotSquare {
                rows: a.rows,
                cols: a.cols,
            });
        }
        let n = a.rows;
        let mut max_diff: f64 = 0.0;
        let mut worst_r = 0;
        let mut worst_c = 0;
        for i in 0..n {
            for j in (i + 1)..n {
                let diff = (a.get(i, j) - a.get(j, i)).abs();
                if diff > max_diff {
                    max_diff = diff;
                    worst_r = i;
                    worst_c = j;
                }
            }
        }
        if max_diff > SYMMETRY_TOL {
            return Err(DecompError::NotSymmetric {
                max_diff,
                row: worst_r,
                col: worst_c,
            });
        }
        Ok(())
    }

    // ── factorisation ───────────────────────────────────────────────────

    /// Compute the Cholesky factorisation A = LLᵀ.
    ///
    /// Returns `Err(NotPositiveDefinite)` if a non-positive pivot is
    /// encountered, and `Err(NotSymmetric)` if the input is not symmetric
    /// within [`SYMMETRY_TOL`].
    pub fn factorize(a: &DenseMatrix) -> DecompResult<Self> {
        Self::check_symmetric(a)?;
        let n = a.rows;

        let mut l = DenseMatrix::zeros(n, n);

        for j in 0..n {
            // Diagonal element
            let mut sum_sq = 0.0;
            for k in 0..j {
                let ljk = l.get(j, k);
                sum_sq += ljk * ljk;
            }
            let diag = a.get(j, j) - sum_sq;
            if diag <= MIN_PIVOT {
                return Err(DecompError::NotPositiveDefinite {
                    context: format!(
                        "pivot at column {} is {:.2e}, not positive",
                        j, diag
                    ),
                });
            }
            let ljj = diag.sqrt();
            l.set(j, j, ljj);

            // Off-diagonal elements in column j, rows i > j
            let inv_ljj = 1.0 / ljj;
            for i in (j + 1)..n {
                let mut dot_sum = 0.0;
                for k in 0..j {
                    dot_sum += l.get(i, k) * l.get(j, k);
                }
                let lij = (a.get(i, j) - dot_sum) * inv_ljj;
                l.set(i, j, lij);
            }
        }

        Ok(Self { l, n })
    }

    // ── solve ───────────────────────────────────────────────────────────

    /// Solve A x = b  via  L y = b  (forward),  Lᵀ x = y  (backward).
    pub fn solve(&self, b: &[f64]) -> DecompResult<Vec<f64>> {
        if b.len() != self.n {
            return Err(DecompError::VectorLengthMismatch {
                expected: self.n,
                actual: b.len(),
            });
        }
        let y = self.forward_substitution(b);
        let x = self.backward_substitution(&y);
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
        let ncols = b.cols;
        let mut result = DenseMatrix::zeros(self.n, ncols);
        for j in 0..ncols {
            let col = b.col(j);
            let x = self.solve(&col)?;
            for i in 0..self.n {
                result.set(i, j, x[i]);
            }
        }
        Ok(result)
    }

    // ── determinant ─────────────────────────────────────────────────────

    /// Compute det(A) = (∏ L[i,i])².
    pub fn determinant(&self) -> f64 {
        let mut prod = 1.0;
        for i in 0..self.n {
            prod *= self.l.get(i, i);
        }
        prod * prod
    }

    /// Compute ln|det(A)| = 2 ∑ ln(L[i,i]).
    ///
    /// More numerically stable than `determinant().ln()` for large matrices.
    pub fn log_determinant(&self) -> f64 {
        let mut s = 0.0;
        for i in 0..self.n {
            s += self.l.get(i, i).ln();
        }
        2.0 * s
    }

    // ── inverse ─────────────────────────────────────────────────────────

    /// Compute A⁻¹ by solving A X = I.
    pub fn inverse(&self) -> DecompResult<DenseMatrix> {
        let eye = DenseMatrix::eye(self.n);
        self.solve_multiple(&eye)
    }

    // ── accessors ───────────────────────────────────────────────────────

    /// Return a clone of the lower-triangular factor L.
    pub fn get_l(&self) -> DenseMatrix {
        self.l.clone()
    }

    /// Matrix dimension.
    pub fn dim(&self) -> usize {
        self.n
    }

    // ── internal substitution routines ──────────────────────────────────

    /// Forward substitution: solve L y = b.
    fn forward_substitution(&self, b: &[f64]) -> Vec<f64> {
        let n = self.n;
        let mut y = vec![0.0; n];
        for i in 0..n {
            let mut s = b[i];
            for k in 0..i {
                s -= self.l.get(i, k) * y[k];
            }
            y[i] = s / self.l.get(i, i);
        }
        y
    }

    /// Backward substitution: solve Lᵀ x = y.
    fn backward_substitution(&self, y: &[f64]) -> Vec<f64> {
        let n = self.n;
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            let mut s = y[i];
            for k in (i + 1)..n {
                s -= self.l.get(k, i) * x[k];
            }
            x[i] = s / self.l.get(i, i);
        }
        x
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. Modified Cholesky (Gill-Murray-Wright)
// ═══════════════════════════════════════════════════════════════════════════

/// Modified Cholesky: A + diag(E) = LLᵀ.
///
/// Uses the Gill-Murray-Wright strategy to perturb the diagonal so that even
/// an indefinite (but symmetric) matrix can be factorised.
#[derive(Clone, Debug)]
pub struct ModifiedCholeskyDecomposition {
    l: DenseMatrix,
    n: usize,
    /// Perturbation added to each diagonal element.
    perturbation_diag: Vec<f64>,
}

impl ModifiedCholeskyDecomposition {
    /// Factorise with modified Cholesky.
    ///
    /// Returns L and the perturbation vector E such that A + diag(E) = LLᵀ.
    /// If A is already positive-definite the perturbation is zero.
    pub fn factorize(a: &DenseMatrix) -> DecompResult<Self> {
        if a.is_empty() {
            return Err(DecompError::EmptyMatrix {
                context: "Modified Cholesky requires a non-empty matrix".into(),
            });
        }
        if !a.is_square() {
            return Err(DecompError::NotSquare {
                rows: a.rows,
                cols: a.cols,
            });
        }

        let n = a.rows;

        // ---------- Gill-Murray-Wright parameters ----------
        // gamma  = max |a_ii|
        // xi     = max |a_ij|, i != j
        // beta^2 = max(gamma, xi / sqrt(n^2-1), eps)
        // delta  = eps * max(gamma + xi, 1)
        let mut gamma: f64 = 0.0;
        for i in 0..n {
            gamma = gamma.max(a.get(i, i).abs());
        }

        let mut xi: f64 = 0.0;
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    xi = xi.max(a.get(i, j).abs());
                }
            }
        }

        let n2m1 = if n > 1 {
            ((n * n - 1) as f64).sqrt()
        } else {
            1.0
        };
        let beta_sq = gamma.max(xi / n2m1).max(MACHINE_EPS);
        let delta = MACHINE_EPS * (gamma + xi).max(1.0);

        // Work on a copy of the matrix (will be overwritten column by column).
        let mut c = DenseMatrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                c.set(i, j, a.get(i, j));
            }
        }

        let mut l = DenseMatrix::zeros(n, n);
        let mut e = vec![0.0_f64; n];

        for j in 0..n {
            // --- compute column j of the Schur complement (already in c) ---
            // The sub-diagonal part of column j of c is already updated.

            // Determine the maximum magnitude of off-diagonal Schur entries
            // in column j to compute the perturbation.
            let mut theta_j: f64 = 0.0;
            for i in (j + 1)..n {
                theta_j = theta_j.max(c.get(i, j).abs());
            }

            // Perturbation: d_j = max(|c_jj|, (theta_j / beta)^2, delta)
            let d_j = (c.get(j, j).abs())
                .max(theta_j * theta_j / beta_sq)
                .max(delta);
            e[j] = d_j - c.get(j, j);

            // Set the diagonal of L
            let ljj = d_j.sqrt();
            l.set(j, j, ljj);

            let inv_ljj = 1.0 / ljj;

            // Off-diagonal entries and Schur-complement update
            for i in (j + 1)..n {
                let lij = c.get(i, j) * inv_ljj;
                l.set(i, j, lij);
            }

            // Update the remaining sub-matrix (Schur complement)
            for i in (j + 1)..n {
                let lij = l.get(i, j);
                for k in (j + 1)..=i {
                    let lik = l.get(k, j);
                    let old = c.get(i, k);
                    c.set(i, k, old - lij * lik);
                    if k != i {
                        c.set(k, i, c.get(i, k));
                    }
                }
            }
        }

        Ok(Self {
            l,
            n,
            perturbation_diag: e,
        })
    }

    /// The per-diagonal perturbation that was added.
    pub fn perturbation(&self) -> &[f64] {
        &self.perturbation_diag
    }

    /// Solve (A + E) x = b.
    pub fn solve(&self, b: &[f64]) -> DecompResult<Vec<f64>> {
        if b.len() != self.n {
            return Err(DecompError::VectorLengthMismatch {
                expected: self.n,
                actual: b.len(),
            });
        }
        let y = self.forward_substitution(b);
        let x = self.backward_substitution(&y);
        Ok(x)
    }

    /// Return the lower-triangular factor.
    pub fn get_l(&self) -> DenseMatrix {
        self.l.clone()
    }

    /// Matrix dimension.
    pub fn dim(&self) -> usize {
        self.n
    }

    // ── internal substitution (same as standard Cholesky) ───────────────

    fn forward_substitution(&self, b: &[f64]) -> Vec<f64> {
        let n = self.n;
        let mut y = vec![0.0; n];
        for i in 0..n {
            let mut s = b[i];
            for k in 0..i {
                s -= self.l.get(i, k) * y[k];
            }
            y[i] = s / self.l.get(i, i);
        }
        y
    }

    fn backward_substitution(&self, y: &[f64]) -> Vec<f64> {
        let n = self.n;
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            let mut s = y[i];
            for k in (i + 1)..n {
                s -= self.l.get(k, i) * x[k];
            }
            x[i] = s / self.l.get(i, i);
        }
        x
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. Incomplete Cholesky – IC(0)
// ═══════════════════════════════════════════════════════════════════════════

/// Incomplete Cholesky factorisation with zero fill-in – IC(0).
///
/// The sparsity pattern of L is restricted to the lower triangle of the
/// sparsity pattern of the input matrix.  This is the standard preconditioner
/// for conjugate gradient on SPD systems.
#[derive(Clone, Debug)]
pub struct IncompleteCholesky {
    /// Sparse lower-triangular factor.
    l: CsrMatrix,
    /// Dimension.
    n: usize,
    /// Diagonal of L (cached for fast forward/back-substitution).
    l_diag: Vec<f64>,
}

impl IncompleteCholesky {
    /// Factorise a sparse SPD matrix using IC(0).
    ///
    /// Only entries whose (row, col) position appears in the lower triangle
    /// of `a` will be retained in L.  Fill-in is discarded.
    pub fn factorize(a: &CsrMatrix) -> DecompResult<Self> {
        if a.rows == 0 {
            return Err(DecompError::EmptyMatrix {
                context: "IC(0) requires a non-empty matrix".into(),
            });
        }
        if !a.is_square() {
            return Err(DecompError::NotSquare {
                rows: a.rows,
                cols: a.cols,
            });
        }

        let n = a.rows;

        // Build the lower-triangular sparsity pattern (row → sorted col list).
        // Also extract the numerical values from A for the lower triangle.
        let mut lower_cols: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut lower_vals: Vec<Vec<f64>> = vec![Vec::new(); n];

        for i in 0..n {
            let start = a.row_ptr[i];
            let end = a.row_ptr[i + 1];
            for idx in start..end {
                let j = a.col_idx[idx];
                if j <= i {
                    lower_cols[i].push(j);
                    lower_vals[i].push(a.values[idx]);
                }
            }
            // Ensure sorted by column index
            let mut order: Vec<usize> = (0..lower_cols[i].len()).collect();
            order.sort_by_key(|&k| lower_cols[i][k]);
            let sorted_cols: Vec<usize> = order.iter().map(|&k| lower_cols[i][k]).collect();
            let sorted_vals: Vec<f64> = order.iter().map(|&k| lower_vals[i][k]).collect();
            lower_cols[i] = sorted_cols;
            lower_vals[i] = sorted_vals;
        }

        // IC(0) factorisation: operate column-by-column.
        // For each column j = 0..n:
        //   L[j,j] = sqrt( A[j,j] - sum_{k<j, (j,k) in pattern} L[j,k]^2 )
        //   For i > j with (i,j) in pattern:
        //     L[i,j] = ( A[i,j] - sum_{k<j, (i,k) and (j,k) in pattern} L[i,k]*L[j,k] ) / L[j,j]

        // For efficient column access, build a map: for each row i and column
        // index in that row, store position so we can read/write quickly.
        // We work on lower_vals in place.

        // Build an index: for row i, col j → position in lower_cols[i]
        let mut col_pos: Vec<std::collections::HashMap<usize, usize>> = Vec::with_capacity(n);
        for i in 0..n {
            let mut m = std::collections::HashMap::new();
            for (pos, &c) in lower_cols[i].iter().enumerate() {
                m.insert(c, pos);
            }
            col_pos.push(m);
        }

        // Helper to get L[i,j] from our working storage.
        // Returns 0.0 if (i,j) not in pattern.
        macro_rules! get_l {
            ($row:expr, $col:expr) => {
                match col_pos[$row].get(&$col) {
                    Some(&pos) => lower_vals[$row][pos],
                    None => 0.0,
                }
            };
        }

        let mut l_diag = vec![0.0_f64; n];

        for j in 0..n {
            // Diagonal: L[j,j]
            let a_jj = get_l!(j, j);
            let mut sum_sq = 0.0_f64;
            // sum over k < j where (j,k) is in the pattern
            for &c in &lower_cols[j] {
                if c >= j {
                    break;
                }
                let ljk = get_l!(j, c);
                sum_sq += ljk * ljk;
            }
            let diag = a_jj - sum_sq;
            if diag <= MIN_PIVOT {
                return Err(DecompError::NotPositiveDefinite {
                    context: format!(
                        "IC(0) pivot at column {} is {:.2e}, not positive",
                        j, diag
                    ),
                });
            }
            let ljj = diag.sqrt();
            // Write L[j,j]
            if let Some(&pos) = col_pos[j].get(&j) {
                lower_vals[j][pos] = ljj;
            }
            l_diag[j] = ljj;

            let inv_ljj = 1.0 / ljj;

            // Off-diagonal: for each row i > j with (i,j) in pattern
            for i in (j + 1)..n {
                if let Some(&pos_ij) = col_pos[i].get(&j) {
                    let a_ij = lower_vals[i][pos_ij]; // currently holds A[i,j]

                    // Compute the dot product of L[i, 0..j] and L[j, 0..j]
                    // but only where both (i,k) and (j,k) are in the pattern.
                    let mut dot_sum = 0.0_f64;
                    for &c in &lower_cols[j] {
                        if c >= j {
                            break;
                        }
                        // c < j, and (j,c) is in pattern
                        if let Some(&pos_ic) = col_pos[i].get(&c) {
                            let lik = lower_vals[i][pos_ic];
                            let ljk = get_l!(j, c);
                            dot_sum += lik * ljk;
                        }
                    }

                    let lij = (a_ij - dot_sum) * inv_ljj;
                    lower_vals[i][pos_ij] = lij;
                }
            }
        }

        // Pack into CsrMatrix
        let mut row_ptr = vec![0usize; n + 1];
        let mut all_cols = Vec::new();
        let mut all_vals = Vec::new();
        for i in 0..n {
            for (idx, &c) in lower_cols[i].iter().enumerate() {
                all_cols.push(c);
                all_vals.push(lower_vals[i][idx]);
            }
            row_ptr[i + 1] = all_cols.len();
        }

        let l = CsrMatrix {
            rows: n,
            cols: n,
            row_ptr,
            col_idx: all_cols,
            values: all_vals,
        };

        Ok(Self { l, n, l_diag })
    }

    /// Solve LLᵀ x = b  (forward + backward substitution with sparse L).
    pub fn solve(&self, b: &[f64]) -> DecompResult<Vec<f64>> {
        if b.len() != self.n {
            return Err(DecompError::VectorLengthMismatch {
                expected: self.n,
                actual: b.len(),
            });
        }
        let y = self.sparse_forward(b);
        let x = self.sparse_backward(&y);
        Ok(x)
    }

    /// Apply the IC(0) preconditioner: solve LLᵀ z = r.
    ///
    /// Equivalent to `solve` – provided as a named convenience for use in
    /// preconditioned iterative solvers.
    pub fn apply_preconditioner(&self, r: &[f64]) -> DecompResult<Vec<f64>> {
        self.solve(r)
    }

    /// Return a reference to the sparse lower-triangular factor.
    pub fn get_l(&self) -> &CsrMatrix {
        &self.l
    }

    // ── sparse forward substitution: L y = b ────────────────────────────

    fn sparse_forward(&self, b: &[f64]) -> Vec<f64> {
        let n = self.n;
        let mut y = vec![0.0; n];
        for i in 0..n {
            let mut s = b[i];
            let start = self.l.row_ptr[i];
            let end = self.l.row_ptr[i + 1];
            for idx in start..end {
                let j = self.l.col_idx[idx];
                if j < i {
                    s -= self.l.values[idx] * y[j];
                }
            }
            y[i] = s / self.l_diag[i];
        }
        y
    }

    // ── sparse backward substitution: Lᵀ x = y ─────────────────────────
    //
    // Lᵀ is upper-triangular.  We process rows of Lᵀ from bottom to top.
    // Since Lᵀ[j,i] = L[i,j], we scatter L row-by-row in reverse.

    fn sparse_backward(&self, y: &[f64]) -> Vec<f64> {
        let n = self.n;
        let mut x = y.to_vec();
        for i in (0..n).rev() {
            x[i] /= self.l_diag[i];
            let start = self.l.row_ptr[i];
            let end = self.l.row_ptr[i + 1];
            for idx in start..end {
                let j = self.l.col_idx[idx];
                if j < i {
                    x[j] -= self.l.values[idx] * x[i];
                }
            }
        }
        x
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. Approximate Minimum Degree Ordering
// ═══════════════════════════════════════════════════════════════════════════

/// Compute a simple approximate-minimum-degree (AMD) ordering.
///
/// Builds the adjacency graph of the sparse matrix (ignoring self-loops),
/// then greedily selects the vertex with the smallest current degree,
/// removes it from the graph, and adds it to the permutation.  Ties are
/// broken by choosing the vertex with the smallest index.
///
/// This is a simplified AMD; it does not implement mass elimination or
/// element absorption, but it produces reasonable fill-reducing orderings
/// for moderate-sized problems.
pub fn approximate_minimum_degree(a: &CsrMatrix) -> Vec<usize> {
    let n = a.rows;
    if n == 0 {
        return Vec::new();
    }

    // Build adjacency sets (undirected, no self-loops).
    let mut adj: Vec<std::collections::BTreeSet<usize>> = vec![std::collections::BTreeSet::new(); n];
    for i in 0..n {
        let start = a.row_ptr[i];
        let end = a.row_ptr[i + 1];
        for idx in start..end {
            let j = a.col_idx[idx];
            if j != i {
                adj[i].insert(j);
                adj[j].insert(i);
            }
        }
    }

    let mut eliminated = vec![false; n];
    let mut perm = Vec::with_capacity(n);

    for _ in 0..n {
        // Find the un-eliminated vertex with the smallest degree.
        let mut best: Option<usize> = None;
        let mut best_deg = usize::MAX;
        for v in 0..n {
            if eliminated[v] {
                continue;
            }
            let deg = adj[v].iter().filter(|&&u| !eliminated[u]).count();
            if deg < best_deg || (deg == best_deg && best.map_or(true, |b| v < b)) {
                best_deg = deg;
                best = Some(v);
            }
        }

        let v = best.unwrap();
        perm.push(v);
        eliminated[v] = true;

        // Add edges between all un-eliminated neighbours of v (fill edges).
        let neighbours: Vec<usize> = adj[v]
            .iter()
            .copied()
            .filter(|&u| !eliminated[u])
            .collect();
        for i in 0..neighbours.len() {
            for j in (i + 1)..neighbours.len() {
                let u = neighbours[i];
                let w = neighbours[j];
                adj[u].insert(w);
                adj[w].insert(u);
            }
        }
    }

    perm
}

/// Permute a CSR matrix according to `perm`: P A Pᵀ.
///
/// `perm[new_index] = old_index`, so row/col `i` in the permuted matrix
/// corresponds to row/col `perm[i]` in the original.
fn permute_symmetric(a: &CsrMatrix, perm: &[usize]) -> DenseMatrix {
    let n = a.rows;
    // Build inverse permutation
    let mut inv_perm = vec![0usize; n];
    for (new, &old) in perm.iter().enumerate() {
        inv_perm[old] = new;
    }

    let mut b = DenseMatrix::zeros(n, n);
    for i in 0..n {
        let start = a.row_ptr[i];
        let end = a.row_ptr[i + 1];
        for idx in start..end {
            let j = a.col_idx[idx];
            let val = a.values[idx];
            let pi = inv_perm[i];
            let pj = inv_perm[j];
            b.set(pi, pj, val);
        }
    }
    b
}

/// Compute a fill-reducing Cholesky factorisation of a sparse SPD matrix.
///
/// 1. Compute an approximate minimum-degree ordering.
/// 2. Permute the matrix: B = P A Pᵀ.
/// 3. Factorise B (dense Cholesky).
///
/// Returns `(chol, perm)` where `chol` is the factorisation of the permuted
/// matrix and `perm` is the ordering used.
///
/// To solve A x = b:
/// ```ignore
/// let (chol, perm) = sparse_cholesky_with_ordering(&a)?;
/// // permute b
/// let pb: Vec<f64> = perm.iter().map(|&i| b[i]).collect(); // wrong direction
/// // actually need inverse perm for RHS
/// let mut inv = vec![0; n];
/// for (new, &old) in perm.iter().enumerate() { inv[old] = new; }
/// let pb: Vec<f64> = (0..n).map(|i| b[perm[i]]).collect();
/// let px = chol.solve(&pb)?;
/// // unpermute
/// let mut x = vec![0.0; n];
/// for (new, &old) in perm.iter().enumerate() { x[old] = px[new]; }
/// ```
pub fn sparse_cholesky_with_ordering(
    a: &CsrMatrix,
) -> DecompResult<(CholeskyDecomposition, Vec<usize>)> {
    if a.rows == 0 {
        return Err(DecompError::EmptyMatrix {
            context: "sparse Cholesky requires a non-empty matrix".into(),
        });
    }
    if !a.is_square() {
        return Err(DecompError::NotSquare {
            rows: a.rows,
            cols: a.cols,
        });
    }
    let perm = approximate_minimum_degree(a);
    let permuted = permute_symmetric(a, &perm);
    let chol = CholeskyDecomposition::factorize(&permuted)?;
    Ok((chol, perm))
}

// ═══════════════════════════════════════════════════════════════════════════
// Helper: build a sparse SPD test matrix (used in tests)
// ═══════════════════════════════════════════════════════════════════════════

/// Build a simple sparse SPD tridiagonal matrix of size n:
///   A[i,i] = 2,  A[i,i±1] = -1.
#[cfg(test)]
fn build_sparse_spd(n: usize) -> CsrMatrix {
    let mut triplets: Vec<(usize, usize, f64)> = Vec::new();
    for i in 0..n {
        triplets.push((i, i, 2.0));
        if i > 0 {
            triplets.push((i, i - 1, -1.0));
        }
        if i + 1 < n {
            triplets.push((i, i + 1, -1.0));
        }
    }
    CsrMatrix::from_triplets(n, n, &triplets)
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;

    /// Helper: ‖x - y‖∞
    fn max_diff(x: &[f64], y: &[f64]) -> f64 {
        x.iter()
            .zip(y.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max)
    }

    /// Helper: ‖A - B‖∞ (element-wise)
    fn mat_max_diff(a: &DenseMatrix, b: &DenseMatrix) -> f64 {
        assert_eq!(a.rows, b.rows);
        assert_eq!(a.cols, b.cols);
        a.data
            .iter()
            .zip(b.data.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f64, f64::max)
    }

    /// 3×3 SPD matrix used in several tests.
    fn spd_3x3() -> DenseMatrix {
        // A = [[4, 2, 2],
        //      [2, 5, 1],
        //      [2, 1, 6]]
        DenseMatrix::from_row_major(
            3,
            3,
            vec![
                4.0, 2.0, 2.0,
                2.0, 5.0, 1.0,
                2.0, 1.0, 6.0,
            ],
        )
    }

    // ── Test 1: factorize & verify LLᵀ = A ─────────────────────────────

    #[test]
    fn test_factorize_llt_equals_a() {
        let a = spd_3x3();
        let chol = CholeskyDecomposition::factorize(&a).unwrap();
        let l = chol.get_l();
        let lt = l.transpose();
        let reconstructed = l.mul_mat(&lt).unwrap();
        assert!(
            mat_max_diff(&a, &reconstructed) < TOL,
            "LLᵀ should equal A"
        );

        // Verify L is lower triangular
        let n = chol.dim();
        for i in 0..n {
            for j in (i + 1)..n {
                assert_eq!(l.get(i, j), 0.0, "L should be lower triangular");
            }
        }
    }

    // ── Test 2: solve a known system ────────────────────────────────────

    #[test]
    fn test_solve_known_system() {
        let a = spd_3x3();
        let chol = CholeskyDecomposition::factorize(&a).unwrap();

        // True solution x = [1, 2, 3]
        let x_true = vec![1.0, 2.0, 3.0];
        let b = a.mul_vec(&x_true).unwrap();
        let x = chol.solve(&b).unwrap();
        assert!(
            max_diff(&x, &x_true) < TOL,
            "solve should recover the true solution"
        );
    }

    // ── Test 3: determinant ─────────────────────────────────────────────

    #[test]
    fn test_determinant() {
        let a = spd_3x3();
        let chol = CholeskyDecomposition::factorize(&a).unwrap();

        // det(A) by cofactor expansion:
        // 4*(30-1) - 2*(12-2) + 2*(2-10) = 116 - 20 - 16 = 80
        let expected_det = 80.0;
        let det = chol.determinant();
        assert!(
            (det - expected_det).abs() < 1e-6,
            "determinant got {}, expected {}",
            det,
            expected_det
        );
    }

    // ── Test 4: log-determinant ─────────────────────────────────────────

    #[test]
    fn test_log_determinant() {
        let a = spd_3x3();
        let chol = CholeskyDecomposition::factorize(&a).unwrap();
        let log_det = chol.log_determinant();
        let expected = 80.0_f64.ln();
        assert!(
            (log_det - expected).abs() < 1e-10,
            "log_determinant got {}, expected {}",
            log_det,
            expected
        );
    }

    // ── Test 5: inverse ─────────────────────────────────────────────────

    #[test]
    fn test_inverse() {
        let a = spd_3x3();
        let chol = CholeskyDecomposition::factorize(&a).unwrap();
        let a_inv = chol.inverse().unwrap();
        let product = a.mul_mat(&a_inv).unwrap();
        let eye = DenseMatrix::eye(3);
        assert!(
            mat_max_diff(&product, &eye) < 1e-10,
            "A * A⁻¹ should be I"
        );
    }

    // ── Test 6: not positive-definite detection ─────────────────────────

    #[test]
    fn test_not_positive_definite() {
        // A symmetric but indefinite matrix
        let a = DenseMatrix::from_row_major(
            3,
            3,
            vec![
                1.0, 2.0, 3.0,
                2.0, 1.0, 4.0,
                3.0, 4.0, 1.0,
            ],
        );
        let result = CholeskyDecomposition::factorize(&a);
        assert!(result.is_err());
        match result.unwrap_err() {
            DecompError::NotPositiveDefinite { .. } => {}
            e => panic!("expected NotPositiveDefinite, got {:?}", e),
        }
    }

    // ── Test 7: not symmetric detection ─────────────────────────────────

    #[test]
    fn test_not_symmetric() {
        let a = DenseMatrix::from_row_major(
            2,
            2,
            vec![4.0, 1.0, 2.0, 5.0], // a[0,1]=1 != a[1,0]=2
        );
        let result = CholeskyDecomposition::factorize(&a);
        assert!(result.is_err());
        match result.unwrap_err() {
            DecompError::NotSymmetric { .. } => {}
            e => panic!("expected NotSymmetric, got {:?}", e),
        }
    }

    // ── Test 8: modified Cholesky on indefinite matrix ──────────────────

    #[test]
    fn test_modified_cholesky_indefinite() {
        // Symmetric indefinite matrix
        let a = DenseMatrix::from_row_major(
            3,
            3,
            vec![
                1.0, 2.0, 0.0,
                2.0, -3.0, 1.0,
                0.0, 1.0, 1.0,
            ],
        );
        // Standard Cholesky should fail (a[1,1] pivot goes negative)
        assert!(CholeskyDecomposition::factorize(&a).is_err());

        // Modified Cholesky should succeed
        let mc = ModifiedCholeskyDecomposition::factorize(&a).unwrap();
        let l = mc.get_l();
        let lt = l.transpose();
        let llt = l.mul_mat(&lt).unwrap();

        // Verify A + E = LLᵀ
        let e = mc.perturbation();
        let mut a_plus_e = a.clone();
        for i in 0..3 {
            let old = a_plus_e.get(i, i);
            a_plus_e.set(i, i, old + e[i]);
        }
        assert!(
            mat_max_diff(&a_plus_e, &llt) < 1e-10,
            "A + E should equal LLᵀ"
        );

        // The perturbation should be non-trivial (at least one non-zero entry)
        let max_e: f64 = e.iter().map(|v| v.abs()).fold(0.0, f64::max);
        assert!(
            max_e > 0.0,
            "perturbation should be nonzero for indefinite matrix"
        );

        // Solve should work
        let b = vec![1.0, 2.0, 3.0];
        let x = mc.solve(&b).unwrap();
        // Verify (A+E)x = b
        let ax = a_plus_e.mul_vec(&x).unwrap();
        assert!(max_diff(&ax, &b) < 1e-10);
    }

    // ── Test 9: IC(0) sparsity pattern preserved ────────────────────────

    #[test]
    fn test_ic0_sparsity_pattern() {
        let a = build_sparse_spd(5);
        let ic = IncompleteCholesky::factorize(&a).unwrap();
        let l = ic.get_l();

        // The sparsity pattern of L should be a subset of the lower triangle
        // of A's sparsity pattern.
        for i in 0..l.rows {
            let start = l.row_ptr[i];
            let end = l.row_ptr[i + 1];
            for idx in start..end {
                let j = l.col_idx[idx];
                assert!(j <= i, "L should be lower triangular");
                // (i, j) should exist in the lower triangle of A
                assert!(
                    a.get(i, j) != 0.0 || i == j,
                    "L[{},{}] not in pattern of A",
                    i,
                    j
                );
            }
        }
    }

    // ── Test 10: IC(0) as preconditioner ────────────────────────────────

    #[test]
    fn test_ic0_preconditioner() {
        let n = 10;
        let a = build_sparse_spd(n);
        let ic = IncompleteCholesky::factorize(&a).unwrap();

        // For a tridiagonal SPD matrix, IC(0) is exact Cholesky (no fill-in
        // beyond the existing pattern), so solve should be quite accurate.
        let b: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
        let x = ic.apply_preconditioner(&b).unwrap();

        // Check residual ‖Ax - b‖ / ‖b‖
        let ax = a.mul_vec(&x).unwrap();
        let res = norm2(
            &ax.iter()
                .zip(b.iter())
                .map(|(a, b)| a - b)
                .collect::<Vec<_>>(),
        );
        let bnorm = norm2(&b);
        let rel_res = res / bnorm;
        assert!(
            rel_res < 1e-8,
            "IC(0) relative residual too large: {:.2e}",
            rel_res
        );
    }

    // ── Test 11: AMD ordering produces valid permutation ────────────────

    #[test]
    fn test_amd_valid_permutation() {
        let a = build_sparse_spd(8);
        let perm = approximate_minimum_degree(&a);

        assert_eq!(perm.len(), 8);

        // Should be a valid permutation: each index 0..n appears exactly once.
        let mut sorted = perm.clone();
        sorted.sort();
        let expected: Vec<usize> = (0..8).collect();
        assert_eq!(sorted, expected, "AMD should produce a valid permutation");
    }

    // ── Test 12: 1×1 matrix ─────────────────────────────────────────────

    #[test]
    fn test_1x1() {
        let a = DenseMatrix::from_row_major(1, 1, vec![9.0]);
        let chol = CholeskyDecomposition::factorize(&a).unwrap();
        let l = chol.get_l();
        assert!((l.get(0, 0) - 3.0).abs() < TOL);
        assert!((chol.determinant() - 9.0).abs() < TOL);

        let x = chol.solve(&[18.0]).unwrap();
        assert!((x[0] - 2.0).abs() < TOL);
    }

    // ── Test 13: 2×2 matrix ─────────────────────────────────────────────

    #[test]
    fn test_2x2() {
        // A = [[4, 2], [2, 5]]
        let a = DenseMatrix::from_row_major(2, 2, vec![4.0, 2.0, 2.0, 5.0]);
        let chol = CholeskyDecomposition::factorize(&a).unwrap();
        let l = chol.get_l();
        let lt = l.transpose();
        let llt = l.mul_mat(&lt).unwrap();
        assert!(mat_max_diff(&a, &llt) < TOL);

        // det = 20 - 4 = 16
        assert!((chol.determinant() - 16.0).abs() < 1e-8);

        // Solve
        let b = vec![8.0, 9.0];
        let x = chol.solve(&b).unwrap();
        let ax = a.mul_vec(&x).unwrap();
        assert!(max_diff(&ax, &b) < TOL);
    }

    // ── Test 14: solve multiple RHS ─────────────────────────────────────

    #[test]
    fn test_solve_multiple() {
        let a = spd_3x3();
        let chol = CholeskyDecomposition::factorize(&a).unwrap();

        // Two RHS columns
        let b = DenseMatrix::from_row_major(
            3,
            2,
            vec![
                14.0, 8.0,
                13.0, 9.0,
                22.0, 5.0,
            ],
        );
        let x = chol.solve_multiple(&b).unwrap();
        // Verify A * X = B
        let ax = a.mul_mat(&x).unwrap();
        assert!(
            mat_max_diff(&ax, &b) < 1e-10,
            "A * X should equal B for multiple RHS"
        );
    }

    // ── Test 15: sparse Cholesky with ordering ──────────────────────────

    #[test]
    fn test_sparse_cholesky_with_ordering() {
        let n = 6;
        let a = build_sparse_spd(n);
        let (chol, perm) = sparse_cholesky_with_ordering(&a).unwrap();

        // Use the ordering to solve Ax = b
        let b: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();

        // Build inverse perm
        let mut inv_perm = vec![0usize; n];
        for (new, &old) in perm.iter().enumerate() {
            inv_perm[old] = new;
        }

        // Permute b
        let pb: Vec<f64> = (0..n).map(|i| b[perm[i]]).collect();
        let px = chol.solve(&pb).unwrap();

        // Un-permute
        let mut x = vec![0.0; n];
        for (new, &old) in perm.iter().enumerate() {
            x[old] = px[new];
        }

        // Verify residual
        let ax = a.mul_vec(&x).unwrap();
        let res: Vec<f64> = ax.iter().zip(b.iter()).map(|(a, b)| a - b).collect();
        let rel_res = norm2(&res) / norm2(&b);
        assert!(
            rel_res < 1e-10,
            "sparse Cholesky+ordering residual too large: {:.2e}",
            rel_res
        );
    }

    // ── Test 16: modified Cholesky on already-SPD matrix has zero perturbation

    #[test]
    fn test_modified_cholesky_spd_zero_perturbation() {
        let a = spd_3x3();
        let mc = ModifiedCholeskyDecomposition::factorize(&a).unwrap();
        let e = mc.perturbation();
        // For an SPD matrix the GMW perturbation should be negligible
        let max_e: f64 = e.iter().map(|v| v.abs()).fold(0.0, f64::max);
        // The perturbation includes the delta floor, but should be very small
        // compared to the matrix entries.
        let a_norm = a.frobenius_norm();
        assert!(
            max_e / a_norm < 1e-12,
            "perturbation should be negligible for SPD, got {:.2e}",
            max_e / a_norm
        );
    }

    // ── Test 17: empty matrix errors ────────────────────────────────────

    #[test]
    fn test_empty_matrix() {
        let a = DenseMatrix::zeros(0, 0);
        assert!(CholeskyDecomposition::factorize(&a).is_err());
        assert!(ModifiedCholeskyDecomposition::factorize(&a).is_err());
    }

    // ── Test 18: non-square matrix errors ───────────────────────────────

    #[test]
    fn test_non_square() {
        let a = DenseMatrix::zeros(3, 4);
        match CholeskyDecomposition::factorize(&a).unwrap_err() {
            DecompError::NotSquare { rows: 3, cols: 4 } => {}
            e => panic!("expected NotSquare, got {:?}", e),
        }
    }

    // ── Test 19: IC(0) on a larger sparse matrix ────────────────────────

    #[test]
    fn test_ic0_larger_matrix() {
        let n = 20;
        let a = build_sparse_spd(n);
        let ic = IncompleteCholesky::factorize(&a).unwrap();

        let b: Vec<f64> = (0..n).map(|i| ((i * 7 + 3) % 13) as f64).collect();
        let x = ic.solve(&b).unwrap();

        let ax = a.mul_vec(&x).unwrap();
        let res: Vec<f64> = ax.iter().zip(b.iter()).map(|(a, b)| a - b).collect();
        let rel_res = norm2(&res) / norm2(&b);
        assert!(
            rel_res < 1e-8,
            "IC(0) on larger matrix: relative residual {:.2e}",
            rel_res
        );
    }

    // ── Test 20: larger SPD matrix (5×5) ────────────────────────────────

    #[test]
    fn test_5x5_spd() {
        // Build a diagonally-dominant SPD matrix
        let n = 5;
        let mut a = DenseMatrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let v = 1.0 / (1.0 + (i as f64 - j as f64).abs());
                a.set(i, j, v);
            }
            // Make diagonally dominant
            a.set(i, i, a.get(i, i) + n as f64);
        }

        let chol = CholeskyDecomposition::factorize(&a).unwrap();
        let l = chol.get_l();
        let lt = l.transpose();
        let llt = l.mul_mat(&lt).unwrap();
        assert!(
            mat_max_diff(&a, &llt) < 1e-10,
            "5×5 LLᵀ reconstruction"
        );

        // Solve
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x = chol.solve(&b).unwrap();
        let ax = a.mul_vec(&x).unwrap();
        assert!(max_diff(&ax, &b) < 1e-10);

        // Inverse
        let inv = chol.inverse().unwrap();
        let eye = DenseMatrix::eye(n);
        let product = a.mul_mat(&inv).unwrap();
        assert!(mat_max_diff(&product, &eye) < 1e-10);
    }

    // ── Test 21: vector length mismatch ─────────────────────────────────

    #[test]
    fn test_vector_length_mismatch() {
        let a = spd_3x3();
        let chol = CholeskyDecomposition::factorize(&a).unwrap();
        let result = chol.solve(&[1.0, 2.0]);
        assert!(result.is_err());
        match result.unwrap_err() {
            DecompError::VectorLengthMismatch {
                expected: 3,
                actual: 2,
            } => {}
            e => panic!("expected VectorLengthMismatch, got {:?}", e),
        }
    }
}
