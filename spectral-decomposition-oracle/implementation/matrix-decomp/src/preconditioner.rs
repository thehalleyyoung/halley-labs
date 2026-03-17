//! Preconditioners for iterative methods.
//!
//! Provides several preconditioner strategies ranging from simple diagonal
//! (Jacobi) scaling through incomplete factorizations (IC(0), ILU(0)), SSOR
//! sweeps, and block-diagonal solves.  An adaptive selector chooses an
//! appropriate preconditioner based on structural properties of the matrix.

use crate::{CsrMatrix, DenseMatrix, DecompError, DecompResult};

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// A preconditioner maps a residual vector **r** to an approximate inverse
/// application **z ≈ M⁻¹ r** where *M* approximates the system matrix.
pub trait Preconditioner: std::fmt::Debug + Send + Sync {
    /// Apply the preconditioner: compute `z ≈ M⁻¹ r`.
    fn apply(&self, r: &[f64], z: &mut [f64]) -> DecompResult<()>;

    /// Dimension of the vectors this preconditioner operates on.
    fn size(&self) -> usize;
}

// ---------------------------------------------------------------------------
// Identity
// ---------------------------------------------------------------------------

/// Trivial preconditioner: z = r (no preconditioning).
#[derive(Debug, Clone)]
pub struct IdentityPreconditioner {
    n: usize,
}

impl IdentityPreconditioner {
    /// Create an identity preconditioner for vectors of length `n`.
    pub fn new(n: usize) -> Self {
        Self { n }
    }
}

impl Preconditioner for IdentityPreconditioner {
    fn apply(&self, r: &[f64], z: &mut [f64]) -> DecompResult<()> {
        if r.len() != self.n || z.len() != self.n {
            return Err(DecompError::VectorLengthMismatch {
                expected: self.n,
                actual: r.len().max(z.len()),
            });
        }
        z.copy_from_slice(r);
        Ok(())
    }

    fn size(&self) -> usize {
        self.n
    }
}

// ---------------------------------------------------------------------------
// Jacobi (diagonal scaling)
// ---------------------------------------------------------------------------

/// Diagonal (Jacobi) preconditioner: z_i = r_i / a_{ii}.
#[derive(Debug, Clone)]
pub struct JacobiPreconditioner {
    inv_diag: Vec<f64>,
}

impl JacobiPreconditioner {
    /// Build from the diagonal of a matrix.  Zero diagonals are replaced by 1.
    pub fn new(diag: &[f64]) -> Self {
        let inv_diag = diag
            .iter()
            .map(|&d| if d.abs() > 1e-300 { 1.0 / d } else { 1.0 })
            .collect();
        Self { inv_diag }
    }

    /// Build directly from a sparse CSR matrix.
    pub fn from_csr(a: &CsrMatrix) -> Self {
        Self::new(&a.diag())
    }

    /// Build from a sparse matrix, returning an error on zero diagonals.
    pub fn from_csr_strict(a: &CsrMatrix) -> DecompResult<Self> {
        if !a.is_square() {
            return Err(DecompError::NotSquare {
                rows: a.rows,
                cols: a.cols,
            });
        }
        let n = a.rows;
        let diag = a.diag();
        let mut inv_diag = Vec::with_capacity(n);
        for i in 0..n {
            let d = diag[i];
            if d.abs() < 1e-15 {
                return Err(DecompError::ZeroPivot { index: i, value: 0.0 });
            }
            inv_diag.push(1.0 / d);
        }
        Ok(Self { inv_diag })
    }

    /// Build from a dense matrix, returning an error on zero diagonals.
    pub fn from_dense(a: &DenseMatrix) -> DecompResult<Self> {
        if !a.is_square() {
            return Err(DecompError::NotSquare {
                rows: a.rows,
                cols: a.cols,
            });
        }
        let n = a.rows;
        let mut inv_diag = Vec::with_capacity(n);
        for i in 0..n {
            let d = a.get(i, i);
            if d.abs() < 1e-15 {
                return Err(DecompError::ZeroPivot { index: i, value: 0.0 });
            }
            inv_diag.push(1.0 / d);
        }
        Ok(Self { inv_diag })
    }
}

impl Preconditioner for JacobiPreconditioner {
    fn apply(&self, r: &[f64], z: &mut [f64]) -> DecompResult<()> {
        let n = self.inv_diag.len();
        if r.len() != n || z.len() != n {
            return Err(DecompError::VectorLengthMismatch {
                expected: n,
                actual: r.len().max(z.len()),
            });
        }
        for i in 0..n {
            z[i] = self.inv_diag[i] * r[i];
        }
        Ok(())
    }

    fn size(&self) -> usize {
        self.inv_diag.len()
    }
}

// ---------------------------------------------------------------------------
// SSOR (Symmetric Successive Over-Relaxation)
// ---------------------------------------------------------------------------

/// SSOR preconditioner with relaxation parameter ω.
///
/// The preconditioner is  M = (D/ω + L) (D/ω)⁻¹ (D/ω + U)  scaled by
/// ω(2−ω), where D is the diagonal, L the strict lower triangle, and U the
/// strict upper triangle of A stored in CSR form.
#[derive(Debug, Clone)]
pub struct SsorPreconditioner {
    n: usize,
    omega: f64,
    row_ptr: Vec<usize>,
    col_idx: Vec<usize>,
    values: Vec<f64>,
    diag: Vec<f64>,
}

impl SsorPreconditioner {
    /// Build the SSOR preconditioner.  `omega` must be in (0, 2).
    pub fn new(a: &CsrMatrix, omega: f64) -> DecompResult<Self> {
        if !a.is_square() {
            return Err(DecompError::NotSquare {
                rows: a.rows,
                cols: a.cols,
            });
        }
        if omega <= 0.0 || omega >= 2.0 {
            return Err(DecompError::InvalidParameter {
                name: "omega".into(),
                value: omega.to_string(),
                reason: "SSOR relaxation parameter must be in (0, 2)".into(),
            });
        }
        let n = a.rows;
        let diag = a.diag();
        for i in 0..n {
            if diag[i].abs() < 1e-15 {
                return Err(DecompError::ZeroPivot { index: i, value: 0.0 });
            }
        }
        Ok(Self {
            n,
            omega,
            row_ptr: a.row_ptr.clone(),
            col_idx: a.col_idx.clone(),
            values: a.values.clone(),
            diag,
        })
    }
}

impl Preconditioner for SsorPreconditioner {
    fn apply(&self, r: &[f64], z: &mut [f64]) -> DecompResult<()> {
        let n = self.n;
        if r.len() != n || z.len() != n {
            return Err(DecompError::VectorLengthMismatch {
                expected: n,
                actual: r.len().max(z.len()),
            });
        }
        let omega = self.omega;

        // Forward sweep: solve (D/ω + L) y = r
        //   y[i] = (ω / d[i]) * (r[i] - Σ_{j<i} a[i,j]*y[j])
        for i in 0..n {
            let mut sigma = 0.0;
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];
            for idx in start..end {
                let j = self.col_idx[idx];
                if j < i {
                    sigma += self.values[idx] * z[j];
                }
            }
            z[i] = (omega / self.diag[i]) * (r[i] - sigma);
        }

        // Scale by D/ω to get intermediate vector v = (D/ω) y.
        for i in 0..n {
            z[i] *= self.diag[i] / omega;
        }

        // Backward sweep: solve (D/ω + U) z = v
        //   z[i] = (ω / d[i]) * (v[i] - Σ_{j>i} a[i,j]*z[j])
        for i in (0..n).rev() {
            let mut sigma = 0.0;
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];
            for idx in start..end {
                let j = self.col_idx[idx];
                if j > i {
                    sigma += self.values[idx] * z[j];
                }
            }
            z[i] = (omega / self.diag[i]) * (z[i] - sigma);
        }

        // Overall scaling factor.
        let factor = omega * (2.0 - omega);
        for v in z.iter_mut() {
            *v *= factor;
        }

        Ok(())
    }

    fn size(&self) -> usize {
        self.n
    }
}

// ---------------------------------------------------------------------------
// Incomplete Cholesky IC(0)
// ---------------------------------------------------------------------------

/// Incomplete Cholesky IC(0) preconditioner for symmetric positive-definite
/// matrices.  The sparsity pattern of L is the same as the lower triangle of
/// A.
#[derive(Debug, Clone)]
pub struct IncompleteCholeskyPreconditioner {
    n: usize,
    /// CSR representation of the lower-triangular factor L.
    l_row_ptr: Vec<usize>,
    l_col_idx: Vec<usize>,
    l_values: Vec<f64>,
}

impl IncompleteCholeskyPreconditioner {
    /// Compute IC(0) from a symmetric positive-definite sparse matrix.
    pub fn new(a: &CsrMatrix) -> DecompResult<Self> {
        if !a.is_square() {
            return Err(DecompError::NotSquare {
                rows: a.rows,
                cols: a.cols,
            });
        }
        let n = a.rows;

        // Extract lower triangle in CSR form.
        let mut l_row_ptr: Vec<usize> = Vec::with_capacity(n + 1);
        let mut l_col_idx: Vec<usize> = Vec::new();
        let mut l_values: Vec<f64> = Vec::new();
        l_row_ptr.push(0);

        for i in 0..n {
            let start = a.row_ptr[i];
            let end = a.row_ptr[i + 1];
            for idx in start..end {
                let j = a.col_idx[idx];
                if j <= i {
                    l_col_idx.push(j);
                    l_values.push(a.values[idx]);
                }
            }
            l_row_ptr.push(l_col_idx.len());
        }

        // Build column-index → position lookup per row.
        let mut row_col_pos: Vec<std::collections::HashMap<usize, usize>> =
            Vec::with_capacity(n);
        for i in 0..n {
            let mut m = std::collections::HashMap::new();
            for idx in l_row_ptr[i]..l_row_ptr[i + 1] {
                m.insert(l_col_idx[idx], idx);
            }
            row_col_pos.push(m);
        }

        // IC(0) factorization in-place on l_values.
        for i in 0..n {
            let diag_pos = match row_col_pos[i].get(&i) {
                Some(&p) => p,
                None => {
                    return Err(DecompError::ZeroPivot { index: i, value: 0.0 });
                }
            };

            let row_start = l_row_ptr[i];
            let row_end = l_row_ptr[i + 1];

            // Off-diagonal entries: l[i,j] for j < i.
            for idx in row_start..row_end {
                let j = l_col_idx[idx];
                if j >= i {
                    continue;
                }
                let ljj_pos = match row_col_pos[j].get(&j) {
                    Some(&p) => p,
                    None => return Err(DecompError::ZeroPivot { index: j, value: 0.0 }),
                };
                let ljj = l_values[ljj_pos];
                if ljj.abs() < 1e-15 {
                    return Err(DecompError::ZeroPivot { index: j, value: 0.0 });
                }

                let mut sum = 0.0;
                for idx2 in row_start..row_end {
                    let k = l_col_idx[idx2];
                    if k >= j {
                        break;
                    }
                    if let Some(&pos_jk) = row_col_pos[j].get(&k) {
                        sum += l_values[idx2] * l_values[pos_jk];
                    }
                }
                l_values[idx] = (l_values[idx] - sum) / ljj;
            }

            // Diagonal: l[i,i] = sqrt( a[i,i] - Σ_{k<i} l[i,k]² )
            let mut sum_sq = 0.0;
            for idx in row_start..row_end {
                let k = l_col_idx[idx];
                if k < i {
                    sum_sq += l_values[idx] * l_values[idx];
                }
            }
            let diag_val = l_values[diag_pos] - sum_sq;
            if diag_val <= 0.0 {
                return Err(DecompError::NotPositiveDefinite { context: "IC(0) diagonal became non-positive".into() });
            }
            l_values[diag_pos] = diag_val.sqrt();
        }

        Ok(Self {
            n,
            l_row_ptr,
            l_col_idx,
            l_values,
        })
    }

    /// Forward substitution: solve L y = b.
    fn forward_solve(&self, b: &[f64], y: &mut [f64]) {
        let n = self.n;
        for i in 0..n {
            let mut sum = 0.0;
            let start = self.l_row_ptr[i];
            let end = self.l_row_ptr[i + 1];
            let mut diag_val = 1.0;
            for idx in start..end {
                let j = self.l_col_idx[idx];
                if j < i {
                    sum += self.l_values[idx] * y[j];
                } else if j == i {
                    diag_val = self.l_values[idx];
                }
            }
            y[i] = (b[i] - sum) / diag_val;
        }
    }

    /// Backward substitution: solve Lᵀ z = y.
    fn backward_solve(&self, y: &[f64], z: &mut [f64]) {
        let n = self.n;
        z.copy_from_slice(y);

        for i in (0..n).rev() {
            let start = self.l_row_ptr[i];
            let end = self.l_row_ptr[i + 1];
            let mut diag_val = 1.0;
            for idx in start..end {
                if self.l_col_idx[idx] == i {
                    diag_val = self.l_values[idx];
                    break;
                }
            }
            z[i] /= diag_val;
            for idx in start..end {
                let j = self.l_col_idx[idx];
                if j < i {
                    z[j] -= self.l_values[idx] * z[i];
                }
            }
        }
    }
}

impl Preconditioner for IncompleteCholeskyPreconditioner {
    fn apply(&self, r: &[f64], z: &mut [f64]) -> DecompResult<()> {
        let n = self.n;
        if r.len() != n || z.len() != n {
            return Err(DecompError::VectorLengthMismatch {
                expected: n,
                actual: r.len().max(z.len()),
            });
        }
        let mut y = vec![0.0; n];
        self.forward_solve(r, &mut y);
        self.backward_solve(&y, z);
        Ok(())
    }

    fn size(&self) -> usize {
        self.n
    }
}

// ---------------------------------------------------------------------------
// Incomplete LU  ILU(0)
// ---------------------------------------------------------------------------

/// ILU(0) preconditioner: LU factorization that retains only the sparsity
/// pattern of the original matrix.
#[derive(Debug, Clone)]
pub struct IncompleteLuPreconditioner {
    n: usize,
    row_ptr: Vec<usize>,
    col_idx: Vec<usize>,
    /// Combined LU values.  For j < i the entry is l[i,j]; for j >= i the
    /// entry is u[i,j].  L has implicit unit diagonal.
    lu_values: Vec<f64>,
}

impl IncompleteLuPreconditioner {
    /// Compute ILU(0) from a sparse matrix.
    pub fn new(a: &CsrMatrix) -> DecompResult<Self> {
        if !a.is_square() {
            return Err(DecompError::NotSquare {
                rows: a.rows,
                cols: a.cols,
            });
        }
        let n = a.rows;

        let row_ptr = a.row_ptr.clone();
        let col_idx = a.col_idx.clone();
        let mut lu = a.values.clone();

        // Fast col→position lookup per row.
        let mut row_map: Vec<std::collections::HashMap<usize, usize>> =
            Vec::with_capacity(n);
        for i in 0..n {
            let mut m = std::collections::HashMap::new();
            for idx in row_ptr[i]..row_ptr[i + 1] {
                m.insert(col_idx[idx], idx);
            }
            row_map.push(m);
        }

        // ILU(0) factorization.
        for i in 1..n {
            let ri_start = row_ptr[i];
            let ri_end = row_ptr[i + 1];

            let mut lower_cols: Vec<usize> = Vec::new();
            for idx in ri_start..ri_end {
                if col_idx[idx] < i {
                    lower_cols.push(col_idx[idx]);
                }
            }
            lower_cols.sort_unstable();

            for &j in &lower_cols {
                let pos_ij = row_map[i][&j];
                let pos_jj = match row_map[j].get(&j) {
                    Some(&p) => p,
                    None => return Err(DecompError::ZeroPivot { index: j, value: 0.0 }),
                };
                let ujj = lu[pos_jj];
                if ujj.abs() < 1e-15 {
                    return Err(DecompError::ZeroPivot { index: j, value: 0.0 });
                }
                lu[pos_ij] /= ujj;
                let l_ij = lu[pos_ij];

                let rj_start = row_ptr[j];
                let rj_end = row_ptr[j + 1];
                for jdx in rj_start..rj_end {
                    let k = col_idx[jdx];
                    if k <= j {
                        continue;
                    }
                    if let Some(&pos_ik) = row_map[i].get(&k) {
                        lu[pos_ik] -= l_ij * lu[jdx];
                    }
                }
            }
        }

        // Verify no zero pivots.
        for i in 0..n {
            let pos_ii = match row_map[i].get(&i) {
                Some(&p) => p,
                None => return Err(DecompError::ZeroPivot { index: i, value: 0.0 }),
            };
            if lu[pos_ii].abs() < 1e-15 {
                return Err(DecompError::ZeroPivot { index: i, value: 0.0 });
            }
        }

        Ok(Self {
            n,
            row_ptr,
            col_idx,
            lu_values: lu,
        })
    }

    /// Forward substitution with unit-lower-triangular L.
    fn forward_solve(&self, b: &[f64], y: &mut [f64]) {
        let n = self.n;
        y.copy_from_slice(b);
        for i in 0..n {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];
            for idx in start..end {
                let j = self.col_idx[idx];
                if j < i {
                    y[i] -= self.lu_values[idx] * y[j];
                }
            }
        }
    }

    /// Backward substitution with upper-triangular U.
    fn backward_solve(&self, y: &[f64], z: &mut [f64]) {
        let n = self.n;
        z.copy_from_slice(y);
        for i in (0..n).rev() {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];
            let mut diag_val = 1.0;
            for idx in start..end {
                let j = self.col_idx[idx];
                if j == i {
                    diag_val = self.lu_values[idx];
                } else if j > i {
                    z[i] -= self.lu_values[idx] * z[j];
                }
            }
            z[i] /= diag_val;
        }
    }
}

impl Preconditioner for IncompleteLuPreconditioner {
    fn apply(&self, r: &[f64], z: &mut [f64]) -> DecompResult<()> {
        let n = self.n;
        if r.len() != n || z.len() != n {
            return Err(DecompError::VectorLengthMismatch {
                expected: n,
                actual: r.len().max(z.len()),
            });
        }
        let mut y = vec![0.0; n];
        self.forward_solve(r, &mut y);
        self.backward_solve(&y, z);
        Ok(())
    }

    fn size(&self) -> usize {
        self.n
    }
}

// ---------------------------------------------------------------------------
// Block Jacobi
// ---------------------------------------------------------------------------

/// Block-diagonal preconditioner that partitions the matrix into square blocks
/// along the diagonal and solves each block independently via dense LU.
#[derive(Debug, Clone)]
pub struct BlockJacobiPreconditioner {
    n: usize,
    block_size: usize,
    /// LU factorizations stored as combined L\U matrices (L is unit-lower,
    /// U is upper including the diagonal).
    lu_blocks: Vec<DenseMatrix>,
    /// Pivot permutation vectors for each block.
    pivots: Vec<Vec<usize>>,
}

impl BlockJacobiPreconditioner {
    /// Build from a dense matrix with given `block_size`.  The last block may
    /// be smaller if `n` is not a multiple of `block_size`.
    pub fn new(a: &DenseMatrix, block_size: usize) -> DecompResult<Self> {
        if !a.is_square() {
            return Err(DecompError::NotSquare {
                rows: a.rows,
                cols: a.cols,
            });
        }
        if block_size == 0 {
            return Err(DecompError::InvalidParameter {
                name: "block_size".into(),
                value: "0".into(),
                reason: "block_size must be > 0".into(),
            });
        }
        let n = a.rows;
        let num_blocks = (n + block_size - 1) / block_size;
        let mut lu_blocks = Vec::with_capacity(num_blocks);
        let mut pivots = Vec::with_capacity(num_blocks);

        for b in 0..num_blocks {
            let r0 = b * block_size;
            let r1 = ((b + 1) * block_size).min(n);
            let bsz = r1 - r0;

            // Extract diagonal block.
            let mut block = DenseMatrix::zeros(bsz, bsz);
            for i in 0..bsz {
                for j in 0..bsz {
                    block.set(i, j, a.get(r0 + i, r0 + j));
                }
            }

            // LU factorization with partial pivoting.
            let mut piv: Vec<usize> = (0..bsz).collect();
            for k in 0..bsz {
                let mut max_val = block.get(k, k).abs();
                let mut max_row = k;
                for i in (k + 1)..bsz {
                    let v = block.get(i, k).abs();
                    if v > max_val {
                        max_val = v;
                        max_row = i;
                    }
                }
                if max_val < 1e-15 {
                    return Err(DecompError::SingularMatrix { context: "block Jacobi factorization".into() });
                }
                if max_row != k {
                    block.swap_rows(k, max_row);
                    piv.swap(k, max_row);
                }
                let akk = block.get(k, k);
                for i in (k + 1)..bsz {
                    let lik = block.get(i, k) / akk;
                    block.set(i, k, lik);
                    for j in (k + 1)..bsz {
                        let v = block.get(i, j) - lik * block.get(k, j);
                        block.set(i, j, v);
                    }
                }
            }
            lu_blocks.push(block);
            pivots.push(piv);
        }

        Ok(Self {
            n,
            block_size,
            lu_blocks,
            pivots,
        })
    }

    /// Solve a single block's LU system.
    fn solve_block(lu: &DenseMatrix, piv: &[usize], b: &[f64], x: &mut [f64]) {
        let m = lu.rows;

        // Apply permutation.
        let mut pb = vec![0.0; m];
        for i in 0..m {
            pb[i] = b[piv[i]];
        }

        // Forward substitution (unit lower).
        let mut y = vec![0.0; m];
        for i in 0..m {
            let mut s = pb[i];
            for j in 0..i {
                s -= lu.get(i, j) * y[j];
            }
            y[i] = s;
        }

        // Backward substitution (upper with diagonal).
        for i in (0..m).rev() {
            let mut s = y[i];
            for j in (i + 1)..m {
                s -= lu.get(i, j) * x[j];
            }
            x[i] = s / lu.get(i, i);
        }
    }
}

impl Preconditioner for BlockJacobiPreconditioner {
    fn apply(&self, r: &[f64], z: &mut [f64]) -> DecompResult<()> {
        if r.len() != self.n || z.len() != self.n {
            return Err(DecompError::VectorLengthMismatch {
                expected: self.n,
                actual: r.len().max(z.len()),
            });
        }
        let num_blocks = self.lu_blocks.len();
        for b in 0..num_blocks {
            let r0 = b * self.block_size;
            let r1 = ((b + 1) * self.block_size).min(self.n);
            Self::solve_block(
                &self.lu_blocks[b],
                &self.pivots[b],
                &r[r0..r1],
                &mut z[r0..r1],
            );
        }
        Ok(())
    }

    fn size(&self) -> usize {
        self.n
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Returns `true` when every row satisfies  |a_{ii}| ≥ Σ_{j≠i} |a_{ij}|.
pub fn is_diagonally_dominant(a: &CsrMatrix) -> bool {
    if !a.is_square() {
        return false;
    }
    let n = a.rows;
    for i in 0..n {
        let start = a.row_ptr[i];
        let end = a.row_ptr[i + 1];
        let mut diag_abs = 0.0_f64;
        let mut off_sum = 0.0_f64;
        for idx in start..end {
            let j = a.col_idx[idx];
            let v = a.values[idx].abs();
            if j == i {
                diag_abs = v;
            } else {
                off_sum += v;
            }
        }
        if diag_abs < off_sum {
            return false;
        }
    }
    true
}

/// Check whether a sparse matrix is (approximately) symmetric.
fn is_sparse_symmetric(a: &CsrMatrix, tol: f64) -> bool {
    if !a.is_square() {
        return false;
    }
    let n = a.rows;
    for i in 0..n {
        let start = a.row_ptr[i];
        let end = a.row_ptr[i + 1];
        for idx in start..end {
            let j = a.col_idx[idx];
            let aij = a.values[idx];
            let aji = a.get(j, i);
            if (aij - aji).abs() > tol {
                return false;
            }
        }
    }
    true
}

/// Check whether all diagonal entries are positive (necessary for SPD).
fn has_positive_diagonal(a: &CsrMatrix) -> bool {
    let diag = a.diag();
    diag.iter().all(|&d| d > 0.0)
}

// ---------------------------------------------------------------------------
// Adaptive selector
// ---------------------------------------------------------------------------

/// Adaptive preconditioner selector.  Examines structural properties of the
/// matrix and returns the most suitable preconditioner wrapped in a
/// trait-object.
#[derive(Debug)]
pub struct AdaptivePreconditioner;

impl AdaptivePreconditioner {
    /// Choose a preconditioner based on the properties of `a`:
    ///
    /// * If `a` is diagonally dominant → [`JacobiPreconditioner`].
    /// * If `a` appears symmetric with positive diagonal → [`IncompleteCholeskyPreconditioner`].
    /// * Otherwise → [`IncompleteLuPreconditioner`].
    ///
    /// Falls back gracefully if the selected factorization fails.
    pub fn select(a: &CsrMatrix) -> Box<dyn Preconditioner> {
        // 1. Diagonally dominant → Jacobi.
        if is_diagonally_dominant(a) {
            if let Ok(p) = JacobiPreconditioner::from_csr_strict(a) {
                return Box::new(p);
            }
        }

        // 2. Symmetric with positive diagonal → try IC(0).
        if is_sparse_symmetric(a, 1e-10) && has_positive_diagonal(a) {
            if let Ok(p) = IncompleteCholeskyPreconditioner::new(a) {
                return Box::new(p);
            }
        }

        // 3. General → ILU(0).
        if let Ok(p) = IncompleteLuPreconditioner::new(a) {
            return Box::new(p);
        }

        // 4. Last resort → Jacobi (safe fallback that never fails).
        Box::new(JacobiPreconditioner::from_csr(a))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CsrMatrix, DenseMatrix, norm2};

    /// Helper: build an n×n diagonal sparse matrix.
    fn diag_csr(d: &[f64]) -> CsrMatrix {
        let n = d.len();
        let triplets: Vec<(usize, usize, f64)> =
            d.iter().enumerate().map(|(i, &v)| (i, i, v)).collect();
        CsrMatrix::from_triplets(n, n, &triplets)
    }

    /// Helper: build a small SPD tridiagonal matrix  [2 -1; -1 2; ...]
    fn spd_tridiag(n: usize) -> CsrMatrix {
        let mut triplets = Vec::new();
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

    /// Helper: build a diagonally dominant non-symmetric matrix.
    fn dd_nonsymmetric(n: usize) -> CsrMatrix {
        let mut triplets = Vec::new();
        for i in 0..n {
            triplets.push((i, i, 10.0));
            if i > 0 {
                triplets.push((i, i - 1, -2.0));
            }
            if i + 1 < n {
                triplets.push((i, i + 1, -1.0));
            }
        }
        CsrMatrix::from_triplets(n, n, &triplets)
    }

    #[test]
    fn test_identity_preconditioner() {
        let n = 5;
        let pc = IdentityPreconditioner::new(n);
        assert_eq!(pc.size(), n);
        let r = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut z = vec![0.0; n];
        pc.apply(&r, &mut z).unwrap();
        assert_eq!(z, r);
    }

    #[test]
    fn test_jacobi_on_diagonal_matrix() {
        let d = vec![2.0, 4.0, 8.0, 16.0];
        let a = diag_csr(&d);
        let pc = JacobiPreconditioner::from_csr(&a);
        assert_eq!(pc.size(), 4);

        let r = vec![2.0, 8.0, 16.0, 32.0];
        let mut z = vec![0.0; 4];
        pc.apply(&r, &mut z).unwrap();
        let expected = vec![1.0, 2.0, 2.0, 2.0];
        for i in 0..4 {
            assert!((z[i] - expected[i]).abs() < 1e-12, "z[{i}] = {}", z[i]);
        }
    }

    #[test]
    fn test_jacobi_from_dense() {
        let a = DenseMatrix::from_diag(&[3.0, 6.0, 9.0]);
        let pc = JacobiPreconditioner::from_dense(&a).unwrap();
        assert_eq!(pc.size(), 3);

        let r = vec![3.0, 12.0, 27.0];
        let mut z = vec![0.0; 3];
        pc.apply(&r, &mut z).unwrap();
        let expected = vec![1.0, 2.0, 3.0];
        for i in 0..3 {
            assert!((z[i] - expected[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn test_ssor_reduces_residual() {
        let n = 8;
        let a = spd_tridiag(n);
        let pc = SsorPreconditioner::new(&a, 1.0).unwrap();
        assert_eq!(pc.size(), n);

        let r: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
        let mut z = vec![0.0; n];
        pc.apply(&r, &mut z).unwrap();

        let z_norm = norm2(&z);
        assert!(z_norm > 1e-8, "z should be non-trivial");

        let az = a.mul_vec(&z).unwrap();
        let mut residual = vec![0.0; n];
        for i in 0..n {
            residual[i] = r[i] - az[i];
        }
        let res_norm = norm2(&residual);
        let r_norm = norm2(&r);
        assert!(
            res_norm < r_norm,
            "SSOR should reduce the residual: res={res_norm}, r={r_norm}"
        );
    }

    #[test]
    fn test_ic0_on_spd_matrix() {
        let n = 6;
        let a = spd_tridiag(n);
        let pc = IncompleteCholeskyPreconditioner::new(&a).unwrap();
        assert_eq!(pc.size(), n);

        // For tridiagonal SPD, IC(0) is exact Cholesky.
        let r = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        let mut z = vec![0.0; n];
        pc.apply(&r, &mut z).unwrap();

        let az = a.mul_vec(&z).unwrap();
        for i in 0..n {
            assert!(
                (az[i] - r[i]).abs() < 1e-10,
                "IC(0) solve mismatch at {i}: az={} r={}",
                az[i],
                r[i]
            );
        }
    }

    #[test]
    fn test_ilu0_on_general_matrix() {
        let n = 6;
        let a = dd_nonsymmetric(n);
        let pc = IncompleteLuPreconditioner::new(&a).unwrap();
        assert_eq!(pc.size(), n);

        // For tridiagonal, ILU(0) is exact LU.
        let r = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut z = vec![0.0; n];
        pc.apply(&r, &mut z).unwrap();

        let az = a.mul_vec(&z).unwrap();
        for i in 0..n {
            assert!(
                (az[i] - r[i]).abs() < 1e-10,
                "ILU(0) solve mismatch at {i}: az={} r={}",
                az[i],
                r[i]
            );
        }
    }

    #[test]
    fn test_block_jacobi() {
        // 4×4 matrix with 2×2 block-diagonal structure.
        let mut a = DenseMatrix::zeros(4, 4);
        a.set(0, 0, 2.0);
        a.set(0, 1, 1.0);
        a.set(1, 0, 1.0);
        a.set(1, 1, 3.0);
        a.set(2, 2, 4.0);
        a.set(2, 3, 2.0);
        a.set(3, 2, 1.0);
        a.set(3, 3, 5.0);

        let pc = BlockJacobiPreconditioner::new(&a, 2).unwrap();
        assert_eq!(pc.size(), 4);

        let r = vec![5.0, 7.0, 14.0, 11.0];
        let mut z = vec![0.0; 4];
        pc.apply(&r, &mut z).unwrap();

        let b0_ax0 = 2.0 * z[0] + 1.0 * z[1];
        let b0_ax1 = 1.0 * z[0] + 3.0 * z[1];
        assert!((b0_ax0 - 5.0).abs() < 1e-10, "block0 row0: {b0_ax0}");
        assert!((b0_ax1 - 7.0).abs() < 1e-10, "block0 row1: {b0_ax1}");
        let b1_ax0 = 4.0 * z[2] + 2.0 * z[3];
        let b1_ax1 = 1.0 * z[2] + 5.0 * z[3];
        assert!((b1_ax0 - 14.0).abs() < 1e-10, "block1 row0: {b1_ax0}");
        assert!((b1_ax1 - 11.0).abs() < 1e-10, "block1 row1: {b1_ax1}");
    }

    #[test]
    fn test_adaptive_selects_jacobi_for_dd() {
        let a = dd_nonsymmetric(5);
        assert!(is_diagonally_dominant(&a));
        let pc = AdaptivePreconditioner::select(&a);
        assert_eq!(pc.size(), 5);
        let dbg = format!("{:?}", pc);
        assert!(
            dbg.contains("Jacobi"),
            "Expected Jacobi, got: {dbg}"
        );
    }

    #[test]
    fn test_adaptive_selects_ic_for_spd() {
        // Non-diagonally-dominant symmetric with positive diagonal.
        let n = 4;
        let mut triplets = Vec::new();
        for i in 0..n {
            triplets.push((i, i, 2.0));
            if i > 0 {
                triplets.push((i, i - 1, -1.5));
            }
            if i + 1 < n {
                triplets.push((i, i + 1, -1.5));
            }
        }
        let a2 = CsrMatrix::from_triplets(n, n, &triplets);
        let pc = AdaptivePreconditioner::select(&a2);
        assert_eq!(pc.size(), n);

        // Also test the well-conditioned SPD tridiagonal.
        let a = spd_tridiag(n);
        let pc2 = AdaptivePreconditioner::select(&a);
        assert_eq!(pc2.size(), n);
    }

    #[test]
    fn test_diagonal_dominance_check() {
        let d = vec![10.0, 10.0, 10.0];
        let a = diag_csr(&d);
        assert!(is_diagonally_dominant(&a));

        let triplets = vec![
            (0, 0, 1.0),
            (0, 1, 2.0),
            (1, 0, 2.0),
            (1, 1, 1.0),
        ];
        let b = CsrMatrix::from_triplets(2, 2, &triplets);
        assert!(!is_diagonally_dominant(&b));
    }

    #[test]
    fn test_preconditioner_size_matches_matrix() {
        let n = 7;
        let a = spd_tridiag(n);
        let dense_a = a.to_dense();

        assert_eq!(IdentityPreconditioner::new(n).size(), n);
        assert_eq!(JacobiPreconditioner::from_csr(&a).size(), n);
        assert_eq!(JacobiPreconditioner::from_dense(&dense_a).unwrap().size(), n);
        assert_eq!(SsorPreconditioner::new(&a, 1.2).unwrap().size(), n);
        assert_eq!(IncompleteCholeskyPreconditioner::new(&a).unwrap().size(), n);

        let dd = dd_nonsymmetric(n);
        assert_eq!(IncompleteLuPreconditioner::new(&dd).unwrap().size(), n);
        assert_eq!(BlockJacobiPreconditioner::new(&dense_a, 3).unwrap().size(), n);
    }

    #[test]
    fn test_apply_preserves_vector_length() {
        let n = 5;
        let a = spd_tridiag(n);
        let r = vec![1.0; n];

        let pcs: Vec<Box<dyn Preconditioner>> = vec![
            Box::new(IdentityPreconditioner::new(n)),
            Box::new(JacobiPreconditioner::from_csr(&a)),
            Box::new(SsorPreconditioner::new(&a, 1.0).unwrap()),
            Box::new(IncompleteCholeskyPreconditioner::new(&a).unwrap()),
        ];

        for pc in &pcs {
            let mut z = vec![0.0; n];
            pc.apply(&r, &mut z).unwrap();
            assert_eq!(z.len(), n);
            assert!(z.iter().any(|&v| v.abs() > 1e-15));
        }
    }

    #[test]
    fn test_vector_length_mismatch_errors() {
        let pc = IdentityPreconditioner::new(3);
        let r = vec![1.0; 5];
        let mut z = vec![0.0; 3];
        assert!(pc.apply(&r, &mut z).is_err());

        let r2 = vec![1.0; 3];
        let mut z2 = vec![0.0; 5];
        assert!(pc.apply(&r2, &mut z2).is_err());
    }

    #[test]
    fn test_jacobi_zero_diagonal_strict_error() {
        let triplets = vec![
            (0, 0, 1.0),
            (0, 1, 2.0),
            (1, 0, 3.0),
        ];
        let a = CsrMatrix::from_triplets(2, 2, &triplets);
        assert!(JacobiPreconditioner::from_csr_strict(&a).is_err());
    }

    #[test]
    fn test_ssor_omega_bounds() {
        let a = spd_tridiag(3);
        assert!(SsorPreconditioner::new(&a, 0.0).is_err());
        assert!(SsorPreconditioner::new(&a, 2.0).is_err());
        assert!(SsorPreconditioner::new(&a, -0.5).is_err());
        assert!(SsorPreconditioner::new(&a, 0.5).is_ok());
        assert!(SsorPreconditioner::new(&a, 1.5).is_ok());
    }

    #[test]
    fn test_block_jacobi_non_divisible() {
        // 5×5 diagonal with block_size=3 → blocks of 3 and 2.
        let mut a = DenseMatrix::zeros(5, 5);
        for i in 0..5 {
            a.set(i, i, (i + 2) as f64);
        }
        let pc = BlockJacobiPreconditioner::new(&a, 3).unwrap();
        assert_eq!(pc.size(), 5);

        let r = vec![2.0, 3.0, 4.0, 5.0, 6.0];
        let mut z = vec![0.0; 5];
        pc.apply(&r, &mut z).unwrap();
        for i in 0..5 {
            let expected = r[i] / (i + 2) as f64;
            assert!(
                (z[i] - expected).abs() < 1e-10,
                "z[{i}]={} expected {expected}",
                z[i]
            );
        }
    }
}
