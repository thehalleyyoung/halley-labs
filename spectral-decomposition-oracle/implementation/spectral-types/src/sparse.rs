//! Complete sparse matrix types: CSR, CSC, COO formats with full operations.

use serde::{Deserialize, Serialize};
use crate::error::{self, MatrixError, SpectralError};
use crate::scalar::Scalar;

/// Coordinate (COO / triplet) format sparse matrix.
///
/// Stores non-zero entries as `(row, col, value)` triplets. This format is
/// ideal for incremental assembly — entries can be [`push`](Self::push)ed in
/// any order and duplicate coordinates are preserved (summed on conversion).
/// Convert to [`CsrMatrix`] or [`CscMatrix`] for efficient arithmetic.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct CooMatrix<T: Scalar> {
    pub rows: usize,
    pub cols: usize,
    pub row_indices: Vec<usize>,
    pub col_indices: Vec<usize>,
    pub values: Vec<T>,
}

impl<T: Scalar> CooMatrix<T> {
    /// Create an empty COO matrix with the given dimensions.
    pub fn new(rows: usize, cols: usize) -> Self {
        Self { rows, cols, row_indices: Vec::new(), col_indices: Vec::new(), values: Vec::new() }
    }

    /// Create an empty COO matrix with pre-allocated capacity for `cap` entries.
    pub fn with_capacity(rows: usize, cols: usize, cap: usize) -> Self {
        Self {
            rows, cols,
            row_indices: Vec::with_capacity(cap),
            col_indices: Vec::with_capacity(cap),
            values: Vec::with_capacity(cap),
        }
    }

    /// Append a single `(row, col, val)` entry. Duplicates are allowed.
    pub fn push(&mut self, row: usize, col: usize, val: T) {
        self.row_indices.push(row);
        self.col_indices.push(col);
        self.values.push(val);
    }

    /// Number of stored entries (may include duplicates).
    pub fn nnz(&self) -> usize { self.values.len() }

    /// Convert to Compressed Sparse Row format, sorting each row by column index.
    pub fn to_csr(&self) -> CsrMatrix<T> {
        let nnz = self.nnz();
        let mut row_counts = vec![0usize; self.rows + 1];
        for &r in &self.row_indices { row_counts[r + 1] += 1; }
        for i in 1..=self.rows { row_counts[i] += row_counts[i - 1]; }

        let mut col_ind = vec![0usize; nnz];
        let mut vals = vec![T::zero(); nnz];
        let mut offset = row_counts.clone();

        for k in 0..nnz {
            let r = self.row_indices[k];
            let pos = offset[r];
            col_ind[pos] = self.col_indices[k];
            vals[pos] = self.values[k];
            offset[r] += 1;
        }

        // Sort each row by column index
        for i in 0..self.rows {
            let start = row_counts[i];
            let end = row_counts[i + 1];
            let slice = &mut col_ind[start..end];
            let vslice = &mut vals[start..end];
            // Simple insertion sort (rows are typically short)
            for j in 1..slice.len() {
                let mut k = j;
                while k > 0 && slice[k - 1] > slice[k] {
                    slice.swap(k - 1, k);
                    vslice.swap(k - 1, k);
                    k -= 1;
                }
            }
        }

        CsrMatrix { rows: self.rows, cols: self.cols, row_ptr: row_counts, col_ind, values: vals }
    }

    /// Convert to Compressed Sparse Column format via an intermediate CSR.
    pub fn to_csc(&self) -> CscMatrix<T> {
        self.to_csr().to_csc()
    }

    /// Return the transpose (swap row and column indices).
    pub fn transpose(&self) -> Self {
        Self {
            rows: self.cols, cols: self.rows,
            row_indices: self.col_indices.clone(),
            col_indices: self.row_indices.clone(),
            values: self.values.clone(),
        }
    }
}

/// Compressed Sparse Row (CSR) matrix.
///
/// The standard format for row-oriented sparse linear algebra. Stores three
/// arrays: `row_ptr` (length `rows + 1`), `col_ind`, and `values`.
/// Entry `A[i][j]` lives at position `k` where
/// `row_ptr[i] <= k < row_ptr[i+1]` and `col_ind[k] == j`.
///
/// Use [`CooMatrix::to_csr`] for construction from triplets, or
/// [`CsrMatrix::new`] / [`CsrMatrix::identity`] for direct creation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct CsrMatrix<T: Scalar> {
    pub rows: usize,
    pub cols: usize,
    pub row_ptr: Vec<usize>,
    pub col_ind: Vec<usize>,
    pub values: Vec<T>,
}

impl<T: Scalar> CsrMatrix<T> {
    /// Construct a CSR matrix from raw arrays, validating lengths.
    pub fn new(rows: usize, cols: usize, row_ptr: Vec<usize>, col_ind: Vec<usize>, values: Vec<T>) -> error::Result<Self> {
        if row_ptr.len() != rows + 1 {
            return Err(SpectralError::Matrix(MatrixError::InvalidSparseFormat {
                reason: format!("row_ptr length {} != rows+1 {}", row_ptr.len(), rows + 1),
            }));
        }
        if col_ind.len() != values.len() {
            return Err(SpectralError::Matrix(MatrixError::InvalidSparseFormat {
                reason: "col_ind and values length mismatch".into(),
            }));
        }
        Ok(Self { rows, cols, row_ptr, col_ind, values })
    }

    /// Create an all-zeros matrix (empty sparsity pattern).
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self { rows, cols, row_ptr: vec![0; rows + 1], col_ind: Vec::new(), values: Vec::new() }
    }

    /// Create an `n × n` identity matrix.
    pub fn identity(n: usize) -> Self {
        let row_ptr: Vec<usize> = (0..=n).collect();
        let col_ind: Vec<usize> = (0..n).collect();
        let values = vec![T::one(); n];
        Self { rows: n, cols: n, row_ptr, col_ind, values }
    }

    /// Number of stored non-zero entries.
    pub fn nnz(&self) -> usize { self.values.len() }
    /// Matrix dimensions as `(rows, cols)`.
    pub fn shape(&self) -> (usize, usize) { (self.rows, self.cols) }
    /// `true` if the matrix is square.
    pub fn is_square(&self) -> bool { self.rows == self.cols }

    /// Fraction of non-zero entries: `nnz / (rows * cols)`.
    pub fn density(&self) -> f64 {
        let total = self.rows * self.cols;
        if total == 0 { 0.0 } else { self.nnz() as f64 / total as f64 }
    }

    pub fn get(&self, row: usize, col: usize) -> T {
        if row >= self.rows { return T::zero(); }
        let start = self.row_ptr[row];
        let end = self.row_ptr[row + 1];
        for k in start..end {
            if self.col_ind[k] == col { return self.values[k]; }
            if self.col_ind[k] > col { break; }
        }
        T::zero()
    }

    /// Sparse matrix-vector multiply: y = A * x
    pub fn mul_vec(&self, x: &[T]) -> error::Result<Vec<T>> {
        if x.len() != self.cols {
            return Err(SpectralError::Matrix(MatrixError::IncompatibleDimensions {
                operation: "spmv".into(), left_rows: self.rows, left_cols: self.cols,
                right_rows: x.len(), right_cols: 1,
            }));
        }
        let mut y = vec![T::zero(); self.rows];
        for i in 0..self.rows {
            for k in self.row_ptr[i]..self.row_ptr[i + 1] {
                y[i] = y[i] + self.values[k] * x[self.col_ind[k]];
            }
        }
        Ok(y)
    }

    /// Sparse matrix-matrix multiply (CSR * CSR -> CSR).
    pub fn mul_mat(&self, other: &CsrMatrix<T>) -> error::Result<CsrMatrix<T>> {
        if self.cols != other.rows {
            return Err(SpectralError::Matrix(MatrixError::IncompatibleDimensions {
                operation: "spmm".into(), left_rows: self.rows, left_cols: self.cols,
                right_rows: other.rows, right_cols: other.cols,
            }));
        }
        let mut coo = CooMatrix::new(self.rows, other.cols);
        let mut work = vec![T::zero(); other.cols];
        let mut marker = vec![false; other.cols];

        for i in 0..self.rows {
            let mut cols_in_row = Vec::new();
            for ka in self.row_ptr[i]..self.row_ptr[i + 1] {
                let k = self.col_ind[ka];
                let a_ik = self.values[ka];
                for kb in other.row_ptr[k]..other.row_ptr[k + 1] {
                    let j = other.col_ind[kb];
                    if !marker[j] { marker[j] = true; cols_in_row.push(j); }
                    work[j] = work[j] + a_ik * other.values[kb];
                }
            }
            for &j in &cols_in_row {
                if !work[j].is_approx_zero(T::zero_threshold()) {
                    coo.push(i, j, work[j]);
                }
                work[j] = T::zero();
                marker[j] = false;
            }
        }
        Ok(coo.to_csr())
    }

    /// Return the transpose as a new CSR matrix (via COO intermediate).
    pub fn transpose(&self) -> Self {
        self.to_coo().transpose().to_csr()
    }

    /// Convert to COO (triplet) format.
    pub fn to_coo(&self) -> CooMatrix<T> {
        let mut coo = CooMatrix::with_capacity(self.rows, self.cols, self.nnz());
        for i in 0..self.rows {
            for k in self.row_ptr[i]..self.row_ptr[i + 1] {
                coo.push(i, self.col_ind[k], self.values[k]);
            }
        }
        coo
    }

    /// Convert to Compressed Sparse Column format.
    pub fn to_csc(&self) -> CscMatrix<T> {
        let nnz = self.nnz();
        let mut col_counts = vec![0usize; self.cols + 1];
        for &c in &self.col_ind { col_counts[c + 1] += 1; }
        for j in 1..=self.cols { col_counts[j] += col_counts[j - 1]; }

        let mut row_ind = vec![0usize; nnz];
        let mut vals = vec![T::zero(); nnz];
        let mut offset = col_counts.clone();

        for i in 0..self.rows {
            for k in self.row_ptr[i]..self.row_ptr[i + 1] {
                let c = self.col_ind[k];
                let pos = offset[c];
                row_ind[pos] = i;
                vals[pos] = self.values[k];
                offset[c] += 1;
            }
        }
        CscMatrix { rows: self.rows, cols: self.cols, col_ptr: col_counts, row_ind, values: vals }
    }

    /// Element-wise addition: `self + other`. Both matrices must have the same shape.
    pub fn add(&self, other: &CsrMatrix<T>) -> error::Result<CsrMatrix<T>> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(SpectralError::Matrix(MatrixError::IncompatibleDimensions {
                operation: "add".into(), left_rows: self.rows, left_cols: self.cols,
                right_rows: other.rows, right_cols: other.cols,
            }));
        }
        let mut coo = CooMatrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            let mut row_vals = std::collections::BTreeMap::new();
            for k in self.row_ptr[i]..self.row_ptr[i + 1] {
                *row_vals.entry(self.col_ind[k]).or_insert(T::zero()) = *row_vals.entry(self.col_ind[k]).or_insert(T::zero()) + self.values[k];
            }
            for k in other.row_ptr[i]..other.row_ptr[i + 1] {
                *row_vals.entry(other.col_ind[k]).or_insert(T::zero()) = *row_vals.entry(other.col_ind[k]).or_insert(T::zero()) + other.values[k];
            }
            for (j, v) in row_vals {
                if !v.is_approx_zero(T::zero_threshold()) { coo.push(i, j, v); }
            }
        }
        Ok(coo.to_csr())
    }

    /// Scalar multiplication: return `s * self` as a new matrix.
    pub fn scale(&self, s: T) -> Self {
        Self {
            rows: self.rows, cols: self.cols,
            row_ptr: self.row_ptr.clone(), col_ind: self.col_ind.clone(),
            values: self.values.iter().map(|&v| v * s).collect(),
        }
    }

    /// Extract the main diagonal as a vector of length `min(rows, cols)`.
    pub fn diagonal(&self) -> Vec<T> {
        let n = self.rows.min(self.cols);
        (0..n).map(|i| self.get(i, i)).collect()
    }

    /// Trace of the matrix: sum of diagonal entries.
    pub fn trace(&self) -> T {
        self.diagonal().iter().copied().fold(T::zero(), |a, b| a + b)
    }

    /// Frobenius norm: `sqrt(Σ v²)` over all stored values.
    pub fn frobenius_norm(&self) -> T {
        self.values.iter().map(|&v| v * v).fold(T::zero(), |a, b| a + b).sqrt()
    }

    /// Number of non-zeros in the given row.
    pub fn row_nnz(&self, row: usize) -> usize {
        if row >= self.rows { return 0; }
        self.row_ptr[row + 1] - self.row_ptr[row]
    }

    /// Column indices of non-zero entries in the given row.
    pub fn row_indices(&self, row: usize) -> &[usize] {
        if row >= self.rows { return &[]; }
        &self.col_ind[self.row_ptr[row]..self.row_ptr[row + 1]]
    }

    /// Values of non-zero entries in the given row.
    pub fn row_values(&self, row: usize) -> &[T] {
        if row >= self.rows { return &[]; }
        &self.values[self.row_ptr[row]..self.row_ptr[row + 1]]
    }

    /// Check whether the matrix is symmetric within tolerance `tol`.
    pub fn is_symmetric(&self, tol: T) -> bool {
        if !self.is_square() { return false; }
        for i in 0..self.rows {
            for k in self.row_ptr[i]..self.row_ptr[i + 1] {
                let j = self.col_ind[k];
                let v = self.values[k];
                if !v.approx_eq(self.get(j, i), tol) { return false; }
            }
        }
        true
    }

    /// Sum of all values in the given row.
    pub fn row_sum(&self, row: usize) -> T {
        if row >= self.rows { return T::zero(); }
        self.values[self.row_ptr[row]..self.row_ptr[row + 1]].iter().copied().fold(T::zero(), |a, b| a + b)
    }

    /// Extract a submatrix by selecting specific rows and columns.
    pub fn submatrix(&self, row_set: &[usize], col_set: &[usize]) -> CsrMatrix<T> {
        let col_map: std::collections::HashMap<usize, usize> =
            col_set.iter().enumerate().map(|(new, &old)| (old, new)).collect();
        let mut coo = CooMatrix::new(row_set.len(), col_set.len());
        for (new_i, &old_i) in row_set.iter().enumerate() {
            if old_i >= self.rows { continue; }
            for k in self.row_ptr[old_i]..self.row_ptr[old_i + 1] {
                if let Some(&new_j) = col_map.get(&self.col_ind[k]) {
                    coo.push(new_i, new_j, self.values[k]);
                }
            }
        }
        coo.to_csr()
    }

    /// Maximum number of non-zeros across all rows.
    pub fn max_row_nnz(&self) -> usize {
        (0..self.rows).map(|i| self.row_nnz(i)).max().unwrap_or(0)
    }

    /// Average number of non-zeros per row.
    pub fn avg_row_nnz(&self) -> f64 {
        if self.rows == 0 { return 0.0; }
        self.nnz() as f64 / self.rows as f64
    }
}

/// Compressed Sparse Column (CSC) matrix.
///
/// The column-oriented counterpart to [`CsrMatrix`]. Stores `col_ptr`
/// (length `cols + 1`), `row_ind`, and `values`. Efficient for
/// column-wise access and column-oriented SpMV.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct CscMatrix<T: Scalar> {
    pub rows: usize,
    pub cols: usize,
    pub col_ptr: Vec<usize>,
    pub row_ind: Vec<usize>,
    pub values: Vec<T>,
}

impl<T: Scalar> CscMatrix<T> {
    /// Construct a CSC matrix from raw arrays, validating lengths.
    pub fn new(rows: usize, cols: usize, col_ptr: Vec<usize>, row_ind: Vec<usize>, values: Vec<T>) -> error::Result<Self> {
        if col_ptr.len() != cols + 1 {
            return Err(SpectralError::Matrix(MatrixError::InvalidSparseFormat {
                reason: format!("col_ptr length {} != cols+1 {}", col_ptr.len(), cols + 1),
            }));
        }
        Ok(Self { rows, cols, col_ptr, row_ind, values })
    }

    /// Number of stored non-zero entries.
    pub fn nnz(&self) -> usize { self.values.len() }
    /// Matrix dimensions as `(rows, cols)`.
    pub fn shape(&self) -> (usize, usize) { (self.rows, self.cols) }

    /// Retrieve `A[row][col]`, returning `T::zero()` for missing entries.
    pub fn get(&self, row: usize, col: usize) -> T {
        if col >= self.cols { return T::zero(); }
        for k in self.col_ptr[col]..self.col_ptr[col + 1] {
            if self.row_ind[k] == row { return self.values[k]; }
            if self.row_ind[k] > row { break; }
        }
        T::zero()
    }

    /// Convert to CSR format.
    pub fn to_csr(&self) -> CsrMatrix<T> {
        let mut coo = CooMatrix::with_capacity(self.rows, self.cols, self.nnz());
        for j in 0..self.cols {
            for k in self.col_ptr[j]..self.col_ptr[j + 1] {
                coo.push(self.row_ind[k], j, self.values[k]);
            }
        }
        coo.to_csr()
    }

    /// Convert to COO (triplet) format.
    pub fn to_coo(&self) -> CooMatrix<T> {
        let mut coo = CooMatrix::with_capacity(self.rows, self.cols, self.nnz());
        for j in 0..self.cols {
            for k in self.col_ptr[j]..self.col_ptr[j + 1] {
                coo.push(self.row_ind[k], j, self.values[k]);
            }
        }
        coo
    }

    /// Return the transpose as a new CSC matrix.
    pub fn transpose(&self) -> Self {
        self.to_csr().to_csc()
    }

    /// Number of non-zeros in the given column.
    pub fn col_nnz(&self, col: usize) -> usize {
        if col >= self.cols { return 0; }
        self.col_ptr[col + 1] - self.col_ptr[col]
    }

    /// Row indices of non-zero entries in the given column.
    pub fn col_indices(&self, col: usize) -> &[usize] {
        if col >= self.cols { return &[]; }
        &self.row_ind[self.col_ptr[col]..self.col_ptr[col + 1]]
    }

    /// Sparse matrix-vector multiply: `y = A * x` (column-oriented accumulation).
    pub fn mul_vec(&self, x: &[T]) -> error::Result<Vec<T>> {
        if x.len() != self.cols {
            return Err(SpectralError::Matrix(MatrixError::IncompatibleDimensions {
                operation: "csc_spmv".into(), left_rows: self.rows, left_cols: self.cols,
                right_rows: x.len(), right_cols: 1,
            }));
        }
        let mut y = vec![T::zero(); self.rows];
        for j in 0..self.cols {
            let xj = x[j];
            if xj.is_approx_zero(T::zero_threshold()) { continue; }
            for k in self.col_ptr[j]..self.col_ptr[j + 1] {
                y[self.row_ind[k]] = y[self.row_ind[k]] + self.values[k] * xj;
            }
        }
        Ok(y)
    }

    /// Extract the main diagonal as a vector of length `min(rows, cols)`.
    pub fn diagonal(&self) -> Vec<T> {
        let n = self.rows.min(self.cols);
        (0..n).map(|i| self.get(i, i)).collect()
    }
}

/// Builder for constructing sparse matrices from triplets incrementally.
///
/// Unlike [`CooMatrix`], `SparseTriple` sums duplicate entries on conversion
/// to CSR/CSC, making it convenient for finite-element-style assembly.
#[derive(Debug, Clone)]
pub struct SparseTriple<T: Scalar> {
    rows: usize,
    cols: usize,
    entries: Vec<(usize, usize, T)>,
}

impl<T: Scalar> SparseTriple<T> {
    /// Create an empty triplet builder for the given dimensions.
    pub fn new(rows: usize, cols: usize) -> Self {
        Self { rows, cols, entries: Vec::new() }
    }

    /// Add a triplet `(row, col, val)`. Duplicates are summed on conversion.
    pub fn add(&mut self, row: usize, col: usize, val: T) {
        self.entries.push((row, col, val));
    }

    /// Number of stored triplets (before duplicate summation).
    pub fn nnz(&self) -> usize { self.entries.len() }

    /// Convert to CSR, summing any duplicate `(row, col)` entries.
    pub fn to_csr(&self) -> CsrMatrix<T> {
        let mut coo = CooMatrix::new(self.rows, self.cols);
        // Sum duplicates
        let mut map: std::collections::HashMap<(usize, usize), T> = std::collections::HashMap::new();
        for &(r, c, v) in &self.entries {
            *map.entry((r, c)).or_insert(T::zero()) = map.get(&(r, c)).copied().unwrap_or(T::zero()) + v;
        }
        for (&(r, c), &v) in &map {
            if !v.is_approx_zero(T::zero_threshold()) { coo.push(r, c, v); }
        }
        coo.to_csr()
    }

    /// Convert to CSC via an intermediate CSR.
    pub fn to_csc(&self) -> CscMatrix<T> {
        self.to_csr().to_csc()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_csr() -> CsrMatrix<f64> {
        // 3x3: [[1,0,2],[0,3,0],[4,0,5]]
        CsrMatrix {
            rows: 3, cols: 3,
            row_ptr: vec![0, 2, 3, 5],
            col_ind: vec![0, 2, 1, 0, 2],
            values: vec![1.0, 2.0, 3.0, 4.0, 5.0],
        }
    }

    #[test] fn test_csr_get() {
        let m = sample_csr();
        assert_eq!(m.get(0, 0), 1.0); assert_eq!(m.get(0, 1), 0.0); assert_eq!(m.get(2, 2), 5.0);
    }

    #[test] fn test_csr_spmv() {
        let m = sample_csr();
        let y = m.mul_vec(&[1.0, 1.0, 1.0]).unwrap();
        assert!((y[0] - 3.0).abs() < 1e-10); assert!((y[1] - 3.0).abs() < 1e-10); assert!((y[2] - 9.0).abs() < 1e-10);
    }

    #[test] fn test_csr_transpose() {
        let m = sample_csr();
        let mt = m.transpose();
        assert_eq!(mt.get(0, 0), 1.0); assert_eq!(mt.get(2, 0), 2.0); assert_eq!(mt.get(0, 2), 4.0);
    }

    #[test] fn test_csr_to_csc() {
        let m = sample_csr();
        let csc = m.to_csc();
        assert_eq!(csc.get(0, 0), 1.0); assert_eq!(csc.get(2, 2), 5.0); assert_eq!(csc.nnz(), 5);
    }

    #[test] fn test_coo_roundtrip() {
        let m = sample_csr();
        let coo = m.to_coo();
        let m2 = coo.to_csr();
        assert_eq!(m2.get(0, 0), 1.0); assert_eq!(m2.get(2, 2), 5.0);
    }

    #[test] fn test_identity() {
        let m: CsrMatrix<f64> = CsrMatrix::identity(4);
        assert_eq!(m.nnz(), 4); assert_eq!(m.get(2, 2), 1.0); assert_eq!(m.get(0, 1), 0.0);
    }

    #[test] fn test_csr_add() {
        let m = sample_csr();
        let sum = m.add(&m).unwrap();
        assert!((sum.get(0, 0) - 2.0).abs() < 1e-10);
    }

    #[test] fn test_csr_scale() {
        let m = sample_csr().scale(2.0_f64);
        assert!((m.get(0, 0) - 2.0).abs() < 1e-10);
    }

    #[test] fn test_diagonal_trace() {
        let m = sample_csr();
        let d = m.diagonal();
        assert_eq!(d, vec![1.0, 3.0, 5.0]);
        assert!((m.trace() - 9.0).abs() < 1e-10);
    }

    #[test] fn test_frobenius() {
        let m = sample_csr();
        let expected = (1.0_f64 + 4.0 + 9.0 + 16.0 + 25.0).sqrt();
        assert!((m.frobenius_norm() - expected).abs() < 1e-10);
    }

    #[test] fn test_symmetric() {
        let sym = CsrMatrix {
            rows: 2, cols: 2,
            row_ptr: vec![0, 2, 4],
            col_ind: vec![0, 1, 0, 1],
            values: vec![1.0_f64, 2.0, 2.0, 3.0],
        };
        assert!(sym.is_symmetric(1e-10));
        assert!(!sample_csr().is_symmetric(1e-10));
    }

    #[test] fn test_spmm() {
        let m: CsrMatrix<f64> = CsrMatrix::identity(3);
        let b = sample_csr();
        let c = m.mul_mat(&b).unwrap();
        assert_eq!(c.get(0, 0), 1.0); assert_eq!(c.get(2, 2), 5.0);
    }

    #[test] fn test_submatrix() {
        let m = sample_csr();
        let sub = m.submatrix(&[0, 2], &[0, 2]);
        assert_eq!(sub.shape(), (2, 2));
        assert_eq!(sub.get(0, 0), 1.0); assert_eq!(sub.get(1, 1), 5.0);
    }

    #[test] fn test_sparse_triple() {
        let mut st = SparseTriple::new(2, 2);
        st.add(0, 0, 1.0_f64); st.add(0, 0, 2.0); st.add(1, 1, 5.0);
        let m = st.to_csr();
        assert!((m.get(0, 0) - 3.0).abs() < 1e-10);
        assert!((m.get(1, 1) - 5.0).abs() < 1e-10);
    }

    #[test] fn test_csc_spmv() {
        let csc = sample_csr().to_csc();
        let y = csc.mul_vec(&[1.0, 1.0, 1.0]).unwrap();
        assert!((y[0] - 3.0).abs() < 1e-10);
    }

    #[test] fn test_row_stats() {
        let m = sample_csr();
        assert_eq!(m.row_nnz(0), 2); assert_eq!(m.max_row_nnz(), 2);
        assert!((m.avg_row_nnz() - 5.0 / 3.0).abs() < 1e-10);
    }

    #[test] fn test_density() {
        let m = sample_csr();
        assert!((m.density() - 5.0 / 9.0).abs() < 1e-10);
    }

    #[test] fn test_row_sum() {
        let m = sample_csr();
        assert!((m.row_sum(0) - 3.0).abs() < 1e-10);
        assert!((m.row_sum(2) - 9.0).abs() < 1e-10);
    }
}
