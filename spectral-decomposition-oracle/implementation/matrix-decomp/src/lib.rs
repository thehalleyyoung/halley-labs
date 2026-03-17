//! Matrix decomposition algorithms for the spectral decomposition oracle.
//!
//! Provides LU, QR, Cholesky, SVD, and eigenvalue decompositions with both
//! dense and sparse matrix support. Includes Lanczos, LOBPCG, randomized
//! methods, and supporting utilities (Householder, Givens, tridiagonal solvers,
//! preconditioners).
//!
//! # Quick start
//! ```ignore
//! use matrix_decomp::{DenseMatrix, lu::LuDecomposition};
//! let a = DenseMatrix::from_row_major(3, 3, vec![
//!     2.0, 1.0, 1.0,
//!     4.0, 3.0, 3.0,
//!     8.0, 7.0, 9.0,
//! ]);
//! let lu = LuDecomposition::factorize(&a).unwrap();
//! let det = lu.determinant();
//! ```

pub mod error;
pub mod householder;
pub mod givens;
pub mod tridiagonal;
pub mod lu;
pub mod qr;
pub mod cholesky;
pub mod svd;
pub mod eigen;
pub mod lanczos;
pub mod lobpcg;
pub mod random_projection;
pub mod preconditioner;

pub use error::{DecompError, DecompResult};

use serde::{Deserialize, Serialize};
use std::fmt;

// ═══════════════════════════════════════════════════════════════════════════
// Dense matrix (row-major)
// ═══════════════════════════════════════════════════════════════════════════

/// Dense matrix stored in row-major order.
///
/// The primary dense matrix type for the `matrix-decomp` crate. Provides
/// element access, basic arithmetic (matrix-vector, matrix-matrix multiply),
/// factorisation entry points, and utility methods (transpose, norms, submatrix
/// extraction).
#[derive(Clone, Serialize, Deserialize)]
pub struct DenseMatrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl fmt::Debug for DenseMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DenseMatrix({}x{}, nnz={})", self.rows, self.cols, self.nnz())
    }
}

impl DenseMatrix {
    /// Create from row-major data.
    pub fn from_row_major(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), rows * cols, "data length must equal rows*cols");
        Self { rows, cols, data }
    }

    /// Create a zero matrix.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    /// Create an identity matrix.
    pub fn eye(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m.data[i * n + i] = 1.0;
        }
        m
    }

    /// Create a diagonal matrix from a vector.
    pub fn from_diag(diag: &[f64]) -> Self {
        let n = diag.len();
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m.data[i * n + i] = diag[i];
        }
        m
    }

    /// Create from a function f(row, col) -> value.
    pub fn from_fn(rows: usize, cols: usize, f: impl Fn(usize, usize) -> f64) -> Self {
        let mut data = Vec::with_capacity(rows * cols);
        for i in 0..rows {
            for j in 0..cols {
                data.push(f(i, j));
            }
        }
        Self { rows, cols, data }
    }

    /// Element access at `(r, c)` without bounds checking.
    #[inline]
    pub fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }

    /// Set element at `(r, c)` without bounds checking.
    #[inline]
    pub fn set(&mut self, r: usize, c: usize, v: f64) {
        self.data[r * self.cols + c] = v;
    }

    /// Element access with bounds checking — returns an error on out-of-bounds.
    #[inline]
    pub fn get_checked(&self, r: usize, c: usize) -> DecompResult<f64> {
        if r >= self.rows || c >= self.cols {
            Err(DecompError::IndexOutOfBounds {
                row: r,
                col: c,
                rows: self.rows,
                cols: self.cols,
            })
        } else {
            Ok(self.data[r * self.cols + c])
        }
    }

    /// Matrix dimensions as `(rows, cols)`.
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// `true` if the matrix is square.
    pub fn is_square(&self) -> bool {
        self.rows == self.cols
    }

    /// `true` if either dimension is zero.
    pub fn is_empty(&self) -> bool {
        self.rows == 0 || self.cols == 0
    }

    /// Number of non-zero elements.
    pub fn nnz(&self) -> usize {
        self.data.iter().filter(|&&v| v != 0.0).count()
    }

    /// Extract column j as a new vector.
    pub fn col(&self, j: usize) -> Vec<f64> {
        (0..self.rows).map(|i| self.get(i, j)).collect()
    }

    /// Extract row i as a slice.
    pub fn row(&self, i: usize) -> &[f64] {
        &self.data[i * self.cols..(i + 1) * self.cols]
    }

    /// Extract row i as a mutable slice.
    pub fn row_mut(&mut self, i: usize) -> &mut [f64] {
        let s = i * self.cols;
        &mut self.data[s..s + self.cols]
    }

    /// Swap rows i and j.
    pub fn swap_rows(&mut self, i: usize, j: usize) {
        if i == j {
            return;
        }
        let c = self.cols;
        for k in 0..c {
            self.data.swap(i * c + k, j * c + k);
        }
    }

    /// Transpose (returns a new matrix).
    pub fn transpose(&self) -> Self {
        let mut t = Self::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                t.set(j, i, self.get(i, j));
            }
        }
        t
    }

    /// Matrix-vector multiply: y = A * x.
    pub fn mul_vec(&self, x: &[f64]) -> DecompResult<Vec<f64>> {
        DecompError::check_vector_len(self.cols, x.len())?;
        let mut y = vec![0.0; self.rows];
        for i in 0..self.rows {
            let row_start = i * self.cols;
            let mut sum = 0.0;
            for j in 0..self.cols {
                sum += self.data[row_start + j] * x[j];
            }
            y[i] = sum;
        }
        Ok(y)
    }

    /// Matrix-matrix multiply: C = A * B.
    pub fn mul_mat(&self, b: &DenseMatrix) -> DecompResult<DenseMatrix> {
        DecompError::check_mul_dims(self.rows, self.cols, b.rows, b.cols)?;
        let mut c = DenseMatrix::zeros(self.rows, b.cols);
        for i in 0..self.rows {
            for k in 0..self.cols {
                let a_ik = self.get(i, k);
                if a_ik == 0.0 {
                    continue;
                }
                for j in 0..b.cols {
                    c.data[i * b.cols + j] += a_ik * b.data[k * b.cols + j];
                }
            }
        }
        Ok(c)
    }

    /// Frobenius norm.
    pub fn frobenius_norm(&self) -> f64 {
        self.data.iter().map(|&v| v * v).sum::<f64>().sqrt()
    }

    /// Trace (sum of diagonal elements).
    pub fn trace(&self) -> f64 {
        let n = self.rows.min(self.cols);
        (0..n).map(|i| self.get(i, i)).sum()
    }

    /// Extract a sub-matrix [r0..r1, c0..c1).
    pub fn submatrix(&self, r0: usize, r1: usize, c0: usize, c1: usize) -> Self {
        let nr = r1 - r0;
        let nc = c1 - c0;
        let mut out = Self::zeros(nr, nc);
        for i in 0..nr {
            for j in 0..nc {
                out.set(i, j, self.get(r0 + i, c0 + j));
            }
        }
        out
    }

    /// Set a sub-matrix at (r0, c0).
    pub fn set_submatrix(&mut self, r0: usize, c0: usize, src: &DenseMatrix) {
        for i in 0..src.rows {
            for j in 0..src.cols {
                self.set(r0 + i, c0 + j, src.get(i, j));
            }
        }
    }

    /// Scale all elements by a factor.
    pub fn scale(&mut self, factor: f64) {
        for v in &mut self.data {
            *v *= factor;
        }
    }

    /// Add another matrix (in-place).
    pub fn add_inplace(&mut self, other: &DenseMatrix) {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        for (a, &b) in self.data.iter_mut().zip(other.data.iter()) {
            *a += b;
        }
    }

    /// Check if symmetric within tolerance.
    pub fn is_symmetric(&self, tol: f64) -> bool {
        if !self.is_square() {
            return false;
        }
        let n = self.rows;
        for i in 0..n {
            for j in (i + 1)..n {
                if (self.get(i, j) - self.get(j, i)).abs() > tol {
                    return false;
                }
            }
        }
        true
    }

    /// Max absolute element.
    pub fn max_abs(&self) -> f64 {
        self.data
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max)
    }

    /// Infinity norm (max row sum of absolute values).
    pub fn inf_norm(&self) -> f64 {
        let mut mx = 0.0_f64;
        for i in 0..self.rows {
            let s: f64 = (0..self.cols).map(|j| self.get(i, j).abs()).sum();
            mx = mx.max(s);
        }
        mx
    }

    /// One-norm (max column sum of absolute values).
    pub fn one_norm(&self) -> f64 {
        let mut mx = 0.0_f64;
        for j in 0..self.cols {
            let s: f64 = (0..self.rows).map(|i| self.get(i, j).abs()).sum();
            mx = mx.max(s);
        }
        mx
    }

    /// Convert to CSR sparse format.
    pub fn to_csr(&self) -> CsrMatrix {
        let mut row_ptr = vec![0usize; self.rows + 1];
        let mut col_idx = Vec::new();
        let mut values = Vec::new();
        for i in 0..self.rows {
            for j in 0..self.cols {
                let v = self.get(i, j);
                if v != 0.0 {
                    col_idx.push(j);
                    values.push(v);
                }
            }
            row_ptr[i + 1] = col_idx.len();
        }
        CsrMatrix {
            rows: self.rows,
            cols: self.cols,
            row_ptr,
            col_idx,
            values,
        }
    }

    /// Diagonal vector.
    pub fn diag(&self) -> Vec<f64> {
        let n = self.rows.min(self.cols);
        (0..n).map(|i| self.get(i, i)).collect()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Dense vector
// ═══════════════════════════════════════════════════════════════════════════

/// Dense vector.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DenseVector {
    pub data: Vec<f64>,
}

impl DenseVector {
    pub fn new(data: Vec<f64>) -> Self {
        Self { data }
    }

    pub fn zeros(n: usize) -> Self {
        Self {
            data: vec![0.0; n],
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn dot(&self, other: &[f64]) -> f64 {
        self.data
            .iter()
            .zip(other.iter())
            .map(|(&a, &b)| a * b)
            .sum()
    }

    pub fn norm2(&self) -> f64 {
        self.data.iter().map(|&v| v * v).sum::<f64>().sqrt()
    }

    pub fn scale(&mut self, factor: f64) {
        for v in &mut self.data {
            *v *= factor;
        }
    }

    pub fn axpy(&mut self, alpha: f64, x: &[f64]) {
        for (a, &b) in self.data.iter_mut().zip(x.iter()) {
            *a += alpha * b;
        }
    }

    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        &mut self.data
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CSR Sparse matrix
// ═══════════════════════════════════════════════════════════════════════════

/// Compressed Sparse Row matrix.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CsrMatrix {
    pub rows: usize,
    pub cols: usize,
    pub row_ptr: Vec<usize>,
    pub col_idx: Vec<usize>,
    pub values: Vec<f64>,
}

impl CsrMatrix {
    /// Create an empty CSR matrix.
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            row_ptr: vec![0; rows + 1],
            col_idx: Vec::new(),
            values: Vec::new(),
        }
    }

    /// Build from COO-style triplets (row, col, value). Duplicates are summed.
    pub fn from_triplets(
        rows: usize,
        cols: usize,
        triplets: &[(usize, usize, f64)],
    ) -> Self {
        let mut sorted = triplets.to_vec();
        sorted.sort_by_key(|&(r, c, _)| (r, c));

        let mut row_ptr = vec![0usize; rows + 1];
        let mut col_idx = Vec::with_capacity(sorted.len());
        let mut values = Vec::with_capacity(sorted.len());

        for &(r, c, v) in &sorted {
            if !col_idx.is_empty() && col_idx.last() == Some(&c) && {
                let last_row_start = row_ptr[r];
                let current_nnz = col_idx.len();
                current_nnz > last_row_start
            } {
                // Check if this is a duplicate in the same row
                let last_idx = col_idx.len() - 1;
                if col_idx[last_idx] == c {
                    values[last_idx] += v;
                    continue;
                }
            }
            col_idx.push(c);
            values.push(v);
            row_ptr[r + 1] = col_idx.len();
        }

        // Fix row_ptr to be monotonically increasing
        for i in 1..=rows {
            if row_ptr[i] < row_ptr[i - 1] {
                row_ptr[i] = row_ptr[i - 1];
            }
        }
        // Ensure last element is correct
        row_ptr[rows] = col_idx.len();

        Self {
            rows,
            cols,
            row_ptr,
            col_idx,
            values,
        }
    }

    /// Number of non-zeros.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    pub fn is_square(&self) -> bool {
        self.rows == self.cols
    }

    /// Get element (r, c). Returns 0.0 for structural zeros.
    pub fn get(&self, r: usize, c: usize) -> f64 {
        let start = self.row_ptr[r];
        let end = self.row_ptr[r + 1];
        for idx in start..end {
            if self.col_idx[idx] == c {
                return self.values[idx];
            }
        }
        0.0
    }

    /// Matrix-vector multiply y = A * x.
    pub fn mul_vec(&self, x: &[f64]) -> DecompResult<Vec<f64>> {
        DecompError::check_vector_len(self.cols, x.len())?;
        let mut y = vec![0.0; self.rows];
        for i in 0..self.rows {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];
            let mut sum = 0.0;
            for idx in start..end {
                sum += self.values[idx] * x[self.col_idx[idx]];
            }
            y[i] = sum;
        }
        Ok(y)
    }

    /// Convert to dense matrix.
    pub fn to_dense(&self) -> DenseMatrix {
        let mut m = DenseMatrix::zeros(self.rows, self.cols);
        for i in 0..self.rows {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];
            for idx in start..end {
                m.set(i, self.col_idx[idx], self.values[idx]);
            }
        }
        m
    }

    /// Diagonal elements.
    pub fn diag(&self) -> Vec<f64> {
        let n = self.rows.min(self.cols);
        (0..n).map(|i| self.get(i, i)).collect()
    }

    /// Transpose to new CSR.
    pub fn transpose(&self) -> Self {
        let mut triplets = Vec::with_capacity(self.nnz());
        for i in 0..self.rows {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];
            for idx in start..end {
                triplets.push((self.col_idx[idx], i, self.values[idx]));
            }
        }
        Self::from_triplets(self.cols, self.rows, &triplets)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Free-standing linear algebra helpers used throughout
// ═══════════════════════════════════════════════════════════════════════════

/// Dot product of two slices.
#[inline]
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// L2 norm of a slice.
#[inline]
pub fn norm2(v: &[f64]) -> f64 {
    v.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

/// Scale a slice in-place.
#[inline]
pub fn scale_vec(v: &mut [f64], s: f64) {
    for x in v.iter_mut() {
        *x *= s;
    }
}

/// axpy: y += alpha * x.
#[inline]
pub fn axpy(alpha: f64, x: &[f64], y: &mut [f64]) {
    for (yi, &xi) in y.iter_mut().zip(x.iter()) {
        *yi += alpha * xi;
    }
}

/// Normalize a vector in-place, returning the original norm.
pub fn normalize(v: &mut [f64]) -> f64 {
    let n = norm2(v);
    if n > 1e-300 {
        let inv = 1.0 / n;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
    n
}

/// Copy src into dst.
#[inline]
pub fn copy_vec(dst: &mut [f64], src: &[f64]) {
    dst.copy_from_slice(src);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dense_matrix_zeros() {
        let m = DenseMatrix::zeros(3, 4);
        assert_eq!(m.rows, 3);
        assert_eq!(m.cols, 4);
        assert!(m.data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_dense_matrix_eye() {
        let m = DenseMatrix::eye(3);
        assert_eq!(m.get(0, 0), 1.0);
        assert_eq!(m.get(0, 1), 0.0);
        assert_eq!(m.get(1, 1), 1.0);
        assert_eq!(m.trace(), 3.0);
    }

    #[test]
    fn test_dense_matrix_mul_vec() {
        let m = DenseMatrix::from_row_major(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let x = vec![1.0, 1.0, 1.0];
        let y = m.mul_vec(&x).unwrap();
        assert!((y[0] - 6.0).abs() < 1e-10);
        assert!((y[1] - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_dense_matrix_mul_mat() {
        let a = DenseMatrix::eye(3);
        let b = DenseMatrix::from_row_major(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let c = a.mul_mat(&b).unwrap();
        assert_eq!(c.rows, 3);
        assert_eq!(c.cols, 2);
        assert!((c.get(0, 0) - 1.0).abs() < 1e-10);
        assert!((c.get(2, 1) - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_dense_matrix_transpose() {
        let m = DenseMatrix::from_row_major(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let t = m.transpose();
        assert_eq!(t.rows, 3);
        assert_eq!(t.cols, 2);
        assert_eq!(t.get(0, 0), 1.0);
        assert_eq!(t.get(2, 1), 6.0);
    }

    #[test]
    fn test_dense_matrix_symmetric() {
        let m = DenseMatrix::from_row_major(2, 2, vec![1.0, 2.0, 2.0, 3.0]);
        assert!(m.is_symmetric(1e-10));
        let m2 = DenseMatrix::from_row_major(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        assert!(!m2.is_symmetric(1e-10));
    }

    #[test]
    fn test_csr_matrix_basic() {
        let triplets = vec![(0, 0, 1.0), (0, 2, 2.0), (1, 1, 3.0), (2, 0, 4.0), (2, 2, 5.0)];
        let csr = CsrMatrix::from_triplets(3, 3, &triplets);
        assert_eq!(csr.nnz(), 5);
        assert!((csr.get(0, 0) - 1.0).abs() < 1e-10);
        assert!((csr.get(0, 1) - 0.0).abs() < 1e-10);
        assert!((csr.get(1, 1) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_csr_mul_vec() {
        let triplets = vec![(0, 0, 2.0), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 3.0)];
        let csr = CsrMatrix::from_triplets(2, 2, &triplets);
        let y = csr.mul_vec(&[1.0, 2.0]).unwrap();
        assert!((y[0] - 4.0).abs() < 1e-10);
        assert!((y[1] - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_csr_to_dense() {
        let triplets = vec![(0, 0, 1.0), (1, 1, 2.0)];
        let csr = CsrMatrix::from_triplets(2, 2, &triplets);
        let d = csr.to_dense();
        assert_eq!(d.get(0, 0), 1.0);
        assert_eq!(d.get(1, 1), 2.0);
        assert_eq!(d.get(0, 1), 0.0);
    }

    #[test]
    fn test_dense_vector_ops() {
        let v = DenseVector::new(vec![3.0, 4.0]);
        assert!((v.norm2() - 5.0).abs() < 1e-10);
        assert!((v.dot(&[1.0, 0.0]) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_free_standing_helpers() {
        assert!((dot(&[1.0, 2.0], &[3.0, 4.0]) - 11.0).abs() < 1e-10);
        assert!((norm2(&[3.0, 4.0]) - 5.0).abs() < 1e-10);
        let mut v = vec![3.0, 4.0];
        let n = normalize(&mut v);
        assert!((n - 5.0).abs() < 1e-10);
        assert!((norm2(&v) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_dense_to_csr_roundtrip() {
        let m = DenseMatrix::from_row_major(2, 2, vec![1.0, 0.0, 0.0, 2.0]);
        let csr = m.to_csr();
        let m2 = csr.to_dense();
        assert!((m2.get(0, 0) - 1.0).abs() < 1e-10);
        assert_eq!(m2.get(0, 1), 0.0);
        assert!((m2.get(1, 1) - 2.0).abs() < 1e-10);
    }
}
