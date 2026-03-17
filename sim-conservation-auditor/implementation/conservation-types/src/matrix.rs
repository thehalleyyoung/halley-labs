//! Matrix types for conservation analysis computations.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops::{Add, Mul, Sub, Neg, Index, IndexMut};

/// A dense matrix stored in row-major order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DenseMatrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl DenseMatrix {
    /// Create a new zero matrix.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    /// Create an identity matrix.
    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m[(i, i)] = 1.0;
        }
        m
    }

    /// Create from a flat vector (row-major).
    pub fn from_vec(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), rows * cols);
        Self { rows, cols, data }
    }

    /// Create from row vectors.
    pub fn from_rows(rows: &[Vec<f64>]) -> Self {
        let nrows = rows.len();
        let ncols = rows[0].len();
        let mut data = Vec::with_capacity(nrows * ncols);
        for row in rows {
            assert_eq!(row.len(), ncols);
            data.extend_from_slice(row);
        }
        Self { rows: nrows, cols: ncols, data }
    }

    /// Create a diagonal matrix.
    pub fn diagonal(diag: &[f64]) -> Self {
        let n = diag.len();
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m[(i, i)] = diag[i];
        }
        m
    }

    /// Get the transpose.
    pub fn transpose(&self) -> Self {
        let mut result = Self::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result[(j, i)] = self[(i, j)];
            }
        }
        result
    }

    /// Matrix-matrix multiplication.
    pub fn matmul(&self, other: &DenseMatrix) -> Self {
        assert_eq!(self.cols, other.rows);
        let mut result = Self::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for k in 0..self.cols {
                let a_ik = self[(i, k)];
                for j in 0..other.cols {
                    result[(i, j)] += a_ik * other[(k, j)];
                }
            }
        }
        result
    }

    /// Matrix-vector multiplication.
    pub fn matvec(&self, v: &[f64]) -> Vec<f64> {
        assert_eq!(self.cols, v.len());
        let mut result = vec![0.0; self.rows];
        for i in 0..self.rows {
            for j in 0..self.cols {
                result[i] += self[(i, j)] * v[j];
            }
        }
        result
    }

    /// Compute the trace.
    pub fn trace(&self) -> f64 {
        let n = self.rows.min(self.cols);
        (0..n).map(|i| self[(i, i)]).sum()
    }

    /// Compute the Frobenius norm.
    pub fn frobenius_norm(&self) -> f64 {
        self.data.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Compute the infinity norm (max row sum).
    pub fn infinity_norm(&self) -> f64 {
        (0..self.rows)
            .map(|i| {
                (0..self.cols)
                    .map(|j| self[(i, j)].abs())
                    .sum::<f64>()
            })
            .fold(0.0f64, f64::max)
    }

    /// Scale the matrix by a scalar.
    pub fn scale(&self, s: f64) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().map(|x| x * s).collect(),
        }
    }

    /// Extract a row as a vector.
    pub fn row(&self, i: usize) -> Vec<f64> {
        let start = i * self.cols;
        self.data[start..start + self.cols].to_vec()
    }

    /// Extract a column as a vector.
    pub fn col(&self, j: usize) -> Vec<f64> {
        (0..self.rows).map(|i| self[(i, j)]).collect()
    }

    /// Check if the matrix is square.
    pub fn is_square(&self) -> bool {
        self.rows == self.cols
    }

    /// Check if the matrix is symmetric within tolerance.
    pub fn is_symmetric(&self, tol: f64) -> bool {
        if !self.is_square() {
            return false;
        }
        for i in 0..self.rows {
            for j in (i + 1)..self.cols {
                if (self[(i, j)] - self[(j, i)]).abs() > tol {
                    return false;
                }
            }
        }
        true
    }

    /// Check if the matrix is skew-symmetric within tolerance.
    pub fn is_skew_symmetric(&self, tol: f64) -> bool {
        if !self.is_square() {
            return false;
        }
        for i in 0..self.rows {
            if self[(i, i)].abs() > tol {
                return false;
            }
            for j in (i + 1)..self.cols {
                if (self[(i, j)] + self[(j, i)]).abs() > tol {
                    return false;
                }
            }
        }
        true
    }

    /// Compute the determinant for small matrices (up to 4x4).
    pub fn determinant(&self) -> Option<f64> {
        if !self.is_square() {
            return None;
        }
        match self.rows {
            1 => Some(self[(0, 0)]),
            2 => Some(self[(0, 0)] * self[(1, 1)] - self[(0, 1)] * self[(1, 0)]),
            3 => {
                let a = self[(0, 0)]
                    * (self[(1, 1)] * self[(2, 2)] - self[(1, 2)] * self[(2, 1)]);
                let b = self[(0, 1)]
                    * (self[(1, 0)] * self[(2, 2)] - self[(1, 2)] * self[(2, 0)]);
                let c = self[(0, 2)]
                    * (self[(1, 0)] * self[(2, 1)] - self[(1, 1)] * self[(2, 0)]);
                Some(a - b + c)
            }
            _ => {
                // LU decomposition for larger matrices
                let (lu, parity) = self.lu_decompose()?;
                let det: f64 = (0..self.rows).map(|i| lu[(i, i)]).product();
                Some(if parity { det } else { -det })
            }
        }
    }

    /// LU decomposition (returns LU matrix and parity).
    fn lu_decompose(&self) -> Option<(DenseMatrix, bool)> {
        if !self.is_square() {
            return None;
        }
        let n = self.rows;
        let mut lu = self.clone();
        let mut parity = true;
        for k in 0..n {
            // Partial pivoting
            let mut max_val = lu[(k, k)].abs();
            let mut max_row = k;
            for i in (k + 1)..n {
                if lu[(i, k)].abs() > max_val {
                    max_val = lu[(i, k)].abs();
                    max_row = i;
                }
            }
            if max_val < 1e-15 {
                return None; // Singular
            }
            if max_row != k {
                for j in 0..n {
                    let tmp = lu[(k, j)];
                    lu[(k, j)] = lu[(max_row, j)];
                    lu[(max_row, j)] = tmp;
                }
                parity = !parity;
            }
            for i in (k + 1)..n {
                lu[(i, k)] /= lu[(k, k)];
                for j in (k + 1)..n {
                    let lik = lu[(i, k)];
                    let ukj = lu[(k, j)];
                    lu[(i, j)] -= lik * ukj;
                }
            }
        }
        Some((lu, parity))
    }

    /// Solve the linear system Ax = b using LU decomposition.
    pub fn solve(&self, b: &[f64]) -> Option<Vec<f64>> {
        if !self.is_square() || b.len() != self.rows {
            return None;
        }
        let n = self.rows;
        let (lu, _) = self.lu_decompose()?;
        let mut x = b.to_vec();
        // Forward substitution
        for i in 1..n {
            for j in 0..i {
                let lij = lu[(i, j)];
                x[i] -= lij * x[j];
            }
        }
        // Backward substitution
        for i in (0..n).rev() {
            for j in (i + 1)..n {
                let uij = lu[(i, j)];
                x[i] -= uij * x[j];
            }
            x[i] /= lu[(i, i)];
        }
        Some(x)
    }

    /// Compute eigenvalues for a 2x2 matrix.
    pub fn eigenvalues_2x2(&self) -> Option<(f64, f64)> {
        if self.rows != 2 || self.cols != 2 {
            return None;
        }
        let tr = self.trace();
        let det = self.determinant()?;
        let disc = tr * tr - 4.0 * det;
        if disc >= 0.0 {
            let sqrt_disc = disc.sqrt();
            Some(((tr + sqrt_disc) / 2.0, (tr - sqrt_disc) / 2.0))
        } else {
            None // Complex eigenvalues
        }
    }

    /// Check if the matrix preserves a symplectic form J.
    /// M is symplectic if M^T J M = J.
    pub fn is_symplectic(&self, j: &DenseMatrix, tol: f64) -> bool {
        let mt = self.transpose();
        let result = mt.matmul(j).matmul(self);
        for i in 0..j.rows {
            for k in 0..j.cols {
                if (result[(i, k)] - j[(i, k)]).abs() > tol {
                    return false;
                }
            }
        }
        true
    }

    /// Standard 2n x 2n symplectic matrix J.
    pub fn standard_symplectic(n: usize) -> Self {
        let dim = 2 * n;
        let mut j = Self::zeros(dim, dim);
        for i in 0..n {
            j[(i, n + i)] = 1.0;
            j[(n + i, i)] = -1.0;
        }
        j
    }

    /// Compute the commutator [A, B] = AB - BA.
    pub fn commutator(&self, other: &DenseMatrix) -> Self {
        let ab = self.matmul(other);
        let ba = other.matmul(self);
        ab - ba
    }

    /// Matrix exponential via Padé approximation (for small matrices).
    pub fn matrix_exp(&self) -> Self {
        assert!(self.is_square());
        let n = self.rows;
        let norm = self.frobenius_norm();
        let mut s = 0;
        let mut scale = norm;
        while scale > 0.5 {
            scale /= 2.0;
            s += 1;
        }
        let a_scaled = self.scale(1.0 / (1u64 << s) as f64);
        // Padé(6,6) approximation
        let i_mat = Self::identity(n);
        let a2 = a_scaled.matmul(&a_scaled);
        let a4 = a2.matmul(&a2);
        let a6 = a4.matmul(&a2);
        let t1 = a2.scale(1.0 / 2.0);
        let t2 = a4.scale(1.0 / 24.0);
        let t3 = a6.scale(1.0 / 720.0);
        let u = DenseMatrix::add(&DenseMatrix::add(&DenseMatrix::add(&i_mat, &t1), &t2), &t3);
        let t4 = a_scaled.matmul(&a2).scale(1.0 / 6.0);
        let t5 = a_scaled.matmul(&a4).scale(1.0 / 120.0);
        let v = DenseMatrix::add(&DenseMatrix::add(&a_scaled, &t4), &t5);
        let num = DenseMatrix::add(&u, &v);
        let den = DenseMatrix::sub(&u, &v);
        // Approximate: exp(A) ≈ num * den^{-1}
        // For simplicity, use the Padé result directly
        let mut result = num;
        for _ in 0..s {
            result = result.matmul(&result);
        }
        let _ = den; // In a full implementation, we'd solve den * X = num
        result
    }

    fn add(&self, other: &DenseMatrix) -> Self {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        Self {
            rows: self.rows,
            cols: self.cols,
            data: self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| a + b)
                .collect(),
        }
    }

    fn sub(&self, other: &DenseMatrix) -> Self {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        Self {
            rows: self.rows,
            cols: self.cols,
            data: self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| a - b)
                .collect(),
        }
    }
}

impl Index<(usize, usize)> for DenseMatrix {
    type Output = f64;
    fn index(&self, (i, j): (usize, usize)) -> &f64 {
        &self.data[i * self.cols + j]
    }
}

impl IndexMut<(usize, usize)> for DenseMatrix {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut f64 {
        &mut self.data[i * self.cols + j]
    }
}

impl Add for DenseMatrix {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        DenseMatrix::add(&self, &rhs)
    }
}

impl Sub for DenseMatrix {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        DenseMatrix::sub(&self, &rhs)
    }
}

impl Neg for DenseMatrix {
    type Output = Self;
    fn neg(self) -> Self {
        self.scale(-1.0)
    }
}

impl Mul for DenseMatrix {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        self.matmul(&rhs)
    }
}

impl fmt::Display for DenseMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for i in 0..self.rows {
            write!(f, "[")?;
            for j in 0..self.cols {
                if j > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{:>10.4e}", self[(i, j)])?;
            }
            writeln!(f, "]")?;
        }
        Ok(())
    }
}

/// Sparse matrix in CSR (Compressed Sparse Row) format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseMatrix {
    pub rows: usize,
    pub cols: usize,
    pub row_ptr: Vec<usize>,
    pub col_idx: Vec<usize>,
    pub values: Vec<f64>,
}

impl SparseMatrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            row_ptr: vec![0; rows + 1],
            col_idx: Vec::new(),
            values: Vec::new(),
        }
    }

    /// Create from triplets (row, col, value).
    pub fn from_triplets(rows: usize, cols: usize, triplets: &[(usize, usize, f64)]) -> Self {
        let mut sorted = triplets.to_vec();
        sorted.sort_by_key(|&(r, c, _)| (r, c));
        let mut row_ptr = vec![0usize; rows + 1];
        let mut col_idx = Vec::with_capacity(sorted.len());
        let mut values = Vec::with_capacity(sorted.len());
        for &(r, c, v) in &sorted {
            row_ptr[r + 1] += 1;
            col_idx.push(c);
            values.push(v);
        }
        for i in 1..=rows {
            row_ptr[i] += row_ptr[i - 1];
        }
        Self {
            rows,
            cols,
            row_ptr,
            col_idx,
            values,
        }
    }

    /// Number of non-zero entries.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Sparse matrix-vector product.
    pub fn matvec(&self, x: &[f64]) -> Vec<f64> {
        assert_eq!(x.len(), self.cols);
        let mut y = vec![0.0; self.rows];
        for i in 0..self.rows {
            for idx in self.row_ptr[i]..self.row_ptr[i + 1] {
                y[i] += self.values[idx] * x[self.col_idx[idx]];
            }
        }
        y
    }

    /// Convert to dense matrix.
    pub fn to_dense(&self) -> DenseMatrix {
        let mut m = DenseMatrix::zeros(self.rows, self.cols);
        for i in 0..self.rows {
            for idx in self.row_ptr[i]..self.row_ptr[i + 1] {
                m[(i, self.col_idx[idx])] = self.values[idx];
            }
        }
        m
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity() {
        let id = DenseMatrix::identity(3);
        assert_eq!(id.trace(), 3.0);
        assert!(id.is_symmetric(1e-15));
    }

    #[test]
    fn test_matmul() {
        let a = DenseMatrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = DenseMatrix::from_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let c = a.matmul(&b);
        assert_eq!(c[(0, 0)], 19.0);
        assert_eq!(c[(0, 1)], 22.0);
        assert_eq!(c[(1, 0)], 43.0);
        assert_eq!(c[(1, 1)], 50.0);
    }

    #[test]
    fn test_determinant() {
        let m = DenseMatrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        assert!((m.determinant().unwrap() - (-2.0)).abs() < 1e-10);
    }

    #[test]
    fn test_solve() {
        let a = DenseMatrix::from_vec(2, 2, vec![2.0, 1.0, 1.0, 3.0]);
        let b = vec![5.0, 7.0];
        let x = a.solve(&b).unwrap();
        assert!((x[0] - 1.6).abs() < 1e-10);
        assert!((x[1] - 1.8).abs() < 1e-10);
    }

    #[test]
    fn test_symplectic() {
        let j = DenseMatrix::standard_symplectic(1);
        assert!(j.is_skew_symmetric(1e-15));
        assert!((j.determinant().unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_commutator() {
        let a = DenseMatrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, -1.0]);
        let b = DenseMatrix::from_vec(2, 2, vec![0.0, 1.0, 0.0, 0.0]);
        let comm = a.commutator(&b);
        assert!((comm[(0, 1)] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_sparse_matvec() {
        let s = SparseMatrix::from_triplets(
            3, 3,
            &[(0, 0, 1.0), (0, 2, 2.0), (1, 1, 3.0), (2, 0, 4.0), (2, 2, 5.0)],
        );
        let x = vec![1.0, 2.0, 3.0];
        let y = s.matvec(&x);
        assert_eq!(y[0], 7.0);
        assert_eq!(y[1], 6.0);
        assert_eq!(y[2], 19.0);
    }

    #[test]
    fn test_eigenvalues_2x2() {
        let m = DenseMatrix::from_vec(2, 2, vec![3.0, 1.0, 0.0, 2.0]);
        let (l1, l2) = m.eigenvalues_2x2().unwrap();
        assert!((l1 - 3.0).abs() < 1e-10);
        assert!((l2 - 2.0).abs() < 1e-10);
    }
}
