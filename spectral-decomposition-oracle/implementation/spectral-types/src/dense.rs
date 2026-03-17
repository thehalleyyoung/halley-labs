//! Dense matrix and vector types with row-major storage.

use serde::{Deserialize, Serialize};
use crate::error::{self, MatrixError, SpectralError};
use crate::scalar::Scalar;

/// Dense matrix in row-major order.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct DenseMatrix<T: Scalar> {
    pub data: Vec<T>,
    pub rows: usize,
    pub cols: usize,
}

impl<T: Scalar> DenseMatrix<T> {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self { data: vec![T::zero(); rows * cols], rows, cols }
    }

    pub fn from_vec(rows: usize, cols: usize, data: Vec<T>) -> error::Result<Self> {
        if data.len() != rows * cols {
            return Err(SpectralError::Matrix(MatrixError::InvalidReshape {
                from_rows: 1, from_cols: data.len(), to_rows: rows, to_cols: cols,
            }));
        }
        Ok(Self { data, rows, cols })
    }

    pub fn from_fn(rows: usize, cols: usize, f: impl Fn(usize, usize) -> T) -> Self {
        let mut data = Vec::with_capacity(rows * cols);
        for i in 0..rows { for j in 0..cols { data.push(f(i, j)); } }
        Self { data, rows, cols }
    }

    pub fn identity(n: usize) -> Self {
        Self::from_fn(n, n, |i, j| if i == j { T::one() } else { T::zero() })
    }

    pub fn diagonal(diag: &[T]) -> Self {
        let n = diag.len();
        Self::from_fn(n, n, |i, j| if i == j { diag[i] } else { T::zero() })
    }

    pub fn ones(rows: usize, cols: usize) -> Self {
        Self { data: vec![T::one(); rows * cols], rows, cols }
    }

    pub fn get(&self, row: usize, col: usize) -> Option<T> {
        if row < self.rows && col < self.cols { Some(self.data[row * self.cols + col]) } else { None }
    }

    pub fn set(&mut self, row: usize, col: usize, val: T) -> error::Result<()> {
        if row >= self.rows || col >= self.cols {
            return Err(SpectralError::Matrix(MatrixError::IndexOutOfBounds {
                row, col, rows: self.rows, cols: self.cols,
            }));
        }
        self.data[row * self.cols + col] = val;
        Ok(())
    }

    pub fn shape(&self) -> (usize, usize) { (self.rows, self.cols) }
    pub fn is_square(&self) -> bool { self.rows == self.cols }

    pub fn transpose(&self) -> Self {
        let mut r = Self::zeros(self.cols, self.rows);
        for i in 0..self.rows { for j in 0..self.cols { r.data[j * self.rows + i] = self.data[i * self.cols + j]; } }
        r
    }

    pub fn add(&self, other: &Self) -> error::Result<Self> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(SpectralError::Matrix(MatrixError::IncompatibleDimensions {
                operation: "add".into(), left_rows: self.rows, left_cols: self.cols,
                right_rows: other.rows, right_cols: other.cols,
            }));
        }
        Ok(Self { data: self.data.iter().zip(&other.data).map(|(&a, &b)| a + b).collect(), rows: self.rows, cols: self.cols })
    }

    pub fn sub(&self, other: &Self) -> error::Result<Self> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(SpectralError::Matrix(MatrixError::IncompatibleDimensions {
                operation: "sub".into(), left_rows: self.rows, left_cols: self.cols,
                right_rows: other.rows, right_cols: other.cols,
            }));
        }
        Ok(Self { data: self.data.iter().zip(&other.data).map(|(&a, &b)| a - b).collect(), rows: self.rows, cols: self.cols })
    }

    pub fn scale(&self, s: T) -> Self {
        Self { data: self.data.iter().map(|&v| v * s).collect(), rows: self.rows, cols: self.cols }
    }

    pub fn scale_in_place(&mut self, s: T) { for v in &mut self.data { *v = *v * s; } }

    pub fn mul_vec(&self, x: &[T]) -> error::Result<Vec<T>> {
        if x.len() != self.cols {
            return Err(SpectralError::Matrix(MatrixError::IncompatibleDimensions {
                operation: "mul_vec".into(), left_rows: self.rows, left_cols: self.cols,
                right_rows: x.len(), right_cols: 1,
            }));
        }
        let mut y = vec![T::zero(); self.rows];
        for i in 0..self.rows {
            let base = i * self.cols;
            for j in 0..self.cols { y[i] = y[i] + self.data[base + j] * x[j]; }
        }
        Ok(y)
    }

    pub fn mul_mat(&self, other: &Self) -> error::Result<Self> {
        if self.cols != other.rows {
            return Err(SpectralError::Matrix(MatrixError::IncompatibleDimensions {
                operation: "mul_mat".into(), left_rows: self.rows, left_cols: self.cols,
                right_rows: other.rows, right_cols: other.cols,
            }));
        }
        let mut result = Self::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for k in 0..self.cols {
                let a = self.data[i * self.cols + k];
                if a.is_approx_zero(T::zero_threshold()) { continue; }
                for j in 0..other.cols {
                    result.data[i * other.cols + j] = result.data[i * other.cols + j] + a * other.data[k * other.cols + j];
                }
            }
        }
        Ok(result)
    }

    pub fn element_wise_mul(&self, other: &Self) -> error::Result<Self> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(SpectralError::Matrix(MatrixError::IncompatibleDimensions {
                operation: "hadamard".into(), left_rows: self.rows, left_cols: self.cols,
                right_rows: other.rows, right_cols: other.cols,
            }));
        }
        Ok(Self { data: self.data.iter().zip(&other.data).map(|(&a, &b)| a * b).collect(), rows: self.rows, cols: self.cols })
    }

    pub fn element_wise_div(&self, other: &Self) -> error::Result<Self> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(SpectralError::Matrix(MatrixError::IncompatibleDimensions {
                operation: "div".into(), left_rows: self.rows, left_cols: self.cols,
                right_rows: other.rows, right_cols: other.cols,
            }));
        }
        Ok(Self {
            data: self.data.iter().zip(&other.data).map(|(&a, &b)| {
                if b.is_approx_zero(T::zero_threshold()) { T::zero() } else { a / b }
            }).collect(),
            rows: self.rows, cols: self.cols,
        })
    }

    pub fn apply(&self, f: impl Fn(T) -> T) -> Self {
        Self { data: self.data.iter().map(|&v| f(v)).collect(), rows: self.rows, cols: self.cols }
    }

    pub fn row(&self, i: usize) -> Option<&[T]> {
        if i < self.rows { Some(&self.data[i * self.cols..(i + 1) * self.cols]) } else { None }
    }

    pub fn col(&self, j: usize) -> Option<Vec<T>> {
        if j >= self.cols { return None; }
        Some((0..self.rows).map(|i| self.data[i * self.cols + j]).collect())
    }

    pub fn submatrix(&self, rs: usize, cs: usize, nr: usize, nc: usize) -> error::Result<Self> {
        if rs + nr > self.rows || cs + nc > self.cols {
            return Err(SpectralError::Matrix(MatrixError::IndexOutOfBounds {
                row: rs + nr, col: cs + nc, rows: self.rows, cols: self.cols,
            }));
        }
        let mut data = Vec::with_capacity(nr * nc);
        for i in rs..rs + nr { for j in cs..cs + nc { data.push(self.data[i * self.cols + j]); } }
        Ok(Self { data, rows: nr, cols: nc })
    }

    pub fn set_row(&mut self, i: usize, vals: &[T]) -> error::Result<()> {
        if i >= self.rows || vals.len() != self.cols {
            return Err(SpectralError::Matrix(MatrixError::IndexOutOfBounds {
                row: i, col: 0, rows: self.rows, cols: self.cols,
            }));
        }
        self.data[i * self.cols..(i + 1) * self.cols].copy_from_slice(vals);
        Ok(())
    }

    pub fn set_col(&mut self, j: usize, vals: &[T]) -> error::Result<()> {
        if j >= self.cols || vals.len() != self.rows {
            return Err(SpectralError::Matrix(MatrixError::IndexOutOfBounds {
                row: 0, col: j, rows: self.rows, cols: self.cols,
            }));
        }
        for i in 0..self.rows { self.data[i * self.cols + j] = vals[i]; }
        Ok(())
    }

    pub fn trace(&self) -> T {
        let n = self.rows.min(self.cols);
        (0..n).map(|i| self.data[i * self.cols + i]).fold(T::zero(), |a, b| a + b)
    }

    pub fn frobenius_norm(&self) -> T {
        self.data.iter().map(|&v| v * v).fold(T::zero(), |a, b| a + b).sqrt()
    }

    pub fn l1_norm(&self) -> T {
        (0..self.cols).map(|j| {
            (0..self.rows).map(|i| self.data[i * self.cols + j].abs()).fold(T::zero(), |a, b| a + b)
        }).fold(T::zero(), |a, b| a.ordered_max(b))
    }

    pub fn linf_norm(&self) -> T {
        (0..self.rows).map(|i| {
            (0..self.cols).map(|j| self.data[i * self.cols + j].abs()).fold(T::zero(), |a, b| a + b)
        }).fold(T::zero(), |a, b| a.ordered_max(b))
    }

    pub fn max_abs(&self) -> T {
        self.data.iter().copied().map(|v| v.abs()).fold(T::zero(), |a, b| a.ordered_max(b))
    }

    pub fn reshape(&self, nr: usize, nc: usize) -> error::Result<Self> {
        if nr * nc != self.rows * self.cols {
            return Err(SpectralError::Matrix(MatrixError::InvalidReshape {
                from_rows: self.rows, from_cols: self.cols, to_rows: nr, to_cols: nc,
            }));
        }
        Ok(Self { data: self.data.clone(), rows: nr, cols: nc })
    }

    pub fn diagonal_vec(&self) -> Vec<T> {
        let n = self.rows.min(self.cols);
        (0..n).map(|i| self.data[i * self.cols + i]).collect()
    }

    pub fn sum(&self) -> T { self.data.iter().copied().fold(T::zero(), |a, b| a + b) }

    pub fn row_sums(&self) -> Vec<T> {
        (0..self.rows).map(|i| {
            self.data[i * self.cols..(i + 1) * self.cols].iter().copied().fold(T::zero(), |a, b| a + b)
        }).collect()
    }

    pub fn col_sums(&self) -> Vec<T> {
        let mut s = vec![T::zero(); self.cols];
        for i in 0..self.rows { for j in 0..self.cols { s[j] = s[j] + self.data[i * self.cols + j]; } }
        s
    }

    pub fn is_symmetric(&self, tol: T) -> bool {
        if !self.is_square() { return false; }
        for i in 0..self.rows {
            for j in (i + 1)..self.cols {
                if (self.data[i * self.cols + j] - self.data[j * self.cols + i]).abs() > tol { return false; }
            }
        }
        true
    }

    pub fn nnz(&self) -> usize {
        self.data.iter().filter(|&&v| !v.is_approx_zero(T::zero_threshold())).count()
    }

    pub fn rank1_update(&mut self, alpha: T, x: &[T], y: &[T]) {
        for i in 0..self.rows.min(x.len()) {
            for j in 0..self.cols.min(y.len()) {
                self.data[i * self.cols + j] = self.data[i * self.cols + j] + alpha * x[i] * y[j];
            }
        }
    }

    pub fn add_row_broadcast(&mut self, v: &[T]) {
        for i in 0..self.rows { for j in 0..self.cols.min(v.len()) {
            self.data[i * self.cols + j] = self.data[i * self.cols + j] + v[j];
        }}
    }

    pub fn add_col_broadcast(&mut self, v: &[T]) {
        for i in 0..self.rows.min(v.len()) { for j in 0..self.cols {
            self.data[i * self.cols + j] = self.data[i * self.cols + j] + v[i];
        }}
    }
}

/// A dense vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct DenseVector<T: Scalar> {
    pub data: Vec<T>,
}

impl<T: Scalar> DenseVector<T> {
    pub fn zeros(n: usize) -> Self { Self { data: vec![T::zero(); n] } }
    pub fn ones(n: usize) -> Self { Self { data: vec![T::one(); n] } }
    pub fn from_vec(data: Vec<T>) -> Self { Self { data } }
    pub fn len(&self) -> usize { self.data.len() }
    pub fn is_empty(&self) -> bool { self.data.is_empty() }
    pub fn get(&self, i: usize) -> Option<T> { self.data.get(i).copied() }
    pub fn set(&mut self, i: usize, val: T) { if i < self.data.len() { self.data[i] = val; } }

    pub fn dot(&self, other: &Self) -> T {
        self.data.iter().zip(&other.data).map(|(&a, &b)| a * b).fold(T::zero(), |a, v| a + v)
    }

    pub fn norm_l2(&self) -> T { self.dot(self).sqrt() }
    pub fn norm_l1(&self) -> T { self.data.iter().map(|&v| v.abs()).fold(T::zero(), |a, b| a + b) }
    pub fn norm_linf(&self) -> T { self.data.iter().map(|&v| v.abs()).fold(T::zero(), |a, b| a.ordered_max(b)) }

    pub fn add(&self, other: &Self) -> Self {
        Self { data: self.data.iter().zip(&other.data).map(|(&a, &b)| a + b).collect() }
    }

    pub fn sub(&self, other: &Self) -> Self {
        Self { data: self.data.iter().zip(&other.data).map(|(&a, &b)| a - b).collect() }
    }

    pub fn scale(&self, s: T) -> Self { Self { data: self.data.iter().map(|&v| v * s).collect() } }
    pub fn scale_in_place(&mut self, s: T) { for v in &mut self.data { *v = *v * s; } }

    pub fn normalize(&mut self) -> T {
        let n = self.norm_l2();
        if !n.is_approx_zero(T::zero_threshold()) { self.scale_in_place(T::one() / n); }
        n
    }

    pub fn outer_product(&self, other: &Self) -> DenseMatrix<T> {
        let rows = self.len(); let cols = other.len();
        let mut data = Vec::with_capacity(rows * cols);
        for &a in &self.data { for &b in &other.data { data.push(a * b); } }
        DenseMatrix { data, rows, cols }
    }

    pub fn sum(&self) -> T { self.data.iter().copied().fold(T::zero(), |a, b| a + b) }

    pub fn mean(&self) -> T {
        if self.data.is_empty() { return T::zero(); }
        self.sum() / T::from_f64_lossy(self.data.len() as f64)
    }

    pub fn element_wise_mul(&self, other: &Self) -> Self {
        Self { data: self.data.iter().zip(&other.data).map(|(&a, &b)| a * b).collect() }
    }

    pub fn apply(&self, f: impl Fn(T) -> T) -> Self {
        Self { data: self.data.iter().map(|&v| f(v)).collect() }
    }

    pub fn max(&self) -> T { self.data.iter().copied().fold(T::neg_infinity(), |a, b| a.ordered_max(b)) }
    pub fn min(&self) -> T { self.data.iter().copied().fold(T::infinity(), |a, b| a.ordered_min(b)) }

    pub fn argmax(&self) -> usize {
        self.data.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal)).map(|(i,_)| i).unwrap_or(0)
    }

    pub fn argmin(&self) -> usize {
        self.data.iter().enumerate().min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal)).map(|(i,_)| i).unwrap_or(0)
    }

    pub fn as_slice(&self) -> &[T] { &self.data }
    pub fn as_mut_slice(&mut self) -> &mut [T] { &mut self.data }

    pub fn axpy(&mut self, alpha: T, other: &Self) {
        for (a, &b) in self.data.iter_mut().zip(&other.data) { *a = *a + alpha * b; }
    }

    pub fn distance_l2(&self, other: &Self) -> T { self.sub(other).norm_l2() }
}

impl<T: Scalar> crate::traits::VectorLike<T> for DenseVector<T> {
    fn len(&self) -> usize { self.data.len() }
    fn get(&self, index: usize) -> Option<T> { self.data.get(index).copied() }
    fn as_slice(&self) -> &[T] { &self.data }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test] fn test_zeros() { let m: DenseMatrix<f64> = DenseMatrix::zeros(3, 4); assert_eq!(m.shape(), (3, 4)); }
    #[test] fn test_identity() { let m: DenseMatrix<f64> = DenseMatrix::identity(3); assert!((m.trace() - 3.0).abs() < 1e-10); }

    #[test] fn test_transpose() {
        let m = DenseMatrix::from_vec(2, 3, vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let t = m.transpose(); assert_eq!(t.shape(), (3, 2)); assert_eq!(t.get(0, 1), Some(4.0));
    }

    #[test] fn test_add() {
        let a: DenseMatrix<f64> = DenseMatrix::ones(2, 2);
        let c = a.add(&a).unwrap(); assert_eq!(c.get(0, 0), Some(2.0));
    }

    #[test] fn test_mul_vec() {
        let m: DenseMatrix<f64> = DenseMatrix::identity(3);
        assert_eq!(m.mul_vec(&[1.0, 2.0, 3.0]).unwrap(), vec![1.0, 2.0, 3.0]);
    }

    #[test] fn test_mul_mat() {
        let a: DenseMatrix<f64> = DenseMatrix::identity(2);
        let b = DenseMatrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let c = a.mul_mat(&b).unwrap(); assert_eq!(c.get(1, 1), Some(4.0));
    }

    #[test] fn test_frobenius() {
        let m = DenseMatrix::from_vec(1, 2, vec![3.0_f64, 4.0]).unwrap();
        assert!((m.frobenius_norm() - 5.0).abs() < 1e-10);
    }

    #[test] fn test_submatrix() {
        let m = DenseMatrix::from_vec(3, 3, vec![1.0_f64,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0]).unwrap();
        let s = m.submatrix(0, 0, 2, 2).unwrap(); assert_eq!(s.get(1, 1), Some(5.0));
    }

    #[test] fn test_reshape() {
        let m = DenseMatrix::from_vec(2, 3, vec![1.0_f64,2.0,3.0,4.0,5.0,6.0]).unwrap();
        assert_eq!(m.reshape(3, 2).unwrap().shape(), (3, 2));
    }

    #[test] fn test_symmetric() {
        let m = DenseMatrix::from_vec(2, 2, vec![1.0_f64, 2.0, 2.0, 3.0]).unwrap();
        assert!(m.is_symmetric(1e-10));
    }

    #[test] fn test_vec_basics() {
        let v: DenseVector<f64> = DenseVector::from_vec(vec![3.0, 4.0]);
        assert!((v.norm_l2() - 5.0).abs() < 1e-10);
        assert!((v.norm_l1() - 7.0).abs() < 1e-10);
    }

    #[test] fn test_vec_dot() {
        let a = DenseVector::from_vec(vec![1.0_f64, 2.0, 3.0]);
        let b = DenseVector::from_vec(vec![4.0, 5.0, 6.0]);
        assert!((a.dot(&b) - 32.0).abs() < 1e-10);
    }

    #[test] fn test_outer() {
        let a = DenseVector::from_vec(vec![1.0_f64, 2.0]);
        let b = DenseVector::from_vec(vec![3.0, 4.0, 5.0]);
        let m = a.outer_product(&b); assert_eq!(m.get(1, 2), Some(10.0));
    }

    #[test] fn test_normalize() {
        let mut v = DenseVector::from_vec(vec![3.0_f64, 4.0]);
        v.normalize(); assert!((v.norm_l2() - 1.0).abs() < 1e-10);
    }

    #[test] fn test_argmax() { let v = DenseVector::from_vec(vec![1.0_f64, 5.0, 3.0]); assert_eq!(v.argmax(), 1); }

    #[test] fn test_norms() {
        let m = DenseMatrix::from_vec(2, 2, vec![1.0_f64, -2.0, 3.0, -4.0]).unwrap();
        assert!((m.l1_norm() - 6.0).abs() < 1e-10); assert!((m.linf_norm() - 7.0).abs() < 1e-10);
    }

    #[test] fn test_axpy() {
        let mut a = DenseVector::from_vec(vec![1.0_f64, 2.0]);
        let b = DenseVector::from_vec(vec![3.0, 4.0]); a.axpy(2.0, &b);
        assert_eq!(a.data, vec![7.0, 10.0]);
    }

    #[test] fn test_distance() {
        let a = DenseVector::from_vec(vec![0.0_f64, 0.0]);
        let b = DenseVector::from_vec(vec![3.0, 4.0]);
        assert!((a.distance_l2(&b) - 5.0).abs() < 1e-10);
    }

    #[test] fn test_rank1() {
        let mut m: DenseMatrix<f64> = DenseMatrix::zeros(2, 2);
        m.rank1_update(1.0, &[1.0, 2.0], &[3.0, 4.0]);
        assert_eq!(m.get(0, 0), Some(3.0)); assert_eq!(m.get(1, 1), Some(8.0));
    }

    #[test] fn test_row_col_sums() {
        let m = DenseMatrix::from_vec(2, 2, vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(m.row_sums(), vec![3.0, 7.0]); assert_eq!(m.col_sums(), vec![4.0, 6.0]);
    }

    #[test] fn test_mean() {
        let v = DenseVector::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0]);
        assert!((v.mean() - 2.5).abs() < 1e-10);
    }

    #[test] fn test_broadcast() {
        let mut m: DenseMatrix<f64> = DenseMatrix::zeros(2, 3);
        m.add_row_broadcast(&[1.0, 2.0, 3.0]);
        assert_eq!(m.get(1, 2), Some(3.0));
    }

    #[test] fn test_diag_vec() {
        let m = DenseMatrix::from_vec(2, 2, vec![1.0_f64, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(m.diagonal_vec(), vec![1.0, 4.0]);
    }

    #[test] fn test_elem_ops() {
        let a = DenseMatrix::from_vec(1, 3, vec![2.0_f64, 4.0, 6.0]).unwrap();
        let b = DenseMatrix::from_vec(1, 3, vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!(a.element_wise_mul(&b).unwrap().get(0, 0), Some(2.0));
        assert_eq!(a.element_wise_div(&b).unwrap().get(0, 0), Some(2.0));
    }
}
