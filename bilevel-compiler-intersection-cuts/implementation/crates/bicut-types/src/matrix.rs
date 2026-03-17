//! Sparse matrix representations for optimization: CSR, CSC, and COO (triplet).
//!
//! Provides conversion between formats, matrix-vector multiply, transpose,
//! and submatrix extraction.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::error::{BicutError, BicutResult, ValidationError};

// ── Triplet (COO) format ───────────────────────────────────────────

/// Coordinate (COO / triplet) sparse matrix.
/// Stores (row, col, value) triples. Duplicates are summed on conversion.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TripletMatrix {
    pub nrows: usize,
    pub ncols: usize,
    pub row_indices: Vec<usize>,
    pub col_indices: Vec<usize>,
    pub values: Vec<f64>,
}

impl TripletMatrix {
    pub fn new(nrows: usize, ncols: usize) -> Self {
        Self {
            nrows,
            ncols,
            row_indices: Vec::new(),
            col_indices: Vec::new(),
            values: Vec::new(),
        }
    }

    pub fn with_capacity(nrows: usize, ncols: usize, nnz_hint: usize) -> Self {
        Self {
            nrows,
            ncols,
            row_indices: Vec::with_capacity(nnz_hint),
            col_indices: Vec::with_capacity(nnz_hint),
            values: Vec::with_capacity(nnz_hint),
        }
    }

    /// Add a triplet entry. Duplicates will be summed on conversion.
    pub fn add(&mut self, row: usize, col: usize, value: f64) {
        self.row_indices.push(row);
        self.col_indices.push(col);
        self.values.push(value);
    }

    /// Number of stored entries (may include duplicates).
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Convert to CSR format, summing duplicate entries.
    pub fn to_csr(&self) -> SparseMatrixCsr {
        let mut row_counts = vec![0usize; self.nrows + 1];
        for &r in &self.row_indices {
            row_counts[r + 1] += 1;
        }
        // prefix sum
        for i in 1..=self.nrows {
            row_counts[i] += row_counts[i - 1];
        }

        let nnz = self.nnz();
        let mut col_indices = vec![0usize; nnz];
        let mut values = vec![0.0f64; nnz];
        let mut offsets = row_counts.clone();

        for k in 0..nnz {
            let r = self.row_indices[k];
            let dest = offsets[r];
            col_indices[dest] = self.col_indices[k];
            values[dest] = self.values[k];
            offsets[r] += 1;
        }

        // Sort within each row and merge duplicates
        let mut csr = SparseMatrixCsr {
            nrows: self.nrows,
            ncols: self.ncols,
            row_offsets: row_counts,
            col_indices,
            values,
        };
        csr.sort_and_merge_duplicates();
        csr
    }

    /// Convert to CSC format.
    pub fn to_csc(&self) -> SparseMatrixCsc {
        self.transpose_triplet().to_csr().to_csc_via_transpose()
    }

    /// Transpose the triplet matrix (swap rows and columns).
    pub fn transpose_triplet(&self) -> TripletMatrix {
        TripletMatrix {
            nrows: self.ncols,
            ncols: self.nrows,
            row_indices: self.col_indices.clone(),
            col_indices: self.row_indices.clone(),
            values: self.values.clone(),
        }
    }

    /// Matrix-vector multiply: y = A * x.
    pub fn mul_vec(&self, x: &[f64]) -> BicutResult<Vec<f64>> {
        if x.len() != self.ncols {
            return Err(BicutError::DimensionMismatch {
                expected: self.ncols,
                got: x.len(),
            });
        }
        let mut y = vec![0.0; self.nrows];
        for k in 0..self.nnz() {
            y[self.row_indices[k]] += self.values[k] * x[self.col_indices[k]];
        }
        Ok(y)
    }

    /// Extract a submatrix for given row and column index sets.
    pub fn submatrix(&self, rows: &[usize], cols: &[usize]) -> TripletMatrix {
        let row_set: std::collections::HashSet<usize> = rows.iter().copied().collect();
        let col_set: std::collections::HashSet<usize> = cols.iter().copied().collect();

        // Build mapping from old index to new index
        let row_map: std::collections::HashMap<usize, usize> = rows
            .iter()
            .enumerate()
            .map(|(new, &old)| (old, new))
            .collect();
        let col_map: std::collections::HashMap<usize, usize> = cols
            .iter()
            .enumerate()
            .map(|(new, &old)| (old, new))
            .collect();

        let mut sub = TripletMatrix::new(rows.len(), cols.len());
        for k in 0..self.nnz() {
            let r = self.row_indices[k];
            let c = self.col_indices[k];
            if row_set.contains(&r) && col_set.contains(&c) {
                sub.add(row_map[&r], col_map[&c], self.values[k]);
            }
        }
        sub
    }

    pub fn validate(&self) -> BicutResult<()> {
        let n = self.values.len();
        if self.row_indices.len() != n || self.col_indices.len() != n {
            return Err(BicutError::Validation(ValidationError::StructuralError {
                detail: "triplet arrays have mismatched lengths".into(),
            }));
        }
        for k in 0..n {
            if self.row_indices[k] >= self.nrows {
                return Err(BicutError::IndexOutOfBounds {
                    index: self.row_indices[k],
                    length: self.nrows,
                });
            }
            if self.col_indices[k] >= self.ncols {
                return Err(BicutError::IndexOutOfBounds {
                    index: self.col_indices[k],
                    length: self.ncols,
                });
            }
        }
        Ok(())
    }

    /// Convert to a dense 2D vector (row-major).
    pub fn to_dense(&self) -> Vec<Vec<f64>> {
        let mut dense = vec![vec![0.0; self.ncols]; self.nrows];
        for k in 0..self.nnz() {
            dense[self.row_indices[k]][self.col_indices[k]] += self.values[k];
        }
        dense
    }

    /// Create from a dense 2D vector.
    pub fn from_dense(dense: &[Vec<f64>]) -> Self {
        let nrows = dense.len();
        let ncols = if nrows > 0 { dense[0].len() } else { 0 };
        let mut triplet = TripletMatrix::new(nrows, ncols);
        for (i, row) in dense.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                if val.abs() > 1e-15 {
                    triplet.add(i, j, val);
                }
            }
        }
        triplet
    }
}

impl Default for TripletMatrix {
    fn default() -> Self {
        Self::new(0, 0)
    }
}

impl fmt::Display for TripletMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TripletMatrix({}x{}, {} entries)",
            self.nrows,
            self.ncols,
            self.nnz()
        )
    }
}

// ── CSR format ─────────────────────────────────────────────────────

/// Compressed Sparse Row matrix.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SparseMatrixCsr {
    pub nrows: usize,
    pub ncols: usize,
    /// Length nrows+1: row_offsets[i]..row_offsets[i+1] gives the range in
    /// col_indices/values for row i.
    pub row_offsets: Vec<usize>,
    pub col_indices: Vec<usize>,
    pub values: Vec<f64>,
}

impl SparseMatrixCsr {
    pub fn new(nrows: usize, ncols: usize) -> Self {
        Self {
            nrows,
            ncols,
            row_offsets: vec![0; nrows + 1],
            col_indices: Vec::new(),
            values: Vec::new(),
        }
    }

    /// Number of stored nonzeros.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Nonzeros in a given row.
    pub fn row_nnz(&self, row: usize) -> usize {
        self.row_offsets[row + 1] - self.row_offsets[row]
    }

    /// Iterate over (col, value) pairs for a given row.
    pub fn row_iter(&self, row: usize) -> impl Iterator<Item = (usize, f64)> + '_ {
        let start = self.row_offsets[row];
        let end = self.row_offsets[row + 1];
        self.col_indices[start..end]
            .iter()
            .zip(&self.values[start..end])
            .map(|(&c, &v)| (c, v))
    }

    /// Matrix-vector multiply: y = A * x.
    pub fn mul_vec(&self, x: &[f64]) -> BicutResult<Vec<f64>> {
        if x.len() != self.ncols {
            return Err(BicutError::DimensionMismatch {
                expected: self.ncols,
                got: x.len(),
            });
        }
        let mut y = vec![0.0; self.nrows];
        for i in 0..self.nrows {
            let start = self.row_offsets[i];
            let end = self.row_offsets[i + 1];
            let mut sum = 0.0;
            for k in start..end {
                sum += self.values[k] * x[self.col_indices[k]];
            }
            y[i] = sum;
        }
        Ok(y)
    }

    /// Transpose: return a CSC representation that is structurally the transpose.
    pub fn transpose(&self) -> SparseMatrixCsr {
        let mut triplet = TripletMatrix::with_capacity(self.ncols, self.nrows, self.nnz());
        for i in 0..self.nrows {
            for (j, v) in self.row_iter(i) {
                triplet.add(j, i, v);
            }
        }
        triplet.to_csr()
    }

    /// Convert to CSC by transposing and reinterpreting.
    fn to_csc_via_transpose(&self) -> SparseMatrixCsc {
        let t = self.transpose();
        SparseMatrixCsc {
            nrows: self.nrows,
            ncols: self.ncols,
            col_offsets: t.row_offsets,
            row_indices: t.col_indices,
            values: t.values,
        }
    }

    /// Convert to CSC format.
    pub fn to_csc(&self) -> SparseMatrixCsc {
        self.to_csc_via_transpose()
    }

    /// Convert to triplet format.
    pub fn to_triplet(&self) -> TripletMatrix {
        let mut triplet = TripletMatrix::with_capacity(self.nrows, self.ncols, self.nnz());
        for i in 0..self.nrows {
            for (j, v) in self.row_iter(i) {
                triplet.add(i, j, v);
            }
        }
        triplet
    }

    /// Sort column indices within each row and merge duplicates.
    fn sort_and_merge_duplicates(&mut self) {
        let mut new_col_indices = Vec::with_capacity(self.values.len());
        let mut new_values = Vec::with_capacity(self.values.len());
        let mut new_row_offsets = Vec::with_capacity(self.nrows + 1);
        new_row_offsets.push(0);

        for i in 0..self.nrows {
            let start = self.row_offsets[i];
            let end = self.row_offsets[i + 1];

            // Collect and sort by column
            let mut entries: Vec<(usize, f64)> = self.col_indices[start..end]
                .iter()
                .zip(&self.values[start..end])
                .map(|(&c, &v)| (c, v))
                .collect();
            entries.sort_by_key(|&(c, _)| c);

            // Merge duplicates
            let mut prev_col: Option<usize> = None;
            for (c, v) in entries {
                if prev_col == Some(c) {
                    if let Some(last) = new_values.last_mut() {
                        *last += v;
                    }
                } else {
                    if v.abs() > 1e-15 || prev_col.is_none() {
                        new_col_indices.push(c);
                        new_values.push(v);
                    }
                    prev_col = Some(c);
                }
            }
            // Remove near-zero entries at end
            let row_start = *new_row_offsets.last().unwrap();
            let mut write = row_start;
            for read in row_start..new_values.len() {
                if new_values[read].abs() > 1e-15 {
                    new_col_indices[write] = new_col_indices[read];
                    new_values[write] = new_values[read];
                    write += 1;
                }
            }
            new_col_indices.truncate(write);
            new_values.truncate(write);
            new_row_offsets.push(new_values.len());
        }

        self.row_offsets = new_row_offsets;
        self.col_indices = new_col_indices;
        self.values = new_values;
    }

    /// Extract rows by index set.
    pub fn extract_rows(&self, rows: &[usize]) -> SparseMatrixCsr {
        let mut triplet = TripletMatrix::new(rows.len(), self.ncols);
        for (new_i, &old_i) in rows.iter().enumerate() {
            for (j, v) in self.row_iter(old_i) {
                triplet.add(new_i, j, v);
            }
        }
        triplet.to_csr()
    }

    /// Convert to dense row-major matrix.
    pub fn to_dense(&self) -> Vec<Vec<f64>> {
        let mut dense = vec![vec![0.0; self.ncols]; self.nrows];
        for i in 0..self.nrows {
            for (j, v) in self.row_iter(i) {
                dense[i][j] = v;
            }
        }
        dense
    }

    pub fn validate(&self) -> BicutResult<()> {
        if self.row_offsets.len() != self.nrows + 1 {
            return Err(BicutError::Validation(ValidationError::DimensionMismatch {
                context: "row_offsets length".into(),
                expected: self.nrows + 1,
                got: self.row_offsets.len(),
            }));
        }
        if self.col_indices.len() != self.values.len() {
            return Err(BicutError::Validation(ValidationError::StructuralError {
                detail: "col_indices and values have different lengths".into(),
            }));
        }
        for i in 0..self.nrows {
            if self.row_offsets[i] > self.row_offsets[i + 1] {
                return Err(BicutError::Validation(ValidationError::StructuralError {
                    detail: format!("row_offsets not monotone at row {}", i),
                }));
            }
        }
        for &c in &self.col_indices {
            if c >= self.ncols {
                return Err(BicutError::IndexOutOfBounds {
                    index: c,
                    length: self.ncols,
                });
            }
        }
        Ok(())
    }

    /// Frobenius norm.
    pub fn frobenius_norm(&self) -> f64 {
        self.values.iter().map(|v| v * v).sum::<f64>().sqrt()
    }
}

impl fmt::Display for SparseMatrixCsr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CSR({}x{}, {} nnz)", self.nrows, self.ncols, self.nnz())
    }
}

// ── CSC format ─────────────────────────────────────────────────────

/// Compressed Sparse Column matrix.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SparseMatrixCsc {
    pub nrows: usize,
    pub ncols: usize,
    /// Length ncols+1.
    pub col_offsets: Vec<usize>,
    pub row_indices: Vec<usize>,
    pub values: Vec<f64>,
}

impl SparseMatrixCsc {
    pub fn new(nrows: usize, ncols: usize) -> Self {
        Self {
            nrows,
            ncols,
            col_offsets: vec![0; ncols + 1],
            row_indices: Vec::new(),
            values: Vec::new(),
        }
    }

    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    pub fn col_nnz(&self, col: usize) -> usize {
        self.col_offsets[col + 1] - self.col_offsets[col]
    }

    /// Iterate over (row, value) pairs for a given column.
    pub fn col_iter(&self, col: usize) -> impl Iterator<Item = (usize, f64)> + '_ {
        let start = self.col_offsets[col];
        let end = self.col_offsets[col + 1];
        self.row_indices[start..end]
            .iter()
            .zip(&self.values[start..end])
            .map(|(&r, &v)| (r, v))
    }

    /// Matrix-vector multiply: y = A * x.
    pub fn mul_vec(&self, x: &[f64]) -> BicutResult<Vec<f64>> {
        if x.len() != self.ncols {
            return Err(BicutError::DimensionMismatch {
                expected: self.ncols,
                got: x.len(),
            });
        }
        let mut y = vec![0.0; self.nrows];
        for j in 0..self.ncols {
            let xj = x[j];
            if xj.abs() > 1e-30 {
                for (i, v) in self.col_iter(j) {
                    y[i] += v * xj;
                }
            }
        }
        Ok(y)
    }

    /// Convert to CSR.
    pub fn to_csr(&self) -> SparseMatrixCsr {
        let mut triplet = TripletMatrix::with_capacity(self.nrows, self.ncols, self.nnz());
        for j in 0..self.ncols {
            for (i, v) in self.col_iter(j) {
                triplet.add(i, j, v);
            }
        }
        triplet.to_csr()
    }

    /// Convert to triplet format.
    pub fn to_triplet(&self) -> TripletMatrix {
        let mut triplet = TripletMatrix::with_capacity(self.nrows, self.ncols, self.nnz());
        for j in 0..self.ncols {
            for (i, v) in self.col_iter(j) {
                triplet.add(i, j, v);
            }
        }
        triplet
    }

    /// Transpose to CSC (which is a CSR of the original).
    pub fn transpose(&self) -> SparseMatrixCsc {
        let csr = self.to_csr();
        SparseMatrixCsc {
            nrows: self.ncols,
            ncols: self.nrows,
            col_offsets: csr.row_offsets,
            row_indices: csr.col_indices,
            values: csr.values,
        }
    }

    /// Extract columns by index set.
    pub fn extract_cols(&self, cols: &[usize]) -> SparseMatrixCsc {
        let mut triplet = TripletMatrix::new(self.nrows, cols.len());
        for (new_j, &old_j) in cols.iter().enumerate() {
            for (i, v) in self.col_iter(old_j) {
                triplet.add(i, new_j, v);
            }
        }
        triplet.to_csc()
    }

    pub fn validate(&self) -> BicutResult<()> {
        if self.col_offsets.len() != self.ncols + 1 {
            return Err(BicutError::Validation(ValidationError::DimensionMismatch {
                context: "col_offsets length".into(),
                expected: self.ncols + 1,
                got: self.col_offsets.len(),
            }));
        }
        if self.row_indices.len() != self.values.len() {
            return Err(BicutError::Validation(ValidationError::StructuralError {
                detail: "row_indices and values have different lengths".into(),
            }));
        }
        for j in 0..self.ncols {
            if self.col_offsets[j] > self.col_offsets[j + 1] {
                return Err(BicutError::Validation(ValidationError::StructuralError {
                    detail: format!("col_offsets not monotone at col {}", j),
                }));
            }
        }
        for &r in &self.row_indices {
            if r >= self.nrows {
                return Err(BicutError::IndexOutOfBounds {
                    index: r,
                    length: self.nrows,
                });
            }
        }
        Ok(())
    }

    /// Convert to dense row-major matrix.
    pub fn to_dense(&self) -> Vec<Vec<f64>> {
        let mut dense = vec![vec![0.0; self.ncols]; self.nrows];
        for j in 0..self.ncols {
            for (i, v) in self.col_iter(j) {
                dense[i][j] = v;
            }
        }
        dense
    }

    pub fn frobenius_norm(&self) -> f64 {
        self.values.iter().map(|v| v * v).sum::<f64>().sqrt()
    }
}

impl fmt::Display for SparseMatrixCsc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CSC({}x{}, {} nnz)", self.nrows, self.ncols, self.nnz())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_triplet() -> TripletMatrix {
        // 2x3 matrix:
        // [1 0 2]
        // [0 3 4]
        let mut t = TripletMatrix::new(2, 3);
        t.add(0, 0, 1.0);
        t.add(0, 2, 2.0);
        t.add(1, 1, 3.0);
        t.add(1, 2, 4.0);
        t
    }

    #[test]
    fn test_triplet_basic() {
        let t = make_test_triplet();
        assert_eq!(t.nrows, 2);
        assert_eq!(t.ncols, 3);
        assert_eq!(t.nnz(), 4);
    }

    #[test]
    fn test_triplet_to_dense() {
        let t = make_test_triplet();
        let dense = t.to_dense();
        assert_eq!(dense[0], vec![1.0, 0.0, 2.0]);
        assert_eq!(dense[1], vec![0.0, 3.0, 4.0]);
    }

    #[test]
    fn test_triplet_mul_vec() {
        let t = make_test_triplet();
        let x = vec![1.0, 2.0, 3.0];
        let y = t.mul_vec(&x).unwrap();
        // [1*1 + 0*2 + 2*3, 0*1 + 3*2 + 4*3] = [7, 18]
        assert!((y[0] - 7.0).abs() < 1e-10);
        assert!((y[1] - 18.0).abs() < 1e-10);
    }

    #[test]
    fn test_triplet_to_csr() {
        let t = make_test_triplet();
        let csr = t.to_csr();
        assert_eq!(csr.nrows, 2);
        assert_eq!(csr.ncols, 3);
        let y = csr.mul_vec(&[1.0, 2.0, 3.0]).unwrap();
        assert!((y[0] - 7.0).abs() < 1e-10);
        assert!((y[1] - 18.0).abs() < 1e-10);
    }

    #[test]
    fn test_csr_to_csc_roundtrip() {
        let t = make_test_triplet();
        let csr = t.to_csr();
        let csc = csr.to_csc();
        let y = csc.mul_vec(&[1.0, 2.0, 3.0]).unwrap();
        assert!((y[0] - 7.0).abs() < 1e-10);
        assert!((y[1] - 18.0).abs() < 1e-10);
    }

    #[test]
    fn test_csr_transpose() {
        let t = make_test_triplet();
        let csr = t.to_csr();
        let at = csr.transpose();
        assert_eq!(at.nrows, 3);
        assert_eq!(at.ncols, 2);
        // A^T * [1, 2] = [1*1+0*2, 0*1+3*2, 2*1+4*2] = [1, 6, 10]
        let y = at.mul_vec(&[1.0, 2.0]).unwrap();
        assert!((y[0] - 1.0).abs() < 1e-10);
        assert!((y[1] - 6.0).abs() < 1e-10);
        assert!((y[2] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_triplet_duplicate_entries() {
        let mut t = TripletMatrix::new(2, 2);
        t.add(0, 0, 1.0);
        t.add(0, 0, 2.0); // duplicate: should sum to 3
        t.add(1, 1, 5.0);
        let csr = t.to_csr();
        let y = csr.mul_vec(&[1.0, 1.0]).unwrap();
        assert!((y[0] - 3.0).abs() < 1e-10);
        assert!((y[1] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_submatrix_extraction() {
        let t = make_test_triplet();
        // Extract row 1, cols 1 and 2 → 1x2 matrix [[3, 4]]
        let sub = t.submatrix(&[1], &[1, 2]);
        assert_eq!(sub.nrows, 1);
        assert_eq!(sub.ncols, 2);
        let dense = sub.to_dense();
        assert!((dense[0][0] - 3.0).abs() < 1e-10);
        assert!((dense[0][1] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_csr_extract_rows() {
        let t = make_test_triplet();
        let csr = t.to_csr();
        let sub = csr.extract_rows(&[1]);
        assert_eq!(sub.nrows, 1);
        let dense = sub.to_dense();
        assert_eq!(dense[0], vec![0.0, 3.0, 4.0]);
    }

    #[test]
    fn test_from_dense() {
        let dense = vec![vec![1.0, 0.0, 2.0], vec![0.0, 3.0, 4.0]];
        let t = TripletMatrix::from_dense(&dense);
        assert_eq!(t.nnz(), 4);
        let back = t.to_dense();
        assert_eq!(back, dense);
    }

    #[test]
    fn test_validate_bad_indices() {
        let mut t = TripletMatrix::new(2, 2);
        t.add(5, 0, 1.0); // row out of bounds
        assert!(t.validate().is_err());
    }

    #[test]
    fn test_frobenius_norm() {
        let t = make_test_triplet();
        let csr = t.to_csr();
        // sqrt(1^2 + 2^2 + 3^2 + 4^2) = sqrt(30)
        let expected = (1.0_f64 + 4.0 + 9.0 + 16.0).sqrt();
        assert!((csr.frobenius_norm() - expected).abs() < 1e-10);
    }
}
