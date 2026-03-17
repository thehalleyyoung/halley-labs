//! Differential matrix operations for storing and analyzing stage×test results.

use serde::{Deserialize, Serialize};
use shared_types::{LocalizerError, Result, StageId, TestCaseId};
use std::fmt;

// ── DifferentialMatrix ──────────────────────────────────────────────────────

/// A dense matrix where rows are test cases and columns are pipeline stages.
/// `data[i][j]` = differential Δ_j for test case i.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DifferentialMatrix {
    pub data: Vec<Vec<f64>>,
    pub num_tests: usize,
    pub num_stages: usize,
    pub stage_ids: Vec<StageId>,
    pub test_ids: Vec<TestCaseId>,
}

impl DifferentialMatrix {
    /// Create from explicit data. `data` is row-major: data[test][stage].
    pub fn new(
        data: Vec<Vec<f64>>,
        stage_ids: Vec<StageId>,
        test_ids: Vec<TestCaseId>,
    ) -> Result<Self> {
        let num_tests = data.len();
        let num_stages = stage_ids.len();
        for (i, row) in data.iter().enumerate() {
            if row.len() != num_stages {
                return Err(LocalizerError::validation("validation", format!(
                    "Row {} has {} columns, expected {}",
                    i,
                    row.len(),
                    num_stages
                )));
            }
        }
        if test_ids.len() != num_tests {
            return Err(LocalizerError::validation("validation", format!(
                "test_ids length {} != data row count {}",
                test_ids.len(),
                num_tests
            )));
        }
        Ok(Self {
            data,
            num_tests,
            num_stages,
            stage_ids,
            test_ids,
        })
    }

    /// Create a zero matrix.
    pub fn zeros(
        num_tests: usize,
        num_stages: usize,
        stage_ids: Vec<StageId>,
        test_ids: Vec<TestCaseId>,
    ) -> Self {
        Self {
            data: vec![vec![0.0; num_stages]; num_tests],
            num_tests,
            num_stages,
            stage_ids,
            test_ids,
        }
    }

    pub fn get(&self, test: usize, stage: usize) -> Option<f64> {
        self.data.get(test).and_then(|row| row.get(stage).copied())
    }

    pub fn set(&mut self, test: usize, stage: usize, value: f64) -> Result<()> {
        if test >= self.num_tests || stage >= self.num_stages {
            return Err(LocalizerError::validation("validation", format!(
                "Index ({}, {}) out of bounds ({}x{})",
                test, stage, self.num_tests, self.num_stages
            )));
        }
        self.data[test][stage] = value;
        Ok(())
    }

    /// Get an entire row (one test case across all stages).
    pub fn row(&self, test: usize) -> Option<&[f64]> {
        self.data.get(test).map(|r| r.as_slice())
    }

    /// Get an entire column (one stage across all test cases).
    pub fn column(&self, stage: usize) -> Option<Vec<f64>> {
        if stage >= self.num_stages {
            return None;
        }
        Some(self.data.iter().map(|row| row[stage]).collect())
    }

    /// Transpose: stages become rows, tests become columns.
    pub fn transpose(&self) -> DifferentialMatrix {
        let mut transposed = vec![vec![0.0; self.num_tests]; self.num_stages];
        for i in 0..self.num_tests {
            for j in 0..self.num_stages {
                transposed[j][i] = self.data[i][j];
            }
        }
        DifferentialMatrix {
            data: transposed,
            num_tests: self.num_stages,
            num_stages: self.num_tests,
            stage_ids: self.test_ids.iter().map(|t| StageId::new(&t.0.to_string())).collect(),
            test_ids: self.stage_ids.iter().map(|s| TestCaseId::new(&s.0.to_string())).collect(),
        }
    }

    /// Mean of a column (stage).
    pub fn column_mean(&self, stage: usize) -> Option<f64> {
        let col = self.column(stage)?;
        if col.is_empty() {
            return Some(0.0);
        }
        Some(col.iter().sum::<f64>() / col.len() as f64)
    }

    /// Standard deviation of a column (stage).
    pub fn column_std(&self, stage: usize) -> Option<f64> {
        let col = self.column(stage)?;
        let n = col.len();
        if n < 2 {
            return Some(0.0);
        }
        let mean = col.iter().sum::<f64>() / n as f64;
        let var = col.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        Some(var.sqrt())
    }

    /// Percentile of a column (stage). `p` in [0.0, 1.0].
    pub fn column_percentile(&self, stage: usize, p: f64) -> Option<f64> {
        let mut col = self.column(stage)?;
        if col.is_empty() {
            return Some(0.0);
        }
        col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((col.len() - 1) as f64 * p.clamp(0.0, 1.0)) as usize;
        Some(col[idx])
    }

    /// Z-score normalization per stage column.
    pub fn normalize_columns(&mut self) {
        for j in 0..self.num_stages {
            let mean = self.column_mean(j).unwrap_or(0.0);
            let std = self.column_std(j).unwrap_or(1.0);
            let std = if std < 1e-12 { 1.0 } else { std };
            for i in 0..self.num_tests {
                self.data[i][j] = (self.data[i][j] - mean) / std;
            }
        }
    }

    /// Subtract per-column baseline means (from calibration).
    pub fn subtract_baseline(&mut self, baselines: &[f64]) {
        let n = baselines.len().min(self.num_stages);
        for j in 0..n {
            for i in 0..self.num_tests {
                self.data[i][j] -= baselines[j];
            }
        }
    }

    /// Extract sub-matrix containing only rows where at least one column exceeds the threshold.
    pub fn filter_violations(&self, threshold: f64) -> DifferentialMatrix {
        let mut filtered_data = Vec::new();
        let mut filtered_test_ids = Vec::new();
        for (i, row) in self.data.iter().enumerate() {
            if row.iter().any(|&v| v > threshold) {
                filtered_data.push(row.clone());
                filtered_test_ids.push(self.test_ids[i].clone());
            }
        }
        let num_tests = filtered_data.len();
        DifferentialMatrix {
            data: filtered_data,
            num_tests,
            num_stages: self.num_stages,
            stage_ids: self.stage_ids.clone(),
            test_ids: filtered_test_ids,
        }
    }

    /// Compute pairwise Pearson correlation between stage columns.
    pub fn correlation_matrix(&self) -> Vec<Vec<f64>> {
        let mut corr = vec![vec![0.0; self.num_stages]; self.num_stages];
        let means: Vec<f64> = (0..self.num_stages)
            .map(|j| self.column_mean(j).unwrap_or(0.0))
            .collect();
        let stds: Vec<f64> = (0..self.num_stages)
            .map(|j| self.column_std(j).unwrap_or(1.0))
            .collect();

        for a in 0..self.num_stages {
            for b in a..self.num_stages {
                if a == b {
                    corr[a][b] = 1.0;
                    continue;
                }
                let std_a = if stds[a] < 1e-12 { 1.0 } else { stds[a] };
                let std_b = if stds[b] < 1e-12 { 1.0 } else { stds[b] };
                let n = self.num_tests as f64;
                if n < 2.0 {
                    corr[a][b] = 0.0;
                    corr[b][a] = 0.0;
                    continue;
                }
                let cov: f64 = self
                    .data
                    .iter()
                    .map(|row| (row[a] - means[a]) * (row[b] - means[b]))
                    .sum::<f64>()
                    / (n - 1.0);
                let r = cov / (std_a * std_b);
                corr[a][b] = r;
                corr[b][a] = r;
            }
        }
        corr
    }

    /// Simple PCA via power iteration. Returns the top `k` principal component vectors.
    pub fn principal_components(&self, k: usize) -> Vec<Vec<f64>> {
        let cov = self.covariance_matrix();
        let dim = cov.len();
        let k = k.min(dim);
        let mut components = Vec::with_capacity(k);
        let mut deflated = cov;

        for _ in 0..k {
            let eigvec = power_iteration(&deflated, 200);
            let eigenvalue = rayleigh_quotient(&deflated, &eigvec);
            // Deflate: A = A - λ * v * v^T
            for i in 0..dim {
                for j in 0..dim {
                    deflated[i][j] -= eigenvalue * eigvec[i] * eigvec[j];
                }
            }
            components.push(eigvec);
        }
        components
    }

    /// Serialize to CSV string.
    pub fn to_csv(&self) -> String {
        let mut out = String::new();
        // Header
        out.push_str("test_id");
        for sid in &self.stage_ids {
            out.push(',');
            out.push_str(&sid.0.to_string());
        }
        out.push('\n');
        // Rows
        for (i, row) in self.data.iter().enumerate() {
            out.push_str(&self.test_ids[i].0.to_string());
            for val in row {
                out.push(',');
                out.push_str(&format!("{:.6}", val));
            }
            out.push('\n');
        }
        out
    }

    /// Parse from CSV string.
    pub fn from_csv(csv: &str) -> Result<Self> {
        let mut lines = csv.lines();
        let header = lines
            .next()
            .ok_or_else(|| LocalizerError::validation("validation", "Empty CSV"))?;
        let headers: Vec<&str> = header.split(',').collect();
        if headers.len() < 2 {
            return Err(LocalizerError::validation("validation", "CSV needs at least 2 columns"));
        }
        let stage_ids: Vec<StageId> = headers[1..].iter().map(|s| StageId::new(*s)).collect();
        let num_stages = stage_ids.len();

        let mut data = Vec::new();
        let mut test_ids = Vec::new();
        for line in lines {
            if line.trim().is_empty() {
                continue;
            }
            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() != num_stages + 1 {
                return Err(LocalizerError::validation("validation", format!(
                    "Row has {} columns, expected {}",
                    parts.len(),
                    num_stages + 1
                )));
            }
            test_ids.push(TestCaseId::new(parts[0]));
            let row: std::result::Result<Vec<f64>, _> =
                parts[1..].iter().map(|s| s.trim().parse::<f64>()).collect();
            let row = row.map_err(|e| {
                LocalizerError::validation("validation", format!("Failed to parse float: {}", e))
            })?;
            data.push(row);
        }
        let num_tests = data.len();
        Ok(Self {
            data,
            num_tests,
            num_stages,
            stage_ids,
            test_ids,
        })
    }

    fn covariance_matrix(&self) -> Vec<Vec<f64>> {
        let dim = self.num_stages;
        let means: Vec<f64> = (0..dim)
            .map(|j| self.column_mean(j).unwrap_or(0.0))
            .collect();
        let n = self.num_tests as f64;
        let denom = if n > 1.0 { n - 1.0 } else { 1.0 };
        let mut cov = vec![vec![0.0; dim]; dim];
        for row in &self.data {
            for a in 0..dim {
                for b in a..dim {
                    let v = (row[a] - means[a]) * (row[b] - means[b]);
                    cov[a][b] += v;
                    if a != b {
                        cov[b][a] += v;
                    }
                }
            }
        }
        for a in 0..dim {
            for b in 0..dim {
                cov[a][b] /= denom;
            }
        }
        cov
    }
}

impl fmt::Display for DifferentialMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Header
        write!(f, "{:>12}", "test\\stage")?;
        for sid in &self.stage_ids {
            write!(f, " {:>10}", sid.0.to_string())?;
        }
        writeln!(f)?;
        // Separator
        let total_width = 12 + self.num_stages * 11;
        writeln!(f, "{}", "-".repeat(total_width))?;
        // Rows
        for (i, row) in self.data.iter().enumerate() {
            let tid = self.test_ids[i].0.to_string();
            let label = if tid.len() > 10 { &tid[..10] } else { &tid };
            write!(f, "{:>12}", label)?;
            for val in row {
                write!(f, " {:>10.4}", val)?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

// ── MatrixSlice ─────────────────────────────────────────────────────────────

/// A lightweight view into a sub-region of a DifferentialMatrix.
#[derive(Debug, Clone)]
pub struct MatrixSlice<'a> {
    source: &'a DifferentialMatrix,
    row_indices: Vec<usize>,
    col_indices: Vec<usize>,
}

impl<'a> MatrixSlice<'a> {
    pub fn new(
        source: &'a DifferentialMatrix,
        row_indices: Vec<usize>,
        col_indices: Vec<usize>,
    ) -> Self {
        Self {
            source,
            row_indices,
            col_indices,
        }
    }

    /// Create a slice from row and column ranges.
    pub fn from_ranges(
        source: &'a DifferentialMatrix,
        row_range: std::ops::Range<usize>,
        col_range: std::ops::Range<usize>,
    ) -> Self {
        Self {
            source,
            row_indices: row_range.collect(),
            col_indices: col_range.collect(),
        }
    }

    pub fn get(&self, row: usize, col: usize) -> Option<f64> {
        let r = *self.row_indices.get(row)?;
        let c = *self.col_indices.get(col)?;
        self.source.get(r, c)
    }

    pub fn num_rows(&self) -> usize {
        self.row_indices.len()
    }

    pub fn num_cols(&self) -> usize {
        self.col_indices.len()
    }

    /// Materialize the slice into a new owned DifferentialMatrix.
    pub fn to_matrix(&self) -> DifferentialMatrix {
        let data: Vec<Vec<f64>> = self
            .row_indices
            .iter()
            .map(|&r| {
                self.col_indices
                    .iter()
                    .map(|&c| self.source.data[r][c])
                    .collect()
            })
            .collect();
        let stage_ids: Vec<StageId> = self
            .col_indices
            .iter()
            .map(|&c| self.source.stage_ids[c].clone())
            .collect();
        let test_ids: Vec<TestCaseId> = self
            .row_indices
            .iter()
            .map(|&r| self.source.test_ids[r].clone())
            .collect();
        let num_tests = data.len();
        let num_stages = stage_ids.len();
        DifferentialMatrix {
            data,
            num_tests,
            num_stages,
            stage_ids,
            test_ids,
        }
    }

    /// Column mean of the slice.
    pub fn column_mean(&self, col: usize) -> Option<f64> {
        if col >= self.col_indices.len() {
            return None;
        }
        let c = self.col_indices[col];
        let vals: Vec<f64> = self.row_indices.iter().map(|&r| self.source.data[r][c]).collect();
        if vals.is_empty() {
            return Some(0.0);
        }
        Some(vals.iter().sum::<f64>() / vals.len() as f64)
    }
}

// ── Helper functions ────────────────────────────────────────────────────────

/// Power iteration to find the dominant eigenvector.
fn power_iteration(matrix: &[Vec<f64>], max_iters: usize) -> Vec<f64> {
    let n = matrix.len();
    if n == 0 {
        return Vec::new();
    }
    let mut v: Vec<f64> = (0..n).map(|i| if i == 0 { 1.0 } else { 0.0 }).collect();

    for _ in 0..max_iters {
        let mut new_v = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                new_v[i] += matrix[i][j] * v[j];
            }
        }
        // Normalize
        let norm: f64 = new_v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-15 {
            return v;
        }
        for x in &mut new_v {
            *x /= norm;
        }
        v = new_v;
    }
    v
}

/// Rayleigh quotient: v^T A v / v^T v.
fn rayleigh_quotient(matrix: &[Vec<f64>], v: &[f64]) -> f64 {
    let n = v.len();
    let mut av = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            av[i] += matrix[i][j] * v[j];
        }
    }
    let vtav: f64 = v.iter().zip(av.iter()).map(|(a, b)| a * b).sum();
    let vtv: f64 = v.iter().map(|x| x * x).sum();
    if vtv < 1e-15 {
        0.0
    } else {
        vtav / vtv
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_matrix() -> DifferentialMatrix {
        let stage_ids = vec![
            StageId::new("tok"),
            StageId::new("pos"),
            StageId::new("ner"),
        ];
        let test_ids = vec![
            TestCaseId::new("t1"),
            TestCaseId::new("t2"),
            TestCaseId::new("t3"),
            TestCaseId::new("t4"),
        ];
        let data = vec![
            vec![0.1, 0.2, 0.3],
            vec![0.4, 0.5, 0.6],
            vec![0.7, 0.8, 0.9],
            vec![0.2, 0.3, 0.4],
        ];
        DifferentialMatrix::new(data, stage_ids, test_ids).unwrap()
    }

    #[test]
    fn test_matrix_construction() {
        let m = sample_matrix();
        assert_eq!(m.num_tests, 4);
        assert_eq!(m.num_stages, 3);
    }

    #[test]
    fn test_matrix_get_set() {
        let mut m = sample_matrix();
        assert_eq!(m.get(0, 0), Some(0.1));
        assert_eq!(m.get(2, 2), Some(0.9));
        assert_eq!(m.get(10, 0), None);
        m.set(0, 0, 9.9).unwrap();
        assert_eq!(m.get(0, 0), Some(9.9));
    }

    #[test]
    fn test_matrix_row_column() {
        let m = sample_matrix();
        assert_eq!(m.row(0), Some(&[0.1, 0.2, 0.3][..]));
        let col = m.column(1).unwrap();
        assert_eq!(col, vec![0.2, 0.5, 0.8, 0.3]);
    }

    #[test]
    fn test_matrix_transpose() {
        let m = sample_matrix();
        let t = m.transpose();
        assert_eq!(t.num_tests, 3);
        assert_eq!(t.num_stages, 4);
        assert_eq!(t.get(0, 0), Some(0.1));
        assert_eq!(t.get(2, 0), Some(0.3));
        assert_eq!(t.get(0, 2), Some(0.7));
    }

    #[test]
    fn test_column_stats() {
        let m = sample_matrix();
        let mean = m.column_mean(0).unwrap();
        assert!((mean - 0.35).abs() < 1e-9);
        let std = m.column_std(0).unwrap();
        assert!(std > 0.0);
        let p50 = m.column_percentile(0, 0.5).unwrap();
        assert!(p50 >= 0.1 && p50 <= 0.7);
    }

    #[test]
    fn test_normalize_columns() {
        let mut m = sample_matrix();
        m.normalize_columns();
        // After normalization, each column should have mean ≈ 0
        for j in 0..m.num_stages {
            let mean = m.column_mean(j).unwrap();
            assert!(mean.abs() < 1e-9, "Column {} mean = {}", j, mean);
        }
    }

    #[test]
    fn test_subtract_baseline() {
        let mut m = sample_matrix();
        m.subtract_baseline(&[0.1, 0.2, 0.3]);
        assert!((m.get(0, 0).unwrap() - 0.0).abs() < 1e-9);
        assert!((m.get(0, 1).unwrap() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_filter_violations() {
        let m = sample_matrix();
        let filtered = m.filter_violations(0.5);
        assert_eq!(filtered.num_tests, 2); // rows with ≥1 value > 0.5
    }

    #[test]
    fn test_correlation_matrix() {
        let m = sample_matrix();
        let corr = m.correlation_matrix();
        assert_eq!(corr.len(), 3);
        for i in 0..3 {
            assert!((corr[i][i] - 1.0).abs() < 1e-9);
        }
        // All columns are perfectly correlated (0.1+0.1k)
        assert!((corr[0][1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_csv_roundtrip() {
        let m = sample_matrix();
        let csv = m.to_csv();
        let m2 = DifferentialMatrix::from_csv(&csv).unwrap();
        assert_eq!(m2.num_tests, m.num_tests);
        assert_eq!(m2.num_stages, m.num_stages);
        for i in 0..m.num_tests {
            for j in 0..m.num_stages {
                assert!((m2.data[i][j] - m.data[i][j]).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn test_principal_components() {
        let m = sample_matrix();
        let pcs = m.principal_components(2);
        assert!(pcs.len() <= 2);
        if !pcs.is_empty() {
            let norm: f64 = pcs[0].iter().map(|x| x * x).sum::<f64>().sqrt();
            assert!((norm - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_matrix_display() {
        let m = sample_matrix();
        let display = format!("{}", m);
        assert!(display.contains("tok"));
        assert!(display.contains("t1"));
    }

    #[test]
    fn test_matrix_slice() {
        let m = sample_matrix();
        let slice = MatrixSlice::from_ranges(&m, 0..2, 0..2);
        assert_eq!(slice.num_rows(), 2);
        assert_eq!(slice.num_cols(), 2);
        assert_eq!(slice.get(0, 0), Some(0.1));
        assert_eq!(slice.get(1, 1), Some(0.5));
    }

    #[test]
    fn test_matrix_slice_to_matrix() {
        let m = sample_matrix();
        let slice = MatrixSlice::from_ranges(&m, 1..3, 0..2);
        let sub = slice.to_matrix();
        assert_eq!(sub.num_tests, 2);
        assert_eq!(sub.num_stages, 2);
        assert_eq!(sub.get(0, 0), Some(0.4));
        assert_eq!(sub.get(1, 1), Some(0.8));
    }

    #[test]
    fn test_zeros_matrix() {
        let m = DifferentialMatrix::zeros(
            3,
            2,
            vec![StageId::new("a"), StageId::new("b")],
            vec![
                TestCaseId::new("t1"),
                TestCaseId::new("t2"),
                TestCaseId::new("t3"),
            ],
        );
        assert_eq!(m.get(0, 0), Some(0.0));
        assert_eq!(m.get(2, 1), Some(0.0));
    }

    #[test]
    fn test_power_iteration_identity() {
        let identity = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let v = power_iteration(&identity, 100);
        assert_eq!(v.len(), 2);
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_slice_column_mean() {
        let m = sample_matrix();
        let slice = MatrixSlice::from_ranges(&m, 0..4, 0..3);
        let mean = slice.column_mean(0).unwrap();
        assert!((mean - 0.35).abs() < 1e-9);
    }
}
