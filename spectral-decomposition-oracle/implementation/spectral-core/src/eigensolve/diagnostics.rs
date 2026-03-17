//! Eigensolve diagnostics: residual checks, orthogonality, spectral gap analysis.

use spectral_types::dense::DenseMatrix;
use spectral_types::sparse::CsrMatrix;

use crate::eigensolve::EigenResult;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Comprehensive diagnostics report for an eigendecomposition.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DiagnosticsReport {
    /// Residual norm ‖Av_i − λ_i v_i‖ for each eigenpair.
    pub residual_norms: Vec<f64>,
    /// Maximum residual across all eigenpairs.
    pub max_residual: f64,
    /// Maximum off-diagonal entry of V^T V (orthogonality error).
    pub orthogonality_error: f64,
    /// Whether eigenvalues are in ascending order.
    pub is_ordered: bool,
    /// Gap between the second and first eigenvalue, if at least two exist.
    pub spectral_gap: Option<f64>,
    /// Number of eigenpairs whose residual is below tolerance.
    pub num_converged: usize,
    /// Groups of eigenvalue indices that are clustered within tolerance.
    pub eigenvalue_clusters: Vec<Vec<usize>>,
    /// Condition number |λ_max / λ_i| for each eigenvalue.
    pub condition_numbers: Vec<f64>,
}

/// History of convergence over iterations.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ConvergenceHistory {
    /// Iteration numbers at which snapshots were taken.
    pub iteration: Vec<usize>,
    /// Residual norms at each snapshot (one inner vec per eigenvalue).
    pub residuals: Vec<Vec<f64>>,
    /// Eigenvalue estimates at each snapshot.
    pub eigenvalue_estimates: Vec<Vec<f64>>,
}

// ---------------------------------------------------------------------------
// EigenDiagnostics
// ---------------------------------------------------------------------------

/// Static helper methods for analysing eigendecomposition quality.
pub struct EigenDiagnostics;

impl EigenDiagnostics {
    /// Compute residual norms ‖Av_i − λ_i v_i‖ for every eigenpair.
    pub fn compute_residuals(
        matrix: &CsrMatrix<f64>,
        eigenvalues: &[f64],
        eigenvectors: &DenseMatrix<f64>,
    ) -> Vec<f64> {
        let (n, k) = eigenvectors.shape();
        let mut residuals = Vec::with_capacity(k);

        for j in 0..k {
            // Extract column j
            let v: Vec<f64> = (0..n).map(|i| eigenvectors.data[i * k + j]).collect();

            // Av
            let av = match matrix.mul_vec(&v) {
                Ok(r) => r,
                Err(_) => {
                    residuals.push(f64::INFINITY);
                    continue;
                }
            };

            let lambda = if j < eigenvalues.len() {
                eigenvalues[j]
            } else {
                0.0
            };

            // ‖Av − λv‖
            let norm: f64 = av
                .iter()
                .zip(v.iter())
                .map(|(a, vi)| {
                    let d = a - lambda * vi;
                    d * d
                })
                .sum::<f64>()
                .sqrt();

            residuals.push(norm);
        }

        residuals
    }

    /// Check orthogonality: returns max |V^T V − I| off-diagonal entry.
    pub fn check_orthogonality(eigenvectors: &DenseMatrix<f64>) -> f64 {
        let (n, k) = eigenvectors.shape();
        let mut max_off_diag: f64 = 0.0;

        for i in 0..k {
            for j in 0..k {
                let dot: f64 = (0..n)
                    .map(|r| eigenvectors.data[r * k + i] * eigenvectors.data[r * k + j])
                    .sum();

                let expected = if i == j { 1.0 } else { 0.0 };
                let err = (dot - expected).abs();
                if i != j {
                    max_off_diag = max_off_diag.max(err);
                }
            }
        }

        max_off_diag
    }

    /// Check whether eigenvalues are sorted in ascending order.
    pub fn check_ordering(eigenvalues: &[f64]) -> bool {
        eigenvalues.windows(2).all(|w| w[0] <= w[1])
    }

    /// Spectral gap λ₂ − λ₁ (requires at least two eigenvalues).
    pub fn spectral_gap(eigenvalues: &[f64]) -> Option<f64> {
        if eigenvalues.len() < 2 {
            return None;
        }
        Some(eigenvalues[1] - eigenvalues[0])
    }

    /// All consecutive gaps λ_{i+1} − λ_i.
    pub fn eigenvalue_gaps(eigenvalues: &[f64]) -> Vec<f64> {
        eigenvalues.windows(2).map(|w| w[1] - w[0]).collect()
    }

    /// Group eigenvalue indices that are within `tol` of each other.
    pub fn cluster_analysis(eigenvalues: &[f64], tol: f64) -> Vec<Vec<usize>> {
        if eigenvalues.is_empty() {
            return Vec::new();
        }

        let mut clusters: Vec<Vec<usize>> = Vec::new();
        let mut current_cluster = vec![0usize];

        for i in 1..eigenvalues.len() {
            if (eigenvalues[i] - eigenvalues[i - 1]).abs() <= tol {
                current_cluster.push(i);
            } else {
                clusters.push(std::mem::take(&mut current_cluster));
                current_cluster.push(i);
            }
        }
        clusters.push(current_cluster);

        clusters
    }

    /// Condition numbers |λ_max / λ_i| for each eigenvalue.
    pub fn condition_numbers(eigenvalues: &[f64]) -> Vec<f64> {
        if eigenvalues.is_empty() {
            return Vec::new();
        }

        let lambda_max = eigenvalues
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max);

        eigenvalues
            .iter()
            .map(|&lam| {
                if lam.abs() < 1e-15 {
                    f64::INFINITY
                } else {
                    lambda_max / lam.abs()
                }
            })
            .collect()
    }

    /// Generate a full diagnostics report.
    pub fn generate_report(
        result: &EigenResult,
        matrix: &CsrMatrix<f64>,
    ) -> DiagnosticsReport {
        let residual_norms =
            Self::compute_residuals(matrix, &result.eigenvalues, &result.eigenvectors);

        let max_residual = residual_norms
            .iter()
            .copied()
            .fold(0.0_f64, f64::max);

        let orthogonality_error = Self::check_orthogonality(&result.eigenvectors);
        let is_ordered = Self::check_ordering(&result.eigenvalues);
        let spectral_gap = Self::spectral_gap(&result.eigenvalues);

        let num_converged = residual_norms
            .iter()
            .zip(result.eigenvalues.iter())
            .filter(|(&r, &lam)| {
                let threshold = 1e-8 * lam.abs().max(1.0);
                r < threshold
            })
            .count();

        let eigenvalue_clusters = Self::cluster_analysis(&result.eigenvalues, 1e-6);
        let condition_numbers = Self::condition_numbers(&result.eigenvalues);

        DiagnosticsReport {
            residual_norms,
            max_residual,
            orthogonality_error,
            is_ordered,
            spectral_gap,
            num_converged,
            eigenvalue_clusters,
            condition_numbers,
        }
    }
}

impl ConvergenceHistory {
    /// Create an empty convergence history.
    pub fn new() -> Self {
        Self {
            iteration: Vec::new(),
            residuals: Vec::new(),
            eigenvalue_estimates: Vec::new(),
        }
    }

    /// Record a snapshot at the given iteration.
    pub fn record(&mut self, iter: usize, residuals: Vec<f64>, estimates: Vec<f64>) {
        self.iteration.push(iter);
        self.residuals.push(residuals);
        self.eigenvalue_estimates.push(estimates);
    }

    /// Number of snapshots recorded.
    pub fn len(&self) -> usize {
        self.iteration.len()
    }

    /// Whether any snapshots have been recorded.
    pub fn is_empty(&self) -> bool {
        self.iteration.is_empty()
    }

    /// Final residual norms, or empty if no snapshots exist.
    pub fn final_residuals(&self) -> &[f64] {
        self.residuals.last().map_or(&[], |v| v.as_slice())
    }
}

impl Default for ConvergenceHistory {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a small diagonal CSR matrix with given diagonal entries.
    fn diagonal_csr(diag: &[f64]) -> CsrMatrix<f64> {
        let n = diag.len();
        let mut row_ptr = Vec::with_capacity(n + 1);
        let mut col_ind = Vec::with_capacity(n);
        let mut values = Vec::with_capacity(n);
        row_ptr.push(0);
        for (i, &d) in diag.iter().enumerate() {
            col_ind.push(i);
            values.push(d);
            row_ptr.push(i + 1);
        }
        CsrMatrix::new(n, n, row_ptr, col_ind, values).unwrap()
    }

    /// Build a DenseMatrix whose columns are the standard basis vectors.
    fn identity_eigenvectors(n: usize) -> DenseMatrix<f64> {
        DenseMatrix::identity(n)
    }

    #[test]
    fn test_residuals_identity() {
        let mat = CsrMatrix::identity(3);
        let evals = vec![1.0, 1.0, 1.0];
        let evecs = identity_eigenvectors(3);
        let res = EigenDiagnostics::compute_residuals(&mat, &evals, &evecs);
        assert_eq!(res.len(), 3);
        for r in &res {
            assert!(*r < 1e-14, "residual too large: {r}");
        }
    }

    #[test]
    fn test_residuals_diagonal() {
        let mat = diagonal_csr(&[2.0, 5.0, 8.0]);
        let evals = vec![2.0, 5.0, 8.0];
        let evecs = identity_eigenvectors(3);
        let res = EigenDiagnostics::compute_residuals(&mat, &evals, &evecs);
        for r in &res {
            assert!(*r < 1e-14, "residual too large: {r}");
        }
    }

    #[test]
    fn test_residuals_nonzero_for_wrong_eigenvalues() {
        let mat = diagonal_csr(&[2.0, 5.0]);
        let evals = vec![3.0, 6.0]; // wrong
        let evecs = identity_eigenvectors(2);
        let res = EigenDiagnostics::compute_residuals(&mat, &evals, &evecs);
        assert!((res[0] - 1.0).abs() < 1e-14);
        assert!((res[1] - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_orthogonality_identity() {
        let evecs = identity_eigenvectors(4);
        let err = EigenDiagnostics::check_orthogonality(&evecs);
        assert!(err < 1e-14);
    }

    #[test]
    fn test_check_ordering_ascending() {
        assert!(EigenDiagnostics::check_ordering(&[1.0, 2.0, 3.0]));
        assert!(EigenDiagnostics::check_ordering(&[1.0, 1.0, 2.0]));
        assert!(EigenDiagnostics::check_ordering(&[]));
        assert!(EigenDiagnostics::check_ordering(&[5.0]));
    }

    #[test]
    fn test_check_ordering_not_ascending() {
        assert!(!EigenDiagnostics::check_ordering(&[3.0, 2.0, 1.0]));
        assert!(!EigenDiagnostics::check_ordering(&[1.0, 3.0, 2.0]));
    }

    #[test]
    fn test_spectral_gap() {
        assert_eq!(EigenDiagnostics::spectral_gap(&[1.0, 3.0, 7.0]), Some(2.0));
        assert_eq!(EigenDiagnostics::spectral_gap(&[5.0]), None);
        assert_eq!(EigenDiagnostics::spectral_gap(&[]), None);
    }

    #[test]
    fn test_eigenvalue_gaps() {
        let gaps = EigenDiagnostics::eigenvalue_gaps(&[1.0, 3.0, 7.0]);
        assert_eq!(gaps.len(), 2);
        assert!((gaps[0] - 2.0).abs() < 1e-14);
        assert!((gaps[1] - 4.0).abs() < 1e-14);
    }

    #[test]
    fn test_cluster_analysis() {
        let evals = vec![1.0, 1.0 + 1e-7, 3.0, 3.0 + 1e-8, 5.0];
        let clusters = EigenDiagnostics::cluster_analysis(&evals, 1e-6);
        assert_eq!(clusters.len(), 3);
        assert_eq!(clusters[0], vec![0, 1]);
        assert_eq!(clusters[1], vec![2, 3]);
        assert_eq!(clusters[2], vec![4]);
    }

    #[test]
    fn test_cluster_analysis_empty() {
        let clusters = EigenDiagnostics::cluster_analysis(&[], 1e-6);
        assert!(clusters.is_empty());
    }

    #[test]
    fn test_condition_numbers() {
        let conds = EigenDiagnostics::condition_numbers(&[1.0, 2.0, 4.0]);
        assert!((conds[0] - 4.0).abs() < 1e-14);
        assert!((conds[1] - 2.0).abs() < 1e-14);
        assert!((conds[2] - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_condition_numbers_with_zero() {
        let conds = EigenDiagnostics::condition_numbers(&[0.0, 1.0, 2.0]);
        assert!(conds[0].is_infinite());
        assert!((conds[1] - 2.0).abs() < 1e-14);
    }

    #[test]
    fn test_generate_report() {
        let mat = diagonal_csr(&[1.0, 2.0, 3.0]);
        let result = EigenResult {
            eigenvalues: vec![1.0, 2.0, 3.0],
            eigenvectors: identity_eigenvectors(3),
            residuals: vec![0.0, 0.0, 0.0],
            iterations: 10,
            converged: true,
            method_used: "test".to_string(),
            time_ms: 1.0,
        };
        let report = EigenDiagnostics::generate_report(&result, &mat);
        assert!(report.max_residual < 1e-14);
        assert!(report.orthogonality_error < 1e-14);
        assert!(report.is_ordered);
        assert_eq!(report.spectral_gap, Some(1.0));
        assert_eq!(report.num_converged, 3);
    }

    #[test]
    fn test_convergence_history() {
        let mut hist = ConvergenceHistory::new();
        assert!(hist.is_empty());
        hist.record(0, vec![1.0, 2.0], vec![0.5, 1.5]);
        hist.record(1, vec![0.1, 0.2], vec![0.9, 1.9]);
        assert_eq!(hist.len(), 2);
        assert!(!hist.is_empty());
        assert_eq!(hist.final_residuals(), &[0.1, 0.2]);
    }

    #[test]
    fn test_diagnostics_report_serialize() {
        let report = DiagnosticsReport {
            residual_norms: vec![1e-10],
            max_residual: 1e-10,
            orthogonality_error: 1e-15,
            is_ordered: true,
            spectral_gap: Some(1.0),
            num_converged: 1,
            eigenvalue_clusters: vec![vec![0]],
            condition_numbers: vec![1.0],
        };
        let json = serde_json::to_string(&report).unwrap();
        let report2: DiagnosticsReport = serde_json::from_str(&json).unwrap();
        assert_eq!(report.num_converged, report2.num_converged);
    }
}
