//! Eigenvalue/eigenvector solvers for sparse Laplacian matrices.
//!
//! This module provides:
//! - [`EigenSolver`] — main dispatcher that selects between Lanczos and LOBPCG
//! - [`LanczosSolver`](arpack::LanczosSolver) — implicitly restarted Lanczos method
//! - [`LobpcgSolver`](lobpcg::LobpcgSolver) — Locally Optimal Block PCG
//! - [`EigenCache`] — LRU cache for eigendecomposition results
//! - [`EigenDiagnostics`] — residual checks, orthogonality, convergence reporting

pub mod solver;
pub mod arpack;
pub mod lobpcg;
pub mod cache;
pub mod diagnostics;

pub use solver::EigenSolver;
pub use cache::EigenCache;
pub use diagnostics::EigenDiagnostics;

use spectral_types::dense::DenseMatrix;

// ---------------------------------------------------------------------------
// Shared result / config types
// ---------------------------------------------------------------------------

/// Eigendecomposition result.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EigenResult {
    /// Eigenvalues in ascending order.
    pub eigenvalues: Vec<f64>,
    /// Eigenvectors stored as columns of a dense matrix (n × k).
    pub eigenvectors: DenseMatrix<f64>,
    /// Residual norms for each eigenpair: ‖Av − λv‖.
    pub residuals: Vec<f64>,
    /// Number of iterations used by the solver.
    pub iterations: usize,
    /// Whether the solver converged within tolerance.
    pub converged: bool,
    /// Name of the method that produced this result.
    pub method_used: String,
    /// Wall-clock time in milliseconds.
    pub time_ms: f64,
}

/// Configuration for eigenvalue solvers.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EigenConfig {
    /// Number of eigenvalues to compute (k).
    pub num_eigenvalues: usize,
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Convergence tolerance.
    pub tolerance: f64,
    /// Shift for shift-invert mode.
    pub shift: f64,
    /// Whether to use shift-invert transformation.
    pub use_shift_invert: bool,
    /// Block size for LOBPCG (defaults to num_eigenvalues).
    pub block_size: usize,
    /// Whether to perform full reorthogonalization.
    pub reorthogonalize: bool,
    /// Whether to emit verbose log messages.
    pub verbose: bool,
}

impl Default for EigenConfig {
    fn default() -> Self {
        Self {
            num_eigenvalues: 20,
            max_iter: 1000,
            tolerance: 1e-8,
            shift: 0.0,
            use_shift_invert: false,
            block_size: 20,
            reorthogonalize: true,
            verbose: false,
        }
    }
}

impl EigenConfig {
    /// Create a config requesting `k` eigenvalues with default settings.
    pub fn with_k(k: usize) -> Self {
        Self {
            num_eigenvalues: k,
            block_size: k,
            ..Default::default()
        }
    }

    /// Set the convergence tolerance.
    pub fn tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    /// Set the maximum number of iterations.
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Enable shift-invert mode with the given shift.
    pub fn shift_invert(mut self, sigma: f64) -> Self {
        self.shift = sigma;
        self.use_shift_invert = true;
        self
    }

    /// Set the block size for LOBPCG.
    pub fn block_size(mut self, bs: usize) -> Self {
        self.block_size = bs;
        self
    }

    /// Enable or disable verbose logging.
    pub fn verbose(mut self, v: bool) -> Self {
        self.verbose = v;
        self
    }

    /// Enable or disable full reorthogonalization.
    pub fn reorthogonalize(mut self, r: bool) -> Self {
        self.reorthogonalize = r;
        self
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = EigenConfig::default();
        assert_eq!(cfg.num_eigenvalues, 20);
        assert_eq!(cfg.max_iter, 1000);
        assert!((cfg.tolerance - 1e-8).abs() < 1e-15);
        assert!(!cfg.use_shift_invert);
        assert!(cfg.reorthogonalize);
        assert!(!cfg.verbose);
    }

    #[test]
    fn test_config_with_k() {
        let cfg = EigenConfig::with_k(5);
        assert_eq!(cfg.num_eigenvalues, 5);
        assert_eq!(cfg.block_size, 5);
    }

    #[test]
    fn test_config_builder_chain() {
        let cfg = EigenConfig::with_k(10)
            .tolerance(1e-12)
            .max_iter(500)
            .shift_invert(1.0)
            .verbose(true)
            .reorthogonalize(false)
            .block_size(15);
        assert_eq!(cfg.num_eigenvalues, 10);
        assert!((cfg.tolerance - 1e-12).abs() < 1e-20);
        assert_eq!(cfg.max_iter, 500);
        assert!(cfg.use_shift_invert);
        assert!((cfg.shift - 1.0).abs() < 1e-15);
        assert!(cfg.verbose);
        assert!(!cfg.reorthogonalize);
        assert_eq!(cfg.block_size, 15);
    }

    #[test]
    fn test_config_serialize_roundtrip() {
        let cfg = EigenConfig::with_k(7).tolerance(1e-6);
        let json = serde_json::to_string(&cfg).unwrap();
        let cfg2: EigenConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(cfg.num_eigenvalues, cfg2.num_eigenvalues);
        assert!((cfg.tolerance - cfg2.tolerance).abs() < 1e-20);
    }

    #[test]
    fn test_eigen_result_serialize_roundtrip() {
        let result = EigenResult {
            eigenvalues: vec![1.0, 2.0, 3.0],
            eigenvectors: DenseMatrix::zeros(3, 3),
            residuals: vec![1e-10, 2e-10, 3e-10],
            iterations: 42,
            converged: true,
            method_used: "lanczos".to_string(),
            time_ms: 123.456,
        };
        let json = serde_json::to_string(&result).unwrap();
        let result2: EigenResult = serde_json::from_str(&json).unwrap();
        assert_eq!(result.eigenvalues, result2.eigenvalues);
        assert_eq!(result.iterations, result2.iterations);
        assert_eq!(result.converged, result2.converged);
        assert_eq!(result.method_used, result2.method_used);
    }

    #[test]
    fn test_eigen_result_fields() {
        let result = EigenResult {
            eigenvalues: vec![0.0, 1.5],
            eigenvectors: DenseMatrix::zeros(4, 2),
            residuals: vec![1e-9, 2e-9],
            iterations: 100,
            converged: true,
            method_used: "lobpcg".to_string(),
            time_ms: 50.0,
        };
        assert_eq!(result.eigenvalues.len(), 2);
        assert_eq!(result.eigenvectors.shape(), (4, 2));
        assert_eq!(result.residuals.len(), 2);
    }

    #[test]
    fn test_config_default_shift() {
        let cfg = EigenConfig::default();
        assert!((cfg.shift - 0.0).abs() < 1e-15);
        assert!(!cfg.use_shift_invert);
    }

    #[test]
    fn test_config_block_size_default_matches_k() {
        let cfg = EigenConfig::with_k(15);
        assert_eq!(cfg.block_size, cfg.num_eigenvalues);
    }

    #[test]
    fn test_eigen_config_clone() {
        let cfg = EigenConfig::with_k(5).tolerance(1e-10);
        let cfg2 = cfg.clone();
        assert_eq!(cfg.num_eigenvalues, cfg2.num_eigenvalues);
        assert!((cfg.tolerance - cfg2.tolerance).abs() < 1e-20);
    }

    #[test]
    fn test_eigen_result_clone() {
        let result = EigenResult {
            eigenvalues: vec![1.0],
            eigenvectors: DenseMatrix::zeros(2, 1),
            residuals: vec![1e-10],
            iterations: 10,
            converged: true,
            method_used: "test".to_string(),
            time_ms: 1.0,
        };
        let r2 = result.clone();
        assert_eq!(result.eigenvalues, r2.eigenvalues);
        assert_eq!(result.method_used, r2.method_used);
    }

    #[test]
    fn test_eigen_result_debug() {
        let result = EigenResult {
            eigenvalues: vec![1.0, 2.0],
            eigenvectors: DenseMatrix::zeros(2, 2),
            residuals: vec![0.0, 0.0],
            iterations: 5,
            converged: true,
            method_used: "test".to_string(),
            time_ms: 0.1,
        };
        let dbg = format!("{result:?}");
        assert!(dbg.contains("EigenResult"));
        assert!(dbg.contains("converged: true"));
    }
}
