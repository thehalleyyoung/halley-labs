//! Error types for matrix decomposition operations.
//!
//! Provides [`DecompError`] covering every failure mode: convergence failures,
//! dimension mismatches, singular matrices, numerical instability, and more.

use thiserror::Error;

/// Result alias for decomposition operations.
pub type DecompResult<T> = std::result::Result<T, DecompError>;

/// Comprehensive error type for matrix decomposition algorithms.
#[derive(Debug, Error)]
pub enum DecompError {
    // ── Dimension errors ────────────────────────────────────────────────
    #[error("Dimension mismatch: expected ({expected_rows}x{expected_cols}), got ({actual_rows}x{actual_cols})")]
    DimensionMismatch {
        expected_rows: usize,
        expected_cols: usize,
        actual_rows: usize,
        actual_cols: usize,
    },

    #[error("Matrix must be square: got ({rows}x{cols})")]
    NotSquare { rows: usize, cols: usize },

    #[error("Incompatible dimensions for {operation}: ({left_rows}x{left_cols}) vs ({right_rows}x{right_cols})")]
    IncompatibleDimensions {
        operation: String,
        left_rows: usize,
        left_cols: usize,
        right_rows: usize,
        right_cols: usize,
    },

    #[error("Vector length mismatch: expected {expected}, got {actual}")]
    VectorLengthMismatch { expected: usize, actual: usize },

    #[error("Empty matrix: {context}")]
    EmptyMatrix { context: String },

    #[error("Index out of bounds: ({row}, {col}) in ({rows}x{cols})")]
    IndexOutOfBounds {
        row: usize,
        col: usize,
        rows: usize,
        cols: usize,
    },

    // ── Singularity / rank errors ───────────────────────────────────────
    #[error("Singular matrix detected: {context}")]
    SingularMatrix { context: String },

    #[error("Near-singular matrix: condition number ~{condition_number:.2e}")]
    NearSingular { condition_number: f64 },

    #[error("Rank deficient: rank {rank} < expected {expected}")]
    RankDeficient { rank: usize, expected: usize },

    #[error("Zero pivot at position {index}: value {value:.2e}")]
    ZeroPivot { index: usize, value: f64 },

    // ── Positive-definiteness ───────────────────────────────────────────
    #[error("Matrix not positive definite: {context}")]
    NotPositiveDefinite { context: String },

    #[error("Matrix not symmetric: max asymmetry {max_diff:.2e} at ({row},{col})")]
    NotSymmetric {
        max_diff: f64,
        row: usize,
        col: usize,
    },

    // ── Convergence failures ────────────────────────────────────────────
    #[error("Convergence failure after {iterations} iterations: {context}")]
    ConvergenceFailure { iterations: usize, context: String },

    #[error("Stagnation detected at iteration {iteration}: residual {residual:.2e}")]
    Stagnation { iteration: usize, residual: f64 },

    #[error("Eigenvalue convergence failure: {converged}/{requested} pairs converged after {iterations} iters")]
    EigenConvergenceFailure {
        converged: usize,
        requested: usize,
        iterations: usize,
    },

    #[error("QR iteration did not converge for element ({row},{col}) after {iterations} iterations")]
    QrIterationFailure {
        row: usize,
        col: usize,
        iterations: usize,
    },

    #[error("SVD convergence failure: {context}")]
    SvdConvergenceFailure { context: String },

    #[error("Lanczos breakdown at step {step}: beta = {beta:.2e}")]
    LanczosBreakdown { step: usize, beta: f64 },

    // ── Numerical instability ───────────────────────────────────────────
    #[error("Numerical instability: {context}")]
    NumericalInstability { context: String },

    #[error("NaN detected in {context}")]
    NanDetected { context: String },

    #[error("Infinity detected in {context}")]
    InfinityDetected { context: String },

    #[error("Loss of orthogonality: max deviation {deviation:.2e}")]
    OrthogonalityLoss { deviation: f64 },

    #[error("Overflow risk: value {value:.2e} in {context}")]
    OverflowRisk { value: f64, context: String },

    // ── Configuration / input errors ────────────────────────────────────
    #[error("Invalid parameter: {name} = {value} ({reason})")]
    InvalidParameter {
        name: String,
        value: String,
        reason: String,
    },

    #[error("Requested {requested} eigenvalues but matrix is only {size}x{size}")]
    TooManyEigenvalues { requested: usize, size: usize },

    #[error("Block size {block_size} exceeds matrix dimension {dim}")]
    BlockSizeTooLarge { block_size: usize, dim: usize },

    #[error("Invalid sparse format: {reason}")]
    InvalidSparseFormat { reason: String },

    // ── Resource limits ─────────────────────────────────────────────────
    #[error("Maximum iterations ({limit}) exceeded")]
    MaxIterationsExceeded { limit: usize },

    #[error("Memory limit exceeded: need ~{needed_mb:.1} MB")]
    MemoryLimitExceeded { needed_mb: f64 },

    // ── Fallthrough ─────────────────────────────────────────────────────
    #[error("Not implemented: {0}")]
    NotImplemented(String),

    #[error("Internal error: {0}")]
    Internal(String),

    #[error(transparent)]
    Spectral(#[from] spectral_types::SpectralError),
}

impl DecompError {
    /// Create a dimension mismatch error for square matrix requirement.
    pub fn require_square(rows: usize, cols: usize) -> Self {
        DecompError::NotSquare { rows, cols }
    }

    /// Create a convergence failure error.
    pub fn convergence(iterations: usize, context: impl Into<String>) -> Self {
        DecompError::ConvergenceFailure {
            iterations,
            context: context.into(),
        }
    }

    /// Create an instability error.
    pub fn instability(context: impl Into<String>) -> Self {
        DecompError::NumericalInstability {
            context: context.into(),
        }
    }

    /// Create a singular matrix error.
    pub fn singular(context: impl Into<String>) -> Self {
        DecompError::SingularMatrix {
            context: context.into(),
        }
    }

    /// Create an empty matrix error.
    pub fn empty(context: impl Into<String>) -> Self {
        DecompError::EmptyMatrix {
            context: context.into(),
        }
    }

    /// Check that a matrix is square, returning an error if not.
    pub fn check_square(rows: usize, cols: usize) -> DecompResult<()> {
        if rows == cols {
            Ok(())
        } else {
            Err(DecompError::NotSquare { rows, cols })
        }
    }

    /// Check that a vector has the expected length.
    pub fn check_vector_len(expected: usize, actual: usize) -> DecompResult<()> {
        if expected == actual {
            Ok(())
        } else {
            Err(DecompError::VectorLengthMismatch { expected, actual })
        }
    }

    /// Check that matrix dimensions are compatible for multiplication.
    pub fn check_mul_dims(
        a_rows: usize,
        a_cols: usize,
        b_rows: usize,
        b_cols: usize,
    ) -> DecompResult<()> {
        if a_cols == b_rows {
            Ok(())
        } else {
            Err(DecompError::IncompatibleDimensions {
                operation: "multiply".into(),
                left_rows: a_rows,
                left_cols: a_cols,
                right_rows: b_rows,
                right_cols: b_cols,
            })
        }
    }

    /// Check that a value is finite.
    pub fn check_finite(value: f64, context: &str) -> DecompResult<()> {
        if value.is_nan() {
            Err(DecompError::NanDetected {
                context: context.into(),
            })
        } else if value.is_infinite() {
            Err(DecompError::InfinityDetected {
                context: context.into(),
            })
        } else {
            Ok(())
        }
    }

    /// Check that a matrix is non-empty.
    pub fn check_non_empty(rows: usize, cols: usize, context: &str) -> DecompResult<()> {
        if rows == 0 || cols == 0 {
            Err(DecompError::EmptyMatrix {
                context: context.into(),
            })
        } else {
            Ok(())
        }
    }
}

impl From<DecompError> for spectral_types::SpectralError {
    fn from(e: DecompError) -> Self {
        spectral_types::SpectralError::Internal(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dimension_mismatch_display() {
        let err = DecompError::DimensionMismatch {
            expected_rows: 3,
            expected_cols: 3,
            actual_rows: 2,
            actual_cols: 4,
        };
        let msg = err.to_string();
        assert!(msg.contains("3x3"));
        assert!(msg.contains("2x4"));
    }

    #[test]
    fn test_not_square_display() {
        let err = DecompError::NotSquare { rows: 3, cols: 5 };
        assert!(msg_contains(&err, "3x5"));
    }

    #[test]
    fn test_singular_display() {
        let err = DecompError::singular("LU pivot");
        assert!(msg_contains(&err, "LU pivot"));
    }

    #[test]
    fn test_convergence_display() {
        let err = DecompError::convergence(100, "QR algorithm");
        let msg = err.to_string();
        assert!(msg.contains("100"));
        assert!(msg.contains("QR algorithm"));
    }

    #[test]
    fn test_check_square_ok() {
        assert!(DecompError::check_square(3, 3).is_ok());
    }

    #[test]
    fn test_check_square_err() {
        assert!(DecompError::check_square(3, 4).is_err());
    }

    #[test]
    fn test_check_vector_len_ok() {
        assert!(DecompError::check_vector_len(5, 5).is_ok());
    }

    #[test]
    fn test_check_vector_len_err() {
        assert!(DecompError::check_vector_len(5, 3).is_err());
    }

    #[test]
    fn test_check_mul_dims_ok() {
        assert!(DecompError::check_mul_dims(3, 4, 4, 5).is_ok());
    }

    #[test]
    fn test_check_mul_dims_err() {
        assert!(DecompError::check_mul_dims(3, 4, 5, 6).is_err());
    }

    #[test]
    fn test_check_finite_ok() {
        assert!(DecompError::check_finite(1.0, "x").is_ok());
    }

    #[test]
    fn test_check_finite_nan() {
        assert!(DecompError::check_finite(f64::NAN, "x").is_err());
    }

    #[test]
    fn test_check_finite_inf() {
        assert!(DecompError::check_finite(f64::INFINITY, "x").is_err());
    }

    #[test]
    fn test_check_non_empty_ok() {
        assert!(DecompError::check_non_empty(3, 3, "test").is_ok());
    }

    #[test]
    fn test_check_non_empty_err() {
        assert!(DecompError::check_non_empty(0, 3, "test").is_err());
    }

    fn msg_contains(err: &DecompError, substr: &str) -> bool {
        err.to_string().contains(substr)
    }
}
