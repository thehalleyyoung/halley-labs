//! Crate-specific error types for `spectral-core`.
//!
//! [`SpectralCoreError`] wraps domain-specific errors arising from hypergraph
//! construction, Laplacian building, eigensolving, feature extraction, and
//! clustering. It converts from the lower-level [`SpectralError`] and
//! [`MatrixError`] types provided by `spectral-types`.

use thiserror::Error;

use spectral_types::error::{MatrixError, SpectralError};

// ---------------------------------------------------------------------------
// Result alias
// ---------------------------------------------------------------------------

/// Convenience result alias for this crate.
pub type Result<T> = std::result::Result<T, SpectralCoreError>;

// ---------------------------------------------------------------------------
// Core error enum
// ---------------------------------------------------------------------------

/// Error type for all operations in `spectral-core`.
#[derive(Debug, Error)]
pub enum SpectralCoreError {
    /// Failure during constraint-hypergraph construction.
    #[error("Hypergraph construction error: {message}")]
    HypergraphConstruction { message: String },

    /// Failure during Laplacian matrix assembly.
    #[error("Laplacian construction error: {message}")]
    LaplacianConstruction { message: String },

    /// Generic eigensolve failure.
    #[error("Eigensolve error: {message}")]
    Eigensolve { message: String },

    /// The eigensolver did not converge within the allowed budget.
    #[error("Eigensolve convergence failure after {iterations} iterations: residual={residual:.2e}")]
    ConvergenceFailure { iterations: usize, residual: f64 },

    /// Failure during spectral feature extraction.
    #[error("Feature extraction error: {message}")]
    FeatureExtraction { message: String },

    /// Failure during spectral clustering.
    #[error("Clustering error: {message}")]
    Clustering { message: String },

    /// Matrix dimensions did not match what was expected.
    #[error("Matrix dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: String, actual: String },

    /// A user-supplied parameter is out of range or otherwise invalid.
    #[error("Invalid parameter: {name}={value} - {reason}")]
    InvalidParameter {
        name: String,
        value: String,
        reason: String,
    },

    /// Numerical computation became unstable (e.g. NaN propagation).
    #[error("Numerical instability: {context}")]
    NumericalInstability { context: String },

    /// An operation received empty input where non-empty was required.
    #[error("Empty input: {what}")]
    EmptyInput { what: String },

    /// Wall-clock timeout.
    #[error("Timeout after {seconds:.1}s: {context}")]
    Timeout { seconds: f64, context: String },

    /// Caching subsystem error.
    #[error("Cache error: {message}")]
    Cache { message: String },

    /// Transparent wrapper for [`std::io::Error`].
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Transparent wrapper for [`serde_json::Error`].
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Transparent wrapper for upstream [`SpectralError`].
    #[error("Spectral types error: {0}")]
    SpectralTypes(#[from] SpectralError),
}

// ---------------------------------------------------------------------------
// From<MatrixError>
// ---------------------------------------------------------------------------

impl From<MatrixError> for SpectralCoreError {
    fn from(err: MatrixError) -> Self {
        // Route through SpectralError so we keep the full chain.
        SpectralCoreError::SpectralTypes(SpectralError::Matrix(err))
    }
}

// ---------------------------------------------------------------------------
// Convenience constructors
// ---------------------------------------------------------------------------

impl SpectralCoreError {
    /// Shorthand for [`SpectralCoreError::HypergraphConstruction`].
    pub fn hypergraph(msg: impl Into<String>) -> Self {
        Self::HypergraphConstruction {
            message: msg.into(),
        }
    }

    /// Shorthand for [`SpectralCoreError::LaplacianConstruction`].
    pub fn laplacian(msg: impl Into<String>) -> Self {
        Self::LaplacianConstruction {
            message: msg.into(),
        }
    }

    /// Shorthand for [`SpectralCoreError::Eigensolve`].
    pub fn eigensolve(msg: impl Into<String>) -> Self {
        Self::Eigensolve {
            message: msg.into(),
        }
    }

    /// Shorthand for [`SpectralCoreError::ConvergenceFailure`].
    pub fn convergence_failure(iterations: usize, residual: f64) -> Self {
        Self::ConvergenceFailure {
            iterations,
            residual,
        }
    }

    /// Shorthand for [`SpectralCoreError::FeatureExtraction`].
    pub fn feature_extraction(msg: impl Into<String>) -> Self {
        Self::FeatureExtraction {
            message: msg.into(),
        }
    }

    /// Shorthand for [`SpectralCoreError::Clustering`].
    pub fn clustering(msg: impl Into<String>) -> Self {
        Self::Clustering {
            message: msg.into(),
        }
    }

    /// Shorthand for [`SpectralCoreError::DimensionMismatch`].
    pub fn dimension_mismatch(
        expected: impl Into<String>,
        actual: impl Into<String>,
    ) -> Self {
        Self::DimensionMismatch {
            expected: expected.into(),
            actual: actual.into(),
        }
    }

    /// Shorthand for [`SpectralCoreError::InvalidParameter`].
    pub fn invalid_parameter(
        name: impl Into<String>,
        value: impl Into<String>,
        reason: impl Into<String>,
    ) -> Self {
        Self::InvalidParameter {
            name: name.into(),
            value: value.into(),
            reason: reason.into(),
        }
    }

    /// Shorthand for [`SpectralCoreError::NumericalInstability`].
    pub fn numerical_instability(context: impl Into<String>) -> Self {
        Self::NumericalInstability {
            context: context.into(),
        }
    }

    /// Shorthand for [`SpectralCoreError::EmptyInput`].
    pub fn empty_input(what: impl Into<String>) -> Self {
        Self::EmptyInput {
            what: what.into(),
        }
    }

    /// Shorthand for [`SpectralCoreError::Timeout`].
    pub fn timeout(seconds: f64, context: impl Into<String>) -> Self {
        Self::Timeout {
            seconds,
            context: context.into(),
        }
    }

    /// Shorthand for [`SpectralCoreError::Cache`].
    pub fn cache(msg: impl Into<String>) -> Self {
        Self::Cache {
            message: msg.into(),
        }
    }
}

// ---------------------------------------------------------------------------
// Classification helpers
// ---------------------------------------------------------------------------

impl SpectralCoreError {
    /// Returns `true` for errors that may succeed on retry or with relaxed
    /// parameters (convergence failures and timeouts).
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self,
            Self::ConvergenceFailure { .. } | Self::Timeout { .. }
        )
    }

    /// Returns a short, machine-readable error code suitable for metrics and
    /// log filtering.
    pub fn error_code(&self) -> &'static str {
        match self {
            Self::HypergraphConstruction { .. } => "E_HYPER",
            Self::LaplacianConstruction { .. } => "E_LAPL",
            Self::Eigensolve { .. } => "E_EIGEN",
            Self::ConvergenceFailure { .. } => "E_CONV",
            Self::FeatureExtraction { .. } => "E_FEAT",
            Self::Clustering { .. } => "E_CLUST",
            Self::DimensionMismatch { .. } => "E_DIM",
            Self::InvalidParameter { .. } => "E_PARAM",
            Self::NumericalInstability { .. } => "E_NUM",
            Self::EmptyInput { .. } => "E_EMPTY",
            Self::Timeout { .. } => "E_TIMEOUT",
            Self::Cache { .. } => "E_CACHE",
            Self::Io(_) => "E_IO",
            Self::Serialization(_) => "E_SERDE",
            Self::SpectralTypes(_) => "E_TYPES",
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Constructor smoke tests -------------------------------------------

    #[test]
    fn test_hypergraph_constructor() {
        let err = SpectralCoreError::hypergraph("bad edge list");
        assert!(matches!(err, SpectralCoreError::HypergraphConstruction { .. }));
        assert!(err.to_string().contains("bad edge list"));
    }

    #[test]
    fn test_laplacian_constructor() {
        let err = SpectralCoreError::laplacian("negative weight");
        assert!(matches!(err, SpectralCoreError::LaplacianConstruction { .. }));
        assert!(err.to_string().contains("negative weight"));
    }

    #[test]
    fn test_eigensolve_constructor() {
        let err = SpectralCoreError::eigensolve("zero matrix");
        assert!(matches!(err, SpectralCoreError::Eigensolve { .. }));
        assert!(err.to_string().contains("zero matrix"));
    }

    #[test]
    fn test_convergence_failure_constructor() {
        let err = SpectralCoreError::convergence_failure(500, 1.2e-4);
        assert!(matches!(err, SpectralCoreError::ConvergenceFailure { iterations: 500, .. }));
        let msg = err.to_string();
        assert!(msg.contains("500"));
        assert!(msg.contains("1.20e-4"));
    }

    #[test]
    fn test_feature_extraction_constructor() {
        let err = SpectralCoreError::feature_extraction("missing eigenvectors");
        assert!(matches!(err, SpectralCoreError::FeatureExtraction { .. }));
        assert!(err.to_string().contains("missing eigenvectors"));
    }

    #[test]
    fn test_clustering_constructor() {
        let err = SpectralCoreError::clustering("k larger than n");
        assert!(matches!(err, SpectralCoreError::Clustering { .. }));
        assert!(err.to_string().contains("k larger than n"));
    }

    #[test]
    fn test_dimension_mismatch_constructor() {
        let err = SpectralCoreError::dimension_mismatch("100x100", "50x50");
        let msg = err.to_string();
        assert!(msg.contains("100x100"));
        assert!(msg.contains("50x50"));
    }

    #[test]
    fn test_invalid_parameter_constructor() {
        let err = SpectralCoreError::invalid_parameter("k", "0", "must be >= 1");
        let msg = err.to_string();
        assert!(msg.contains("k=0"));
        assert!(msg.contains("must be >= 1"));
    }

    #[test]
    fn test_numerical_instability_constructor() {
        let err = SpectralCoreError::numerical_instability("NaN in row 3");
        assert!(err.to_string().contains("NaN in row 3"));
    }

    #[test]
    fn test_empty_input_constructor() {
        let err = SpectralCoreError::empty_input("constraint matrix");
        assert!(err.to_string().contains("constraint matrix"));
    }

    #[test]
    fn test_timeout_constructor() {
        let err = SpectralCoreError::timeout(30.5, "eigensolve");
        let msg = err.to_string();
        assert!(msg.contains("30.5"));
        assert!(msg.contains("eigensolve"));
    }

    #[test]
    fn test_cache_constructor() {
        let err = SpectralCoreError::cache("corrupted entry");
        assert!(err.to_string().contains("corrupted entry"));
    }

    // -- Error codes -------------------------------------------------------

    #[test]
    fn test_error_codes_unique_and_prefixed() {
        let variants: Vec<SpectralCoreError> = vec![
            SpectralCoreError::hypergraph(""),
            SpectralCoreError::laplacian(""),
            SpectralCoreError::eigensolve(""),
            SpectralCoreError::convergence_failure(0, 0.0),
            SpectralCoreError::feature_extraction(""),
            SpectralCoreError::clustering(""),
            SpectralCoreError::dimension_mismatch("", ""),
            SpectralCoreError::invalid_parameter("", "", ""),
            SpectralCoreError::numerical_instability(""),
            SpectralCoreError::empty_input(""),
            SpectralCoreError::timeout(0.0, ""),
            SpectralCoreError::cache(""),
        ];

        let mut codes: Vec<&str> = variants.iter().map(|e| e.error_code()).collect();
        let len_before = codes.len();
        codes.sort();
        codes.dedup();
        assert_eq!(codes.len(), len_before, "error codes must be unique");

        for code in &codes {
            assert!(code.starts_with("E_"), "code {code} must start with E_");
        }
    }

    #[test]
    fn test_specific_error_codes() {
        assert_eq!(SpectralCoreError::hypergraph("").error_code(), "E_HYPER");
        assert_eq!(SpectralCoreError::laplacian("").error_code(), "E_LAPL");
        assert_eq!(SpectralCoreError::eigensolve("").error_code(), "E_EIGEN");
        assert_eq!(
            SpectralCoreError::convergence_failure(0, 0.0).error_code(),
            "E_CONV"
        );
        assert_eq!(
            SpectralCoreError::feature_extraction("").error_code(),
            "E_FEAT"
        );
        assert_eq!(SpectralCoreError::clustering("").error_code(), "E_CLUST");
        assert_eq!(
            SpectralCoreError::dimension_mismatch("", "").error_code(),
            "E_DIM"
        );
        assert_eq!(
            SpectralCoreError::invalid_parameter("", "", "").error_code(),
            "E_PARAM"
        );
        assert_eq!(
            SpectralCoreError::numerical_instability("").error_code(),
            "E_NUM"
        );
        assert_eq!(SpectralCoreError::empty_input("").error_code(), "E_EMPTY");
        assert_eq!(
            SpectralCoreError::timeout(0.0, "").error_code(),
            "E_TIMEOUT"
        );
        assert_eq!(SpectralCoreError::cache("").error_code(), "E_CACHE");
    }

    // -- Recoverability ----------------------------------------------------

    #[test]
    fn test_convergence_failure_is_recoverable() {
        let err = SpectralCoreError::convergence_failure(100, 1e-3);
        assert!(err.is_recoverable());
    }

    #[test]
    fn test_timeout_is_recoverable() {
        let err = SpectralCoreError::timeout(60.0, "laplacian eigensolve");
        assert!(err.is_recoverable());
    }

    #[test]
    fn test_non_recoverable_errors() {
        let non_recoverable = vec![
            SpectralCoreError::hypergraph("bad"),
            SpectralCoreError::laplacian("bad"),
            SpectralCoreError::eigensolve("bad"),
            SpectralCoreError::feature_extraction("bad"),
            SpectralCoreError::clustering("bad"),
            SpectralCoreError::dimension_mismatch("a", "b"),
            SpectralCoreError::invalid_parameter("k", "0", "r"),
            SpectralCoreError::numerical_instability("nan"),
            SpectralCoreError::empty_input("vec"),
            SpectralCoreError::cache("miss"),
        ];
        for err in &non_recoverable {
            assert!(
                !err.is_recoverable(),
                "{} should not be recoverable",
                err.error_code()
            );
        }
    }

    // -- From conversions --------------------------------------------------

    #[test]
    fn test_from_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file missing");
        let core_err: SpectralCoreError = io_err.into();
        assert!(matches!(core_err, SpectralCoreError::Io(_)));
        assert_eq!(core_err.error_code(), "E_IO");
        assert!(!core_err.is_recoverable());
    }

    #[test]
    fn test_from_serde_error() {
        let json_err = serde_json::from_str::<serde_json::Value>("not json").unwrap_err();
        let core_err: SpectralCoreError = json_err.into();
        assert!(matches!(core_err, SpectralCoreError::Serialization(_)));
        assert_eq!(core_err.error_code(), "E_SERDE");
    }

    #[test]
    fn test_from_matrix_error() {
        let mat_err = MatrixError::NotSquare { rows: 3, cols: 5 };
        let core_err: SpectralCoreError = mat_err.into();
        assert!(matches!(core_err, SpectralCoreError::SpectralTypes(_)));
        assert_eq!(core_err.error_code(), "E_TYPES");
        assert!(core_err.to_string().contains("3"));
    }

    #[test]
    fn test_from_spectral_error() {
        let se = SpectralError::Internal("boom".into());
        let core_err: SpectralCoreError = se.into();
        assert!(matches!(core_err, SpectralCoreError::SpectralTypes(_)));
        assert!(core_err.to_string().contains("boom"));
    }

    // -- Display formatting ------------------------------------------------

    #[test]
    fn test_display_formats() {
        let err = SpectralCoreError::convergence_failure(1000, 2.5e-6);
        let msg = format!("{err}");
        assert!(msg.starts_with("Eigensolve convergence failure"));
        assert!(msg.contains("1000"));
        assert!(msg.contains("2.50e-6"));

        let err = SpectralCoreError::timeout(12.345, "Lanczos");
        let msg = format!("{err}");
        assert!(msg.contains("12.3"));
        assert!(msg.contains("Lanczos"));
    }

    // -- Result alias ------------------------------------------------------

    #[test]
    fn test_result_alias_ok() {
        let r: Result<i32> = Ok(42);
        assert_eq!(r.unwrap(), 42);
    }

    #[test]
    fn test_result_alias_err() {
        let r: Result<i32> = Err(SpectralCoreError::empty_input("nothing"));
        assert!(r.is_err());
    }

    // -- Debug impl -------------------------------------------------------

    #[test]
    fn test_debug_impl() {
        let err = SpectralCoreError::eigensolve("test debug");
        let debug = format!("{err:?}");
        assert!(debug.contains("Eigensolve"));
        assert!(debug.contains("test debug"));
    }
}
