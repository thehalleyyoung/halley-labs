//! Comprehensive error types for the spectral decomposition oracle.
//!
//! Provides a hierarchy of domain-specific errors using `thiserror`,
//! covering every subsystem: spectral analysis, matrix operations,
//! decomposition, oracle prediction, optimization, certificates, I/O, and configuration.

use std::fmt;
use thiserror::Error;

/// Top-level result alias using [`SpectralError`].
pub type Result<T> = std::result::Result<T, SpectralError>;

/// Result alias for matrix operations.
pub type MatrixResult<T> = std::result::Result<T, MatrixError>;

/// Result alias for decomposition operations.
pub type DecompositionResult<T> = std::result::Result<T, DecompositionError>;

/// Result alias for oracle operations.
pub type OracleResult<T> = std::result::Result<T, OracleError>;

/// Result alias for optimization operations.
pub type OptimizationResult<T> = std::result::Result<T, OptimizationError>;

/// Result alias for I/O operations.
pub type IoResult<T> = std::result::Result<T, IoError>;

/// Result alias for configuration operations.
pub type ConfigResult<T> = std::result::Result<T, ConfigError>;

/// Top-level error type encompassing all subsystem errors.
#[derive(Debug, Error)]
pub enum SpectralError {
    #[error("Matrix error: {0}")]
    Matrix(#[from] MatrixError),

    #[error("Decomposition error: {0}")]
    Decomposition(#[from] DecompositionError),

    #[error("Oracle error: {0}")]
    Oracle(#[from] OracleError),

    #[error("Optimization error: {0}")]
    Optimization(#[from] OptimizationError),

    #[error("Certificate error: {0}")]
    Certificate(#[from] CertificateError),

    #[error("I/O error: {0}")]
    Io(#[from] IoError),

    #[error("Configuration error: {0}")]
    Config(#[from] ConfigError),

    #[error("Validation error: {0}")]
    Validation(#[from] ValidationError),

    #[error("Feature extraction error: {0}")]
    FeatureExtraction(String),

    #[error("Graph construction error: {0}")]
    GraphConstruction(String),

    #[error("Partition error: {0}")]
    Partition(String),

    #[error("Internal error: {0}")]
    Internal(String),

    #[error("Not implemented: {0}")]
    NotImplemented(String),

    #[error("Timeout after {seconds} seconds: {context}")]
    Timeout { seconds: f64, context: String },

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

/// Errors arising from matrix operations (sparse and dense).
#[derive(Debug, Error)]
pub enum MatrixError {
    #[error("Dimension mismatch: expected ({expected_rows}x{expected_cols}), got ({actual_rows}x{actual_cols})")]
    DimensionMismatch {
        expected_rows: usize,
        expected_cols: usize,
        actual_rows: usize,
        actual_cols: usize,
    },

    #[error("Incompatible dimensions for {operation}: left ({left_rows}x{left_cols}), right ({right_rows}x{right_cols})")]
    IncompatibleDimensions {
        operation: String,
        left_rows: usize,
        left_cols: usize,
        right_rows: usize,
        right_cols: usize,
    },

    #[error("Index out of bounds: ({row}, {col}) in matrix of size ({rows}x{cols})")]
    IndexOutOfBounds {
        row: usize,
        col: usize,
        rows: usize,
        cols: usize,
    },

    #[error("Singular matrix: {context}")]
    SingularMatrix { context: String },

    #[error("Not positive definite: {context}")]
    NotPositiveDefinite { context: String },

    #[error("Not symmetric: maximum asymmetry {max_diff} at ({row}, {col})")]
    NotSymmetric {
        max_diff: f64,
        row: usize,
        col: usize,
    },

    #[error("Invalid sparse format: {reason}")]
    InvalidSparseFormat { reason: String },

    #[error("Duplicate entry at ({row}, {col})")]
    DuplicateEntry { row: usize, col: usize },

    #[error("NaN or Infinity detected in matrix at ({row}, {col})")]
    NonFiniteValue { row: usize, col: usize },

    #[error("Empty matrix: {context}")]
    EmptyMatrix { context: String },

    #[error("Matrix is not square: ({rows}x{cols})")]
    NotSquare { rows: usize, cols: usize },

    #[error("Convergence failure after {iterations} iterations: {context}")]
    ConvergenceFailure { iterations: usize, context: String },

    #[error("Zero diagonal entry at position {index}")]
    ZeroDiagonal { index: usize },

    #[error("Invalid reshape: cannot reshape {from_rows}x{from_cols} into {to_rows}x{to_cols}")]
    InvalidReshape {
        from_rows: usize,
        from_cols: usize,
        to_rows: usize,
        to_cols: usize,
    },
}

/// Errors from decomposition detection and application.
#[derive(Debug, Error)]
pub enum DecompositionError {
    #[error("No decomposable structure detected: {reason}")]
    NoStructureDetected { reason: String },

    #[error("Invalid block structure: {reason}")]
    InvalidBlockStructure { reason: String },

    #[error("Block count mismatch: expected {expected}, got {actual}")]
    BlockCountMismatch { expected: usize, actual: usize },

    #[error("Decomposition infeasible: {reason}")]
    Infeasible { reason: String },

    #[error("Subproblem failure in block {block}: {reason}")]
    SubproblemFailure { block: usize, reason: String },

    #[error("Master problem failure: {reason}")]
    MasterProblemFailure { reason: String },

    #[error("Cut generation failure: {reason}")]
    CutGenerationFailure { reason: String },

    #[error("Column generation failure: {reason}")]
    ColumnGenerationFailure { reason: String },

    #[error("Dual bound divergence: {value}")]
    DualBoundDivergence { value: f64 },

    #[error("Gap did not close: final gap {gap:.6} after {iterations} iterations")]
    GapNotClosed { gap: f64, iterations: usize },

    #[error("Invalid partition for decomposition: {reason}")]
    InvalidPartition { reason: String },

    #[error("Linking constraint violation: constraint {index}, violation {violation:.6}")]
    LinkingConstraintViolation { index: usize, violation: f64 },
}

/// Errors from the oracle prediction system.
#[derive(Debug, Error)]
pub enum OracleError {
    #[error("Model not trained: {context}")]
    ModelNotTrained { context: String },

    #[error("Feature dimension mismatch: model expects {expected}, got {actual}")]
    FeatureDimensionMismatch { expected: usize, actual: usize },

    #[error("Invalid prediction: {reason}")]
    InvalidPrediction { reason: String },

    #[error("Confidence below threshold: {confidence:.4} < {threshold:.4}")]
    LowConfidence { confidence: f64, threshold: f64 },

    #[error("Classification ambiguous: top-2 margin {margin:.4}")]
    AmbiguousClassification { margin: f64 },

    #[error("Model loading error: {reason}")]
    ModelLoadError { reason: String },

    #[error("Model serialization error: {reason}")]
    ModelSerializationError { reason: String },

    #[error("Training data insufficient: {count} samples, minimum {minimum}")]
    InsufficientTrainingData { count: usize, minimum: usize },

    #[error("Feature contains NaN: feature index {index}, name '{name}'")]
    NanFeature { index: usize, name: String },

    #[error("Calibration error: {reason}")]
    CalibrationError { reason: String },
}

/// Errors from MIP optimization.
#[derive(Debug, Error)]
pub enum OptimizationError {
    #[error("Problem is infeasible: {reason}")]
    Infeasible { reason: String },

    #[error("Problem is unbounded: {reason}")]
    Unbounded { reason: String },

    #[error("Solver error: {solver} returned code {code}: {message}")]
    SolverError {
        solver: String,
        code: i32,
        message: String,
    },

    #[error("Numerical instability: {context}")]
    NumericalInstability { context: String },

    #[error("Time limit exceeded: {elapsed:.2}s > {limit:.2}s")]
    TimeLimitExceeded { elapsed: f64, limit: f64 },

    #[error("Iteration limit exceeded: {iterations} > {limit}")]
    IterationLimitExceeded { iterations: usize, limit: usize },

    #[error("Node limit exceeded: {nodes} > {limit}")]
    NodeLimitExceeded { nodes: usize, limit: usize },

    #[error("Memory limit exceeded: {used_mb:.1} MB > {limit_mb:.1} MB")]
    MemoryLimitExceeded { used_mb: f64, limit_mb: f64 },

    #[error("Invalid bound: variable {variable}, lb={lb} > ub={ub}")]
    InvalidBound {
        variable: usize,
        lb: f64,
        ub: f64,
    },

    #[error("Objective value is non-finite: {value}")]
    NonFiniteObjective { value: f64 },

    #[error("Solution violates constraint {index} by {violation:.6}")]
    ConstraintViolation { index: usize, violation: f64 },

    #[error("Integrality violation: variable {variable}, value {value:.6}")]
    IntegralityViolation { variable: usize, value: f64 },
}

/// Errors related to decomposition quality certificates.
#[derive(Debug, Error)]
pub enum CertificateError {
    #[error("Certificate generation failed: {reason}")]
    GenerationFailed { reason: String },

    #[error("Certificate verification failed: {reason}")]
    VerificationFailed { reason: String },

    #[error("Bound certificate invalid: claimed {claimed:.6}, actual {actual:.6}")]
    InvalidBoundCertificate { claimed: f64, actual: f64 },

    #[error("Spectral certificate invalid: eigenvalue error {error:.6e}")]
    InvalidSpectralCertificate { error: f64 },

    #[error("Certificate expired: generated at {generated}, current {current}")]
    Expired { generated: String, current: String },

    #[error("Certificate signature mismatch")]
    SignatureMismatch,

    #[error("Missing certificate component: {component}")]
    MissingComponent { component: String },

    #[error("Certificate format version unsupported: {version}")]
    UnsupportedVersion { version: u32 },
}

/// I/O errors for file parsing and serialization.
#[derive(Debug, Error)]
pub enum IoError {
    #[error("File not found: {path}")]
    FileNotFound { path: String },

    #[error("Permission denied: {path}")]
    PermissionDenied { path: String },

    #[error("Parse error in {path} at line {line}: {reason}")]
    ParseError {
        path: String,
        line: usize,
        reason: String,
    },

    #[error("Invalid MPS format: {reason}")]
    InvalidMpsFormat { reason: String },

    #[error("Invalid LP format: {reason}")]
    InvalidLpFormat { reason: String },

    #[error("Invalid DEC format: {reason}")]
    InvalidDecFormat { reason: String },

    #[error("JSON serialization error: {reason}")]
    JsonError { reason: String },

    #[error("File too large: {path} is {size_mb:.1} MB, limit is {limit_mb:.1} MB")]
    FileTooLarge {
        path: String,
        size_mb: f64,
        limit_mb: f64,
    },

    #[error("Encoding error: {reason}")]
    EncodingError { reason: String },

    #[error("I/O error: {0}")]
    StdIo(#[from] std::io::Error),

    #[error("Serde JSON error: {0}")]
    SerdeJson(#[from] serde_json::Error),
}

/// Configuration-related errors.
#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("Missing required field: {field}")]
    MissingField { field: String },

    #[error("Invalid value for {field}: {value} (expected {expected})")]
    InvalidValue {
        field: String,
        value: String,
        expected: String,
    },

    #[error("Value out of range for {field}: {value} not in [{min}, {max}]")]
    OutOfRange {
        field: String,
        value: f64,
        min: f64,
        max: f64,
    },

    #[error("Conflicting configuration: {field_a} and {field_b}: {reason}")]
    Conflict {
        field_a: String,
        field_b: String,
        reason: String,
    },

    #[error("Unknown configuration key: {key}")]
    UnknownKey { key: String },

    #[error("Configuration file error: {reason}")]
    FileError { reason: String },

    #[error("Schema version mismatch: expected {expected}, got {actual}")]
    SchemaMismatch { expected: u32, actual: u32 },

    #[error("Environment variable {var} not set")]
    EnvVarMissing { var: String },

    #[error("Type conversion error for {field}: cannot convert '{value}' to {target_type}")]
    TypeConversion {
        field: String,
        value: String,
        target_type: String,
    },
}

/// Validation errors for data integrity checks.
#[derive(Debug, Error)]
pub enum ValidationError {
    #[error("Empty input: {context}")]
    EmptyInput { context: String },

    #[error("Invalid dimension: {name} = {value}, must be > 0")]
    InvalidDimension { name: String, value: usize },

    #[error("NaN detected in {context}")]
    NanDetected { context: String },

    #[error("Infinity detected in {context}")]
    InfinityDetected { context: String },

    #[error("Negative value where non-negative required: {name} = {value}")]
    NegativeValue { name: String, value: f64 },

    #[error("Invariant violated: {invariant}")]
    InvariantViolated { invariant: String },

    #[error("Constraint violated: {constraint}")]
    ConstraintViolated { constraint: String },

    #[error("Duplicate identifier: {id} in {context}")]
    DuplicateId { id: String, context: String },

    #[error("Referential integrity: {reference} not found in {context}")]
    ReferentialIntegrity { reference: String, context: String },

    #[error("Sum does not equal expected: got {actual:.6}, expected {expected:.6} in {context}")]
    SumMismatch {
        actual: f64,
        expected: f64,
        context: String,
    },

    #[error("Array length mismatch: {name_a}({len_a}) != {name_b}({len_b})")]
    LengthMismatch {
        name_a: String,
        len_a: usize,
        name_b: String,
        len_b: usize,
    },
}

/// Severity level for errors that may be warnings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ErrorSeverity {
    Warning,
    Error,
    Fatal,
}

impl fmt::Display for ErrorSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorSeverity::Warning => write!(f, "WARNING"),
            ErrorSeverity::Error => write!(f, "ERROR"),
            ErrorSeverity::Fatal => write!(f, "FATAL"),
        }
    }
}

/// An error with associated severity and optional context chain.
#[derive(Debug)]
pub struct AnnotatedError {
    pub severity: ErrorSeverity,
    pub error: SpectralError,
    pub context: Vec<String>,
}

impl AnnotatedError {
    pub fn new(severity: ErrorSeverity, error: SpectralError) -> Self {
        Self {
            severity,
            error,
            context: Vec::new(),
        }
    }

    pub fn with_context(mut self, ctx: impl Into<String>) -> Self {
        self.context.push(ctx.into());
        self
    }

    pub fn is_fatal(&self) -> bool {
        self.severity == ErrorSeverity::Fatal
    }

    pub fn context_chain(&self) -> String {
        if self.context.is_empty() {
            return self.error.to_string();
        }
        let chain: Vec<&str> = self.context.iter().map(|s| s.as_str()).collect();
        format!("{}: {}", chain.join(" -> "), self.error)
    }
}

impl fmt::Display for AnnotatedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.severity, self.context_chain())
    }
}

impl std::error::Error for AnnotatedError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

/// Collect multiple validation errors.
#[derive(Debug, Default)]
pub struct ErrorCollector {
    errors: Vec<AnnotatedError>,
}

impl ErrorCollector {
    pub fn new() -> Self {
        Self { errors: Vec::new() }
    }

    pub fn push(&mut self, severity: ErrorSeverity, error: SpectralError) {
        self.errors.push(AnnotatedError::new(severity, error));
    }

    pub fn push_with_context(
        &mut self,
        severity: ErrorSeverity,
        error: SpectralError,
        ctx: impl Into<String>,
    ) {
        self.errors
            .push(AnnotatedError::new(severity, error).with_context(ctx));
    }

    pub fn push_validation(&mut self, error: ValidationError) {
        self.errors.push(AnnotatedError::new(
            ErrorSeverity::Error,
            SpectralError::Validation(error),
        ));
    }

    pub fn has_errors(&self) -> bool {
        self.errors
            .iter()
            .any(|e| matches!(e.severity, ErrorSeverity::Error | ErrorSeverity::Fatal))
    }

    pub fn has_fatal(&self) -> bool {
        self.errors
            .iter()
            .any(|e| e.severity == ErrorSeverity::Fatal)
    }

    pub fn error_count(&self) -> usize {
        self.errors
            .iter()
            .filter(|e| matches!(e.severity, ErrorSeverity::Error | ErrorSeverity::Fatal))
            .count()
    }

    pub fn warning_count(&self) -> usize {
        self.errors
            .iter()
            .filter(|e| e.severity == ErrorSeverity::Warning)
            .count()
    }

    pub fn is_empty(&self) -> bool {
        self.errors.is_empty()
    }

    pub fn len(&self) -> usize {
        self.errors.len()
    }

    pub fn errors(&self) -> &[AnnotatedError] {
        &self.errors
    }

    pub fn into_result(self) -> Result<()> {
        if let Some(first_err) = self
            .errors
            .into_iter()
            .find(|e| matches!(e.severity, ErrorSeverity::Error | ErrorSeverity::Fatal))
        {
            Err(first_err.error)
        } else {
            Ok(())
        }
    }

    pub fn merge(&mut self, other: ErrorCollector) {
        self.errors.extend(other.errors);
    }

    pub fn summary(&self) -> String {
        format!(
            "{} error(s), {} warning(s)",
            self.error_count(),
            self.warning_count()
        )
    }
}

impl fmt::Display for ErrorCollector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, e) in self.errors.iter().enumerate() {
            if i > 0 {
                writeln!(f)?;
            }
            write!(f, "  {}", e)?;
        }
        Ok(())
    }
}

/// Extension trait for adding context to Results.
pub trait ResultExt<T> {
    fn with_context_str(self, ctx: &str) -> Result<T>;
    fn with_spectral_context(self, f: impl FnOnce() -> String) -> Result<T>;
}

impl<T, E: Into<SpectralError>> ResultExt<T> for std::result::Result<T, E> {
    fn with_context_str(self, ctx: &str) -> Result<T> {
        self.map_err(|e| {
            let base = e.into();
            SpectralError::Internal(format!("{}: {}", ctx, base))
        })
    }

    fn with_spectral_context(self, f: impl FnOnce() -> String) -> Result<T> {
        self.map_err(|e| {
            let base = e.into();
            SpectralError::Internal(format!("{}: {}", f(), base))
        })
    }
}

/// Convenience function to validate a condition.
pub fn ensure(condition: bool, error: impl FnOnce() -> SpectralError) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(error())
    }
}

/// Validate that a value is finite (not NaN or infinity).
pub fn ensure_finite(value: f64, name: &str) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else if value.is_nan() {
        Err(SpectralError::Validation(ValidationError::NanDetected {
            context: name.to_string(),
        }))
    } else {
        Err(SpectralError::Validation(
            ValidationError::InfinityDetected {
                context: name.to_string(),
            },
        ))
    }
}

/// Validate that a dimension is positive.
pub fn ensure_positive_dimension(value: usize, name: &str) -> Result<()> {
    if value > 0 {
        Ok(())
    } else {
        Err(SpectralError::Validation(
            ValidationError::InvalidDimension {
                name: name.to_string(),
                value,
            },
        ))
    }
}

/// Validate that two lengths match.
pub fn ensure_lengths_match(
    name_a: &str,
    len_a: usize,
    name_b: &str,
    len_b: usize,
) -> Result<()> {
    if len_a == len_b {
        Ok(())
    } else {
        Err(SpectralError::Validation(ValidationError::LengthMismatch {
            name_a: name_a.to_string(),
            len_a,
            name_b: name_b.to_string(),
            len_b,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spectral_error_display() {
        let err = SpectralError::Internal("test".to_string());
        assert!(err.to_string().contains("test"));
    }

    #[test]
    fn test_matrix_error_dimension_mismatch() {
        let err = MatrixError::DimensionMismatch {
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
    fn test_matrix_error_into_spectral() {
        let merr = MatrixError::NotSquare { rows: 3, cols: 4 };
        let serr: SpectralError = merr.into();
        assert!(matches!(serr, SpectralError::Matrix(_)));
    }

    #[test]
    fn test_optimization_error_display() {
        let err = OptimizationError::TimeLimitExceeded {
            elapsed: 100.5,
            limit: 60.0,
        };
        assert!(err.to_string().contains("100.50"));
    }

    #[test]
    fn test_error_collector_empty() {
        let ec = ErrorCollector::new();
        assert!(ec.is_empty());
        assert!(!ec.has_errors());
        assert_eq!(ec.len(), 0);
    }

    #[test]
    fn test_error_collector_push() {
        let mut ec = ErrorCollector::new();
        ec.push(
            ErrorSeverity::Warning,
            SpectralError::Internal("warn".to_string()),
        );
        assert_eq!(ec.len(), 1);
        assert_eq!(ec.warning_count(), 1);
        assert_eq!(ec.error_count(), 0);
        assert!(!ec.has_errors());
    }

    #[test]
    fn test_error_collector_has_errors() {
        let mut ec = ErrorCollector::new();
        ec.push(
            ErrorSeverity::Error,
            SpectralError::Internal("err".to_string()),
        );
        assert!(ec.has_errors());
        assert_eq!(ec.error_count(), 1);
    }

    #[test]
    fn test_error_collector_into_result_ok() {
        let mut ec = ErrorCollector::new();
        ec.push(
            ErrorSeverity::Warning,
            SpectralError::Internal("warn".to_string()),
        );
        assert!(ec.into_result().is_ok());
    }

    #[test]
    fn test_error_collector_into_result_err() {
        let mut ec = ErrorCollector::new();
        ec.push(
            ErrorSeverity::Error,
            SpectralError::Internal("fail".to_string()),
        );
        assert!(ec.into_result().is_err());
    }

    #[test]
    fn test_annotated_error_context_chain() {
        let ae = AnnotatedError::new(
            ErrorSeverity::Error,
            SpectralError::Internal("base".to_string()),
        )
        .with_context("outer")
        .with_context("inner");
        let chain = ae.context_chain();
        assert!(chain.contains("outer"));
        assert!(chain.contains("inner"));
        assert!(chain.contains("base"));
    }

    #[test]
    fn test_ensure_finite_ok() {
        assert!(ensure_finite(1.0, "x").is_ok());
    }

    #[test]
    fn test_ensure_finite_nan() {
        assert!(ensure_finite(f64::NAN, "x").is_err());
    }

    #[test]
    fn test_ensure_finite_inf() {
        assert!(ensure_finite(f64::INFINITY, "x").is_err());
    }

    #[test]
    fn test_ensure_positive_dimension() {
        assert!(ensure_positive_dimension(5, "n").is_ok());
        assert!(ensure_positive_dimension(0, "n").is_err());
    }

    #[test]
    fn test_ensure_lengths_match() {
        assert!(ensure_lengths_match("a", 3, "b", 3).is_ok());
        assert!(ensure_lengths_match("a", 3, "b", 4).is_err());
    }

    #[test]
    fn test_error_collector_merge() {
        let mut ec1 = ErrorCollector::new();
        ec1.push(
            ErrorSeverity::Warning,
            SpectralError::Internal("w".to_string()),
        );
        let mut ec2 = ErrorCollector::new();
        ec2.push(
            ErrorSeverity::Error,
            SpectralError::Internal("e".to_string()),
        );
        ec1.merge(ec2);
        assert_eq!(ec1.len(), 2);
    }

    #[test]
    fn test_error_severity_display() {
        assert_eq!(ErrorSeverity::Warning.to_string(), "WARNING");
        assert_eq!(ErrorSeverity::Error.to_string(), "ERROR");
        assert_eq!(ErrorSeverity::Fatal.to_string(), "FATAL");
    }

    #[test]
    fn test_io_error_from_std() {
        let std_err = std::io::Error::new(std::io::ErrorKind::NotFound, "gone");
        let io_err: IoError = std_err.into();
        assert!(matches!(io_err, IoError::StdIo(_)));
    }

    #[test]
    fn test_ensure_fn() {
        assert!(ensure(true, || SpectralError::Internal("x".into())).is_ok());
        assert!(ensure(false, || SpectralError::Internal("x".into())).is_err());
    }

    #[test]
    fn test_collector_summary() {
        let mut ec = ErrorCollector::new();
        ec.push(ErrorSeverity::Warning, SpectralError::Internal("w".into()));
        ec.push(ErrorSeverity::Error, SpectralError::Internal("e".into()));
        let s = ec.summary();
        assert!(s.contains("1 error(s)"));
        assert!(s.contains("1 warning(s)"));
    }
}
