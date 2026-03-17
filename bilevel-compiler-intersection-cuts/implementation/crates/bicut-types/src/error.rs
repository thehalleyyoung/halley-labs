//! Error types for the BiCut bilevel optimization compiler.
//!
//! Provides a comprehensive error hierarchy using `thiserror` for all
//! error conditions across the workspace: parsing, solving, validation,
//! certificates, and numerical issues.

use std::fmt;
use thiserror::Error;

// ── Top-level error ────────────────────────────────────────────────

/// The master error type for the BiCut compiler workspace.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum BicutError {
    #[error("parse error: {0}")]
    Parse(#[from] ParseError),

    #[error("solver error: {0}")]
    Solver(#[from] SolverError),

    #[error("infeasible: {0}")]
    Infeasible(#[from] InfeasibleError),

    #[error("numerical error: {0}")]
    Numerical(#[from] NumericalError),

    #[error("certificate error: {0}")]
    Certificate(#[from] CertificateError),

    #[error("validation error: {0}")]
    Validation(#[from] ValidationError),

    #[error("internal error: {message}")]
    Internal { message: String },

    #[error("not implemented: {feature}")]
    NotImplemented { feature: String },

    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("index out of bounds: index {index}, length {length}")]
    IndexOutOfBounds { index: usize, length: usize },
}

// ── Parse errors ───────────────────────────────────────────────────

/// Errors arising when parsing problem files (MPS, LP, custom formats).
#[derive(Error, Debug, Clone, PartialEq)]
pub enum ParseError {
    #[error("unexpected token '{token}' at line {line}, column {column}")]
    UnexpectedToken {
        token: String,
        line: usize,
        column: usize,
    },

    #[error("missing required section '{section}'")]
    MissingSection { section: String },

    #[error("duplicate definition of '{name}'")]
    DuplicateDefinition { name: String },

    #[error("invalid number format: '{value}'")]
    InvalidNumber { value: String },

    #[error("unknown variable reference: '{name}'")]
    UnknownVariable { name: String },

    #[error("unknown constraint reference: '{name}'")]
    UnknownConstraint { name: String },

    #[error("malformed input at line {line}: {detail}")]
    MalformedInput { line: usize, detail: String },

    #[error("unsupported format: '{format}'")]
    UnsupportedFormat { format: String },

    #[error("empty input")]
    EmptyInput,
}

// ── Solver errors ──────────────────────────────────────────────────

/// Errors from the underlying LP/MIP solvers.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum SolverError {
    #[error("solver returned status: {status}")]
    BadStatus { status: String },

    #[error("iteration limit exceeded ({limit} iterations)")]
    IterationLimit { limit: u64 },

    #[error("time limit exceeded ({seconds:.1}s)")]
    TimeLimit { seconds: f64 },

    #[error("unbounded problem detected")]
    Unbounded,

    #[error("no feasible solution found")]
    NoFeasibleSolution,

    #[error("solver not available: '{solver}'")]
    SolverNotAvailable { solver: String },

    #[error("licensing error for solver '{solver}': {detail}")]
    LicenseError { solver: String, detail: String },

    #[error("callback error: {message}")]
    CallbackError { message: String },

    #[error("parameter error: invalid value '{value}' for parameter '{param}'")]
    ParameterError { param: String, value: String },
}

// ── Infeasibility errors ───────────────────────────────────────────

/// Details about infeasibility detection.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum InfeasibleError {
    #[error("primal infeasible: {detail}")]
    PrimalInfeasible { detail: String },

    #[error("dual infeasible (primal unbounded): {detail}")]
    DualInfeasible { detail: String },

    #[error("bilevel infeasible: no solution satisfies both levels")]
    BilevelInfeasible,

    #[error("infeasible subsystem identified: constraints {constraints:?}")]
    IrreducibleInfeasibleSubsystem { constraints: Vec<String> },

    #[error("conflicting bounds on variable '{variable}': lb={lb}, ub={ub}")]
    ConflictingBounds { variable: String, lb: f64, ub: f64 },
}

// ── Numerical errors ───────────────────────────────────────────────

/// Numerical stability and precision issues.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum NumericalError {
    #[error("near-zero pivot ({value:.2e}) below tolerance ({tolerance:.2e})")]
    NearZeroPivot { value: f64, tolerance: f64 },

    #[error("condition number too large: {condition_number:.2e}")]
    IllConditioned { condition_number: f64 },

    #[error("overflow detected in {operation}")]
    Overflow { operation: String },

    #[error("NaN encountered in {context}")]
    NaN { context: String },

    #[error("big-M value {big_m:.2e} exceeds safe threshold {threshold:.2e}")]
    BigMTooLarge { big_m: f64, threshold: f64 },

    #[error("tolerance violation: residual {residual:.2e} exceeds tolerance {tolerance:.2e}")]
    ToleranceViolation { residual: f64, tolerance: f64 },

    #[error("matrix is singular or near-singular")]
    SingularMatrix,
}

// ── Certificate errors ─────────────────────────────────────────────

/// Errors related to correctness certificates.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum CertificateError {
    #[error("verification failed: {reason}")]
    VerificationFailed { reason: String },

    #[error("hash mismatch: expected {expected}, got {got}")]
    HashMismatch { expected: String, got: String },

    #[error("missing certificate entry: '{entry}'")]
    MissingEntry { entry: String },

    #[error("stale certificate: problem has been modified since certification")]
    StaleCertificate,

    #[error("unsupported reformulation type: '{reformulation}'")]
    UnsupportedReformulation { reformulation: String },

    #[error("constraint qualification '{cq}' not satisfied")]
    CqNotSatisfied { cq: String },

    #[error("incomplete certificate: missing {missing_steps:?}")]
    Incomplete { missing_steps: Vec<String> },
}

// ── Validation errors ──────────────────────────────────────────────

/// Errors from structural/semantic validation of problems.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum ValidationError {
    #[error("dimension mismatch in {context}: expected {expected}, got {got}")]
    DimensionMismatch {
        context: String,
        expected: usize,
        got: usize,
    },

    #[error("empty {entity}: at least one is required")]
    Empty { entity: String },

    #[error("variable '{name}' has invalid bounds: lb={lb}, ub={ub}")]
    InvalidBounds { name: String, lb: f64, ub: f64 },

    #[error("duplicate name '{name}' in {context}")]
    DuplicateName { name: String, context: String },

    #[error("dangling reference to '{name}' in {context}")]
    DanglingReference { name: String, context: String },

    #[error("constraint '{name}' references variable not in scope")]
    ScopeViolation { name: String },

    #[error("objective function is missing for {level}")]
    MissingObjective { level: String },

    #[error("problem is structurally invalid: {detail}")]
    StructuralError { detail: String },

    #[error("configuration error: {detail}")]
    ConfigError { detail: String },
}

// ── Result alias ───────────────────────────────────────────────────

/// Convenience result type for BiCut operations.
pub type BicutResult<T> = Result<T, BicutError>;

// ── Display helpers ────────────────────────────────────────────────

/// A structured error report that can carry multiple diagnostics.
#[derive(Debug, Clone, PartialEq)]
pub struct ErrorReport {
    pub errors: Vec<BicutError>,
}

impl ErrorReport {
    pub fn new() -> Self {
        Self { errors: Vec::new() }
    }

    pub fn push(&mut self, error: BicutError) {
        self.errors.push(error);
    }

    pub fn is_empty(&self) -> bool {
        self.errors.is_empty()
    }

    pub fn len(&self) -> usize {
        self.errors.len()
    }

    pub fn into_result<T>(self, value: T) -> BicutResult<T> {
        if self.errors.is_empty() {
            Ok(value)
        } else {
            Err(self.errors.into_iter().next().unwrap())
        }
    }

    pub fn merge(&mut self, other: ErrorReport) {
        self.errors.extend(other.errors);
    }
}

impl Default for ErrorReport {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for ErrorReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.errors.is_empty() {
            write!(f, "no errors")
        } else {
            writeln!(f, "{} error(s):", self.errors.len())?;
            for (i, e) in self.errors.iter().enumerate() {
                writeln!(f, "  [{}] {}", i + 1, e)?;
            }
            Ok(())
        }
    }
}

// ── Severity for diagnostics ───────────────────────────────────────

/// Diagnostic severity levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Severity {
    Info,
    Warning,
    Error,
    Fatal,
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Severity::Info => write!(f, "info"),
            Severity::Warning => write!(f, "warning"),
            Severity::Error => write!(f, "error"),
            Severity::Fatal => write!(f, "fatal"),
        }
    }
}

/// A single diagnostic message with location and severity.
#[derive(Debug, Clone, PartialEq)]
pub struct Diagnostic {
    pub severity: Severity,
    pub message: String,
    pub context: Option<String>,
}

impl Diagnostic {
    pub fn info(message: impl Into<String>) -> Self {
        Self {
            severity: Severity::Info,
            message: message.into(),
            context: None,
        }
    }

    pub fn warning(message: impl Into<String>) -> Self {
        Self {
            severity: Severity::Warning,
            message: message.into(),
            context: None,
        }
    }

    pub fn error(message: impl Into<String>) -> Self {
        Self {
            severity: Severity::Error,
            message: message.into(),
            context: None,
        }
    }

    pub fn with_context(mut self, ctx: impl Into<String>) -> Self {
        self.context = Some(ctx.into());
        self
    }
}

impl fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.severity, self.message)?;
        if let Some(ref ctx) = self.context {
            write!(f, " (in {})", ctx)?;
        }
        Ok(())
    }
}

// ── Helper constructors ────────────────────────────────────────────

impl BicutError {
    pub fn internal(msg: impl Into<String>) -> Self {
        BicutError::Internal {
            message: msg.into(),
        }
    }

    pub fn not_implemented(feature: impl Into<String>) -> Self {
        BicutError::NotImplemented {
            feature: feature.into(),
        }
    }

    pub fn dim_mismatch(expected: usize, got: usize) -> Self {
        BicutError::DimensionMismatch { expected, got }
    }

    pub fn index_oob(index: usize, length: usize) -> Self {
        BicutError::IndexOutOfBounds { index, length }
    }

    /// Whether this error indicates a fundamentally unrecoverable situation.
    pub fn is_fatal(&self) -> bool {
        matches!(
            self,
            BicutError::Internal { .. } | BicutError::NotImplemented { .. }
        )
    }

    /// Whether this error might be resolved by relaxing tolerances.
    pub fn is_numerical(&self) -> bool {
        matches!(self, BicutError::Numerical(_))
    }
}

impl ParseError {
    pub fn unexpected(token: impl Into<String>, line: usize, column: usize) -> Self {
        ParseError::UnexpectedToken {
            token: token.into(),
            line,
            column,
        }
    }

    pub fn missing_section(section: impl Into<String>) -> Self {
        ParseError::MissingSection {
            section: section.into(),
        }
    }
}

impl SolverError {
    pub fn bad_status(status: impl Into<String>) -> Self {
        SolverError::BadStatus {
            status: status.into(),
        }
    }
}

impl ValidationError {
    pub fn dim_mismatch(context: impl Into<String>, expected: usize, got: usize) -> Self {
        ValidationError::DimensionMismatch {
            context: context.into(),
            expected,
            got,
        }
    }

    pub fn empty(entity: impl Into<String>) -> Self {
        ValidationError::Empty {
            entity: entity.into(),
        }
    }

    pub fn structural(detail: impl Into<String>) -> Self {
        ValidationError::StructuralError {
            detail: detail.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bicut_error_display() {
        let err = BicutError::internal("something broke");
        assert_eq!(err.to_string(), "internal error: something broke");
    }

    #[test]
    fn test_parse_error_conversion() {
        let pe = ParseError::unexpected("FOO", 10, 5);
        let be: BicutError = pe.into();
        assert!(be.to_string().contains("FOO"));
        assert!(be.to_string().contains("line 10"));
    }

    #[test]
    fn test_solver_error_variants() {
        let e = SolverError::Unbounded;
        assert_eq!(e.to_string(), "unbounded problem detected");

        let e = SolverError::TimeLimit { seconds: 3600.0 };
        assert!(e.to_string().contains("3600.0"));
    }

    #[test]
    fn test_infeasible_error_display() {
        let e = InfeasibleError::ConflictingBounds {
            variable: "x1".into(),
            lb: 5.0,
            ub: 3.0,
        };
        assert!(e.to_string().contains("x1"));
    }

    #[test]
    fn test_numerical_error_display() {
        let e = NumericalError::BigMTooLarge {
            big_m: 1e12,
            threshold: 1e8,
        };
        assert!(e.to_string().contains("big-M"));
    }

    #[test]
    fn test_certificate_error_display() {
        let e = CertificateError::CqNotSatisfied { cq: "LICQ".into() };
        assert!(e.to_string().contains("LICQ"));
    }

    #[test]
    fn test_validation_error_display() {
        let e = ValidationError::dim_mismatch("matrix A", 10, 8);
        assert!(e.to_string().contains("matrix A"));
    }

    #[test]
    fn test_error_report() {
        let mut report = ErrorReport::new();
        assert!(report.is_empty());

        report.push(BicutError::internal("err1"));
        report.push(BicutError::internal("err2"));
        assert_eq!(report.len(), 2);
        assert!(!report.is_empty());

        let display = report.to_string();
        assert!(display.contains("2 error(s)"));
    }

    #[test]
    fn test_error_report_into_result() {
        let empty = ErrorReport::new();
        assert!(empty.into_result(42).is_ok());

        let mut bad = ErrorReport::new();
        bad.push(BicutError::internal("oops"));
        assert!(bad.into_result(42).is_err());
    }

    #[test]
    fn test_diagnostic() {
        let d = Diagnostic::warning("possible issue").with_context("constraint c1");
        assert_eq!(d.severity, Severity::Warning);
        assert!(d.to_string().contains("possible issue"));
        assert!(d.to_string().contains("constraint c1"));
    }

    #[test]
    fn test_is_fatal_and_numerical() {
        assert!(BicutError::internal("x").is_fatal());
        assert!(BicutError::not_implemented("y").is_fatal());
        assert!(!BicutError::dim_mismatch(1, 2).is_fatal());

        let ne = NumericalError::SingularMatrix;
        let be: BicutError = ne.into();
        assert!(be.is_numerical());
    }
}
