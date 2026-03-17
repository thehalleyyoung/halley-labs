//! Error types for the conservation analysis pipeline.

use thiserror::Error;
use std::fmt;

/// Primary result type for conservation analysis operations.
pub type Result<T> = std::result::Result<T, ConservationError>;

/// Comprehensive error type covering all failure modes in conservation analysis.
#[derive(Error, Debug, Clone)]
pub enum ConservationError {
    #[error("Phase space error: {0}")]
    PhaseSpace(#[from] PhaseSpaceError),

    #[error("Symmetry analysis error: {0}")]
    Symmetry(#[from] SymmetryError),

    #[error("Symbolic computation error: {0}")]
    Symbolic(#[from] SymbolicError),

    #[error("Numerical error: {0}")]
    Numerical(#[from] NumericalError),

    #[error("IR construction error: {0}")]
    IrConstruction(#[from] IrError),

    #[error("BCH expansion error: {0}")]
    BchExpansion(#[from] BchError),

    #[error("Certificate generation error: {0}")]
    Certificate(#[from] CertificateError),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("IO error: {0}")]
    Io(String),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Internal error: {0}")]
    Internal(String),
}

/// Errors related to phase space operations.
#[derive(Error, Debug, Clone)]
pub enum PhaseSpaceError {
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Invalid coordinate index {index} for dimension {dimension}")]
    InvalidCoordinate { index: usize, dimension: usize },

    #[error("Incompatible phase space structures: {0}")]
    IncompatibleStructures(String),

    #[error("Symplectic form is degenerate at point {0}")]
    DegenerateSymplecticForm(String),

    #[error("Metric tensor is not positive definite: {0}")]
    NonPositiveMetric(String),

    #[error("Phase space declaration missing for variable: {0}")]
    MissingDeclaration(String),

    #[error("Coordinate transformation failed: {0}")]
    TransformationFailed(String),
}

/// Errors related to symmetry analysis.
#[derive(Error, Debug, Clone)]
pub enum SymmetryError {
    #[error("No symmetries found for the given system")]
    NoSymmetries,

    #[error("Lie algebra computation failed: {0}")]
    LieAlgebraFailed(String),

    #[error("Determining equations are inconsistent: {0}")]
    InconsistentDetermining(String),

    #[error("Symmetry generator is not well-defined: {0}")]
    IllDefinedGenerator(String),

    #[error("Prolongation computation failed at order {order}: {reason}")]
    ProlongationFailed { order: usize, reason: String },

    #[error("Symmetry group structure is not closed: {0}")]
    NotClosed(String),

    #[error("Ansatz exhausted without finding generators")]
    AnsatzExhausted,
}

/// Errors in symbolic computation.
#[derive(Error, Debug, Clone)]
pub enum SymbolicError {
    #[error("Division by zero in symbolic expression")]
    DivisionByZero,

    #[error("Polynomial degree exceeds maximum: {degree} > {max}")]
    DegreeTooHigh { degree: usize, max: usize },

    #[error("Variable not found: {0}")]
    VariableNotFound(String),

    #[error("Expression simplification failed: {0}")]
    SimplificationFailed(String),

    #[error("Differentiation failed for expression: {0}")]
    DifferentiationFailed(String),

    #[error("Symbolic integration not supported: {0}")]
    IntegrationUnsupported(String),

    #[error("Expression too complex: {0}")]
    TooComplex(String),
}

/// Numerical computation errors.
#[derive(Error, Debug, Clone)]
pub enum NumericalError {
    #[error("Convergence failed after {iterations} iterations (residual: {residual})")]
    ConvergenceFailed { iterations: usize, residual: f64 },

    #[error("Numerical overflow in computation: {0}")]
    Overflow(String),

    #[error("Numerical underflow in computation: {0}")]
    Underflow(String),

    #[error("Matrix is singular: condition number = {condition}")]
    SingularMatrix { condition: f64 },

    #[error("Eigenvalue computation failed: {0}")]
    EigenvalueFailed(String),

    #[error("Tolerance not met: required {required}, achieved {achieved}")]
    ToleranceNotMet { required: f64, achieved: f64 },

    #[error("NaN encountered in computation at step {0}")]
    NanEncountered(usize),

    #[error("Drift exceeds threshold: {drift} > {threshold}")]
    ExcessiveDrift { drift: f64, threshold: f64 },
}

/// IR construction errors.
#[derive(Error, Debug, Clone)]
pub enum IrError {
    #[error("Unliftable construct: {0}")]
    Unliftable(String),

    #[error("Unsupported operation: {0}")]
    UnsupportedOp(String),

    #[error("Type inference failed for node {node_id}: {reason}")]
    TypeInference { node_id: String, reason: String },

    #[error("Control flow too complex: nesting depth {depth} > max {max}")]
    ComplexControlFlow { depth: usize, max: usize },

    #[error("Missing annotation: {0}")]
    MissingAnnotation(String),

    #[error("Pattern recognition failed: {0}")]
    PatternFailed(String),

    #[error("Array access not analyzable: {0}")]
    ArrayAccess(String),
}

/// BCH expansion errors.
#[derive(Error, Debug, Clone)]
pub enum BchError {
    #[error("BCH order {order} exceeds maximum {max}")]
    OrderTooHigh { order: usize, max: usize },

    #[error("BCH expansion did not converge at order {0}")]
    NonConvergent(usize),

    #[error("Operator composition failed: {0}")]
    CompositionFailed(String),

    #[error("Tag algebra overflow: {0}")]
    TagOverflow(String),

    #[error("Commutator computation failed: {0}")]
    CommutatorFailed(String),
}

/// Certificate generation errors.
#[derive(Error, Debug, Clone)]
pub enum CertificateError {
    #[error("Obstruction analysis failed: {0}")]
    ObstructionFailed(String),

    #[error("Groebner basis computation timed out")]
    GroebnerTimeout,

    #[error("Certificate validation failed: {0}")]
    ValidationFailed(String),

    #[error("Insufficient evidence for certificate: {0}")]
    InsufficientEvidence(String),
}

impl ConservationError {
    pub fn config(msg: impl Into<String>) -> Self {
        ConservationError::Config(msg.into())
    }

    pub fn io(msg: impl Into<String>) -> Self {
        ConservationError::Io(msg.into())
    }

    pub fn internal(msg: impl Into<String>) -> Self {
        ConservationError::Internal(msg.into())
    }

    pub fn is_recoverable(&self) -> bool {
        match self {
            ConservationError::Numerical(NumericalError::ConvergenceFailed { .. }) => true,
            ConservationError::Numerical(NumericalError::ToleranceNotMet { .. }) => true,
            ConservationError::Symmetry(SymmetryError::AnsatzExhausted) => true,
            ConservationError::BchExpansion(BchError::OrderTooHigh { .. }) => true,
            _ => false,
        }
    }

    pub fn error_code(&self) -> &'static str {
        match self {
            ConservationError::PhaseSpace(_) => "E001",
            ConservationError::Symmetry(_) => "E002",
            ConservationError::Symbolic(_) => "E003",
            ConservationError::Numerical(_) => "E004",
            ConservationError::IrConstruction(_) => "E005",
            ConservationError::BchExpansion(_) => "E006",
            ConservationError::Certificate(_) => "E007",
            ConservationError::Config(_) => "E008",
            ConservationError::Io(_) => "E009",
            ConservationError::Serialization(_) => "E010",
            ConservationError::Internal(_) => "E999",
        }
    }
}

/// A warning that does not prevent analysis from continuing.
#[derive(Debug, Clone)]
pub struct AnalysisWarning {
    pub code: String,
    pub message: String,
    pub location: Option<String>,
    pub suggestion: Option<String>,
}

impl fmt::Display for AnalysisWarning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.code, self.message)?;
        if let Some(ref loc) = self.location {
            write!(f, " at {}", loc)?;
        }
        if let Some(ref sug) = self.suggestion {
            write!(f, " (suggestion: {})", sug)?;
        }
        Ok(())
    }
}

impl AnalysisWarning {
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
            location: None,
            suggestion: None,
        }
    }

    pub fn with_location(mut self, loc: impl Into<String>) -> Self {
        self.location = Some(loc.into());
        self
    }

    pub fn with_suggestion(mut self, sug: impl Into<String>) -> Self {
        self.suggestion = Some(sug.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_codes() {
        let e = ConservationError::Config("test".to_string());
        assert_eq!(e.error_code(), "E008");
    }

    #[test]
    fn test_recoverable() {
        let e = ConservationError::Numerical(NumericalError::ConvergenceFailed {
            iterations: 100,
            residual: 1e-3,
        });
        assert!(e.is_recoverable());

        let e2 = ConservationError::Internal("fatal".to_string());
        assert!(!e2.is_recoverable());
    }

    #[test]
    fn test_warning_display() {
        let w = AnalysisWarning::new("W001", "potential drift")
            .with_location("main.py:42")
            .with_suggestion("use symplectic integrator");
        let s = format!("{}", w);
        assert!(s.contains("W001"));
        assert!(s.contains("potential drift"));
        assert!(s.contains("main.py:42"));
    }

    #[test]
    fn test_error_display() {
        let e = ConservationError::PhaseSpace(PhaseSpaceError::DimensionMismatch {
            expected: 6,
            actual: 4,
        });
        let msg = format!("{}", e);
        assert!(msg.contains("Dimension mismatch"));
        assert!(msg.contains("6"));
        assert!(msg.contains("4"));
    }
}
