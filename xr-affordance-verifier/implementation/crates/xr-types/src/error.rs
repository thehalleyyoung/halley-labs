//! Error types for the XR Affordance Verifier.

use thiserror::Error;

/// Main error type for the verification system.
#[derive(Error, Debug)]
pub enum VerifierError {
    #[error("Scene parsing error: {0}")]
    SceneParsing(String),

    #[error("Invalid scene structure: {0}")]
    InvalidScene(String),

    #[error("Kinematic model error: {0}")]
    KinematicModel(String),

    #[error("Joint limit violation: joint {joint_index}, angle {angle} outside [{min}, {max}]")]
    JointLimitViolation {
        joint_index: usize,
        angle: f64,
        min: f64,
        max: f64,
    },

    #[error("Forward kinematics computation failed: {0}")]
    ForwardKinematics(String),

    #[error("Inverse kinematics failed to converge after {iterations} iterations")]
    InverseKinematicsConvergence { iterations: usize },

    #[error("Device configuration error: {0}")]
    DeviceConfig(String),

    #[error("Tracking volume exceeded for device {device_name}")]
    TrackingVolumeExceeded { device_name: String },

    #[error("SMT encoding error: {0}")]
    SmtEncoding(String),

    #[error("SMT solver error: {0}")]
    SmtSolver(String),

    #[error("SMT solver timeout after {seconds}s")]
    SmtTimeout { seconds: f64 },

    #[error("Linearization error exceeds bound: {actual} > {bound}")]
    LinearizationError { actual: f64, bound: f64 },

    #[error("Interval arithmetic overflow in {operation}")]
    IntervalOverflow { operation: String },

    #[error("Affine arithmetic error: {0}")]
    AffineArithmetic(String),

    #[error("Certificate generation failed: {0}")]
    CertificateGeneration(String),

    #[error("Insufficient samples: need {needed}, have {have}")]
    InsufficientSamples { needed: usize, have: usize },

    #[error("Coverage target not met: {achieved:.4} < {target:.4}")]
    CoverageNotMet { achieved: f64, target: f64 },

    #[error("Accessibility violation: element {element_name} unreachable for body params {body_params:?}")]
    AccessibilityViolation {
        element_name: String,
        body_params: Vec<f64>,
    },

    #[error("Configuration error: {0}")]
    Configuration(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("Numeric error: {0}")]
    Numeric(String),

    #[error("Topology error: {0}")]
    Topology(String),

    #[error("DSL parse error at line {line}, column {column}: {message}")]
    DslParse {
        line: usize,
        column: usize,
        message: String,
    },

    #[error("DSL type error: {0}")]
    DslType(String),

    #[error("Decomposition error: {0}")]
    Decomposition(String),

    #[error("Zone abstraction error: {0}")]
    ZoneAbstraction(String),

    #[error("Internal error: {0}")]
    Internal(String),
}

/// Convenience result type for the verifier.
pub type VerifierResult<T> = Result<T, VerifierError>;

/// Severity level for verification diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize)]
pub enum Severity {
    /// Informational message.
    Info,
    /// Warning that may indicate an issue.
    Warning,
    /// Error that must be addressed.
    Error,
    /// Critical error that blocks certification.
    Critical,
}

impl std::fmt::Display for Severity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Severity::Info => write!(f, "INFO"),
            Severity::Warning => write!(f, "WARN"),
            Severity::Error => write!(f, "ERROR"),
            Severity::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// A diagnostic message from the verification pipeline.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Diagnostic {
    /// Severity of the diagnostic.
    pub severity: Severity,
    /// Short code identifying the diagnostic type.
    pub code: String,
    /// Human-readable message.
    pub message: String,
    /// Optional element ID this diagnostic pertains to.
    pub element_id: Option<uuid::Uuid>,
    /// Optional source location or context.
    pub context: Option<String>,
    /// Suggested fix, if any.
    pub suggestion: Option<String>,
}

impl Diagnostic {
    /// Create a new info diagnostic.
    pub fn info(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            severity: Severity::Info,
            code: code.into(),
            message: message.into(),
            element_id: None,
            context: None,
            suggestion: None,
        }
    }

    /// Create a new warning diagnostic.
    pub fn warning(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            severity: Severity::Warning,
            code: code.into(),
            message: message.into(),
            element_id: None,
            context: None,
            suggestion: None,
        }
    }

    /// Create a new error diagnostic.
    pub fn error(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            severity: Severity::Error,
            code: code.into(),
            message: message.into(),
            element_id: None,
            context: None,
            suggestion: None,
        }
    }

    /// Create a new critical diagnostic.
    pub fn critical(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            severity: Severity::Critical,
            code: code.into(),
            message: message.into(),
            element_id: None,
            context: None,
            suggestion: None,
        }
    }

    /// Set the element ID.
    pub fn with_element(mut self, id: uuid::Uuid) -> Self {
        self.element_id = Some(id);
        self
    }

    /// Set the context.
    pub fn with_context(mut self, ctx: impl Into<String>) -> Self {
        self.context = Some(ctx.into());
        self
    }

    /// Set the suggestion.
    pub fn with_suggestion(mut self, suggestion: impl Into<String>) -> Self {
        self.suggestion = Some(suggestion.into());
        self
    }
}

impl std::fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {}: {}", self.severity, self.code, self.message)?;
        if let Some(ref ctx) = self.context {
            write!(f, " ({})", ctx)?;
        }
        if let Some(ref suggestion) = self.suggestion {
            write!(f, " -> {}", suggestion)?;
        }
        Ok(())
    }
}

/// Collection of diagnostics accumulated during verification.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct DiagnosticCollection {
    diagnostics: Vec<Diagnostic>,
}

impl DiagnosticCollection {
    /// Create a new empty collection.
    pub fn new() -> Self {
        Self {
            diagnostics: Vec::new(),
        }
    }

    /// Add a diagnostic.
    pub fn push(&mut self, diag: Diagnostic) {
        self.diagnostics.push(diag);
    }

    /// Get all diagnostics.
    pub fn all(&self) -> &[Diagnostic] {
        &self.diagnostics
    }

    /// Get diagnostics filtered by severity.
    pub fn by_severity(&self, severity: Severity) -> Vec<&Diagnostic> {
        self.diagnostics
            .iter()
            .filter(|d| d.severity == severity)
            .collect()
    }

    /// Check if there are any errors or critical diagnostics.
    pub fn has_errors(&self) -> bool {
        self.diagnostics
            .iter()
            .any(|d| d.severity >= Severity::Error)
    }

    /// Count diagnostics of a given severity.
    pub fn count(&self, severity: Severity) -> usize {
        self.diagnostics
            .iter()
            .filter(|d| d.severity == severity)
            .count()
    }

    /// Total number of diagnostics.
    pub fn len(&self) -> usize {
        self.diagnostics.len()
    }

    /// Check if the collection is empty.
    pub fn is_empty(&self) -> bool {
        self.diagnostics.is_empty()
    }

    /// Merge another collection into this one.
    pub fn merge(&mut self, other: DiagnosticCollection) {
        self.diagnostics.extend(other.diagnostics);
    }

    /// Get the highest severity in the collection.
    pub fn max_severity(&self) -> Option<Severity> {
        self.diagnostics.iter().map(|d| d.severity).max()
    }

    /// Filter diagnostics for a specific element.
    pub fn for_element(&self, id: uuid::Uuid) -> Vec<&Diagnostic> {
        self.diagnostics
            .iter()
            .filter(|d| d.element_id == Some(id))
            .collect()
    }

    /// Remove all diagnostics below a given severity.
    pub fn filter_min_severity(&mut self, min: Severity) {
        self.diagnostics.retain(|d| d.severity >= min);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diagnostic_creation() {
        let d = Diagnostic::info("XR001", "Test message")
            .with_context("test context")
            .with_suggestion("fix it");
        assert_eq!(d.severity, Severity::Info);
        assert_eq!(d.code, "XR001");
        assert_eq!(d.message, "Test message");
        assert_eq!(d.context.as_deref(), Some("test context"));
        assert_eq!(d.suggestion.as_deref(), Some("fix it"));
    }

    #[test]
    fn test_diagnostic_collection() {
        let mut coll = DiagnosticCollection::new();
        coll.push(Diagnostic::info("XR001", "info"));
        coll.push(Diagnostic::warning("XR002", "warning"));
        coll.push(Diagnostic::error("XR003", "error"));

        assert_eq!(coll.len(), 3);
        assert!(coll.has_errors());
        assert_eq!(coll.count(Severity::Info), 1);
        assert_eq!(coll.count(Severity::Warning), 1);
        assert_eq!(coll.count(Severity::Error), 1);
        assert_eq!(coll.max_severity(), Some(Severity::Error));
    }

    #[test]
    fn test_severity_ordering() {
        assert!(Severity::Info < Severity::Warning);
        assert!(Severity::Warning < Severity::Error);
        assert!(Severity::Error < Severity::Critical);
    }

    #[test]
    fn test_filter_min_severity() {
        let mut coll = DiagnosticCollection::new();
        coll.push(Diagnostic::info("XR001", "info"));
        coll.push(Diagnostic::warning("XR002", "warning"));
        coll.push(Diagnostic::error("XR003", "error"));
        coll.filter_min_severity(Severity::Warning);
        assert_eq!(coll.len(), 2);
    }
}
