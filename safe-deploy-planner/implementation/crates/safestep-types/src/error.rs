// Error types for the SafeStep deployment planner.

use std::fmt;

use serde::{Deserialize, Serialize};

/// Severity levels for diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ErrorSeverity {
    Info,
    Warning,
    Error,
    Fatal,
}

impl fmt::Display for ErrorSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Info => write!(f, "info"),
            Self::Warning => write!(f, "warning"),
            Self::Error => write!(f, "error"),
            Self::Fatal => write!(f, "fatal"),
        }
    }
}

/// Source location for diagnostics.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SourceLocation {
    pub file: String,
    pub line: Option<u32>,
    pub column: Option<u32>,
    pub context: Option<String>,
}

impl SourceLocation {
    pub fn new(file: impl Into<String>) -> Self {
        Self {
            file: file.into(),
            line: None,
            column: None,
            context: None,
        }
    }

    pub fn with_line(mut self, line: u32) -> Self {
        self.line = Some(line);
        self
    }

    pub fn with_column(mut self, column: u32) -> Self {
        self.column = Some(column);
        self
    }

    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.context = Some(context.into());
        self
    }
}

impl fmt::Display for SourceLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.file)?;
        if let Some(line) = self.line {
            write!(f, ":{}", line)?;
            if let Some(col) = self.column {
                write!(f, ":{}", col)?;
            }
        }
        Ok(())
    }
}

/// A structured diagnostic message with source location, severity, and suggestions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticMessage {
    pub severity: ErrorSeverity,
    pub code: Option<String>,
    pub message: String,
    pub location: Option<SourceLocation>,
    pub suggestions: Vec<String>,
    pub notes: Vec<String>,
    pub related: Vec<DiagnosticMessage>,
}

impl DiagnosticMessage {
    pub fn new(severity: ErrorSeverity, message: impl Into<String>) -> Self {
        Self {
            severity,
            code: None,
            message: message.into(),
            location: None,
            suggestions: Vec::new(),
            notes: Vec::new(),
            related: Vec::new(),
        }
    }

    pub fn info(message: impl Into<String>) -> Self {
        Self::new(ErrorSeverity::Info, message)
    }

    pub fn warning(message: impl Into<String>) -> Self {
        Self::new(ErrorSeverity::Warning, message)
    }

    pub fn error(message: impl Into<String>) -> Self {
        Self::new(ErrorSeverity::Error, message)
    }

    pub fn fatal(message: impl Into<String>) -> Self {
        Self::new(ErrorSeverity::Fatal, message)
    }

    pub fn with_code(mut self, code: impl Into<String>) -> Self {
        self.code = Some(code.into());
        self
    }

    pub fn with_location(mut self, location: SourceLocation) -> Self {
        self.location = Some(location);
        self
    }

    pub fn with_suggestion(mut self, suggestion: impl Into<String>) -> Self {
        self.suggestions.push(suggestion.into());
        self
    }

    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }

    pub fn with_related(mut self, related: DiagnosticMessage) -> Self {
        self.related.push(related);
        self
    }

    pub fn is_fatal(&self) -> bool {
        self.severity == ErrorSeverity::Fatal
    }

    pub fn is_error_or_above(&self) -> bool {
        self.severity >= ErrorSeverity::Error
    }
}

impl fmt::Display for DiagnosticMessage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.severity)?;
        if let Some(code) = &self.code {
            write!(f, "[{}]", code)?;
        }
        write!(f, ": {}", self.message)?;
        if let Some(loc) = &self.location {
            write!(f, " at {}", loc)?;
        }
        for note in &self.notes {
            write!(f, "\n  note: {}", note)?;
        }
        for suggestion in &self.suggestions {
            write!(f, "\n  suggestion: {}", suggestion)?;
        }
        for related in &self.related {
            write!(f, "\n  related: {}", related)?;
        }
        Ok(())
    }
}

/// Collection of diagnostics from a single operation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ErrorReport {
    pub diagnostics: Vec<DiagnosticMessage>,
    pub context: Option<String>,
}

impl ErrorReport {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.context = Some(context.into());
        self
    }

    pub fn push(&mut self, diagnostic: DiagnosticMessage) {
        self.diagnostics.push(diagnostic);
    }

    pub fn extend(&mut self, other: ErrorReport) {
        self.diagnostics.extend(other.diagnostics);
    }

    pub fn is_empty(&self) -> bool {
        self.diagnostics.is_empty()
    }

    pub fn len(&self) -> usize {
        self.diagnostics.len()
    }

    pub fn has_errors(&self) -> bool {
        self.diagnostics.iter().any(|d| d.is_error_or_above())
    }

    pub fn has_fatal(&self) -> bool {
        self.diagnostics.iter().any(|d| d.is_fatal())
    }

    pub fn errors(&self) -> impl Iterator<Item = &DiagnosticMessage> {
        self.diagnostics.iter().filter(|d| d.is_error_or_above())
    }

    pub fn warnings(&self) -> impl Iterator<Item = &DiagnosticMessage> {
        self.diagnostics
            .iter()
            .filter(|d| d.severity == ErrorSeverity::Warning)
    }

    pub fn infos(&self) -> impl Iterator<Item = &DiagnosticMessage> {
        self.diagnostics
            .iter()
            .filter(|d| d.severity == ErrorSeverity::Info)
    }

    pub fn by_severity(&self, severity: ErrorSeverity) -> Vec<&DiagnosticMessage> {
        self.diagnostics
            .iter()
            .filter(|d| d.severity == severity)
            .collect()
    }

    pub fn error_count(&self) -> usize {
        self.diagnostics
            .iter()
            .filter(|d| d.is_error_or_above())
            .count()
    }

    pub fn warning_count(&self) -> usize {
        self.diagnostics
            .iter()
            .filter(|d| d.severity == ErrorSeverity::Warning)
            .count()
    }

    /// Summary string: "N errors, M warnings"
    pub fn summary(&self) -> String {
        let errors = self.error_count();
        let warnings = self.warning_count();
        let fatals = self.diagnostics.iter().filter(|d| d.is_fatal()).count();
        if fatals > 0 {
            format!(
                "{} fatal, {} errors, {} warnings",
                fatals, errors, warnings
            )
        } else {
            format!("{} errors, {} warnings", errors, warnings)
        }
    }

    /// Convert to a Result - Ok if no errors, Err if any errors.
    pub fn into_result(self) -> std::result::Result<(), SafeStepError> {
        if self.has_errors() {
            Err(SafeStepError::Diagnostic(self))
        } else {
            Ok(())
        }
    }
}

impl fmt::Display for ErrorReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(ctx) = &self.context {
            writeln!(f, "Context: {}", ctx)?;
        }
        for diag in &self.diagnostics {
            writeln!(f, "{}", diag)?;
        }
        write!(f, "Summary: {}", self.summary())
    }
}

/// Error context for chain-of-errors support.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorContext {
    pub operation: String,
    pub details: Option<String>,
    pub location: Option<SourceLocation>,
}

impl ErrorContext {
    pub fn new(operation: impl Into<String>) -> Self {
        Self {
            operation: operation.into(),
            details: None,
            location: None,
        }
    }

    pub fn with_details(mut self, details: impl Into<String>) -> Self {
        self.details = Some(details.into());
        self
    }

    pub fn with_location(mut self, location: SourceLocation) -> Self {
        self.location = Some(location);
        self
    }
}

impl fmt::Display for ErrorContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.operation)?;
        if let Some(details) = &self.details {
            write!(f, ": {}", details)?;
        }
        if let Some(loc) = &self.location {
            write!(f, " at {}", loc)?;
        }
        Ok(())
    }
}

/// The primary error type for SafeStep.
#[derive(Debug, thiserror::Error)]
pub enum SafeStepError {
    #[error("Version parse error: {message}")]
    VersionParse {
        message: String,
        input: String,
        context: Option<ErrorContext>,
    },

    #[error("Constraint violation: {message}")]
    ConstraintViolation {
        message: String,
        constraint_id: Option<String>,
        context: Option<ErrorContext>,
    },

    #[error("Solver timeout after {elapsed_ms}ms (limit: {timeout_ms}ms)")]
    SolverTimeout {
        elapsed_ms: u64,
        timeout_ms: u64,
        partial_result: Option<String>,
        context: Option<ErrorContext>,
    },

    #[error("Infeasible plan: {reason}")]
    InfeasiblePlan {
        reason: String,
        blocking_constraints: Vec<String>,
        context: Option<ErrorContext>,
    },

    #[error("Encoding error: {message}")]
    EncodingError {
        message: String,
        encoding_type: String,
        context: Option<ErrorContext>,
    },

    #[error("Schema error: {message}")]
    SchemaError {
        message: String,
        service: Option<String>,
        context: Option<ErrorContext>,
    },

    #[error("Kubernetes error: {message}")]
    K8sError {
        message: String,
        resource: Option<String>,
        namespace: Option<String>,
        context: Option<ErrorContext>,
    },

    #[error("Configuration error: {message}")]
    ConfigError {
        message: String,
        field: Option<String>,
        context: Option<ErrorContext>,
    },

    #[error("IO error: {source}")]
    IoError {
        #[from]
        source: std::io::Error,
    },

    #[error("JSON error: {source}")]
    JsonError {
        #[from]
        source: serde_json::Error,
    },

    #[error("Internal error: {message}")]
    InternalError {
        message: String,
        context: Option<ErrorContext>,
    },

    #[error("Invalid identifier: {message}")]
    InvalidIdentifier {
        message: String,
        input: String,
        context: Option<ErrorContext>,
    },

    #[error("Resource limit exceeded: {message}")]
    ResourceLimitExceeded {
        message: String,
        limit: String,
        actual: String,
        context: Option<ErrorContext>,
    },

    #[error("Graph error: {message}")]
    GraphError {
        message: String,
        context: Option<ErrorContext>,
    },

    #[error("Plan validation error: {message}")]
    PlanValidation {
        message: String,
        step_index: Option<usize>,
        context: Option<ErrorContext>,
    },

    #[error("Diagnostic report: {0}")]
    Diagnostic(ErrorReport),
}

impl SafeStepError {
    pub fn version_parse(message: impl Into<String>, input: impl Into<String>) -> Self {
        Self::VersionParse {
            message: message.into(),
            input: input.into(),
            context: None,
        }
    }

    pub fn constraint_violation(message: impl Into<String>) -> Self {
        Self::ConstraintViolation {
            message: message.into(),
            constraint_id: None,
            context: None,
        }
    }

    pub fn solver_timeout(elapsed_ms: u64, timeout_ms: u64) -> Self {
        Self::SolverTimeout {
            elapsed_ms,
            timeout_ms,
            partial_result: None,
            context: None,
        }
    }

    pub fn infeasible(reason: impl Into<String>) -> Self {
        Self::InfeasiblePlan {
            reason: reason.into(),
            blocking_constraints: Vec::new(),
            context: None,
        }
    }

    pub fn encoding(message: impl Into<String>, encoding_type: impl Into<String>) -> Self {
        Self::EncodingError {
            message: message.into(),
            encoding_type: encoding_type.into(),
            context: None,
        }
    }

    pub fn schema(message: impl Into<String>) -> Self {
        Self::SchemaError {
            message: message.into(),
            service: None,
            context: None,
        }
    }

    pub fn k8s(message: impl Into<String>) -> Self {
        Self::K8sError {
            message: message.into(),
            resource: None,
            namespace: None,
            context: None,
        }
    }

    pub fn config(message: impl Into<String>) -> Self {
        Self::ConfigError {
            message: message.into(),
            field: None,
            context: None,
        }
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self::InternalError {
            message: message.into(),
            context: None,
        }
    }

    pub fn invalid_id(message: impl Into<String>, input: impl Into<String>) -> Self {
        Self::InvalidIdentifier {
            message: message.into(),
            input: input.into(),
            context: None,
        }
    }

    pub fn graph(message: impl Into<String>) -> Self {
        Self::GraphError {
            message: message.into(),
            context: None,
        }
    }

    pub fn plan_validation(message: impl Into<String>) -> Self {
        Self::PlanValidation {
            message: message.into(),
            step_index: None,
            context: None,
        }
    }

    /// Add context to this error.
    pub fn with_context(mut self, ctx: ErrorContext) -> Self {
        match &mut self {
            Self::VersionParse { context, .. }
            | Self::ConstraintViolation { context, .. }
            | Self::SolverTimeout { context, .. }
            | Self::InfeasiblePlan { context, .. }
            | Self::EncodingError { context, .. }
            | Self::SchemaError { context, .. }
            | Self::K8sError { context, .. }
            | Self::ConfigError { context, .. }
            | Self::InternalError { context, .. }
            | Self::InvalidIdentifier { context, .. }
            | Self::ResourceLimitExceeded { context, .. }
            | Self::GraphError { context, .. }
            | Self::PlanValidation { context, .. } => {
                *context = Some(ctx);
            }
            Self::IoError { .. } | Self::JsonError { .. } | Self::Diagnostic(_) => {}
        }
        self
    }

    /// Return the severity of this error.
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            Self::InternalError { .. } => ErrorSeverity::Fatal,
            Self::SolverTimeout { .. } => ErrorSeverity::Error,
            Self::IoError { .. } | Self::JsonError { .. } => ErrorSeverity::Error,
            Self::VersionParse { .. }
            | Self::ConstraintViolation { .. }
            | Self::InfeasiblePlan { .. }
            | Self::EncodingError { .. }
            | Self::SchemaError { .. }
            | Self::K8sError { .. }
            | Self::ConfigError { .. }
            | Self::InvalidIdentifier { .. }
            | Self::ResourceLimitExceeded { .. }
            | Self::GraphError { .. }
            | Self::PlanValidation { .. } => ErrorSeverity::Error,
            Self::Diagnostic(report) => {
                if report.has_fatal() {
                    ErrorSeverity::Fatal
                } else if report.has_errors() {
                    ErrorSeverity::Error
                } else {
                    ErrorSeverity::Warning
                }
            }
        }
    }

    /// Convert to a DiagnosticMessage.
    pub fn to_diagnostic(&self) -> DiagnosticMessage {
        let mut diag = DiagnosticMessage::new(self.severity(), self.to_string());
        match self {
            Self::VersionParse { input, .. } => {
                diag = diag.with_note(format!("Input was: {:?}", input));
            }
            Self::InfeasiblePlan {
                blocking_constraints,
                ..
            } => {
                for c in blocking_constraints {
                    diag = diag.with_note(format!("Blocking: {}", c));
                }
            }
            Self::SolverTimeout { partial_result, .. } => {
                if let Some(pr) = partial_result {
                    diag = diag.with_note(format!("Partial result: {}", pr));
                }
            }
            _ => {}
        }
        diag
    }
}

/// Convenience result type alias.
pub type Result<T> = std::result::Result<T, SafeStepError>;

/// Extension trait for Result to add error context.
pub trait ResultExt<T> {
    fn with_context(self, ctx: ErrorContext) -> Result<T>;
    fn with_operation(self, operation: impl Into<String>) -> Result<T>;
}

impl<T> ResultExt<T> for Result<T> {
    fn with_context(self, ctx: ErrorContext) -> Result<T> {
        self.map_err(|e| e.with_context(ctx))
    }

    fn with_operation(self, operation: impl Into<String>) -> Result<T> {
        self.map_err(|e| e.with_context(ErrorContext::new(operation)))
    }
}

/// Convert from a string parse error.
impl From<std::num::ParseIntError> for SafeStepError {
    fn from(e: std::num::ParseIntError) -> Self {
        Self::VersionParse {
            message: e.to_string(),
            input: String::new(),
            context: None,
        }
    }
}

impl From<std::num::ParseFloatError> for SafeStepError {
    fn from(e: std::num::ParseFloatError) -> Self {
        Self::ConfigError {
            message: format!("Failed to parse float: {}", e),
            field: None,
            context: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_severity_ordering() {
        assert!(ErrorSeverity::Info < ErrorSeverity::Warning);
        assert!(ErrorSeverity::Warning < ErrorSeverity::Error);
        assert!(ErrorSeverity::Error < ErrorSeverity::Fatal);
    }

    #[test]
    fn test_severity_display() {
        assert_eq!(ErrorSeverity::Info.to_string(), "info");
        assert_eq!(ErrorSeverity::Fatal.to_string(), "fatal");
    }

    #[test]
    fn test_source_location() {
        let loc = SourceLocation::new("test.rs")
            .with_line(42)
            .with_column(10);
        assert_eq!(loc.to_string(), "test.rs:42:10");
    }

    #[test]
    fn test_source_location_file_only() {
        let loc = SourceLocation::new("test.rs");
        assert_eq!(loc.to_string(), "test.rs");
    }

    #[test]
    fn test_diagnostic_message() {
        let diag = DiagnosticMessage::error("Something went wrong")
            .with_code("E001")
            .with_suggestion("Try restarting")
            .with_note("This is a known issue");
        assert!(diag.is_error_or_above());
        assert!(!diag.is_fatal());
        let s = diag.to_string();
        assert!(s.contains("E001"));
        assert!(s.contains("Something went wrong"));
        assert!(s.contains("Try restarting"));
    }

    #[test]
    fn test_diagnostic_fatal() {
        let diag = DiagnosticMessage::fatal("Critical failure");
        assert!(diag.is_fatal());
        assert!(diag.is_error_or_above());
    }

    #[test]
    fn test_error_report() {
        let mut report = ErrorReport::new().with_context("test operation");
        assert!(report.is_empty());
        assert!(!report.has_errors());

        report.push(DiagnosticMessage::warning("minor issue"));
        assert!(!report.has_errors());
        assert_eq!(report.warning_count(), 1);

        report.push(DiagnosticMessage::error("big problem"));
        assert!(report.has_errors());
        assert_eq!(report.error_count(), 1);
        assert_eq!(report.len(), 2);

        let summary = report.summary();
        assert!(summary.contains("1 errors"));
        assert!(summary.contains("1 warnings"));
    }

    #[test]
    fn test_error_report_into_result_ok() {
        let report = ErrorReport::new();
        assert!(report.into_result().is_ok());
    }

    #[test]
    fn test_error_report_into_result_err() {
        let mut report = ErrorReport::new();
        report.push(DiagnosticMessage::error("bad"));
        assert!(report.into_result().is_err());
    }

    #[test]
    fn test_safestep_error_constructors() {
        let e = SafeStepError::version_parse("bad version", "1.x.y");
        assert!(matches!(e, SafeStepError::VersionParse { .. }));
        assert_eq!(e.severity(), ErrorSeverity::Error);

        let e = SafeStepError::solver_timeout(5000, 3000);
        assert!(matches!(e, SafeStepError::SolverTimeout { .. }));

        let e = SafeStepError::internal("oops");
        assert_eq!(e.severity(), ErrorSeverity::Fatal);
    }

    #[test]
    fn test_error_with_context() {
        let e = SafeStepError::config("missing field")
            .with_context(ErrorContext::new("loading config").with_details("from file test.yaml"));
        let s = e.to_string();
        assert!(s.contains("missing field"));
    }

    #[test]
    fn test_error_to_diagnostic() {
        let e = SafeStepError::infeasible("no valid path");
        let diag = e.to_diagnostic();
        assert_eq!(diag.severity, ErrorSeverity::Error);
        assert!(diag.message.contains("no valid path"));
    }

    #[test]
    fn test_result_ext() {
        let r: Result<()> = Err(SafeStepError::config("bad"));
        let r2 = r.with_operation("loading config");
        assert!(r2.is_err());
    }

    #[test]
    fn test_error_report_by_severity() {
        let mut report = ErrorReport::new();
        report.push(DiagnosticMessage::info("fyi"));
        report.push(DiagnosticMessage::warning("careful"));
        report.push(DiagnosticMessage::error("broken"));
        report.push(DiagnosticMessage::fatal("dead"));

        assert_eq!(report.by_severity(ErrorSeverity::Info).len(), 1);
        assert_eq!(report.by_severity(ErrorSeverity::Warning).len(), 1);
        assert!(report.has_fatal());
    }

    #[test]
    fn test_error_report_extend() {
        let mut r1 = ErrorReport::new();
        r1.push(DiagnosticMessage::error("e1"));
        let mut r2 = ErrorReport::new();
        r2.push(DiagnosticMessage::warning("w1"));
        r1.extend(r2);
        assert_eq!(r1.len(), 2);
    }

    #[test]
    fn test_from_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let e: SafeStepError = io_err.into();
        assert!(matches!(e, SafeStepError::IoError { .. }));
    }

    #[test]
    fn test_error_context_display() {
        let ctx = ErrorContext::new("parsing")
            .with_details("version string")
            .with_location(SourceLocation::new("input.yaml").with_line(5));
        let s = ctx.to_string();
        assert!(s.contains("parsing"));
        assert!(s.contains("version string"));
        assert!(s.contains("input.yaml:5"));
    }

    #[test]
    fn test_diagnostic_related() {
        let related = DiagnosticMessage::info("related info");
        let diag = DiagnosticMessage::error("main error").with_related(related);
        assert_eq!(diag.related.len(), 1);
        let s = diag.to_string();
        assert!(s.contains("related"));
    }
}
