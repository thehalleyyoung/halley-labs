//! Error types for the MutSpec system.
//!
//! All fallible operations in MutSpec return [`MutSpecError`].  Each variant
//! carries structured context so that errors can be rendered with source
//! locations, operator names, and other diagnostics.

use std::fmt;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ---------------------------------------------------------------------------
// SourceLocation / SpanInfo
// ---------------------------------------------------------------------------

/// A point in a source file.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SourceLocation {
    pub file: PathBuf,
    pub line: usize,
    pub column: usize,
}

impl SourceLocation {
    pub fn new(file: impl Into<PathBuf>, line: usize, column: usize) -> Self {
        Self {
            file: file.into(),
            line,
            column,
        }
    }

    pub fn unknown() -> Self {
        Self {
            file: PathBuf::from("<unknown>"),
            line: 0,
            column: 0,
        }
    }

    pub fn is_unknown(&self) -> bool {
        self.line == 0 && self.column == 0
    }
}

impl fmt::Display for SourceLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}:{}", self.file.display(), self.line, self.column)
    }
}

/// A span between two points in a source file.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SpanInfo {
    pub start: SourceLocation,
    pub end: SourceLocation,
}

impl SpanInfo {
    pub fn new(start: SourceLocation, end: SourceLocation) -> Self {
        Self { start, end }
    }

    pub fn point(loc: SourceLocation) -> Self {
        Self {
            start: loc.clone(),
            end: loc,
        }
    }

    pub fn unknown() -> Self {
        Self {
            start: SourceLocation::unknown(),
            end: SourceLocation::unknown(),
        }
    }

    pub fn is_unknown(&self) -> bool {
        self.start.is_unknown() && self.end.is_unknown()
    }

    /// Returns true if this span contains the given location.
    pub fn contains_location(&self, loc: &SourceLocation) -> bool {
        if self.start.file != loc.file {
            return false;
        }
        let after_start = loc.line > self.start.line
            || (loc.line == self.start.line && loc.column >= self.start.column);
        let before_end = loc.line < self.end.line
            || (loc.line == self.end.line && loc.column <= self.end.column);
        after_start && before_end
    }

    /// Merge two spans into one that covers both.
    pub fn merge(&self, other: &SpanInfo) -> SpanInfo {
        let start =
            if (self.start.line, self.start.column) <= (other.start.line, other.start.column) {
                self.start.clone()
            } else {
                other.start.clone()
            };
        let end = if (self.end.line, self.end.column) >= (other.end.line, other.end.column) {
            self.end.clone()
        } else {
            other.end.clone()
        };
        SpanInfo { start, end }
    }

    /// Number of lines this span covers.
    pub fn line_count(&self) -> usize {
        if self.end.line >= self.start.line {
            self.end.line - self.start.line + 1
        } else {
            1
        }
    }
}

impl fmt::Display for SpanInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.start == self.end {
            write!(f, "{}", self.start)
        } else {
            write!(f, "{} - {}", self.start, self.end)
        }
    }
}

// ---------------------------------------------------------------------------
// ErrorContext
// ---------------------------------------------------------------------------

/// Additional context attached to an error.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErrorContext {
    pub location: Option<SpanInfo>,
    pub function_name: Option<String>,
    pub phase: Option<String>,
    pub notes: Vec<String>,
}

impl ErrorContext {
    pub fn new() -> Self {
        Self {
            location: None,
            function_name: None,
            phase: None,
            notes: Vec::new(),
        }
    }

    pub fn with_span(mut self, span: SpanInfo) -> Self {
        self.location = Some(span);
        self
    }

    pub fn with_function(mut self, name: impl Into<String>) -> Self {
        self.function_name = Some(name.into());
        self
    }

    pub fn with_phase(mut self, phase: impl Into<String>) -> Self {
        self.phase = Some(phase.into());
        self
    }

    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }

    pub fn has_location(&self) -> bool {
        self.location.is_some()
    }
}

impl Default for ErrorContext {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for ErrorContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(ref loc) = self.location {
            write!(f, " at {loc}")?;
        }
        if let Some(ref func) = self.function_name {
            write!(f, " in function `{func}`")?;
        }
        if let Some(ref phase) = self.phase {
            write!(f, " during {phase}")?;
        }
        for note in &self.notes {
            write!(f, "\n  note: {note}")?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// MutSpecError
// ---------------------------------------------------------------------------

/// Top-level error type for the MutSpec system.
#[derive(Debug, Error)]
pub enum MutSpecError {
    #[error("parse error: {message}{context}")]
    Parse {
        message: String,
        context: ErrorContext,
    },

    #[error("type-check error: {message}{context}")]
    TypeCheck {
        message: String,
        expected: Option<String>,
        found: Option<String>,
        context: ErrorContext,
    },

    #[error("mutation error: {message}{context}")]
    Mutation {
        message: String,
        operator: Option<String>,
        context: ErrorContext,
    },

    #[error("analysis error: {message}{context}")]
    Analysis {
        message: String,
        context: ErrorContext,
    },

    #[error("SMT solver error: {message}{context}")]
    SmtSolver {
        message: String,
        solver_output: Option<String>,
        context: ErrorContext,
    },

    #[error("synthesis error: {message}{context}")]
    Synthesis {
        message: String,
        tier: Option<String>,
        context: ErrorContext,
    },

    #[error("coverage error: {message}{context}")]
    Coverage {
        message: String,
        context: ErrorContext,
    },

    #[error("I/O error: {message}")]
    Io {
        message: String,
        path: Option<PathBuf>,
        #[source]
        source: Option<std::io::Error>,
    },

    #[error("configuration error: {message}")]
    Config {
        message: String,
        key: Option<String>,
    },

    #[error("internal error: {message}{context}")]
    Internal {
        message: String,
        context: ErrorContext,
    },

    #[error("timeout after {duration_secs:.1}s: {message}")]
    Timeout {
        message: String,
        duration_secs: f64,
        context: ErrorContext,
    },

    #[error("unsupported feature: {feature}{context}")]
    UnsupportedFeature {
        feature: String,
        context: ErrorContext,
    },
}

impl MutSpecError {
    // -- Convenient constructors -------------------------------------------

    pub fn parse(msg: impl Into<String>) -> Self {
        MutSpecError::Parse {
            message: msg.into(),
            context: ErrorContext::new(),
        }
    }

    pub fn parse_at(msg: impl Into<String>, span: SpanInfo) -> Self {
        MutSpecError::Parse {
            message: msg.into(),
            context: ErrorContext::new().with_span(span),
        }
    }

    pub fn type_check(msg: impl Into<String>) -> Self {
        MutSpecError::TypeCheck {
            message: msg.into(),
            expected: None,
            found: None,
            context: ErrorContext::new(),
        }
    }

    pub fn type_mismatch(
        msg: impl Into<String>,
        expected: impl Into<String>,
        found: impl Into<String>,
    ) -> Self {
        MutSpecError::TypeCheck {
            message: msg.into(),
            expected: Some(expected.into()),
            found: Some(found.into()),
            context: ErrorContext::new(),
        }
    }

    pub fn mutation(msg: impl Into<String>) -> Self {
        MutSpecError::Mutation {
            message: msg.into(),
            operator: None,
            context: ErrorContext::new(),
        }
    }

    pub fn mutation_op(msg: impl Into<String>, op: impl Into<String>) -> Self {
        MutSpecError::Mutation {
            message: msg.into(),
            operator: Some(op.into()),
            context: ErrorContext::new(),
        }
    }

    pub fn analysis(msg: impl Into<String>) -> Self {
        MutSpecError::Analysis {
            message: msg.into(),
            context: ErrorContext::new(),
        }
    }

    pub fn smt(msg: impl Into<String>) -> Self {
        MutSpecError::SmtSolver {
            message: msg.into(),
            solver_output: None,
            context: ErrorContext::new(),
        }
    }

    pub fn smt_with_output(msg: impl Into<String>, output: impl Into<String>) -> Self {
        MutSpecError::SmtSolver {
            message: msg.into(),
            solver_output: Some(output.into()),
            context: ErrorContext::new(),
        }
    }

    pub fn synthesis(msg: impl Into<String>) -> Self {
        MutSpecError::Synthesis {
            message: msg.into(),
            tier: None,
            context: ErrorContext::new(),
        }
    }

    pub fn synthesis_tier(msg: impl Into<String>, tier: impl Into<String>) -> Self {
        MutSpecError::Synthesis {
            message: msg.into(),
            tier: Some(tier.into()),
            context: ErrorContext::new(),
        }
    }

    pub fn coverage(msg: impl Into<String>) -> Self {
        MutSpecError::Coverage {
            message: msg.into(),
            context: ErrorContext::new(),
        }
    }

    pub fn io(msg: impl Into<String>) -> Self {
        MutSpecError::Io {
            message: msg.into(),
            path: None,
            source: None,
        }
    }

    pub fn io_path(msg: impl Into<String>, path: impl Into<PathBuf>) -> Self {
        MutSpecError::Io {
            message: msg.into(),
            path: Some(path.into()),
            source: None,
        }
    }

    pub fn config(msg: impl Into<String>) -> Self {
        MutSpecError::Config {
            message: msg.into(),
            key: None,
        }
    }

    pub fn config_key(msg: impl Into<String>, key: impl Into<String>) -> Self {
        MutSpecError::Config {
            message: msg.into(),
            key: Some(key.into()),
        }
    }

    pub fn internal(msg: impl Into<String>) -> Self {
        MutSpecError::Internal {
            message: msg.into(),
            context: ErrorContext::new(),
        }
    }

    pub fn timeout(msg: impl Into<String>, duration_secs: f64) -> Self {
        MutSpecError::Timeout {
            message: msg.into(),
            duration_secs,
            context: ErrorContext::new(),
        }
    }

    pub fn unsupported(feature: impl Into<String>) -> Self {
        MutSpecError::UnsupportedFeature {
            feature: feature.into(),
            context: ErrorContext::new(),
        }
    }

    // -- Context accessors -------------------------------------------------

    /// Get the error context, if any.
    pub fn context(&self) -> Option<&ErrorContext> {
        match self {
            MutSpecError::Parse { context, .. }
            | MutSpecError::TypeCheck { context, .. }
            | MutSpecError::Mutation { context, .. }
            | MutSpecError::Analysis { context, .. }
            | MutSpecError::SmtSolver { context, .. }
            | MutSpecError::Synthesis { context, .. }
            | MutSpecError::Coverage { context, .. }
            | MutSpecError::Internal { context, .. }
            | MutSpecError::Timeout { context, .. }
            | MutSpecError::UnsupportedFeature { context, .. } => Some(context),
            MutSpecError::Io { .. } | MutSpecError::Config { .. } => None,
        }
    }

    /// Returns a short category string for this error.
    pub fn category(&self) -> &'static str {
        match self {
            MutSpecError::Parse { .. } => "parse",
            MutSpecError::TypeCheck { .. } => "type-check",
            MutSpecError::Mutation { .. } => "mutation",
            MutSpecError::Analysis { .. } => "analysis",
            MutSpecError::SmtSolver { .. } => "smt-solver",
            MutSpecError::Synthesis { .. } => "synthesis",
            MutSpecError::Coverage { .. } => "coverage",
            MutSpecError::Io { .. } => "io",
            MutSpecError::Config { .. } => "config",
            MutSpecError::Internal { .. } => "internal",
            MutSpecError::Timeout { .. } => "timeout",
            MutSpecError::UnsupportedFeature { .. } => "unsupported",
        }
    }

    /// Returns true if this is a transient error that might succeed on retry.
    pub fn is_transient(&self) -> bool {
        matches!(self, MutSpecError::Timeout { .. } | MutSpecError::Io { .. })
    }

    /// Add a note to this error (if it has a context).
    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        match &mut self {
            MutSpecError::Parse { context, .. }
            | MutSpecError::TypeCheck { context, .. }
            | MutSpecError::Mutation { context, .. }
            | MutSpecError::Analysis { context, .. }
            | MutSpecError::SmtSolver { context, .. }
            | MutSpecError::Synthesis { context, .. }
            | MutSpecError::Coverage { context, .. }
            | MutSpecError::Internal { context, .. }
            | MutSpecError::Timeout { context, .. }
            | MutSpecError::UnsupportedFeature { context, .. } => {
                context.notes.push(note.into());
            }
            _ => {}
        }
        self
    }

    /// Add a function name to this error (if it has a context).
    pub fn in_function(mut self, name: impl Into<String>) -> Self {
        match &mut self {
            MutSpecError::Parse { context, .. }
            | MutSpecError::TypeCheck { context, .. }
            | MutSpecError::Mutation { context, .. }
            | MutSpecError::Analysis { context, .. }
            | MutSpecError::SmtSolver { context, .. }
            | MutSpecError::Synthesis { context, .. }
            | MutSpecError::Coverage { context, .. }
            | MutSpecError::Internal { context, .. }
            | MutSpecError::Timeout { context, .. }
            | MutSpecError::UnsupportedFeature { context, .. } => {
                context.function_name = Some(name.into());
            }
            _ => {}
        }
        self
    }
}

// -- From impls for common error types -------------------------------------

impl From<std::io::Error> for MutSpecError {
    fn from(err: std::io::Error) -> Self {
        MutSpecError::Io {
            message: err.to_string(),
            path: None,
            source: Some(err),
        }
    }
}

impl From<serde_json::Error> for MutSpecError {
    fn from(err: serde_json::Error) -> Self {
        MutSpecError::Config {
            message: format!("JSON error: {err}"),
            key: None,
        }
    }
}

impl From<toml::de::Error> for MutSpecError {
    fn from(err: toml::de::Error) -> Self {
        MutSpecError::Config {
            message: format!("TOML parse error: {err}"),
            key: None,
        }
    }
}

impl From<fmt::Error> for MutSpecError {
    fn from(err: fmt::Error) -> Self {
        MutSpecError::Internal {
            message: format!("formatting error: {err}"),
            context: ErrorContext::new(),
        }
    }
}

/// Crate-level result alias.
pub type Result<T> = std::result::Result<T, MutSpecError>;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_source_location_display() {
        let loc = SourceLocation::new("test.ms", 10, 5);
        assert_eq!(loc.to_string(), "test.ms:10:5");
    }

    #[test]
    fn test_source_location_unknown() {
        let loc = SourceLocation::unknown();
        assert!(loc.is_unknown());
    }

    #[test]
    fn test_span_point() {
        let loc = SourceLocation::new("a.ms", 1, 1);
        let span = SpanInfo::point(loc.clone());
        assert_eq!(span.start, span.end);
        assert_eq!(span.line_count(), 1);
    }

    #[test]
    fn test_span_contains() {
        let span = SpanInfo::new(
            SourceLocation::new("a.ms", 5, 1),
            SourceLocation::new("a.ms", 10, 20),
        );
        assert!(span.contains_location(&SourceLocation::new("a.ms", 7, 5)));
        assert!(!span.contains_location(&SourceLocation::new("a.ms", 3, 1)));
        assert!(!span.contains_location(&SourceLocation::new("b.ms", 7, 5)));
    }

    #[test]
    fn test_span_merge() {
        let a = SpanInfo::new(
            SourceLocation::new("a.ms", 5, 1),
            SourceLocation::new("a.ms", 8, 10),
        );
        let b = SpanInfo::new(
            SourceLocation::new("a.ms", 3, 5),
            SourceLocation::new("a.ms", 12, 1),
        );
        let merged = a.merge(&b);
        assert_eq!(merged.start.line, 3);
        assert_eq!(merged.end.line, 12);
    }

    #[test]
    fn test_span_line_count() {
        let span = SpanInfo::new(
            SourceLocation::new("a.ms", 5, 1),
            SourceLocation::new("a.ms", 10, 1),
        );
        assert_eq!(span.line_count(), 6);
    }

    #[test]
    fn test_span_display_point() {
        let loc = SourceLocation::new("a.ms", 1, 1);
        let span = SpanInfo::point(loc);
        let s = span.to_string();
        assert!(s.contains("a.ms:1:1"));
        assert!(!s.contains(" - "));
    }

    #[test]
    fn test_span_display_range() {
        let span = SpanInfo::new(
            SourceLocation::new("a.ms", 1, 1),
            SourceLocation::new("a.ms", 5, 10),
        );
        let s = span.to_string();
        assert!(s.contains(" - "));
    }

    #[test]
    fn test_error_context_display() {
        let ctx = ErrorContext::new()
            .with_function("foo")
            .with_phase("type checking")
            .with_note("did you mean `int`?");
        let s = ctx.to_string();
        assert!(s.contains("foo"));
        assert!(s.contains("type checking"));
        assert!(s.contains("did you mean"));
    }

    #[test]
    fn test_error_parse() {
        let err = MutSpecError::parse("unexpected token");
        assert_eq!(err.category(), "parse");
        let msg = err.to_string();
        assert!(msg.contains("unexpected token"));
    }

    #[test]
    fn test_error_type_mismatch() {
        let err = MutSpecError::type_mismatch("bad arg", "int", "boolean");
        assert_eq!(err.category(), "type-check");
        match &err {
            MutSpecError::TypeCheck {
                expected, found, ..
            } => {
                assert_eq!(expected.as_deref(), Some("int"));
                assert_eq!(found.as_deref(), Some("boolean"));
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_error_smt_output() {
        let err = MutSpecError::smt_with_output("failed", "(error line 1)");
        match &err {
            MutSpecError::SmtSolver { solver_output, .. } => {
                assert!(solver_output.is_some());
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_error_timeout() {
        let err = MutSpecError::timeout("solver timed out", 30.0);
        assert!(err.is_transient());
        let msg = err.to_string();
        assert!(msg.contains("30.0"));
    }

    #[test]
    fn test_error_with_note() {
        let err = MutSpecError::analysis("failed")
            .with_note("check preconditions")
            .in_function("verify");
        let ctx = err.context().unwrap();
        assert_eq!(ctx.notes.len(), 1);
        assert_eq!(ctx.function_name.as_deref(), Some("verify"));
    }

    #[test]
    fn test_error_io_no_context() {
        let err = MutSpecError::io("read failed");
        assert!(err.context().is_none());
        assert!(err.is_transient());
    }

    #[test]
    fn test_error_config() {
        let err = MutSpecError::config_key("invalid value", "solver.timeout");
        assert_eq!(err.category(), "config");
        assert!(!err.is_transient());
    }

    #[test]
    fn test_error_unsupported() {
        let err = MutSpecError::unsupported("loops");
        assert_eq!(err.category(), "unsupported");
        let msg = err.to_string();
        assert!(msg.contains("loops"));
    }

    #[test]
    fn test_error_from_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "missing");
        let err: MutSpecError = io_err.into();
        assert_eq!(err.category(), "io");
    }

    #[test]
    fn test_error_from_fmt() {
        let fmt_err = fmt::Error;
        let err: MutSpecError = fmt_err.into();
        assert_eq!(err.category(), "internal");
    }

    #[test]
    fn test_error_context_default() {
        let ctx = ErrorContext::default();
        assert!(!ctx.has_location());
        assert!(ctx.function_name.is_none());
        assert!(ctx.notes.is_empty());
    }

    #[test]
    fn test_error_internal() {
        let err = MutSpecError::internal("assertion violated");
        assert_eq!(err.category(), "internal");
    }

    #[test]
    fn test_error_synthesis_tier() {
        let err = MutSpecError::synthesis_tier("no template match", "Tier2");
        match &err {
            MutSpecError::Synthesis { tier, .. } => {
                assert_eq!(tier.as_deref(), Some("Tier2"));
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_span_unknown() {
        let span = SpanInfo::unknown();
        assert!(span.is_unknown());
    }
}
