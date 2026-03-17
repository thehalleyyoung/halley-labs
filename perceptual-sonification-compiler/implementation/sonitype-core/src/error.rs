//! Comprehensive error types for the SoniType compiler.
//!
//! Provides a hierarchy of strongly-typed errors covering every compiler stage:
//! parsing, type checking, optimization, code generation, runtime, and
//! psychoacoustic constraint validation.

use serde::{Deserialize, Serialize};
use std::fmt;
use thiserror::Error;

// ===========================================================================
// Top-level error
// ===========================================================================

/// Top-level error type for the SoniType compiler.
#[derive(Debug, Error)]
pub enum SoniTypeError {
    #[error("Parse error: {0}")]
    Parse(#[from] ParseError),

    #[error("Type error: {0}")]
    Type(#[from] TypeError),

    #[error("Optimization error: {0}")]
    Optimization(#[from] OptimizationError),

    #[error("Code generation error: {0}")]
    Codegen(#[from] CodegenError),

    #[error("Runtime error: {0}")]
    Runtime(#[from] RuntimeError),

    #[error("Psychoacoustic error: {0}")]
    Psychoacoustic(#[from] PsychoacousticError),

    #[error("I/O error: {0}")]
    Io(String),

    #[error("Configuration error: {0}")]
    Config(#[from] ConfigError),

    #[error("Validation error: {0}")]
    Validation(#[from] ValidationError),
}

impl From<std::io::Error> for SoniTypeError {
    fn from(e: std::io::Error) -> Self {
        SoniTypeError::Io(e.to_string())
    }
}

// ===========================================================================
// Parse error
// ===========================================================================

/// Source location for error reporting.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceLocation {
    pub file: String,
    pub line: usize,
    pub column: usize,
}

impl SourceLocation {
    pub fn new(file: impl Into<String>, line: usize, column: usize) -> Self {
        Self { file: file.into(), line, column }
    }

    pub fn unknown() -> Self {
        Self { file: "<unknown>".into(), line: 0, column: 0 }
    }
}

impl fmt::Display for SourceLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}:{}", self.file, self.line, self.column)
    }
}

/// Error during parsing of SoniType mapping specifications.
#[derive(Clone, Debug, Error, Serialize, Deserialize)]
#[error("{message} at {location}")]
pub struct ParseError {
    pub message: String,
    pub location: SourceLocation,
    pub context: Option<String>,
}

impl ParseError {
    pub fn new(message: impl Into<String>, location: SourceLocation) -> Self {
        Self { message: message.into(), location, context: None }
    }

    pub fn with_context(mut self, ctx: impl Into<String>) -> Self {
        self.context = Some(ctx.into());
        self
    }

    pub fn at_line(message: impl Into<String>, file: impl Into<String>, line: usize) -> Self {
        Self::new(message, SourceLocation::new(file, line, 0))
    }

    pub fn unexpected_token(token: &str, expected: &str, loc: SourceLocation) -> Self {
        Self::new(
            format!("unexpected token '{token}', expected {expected}"),
            loc,
        )
    }

    pub fn unexpected_eof(loc: SourceLocation) -> Self {
        Self::new("unexpected end of input", loc)
    }
}

// ===========================================================================
// Type error
// ===========================================================================

/// Kind of constraint that was violated.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConstraintViolationKind {
    RangeBound,
    Uniqueness,
    Compatibility,
    Subtype,
    MissingField,
    ExtraField,
    Custom(String),
}

impl fmt::Display for ConstraintViolationKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConstraintViolationKind::RangeBound => write!(f, "range bound"),
            ConstraintViolationKind::Uniqueness => write!(f, "uniqueness"),
            ConstraintViolationKind::Compatibility => write!(f, "compatibility"),
            ConstraintViolationKind::Subtype => write!(f, "subtype"),
            ConstraintViolationKind::MissingField => write!(f, "missing field"),
            ConstraintViolationKind::ExtraField => write!(f, "extra field"),
            ConstraintViolationKind::Custom(s) => write!(f, "{s}"),
        }
    }
}

/// Error when perceptual type constraints are violated.
#[derive(Clone, Debug, Error, Serialize, Deserialize)]
#[error("type error: expected {expected}, found {found}{}", format_constraint(.constraint))]
pub struct TypeError {
    pub expected: String,
    pub found: String,
    pub constraint: Option<ConstraintViolationKind>,
    pub location: Option<SourceLocation>,
    pub notes: Vec<String>,
}

fn format_constraint(c: &Option<ConstraintViolationKind>) -> String {
    match c {
        Some(k) => format!(" (violated: {k})"),
        None => String::new(),
    }
}

impl TypeError {
    pub fn new(expected: impl Into<String>, found: impl Into<String>) -> Self {
        Self {
            expected: expected.into(), found: found.into(),
            constraint: None, location: None, notes: Vec::new(),
        }
    }

    pub fn with_constraint(mut self, kind: ConstraintViolationKind) -> Self {
        self.constraint = Some(kind);
        self
    }

    pub fn with_location(mut self, loc: SourceLocation) -> Self {
        self.location = Some(loc);
        self
    }

    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }

    pub fn incompatible_mapping(data_type: &str, audio_param: &str) -> Self {
        Self::new(
            format!("compatible mapping for {audio_param}"),
            format!("data type {data_type}"),
        ).with_constraint(ConstraintViolationKind::Compatibility)
    }
}

// ===========================================================================
// Optimization error
// ===========================================================================

/// Reason an optimization failed.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationFailureReason {
    Infeasible,
    Timeout,
    ConvergenceFailure,
    NumericalInstability,
    NoImprovement,
}

impl fmt::Display for OptimizationFailureReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Infeasible => write!(f, "infeasible constraints"),
            Self::Timeout => write!(f, "time budget exceeded"),
            Self::ConvergenceFailure => write!(f, "failed to converge"),
            Self::NumericalInstability => write!(f, "numerical instability"),
            Self::NoImprovement => write!(f, "no improvement found"),
        }
    }
}

/// Error during optimization of sonification mappings.
#[derive(Clone, Debug, Error, Serialize, Deserialize)]
#[error("optimization error: {reason} after {iterations} iterations (best cost: {best_cost:.6})")]
pub struct OptimizationError {
    pub reason: OptimizationFailureReason,
    pub iterations: usize,
    pub best_cost: f64,
    pub message: String,
}

impl OptimizationError {
    pub fn new(reason: OptimizationFailureReason, message: impl Into<String>) -> Self {
        Self { reason, iterations: 0, best_cost: f64::INFINITY, message: message.into() }
    }

    pub fn infeasible(msg: impl Into<String>) -> Self {
        Self::new(OptimizationFailureReason::Infeasible, msg)
    }

    pub fn timeout(iterations: usize, best_cost: f64) -> Self {
        Self {
            reason: OptimizationFailureReason::Timeout,
            iterations, best_cost,
            message: "optimizer time budget exceeded".into(),
        }
    }

    pub fn convergence_failure(iterations: usize, best_cost: f64) -> Self {
        Self {
            reason: OptimizationFailureReason::ConvergenceFailure,
            iterations, best_cost,
            message: "optimizer failed to converge".into(),
        }
    }
}

// ===========================================================================
// Code generation error
// ===========================================================================

/// Error during code generation.
#[derive(Clone, Debug, Error, Serialize, Deserialize)]
#[error("codegen error: {message}")]
pub struct CodegenError {
    pub message: String,
    pub phase: String,
    pub node_id: Option<u64>,
}

impl CodegenError {
    pub fn new(message: impl Into<String>, phase: impl Into<String>) -> Self {
        Self { message: message.into(), phase: phase.into(), node_id: None }
    }

    pub fn with_node(mut self, id: u64) -> Self {
        self.node_id = Some(id);
        self
    }

    pub fn unsupported_node(node_type: &str) -> Self {
        Self::new(format!("unsupported node type: {node_type}"), "ir_lowering")
    }
}

// ===========================================================================
// Psychoacoustic error
// ===========================================================================

/// Specific psychoacoustic constraint that was violated.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PsychoacousticViolation {
    MaskingViolation,
    JndViolation,
    SegregationFailure,
    CognitiveOverload,
    FrequencyRangeExceeded,
    TemporalResolutionExceeded,
}

impl fmt::Display for PsychoacousticViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MaskingViolation => write!(f, "masking threshold violated"),
            Self::JndViolation => write!(f, "below just-noticeable difference"),
            Self::SegregationFailure => write!(f, "stream segregation failure"),
            Self::CognitiveOverload => write!(f, "cognitive load exceeded"),
            Self::FrequencyRangeExceeded => write!(f, "frequency outside audible range"),
            Self::TemporalResolutionExceeded => write!(f, "temporal resolution exceeded"),
        }
    }
}

/// Error when psychoacoustic constraints are violated.
#[derive(Clone, Debug, Error, Serialize, Deserialize)]
#[error("psychoacoustic error: {violation} - {detail}")]
pub struct PsychoacousticError {
    pub violation: PsychoacousticViolation,
    pub detail: String,
    pub severity: f64,
    pub stream_ids: Vec<u64>,
}

impl PsychoacousticError {
    pub fn new(violation: PsychoacousticViolation, detail: impl Into<String>) -> Self {
        Self { violation, detail: detail.into(), severity: 1.0, stream_ids: Vec::new() }
    }

    pub fn with_severity(mut self, s: f64) -> Self {
        self.severity = s;
        self
    }

    pub fn with_streams(mut self, ids: Vec<u64>) -> Self {
        self.stream_ids = ids;
        self
    }

    pub fn masking(detail: impl Into<String>) -> Self {
        Self::new(PsychoacousticViolation::MaskingViolation, detail)
    }

    pub fn jnd(detail: impl Into<String>) -> Self {
        Self::new(PsychoacousticViolation::JndViolation, detail)
    }

    pub fn segregation(detail: impl Into<String>) -> Self {
        Self::new(PsychoacousticViolation::SegregationFailure, detail)
    }

    pub fn cognitive_overload(current: f64, max: f64) -> Self {
        Self::new(
            PsychoacousticViolation::CognitiveOverload,
            format!("cognitive load {current:.2} exceeds budget {max:.2}"),
        )
    }
}

// ===========================================================================
// Runtime error
// ===========================================================================

/// Error during audio rendering at runtime.
#[derive(Clone, Debug, Error, Serialize, Deserialize)]
pub enum RuntimeError {
    #[error("buffer underrun: needed {needed} samples, only {available} available")]
    BufferUnderrun { needed: usize, available: usize },

    #[error("WCET violation: node {node_id} took {actual_us} us, budget was {budget_us} us")]
    WcetViolation { node_id: u64, actual_us: u64, budget_us: u64 },

    #[error("sample rate mismatch: expected {expected}, got {actual}")]
    SampleRateMismatch { expected: u32, actual: u32 },

    #[error("channel count mismatch: expected {expected}, got {actual}")]
    ChannelMismatch { expected: usize, actual: usize },

    #[error("audio device error: {message}")]
    DeviceError { message: String },

    #[error("graph cycle detected involving node {node_id}")]
    GraphCycle { node_id: u64 },

    #[error("runtime error: {0}")]
    Other(String),
}

impl RuntimeError {
    pub fn buffer_underrun(needed: usize, available: usize) -> Self {
        Self::BufferUnderrun { needed, available }
    }

    pub fn wcet_violation(node_id: u64, actual_us: u64, budget_us: u64) -> Self {
        Self::WcetViolation { node_id, actual_us, budget_us }
    }

    pub fn sample_rate_mismatch(expected: u32, actual: u32) -> Self {
        Self::SampleRateMismatch { expected, actual }
    }

    pub fn device_error(msg: impl Into<String>) -> Self {
        Self::DeviceError { message: msg.into() }
    }
}

// ===========================================================================
// Configuration error
// ===========================================================================

/// Error in compiler configuration.
#[derive(Clone, Debug, Error, Serialize, Deserialize)]
#[error("config error: {message}")]
pub struct ConfigError {
    pub message: String,
    pub field: Option<String>,
}

impl ConfigError {
    pub fn new(message: impl Into<String>) -> Self {
        Self { message: message.into(), field: None }
    }

    pub fn invalid_field(field: impl Into<String>, reason: impl Into<String>) -> Self {
        let f = field.into();
        Self { message: format!("invalid value for '{f}': {}", reason.into()), field: Some(f) }
    }
}

// ===========================================================================
// Validation error
// ===========================================================================

/// Kind of validation failure.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValidationKind {
    SchemaMismatch,
    RangeViolation,
    ConstraintUnsatisfiable,
    MissingRequired,
    InvalidFormat,
}

impl fmt::Display for ValidationKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SchemaMismatch => write!(f, "schema mismatch"),
            Self::RangeViolation => write!(f, "range violation"),
            Self::ConstraintUnsatisfiable => write!(f, "constraint unsatisfiable"),
            Self::MissingRequired => write!(f, "missing required"),
            Self::InvalidFormat => write!(f, "invalid format"),
        }
    }
}

/// Error during input validation.
#[derive(Clone, Debug, Error, Serialize, Deserialize)]
#[error("validation error ({kind}): {message}")]
pub struct ValidationError {
    pub kind: ValidationKind,
    pub message: String,
    pub field: Option<String>,
}

impl ValidationError {
    pub fn new(kind: ValidationKind, message: impl Into<String>) -> Self {
        Self { kind, message: message.into(), field: None }
    }

    pub fn with_field(mut self, field: impl Into<String>) -> Self {
        self.field = Some(field.into());
        self
    }

    pub fn schema_mismatch(msg: impl Into<String>) -> Self {
        Self::new(ValidationKind::SchemaMismatch, msg)
    }

    pub fn range_violation(field: &str, value: f64, min: f64, max: f64) -> Self {
        Self::new(
            ValidationKind::RangeViolation,
            format!("'{field}' value {value} outside [{min}, {max}]"),
        ).with_field(field)
    }

    pub fn constraint_unsatisfiable(msg: impl Into<String>) -> Self {
        Self::new(ValidationKind::ConstraintUnsatisfiable, msg)
    }

    pub fn missing_required(field: impl Into<String>) -> Self {
        let f = field.into();
        Self::new(ValidationKind::MissingRequired, format!("required field '{f}' is missing"))
            .with_field(f)
    }
}

// ===========================================================================
// Error context wrapper
// ===========================================================================

/// Wraps an error with additional context.
#[derive(Debug)]
pub struct ErrorContext<E: std::error::Error> {
    pub context: String,
    pub source: E,
}

impl<E: std::error::Error> fmt::Display for ErrorContext<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.context, self.source)
    }
}

impl<E: std::error::Error + 'static> std::error::Error for ErrorContext<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.source)
    }
}

/// Extension trait to add context to any error.
pub trait WithContext<T, E: std::error::Error> {
    fn with_context(self, ctx: impl Into<String>) -> Result<T, ErrorContext<E>>;
}

impl<T, E: std::error::Error> WithContext<T, E> for Result<T, E> {
    fn with_context(self, ctx: impl Into<String>) -> Result<T, ErrorContext<E>> {
        self.map_err(|e| ErrorContext { context: ctx.into(), source: e })
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_error_display() {
        let e = ParseError::new("missing semicolon", SourceLocation::new("main.st", 10, 5));
        let s = format!("{e}");
        assert!(s.contains("missing semicolon"));
        assert!(s.contains("main.st:10:5"));
    }

    #[test]
    fn parse_error_with_context() {
        let e = ParseError::new("bad token", SourceLocation::unknown())
            .with_context("while parsing mapping block");
        assert!(e.context.is_some());
    }

    #[test]
    fn type_error_display() {
        let e = TypeError::new("Frequency", "String");
        let s = format!("{e}");
        assert!(s.contains("Frequency"));
        assert!(s.contains("String"));
    }

    #[test]
    fn type_error_with_constraint() {
        let e = TypeError::new("Bark band 0-23", "25")
            .with_constraint(ConstraintViolationKind::RangeBound);
        let s = format!("{e}");
        assert!(s.contains("range bound"));
    }

    #[test]
    fn optimization_error_infeasible() {
        let e = OptimizationError::infeasible("no valid pitch mapping exists");
        assert_eq!(e.reason, OptimizationFailureReason::Infeasible);
    }

    #[test]
    fn optimization_error_timeout() {
        let e = OptimizationError::timeout(500, 3.14);
        assert_eq!(e.iterations, 500);
        assert!((e.best_cost - 3.14).abs() < 1e-10);
    }

    #[test]
    fn codegen_error_display() {
        let e = CodegenError::unsupported_node("ConvolveNode");
        assert!(format!("{e}").contains("ConvolveNode"));
    }

    #[test]
    fn psychoacoustic_error_masking() {
        let e = PsychoacousticError::masking("stream A masks stream B at 1 kHz")
            .with_severity(0.8)
            .with_streams(vec![1, 2]);
        assert_eq!(e.stream_ids, vec![1, 2]);
        assert!((e.severity - 0.8).abs() < 1e-10);
    }

    #[test]
    fn psychoacoustic_error_cognitive() {
        let e = PsychoacousticError::cognitive_overload(5.5, 4.0);
        assert!(format!("{e}").contains("5.50"));
    }

    #[test]
    fn runtime_error_variants() {
        let e = RuntimeError::buffer_underrun(1024, 512);
        assert!(format!("{e}").contains("1024"));

        let e = RuntimeError::wcet_violation(7, 1500, 1000);
        assert!(format!("{e}").contains("1500"));

        let e = RuntimeError::sample_rate_mismatch(44100, 48000);
        assert!(format!("{e}").contains("44100"));
    }

    #[test]
    fn config_error_field() {
        let e = ConfigError::invalid_field("buffer_size", "must be power of two");
        assert!(e.field.is_some());
        assert!(format!("{e}").contains("buffer_size"));
    }

    #[test]
    fn validation_error_kinds() {
        let e = ValidationError::range_violation("frequency", 25000.0, 20.0, 20000.0);
        assert_eq!(e.kind, ValidationKind::RangeViolation);
        assert!(e.field.is_some());

        let e = ValidationError::missing_required("sample_rate");
        assert_eq!(e.kind, ValidationKind::MissingRequired);
    }

    #[test]
    fn sonitype_error_from_parse() {
        let pe = ParseError::new("bad", SourceLocation::unknown());
        let se: SoniTypeError = pe.into();
        assert!(matches!(se, SoniTypeError::Parse(_)));
    }

    #[test]
    fn sonitype_error_from_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let se: SoniTypeError = io_err.into();
        assert!(matches!(se, SoniTypeError::Io(_)));
    }

    #[test]
    fn error_context_chain() {
        use std::error::Error;
        let inner = ConfigError::new("bad value");
        let wrapped: Result<(), _> = Err(inner);
        let ctx = wrapped.with_context("loading config file");
        assert!(ctx.is_err());
        let e = ctx.unwrap_err();
        assert!(format!("{e}").contains("loading config file"));
        assert!(e.source().is_some());
    }

    #[test]
    fn source_location_display() {
        let loc = SourceLocation::new("test.st", 42, 7);
        assert_eq!(format!("{loc}"), "test.st:42:7");
    }
}
