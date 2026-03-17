//! Comprehensive error types for the NegSynth pipeline.
//!
//! Provides a unified error hierarchy covering all phases of the analysis
//! pipeline: slicing, merging, extraction, encoding, and concretization.

use std::fmt;

/// Top-level error type for the NegSynth pipeline.
#[derive(Debug, thiserror::Error)]
pub enum NegSynthError {
    #[error("slicer error: {0}")]
    Slicer(#[from] SlicerError),

    #[error("merge error: {0}")]
    Merge(#[from] MergeError),

    #[error("extraction error: {0}")]
    Extraction(#[from] ExtractionError),

    #[error("encoding error: {0}")]
    Encoding(#[from] EncodingError),

    #[error("concretization error: {0}")]
    Concretization(#[from] ConcretizationError),

    #[error("protocol error: {0}")]
    Protocol(#[from] ProtocolError),

    #[error("graph error: {0}")]
    Graph(#[from] GraphError),

    #[error("SMT error: {0}")]
    Smt(#[from] SmtError),

    #[error("certificate error: {0}")]
    Certificate(#[from] CertificateError),

    #[error("configuration error: {0}")]
    Config(#[from] ConfigError),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("internal error: {message}")]
    Internal {
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error("timeout after {duration_ms}ms in phase {phase}")]
    Timeout { phase: String, duration_ms: u64 },

    #[error("resource exhaustion: {resource} (limit: {limit}, used: {used})")]
    ResourceExhaustion {
        resource: String,
        limit: u64,
        used: u64,
    },
}

impl NegSynthError {
    pub fn internal(message: impl Into<String>) -> Self {
        NegSynthError::Internal {
            message: message.into(),
            source: None,
        }
    }

    pub fn internal_with_source(
        message: impl Into<String>,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        NegSynthError::Internal {
            message: message.into(),
            source: Some(Box::new(source)),
        }
    }

    pub fn timeout(phase: impl Into<String>, duration_ms: u64) -> Self {
        NegSynthError::Timeout {
            phase: phase.into(),
            duration_ms,
        }
    }

    pub fn resource_exhaustion(resource: impl Into<String>, limit: u64, used: u64) -> Self {
        NegSynthError::ResourceExhaustion {
            resource: resource.into(),
            limit,
            used,
        }
    }

    /// Returns the pipeline phase where this error occurred.
    pub fn phase(&self) -> &str {
        match self {
            NegSynthError::Slicer(_) => "slicer",
            NegSynthError::Merge(_) => "merge",
            NegSynthError::Extraction(_) => "extraction",
            NegSynthError::Encoding(_) => "encoding",
            NegSynthError::Concretization(_) => "concretization",
            NegSynthError::Protocol(_) => "protocol",
            NegSynthError::Graph(_) => "graph",
            NegSynthError::Smt(_) => "smt",
            NegSynthError::Certificate(_) => "certificate",
            NegSynthError::Config(_) => "config",
            NegSynthError::Io(_) => "io",
            NegSynthError::Serialization(_) => "serialization",
            NegSynthError::Internal { .. } => "internal",
            NegSynthError::Timeout { phase, .. } => phase,
            NegSynthError::ResourceExhaustion { .. } => "resource",
        }
    }

    /// Whether this error is recoverable (the pipeline can continue).
    pub fn is_recoverable(&self) -> bool {
        match self {
            NegSynthError::Merge(MergeError::IncompatibleStates { .. }) => true,
            NegSynthError::Timeout { .. } => true,
            NegSynthError::Smt(SmtError::Unknown { .. }) => true,
            _ => false,
        }
    }
}

/// Error context wrapper that attaches location and phase info.
#[derive(Debug)]
pub struct ErrorContext {
    pub error: NegSynthError,
    pub phase: String,
    pub location: Option<String>,
    pub details: Vec<String>,
}

impl ErrorContext {
    pub fn new(error: NegSynthError, phase: impl Into<String>) -> Self {
        ErrorContext {
            error,
            phase: phase.into(),
            location: None,
            details: Vec::new(),
        }
    }

    pub fn with_location(mut self, location: impl Into<String>) -> Self {
        self.location = Some(location.into());
        self
    }

    pub fn with_detail(mut self, detail: impl Into<String>) -> Self {
        self.details.push(detail.into());
        self
    }
}

impl fmt::Display for ErrorContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}]", self.phase)?;
        if let Some(ref loc) = self.location {
            write!(f, " at {}", loc)?;
        }
        write!(f, " {}", self.error)?;
        for detail in &self.details {
            write!(f, "\n  detail: {}", detail)?;
        }
        Ok(())
    }
}

impl std::error::Error for ErrorContext {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

// ── Sub-error types ──────────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum SlicerError {
    #[error("failed to parse binary at offset 0x{offset:x}: {reason}")]
    ParseFailure { offset: u64, reason: String },

    #[error("unsupported instruction: {mnemonic} at 0x{address:x}")]
    UnsupportedInstruction { address: u64, mnemonic: String },

    #[error("CFG construction failed: {reason}")]
    CfgConstructionFailed { reason: String },

    #[error("function {name} not found in binary")]
    FunctionNotFound { name: String },

    #[error("slice criterion invalid: {reason}")]
    InvalidCriterion { reason: String },

    #[error("dependency analysis exceeded depth {depth}")]
    DepthExceeded { depth: u32 },

    #[error("unsupported binary format: {format}")]
    UnsupportedFormat { format: String },
}

impl SlicerError {
    pub fn parse_failure(offset: u64, reason: impl Into<String>) -> Self {
        SlicerError::ParseFailure {
            offset,
            reason: reason.into(),
        }
    }

    pub fn function_not_found(name: impl Into<String>) -> Self {
        SlicerError::FunctionNotFound { name: name.into() }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum MergeError {
    #[error("incompatible states at merge point: {reason}")]
    IncompatibleStates { reason: String },

    #[error("merge budget exceeded ({used}/{limit} merges)")]
    BudgetExceeded { used: u32, limit: u32 },

    #[error("widening failed at iteration {iteration}: {reason}")]
    WideningFailed { iteration: u32, reason: String },

    #[error("merge predicate evaluation failed: {reason}")]
    PredicateFailed { reason: String },

    #[error("states at addresses {addr_a:#x} and {addr_b:#x} are structurally incompatible")]
    StructuralMismatch { addr_a: u64, addr_b: u64 },

    #[error("fixed point not reached after {iterations} iterations")]
    NoFixedPoint { iterations: u32 },

    #[error("expression complexity exceeded: {reason} (complexity: {complexity}, limit: {limit})")]
    ComplexityExceeded { reason: String, complexity: u32, limit: u32 },

    #[error("constraint limit exceeded: {count} constraints (max: {limit})")]
    ConstraintLimitExceeded { count: u32, limit: u32 },
}

impl MergeError {
    pub fn incompatible(reason: impl Into<String>) -> Self {
        MergeError::IncompatibleStates {
            reason: reason.into(),
        }
    }

    pub fn budget_exceeded(used: u32, limit: u32) -> Self {
        MergeError::BudgetExceeded { used, limit }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ExtractionError {
    #[error("no negotiation protocol found in slice")]
    NoProtocolFound,

    #[error("ambiguous negotiation: {count} candidates found")]
    AmbiguousProtocol { count: usize },

    #[error("cipher suite extraction failed: {reason}")]
    CipherSuiteExtractionFailed { reason: String },

    #[error("version negotiation pattern not recognized: {pattern}")]
    UnrecognizedPattern { pattern: String },

    #[error("incomplete extraction: missing {field}")]
    IncompleteExtraction { field: String },

    #[error("type inference failed for variable {var}: {reason}")]
    TypeInferenceFailed { var: String, reason: String },
}

impl ExtractionError {
    pub fn cipher_suite_failed(reason: impl Into<String>) -> Self {
        ExtractionError::CipherSuiteExtractionFailed {
            reason: reason.into(),
        }
    }

    pub fn incomplete(field: impl Into<String>) -> Self {
        ExtractionError::IncompleteExtraction {
            field: field.into(),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum EncodingError {
    #[error("unsupported SMT sort: {sort}")]
    UnsupportedSort { sort: String },

    #[error("encoding overflow: expression depth {depth} exceeds limit {limit}")]
    DepthOverflow { depth: u32, limit: u32 },

    #[error("unresolved symbolic variable: {name}")]
    UnresolvedVariable { name: String },

    #[error("bit-width mismatch: expected {expected}, got {actual}")]
    BitWidthMismatch { expected: u32, actual: u32 },

    #[error("quantifier encoding failed: {reason}")]
    QuantifierFailed { reason: String },

    #[error("theory combination unsupported: {theories}")]
    TheoryCombination { theories: String },
}

impl EncodingError {
    pub fn unsupported_sort(sort: impl Into<String>) -> Self {
        EncodingError::UnsupportedSort { sort: sort.into() }
    }

    pub fn bit_width_mismatch(expected: u32, actual: u32) -> Self {
        EncodingError::BitWidthMismatch { expected, actual }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ConcretizationError {
    #[error("model value out of range for {field}: {value}")]
    ValueOutOfRange { field: String, value: String },

    #[error("no concrete attack: adversary actions exceed budget")]
    BudgetViolation,

    #[error("replay failed at step {step}: {reason}")]
    ReplayFailed { step: usize, reason: String },

    #[error("network constraint violation: {reason}")]
    NetworkConstraint { reason: String },

    #[error("timing constraint unsatisfiable: {reason}")]
    TimingConstraint { reason: String },

    #[error("concretization produced invalid cipher suite id: 0x{id:04x}")]
    InvalidCipherSuite { id: u16 },
}

impl ConcretizationError {
    pub fn value_out_of_range(field: impl Into<String>, value: impl Into<String>) -> Self {
        ConcretizationError::ValueOutOfRange {
            field: field.into(),
            value: value.into(),
        }
    }

    pub fn replay_failed(step: usize, reason: impl Into<String>) -> Self {
        ConcretizationError::ReplayFailed {
            step,
            reason: reason.into(),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ProtocolError {
    #[error("unknown cipher suite: 0x{id:04x}")]
    UnknownCipherSuite { id: u16 },

    #[error("invalid protocol version: {major}.{minor}")]
    InvalidVersion { major: u8, minor: u8 },

    #[error("invalid state transition from {from} to {to}")]
    InvalidTransition { from: String, to: String },

    #[error("extension {id} not supported in {context}")]
    UnsupportedExtension { id: u16, context: String },

    #[error("negotiation failed: {reason}")]
    NegotiationFailed { reason: String },
}

impl ProtocolError {
    pub fn unknown_cipher_suite(id: u16) -> Self {
        ProtocolError::UnknownCipherSuite { id }
    }

    pub fn invalid_version(major: u8, minor: u8) -> Self {
        ProtocolError::InvalidVersion { major, minor }
    }

    pub fn invalid_transition(from: impl Into<String>, to: impl Into<String>) -> Self {
        ProtocolError::InvalidTransition {
            from: from.into(),
            to: to.into(),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum GraphError {
    #[error("state {0} not found in graph")]
    StateNotFound(u32),

    #[error("transition {0} not found in graph")]
    TransitionNotFound(u32),

    #[error("graph has no initial state")]
    NoInitialState,

    #[error("cycle detected involving state {0}")]
    CycleDetected(u32),

    #[error("bisimulation computation diverged after {0} iterations")]
    BisimulationDiverged(u32),

    #[error("quotient construction failed: {0}")]
    QuotientFailed(String),

    #[error("graph too large: {states} states, {transitions} transitions (limit: {limit})")]
    TooLarge {
        states: usize,
        transitions: usize,
        limit: usize,
    },
}

#[derive(Debug, thiserror::Error)]
pub enum SmtError {
    #[error("solver returned unknown: {reason}")]
    Unknown { reason: String },

    #[error("sort mismatch: expected {expected}, got {actual}")]
    SortMismatch { expected: String, actual: String },

    #[error("solver timeout after {ms}ms")]
    Timeout { ms: u64 },

    #[error("invalid model: {reason}")]
    InvalidModel { reason: String },

    #[error("unsupported theory: {theory}")]
    UnsupportedTheory { theory: String },

    #[error("expression too complex: {nodes} nodes (limit: {limit})")]
    ExpressionTooComplex { nodes: usize, limit: usize },
}

impl SmtError {
    pub fn unknown(reason: impl Into<String>) -> Self {
        SmtError::Unknown {
            reason: reason.into(),
        }
    }

    pub fn sort_mismatch(expected: impl Into<String>, actual: impl Into<String>) -> Self {
        SmtError::SortMismatch {
            expected: expected.into(),
            actual: actual.into(),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum CertificateError {
    #[error("certificate verification failed: {reason}")]
    VerificationFailed { reason: String },

    #[error("certificate expired: issued {issued}, expired {expired}")]
    Expired { issued: String, expired: String },

    #[error("invalid certificate chain at position {position}: {reason}")]
    InvalidChain { position: usize, reason: String },

    #[error("hash mismatch: expected {expected}, got {actual}")]
    HashMismatch { expected: String, actual: String },

    #[error("missing required field: {field}")]
    MissingField { field: String },

    #[error("unsupported certificate version: {version}")]
    UnsupportedVersion { version: u32 },
}

impl CertificateError {
    pub fn verification_failed(reason: impl Into<String>) -> Self {
        CertificateError::VerificationFailed {
            reason: reason.into(),
        }
    }

    pub fn hash_mismatch(expected: impl Into<String>, actual: impl Into<String>) -> Self {
        CertificateError::HashMismatch {
            expected: expected.into(),
            actual: actual.into(),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("invalid configuration: {field} - {reason}")]
    InvalidField { field: String, reason: String },

    #[error("missing required configuration: {field}")]
    MissingField { field: String },

    #[error("configuration conflict: {a} and {b} are mutually exclusive")]
    Conflict { a: String, b: String },

    #[error("value out of range for {field}: {value} (range: {min}..{max})")]
    OutOfRange {
        field: String,
        value: String,
        min: String,
        max: String,
    },
}

impl ConfigError {
    pub fn invalid_field(field: impl Into<String>, reason: impl Into<String>) -> Self {
        ConfigError::InvalidField {
            field: field.into(),
            reason: reason.into(),
        }
    }

    pub fn missing_field(field: impl Into<String>) -> Self {
        ConfigError::MissingField {
            field: field.into(),
        }
    }
}

/// Convenience result type alias.
pub type NegSynthResult<T> = Result<T, NegSynthError>;

/// Extension trait for adding context to results.
pub trait ResultExt<T> {
    fn with_phase(self, phase: impl Into<String>) -> Result<T, ErrorContext>;
    fn with_context_detail(self, phase: impl Into<String>, detail: impl Into<String>) -> Result<T, ErrorContext>;
}

impl<T> ResultExt<T> for NegSynthResult<T> {
    fn with_phase(self, phase: impl Into<String>) -> Result<T, ErrorContext> {
        self.map_err(|e| ErrorContext::new(e, phase))
    }

    fn with_context_detail(self, phase: impl Into<String>, detail: impl Into<String>) -> Result<T, ErrorContext> {
        self.map_err(|e| ErrorContext::new(e, phase).with_detail(detail))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_phase() {
        let err = NegSynthError::Slicer(SlicerError::FunctionNotFound {
            name: "test".into(),
        });
        assert_eq!(err.phase(), "slicer");

        let err = NegSynthError::timeout("encoding", 5000);
        assert_eq!(err.phase(), "encoding");
    }

    #[test]
    fn test_error_display() {
        let err = SlicerError::parse_failure(0x1234, "unexpected byte");
        let msg = format!("{}", err);
        assert!(msg.contains("0x1234"));
        assert!(msg.contains("unexpected byte"));
    }

    #[test]
    fn test_error_context() {
        let err = NegSynthError::internal("something broke");
        let ctx = ErrorContext::new(err, "slicer")
            .with_location("main.rs:42")
            .with_detail("while processing function foo");
        let msg = format!("{}", ctx);
        assert!(msg.contains("[slicer]"));
        assert!(msg.contains("main.rs:42"));
        assert!(msg.contains("something broke"));
    }

    #[test]
    fn test_recoverable() {
        let err = NegSynthError::Merge(MergeError::incompatible("types differ"));
        assert!(err.is_recoverable());

        let err = NegSynthError::internal("fatal");
        assert!(!err.is_recoverable());
    }

    #[test]
    fn test_error_conversions() {
        let slicer_err = SlicerError::function_not_found("tls_negotiate");
        let neg_err: NegSynthError = slicer_err.into();
        assert_eq!(neg_err.phase(), "slicer");

        let merge_err = MergeError::budget_exceeded(100, 50);
        let neg_err: NegSynthError = merge_err.into();
        assert_eq!(neg_err.phase(), "merge");
    }

    #[test]
    fn test_result_ext() {
        let result: NegSynthResult<i32> = Err(NegSynthError::internal("oops"));
        let ctx_result = result.with_phase("test-phase");
        assert!(ctx_result.is_err());
        let err = ctx_result.unwrap_err();
        assert_eq!(err.phase, "test-phase");
    }

    #[test]
    fn test_sub_error_constructors() {
        let e = EncodingError::bit_width_mismatch(32, 64);
        let msg = format!("{}", e);
        assert!(msg.contains("32"));
        assert!(msg.contains("64"));

        let e = ConcretizationError::replay_failed(5, "timeout");
        let msg = format!("{}", e);
        assert!(msg.contains("step 5"));

        let e = SmtError::sort_mismatch("Bool", "BitVec(32)");
        let msg = format!("{}", e);
        assert!(msg.contains("Bool"));
    }

    #[test]
    fn test_resource_exhaustion() {
        let err = NegSynthError::resource_exhaustion("memory", 1024, 2048);
        let msg = format!("{}", err);
        assert!(msg.contains("memory"));
        assert!(msg.contains("1024"));
        assert!(msg.contains("2048"));
    }

    #[test]
    fn test_graph_error_display() {
        let err = GraphError::TooLarge {
            states: 10000,
            transitions: 50000,
            limit: 5000,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("10000"));
        assert!(msg.contains("50000"));
    }

    #[test]
    fn test_certificate_error() {
        let err = CertificateError::hash_mismatch("abc123", "def456");
        let msg = format!("{}", err);
        assert!(msg.contains("abc123"));
        assert!(msg.contains("def456"));
    }
}
