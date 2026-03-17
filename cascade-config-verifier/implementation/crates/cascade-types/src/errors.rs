use serde::{Deserialize, Serialize};
use thiserror::Error;

use std::fmt;

// ---------------------------------------------------------------------------
// CascadeError – top-level error enum
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum CascadeError {
    #[error("configuration error: {0}")]
    Config(#[from] ConfigError),

    #[error("parse error: {0}")]
    Parse(#[from] ParseError),

    #[error("analysis error: {0}")]
    Analysis(#[from] AnalysisError),

    #[error("SMT error: {0}")]
    Smt(#[from] SmtError),

    #[error("repair error: {0}")]
    Repair(#[from] RepairError),

    #[error("validation error: {0}")]
    Validation(#[from] ValidationError),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("serialization error: {0}")]
    Serialization(String),

    #[error("internal error: {0}")]
    Internal(String),
}

// ---------------------------------------------------------------------------
// ConfigError
// ---------------------------------------------------------------------------

#[derive(Debug, Error, Clone, Serialize, Deserialize)]
pub enum ConfigError {
    #[error("missing required field `{field}` in {context}")]
    MissingField { field: String, context: String },

    #[error("invalid value `{value}` for field `{field}`: {reason}")]
    InvalidValue {
        field: String,
        value: String,
        reason: String,
    },

    #[error("conflicting configuration: {0}")]
    Conflict(String),

    #[error("unsupported config version `{version}` in {config_source}")]
    UnsupportedVersion { version: String, config_source: String },

    #[error("duplicate key `{key}` in {context}")]
    DuplicateKey { key: String, context: String },

    #[error("config source not found: {0}")]
    NotFound(String),

    #[error("config format error: {0}")]
    Format(String),
}

// ---------------------------------------------------------------------------
// ParseError
// ---------------------------------------------------------------------------

#[derive(Debug, Error, Clone, Serialize, Deserialize)]
pub enum ParseError {
    #[error("YAML parse error at line {line}: {message}")]
    Yaml { line: usize, message: String },

    #[error("JSON parse error: {0}")]
    Json(String),

    #[error("invalid service reference `{reference}`: {reason}")]
    InvalidReference { reference: String, reason: String },

    #[error("malformed endpoint `{endpoint}`: {reason}")]
    MalformedEndpoint { endpoint: String, reason: String },

    #[error("unexpected token `{token}` at position {position}")]
    UnexpectedToken { token: String, position: usize },

    #[error("incomplete input: expected {expected}")]
    IncompleteInput { expected: String },

    #[error("encoding error: {0}")]
    Encoding(String),
}

// ---------------------------------------------------------------------------
// AnalysisError
// ---------------------------------------------------------------------------

#[derive(Debug, Error, Clone, Serialize, Deserialize)]
pub enum AnalysisError {
    #[error("service `{0}` not found in topology")]
    ServiceNotFound(String),

    #[error("cycle detected in topology: {0}")]
    CycleDetected(String),

    #[error("analysis timeout after {0}ms")]
    Timeout(u64),

    #[error("exceeded maximum depth {max_depth} at service `{service}`")]
    MaxDepthExceeded { service: String, max_depth: usize },

    #[error("empty topology: no services to analyze")]
    EmptyTopology,

    #[error("convergence failure: {0}")]
    ConvergenceFailure(String),

    #[error("capacity overflow at service `{service}`: load {load} exceeds capacity {capacity}")]
    CapacityOverflow {
        service: String,
        load: f64,
        capacity: f64,
    },

    #[error("invalid failure set: {0}")]
    InvalidFailureSet(String),

    #[error("unsupported analysis mode: {0}")]
    UnsupportedMode(String),
}

// ---------------------------------------------------------------------------
// SmtError
// ---------------------------------------------------------------------------

#[derive(Debug, Error, Clone, Serialize, Deserialize)]
pub enum SmtError {
    #[error("SMT encoding error: {0}")]
    Encoding(String),

    #[error("sort mismatch: expected {expected}, found {found}")]
    SortMismatch { expected: String, found: String },

    #[error("unbound variable `{0}`")]
    UnboundVariable(String),

    #[error("solver timeout after {0}ms")]
    Timeout(u64),

    #[error("solver returned unknown: {0}")]
    Unknown(String),

    #[error("invalid model: {0}")]
    InvalidModel(String),

    #[error("unsupported operation: {0}")]
    UnsupportedOp(String),

    #[error("unsat core extraction failed: {0}")]
    UnsatCoreFailed(String),

    #[error("variable limit exceeded: {count} > {max}")]
    VariableLimitExceeded { count: usize, max: usize },
}

// ---------------------------------------------------------------------------
// RepairError
// ---------------------------------------------------------------------------

#[derive(Debug, Error, Clone, Serialize, Deserialize)]
pub enum RepairError {
    #[error("no feasible repair found within constraints")]
    Infeasible,

    #[error("repair constraint violated: {0}")]
    ConstraintViolated(String),

    #[error("parameter `{param}` out of range [{min}, {max}]")]
    OutOfRange {
        param: String,
        min: f64,
        max: f64,
    },

    #[error("repair budget exceeded: cost {cost} > budget {budget}")]
    BudgetExceeded { cost: f64, budget: f64 },

    #[error("conflicting repairs: {0}")]
    Conflict(String),

    #[error("repair validation failed: {0}")]
    ValidationFailed(String),

    #[error("optimization did not converge after {iterations} iterations")]
    NoConvergence { iterations: usize },

    #[error("empty repair plan")]
    EmptyPlan,
}

// ---------------------------------------------------------------------------
// ValidationError
// ---------------------------------------------------------------------------

#[derive(Debug, Error, Clone, Serialize, Deserialize)]
pub enum ValidationError {
    #[error("schema validation failed: {0}")]
    Schema(String),

    #[error("constraint violation: {constraint} — {details}")]
    ConstraintViolation {
        constraint: String,
        details: String,
    },

    #[error("type mismatch: expected {expected}, found {found}")]
    TypeMismatch { expected: String, found: String },

    #[error("range violation: {field} = {value} not in [{min}, {max}]")]
    RangeViolation {
        field: String,
        value: String,
        min: String,
        max: String,
    },

    #[error("missing dependency: {0}")]
    MissingDependency(String),

    #[error("circular dependency: {0}")]
    CircularDependency(String),

    #[error("invariant violated: {0}")]
    InvariantViolated(String),
}

// ---------------------------------------------------------------------------
// ErrorContext – attaches contextual information to any error
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorContext {
    pub message: String,
    pub source_file: Option<String>,
    pub line: Option<usize>,
    pub service: Option<String>,
    pub phase: Option<String>,
}

impl ErrorContext {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            source_file: None,
            line: None,
            service: None,
            phase: None,
        }
    }

    pub fn with_source_file(mut self, file: impl Into<String>) -> Self {
        self.source_file = Some(file.into());
        self
    }

    pub fn with_line(mut self, line: usize) -> Self {
        self.line = Some(line);
        self
    }

    pub fn with_service(mut self, service: impl Into<String>) -> Self {
        self.service = Some(service.into());
        self
    }

    pub fn with_phase(mut self, phase: impl Into<String>) -> Self {
        self.phase = Some(phase.into());
        self
    }
}

impl fmt::Display for ErrorContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)?;
        if let Some(ref file) = self.source_file {
            write!(f, " (file: {file}")?;
            if let Some(line) = self.line {
                write!(f, ":{line}")?;
            }
            write!(f, ")")?;
        }
        if let Some(ref svc) = self.service {
            write!(f, " [service: {svc}]")?;
        }
        if let Some(ref phase) = self.phase {
            write!(f, " [phase: {phase}]")?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ErrorChain – ordered list of contextual error frames
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorChain {
    pub contexts: Vec<ErrorContext>,
}

impl ErrorChain {
    pub fn new() -> Self {
        Self {
            contexts: Vec::new(),
        }
    }

    pub fn push(&mut self, ctx: ErrorContext) {
        self.contexts.push(ctx);
    }

    pub fn with(mut self, ctx: ErrorContext) -> Self {
        self.contexts.push(ctx);
        self
    }

    pub fn is_empty(&self) -> bool {
        self.contexts.is_empty()
    }

    pub fn len(&self) -> usize {
        self.contexts.len()
    }

    pub fn root_cause(&self) -> Option<&ErrorContext> {
        self.contexts.last()
    }

    pub fn top(&self) -> Option<&ErrorContext> {
        self.contexts.first()
    }

    pub fn iter(&self) -> impl Iterator<Item = &ErrorContext> {
        self.contexts.iter()
    }
}

impl Default for ErrorChain {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for ErrorChain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, ctx) in self.contexts.iter().enumerate() {
            if i > 0 {
                write!(f, "\n  caused by: ")?;
            }
            write!(f, "{ctx}")?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Result aliases
// ---------------------------------------------------------------------------

pub type CascadeResult<T> = std::result::Result<T, CascadeError>;
pub type ConfigResult<T> = std::result::Result<T, ConfigError>;
pub type AnalysisResult<T> = std::result::Result<T, AnalysisError>;
pub type SmtResult<T> = std::result::Result<T, SmtError>;
pub type RepairResult<T> = std::result::Result<T, RepairError>;
pub type ValidationResult<T> = std::result::Result<T, ValidationError>;

// ---------------------------------------------------------------------------
// Convenience conversion: serde_json::Error -> CascadeError
// ---------------------------------------------------------------------------

impl From<serde_json::Error> for CascadeError {
    fn from(e: serde_json::Error) -> Self {
        CascadeError::Serialization(e.to_string())
    }
}

impl From<serde_yaml::Error> for CascadeError {
    fn from(e: serde_yaml::Error) -> Self {
        CascadeError::Serialization(e.to_string())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_error_display_missing_field() {
        let e = ConfigError::MissingField {
            field: "retries".into(),
            context: "RetryPolicy".into(),
        };
        assert!(e.to_string().contains("retries"));
        assert!(e.to_string().contains("RetryPolicy"));
    }

    #[test]
    fn test_config_error_display_invalid_value() {
        let e = ConfigError::InvalidValue {
            field: "timeout".into(),
            value: "-5".into(),
            reason: "must be positive".into(),
        };
        let s = e.to_string();
        assert!(s.contains("timeout") && s.contains("-5"));
    }

    #[test]
    fn test_parse_error_yaml() {
        let e = ParseError::Yaml {
            line: 42,
            message: "unexpected mapping".into(),
        };
        assert!(e.to_string().contains("42"));
    }

    #[test]
    fn test_analysis_error_service_not_found() {
        let e = AnalysisError::ServiceNotFound("gateway".into());
        assert!(e.to_string().contains("gateway"));
    }

    #[test]
    fn test_smt_error_sort_mismatch() {
        let e = SmtError::SortMismatch {
            expected: "Bool".into(),
            found: "Int".into(),
        };
        assert!(e.to_string().contains("Bool") && e.to_string().contains("Int"));
    }

    #[test]
    fn test_repair_error_infeasible() {
        let e = RepairError::Infeasible;
        assert!(e.to_string().contains("no feasible"));
    }

    #[test]
    fn test_validation_error_range() {
        let e = ValidationError::RangeViolation {
            field: "retries".into(),
            value: "100".into(),
            min: "0".into(),
            max: "10".into(),
        };
        assert!(e.to_string().contains("retries"));
    }

    #[test]
    fn test_cascade_error_from_config() {
        let ce: CascadeError = ConfigError::NotFound("foo.yaml".into()).into();
        assert!(ce.to_string().contains("foo.yaml"));
    }

    #[test]
    fn test_cascade_error_from_analysis() {
        let ce: CascadeError = AnalysisError::EmptyTopology.into();
        assert!(ce.to_string().contains("empty topology"));
    }

    #[test]
    fn test_error_context_display() {
        let ctx = ErrorContext::new("something broke")
            .with_source_file("service.yaml")
            .with_line(10)
            .with_service("gateway")
            .with_phase("parsing");
        let s = ctx.to_string();
        assert!(s.contains("something broke"));
        assert!(s.contains("service.yaml"));
        assert!(s.contains("10"));
        assert!(s.contains("gateway"));
        assert!(s.contains("parsing"));
    }

    #[test]
    fn test_error_chain_basics() {
        let mut chain = ErrorChain::new();
        assert!(chain.is_empty());
        chain.push(ErrorContext::new("top"));
        chain.push(ErrorContext::new("root"));
        assert_eq!(chain.len(), 2);
        assert_eq!(chain.top().unwrap().message, "top");
        assert_eq!(chain.root_cause().unwrap().message, "root");
    }

    #[test]
    fn test_error_chain_display() {
        let chain = ErrorChain::new()
            .with(ErrorContext::new("outer"))
            .with(ErrorContext::new("inner"));
        let s = chain.to_string();
        assert!(s.contains("outer"));
        assert!(s.contains("caused by"));
        assert!(s.contains("inner"));
    }

    #[test]
    fn test_error_chain_iter() {
        let chain = ErrorChain::new()
            .with(ErrorContext::new("a"))
            .with(ErrorContext::new("b"))
            .with(ErrorContext::new("c"));
        let msgs: Vec<_> = chain.iter().map(|c| c.message.as_str()).collect();
        assert_eq!(msgs, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_cascade_error_from_io() {
        let io = std::io::Error::new(std::io::ErrorKind::NotFound, "gone");
        let ce: CascadeError = io.into();
        assert!(ce.to_string().contains("gone"));
    }

    #[test]
    fn test_result_aliases_compile() {
        fn config_fn() -> ConfigResult<()> {
            Ok(())
        }
        fn analysis_fn() -> AnalysisResult<u32> {
            Ok(42)
        }
        assert!(config_fn().is_ok());
        assert_eq!(analysis_fn().unwrap(), 42);
    }
}
