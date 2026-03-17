use std::fmt;

use serde::{Deserialize, Serialize};

/// All error categories produced by the RegSynth pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RegSynthError {
    Parse(ParseError),
    TypeCheck(TypeCheckError),
    Encoding(EncodingError),
    Solving(SolvingError),
    Certificate(CertificateError),
    Temporal(TemporalError),
    Constraint(ConstraintError),
    Config(ConfigError),
    Io(IoError),
    Internal(InternalError),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParseError {
    pub message: String,
    pub source_location: Option<SourceLocation>,
    pub hint: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceLocation {
    pub file: String,
    pub line: usize,
    pub column: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeCheckError {
    pub message: String,
    pub expected: String,
    pub found: String,
    pub context: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodingError {
    pub message: String,
    pub variable: Option<String>,
    pub clause_index: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolvingError {
    pub message: String,
    pub solver_status: Option<String>,
    pub iterations: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertificateError {
    pub message: String,
    pub certificate_id: Option<String>,
    pub verification_step: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalError {
    pub message: String,
    pub interval_desc: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintError {
    pub message: String,
    pub constraint_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigError {
    pub message: String,
    pub field: Option<String>,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IoError {
    pub message: String,
    pub path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InternalError {
    pub message: String,
    pub backtrace: Option<String>,
}

/// Contextual wrapper that accumulates error provenance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorContext {
    pub error: RegSynthError,
    pub chain: Vec<String>,
}

impl ErrorContext {
    pub fn new(error: RegSynthError) -> Self {
        Self {
            error,
            chain: Vec::new(),
        }
    }

    pub fn with_context(mut self, ctx: impl Into<String>) -> Self {
        self.chain.push(ctx.into());
        self
    }

    pub fn root_message(&self) -> &str {
        match &self.error {
            RegSynthError::Parse(e) => &e.message,
            RegSynthError::TypeCheck(e) => &e.message,
            RegSynthError::Encoding(e) => &e.message,
            RegSynthError::Solving(e) => &e.message,
            RegSynthError::Certificate(e) => &e.message,
            RegSynthError::Temporal(e) => &e.message,
            RegSynthError::Constraint(e) => &e.message,
            RegSynthError::Config(e) => &e.message,
            RegSynthError::Io(e) => &e.message,
            RegSynthError::Internal(e) => &e.message,
        }
    }

    pub fn full_chain(&self) -> String {
        let mut parts = vec![self.root_message().to_string()];
        for ctx in &self.chain {
            parts.push(format!("  caused by: {}", ctx));
        }
        parts.join("\n")
    }
}

impl fmt::Display for RegSynthError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse(e) => write!(f, "Parse error: {}", e.message),
            Self::TypeCheck(e) => {
                write!(f, "Type error: {} (expected {}, found {})", e.message, e.expected, e.found)
            }
            Self::Encoding(e) => write!(f, "Encoding error: {}", e.message),
            Self::Solving(e) => write!(f, "Solving error: {}", e.message),
            Self::Certificate(e) => write!(f, "Certificate error: {}", e.message),
            Self::Temporal(e) => write!(f, "Temporal error: {}", e.message),
            Self::Constraint(e) => write!(f, "Constraint error: {}", e.message),
            Self::Config(e) => write!(f, "Config error: {}", e.message),
            Self::Io(e) => write!(f, "IO error: {}", e.message),
            Self::Internal(e) => write!(f, "Internal error: {}", e.message),
        }
    }
}

impl std::error::Error for RegSynthError {}

impl fmt::Display for ErrorContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.full_chain())
    }
}

impl fmt::Display for SourceLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}:{}", self.file, self.line, self.column)
    }
}

// Convenience constructors
impl RegSynthError {
    pub fn parse(msg: impl Into<String>) -> Self {
        Self::Parse(ParseError {
            message: msg.into(),
            source_location: None,
            hint: None,
        })
    }

    pub fn type_check(msg: impl Into<String>, expected: impl Into<String>, found: impl Into<String>) -> Self {
        Self::TypeCheck(TypeCheckError {
            message: msg.into(),
            expected: expected.into(),
            found: found.into(),
            context: Vec::new(),
        })
    }

    pub fn encoding(msg: impl Into<String>) -> Self {
        Self::Encoding(EncodingError {
            message: msg.into(),
            variable: None,
            clause_index: None,
        })
    }

    pub fn solving(msg: impl Into<String>) -> Self {
        Self::Solving(SolvingError {
            message: msg.into(),
            solver_status: None,
            iterations: None,
        })
    }

    pub fn certificate(msg: impl Into<String>) -> Self {
        Self::Certificate(CertificateError {
            message: msg.into(),
            certificate_id: None,
            verification_step: None,
        })
    }

    pub fn temporal(msg: impl Into<String>) -> Self {
        Self::Temporal(TemporalError {
            message: msg.into(),
            interval_desc: None,
        })
    }

    pub fn constraint(msg: impl Into<String>) -> Self {
        Self::Constraint(ConstraintError {
            message: msg.into(),
            constraint_id: None,
        })
    }

    pub fn config(msg: impl Into<String>) -> Self {
        Self::Config(ConfigError {
            message: msg.into(),
            field: None,
            reason: None,
        })
    }

    pub fn io(msg: impl Into<String>) -> Self {
        Self::Io(IoError {
            message: msg.into(),
            path: None,
        })
    }

    pub fn internal(msg: impl Into<String>) -> Self {
        Self::Internal(InternalError {
            message: msg.into(),
            backtrace: None,
        })
    }
}

pub type RegSynthResult<T> = Result<T, RegSynthError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = RegSynthError::parse("unexpected token 'BLAH'");
        assert!(err.to_string().contains("unexpected token"));
    }

    #[test]
    fn test_type_check_error() {
        let err = RegSynthError::type_check("mismatch", "Obligation", "Strategy");
        assert!(err.to_string().contains("expected Obligation"));
    }

    #[test]
    fn test_error_context_chain() {
        let ctx = ErrorContext::new(RegSynthError::solving("infeasible"))
            .with_context("while processing jurisdiction EU-AI-Act")
            .with_context("in pipeline stage 3");
        let full = ctx.full_chain();
        assert!(full.contains("infeasible"));
        assert!(full.contains("jurisdiction EU-AI-Act"));
        assert!(full.contains("pipeline stage 3"));
    }

    #[test]
    fn test_error_serialization() {
        let err = RegSynthError::encoding("variable overflow");
        let json = serde_json::to_string(&err).unwrap();
        let deser: RegSynthError = serde_json::from_str(&json).unwrap();
        assert!(deser.to_string().contains("variable overflow"));
    }

    #[test]
    fn test_result_type() {
        let ok: RegSynthResult<i32> = Ok(42);
        assert_eq!(ok.unwrap(), 42);
        let err: RegSynthResult<i32> = Err(RegSynthError::internal("oops"));
        assert!(err.is_err());
    }
}
