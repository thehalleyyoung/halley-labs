//! Error types for the analysis framework.

use serde::{Deserialize, Serialize};
use thiserror::Error;
use std::fmt;

/// Main error type for analysis operations.
#[derive(Error, Debug)]
pub enum AnalysisError {
    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Binary parsing error: {0}")]
    BinaryParse(String),

    #[error("Unsupported instruction at {address}: {mnemonic}")]
    UnsupportedInstruction { address: u64, mnemonic: String },

    #[error("Fixpoint computation failed: {0}")]
    FixpointFailure(String),

    #[error("Widening error: {0}")]
    WideningError(String),

    #[error("Domain operation error: {kind}: {message}")]
    DomainError { kind: ErrorKind, message: String },

    #[error("Contract extraction failed: {0}")]
    ContractExtraction(String),

    #[error("Composition error: {0}")]
    Composition(String),

    #[error("SMT solver error: {0}")]
    SmtError(String),

    #[error("Certificate error: {0}")]
    CertificateError(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Timeout after {seconds}s")]
    Timeout { seconds: u64 },

    #[error("Resource limit exceeded: {resource} ({used}/{limit})")]
    ResourceLimit { resource: String, used: u64, limit: u64 },

    #[error("Internal error: {0}")]
    Internal(String),

    #[error("Verification failed: {0}")]
    Verification(String),

    #[error("Not implemented: {0}")]
    NotImplemented(String),
}

impl AnalysisError {
    pub fn domain(kind: ErrorKind, msg: impl Into<String>) -> Self {
        Self::DomainError { kind, message: msg.into() }
    }

    pub fn config(msg: impl Into<String>) -> Self {
        Self::Config(msg.into())
    }

    pub fn internal(msg: impl Into<String>) -> Self {
        Self::Internal(msg.into())
    }

    pub fn is_recoverable(&self) -> bool {
        matches!(self,
            Self::UnsupportedInstruction { .. }
            | Self::WideningError(_)
            | Self::Timeout { .. }
        )
    }

    pub fn severity(&self) -> ErrorSeverity {
        match self {
            Self::Internal(_) | Self::Io(_) => ErrorSeverity::Fatal,
            Self::FixpointFailure(_) | Self::DomainError { .. } => ErrorSeverity::Error,
            Self::UnsupportedInstruction { .. } => ErrorSeverity::Warning,
            Self::Timeout { .. } | Self::ResourceLimit { .. } => ErrorSeverity::Error,
            _ => ErrorSeverity::Error,
        }
    }
}

/// Error classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ErrorKind {
    LatticeViolation,
    MonotonicityViolation,
    ConvergenceFailure,
    InvalidState,
    InvalidTransfer,
    InvalidReduction,
    InvalidWidening,
    InvalidJoin,
    InvalidMeet,
    BoundsExceeded,
    InvariantViolation,
    UnsoundApproximation,
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LatticeViolation => write!(f, "lattice violation"),
            Self::MonotonicityViolation => write!(f, "monotonicity violation"),
            Self::ConvergenceFailure => write!(f, "convergence failure"),
            Self::InvalidState => write!(f, "invalid state"),
            Self::InvalidTransfer => write!(f, "invalid transfer"),
            Self::InvalidReduction => write!(f, "invalid reduction"),
            Self::InvalidWidening => write!(f, "invalid widening"),
            Self::InvalidJoin => write!(f, "invalid join"),
            Self::InvalidMeet => write!(f, "invalid meet"),
            Self::BoundsExceeded => write!(f, "bounds exceeded"),
            Self::InvariantViolation => write!(f, "invariant violation"),
            Self::UnsoundApproximation => write!(f, "unsound approximation"),
        }
    }
}

/// Error severity level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ErrorSeverity {
    Warning,
    Error,
    Fatal,
}

/// Result type alias for the analysis framework.
pub type AnalysisResult<T> = Result<T, AnalysisError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = AnalysisError::config("missing cache config");
        assert!(format!("{}", err).contains("missing cache config"));
    }

    #[test]
    fn test_error_severity() {
        let err = AnalysisError::Internal("bug".into());
        assert_eq!(err.severity(), ErrorSeverity::Fatal);
    }

    #[test]
    fn test_recoverable() {
        let err = AnalysisError::UnsupportedInstruction {
            address: 0x1000,
            mnemonic: "vgatherdps".into(),
        };
        assert!(err.is_recoverable());
        let err2 = AnalysisError::Internal("critical".into());
        assert!(!err2.is_recoverable());
    }
}
