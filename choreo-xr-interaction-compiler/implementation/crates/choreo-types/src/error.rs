//! Error types.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Span {
    pub start: usize,
    pub end: usize,
    pub file: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Severity {
    Error,
    Warning,
    Info,
    Hint,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Diagnostic {
    pub severity: Severity,
    pub message: String,
    pub span: Option<Span>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticBag {
    pub diagnostics: Vec<Diagnostic>,
}

impl DiagnosticBag {
    pub fn new() -> Self {
        Self {
            diagnostics: Vec::new(),
        }
    }
}

impl Default for DiagnosticBag {
    fn default() -> Self {
        Self::new()
    }
}

/// Top-level error type for the Choreo compiler.
#[derive(Debug, thiserror::Error)]
pub enum ChoreoError {
    #[error("Parse error: {0}")]
    Parse(String),
    #[error("Type error: {0}")]
    Type(String),
    #[error("Spatial error: {0}")]
    Spatial(String),
    #[error("Temporal error: {0}")]
    Temporal(String),
    #[error("Verification error: {0}")]
    Verification(String),
    #[error("Code generation error: {0}")]
    CodeGen(String),
    #[error("Internal error: {0}")]
    Internal(String),
}
