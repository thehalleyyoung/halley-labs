//! Comprehensive error types for the NLP metamorphic fault localizer.

use crate::types::StageId;
use std::fmt;

/// Convenience alias used throughout the workspace.
pub type Result<T> = std::result::Result<T, LocalizerError>;

/// Top-level error type carrying structured context for every failure mode.
#[derive(Debug, thiserror::Error)]
pub enum LocalizerError {
    /// A pipeline stage failed during execution.
    #[error("pipeline error at stage '{stage}': {message}")]
    PipelineError {
        stage: String,
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// A metamorphic transformation could not be applied.
    #[error("transformation error '{transformation}': {message}")]
    TransformationError {
        transformation: String,
        message: String,
    },

    /// Input failed a grammar / well-formedness check.
    #[error("grammar error: {message}")]
    GrammarError { message: String, position: Option<usize> },

    /// The input-shrinking phase failed to converge.
    #[error("shrinking error after {iterations} iterations: {message}")]
    ShrinkingError { message: String, iterations: u32 },

    /// Invalid or missing configuration.
    #[error("config error for key '{key}': {message}")]
    ConfigError { key: String, message: String },

    /// File-system or other I/O failure.
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// JSON (de)serialization failure.
    #[error("serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    /// A domain-level validation check did not pass.
    #[error("validation error in '{context}': {message}")]
    ValidationError { context: String, message: String },

    /// Calibration failed (e.g. singular matrix, ill-conditioned data).
    #[error("calibration error: {message} (condition_number={condition_number:.4})")]
    CalibrationError { message: String, condition_number: f64 },

    /// Token / span alignment between original and transformed sentence failed.
    #[error("alignment error between '{source_text}' and '{target_text}': {message}")]
    AlignmentError {
        source_text: String,
        target_text: String,
        message: String,
    },

    /// A numeric / matrix operation failed.
    #[error("matrix error: {message} (dimensions={rows}x{cols})")]
    MatrixError {
        message: String,
        rows: usize,
        cols: usize,
    },

    /// The metamorphic oracle detected an unexpected relation violation.
    #[error("oracle error for relation '{relation}': {message}")]
    OracleError { relation: String, message: String },
}

// ── Convenience constructors ────────────────────────────────────────────────

impl LocalizerError {
    pub fn pipeline(stage: impl Into<String>, message: impl Into<String>) -> Self {
        Self::PipelineError {
            stage: stage.into(),
            message: message.into(),
            source: None,
        }
    }

    pub fn pipeline_with_source(
        stage: impl Into<String>,
        message: impl Into<String>,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::PipelineError {
            stage: stage.into(),
            message: message.into(),
            source: Some(Box::new(source)),
        }
    }

    pub fn transformation(name: impl Into<String>, message: impl Into<String>) -> Self {
        Self::TransformationError {
            transformation: name.into(),
            message: message.into(),
        }
    }

    pub fn grammar(message: impl Into<String>) -> Self {
        Self::GrammarError {
            message: message.into(),
            position: None,
        }
    }

    pub fn grammar_at(message: impl Into<String>, position: usize) -> Self {
        Self::GrammarError {
            message: message.into(),
            position: Some(position),
        }
    }

    pub fn shrinking(message: impl Into<String>, iterations: u32) -> Self {
        Self::ShrinkingError {
            message: message.into(),
            iterations,
        }
    }

    pub fn config(key: impl Into<String>, message: impl Into<String>) -> Self {
        Self::ConfigError {
            key: key.into(),
            message: message.into(),
        }
    }

    pub fn validation(context: impl Into<String>, message: impl Into<String>) -> Self {
        Self::ValidationError {
            context: context.into(),
            message: message.into(),
        }
    }

    pub fn calibration(message: impl Into<String>, condition_number: f64) -> Self {
        Self::CalibrationError {
            message: message.into(),
            condition_number,
        }
    }

    pub fn alignment(
        source_text: impl Into<String>,
        target_text: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self::AlignmentError {
            source_text: source_text.into(),
            target_text: target_text.into(),
            message: message.into(),
        }
    }

    pub fn matrix(message: impl Into<String>, rows: usize, cols: usize) -> Self {
        Self::MatrixError {
            message: message.into(),
            rows,
            cols,
        }
    }

    pub fn oracle(relation: impl Into<String>, message: impl Into<String>) -> Self {
        Self::OracleError {
            relation: relation.into(),
            message: message.into(),
        }
    }

    /// True when the error comes from a specific pipeline stage.
    pub fn is_stage_error(&self, stage_name: &str) -> bool {
        match self {
            Self::PipelineError { stage, .. } => stage == stage_name,
            _ => false,
        }
    }

    /// Returns the pipeline stage name if this is a pipeline error.
    pub fn stage_name(&self) -> Option<&str> {
        match self {
            Self::PipelineError { stage, .. } => Some(stage.as_str()),
            _ => None,
        }
    }

    /// True for errors that are transient and may succeed on retry.
    pub fn is_retryable(&self) -> bool {
        matches!(self, Self::IoError(_) | Self::PipelineError { .. })
    }

    /// Broad category name useful for metrics / logging.
    pub fn category(&self) -> &'static str {
        match self {
            Self::PipelineError { .. } => "pipeline",
            Self::TransformationError { .. } => "transformation",
            Self::GrammarError { .. } => "grammar",
            Self::ShrinkingError { .. } => "shrinking",
            Self::ConfigError { .. } => "config",
            Self::IoError(_) => "io",
            Self::SerializationError(_) => "serialization",
            Self::ValidationError { .. } => "validation",
            Self::CalibrationError { .. } => "calibration",
            Self::AlignmentError { .. } => "alignment",
            Self::MatrixError { .. } => "matrix",
            Self::OracleError { .. } => "oracle",
        }
    }
}

// ── From impls for common foreign types ─────────────────────────────────────

impl From<uuid::Error> for LocalizerError {
    fn from(e: uuid::Error) -> Self {
        Self::ValidationError {
            context: "uuid".into(),
            message: e.to_string(),
        }
    }
}

impl From<std::num::ParseFloatError> for LocalizerError {
    fn from(e: std::num::ParseFloatError) -> Self {
        Self::ValidationError {
            context: "parse_float".into(),
            message: e.to_string(),
        }
    }
}

impl From<std::num::ParseIntError> for LocalizerError {
    fn from(e: std::num::ParseIntError) -> Self {
        Self::ValidationError {
            context: "parse_int".into(),
            message: e.to_string(),
        }
    }
}

impl From<chrono::ParseError> for LocalizerError {
    fn from(e: chrono::ParseError) -> Self {
        Self::ValidationError {
            context: "datetime_parse".into(),
            message: e.to_string(),
        }
    }
}

impl From<StageId> for String {
    fn from(id: StageId) -> Self {
        id.0.to_string()
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_error_display() {
        let err = LocalizerError::pipeline("tokenizer", "unexpected EOF");
        assert!(err.to_string().contains("tokenizer"));
        assert!(err.to_string().contains("unexpected EOF"));
    }

    #[test]
    fn test_transformation_error_display() {
        let err = LocalizerError::transformation("passivize", "no transitive verb found");
        assert!(err.to_string().contains("passivize"));
    }

    #[test]
    fn test_grammar_error_with_position() {
        let err = LocalizerError::grammar_at("unmatched paren", 42);
        match &err {
            LocalizerError::GrammarError { position, .. } => assert_eq!(*position, Some(42)),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_shrinking_error() {
        let err = LocalizerError::shrinking("no progress", 100);
        match &err {
            LocalizerError::ShrinkingError { iterations, .. } => assert_eq!(*iterations, 100),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_config_error() {
        let err = LocalizerError::config("timeout", "must be positive");
        assert!(err.to_string().contains("timeout"));
    }

    #[test]
    fn test_io_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file missing");
        let err: LocalizerError = io_err.into();
        assert!(matches!(err, LocalizerError::IoError(_)));
    }

    #[test]
    fn test_is_stage_error() {
        let err = LocalizerError::pipeline("ner", "timeout");
        assert!(err.is_stage_error("ner"));
        assert!(!err.is_stage_error("tokenizer"));
    }

    #[test]
    fn test_is_retryable() {
        let io_err: LocalizerError =
            std::io::Error::new(std::io::ErrorKind::TimedOut, "timeout").into();
        assert!(io_err.is_retryable());

        let cfg_err = LocalizerError::config("k", "bad");
        assert!(!cfg_err.is_retryable());
    }

    #[test]
    fn test_category() {
        assert_eq!(LocalizerError::pipeline("a", "b").category(), "pipeline");
        assert_eq!(
            LocalizerError::calibration("x", 1.0).category(),
            "calibration"
        );
        assert_eq!(LocalizerError::oracle("r", "m").category(), "oracle");
    }

    #[test]
    fn test_matrix_error_display() {
        let err = LocalizerError::matrix("singular", 3, 3);
        let s = err.to_string();
        assert!(s.contains("3x3"));
        assert!(s.contains("singular"));
    }
}
