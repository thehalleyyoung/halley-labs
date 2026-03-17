// Oracle error types for the decomposition selection oracle.

use std::fmt;

/// Primary error type for the oracle crate.
#[derive(Debug, thiserror::Error)]
pub enum OracleError {
    #[error("Feature extraction failed: {message}")]
    FeatureExtractionFailed {
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error("Classification failed: {message}")]
    ClassificationFailed {
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error("Futility check failed: {message}")]
    FutilityCheckFailed {
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error("Partition failed: {message}")]
    PartitionFailed {
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error("Model not trained: {message}")]
    ModelNotTrained { message: String },

    #[error("Invalid input: {message}")]
    InvalidInput {
        message: String,
        field: Option<String>,
    },

    #[error("Serialization error: {message}")]
    SerializationError {
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },
}

impl OracleError {
    pub fn feature_extraction(msg: impl Into<String>) -> Self {
        OracleError::FeatureExtractionFailed {
            message: msg.into(),
            source: None,
        }
    }

    pub fn feature_extraction_with_source(
        msg: impl Into<String>,
        src: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        OracleError::FeatureExtractionFailed {
            message: msg.into(),
            source: Some(Box::new(src)),
        }
    }

    pub fn classification(msg: impl Into<String>) -> Self {
        OracleError::ClassificationFailed {
            message: msg.into(),
            source: None,
        }
    }

    pub fn classification_with_source(
        msg: impl Into<String>,
        src: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        OracleError::ClassificationFailed {
            message: msg.into(),
            source: Some(Box::new(src)),
        }
    }

    pub fn futility_check(msg: impl Into<String>) -> Self {
        OracleError::FutilityCheckFailed {
            message: msg.into(),
            source: None,
        }
    }

    pub fn futility_check_with_source(
        msg: impl Into<String>,
        src: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        OracleError::FutilityCheckFailed {
            message: msg.into(),
            source: Some(Box::new(src)),
        }
    }

    pub fn partition(msg: impl Into<String>) -> Self {
        OracleError::PartitionFailed {
            message: msg.into(),
            source: None,
        }
    }

    pub fn partition_with_source(
        msg: impl Into<String>,
        src: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        OracleError::PartitionFailed {
            message: msg.into(),
            source: Some(Box::new(src)),
        }
    }

    pub fn model_not_trained(msg: impl Into<String>) -> Self {
        OracleError::ModelNotTrained {
            message: msg.into(),
        }
    }

    pub fn invalid_input(msg: impl Into<String>) -> Self {
        OracleError::InvalidInput {
            message: msg.into(),
            field: None,
        }
    }

    pub fn invalid_input_field(msg: impl Into<String>, field: impl Into<String>) -> Self {
        OracleError::InvalidInput {
            message: msg.into(),
            field: Some(field.into()),
        }
    }

    pub fn serialization(msg: impl Into<String>) -> Self {
        OracleError::SerializationError {
            message: msg.into(),
            source: None,
        }
    }

    pub fn serialization_with_source(
        msg: impl Into<String>,
        src: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        OracleError::SerializationError {
            message: msg.into(),
            source: Some(Box::new(src)),
        }
    }

    /// Returns a short error code string for categorization.
    pub fn error_code(&self) -> &'static str {
        match self {
            OracleError::FeatureExtractionFailed { .. } => "FEAT_EXTRACT",
            OracleError::ClassificationFailed { .. } => "CLASSIFY",
            OracleError::FutilityCheckFailed { .. } => "FUTILITY",
            OracleError::PartitionFailed { .. } => "PARTITION",
            OracleError::ModelNotTrained { .. } => "NOT_TRAINED",
            OracleError::InvalidInput { .. } => "INVALID_INPUT",
            OracleError::SerializationError { .. } => "SERIALIZATION",
        }
    }

    /// Returns true if the error is recoverable in a pipeline context.
    pub fn is_recoverable(&self) -> bool {
        match self {
            OracleError::FeatureExtractionFailed { .. } => false,
            OracleError::ClassificationFailed { .. } => true,
            OracleError::FutilityCheckFailed { .. } => true,
            OracleError::PartitionFailed { .. } => true,
            OracleError::ModelNotTrained { .. } => false,
            OracleError::InvalidInput { .. } => false,
            OracleError::SerializationError { .. } => true,
        }
    }

    /// Returns the severity level as a string.
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            OracleError::FeatureExtractionFailed { .. } => ErrorSeverity::Critical,
            OracleError::ClassificationFailed { .. } => ErrorSeverity::Error,
            OracleError::FutilityCheckFailed { .. } => ErrorSeverity::Warning,
            OracleError::PartitionFailed { .. } => ErrorSeverity::Error,
            OracleError::ModelNotTrained { .. } => ErrorSeverity::Critical,
            OracleError::InvalidInput { .. } => ErrorSeverity::Error,
            OracleError::SerializationError { .. } => ErrorSeverity::Warning,
        }
    }
}

/// Error severity levels for logging and reporting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ErrorSeverity {
    Warning,
    Error,
    Critical,
}

impl fmt::Display for ErrorSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorSeverity::Warning => write!(f, "WARNING"),
            ErrorSeverity::Error => write!(f, "ERROR"),
            ErrorSeverity::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// Context wrapper that annotates an OracleError with pipeline stage information.
#[derive(Debug)]
pub struct ErrorContext {
    pub error: OracleError,
    pub stage: String,
    pub instance_name: Option<String>,
    pub elapsed_secs: Option<f64>,
}

impl fmt::Display for ErrorContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.stage, self.error)?;
        if let Some(ref name) = self.instance_name {
            write!(f, " (instance: {})", name)?;
        }
        if let Some(elapsed) = self.elapsed_secs {
            write!(f, " (after {:.2}s)", elapsed)?;
        }
        Ok(())
    }
}

impl std::error::Error for ErrorContext {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

impl ErrorContext {
    pub fn new(error: OracleError, stage: impl Into<String>) -> Self {
        Self {
            error,
            stage: stage.into(),
            instance_name: None,
            elapsed_secs: None,
        }
    }

    pub fn with_instance(mut self, name: impl Into<String>) -> Self {
        self.instance_name = Some(name.into());
        self
    }

    pub fn with_elapsed(mut self, secs: f64) -> Self {
        self.elapsed_secs = Some(secs);
        self
    }
}

/// Conversion from serde_json errors.
impl From<serde_json::Error> for OracleError {
    fn from(e: serde_json::Error) -> Self {
        OracleError::SerializationError {
            message: e.to_string(),
            source: Some(Box::new(e)),
        }
    }
}

/// Conversion from std::io::Error.
impl From<std::io::Error> for OracleError {
    fn from(e: std::io::Error) -> Self {
        OracleError::SerializationError {
            message: format!("I/O error: {}", e),
            source: Some(Box::new(e)),
        }
    }
}

pub type OracleResult<T> = Result<T, OracleError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display_feature_extraction() {
        let err = OracleError::feature_extraction("eigenvalue computation diverged");
        assert!(err.to_string().contains("eigenvalue computation diverged"));
    }

    #[test]
    fn test_error_display_classification() {
        let err = OracleError::classification("no trained model available");
        assert!(err.to_string().contains("no trained model available"));
    }

    #[test]
    fn test_error_code() {
        assert_eq!(OracleError::feature_extraction("x").error_code(), "FEAT_EXTRACT");
        assert_eq!(OracleError::classification("x").error_code(), "CLASSIFY");
        assert_eq!(OracleError::futility_check("x").error_code(), "FUTILITY");
        assert_eq!(OracleError::partition("x").error_code(), "PARTITION");
        assert_eq!(OracleError::model_not_trained("x").error_code(), "NOT_TRAINED");
        assert_eq!(OracleError::invalid_input("x").error_code(), "INVALID_INPUT");
        assert_eq!(OracleError::serialization("x").error_code(), "SERIALIZATION");
    }

    #[test]
    fn test_error_recoverability() {
        assert!(!OracleError::feature_extraction("x").is_recoverable());
        assert!(OracleError::classification("x").is_recoverable());
        assert!(OracleError::futility_check("x").is_recoverable());
        assert!(!OracleError::model_not_trained("x").is_recoverable());
        assert!(!OracleError::invalid_input("x").is_recoverable());
    }

    #[test]
    fn test_error_severity() {
        assert_eq!(OracleError::feature_extraction("x").severity(), ErrorSeverity::Critical);
        assert_eq!(OracleError::classification("x").severity(), ErrorSeverity::Error);
        assert_eq!(OracleError::futility_check("x").severity(), ErrorSeverity::Warning);
    }

    #[test]
    fn test_invalid_input_with_field() {
        let err = OracleError::invalid_input_field("negative value", "n_trees");
        match &err {
            OracleError::InvalidInput { field, .. } => {
                assert_eq!(field.as_deref(), Some("n_trees"));
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn test_error_context() {
        let err = OracleError::feature_extraction("failed");
        let ctx = ErrorContext::new(err, "spectral_analysis")
            .with_instance("neos-1234")
            .with_elapsed(12.5);
        let msg = ctx.to_string();
        assert!(msg.contains("spectral_analysis"));
        assert!(msg.contains("neos-1234"));
        assert!(msg.contains("12.50"));
    }

    #[test]
    fn test_from_serde_json_error() {
        let json_err = serde_json::from_str::<i32>("not_json").unwrap_err();
        let oracle_err: OracleError = json_err.into();
        assert_eq!(oracle_err.error_code(), "SERIALIZATION");
    }

    #[test]
    fn test_severity_ordering() {
        assert!(ErrorSeverity::Warning < ErrorSeverity::Error);
        assert!(ErrorSeverity::Error < ErrorSeverity::Critical);
    }

    #[test]
    fn test_severity_display() {
        assert_eq!(ErrorSeverity::Warning.to_string(), "WARNING");
        assert_eq!(ErrorSeverity::Error.to_string(), "ERROR");
        assert_eq!(ErrorSeverity::Critical.to_string(), "CRITICAL");
    }

    #[test]
    fn test_error_with_source() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file missing");
        let err = OracleError::feature_extraction_with_source("could not read", io_err);
        assert!(err.to_string().contains("could not read"));
    }
}
