//! Error types for the fhir-interop crate.

use thiserror::Error;

/// Errors that can occur during FHIR/CDS Hooks/RxNorm processing.
#[derive(Debug, Error)]
pub enum FhirInteropError {
    /// JSON parsing failure.
    #[error("JSON parse error: {0}")]
    JsonParse(#[from] serde_json::Error),

    /// A FHIR resource had an unexpected resourceType.
    #[error("Expected FHIR resourceType '{expected}', got '{got}'")]
    ResourceTypeMismatch {
        expected: String,
        got: String,
    },

    /// A required field was missing from a FHIR resource.
    #[error("Missing required field: {0}")]
    MissingField(String),

    /// An RxNorm concept could not be resolved.
    #[error("RxNorm concept not found: {0}")]
    RxNormNotFound(String),

    /// A code system was not recognized.
    #[error("Unrecognized code system: {0}")]
    UnrecognizedCodeSystem(String),

    /// Generic wrapper for other errors.
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}
