//! PIT-specific error types for the MutSpec PIT integration crate.
//!
//! Provides structured error variants covering every failure mode encountered
//! when parsing PIT reports, converting mutation descriptors, building kill
//! matrices, or executing the PIT runner.  All variants carry enough context
//! for actionable diagnostics.

use std::fmt;
use std::path::PathBuf;

use thiserror::Error;

// ---------------------------------------------------------------------------
// PitErrorKind — fine-grained classification
// ---------------------------------------------------------------------------

/// Fine-grained classification of PIT integration failures.
///
/// Used as an inner discriminant so callers can programmatically match on
/// error categories without pattern-matching the full [`PitError`] enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PitErrorKind {
    /// An I/O error occurred while reading a PIT report file.
    Io,
    /// The XML document is syntactically malformed.
    XmlSyntax,
    /// A required XML element or attribute is missing.
    MissingElement,
    /// An XML attribute value could not be parsed to the expected type.
    InvalidAttribute,
    /// A CSV record is malformed or has the wrong number of columns.
    CsvFormat,
    /// A PIT mutator class name could not be mapped to a MutSpec operator.
    UnknownMutator,
    /// A PIT mutation status string is not recognized.
    UnknownStatus,
    /// A method descriptor (JVM signature) could not be parsed.
    InvalidDescriptor,
    /// Line number is missing or out of range for the source file.
    InvalidLineNumber,
    /// The kill matrix has an inconsistency (e.g., duplicate mutant ids).
    KillMatrixInconsistency,
    /// A source replay operation failed.
    ReplayFailure,
    /// A configuration value is invalid or missing.
    ConfigError,
    /// The PIT runner process returned a non-zero exit code.
    RunnerFailure,
    /// A timeout expired while waiting for PIT to finish.
    Timeout,
}

impl fmt::Display for PitErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            PitErrorKind::Io => "I/O",
            PitErrorKind::XmlSyntax => "XML syntax",
            PitErrorKind::MissingElement => "missing element",
            PitErrorKind::InvalidAttribute => "invalid attribute",
            PitErrorKind::CsvFormat => "CSV format",
            PitErrorKind::UnknownMutator => "unknown mutator",
            PitErrorKind::UnknownStatus => "unknown status",
            PitErrorKind::InvalidDescriptor => "invalid descriptor",
            PitErrorKind::InvalidLineNumber => "invalid line number",
            PitErrorKind::KillMatrixInconsistency => "kill-matrix inconsistency",
            PitErrorKind::ReplayFailure => "replay failure",
            PitErrorKind::ConfigError => "config error",
            PitErrorKind::RunnerFailure => "runner failure",
            PitErrorKind::Timeout => "timeout",
        };
        write!(f, "{label}")
    }
}

// ---------------------------------------------------------------------------
// PitError
// ---------------------------------------------------------------------------

/// Top-level error type for the PIT integration crate.
///
/// Each variant corresponds to a distinct failure scenario and carries
/// structured context (file paths, element names, line numbers, etc.) that
/// aids debugging without requiring the caller to parse free-form strings.
///
/// # Converting to `MutSpecError`
///
/// A blanket `From<PitError>` conversion into
/// [`shared_types::MutSpecError`] is provided via the [`Into`] trait so
/// that PIT errors compose naturally with the rest of the MutSpec pipeline.
#[derive(Debug, Error)]
pub enum PitError {
    /// I/O failure (e.g. file not found, permission denied).
    #[error("PIT I/O error reading `{path}`: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// The XML document is not well-formed.
    #[error("PIT XML syntax error in `{path}`: {message}")]
    XmlSyntax { path: PathBuf, message: String },

    /// A required element or attribute is absent in the XML report.
    #[error("PIT missing element `{element}` in `{path}` (line ~{line})")]
    MissingElement {
        path: PathBuf,
        element: String,
        line: usize,
    },

    /// An attribute value could not be converted to the expected type.
    #[error("PIT invalid attribute `{attribute}={value}` in `{path}`: {reason}")]
    InvalidAttribute {
        path: PathBuf,
        attribute: String,
        value: String,
        reason: String,
    },

    /// A CSV record is malformed.
    #[error("PIT CSV format error in `{path}` at record {record}: {message}")]
    CsvFormat {
        path: PathBuf,
        record: usize,
        message: String,
    },

    /// A PIT mutator class name has no MutSpec equivalent.
    #[error("PIT unknown mutator `{mutator_class}` in `{path}`")]
    UnknownMutator {
        path: PathBuf,
        mutator_class: String,
    },

    /// A PIT mutation detection status string is unrecognized.
    #[error("PIT unknown status `{status}` in `{path}`")]
    UnknownStatus { path: PathBuf, status: String },

    /// A JVM method descriptor could not be parsed.
    #[error("PIT invalid method descriptor `{descriptor}`: {reason}")]
    InvalidDescriptor { descriptor: String, reason: String },

    /// A line-number value is out of range or missing.
    #[error("PIT invalid line number {line} for `{file}`: {reason}")]
    InvalidLineNumber {
        file: String,
        line: i64,
        reason: String,
    },

    /// The kill matrix contains an inconsistency.
    #[error("PIT kill-matrix inconsistency: {message}")]
    KillMatrixInconsistency { message: String },

    /// Source-level replay could not reconstruct the mutation.
    #[error("PIT replay failure for mutant `{mutant_id}`: {message}")]
    ReplayFailure { mutant_id: String, message: String },

    /// A configuration value is invalid.
    #[error("PIT config error: {message}")]
    ConfigError { message: String },

    /// The PIT runner exited with a non-zero code.
    #[error("PIT runner failed (exit {exit_code}): {stderr}")]
    RunnerFailure { exit_code: i32, stderr: String },

    /// Waiting for PIT timed out.
    #[error("PIT timed out after {seconds}s")]
    Timeout { seconds: u64 },
}

impl PitError {
    /// Returns the [`PitErrorKind`] discriminant for this error.
    pub fn kind(&self) -> PitErrorKind {
        match self {
            PitError::Io { .. } => PitErrorKind::Io,
            PitError::XmlSyntax { .. } => PitErrorKind::XmlSyntax,
            PitError::MissingElement { .. } => PitErrorKind::MissingElement,
            PitError::InvalidAttribute { .. } => PitErrorKind::InvalidAttribute,
            PitError::CsvFormat { .. } => PitErrorKind::CsvFormat,
            PitError::UnknownMutator { .. } => PitErrorKind::UnknownMutator,
            PitError::UnknownStatus { .. } => PitErrorKind::UnknownStatus,
            PitError::InvalidDescriptor { .. } => PitErrorKind::InvalidDescriptor,
            PitError::InvalidLineNumber { .. } => PitErrorKind::InvalidLineNumber,
            PitError::KillMatrixInconsistency { .. } => PitErrorKind::KillMatrixInconsistency,
            PitError::ReplayFailure { .. } => PitErrorKind::ReplayFailure,
            PitError::ConfigError { .. } => PitErrorKind::ConfigError,
            PitError::RunnerFailure { .. } => PitErrorKind::RunnerFailure,
            PitError::Timeout { .. } => PitErrorKind::Timeout,
        }
    }

    /// Returns `true` when the error is recoverable (e.g. a single unknown
    /// mutator does not invalidate the whole report).
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self.kind(),
            PitErrorKind::UnknownMutator
                | PitErrorKind::UnknownStatus
                | PitErrorKind::InvalidLineNumber
        )
    }
}

// ---------------------------------------------------------------------------
// Conversion helpers
// ---------------------------------------------------------------------------

/// Convenience alias used throughout this crate.
pub type PitResult<T> = std::result::Result<T, PitError>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::io;

    #[test]
    fn test_error_kind_display() {
        assert_eq!(PitErrorKind::Io.to_string(), "I/O");
        assert_eq!(PitErrorKind::XmlSyntax.to_string(), "XML syntax");
        assert_eq!(PitErrorKind::CsvFormat.to_string(), "CSV format");
        assert_eq!(PitErrorKind::Timeout.to_string(), "timeout");
    }

    #[test]
    fn test_io_error() {
        let err = PitError::Io {
            path: PathBuf::from("/tmp/mutations.xml"),
            source: io::Error::new(io::ErrorKind::NotFound, "not found"),
        };
        assert_eq!(err.kind(), PitErrorKind::Io);
        assert!(!err.is_recoverable());
        let msg = err.to_string();
        assert!(msg.contains("mutations.xml"));
    }

    #[test]
    fn test_unknown_mutator_is_recoverable() {
        let err = PitError::UnknownMutator {
            path: PathBuf::from("report.xml"),
            mutator_class: "com.example.FooMutator".into(),
        };
        assert!(err.is_recoverable());
        assert_eq!(err.kind(), PitErrorKind::UnknownMutator);
    }

    #[test]
    fn test_xml_syntax_error() {
        let err = PitError::XmlSyntax {
            path: PathBuf::from("bad.xml"),
            message: "unexpected EOF".into(),
        };
        assert!(!err.is_recoverable());
        assert!(err.to_string().contains("unexpected EOF"));
    }

    #[test]
    fn test_runner_failure() {
        let err = PitError::RunnerFailure {
            exit_code: 1,
            stderr: "OOM".into(),
        };
        assert_eq!(err.kind(), PitErrorKind::RunnerFailure);
        assert!(err.to_string().contains("exit 1"));
    }

    #[test]
    fn test_timeout_error() {
        let err = PitError::Timeout { seconds: 300 };
        assert_eq!(err.kind(), PitErrorKind::Timeout);
        assert!(err.to_string().contains("300s"));
    }

    #[test]
    fn test_csv_format_error() {
        let err = PitError::CsvFormat {
            path: PathBuf::from("kill.csv"),
            record: 42,
            message: "wrong column count".into(),
        };
        assert_eq!(err.kind(), PitErrorKind::CsvFormat);
        assert!(err.to_string().contains("record 42"));
    }

    #[test]
    fn test_missing_element_error() {
        let err = PitError::MissingElement {
            path: PathBuf::from("r.xml"),
            element: "mutatedClass".into(),
            line: 15,
        };
        assert!(err.to_string().contains("mutatedClass"));
        assert!(err.to_string().contains("line ~15"));
    }

    #[test]
    fn test_replay_failure() {
        let err = PitError::ReplayFailure {
            mutant_id: "abc123".into(),
            message: "line not found".into(),
        };
        assert!(err.to_string().contains("abc123"));
    }
}
