//! # isospec-format
//!
//! Wire protocol parsers and format converters for IsoSpec trace collection.
//!
//! Supports:
//! - **PostgreSQL wire protocol** message parsing for trace extraction
//! - **MySQL wire protocol** packet parsing for trace extraction
//! - **SQL trace logs** (pgaudit, MySQL general log, custom formats)
//! - **Jepsen history format** (EDN) import/export
//! - **IsoSpec native JSON** format

pub mod edn;
pub mod jepsen;
pub mod mysql_wire;
pub mod pg_wire;
pub mod sql_trace;

use isospec_types::error::IsoSpecError;

/// Errors specific to format parsing
#[derive(Debug, thiserror::Error)]
pub enum FormatError {
    #[error("Parse error at byte {offset}: {message}")]
    ParseError { offset: usize, message: String },

    #[error("Unsupported protocol version: {0}")]
    UnsupportedVersion(String),

    #[error("Incomplete message: expected {expected} bytes, got {actual}")]
    IncompleteMessage { expected: usize, actual: usize },

    #[error("Invalid EDN syntax at position {position}: {detail}")]
    EdnSyntaxError { position: usize, detail: String },

    #[error("Missing required field: {0}")]
    MissingField(String),

    #[error("Invalid trace format: {0}")]
    InvalidFormat(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("IsoSpec error: {0}")]
    IsoSpec(#[from] IsoSpecError),
}

pub type FormatResult<T> = Result<T, FormatError>;
