//! CLI interface for the Certified Leakage Contracts framework.
//!
//! Provides command-line tools for analyzing, composing, certifying,
//! and regression-testing leakage contracts.

use thiserror::Error;

/// CLI errors.
#[derive(Debug, Error)]
pub enum CliError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("analysis error: {0}")]
    Analysis(String),
    #[error("configuration error: {0}")]
    Config(String),
}

pub type CliResult<T> = Result<T, CliError>;
