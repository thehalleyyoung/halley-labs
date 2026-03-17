//! Error types for the IsoSpec system.
use thiserror::Error;
use crate::identifier::TransactionId;
use crate::isolation::{AnomalyClass, IsolationLevel};

#[derive(Debug, Error)]
pub enum IsoSpecError {
    #[error("Transaction error: {0}")]
    Transaction(#[from] crate::transaction::TransactionError),
    #[error("Schema error: {msg}")]
    Schema { msg: String },
    #[error("Predicate error: {msg}")]
    Predicate { msg: String },
    #[error("SMT solver error: {msg}")]
    SmtSolver { msg: String },
    #[error("SMT encoding error: {msg}")]
    SmtEncoding { msg: String },
    #[error("Engine model error: {engine}: {msg}")]
    EngineModel { engine: String, msg: String },
    #[error("Anomaly detection error: {0}")]
    AnomalyDetection(String),
    #[error("Witness synthesis error: {0}")]
    WitnessSynthesis(String),
    #[error("Configuration error: {0}")]
    Configuration(String),
    #[error("Serialization failure for transaction {txn_id} under {level}")]
    SerializationFailure { txn_id: TransactionId, level: IsolationLevel },
    #[error("Lock conflict: transaction {holder} holds lock needed by {requester}")]
    LockConflict { holder: TransactionId, requester: TransactionId },
    #[error("Deadlock detected: {0:?}")]
    Deadlock(Vec<TransactionId>),
    #[error("Timeout after {seconds}s")]
    Timeout { seconds: u64 },
    #[error("Anomaly {anomaly} detected with {txn_count} transactions")]
    AnomalyFound { anomaly: AnomalyClass, txn_count: usize },
    #[error("Portability violation: {msg}")]
    PortabilityViolation { msg: String },
    #[error("Internal error: {0}")]
    Internal(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

impl IsoSpecError {
    pub fn schema(msg: impl Into<String>) -> Self {
        Self::Schema { msg: msg.into() }
    }
    pub fn predicate(msg: impl Into<String>) -> Self {
        Self::Predicate { msg: msg.into() }
    }
    pub fn smt_solver(msg: impl Into<String>) -> Self {
        Self::SmtSolver { msg: msg.into() }
    }
    pub fn engine_model(engine: impl Into<String>, msg: impl Into<String>) -> Self {
        Self::EngineModel { engine: engine.into(), msg: msg.into() }
    }
    pub fn internal(msg: impl Into<String>) -> Self {
        Self::Internal(msg.into())
    }
    pub fn is_recoverable(&self) -> bool {
        matches!(self, Self::LockConflict { .. } | Self::Timeout { .. } | Self::SerializationFailure { .. })
    }
}

pub type IsoSpecResult<T> = Result<T, IsoSpecError>;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_error_display() {
        let err = IsoSpecError::schema("table not found");
        assert!(format!("{}", err).contains("table not found"));
    }
    #[test]
    fn test_error_recoverable() {
        let err = IsoSpecError::Timeout { seconds: 30 };
        assert!(err.is_recoverable());
        let err2 = IsoSpecError::internal("bug");
        assert!(!err2.is_recoverable());
    }
}
