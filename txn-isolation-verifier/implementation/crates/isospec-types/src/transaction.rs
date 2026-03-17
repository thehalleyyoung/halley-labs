//! Transaction types and lifecycle modeling.
use serde::{Deserialize, Serialize};
use crate::identifier::{TransactionId, OperationId};
use crate::isolation::IsolationLevel;
use crate::operation::Operation;
use std::collections::BTreeMap;
use std::fmt;

/// The status of a transaction in its lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TransactionStatus {
    Active,
    Preparing,
    Committed,
    Aborted,
    WaitingForLock,
}

impl TransactionStatus {
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Committed | Self::Aborted)
    }

    pub fn is_active(self) -> bool {
        matches!(self, Self::Active | Self::Preparing | Self::WaitingForLock)
    }

    pub fn can_transition_to(self, target: TransactionStatus) -> bool {
        match (self, target) {
            (Self::Active, Self::Preparing) => true,
            (Self::Active, Self::Committed) => true,
            (Self::Active, Self::Aborted) => true,
            (Self::Active, Self::WaitingForLock) => true,
            (Self::Preparing, Self::Committed) => true,
            (Self::Preparing, Self::Aborted) => true,
            (Self::WaitingForLock, Self::Active) => true,
            (Self::WaitingForLock, Self::Aborted) => true,
            _ => false,
        }
    }
}

impl fmt::Display for TransactionStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Active => write!(f, "ACTIVE"),
            Self::Preparing => write!(f, "PREPARING"),
            Self::Committed => write!(f, "COMMITTED"),
            Self::Aborted => write!(f, "ABORTED"),
            Self::WaitingForLock => write!(f, "WAITING"),
        }
    }
}

/// A transaction in the system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transaction {
    pub id: TransactionId,
    pub isolation_level: IsolationLevel,
    pub status: TransactionStatus,
    pub operations: Vec<Operation>,
    pub read_set: Vec<OperationId>,
    pub write_set: Vec<OperationId>,
    pub start_timestamp: u64,
    pub commit_timestamp: Option<u64>,
    pub metadata: TransactionMetadata,
}

/// Additional metadata about a transaction.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TransactionMetadata {
    pub label: Option<String>,
    pub read_only: bool,
    pub deferrable: bool,
    pub retry_count: u32,
    pub session_id: Option<String>,
    pub application_name: Option<String>,
}

impl Transaction {
    pub fn new(id: TransactionId, isolation_level: IsolationLevel) -> Self {
        Self {
            id,
            isolation_level,
            status: TransactionStatus::Active,
            operations: Vec::new(),
            read_set: Vec::new(),
            write_set: Vec::new(),
            start_timestamp: 0,
            commit_timestamp: None,
            metadata: TransactionMetadata::default(),
        }
    }

    pub fn with_timestamp(mut self, ts: u64) -> Self {
        self.start_timestamp = ts;
        self
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.metadata.label = Some(label.into());
        self
    }

    pub fn with_read_only(mut self, read_only: bool) -> Self {
        self.metadata.read_only = read_only;
        self
    }

    pub fn add_operation(&mut self, op: Operation) {
        let op_id = OperationId::new(self.operations.len() as u64);
        if op.is_read() {
            self.read_set.push(op_id);
        }
        if op.is_write() {
            self.write_set.push(op_id);
        }
        self.operations.push(op);
    }

    pub fn commit(&mut self, timestamp: u64) -> Result<(), TransactionError> {
        if !self.status.can_transition_to(TransactionStatus::Committed) {
            return Err(TransactionError::InvalidTransition {
                from: self.status,
                to: TransactionStatus::Committed,
            });
        }
        self.status = TransactionStatus::Committed;
        self.commit_timestamp = Some(timestamp);
        Ok(())
    }

    pub fn abort(&mut self) -> Result<(), TransactionError> {
        if self.status.is_terminal() {
            return Err(TransactionError::InvalidTransition {
                from: self.status,
                to: TransactionStatus::Aborted,
            });
        }
        self.status = TransactionStatus::Aborted;
        Ok(())
    }

    pub fn is_committed(&self) -> bool {
        self.status == TransactionStatus::Committed
    }

    pub fn is_aborted(&self) -> bool {
        self.status == TransactionStatus::Aborted
    }

    pub fn is_active(&self) -> bool {
        self.status.is_active()
    }

    pub fn is_read_only(&self) -> bool {
        self.metadata.read_only || self.write_set.is_empty()
    }

    pub fn operation_count(&self) -> usize {
        self.operations.len()
    }

    pub fn read_count(&self) -> usize {
        self.read_set.len()
    }

    pub fn write_count(&self) -> usize {
        self.write_set.len()
    }

    pub fn overlaps_with(&self, other: &Transaction) -> bool {
        if let (Some(my_commit), Some(their_commit)) = (self.commit_timestamp, other.commit_timestamp) {
            self.start_timestamp < their_commit && other.start_timestamp < my_commit
        } else {
            true
        }
    }

    pub fn get_operation(&self, idx: usize) -> Option<&Operation> {
        self.operations.get(idx)
    }
}

/// Transaction-level errors.
#[derive(Debug, Clone, thiserror::Error)]
pub enum TransactionError {
    #[error("Invalid state transition from {from} to {to}")]
    InvalidTransition {
        from: TransactionStatus,
        to: TransactionStatus,
    },
    #[error("Transaction {0} is already terminal")]
    AlreadyTerminal(TransactionId),
    #[error("Operation not allowed in state {0}")]
    OperationNotAllowed(TransactionStatus),
    #[error("Serialization failure for transaction {0}")]
    SerializationFailure(TransactionId),
    #[error("Deadlock detected involving transaction {0}")]
    DeadlockDetected(TransactionId),
    #[error("Lock timeout for transaction {0}")]
    LockTimeout(TransactionId),
}

/// A transaction template for workload generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionTemplate {
    pub label: String,
    pub isolation_level: IsolationLevel,
    pub operation_templates: Vec<OperationTemplate>,
    pub read_only: bool,
}

/// An operation template within a transaction template.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationTemplate {
    pub kind: OperationKind,
    pub table: String,
    pub columns: Vec<String>,
    pub predicate_template: Option<String>,
}

/// The kind of operation in a template.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OperationKind {
    Read,
    Write,
    Insert,
    Delete,
    PredicateRead,
    PredicateWrite,
}

/// A set of concurrent transactions for analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionSet {
    pub transactions: BTreeMap<TransactionId, Transaction>,
    pub ordering: Vec<TransactionId>,
}

impl TransactionSet {
    pub fn new() -> Self {
        Self {
            transactions: BTreeMap::new(),
            ordering: Vec::new(),
        }
    }

    pub fn add(&mut self, txn: Transaction) {
        let id = txn.id;
        self.ordering.push(id);
        self.transactions.insert(id, txn);
    }

    pub fn get(&self, id: TransactionId) -> Option<&Transaction> {
        self.transactions.get(&id)
    }

    pub fn get_mut(&mut self, id: TransactionId) -> Option<&mut Transaction> {
        self.transactions.get_mut(&id)
    }

    pub fn len(&self) -> usize {
        self.transactions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.transactions.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&TransactionId, &Transaction)> {
        self.transactions.iter()
    }

    pub fn active_transactions(&self) -> Vec<TransactionId> {
        self.transactions
            .iter()
            .filter(|(_, t)| t.is_active())
            .map(|(id, _)| *id)
            .collect()
    }

    pub fn committed_transactions(&self) -> Vec<TransactionId> {
        self.transactions
            .iter()
            .filter(|(_, t)| t.is_committed())
            .map(|(id, _)| *id)
            .collect()
    }
}

impl Default for TransactionSet {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::isolation::IsolationLevel;

    #[test]
    fn test_transaction_creation() {
        let txn = Transaction::new(TransactionId::new(1), IsolationLevel::Serializable);
        assert!(txn.is_active());
        assert!(!txn.is_committed());
        assert!(!txn.is_aborted());
        assert_eq!(txn.operation_count(), 0);
    }

    #[test]
    fn test_transaction_commit() {
        let mut txn = Transaction::new(TransactionId::new(1), IsolationLevel::ReadCommitted);
        assert!(txn.commit(100).is_ok());
        assert!(txn.is_committed());
        assert_eq!(txn.commit_timestamp, Some(100));
    }

    #[test]
    fn test_transaction_abort() {
        let mut txn = Transaction::new(TransactionId::new(1), IsolationLevel::RepeatableRead);
        assert!(txn.abort().is_ok());
        assert!(txn.is_aborted());
    }

    #[test]
    fn test_invalid_transition() {
        let mut txn = Transaction::new(TransactionId::new(1), IsolationLevel::Serializable);
        txn.commit(100).unwrap();
        assert!(txn.abort().is_err());
    }

    #[test]
    fn test_transaction_overlap() {
        let mut t1 = Transaction::new(TransactionId::new(1), IsolationLevel::Serializable)
            .with_timestamp(10);
        t1.commit(20).unwrap();
        let mut t2 = Transaction::new(TransactionId::new(2), IsolationLevel::Serializable)
            .with_timestamp(15);
        t2.commit(25).unwrap();
        assert!(t1.overlaps_with(&t2));

        let mut t3 = Transaction::new(TransactionId::new(3), IsolationLevel::Serializable)
            .with_timestamp(25);
        t3.commit(30).unwrap();
        assert!(!t1.overlaps_with(&t3));
    }

    #[test]
    fn test_transaction_set() {
        let mut set = TransactionSet::new();
        set.add(Transaction::new(TransactionId::new(1), IsolationLevel::Serializable));
        set.add(Transaction::new(TransactionId::new(2), IsolationLevel::ReadCommitted));
        assert_eq!(set.len(), 2);
        assert_eq!(set.active_transactions().len(), 2);
    }

    #[test]
    fn test_status_display() {
        assert_eq!(format!("{}", TransactionStatus::Active), "ACTIVE");
        assert_eq!(format!("{}", TransactionStatus::Committed), "COMMITTED");
        assert_eq!(format!("{}", TransactionStatus::Aborted), "ABORTED");
    }

    #[test]
    fn test_read_only_detection() {
        let txn = Transaction::new(TransactionId::new(1), IsolationLevel::Serializable)
            .with_read_only(true);
        assert!(txn.is_read_only());
    }
}
