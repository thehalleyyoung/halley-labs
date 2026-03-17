//! Operation types for transaction modeling.
use serde::{Deserialize, Serialize};
use crate::identifier::{TransactionId, ItemId, TableId, OperationId};
use crate::predicate::Predicate;
use crate::value::Value;
use std::fmt;

/// A single operation within a transaction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Operation {
    pub id: OperationId,
    pub txn_id: TransactionId,
    pub kind: OpKind,
    pub timestamp: u64,
}

/// The specific kind of operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OpKind {
    /// Read a single item by key.
    Read(ReadOp),
    /// Write (update) a single item.
    Write(WriteOp),
    /// Insert a new tuple.
    Insert(InsertOp),
    /// Delete tuple(s) matching predicate.
    Delete(DeleteOp),
    /// Predicate read (range scan).
    PredicateRead(PredicateReadOp),
    /// Predicate write (range update).
    PredicateWrite(PredicateWriteOp),
    /// Acquire a lock explicitly.
    Lock(LockOp),
    /// Begin transaction.
    Begin(BeginOp),
    /// Commit transaction.
    Commit(CommitOp),
    /// Abort transaction.
    Abort(AbortOp),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadOp {
    pub table: TableId,
    pub item: ItemId,
    pub columns: Vec<String>,
    pub value_read: Option<Value>,
    pub version_read: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WriteOp {
    pub table: TableId,
    pub item: ItemId,
    pub columns: Vec<String>,
    pub old_value: Option<Value>,
    pub new_value: Value,
    pub version_written: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InsertOp {
    pub table: TableId,
    pub item: ItemId,
    pub values: indexmap::IndexMap<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeleteOp {
    pub table: TableId,
    pub predicate: Predicate,
    pub deleted_items: Vec<ItemId>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredicateReadOp {
    pub table: TableId,
    pub predicate: Predicate,
    pub items_read: Vec<ItemId>,
    pub result_count: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredicateWriteOp {
    pub table: TableId,
    pub predicate: Predicate,
    pub columns: Vec<String>,
    pub new_value: Value,
    pub items_written: Vec<ItemId>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LockOp {
    pub table: TableId,
    pub item: Option<ItemId>,
    pub predicate: Option<Predicate>,
    pub mode: LockMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LockMode {
    Shared,
    Exclusive,
    IntentShared,
    IntentExclusive,
    Update,
    SIRead,
    /// SQL Server key-range lock modes
    RangeSharedShared,
    RangeSharedUpdate,
    RangeInsertNull,
    RangeExclusiveExclusive,
}

impl LockMode {
    pub fn is_compatible_with(self, other: LockMode) -> bool {
        use LockMode::*;
        matches!(
            (self, other),
            (Shared, Shared) | (Shared, IntentShared) | (IntentShared, Shared)
            | (IntentShared, IntentShared) | (IntentShared, IntentExclusive)
            | (IntentExclusive, IntentShared) | (IntentExclusive, IntentExclusive)
            | (SIRead, Shared) | (Shared, SIRead) | (SIRead, SIRead)
            | (SIRead, IntentShared) | (IntentShared, SIRead)
        )
    }

    pub fn strength(self) -> u8 {
        match self {
            Self::SIRead => 0,
            Self::IntentShared => 1,
            Self::Shared => 2,
            Self::Update => 3,
            Self::IntentExclusive => 4,
            Self::RangeSharedShared => 5,
            Self::RangeSharedUpdate => 6,
            Self::RangeInsertNull => 7,
            Self::RangeExclusiveExclusive => 8,
            Self::Exclusive => 9,
        }
    }
}

impl fmt::Display for LockMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shared => write!(f, "S"),
            Self::Exclusive => write!(f, "X"),
            Self::IntentShared => write!(f, "IS"),
            Self::IntentExclusive => write!(f, "IX"),
            Self::Update => write!(f, "U"),
            Self::SIRead => write!(f, "SIREAD"),
            Self::RangeSharedShared => write!(f, "RangeS-S"),
            Self::RangeSharedUpdate => write!(f, "RangeS-U"),
            Self::RangeInsertNull => write!(f, "RangeI-N"),
            Self::RangeExclusiveExclusive => write!(f, "RangeX-X"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeginOp {
    pub isolation_level: crate::isolation::IsolationLevel,
    pub read_only: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitOp {
    pub commit_timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbortOp {
    pub reason: Option<String>,
}

impl Operation {
    pub fn read(
        id: OperationId,
        txn_id: TransactionId,
        table: TableId,
        item: ItemId,
        timestamp: u64,
    ) -> Self {
        Self {
            id,
            txn_id,
            kind: OpKind::Read(ReadOp {
                table,
                item,
                columns: Vec::new(),
                value_read: None,
                version_read: None,
            }),
            timestamp,
        }
    }

    pub fn write(
        id: OperationId,
        txn_id: TransactionId,
        table: TableId,
        item: ItemId,
        value: Value,
        timestamp: u64,
    ) -> Self {
        Self {
            id,
            txn_id,
            kind: OpKind::Write(WriteOp {
                table,
                item,
                columns: Vec::new(),
                old_value: None,
                new_value: value,
                version_written: None,
            }),
            timestamp,
        }
    }

    pub fn insert(
        id: OperationId,
        txn_id: TransactionId,
        table: TableId,
        item: ItemId,
        values: indexmap::IndexMap<String, Value>,
        timestamp: u64,
    ) -> Self {
        Self {
            id,
            txn_id,
            kind: OpKind::Insert(InsertOp { table, item, values }),
            timestamp,
        }
    }

    pub fn predicate_read(
        id: OperationId,
        txn_id: TransactionId,
        table: TableId,
        predicate: Predicate,
        timestamp: u64,
    ) -> Self {
        Self {
            id,
            txn_id,
            kind: OpKind::PredicateRead(PredicateReadOp {
                table,
                predicate,
                items_read: Vec::new(),
                result_count: None,
            }),
            timestamp,
        }
    }

    pub fn begin(
        id: OperationId,
        txn_id: TransactionId,
        level: crate::isolation::IsolationLevel,
        timestamp: u64,
    ) -> Self {
        Self {
            id,
            txn_id,
            kind: OpKind::Begin(BeginOp {
                isolation_level: level,
                read_only: false,
            }),
            timestamp,
        }
    }

    pub fn commit(id: OperationId, txn_id: TransactionId, timestamp: u64) -> Self {
        Self {
            id,
            txn_id,
            kind: OpKind::Commit(CommitOp {
                commit_timestamp: timestamp,
            }),
            timestamp,
        }
    }

    pub fn abort(id: OperationId, txn_id: TransactionId, timestamp: u64, reason: Option<String>) -> Self {
        Self {
            id,
            txn_id,
            kind: OpKind::Abort(AbortOp { reason }),
            timestamp,
        }
    }

    pub fn is_read(&self) -> bool {
        matches!(self.kind, OpKind::Read(_) | OpKind::PredicateRead(_))
    }

    pub fn is_write(&self) -> bool {
        matches!(
            self.kind,
            OpKind::Write(_) | OpKind::Insert(_) | OpKind::Delete(_) | OpKind::PredicateWrite(_)
        )
    }

    pub fn is_data_operation(&self) -> bool {
        self.is_read() || self.is_write()
    }

    pub fn is_control(&self) -> bool {
        matches!(
            self.kind,
            OpKind::Begin(_) | OpKind::Commit(_) | OpKind::Abort(_)
        )
    }

    pub fn table_id(&self) -> Option<TableId> {
        match &self.kind {
            OpKind::Read(r) => Some(r.table),
            OpKind::Write(w) => Some(w.table),
            OpKind::Insert(i) => Some(i.table),
            OpKind::Delete(d) => Some(d.table),
            OpKind::PredicateRead(pr) => Some(pr.table),
            OpKind::PredicateWrite(pw) => Some(pw.table),
            OpKind::Lock(l) => Some(l.table),
            _ => None,
        }
    }

    pub fn item_id(&self) -> Option<ItemId> {
        match &self.kind {
            OpKind::Read(r) => Some(r.item),
            OpKind::Write(w) => Some(w.item),
            OpKind::Insert(i) => Some(i.item),
            _ => None,
        }
    }

    pub fn predicate(&self) -> Option<&Predicate> {
        match &self.kind {
            OpKind::PredicateRead(pr) => Some(&pr.predicate),
            OpKind::PredicateWrite(pw) => Some(&pw.predicate),
            OpKind::Delete(d) => Some(&d.predicate),
            OpKind::Lock(l) => l.predicate.as_ref(),
            _ => None,
        }
    }

    pub fn label(&self) -> &'static str {
        match &self.kind {
            OpKind::Read(_) => "read",
            OpKind::Write(_) => "write",
            OpKind::Insert(_) => "insert",
            OpKind::Delete(_) => "delete",
            OpKind::PredicateRead(_) => "pred_read",
            OpKind::PredicateWrite(_) => "pred_write",
            OpKind::Lock(_) => "lock",
            OpKind::Begin(_) => "begin",
            OpKind::Commit(_) => "commit",
            OpKind::Abort(_) => "abort",
        }
    }
}

impl fmt::Display for Operation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}({}, {})", self.label(), self.txn_id, self.id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identifier::*;

    fn tid(n: u64) -> TransactionId { TransactionId::new(n) }
    fn oid(n: u64) -> OperationId { OperationId::new(n) }
    fn tblid(n: u64) -> TableId { TableId::new(n) }
    fn iid(n: u64) -> ItemId { ItemId::new(n) }

    #[test]
    fn test_read_operation() {
        let op = Operation::read(oid(0), tid(1), tblid(0), iid(1), 100);
        assert!(op.is_read());
        assert!(!op.is_write());
        assert!(op.is_data_operation());
        assert!(!op.is_control());
        assert_eq!(op.table_id(), Some(tblid(0)));
        assert_eq!(op.item_id(), Some(iid(1)));
    }

    #[test]
    fn test_write_operation() {
        let op = Operation::write(oid(0), tid(1), tblid(0), iid(1), Value::Integer(42), 100);
        assert!(op.is_write());
        assert!(!op.is_read());
    }

    #[test]
    fn test_lock_compatibility() {
        assert!(LockMode::Shared.is_compatible_with(LockMode::Shared));
        assert!(!LockMode::Shared.is_compatible_with(LockMode::Exclusive));
        assert!(!LockMode::Exclusive.is_compatible_with(LockMode::Exclusive));
        assert!(LockMode::IntentShared.is_compatible_with(LockMode::IntentExclusive));
    }

    #[test]
    fn test_lock_strength() {
        assert!(LockMode::Exclusive.strength() > LockMode::Shared.strength());
        assert!(LockMode::Shared.strength() > LockMode::IntentShared.strength());
    }

    #[test]
    fn test_control_operations() {
        let begin = Operation::begin(oid(0), tid(1), crate::isolation::IsolationLevel::Serializable, 0);
        assert!(begin.is_control());
        assert!(!begin.is_data_operation());

        let commit = Operation::commit(oid(1), tid(1), 100);
        assert!(commit.is_control());
    }
}
