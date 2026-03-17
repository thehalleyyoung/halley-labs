//! Dependency types for the Direct Serialization Graph (DSG).
use serde::{Deserialize, Serialize};
use crate::identifier::*;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DependencyType {
    WriteWrite,
    WriteRead,
    ReadWrite,
    PredicateWriteRead,
    PredicateReadWrite,
    WritePredicateRead,
    PredicateWriteWrite,
}

impl DependencyType {
    pub fn short_name(self) -> &'static str {
        match self {
            Self::WriteWrite => "ww",
            Self::WriteRead => "wr",
            Self::ReadWrite => "rw",
            Self::PredicateWriteRead => "pwr",
            Self::PredicateReadWrite => "prw",
            Self::WritePredicateRead => "wpr",
            Self::PredicateWriteWrite => "pww",
        }
    }
    pub fn is_anti_dependency(self) -> bool {
        matches!(self, Self::ReadWrite | Self::PredicateReadWrite)
    }
    pub fn is_predicate_level(self) -> bool {
        matches!(self, Self::PredicateWriteRead | Self::PredicateReadWrite | Self::WritePredicateRead | Self::PredicateWriteWrite)
    }
}

impl fmt::Display for DependencyType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.short_name())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dependency {
    pub from_txn: TransactionId,
    pub to_txn: TransactionId,
    pub dep_type: DependencyType,
    pub from_op: Option<OperationId>,
    pub to_op: Option<OperationId>,
    pub item_id: Option<ItemId>,
    pub table_id: Option<TableId>,
}

impl Dependency {
    pub fn new(from: TransactionId, to: TransactionId, dep_type: DependencyType) -> Self {
        Self {
            from_txn: from, to_txn: to, dep_type,
            from_op: None, to_op: None, item_id: None, table_id: None,
        }
    }
    pub fn with_ops(mut self, from_op: OperationId, to_op: OperationId) -> Self {
        self.from_op = Some(from_op);
        self.to_op = Some(to_op);
        self
    }
    pub fn with_item(mut self, item: ItemId, table: TableId) -> Self {
        self.item_id = Some(item);
        self.table_id = Some(table);
        self
    }
}

impl fmt::Display for Dependency {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} --{}-> {}", self.from_txn, self.dep_type, self.to_txn)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_dependency_types() {
        assert!(DependencyType::ReadWrite.is_anti_dependency());
        assert!(!DependencyType::WriteRead.is_anti_dependency());
        assert!(DependencyType::PredicateReadWrite.is_predicate_level());
    }
    #[test]
    fn test_dependency_display() {
        let dep = Dependency::new(TransactionId::new(1), TransactionId::new(2), DependencyType::WriteRead);
        assert!(format!("{}", dep).contains("wr"));
    }
}
