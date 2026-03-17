//! Transaction Intermediate Representation (IR).
use serde::{Deserialize, Serialize};
use crate::identifier::*;
use crate::isolation::IsolationLevel;
use crate::predicate::Predicate;
use crate::value::Value;
use std::collections::HashMap;

/// A complete transaction IR program (workload).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrProgram {
    pub id: WorkloadId,
    pub name: String,
    pub transactions: Vec<IrTransaction>,
    pub schema_name: String,
    pub metadata: HashMap<String, String>,
}

/// A single transaction in the IR.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrTransaction {
    pub id: TransactionId,
    pub label: String,
    pub isolation_level: IsolationLevel,
    pub statements: Vec<IrStatement>,
    pub read_only: bool,
}

/// An IR statement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IrStatement {
    Select(IrSelect),
    Update(IrUpdate),
    Insert(IrInsert),
    Delete(IrDelete),
    Lock(IrLock),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrSelect {
    pub table: String,
    pub columns: Vec<String>,
    pub predicate: Predicate,
    pub for_update: bool,
    pub for_share: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrUpdate {
    pub table: String,
    pub assignments: Vec<(String, IrExpr)>,
    pub predicate: Predicate,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrInsert {
    pub table: String,
    pub columns: Vec<String>,
    pub values: Vec<Vec<IrExpr>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrDelete {
    pub table: String,
    pub predicate: Predicate,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrLock {
    pub table: String,
    pub mode: String,
    pub predicate: Option<Predicate>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IrExpr {
    Literal(Value),
    ColumnRef(String),
    BinaryOp { left: Box<IrExpr>, op: String, right: Box<IrExpr> },
    Function { name: String, args: Vec<IrExpr> },
    Null,
}

impl IrProgram {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            id: WorkloadId::new(0),
            name: name.into(),
            transactions: Vec::new(),
            schema_name: "public".into(),
            metadata: HashMap::new(),
        }
    }
    pub fn add_transaction(&mut self, txn: IrTransaction) {
        self.transactions.push(txn);
    }
    pub fn transaction_count(&self) -> usize {
        self.transactions.len()
    }
    pub fn total_statements(&self) -> usize {
        self.transactions.iter().map(|t| t.statements.len()).sum()
    }
    pub fn tables_accessed(&self) -> Vec<String> {
        let mut tables = Vec::new();
        for txn in &self.transactions {
            for stmt in &txn.statements {
                let table = match stmt {
                    IrStatement::Select(s) => s.table.clone(),
                    IrStatement::Update(u) => u.table.clone(),
                    IrStatement::Insert(i) => i.table.clone(),
                    IrStatement::Delete(d) => d.table.clone(),
                    IrStatement::Lock(l) => l.table.clone(),
                };
                if !tables.contains(&table) {
                    tables.push(table);
                }
            }
        }
        tables
    }
}

impl IrTransaction {
    pub fn new(id: TransactionId, label: impl Into<String>, level: IsolationLevel) -> Self {
        Self {
            id,
            label: label.into(),
            isolation_level: level,
            statements: Vec::new(),
            read_only: false,
        }
    }
    pub fn add_statement(&mut self, stmt: IrStatement) {
        self.statements.push(stmt);
    }
    pub fn has_writes(&self) -> bool {
        self.statements.iter().any(|s| matches!(s, IrStatement::Update(_) | IrStatement::Insert(_) | IrStatement::Delete(_)))
    }
    pub fn has_predicate_operations(&self) -> bool {
        self.statements.iter().any(|s| match s {
            IrStatement::Select(sel) => !matches!(sel.predicate, Predicate::True),
            IrStatement::Update(upd) => !matches!(upd.predicate, Predicate::True),
            IrStatement::Delete(del) => !matches!(del.predicate, Predicate::True),
            _ => false,
        })
    }
    pub fn statement_count(&self) -> usize {
        self.statements.len()
    }
}

impl IrExpr {
    pub fn literal(v: Value) -> Self { Self::Literal(v) }
    pub fn column(name: impl Into<String>) -> Self { Self::ColumnRef(name.into()) }
    pub fn add(left: IrExpr, right: IrExpr) -> Self {
        Self::BinaryOp { left: Box::new(left), op: "+".into(), right: Box::new(right) }
    }
    pub fn sub(left: IrExpr, right: IrExpr) -> Self {
        Self::BinaryOp { left: Box::new(left), op: "-".into(), right: Box::new(right) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::predicate::ColumnRef;

    #[test]
    fn test_ir_program() {
        let mut prog = IrProgram::new("test_workload");
        let mut txn = IrTransaction::new(TransactionId::new(0), "T1", IsolationLevel::Serializable);
        txn.add_statement(IrStatement::Select(IrSelect {
            table: "accounts".into(),
            columns: vec!["balance".into()],
            predicate: Predicate::eq(ColumnRef::new("id"), Value::Integer(1)),
            for_update: false,
            for_share: false,
        }));
        txn.add_statement(IrStatement::Update(IrUpdate {
            table: "accounts".into(),
            assignments: vec![("balance".into(), IrExpr::literal(Value::Integer(100)))],
            predicate: Predicate::eq(ColumnRef::new("id"), Value::Integer(1)),
        }));
        prog.add_transaction(txn);
        assert_eq!(prog.transaction_count(), 1);
        assert_eq!(prog.total_statements(), 2);
        assert_eq!(prog.tables_accessed(), vec!["accounts"]);
    }
    #[test]
    fn test_ir_transaction_writes() {
        let mut txn = IrTransaction::new(TransactionId::new(0), "T1", IsolationLevel::ReadCommitted);
        assert!(!txn.has_writes());
        txn.add_statement(IrStatement::Insert(IrInsert {
            table: "t".into(),
            columns: vec!["a".into()],
            values: vec![vec![IrExpr::literal(Value::Integer(1))]],
        }));
        assert!(txn.has_writes());
    }
}
