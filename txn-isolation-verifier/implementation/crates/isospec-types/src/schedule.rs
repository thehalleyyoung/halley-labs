//! Schedule types for representing execution orderings.
use serde::{Deserialize, Serialize};
use crate::identifier::*;
use crate::operation::Operation;
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Schedule {
    pub steps: Vec<ScheduleStep>,
    pub transaction_order: Vec<TransactionId>,
    pub metadata: ScheduleMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduleStep {
    pub id: ScheduleStepId,
    pub txn_id: TransactionId,
    pub operation: Operation,
    pub position: usize,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScheduleMetadata {
    pub engine: Option<String>,
    pub isolation_level: Option<crate::isolation::IsolationLevel>,
    pub is_witness: bool,
    pub anomaly_class: Option<crate::isolation::AnomalyClass>,
    pub generation_method: Option<String>,
}

impl Schedule {
    pub fn new() -> Self {
        Self { steps: Vec::new(), transaction_order: Vec::new(), metadata: ScheduleMetadata::default() }
    }

    pub fn add_step(&mut self, txn_id: TransactionId, op: Operation) {
        let pos = self.steps.len();
        let id = ScheduleStepId::new(pos as u64);
        if !self.transaction_order.contains(&txn_id) {
            self.transaction_order.push(txn_id);
        }
        self.steps.push(ScheduleStep { id, txn_id, operation: op, position: pos });
    }

    pub fn len(&self) -> usize { self.steps.len() }
    pub fn is_empty(&self) -> bool { self.steps.is_empty() }

    pub fn operations_for(&self, txn_id: TransactionId) -> Vec<&ScheduleStep> {
        self.steps.iter().filter(|s| s.txn_id == txn_id).collect()
    }

    pub fn transaction_ids(&self) -> Vec<TransactionId> {
        self.transaction_order.clone()
    }

    pub fn is_serial(&self) -> bool {
        let mut current_txn = None;
        let mut seen = std::collections::HashSet::new();
        for step in &self.steps {
            if Some(step.txn_id) != current_txn {
                if seen.contains(&step.txn_id) { return false; }
                if let Some(prev) = current_txn { seen.insert(prev); }
                current_txn = Some(step.txn_id);
            }
        }
        true
    }

    pub fn interleaving_count(&self) -> usize {
        let mut switches = 0;
        for i in 1..self.steps.len() {
            if self.steps[i].txn_id != self.steps[i - 1].txn_id {
                switches += 1;
            }
        }
        switches
    }

    pub fn prefix(&self, n: usize) -> Schedule {
        let mut s = Schedule::new();
        for step in self.steps.iter().take(n) {
            s.add_step(step.txn_id, step.operation.clone());
        }
        s.metadata = self.metadata.clone();
        s
    }

    pub fn committed_transactions(&self) -> Vec<TransactionId> {
        self.steps.iter()
            .filter(|s| matches!(s.operation.kind, crate::operation::OpKind::Commit(_)))
            .map(|s| s.txn_id)
            .collect()
    }

    pub fn aborted_transactions(&self) -> Vec<TransactionId> {
        self.steps.iter()
            .filter(|s| matches!(s.operation.kind, crate::operation::OpKind::Abort(_)))
            .map(|s| s.txn_id)
            .collect()
    }
}

impl Default for Schedule {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::Value;
    use crate::operation::Operation;

    #[test]
    fn test_schedule_serial_check() {
        let mut s = Schedule::new();
        let t1 = TransactionId::new(1);
        let t2 = TransactionId::new(2);
        s.add_step(t1, Operation::read(OperationId::new(0), t1, TableId::new(0), ItemId::new(0), 0));
        s.add_step(t1, Operation::write(OperationId::new(1), t1, TableId::new(0), ItemId::new(0), Value::Integer(1), 1));
        s.add_step(t2, Operation::read(OperationId::new(2), t2, TableId::new(0), ItemId::new(0), 2));
        assert!(s.is_serial());
    }

    #[test]
    fn test_schedule_non_serial() {
        let mut s = Schedule::new();
        let t1 = TransactionId::new(1);
        let t2 = TransactionId::new(2);
        s.add_step(t1, Operation::read(OperationId::new(0), t1, TableId::new(0), ItemId::new(0), 0));
        s.add_step(t2, Operation::read(OperationId::new(1), t2, TableId::new(0), ItemId::new(0), 1));
        s.add_step(t1, Operation::write(OperationId::new(2), t1, TableId::new(0), ItemId::new(0), Value::Integer(1), 2));
        assert!(!s.is_serial());
    }

    #[test]
    fn test_interleaving_count() {
        let mut s = Schedule::new();
        let t1 = TransactionId::new(1);
        let t2 = TransactionId::new(2);
        s.add_step(t1, Operation::read(OperationId::new(0), t1, TableId::new(0), ItemId::new(0), 0));
        s.add_step(t2, Operation::read(OperationId::new(1), t2, TableId::new(0), ItemId::new(0), 1));
        s.add_step(t1, Operation::write(OperationId::new(2), t1, TableId::new(0), ItemId::new(0), Value::Integer(1), 2));
        assert_eq!(s.interleaving_count(), 2);
    }
}
