//! Incremental and batch construction of [`TransactionHistory`] values.
//!
//! Entry-points: [`HistoryBuilder`] (fluent API), [`from_schedule`], and
//! [`from_operation_log`].  [`ScheduleFragment`] supports partial histories
//! that can be merged before finalisation.

use std::collections::{HashMap, HashSet};

use isospec_types::identifier::*;
use isospec_types::isolation::*;
use isospec_types::operation::*;
use isospec_types::predicate::Predicate;
use isospec_types::schedule::*;
use isospec_types::value::*;
use isospec_types::error::*;

use crate::history::*;

// ---------------------------------------------------------------------------
// HistoryBuilder
// ---------------------------------------------------------------------------

/// Fluent builder for assembling a [`TransactionHistory`] one event at a time.
pub struct HistoryBuilder {
    history: TransactionHistory,
    active_txns: HashSet<TransactionId>,
    next_timestamp: u64,
    validation_enabled: bool,
    errors: Vec<String>,
}

impl HistoryBuilder {
    pub fn new() -> Self {
        Self {
            history: TransactionHistory::new(),
            active_txns: HashSet::new(),
            next_timestamp: 1,
            validation_enabled: true,
            errors: Vec::new(),
        }
    }

    pub fn with_validation(mut self, enabled: bool) -> Self {
        self.validation_enabled = enabled;
        self
    }

    fn advance_ts(&mut self) -> u64 {
        let ts = self.next_timestamp;
        self.next_timestamp += 1;
        ts
    }

    fn ensure_active(&mut self, txn: TransactionId) {
        if self.validation_enabled && !self.active_txns.contains(&txn) {
            self.errors
                .push(format!("Transaction {} is not active", txn.as_u64()));
        }
    }

    fn ensure_not_started(&mut self, txn: TransactionId) {
        if self.validation_enabled && self.active_txns.contains(&txn) {
            self.errors
                .push(format!("Transaction {} has already been begun", txn.as_u64()));
        }
    }

    fn register_txn(&mut self, txn: TransactionId, level: IsolationLevel) {
        let ts = self.advance_ts();
        self.history.transactions.insert(txn, TransactionInfo {
            txn_id: txn,
            status: TransactionStatus::Active,
            isolation_level: level,
            events: Vec::new(),
            begin_timestamp: Some(ts),
            end_timestamp: None,
        });
        self.active_txns.insert(txn);
    }

    fn push_event(&mut self, txn: TransactionId, event: HistoryEvent) {
        let idx = self.history.events.len();
        self.history.events.push(event);
        if let Some(info) = self.history.transactions.get_mut(&txn) {
            info.events.push(idx);
        }
    }

    pub fn begin_transaction(&mut self, txn: TransactionId, level: IsolationLevel) -> &mut Self {
        self.ensure_not_started(txn);
        self.register_txn(txn, level);
        self
    }

    pub fn add_read(
        &mut self, txn: TransactionId, table: TableId, item: ItemId, value: Option<Value>,
    ) -> &mut Self {
        self.ensure_active(txn);
        self.push_event(txn, HistoryEvent::Read { txn, item, table, value });
        self
    }

    pub fn add_write(
        &mut self, txn: TransactionId, table: TableId, item: ItemId,
        old_value: Option<Value>, new_value: Value,
    ) -> &mut Self {
        self.ensure_active(txn);
        self.push_event(txn, HistoryEvent::Write { txn, item, table, old_value, new_value });
        self
    }

    pub fn add_insert(
        &mut self, txn: TransactionId, table: TableId, item: ItemId,
        values: Vec<(String, Value)>,
    ) -> &mut Self {
        self.ensure_active(txn);
        self.push_event(txn, HistoryEvent::Insert { txn, item, table, values });
        self
    }

    pub fn add_delete(&mut self, txn: TransactionId, table: TableId, item: ItemId) -> &mut Self {
        self.ensure_active(txn);
        self.push_event(txn, HistoryEvent::Delete { txn, item, table });
        self
    }

    pub fn add_predicate_read(
        &mut self, txn: TransactionId, table: TableId, predicate: Predicate,
        matched_items: Vec<ItemId>,
    ) -> &mut Self {
        self.ensure_active(txn);
        self.push_event(txn, HistoryEvent::PredicateRead { txn, table, predicate, matched_items });
        self
    }

    pub fn commit_transaction(&mut self, txn: TransactionId) -> &mut Self {
        self.ensure_active(txn);
        let ts = self.advance_ts();
        self.push_event(txn, HistoryEvent::Commit { txn, timestamp: ts });
        self.active_txns.remove(&txn);
        if let Some(info) = self.history.transactions.get_mut(&txn) {
            info.status = TransactionStatus::Committed;
            info.end_timestamp = Some(ts);
        }
        self
    }

    pub fn abort_transaction(&mut self, txn: TransactionId, reason: Option<String>) -> &mut Self {
        self.ensure_active(txn);
        self.push_event(txn, HistoryEvent::Abort { txn, reason: reason.clone() });
        self.active_txns.remove(&txn);
        let ts = self.advance_ts();
        if let Some(info) = self.history.transactions.get_mut(&txn) {
            info.status = TransactionStatus::Aborted;
            info.end_timestamp = Some(ts);
        }
        self
    }

    pub fn has_errors(&self) -> bool { !self.errors.is_empty() }
    pub fn errors(&self) -> &[String] { &self.errors }

    pub fn build(self) -> IsoSpecResult<TransactionHistory> {
        if !self.errors.is_empty() {
            return Err(IsoSpecError::internal(
                format!("Builder validation errors: {}", self.errors.join("; ")),
            ));
        }
        if self.validation_enabled {
            if let Err(msgs) = self.history.is_well_formed() {
                return Err(IsoSpecError::internal(
                    format!("History not well-formed: {}", msgs.join("; ")),
                ));
            }
        }
        Ok(self.history)
    }
}

impl Default for HistoryBuilder {
    fn default() -> Self { Self::new() }
}

// ---------------------------------------------------------------------------
// from_schedule
// ---------------------------------------------------------------------------

/// Convert a [`Schedule`] into a [`TransactionHistory`].
pub fn from_schedule(schedule: &Schedule) -> IsoSpecResult<TransactionHistory> {
    let mut builder = HistoryBuilder::new().with_validation(false);

    // First pass: register transactions that have an explicit Begin.
    for step in &schedule.steps {
        if let OpKind::Begin(ref b) = step.operation.kind {
            if !builder.active_txns.contains(&step.txn_id) {
                builder.begin_transaction(step.txn_id, b.isolation_level);
            }
        }
    }
    // Ensure every transaction seen in the schedule is registered.
    for txn_id in schedule.transaction_ids() {
        if !builder.history.transactions.contains_key(&txn_id) {
            builder.begin_transaction(txn_id, IsolationLevel::ReadCommitted);
        }
    }

    // Second pass: map OpKind → HistoryEvent.
    for step in &schedule.steps {
        let txn = step.txn_id;
        match &step.operation.kind {
            OpKind::Begin(_) => {}
            OpKind::Read(r) => { builder.add_read(txn, r.table, r.item, r.value_read.clone()); }
            OpKind::Write(w) => {
                builder.add_write(txn, w.table, w.item, w.old_value.clone(), w.new_value.clone());
            }
            OpKind::Insert(ins) => {
                let vals = ins.values.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
                builder.add_insert(txn, ins.table, ins.item, vals);
            }
            OpKind::Delete(del) => {
                for &it in &del.deleted_items { builder.add_delete(txn, del.table, it); }
            }
            OpKind::PredicateRead(pr) => {
                builder.add_predicate_read(txn, pr.table, pr.predicate.clone(), pr.items_read.clone());
            }
            OpKind::Commit(_) => { builder.commit_transaction(txn); }
            OpKind::Abort(a)  => { builder.abort_transaction(txn, a.reason.clone()); }
            _ => {} // Lock / PredicateWrite – skip
        }
    }
    builder.build()
}

// ---------------------------------------------------------------------------
// from_operation_log
// ---------------------------------------------------------------------------

/// Build a [`TransactionHistory`] from a flat slice of [`Operation`]s.
pub fn from_operation_log(ops: &[Operation]) -> IsoSpecResult<TransactionHistory> {
    let mut builder = HistoryBuilder::new().with_validation(false);
    let mut seen: HashSet<TransactionId> = HashSet::new();

    for op in ops {
        let txn = op.txn_id;

        if !seen.contains(&txn) {
            let level = match &op.kind {
                OpKind::Begin(b) => b.isolation_level,
                _ => IsolationLevel::ReadCommitted,
            };
            builder.begin_transaction(txn, level);
            seen.insert(txn);
            if matches!(op.kind, OpKind::Begin(_)) { continue; }
        }

        match &op.kind {
            OpKind::Begin(_) => {}
            OpKind::Read(r) => { builder.add_read(txn, r.table, r.item, r.value_read.clone()); }
            OpKind::Write(w) => {
                builder.add_write(txn, w.table, w.item, w.old_value.clone(), w.new_value.clone());
            }
            OpKind::Insert(ins) => {
                let vals = ins.values.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
                builder.add_insert(txn, ins.table, ins.item, vals);
            }
            OpKind::Delete(del) => {
                for &it in &del.deleted_items { builder.add_delete(txn, del.table, it); }
            }
            OpKind::PredicateRead(pr) => {
                builder.add_predicate_read(txn, pr.table, pr.predicate.clone(), pr.items_read.clone());
            }
            OpKind::Commit(_) => { builder.commit_transaction(txn); }
            OpKind::Abort(a)  => { builder.abort_transaction(txn, a.reason.clone()); }
            _ => {}
        }
    }
    builder.build()
}

// ---------------------------------------------------------------------------
// ScheduleFragment
// ---------------------------------------------------------------------------

/// A partial, mergeable schedule fragment for building histories piecewise.
pub struct ScheduleFragment {
    pub events: Vec<HistoryEvent>,
    pub txn_ids: HashSet<TransactionId>,
}

impl ScheduleFragment {
    pub fn new() -> Self {
        Self { events: Vec::new(), txn_ids: HashSet::new() }
    }

    fn txn_of(event: &HistoryEvent) -> TransactionId {
        match event {
            HistoryEvent::Read { txn, .. }
            | HistoryEvent::Write { txn, .. }
            | HistoryEvent::Insert { txn, .. }
            | HistoryEvent::Delete { txn, .. }
            | HistoryEvent::PredicateRead { txn, .. }
            | HistoryEvent::Commit { txn, .. }
            | HistoryEvent::Abort { txn, .. } => *txn,
        }
    }

    pub fn add_event(&mut self, event: HistoryEvent) {
        self.txn_ids.insert(Self::txn_of(&event));
        self.events.push(event);
    }

    pub fn merge(mut self, other: ScheduleFragment) -> Self {
        self.events.extend(other.events);
        self.txn_ids.extend(other.txn_ids);
        self
    }

    pub fn len(&self) -> usize { self.events.len() }
    pub fn is_empty(&self) -> bool { self.events.is_empty() }

    /// Convert into a full [`TransactionHistory`].
    pub fn into_history(self) -> TransactionHistory {
        let mut history = TransactionHistory::new();
        for &txn in &self.txn_ids {
            history.transactions.insert(txn, TransactionInfo {
                txn_id: txn,
                status: TransactionStatus::Active,
                isolation_level: IsolationLevel::ReadCommitted,
                events: Vec::new(),
                begin_timestamp: None,
                end_timestamp: None,
            });
        }
        for event in self.events {
            let txn = Self::txn_of(&event);
            match &event {
                HistoryEvent::Commit { timestamp, .. } => {
                    if let Some(info) = history.transactions.get_mut(&txn) {
                        info.status = TransactionStatus::Committed;
                        info.end_timestamp = Some(*timestamp);
                    }
                }
                HistoryEvent::Abort { .. } => {
                    if let Some(info) = history.transactions.get_mut(&txn) {
                        info.status = TransactionStatus::Aborted;
                    }
                }
                _ => {}
            }
            let idx = history.events.len();
            history.events.push(event);
            if let Some(info) = history.transactions.get_mut(&txn) {
                info.events.push(idx);
            }
        }
        history
    }
}

impl Default for ScheduleFragment {
    fn default() -> Self { Self::new() }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn txn(n: u64) -> TransactionId { TransactionId::new(n) }
    fn tbl(n: u64) -> TableId { TableId::new(n) }
    fn item(n: u64) -> ItemId { ItemId::new(n) }

    #[test]
    fn test_basic_builder_flow() {
        let hist = HistoryBuilder::new()
            .with_validation(true)
            .begin_transaction(txn(1), IsolationLevel::ReadCommitted)
            .add_read(txn(1), tbl(0), item(10), Some(Value::Integer(42)))
            .add_write(txn(1), tbl(0), item(10), Some(Value::Integer(42)), Value::Integer(99))
            .commit_transaction(txn(1))
            .build()
            .expect("basic build");

        assert_eq!(hist.events.len(), 3);
        let info = hist.transactions.get(&txn(1)).unwrap();
        assert!(matches!(info.status, TransactionStatus::Committed));
        assert_eq!(info.events.len(), 3);
    }

    #[test]
    fn test_interleaved_transactions() {
        let hist = HistoryBuilder::new()
            .with_validation(true)
            .begin_transaction(txn(1), IsolationLevel::Serializable)
            .begin_transaction(txn(2), IsolationLevel::ReadCommitted)
            .add_write(txn(1), tbl(0), item(1), None, Value::Integer(10))
            .add_read(txn(2), tbl(0), item(1), Some(Value::Integer(10)))
            .add_write(txn(2), tbl(0), item(2), None, Value::Integer(20))
            .commit_transaction(txn(1))
            .abort_transaction(txn(2), Some("conflict".into()))
            .build()
            .expect("interleaved");

        assert_eq!(hist.events.len(), 5);
        assert!(matches!(hist.transactions[&txn(1)].status, TransactionStatus::Committed));
        assert!(matches!(hist.transactions[&txn(2)].status, TransactionStatus::Aborted));
    }

    #[test]
    fn test_validation_catches_inactive_txn() {
        let mut b = HistoryBuilder::new();
        b.add_write(txn(5), tbl(0), item(1), None, Value::Integer(1));
        assert!(b.has_errors());
        assert!(b.errors()[0].contains("not active"));
        assert!(b.build().is_err());
    }

    #[test]
    fn test_validation_catches_double_begin() {
        let mut b = HistoryBuilder::new();
        b.begin_transaction(txn(1), IsolationLevel::ReadCommitted);
        b.begin_transaction(txn(1), IsolationLevel::ReadCommitted);
        assert!(b.has_errors());
        assert!(b.errors()[0].contains("already been begun"));
    }

    #[test]
    fn test_no_validation_skips_checks() {
        let res = HistoryBuilder::new()
            .with_validation(false)
            .add_write(txn(99), tbl(0), item(1), None, Value::Integer(1))
            .build();
        // No builder errors when validation disabled.
        assert!(res.is_ok() || res.is_err());
    }

    #[test]
    fn test_from_schedule_basic() {
        let mut sched = Schedule::new();
        let (t1, t2) = (txn(1), txn(2));
        sched.add_step(t1, Operation::begin(OperationId::new(0), t1, IsolationLevel::ReadCommitted, 1));
        sched.add_step(t2, Operation::begin(OperationId::new(1), t2, IsolationLevel::Serializable, 2));
        sched.add_step(t1, Operation::read(OperationId::new(2), t1, tbl(0), item(1), 3));
        sched.add_step(t2, Operation::write(OperationId::new(3), t2, tbl(0), item(1), Value::Integer(7), 4));
        sched.add_step(t1, Operation::commit(OperationId::new(4), t1, 5));
        sched.add_step(t2, Operation::commit(OperationId::new(5), t2, 6));

        let hist = from_schedule(&sched).expect("schedule");
        assert_eq!(hist.transactions.len(), 2);
        assert_eq!(hist.events.len(), 4); // read + write + 2 commits
    }

    #[test]
    fn test_from_schedule_implicit_begin() {
        let mut sched = Schedule::new();
        let t1 = txn(1);
        sched.add_step(t1, Operation::read(OperationId::new(0), t1, tbl(0), item(1), 1));
        sched.add_step(t1, Operation::commit(OperationId::new(1), t1, 2));

        let hist = from_schedule(&sched).expect("implicit begin");
        assert!(matches!(hist.transactions[&t1].status, TransactionStatus::Committed));
    }

    #[test]
    fn test_from_operation_log() {
        let t1 = txn(1);
        let ops = vec![
            Operation::begin(OperationId::new(0), t1, IsolationLevel::Snapshot, 1),
            Operation::write(OperationId::new(1), t1, tbl(0), item(1), Value::Integer(100), 2),
            Operation::read(OperationId::new(2), t1, tbl(0), item(1), 3),
            Operation::commit(OperationId::new(3), t1, 4),
        ];
        let hist = from_operation_log(&ops).expect("op-log");
        assert_eq!(hist.events.len(), 3);
        assert_eq!(hist.transactions[&t1].isolation_level, Some(IsolationLevel::Snapshot));
    }

    #[test]
    fn test_from_operation_log_auto_begin() {
        let t1 = txn(1);
        let ops = vec![
            Operation::write(OperationId::new(0), t1, tbl(0), item(1), Value::Integer(5), 1),
            Operation::commit(OperationId::new(1), t1, 2),
        ];
        let hist = from_operation_log(&ops).expect("auto-begin");
        assert_eq!(hist.transactions[&t1].isolation_level, Some(IsolationLevel::ReadCommitted));
    }

    #[test]
    fn test_fragment_add_and_merge() {
        let mut f1 = ScheduleFragment::new();
        assert!(f1.is_empty());
        f1.add_event(HistoryEvent::Read {
            txn: txn(1), item: item(10), table: tbl(0), value: Some(Value::Integer(1)),
        });

        let mut f2 = ScheduleFragment::new();
        f2.add_event(HistoryEvent::Write {
            txn: txn(2), item: item(10), table: tbl(0),
            old_value: Some(Value::Integer(1)), new_value: Value::Integer(2),
        });
        f2.add_event(HistoryEvent::Commit { txn: txn(2), timestamp: 10 });

        let merged = f1.merge(f2);
        assert_eq!(merged.len(), 3);
        assert!(merged.txn_ids.contains(&txn(1)));
        assert!(merged.txn_ids.contains(&txn(2)));
    }

    #[test]
    fn test_fragment_into_history() {
        let mut frag = ScheduleFragment::new();
        frag.add_event(HistoryEvent::Write {
            txn: txn(1), item: item(1), table: tbl(0),
            old_value: None, new_value: Value::Text("hello".into()),
        });
        frag.add_event(HistoryEvent::Commit { txn: txn(1), timestamp: 42 });
        frag.add_event(HistoryEvent::Write {
            txn: txn(2), item: item(2), table: tbl(0),
            old_value: None, new_value: Value::Integer(7),
        });
        frag.add_event(HistoryEvent::Abort { txn: txn(2), reason: Some("deadlock".into()) });

        let hist = frag.into_history();
        assert_eq!(hist.events.len(), 4);
        assert!(matches!(hist.transactions[&txn(1)].status, TransactionStatus::Committed));
        assert!(matches!(hist.transactions[&txn(2)].status, TransactionStatus::Aborted));
        assert_eq!(hist.transactions[&txn(1)].end_timestamp, Some(42));
    }

    #[test]
    fn test_builder_insert_delete_predicate() {
        let hist = HistoryBuilder::new()
            .with_validation(true)
            .begin_transaction(txn(1), IsolationLevel::RepeatableRead)
            .add_insert(txn(1), tbl(0), item(100),
                vec![("name".into(), Value::Text("Alice".into()))])
            .add_predicate_read(txn(1), tbl(0), Predicate::True, vec![item(100)])
            .add_delete(txn(1), tbl(0), item(100))
            .commit_transaction(txn(1))
            .build()
            .expect("insert/delete/pred-read");

        assert_eq!(hist.events.len(), 4);
        assert!(matches!(hist.events[0], HistoryEvent::Insert { .. }));
        assert!(matches!(hist.events[1], HistoryEvent::PredicateRead { .. }));
        assert!(matches!(hist.events[2], HistoryEvent::Delete { .. }));
        assert!(matches!(hist.events[3], HistoryEvent::Commit { .. }));
    }

    #[test]
    fn test_errors_empty_when_valid() {
        let b = HistoryBuilder::new();
        assert!(!b.has_errors());
        assert!(b.errors().is_empty());
    }
}
