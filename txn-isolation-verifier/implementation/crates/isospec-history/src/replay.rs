//! Schedule replay engine: executes a `Schedule` step-by-step against an
//! in-memory versioned data store, producing a `TransactionHistory` and trace.

use std::collections::HashMap;
use isospec_types::identifier::*;
use isospec_types::isolation::*;
use isospec_types::operation::*;
use isospec_types::schedule::*;
use isospec_types::value::*;
use isospec_types::error::*;
use crate::history::*;
use crate::trace::*;

// ---------------------------------------------------------------------------
// ItemVersion / VersionedItem
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ItemVersion {
    pub version: u64,
    pub value: Value,
    pub written_by: TransactionId,
    pub committed: bool,
}

#[derive(Debug, Clone)]
pub struct VersionedItem {
    pub versions: Vec<ItemVersion>,
    pub deleted: bool,
}

impl VersionedItem {
    fn new() -> Self { Self { versions: Vec::new(), deleted: false } }

    fn visible_version(
        &self, reader: TransactionId, isolation: IsolationLevel,
        committed_txns: &[TransactionId],
    ) -> Option<&ItemVersion> {
        for v in self.versions.iter().rev() {
            if v.written_by == reader { return Some(v); }
            match isolation {
                IsolationLevel::ReadUncommitted => return Some(v),
                _ => {
                    if v.committed || committed_txns.contains(&v.written_by) {
                        return Some(v);
                    }
                }
            }
        }
        None
    }

    fn latest_version(&self) -> Option<&ItemVersion> { self.versions.last() }
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum ReadResult {
    Found { value: Value, version: u64 },
    NotFound,
}

#[derive(Debug, Clone)]
pub enum WriteResult {
    Ok { old: Option<Value>, new_value: Value, version: u64 },
    Conflict { holder: TransactionId },
}

#[derive(Debug, Clone)]
pub enum InsertResult { Ok { version: u64 }, AlreadyExists }

#[derive(Debug, Clone)]
pub enum DeleteResult { Ok, NotFound, Conflict { holder: TransactionId } }

// ---------------------------------------------------------------------------
// DataStore
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct DataStore {
    items: HashMap<(TableId, ItemId), VersionedItem>,
    next_version: u64,
    committed_txns: Vec<TransactionId>,
}

impl DataStore {
    pub fn new() -> Self {
        Self { items: HashMap::new(), next_version: 1, committed_txns: Vec::new() }
    }

    fn alloc_version(&mut self) -> u64 {
        let v = self.next_version; self.next_version += 1; v
    }

    pub fn read(&self, table: TableId, item: ItemId, txn: TransactionId, isolation: IsolationLevel) -> ReadResult {
        match self.items.get(&(table, item)) {
            Some(vi) if !vi.deleted || vi.versions.iter().any(|v| v.written_by == txn) => {
                match vi.visible_version(txn, isolation, &self.committed_txns) {
                    Some(v) => ReadResult::Found { value: v.value.clone(), version: v.version },
                    None => ReadResult::NotFound,
                }
            }
            _ => ReadResult::NotFound,
        }
    }

    pub fn write(&mut self, table: TableId, item: ItemId, value: Value, txn: TransactionId) -> WriteResult {
        let ver = self.alloc_version();
        let entry = self.items.entry((table, item)).or_insert_with(VersionedItem::new);
        if let Some(latest) = entry.latest_version() {
            if !latest.committed && latest.written_by != txn
                && !self.committed_txns.contains(&latest.written_by) {
                return WriteResult::Conflict { holder: latest.written_by };
            }
        }
        let old = entry.visible_version(txn, IsolationLevel::ReadCommitted, &self.committed_txns)
            .map(|v| v.value.clone());
        entry.versions.push(ItemVersion { version: ver, value: value.clone(), written_by: txn, committed: false });
        entry.deleted = false;
        WriteResult::Ok { old, new_value: value, version: ver }
    }

    pub fn insert(&mut self, table: TableId, item: ItemId, value: Value, txn: TransactionId) -> InsertResult {
        let key = (table, item);
        if let Some(existing) = self.items.get(&key) {
            if !existing.deleted && existing.versions.iter()
                .any(|v| v.committed || self.committed_txns.contains(&v.written_by)) {
                return InsertResult::AlreadyExists;
            }
        }
        let ver = self.alloc_version();
        let entry = self.items.entry(key).or_insert_with(VersionedItem::new);
        entry.deleted = false;
        entry.versions.push(ItemVersion { version: ver, value, written_by: txn, committed: false });
        InsertResult::Ok { version: ver }
    }

    pub fn delete(&mut self, table: TableId, item: ItemId, txn: TransactionId) -> DeleteResult {
        match self.items.get_mut(&(table, item)) {
            None => DeleteResult::NotFound,
            Some(vi) => {
                if let Some(latest) = vi.latest_version() {
                    if !latest.committed && latest.written_by != txn
                        && !self.committed_txns.contains(&latest.written_by) {
                        return DeleteResult::Conflict { holder: latest.written_by };
                    }
                }
                vi.deleted = true;
                DeleteResult::Ok
            }
        }
    }

    pub fn commit(&mut self, txn: TransactionId) {
        if !self.committed_txns.contains(&txn) { self.committed_txns.push(txn); }
        for vi in self.items.values_mut() {
            for v in vi.versions.iter_mut() {
                if v.written_by == txn { v.committed = true; }
            }
        }
    }

    pub fn abort(&mut self, txn: TransactionId) {
        for vi in self.items.values_mut() {
            vi.versions.retain(|v| v.written_by != txn || v.committed);
            if vi.deleted && !vi.versions.is_empty() { vi.deleted = false; }
        }
    }

    pub fn snapshot(&self, txn: TransactionId) -> HashMap<(TableId, ItemId), Value> {
        let mut snap = HashMap::new();
        for (&key, vi) in &self.items {
            if vi.deleted { continue; }
            if let Some(v) = vi.visible_version(txn, IsolationLevel::ReadCommitted, &self.committed_txns) {
                snap.insert(key, v.value.clone());
            }
        }
        snap
    }

    pub fn item_exists(&self, table: TableId, item: ItemId) -> bool {
        match self.items.get(&(table, item)) {
            Some(vi) => !vi.deleted && vi.versions.iter().any(|v| v.committed),
            None => false,
        }
    }
}

impl Default for DataStore { fn default() -> Self { Self::new() } }

// ---------------------------------------------------------------------------
// StoreSnapshot
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct StoreSnapshot { pub items: HashMap<(TableId, ItemId), Value> }

impl StoreSnapshot {
    fn capture(store: &DataStore, viewer: TransactionId) -> Self {
        Self { items: store.snapshot(viewer) }
    }
}

// ---------------------------------------------------------------------------
// StepResult / ReplayStep
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum StepResult {
    Success(String),
    ReadValue(Value),
    WriteOk { old: Option<Value>, new: Value },
    Conflict(String),
    Skipped(String),
}

#[derive(Debug, Clone)]
pub struct ReplayStep {
    pub step_index: usize,
    pub txn_id: TransactionId,
    pub operation_label: String,
    pub result: StepResult,
    pub state_before: StoreSnapshot,
    pub state_after: StoreSnapshot,
}

// ---------------------------------------------------------------------------
// ReplayResult
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ReplayResult {
    pub steps: Vec<ReplayStep>,
    pub final_state: StoreSnapshot,
    pub committed_txns: Vec<TransactionId>,
    pub aborted_txns: Vec<TransactionId>,
    pub conflicts: Vec<String>,
    pub history: TransactionHistory,
}

// ---------------------------------------------------------------------------
// ReplayEngine
// ---------------------------------------------------------------------------

pub struct ReplayEngine {
    store: DataStore,
    recorder: TraceRecorder,
    steps: Vec<ReplayStep>,
    active_txns: HashMap<TransactionId, IsolationLevel>,
    conflicts: Vec<String>,
    history: TransactionHistory,
}

impl ReplayEngine {
    pub fn new() -> Self {
        let mut recorder = TraceRecorder::new();
        recorder.start();
        Self {
            store: DataStore::new(), recorder, steps: Vec::new(),
            active_txns: HashMap::new(), conflicts: Vec::new(),
            history: TransactionHistory::new(),
        }
    }

    pub fn replay_schedule(&mut self, schedule: &Schedule) -> IsoSpecResult<ReplayResult> {
        for step in &schedule.steps {
            let replay_step = self.execute_step(step)?;
            self.steps.push(replay_step);
        }
        self.recorder.stop();
        Ok(ReplayResult {
            steps: self.steps.clone(),
            final_state: self.take_snapshot(),
            committed_txns: self.history.committed_transactions(),
            aborted_txns: self.history.aborted_transactions(),
            conflicts: self.conflicts.clone(),
            history: self.history.clone(),
        })
    }

    pub fn execute_step(&mut self, step: &ScheduleStep) -> IsoSpecResult<ReplayStep> {
        let txn = step.txn_id;
        let isolation = self.active_txns.get(&txn).copied()
            .unwrap_or(IsolationLevel::Serializable);
        let state_before = self.take_snapshot();

        let (label, result) = match &step.operation.kind {
            OpKind::Begin(begin) => {
                self.active_txns.insert(txn, begin.isolation_level);
                self.recorder.record(TraceEventKind::BeginTransaction {
                    txn, isolation: begin.isolation_level,
                });
                ("begin".into(), StepResult::Success(format!("begin {:?}", begin.isolation_level)))
            }
            OpKind::Read(read) => self.exec_read(txn, read, isolation),
            OpKind::Write(write) => self.exec_write(txn, write, isolation),
            OpKind::Insert(ins) => self.exec_insert(txn, ins, isolation),
            OpKind::Delete(del) => {
                match self.exec_delete(txn, del, isolation, &state_before)? {
                    Some(early) => return Ok(early),
                    None => unreachable!(),
                }
            }
            OpKind::PredicateRead(pr) => {
                self.history.add_event(HistoryEvent::PredicateRead {
                    txn, table: pr.table, predicate: pr.predicate.clone(),
                    matched_items: pr.items_read.clone(),
                }, isolation);
                let n = pr.result_count.unwrap_or(pr.items_read.len());
                (format!("predicate_read {} => {} items", pr.table, n),
                 StepResult::Success(format!("{} items matched", n)))
            }
            OpKind::Commit(c) => {
                self.store.commit(txn);
                self.active_txns.remove(&txn);
                self.recorder.record(TraceEventKind::CommitTransaction { txn });
                self.history.add_event(HistoryEvent::Commit { txn, timestamp: c.commit_timestamp }, isolation);
                ("commit".into(), StepResult::Success("committed".into()))
            }
            OpKind::Abort(a) => {
                self.store.abort(txn);
                self.active_txns.remove(&txn);
                self.recorder.record(TraceEventKind::AbortTransaction { txn, reason: a.reason.clone() });
                self.history.add_event(HistoryEvent::Abort { txn, reason: a.reason.clone() }, isolation);
                ("abort".into(), StepResult::Success("aborted".into()))
            }
            OpKind::PredicateWrite(_) | OpKind::Lock(_) => {
                ("skipped".into(), StepResult::Skipped("unsupported op kind".into()))
            }
        };

        let state_after = self.take_snapshot();
        Ok(ReplayStep {
            step_index: self.steps.len(), txn_id: txn, operation_label: label,
            result, state_before, state_after,
        })
    }

    fn exec_read(&mut self, txn: TransactionId, read: &ReadOp, iso: IsolationLevel) -> (String, StepResult) {
        let rr = self.store.read(read.table, read.item, txn, iso);
        let (val_opt, lbl) = match &rr {
            ReadResult::Found { value, version } =>
                (Some(value.clone()), format!("read {}.{} v{}={}", read.table, read.item, version, value)),
            ReadResult::NotFound =>
                (None, format!("read {}.{} => NULL", read.table, read.item)),
        };
        self.recorder.record(TraceEventKind::ReadItem { txn, table: read.table, item: read.item, value: val_opt.clone() });
        self.history.add_event(HistoryEvent::Read { txn, item: read.item, table: read.table, value: val_opt.clone() }, iso);
        let sr = match val_opt { Some(v) => StepResult::ReadValue(v), None => StepResult::ReadValue(Value::Null) };
        (lbl, sr)
    }

    fn exec_write(&mut self, txn: TransactionId, write: &WriteOp, iso: IsolationLevel) -> (String, StepResult) {
        let wr = self.store.write(write.table, write.item, write.new_value.clone(), txn);
        match wr {
            WriteResult::Ok { old, new_value, .. } => {
                self.recorder.record(TraceEventKind::WriteItem {
                    txn, table: write.table, item: write.item,
                    old_value: old.clone(), new_value: new_value.clone(),
                });
                self.history.add_event(HistoryEvent::Write {
                    txn, item: write.item, table: write.table,
                    old_value: old.clone(), new_value: new_value.clone(),
                }, iso);
                (format!("write {}.{} = {}", write.table, write.item, new_value),
                 StepResult::WriteOk { old, new: new_value })
            }
            WriteResult::Conflict { holder } => {
                let msg = format!("write-write conflict on {}.{}: held by {}", write.table, write.item, holder);
                self.conflicts.push(msg.clone());
                (format!("write {}.{} CONFLICT", write.table, write.item), StepResult::Conflict(msg))
            }
        }
    }

    fn exec_insert(&mut self, txn: TransactionId, ins: &InsertOp, iso: IsolationLevel) -> (String, StepResult) {
        let combined = ins.values.values().next().cloned().unwrap_or(Value::Null);
        match self.store.insert(ins.table, ins.item, combined, txn) {
            InsertResult::Ok { .. } => {
                self.recorder.record(TraceEventKind::InsertItem { txn, table: ins.table, item: ins.item });
                let vals: Vec<(String, Value)> = ins.values.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
                self.history.add_event(HistoryEvent::Insert { txn, item: ins.item, table: ins.table, values: vals }, iso);
                (format!("insert {}.{}", ins.table, ins.item), StepResult::Success("inserted".into()))
            }
            InsertResult::AlreadyExists => {
                let msg = format!("insert {}.{} already exists", ins.table, ins.item);
                self.conflicts.push(msg.clone());
                (msg.clone(), StepResult::Conflict(msg))
            }
        }
    }

    fn exec_delete(
        &mut self, txn: TransactionId, del: &DeleteOp, iso: IsolationLevel,
        state_before: &StoreSnapshot,
    ) -> IsoSpecResult<Option<ReplayStep>> {
        let mut deleted_any = false;
        for &item in &del.deleted_items {
            match self.store.delete(del.table, item, txn) {
                DeleteResult::Ok => {
                    self.recorder.record(TraceEventKind::DeleteItem { txn, table: del.table, item });
                    self.history.add_event(HistoryEvent::Delete { txn, item, table: del.table }, iso);
                    deleted_any = true;
                }
                DeleteResult::NotFound => {}
                DeleteResult::Conflict { holder } => {
                    let msg = format!("delete conflict on {}.{}: held by {}", del.table, item, holder);
                    self.conflicts.push(msg.clone());
                    return Ok(Some(ReplayStep {
                        step_index: self.steps.len(), txn_id: txn,
                        operation_label: format!("delete {}", del.table),
                        result: StepResult::Conflict(msg),
                        state_before: state_before.clone(), state_after: self.take_snapshot(),
                    }));
                }
            }
        }
        let sr = if deleted_any { StepResult::Success("deleted".into()) }
                 else { StepResult::Skipped("nothing to delete".into()) };
        let label = format!("delete {} items from {}", del.deleted_items.len(), del.table);
        Ok(Some(ReplayStep {
            step_index: self.steps.len(), txn_id: txn, operation_label: label,
            result: sr, state_before: state_before.clone(), state_after: self.take_snapshot(),
        }))
    }

    pub fn take_snapshot(&self) -> StoreSnapshot {
        StoreSnapshot::capture(&self.store, TransactionId::new(u64::MAX))
    }

    pub fn get_history(&self) -> TransactionHistory { self.history.clone() }
}

impl Default for ReplayEngine { fn default() -> Self { Self::new() } }

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn op(id: u64, txn: u64, kind: OpKind) -> Operation {
        Operation { id: OperationId::new(id), txn_id: TransactionId::new(txn), kind, timestamp: id }
    }
    fn begin(iso: IsolationLevel) -> OpKind {
        OpKind::Begin(BeginOp { isolation_level: iso, read_only: false })
    }
    fn w(tbl: u64, itm: u64, val: i64) -> OpKind {
        OpKind::Write(WriteOp { table: TableId::new(tbl), item: ItemId::new(itm),
            columns: vec!["x".into()], old_value: None, new_value: Value::Integer(val), version_written: None })
    }
    fn r(tbl: u64, itm: u64) -> OpKind {
        OpKind::Read(ReadOp { table: TableId::new(tbl), item: ItemId::new(itm),
            columns: vec!["x".into()], value_read: None, version_read: None })
    }
    fn commit(ts: u64) -> OpKind { OpKind::Commit(CommitOp { commit_timestamp: ts }) }
    fn abort(reason: Option<&str>) -> OpKind { OpKind::Abort(AbortOp { reason: reason.map(String::from) }) }

    #[test]
    fn test_serial_schedule() {
        let mut s = Schedule::new();
        let t1 = TransactionId::new(1);
        s.add_step(t1, op(1, 1, begin(IsolationLevel::Serializable)));
        s.add_step(t1, op(2, 1, w(1, 1, 10)));
        s.add_step(t1, op(3, 1, commit(100)));
        let res = ReplayEngine::new().replay_schedule(&s).unwrap();
        assert_eq!(res.steps.len(), 3);
        assert_eq!(res.committed_txns.len(), 1);
        assert!(res.conflicts.is_empty());
        assert_eq!(res.final_state.items.get(&(TableId::new(1), ItemId::new(1))), Some(&Value::Integer(10)));
    }

    #[test]
    fn test_interleaved_schedule() {
        let (mut s, t1, t2) = (Schedule::new(), TransactionId::new(1), TransactionId::new(2));
        s.add_step(t1, op(1, 1, begin(IsolationLevel::ReadCommitted)));
        s.add_step(t1, op(2, 1, w(1, 1, 10)));
        s.add_step(t2, op(3, 2, begin(IsolationLevel::ReadCommitted)));
        s.add_step(t2, op(4, 2, w(1, 2, 20)));
        s.add_step(t1, op(5, 1, commit(200)));
        s.add_step(t2, op(6, 2, commit(201)));
        let res = ReplayEngine::new().replay_schedule(&s).unwrap();
        assert_eq!(res.committed_txns.len(), 2);
        assert!(res.conflicts.is_empty());
        assert_eq!(res.final_state.items.get(&(TableId::new(1), ItemId::new(1))), Some(&Value::Integer(10)));
        assert_eq!(res.final_state.items.get(&(TableId::new(1), ItemId::new(2))), Some(&Value::Integer(20)));
    }

    #[test]
    fn test_commit_and_abort() {
        let (mut s, t1, t2) = (Schedule::new(), TransactionId::new(1), TransactionId::new(2));
        s.add_step(t1, op(1, 1, begin(IsolationLevel::Serializable)));
        s.add_step(t1, op(2, 1, w(1, 1, 42)));
        s.add_step(t1, op(3, 1, commit(100)));
        s.add_step(t2, op(4, 2, begin(IsolationLevel::Serializable)));
        s.add_step(t2, op(5, 2, w(1, 1, 99)));
        s.add_step(t2, op(6, 2, abort(Some("rollback"))));
        let res = ReplayEngine::new().replay_schedule(&s).unwrap();
        assert_eq!(res.committed_txns, vec![t1]);
        assert_eq!(res.aborted_txns, vec![t2]);
        assert_eq!(res.final_state.items.get(&(TableId::new(1), ItemId::new(1))), Some(&Value::Integer(42)));
    }

    #[test]
    fn test_write_write_conflict() {
        let (mut s, t1, t2) = (Schedule::new(), TransactionId::new(1), TransactionId::new(2));
        s.add_step(t1, op(1, 1, begin(IsolationLevel::Serializable)));
        s.add_step(t1, op(2, 1, w(1, 1, 10)));
        s.add_step(t2, op(3, 2, begin(IsolationLevel::Serializable)));
        s.add_step(t2, op(4, 2, w(1, 1, 20)));
        let res = ReplayEngine::new().replay_schedule(&s).unwrap();
        assert!(!res.conflicts.is_empty());
        assert!(matches!(res.steps[3].result, StepResult::Conflict(_)));
    }

    #[test]
    fn test_read_returns_written_value() {
        let mut s = Schedule::new();
        let t1 = TransactionId::new(1);
        s.add_step(t1, op(1, 1, begin(IsolationLevel::Serializable)));
        s.add_step(t1, op(2, 1, w(1, 1, 55)));
        s.add_step(t1, op(3, 1, r(1, 1)));
        s.add_step(t1, op(4, 1, commit(300)));
        let res = ReplayEngine::new().replay_schedule(&s).unwrap();
        match &res.steps[2].result {
            StepResult::ReadValue(v) => assert_eq!(*v, Value::Integer(55)),
            other => panic!("expected ReadValue, got {:?}", other),
        }
    }

    #[test]
    fn test_read_uncommitted_sees_dirty() {
        let (mut s, t1, t2) = (Schedule::new(), TransactionId::new(1), TransactionId::new(2));
        s.add_step(t1, op(1, 1, begin(IsolationLevel::Serializable)));
        s.add_step(t1, op(2, 1, w(1, 1, 77)));
        s.add_step(t2, op(3, 2, begin(IsolationLevel::ReadUncommitted)));
        s.add_step(t2, op(4, 2, r(1, 1)));
        s.add_step(t1, op(5, 1, commit(400)));
        s.add_step(t2, op(6, 2, commit(401)));
        let res = ReplayEngine::new().replay_schedule(&s).unwrap();
        match &res.steps[3].result {
            StepResult::ReadValue(v) => assert_eq!(*v, Value::Integer(77)),
            other => panic!("expected ReadValue(77), got {:?}", other),
        }
    }

    #[test]
    fn test_read_committed_hides_dirty() {
        let (mut s, t1, t2) = (Schedule::new(), TransactionId::new(1), TransactionId::new(2));
        s.add_step(t1, op(1, 1, begin(IsolationLevel::Serializable)));
        s.add_step(t1, op(2, 1, w(1, 1, 77)));
        s.add_step(t2, op(3, 2, begin(IsolationLevel::ReadCommitted)));
        s.add_step(t2, op(4, 2, r(1, 1)));
        s.add_step(t1, op(5, 1, commit(400)));
        s.add_step(t2, op(6, 2, commit(401)));
        let res = ReplayEngine::new().replay_schedule(&s).unwrap();
        match &res.steps[3].result {
            StepResult::ReadValue(v) => assert_eq!(*v, Value::Null),
            other => panic!("expected ReadValue(Null), got {:?}", other),
        }
    }

    #[test]
    fn test_datastore_insert_and_read() {
        let mut store = DataStore::new();
        let (t, tbl, itm) = (TransactionId::new(1), TableId::new(0), ItemId::new(0));
        assert!(matches!(store.insert(tbl, itm, Value::Text("hello".into()), t), InsertResult::Ok { .. }));
        assert!(matches!(store.read(tbl, itm, t, IsolationLevel::Serializable), ReadResult::Found { .. }));
        store.commit(t);
        match store.read(tbl, itm, TransactionId::new(2), IsolationLevel::ReadCommitted) {
            ReadResult::Found { value, .. } => assert_eq!(value, Value::Text("hello".into())),
            _ => panic!("expected Found"),
        }
    }

    #[test]
    fn test_datastore_abort_removes_versions() {
        let mut store = DataStore::new();
        let (t, tbl, itm) = (TransactionId::new(1), TableId::new(0), ItemId::new(0));
        store.write(tbl, itm, Value::Integer(100), t);
        store.abort(t);
        assert!(matches!(store.read(tbl, itm, TransactionId::new(2), IsolationLevel::ReadCommitted), ReadResult::NotFound));
    }
}
