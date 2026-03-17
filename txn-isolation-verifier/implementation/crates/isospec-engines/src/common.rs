//! Common engine state structures shared across engine implementations.
//!
//! Provides MVCC version-store helpers, lock-table conflict detection,
//! dependency tracking, and timestamp management.

use isospec_types::dependency::{Dependency, DependencyType};
use isospec_types::identifier::*;
use isospec_types::isolation::IsolationLevel;
use isospec_types::operation::{LockMode, OpKind, Operation};
use isospec_types::snapshot::Snapshot;
use isospec_types::transaction::TransactionStatus;
use isospec_types::value::Value;
use isospec_types::version::VersionStore;
use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Engine timestamp
// ---------------------------------------------------------------------------

/// Monotonically increasing logical clock shared by an engine instance.
#[derive(Debug, Clone)]
pub struct EngineTimestamp {
    current: u64,
}

impl EngineTimestamp {
    pub fn new() -> Self {
        Self { current: 0 }
    }

    pub fn tick(&mut self) -> u64 {
        self.current += 1;
        self.current
    }

    pub fn current(&self) -> u64 {
        self.current
    }

    pub fn advance_to(&mut self, ts: u64) {
        if ts > self.current {
            self.current = ts;
        }
    }
}

impl Default for EngineTimestamp {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Transaction state (per-txn metadata kept by engines)
// ---------------------------------------------------------------------------

/// Engine-local view of a transaction's lifecycle.
#[derive(Debug, Clone)]
pub struct TransactionState {
    pub txn_id: TransactionId,
    pub status: TransactionStatus,
    pub isolation_level: IsolationLevel,
    pub start_ts: u64,
    pub commit_ts: Option<u64>,
    pub read_only: bool,
    pub read_set: Vec<ReadEntry>,
    pub write_set: Vec<WriteEntry>,
}

/// One item that was read by a transaction.
#[derive(Debug, Clone)]
pub struct ReadEntry {
    pub table: TableId,
    pub item: ItemId,
    pub version_read: Option<VersionId>,
    pub op_id: OperationId,
    pub timestamp: u64,
}

/// One item that was written by a transaction.
#[derive(Debug, Clone)]
pub struct WriteEntry {
    pub table: TableId,
    pub item: ItemId,
    pub version_written: Option<VersionId>,
    pub op_id: OperationId,
    pub old_value: Option<Value>,
    pub new_value: Value,
    pub timestamp: u64,
}

impl TransactionState {
    pub fn new(txn_id: TransactionId, level: IsolationLevel, start_ts: u64) -> Self {
        Self {
            txn_id,
            status: TransactionStatus::Active,
            isolation_level: level,
            start_ts,
            commit_ts: None,
            read_only: false,
            read_set: Vec::new(),
            write_set: Vec::new(),
        }
    }

    pub fn is_active(&self) -> bool {
        self.status.is_active()
    }

    pub fn is_committed(&self) -> bool {
        self.status == TransactionStatus::Committed
    }

    pub fn record_read(
        &mut self,
        table: TableId,
        item: ItemId,
        version: Option<VersionId>,
        op_id: OperationId,
        ts: u64,
    ) {
        self.read_set.push(ReadEntry {
            table,
            item,
            version_read: version,
            op_id,
            timestamp: ts,
        });
    }

    pub fn record_write(
        &mut self,
        table: TableId,
        item: ItemId,
        version: Option<VersionId>,
        op_id: OperationId,
        old_value: Option<Value>,
        new_value: Value,
        ts: u64,
    ) {
        self.write_set.push(WriteEntry {
            table,
            item,
            version_written: version,
            op_id,
            old_value,
            new_value,
            timestamp: ts,
        });
    }

    pub fn wrote_item(&self, table: TableId, item: ItemId) -> bool {
        self.write_set.iter().any(|w| w.table == table && w.item == item)
    }

    pub fn read_item(&self, table: TableId, item: ItemId) -> bool {
        self.read_set.iter().any(|r| r.table == table && r.item == item)
    }
}

// ---------------------------------------------------------------------------
// Version-store common helpers (wraps isospec_types::version::VersionStore)
// ---------------------------------------------------------------------------

/// Extended MVCC helpers on top of the core `VersionStore`.
#[derive(Debug, Clone, Default)]
pub struct VersionStoreCommon {
    pub store: VersionStore,
}

impl VersionStoreCommon {
    pub fn new() -> Self {
        Self {
            store: VersionStore::new(),
        }
    }

    /// Write a new version; returns the `VersionId`.
    pub fn write_version(
        &mut self,
        item: ItemId,
        table: TableId,
        txn: TransactionId,
        ts: u64,
        value: Value,
    ) -> VersionId {
        self.store.create_version(item, table, txn, ts, value)
    }

    /// Read the version visible under `snapshot_time` for committed txns.
    pub fn read_visible(
        &self,
        item: ItemId,
        table: TableId,
        snapshot_time: u64,
        committed: &[TransactionId],
    ) -> Option<(VersionId, Value)> {
        self.store
            .visible_version(item, table, snapshot_time, committed)
            .map(|e| (e.version_id, e.value.clone()))
    }

    /// Check if `txn` wrote to `(table, item)` after `since_ts`.
    pub fn has_write_conflict(
        &self,
        table: TableId,
        item: ItemId,
        since_ts: u64,
        exclude_txn: TransactionId,
        committed: &[TransactionId],
    ) -> Option<TransactionId> {
        let versions = self.store.all_versions(item, table);
        for v in versions.iter().rev() {
            if v.created_at > since_ts
                && v.created_by != exclude_txn
                && committed.contains(&v.created_by)
            {
                return Some(v.created_by);
            }
        }
        None
    }

    /// Retrieve writer of the latest version for an item.
    pub fn latest_writer(
        &self,
        table: TableId,
        item: ItemId,
    ) -> Option<TransactionId> {
        let versions = self.store.all_versions(item, table);
        versions.last().map(|v| v.created_by)
    }

    pub fn version_count(&self) -> usize {
        self.store.version_count()
    }
}

// ---------------------------------------------------------------------------
// Lock-table common (with conflict detection)
// ---------------------------------------------------------------------------

/// A lock entry with additional engine-specific tag.
#[derive(Debug, Clone)]
pub struct EngineLockEntry {
    pub lock_id: LockId,
    pub txn_id: TransactionId,
    pub table: TableId,
    pub item: Option<ItemId>,
    pub mode: LockMode,
    pub granted: bool,
    pub timestamp: u64,
    pub tag: LockTag,
}

/// Engine-specific lock classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LockTag {
    Regular,
    SIRead,
    Gap,
    NextKey,
    KeyRange,
    InsertIntention,
}

/// Lock table with configurable compatibility and conflict detection.
#[derive(Debug, Clone, Default)]
pub struct LockTableCommon {
    pub entries: Vec<EngineLockEntry>,
    pub wait_for: HashMap<TransactionId, HashSet<TransactionId>>,
    next_id: u64,
}

impl LockTableCommon {
    pub fn new() -> Self {
        Self::default()
    }

    fn alloc_id(&mut self) -> LockId {
        let id = LockId::new(self.next_id);
        self.next_id += 1;
        id
    }

    /// Try to acquire a lock; returns Ok(LockId) on grant, Err(blocker) on conflict.
    pub fn acquire(
        &mut self,
        txn_id: TransactionId,
        table: TableId,
        item: Option<ItemId>,
        mode: LockMode,
        tag: LockTag,
        ts: u64,
    ) -> Result<LockId, TransactionId> {
        // Check for conflicting held locks
        for entry in &self.entries {
            if !entry.granted {
                continue;
            }
            if entry.txn_id == txn_id {
                continue;
            }
            if entry.table != table {
                continue;
            }
            if item.is_some() && entry.item.is_some() && entry.item != item {
                continue;
            }
            // SIRead locks never block
            if tag == LockTag::SIRead || entry.tag == LockTag::SIRead {
                continue;
            }
            if !entry.mode.is_compatible_with(mode) {
                self.wait_for
                    .entry(txn_id)
                    .or_default()
                    .insert(entry.txn_id);
                return Err(entry.txn_id);
            }
        }
        let lid = self.alloc_id();
        self.entries.push(EngineLockEntry {
            lock_id: lid,
            txn_id,
            table,
            item,
            mode,
            granted: true,
            timestamp: ts,
            tag,
        });
        Ok(lid)
    }

    /// Release all locks held by a transaction.
    pub fn release_all(&mut self, txn_id: TransactionId) {
        self.entries.retain(|e| e.txn_id != txn_id);
        self.wait_for.remove(&txn_id);
        for waiters in self.wait_for.values_mut() {
            waiters.remove(&txn_id);
        }
    }

    /// Returns all lock entries held by `txn_id`.
    pub fn held_by(&self, txn_id: TransactionId) -> Vec<&EngineLockEntry> {
        self.entries
            .iter()
            .filter(|e| e.txn_id == txn_id && e.granted)
            .collect()
    }

    /// Returns true if `txn_id` already holds a lock with the given mode or
    /// stronger on `(table, item)`.
    pub fn already_holds(
        &self,
        txn_id: TransactionId,
        table: TableId,
        item: Option<ItemId>,
        mode: LockMode,
    ) -> bool {
        self.entries.iter().any(|e| {
            e.txn_id == txn_id
                && e.granted
                && e.table == table
                && e.item == item
                && e.mode.strength() >= mode.strength()
        })
    }

    /// Simple deadlock check (cycle in wait-for graph).
    pub fn has_deadlock(&self) -> Option<Vec<TransactionId>> {
        for &start in self.wait_for.keys() {
            let mut visited = HashSet::new();
            let mut stack = vec![start];
            visited.insert(start);
            while let Some(cur) = stack.pop() {
                if let Some(targets) = self.wait_for.get(&cur) {
                    for &t in targets {
                        if t == start {
                            let mut cycle: Vec<TransactionId> = visited.iter().copied().collect();
                            cycle.sort();
                            return Some(cycle);
                        }
                        if visited.insert(t) {
                            stack.push(t);
                        }
                    }
                }
            }
        }
        None
    }

    pub fn lock_count(&self) -> usize {
        self.entries.len()
    }
}

// ---------------------------------------------------------------------------
// Dependency tracker
// ---------------------------------------------------------------------------

/// Tracks rw/ww/wr dependencies between transactions.
#[derive(Debug, Clone, Default)]
pub struct DependencyTracker {
    deps: Vec<Dependency>,
    seen: HashSet<(TransactionId, TransactionId, DependencyType)>,
}

impl DependencyTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a dependency if it hasn't been seen before.
    pub fn add(
        &mut self,
        from: TransactionId,
        to: TransactionId,
        dep_type: DependencyType,
        from_op: Option<OperationId>,
        to_op: Option<OperationId>,
        item: Option<ItemId>,
        table: Option<TableId>,
    ) {
        if from == to {
            return;
        }
        let key = (from, to, dep_type);
        if self.seen.contains(&key) {
            return;
        }
        self.seen.insert(key);
        let mut dep = Dependency::new(from, to, dep_type);
        if let (Some(fo), Some(to_o)) = (from_op, to_op) {
            dep = dep.with_ops(fo, to_o);
        }
        if let (Some(i), Some(t)) = (item, table) {
            dep = dep.with_item(i, t);
        }
        self.deps.push(dep);
    }

    /// Convenience: record a write-write dependency.
    pub fn add_ww(&mut self, from: TransactionId, to: TransactionId) {
        self.add(from, to, DependencyType::WriteWrite, None, None, None, None);
    }

    /// Convenience: record a write-read dependency.
    pub fn add_wr(&mut self, from: TransactionId, to: TransactionId) {
        self.add(from, to, DependencyType::WriteRead, None, None, None, None);
    }

    /// Convenience: record a read-write (anti-) dependency.
    pub fn add_rw(&mut self, from: TransactionId, to: TransactionId) {
        self.add(from, to, DependencyType::ReadWrite, None, None, None, None);
    }

    pub fn all(&self) -> &[Dependency] {
        &self.deps
    }

    pub fn into_vec(self) -> Vec<Dependency> {
        self.deps
    }

    pub fn clear(&mut self) {
        self.deps.clear();
        self.seen.clear();
    }

    /// Collect rw-dependencies outgoing from `txn`.
    pub fn rw_from(&self, txn: TransactionId) -> Vec<&Dependency> {
        self.deps
            .iter()
            .filter(|d| d.from_txn == txn && d.dep_type.is_anti_dependency())
            .collect()
    }

    /// Collect rw-dependencies incoming to `txn`.
    pub fn rw_to(&self, txn: TransactionId) -> Vec<&Dependency> {
        self.deps
            .iter()
            .filter(|d| d.to_txn == txn && d.dep_type.is_anti_dependency())
            .collect()
    }

    pub fn has_dependency(
        &self,
        from: TransactionId,
        to: TransactionId,
        dep_type: DependencyType,
    ) -> bool {
        self.seen.contains(&(from, to, dep_type))
    }

    pub fn count(&self) -> usize {
        self.deps.len()
    }
}

// ---------------------------------------------------------------------------
// Helpers: snapshot-based committed list
// ---------------------------------------------------------------------------

/// Build the list of committed transaction IDs from a map of states.
pub fn committed_txn_list(txns: &HashMap<TransactionId, TransactionState>) -> Vec<TransactionId> {
    txns.iter()
        .filter(|(_, st)| st.status == TransactionStatus::Committed)
        .map(|(&id, _)| id)
        .collect()
}

/// Build the set of active transaction IDs.
pub fn active_txn_set(txns: &HashMap<TransactionId, TransactionState>) -> HashSet<TransactionId> {
    txns.iter()
        .filter(|(_, st)| st.is_active())
        .map(|(&id, _)| id)
        .collect()
}

/// Detect all ww/wr/rw dependencies among a set of transaction states.
pub fn detect_dependencies(
    txns: &HashMap<TransactionId, TransactionState>,
    tracker: &mut DependencyTracker,
) {
    let ids: Vec<TransactionId> = txns.keys().copied().collect();
    for &t1 in &ids {
        for &t2 in &ids {
            if t1 == t2 {
                continue;
            }
            let s1 = &txns[&t1];
            let s2 = &txns[&t2];
            // ww: t1 wrote X, t2 wrote X, t1 committed before t2
            for w1 in &s1.write_set {
                for w2 in &s2.write_set {
                    if w1.table == w2.table && w1.item == w2.item && w1.timestamp < w2.timestamp {
                        tracker.add(
                            t1,
                            t2,
                            DependencyType::WriteWrite,
                            Some(w1.op_id),
                            Some(w2.op_id),
                            Some(w1.item),
                            Some(w1.table),
                        );
                    }
                }
                // wr: t1 wrote X, t2 read X written by t1
                for r2 in &s2.read_set {
                    if w1.table == r2.table
                        && w1.item == r2.item
                        && w1.version_written.is_some()
                        && w1.version_written == r2.version_read
                    {
                        tracker.add(
                            t1,
                            t2,
                            DependencyType::WriteRead,
                            Some(w1.op_id),
                            Some(r2.op_id),
                            Some(w1.item),
                            Some(w1.table),
                        );
                    }
                }
            }
            // rw: t1 read X, t2 wrote a new version of X after t1's read
            for r1 in &s1.read_set {
                for w2 in &s2.write_set {
                    if r1.table == w2.table
                        && r1.item == w2.item
                        && r1.timestamp < w2.timestamp
                    {
                        tracker.add(
                            t1,
                            t2,
                            DependencyType::ReadWrite,
                            Some(r1.op_id),
                            Some(w2.op_id),
                            Some(r1.item),
                            Some(r1.table),
                        );
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_timestamp() {
        let mut ts = EngineTimestamp::new();
        assert_eq!(ts.current(), 0);
        assert_eq!(ts.tick(), 1);
        assert_eq!(ts.tick(), 2);
        ts.advance_to(10);
        assert_eq!(ts.current(), 10);
        ts.advance_to(5); // should not go backward
        assert_eq!(ts.current(), 10);
    }

    #[test]
    fn test_transaction_state_read_write_tracking() {
        let mut st = TransactionState::new(
            TransactionId::new(1),
            IsolationLevel::Serializable,
            1,
        );
        st.record_read(TableId::new(0), ItemId::new(1), None, OperationId::new(0), 2);
        st.record_write(
            TableId::new(0),
            ItemId::new(1),
            None,
            OperationId::new(1),
            None,
            Value::Integer(42),
            3,
        );
        assert!(st.read_item(TableId::new(0), ItemId::new(1)));
        assert!(st.wrote_item(TableId::new(0), ItemId::new(1)));
        assert!(!st.read_item(TableId::new(0), ItemId::new(99)));
    }

    #[test]
    fn test_version_store_common_write_read() {
        let mut vs = VersionStoreCommon::new();
        let t1 = TransactionId::new(1);
        let vid = vs.write_version(ItemId::new(1), TableId::new(0), t1, 10, Value::Integer(100));
        assert_eq!(vs.version_count(), 1);
        let read = vs.read_visible(ItemId::new(1), TableId::new(0), 15, &[t1]);
        assert!(read.is_some());
        let (rv, val) = read.unwrap();
        assert_eq!(rv, vid);
        assert_eq!(val, Value::Integer(100));
    }

    #[test]
    fn test_version_store_write_conflict() {
        let mut vs = VersionStoreCommon::new();
        let t1 = TransactionId::new(1);
        let t2 = TransactionId::new(2);
        vs.write_version(ItemId::new(1), TableId::new(0), t1, 10, Value::Integer(100));
        vs.write_version(ItemId::new(1), TableId::new(0), t2, 20, Value::Integer(200));
        let conflict =
            vs.has_write_conflict(TableId::new(0), ItemId::new(1), 5, t1, &[t1, t2]);
        assert!(conflict.is_some());
        // t2 wrote after ts=5 and is not t1
        assert_eq!(conflict.unwrap(), t2);
    }

    #[test]
    fn test_lock_table_common_acquire_release() {
        let mut lt = LockTableCommon::new();
        let t1 = TransactionId::new(1);
        let res = lt.acquire(
            t1,
            TableId::new(0),
            Some(ItemId::new(1)),
            LockMode::Shared,
            LockTag::Regular,
            1,
        );
        assert!(res.is_ok());
        assert_eq!(lt.lock_count(), 1);
        lt.release_all(t1);
        assert_eq!(lt.lock_count(), 0);
    }

    #[test]
    fn test_lock_table_common_conflict() {
        let mut lt = LockTableCommon::new();
        let t1 = TransactionId::new(1);
        let t2 = TransactionId::new(2);
        lt.acquire(t1, TableId::new(0), Some(ItemId::new(1)), LockMode::Exclusive, LockTag::Regular, 1)
            .unwrap();
        let res = lt.acquire(
            t2,
            TableId::new(0),
            Some(ItemId::new(1)),
            LockMode::Shared,
            LockTag::Regular,
            2,
        );
        assert!(res.is_err());
        assert_eq!(res.unwrap_err(), t1);
    }

    #[test]
    fn test_lock_table_siread_no_block() {
        let mut lt = LockTableCommon::new();
        let t1 = TransactionId::new(1);
        let t2 = TransactionId::new(2);
        lt.acquire(t1, TableId::new(0), Some(ItemId::new(1)), LockMode::SIRead, LockTag::SIRead, 1)
            .unwrap();
        let res = lt.acquire(
            t2,
            TableId::new(0),
            Some(ItemId::new(1)),
            LockMode::Exclusive,
            LockTag::Regular,
            2,
        );
        assert!(res.is_ok());
    }

    #[test]
    fn test_lock_table_deadlock() {
        let mut lt = LockTableCommon::new();
        let t1 = TransactionId::new(1);
        let t2 = TransactionId::new(2);
        lt.wait_for.entry(t1).or_default().insert(t2);
        lt.wait_for.entry(t2).or_default().insert(t1);
        assert!(lt.has_deadlock().is_some());
    }

    #[test]
    fn test_dependency_tracker() {
        let mut dt = DependencyTracker::new();
        let t1 = TransactionId::new(1);
        let t2 = TransactionId::new(2);
        dt.add_ww(t1, t2);
        dt.add_wr(t1, t2);
        dt.add_rw(t2, t1);
        assert_eq!(dt.count(), 3);
        // Duplicate should be ignored
        dt.add_ww(t1, t2);
        assert_eq!(dt.count(), 3);
        // Self-dep ignored
        dt.add_ww(t1, t1);
        assert_eq!(dt.count(), 3);
        assert!(dt.has_dependency(t1, t2, DependencyType::WriteWrite));
        assert!(!dt.rw_from(t2).is_empty());
    }

    #[test]
    fn test_detect_dependencies() {
        let mut txns = HashMap::new();
        let t1 = TransactionId::new(1);
        let t2 = TransactionId::new(2);
        let mut s1 = TransactionState::new(t1, IsolationLevel::Serializable, 1);
        let vid = VersionId::new(0);
        s1.record_read(TableId::new(0), ItemId::new(1), Some(vid), OperationId::new(0), 2);
        s1.status = TransactionStatus::Committed;
        s1.commit_ts = Some(5);
        let mut s2 = TransactionState::new(t2, IsolationLevel::Serializable, 3);
        s2.record_write(
            TableId::new(0),
            ItemId::new(1),
            Some(VersionId::new(1)),
            OperationId::new(1),
            None,
            Value::Integer(99),
            4,
        );
        s2.status = TransactionStatus::Committed;
        s2.commit_ts = Some(6);
        txns.insert(t1, s1);
        txns.insert(t2, s2);
        let mut tracker = DependencyTracker::new();
        detect_dependencies(&txns, &mut tracker);
        assert!(tracker.has_dependency(t1, t2, DependencyType::ReadWrite));
    }

    #[test]
    fn test_committed_and_active_helpers() {
        let mut txns = HashMap::new();
        let t1 = TransactionId::new(1);
        let t2 = TransactionId::new(2);
        let mut s1 = TransactionState::new(t1, IsolationLevel::ReadCommitted, 1);
        s1.status = TransactionStatus::Committed;
        let s2 = TransactionState::new(t2, IsolationLevel::ReadCommitted, 2);
        txns.insert(t1, s1);
        txns.insert(t2, s2);
        let committed = committed_txn_list(&txns);
        assert_eq!(committed.len(), 1);
        assert!(committed.contains(&t1));
        let active = active_txn_set(&txns);
        assert_eq!(active.len(), 1);
        assert!(active.contains(&t2));
    }
}
