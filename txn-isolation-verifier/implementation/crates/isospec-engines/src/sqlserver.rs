//! SQL Server 2022 dual-mode engine model.
//!
//! State: PessimisticState | OptimisticState selected by READ_COMMITTED_SNAPSHOT flag.
//! - Pessimistic: (LockTable, KeyRangeLocks, EscalationCounters)
//! - Optimistic: (VersionStore_tempdb, TxnSnapshots, ConflictDetection)
//! - Key-range locks: RangeS-S, RangeS-U, RangeI-N, RangeX-X
//! - Lock escalation from row → page → table

use crate::common::*;
use isospec_core::engine_traits::*;
use isospec_types::config::EngineKind;
use isospec_types::constraint::*;
use isospec_types::dependency::*;
use isospec_types::error::*;
use isospec_types::identifier::*;
use isospec_types::isolation::IsolationLevel;
use isospec_types::operation::*;
use isospec_types::snapshot::{Snapshot, SnapshotManager};
use isospec_types::transaction::TransactionStatus;
use isospec_types::value::Value;
use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Key-range lock manager
// ---------------------------------------------------------------------------

/// A key-range lock entry.
#[derive(Debug, Clone)]
pub struct KeyRangeLockEntry {
    pub txn_id: TransactionId,
    pub table: TableId,
    pub item: ItemId,
    pub mode: KeyRangeLockMode,
    pub timestamp: u64,
}

/// SQL Server key-range lock modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KeyRangeLockMode {
    /// RangeS-S: shared range, shared key
    RangeSS,
    /// RangeS-U: shared range, update key
    RangeSU,
    /// RangeI-N: insert range, null key (insert into range)
    RangeIN,
    /// RangeX-X: exclusive range, exclusive key
    RangeXX,
}

impl KeyRangeLockMode {
    /// Compatibility matrix for key-range locks.
    pub fn is_compatible_with(self, other: KeyRangeLockMode) -> bool {
        use KeyRangeLockMode::*;
        match (self, other) {
            (RangeSS, RangeSS) => true,
            (RangeSS, RangeSU) => true,
            (RangeSU, RangeSS) => true,
            (RangeIN, RangeIN) => true,
            (RangeIN, RangeSS) => true,
            (RangeSS, RangeIN) => true,
            // All other combinations conflict
            _ => false,
        }
    }

    /// Map to the isospec_types LockMode equivalent.
    pub fn to_lock_mode(self) -> LockMode {
        match self {
            Self::RangeSS => LockMode::RangeSharedShared,
            Self::RangeSU => LockMode::RangeSharedUpdate,
            Self::RangeIN => LockMode::RangeInsertNull,
            Self::RangeXX => LockMode::RangeExclusiveExclusive,
        }
    }
}

/// Manages key-range locks for SQL Server SERIALIZABLE.
#[derive(Debug, Clone, Default)]
pub struct KeyRangeLockManager {
    entries: Vec<KeyRangeLockEntry>,
}

impl KeyRangeLockManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// Try to acquire a key-range lock.
    pub fn acquire(
        &mut self,
        txn_id: TransactionId,
        table: TableId,
        item: ItemId,
        mode: KeyRangeLockMode,
        ts: u64,
    ) -> Result<(), TransactionId> {
        for entry in &self.entries {
            if entry.txn_id == txn_id {
                continue;
            }
            if entry.table != table || entry.item != item {
                continue;
            }
            if !mode.is_compatible_with(entry.mode) {
                return Err(entry.txn_id);
            }
        }
        self.entries.push(KeyRangeLockEntry {
            txn_id,
            table,
            item,
            mode,
            timestamp: ts,
        });
        Ok(())
    }

    pub fn release_all(&mut self, txn_id: TransactionId) {
        self.entries.retain(|e| e.txn_id != txn_id);
    }

    pub fn count(&self) -> usize {
        self.entries.len()
    }
}

// ---------------------------------------------------------------------------
// Lock escalation
// ---------------------------------------------------------------------------

/// Tracks per-(txn, table) lock counts for escalation decisions.
#[derive(Debug, Clone, Default)]
pub struct EscalationCounters {
    counters: HashMap<(TransactionId, TableId), usize>,
    threshold: usize,
}

impl EscalationCounters {
    pub fn new(threshold: usize) -> Self {
        Self {
            counters: HashMap::new(),
            threshold,
        }
    }

    /// Increment the lock count; returns true if escalation should occur.
    pub fn increment(&mut self, txn_id: TransactionId, table: TableId) -> bool {
        let count = self
            .counters
            .entry((txn_id, table))
            .or_insert(0);
        *count += 1;
        *count >= self.threshold
    }

    pub fn reset_txn(&mut self, txn_id: TransactionId) {
        self.counters.retain(|&(t, _), _| t != txn_id);
    }

    pub fn count(&self, txn_id: TransactionId, table: TableId) -> usize {
        self.counters
            .get(&(txn_id, table))
            .copied()
            .unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// Pessimistic mode
// ---------------------------------------------------------------------------

/// Pessimistic-mode state: traditional row/page/table locks with
/// key-range locks at SERIALIZABLE.
#[derive(Debug, Clone)]
pub struct PessimisticMode {
    pub lock_table: LockTableCommon,
    pub key_range_locks: KeyRangeLockManager,
    pub escalation: EscalationCounters,
}

impl PessimisticMode {
    pub fn new(escalation_threshold: usize) -> Self {
        Self {
            lock_table: LockTableCommon::new(),
            key_range_locks: KeyRangeLockManager::new(),
            escalation: EscalationCounters::new(escalation_threshold),
        }
    }

    /// Acquire a row-level lock, potentially escalating to table-level.
    pub fn acquire_row_lock(
        &mut self,
        txn_id: TransactionId,
        table: TableId,
        item: ItemId,
        mode: LockMode,
        ts: u64,
    ) -> Result<LockId, TransactionId> {
        let should_escalate = self.escalation.increment(txn_id, table);
        if should_escalate {
            // Table-level lock
            self.lock_table
                .acquire(txn_id, table, None, mode, LockTag::Regular, ts)
        } else {
            self.lock_table
                .acquire(txn_id, table, Some(item), mode, LockTag::Regular, ts)
        }
    }

    pub fn release_all(&mut self, txn_id: TransactionId) {
        self.lock_table.release_all(txn_id);
        self.key_range_locks.release_all(txn_id);
        self.escalation.reset_txn(txn_id);
    }
}

// ---------------------------------------------------------------------------
// Optimistic mode
// ---------------------------------------------------------------------------

/// Optimistic-mode state: tempdb-based version store with snapshot reads
/// and first-committer-wins conflict detection.
#[derive(Debug, Clone)]
pub struct OptimisticMode {
    pub version_store: VersionStoreCommon,
    pub snapshot_mgr: SnapshotManager,
    pub snapshots: HashMap<TransactionId, Snapshot>,
}

impl OptimisticMode {
    pub fn new() -> Self {
        Self {
            version_store: VersionStoreCommon::new(),
            snapshot_mgr: SnapshotManager::new(),
            snapshots: HashMap::new(),
        }
    }

    pub fn take_snapshot(
        &mut self,
        txn_id: TransactionId,
        active: HashSet<TransactionId>,
        committed: HashSet<TransactionId>,
    ) -> Snapshot {
        let snap = self
            .snapshot_mgr
            .take_snapshot(txn_id, active, committed);
        self.snapshots.insert(txn_id, snap.clone());
        snap
    }

    pub fn snapshot_time(&self, txn_id: TransactionId) -> u64 {
        self.snapshots
            .get(&txn_id)
            .map(|s| s.snapshot_time)
            .unwrap_or(0)
    }

    /// First-committer-wins check for optimistic mode.
    pub fn check_write_conflict(
        &self,
        table: TableId,
        item: ItemId,
        since_ts: u64,
        txn_id: TransactionId,
        committed: &[TransactionId],
    ) -> Option<TransactionId> {
        self.version_store
            .has_write_conflict(table, item, since_ts, txn_id, committed)
    }

    pub fn remove_snapshot(&mut self, txn_id: TransactionId) {
        self.snapshots.remove(&txn_id);
    }
}

impl Default for OptimisticMode {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// SQL Server engine state
// ---------------------------------------------------------------------------

/// Whether we're in pessimistic or optimistic mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SqlServerMode {
    Pessimistic,
    Optimistic,
}

/// Full mutable state for the SQL Server engine.
pub struct SqlServerState {
    pub mode: SqlServerMode,
    pub txn_states: HashMap<TransactionId, TransactionState>,
    /// Shared version store (used in optimistic mode; also tracks versions in
    /// pessimistic mode for dependency analysis).
    pub version_store: VersionStoreCommon,
    pub pessimistic: PessimisticMode,
    pub optimistic: OptimisticMode,
    pub dep_tracker: DependencyTracker,
    pub clock: EngineTimestamp,
    pub committed_set: HashSet<TransactionId>,
    pub isolation_level: IsolationLevel,
    /// Snapshot manager used in pessimistic mode at SNAPSHOT isolation.
    pub snapshot_mgr: SnapshotManager,
    pub snapshots: HashMap<TransactionId, Snapshot>,
}

impl SqlServerState {
    pub fn new(isolation_level: IsolationLevel, rcsi_enabled: bool) -> Self {
        let mode = if rcsi_enabled {
            SqlServerMode::Optimistic
        } else {
            SqlServerMode::Pessimistic
        };
        Self {
            mode,
            txn_states: HashMap::new(),
            version_store: VersionStoreCommon::new(),
            pessimistic: PessimisticMode::new(5000),
            optimistic: OptimisticMode::new(),
            dep_tracker: DependencyTracker::new(),
            clock: EngineTimestamp::new(),
            committed_set: HashSet::new(),
            isolation_level,
            snapshot_mgr: SnapshotManager::new(),
            snapshots: HashMap::new(),
        }
    }

    fn committed_list(&self) -> Vec<TransactionId> {
        self.committed_set.iter().copied().collect()
    }

    fn active_set(&self) -> HashSet<TransactionId> {
        self.txn_states
            .iter()
            .filter(|(_, s)| s.is_active())
            .map(|(&id, _)| id)
            .collect()
    }

    // -- Pessimistic read --
    fn pessimistic_read(
        &mut self,
        txn_id: TransactionId,
        table: TableId,
        item: ItemId,
        op_id: OperationId,
    ) -> IsoSpecResult<OperationOutcome> {
        let ts = self.clock.tick();

        // At RC: S lock acquired and released immediately after read
        // At RR: S lock held until end of transaction
        // At SERIALIZABLE: RangeS-S key-range lock held until end
        let lock_mode = LockMode::Shared;
        let res = self.pessimistic.acquire_row_lock(txn_id, table, item, lock_mode, ts);
        if let Err(blocker) = res {
            return Ok(OperationOutcome::blocked(blocker));
        }

        // At SERIALIZABLE, acquire key-range lock
        if self.isolation_level == IsolationLevel::Serializable {
            if let Err(blocker) = self.pessimistic.key_range_locks.acquire(
                txn_id,
                table,
                item,
                KeyRangeLockMode::RangeSS,
                ts,
            ) {
                return Ok(OperationOutcome::blocked(blocker));
            }
        }

        // Read the latest committed version
        let committed = self.committed_list();
        let snap_time = if self.isolation_level == IsolationLevel::Snapshot {
            self.snapshots
                .get(&txn_id)
                .map(|s| s.snapshot_time)
                .unwrap_or(ts)
        } else {
            ts // pessimistic reads current data
        };

        let visible = self
            .version_store
            .read_visible(item, table, snap_time, &committed);

        // At RC, release the S lock immediately
        if self.isolation_level == IsolationLevel::ReadCommitted {
            // Remove just-acquired S lock for this item
            self.pessimistic.lock_table.entries.retain(|e| {
                !(e.txn_id == txn_id
                    && e.table == table
                    && e.item == Some(item)
                    && e.mode == LockMode::Shared
                    && e.timestamp == ts)
            });
        }

        if let Some((vid, val)) = visible {
            if let Some(st) = self.txn_states.get_mut(&txn_id) {
                st.record_read(table, item, Some(vid), op_id, ts);
            }
            if let Some(writer) = self.version_store.latest_writer(table, item) {
                if writer != txn_id && self.committed_set.contains(&writer) {
                    self.dep_tracker.add_wr(writer, txn_id);
                }
            }
            Ok(OperationOutcome::success().with_value(val))
        } else {
            if let Some(st) = self.txn_states.get_mut(&txn_id) {
                st.record_read(table, item, None, op_id, ts);
            }
            Ok(OperationOutcome::success().with_value(Value::Null))
        }
    }

    // -- Optimistic read --
    fn optimistic_read(
        &mut self,
        txn_id: TransactionId,
        table: TableId,
        item: ItemId,
        op_id: OperationId,
    ) -> IsoSpecResult<OperationOutcome> {
        let ts = self.clock.tick();
        let snap_time = self.optimistic.snapshot_time(txn_id);
        let committed = self.committed_list();

        let visible = self
            .version_store
            .read_visible(item, table, snap_time, &committed);

        if let Some((vid, val)) = visible {
            if let Some(st) = self.txn_states.get_mut(&txn_id) {
                st.record_read(table, item, Some(vid), op_id, ts);
            }
            if let Some(writer) = self.version_store.latest_writer(table, item) {
                if writer != txn_id && self.committed_set.contains(&writer) {
                    self.dep_tracker.add_wr(writer, txn_id);
                }
            }
            Ok(OperationOutcome::success().with_value(val))
        } else {
            if let Some(st) = self.txn_states.get_mut(&txn_id) {
                st.record_read(table, item, None, op_id, ts);
            }
            Ok(OperationOutcome::success().with_value(Value::Null))
        }
    }

    // -- Write (shared between modes) --
    fn handle_write(
        &mut self,
        txn_id: TransactionId,
        table: TableId,
        item: ItemId,
        value: Value,
        op_id: OperationId,
    ) -> IsoSpecResult<OperationOutcome> {
        let ts = self.clock.tick();

        // Both modes acquire X lock on writes
        if self.mode == SqlServerMode::Pessimistic {
            let res = self
                .pessimistic
                .acquire_row_lock(txn_id, table, item, LockMode::Exclusive, ts);
            if let Err(blocker) = res {
                return Ok(OperationOutcome::blocked(blocker));
            }
            // At SERIALIZABLE, acquire RangeX-X
            if self.isolation_level == IsolationLevel::Serializable {
                if let Err(blocker) = self.pessimistic.key_range_locks.acquire(
                    txn_id,
                    table,
                    item,
                    KeyRangeLockMode::RangeXX,
                    ts,
                ) {
                    return Ok(OperationOutcome::blocked(blocker));
                }
            }
        }

        // rw-dependency detection
        for (&other_id, other_st) in &self.txn_states {
            if other_id == txn_id {
                continue;
            }
            if other_st.read_item(table, item) {
                self.dep_tracker.add_rw(other_id, txn_id);
            }
        }

        let vid = self
            .version_store
            .write_version(item, table, txn_id, ts, value.clone());

        if let Some(st) = self.txn_states.get_mut(&txn_id) {
            st.record_write(table, item, Some(vid), op_id, None, value, ts);
        }

        Ok(OperationOutcome::success())
    }

    fn handle_insert(
        &mut self,
        txn_id: TransactionId,
        table: TableId,
        item: ItemId,
        values: &indexmap::IndexMap<String, Value>,
        op_id: OperationId,
    ) -> IsoSpecResult<OperationOutcome> {
        let ts = self.clock.tick();

        // At SERIALIZABLE in pessimistic mode, acquire RangeI-N
        if self.mode == SqlServerMode::Pessimistic
            && self.isolation_level == IsolationLevel::Serializable
        {
            if let Err(blocker) = self.pessimistic.key_range_locks.acquire(
                txn_id,
                table,
                item,
                KeyRangeLockMode::RangeIN,
                ts,
            ) {
                return Ok(OperationOutcome::blocked(blocker));
            }
        }

        let composite_value = if values.len() == 1 {
            values.values().next().cloned().unwrap_or(Value::Null)
        } else {
            Value::Array(values.values().cloned().collect())
        };
        self.handle_write(txn_id, table, item, composite_value, op_id)
    }

    fn handle_predicate_read(
        &mut self,
        txn_id: TransactionId,
        table: TableId,
        items: &[ItemId],
        op_id: OperationId,
    ) -> IsoSpecResult<OperationOutcome> {
        // At SERIALIZABLE, acquire RangeS-S on each key in the range
        if self.mode == SqlServerMode::Pessimistic
            && self.isolation_level == IsolationLevel::Serializable
        {
            let ts = self.clock.current();
            for &item_id in items {
                if let Err(blocker) = self.pessimistic.key_range_locks.acquire(
                    txn_id,
                    table,
                    item_id,
                    KeyRangeLockMode::RangeSS,
                    ts,
                ) {
                    return Ok(OperationOutcome::blocked(blocker));
                }
            }
        }

        let mut outcome = OperationOutcome::success();
        for &item_id in items {
            let sub = if self.mode == SqlServerMode::Pessimistic {
                self.pessimistic_read(txn_id, table, item_id, op_id)?
            } else {
                self.optimistic_read(txn_id, table, item_id, op_id)?
            };
            if sub.blocked {
                return Ok(sub);
            }
            for v in sub.values_read {
                outcome = outcome.with_value(v);
            }
        }
        Ok(outcome)
    }
}

impl EngineState for SqlServerState {
    fn begin_transaction(
        &mut self,
        txn_id: TransactionId,
        level: IsolationLevel,
    ) -> IsoSpecResult<()> {
        let ts = self.clock.tick();
        let st = TransactionState::new(txn_id, level, ts);
        self.txn_states.insert(txn_id, st);

        match self.mode {
            SqlServerMode::Pessimistic => {
                // At SNAPSHOT isolation, take a snapshot
                if level == IsolationLevel::Snapshot {
                    let active = self.active_set();
                    let committed = self.committed_set.clone();
                    let snap = self.snapshot_mgr.take_snapshot(txn_id, active, committed);
                    self.snapshots.insert(txn_id, snap);
                }
            }
            SqlServerMode::Optimistic => {
                let active = self.active_set();
                let committed = self.committed_set.clone();
                self.optimistic.take_snapshot(txn_id, active, committed);
            }
        }
        Ok(())
    }

    fn execute_operation(&mut self, op: &Operation) -> IsoSpecResult<OperationOutcome> {
        let txn_id = op.txn_id;
        if !self.txn_states.contains_key(&txn_id) {
            return Err(IsoSpecError::engine_model(
                "SQL Server",
                format!("Transaction {} not found", txn_id),
            ));
        }
        match &op.kind {
            OpKind::Read(r) => {
                if self.mode == SqlServerMode::Pessimistic {
                    self.pessimistic_read(txn_id, r.table, r.item, op.id)
                } else {
                    self.optimistic_read(txn_id, r.table, r.item, op.id)
                }
            }
            OpKind::Write(w) => {
                self.handle_write(txn_id, w.table, w.item, w.new_value.clone(), op.id)
            }
            OpKind::Insert(i) => self.handle_insert(txn_id, i.table, i.item, &i.values, op.id),
            OpKind::Delete(d) => {
                for &item_id in &d.deleted_items {
                    let res = self.handle_write(txn_id, d.table, item_id, Value::Null, op.id)?;
                    if !res.success {
                        return Ok(res);
                    }
                }
                Ok(OperationOutcome::success())
            }
            OpKind::PredicateRead(pr) => {
                self.handle_predicate_read(txn_id, pr.table, &pr.items_read, op.id)
            }
            OpKind::PredicateWrite(pw) => {
                for &item_id in &pw.items_written {
                    let res =
                        self.handle_write(txn_id, pw.table, item_id, pw.new_value.clone(), op.id)?;
                    if !res.success {
                        return Ok(res);
                    }
                }
                Ok(OperationOutcome::success())
            }
            OpKind::Lock(l) => {
                let ts = self.clock.tick();
                if self.mode == SqlServerMode::Pessimistic {
                    match self.pessimistic.lock_table.acquire(
                        txn_id,
                        l.table,
                        l.item,
                        l.mode,
                        LockTag::Regular,
                        ts,
                    ) {
                        Ok(_) => Ok(OperationOutcome::success()),
                        Err(blocker) => Ok(OperationOutcome::blocked(blocker)),
                    }
                } else {
                    Ok(OperationOutcome::success())
                }
            }
            OpKind::Begin(_) => Ok(OperationOutcome::success()),
            OpKind::Commit(_) => {
                let outcome = self.commit_transaction(txn_id)?;
                if outcome.committed {
                    Ok(OperationOutcome::success())
                } else {
                    Ok(OperationOutcome {
                        success: false,
                        values_read: Vec::new(),
                        locks_acquired: Vec::new(),
                        blocked: false,
                        blocked_by: None,
                        message: outcome.abort_reason,
                    })
                }
            }
            OpKind::Abort(_) => {
                self.abort_transaction(txn_id)?;
                Ok(OperationOutcome::success())
            }
        }
    }

    fn commit_transaction(
        &mut self,
        txn_id: TransactionId,
    ) -> IsoSpecResult<CommitOutcome> {
        let ts = self.clock.tick();

        // Optimistic mode: first-committer-wins validation at SNAPSHOT
        if self.mode == SqlServerMode::Optimistic {
            if self.isolation_level == IsolationLevel::Snapshot
                || self.isolation_level == IsolationLevel::SqlServerRCSI
            {
                let snap_time = self.optimistic.snapshot_time(txn_id);
                let committed = self.committed_list();
                if let Some(st) = self.txn_states.get(&txn_id) {
                    for w in st.write_set.clone() {
                        if let Some(conflict) = self.optimistic.check_write_conflict(
                            w.table,
                            w.item,
                            snap_time,
                            txn_id,
                            &committed,
                        ) {
                            if conflict != txn_id {
                                self.abort_transaction(txn_id)?;
                                return Ok(CommitOutcome::aborted(
                                    "update conflict: another transaction modified the same row",
                                ));
                            }
                        }
                    }
                }
            }
        }

        if let Some(st) = self.txn_states.get_mut(&txn_id) {
            st.status = TransactionStatus::Committed;
            st.commit_ts = Some(ts);
        }
        self.committed_set.insert(txn_id);

        // Release locks (pessimistic) or snapshot (optimistic)
        match self.mode {
            SqlServerMode::Pessimistic => {
                self.pessimistic.release_all(txn_id);
                self.snapshots.remove(&txn_id);
            }
            SqlServerMode::Optimistic => {
                self.optimistic.remove_snapshot(txn_id);
            }
        }

        let deps = self.dep_tracker.all().to_vec();
        Ok(CommitOutcome {
            committed: true,
            abort_reason: None,
            commit_timestamp: ts,
            dependencies: deps,
        })
    }

    fn abort_transaction(&mut self, txn_id: TransactionId) -> IsoSpecResult<()> {
        if let Some(st) = self.txn_states.get_mut(&txn_id) {
            st.status = TransactionStatus::Aborted;
        }
        match self.mode {
            SqlServerMode::Pessimistic => {
                self.pessimistic.release_all(txn_id);
                self.snapshots.remove(&txn_id);
            }
            SqlServerMode::Optimistic => {
                self.optimistic.remove_snapshot(txn_id);
            }
        }
        Ok(())
    }

    fn extract_dependencies(&self) -> Vec<Dependency> {
        self.dep_tracker.all().to_vec()
    }

    fn active_transactions(&self) -> Vec<TransactionId> {
        self.txn_states
            .iter()
            .filter(|(_, s)| s.is_active())
            .map(|(&id, _)| id)
            .collect()
    }

    fn transaction_status(
        &self,
        txn_id: TransactionId,
    ) -> Option<TransactionStatus> {
        self.txn_states.get(&txn_id).map(|s| s.status)
    }

    fn snapshot_info(&self, txn_id: TransactionId) -> Option<SnapshotInfo> {
        // Check optimistic snapshots first, then pessimistic
        let snap = self
            .optimistic
            .snapshots
            .get(&txn_id)
            .or_else(|| self.snapshots.get(&txn_id));
        snap.map(|s| SnapshotInfo {
            snapshot_time: s.snapshot_time,
            active_txns: s.active_txns.iter().copied().collect(),
            committed_before_snapshot: s.committed_txns.iter().copied().collect(),
        })
    }

    fn reset(&mut self) {
        self.txn_states.clear();
        self.version_store = VersionStoreCommon::new();
        self.pessimistic = PessimisticMode::new(5000);
        self.optimistic = OptimisticMode::new();
        self.dep_tracker.clear();
        self.clock = EngineTimestamp::new();
        self.committed_set.clear();
        self.snapshot_mgr = SnapshotManager::new();
        self.snapshots.clear();
    }
}

// ---------------------------------------------------------------------------
// SQL Server engine model
// ---------------------------------------------------------------------------

/// SQL Server 2022 engine model.
pub struct SqlServerModel {
    version: String,
    rcsi_enabled: bool,
    lock_escalation_threshold: usize,
}

impl SqlServerModel {
    pub fn new() -> Self {
        Self {
            version: "2022".to_string(),
            rcsi_enabled: false,
            lock_escalation_threshold: 5000,
        }
    }

    pub fn with_version(mut self, v: impl Into<String>) -> Self {
        self.version = v.into();
        self
    }

    pub fn with_rcsi(mut self, enabled: bool) -> Self {
        self.rcsi_enabled = enabled;
        self
    }

    pub fn with_escalation_threshold(mut self, threshold: usize) -> Self {
        self.lock_escalation_threshold = threshold;
        self
    }

    fn encode_sqlserver_constraints(
        &self,
        txn_count: usize,
        _op_count: usize,
    ) -> SmtConstraintSet {
        let mut cs = SmtConstraintSet::new("QF_LIA");

        for i in 0..txn_count {
            let start = format!("ss_start_{}", i);
            let commit = format!("ss_commit_{}", i);
            cs.declare(&start, SmtSort::Int);
            cs.declare(&commit, SmtSort::Int);
            cs.assert(SmtExpr::lt(
                SmtExpr::int_var(&start),
                SmtExpr::int_var(&commit),
            ));
        }

        if self.rcsi_enabled {
            // Optimistic mode: snapshot variables and conflict detection
            for i in 0..txn_count {
                let snap = format!("ss_snap_{}", i);
                cs.declare(&snap, SmtSort::Int);
                cs.assert(SmtExpr::eq(
                    SmtExpr::int_var(&snap),
                    SmtExpr::int_var(&format!("ss_start_{}", i)),
                ));
            }
            // First-committer-wins: for overlapping txns writing same item,
            // one must abort
            for i in 0..txn_count {
                for j in (i + 1)..txn_count {
                    let conflict = format!("ss_ww_conflict_{}_{}", i, j);
                    let abort_i = format!("ss_abort_{}", i);
                    let abort_j = format!("ss_abort_{}", j);
                    cs.declare(&conflict, SmtSort::Bool);
                    cs.declare(&abort_i, SmtSort::Bool);
                    cs.declare(&abort_j, SmtSort::Bool);
                    // If conflict, one must abort
                    cs.assert(SmtExpr::implies(
                        SmtExpr::bool_var(&conflict),
                        SmtExpr::or(vec![
                            SmtExpr::bool_var(&abort_i),
                            SmtExpr::bool_var(&abort_j),
                        ]),
                    ));
                }
            }
        } else {
            // Pessimistic mode: lock variables
            for i in 0..txn_count {
                for j in 0..txn_count {
                    if i == j {
                        continue;
                    }
                    let lock = format!("ss_lock_{}_{}", i, j);
                    cs.declare(&lock, SmtSort::Bool);
                    let wait = format!("ss_wait_{}_{}", j, i);
                    cs.declare(&wait, SmtSort::Bool);
                    cs.assert(SmtExpr::implies(
                        SmtExpr::bool_var(&lock),
                        SmtExpr::bool_var(&wait),
                    ));
                }
            }

            // Key-range lock constraints at SERIALIZABLE
            for i in 0..txn_count {
                let kr = format!("ss_keyrange_{}", i);
                cs.declare(&kr, SmtSort::Bool);
                for j in 0..txn_count {
                    if i == j {
                        continue;
                    }
                    let blocked = format!("ss_kr_blocked_{}_{}", j, i);
                    cs.declare(&blocked, SmtSort::Bool);
                    cs.assert(SmtExpr::implies(
                        SmtExpr::bool_var(&kr),
                        SmtExpr::bool_var(&blocked),
                    ));
                }
            }

            // Lock escalation
            for i in 0..txn_count {
                let row_count = format!("ss_rowlocks_{}", i);
                let escalated = format!("ss_escalated_{}", i);
                cs.declare(&row_count, SmtSort::Int);
                cs.declare(&escalated, SmtSort::Bool);
                cs.assert(SmtExpr::Iff(
                    Box::new(SmtExpr::bool_var(&escalated)),
                    Box::new(SmtExpr::Ge(
                        Box::new(SmtExpr::int_var(&row_count)),
                        Box::new(SmtExpr::IntLit(self.lock_escalation_threshold as i64)),
                    )),
                ));
            }
        }

        cs
    }
}

impl Default for SqlServerModel {
    fn default() -> Self {
        Self::new()
    }
}

impl EngineModel for SqlServerModel {
    fn engine_kind(&self) -> EngineKind {
        EngineKind::SqlServer
    }

    fn supported_isolation_levels(&self) -> Vec<IsolationLevel> {
        vec![
            IsolationLevel::ReadUncommitted,
            IsolationLevel::ReadCommitted,
            IsolationLevel::RepeatableRead,
            IsolationLevel::Serializable,
            IsolationLevel::Snapshot,
        ]
    }

    fn create_state(&self, isolation_level: IsolationLevel) -> Box<dyn EngineState> {
        let mut state = SqlServerState::new(isolation_level, self.rcsi_enabled);
        state.pessimistic.escalation =
            EscalationCounters::new(self.lock_escalation_threshold);
        Box::new(state)
    }

    fn encode_constraints(
        &self,
        _isolation_level: IsolationLevel,
        txn_count: usize,
        op_count: usize,
    ) -> IsoSpecResult<SmtConstraintSet> {
        Ok(self.encode_sqlserver_constraints(txn_count, op_count))
    }

    fn version_string(&self) -> &str {
        &self.version
    }

    fn validate_schedule(
        &self,
        schedule: &isospec_types::schedule::Schedule,
        level: IsolationLevel,
    ) -> IsoSpecResult<ValidationResult> {
        let mut state = SqlServerState::new(level, self.rcsi_enabled);
        state.pessimistic.escalation =
            EscalationCounters::new(self.lock_escalation_threshold);
        let mut aborted = Vec::new();
        let mut blocked_txns: HashSet<TransactionId> = HashSet::new();

        for &txn_id in &schedule.transaction_order {
            state.begin_transaction(txn_id, level)?;
        }

        for step in &schedule.steps {
            if blocked_txns.contains(&step.txn_id) {
                continue;
            }
            if state
                .txn_states
                .get(&step.txn_id)
                .map_or(true, |s| s.status == TransactionStatus::Aborted)
            {
                continue;
            }
            let outcome = state.execute_operation(&step.operation)?;
            if outcome.blocked {
                blocked_txns.insert(step.txn_id);
            }
            if !outcome.success && !outcome.blocked {
                aborted.push(step.txn_id);
            }
        }

        let deps = state.extract_dependencies();
        let violations: Vec<String> = blocked_txns
            .iter()
            .map(|t| format!("Transaction {} blocked by lock conflict", t))
            .chain(aborted.iter().map(|t| format!("Transaction {} aborted", t)))
            .collect();
        Ok(ValidationResult {
            valid: violations.is_empty(),
            violations,
            schedule_accepted: aborted.is_empty() && blocked_txns.is_empty(),
            transactions_aborted: aborted,
            dependencies_found: deps,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use isospec_types::identifier::*;
    use isospec_types::operation::Operation;
    use isospec_types::value::Value;

    fn tid(n: u64) -> TransactionId { TransactionId::new(n) }
    fn oid(n: u64) -> OperationId { OperationId::new(n) }
    fn tbl() -> TableId { TableId::new(0) }
    fn item(n: u64) -> ItemId { ItemId::new(n) }

    #[test]
    fn test_key_range_lock_compatibility() {
        use KeyRangeLockMode::*;
        assert!(RangeSS.is_compatible_with(RangeSS));
        assert!(RangeSS.is_compatible_with(RangeSU));
        assert!(!RangeXX.is_compatible_with(RangeSS));
        assert!(!RangeXX.is_compatible_with(RangeXX));
        assert!(RangeIN.is_compatible_with(RangeIN));
        assert!(RangeIN.is_compatible_with(RangeSS));
    }

    #[test]
    fn test_key_range_lock_manager() {
        let mut mgr = KeyRangeLockManager::new();
        assert!(mgr.acquire(tid(1), tbl(), item(5), KeyRangeLockMode::RangeSS, 1).is_ok());
        assert!(mgr.acquire(tid(2), tbl(), item(5), KeyRangeLockMode::RangeSS, 2).is_ok());
        assert!(mgr.acquire(tid(3), tbl(), item(5), KeyRangeLockMode::RangeXX, 3).is_err());
        assert_eq!(mgr.count(), 2);
        mgr.release_all(tid(1));
        mgr.release_all(tid(2));
        assert!(mgr.acquire(tid(3), tbl(), item(5), KeyRangeLockMode::RangeXX, 4).is_ok());
    }

    #[test]
    fn test_escalation_counters() {
        let mut ec = EscalationCounters::new(3);
        assert!(!ec.increment(tid(1), tbl()));
        assert!(!ec.increment(tid(1), tbl()));
        assert!(ec.increment(tid(1), tbl())); // hits threshold
        assert_eq!(ec.count(tid(1), tbl()), 3);
        ec.reset_txn(tid(1));
        assert_eq!(ec.count(tid(1), tbl()), 0);
    }

    #[test]
    fn test_pessimistic_mode_basic() {
        let mut state = SqlServerState::new(IsolationLevel::ReadCommitted, false);
        state.begin_transaction(tid(1), IsolationLevel::ReadCommitted).unwrap();

        state.version_store.write_version(item(1), tbl(), tid(0), 0, Value::Integer(42));
        state.committed_set.insert(tid(0));

        let op = Operation::read(oid(0), tid(1), tbl(), item(1), 1);
        let outcome = state.execute_operation(&op).unwrap();
        assert!(outcome.success);
        assert_eq!(outcome.values_read, vec![Value::Integer(42)]);
    }

    #[test]
    fn test_optimistic_mode_basic() {
        let mut state = SqlServerState::new(IsolationLevel::Snapshot, true);
        state.begin_transaction(tid(1), IsolationLevel::Snapshot).unwrap();

        state.version_store.write_version(item(1), tbl(), tid(0), 0, Value::Integer(55));
        state.committed_set.insert(tid(0));

        let op = Operation::read(oid(0), tid(1), tbl(), item(1), 1);
        let outcome = state.execute_operation(&op).unwrap();
        assert!(outcome.success);
    }

    #[test]
    fn test_pessimistic_write_lock_conflict() {
        let mut state = SqlServerState::new(IsolationLevel::RepeatableRead, false);
        state.begin_transaction(tid(1), IsolationLevel::RepeatableRead).unwrap();
        state.begin_transaction(tid(2), IsolationLevel::RepeatableRead).unwrap();

        let w1 = Operation::write(oid(0), tid(1), tbl(), item(1), Value::Integer(10), 1);
        let r1 = state.execute_operation(&w1).unwrap();
        assert!(r1.success);

        let w2 = Operation::write(oid(1), tid(2), tbl(), item(1), Value::Integer(20), 2);
        let r2 = state.execute_operation(&w2).unwrap();
        assert!(r2.blocked);
        assert_eq!(r2.blocked_by, Some(tid(1)));
    }

    #[test]
    fn test_optimistic_write_conflict_at_commit() {
        let mut state = SqlServerState::new(IsolationLevel::Snapshot, true);
        state.begin_transaction(tid(1), IsolationLevel::Snapshot).unwrap();
        state.begin_transaction(tid(2), IsolationLevel::Snapshot).unwrap();

        // Both write to same item
        let w1 = Operation::write(oid(0), tid(1), tbl(), item(1), Value::Integer(10), 1);
        state.execute_operation(&w1).unwrap();
        let c1 = state.commit_transaction(tid(1)).unwrap();
        assert!(c1.committed);

        let w2 = Operation::write(oid(1), tid(2), tbl(), item(1), Value::Integer(20), 2);
        state.execute_operation(&w2).unwrap();
        let c2 = state.commit_transaction(tid(2)).unwrap();
        // T2 should be aborted due to write conflict
        assert!(!c2.committed);
        assert!(c2.abort_reason.unwrap().contains("update conflict"));
    }

    #[test]
    fn test_sqlserver_model_traits() {
        let model = SqlServerModel::new();
        assert_eq!(model.engine_kind(), EngineKind::SqlServer);
        assert_eq!(model.supported_isolation_levels().len(), 5);
        assert_eq!(model.version_string(), "2022");
    }

    #[test]
    fn test_sqlserver_encode_constraints_pessimistic() {
        let model = SqlServerModel::new(); // rcsi=false
        let cs = model
            .encode_constraints(IsolationLevel::Serializable, 3, 6)
            .unwrap();
        assert!(cs.variable_count() > 0);
        let smtlib = cs.to_smtlib2();
        assert!(smtlib.contains("ss_start_0"));
        assert!(smtlib.contains("ss_lock_"));
        assert!(smtlib.contains("ss_keyrange_"));
    }

    #[test]
    fn test_sqlserver_encode_constraints_optimistic() {
        let model = SqlServerModel::new().with_rcsi(true);
        let cs = model
            .encode_constraints(IsolationLevel::Snapshot, 3, 6)
            .unwrap();
        let smtlib = cs.to_smtlib2();
        assert!(smtlib.contains("ss_snap_"));
        assert!(smtlib.contains("ss_ww_conflict_"));
    }

    #[test]
    fn test_sqlserver_state_reset() {
        let mut state = SqlServerState::new(IsolationLevel::ReadCommitted, false);
        state.begin_transaction(tid(1), IsolationLevel::ReadCommitted).unwrap();
        state.reset();
        assert!(state.active_transactions().is_empty());
        assert_eq!(state.version_store.version_count(), 0);
    }
}
