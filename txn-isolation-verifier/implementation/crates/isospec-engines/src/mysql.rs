//! MySQL 8.0 InnoDB engine model.
//!
//! Key mechanisms:
//! - Next-key locks: record lock + gap lock on gap before record
//! - Gap locks depend on index selection by query optimizer
//! - Sound over-approximation: GapLockSet = union over all possible index choices
//! - Lock compatibility matrix: S/X/IS/IX/AI modes

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
// Gap lock representation
// ---------------------------------------------------------------------------

/// A gap in the key space, identified by the (table, next_key) pair.
/// The gap is the open interval before `next_key`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GapId {
    pub table: TableId,
    pub next_key: ItemId,
}

/// A single gap-lock entry.
#[derive(Debug, Clone)]
pub struct GapLockEntry {
    pub txn_id: TransactionId,
    pub gap: GapId,
    pub mode: GapLockMode,
    pub timestamp: u64,
}

/// Gap lock modes in InnoDB.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GapLockMode {
    /// Pure gap lock (prevents inserts into the gap).
    Gap,
    /// Next-key lock = record lock + gap lock.
    NextKey,
    /// Insert intention lock (compatible with gap locks from other txns as
    /// long as the insert positions don't overlap).
    InsertIntention,
}

impl GapLockMode {
    /// Gap locks are compatible with each other; insert-intention conflicts
    /// with gap/next-key held by another txn.
    pub fn is_compatible_with(self, other: GapLockMode) -> bool {
        match (self, other) {
            (GapLockMode::Gap, GapLockMode::Gap) => true,
            (GapLockMode::Gap, GapLockMode::NextKey) => true,
            (GapLockMode::NextKey, GapLockMode::Gap) => true,
            (GapLockMode::NextKey, GapLockMode::NextKey) => true,
            (GapLockMode::InsertIntention, GapLockMode::InsertIntention) => true,
            // Insert intention conflicts with gap/next-key
            (GapLockMode::InsertIntention, _) => false,
            (_, GapLockMode::InsertIntention) => false,
        }
    }
}

/// Manages the gap-lock set for InnoDB.
#[derive(Debug, Clone, Default)]
pub struct GapLockManager {
    entries: Vec<GapLockEntry>,
}

impl GapLockManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// Try to acquire a gap lock. Returns Ok(()) or Err(blocking_txn).
    pub fn acquire(
        &mut self,
        txn_id: TransactionId,
        gap: GapId,
        mode: GapLockMode,
        ts: u64,
    ) -> Result<(), TransactionId> {
        for entry in &self.entries {
            if entry.txn_id == txn_id {
                continue;
            }
            if entry.gap != gap {
                continue;
            }
            if !mode.is_compatible_with(entry.mode) {
                return Err(entry.txn_id);
            }
        }
        self.entries.push(GapLockEntry {
            txn_id,
            gap,
            mode,
            timestamp: ts,
        });
        Ok(())
    }

    /// Release all gap locks held by `txn_id`.
    pub fn release_all(&mut self, txn_id: TransactionId) {
        self.entries.retain(|e| e.txn_id != txn_id);
    }

    /// Check if a gap lock is held on the given gap by any other txn.
    pub fn is_gap_locked(&self, gap: &GapId, exclude: TransactionId) -> Option<TransactionId> {
        self.entries
            .iter()
            .find(|e| e.txn_id != exclude && e.gap == *gap && e.mode != GapLockMode::InsertIntention)
            .map(|e| e.txn_id)
    }

    pub fn count(&self) -> usize {
        self.entries.len()
    }
}

// ---------------------------------------------------------------------------
// Next-key lock set
// ---------------------------------------------------------------------------

/// A next-key lock entry combines a record lock and a gap lock.
#[derive(Debug, Clone)]
pub struct NextKeyLockEntry {
    pub txn_id: TransactionId,
    pub table: TableId,
    pub item: ItemId,
    pub lock_mode: LockMode,
    pub timestamp: u64,
}

/// Manages next-key locks (record + gap before the record).
#[derive(Debug, Clone, Default)]
pub struct NextKeyLockSet {
    entries: Vec<NextKeyLockEntry>,
}

impl NextKeyLockSet {
    pub fn new() -> Self {
        Self::default()
    }

    /// Acquire a next-key lock; also implicitly creates a gap lock in the
    /// gap manager.
    pub fn acquire(
        &mut self,
        txn_id: TransactionId,
        table: TableId,
        item: ItemId,
        mode: LockMode,
        ts: u64,
        gap_mgr: &mut GapLockManager,
    ) -> Result<(), TransactionId> {
        // Check record-level conflict
        for entry in &self.entries {
            if entry.txn_id == txn_id {
                continue;
            }
            if entry.table != table || entry.item != item {
                continue;
            }
            if !entry.lock_mode.is_compatible_with(mode) {
                return Err(entry.txn_id);
            }
        }
        // Acquire gap lock for the gap before this record
        let gap = GapId {
            table,
            next_key: item,
        };
        let gap_mode = if mode == LockMode::Shared {
            GapLockMode::NextKey
        } else {
            GapLockMode::NextKey
        };
        gap_mgr.acquire(txn_id, gap, gap_mode, ts)?;

        self.entries.push(NextKeyLockEntry {
            txn_id,
            table,
            item,
            lock_mode: mode,
            timestamp: ts,
        });
        Ok(())
    }

    /// Release all next-key locks for `txn_id`.
    pub fn release_all(&mut self, txn_id: TransactionId) {
        self.entries.retain(|e| e.txn_id != txn_id);
    }

    /// Check if `txn_id` already holds a next-key lock on `(table, item)`.
    pub fn holds_lock(
        &self,
        txn_id: TransactionId,
        table: TableId,
        item: ItemId,
    ) -> bool {
        self.entries.iter().any(|e| {
            e.txn_id == txn_id && e.table == table && e.item == item
        })
    }

    pub fn count(&self) -> usize {
        self.entries.len()
    }
}

// ---------------------------------------------------------------------------
// MySQL InnoDB Engine State
// ---------------------------------------------------------------------------

/// Full mutable state for the MySQL InnoDB engine.
pub struct MySqlInnoDBState {
    pub txn_states: HashMap<TransactionId, TransactionState>,
    pub version_store: VersionStoreCommon,
    pub lock_table: LockTableCommon,
    pub gap_locks: GapLockManager,
    pub next_key_locks: NextKeyLockSet,
    pub snapshot_mgr: SnapshotManager,
    pub snapshots: HashMap<TransactionId, Snapshot>,
    pub dep_tracker: DependencyTracker,
    pub clock: EngineTimestamp,
    pub committed_set: HashSet<TransactionId>,
    pub isolation_level: IsolationLevel,
    pub over_approximate_indexes: bool,
}

impl MySqlInnoDBState {
    pub fn new(isolation_level: IsolationLevel) -> Self {
        Self {
            txn_states: HashMap::new(),
            version_store: VersionStoreCommon::new(),
            lock_table: LockTableCommon::new(),
            gap_locks: GapLockManager::new(),
            next_key_locks: NextKeyLockSet::new(),
            snapshot_mgr: SnapshotManager::new(),
            snapshots: HashMap::new(),
            dep_tracker: DependencyTracker::new(),
            clock: EngineTimestamp::new(),
            committed_set: HashSet::new(),
            isolation_level,
            over_approximate_indexes: true,
        }
    }

    fn committed_list(&self) -> Vec<TransactionId> {
        self.committed_set.iter().copied().collect()
    }

    /// At RR, InnoDB uses a snapshot taken at the first consistent read.
    /// At RC, InnoDB takes a new snapshot for each statement (simplified: per read).
    fn snapshot_time(&mut self, txn_id: TransactionId) -> u64 {
        if self.isolation_level == IsolationLevel::ReadCommitted
            || self.isolation_level == IsolationLevel::PgReadCommitted
        {
            // Fresh snapshot per read
            self.clock.current()
        } else {
            // RR/Serializable: snapshot at begin
            self.snapshots
                .get(&txn_id)
                .map(|s| s.snapshot_time)
                .unwrap_or(self.clock.current())
        }
    }

    fn handle_read(
        &mut self,
        txn_id: TransactionId,
        table: TableId,
        item: ItemId,
        op_id: OperationId,
    ) -> IsoSpecResult<OperationOutcome> {
        let ts = self.clock.tick();
        let snap_time = self.snapshot_time(txn_id);
        let committed = self.committed_list();

        // At SERIALIZABLE, InnoDB acquires S next-key locks on reads
        if self.isolation_level == IsolationLevel::Serializable {
            let res = self.next_key_locks.acquire(
                txn_id,
                table,
                item,
                LockMode::Shared,
                ts,
                &mut self.gap_locks,
            );
            if let Err(blocker) = res {
                return Ok(OperationOutcome::blocked(blocker));
            }
        }

        // At RR, InnoDB uses a consistent read (no locks on plain SELECTs)
        let visible = self.version_store.read_visible(item, table, snap_time, &committed);

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

    fn handle_write(
        &mut self,
        txn_id: TransactionId,
        table: TableId,
        item: ItemId,
        value: Value,
        op_id: OperationId,
    ) -> IsoSpecResult<OperationOutcome> {
        let ts = self.clock.tick();

        // InnoDB acquires X record lock on writes at all isolation levels
        let lock_res = self.lock_table.acquire(
            txn_id,
            table,
            Some(item),
            LockMode::Exclusive,
            LockTag::Regular,
            ts,
        );
        if let Err(blocker) = lock_res {
            return Ok(OperationOutcome::blocked(blocker));
        }

        // At RR/Serializable, also acquire next-key lock
        if self.isolation_level == IsolationLevel::RepeatableRead
            || self.isolation_level == IsolationLevel::MySqlRepeatableRead
            || self.isolation_level == IsolationLevel::Serializable
        {
            let nk_res = self.next_key_locks.acquire(
                txn_id,
                table,
                item,
                LockMode::Exclusive,
                ts,
                &mut self.gap_locks,
            );
            if let Err(blocker) = nk_res {
                return Ok(OperationOutcome::blocked(blocker));
            }
        }

        // Track rw-dependency: if anyone previously read this item
        for (&other_id, other_st) in &self.txn_states {
            if other_id == txn_id {
                continue;
            }
            if other_st.read_item(table, item) {
                self.dep_tracker.add_rw(other_id, txn_id);
            }
        }

        let vid = self.version_store.write_version(item, table, txn_id, ts, value.clone());

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

        // Insert intention lock
        let gap = GapId {
            table,
            next_key: item,
        };
        if let Err(blocker) =
            self.gap_locks
                .acquire(txn_id, gap, GapLockMode::InsertIntention, ts)
        {
            return Ok(OperationOutcome::blocked(blocker));
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
        let ts = self.clock.tick();
        let snap_time = self.snapshot_time(txn_id);
        let committed = self.committed_list();

        // At RR/Serializable, next-key lock on each item in the range
        if self.isolation_level == IsolationLevel::RepeatableRead
            || self.isolation_level == IsolationLevel::MySqlRepeatableRead
            || self.isolation_level == IsolationLevel::Serializable
        {
            for &item_id in items {
                let res = self.next_key_locks.acquire(
                    txn_id,
                    table,
                    item_id,
                    LockMode::Shared,
                    ts,
                    &mut self.gap_locks,
                );
                if let Err(blocker) = res {
                    return Ok(OperationOutcome::blocked(blocker));
                }
            }

            // Over-approximate: lock the gaps between all items
            if self.over_approximate_indexes && !items.is_empty() {
                // We add gap locks on all items in the predicate range
                for &item_id in items {
                    let gap = GapId {
                        table,
                        next_key: item_id,
                    };
                    let _ = self.gap_locks.acquire(txn_id, gap, GapLockMode::Gap, ts);
                }
            }
        }

        let mut outcome = OperationOutcome::success();
        for &item_id in items {
            if let Some((vid, val)) =
                self.version_store.read_visible(item_id, table, snap_time, &committed)
            {
                if let Some(st) = self.txn_states.get_mut(&txn_id) {
                    st.record_read(table, item_id, Some(vid), op_id, ts);
                }
                outcome = outcome.with_value(val);
            }
        }
        Ok(outcome)
    }
}

impl EngineState for MySqlInnoDBState {
    fn begin_transaction(
        &mut self,
        txn_id: TransactionId,
        level: IsolationLevel,
    ) -> IsoSpecResult<()> {
        let ts = self.clock.tick();
        let st = TransactionState::new(txn_id, level, ts);
        self.txn_states.insert(txn_id, st);

        // InnoDB takes snapshot at the first consistent read for RR;
        // for RC, snapshot is per-statement. We take it here for RR.
        if level != IsolationLevel::ReadCommitted {
            let active: HashSet<TransactionId> = self
                .txn_states
                .iter()
                .filter(|(id, s)| **id != txn_id && s.is_active())
                .map(|(&id, _)| id)
                .collect();
            let committed = self.committed_set.clone();
            let snap = self.snapshot_mgr.take_snapshot(txn_id, active, committed);
            self.snapshots.insert(txn_id, snap);
        }
        Ok(())
    }

    fn execute_operation(&mut self, op: &Operation) -> IsoSpecResult<OperationOutcome> {
        let txn_id = op.txn_id;
        if !self.txn_states.contains_key(&txn_id) {
            return Err(IsoSpecError::engine_model(
                "MySQL",
                format!("Transaction {} not found", txn_id),
            ));
        }
        match &op.kind {
            OpKind::Read(r) => self.handle_read(txn_id, r.table, r.item, op.id),
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
                let res = self.lock_table.acquire(
                    txn_id,
                    l.table,
                    l.item,
                    l.mode,
                    LockTag::Regular,
                    ts,
                );
                match res {
                    Ok(_) => Ok(OperationOutcome::success()),
                    Err(blocker) => Ok(OperationOutcome::blocked(blocker)),
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

        if let Some(st) = self.txn_states.get_mut(&txn_id) {
            st.status = TransactionStatus::Committed;
            st.commit_ts = Some(ts);
        }
        self.committed_set.insert(txn_id);

        // Release all locks
        self.lock_table.release_all(txn_id);
        self.gap_locks.release_all(txn_id);
        self.next_key_locks.release_all(txn_id);

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
        self.lock_table.release_all(txn_id);
        self.gap_locks.release_all(txn_id);
        self.next_key_locks.release_all(txn_id);
        self.snapshots.remove(&txn_id);
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
        self.snapshots.get(&txn_id).map(|snap| SnapshotInfo {
            snapshot_time: snap.snapshot_time,
            active_txns: snap.active_txns.iter().copied().collect(),
            committed_before_snapshot: snap.committed_txns.iter().copied().collect(),
        })
    }

    fn reset(&mut self) {
        self.txn_states.clear();
        self.version_store = VersionStoreCommon::new();
        self.lock_table = LockTableCommon::new();
        self.gap_locks = GapLockManager::new();
        self.next_key_locks = NextKeyLockSet::new();
        self.snapshot_mgr = SnapshotManager::new();
        self.snapshots.clear();
        self.dep_tracker.clear();
        self.clock = EngineTimestamp::new();
        self.committed_set.clear();
    }
}

// ---------------------------------------------------------------------------
// MySQL engine model
// ---------------------------------------------------------------------------

/// MySQL 8.0 InnoDB engine model.
pub struct MySqlModel {
    version: String,
    over_approximate_indexes: bool,
}

impl MySqlModel {
    pub fn new() -> Self {
        Self {
            version: "8.0".to_string(),
            over_approximate_indexes: true,
        }
    }

    pub fn with_version(mut self, v: impl Into<String>) -> Self {
        self.version = v.into();
        self
    }

    pub fn with_index_over_approximation(mut self, enabled: bool) -> Self {
        self.over_approximate_indexes = enabled;
        self
    }

    fn encode_innodb_constraints(
        &self,
        txn_count: usize,
        _op_count: usize,
    ) -> SmtConstraintSet {
        let mut cs = SmtConstraintSet::new("QF_LIA");

        for i in 0..txn_count {
            let start = format!("my_start_{}", i);
            let commit = format!("my_commit_{}", i);
            cs.declare(&start, SmtSort::Int);
            cs.declare(&commit, SmtSort::Int);
            cs.assert(SmtExpr::lt(
                SmtExpr::int_var(&start),
                SmtExpr::int_var(&commit),
            ));
        }

        // Lock variables: lock_held_i_j means txn i holds X-lock when j tries
        for i in 0..txn_count {
            for j in 0..txn_count {
                if i == j {
                    continue;
                }
                let lock_var = format!("my_xlock_{}_{}", i, j);
                cs.declare(&lock_var, SmtSort::Bool);
                // If i holds X-lock and j wants S or X lock, j must wait
                let wait_var = format!("my_wait_{}_{}", j, i);
                cs.declare(&wait_var, SmtSort::Bool);
                cs.assert(SmtExpr::implies(
                    SmtExpr::bool_var(&lock_var),
                    SmtExpr::bool_var(&wait_var),
                ));
            }
        }

        // Gap lock constraint: next-key lock prevents phantom inserts
        for i in 0..txn_count {
            let nk_lock = format!("my_nklock_{}", i);
            cs.declare(&nk_lock, SmtSort::Bool);
            for j in 0..txn_count {
                if i == j {
                    continue;
                }
                let insert_blocked = format!("my_insert_blocked_{}_{}", j, i);
                cs.declare(&insert_blocked, SmtSort::Bool);
                cs.assert(SmtExpr::implies(
                    SmtExpr::bool_var(&nk_lock),
                    SmtExpr::bool_var(&insert_blocked),
                ));
            }
        }

        // No deadlocks: encode acyclicity of wait-for graph
        // wait_order_i < wait_order_j if j waits for i
        for i in 0..txn_count {
            let ord = format!("my_wait_order_{}", i);
            cs.declare(&ord, SmtSort::Int);
            cs.assert(SmtExpr::Ge(
                Box::new(SmtExpr::int_var(&ord)),
                Box::new(SmtExpr::IntLit(0)),
            ));
        }
        for i in 0..txn_count {
            for j in 0..txn_count {
                if i == j {
                    continue;
                }
                let wait_var = format!("my_wait_{}_{}", j, i);
                let ord_i = format!("my_wait_order_{}", i);
                let ord_j = format!("my_wait_order_{}", j);
                cs.assert(SmtExpr::implies(
                    SmtExpr::bool_var(&wait_var),
                    SmtExpr::lt(
                        SmtExpr::int_var(&ord_i),
                        SmtExpr::int_var(&ord_j),
                    ),
                ));
            }
        }

        cs
    }
}

impl Default for MySqlModel {
    fn default() -> Self {
        Self::new()
    }
}

impl EngineModel for MySqlModel {
    fn engine_kind(&self) -> EngineKind {
        EngineKind::MySQL
    }

    fn supported_isolation_levels(&self) -> Vec<IsolationLevel> {
        vec![
            IsolationLevel::ReadUncommitted,
            IsolationLevel::ReadCommitted,
            IsolationLevel::RepeatableRead,
            IsolationLevel::Serializable,
        ]
    }

    fn create_state(&self, isolation_level: IsolationLevel) -> Box<dyn EngineState> {
        let mut state = MySqlInnoDBState::new(isolation_level);
        state.over_approximate_indexes = self.over_approximate_indexes;
        Box::new(state)
    }

    fn encode_constraints(
        &self,
        _isolation_level: IsolationLevel,
        txn_count: usize,
        op_count: usize,
    ) -> IsoSpecResult<SmtConstraintSet> {
        Ok(self.encode_innodb_constraints(txn_count, op_count))
    }

    fn version_string(&self) -> &str {
        &self.version
    }

    fn validate_schedule(
        &self,
        schedule: &isospec_types::schedule::Schedule,
        level: IsolationLevel,
    ) -> IsoSpecResult<ValidationResult> {
        let mut state = MySqlInnoDBState::new(level);
        state.over_approximate_indexes = self.over_approximate_indexes;
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
    fn test_gap_lock_compatibility() {
        assert!(GapLockMode::Gap.is_compatible_with(GapLockMode::Gap));
        assert!(GapLockMode::Gap.is_compatible_with(GapLockMode::NextKey));
        assert!(!GapLockMode::InsertIntention.is_compatible_with(GapLockMode::Gap));
        assert!(GapLockMode::InsertIntention.is_compatible_with(GapLockMode::InsertIntention));
    }

    #[test]
    fn test_gap_lock_manager_basic() {
        let mut mgr = GapLockManager::new();
        let gap = GapId { table: tbl(), next_key: item(5) };
        assert!(mgr.acquire(tid(1), gap.clone(), GapLockMode::Gap, 1).is_ok());
        // Another gap lock is compatible
        assert!(mgr.acquire(tid(2), gap.clone(), GapLockMode::Gap, 2).is_ok());
        assert_eq!(mgr.count(), 2);
        // Insert intention conflicts
        assert!(mgr.acquire(tid(3), gap.clone(), GapLockMode::InsertIntention, 3).is_err());
    }

    #[test]
    fn test_gap_lock_release() {
        let mut mgr = GapLockManager::new();
        let gap = GapId { table: tbl(), next_key: item(5) };
        mgr.acquire(tid(1), gap.clone(), GapLockMode::Gap, 1).unwrap();
        mgr.release_all(tid(1));
        assert_eq!(mgr.count(), 0);
        // Now insert intention should succeed
        assert!(mgr.acquire(tid(2), gap.clone(), GapLockMode::InsertIntention, 2).is_ok());
    }

    #[test]
    fn test_next_key_lock_set() {
        let mut nk = NextKeyLockSet::new();
        let mut gap_mgr = GapLockManager::new();
        assert!(nk.acquire(tid(1), tbl(), item(5), LockMode::Shared, 1, &mut gap_mgr).is_ok());
        assert!(nk.holds_lock(tid(1), tbl(), item(5)));
        assert!(!nk.holds_lock(tid(2), tbl(), item(5)));
        assert_eq!(nk.count(), 1);
        assert_eq!(gap_mgr.count(), 1); // gap lock also acquired
    }

    #[test]
    fn test_mysql_state_begin_and_read() {
        let mut state = MySqlInnoDBState::new(IsolationLevel::RepeatableRead);
        state.begin_transaction(tid(1), IsolationLevel::RepeatableRead).unwrap();

        // Pre-populate
        state.version_store.write_version(item(1), tbl(), tid(0), 0, Value::Integer(77));
        state.committed_set.insert(tid(0));

        let op = Operation::read(oid(0), tid(1), tbl(), item(1), 1);
        let outcome = state.execute_operation(&op).unwrap();
        assert!(outcome.success);
        assert_eq!(outcome.values_read, vec![Value::Integer(77)]);
    }

    #[test]
    fn test_mysql_state_write_lock_conflict() {
        let mut state = MySqlInnoDBState::new(IsolationLevel::RepeatableRead);
        state.begin_transaction(tid(1), IsolationLevel::RepeatableRead).unwrap();
        state.begin_transaction(tid(2), IsolationLevel::RepeatableRead).unwrap();

        // T1 writes item 1 → X lock
        let w1 = Operation::write(oid(0), tid(1), tbl(), item(1), Value::Integer(10), 1);
        let r1 = state.execute_operation(&w1).unwrap();
        assert!(r1.success);

        // T2 tries to write same item → blocked
        let w2 = Operation::write(oid(1), tid(2), tbl(), item(1), Value::Integer(20), 2);
        let r2 = state.execute_operation(&w2).unwrap();
        assert!(r2.blocked);
        assert_eq!(r2.blocked_by, Some(tid(1)));
    }

    #[test]
    fn test_mysql_state_commit_releases_locks() {
        let mut state = MySqlInnoDBState::new(IsolationLevel::RepeatableRead);
        state.begin_transaction(tid(1), IsolationLevel::RepeatableRead).unwrap();
        state.begin_transaction(tid(2), IsolationLevel::RepeatableRead).unwrap();

        let w1 = Operation::write(oid(0), tid(1), tbl(), item(1), Value::Integer(10), 1);
        state.execute_operation(&w1).unwrap();
        state.commit_transaction(tid(1)).unwrap();

        // After T1 commits, T2 should be able to write
        let w2 = Operation::write(oid(1), tid(2), tbl(), item(1), Value::Integer(20), 2);
        let r2 = state.execute_operation(&w2).unwrap();
        assert!(r2.success);
    }

    #[test]
    fn test_mysql_model_traits() {
        let model = MySqlModel::new();
        assert_eq!(model.engine_kind(), EngineKind::MySQL);
        assert_eq!(model.supported_isolation_levels().len(), 4);
        assert_eq!(model.version_string(), "8.0");
    }

    #[test]
    fn test_mysql_encode_constraints() {
        let model = MySqlModel::new();
        let cs = model
            .encode_constraints(IsolationLevel::RepeatableRead, 3, 6)
            .unwrap();
        assert!(cs.variable_count() > 0);
        assert!(cs.constraint_count() > 0);
        let smtlib = cs.to_smtlib2();
        assert!(smtlib.contains("my_start_0"));
        assert!(smtlib.contains("my_xlock_"));
    }

    #[test]
    fn test_mysql_state_reset() {
        let mut state = MySqlInnoDBState::new(IsolationLevel::RepeatableRead);
        state.begin_transaction(tid(1), IsolationLevel::RepeatableRead).unwrap();
        state.reset();
        assert!(state.active_transactions().is_empty());
        assert_eq!(state.version_store.version_count(), 0);
        assert_eq!(state.lock_table.lock_count(), 0);
    }
}
