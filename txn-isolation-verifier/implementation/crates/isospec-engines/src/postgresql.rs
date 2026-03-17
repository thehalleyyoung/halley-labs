//! PostgreSQL 16.x Serializable Snapshot Isolation (SSI) engine model.
//!
//! Key mechanisms:
//! - SIREAD predicate-level read locks for rw-dependency detection
//! - Dangerous structure detection: consecutive rw-deps T1→T2→T3 where T1
//!   committed before T3's snapshot
//! - Read-only optimization: read-only txn with snapshot preceding all
//!   concurrent writers skips abort check
//! - Granularity escalation: tuple → page → relation under memory pressure

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
// SIRead lock granularity
// ---------------------------------------------------------------------------

/// Granularity of a SIREAD lock.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SIReadGranularity {
    Tuple,
    Page,
    Relation,
}

/// A single SIREAD lock entry.
#[derive(Debug, Clone)]
pub struct SIReadLockEntry {
    pub txn_id: TransactionId,
    pub table: TableId,
    pub item: Option<ItemId>,
    pub granularity: SIReadGranularity,
    pub timestamp: u64,
}

/// Manages the SIREAD lock set used by PostgreSQL SSI.
#[derive(Debug, Clone, Default)]
pub struct SIReadLockSet {
    entries: Vec<SIReadLockEntry>,
    /// Per-table tuple-lock count for escalation decisions.
    tuple_counts: HashMap<(TransactionId, TableId), usize>,
    escalation_threshold: usize,
}

impl SIReadLockSet {
    pub fn new(escalation_threshold: usize) -> Self {
        Self {
            entries: Vec::new(),
            tuple_counts: HashMap::new(),
            escalation_threshold,
        }
    }

    /// Record a SIREAD lock on a tuple. May escalate if threshold exceeded.
    pub fn add_tuple_lock(
        &mut self,
        txn_id: TransactionId,
        table: TableId,
        item: ItemId,
        ts: u64,
    ) {
        let count = self
            .tuple_counts
            .entry((txn_id, table))
            .or_insert(0);
        *count += 1;

        if *count > self.escalation_threshold {
            self.escalate_to_relation(txn_id, table, ts);
            return;
        }

        // Don't add duplicate tuple locks
        let already = self.entries.iter().any(|e| {
            e.txn_id == txn_id
                && e.table == table
                && e.item == Some(item)
                && e.granularity == SIReadGranularity::Tuple
        });
        if !already {
            self.entries.push(SIReadLockEntry {
                txn_id,
                table,
                item: Some(item),
                granularity: SIReadGranularity::Tuple,
                timestamp: ts,
            });
        }
    }

    /// Add a relation-level SIREAD lock, removing all finer-grained locks.
    pub fn escalate_to_relation(
        &mut self,
        txn_id: TransactionId,
        table: TableId,
        ts: u64,
    ) {
        // Already have relation lock?
        let has_rel = self.entries.iter().any(|e| {
            e.txn_id == txn_id
                && e.table == table
                && e.granularity == SIReadGranularity::Relation
        });
        if has_rel {
            return;
        }
        // Remove finer-grained
        self.entries
            .retain(|e| !(e.txn_id == txn_id && e.table == table));
        self.entries.push(SIReadLockEntry {
            txn_id,
            table,
            item: None,
            granularity: SIReadGranularity::Relation,
            timestamp: ts,
        });
    }

    /// Check if a write by `writer` to `(table, item)` conflicts with any
    /// SIREAD lock held by a *different* transaction. Returns conflicting txn.
    pub fn check_rw_conflict(
        &self,
        writer: TransactionId,
        table: TableId,
        item: ItemId,
    ) -> Vec<TransactionId> {
        let mut result = Vec::new();
        for entry in &self.entries {
            if entry.txn_id == writer {
                continue;
            }
            if entry.table != table {
                continue;
            }
            let conflicts = match entry.granularity {
                SIReadGranularity::Relation => true,
                SIReadGranularity::Page => true, // conservative
                SIReadGranularity::Tuple => entry.item == Some(item),
            };
            if conflicts && !result.contains(&entry.txn_id) {
                result.push(entry.txn_id);
            }
        }
        result
    }

    /// Remove all locks held by `txn`.
    pub fn remove_txn(&mut self, txn_id: TransactionId) {
        self.entries.retain(|e| e.txn_id != txn_id);
        self.tuple_counts.retain(|&(t, _), _| t != txn_id);
    }

    pub fn count(&self) -> usize {
        self.entries.len()
    }
}

// ---------------------------------------------------------------------------
// Dangerous-structure detector
// ---------------------------------------------------------------------------

/// An rw-dependency edge in the SSI graph.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RWEdge {
    pub from: TransactionId,
    pub to: TransactionId,
}

/// Detects "dangerous structures": T1→(rw)→T2→(rw)→T3 where T1 committed
/// before T3 took its snapshot, meaning the cycle cannot be broken by
/// aborting T1.
#[derive(Debug, Clone, Default)]
pub struct DangerousStructureDetector {
    rw_edges: Vec<RWEdge>,
}

impl DangerousStructureDetector {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_rw_edge(&mut self, from: TransactionId, to: TransactionId) {
        let edge = RWEdge { from, to };
        if !self.rw_edges.contains(&edge) {
            self.rw_edges.push(edge);
        }
    }

    /// Find all dangerous structures (consecutive rw-dep pairs).
    pub fn find_dangerous_structures(
        &self,
        txn_states: &HashMap<TransactionId, TransactionState>,
        snapshots: &HashMap<TransactionId, Snapshot>,
    ) -> Vec<(TransactionId, TransactionId, TransactionId)> {
        let mut structures = Vec::new();
        // T1 -rw-> T2 -rw-> T3
        for e1 in &self.rw_edges {
            for e2 in &self.rw_edges {
                if e1.to != e2.from {
                    continue;
                }
                let t1 = e1.from;
                let t2 = e1.to;
                let t3 = e2.to;
                if t1 == t3 {
                    continue;
                }
                // Check: T1 committed before T3's snapshot
                let t1_committed = txn_states
                    .get(&t1)
                    .and_then(|s| s.commit_ts);
                let t3_snap_time = snapshots
                    .get(&t3)
                    .map(|s| s.snapshot_time);
                let is_dangerous = match (t1_committed, t3_snap_time) {
                    (Some(ct), Some(st)) => ct < st,
                    _ => true, // conservative: treat unknown as dangerous
                };
                if is_dangerous {
                    structures.push((t1, t2, t3));
                }
            }
        }
        structures
    }

    /// Check whether the pivot (T2) should be aborted.
    pub fn should_abort_pivot(
        &self,
        pivot: TransactionId,
        txn_states: &HashMap<TransactionId, TransactionState>,
        snapshots: &HashMap<TransactionId, Snapshot>,
        read_only_optimization: bool,
    ) -> bool {
        let structures = self.find_dangerous_structures(txn_states, snapshots);
        for (t1, t2, t3) in &structures {
            if *t2 != pivot {
                continue;
            }
            // Read-only optimization: if T2 is read-only and its snapshot
            // precedes all concurrent writers, skip abort.
            if read_only_optimization {
                if let Some(st) = txn_states.get(t2) {
                    if st.read_only && st.write_set.is_empty() {
                        let snap_ts = snapshots.get(t2).map(|s| s.snapshot_time).unwrap_or(0);
                        let all_writers_later = txn_states.values().all(|other| {
                            if other.txn_id == *t2 {
                                return true;
                            }
                            if other.write_set.is_empty() {
                                return true;
                            }
                            other.start_ts >= snap_ts
                        });
                        if all_writers_later {
                            continue;
                        }
                    }
                }
            }
            return true;
        }
        false
    }

    pub fn clear(&mut self) {
        self.rw_edges.clear();
    }

    pub fn edge_count(&self) -> usize {
        self.rw_edges.len()
    }
}

// ---------------------------------------------------------------------------
// PostgreSQL SSI Engine State
// ---------------------------------------------------------------------------

/// Full mutable state for the PostgreSQL SSI engine.
pub struct PostgresSSIState {
    pub txn_states: HashMap<TransactionId, TransactionState>,
    pub version_store: VersionStoreCommon,
    pub siread_locks: SIReadLockSet,
    pub dangerous_detector: DangerousStructureDetector,
    pub snapshot_mgr: SnapshotManager,
    pub snapshots: HashMap<TransactionId, Snapshot>,
    pub dep_tracker: DependencyTracker,
    pub clock: EngineTimestamp,
    pub committed_set: HashSet<TransactionId>,
    pub read_only_optimization: bool,
    pub isolation_level: IsolationLevel,
}

impl PostgresSSIState {
    pub fn new(isolation_level: IsolationLevel) -> Self {
        Self {
            txn_states: HashMap::new(),
            version_store: VersionStoreCommon::new(),
            siread_locks: SIReadLockSet::new(10_000),
            dangerous_detector: DangerousStructureDetector::new(),
            snapshot_mgr: SnapshotManager::new(),
            snapshots: HashMap::new(),
            dep_tracker: DependencyTracker::new(),
            clock: EngineTimestamp::new(),
            committed_set: HashSet::new(),
            read_only_optimization: true,
            isolation_level,
        }
    }

    fn committed_list(&self) -> Vec<TransactionId> {
        self.committed_set.iter().copied().collect()
    }

    fn handle_read(
        &mut self,
        txn_id: TransactionId,
        table: TableId,
        item: ItemId,
        op_id: OperationId,
    ) -> IsoSpecResult<OperationOutcome> {
        let ts = self.clock.tick();

        // Determine snapshot time for visibility
        let snap_time = self
            .snapshots
            .get(&txn_id)
            .map(|s| s.snapshot_time)
            .unwrap_or(ts);
        let committed = self.committed_list();

        let visible = self
            .version_store
            .read_visible(item, table, snap_time, &committed);

        // Record SIREAD lock at SERIALIZABLE level
        if self.isolation_level == IsolationLevel::Serializable {
            self.siread_locks.add_tuple_lock(txn_id, table, item, ts);
        }

        if let Some((vid, val)) = visible {
            if let Some(st) = self.txn_states.get_mut(&txn_id) {
                st.record_read(table, item, Some(vid), op_id, ts);
            }
            // Detect wr-dependency from the writer to this reader
            if let Some(writer) = self.version_store.latest_writer(table, item) {
                if writer != txn_id && self.committed_set.contains(&writer) {
                    self.dep_tracker.add_wr(writer, txn_id);
                }
            }
            Ok(OperationOutcome::success().with_value(val))
        } else {
            // No visible version → return NULL
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

        // First-committer-wins: check for write conflicts at Serializable
        if self.isolation_level == IsolationLevel::Serializable
            || self.isolation_level == IsolationLevel::RepeatableRead
        {
            let snap_time = self
                .snapshots
                .get(&txn_id)
                .map(|s| s.snapshot_time)
                .unwrap_or(0);
            let committed = self.committed_list();
            if let Some(conflicting) =
                self.version_store
                    .has_write_conflict(table, item, snap_time, txn_id, &committed)
            {
                self.dep_tracker.add_ww(conflicting, txn_id);
            }
        }

        // Check SIREAD conflicts: if another txn holds a SIREAD lock on this
        // item, we create an rw-dependency edge from reader→writer.
        if self.isolation_level == IsolationLevel::Serializable {
            let conflicting_readers = self.siread_locks.check_rw_conflict(txn_id, table, item);
            for reader in &conflicting_readers {
                self.dep_tracker.add_rw(*reader, txn_id);
                self.dangerous_detector.add_rw_edge(*reader, txn_id);
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
        let snap_time = self
            .snapshots
            .get(&txn_id)
            .map(|s| s.snapshot_time)
            .unwrap_or(ts);
        let committed = self.committed_list();

        // At SERIALIZABLE: escalate to relation-level SIREAD lock
        if self.isolation_level == IsolationLevel::Serializable {
            self.siread_locks.escalate_to_relation(txn_id, table, ts);
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

impl EngineState for PostgresSSIState {
    fn begin_transaction(
        &mut self,
        txn_id: TransactionId,
        level: IsolationLevel,
    ) -> IsoSpecResult<()> {
        let ts = self.clock.tick();
        let st = TransactionState::new(txn_id, level, ts);
        self.txn_states.insert(txn_id, st);

        // Take snapshot
        let active: HashSet<TransactionId> = self
            .txn_states
            .iter()
            .filter(|(id, s)| **id != txn_id && s.is_active())
            .map(|(&id, _)| id)
            .collect();
        let committed = self.committed_set.clone();
        let snap = self.snapshot_mgr.take_snapshot(txn_id, active, committed);
        self.snapshots.insert(txn_id, snap);
        Ok(())
    }

    fn execute_operation(&mut self, op: &Operation) -> IsoSpecResult<OperationOutcome> {
        let txn_id = op.txn_id;
        if !self.txn_states.contains_key(&txn_id) {
            return Err(IsoSpecError::engine_model(
                "PostgreSQL",
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
                    self.handle_write(txn_id, d.table, item_id, Value::Null, op.id)?;
                }
                Ok(OperationOutcome::success())
            }
            OpKind::PredicateRead(pr) => {
                self.handle_predicate_read(txn_id, pr.table, &pr.items_read, op.id)
            }
            OpKind::PredicateWrite(pw) => {
                for &item_id in &pw.items_written {
                    self.handle_write(txn_id, pw.table, item_id, pw.new_value.clone(), op.id)?;
                }
                Ok(OperationOutcome::success())
            }
            OpKind::Lock(_) => Ok(OperationOutcome::success()),
            OpKind::Begin(_) => {
                // Already handled by begin_transaction
                Ok(OperationOutcome::success())
            }
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

        // SSI dangerous-structure check at SERIALIZABLE
        if self.isolation_level == IsolationLevel::Serializable {
            let should_abort = self.dangerous_detector.should_abort_pivot(
                txn_id,
                &self.txn_states,
                &self.snapshots,
                self.read_only_optimization,
            );
            if should_abort {
                self.abort_transaction(txn_id)?;
                return Ok(CommitOutcome::aborted(
                    "serialization failure: dangerous structure detected",
                ));
            }
        }

        // First-committer-wins at RepeatableRead / Serializable
        if self.isolation_level == IsolationLevel::RepeatableRead
            || self.isolation_level == IsolationLevel::Serializable
        {
            let snap_time = self
                .snapshots
                .get(&txn_id)
                .map(|s| s.snapshot_time)
                .unwrap_or(0);
            let committed = self.committed_list();
            if let Some(st) = self.txn_states.get(&txn_id) {
                for w in &st.write_set {
                    if let Some(conflict) = self.version_store.has_write_conflict(
                        w.table,
                        w.item,
                        snap_time,
                        txn_id,
                        &committed,
                    ) {
                        if conflict != txn_id {
                            self.abort_transaction(txn_id)?;
                            return Ok(CommitOutcome::aborted(
                                "serialization failure: write conflict",
                            ));
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
        // SIREAD locks are kept until all concurrent txns finish (simplified:
        // remove them at commit time since our model is trace-driven).

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
        self.siread_locks.remove_txn(txn_id);
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
        self.siread_locks = SIReadLockSet::new(self.siread_locks.escalation_threshold);
        self.dangerous_detector.clear();
        self.snapshot_mgr = SnapshotManager::new();
        self.snapshots.clear();
        self.dep_tracker.clear();
        self.clock = EngineTimestamp::new();
        self.committed_set.clear();
    }
}

// ---------------------------------------------------------------------------
// PostgreSQL engine model
// ---------------------------------------------------------------------------

/// PostgreSQL 16.x SSI engine model.
pub struct PostgresModel {
    version: String,
    read_only_opt: bool,
}

impl PostgresModel {
    pub fn new() -> Self {
        Self {
            version: "16.0".to_string(),
            read_only_opt: true,
        }
    }

    pub fn with_version(mut self, v: impl Into<String>) -> Self {
        self.version = v.into();
        self
    }

    pub fn with_read_only_optimization(mut self, enabled: bool) -> Self {
        self.read_only_opt = enabled;
        self
    }

    /// Generate SMT constraints encoding PG SSI rules.
    fn encode_ssi_constraints(
        &self,
        txn_count: usize,
        op_count: usize,
    ) -> SmtConstraintSet {
        let mut cs = SmtConstraintSet::new("QF_LIA");

        // Declare timestamp variables for each transaction
        for i in 0..txn_count {
            let start = format!("pg_start_{}", i);
            let commit = format!("pg_commit_{}", i);
            let snap = format!("pg_snap_{}", i);
            cs.declare(&start, SmtSort::Int);
            cs.declare(&commit, SmtSort::Int);
            cs.declare(&snap, SmtSort::Int);
            // start < commit
            cs.assert(SmtExpr::lt(
                SmtExpr::int_var(&start),
                SmtExpr::int_var(&commit),
            ));
            // snap == start (snapshot taken at begin)
            cs.assert(SmtExpr::eq(
                SmtExpr::int_var(&snap),
                SmtExpr::int_var(&start),
            ));
        }

        // Declare rw-dependency booleans for each pair
        for i in 0..txn_count {
            for j in 0..txn_count {
                if i == j {
                    continue;
                }
                let rw_var = format!("pg_rw_{}_{}", i, j);
                cs.declare(&rw_var, SmtSort::Bool);
            }
        }

        // Dangerous structure: rw(i,j) ∧ rw(j,k) ∧ commit_i < snap_k ⇒ abort_j
        for i in 0..txn_count {
            for j in 0..txn_count {
                if i == j {
                    continue;
                }
                for k in 0..txn_count {
                    if k == i || k == j {
                        continue;
                    }
                    let rw_ij = format!("pg_rw_{}_{}", i, j);
                    let rw_jk = format!("pg_rw_{}_{}", j, k);
                    let commit_i = format!("pg_commit_{}", i);
                    let snap_k = format!("pg_snap_{}", k);
                    let abort_j = format!("pg_abort_{}", j);
                    cs.declare(&abort_j, SmtSort::Bool);

                    let dangerous = SmtExpr::and(vec![
                        SmtExpr::bool_var(&rw_ij),
                        SmtExpr::bool_var(&rw_jk),
                        SmtExpr::lt(
                            SmtExpr::int_var(&commit_i),
                            SmtExpr::int_var(&snap_k),
                        ),
                    ]);
                    cs.assert(SmtExpr::implies(dangerous, SmtExpr::bool_var(&abort_j)));
                }
            }
        }

        // Visibility: a read in txn i sees writes from txn j iff
        // commit_j < snap_i
        for i in 0..txn_count {
            for j in 0..txn_count {
                if i == j {
                    continue;
                }
                let vis = format!("pg_vis_{}_{}", j, i);
                cs.declare(&vis, SmtSort::Bool);
                let snap_i = format!("pg_snap_{}", i);
                let commit_j = format!("pg_commit_{}", j);
                cs.assert(SmtExpr::Iff(
                    Box::new(SmtExpr::bool_var(&vis)),
                    Box::new(SmtExpr::lt(
                        SmtExpr::int_var(&commit_j),
                        SmtExpr::int_var(&snap_i),
                    )),
                ));
            }
        }

        cs
    }
}

impl Default for PostgresModel {
    fn default() -> Self {
        Self::new()
    }
}

impl EngineModel for PostgresModel {
    fn engine_kind(&self) -> EngineKind {
        EngineKind::PostgreSQL
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
        let mut state = PostgresSSIState::new(isolation_level);
        state.read_only_optimization = self.read_only_opt;
        Box::new(state)
    }

    fn encode_constraints(
        &self,
        isolation_level: IsolationLevel,
        txn_count: usize,
        op_count: usize,
    ) -> IsoSpecResult<SmtConstraintSet> {
        let _ = isolation_level; // constraints are SSI-focused
        Ok(self.encode_ssi_constraints(txn_count, op_count))
    }

    fn version_string(&self) -> &str {
        &self.version
    }

    fn validate_schedule(
        &self,
        schedule: &isospec_types::schedule::Schedule,
        level: IsolationLevel,
    ) -> IsoSpecResult<ValidationResult> {
        let mut state = PostgresSSIState::new(level);
        state.read_only_optimization = self.read_only_opt;
        let mut aborted = Vec::new();

        // Begin all transactions first
        for &txn_id in &schedule.transaction_order {
            state.begin_transaction(txn_id, level)?;
        }

        // Execute steps
        for step in &schedule.steps {
            if state
                .txn_states
                .get(&step.txn_id)
                .map_or(true, |s| s.status == TransactionStatus::Aborted)
            {
                continue;
            }
            let outcome = state.execute_operation(&step.operation)?;
            if !outcome.success {
                aborted.push(step.txn_id);
            }
        }

        let deps = state.extract_dependencies();
        let violations: Vec<String> = aborted
            .iter()
            .map(|t| format!("Transaction {} aborted by SSI", t))
            .collect();
        Ok(ValidationResult {
            valid: violations.is_empty(),
            violations,
            schedule_accepted: aborted.is_empty(),
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
    fn test_siread_lock_set_basic() {
        let mut locks = SIReadLockSet::new(100);
        locks.add_tuple_lock(tid(1), tbl(), item(1), 10);
        assert_eq!(locks.count(), 1);
        let conflicts = locks.check_rw_conflict(tid(2), tbl(), item(1));
        assert_eq!(conflicts, vec![tid(1)]);
        let no_conflict = locks.check_rw_conflict(tid(2), tbl(), item(99));
        assert!(no_conflict.is_empty());
    }

    #[test]
    fn test_siread_lock_escalation() {
        let mut locks = SIReadLockSet::new(2);
        locks.add_tuple_lock(tid(1), tbl(), item(1), 10);
        locks.add_tuple_lock(tid(1), tbl(), item(2), 11);
        // Not yet escalated (threshold is 2, count is 2)
        locks.add_tuple_lock(tid(1), tbl(), item(3), 12);
        // Now escalated to relation
        let entry = locks.entries.iter().find(|e| {
            e.txn_id == tid(1) && e.granularity == SIReadGranularity::Relation
        });
        assert!(entry.is_some());
        // Relation lock conflicts with any item
        let conflicts = locks.check_rw_conflict(tid(2), tbl(), item(999));
        assert!(conflicts.contains(&tid(1)));
    }

    #[test]
    fn test_dangerous_structure_detection() {
        let mut det = DangerousStructureDetector::new();
        det.add_rw_edge(tid(1), tid(2));
        det.add_rw_edge(tid(2), tid(3));

        let mut txn_states = HashMap::new();
        let mut s1 = TransactionState::new(tid(1), IsolationLevel::Serializable, 1);
        s1.status = TransactionStatus::Committed;
        s1.commit_ts = Some(5);
        txn_states.insert(tid(1), s1);
        let s2 = TransactionState::new(tid(2), IsolationLevel::Serializable, 3);
        txn_states.insert(tid(2), s2);
        let s3 = TransactionState::new(tid(3), IsolationLevel::Serializable, 6);
        txn_states.insert(tid(3), s3);

        let mut snaps = HashMap::new();
        snaps.insert(tid(3), Snapshot::new(10, tid(3)));

        let structures = det.find_dangerous_structures(&txn_states, &snaps);
        assert!(!structures.is_empty());
        assert_eq!(structures[0], (tid(1), tid(2), tid(3)));
    }

    #[test]
    fn test_read_only_optimization_skips_abort() {
        let mut det = DangerousStructureDetector::new();
        det.add_rw_edge(tid(1), tid(2));
        det.add_rw_edge(tid(2), tid(3));

        let mut txn_states = HashMap::new();
        let mut s1 = TransactionState::new(tid(1), IsolationLevel::Serializable, 1);
        s1.status = TransactionStatus::Committed;
        s1.commit_ts = Some(5);
        txn_states.insert(tid(1), s1);
        let mut s2 = TransactionState::new(tid(2), IsolationLevel::Serializable, 0);
        s2.read_only = true;
        txn_states.insert(tid(2), s2);
        let s3 = TransactionState::new(tid(3), IsolationLevel::Serializable, 6);
        txn_states.insert(tid(3), s3);

        let mut snaps = HashMap::new();
        // T2's snapshot is at time 0, which precedes all writers
        snaps.insert(tid(2), Snapshot::new(0, tid(2)));
        snaps.insert(tid(3), Snapshot::new(10, tid(3)));

        let abort = det.should_abort_pivot(tid(2), &txn_states, &snaps, true);
        assert!(!abort);
    }

    #[test]
    fn test_postgres_state_begin_and_read() {
        let mut state = PostgresSSIState::new(IsolationLevel::Serializable);
        state.begin_transaction(tid(1), IsolationLevel::Serializable).unwrap();

        // Pre-populate a version
        state.version_store.write_version(
            item(1), tbl(), tid(0), 0, Value::Integer(42),
        );
        state.committed_set.insert(tid(0));

        let op = Operation::read(oid(0), tid(1), tbl(), item(1), 1);
        let outcome = state.execute_operation(&op).unwrap();
        assert!(outcome.success);
        assert_eq!(outcome.values_read, vec![Value::Integer(42)]);
    }

    #[test]
    fn test_postgres_state_write_creates_version() {
        let mut state = PostgresSSIState::new(IsolationLevel::Serializable);
        state.begin_transaction(tid(1), IsolationLevel::Serializable).unwrap();

        let op = Operation::write(oid(0), tid(1), tbl(), item(1), Value::Integer(99), 1);
        let outcome = state.execute_operation(&op).unwrap();
        assert!(outcome.success);
        assert_eq!(state.version_store.version_count(), 1);
    }

    #[test]
    fn test_postgres_state_commit_extracts_deps() {
        let mut state = PostgresSSIState::new(IsolationLevel::Serializable);
        state.begin_transaction(tid(1), IsolationLevel::Serializable).unwrap();
        state.begin_transaction(tid(2), IsolationLevel::Serializable).unwrap();

        // T1 writes item 1
        let w = Operation::write(oid(0), tid(1), tbl(), item(1), Value::Integer(10), 1);
        state.execute_operation(&w).unwrap();
        state.commit_transaction(tid(1)).unwrap();

        // T2 reads item 1 (should see T1's write since T1 committed)
        let r = Operation::read(oid(1), tid(2), tbl(), item(1), 5);
        state.execute_operation(&r).unwrap();

        let deps = state.extract_dependencies();
        // Should have at least a wr dependency T1 -> T2
        let has_wr = deps.iter().any(|d| {
            d.from_txn == tid(1)
                && d.to_txn == tid(2)
                && d.dep_type == DependencyType::WriteRead
        });
        assert!(has_wr);
    }

    #[test]
    fn test_postgres_model_traits() {
        let model = PostgresModel::new();
        assert_eq!(model.engine_kind(), EngineKind::PostgreSQL);
        assert_eq!(model.supported_isolation_levels().len(), 4);
        assert_eq!(model.version_string(), "16.0");
    }

    #[test]
    fn test_postgres_encode_constraints() {
        let model = PostgresModel::new();
        let cs = model
            .encode_constraints(IsolationLevel::Serializable, 3, 6)
            .unwrap();
        // Should have timestamp declarations and dangerous-structure assertions
        assert!(cs.variable_count() > 0);
        assert!(cs.constraint_count() > 0);
        let smtlib = cs.to_smtlib2();
        assert!(smtlib.contains("pg_start_0"));
        assert!(smtlib.contains("pg_rw_"));
    }

    #[test]
    fn test_postgres_state_reset() {
        let mut state = PostgresSSIState::new(IsolationLevel::Serializable);
        state.begin_transaction(tid(1), IsolationLevel::Serializable).unwrap();
        state.reset();
        assert!(state.active_transactions().is_empty());
        assert_eq!(state.version_store.version_count(), 0);
    }
}
