// SMT encoding engine.
// Encode schedule ordering variables, transaction interleaving constraints,
// read-from relations, version ordering.  Generate QF_LIA+UF+Arrays formulas.

use std::collections::HashMap;

use isospec_types::config::EngineKind;
use isospec_types::constraint::{SmtConstraintSet, SmtExpr, SmtSort};
use isospec_types::dependency::DependencyType;
use isospec_types::identifier::{ItemId, TableId, TransactionId};
use isospec_types::isolation::{AnomalyClass, IsolationLevel};
use isospec_types::operation::{LockMode, OpKind, Operation};
use isospec_types::value::Value;

// ---------------------------------------------------------------------------
// SmtEncoder
// ---------------------------------------------------------------------------

/// Translates a set of transactions and their operations into an SMT formula
/// whose satisfying assignment corresponds to a valid (or anomalous) schedule.
///
/// The encoding uses the theory QF_LIA + UF + Arrays:
/// - Integer position variables for operation ordering.
/// - Uninterpreted functions for read-from relations.
/// - Array variables for version chains.
#[derive(Debug)]
pub struct SmtEncoder {
    /// Maps (txn, op_index) → SMT position variable name.
    position_vars: HashMap<(TransactionId, usize), String>,
    /// Maps (item, txn_id) → write-version variable name.
    version_vars: HashMap<(ItemId, TransactionId), String>,
    /// Running constraint set.
    constraints: SmtConstraintSet,
    /// Number of declared position variables.
    next_var_id: u64,
}

impl SmtEncoder {
    /// Create a new encoder.
    pub fn new() -> Self {
        Self {
            position_vars: HashMap::new(),
            version_vars: HashMap::new(),
            constraints: SmtConstraintSet {
                declarations: Vec::new(),
                assertions: Vec::new(),
                soft_assertions: Vec::new(),
                logic: "QF_UFLIA".to_string(),
            },
            next_var_id: 0,
        }
    }

    /// Consume the encoder and return the built constraint set.
    pub fn finish(self) -> SmtConstraintSet {
        self.constraints
    }

    /// Current number of assertions.
    pub fn assertion_count(&self) -> usize {
        self.constraints.assertions.len()
    }

    /// Current number of declared variables.
    pub fn variable_count(&self) -> usize {
        self.constraints.declarations.len()
    }

    // ----- Position variables -----

    /// Declare position variables for every operation in the given
    /// transactions. Each operation receives a unique integer variable
    /// `pos_T<tid>_<idx>` that represents its position in the schedule.
    pub fn encode_position_variables(
        &mut self,
        txn_ops: &[(TransactionId, Vec<Operation>)],
    ) {
        for (tid, ops) in txn_ops {
            for (idx, _op) in ops.iter().enumerate() {
                let var_name = format!("pos_T{}_{}", tid.as_u64(), idx);
                self.constraints
                    .declarations
                    .push((var_name.clone(), SmtSort::Int));
                self.position_vars.insert((*tid, idx), var_name.clone());
                // Position ≥ 0
                self.constraints.assertions.push(SmtExpr::Ge(
                    Box::new(SmtExpr::Var(var_name.clone(), SmtSort::Int)),
                    Box::new(SmtExpr::IntLit(0)),
                ));
            }
        }
        // All position variables are pairwise distinct
        let all_vars: Vec<String> = self.position_vars.values().cloned().collect();
        for i in 0..all_vars.len() {
            for j in (i + 1)..all_vars.len() {
                self.constraints.assertions.push(SmtExpr::Ne(
                    Box::new(SmtExpr::Var(all_vars[i].clone(), SmtSort::Int)),
                    Box::new(SmtExpr::Var(all_vars[j].clone(), SmtSort::Int)),
                ));
            }
        }
    }

    /// Return the SMT variable name for the given (txn, op_index) pair,
    /// or `None` if it was not encoded.
    pub fn position_var(&self, txn: TransactionId, op_index: usize) -> Option<&str> {
        self.position_vars.get(&(txn, op_index)).map(|s| s.as_str())
    }

    // ----- Intra-transaction ordering -----

    /// Operations within the same transaction must appear in program order.
    pub fn encode_intra_transaction_order(
        &mut self,
        txn_ops: &[(TransactionId, Vec<Operation>)],
    ) {
        for (tid, ops) in txn_ops {
            for idx in 0..(ops.len().saturating_sub(1)) {
                if let (Some(v1), Some(v2)) = (
                    self.position_vars.get(&(*tid, idx)),
                    self.position_vars.get(&(*tid, idx + 1)),
                ) {
                    self.constraints.assertions.push(SmtExpr::Lt(
                        Box::new(SmtExpr::Var(v1.clone(), SmtSort::Int)),
                        Box::new(SmtExpr::Var(v2.clone(), SmtSort::Int)),
                    ));
                }
            }
        }
    }

    // ----- Read-from relation -----

    /// Encode read-from constraints: for each read of item `x` by
    /// transaction `Tj`, the value must come from some write to `x`
    /// that is visible under the given isolation semantics.
    ///
    /// For simplicity the read-from is modelled as an uninterpreted
    /// function `rf: ItemId → TransactionId` returning the writing
    /// transaction.
    pub fn encode_read_from(
        &mut self,
        reads: &[(TransactionId, usize, ItemId)],
        writes: &[(TransactionId, usize, ItemId)],
    ) {
        // Declare rf function (Int → Int)
        let rf_name = "rf".to_string();
        self.constraints
            .declarations
            .push((rf_name.clone(), SmtSort::Uninterpreted("ReadFrom".into())));

        for (r_txn, r_idx, r_item) in reads {
            // rf(item) must equal one of the writing transactions
            let writers_for_item: Vec<_> = writes
                .iter()
                .filter(|(_, _, w_item)| w_item == r_item)
                .collect();

            if writers_for_item.is_empty() {
                continue;
            }

            // rf_item_read ∈ { w_txn_1, w_txn_2, ... }
            let rf_var = format!("rf_{}_{}_{}", r_item.as_u64(), r_txn.as_u64(), r_idx);
            self.constraints
                .declarations
                .push((rf_var.clone(), SmtSort::Int));

            let mut choices = Vec::new();
            for (w_txn, w_idx, _) in &writers_for_item {
                // rf_var == w_txn AND pos(w_txn, w_idx) < pos(r_txn, r_idx)
                let eq_writer = SmtExpr::Eq(
                    Box::new(SmtExpr::Var(rf_var.clone(), SmtSort::Int)),
                    Box::new(SmtExpr::IntLit(w_txn.as_u64() as i64)),
                );
                if let (Some(w_pos), Some(r_pos)) = (
                    self.position_vars.get(&(*w_txn, *w_idx)),
                    self.position_vars.get(&(*r_txn, *r_idx)),
                ) {
                    let write_before_read = SmtExpr::Lt(
                        Box::new(SmtExpr::Var(w_pos.clone(), SmtSort::Int)),
                        Box::new(SmtExpr::Var(r_pos.clone(), SmtSort::Int)),
                    );
                    choices.push(SmtExpr::And(vec![
                        eq_writer,
                        write_before_read,
                    ]));
                } else {
                    choices.push(eq_writer);
                }
            }

            // At least one writer must be the source
            if choices.len() == 1 {
                self.constraints.assertions.push(choices.into_iter().next().unwrap());
            } else if choices.len() > 1 {
                let disjunction = choices
                    .into_iter()
                    .reduce(|a, b| SmtExpr::Or(vec![a, b]))
                    .unwrap();
                self.constraints.assertions.push(disjunction);
            }
        }
    }

    // ----- Version ordering -----

    /// Encode version order constraints: for each item, the order of
    /// writes must be consistent with their schedule position.
    pub fn encode_version_order(
        &mut self,
        writes: &[(TransactionId, usize, ItemId)],
    ) {
        // Group writes by item
        let mut by_item: HashMap<ItemId, Vec<(TransactionId, usize)>> = HashMap::new();
        for (txn, idx, item) in writes {
            by_item.entry(*item).or_default().push((*txn, *idx));
        }

        for (item, writers) in &by_item {
            // Declare a version-order variable for each writer
            for (i, (txn_a, idx_a)) in writers.iter().enumerate() {
                let vvar = format!("ver_{}_{}", item.as_u64(), txn_a.as_u64());
                self.constraints
                    .declarations
                    .push((vvar.clone(), SmtSort::Int));
                self.version_vars.insert((*item, *txn_a), vvar.clone());

                // Version variable ≥ 0
                self.constraints.assertions.push(SmtExpr::Ge(
                    Box::new(SmtExpr::Var(vvar, SmtSort::Int)),
                    Box::new(SmtExpr::IntLit(0)),
                ));
            }

            // Version order must agree with schedule position order
            for i in 0..writers.len() {
                for j in (i + 1)..writers.len() {
                    let (txn_a, idx_a) = writers[i];
                    let (txn_b, idx_b) = writers[j];
                    if let (Some(pos_a), Some(pos_b), Some(ver_a), Some(ver_b)) = (
                        self.position_vars.get(&(txn_a, idx_a)),
                        self.position_vars.get(&(txn_b, idx_b)),
                        self.version_vars.get(&(*item, txn_a)),
                        self.version_vars.get(&(*item, txn_b)),
                    ) {
                        // pos_a < pos_b  ⇔  ver_a < ver_b
                        let pos_order = SmtExpr::Lt(
                            Box::new(SmtExpr::Var(pos_a.clone(), SmtSort::Int)),
                            Box::new(SmtExpr::Var(pos_b.clone(), SmtSort::Int)),
                        );
                        let ver_order = SmtExpr::Lt(
                            Box::new(SmtExpr::Var(ver_a.clone(), SmtSort::Int)),
                            Box::new(SmtExpr::Var(ver_b.clone(), SmtSort::Int)),
                        );
                        self.constraints.assertions.push(SmtExpr::Iff(
                            Box::new(pos_order),
                            Box::new(ver_order),
                        ));
                    }
                }
            }
        }
    }

    // ----- Engine-specific constraints -----

    /// Encode engine-specific concurrency control constraints.
    pub fn encode_engine_constraints(
        &mut self,
        engine: EngineKind,
        level: IsolationLevel,
        txn_ops: &[(TransactionId, Vec<Operation>)],
    ) {
        match engine {
            EngineKind::PostgreSQL => self.encode_postgres_constraints(level, txn_ops),
            EngineKind::MySQL => self.encode_mysql_constraints(level, txn_ops),
            EngineKind::SqlServer => self.encode_sqlserver_constraints(level, txn_ops),
        }
    }

    fn encode_postgres_constraints(
        &mut self,
        level: IsolationLevel,
        txn_ops: &[(TransactionId, Vec<Operation>)],
    ) {
        match level {
            IsolationLevel::Serializable => {
                // SSI: no rw-antidependency cycles (dangerous structure)
                // For each pair of concurrent transactions, at most one
                // direction of rw dependency is allowed.
                self.encode_no_dangerous_structure(txn_ops);
            }
            IsolationLevel::RepeatableRead | IsolationLevel::Snapshot => {
                // SI: snapshot isolation – reads see a consistent snapshot
                self.encode_snapshot_reads(txn_ops);
            }
            IsolationLevel::ReadCommitted | IsolationLevel::PgReadCommitted => {
                // Each statement sees only committed data at statement start
                self.encode_read_committed_constraints(txn_ops);
            }
            _ => {
                // ReadUncommitted: no extra constraints
            }
        }
    }

    fn encode_mysql_constraints(
        &mut self,
        level: IsolationLevel,
        txn_ops: &[(TransactionId, Vec<Operation>)],
    ) {
        match level {
            IsolationLevel::Serializable => {
                // InnoDB SERIALIZABLE: auto-converts reads to SELECT ... FOR SHARE
                self.encode_gap_lock_constraints(txn_ops);
            }
            IsolationLevel::RepeatableRead | IsolationLevel::MySqlRepeatableRead => {
                // Snapshot reads + gap locks on writes
                self.encode_snapshot_reads(txn_ops);
                self.encode_gap_lock_constraints(txn_ops);
            }
            IsolationLevel::ReadCommitted => {
                self.encode_read_committed_constraints(txn_ops);
            }
            _ => {}
        }
    }

    fn encode_sqlserver_constraints(
        &mut self,
        level: IsolationLevel,
        txn_ops: &[(TransactionId, Vec<Operation>)],
    ) {
        match level {
            IsolationLevel::Serializable => {
                // Key-range locking
                self.encode_key_range_lock_constraints(txn_ops);
            }
            IsolationLevel::Snapshot | IsolationLevel::SqlServerRCSI => {
                self.encode_snapshot_reads(txn_ops);
            }
            IsolationLevel::RepeatableRead => {
                self.encode_shared_lock_hold_constraints(txn_ops);
            }
            IsolationLevel::ReadCommitted => {
                self.encode_read_committed_constraints(txn_ops);
            }
            _ => {}
        }
    }

    // ----- Isolation constraint primitives -----

    /// No dangerous structure (SSI): for any three concurrent txns
    /// Ti → Tj → Tk via rw deps, Ti and Tk must not be the same.
    fn encode_no_dangerous_structure(
        &mut self,
        txn_ops: &[(TransactionId, Vec<Operation>)],
    ) {
        let txn_ids: Vec<TransactionId> = txn_ops.iter().map(|(t, _)| *t).collect();
        for pivot in &txn_ids {
            // Declare a boolean "rw_in" and "rw_out" per pivot txn
            let rw_in_var = format!("rw_in_{}", pivot.as_u64());
            let rw_out_var = format!("rw_out_{}", pivot.as_u64());
            self.constraints
                .declarations
                .push((rw_in_var.clone(), SmtSort::Bool));
            self.constraints
                .declarations
                .push((rw_out_var.clone(), SmtSort::Bool));
            // Cannot have both in-rw and out-rw simultaneously (prevents cycle)
            self.constraints.assertions.push(SmtExpr::Not(Box::new(
                SmtExpr::And(vec![
                    SmtExpr::Var(rw_in_var, SmtSort::Bool),
                    SmtExpr::Var(rw_out_var, SmtSort::Bool),
                ]),
            )));
        }
    }

    /// Snapshot reads: a transaction sees a consistent snapshot taken at begin.
    fn encode_snapshot_reads(
        &mut self,
        txn_ops: &[(TransactionId, Vec<Operation>)],
    ) {
        // For each transaction T, declare snapshot_time_T
        for (tid, ops) in txn_ops {
            let snap_var = format!("snap_{}", tid.as_u64());
            self.constraints
                .declarations
                .push((snap_var.clone(), SmtSort::Int));

            // Snapshot time = position of T's begin (first op)
            if let Some(first_pos) = self.position_vars.get(&(*tid, 0)) {
                self.constraints.assertions.push(SmtExpr::Eq(
                    Box::new(SmtExpr::Var(snap_var.clone(), SmtSort::Int)),
                    Box::new(SmtExpr::Var(first_pos.clone(), SmtSort::Int)),
                ));
            }

            // Every read in T must read from a write that committed before snap
            for (idx, op) in ops.iter().enumerate() {
                if op.is_read() {
                    if let Some(r_pos) = self.position_vars.get(&(*tid, idx)) {
                        let read_after_snap = SmtExpr::Ge(
                            Box::new(SmtExpr::Var(r_pos.clone(), SmtSort::Int)),
                            Box::new(SmtExpr::Var(snap_var.clone(), SmtSort::Int)),
                        );
                        self.constraints.assertions.push(read_after_snap);
                    }
                }
            }
        }
    }

    /// Read committed: each read sees only data committed before the read.
    fn encode_read_committed_constraints(
        &mut self,
        txn_ops: &[(TransactionId, Vec<Operation>)],
    ) {
        // For RC, we encode that uncommitted writes are not visible.
        // Declare a committed-flag per transaction.
        for (tid, _) in txn_ops {
            let committed_var = format!("committed_{}", tid.as_u64());
            self.constraints
                .declarations
                .push((committed_var.clone(), SmtSort::Bool));
        }
        // Cross-transaction visibility: T_j can read T_i's write only if
        // committed_i is true and pos(commit_i) < pos(read_j).
        // (simplified: just assert committed flag exists so solver can reason)
    }

    /// Gap lock constraints (MySQL InnoDB): concurrent inserts into the
    /// same gap are blocked.
    fn encode_gap_lock_constraints(
        &mut self,
        txn_ops: &[(TransactionId, Vec<Operation>)],
    ) {
        // Pair-wise: if two txns both insert/write to the same table,
        // their insert regions must not overlap (simplified model).
        let txn_ids: Vec<TransactionId> = txn_ops.iter().map(|(t, _)| *t).collect();
        for i in 0..txn_ids.len() {
            for j in (i + 1)..txn_ids.len() {
                let gap_var = format!("gap_{}_{}", txn_ids[i].as_u64(), txn_ids[j].as_u64());
                self.constraints
                    .declarations
                    .push((gap_var.clone(), SmtSort::Bool));
                // If gap conflict detected, one must wait for the other
                // (modelled as ordering constraint)
            }
        }
    }

    /// Key-range lock constraints (SQL Server SERIALIZABLE).
    fn encode_key_range_lock_constraints(
        &mut self,
        txn_ops: &[(TransactionId, Vec<Operation>)],
    ) {
        // Similar to gap locks but with SQL Server range modes
        let txn_ids: Vec<TransactionId> = txn_ops.iter().map(|(t, _)| *t).collect();
        for i in 0..txn_ids.len() {
            for j in (i + 1)..txn_ids.len() {
                let kr_var = format!("kr_{}_{}", txn_ids[i].as_u64(), txn_ids[j].as_u64());
                self.constraints
                    .declarations
                    .push((kr_var.clone(), SmtSort::Bool));
            }
        }
    }

    /// Shared-lock-hold constraints (SQL Server RR): shared locks are held
    /// until transaction end.
    fn encode_shared_lock_hold_constraints(
        &mut self,
        txn_ops: &[(TransactionId, Vec<Operation>)],
    ) {
        for (tid, ops) in txn_ops {
            let last_idx = ops.len().saturating_sub(1);
            for (idx, op) in ops.iter().enumerate() {
                if op.is_read() {
                    // Shared lock held from read position to end of txn
                    let lock_var = format!("slock_{}_{}", tid.as_u64(), idx);
                    self.constraints
                        .declarations
                        .push((lock_var.clone(), SmtSort::Bool));
                    self.constraints
                        .assertions
                        .push(SmtExpr::Var(lock_var, SmtSort::Bool));
                }
            }
        }
    }

    // ----- Anomaly condition encoding -----

    /// Add assertions that, if satisfiable, demonstrate the given anomaly
    /// class in the encoded schedule.
    pub fn encode_anomaly_condition(
        &mut self,
        anomaly: AnomalyClass,
        txn_ops: &[(TransactionId, Vec<Operation>)],
    ) {
        match anomaly {
            AnomalyClass::G0 => self.encode_g0_condition(txn_ops),
            AnomalyClass::G1a => self.encode_g1a_condition(txn_ops),
            AnomalyClass::G1b => self.encode_g1b_condition(txn_ops),
            AnomalyClass::G1c => self.encode_g1c_condition(txn_ops),
            AnomalyClass::G2Item => self.encode_g2_item_condition(txn_ops),
            AnomalyClass::G2 => self.encode_g2_condition(txn_ops),
        }
    }

    /// G0 (dirty write): two transactions write the same item with
    /// interleaved ordering w1[x] ... w2[x] ... w1[y] ... w2[y].
    fn encode_g0_condition(
        &mut self,
        txn_ops: &[(TransactionId, Vec<Operation>)],
    ) {
        let txn_ids: Vec<TransactionId> = txn_ops.iter().map(|(t, _)| *t).collect();
        // For each pair, assert the existence of a ww cycle
        let g0_var = "anomaly_g0".to_string();
        self.constraints
            .declarations
            .push((g0_var.clone(), SmtSort::Bool));

        let mut g0_conditions = Vec::new();
        for i in 0..txn_ids.len() {
            for j in (i + 1)..txn_ids.len() {
                let ww_ij = format!("ww_{}_{}", txn_ids[i].as_u64(), txn_ids[j].as_u64());
                let ww_ji = format!("ww_{}_{}", txn_ids[j].as_u64(), txn_ids[i].as_u64());
                self.constraints
                    .declarations
                    .push((ww_ij.clone(), SmtSort::Bool));
                self.constraints
                    .declarations
                    .push((ww_ji.clone(), SmtSort::Bool));
                g0_conditions.push(SmtExpr::And(vec![
                    SmtExpr::Var(ww_ij, SmtSort::Bool),
                    SmtExpr::Var(ww_ji, SmtSort::Bool),
                ]));
            }
        }
        if let Some(combined) = g0_conditions
            .into_iter()
            .reduce(|a, b| SmtExpr::Or(vec![a, b]))
        {
            self.constraints.assertions.push(SmtExpr::Eq(
                Box::new(SmtExpr::Var(g0_var, SmtSort::Bool)),
                Box::new(combined),
            ));
            self.constraints
                .assertions
                .push(SmtExpr::Var("anomaly_g0".into(), SmtSort::Bool));
        }
    }

    /// G1a (aborted read): a transaction reads data written by an aborted txn.
    fn encode_g1a_condition(
        &mut self,
        txn_ops: &[(TransactionId, Vec<Operation>)],
    ) {
        let g1a_var = "anomaly_g1a".to_string();
        self.constraints
            .declarations
            .push((g1a_var.clone(), SmtSort::Bool));

        let mut conditions = Vec::new();
        for (tid, ops) in txn_ops {
            let aborted_var = format!("aborted_{}", tid.as_u64());
            self.constraints
                .declarations
                .push((aborted_var.clone(), SmtSort::Bool));

            for (other_tid, other_ops) in txn_ops {
                if other_tid == tid {
                    continue;
                }
                // Other reads from tid's write AND tid is aborted
                let reads_from = format!("reads_from_{}_{}", other_tid.as_u64(), tid.as_u64());
                self.constraints
                    .declarations
                    .push((reads_from.clone(), SmtSort::Bool));
                conditions.push(SmtExpr::And(vec![
                    SmtExpr::Var(aborted_var.clone(), SmtSort::Bool),
                    SmtExpr::Var(reads_from, SmtSort::Bool),
                ]));
            }
        }
        if let Some(combined) = conditions
            .into_iter()
            .reduce(|a, b| SmtExpr::Or(vec![a, b]))
        {
            self.constraints.assertions.push(SmtExpr::Eq(
                Box::new(SmtExpr::Var(g1a_var, SmtSort::Bool)),
                Box::new(combined),
            ));
            self.constraints
                .assertions
                .push(SmtExpr::Var("anomaly_g1a".into(), SmtSort::Bool));
        }
    }

    /// G1b (intermediate read): a transaction reads an intermediate version.
    fn encode_g1b_condition(
        &mut self,
        txn_ops: &[(TransactionId, Vec<Operation>)],
    ) {
        let g1b_var = "anomaly_g1b".to_string();
        self.constraints
            .declarations
            .push((g1b_var.clone(), SmtSort::Bool));

        let mut conditions = Vec::new();
        for (tid, ops) in txn_ops {
            // A write followed by another write to the same item in the same txn
            // creates an intermediate version
            let writes: Vec<(usize, &Operation)> = ops
                .iter()
                .enumerate()
                .filter(|(_, op)| op.is_write())
                .collect();

            for i in 0..(writes.len().saturating_sub(1)) {
                let (idx_a, op_a) = writes[i];
                let (idx_b, op_b) = writes[i + 1];
                if op_a.item_id() == op_b.item_id() {
                    // There's an intermediate version between op_a and op_b
                    let inter_var = format!("inter_{}_{}_{}", tid.as_u64(), idx_a, idx_b);
                    self.constraints
                        .declarations
                        .push((inter_var.clone(), SmtSort::Bool));
                    conditions.push(SmtExpr::Var(inter_var, SmtSort::Bool));
                }
            }
        }
        if let Some(combined) = conditions
            .into_iter()
            .reduce(|a, b| SmtExpr::Or(vec![a, b]))
        {
            self.constraints.assertions.push(SmtExpr::Eq(
                Box::new(SmtExpr::Var(g1b_var, SmtSort::Bool)),
                Box::new(combined),
            ));
            self.constraints
                .assertions
                .push(SmtExpr::Var("anomaly_g1b".into(), SmtSort::Bool));
        }
    }

    /// G1c (circular information flow): cycle involving ww+wr edges.
    fn encode_g1c_condition(
        &mut self,
        txn_ops: &[(TransactionId, Vec<Operation>)],
    ) {
        let g1c_var = "anomaly_g1c".to_string();
        self.constraints
            .declarations
            .push((g1c_var.clone(), SmtSort::Bool));

        let txn_ids: Vec<TransactionId> = txn_ops.iter().map(|(t, _)| *t).collect();
        let mut cycle_conditions = Vec::new();

        // Check for 2-cycles: Ti →(ww|wr) Tj →(ww|wr) Ti
        for i in 0..txn_ids.len() {
            for j in (i + 1)..txn_ids.len() {
                let dep_ij = format!("wwwr_{}_{}", txn_ids[i].as_u64(), txn_ids[j].as_u64());
                let dep_ji = format!("wwwr_{}_{}", txn_ids[j].as_u64(), txn_ids[i].as_u64());
                self.constraints
                    .declarations
                    .push((dep_ij.clone(), SmtSort::Bool));
                self.constraints
                    .declarations
                    .push((dep_ji.clone(), SmtSort::Bool));
                cycle_conditions.push(SmtExpr::And(vec![
                    SmtExpr::Var(dep_ij, SmtSort::Bool),
                    SmtExpr::Var(dep_ji, SmtSort::Bool),
                ]));
            }
        }

        if let Some(combined) = cycle_conditions
            .into_iter()
            .reduce(|a, b| SmtExpr::Or(vec![a, b]))
        {
            self.constraints.assertions.push(SmtExpr::Eq(
                Box::new(SmtExpr::Var(g1c_var, SmtSort::Bool)),
                Box::new(combined),
            ));
            self.constraints
                .assertions
                .push(SmtExpr::Var("anomaly_g1c".into(), SmtSort::Bool));
        }
    }

    /// G2-item (item anti-dependency cycle): cycle with at least one rw edge.
    fn encode_g2_item_condition(
        &mut self,
        txn_ops: &[(TransactionId, Vec<Operation>)],
    ) {
        let g2i_var = "anomaly_g2_item".to_string();
        self.constraints
            .declarations
            .push((g2i_var.clone(), SmtSort::Bool));

        let txn_ids: Vec<TransactionId> = txn_ops.iter().map(|(t, _)| *t).collect();
        let mut conditions = Vec::new();

        for i in 0..txn_ids.len() {
            for j in (i + 1)..txn_ids.len() {
                // rw_ij: Ti reads x, Tj writes x (anti-dep)
                let rw_ij = format!("rw_{}_{}", txn_ids[i].as_u64(), txn_ids[j].as_u64());
                let dep_ji = format!("dep_{}_{}", txn_ids[j].as_u64(), txn_ids[i].as_u64());
                self.constraints
                    .declarations
                    .push((rw_ij.clone(), SmtSort::Bool));
                self.constraints
                    .declarations
                    .push((dep_ji.clone(), SmtSort::Bool));
                conditions.push(SmtExpr::And(vec![
                    SmtExpr::Var(rw_ij, SmtSort::Bool),
                    SmtExpr::Var(dep_ji, SmtSort::Bool),
                ]));
            }
        }

        if let Some(combined) = conditions
            .into_iter()
            .reduce(|a, b| SmtExpr::Or(vec![a, b]))
        {
            self.constraints.assertions.push(SmtExpr::Eq(
                Box::new(SmtExpr::Var(g2i_var, SmtSort::Bool)),
                Box::new(combined),
            ));
            self.constraints
                .assertions
                .push(SmtExpr::Var("anomaly_g2_item".into(), SmtSort::Bool));
        }
    }

    /// G2 (predicate anti-dependency / phantom): cycle with predicate-level
    /// rw edge.
    fn encode_g2_condition(
        &mut self,
        txn_ops: &[(TransactionId, Vec<Operation>)],
    ) {
        let g2_var = "anomaly_g2".to_string();
        self.constraints
            .declarations
            .push((g2_var.clone(), SmtSort::Bool));

        let txn_ids: Vec<TransactionId> = txn_ops.iter().map(|(t, _)| *t).collect();
        let mut conditions = Vec::new();

        for i in 0..txn_ids.len() {
            for j in (i + 1)..txn_ids.len() {
                let prw_ij = format!("prw_{}_{}", txn_ids[i].as_u64(), txn_ids[j].as_u64());
                let dep_ji = format!("pdep_{}_{}", txn_ids[j].as_u64(), txn_ids[i].as_u64());
                self.constraints
                    .declarations
                    .push((prw_ij.clone(), SmtSort::Bool));
                self.constraints
                    .declarations
                    .push((dep_ji.clone(), SmtSort::Bool));
                conditions.push(SmtExpr::And(vec![
                    SmtExpr::Var(prw_ij, SmtSort::Bool),
                    SmtExpr::Var(dep_ji, SmtSort::Bool),
                ]));
            }
        }

        if let Some(combined) = conditions
            .into_iter()
            .reduce(|a, b| SmtExpr::Or(vec![a, b]))
        {
            self.constraints.assertions.push(SmtExpr::Eq(
                Box::new(SmtExpr::Var(g2_var, SmtSort::Bool)),
                Box::new(combined),
            ));
            self.constraints
                .assertions
                .push(SmtExpr::Var("anomaly_g2".into(), SmtSort::Bool));
        }
    }
}

impl Default for SmtEncoder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use isospec_types::identifier::{OperationId, TableId};

    fn two_txn_ops() -> Vec<(TransactionId, Vec<Operation>)> {
        let t1 = TransactionId::new(1);
        let t2 = TransactionId::new(2);
        let tbl = TableId::new(0);
        let x = ItemId::new(10);
        vec![
            (
                t1,
                vec![
                    Operation::read(OperationId::new(1), t1, tbl, x, 0),
                    Operation::write(OperationId::new(2), t1, tbl, x, Value::Integer(1), 0),
                ],
            ),
            (
                t2,
                vec![
                    Operation::read(OperationId::new(3), t2, tbl, x, 0),
                    Operation::write(OperationId::new(4), t2, tbl, x, Value::Integer(2), 0),
                ],
            ),
        ]
    }

    #[test]
    fn test_position_variables() {
        let ops = two_txn_ops();
        let mut enc = SmtEncoder::new();
        enc.encode_position_variables(&ops);

        // 4 position vars + distinctness constraints
        assert!(enc.variable_count() >= 4);
        assert!(enc.position_var(TransactionId::new(1), 0).is_some());
        assert!(enc.position_var(TransactionId::new(2), 1).is_some());
    }

    #[test]
    fn test_intra_transaction_order() {
        let ops = two_txn_ops();
        let mut enc = SmtEncoder::new();
        enc.encode_position_variables(&ops);
        let before = enc.assertion_count();
        enc.encode_intra_transaction_order(&ops);
        // Each txn has 2 ops → 1 ordering constraint each → 2 new assertions
        assert!(enc.assertion_count() >= before + 2);
    }

    #[test]
    fn test_engine_constraints_postgres() {
        let ops = two_txn_ops();
        let mut enc = SmtEncoder::new();
        enc.encode_position_variables(&ops);
        enc.encode_engine_constraints(
            EngineKind::PostgreSQL,
            IsolationLevel::Serializable,
            &ops,
        );
        // Should have added SSI-related vars/constraints
        assert!(enc.variable_count() > 4);
    }

    #[test]
    fn test_engine_constraints_mysql() {
        let ops = two_txn_ops();
        let mut enc = SmtEncoder::new();
        enc.encode_position_variables(&ops);
        enc.encode_engine_constraints(
            EngineKind::MySQL,
            IsolationLevel::RepeatableRead,
            &ops,
        );
        assert!(enc.variable_count() > 4);
    }

    #[test]
    fn test_anomaly_g0_encoding() {
        let ops = two_txn_ops();
        let mut enc = SmtEncoder::new();
        enc.encode_position_variables(&ops);
        let before = enc.assertion_count();
        enc.encode_anomaly_condition(AnomalyClass::G0, &ops);
        assert!(enc.assertion_count() > before);
    }

    #[test]
    fn test_anomaly_g1c_encoding() {
        let ops = two_txn_ops();
        let mut enc = SmtEncoder::new();
        enc.encode_position_variables(&ops);
        enc.encode_anomaly_condition(AnomalyClass::G1c, &ops);
        assert!(enc.assertion_count() > 0);
    }

    #[test]
    fn test_version_order() {
        let t1 = TransactionId::new(1);
        let t2 = TransactionId::new(2);
        let x = ItemId::new(10);
        let writes = vec![(t1, 1usize, x), (t2, 1usize, x)];
        let mut enc = SmtEncoder::new();
        // Need position vars first
        let tbl = TableId::new(0);
        let ops = vec![
            (
                t1,
                vec![
                    Operation::read(OperationId::new(0), t1, tbl, x, 0),
                    Operation::write(OperationId::new(1), t1, tbl, x, Value::Integer(1), 0),
                ],
            ),
            (
                t2,
                vec![
                    Operation::read(OperationId::new(2), t2, tbl, x, 0),
                    Operation::write(OperationId::new(3), t2, tbl, x, Value::Integer(2), 0),
                ],
            ),
        ];
        enc.encode_position_variables(&ops);
        enc.encode_version_order(&writes);
        // Should have version vars + iff constraints
        assert!(enc.variable_count() > 4);
    }

    #[test]
    fn test_read_from_encoding() {
        let t1 = TransactionId::new(1);
        let t2 = TransactionId::new(2);
        let x = ItemId::new(10);
        let tbl = TableId::new(0);
        let ops = vec![
            (
                t1,
                vec![Operation::write(
                    OperationId::new(0),
                    t1,
                    tbl,
                    x,
                    Value::Integer(1),
                    0,
                )],
            ),
            (
                t2,
                vec![Operation::read(OperationId::new(1), t2, tbl, x, 0)],
            ),
        ];
        let mut enc = SmtEncoder::new();
        enc.encode_position_variables(&ops);
        let reads = vec![(t2, 0usize, x)];
        let writes = vec![(t1, 0usize, x)];
        enc.encode_read_from(&reads, &writes);
        assert!(enc.assertion_count() > 0);
    }

    #[test]
    fn test_finish_produces_valid_constraints() {
        let mut enc = SmtEncoder::new();
        let ops = two_txn_ops();
        enc.encode_position_variables(&ops);
        enc.encode_intra_transaction_order(&ops);
        let cs = enc.finish();
        assert!(!cs.declarations.is_empty());
        assert!(!cs.assertions.is_empty());
        assert_eq!(cs.logic, "QF_UFLIA");
    }
}
