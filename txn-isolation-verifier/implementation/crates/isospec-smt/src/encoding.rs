//! Schedule encoding for bounded model checking via SMT.
//!
//! Encodes transaction schedules as SMT constraints so that a solver can
//! search for interleavings that exhibit (or are free from) isolation anomalies.

use std::collections::HashMap;

use isospec_types::constraint::{SmtConstraintSet, SmtExpr};
use isospec_types::error::{IsoSpecError, IsoSpecResult};
use isospec_types::identifier::{OperationId, TransactionId};
use isospec_types::isolation::IsolationLevel;
use isospec_types::operation::OpKind;

// ---------------------------------------------------------------------------
// Encoding parameters
// ---------------------------------------------------------------------------

/// Parameters controlling the size of the bounded model.
#[derive(Debug, Clone)]
pub struct EncodingBounds {
    /// Maximum number of transactions.
    pub max_transactions: usize,
    /// Maximum operations per transaction.
    pub max_ops_per_txn: usize,
    /// Maximum number of distinct data items.
    pub max_data_items: usize,
    /// Maximum integer value for data.
    pub max_value: i64,
    /// Whether to encode predicate-based operations.
    pub encode_predicates: bool,
}

impl Default for EncodingBounds {
    fn default() -> Self {
        Self {
            max_transactions: 4,
            max_ops_per_txn: 6,
            max_data_items: 8,
            max_value: 100,
            encode_predicates: false,
        }
    }
}

impl EncodingBounds {
    pub fn total_operations(&self) -> usize {
        self.max_transactions * self.max_ops_per_txn
    }
}

// ---------------------------------------------------------------------------
// Variable naming helpers
// ---------------------------------------------------------------------------

/// Generates standardized variable names for the SMT encoding.
#[derive(Debug, Clone)]
pub struct VarNaming {
    prefix: String,
}

impl VarNaming {
    pub fn new(prefix: &str) -> Self {
        Self {
            prefix: prefix.to_string(),
        }
    }

    /// Position variable for operation (txn, op_index) in the total order.
    pub fn position(&self, txn: usize, op: usize) -> String {
        format!("{}_pos_t{}_o{}", self.prefix, txn, op)
    }

    /// Whether operation (txn, op) is active (exists in the schedule).
    pub fn active(&self, txn: usize, op: usize) -> String {
        format!("{}_act_t{}_o{}", self.prefix, txn, op)
    }

    /// The data item accessed by operation (txn, op).
    pub fn item(&self, txn: usize, op: usize) -> String {
        format!("{}_item_t{}_o{}", self.prefix, txn, op)
    }

    /// The value read by operation (txn, op).
    pub fn read_value(&self, txn: usize, op: usize) -> String {
        format!("{}_rv_t{}_o{}", self.prefix, txn, op)
    }

    /// The value written by operation (txn, op).
    pub fn write_value(&self, txn: usize, op: usize) -> String {
        format!("{}_wv_t{}_o{}", self.prefix, txn, op)
    }

    /// Whether operation (txn, op) is a read.
    pub fn is_read(&self, txn: usize, op: usize) -> String {
        format!("{}_rd_t{}_o{}", self.prefix, txn, op)
    }

    /// Whether operation (txn, op) is a write.
    pub fn is_write(&self, txn: usize, op: usize) -> String {
        format!("{}_wr_t{}_o{}", self.prefix, txn, op)
    }

    /// Whether txn i reads from txn j on some item.
    pub fn reads_from(&self, reader_txn: usize, writer_txn: usize) -> String {
        format!("{}_rf_t{}_t{}", self.prefix, reader_txn, writer_txn)
    }

    /// Whether txn i committed.
    pub fn committed(&self, txn: usize) -> String {
        format!("{}_commit_t{}", self.prefix, txn)
    }

    /// Whether txn i aborted.
    pub fn aborted(&self, txn: usize) -> String {
        format!("{}_abort_t{}", self.prefix, txn)
    }

    /// Version order: whether write by txn i on item d is before write by txn j on item d.
    pub fn version_order(&self, txn_i: usize, txn_j: usize, item: usize) -> String {
        format!("{}_vo_t{}_t{}_d{}", self.prefix, txn_i, txn_j, item)
    }
}

impl Default for VarNaming {
    fn default() -> Self {
        Self::new("s")
    }
}

// ---------------------------------------------------------------------------
// ScheduleEncoder
// ---------------------------------------------------------------------------

/// Encodes a bounded schedule exploration problem as SMT constraints.
pub struct ScheduleEncoder {
    bounds: EncodingBounds,
    naming: VarNaming,
    constraints: SmtConstraintSet,
    /// Operation types keyed by (txn, op_idx).
    op_types: HashMap<(usize, usize), OpKind>,
    /// Required isolation level per transaction.
    isolation_levels: HashMap<usize, IsolationLevel>,
}

impl ScheduleEncoder {
    pub fn new(bounds: EncodingBounds) -> Self {
        Self {
            bounds,
            naming: VarNaming::default(),
            constraints: SmtConstraintSet::new("QF_UFLIA"),
            op_types: HashMap::new(),
            isolation_levels: HashMap::new(),
        }
    }

    pub fn with_naming(mut self, naming: VarNaming) -> Self {
        self.naming = naming;
        self
    }

    /// Set the operation type for a specific slot.
    pub fn set_op_type(&mut self, txn: usize, op: usize, kind: OpKind) {
        self.op_types.insert((txn, op), kind);
    }

    /// Set the required isolation level for a transaction.
    pub fn set_isolation(&mut self, txn: usize, level: IsolationLevel) {
        self.isolation_levels.insert(txn, level);
    }

    /// Build the full constraint set for bounded schedule exploration.
    pub fn encode(&mut self) -> IsoSpecResult<SmtConstraintSet> {
        self.encode_declarations();
        self.encode_position_constraints();
        self.encode_operation_type_constraints();
        self.encode_transaction_structure();
        self.encode_read_from_relations();
        self.encode_version_ordering();
        self.encode_data_flow_constraints();
        Ok(self.constraints.clone())
    }

    /// Declare all SMT variables.
    fn encode_declarations(&mut self) {
        let k = self.bounds.max_transactions;
        let n = self.bounds.max_ops_per_txn;

        for t in 0..k {
            self.constraints
                .add_declaration(self.naming.committed(t), "Bool".to_string());
            self.constraints
                .add_declaration(self.naming.aborted(t), "Bool".to_string());

            for o in 0..n {
                self.constraints
                    .add_declaration(self.naming.position(t, o), "Int".to_string());
                self.constraints
                    .add_declaration(self.naming.active(t, o), "Bool".to_string());
                self.constraints
                    .add_declaration(self.naming.item(t, o), "Int".to_string());
                self.constraints
                    .add_declaration(self.naming.read_value(t, o), "Int".to_string());
                self.constraints
                    .add_declaration(self.naming.write_value(t, o), "Int".to_string());
                self.constraints
                    .add_declaration(self.naming.is_read(t, o), "Bool".to_string());
                self.constraints
                    .add_declaration(self.naming.is_write(t, o), "Bool".to_string());
            }
        }

        // Read-from relations
        for i in 0..k {
            for j in 0..k {
                if i != j {
                    self.constraints
                        .add_declaration(self.naming.reads_from(i, j), "Bool".to_string());
                }
            }
        }

        // Version ordering for each pair of txns and each item
        for i in 0..k {
            for j in 0..k {
                if i != j {
                    for d in 0..self.bounds.max_data_items {
                        self.constraints.add_declaration(
                            self.naming.version_order(i, j, d),
                            "Bool".to_string(),
                        );
                    }
                }
            }
        }
    }

    /// Position variables define a total order on operations.
    fn encode_position_constraints(&mut self) {
        let k = self.bounds.max_transactions;
        let n = self.bounds.max_ops_per_txn;
        let total = self.bounds.total_operations();

        // Each active operation has a position in [0, total)
        for t in 0..k {
            for o in 0..n {
                let pos = SmtExpr::Const(self.naming.position(t, o));
                let act = SmtExpr::Const(self.naming.active(t, o));
                // active => 0 <= pos < total
                let lower = SmtExpr::Ge(Box::new(pos.clone()), Box::new(SmtExpr::IntLit(0)));
                let upper = SmtExpr::Lt(
                    Box::new(pos.clone()),
                    Box::new(SmtExpr::IntLit(total as i64)),
                );
                let bounded = SmtExpr::And(vec![lower, upper]);
                self.constraints
                    .add_assertion(SmtExpr::Implies(Box::new(act), Box::new(bounded)));
            }
        }

        // Intra-transaction ordering: ops within same txn are ordered by index
        for t in 0..k {
            for o in 0..n.saturating_sub(1) {
                let pos_curr = SmtExpr::Const(self.naming.position(t, o));
                let pos_next = SmtExpr::Const(self.naming.position(t, o + 1));
                let act_curr = SmtExpr::Const(self.naming.active(t, o));
                let act_next = SmtExpr::Const(self.naming.active(t, o + 1));
                // both active => pos_curr < pos_next
                let both_active = SmtExpr::And(vec![act_curr, act_next]);
                let ordered = SmtExpr::Lt(Box::new(pos_curr), Box::new(pos_next));
                self.constraints
                    .add_assertion(SmtExpr::Implies(Box::new(both_active), Box::new(ordered)));
            }
        }

        // All active operations have distinct positions
        let mut all_positions = Vec::new();
        for t in 0..k {
            for o in 0..n {
                all_positions.push((t, o));
            }
        }
        for i in 0..all_positions.len() {
            for j in (i + 1)..all_positions.len() {
                let (t1, o1) = all_positions[i];
                let (t2, o2) = all_positions[j];
                let pos1 = SmtExpr::Const(self.naming.position(t1, o1));
                let pos2 = SmtExpr::Const(self.naming.position(t2, o2));
                let act1 = SmtExpr::Const(self.naming.active(t1, o1));
                let act2 = SmtExpr::Const(self.naming.active(t2, o2));
                let both = SmtExpr::And(vec![act1, act2]);
                let diff = SmtExpr::Not(Box::new(SmtExpr::Eq(Box::new(pos1), Box::new(pos2))));
                self.constraints
                    .add_assertion(SmtExpr::Implies(Box::new(both), Box::new(diff)));
            }
        }
    }

    /// Encode operation type constraints from known op_types.
    fn encode_operation_type_constraints(&mut self) {
        let k = self.bounds.max_transactions;
        let n = self.bounds.max_ops_per_txn;

        for t in 0..k {
            for o in 0..n {
                let is_rd = SmtExpr::Const(self.naming.is_read(t, o));
                let is_wr = SmtExpr::Const(self.naming.is_write(t, o));
                let active = SmtExpr::Const(self.naming.active(t, o));

                if let Some(kind) = self.op_types.get(&(t, o)) {
                    match kind {
                        OpKind::Read(_) | OpKind::PredicateRead(_) => {
                            self.constraints.add_assertion(SmtExpr::Implies(
                                Box::new(active.clone()),
                                Box::new(is_rd.clone()),
                            ));
                            self.constraints.add_assertion(SmtExpr::Implies(
                                Box::new(active),
                                Box::new(SmtExpr::Not(Box::new(is_wr))),
                            ));
                        }
                        OpKind::Write(_) | OpKind::Insert(_) | OpKind::Delete(_) => {
                            self.constraints.add_assertion(SmtExpr::Implies(
                                Box::new(active.clone()),
                                Box::new(is_wr.clone()),
                            ));
                            self.constraints.add_assertion(SmtExpr::Implies(
                                Box::new(active),
                                Box::new(SmtExpr::Not(Box::new(is_rd))),
                            ));
                        }
                        _ => {
                            // Begin, Commit, Abort, Lock: neither read nor write
                            self.constraints.add_assertion(SmtExpr::Implies(
                                Box::new(active.clone()),
                                Box::new(SmtExpr::Not(Box::new(is_rd))),
                            ));
                            self.constraints.add_assertion(SmtExpr::Implies(
                                Box::new(active),
                                Box::new(SmtExpr::Not(Box::new(is_wr))),
                            ));
                        }
                    }
                } else {
                    // Unknown: at most one of read/write
                    let mutex = SmtExpr::Not(Box::new(SmtExpr::And(vec![
                        is_rd.clone(),
                        is_wr.clone(),
                    ])));
                    self.constraints.add_assertion(mutex);
                }
            }
        }
    }

    /// Transaction structure: committed XOR aborted, inactive ops after gap.
    fn encode_transaction_structure(&mut self) {
        let k = self.bounds.max_transactions;
        let n = self.bounds.max_ops_per_txn;

        for t in 0..k {
            let committed = SmtExpr::Const(self.naming.committed(t));
            let aborted = SmtExpr::Const(self.naming.aborted(t));

            // committed XOR aborted
            let xor = SmtExpr::And(vec![
                SmtExpr::Or(vec![committed.clone(), aborted.clone()]),
                SmtExpr::Not(Box::new(SmtExpr::And(vec![
                    committed.clone(),
                    aborted.clone(),
                ]))),
            ]);
            self.constraints.add_assertion(xor);

            // If op o is inactive, all subsequent ops must be inactive
            for o in 0..n.saturating_sub(1) {
                let act_curr = SmtExpr::Const(self.naming.active(t, o));
                let act_next = SmtExpr::Const(self.naming.active(t, o + 1));
                // !act_curr => !act_next
                let gap_rule = SmtExpr::Implies(
                    Box::new(SmtExpr::Not(Box::new(act_curr))),
                    Box::new(SmtExpr::Not(Box::new(act_next))),
                );
                self.constraints.add_assertion(gap_rule);
            }

            // Each transaction must have at least one active operation
            let at_least_one = SmtExpr::Const(self.naming.active(t, 0));
            self.constraints.add_assertion(at_least_one);
        }
    }

    /// Read-from relations: if txn i reads item d written by txn j,
    /// j's write must precede i's read in the schedule.
    fn encode_read_from_relations(&mut self) {
        let k = self.bounds.max_transactions;
        let n = self.bounds.max_ops_per_txn;

        for reader in 0..k {
            for writer in 0..k {
                if reader == writer {
                    continue;
                }
                let rf = SmtExpr::Const(self.naming.reads_from(reader, writer));

                // rf => exists ops r in reader, w in writer such that:
                //   is_read(r) && is_write(w) && item(r) == item(w)
                //   && pos(w) < pos(r) && read_value(r) == write_value(w)
                let mut matching_pairs = Vec::new();
                for ro in 0..n {
                    for wo in 0..n {
                        let r_read = SmtExpr::Const(self.naming.is_read(reader, ro));
                        let w_write = SmtExpr::Const(self.naming.is_write(writer, wo));
                        let r_act = SmtExpr::Const(self.naming.active(reader, ro));
                        let w_act = SmtExpr::Const(self.naming.active(writer, wo));
                        let same_item = SmtExpr::Eq(
                            Box::new(SmtExpr::Const(self.naming.item(reader, ro))),
                            Box::new(SmtExpr::Const(self.naming.item(writer, wo))),
                        );
                        let write_before_read = SmtExpr::Lt(
                            Box::new(SmtExpr::Const(self.naming.position(writer, wo))),
                            Box::new(SmtExpr::Const(self.naming.position(reader, ro))),
                        );
                        let value_match = SmtExpr::Eq(
                            Box::new(SmtExpr::Const(self.naming.read_value(reader, ro))),
                            Box::new(SmtExpr::Const(self.naming.write_value(writer, wo))),
                        );
                        matching_pairs.push(SmtExpr::And(vec![
                            r_read,
                            w_write,
                            r_act,
                            w_act,
                            same_item,
                            write_before_read,
                            value_match,
                        ]));
                    }
                }
                if !matching_pairs.is_empty() {
                    let exists_pair = SmtExpr::Or(matching_pairs);
                    self.constraints
                        .add_assertion(SmtExpr::Implies(Box::new(rf), Box::new(exists_pair)));
                }
            }
        }
    }

    /// Version ordering constraints for writes to the same item.
    fn encode_version_ordering(&mut self) {
        let k = self.bounds.max_transactions;
        let n = self.bounds.max_ops_per_txn;
        let d_max = self.bounds.max_data_items;

        for i in 0..k {
            for j in 0..k {
                if i == j {
                    continue;
                }
                for d in 0..d_max {
                    let vo_ij = SmtExpr::Const(self.naming.version_order(i, j, d));

                    // vo(i,j,d) => there exist writes w_i in txn i and w_j in txn j
                    // both on item d, and pos(w_i) < pos(w_j)
                    let mut write_pairs = Vec::new();
                    for oi in 0..n {
                        for oj in 0..n {
                            let wi = SmtExpr::Const(self.naming.is_write(i, oi));
                            let wj = SmtExpr::Const(self.naming.is_write(j, oj));
                            let ai = SmtExpr::Const(self.naming.active(i, oi));
                            let aj = SmtExpr::Const(self.naming.active(j, oj));
                            let item_i_is_d = SmtExpr::Eq(
                                Box::new(SmtExpr::Const(self.naming.item(i, oi))),
                                Box::new(SmtExpr::IntLit(d as i64)),
                            );
                            let item_j_is_d = SmtExpr::Eq(
                                Box::new(SmtExpr::Const(self.naming.item(j, oj))),
                                Box::new(SmtExpr::IntLit(d as i64)),
                            );
                            let i_before_j = SmtExpr::Lt(
                                Box::new(SmtExpr::Const(self.naming.position(i, oi))),
                                Box::new(SmtExpr::Const(self.naming.position(j, oj))),
                            );
                            write_pairs.push(SmtExpr::And(vec![
                                wi,
                                wj,
                                ai,
                                aj,
                                item_i_is_d,
                                item_j_is_d,
                                i_before_j,
                            ]));
                        }
                    }
                    if !write_pairs.is_empty() {
                        self.constraints.add_assertion(SmtExpr::Implies(
                            Box::new(vo_ij.clone()),
                            Box::new(SmtExpr::Or(write_pairs)),
                        ));
                    }

                    // Anti-symmetry: not (vo(i,j,d) and vo(j,i,d))
                    if i < j {
                        let vo_ji = SmtExpr::Const(self.naming.version_order(j, i, d));
                        self.constraints.add_assertion(SmtExpr::Not(Box::new(
                            SmtExpr::And(vec![vo_ij, vo_ji]),
                        )));
                    }
                }
            }
        }
    }

    /// Data flow constraints: item indices and values are bounded.
    fn encode_data_flow_constraints(&mut self) {
        let k = self.bounds.max_transactions;
        let n = self.bounds.max_ops_per_txn;
        let d_max = self.bounds.max_data_items as i64;
        let v_max = self.bounds.max_value;

        for t in 0..k {
            for o in 0..n {
                let active = SmtExpr::Const(self.naming.active(t, o));
                let item = SmtExpr::Const(self.naming.item(t, o));
                let rv = SmtExpr::Const(self.naming.read_value(t, o));
                let wv = SmtExpr::Const(self.naming.write_value(t, o));

                // item in [0, max_data_items)
                let item_lower =
                    SmtExpr::Ge(Box::new(item.clone()), Box::new(SmtExpr::IntLit(0)));
                let item_upper = SmtExpr::Lt(Box::new(item), Box::new(SmtExpr::IntLit(d_max)));
                self.constraints.add_assertion(SmtExpr::Implies(
                    Box::new(active.clone()),
                    Box::new(SmtExpr::And(vec![item_lower, item_upper])),
                ));

                // values in [0, max_value]
                let rv_bounded = SmtExpr::And(vec![
                    SmtExpr::Ge(Box::new(rv.clone()), Box::new(SmtExpr::IntLit(0))),
                    SmtExpr::Le(Box::new(rv), Box::new(SmtExpr::IntLit(v_max))),
                ]);
                let wv_bounded = SmtExpr::And(vec![
                    SmtExpr::Ge(Box::new(wv.clone()), Box::new(SmtExpr::IntLit(0))),
                    SmtExpr::Le(Box::new(wv), Box::new(SmtExpr::IntLit(v_max))),
                ]);
                self.constraints.add_assertion(SmtExpr::Implies(
                    Box::new(active.clone()),
                    Box::new(rv_bounded),
                ));
                self.constraints
                    .add_assertion(SmtExpr::Implies(Box::new(active), Box::new(wv_bounded)));
            }
        }
    }

    /// Return the number of constraints generated so far.
    pub fn constraint_count(&self) -> usize {
        self.constraints.assertions.len()
    }

    /// Return the number of declarations generated so far.
    pub fn declaration_count(&self) -> usize {
        self.constraints.declarations.len()
    }
}

// ---------------------------------------------------------------------------
// Anomaly constraint helpers
// ---------------------------------------------------------------------------

/// Adds constraints that assert the existence of a specific anomaly pattern.
pub struct AnomalyEncoder {
    naming: VarNaming,
}

impl AnomalyEncoder {
    pub fn new(naming: VarNaming) -> Self {
        Self { naming }
    }

    /// G0 (Dirty Write): Two txns write the same item, neither aborts,
    /// and the version order is inconsistent with the commit order.
    pub fn encode_g0(&self, txn_i: usize, txn_j: usize) -> Vec<SmtExpr> {
        let mut constraints = Vec::new();
        let ci = SmtExpr::Const(self.naming.committed(txn_i));
        let cj = SmtExpr::Const(self.naming.committed(txn_j));
        constraints.push(ci);
        constraints.push(cj);

        // There must exist writes by both txns on the same item
        // with conflicting version orders on two different items
        let vo_ij_0 = SmtExpr::Const(self.naming.version_order(txn_i, txn_j, 0));
        let vo_ji_1 = SmtExpr::Const(self.naming.version_order(txn_j, txn_i, 1));
        constraints.push(SmtExpr::And(vec![vo_ij_0, vo_ji_1]));

        constraints
    }

    /// G1a (Aborted Read): txn i reads data written by an aborted txn j.
    pub fn encode_g1a(&self, reader: usize, aborted_writer: usize) -> Vec<SmtExpr> {
        let mut constraints = Vec::new();
        let rf = SmtExpr::Const(self.naming.reads_from(reader, aborted_writer));
        let aborted = SmtExpr::Const(self.naming.aborted(aborted_writer));
        let reader_committed = SmtExpr::Const(self.naming.committed(reader));
        constraints.push(rf);
        constraints.push(aborted);
        constraints.push(reader_committed);
        constraints
    }

    /// G1b (Intermediate Read): txn i reads an intermediate value from txn j
    /// (not the final write).
    pub fn encode_g1b(&self, reader: usize, writer: usize, max_ops: usize) -> Vec<SmtExpr> {
        let mut constraints = Vec::new();
        let ci = SmtExpr::Const(self.naming.committed(reader));
        let cj = SmtExpr::Const(self.naming.committed(writer));
        constraints.push(ci);
        constraints.push(cj);

        // reader reads from a non-final write of writer
        // writer has at least 2 writes to the same item
        if max_ops >= 2 {
            let w0 = SmtExpr::Const(self.naming.is_write(writer, 0));
            let w1 = SmtExpr::Const(self.naming.is_write(writer, 1));
            let same_item = SmtExpr::Eq(
                Box::new(SmtExpr::Const(self.naming.item(writer, 0))),
                Box::new(SmtExpr::Const(self.naming.item(writer, 1))),
            );
            let both_active = SmtExpr::And(vec![
                SmtExpr::Const(self.naming.active(writer, 0)),
                SmtExpr::Const(self.naming.active(writer, 1)),
            ]);
            let reader_reads_first = SmtExpr::Eq(
                Box::new(SmtExpr::Const(self.naming.read_value(reader, 0))),
                Box::new(SmtExpr::Const(self.naming.write_value(writer, 0))),
            );
            constraints.push(SmtExpr::And(vec![
                w0,
                w1,
                same_item,
                both_active,
                reader_reads_first,
            ]));
        }

        constraints
    }

    /// G1c (Circular Information Flow): cycle in the "reads-from committed" graph.
    pub fn encode_g1c(&self, txn_i: usize, txn_j: usize) -> Vec<SmtExpr> {
        let mut constraints = Vec::new();
        let ci = SmtExpr::Const(self.naming.committed(txn_i));
        let cj = SmtExpr::Const(self.naming.committed(txn_j));
        let rf_ij = SmtExpr::Const(self.naming.reads_from(txn_i, txn_j));
        let rf_ji = SmtExpr::Const(self.naming.reads_from(txn_j, txn_i));
        constraints.push(ci);
        constraints.push(cj);
        constraints.push(rf_ij);
        constraints.push(rf_ji);
        constraints
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoding_bounds_defaults() {
        let bounds = EncodingBounds::default();
        assert_eq!(bounds.max_transactions, 4);
        assert_eq!(bounds.max_ops_per_txn, 6);
        assert_eq!(bounds.total_operations(), 24);
    }

    #[test]
    fn test_var_naming() {
        let naming = VarNaming::new("test");
        assert_eq!(naming.position(0, 1), "test_pos_t0_o1");
        assert_eq!(naming.active(2, 3), "test_act_t2_o3");
        assert_eq!(naming.reads_from(0, 1), "test_rf_t0_t1");
        assert_eq!(naming.version_order(1, 2, 0), "test_vo_t1_t2_d0");
    }

    #[test]
    fn test_small_encoding() {
        let bounds = EncodingBounds {
            max_transactions: 2,
            max_ops_per_txn: 2,
            max_data_items: 2,
            max_value: 10,
            encode_predicates: false,
        };
        let mut encoder = ScheduleEncoder::new(bounds);
        let cs = encoder.encode().unwrap();
        // Should have declarations and assertions
        assert!(cs.declarations.len() > 0);
        assert!(cs.assertions.len() > 0);
    }

    #[test]
    fn test_encoding_with_op_types() {
        let bounds = EncodingBounds {
            max_transactions: 2,
            max_ops_per_txn: 2,
            max_data_items: 2,
            max_value: 10,
            encode_predicates: false,
        };
        let mut encoder = ScheduleEncoder::new(bounds);
        encoder.set_op_type(
            0,
            0,
            OpKind::Read(isospec_types::operation::ReadOp {
                item_id: isospec_types::identifier::ItemId::new(0),
                value: None,
            }),
        );
        encoder.set_op_type(
            1,
            0,
            OpKind::Write(isospec_types::operation::WriteOp {
                item_id: isospec_types::identifier::ItemId::new(0),
                value: isospec_types::value::Value::Integer(1),
            }),
        );
        let cs = encoder.encode().unwrap();
        assert!(cs.assertions.len() > 10);
    }

    #[test]
    fn test_anomaly_encoder_g0() {
        let naming = VarNaming::default();
        let ae = AnomalyEncoder::new(naming);
        let g0 = ae.encode_g0(0, 1);
        assert_eq!(g0.len(), 3);
    }

    #[test]
    fn test_anomaly_encoder_g1a() {
        let naming = VarNaming::default();
        let ae = AnomalyEncoder::new(naming);
        let g1a = ae.encode_g1a(0, 1);
        assert_eq!(g1a.len(), 3);
    }

    #[test]
    fn test_anomaly_encoder_g1c() {
        let naming = VarNaming::default();
        let ae = AnomalyEncoder::new(naming);
        let g1c = ae.encode_g1c(0, 1);
        assert_eq!(g1c.len(), 4);
    }

    #[test]
    fn test_declaration_count() {
        let bounds = EncodingBounds {
            max_transactions: 2,
            max_ops_per_txn: 2,
            max_data_items: 2,
            max_value: 10,
            encode_predicates: false,
        };
        let mut encoder = ScheduleEncoder::new(bounds);
        encoder.encode().unwrap();
        // 2 txns * (committed + aborted) = 4
        // 2 txns * 2 ops * 7 vars = 28
        // 2 read-from = 2
        // 2 * 2 version_order = 4
        assert!(encoder.declaration_count() >= 30);
    }
}
