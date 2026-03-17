// Conflict detection and resolution.
// ConflictMatrix, lock compatibility, predicate-level conflict, gap lock analysis.

use std::collections::{HashMap, HashSet};

use isospec_types::identifier::{ItemId, TableId, TransactionId};
use isospec_types::isolation::IsolationLevel;
use isospec_types::operation::{LockMode, OpKind, Operation};
use isospec_types::predicate::{ColumnRef, ComparisonOp, Predicate};
use isospec_types::value::Value;

// ---------------------------------------------------------------------------
// Lock compatibility
// ---------------------------------------------------------------------------

/// Full lock-mode compatibility matrix.
/// Returns `true` when `held` and `requested` can coexist on the same item.
pub fn locks_compatible(held: LockMode, requested: LockMode) -> bool {
    use LockMode::*;
    match (held, requested) {
        // Shared ↔ Shared is always fine
        (Shared, Shared) => true,
        (Shared, IntentShared) | (IntentShared, Shared) => true,
        (IntentShared, IntentShared) => true,
        (IntentShared, IntentExclusive) | (IntentExclusive, IntentShared) => true,
        (Shared, Update) => true,
        (Update, Shared) => false,
        // Intent-exclusive with intent-shared
        (IntentExclusive, IntentExclusive) => true,
        // Exclusive is never compatible with anything except IS
        (Exclusive, IntentShared) | (IntentShared, Exclusive) => false,
        (Exclusive, _) | (_, Exclusive) => false,
        // Update blocks update
        (Update, Update) => false,
        (Update, IntentShared) | (IntentShared, Update) => true,
        (Update, IntentExclusive) | (IntentExclusive, Update) => false,
        // SIRead (predicate read lock in SSI) is compatible with everything
        // except exclusive writes that modify the predicate range
        (SIRead, SIRead) => true,
        (SIRead, Shared) | (Shared, SIRead) => true,
        (SIRead, Exclusive) | (Exclusive, SIRead) => false,
        (SIRead, _) | (_, SIRead) => true,
        // Key-range lock modes
        (RangeSharedShared, RangeSharedShared) => true,
        (RangeSharedShared, RangeSharedUpdate) => true,
        (RangeSharedUpdate, RangeSharedShared) => true,
        (RangeSharedUpdate, RangeSharedUpdate) => false,
        (RangeExclusiveExclusive, _) | (_, RangeExclusiveExclusive) => false,
        (RangeInsertNull, RangeInsertNull) => true,
        (RangeInsertNull, RangeSharedShared) | (RangeSharedShared, RangeInsertNull) => true,
        (RangeInsertNull, _) | (_, RangeInsertNull) => false,
        // Shared vs intent-exclusive
        (Shared, IntentExclusive) | (IntentExclusive, Shared) => false,
        // Remaining range vs non-range
        (RangeSharedShared, _) | (_, RangeSharedShared) => false,
        (RangeSharedUpdate, _) | (_, RangeSharedUpdate) => false,
    }
}

// ---------------------------------------------------------------------------
// ConflictKind / ConflictInfo
// ---------------------------------------------------------------------------

/// Classification of a pairwise conflict.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ConflictKind {
    WriteWrite,
    WriteRead,
    ReadWrite,
    PredicateConflict,
    GapLockConflict,
    LockIncompatibility,
}

/// Details about a detected conflict.
#[derive(Debug, Clone)]
pub struct ConflictInfo {
    pub kind: ConflictKind,
    pub txn_a: TransactionId,
    pub txn_b: TransactionId,
    pub item: Option<ItemId>,
    pub table: Option<TableId>,
    pub description: String,
}

// ---------------------------------------------------------------------------
// ConflictMatrix – pairwise conflict summary
// ---------------------------------------------------------------------------

/// Conflict matrix recording all conflicts among a set of transactions.
#[derive(Debug)]
pub struct ConflictMatrix {
    txn_ids: Vec<TransactionId>,
    /// (i, j) → set of conflict kinds between txn_ids[i] and txn_ids[j].
    matrix: HashMap<(usize, usize), Vec<ConflictInfo>>,
}

impl ConflictMatrix {
    /// Build a conflict matrix from a slice of operations grouped by
    /// transaction (each inner `Vec` is one transaction's ops in order).
    pub fn build(txn_ops: &[(TransactionId, Vec<Operation>)]) -> Self {
        let txn_ids: Vec<TransactionId> = txn_ops.iter().map(|(id, _)| *id).collect();
        let mut matrix: HashMap<(usize, usize), Vec<ConflictInfo>> = HashMap::new();

        for i in 0..txn_ops.len() {
            for j in (i + 1)..txn_ops.len() {
                let conflicts = detect_pairwise_conflicts(
                    txn_ops[i].0,
                    &txn_ops[i].1,
                    txn_ops[j].0,
                    &txn_ops[j].1,
                );
                if !conflicts.is_empty() {
                    matrix.insert((i, j), conflicts);
                }
            }
        }

        Self { txn_ids, matrix }
    }

    /// All conflicts between transactions at indices `i` and `j`.
    pub fn conflicts_between(&self, i: usize, j: usize) -> &[ConflictInfo] {
        let key = if i < j { (i, j) } else { (j, i) };
        self.matrix.get(&key).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Whether any conflict exists between two transactions.
    pub fn has_conflict(&self, i: usize, j: usize) -> bool {
        !self.conflicts_between(i, j).is_empty()
    }

    /// All unique conflict kinds across the whole matrix.
    pub fn all_conflict_kinds(&self) -> HashSet<ConflictKind> {
        self.matrix
            .values()
            .flat_map(|v| v.iter().map(|c| c.kind.clone()))
            .collect()
    }

    /// Transaction identifiers in matrix order.
    pub fn transaction_ids(&self) -> &[TransactionId] {
        &self.txn_ids
    }

    /// Total number of conflicts in the matrix.
    pub fn total_conflicts(&self) -> usize {
        self.matrix.values().map(|v| v.len()).sum()
    }

    /// Whether the matrix is fully conflict-free.
    pub fn is_conflict_free(&self) -> bool {
        self.matrix.is_empty()
    }

    /// Density of the conflict matrix (fraction of pairs with ≥1 conflict).
    pub fn density(&self) -> f64 {
        let n = self.txn_ids.len();
        if n < 2 {
            return 0.0;
        }
        let total_pairs = n * (n - 1) / 2;
        self.matrix.len() as f64 / total_pairs as f64
    }
}

// ---------------------------------------------------------------------------
// Pairwise operation-level conflict detection
// ---------------------------------------------------------------------------

fn detect_pairwise_conflicts(
    txn_a: TransactionId,
    ops_a: &[Operation],
    txn_b: TransactionId,
    ops_b: &[Operation],
) -> Vec<ConflictInfo> {
    let mut conflicts = Vec::new();

    for oa in ops_a {
        for ob in ops_b {
            // Only data operations create conflicts
            if !oa.is_data_operation() || !ob.is_data_operation() {
                continue;
            }
            // Must touch the same item (if both have item ids)
            let same_item = match (oa.item_id(), ob.item_id()) {
                (Some(a), Some(b)) => a == b,
                _ => false,
            };
            let same_table = match (oa.table_id(), ob.table_id()) {
                (Some(a), Some(b)) => a == b,
                _ => false,
            };
            if !same_item && !same_table {
                continue;
            }
            // WW
            if oa.is_write() && ob.is_write() && same_item {
                conflicts.push(ConflictInfo {
                    kind: ConflictKind::WriteWrite,
                    txn_a,
                    txn_b,
                    item: oa.item_id(),
                    table: oa.table_id(),
                    description: format!(
                        "WW conflict on {:?} between {} and {}",
                        oa.item_id(),
                        txn_a,
                        txn_b
                    ),
                });
            }
            // WR
            if oa.is_write() && ob.is_read() && same_item {
                conflicts.push(ConflictInfo {
                    kind: ConflictKind::WriteRead,
                    txn_a,
                    txn_b,
                    item: oa.item_id(),
                    table: oa.table_id(),
                    description: format!(
                        "WR conflict on {:?} between {} and {}",
                        oa.item_id(),
                        txn_a,
                        txn_b
                    ),
                });
            }
            // RW
            if oa.is_read() && ob.is_write() && same_item {
                conflicts.push(ConflictInfo {
                    kind: ConflictKind::ReadWrite,
                    txn_a,
                    txn_b,
                    item: oa.item_id(),
                    table: oa.table_id(),
                    description: format!(
                        "RW conflict on {:?} between {} and {}",
                        oa.item_id(),
                        txn_a,
                        txn_b
                    ),
                });
            }
            // Predicate-level: at least one is a predicate operation, same table
            if same_table
                && (matches!(oa.kind, OpKind::PredicateRead(_) | OpKind::PredicateWrite(_))
                    || matches!(
                        ob.kind,
                        OpKind::PredicateRead(_) | OpKind::PredicateWrite(_)
                    ))
            {
                let p1 = oa.predicate();
                let p2 = ob.predicate();
                if let (Some(pred_a), Some(pred_b)) = (p1, p2) {
                    if predicates_may_conflict(pred_a, pred_b) {
                        conflicts.push(ConflictInfo {
                            kind: ConflictKind::PredicateConflict,
                            txn_a,
                            txn_b,
                            item: None,
                            table: oa.table_id(),
                            description: format!(
                                "Predicate conflict on table {:?} between {} and {}",
                                oa.table_id(),
                                txn_a,
                                txn_b
                            ),
                        });
                    }
                }
            }
        }
    }

    conflicts
}

// ---------------------------------------------------------------------------
// Predicate conflict detection
// ---------------------------------------------------------------------------

/// Conservative check: can two predicates select overlapping rows?
pub fn predicates_may_conflict(p1: &Predicate, p2: &Predicate) -> bool {
    // If either is True/False we can decide trivially
    if matches!(p1, Predicate::False) || matches!(p2, Predicate::False) {
        return false;
    }
    if matches!(p1, Predicate::True) || matches!(p2, Predicate::True) {
        return true;
    }
    // Try interval-based check: extract per-column intervals and check overlap
    let intervals_a = p1.to_interval_constraints();
    let intervals_b = p2.to_interval_constraints();

    // If we cannot extract intervals, conservatively say they may conflict
    if intervals_a.is_empty() && intervals_b.is_empty() {
        return true;
    }

    // For each column that both predicates constrain, check overlap
    for (col, int_a) in &intervals_a {
        if let Some(int_b) = intervals_b.get(col) {
            if !int_a.overlaps(int_b) {
                return false;
            }
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Gap lock conflict analysis (InnoDB-style)
// ---------------------------------------------------------------------------

/// Represents a key range for gap lock analysis.
#[derive(Debug, Clone)]
pub struct KeyRange {
    pub table: TableId,
    pub lower: Option<Value>,
    pub upper: Option<Value>,
    pub lower_inclusive: bool,
    pub upper_inclusive: bool,
}

impl KeyRange {
    /// Create an unbounded range covering all keys in a table.
    pub fn all(table: TableId) -> Self {
        Self {
            table,
            lower: None,
            upper: None,
            lower_inclusive: false,
            upper_inclusive: false,
        }
    }

    /// Create a point range (single key).
    pub fn point(table: TableId, key: Value) -> Self {
        Self {
            table,
            lower: Some(key.clone()),
            upper: Some(key),
            lower_inclusive: true,
            upper_inclusive: true,
        }
    }

    /// Create a bounded range.
    pub fn bounded(
        table: TableId,
        lower: Value,
        upper: Value,
        lower_inclusive: bool,
        upper_inclusive: bool,
    ) -> Self {
        Self {
            table,
            lower: Some(lower),
            upper: Some(upper),
            lower_inclusive,
            upper_inclusive,
        }
    }

    /// Whether two ranges on the same table may overlap.
    pub fn overlaps(&self, other: &KeyRange) -> bool {
        if self.table != other.table {
            return false;
        }
        // If either is unbounded, they overlap
        let (s_lo, s_hi) = (&self.lower, &self.upper);
        let (o_lo, o_hi) = (&other.lower, &other.upper);

        // self.upper < other.lower → no overlap
        if let (Some(s_hi_val), Some(o_lo_val)) = (s_hi, o_lo) {
            if let Some(ord) = s_hi_val.compare(o_lo_val) {
                match ord {
                    std::cmp::Ordering::Less => return false,
                    std::cmp::Ordering::Equal => {
                        if !self.upper_inclusive || !other.lower_inclusive {
                            return false;
                        }
                    }
                    _ => {}
                }
            }
        }
        // other.upper < self.lower → no overlap
        if let (Some(o_hi_val), Some(s_lo_val)) = (o_hi, s_lo) {
            if let Some(ord) = o_hi_val.compare(s_lo_val) {
                match ord {
                    std::cmp::Ordering::Less => return false,
                    std::cmp::Ordering::Equal => {
                        if !other.upper_inclusive || !self.lower_inclusive {
                            return false;
                        }
                    }
                    _ => {}
                }
            }
        }
        true
    }

    /// Whether this range is a single point.
    pub fn is_point(&self) -> bool {
        match (&self.lower, &self.upper) {
            (Some(lo), Some(hi)) => {
                self.lower_inclusive
                    && self.upper_inclusive
                    && lo.compare(hi) == Some(std::cmp::Ordering::Equal)
            }
            _ => false,
        }
    }
}

/// Held gap lock.
#[derive(Debug, Clone)]
pub struct GapLock {
    pub txn_id: TransactionId,
    pub range: KeyRange,
    pub mode: LockMode,
}

/// Analyser for gap-lock conflicts (MySQL/InnoDB next-key locking).
#[derive(Debug)]
pub struct GapLockAnalyzer {
    locks: Vec<GapLock>,
}

impl GapLockAnalyzer {
    pub fn new() -> Self {
        Self { locks: Vec::new() }
    }

    /// Record a gap lock.
    pub fn add_lock(&mut self, lock: GapLock) {
        self.locks.push(lock);
    }

    /// Find all conflicting lock pairs.
    pub fn find_conflicts(&self) -> Vec<ConflictInfo> {
        let mut conflicts = Vec::new();
        for i in 0..self.locks.len() {
            for j in (i + 1)..self.locks.len() {
                let la = &self.locks[i];
                let lb = &self.locks[j];
                if la.txn_id == lb.txn_id {
                    continue;
                }
                if !la.range.overlaps(&lb.range) {
                    continue;
                }
                if !locks_compatible(la.mode, lb.mode) {
                    conflicts.push(ConflictInfo {
                        kind: ConflictKind::GapLockConflict,
                        txn_a: la.txn_id,
                        txn_b: lb.txn_id,
                        item: None,
                        table: Some(la.range.table),
                        description: format!(
                            "Gap lock conflict: {} ({:?}) vs {} ({:?}) on table {:?}",
                            la.txn_id, la.mode, lb.txn_id, lb.mode, la.range.table
                        ),
                    });
                }
            }
        }
        conflicts
    }

    /// Number of tracked gap locks.
    pub fn lock_count(&self) -> usize {
        self.locks.len()
    }

    /// Clear all locks.
    pub fn clear(&mut self) {
        self.locks.clear();
    }
}

impl Default for GapLockAnalyzer {
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
    use isospec_types::identifier::{ItemId, OperationId, TableId, TransactionId};
    use isospec_types::operation::Operation;

    fn make_write(txn: TransactionId, table: TableId, item: ItemId) -> Operation {
        Operation::write(
            OperationId::new(0),
            txn,
            table,
            item,
            Value::Integer(1),
            0,
        )
    }

    fn make_read(txn: TransactionId, table: TableId, item: ItemId) -> Operation {
        Operation::read(OperationId::new(0), txn, table, item, 0)
    }

    #[test]
    fn test_lock_compatibility_shared() {
        assert!(locks_compatible(LockMode::Shared, LockMode::Shared));
        assert!(!locks_compatible(LockMode::Shared, LockMode::Exclusive));
        assert!(!locks_compatible(LockMode::Exclusive, LockMode::Exclusive));
    }

    #[test]
    fn test_lock_compatibility_intent() {
        assert!(locks_compatible(LockMode::IntentShared, LockMode::IntentShared));
        assert!(locks_compatible(
            LockMode::IntentShared,
            LockMode::IntentExclusive
        ));
        assert!(locks_compatible(
            LockMode::IntentExclusive,
            LockMode::IntentExclusive
        ));
    }

    #[test]
    fn test_conflict_matrix_basic() {
        let t1 = TransactionId::new(1);
        let t2 = TransactionId::new(2);
        let tbl = TableId::new(0);
        let item = ItemId::new(10);

        let ops1 = vec![make_write(t1, tbl, item)];
        let ops2 = vec![make_read(t2, tbl, item)];
        let matrix = ConflictMatrix::build(&[(t1, ops1), (t2, ops2)]);

        assert!(!matrix.is_conflict_free());
        assert!(matrix.has_conflict(0, 1));
        assert!(matrix
            .all_conflict_kinds()
            .contains(&ConflictKind::WriteRead));
    }

    #[test]
    fn test_conflict_matrix_no_conflict() {
        let t1 = TransactionId::new(1);
        let t2 = TransactionId::new(2);
        let tbl = TableId::new(0);
        let item_a = ItemId::new(10);
        let item_b = ItemId::new(20);

        let ops1 = vec![make_write(t1, tbl, item_a)];
        let ops2 = vec![make_write(t2, tbl, item_b)];
        let matrix = ConflictMatrix::build(&[(t1, ops1), (t2, ops2)]);

        assert!(matrix.is_conflict_free());
    }

    #[test]
    fn test_predicate_conflict_true() {
        assert!(predicates_may_conflict(&Predicate::True, &Predicate::True));
        assert!(!predicates_may_conflict(
            &Predicate::False,
            &Predicate::True
        ));
    }

    #[test]
    fn test_key_range_overlap() {
        let tbl = TableId::new(0);
        let r1 = KeyRange::bounded(
            tbl,
            Value::Integer(1),
            Value::Integer(10),
            true,
            true,
        );
        let r2 = KeyRange::bounded(
            tbl,
            Value::Integer(5),
            Value::Integer(15),
            true,
            true,
        );
        let r3 = KeyRange::bounded(
            tbl,
            Value::Integer(11),
            Value::Integer(20),
            true,
            true,
        );
        assert!(r1.overlaps(&r2));
        assert!(!r1.overlaps(&r3));
    }

    #[test]
    fn test_gap_lock_analyzer() {
        let tbl = TableId::new(0);
        let mut analyzer = GapLockAnalyzer::new();
        analyzer.add_lock(GapLock {
            txn_id: TransactionId::new(1),
            range: KeyRange::bounded(
                tbl,
                Value::Integer(1),
                Value::Integer(10),
                true,
                true,
            ),
            mode: LockMode::Exclusive,
        });
        analyzer.add_lock(GapLock {
            txn_id: TransactionId::new(2),
            range: KeyRange::bounded(
                tbl,
                Value::Integer(5),
                Value::Integer(15),
                true,
                true,
            ),
            mode: LockMode::Exclusive,
        });
        let conflicts = analyzer.find_conflicts();
        assert_eq!(conflicts.len(), 1);
        assert_eq!(conflicts[0].kind, ConflictKind::GapLockConflict);
    }

    #[test]
    fn test_key_range_point() {
        let tbl = TableId::new(0);
        let pt = KeyRange::point(tbl, Value::Integer(42));
        assert!(pt.is_point());
        let rng = KeyRange::bounded(
            tbl,
            Value::Integer(1),
            Value::Integer(10),
            true,
            true,
        );
        assert!(!rng.is_point());
    }
}
