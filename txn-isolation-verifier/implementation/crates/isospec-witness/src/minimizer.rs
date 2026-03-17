//! Witness minimization via iterative deletion-based MUS extraction.
//!
//! Produces a minimal witness by removing operations one at a time and
//! re-checking if the anomaly is still present.

use std::collections::HashSet;
use std::fmt;
use std::time::{Duration, Instant};

use isospec_types::error::{IsoSpecError, IsoSpecResult};
use isospec_types::identifier::{OperationId, TransactionId};
use isospec_types::isolation::AnomalyClass;
use isospec_types::operation::OpKind;
use isospec_types::schedule::{Schedule, ScheduleMetadata, ScheduleStep};

// ---------------------------------------------------------------------------
// MinimizerConfig
// ---------------------------------------------------------------------------

/// Configuration for the witness minimizer.
#[derive(Debug, Clone)]
pub struct MinimizerConfig {
    /// Maximum number of minimization iterations.
    pub max_iterations: usize,
    /// Timeout for the entire minimization process.
    pub timeout: Duration,
    /// Whether to try removing entire transactions (not just operations).
    pub try_transaction_removal: bool,
    /// Whether to attempt reordering operations within transactions.
    pub try_reordering: bool,
}

impl Default for MinimizerConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            timeout: Duration::from_secs(60),
            try_transaction_removal: true,
            try_reordering: false,
        }
    }
}

// ---------------------------------------------------------------------------
// AnomalyChecker trait
// ---------------------------------------------------------------------------

/// Trait for checking whether a schedule exhibits a given anomaly.
/// Abstracts the SMT solver so we can test minimization independently.
pub trait AnomalyChecker: Send {
    fn check(&mut self, schedule: &Schedule, anomaly: &AnomalyClass) -> IsoSpecResult<bool>;
}

/// A simple structural anomaly checker that verifies schedule properties
/// without invoking an SMT solver.
pub struct StructuralAnomalyChecker;

impl StructuralAnomalyChecker {
    pub fn new() -> Self {
        Self
    }

    /// Check if the schedule has conflicting operations (reads and writes
    /// on the same item from different transactions).
    fn has_rw_conflict(schedule: &Schedule) -> bool {
        for i in 0..schedule.steps.len() {
            for j in (i + 1)..schedule.steps.len() {
                let a = &schedule.steps[i];
                let b = &schedule.steps[j];
                if a.operation.txn_id == b.operation.txn_id {
                    continue;
                }
                let a_item = Self::op_item(&a.operation.kind);
                let b_item = Self::op_item(&b.operation.kind);
                if a_item.is_some() && a_item == b_item {
                    let a_write = Self::is_write_op(&a.operation.kind);
                    let b_write = Self::is_write_op(&b.operation.kind);
                    if a_write || b_write {
                        return true;
                    }
                }
            }
        }
        false
    }

    fn op_item(kind: &OpKind) -> Option<u64> {
        match kind {
            OpKind::Read(r) => Some(r.item.as_u64()),
            OpKind::Write(w) => Some(w.item.as_u64()),
            OpKind::Insert(i) => Some(i.item.as_u64()),
            OpKind::Delete(d) => d.deleted_items.first().map(|id| id.as_u64()),
            OpKind::Lock(l) => l.item.map(|id| id.as_u64()),
            _ => None,
        }
    }

    fn is_write_op(kind: &OpKind) -> bool {
        matches!(kind, OpKind::Write(_) | OpKind::Insert(_) | OpKind::Delete(_))
    }
}

impl AnomalyChecker for StructuralAnomalyChecker {
    fn check(&mut self, schedule: &Schedule, anomaly: &AnomalyClass) -> IsoSpecResult<bool> {
        // Minimal structural check: there must be at least 2 transactions
        // with conflicting operations for any anomaly to exist.
        let txn_ids: HashSet<_> = schedule
            .steps
            .iter()
            .map(|s| s.operation.txn_id)
            .collect();
        if txn_ids.len() < 2 {
            return Ok(false);
        }

        // Check for conflicting operations
        let has_conflict = Self::has_rw_conflict(schedule);
        if !has_conflict {
            return Ok(false);
        }

        // For structural checking, if there are conflicts we assume the anomaly
        // is still possible. A real implementation would re-invoke the SMT solver.
        Ok(true)
    }
}

// ---------------------------------------------------------------------------
// MinimizationResult
// ---------------------------------------------------------------------------

/// Result of the minimization process.
#[derive(Debug, Clone)]
pub struct MinimizationResult {
    /// The minimized schedule.
    pub schedule: Schedule,
    /// Number of operations removed.
    pub ops_removed: usize,
    /// Number of transactions removed.
    pub txns_removed: usize,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Total time spent.
    pub elapsed: Duration,
    /// Whether the minimization completed within the budget.
    pub completed: bool,
    /// Original size (operations).
    pub original_size: usize,
}

impl MinimizationResult {
    pub fn reduction_ratio(&self) -> f64 {
        if self.original_size == 0 {
            0.0
        } else {
            self.ops_removed as f64 / self.original_size as f64
        }
    }
}

impl fmt::Display for MinimizationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "minimized: {} -> {} ops ({} removed, {:.0}% reduction, {} iters, {:.2}s)",
            self.original_size,
            self.schedule.steps.len(),
            self.ops_removed,
            self.reduction_ratio() * 100.0,
            self.iterations,
            self.elapsed.as_secs_f64(),
        )
    }
}

// ---------------------------------------------------------------------------
// MUSExtractor
// ---------------------------------------------------------------------------

/// Extracts a Minimal Unsatisfiable Subset (MUS) of operations that still
/// demonstrates the anomaly, using iterative deletion.
pub struct MUSExtractor<C: AnomalyChecker> {
    checker: C,
    config: MinimizerConfig,
}

impl<C: AnomalyChecker> MUSExtractor<C> {
    pub fn new(checker: C, config: MinimizerConfig) -> Self {
        Self { checker, config }
    }

    /// Minimize a schedule by iteratively removing operations.
    pub fn minimize(
        &mut self,
        schedule: &Schedule,
        anomaly: &AnomalyClass,
    ) -> IsoSpecResult<MinimizationResult> {
        let start = Instant::now();
        let original_size = schedule.steps.len();
        let mut current = schedule.clone();
        let mut iterations = 0;
        let mut txns_removed = 0;

        // Phase 1: Try removing entire transactions
        if self.config.try_transaction_removal {
            let txn_ids: Vec<TransactionId> = current.transaction_ids();
            for txn_id in &txn_ids {
                if start.elapsed() > self.config.timeout {
                    break;
                }
                let candidate = remove_transaction(&current, txn_id);
                if candidate.steps.len() < 2 {
                    continue;
                }
                iterations += 1;
                if self.checker.check(&candidate, anomaly)? {
                    txns_removed += 1;
                    current = candidate;
                }
            }
        }

        // Phase 2: Try removing individual operations (in reverse order)
        let mut changed = true;
        while changed && iterations < self.config.max_iterations {
            if start.elapsed() > self.config.timeout {
                break;
            }
            changed = false;

            let step_count = current.steps.len();
            for idx in (0..step_count).rev() {
                if start.elapsed() > self.config.timeout || iterations >= self.config.max_iterations
                {
                    break;
                }
                iterations += 1;

                let candidate = remove_step(&current, idx);
                // Must keep at least one op per transaction
                if !all_txns_have_ops(&candidate) {
                    continue;
                }
                if self.checker.check(&candidate, anomaly)? {
                    current = candidate;
                    changed = true;
                    break; // restart from end
                }
            }
        }

        // Phase 3: Compact positions
        current = compact_positions(&current);

        let ops_removed = original_size - current.steps.len();
        let completed = iterations < self.config.max_iterations
            && start.elapsed() <= self.config.timeout;

        Ok(MinimizationResult {
            schedule: current,
            ops_removed,
            txns_removed,
            iterations,
            elapsed: start.elapsed(),
            completed,
            original_size,
        })
    }
}

/// Remove all operations belonging to a transaction from a schedule.
fn remove_transaction(schedule: &Schedule, txn_id: &TransactionId) -> Schedule {
    let steps: Vec<ScheduleStep> = schedule
        .steps
        .iter()
        .filter(|s| s.operation.txn_id != *txn_id)
        .cloned()
        .collect();
    let transaction_order: Vec<TransactionId> = schedule
        .transaction_order
        .iter()
        .filter(|t| *t != txn_id)
        .copied()
        .collect();
    Schedule {
        steps,
        transaction_order,
        metadata: schedule.metadata.clone(),
    }
}

/// Remove a single step by index.
fn remove_step(schedule: &Schedule, idx: usize) -> Schedule {
    let mut steps = schedule.steps.clone();
    if idx < steps.len() {
        steps.remove(idx);
    }
    // Update transaction list: remove any txn with no remaining ops
    let remaining_txns: HashSet<TransactionId> =
        steps.iter().map(|s| s.operation.txn_id).collect();
    let transaction_order: Vec<TransactionId> = schedule
        .transaction_order
        .iter()
        .filter(|t| remaining_txns.contains(t))
        .copied()
        .collect();
    Schedule {
        steps,
        transaction_order,
        metadata: schedule.metadata.clone(),
    }
}

/// Check that every listed transaction has at least one operation.
fn all_txns_have_ops(schedule: &Schedule) -> bool {
    let ops_txns: HashSet<TransactionId> = schedule
        .steps
        .iter()
        .map(|s| s.operation.txn_id)
        .collect();
    schedule.transaction_order.iter().all(|t| ops_txns.contains(t))
}

/// Reassign positions to be contiguous starting from 0.
fn compact_positions(schedule: &Schedule) -> Schedule {
    let mut steps = schedule.steps.clone();
    steps.sort_by_key(|s| s.position);
    for (i, step) in steps.iter_mut().enumerate() {
        step.position = i;
    }
    Schedule {
        steps,
        transaction_order: schedule.transaction_order.clone(),
        metadata: schedule.metadata.clone(),
    }
}

// ---------------------------------------------------------------------------
// DeltaDebugging – alternative minimization strategy
// ---------------------------------------------------------------------------

/// Delta-debugging minimization: recursively halves the set of operations
/// to find a minimal failing subset.
pub struct DeltaDebugger<C: AnomalyChecker> {
    checker: C,
    max_depth: usize,
}

impl<C: AnomalyChecker> DeltaDebugger<C> {
    pub fn new(checker: C, max_depth: usize) -> Self {
        Self { checker, max_depth }
    }

    /// Run delta debugging on the schedule.
    pub fn minimize(
        &mut self,
        schedule: &Schedule,
        anomaly: &AnomalyClass,
    ) -> IsoSpecResult<Schedule> {
        let n = schedule.steps.len();
        if n <= 2 {
            return Ok(schedule.clone());
        }
        self.dd_rec(schedule, anomaly, 2, 0)
    }

    fn dd_rec(
        &mut self,
        schedule: &Schedule,
        anomaly: &AnomalyClass,
        granularity: usize,
        depth: usize,
    ) -> IsoSpecResult<Schedule> {
        if depth >= self.max_depth || schedule.steps.len() <= 2 {
            return Ok(schedule.clone());
        }

        let n = schedule.steps.len();
        let chunk_size = (n + granularity - 1) / granularity;

        // Try each complement (removing one chunk at a time)
        for chunk_idx in 0..granularity {
            let start = chunk_idx * chunk_size;
            let end = ((chunk_idx + 1) * chunk_size).min(n);

            let complement = self.remove_range(schedule, start, end);
            if complement.steps.len() < 2 {
                continue;
            }
            if !all_txns_have_ops(&complement) {
                continue;
            }
            if self.checker.check(&complement, anomaly)? {
                // Recurse with same granularity minus 1
                let new_gran = (granularity - 1).max(2);
                return self.dd_rec(&complement, anomaly, new_gran, depth + 1);
            }
        }

        // Try increasing granularity
        if granularity < n {
            let new_gran = (granularity * 2).min(n);
            if new_gran > granularity {
                return self.dd_rec(schedule, anomaly, new_gran, depth + 1);
            }
        }

        Ok(schedule.clone())
    }

    fn remove_range(&self, schedule: &Schedule, start: usize, end: usize) -> Schedule {
        let steps: Vec<ScheduleStep> = schedule
            .steps
            .iter()
            .enumerate()
            .filter(|(i, _)| *i < start || *i >= end)
            .map(|(_, s)| s.clone())
            .collect();
        let remaining_txns: HashSet<TransactionId> =
            steps.iter().map(|s| s.operation.txn_id).collect();
        let transaction_order: Vec<TransactionId> = schedule
            .transaction_order
            .iter()
            .filter(|t| remaining_txns.contains(t))
            .copied()
            .collect();
        Schedule {
            steps,
            transaction_order,
            metadata: schedule.metadata.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use isospec_types::identifier::ItemId;
    use isospec_types::operation::{Operation, ReadOp, WriteOp};
    use isospec_types::value::Value;

    fn make_schedule_with_conflict() -> Schedule {
        let t0 = TransactionId::new(0);
        let t1 = TransactionId::new(1);

        let mut s = Schedule::new();
        s.add_step(t0, Operation::read(OperationId::new(0), t0, isospec_types::identifier::TableId::new(0), ItemId::new(0), 0));
        s.add_step(t1, Operation::write(OperationId::new(1), t1, isospec_types::identifier::TableId::new(0), ItemId::new(0), Value::Integer(1), 1));
        s.add_step(t0, Operation::read(OperationId::new(2), t0, isospec_types::identifier::TableId::new(0), ItemId::new(0), 2));
        s.add_step(t1, Operation::write(OperationId::new(3), t1, isospec_types::identifier::TableId::new(0), ItemId::new(1), Value::Integer(99), 3));
        s
    }

    #[test]
    fn test_structural_checker_conflict() {
        let schedule = make_schedule_with_conflict();
        let mut checker = StructuralAnomalyChecker::new();
        assert!(checker.check(&schedule, &AnomalyClass::G1a).unwrap());
    }

    #[test]
    fn test_structural_checker_no_conflict() {
        let t0 = TransactionId::new(0);
        let t1 = TransactionId::new(1);
        let mut schedule = Schedule::new();
        schedule.add_step(t0, Operation::read(OperationId::new(0), t0, isospec_types::identifier::TableId::new(0), ItemId::new(0), 0));
        schedule.add_step(t1, Operation::read(OperationId::new(1), t1, isospec_types::identifier::TableId::new(0), ItemId::new(1), 1));
        let mut checker = StructuralAnomalyChecker::new();
        assert!(!checker.check(&schedule, &AnomalyClass::G0).unwrap());
    }

    #[test]
    fn test_mus_extractor_basic() {
        let schedule = make_schedule_with_conflict();
        let checker = StructuralAnomalyChecker::new();
        let config = MinimizerConfig::default();
        let mut extractor = MUSExtractor::new(checker, config);
        let result = extractor
            .minimize(&schedule, &AnomalyClass::G1a)
            .unwrap();
        assert!(result.schedule.steps.len() <= schedule.steps.len());
        assert!(result.schedule.steps.len() >= 2);
    }

    #[test]
    fn test_minimization_result_display() {
        let result = MinimizationResult {
            schedule: Schedule::new(),
            ops_removed: 3,
            txns_removed: 1,
            iterations: 5,
            elapsed: Duration::from_millis(100),
            completed: true,
            original_size: 6,
        };
        let display = format!("{}", result);
        assert!(display.contains("6 -> 0"));
        assert!(display.contains("50%"));
    }

    #[test]
    fn test_remove_transaction() {
        let schedule = make_schedule_with_conflict();
        let t1 = TransactionId::new(1);
        let reduced = remove_transaction(&schedule, &t1);
        assert_eq!(reduced.steps.len(), 2);
        assert!(reduced
            .steps
            .iter()
            .all(|s| s.operation.txn_id != t1));
    }

    #[test]
    fn test_remove_step() {
        let schedule = make_schedule_with_conflict();
        let reduced = remove_step(&schedule, 1);
        assert_eq!(reduced.steps.len(), 3);
    }

    #[test]
    fn test_compact_positions() {
        let t0 = TransactionId::new(0);
        let mut schedule = Schedule::new();
        // Manually create steps with non-contiguous positions
        schedule.steps.push(ScheduleStep {
            id: isospec_types::identifier::ScheduleStepId::new(0),
            txn_id: t0,
            operation: Operation::read(OperationId::new(0), t0, isospec_types::identifier::TableId::new(0), ItemId::new(0), 0),
            position: 5,
        });
        schedule.steps.push(ScheduleStep {
            id: isospec_types::identifier::ScheduleStepId::new(1),
            txn_id: t0,
            operation: Operation::read(OperationId::new(1), t0, isospec_types::identifier::TableId::new(0), ItemId::new(1), 1),
            position: 10,
        });
        schedule.transaction_order.push(t0);
        let compacted = compact_positions(&schedule);
        assert_eq!(compacted.steps[0].position, 0);
        assert_eq!(compacted.steps[1].position, 1);
    }

    #[test]
    fn test_delta_debugger() {
        let schedule = make_schedule_with_conflict();
        let checker = StructuralAnomalyChecker::new();
        let mut dd = DeltaDebugger::new(checker, 10);
        let minimized = dd.minimize(&schedule, &AnomalyClass::G1a).unwrap();
        assert!(minimized.steps.len() <= schedule.steps.len());
    }

    #[test]
    fn test_reduction_ratio() {
        let result = MinimizationResult {
            schedule: Schedule::new(),
            ops_removed: 4,
            txns_removed: 0,
            iterations: 1,
            elapsed: Duration::ZERO,
            completed: true,
            original_size: 8,
        };
        assert!((result.reduction_ratio() - 0.5).abs() < 0.001);
    }
}
