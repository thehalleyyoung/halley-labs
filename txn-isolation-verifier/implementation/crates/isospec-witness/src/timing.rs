//! Timing annotation for witness schedules.
//!
//! Adds advisory locks and barrier-based synchronization to witness scripts
//! so that concurrent transactions interleave in the correct order when
//! executed against a real database.

use std::collections::HashMap;

use isospec_types::error::{IsoSpecError, IsoSpecResult};
use isospec_types::identifier::TransactionId;
use isospec_types::isolation::IsolationLevel;
use isospec_types::operation::OpKind;
use isospec_types::schedule::{Schedule, ScheduleStep};

use crate::sql_gen::{SqlGenerator, TargetDialect};

// ---------------------------------------------------------------------------
// SyncStrategy
// ---------------------------------------------------------------------------

/// Strategy for synchronizing concurrent transaction execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncStrategy {
    /// Use advisory locks to gate each step.
    AdvisoryLocks,
    /// Use sleep-based timing with calibrated delays.
    SleepBased,
    /// Use a combination of advisory locks and sleeps.
    Hybrid,
}

impl Default for SyncStrategy {
    fn default() -> Self {
        SyncStrategy::AdvisoryLocks
    }
}

// ---------------------------------------------------------------------------
// TimingAnnotation
// ---------------------------------------------------------------------------

/// A timing annotation for a single step in a transaction.
#[derive(Debug, Clone)]
pub struct TimingAnnotation {
    /// The global step index this annotation applies to.
    pub global_step: u64,
    /// Lock IDs to acquire before executing this step.
    pub acquire_locks: Vec<i64>,
    /// Lock IDs to release after executing this step.
    pub release_locks: Vec<i64>,
    /// Optional sleep duration before the step (in seconds).
    pub pre_sleep: Option<f64>,
    /// Optional sleep duration after the step.
    pub post_sleep: Option<f64>,
    /// Comment describing the synchronization point.
    pub comment: String,
}

impl TimingAnnotation {
    pub fn empty(step: u64) -> Self {
        Self {
            global_step: step,
            acquire_locks: Vec::new(),
            release_locks: Vec::new(),
            pre_sleep: None,
            post_sleep: None,
            comment: String::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// AnnotatedTransaction
// ---------------------------------------------------------------------------

/// A transaction's SQL statements interleaved with timing annotations.
#[derive(Debug, Clone)]
pub struct AnnotatedTransaction {
    pub txn_index: usize,
    /// Interleaved sequence of (annotation, sql_statement) pairs.
    pub steps: Vec<(TimingAnnotation, String)>,
    /// Preamble SQL (SET ISOLATION LEVEL, BEGIN).
    pub preamble: Vec<String>,
    /// Postamble SQL (COMMIT/ROLLBACK).
    pub postamble: Vec<String>,
}

impl AnnotatedTransaction {
    /// Render to a complete SQL script.
    pub fn render(&self, gen: &dyn SqlGenerator) -> Vec<String> {
        let mut lines = Vec::new();
        lines.push(format!("-- Transaction {}", self.txn_index));

        for stmt in &self.preamble {
            lines.push(stmt.clone());
        }

        for (annotation, sql) in &self.steps {
            if !annotation.comment.is_empty() {
                lines.push(format!("-- {}", annotation.comment));
            }
            for lock_id in &annotation.acquire_locks {
                lines.push(gen.advisory_lock(*lock_id));
            }
            if let Some(delay) = annotation.pre_sleep {
                lines.push(gen.sleep(delay));
            }
            lines.push(sql.clone());
            if let Some(delay) = annotation.post_sleep {
                lines.push(gen.sleep(delay));
            }
            for lock_id in &annotation.release_locks {
                lines.push(gen.advisory_unlock(*lock_id));
            }
        }

        for stmt in &self.postamble {
            lines.push(stmt.clone());
        }

        lines
    }
}

// ---------------------------------------------------------------------------
// TimingAnnotator
// ---------------------------------------------------------------------------

/// Annotates a schedule with synchronization primitives.
pub struct TimingAnnotator {
    strategy: SyncStrategy,
    /// Base delay for sleep-based synchronization (seconds).
    base_delay: f64,
    /// Lock ID offset to avoid collisions.
    lock_offset: i64,
}

impl TimingAnnotator {
    pub fn new(strategy: SyncStrategy) -> Self {
        Self {
            strategy,
            base_delay: 0.1,
            lock_offset: 1000,
        }
    }

    pub fn with_base_delay(mut self, delay: f64) -> Self {
        self.base_delay = delay;
        self
    }

    pub fn with_lock_offset(mut self, offset: i64) -> Self {
        self.lock_offset = offset;
        self
    }

    /// Generate timing annotations for all steps in a schedule.
    pub fn annotate(&self, schedule: &Schedule) -> HashMap<usize, TimingAnnotation> {
        match self.strategy {
            SyncStrategy::AdvisoryLocks => self.annotate_advisory(schedule),
            SyncStrategy::SleepBased => self.annotate_sleep(schedule),
            SyncStrategy::Hybrid => self.annotate_hybrid(schedule),
        }
    }

    /// Advisory lock-based annotation: each step acquires a "turn" lock and
    /// releases the next step's lock.
    fn annotate_advisory(&self, schedule: &Schedule) -> HashMap<usize, TimingAnnotation> {
        let mut annotations = HashMap::new();
        let mut sorted_steps: Vec<&ScheduleStep> = schedule.steps.iter().collect();
        sorted_steps.sort_by_key(|s| s.position);

        for (idx, step) in sorted_steps.iter().enumerate() {
            let pos = step.position;
            let lock_id = self.lock_offset + idx as i64;
            let next_lock_id = self.lock_offset + (idx + 1) as i64;

            let mut annotation = TimingAnnotation::empty(pos as u64);

            if idx > 0 {
                // Wait for the previous step to release our lock
                annotation.acquire_locks.push(lock_id);
                annotation.comment = format!(
                    "Step {}: wait for lock {} (T{})",
                    idx,
                    lock_id,
                    step.operation.txn_id
                );
            } else {
                annotation.comment = format!(
                    "Step {}: first step (T{})",
                    idx,
                    step.operation.txn_id
                );
            }

            if idx < sorted_steps.len() - 1 {
                // Release the next step's lock after execution
                annotation.release_locks.push(next_lock_id);
            }

            annotations.insert(pos, annotation);
        }

        annotations
    }

    /// Sleep-based annotation: each step sleeps proportionally to its position.
    fn annotate_sleep(&self, schedule: &Schedule) -> HashMap<usize, TimingAnnotation> {
        let mut annotations = HashMap::new();
        let mut sorted_steps: Vec<&ScheduleStep> = schedule.steps.iter().collect();
        sorted_steps.sort_by_key(|s| s.position);

        for (idx, step) in sorted_steps.iter().enumerate() {
            let pos = step.position;
            let mut annotation = TimingAnnotation::empty(pos as u64);
            let delay = self.base_delay * idx as f64;
            if delay > 0.0 {
                annotation.pre_sleep = Some(delay);
            }
            annotation.comment = format!(
                "Step {}: sleep {:.2}s then execute (T{})",
                idx,
                delay,
                step.operation.txn_id
            );
            annotations.insert(pos, annotation);
        }

        annotations
    }

    /// Hybrid: advisory locks for cross-transaction steps, sleeps within.
    fn annotate_hybrid(&self, schedule: &Schedule) -> HashMap<usize, TimingAnnotation> {
        let mut annotations = HashMap::new();
        let mut sorted_steps: Vec<&ScheduleStep> = schedule.steps.iter().collect();
        sorted_steps.sort_by_key(|s| s.position);

        let mut prev_txn: Option<TransactionId> = None;

        for (idx, step) in sorted_steps.iter().enumerate() {
            let pos = step.position;
            let lock_id = self.lock_offset + idx as i64;
            let next_lock_id = self.lock_offset + (idx + 1) as i64;
            let mut annotation = TimingAnnotation::empty(pos as u64);

            let is_txn_switch = prev_txn.map_or(false, |prev| prev != step.operation.txn_id);

            if is_txn_switch && idx > 0 {
                // Use advisory locks at context switches
                annotation.acquire_locks.push(lock_id);
                annotation.comment = format!(
                    "Step {}: context switch to T{}, acquire lock {}",
                    idx,
                    step.operation.txn_id,
                    lock_id
                );
            } else if idx > 0 {
                // Same transaction: use a small sleep
                annotation.pre_sleep = Some(self.base_delay * 0.5);
                annotation.comment = format!(
                    "Step {}: continue T{}",
                    idx,
                    step.operation.txn_id
                );
            } else {
                annotation.comment = format!(
                    "Step {}: start T{}",
                    idx,
                    step.operation.txn_id
                );
            }

            if idx < sorted_steps.len() - 1 {
                let next_step = sorted_steps[idx + 1];
                if next_step.operation.txn_id != step.operation.txn_id {
                    annotation.release_locks.push(next_lock_id);
                }
            }

            prev_txn = Some(step.operation.txn_id);
            annotations.insert(pos, annotation);
        }

        annotations
    }

    /// Build fully annotated transactions from a schedule and SQL mapping.
    pub fn build_annotated_transactions(
        &self,
        schedule: &Schedule,
        txn_sql: &HashMap<usize, Vec<String>>,
        gen: &dyn SqlGenerator,
        isolation: &IsolationLevel,
    ) -> Vec<AnnotatedTransaction> {
        let annotations = self.annotate(schedule);

        let mut txn_step_map: HashMap<TransactionId, Vec<&ScheduleStep>> = HashMap::new();
        let mut sorted = schedule.steps.clone();
        sorted.sort_by_key(|s| s.position);

        for step in &sorted {
            txn_step_map
                .entry(step.operation.txn_id)
                .or_default()
                .push(step);
        }

        let mut result = Vec::new();
        let mut txn_ids: Vec<_> = txn_step_map.keys().copied().collect();
        txn_ids.sort_by_key(|id| format!("{}", id));

        for (idx, txn_id) in txn_ids.iter().enumerate() {
            let steps_for_txn = txn_step_map.get(txn_id).cloned().unwrap_or_default();
            let sql_stmts = txn_sql
                .get(&idx)
                .cloned()
                .unwrap_or_default();

            let preamble = gen.begin_transaction(isolation);
            let postamble = vec![gen.commit()];

            // Filter out preamble/postamble from sql_stmts
            let body_stmts: Vec<String> = sql_stmts
                .iter()
                .filter(|s| {
                    !s.starts_with("--")
                        && !s.starts_with("SET TRANSACTION")
                        && !s.starts_with("BEGIN")
                        && !s.starts_with("COMMIT")
                        && !s.starts_with("ROLLBACK")
                        && !s.starts_with("START TRANSACTION")
                })
                .cloned()
                .collect();

            let mut annotated_steps = Vec::new();
            for (step_idx, step) in steps_for_txn.iter().enumerate() {
                let annotation = annotations
                    .get(&step.position)
                    .cloned()
                    .unwrap_or_else(|| TimingAnnotation::empty(step.position as u64));
                let sql = body_stmts
                    .get(step_idx)
                    .cloned()
                    .unwrap_or_else(|| format!("-- step {} (no SQL)", step.position));
                annotated_steps.push((annotation, sql));
            }

            result.push(AnnotatedTransaction {
                txn_index: idx,
                steps: annotated_steps,
                preamble,
                postamble,
            });
        }

        result
    }
}

// ---------------------------------------------------------------------------
// BarrierScript – generates a coordinator script
// ---------------------------------------------------------------------------

/// Generates a barrier-based coordinator script that orchestrates multiple
/// database connections.
pub struct BarrierScriptGenerator;

impl BarrierScriptGenerator {
    /// Generate a shell script that coordinates transaction execution.
    pub fn generate_shell_script(
        annotated_txns: &[AnnotatedTransaction],
        gen: &dyn SqlGenerator,
    ) -> String {
        let mut lines = Vec::new();
        lines.push("#!/bin/bash".to_string());
        lines.push("# Auto-generated witness execution script".to_string());
        lines.push("set -e".to_string());
        lines.push(String::new());

        // Pre-acquire all advisory locks
        let total_steps: usize = annotated_txns.iter().map(|t| t.steps.len()).sum();
        lines.push(format!("echo 'Total steps: {}'", total_steps));
        lines.push(String::new());

        for txn in annotated_txns {
            let rendered = txn.render(gen);
            lines.push(format!(
                "# --- Transaction {} ({} steps) ---",
                txn.txn_index,
                txn.steps.len()
            ));

            let sql_content = rendered.join("\n");
            let escaped = sql_content.replace('\'', "'\\''");
            lines.push(format!(
                "TXN{}_SQL='{}'",
                txn.txn_index, escaped
            ));
            lines.push(String::new());
        }

        // Launch each transaction in background
        for txn in annotated_txns {
            lines.push(format!(
                "echo \"$TXN{}_SQL\" | $DB_CLIENT &",
                txn.txn_index
            ));
            lines.push(format!("TXN{}_PID=$!", txn.txn_index));
        }
        lines.push(String::new());

        // Wait for all
        for txn in annotated_txns {
            lines.push(format!("wait $TXN{}_PID", txn.txn_index));
            lines.push(format!(
                "echo 'Transaction {} completed'",
                txn.txn_index
            ));
        }

        lines.push(String::new());
        lines.push("echo 'All transactions completed'".to_string());
        lines.join("\n")
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use isospec_types::identifier::{ItemId, OperationId};
    use isospec_types::operation::{Operation, ReadOp, WriteOp};
    use isospec_types::schedule::ScheduleMetadata;
    use isospec_types::value::Value;

    fn make_interleaved_schedule() -> Schedule {
        let t0 = TransactionId::new(0);
        let t1 = TransactionId::new(1);
        let mut s = Schedule::new();
        s.add_step(t0, Operation::read(OperationId::new(0), t0, isospec_types::identifier::TableId::new(0), ItemId::new(0), 0));
        s.add_step(t1, Operation::write(OperationId::new(1), t1, isospec_types::identifier::TableId::new(0), ItemId::new(0), Value::Integer(1), 1));
        s.add_step(t0, Operation::read(OperationId::new(2), t0, isospec_types::identifier::TableId::new(0), ItemId::new(0), 2));
        s.add_step(t1, Operation::write(OperationId::new(3), t1, isospec_types::identifier::TableId::new(0), ItemId::new(1), Value::Integer(2), 3));
        s
    }

    #[test]
    fn test_advisory_lock_annotation() {
        let schedule = make_interleaved_schedule();
        let annotator = TimingAnnotator::new(SyncStrategy::AdvisoryLocks);
        let annotations = annotator.annotate(&schedule);
        assert_eq!(annotations.len(), 4);
        // First step should have no acquire locks
        let first = annotations.get(&0).unwrap();
        assert!(first.acquire_locks.is_empty());
        // Second step should acquire a lock
        let second = annotations.get(&1).unwrap();
        assert_eq!(second.acquire_locks.len(), 1);
    }

    #[test]
    fn test_sleep_annotation() {
        let schedule = make_interleaved_schedule();
        let annotator = TimingAnnotator::new(SyncStrategy::SleepBased)
            .with_base_delay(0.5);
        let annotations = annotator.annotate(&schedule);
        let first = annotations.get(&0).unwrap();
        assert!(first.pre_sleep.is_none());
        let third = annotations.get(&2).unwrap();
        assert!(third.pre_sleep.is_some());
        assert!((third.pre_sleep.unwrap() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_hybrid_annotation() {
        let schedule = make_interleaved_schedule();
        let annotator = TimingAnnotator::new(SyncStrategy::Hybrid);
        let annotations = annotator.annotate(&schedule);
        assert_eq!(annotations.len(), 4);
        // Step 1 (T1 after T0) should have advisory lock (context switch)
        let second = annotations.get(&1).unwrap();
        assert!(!second.acquire_locks.is_empty());
    }

    #[test]
    fn test_annotated_transaction_render() {
        let gen = crate::sql_gen::PostgreSqlGenerator;
        let annotation = TimingAnnotation {
            global_step: 0,
            acquire_locks: vec![1000],
            release_locks: vec![1001],
            pre_sleep: None,
            post_sleep: None,
            comment: "step 0".to_string(),
        };
        let txn = AnnotatedTransaction {
            txn_index: 0,
            steps: vec![(annotation, "SELECT val FROM t WHERE id = 0;".to_string())],
            preamble: gen.begin_transaction(&IsolationLevel::ReadCommitted),
            postamble: vec![gen.commit()],
        };
        let rendered = txn.render(&gen);
        assert!(rendered.iter().any(|l| l.contains("pg_advisory_lock")));
        assert!(rendered.iter().any(|l| l.contains("pg_advisory_unlock")));
        assert!(rendered.iter().any(|l| l.contains("SELECT val")));
    }

    #[test]
    fn test_barrier_script_generation() {
        let gen = crate::sql_gen::PostgreSqlGenerator;
        let txn0 = AnnotatedTransaction {
            txn_index: 0,
            steps: vec![(
                TimingAnnotation::empty(0),
                "SELECT 1;".to_string(),
            )],
            preamble: vec!["BEGIN;".to_string()],
            postamble: vec!["COMMIT;".to_string()],
        };
        let txn1 = AnnotatedTransaction {
            txn_index: 1,
            steps: vec![(
                TimingAnnotation::empty(1),
                "SELECT 2;".to_string(),
            )],
            preamble: vec!["BEGIN;".to_string()],
            postamble: vec!["COMMIT;".to_string()],
        };
        let script = BarrierScriptGenerator::generate_shell_script(&[txn0, txn1], &gen);
        assert!(script.contains("#!/bin/bash"));
        assert!(script.contains("TXN0_PID"));
        assert!(script.contains("TXN1_PID"));
        assert!(script.contains("wait"));
    }

    #[test]
    fn test_timing_annotation_empty() {
        let ann = TimingAnnotation::empty(5);
        assert_eq!(ann.global_step, 5);
        assert!(ann.acquire_locks.is_empty());
        assert!(ann.release_locks.is_empty());
        assert!(ann.pre_sleep.is_none());
    }

    #[test]
    fn test_sync_strategy_default() {
        assert_eq!(SyncStrategy::default(), SyncStrategy::AdvisoryLocks);
    }
}
