//! Concurrent SQL executor for witness schedules.
//!
//! Executes witness schedules against real databases using a
//! thread-per-connection model with barrier-based synchronization.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Barrier, Mutex};
use std::time::{Duration, Instant};

use isospec_types::config::EngineKind;
use isospec_types::error::{IsoSpecError, IsoSpecResult};
use isospec_types::identifier::TransactionId;
use isospec_types::value::Value;

use crate::adapter::{AdapterConfig, DatabaseAdapter, QueryResult};

// ---------------------------------------------------------------------------
// ExecutionPlan
// ---------------------------------------------------------------------------

/// A plan for executing a witness schedule against a real database.
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    /// SQL statements per transaction, in execution order.
    pub transactions: HashMap<usize, Vec<SqlStep>>,
    /// Schema setup SQL.
    pub setup_sql: Vec<String>,
    /// Teardown SQL.
    pub teardown_sql: Vec<String>,
    /// Number of concurrent connections needed.
    pub concurrency: usize,
}

/// A single SQL step with synchronization metadata.
#[derive(Debug, Clone)]
pub struct SqlStep {
    /// The SQL statement to execute.
    pub sql: String,
    /// Global position in the schedule (for ordering).
    pub global_position: u64,
    /// Barrier ID to synchronize on before executing (if any).
    pub sync_before: Option<u64>,
    /// Barrier ID to synchronize on after executing (if any).
    pub sync_after: Option<u64>,
    /// Whether this step's result should be captured.
    pub capture_result: bool,
    /// Expected value (for reads).
    pub expected: Option<Value>,
}

impl SqlStep {
    pub fn new(sql: &str, position: u64) -> Self {
        Self {
            sql: sql.to_string(),
            global_position: position,
            sync_before: None,
            sync_after: None,
            capture_result: false,
            expected: None,
        }
    }

    pub fn with_sync_before(mut self, barrier: u64) -> Self {
        self.sync_before = Some(barrier);
        self
    }

    pub fn with_sync_after(mut self, barrier: u64) -> Self {
        self.sync_after = Some(barrier);
        self
    }

    pub fn with_capture(mut self) -> Self {
        self.capture_result = true;
        self
    }

    pub fn with_expected(mut self, val: Value) -> Self {
        self.expected = Some(val);
        self.capture_result = true;
        self
    }
}

impl ExecutionPlan {
    pub fn new() -> Self {
        Self {
            transactions: HashMap::new(),
            setup_sql: Vec::new(),
            teardown_sql: Vec::new(),
            concurrency: 0,
        }
    }

    pub fn add_transaction(&mut self, txn_idx: usize, steps: Vec<SqlStep>) {
        self.transactions.insert(txn_idx, steps);
        self.concurrency = self.transactions.len();
    }

    pub fn total_steps(&self) -> usize {
        self.transactions.values().map(|v| v.len()).sum()
    }
}

impl Default for ExecutionPlan {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// StepResult
// ---------------------------------------------------------------------------

/// Result of executing a single SQL step.
#[derive(Debug, Clone)]
pub struct StepResult {
    pub txn_index: usize,
    pub step_index: usize,
    pub global_position: u64,
    pub query_result: Option<QueryResult>,
    pub error: Option<String>,
    pub elapsed: Duration,
    pub matched_expected: Option<bool>,
}

impl StepResult {
    pub fn success(txn: usize, step: usize, pos: u64, elapsed: Duration) -> Self {
        Self {
            txn_index: txn,
            step_index: step,
            global_position: pos,
            query_result: None,
            error: None,
            elapsed,
            matched_expected: None,
        }
    }

    pub fn with_result(mut self, result: QueryResult) -> Self {
        self.query_result = Some(result);
        self
    }

    pub fn failure(txn: usize, step: usize, pos: u64, error: String) -> Self {
        Self {
            txn_index: txn,
            step_index: step,
            global_position: pos,
            query_result: None,
            error: Some(error),
            elapsed: Duration::ZERO,
            matched_expected: None,
        }
    }

    pub fn is_success(&self) -> bool {
        self.error.is_none()
    }
}

// ---------------------------------------------------------------------------
// ExecutionResult
// ---------------------------------------------------------------------------

/// Aggregated result of executing a full witness schedule.
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    /// Per-transaction step results.
    pub step_results: HashMap<usize, Vec<StepResult>>,
    /// Total execution time.
    pub total_elapsed: Duration,
    /// Whether all steps completed successfully.
    pub all_success: bool,
    /// Number of mismatched expectations.
    pub mismatches: usize,
    /// Errors encountered.
    pub errors: Vec<String>,
}

impl ExecutionResult {
    pub fn total_steps(&self) -> usize {
        self.step_results.values().map(|v| v.len()).sum()
    }

    pub fn successful_steps(&self) -> usize {
        self.step_results
            .values()
            .flat_map(|v| v.iter())
            .filter(|r| r.is_success())
            .count()
    }

    pub fn failed_steps(&self) -> usize {
        self.total_steps() - self.successful_steps()
    }
}

impl fmt::Display for ExecutionResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Execution: {}/{} steps ok, {} mismatches, {} errors, {:.3}s",
            self.successful_steps(),
            self.total_steps(),
            self.mismatches,
            self.errors.len(),
            self.total_elapsed.as_secs_f64(),
        )
    }
}

// ---------------------------------------------------------------------------
// SequentialExecutor – single-threaded executor for testing
// ---------------------------------------------------------------------------

/// Executes a plan sequentially (single-threaded) for simpler testing.
pub struct SequentialExecutor;

impl SequentialExecutor {
    /// Execute a plan against a database adapter sequentially, interleaving
    /// transactions in global position order.
    pub fn execute(
        plan: &ExecutionPlan,
        adapter: &dyn DatabaseAdapter,
    ) -> IsoSpecResult<ExecutionResult> {
        let start = Instant::now();
        let mut step_results: HashMap<usize, Vec<StepResult>> = HashMap::new();
        let mut errors = Vec::new();
        let mut mismatches = 0;

        // Run setup
        for sql in &plan.setup_sql {
            adapter.execute(sql)?;
        }

        // Collect all steps with their transaction index and step index
        let mut all_steps: Vec<(usize, usize, &SqlStep)> = Vec::new();
        for (txn_idx, steps) in &plan.transactions {
            for (step_idx, step) in steps.iter().enumerate() {
                all_steps.push((*txn_idx, step_idx, step));
            }
        }
        // Sort by global position for interleaved execution
        all_steps.sort_by_key(|(_, _, s)| s.global_position);

        for (txn_idx, step_idx, sql_step) in &all_steps {
            let step_start = Instant::now();
            match adapter.execute(&sql_step.sql) {
                Ok(result) => {
                    let mut sr =
                        StepResult::success(*txn_idx, *step_idx, sql_step.global_position, step_start.elapsed());

                    if sql_step.capture_result {
                        if let Some(ref expected) = sql_step.expected {
                            let actual = result.get_value(0, 0);
                            let matched = actual.map_or(false, |a| a == expected);
                            sr.matched_expected = Some(matched);
                            if !matched {
                                mismatches += 1;
                            }
                        }
                        sr = sr.with_result(result);
                    }

                    step_results.entry(*txn_idx).or_default().push(sr);
                }
                Err(e) => {
                    let err_msg = format!("txn {} step {}: {}", txn_idx, step_idx, e);
                    errors.push(err_msg.clone());
                    step_results
                        .entry(*txn_idx)
                        .or_default()
                        .push(StepResult::failure(
                            *txn_idx,
                            *step_idx,
                            sql_step.global_position,
                            err_msg,
                        ));
                }
            }
        }

        // Run teardown
        for sql in &plan.teardown_sql {
            let _ = adapter.execute(sql);
        }

        let all_success = errors.is_empty();
        Ok(ExecutionResult {
            step_results,
            total_elapsed: start.elapsed(),
            all_success,
            mismatches,
            errors,
        })
    }
}

// ---------------------------------------------------------------------------
// ConcurrentExecutor – multi-threaded barrier-based executor
// ---------------------------------------------------------------------------

/// Describes the synchronization strategy for concurrent execution.
#[derive(Debug, Clone)]
pub struct ConcurrencyConfig {
    /// Number of threads to use.
    pub thread_count: usize,
    /// Timeout for barrier synchronization.
    pub barrier_timeout: Duration,
    /// Delay between steps (to improve interleaving reliability).
    pub inter_step_delay: Duration,
}

impl Default for ConcurrencyConfig {
    fn default() -> Self {
        Self {
            thread_count: 4,
            barrier_timeout: Duration::from_secs(10),
            inter_step_delay: Duration::from_millis(50),
        }
    }
}

/// A concurrent executor that manages barriers for synchronization.
/// Each transaction runs in its own thread with a dedicated connection.
pub struct ConcurrentExecutor {
    config: ConcurrencyConfig,
}

impl ConcurrentExecutor {
    pub fn new(config: ConcurrencyConfig) -> Self {
        Self { config }
    }

    /// Build a set of barriers for synchronizing execution points.
    pub fn build_barriers(
        &self,
        plan: &ExecutionPlan,
    ) -> HashMap<u64, Arc<Barrier>> {
        let mut barrier_ids: std::collections::HashSet<u64> = std::collections::HashSet::new();
        for steps in plan.transactions.values() {
            for step in steps {
                if let Some(b) = step.sync_before {
                    barrier_ids.insert(b);
                }
                if let Some(b) = step.sync_after {
                    barrier_ids.insert(b);
                }
            }
        }

        let num_threads = plan.concurrency;
        barrier_ids
            .into_iter()
            .map(|id| (id, Arc::new(Barrier::new(num_threads))))
            .collect()
    }

    /// Execute a single transaction's steps with barrier synchronization.
    /// This is meant to be called from a per-transaction thread.
    pub fn execute_transaction_steps(
        steps: &[SqlStep],
        txn_idx: usize,
        adapter: &dyn DatabaseAdapter,
        barriers: &HashMap<u64, Arc<Barrier>>,
        inter_step_delay: Duration,
    ) -> Vec<StepResult> {
        let mut results = Vec::new();

        for (step_idx, step) in steps.iter().enumerate() {
            // Sync before
            if let Some(barrier_id) = step.sync_before {
                if let Some(barrier) = barriers.get(&barrier_id) {
                    barrier.wait();
                }
            }

            if !inter_step_delay.is_zero() {
                std::thread::sleep(inter_step_delay);
            }

            let step_start = Instant::now();
            match adapter.execute(&step.sql) {
                Ok(result) => {
                    let mut sr = StepResult::success(
                        txn_idx,
                        step_idx,
                        step.global_position,
                        step_start.elapsed(),
                    );
                    if step.capture_result {
                        sr = sr.with_result(result);
                    }
                    results.push(sr);
                }
                Err(e) => {
                    results.push(StepResult::failure(
                        txn_idx,
                        step_idx,
                        step.global_position,
                        format!("{}", e),
                    ));
                }
            }

            // Sync after
            if let Some(barrier_id) = step.sync_after {
                if let Some(barrier) = barriers.get(&barrier_id) {
                    barrier.wait();
                }
            }
        }

        results
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::MockAdapter;

    #[test]
    fn test_sql_step_builder() {
        let step = SqlStep::new("SELECT 1;", 0)
            .with_sync_before(100)
            .with_sync_after(101)
            .with_capture()
            .with_expected(Value::Integer(1));
        assert_eq!(step.sql, "SELECT 1;");
        assert_eq!(step.sync_before, Some(100));
        assert_eq!(step.sync_after, Some(101));
        assert!(step.capture_result);
        assert_eq!(step.expected, Some(Value::Integer(1)));
    }

    #[test]
    fn test_execution_plan() {
        let mut plan = ExecutionPlan::new();
        plan.setup_sql = vec!["CREATE TABLE t (id INT);".into()];
        plan.teardown_sql = vec!["DROP TABLE t;".into()];
        plan.add_transaction(0, vec![
            SqlStep::new("INSERT INTO t VALUES (1);", 0),
            SqlStep::new("COMMIT;", 2),
        ]);
        plan.add_transaction(1, vec![
            SqlStep::new("INSERT INTO t VALUES (2);", 1),
            SqlStep::new("COMMIT;", 3),
        ]);
        assert_eq!(plan.concurrency, 2);
        assert_eq!(plan.total_steps(), 4);
    }

    #[test]
    fn test_sequential_executor() {
        let adapter = MockAdapter::new(EngineKind::PostgreSQL);
        let mut plan = ExecutionPlan::new();
        plan.add_transaction(0, vec![
            SqlStep::new("SELECT 1;", 0),
            SqlStep::new("SELECT 2;", 2),
        ]);
        plan.add_transaction(1, vec![SqlStep::new("SELECT 3;", 1)]);

        let result = SequentialExecutor::execute(&plan, &adapter).unwrap();
        assert!(result.all_success);
        assert_eq!(result.total_steps(), 3);
        assert_eq!(result.successful_steps(), 3);
    }

    #[test]
    fn test_step_result_success() {
        let sr = StepResult::success(0, 0, 0, Duration::from_millis(5));
        assert!(sr.is_success());
        assert!(sr.error.is_none());
    }

    #[test]
    fn test_step_result_failure() {
        let sr = StepResult::failure(0, 0, 0, "timeout".into());
        assert!(!sr.is_success());
        assert_eq!(sr.error, Some("timeout".into()));
    }

    #[test]
    fn test_execution_result_display() {
        let result = ExecutionResult {
            step_results: HashMap::new(),
            total_elapsed: Duration::from_millis(100),
            all_success: true,
            mismatches: 0,
            errors: Vec::new(),
        };
        let display = format!("{}", result);
        assert!(display.contains("0/0 steps ok"));
    }

    #[test]
    fn test_concurrent_executor_barriers() {
        let executor = ConcurrentExecutor::new(ConcurrencyConfig::default());
        let mut plan = ExecutionPlan::new();
        plan.add_transaction(0, vec![
            SqlStep::new("SELECT 1;", 0).with_sync_after(1),
        ]);
        plan.add_transaction(1, vec![
            SqlStep::new("SELECT 2;", 1).with_sync_before(1),
        ]);
        let barriers = executor.build_barriers(&plan);
        assert_eq!(barriers.len(), 1);
        assert!(barriers.contains_key(&1));
    }

    #[test]
    fn test_sequential_with_capture() {
        let adapter = MockAdapter::new(EngineKind::PostgreSQL);
        adapter.add_response(
            "SELECT val",
            QueryResult::with_rows(
                vec!["val".into()],
                vec![vec![Value::Integer(42)]],
            ),
        );

        let mut plan = ExecutionPlan::new();
        plan.add_transaction(0, vec![
            SqlStep::new("SELECT val FROM t WHERE id = 1;", 0)
                .with_capture()
                .with_expected(Value::Integer(42)),
        ]);

        let result = SequentialExecutor::execute(&plan, &adapter).unwrap();
        assert!(result.all_success);
        assert_eq!(result.mismatches, 0);
    }

    #[test]
    fn test_sequential_with_mismatch() {
        let adapter = MockAdapter::new(EngineKind::PostgreSQL);
        adapter.add_response(
            "SELECT val",
            QueryResult::with_rows(
                vec!["val".into()],
                vec![vec![Value::Integer(99)]],
            ),
        );

        let mut plan = ExecutionPlan::new();
        plan.add_transaction(0, vec![
            SqlStep::new("SELECT val FROM t WHERE id = 1;", 0)
                .with_expected(Value::Integer(42)),
        ]);

        let result = SequentialExecutor::execute(&plan, &adapter).unwrap();
        assert_eq!(result.mismatches, 1);
    }
}
