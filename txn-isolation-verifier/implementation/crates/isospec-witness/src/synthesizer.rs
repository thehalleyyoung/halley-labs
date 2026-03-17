//! Witness synthesis from SAT SMT models.
//!
//! Takes a satisfying SMT model and produces a minimal runnable witness
//! consisting of SQL scripts that demonstrate the anomaly.

use std::collections::HashMap;
use std::fmt;

use isospec_types::constraint::SmtExpr;
use isospec_types::error::{IsoSpecError, IsoSpecResult};
use isospec_types::identifier::{ItemId, OperationId, TransactionId};
use isospec_types::isolation::{AnomalyClass, IsolationLevel};
use isospec_types::operation::{OpKind, Operation, ReadOp, WriteOp};
use isospec_types::schedule::{Schedule, ScheduleMetadata, ScheduleStep};
use isospec_types::schema::TableSchema;
use isospec_types::value::Value;

// ---------------------------------------------------------------------------
// WitnessResult
// ---------------------------------------------------------------------------

/// A complete witness demonstrating an isolation anomaly.
#[derive(Debug, Clone)]
pub struct WitnessResult {
    /// The schedule that exhibits the anomaly.
    pub schedule: Schedule,
    /// The anomaly class demonstrated.
    pub anomaly: AnomalyClass,
    /// SQL scripts for each transaction (keyed by transaction index).
    pub sql_scripts: HashMap<usize, Vec<String>>,
    /// Schema setup SQL.
    pub schema_sql: Vec<String>,
    /// Teardown SQL.
    pub teardown_sql: Vec<String>,
    /// Human-readable description.
    pub description: String,
    /// The isolation level under which this anomaly appears.
    pub isolation_level: IsolationLevel,
}

impl WitnessResult {
    pub fn num_transactions(&self) -> usize {
        self.sql_scripts.len()
    }

    pub fn total_sql_statements(&self) -> usize {
        self.sql_scripts.values().map(|v| v.len()).sum::<usize>()
            + self.schema_sql.len()
            + self.teardown_sql.len()
    }

    /// Render the complete witness as a single SQL script with comments.
    pub fn render_full_script(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!("-- Witness for anomaly: {:?}", self.anomaly));
        lines.push(format!(
            "-- Isolation level: {:?}",
            self.isolation_level
        ));
        lines.push(format!("-- {}", self.description));
        lines.push(String::new());

        lines.push("-- === Schema Setup ===".to_string());
        for stmt in &self.schema_sql {
            lines.push(stmt.clone());
        }
        lines.push(String::new());

        let mut sorted_txns: Vec<_> = self.sql_scripts.keys().collect();
        sorted_txns.sort();
        for txn_idx in sorted_txns {
            let stmts = &self.sql_scripts[txn_idx];
            lines.push(format!("-- === Transaction {} ===", txn_idx));
            for stmt in stmts {
                lines.push(stmt.clone());
            }
            lines.push(String::new());
        }

        lines.push("-- === Teardown ===".to_string());
        for stmt in &self.teardown_sql {
            lines.push(stmt.clone());
        }

        lines.join("\n")
    }
}

impl fmt::Display for WitnessResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Witness({:?} at {:?}, {} txns, {} stmts)",
            self.anomaly,
            self.isolation_level,
            self.num_transactions(),
            self.total_sql_statements(),
        )
    }
}

// ---------------------------------------------------------------------------
// ScheduleToWitness – extract operations from schedule
// ---------------------------------------------------------------------------

/// Extracts a per-transaction view from a flat schedule.
#[derive(Debug)]
pub struct ScheduleView {
    /// Operations per transaction, in execution order.
    pub txn_ops: HashMap<TransactionId, Vec<ScheduleStep>>,
    /// Global execution order.
    pub global_order: Vec<(TransactionId, ScheduleStep)>,
}

impl ScheduleView {
    pub fn from_schedule(schedule: &Schedule) -> Self {
        let mut txn_ops: HashMap<TransactionId, Vec<ScheduleStep>> = HashMap::new();
        let mut global_order = Vec::new();

        let mut sorted_steps = schedule.steps.clone();
        sorted_steps.sort_by_key(|s| s.position);

        for step in &sorted_steps {
            let txn_id = step.operation.txn_id;
            txn_ops
                .entry(txn_id)
                .or_default()
                .push(step.clone());
            global_order.push((txn_id, step.clone()));
        }

        Self {
            txn_ops,
            global_order,
        }
    }

    pub fn transaction_ids(&self) -> Vec<TransactionId> {
        let mut ids: Vec<_> = self.txn_ops.keys().copied().collect();
        ids.sort_by_key(|id| format!("{}", id));
        ids
    }

    pub fn ops_for(&self, txn: &TransactionId) -> &[ScheduleStep] {
        self.txn_ops
            .get(txn)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }
}

// ---------------------------------------------------------------------------
// WitnessSynthesizer
// ---------------------------------------------------------------------------

/// Configuration for witness synthesis.
#[derive(Debug, Clone)]
pub struct SynthesizerConfig {
    /// The table name to use for the witness schema.
    pub table_name: String,
    /// The key column name.
    pub key_column: String,
    /// The value column name.
    pub value_column: String,
    /// Initial values to seed the table with.
    pub initial_values: HashMap<i64, Value>,
    /// Isolation level for the witness transactions.
    pub isolation_level: IsolationLevel,
}

impl Default for SynthesizerConfig {
    fn default() -> Self {
        let mut initial = HashMap::new();
        for i in 0..8 {
            initial.insert(i, Value::Integer(0));
        }
        Self {
            table_name: "witness_data".to_string(),
            key_column: "id".to_string(),
            value_column: "val".to_string(),
            initial_values: initial,
            isolation_level: IsolationLevel::ReadCommitted,
        }
    }
}

/// Synthesizes runnable SQL witness scripts from a schedule.
pub struct WitnessSynthesizer {
    config: SynthesizerConfig,
}

impl WitnessSynthesizer {
    pub fn new(config: SynthesizerConfig) -> Self {
        Self { config }
    }

    pub fn with_default_config() -> Self {
        Self::new(SynthesizerConfig::default())
    }

    /// Synthesize a complete witness from a schedule and anomaly classification.
    pub fn synthesize(
        &self,
        schedule: &Schedule,
        anomaly: AnomalyClass,
    ) -> IsoSpecResult<WitnessResult> {
        let view = ScheduleView::from_schedule(schedule);
        let schema_sql = self.generate_schema_sql();
        let teardown_sql = self.generate_teardown_sql();

        let mut sql_scripts: HashMap<usize, Vec<String>> = HashMap::new();

        for (idx, txn_id) in view.transaction_ids().iter().enumerate() {
            let ops = view.ops_for(txn_id);
            let stmts = self.ops_to_sql(ops, idx);
            sql_scripts.insert(idx, stmts);
        }

        let description = format!(
            "Witness schedule demonstrating {:?} anomaly with {} transactions and {} operations",
            anomaly,
            view.transaction_ids().len(),
            schedule.steps.len(),
        );

        Ok(WitnessResult {
            schedule: schedule.clone(),
            anomaly,
            sql_scripts,
            schema_sql,
            teardown_sql,
            description,
            isolation_level: self.config.isolation_level,
        })
    }

    /// Generate the schema creation SQL.
    fn generate_schema_sql(&self) -> Vec<String> {
        let mut stmts = Vec::new();
        stmts.push(format!(
            "DROP TABLE IF EXISTS {};",
            self.config.table_name
        ));
        stmts.push(format!(
            "CREATE TABLE {} ({} INTEGER PRIMARY KEY, {} INTEGER NOT NULL);",
            self.config.table_name, self.config.key_column, self.config.value_column
        ));

        let mut sorted_entries: Vec<_> = self.config.initial_values.iter().collect();
        sorted_entries.sort_by_key(|(k, _)| *k);

        for (key, val) in sorted_entries {
            let val_str = match val {
                Value::Integer(i) => i.to_string(),
                Value::Float(f) => f.to_string(),
                Value::Text(s) => format!("'{}'", s.replace('\'', "''")),
                Value::Boolean(b) => if *b { "1" } else { "0" }.to_string(),
                Value::Null => "NULL".to_string(),
                _ => "0".to_string(),
            };
            stmts.push(format!(
                "INSERT INTO {} ({}, {}) VALUES ({}, {});",
                self.config.table_name,
                self.config.key_column,
                self.config.value_column,
                key,
                val_str
            ));
        }
        stmts
    }

    /// Generate the teardown SQL.
    fn generate_teardown_sql(&self) -> Vec<String> {
        vec![format!("DROP TABLE IF EXISTS {};", self.config.table_name)]
    }

    /// Convert a list of schedule steps into SQL statements for one transaction.
    fn ops_to_sql(&self, ops: &[ScheduleStep], txn_idx: usize) -> Vec<String> {
        let mut stmts = Vec::new();
        let iso_str = isolation_level_sql(&self.config.isolation_level);
        stmts.push(format!(
            "-- Transaction {} at isolation level {}",
            txn_idx, iso_str
        ));
        stmts.push(format!("SET TRANSACTION ISOLATION LEVEL {};", iso_str));
        stmts.push("BEGIN;".to_string());

        for step in ops {
            match &step.operation.kind {
                OpKind::Read(read_op) => {
                    let item_id = read_op.item;
                    stmts.push(format!(
                        "SELECT {} FROM {} WHERE {} = {}; -- expect {:?}",
                        self.config.value_column,
                        self.config.table_name,
                        self.config.key_column,
                        item_id,
                        read_op.value_read,
                    ));
                }
                OpKind::Write(write_op) => {
                    let item_id = write_op.item;
                    let val_str = value_to_sql(&write_op.new_value);
                    stmts.push(format!(
                        "UPDATE {} SET {} = {} WHERE {} = {};",
                        self.config.table_name,
                        self.config.value_column,
                        val_str,
                        self.config.key_column,
                        item_id,
                    ));
                }
                OpKind::Insert(insert_op) => {
                    let item_id = insert_op.item;
                    let val_str = insert_op.values.values().next()
                        .map(|v| value_to_sql(v))
                        .unwrap_or_else(|| "0".to_string());
                    stmts.push(format!(
                        "INSERT INTO {} ({}, {}) VALUES ({}, {});",
                        self.config.table_name,
                        self.config.key_column,
                        self.config.value_column,
                        item_id,
                        val_str,
                    ));
                }
                OpKind::Delete(delete_op) => {
                    if let Some(item_id) = delete_op.deleted_items.first() {
                        stmts.push(format!(
                            "DELETE FROM {} WHERE {} = {};",
                            self.config.table_name,
                            self.config.key_column,
                            item_id,
                        ));
                    }
                }
                OpKind::Begin(_) => {
                    // Already handled above
                }
                OpKind::Commit(_) => {
                    stmts.push("COMMIT;".to_string());
                }
                OpKind::Abort(_) => {
                    stmts.push("ROLLBACK;".to_string());
                }
                OpKind::Lock(lock_op) => {
                    if let Some(item_id) = lock_op.item {
                        stmts.push(format!(
                            "SELECT {} FROM {} WHERE {} = {} FOR UPDATE;",
                            self.config.value_column,
                            self.config.table_name,
                            self.config.key_column,
                            item_id,
                        ));
                    }
                }
                OpKind::PredicateRead(pred_read) => {
                    stmts.push(format!(
                        "SELECT {} FROM {} WHERE {} > 0;",
                        self.config.value_column,
                        self.config.table_name,
                        self.config.value_column,
                    ));
                }
                OpKind::PredicateWrite(_) => {
                    stmts.push(format!(
                        "UPDATE {} SET {} = {} + 1 WHERE {} > 0;",
                        self.config.table_name,
                        self.config.value_column,
                        self.config.value_column,
                        self.config.value_column,
                    ));
                }
            }
        }

        // Add commit if not already present
        let has_end = ops
            .iter()
            .any(|s| matches!(&s.operation.kind, OpKind::Commit(_) | OpKind::Abort(_)));
        if !has_end {
            stmts.push("COMMIT;".to_string());
        }

        stmts
    }
}

/// Convert an isolation level to SQL syntax.
fn isolation_level_sql(level: &IsolationLevel) -> &'static str {
    match level {
        IsolationLevel::ReadUncommitted => "READ UNCOMMITTED",
        IsolationLevel::ReadCommitted => "READ COMMITTED",
        IsolationLevel::RepeatableRead => "REPEATABLE READ",
        IsolationLevel::Serializable => "SERIALIZABLE",
        _ => "SERIALIZABLE",
    }
}

/// Convert a Value to a SQL literal string.
fn value_to_sql(val: &Value) -> String {
    match val {
        Value::Integer(i) => i.to_string(),
        Value::Float(f) => f.to_string(),
        Value::Text(s) => format!("'{}'", s.replace('\'', "''")),
        Value::Boolean(b) => if *b { "TRUE" } else { "FALSE" }.to_string(),
        Value::Null => "NULL".to_string(),
        _ => "0".to_string(),
    }
}

// ---------------------------------------------------------------------------
// WitnessVerifier – verifies expected outcomes
// ---------------------------------------------------------------------------

/// A description of an expected read outcome in the witness.
#[derive(Debug, Clone)]
pub struct ExpectedRead {
    pub txn_index: usize,
    pub step_index: usize,
    pub item_id: ItemId,
    pub expected_value: Value,
}

/// Extracts expected read outcomes from a witness schedule.
pub struct ExpectedOutcomeExtractor;

impl ExpectedOutcomeExtractor {
    pub fn extract(witness: &WitnessResult) -> Vec<ExpectedRead> {
        let mut expected = Vec::new();
        let view = ScheduleView::from_schedule(&witness.schedule);

        for (idx, txn_id) in view.transaction_ids().iter().enumerate() {
            for (step_idx, step) in view.ops_for(txn_id).iter().enumerate() {
                if let OpKind::Read(read_op) = &step.operation.kind {
                    if let Some(ref val) = read_op.value_read {
                        expected.push(ExpectedRead {
                            txn_index: idx,
                            step_index: step_idx,
                            item_id: read_op.item,
                            expected_value: val.clone(),
                        });
                    }
                }
            }
        }

        expected
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use isospec_types::operation::{BeginOp, CommitOp};

    fn make_test_schedule() -> Schedule {
        let t0 = TransactionId::new(0);
        let t1 = TransactionId::new(1);

        let mut s = Schedule::new();
        s.add_step(t0, Operation::read(OperationId::new(0), t0, isospec_types::identifier::TableId::new(0), ItemId::new(0), 0));
        s.add_step(t1, Operation::write(OperationId::new(1), t1, isospec_types::identifier::TableId::new(0), ItemId::new(0), Value::Integer(42), 1));
        s.add_step(t0, Operation::read(OperationId::new(2), t0, isospec_types::identifier::TableId::new(0), ItemId::new(0), 2));
        s
    }

    #[test]
    fn test_schedule_view() {
        let schedule = make_test_schedule();
        let view = ScheduleView::from_schedule(&schedule);
        assert_eq!(view.transaction_ids().len(), 2);
        assert_eq!(view.global_order.len(), 3);
    }

    #[test]
    fn test_synthesize_basic() {
        let schedule = make_test_schedule();
        let synth = WitnessSynthesizer::with_default_config();
        let witness = synth.synthesize(&schedule, AnomalyClass::G1c).unwrap();
        assert_eq!(witness.num_transactions(), 2);
        assert!(witness.total_sql_statements() > 0);
        assert!(witness.description.contains("G1c"));
    }

    #[test]
    fn test_witness_render_script() {
        let schedule = make_test_schedule();
        let synth = WitnessSynthesizer::with_default_config();
        let witness = synth.synthesize(&schedule, AnomalyClass::G1a).unwrap();
        let script = witness.render_full_script();
        assert!(script.contains("CREATE TABLE"));
        assert!(script.contains("BEGIN"));
        assert!(script.contains("SELECT"));
        assert!(script.contains("UPDATE"));
    }

    #[test]
    fn test_schema_sql_generation() {
        let synth = WitnessSynthesizer::with_default_config();
        let schema = synth.generate_schema_sql();
        assert!(schema.len() >= 2); // DROP + CREATE + INSERTs
        assert!(schema[0].contains("DROP TABLE"));
        assert!(schema[1].contains("CREATE TABLE"));
    }

    #[test]
    fn test_expected_outcome_extraction() {
        let schedule = make_test_schedule();
        let synth = WitnessSynthesizer::with_default_config();
        let witness = synth.synthesize(&schedule, AnomalyClass::G1c).unwrap();
        let expected = ExpectedOutcomeExtractor::extract(&witness);
        assert_eq!(expected.len(), 2); // two reads with expected values
    }

    #[test]
    fn test_witness_display() {
        let schedule = make_test_schedule();
        let synth = WitnessSynthesizer::with_default_config();
        let witness = synth.synthesize(&schedule, AnomalyClass::G0).unwrap();
        let display = format!("{}", witness);
        assert!(display.contains("G0"));
    }

    #[test]
    fn test_isolation_level_sql() {
        assert_eq!(
            isolation_level_sql(&IsolationLevel::ReadCommitted),
            "READ COMMITTED"
        );
        assert_eq!(
            isolation_level_sql(&IsolationLevel::Serializable),
            "SERIALIZABLE"
        );
    }

    #[test]
    fn test_value_to_sql() {
        assert_eq!(value_to_sql(&Value::Integer(42)), "42");
        assert_eq!(value_to_sql(&Value::Text("hello".into())), "'hello'");
        assert_eq!(value_to_sql(&Value::Boolean(true)), "TRUE");
        assert_eq!(value_to_sql(&Value::Null), "NULL");
    }
}
