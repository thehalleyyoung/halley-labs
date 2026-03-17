//! Model extraction and interpretation from SMT SAT results.
//!
//! Converts raw SMT models into structured schedule representations.

use std::collections::HashMap;

use isospec_types::error::{IsoSpecError, IsoSpecResult};
use isospec_types::identifier::{ItemId, OperationId, ScheduleStepId, TableId, TransactionId};
use isospec_types::isolation::IsolationLevel;
use isospec_types::operation::{OpKind, Operation, ReadOp, WriteOp};
use isospec_types::schedule::{Schedule, ScheduleMetadata, ScheduleStep};
use isospec_types::value::Value;

use crate::encoding::VarNaming;
use crate::solver::{ModelValue, RawModel};

// ---------------------------------------------------------------------------
// SmtModel – typed wrapper around RawModel
// ---------------------------------------------------------------------------

/// A typed wrapper that provides convenient accessors for schedule-related
/// variables extracted from an SMT SAT result.
#[derive(Debug, Clone)]
pub struct SmtModel {
    raw: RawModel,
    naming: VarNaming,
    num_transactions: usize,
    ops_per_txn: usize,
}

impl SmtModel {
    pub fn new(raw: RawModel, naming: VarNaming, num_transactions: usize, ops_per_txn: usize) -> Self {
        Self {
            raw,
            naming,
            num_transactions,
            ops_per_txn,
        }
    }

    pub fn raw(&self) -> &RawModel {
        &self.raw
    }

    /// Get the position of operation (txn, op) in the total order.
    pub fn position(&self, txn: usize, op: usize) -> Option<i64> {
        let name = self.naming.position(txn, op);
        self.raw.get_int(&name)
    }

    /// Check if operation (txn, op) is active.
    pub fn is_active(&self, txn: usize, op: usize) -> bool {
        let name = self.naming.active(txn, op);
        self.raw.get_bool(&name).unwrap_or(false)
    }

    /// Get the data item index for operation (txn, op).
    pub fn item_index(&self, txn: usize, op: usize) -> Option<i64> {
        let name = self.naming.item(txn, op);
        self.raw.get_int(&name)
    }

    /// Get the read value for operation (txn, op).
    pub fn read_value(&self, txn: usize, op: usize) -> Option<i64> {
        let name = self.naming.read_value(txn, op);
        self.raw.get_int(&name)
    }

    /// Get the write value for operation (txn, op).
    pub fn write_value(&self, txn: usize, op: usize) -> Option<i64> {
        let name = self.naming.write_value(txn, op);
        self.raw.get_int(&name)
    }

    /// Check if operation (txn, op) is a read.
    pub fn is_read(&self, txn: usize, op: usize) -> bool {
        let name = self.naming.is_read(txn, op);
        self.raw.get_bool(&name).unwrap_or(false)
    }

    /// Check if operation (txn, op) is a write.
    pub fn is_write(&self, txn: usize, op: usize) -> bool {
        let name = self.naming.is_write(txn, op);
        self.raw.get_bool(&name).unwrap_or(false)
    }

    /// Check if txn i reads from txn j.
    pub fn reads_from(&self, reader: usize, writer: usize) -> bool {
        let name = self.naming.reads_from(reader, writer);
        self.raw.get_bool(&name).unwrap_or(false)
    }

    /// Check if a transaction committed.
    pub fn is_committed(&self, txn: usize) -> bool {
        let name = self.naming.committed(txn);
        self.raw.get_bool(&name).unwrap_or(false)
    }

    /// Check if a transaction aborted.
    pub fn is_aborted(&self, txn: usize) -> bool {
        let name = self.naming.aborted(txn);
        self.raw.get_bool(&name).unwrap_or(false)
    }

    /// Check version ordering between two transactions on a data item.
    pub fn version_order(&self, txn_i: usize, txn_j: usize, item: usize) -> bool {
        let name = self.naming.version_order(txn_i, txn_j, item);
        self.raw.get_bool(&name).unwrap_or(false)
    }

    /// Count active operations across all transactions.
    pub fn total_active_ops(&self) -> usize {
        let mut count = 0;
        for t in 0..self.num_transactions {
            for o in 0..self.ops_per_txn {
                if self.is_active(t, o) {
                    count += 1;
                }
            }
        }
        count
    }

    /// Get the list of active operation slots sorted by position.
    pub fn sorted_operations(&self) -> Vec<(usize, usize, i64)> {
        let mut ops = Vec::new();
        for t in 0..self.num_transactions {
            for o in 0..self.ops_per_txn {
                if self.is_active(t, o) {
                    if let Some(pos) = self.position(t, o) {
                        ops.push((t, o, pos));
                    }
                }
            }
        }
        ops.sort_by_key(|&(_, _, pos)| pos);
        ops
    }

    /// Identify read-from dependencies in the model.
    pub fn read_from_pairs(&self) -> Vec<(usize, usize)> {
        let mut pairs = Vec::new();
        for i in 0..self.num_transactions {
            for j in 0..self.num_transactions {
                if i != j && self.reads_from(i, j) {
                    pairs.push((i, j));
                }
            }
        }
        pairs
    }
}

// ---------------------------------------------------------------------------
// ModelInterpreter – interpret raw string values
// ---------------------------------------------------------------------------

/// Interprets arbitrary variable assignments from an SMT model.
pub struct ModelInterpreter;

impl ModelInterpreter {
    /// Parse a SMTLIB2 boolean value string.
    pub fn parse_bool(s: &str) -> Option<bool> {
        match s.trim() {
            "true" => Some(true),
            "false" => Some(false),
            _ => None,
        }
    }

    /// Parse a SMTLIB2 integer value string, including negative (- N).
    pub fn parse_int(s: &str) -> Option<i64> {
        let trimmed = s.trim();
        if let Ok(v) = trimmed.parse::<i64>() {
            return Some(v);
        }
        // Handle (- N) form
        let inner = trimmed.strip_prefix("(-")?.strip_suffix(')')?.trim();
        let v = inner.parse::<i64>().ok()?;
        Some(-v)
    }

    /// Parse a bitvector literal.
    pub fn parse_bitvec(s: &str) -> Option<(u64, u32)> {
        let trimmed = s.trim();
        if let Some(bv) = trimmed.strip_prefix("#b") {
            let width = bv.len() as u32;
            let value = u64::from_str_radix(bv, 2).ok()?;
            return Some((value, width));
        }
        if let Some(hex) = trimmed.strip_prefix("#x") {
            let width = (hex.len() as u32) * 4;
            let value = u64::from_str_radix(hex, 16).ok()?;
            return Some((value, width));
        }
        None
    }

    /// Convert a model value to an isospec Value.
    pub fn to_isospec_value(mv: &ModelValue) -> Value {
        match mv {
            ModelValue::Bool(b) => Value::Boolean(*b),
            ModelValue::Int(i) => Value::Integer(*i),
            ModelValue::Str(s) => Value::Text(s.clone()),
            ModelValue::BitVec { value, .. } => Value::Integer(*value as i64),
        }
    }

    /// Extract all integer assignments from a raw model with a given prefix.
    pub fn extract_ints_with_prefix(model: &RawModel, prefix: &str) -> HashMap<String, i64> {
        let mut result = HashMap::new();
        for (name, value) in model.iter() {
            if name.starts_with(prefix) {
                if let Some(i) = value.as_int() {
                    result.insert(name.clone(), i);
                }
            }
        }
        result
    }

    /// Extract all boolean assignments from a raw model with a given prefix.
    pub fn extract_bools_with_prefix(model: &RawModel, prefix: &str) -> HashMap<String, bool> {
        let mut result = HashMap::new();
        for (name, value) in model.iter() {
            if name.starts_with(prefix) {
                if let Some(b) = value.as_bool() {
                    result.insert(name.clone(), b);
                }
            }
        }
        result
    }
}

// ---------------------------------------------------------------------------
// ScheduleExtractor
// ---------------------------------------------------------------------------

/// Extracts a structured `Schedule` from an `SmtModel`.
pub struct ScheduleExtractor {
    naming: VarNaming,
}

impl ScheduleExtractor {
    pub fn new(naming: VarNaming) -> Self {
        Self { naming }
    }

    /// Convert an SMT model into a `Schedule`.
    pub fn extract(&self, model: &SmtModel) -> IsoSpecResult<Schedule> {
        let sorted = model.sorted_operations();
        if sorted.is_empty() {
            return Err(IsoSpecError::smt_solver("model has no active operations"));
        }

        let mut steps = Vec::new();
        let mut op_counter: u64 = 0;

        for (txn_idx, op_idx, _pos) in &sorted {
            let txn_id = TransactionId::new(*txn_idx as u64);
            let op_id = OperationId::new(op_counter);
            op_counter += 1;

            let item_idx = model.item_index(*txn_idx, *op_idx).unwrap_or(0);
            let item_id = ItemId::new(item_idx as u64);
            let table_id = TableId::new(0);

            let kind = if model.is_read(*txn_idx, *op_idx) {
                let val = model.read_value(*txn_idx, *op_idx);
                OpKind::Read(ReadOp {
                    table: table_id,
                    item: item_id,
                    columns: Vec::new(),
                    value_read: val.map(Value::Integer),
                    version_read: None,
                })
            } else if model.is_write(*txn_idx, *op_idx) {
                let val = model
                    .write_value(*txn_idx, *op_idx)
                    .unwrap_or(0);
                OpKind::Write(WriteOp {
                    table: table_id,
                    item: item_id,
                    columns: Vec::new(),
                    old_value: None,
                    new_value: Value::Integer(val),
                    version_written: None,
                })
            } else {
                OpKind::Read(ReadOp {
                    table: table_id,
                    item: item_id,
                    columns: Vec::new(),
                    value_read: None,
                    version_read: None,
                })
            };

            let operation = Operation {
                id: op_id,
                txn_id,
                kind,
                timestamp: *_pos as u64,
            };

            let step = ScheduleStep {
                id: ScheduleStepId::new(op_counter - 1),
                txn_id,
                operation,
                position: *_pos as usize,
            };
            steps.push(step);
        }

        // Collect transaction IDs
        let mut txn_ids: Vec<TransactionId> = Vec::new();
        for t in 0..model.num_transactions {
            if model.is_active(t, 0) {
                txn_ids.push(TransactionId::new(t as u64));
            }
        }

        let metadata = ScheduleMetadata {
            engine: None,
            isolation_level: None,
            is_witness: false,
            anomaly_class: None,
            generation_method: Some("extracted from SMT model".to_string()),
        };

        Ok(Schedule {
            steps,
            transaction_order: txn_ids,
            metadata,
        })
    }

    /// Extract a summary of read-from dependencies.
    pub fn extract_read_from_summary(
        &self,
        model: &SmtModel,
    ) -> Vec<ReadFromDependency> {
        let mut deps = Vec::new();
        for (reader, writer) in model.read_from_pairs() {
            // Find which operations are involved
            for ro in 0..model.ops_per_txn {
                if !model.is_active(reader, ro) || !model.is_read(reader, ro) {
                    continue;
                }
                for wo in 0..model.ops_per_txn {
                    if !model.is_active(writer, wo) || !model.is_write(writer, wo) {
                        continue;
                    }
                    let r_item = model.item_index(reader, ro);
                    let w_item = model.item_index(writer, wo);
                    if r_item == w_item {
                        let r_val = model.read_value(reader, ro);
                        let w_val = model.write_value(writer, wo);
                        if r_val == w_val {
                            deps.push(ReadFromDependency {
                                reader_txn: reader,
                                reader_op: ro,
                                writer_txn: writer,
                                writer_op: wo,
                                item: r_item.unwrap_or(0) as usize,
                                value: r_val.unwrap_or(0),
                            });
                        }
                    }
                }
            }
        }
        deps
    }
}

/// A read-from dependency between two operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadFromDependency {
    pub reader_txn: usize,
    pub reader_op: usize,
    pub writer_txn: usize,
    pub writer_op: usize,
    pub item: usize,
    pub value: i64,
}

// ---------------------------------------------------------------------------
// ScheduleFormatter – pretty print extracted schedules
// ---------------------------------------------------------------------------

/// Formats a `Schedule` for human-readable display.
pub struct ScheduleFormatter;

impl ScheduleFormatter {
    /// Format a schedule as a text table.
    pub fn format_text(schedule: &Schedule) -> String {
        let mut lines = Vec::new();
        lines.push(format!(
            "Schedule ({} steps, {} transactions)",
            schedule.steps.len(),
            schedule.transaction_order.len(),
        ));
        lines.push("-".repeat(60));
        lines.push(format!(
            "{:<6} {:<10} {:<12} {:<20}",
            "Pos", "Txn", "Type", "Details"
        ));
        lines.push("-".repeat(60));

        for step in &schedule.steps {
            let op = &step.operation;
            let type_str = match &op.kind {
                OpKind::Read(_) => "READ",
                OpKind::Write(_) => "WRITE",
                OpKind::Insert(_) => "INSERT",
                OpKind::Delete(_) => "DELETE",
                OpKind::Begin(_) => "BEGIN",
                OpKind::Commit(_) => "COMMIT",
                OpKind::Abort(_) => "ABORT",
                OpKind::Lock(_) => "LOCK",
                OpKind::PredicateRead(_) => "PRED_READ",
                OpKind::PredicateWrite(_) => "PRED_WRITE",
            };
            let details = match &op.kind {
                OpKind::Read(r) => {
                    let val_str = match &r.value_read {
                        Some(v) => format!("{}", v),
                        None => "?".to_string(),
                    };
                    format!("item={} val={}", r.item, val_str)
                }
                OpKind::Write(w) => format!("item={} val={}", w.item, w.new_value),
                _ => String::new(),
            };

            lines.push(format!(
                "{:<6} {:<10} {:<12} {}",
                step.position,
                format!("T{}", op.txn_id),
                type_str,
                details
            ));
        }

        if let Some(ref desc) = schedule.metadata.generation_method {
            lines.push(String::new());
            lines.push(format!("Description: {}", desc));
        }

        lines.join("\n")
    }

    /// Format a schedule as a DOT digraph for visualization.
    pub fn format_dot(schedule: &Schedule) -> String {
        let mut lines = Vec::new();
        lines.push("digraph schedule {".to_string());
        lines.push("  rankdir=TB;".to_string());
        lines.push("  node [shape=record];".to_string());

        for step in &schedule.steps {
            let op = &step.operation;
            let label = match &op.kind {
                OpKind::Read(r) => format!("R({})", r.item),
                OpKind::Write(w) => format!("W({})", w.item),
                _ => format!("{:?}", op.kind),
            };
            lines.push(format!(
                "  op{} [label=\"{{T{}|{}|pos={}}}\"];",
                step.position, op.txn_id, label, step.position,
            ));
        }

        // Edges for sequential order
        for i in 0..schedule.steps.len().saturating_sub(1) {
            lines.push(format!(
                "  op{} -> op{};",
                schedule.steps[i].position,
                schedule.steps[i + 1].position,
            ));
        }

        lines.push("}".to_string());
        lines.join("\n")
    }
}

// ---------------------------------------------------------------------------
// ModelValidator – sanity checks on extracted models
// ---------------------------------------------------------------------------

/// Validates that a model satisfies expected structural invariants.
pub struct ModelValidator;

impl ModelValidator {
    /// Check that all positions are unique among active operations.
    pub fn check_unique_positions(model: &SmtModel) -> Vec<String> {
        let mut errors = Vec::new();
        let sorted = model.sorted_operations();
        for i in 0..sorted.len() {
            for j in (i + 1)..sorted.len() {
                if sorted[i].2 == sorted[j].2 {
                    errors.push(format!(
                        "duplicate position {}: T{}:Op{} and T{}:Op{}",
                        sorted[i].2, sorted[i].0, sorted[i].1, sorted[j].0, sorted[j].1,
                    ));
                }
            }
        }
        errors
    }

    /// Check that operations within the same transaction are correctly ordered.
    pub fn check_intra_txn_order(model: &SmtModel) -> Vec<String> {
        let mut errors = Vec::new();
        for t in 0..model.num_transactions {
            let mut prev_pos: Option<i64> = None;
            for o in 0..model.ops_per_txn {
                if !model.is_active(t, o) {
                    break;
                }
                if let Some(pos) = model.position(t, o) {
                    if let Some(prev) = prev_pos {
                        if pos <= prev {
                            errors.push(format!(
                                "T{}: op {} at pos {} not after op {} at pos {}",
                                t,
                                o,
                                pos,
                                o - 1,
                                prev,
                            ));
                        }
                    }
                    prev_pos = Some(pos);
                }
            }
        }
        errors
    }

    /// Check that each transaction is either committed or aborted, not both.
    pub fn check_txn_status(model: &SmtModel) -> Vec<String> {
        let mut errors = Vec::new();
        for t in 0..model.num_transactions {
            let c = model.is_committed(t);
            let a = model.is_aborted(t);
            if c && a {
                errors.push(format!("T{}: both committed and aborted", t));
            }
            if !c && !a {
                errors.push(format!("T{}: neither committed nor aborted", t));
            }
        }
        errors
    }

    /// Run all validations and return combined errors.
    pub fn validate_all(model: &SmtModel) -> Vec<String> {
        let mut errors = Vec::new();
        errors.extend(Self::check_unique_positions(model));
        errors.extend(Self::check_intra_txn_order(model));
        errors.extend(Self::check_txn_status(model));
        errors
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::ModelValue;

    fn make_test_model() -> SmtModel {
        let naming = VarNaming::default();
        let mut raw = RawModel::new();

        // Two transactions, 2 ops each
        // T0: read item 0, write item 0 (positions 0, 2)
        // T1: write item 0, read item 0 (positions 1, 3)
        raw.insert(naming.active(0, 0), ModelValue::Bool(true));
        raw.insert(naming.active(0, 1), ModelValue::Bool(true));
        raw.insert(naming.active(1, 0), ModelValue::Bool(true));
        raw.insert(naming.active(1, 1), ModelValue::Bool(true));

        raw.insert(naming.position(0, 0), ModelValue::Int(0));
        raw.insert(naming.position(0, 1), ModelValue::Int(2));
        raw.insert(naming.position(1, 0), ModelValue::Int(1));
        raw.insert(naming.position(1, 1), ModelValue::Int(3));

        raw.insert(naming.is_read(0, 0), ModelValue::Bool(true));
        raw.insert(naming.is_write(0, 0), ModelValue::Bool(false));
        raw.insert(naming.is_read(0, 1), ModelValue::Bool(false));
        raw.insert(naming.is_write(0, 1), ModelValue::Bool(true));

        raw.insert(naming.is_read(1, 0), ModelValue::Bool(false));
        raw.insert(naming.is_write(1, 0), ModelValue::Bool(true));
        raw.insert(naming.is_read(1, 1), ModelValue::Bool(true));
        raw.insert(naming.is_write(1, 1), ModelValue::Bool(false));

        raw.insert(naming.item(0, 0), ModelValue::Int(0));
        raw.insert(naming.item(0, 1), ModelValue::Int(0));
        raw.insert(naming.item(1, 0), ModelValue::Int(0));
        raw.insert(naming.item(1, 1), ModelValue::Int(0));

        raw.insert(naming.read_value(0, 0), ModelValue::Int(10));
        raw.insert(naming.write_value(0, 1), ModelValue::Int(20));
        raw.insert(naming.write_value(1, 0), ModelValue::Int(30));
        raw.insert(naming.read_value(1, 1), ModelValue::Int(20));

        raw.insert(naming.committed(0), ModelValue::Bool(true));
        raw.insert(naming.aborted(0), ModelValue::Bool(false));
        raw.insert(naming.committed(1), ModelValue::Bool(true));
        raw.insert(naming.aborted(1), ModelValue::Bool(false));

        SmtModel::new(raw, naming, 2, 2)
    }

    #[test]
    fn test_smt_model_accessors() {
        let model = make_test_model();
        assert!(model.is_active(0, 0));
        assert!(model.is_active(1, 1));
        assert_eq!(model.position(0, 0), Some(0));
        assert_eq!(model.position(1, 0), Some(1));
        assert!(model.is_read(0, 0));
        assert!(model.is_write(0, 1));
        assert!(model.is_committed(0));
        assert!(!model.is_aborted(0));
    }

    #[test]
    fn test_sorted_operations() {
        let model = make_test_model();
        let sorted = model.sorted_operations();
        assert_eq!(sorted.len(), 4);
        assert_eq!(sorted[0], (0, 0, 0));
        assert_eq!(sorted[1], (1, 0, 1));
        assert_eq!(sorted[2], (0, 1, 2));
        assert_eq!(sorted[3], (1, 1, 3));
    }

    #[test]
    fn test_total_active_ops() {
        let model = make_test_model();
        assert_eq!(model.total_active_ops(), 4);
    }

    #[test]
    fn test_schedule_extraction() {
        let model = make_test_model();
        let extractor = ScheduleExtractor::new(VarNaming::default());
        let schedule = extractor.extract(&model).unwrap();
        assert_eq!(schedule.steps.len(), 4);
        assert_eq!(schedule.transaction_order.len(), 2);
        assert_eq!(schedule.steps[0].position, 0);
        assert_eq!(schedule.steps[3].position, 3);
    }

    #[test]
    fn test_model_interpreter_parse_bool() {
        assert_eq!(ModelInterpreter::parse_bool("true"), Some(true));
        assert_eq!(ModelInterpreter::parse_bool("false"), Some(false));
        assert_eq!(ModelInterpreter::parse_bool("maybe"), None);
    }

    #[test]
    fn test_model_interpreter_parse_int() {
        assert_eq!(ModelInterpreter::parse_int("42"), Some(42));
        assert_eq!(ModelInterpreter::parse_int("(- 5)"), Some(-5));
        assert_eq!(ModelInterpreter::parse_int("abc"), None);
    }

    #[test]
    fn test_model_interpreter_parse_bitvec() {
        assert_eq!(ModelInterpreter::parse_bitvec("#b1010"), Some((10, 4)));
        assert_eq!(ModelInterpreter::parse_bitvec("#xff"), Some((255, 8)));
    }

    #[test]
    fn test_model_validator_unique_positions() {
        let model = make_test_model();
        let errors = ModelValidator::check_unique_positions(&model);
        assert!(errors.is_empty(), "expected no duplicate positions");
    }

    #[test]
    fn test_model_validator_intra_order() {
        let model = make_test_model();
        let errors = ModelValidator::check_intra_txn_order(&model);
        assert!(errors.is_empty(), "expected correct intra-txn order");
    }

    #[test]
    fn test_model_validator_txn_status() {
        let model = make_test_model();
        let errors = ModelValidator::check_txn_status(&model);
        assert!(errors.is_empty(), "expected valid txn status");
    }

    #[test]
    fn test_schedule_formatter_text() {
        let model = make_test_model();
        let extractor = ScheduleExtractor::new(VarNaming::default());
        let schedule = extractor.extract(&model).unwrap();
        let text = ScheduleFormatter::format_text(&schedule);
        assert!(text.contains("Schedule"));
        assert!(text.contains("READ"));
        assert!(text.contains("WRITE"));
    }

    #[test]
    fn test_schedule_formatter_dot() {
        let model = make_test_model();
        let extractor = ScheduleExtractor::new(VarNaming::default());
        let schedule = extractor.extract(&model).unwrap();
        let dot = ScheduleFormatter::format_dot(&schedule);
        assert!(dot.contains("digraph schedule"));
        assert!(dot.contains("->"));
    }

    #[test]
    fn test_extract_ints_with_prefix() {
        let naming = VarNaming::default();
        let mut raw = RawModel::new();
        raw.insert(naming.position(0, 0), ModelValue::Int(5));
        raw.insert(naming.position(0, 1), ModelValue::Int(7));
        raw.insert(naming.active(0, 0), ModelValue::Bool(true));

        let ints = ModelInterpreter::extract_ints_with_prefix(&raw, "s_pos_");
        assert_eq!(ints.len(), 2);
        assert_eq!(ints[&naming.position(0, 0)], 5);
    }
}
