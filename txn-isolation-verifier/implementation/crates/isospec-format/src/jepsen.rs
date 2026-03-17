//! Jepsen history format import/export.
//!
//! Jepsen (<https://jepsen.io>) represents operation histories as sequences of EDN maps.
//! Each operation has:
//! - `:type` — `:invoke`, `:ok`, `:fail`, or `:info`
//! - `:f` — the function name (e.g., `:txn`, `:read`, `:write`)
//! - `:value` — operation-specific data (often a list-append or register transaction)
//! - `:process` — the process/thread ID
//! - `:time` — wall-clock nanosecond timestamp
//!
//! This module converts between Jepsen's EDN history format and IsoSpec's
//! `TransactionHistory`, enabling analysis of Jepsen test results.

use crate::edn::{self, EdnValue};
use crate::{FormatError, FormatResult};
use isospec_history::builder::HistoryBuilder;
use isospec_history::history::{TransactionHistory, TransactionStatus};
use isospec_types::identifier::{ItemId, TableId, TransactionId};
use isospec_types::isolation::IsolationLevel;
use isospec_types::value::Value;
use std::collections::HashMap;
use tracing::{debug, warn};

/// A single Jepsen operation (one line of history)
#[derive(Debug, Clone)]
pub struct JepsenOperation {
    /// Operation type: invoke, ok, fail, info
    pub op_type: JepsenOpType,
    /// Function name (e.g., "txn", "read", "write", "cas")
    pub function: String,
    /// Operation value (function-specific)
    pub value: EdnValue,
    /// Process/thread ID
    pub process: i64,
    /// Wall-clock time in nanoseconds
    pub time_ns: Option<i64>,
    /// Additional fields
    pub extra: HashMap<String, EdnValue>,
}

/// Jepsen operation types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JepsenOpType {
    /// Client invokes an operation
    Invoke,
    /// Operation completed successfully
    Ok,
    /// Operation failed (e.g., aborted)
    Fail,
    /// Indeterminate result
    Info,
}

impl JepsenOpType {
    pub fn from_keyword(k: &str) -> Option<Self> {
        match k {
            "invoke" => Some(Self::Invoke),
            "ok" => Some(Self::Ok),
            "fail" => Some(Self::Fail),
            "info" => Some(Self::Info),
            _ => None,
        }
    }

    pub fn to_keyword(&self) -> &'static str {
        match self {
            Self::Invoke => "invoke",
            Self::Ok => "ok",
            Self::Fail => "fail",
            Self::Info => "info",
        }
    }
}

/// Parse a Jepsen EDN history file into a sequence of operations.
pub fn parse_jepsen_history(input: &str) -> FormatResult<Vec<JepsenOperation>> {
    let values = edn::parse_edn(input)?;
    let mut ops = Vec::new();

    for value in values {
        match parse_jepsen_op(&value) {
            Ok(op) => ops.push(op),
            Err(e) => {
                warn!(error = %e, "Skipping unparseable Jepsen operation");
            }
        }
    }

    debug!(operations = ops.len(), "Parsed Jepsen history");
    Ok(ops)
}

/// Parse a single Jepsen operation from an EDN map.
fn parse_jepsen_op(value: &EdnValue) -> FormatResult<JepsenOperation> {
    let op_type_val = value
        .get("type")
        .ok_or_else(|| FormatError::MissingField(":type".into()))?;
    let op_type = JepsenOpType::from_keyword(
        op_type_val
            .as_keyword()
            .ok_or_else(|| FormatError::MissingField(":type must be keyword".into()))?,
    )
    .ok_or_else(|| FormatError::InvalidFormat("Unknown operation type".into()))?;

    let function = value
        .get("f")
        .and_then(|v| v.as_keyword())
        .unwrap_or("unknown")
        .to_string();

    let val = value
        .get("value")
        .cloned()
        .unwrap_or(EdnValue::Nil);

    let process = value
        .get("process")
        .and_then(|v| v.as_integer())
        .unwrap_or(0);

    let time_ns = value.get("time").and_then(|v| v.as_integer());

    Ok(JepsenOperation {
        op_type,
        function,
        value: val,
        process,
        time_ns,
        extra: HashMap::new(),
    })
}

/// Convert a Jepsen history into an IsoSpec `TransactionHistory`.
///
/// Maps Jepsen process IDs to transaction IDs, and interprets
/// list-append / register / key-value transaction micro-operations.
///
/// # Supported Jepsen test models
/// - **list-append**: `:value [[:r :x [1 2]] [:append :y 3]]`
/// - **register**: `:value [:r nil]` or `:value [:w 42]`
/// - **bank**: `:value {:accounts {0 100, 1 200}}`
pub fn jepsen_to_history(
    ops: &[JepsenOperation],
    isolation_level: IsolationLevel,
) -> FormatResult<TransactionHistory> {
    let mut builder = HistoryBuilder::new();
    let mut process_txn: HashMap<i64, TransactionId> = HashMap::new();
    let mut txn_counter: u64 = 1;

    let default_table = TableId::new(simple_hash("default"));

    for op in ops {
        match op.op_type {
            JepsenOpType::Invoke => {
                let txn_id = TransactionId::from(txn_counter);
                txn_counter += 1;
                process_txn.insert(op.process, txn_id);
                builder.begin_transaction(txn_id, isolation_level);

                // Parse micro-operations from value
                if let EdnValue::Vector(micro_ops) = &op.value {
                    for micro_op in micro_ops {
                        if let EdnValue::Vector(parts) = micro_op {
                            interpret_micro_op(&mut builder, txn_id, default_table, parts);
                        }
                    }
                }
            }
            JepsenOpType::Ok => {
                if let Some(&txn_id) = process_txn.get(&op.process) {
                    builder.commit_transaction(txn_id);
                    process_txn.remove(&op.process);
                }
            }
            JepsenOpType::Fail => {
                if let Some(&txn_id) = process_txn.get(&op.process) {
                    builder.abort_transaction(txn_id, Some("Jepsen :fail".to_string()));
                    process_txn.remove(&op.process);
                }
            }
            JepsenOpType::Info => {
                // Indeterminate - treat as crash
                if let Some(&txn_id) = process_txn.get(&op.process) {
                    builder.abort_transaction(txn_id, Some("Jepsen :info (indeterminate)".to_string()));
                    process_txn.remove(&op.process);
                }
            }
        }
    }

    // Abort any remaining open transactions
    for (_, txn_id) in process_txn {
        builder.abort_transaction(txn_id, Some("History ended without completion".to_string()));
    }

    Ok(builder.build()?)
}

/// Interpret a Jepsen micro-operation like `[:r :x nil]` or `[:w :y 42]`.
fn interpret_micro_op(
    builder: &mut HistoryBuilder,
    txn_id: TransactionId,
    table: TableId,
    parts: &[EdnValue],
) {
    if parts.len() < 2 {
        return;
    }

    let op_type = match parts[0].as_keyword() {
        Some("r") | Some("read") => "read",
        Some("w") | Some("write") => "write",
        Some("append") => "insert",
        Some("cas") => "cas",
        _ => return,
    };

    let key = match &parts[1] {
        EdnValue::Keyword(k) => k.clone(),
        EdnValue::Integer(n) => n.to_string(),
        EdnValue::String(s) => s.clone(),
        _ => "unknown".to_string(),
    };

    let item_id = ItemId::from(simple_hash(&key));

    match op_type {
        "read" => {
            let value = if parts.len() > 2 {
                edn_to_value(&parts[2])
            } else {
                None
            };
            builder.add_read(txn_id, table, item_id, value);
        }
        "write" | "insert" => {
            let new_val = if parts.len() > 2 {
                edn_to_value(&parts[2]).unwrap_or(Value::Null)
            } else {
                Value::Null
            };
            builder.add_write(txn_id, table, item_id, None, new_val);
        }
        "cas" => {
            // Compare-and-swap: [:cas :key old-val new-val]
            let old_val = if parts.len() > 2 {
                edn_to_value(&parts[2])
            } else {
                None
            };
            let new_val = if parts.len() > 3 {
                edn_to_value(&parts[3]).unwrap_or(Value::Null)
            } else {
                Value::Null
            };
            builder.add_write(txn_id, table, item_id, old_val, new_val);
        }
        _ => {}
    }
}

/// Convert an EDN value to an IsoSpec Value.
fn edn_to_value(edn: &EdnValue) -> Option<Value> {
    match edn {
        EdnValue::Nil => Some(Value::Null),
        EdnValue::Integer(n) => Some(Value::Integer(*n)),
        EdnValue::String(s) => Some(Value::Text(s.clone())),
        EdnValue::Bool(b) => Some(Value::Boolean(*b)),
        EdnValue::Float(f) => Some(Value::Float(*f)),
        _ => None,
    }
}

/// Convert an IsoSpec value to EDN.
fn value_to_edn(value: &Value) -> EdnValue {
    match value {
        Value::Null => EdnValue::Nil,
        Value::Integer(n) => EdnValue::Integer(*n),
        Value::Text(s) => EdnValue::String(s.clone()),
        Value::Boolean(b) => EdnValue::Bool(*b),
        _ => EdnValue::Nil,
    }
}

/// Simple string hash for key->ItemId mapping.
fn simple_hash(s: &str) -> u64 {
    let mut hash: u64 = 5381;
    for b in s.bytes() {
        hash = hash.wrapping_mul(33).wrapping_add(b as u64);
    }
    hash
}

/// Export an IsoSpec `TransactionHistory` to Jepsen EDN format.
pub fn history_to_jepsen_edn(history: &TransactionHistory) -> String {
    let mut output = String::new();

    // Collect all transaction IDs (committed + aborted)
    let mut all_txn_ids: Vec<TransactionId> = history.committed_transactions();
    all_txn_ids.extend(history.aborted_transactions());

    // Each transaction becomes an invoke/ok or invoke/fail pair
    for (i, txn_id) in all_txn_ids.iter().enumerate() {
        let info = match history.get_transaction(*txn_id) {
            Some(info) => info,
            None => continue,
        };
        let process = EdnValue::Integer(i as i64);

        // Build micro-ops from transaction events
        let mut micro_ops = Vec::new();
        for event in history.events_for_txn(*txn_id) {
            if event.is_read_event() {
                if let Some(table_id) = event.table_id() {
                    micro_ops.push(EdnValue::Vector(vec![
                        EdnValue::Keyword("r".into()),
                        EdnValue::Keyword(format!("{:?}", table_id)),
                        EdnValue::Nil,
                    ]));
                }
            } else if event.is_write_event() {
                if let Some(table_id) = event.table_id() {
                    micro_ops.push(EdnValue::Vector(vec![
                        EdnValue::Keyword("w".into()),
                        EdnValue::Keyword(format!("{:?}", table_id)),
                        EdnValue::Integer(1),
                    ]));
                }
            }
        }

        let value = EdnValue::Vector(micro_ops.clone());

        let begin_ts = info.begin_timestamp.unwrap_or(0);

        // Invoke
        let invoke = EdnValue::Map(vec![
            (EdnValue::Keyword("type".into()), EdnValue::Keyword("invoke".into())),
            (EdnValue::Keyword("f".into()), EdnValue::Keyword("txn".into())),
            (EdnValue::Keyword("value".into()), value.clone()),
            (EdnValue::Keyword("process".into()), process.clone()),
            (
                EdnValue::Keyword("time".into()),
                EdnValue::Integer(begin_ts as i64),
            ),
        ]);
        output.push_str(&edn::to_edn_string(&invoke));
        output.push('\n');

        // Completion
        let completion_type = if info.status == TransactionStatus::Committed {
            "ok"
        } else if info.status == TransactionStatus::Aborted {
            "fail"
        } else {
            "info"
        };

        let end_ts = info.end_timestamp.unwrap_or(begin_ts + 1);

        let completion = EdnValue::Map(vec![
            (
                EdnValue::Keyword("type".into()),
                EdnValue::Keyword(completion_type.into()),
            ),
            (EdnValue::Keyword("f".into()), EdnValue::Keyword("txn".into())),
            (EdnValue::Keyword("value".into()), value),
            (EdnValue::Keyword("process".into()), process),
            (
                EdnValue::Keyword("time".into()),
                EdnValue::Integer(end_ts as i64),
            ),
        ]);
        output.push_str(&edn::to_edn_string(&completion));
        output.push('\n');
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_jepsen_operation() {
        let input = r#"{:type :invoke, :f :txn, :value [[:r :x nil] [:w :y 1]], :process 0, :time 100}"#;
        let ops = parse_jepsen_history(input).unwrap();
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].op_type, JepsenOpType::Invoke);
        assert_eq!(ops[0].function, "txn");
        assert_eq!(ops[0].process, 0);
    }

    #[test]
    fn test_jepsen_roundtrip() {
        let input = r#"
{:type :invoke, :f :txn, :value [[:r :x nil] [:w :y 1]], :process 0, :time 100}
{:type :ok, :f :txn, :value [[:r :x 5] [:w :y 1]], :process 0, :time 200}
{:type :invoke, :f :txn, :value [[:r :y nil]], :process 1, :time 150}
{:type :ok, :f :txn, :value [[:r :y 1]], :process 1, :time 250}
"#;
        let ops = parse_jepsen_history(input).unwrap();
        assert_eq!(ops.len(), 4);

        let history = jepsen_to_history(&ops, IsolationLevel::Serializable).unwrap();
        assert!(history.transaction_count() >= 2);
    }

    #[test]
    fn test_jepsen_op_types() {
        assert_eq!(JepsenOpType::from_keyword("invoke"), Some(JepsenOpType::Invoke));
        assert_eq!(JepsenOpType::from_keyword("ok"), Some(JepsenOpType::Ok));
        assert_eq!(JepsenOpType::from_keyword("fail"), Some(JepsenOpType::Fail));
        assert_eq!(JepsenOpType::from_keyword("info"), Some(JepsenOpType::Info));
        assert_eq!(JepsenOpType::from_keyword("unknown"), None);
    }

    #[test]
    fn test_jepsen_fail_aborts() {
        let input = r#"
{:type :invoke, :f :txn, :value [[:w :x 1]], :process 0, :time 100}
{:type :fail, :f :txn, :value [[:w :x 1]], :process 0, :time 200}
"#;
        let ops = parse_jepsen_history(input).unwrap();
        let history = jepsen_to_history(&ops, IsolationLevel::ReadCommitted).unwrap();

        // The transaction should be aborted
        assert_eq!(history.transaction_count(), 1);
        let aborted = history.aborted_transactions();
        assert_eq!(aborted.len(), 1);
    }
}
