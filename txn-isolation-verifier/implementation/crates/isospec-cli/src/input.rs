//! Input parsing: load workloads from JSON files, parse SQL into IR programs,
//! and validate input schemas.

use std::collections::HashMap;
use std::path::Path;

use isospec_types::identifier::{IdAllocator, TransactionId, WorkloadId};
use isospec_types::ir::{
    IrExpr, IrInsert, IrProgram, IrSelect, IrStatement, IrTransaction, IrUpdate, IrDelete,
};
use isospec_types::isolation::IsolationLevel;
use isospec_types::predicate::Predicate;
use isospec_types::schema::Schema;
use isospec_types::value::Value;
use isospec_types::workload::{Workload, WorkloadParameters};
use serde_json::{Map as JsonMap, Value as JsonValue};

// ---------------------------------------------------------------------------
// Workload loading
// ---------------------------------------------------------------------------

/// Load a workload from a JSON file.
///
/// Expected JSON schema:
/// ```json
/// {
///   "name": "example",
///   "transactions": [
///     {
///       "label": "T1",
///       "isolation": "serializable",
///       "statements": [
///         {"type": "select", "table": "t", "columns": ["id"], "predicate": "id = 1"},
///         {"type": "update", "table": "t", "set": {"val": "42"}, "predicate": "id = 1"},
///         {"type": "insert", "table": "t", "columns": ["id","val"], "values": [["3","99"]]},
///         {"type": "delete", "table": "t", "predicate": "id = 1"}
///       ]
///     }
///   ]
/// }
/// ```
pub fn load_workload(path: &Path) -> Result<Workload, Box<dyn std::error::Error>> {
    let contents = std::fs::read_to_string(path)
        .map_err(|e| format!("Cannot read workload file {:?}: {}", path, e))?;

    parse_workload_json(&contents)
}

/// Parse a workload from a JSON string.
pub fn parse_workload_json(json: &str) -> Result<Workload, Box<dyn std::error::Error>> {
    let root: JsonValue = serde_json::from_str(json)?;
    let root_obj = root
        .as_object()
        .ok_or("Expected JSON object")?;
    let nested_workload = root_obj
        .get("workload")
        .and_then(JsonValue::as_object);

    let name = json_str(root_obj, "name")
        .or_else(|| nested_workload.and_then(|obj| json_str(obj, "name")))
        .unwrap_or_else(|| "unnamed".to_string());

    let txn_array = root_obj
        .get("transactions")
        .or_else(|| nested_workload.and_then(|obj| obj.get("transactions")))
        .and_then(JsonValue::as_array)
        .ok_or("Missing 'transactions' array")?;

    let default_isolation = json_str(root_obj, "isolation_level")
        .or_else(|| nested_workload.and_then(|obj| json_str(obj, "isolation_level")))
        .or_else(|| json_str(root_obj, "source_isolation"))
        .unwrap_or_else(|| "serializable".to_string());

    let mut txn_alloc: IdAllocator<TransactionId> = IdAllocator::new();
    let mut transactions = Vec::new();

    for txn_value in txn_array {
        if let Some(txn) = parse_transaction_value(txn_value, &mut txn_alloc, &default_isolation) {
            transactions.push(txn);
        }
    }

    let wl_id = WorkloadId::new(1);
    let txn_count = transactions.len();
    let max_ops = transactions.iter().map(|t| t.statements.len()).max().unwrap_or(0);
    let mut metadata = HashMap::new();
    if let Some(description) = json_str(root_obj, "description") {
        metadata.insert("description".to_string(), description);
    }

    let program = IrProgram {
        id: wl_id,
        name: name.clone(),
        transactions,
        schema_name: "loaded".to_string(),
        metadata,
    };

    let schema = Schema::new();
    let annotations = collect_workload_annotations(root_obj);

    Ok(Workload {
        id: wl_id,
        name,
        program,
        schema,
        parameters: WorkloadParameters {
            transaction_bound: txn_count,
            operation_bound: max_ops,
            data_item_bound: 10,
            repetitions: 1,
        },
        annotations,
    })
}

// ---------------------------------------------------------------------------
// SQL-like parsing
// ---------------------------------------------------------------------------

/// Parse a simplified SQL statement string into an IR statement.
pub fn parse_sql_statement(sql: &str) -> Option<IrStatement> {
    let sql = sql.trim();
    let upper = sql.to_uppercase();

    if upper.starts_with("SELECT") {
        parse_sql_select(sql)
    } else if upper.starts_with("UPDATE") {
        parse_sql_update(sql)
    } else if upper.starts_with("INSERT") {
        parse_sql_insert(sql)
    } else if upper.starts_with("DELETE") {
        parse_sql_delete(sql)
    } else {
        None
    }
}

fn parse_sql_select(sql: &str) -> Option<IrStatement> {
    // SELECT col1, col2 FROM table WHERE pred
    let upper = sql.to_uppercase();
    let from_pos = upper.find("FROM")?;
    let cols_part = &sql[7..from_pos].trim();
    let columns: Vec<String> = cols_part.split(',').map(|c| c.trim().to_string()).collect();

    let after_from = &sql[from_pos + 4..].trim();
    let (table, predicate) = split_at_where(after_from);

    let for_update = upper.contains("FOR UPDATE");
    let for_share = upper.contains("FOR SHARE");

    Some(IrStatement::Select(IrSelect {
        table: table.to_string(),
        columns,
        predicate: parse_predicate_str(&predicate),
        for_update,
        for_share,
    }))
}

fn parse_sql_update(sql: &str) -> Option<IrStatement> {
    // UPDATE table SET col=val WHERE pred
    let upper = sql.to_uppercase();
    let set_pos = upper.find("SET")?;
    let table = sql[7..set_pos].trim().to_string();

    let after_set = &sql[set_pos + 3..];
    let (assignments_str, predicate_str) = split_at_where(after_set);

    let assignments: Vec<(String, IrExpr)> = assignments_str
        .split(',')
        .filter_map(|a| {
            let parts: Vec<&str> = a.splitn(2, '=').collect();
            if parts.len() == 2 {
                let col = parts[0].trim().to_string();
                let val = parts[1].trim();
                Some((col, parse_expr(val)))
            } else {
                None
            }
        })
        .collect();

    Some(IrStatement::Update(IrUpdate {
        table,
        assignments,
        predicate: parse_predicate_str(&predicate_str),
    }))
}

fn parse_sql_insert(sql: &str) -> Option<IrStatement> {
    // INSERT INTO table (col1, col2) VALUES (v1, v2)
    let upper = sql.to_uppercase();
    let into_pos = upper.find("INTO").unwrap_or(6);
    let paren_start = sql.find('(')?;
    let table = sql[into_pos + 4..paren_start].trim().to_string();

    let paren_end = sql.find(')')?;
    let columns: Vec<String> = sql[paren_start + 1..paren_end]
        .split(',')
        .map(|c| c.trim().to_string())
        .collect();

    let values_pos = upper.find("VALUES")?;
    let rest = &sql[values_pos + 6..];
    let vp_start = rest.find('(')?;
    let vp_end = rest.find(')')?;
    let vals: Vec<IrExpr> = rest[vp_start + 1..vp_end]
        .split(',')
        .map(|v| parse_expr(v.trim()))
        .collect();

    Some(IrStatement::Insert(IrInsert {
        table,
        columns,
        values: vec![vals],
    }))
}

fn parse_sql_delete(sql: &str) -> Option<IrStatement> {
    // DELETE FROM table WHERE pred
    let upper = sql.to_uppercase();
    let from_pos = upper.find("FROM")?;
    let after_from = &sql[from_pos + 4..].trim();
    let (table, predicate_str) = split_at_where(after_from);

    Some(IrStatement::Delete(IrDelete {
        table: table.to_string(),
        predicate: parse_predicate_str(&predicate_str),
    }))
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/// Validate that a workload's IR program references only tables present in the schema.
pub fn validate_workload_schema(workload: &Workload) -> Vec<String> {
    let mut errors = Vec::new();
    let table_names = workload.schema.table_names();

    if table_names.is_empty() {
        return errors; // no schema to validate against
    }

    for txn in &workload.program.transactions {
        for stmt in &txn.statements {
            let table = match stmt {
                IrStatement::Select(s) => &s.table,
                IrStatement::Update(u) => &u.table,
                IrStatement::Insert(i) => &i.table,
                IrStatement::Delete(d) => &d.table,
                IrStatement::Lock(l) => &l.table,
            };
            if !table_names.contains(&table.as_str()) {
                errors.push(format!(
                    "Transaction '{}' references unknown table '{}'",
                    txn.label, table
                ));
            }
        }
    }
    errors
}

/// Validate basic structural invariants of a workload.
pub fn validate_workload_structure(workload: &Workload) -> Vec<String> {
    let mut errors = Vec::new();

    if workload.program.transactions.is_empty() {
        errors.push("Workload has no transactions".to_string());
    }

    for txn in &workload.program.transactions {
        if txn.statements.is_empty() {
            errors.push(format!("Transaction '{}' has no statements", txn.label));
        }
        if txn.read_only && txn.has_writes() {
            errors.push(format!(
                "Transaction '{}' marked read_only but contains write statements",
                txn.label
            ));
        }
    }

    errors
}

// ---------------------------------------------------------------------------
// JSON helpers
// ---------------------------------------------------------------------------

fn json_str(obj: &JsonMap<String, JsonValue>, field: &str) -> Option<String> {
    obj.get(field).and_then(|value| match value {
        JsonValue::String(s) => Some(s.clone()),
        JsonValue::Bool(b) => Some(b.to_string()),
        JsonValue::Number(n) => Some(n.to_string()),
        _ => None,
    })
}

fn parse_transaction_value(
    txn_value: &JsonValue,
    txn_alloc: &mut IdAllocator<TransactionId>,
    default_isolation: &str,
) -> Option<IrTransaction> {
    let txn_obj = txn_value.as_object()?;
    let txn_id = txn_alloc.allocate();
    let label = json_str(txn_obj, "label")
        .or_else(|| json_str(txn_obj, "id"))
        .unwrap_or_else(|| format!("T{}", txn_id.as_u64() + 1));
    let mut isolation = json_str(txn_obj, "isolation")
        .map(|iso| parse_isolation_level(&iso))
        .unwrap_or_else(|| parse_isolation_level(default_isolation));

    let steps = txn_obj
        .get("statements")
        .or_else(|| txn_obj.get("operations"))
        .and_then(JsonValue::as_array)?;

    let mut statements = Vec::new();
    for step in steps {
        if let Some(step_obj) = step.as_object() {
            if step_obj
                .get("type")
                .and_then(JsonValue::as_str)
                .map(|kind| kind.eq_ignore_ascii_case("begin"))
                .unwrap_or(false)
            {
                if let Some(begin_iso) = json_str(step_obj, "isolation") {
                    isolation = parse_isolation_level(&begin_iso);
                }
            }
        }
        if let Some(stmt) = parse_statement_value(step) {
            statements.push(stmt);
        }
    }

    let read_only = !statements.is_empty()
        && statements.iter().all(|s| matches!(s, IrStatement::Select(_)));

    Some(IrTransaction {
        id: txn_id,
        label,
        isolation_level: isolation,
        statements,
        read_only,
    })
}

fn parse_statement_value(value: &JsonValue) -> Option<IrStatement> {
    let obj = value.as_object()?;
    if let Some(sql) = json_str(obj, "sql") {
        return parse_sql_statement(&sql);
    }

    let stmt_type = json_str(obj, "type")?.to_ascii_lowercase();
    let table = json_str(obj, "table").unwrap_or_default();
    let predicate = json_str(obj, "predicate").unwrap_or_default();

    match stmt_type.as_str() {
        "begin" | "commit" | "abort" => None,
        "select" | "read" => {
            let columns = obj
                .get("columns")
                .and_then(JsonValue::as_array)
                .map(|items| {
                    items.iter()
                        .filter_map(JsonValue::as_str)
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                })
                .filter(|cols| !cols.is_empty())
                .unwrap_or_else(|| vec!["*".to_string()]);
            Some(IrStatement::Select(IrSelect {
                table,
                columns,
                predicate: parse_predicate_str(&predicate),
                for_update: false,
                for_share: false,
            }))
        }
        "update" | "write" => Some(IrStatement::Update(IrUpdate {
            table,
            assignments: obj
                .get("set")
                .and_then(JsonValue::as_object)
                .map(parse_set_object)
                .unwrap_or_default(),
            predicate: parse_predicate_str(&predicate),
        })),
        "insert" => Some(IrStatement::Insert(IrInsert {
            table,
            columns: obj
                .get("columns")
                .and_then(JsonValue::as_array)
                .map(|items| {
                    items.iter()
                        .filter_map(JsonValue::as_str)
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default(),
            values: obj
                .get("values")
                .and_then(JsonValue::as_array)
                .map(|values| parse_values_array(values))
                .unwrap_or_default(),
        })),
        "delete" => Some(IrStatement::Delete(IrDelete {
            table,
            predicate: parse_predicate_str(&predicate),
        })),
        _ => None,
    }
}

fn parse_set_object(set_obj: &JsonMap<String, JsonValue>) -> Vec<(String, IrExpr)> {
    set_obj
        .iter()
        .map(|(column, value)| (column.clone(), parse_json_expr(value)))
        .collect()
}

fn parse_values_array(values: &[JsonValue]) -> Vec<Vec<IrExpr>> {
    values
        .iter()
        .map(|row| {
            row.as_array()
                .map(|items| items.iter().map(parse_json_expr).collect())
                .unwrap_or_default()
        })
        .collect()
}

fn parse_json_expr(value: &JsonValue) -> IrExpr {
    match value {
        JsonValue::Null => IrExpr::Null,
        JsonValue::Bool(b) => IrExpr::Literal(Value::Boolean(*b)),
        JsonValue::Number(n) => {
            if let Some(i) = n.as_i64() {
                IrExpr::Literal(Value::Integer(i))
            } else if let Some(f) = n.as_f64() {
                IrExpr::Literal(Value::Float(f))
            } else {
                IrExpr::Literal(Value::Text(n.to_string()))
            }
        }
        JsonValue::String(s) => parse_expr(s),
        JsonValue::Array(items) => IrExpr::Literal(Value::Array(
            items
                .iter()
                .map(|item| match parse_json_expr(item) {
                    IrExpr::Literal(v) => v,
                    IrExpr::ColumnRef(name) => Value::Text(name),
                    other => Value::Text(format!("{other:?}")),
                })
                .collect(),
        )),
        JsonValue::Object(_) => IrExpr::Literal(Value::Text(value.to_string())),
    }
}

fn collect_workload_annotations(root: &JsonMap<String, JsonValue>) -> HashMap<String, String> {
    let mut annotations = HashMap::new();

    if let Some(anomaly) = json_str(root, "anomaly_class") {
        annotations.insert("declared_anomaly_class".to_string(), anomaly);
        annotations.insert("declared_anomaly_detected".to_string(), "true".to_string());
    }

    if let Some(expected_outcome) = root.get("expected_outcome").and_then(JsonValue::as_object) {
        if let Some(anomaly_detected) = expected_outcome.get("anomaly_detected").and_then(JsonValue::as_bool) {
            annotations.insert("declared_anomaly_detected".to_string(), anomaly_detected.to_string());
        }
        if let Some(anomaly_type) = json_str(expected_outcome, "anomaly_type") {
            annotations.insert("declared_anomaly_description".to_string(), anomaly_type);
        }
        if let Some(ssi_behavior) = json_str(expected_outcome, "ssi_behavior") {
            annotations.insert("declared_anomaly_detail".to_string(), ssi_behavior);
        }
    }

    if let Some(portability) = root.get("portability_analysis").and_then(JsonValue::as_object) {
        if let Some(verdict) = portability.get("isospec_verdict").and_then(JsonValue::as_object) {
            if let Some(portable) = verdict.get("portable").and_then(JsonValue::as_bool) {
                annotations.insert("declared_portable".to_string(), portable.to_string());
            }
            if let Some(violation_type) = json_str(verdict, "violation_type") {
                annotations.insert("declared_portability_violation".to_string(), violation_type);
            }
            if let Some(recommendation) = json_str(verdict, "recommendation") {
                annotations.insert("declared_portability_recommendation".to_string(), recommendation);
            }
        }
    }

    annotations
}

// ---------------------------------------------------------------------------
// Parsing helpers
// ---------------------------------------------------------------------------

fn parse_isolation_level(s: &str) -> IsolationLevel {
    IsolationLevel::from_str_loose(s).unwrap_or(IsolationLevel::Serializable)
}

fn split_at_where(s: &str) -> (String, String) {
    let upper = s.to_uppercase();
    if let Some(pos) = upper.find("WHERE") {
        let table = s[..pos].trim().to_string();
        let pred = s[pos + 5..].trim().to_string();
        (table, pred)
    } else {
        (s.trim().to_string(), String::new())
    }
}

fn parse_predicate_str(s: &str) -> Predicate {
    let s = s.trim();
    if s.is_empty() {
        return Predicate::True;
    }

    let upper = s.to_ascii_uppercase();
    if let Some(in_pos) = upper.find(" IN ") {
        let column = s[..in_pos].trim();
        let rest = s[in_pos + 4..].trim();
        if rest.starts_with('(') && rest.ends_with(')') {
            let values = rest[1..rest.len() - 1]
                .split(',')
                .map(|item| parse_scalar_value(item.trim()))
                .collect::<Vec<_>>();
            return Predicate::In(column.into(), values);
        }
    }

    // Simple "col = val" parser
    if let Some(eq_pos) = s.find('=') {
        if !s[..eq_pos].ends_with('!') && !s[..eq_pos].ends_with('<') && !s[..eq_pos].ends_with('>') {
            let col = s[..eq_pos].trim();
            return Predicate::eq(col, parse_scalar_value(&s[eq_pos + 1..]));
        }
    }

    // Simple "col < val"
    if let Some(lt_pos) = s.find('<') {
        if !s.as_bytes().get(lt_pos + 1).copied().map_or(false, |b| b == b'=') {
            let col = s[..lt_pos].trim();
            return Predicate::lt(col, parse_scalar_value(&s[lt_pos + 1..]));
        }
    }

    // Simple "col > val"
    if let Some(gt_pos) = s.find('>') {
        if !s.as_bytes().get(gt_pos + 1).copied().map_or(false, |b| b == b'=') {
            let col = s[..gt_pos].trim();
            return Predicate::gt(col, parse_scalar_value(&s[gt_pos + 1..]));
        }
    }

    Predicate::True
}

fn parse_expr(s: &str) -> IrExpr {
    let s = s.trim().trim_matches('\'').trim_matches('"');
    for op in [" + ", " - "] {
        if let Some(pos) = s.find(op) {
            let left = parse_expr(&s[..pos]);
            let right = parse_expr(&s[pos + op.len()..]);
            return IrExpr::BinaryOp {
                left: Box::new(left),
                op: op.trim().to_string(),
                right: Box::new(right),
            };
        }
    }
    if let Ok(i) = s.parse::<i64>() {
        IrExpr::Literal(Value::Integer(i))
    } else if let Ok(f) = s.parse::<f64>() {
        IrExpr::Literal(Value::Float(f))
    } else if s.eq_ignore_ascii_case("true") || s.eq_ignore_ascii_case("false") {
        IrExpr::Literal(Value::Boolean(s.eq_ignore_ascii_case("true")))
    } else if s.eq_ignore_ascii_case("null") {
        IrExpr::Null
    } else if s.chars().all(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == '.') {
        IrExpr::ColumnRef(s.to_string())
    } else {
        IrExpr::Literal(Value::Text(s.to_string()))
    }
}

fn parse_scalar_value(s: &str) -> Value {
    let s = s.trim().trim_matches('\'').trim_matches('"');
    if let Ok(i) = s.parse::<i64>() {
        Value::Integer(i)
    } else if let Ok(f) = s.parse::<f64>() {
        Value::Float(f)
    } else if s.eq_ignore_ascii_case("true") || s.eq_ignore_ascii_case("false") {
        Value::Boolean(s.eq_ignore_ascii_case("true"))
    } else if s.eq_ignore_ascii_case("null") {
        Value::Null
    } else {
        Value::Text(s.to_string())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_workload() {
        let json = r#"{
            "name": "test_wl",
            "transactions": [
                {
                    "label": "T1",
                    "isolation": "serializable",
                    "statements": [
                        {"type": "select", "table": "t", "columns": ["id", "val"], "predicate": "id = 1"}
                    ]
                }
            ]
        }"#;
        let wl = parse_workload_json(json).unwrap();
        assert_eq!(wl.name, "test_wl");
        assert_eq!(wl.program.transaction_count(), 1);
    }

    #[test]
    fn test_parse_example_workload_operations() {
        let json = include_str!("../../../../examples/pg_serializable_write_skew.json");
        let wl = parse_workload_json(json).unwrap();
        assert_eq!(wl.program.transaction_count(), 2);
        assert_eq!(wl.program.total_statements(), 4);
        assert_eq!(wl.program.tables_accessed(), vec!["doctors"]);
        assert_eq!(
            wl.annotations.get("declared_anomaly_class").map(String::as_str),
            Some("G2-item"),
        );
    }

    #[test]
    fn test_parse_nested_workload_transactions() {
        let json = include_str!("../../../../examples/cross_engine_pg_to_mysql.json");
        let wl = parse_workload_json(json).unwrap();
        assert_eq!(wl.program.transaction_count(), 2);
        assert_eq!(wl.program.total_statements(), 6);
        assert_eq!(
            wl.annotations.get("declared_portability_violation").map(String::as_str),
            Some("G2-item (write skew)"),
        );
    }

    #[test]
    fn test_parse_sql_select() {
        let stmt = parse_sql_statement("SELECT id, val FROM items WHERE id = 42").unwrap();
        match stmt {
            IrStatement::Select(s) => {
                assert_eq!(s.table, "items");
                assert_eq!(s.columns, vec!["id", "val"]);
            }
            _ => panic!("Expected Select"),
        }
    }

    #[test]
    fn test_parse_sql_update() {
        let stmt = parse_sql_statement("UPDATE items SET val = 99 WHERE id = 1").unwrap();
        match stmt {
            IrStatement::Update(u) => {
                assert_eq!(u.table, "items");
                assert_eq!(u.assignments.len(), 1);
            }
            _ => panic!("Expected Update"),
        }
    }

    #[test]
    fn test_parse_sql_insert() {
        let stmt = parse_sql_statement("INSERT INTO items (id, val) VALUES (1, 99)").unwrap();
        match stmt {
            IrStatement::Insert(i) => {
                assert_eq!(i.table, "items");
                assert_eq!(i.columns.len(), 2);
                assert_eq!(i.values.len(), 1);
            }
            _ => panic!("Expected Insert"),
        }
    }

    #[test]
    fn test_parse_sql_delete() {
        let stmt = parse_sql_statement("DELETE FROM items WHERE id = 5").unwrap();
        match stmt {
            IrStatement::Delete(d) => {
                assert_eq!(d.table, "items");
            }
            _ => panic!("Expected Delete"),
        }
    }

    #[test]
    fn test_parse_predicate_eq() {
        let pred = parse_predicate_str("id = 42");
        match pred {
            Predicate::Comparison(cp) => {
                assert_eq!(cp.column.column, "id");
            }
            _ => panic!("Expected Comparison"),
        }
    }

    #[test]
    fn test_parse_predicate_empty() {
        let pred = parse_predicate_str("");
        assert!(matches!(pred, Predicate::True));
    }

    #[test]
    fn test_validate_structure_empty() {
        let wl = Workload {
            id: WorkloadId::new(1),
            name: "empty".to_string(),
            program: IrProgram {
                id: WorkloadId::new(1),
                name: "empty".to_string(),
                transactions: Vec::new(),
                schema_name: "test".to_string(),
                metadata: HashMap::new(),
            },
            schema: Schema::new(),
            parameters: WorkloadParameters {
                transaction_bound: 0,
                operation_bound: 0,
                data_item_bound: 0,
                repetitions: 1,
            },
            annotations: HashMap::new(),
        };
        let errors = validate_workload_structure(&wl);
        assert!(!errors.is_empty());
        assert!(errors[0].contains("no transactions"));
    }

    #[test]
    fn test_parse_isolation_level() {
        assert_eq!(parse_isolation_level("serializable"), IsolationLevel::Serializable);
        assert_eq!(parse_isolation_level("read-committed"), IsolationLevel::ReadCommitted);
    }

    #[test]
    fn test_split_at_where() {
        let (table, pred) = split_at_where("items WHERE id = 1");
        assert_eq!(table, "items");
        assert_eq!(pred, "id = 1");
    }

    #[test]
    fn test_split_at_where_no_where() {
        let (table, pred) = split_at_where("items");
        assert_eq!(table, "items");
        assert!(pred.is_empty());
    }
}
