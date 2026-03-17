//! SQL result parser.
//!
//! Parses query results, EXPLAIN output, and engine-specific diagnostic output
//! into structured representations.

use std::collections::HashMap;
use std::time::Duration;

use isospec_types::config::EngineKind;
use isospec_types::error::{IsoSpecError, IsoSpecResult};
use isospec_types::value::Value;

// ---------------------------------------------------------------------------
// Value parsing
// ---------------------------------------------------------------------------

/// Parse a string cell value into a typed `Value`.
pub fn parse_cell_value(raw: &str, type_hint: Option<&str>) -> Value {
    let trimmed = raw.trim();

    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("null") {
        return Value::Null;
    }

    // Try type hint first
    if let Some(hint) = type_hint {
        let hint_lower = hint.to_lowercase();
        if hint_lower.contains("int") || hint_lower.contains("serial") {
            if let Ok(i) = trimmed.parse::<i64>() {
                return Value::Integer(i);
            }
        }
        if hint_lower.contains("bool") || hint_lower.contains("bit") {
            return match trimmed {
                "t" | "true" | "TRUE" | "1" => Value::Boolean(true),
                "f" | "false" | "FALSE" | "0" => Value::Boolean(false),
                _ => Value::Text(trimmed.to_string()),
            };
        }
        if hint_lower.contains("float")
            || hint_lower.contains("double")
            || hint_lower.contains("real")
            || hint_lower.contains("numeric")
            || hint_lower.contains("decimal")
        {
            if let Ok(f) = trimmed.parse::<f64>() {
                return Value::Float(f);
            }
        }
    }

    // Auto-detect type
    if let Ok(i) = trimmed.parse::<i64>() {
        return Value::Integer(i);
    }
    if let Ok(f) = trimmed.parse::<f64>() {
        if trimmed.contains('.') {
            return Value::Float(f);
        }
    }
    if trimmed == "t" || trimmed == "true" || trimmed == "TRUE" {
        return Value::Boolean(true);
    }
    if trimmed == "f" || trimmed == "false" || trimmed == "FALSE" {
        return Value::Boolean(false);
    }

    Value::Text(trimmed.to_string())
}

/// Parse a delimited result set (e.g., from psql or mysql CLI output).
pub fn parse_delimited_result(
    output: &str,
    delimiter: char,
    has_header: bool,
) -> ParsedResultSet {
    let lines: Vec<&str> = output
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect();

    if lines.is_empty() {
        return ParsedResultSet {
            columns: Vec::new(),
            rows: Vec::new(),
        };
    }

    let start_idx;
    let columns;

    if has_header {
        let header = lines[0];
        columns = header
            .split(delimiter)
            .map(|c| c.trim().to_string())
            .collect::<Vec<_>>();
        // Skip separator line if present (e.g., +----+----+)
        start_idx = if lines.len() > 1 && lines[1].contains("---") {
            2
        } else {
            1
        };
    } else {
        columns = Vec::new();
        start_idx = 0;
    }

    let mut rows = Vec::new();
    for line in &lines[start_idx..] {
        // Skip separator lines
        if line.starts_with('+') || line.starts_with('-') {
            continue;
        }
        // Skip footer lines like "(N rows)"
        if line.starts_with('(') && line.ends_with(')') {
            continue;
        }
        let cells: Vec<Value> = line
            .split(delimiter)
            .map(|c| parse_cell_value(c, None))
            .collect();
        if !cells.is_empty() {
            rows.push(cells);
        }
    }

    ParsedResultSet { columns, rows }
}

/// A parsed result set with columns and rows.
#[derive(Debug, Clone)]
pub struct ParsedResultSet {
    pub columns: Vec<String>,
    pub rows: Vec<Vec<Value>>,
}

impl ParsedResultSet {
    pub fn row_count(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    pub fn get(&self, row: usize, col: usize) -> Option<&Value> {
        self.rows.get(row).and_then(|r| r.get(col))
    }
}

// ---------------------------------------------------------------------------
// EXPLAIN output parsing
// ---------------------------------------------------------------------------

/// Parsed EXPLAIN plan node.
#[derive(Debug, Clone)]
pub struct ExplainNode {
    pub node_type: String,
    pub relation: Option<String>,
    pub startup_cost: Option<f64>,
    pub total_cost: Option<f64>,
    pub rows: Option<u64>,
    pub width: Option<u64>,
    pub actual_time: Option<f64>,
    pub actual_rows: Option<u64>,
    pub children: Vec<ExplainNode>,
    pub extra: HashMap<String, String>,
}

impl ExplainNode {
    pub fn new(node_type: &str) -> Self {
        Self {
            node_type: node_type.to_string(),
            relation: None,
            startup_cost: None,
            total_cost: None,
            rows: None,
            width: None,
            actual_time: None,
            actual_rows: None,
            children: Vec::new(),
            extra: HashMap::new(),
        }
    }

    pub fn is_seq_scan(&self) -> bool {
        self.node_type.contains("Seq Scan") || self.node_type.contains("ALL")
    }

    pub fn is_index_scan(&self) -> bool {
        self.node_type.contains("Index") || self.node_type.contains("ref")
    }

    pub fn total_estimated_rows(&self) -> u64 {
        let own = self.rows.unwrap_or(0);
        let child_sum: u64 = self.children.iter().map(|c| c.total_estimated_rows()).sum();
        own + child_sum
    }
}

/// Parse PostgreSQL text-format EXPLAIN output.
pub fn parse_pg_explain(output: &str) -> Vec<ExplainNode> {
    let mut nodes = Vec::new();
    let lines: Vec<&str> = output.lines().collect();

    for line in &lines {
        let trimmed = line.trim_start_matches(|c: char| c == ' ' || c == '-' || c == '>');
        if trimmed.is_empty() {
            continue;
        }

        // Extract node type (text before parentheses)
        if let Some(paren_pos) = trimmed.find('(') {
            let node_type = trimmed[..paren_pos].trim().to_string();
            let mut node = ExplainNode::new(&node_type);

            // Parse cost info: (cost=X..Y rows=N width=W)
            let cost_section = &trimmed[paren_pos..];
            if let Some(cost_str) = extract_between(cost_section, "cost=", "..") {
                node.startup_cost = cost_str.parse().ok();
            }
            if let Some(total_str) =
                extract_between(cost_section, "..", " rows=").or_else(|| extract_between(cost_section, "..", ")"))
            {
                node.total_cost = total_str.parse().ok();
            }
            if let Some(rows_str) = extract_between(cost_section, "rows=", " ") {
                node.rows = rows_str.parse().ok();
            }
            if let Some(width_str) = extract_between(cost_section, "width=", ")") {
                node.width = width_str.parse().ok();
            }

            // Check for table name
            if node_type.contains(" on ") {
                let parts: Vec<&str> = node_type.splitn(2, " on ").collect();
                if parts.len() == 2 {
                    node.relation = Some(parts[1].trim().to_string());
                }
            }

            nodes.push(node);
        } else if trimmed.starts_with("Planning") || trimmed.starts_with("Execution") {
            // Skip planning/execution time lines
        } else {
            // Extra info line, attach to last node
            if let Some(last) = nodes.last_mut() {
                if let Some((key, val)) = trimmed.split_once(':') {
                    last.extra
                        .insert(key.trim().to_string(), val.trim().to_string());
                }
            }
        }
    }

    nodes
}

/// Parse MySQL EXPLAIN output (tabular format).
pub fn parse_mysql_explain(output: &str) -> Vec<ExplainNode> {
    let result = parse_delimited_result(output, '|', true);
    let mut nodes = Vec::new();

    let type_col = result.columns.iter().position(|c| c == "type");
    let table_col = result.columns.iter().position(|c| c == "table");
    let rows_col = result.columns.iter().position(|c| c == "rows");
    let extra_col = result.columns.iter().position(|c| c.starts_with("Extra"));

    for row in &result.rows {
        let node_type = type_col
            .and_then(|i| row.get(i))
            .map(|v| match v {
                Value::Text(s) => s.clone(),
                _ => format!("{:?}", v),
            })
            .unwrap_or_else(|| "unknown".to_string());

        let mut node = ExplainNode::new(&node_type);

        node.relation = table_col.and_then(|i| row.get(i)).and_then(|v| match v {
            Value::Text(s) => Some(s.clone()),
            _ => None,
        });

        node.rows = rows_col.and_then(|i| row.get(i)).and_then(|v| match v {
            Value::Integer(i) => Some(*i as u64),
            Value::Text(s) => s.parse().ok(),
            _ => None,
        });

        if let Some(idx) = extra_col {
            if let Some(Value::Text(s)) = row.get(idx) {
                node.extra.insert("Extra".to_string(), s.clone());
            }
        }

        nodes.push(node);
    }

    nodes
}

/// Extract the substring between two markers.
fn extract_between<'a>(s: &'a str, start: &str, end: &str) -> Option<&'a str> {
    let start_pos = s.find(start)?;
    let after_start = start_pos + start.len();
    let remaining = &s[after_start..];
    let end_pos = remaining.find(end)?;
    Some(&remaining[..end_pos])
}

// ---------------------------------------------------------------------------
// Engine-specific diagnostic parsing
// ---------------------------------------------------------------------------

/// Parse lock wait information from engine-specific output.
pub fn parse_lock_waits(output: &str, engine: &EngineKind) -> Vec<LockWaitInfo> {
    match engine {
        EngineKind::PostgreSQL => parse_pg_lock_waits(output),
        EngineKind::MySQL => parse_mysql_lock_waits(output),
        EngineKind::SqlServer => parse_mssql_lock_waits(output),
    }
}

/// Lock wait information.
#[derive(Debug, Clone)]
pub struct LockWaitInfo {
    pub waiting_pid: String,
    pub blocking_pid: String,
    pub lock_type: String,
    pub relation: Option<String>,
    pub duration: Option<Duration>,
}

fn parse_pg_lock_waits(output: &str) -> Vec<LockWaitInfo> {
    let mut waits = Vec::new();
    let result = parse_delimited_result(output, '|', true);

    for row in &result.rows {
        if row.len() >= 3 {
            let waiting = match &row[0] {
                Value::Integer(i) => i.to_string(),
                Value::Text(s) => s.clone(),
                _ => continue,
            };
            let blocking = match &row[1] {
                Value::Integer(i) => i.to_string(),
                Value::Text(s) => s.clone(),
                _ => continue,
            };
            let lock_type = match &row[2] {
                Value::Text(s) => s.clone(),
                _ => "unknown".to_string(),
            };
            waits.push(LockWaitInfo {
                waiting_pid: waiting,
                blocking_pid: blocking,
                lock_type,
                relation: row.get(3).and_then(|v| match v {
                    Value::Text(s) => Some(s.clone()),
                    _ => None,
                }),
                duration: None,
            });
        }
    }
    waits
}

fn parse_mysql_lock_waits(output: &str) -> Vec<LockWaitInfo> {
    let mut waits = Vec::new();
    // Parse InnoDB status output looking for LOCK WAIT sections
    let mut in_lock_section = false;
    let mut current_waiting = String::new();
    let mut current_blocking = String::new();

    for line in output.lines() {
        let trimmed = line.trim();
        if trimmed.contains("LOCK WAIT") {
            in_lock_section = true;
            current_waiting.clear();
            current_blocking.clear();
        }
        if in_lock_section {
            if trimmed.starts_with("---TRANSACTION") {
                if let Some(id) = trimmed.split_whitespace().nth(1) {
                    if current_waiting.is_empty() {
                        current_waiting = id.to_string();
                    }
                }
            }
            if trimmed.contains("HOLDS THE LOCK") || trimmed.contains("blocking") {
                if let Some(id) = trimmed.split_whitespace().last() {
                    current_blocking = id.to_string();
                }
            }
            if !current_waiting.is_empty() && !current_blocking.is_empty() {
                waits.push(LockWaitInfo {
                    waiting_pid: current_waiting.clone(),
                    blocking_pid: current_blocking.clone(),
                    lock_type: "row".to_string(),
                    relation: None,
                    duration: None,
                });
                in_lock_section = false;
            }
        }
    }
    waits
}

fn parse_mssql_lock_waits(output: &str) -> Vec<LockWaitInfo> {
    let result = parse_delimited_result(output, '|', true);
    let mut waits = Vec::new();

    for row in &result.rows {
        if row.len() >= 3 {
            waits.push(LockWaitInfo {
                waiting_pid: format!("{:?}", row[0]),
                blocking_pid: format!("{:?}", row[1]),
                lock_type: match &row[2] {
                    Value::Text(s) => s.clone(),
                    _ => "unknown".to_string(),
                },
                relation: None,
                duration: None,
            });
        }
    }
    waits
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_cell_value_int() {
        assert_eq!(parse_cell_value("42", None), Value::Integer(42));
        assert_eq!(parse_cell_value("-7", None), Value::Integer(-7));
    }

    #[test]
    fn test_parse_cell_value_float() {
        assert_eq!(parse_cell_value("3.14", None), Value::Float(3.14));
    }

    #[test]
    fn test_parse_cell_value_bool() {
        assert_eq!(parse_cell_value("true", None), Value::Boolean(true));
        assert_eq!(parse_cell_value("false", None), Value::Boolean(false));
        assert_eq!(parse_cell_value("t", None), Value::Boolean(true));
    }

    #[test]
    fn test_parse_cell_value_null() {
        assert_eq!(parse_cell_value("NULL", None), Value::Null);
        assert_eq!(parse_cell_value("", None), Value::Null);
    }

    #[test]
    fn test_parse_cell_value_with_hint() {
        assert_eq!(
            parse_cell_value("1", Some("BOOLEAN")),
            Value::Boolean(true)
        );
        assert_eq!(
            parse_cell_value("42", Some("INT")),
            Value::Integer(42)
        );
        assert_eq!(
            parse_cell_value("3.14", Some("FLOAT")),
            Value::Float(3.14)
        );
    }

    #[test]
    fn test_parse_cell_value_text() {
        assert_eq!(
            parse_cell_value("hello", None),
            Value::Text("hello".to_string())
        );
    }

    #[test]
    fn test_parse_delimited_result() {
        let output = "id|val\n---+---\n1|42\n2|99\n(2 rows)";
        let result = parse_delimited_result(output, '|', true);
        assert_eq!(result.columns, vec!["id", "val"]);
        assert_eq!(result.row_count(), 2);
        assert_eq!(result.get(0, 0), Some(&Value::Integer(1)));
        assert_eq!(result.get(0, 1), Some(&Value::Integer(42)));
    }

    #[test]
    fn test_parse_delimited_no_header() {
        let output = "1|hello\n2|world";
        let result = parse_delimited_result(output, '|', false);
        assert!(result.columns.is_empty());
        assert_eq!(result.row_count(), 2);
    }

    #[test]
    fn test_parse_delimited_empty() {
        let result = parse_delimited_result("", '|', true);
        assert!(result.is_empty());
    }

    #[test]
    fn test_parse_pg_explain() {
        let explain_output = r#"
Seq Scan on test_data  (cost=0.00..35.50 rows=10 width=8)
  Filter: (val > 0)
"#;
        let nodes = parse_pg_explain(explain_output);
        assert!(!nodes.is_empty());
        let first = &nodes[0];
        assert!(first.is_seq_scan());
    }

    #[test]
    fn test_explain_node() {
        let mut node = ExplainNode::new("Index Scan on users");
        node.rows = Some(100);
        node.relation = Some("users".to_string());
        assert!(node.is_index_scan());
        assert!(!node.is_seq_scan());
        assert_eq!(node.total_estimated_rows(), 100);
    }

    #[test]
    fn test_explain_node_with_children() {
        let mut parent = ExplainNode::new("Hash Join");
        parent.rows = Some(50);
        let mut child = ExplainNode::new("Seq Scan");
        child.rows = Some(100);
        parent.children.push(child);
        assert_eq!(parent.total_estimated_rows(), 150);
    }

    #[test]
    fn test_extract_between() {
        assert_eq!(extract_between("cost=1.00..5.50", "cost=", ".."), Some("1.00"));
        assert_eq!(extract_between("rows=100 width", "rows=", " "), Some("100"));
        assert_eq!(extract_between("abc", "x", "y"), None);
    }

    #[test]
    fn test_lock_wait_info() {
        let info = LockWaitInfo {
            waiting_pid: "1234".into(),
            blocking_pid: "5678".into(),
            lock_type: "RowExclusive".into(),
            relation: Some("test_data".into()),
            duration: None,
        };
        assert_eq!(info.waiting_pid, "1234");
        assert_eq!(info.lock_type, "RowExclusive");
    }

    #[test]
    fn test_parse_pg_lock_waits() {
        let output = "waiting|blocking|lock_type|relation\n---+---+---+---\n100|200|RowExclusive|test_data";
        let waits = parse_pg_lock_waits(output);
        assert_eq!(waits.len(), 1);
        assert_eq!(waits[0].waiting_pid, "100");
        assert_eq!(waits[0].blocking_pid, "200");
    }

    #[test]
    fn test_parsed_result_set() {
        let rs = ParsedResultSet {
            columns: vec!["a".into(), "b".into()],
            rows: vec![vec![Value::Integer(1), Value::Integer(2)]],
        };
        assert_eq!(rs.row_count(), 1);
        assert!(!rs.is_empty());
        assert_eq!(rs.get(0, 0), Some(&Value::Integer(1)));
    }
}
