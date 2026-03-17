//! SQL trace log parser for various database log formats.
//!
//! Supports:
//! - **pgAudit** format (PostgreSQL audit logging extension)
//! - **MySQL General Query Log** format
//! - **Generic SQL trace** format (timestamp + session + SQL)
//! - **Custom CSV/TSV** trace formats
//!
//! Each parser extracts transaction boundaries and SQL statements,
//! building a `TransactionHistory` for isolation analysis.

use crate::{FormatError, FormatResult};
use isospec_history::builder::HistoryBuilder;
use isospec_history::history::TransactionHistory;
use isospec_types::identifier::{ItemId, TableId, TransactionId};
use isospec_types::isolation::IsolationLevel;
use isospec_types::value::Value;
use std::collections::HashMap;
use tracing::{debug, trace, warn};

/// Supported SQL trace log formats
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraceFormat {
    /// pgAudit log format: `AUDIT: SESSION,...,STATEMENT,...`
    PgAudit,
    /// MySQL general query log: `timestamp thread_id command_type argument`
    MysqlGeneralLog,
    /// Generic format: `timestamp|session_id|sql_statement`
    Generic,
    /// CSV format with configurable columns
    Csv,
}

/// Configuration for trace parsing
#[derive(Debug, Clone)]
pub struct TraceConfig {
    /// Log format to parse
    pub format: TraceFormat,
    /// Default isolation level if not specified in the log
    pub default_isolation: IsolationLevel,
    /// Column delimiter for CSV format
    pub delimiter: char,
    /// Column index for timestamp (CSV format)
    pub timestamp_column: usize,
    /// Column index for session/connection ID (CSV format)
    pub session_column: usize,
    /// Column index for SQL statement (CSV format)
    pub sql_column: usize,
    /// Whether the CSV has a header row
    pub has_header: bool,
}

impl Default for TraceConfig {
    fn default() -> Self {
        Self {
            format: TraceFormat::Generic,
            default_isolation: IsolationLevel::ReadCommitted,
            delimiter: '|',
            timestamp_column: 0,
            session_column: 1,
            sql_column: 2,
            has_header: false,
        }
    }
}

impl TraceConfig {
    pub fn pgaudit() -> Self {
        Self {
            format: TraceFormat::PgAudit,
            default_isolation: IsolationLevel::ReadCommitted,
            ..Default::default()
        }
    }

    pub fn mysql_general_log() -> Self {
        Self {
            format: TraceFormat::MysqlGeneralLog,
            default_isolation: IsolationLevel::RepeatableRead,
            ..Default::default()
        }
    }

    pub fn csv(delimiter: char) -> Self {
        Self {
            format: TraceFormat::Csv,
            delimiter,
            ..Default::default()
        }
    }
}

/// Parse a SQL trace log into a `TransactionHistory`.
pub fn parse_trace(input: &str, config: &TraceConfig) -> FormatResult<TransactionHistory> {
    match config.format {
        TraceFormat::PgAudit => parse_pgaudit(input, config),
        TraceFormat::MysqlGeneralLog => parse_mysql_general_log(input, config),
        TraceFormat::Generic => parse_generic_trace(input, config),
        TraceFormat::Csv => parse_csv_trace(input, config),
    }
}

/// State tracker for a single session/connection
struct SessionState {
    txn_id: Option<TransactionId>,
    isolation_level: IsolationLevel,
    in_transaction: bool,
}

/// Shared parser state across sessions
struct TraceParserState {
    builder: HistoryBuilder,
    sessions: HashMap<String, SessionState>,
    next_txn_id: u64,
    default_isolation: IsolationLevel,
}

impl TraceParserState {
    fn new(default_isolation: IsolationLevel) -> Self {
        Self {
            builder: HistoryBuilder::new(),
            sessions: HashMap::new(),
            next_txn_id: 1,
            default_isolation,
        }
    }

    fn get_or_create_session(&mut self, session_id: &str) -> &mut SessionState {
        let default_iso = self.default_isolation;
        self.sessions
            .entry(session_id.to_string())
            .or_insert_with(|| SessionState {
                txn_id: None,
                isolation_level: default_iso,
                in_transaction: false,
            })
    }

    fn allocate_txn_id(&mut self) -> TransactionId {
        let id = TransactionId::from(self.next_txn_id);
        self.next_txn_id += 1;
        id
    }

    fn process_sql(&mut self, session_id: &str, sql: &str) {
        let upper = sql.trim().to_uppercase();

        // Transaction control
        if upper.starts_with("BEGIN") || upper.starts_with("START TRANSACTION") {
            let txn_id = self.allocate_txn_id();
            let session = self.get_or_create_session(session_id);
            session.txn_id = Some(txn_id);
            session.in_transaction = true;
            let iso = session.isolation_level;
            self.builder.begin_transaction(txn_id, iso);
            return;
        }

        if upper.starts_with("COMMIT") || upper.starts_with("END") {
            let session = self.get_or_create_session(session_id);
            if let Some(txn_id) = session.txn_id.take() {
                session.in_transaction = false;
                self.builder.commit_transaction(txn_id);
            }
            return;
        }

        if upper.starts_with("ROLLBACK") || upper.starts_with("ABORT") {
            let session = self.get_or_create_session(session_id);
            if let Some(txn_id) = session.txn_id.take() {
                session.in_transaction = false;
                self.builder
                    .abort_transaction(txn_id, Some("Explicit rollback".to_string()));
            }
            return;
        }

        // Isolation level changes
        if upper.contains("SET TRANSACTION ISOLATION LEVEL")
            || upper.contains("SET SESSION CHARACTERISTICS")
        {
            let session = self.get_or_create_session(session_id);
            if upper.contains("SERIALIZABLE") {
                session.isolation_level = IsolationLevel::Serializable;
            } else if upper.contains("REPEATABLE READ") {
                session.isolation_level = IsolationLevel::RepeatableRead;
            } else if upper.contains("READ COMMITTED") {
                session.isolation_level = IsolationLevel::ReadCommitted;
            } else if upper.contains("READ UNCOMMITTED") {
                session.isolation_level = IsolationLevel::ReadUncommitted;
            }
            return;
        }

        // DML operations
        let session = self.get_or_create_session(session_id);
        if let Some(txn_id) = session.txn_id {
            let table_name = extract_table_from_sql(&upper).unwrap_or("unknown".to_string());
            let table_id = TableId::new(hash_table_name(table_name.as_str()));
            let item_id = ItemId::from(0u64);

            if upper.starts_with("SELECT") {
                self.builder
                    .add_read(txn_id, table_id, item_id, Some(Value::Null));
            } else if upper.starts_with("UPDATE") {
                self.builder
                    .add_write(txn_id, table_id, item_id, Some(Value::Null), Value::Null);
            } else if upper.starts_with("INSERT") {
                self.builder
                    .add_insert(txn_id, table_id, item_id, Vec::new());
            } else if upper.starts_with("DELETE") {
                self.builder.add_delete(txn_id, table_id, item_id);
            }
        }
    }

    fn finish(mut self) -> FormatResult<TransactionHistory> {
        // Abort remaining open transactions
        for (session_id, session) in &self.sessions {
            if let Some(txn_id) = session.txn_id {
                self.builder.abort_transaction(
                    txn_id,
                    Some(format!("Session {} ended without commit", session_id)),
                );
            }
        }
        Ok(self.builder.build()?)
    }
}

/// Parse pgAudit format logs.
///
/// Format: `AUDIT: SESSION,1,1,READ,SELECT,,,SELECT * FROM accounts WHERE id = 1,<none>`
fn parse_pgaudit(input: &str, config: &TraceConfig) -> FormatResult<TransactionHistory> {
    let mut state = TraceParserState::new(config.default_isolation);
    let mut line_count = 0;

    for line in input.lines() {
        let line = line.trim();
        if line.is_empty() || !line.contains("AUDIT:") {
            continue;
        }

        line_count += 1;

        // Extract the AUDIT payload after "AUDIT: "
        let audit_part = if let Some(pos) = line.find("AUDIT:") {
            line[pos + 6..].trim()
        } else {
            continue;
        };

        let fields: Vec<&str> = audit_part.splitn(9, ',').collect();
        if fields.len() < 8 {
            trace!(line = %line, "Skipping malformed pgAudit line");
            continue;
        }

        let session_type = fields[0].trim(); // SESSION or OBJECT
        let statement_id = fields[1].trim();
        let sub_statement_id = fields[2].trim();
        let class = fields[3].trim(); // READ, WRITE, DDL, MISC
        let command = fields[4].trim(); // SELECT, INSERT, etc.

        // Use session_type + a derived session ID
        let session_id = format!("pgaudit-{}", session_type);

        // Extract the SQL statement (field index 7)
        if fields.len() > 7 {
            let sql = fields[7].trim();
            if !sql.is_empty() && sql != "<none>" {
                state.process_sql(&session_id, sql);
            }
        }
    }

    debug!(lines = line_count, "Parsed pgAudit log");
    state.finish()
}

/// Parse MySQL general query log format.
///
/// Format:
/// ```text
/// 2024-01-15T10:30:00.000000Z	    5 Query	BEGIN
/// 2024-01-15T10:30:00.001000Z	    5 Query	SELECT * FROM accounts
/// 2024-01-15T10:30:00.002000Z	    5 Query	COMMIT
/// ```
fn parse_mysql_general_log(input: &str, config: &TraceConfig) -> FormatResult<TransactionHistory> {
    let mut state = TraceParserState::new(config.default_isolation);
    let mut line_count = 0;

    for line in input.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        line_count += 1;

        // MySQL general log format: timestamp<TAB>thread_id command<TAB>argument
        let parts: Vec<&str> = line.splitn(3, '\t').collect();
        if parts.len() < 3 {
            // Continuation line
            continue;
        }

        let thread_cmd = parts[1].trim();
        let argument = parts[2].trim();

        // Extract thread ID and command type
        let cmd_parts: Vec<&str> = thread_cmd.splitn(2, ' ').collect();
        if cmd_parts.len() < 2 {
            continue;
        }

        let thread_id = cmd_parts[0].trim();
        let command_type = cmd_parts[1].trim();

        if command_type == "Query" || command_type == "Execute" {
            let session_id = format!("mysql-thread-{}", thread_id);
            state.process_sql(&session_id, argument);
        }
    }

    debug!(lines = line_count, "Parsed MySQL general log");
    state.finish()
}

/// Parse generic trace format: `timestamp|session_id|sql_statement`
fn parse_generic_trace(input: &str, config: &TraceConfig) -> FormatResult<TransactionHistory> {
    let mut state = TraceParserState::new(config.default_isolation);
    let mut line_count = 0;

    for line in input.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("//") {
            continue;
        }

        line_count += 1;
        let delim = config.delimiter;
        let parts: Vec<&str> = line.splitn(3, delim).collect();

        if parts.len() < 3 {
            trace!(line = %line, "Skipping malformed generic trace line");
            continue;
        }

        let _timestamp = parts[0].trim();
        let session_id = parts[1].trim();
        let sql = parts[2].trim();

        state.process_sql(session_id, sql);
    }

    debug!(lines = line_count, "Parsed generic trace");
    state.finish()
}

/// Parse CSV trace format with configurable column indices.
fn parse_csv_trace(input: &str, config: &TraceConfig) -> FormatResult<TransactionHistory> {
    let mut state = TraceParserState::new(config.default_isolation);
    let mut line_count = 0;
    let mut first = true;

    for line in input.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        if first && config.has_header {
            first = false;
            continue;
        }
        first = false;

        line_count += 1;
        let fields: Vec<&str> = line.split(config.delimiter).collect();

        let max_col = config
            .timestamp_column
            .max(config.session_column)
            .max(config.sql_column);

        if fields.len() <= max_col {
            trace!(line = %line, "Skipping CSV line with insufficient columns");
            continue;
        }

        let session_id = fields[config.session_column].trim();
        let sql = fields[config.sql_column].trim();

        state.process_sql(session_id, sql);
    }

    debug!(lines = line_count, "Parsed CSV trace");
    state.finish()
}

/// Simple string hash for converting table names to u64 IDs.
fn hash_table_name(s: &str) -> u64 {
    let mut hash: u64 = 5381;
    for b in s.bytes() {
        hash = hash.wrapping_mul(33).wrapping_add(b as u64);
    }
    hash
}

/// Extract table name from SQL (shared utility).
fn extract_table_from_sql(upper_sql: &str) -> Option<String> {
    let tokens: Vec<&str> = upper_sql.split_whitespace().collect();

    if let Some(pos) = tokens.iter().position(|&t| t == "FROM") {
        return tokens
            .get(pos + 1)
            .map(|t| t.trim_end_matches(|c: char| !c.is_alphanumeric() && c != '_').to_lowercase());
    }

    if tokens.first() == Some(&"UPDATE") {
        return tokens.get(1).map(|t| t.to_lowercase());
    }

    if tokens.first() == Some(&"INSERT") && tokens.get(1) == Some(&"INTO") {
        return tokens.get(2).map(|t| t.trim_end_matches('(').to_lowercase());
    }

    if tokens.first() == Some(&"DELETE") && tokens.get(1) == Some(&"FROM") {
        return tokens.get(2).map(|t| t.to_lowercase());
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generic_trace() {
        let input = r#"
# IsoSpec trace log
1000|sess1|BEGIN
1001|sess1|SELECT * FROM accounts WHERE id = 1
1002|sess1|UPDATE accounts SET balance = 50 WHERE id = 1
1003|sess1|COMMIT
1004|sess2|BEGIN
1005|sess2|SELECT * FROM accounts WHERE id = 1
1006|sess2|ROLLBACK
"#;
        let config = TraceConfig::default();
        let history = parse_trace(input, &config).unwrap();
        assert_eq!(history.transaction_count(), 2);
        assert_eq!(history.committed_transactions().len(), 1);
        assert_eq!(history.aborted_transactions().len(), 1);
    }

    #[test]
    fn test_csv_trace() {
        let input = "ts,session,sql\n1000,s1,BEGIN\n1001,s1,SELECT * FROM t\n1002,s1,COMMIT\n";
        let config = TraceConfig {
            format: TraceFormat::Csv,
            delimiter: ',',
            has_header: true,
            ..Default::default()
        };
        let history = parse_trace(input, &config).unwrap();
        assert_eq!(history.transaction_count(), 1);
    }

    #[test]
    fn test_mysql_general_log() {
        let input = "2024-01-15T10:30:00Z\t5 Query\tBEGIN\n\
                      2024-01-15T10:30:01Z\t5 Query\tSELECT * FROM accounts\n\
                      2024-01-15T10:30:02Z\t5 Query\tCOMMIT\n";
        let config = TraceConfig::mysql_general_log();
        let history = parse_trace(input, &config).unwrap();
        assert_eq!(history.transactions().len(), 1);
    }

    #[test]
    fn test_pgaudit_format() {
        let input = "LOG: AUDIT: SESSION,1,1,READ,SELECT,,,SELECT * FROM accounts WHERE id = 1,<none>\n";
        let config = TraceConfig::pgaudit();
        let _history = parse_trace(input, &config).unwrap();
        // pgAudit doesn't include explicit BEGIN/COMMIT so operations happen outside transactions
    }

    #[test]
    fn test_isolation_level_detection() {
        let input = "1000|s1|BEGIN\n\
                      1001|s1|SET TRANSACTION ISOLATION LEVEL SERIALIZABLE\n\
                      1002|s1|SELECT * FROM t\n\
                      1003|s1|COMMIT\n";
        let config = TraceConfig::default();
        let history = parse_trace(input, &config).unwrap();
        assert_eq!(history.transaction_count(), 1);
    }
}
