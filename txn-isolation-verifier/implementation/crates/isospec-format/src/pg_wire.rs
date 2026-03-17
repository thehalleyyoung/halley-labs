//! PostgreSQL wire protocol (v3) message parser for trace extraction.
//!
//! Parses the frontend/backend message flow to extract transaction events:
//! - Query messages (simple and extended protocol)
//! - Parse/Bind/Execute sequences
//! - ReadyForQuery transaction status indicators
//! - ErrorResponse for abort detection
//! - CommandComplete for commit/row-count extraction
//!
//! Reference: <https://www.postgresql.org/docs/16/protocol-message-formats.html>

use crate::{FormatError, FormatResult};
use isospec_history::builder::HistoryBuilder;
use isospec_types::identifier::TransactionId;
use isospec_types::isolation::IsolationLevel;
use isospec_types::value::Value;
use std::collections::HashMap;
use tracing::{debug, trace, warn};

/// PostgreSQL wire protocol message types (backend)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PgBackendMessage {
    Authentication,      // 'R'
    ParameterStatus,     // 'S'
    ReadyForQuery,       // 'Z'
    RowDescription,      // 'T'
    DataRow,             // 'D'
    CommandComplete,     // 'C'
    ErrorResponse,       // 'E'
    NoticeResponse,      // 'N'
    ParseComplete,       // '1'
    BindComplete,        // '2'
    CloseComplete,       // '3'
    EmptyQueryResponse,  // 'I'
    ParameterDescription,// 't'
    NoData,              // 'n'
    NotificationResponse,// 'A'
    CopyInResponse,      // 'G'
    CopyOutResponse,     // 'H'
    CopyDone,            // 'c'
}

/// PostgreSQL wire protocol message types (frontend)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PgFrontendMessage {
    Query,       // 'Q'
    Parse,       // 'P'
    Bind,        // 'B'
    Execute,     // 'E'
    Describe,    // 'D'
    Close,       // 'C'
    Sync,        // 'S'
    Flush,       // 'H'
    Terminate,   // 'X'
    CopyData,    // 'd'
    CopyDone,    // 'c'
    CopyFail,    // 'f'
}

/// Transaction status from ReadyForQuery
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PgTransactionStatus {
    Idle,           // 'I' - not in a transaction
    InTransaction,  // 'T' - in a transaction block
    Failed,         // 'E' - in a failed transaction block
}

/// A parsed PostgreSQL wire protocol message
#[derive(Debug, Clone)]
pub struct PgMessage {
    /// Message direction
    pub direction: MessageDirection,
    /// Raw message type byte
    pub type_byte: u8,
    /// Message payload (excluding type byte and length)
    pub payload: Vec<u8>,
    /// Timestamp when the message was captured (nanoseconds)
    pub timestamp_ns: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageDirection {
    Frontend,
    Backend,
}

/// Tracks state for a single PostgreSQL connection
#[derive(Debug)]
struct ConnectionState {
    txn_id: Option<TransactionId>,
    isolation_level: IsolationLevel,
    in_transaction: bool,
    next_txn_counter: u64,
    current_query: Option<String>,
    /// Named prepared statements -> SQL text
    prepared_statements: HashMap<String, String>,
    session_id: String,
}

impl ConnectionState {
    fn new(session_id: String) -> Self {
        Self {
            txn_id: None,
            isolation_level: IsolationLevel::ReadCommitted,
            in_transaction: false,
            next_txn_counter: 1,
            current_query: None,
            prepared_statements: HashMap::new(),
            session_id,
        }
    }

    fn allocate_txn_id(&mut self) -> TransactionId {
        let id = TransactionId::from(self.next_txn_counter);
        self.next_txn_counter += 1;
        id
    }
}

/// Parser for PostgreSQL wire protocol v3 messages.
///
/// Extracts transaction events from a stream of wire protocol messages,
/// building a `TransactionHistory` suitable for isolation analysis.
///
/// # Example
/// ```no_run
/// use isospec_format::pg_wire::PgWireParser;
///
/// let mut parser = PgWireParser::new("conn-1");
/// // Feed captured packets...
/// // let history = parser.finish().unwrap();
/// ```
pub struct PgWireParser {
    state: ConnectionState,
    builder: HistoryBuilder,
    message_count: u64,
}

impl PgWireParser {
    /// Create a new parser for a PostgreSQL connection.
    pub fn new(session_id: &str) -> Self {
        Self {
            state: ConnectionState::new(session_id.to_string()),
            builder: HistoryBuilder::new(),
            message_count: 0,
        }
    }

    /// Parse a raw wire protocol message from bytes.
    ///
    /// The buffer must contain a complete message: 1-byte type + 4-byte length + payload.
    /// Returns the number of bytes consumed.
    pub fn parse_backend_message(&mut self, buf: &[u8]) -> FormatResult<usize> {
        if buf.len() < 5 {
            return Err(FormatError::IncompleteMessage {
                expected: 5,
                actual: buf.len(),
            });
        }

        let type_byte = buf[0];
        let length = u32::from_be_bytes([buf[1], buf[2], buf[3], buf[4]]) as usize;

        if buf.len() < 1 + length {
            return Err(FormatError::IncompleteMessage {
                expected: 1 + length,
                actual: buf.len(),
            });
        }

        let payload = &buf[5..1 + length];
        self.message_count += 1;

        match type_byte {
            b'Z' => self.handle_ready_for_query(payload)?,
            b'C' => self.handle_command_complete(payload)?,
            b'E' => self.handle_error_response(payload)?,
            b'T' => self.handle_row_description(payload)?,
            b'D' => self.handle_data_row(payload)?,
            _ => {
                trace!(type_byte = type_byte, "Skipping unhandled backend message");
            }
        }

        Ok(1 + length)
    }

    /// Parse a frontend (client->server) message.
    pub fn parse_frontend_message(&mut self, buf: &[u8]) -> FormatResult<usize> {
        if buf.len() < 5 {
            return Err(FormatError::IncompleteMessage {
                expected: 5,
                actual: buf.len(),
            });
        }

        let type_byte = buf[0];
        let length = u32::from_be_bytes([buf[1], buf[2], buf[3], buf[4]]) as usize;

        if buf.len() < 1 + length {
            return Err(FormatError::IncompleteMessage {
                expected: 1 + length,
                actual: buf.len(),
            });
        }

        let payload = &buf[5..1 + length];

        match type_byte {
            b'Q' => self.handle_query(payload)?,
            b'P' => self.handle_parse(payload)?,
            b'B' => self.handle_bind(payload)?,
            b'E' => self.handle_execute(payload)?,
            _ => {
                trace!(type_byte = type_byte, "Skipping unhandled frontend message");
            }
        }

        Ok(1 + length)
    }

    /// Finish parsing and build the transaction history.
    pub fn finish(self) -> FormatResult<isospec_history::history::TransactionHistory> {
        debug!(
            messages = self.message_count,
            session = %self.state.session_id,
            "PgWireParser finished"
        );
        Ok(self.builder.build()?)
    }

    /// Number of messages processed so far.
    pub fn message_count(&self) -> u64 {
        self.message_count
    }

    fn handle_ready_for_query(&mut self, payload: &[u8]) -> FormatResult<()> {
        if payload.is_empty() {
            return Err(FormatError::ParseError {
                offset: 0,
                message: "Empty ReadyForQuery payload".into(),
            });
        }

        let status = match payload[0] {
            b'I' => PgTransactionStatus::Idle,
            b'T' => PgTransactionStatus::InTransaction,
            b'E' => PgTransactionStatus::Failed,
            other => {
                warn!(status = other, "Unknown transaction status byte");
                return Ok(());
            }
        };

        match status {
            PgTransactionStatus::Idle => {
                if self.state.in_transaction {
                    // Transaction just committed (implicit or explicit)
                    if let Some(txn_id) = self.state.txn_id.take() {
                        self.builder.commit_transaction(txn_id);
                        debug!(txn = ?txn_id, "Transaction committed");
                    }
                    self.state.in_transaction = false;
                }
            }
            PgTransactionStatus::InTransaction => {
                if !self.state.in_transaction {
                    let txn_id = self.state.allocate_txn_id();
                    self.builder
                        .begin_transaction(txn_id, self.state.isolation_level);
                    self.state.txn_id = Some(txn_id);
                    self.state.in_transaction = true;
                    debug!(txn = ?txn_id, "Transaction started");
                }
            }
            PgTransactionStatus::Failed => {
                if let Some(txn_id) = self.state.txn_id.take() {
                    self.builder
                        .abort_transaction(txn_id, Some("Transaction entered failed state".to_string()));
                    debug!(txn = ?txn_id, "Transaction aborted (failed state)");
                }
                self.state.in_transaction = false;
            }
        }

        Ok(())
    }

    fn handle_command_complete(&mut self, payload: &[u8]) -> FormatResult<()> {
        let tag = extract_cstring(payload);
        trace!(tag = %tag, "CommandComplete");

        // Parse command tags to detect SET TRANSACTION ISOLATION LEVEL
        let upper = tag.to_uppercase();
        if upper.starts_with("SET") {
            if let Some(query) = &self.state.current_query {
                let q = query.to_uppercase();
                if q.contains("ISOLATION LEVEL") {
                    if q.contains("SERIALIZABLE") {
                        self.state.isolation_level = IsolationLevel::Serializable;
                    } else if q.contains("REPEATABLE READ") {
                        self.state.isolation_level = IsolationLevel::RepeatableRead;
                    } else if q.contains("READ COMMITTED") {
                        self.state.isolation_level = IsolationLevel::ReadCommitted;
                    } else if q.contains("READ UNCOMMITTED") {
                        self.state.isolation_level = IsolationLevel::ReadUncommitted;
                    }
                    debug!(level = ?self.state.isolation_level, "Isolation level changed");
                }
            }
        }

        Ok(())
    }

    fn handle_error_response(&mut self, payload: &[u8]) -> FormatResult<()> {
        let fields = parse_error_fields(payload);
        let severity = fields.get(&b'S').cloned().unwrap_or_default();
        let code = fields.get(&b'C').cloned().unwrap_or_default();
        let message = fields.get(&b'M').cloned().unwrap_or_default();

        debug!(
            severity = %severity, code = %code, message = %message,
            "ErrorResponse"
        );

        // Serialization failure codes indicate isolation violations
        if code == "40001" || code == "40P01" {
            if let Some(txn_id) = self.state.txn_id {
                debug!(txn = ?txn_id, code = %code, "Serialization/deadlock failure detected");
            }
        }

        Ok(())
    }

    fn handle_row_description(&mut self, _payload: &[u8]) -> FormatResult<()> {
        // Row descriptions help us interpret DataRow messages but we
        // don't need them for basic trace extraction
        Ok(())
    }

    fn handle_data_row(&mut self, _payload: &[u8]) -> FormatResult<()> {
        // Data rows would need RowDescription context to interpret;
        // for trace-level analysis we track the read event from the query
        Ok(())
    }

    fn handle_query(&mut self, payload: &[u8]) -> FormatResult<()> {
        let sql = extract_cstring(payload);
        self.state.current_query = Some(sql.clone());
        trace!(sql = %sql, "Simple query");

        self.process_sql_statement(&sql)?;
        Ok(())
    }

    fn handle_parse(&mut self, payload: &[u8]) -> FormatResult<()> {
        // Parse: statement_name (cstring) + query (cstring) + param types
        let (name, rest) = split_cstring(payload);
        let (sql, _) = split_cstring(rest);

        if !name.is_empty() {
            self.state
                .prepared_statements
                .insert(name.to_string(), sql.to_string());
        }
        self.state.current_query = Some(sql.to_string());
        trace!(name = %name, sql = %sql, "Parse");
        Ok(())
    }

    fn handle_bind(&mut self, _payload: &[u8]) -> FormatResult<()> {
        // Bind associates parameters with a prepared statement
        // For trace extraction, the SQL from Parse is sufficient
        Ok(())
    }

    fn handle_execute(&mut self, _payload: &[u8]) -> FormatResult<()> {
        // Execute runs a bound portal; the SQL was captured at Parse time
        if let Some(sql) = self.state.current_query.clone() {
            self.process_sql_statement(&sql)?;
        }
        Ok(())
    }

    /// Classify a SQL statement and record the appropriate trace event.
    fn process_sql_statement(&mut self, sql: &str) -> FormatResult<()> {
        let upper = sql.trim().to_uppercase();

        if upper.starts_with("BEGIN") || upper.starts_with("START TRANSACTION") {
            // Explicit transaction start - isolation level may be specified inline
            if upper.contains("ISOLATION LEVEL SERIALIZABLE") {
                self.state.isolation_level = IsolationLevel::Serializable;
            } else if upper.contains("ISOLATION LEVEL REPEATABLE READ") {
                self.state.isolation_level = IsolationLevel::RepeatableRead;
            } else if upper.contains("ISOLATION LEVEL READ COMMITTED") {
                self.state.isolation_level = IsolationLevel::ReadCommitted;
            }
            return Ok(());
        }

        if upper.starts_with("COMMIT") || upper.starts_with("END") {
            return Ok(());
        }

        if upper.starts_with("ROLLBACK") || upper.starts_with("ABORT") {
            return Ok(());
        }

        // For DML statements, extract table name for trace recording
        if let Some(txn_id) = self.state.txn_id {
            let table = extract_table_name(&upper);
            let table_id = isospec_types::identifier::TableId::new(
                hash_table_name(table.as_deref().unwrap_or("unknown")),
            );

            if upper.starts_with("SELECT") {
                let item_id = isospec_types::identifier::ItemId::from(0u64);
                self.builder
                    .add_read(txn_id, table_id, item_id, Some(Value::Null));
            } else if upper.starts_with("UPDATE") {
                let item_id = isospec_types::identifier::ItemId::from(0u64);
                self.builder.add_write(
                    txn_id,
                    table_id,
                    item_id,
                    Some(Value::Null),
                    Value::Null,
                );
            } else if upper.starts_with("INSERT") {
                let item_id = isospec_types::identifier::ItemId::from(0u64);
                self.builder
                    .add_insert(txn_id, table_id, item_id, Vec::new());
            } else if upper.starts_with("DELETE") {
                let item_id = isospec_types::identifier::ItemId::from(0u64);
                self.builder.add_delete(txn_id, table_id, item_id);
            }
        }

        Ok(())
    }
}

/// Extract a null-terminated C string from a byte slice.
fn extract_cstring(buf: &[u8]) -> String {
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..end]).to_string()
}

/// Split a byte slice at the first null terminator, returning (string, rest).
fn split_cstring(buf: &[u8]) -> (String, &[u8]) {
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    let s = String::from_utf8_lossy(&buf[..end]).to_string();
    let rest = if end + 1 < buf.len() {
        &buf[end + 1..]
    } else {
        &[]
    };
    (s, rest)
}

/// Parse ErrorResponse/NoticeResponse field list.
fn parse_error_fields(payload: &[u8]) -> HashMap<u8, String> {
    let mut fields = HashMap::new();
    let mut pos = 0;
    while pos < payload.len() {
        let field_type = payload[pos];
        if field_type == 0 {
            break;
        }
        pos += 1;
        let (value, rest) = split_cstring(&payload[pos..]);
        fields.insert(field_type, value);
        pos = payload.len() - rest.len();
    }
    fields
}

/// Simple string hash for converting table names to u64 IDs.
fn hash_table_name(s: &str) -> u64 {
    let mut hash: u64 = 5381;
    for b in s.bytes() {
        hash = hash.wrapping_mul(33).wrapping_add(b as u64);
    }
    hash
}

/// Extract table name from a SQL statement (best-effort).
fn extract_table_name(upper_sql: &str) -> Option<String> {
    let tokens: Vec<&str> = upper_sql.split_whitespace().collect();

    // SELECT ... FROM table
    if let Some(pos) = tokens.iter().position(|&t| t == "FROM") {
        return tokens.get(pos + 1).map(|t| t.trim_end_matches(|c: char| !c.is_alphanumeric() && c != '_').to_lowercase());
    }

    // UPDATE table SET ...
    if tokens.first() == Some(&"UPDATE") {
        return tokens.get(1).map(|t| t.to_lowercase());
    }

    // INSERT INTO table
    if tokens.first() == Some(&"INSERT") && tokens.get(1) == Some(&"INTO") {
        return tokens.get(2).map(|t| t.trim_end_matches('(').to_lowercase());
    }

    // DELETE FROM table
    if tokens.first() == Some(&"DELETE") && tokens.get(1) == Some(&"FROM") {
        return tokens.get(2).map(|t| t.to_lowercase());
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_cstring() {
        assert_eq!(extract_cstring(b"hello\0world"), "hello");
        assert_eq!(extract_cstring(b"no null"), "no null");
        assert_eq!(extract_cstring(b"\0"), "");
    }

    #[test]
    fn test_split_cstring() {
        let (s, rest) = split_cstring(b"hello\0world\0");
        assert_eq!(s, "hello");
        assert_eq!(rest, b"world\0");
    }

    #[test]
    fn test_extract_table_name() {
        assert_eq!(
            extract_table_name("SELECT * FROM accounts WHERE id = 1"),
            Some("accounts".to_string())
        );
        assert_eq!(
            extract_table_name("UPDATE accounts SET balance = 100"),
            Some("accounts".to_string())
        );
        assert_eq!(
            extract_table_name("INSERT INTO orders(id, amount) VALUES(1, 50)"),
            Some("orders".to_string())
        );
        assert_eq!(
            extract_table_name("DELETE FROM sessions WHERE expired = TRUE"),
            Some("sessions".to_string())
        );
    }

    #[test]
    fn test_pg_wire_parser_creation() {
        let parser = PgWireParser::new("test-session");
        assert_eq!(parser.message_count(), 0);
    }

    #[test]
    fn test_incomplete_message() {
        let mut parser = PgWireParser::new("test");
        let result = parser.parse_backend_message(&[b'Z', 0, 0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_ready_for_query_idle() {
        let mut parser = PgWireParser::new("test");
        // ReadyForQuery: type='Z', length=5, status='I'
        let msg = [b'Z', 0, 0, 0, 5, b'I'];
        let consumed = parser.parse_backend_message(&msg).unwrap();
        assert_eq!(consumed, 6);
    }
}
