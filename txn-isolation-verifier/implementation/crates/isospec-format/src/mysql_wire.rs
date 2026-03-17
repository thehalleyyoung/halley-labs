//! MySQL wire protocol packet parser for trace extraction.
//!
//! Parses MySQL client/server protocol packets to extract transaction events:
//! - COM_QUERY commands
//! - COM_STMT_PREPARE / COM_STMT_EXECUTE sequences
//! - OK/ERR/EOF response packets
//! - Transaction state changes
//!
//! Reference: <https://dev.mysql.com/doc/dev/mysql-server/latest/page_protocol_basic_packets.html>

use crate::{FormatError, FormatResult};
use isospec_history::builder::HistoryBuilder;
use isospec_types::identifier::{ItemId, TableId, TransactionId};
use isospec_types::isolation::IsolationLevel;
use isospec_types::value::Value;
use std::collections::HashMap;
use tracing::{debug, trace, warn};

/// MySQL command bytes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MysqlCommand {
    ComSleep = 0x00,
    ComQuit = 0x01,
    ComInitDb = 0x02,
    ComQuery = 0x03,
    ComFieldList = 0x04,
    ComCreateDb = 0x05,
    ComDropDb = 0x06,
    ComRefresh = 0x07,
    ComStmtPrepare = 0x16,
    ComStmtExecute = 0x17,
    ComStmtClose = 0x19,
    ComStmtReset = 0x1a,
    ComSetOption = 0x1b,
    ComStmtFetch = 0x1c,
}

impl MysqlCommand {
    pub fn from_byte(b: u8) -> Option<Self> {
        match b {
            0x00 => Some(Self::ComSleep),
            0x01 => Some(Self::ComQuit),
            0x02 => Some(Self::ComInitDb),
            0x03 => Some(Self::ComQuery),
            0x04 => Some(Self::ComFieldList),
            0x05 => Some(Self::ComCreateDb),
            0x06 => Some(Self::ComDropDb),
            0x07 => Some(Self::ComRefresh),
            0x16 => Some(Self::ComStmtPrepare),
            0x17 => Some(Self::ComStmtExecute),
            0x19 => Some(Self::ComStmtClose),
            0x1a => Some(Self::ComStmtReset),
            0x1b => Some(Self::ComSetOption),
            0x1c => Some(Self::ComStmtFetch),
            _ => None,
        }
    }
}

/// MySQL server status flags (from OK packet)
bitflags::bitflags! {
    /// Server status flags from MySQL OK/EOF packets.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct ServerStatus: u16 {
        const IN_TRANS = 0x0001;
        const AUTOCOMMIT = 0x0002;
        const MORE_RESULTS_EXISTS = 0x0008;
        const NO_GOOD_INDEX_USED = 0x0010;
        const NO_INDEX_USED = 0x0020;
        const CURSOR_EXISTS = 0x0040;
        const LAST_ROW_SENT = 0x0080;
        const DB_DROPPED = 0x0100;
        const NO_BACKSLASH_ESCAPES = 0x0200;
        const METADATA_CHANGED = 0x0400;
        const QUERY_WAS_SLOW = 0x0800;
        const PS_OUT_PARAMS = 0x1000;
        const IN_TRANS_READONLY = 0x2000;
        const SESSION_STATE_CHANGED = 0x4000;
    }
}

/// A parsed MySQL protocol packet
#[derive(Debug, Clone)]
pub struct MysqlPacket {
    /// Sequence ID (0-255, wrapping)
    pub sequence_id: u8,
    /// Packet payload
    pub payload: Vec<u8>,
    /// Capture timestamp (nanoseconds)
    pub timestamp_ns: u64,
}

/// MySQL response packet type
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MysqlResponse {
    Ok {
        affected_rows: u64,
        last_insert_id: u64,
        status_flags: u16,
        warnings: u16,
    },
    Err {
        error_code: u16,
        sql_state: String,
        message: String,
    },
    Eof {
        warnings: u16,
        status_flags: u16,
    },
    ResultSet,
}

/// Tracks state for a MySQL connection
#[derive(Debug)]
struct MysqlConnectionState {
    txn_id: Option<TransactionId>,
    isolation_level: IsolationLevel,
    in_transaction: bool,
    autocommit: bool,
    next_txn_counter: u64,
    current_query: Option<String>,
    prepared_statements: HashMap<u32, String>,
    session_id: String,
}

impl MysqlConnectionState {
    fn new(session_id: String) -> Self {
        Self {
            txn_id: None,
            isolation_level: IsolationLevel::RepeatableRead, // MySQL default
            in_transaction: false,
            autocommit: true,
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

/// Parser for MySQL wire protocol packets.
///
/// Extracts transaction events from MySQL client/server protocol traffic,
/// building a `TransactionHistory` for isolation analysis.
///
/// # Example
/// ```no_run
/// use isospec_format::mysql_wire::MysqlWireParser;
///
/// let mut parser = MysqlWireParser::new("conn-1");
/// // Feed captured packets...
/// // let history = parser.finish().unwrap();
/// ```
pub struct MysqlWireParser {
    state: MysqlConnectionState,
    builder: HistoryBuilder,
    packet_count: u64,
}

impl MysqlWireParser {
    /// Create a new MySQL wire protocol parser.
    pub fn new(session_id: &str) -> Self {
        Self {
            state: MysqlConnectionState::new(session_id.to_string()),
            builder: HistoryBuilder::new(),
            packet_count: 0,
        }
    }

    /// Parse a raw MySQL packet from bytes.
    ///
    /// MySQL packet format: 3-byte length (LE) + 1-byte sequence_id + payload
    /// Returns number of bytes consumed.
    pub fn parse_packet(&mut self, buf: &[u8]) -> FormatResult<(usize, MysqlPacket)> {
        if buf.len() < 4 {
            return Err(FormatError::IncompleteMessage {
                expected: 4,
                actual: buf.len(),
            });
        }

        let payload_len =
            buf[0] as usize | (buf[1] as usize) << 8 | (buf[2] as usize) << 16;
        let sequence_id = buf[3];

        let total_len = 4 + payload_len;
        if buf.len() < total_len {
            return Err(FormatError::IncompleteMessage {
                expected: total_len,
                actual: buf.len(),
            });
        }

        let packet = MysqlPacket {
            sequence_id,
            payload: buf[4..total_len].to_vec(),
            timestamp_ns: 0,
        };

        self.packet_count += 1;
        Ok((total_len, packet))
    }

    /// Process a client command packet.
    pub fn process_client_packet(&mut self, packet: &MysqlPacket) -> FormatResult<()> {
        if packet.payload.is_empty() {
            return Ok(());
        }

        let cmd_byte = packet.payload[0];
        match MysqlCommand::from_byte(cmd_byte) {
            Some(MysqlCommand::ComQuery) => {
                let sql = String::from_utf8_lossy(&packet.payload[1..]).to_string();
                self.state.current_query = Some(sql.clone());
                trace!(sql = %sql, "COM_QUERY");
                self.process_sql(&sql)?;
            }
            Some(MysqlCommand::ComStmtPrepare) => {
                let sql = String::from_utf8_lossy(&packet.payload[1..]).to_string();
                trace!(sql = %sql, "COM_STMT_PREPARE");
            }
            Some(MysqlCommand::ComStmtExecute) if packet.payload.len() >= 5 => {
                let stmt_id = u32::from_le_bytes([
                    packet.payload[1],
                    packet.payload[2],
                    packet.payload[3],
                    packet.payload[4],
                ]);
                if let Some(sql) = self.state.prepared_statements.get(&stmt_id).cloned() {
                    self.process_sql(&sql)?;
                }
            }
            Some(MysqlCommand::ComQuit) => {
                // Connection closing; commit any open transaction
                if let Some(txn_id) = self.state.txn_id.take() {
                    self.builder
                        .abort_transaction(txn_id, Some("Connection closed".to_string()));
                    self.state.in_transaction = false;
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// Process a server response packet.
    pub fn process_server_packet(&mut self, packet: &MysqlPacket) -> FormatResult<()> {
        if packet.payload.is_empty() {
            return Ok(());
        }

        match packet.payload[0] {
            0x00 => {
                // OK packet
                let status_flags = if packet.payload.len() > 3 {
                    // Simplified: real parsing needs length-encoded integers
                    let pos = 1; // skip header
                    let (_affected, _pos2) = read_lenenc_int(&packet.payload[pos..]);
                    // Status flags are later in the packet
                    0u16
                } else {
                    0u16
                };
                self.handle_ok_packet(status_flags);
            }
            0xff => {
                // ERR packet
                if packet.payload.len() >= 3 {
                    let error_code =
                        u16::from_le_bytes([packet.payload[1], packet.payload[2]]);
                    debug!(error_code = error_code, "MySQL ERR packet");

                    // 1213 = ER_LOCK_DEADLOCK, 1205 = ER_LOCK_WAIT_TIMEOUT
                    if error_code == 1213 || error_code == 1205 {
                        if let Some(txn_id) = self.state.txn_id.take() {
                            self.builder.abort_transaction(
                                txn_id,
                                Some("Deadlock or lock timeout".to_string()),
                            );
                            self.state.in_transaction = false;
                        }
                    }
                }
            }
            0xfe if packet.payload.len() < 9 => {
                // EOF packet (only in pre-deprecation protocol)
            }
            _ => {
                // Result set or other response
            }
        }
        Ok(())
    }

    /// Finish parsing and build the transaction history.
    pub fn finish(self) -> FormatResult<isospec_history::history::TransactionHistory> {
        debug!(
            packets = self.packet_count,
            session = %self.state.session_id,
            "MysqlWireParser finished"
        );
        Ok(self.builder.build()?)
    }

    /// Number of packets processed.
    pub fn packet_count(&self) -> u64 {
        self.packet_count
    }

    fn handle_ok_packet(&mut self, status_flags: u16) {
        let in_trans = status_flags & 0x0001 != 0;
        let autocommit = status_flags & 0x0002 != 0;

        if self.state.in_transaction && !in_trans {
            // Transaction ended
            if let Some(txn_id) = self.state.txn_id.take() {
                self.builder.commit_transaction(txn_id);
                debug!(txn = ?txn_id, "MySQL transaction committed");
            }
            self.state.in_transaction = false;
        } else if !self.state.in_transaction && in_trans {
            // Transaction started
            let txn_id = self.state.allocate_txn_id();
            self.builder
                .begin_transaction(txn_id, self.state.isolation_level);
            self.state.txn_id = Some(txn_id);
            self.state.in_transaction = true;
            debug!(txn = ?txn_id, "MySQL transaction started");
        }

        self.state.autocommit = autocommit;
    }

    fn process_sql(&mut self, sql: &str) -> FormatResult<()> {
        let upper = sql.trim().to_uppercase();

        // Detect isolation level changes
        if upper.contains("SET TRANSACTION ISOLATION LEVEL")
            || upper.contains("SET SESSION TRANSACTION ISOLATION LEVEL")
        {
            if upper.contains("SERIALIZABLE") {
                self.state.isolation_level = IsolationLevel::Serializable;
            } else if upper.contains("REPEATABLE READ") {
                self.state.isolation_level = IsolationLevel::RepeatableRead;
            } else if upper.contains("READ COMMITTED") {
                self.state.isolation_level = IsolationLevel::ReadCommitted;
            } else if upper.contains("READ UNCOMMITTED") {
                self.state.isolation_level = IsolationLevel::ReadUncommitted;
            }
            debug!(level = ?self.state.isolation_level, "MySQL isolation level changed");
            return Ok(());
        }

        if upper.starts_with("BEGIN") || upper.starts_with("START TRANSACTION") {
            if !self.state.in_transaction {
                let txn_id = self.state.allocate_txn_id();
                self.builder
                    .begin_transaction(txn_id, self.state.isolation_level);
                self.state.txn_id = Some(txn_id);
                self.state.in_transaction = true;
            }
            return Ok(());
        }

        if upper.starts_with("COMMIT") {
            if let Some(txn_id) = self.state.txn_id.take() {
                self.builder.commit_transaction(txn_id);
                self.state.in_transaction = false;
            }
            return Ok(());
        }

        if upper.starts_with("ROLLBACK") {
            if let Some(txn_id) = self.state.txn_id.take() {
                self.builder
                    .abort_transaction(txn_id, Some("Explicit ROLLBACK".to_string()));
                self.state.in_transaction = false;
            }
            return Ok(());
        }

        // DML in transaction
        if let Some(txn_id) = self.state.txn_id {
            let table = extract_mysql_table(&upper);
            let table_id = TableId::new(hash_table_name(table.as_deref().unwrap_or("unknown")));
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

        Ok(())
    }
}

/// Read a MySQL length-encoded integer.
fn read_lenenc_int(buf: &[u8]) -> (u64, usize) {
    if buf.is_empty() {
        return (0, 0);
    }
    match buf[0] {
        0..=0xfb => (buf[0] as u64, 1),
        0xfc if buf.len() >= 3 => {
            let val = u16::from_le_bytes([buf[1], buf[2]]) as u64;
            (val, 3)
        }
        0xfd if buf.len() >= 4 => {
            let val = buf[1] as u64 | (buf[2] as u64) << 8 | (buf[3] as u64) << 16;
            (val, 4)
        }
        0xfe if buf.len() >= 9 => {
            let val = u64::from_le_bytes([
                buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7], buf[8],
            ]);
            (val, 9)
        }
        _ => (0, 1),
    }
}

/// Simple string hash for converting table names to u64 IDs.
fn hash_table_name(s: &str) -> u64 {
    let mut hash: u64 = 5381;
    for b in s.bytes() {
        hash = hash.wrapping_mul(33).wrapping_add(b as u64);
    }
    hash
}

/// Extract table name from MySQL SQL (best-effort).
fn extract_mysql_table(upper_sql: &str) -> Option<String> {
    let tokens: Vec<&str> = upper_sql.split_whitespace().collect();

    if let Some(pos) = tokens.iter().position(|&t| t == "FROM") {
        return tokens.get(pos + 1).map(|t| {
            t.trim_matches(|c: char| c == '`' || c == '\'' || c == '"')
                .to_lowercase()
        });
    }

    if tokens.first() == Some(&"UPDATE") {
        return tokens.get(1).map(|t| t.trim_matches('`').to_lowercase());
    }

    if tokens.first() == Some(&"INSERT") && tokens.get(1) == Some(&"INTO") {
        return tokens
            .get(2)
            .map(|t| t.trim_matches(|c: char| c == '`' || c == '(').to_lowercase());
    }

    if tokens.first() == Some(&"DELETE") && tokens.get(1) == Some(&"FROM") {
        return tokens.get(2).map(|t| t.trim_matches('`').to_lowercase());
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mysql_command_from_byte() {
        assert_eq!(MysqlCommand::from_byte(0x03), Some(MysqlCommand::ComQuery));
        assert_eq!(
            MysqlCommand::from_byte(0x16),
            Some(MysqlCommand::ComStmtPrepare)
        );
        assert_eq!(MysqlCommand::from_byte(0xFF), None);
    }

    #[test]
    fn test_read_lenenc_int() {
        assert_eq!(read_lenenc_int(&[42]), (42, 1));
        assert_eq!(read_lenenc_int(&[0xfc, 0x01, 0x00]), (1, 3));
        assert_eq!(read_lenenc_int(&[0]), (0, 1));
    }

    #[test]
    fn test_extract_mysql_table() {
        assert_eq!(
            extract_mysql_table("SELECT * FROM `accounts` WHERE id = 1"),
            Some("accounts".to_string())
        );
        assert_eq!(
            extract_mysql_table("INSERT INTO `orders`(`id`) VALUES(1)"),
            Some("orders".to_string())
        );
    }

    #[test]
    fn test_mysql_wire_parser_creation() {
        let parser = MysqlWireParser::new("test-session");
        assert_eq!(parser.packet_count(), 0);
    }

    #[test]
    fn test_parse_packet() {
        let mut parser = MysqlWireParser::new("test");
        // 3-byte length (5) + seq_id (0) + 5 bytes payload
        let buf = [5, 0, 0, 0, 0x03, b'B', b'E', b'G', b'I'];
        let (consumed, packet) = parser.parse_packet(&buf).unwrap();
        assert_eq!(consumed, 9);
        assert_eq!(packet.sequence_id, 0);
        assert_eq!(packet.payload.len(), 5);
    }
}
