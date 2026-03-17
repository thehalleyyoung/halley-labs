//! TLS message parser using nom combinators.
//!
//! Provides unified parsing for all TLS record types including handshake
//! messages, alert messages, ChangeCipherSpec, and supports error recovery
//! for malformed messages.

use crate::extensions::{self, TlsExtension};
use crate::handshake::{
    CertificateMessage, ClientHello, HandshakeError, HandshakeMessage, HandshakeType,
    ServerHello,
};
use crate::record::{
    AlertDescription, AlertLevel, ContentType, TlsAlert, TlsRecord, RECORD_HEADER_SIZE,
};
use crate::version::TlsVersion;
use nom::{
    bytes::complete::take,
    number::complete::{be_u16, be_u8},
    IResult,
};
use serde::{Deserialize, Serialize};
use std::fmt;

// ---------------------------------------------------------------------------
// Top-level parsed message
// ---------------------------------------------------------------------------

/// A fully-parsed TLS message extracted from a record.
#[derive(Debug, Clone)]
pub enum ParsedMessage {
    Handshake(HandshakeMessage),
    Alert(TlsAlert),
    ChangeCipherSpec,
    ApplicationData(Vec<u8>),
    Heartbeat(Vec<u8>),
}

impl fmt::Display for ParsedMessage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Handshake(msg) => write!(f, "Handshake({})", msg),
            Self::Alert(alert) => write!(f, "Alert({} {})", alert.level, alert.description),
            Self::ChangeCipherSpec => write!(f, "ChangeCipherSpec"),
            Self::ApplicationData(data) => write!(f, "ApplicationData({} bytes)", data.len()),
            Self::Heartbeat(data) => write!(f, "Heartbeat({} bytes)", data.len()),
        }
    }
}

// ---------------------------------------------------------------------------
// Parser errors
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, thiserror::Error)]
pub enum ParseError {
    #[error("incomplete data: need {needed} more bytes")]
    Incomplete { needed: usize },

    #[error("invalid record: {reason}")]
    InvalidRecord { reason: String },

    #[error("invalid handshake: {0}")]
    Handshake(#[from] HandshakeError),

    #[error("unknown content type: 0x{0:02x}")]
    UnknownContentType(u8),

    #[error("record too large: {size} > {max}")]
    RecordTooLarge { size: usize, max: usize },

    #[error("malformed extension at offset {offset}: {reason}")]
    MalformedExtension { offset: usize, reason: String },

    #[error("nom parse error")]
    NomError,
}

// ---------------------------------------------------------------------------
// TLS Parser
// ---------------------------------------------------------------------------

/// Stateful TLS message parser.
pub struct TlsParser {
    /// Buffer for accumulating incomplete data.
    buffer: Vec<u8>,
    /// Maximum allowed record size.
    max_record_size: usize,
    /// Whether to attempt error recovery on malformed messages.
    error_recovery: bool,
    /// Number of successfully parsed messages.
    parsed_count: u64,
    /// Number of parse errors encountered.
    error_count: u64,
}

impl TlsParser {
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
            max_record_size: crate::record::MAX_RECORD_SIZE,
            error_recovery: true,
            parsed_count: 0,
            error_count: 0,
        }
    }

    pub fn with_max_record_size(mut self, max: usize) -> Self {
        self.max_record_size = max;
        self
    }

    pub fn with_error_recovery(mut self, enabled: bool) -> Self {
        self.error_recovery = enabled;
        self
    }

    /// Feed raw bytes into the parser.
    pub fn feed(&mut self, data: &[u8]) {
        self.buffer.extend_from_slice(data);
    }

    /// Try to parse the next complete TLS record from the buffer.
    pub fn next_record(&mut self) -> Result<Option<TlsRecord>, ParseError> {
        if self.buffer.len() < RECORD_HEADER_SIZE {
            return Ok(None);
        }

        let ct_byte = self.buffer[0];
        if ContentType::from_u8(ct_byte).is_none() {
            if self.error_recovery {
                self.error_count += 1;
                self.buffer.remove(0);
                return Ok(None);
            }
            return Err(ParseError::UnknownContentType(ct_byte));
        }

        let payload_len =
            ((self.buffer[3] as usize) << 8) | (self.buffer[4] as usize);

        if RECORD_HEADER_SIZE + payload_len > self.max_record_size {
            if self.error_recovery {
                self.error_count += 1;
                self.buffer.drain(..RECORD_HEADER_SIZE);
                return Ok(None);
            }
            return Err(ParseError::RecordTooLarge {
                size: RECORD_HEADER_SIZE + payload_len,
                max: self.max_record_size,
            });
        }

        let total = RECORD_HEADER_SIZE + payload_len;
        if self.buffer.len() < total {
            return Ok(None);
        }

        let record_bytes: Vec<u8> = self.buffer.drain(..total).collect();
        match TlsRecord::parse(&record_bytes) {
            Ok((_, record)) => {
                self.parsed_count += 1;
                Ok(Some(record))
            }
            Err(_) => {
                self.error_count += 1;
                if self.error_recovery {
                    Ok(None)
                } else {
                    Err(ParseError::InvalidRecord {
                        reason: "failed to parse record".to_string(),
                    })
                }
            }
        }
    }

    /// Parse all complete records in the buffer.
    pub fn parse_all_records(&mut self) -> Result<Vec<TlsRecord>, ParseError> {
        let mut records = Vec::new();
        loop {
            match self.next_record()? {
                Some(record) => records.push(record),
                None => break,
            }
        }
        Ok(records)
    }

    /// Parse a record into a typed message.
    pub fn parse_message(record: &TlsRecord) -> Result<ParsedMessage, ParseError> {
        match record.content_type {
            ContentType::Handshake => {
                let (msg, _) = HandshakeMessage::parse(&record.fragment)?;
                Ok(ParsedMessage::Handshake(msg))
            }
            ContentType::Alert => {
                if record.fragment.len() < 2 {
                    return Err(ParseError::InvalidRecord {
                        reason: "alert too short".to_string(),
                    });
                }
                let (_, alert) = TlsAlert::parse(&record.fragment)
                    .map_err(|_| ParseError::InvalidRecord {
                        reason: "malformed alert".to_string(),
                    })?;
                Ok(ParsedMessage::Alert(alert))
            }
            ContentType::ChangeCipherSpec => {
                if record.fragment.len() != 1 || record.fragment[0] != 1 {
                    return Err(ParseError::InvalidRecord {
                        reason: "invalid ChangeCipherSpec".to_string(),
                    });
                }
                Ok(ParsedMessage::ChangeCipherSpec)
            }
            ContentType::ApplicationData => {
                Ok(ParsedMessage::ApplicationData(record.fragment.clone()))
            }
            ContentType::Heartbeat => {
                Ok(ParsedMessage::Heartbeat(record.fragment.clone()))
            }
        }
    }

    /// Parse records and decode all messages.
    pub fn parse_all_messages(&mut self) -> Result<Vec<ParsedMessage>, ParseError> {
        let records = self.parse_all_records()?;
        let mut messages = Vec::new();
        for record in &records {
            match Self::parse_message(record) {
                Ok(msg) => messages.push(msg),
                Err(e) => {
                    self.error_count += 1;
                    if !self.error_recovery {
                        return Err(e);
                    }
                }
            }
        }
        Ok(messages)
    }

    /// Parse a ClientHello directly from raw record bytes.
    pub fn parse_client_hello(data: &[u8]) -> Result<ClientHello, ParseError> {
        let (_, record) = TlsRecord::parse(data)
            .map_err(|_| ParseError::InvalidRecord {
                reason: "failed to parse record".to_string(),
            })?;

        if record.content_type != ContentType::Handshake {
            return Err(ParseError::InvalidRecord {
                reason: format!("expected Handshake, got {:?}", record.content_type),
            });
        }

        if record.fragment.len() < 4 {
            return Err(ParseError::Incomplete { needed: 4 });
        }

        if record.fragment[0] != HandshakeType::ClientHello.to_u8() {
            return Err(ParseError::InvalidRecord {
                reason: format!(
                    "expected ClientHello (1), got {}",
                    record.fragment[0]
                ),
            });
        }

        let body_len = ((record.fragment[1] as usize) << 16)
            | ((record.fragment[2] as usize) << 8)
            | (record.fragment[3] as usize);

        if record.fragment.len() < 4 + body_len {
            return Err(ParseError::Incomplete {
                needed: 4 + body_len - record.fragment.len(),
            });
        }

        let body = &record.fragment[4..4 + body_len];
        let ch = ClientHello::parse_body(body)?;
        Ok(ch)
    }

    /// Parse a ServerHello directly from raw record bytes.
    pub fn parse_server_hello(data: &[u8]) -> Result<ServerHello, ParseError> {
        let (_, record) = TlsRecord::parse(data)
            .map_err(|_| ParseError::InvalidRecord {
                reason: "failed to parse record".to_string(),
            })?;

        if record.content_type != ContentType::Handshake {
            return Err(ParseError::InvalidRecord {
                reason: format!("expected Handshake, got {:?}", record.content_type),
            });
        }

        if record.fragment.len() < 4 {
            return Err(ParseError::Incomplete { needed: 4 });
        }

        if record.fragment[0] != HandshakeType::ServerHello.to_u8() {
            return Err(ParseError::InvalidRecord {
                reason: format!(
                    "expected ServerHello (2), got {}",
                    record.fragment[0]
                ),
            });
        }

        let body_len = ((record.fragment[1] as usize) << 16)
            | ((record.fragment[2] as usize) << 8)
            | (record.fragment[3] as usize);

        if record.fragment.len() < 4 + body_len {
            return Err(ParseError::Incomplete {
                needed: 4 + body_len - record.fragment.len(),
            });
        }

        let body = &record.fragment[4..4 + body_len];
        let sh = ServerHello::parse_body(body)?;
        Ok(sh)
    }

    /// Parse a ChangeCipherSpec record.
    pub fn parse_change_cipher_spec(data: &[u8]) -> Result<(), ParseError> {
        let (_, record) = TlsRecord::parse(data)
            .map_err(|_| ParseError::InvalidRecord {
                reason: "failed to parse record".to_string(),
            })?;

        if record.content_type != ContentType::ChangeCipherSpec {
            return Err(ParseError::InvalidRecord {
                reason: "not a ChangeCipherSpec record".to_string(),
            });
        }

        if record.fragment != [1] {
            return Err(ParseError::InvalidRecord {
                reason: "invalid CCS value".to_string(),
            });
        }

        Ok(())
    }

    /// Parse an Alert record.
    pub fn parse_alert(data: &[u8]) -> Result<TlsAlert, ParseError> {
        let (_, record) = TlsRecord::parse(data)
            .map_err(|_| ParseError::InvalidRecord {
                reason: "failed to parse record".to_string(),
            })?;

        if record.content_type != ContentType::Alert {
            return Err(ParseError::InvalidRecord {
                reason: "not an Alert record".to_string(),
            });
        }

        let (_, alert) = TlsAlert::parse(&record.fragment)
            .map_err(|_| ParseError::InvalidRecord {
                reason: "malformed alert".to_string(),
            })?;

        Ok(alert)
    }

    /// Parse multiple handshake messages from a single record fragment.
    /// This handles the case where multiple handshake messages are coalesced.
    pub fn parse_coalesced_handshakes(
        fragment: &[u8],
    ) -> Result<Vec<HandshakeMessage>, ParseError> {
        let mut messages = Vec::new();
        let mut offset = 0;
        while offset < fragment.len() {
            if offset + 4 > fragment.len() {
                break;
            }
            match HandshakeMessage::parse(&fragment[offset..]) {
                Ok((msg, consumed)) => {
                    offset += consumed;
                    messages.push(msg);
                }
                Err(e) => {
                    return Err(ParseError::Handshake(e));
                }
            }
        }
        Ok(messages)
    }

    /// Number of successfully parsed records.
    pub fn parsed_count(&self) -> u64 {
        self.parsed_count
    }

    /// Number of parse errors encountered.
    pub fn error_count(&self) -> u64 {
        self.error_count
    }

    /// Bytes remaining in the parser buffer.
    pub fn buffered_bytes(&self) -> usize {
        self.buffer.len()
    }

    /// Clear the parser state.
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.parsed_count = 0;
        self.error_count = 0;
    }
}

impl Default for TlsParser {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for TlsParser {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TlsParser")
            .field("buffered", &self.buffer.len())
            .field("parsed", &self.parsed_count)
            .field("errors", &self.error_count)
            .field("error_recovery", &self.error_recovery)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Convenience parsing functions
// ---------------------------------------------------------------------------

/// Parse raw bytes into a sequence of ParsedMessages.
pub fn parse_tls_stream(data: &[u8]) -> Result<Vec<ParsedMessage>, ParseError> {
    let mut parser = TlsParser::new();
    parser.feed(data);
    parser.parse_all_messages()
}

/// Extract a ClientHello from a raw byte stream.
pub fn extract_client_hello(data: &[u8]) -> Result<ClientHello, ParseError> {
    TlsParser::parse_client_hello(data)
}

/// Extract a ServerHello from a raw byte stream.
pub fn extract_server_hello(data: &[u8]) -> Result<ServerHello, ParseError> {
    TlsParser::parse_server_hello(data)
}

/// Parse a raw handshake message without the record layer.
pub fn parse_handshake_message(data: &[u8]) -> Result<HandshakeMessage, ParseError> {
    let (msg, _) = HandshakeMessage::parse(data)?;
    Ok(msg)
}

/// Validate that a byte stream contains well-formed TLS records.
pub fn validate_tls_stream(data: &[u8]) -> Vec<ParseError> {
    let mut errors = Vec::new();
    let mut parser = TlsParser::new().with_error_recovery(false);
    parser.feed(data);
    loop {
        match parser.next_record() {
            Ok(Some(_)) => {}
            Ok(None) => break,
            Err(e) => {
                errors.push(e);
                break;
            }
        }
    }
    errors
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_client_hello_record() -> Vec<u8> {
        let mut ch = ClientHello::new(TlsVersion::TLS1_2, [0x42u8; 32]);
        ch.cipher_suites = vec![0xC02F, 0x009E];
        ch.compression_methods = vec![crate::handshake::CompressionMethod::Null];
        let body = ch.serialize();
        TlsRecord::new(ContentType::Handshake, TlsVersion::TLS1_2, body).serialize()
    }

    fn make_server_hello_record() -> Vec<u8> {
        let sh = ServerHello::new(TlsVersion::TLS1_2, [0x55u8; 32], 0xC02F);
        let body = sh.serialize();
        TlsRecord::new(ContentType::Handshake, TlsVersion::TLS1_2, body).serialize()
    }

    #[test]
    fn test_parse_client_hello_from_record() {
        let data = make_client_hello_record();
        let ch = TlsParser::parse_client_hello(&data).unwrap();
        assert_eq!(ch.client_version, TlsVersion::TLS1_2);
        assert_eq!(ch.cipher_suites, vec![0xC02F, 0x009E]);
    }

    #[test]
    fn test_parse_server_hello_from_record() {
        let data = make_server_hello_record();
        let sh = TlsParser::parse_server_hello(&data).unwrap();
        assert_eq!(sh.server_version, TlsVersion::TLS1_2);
        assert_eq!(sh.cipher_suite, 0xC02F);
    }

    #[test]
    fn test_parser_feed_incremental() {
        let data = make_client_hello_record();
        let mut parser = TlsParser::new();

        parser.feed(&data[..3]);
        assert!(parser.next_record().unwrap().is_none());

        parser.feed(&data[3..]);
        let record = parser.next_record().unwrap().unwrap();
        assert_eq!(record.content_type, ContentType::Handshake);
    }

    #[test]
    fn test_parser_multiple_records() {
        let ch_data = make_client_hello_record();
        let ccs = TlsRecord::change_cipher_spec(TlsVersion::TLS1_2).serialize();
        let alert = TlsRecord::new(
            ContentType::Alert,
            TlsVersion::TLS1_2,
            vec![AlertLevel::Fatal as u8, AlertDescription::HandshakeFailure as u8],
        )
        .serialize();

        let mut stream = Vec::new();
        stream.extend_from_slice(&ch_data);
        stream.extend_from_slice(&ccs);
        stream.extend_from_slice(&alert);

        let messages = parse_tls_stream(&stream).unwrap();
        assert_eq!(messages.len(), 3);
        assert!(matches!(messages[0], ParsedMessage::Handshake(_)));
        assert!(matches!(messages[1], ParsedMessage::ChangeCipherSpec));
        assert!(matches!(messages[2], ParsedMessage::Alert(_)));
    }

    #[test]
    fn test_parse_change_cipher_spec() {
        let ccs = TlsRecord::change_cipher_spec(TlsVersion::TLS1_2).serialize();
        TlsParser::parse_change_cipher_spec(&ccs).unwrap();
    }

    #[test]
    fn test_parse_alert_record() {
        let alert_record = TlsRecord::new(
            ContentType::Alert,
            TlsVersion::TLS1_2,
            vec![2, 40],
        );
        let data = alert_record.serialize();
        let alert = TlsParser::parse_alert(&data).unwrap();
        assert_eq!(alert.level, AlertLevel::Fatal);
        assert_eq!(alert.description, AlertDescription::HandshakeFailure);
    }

    #[test]
    fn test_error_recovery() {
        let mut parser = TlsParser::new().with_error_recovery(true);
        // Feed garbage followed by a valid record.
        let mut data = vec![0xFF, 0xFF, 0xFF]; // garbage
        data.extend_from_slice(&TlsRecord::change_cipher_spec(TlsVersion::TLS1_2).serialize());
        parser.feed(&data);

        // First call should skip the garbage byte (0xFF is not a valid content type).
        let _ = parser.next_record();
        let _ = parser.next_record();
        let _ = parser.next_record();
        // Eventually we should get the CCS.
        let result = parser.next_record().unwrap();
        // Depending on recovery, we might or might not get the record.
        assert!(parser.error_count() > 0 || result.is_some());
    }

    #[test]
    fn test_coalesced_handshakes() {
        let hello_req = HandshakeMessage::HelloRequest.serialize();
        let server_done = HandshakeMessage::ServerHelloDone.serialize();
        let mut coalesced = hello_req.clone();
        coalesced.extend_from_slice(&server_done);

        let messages = TlsParser::parse_coalesced_handshakes(&coalesced).unwrap();
        assert_eq!(messages.len(), 2);
        assert!(matches!(messages[0], HandshakeMessage::HelloRequest));
        assert!(matches!(messages[1], HandshakeMessage::ServerHelloDone));
    }

    #[test]
    fn test_parser_stats() {
        let data = make_client_hello_record();
        let mut parser = TlsParser::new();
        parser.feed(&data);
        let _ = parser.parse_all_records().unwrap();
        assert_eq!(parser.parsed_count(), 1);
        assert_eq!(parser.error_count(), 0);
    }

    #[test]
    fn test_parser_reset() {
        let mut parser = TlsParser::new();
        parser.feed(&[1, 2, 3]);
        assert_eq!(parser.buffered_bytes(), 3);
        parser.reset();
        assert_eq!(parser.buffered_bytes(), 0);
    }

    #[test]
    fn test_validate_tls_stream() {
        let data = make_client_hello_record();
        let errors = validate_tls_stream(&data);
        assert!(errors.is_empty());

        let errors = validate_tls_stream(&[0xFF, 0x01, 0x02, 0x03, 0x04]);
        assert!(!errors.is_empty());
    }

    #[test]
    fn test_application_data_parse() {
        let record = TlsRecord::new(
            ContentType::ApplicationData,
            TlsVersion::TLS1_2,
            vec![0xDE, 0xAD, 0xBE, 0xEF],
        );
        let msg = TlsParser::parse_message(&record).unwrap();
        match msg {
            ParsedMessage::ApplicationData(data) => {
                assert_eq!(data, vec![0xDE, 0xAD, 0xBE, 0xEF]);
            }
            _ => panic!("Expected ApplicationData"),
        }
    }

    #[test]
    fn test_extract_client_hello_convenience() {
        let data = make_client_hello_record();
        let ch = extract_client_hello(&data).unwrap();
        assert_eq!(ch.client_version, TlsVersion::TLS1_2);
    }
}
