//! TLS record layer (RFC 5246 §6, RFC 8446 §5).
//!
//! Handles record framing, content type identification, version validation,
//! fragmentation, reassembly, and serialization.

use crate::version::TlsVersion;
use bytes::{Buf, BufMut, BytesMut};
use nom::{
    bytes::complete::take,
    combinator::map,
    number::complete::{be_u16, be_u8},
    IResult,
};
use serde::{Deserialize, Serialize};
use std::fmt;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum TLS record payload size (2^14 = 16384 bytes).
pub const MAX_RECORD_PAYLOAD: usize = 16384;

/// Maximum TLS record payload with expansion for TLS 1.3 (16384 + 256).
pub const MAX_RECORD_PAYLOAD_TLS13: usize = 16640;

/// TLS record header is always 5 bytes: content_type(1) + version(2) + length(2).
pub const RECORD_HEADER_SIZE: usize = 5;

/// Maximum total record size (header + payload).
pub const MAX_RECORD_SIZE: usize = RECORD_HEADER_SIZE + MAX_RECORD_PAYLOAD_TLS13;

// ---------------------------------------------------------------------------
// Content type
// ---------------------------------------------------------------------------

/// TLS record content type (RFC 5246 §6.2.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum ContentType {
    ChangeCipherSpec = 20,
    Alert = 21,
    Handshake = 22,
    ApplicationData = 23,
    Heartbeat = 24,
}

impl ContentType {
    pub fn from_u8(val: u8) -> Option<Self> {
        match val {
            20 => Some(Self::ChangeCipherSpec),
            21 => Some(Self::Alert),
            22 => Some(Self::Handshake),
            23 => Some(Self::ApplicationData),
            24 => Some(Self::Heartbeat),
            _ => None,
        }
    }

    pub fn to_u8(self) -> u8 {
        self as u8
    }

    pub fn is_handshake(self) -> bool {
        matches!(self, Self::Handshake)
    }

    pub fn is_application_data(self) -> bool {
        matches!(self, Self::ApplicationData)
    }
}

impl fmt::Display for ContentType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ChangeCipherSpec => write!(f, "ChangeCipherSpec(20)"),
            Self::Alert => write!(f, "Alert(21)"),
            Self::Handshake => write!(f, "Handshake(22)"),
            Self::ApplicationData => write!(f, "ApplicationData(23)"),
            Self::Heartbeat => write!(f, "Heartbeat(24)"),
        }
    }
}

// ---------------------------------------------------------------------------
// Alert types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum AlertLevel {
    Warning = 1,
    Fatal = 2,
}

impl AlertLevel {
    pub fn from_u8(val: u8) -> Option<Self> {
        match val {
            1 => Some(Self::Warning),
            2 => Some(Self::Fatal),
            _ => None,
        }
    }
}

impl fmt::Display for AlertLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Warning => write!(f, "Warning"),
            Self::Fatal => write!(f, "Fatal"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum AlertDescription {
    CloseNotify = 0,
    UnexpectedMessage = 10,
    BadRecordMac = 20,
    DecryptionFailed = 21,
    RecordOverflow = 22,
    DecompressionFailure = 30,
    HandshakeFailure = 40,
    NoCertificate = 41,
    BadCertificate = 42,
    UnsupportedCertificate = 43,
    CertificateRevoked = 44,
    CertificateExpired = 45,
    CertificateUnknown = 46,
    IllegalParameter = 47,
    UnknownCa = 48,
    AccessDenied = 49,
    DecodeError = 50,
    DecryptError = 51,
    ExportRestriction = 60,
    ProtocolVersion = 70,
    InsufficientSecurity = 71,
    InternalError = 80,
    InappropriateFallback = 86,
    UserCanceled = 90,
    NoRenegotiation = 100,
    MissingExtension = 109,
    UnsupportedExtension = 110,
    UnrecognizedName = 112,
    BadCertificateStatusResponse = 113,
    UnknownPskIdentity = 115,
    CertificateRequired = 116,
    NoApplicationProtocol = 120,
}

impl AlertDescription {
    pub fn from_u8(val: u8) -> Option<Self> {
        match val {
            0 => Some(Self::CloseNotify),
            10 => Some(Self::UnexpectedMessage),
            20 => Some(Self::BadRecordMac),
            21 => Some(Self::DecryptionFailed),
            22 => Some(Self::RecordOverflow),
            30 => Some(Self::DecompressionFailure),
            40 => Some(Self::HandshakeFailure),
            41 => Some(Self::NoCertificate),
            42 => Some(Self::BadCertificate),
            43 => Some(Self::UnsupportedCertificate),
            44 => Some(Self::CertificateRevoked),
            45 => Some(Self::CertificateExpired),
            46 => Some(Self::CertificateUnknown),
            47 => Some(Self::IllegalParameter),
            48 => Some(Self::UnknownCa),
            49 => Some(Self::AccessDenied),
            50 => Some(Self::DecodeError),
            51 => Some(Self::DecryptError),
            60 => Some(Self::ExportRestriction),
            70 => Some(Self::ProtocolVersion),
            71 => Some(Self::InsufficientSecurity),
            80 => Some(Self::InternalError),
            86 => Some(Self::InappropriateFallback),
            90 => Some(Self::UserCanceled),
            100 => Some(Self::NoRenegotiation),
            109 => Some(Self::MissingExtension),
            110 => Some(Self::UnsupportedExtension),
            112 => Some(Self::UnrecognizedName),
            113 => Some(Self::BadCertificateStatusResponse),
            115 => Some(Self::UnknownPskIdentity),
            116 => Some(Self::CertificateRequired),
            120 => Some(Self::NoApplicationProtocol),
            _ => None,
        }
    }

    pub fn to_u8(self) -> u8 {
        self as u8
    }
}

impl fmt::Display for AlertDescription {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}({})", self, *self as u8)
    }
}

/// A parsed TLS alert message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TlsAlert {
    pub level: AlertLevel,
    pub description: AlertDescription,
}

impl TlsAlert {
    pub fn new(level: AlertLevel, description: AlertDescription) -> Self {
        Self { level, description }
    }

    pub fn fatal(desc: AlertDescription) -> Self {
        Self::new(AlertLevel::Fatal, desc)
    }

    pub fn warning(desc: AlertDescription) -> Self {
        Self::new(AlertLevel::Warning, desc)
    }

    pub fn serialize(&self) -> Vec<u8> {
        vec![self.level as u8, self.description as u8]
    }

    pub fn parse(input: &[u8]) -> IResult<&[u8], Self> {
        let (input, level_byte) = be_u8(input)?;
        let (input, desc_byte) = be_u8(input)?;
        let level = AlertLevel::from_u8(level_byte)
            .unwrap_or(AlertLevel::Fatal);
        let description = AlertDescription::from_u8(desc_byte)
            .unwrap_or(AlertDescription::InternalError);
        Ok((input, TlsAlert { level, description }))
    }
}

// ---------------------------------------------------------------------------
// TLS Record
// ---------------------------------------------------------------------------

/// A single TLS record (RFC 5246 §6.2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TlsRecord {
    pub content_type: ContentType,
    pub protocol_version: TlsVersion,
    pub fragment: Vec<u8>,
}

impl TlsRecord {
    pub fn new(content_type: ContentType, version: TlsVersion, fragment: Vec<u8>) -> Self {
        Self {
            content_type,
            protocol_version: version,
            fragment,
        }
    }

    /// Length of the fragment (payload).
    pub fn payload_length(&self) -> usize {
        self.fragment.len()
    }

    /// Total wire size of this record (header + payload).
    pub fn wire_size(&self) -> usize {
        RECORD_HEADER_SIZE + self.fragment.len()
    }

    /// Serialize this record to bytes.
    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(self.wire_size());
        buf.push(self.content_type.to_u8());
        buf.push(self.protocol_version.major);
        buf.push(self.protocol_version.minor);
        let len = self.fragment.len() as u16;
        buf.push((len >> 8) as u8);
        buf.push((len & 0xFF) as u8);
        buf.extend_from_slice(&self.fragment);
        buf
    }

    /// Serialize into a BytesMut buffer.
    pub fn serialize_into(&self, buf: &mut BytesMut) {
        buf.put_u8(self.content_type.to_u8());
        buf.put_u8(self.protocol_version.major);
        buf.put_u8(self.protocol_version.minor);
        buf.put_u16(self.fragment.len() as u16);
        buf.put_slice(&self.fragment);
    }

    /// Parse a TLS record from a byte slice using nom.
    pub fn parse(input: &[u8]) -> IResult<&[u8], Self> {
        let (input, ct_byte) = be_u8(input)?;
        let (input, major) = be_u8(input)?;
        let (input, minor) = be_u8(input)?;
        let (input, length) = be_u16(input)?;
        let (input, fragment) = take(length as usize)(input)?;

        let content_type = ContentType::from_u8(ct_byte).ok_or_else(|| {
            nom::Err::Error(nom::error::Error::new(input, nom::error::ErrorKind::Tag))
        })?;

        let version = TlsVersion::new(major, minor);

        Ok((
            input,
            TlsRecord {
                content_type,
                protocol_version: version,
                fragment: fragment.to_vec(),
            },
        ))
    }

    /// Validate this record according to TLS rules.
    pub fn validate(&self) -> Result<(), RecordError> {
        if !self.protocol_version.is_known() && self.protocol_version != TlsVersion::new(3, 3) {
            return Err(RecordError::UnknownVersion(self.protocol_version));
        }
        if self.fragment.len() > MAX_RECORD_PAYLOAD_TLS13 {
            return Err(RecordError::PayloadTooLarge {
                size: self.fragment.len(),
                max: MAX_RECORD_PAYLOAD_TLS13,
            });
        }
        if self.fragment.is_empty() && self.content_type != ContentType::ApplicationData {
            return Err(RecordError::EmptyFragment(self.content_type));
        }
        Ok(())
    }

    /// Check if this record contains a ChangeCipherSpec message.
    pub fn is_change_cipher_spec(&self) -> bool {
        self.content_type == ContentType::ChangeCipherSpec
            && self.fragment.len() == 1
            && self.fragment[0] == 1
    }

    /// Create a ChangeCipherSpec record.
    pub fn change_cipher_spec(version: TlsVersion) -> Self {
        Self::new(ContentType::ChangeCipherSpec, version, vec![1])
    }
}

impl fmt::Display for TlsRecord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TlsRecord({}, {}, {} bytes)",
            self.content_type,
            self.protocol_version,
            self.fragment.len()
        )
    }
}

// ---------------------------------------------------------------------------
// Record errors
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, thiserror::Error)]
pub enum RecordError {
    #[error("unknown protocol version: {0}")]
    UnknownVersion(TlsVersion),

    #[error("record payload too large: {size} bytes (max {max})")]
    PayloadTooLarge { size: usize, max: usize },

    #[error("empty fragment for content type {0}")]
    EmptyFragment(ContentType),

    #[error("incomplete record: need {needed} bytes, have {available}")]
    Incomplete { needed: usize, available: usize },

    #[error("invalid content type byte: 0x{0:02x}")]
    InvalidContentType(u8),

    #[error("reassembly buffer overflow: {size} > {max}")]
    ReassemblyOverflow { size: usize, max: usize },
}

// ---------------------------------------------------------------------------
// Record layer (stateful framing)
// ---------------------------------------------------------------------------

/// Stateful record layer that handles framing, fragmentation, and reassembly.
pub struct RecordLayer {
    /// Expected protocol version for incoming records.
    pub version: TlsVersion,
    /// Read buffer for incremental record parsing.
    read_buf: BytesMut,
    /// Reassembly buffers per content type.
    reassembly: std::collections::HashMap<u8, Vec<u8>>,
    /// Maximum fragment size for outgoing records.
    max_fragment_size: usize,
    /// Whether we've seen a ChangeCipherSpec (encryption active).
    cipher_active: bool,
    /// Sequence number for outgoing records.
    write_seq: u64,
    /// Sequence number for incoming records.
    read_seq: u64,
    /// Total bytes read through this layer.
    bytes_read: u64,
    /// Total bytes written through this layer.
    bytes_written: u64,
}

impl RecordLayer {
    pub fn new(version: TlsVersion) -> Self {
        Self {
            version,
            read_buf: BytesMut::with_capacity(MAX_RECORD_SIZE * 2),
            reassembly: std::collections::HashMap::new(),
            max_fragment_size: MAX_RECORD_PAYLOAD,
            cipher_active: false,
            write_seq: 0,
            read_seq: 0,
            bytes_read: 0,
            bytes_written: 0,
        }
    }

    /// Set the maximum outgoing fragment size.
    pub fn set_max_fragment_size(&mut self, size: usize) {
        self.max_fragment_size = size.min(MAX_RECORD_PAYLOAD);
    }

    /// Feed raw bytes into the read buffer.
    pub fn feed(&mut self, data: &[u8]) {
        self.read_buf.extend_from_slice(data);
        self.bytes_read += data.len() as u64;
    }

    /// Try to read a complete TLS record from the buffer.
    pub fn try_read_record(&mut self) -> Result<Option<TlsRecord>, RecordError> {
        if self.read_buf.len() < RECORD_HEADER_SIZE {
            return Ok(None);
        }

        let ct_byte = self.read_buf[0];
        let _major = self.read_buf[1];
        let _minor = self.read_buf[2];
        let length = ((self.read_buf[3] as usize) << 8) | (self.read_buf[4] as usize);

        if length > MAX_RECORD_PAYLOAD_TLS13 {
            return Err(RecordError::PayloadTooLarge {
                size: length,
                max: MAX_RECORD_PAYLOAD_TLS13,
            });
        }

        let total_needed = RECORD_HEADER_SIZE + length;
        if self.read_buf.len() < total_needed {
            return Ok(None);
        }

        let record_bytes = self.read_buf.split_to(total_needed);
        match TlsRecord::parse(&record_bytes) {
            Ok((_, record)) => {
                if ContentType::from_u8(ct_byte).is_none() {
                    return Err(RecordError::InvalidContentType(ct_byte));
                }
                self.read_seq += 1;
                Ok(Some(record))
            }
            Err(_) => Err(RecordError::InvalidContentType(ct_byte)),
        }
    }

    /// Read all complete records currently in the buffer.
    pub fn read_all_records(&mut self) -> Result<Vec<TlsRecord>, RecordError> {
        let mut records = Vec::new();
        loop {
            match self.try_read_record()? {
                Some(record) => records.push(record),
                None => break,
            }
        }
        Ok(records)
    }

    /// Fragment data into appropriately-sized TLS records.
    pub fn fragment(
        &mut self,
        content_type: ContentType,
        data: &[u8],
    ) -> Vec<TlsRecord> {
        let mut records = Vec::new();
        if data.is_empty() {
            if content_type == ContentType::ApplicationData {
                records.push(TlsRecord::new(content_type, self.version, Vec::new()));
                self.write_seq += 1;
            }
            return records;
        }

        for chunk in data.chunks(self.max_fragment_size) {
            let record = TlsRecord::new(content_type, self.version, chunk.to_vec());
            self.bytes_written += record.wire_size() as u64;
            self.write_seq += 1;
            records.push(record);
        }
        records
    }

    /// Reassemble fragmented handshake records into a complete message.
    pub fn reassemble_handshake(
        &mut self,
        records: &[TlsRecord],
    ) -> Result<Vec<u8>, RecordError> {
        let key = ContentType::Handshake.to_u8();
        let buf = self.reassembly.entry(key).or_default();

        for record in records {
            if record.content_type != ContentType::Handshake {
                continue;
            }
            let new_len = buf.len() + record.fragment.len();
            if new_len > MAX_RECORD_PAYLOAD * 16 {
                return Err(RecordError::ReassemblyOverflow {
                    size: new_len,
                    max: MAX_RECORD_PAYLOAD * 16,
                });
            }
            buf.extend_from_slice(&record.fragment);
        }

        // Check if we have a complete handshake message.
        // Handshake header: type(1) + length(3) = 4 bytes minimum.
        let result = if buf.len() >= 4 {
            let msg_len =
                ((buf[1] as usize) << 16) | ((buf[2] as usize) << 8) | (buf[3] as usize);
            let total_needed = 4 + msg_len;
            if buf.len() >= total_needed {
                let message = buf[..total_needed].to_vec();
                let remaining = buf[total_needed..].to_vec();
                self.reassembly.insert(key, remaining);
                Ok(message)
            } else {
                Ok(Vec::new())
            }
        } else {
            Ok(Vec::new())
        };

        result
    }

    /// Activate cipher (after ChangeCipherSpec).
    pub fn activate_cipher(&mut self) {
        self.cipher_active = true;
        self.write_seq = 0;
        self.read_seq = 0;
    }

    pub fn is_cipher_active(&self) -> bool {
        self.cipher_active
    }

    pub fn write_sequence(&self) -> u64 {
        self.write_seq
    }

    pub fn read_sequence(&self) -> u64 {
        self.read_seq
    }

    pub fn bytes_read(&self) -> u64 {
        self.bytes_read
    }

    pub fn bytes_written(&self) -> u64 {
        self.bytes_written
    }

    /// How many bytes remain in the read buffer.
    pub fn buffered_bytes(&self) -> usize {
        self.read_buf.len()
    }

    /// Clear the read buffer and reassembly state.
    pub fn reset(&mut self) {
        self.read_buf.clear();
        self.reassembly.clear();
        self.cipher_active = false;
        self.write_seq = 0;
        self.read_seq = 0;
    }

    /// Serialize a single record to bytes.
    pub fn serialize_record(&self, record: &TlsRecord) -> Vec<u8> {
        record.serialize()
    }

    /// Serialize multiple records to a contiguous byte stream.
    pub fn serialize_records(&self, records: &[TlsRecord]) -> Vec<u8> {
        let total_size: usize = records.iter().map(|r| r.wire_size()).sum();
        let mut buf = Vec::with_capacity(total_size);
        for record in records {
            buf.extend_from_slice(&record.serialize());
        }
        buf
    }
}

impl fmt::Debug for RecordLayer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RecordLayer")
            .field("version", &self.version)
            .field("cipher_active", &self.cipher_active)
            .field("write_seq", &self.write_seq)
            .field("read_seq", &self.read_seq)
            .field("buffered", &self.read_buf.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Multi-record parsing
// ---------------------------------------------------------------------------

/// Parse multiple TLS records from a byte slice.
pub fn parse_records(input: &[u8]) -> IResult<&[u8], Vec<TlsRecord>> {
    let mut records = Vec::new();
    let mut remaining = input;
    while remaining.len() >= RECORD_HEADER_SIZE {
        let length_offset = 3;
        if remaining.len() < 5 {
            break;
        }
        let payload_len =
            ((remaining[length_offset] as usize) << 8) | (remaining[length_offset + 1] as usize);
        if remaining.len() < RECORD_HEADER_SIZE + payload_len {
            break;
        }
        match TlsRecord::parse(remaining) {
            Ok((rest, record)) => {
                records.push(record);
                remaining = rest;
            }
            Err(_) => break,
        }
    }
    Ok((remaining, records))
}

/// Validate a sequence of records for version consistency.
pub fn validate_record_versions(
    records: &[TlsRecord],
    expected: TlsVersion,
) -> Vec<RecordError> {
    let mut errors = Vec::new();
    for record in records {
        if record.protocol_version != expected
            && !(expected >= TlsVersion::TLS1_3
                && record.protocol_version == TlsVersion::TLS1_2)
        {
            errors.push(RecordError::UnknownVersion(record.protocol_version));
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

    #[test]
    fn test_content_type_roundtrip() {
        for ct in [
            ContentType::ChangeCipherSpec,
            ContentType::Alert,
            ContentType::Handshake,
            ContentType::ApplicationData,
            ContentType::Heartbeat,
        ] {
            let byte = ct.to_u8();
            let decoded = ContentType::from_u8(byte).unwrap();
            assert_eq!(ct, decoded);
        }
    }

    #[test]
    fn test_content_type_invalid() {
        assert!(ContentType::from_u8(0).is_none());
        assert!(ContentType::from_u8(19).is_none());
        assert!(ContentType::from_u8(255).is_none());
    }

    #[test]
    fn test_record_serialize_parse_roundtrip() {
        let record = TlsRecord::new(
            ContentType::Handshake,
            TlsVersion::TLS1_2,
            vec![0x01, 0x00, 0x00, 0x05, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE],
        );
        let bytes = record.serialize();
        assert_eq!(bytes[0], 22); // Handshake
        assert_eq!(bytes[1], 3);
        assert_eq!(bytes[2], 3); // TLS 1.2
        let (remaining, parsed) = TlsRecord::parse(&bytes).unwrap();
        assert!(remaining.is_empty());
        assert_eq!(parsed, record);
    }

    #[test]
    fn test_record_validate() {
        let good = TlsRecord::new(
            ContentType::Handshake,
            TlsVersion::TLS1_2,
            vec![0x01],
        );
        assert!(good.validate().is_ok());

        let bad_version = TlsRecord::new(
            ContentType::Handshake,
            TlsVersion::new(2, 0),
            vec![0x01],
        );
        assert!(bad_version.validate().is_err());

        let too_large = TlsRecord::new(
            ContentType::Handshake,
            TlsVersion::TLS1_2,
            vec![0u8; MAX_RECORD_PAYLOAD_TLS13 + 1],
        );
        assert!(too_large.validate().is_err());
    }

    #[test]
    fn test_record_layer_fragment() {
        let mut layer = RecordLayer::new(TlsVersion::TLS1_2);
        layer.set_max_fragment_size(100);
        let data = vec![0xABu8; 250];
        let records = layer.fragment(ContentType::ApplicationData, &data);
        assert_eq!(records.len(), 3);
        assert_eq!(records[0].fragment.len(), 100);
        assert_eq!(records[1].fragment.len(), 100);
        assert_eq!(records[2].fragment.len(), 50);
    }

    #[test]
    fn test_record_layer_feed_and_read() {
        let mut layer = RecordLayer::new(TlsVersion::TLS1_2);
        let record = TlsRecord::new(
            ContentType::Handshake,
            TlsVersion::TLS1_2,
            vec![0x01, 0x00, 0x00, 0x00],
        );
        let bytes = record.serialize();
        layer.feed(&bytes);
        let parsed = layer.try_read_record().unwrap().unwrap();
        assert_eq!(parsed.content_type, ContentType::Handshake);
        assert_eq!(parsed.fragment, vec![0x01, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn test_record_layer_incremental_feed() {
        let mut layer = RecordLayer::new(TlsVersion::TLS1_2);
        let record = TlsRecord::new(
            ContentType::Handshake,
            TlsVersion::TLS1_2,
            vec![0xFF; 10],
        );
        let bytes = record.serialize();

        // Feed partial header.
        layer.feed(&bytes[..3]);
        assert!(layer.try_read_record().unwrap().is_none());

        // Feed rest of header but not full payload.
        layer.feed(&bytes[3..7]);
        assert!(layer.try_read_record().unwrap().is_none());

        // Feed remaining.
        layer.feed(&bytes[7..]);
        let parsed = layer.try_read_record().unwrap().unwrap();
        assert_eq!(parsed.fragment.len(), 10);
    }

    #[test]
    fn test_record_layer_multiple_records() {
        let mut layer = RecordLayer::new(TlsVersion::TLS1_2);
        let r1 = TlsRecord::new(ContentType::Handshake, TlsVersion::TLS1_2, vec![0x01]);
        let r2 = TlsRecord::new(ContentType::Alert, TlsVersion::TLS1_2, vec![0x02, 0x28]);
        let mut bytes = r1.serialize();
        bytes.extend_from_slice(&r2.serialize());
        layer.feed(&bytes);

        let records = layer.read_all_records().unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].content_type, ContentType::Handshake);
        assert_eq!(records[1].content_type, ContentType::Alert);
    }

    #[test]
    fn test_record_layer_reassemble_handshake() {
        let mut layer = RecordLayer::new(TlsVersion::TLS1_2);
        // Handshake message: type=1 (ClientHello), length=6, body=[AA;6]
        let msg = vec![0x01, 0x00, 0x00, 0x06, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA, 0xAA];
        let r1 = TlsRecord::new(ContentType::Handshake, TlsVersion::TLS1_2, msg[..5].to_vec());
        let r2 = TlsRecord::new(ContentType::Handshake, TlsVersion::TLS1_2, msg[5..].to_vec());

        // First fragment should not complete the message.
        let result1 = layer.reassemble_handshake(&[r1]).unwrap();
        assert!(result1.is_empty());

        // Second fragment completes it.
        let result2 = layer.reassemble_handshake(&[r2]).unwrap();
        assert_eq!(result2, msg);
    }

    #[test]
    fn test_change_cipher_spec_record() {
        let record = TlsRecord::change_cipher_spec(TlsVersion::TLS1_2);
        assert!(record.is_change_cipher_spec());
        assert_eq!(record.content_type, ContentType::ChangeCipherSpec);
        assert_eq!(record.fragment, vec![1]);
    }

    #[test]
    fn test_parse_multiple_records() {
        let r1 = TlsRecord::new(ContentType::Handshake, TlsVersion::TLS1_2, vec![0x01, 0x00, 0x00, 0x00]);
        let r2 = TlsRecord::new(ContentType::ApplicationData, TlsVersion::TLS1_2, vec![0xFF; 5]);
        let mut bytes = r1.serialize();
        bytes.extend_from_slice(&r2.serialize());

        let (remaining, records) = parse_records(&bytes).unwrap();
        assert!(remaining.is_empty());
        assert_eq!(records.len(), 2);
    }

    #[test]
    fn test_alert_serialize_parse() {
        let alert = TlsAlert::fatal(AlertDescription::HandshakeFailure);
        let bytes = alert.serialize();
        assert_eq!(bytes, vec![2, 40]);
        let (_, parsed) = TlsAlert::parse(&bytes).unwrap();
        assert_eq!(parsed.level, AlertLevel::Fatal);
        assert_eq!(parsed.description, AlertDescription::HandshakeFailure);
    }

    #[test]
    fn test_record_layer_sequence_numbers() {
        let mut layer = RecordLayer::new(TlsVersion::TLS1_2);
        assert_eq!(layer.write_sequence(), 0);
        assert_eq!(layer.read_sequence(), 0);

        let _ = layer.fragment(ContentType::ApplicationData, &[1, 2, 3]);
        assert_eq!(layer.write_sequence(), 1);

        let record = TlsRecord::new(ContentType::Handshake, TlsVersion::TLS1_2, vec![0x01]);
        layer.feed(&record.serialize());
        let _ = layer.try_read_record().unwrap();
        assert_eq!(layer.read_sequence(), 1);
    }

    #[test]
    fn test_record_wire_size() {
        let record = TlsRecord::new(ContentType::Handshake, TlsVersion::TLS1_2, vec![0u8; 100]);
        assert_eq!(record.wire_size(), RECORD_HEADER_SIZE + 100);
        assert_eq!(record.payload_length(), 100);
    }

    #[test]
    fn test_record_layer_reset() {
        let mut layer = RecordLayer::new(TlsVersion::TLS1_2);
        layer.feed(&[1, 2, 3]);
        layer.activate_cipher();
        assert!(layer.is_cipher_active());
        layer.reset();
        assert!(!layer.is_cipher_active());
        assert_eq!(layer.buffered_bytes(), 0);
    }
}
