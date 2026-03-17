//! TLS protocol message format parsing and serialization.
//!
//! Provides wire-format parsing and encoding for core TLS handshake messages
//! (ClientHello, ServerHello, Certificate) and extensions using nom combinators.
//! Complements the higher-level handshake module with byte-level access.

use crate::extensions::TlsExtension;
use crate::version::TlsVersion;
use nom::{
    bytes::complete::take,
    multi::count,
    number::complete::{be_u16, be_u24, be_u8},
    IResult,
};
use serde::{Deserialize, Serialize};
use std::fmt;

// ---------------------------------------------------------------------------
// Wire format errors
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, thiserror::Error)]
pub enum MessageFormatError {
    #[error("truncated message: expected {expected} bytes, got {actual}")]
    Truncated { expected: usize, actual: usize },

    #[error("invalid handshake type: 0x{0:02x}")]
    InvalidHandshakeType(u8),

    #[error("invalid extension: type=0x{ext_type:04x}, {reason}")]
    InvalidExtension { ext_type: u16, reason: String },

    #[error("message too large: {size} bytes exceeds limit of {max}")]
    TooLarge { size: usize, max: usize },

    #[error("parse error: {0}")]
    Parse(String),
}

// ---------------------------------------------------------------------------
// ClientHello wire format
// ---------------------------------------------------------------------------

/// Raw ClientHello message parsed from wire format.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WireClientHello {
    /// Legacy version field (2 bytes).
    pub client_version: TlsVersion,
    /// 32 bytes of client random.
    pub random: [u8; 32],
    /// Session ID (0-32 bytes).
    pub session_id: Vec<u8>,
    /// Cipher suite IDs (2 bytes each).
    pub cipher_suites: Vec<u16>,
    /// Compression method IDs (1 byte each).
    pub compression_methods: Vec<u8>,
    /// Raw extension data as (type, data) pairs.
    pub extensions: Vec<RawExtension>,
}

impl WireClientHello {
    /// Parse a ClientHello from wire bytes (after the handshake header).
    pub fn parse(input: &[u8]) -> Result<Self, MessageFormatError> {
        match parse_wire_client_hello(input) {
            Ok((_, ch)) => Ok(ch),
            Err(_) => Err(MessageFormatError::Parse("ClientHello parse failed".into())),
        }
    }

    /// Serialize to wire format bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();

        // client_version
        buf.push(self.client_version.major);
        buf.push(self.client_version.minor);

        // random
        buf.extend_from_slice(&self.random);

        // session_id
        buf.push(self.session_id.len() as u8);
        buf.extend_from_slice(&self.session_id);

        // cipher_suites
        let cs_len = (self.cipher_suites.len() * 2) as u16;
        buf.extend_from_slice(&cs_len.to_be_bytes());
        for &cs in &self.cipher_suites {
            buf.extend_from_slice(&cs.to_be_bytes());
        }

        // compression_methods
        buf.push(self.compression_methods.len() as u8);
        buf.extend_from_slice(&self.compression_methods);

        // extensions
        if !self.extensions.is_empty() {
            let ext_bytes = serialize_raw_extensions(&self.extensions);
            let ext_len = ext_bytes.len() as u16;
            buf.extend_from_slice(&ext_len.to_be_bytes());
            buf.extend(ext_bytes);
        }

        buf
    }
}

// ---------------------------------------------------------------------------
// ServerHello wire format
// ---------------------------------------------------------------------------

/// Raw ServerHello message parsed from wire format.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WireServerHello {
    /// Server version (2 bytes).
    pub server_version: TlsVersion,
    /// 32 bytes of server random.
    pub random: [u8; 32],
    /// Session ID (0-32 bytes).
    pub session_id: Vec<u8>,
    /// Selected cipher suite ID.
    pub cipher_suite: u16,
    /// Selected compression method.
    pub compression_method: u8,
    /// Raw extension data.
    pub extensions: Vec<RawExtension>,
}

impl WireServerHello {
    /// Parse a ServerHello from wire bytes (after the handshake header).
    pub fn parse(input: &[u8]) -> Result<Self, MessageFormatError> {
        match parse_wire_server_hello(input) {
            Ok((_, sh)) => Ok(sh),
            Err(_) => Err(MessageFormatError::Parse("ServerHello parse failed".into())),
        }
    }

    /// Serialize to wire format bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();

        // server_version
        buf.push(self.server_version.major);
        buf.push(self.server_version.minor);

        // random
        buf.extend_from_slice(&self.random);

        // session_id
        buf.push(self.session_id.len() as u8);
        buf.extend_from_slice(&self.session_id);

        // cipher_suite
        buf.extend_from_slice(&self.cipher_suite.to_be_bytes());

        // compression_method
        buf.push(self.compression_method);

        // extensions
        if !self.extensions.is_empty() {
            let ext_bytes = serialize_raw_extensions(&self.extensions);
            let ext_len = ext_bytes.len() as u16;
            buf.extend_from_slice(&ext_len.to_be_bytes());
            buf.extend(ext_bytes);
        }

        buf
    }
}

// ---------------------------------------------------------------------------
// Certificate message wire format
// ---------------------------------------------------------------------------

/// Raw Certificate message: a list of DER-encoded certificate entries.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WireCertificateMessage {
    /// Individual DER-encoded certificates (leaf first, then intermediates).
    pub certificates: Vec<Vec<u8>>,
}

impl WireCertificateMessage {
    /// Parse a Certificate message from wire bytes.
    pub fn parse(input: &[u8]) -> Result<Self, MessageFormatError> {
        match parse_wire_certificate(input) {
            Ok((_, msg)) => Ok(msg),
            Err(_) => Err(MessageFormatError::Parse("Certificate parse failed".into())),
        }
    }

    /// Serialize to wire format bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut cert_data = Vec::new();
        for cert in &self.certificates {
            let len = cert.len() as u32;
            cert_data.push((len >> 16) as u8);
            cert_data.push((len >> 8) as u8);
            cert_data.push(len as u8);
            cert_data.extend_from_slice(cert);
        }

        let mut buf = Vec::new();
        let total_len = cert_data.len() as u32;
        buf.push((total_len >> 16) as u8);
        buf.push((total_len >> 8) as u8);
        buf.push(total_len as u8);
        buf.extend(cert_data);
        buf
    }

    /// Total size of all certificates in bytes.
    pub fn total_size(&self) -> usize {
        self.certificates.iter().map(|c| c.len()).sum()
    }
}

// ---------------------------------------------------------------------------
// Raw extension
// ---------------------------------------------------------------------------

/// A raw TLS extension as (type, opaque data).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RawExtension {
    /// Extension type code.
    pub ext_type: u16,
    /// Raw extension data bytes.
    pub data: Vec<u8>,
}

impl RawExtension {
    pub fn new(ext_type: u16, data: Vec<u8>) -> Self {
        Self { ext_type, data }
    }

    /// Whether this is the supported_versions extension (type 43).
    pub fn is_supported_versions(&self) -> bool {
        self.ext_type == 43
    }

    /// Whether this is the supported_groups extension (type 10).
    pub fn is_supported_groups(&self) -> bool {
        self.ext_type == 10
    }

    /// Whether this is the signature_algorithms extension (type 13).
    pub fn is_signature_algorithms(&self) -> bool {
        self.ext_type == 13
    }

    /// Whether this is the key_share extension (type 51).
    pub fn is_key_share(&self) -> bool {
        self.ext_type == 51
    }

    /// Whether this is the server_name (SNI) extension (type 0).
    pub fn is_server_name(&self) -> bool {
        self.ext_type == 0
    }
}

impl fmt::Display for RawExtension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Extension(type=0x{:04X}, {} bytes)", self.ext_type, self.data.len())
    }
}

// ---------------------------------------------------------------------------
// nom parsers
// ---------------------------------------------------------------------------

fn parse_wire_client_hello(input: &[u8]) -> IResult<&[u8], WireClientHello> {
    // client_version (2 bytes)
    let (rest, major) = be_u8(input)?;
    let (rest, minor) = be_u8(rest)?;
    let client_version = TlsVersion::new(major, minor);

    // random (32 bytes)
    let (rest, random_bytes) = take(32usize)(rest)?;
    let mut random = [0u8; 32];
    random.copy_from_slice(random_bytes);

    // session_id (1-byte length + data)
    let (rest, sid_len) = be_u8(rest)?;
    let (rest, session_id_bytes) = take(sid_len as usize)(rest)?;
    let session_id = session_id_bytes.to_vec();

    // cipher_suites (2-byte length + data)
    let (rest, cs_len) = be_u16(rest)?;
    let cs_count = (cs_len / 2) as usize;
    let (rest, cipher_suites) = count(be_u16, cs_count)(rest)?;

    // compression_methods (1-byte length + data)
    let (rest, comp_len) = be_u8(rest)?;
    let (rest, comp_bytes) = take(comp_len as usize)(rest)?;
    let compression_methods = comp_bytes.to_vec();

    // extensions (optional: 2-byte length + data)
    let extensions = if !rest.is_empty() {
        let (rest2, ext_len) = be_u16(rest)?;
        let (rest2, ext_data) = take(ext_len as usize)(rest2)?;
        parse_raw_extensions(ext_data)
            .map(|(_, exts)| exts)
            .unwrap_or_default()
    } else {
        Vec::new()
    };

    Ok((&[], WireClientHello {
        client_version,
        random,
        session_id,
        cipher_suites,
        compression_methods,
        extensions,
    }))
}

fn parse_wire_server_hello(input: &[u8]) -> IResult<&[u8], WireServerHello> {
    // server_version (2 bytes)
    let (rest, major) = be_u8(input)?;
    let (rest, minor) = be_u8(rest)?;
    let server_version = TlsVersion::new(major, minor);

    // random (32 bytes)
    let (rest, random_bytes) = take(32usize)(rest)?;
    let mut random = [0u8; 32];
    random.copy_from_slice(random_bytes);

    // session_id (1-byte length + data)
    let (rest, sid_len) = be_u8(rest)?;
    let (rest, session_id_bytes) = take(sid_len as usize)(rest)?;
    let session_id = session_id_bytes.to_vec();

    // cipher_suite (2 bytes)
    let (rest, cipher_suite) = be_u16(rest)?;

    // compression_method (1 byte)
    let (rest, compression_method) = be_u8(rest)?;

    // extensions (optional)
    let extensions = if !rest.is_empty() {
        let (rest2, ext_len) = be_u16(rest)?;
        let (rest2, ext_data) = take(ext_len as usize)(rest2)?;
        parse_raw_extensions(ext_data)
            .map(|(_, exts)| exts)
            .unwrap_or_default()
    } else {
        Vec::new()
    };

    Ok((&[], WireServerHello {
        server_version,
        random,
        session_id,
        cipher_suite,
        compression_method,
        extensions,
    }))
}

fn parse_wire_certificate(input: &[u8]) -> IResult<&[u8], WireCertificateMessage> {
    // Total certificates length (3 bytes)
    let (rest, total_len) = be_u24(input)?;
    let (rest, cert_data) = take(total_len as usize)(rest)?;

    let mut certificates = Vec::new();
    let mut pos = cert_data;

    while !pos.is_empty() {
        let (remaining, cert_len) = be_u24(pos)?;
        let (remaining, cert_bytes) = take(cert_len as usize)(remaining)?;
        certificates.push(cert_bytes.to_vec());
        pos = remaining;
    }

    Ok((rest, WireCertificateMessage { certificates }))
}

fn parse_raw_extensions(mut input: &[u8]) -> IResult<&[u8], Vec<RawExtension>> {
    let mut extensions = Vec::new();

    while !input.is_empty() {
        let (rest, ext_type) = be_u16(input)?;
        let (rest, ext_len) = be_u16(rest)?;
        let (rest, ext_data) = take(ext_len as usize)(rest)?;

        extensions.push(RawExtension {
            ext_type,
            data: ext_data.to_vec(),
        });

        input = rest;
    }

    Ok((&[], extensions))
}

fn serialize_raw_extensions(extensions: &[RawExtension]) -> Vec<u8> {
    let mut buf = Vec::new();
    for ext in extensions {
        buf.extend_from_slice(&ext.ext_type.to_be_bytes());
        let len = ext.data.len() as u16;
        buf.extend_from_slice(&len.to_be_bytes());
        buf.extend_from_slice(&ext.data);
    }
    buf
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_client_hello_bytes() -> Vec<u8> {
        let ch = WireClientHello {
            client_version: TlsVersion::TLS1_2,
            random: [0xAB; 32],
            session_id: vec![0x01, 0x02, 0x03, 0x04],
            cipher_suites: vec![0xC02F, 0x1301],
            compression_methods: vec![0x00],
            extensions: vec![
                RawExtension::new(43, vec![0x03, 0x04]), // supported_versions: TLS 1.3
            ],
        };
        ch.to_bytes()
    }

    fn make_server_hello_bytes() -> Vec<u8> {
        let sh = WireServerHello {
            server_version: TlsVersion::TLS1_2,
            random: [0xCD; 32],
            session_id: vec![0x01, 0x02, 0x03, 0x04],
            cipher_suite: 0x1301,
            compression_method: 0x00,
            extensions: vec![],
        };
        sh.to_bytes()
    }

    #[test]
    fn test_client_hello_roundtrip() {
        let bytes = make_client_hello_bytes();
        let parsed = WireClientHello::parse(&bytes).unwrap();

        assert_eq!(parsed.client_version, TlsVersion::TLS1_2);
        assert_eq!(parsed.random, [0xAB; 32]);
        assert_eq!(parsed.session_id, vec![0x01, 0x02, 0x03, 0x04]);
        assert_eq!(parsed.cipher_suites, vec![0xC02F, 0x1301]);
        assert_eq!(parsed.compression_methods, vec![0x00]);

        // Re-serialize and compare
        let reserialized = parsed.to_bytes();
        assert_eq!(bytes, reserialized);
    }

    #[test]
    fn test_server_hello_roundtrip() {
        let bytes = make_server_hello_bytes();
        let parsed = WireServerHello::parse(&bytes).unwrap();

        assert_eq!(parsed.server_version, TlsVersion::TLS1_2);
        assert_eq!(parsed.random, [0xCD; 32]);
        assert_eq!(parsed.cipher_suite, 0x1301);
        assert_eq!(parsed.compression_method, 0x00);

        let reserialized = parsed.to_bytes();
        assert_eq!(bytes, reserialized);
    }

    #[test]
    fn test_certificate_message_roundtrip() {
        let msg = WireCertificateMessage {
            certificates: vec![
                vec![0x30, 0x82, 0x01, 0x00], // dummy DER cert 1
                vec![0x30, 0x82, 0x02, 0x00], // dummy DER cert 2
            ],
        };

        let bytes = msg.to_bytes();
        let parsed = WireCertificateMessage::parse(&bytes).unwrap();

        assert_eq!(parsed.certificates.len(), 2);
        assert_eq!(parsed.certificates[0], vec![0x30, 0x82, 0x01, 0x00]);
        assert_eq!(parsed.certificates[1], vec![0x30, 0x82, 0x02, 0x00]);
    }

    #[test]
    fn test_certificate_total_size() {
        let msg = WireCertificateMessage {
            certificates: vec![vec![0; 100], vec![0; 200]],
        };
        assert_eq!(msg.total_size(), 300);
    }

    #[test]
    fn test_raw_extension_methods() {
        let ext = RawExtension::new(43, vec![0x03, 0x04]);
        assert!(ext.is_supported_versions());
        assert!(!ext.is_key_share());
        assert!(!ext.is_server_name());

        let sni = RawExtension::new(0, vec![]);
        assert!(sni.is_server_name());
        assert!(!sni.is_supported_versions());
    }

    #[test]
    fn test_raw_extension_display() {
        let ext = RawExtension::new(0x002B, vec![0x03, 0x04]);
        let s = ext.to_string();
        assert!(s.contains("002B"));
        assert!(s.contains("2 bytes"));
    }

    #[test]
    fn test_client_hello_no_extensions() {
        let ch = WireClientHello {
            client_version: TlsVersion::TLS1_0,
            random: [0; 32],
            session_id: vec![],
            cipher_suites: vec![0x002F],
            compression_methods: vec![0x00],
            extensions: vec![],
        };

        let bytes = ch.to_bytes();
        let parsed = WireClientHello::parse(&bytes).unwrap();
        assert!(parsed.extensions.is_empty());
        assert_eq!(parsed.cipher_suites, vec![0x002F]);
    }

    #[test]
    fn test_server_hello_with_extensions() {
        let sh = WireServerHello {
            server_version: TlsVersion::TLS1_2,
            random: [0xFF; 32],
            session_id: vec![],
            cipher_suite: 0xC02F,
            compression_method: 0x00,
            extensions: vec![
                RawExtension::new(43, vec![0x03, 0x03]), // supported_versions: TLS 1.2
                RawExtension::new(0xFF01, vec![0x00]),    // renegotiation_info
            ],
        };

        let bytes = sh.to_bytes();
        let parsed = WireServerHello::parse(&bytes).unwrap();
        assert_eq!(parsed.extensions.len(), 2);
        assert!(parsed.extensions[0].is_supported_versions());
    }

    #[test]
    fn test_empty_certificate_message() {
        let msg = WireCertificateMessage {
            certificates: vec![],
        };
        let bytes = msg.to_bytes();
        assert_eq!(bytes, vec![0x00, 0x00, 0x00]); // 3 bytes for zero length
        let parsed = WireCertificateMessage::parse(&bytes).unwrap();
        assert!(parsed.certificates.is_empty());
    }

    #[test]
    fn test_client_hello_parse_error() {
        let bad_bytes = vec![0x03, 0x03]; // too short
        let result = WireClientHello::parse(&bad_bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_message_format_error_display() {
        let err = MessageFormatError::Truncated {
            expected: 100,
            actual: 50,
        };
        assert!(err.to_string().contains("100"));
        assert!(err.to_string().contains("50"));
    }
}
