//! TLS handshake messages (RFC 5246 §7, RFC 8446 §4).
//!
//! Defines all handshake message types, ClientHello/ServerHello structs,
//! parsing, serialization, and handshake hash computation.

use crate::extensions::TlsExtension;
use crate::version::TlsVersion;
use sha2::{Digest, Sha256};
use serde::{Deserialize, Serialize};
use std::fmt;

// ---------------------------------------------------------------------------
// Handshake message types (RFC 5246 §7.4)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum HandshakeType {
    HelloRequest = 0,
    ClientHello = 1,
    ServerHello = 2,
    NewSessionTicket = 4,
    EndOfEarlyData = 5,
    EncryptedExtensions = 8,
    Certificate = 11,
    ServerKeyExchange = 12,
    CertificateRequest = 13,
    ServerHelloDone = 14,
    CertificateVerify = 15,
    ClientKeyExchange = 16,
    Finished = 20,
    KeyUpdate = 24,
    MessageHash = 254,
}

impl HandshakeType {
    pub fn from_u8(val: u8) -> Option<Self> {
        match val {
            0 => Some(Self::HelloRequest),
            1 => Some(Self::ClientHello),
            2 => Some(Self::ServerHello),
            4 => Some(Self::NewSessionTicket),
            5 => Some(Self::EndOfEarlyData),
            8 => Some(Self::EncryptedExtensions),
            11 => Some(Self::Certificate),
            12 => Some(Self::ServerKeyExchange),
            13 => Some(Self::CertificateRequest),
            14 => Some(Self::ServerHelloDone),
            15 => Some(Self::CertificateVerify),
            16 => Some(Self::ClientKeyExchange),
            20 => Some(Self::Finished),
            24 => Some(Self::KeyUpdate),
            254 => Some(Self::MessageHash),
            _ => None,
        }
    }

    pub fn to_u8(self) -> u8 {
        self as u8
    }
}

impl fmt::Display for HandshakeType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}({})", self, *self as u8)
    }
}

// ---------------------------------------------------------------------------
// Compression methods
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CompressionMethod {
    Null,
    Deflate,
    Unknown(u8),
}

impl CompressionMethod {
    pub fn from_u8(val: u8) -> Self {
        match val {
            0 => Self::Null,
            1 => Self::Deflate,
            v => Self::Unknown(v),
        }
    }

    pub fn to_u8(self) -> u8 {
        match self {
            Self::Null => 0,
            Self::Deflate => 1,
            Self::Unknown(v) => v,
        }
    }
}

// ---------------------------------------------------------------------------
// ClientHello
// ---------------------------------------------------------------------------

/// TLS ClientHello message (RFC 5246 §7.4.1.2, RFC 8446 §4.1.2).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClientHello {
    /// Client's maximum supported version (legacy in TLS 1.3).
    pub client_version: TlsVersion,
    /// 32 bytes of random data.
    pub random: [u8; 32],
    /// Session ID for resumption (0-32 bytes).
    pub session_id: Vec<u8>,
    /// Offered cipher suite IDs.
    pub cipher_suites: Vec<u16>,
    /// Offered compression methods.
    pub compression_methods: Vec<CompressionMethod>,
    /// Extensions.
    pub extensions: Vec<TlsExtension>,
}

impl ClientHello {
    pub fn new(version: TlsVersion, random: [u8; 32]) -> Self {
        Self {
            client_version: version,
            random,
            session_id: Vec::new(),
            cipher_suites: Vec::new(),
            compression_methods: vec![CompressionMethod::Null],
            extensions: Vec::new(),
        }
    }

    /// Serialize to handshake message body (without handshake header).
    pub fn serialize_body(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(512);

        // Version.
        buf.push(self.client_version.major);
        buf.push(self.client_version.minor);

        // Random.
        buf.extend_from_slice(&self.random);

        // Session ID.
        buf.push(self.session_id.len() as u8);
        buf.extend_from_slice(&self.session_id);

        // Cipher suites.
        let cs_len = (self.cipher_suites.len() * 2) as u16;
        buf.push((cs_len >> 8) as u8);
        buf.push((cs_len & 0xFF) as u8);
        for &cs in &self.cipher_suites {
            buf.push((cs >> 8) as u8);
            buf.push((cs & 0xFF) as u8);
        }

        // Compression methods.
        buf.push(self.compression_methods.len() as u8);
        for cm in &self.compression_methods {
            buf.push(cm.to_u8());
        }

        // Extensions.
        if !self.extensions.is_empty() {
            let ext_data: Vec<u8> = self.extensions.iter().flat_map(|e| e.serialize()).collect();
            let ext_len = ext_data.len() as u16;
            buf.push((ext_len >> 8) as u8);
            buf.push((ext_len & 0xFF) as u8);
            buf.extend_from_slice(&ext_data);
        }

        buf
    }

    /// Serialize as a complete handshake message (with header).
    pub fn serialize(&self) -> Vec<u8> {
        let body = self.serialize_body();
        let mut msg = Vec::with_capacity(4 + body.len());
        msg.push(HandshakeType::ClientHello.to_u8());
        let len = body.len();
        msg.push((len >> 16) as u8);
        msg.push((len >> 8) as u8);
        msg.push((len & 0xFF) as u8);
        msg.extend_from_slice(&body);
        msg
    }

    /// Parse from handshake message body bytes.
    pub fn parse_body(input: &[u8]) -> Result<Self, HandshakeError> {
        if input.len() < 38 {
            return Err(HandshakeError::TooShort {
                expected: 38,
                actual: input.len(),
            });
        }

        let client_version = TlsVersion::new(input[0], input[1]);
        let mut random = [0u8; 32];
        random.copy_from_slice(&input[2..34]);

        let mut offset = 34;

        // Session ID.
        if offset >= input.len() {
            return Err(HandshakeError::TooShort { expected: offset + 1, actual: input.len() });
        }
        let sid_len = input[offset] as usize;
        offset += 1;
        if offset + sid_len > input.len() {
            return Err(HandshakeError::TooShort { expected: offset + sid_len, actual: input.len() });
        }
        let session_id = input[offset..offset + sid_len].to_vec();
        offset += sid_len;

        // Cipher suites.
        if offset + 2 > input.len() {
            return Err(HandshakeError::TooShort { expected: offset + 2, actual: input.len() });
        }
        let cs_len = ((input[offset] as usize) << 8) | (input[offset + 1] as usize);
        offset += 2;
        if offset + cs_len > input.len() {
            return Err(HandshakeError::TooShort { expected: offset + cs_len, actual: input.len() });
        }
        let cs_count = cs_len / 2;
        let mut cipher_suites = Vec::with_capacity(cs_count);
        for _ in 0..cs_count {
            let cs = ((input[offset] as u16) << 8) | (input[offset + 1] as u16);
            cipher_suites.push(cs);
            offset += 2;
        }

        // Compression methods.
        if offset >= input.len() {
            return Err(HandshakeError::TooShort { expected: offset + 1, actual: input.len() });
        }
        let cm_len = input[offset] as usize;
        offset += 1;
        if offset + cm_len > input.len() {
            return Err(HandshakeError::TooShort { expected: offset + cm_len, actual: input.len() });
        }
        let mut compression_methods = Vec::with_capacity(cm_len);
        for i in 0..cm_len {
            compression_methods.push(CompressionMethod::from_u8(input[offset + i]));
        }
        offset += cm_len;

        // Extensions (optional).
        let mut extensions = Vec::new();
        if offset + 2 <= input.len() {
            let ext_data = &input[offset..];
            match crate::extensions::parse_extensions(ext_data) {
                Ok((_, exts)) => extensions = exts,
                Err(_) => {}
            }
        }

        Ok(ClientHello {
            client_version,
            random,
            session_id,
            cipher_suites,
            compression_methods,
            extensions,
        })
    }

    /// Get the effective TLS version, considering supported_versions extension.
    pub fn effective_version(&self) -> TlsVersion {
        for ext in &self.extensions {
            if let TlsExtension::SupportedVersions(versions) = ext {
                if let Some(max) = versions.iter().max() {
                    return *max;
                }
            }
        }
        self.client_version
    }

    /// Check if TLS_FALLBACK_SCSV is included.
    pub fn has_fallback_scsv(&self) -> bool {
        self.cipher_suites.contains(&crate::version::TLS_FALLBACK_SCSV)
    }

    /// Check if renegotiation info SCSV is included.
    pub fn has_renegotiation_scsv(&self) -> bool {
        self.cipher_suites.contains(&crate::version::TLS_EMPTY_RENEGOTIATION_INFO_SCSV)
    }

    /// Find a specific extension.
    pub fn find_extension(&self, type_id: u16) -> Option<&TlsExtension> {
        self.extensions.iter().find(|e| e.type_id() == type_id)
    }

    /// Get SNI hostnames if present.
    pub fn sni_hostnames(&self) -> Vec<&str> {
        for ext in &self.extensions {
            if let TlsExtension::ServerName(names) = ext {
                return names.iter().map(|s| s.as_str()).collect();
            }
        }
        Vec::new()
    }

    /// Get supported versions from extension.
    pub fn supported_versions(&self) -> Option<&[TlsVersion]> {
        for ext in &self.extensions {
            if let TlsExtension::SupportedVersions(versions) = ext {
                return Some(versions);
            }
        }
        None
    }
}

impl fmt::Display for ClientHello {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ClientHello(version={}, ciphers={}, exts={})",
            self.client_version,
            self.cipher_suites.len(),
            self.extensions.len()
        )
    }
}

// ---------------------------------------------------------------------------
// ServerHello
// ---------------------------------------------------------------------------

/// TLS ServerHello message (RFC 5246 §7.4.1.3, RFC 8446 §4.1.3).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServerHello {
    /// Selected protocol version (legacy in TLS 1.3).
    pub server_version: TlsVersion,
    /// 32 bytes of server random.
    pub random: [u8; 32],
    /// Session ID.
    pub session_id: Vec<u8>,
    /// Selected cipher suite.
    pub cipher_suite: u16,
    /// Selected compression method.
    pub compression_method: CompressionMethod,
    /// Extensions.
    pub extensions: Vec<TlsExtension>,
}

impl ServerHello {
    pub fn new(version: TlsVersion, random: [u8; 32], cipher_suite: u16) -> Self {
        Self {
            server_version: version,
            random,
            session_id: Vec::new(),
            cipher_suite,
            compression_method: CompressionMethod::Null,
            extensions: Vec::new(),
        }
    }

    /// Serialize to handshake message body.
    pub fn serialize_body(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(256);

        buf.push(self.server_version.major);
        buf.push(self.server_version.minor);
        buf.extend_from_slice(&self.random);

        buf.push(self.session_id.len() as u8);
        buf.extend_from_slice(&self.session_id);

        buf.push((self.cipher_suite >> 8) as u8);
        buf.push((self.cipher_suite & 0xFF) as u8);
        buf.push(self.compression_method.to_u8());

        if !self.extensions.is_empty() {
            let ext_data: Vec<u8> = self.extensions.iter().flat_map(|e| e.serialize()).collect();
            let ext_len = ext_data.len() as u16;
            buf.push((ext_len >> 8) as u8);
            buf.push((ext_len & 0xFF) as u8);
            buf.extend_from_slice(&ext_data);
        }

        buf
    }

    /// Serialize as complete handshake message.
    pub fn serialize(&self) -> Vec<u8> {
        let body = self.serialize_body();
        let mut msg = Vec::with_capacity(4 + body.len());
        msg.push(HandshakeType::ServerHello.to_u8());
        let len = body.len();
        msg.push((len >> 16) as u8);
        msg.push((len >> 8) as u8);
        msg.push((len & 0xFF) as u8);
        msg.extend_from_slice(&body);
        msg
    }

    /// Parse from handshake body bytes.
    pub fn parse_body(input: &[u8]) -> Result<Self, HandshakeError> {
        if input.len() < 38 {
            return Err(HandshakeError::TooShort {
                expected: 38,
                actual: input.len(),
            });
        }

        let server_version = TlsVersion::new(input[0], input[1]);
        let mut random = [0u8; 32];
        random.copy_from_slice(&input[2..34]);

        let mut offset = 34;

        let sid_len = input[offset] as usize;
        offset += 1;
        if offset + sid_len > input.len() {
            return Err(HandshakeError::TooShort { expected: offset + sid_len, actual: input.len() });
        }
        let session_id = input[offset..offset + sid_len].to_vec();
        offset += sid_len;

        if offset + 3 > input.len() {
            return Err(HandshakeError::TooShort { expected: offset + 3, actual: input.len() });
        }
        let cipher_suite = ((input[offset] as u16) << 8) | (input[offset + 1] as u16);
        offset += 2;
        let compression_method = CompressionMethod::from_u8(input[offset]);
        offset += 1;

        let mut extensions = Vec::new();
        if offset + 2 <= input.len() {
            let ext_data = &input[offset..];
            match crate::extensions::parse_extensions(ext_data) {
                Ok((_, exts)) => extensions = exts,
                Err(_) => {}
            }
        }

        Ok(ServerHello {
            server_version,
            random,
            session_id,
            cipher_suite,
            compression_method,
            extensions,
        })
    }

    /// Get the negotiated version, considering supported_versions extension.
    pub fn negotiated_version(&self) -> TlsVersion {
        for ext in &self.extensions {
            if let TlsExtension::SupportedVersions(versions) = ext {
                if let Some(&v) = versions.first() {
                    return v;
                }
            }
        }
        self.server_version
    }

    /// Check for downgrade sentinel in random.
    pub fn has_downgrade_sentinel(&self) -> bool {
        crate::extensions::check_anti_downgrade_sentinel(&self.random).is_some()
    }

    pub fn find_extension(&self, type_id: u16) -> Option<&TlsExtension> {
        self.extensions.iter().find(|e| e.type_id() == type_id)
    }
}

impl fmt::Display for ServerHello {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ServerHello(version={}, cipher=0x{:04X}, exts={})",
            self.server_version,
            self.cipher_suite,
            self.extensions.len()
        )
    }
}

// ---------------------------------------------------------------------------
// Other handshake messages
// ---------------------------------------------------------------------------

/// A server key exchange message (RFC 5246 §7.4.3).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServerKeyExchange {
    pub params: Vec<u8>,
}

/// A certificate message (RFC 5246 §7.4.2).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CertificateMessage {
    pub certificates: Vec<Vec<u8>>,
}

impl CertificateMessage {
    pub fn new(certificates: Vec<Vec<u8>>) -> Self {
        Self { certificates }
    }

    pub fn serialize_body(&self) -> Vec<u8> {
        let mut certs_data = Vec::new();
        for cert in &self.certificates {
            let len = cert.len();
            certs_data.push((len >> 16) as u8);
            certs_data.push((len >> 8) as u8);
            certs_data.push((len & 0xFF) as u8);
            certs_data.extend_from_slice(cert);
        }
        let total_len = certs_data.len();
        let mut buf = Vec::with_capacity(3 + total_len);
        buf.push((total_len >> 16) as u8);
        buf.push((total_len >> 8) as u8);
        buf.push((total_len & 0xFF) as u8);
        buf.extend_from_slice(&certs_data);
        buf
    }

    pub fn parse_body(input: &[u8]) -> Result<Self, HandshakeError> {
        if input.len() < 3 {
            return Err(HandshakeError::TooShort { expected: 3, actual: input.len() });
        }
        let total_len =
            ((input[0] as usize) << 16) | ((input[1] as usize) << 8) | (input[2] as usize);
        let mut offset = 3;
        let mut certificates = Vec::new();
        let end = (3 + total_len).min(input.len());
        while offset + 3 <= end {
            let cert_len =
                ((input[offset] as usize) << 16)
                    | ((input[offset + 1] as usize) << 8)
                    | (input[offset + 2] as usize);
            offset += 3;
            if offset + cert_len > end {
                break;
            }
            certificates.push(input[offset..offset + cert_len].to_vec());
            offset += cert_len;
        }
        Ok(Self { certificates })
    }

    pub fn certificate_count(&self) -> usize {
        self.certificates.len()
    }
}

/// A certificate request message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CertificateRequest {
    pub certificate_types: Vec<u8>,
    pub signature_algorithms: Vec<u16>,
    pub certificate_authorities: Vec<Vec<u8>>,
}

/// A Finished message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FinishedMessage {
    pub verify_data: Vec<u8>,
}

/// A NewSessionTicket message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NewSessionTicket {
    pub lifetime: u32,
    pub ticket: Vec<u8>,
    pub age_add: Option<u32>,
    pub nonce: Option<Vec<u8>>,
}

// ---------------------------------------------------------------------------
// Handshake message enum
// ---------------------------------------------------------------------------

/// A parsed TLS handshake message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HandshakeMessage {
    HelloRequest,
    ClientHello(ClientHello),
    ServerHello(ServerHello),
    NewSessionTicket(NewSessionTicket),
    EndOfEarlyData,
    EncryptedExtensions(Vec<TlsExtension>),
    Certificate(CertificateMessage),
    ServerKeyExchange(ServerKeyExchange),
    CertificateRequest(CertificateRequest),
    ServerHelloDone,
    CertificateVerify(Vec<u8>),
    ClientKeyExchange(Vec<u8>),
    Finished(FinishedMessage),
    KeyUpdate(u8),
    Unknown { msg_type: u8, data: Vec<u8> },
}

impl HandshakeMessage {
    /// Returns the handshake message type.
    pub fn msg_type(&self) -> HandshakeType {
        match self {
            Self::HelloRequest => HandshakeType::HelloRequest,
            Self::ClientHello(_) => HandshakeType::ClientHello,
            Self::ServerHello(_) => HandshakeType::ServerHello,
            Self::NewSessionTicket(_) => HandshakeType::NewSessionTicket,
            Self::EndOfEarlyData => HandshakeType::EndOfEarlyData,
            Self::EncryptedExtensions(_) => HandshakeType::EncryptedExtensions,
            Self::Certificate(_) => HandshakeType::Certificate,
            Self::ServerKeyExchange(_) => HandshakeType::ServerKeyExchange,
            Self::CertificateRequest(_) => HandshakeType::CertificateRequest,
            Self::ServerHelloDone => HandshakeType::ServerHelloDone,
            Self::CertificateVerify(_) => HandshakeType::CertificateVerify,
            Self::ClientKeyExchange(_) => HandshakeType::ClientKeyExchange,
            Self::Finished(_) => HandshakeType::Finished,
            Self::KeyUpdate(_) => HandshakeType::KeyUpdate,
            Self::Unknown { msg_type, .. } => {
                HandshakeType::from_u8(*msg_type).unwrap_or(HandshakeType::HelloRequest)
            }
        }
    }

    /// Serialize to the complete handshake message bytes (type + length + body).
    pub fn serialize(&self) -> Vec<u8> {
        match self {
            Self::ClientHello(ch) => ch.serialize(),
            Self::ServerHello(sh) => sh.serialize(),
            Self::HelloRequest => {
                vec![HandshakeType::HelloRequest.to_u8(), 0, 0, 0]
            }
            Self::ServerHelloDone => {
                vec![HandshakeType::ServerHelloDone.to_u8(), 0, 0, 0]
            }
            Self::Certificate(cert) => {
                let body = cert.serialize_body();
                let mut msg = Vec::with_capacity(4 + body.len());
                msg.push(HandshakeType::Certificate.to_u8());
                let len = body.len();
                msg.push((len >> 16) as u8);
                msg.push((len >> 8) as u8);
                msg.push((len & 0xFF) as u8);
                msg.extend_from_slice(&body);
                msg
            }
            Self::Finished(fin) => {
                let mut msg = Vec::with_capacity(4 + fin.verify_data.len());
                msg.push(HandshakeType::Finished.to_u8());
                let len = fin.verify_data.len();
                msg.push((len >> 16) as u8);
                msg.push((len >> 8) as u8);
                msg.push((len & 0xFF) as u8);
                msg.extend_from_slice(&fin.verify_data);
                msg
            }
            Self::ServerKeyExchange(ske) => {
                let mut msg = Vec::with_capacity(4 + ske.params.len());
                msg.push(HandshakeType::ServerKeyExchange.to_u8());
                let len = ske.params.len();
                msg.push((len >> 16) as u8);
                msg.push((len >> 8) as u8);
                msg.push((len & 0xFF) as u8);
                msg.extend_from_slice(&ske.params);
                msg
            }
            Self::ClientKeyExchange(data) => {
                let mut msg = Vec::with_capacity(4 + data.len());
                msg.push(HandshakeType::ClientKeyExchange.to_u8());
                let len = data.len();
                msg.push((len >> 16) as u8);
                msg.push((len >> 8) as u8);
                msg.push((len & 0xFF) as u8);
                msg.extend_from_slice(data);
                msg
            }
            Self::CertificateVerify(data) => {
                let mut msg = Vec::with_capacity(4 + data.len());
                msg.push(HandshakeType::CertificateVerify.to_u8());
                let len = data.len();
                msg.push((len >> 16) as u8);
                msg.push((len >> 8) as u8);
                msg.push((len & 0xFF) as u8);
                msg.extend_from_slice(data);
                msg
            }
            Self::EncryptedExtensions(exts) => {
                let ext_data: Vec<u8> = exts.iter().flat_map(|e| e.serialize()).collect();
                let mut body = Vec::with_capacity(2 + ext_data.len());
                let ext_len = ext_data.len() as u16;
                body.push((ext_len >> 8) as u8);
                body.push((ext_len & 0xFF) as u8);
                body.extend_from_slice(&ext_data);
                let mut msg = Vec::with_capacity(4 + body.len());
                msg.push(HandshakeType::EncryptedExtensions.to_u8());
                let len = body.len();
                msg.push((len >> 16) as u8);
                msg.push((len >> 8) as u8);
                msg.push((len & 0xFF) as u8);
                msg.extend_from_slice(&body);
                msg
            }
            Self::EndOfEarlyData => {
                vec![HandshakeType::EndOfEarlyData.to_u8(), 0, 0, 0]
            }
            Self::KeyUpdate(mode) => {
                vec![HandshakeType::KeyUpdate.to_u8(), 0, 0, 1, *mode]
            }
            Self::NewSessionTicket(nst) => {
                let mut body = Vec::new();
                body.push((nst.lifetime >> 24) as u8);
                body.push((nst.lifetime >> 16) as u8);
                body.push((nst.lifetime >> 8) as u8);
                body.push((nst.lifetime & 0xFF) as u8);
                if let Some(age_add) = nst.age_add {
                    body.push((age_add >> 24) as u8);
                    body.push((age_add >> 16) as u8);
                    body.push((age_add >> 8) as u8);
                    body.push((age_add & 0xFF) as u8);
                }
                if let Some(ref nonce) = nst.nonce {
                    body.push(nonce.len() as u8);
                    body.extend_from_slice(nonce);
                }
                let tlen = nst.ticket.len() as u16;
                body.push((tlen >> 8) as u8);
                body.push((tlen & 0xFF) as u8);
                body.extend_from_slice(&nst.ticket);
                let mut msg = Vec::with_capacity(4 + body.len());
                msg.push(HandshakeType::NewSessionTicket.to_u8());
                let len = body.len();
                msg.push((len >> 16) as u8);
                msg.push((len >> 8) as u8);
                msg.push((len & 0xFF) as u8);
                msg.extend_from_slice(&body);
                msg
            }
            Self::CertificateRequest(cr) => {
                let mut body = Vec::new();
                body.push(cr.certificate_types.len() as u8);
                body.extend_from_slice(&cr.certificate_types);
                let sa_len = (cr.signature_algorithms.len() * 2) as u16;
                body.push((sa_len >> 8) as u8);
                body.push((sa_len & 0xFF) as u8);
                for &sa in &cr.signature_algorithms {
                    body.push((sa >> 8) as u8);
                    body.push((sa & 0xFF) as u8);
                }
                let mut ca_data = Vec::new();
                for ca in &cr.certificate_authorities {
                    let len = ca.len() as u16;
                    ca_data.push((len >> 8) as u8);
                    ca_data.push((len & 0xFF) as u8);
                    ca_data.extend_from_slice(ca);
                }
                let ca_total = ca_data.len() as u16;
                body.push((ca_total >> 8) as u8);
                body.push((ca_total & 0xFF) as u8);
                body.extend_from_slice(&ca_data);
                let mut msg = Vec::with_capacity(4 + body.len());
                msg.push(HandshakeType::CertificateRequest.to_u8());
                let len = body.len();
                msg.push((len >> 16) as u8);
                msg.push((len >> 8) as u8);
                msg.push((len & 0xFF) as u8);
                msg.extend_from_slice(&body);
                msg
            }
            Self::Unknown { msg_type, data } => {
                let mut msg = Vec::with_capacity(4 + data.len());
                msg.push(*msg_type);
                let len = data.len();
                msg.push((len >> 16) as u8);
                msg.push((len >> 8) as u8);
                msg.push((len & 0xFF) as u8);
                msg.extend_from_slice(data);
                msg
            }
        }
    }

    /// Parse a handshake message from raw bytes (type + length + body).
    pub fn parse(input: &[u8]) -> Result<(Self, usize), HandshakeError> {
        if input.len() < 4 {
            return Err(HandshakeError::TooShort {
                expected: 4,
                actual: input.len(),
            });
        }

        let msg_type_byte = input[0];
        let body_len =
            ((input[1] as usize) << 16) | ((input[2] as usize) << 8) | (input[3] as usize);
        let total_len = 4 + body_len;

        if input.len() < total_len {
            return Err(HandshakeError::TooShort {
                expected: total_len,
                actual: input.len(),
            });
        }

        let body = &input[4..total_len];

        let msg = match HandshakeType::from_u8(msg_type_byte) {
            Some(HandshakeType::HelloRequest) => HandshakeMessage::HelloRequest,
            Some(HandshakeType::ClientHello) => {
                HandshakeMessage::ClientHello(ClientHello::parse_body(body)?)
            }
            Some(HandshakeType::ServerHello) => {
                HandshakeMessage::ServerHello(ServerHello::parse_body(body)?)
            }
            Some(HandshakeType::Certificate) => {
                HandshakeMessage::Certificate(CertificateMessage::parse_body(body)?)
            }
            Some(HandshakeType::ServerHelloDone) => HandshakeMessage::ServerHelloDone,
            Some(HandshakeType::ServerKeyExchange) => {
                HandshakeMessage::ServerKeyExchange(ServerKeyExchange {
                    params: body.to_vec(),
                })
            }
            Some(HandshakeType::ClientKeyExchange) => {
                HandshakeMessage::ClientKeyExchange(body.to_vec())
            }
            Some(HandshakeType::CertificateVerify) => {
                HandshakeMessage::CertificateVerify(body.to_vec())
            }
            Some(HandshakeType::Finished) => HandshakeMessage::Finished(FinishedMessage {
                verify_data: body.to_vec(),
            }),
            Some(HandshakeType::EndOfEarlyData) => HandshakeMessage::EndOfEarlyData,
            Some(HandshakeType::KeyUpdate) => {
                let mode = if body.is_empty() { 0 } else { body[0] };
                HandshakeMessage::KeyUpdate(mode)
            }
            _ => HandshakeMessage::Unknown {
                msg_type: msg_type_byte,
                data: body.to_vec(),
            },
        };

        Ok((msg, total_len))
    }
}

impl fmt::Display for HandshakeMessage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ClientHello(ch) => write!(f, "{}", ch),
            Self::ServerHello(sh) => write!(f, "{}", sh),
            Self::Certificate(cert) => {
                write!(f, "Certificate({} certs)", cert.certificate_count())
            }
            Self::Finished(fin) => {
                write!(f, "Finished({} bytes)", fin.verify_data.len())
            }
            _ => write!(f, "{:?}", self.msg_type()),
        }
    }
}

// ---------------------------------------------------------------------------
// Handshake errors
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, thiserror::Error)]
pub enum HandshakeError {
    #[error("message too short: expected {expected}, got {actual}")]
    TooShort { expected: usize, actual: usize },

    #[error("invalid handshake type: 0x{0:02X}")]
    InvalidType(u8),

    #[error("body length mismatch: header says {header_len}, body is {body_len}")]
    LengthMismatch { header_len: usize, body_len: usize },

    #[error("parse error: {0}")]
    ParseError(String),
}

// ---------------------------------------------------------------------------
// Handshake hash computation
// ---------------------------------------------------------------------------

/// Accumulates handshake messages for hash computation.
#[derive(Debug, Clone)]
pub struct HandshakeHash {
    buffer: Vec<u8>,
}

impl HandshakeHash {
    pub fn new() -> Self {
        Self { buffer: Vec::new() }
    }

    /// Add a raw handshake message (including type + length header).
    pub fn update(&mut self, data: &[u8]) {
        self.buffer.extend_from_slice(data);
    }

    /// Add a HandshakeMessage.
    pub fn update_message(&mut self, msg: &HandshakeMessage) {
        let bytes = msg.serialize();
        self.buffer.extend_from_slice(&bytes);
    }

    /// Compute SHA-256 hash of all accumulated messages.
    pub fn finish_sha256(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(&self.buffer);
        let result = hasher.finalize();
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&result);
        hash
    }

    /// Compute SHA-256 hash and return as hex string.
    pub fn finish_sha256_hex(&self) -> String {
        hex::encode(self.finish_sha256())
    }

    /// Get the current transcript length.
    pub fn transcript_length(&self) -> usize {
        self.buffer.len()
    }

    /// Reset the hash accumulator.
    pub fn reset(&mut self) {
        self.buffer.clear();
    }

    /// Get a snapshot of the current hash state.
    pub fn current_hash_sha256(&self) -> [u8; 32] {
        self.finish_sha256()
    }
}

impl Default for HandshakeHash {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extensions::TlsExtension;

    #[test]
    fn test_client_hello_serialize_parse_roundtrip() {
        let mut ch = ClientHello::new(TlsVersion::TLS1_2, [0x42u8; 32]);
        ch.cipher_suites = vec![0xC02F, 0x009E, 0x0035];
        ch.session_id = vec![0xAB; 16];
        ch.extensions.push(TlsExtension::ExtendedMasterSecret);
        ch.extensions.push(TlsExtension::RenegotiationInfo(vec![]));

        let bytes = ch.serialize();
        let (parsed_msg, consumed) = HandshakeMessage::parse(&bytes).unwrap();
        assert_eq!(consumed, bytes.len());

        match parsed_msg {
            HandshakeMessage::ClientHello(parsed_ch) => {
                assert_eq!(parsed_ch.client_version, TlsVersion::TLS1_2);
                assert_eq!(parsed_ch.random, [0x42u8; 32]);
                assert_eq!(parsed_ch.cipher_suites, vec![0xC02F, 0x009E, 0x0035]);
                assert_eq!(parsed_ch.session_id, vec![0xAB; 16]);
                assert_eq!(parsed_ch.extensions.len(), 2);
            }
            _ => panic!("Expected ClientHello"),
        }
    }

    #[test]
    fn test_server_hello_serialize_parse_roundtrip() {
        let mut sh = ServerHello::new(TlsVersion::TLS1_2, [0x55u8; 32], 0xC02F);
        sh.session_id = vec![0xBB; 32];
        sh.extensions.push(TlsExtension::ExtendedMasterSecret);

        let bytes = sh.serialize();
        let (parsed_msg, _) = HandshakeMessage::parse(&bytes).unwrap();

        match parsed_msg {
            HandshakeMessage::ServerHello(parsed_sh) => {
                assert_eq!(parsed_sh.server_version, TlsVersion::TLS1_2);
                assert_eq!(parsed_sh.cipher_suite, 0xC02F);
                assert_eq!(parsed_sh.session_id, vec![0xBB; 32]);
            }
            _ => panic!("Expected ServerHello"),
        }
    }

    #[test]
    fn test_server_hello_done() {
        let msg = HandshakeMessage::ServerHelloDone;
        let bytes = msg.serialize();
        assert_eq!(bytes, vec![14, 0, 0, 0]);
        let (parsed, _) = HandshakeMessage::parse(&bytes).unwrap();
        assert!(matches!(parsed, HandshakeMessage::ServerHelloDone));
    }

    #[test]
    fn test_certificate_message() {
        let cert = CertificateMessage::new(vec![vec![0x30; 100], vec![0x31; 50]]);
        let body = cert.serialize_body();
        let parsed = CertificateMessage::parse_body(&body).unwrap();
        assert_eq!(parsed.certificates.len(), 2);
        assert_eq!(parsed.certificates[0].len(), 100);
        assert_eq!(parsed.certificates[1].len(), 50);
    }

    #[test]
    fn test_finished_message() {
        let fin = HandshakeMessage::Finished(FinishedMessage {
            verify_data: vec![0xDE; 12],
        });
        let bytes = fin.serialize();
        let (parsed, _) = HandshakeMessage::parse(&bytes).unwrap();
        match parsed {
            HandshakeMessage::Finished(f) => {
                assert_eq!(f.verify_data, vec![0xDE; 12]);
            }
            _ => panic!("Expected Finished"),
        }
    }

    #[test]
    fn test_handshake_type_from_u8() {
        assert_eq!(HandshakeType::from_u8(1), Some(HandshakeType::ClientHello));
        assert_eq!(HandshakeType::from_u8(2), Some(HandshakeType::ServerHello));
        assert_eq!(HandshakeType::from_u8(20), Some(HandshakeType::Finished));
        assert_eq!(HandshakeType::from_u8(99), None);
    }

    #[test]
    fn test_effective_version() {
        let mut ch = ClientHello::new(TlsVersion::TLS1_2, [0u8; 32]);
        assert_eq!(ch.effective_version(), TlsVersion::TLS1_2);

        ch.extensions.push(TlsExtension::SupportedVersions(vec![
            TlsVersion::TLS1_3,
            TlsVersion::TLS1_2,
        ]));
        assert_eq!(ch.effective_version(), TlsVersion::TLS1_3);
    }

    #[test]
    fn test_has_fallback_scsv() {
        let mut ch = ClientHello::new(TlsVersion::TLS1_2, [0u8; 32]);
        ch.cipher_suites = vec![0xC02F, 0x0035];
        assert!(!ch.has_fallback_scsv());

        ch.cipher_suites.push(crate::version::TLS_FALLBACK_SCSV);
        assert!(ch.has_fallback_scsv());
    }

    #[test]
    fn test_handshake_hash() {
        let mut hash = HandshakeHash::new();
        let ch = ClientHello::new(TlsVersion::TLS1_2, [0x42u8; 32]);
        hash.update_message(&HandshakeMessage::ClientHello(ch));
        let h1 = hash.finish_sha256();

        assert!(hash.transcript_length() > 0);
        assert_ne!(h1, [0u8; 32]);

        let hex_str = hash.finish_sha256_hex();
        assert_eq!(hex_str.len(), 64);
    }

    #[test]
    fn test_compression_method() {
        assert_eq!(CompressionMethod::from_u8(0), CompressionMethod::Null);
        assert_eq!(CompressionMethod::from_u8(1), CompressionMethod::Deflate);
        assert_eq!(CompressionMethod::Null.to_u8(), 0);
    }

    #[test]
    fn test_sni_hostnames() {
        let mut ch = ClientHello::new(TlsVersion::TLS1_2, [0u8; 32]);
        ch.extensions.push(TlsExtension::ServerName(vec![
            "example.com".to_string(),
        ]));
        assert_eq!(ch.sni_hostnames(), vec!["example.com"]);
    }

    #[test]
    fn test_parse_too_short() {
        let err = HandshakeMessage::parse(&[0x01]).unwrap_err();
        match err {
            HandshakeError::TooShort { .. } => {}
            _ => panic!("Expected TooShort"),
        }
    }

    #[test]
    fn test_server_hello_negotiated_version() {
        let mut sh = ServerHello::new(TlsVersion::TLS1_2, [0u8; 32], 0x1301);
        assert_eq!(sh.negotiated_version(), TlsVersion::TLS1_2);

        sh.extensions.push(TlsExtension::SupportedVersions(vec![TlsVersion::TLS1_3]));
        assert_eq!(sh.negotiated_version(), TlsVersion::TLS1_3);
    }
}
