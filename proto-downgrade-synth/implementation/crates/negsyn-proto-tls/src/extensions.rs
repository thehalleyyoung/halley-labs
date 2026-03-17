//! TLS extensions (RFC 5246, RFC 6066, RFC 8446, and others).
//!
//! Parsing, serialization, and negotiation logic for all common TLS extensions
//! including SNI, supported_versions, key_share, signature_algorithms, ALPN,
//! and the TLS 1.3 anti-downgrade sentinel mechanism.

use crate::version::TlsVersion;
use nom::{
    bytes::complete::take,
    multi::count,
    number::complete::{be_u16, be_u8},
    IResult,
};
use serde::{Deserialize, Serialize};
use std::fmt;

// ---------------------------------------------------------------------------
// Extension type IDs (IANA registry)
// ---------------------------------------------------------------------------

pub const EXT_SERVER_NAME: u16 = 0;
pub const EXT_MAX_FRAGMENT_LENGTH: u16 = 1;
pub const EXT_CLIENT_CERTIFICATE_URL: u16 = 2;
pub const EXT_TRUSTED_CA_KEYS: u16 = 3;
pub const EXT_TRUNCATED_HMAC: u16 = 4;
pub const EXT_STATUS_REQUEST: u16 = 5;
pub const EXT_SUPPORTED_GROUPS: u16 = 10;
pub const EXT_EC_POINT_FORMATS: u16 = 11;
pub const EXT_SIGNATURE_ALGORITHMS: u16 = 13;
pub const EXT_USE_SRTP: u16 = 14;
pub const EXT_HEARTBEAT: u16 = 15;
pub const EXT_ALPN: u16 = 16;
pub const EXT_SCT: u16 = 18;
pub const EXT_CLIENT_CERTIFICATE_TYPE: u16 = 19;
pub const EXT_SERVER_CERTIFICATE_TYPE: u16 = 20;
pub const EXT_PADDING: u16 = 21;
pub const EXT_ENCRYPT_THEN_MAC: u16 = 22;
pub const EXT_EXTENDED_MASTER_SECRET: u16 = 23;
pub const EXT_SESSION_TICKET: u16 = 35;
pub const EXT_PRE_SHARED_KEY: u16 = 41;
pub const EXT_EARLY_DATA: u16 = 42;
pub const EXT_SUPPORTED_VERSIONS: u16 = 43;
pub const EXT_COOKIE: u16 = 44;
pub const EXT_PSK_KEY_EXCHANGE_MODES: u16 = 45;
pub const EXT_CERTIFICATE_AUTHORITIES: u16 = 47;
pub const EXT_OID_FILTERS: u16 = 48;
pub const EXT_POST_HANDSHAKE_AUTH: u16 = 49;
pub const EXT_SIGNATURE_ALGORITHMS_CERT: u16 = 50;
pub const EXT_KEY_SHARE: u16 = 51;
pub const EXT_RENEGOTIATION_INFO: u16 = 0xFF01;

// ---------------------------------------------------------------------------
// Named groups (for supported_groups extension)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NamedGroup {
    Secp256r1,
    Secp384r1,
    Secp521r1,
    X25519,
    X448,
    Ffdhe2048,
    Ffdhe3072,
    Ffdhe4096,
    Ffdhe6144,
    Ffdhe8192,
    Unknown(u16),
}

impl NamedGroup {
    pub fn from_u16(val: u16) -> Self {
        match val {
            0x0017 => Self::Secp256r1,
            0x0018 => Self::Secp384r1,
            0x0019 => Self::Secp521r1,
            0x001D => Self::X25519,
            0x001E => Self::X448,
            0x0100 => Self::Ffdhe2048,
            0x0101 => Self::Ffdhe3072,
            0x0102 => Self::Ffdhe4096,
            0x0103 => Self::Ffdhe6144,
            0x0104 => Self::Ffdhe8192,
            v => Self::Unknown(v),
        }
    }

    pub fn to_u16(self) -> u16 {
        match self {
            Self::Secp256r1 => 0x0017,
            Self::Secp384r1 => 0x0018,
            Self::Secp521r1 => 0x0019,
            Self::X25519 => 0x001D,
            Self::X448 => 0x001E,
            Self::Ffdhe2048 => 0x0100,
            Self::Ffdhe3072 => 0x0101,
            Self::Ffdhe4096 => 0x0102,
            Self::Ffdhe6144 => 0x0103,
            Self::Ffdhe8192 => 0x0104,
            Self::Unknown(v) => v,
        }
    }

    pub fn is_ecdhe(&self) -> bool {
        matches!(
            self,
            Self::Secp256r1 | Self::Secp384r1 | Self::Secp521r1 | Self::X25519 | Self::X448
        )
    }

    pub fn is_ffdhe(&self) -> bool {
        matches!(
            self,
            Self::Ffdhe2048
                | Self::Ffdhe3072
                | Self::Ffdhe4096
                | Self::Ffdhe6144
                | Self::Ffdhe8192
        )
    }

    pub fn key_bits(&self) -> u32 {
        match self {
            Self::Secp256r1 | Self::X25519 => 256,
            Self::Secp384r1 => 384,
            Self::Secp521r1 | Self::X448 => 521,
            Self::Ffdhe2048 => 2048,
            Self::Ffdhe3072 => 3072,
            Self::Ffdhe4096 => 4096,
            Self::Ffdhe6144 => 6144,
            Self::Ffdhe8192 => 8192,
            Self::Unknown(_) => 0,
        }
    }
}

impl fmt::Display for NamedGroup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Secp256r1 => write!(f, "secp256r1"),
            Self::Secp384r1 => write!(f, "secp384r1"),
            Self::Secp521r1 => write!(f, "secp521r1"),
            Self::X25519 => write!(f, "x25519"),
            Self::X448 => write!(f, "x448"),
            Self::Ffdhe2048 => write!(f, "ffdhe2048"),
            Self::Ffdhe3072 => write!(f, "ffdhe3072"),
            Self::Ffdhe4096 => write!(f, "ffdhe4096"),
            Self::Ffdhe6144 => write!(f, "ffdhe6144"),
            Self::Ffdhe8192 => write!(f, "ffdhe8192"),
            Self::Unknown(v) => write!(f, "unknown(0x{:04X})", v),
        }
    }
}

// ---------------------------------------------------------------------------
// Signature algorithms
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SignatureScheme {
    RsaPkcs1Sha256,
    RsaPkcs1Sha384,
    RsaPkcs1Sha512,
    EcdsaSecp256r1Sha256,
    EcdsaSecp384r1Sha384,
    EcdsaSecp521r1Sha512,
    RsaPssRsaeSha256,
    RsaPssRsaeSha384,
    RsaPssRsaeSha512,
    Ed25519,
    Ed448,
    RsaPkcs1Sha1,
    EcdsaSha1,
    Unknown(u16),
}

impl SignatureScheme {
    pub fn from_u16(val: u16) -> Self {
        match val {
            0x0401 => Self::RsaPkcs1Sha256,
            0x0501 => Self::RsaPkcs1Sha384,
            0x0601 => Self::RsaPkcs1Sha512,
            0x0403 => Self::EcdsaSecp256r1Sha256,
            0x0503 => Self::EcdsaSecp384r1Sha384,
            0x0603 => Self::EcdsaSecp521r1Sha512,
            0x0804 => Self::RsaPssRsaeSha256,
            0x0805 => Self::RsaPssRsaeSha384,
            0x0806 => Self::RsaPssRsaeSha512,
            0x0807 => Self::Ed25519,
            0x0808 => Self::Ed448,
            0x0201 => Self::RsaPkcs1Sha1,
            0x0203 => Self::EcdsaSha1,
            v => Self::Unknown(v),
        }
    }

    pub fn to_u16(self) -> u16 {
        match self {
            Self::RsaPkcs1Sha256 => 0x0401,
            Self::RsaPkcs1Sha384 => 0x0501,
            Self::RsaPkcs1Sha512 => 0x0601,
            Self::EcdsaSecp256r1Sha256 => 0x0403,
            Self::EcdsaSecp384r1Sha384 => 0x0503,
            Self::EcdsaSecp521r1Sha512 => 0x0603,
            Self::RsaPssRsaeSha256 => 0x0804,
            Self::RsaPssRsaeSha384 => 0x0805,
            Self::RsaPssRsaeSha512 => 0x0806,
            Self::Ed25519 => 0x0807,
            Self::Ed448 => 0x0808,
            Self::RsaPkcs1Sha1 => 0x0201,
            Self::EcdsaSha1 => 0x0203,
            Self::Unknown(v) => v,
        }
    }

    pub fn is_legacy(&self) -> bool {
        matches!(self, Self::RsaPkcs1Sha1 | Self::EcdsaSha1)
    }
}

impl fmt::Display for SignatureScheme {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

// ---------------------------------------------------------------------------
// EC point formats
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ECPointFormat {
    Uncompressed,
    AnsiX962CompressedPrime,
    AnsiX962CompressedChar2,
    Unknown(u8),
}

impl ECPointFormat {
    pub fn from_u8(val: u8) -> Self {
        match val {
            0 => Self::Uncompressed,
            1 => Self::AnsiX962CompressedPrime,
            2 => Self::AnsiX962CompressedChar2,
            v => Self::Unknown(v),
        }
    }

    pub fn to_u8(self) -> u8 {
        match self {
            Self::Uncompressed => 0,
            Self::AnsiX962CompressedPrime => 1,
            Self::AnsiX962CompressedChar2 => 2,
            Self::Unknown(v) => v,
        }
    }
}

// ---------------------------------------------------------------------------
// PSK key exchange modes
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PskKeyExchangeMode {
    PskKe,
    PskDheKe,
    Unknown(u8),
}

impl PskKeyExchangeMode {
    pub fn from_u8(val: u8) -> Self {
        match val {
            0 => Self::PskKe,
            1 => Self::PskDheKe,
            v => Self::Unknown(v),
        }
    }

    pub fn to_u8(self) -> u8 {
        match self {
            Self::PskKe => 0,
            Self::PskDheKe => 1,
            Self::Unknown(v) => v,
        }
    }
}

// ---------------------------------------------------------------------------
// Max fragment length
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MaxFragmentLength {
    Bits9,   // 512
    Bits10,  // 1024
    Bits11,  // 2048
    Bits12,  // 4096
}

impl MaxFragmentLength {
    pub fn from_u8(val: u8) -> Option<Self> {
        match val {
            1 => Some(Self::Bits9),
            2 => Some(Self::Bits10),
            3 => Some(Self::Bits11),
            4 => Some(Self::Bits12),
            _ => None,
        }
    }

    pub fn to_u8(self) -> u8 {
        match self {
            Self::Bits9 => 1,
            Self::Bits10 => 2,
            Self::Bits11 => 3,
            Self::Bits12 => 4,
        }
    }

    pub fn max_bytes(&self) -> usize {
        match self {
            Self::Bits9 => 512,
            Self::Bits10 => 1024,
            Self::Bits11 => 2048,
            Self::Bits12 => 4096,
        }
    }
}

// ---------------------------------------------------------------------------
// Key share entry
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct KeyShareEntry {
    pub group: NamedGroup,
    pub key_exchange: Vec<u8>,
}

impl KeyShareEntry {
    pub fn new(group: NamedGroup, key_exchange: Vec<u8>) -> Self {
        Self {
            group,
            key_exchange,
        }
    }

    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        let group_id = self.group.to_u16();
        buf.push((group_id >> 8) as u8);
        buf.push((group_id & 0xFF) as u8);
        let len = self.key_exchange.len() as u16;
        buf.push((len >> 8) as u8);
        buf.push((len & 0xFF) as u8);
        buf.extend_from_slice(&self.key_exchange);
        buf
    }

    pub fn parse(input: &[u8]) -> IResult<&[u8], Self> {
        let (input, group_id) = be_u16(input)?;
        let (input, ke_len) = be_u16(input)?;
        let (input, ke_data) = take(ke_len as usize)(input)?;
        Ok((
            input,
            KeyShareEntry {
                group: NamedGroup::from_u16(group_id),
                key_exchange: ke_data.to_vec(),
            },
        ))
    }
}

// ---------------------------------------------------------------------------
// TLS Extension enum
// ---------------------------------------------------------------------------

/// A parsed TLS extension.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TlsExtension {
    /// Server Name Indication (SNI).
    ServerName(Vec<String>),

    /// Maximum fragment length negotiation.
    MaxFragmentLength(MaxFragmentLength),

    /// OCSP status request.
    StatusRequest,

    /// Supported elliptic curve groups.
    SupportedGroups(Vec<NamedGroup>),

    /// EC point format negotiation.
    ECPointFormats(Vec<ECPointFormat>),

    /// Signature algorithms the client supports.
    SignatureAlgorithms(Vec<SignatureScheme>),

    /// Signature algorithms for certificates (TLS 1.3).
    SignatureAlgorithmsCert(Vec<SignatureScheme>),

    /// Application-Layer Protocol Negotiation.
    Alpn(Vec<String>),

    /// Signed Certificate Timestamp (SCT).
    SignedCertificateTimestamp(Vec<u8>),

    /// Padding extension.
    Padding(u16),

    /// Encrypt-then-MAC (RFC 7366).
    EncryptThenMac,

    /// Extended Master Secret (RFC 7627).
    ExtendedMasterSecret,

    /// Session ticket (RFC 5077).
    SessionTicket(Vec<u8>),

    /// Supported versions (TLS 1.3).
    SupportedVersions(Vec<TlsVersion>),

    /// PSK key exchange modes (TLS 1.3).
    PskKeyExchangeModes(Vec<PskKeyExchangeMode>),

    /// Key share (TLS 1.3).
    KeyShareClientHello(Vec<KeyShareEntry>),

    /// Key share (ServerHello response).
    KeyShareServerHello(KeyShareEntry),

    /// Key share (HelloRetryRequest).
    KeyShareHelloRetryRequest(NamedGroup),

    /// Pre-shared key (TLS 1.3).
    PreSharedKey(Vec<u8>),

    /// Early data indication (TLS 1.3 0-RTT).
    EarlyData(Option<u32>),

    /// Cookie (TLS 1.3).
    Cookie(Vec<u8>),

    /// Post-handshake authentication (TLS 1.3).
    PostHandshakeAuth,

    /// Renegotiation info (RFC 5746).
    RenegotiationInfo(Vec<u8>),

    /// Certificate authorities (TLS 1.3).
    CertificateAuthorities(Vec<Vec<u8>>),

    /// Unknown extension with raw data.
    Unknown { type_id: u16, data: Vec<u8> },
}

impl TlsExtension {
    /// Returns the IANA extension type ID.
    pub fn type_id(&self) -> u16 {
        match self {
            Self::ServerName(_) => EXT_SERVER_NAME,
            Self::MaxFragmentLength(_) => EXT_MAX_FRAGMENT_LENGTH,
            Self::StatusRequest => EXT_STATUS_REQUEST,
            Self::SupportedGroups(_) => EXT_SUPPORTED_GROUPS,
            Self::ECPointFormats(_) => EXT_EC_POINT_FORMATS,
            Self::SignatureAlgorithms(_) => EXT_SIGNATURE_ALGORITHMS,
            Self::SignatureAlgorithmsCert(_) => EXT_SIGNATURE_ALGORITHMS_CERT,
            Self::Alpn(_) => EXT_ALPN,
            Self::SignedCertificateTimestamp(_) => EXT_SCT,
            Self::Padding(_) => EXT_PADDING,
            Self::EncryptThenMac => EXT_ENCRYPT_THEN_MAC,
            Self::ExtendedMasterSecret => EXT_EXTENDED_MASTER_SECRET,
            Self::SessionTicket(_) => EXT_SESSION_TICKET,
            Self::SupportedVersions(_) => EXT_SUPPORTED_VERSIONS,
            Self::PskKeyExchangeModes(_) => EXT_PSK_KEY_EXCHANGE_MODES,
            Self::KeyShareClientHello(_) => EXT_KEY_SHARE,
            Self::KeyShareServerHello(_) => EXT_KEY_SHARE,
            Self::KeyShareHelloRetryRequest(_) => EXT_KEY_SHARE,
            Self::PreSharedKey(_) => EXT_PRE_SHARED_KEY,
            Self::EarlyData(_) => EXT_EARLY_DATA,
            Self::Cookie(_) => EXT_COOKIE,
            Self::PostHandshakeAuth => EXT_POST_HANDSHAKE_AUTH,
            Self::RenegotiationInfo(_) => EXT_RENEGOTIATION_INFO,
            Self::CertificateAuthorities(_) => EXT_CERTIFICATE_AUTHORITIES,
            Self::Unknown { type_id, .. } => *type_id,
        }
    }

    /// Serialize the extension to bytes (type + length + data).
    pub fn serialize(&self) -> Vec<u8> {
        let data = self.serialize_data();
        let mut buf = Vec::with_capacity(4 + data.len());
        let type_id = self.type_id();
        buf.push((type_id >> 8) as u8);
        buf.push((type_id & 0xFF) as u8);
        let len = data.len() as u16;
        buf.push((len >> 8) as u8);
        buf.push((len & 0xFF) as u8);
        buf.extend_from_slice(&data);
        buf
    }

    /// Serialize just the extension data (without type and length header).
    pub fn serialize_data(&self) -> Vec<u8> {
        match self {
            Self::ServerName(names) => {
                let mut list_data = Vec::new();
                for name in names {
                    list_data.push(0); // host_name type
                    let name_bytes = name.as_bytes();
                    let nlen = name_bytes.len() as u16;
                    list_data.push((nlen >> 8) as u8);
                    list_data.push((nlen & 0xFF) as u8);
                    list_data.extend_from_slice(name_bytes);
                }
                let list_len = list_data.len() as u16;
                let mut buf = Vec::new();
                buf.push((list_len >> 8) as u8);
                buf.push((list_len & 0xFF) as u8);
                buf.extend_from_slice(&list_data);
                buf
            }
            Self::MaxFragmentLength(mfl) => vec![mfl.to_u8()],
            Self::StatusRequest => {
                // OCSP status request: type(1) + responder_id_list(2+0) + request_extensions(2+0)
                vec![0x01, 0x00, 0x00, 0x00, 0x00]
            }
            Self::SupportedGroups(groups) => {
                let list_len = (groups.len() * 2) as u16;
                let mut buf = Vec::with_capacity(2 + list_len as usize);
                buf.push((list_len >> 8) as u8);
                buf.push((list_len & 0xFF) as u8);
                for g in groups {
                    let id = g.to_u16();
                    buf.push((id >> 8) as u8);
                    buf.push((id & 0xFF) as u8);
                }
                buf
            }
            Self::ECPointFormats(formats) => {
                let mut buf = Vec::with_capacity(1 + formats.len());
                buf.push(formats.len() as u8);
                for f in formats {
                    buf.push(f.to_u8());
                }
                buf
            }
            Self::SignatureAlgorithms(schemes) | Self::SignatureAlgorithmsCert(schemes) => {
                let list_len = (schemes.len() * 2) as u16;
                let mut buf = Vec::with_capacity(2 + list_len as usize);
                buf.push((list_len >> 8) as u8);
                buf.push((list_len & 0xFF) as u8);
                for s in schemes {
                    let id = s.to_u16();
                    buf.push((id >> 8) as u8);
                    buf.push((id & 0xFF) as u8);
                }
                buf
            }
            Self::Alpn(protocols) => {
                let mut list_data = Vec::new();
                for proto in protocols {
                    let bytes = proto.as_bytes();
                    list_data.push(bytes.len() as u8);
                    list_data.extend_from_slice(bytes);
                }
                let list_len = list_data.len() as u16;
                let mut buf = Vec::with_capacity(2 + list_data.len());
                buf.push((list_len >> 8) as u8);
                buf.push((list_len & 0xFF) as u8);
                buf.extend_from_slice(&list_data);
                buf
            }
            Self::SignedCertificateTimestamp(data) => data.clone(),
            Self::Padding(len) => vec![0u8; *len as usize],
            Self::EncryptThenMac => Vec::new(),
            Self::ExtendedMasterSecret => Vec::new(),
            Self::SessionTicket(data) => data.clone(),
            Self::SupportedVersions(versions) => {
                let list_len = (versions.len() * 2) as u8;
                let mut buf = Vec::with_capacity(1 + list_len as usize);
                buf.push(list_len);
                for v in versions {
                    buf.push(v.major);
                    buf.push(v.minor);
                }
                buf
            }
            Self::PskKeyExchangeModes(modes) => {
                let mut buf = Vec::with_capacity(1 + modes.len());
                buf.push(modes.len() as u8);
                for m in modes {
                    buf.push(m.to_u8());
                }
                buf
            }
            Self::KeyShareClientHello(entries) => {
                let mut entries_data = Vec::new();
                for e in entries {
                    entries_data.extend_from_slice(&e.serialize());
                }
                let list_len = entries_data.len() as u16;
                let mut buf = Vec::with_capacity(2 + entries_data.len());
                buf.push((list_len >> 8) as u8);
                buf.push((list_len & 0xFF) as u8);
                buf.extend_from_slice(&entries_data);
                buf
            }
            Self::KeyShareServerHello(entry) => entry.serialize(),
            Self::KeyShareHelloRetryRequest(group) => {
                let id = group.to_u16();
                vec![(id >> 8) as u8, (id & 0xFF) as u8]
            }
            Self::PreSharedKey(data) => data.clone(),
            Self::EarlyData(max_size) => {
                if let Some(ms) = max_size {
                    let mut buf = Vec::with_capacity(4);
                    buf.push((*ms >> 24) as u8);
                    buf.push((*ms >> 16) as u8);
                    buf.push((*ms >> 8) as u8);
                    buf.push((*ms & 0xFF) as u8);
                    buf
                } else {
                    Vec::new()
                }
            }
            Self::Cookie(data) => {
                let len = data.len() as u16;
                let mut buf = Vec::with_capacity(2 + data.len());
                buf.push((len >> 8) as u8);
                buf.push((len & 0xFF) as u8);
                buf.extend_from_slice(data);
                buf
            }
            Self::PostHandshakeAuth => Vec::new(),
            Self::RenegotiationInfo(data) => {
                let mut buf = Vec::with_capacity(1 + data.len());
                buf.push(data.len() as u8);
                buf.extend_from_slice(data);
                buf
            }
            Self::CertificateAuthorities(cas) => {
                let mut list_data = Vec::new();
                for ca in cas {
                    let len = ca.len() as u16;
                    list_data.push((len >> 8) as u8);
                    list_data.push((len & 0xFF) as u8);
                    list_data.extend_from_slice(ca);
                }
                let list_len = list_data.len() as u16;
                let mut buf = Vec::with_capacity(2 + list_data.len());
                buf.push((list_len >> 8) as u8);
                buf.push((list_len & 0xFF) as u8);
                buf.extend_from_slice(&list_data);
                buf
            }
            Self::Unknown { data, .. } => data.clone(),
        }
    }

    /// Whether this extension is required in TLS 1.3 ClientHello.
    pub fn is_required_tls13_client_hello(&self) -> bool {
        matches!(
            self,
            Self::SupportedVersions(_)
                | Self::KeyShareClientHello(_)
                | Self::SignatureAlgorithms(_)
                | Self::SupportedGroups(_)
        )
    }

    /// Whether this extension is allowed in a ServerHello.
    pub fn is_allowed_in_server_hello(&self) -> bool {
        matches!(
            self,
            Self::SupportedVersions(_)
                | Self::KeyShareServerHello(_)
                | Self::KeyShareHelloRetryRequest(_)
                | Self::PreSharedKey(_)
                | Self::MaxFragmentLength(_)
                | Self::EncryptThenMac
                | Self::ExtendedMasterSecret
                | Self::SessionTicket(_)
                | Self::RenegotiationInfo(_)
                | Self::Alpn(_)
                | Self::ECPointFormats(_)
                | Self::StatusRequest
        )
    }

    /// Convert to negsyn_types Extension.
    pub fn to_negsyn_extension(&self) -> negsyn_types::Extension {
        let data = self.serialize_data();
        let name = self.name();
        let critical = self.is_required_tls13_client_hello();
        negsyn_types::Extension::new(self.type_id(), name, data, critical)
    }

    /// Human-readable extension name.
    pub fn name(&self) -> String {
        match self {
            Self::ServerName(_) => "server_name".to_string(),
            Self::MaxFragmentLength(_) => "max_fragment_length".to_string(),
            Self::StatusRequest => "status_request".to_string(),
            Self::SupportedGroups(_) => "supported_groups".to_string(),
            Self::ECPointFormats(_) => "ec_point_formats".to_string(),
            Self::SignatureAlgorithms(_) => "signature_algorithms".to_string(),
            Self::SignatureAlgorithmsCert(_) => "signature_algorithms_cert".to_string(),
            Self::Alpn(_) => "application_layer_protocol_negotiation".to_string(),
            Self::SignedCertificateTimestamp(_) => "signed_certificate_timestamp".to_string(),
            Self::Padding(_) => "padding".to_string(),
            Self::EncryptThenMac => "encrypt_then_mac".to_string(),
            Self::ExtendedMasterSecret => "extended_master_secret".to_string(),
            Self::SessionTicket(_) => "session_ticket".to_string(),
            Self::SupportedVersions(_) => "supported_versions".to_string(),
            Self::PskKeyExchangeModes(_) => "psk_key_exchange_modes".to_string(),
            Self::KeyShareClientHello(_) | Self::KeyShareServerHello(_) | Self::KeyShareHelloRetryRequest(_) => {
                "key_share".to_string()
            }
            Self::PreSharedKey(_) => "pre_shared_key".to_string(),
            Self::EarlyData(_) => "early_data".to_string(),
            Self::Cookie(_) => "cookie".to_string(),
            Self::PostHandshakeAuth => "post_handshake_auth".to_string(),
            Self::RenegotiationInfo(_) => "renegotiation_info".to_string(),
            Self::CertificateAuthorities(_) => "certificate_authorities".to_string(),
            Self::Unknown { type_id, .. } => format!("unknown(0x{:04X})", type_id),
        }
    }
}

impl fmt::Display for TlsExtension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}(0x{:04X})", self.name(), self.type_id())
    }
}

// ---------------------------------------------------------------------------
// Extension parsing
// ---------------------------------------------------------------------------

/// Parse a single raw extension (type_id + length + data).
pub fn parse_extension_raw(input: &[u8]) -> IResult<&[u8], (u16, Vec<u8>)> {
    let (input, type_id) = be_u16(input)?;
    let (input, length) = be_u16(input)?;
    let (input, data) = take(length as usize)(input)?;
    Ok((input, (type_id, data.to_vec())))
}

/// Parse a list of extensions from a ClientHello/ServerHello.
pub fn parse_extensions(input: &[u8]) -> IResult<&[u8], Vec<TlsExtension>> {
    if input.is_empty() {
        return Ok((input, Vec::new()));
    }
    let (input, extensions_len) = be_u16(input)?;
    let (remaining, ext_data) = take(extensions_len as usize)(input)?;

    let mut extensions = Vec::new();
    let mut cursor = ext_data;
    while !cursor.is_empty() {
        let (rest, (type_id, data)) = parse_extension_raw(cursor)?;
        let ext = decode_extension(type_id, &data);
        extensions.push(ext);
        cursor = rest;
    }

    Ok((remaining, extensions))
}

/// Decode a raw extension into a typed TlsExtension.
pub fn decode_extension(type_id: u16, data: &[u8]) -> TlsExtension {
    match type_id {
        EXT_SERVER_NAME => decode_server_name(data),
        EXT_MAX_FRAGMENT_LENGTH => {
            if let Some(&byte) = data.first() {
                if let Some(mfl) = MaxFragmentLength::from_u8(byte) {
                    return TlsExtension::MaxFragmentLength(mfl);
                }
            }
            TlsExtension::Unknown {
                type_id,
                data: data.to_vec(),
            }
        }
        EXT_STATUS_REQUEST => TlsExtension::StatusRequest,
        EXT_SUPPORTED_GROUPS => decode_supported_groups(data),
        EXT_EC_POINT_FORMATS => decode_ec_point_formats(data),
        EXT_SIGNATURE_ALGORITHMS => decode_signature_algorithms(data, false),
        EXT_SIGNATURE_ALGORITHMS_CERT => decode_signature_algorithms(data, true),
        EXT_ALPN => decode_alpn(data),
        EXT_SCT => TlsExtension::SignedCertificateTimestamp(data.to_vec()),
        EXT_PADDING => TlsExtension::Padding(data.len() as u16),
        EXT_ENCRYPT_THEN_MAC => TlsExtension::EncryptThenMac,
        EXT_EXTENDED_MASTER_SECRET => TlsExtension::ExtendedMasterSecret,
        EXT_SESSION_TICKET => TlsExtension::SessionTicket(data.to_vec()),
        EXT_SUPPORTED_VERSIONS => decode_supported_versions(data),
        EXT_PSK_KEY_EXCHANGE_MODES => decode_psk_key_exchange_modes(data),
        EXT_KEY_SHARE => decode_key_share_client(data),
        EXT_PRE_SHARED_KEY => TlsExtension::PreSharedKey(data.to_vec()),
        EXT_EARLY_DATA => {
            if data.len() == 4 {
                let max = ((data[0] as u32) << 24)
                    | ((data[1] as u32) << 16)
                    | ((data[2] as u32) << 8)
                    | (data[3] as u32);
                TlsExtension::EarlyData(Some(max))
            } else {
                TlsExtension::EarlyData(None)
            }
        }
        EXT_COOKIE => {
            if data.len() >= 2 {
                let len = ((data[0] as usize) << 8) | (data[1] as usize);
                if data.len() >= 2 + len {
                    TlsExtension::Cookie(data[2..2 + len].to_vec())
                } else {
                    TlsExtension::Cookie(data[2..].to_vec())
                }
            } else {
                TlsExtension::Cookie(Vec::new())
            }
        }
        EXT_POST_HANDSHAKE_AUTH => TlsExtension::PostHandshakeAuth,
        EXT_RENEGOTIATION_INFO => {
            if !data.is_empty() {
                let len = data[0] as usize;
                if data.len() > len {
                    TlsExtension::RenegotiationInfo(data[1..1 + len].to_vec())
                } else {
                    TlsExtension::RenegotiationInfo(data[1..].to_vec())
                }
            } else {
                TlsExtension::RenegotiationInfo(Vec::new())
            }
        }
        _ => TlsExtension::Unknown {
            type_id,
            data: data.to_vec(),
        },
    }
}

fn decode_server_name(data: &[u8]) -> TlsExtension {
    let mut names = Vec::new();
    if data.len() < 2 {
        return TlsExtension::ServerName(names);
    }
    let list_len = ((data[0] as usize) << 8) | (data[1] as usize);
    let mut cursor = &data[2..];
    let end = list_len.min(cursor.len());
    let mut consumed = 0;
    while consumed < end && cursor.len() >= 3 {
        let _name_type = cursor[0];
        let name_len = ((cursor[1] as usize) << 8) | (cursor[2] as usize);
        cursor = &cursor[3..];
        consumed += 3;
        if cursor.len() >= name_len {
            if let Ok(name) = std::str::from_utf8(&cursor[..name_len]) {
                names.push(name.to_string());
            }
            cursor = &cursor[name_len..];
            consumed += name_len;
        } else {
            break;
        }
    }
    TlsExtension::ServerName(names)
}

fn decode_supported_groups(data: &[u8]) -> TlsExtension {
    let mut groups = Vec::new();
    if data.len() < 2 {
        return TlsExtension::SupportedGroups(groups);
    }
    let list_len = ((data[0] as usize) << 8) | (data[1] as usize);
    let count = list_len / 2;
    let mut cursor = &data[2..];
    for _ in 0..count {
        if cursor.len() < 2 {
            break;
        }
        let id = ((cursor[0] as u16) << 8) | (cursor[1] as u16);
        groups.push(NamedGroup::from_u16(id));
        cursor = &cursor[2..];
    }
    TlsExtension::SupportedGroups(groups)
}

fn decode_ec_point_formats(data: &[u8]) -> TlsExtension {
    let mut formats = Vec::new();
    if data.is_empty() {
        return TlsExtension::ECPointFormats(formats);
    }
    let count = data[0] as usize;
    for i in 0..count {
        if i + 1 < data.len() {
            formats.push(ECPointFormat::from_u8(data[i + 1]));
        }
    }
    TlsExtension::ECPointFormats(formats)
}

fn decode_signature_algorithms(data: &[u8], is_cert: bool) -> TlsExtension {
    let mut schemes = Vec::new();
    if data.len() < 2 {
        if is_cert {
            return TlsExtension::SignatureAlgorithmsCert(schemes);
        }
        return TlsExtension::SignatureAlgorithms(schemes);
    }
    let list_len = ((data[0] as usize) << 8) | (data[1] as usize);
    let count = list_len / 2;
    let mut cursor = &data[2..];
    for _ in 0..count {
        if cursor.len() < 2 {
            break;
        }
        let id = ((cursor[0] as u16) << 8) | (cursor[1] as u16);
        schemes.push(SignatureScheme::from_u16(id));
        cursor = &cursor[2..];
    }
    if is_cert {
        TlsExtension::SignatureAlgorithmsCert(schemes)
    } else {
        TlsExtension::SignatureAlgorithms(schemes)
    }
}

fn decode_alpn(data: &[u8]) -> TlsExtension {
    let mut protocols = Vec::new();
    if data.len() < 2 {
        return TlsExtension::Alpn(protocols);
    }
    let _list_len = ((data[0] as usize) << 8) | (data[1] as usize);
    let mut cursor = &data[2..];
    while !cursor.is_empty() {
        let proto_len = cursor[0] as usize;
        cursor = &cursor[1..];
        if cursor.len() >= proto_len {
            if let Ok(proto) = std::str::from_utf8(&cursor[..proto_len]) {
                protocols.push(proto.to_string());
            }
            cursor = &cursor[proto_len..];
        } else {
            break;
        }
    }
    TlsExtension::Alpn(protocols)
}

fn decode_supported_versions(data: &[u8]) -> TlsExtension {
    let mut versions = Vec::new();
    if data.is_empty() {
        return TlsExtension::SupportedVersions(versions);
    }
    // ClientHello format: length(1) + versions list.
    // ServerHello format: just 2 bytes.
    if data.len() == 2 {
        let ver = TlsVersion::new(data[0], data[1]);
        versions.push(ver);
    } else {
        let list_len = data[0] as usize;
        let mut cursor = &data[1..];
        let count = list_len / 2;
        for _ in 0..count {
            if cursor.len() < 2 {
                break;
            }
            versions.push(TlsVersion::new(cursor[0], cursor[1]));
            cursor = &cursor[2..];
        }
    }
    TlsExtension::SupportedVersions(versions)
}

fn decode_psk_key_exchange_modes(data: &[u8]) -> TlsExtension {
    let mut modes = Vec::new();
    if data.is_empty() {
        return TlsExtension::PskKeyExchangeModes(modes);
    }
    let count = data[0] as usize;
    for i in 0..count {
        if i + 1 < data.len() {
            modes.push(PskKeyExchangeMode::from_u8(data[i + 1]));
        }
    }
    TlsExtension::PskKeyExchangeModes(modes)
}

fn decode_key_share_client(data: &[u8]) -> TlsExtension {
    let mut entries = Vec::new();
    if data.len() < 2 {
        return TlsExtension::KeyShareClientHello(entries);
    }
    let list_len = ((data[0] as usize) << 8) | (data[1] as usize);
    let mut cursor = &data[2..];
    let mut consumed = 0;
    while consumed < list_len && cursor.len() >= 4 {
        match KeyShareEntry::parse(cursor) {
            Ok((rest, entry)) => {
                let entry_size = cursor.len() - rest.len();
                consumed += entry_size;
                entries.push(entry);
                cursor = rest;
            }
            Err(_) => break,
        }
    }
    TlsExtension::KeyShareClientHello(entries)
}

// ---------------------------------------------------------------------------
// Extension negotiation
// ---------------------------------------------------------------------------

/// Negotiate extensions between client and server.
/// Returns the list of extensions the server should include in its response.
pub fn negotiate_extensions(
    client_extensions: &[TlsExtension],
    server_config: &ExtensionNegotiationConfig,
) -> Vec<TlsExtension> {
    let mut result = Vec::new();

    for ext in client_extensions {
        match ext {
            TlsExtension::ServerName(names) => {
                if server_config.accept_sni && !names.is_empty() {
                    // Server echoes empty SNI to acknowledge.
                    result.push(TlsExtension::ServerName(Vec::new()));
                }
            }
            TlsExtension::MaxFragmentLength(mfl) => {
                if server_config.accept_max_fragment_length {
                    result.push(TlsExtension::MaxFragmentLength(*mfl));
                }
            }
            TlsExtension::SupportedGroups(_) => {
                // Server doesn't echo supported_groups in ServerHello.
            }
            TlsExtension::ECPointFormats(formats) => {
                if formats.contains(&ECPointFormat::Uncompressed) {
                    result.push(TlsExtension::ECPointFormats(vec![ECPointFormat::Uncompressed]));
                }
            }
            TlsExtension::Alpn(protocols) => {
                if let Some(ref server_protocols) = server_config.alpn_protocols {
                    for sp in server_protocols {
                        if protocols.iter().any(|cp| cp == sp) {
                            result.push(TlsExtension::Alpn(vec![sp.clone()]));
                            break;
                        }
                    }
                }
            }
            TlsExtension::EncryptThenMac => {
                if server_config.accept_encrypt_then_mac {
                    result.push(TlsExtension::EncryptThenMac);
                }
            }
            TlsExtension::ExtendedMasterSecret => {
                if server_config.accept_extended_master_secret {
                    result.push(TlsExtension::ExtendedMasterSecret);
                }
            }
            TlsExtension::SessionTicket(_) => {
                if server_config.accept_session_tickets {
                    result.push(TlsExtension::SessionTicket(Vec::new()));
                }
            }
            TlsExtension::SupportedVersions(versions) => {
                if let Some(ref sv) = server_config.supported_version_response {
                    if versions.contains(sv) {
                        result.push(TlsExtension::SupportedVersions(vec![*sv]));
                    }
                }
            }
            TlsExtension::RenegotiationInfo(data) => {
                if server_config.accept_renegotiation_info {
                    // Echo empty for initial handshake.
                    if data.is_empty() {
                        result.push(TlsExtension::RenegotiationInfo(Vec::new()));
                    }
                }
            }
            TlsExtension::StatusRequest => {
                if server_config.accept_status_request {
                    result.push(TlsExtension::StatusRequest);
                }
            }
            _ => {}
        }
    }

    result
}

/// Configuration for server-side extension negotiation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtensionNegotiationConfig {
    pub accept_sni: bool,
    pub accept_max_fragment_length: bool,
    pub accept_encrypt_then_mac: bool,
    pub accept_extended_master_secret: bool,
    pub accept_session_tickets: bool,
    pub accept_renegotiation_info: bool,
    pub accept_status_request: bool,
    pub alpn_protocols: Option<Vec<String>>,
    pub supported_version_response: Option<TlsVersion>,
}

impl Default for ExtensionNegotiationConfig {
    fn default() -> Self {
        Self {
            accept_sni: true,
            accept_max_fragment_length: false,
            accept_encrypt_then_mac: true,
            accept_extended_master_secret: true,
            accept_session_tickets: true,
            accept_renegotiation_info: true,
            accept_status_request: false,
            alpn_protocols: None,
            supported_version_response: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Anti-downgrade sentinel check
// ---------------------------------------------------------------------------

/// Check for TLS 1.3 anti-downgrade sentinel in ServerHello.random.
/// Returns true if a sentinel is detected, indicating a potential downgrade.
pub fn check_anti_downgrade_sentinel(server_random: &[u8; 32]) -> Option<&'static str> {
    let tail: &[u8] = &server_random[24..32];
    if tail == crate::version::DOWNGRADE_SENTINEL_TLS12 {
        Some("TLS 1.2 downgrade sentinel detected")
    } else if tail == crate::version::DOWNGRADE_SENTINEL_TLS11 {
        Some("TLS 1.1 (or below) downgrade sentinel detected")
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sni_serialize_roundtrip() {
        let ext = TlsExtension::ServerName(vec!["example.com".to_string()]);
        let bytes = ext.serialize();
        assert_eq!(bytes[0], 0x00); // type high
        assert_eq!(bytes[1], 0x00); // type low
        let (_, exts) = parse_extension_raw(&bytes).unwrap();
        assert_eq!(exts.0, EXT_SERVER_NAME);
        let decoded = decode_extension(exts.0, &exts.1);
        match decoded {
            TlsExtension::ServerName(names) => {
                assert_eq!(names, vec!["example.com"]);
            }
            _ => panic!("Expected ServerName"),
        }
    }

    #[test]
    fn test_supported_groups_roundtrip() {
        let ext = TlsExtension::SupportedGroups(vec![
            NamedGroup::X25519,
            NamedGroup::Secp256r1,
            NamedGroup::Secp384r1,
        ]);
        let bytes = ext.serialize_data();
        let decoded = decode_supported_groups(&bytes);
        match decoded {
            TlsExtension::SupportedGroups(groups) => {
                assert_eq!(groups.len(), 3);
                assert_eq!(groups[0], NamedGroup::X25519);
            }
            _ => panic!("Expected SupportedGroups"),
        }
    }

    #[test]
    fn test_signature_algorithms_roundtrip() {
        let ext = TlsExtension::SignatureAlgorithms(vec![
            SignatureScheme::EcdsaSecp256r1Sha256,
            SignatureScheme::RsaPssRsaeSha256,
        ]);
        let bytes = ext.serialize_data();
        let decoded = decode_signature_algorithms(&bytes, false);
        match decoded {
            TlsExtension::SignatureAlgorithms(schemes) => {
                assert_eq!(schemes.len(), 2);
                assert_eq!(schemes[0], SignatureScheme::EcdsaSecp256r1Sha256);
            }
            _ => panic!("Expected SignatureAlgorithms"),
        }
    }

    #[test]
    fn test_alpn_roundtrip() {
        let ext = TlsExtension::Alpn(vec!["h2".to_string(), "http/1.1".to_string()]);
        let bytes = ext.serialize_data();
        let decoded = decode_alpn(&bytes);
        match decoded {
            TlsExtension::Alpn(protos) => {
                assert_eq!(protos, vec!["h2", "http/1.1"]);
            }
            _ => panic!("Expected ALPN"),
        }
    }

    #[test]
    fn test_supported_versions_roundtrip() {
        let ext = TlsExtension::SupportedVersions(vec![
            TlsVersion::TLS1_3,
            TlsVersion::TLS1_2,
        ]);
        let bytes = ext.serialize_data();
        let decoded = decode_supported_versions(&bytes);
        match decoded {
            TlsExtension::SupportedVersions(versions) => {
                assert_eq!(versions.len(), 2);
                assert_eq!(versions[0], TlsVersion::TLS1_3);
            }
            _ => panic!("Expected SupportedVersions"),
        }
    }

    #[test]
    fn test_key_share_entry_roundtrip() {
        let entry = KeyShareEntry::new(NamedGroup::X25519, vec![0xAB; 32]);
        let bytes = entry.serialize();
        let (_, parsed) = KeyShareEntry::parse(&bytes).unwrap();
        assert_eq!(parsed.group, NamedGroup::X25519);
        assert_eq!(parsed.key_exchange.len(), 32);
    }

    #[test]
    fn test_extension_type_ids() {
        assert_eq!(TlsExtension::EncryptThenMac.type_id(), EXT_ENCRYPT_THEN_MAC);
        assert_eq!(TlsExtension::ExtendedMasterSecret.type_id(), EXT_EXTENDED_MASTER_SECRET);
        assert_eq!(
            TlsExtension::SupportedVersions(vec![]).type_id(),
            EXT_SUPPORTED_VERSIONS
        );
    }

    #[test]
    fn test_extension_name() {
        assert_eq!(TlsExtension::EncryptThenMac.name(), "encrypt_then_mac");
        assert_eq!(
            TlsExtension::ServerName(vec![]).name(),
            "server_name"
        );
    }

    #[test]
    fn test_negotiate_extensions_basic() {
        let client = vec![
            TlsExtension::ServerName(vec!["example.com".to_string()]),
            TlsExtension::EncryptThenMac,
            TlsExtension::ExtendedMasterSecret,
            TlsExtension::Alpn(vec!["h2".to_string(), "http/1.1".to_string()]),
        ];
        let config = ExtensionNegotiationConfig {
            alpn_protocols: Some(vec!["h2".to_string()]),
            ..Default::default()
        };
        let server_exts = negotiate_extensions(&client, &config);

        let has_sni = server_exts.iter().any(|e| matches!(e, TlsExtension::ServerName(_)));
        let has_etm = server_exts.iter().any(|e| matches!(e, TlsExtension::EncryptThenMac));
        let has_ems = server_exts.iter().any(|e| matches!(e, TlsExtension::ExtendedMasterSecret));
        let alpn_match = server_exts.iter().find_map(|e| match e {
            TlsExtension::Alpn(protos) => Some(protos.clone()),
            _ => None,
        });

        assert!(has_sni);
        assert!(has_etm);
        assert!(has_ems);
        assert_eq!(alpn_match, Some(vec!["h2".to_string()]));
    }

    #[test]
    fn test_named_group_properties() {
        assert!(NamedGroup::X25519.is_ecdhe());
        assert!(!NamedGroup::X25519.is_ffdhe());
        assert!(NamedGroup::Ffdhe2048.is_ffdhe());
        assert_eq!(NamedGroup::X25519.key_bits(), 256);
        assert_eq!(NamedGroup::Ffdhe4096.key_bits(), 4096);
    }

    #[test]
    fn test_anti_downgrade_sentinel() {
        let mut random = [0x42u8; 32];
        assert!(check_anti_downgrade_sentinel(&random).is_none());

        random[24..32].copy_from_slice(&crate::version::DOWNGRADE_SENTINEL_TLS12);
        assert!(check_anti_downgrade_sentinel(&random).is_some());
    }

    #[test]
    fn test_psk_key_exchange_modes_roundtrip() {
        let ext = TlsExtension::PskKeyExchangeModes(vec![
            PskKeyExchangeMode::PskDheKe,
        ]);
        let bytes = ext.serialize_data();
        let decoded = decode_psk_key_exchange_modes(&bytes);
        match decoded {
            TlsExtension::PskKeyExchangeModes(modes) => {
                assert_eq!(modes, vec![PskKeyExchangeMode::PskDheKe]);
            }
            _ => panic!("Expected PskKeyExchangeModes"),
        }
    }

    #[test]
    fn test_renegotiation_info_roundtrip() {
        let ext = TlsExtension::RenegotiationInfo(vec![]);
        let data = ext.serialize_data();
        assert_eq!(data, vec![0x00]); // length byte = 0
        let decoded = decode_extension(EXT_RENEGOTIATION_INFO, &data);
        match decoded {
            TlsExtension::RenegotiationInfo(ri) => assert!(ri.is_empty()),
            _ => panic!("Expected RenegotiationInfo"),
        }
    }

    #[test]
    fn test_is_allowed_in_server_hello() {
        assert!(TlsExtension::EncryptThenMac.is_allowed_in_server_hello());
        assert!(TlsExtension::ExtendedMasterSecret.is_allowed_in_server_hello());
        assert!(!TlsExtension::SignatureAlgorithms(vec![]).is_allowed_in_server_hello());
    }

    #[test]
    fn test_to_negsyn_extension() {
        let ext = TlsExtension::EncryptThenMac;
        let ne = ext.to_negsyn_extension();
        assert_eq!(ne.id, EXT_ENCRYPT_THEN_MAC);
        assert_eq!(ne.name, "encrypt_then_mac");
    }
}
