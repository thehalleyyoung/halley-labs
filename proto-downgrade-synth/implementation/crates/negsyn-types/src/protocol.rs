//! Core protocol types for TLS/SSH negotiation analysis.
//!
//! Implements Definition D1 (Negotiation Protocol LTS) and related types
//! for modeling protocol handshake state machines, cipher suites, and
//! version negotiation.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;

// ── Handshake Phase ──────────────────────────────────────────────────────

/// Handshake phases for TLS/SSH negotiation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum HandshakePhase {
    Init,
    /// Alias used by downstream crates for the initial state.
    Initial,
    ClientHelloSent,
    /// Simplified alias for ClientHelloSent.
    ClientHello,
    ServerHelloReceived,
    /// Simplified alias for ServerHelloReceived.
    ServerHello,
    /// Certificate exchange phase.
    Certificate,
    /// Key exchange phase.
    KeyExchange,
    Negotiated,
    /// Change cipher spec phase.
    ChangeCipherSpec,
    /// Handshake finished phase.
    Finished,
    Done,
    /// Application data phase (post-handshake).
    ApplicationData,
    Abort,
    /// Alert received/sent.
    Alert,
    /// Renegotiation phase.
    Renegotiation,
}

impl HandshakePhase {
    /// Whether this phase is terminal (no further transitions).
    pub fn is_terminal(&self) -> bool {
        matches!(self, HandshakePhase::Done | HandshakePhase::Abort | HandshakePhase::Finished | HandshakePhase::Alert)
    }

    /// Whether this phase represents a successful outcome.
    pub fn is_success(&self) -> bool {
        matches!(self, HandshakePhase::Done | HandshakePhase::Finished | HandshakePhase::ApplicationData)
    }

    /// Valid successor phases from this phase.
    pub fn valid_successors(&self) -> &[HandshakePhase] {
        match self {
            HandshakePhase::Init | HandshakePhase::Initial => &[HandshakePhase::ClientHelloSent, HandshakePhase::ClientHello, HandshakePhase::Abort],
            HandshakePhase::ClientHelloSent | HandshakePhase::ClientHello => &[
                HandshakePhase::ServerHelloReceived,
                HandshakePhase::ServerHello,
                HandshakePhase::Abort,
                HandshakePhase::Alert,
            ],
            HandshakePhase::ServerHelloReceived | HandshakePhase::ServerHello => {
                &[HandshakePhase::Certificate, HandshakePhase::Negotiated, HandshakePhase::Abort, HandshakePhase::Alert]
            }
            HandshakePhase::Certificate => &[HandshakePhase::KeyExchange, HandshakePhase::Negotiated, HandshakePhase::Abort, HandshakePhase::Alert],
            HandshakePhase::KeyExchange => &[HandshakePhase::ChangeCipherSpec, HandshakePhase::Negotiated, HandshakePhase::Abort, HandshakePhase::Alert],
            HandshakePhase::ChangeCipherSpec => &[HandshakePhase::Finished, HandshakePhase::Done, HandshakePhase::Abort, HandshakePhase::Alert],
            HandshakePhase::Negotiated => &[HandshakePhase::Done, HandshakePhase::Finished, HandshakePhase::Abort, HandshakePhase::Alert],
            HandshakePhase::Finished => &[HandshakePhase::ApplicationData, HandshakePhase::Done],
            HandshakePhase::Done => &[HandshakePhase::Renegotiation],
            HandshakePhase::ApplicationData => &[HandshakePhase::Renegotiation, HandshakePhase::Alert],
            HandshakePhase::Abort | HandshakePhase::Alert => &[],
            HandshakePhase::Renegotiation => &[HandshakePhase::ClientHello, HandshakePhase::ClientHelloSent, HandshakePhase::Abort, HandshakePhase::Alert],
        }
    }

    /// Check whether transition to `next` is valid.
    pub fn can_transition_to(&self, next: HandshakePhase) -> bool {
        self.valid_successors().contains(&next)
    }

    pub fn all_phases() -> &'static [HandshakePhase] {
        &[
            HandshakePhase::Init,
            HandshakePhase::Initial,
            HandshakePhase::ClientHelloSent,
            HandshakePhase::ClientHello,
            HandshakePhase::ServerHelloReceived,
            HandshakePhase::ServerHello,
            HandshakePhase::Certificate,
            HandshakePhase::KeyExchange,
            HandshakePhase::Negotiated,
            HandshakePhase::ChangeCipherSpec,
            HandshakePhase::Finished,
            HandshakePhase::Done,
            HandshakePhase::ApplicationData,
            HandshakePhase::Abort,
            HandshakePhase::Alert,
            HandshakePhase::Renegotiation,
        ]
    }

    /// Return a numeric index for ordering phases.
    pub fn order_index(&self) -> u8 {
        match self {
            HandshakePhase::Init | HandshakePhase::Initial => 0,
            HandshakePhase::ClientHelloSent | HandshakePhase::ClientHello => 1,
            HandshakePhase::ServerHelloReceived | HandshakePhase::ServerHello => 2,
            HandshakePhase::Certificate => 3,
            HandshakePhase::KeyExchange => 4,
            HandshakePhase::Negotiated => 5,
            HandshakePhase::ChangeCipherSpec => 6,
            HandshakePhase::Finished => 7,
            HandshakePhase::Done => 8,
            HandshakePhase::ApplicationData => 9,
            HandshakePhase::Alert => 10,
            HandshakePhase::Abort => 11,
            HandshakePhase::Renegotiation => 12,
        }
    }
}

impl fmt::Display for HandshakePhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HandshakePhase::Init => write!(f, "Init"),
            HandshakePhase::Initial => write!(f, "Initial"),
            HandshakePhase::ClientHelloSent => write!(f, "ClientHelloSent"),
            HandshakePhase::ClientHello => write!(f, "ClientHello"),
            HandshakePhase::ServerHelloReceived => write!(f, "ServerHelloReceived"),
            HandshakePhase::ServerHello => write!(f, "ServerHello"),
            HandshakePhase::Certificate => write!(f, "Certificate"),
            HandshakePhase::KeyExchange => write!(f, "KeyExchange"),
            HandshakePhase::Negotiated => write!(f, "Negotiated"),
            HandshakePhase::ChangeCipherSpec => write!(f, "ChangeCipherSpec"),
            HandshakePhase::Finished => write!(f, "Finished"),
            HandshakePhase::Done => write!(f, "Done"),
            HandshakePhase::ApplicationData => write!(f, "ApplicationData"),
            HandshakePhase::Abort => write!(f, "Abort"),
            HandshakePhase::Alert => write!(f, "Alert"),
            HandshakePhase::Renegotiation => write!(f, "Renegotiation"),
        }
    }
}

// ── Security Level ───────────────────────────────────────────────────────

/// Security level classification for cipher suites and algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SecurityLevel {
    /// Formally broken (e.g., RC4, DES, export ciphers).
    Broken,
    /// Deprecated, known weaknesses (e.g., 3DES, SHA1).
    Weak,
    /// Legacy, acceptable for backward compat (e.g., AES-CBC with SHA256).
    Legacy,
    /// Current standard (e.g., AES-GCM with SHA384).
    Standard,
    /// Best available (e.g., ChaCha20-Poly1305, AES-256-GCM).
    High,
}

impl SecurityLevel {
    fn ordinal(&self) -> u8 {
        match self {
            SecurityLevel::Broken => 0,
            SecurityLevel::Weak => 1,
            SecurityLevel::Legacy => 2,
            SecurityLevel::Standard => 3,
            SecurityLevel::High => 4,
        }
    }

    /// Whether this level is considered secure for modern use.
    pub fn is_secure(&self) -> bool {
        self.ordinal() >= SecurityLevel::Standard.ordinal()
    }

    /// Whether this level indicates a known vulnerability.
    pub fn is_vulnerable(&self) -> bool {
        matches!(self, SecurityLevel::Broken)
    }
}

impl PartialOrd for SecurityLevel {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SecurityLevel {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.ordinal().cmp(&other.ordinal())
    }
}

impl fmt::Display for SecurityLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SecurityLevel::Broken => write!(f, "BROKEN"),
            SecurityLevel::Weak => write!(f, "WEAK"),
            SecurityLevel::Legacy => write!(f, "LEGACY"),
            SecurityLevel::Standard => write!(f, "STANDARD"),
            SecurityLevel::High => write!(f, "HIGH"),
        }
    }
}

// ── Key Exchange / Auth / Encryption / MAC algorithms ────────────────────

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KeyExchange {
    RSA,
    DHE,
    ECDHE,
    PSK,
    DHEPSK,
    ECDHEPSK,
    SRP,
    Kerberos,
    NULL,
    // Aliases used by downstream crates
    Rsa,
    Dhe,
    Ecdhe,
    Psk,
}

impl fmt::Display for KeyExchange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KeyExchange::RSA | KeyExchange::Rsa => write!(f, "RSA"),
            KeyExchange::DHE | KeyExchange::Dhe => write!(f, "DHE"),
            KeyExchange::ECDHE | KeyExchange::Ecdhe => write!(f, "ECDHE"),
            KeyExchange::PSK | KeyExchange::Psk => write!(f, "PSK"),
            KeyExchange::DHEPSK => write!(f, "DHE_PSK"),
            KeyExchange::ECDHEPSK => write!(f, "ECDHE_PSK"),
            KeyExchange::SRP => write!(f, "SRP"),
            KeyExchange::Kerberos => write!(f, "KRB5"),
            KeyExchange::NULL => write!(f, "NULL"),
        }
    }
}

impl KeyExchange {
    pub fn provides_forward_secrecy(&self) -> bool {
        matches!(self, KeyExchange::DHE | KeyExchange::Dhe | KeyExchange::ECDHE | KeyExchange::Ecdhe | KeyExchange::DHEPSK | KeyExchange::ECDHEPSK)
    }
}

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AuthAlgorithm {
    RSA,
    DSS,
    ECDSA,
    PSK,
    NULL,
    Anon,
    SHA256,
    SHA384,
    // Aliases used by downstream crates
    Rsa,
    Ecdsa,
    Dss,
}

impl fmt::Display for AuthAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AuthAlgorithm::RSA | AuthAlgorithm::Rsa => write!(f, "RSA"),
            AuthAlgorithm::DSS | AuthAlgorithm::Dss => write!(f, "DSS"),
            AuthAlgorithm::ECDSA | AuthAlgorithm::Ecdsa => write!(f, "ECDSA"),
            AuthAlgorithm::PSK => write!(f, "PSK"),
            AuthAlgorithm::NULL => write!(f, "NULL"),
            AuthAlgorithm::Anon => write!(f, "anon"),
            AuthAlgorithm::SHA256 => write!(f, "SHA256"),
            AuthAlgorithm::SHA384 => write!(f, "SHA384"),
        }
    }
}

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EncryptionAlgorithm {
    AES128GCM,
    AES256GCM,
    AES128CBC,
    AES256CBC,
    ChaCha20Poly1305,
    TripleDESCBC,
    RC4_128,
    DESCBC,
    AES128CCM,
    AES256CCM,
    Camellia128CBC,
    Camellia256CBC,
    Camellia128GCM,
    Camellia256GCM,
    SEED_CBC,
    ARIA128GCM,
    ARIA256GCM,
    NULL,
    // Aliases used by downstream crates
    Aes128,
    Aes256,
    Aes128Gcm,
    Aes256Gcm,
    Aes128Cbc,
    Aes256Cbc,
    TripleDes,
    Rc4,
    Des,
}

impl EncryptionAlgorithm {
    pub fn key_bits(&self) -> u32 {
        match self {
            EncryptionAlgorithm::AES128GCM | EncryptionAlgorithm::AES128CBC
            | EncryptionAlgorithm::AES128CCM | EncryptionAlgorithm::Camellia128CBC
            | EncryptionAlgorithm::Camellia128GCM | EncryptionAlgorithm::SEED_CBC
            | EncryptionAlgorithm::ARIA128GCM
            | EncryptionAlgorithm::Aes128 | EncryptionAlgorithm::Aes128Gcm
            | EncryptionAlgorithm::Aes128Cbc => 128,
            EncryptionAlgorithm::AES256GCM | EncryptionAlgorithm::AES256CBC
            | EncryptionAlgorithm::AES256CCM | EncryptionAlgorithm::ChaCha20Poly1305
            | EncryptionAlgorithm::Camellia256CBC | EncryptionAlgorithm::Camellia256GCM
            | EncryptionAlgorithm::ARIA256GCM
            | EncryptionAlgorithm::Aes256 | EncryptionAlgorithm::Aes256Gcm
            | EncryptionAlgorithm::Aes256Cbc => 256,
            EncryptionAlgorithm::TripleDESCBC | EncryptionAlgorithm::TripleDes => 168,
            EncryptionAlgorithm::RC4_128 | EncryptionAlgorithm::Rc4 => 128,
            EncryptionAlgorithm::DESCBC | EncryptionAlgorithm::Des => 56,
            EncryptionAlgorithm::NULL => 0,
        }
    }

    pub fn is_aead(&self) -> bool {
        matches!(
            self,
            EncryptionAlgorithm::AES128GCM
                | EncryptionAlgorithm::AES256GCM
                | EncryptionAlgorithm::ChaCha20Poly1305
                | EncryptionAlgorithm::AES128CCM
                | EncryptionAlgorithm::AES256CCM
                | EncryptionAlgorithm::Camellia128GCM
                | EncryptionAlgorithm::Camellia256GCM
                | EncryptionAlgorithm::ARIA128GCM
                | EncryptionAlgorithm::ARIA256GCM
                | EncryptionAlgorithm::Aes128Gcm
                | EncryptionAlgorithm::Aes256Gcm
        )
    }
}

impl fmt::Display for EncryptionAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EncryptionAlgorithm::AES128GCM | EncryptionAlgorithm::Aes128Gcm => write!(f, "AES_128_GCM"),
            EncryptionAlgorithm::AES256GCM | EncryptionAlgorithm::Aes256Gcm => write!(f, "AES_256_GCM"),
            EncryptionAlgorithm::AES128CBC | EncryptionAlgorithm::Aes128Cbc | EncryptionAlgorithm::Aes128 => write!(f, "AES_128_CBC"),
            EncryptionAlgorithm::AES256CBC | EncryptionAlgorithm::Aes256Cbc | EncryptionAlgorithm::Aes256 => write!(f, "AES_256_CBC"),
            EncryptionAlgorithm::ChaCha20Poly1305 => write!(f, "CHACHA20_POLY1305"),
            EncryptionAlgorithm::TripleDESCBC | EncryptionAlgorithm::TripleDes => write!(f, "3DES_EDE_CBC"),
            EncryptionAlgorithm::RC4_128 | EncryptionAlgorithm::Rc4 => write!(f, "RC4_128"),
            EncryptionAlgorithm::DESCBC | EncryptionAlgorithm::Des => write!(f, "DES_CBC"),
            EncryptionAlgorithm::AES128CCM => write!(f, "AES_128_CCM"),
            EncryptionAlgorithm::AES256CCM => write!(f, "AES_256_CCM"),
            EncryptionAlgorithm::Camellia128CBC => write!(f, "CAMELLIA_128_CBC"),
            EncryptionAlgorithm::Camellia256CBC => write!(f, "CAMELLIA_256_CBC"),
            EncryptionAlgorithm::Camellia128GCM => write!(f, "CAMELLIA_128_GCM"),
            EncryptionAlgorithm::Camellia256GCM => write!(f, "CAMELLIA_256_GCM"),
            EncryptionAlgorithm::SEED_CBC => write!(f, "SEED_CBC"),
            EncryptionAlgorithm::ARIA128GCM => write!(f, "ARIA_128_GCM"),
            EncryptionAlgorithm::ARIA256GCM => write!(f, "ARIA_256_GCM"),
            EncryptionAlgorithm::NULL => write!(f, "NULL"),
        }
    }
}

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MacAlgorithm {
    SHA256,
    SHA384,
    SHA1,
    MD5,
    AEAD,
    NULL,
    // Aliases used by downstream crates
    HmacSha1,
    HmacSha256,
    HmacSha384,
    Aead,
}

impl MacAlgorithm {
    pub fn digest_bits(&self) -> u32 {
        match self {
            MacAlgorithm::SHA256 | MacAlgorithm::HmacSha256 => 256,
            MacAlgorithm::SHA384 | MacAlgorithm::HmacSha384 => 384,
            MacAlgorithm::SHA1 | MacAlgorithm::HmacSha1 => 160,
            MacAlgorithm::MD5 => 128,
            MacAlgorithm::AEAD | MacAlgorithm::Aead => 0,
            MacAlgorithm::NULL => 0,
        }
    }
}

impl fmt::Display for MacAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MacAlgorithm::SHA256 | MacAlgorithm::HmacSha256 => write!(f, "SHA256"),
            MacAlgorithm::SHA384 | MacAlgorithm::HmacSha384 => write!(f, "SHA384"),
            MacAlgorithm::SHA1 | MacAlgorithm::HmacSha1 => write!(f, "SHA1"),
            MacAlgorithm::MD5 => write!(f, "MD5"),
            MacAlgorithm::AEAD | MacAlgorithm::Aead => write!(f, "AEAD"),
            MacAlgorithm::NULL => write!(f, "NULL"),
        }
    }
}

// ── CipherSuite ──────────────────────────────────────────────────────────

/// A TLS cipher suite with IANA parameters and security classification.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CipherSuite {
    pub iana_id: u16,
    pub name: String,
    pub key_exchange: KeyExchange,
    pub auth: AuthAlgorithm,
    /// Alias for `auth`, used by downstream crates as `authentication`.
    pub authentication: AuthAlgorithm,
    pub encryption: EncryptionAlgorithm,
    pub mac: MacAlgorithm,
    pub security_level: SecurityLevel,
    /// Whether this suite is FIPS-140 approved.
    pub is_fips_approved: bool,
}

impl CipherSuite {
    pub fn new(
        iana_id: u16,
        name: impl Into<String>,
        key_exchange: KeyExchange,
        auth: AuthAlgorithm,
        encryption: EncryptionAlgorithm,
        mac: MacAlgorithm,
        security_level: SecurityLevel,
    ) -> Self {
        let is_fips = matches!(
            encryption,
            EncryptionAlgorithm::AES128GCM | EncryptionAlgorithm::AES256GCM
            | EncryptionAlgorithm::AES128CBC | EncryptionAlgorithm::AES256CBC
            | EncryptionAlgorithm::AES128CCM | EncryptionAlgorithm::AES256CCM
        );
        CipherSuite {
            iana_id,
            name: name.into(),
            key_exchange,
            authentication: auth,
            auth,
            encryption,
            mac,
            security_level,
            is_fips_approved: is_fips,
        }
    }

    /// Create a null/empty cipher suite.
    pub fn null_suite() -> Self {
        Self::new(0x0000, "TLS_NULL_WITH_NULL_NULL", KeyExchange::NULL, AuthAlgorithm::NULL, EncryptionAlgorithm::NULL, MacAlgorithm::NULL, SecurityLevel::Broken)
    }

    /// IANA ID accessor (for code that uses `.id` instead of `.iana_id`).
    pub fn id(&self) -> u16 {
        self.iana_id
    }

    /// Whether this cipher provides forward secrecy.
    pub fn has_forward_secrecy(&self) -> bool {
        self.key_exchange.provides_forward_secrecy()
    }

    /// Whether this cipher uses authenticated encryption.
    pub fn has_aead(&self) -> bool {
        self.encryption.is_aead()
    }

    /// Effective key strength in bits.
    pub fn effective_key_bits(&self) -> u32 {
        self.encryption.key_bits()
    }

    /// Whether a downgrade from `self` to `other` is a security concern.
    pub fn is_downgrade_from(&self, other: &CipherSuite) -> bool {
        self.security_level < other.security_level
    }

    /// Compute a composite security score (higher = more secure).
    pub fn security_score(&self) -> u32 {
        let level_score = self.security_level.ordinal() as u32 * 100;
        let key_score = self.encryption.key_bits() / 4;
        let fs_bonus = if self.has_forward_secrecy() { 50 } else { 0 };
        let aead_bonus = if self.has_aead() { 30 } else { 0 };
        level_score + key_score + fs_bonus + aead_bonus
    }
}

impl PartialOrd for CipherSuite {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CipherSuite {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.security_score().cmp(&other.security_score())
    }
}

impl fmt::Display for CipherSuite {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} (0x{:04X}) [{}]", self.name, self.iana_id, self.security_level)
    }
}

impl fmt::LowerHex for CipherSuite {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:04x}", self.iana_id)
    }
}

// ── IANA Cipher Suite Constants ──────────────────────────────────────────

macro_rules! define_cipher_suite {
    ($name:ident, $id:expr, $sname:expr, $kx:expr, $auth:expr, $enc:expr, $mac:expr, $sec:expr) => {
        pub const $name: CipherSuite = CipherSuite {
            iana_id: $id,
            name: String::new(), // We'll use a function to get the actual name
            key_exchange: $kx,
            auth: $auth,
            encryption: $enc,
            mac: $mac,
            security_level: $sec,
        };
    };
}

/// Registry of common IANA cipher suites (50+).
pub struct CipherSuiteRegistry;

impl CipherSuiteRegistry {
    /// Returns all known cipher suites.
    pub fn all() -> Vec<CipherSuite> {
        vec![
            // TLS 1.3 cipher suites
            CipherSuite::new(0x1301, "TLS_AES_128_GCM_SHA256", KeyExchange::ECDHE, AuthAlgorithm::SHA256, EncryptionAlgorithm::AES128GCM, MacAlgorithm::AEAD, SecurityLevel::High),
            CipherSuite::new(0x1302, "TLS_AES_256_GCM_SHA384", KeyExchange::ECDHE, AuthAlgorithm::SHA384, EncryptionAlgorithm::AES256GCM, MacAlgorithm::AEAD, SecurityLevel::High),
            CipherSuite::new(0x1303, "TLS_CHACHA20_POLY1305_SHA256", KeyExchange::ECDHE, AuthAlgorithm::SHA256, EncryptionAlgorithm::ChaCha20Poly1305, MacAlgorithm::AEAD, SecurityLevel::High),
            CipherSuite::new(0x1304, "TLS_AES_128_CCM_SHA256", KeyExchange::ECDHE, AuthAlgorithm::SHA256, EncryptionAlgorithm::AES128CCM, MacAlgorithm::AEAD, SecurityLevel::High),
            CipherSuite::new(0x1305, "TLS_AES_128_CCM_8_SHA256", KeyExchange::ECDHE, AuthAlgorithm::SHA256, EncryptionAlgorithm::AES128CCM, MacAlgorithm::AEAD, SecurityLevel::Standard),

            // ECDHE+ECDSA suites
            CipherSuite::new(0xC02B, "TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256", KeyExchange::ECDHE, AuthAlgorithm::ECDSA, EncryptionAlgorithm::AES128GCM, MacAlgorithm::AEAD, SecurityLevel::High),
            CipherSuite::new(0xC02C, "TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384", KeyExchange::ECDHE, AuthAlgorithm::ECDSA, EncryptionAlgorithm::AES256GCM, MacAlgorithm::AEAD, SecurityLevel::High),
            CipherSuite::new(0xC023, "TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA256", KeyExchange::ECDHE, AuthAlgorithm::ECDSA, EncryptionAlgorithm::AES128CBC, MacAlgorithm::SHA256, SecurityLevel::Standard),
            CipherSuite::new(0xC024, "TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA384", KeyExchange::ECDHE, AuthAlgorithm::ECDSA, EncryptionAlgorithm::AES256CBC, MacAlgorithm::SHA384, SecurityLevel::Standard),
            CipherSuite::new(0xC009, "TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA", KeyExchange::ECDHE, AuthAlgorithm::ECDSA, EncryptionAlgorithm::AES128CBC, MacAlgorithm::SHA1, SecurityLevel::Legacy),
            CipherSuite::new(0xC00A, "TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA", KeyExchange::ECDHE, AuthAlgorithm::ECDSA, EncryptionAlgorithm::AES256CBC, MacAlgorithm::SHA1, SecurityLevel::Legacy),
            CipherSuite::new(0xCCA9, "TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256", KeyExchange::ECDHE, AuthAlgorithm::ECDSA, EncryptionAlgorithm::ChaCha20Poly1305, MacAlgorithm::AEAD, SecurityLevel::High),

            // ECDHE+RSA suites
            CipherSuite::new(0xC02F, "TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256", KeyExchange::ECDHE, AuthAlgorithm::RSA, EncryptionAlgorithm::AES128GCM, MacAlgorithm::AEAD, SecurityLevel::High),
            CipherSuite::new(0xC030, "TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384", KeyExchange::ECDHE, AuthAlgorithm::RSA, EncryptionAlgorithm::AES256GCM, MacAlgorithm::AEAD, SecurityLevel::High),
            CipherSuite::new(0xC027, "TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA256", KeyExchange::ECDHE, AuthAlgorithm::RSA, EncryptionAlgorithm::AES128CBC, MacAlgorithm::SHA256, SecurityLevel::Standard),
            CipherSuite::new(0xC028, "TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA384", KeyExchange::ECDHE, AuthAlgorithm::RSA, EncryptionAlgorithm::AES256CBC, MacAlgorithm::SHA384, SecurityLevel::Standard),
            CipherSuite::new(0xC013, "TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA", KeyExchange::ECDHE, AuthAlgorithm::RSA, EncryptionAlgorithm::AES128CBC, MacAlgorithm::SHA1, SecurityLevel::Legacy),
            CipherSuite::new(0xC014, "TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA", KeyExchange::ECDHE, AuthAlgorithm::RSA, EncryptionAlgorithm::AES256CBC, MacAlgorithm::SHA1, SecurityLevel::Legacy),
            CipherSuite::new(0xCCA8, "TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256", KeyExchange::ECDHE, AuthAlgorithm::RSA, EncryptionAlgorithm::ChaCha20Poly1305, MacAlgorithm::AEAD, SecurityLevel::High),

            // DHE+RSA suites
            CipherSuite::new(0x009E, "TLS_DHE_RSA_WITH_AES_128_GCM_SHA256", KeyExchange::DHE, AuthAlgorithm::RSA, EncryptionAlgorithm::AES128GCM, MacAlgorithm::AEAD, SecurityLevel::High),
            CipherSuite::new(0x009F, "TLS_DHE_RSA_WITH_AES_256_GCM_SHA384", KeyExchange::DHE, AuthAlgorithm::RSA, EncryptionAlgorithm::AES256GCM, MacAlgorithm::AEAD, SecurityLevel::High),
            CipherSuite::new(0x0067, "TLS_DHE_RSA_WITH_AES_128_CBC_SHA256", KeyExchange::DHE, AuthAlgorithm::RSA, EncryptionAlgorithm::AES128CBC, MacAlgorithm::SHA256, SecurityLevel::Standard),
            CipherSuite::new(0x006B, "TLS_DHE_RSA_WITH_AES_256_CBC_SHA256", KeyExchange::DHE, AuthAlgorithm::RSA, EncryptionAlgorithm::AES256CBC, MacAlgorithm::SHA256, SecurityLevel::Standard),
            CipherSuite::new(0x0033, "TLS_DHE_RSA_WITH_AES_128_CBC_SHA", KeyExchange::DHE, AuthAlgorithm::RSA, EncryptionAlgorithm::AES128CBC, MacAlgorithm::SHA1, SecurityLevel::Legacy),
            CipherSuite::new(0x0039, "TLS_DHE_RSA_WITH_AES_256_CBC_SHA", KeyExchange::DHE, AuthAlgorithm::RSA, EncryptionAlgorithm::AES256CBC, MacAlgorithm::SHA1, SecurityLevel::Legacy),
            CipherSuite::new(0xCCAA, "TLS_DHE_RSA_WITH_CHACHA20_POLY1305_SHA256", KeyExchange::DHE, AuthAlgorithm::RSA, EncryptionAlgorithm::ChaCha20Poly1305, MacAlgorithm::AEAD, SecurityLevel::High),

            // RSA (no forward secrecy) suites
            CipherSuite::new(0x009C, "TLS_RSA_WITH_AES_128_GCM_SHA256", KeyExchange::RSA, AuthAlgorithm::RSA, EncryptionAlgorithm::AES128GCM, MacAlgorithm::AEAD, SecurityLevel::Standard),
            CipherSuite::new(0x009D, "TLS_RSA_WITH_AES_256_GCM_SHA384", KeyExchange::RSA, AuthAlgorithm::RSA, EncryptionAlgorithm::AES256GCM, MacAlgorithm::AEAD, SecurityLevel::Standard),
            CipherSuite::new(0x003C, "TLS_RSA_WITH_AES_128_CBC_SHA256", KeyExchange::RSA, AuthAlgorithm::RSA, EncryptionAlgorithm::AES128CBC, MacAlgorithm::SHA256, SecurityLevel::Legacy),
            CipherSuite::new(0x003D, "TLS_RSA_WITH_AES_256_CBC_SHA256", KeyExchange::RSA, AuthAlgorithm::RSA, EncryptionAlgorithm::AES256CBC, MacAlgorithm::SHA256, SecurityLevel::Legacy),
            CipherSuite::new(0x002F, "TLS_RSA_WITH_AES_128_CBC_SHA", KeyExchange::RSA, AuthAlgorithm::RSA, EncryptionAlgorithm::AES128CBC, MacAlgorithm::SHA1, SecurityLevel::Legacy),
            CipherSuite::new(0x0035, "TLS_RSA_WITH_AES_256_CBC_SHA", KeyExchange::RSA, AuthAlgorithm::RSA, EncryptionAlgorithm::AES256CBC, MacAlgorithm::SHA1, SecurityLevel::Legacy),

            // Weak / broken suites (for downgrade detection)
            CipherSuite::new(0x000A, "TLS_RSA_WITH_3DES_EDE_CBC_SHA", KeyExchange::RSA, AuthAlgorithm::RSA, EncryptionAlgorithm::TripleDESCBC, MacAlgorithm::SHA1, SecurityLevel::Weak),
            CipherSuite::new(0x0005, "TLS_RSA_WITH_RC4_128_SHA", KeyExchange::RSA, AuthAlgorithm::RSA, EncryptionAlgorithm::RC4_128, MacAlgorithm::SHA1, SecurityLevel::Broken),
            CipherSuite::new(0x0004, "TLS_RSA_WITH_RC4_128_MD5", KeyExchange::RSA, AuthAlgorithm::RSA, EncryptionAlgorithm::RC4_128, MacAlgorithm::MD5, SecurityLevel::Broken),
            CipherSuite::new(0x000D, "TLS_DH_DSS_WITH_3DES_EDE_CBC_SHA", KeyExchange::DHE, AuthAlgorithm::DSS, EncryptionAlgorithm::TripleDESCBC, MacAlgorithm::SHA1, SecurityLevel::Weak),
            CipherSuite::new(0x0009, "TLS_RSA_WITH_DES_CBC_SHA", KeyExchange::RSA, AuthAlgorithm::RSA, EncryptionAlgorithm::DESCBC, MacAlgorithm::SHA1, SecurityLevel::Broken),

            // NULL cipher suites
            CipherSuite::new(0x0000, "TLS_NULL_WITH_NULL_NULL", KeyExchange::NULL, AuthAlgorithm::NULL, EncryptionAlgorithm::NULL, MacAlgorithm::NULL, SecurityLevel::Broken),
            CipherSuite::new(0x0001, "TLS_RSA_WITH_NULL_MD5", KeyExchange::RSA, AuthAlgorithm::RSA, EncryptionAlgorithm::NULL, MacAlgorithm::MD5, SecurityLevel::Broken),
            CipherSuite::new(0x0002, "TLS_RSA_WITH_NULL_SHA", KeyExchange::RSA, AuthAlgorithm::RSA, EncryptionAlgorithm::NULL, MacAlgorithm::SHA1, SecurityLevel::Broken),
            CipherSuite::new(0x003B, "TLS_RSA_WITH_NULL_SHA256", KeyExchange::RSA, AuthAlgorithm::RSA, EncryptionAlgorithm::NULL, MacAlgorithm::SHA256, SecurityLevel::Broken),

            // ECDHE anon (no auth)
            CipherSuite::new(0xC018, "TLS_ECDH_anon_WITH_RC4_128_SHA", KeyExchange::ECDHE, AuthAlgorithm::Anon, EncryptionAlgorithm::RC4_128, MacAlgorithm::SHA1, SecurityLevel::Broken),
            CipherSuite::new(0xC019, "TLS_ECDH_anon_WITH_3DES_EDE_CBC_SHA", KeyExchange::ECDHE, AuthAlgorithm::Anon, EncryptionAlgorithm::TripleDESCBC, MacAlgorithm::SHA1, SecurityLevel::Broken),

            // Camellia suites
            CipherSuite::new(0x00BE, "TLS_DHE_RSA_WITH_CAMELLIA_128_CBC_SHA256", KeyExchange::DHE, AuthAlgorithm::RSA, EncryptionAlgorithm::Camellia128CBC, MacAlgorithm::SHA256, SecurityLevel::Standard),
            CipherSuite::new(0x00C4, "TLS_DHE_RSA_WITH_CAMELLIA_256_CBC_SHA256", KeyExchange::DHE, AuthAlgorithm::RSA, EncryptionAlgorithm::Camellia256CBC, MacAlgorithm::SHA256, SecurityLevel::Standard),
            CipherSuite::new(0xC07C, "TLS_DHE_RSA_WITH_CAMELLIA_128_GCM_SHA256", KeyExchange::DHE, AuthAlgorithm::RSA, EncryptionAlgorithm::Camellia128GCM, MacAlgorithm::AEAD, SecurityLevel::High),
            CipherSuite::new(0xC07D, "TLS_DHE_RSA_WITH_CAMELLIA_256_GCM_SHA384", KeyExchange::DHE, AuthAlgorithm::RSA, EncryptionAlgorithm::Camellia256GCM, MacAlgorithm::AEAD, SecurityLevel::High),

            // PSK suites
            CipherSuite::new(0x00A8, "TLS_PSK_WITH_AES_128_GCM_SHA256", KeyExchange::PSK, AuthAlgorithm::PSK, EncryptionAlgorithm::AES128GCM, MacAlgorithm::AEAD, SecurityLevel::Standard),
            CipherSuite::new(0x00A9, "TLS_PSK_WITH_AES_256_GCM_SHA384", KeyExchange::PSK, AuthAlgorithm::PSK, EncryptionAlgorithm::AES256GCM, MacAlgorithm::AEAD, SecurityLevel::Standard),
            CipherSuite::new(0x00AE, "TLS_PSK_WITH_AES_128_CBC_SHA256", KeyExchange::PSK, AuthAlgorithm::PSK, EncryptionAlgorithm::AES128CBC, MacAlgorithm::SHA256, SecurityLevel::Legacy),
            CipherSuite::new(0x008C, "TLS_PSK_WITH_AES_128_CBC_SHA", KeyExchange::PSK, AuthAlgorithm::PSK, EncryptionAlgorithm::AES128CBC, MacAlgorithm::SHA1, SecurityLevel::Legacy),

            // ARIA suites
            CipherSuite::new(0xC050, "TLS_ECDHE_ECDSA_WITH_ARIA_128_GCM_SHA256", KeyExchange::ECDHE, AuthAlgorithm::ECDSA, EncryptionAlgorithm::ARIA128GCM, MacAlgorithm::AEAD, SecurityLevel::High),
            CipherSuite::new(0xC051, "TLS_ECDHE_ECDSA_WITH_ARIA_256_GCM_SHA384", KeyExchange::ECDHE, AuthAlgorithm::ECDSA, EncryptionAlgorithm::ARIA256GCM, MacAlgorithm::AEAD, SecurityLevel::High),

            // SEED suite
            CipherSuite::new(0x009A, "TLS_DHE_RSA_WITH_SEED_CBC_SHA", KeyExchange::DHE, AuthAlgorithm::RSA, EncryptionAlgorithm::SEED_CBC, MacAlgorithm::SHA1, SecurityLevel::Legacy),
        ]
    }

    /// Look up a cipher suite by IANA identifier.
    pub fn lookup(iana_id: u16) -> Option<CipherSuite> {
        Self::all().into_iter().find(|cs| cs.iana_id == iana_id)
    }

    /// All cipher suites at or above the given security level.
    pub fn at_level(min_level: SecurityLevel) -> Vec<CipherSuite> {
        Self::all()
            .into_iter()
            .filter(|cs| cs.security_level >= min_level)
            .collect()
    }

    /// All cipher suites that provide forward secrecy.
    pub fn with_forward_secrecy() -> Vec<CipherSuite> {
        Self::all()
            .into_iter()
            .filter(|cs| cs.has_forward_secrecy())
            .collect()
    }

    /// All broken cipher suites (potential downgrade targets).
    pub fn broken() -> Vec<CipherSuite> {
        Self::all()
            .into_iter()
            .filter(|cs| cs.security_level == SecurityLevel::Broken)
            .collect()
    }
}

// ── Protocol Version ─────────────────────────────────────────────────────

/// A protocol version (TLS or SSH).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProtocolVersion {
    pub major: u8,
    pub minor: u8,
    pub protocol: ProtocolFamily,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProtocolFamily {
    TLS,
    SSH,
    DTLS,
}

impl ProtocolVersion {
    pub fn new(major: u8, minor: u8, protocol: ProtocolFamily) -> Self {
        ProtocolVersion { major, minor, protocol }
    }

    pub fn tls(major: u8, minor: u8) -> Self {
        Self::new(major, minor, ProtocolFamily::TLS)
    }

    pub fn ssh(major: u8, minor: u8) -> Self {
        Self::new(major, minor, ProtocolFamily::SSH)
    }

    pub fn dtls(major: u8, minor: u8) -> Self {
        Self::new(major, minor, ProtocolFamily::DTLS)
    }

    // Well-known versions
    pub fn tls10() -> Self { Self::tls(3, 1) }
    pub fn tls11() -> Self { Self::tls(3, 2) }
    pub fn tls12() -> Self { Self::tls(3, 3) }
    pub fn tls13() -> Self { Self::tls(3, 4) }
    pub fn ssl30() -> Self { Self::tls(3, 0) }
    pub fn ssh2()  -> Self { Self::ssh(2, 0) }
    pub fn dtls12() -> Self { Self::dtls(3, 3) }
    pub fn dtls10() -> Self { Self::dtls(3, 1) }

    // Associated-constant-style constructors for downstream use
    #[allow(non_upper_case_globals)]
    pub const Tls10: Self = ProtocolVersion { major: 3, minor: 1, protocol: ProtocolFamily::TLS };
    #[allow(non_upper_case_globals)]
    pub const Tls11: Self = ProtocolVersion { major: 3, minor: 2, protocol: ProtocolFamily::TLS };
    #[allow(non_upper_case_globals)]
    pub const Tls12: Self = ProtocolVersion { major: 3, minor: 3, protocol: ProtocolFamily::TLS };
    #[allow(non_upper_case_globals)]
    pub const Tls13: Self = ProtocolVersion { major: 3, minor: 4, protocol: ProtocolFamily::TLS };
    #[allow(non_upper_case_globals)]
    pub const Ssl30: Self = ProtocolVersion { major: 3, minor: 0, protocol: ProtocolFamily::TLS };
    #[allow(non_upper_case_globals)]
    pub const Ssh2: Self = ProtocolVersion { major: 2, minor: 0, protocol: ProtocolFamily::SSH };
    #[allow(non_upper_case_globals)]
    pub const Dtls10: Self = ProtocolVersion { major: 3, minor: 1, protocol: ProtocolFamily::DTLS };
    #[allow(non_upper_case_globals)]
    pub const Dtls12: Self = ProtocolVersion { major: 3, minor: 3, protocol: ProtocolFamily::DTLS };
    #[allow(non_upper_case_globals)]
    pub const Unknown: Self = ProtocolVersion { major: 0, minor: 0, protocol: ProtocolFamily::TLS };

    pub fn security_level(&self) -> SecurityLevel {
        match self.protocol {
            ProtocolFamily::TLS => match (self.major, self.minor) {
                (3, 4) => SecurityLevel::High,
                (3, 3) => SecurityLevel::Standard,
                (3, 2) => SecurityLevel::Weak,
                (3, 1) => SecurityLevel::Weak,
                (3, 0) => SecurityLevel::Broken,
                _ => SecurityLevel::Broken,
            },
            ProtocolFamily::SSH => match (self.major, self.minor) {
                (2, _) => SecurityLevel::Standard,
                (1, _) => SecurityLevel::Broken,
                _ => SecurityLevel::Broken,
            },
            ProtocolFamily::DTLS => match (self.major, self.minor) {
                (3, 3) => SecurityLevel::Standard,
                _ => SecurityLevel::Legacy,
            },
        }
    }

    pub fn is_downgrade_from(&self, other: &ProtocolVersion) -> bool {
        if self.protocol != other.protocol {
            return false;
        }
        self.security_level() < other.security_level()
    }

    /// Wire encoding as u16 (TLS style).
    pub fn to_wire(&self) -> u16 {
        ((self.major as u16) << 8) | (self.minor as u16)
    }

    pub fn from_wire(value: u16, protocol: ProtocolFamily) -> Self {
        ProtocolVersion {
            major: (value >> 8) as u8,
            minor: (value & 0xFF) as u8,
            protocol,
        }
    }
}

impl PartialOrd for ProtocolVersion {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if self.protocol != other.protocol {
            return None;
        }
        Some(self.cmp_same_family(other))
    }
}

impl Ord for ProtocolVersion {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Compare by protocol family first, then by version
        let family_ord = format!("{:?}", self.protocol).cmp(&format!("{:?}", other.protocol));
        if family_ord != std::cmp::Ordering::Equal {
            return family_ord;
        }
        self.cmp_same_family(other)
    }
}

impl ProtocolVersion {
    fn cmp_same_family(&self, other: &Self) -> std::cmp::Ordering {
        (self.major, self.minor).cmp(&(other.major, other.minor))
    }
}

impl fmt::Display for ProtocolVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.protocol {
            ProtocolFamily::TLS => match (self.major, self.minor) {
                (3, 0) => write!(f, "SSLv3"),
                (3, 1) => write!(f, "TLSv1.0"),
                (3, 2) => write!(f, "TLSv1.1"),
                (3, 3) => write!(f, "TLSv1.2"),
                (3, 4) => write!(f, "TLSv1.3"),
                _ => write!(f, "TLS({}.{})", self.major, self.minor),
            },
            ProtocolFamily::SSH => write!(f, "SSHv{}.{}", self.major, self.minor),
            ProtocolFamily::DTLS => write!(f, "DTLSv{}.{}", self.major, self.minor),
        }
    }
}

// ── Extension ────────────────────────────────────────────────────────────

/// A protocol extension (TLS extension, SSH extension).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Extension {
    pub id: u16,
    pub name: String,
    pub data: Vec<u8>,
    pub critical: bool,
    /// Alias for `critical` used by downstream crates.
    pub is_critical: bool,
}

impl Extension {
    pub fn new(id: u16, name: impl Into<String>, data: Vec<u8>, critical: bool) -> Self {
        Extension {
            id,
            name: name.into(),
            data,
            critical,
            is_critical: critical,
        }
    }

    pub fn data_len(&self) -> usize {
        self.data.len()
    }

    // Well-known TLS extensions
    pub fn server_name(name: &str) -> Self {
        Self::new(0x0000, "server_name", name.as_bytes().to_vec(), false)
    }

    pub fn supported_versions(versions: &[ProtocolVersion]) -> Self {
        let data: Vec<u8> = versions
            .iter()
            .flat_map(|v| v.to_wire().to_be_bytes())
            .collect();
        Self::new(0x002B, "supported_versions", data, true)
    }

    pub fn signature_algorithms(algs: &[u16]) -> Self {
        let data: Vec<u8> = algs.iter().flat_map(|a| a.to_be_bytes()).collect();
        Self::new(0x000D, "signature_algorithms", data, true)
    }

    pub fn supported_groups(groups: &[u16]) -> Self {
        let data: Vec<u8> = groups.iter().flat_map(|g| g.to_be_bytes()).collect();
        Self::new(0x000A, "supported_groups", data, false)
    }

    pub fn key_share(data: Vec<u8>) -> Self {
        Self::new(0x0033, "key_share", data, true)
    }

    pub fn renegotiation_info(data: Vec<u8>) -> Self {
        Self::new(0xFF01, "renegotiation_info", data, false)
    }
}

impl fmt::Display for Extension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ext:{}(0x{:04X}, {} bytes{})",
            self.name,
            self.id,
            self.data.len(),
            if self.critical { ", critical" } else { "" }
        )
    }
}

// ── Negotiation State ────────────────────────────────────────────────────

/// The state of a protocol negotiation in progress.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NegotiationState {
    pub phase: HandshakePhase,
    pub offered_ciphers: Vec<CipherSuite>,
    pub selected_cipher: Option<CipherSuite>,
    pub version: Option<ProtocolVersion>,
    pub extensions: Vec<Extension>,
    pub random_client: Option<[u8; 32]>,
    pub random_server: Option<[u8; 32]>,
    pub session_id: Option<Vec<u8>>,
    /// Whether this negotiation is a session resumption.
    pub is_resumption: bool,
}

impl NegotiationState {
    pub fn initial() -> Self {
        NegotiationState {
            phase: HandshakePhase::Init,
            offered_ciphers: Vec::new(),
            selected_cipher: None,
            version: None,
            extensions: Vec::new(),
            random_client: None,
            random_server: None,
            session_id: None,
            is_resumption: false,
        }
    }

    /// Alias for `initial()` used by some downstream crates.
    pub fn new() -> Self {
        Self::initial()
    }

    /// Check whether the negotiation is complete (success or failure).
    pub fn is_terminal(&self) -> bool {
        self.phase.is_terminal()
    }

    /// Check whether a cipher suite was offered.
    pub fn was_offered(&self, iana_id: u16) -> bool {
        self.offered_ciphers.iter().any(|cs| cs.iana_id == iana_id)
    }

    /// Check whether the selected cipher is weaker than the best offered.
    pub fn is_downgraded(&self) -> bool {
        if let (Some(selected), Some(best)) = (&self.selected_cipher, self.best_offered()) {
            selected.security_level < best.security_level
        } else {
            false
        }
    }

    /// Returns the strongest offered cipher suite.
    pub fn best_offered(&self) -> Option<&CipherSuite> {
        self.offered_ciphers.iter().max()
    }

    /// Returns the weakest offered cipher suite.
    pub fn worst_offered(&self) -> Option<&CipherSuite> {
        self.offered_ciphers.iter().min()
    }

    /// Check if a particular extension is present.
    pub fn has_extension(&self, id: u16) -> bool {
        self.extensions.iter().any(|e| e.id == id)
    }
}

// ── Negotiation Outcome ──────────────────────────────────────────────────

/// The final outcome of a completed negotiation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NegotiationOutcome {
    pub selected_cipher: CipherSuite,
    pub version: ProtocolVersion,
    pub extensions: Vec<Extension>,
    pub session_resumed: bool,
}

impl NegotiationOutcome {
    pub fn security_level(&self) -> SecurityLevel {
        std::cmp::min(
            self.selected_cipher.security_level,
            self.version.security_level(),
        )
    }

    pub fn has_forward_secrecy(&self) -> bool {
        self.selected_cipher.has_forward_secrecy()
    }

    pub fn has_aead(&self) -> bool {
        self.selected_cipher.has_aead()
    }
}

impl fmt::Display for NegotiationOutcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} with {} ({} extensions{})",
            self.version,
            self.selected_cipher.name,
            self.extensions.len(),
            if self.session_resumed { ", resumed" } else { "" }
        )
    }
}

// ── Transition Labels ────────────────────────────────────────────────────

/// Labels on transitions in the negotiation LTS.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TransitionLabel {
    /// Client-initiated action.
    ClientAction(ClientActionKind),
    /// Server-initiated action.
    ServerAction(ServerActionKind),
    /// Adversary-initiated action.
    AdversaryAction(AdversaryActionKind),
    /// Internal / tau transition.
    Tau,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ClientActionKind {
    SendClientHello { ciphers: Vec<u16>, version: u16 },
    SendCertificate,
    SendKeyExchange { group: u16 },
    SendChangeCipherSpec,
    SendFinished,
    SendAlert { level: u8, desc: u8 },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ServerActionKind {
    SendServerHello { cipher: u16, version: u16 },
    SendCertificate,
    SendKeyExchange,
    SendHelloRetryRequest { cipher: u16 },
    SendChangeCipherSpec,
    SendFinished,
    SendAlert { level: u8, desc: u8 },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AdversaryActionKind {
    Drop,
    Intercept,
    Inject { data: Vec<u8> },
    Modify { field: String, old_value: Vec<u8>, new_value: Vec<u8> },
    Replay { message_index: usize },
    Reorder { from: usize, to: usize },
}

impl TransitionLabel {
    pub fn is_adversary(&self) -> bool {
        matches!(self, TransitionLabel::AdversaryAction(_))
    }

    pub fn is_client(&self) -> bool {
        matches!(self, TransitionLabel::ClientAction(_))
    }

    pub fn is_server(&self) -> bool {
        matches!(self, TransitionLabel::ServerAction(_))
    }
}

impl fmt::Display for TransitionLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TransitionLabel::ClientAction(a) => write!(f, "C:{:?}", a),
            TransitionLabel::ServerAction(a) => write!(f, "S:{:?}", a),
            TransitionLabel::AdversaryAction(a) => write!(f, "A:{:?}", a),
            TransitionLabel::Tau => write!(f, "τ"),
        }
    }
}

// ── Negotiation LTS (Definition D1) ─────────────────────────────────────

/// Labelled Transition System for protocol negotiation (Definition D1).
///
/// An LTS is a tuple (S, A, →, s₀, O) where:
/// - S is a set of states
/// - A is a set of actions (transition labels)
/// - → ⊆ S × A × S is the transition relation
/// - s₀ is the initial state
/// - O is the observation function
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NegotiationLTS {
    pub states: Vec<NegotiationState>,
    pub transitions: Vec<(usize, TransitionLabel, usize)>,
    pub initial_state: usize,
    observations: HashMap<usize, NegotiationOutcome>,
}

impl NegotiationLTS {
    pub fn new(initial: NegotiationState) -> Self {
        NegotiationLTS {
            states: vec![initial],
            transitions: Vec::new(),
            initial_state: 0,
            observations: HashMap::new(),
        }
    }

    /// Add a new state, returning its index.
    pub fn add_state(&mut self, state: NegotiationState) -> usize {
        let idx = self.states.len();
        self.states.push(state);
        idx
    }

    /// Add a transition between states.
    pub fn add_transition(&mut self, from: usize, label: TransitionLabel, to: usize) -> bool {
        if from >= self.states.len() || to >= self.states.len() {
            return false;
        }
        self.transitions.push((from, label, to));
        true
    }

    /// Set the observation (outcome) for a terminal state.
    pub fn set_observation(&mut self, state_idx: usize, outcome: NegotiationOutcome) {
        self.observations.insert(state_idx, outcome);
    }

    /// Get the observation for a state.
    pub fn observation(&self, state_idx: usize) -> Option<&NegotiationOutcome> {
        self.observations.get(&state_idx)
    }

    /// Returns all states reachable from the initial state.
    pub fn reachable_states(&self) -> BTreeSet<usize> {
        let mut visited = BTreeSet::new();
        let mut stack = vec![self.initial_state];
        while let Some(s) = stack.pop() {
            if visited.insert(s) {
                for (from, _, to) in &self.transitions {
                    if *from == s && !visited.contains(to) {
                        stack.push(*to);
                    }
                }
            }
        }
        visited
    }

    /// All transitions from a given state.
    pub fn transitions_from(&self, state: usize) -> Vec<&(usize, TransitionLabel, usize)> {
        self.transitions.iter().filter(|(f, _, _)| *f == state).collect()
    }

    /// All transitions that involve adversary actions.
    pub fn adversary_transitions(&self) -> Vec<&(usize, TransitionLabel, usize)> {
        self.transitions
            .iter()
            .filter(|(_, l, _)| l.is_adversary())
            .collect()
    }

    /// Terminal (Done/Abort) states.
    pub fn terminal_states(&self) -> Vec<usize> {
        self.states
            .iter()
            .enumerate()
            .filter(|(_, s)| s.is_terminal())
            .map(|(i, _)| i)
            .collect()
    }

    /// Count of states and transitions.
    pub fn size(&self) -> (usize, usize) {
        (self.states.len(), self.transitions.len())
    }

    /// Whether the LTS contains any adversary transitions.
    pub fn has_adversary_actions(&self) -> bool {
        self.transitions.iter().any(|(_, l, _)| l.is_adversary())
    }

    /// Find all paths from initial to a terminal state via adversary actions
    /// that result in a downgrade.
    pub fn find_downgrade_paths(&self, max_depth: usize) -> Vec<Vec<(usize, TransitionLabel, usize)>> {
        let mut results = Vec::new();
        let mut stack: Vec<(usize, Vec<(usize, TransitionLabel, usize)>)> =
            vec![(self.initial_state, Vec::new())];

        while let Some((current, path)) = stack.pop() {
            if path.len() > max_depth {
                continue;
            }
            let state = &self.states[current];
            if state.is_terminal() && state.is_downgraded() {
                results.push(path.clone());
                continue;
            }
            for (from, label, to) in &self.transitions {
                if *from == current {
                    let mut new_path = path.clone();
                    new_path.push((*from, label.clone(), *to));
                    stack.push((*to, new_path));
                }
            }
        }
        results
    }
}

impl fmt::Display for NegotiationLTS {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "NegotiationLTS({} states, {} transitions, initial={})",
            self.states.len(),
            self.transitions.len(),
            self.initial_state
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_handshake_phase_transitions() {
        assert!(HandshakePhase::Init.can_transition_to(HandshakePhase::ClientHelloSent));
        assert!(!HandshakePhase::Init.can_transition_to(HandshakePhase::Done));
        assert!(HandshakePhase::Done.is_terminal());
        assert!(HandshakePhase::Abort.is_terminal());
        assert!(!HandshakePhase::Init.is_terminal());
    }

    #[test]
    fn test_security_level_ordering() {
        assert!(SecurityLevel::Broken < SecurityLevel::Weak);
        assert!(SecurityLevel::Weak < SecurityLevel::Legacy);
        assert!(SecurityLevel::Legacy < SecurityLevel::Standard);
        assert!(SecurityLevel::Standard < SecurityLevel::High);
        assert!(!SecurityLevel::Broken.is_secure());
        assert!(SecurityLevel::High.is_secure());
    }

    #[test]
    fn test_cipher_suite_registry() {
        let all = CipherSuiteRegistry::all();
        assert!(all.len() >= 50, "Expected at least 50 cipher suites, got {}", all.len());

        let tls13 = CipherSuiteRegistry::lookup(0x1301);
        assert!(tls13.is_some());
        let tls13 = tls13.unwrap();
        assert_eq!(tls13.security_level, SecurityLevel::High);
        assert!(tls13.has_aead());
    }

    #[test]
    fn test_cipher_suite_ordering() {
        let strong = CipherSuiteRegistry::lookup(0x1302).unwrap(); // AES_256_GCM
        let weak = CipherSuiteRegistry::lookup(0x0005).unwrap();   // RC4_128_SHA
        assert!(strong > weak);
        assert!(weak.is_downgrade_from(&strong));
    }

    #[test]
    fn test_cipher_suite_security_score() {
        let high = CipherSuiteRegistry::lookup(0x1302).unwrap();
        let broken = CipherSuiteRegistry::lookup(0x0000).unwrap();
        assert!(high.security_score() > broken.security_score());
    }

    #[test]
    fn test_forward_secrecy() {
        let fs_suites = CipherSuiteRegistry::with_forward_secrecy();
        for cs in &fs_suites {
            assert!(cs.has_forward_secrecy());
        }
        let rsa = CipherSuiteRegistry::lookup(0x009C).unwrap();
        assert!(!rsa.has_forward_secrecy());
    }

    #[test]
    fn test_protocol_version() {
        let tls12 = ProtocolVersion::tls12();
        let tls13 = ProtocolVersion::tls13();
        assert!(tls12 < tls13);
        assert_eq!(tls12.to_wire(), 0x0303);
        assert_eq!(tls13.to_wire(), 0x0304);
        assert_eq!(format!("{}", tls12), "TLSv1.2");
        assert_eq!(format!("{}", tls13), "TLSv1.3");
    }

    #[test]
    fn test_protocol_version_security() {
        assert_eq!(ProtocolVersion::tls13().security_level(), SecurityLevel::High);
        assert_eq!(ProtocolVersion::ssl30().security_level(), SecurityLevel::Broken);
        assert!(ProtocolVersion::tls11().is_downgrade_from(&ProtocolVersion::tls13()));
    }

    #[test]
    fn test_version_cross_family() {
        let tls = ProtocolVersion::tls12();
        let ssh = ProtocolVersion::ssh2();
        assert_eq!(tls.partial_cmp(&ssh), None);
        assert!(!tls.is_downgrade_from(&ssh));
    }

    #[test]
    fn test_extension() {
        let sni = Extension::server_name("example.com");
        assert_eq!(sni.id, 0x0000);
        assert!(!sni.critical);
        assert_eq!(sni.data, b"example.com");

        let sv = Extension::supported_versions(&[ProtocolVersion::tls13(), ProtocolVersion::tls12()]);
        assert!(sv.critical);
        assert_eq!(sv.data.len(), 4);
    }

    #[test]
    fn test_negotiation_state() {
        let mut state = NegotiationState::initial();
        assert!(!state.is_terminal());

        state.offered_ciphers.push(CipherSuiteRegistry::lookup(0x1301).unwrap());
        state.offered_ciphers.push(CipherSuiteRegistry::lookup(0x0005).unwrap());
        assert!(state.was_offered(0x1301));
        assert!(!state.was_offered(0xFFFF));

        state.selected_cipher = Some(CipherSuiteRegistry::lookup(0x0005).unwrap());
        assert!(state.is_downgraded());
    }

    #[test]
    fn test_negotiation_lts() {
        let init = NegotiationState::initial();
        let mut lts = NegotiationLTS::new(init);
        assert_eq!(lts.size(), (1, 0));

        let mut hello_sent = NegotiationState::initial();
        hello_sent.phase = HandshakePhase::ClientHelloSent;
        let s1 = lts.add_state(hello_sent);

        let label = TransitionLabel::ClientAction(ClientActionKind::SendClientHello {
            ciphers: vec![0x1301],
            version: 0x0304,
        });
        assert!(lts.add_transition(0, label, s1));
        assert_eq!(lts.size(), (2, 1));

        let reachable = lts.reachable_states();
        assert!(reachable.contains(&0));
        assert!(reachable.contains(&1));
    }

    #[test]
    fn test_transition_label_classification() {
        let client = TransitionLabel::ClientAction(ClientActionKind::SendFinished);
        assert!(client.is_client());
        assert!(!client.is_server());
        assert!(!client.is_adversary());

        let adv = TransitionLabel::AdversaryAction(AdversaryActionKind::Drop);
        assert!(adv.is_adversary());
    }

    #[test]
    fn test_negotiation_outcome() {
        let outcome = NegotiationOutcome {
            selected_cipher: CipherSuiteRegistry::lookup(0x1301).unwrap(),
            version: ProtocolVersion::tls13(),
            extensions: vec![],
            session_resumed: false,
        };
        assert_eq!(outcome.security_level(), SecurityLevel::High);
        assert!(outcome.has_forward_secrecy());
        assert!(outcome.has_aead());
    }

    #[test]
    fn test_version_from_wire() {
        let v = ProtocolVersion::from_wire(0x0303, ProtocolFamily::TLS);
        assert_eq!(v, ProtocolVersion::tls12());

        let v = ProtocolVersion::from_wire(0x0304, ProtocolFamily::TLS);
        assert_eq!(v, ProtocolVersion::tls13());
    }

    #[test]
    fn test_broken_cipher_suites() {
        let broken = CipherSuiteRegistry::broken();
        assert!(!broken.is_empty());
        for cs in &broken {
            assert_eq!(cs.security_level, SecurityLevel::Broken);
        }
    }

    #[test]
    fn test_encryption_properties() {
        assert!(EncryptionAlgorithm::AES256GCM.is_aead());
        assert!(!EncryptionAlgorithm::AES128CBC.is_aead());
        assert_eq!(EncryptionAlgorithm::AES256GCM.key_bits(), 256);
        assert_eq!(EncryptionAlgorithm::TripleDESCBC.key_bits(), 168);
    }
}
