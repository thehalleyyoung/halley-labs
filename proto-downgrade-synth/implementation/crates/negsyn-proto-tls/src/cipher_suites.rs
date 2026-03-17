//! TLS cipher suite definitions, IANA registry entries, and negotiation logic.
//!
//! Provides complete cipher suite decomposition by key exchange, authentication,
//! bulk encryption, and MAC algorithm. Includes security classification,
//! export cipher detection, and suite selection algorithms.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// Component algorithm enums
// ---------------------------------------------------------------------------

/// Key exchange algorithm used in the cipher suite.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum KeyExchange {
    RSA,
    DHE_RSA,
    DHE_DSS,
    ECDHE_RSA,
    ECDHE_ECDSA,
    DH_anon,
    ECDH_anon,
    PSK,
    DHE_PSK,
    RSA_PSK,
    ECDHE_PSK,
    RSA_EXPORT,
    DH_DSS_EXPORT,
    DH_RSA_EXPORT,
    DHE_DSS_EXPORT,
    DHE_RSA_EXPORT,
    DH_DSS,
    DH_RSA,
    ECDH_RSA,
    ECDH_ECDSA,
    SRP_SHA,
    SRP_SHA_RSA,
    SRP_SHA_DSS,
    NULL,
    TLS13,
}

impl KeyExchange {
    /// Whether this key exchange is an export-grade algorithm.
    pub fn is_export(&self) -> bool {
        matches!(
            self,
            Self::RSA_EXPORT
                | Self::DH_DSS_EXPORT
                | Self::DH_RSA_EXPORT
                | Self::DHE_DSS_EXPORT
                | Self::DHE_RSA_EXPORT
        )
    }

    /// Whether this provides forward secrecy.
    pub fn has_forward_secrecy(&self) -> bool {
        matches!(
            self,
            Self::DHE_RSA
                | Self::DHE_DSS
                | Self::ECDHE_RSA
                | Self::ECDHE_ECDSA
                | Self::DHE_PSK
                | Self::ECDHE_PSK
                | Self::TLS13
        )
    }

    /// Security strength rating (0 = none, 4 = best).
    pub fn strength(&self) -> u32 {
        match self {
            Self::NULL => 0,
            Self::RSA_EXPORT | Self::DH_DSS_EXPORT | Self::DH_RSA_EXPORT
            | Self::DHE_DSS_EXPORT | Self::DHE_RSA_EXPORT => 0,
            Self::DH_anon | Self::ECDH_anon => 1,
            Self::RSA | Self::DH_DSS | Self::DH_RSA | Self::PSK | Self::RSA_PSK => 2,
            Self::DHE_RSA | Self::DHE_DSS | Self::DHE_PSK
            | Self::SRP_SHA | Self::SRP_SHA_RSA | Self::SRP_SHA_DSS => 3,
            Self::ECDHE_RSA | Self::ECDHE_ECDSA | Self::ECDHE_PSK
            | Self::ECDH_RSA | Self::ECDH_ECDSA | Self::TLS13 => 4,
        }
    }
}

impl fmt::Display for KeyExchange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// Authentication algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Authentication {
    RSA,
    DSS,
    ECDSA,
    Anonymous,
    PSK,
    NULL,
}

impl Authentication {
    pub fn is_anonymous(&self) -> bool {
        matches!(self, Self::Anonymous | Self::NULL)
    }

    pub fn strength(&self) -> u32 {
        match self {
            Self::NULL | Self::Anonymous => 0,
            Self::PSK => 2,
            Self::RSA | Self::DSS => 3,
            Self::ECDSA => 4,
        }
    }
}

impl fmt::Display for Authentication {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// Bulk encryption algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum BulkEncryption {
    NULL,
    RC4_40,
    RC4_128,
    DES40_CBC,
    DES_CBC,
    DES_EDE3_CBC,
    IDEA_CBC,
    AES_128_CBC,
    AES_256_CBC,
    AES_128_GCM,
    AES_256_GCM,
    AES_128_CCM,
    AES_256_CCM,
    AES_128_CCM_8,
    CAMELLIA_128_CBC,
    CAMELLIA_256_CBC,
    CAMELLIA_128_GCM,
    CAMELLIA_256_GCM,
    SEED_CBC,
    ARIA_128_CBC,
    ARIA_256_CBC,
    ARIA_128_GCM,
    ARIA_256_GCM,
    CHACHA20_POLY1305,
}

impl BulkEncryption {
    /// Whether this is null / no encryption.
    pub fn is_null(&self) -> bool {
        matches!(self, Self::NULL)
    }

    /// Whether this is an export-grade cipher (40-bit keys).
    pub fn is_export_grade(&self) -> bool {
        matches!(self, Self::RC4_40 | Self::DES40_CBC)
    }

    /// Whether this is an AEAD cipher.
    pub fn is_aead(&self) -> bool {
        matches!(
            self,
            Self::AES_128_GCM
                | Self::AES_256_GCM
                | Self::AES_128_CCM
                | Self::AES_256_CCM
                | Self::AES_128_CCM_8
                | Self::CHACHA20_POLY1305
                | Self::CAMELLIA_128_GCM
                | Self::CAMELLIA_256_GCM
                | Self::ARIA_128_GCM
                | Self::ARIA_256_GCM
        )
    }

    /// Whether this cipher is considered weak.
    pub fn is_weak(&self) -> bool {
        matches!(
            self,
            Self::NULL
                | Self::RC4_40
                | Self::RC4_128
                | Self::DES40_CBC
                | Self::DES_CBC
                | Self::DES_EDE3_CBC
                | Self::IDEA_CBC
        )
    }

    /// Effective key length in bits.
    pub fn key_bits(&self) -> u32 {
        match self {
            Self::NULL => 0,
            Self::RC4_40 | Self::DES40_CBC => 40,
            Self::DES_CBC => 56,
            Self::RC4_128 => 128,
            Self::DES_EDE3_CBC => 168,
            Self::IDEA_CBC => 128,
            Self::AES_128_CBC | Self::AES_128_GCM | Self::AES_128_CCM | Self::AES_128_CCM_8 => 128,
            Self::AES_256_CBC | Self::AES_256_GCM | Self::AES_256_CCM => 256,
            Self::CAMELLIA_128_CBC | Self::CAMELLIA_128_GCM => 128,
            Self::CAMELLIA_256_CBC | Self::CAMELLIA_256_GCM => 256,
            Self::SEED_CBC => 128,
            Self::ARIA_128_CBC | Self::ARIA_128_GCM => 128,
            Self::ARIA_256_CBC | Self::ARIA_256_GCM => 256,
            Self::CHACHA20_POLY1305 => 256,
        }
    }

    /// Security strength rating.
    pub fn strength(&self) -> u32 {
        match self.key_bits() {
            0 => 0,
            40 => 1,
            56 => 1,
            128 if self.is_aead() => 4,
            128 => 3,
            168 => 2,
            256 if self.is_aead() => 5,
            256 => 4,
            _ => 2,
        }
    }
}

impl fmt::Display for BulkEncryption {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// MAC algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MacAlgorithm {
    NULL,
    MD5,
    SHA1,
    SHA256,
    SHA384,
    AEAD,
}

impl MacAlgorithm {
    pub fn is_aead(&self) -> bool {
        matches!(self, Self::AEAD)
    }

    pub fn hash_length(&self) -> usize {
        match self {
            Self::NULL => 0,
            Self::MD5 => 16,
            Self::SHA1 => 20,
            Self::SHA256 => 32,
            Self::SHA384 => 48,
            Self::AEAD => 0,
        }
    }

    pub fn strength(&self) -> u32 {
        match self {
            Self::NULL => 0,
            Self::MD5 => 1,
            Self::SHA1 => 2,
            Self::SHA256 | Self::AEAD => 3,
            Self::SHA384 => 4,
        }
    }
}

impl fmt::Display for MacAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

// ---------------------------------------------------------------------------
// Security level
// ---------------------------------------------------------------------------

/// Overall security classification for a cipher suite.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SecurityLevel {
    /// No security (NULL ciphers, anonymous key exchange).
    Insecure,
    /// Export-grade (40-bit keys).
    Export,
    /// Weak but not export (DES, RC4, 3DES).
    Weak,
    /// Legacy acceptable (CBC mode AES without AEAD).
    Legacy,
    /// Modern and secure (AEAD ciphers with forward secrecy).
    Secure,
    /// Best practice (TLS 1.3 suites, ECDHE + AEAD).
    Recommended,
}

impl SecurityLevel {
    pub fn numeric(&self) -> u32 {
        match self {
            Self::Insecure => 0,
            Self::Export => 1,
            Self::Weak => 2,
            Self::Legacy => 3,
            Self::Secure => 4,
            Self::Recommended => 5,
        }
    }
}

impl fmt::Display for SecurityLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

// ---------------------------------------------------------------------------
// TlsCipherSuite
// ---------------------------------------------------------------------------

/// A fully-decomposed TLS cipher suite.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TlsCipherSuite {
    /// IANA-assigned 16-bit identifier.
    pub id: u16,
    /// IANA name (e.g., "TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256").
    pub name: &'static str,
    /// Key exchange algorithm.
    pub key_exchange: KeyExchange,
    /// Authentication algorithm.
    pub authentication: Authentication,
    /// Bulk encryption algorithm.
    pub encryption: BulkEncryption,
    /// MAC algorithm.
    pub mac: MacAlgorithm,
    /// Minimum TLS version required.
    pub min_version: Option<crate::version::TlsVersion>,
    /// Maximum TLS version (None = no limit).
    pub max_version: Option<crate::version::TlsVersion>,
}

impl TlsCipherSuite {
    pub const fn new(
        id: u16,
        name: &'static str,
        kx: KeyExchange,
        auth: Authentication,
        enc: BulkEncryption,
        mac: MacAlgorithm,
    ) -> Self {
        Self {
            id,
            name,
            key_exchange: kx,
            authentication: auth,
            encryption: enc,
            mac,
            min_version: None,
            max_version: None,
        }
    }

    /// Classify the overall security level.
    pub fn security_level(&self) -> SecurityLevel {
        if self.encryption.is_null() || self.authentication.is_anonymous() {
            return SecurityLevel::Insecure;
        }
        if self.key_exchange.is_export() || self.encryption.is_export_grade() {
            return SecurityLevel::Export;
        }
        if self.encryption.is_weak() {
            return SecurityLevel::Weak;
        }
        if self.encryption.is_aead() && self.key_exchange.has_forward_secrecy() {
            if matches!(self.key_exchange, KeyExchange::TLS13) {
                return SecurityLevel::Recommended;
            }
            return SecurityLevel::Secure;
        }
        if self.encryption.is_aead() {
            return SecurityLevel::Secure;
        }
        SecurityLevel::Legacy
    }

    /// Whether this is an export cipher suite.
    pub fn is_export(&self) -> bool {
        self.key_exchange.is_export() || self.encryption.is_export_grade()
    }

    /// Whether this suite is considered weak.
    pub fn is_weak(&self) -> bool {
        let level = self.security_level();
        matches!(level, SecurityLevel::Insecure | SecurityLevel::Export | SecurityLevel::Weak)
    }

    /// Whether this provides forward secrecy.
    pub fn has_forward_secrecy(&self) -> bool {
        self.key_exchange.has_forward_secrecy()
    }

    /// Whether this is a TLS 1.3 cipher suite.
    pub fn is_tls13(&self) -> bool {
        matches!(self.key_exchange, KeyExchange::TLS13)
    }

    /// Whether this suite is FIPS 140-2 approved.
    pub fn is_fips_approved(&self) -> bool {
        let enc_ok = matches!(
            self.encryption,
            BulkEncryption::AES_128_CBC
                | BulkEncryption::AES_256_CBC
                | BulkEncryption::AES_128_GCM
                | BulkEncryption::AES_256_GCM
                | BulkEncryption::AES_128_CCM
                | BulkEncryption::AES_256_CCM
                | BulkEncryption::DES_EDE3_CBC
        );
        let mac_ok = matches!(
            self.mac,
            MacAlgorithm::SHA1 | MacAlgorithm::SHA256 | MacAlgorithm::SHA384 | MacAlgorithm::AEAD
        );
        enc_ok && mac_ok && !self.key_exchange.is_export()
    }

    /// Effective key length in bits.
    pub fn effective_key_bits(&self) -> u32 {
        self.encryption.key_bits()
    }

    /// Composite security score (higher is better).
    pub fn security_score(&self) -> u32 {
        self.key_exchange.strength()
            + self.authentication.strength()
            + self.encryption.strength()
            + self.mac.strength()
    }

    /// Convert to the negsyn_types CipherSuite.
    pub fn to_negsyn_cipher_suite(&self) -> negsyn_types::CipherSuite {
        use negsyn_types::protocol::{
            AuthAlgorithm as NAuth, EncryptionAlgorithm as NEnc, KeyExchange as NKx,
            MacAlgorithm as NMac,
        };
        let kx = match self.key_exchange {
            KeyExchange::RSA | KeyExchange::RSA_EXPORT => NKx::RSA,
            KeyExchange::DHE_RSA | KeyExchange::DHE_DSS
            | KeyExchange::DHE_RSA_EXPORT | KeyExchange::DHE_DSS_EXPORT
            | KeyExchange::DH_DSS | KeyExchange::DH_RSA
            | KeyExchange::DH_DSS_EXPORT | KeyExchange::DH_RSA_EXPORT
            | KeyExchange::DH_anon => NKx::DHE,
            KeyExchange::ECDHE_RSA | KeyExchange::ECDHE_ECDSA
            | KeyExchange::ECDH_RSA | KeyExchange::ECDH_ECDSA
            | KeyExchange::ECDH_anon => NKx::ECDHE,
            KeyExchange::PSK | KeyExchange::RSA_PSK => NKx::PSK,
            KeyExchange::DHE_PSK => NKx::DHEPSK,
            KeyExchange::ECDHE_PSK => NKx::ECDHEPSK,
            KeyExchange::SRP_SHA | KeyExchange::SRP_SHA_RSA | KeyExchange::SRP_SHA_DSS => {
                NKx::SRP
            }
            KeyExchange::NULL => NKx::NULL,
            KeyExchange::TLS13 => NKx::ECDHE,
        };
        let auth = match self.authentication {
            Authentication::RSA => NAuth::RSA,
            Authentication::DSS => NAuth::DSS,
            Authentication::ECDSA => NAuth::ECDSA,
            Authentication::PSK => NAuth::PSK,
            Authentication::Anonymous => NAuth::Anon,
            Authentication::NULL => NAuth::NULL,
        };
        let enc = match self.encryption {
            BulkEncryption::NULL => NEnc::NULL,
            BulkEncryption::RC4_40 | BulkEncryption::RC4_128 => NEnc::RC4_128,
            BulkEncryption::DES40_CBC | BulkEncryption::DES_CBC => NEnc::DESCBC,
            BulkEncryption::DES_EDE3_CBC => NEnc::TripleDESCBC,
            BulkEncryption::AES_128_CBC => NEnc::AES128CBC,
            BulkEncryption::AES_256_CBC => NEnc::AES256CBC,
            BulkEncryption::AES_128_GCM => NEnc::AES128GCM,
            BulkEncryption::AES_256_GCM => NEnc::AES256GCM,
            BulkEncryption::AES_128_CCM | BulkEncryption::AES_128_CCM_8 => NEnc::AES128CCM,
            BulkEncryption::AES_256_CCM => NEnc::AES256CCM,
            BulkEncryption::CHACHA20_POLY1305 => NEnc::ChaCha20Poly1305,
            BulkEncryption::CAMELLIA_128_CBC => NEnc::Camellia128CBC,
            BulkEncryption::CAMELLIA_256_CBC => NEnc::Camellia256CBC,
            BulkEncryption::CAMELLIA_128_GCM => NEnc::Camellia128GCM,
            BulkEncryption::CAMELLIA_256_GCM => NEnc::Camellia256GCM,
            BulkEncryption::SEED_CBC | BulkEncryption::IDEA_CBC => NEnc::SEED_CBC,
            BulkEncryption::ARIA_128_CBC | BulkEncryption::ARIA_128_GCM => NEnc::ARIA128GCM,
            BulkEncryption::ARIA_256_CBC | BulkEncryption::ARIA_256_GCM => NEnc::ARIA256GCM,
        };
        let mac = match self.mac {
            MacAlgorithm::NULL => NMac::NULL,
            MacAlgorithm::MD5 => NMac::MD5,
            MacAlgorithm::SHA1 => NMac::SHA1,
            MacAlgorithm::SHA256 => NMac::SHA256,
            MacAlgorithm::SHA384 => NMac::SHA384,
            MacAlgorithm::AEAD => NMac::AEAD,
        };
        let security_level = match self.security_level() {
            SecurityLevel::Insecure | SecurityLevel::Export => negsyn_types::SecurityLevel::Broken,
            SecurityLevel::Weak => negsyn_types::SecurityLevel::Weak,
            SecurityLevel::Legacy => negsyn_types::SecurityLevel::Legacy,
            SecurityLevel::Secure => negsyn_types::SecurityLevel::Standard,
            SecurityLevel::Recommended => negsyn_types::SecurityLevel::High,
        };
        negsyn_types::CipherSuite::new(
            self.id,
            self.name,
            kx,
            auth,
            enc,
            mac,
            security_level,
        )
    }
}

impl fmt::Display for TlsCipherSuite {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "0x{:04X} {}", self.id, self.name)
    }
}

impl PartialOrd for TlsCipherSuite {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.id.cmp(&other.id))
    }
}

impl Ord for TlsCipherSuite {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.id.cmp(&other.id)
    }
}

// ---------------------------------------------------------------------------
// IANA cipher suite registry (80+ entries)
// ---------------------------------------------------------------------------

macro_rules! suite {
    ($id:expr, $name:expr, $kx:ident, $auth:ident, $enc:ident, $mac:ident) => {
        TlsCipherSuite::new(
            $id,
            $name,
            KeyExchange::$kx,
            Authentication::$auth,
            BulkEncryption::$enc,
            MacAlgorithm::$mac,
        )
    };
}

/// All known cipher suites.
pub fn all_cipher_suites() -> Vec<TlsCipherSuite> {
    vec![
        // NULL
        suite!(0x0000, "TLS_NULL_WITH_NULL_NULL", NULL, NULL, NULL, NULL),
        // RSA
        suite!(0x0001, "TLS_RSA_WITH_NULL_MD5", RSA, RSA, NULL, MD5),
        suite!(0x0002, "TLS_RSA_WITH_NULL_SHA", RSA, RSA, NULL, SHA1),
        suite!(0x003B, "TLS_RSA_WITH_NULL_SHA256", RSA, RSA, NULL, SHA256),
        suite!(0x0003, "TLS_RSA_EXPORT_WITH_RC4_40_MD5", RSA_EXPORT, RSA, RC4_40, MD5),
        suite!(0x0004, "TLS_RSA_WITH_RC4_128_MD5", RSA, RSA, RC4_128, MD5),
        suite!(0x0005, "TLS_RSA_WITH_RC4_128_SHA", RSA, RSA, RC4_128, SHA1),
        suite!(0x0006, "TLS_RSA_EXPORT_WITH_RC2_CBC_40_MD5", RSA_EXPORT, RSA, DES40_CBC, MD5),
        suite!(0x0007, "TLS_RSA_WITH_IDEA_CBC_SHA", RSA, RSA, IDEA_CBC, SHA1),
        suite!(0x0008, "TLS_RSA_EXPORT_WITH_DES40_CBC_SHA", RSA_EXPORT, RSA, DES40_CBC, SHA1),
        suite!(0x0009, "TLS_RSA_WITH_DES_CBC_SHA", RSA, RSA, DES_CBC, SHA1),
        suite!(0x000A, "TLS_RSA_WITH_3DES_EDE_CBC_SHA", RSA, RSA, DES_EDE3_CBC, SHA1),
        // DHE DSS
        suite!(0x000D, "TLS_DH_DSS_WITH_3DES_EDE_CBC_SHA", DH_DSS, DSS, DES_EDE3_CBC, SHA1),
        suite!(0x000E, "TLS_DHE_DSS_EXPORT_WITH_DES40_CBC_SHA", DHE_DSS_EXPORT, DSS, DES40_CBC, SHA1),
        suite!(0x0010, "TLS_DH_RSA_WITH_3DES_EDE_CBC_SHA", DH_RSA, RSA, DES_EDE3_CBC, SHA1),
        suite!(0x0011, "TLS_DHE_DSS_WITH_DES_CBC_SHA", DHE_DSS, DSS, DES_CBC, SHA1),
        suite!(0x0012, "TLS_DHE_DSS_WITH_3DES_EDE_CBC_SHA", DHE_DSS, DSS, DES_EDE3_CBC, SHA1),
        suite!(0x0013, "TLS_DHE_RSA_EXPORT_WITH_DES40_CBC_SHA", DHE_RSA_EXPORT, RSA, DES40_CBC, SHA1),
        suite!(0x0014, "TLS_DHE_RSA_WITH_DES_CBC_SHA", DHE_RSA, RSA, DES_CBC, SHA1),
        suite!(0x0015, "TLS_DHE_RSA_WITH_3DES_EDE_CBC_SHA", DHE_RSA, RSA, DES_EDE3_CBC, SHA1),
        // DH anon
        suite!(0x0018, "TLS_DH_anon_WITH_RC4_128_MD5", DH_anon, Anonymous, RC4_128, MD5),
        suite!(0x001B, "TLS_DH_anon_WITH_3DES_EDE_CBC_SHA", DH_anon, Anonymous, DES_EDE3_CBC, SHA1),
        // AES CBC with RSA
        suite!(0x002F, "TLS_RSA_WITH_AES_128_CBC_SHA", RSA, RSA, AES_128_CBC, SHA1),
        suite!(0x0030, "TLS_DH_DSS_WITH_AES_128_CBC_SHA", DH_DSS, DSS, AES_128_CBC, SHA1),
        suite!(0x0031, "TLS_DH_RSA_WITH_AES_128_CBC_SHA", DH_RSA, RSA, AES_128_CBC, SHA1),
        suite!(0x0032, "TLS_DHE_DSS_WITH_AES_128_CBC_SHA", DHE_DSS, DSS, AES_128_CBC, SHA1),
        suite!(0x0033, "TLS_DHE_RSA_WITH_AES_128_CBC_SHA", DHE_RSA, RSA, AES_128_CBC, SHA1),
        suite!(0x0034, "TLS_DH_anon_WITH_AES_128_CBC_SHA", DH_anon, Anonymous, AES_128_CBC, SHA1),
        suite!(0x0035, "TLS_RSA_WITH_AES_256_CBC_SHA", RSA, RSA, AES_256_CBC, SHA1),
        suite!(0x0036, "TLS_DH_DSS_WITH_AES_256_CBC_SHA", DH_DSS, DSS, AES_256_CBC, SHA1),
        suite!(0x0037, "TLS_DH_RSA_WITH_AES_256_CBC_SHA", DH_RSA, RSA, AES_256_CBC, SHA1),
        suite!(0x0038, "TLS_DHE_DSS_WITH_AES_256_CBC_SHA", DHE_DSS, DSS, AES_256_CBC, SHA1),
        suite!(0x0039, "TLS_DHE_RSA_WITH_AES_256_CBC_SHA", DHE_RSA, RSA, AES_256_CBC, SHA1),
        suite!(0x003A, "TLS_DH_anon_WITH_AES_256_CBC_SHA", DH_anon, Anonymous, AES_256_CBC, SHA1),
        // AES CBC SHA256
        suite!(0x003C, "TLS_RSA_WITH_AES_128_CBC_SHA256", RSA, RSA, AES_128_CBC, SHA256),
        suite!(0x003D, "TLS_RSA_WITH_AES_256_CBC_SHA256", RSA, RSA, AES_256_CBC, SHA256),
        suite!(0x0040, "TLS_DH_DSS_WITH_AES_128_CBC_SHA256", DH_DSS, DSS, AES_128_CBC, SHA256),
        suite!(0x0067, "TLS_DHE_RSA_WITH_AES_128_CBC_SHA256", DHE_RSA, RSA, AES_128_CBC, SHA256),
        suite!(0x006B, "TLS_DHE_RSA_WITH_AES_256_CBC_SHA256", DHE_RSA, RSA, AES_256_CBC, SHA256),
        suite!(0x006A, "TLS_DH_DSS_WITH_AES_256_CBC_SHA256", DH_DSS, DSS, AES_256_CBC, SHA256),
        suite!(0x006C, "TLS_DH_anon_WITH_AES_128_CBC_SHA256", DH_anon, Anonymous, AES_128_CBC, SHA256),
        suite!(0x006D, "TLS_DH_anon_WITH_AES_256_CBC_SHA256", DH_anon, Anonymous, AES_256_CBC, SHA256),
        // Camellia
        suite!(0x0041, "TLS_RSA_WITH_CAMELLIA_128_CBC_SHA", RSA, RSA, CAMELLIA_128_CBC, SHA1),
        suite!(0x0045, "TLS_DHE_RSA_WITH_CAMELLIA_128_CBC_SHA", DHE_RSA, RSA, CAMELLIA_128_CBC, SHA1),
        suite!(0x0084, "TLS_RSA_WITH_CAMELLIA_256_CBC_SHA", RSA, RSA, CAMELLIA_256_CBC, SHA1),
        suite!(0x0088, "TLS_DHE_RSA_WITH_CAMELLIA_256_CBC_SHA", DHE_RSA, RSA, CAMELLIA_256_CBC, SHA1),
        // SEED
        suite!(0x0096, "TLS_RSA_WITH_SEED_CBC_SHA", RSA, RSA, SEED_CBC, SHA1),
        // AES GCM
        suite!(0x009C, "TLS_RSA_WITH_AES_128_GCM_SHA256", RSA, RSA, AES_128_GCM, AEAD),
        suite!(0x009D, "TLS_RSA_WITH_AES_256_GCM_SHA384", RSA, RSA, AES_256_GCM, AEAD),
        suite!(0x009E, "TLS_DHE_RSA_WITH_AES_128_GCM_SHA256", DHE_RSA, RSA, AES_128_GCM, AEAD),
        suite!(0x009F, "TLS_DHE_RSA_WITH_AES_256_GCM_SHA384", DHE_RSA, RSA, AES_256_GCM, AEAD),
        suite!(0x00A0, "TLS_DH_RSA_WITH_AES_128_GCM_SHA256", DH_RSA, RSA, AES_128_GCM, AEAD),
        suite!(0x00A1, "TLS_DH_RSA_WITH_AES_256_GCM_SHA384", DH_RSA, RSA, AES_256_GCM, AEAD),
        suite!(0x00A2, "TLS_DHE_DSS_WITH_AES_128_GCM_SHA256", DHE_DSS, DSS, AES_128_GCM, AEAD),
        suite!(0x00A3, "TLS_DHE_DSS_WITH_AES_256_GCM_SHA384", DHE_DSS, DSS, AES_256_GCM, AEAD),
        suite!(0x00A6, "TLS_DH_anon_WITH_AES_128_GCM_SHA256", DH_anon, Anonymous, AES_128_GCM, AEAD),
        suite!(0x00A7, "TLS_DH_anon_WITH_AES_256_GCM_SHA384", DH_anon, Anonymous, AES_256_GCM, AEAD),
        // PSK
        suite!(0x008A, "TLS_PSK_WITH_RC4_128_SHA", PSK, PSK, RC4_128, SHA1),
        suite!(0x008B, "TLS_PSK_WITH_3DES_EDE_CBC_SHA", PSK, PSK, DES_EDE3_CBC, SHA1),
        suite!(0x008C, "TLS_PSK_WITH_AES_128_CBC_SHA", PSK, PSK, AES_128_CBC, SHA1),
        suite!(0x008D, "TLS_PSK_WITH_AES_256_CBC_SHA", PSK, PSK, AES_256_CBC, SHA1),
        suite!(0x00A8, "TLS_PSK_WITH_AES_128_GCM_SHA256", PSK, PSK, AES_128_GCM, AEAD),
        suite!(0x00A9, "TLS_PSK_WITH_AES_256_GCM_SHA384", PSK, PSK, AES_256_GCM, AEAD),
        suite!(0x00AE, "TLS_PSK_WITH_AES_128_CBC_SHA256", PSK, PSK, AES_128_CBC, SHA256),
        suite!(0x00AF, "TLS_PSK_WITH_AES_256_CBC_SHA384", PSK, PSK, AES_256_CBC, SHA384),
        // DHE PSK
        suite!(0x0090, "TLS_DHE_PSK_WITH_AES_128_CBC_SHA", DHE_PSK, PSK, AES_128_CBC, SHA1),
        suite!(0x0091, "TLS_DHE_PSK_WITH_AES_256_CBC_SHA", DHE_PSK, PSK, AES_256_CBC, SHA1),
        suite!(0x00AA, "TLS_DHE_PSK_WITH_AES_128_GCM_SHA256", DHE_PSK, PSK, AES_128_GCM, AEAD),
        suite!(0x00AB, "TLS_DHE_PSK_WITH_AES_256_GCM_SHA384", DHE_PSK, PSK, AES_256_GCM, AEAD),
        // RSA PSK
        suite!(0x0094, "TLS_RSA_PSK_WITH_AES_128_CBC_SHA", RSA_PSK, PSK, AES_128_CBC, SHA1),
        suite!(0x0095, "TLS_RSA_PSK_WITH_AES_256_CBC_SHA", RSA_PSK, PSK, AES_256_CBC, SHA1),
        suite!(0x00AC, "TLS_RSA_PSK_WITH_AES_128_GCM_SHA256", RSA_PSK, PSK, AES_128_GCM, AEAD),
        suite!(0x00AD, "TLS_RSA_PSK_WITH_AES_256_GCM_SHA384", RSA_PSK, PSK, AES_256_GCM, AEAD),
        // ECDHE ECDSA
        suite!(0xC006, "TLS_ECDHE_ECDSA_WITH_NULL_SHA", ECDHE_ECDSA, ECDSA, NULL, SHA1),
        suite!(0xC007, "TLS_ECDHE_ECDSA_WITH_RC4_128_SHA", ECDHE_ECDSA, ECDSA, RC4_128, SHA1),
        suite!(0xC008, "TLS_ECDHE_ECDSA_WITH_3DES_EDE_CBC_SHA", ECDHE_ECDSA, ECDSA, DES_EDE3_CBC, SHA1),
        suite!(0xC009, "TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA", ECDHE_ECDSA, ECDSA, AES_128_CBC, SHA1),
        suite!(0xC00A, "TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA", ECDHE_ECDSA, ECDSA, AES_256_CBC, SHA1),
        // ECDHE RSA
        suite!(0xC011, "TLS_ECDHE_RSA_WITH_NULL_SHA", ECDHE_RSA, RSA, NULL, SHA1),
        suite!(0xC012, "TLS_ECDHE_RSA_WITH_RC4_128_SHA", ECDHE_RSA, RSA, RC4_128, SHA1),
        suite!(0xC013, "TLS_ECDHE_RSA_WITH_3DES_EDE_CBC_SHA", ECDHE_RSA, RSA, DES_EDE3_CBC, SHA1),
        suite!(0xC014, "TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA", ECDHE_RSA, RSA, AES_128_CBC, SHA1),
        suite!(0xC015, "TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA", ECDHE_RSA, RSA, AES_256_CBC, SHA1),
        // ECDH anon
        suite!(0xC018, "TLS_ECDH_anon_WITH_RC4_128_SHA", ECDH_anon, Anonymous, RC4_128, SHA1),
        suite!(0xC019, "TLS_ECDH_anon_WITH_3DES_EDE_CBC_SHA", ECDH_anon, Anonymous, DES_EDE3_CBC, SHA1),
        suite!(0xC01A, "TLS_ECDH_anon_WITH_AES_128_CBC_SHA", ECDH_anon, Anonymous, AES_128_CBC, SHA1),
        suite!(0xC01B, "TLS_ECDH_anon_WITH_AES_256_CBC_SHA", ECDH_anon, Anonymous, AES_256_CBC, SHA1),
        // ECDHE with SHA256/SHA384
        suite!(0xC023, "TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA256", ECDHE_ECDSA, ECDSA, AES_128_CBC, SHA256),
        suite!(0xC024, "TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA384", ECDHE_ECDSA, ECDSA, AES_256_CBC, SHA384),
        suite!(0xC027, "TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA256", ECDHE_RSA, RSA, AES_128_CBC, SHA256),
        suite!(0xC028, "TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA384", ECDHE_RSA, RSA, AES_256_CBC, SHA384),
        // ECDHE GCM
        suite!(0xC02B, "TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256", ECDHE_ECDSA, ECDSA, AES_128_GCM, AEAD),
        suite!(0xC02C, "TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384", ECDHE_ECDSA, ECDSA, AES_256_GCM, AEAD),
        suite!(0xC02F, "TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256", ECDHE_RSA, RSA, AES_128_GCM, AEAD),
        suite!(0xC030, "TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384", ECDHE_RSA, RSA, AES_256_GCM, AEAD),
        // ECDHE PSK
        suite!(0xC035, "TLS_ECDHE_PSK_WITH_AES_128_CBC_SHA", ECDHE_PSK, PSK, AES_128_CBC, SHA1),
        suite!(0xC036, "TLS_ECDHE_PSK_WITH_AES_256_CBC_SHA", ECDHE_PSK, PSK, AES_256_CBC, SHA1),
        suite!(0xC037, "TLS_ECDHE_PSK_WITH_AES_128_CBC_SHA256", ECDHE_PSK, PSK, AES_128_CBC, SHA256),
        // AES CCM (RFC 6655)
        suite!(0xC09C, "TLS_RSA_WITH_AES_128_CCM", RSA, RSA, AES_128_CCM, AEAD),
        suite!(0xC09D, "TLS_RSA_WITH_AES_256_CCM", RSA, RSA, AES_256_CCM, AEAD),
        suite!(0xC09E, "TLS_DHE_RSA_WITH_AES_128_CCM", DHE_RSA, RSA, AES_128_CCM, AEAD),
        suite!(0xC09F, "TLS_DHE_RSA_WITH_AES_256_CCM", DHE_RSA, RSA, AES_256_CCM, AEAD),
        suite!(0xC0A0, "TLS_RSA_WITH_AES_128_CCM_8", RSA, RSA, AES_128_CCM_8, AEAD),
        // ChaCha20-Poly1305 (RFC 7905)
        suite!(0xCCA8, "TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256", ECDHE_RSA, RSA, CHACHA20_POLY1305, AEAD),
        suite!(0xCCA9, "TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256", ECDHE_ECDSA, ECDSA, CHACHA20_POLY1305, AEAD),
        suite!(0xCCAA, "TLS_DHE_RSA_WITH_CHACHA20_POLY1305_SHA256", DHE_RSA, RSA, CHACHA20_POLY1305, AEAD),
        suite!(0xCCAB, "TLS_PSK_WITH_CHACHA20_POLY1305_SHA256", PSK, PSK, CHACHA20_POLY1305, AEAD),
        suite!(0xCCAC, "TLS_ECDHE_PSK_WITH_CHACHA20_POLY1305_SHA256", ECDHE_PSK, PSK, CHACHA20_POLY1305, AEAD),
        suite!(0xCCAD, "TLS_DHE_PSK_WITH_CHACHA20_POLY1305_SHA256", DHE_PSK, PSK, CHACHA20_POLY1305, AEAD),
        // TLS 1.3 cipher suites (RFC 8446)
        suite!(0x1301, "TLS_AES_128_GCM_SHA256", TLS13, NULL, AES_128_GCM, AEAD),
        suite!(0x1302, "TLS_AES_256_GCM_SHA384", TLS13, NULL, AES_256_GCM, AEAD),
        suite!(0x1303, "TLS_CHACHA20_POLY1305_SHA256", TLS13, NULL, CHACHA20_POLY1305, AEAD),
        suite!(0x1304, "TLS_AES_128_CCM_SHA256", TLS13, NULL, AES_128_CCM, AEAD),
        suite!(0x1305, "TLS_AES_128_CCM_8_SHA256", TLS13, NULL, AES_128_CCM_8, AEAD),
    ]
}

// ---------------------------------------------------------------------------
// Cipher suite registry (lookup)
// ---------------------------------------------------------------------------

/// A registry for looking up cipher suites by ID or name.
pub struct CipherSuiteRegistry {
    by_id: HashMap<u16, TlsCipherSuite>,
    by_name: HashMap<String, u16>,
}

impl CipherSuiteRegistry {
    /// Build the default registry from all known suites.
    pub fn new() -> Self {
        let suites = all_cipher_suites();
        let mut by_id = HashMap::with_capacity(suites.len());
        let mut by_name = HashMap::with_capacity(suites.len());
        for suite in suites {
            by_name.insert(suite.name.to_string(), suite.id);
            by_id.insert(suite.id, suite);
        }
        Self { by_id, by_name }
    }

    pub fn lookup_by_id(&self, id: u16) -> Option<&TlsCipherSuite> {
        self.by_id.get(&id)
    }

    pub fn lookup_by_name(&self, name: &str) -> Option<&TlsCipherSuite> {
        self.by_name.get(name).and_then(|id| self.by_id.get(id))
    }

    pub fn all_suites(&self) -> Vec<&TlsCipherSuite> {
        self.by_id.values().collect()
    }

    pub fn count(&self) -> usize {
        self.by_id.len()
    }

    /// Filter suites by minimum security level.
    pub fn filter_by_security(&self, min_level: SecurityLevel) -> Vec<&TlsCipherSuite> {
        self.by_id
            .values()
            .filter(|s| s.security_level() >= min_level)
            .collect()
    }

    /// Get all export cipher suites.
    pub fn export_suites(&self) -> Vec<&TlsCipherSuite> {
        self.by_id.values().filter(|s| s.is_export()).collect()
    }

    /// Get all weak cipher suites.
    pub fn weak_suites(&self) -> Vec<&TlsCipherSuite> {
        self.by_id.values().filter(|s| s.is_weak()).collect()
    }

    /// Get TLS 1.3 suites only.
    pub fn tls13_suites(&self) -> Vec<&TlsCipherSuite> {
        self.by_id.values().filter(|s| s.is_tls13()).collect()
    }

    /// Get FIPS-approved suites.
    pub fn fips_suites(&self) -> Vec<&TlsCipherSuite> {
        self.by_id.values().filter(|s| s.is_fips_approved()).collect()
    }
}

impl Default for CipherSuiteRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Cipher suite selection
// ---------------------------------------------------------------------------

/// Select a cipher suite from the server's perspective.
/// `client_suites` is the client's ordered preference list.
/// `server_suites` is the server's ordered preference list.
/// `server_preference` determines whose order wins.
pub fn select_cipher_suite(
    client_suites: &[u16],
    server_suites: &[u16],
    server_preference: bool,
) -> Option<u16> {
    if server_preference {
        for &server_id in server_suites {
            if client_suites.contains(&server_id) {
                return Some(server_id);
            }
        }
    } else {
        for &client_id in client_suites {
            if server_suites.contains(&client_id) {
                return Some(client_id);
            }
        }
    }
    None
}

/// Select a cipher suite with security filtering.
pub fn select_cipher_suite_filtered(
    client_suites: &[u16],
    server_suites: &[u16],
    registry: &CipherSuiteRegistry,
    min_level: SecurityLevel,
    server_preference: bool,
) -> Option<u16> {
    let preference_list = if server_preference {
        server_suites
    } else {
        client_suites
    };
    let other_list = if server_preference {
        client_suites
    } else {
        server_suites
    };

    for &id in preference_list {
        if !other_list.contains(&id) {
            continue;
        }
        if let Some(suite) = registry.lookup_by_id(id) {
            if suite.security_level() >= min_level {
                return Some(id);
            }
        }
    }
    None
}

/// Detect export cipher suites in a list.
pub fn detect_export_ciphers(suite_ids: &[u16], registry: &CipherSuiteRegistry) -> Vec<u16> {
    suite_ids
        .iter()
        .copied()
        .filter(|id| {
            registry
                .lookup_by_id(*id)
                .map_or(false, |s| s.is_export())
        })
        .collect()
}

/// Detect weak cipher suites in a list.
pub fn detect_weak_ciphers(suite_ids: &[u16], registry: &CipherSuiteRegistry) -> Vec<u16> {
    suite_ids
        .iter()
        .copied()
        .filter(|id| {
            registry
                .lookup_by_id(*id)
                .map_or(false, |s| s.is_weak())
        })
        .collect()
}

/// Sort cipher suites by security score (descending).
pub fn sort_by_security(suite_ids: &[u16], registry: &CipherSuiteRegistry) -> Vec<u16> {
    let mut sorted: Vec<(u16, u32)> = suite_ids
        .iter()
        .map(|&id| {
            let score = registry
                .lookup_by_id(id)
                .map_or(0, |s| s.security_score());
            (id, score)
        })
        .collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1));
    sorted.into_iter().map(|(id, _)| id).collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_has_at_least_80_suites() {
        let reg = CipherSuiteRegistry::new();
        assert!(
            reg.count() >= 80,
            "Expected at least 80 suites, got {}",
            reg.count()
        );
    }

    #[test]
    fn test_lookup_by_id() {
        let reg = CipherSuiteRegistry::new();
        let suite = reg.lookup_by_id(0xC02F).unwrap();
        assert_eq!(suite.name, "TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256");
        assert_eq!(suite.key_exchange, KeyExchange::ECDHE_RSA);
        assert_eq!(suite.encryption, BulkEncryption::AES_128_GCM);
        assert!(suite.has_forward_secrecy());
    }

    #[test]
    fn test_lookup_by_name() {
        let reg = CipherSuiteRegistry::new();
        let suite = reg.lookup_by_name("TLS_AES_128_GCM_SHA256").unwrap();
        assert_eq!(suite.id, 0x1301);
        assert!(suite.is_tls13());
    }

    #[test]
    fn test_security_level_classification() {
        let reg = CipherSuiteRegistry::new();

        // NULL cipher => Insecure.
        let null_suite = reg.lookup_by_id(0x0000).unwrap();
        assert_eq!(null_suite.security_level(), SecurityLevel::Insecure);

        // Export cipher => Export.
        let export_suite = reg.lookup_by_id(0x0003).unwrap();
        assert_eq!(export_suite.security_level(), SecurityLevel::Export);

        // RC4 => Weak.
        let rc4_suite = reg.lookup_by_id(0x0004).unwrap();
        assert_eq!(rc4_suite.security_level(), SecurityLevel::Weak);

        // ECDHE + GCM => Secure.
        let modern = reg.lookup_by_id(0xC02F).unwrap();
        assert_eq!(modern.security_level(), SecurityLevel::Secure);

        // TLS 1.3 => Recommended.
        let tls13 = reg.lookup_by_id(0x1301).unwrap();
        assert_eq!(tls13.security_level(), SecurityLevel::Recommended);
    }

    #[test]
    fn test_export_detection() {
        let reg = CipherSuiteRegistry::new();
        let suites = vec![0xC02F, 0x0003, 0x0008, 0x1301];
        let exports = detect_export_ciphers(&suites, &reg);
        assert_eq!(exports.len(), 2);
        assert!(exports.contains(&0x0003));
        assert!(exports.contains(&0x0008));
    }

    #[test]
    fn test_weak_detection() {
        let reg = CipherSuiteRegistry::new();
        let suites = vec![0xC02F, 0x0004, 0x0005, 0x1301];
        let weak = detect_weak_ciphers(&suites, &reg);
        assert_eq!(weak.len(), 2);
    }

    #[test]
    fn test_select_cipher_suite_server_preference() {
        let client = vec![0x0035, 0xC02F, 0x1301];
        let server = vec![0x1301, 0xC02F, 0x0035];
        let selected = select_cipher_suite(&client, &server, true);
        assert_eq!(selected, Some(0x1301));
    }

    #[test]
    fn test_select_cipher_suite_client_preference() {
        let client = vec![0x0035, 0xC02F, 0x1301];
        let server = vec![0x1301, 0xC02F, 0x0035];
        let selected = select_cipher_suite(&client, &server, false);
        assert_eq!(selected, Some(0x0035));
    }

    #[test]
    fn test_select_cipher_suite_no_match() {
        let client = vec![0x0035];
        let server = vec![0xC02F];
        let selected = select_cipher_suite(&client, &server, true);
        assert!(selected.is_none());
    }

    #[test]
    fn test_select_cipher_suite_filtered() {
        let reg = CipherSuiteRegistry::new();
        let client = vec![0x0004, 0x0035, 0xC02F, 0x1301];
        let server = vec![0x0004, 0x0035, 0xC02F, 0x1301];
        let selected = select_cipher_suite_filtered(
            &client,
            &server,
            &reg,
            SecurityLevel::Secure,
            true,
        );
        // 0x0004 (RC4) and 0x0035 (AES CBC) are below Secure.
        // 0xC02F is first Secure suite in server order, but 0x0004 is first in both lists.
        // With server preference, iterate server list: 0x0004 (Weak) skip, 0x0035 (Legacy) skip, 0xC02F (Secure) match.
        assert_eq!(selected, Some(0xC02F));
    }

    #[test]
    fn test_sort_by_security() {
        let reg = CipherSuiteRegistry::new();
        let suites = vec![0x0004, 0xC02F, 0x1301, 0x0035];
        let sorted = sort_by_security(&suites, &reg);
        // TLS 1.3 and ECDHE+GCM should be at the top.
        assert!(sorted[0] == 0x1301 || sorted[0] == 0xC02F);
    }

    #[test]
    fn test_fips_suites() {
        let reg = CipherSuiteRegistry::new();
        let fips = reg.fips_suites();
        assert!(!fips.is_empty());
        for s in &fips {
            assert!(s.is_fips_approved(), "{} not FIPS", s.name);
        }
    }

    #[test]
    fn test_tls13_suites() {
        let reg = CipherSuiteRegistry::new();
        let tls13 = reg.tls13_suites();
        assert_eq!(tls13.len(), 5);
        for s in &tls13 {
            assert!(s.is_tls13());
        }
    }

    #[test]
    fn test_key_exchange_properties() {
        assert!(KeyExchange::ECDHE_RSA.has_forward_secrecy());
        assert!(!KeyExchange::RSA.has_forward_secrecy());
        assert!(KeyExchange::RSA_EXPORT.is_export());
        assert!(!KeyExchange::RSA.is_export());
    }

    #[test]
    fn test_bulk_encryption_properties() {
        assert!(BulkEncryption::AES_128_GCM.is_aead());
        assert!(!BulkEncryption::AES_128_CBC.is_aead());
        assert!(BulkEncryption::RC4_40.is_export_grade());
        assert_eq!(BulkEncryption::AES_256_GCM.key_bits(), 256);
        assert_eq!(BulkEncryption::NULL.key_bits(), 0);
    }

    #[test]
    fn test_to_negsyn_cipher_suite() {
        let reg = CipherSuiteRegistry::new();
        let suite = reg.lookup_by_id(0xC02F).unwrap();
        let ns = suite.to_negsyn_cipher_suite();
        assert_eq!(ns.iana_id, 0xC02F);
        assert_eq!(ns.key_exchange, negsyn_types::protocol::KeyExchange::ECDHE);
        assert_eq!(ns.auth, negsyn_types::protocol::AuthAlgorithm::RSA);
    }

    #[test]
    fn test_display() {
        let reg = CipherSuiteRegistry::new();
        let suite = reg.lookup_by_id(0x1301).unwrap();
        let display = format!("{}", suite);
        assert!(display.contains("0x1301"));
        assert!(display.contains("TLS_AES_128_GCM_SHA256"));
    }
}
