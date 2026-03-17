//! SSH algorithm taxonomy with security classification.
//!
//! Every algorithm variant carries metadata such as key size, security level,
//! whether it is deprecated, and whether it is FIPS-approved.  The module also
//! provides canonical preference orderings and a central registry.

use serde::{Deserialize, Serialize};
use std::fmt;

// ---------------------------------------------------------------------------
// Security classification
// ---------------------------------------------------------------------------

/// Security classification for an algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum SecurityClassification {
    /// Known-broken — MUST NOT use.
    Broken,
    /// Weak — SHOULD NOT use; acceptable only for legacy interop.
    Weak,
    /// Acceptable — meets minimum bar but better options exist.
    Acceptable,
    /// Recommended — current best practice.
    Recommended,
}

impl SecurityClassification {
    /// Numeric score (higher = more secure).
    pub fn score(self) -> u32 {
        match self {
            Self::Broken => 0,
            Self::Weak => 1,
            Self::Acceptable => 2,
            Self::Recommended => 3,
        }
    }

    pub fn is_safe(self) -> bool {
        matches!(self, Self::Acceptable | Self::Recommended)
    }

    pub fn is_deprecated(self) -> bool {
        matches!(self, Self::Broken | Self::Weak)
    }
}

impl fmt::Display for SecurityClassification {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Broken => write!(f, "BROKEN"),
            Self::Weak => write!(f, "WEAK"),
            Self::Acceptable => write!(f, "ACCEPTABLE"),
            Self::Recommended => write!(f, "RECOMMENDED"),
        }
    }
}

// ---------------------------------------------------------------------------
// Host key algorithms
// ---------------------------------------------------------------------------

/// SSH host key / public key algorithms (RFC 4253 §6.6).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum HostKeyAlgorithm {
    SshRsa,
    SshEd25519,
    EcdsaSha2Nistp256,
    EcdsaSha2Nistp384,
    EcdsaSha2Nistp521,
    RsaSha2_256,
    RsaSha2_512,
    SshDss,
}

impl HostKeyAlgorithm {
    pub fn wire_name(self) -> &'static str {
        match self {
            Self::SshRsa => "ssh-rsa",
            Self::SshEd25519 => "ssh-ed25519",
            Self::EcdsaSha2Nistp256 => "ecdsa-sha2-nistp256",
            Self::EcdsaSha2Nistp384 => "ecdsa-sha2-nistp384",
            Self::EcdsaSha2Nistp521 => "ecdsa-sha2-nistp521",
            Self::RsaSha2_256 => "rsa-sha2-256",
            Self::RsaSha2_512 => "rsa-sha2-512",
            Self::SshDss => "ssh-dss",
        }
    }

    pub fn from_wire_name(name: &str) -> Option<Self> {
        match name {
            "ssh-rsa" => Some(Self::SshRsa),
            "ssh-ed25519" => Some(Self::SshEd25519),
            "ecdsa-sha2-nistp256" => Some(Self::EcdsaSha2Nistp256),
            "ecdsa-sha2-nistp384" => Some(Self::EcdsaSha2Nistp384),
            "ecdsa-sha2-nistp521" => Some(Self::EcdsaSha2Nistp521),
            "rsa-sha2-256" => Some(Self::RsaSha2_256),
            "rsa-sha2-512" => Some(Self::RsaSha2_512),
            "ssh-dss" => Some(Self::SshDss),
            _ => None,
        }
    }

    pub fn security(self) -> SecurityClassification {
        match self {
            Self::SshEd25519 => SecurityClassification::Recommended,
            Self::RsaSha2_256 | Self::RsaSha2_512 => SecurityClassification::Recommended,
            Self::EcdsaSha2Nistp256 | Self::EcdsaSha2Nistp384 | Self::EcdsaSha2Nistp521 => {
                SecurityClassification::Recommended
            }
            Self::SshRsa => SecurityClassification::Weak, // SHA-1
            Self::SshDss => SecurityClassification::Broken,
        }
    }

    pub fn key_bits(self) -> u32 {
        match self {
            Self::SshRsa | Self::RsaSha2_256 | Self::RsaSha2_512 => 3072,
            Self::SshEd25519 => 256,
            Self::EcdsaSha2Nistp256 => 256,
            Self::EcdsaSha2Nistp384 => 384,
            Self::EcdsaSha2Nistp521 => 521,
            Self::SshDss => 1024,
        }
    }

    /// Canonical preference order (most preferred first).
    pub fn preference_order() -> &'static [Self] {
        &[
            Self::SshEd25519,
            Self::EcdsaSha2Nistp521,
            Self::EcdsaSha2Nistp384,
            Self::EcdsaSha2Nistp256,
            Self::RsaSha2_512,
            Self::RsaSha2_256,
            Self::SshRsa,
            Self::SshDss,
        ]
    }

    pub fn all_variants() -> &'static [Self] {
        Self::preference_order()
    }

    pub fn is_deprecated(self) -> bool {
        self.security().is_deprecated()
    }

    pub fn is_fips_approved(self) -> bool {
        matches!(
            self,
            Self::RsaSha2_256
                | Self::RsaSha2_512
                | Self::EcdsaSha2Nistp256
                | Self::EcdsaSha2Nistp384
                | Self::EcdsaSha2Nistp521
        )
    }
}

impl fmt::Display for HostKeyAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.wire_name())
    }
}

// ---------------------------------------------------------------------------
// Encryption algorithms
// ---------------------------------------------------------------------------

/// SSH encryption (cipher) algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum EncryptionAlgorithm {
    Aes128Ctr,
    Aes192Ctr,
    Aes256Ctr,
    Aes128Gcm,
    Aes256Gcm,
    Chacha20Poly1305,
    Aes128Cbc,
    Aes192Cbc,
    Aes256Cbc,
    TripleDesCbc,
    None,
}

impl EncryptionAlgorithm {
    pub fn wire_name(self) -> &'static str {
        match self {
            Self::Aes128Ctr => "aes128-ctr",
            Self::Aes192Ctr => "aes192-ctr",
            Self::Aes256Ctr => "aes256-ctr",
            Self::Aes128Gcm => "aes128-gcm@openssh.com",
            Self::Aes256Gcm => "aes256-gcm@openssh.com",
            Self::Chacha20Poly1305 => "chacha20-poly1305@openssh.com",
            Self::Aes128Cbc => "aes128-cbc",
            Self::Aes192Cbc => "aes192-cbc",
            Self::Aes256Cbc => "aes256-cbc",
            Self::TripleDesCbc => "3des-cbc",
            Self::None => "none",
        }
    }

    pub fn from_wire_name(name: &str) -> Option<Self> {
        match name {
            "aes128-ctr" => Some(Self::Aes128Ctr),
            "aes192-ctr" => Some(Self::Aes192Ctr),
            "aes256-ctr" => Some(Self::Aes256Ctr),
            "aes128-gcm@openssh.com" => Some(Self::Aes128Gcm),
            "aes256-gcm@openssh.com" => Some(Self::Aes256Gcm),
            "chacha20-poly1305@openssh.com" => Some(Self::Chacha20Poly1305),
            "aes128-cbc" => Some(Self::Aes128Cbc),
            "aes192-cbc" => Some(Self::Aes192Cbc),
            "aes256-cbc" => Some(Self::Aes256Cbc),
            "3des-cbc" => Some(Self::TripleDesCbc),
            "none" => Some(Self::None),
            _ => None,
        }
    }

    pub fn security(self) -> SecurityClassification {
        match self {
            Self::Chacha20Poly1305 | Self::Aes256Gcm => SecurityClassification::Recommended,
            Self::Aes128Gcm | Self::Aes256Ctr => SecurityClassification::Recommended,
            Self::Aes128Ctr | Self::Aes192Ctr => SecurityClassification::Acceptable,
            Self::Aes128Cbc | Self::Aes192Cbc | Self::Aes256Cbc => SecurityClassification::Weak,
            Self::TripleDesCbc => SecurityClassification::Weak,
            Self::None => SecurityClassification::Broken,
        }
    }

    pub fn key_bits(self) -> u32 {
        match self {
            Self::Aes128Ctr | Self::Aes128Gcm | Self::Aes128Cbc => 128,
            Self::Aes192Ctr | Self::Aes192Cbc => 192,
            Self::Aes256Ctr | Self::Aes256Gcm | Self::Aes256Cbc => 256,
            Self::Chacha20Poly1305 => 256,
            Self::TripleDesCbc => 168,
            Self::None => 0,
        }
    }

    pub fn block_size(self) -> usize {
        match self {
            Self::Chacha20Poly1305 => 8, // stream, but SSH uses 8
            Self::Aes128Gcm | Self::Aes256Gcm => 16,
            Self::Aes128Ctr | Self::Aes192Ctr | Self::Aes256Ctr => 16,
            Self::Aes128Cbc | Self::Aes192Cbc | Self::Aes256Cbc => 16,
            Self::TripleDesCbc => 8,
            Self::None => 8,
        }
    }

    pub fn is_aead(self) -> bool {
        matches!(
            self,
            Self::Aes128Gcm | Self::Aes256Gcm | Self::Chacha20Poly1305
        )
    }

    pub fn is_cbc(self) -> bool {
        matches!(
            self,
            Self::Aes128Cbc | Self::Aes192Cbc | Self::Aes256Cbc | Self::TripleDesCbc
        )
    }

    pub fn is_ctr(self) -> bool {
        matches!(self, Self::Aes128Ctr | Self::Aes192Ctr | Self::Aes256Ctr)
    }

    pub fn mac_length(self) -> usize {
        match self {
            Self::Aes128Gcm | Self::Aes256Gcm => 16,
            Self::Chacha20Poly1305 => 16,
            _ => 0,
        }
    }

    pub fn iv_length(self) -> usize {
        match self {
            Self::Aes128Gcm | Self::Aes256Gcm => 12,
            Self::Chacha20Poly1305 => 0, // derived from seqno
            Self::Aes128Ctr | Self::Aes192Ctr | Self::Aes256Ctr => 16,
            Self::Aes128Cbc | Self::Aes192Cbc | Self::Aes256Cbc => 16,
            Self::TripleDesCbc => 8,
            Self::None => 0,
        }
    }

    pub fn preference_order() -> &'static [Self] {
        &[
            Self::Chacha20Poly1305,
            Self::Aes256Gcm,
            Self::Aes128Gcm,
            Self::Aes256Ctr,
            Self::Aes192Ctr,
            Self::Aes128Ctr,
            Self::Aes256Cbc,
            Self::Aes192Cbc,
            Self::Aes128Cbc,
            Self::TripleDesCbc,
            Self::None,
        ]
    }

    pub fn all_variants() -> &'static [Self] {
        Self::preference_order()
    }

    pub fn is_deprecated(self) -> bool {
        self.security().is_deprecated()
    }

    pub fn is_fips_approved(self) -> bool {
        matches!(
            self,
            Self::Aes128Ctr
                | Self::Aes192Ctr
                | Self::Aes256Ctr
                | Self::Aes128Gcm
                | Self::Aes256Gcm
                | Self::Aes128Cbc
                | Self::Aes192Cbc
                | Self::Aes256Cbc
        )
    }
}

impl fmt::Display for EncryptionAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.wire_name())
    }
}

// ---------------------------------------------------------------------------
// MAC algorithms
// ---------------------------------------------------------------------------

/// SSH MAC algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum MacAlgorithm {
    HmacSha2_256,
    HmacSha2_512,
    HmacSha2_256Etm,
    HmacSha2_512Etm,
    HmacSha1,
    HmacSha1Etm,
    HmacMd5,
    HmacMd5Etm,
    None,
}

impl MacAlgorithm {
    pub fn wire_name(self) -> &'static str {
        match self {
            Self::HmacSha2_256 => "hmac-sha2-256",
            Self::HmacSha2_512 => "hmac-sha2-512",
            Self::HmacSha2_256Etm => "hmac-sha2-256-etm@openssh.com",
            Self::HmacSha2_512Etm => "hmac-sha2-512-etm@openssh.com",
            Self::HmacSha1 => "hmac-sha1",
            Self::HmacSha1Etm => "hmac-sha1-etm@openssh.com",
            Self::HmacMd5 => "hmac-md5",
            Self::HmacMd5Etm => "hmac-md5-etm@openssh.com",
            Self::None => "none",
        }
    }

    pub fn from_wire_name(name: &str) -> Option<Self> {
        match name {
            "hmac-sha2-256" => Some(Self::HmacSha2_256),
            "hmac-sha2-512" => Some(Self::HmacSha2_512),
            "hmac-sha2-256-etm@openssh.com" => Some(Self::HmacSha2_256Etm),
            "hmac-sha2-512-etm@openssh.com" => Some(Self::HmacSha2_512Etm),
            "hmac-sha1" => Some(Self::HmacSha1),
            "hmac-sha1-etm@openssh.com" => Some(Self::HmacSha1Etm),
            "hmac-md5" => Some(Self::HmacMd5),
            "hmac-md5-etm@openssh.com" => Some(Self::HmacMd5Etm),
            "none" => Some(Self::None),
            _ => None,
        }
    }

    pub fn security(self) -> SecurityClassification {
        match self {
            Self::HmacSha2_256Etm | Self::HmacSha2_512Etm => SecurityClassification::Recommended,
            Self::HmacSha2_256 | Self::HmacSha2_512 => SecurityClassification::Acceptable,
            Self::HmacSha1 | Self::HmacSha1Etm => SecurityClassification::Weak,
            Self::HmacMd5 | Self::HmacMd5Etm => SecurityClassification::Broken,
            Self::None => SecurityClassification::Broken,
        }
    }

    pub fn digest_length(self) -> usize {
        match self {
            Self::HmacSha2_256 | Self::HmacSha2_256Etm => 32,
            Self::HmacSha2_512 | Self::HmacSha2_512Etm => 64,
            Self::HmacSha1 | Self::HmacSha1Etm => 20,
            Self::HmacMd5 | Self::HmacMd5Etm => 16,
            Self::None => 0,
        }
    }

    pub fn key_length(self) -> usize {
        match self {
            Self::HmacSha2_256 | Self::HmacSha2_256Etm => 32,
            Self::HmacSha2_512 | Self::HmacSha2_512Etm => 64,
            Self::HmacSha1 | Self::HmacSha1Etm => 20,
            Self::HmacMd5 | Self::HmacMd5Etm => 16,
            Self::None => 0,
        }
    }

    pub fn is_etm(self) -> bool {
        matches!(
            self,
            Self::HmacSha2_256Etm
                | Self::HmacSha2_512Etm
                | Self::HmacSha1Etm
                | Self::HmacMd5Etm
        )
    }

    pub fn preference_order() -> &'static [Self] {
        &[
            Self::HmacSha2_512Etm,
            Self::HmacSha2_256Etm,
            Self::HmacSha2_512,
            Self::HmacSha2_256,
            Self::HmacSha1Etm,
            Self::HmacSha1,
            Self::HmacMd5Etm,
            Self::HmacMd5,
            Self::None,
        ]
    }

    pub fn all_variants() -> &'static [Self] {
        Self::preference_order()
    }

    pub fn is_deprecated(self) -> bool {
        self.security().is_deprecated()
    }

    pub fn is_fips_approved(self) -> bool {
        matches!(
            self,
            Self::HmacSha2_256
                | Self::HmacSha2_512
                | Self::HmacSha2_256Etm
                | Self::HmacSha2_512Etm
                | Self::HmacSha1
                | Self::HmacSha1Etm
        )
    }
}

impl fmt::Display for MacAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.wire_name())
    }
}

// ---------------------------------------------------------------------------
// Compression algorithms
// ---------------------------------------------------------------------------

/// SSH compression algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    None,
    Zlib,
    ZlibOpenssh,
}

impl CompressionAlgorithm {
    pub fn wire_name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Zlib => "zlib",
            Self::ZlibOpenssh => "zlib@openssh.com",
        }
    }

    pub fn from_wire_name(name: &str) -> Option<Self> {
        match name {
            "none" => Some(Self::None),
            "zlib" => Some(Self::Zlib),
            "zlib@openssh.com" => Some(Self::ZlibOpenssh),
            _ => None,
        }
    }

    pub fn security(self) -> SecurityClassification {
        match self {
            Self::None => SecurityClassification::Recommended,
            Self::ZlibOpenssh => SecurityClassification::Acceptable,
            Self::Zlib => SecurityClassification::Weak, // pre-auth compression = CRIME-like
        }
    }

    pub fn preference_order() -> &'static [Self] {
        &[Self::None, Self::ZlibOpenssh, Self::Zlib]
    }

    pub fn all_variants() -> &'static [Self] {
        Self::preference_order()
    }

    pub fn is_deprecated(self) -> bool {
        self.security().is_deprecated()
    }
}

impl fmt::Display for CompressionAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.wire_name())
    }
}

// ---------------------------------------------------------------------------
// Algorithm properties / registry
// ---------------------------------------------------------------------------

/// Properties of an SSH algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmProperties {
    pub name: String,
    pub category: AlgorithmCategory,
    pub security: SecurityClassification,
    pub key_bits: u32,
    pub is_fips_approved: bool,
    pub is_deprecated: bool,
    pub rfc: Option<String>,
    pub notes: Option<String>,
}

/// Algorithm category for grouping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AlgorithmCategory {
    KeyExchange,
    HostKey,
    Encryption,
    Mac,
    Compression,
}

impl fmt::Display for AlgorithmCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::KeyExchange => write!(f, "kex"),
            Self::HostKey => write!(f, "host_key"),
            Self::Encryption => write!(f, "encryption"),
            Self::Mac => write!(f, "mac"),
            Self::Compression => write!(f, "compression"),
        }
    }
}

/// Central algorithm registry.
#[derive(Debug, Clone)]
pub struct AlgorithmRegistry {
    entries: Vec<AlgorithmProperties>,
}

impl AlgorithmRegistry {
    /// Build the default registry with all known algorithms.
    pub fn new() -> Self {
        let mut entries = Vec::new();

        // Host key algorithms
        for alg in HostKeyAlgorithm::all_variants() {
            entries.push(AlgorithmProperties {
                name: alg.wire_name().to_string(),
                category: AlgorithmCategory::HostKey,
                security: alg.security(),
                key_bits: alg.key_bits(),
                is_fips_approved: alg.is_fips_approved(),
                is_deprecated: alg.is_deprecated(),
                rfc: Some("RFC 4253".into()),
                notes: None,
            });
        }

        // Encryption algorithms
        for alg in EncryptionAlgorithm::all_variants() {
            entries.push(AlgorithmProperties {
                name: alg.wire_name().to_string(),
                category: AlgorithmCategory::Encryption,
                security: alg.security(),
                key_bits: alg.key_bits(),
                is_fips_approved: alg.is_fips_approved(),
                is_deprecated: alg.is_deprecated(),
                rfc: if alg.is_aead() {
                    Some("RFC 5647 / OpenSSH".into())
                } else {
                    Some("RFC 4253".into())
                },
                notes: if alg.is_cbc() {
                    Some("CBC mode vulnerable to plaintext recovery".into())
                } else {
                    None
                },
            });
        }

        // MAC algorithms
        for alg in MacAlgorithm::all_variants() {
            entries.push(AlgorithmProperties {
                name: alg.wire_name().to_string(),
                category: AlgorithmCategory::Mac,
                security: alg.security(),
                key_bits: (alg.key_length() * 8) as u32,
                is_fips_approved: alg.is_fips_approved(),
                is_deprecated: alg.is_deprecated(),
                rfc: if alg.is_etm() {
                    Some("OpenSSH ETM extension".into())
                } else {
                    Some("RFC 4253".into())
                },
                notes: if alg.is_etm() {
                    Some("Encrypt-then-MAC provides better security".into())
                } else {
                    None
                },
            });
        }

        // Compression algorithms
        for alg in CompressionAlgorithm::all_variants() {
            entries.push(AlgorithmProperties {
                name: alg.wire_name().to_string(),
                category: AlgorithmCategory::Compression,
                security: alg.security(),
                key_bits: 0,
                is_fips_approved: false,
                is_deprecated: alg.is_deprecated(),
                rfc: Some("RFC 4253".into()),
                notes: if matches!(alg, CompressionAlgorithm::Zlib) {
                    Some("Pre-auth compression vulnerable to CRIME-like attacks".into())
                } else {
                    None
                },
            });
        }

        Self { entries }
    }

    pub fn lookup(&self, name: &str) -> Option<&AlgorithmProperties> {
        self.entries.iter().find(|e| e.name == name)
    }

    pub fn by_category(&self, cat: AlgorithmCategory) -> Vec<&AlgorithmProperties> {
        self.entries.iter().filter(|e| e.category == cat).collect()
    }

    pub fn deprecated(&self) -> Vec<&AlgorithmProperties> {
        self.entries.iter().filter(|e| e.is_deprecated).collect()
    }

    pub fn fips_approved(&self) -> Vec<&AlgorithmProperties> {
        self.entries.iter().filter(|e| e.is_fips_approved).collect()
    }

    pub fn all(&self) -> &[AlgorithmProperties] {
        &self.entries
    }

    /// Returns the security classification for a wire name, regardless of category.
    pub fn classify(&self, name: &str) -> Option<SecurityClassification> {
        self.lookup(name).map(|p| p.security)
    }

    /// Returns all weak or broken algorithms.
    pub fn weak_algorithms(&self) -> Vec<&AlgorithmProperties> {
        self.entries
            .iter()
            .filter(|e| e.security.is_deprecated())
            .collect()
    }

    /// Returns the minimum security level across a list of wire names.
    pub fn minimum_security<'a>(
        &self,
        names: impl IntoIterator<Item = &'a str>,
    ) -> SecurityClassification {
        let mut min = SecurityClassification::Recommended;
        for name in names {
            if let Some(props) = self.lookup(name) {
                if props.security < min {
                    min = props.security;
                }
            }
        }
        min
    }
}

impl Default for AlgorithmRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helpers for name-list ↔ enum conversion
// ---------------------------------------------------------------------------

/// Parse a comma-separated name-list into host key variants, skipping unknowns.
pub fn parse_host_key_list(name_list: &str) -> Vec<HostKeyAlgorithm> {
    name_list
        .split(',')
        .filter_map(|n| HostKeyAlgorithm::from_wire_name(n.trim()))
        .collect()
}

/// Parse a comma-separated name-list into encryption variants.
pub fn parse_encryption_list(name_list: &str) -> Vec<EncryptionAlgorithm> {
    name_list
        .split(',')
        .filter_map(|n| EncryptionAlgorithm::from_wire_name(n.trim()))
        .collect()
}

/// Parse a comma-separated name-list into MAC variants.
pub fn parse_mac_list(name_list: &str) -> Vec<MacAlgorithm> {
    name_list
        .split(',')
        .filter_map(|n| MacAlgorithm::from_wire_name(n.trim()))
        .collect()
}

/// Parse a comma-separated name-list into compression variants.
pub fn parse_compression_list(name_list: &str) -> Vec<CompressionAlgorithm> {
    name_list
        .split(',')
        .filter_map(|n| CompressionAlgorithm::from_wire_name(n.trim()))
        .collect()
}

/// Serialize a list of host key algorithms into a name-list.
pub fn host_key_name_list(algs: &[HostKeyAlgorithm]) -> String {
    algs.iter()
        .map(|a| a.wire_name())
        .collect::<Vec<_>>()
        .join(",")
}

/// Serialize a list of encryption algorithms into a name-list.
pub fn encryption_name_list(algs: &[EncryptionAlgorithm]) -> String {
    algs.iter()
        .map(|a| a.wire_name())
        .collect::<Vec<_>>()
        .join(",")
}

/// Serialize a list of MAC algorithms into a name-list.
pub fn mac_name_list(algs: &[MacAlgorithm]) -> String {
    algs.iter()
        .map(|a| a.wire_name())
        .collect::<Vec<_>>()
        .join(",")
}

/// Serialize a list of compression algorithms into a name-list.
pub fn compression_name_list(algs: &[CompressionAlgorithm]) -> String {
    algs.iter()
        .map(|a| a.wire_name())
        .collect::<Vec<_>>()
        .join(",")
}

/// Returns true if ANY algorithm in the combined set is deprecated.
pub fn has_weak_algorithm(
    host_keys: &[HostKeyAlgorithm],
    encryptions: &[EncryptionAlgorithm],
    macs: &[MacAlgorithm],
) -> bool {
    host_keys.iter().any(|a| a.is_deprecated())
        || encryptions.iter().any(|a| a.is_deprecated())
        || macs.iter().any(|a| a.is_deprecated())
}

/// Compute the overall security classification as the minimum across all selected
/// algorithms.
pub fn overall_security(
    host_key: HostKeyAlgorithm,
    encryption: EncryptionAlgorithm,
    mac: MacAlgorithm,
    compression: CompressionAlgorithm,
) -> SecurityClassification {
    let levels = [
        host_key.security(),
        encryption.security(),
        mac.security(),
        compression.security(),
    ];
    levels.into_iter().min().unwrap_or(SecurityClassification::Broken)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_key_roundtrip() {
        for alg in HostKeyAlgorithm::all_variants() {
            let name = alg.wire_name();
            assert_eq!(HostKeyAlgorithm::from_wire_name(name), Some(*alg));
        }
    }

    #[test]
    fn encryption_roundtrip() {
        for alg in EncryptionAlgorithm::all_variants() {
            let name = alg.wire_name();
            assert_eq!(EncryptionAlgorithm::from_wire_name(name), Some(*alg));
        }
    }

    #[test]
    fn mac_roundtrip() {
        for alg in MacAlgorithm::all_variants() {
            let name = alg.wire_name();
            assert_eq!(MacAlgorithm::from_wire_name(name), Some(*alg));
        }
    }

    #[test]
    fn compression_roundtrip() {
        for alg in CompressionAlgorithm::all_variants() {
            let name = alg.wire_name();
            assert_eq!(CompressionAlgorithm::from_wire_name(name), Some(*alg));
        }
    }

    #[test]
    fn security_ordering() {
        assert!(SecurityClassification::Broken < SecurityClassification::Weak);
        assert!(SecurityClassification::Weak < SecurityClassification::Acceptable);
        assert!(SecurityClassification::Acceptable < SecurityClassification::Recommended);
    }

    #[test]
    fn ed25519_is_recommended() {
        assert_eq!(
            HostKeyAlgorithm::SshEd25519.security(),
            SecurityClassification::Recommended
        );
    }

    #[test]
    fn cbc_is_weak() {
        assert!(EncryptionAlgorithm::Aes128Cbc.is_cbc());
        assert!(EncryptionAlgorithm::Aes128Cbc.is_deprecated());
    }

    #[test]
    fn etm_detection() {
        assert!(MacAlgorithm::HmacSha2_256Etm.is_etm());
        assert!(!MacAlgorithm::HmacSha2_256.is_etm());
    }

    #[test]
    fn aead_detection() {
        assert!(EncryptionAlgorithm::Aes256Gcm.is_aead());
        assert!(EncryptionAlgorithm::Chacha20Poly1305.is_aead());
        assert!(!EncryptionAlgorithm::Aes256Ctr.is_aead());
    }

    #[test]
    fn name_list_roundtrip() {
        let algs = vec![
            HostKeyAlgorithm::SshEd25519,
            HostKeyAlgorithm::RsaSha2_256,
        ];
        let nl = host_key_name_list(&algs);
        assert_eq!(nl, "ssh-ed25519,rsa-sha2-256");
        let parsed = parse_host_key_list(&nl);
        assert_eq!(parsed, algs);
    }

    #[test]
    fn overall_security_min() {
        let sec = overall_security(
            HostKeyAlgorithm::SshEd25519,
            EncryptionAlgorithm::Aes256Gcm,
            MacAlgorithm::HmacMd5, // broken
            CompressionAlgorithm::None,
        );
        assert_eq!(sec, SecurityClassification::Broken);
    }

    #[test]
    fn registry_lookup() {
        let reg = AlgorithmRegistry::new();
        let props = reg.lookup("ssh-ed25519").unwrap();
        assert_eq!(props.security, SecurityClassification::Recommended);
        assert_eq!(props.category, AlgorithmCategory::HostKey);
    }

    #[test]
    fn registry_weak() {
        let reg = AlgorithmRegistry::new();
        let weak = reg.weak_algorithms();
        assert!(weak.iter().any(|p| p.name == "ssh-dss"));
        assert!(weak.iter().any(|p| p.name == "hmac-md5"));
    }

    #[test]
    fn registry_fips() {
        let reg = AlgorithmRegistry::new();
        let fips = reg.fips_approved();
        assert!(fips.iter().any(|p| p.name == "rsa-sha2-256"));
        assert!(!fips.iter().any(|p| p.name == "ssh-ed25519"));
    }

    #[test]
    fn has_weak_detects_deprecated() {
        assert!(has_weak_algorithm(
            &[HostKeyAlgorithm::SshDss],
            &[EncryptionAlgorithm::Aes256Gcm],
            &[MacAlgorithm::HmacSha2_256],
        ));
        assert!(!has_weak_algorithm(
            &[HostKeyAlgorithm::SshEd25519],
            &[EncryptionAlgorithm::Aes256Gcm],
            &[MacAlgorithm::HmacSha2_256],
        ));
    }

    #[test]
    fn minimum_security_across_names() {
        let reg = AlgorithmRegistry::new();
        let min = reg.minimum_security(["ssh-ed25519", "hmac-md5"].iter().copied());
        assert_eq!(min, SecurityClassification::Broken);
    }

    #[test]
    fn unknown_wire_name_returns_none() {
        assert!(HostKeyAlgorithm::from_wire_name("unknown-alg").is_none());
        assert!(EncryptionAlgorithm::from_wire_name("blowfish-cbc").is_none());
        assert!(MacAlgorithm::from_wire_name("hmac-ripemd160").is_none());
        assert!(CompressionAlgorithm::from_wire_name("lz4").is_none());
    }

    #[test]
    fn category_display() {
        assert_eq!(AlgorithmCategory::KeyExchange.to_string(), "kex");
        assert_eq!(AlgorithmCategory::Encryption.to_string(), "encryption");
    }

    #[test]
    fn parse_encryption_list_skips_unknowns() {
        let list = "aes256-ctr,blowfish-cbc,aes128-ctr";
        let parsed = parse_encryption_list(list);
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0], EncryptionAlgorithm::Aes256Ctr);
        assert_eq!(parsed[1], EncryptionAlgorithm::Aes128Ctr);
    }
}
