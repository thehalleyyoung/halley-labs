//! SSH key exchange (KEX) types and parsing — RFC 4253 §7.
//!
//! The SSH_MSG_KEXINIT message carries:
//!   - 16-byte cookie
//!   - 10 name-lists (kex algorithms, server host key, encryption c2s/s2c,
//!     mac c2s/s2c, compression c2s/s2c, languages c2s/s2c)
//!   - boolean first_kex_packet_follows
//!   - uint32 reserved (0)

use crate::algorithms::{
    CompressionAlgorithm, EncryptionAlgorithm, HostKeyAlgorithm,
    MacAlgorithm as SshMac,
};
use crate::constants::SSH_MSG_KEXINIT;
use crate::{SshError, SshResult};
use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::io::{Cursor, Read, Write};

// ---------------------------------------------------------------------------
// KexAlgorithm
// ---------------------------------------------------------------------------

/// SSH key-exchange algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum KexAlgorithm {
    Curve25519Sha256,
    Curve25519Sha256Libssh,
    EcdhSha2Nistp256,
    EcdhSha2Nistp384,
    EcdhSha2Nistp521,
    DiffieHellmanGroup14Sha256,
    DiffieHellmanGroup16Sha512,
    DiffieHellmanGroup18Sha512,
    DiffieHellmanGroup14Sha1,
    DiffieHellmanGroup1Sha1,
    DiffieHellmanGroupExchangeSha256,
    DiffieHellmanGroupExchangeSha1,
    /// Pseudo-algorithm: strict KEX extension marker (client).
    ExtInfoC,
    /// Pseudo-algorithm: strict KEX extension marker (server).
    ExtInfoS,
    /// Pseudo-algorithm: strict KEX (client) — Terrapin mitigation.
    KexStrictClientV00Openssh,
    /// Pseudo-algorithm: strict KEX (server) — Terrapin mitigation.
    KexStrictServerV00Openssh,
}

impl KexAlgorithm {
    pub fn wire_name(self) -> &'static str {
        match self {
            Self::Curve25519Sha256 => "curve25519-sha256",
            Self::Curve25519Sha256Libssh => "curve25519-sha256@libssh.org",
            Self::EcdhSha2Nistp256 => "ecdh-sha2-nistp256",
            Self::EcdhSha2Nistp384 => "ecdh-sha2-nistp384",
            Self::EcdhSha2Nistp521 => "ecdh-sha2-nistp521",
            Self::DiffieHellmanGroup14Sha256 => "diffie-hellman-group14-sha256",
            Self::DiffieHellmanGroup16Sha512 => "diffie-hellman-group16-sha512",
            Self::DiffieHellmanGroup18Sha512 => "diffie-hellman-group18-sha512",
            Self::DiffieHellmanGroup14Sha1 => "diffie-hellman-group14-sha1",
            Self::DiffieHellmanGroup1Sha1 => "diffie-hellman-group1-sha1",
            Self::DiffieHellmanGroupExchangeSha256 => "diffie-hellman-group-exchange-sha256",
            Self::DiffieHellmanGroupExchangeSha1 => "diffie-hellman-group-exchange-sha1",
            Self::ExtInfoC => "ext-info-c",
            Self::ExtInfoS => "ext-info-s",
            Self::KexStrictClientV00Openssh => "kex-strict-c-v00@openssh.com",
            Self::KexStrictServerV00Openssh => "kex-strict-s-v00@openssh.com",
        }
    }

    pub fn from_wire_name(name: &str) -> Option<Self> {
        match name {
            "curve25519-sha256" => Some(Self::Curve25519Sha256),
            "curve25519-sha256@libssh.org" => Some(Self::Curve25519Sha256Libssh),
            "ecdh-sha2-nistp256" => Some(Self::EcdhSha2Nistp256),
            "ecdh-sha2-nistp384" => Some(Self::EcdhSha2Nistp384),
            "ecdh-sha2-nistp521" => Some(Self::EcdhSha2Nistp521),
            "diffie-hellman-group14-sha256" => Some(Self::DiffieHellmanGroup14Sha256),
            "diffie-hellman-group16-sha512" => Some(Self::DiffieHellmanGroup16Sha512),
            "diffie-hellman-group18-sha512" => Some(Self::DiffieHellmanGroup18Sha512),
            "diffie-hellman-group14-sha1" => Some(Self::DiffieHellmanGroup14Sha1),
            "diffie-hellman-group1-sha1" => Some(Self::DiffieHellmanGroup1Sha1),
            "diffie-hellman-group-exchange-sha256" => {
                Some(Self::DiffieHellmanGroupExchangeSha256)
            }
            "diffie-hellman-group-exchange-sha1" => {
                Some(Self::DiffieHellmanGroupExchangeSha1)
            }
            "ext-info-c" => Some(Self::ExtInfoC),
            "ext-info-s" => Some(Self::ExtInfoS),
            "kex-strict-c-v00@openssh.com" => Some(Self::KexStrictClientV00Openssh),
            "kex-strict-s-v00@openssh.com" => Some(Self::KexStrictServerV00Openssh),
            _ => None,
        }
    }

    pub fn security(self) -> crate::algorithms::SecurityClassification {
        use crate::algorithms::SecurityClassification;
        match self {
            Self::Curve25519Sha256 | Self::Curve25519Sha256Libssh => {
                SecurityClassification::Recommended
            }
            Self::EcdhSha2Nistp256 | Self::EcdhSha2Nistp384 | Self::EcdhSha2Nistp521 => {
                SecurityClassification::Recommended
            }
            Self::DiffieHellmanGroup16Sha512 | Self::DiffieHellmanGroup18Sha512 => {
                SecurityClassification::Recommended
            }
            Self::DiffieHellmanGroup14Sha256 => SecurityClassification::Acceptable,
            Self::DiffieHellmanGroupExchangeSha256 => SecurityClassification::Acceptable,
            Self::DiffieHellmanGroup14Sha1 | Self::DiffieHellmanGroupExchangeSha1 => {
                SecurityClassification::Weak
            }
            Self::DiffieHellmanGroup1Sha1 => SecurityClassification::Broken,
            // Pseudo-algorithms have no inherent security level
            Self::ExtInfoC | Self::ExtInfoS => SecurityClassification::Recommended,
            Self::KexStrictClientV00Openssh | Self::KexStrictServerV00Openssh => {
                SecurityClassification::Recommended
            }
        }
    }

    /// Returns the DH group size in bits (0 for non-DH algorithms).
    pub fn group_bits(self) -> u32 {
        match self {
            Self::DiffieHellmanGroup1Sha1 => 1024,
            Self::DiffieHellmanGroup14Sha1 | Self::DiffieHellmanGroup14Sha256 => 2048,
            Self::DiffieHellmanGroup16Sha512 => 4096,
            Self::DiffieHellmanGroup18Sha512 => 8192,
            Self::Curve25519Sha256 | Self::Curve25519Sha256Libssh => 256,
            Self::EcdhSha2Nistp256 => 256,
            Self::EcdhSha2Nistp384 => 384,
            Self::EcdhSha2Nistp521 => 521,
            _ => 0,
        }
    }

    pub fn is_pseudo(self) -> bool {
        matches!(
            self,
            Self::ExtInfoC
                | Self::ExtInfoS
                | Self::KexStrictClientV00Openssh
                | Self::KexStrictServerV00Openssh
        )
    }

    pub fn is_deprecated(self) -> bool {
        self.security().is_deprecated()
    }

    /// Canonical preference order (most preferred first).
    pub fn preference_order() -> &'static [Self] {
        &[
            Self::Curve25519Sha256,
            Self::Curve25519Sha256Libssh,
            Self::EcdhSha2Nistp521,
            Self::EcdhSha2Nistp384,
            Self::EcdhSha2Nistp256,
            Self::DiffieHellmanGroup18Sha512,
            Self::DiffieHellmanGroup16Sha512,
            Self::DiffieHellmanGroupExchangeSha256,
            Self::DiffieHellmanGroup14Sha256,
            Self::DiffieHellmanGroup14Sha1,
            Self::DiffieHellmanGroupExchangeSha1,
            Self::DiffieHellmanGroup1Sha1,
        ]
    }

    pub fn all_variants() -> &'static [Self] {
        &[
            Self::Curve25519Sha256,
            Self::Curve25519Sha256Libssh,
            Self::EcdhSha2Nistp256,
            Self::EcdhSha2Nistp384,
            Self::EcdhSha2Nistp521,
            Self::DiffieHellmanGroup14Sha256,
            Self::DiffieHellmanGroup16Sha512,
            Self::DiffieHellmanGroup18Sha512,
            Self::DiffieHellmanGroup14Sha1,
            Self::DiffieHellmanGroup1Sha1,
            Self::DiffieHellmanGroupExchangeSha256,
            Self::DiffieHellmanGroupExchangeSha1,
            Self::ExtInfoC,
            Self::ExtInfoS,
            Self::KexStrictClientV00Openssh,
            Self::KexStrictServerV00Openssh,
        ]
    }
}

impl fmt::Display for KexAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.wire_name())
    }
}

// ---------------------------------------------------------------------------
// KexInit
// ---------------------------------------------------------------------------

/// Parsed SSH_MSG_KEXINIT message (RFC 4253 §7.1).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KexInit {
    /// 16-byte random cookie.
    pub cookie: [u8; 16],
    /// Key exchange algorithms.
    pub kex_algorithms: Vec<String>,
    /// Server host key algorithms.
    pub server_host_key_algorithms: Vec<String>,
    /// Encryption algorithms client→server.
    pub encryption_algorithms_client_to_server: Vec<String>,
    /// Encryption algorithms server→client.
    pub encryption_algorithms_server_to_client: Vec<String>,
    /// MAC algorithms client→server.
    pub mac_algorithms_client_to_server: Vec<String>,
    /// MAC algorithms server→client.
    pub mac_algorithms_server_to_client: Vec<String>,
    /// Compression algorithms client→server.
    pub compression_algorithms_client_to_server: Vec<String>,
    /// Compression algorithms server→client.
    pub compression_algorithms_server_to_client: Vec<String>,
    /// Languages client→server (usually empty).
    pub languages_client_to_server: Vec<String>,
    /// Languages server→client (usually empty).
    pub languages_server_to_client: Vec<String>,
    /// Whether the sender's guess for the key exchange is included.
    pub first_kex_packet_follows: bool,
    /// Reserved — must be 0.
    pub reserved: u32,
}

impl KexInit {
    /// Does this KEXINIT indicate support for ext-info-c?
    pub fn has_ext_info_c(&self) -> bool {
        self.kex_algorithms.iter().any(|a| a == "ext-info-c")
    }

    /// Does this KEXINIT indicate support for ext-info-s?
    pub fn has_ext_info_s(&self) -> bool {
        self.kex_algorithms.iter().any(|a| a == "ext-info-s")
    }

    /// Does this KEXINIT include the strict-KEX client marker?
    pub fn has_strict_kex_client(&self) -> bool {
        self.kex_algorithms
            .iter()
            .any(|a| a == "kex-strict-c-v00@openssh.com")
    }

    /// Does this KEXINIT include the strict-KEX server marker?
    pub fn has_strict_kex_server(&self) -> bool {
        self.kex_algorithms
            .iter()
            .any(|a| a == "kex-strict-s-v00@openssh.com")
    }

    /// Returns all real (non-pseudo) KEX algorithm names.
    pub fn real_kex_algorithms(&self) -> Vec<&str> {
        self.kex_algorithms
            .iter()
            .filter(|name| {
                !matches!(
                    name.as_str(),
                    "ext-info-c"
                        | "ext-info-s"
                        | "kex-strict-c-v00@openssh.com"
                        | "kex-strict-s-v00@openssh.com"
                )
            })
            .map(|s| s.as_str())
            .collect()
    }

    /// Returns typed KEX algorithm list, skipping unknowns.
    pub fn typed_kex_algorithms(&self) -> Vec<KexAlgorithm> {
        self.kex_algorithms
            .iter()
            .filter_map(|n| KexAlgorithm::from_wire_name(n))
            .collect()
    }

    /// Returns typed host key algorithm list.
    pub fn typed_host_key_algorithms(&self) -> Vec<HostKeyAlgorithm> {
        self.server_host_key_algorithms
            .iter()
            .filter_map(|n| HostKeyAlgorithm::from_wire_name(n))
            .collect()
    }

    /// Returns typed encryption algorithms c→s.
    pub fn typed_encryption_c2s(&self) -> Vec<EncryptionAlgorithm> {
        self.encryption_algorithms_client_to_server
            .iter()
            .filter_map(|n| EncryptionAlgorithm::from_wire_name(n))
            .collect()
    }

    /// Returns typed encryption algorithms s→c.
    pub fn typed_encryption_s2c(&self) -> Vec<EncryptionAlgorithm> {
        self.encryption_algorithms_server_to_client
            .iter()
            .filter_map(|n| EncryptionAlgorithm::from_wire_name(n))
            .collect()
    }

    /// Returns typed MAC algorithms c→s.
    pub fn typed_mac_c2s(&self) -> Vec<SshMac> {
        self.mac_algorithms_client_to_server
            .iter()
            .filter_map(|n| SshMac::from_wire_name(n))
            .collect()
    }

    /// Returns typed MAC algorithms s→c.
    pub fn typed_mac_s2c(&self) -> Vec<SshMac> {
        self.mac_algorithms_server_to_client
            .iter()
            .filter_map(|n| SshMac::from_wire_name(n))
            .collect()
    }

    /// Returns typed compression c→s.
    pub fn typed_compression_c2s(&self) -> Vec<CompressionAlgorithm> {
        self.compression_algorithms_client_to_server
            .iter()
            .filter_map(|n| CompressionAlgorithm::from_wire_name(n))
            .collect()
    }

    /// Returns typed compression s→c.
    pub fn typed_compression_s2c(&self) -> Vec<CompressionAlgorithm> {
        self.compression_algorithms_server_to_client
            .iter()
            .filter_map(|n| CompressionAlgorithm::from_wire_name(n))
            .collect()
    }

    /// Serialize to the raw payload bytes (including the KEXINIT message type byte).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf: Vec<u8> = Vec::with_capacity(512);
        buf.push(SSH_MSG_KEXINIT);
        buf.extend_from_slice(&self.cookie);

        let lists: [&[String]; 10] = [
            &self.kex_algorithms,
            &self.server_host_key_algorithms,
            &self.encryption_algorithms_client_to_server,
            &self.encryption_algorithms_server_to_client,
            &self.mac_algorithms_client_to_server,
            &self.mac_algorithms_server_to_client,
            &self.compression_algorithms_client_to_server,
            &self.compression_algorithms_server_to_client,
            &self.languages_client_to_server,
            &self.languages_server_to_client,
        ];

        for list in &lists {
            let joined: String = list.join(",");
            let bytes = joined.as_bytes();
            buf.write_u32::<BigEndian>(bytes.len() as u32).unwrap();
            buf.write_all(bytes).unwrap();
        }

        buf.push(if self.first_kex_packet_follows { 1 } else { 0 });
        buf.write_u32::<BigEndian>(self.reserved).unwrap();

        buf
    }
}

impl fmt::Display for KexInit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "SSH_MSG_KEXINIT:")?;
        writeln!(f, "  kex: {}", self.kex_algorithms.join(","))?;
        writeln!(
            f,
            "  host_key: {}",
            self.server_host_key_algorithms.join(",")
        )?;
        writeln!(
            f,
            "  enc_c2s: {}",
            self.encryption_algorithms_client_to_server.join(",")
        )?;
        writeln!(
            f,
            "  enc_s2c: {}",
            self.encryption_algorithms_server_to_client.join(",")
        )?;
        writeln!(
            f,
            "  mac_c2s: {}",
            self.mac_algorithms_client_to_server.join(",")
        )?;
        writeln!(
            f,
            "  mac_s2c: {}",
            self.mac_algorithms_server_to_client.join(",")
        )?;
        writeln!(
            f,
            "  comp_c2s: {}",
            self.compression_algorithms_client_to_server.join(",")
        )?;
        writeln!(
            f,
            "  comp_s2c: {}",
            self.compression_algorithms_server_to_client.join(",")
        )?;
        writeln!(
            f,
            "  first_kex_packet_follows: {}",
            self.first_kex_packet_follows
        )?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// KexInitParser
// ---------------------------------------------------------------------------

/// Parses SSH_MSG_KEXINIT messages from raw payload bytes.
pub struct KexInitParser;

impl KexInitParser {
    /// Parse from the payload of an SSH packet (first byte should be SSH_MSG_KEXINIT).
    pub fn parse(data: &[u8]) -> SshResult<KexInit> {
        if data.is_empty() {
            return Err(SshError::ParseError("empty KEXINIT payload".into()));
        }

        let mut cursor = Cursor::new(data);
        let msg_type = cursor.read_u8().map_err(|e| SshError::ParseError(e.to_string()))?;
        if msg_type != SSH_MSG_KEXINIT {
            return Err(SshError::UnexpectedMessage {
                msg_type,
                state: "KEXINIT parsing".into(),
            });
        }

        // Cookie
        let mut cookie = [0u8; 16];
        cursor
            .read_exact(&mut cookie)
            .map_err(|e| SshError::ParseError(format!("cookie: {}", e)))?;

        // 10 name-lists
        let kex_algorithms = Self::read_name_list(&mut cursor)?;
        let server_host_key_algorithms = Self::read_name_list(&mut cursor)?;
        let enc_c2s = Self::read_name_list(&mut cursor)?;
        let enc_s2c = Self::read_name_list(&mut cursor)?;
        let mac_c2s = Self::read_name_list(&mut cursor)?;
        let mac_s2c = Self::read_name_list(&mut cursor)?;
        let comp_c2s = Self::read_name_list(&mut cursor)?;
        let comp_s2c = Self::read_name_list(&mut cursor)?;
        let lang_c2s = Self::read_name_list(&mut cursor)?;
        let lang_s2c = Self::read_name_list(&mut cursor)?;

        let first_kex_packet_follows = cursor
            .read_u8()
            .map_err(|e| SshError::ParseError(format!("first_kex_packet_follows: {}", e)))?
            != 0;

        let reserved = cursor
            .read_u32::<BigEndian>()
            .map_err(|e| SshError::ParseError(format!("reserved: {}", e)))?;

        Ok(KexInit {
            cookie,
            kex_algorithms,
            server_host_key_algorithms,
            encryption_algorithms_client_to_server: enc_c2s,
            encryption_algorithms_server_to_client: enc_s2c,
            mac_algorithms_client_to_server: mac_c2s,
            mac_algorithms_server_to_client: mac_s2c,
            compression_algorithms_client_to_server: comp_c2s,
            compression_algorithms_server_to_client: comp_s2c,
            languages_client_to_server: lang_c2s,
            languages_server_to_client: lang_s2c,
            first_kex_packet_follows,
            reserved,
        })
    }

    /// Read a name-list: uint32 length, then comma-separated ASCII names.
    fn read_name_list(cursor: &mut Cursor<&[u8]>) -> SshResult<Vec<String>> {
        let len = cursor
            .read_u32::<BigEndian>()
            .map_err(|e| SshError::ParseError(format!("name-list length: {}", e)))?;

        if len == 0 {
            return Ok(Vec::new());
        }

        let mut buf = vec![0u8; len as usize];
        cursor
            .read_exact(&mut buf)
            .map_err(|e| SshError::ParseError(format!("name-list data: {}", e)))?;

        let s = std::str::from_utf8(&buf)
            .map_err(|e| SshError::ParseError(format!("name-list utf8: {}", e)))?;

        Ok(s.split(',').map(|n| n.to_string()).collect())
    }
}

// ---------------------------------------------------------------------------
// KexInitBuilder
// ---------------------------------------------------------------------------

/// Builder for constructing SSH_MSG_KEXINIT messages.
#[derive(Debug, Clone)]
pub struct KexInitBuilder {
    cookie: [u8; 16],
    kex_algorithms: Vec<String>,
    server_host_key_algorithms: Vec<String>,
    encryption_c2s: Vec<String>,
    encryption_s2c: Vec<String>,
    mac_c2s: Vec<String>,
    mac_s2c: Vec<String>,
    compression_c2s: Vec<String>,
    compression_s2c: Vec<String>,
    languages_c2s: Vec<String>,
    languages_s2c: Vec<String>,
    first_kex_packet_follows: bool,
    strict_kex_client: bool,
    strict_kex_server: bool,
    ext_info_c: bool,
    ext_info_s: bool,
}

impl KexInitBuilder {
    pub fn new() -> Self {
        Self {
            cookie: [0u8; 16],
            kex_algorithms: Vec::new(),
            server_host_key_algorithms: Vec::new(),
            encryption_c2s: Vec::new(),
            encryption_s2c: Vec::new(),
            mac_c2s: Vec::new(),
            mac_s2c: Vec::new(),
            compression_c2s: vec!["none".into()],
            compression_s2c: vec!["none".into()],
            languages_c2s: Vec::new(),
            languages_s2c: Vec::new(),
            first_kex_packet_follows: false,
            strict_kex_client: false,
            strict_kex_server: false,
            ext_info_c: false,
            ext_info_s: false,
        }
    }

    pub fn cookie(mut self, cookie: [u8; 16]) -> Self {
        self.cookie = cookie;
        self
    }

    pub fn kex_algorithms(mut self, algs: Vec<String>) -> Self {
        self.kex_algorithms = algs;
        self
    }

    pub fn kex_algorithms_typed(mut self, algs: &[KexAlgorithm]) -> Self {
        self.kex_algorithms = algs.iter().map(|a| a.wire_name().to_string()).collect();
        self
    }

    pub fn server_host_key_algorithms(mut self, algs: Vec<String>) -> Self {
        self.server_host_key_algorithms = algs;
        self
    }

    pub fn host_key_algorithms_typed(mut self, algs: &[HostKeyAlgorithm]) -> Self {
        self.server_host_key_algorithms =
            algs.iter().map(|a| a.wire_name().to_string()).collect();
        self
    }

    pub fn encryption_c2s(mut self, algs: Vec<String>) -> Self {
        self.encryption_c2s = algs;
        self
    }

    pub fn encryption_c2s_typed(mut self, algs: &[EncryptionAlgorithm]) -> Self {
        self.encryption_c2s = algs.iter().map(|a| a.wire_name().to_string()).collect();
        self
    }

    pub fn encryption_s2c(mut self, algs: Vec<String>) -> Self {
        self.encryption_s2c = algs;
        self
    }

    pub fn encryption_s2c_typed(mut self, algs: &[EncryptionAlgorithm]) -> Self {
        self.encryption_s2c = algs.iter().map(|a| a.wire_name().to_string()).collect();
        self
    }

    pub fn mac_c2s(mut self, algs: Vec<String>) -> Self {
        self.mac_c2s = algs;
        self
    }

    pub fn mac_c2s_typed(mut self, algs: &[SshMac]) -> Self {
        self.mac_c2s = algs.iter().map(|a| a.wire_name().to_string()).collect();
        self
    }

    pub fn mac_s2c(mut self, algs: Vec<String>) -> Self {
        self.mac_s2c = algs;
        self
    }

    pub fn mac_s2c_typed(mut self, algs: &[SshMac]) -> Self {
        self.mac_s2c = algs.iter().map(|a| a.wire_name().to_string()).collect();
        self
    }

    pub fn compression_c2s(mut self, algs: Vec<String>) -> Self {
        self.compression_c2s = algs;
        self
    }

    pub fn compression_c2s_typed(mut self, algs: &[CompressionAlgorithm]) -> Self {
        self.compression_c2s = algs.iter().map(|a| a.wire_name().to_string()).collect();
        self
    }

    pub fn compression_s2c(mut self, algs: Vec<String>) -> Self {
        self.compression_s2c = algs;
        self
    }

    pub fn compression_s2c_typed(mut self, algs: &[CompressionAlgorithm]) -> Self {
        self.compression_s2c = algs.iter().map(|a| a.wire_name().to_string()).collect();
        self
    }

    pub fn first_kex_packet_follows(mut self, v: bool) -> Self {
        self.first_kex_packet_follows = v;
        self
    }

    pub fn with_strict_kex_client(mut self) -> Self {
        self.strict_kex_client = true;
        self
    }

    pub fn with_strict_kex_server(mut self) -> Self {
        self.strict_kex_server = true;
        self
    }

    pub fn with_ext_info_c(mut self) -> Self {
        self.ext_info_c = true;
        self
    }

    pub fn with_ext_info_s(mut self) -> Self {
        self.ext_info_s = true;
        self
    }

    /// Build the `KexInit` struct, appending pseudo-algorithm markers.
    pub fn build(self) -> KexInit {
        let mut kex_algs = self.kex_algorithms;

        if self.ext_info_c && !kex_algs.contains(&"ext-info-c".to_string()) {
            kex_algs.push("ext-info-c".to_string());
        }
        if self.ext_info_s && !kex_algs.contains(&"ext-info-s".to_string()) {
            kex_algs.push("ext-info-s".to_string());
        }
        if self.strict_kex_client
            && !kex_algs.contains(&"kex-strict-c-v00@openssh.com".to_string())
        {
            kex_algs.push("kex-strict-c-v00@openssh.com".to_string());
        }
        if self.strict_kex_server
            && !kex_algs.contains(&"kex-strict-s-v00@openssh.com".to_string())
        {
            kex_algs.push("kex-strict-s-v00@openssh.com".to_string());
        }

        KexInit {
            cookie: self.cookie,
            kex_algorithms: kex_algs,
            server_host_key_algorithms: self.server_host_key_algorithms,
            encryption_algorithms_client_to_server: self.encryption_c2s,
            encryption_algorithms_server_to_client: self.encryption_s2c,
            mac_algorithms_client_to_server: self.mac_c2s,
            mac_algorithms_server_to_client: self.mac_s2c,
            compression_algorithms_client_to_server: self.compression_c2s,
            compression_algorithms_server_to_client: self.compression_s2c,
            languages_client_to_server: self.languages_c2s,
            languages_server_to_client: self.languages_s2c,
            first_kex_packet_follows: self.first_kex_packet_follows,
            reserved: 0,
        }
    }

    /// Build and immediately serialize to bytes.
    pub fn build_bytes(self) -> Vec<u8> {
        self.build().to_bytes()
    }
}

impl Default for KexInitBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Creates a default OpenSSH-like client KexInit for testing/reference.
pub fn default_client_kex_init() -> KexInit {
    KexInitBuilder::new()
        .kex_algorithms_typed(&[
            KexAlgorithm::Curve25519Sha256,
            KexAlgorithm::Curve25519Sha256Libssh,
            KexAlgorithm::EcdhSha2Nistp256,
            KexAlgorithm::DiffieHellmanGroup16Sha512,
            KexAlgorithm::DiffieHellmanGroup14Sha256,
        ])
        .host_key_algorithms_typed(&[
            HostKeyAlgorithm::SshEd25519,
            HostKeyAlgorithm::EcdsaSha2Nistp256,
            HostKeyAlgorithm::RsaSha2_512,
            HostKeyAlgorithm::RsaSha2_256,
            HostKeyAlgorithm::SshRsa,
        ])
        .encryption_c2s_typed(&[
            EncryptionAlgorithm::Chacha20Poly1305,
            EncryptionAlgorithm::Aes256Gcm,
            EncryptionAlgorithm::Aes128Gcm,
            EncryptionAlgorithm::Aes256Ctr,
            EncryptionAlgorithm::Aes128Ctr,
        ])
        .encryption_s2c_typed(&[
            EncryptionAlgorithm::Chacha20Poly1305,
            EncryptionAlgorithm::Aes256Gcm,
            EncryptionAlgorithm::Aes128Gcm,
            EncryptionAlgorithm::Aes256Ctr,
            EncryptionAlgorithm::Aes128Ctr,
        ])
        .mac_c2s_typed(&[
            SshMac::HmacSha2_512Etm,
            SshMac::HmacSha2_256Etm,
            SshMac::HmacSha2_512,
            SshMac::HmacSha2_256,
        ])
        .mac_s2c_typed(&[
            SshMac::HmacSha2_512Etm,
            SshMac::HmacSha2_256Etm,
            SshMac::HmacSha2_512,
            SshMac::HmacSha2_256,
        ])
        .with_ext_info_c()
        .with_strict_kex_client()
        .build()
}

/// Creates a default OpenSSH-like server KexInit for testing/reference.
pub fn default_server_kex_init() -> KexInit {
    KexInitBuilder::new()
        .kex_algorithms_typed(&[
            KexAlgorithm::Curve25519Sha256,
            KexAlgorithm::Curve25519Sha256Libssh,
            KexAlgorithm::EcdhSha2Nistp256,
            KexAlgorithm::DiffieHellmanGroup16Sha512,
            KexAlgorithm::DiffieHellmanGroup14Sha256,
            KexAlgorithm::DiffieHellmanGroup14Sha1,
        ])
        .host_key_algorithms_typed(&[
            HostKeyAlgorithm::SshEd25519,
            HostKeyAlgorithm::RsaSha2_512,
            HostKeyAlgorithm::RsaSha2_256,
            HostKeyAlgorithm::SshRsa,
        ])
        .encryption_c2s_typed(&[
            EncryptionAlgorithm::Chacha20Poly1305,
            EncryptionAlgorithm::Aes256Gcm,
            EncryptionAlgorithm::Aes128Gcm,
            EncryptionAlgorithm::Aes256Ctr,
            EncryptionAlgorithm::Aes128Ctr,
            EncryptionAlgorithm::Aes256Cbc,
            EncryptionAlgorithm::Aes128Cbc,
        ])
        .encryption_s2c_typed(&[
            EncryptionAlgorithm::Chacha20Poly1305,
            EncryptionAlgorithm::Aes256Gcm,
            EncryptionAlgorithm::Aes128Gcm,
            EncryptionAlgorithm::Aes256Ctr,
            EncryptionAlgorithm::Aes128Ctr,
            EncryptionAlgorithm::Aes256Cbc,
            EncryptionAlgorithm::Aes128Cbc,
        ])
        .mac_c2s_typed(&[
            SshMac::HmacSha2_512Etm,
            SshMac::HmacSha2_256Etm,
            SshMac::HmacSha2_512,
            SshMac::HmacSha2_256,
            SshMac::HmacSha1,
        ])
        .mac_s2c_typed(&[
            SshMac::HmacSha2_512Etm,
            SshMac::HmacSha2_256Etm,
            SshMac::HmacSha2_512,
            SshMac::HmacSha2_256,
            SshMac::HmacSha1,
        ])
        .with_ext_info_s()
        .with_strict_kex_server()
        .build()
}

// ---------------------------------------------------------------------------
// Rekey state
// ---------------------------------------------------------------------------

/// Tracks re-keying state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RekeyState {
    /// Number of bytes transferred since last rekey.
    pub bytes_since_rekey: u64,
    /// Number of packets since last rekey.
    pub packets_since_rekey: u64,
    /// Threshold in bytes before recommending rekey (default 1 GiB).
    pub byte_threshold: u64,
    /// Threshold in packets before recommending rekey.
    pub packet_threshold: u64,
    /// Number of rekeys that have occurred.
    pub rekey_count: u32,
}

impl RekeyState {
    pub fn new() -> Self {
        Self {
            bytes_since_rekey: 0,
            packets_since_rekey: 0,
            byte_threshold: 1 << 30, // 1 GiB
            packet_threshold: 1 << 31,
            rekey_count: 0,
        }
    }

    pub fn record_packet(&mut self, size: u64) {
        self.bytes_since_rekey += size;
        self.packets_since_rekey += 1;
    }

    pub fn should_rekey(&self) -> bool {
        self.bytes_since_rekey >= self.byte_threshold
            || self.packets_since_rekey >= self.packet_threshold
    }

    pub fn complete_rekey(&mut self) {
        self.bytes_since_rekey = 0;
        self.packets_since_rekey = 0;
        self.rekey_count += 1;
    }
}

impl Default for RekeyState {
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

    #[test]
    fn kex_algorithm_roundtrip() {
        for alg in KexAlgorithm::all_variants() {
            let name = alg.wire_name();
            assert_eq!(
                KexAlgorithm::from_wire_name(name),
                Some(*alg),
                "roundtrip failed for {}",
                name
            );
        }
    }

    #[test]
    fn kex_init_serialize_parse_roundtrip() {
        let original = default_client_kex_init();
        let bytes = original.to_bytes();
        let parsed = KexInitParser::parse(&bytes).unwrap();

        assert_eq!(parsed.cookie, original.cookie);
        assert_eq!(parsed.kex_algorithms, original.kex_algorithms);
        assert_eq!(
            parsed.server_host_key_algorithms,
            original.server_host_key_algorithms
        );
        assert_eq!(
            parsed.encryption_algorithms_client_to_server,
            original.encryption_algorithms_client_to_server
        );
        assert_eq!(
            parsed.encryption_algorithms_server_to_client,
            original.encryption_algorithms_server_to_client
        );
        assert_eq!(
            parsed.mac_algorithms_client_to_server,
            original.mac_algorithms_client_to_server
        );
        assert_eq!(
            parsed.mac_algorithms_server_to_client,
            original.mac_algorithms_server_to_client
        );
        assert_eq!(
            parsed.compression_algorithms_client_to_server,
            original.compression_algorithms_client_to_server
        );
        assert_eq!(
            parsed.first_kex_packet_follows,
            original.first_kex_packet_follows
        );
    }

    #[test]
    fn kex_init_ext_info_markers() {
        let client = default_client_kex_init();
        assert!(client.has_ext_info_c());
        assert!(!client.has_ext_info_s());
        assert!(client.has_strict_kex_client());
        assert!(!client.has_strict_kex_server());

        let server = default_server_kex_init();
        assert!(!server.has_ext_info_c());
        assert!(server.has_ext_info_s());
        assert!(!server.has_strict_kex_client());
        assert!(server.has_strict_kex_server());
    }

    #[test]
    fn real_kex_algorithms_excludes_pseudo() {
        let ki = default_client_kex_init();
        let real = ki.real_kex_algorithms();
        assert!(
            !real.contains(&"ext-info-c"),
            "pseudo-alg should be filtered"
        );
        assert!(
            !real.contains(&"kex-strict-c-v00@openssh.com"),
            "pseudo-alg should be filtered"
        );
        assert!(real.contains(&"curve25519-sha256"));
    }

    #[test]
    fn kex_builder_default_compression() {
        let ki = KexInitBuilder::new().build();
        assert_eq!(
            ki.compression_algorithms_client_to_server,
            vec!["none".to_string()]
        );
    }

    #[test]
    fn kex_builder_no_duplicate_markers() {
        let ki = KexInitBuilder::new()
            .kex_algorithms(vec![
                "curve25519-sha256".into(),
                "ext-info-c".into(),
            ])
            .with_ext_info_c()
            .build();
        let count = ki
            .kex_algorithms
            .iter()
            .filter(|a| a.as_str() == "ext-info-c")
            .count();
        assert_eq!(count, 1, "ext-info-c should not be duplicated");
    }

    #[test]
    fn rekey_state() {
        let mut rs = RekeyState::new();
        assert!(!rs.should_rekey());

        for _ in 0..100 {
            rs.record_packet(100_000_000);
        }
        // 100 * 100MB = 10 GB > 1 GiB
        assert!(rs.should_rekey());

        rs.complete_rekey();
        assert!(!rs.should_rekey());
        assert_eq!(rs.rekey_count, 1);
    }

    #[test]
    fn kex_group_bits() {
        assert_eq!(KexAlgorithm::DiffieHellmanGroup1Sha1.group_bits(), 1024);
        assert_eq!(KexAlgorithm::DiffieHellmanGroup14Sha256.group_bits(), 2048);
        assert_eq!(KexAlgorithm::DiffieHellmanGroup16Sha512.group_bits(), 4096);
        assert_eq!(KexAlgorithm::Curve25519Sha256.group_bits(), 256);
    }

    #[test]
    fn kex_security_classification() {
        assert!(KexAlgorithm::Curve25519Sha256.security().is_safe());
        assert!(KexAlgorithm::DiffieHellmanGroup1Sha1.is_deprecated());
        assert!(!KexAlgorithm::DiffieHellmanGroup16Sha512.is_deprecated());
    }

    #[test]
    fn kex_init_parse_error_on_wrong_type() {
        let mut data = default_client_kex_init().to_bytes();
        data[0] = 99; // wrong msg type
        assert!(KexInitParser::parse(&data).is_err());
    }

    #[test]
    fn kex_init_parse_error_on_empty() {
        assert!(KexInitParser::parse(&[]).is_err());
    }

    #[test]
    fn typed_algorithms() {
        let ki = default_client_kex_init();
        let typed = ki.typed_encryption_c2s();
        assert_eq!(typed[0], EncryptionAlgorithm::Chacha20Poly1305);

        let macs = ki.typed_mac_c2s();
        assert_eq!(macs[0], SshMac::HmacSha2_512Etm);

        let hk = ki.typed_host_key_algorithms();
        assert_eq!(hk[0], HostKeyAlgorithm::SshEd25519);
    }

    #[test]
    fn kex_display() {
        let ki = default_client_kex_init();
        let s = format!("{}", ki);
        assert!(s.contains("SSH_MSG_KEXINIT"));
        assert!(s.contains("curve25519-sha256"));
    }
}
