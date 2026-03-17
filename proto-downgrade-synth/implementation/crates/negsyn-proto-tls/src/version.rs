//! TLS version negotiation and management.
//!
//! Implements TLS version types, ordering, negotiation logic per RFC 5246
//! and RFC 8446, version fallback mechanisms including TLS_FALLBACK_SCSV,
//! and downgrade protection sentinels.

use serde::{Deserialize, Serialize};
use std::fmt;

// ---------------------------------------------------------------------------
// TLS version type
// ---------------------------------------------------------------------------

/// Wire-level TLS protocol version as (major, minor) pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TlsVersion {
    pub major: u8,
    pub minor: u8,
}

impl TlsVersion {
    pub const SSL3_0: TlsVersion = TlsVersion { major: 3, minor: 0 };
    pub const TLS1_0: TlsVersion = TlsVersion { major: 3, minor: 1 };
    pub const TLS1_1: TlsVersion = TlsVersion { major: 3, minor: 2 };
    pub const TLS1_2: TlsVersion = TlsVersion { major: 3, minor: 3 };
    pub const TLS1_3: TlsVersion = TlsVersion { major: 3, minor: 4 };

    pub const fn new(major: u8, minor: u8) -> Self {
        TlsVersion { major, minor }
    }

    /// Encode as a 16-bit wire value: (major << 8) | minor.
    pub const fn to_u16(self) -> u16 {
        (self.major as u16) << 8 | self.minor as u16
    }

    /// Decode from 16-bit wire value.
    pub const fn from_u16(val: u16) -> Self {
        TlsVersion {
            major: (val >> 8) as u8,
            minor: (val & 0xFF) as u8,
        }
    }

    /// Whether this is a known, valid TLS/SSL version.
    pub fn is_known(&self) -> bool {
        matches!(
            (self.major, self.minor),
            (3, 0) | (3, 1) | (3, 2) | (3, 3) | (3, 4)
        )
    }

    /// Whether this version is deprecated (SSL 3.0, TLS 1.0, TLS 1.1).
    pub fn is_deprecated(&self) -> bool {
        matches!(
            (self.major, self.minor),
            (3, 0) | (3, 1) | (3, 2)
        )
    }

    /// Whether this version supports AEAD cipher suites natively.
    pub fn supports_aead(&self) -> bool {
        *self >= Self::TLS1_2
    }

    /// Whether this version uses the new TLS 1.3 handshake structure.
    pub fn is_tls13_or_later(&self) -> bool {
        *self >= Self::TLS1_3
    }

    /// Whether this version supports the extended master secret extension.
    pub fn supports_extended_master_secret(&self) -> bool {
        *self >= Self::TLS1_0
    }

    /// Whether this version supports encrypt-then-MAC.
    pub fn supports_encrypt_then_mac(&self) -> bool {
        *self >= Self::TLS1_0 && *self < Self::TLS1_3
    }

    /// Security level on a 0-4 scale.
    pub fn security_level(&self) -> u32 {
        match (self.major, self.minor) {
            (3, 0) => 0,
            (3, 1) => 1,
            (3, 2) => 2,
            (3, 3) => 3,
            (3, 4) => 4,
            _ => 0,
        }
    }

    /// Returns the set of features available for this version.
    pub fn feature_set(&self) -> VersionFeatureSet {
        VersionFeatureSet::for_version(*self)
    }

    /// Convert to `negsyn_types::ProtocolVersion`.
    pub fn to_protocol_version(self) -> negsyn_types::ProtocolVersion {
        negsyn_types::ProtocolVersion::tls(self.major, self.minor)
    }

    /// Convert from `negsyn_types::ProtocolVersion`.
    pub fn from_protocol_version(pv: negsyn_types::ProtocolVersion) -> Option<Self> {
        if pv.protocol != negsyn_types::ProtocolFamily::TLS {
            return None;
        }
        let v = TlsVersion::new(pv.major, pv.minor);
        if v.is_known() {
            Some(v)
        } else {
            None
        }
    }

    /// All known TLS versions in ascending order.
    pub fn all_versions() -> &'static [TlsVersion] {
        &[
            Self::SSL3_0,
            Self::TLS1_0,
            Self::TLS1_1,
            Self::TLS1_2,
            Self::TLS1_3,
        ]
    }

    /// Returns the record-layer version to use for the initial ClientHello.
    /// In TLS 1.3, the record layer version is set to TLS 1.0 (3,1) for
    /// compatibility, while the actual version is negotiated via
    /// supported_versions extension.
    pub fn record_layer_version(&self) -> TlsVersion {
        if *self >= Self::TLS1_3 {
            Self::TLS1_0
        } else {
            *self
        }
    }
}

impl PartialOrd for TlsVersion {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TlsVersion {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.to_u16().cmp(&other.to_u16())
    }
}

impl fmt::Display for TlsVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (self.major, self.minor) {
            (3, 0) => write!(f, "SSL 3.0"),
            (3, 1) => write!(f, "TLS 1.0"),
            (3, 2) => write!(f, "TLS 1.1"),
            (3, 3) => write!(f, "TLS 1.2"),
            (3, 4) => write!(f, "TLS 1.3"),
            _ => write!(f, "Unknown({}.{})", self.major, self.minor),
        }
    }
}

// ---------------------------------------------------------------------------
// Version feature sets
// ---------------------------------------------------------------------------

/// Features available in a particular TLS version.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VersionFeatureSet {
    pub version: TlsVersion,
    pub has_explicit_iv: bool,
    pub has_aead_support: bool,
    pub has_sha256_prf: bool,
    pub has_session_tickets: bool,
    pub has_extensions: bool,
    pub has_signature_algorithms: bool,
    pub has_supported_versions_ext: bool,
    pub has_key_share_ext: bool,
    pub has_psk_support: bool,
    pub has_zero_rtt: bool,
    pub has_encrypted_extensions: bool,
    pub has_certificate_verify_always: bool,
    pub has_renegotiation: bool,
    pub has_compression: bool,
    pub has_change_cipher_spec: bool,
    pub max_record_size: usize,
    pub prf_algorithm: PrfAlgorithm,
}

/// Pseudorandom function used by each TLS version.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PrfAlgorithm {
    Md5Sha1Combined,
    Sha256,
    Sha384,
    HkdfSha256,
    HkdfSha384,
}

impl fmt::Display for PrfAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Md5Sha1Combined => write!(f, "MD5+SHA-1"),
            Self::Sha256 => write!(f, "SHA-256"),
            Self::Sha384 => write!(f, "SHA-384"),
            Self::HkdfSha256 => write!(f, "HKDF-SHA-256"),
            Self::HkdfSha384 => write!(f, "HKDF-SHA-384"),
        }
    }
}

impl VersionFeatureSet {
    pub fn for_version(version: TlsVersion) -> Self {
        match (version.major, version.minor) {
            (3, 0) => Self {
                version,
                has_explicit_iv: false,
                has_aead_support: false,
                has_sha256_prf: false,
                has_session_tickets: false,
                has_extensions: false,
                has_signature_algorithms: false,
                has_supported_versions_ext: false,
                has_key_share_ext: false,
                has_psk_support: false,
                has_zero_rtt: false,
                has_encrypted_extensions: false,
                has_certificate_verify_always: false,
                has_renegotiation: true,
                has_compression: true,
                has_change_cipher_spec: true,
                max_record_size: 16384,
                prf_algorithm: PrfAlgorithm::Md5Sha1Combined,
            },
            (3, 1) => Self {
                version,
                has_explicit_iv: false,
                has_aead_support: false,
                has_sha256_prf: false,
                has_session_tickets: true,
                has_extensions: true,
                has_signature_algorithms: false,
                has_supported_versions_ext: false,
                has_key_share_ext: false,
                has_psk_support: false,
                has_zero_rtt: false,
                has_encrypted_extensions: false,
                has_certificate_verify_always: false,
                has_renegotiation: true,
                has_compression: true,
                has_change_cipher_spec: true,
                max_record_size: 16384,
                prf_algorithm: PrfAlgorithm::Md5Sha1Combined,
            },
            (3, 2) => Self {
                version,
                has_explicit_iv: true,
                has_aead_support: false,
                has_sha256_prf: false,
                has_session_tickets: true,
                has_extensions: true,
                has_signature_algorithms: false,
                has_supported_versions_ext: false,
                has_key_share_ext: false,
                has_psk_support: false,
                has_zero_rtt: false,
                has_encrypted_extensions: false,
                has_certificate_verify_always: false,
                has_renegotiation: true,
                has_compression: true,
                has_change_cipher_spec: true,
                max_record_size: 16384,
                prf_algorithm: PrfAlgorithm::Md5Sha1Combined,
            },
            (3, 3) => Self {
                version,
                has_explicit_iv: true,
                has_aead_support: true,
                has_sha256_prf: true,
                has_session_tickets: true,
                has_extensions: true,
                has_signature_algorithms: true,
                has_supported_versions_ext: false,
                has_key_share_ext: false,
                has_psk_support: false,
                has_zero_rtt: false,
                has_encrypted_extensions: false,
                has_certificate_verify_always: false,
                has_renegotiation: true,
                has_compression: true,
                has_change_cipher_spec: true,
                max_record_size: 16384,
                prf_algorithm: PrfAlgorithm::Sha256,
            },
            (3, 4) => Self {
                version,
                has_explicit_iv: true,
                has_aead_support: true,
                has_sha256_prf: true,
                has_session_tickets: true,
                has_extensions: true,
                has_signature_algorithms: true,
                has_supported_versions_ext: true,
                has_key_share_ext: true,
                has_psk_support: true,
                has_zero_rtt: true,
                has_encrypted_extensions: true,
                has_certificate_verify_always: true,
                has_renegotiation: false,
                has_compression: false,
                has_change_cipher_spec: false,
                max_record_size: 16384 + 256,
                prf_algorithm: PrfAlgorithm::HkdfSha256,
            },
            _ => Self {
                version,
                has_explicit_iv: false,
                has_aead_support: false,
                has_sha256_prf: false,
                has_session_tickets: false,
                has_extensions: false,
                has_signature_algorithms: false,
                has_supported_versions_ext: false,
                has_key_share_ext: false,
                has_psk_support: false,
                has_zero_rtt: false,
                has_encrypted_extensions: false,
                has_certificate_verify_always: false,
                has_renegotiation: false,
                has_compression: false,
                has_change_cipher_spec: false,
                max_record_size: 16384,
                prf_algorithm: PrfAlgorithm::Md5Sha1Combined,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// TLS_FALLBACK_SCSV
// ---------------------------------------------------------------------------

/// The TLS_FALLBACK_SCSV signaling cipher suite value (RFC 7507).
pub const TLS_FALLBACK_SCSV: u16 = 0x5600;

/// The TLS_EMPTY_RENEGOTIATION_INFO_SCSV value (RFC 5746).
pub const TLS_EMPTY_RENEGOTIATION_INFO_SCSV: u16 = 0x00FF;

// ---------------------------------------------------------------------------
// Downgrade sentinels (RFC 8446 §4.1.3)
// ---------------------------------------------------------------------------

/// The last 8 bytes of ServerHello.random when TLS 1.3 server negotiates
/// TLS 1.2 with a TLS 1.3-capable client.
pub const DOWNGRADE_SENTINEL_TLS12: [u8; 8] = [0x44, 0x4F, 0x57, 0x4E, 0x47, 0x52, 0x44, 0x01];

/// The last 8 bytes of ServerHello.random when TLS 1.3 server negotiates
/// TLS 1.1 or below.
pub const DOWNGRADE_SENTINEL_TLS11: [u8; 8] = [0x44, 0x4F, 0x57, 0x4E, 0x47, 0x52, 0x44, 0x00];

// ---------------------------------------------------------------------------
// Version negotiation engine
// ---------------------------------------------------------------------------

/// Configuration for version negotiation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionNegotiationConfig {
    pub min_version: TlsVersion,
    pub max_version: TlsVersion,
    pub allow_fallback: bool,
    pub enforce_downgrade_sentinel: bool,
    pub supported_versions: Vec<TlsVersion>,
}

impl Default for VersionNegotiationConfig {
    fn default() -> Self {
        Self {
            min_version: TlsVersion::TLS1_2,
            max_version: TlsVersion::TLS1_3,
            allow_fallback: false,
            enforce_downgrade_sentinel: true,
            supported_versions: vec![TlsVersion::TLS1_2, TlsVersion::TLS1_3],
        }
    }
}

/// Result of version negotiation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VersionNegotiationResult {
    /// Successfully negotiated a version.
    Success(TlsVersion),
    /// Server used supported_versions extension (TLS 1.3 style).
    SuccessViaSupportedVersions(TlsVersion),
    /// No common version; negotiation failed.
    NoCommonVersion,
    /// Detected an inappropriate fallback (SCSV triggered).
    InappropriateFallback {
        attempted: TlsVersion,
        server_max: TlsVersion,
    },
    /// Downgrade sentinel detected in random.
    DowngradeSentinelDetected {
        negotiated: TlsVersion,
        sentinel_for: TlsVersion,
    },
}

/// Engine that handles TLS version negotiation.
#[derive(Debug, Clone)]
pub struct VersionNegotiator {
    config: VersionNegotiationConfig,
}

impl VersionNegotiator {
    pub fn new(config: VersionNegotiationConfig) -> Self {
        Self { config }
    }

    pub fn with_defaults() -> Self {
        Self::new(VersionNegotiationConfig::default())
    }

    /// Negotiate version from a client perspective.
    /// `client_max` is the highest version the client supports.
    /// `server_version` is what the server replied with.
    /// `server_random` is the 32-byte ServerHello random (for sentinel check).
    /// `server_supported_versions` is from the supported_versions extension, if present.
    pub fn negotiate_client_side(
        &self,
        client_max: TlsVersion,
        server_version: TlsVersion,
        server_random: &[u8; 32],
        server_supported_versions: Option<TlsVersion>,
    ) -> VersionNegotiationResult {
        // If supported_versions extension is present, use it (TLS 1.3 mechanism).
        if let Some(sv) = server_supported_versions {
            if !self.config.supported_versions.contains(&sv) {
                return VersionNegotiationResult::NoCommonVersion;
            }
            if sv < self.config.min_version {
                return VersionNegotiationResult::NoCommonVersion;
            }
            // Check downgrade sentinel even when using supported_versions.
            if self.config.enforce_downgrade_sentinel && sv < TlsVersion::TLS1_3 {
                if let Some(sentinel_ver) = self.check_downgrade_sentinel(server_random) {
                    return VersionNegotiationResult::DowngradeSentinelDetected {
                        negotiated: sv,
                        sentinel_for: sentinel_ver,
                    };
                }
            }
            return VersionNegotiationResult::SuccessViaSupportedVersions(sv);
        }

        // Legacy negotiation: server picks min(client_max, server_max).
        if !server_version.is_known() {
            return VersionNegotiationResult::NoCommonVersion;
        }
        if server_version > client_max {
            return VersionNegotiationResult::NoCommonVersion;
        }
        if server_version < self.config.min_version {
            return VersionNegotiationResult::NoCommonVersion;
        }

        // Check downgrade sentinel.
        if self.config.enforce_downgrade_sentinel && client_max >= TlsVersion::TLS1_3 {
            if let Some(sentinel_ver) = self.check_downgrade_sentinel(server_random) {
                return VersionNegotiationResult::DowngradeSentinelDetected {
                    negotiated: server_version,
                    sentinel_for: sentinel_ver,
                };
            }
        }

        VersionNegotiationResult::Success(server_version)
    }

    /// Negotiate version from a server perspective.
    /// `client_version` is the client's advertised maximum.
    /// `client_cipher_suites` may include TLS_FALLBACK_SCSV.
    /// `client_supported_versions` is from the supported_versions extension.
    pub fn negotiate_server_side(
        &self,
        client_version: TlsVersion,
        client_cipher_suites: &[u16],
        client_supported_versions: Option<&[TlsVersion]>,
    ) -> VersionNegotiationResult {
        let has_fallback_scsv = client_cipher_suites.contains(&TLS_FALLBACK_SCSV);

        // If client sends supported_versions extension, pick the highest common.
        if let Some(client_versions) = client_supported_versions {
            let mut best: Option<TlsVersion> = None;
            for &cv in client_versions {
                if self.config.supported_versions.contains(&cv)
                    && cv >= self.config.min_version
                    && cv <= self.config.max_version
                {
                    if best.map_or(true, |b| cv > b) {
                        best = Some(cv);
                    }
                }
            }
            return match best {
                Some(v) => VersionNegotiationResult::SuccessViaSupportedVersions(v),
                None => VersionNegotiationResult::NoCommonVersion,
            };
        }

        // Legacy negotiation.
        let negotiated = std::cmp::min(client_version, self.config.max_version);
        if negotiated < self.config.min_version {
            return VersionNegotiationResult::NoCommonVersion;
        }

        // Check TLS_FALLBACK_SCSV: if client includes it and is offering
        // a lower version than the server's maximum, reject.
        if has_fallback_scsv && negotiated < self.config.max_version {
            return VersionNegotiationResult::InappropriateFallback {
                attempted: negotiated,
                server_max: self.config.max_version,
            };
        }

        VersionNegotiationResult::Success(negotiated)
    }

    /// Check the last 8 bytes of server random for downgrade sentinel.
    /// Returns the version the sentinel corresponds to, if detected.
    fn check_downgrade_sentinel(&self, random: &[u8; 32]) -> Option<TlsVersion> {
        let tail: [u8; 8] = random[24..32].try_into().unwrap();
        if tail == DOWNGRADE_SENTINEL_TLS12 {
            Some(TlsVersion::TLS1_2)
        } else if tail == DOWNGRADE_SENTINEL_TLS11 {
            Some(TlsVersion::TLS1_1)
        } else {
            None
        }
    }

    /// Generate the server random with appropriate downgrade sentinel embedded.
    /// `base_random` is the 32-byte random value to start with.
    /// `negotiated` is the version being negotiated.
    /// `server_max` is the server's highest supported version.
    pub fn embed_downgrade_sentinel(
        base_random: &mut [u8; 32],
        negotiated: TlsVersion,
        server_max: TlsVersion,
    ) {
        if server_max >= TlsVersion::TLS1_3 && negotiated <= TlsVersion::TLS1_2 {
            if negotiated == TlsVersion::TLS1_2 {
                base_random[24..32].copy_from_slice(&DOWNGRADE_SENTINEL_TLS12);
            } else {
                base_random[24..32].copy_from_slice(&DOWNGRADE_SENTINEL_TLS11);
            }
        }
    }

    /// Returns the versions this negotiator supports.
    pub fn supported_versions(&self) -> &[TlsVersion] {
        &self.config.supported_versions
    }

    /// Check if a version is within the configured range.
    pub fn is_version_acceptable(&self, version: TlsVersion) -> bool {
        version >= self.config.min_version
            && version <= self.config.max_version
            && self.config.supported_versions.contains(&version)
    }
}

// ---------------------------------------------------------------------------
// Version comparison utilities
// ---------------------------------------------------------------------------

/// Returns the highest version from a list, or None if empty.
pub fn highest_version(versions: &[TlsVersion]) -> Option<TlsVersion> {
    versions.iter().copied().max()
}

/// Returns the lowest version from a list, or None if empty.
pub fn lowest_version(versions: &[TlsVersion]) -> Option<TlsVersion> {
    versions.iter().copied().min()
}

/// Filter versions to only those within [min, max].
pub fn filter_version_range(
    versions: &[TlsVersion],
    min: TlsVersion,
    max: TlsVersion,
) -> Vec<TlsVersion> {
    versions
        .iter()
        .copied()
        .filter(|v| *v >= min && *v <= max)
        .collect()
}

/// Check if a version downgrade occurred.
pub fn is_downgrade(offered_max: TlsVersion, negotiated: TlsVersion) -> bool {
    negotiated < offered_max
}

/// Compute the "downgrade distance" (number of version steps).
pub fn downgrade_distance(from: TlsVersion, to: TlsVersion) -> i32 {
    from.security_level() as i32 - to.security_level() as i32
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_ordering() {
        assert!(TlsVersion::SSL3_0 < TlsVersion::TLS1_0);
        assert!(TlsVersion::TLS1_0 < TlsVersion::TLS1_1);
        assert!(TlsVersion::TLS1_1 < TlsVersion::TLS1_2);
        assert!(TlsVersion::TLS1_2 < TlsVersion::TLS1_3);
    }

    #[test]
    fn test_version_u16_roundtrip() {
        for &v in TlsVersion::all_versions() {
            let encoded = v.to_u16();
            let decoded = TlsVersion::from_u16(encoded);
            assert_eq!(v, decoded);
        }
    }

    #[test]
    fn test_version_is_known() {
        assert!(TlsVersion::SSL3_0.is_known());
        assert!(TlsVersion::TLS1_3.is_known());
        assert!(!TlsVersion::new(2, 0).is_known());
        assert!(!TlsVersion::new(3, 5).is_known());
    }

    #[test]
    fn test_version_deprecated() {
        assert!(TlsVersion::SSL3_0.is_deprecated());
        assert!(TlsVersion::TLS1_0.is_deprecated());
        assert!(TlsVersion::TLS1_1.is_deprecated());
        assert!(!TlsVersion::TLS1_2.is_deprecated());
        assert!(!TlsVersion::TLS1_3.is_deprecated());
    }

    #[test]
    fn test_record_layer_version() {
        assert_eq!(TlsVersion::TLS1_2.record_layer_version(), TlsVersion::TLS1_2);
        assert_eq!(TlsVersion::TLS1_3.record_layer_version(), TlsVersion::TLS1_0);
    }

    #[test]
    fn test_security_level() {
        assert_eq!(TlsVersion::SSL3_0.security_level(), 0);
        assert_eq!(TlsVersion::TLS1_0.security_level(), 1);
        assert_eq!(TlsVersion::TLS1_3.security_level(), 4);
    }

    #[test]
    fn test_feature_set_ssl3() {
        let fs = TlsVersion::SSL3_0.feature_set();
        assert!(!fs.has_extensions);
        assert!(!fs.has_aead_support);
        assert!(fs.has_compression);
        assert!(fs.has_renegotiation);
    }

    #[test]
    fn test_feature_set_tls13() {
        let fs = TlsVersion::TLS1_3.feature_set();
        assert!(fs.has_extensions);
        assert!(fs.has_aead_support);
        assert!(fs.has_zero_rtt);
        assert!(!fs.has_compression);
        assert!(!fs.has_renegotiation);
        assert!(fs.has_key_share_ext);
    }

    #[test]
    fn test_downgrade_sentinel_embed_and_detect() {
        let mut random = [0u8; 32];
        VersionNegotiator::embed_downgrade_sentinel(
            &mut random,
            TlsVersion::TLS1_2,
            TlsVersion::TLS1_3,
        );
        let tail: [u8; 8] = random[24..32].try_into().unwrap();
        assert_eq!(tail, DOWNGRADE_SENTINEL_TLS12);

        let mut random2 = [0u8; 32];
        VersionNegotiator::embed_downgrade_sentinel(
            &mut random2,
            TlsVersion::TLS1_1,
            TlsVersion::TLS1_3,
        );
        let tail2: [u8; 8] = random2[24..32].try_into().unwrap();
        assert_eq!(tail2, DOWNGRADE_SENTINEL_TLS11);
    }

    #[test]
    fn test_server_negotiation_basic() {
        let negotiator = VersionNegotiator::with_defaults();
        let result = negotiator.negotiate_server_side(
            TlsVersion::TLS1_3,
            &[0x1301, 0x1302],
            Some(&[TlsVersion::TLS1_3, TlsVersion::TLS1_2]),
        );
        assert_eq!(
            result,
            VersionNegotiationResult::SuccessViaSupportedVersions(TlsVersion::TLS1_3)
        );
    }

    #[test]
    fn test_server_negotiation_no_common() {
        let config = VersionNegotiationConfig {
            min_version: TlsVersion::TLS1_3,
            max_version: TlsVersion::TLS1_3,
            supported_versions: vec![TlsVersion::TLS1_3],
            ..Default::default()
        };
        let negotiator = VersionNegotiator::new(config);
        let result = negotiator.negotiate_server_side(
            TlsVersion::TLS1_2,
            &[0xC02F],
            None,
        );
        assert_eq!(result, VersionNegotiationResult::NoCommonVersion);
    }

    #[test]
    fn test_fallback_scsv_detection() {
        let config = VersionNegotiationConfig {
            min_version: TlsVersion::TLS1_0,
            max_version: TlsVersion::TLS1_3,
            supported_versions: vec![
                TlsVersion::TLS1_0,
                TlsVersion::TLS1_1,
                TlsVersion::TLS1_2,
                TlsVersion::TLS1_3,
            ],
            ..Default::default()
        };
        let negotiator = VersionNegotiator::new(config);
        let result = negotiator.negotiate_server_side(
            TlsVersion::TLS1_2,
            &[0xC02F, TLS_FALLBACK_SCSV],
            None,
        );
        match result {
            VersionNegotiationResult::InappropriateFallback { attempted, server_max } => {
                assert_eq!(attempted, TlsVersion::TLS1_2);
                assert_eq!(server_max, TlsVersion::TLS1_3);
            }
            _ => panic!("Expected InappropriateFallback, got {:?}", result),
        }
    }

    #[test]
    fn test_client_negotiation_with_sentinel() {
        let negotiator = VersionNegotiator::with_defaults();
        let mut random = [0xABu8; 32];
        random[24..32].copy_from_slice(&DOWNGRADE_SENTINEL_TLS12);
        let result = negotiator.negotiate_client_side(
            TlsVersion::TLS1_3,
            TlsVersion::TLS1_2,
            &random,
            None,
        );
        match result {
            VersionNegotiationResult::DowngradeSentinelDetected { .. } => {}
            _ => panic!("Expected DowngradeSentinelDetected, got {:?}", result),
        }
    }

    #[test]
    fn test_client_negotiation_success() {
        let negotiator = VersionNegotiator::with_defaults();
        let random = [0x42u8; 32];
        let result = negotiator.negotiate_client_side(
            TlsVersion::TLS1_3,
            TlsVersion::TLS1_2,
            &random,
            Some(TlsVersion::TLS1_3),
        );
        assert_eq!(
            result,
            VersionNegotiationResult::SuccessViaSupportedVersions(TlsVersion::TLS1_3)
        );
    }

    #[test]
    fn test_is_downgrade() {
        assert!(is_downgrade(TlsVersion::TLS1_3, TlsVersion::TLS1_2));
        assert!(!is_downgrade(TlsVersion::TLS1_2, TlsVersion::TLS1_2));
        assert!(!is_downgrade(TlsVersion::TLS1_2, TlsVersion::TLS1_3));
    }

    #[test]
    fn test_downgrade_distance() {
        assert_eq!(downgrade_distance(TlsVersion::TLS1_3, TlsVersion::SSL3_0), 4);
        assert_eq!(downgrade_distance(TlsVersion::TLS1_2, TlsVersion::TLS1_2), 0);
    }

    #[test]
    fn test_highest_lowest_version() {
        let versions = vec![TlsVersion::TLS1_0, TlsVersion::TLS1_3, TlsVersion::TLS1_1];
        assert_eq!(highest_version(&versions), Some(TlsVersion::TLS1_3));
        assert_eq!(lowest_version(&versions), Some(TlsVersion::TLS1_0));
        assert_eq!(highest_version(&[]), None);
    }

    #[test]
    fn test_filter_version_range() {
        let all = vec![
            TlsVersion::SSL3_0,
            TlsVersion::TLS1_0,
            TlsVersion::TLS1_1,
            TlsVersion::TLS1_2,
            TlsVersion::TLS1_3,
        ];
        let filtered = filter_version_range(&all, TlsVersion::TLS1_1, TlsVersion::TLS1_2);
        assert_eq!(filtered, vec![TlsVersion::TLS1_1, TlsVersion::TLS1_2]);
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", TlsVersion::SSL3_0), "SSL 3.0");
        assert_eq!(format!("{}", TlsVersion::TLS1_0), "TLS 1.0");
        assert_eq!(format!("{}", TlsVersion::TLS1_3), "TLS 1.3");
    }

    #[test]
    fn test_to_from_protocol_version() {
        let pv = TlsVersion::TLS1_2.to_protocol_version();
        assert_eq!(pv, negsyn_types::ProtocolVersion::tls12());
        let back = TlsVersion::from_protocol_version(pv);
        assert_eq!(back, Some(TlsVersion::TLS1_2));
    }
}
