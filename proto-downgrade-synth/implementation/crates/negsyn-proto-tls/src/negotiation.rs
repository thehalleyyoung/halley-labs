//! TLS negotiation engine.
//!
//! Implements cipher suite selection, extension negotiation, compression
//! method selection, FIPS mode filtering, and policy-based controls for
//! both client and server perspectives.

use crate::cipher_suites::{
    CipherSuiteRegistry, KeyExchange, SecurityLevel, TlsCipherSuite,
    select_cipher_suite, select_cipher_suite_filtered,
};
use crate::extensions::{
    ExtensionNegotiationConfig, TlsExtension, NamedGroup, SignatureScheme,
    negotiate_extensions,
};
use crate::handshake::{ClientHello, CompressionMethod, ServerHello};
use crate::version::{
    TlsVersion, VersionNegotiationConfig, VersionNegotiator, VersionNegotiationResult,
    TLS_FALLBACK_SCSV,
};
use serde::{Deserialize, Serialize};
use std::fmt;

// ---------------------------------------------------------------------------
// Negotiation policy
// ---------------------------------------------------------------------------

/// Policy that constrains what can be negotiated.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NegotiationPolicy {
    /// Minimum acceptable security level for cipher suites.
    pub min_security_level: SecurityLevel,
    /// Whether to require forward secrecy.
    pub require_forward_secrecy: bool,
    /// Whether FIPS-only mode is enabled.
    pub fips_mode: bool,
    /// Whether to allow renegotiation.
    pub allow_renegotiation: bool,
    /// Whether to require extended master secret.
    pub require_extended_master_secret: bool,
    /// Whether to allow compression (CRIME mitigation).
    pub allow_compression: bool,
    /// Minimum TLS version.
    pub min_version: TlsVersion,
    /// Maximum TLS version.
    pub max_version: TlsVersion,
    /// Explicitly allowed cipher suite IDs (empty = all allowed).
    pub allowed_cipher_suites: Vec<u16>,
    /// Explicitly blocked cipher suite IDs.
    pub blocked_cipher_suites: Vec<u16>,
    /// Whether server preference ordering is used.
    pub server_preference: bool,
    /// Allowed named groups (empty = all).
    pub allowed_groups: Vec<NamedGroup>,
    /// Allowed signature schemes (empty = all).
    pub allowed_signatures: Vec<SignatureScheme>,
}

impl Default for NegotiationPolicy {
    fn default() -> Self {
        Self {
            min_security_level: SecurityLevel::Legacy,
            require_forward_secrecy: false,
            fips_mode: false,
            allow_renegotiation: true,
            require_extended_master_secret: false,
            allow_compression: false,
            min_version: TlsVersion::TLS1_2,
            max_version: TlsVersion::TLS1_3,
            allowed_cipher_suites: Vec::new(),
            blocked_cipher_suites: Vec::new(),
            server_preference: true,
            allowed_groups: Vec::new(),
            allowed_signatures: Vec::new(),
        }
    }
}

impl NegotiationPolicy {
    /// A strict modern policy (TLS 1.2+ with AEAD and forward secrecy).
    pub fn strict() -> Self {
        Self {
            min_security_level: SecurityLevel::Secure,
            require_forward_secrecy: true,
            fips_mode: false,
            allow_renegotiation: false,
            require_extended_master_secret: true,
            allow_compression: false,
            min_version: TlsVersion::TLS1_2,
            max_version: TlsVersion::TLS1_3,
            allowed_cipher_suites: Vec::new(),
            blocked_cipher_suites: Vec::new(),
            server_preference: true,
            allowed_groups: vec![
                NamedGroup::X25519,
                NamedGroup::Secp256r1,
                NamedGroup::Secp384r1,
            ],
            allowed_signatures: vec![
                SignatureScheme::EcdsaSecp256r1Sha256,
                SignatureScheme::RsaPssRsaeSha256,
                SignatureScheme::RsaPssRsaeSha384,
                SignatureScheme::Ed25519,
            ],
        }
    }

    /// A FIPS-compliant policy.
    pub fn fips() -> Self {
        Self {
            min_security_level: SecurityLevel::Legacy,
            require_forward_secrecy: false,
            fips_mode: true,
            allow_renegotiation: true,
            require_extended_master_secret: true,
            allow_compression: false,
            min_version: TlsVersion::TLS1_2,
            max_version: TlsVersion::TLS1_3,
            allowed_cipher_suites: Vec::new(),
            blocked_cipher_suites: Vec::new(),
            server_preference: true,
            allowed_groups: vec![
                NamedGroup::Secp256r1,
                NamedGroup::Secp384r1,
                NamedGroup::Secp521r1,
            ],
            allowed_signatures: vec![
                SignatureScheme::RsaPkcs1Sha256,
                SignatureScheme::RsaPkcs1Sha384,
                SignatureScheme::EcdsaSecp256r1Sha256,
                SignatureScheme::EcdsaSecp384r1Sha384,
                SignatureScheme::RsaPssRsaeSha256,
                SignatureScheme::RsaPssRsaeSha384,
            ],
        }
    }

    /// A permissive legacy policy (for testing, not recommended).
    pub fn permissive() -> Self {
        Self {
            min_security_level: SecurityLevel::Insecure,
            require_forward_secrecy: false,
            fips_mode: false,
            allow_renegotiation: true,
            require_extended_master_secret: false,
            allow_compression: true,
            min_version: TlsVersion::SSL3_0,
            max_version: TlsVersion::TLS1_3,
            allowed_cipher_suites: Vec::new(),
            blocked_cipher_suites: Vec::new(),
            server_preference: false,
            allowed_groups: Vec::new(),
            allowed_signatures: Vec::new(),
        }
    }

    /// Check if a cipher suite ID is acceptable under this policy.
    pub fn is_cipher_suite_acceptable(
        &self,
        id: u16,
        registry: &CipherSuiteRegistry,
    ) -> bool {
        if self.blocked_cipher_suites.contains(&id) {
            return false;
        }
        if !self.allowed_cipher_suites.is_empty() && !self.allowed_cipher_suites.contains(&id) {
            return false;
        }

        if let Some(suite) = registry.lookup_by_id(id) {
            if suite.security_level() < self.min_security_level {
                return false;
            }
            if self.require_forward_secrecy && !suite.has_forward_secrecy() {
                return false;
            }
            if self.fips_mode && !suite.is_fips_approved() {
                return false;
            }
            true
        } else {
            // Unknown suite: reject if we have an explicit allow list.
            self.allowed_cipher_suites.is_empty()
        }
    }

    /// Filter a list of cipher suite IDs according to this policy.
    pub fn filter_cipher_suites(
        &self,
        suites: &[u16],
        registry: &CipherSuiteRegistry,
    ) -> Vec<u16> {
        suites
            .iter()
            .copied()
            .filter(|id| self.is_cipher_suite_acceptable(*id, registry))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Negotiation result
// ---------------------------------------------------------------------------

/// The outcome of a full TLS negotiation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NegotiationResult {
    pub success: bool,
    pub version: Option<TlsVersion>,
    pub cipher_suite: Option<u16>,
    pub cipher_suite_name: Option<String>,
    pub compression: CompressionMethod,
    pub server_extensions: Vec<TlsExtension>,
    pub warnings: Vec<String>,
    pub error: Option<String>,
}

impl NegotiationResult {
    pub fn success(
        version: TlsVersion,
        cipher_suite: u16,
        cipher_name: String,
        extensions: Vec<TlsExtension>,
    ) -> Self {
        Self {
            success: true,
            version: Some(version),
            cipher_suite: Some(cipher_suite),
            cipher_suite_name: Some(cipher_name),
            compression: CompressionMethod::Null,
            server_extensions: extensions,
            warnings: Vec::new(),
            error: None,
        }
    }

    pub fn failure(error: impl Into<String>) -> Self {
        Self {
            success: false,
            version: None,
            cipher_suite: None,
            cipher_suite_name: None,
            compression: CompressionMethod::Null,
            server_extensions: Vec::new(),
            warnings: Vec::new(),
            error: Some(error.into()),
        }
    }

    pub fn with_warning(mut self, warning: impl Into<String>) -> Self {
        self.warnings.push(warning.into());
        self
    }
}

impl fmt::Display for NegotiationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.success {
            write!(
                f,
                "Negotiated: version={}, cipher={}",
                self.version.map_or("none".to_string(), |v| v.to_string()),
                self.cipher_suite_name.as_deref().unwrap_or("none")
            )
        } else {
            write!(
                f,
                "Failed: {}",
                self.error.as_deref().unwrap_or("unknown")
            )
        }
    }
}

// ---------------------------------------------------------------------------
// Negotiation engine
// ---------------------------------------------------------------------------

/// Engine that performs complete TLS negotiation from the server side.
pub struct NegotiationEngine {
    registry: CipherSuiteRegistry,
    policy: NegotiationPolicy,
    version_config: VersionNegotiationConfig,
    extension_config: ExtensionNegotiationConfig,
}

impl NegotiationEngine {
    pub fn new(policy: NegotiationPolicy) -> Self {
        let version_config = VersionNegotiationConfig {
            min_version: policy.min_version,
            max_version: policy.max_version,
            allow_fallback: false,
            enforce_downgrade_sentinel: true,
            supported_versions: TlsVersion::all_versions()
                .iter()
                .copied()
                .filter(|v| *v >= policy.min_version && *v <= policy.max_version)
                .collect(),
        };

        let extension_config = ExtensionNegotiationConfig {
            accept_sni: true,
            accept_max_fragment_length: false,
            accept_encrypt_then_mac: !policy.fips_mode,
            accept_extended_master_secret: true,
            accept_session_tickets: true,
            accept_renegotiation_info: policy.allow_renegotiation,
            accept_status_request: false,
            alpn_protocols: None,
            supported_version_response: if policy.max_version >= TlsVersion::TLS1_3 {
                Some(TlsVersion::TLS1_3)
            } else {
                None
            },
        };

        Self {
            registry: CipherSuiteRegistry::new(),
            policy,
            version_config,
            extension_config,
        }
    }

    pub fn with_default_policy() -> Self {
        Self::new(NegotiationPolicy::default())
    }

    pub fn with_strict_policy() -> Self {
        Self::new(NegotiationPolicy::strict())
    }

    pub fn with_fips_policy() -> Self {
        Self::new(NegotiationPolicy::fips())
    }

    /// Set the server's ALPN protocols.
    pub fn set_alpn_protocols(&mut self, protocols: Vec<String>) {
        self.extension_config.alpn_protocols = Some(protocols);
    }

    /// Get the current policy.
    pub fn policy(&self) -> &NegotiationPolicy {
        &self.policy
    }

    /// Get the cipher suite registry.
    pub fn registry(&self) -> &CipherSuiteRegistry {
        &self.registry
    }

    /// Perform server-side negotiation given a ClientHello.
    pub fn negotiate(&self, client_hello: &ClientHello) -> NegotiationResult {
        let mut warnings = Vec::new();

        // Step 1: Version negotiation.
        let client_supported = client_hello.supported_versions();
        let version_negotiator = VersionNegotiator::new(self.version_config.clone());

        let version_result = version_negotiator.negotiate_server_side(
            client_hello.client_version,
            &client_hello.cipher_suites,
            client_supported,
        );

        let negotiated_version = match version_result {
            VersionNegotiationResult::Success(v) => v,
            VersionNegotiationResult::SuccessViaSupportedVersions(v) => v,
            VersionNegotiationResult::NoCommonVersion => {
                return NegotiationResult::failure("No common TLS version");
            }
            VersionNegotiationResult::InappropriateFallback { attempted, server_max } => {
                return NegotiationResult::failure(format!(
                    "Inappropriate fallback: client offered {} but server supports {}",
                    attempted, server_max
                ));
            }
            VersionNegotiationResult::DowngradeSentinelDetected { negotiated, sentinel_for } => {
                warnings.push(format!(
                    "Downgrade sentinel detected for {} at version {}",
                    sentinel_for, negotiated
                ));
                negotiated
            }
        };

        // Step 2: Cipher suite selection.
        let acceptable_client_suites =
            self.policy.filter_cipher_suites(&client_hello.cipher_suites, &self.registry);

        if acceptable_client_suites.is_empty() {
            return NegotiationResult::failure(
                "No acceptable cipher suites after policy filtering",
            );
        }

        // Build server preference list.
        let server_suites = self.build_server_cipher_list(negotiated_version);

        let selected_suite = select_cipher_suite(
            &acceptable_client_suites,
            &server_suites,
            self.policy.server_preference,
        );

        let (cipher_id, cipher_name) = match selected_suite {
            Some(id) => {
                let name = self
                    .registry
                    .lookup_by_id(id)
                    .map(|s| s.name.to_string())
                    .unwrap_or_else(|| format!("0x{:04X}", id));
                (id, name)
            }
            None => {
                return NegotiationResult::failure("No common cipher suite");
            }
        };

        // Step 3: Compression method (always null in modern TLS).
        let compression = if self.policy.allow_compression
            && client_hello.compression_methods.contains(&CompressionMethod::Deflate)
        {
            warnings.push("DEFLATE compression enabled (CRIME risk)".to_string());
            CompressionMethod::Deflate
        } else {
            CompressionMethod::Null
        };

        // Step 4: Extension negotiation.
        let mut ext_config = self.extension_config.clone();
        if negotiated_version >= TlsVersion::TLS1_3 {
            ext_config.supported_version_response = Some(negotiated_version);
        }
        let server_extensions = negotiate_extensions(&client_hello.extensions, &ext_config);

        // Step 5: Policy checks.
        if self.policy.require_extended_master_secret {
            let has_ems = client_hello
                .extensions
                .iter()
                .any(|e| matches!(e, TlsExtension::ExtendedMasterSecret));
            if !has_ems && negotiated_version < TlsVersion::TLS1_3 {
                return NegotiationResult::failure(
                    "Client does not support Extended Master Secret (required by policy)",
                );
            }
        }

        // Check SCSV.
        if client_hello.has_fallback_scsv()
            && negotiated_version < self.version_config.max_version
        {
            warnings.push("TLS_FALLBACK_SCSV detected but version was downgraded".to_string());
        }

        let mut result = NegotiationResult::success(
            negotiated_version,
            cipher_id,
            cipher_name,
            server_extensions,
        );
        result.compression = compression;
        for w in warnings {
            result = result.with_warning(w);
        }

        result
    }

    /// Build the server's ordered cipher suite list for a given version.
    fn build_server_cipher_list(&self, version: TlsVersion) -> Vec<u16> {
        let mut suites: Vec<&TlsCipherSuite> = self
            .registry
            .all_suites()
            .into_iter()
            .filter(|s| {
                self.policy.is_cipher_suite_acceptable(s.id, &self.registry)
            })
            .filter(|s| {
                if version >= TlsVersion::TLS1_3 {
                    s.is_tls13()
                } else {
                    !s.is_tls13()
                }
            })
            .collect();

        // Sort by security score descending.
        suites.sort_by(|a, b| b.security_score().cmp(&a.security_score()));

        suites.iter().map(|s| s.id).collect()
    }

    /// Simulate the complete negotiation and build a ServerHello.
    pub fn build_server_hello(
        &self,
        client_hello: &ClientHello,
        server_random: [u8; 32],
    ) -> Result<ServerHello, String> {
        let result = self.negotiate(client_hello);
        if !result.success {
            return Err(result.error.unwrap_or_else(|| "negotiation failed".to_string()));
        }

        let version = result.version.unwrap();
        let cipher = result.cipher_suite.unwrap();

        let mut random = server_random;
        // Embed downgrade sentinel if needed.
        if self.version_config.max_version >= TlsVersion::TLS1_3
            && version < TlsVersion::TLS1_3
        {
            VersionNegotiator::embed_downgrade_sentinel(
                &mut random,
                version,
                self.version_config.max_version,
            );
        }

        let record_version = if version >= TlsVersion::TLS1_3 {
            TlsVersion::TLS1_2
        } else {
            version
        };

        let mut sh = ServerHello::new(record_version, random, cipher);
        sh.session_id = client_hello.session_id.clone();
        sh.compression_method = result.compression;
        sh.extensions = result.server_extensions;

        Ok(sh)
    }

    /// Validate a negotiation result against the policy.
    pub fn validate_result(&self, result: &NegotiationResult) -> Vec<String> {
        let mut violations = Vec::new();

        if let Some(version) = result.version {
            if version < self.policy.min_version {
                violations.push(format!(
                    "Version {} below minimum {}",
                    version, self.policy.min_version
                ));
            }
        }

        if let Some(cipher_id) = result.cipher_suite {
            if !self.policy.is_cipher_suite_acceptable(cipher_id, &self.registry) {
                violations.push(format!(
                    "Cipher suite 0x{:04X} not acceptable under policy",
                    cipher_id
                ));
            }
        }

        violations
    }
}

impl fmt::Debug for NegotiationEngine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NegotiationEngine")
            .field("policy_min_version", &self.policy.min_version)
            .field("policy_max_version", &self.policy.max_version)
            .field("fips_mode", &self.policy.fips_mode)
            .field("server_preference", &self.policy.server_preference)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Client-side preference builder
// ---------------------------------------------------------------------------

/// Builds a client cipher suite preference list.
pub struct ClientPreferenceBuilder {
    registry: CipherSuiteRegistry,
    policy: NegotiationPolicy,
}

impl ClientPreferenceBuilder {
    pub fn new(policy: NegotiationPolicy) -> Self {
        Self {
            registry: CipherSuiteRegistry::new(),
            policy,
        }
    }

    /// Build an ordered list of cipher suite IDs for a ClientHello.
    pub fn build_cipher_list(&self, version: TlsVersion) -> Vec<u16> {
        let mut suites: Vec<&TlsCipherSuite> = self
            .registry
            .all_suites()
            .into_iter()
            .filter(|s| self.policy.is_cipher_suite_acceptable(s.id, &self.registry))
            .filter(|s| {
                if version >= TlsVersion::TLS1_3 {
                    // Include both TLS 1.3 and pre-1.3 suites for compatibility.
                    true
                } else {
                    !s.is_tls13()
                }
            })
            .collect();

        suites.sort_by(|a, b| b.security_score().cmp(&a.security_score()));
        suites.iter().map(|s| s.id).collect()
    }

    /// Build a complete ClientHello with the policy applied.
    pub fn build_client_hello(
        &self,
        version: TlsVersion,
        random: [u8; 32],
        sni: Option<&str>,
    ) -> ClientHello {
        let mut ch = ClientHello::new(
            if version >= TlsVersion::TLS1_3 {
                TlsVersion::TLS1_2 // Legacy field
            } else {
                version
            },
            random,
        );

        ch.cipher_suites = self.build_cipher_list(version);

        // Add supported_versions extension for TLS 1.3.
        if version >= TlsVersion::TLS1_3 {
            let versions: Vec<TlsVersion> = TlsVersion::all_versions()
                .iter()
                .copied()
                .filter(|v| *v >= self.policy.min_version && *v <= self.policy.max_version)
                .rev()
                .collect();
            ch.extensions.push(TlsExtension::SupportedVersions(versions));
        }

        // SNI.
        if let Some(hostname) = sni {
            ch.extensions
                .push(TlsExtension::ServerName(vec![hostname.to_string()]));
        }

        // Supported groups.
        let groups = if self.policy.allowed_groups.is_empty() {
            vec![NamedGroup::X25519, NamedGroup::Secp256r1, NamedGroup::Secp384r1]
        } else {
            self.policy.allowed_groups.clone()
        };
        ch.extensions.push(TlsExtension::SupportedGroups(groups));

        // Signature algorithms.
        let sig_algs = if self.policy.allowed_signatures.is_empty() {
            vec![
                SignatureScheme::EcdsaSecp256r1Sha256,
                SignatureScheme::RsaPssRsaeSha256,
                SignatureScheme::RsaPkcs1Sha256,
                SignatureScheme::EcdsaSecp384r1Sha384,
                SignatureScheme::RsaPssRsaeSha384,
                SignatureScheme::RsaPkcs1Sha384,
            ]
        } else {
            self.policy.allowed_signatures.clone()
        };
        ch.extensions.push(TlsExtension::SignatureAlgorithms(sig_algs));

        // Extended master secret (pre-1.3).
        if version < TlsVersion::TLS1_3 {
            ch.extensions.push(TlsExtension::ExtendedMasterSecret);
            ch.extensions.push(TlsExtension::EncryptThenMac);
            ch.extensions.push(TlsExtension::RenegotiationInfo(vec![]));
        }

        ch
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_negotiation() {
        let engine = NegotiationEngine::with_default_policy();
        let builder = ClientPreferenceBuilder::new(NegotiationPolicy::default());
        let ch = builder.build_client_hello(TlsVersion::TLS1_3, [0x42u8; 32], Some("example.com"));

        let result = engine.negotiate(&ch);
        assert!(result.success, "Negotiation failed: {:?}", result.error);
        assert!(result.version.is_some());
        assert!(result.cipher_suite.is_some());
    }

    #[test]
    fn test_strict_policy_rejects_weak() {
        let engine = NegotiationEngine::with_strict_policy();
        let mut ch = ClientHello::new(TlsVersion::TLS1_2, [0u8; 32]);
        ch.cipher_suites = vec![0x000A]; // 3DES only
        ch.extensions.push(TlsExtension::ExtendedMasterSecret);

        let result = engine.negotiate(&ch);
        assert!(!result.success);
    }

    #[test]
    fn test_fips_policy() {
        let engine = NegotiationEngine::with_fips_policy();
        let mut ch = ClientHello::new(TlsVersion::TLS1_2, [0u8; 32]);
        ch.cipher_suites = vec![0x009C, 0x009E, 0xC02F]; // AES-GCM suites
        ch.extensions.push(TlsExtension::ExtendedMasterSecret);

        let result = engine.negotiate(&ch);
        assert!(result.success, "FIPS negotiation failed: {:?}", result.error);
    }

    #[test]
    fn test_no_common_cipher() {
        let engine = NegotiationEngine::with_strict_policy();
        let mut ch = ClientHello::new(TlsVersion::TLS1_2, [0u8; 32]);
        ch.cipher_suites = vec![0x0004]; // RC4_128_MD5 (weak)
        ch.extensions.push(TlsExtension::ExtendedMasterSecret);

        let result = engine.negotiate(&ch);
        assert!(!result.success);
    }

    #[test]
    fn test_version_mismatch() {
        let policy = NegotiationPolicy {
            min_version: TlsVersion::TLS1_3,
            max_version: TlsVersion::TLS1_3,
            ..Default::default()
        };
        let engine = NegotiationEngine::new(policy);
        let mut ch = ClientHello::new(TlsVersion::TLS1_2, [0u8; 32]);
        ch.cipher_suites = vec![0xC02F];

        let result = engine.negotiate(&ch);
        assert!(!result.success);
    }

    #[test]
    fn test_build_server_hello() {
        let engine = NegotiationEngine::with_default_policy();
        let builder = ClientPreferenceBuilder::new(NegotiationPolicy::default());
        let ch = builder.build_client_hello(TlsVersion::TLS1_2, [0x42u8; 32], None);

        let sh = engine.build_server_hello(&ch, [0x55u8; 32]).unwrap();
        assert!(sh.cipher_suite != 0);
    }

    #[test]
    fn test_policy_filter_cipher_suites() {
        let reg = CipherSuiteRegistry::new();
        let policy = NegotiationPolicy::strict();
        let suites = vec![0x0004, 0x0003, 0xC02F, 0x1301];
        let filtered = policy.filter_cipher_suites(&suites, &reg);
        // Strict policy should reject 0x0004 (RC4) and 0x0003 (export).
        assert!(!filtered.contains(&0x0004));
        assert!(!filtered.contains(&0x0003));
        assert!(filtered.contains(&0xC02F) || filtered.contains(&0x1301));
    }

    #[test]
    fn test_client_preference_builder() {
        let builder = ClientPreferenceBuilder::new(NegotiationPolicy::default());
        let ciphers = builder.build_cipher_list(TlsVersion::TLS1_2);
        assert!(!ciphers.is_empty());
        // Should not contain export or weak ciphers.
        let reg = CipherSuiteRegistry::new();
        for &id in &ciphers {
            if let Some(s) = reg.lookup_by_id(id) {
                assert!(
                    s.security_level() >= SecurityLevel::Legacy,
                    "Suite {} ({}) below Legacy",
                    s.name,
                    s.security_level()
                );
            }
        }
    }

    #[test]
    fn test_client_hello_has_sni() {
        let builder = ClientPreferenceBuilder::new(NegotiationPolicy::default());
        let ch = builder.build_client_hello(TlsVersion::TLS1_3, [0u8; 32], Some("test.example.com"));
        let sni = ch.sni_hostnames();
        assert_eq!(sni, vec!["test.example.com"]);
    }

    #[test]
    fn test_compression_always_null() {
        let engine = NegotiationEngine::with_default_policy();
        let mut ch = ClientHello::new(TlsVersion::TLS1_2, [0u8; 32]);
        ch.cipher_suites = vec![0xC02F, 0x009C];
        ch.compression_methods = vec![CompressionMethod::Deflate, CompressionMethod::Null];

        let result = engine.negotiate(&ch);
        if result.success {
            assert_eq!(result.compression, CompressionMethod::Null);
        }
    }

    #[test]
    fn test_validate_result() {
        let engine = NegotiationEngine::with_strict_policy();
        let good = NegotiationResult::success(
            TlsVersion::TLS1_3,
            0x1301,
            "TLS_AES_128_GCM_SHA256".to_string(),
            vec![],
        );
        let violations = engine.validate_result(&good);
        assert!(violations.is_empty());
    }

    #[test]
    fn test_negotiation_result_display() {
        let result = NegotiationResult::success(
            TlsVersion::TLS1_3,
            0x1301,
            "TLS_AES_128_GCM_SHA256".to_string(),
            vec![],
        );
        let display = format!("{}", result);
        assert!(display.contains("TLS 1.3"));
        assert!(display.contains("TLS_AES_128_GCM_SHA256"));
    }

    #[test]
    fn test_permissive_policy_allows_weak() {
        let reg = CipherSuiteRegistry::new();
        let policy = NegotiationPolicy::permissive();
        assert!(policy.is_cipher_suite_acceptable(0x0004, &reg)); // RC4
        assert!(policy.is_cipher_suite_acceptable(0x0003, &reg)); // Export
    }

    #[test]
    fn test_blocked_cipher_suites() {
        let reg = CipherSuiteRegistry::new();
        let policy = NegotiationPolicy {
            blocked_cipher_suites: vec![0xC02F],
            ..NegotiationPolicy::permissive()
        };
        assert!(!policy.is_cipher_suite_acceptable(0xC02F, &reg));
        assert!(policy.is_cipher_suite_acceptable(0x1301, &reg));
    }
}
