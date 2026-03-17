//! Known TLS/SSL vulnerability patterns for downgrade attack synthesis.
//!
//! Models FREAK, Logjam, POODLE, Terrapin, DROWN, CCS Injection, and
//! SSLv2 override vulnerabilities with pattern matching for detection
//! and attack trace templates for each CVE.

use crate::cipher_suites::{
    BulkEncryption, CipherSuiteRegistry, KeyExchange, SecurityLevel, TlsCipherSuite,
};
use crate::extensions::{self, TlsExtension};
use crate::handshake::{ClientHello, HandshakeMessage, ServerHello};
use crate::record::{ContentType, TlsRecord};
use crate::version::TlsVersion;
use serde::{Deserialize, Serialize};
use std::fmt;

// ---------------------------------------------------------------------------
// CVE identifiers
// ---------------------------------------------------------------------------

/// Known TLS/SSL vulnerabilities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KnownVulnerability {
    /// FREAK: RSA_EXPORT cipher injection (CVE-2015-0204).
    Freak,
    /// Logjam: DHE_EXPORT parameter downgrade (CVE-2015-4000).
    Logjam,
    /// POODLE: Version downgrade to SSL 3.0 with CBC padding oracle (CVE-2014-3566).
    Poodle,
    /// Terrapin: SSH extension stripping via sequence number manipulation (CVE-2023-48795).
    Terrapin,
    /// DROWN: SSLv2 cross-protocol attack (CVE-2016-0703).
    Drown,
    /// CCS Injection: ChangeCipherSpec ordering attack (CVE-2014-0224).
    CcsInjection,
    /// SSLv2 override: disabled cipher suite bypass (CVE-2015-3197).
    Ssl2Override,
}

impl KnownVulnerability {
    /// The CVE identifier for this vulnerability.
    pub fn cve_id(&self) -> &'static str {
        match self {
            Self::Freak => "CVE-2015-0204",
            Self::Logjam => "CVE-2015-4000",
            Self::Poodle => "CVE-2014-3566",
            Self::Terrapin => "CVE-2023-48795",
            Self::Drown => "CVE-2016-0703",
            Self::CcsInjection => "CVE-2014-0224",
            Self::Ssl2Override => "CVE-2015-3197",
        }
    }

    /// Human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Freak => "FREAK",
            Self::Logjam => "Logjam",
            Self::Poodle => "POODLE",
            Self::Terrapin => "Terrapin",
            Self::Drown => "DROWN",
            Self::CcsInjection => "CCS Injection",
            Self::Ssl2Override => "SSLv2 Override",
        }
    }

    /// Severity score (CVSS-like, 0.0 - 10.0).
    pub fn severity(&self) -> f64 {
        match self {
            Self::Freak => 7.4,
            Self::Logjam => 6.8,
            Self::Poodle => 7.5,
            Self::Terrapin => 5.9,
            Self::Drown => 7.4,
            Self::CcsInjection => 7.4,
            Self::Ssl2Override => 5.9,
        }
    }

    /// Whether this is a protocol downgrade attack.
    pub fn is_downgrade_attack(&self) -> bool {
        matches!(self, Self::Freak | Self::Logjam | Self::Poodle | Self::Drown)
    }

    /// Which TLS versions are affected.
    pub fn affected_versions(&self) -> Vec<TlsVersion> {
        match self {
            Self::Freak => vec![TlsVersion::SSL3_0, TlsVersion::TLS1_0, TlsVersion::TLS1_1, TlsVersion::TLS1_2],
            Self::Logjam => vec![TlsVersion::SSL3_0, TlsVersion::TLS1_0, TlsVersion::TLS1_1, TlsVersion::TLS1_2],
            Self::Poodle => vec![TlsVersion::SSL3_0],
            Self::Terrapin => vec![], // SSH, not TLS
            Self::Drown => vec![TlsVersion::SSL3_0], // SSLv2 technically
            Self::CcsInjection => vec![TlsVersion::SSL3_0, TlsVersion::TLS1_0, TlsVersion::TLS1_1, TlsVersion::TLS1_2],
            Self::Ssl2Override => vec![TlsVersion::SSL3_0],
        }
    }

    /// All known vulnerabilities.
    pub fn all() -> &'static [KnownVulnerability] {
        &[
            Self::Freak,
            Self::Logjam,
            Self::Poodle,
            Self::Terrapin,
            Self::Drown,
            Self::CcsInjection,
            Self::Ssl2Override,
        ]
    }

    /// Description of the attack mechanism.
    pub fn description(&self) -> &'static str {
        match self {
            Self::Freak => "MITM injects RSA_EXPORT cipher suites into ClientHello, causing server to use 512-bit RSA export keys that can be factored in real time.",
            Self::Logjam => "MITM downgrades DHE key exchange to use export-grade 512-bit DH parameters. Precomputation allows breaking 512-bit DH groups.",
            Self::Poodle => "MITM forces protocol downgrade to SSL 3.0 and exploits CBC padding oracle to decrypt data one byte at a time.",
            Self::Terrapin => "Attacker manipulates SSH sequence numbers by injecting/deleting messages before encryption starts, stripping security extensions.",
            Self::Drown => "SSLv2 server shares RSA key with TLS server. Bleichenbacher-style attack on SSLv2 recovers TLS session keys.",
            Self::CcsInjection => "Injecting ChangeCipherSpec before key material is established causes both sides to derive weak keys from empty master secret.",
            Self::Ssl2Override => "Even with SSLv2 disabled, certain OpenSSL versions still accept SSLv2 handshakes if matching ciphers exist.",
        }
    }
}

impl fmt::Display for KnownVulnerability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.name(), self.cve_id())
    }
}

// ---------------------------------------------------------------------------
// Vulnerability detection result
// ---------------------------------------------------------------------------

/// Result of a vulnerability scan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VulnerabilityDetection {
    pub vulnerability: KnownVulnerability,
    pub confidence: f64,
    pub evidence: Vec<String>,
    pub affected_cipher_suites: Vec<u16>,
    pub affected_version: Option<TlsVersion>,
    pub recommended_mitigation: String,
}

impl VulnerabilityDetection {
    pub fn new(vuln: KnownVulnerability, confidence: f64) -> Self {
        Self {
            vulnerability: vuln,
            confidence,
            evidence: Vec::new(),
            affected_cipher_suites: Vec::new(),
            affected_version: None,
            recommended_mitigation: default_mitigation(vuln),
        }
    }

    pub fn with_evidence(mut self, evidence: impl Into<String>) -> Self {
        self.evidence.push(evidence.into());
        self
    }

    pub fn with_cipher_suites(mut self, suites: Vec<u16>) -> Self {
        self.affected_cipher_suites = suites;
        self
    }

    pub fn with_version(mut self, version: TlsVersion) -> Self {
        self.affected_version = Some(version);
        self
    }
}

impl fmt::Display for VulnerabilityDetection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: confidence={:.1}%, evidence={}",
            self.vulnerability,
            self.confidence * 100.0,
            self.evidence.len()
        )
    }
}

fn default_mitigation(vuln: KnownVulnerability) -> String {
    match vuln {
        KnownVulnerability::Freak => {
            "Disable RSA_EXPORT cipher suites and remove export-grade key support.".to_string()
        }
        KnownVulnerability::Logjam => {
            "Use 2048-bit or larger DH groups. Disable DHE_EXPORT cipher suites.".to_string()
        }
        KnownVulnerability::Poodle => {
            "Disable SSL 3.0. Use TLS_FALLBACK_SCSV. Prefer TLS 1.2+.".to_string()
        }
        KnownVulnerability::Terrapin => {
            "Apply strict key exchange mode. Update SSH implementations.".to_string()
        }
        KnownVulnerability::Drown => {
            "Disable SSLv2 on all servers sharing the RSA key. Use separate keys.".to_string()
        }
        KnownVulnerability::CcsInjection => {
            "Update OpenSSL to patched version. Reject out-of-order CCS messages.".to_string()
        }
        KnownVulnerability::Ssl2Override => {
            "Update OpenSSL. Ensure SSLv2 is fully disabled at the protocol level.".to_string()
        }
    }
}

// ---------------------------------------------------------------------------
// Vulnerability scanner
// ---------------------------------------------------------------------------

/// Scanner that checks for known vulnerability patterns.
pub struct VulnerabilityScanner {
    registry: CipherSuiteRegistry,
}

impl VulnerabilityScanner {
    pub fn new() -> Self {
        Self {
            registry: CipherSuiteRegistry::new(),
        }
    }

    pub fn with_registry(registry: CipherSuiteRegistry) -> Self {
        Self { registry }
    }

    /// Scan a ClientHello for vulnerability indicators.
    pub fn scan_client_hello(&self, ch: &ClientHello) -> Vec<VulnerabilityDetection> {
        let mut detections = Vec::new();

        self.check_freak_client(ch, &mut detections);
        self.check_logjam_client(ch, &mut detections);
        self.check_poodle_client(ch, &mut detections);
        self.check_drown_client(ch, &mut detections);

        detections
    }

    /// Scan a ServerHello for vulnerability indicators.
    pub fn scan_server_hello(&self, sh: &ServerHello) -> Vec<VulnerabilityDetection> {
        let mut detections = Vec::new();

        self.check_freak_server(sh, &mut detections);
        self.check_logjam_server(sh, &mut detections);
        self.check_poodle_server(sh, &mut detections);

        detections
    }

    /// Scan a full handshake for CCS injection vulnerability.
    pub fn scan_handshake_for_ccs_injection(
        &self,
        messages: &[(ContentType, Vec<u8>)],
    ) -> Vec<VulnerabilityDetection> {
        let mut detections = Vec::new();
        self.check_ccs_injection(messages, &mut detections);
        detections
    }

    /// Scan a list of supported cipher suites for vulnerabilities.
    pub fn scan_cipher_suites(&self, suites: &[u16]) -> Vec<VulnerabilityDetection> {
        let mut detections = Vec::new();

        let export_suites: Vec<u16> = suites
            .iter()
            .copied()
            .filter(|id| {
                self.registry
                    .lookup_by_id(*id)
                    .map_or(false, |s| s.is_export())
            })
            .collect();

        if !export_suites.is_empty() {
            let rsa_exports: Vec<u16> = export_suites
                .iter()
                .copied()
                .filter(|id| {
                    self.registry.lookup_by_id(*id).map_or(false, |s| {
                        matches!(s.key_exchange, KeyExchange::RSA_EXPORT)
                    })
                })
                .collect();

            let dhe_exports: Vec<u16> = export_suites
                .iter()
                .copied()
                .filter(|id| {
                    self.registry.lookup_by_id(*id).map_or(false, |s| {
                        matches!(
                            s.key_exchange,
                            KeyExchange::DHE_RSA_EXPORT | KeyExchange::DHE_DSS_EXPORT
                        )
                    })
                })
                .collect();

            if !rsa_exports.is_empty() {
                detections.push(
                    VulnerabilityDetection::new(KnownVulnerability::Freak, 0.9)
                        .with_evidence("RSA_EXPORT cipher suites present")
                        .with_cipher_suites(rsa_exports),
                );
            }

            if !dhe_exports.is_empty() {
                detections.push(
                    VulnerabilityDetection::new(KnownVulnerability::Logjam, 0.9)
                        .with_evidence("DHE_EXPORT cipher suites present")
                        .with_cipher_suites(dhe_exports),
                );
            }
        }

        detections
    }

    // -- FREAK checks --

    fn check_freak_client(&self, ch: &ClientHello, detections: &mut Vec<VulnerabilityDetection>) {
        let export_rsa: Vec<u16> = ch
            .cipher_suites
            .iter()
            .copied()
            .filter(|id| {
                self.registry.lookup_by_id(*id).map_or(false, |s| {
                    matches!(s.key_exchange, KeyExchange::RSA_EXPORT)
                })
            })
            .collect();

        if !export_rsa.is_empty() {
            detections.push(
                VulnerabilityDetection::new(KnownVulnerability::Freak, 0.8)
                    .with_evidence(format!(
                        "ClientHello offers {} RSA_EXPORT suites",
                        export_rsa.len()
                    ))
                    .with_cipher_suites(export_rsa),
            );
        }
    }

    fn check_freak_server(&self, sh: &ServerHello, detections: &mut Vec<VulnerabilityDetection>) {
        if let Some(suite) = self.registry.lookup_by_id(sh.cipher_suite) {
            if matches!(suite.key_exchange, KeyExchange::RSA_EXPORT) {
                detections.push(
                    VulnerabilityDetection::new(KnownVulnerability::Freak, 0.95)
                        .with_evidence(format!(
                            "Server selected RSA_EXPORT cipher: {}",
                            suite.name
                        ))
                        .with_cipher_suites(vec![sh.cipher_suite]),
                );
            }
        }
    }

    // -- Logjam checks --

    fn check_logjam_client(&self, ch: &ClientHello, detections: &mut Vec<VulnerabilityDetection>) {
        let export_dhe: Vec<u16> = ch
            .cipher_suites
            .iter()
            .copied()
            .filter(|id| {
                self.registry.lookup_by_id(*id).map_or(false, |s| {
                    matches!(
                        s.key_exchange,
                        KeyExchange::DHE_RSA_EXPORT | KeyExchange::DHE_DSS_EXPORT
                    )
                })
            })
            .collect();

        if !export_dhe.is_empty() {
            detections.push(
                VulnerabilityDetection::new(KnownVulnerability::Logjam, 0.8)
                    .with_evidence(format!(
                        "ClientHello offers {} DHE_EXPORT suites",
                        export_dhe.len()
                    ))
                    .with_cipher_suites(export_dhe),
            );
        }
    }

    fn check_logjam_server(&self, sh: &ServerHello, detections: &mut Vec<VulnerabilityDetection>) {
        if let Some(suite) = self.registry.lookup_by_id(sh.cipher_suite) {
            if matches!(
                suite.key_exchange,
                KeyExchange::DHE_RSA_EXPORT | KeyExchange::DHE_DSS_EXPORT
            ) {
                detections.push(
                    VulnerabilityDetection::new(KnownVulnerability::Logjam, 0.95)
                        .with_evidence(format!(
                            "Server selected DHE_EXPORT cipher: {}",
                            suite.name
                        ))
                        .with_cipher_suites(vec![sh.cipher_suite]),
                );
            }
        }
    }

    // -- POODLE checks --

    fn check_poodle_client(&self, ch: &ClientHello, detections: &mut Vec<VulnerabilityDetection>) {
        if ch.client_version == TlsVersion::SSL3_0 || ch.client_version < TlsVersion::TLS1_0 {
            let has_cbc: bool = ch.cipher_suites.iter().any(|id| {
                self.registry.lookup_by_id(*id).map_or(false, |s| {
                    matches!(
                        s.encryption,
                        BulkEncryption::AES_128_CBC
                            | BulkEncryption::AES_256_CBC
                            | BulkEncryption::DES_CBC
                            | BulkEncryption::DES_EDE3_CBC
                    )
                })
            });

            if has_cbc {
                let cbc_suites: Vec<u16> = ch
                    .cipher_suites
                    .iter()
                    .copied()
                    .filter(|id| {
                        self.registry.lookup_by_id(*id).map_or(false, |s| {
                            matches!(
                                s.encryption,
                                BulkEncryption::AES_128_CBC
                                    | BulkEncryption::AES_256_CBC
                                    | BulkEncryption::DES_CBC
                                    | BulkEncryption::DES_EDE3_CBC
                            )
                        })
                    })
                    .collect();

                detections.push(
                    VulnerabilityDetection::new(KnownVulnerability::Poodle, 0.85)
                        .with_evidence("SSL 3.0 ClientHello with CBC cipher suites")
                        .with_version(TlsVersion::SSL3_0)
                        .with_cipher_suites(cbc_suites),
                );
            }
        }

        // Also check: no TLS_FALLBACK_SCSV when version < max supported.
        if !ch.has_fallback_scsv() && ch.client_version < TlsVersion::TLS1_2 {
            detections.push(
                VulnerabilityDetection::new(KnownVulnerability::Poodle, 0.4)
                    .with_evidence("Missing TLS_FALLBACK_SCSV in downgraded ClientHello"),
            );
        }
    }

    fn check_poodle_server(&self, sh: &ServerHello, detections: &mut Vec<VulnerabilityDetection>) {
        if sh.server_version == TlsVersion::SSL3_0 || sh.negotiated_version() == TlsVersion::SSL3_0 {
            if let Some(suite) = self.registry.lookup_by_id(sh.cipher_suite) {
                if matches!(
                    suite.encryption,
                    BulkEncryption::AES_128_CBC
                        | BulkEncryption::AES_256_CBC
                        | BulkEncryption::DES_CBC
                        | BulkEncryption::DES_EDE3_CBC
                ) {
                    detections.push(
                        VulnerabilityDetection::new(KnownVulnerability::Poodle, 0.95)
                            .with_evidence(format!(
                                "SSL 3.0 ServerHello with CBC cipher: {}",
                                suite.name
                            ))
                            .with_version(TlsVersion::SSL3_0)
                            .with_cipher_suites(vec![sh.cipher_suite]),
                    );
                }
            }
        }
    }

    // -- DROWN checks --

    fn check_drown_client(&self, ch: &ClientHello, detections: &mut Vec<VulnerabilityDetection>) {
        // DROWN exploits SSLv2 support.
        // If we see very old version or known SSLv2 cipher IDs, flag it.
        if ch.client_version < TlsVersion::SSL3_0 {
            detections.push(
                VulnerabilityDetection::new(KnownVulnerability::Drown, 0.7)
                    .with_evidence("Client version below SSL 3.0 (potential SSLv2)"),
            );
        }
    }

    // -- CCS Injection checks --

    fn check_ccs_injection(
        &self,
        messages: &[(ContentType, Vec<u8>)],
        detections: &mut Vec<VulnerabilityDetection>,
    ) {
        // CCS Injection: ChangeCipherSpec sent before the handshake is complete.
        // Pattern: CCS appears before Certificate or ServerKeyExchange messages.
        let mut saw_ccs = false;
        let mut saw_key_exchange = false;
        let mut premature_ccs = false;

        for (ct, data) in messages {
            match ct {
                ContentType::ChangeCipherSpec => {
                    if !saw_key_exchange {
                        premature_ccs = true;
                    }
                    saw_ccs = true;
                }
                ContentType::Handshake => {
                    if !data.is_empty() {
                        let msg_type = data[0];
                        // ServerKeyExchange(12) or ClientKeyExchange(16).
                        if msg_type == 12 || msg_type == 16 {
                            saw_key_exchange = true;
                        }
                    }
                }
                _ => {}
            }
        }

        if premature_ccs {
            detections.push(
                VulnerabilityDetection::new(KnownVulnerability::CcsInjection, 0.9)
                    .with_evidence("ChangeCipherSpec received before key exchange"),
            );
        }
    }
}

impl Default for VulnerabilityScanner {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Attack trace templates
// ---------------------------------------------------------------------------

/// A step in an attack trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttackStep {
    pub step_number: u32,
    pub actor: AttackActor,
    pub action: String,
    pub message_type: Option<String>,
    pub description: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttackActor {
    Client,
    Server,
    Attacker,
}

impl fmt::Display for AttackActor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Client => write!(f, "Client"),
            Self::Server => write!(f, "Server"),
            Self::Attacker => write!(f, "Attacker"),
        }
    }
}

/// An attack trace template for a known vulnerability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttackTrace {
    pub vulnerability: KnownVulnerability,
    pub steps: Vec<AttackStep>,
    pub preconditions: Vec<String>,
    pub postconditions: Vec<String>,
}

impl AttackTrace {
    pub fn new(vuln: KnownVulnerability) -> Self {
        Self {
            vulnerability: vuln,
            steps: Vec::new(),
            preconditions: Vec::new(),
            postconditions: Vec::new(),
        }
    }

    pub fn step_count(&self) -> usize {
        self.steps.len()
    }
}

/// Generate the attack trace template for FREAK.
pub fn freak_attack_trace() -> AttackTrace {
    AttackTrace {
        vulnerability: KnownVulnerability::Freak,
        steps: vec![
            AttackStep {
                step_number: 1,
                actor: AttackActor::Client,
                action: "send_client_hello".to_string(),
                message_type: Some("ClientHello".to_string()),
                description: "Client sends ClientHello with strong cipher suites".to_string(),
            },
            AttackStep {
                step_number: 2,
                actor: AttackActor::Attacker,
                action: "modify_client_hello".to_string(),
                message_type: Some("ClientHello".to_string()),
                description: "MITM replaces cipher suites with RSA_EXPORT ciphers".to_string(),
            },
            AttackStep {
                step_number: 3,
                actor: AttackActor::Server,
                action: "select_export_cipher".to_string(),
                message_type: Some("ServerHello".to_string()),
                description: "Server selects RSA_EXPORT cipher, sends 512-bit RSA key".to_string(),
            },
            AttackStep {
                step_number: 4,
                actor: AttackActor::Attacker,
                action: "factor_rsa_key".to_string(),
                message_type: None,
                description: "Attacker factors 512-bit RSA key in real time".to_string(),
            },
            AttackStep {
                step_number: 5,
                actor: AttackActor::Attacker,
                action: "forge_server_key_exchange".to_string(),
                message_type: Some("ServerKeyExchange".to_string()),
                description: "Attacker forges ServerKeyExchange with strong cipher using factored key".to_string(),
            },
            AttackStep {
                step_number: 6,
                actor: AttackActor::Attacker,
                action: "decrypt_traffic".to_string(),
                message_type: None,
                description: "Attacker decrypts session using recovered key material".to_string(),
            },
        ],
        preconditions: vec![
            "Server supports RSA_EXPORT cipher suites".to_string(),
            "MITM position between client and server".to_string(),
        ],
        postconditions: vec![
            "Attacker can decrypt session traffic".to_string(),
            "Client believes it has a strong connection".to_string(),
        ],
    }
}

/// Generate the attack trace template for Logjam.
pub fn logjam_attack_trace() -> AttackTrace {
    AttackTrace {
        vulnerability: KnownVulnerability::Logjam,
        steps: vec![
            AttackStep {
                step_number: 1,
                actor: AttackActor::Client,
                action: "send_client_hello".to_string(),
                message_type: Some("ClientHello".to_string()),
                description: "Client sends ClientHello with DHE cipher suites".to_string(),
            },
            AttackStep {
                step_number: 2,
                actor: AttackActor::Attacker,
                action: "modify_client_hello".to_string(),
                message_type: Some("ClientHello".to_string()),
                description: "MITM replaces DHE suites with DHE_EXPORT suites".to_string(),
            },
            AttackStep {
                step_number: 3,
                actor: AttackActor::Server,
                action: "select_export_dhe".to_string(),
                message_type: Some("ServerHello".to_string()),
                description: "Server selects DHE_EXPORT, sends 512-bit DH params".to_string(),
            },
            AttackStep {
                step_number: 4,
                actor: AttackActor::Attacker,
                action: "compute_discrete_log".to_string(),
                message_type: None,
                description: "Attacker uses precomputed tables to solve 512-bit DLP".to_string(),
            },
            AttackStep {
                step_number: 5,
                actor: AttackActor::Attacker,
                action: "complete_handshake".to_string(),
                message_type: None,
                description: "Attacker completes handshake on both sides".to_string(),
            },
        ],
        preconditions: vec![
            "Server supports DHE_EXPORT cipher suites".to_string(),
            "Attacker has precomputed NFS tables for common 512-bit DH groups".to_string(),
            "MITM position".to_string(),
        ],
        postconditions: vec![
            "Attacker can read and modify traffic".to_string(),
        ],
    }
}

/// Generate the attack trace template for POODLE.
pub fn poodle_attack_trace() -> AttackTrace {
    AttackTrace {
        vulnerability: KnownVulnerability::Poodle,
        steps: vec![
            AttackStep {
                step_number: 1,
                actor: AttackActor::Attacker,
                action: "force_version_downgrade".to_string(),
                message_type: None,
                description: "MITM causes TLS handshake failures to trigger version fallback".to_string(),
            },
            AttackStep {
                step_number: 2,
                actor: AttackActor::Client,
                action: "retry_ssl30".to_string(),
                message_type: Some("ClientHello".to_string()),
                description: "Client retries with SSL 3.0 after TLS failures".to_string(),
            },
            AttackStep {
                step_number: 3,
                actor: AttackActor::Server,
                action: "accept_ssl30".to_string(),
                message_type: Some("ServerHello".to_string()),
                description: "Server accepts SSL 3.0 with CBC cipher".to_string(),
            },
            AttackStep {
                step_number: 4,
                actor: AttackActor::Attacker,
                action: "cbc_padding_oracle".to_string(),
                message_type: None,
                description: "Attacker exploits SSL 3.0 CBC padding oracle to decrypt bytes".to_string(),
            },
            AttackStep {
                step_number: 5,
                actor: AttackActor::Attacker,
                action: "extract_secrets".to_string(),
                message_type: None,
                description: "Attacker recovers session cookies/tokens one byte at a time".to_string(),
            },
        ],
        preconditions: vec![
            "Server supports SSL 3.0".to_string(),
            "Client performs version fallback without TLS_FALLBACK_SCSV".to_string(),
            "CBC cipher suite negotiated".to_string(),
        ],
        postconditions: vec![
            "Attacker recovers plaintext data (e.g., cookies)".to_string(),
        ],
    }
}

/// Generate the attack trace template for CCS Injection.
pub fn ccs_injection_attack_trace() -> AttackTrace {
    AttackTrace {
        vulnerability: KnownVulnerability::CcsInjection,
        steps: vec![
            AttackStep {
                step_number: 1,
                actor: AttackActor::Client,
                action: "send_client_hello".to_string(),
                message_type: Some("ClientHello".to_string()),
                description: "Normal TLS handshake begins".to_string(),
            },
            AttackStep {
                step_number: 2,
                actor: AttackActor::Server,
                action: "send_server_hello".to_string(),
                message_type: Some("ServerHello".to_string()),
                description: "Server responds with ServerHello, Certificate, etc.".to_string(),
            },
            AttackStep {
                step_number: 3,
                actor: AttackActor::Attacker,
                action: "inject_ccs".to_string(),
                message_type: Some("ChangeCipherSpec".to_string()),
                description: "MITM injects premature ChangeCipherSpec before key exchange completes".to_string(),
            },
            AttackStep {
                step_number: 4,
                actor: AttackActor::Server,
                action: "accept_premature_ccs".to_string(),
                message_type: None,
                description: "Vulnerable server accepts CCS and derives keys from empty master secret".to_string(),
            },
            AttackStep {
                step_number: 5,
                actor: AttackActor::Attacker,
                action: "decrypt_with_weak_keys".to_string(),
                message_type: None,
                description: "Attacker knows the key material (derived from empty/predictable state)".to_string(),
            },
        ],
        preconditions: vec![
            "Server runs vulnerable OpenSSL (before 1.0.1h, 1.0.0m, 0.9.8za)".to_string(),
            "MITM position".to_string(),
        ],
        postconditions: vec![
            "Attacker can decrypt and modify traffic".to_string(),
        ],
    }
}

/// Generate the attack trace template for DROWN.
pub fn drown_attack_trace() -> AttackTrace {
    AttackTrace {
        vulnerability: KnownVulnerability::Drown,
        steps: vec![
            AttackStep {
                step_number: 1,
                actor: AttackActor::Attacker,
                action: "capture_tls_session".to_string(),
                message_type: None,
                description: "Attacker captures TLS session using RSA key exchange".to_string(),
            },
            AttackStep {
                step_number: 2,
                actor: AttackActor::Attacker,
                action: "connect_sslv2_server".to_string(),
                message_type: None,
                description: "Attacker connects to SSLv2-enabled server sharing the same RSA key".to_string(),
            },
            AttackStep {
                step_number: 3,
                actor: AttackActor::Attacker,
                action: "bleichenbacher_oracle".to_string(),
                message_type: None,
                description: "Attacker uses SSLv2 Bleichenbacher oracle to decrypt RSA ciphertext".to_string(),
            },
            AttackStep {
                step_number: 4,
                actor: AttackActor::Attacker,
                action: "recover_premaster_secret".to_string(),
                message_type: None,
                description: "Attacker recovers TLS premaster secret via cross-protocol attack".to_string(),
            },
            AttackStep {
                step_number: 5,
                actor: AttackActor::Attacker,
                action: "decrypt_captured_session".to_string(),
                message_type: None,
                description: "Attacker decrypts captured TLS session".to_string(),
            },
        ],
        preconditions: vec![
            "SSLv2-enabled server shares RSA key with TLS server".to_string(),
            "TLS session uses RSA key exchange (no forward secrecy)".to_string(),
        ],
        postconditions: vec![
            "Attacker decrypts captured TLS session".to_string(),
        ],
    }
}

/// Generate the attack trace for Terrapin (SSH, included for reference).
pub fn terrapin_attack_trace() -> AttackTrace {
    AttackTrace {
        vulnerability: KnownVulnerability::Terrapin,
        steps: vec![
            AttackStep {
                step_number: 1,
                actor: AttackActor::Attacker,
                action: "intercept_handshake".to_string(),
                message_type: None,
                description: "MITM intercepts SSH handshake before encryption activates".to_string(),
            },
            AttackStep {
                step_number: 2,
                actor: AttackActor::Attacker,
                action: "delete_ext_info".to_string(),
                message_type: None,
                description: "Attacker deletes SSH_MSG_EXT_INFO message to strip security extensions".to_string(),
            },
            AttackStep {
                step_number: 3,
                actor: AttackActor::Attacker,
                action: "adjust_sequence_numbers".to_string(),
                message_type: None,
                description: "Attacker injects IGNORE message to realign sequence numbers".to_string(),
            },
            AttackStep {
                step_number: 4,
                actor: AttackActor::Client,
                action: "continue_without_extensions".to_string(),
                message_type: None,
                description: "Client/server continue without security extensions (e.g., no server-sig-algs)".to_string(),
            },
        ],
        preconditions: vec![
            "SSH uses ChaCha20-Poly1305 or CBC-EtM cipher".to_string(),
            "MITM position during initial handshake".to_string(),
        ],
        postconditions: vec![
            "Security extensions stripped from session".to_string(),
        ],
    }
}

/// Get the attack trace template for any known vulnerability.
pub fn get_attack_trace(vuln: KnownVulnerability) -> AttackTrace {
    match vuln {
        KnownVulnerability::Freak => freak_attack_trace(),
        KnownVulnerability::Logjam => logjam_attack_trace(),
        KnownVulnerability::Poodle => poodle_attack_trace(),
        KnownVulnerability::Terrapin => terrapin_attack_trace(),
        KnownVulnerability::Drown => drown_attack_trace(),
        KnownVulnerability::CcsInjection => ccs_injection_attack_trace(),
        KnownVulnerability::Ssl2Override => {
            let mut trace = AttackTrace::new(KnownVulnerability::Ssl2Override);
            trace.steps.push(AttackStep {
                step_number: 1,
                actor: AttackActor::Attacker,
                action: "send_sslv2_client_hello".to_string(),
                message_type: Some("SSLv2 ClientHello".to_string()),
                description: "Attacker sends SSLv2 ClientHello to server with SSLv2 disabled but vulnerable".to_string(),
            });
            trace.steps.push(AttackStep {
                step_number: 2,
                actor: AttackActor::Server,
                action: "accept_sslv2_despite_config".to_string(),
                message_type: Some("SSLv2 ServerHello".to_string()),
                description: "Bug: server accepts SSLv2 handshake despite configuration".to_string(),
            });
            trace.preconditions.push("OpenSSL with SSLv2 override bug".to_string());
            trace.postconditions.push("SSLv2 session established despite disabled config".to_string());
            trace
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vulnerability_cve_ids() {
        assert_eq!(KnownVulnerability::Freak.cve_id(), "CVE-2015-0204");
        assert_eq!(KnownVulnerability::Poodle.cve_id(), "CVE-2014-3566");
        assert_eq!(KnownVulnerability::Terrapin.cve_id(), "CVE-2023-48795");
    }

    #[test]
    fn test_vulnerability_names() {
        assert_eq!(KnownVulnerability::Logjam.name(), "Logjam");
        assert_eq!(KnownVulnerability::CcsInjection.name(), "CCS Injection");
    }

    #[test]
    fn test_all_vulnerabilities() {
        let all = KnownVulnerability::all();
        assert_eq!(all.len(), 7);
    }

    #[test]
    fn test_is_downgrade_attack() {
        assert!(KnownVulnerability::Freak.is_downgrade_attack());
        assert!(KnownVulnerability::Poodle.is_downgrade_attack());
        assert!(!KnownVulnerability::CcsInjection.is_downgrade_attack());
        assert!(!KnownVulnerability::Terrapin.is_downgrade_attack());
    }

    #[test]
    fn test_scan_client_hello_freak() {
        let scanner = VulnerabilityScanner::new();
        let mut ch = ClientHello::new(TlsVersion::TLS1_2, [0u8; 32]);
        ch.cipher_suites = vec![0xC02F, 0x0003]; // 0x0003 = RSA_EXPORT_WITH_RC4_40_MD5

        let detections = scanner.scan_client_hello(&ch);
        let freak = detections.iter().find(|d| d.vulnerability == KnownVulnerability::Freak);
        assert!(freak.is_some());
    }

    #[test]
    fn test_scan_client_hello_no_vuln() {
        let scanner = VulnerabilityScanner::new();
        let mut ch = ClientHello::new(TlsVersion::TLS1_2, [0u8; 32]);
        ch.cipher_suites = vec![0xC02F, 0x1301];

        let detections = scanner.scan_client_hello(&ch);
        assert!(detections.is_empty());
    }

    #[test]
    fn test_scan_server_hello_freak() {
        let scanner = VulnerabilityScanner::new();
        let sh = ServerHello::new(TlsVersion::TLS1_0, [0u8; 32], 0x0003);

        let detections = scanner.scan_server_hello(&sh);
        let freak = detections.iter().find(|d| d.vulnerability == KnownVulnerability::Freak);
        assert!(freak.is_some());
    }

    #[test]
    fn test_scan_poodle_ssl30_cbc() {
        let scanner = VulnerabilityScanner::new();
        let sh = ServerHello::new(TlsVersion::SSL3_0, [0u8; 32], 0x002F); // AES_128_CBC_SHA
        let detections = scanner.scan_server_hello(&sh);
        let poodle = detections.iter().find(|d| d.vulnerability == KnownVulnerability::Poodle);
        assert!(poodle.is_some());
    }

    #[test]
    fn test_scan_ccs_injection() {
        let scanner = VulnerabilityScanner::new();
        let messages = vec![
            (ContentType::Handshake, vec![1, 0, 0, 0]),    // ClientHello
            (ContentType::Handshake, vec![2, 0, 0, 0]),    // ServerHello
            (ContentType::ChangeCipherSpec, vec![1]),       // CCS before key exchange!
            (ContentType::Handshake, vec![12, 0, 0, 1, 0]), // ServerKeyExchange
        ];
        let detections = scanner.scan_handshake_for_ccs_injection(&messages);
        let ccs = detections.iter().find(|d| d.vulnerability == KnownVulnerability::CcsInjection);
        assert!(ccs.is_some());
    }

    #[test]
    fn test_scan_cipher_suites() {
        let scanner = VulnerabilityScanner::new();
        let suites = vec![0x0003, 0x0013, 0xC02F]; // RSA_EXPORT, DHE_RSA_EXPORT
        let detections = scanner.scan_cipher_suites(&suites);
        assert!(!detections.is_empty());
    }

    #[test]
    fn test_attack_traces() {
        for &vuln in KnownVulnerability::all() {
            let trace = get_attack_trace(vuln);
            assert_eq!(trace.vulnerability, vuln);
            assert!(!trace.steps.is_empty(), "Empty trace for {}", vuln);
        }
    }

    #[test]
    fn test_freak_attack_trace_structure() {
        let trace = freak_attack_trace();
        assert_eq!(trace.vulnerability, KnownVulnerability::Freak);
        assert_eq!(trace.steps.len(), 6);
        assert_eq!(trace.steps[0].actor, AttackActor::Client);
        assert_eq!(trace.steps[1].actor, AttackActor::Attacker);
        assert!(!trace.preconditions.is_empty());
        assert!(!trace.postconditions.is_empty());
    }

    #[test]
    fn test_vulnerability_severity() {
        for &vuln in KnownVulnerability::all() {
            let sev = vuln.severity();
            assert!(sev >= 0.0 && sev <= 10.0, "Bad severity for {}: {}", vuln, sev);
        }
    }

    #[test]
    fn test_vulnerability_display() {
        let display = format!("{}", KnownVulnerability::Freak);
        assert!(display.contains("FREAK"));
        assert!(display.contains("CVE-2015-0204"));
    }

    #[test]
    fn test_detection_display() {
        let det = VulnerabilityDetection::new(KnownVulnerability::Poodle, 0.85)
            .with_evidence("SSL 3.0 negotiated");
        let display = format!("{}", det);
        assert!(display.contains("POODLE"));
        assert!(display.contains("85.0%"));
    }
}
