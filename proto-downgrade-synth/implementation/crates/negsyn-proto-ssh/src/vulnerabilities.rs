//! Known SSH vulnerability patterns and detection heuristics.
//!
//! Each vulnerability has:
//! - A CVE identifier (where applicable)
//! - A description
//! - Detection logic that checks negotiation parameters
//! - An attack trace template for the synthesis engine

use crate::algorithms::{EncryptionAlgorithm, SecurityClassification};
use crate::extensions::StrictKex;
use crate::kex::{KexAlgorithm, KexInit};
use serde::{Deserialize, Serialize};
use std::fmt;

// ---------------------------------------------------------------------------
// SshVulnerability
// ---------------------------------------------------------------------------

/// Known SSH vulnerabilities.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SshVulnerability {
    /// CVE-2023-48795 — Terrapin prefix truncation attack.
    /// Exploits sequence number manipulation when using ChaCha20-Poly1305
    /// or any CBC cipher with ETM MAC.
    Terrapin,

    /// ChaCha20-Poly1305 sequence number wrapping after 2^32 packets
    /// without re-keying.
    Chacha20SequenceWrap,

    /// CBC mode plaintext recovery (Bellare et al., 2004; CVE-2008-5161).
    CbcPlaintextRecovery,

    /// Algorithm downgrade from strong to weak cipher via MITM KEX
    /// manipulation.
    AlgorithmDowngrade {
        category: String,
        strong: String,
        weak: String,
    },

    /// Weak key exchange using DH Group 1 (1024-bit) — subject to
    /// Logjam-class attacks.
    WeakKeyExchange { algorithm: String },

    /// Use of SHA-1 based algorithms (e.g., diffie-hellman-group14-sha1,
    /// ssh-rsa) which are collision-vulnerable.
    Sha1Usage { algorithm: String },

    /// HMAC-MD5 usage — collision attacks trivial.
    Md5Mac,

    /// No encryption (cipher "none") — complete confidentiality loss.
    NoEncryption,

    /// Pre-auth compression (zlib without @openssh.com) can leak
    /// plaintext length information (CRIME-like).
    PreAuthCompression,

    /// Strict-KEX not supported — enables Terrapin even if the cipher
    /// would otherwise mitigate it.
    NoStrictKex,

    /// SSH-RSA host key with SHA-1 signature — collision attacks feasible.
    SshRsaSha1HostKey,

    /// DSA host key (1024-bit, broken by modern standards).
    DsaHostKey,
}

impl SshVulnerability {
    /// Returns the CVE identifier if one exists.
    pub fn cve(&self) -> Option<&'static str> {
        match self {
            Self::Terrapin => Some("CVE-2023-48795"),
            Self::CbcPlaintextRecovery => Some("CVE-2008-5161"),
            _ => None,
        }
    }

    /// Returns a human-readable description.
    pub fn description(&self) -> String {
        match self {
            Self::Terrapin => {
                "Terrapin attack: prefix truncation via sequence number manipulation. \
                 An active MITM can delete the EXT_INFO message sent immediately after \
                 NEWKEYS, potentially disabling countermeasures like server-sig-algs. \
                 Affects ChaCha20-Poly1305 and CBC+ETM ciphers."
                    .into()
            }
            Self::Chacha20SequenceWrap => {
                "ChaCha20-Poly1305 derives per-packet nonces from the 32-bit sequence \
                 number. After 2^32 packets without re-keying, nonces repeat, \
                 enabling key recovery."
                    .into()
            }
            Self::CbcPlaintextRecovery => {
                "CBC mode encryption is vulnerable to plaintext recovery attacks \
                 (Bellare et al., 2004). An attacker can recover 14 bits of plaintext \
                 per block with ~2^14 chosen ciphertexts."
                    .into()
            }
            Self::AlgorithmDowngrade {
                category,
                strong,
                weak,
            } => {
                format!(
                    "Algorithm downgrade: {} can be forced from '{}' to '{}' \
                     by a MITM modifying the KEXINIT name-lists.",
                    category, strong, weak
                )
            }
            Self::WeakKeyExchange { algorithm } => {
                format!(
                    "Weak key exchange '{}' uses insufficient group size. \
                     Susceptible to precomputation attacks (Logjam-class).",
                    algorithm
                )
            }
            Self::Sha1Usage { algorithm } => {
                format!(
                    "Algorithm '{}' uses SHA-1 which is collision-vulnerable. \
                     Chosen-prefix collisions are practical.",
                    algorithm
                )
            }
            Self::Md5Mac => {
                "HMAC-MD5 MAC: MD5 is cryptographically broken. \
                 Collision attacks are trivial."
                    .into()
            }
            Self::NoEncryption => {
                "Cipher 'none' selected — all traffic transmitted in cleartext.".into()
            }
            Self::PreAuthCompression => {
                "Pre-authentication zlib compression enabled. \
                 Vulnerable to CRIME-like side-channel attacks that \
                 leak plaintext content via compressed length."
                    .into()
            }
            Self::NoStrictKex => {
                "Neither client nor server supports strict-KEX. \
                 The connection is vulnerable to Terrapin (CVE-2023-48795) \
                 if a susceptible cipher is negotiated."
                    .into()
            }
            Self::SshRsaSha1HostKey => {
                "ssh-rsa host key uses SHA-1 signatures. \
                 Chosen-prefix collision attacks against SHA-1 are practical, \
                 enabling host key impersonation."
                    .into()
            }
            Self::DsaHostKey => {
                "ssh-dss (DSA) host key uses only 1024-bit keys. \
                 Discrete-log attacks at this size are within reach of \
                 well-resourced adversaries."
                    .into()
            }
        }
    }

    /// Severity score (0 = informational, 10 = critical).
    pub fn severity(&self) -> f64 {
        match self {
            Self::Terrapin => 5.9,
            Self::Chacha20SequenceWrap => 4.0,
            Self::CbcPlaintextRecovery => 4.3,
            Self::AlgorithmDowngrade { .. } => 7.4,
            Self::WeakKeyExchange { .. } => 6.8,
            Self::Sha1Usage { .. } => 5.3,
            Self::Md5Mac => 6.0,
            Self::NoEncryption => 9.8,
            Self::PreAuthCompression => 3.5,
            Self::NoStrictKex => 5.9,
            Self::SshRsaSha1HostKey => 5.3,
            Self::DsaHostKey => 7.0,
        }
    }

    /// Whether this vulnerability allows active exploitation (MITM required).
    pub fn requires_active_attacker(&self) -> bool {
        match self {
            Self::Terrapin => true,
            Self::Chacha20SequenceWrap => false,
            Self::CbcPlaintextRecovery => true,
            Self::AlgorithmDowngrade { .. } => true,
            Self::WeakKeyExchange { .. } => false,
            Self::Sha1Usage { .. } => true,
            Self::Md5Mac => true,
            Self::NoEncryption => false,
            Self::PreAuthCompression => true,
            Self::NoStrictKex => true,
            Self::SshRsaSha1HostKey => true,
            Self::DsaHostKey => false,
        }
    }

    /// Returns a MITRE ATT&CK technique ID if applicable.
    pub fn mitre_technique(&self) -> Option<&'static str> {
        match self {
            Self::Terrapin | Self::AlgorithmDowngrade { .. } => {
                Some("T1557 — Adversary-in-the-Middle")
            }
            Self::NoEncryption => Some("T1040 — Network Sniffing"),
            Self::WeakKeyExchange { .. } | Self::DsaHostKey => {
                Some("T1600 — Weaken Encryption")
            }
            _ => None,
        }
    }
}

impl fmt::Display for SshVulnerability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.cve() {
            Some(cve) => write!(f, "{} ({})", self.short_name(), cve),
            None => write!(f, "{}", self.short_name()),
        }
    }
}

impl SshVulnerability {
    fn short_name(&self) -> &str {
        match self {
            Self::Terrapin => "Terrapin",
            Self::Chacha20SequenceWrap => "ChaCha20-SeqWrap",
            Self::CbcPlaintextRecovery => "CBC-PlaintextRecovery",
            Self::AlgorithmDowngrade { .. } => "AlgorithmDowngrade",
            Self::WeakKeyExchange { .. } => "WeakKeyExchange",
            Self::Sha1Usage { .. } => "SHA1-Usage",
            Self::Md5Mac => "MD5-MAC",
            Self::NoEncryption => "NoEncryption",
            Self::PreAuthCompression => "PreAuthCompression",
            Self::NoStrictKex => "NoStrictKex",
            Self::SshRsaSha1HostKey => "SSH-RSA-SHA1",
            Self::DsaHostKey => "DSA-HostKey",
        }
    }
}

// ---------------------------------------------------------------------------
// Attack trace templates
// ---------------------------------------------------------------------------

/// A step in an attack trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttackStep {
    /// Step number (1-indexed).
    pub step: u32,
    /// Who performs this step.
    pub actor: AttackActor,
    /// Description of what happens.
    pub action: String,
    /// The SSH message type involved (if any).
    pub msg_type: Option<u8>,
    /// Whether this step requires modifying a message in transit.
    pub requires_modification: bool,
}

/// Actors in an attack trace.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
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
            Self::Attacker => write!(f, "Attacker (MITM)"),
        }
    }
}

/// An attack trace for a specific vulnerability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttackTrace {
    pub vulnerability: SshVulnerability,
    pub preconditions: Vec<String>,
    pub steps: Vec<AttackStep>,
    pub postconditions: Vec<String>,
    pub impact: String,
}

impl AttackTrace {
    /// Build the Terrapin attack trace.
    pub fn terrapin() -> Self {
        Self {
            vulnerability: SshVulnerability::Terrapin,
            preconditions: vec![
                "MITM position between client and server".into(),
                "ChaCha20-Poly1305 or CBC+ETM cipher negotiated".into(),
                "Neither side enforces strict-KEX".into(),
            ],
            steps: vec![
                AttackStep {
                    step: 1,
                    actor: AttackActor::Client,
                    action: "Client sends SSH_MSG_KEXINIT".into(),
                    msg_type: Some(crate::constants::SSH_MSG_KEXINIT),
                    requires_modification: false,
                },
                AttackStep {
                    step: 2,
                    actor: AttackActor::Server,
                    action: "Server sends SSH_MSG_KEXINIT".into(),
                    msg_type: Some(crate::constants::SSH_MSG_KEXINIT),
                    requires_modification: false,
                },
                AttackStep {
                    step: 3,
                    actor: AttackActor::Attacker,
                    action: "Attacker records both KEXINITs and allows KEX to proceed normally"
                        .into(),
                    msg_type: None,
                    requires_modification: false,
                },
                AttackStep {
                    step: 4,
                    actor: AttackActor::Attacker,
                    action: "After NEWKEYS, attacker injects SSH_MSG_IGNORE to client, \
                             incrementing the client's receive sequence number by 1"
                        .into(),
                    msg_type: Some(crate::constants::SSH_MSG_IGNORE),
                    requires_modification: false,
                },
                AttackStep {
                    step: 5,
                    actor: AttackActor::Server,
                    action: "Server sends SSH_MSG_EXT_INFO (with server-sig-algs)".into(),
                    msg_type: Some(crate::constants::SSH_MSG_EXT_INFO),
                    requires_modification: false,
                },
                AttackStep {
                    step: 6,
                    actor: AttackActor::Attacker,
                    action: "Attacker drops the SSH_MSG_EXT_INFO. Because the client's \
                             sequence number was incremented by the injected IGNORE, \
                             the sequence numbers remain synchronized despite the \
                             dropped message."
                        .into(),
                    msg_type: Some(crate::constants::SSH_MSG_EXT_INFO),
                    requires_modification: true,
                },
                AttackStep {
                    step: 7,
                    actor: AttackActor::Client,
                    action: "Client proceeds without server-sig-algs, potentially \
                             falling back to weaker signature algorithms (e.g., ssh-rsa \
                             with SHA-1 instead of rsa-sha2-256)."
                        .into(),
                    msg_type: None,
                    requires_modification: false,
                },
            ],
            postconditions: vec![
                "Client may use weak signature algorithms for host key verification".into(),
                "Server's EXT_INFO extensions are silently suppressed".into(),
            ],
            impact: "Extension downgrade; may enable RSA/SHA-1 signature usage \
                     which is vulnerable to chosen-prefix collision attacks."
                .into(),
        }
    }

    /// Build an algorithm downgrade attack trace.
    pub fn algorithm_downgrade(category: &str, strong: &str, weak: &str) -> Self {
        Self {
            vulnerability: SshVulnerability::AlgorithmDowngrade {
                category: category.into(),
                strong: strong.into(),
                weak: weak.into(),
            },
            preconditions: vec![
                "MITM position between client and server".into(),
                format!("Both client and server support '{}'", weak),
                format!("Client prefers '{}' but also lists '{}'", strong, weak),
            ],
            steps: vec![
                AttackStep {
                    step: 1,
                    actor: AttackActor::Client,
                    action: format!(
                        "Client sends KEXINIT with {} list: [{}, ..., {}]",
                        category, strong, weak
                    ),
                    msg_type: Some(crate::constants::SSH_MSG_KEXINIT),
                    requires_modification: false,
                },
                AttackStep {
                    step: 2,
                    actor: AttackActor::Attacker,
                    action: format!(
                        "Attacker modifies client's KEXINIT: removes '{}' from {} list, \
                         leaving only '{}' (and other weak options)",
                        strong, category, weak
                    ),
                    msg_type: Some(crate::constants::SSH_MSG_KEXINIT),
                    requires_modification: true,
                },
                AttackStep {
                    step: 3,
                    actor: AttackActor::Server,
                    action: format!(
                        "Server receives modified KEXINIT and selects '{}' as the \
                         first matching {} algorithm",
                        weak, category
                    ),
                    msg_type: None,
                    requires_modification: false,
                },
                AttackStep {
                    step: 4,
                    actor: AttackActor::Server,
                    action: "Server sends its KEXINIT (attacker forwards unmodified)".into(),
                    msg_type: Some(crate::constants::SSH_MSG_KEXINIT),
                    requires_modification: false,
                },
                AttackStep {
                    step: 5,
                    actor: AttackActor::Attacker,
                    action: "KEX proceeds with the weak algorithm. Attacker may exploit \
                             the weakness depending on the specific algorithm."
                        .into(),
                    msg_type: None,
                    requires_modification: false,
                },
            ],
            postconditions: vec![
                format!("Connection uses '{}' instead of '{}'", weak, strong),
                "Security level reduced".into(),
            ],
            impact: format!(
                "Downgrade from {} ({}) to {} ({}) in {}",
                strong,
                "recommended",
                weak,
                "weak/broken",
                category
            ),
        }
    }

    /// Build the CBC plaintext recovery attack trace.
    pub fn cbc_plaintext_recovery(cipher: &str) -> Self {
        Self {
            vulnerability: SshVulnerability::CbcPlaintextRecovery,
            preconditions: vec![
                "MITM position between client and server".into(),
                format!("CBC cipher '{}' negotiated", cipher),
                "Attacker can observe encrypted traffic".into(),
            ],
            steps: vec![
                AttackStep {
                    step: 1,
                    actor: AttackActor::Attacker,
                    action: "Attacker captures an encrypted block C[i] from the stream".into(),
                    msg_type: None,
                    requires_modification: false,
                },
                AttackStep {
                    step: 2,
                    actor: AttackActor::Attacker,
                    action: "Attacker replaces the last block of a new packet with C[i-1] || C[i], \
                             where C[i-1] is the preceding block"
                        .into(),
                    msg_type: None,
                    requires_modification: true,
                },
                AttackStep {
                    step: 3,
                    actor: AttackActor::Server,
                    action: "Server decrypts the modified packet. The decryption of C[i] \
                             yields P[i] XOR IV (where IV = C[i-1]). The server checks \
                             padding and MAC."
                        .into(),
                    msg_type: None,
                    requires_modification: false,
                },
                AttackStep {
                    step: 4,
                    actor: AttackActor::Attacker,
                    action: "By observing whether the server sends a MAC error (immediately) \
                             or a padding error (with different timing), the attacker learns \
                             ~14 bits of P[i] per attempt."
                        .into(),
                    msg_type: None,
                    requires_modification: false,
                },
            ],
            postconditions: vec![
                "Partial plaintext recovery (~14 bits per attempt)".into(),
                "Connection likely terminates after each attempt".into(),
            ],
            impact: format!(
                "Partial plaintext recovery from {} encrypted traffic. \
                 Requires ~2^14 reconnections per full block recovery.",
                cipher
            ),
        }
    }

    /// Total number of steps.
    pub fn step_count(&self) -> usize {
        self.steps.len()
    }

    /// Steps that require message modification.
    pub fn modification_steps(&self) -> Vec<&AttackStep> {
        self.steps.iter().filter(|s| s.requires_modification).collect()
    }
}

impl fmt::Display for AttackTrace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Attack: {}", self.vulnerability)?;
        writeln!(f, "Preconditions:")?;
        for p in &self.preconditions {
            writeln!(f, "  - {}", p)?;
        }
        writeln!(f, "Steps:")?;
        for step in &self.steps {
            writeln!(f, "  {}. [{}] {}", step.step, step.actor, step.action)?;
        }
        writeln!(f, "Impact: {}", self.impact)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Vulnerability scanner
// ---------------------------------------------------------------------------

/// Scans negotiated parameters for known vulnerabilities.
#[derive(Debug, Clone)]
pub struct VulnerabilityScanner;

impl VulnerabilityScanner {
    /// Scan a pair of KEXINIT messages and strict-KEX state for vulnerabilities.
    pub fn scan(
        client_ki: &KexInit,
        server_ki: &KexInit,
        strict_kex: &StrictKex,
    ) -> Vec<SshVulnerabilityReport> {
        let mut reports = Vec::new();

        // Check for Terrapin vulnerability
        if let Some(r) = Self::check_terrapin(client_ki, server_ki, strict_kex) {
            reports.push(r);
        }

        // Check for weak KEX algorithms
        reports.extend(Self::check_weak_kex(client_ki));
        reports.extend(Self::check_weak_kex(server_ki));

        // Check for CBC ciphers
        reports.extend(Self::check_cbc_ciphers(client_ki));

        // Check for weak MACs
        reports.extend(Self::check_weak_macs(client_ki));

        // Check for no encryption
        reports.extend(Self::check_no_encryption(client_ki));
        reports.extend(Self::check_no_encryption(server_ki));

        // Check for SHA-1 algorithms
        reports.extend(Self::check_sha1(client_ki, server_ki));

        // Check for weak host keys
        reports.extend(Self::check_weak_host_keys(server_ki));

        // Check for pre-auth compression
        reports.extend(Self::check_compression(client_ki));
        reports.extend(Self::check_compression(server_ki));

        // Check strict-KEX support
        if !strict_kex.client_supported || !strict_kex.server_supported {
            reports.push(SshVulnerabilityReport {
                vulnerability: SshVulnerability::NoStrictKex,
                confidence: VulnerabilityConfidence::Confirmed,
                affected_algorithms: Vec::new(),
                recommendation: "Enable strict-KEX (kex-strict-{c,s}-v00@openssh.com) \
                                 on both client and server to mitigate Terrapin."
                    .into(),
            });
        }

        // Deduplicate
        reports.sort_by(|a, b| {
            b.vulnerability
                .severity()
                .partial_cmp(&a.vulnerability.severity())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        reports
    }

    fn check_terrapin(
        client_ki: &KexInit,
        server_ki: &KexInit,
        strict_kex: &StrictKex,
    ) -> Option<SshVulnerabilityReport> {
        if strict_kex.active {
            return None; // mitigated
        }

        // Check if a Terrapin-susceptible cipher could be negotiated
        let client_enc: Vec<&str> = client_ki
            .encryption_algorithms_client_to_server
            .iter()
            .map(|s| s.as_str())
            .collect();
        let server_enc: Vec<&str> = server_ki
            .encryption_algorithms_client_to_server
            .iter()
            .map(|s| s.as_str())
            .collect();

        let susceptible_ciphers: Vec<String> = client_enc
            .iter()
            .filter(|c| server_enc.contains(c))
            .filter(|c| is_terrapin_susceptible(c))
            .map(|c| c.to_string())
            .collect();

        if susceptible_ciphers.is_empty() {
            return None;
        }

        // Also check if CBC+ETM is possible
        let client_has_etm = client_ki
            .mac_algorithms_client_to_server
            .iter()
            .any(|m| m.contains("-etm@"));
        let server_has_etm = server_ki
            .mac_algorithms_client_to_server
            .iter()
            .any(|m| m.contains("-etm@"));

        let client_has_cbc = client_enc.iter().any(|c| c.contains("-cbc"));
        let server_has_cbc = server_enc.iter().any(|c| c.contains("-cbc"));

        let has_chacha = susceptible_ciphers
            .iter()
            .any(|c| c.contains("chacha20"));
        let has_cbc_etm =
            (client_has_cbc && server_has_cbc) && (client_has_etm && server_has_etm);

        if !has_chacha && !has_cbc_etm {
            return None;
        }

        Some(SshVulnerabilityReport {
            vulnerability: SshVulnerability::Terrapin,
            confidence: if has_chacha {
                VulnerabilityConfidence::Confirmed
            } else {
                VulnerabilityConfidence::Likely
            },
            affected_algorithms: susceptible_ciphers,
            recommendation: "Enable strict-KEX on both sides, or disable \
                             ChaCha20-Poly1305 and CBC+ETM cipher/MAC combinations."
                .into(),
        })
    }

    fn check_weak_kex(ki: &KexInit) -> Vec<SshVulnerabilityReport> {
        let mut reports = Vec::new();

        for alg_name in &ki.kex_algorithms {
            if let Some(alg) = KexAlgorithm::from_wire_name(alg_name) {
                if alg.is_pseudo() {
                    continue;
                }
                match alg.security() {
                    SecurityClassification::Broken => {
                        reports.push(SshVulnerabilityReport {
                            vulnerability: SshVulnerability::WeakKeyExchange {
                                algorithm: alg_name.clone(),
                            },
                            confidence: VulnerabilityConfidence::Confirmed,
                            affected_algorithms: vec![alg_name.clone()],
                            recommendation: format!("Remove '{}' from kex algorithms", alg_name),
                        });
                    }
                    SecurityClassification::Weak => {
                        if alg_name.contains("sha1") {
                            reports.push(SshVulnerabilityReport {
                                vulnerability: SshVulnerability::Sha1Usage {
                                    algorithm: alg_name.clone(),
                                },
                                confidence: VulnerabilityConfidence::Confirmed,
                                affected_algorithms: vec![alg_name.clone()],
                                recommendation: format!(
                                    "Replace '{}' with SHA-256/SHA-512 variant",
                                    alg_name
                                ),
                            });
                        }
                    }
                    _ => {}
                }
            }
        }

        reports
    }

    fn check_cbc_ciphers(ki: &KexInit) -> Vec<SshVulnerabilityReport> {
        let mut reports = Vec::new();
        let all_enc = ki
            .encryption_algorithms_client_to_server
            .iter()
            .chain(ki.encryption_algorithms_server_to_client.iter());

        for alg_name in all_enc {
            if let Some(alg) = EncryptionAlgorithm::from_wire_name(alg_name) {
                if alg.is_cbc() {
                    reports.push(SshVulnerabilityReport {
                        vulnerability: SshVulnerability::CbcPlaintextRecovery,
                        confidence: VulnerabilityConfidence::Likely,
                        affected_algorithms: vec![alg_name.clone()],
                        recommendation: format!(
                            "Replace '{}' with CTR or AEAD mode cipher",
                            alg_name
                        ),
                    });
                }
            }
        }

        // Deduplicate CBC reports
        reports.dedup_by(|a, b| {
            matches!(
                (&a.vulnerability, &b.vulnerability),
                (
                    SshVulnerability::CbcPlaintextRecovery,
                    SshVulnerability::CbcPlaintextRecovery
                )
            )
        });

        reports
    }

    fn check_weak_macs(ki: &KexInit) -> Vec<SshVulnerabilityReport> {
        let mut reports = Vec::new();
        let all_macs = ki
            .mac_algorithms_client_to_server
            .iter()
            .chain(ki.mac_algorithms_server_to_client.iter());

        for alg_name in all_macs {
            if alg_name.contains("md5") {
                reports.push(SshVulnerabilityReport {
                    vulnerability: SshVulnerability::Md5Mac,
                    confidence: VulnerabilityConfidence::Confirmed,
                    affected_algorithms: vec![alg_name.clone()],
                    recommendation: "Remove MD5-based MAC algorithms; use hmac-sha2-256 or better"
                        .into(),
                });
            }
        }

        reports.dedup_by(|a, b| {
            matches!(
                (&a.vulnerability, &b.vulnerability),
                (SshVulnerability::Md5Mac, SshVulnerability::Md5Mac)
            )
        });

        reports
    }

    fn check_no_encryption(ki: &KexInit) -> Vec<SshVulnerabilityReport> {
        let has_none = ki
            .encryption_algorithms_client_to_server
            .iter()
            .chain(ki.encryption_algorithms_server_to_client.iter())
            .any(|a| a == "none");

        if has_none {
            vec![SshVulnerabilityReport {
                vulnerability: SshVulnerability::NoEncryption,
                confidence: VulnerabilityConfidence::Confirmed,
                affected_algorithms: vec!["none".into()],
                recommendation: "Remove 'none' from encryption algorithm lists".into(),
            }]
        } else {
            Vec::new()
        }
    }

    fn check_sha1(client_ki: &KexInit, server_ki: &KexInit) -> Vec<SshVulnerabilityReport> {
        let mut reports = Vec::new();

        // Check host key algorithms for ssh-rsa (SHA-1)
        let both_support_ssh_rsa = client_ki
            .server_host_key_algorithms
            .contains(&"ssh-rsa".to_string())
            && server_ki
                .server_host_key_algorithms
                .contains(&"ssh-rsa".to_string());

        if both_support_ssh_rsa {
            reports.push(SshVulnerabilityReport {
                vulnerability: SshVulnerability::SshRsaSha1HostKey,
                confidence: VulnerabilityConfidence::Likely,
                affected_algorithms: vec!["ssh-rsa".into()],
                recommendation: "Use rsa-sha2-256 or rsa-sha2-512 instead of ssh-rsa".into(),
            });
        }

        reports
    }

    fn check_weak_host_keys(server_ki: &KexInit) -> Vec<SshVulnerabilityReport> {
        let mut reports = Vec::new();

        if server_ki
            .server_host_key_algorithms
            .contains(&"ssh-dss".to_string())
        {
            reports.push(SshVulnerabilityReport {
                vulnerability: SshVulnerability::DsaHostKey,
                confidence: VulnerabilityConfidence::Confirmed,
                affected_algorithms: vec!["ssh-dss".into()],
                recommendation: "Remove ssh-dss; use ssh-ed25519 or ecdsa-sha2-nistp256".into(),
            });
        }

        reports
    }

    fn check_compression(ki: &KexInit) -> Vec<SshVulnerabilityReport> {
        let has_preauth_zlib = ki
            .compression_algorithms_client_to_server
            .iter()
            .chain(ki.compression_algorithms_server_to_client.iter())
            .any(|a| a == "zlib");

        if has_preauth_zlib {
            vec![SshVulnerabilityReport {
                vulnerability: SshVulnerability::PreAuthCompression,
                confidence: VulnerabilityConfidence::Likely,
                affected_algorithms: vec!["zlib".into()],
                recommendation: "Use 'zlib@openssh.com' (post-auth) instead of 'zlib'".into(),
            }]
        } else {
            Vec::new()
        }
    }

    /// Scan a single negotiated algorithm set for vulnerabilities.
    pub fn scan_negotiated(
        kex: &str,
        host_key: &str,
        enc_c2s: &str,
        enc_s2c: &str,
        mac_c2s: &str,
        mac_s2c: &str,
        comp_c2s: &str,
        comp_s2c: &str,
        strict_kex_active: bool,
    ) -> Vec<SshVulnerabilityReport> {
        let mut reports = Vec::new();

        // KEX
        if let Some(alg) = KexAlgorithm::from_wire_name(kex) {
            if alg.security() == SecurityClassification::Broken {
                reports.push(SshVulnerabilityReport {
                    vulnerability: SshVulnerability::WeakKeyExchange {
                        algorithm: kex.to_string(),
                    },
                    confidence: VulnerabilityConfidence::Confirmed,
                    affected_algorithms: vec![kex.to_string()],
                    recommendation: "Use curve25519-sha256 or equivalent".into(),
                });
            }
        }

        // Encryption
        for enc in [enc_c2s, enc_s2c] {
            if let Some(alg) = EncryptionAlgorithm::from_wire_name(enc) {
                if alg.is_cbc() {
                    reports.push(SshVulnerabilityReport {
                        vulnerability: SshVulnerability::CbcPlaintextRecovery,
                        confidence: VulnerabilityConfidence::Confirmed,
                        affected_algorithms: vec![enc.to_string()],
                        recommendation: "Use CTR or AEAD cipher".into(),
                    });
                }
                if alg == EncryptionAlgorithm::None {
                    reports.push(SshVulnerabilityReport {
                        vulnerability: SshVulnerability::NoEncryption,
                        confidence: VulnerabilityConfidence::Confirmed,
                        affected_algorithms: vec!["none".into()],
                        recommendation: "Select a real encryption algorithm".into(),
                    });
                }
            }
        }

        // MAC
        for mac in [mac_c2s, mac_s2c] {
            if mac.contains("md5") {
                reports.push(SshVulnerabilityReport {
                    vulnerability: SshVulnerability::Md5Mac,
                    confidence: VulnerabilityConfidence::Confirmed,
                    affected_algorithms: vec![mac.to_string()],
                    recommendation: "Use hmac-sha2-256-etm@openssh.com or better".into(),
                });
            }
        }

        // Terrapin on negotiated ciphers
        if !strict_kex_active {
            let susceptible_enc = [enc_c2s, enc_s2c]
                .iter()
                .any(|e| is_terrapin_susceptible(e));
            let etm_mac = [mac_c2s, mac_s2c].iter().any(|m| m.contains("-etm@"));
            let cbc_enc = [enc_c2s, enc_s2c].iter().any(|e| e.contains("-cbc"));

            if susceptible_enc || (cbc_enc && etm_mac) {
                reports.push(SshVulnerabilityReport {
                    vulnerability: SshVulnerability::Terrapin,
                    confidence: VulnerabilityConfidence::Confirmed,
                    affected_algorithms: vec![enc_c2s.to_string(), enc_s2c.to_string()],
                    recommendation: "Enable strict-KEX or change cipher".into(),
                });
            }
        }

        // Host key
        if host_key == "ssh-rsa" {
            reports.push(SshVulnerabilityReport {
                vulnerability: SshVulnerability::SshRsaSha1HostKey,
                confidence: VulnerabilityConfidence::Confirmed,
                affected_algorithms: vec!["ssh-rsa".into()],
                recommendation: "Use rsa-sha2-256 or rsa-sha2-512".into(),
            });
        }
        if host_key == "ssh-dss" {
            reports.push(SshVulnerabilityReport {
                vulnerability: SshVulnerability::DsaHostKey,
                confidence: VulnerabilityConfidence::Confirmed,
                affected_algorithms: vec!["ssh-dss".into()],
                recommendation: "Use ssh-ed25519 or ecdsa-sha2-nistp256".into(),
            });
        }

        // Compression
        if comp_c2s == "zlib" || comp_s2c == "zlib" {
            reports.push(SshVulnerabilityReport {
                vulnerability: SshVulnerability::PreAuthCompression,
                confidence: VulnerabilityConfidence::Confirmed,
                affected_algorithms: vec!["zlib".into()],
                recommendation: "Use zlib@openssh.com or none".into(),
            });
        }

        reports
    }
}

/// A vulnerability report from the scanner.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SshVulnerabilityReport {
    pub vulnerability: SshVulnerability,
    pub confidence: VulnerabilityConfidence,
    pub affected_algorithms: Vec<String>,
    pub recommendation: String,
}

impl fmt::Display for SshVulnerabilityReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{:.1}] {} ({:?}): {} — Affected: [{}]",
            self.vulnerability.severity(),
            self.vulnerability,
            self.confidence,
            self.recommendation,
            self.affected_algorithms.join(", ")
        )
    }
}

/// Confidence level for a vulnerability finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VulnerabilityConfidence {
    /// Definitely exploitable with the negotiated parameters.
    Confirmed,
    /// Likely exploitable (vulnerable algorithms offered but may not be selected).
    Likely,
    /// Possibly exploitable under specific conditions.
    Possible,
}

/// Returns true if a cipher is susceptible to the Terrapin attack.
fn is_terrapin_susceptible(cipher: &str) -> bool {
    cipher == "chacha20-poly1305@openssh.com"
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kex::KexInitBuilder;

    fn vulnerable_client_ki() -> KexInit {
        KexInitBuilder::new()
            .kex_algorithms(vec!["curve25519-sha256".into()])
            .server_host_key_algorithms(vec!["ssh-ed25519".into(), "ssh-rsa".into()])
            .encryption_c2s(vec![
                "chacha20-poly1305@openssh.com".into(),
                "aes256-cbc".into(),
            ])
            .encryption_s2c(vec![
                "chacha20-poly1305@openssh.com".into(),
                "aes256-cbc".into(),
            ])
            .mac_c2s(vec![
                "hmac-sha2-256-etm@openssh.com".into(),
                "hmac-md5".into(),
            ])
            .mac_s2c(vec![
                "hmac-sha2-256-etm@openssh.com".into(),
                "hmac-md5".into(),
            ])
            .build()
    }

    fn vulnerable_server_ki() -> KexInit {
        KexInitBuilder::new()
            .kex_algorithms(vec![
                "curve25519-sha256".into(),
                "diffie-hellman-group1-sha1".into(),
            ])
            .server_host_key_algorithms(vec![
                "ssh-ed25519".into(),
                "ssh-rsa".into(),
                "ssh-dss".into(),
            ])
            .encryption_c2s(vec![
                "chacha20-poly1305@openssh.com".into(),
                "aes256-cbc".into(),
                "none".into(),
            ])
            .encryption_s2c(vec![
                "chacha20-poly1305@openssh.com".into(),
                "aes256-cbc".into(),
            ])
            .mac_c2s(vec![
                "hmac-sha2-256-etm@openssh.com".into(),
                "hmac-md5".into(),
            ])
            .mac_s2c(vec!["hmac-sha2-256-etm@openssh.com".into()])
            .compression_c2s(vec!["none".into(), "zlib".into()])
            .compression_s2c(vec!["none".into()])
            .build()
    }

    #[test]
    fn scan_detects_terrapin() {
        let strict_kex = StrictKex::new(); // not active
        let reports = VulnerabilityScanner::scan(
            &vulnerable_client_ki(),
            &vulnerable_server_ki(),
            &strict_kex,
        );

        let has_terrapin = reports
            .iter()
            .any(|r| matches!(r.vulnerability, SshVulnerability::Terrapin));
        assert!(has_terrapin, "should detect Terrapin");
    }

    #[test]
    fn scan_no_terrapin_with_strict_kex() {
        let mut strict_kex = StrictKex::new();
        strict_kex.active = true;

        let reports = VulnerabilityScanner::scan(
            &vulnerable_client_ki(),
            &vulnerable_server_ki(),
            &strict_kex,
        );

        let has_terrapin = reports
            .iter()
            .any(|r| matches!(r.vulnerability, SshVulnerability::Terrapin));
        assert!(!has_terrapin, "strict-KEX should mitigate Terrapin");
    }

    #[test]
    fn scan_detects_weak_kex() {
        let strict_kex = StrictKex::new();
        let reports = VulnerabilityScanner::scan(
            &vulnerable_client_ki(),
            &vulnerable_server_ki(),
            &strict_kex,
        );

        let has_weak_kex = reports
            .iter()
            .any(|r| matches!(r.vulnerability, SshVulnerability::WeakKeyExchange { .. }));
        assert!(has_weak_kex, "should detect weak KEX (DH group1)");
    }

    #[test]
    fn scan_detects_cbc() {
        let strict_kex = StrictKex::new();
        let reports = VulnerabilityScanner::scan(
            &vulnerable_client_ki(),
            &vulnerable_server_ki(),
            &strict_kex,
        );

        let has_cbc = reports
            .iter()
            .any(|r| matches!(r.vulnerability, SshVulnerability::CbcPlaintextRecovery));
        assert!(has_cbc, "should detect CBC cipher");
    }

    #[test]
    fn scan_detects_md5() {
        let strict_kex = StrictKex::new();
        let reports = VulnerabilityScanner::scan(
            &vulnerable_client_ki(),
            &vulnerable_server_ki(),
            &strict_kex,
        );

        let has_md5 = reports
            .iter()
            .any(|r| matches!(r.vulnerability, SshVulnerability::Md5Mac));
        assert!(has_md5, "should detect MD5 MAC");
    }

    #[test]
    fn scan_detects_no_encryption() {
        let strict_kex = StrictKex::new();
        let reports = VulnerabilityScanner::scan(
            &vulnerable_client_ki(),
            &vulnerable_server_ki(),
            &strict_kex,
        );

        let has_none = reports
            .iter()
            .any(|r| matches!(r.vulnerability, SshVulnerability::NoEncryption));
        assert!(has_none, "should detect 'none' cipher");
    }

    #[test]
    fn scan_detects_dsa() {
        let strict_kex = StrictKex::new();
        let reports = VulnerabilityScanner::scan(
            &vulnerable_client_ki(),
            &vulnerable_server_ki(),
            &strict_kex,
        );

        let has_dsa = reports
            .iter()
            .any(|r| matches!(r.vulnerability, SshVulnerability::DsaHostKey));
        assert!(has_dsa, "should detect DSA host key");
    }

    #[test]
    fn scan_detects_preauth_compression() {
        let strict_kex = StrictKex::new();
        let reports = VulnerabilityScanner::scan(
            &vulnerable_client_ki(),
            &vulnerable_server_ki(),
            &strict_kex,
        );

        let has_zlib = reports
            .iter()
            .any(|r| matches!(r.vulnerability, SshVulnerability::PreAuthCompression));
        assert!(has_zlib, "should detect pre-auth zlib");
    }

    #[test]
    fn vulnerability_severity() {
        assert!(SshVulnerability::NoEncryption.severity() > 9.0);
        assert!(SshVulnerability::Terrapin.severity() > 5.0);
        assert!(SshVulnerability::PreAuthCompression.severity() < 5.0);
    }

    #[test]
    fn vulnerability_cve() {
        assert_eq!(SshVulnerability::Terrapin.cve(), Some("CVE-2023-48795"));
        assert_eq!(
            SshVulnerability::CbcPlaintextRecovery.cve(),
            Some("CVE-2008-5161")
        );
        assert_eq!(SshVulnerability::Md5Mac.cve(), None);
    }

    #[test]
    fn attack_trace_terrapin() {
        let trace = AttackTrace::terrapin();
        assert!(trace.steps.len() >= 5);
        assert!(trace.modification_steps().len() >= 1);
        assert!(trace.preconditions.len() >= 2);
    }

    #[test]
    fn attack_trace_downgrade() {
        let trace = AttackTrace::algorithm_downgrade(
            "encryption",
            "aes256-gcm@openssh.com",
            "3des-cbc",
        );
        assert!(trace.steps.len() >= 3);
        assert_eq!(
            trace.vulnerability,
            SshVulnerability::AlgorithmDowngrade {
                category: "encryption".into(),
                strong: "aes256-gcm@openssh.com".into(),
                weak: "3des-cbc".into(),
            }
        );
    }

    #[test]
    fn attack_trace_display() {
        let trace = AttackTrace::terrapin();
        let s = format!("{}", trace);
        assert!(s.contains("Terrapin"));
        assert!(s.contains("MITM"));
    }

    #[test]
    fn scan_negotiated_detects_issues() {
        let reports = VulnerabilityScanner::scan_negotiated(
            "diffie-hellman-group1-sha1",
            "ssh-rsa",
            "aes128-cbc",
            "aes128-cbc",
            "hmac-md5",
            "hmac-sha2-256",
            "zlib",
            "none",
            false,
        );

        assert!(reports.len() >= 4, "should detect multiple issues, got {}", reports.len());
    }

    #[test]
    fn vulnerability_display() {
        let v = SshVulnerability::Terrapin;
        let s = format!("{}", v);
        assert!(s.contains("CVE-2023-48795"));
    }

    #[test]
    fn report_display() {
        let report = SshVulnerabilityReport {
            vulnerability: SshVulnerability::Md5Mac,
            confidence: VulnerabilityConfidence::Confirmed,
            affected_algorithms: vec!["hmac-md5".into()],
            recommendation: "Remove MD5".into(),
        };
        let s = format!("{}", report);
        assert!(s.contains("MD5"));
    }

    #[test]
    fn mitre_technique() {
        assert!(SshVulnerability::Terrapin.mitre_technique().is_some());
        assert!(SshVulnerability::NoEncryption.mitre_technique().is_some());
    }
}
