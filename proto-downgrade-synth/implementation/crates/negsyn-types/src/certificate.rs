//! Certificate types based on Definition D6.
//!
//! Provides certificate structures for recording analysis results:
//! either a discovered attack trace or a safety certificate proving
//! the absence of downgrade attacks under given bounds.

use crate::adversary::{AdversaryBudget, AdversaryTrace, DowngradeInfo};
use crate::protocol::{CipherSuite, NegotiationOutcome, ProtocolVersion, TransitionLabel};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt;

// ── Library Identifier ───────────────────────────────────────────────────

/// Identifies the protocol library being analyzed.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LibraryIdentifier {
    pub name: String,
    pub version: String,
    pub commit_hash: Option<String>,
    pub binary_hash: Option<String>,
}

impl LibraryIdentifier {
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Self {
        LibraryIdentifier {
            name: name.into(),
            version: version.into(),
            commit_hash: None,
            binary_hash: None,
        }
    }

    pub fn with_commit(mut self, hash: impl Into<String>) -> Self {
        self.commit_hash = Some(hash.into());
        self
    }

    pub fn with_binary_hash(mut self, hash: impl Into<String>) -> Self {
        self.binary_hash = Some(hash.into());
        self
    }

    /// Compute a stable fingerprint for this library.
    pub fn fingerprint(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.name.as_bytes());
        hasher.update(b":");
        hasher.update(self.version.as_bytes());
        if let Some(ref commit) = self.commit_hash {
            hasher.update(b":");
            hasher.update(commit.as_bytes());
        }
        if let Some(ref bin_hash) = self.binary_hash {
            hasher.update(b":");
            hasher.update(bin_hash.as_bytes());
        }
        hex::encode(hasher.finalize())
    }
}

impl fmt::Display for LibraryIdentifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} v{}", self.name, self.version)?;
        if let Some(ref hash) = self.commit_hash {
            write!(f, " ({})", &hash[..7.min(hash.len())])?;
        }
        Ok(())
    }
}

// ── Bounds Specification ─────────────────────────────────────────────────

/// Bounds under which the analysis was performed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoundsSpec {
    /// Dolev-Yao depth bound (k).
    pub depth_k: u32,
    /// Action count bound (n).
    pub actions_n: u32,
    /// State-space coverage achieved (0.0..=100.0).
    pub coverage_pct: f64,
    /// Path coverage percentage.
    pub path_coverage_pct: f64,
    /// Maximum symbolic execution depth.
    pub max_symex_depth: u32,
}

impl BoundsSpec {
    pub fn new(depth_k: u32, actions_n: u32) -> Self {
        BoundsSpec {
            depth_k,
            actions_n,
            coverage_pct: 0.0,
            path_coverage_pct: 0.0,
            max_symex_depth: 0,
        }
    }

    pub fn with_coverage(mut self, state_pct: f64, path_pct: f64) -> Self {
        self.coverage_pct = state_pct.clamp(0.0, 100.0);
        self.path_coverage_pct = path_pct.clamp(0.0, 100.0);
        self
    }

    pub fn with_symex_depth(mut self, depth: u32) -> Self {
        self.max_symex_depth = depth;
        self
    }

    /// Whether the analysis achieved full coverage under these bounds.
    pub fn is_complete(&self) -> bool {
        self.coverage_pct >= 100.0 && self.path_coverage_pct >= 100.0
    }

    pub fn to_budget(&self) -> AdversaryBudget {
        AdversaryBudget::new(self.depth_k, self.actions_n)
    }
}

impl fmt::Display for BoundsSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "k={}, n={}, coverage={:.1}%/paths={:.1}%",
            self.depth_k, self.actions_n, self.coverage_pct, self.path_coverage_pct
        )
    }
}

// ── Certificate Validity ─────────────────────────────────────────────────

/// Validity status of a certificate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CertificateValidity {
    Valid,
    Expired { reason: String },
    Invalid { reason: String },
    Revoked { reason: String },
}

impl CertificateValidity {
    pub fn is_valid(&self) -> bool {
        matches!(self, CertificateValidity::Valid)
    }
}

impl fmt::Display for CertificateValidity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CertificateValidity::Valid => write!(f, "VALID"),
            CertificateValidity::Expired { reason } => write!(f, "EXPIRED: {}", reason),
            CertificateValidity::Invalid { reason } => write!(f, "INVALID: {}", reason),
            CertificateValidity::Revoked { reason } => write!(f, "REVOKED: {}", reason),
        }
    }
}

// ── Attack Trace ─────────────────────────────────────────────────────────

/// A concrete attack trace demonstrating a downgrade.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttackTrace {
    pub adversary_trace: AdversaryTrace,
    pub downgrade: DowngradeInfo,
    pub expected_outcome: NegotiationOutcome,
    pub actual_outcome: NegotiationOutcome,
    pub protocol_steps: Vec<ProtocolStep>,
}

/// A single step in the protocol execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolStep {
    pub step_number: usize,
    pub actor: StepActor,
    pub action: String,
    pub data_summary: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StepActor {
    Client,
    Server,
    Adversary,
}

impl AttackTrace {
    pub fn new(
        adversary_trace: AdversaryTrace,
        downgrade: DowngradeInfo,
        expected: NegotiationOutcome,
        actual: NegotiationOutcome,
    ) -> Self {
        AttackTrace {
            adversary_trace,
            downgrade,
            expected_outcome: expected,
            actual_outcome: actual,
            protocol_steps: Vec::new(),
        }
    }

    pub fn add_step(&mut self, actor: StepActor, action: impl Into<String>, data: impl Into<String>) {
        let step_number = self.protocol_steps.len();
        self.protocol_steps.push(ProtocolStep {
            step_number,
            actor,
            action: action.into(),
            data_summary: data.into(),
        });
    }

    /// Number of adversary actions in the attack.
    pub fn adversary_action_count(&self) -> usize {
        self.adversary_trace.active_action_count()
    }

    /// Total protocol steps.
    pub fn total_steps(&self) -> usize {
        self.protocol_steps.len()
    }

    /// Severity based on the security level gap.
    pub fn severity(&self) -> AttackSeverity {
        let from = self.downgrade.from_level;
        let to = self.downgrade.to_level;
        use crate::protocol::SecurityLevel;
        match (from, to) {
            (SecurityLevel::High, SecurityLevel::Broken) | (SecurityLevel::Standard, SecurityLevel::Broken) => {
                AttackSeverity::Critical
            }
            (SecurityLevel::High, SecurityLevel::Weak) | (SecurityLevel::Standard, SecurityLevel::Weak) => {
                AttackSeverity::High
            }
            (SecurityLevel::High, SecurityLevel::Legacy) | (SecurityLevel::Standard, SecurityLevel::Legacy) => {
                AttackSeverity::Medium
            }
            _ => AttackSeverity::Low,
        }
    }

    /// Generate a human-readable summary.
    pub fn summary(&self) -> String {
        format!(
            "Downgrade attack ({:?}): {} → {} via {} adversary actions\nExpected: {}\nActual: {}",
            self.severity(),
            self.downgrade.from_level,
            self.downgrade.to_level,
            self.adversary_action_count(),
            self.expected_outcome,
            self.actual_outcome,
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttackSeverity {
    Critical,
    High,
    Medium,
    Low,
}

impl fmt::Display for AttackSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AttackSeverity::Critical => write!(f, "CRITICAL"),
            AttackSeverity::High => write!(f, "HIGH"),
            AttackSeverity::Medium => write!(f, "MEDIUM"),
            AttackSeverity::Low => write!(f, "LOW"),
        }
    }
}

impl fmt::Display for AttackTrace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Attack Trace ===")?;
        writeln!(f, "Severity: {}", self.severity())?;
        writeln!(f, "Downgrade: {}", self.downgrade)?;
        writeln!(f, "Steps:")?;
        for step in &self.protocol_steps {
            writeln!(
                f,
                "  [{:>3}] {:?}: {} ({})",
                step.step_number, step.actor, step.action, step.data_summary
            )?;
        }
        writeln!(f, "Adversary trace: {}", self.adversary_trace)?;
        Ok(())
    }
}

// ── Certificate ──────────────────────────────────────────────────────────

/// A safety certificate (Definition D6) or attack report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Certificate {
    pub id: String,
    pub library: LibraryIdentifier,
    pub bounds: BoundsSpec,
    pub proof_hash: String,
    pub issued_at: String,
    pub expires_at: Option<String>,
    pub analysis_version: String,
    pub notes: Vec<String>,
}

impl Certificate {
    pub fn new(
        library: LibraryIdentifier,
        bounds: BoundsSpec,
        analysis_version: impl Into<String>,
    ) -> Self {
        let issued_at = chrono::Utc::now().to_rfc3339();
        let id = uuid::Uuid::new_v4().to_string();

        // Compute proof hash from library + bounds
        let mut hasher = Sha256::new();
        hasher.update(library.fingerprint().as_bytes());
        hasher.update(format!("{}:{}", bounds.depth_k, bounds.actions_n).as_bytes());
        hasher.update(issued_at.as_bytes());
        let proof_hash = hex::encode(hasher.finalize());

        Certificate {
            id,
            library,
            bounds,
            proof_hash,
            issued_at,
            expires_at: None,
            analysis_version: analysis_version.into(),
            notes: Vec::new(),
        }
    }

    pub fn with_expiry(mut self, expiry: impl Into<String>) -> Self {
        self.expires_at = Some(expiry.into());
        self
    }

    pub fn add_note(&mut self, note: impl Into<String>) {
        self.notes.push(note.into());
    }

    /// Validate the certificate.
    pub fn validate(&self) -> CertificateValidity {
        // Check hash integrity
        let mut hasher = Sha256::new();
        hasher.update(self.library.fingerprint().as_bytes());
        hasher.update(format!("{}:{}", self.bounds.depth_k, self.bounds.actions_n).as_bytes());
        hasher.update(self.issued_at.as_bytes());
        let expected_hash = hex::encode(hasher.finalize());

        if self.proof_hash != expected_hash {
            return CertificateValidity::Invalid {
                reason: "proof hash mismatch".into(),
            };
        }

        // Check expiry
        if let Some(ref expires) = self.expires_at {
            if let (Ok(now), Ok(exp)) = (
                chrono::DateTime::parse_from_rfc3339(&chrono::Utc::now().to_rfc3339()),
                chrono::DateTime::parse_from_rfc3339(expires),
            ) {
                if now > exp {
                    return CertificateValidity::Expired {
                        reason: format!("expired at {}", expires),
                    };
                }
            }
        }

        CertificateValidity::Valid
    }

    /// Serialize to JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize from JSON.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

impl fmt::Display for Certificate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Certificate {}", self.id)?;
        writeln!(f, "  Library: {}", self.library)?;
        writeln!(f, "  Bounds:  {}", self.bounds)?;
        writeln!(f, "  Issued:  {}", self.issued_at)?;
        if let Some(ref exp) = self.expires_at {
            writeln!(f, "  Expires: {}", exp)?;
        }
        writeln!(f, "  Hash:    {}...", &self.proof_hash[..16])?;
        Ok(())
    }
}

// ── Certificate Chain ────────────────────────────────────────────────────

/// A chain of certificates for incremental analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertificateChain {
    pub certificates: Vec<Certificate>,
}

impl CertificateChain {
    pub fn new() -> Self {
        CertificateChain {
            certificates: Vec::new(),
        }
    }

    pub fn push(&mut self, cert: Certificate) {
        self.certificates.push(cert);
    }

    /// Validate the entire chain.
    pub fn validate(&self) -> CertificateValidity {
        if self.certificates.is_empty() {
            return CertificateValidity::Invalid {
                reason: "empty certificate chain".into(),
            };
        }

        for (i, cert) in self.certificates.iter().enumerate() {
            let validity = cert.validate();
            if !validity.is_valid() {
                return CertificateValidity::Invalid {
                    reason: format!("certificate {} failed: {}", i, validity),
                };
            }
        }

        // Check that consecutive certificates are for the same library
        for window in self.certificates.windows(2) {
            if window[0].library.name != window[1].library.name {
                return CertificateValidity::Invalid {
                    reason: format!(
                        "library mismatch: {} vs {}",
                        window[0].library.name, window[1].library.name
                    ),
                };
            }
        }

        CertificateValidity::Valid
    }

    /// Maximum bounds covered by the chain.
    pub fn max_bounds(&self) -> Option<&BoundsSpec> {
        self.certificates
            .iter()
            .max_by(|a, b| {
                (a.bounds.depth_k, a.bounds.actions_n)
                    .cmp(&(b.bounds.depth_k, b.bounds.actions_n))
            })
            .map(|c| &c.bounds)
    }

    pub fn len(&self) -> usize {
        self.certificates.len()
    }

    pub fn is_empty(&self) -> bool {
        self.certificates.is_empty()
    }
}

impl Default for CertificateChain {
    fn default() -> Self {
        Self::new()
    }
}

// ── Analysis Result ──────────────────────────────────────────────────────

/// The top-level result of the NegSynth analysis pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnalysisResult {
    /// An attack was found.
    AttackFound(AttackTrace),
    /// The library is certified safe under the given bounds.
    CertifiedSafe(Certificate),
    /// Analysis was inconclusive.
    Inconclusive(InconclusiveReason),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InconclusiveReason {
    pub reason: String,
    pub partial_coverage: f64,
    pub suggestion: String,
}

impl AnalysisResult {
    pub fn is_attack_found(&self) -> bool {
        matches!(self, AnalysisResult::AttackFound(_))
    }

    pub fn is_safe(&self) -> bool {
        matches!(self, AnalysisResult::CertifiedSafe(_))
    }

    pub fn is_inconclusive(&self) -> bool {
        matches!(self, AnalysisResult::Inconclusive(_))
    }

    pub fn attack_trace(&self) -> Option<&AttackTrace> {
        match self {
            AnalysisResult::AttackFound(t) => Some(t),
            _ => None,
        }
    }

    pub fn certificate(&self) -> Option<&Certificate> {
        match self {
            AnalysisResult::CertifiedSafe(c) => Some(c),
            _ => None,
        }
    }

    /// Serialize to JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize from JSON.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

impl fmt::Display for AnalysisResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AnalysisResult::AttackFound(trace) => {
                write!(f, "ATTACK FOUND: {}", trace.downgrade)
            }
            AnalysisResult::CertifiedSafe(cert) => {
                write!(f, "CERTIFIED SAFE: {} ({})", cert.library, cert.bounds)
            }
            AnalysisResult::Inconclusive(reason) => {
                write!(
                    f,
                    "INCONCLUSIVE: {} ({:.1}% coverage)",
                    reason.reason, reason.partial_coverage
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adversary::{AdversaryBudget, AdversaryTrace, DowngradeInfo, DowngradeKind};
    use crate::protocol::{
        CipherSuiteRegistry, ProtocolVersion, SecurityLevel,
    };

    fn make_test_library() -> LibraryIdentifier {
        LibraryIdentifier::new("openssl", "3.0.12")
            .with_commit("abc123def456")
    }

    #[test]
    fn test_library_identifier() {
        let lib = make_test_library();
        assert_eq!(lib.name, "openssl");
        let display = format!("{}", lib);
        assert!(display.contains("openssl"));
        assert!(display.contains("3.0.12"));
        assert!(display.contains("abc123d"));
    }

    #[test]
    fn test_library_fingerprint() {
        let lib1 = LibraryIdentifier::new("openssl", "3.0.12");
        let lib2 = LibraryIdentifier::new("openssl", "3.0.13");
        assert_ne!(lib1.fingerprint(), lib2.fingerprint());

        let lib3 = LibraryIdentifier::new("openssl", "3.0.12");
        assert_eq!(lib1.fingerprint(), lib3.fingerprint());
    }

    #[test]
    fn test_bounds_spec() {
        let bounds = BoundsSpec::new(5, 20)
            .with_coverage(95.5, 88.0)
            .with_symex_depth(100);
        assert!(!bounds.is_complete());
        assert_eq!(bounds.depth_k, 5);
        assert_eq!(bounds.actions_n, 20);

        let budget = bounds.to_budget();
        assert_eq!(budget.depth_bound, 5);
        assert_eq!(budget.action_bound, 20);
    }

    #[test]
    fn test_bounds_complete() {
        let bounds = BoundsSpec::new(3, 10).with_coverage(100.0, 100.0);
        assert!(bounds.is_complete());
    }

    #[test]
    fn test_certificate_creation_and_validation() {
        let lib = make_test_library();
        let bounds = BoundsSpec::new(5, 20).with_coverage(100.0, 100.0);
        let cert = Certificate::new(lib, bounds, "negsyn-0.1.0");

        let validity = cert.validate();
        assert!(validity.is_valid());
    }

    #[test]
    fn test_certificate_json_roundtrip() {
        let lib = make_test_library();
        let bounds = BoundsSpec::new(5, 20);
        let cert = Certificate::new(lib, bounds, "negsyn-0.1.0");

        let json = cert.to_json().unwrap();
        let cert2 = Certificate::from_json(&json).unwrap();
        assert_eq!(cert.id, cert2.id);
        assert_eq!(cert.library, cert2.library);
    }

    #[test]
    fn test_certificate_invalid_hash() {
        let lib = make_test_library();
        let bounds = BoundsSpec::new(5, 20);
        let mut cert = Certificate::new(lib, bounds, "negsyn-0.1.0");
        cert.proof_hash = "deadbeef".into();

        let validity = cert.validate();
        assert!(!validity.is_valid());
    }

    #[test]
    fn test_certificate_chain() {
        let lib = make_test_library();
        let mut chain = CertificateChain::new();
        chain.push(Certificate::new(lib.clone(), BoundsSpec::new(3, 10), "0.1"));
        chain.push(Certificate::new(lib.clone(), BoundsSpec::new(5, 20), "0.1"));

        assert_eq!(chain.len(), 2);
        let validity = chain.validate();
        assert!(validity.is_valid());

        let max = chain.max_bounds().unwrap();
        assert_eq!(max.depth_k, 5);
    }

    #[test]
    fn test_empty_chain_invalid() {
        let chain = CertificateChain::new();
        assert!(!chain.validate().is_valid());
    }

    #[test]
    fn test_attack_trace() {
        let budget = AdversaryBudget::new(5, 20);
        let trace = AdversaryTrace::new(budget);

        let downgrade = DowngradeInfo {
            kind: DowngradeKind::CipherSuite,
            from_level: SecurityLevel::High,
            to_level: SecurityLevel::Broken,
            description: "test downgrade".into(),
        };

        let expected = crate::protocol::NegotiationOutcome {
            selected_cipher: CipherSuiteRegistry::lookup(0x1301).unwrap(),
            version: ProtocolVersion::tls13(),
            extensions: vec![],
            session_resumed: false,
        };
        let actual = crate::protocol::NegotiationOutcome {
            selected_cipher: CipherSuiteRegistry::lookup(0x0005).unwrap(),
            version: ProtocolVersion::tls12(),
            extensions: vec![],
            session_resumed: false,
        };

        let mut attack = AttackTrace::new(trace, downgrade, expected, actual);
        attack.add_step(StepActor::Client, "ClientHello", "TLS 1.3, 5 ciphers");
        attack.add_step(StepActor::Adversary, "Modify", "downgrade cipher list");
        attack.add_step(StepActor::Server, "ServerHello", "RC4_128_SHA");

        assert_eq!(attack.total_steps(), 3);
        assert_eq!(attack.severity(), AttackSeverity::Critical);
    }

    #[test]
    fn test_analysis_result() {
        let lib = make_test_library();
        let bounds = BoundsSpec::new(5, 20);
        let cert = Certificate::new(lib, bounds, "negsyn-0.1.0");

        let result = AnalysisResult::CertifiedSafe(cert);
        assert!(result.is_safe());
        assert!(!result.is_attack_found());
        assert!(result.certificate().is_some());

        let json = result.to_json().unwrap();
        let result2 = AnalysisResult::from_json(&json).unwrap();
        assert!(result2.is_safe());
    }

    #[test]
    fn test_inconclusive_result() {
        let result = AnalysisResult::Inconclusive(InconclusiveReason {
            reason: "timeout".into(),
            partial_coverage: 45.0,
            suggestion: "increase timeout".into(),
        });
        assert!(result.is_inconclusive());
        let display = format!("{}", result);
        assert!(display.contains("INCONCLUSIVE"));
        assert!(display.contains("45.0%"));
    }

    #[test]
    fn test_certificate_validity_display() {
        assert_eq!(format!("{}", CertificateValidity::Valid), "VALID");
        let exp = CertificateValidity::Expired {
            reason: "too old".into(),
        };
        assert!(format!("{}", exp).contains("EXPIRED"));
    }

    #[test]
    fn test_attack_severity() {
        let check = |from: SecurityLevel, to: SecurityLevel, expected: AttackSeverity| {
            let budget = AdversaryBudget::new(5, 20);
            let trace = AdversaryTrace::new(budget);
            let di = DowngradeInfo {
                kind: DowngradeKind::CipherSuite,
                from_level: from,
                to_level: to,
                description: "test".into(),
            };
            let outcome = crate::protocol::NegotiationOutcome {
                selected_cipher: CipherSuiteRegistry::lookup(0x1301).unwrap(),
                version: ProtocolVersion::tls13(),
                extensions: vec![],
                session_resumed: false,
            };
            let attack = AttackTrace::new(trace, di, outcome.clone(), outcome);
            assert_eq!(attack.severity(), expected);
        };
        check(SecurityLevel::High, SecurityLevel::Broken, AttackSeverity::Critical);
        check(SecurityLevel::High, SecurityLevel::Weak, AttackSeverity::High);
        check(SecurityLevel::High, SecurityLevel::Legacy, AttackSeverity::Medium);
        check(SecurityLevel::Weak, SecurityLevel::Broken, AttackSeverity::Low);
    }
}
