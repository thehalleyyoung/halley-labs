//! Bounded-completeness certificate generation (Definition D6).
//!
//! Produces certificates that attest to the completeness of the analysis
//! within the given bounds.  A certificate either:
//! - Accompanies a concrete attack (SAT certificate)
//! - Attests that no attack exists within the bounds (UNSAT certificate)

use crate::cegar::{CegarResult, CegarStats};
use crate::refinement::RefinementHistory;
use crate::trace::ConcreteTrace;
use crate::validation::{TraceValidator, ValidationReport};
use crate::{ConcreteError, ConcreteResult, LibraryId, SmtFormula, UnsatProof};
use crate::ProtocolVersion;
use serde::{Deserialize, Serialize};
use sha2::{Sha256, Digest};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

// ── BoundedCertificate ───────────────────────────────────────────────────

/// A bounded-completeness certificate (Definition D6).
///
/// Certifies that the analysis has explored the search space within
/// specified bounds and either found an attack or proved absence of attacks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundedCertificate {
    /// Unique certificate identifier.
    pub certificate_id: String,
    /// Version of the certificate format.
    pub format_version: u32,
    /// Library being analyzed.
    pub library: LibraryId,
    /// Analysis bounds.
    pub bounds: AnalysisBounds,
    /// Certificate verdict.
    pub verdict: CertificateVerdict,
    /// UNSAT proof (if verdict is Safe).
    pub unsat_proof: Option<UnsatProof>,
    /// Attack trace (if verdict is Vulnerable).
    pub attack_trace: Option<ConcreteTrace>,
    /// Coverage metrics.
    pub coverage: CoverageMetrics,
    /// CEGAR statistics.
    pub cegar_stats: CegarStats,
    /// Refinement summary.
    pub refinement_summary: RefinementSummary,
    /// Timestamp of certificate generation.
    pub timestamp: u64,
    /// SHA-256 hash of the certificate content (self-signed).
    pub content_hash: String,
    /// Chain link: hash of parent certificate (if part of a chain).
    pub parent_hash: Option<String>,
}

/// Analysis bounds that scope the certificate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisBounds {
    /// Maximum number of protocol messages considered.
    pub max_messages: usize,
    /// Maximum number of cipher suites considered.
    pub max_cipher_suites: usize,
    /// Protocol versions in scope.
    pub protocol_versions: Vec<ProtocolVersion>,
    /// Maximum adversary actions.
    pub max_adversary_actions: usize,
    /// CEGAR iteration bound.
    pub cegar_iterations: usize,
    /// Solver timeout per query.
    pub solver_timeout_ms: u64,
}

impl Default for AnalysisBounds {
    fn default() -> Self {
        Self {
            max_messages: 50,
            max_cipher_suites: 350,
            protocol_versions: vec![
                ProtocolVersion::Ssl30,
                ProtocolVersion::Tls10,
                ProtocolVersion::Tls11,
                ProtocolVersion::Tls12,
                ProtocolVersion::Tls13,
            ],
            max_adversary_actions: 100,
            cegar_iterations: 1000,
            solver_timeout_ms: 30_000,
        }
    }
}

/// Certificate verdict.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CertificateVerdict {
    /// The library is vulnerable: a concrete attack was found.
    Vulnerable {
        severity: u32,
        downgrade_from: ProtocolVersion,
        downgrade_to: ProtocolVersion,
    },
    /// The library is safe within the given bounds.
    Safe {
        proof_core_size: usize,
    },
    /// Analysis was inconclusive (timeout / resource limit).
    Inconclusive {
        reason: String,
    },
}

impl fmt::Display for CertificateVerdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Vulnerable { severity, downgrade_from, downgrade_to } => {
                write!(f, "VULNERABLE (severity={}, {} → {})", severity, downgrade_from, downgrade_to)
            }
            Self::Safe { proof_core_size } => {
                write!(f, "SAFE (proof core: {} clauses)", proof_core_size)
            }
            Self::Inconclusive { reason } => {
                write!(f, "INCONCLUSIVE: {}", reason)
            }
        }
    }
}

/// Coverage metrics showing what fraction of the state space was explored.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CoverageMetrics {
    /// Total reachable states estimated.
    pub estimated_reachable_states: usize,
    /// States actually explored.
    pub explored_states: usize,
    /// Cipher suites explored.
    pub cipher_suites_explored: usize,
    /// Cipher suites excluded by refinement.
    pub cipher_suites_excluded: usize,
    /// Version combinations explored.
    pub version_combinations_explored: usize,
    /// Coverage as a fraction [0, 1].
    pub coverage_fraction: f64,
}

impl CoverageMetrics {
    /// Compute coverage from exploration data.
    pub fn compute(
        explored: usize,
        total_cipher_suites: usize,
        total_versions: usize,
        max_messages: usize,
        refinement_count: usize,
    ) -> Self {
        let estimated = total_cipher_suites
            .saturating_mul(total_versions)
            .saturating_mul(max_messages.max(1));
        let estimated = estimated.max(1);
        let coverage = (explored as f64) / (estimated as f64);

        Self {
            estimated_reachable_states: estimated,
            explored_states: explored,
            cipher_suites_explored: total_cipher_suites.saturating_sub(refinement_count),
            cipher_suites_excluded: refinement_count,
            version_combinations_explored: total_versions,
            coverage_fraction: coverage.min(1.0),
        }
    }

    /// Coverage as a percentage string.
    pub fn coverage_percent(&self) -> String {
        format!("{:.1}%", self.coverage_fraction * 100.0)
    }
}

/// Summary of refinements applied during CEGAR.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RefinementSummary {
    pub total_iterations: usize,
    pub total_predicates: usize,
    pub predicate_kinds: BTreeMap<String, usize>,
    pub minimized_predicates: usize,
}

// ── CertificateGenerator ─────────────────────────────────────────────────

/// Generates bounded-completeness certificates from CEGAR results.
pub struct CertificateGenerator {
    library: LibraryId,
    bounds: AnalysisBounds,
}

impl CertificateGenerator {
    pub fn new(library: LibraryId, bounds: AnalysisBounds) -> Self {
        Self { library, bounds }
    }

    /// Generate a certificate from a CEGAR result.
    pub fn generate(&self, result: &CegarResult, history: &RefinementHistory) -> ConcreteResult<BoundedCertificate> {
        let (verdict, unsat_proof, attack_trace) = match result {
            CegarResult::ConcreteAttack { trace, .. } => {
                let severity = trace.downgrade_severity();
                (
                    CertificateVerdict::Vulnerable {
                        severity,
                        downgrade_from: trace.initial_version,
                        downgrade_to: trace.downgraded_version,
                    },
                    None,
                    Some(trace.clone()),
                )
            }
            CegarResult::CertifiedSafe { proof, .. } => {
                (
                    CertificateVerdict::Safe {
                        proof_core_size: proof.core_size(),
                    },
                    Some(proof.clone()),
                    None,
                )
            }
            CegarResult::Timeout { reason, .. } => {
                (
                    CertificateVerdict::Inconclusive {
                        reason: reason.clone(),
                    },
                    None,
                    None,
                )
            }
        };

        let coverage = CoverageMetrics::compute(
            result.stats().total_iterations,
            self.bounds.max_cipher_suites,
            self.bounds.protocol_versions.len(),
            self.bounds.max_messages,
            history.total_predicates(),
        );

        let refinement_summary = RefinementSummary {
            total_iterations: history.iteration_count(),
            total_predicates: history.total_predicates(),
            predicate_kinds: history.kind_distribution().clone(),
            minimized_predicates: history.minimize().len(),
        };

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let certificate_id = format!(
            "cert-{}-{}-{}",
            self.library.name,
            self.library.version,
            timestamp
        );

        let mut cert = BoundedCertificate {
            certificate_id,
            format_version: 1,
            library: self.library.clone(),
            bounds: self.bounds.clone(),
            verdict,
            unsat_proof,
            attack_trace,
            coverage,
            cegar_stats: result.stats().clone(),
            refinement_summary,
            timestamp,
            content_hash: String::new(),
            parent_hash: None,
        };

        // Compute content hash
        cert.content_hash = self.compute_hash(&cert)?;

        Ok(cert)
    }

    /// Compute SHA-256 hash of certificate content.
    fn compute_hash(&self, cert: &BoundedCertificate) -> ConcreteResult<String> {
        let mut hasher = Sha256::new();

        // Hash key fields (not including the hash itself)
        hasher.update(cert.certificate_id.as_bytes());
        hasher.update(cert.format_version.to_le_bytes());
        hasher.update(cert.library.name.as_bytes());
        hasher.update(cert.library.version.as_bytes());
        hasher.update(cert.timestamp.to_le_bytes());
        hasher.update(format!("{:?}", cert.verdict).as_bytes());
        hasher.update(cert.coverage.explored_states.to_le_bytes());
        hasher.update(cert.cegar_stats.total_iterations.to_le_bytes());

        if let Some(ref proof) = cert.unsat_proof {
            for clause in &proof.core {
                hasher.update(clause.as_bytes());
            }
        }

        if let Some(ref trace) = cert.attack_trace {
            hasher.update(trace.message_count().to_le_bytes());
            for msg in &trace.messages {
                hasher.update(&msg.raw_bytes);
            }
        }

        let result = hasher.finalize();
        Ok(hex::encode(result))
    }

    /// Verify a certificate's integrity.
    pub fn verify(&self, cert: &BoundedCertificate) -> ConcreteResult<CertificateVerification> {
        let mut issues = Vec::new();

        // 1. Verify hash
        let expected_hash = self.compute_hash(cert)?;
        if cert.content_hash != expected_hash {
            issues.push("Content hash mismatch".into());
        }

        // 2. Verify format version
        if cert.format_version != 1 {
            issues.push(format!("Unsupported format version: {}", cert.format_version));
        }

        // 3. Verify bounds are reasonable
        if cert.bounds.max_cipher_suites == 0 {
            issues.push("Invalid bounds: max_cipher_suites = 0".into());
        }
        if cert.bounds.protocol_versions.is_empty() {
            issues.push("Invalid bounds: no protocol versions".into());
        }

        // 4. Verify verdict consistency
        match &cert.verdict {
            CertificateVerdict::Vulnerable { .. } => {
                if cert.attack_trace.is_none() {
                    issues.push("Vulnerable verdict but no attack trace".into());
                }
            }
            CertificateVerdict::Safe { .. } => {
                if cert.unsat_proof.is_none() {
                    issues.push("Safe verdict but no UNSAT proof".into());
                }
            }
            CertificateVerdict::Inconclusive { .. } => {
                // No additional requirements
            }
        }

        // 5. Verify coverage metrics
        if cert.coverage.coverage_fraction < 0.0 || cert.coverage.coverage_fraction > 1.0 {
            issues.push(format!(
                "Invalid coverage fraction: {}",
                cert.coverage.coverage_fraction
            ));
        }

        // 6. Verify UNSAT proof if present
        if let Some(ref proof) = cert.unsat_proof {
            if !proof.is_valid {
                issues.push("UNSAT proof marked as invalid".into());
            }
            if proof.core.is_empty() {
                issues.push("UNSAT proof has empty core".into());
            }
        }

        // 7. Verify attack trace if present
        if let Some(ref trace) = cert.attack_trace {
            if trace.messages.is_empty() {
                issues.push("Attack trace has no messages".into());
            }
            if trace.downgrade_severity() == 0 {
                if let CertificateVerdict::Vulnerable { .. } = cert.verdict {
                    issues.push("Vulnerable verdict but no downgrade in trace".into());
                }
            }
        }

        // 8. Check parent hash if in chain
        if let Some(ref parent) = cert.parent_hash {
            if parent.is_empty() {
                issues.push("Parent hash is empty".into());
            }
        }

        let is_valid = issues.is_empty();
        Ok(CertificateVerification {
            is_valid,
            issues,
            certificate_id: cert.certificate_id.clone(),
        })
    }
}

/// Result of certificate verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertificateVerification {
    pub is_valid: bool,
    pub issues: Vec<String>,
    pub certificate_id: String,
}

impl fmt::Display for CertificateVerification {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Certificate {}: {}",
            self.certificate_id,
            if self.is_valid { "VALID" } else { "INVALID" }
        )?;
        for issue in &self.issues {
            write!(f, "\n  - {}", issue)?;
        }
        Ok(())
    }
}

// ── CertificateBuilder ──────────────────────────────────────────────────

/// Builder for constructing certificates step by step.
pub struct CertificateBuilder {
    library: Option<LibraryId>,
    bounds: AnalysisBounds,
    verdict: Option<CertificateVerdict>,
    unsat_proof: Option<UnsatProof>,
    attack_trace: Option<ConcreteTrace>,
    coverage: Option<CoverageMetrics>,
    cegar_stats: CegarStats,
    refinement_summary: RefinementSummary,
    parent_hash: Option<String>,
}

impl CertificateBuilder {
    pub fn new() -> Self {
        Self {
            library: None,
            bounds: AnalysisBounds::default(),
            verdict: None,
            unsat_proof: None,
            attack_trace: None,
            coverage: None,
            cegar_stats: CegarStats::default(),
            refinement_summary: RefinementSummary::default(),
            parent_hash: None,
        }
    }

    pub fn library(mut self, library: LibraryId) -> Self {
        self.library = Some(library);
        self
    }

    pub fn bounds(mut self, bounds: AnalysisBounds) -> Self {
        self.bounds = bounds;
        self
    }

    pub fn verdict(mut self, verdict: CertificateVerdict) -> Self {
        self.verdict = Some(verdict);
        self
    }

    pub fn unsat_proof(mut self, proof: UnsatProof) -> Self {
        self.unsat_proof = Some(proof);
        self
    }

    pub fn attack_trace(mut self, trace: ConcreteTrace) -> Self {
        self.attack_trace = Some(trace);
        self
    }

    pub fn coverage(mut self, coverage: CoverageMetrics) -> Self {
        self.coverage = Some(coverage);
        self
    }

    pub fn cegar_stats(mut self, stats: CegarStats) -> Self {
        self.cegar_stats = stats;
        self
    }

    pub fn refinement_summary(mut self, summary: RefinementSummary) -> Self {
        self.refinement_summary = summary;
        self
    }

    pub fn parent_hash(mut self, hash: impl Into<String>) -> Self {
        self.parent_hash = Some(hash.into());
        self
    }

    /// Build the certificate.
    pub fn build(self) -> ConcreteResult<BoundedCertificate> {
        let library = self
            .library
            .ok_or_else(|| ConcreteError::Certificate("Library ID required".into()))?;
        let verdict = self
            .verdict
            .ok_or_else(|| ConcreteError::Certificate("Verdict required".into()))?;

        let coverage = self.coverage.unwrap_or_default();

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let certificate_id = format!(
            "cert-{}-{}-{}",
            library.name, library.version, timestamp
        );

        let mut cert = BoundedCertificate {
            certificate_id,
            format_version: 1,
            library: library.clone(),
            bounds: self.bounds,
            verdict,
            unsat_proof: self.unsat_proof,
            attack_trace: self.attack_trace,
            coverage,
            cegar_stats: self.cegar_stats,
            refinement_summary: self.refinement_summary,
            timestamp,
            content_hash: String::new(),
            parent_hash: self.parent_hash,
        };

        // Compute hash
        let gen = CertificateGenerator::new(library, AnalysisBounds::default());
        cert.content_hash = gen.compute_hash(&cert)?;

        Ok(cert)
    }
}

impl Default for CertificateBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ── UNSAT proof extraction ───────────────────────────────────────────────

/// Extract and package an UNSAT proof from solver output.
pub fn extract_unsat_proof(
    core_clauses: Vec<String>,
    resolution_log: &[(usize, usize, String, usize)],
) -> UnsatProof {
    let resolution_steps: Vec<crate::ResolutionStep> = resolution_log
        .iter()
        .map(|(a, b, pivot, result)| crate::ResolutionStep {
            clause_a: *a,
            clause_b: *b,
            pivot: pivot.clone(),
            result: *result,
        })
        .collect();

    let is_valid = !core_clauses.is_empty() && validate_resolution_proof(&core_clauses, &resolution_steps);

    UnsatProof {
        core: core_clauses,
        resolution_steps,
        is_valid,
    }
}

/// Basic validation of a resolution proof.
fn validate_resolution_proof(
    core: &[String],
    steps: &[crate::ResolutionStep],
) -> bool {
    if core.is_empty() {
        return false;
    }

    // Check that all resolution steps reference valid clause indices
    let max_clause = core.len() + steps.len();
    for step in steps {
        if step.clause_a >= max_clause || step.clause_b >= max_clause {
            return false;
        }
        if step.result >= max_clause + steps.len() {
            return false;
        }
    }

    true
}

// ── Certificate serialization ────────────────────────────────────────────

impl BoundedCertificate {
    /// Serialize to JSON.
    pub fn to_json(&self) -> ConcreteResult<String> {
        serde_json::to_string_pretty(self).map_err(ConcreteError::Json)
    }

    /// Deserialize from JSON.
    pub fn from_json(json: &str) -> ConcreteResult<Self> {
        serde_json::from_str(json).map_err(ConcreteError::Json)
    }

    /// Generate a human-readable summary.
    pub fn summary(&self) -> String {
        let mut s = String::with_capacity(1024);
        s.push_str("══════════════════════════════════════════\n");
        s.push_str("     BOUNDED-COMPLETENESS CERTIFICATE     \n");
        s.push_str("══════════════════════════════════════════\n");
        s.push_str(&format!("ID:       {}\n", self.certificate_id));
        s.push_str(&format!("Library:  {}\n", self.library));
        s.push_str(&format!("Verdict:  {}\n", self.verdict));
        s.push_str(&format!("Coverage: {}\n", self.coverage.coverage_percent()));
        s.push_str(&format!(
            "Bounds:   {} ciphers × {} versions × {} msgs\n",
            self.bounds.max_cipher_suites,
            self.bounds.protocol_versions.len(),
            self.bounds.max_messages,
        ));
        s.push_str(&format!("CEGAR:    {}\n", self.cegar_stats.summary()));
        s.push_str(&format!("Hash:     {}\n", &self.content_hash[..16]));
        if let Some(ref parent) = self.parent_hash {
            s.push_str(&format!("Parent:   {}\n", &parent[..16.min(parent.len())]));
        }
        s.push_str("══════════════════════════════════════════\n");
        s
    }
}

impl fmt::Display for BoundedCertificate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.summary())
    }
}

// ── Certificate chain for multi-library analysis ─────────────────────────

/// A chain of certificates from multi-library analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertificateChain {
    pub chain_id: String,
    pub certificates: Vec<BoundedCertificate>,
    pub chain_hash: String,
}

impl CertificateChain {
    /// Build a new chain from ordered certificates.
    pub fn new(certificates: Vec<BoundedCertificate>) -> ConcreteResult<Self> {
        if certificates.is_empty() {
            return Err(ConcreteError::Certificate("Cannot create empty chain".into()));
        }

        // Compute chain hash
        let mut hasher = Sha256::new();
        for cert in &certificates {
            hasher.update(cert.content_hash.as_bytes());
        }
        let chain_hash = hex::encode(hasher.finalize());

        let chain_id = format!(
            "chain-{}-{}",
            certificates.len(),
            &chain_hash[..8]
        );

        Ok(Self {
            chain_id,
            certificates,
            chain_hash,
        })
    }

    /// Verify the entire chain.
    pub fn verify(&self) -> ConcreteResult<ChainVerification> {
        let mut all_valid = true;
        let mut cert_results = Vec::new();
        let mut chain_hash_valid = true;

        // Verify each certificate
        for (i, cert) in self.certificates.iter().enumerate() {
            let gen = CertificateGenerator::new(cert.library.clone(), cert.bounds.clone());
            let verification = gen.verify(cert)?;
            if !verification.is_valid {
                all_valid = false;
            }

            // Check chain linkage
            if i > 0 {
                if let Some(ref parent) = cert.parent_hash {
                    if *parent != self.certificates[i - 1].content_hash {
                        all_valid = false;
                        cert_results.push(CertificateVerification {
                            is_valid: false,
                            issues: vec![format!(
                                "Parent hash mismatch at position {}",
                                i
                            )],
                            certificate_id: cert.certificate_id.clone(),
                        });
                        continue;
                    }
                }
            }

            cert_results.push(verification);
        }

        // Verify chain hash
        let mut hasher = Sha256::new();
        for cert in &self.certificates {
            hasher.update(cert.content_hash.as_bytes());
        }
        let computed_hash = hex::encode(hasher.finalize());
        if computed_hash != self.chain_hash {
            chain_hash_valid = false;
            all_valid = false;
        }

        Ok(ChainVerification {
            is_valid: all_valid,
            chain_hash_valid,
            certificate_results: cert_results,
            chain_id: self.chain_id.clone(),
        })
    }

    /// Get all vulnerable libraries in the chain.
    pub fn vulnerable_libraries(&self) -> Vec<&LibraryId> {
        self.certificates
            .iter()
            .filter(|c| matches!(c.verdict, CertificateVerdict::Vulnerable { .. }))
            .map(|c| &c.library)
            .collect()
    }

    /// Get all safe libraries in the chain.
    pub fn safe_libraries(&self) -> Vec<&LibraryId> {
        self.certificates
            .iter()
            .filter(|c| matches!(c.verdict, CertificateVerdict::Safe { .. }))
            .map(|c| &c.library)
            .collect()
    }

    pub fn to_json(&self) -> ConcreteResult<String> {
        serde_json::to_string_pretty(self).map_err(ConcreteError::Json)
    }
}

/// Result of chain verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainVerification {
    pub is_valid: bool,
    pub chain_hash_valid: bool,
    pub certificate_results: Vec<CertificateVerification>,
    pub chain_id: String,
}

impl fmt::Display for ChainVerification {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Chain {}: {} ({} certificates, chain hash: {})",
            self.chain_id,
            if self.is_valid { "VALID" } else { "INVALID" },
            self.certificate_results.len(),
            if self.chain_hash_valid { "OK" } else { "MISMATCH" },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cegar::CegarStats;
    use crate::trace::TraceBuilder;

    fn make_test_library() -> LibraryId {
        LibraryId::new("openssl", "1.1.1", "TLS")
    }

    fn make_test_trace() -> ConcreteTrace {
        let mut builder = TraceBuilder::new(ProtocolVersion::Tls12, ProtocolVersion::Ssl30)
            .with_library("openssl", "1.1.1");
        builder.client_send(
            vec![0x16, 0x03, 0x03, 0x00, 0x02, 0x01, 0x00],
            ProtocolVersion::Tls12,
            crate::HandshakePhase::ClientHello,
        );
        builder.server_send(
            vec![0x16, 0x03, 0x00, 0x00, 0x02, 0x02, 0x00],
            ProtocolVersion::Ssl30,
            crate::HandshakePhase::ServerHello,
        );
        builder.set_negotiated_cipher(crate::CipherSuite::new(
            0x002f, "TLS_RSA_WITH_AES_128_CBC_SHA",
            crate::KeyExchange::Rsa,
            crate::AuthAlgorithm::Rsa,
            crate::BulkEncryption::Aes128,
            crate::MacAlgorithm::HmacSha1,
        ));
        builder.build()
    }

    fn make_test_proof() -> UnsatProof {
        UnsatProof::new(vec![
            "cipher_constraint_1".into(),
            "version_constraint_2".into(),
            "ordering_constraint_3".into(),
        ])
    }

    #[test]
    fn test_certificate_generation_vulnerable() {
        let library = make_test_library();
        let trace = make_test_trace();
        let stats = CegarStats {
            total_iterations: 5,
            sat_results: 3,
            genuine_traces: 1,
            ..Default::default()
        };
        let result = CegarResult::ConcreteAttack {
            trace,
            iterations: 5,
            stats,
        };

        let history = RefinementHistory::new();
        let gen = CertificateGenerator::new(library, AnalysisBounds::default());
        let cert = gen.generate(&result, &history).unwrap();

        assert!(matches!(cert.verdict, CertificateVerdict::Vulnerable { .. }));
        assert!(cert.attack_trace.is_some());
        assert!(!cert.content_hash.is_empty());
    }

    #[test]
    fn test_certificate_generation_safe() {
        let library = make_test_library();
        let proof = make_test_proof();
        let stats = CegarStats {
            total_iterations: 10,
            unsat_results: 1,
            ..Default::default()
        };
        let result = CegarResult::CertifiedSafe {
            proof,
            iterations: 10,
            stats,
        };

        let history = RefinementHistory::new();
        let gen = CertificateGenerator::new(library, AnalysisBounds::default());
        let cert = gen.generate(&result, &history).unwrap();

        assert!(matches!(cert.verdict, CertificateVerdict::Safe { .. }));
        assert!(cert.unsat_proof.is_some());
    }

    #[test]
    fn test_certificate_verification() {
        let library = make_test_library();
        let trace = make_test_trace();
        let stats = CegarStats::default();
        let result = CegarResult::ConcreteAttack {
            trace,
            iterations: 1,
            stats,
        };

        let history = RefinementHistory::new();
        let gen = CertificateGenerator::new(library, AnalysisBounds::default());
        let cert = gen.generate(&result, &history).unwrap();
        let verification = gen.verify(&cert).unwrap();
        assert!(verification.is_valid, "Certificate should verify: {:?}", verification.issues);
    }

    #[test]
    fn test_certificate_tamper_detection() {
        let library = make_test_library();
        let proof = make_test_proof();
        let stats = CegarStats::default();
        let result = CegarResult::CertifiedSafe {
            proof,
            iterations: 1,
            stats,
        };

        let history = RefinementHistory::new();
        let gen = CertificateGenerator::new(library.clone(), AnalysisBounds::default());
        let mut cert = gen.generate(&result, &history).unwrap();

        // Tamper with the certificate
        cert.content_hash = "deadbeef".into();
        let verification = gen.verify(&cert).unwrap();
        assert!(!verification.is_valid);
        assert!(verification.issues.iter().any(|i| i.contains("hash")));
    }

    #[test]
    fn test_certificate_json_roundtrip() {
        let library = make_test_library();
        let proof = make_test_proof();
        let result = CegarResult::CertifiedSafe {
            proof,
            iterations: 1,
            stats: CegarStats::default(),
        };

        let history = RefinementHistory::new();
        let gen = CertificateGenerator::new(library, AnalysisBounds::default());
        let cert = gen.generate(&result, &history).unwrap();

        let json = cert.to_json().unwrap();
        let deserialized = BoundedCertificate::from_json(&json).unwrap();
        assert_eq!(deserialized.certificate_id, cert.certificate_id);
        assert_eq!(deserialized.content_hash, cert.content_hash);
    }

    #[test]
    fn test_certificate_builder() {
        let cert = CertificateBuilder::new()
            .library(make_test_library())
            .verdict(CertificateVerdict::Safe { proof_core_size: 5 })
            .unsat_proof(make_test_proof())
            .build()
            .unwrap();

        assert!(matches!(cert.verdict, CertificateVerdict::Safe { .. }));
        assert!(!cert.content_hash.is_empty());
    }

    #[test]
    fn test_certificate_builder_missing_library() {
        let result = CertificateBuilder::new()
            .verdict(CertificateVerdict::Inconclusive { reason: "test".into() })
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn test_certificate_builder_missing_verdict() {
        let result = CertificateBuilder::new()
            .library(make_test_library())
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn test_coverage_metrics() {
        let coverage = CoverageMetrics::compute(50, 350, 5, 10, 20);
        assert!(coverage.coverage_fraction > 0.0);
        assert!(coverage.coverage_fraction <= 1.0);
        assert_eq!(coverage.cipher_suites_excluded, 20);
        assert_eq!(coverage.cipher_suites_explored, 330);
    }

    #[test]
    fn test_certificate_chain() {
        let library1 = LibraryId::new("openssl", "1.1.1", "TLS");
        let library2 = LibraryId::new("gnutls", "3.7.0", "TLS");

        let cert1 = CertificateBuilder::new()
            .library(library1)
            .verdict(CertificateVerdict::Vulnerable {
                severity: 3,
                downgrade_from: ProtocolVersion::Tls12,
                downgrade_to: ProtocolVersion::Ssl30,
            })
            .attack_trace(make_test_trace())
            .build()
            .unwrap();

        let cert2 = CertificateBuilder::new()
            .library(library2)
            .verdict(CertificateVerdict::Safe { proof_core_size: 10 })
            .unsat_proof(make_test_proof())
            .parent_hash(&cert1.content_hash)
            .build()
            .unwrap();

        let chain = CertificateChain::new(vec![cert1, cert2]).unwrap();
        assert_eq!(chain.certificates.len(), 2);

        let vulnerable = chain.vulnerable_libraries();
        assert_eq!(vulnerable.len(), 1);
        assert_eq!(vulnerable[0].name, "openssl");

        let safe = chain.safe_libraries();
        assert_eq!(safe.len(), 1);
        assert_eq!(safe[0].name, "gnutls");
    }

    #[test]
    fn test_certificate_chain_verification() {
        let cert1 = CertificateBuilder::new()
            .library(make_test_library())
            .verdict(CertificateVerdict::Safe { proof_core_size: 5 })
            .unsat_proof(make_test_proof())
            .build()
            .unwrap();

        let chain = CertificateChain::new(vec![cert1]).unwrap();
        let verification = chain.verify().unwrap();
        assert!(verification.is_valid, "Chain should be valid: {:?}", verification);
        assert!(verification.chain_hash_valid);
    }

    #[test]
    fn test_certificate_summary() {
        let cert = CertificateBuilder::new()
            .library(make_test_library())
            .verdict(CertificateVerdict::Vulnerable {
                severity: 3,
                downgrade_from: ProtocolVersion::Tls12,
                downgrade_to: ProtocolVersion::Ssl30,
            })
            .attack_trace(make_test_trace())
            .build()
            .unwrap();

        let summary = cert.summary();
        assert!(summary.contains("CERTIFICATE"));
        assert!(summary.contains("VULNERABLE"));
        assert!(summary.contains("openssl"));
    }

    #[test]
    fn test_extract_unsat_proof() {
        let core = vec!["c1".into(), "c2".into(), "c3".into()];
        let resolution = vec![
            (0usize, 1, "x".to_string(), 3usize),
        ];
        let proof = extract_unsat_proof(core, &resolution);
        assert!(proof.is_valid);
        assert_eq!(proof.core_size(), 3);
        assert_eq!(proof.resolution_steps.len(), 1);
    }

    #[test]
    fn test_verdict_display() {
        let v = CertificateVerdict::Vulnerable {
            severity: 3,
            downgrade_from: ProtocolVersion::Tls12,
            downgrade_to: ProtocolVersion::Ssl30,
        };
        let s = format!("{}", v);
        assert!(s.contains("VULNERABLE"));
        assert!(s.contains("severity=3"));

        let v = CertificateVerdict::Safe { proof_core_size: 42 };
        let s = format!("{}", v);
        assert!(s.contains("SAFE"));
        assert!(s.contains("42"));
    }

    #[test]
    fn test_empty_chain_error() {
        let result = CertificateChain::new(vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_chain_json_roundtrip() {
        let cert = CertificateBuilder::new()
            .library(make_test_library())
            .verdict(CertificateVerdict::Safe { proof_core_size: 5 })
            .unsat_proof(make_test_proof())
            .build()
            .unwrap();

        let chain = CertificateChain::new(vec![cert]).unwrap();
        let json = chain.to_json().unwrap();
        assert!(json.contains("chain"));
    }
}
