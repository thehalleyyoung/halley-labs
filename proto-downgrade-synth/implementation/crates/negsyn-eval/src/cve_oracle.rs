//! CVE oracle testing: ground truth database for known protocol downgrade vulnerabilities.

use crate::{
    AnalysisCertificate, AnalysisPipeline, AttackTrace, CegarResult, Lts, LtsState,
    LtsTransition, PipelineResult, SmtResult,
};
use crate::pipeline::PipelineConfig;

use chrono::Utc;
use log::{debug, info, warn};
use negsyn_types::{HandshakePhase, ProtocolVersion};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::time::Instant;

/// Vulnerability type classification for CVEs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VulnerabilityType {
    CipherDowngrade,
    VersionDowngrade,
    ExportCipherForcing,
    PaddingOracle,
    KeyExchangeWeakness,
    ProtocolConfusion,
    SequenceManipulation,
    SignatureBypass,
}

impl VulnerabilityType {
    pub fn name(&self) -> &'static str {
        match self {
            VulnerabilityType::CipherDowngrade => "cipher_downgrade",
            VulnerabilityType::VersionDowngrade => "version_downgrade",
            VulnerabilityType::ExportCipherForcing => "export_cipher_forcing",
            VulnerabilityType::PaddingOracle => "padding_oracle",
            VulnerabilityType::KeyExchangeWeakness => "key_exchange_weakness",
            VulnerabilityType::ProtocolConfusion => "protocol_confusion",
            VulnerabilityType::SequenceManipulation => "sequence_manipulation",
            VulnerabilityType::SignatureBypass => "signature_bypass",
        }
    }
}

impl fmt::Display for VulnerabilityType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Minimum bounds (k, n) from the spec: k = adversary budget, n = cipher suite count.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MinBounds {
    pub k: u32,
    pub n: u32,
}

impl MinBounds {
    pub fn new(k: u32, n: u32) -> Self {
        Self { k, n }
    }
}

/// A CVE entry describing a known vulnerability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CveEntry {
    pub cve_id: String,
    pub name: String,
    pub description: String,
    pub vulnerability_type: VulnerabilityType,
    pub affected_libraries: Vec<String>,
    pub affected_versions: Vec<AffectedVersion>,
    pub affected_protocol: ProtocolVersion,
    pub weak_cipher_suites: Vec<u16>,
    pub strong_cipher_suites: Vec<u16>,
    pub expected_attack_phases: Vec<HandshakePhase>,
    pub expected_downgrade_from: Option<u16>,
    pub expected_downgrade_to: Option<u16>,
    pub minimum_bounds: MinBounds,
    pub reachability_states: Vec<String>,
    pub severity: CveSeverity,
    pub year: u32,
}

impl CveEntry {
    /// Check if a given library and version is affected.
    pub fn affects(&self, library: &str, version: &str) -> bool {
        let lib_match = self.affected_libraries.iter().any(|l| {
            l.eq_ignore_ascii_case(library)
                || library.to_lowercase().contains(&l.to_lowercase())
        });
        if !lib_match {
            return false;
        }
        self.affected_versions.iter().any(|av| av.matches(version))
    }

    /// Check if an attack trace matches the expected vulnerability pattern.
    pub fn matches_trace(&self, trace: &AttackTrace) -> TraceMatchResult {
        let mut score = 0.0f64;
        let mut reasons = Vec::new();

        if let Some(expected_from) = self.expected_downgrade_from {
            if trace.downgraded_from == expected_from {
                score += 0.3;
                reasons.push("downgrade_from matches".into());
            }
        } else {
            score += 0.15;
        }

        if let Some(expected_to) = self.expected_downgrade_to {
            if trace.downgraded_to == expected_to {
                score += 0.3;
                reasons.push("downgrade_to matches".into());
            } else if self.weak_cipher_suites.contains(&trace.downgraded_to) {
                score += 0.2;
                reasons.push("downgrade_to is a known weak cipher".into());
            }
        } else if self.weak_cipher_suites.contains(&trace.downgraded_to) {
            score += 0.25;
            reasons.push("downgrade_to is in weak cipher set".into());
        }

        if trace.adversary_budget <= self.minimum_bounds.k {
            score += 0.2;
            reasons.push("budget within bounds".into());
        }

        let vuln_type_str = self.vulnerability_type.name();
        if trace.vulnerability_type.contains(vuln_type_str)
            || vuln_type_str.contains(&trace.vulnerability_type)
        {
            score += 0.2;
            reasons.push("vulnerability type matches".into());
        }

        TraceMatchResult {
            score,
            is_match: score >= 0.5,
            reasons,
        }
    }

    /// Validate that a given LTS can reach the vulnerability.
    pub fn validate_reachability(&self, lts: &Lts) -> ReachabilityResult {
        let reachable = lts.reachable_states();
        let all_states: Vec<String> = lts
            .states
            .iter()
            .filter(|s| reachable.contains(&s.id))
            .map(|s| s.label.clone())
            .collect();

        let mut found_phases = Vec::new();
        let mut missing_phases = Vec::new();

        for expected in &self.expected_attack_phases {
            let phase_str = format!("{:?}", expected);
            let found = lts.states.iter().any(|s| {
                s.phase == *expected && reachable.contains(&s.id)
            });
            if found {
                found_phases.push(*expected);
            } else {
                missing_phases.push(*expected);
            }
        }

        let has_downgrade_path = lts.transitions.iter().any(|t| t.is_downgrade);

        let required_found = found_phases.len();
        let required_total = self.expected_attack_phases.len();
        let coverage = if required_total > 0 {
            required_found as f64 / required_total as f64
        } else {
            1.0
        };

        ReachabilityResult {
            is_reachable: coverage >= 0.75 && (has_downgrade_path || missing_phases.is_empty()),
            found_phases,
            missing_phases,
            phase_coverage: coverage,
            has_downgrade_path,
            total_reachable_states: reachable.len(),
        }
    }
}

/// Version range for affected software.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AffectedVersion {
    pub library: String,
    pub min_version: Option<String>,
    pub max_version: Option<String>,
    pub exact_versions: Vec<String>,
}

impl AffectedVersion {
    pub fn matches(&self, version: &str) -> bool {
        if self.exact_versions.contains(&version.to_string()) {
            return true;
        }
        if let Some(ref min) = self.min_version {
            if version_compare(version, min) < 0 {
                return false;
            }
        }
        if let Some(ref max) = self.max_version {
            if version_compare(version, max) > 0 {
                return false;
            }
        }
        self.min_version.is_some() || self.max_version.is_some()
    }
}

fn version_compare(a: &str, b: &str) -> i32 {
    let parse = |s: &str| -> Vec<u64> {
        s.split('.')
            .filter_map(|p| p.trim_start_matches('v').parse::<u64>().ok())
            .collect()
    };
    let va = parse(a);
    let vb = parse(b);
    let max_len = va.len().max(vb.len());
    for i in 0..max_len {
        let pa = va.get(i).copied().unwrap_or(0);
        let pb = vb.get(i).copied().unwrap_or(0);
        if pa < pb {
            return -1;
        }
        if pa > pb {
            return 1;
        }
    }
    0
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CveSeverity {
    Critical,
    High,
    Medium,
    Low,
}

impl fmt::Display for CveSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CveSeverity::Critical => write!(f, "CRITICAL"),
            CveSeverity::High => write!(f, "HIGH"),
            CveSeverity::Medium => write!(f, "MEDIUM"),
            CveSeverity::Low => write!(f, "LOW"),
        }
    }
}

/// Result of comparing a trace against expected vulnerability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceMatchResult {
    pub score: f64,
    pub is_match: bool,
    pub reasons: Vec<String>,
}

/// Result of checking vulnerability reachability in an LTS.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReachabilityResult {
    pub is_reachable: bool,
    pub found_phases: Vec<HandshakePhase>,
    pub missing_phases: Vec<HandshakePhase>,
    pub phase_coverage: f64,
    pub has_downgrade_path: bool,
    pub total_reachable_states: usize,
}

/// Oracle test classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OracleClassification {
    TruePositive,
    FalsePositive,
    TrueNegative,
    FalseNegative,
}

impl fmt::Display for OracleClassification {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OracleClassification::TruePositive => write!(f, "TP"),
            OracleClassification::FalsePositive => write!(f, "FP"),
            OracleClassification::TrueNegative => write!(f, "TN"),
            OracleClassification::FalseNegative => write!(f, "FN"),
        }
    }
}

/// Result of an oracle test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OracleResult {
    pub cve_id: String,
    pub classification: OracleClassification,
    pub expected_vulnerable: bool,
    pub detected_vulnerable: bool,
    pub trace_match: Option<TraceMatchResult>,
    pub reachability: Option<ReachabilityResult>,
    pub bounds_satisfied: bool,
    pub duration_ms: u64,
    pub details: String,
}

impl OracleResult {
    pub fn is_correct(&self) -> bool {
        matches!(
            self.classification,
            OracleClassification::TruePositive | OracleClassification::TrueNegative
        )
    }
}

/// Aggregate oracle results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OracleReport {
    pub results: Vec<OracleResult>,
    pub true_positives: usize,
    pub false_positives: usize,
    pub true_negatives: usize,
    pub false_negatives: usize,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub accuracy: f64,
    pub total_duration_ms: u64,
}

impl OracleReport {
    pub fn from_results(results: Vec<OracleResult>) -> Self {
        let tp = results
            .iter()
            .filter(|r| r.classification == OracleClassification::TruePositive)
            .count();
        let fp = results
            .iter()
            .filter(|r| r.classification == OracleClassification::FalsePositive)
            .count();
        let tn = results
            .iter()
            .filter(|r| r.classification == OracleClassification::TrueNegative)
            .count();
        let fn_ = results
            .iter()
            .filter(|r| r.classification == OracleClassification::FalseNegative)
            .count();

        let precision = if tp + fp > 0 {
            tp as f64 / (tp + fp) as f64
        } else {
            0.0
        };
        let recall = if tp + fn_ > 0 {
            tp as f64 / (tp + fn_) as f64
        } else {
            0.0
        };
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };
        let total = tp + fp + tn + fn_;
        let accuracy = if total > 0 {
            (tp + tn) as f64 / total as f64
        } else {
            0.0
        };
        let total_duration = results.iter().map(|r| r.duration_ms).sum();

        OracleReport {
            results,
            true_positives: tp,
            false_positives: fp,
            true_negatives: tn,
            false_negatives: fn_,
            precision,
            recall,
            f1_score: f1,
            accuracy,
            total_duration_ms: total_duration,
        }
    }
}

/// An oracle test case: run the pipeline and check the result against the CVE entry.
#[derive(Debug, Clone)]
pub struct OracleTest {
    pub cve: CveEntry,
    pub library_name: String,
    pub library_version: String,
    pub should_be_vulnerable: bool,
}

impl OracleTest {
    pub fn new(cve: CveEntry, library: impl Into<String>, version: impl Into<String>) -> Self {
        let lib = library.into();
        let ver = version.into();
        let should_be_vulnerable = cve.affects(&lib, &ver);
        Self {
            cve,
            library_name: lib,
            library_version: ver,
            should_be_vulnerable,
        }
    }

    /// Run the test against a pipeline result.
    pub fn evaluate(&self, result: &PipelineResult) -> OracleResult {
        let start = Instant::now();
        let detected = result.has_vulnerability();

        let trace_match = result.attack_traces.first().map(|t| self.cve.matches_trace(t));

        let reachability = result
            .lts
            .as_ref()
            .map(|lts| self.cve.validate_reachability(lts));

        let bounds_satisfied = self.check_bounds(result);

        let classification = match (self.should_be_vulnerable, detected) {
            (true, true) => {
                let trace_ok = trace_match
                    .as_ref()
                    .map(|tm| tm.is_match)
                    .unwrap_or(false);
                if trace_ok {
                    OracleClassification::TruePositive
                } else {
                    OracleClassification::FalsePositive
                }
            }
            (true, false) => OracleClassification::FalseNegative,
            (false, true) => OracleClassification::FalsePositive,
            (false, false) => OracleClassification::TrueNegative,
        };

        let details = self.build_details(&classification, &trace_match, &reachability);
        let duration = start.elapsed().as_millis() as u64;

        OracleResult {
            cve_id: self.cve.cve_id.clone(),
            classification,
            expected_vulnerable: self.should_be_vulnerable,
            detected_vulnerable: detected,
            trace_match,
            reachability,
            bounds_satisfied,
            duration_ms: duration,
            details,
        }
    }

    fn check_bounds(&self, result: &PipelineResult) -> bool {
        if let Some(ref lts) = result.lts {
            let cipher_count = lts
                .transitions
                .iter()
                .filter_map(|t| t.cipher_suite_id)
                .collect::<std::collections::BTreeSet<_>>()
                .len() as u32;

            cipher_count >= self.cve.minimum_bounds.n
        } else {
            false
        }
    }

    fn build_details(
        &self,
        classification: &OracleClassification,
        trace_match: &Option<TraceMatchResult>,
        reachability: &Option<ReachabilityResult>,
    ) -> String {
        let mut details = format!(
            "CVE: {}, Library: {} v{}, Classification: {}",
            self.cve.cve_id, self.library_name, self.library_version, classification
        );

        if let Some(ref tm) = trace_match {
            details.push_str(&format!(
                "\n  Trace match score: {:.2}, reasons: {}",
                tm.score,
                tm.reasons.join(", ")
            ));
        }

        if let Some(ref reach) = reachability {
            details.push_str(&format!(
                "\n  Reachability: {}, phase coverage: {:.0}%, downgrade path: {}",
                reach.is_reachable,
                reach.phase_coverage * 100.0,
                reach.has_downgrade_path
            ));
            if !reach.missing_phases.is_empty() {
                details.push_str(&format!(
                    "\n  Missing phases: {:?}",
                    reach.missing_phases
                ));
            }
        }

        details
    }
}

/// The CVE oracle: ground truth database for known vulnerabilities.
pub struct CveOracle {
    entries: Vec<CveEntry>,
    entries_by_id: HashMap<String, usize>,
}

impl CveOracle {
    /// Create a new oracle with the built-in CVE database.
    pub fn new() -> Self {
        let entries = Self::build_database();
        let entries_by_id: HashMap<String, usize> = entries
            .iter()
            .enumerate()
            .map(|(i, e)| (e.cve_id.clone(), i))
            .collect();
        Self {
            entries,
            entries_by_id,
        }
    }

    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    pub fn get_entry(&self, cve_id: &str) -> Option<&CveEntry> {
        self.entries_by_id.get(cve_id).map(|&i| &self.entries[i])
    }

    pub fn entries(&self) -> &[CveEntry] {
        &self.entries
    }

    pub fn entries_by_type(&self, vtype: VulnerabilityType) -> Vec<&CveEntry> {
        self.entries
            .iter()
            .filter(|e| e.vulnerability_type == vtype)
            .collect()
    }

    pub fn entries_for_library(&self, library: &str) -> Vec<&CveEntry> {
        self.entries
            .iter()
            .filter(|e| {
                e.affected_libraries
                    .iter()
                    .any(|l| l.eq_ignore_ascii_case(library))
            })
            .collect()
    }

    /// Run oracle tests for a specific library against all CVEs.
    pub fn test_library(
        &self,
        library: &str,
        version: &str,
        result: &PipelineResult,
    ) -> OracleReport {
        let mut results = Vec::new();
        for entry in &self.entries {
            let test = OracleTest::new(entry.clone(), library, version);
            let oracle_result = test.evaluate(result);
            results.push(oracle_result);
        }
        OracleReport::from_results(results)
    }

    /// Run oracle tests for a specific CVE across multiple results.
    pub fn test_cve(
        &self,
        cve_id: &str,
        library_results: &[(String, String, PipelineResult)],
    ) -> Vec<OracleResult> {
        let entry = match self.get_entry(cve_id) {
            Some(e) => e,
            None => return Vec::new(),
        };

        library_results
            .iter()
            .map(|(lib, ver, result)| {
                let test = OracleTest::new(entry.clone(), lib.as_str(), ver.as_str());
                test.evaluate(result)
            })
            .collect()
    }

    /// Build the full CVE database.
    fn build_database() -> Vec<CveEntry> {
        vec![
            Self::cve_freak(),
            Self::cve_logjam(),
            Self::cve_poodle(),
            Self::cve_terrapin(),
            Self::cve_drown(),
            Self::cve_ccs_injection(),
            Self::cve_sslv2_override(),
            Self::cve_sigalgs_dos(),
        ]
    }

    fn cve_freak() -> CveEntry {
        CveEntry {
            cve_id: "CVE-2015-0204".into(),
            name: "FREAK".into(),
            description: "Factoring RSA Export Keys: MITM attacker can force use of \
                export-grade RSA keys in TLS connections"
                .into(),
            vulnerability_type: VulnerabilityType::ExportCipherForcing,
            affected_libraries: vec![
                "openssl".into(),
                "boringssl".into(),
                "apple-securetransport".into(),
            ],
            affected_versions: vec![
                AffectedVersion {
                    library: "openssl".into(),
                    min_version: Some("0.9.8".into()),
                    max_version: Some("1.0.1k".into()),
                    exact_versions: vec![],
                },
            ],
            affected_protocol: ProtocolVersion::tls12(),
            weak_cipher_suites: vec![0x0003, 0x0006, 0x0008, 0x000B, 0x000E, 0x0011, 0x0014],
            strong_cipher_suites: vec![0x002F, 0x0035, 0x009C, 0x009D],
            expected_attack_phases: vec![
                HandshakePhase::ClientHelloSent,
                HandshakePhase::ServerHelloReceived,
                HandshakePhase::Negotiated,
            ],
            expected_downgrade_from: Some(0x002F),
            expected_downgrade_to: Some(0x0003),
            minimum_bounds: MinBounds::new(2, 4),
            reachability_states: vec![
                "client_hello_with_export".into(),
                "server_accepts_export".into(),
                "weak_key_exchange".into(),
            ],
            severity: CveSeverity::High,
            year: 2015,
        }
    }

    fn cve_logjam() -> CveEntry {
        CveEntry {
            cve_id: "CVE-2015-4000".into(),
            name: "Logjam".into(),
            description: "MITM attacker can downgrade TLS connections to use 512-bit \
                export-grade Diffie-Hellman key exchange"
                .into(),
            vulnerability_type: VulnerabilityType::KeyExchangeWeakness,
            affected_libraries: vec![
                "openssl".into(),
                "gnutls".into(),
                "nss".into(),
                "boringssl".into(),
            ],
            affected_versions: vec![
                AffectedVersion {
                    library: "openssl".into(),
                    min_version: Some("0.9.8".into()),
                    max_version: Some("1.0.2a".into()),
                    exact_versions: vec![],
                },
            ],
            affected_protocol: ProtocolVersion::tls12(),
            weak_cipher_suites: vec![0x0011, 0x0014, 0x0017, 0x0019],
            strong_cipher_suites: vec![0x009E, 0x009F, 0xC02B, 0xC02F],
            expected_attack_phases: vec![
                HandshakePhase::ClientHelloSent,
                HandshakePhase::ServerHelloReceived,
                HandshakePhase::Negotiated,
                HandshakePhase::Negotiated,
            ],
            expected_downgrade_from: Some(0x009E),
            expected_downgrade_to: Some(0x0011),
            minimum_bounds: MinBounds::new(2, 6),
            reachability_states: vec![
                "client_offers_dhe".into(),
                "server_selects_export_dhe".into(),
                "weak_dh_params".into(),
            ],
            severity: CveSeverity::High,
            year: 2015,
        }
    }

    fn cve_poodle() -> CveEntry {
        CveEntry {
            cve_id: "CVE-2014-3566".into(),
            name: "POODLE".into(),
            description: "Padding Oracle On Downgraded Legacy Encryption: \
                exploits SSL 3.0 CBC padding to decrypt traffic"
                .into(),
            vulnerability_type: VulnerabilityType::PaddingOracle,
            affected_libraries: vec![
                "openssl".into(),
                "gnutls".into(),
                "nss".into(),
                "mbedtls".into(),
            ],
            affected_versions: vec![
                AffectedVersion {
                    library: "openssl".into(),
                    min_version: Some("0.9.8".into()),
                    max_version: Some("1.0.1j".into()),
                    exact_versions: vec![],
                },
            ],
            affected_protocol: ProtocolVersion::ssl30(),
            weak_cipher_suites: vec![0x0004, 0x0005, 0x000A, 0x002F, 0x0035],
            strong_cipher_suites: vec![0x009C, 0x009D, 0xC02B, 0xC02F],
            expected_attack_phases: vec![
                HandshakePhase::ClientHelloSent,
                HandshakePhase::ServerHelloReceived,
                HandshakePhase::Negotiated,
                HandshakePhase::Done,
            ],
            expected_downgrade_from: Some(0x009C),
            expected_downgrade_to: Some(0x0005),
            minimum_bounds: MinBounds::new(1, 3),
            reachability_states: vec![
                "fallback_to_ssl3".into(),
                "cbc_mode_selected".into(),
                "padding_exploitable".into(),
            ],
            severity: CveSeverity::Medium,
            year: 2014,
        }
    }

    fn cve_terrapin() -> CveEntry {
        CveEntry {
            cve_id: "CVE-2023-48795".into(),
            name: "Terrapin".into(),
            description: "SSH prefix truncation attack that manipulates handshake \
                sequence numbers to downgrade connection security"
                .into(),
            vulnerability_type: VulnerabilityType::SequenceManipulation,
            affected_libraries: vec![
                "openssh".into(),
                "libssh".into(),
                "paramiko".into(),
                "putty".into(),
                "golang-ssh".into(),
            ],
            affected_versions: vec![
                AffectedVersion {
                    library: "openssh".into(),
                    min_version: Some("2.0".into()),
                    max_version: Some("9.5".into()),
                    exact_versions: vec![],
                },
                AffectedVersion {
                    library: "libssh".into(),
                    min_version: Some("0.9.0".into()),
                    max_version: Some("0.10.5".into()),
                    exact_versions: vec![],
                },
            ],
            affected_protocol: ProtocolVersion::ssh2(),
            weak_cipher_suites: vec![],
            strong_cipher_suites: vec![],
            expected_attack_phases: vec![
                HandshakePhase::Init,
                HandshakePhase::Negotiated,
                HandshakePhase::Negotiated,
            ],
            expected_downgrade_from: None,
            expected_downgrade_to: None,
            minimum_bounds: MinBounds::new(1, 2),
            reachability_states: vec![
                "binary_packet_protocol".into(),
                "sequence_number_manipulation".into(),
                "extension_negotiation_truncated".into(),
            ],
            severity: CveSeverity::Medium,
            year: 2023,
        }
    }

    fn cve_drown() -> CveEntry {
        CveEntry {
            cve_id: "CVE-2016-0800".into(),
            name: "DROWN".into(),
            description: "Decrypting RSA with Obsolete and Weakened eNcryption: \
                cross-protocol attack using SSLv2 to decrypt TLS sessions"
                .into(),
            vulnerability_type: VulnerabilityType::ProtocolConfusion,
            affected_libraries: vec!["openssl".into(), "nss".into()],
            affected_versions: vec![AffectedVersion {
                library: "openssl".into(),
                min_version: Some("0.9.8".into()),
                max_version: Some("1.0.2f".into()),
                exact_versions: vec![],
            }],
            affected_protocol: ProtocolVersion::tls12(),
            weak_cipher_suites: vec![0x0001, 0x0002, 0x0003, 0x0004, 0x0005, 0x0006],
            strong_cipher_suites: vec![0x002F, 0x0035, 0x009C],
            expected_attack_phases: vec![
                HandshakePhase::ClientHelloSent,
                HandshakePhase::ServerHelloReceived,
                HandshakePhase::Negotiated,
                HandshakePhase::Negotiated,
            ],
            expected_downgrade_from: Some(0x009C),
            expected_downgrade_to: Some(0x0002),
            minimum_bounds: MinBounds::new(3, 5),
            reachability_states: vec![
                "sslv2_enabled".into(),
                "shared_rsa_key".into(),
                "bleichenbacher_oracle".into(),
            ],
            severity: CveSeverity::Critical,
            year: 2016,
        }
    }

    fn cve_ccs_injection() -> CveEntry {
        CveEntry {
            cve_id: "CVE-2014-0224".into(),
            name: "CCS Injection".into(),
            description: "ChangeCipherSpec message processed before key material \
                is established, allowing MITM to force use of weak keys"
                .into(),
            vulnerability_type: VulnerabilityType::CipherDowngrade,
            affected_libraries: vec!["openssl".into()],
            affected_versions: vec![AffectedVersion {
                library: "openssl".into(),
                min_version: Some("0.9.8".into()),
                max_version: Some("1.0.1g".into()),
                exact_versions: vec![],
            }],
            affected_protocol: ProtocolVersion::tls12(),
            weak_cipher_suites: vec![0x0001, 0x0002, 0x0003],
            strong_cipher_suites: vec![0x002F, 0x0035, 0x009C],
            expected_attack_phases: vec![
                HandshakePhase::ClientHelloSent,
                HandshakePhase::ServerHelloReceived,
                HandshakePhase::Negotiated,
            ],
            expected_downgrade_from: Some(0x002F),
            expected_downgrade_to: Some(0x0001),
            minimum_bounds: MinBounds::new(1, 3),
            reachability_states: vec![
                "early_ccs_accepted".into(),
                "null_key_material".into(),
                "weak_encryption_active".into(),
            ],
            severity: CveSeverity::High,
            year: 2014,
        }
    }

    fn cve_sslv2_override() -> CveEntry {
        CveEntry {
            cve_id: "CVE-2021-3449".into(),
            name: "SSLv2 Override".into(),
            description: "Server crash via malformed renegotiation ClientHello that \
                forces SSLv2 protocol selection despite being disabled"
                .into(),
            vulnerability_type: VulnerabilityType::VersionDowngrade,
            affected_libraries: vec!["openssl".into()],
            affected_versions: vec![AffectedVersion {
                library: "openssl".into(),
                min_version: None,
                max_version: None,
                exact_versions: vec!["1.1.1j".into(), "1.1.1k".into()],
            }],
            affected_protocol: ProtocolVersion::tls12(),
            weak_cipher_suites: vec![],
            strong_cipher_suites: vec![0x002F, 0x0035],
            expected_attack_phases: vec![
                HandshakePhase::ClientHelloSent,
                HandshakePhase::Init,
            ],
            expected_downgrade_from: None,
            expected_downgrade_to: None,
            minimum_bounds: MinBounds::new(1, 2),
            reachability_states: vec![
                "renegotiation_initiated".into(),
                "malformed_client_hello".into(),
                "version_override".into(),
            ],
            severity: CveSeverity::High,
            year: 2021,
        }
    }

    fn cve_sigalgs_dos() -> CveEntry {
        CveEntry {
            cve_id: "CVE-2022-4450".into(),
            name: "SigAlgs DoS".into(),
            description: "Double free vulnerability triggered by crafted signature_algorithms \
                extension that can force algorithm downgrade"
                .into(),
            vulnerability_type: VulnerabilityType::SignatureBypass,
            affected_libraries: vec!["openssl".into()],
            affected_versions: vec![AffectedVersion {
                library: "openssl".into(),
                min_version: Some("3.0.0".into()),
                max_version: Some("3.0.7".into()),
                exact_versions: vec![],
            }],
            affected_protocol: ProtocolVersion::tls13(),
            weak_cipher_suites: vec![],
            strong_cipher_suites: vec![0x1301, 0x1302, 0x1303],
            expected_attack_phases: vec![
                HandshakePhase::ClientHelloSent,
                HandshakePhase::Negotiated,
            ],
            expected_downgrade_from: None,
            expected_downgrade_to: None,
            minimum_bounds: MinBounds::new(1, 1),
            reachability_states: vec![
                "sigalgs_extension_parsed".into(),
                "double_free_triggered".into(),
            ],
            severity: CveSeverity::High,
            year: 2022,
        }
    }
}

impl Default for CveOracle {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::PipelineConfig;

    #[test]
    fn test_oracle_database_size() {
        let oracle = CveOracle::new();
        assert_eq!(oracle.entry_count(), 8);
    }

    #[test]
    fn test_oracle_lookup_by_id() {
        let oracle = CveOracle::new();
        let freak = oracle.get_entry("CVE-2015-0204");
        assert!(freak.is_some());
        assert_eq!(freak.unwrap().name, "FREAK");
    }

    #[test]
    fn test_oracle_lookup_by_type() {
        let oracle = CveOracle::new();
        let padding = oracle.entries_by_type(VulnerabilityType::PaddingOracle);
        assert_eq!(padding.len(), 1);
        assert_eq!(padding[0].cve_id, "CVE-2014-3566");
    }

    #[test]
    fn test_oracle_library_filter() {
        let oracle = CveOracle::new();
        let openssl = oracle.entries_for_library("openssl");
        assert!(openssl.len() >= 6);
    }

    #[test]
    fn test_cve_affects() {
        let oracle = CveOracle::new();
        let freak = oracle.get_entry("CVE-2015-0204").unwrap();
        assert!(freak.affects("openssl", "1.0.1a"));
        assert!(!freak.affects("openssl", "1.0.2a"));
        assert!(!freak.affects("unknownlib", "1.0.0"));
    }

    #[test]
    fn test_version_compare() {
        assert_eq!(version_compare("1.0.0", "1.0.0"), 0);
        assert_eq!(version_compare("1.0.1", "1.0.0"), 1);
        assert_eq!(version_compare("1.0.0", "1.0.1"), -1);
        assert_eq!(version_compare("2.0", "1.9.9"), 1);
        assert_eq!(version_compare("1.0", "1.0.0"), 0);
    }

    #[test]
    fn test_affected_version_exact() {
        let av = AffectedVersion {
            library: "openssl".into(),
            min_version: None,
            max_version: None,
            exact_versions: vec!["1.1.1j".into()],
        };
        assert!(av.matches("1.1.1j"));
        assert!(!av.matches("1.1.1k"));
    }

    #[test]
    fn test_affected_version_range() {
        let av = AffectedVersion {
            library: "openssl".into(),
            min_version: Some("1.0.0".into()),
            max_version: Some("1.0.2".into()),
            exact_versions: vec![],
        };
        assert!(av.matches("1.0.1"));
        assert!(!av.matches("1.0.3"));
        assert!(!av.matches("0.9.8"));
    }

    #[test]
    fn test_trace_match_scoring() {
        let oracle = CveOracle::new();
        let freak = oracle.get_entry("CVE-2015-0204").unwrap();

        let trace = crate::AttackTrace {
            steps: vec![],
            downgraded_from: 0x002F,
            downgraded_to: 0x0003,
            adversary_budget: 2,
            vulnerability_type: "export_cipher_forcing".into(),
        };
        let result = freak.matches_trace(&trace);
        assert!(result.is_match);
        assert!(result.score >= 0.8);
    }

    #[test]
    fn test_trace_match_no_match() {
        let oracle = CveOracle::new();
        let freak = oracle.get_entry("CVE-2015-0204").unwrap();

        let trace = crate::AttackTrace {
            steps: vec![],
            downgraded_from: 0xFFFF,
            downgraded_to: 0xFFFE,
            adversary_budget: 100,
            vulnerability_type: "unrelated".into(),
        };
        let result = freak.matches_trace(&trace);
        assert!(!result.is_match);
    }

    #[test]
    fn test_oracle_result_classification() {
        let r = OracleResult {
            cve_id: "CVE-TEST".into(),
            classification: OracleClassification::TruePositive,
            expected_vulnerable: true,
            detected_vulnerable: true,
            trace_match: None,
            reachability: None,
            bounds_satisfied: true,
            duration_ms: 10,
            details: String::new(),
        };
        assert!(r.is_correct());

        let r2 = OracleResult {
            classification: OracleClassification::FalseNegative,
            ..r
        };
        assert!(!r2.is_correct());
    }

    #[test]
    fn test_oracle_report_metrics() {
        let results = vec![
            OracleResult {
                cve_id: "CVE-1".into(),
                classification: OracleClassification::TruePositive,
                expected_vulnerable: true,
                detected_vulnerable: true,
                trace_match: None,
                reachability: None,
                bounds_satisfied: true,
                duration_ms: 10,
                details: String::new(),
            },
            OracleResult {
                cve_id: "CVE-2".into(),
                classification: OracleClassification::TrueNegative,
                expected_vulnerable: false,
                detected_vulnerable: false,
                trace_match: None,
                reachability: None,
                bounds_satisfied: false,
                duration_ms: 5,
                details: String::new(),
            },
            OracleResult {
                cve_id: "CVE-3".into(),
                classification: OracleClassification::FalseNegative,
                expected_vulnerable: true,
                detected_vulnerable: false,
                trace_match: None,
                reachability: None,
                bounds_satisfied: false,
                duration_ms: 8,
                details: String::new(),
            },
        ];

        let report = OracleReport::from_results(results);
        assert_eq!(report.true_positives, 1);
        assert_eq!(report.true_negatives, 1);
        assert_eq!(report.false_negatives, 1);
        assert_eq!(report.false_positives, 0);
        assert!((report.precision - 1.0).abs() < 0.01);
        assert!((report.recall - 0.5).abs() < 0.01);
        assert!((report.accuracy - 2.0 / 3.0).abs() < 0.01);
        assert_eq!(report.total_duration_ms, 23);
    }

    #[test]
    fn test_reachability_validation() {
        let oracle = CveOracle::new();
        let ccs = oracle.get_entry("CVE-2014-0224").unwrap();

        let mut lts = Lts::new("openssl");
        lts.add_state(LtsState::new(0, "initial", HandshakePhase::Init));
        let mut s1 = LtsState::new(1, "ClientHello", HandshakePhase::ClientHelloSent);
        lts.add_state(s1);
        let mut s2 = LtsState::new(2, "ServerHello", HandshakePhase::ServerHelloReceived);
        lts.add_state(s2);
        let mut s3 = LtsState::new(3, "ChangeCipherSpec", HandshakePhase::Negotiated);
        lts.add_state(s3);

        lts.add_transition(LtsTransition::new(0, 0, 1, "init"));
        lts.add_transition(LtsTransition::new(1, 1, 2, "hello"));
        let mut t = LtsTransition::new(2, 2, 3, "ccs");
        t.is_downgrade = true;
        lts.add_transition(t);

        let reach = ccs.validate_reachability(&lts);
        assert!(reach.is_reachable);
        assert!(reach.has_downgrade_path);
        assert!(reach.phase_coverage >= 0.66);
    }

    #[test]
    fn test_vulnerability_type_names() {
        assert_eq!(
            VulnerabilityType::CipherDowngrade.name(),
            "cipher_downgrade"
        );
        assert_eq!(
            VulnerabilityType::SequenceManipulation.name(),
            "sequence_manipulation"
        );
    }

    #[test]
    fn test_oracle_evaluate_with_pipeline_result() {
        let oracle = CveOracle::new();
        let ccs = oracle.get_entry("CVE-2014-0224").unwrap();
        let test = OracleTest::new(ccs.clone(), "openssl", "1.0.1a");
        assert!(test.should_be_vulnerable);

        let result = PipelineResult::new("openssl");
        let oracle_result = test.evaluate(&result);
        assert_eq!(oracle_result.classification, OracleClassification::FalseNegative);
    }

    #[test]
    fn test_min_bounds() {
        let bounds = MinBounds::new(3, 5);
        assert_eq!(bounds.k, 3);
        assert_eq!(bounds.n, 5);
    }

    #[test]
    fn test_severity_display() {
        assert_eq!(format!("{}", CveSeverity::Critical), "CRITICAL");
        assert_eq!(format!("{}", CveSeverity::Low), "LOW");
    }
}
