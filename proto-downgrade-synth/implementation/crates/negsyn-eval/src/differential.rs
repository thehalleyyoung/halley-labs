//! Cross-library differential analysis for protocol negotiation state machines.

use crate::{AnalysisCertificate, Lts, LtsState, LtsTransition};

use chrono::Utc;
use itertools::Itertools;
use log::{debug, info, warn};
use negsyn_types::{HandshakePhase, ProtocolVersion};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use uuid::Uuid;

/// Aligns protocol outputs across libraries using IANA cipher suite IDs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireProtocolAlignment {
    pub cipher_id: u16,
    pub cipher_name: String,
    pub library_behaviors: BTreeMap<String, CipherBehavior>,
}

/// How a specific library handles a given cipher suite.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CipherBehavior {
    pub supported: bool,
    pub offered_by_default: bool,
    pub priority_rank: Option<u32>,
    pub requires_extensions: Vec<u16>,
    pub negotiation_outcome: NegotiationOutcome,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NegotiationOutcome {
    Accepted,
    Rejected,
    FallbackToWeaker(u16),
    Error(String),
    NotTested,
}

impl std::fmt::Display for NegotiationOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NegotiationOutcome::Accepted => write!(f, "ACCEPTED"),
            NegotiationOutcome::Rejected => write!(f, "REJECTED"),
            NegotiationOutcome::FallbackToWeaker(id) => write!(f, "FALLBACK({:#06x})", id),
            NegotiationOutcome::Error(e) => write!(f, "ERROR({})", e),
            NegotiationOutcome::NotTested => write!(f, "NOT_TESTED"),
        }
    }
}

/// A behavioral deviation between two libraries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehavioralDeviation {
    pub id: String,
    pub library_a: String,
    pub library_b: String,
    pub deviation_type: DeviationType,
    pub description: String,
    pub cipher_suite_id: Option<u16>,
    pub state_a: Option<u32>,
    pub state_b: Option<u32>,
    pub phase: Option<HandshakePhase>,
    pub security_impact: SecurityImpact,
    pub confidence: f64,
    pub evidence: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DeviationType {
    CipherSuiteAcceptance,
    VersionNegotiation,
    ExtensionHandling,
    TransitionOrdering,
    ErrorHandling,
    FallbackBehavior,
    StateReachability,
    TimingDifference,
}

impl std::fmt::Display for DeviationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeviationType::CipherSuiteAcceptance => write!(f, "cipher_suite_acceptance"),
            DeviationType::VersionNegotiation => write!(f, "version_negotiation"),
            DeviationType::ExtensionHandling => write!(f, "extension_handling"),
            DeviationType::TransitionOrdering => write!(f, "transition_ordering"),
            DeviationType::ErrorHandling => write!(f, "error_handling"),
            DeviationType::FallbackBehavior => write!(f, "fallback_behavior"),
            DeviationType::StateReachability => write!(f, "state_reachability"),
            DeviationType::TimingDifference => write!(f, "timing_difference"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum SecurityImpact {
    None,
    Low,
    Medium,
    High,
    Critical,
}

impl SecurityImpact {
    pub fn score(&self) -> f64 {
        match self {
            SecurityImpact::None => 0.0,
            SecurityImpact::Low => 2.5,
            SecurityImpact::Medium => 5.0,
            SecurityImpact::High => 7.5,
            SecurityImpact::Critical => 10.0,
        }
    }
}

impl std::fmt::Display for SecurityImpact {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SecurityImpact::None => write!(f, "NONE"),
            SecurityImpact::Low => write!(f, "LOW"),
            SecurityImpact::Medium => write!(f, "MEDIUM"),
            SecurityImpact::High => write!(f, "HIGH"),
            SecurityImpact::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// Ranks deviations by security impact.
pub struct DeviationRanker {
    weight_cipher_downgrade: f64,
    weight_version_downgrade: f64,
    weight_fallback: f64,
    weight_error_diff: f64,
    deprecated_ciphers: BTreeSet<u16>,
    deprecated_versions: BTreeSet<ProtocolVersion>,
}

impl DeviationRanker {
    pub fn new() -> Self {
        Self {
            weight_cipher_downgrade: 10.0,
            weight_version_downgrade: 9.0,
            weight_fallback: 8.0,
            weight_error_diff: 5.0,
            deprecated_ciphers: [
                0x0001, 0x0002, 0x0003, 0x0004, 0x0005, 0x0006, 0x0007, 0x0008, 0x0009, 0x000A,
                0x002C, 0x002D, 0x002E, 0x0060, 0x0061, 0x0062, 0x0063, 0x0064,
            ]
            .iter()
            .copied()
            .collect(),
            deprecated_versions: [ProtocolVersion::ssl30(), ProtocolVersion::tls10(), ProtocolVersion::tls11()]
                .iter()
                .cloned()
                .collect(),
        }
    }

    pub fn rank(&self, deviations: &mut [BehavioralDeviation]) {
        for dev in deviations.iter_mut() {
            dev.security_impact = self.classify_impact(dev);
            dev.confidence = self.compute_confidence(dev);
        }
        deviations.sort_by(|a, b| {
            let score_a = a.security_impact.score() * a.confidence;
            let score_b = b.security_impact.score() * b.confidence;
            score_b
                .partial_cmp(&score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    fn classify_impact(&self, dev: &BehavioralDeviation) -> SecurityImpact {
        match dev.deviation_type {
            DeviationType::CipherSuiteAcceptance => {
                if let Some(cipher_id) = dev.cipher_suite_id {
                    if self.deprecated_ciphers.contains(&cipher_id) {
                        return SecurityImpact::Critical;
                    }
                    if cipher_id < 0x0030 {
                        return SecurityImpact::High;
                    }
                }
                SecurityImpact::Medium
            }
            DeviationType::VersionNegotiation => SecurityImpact::High,
            DeviationType::FallbackBehavior => {
                if let Some(cipher_id) = dev.cipher_suite_id {
                    if self.deprecated_ciphers.contains(&cipher_id) {
                        return SecurityImpact::Critical;
                    }
                }
                SecurityImpact::High
            }
            DeviationType::ErrorHandling => SecurityImpact::Medium,
            DeviationType::TransitionOrdering => SecurityImpact::Medium,
            DeviationType::ExtensionHandling => SecurityImpact::Low,
            DeviationType::StateReachability => SecurityImpact::Low,
            DeviationType::TimingDifference => SecurityImpact::None,
        }
    }

    fn compute_confidence(&self, dev: &BehavioralDeviation) -> f64 {
        let base = match dev.evidence.len() {
            0 => 0.3,
            1 => 0.5,
            2 => 0.7,
            3 => 0.85,
            _ => 0.95,
        };

        let type_bonus = match dev.deviation_type {
            DeviationType::CipherSuiteAcceptance => 0.05,
            DeviationType::FallbackBehavior => 0.05,
            _ => 0.0,
        };

        let score: f64 = base + type_bonus;
        score.min(1.0)
    }

    pub fn is_security_relevant(&self, dev: &BehavioralDeviation) -> bool {
        dev.security_impact >= SecurityImpact::Medium
            || (dev.security_impact >= SecurityImpact::Low && dev.confidence >= 0.8)
    }
}

impl Default for DeviationRanker {
    fn default() -> Self {
        Self::new()
    }
}

/// Generates test scenarios using covering designs for systematic differential testing.
pub struct CoveringDesignGenerator {
    strength: usize,
    cipher_suites: Vec<u16>,
    versions: Vec<ProtocolVersion>,
    extensions: Vec<u16>,
}

impl CoveringDesignGenerator {
    pub fn new(
        strength: usize,
        cipher_suites: Vec<u16>,
        versions: Vec<ProtocolVersion>,
        extensions: Vec<u16>,
    ) -> Self {
        Self {
            strength,
            cipher_suites,
            versions,
            extensions,
        }
    }

    /// Generate covering array rows that achieve t-way coverage.
    pub fn generate(&self) -> Vec<CoveringDesignRow> {
        let mut rows = Vec::new();

        if self.strength <= 1 {
            rows.extend(self.generate_1_way());
        } else {
            rows.extend(self.generate_pairwise());
        }

        rows.extend(self.generate_boundary_cases());
        self.deduplicate(&mut rows);
        rows
    }

    fn generate_1_way(&self) -> Vec<CoveringDesignRow> {
        let max_len = self
            .cipher_suites
            .len()
            .max(self.versions.len())
            .max(self.extensions.len())
            .max(1);

        (0..max_len)
            .map(|i| CoveringDesignRow {
                cipher_suite: self.cipher_suites.get(i % self.cipher_suites.len().max(1)).copied(),
                version: self.versions.get(i % self.versions.len().max(1)).cloned(),
                extension_ids: if !self.extensions.is_empty() {
                    vec![self.extensions[i % self.extensions.len()]]
                } else {
                    vec![]
                },
                is_boundary: false,
            })
            .collect()
    }

    fn generate_pairwise(&self) -> Vec<CoveringDesignRow> {
        let mut rows = Vec::new();
        let mut covered_pairs: HashSet<(Option<u16>, Option<ProtocolVersion>)> = HashSet::new();

        for &cipher in &self.cipher_suites {
            for &version in &self.versions {
                let pair = (Some(cipher), Some(version));
                if covered_pairs.insert(pair) {
                    let ext_ids = if !self.extensions.is_empty() {
                        vec![self.extensions[rows.len() % self.extensions.len()]]
                    } else {
                        vec![]
                    };
                    rows.push(CoveringDesignRow {
                        cipher_suite: Some(cipher),
                        version: Some(version),
                        extension_ids: ext_ids,
                        is_boundary: false,
                    });
                }
            }
        }

        if self.strength >= 3 && !self.extensions.is_empty() {
            for &cipher in &self.cipher_suites {
                for &ext in &self.extensions {
                    let version = self.versions.first().cloned();
                    rows.push(CoveringDesignRow {
                        cipher_suite: Some(cipher),
                        version,
                        extension_ids: vec![ext],
                        is_boundary: false,
                    });
                }
            }
        }

        rows
    }

    fn generate_boundary_cases(&self) -> Vec<CoveringDesignRow> {
        let mut cases = Vec::new();

        if let Some(&first) = self.cipher_suites.first() {
            cases.push(CoveringDesignRow {
                cipher_suite: Some(first),
                version: self.versions.first().cloned(),
                extension_ids: vec![],
                is_boundary: true,
            });
        }

        if let Some(&last) = self.cipher_suites.last() {
            cases.push(CoveringDesignRow {
                cipher_suite: Some(last),
                version: self.versions.last().cloned(),
                extension_ids: self.extensions.clone(),
                is_boundary: true,
            });
        }

        cases.push(CoveringDesignRow {
            cipher_suite: None,
            version: None,
            extension_ids: vec![],
            is_boundary: true,
        });

        cases
    }

    fn deduplicate(&self, rows: &mut Vec<CoveringDesignRow>) {
        let mut seen = HashSet::new();
        rows.retain(|r| {
            let key = format!("{:?}|{:?}|{:?}", r.cipher_suite, r.version, r.extension_ids);
            seen.insert(key)
        });
    }
}

/// A row in a covering design.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoveringDesignRow {
    pub cipher_suite: Option<u16>,
    pub version: Option<ProtocolVersion>,
    pub extension_ids: Vec<u16>,
    pub is_boundary: bool,
}

/// Certificate attesting cross-library interoperability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossLibraryCertificate {
    pub id: String,
    pub libraries: Vec<String>,
    pub timestamp: String,
    pub total_scenarios: usize,
    pub deviations_found: usize,
    pub security_deviations: usize,
    pub benign_deviations: usize,
    pub cipher_suites_tested: usize,
    pub versions_tested: usize,
    pub interop_score: f64,
    pub hash: String,
}

/// The main differential analyzer.
pub struct DifferentialAnalyzer {
    library_lts: BTreeMap<String, Lts>,
    ranker: DeviationRanker,
    alignments: Vec<WireProtocolAlignment>,
    deviations: Vec<BehavioralDeviation>,
}

impl DifferentialAnalyzer {
    pub fn new() -> Self {
        Self {
            library_lts: BTreeMap::new(),
            ranker: DeviationRanker::new(),
            alignments: Vec::new(),
            deviations: Vec::new(),
        }
    }

    pub fn add_library(&mut self, name: impl Into<String>, lts: Lts) {
        self.library_lts.insert(name.into(), lts);
    }

    pub fn library_count(&self) -> usize {
        self.library_lts.len()
    }

    /// Run full differential analysis across all loaded libraries.
    pub fn analyze(&mut self) -> DifferentialResult {
        info!(
            "Starting differential analysis across {} libraries",
            self.library_lts.len()
        );

        self.alignments = self.compute_alignments();
        self.deviations = self.find_deviations();
        self.ranker.rank(&mut self.deviations);

        let security_devs: Vec<&BehavioralDeviation> = self
            .deviations
            .iter()
            .filter(|d| self.ranker.is_security_relevant(d))
            .collect();

        let benign_devs = self.deviations.len() - security_devs.len();

        let cert = self.generate_certificate(security_devs.len(), benign_devs);

        DifferentialResult {
            libraries: self.library_lts.keys().cloned().collect(),
            alignments: self.alignments.clone(),
            deviations: self.deviations.clone(),
            security_deviations: security_devs.len(),
            benign_deviations: benign_devs,
            certificate: cert,
        }
    }

    fn compute_alignments(&self) -> Vec<WireProtocolAlignment> {
        let mut all_ciphers: BTreeSet<u16> = BTreeSet::new();

        for lts in self.library_lts.values() {
            for trans in &lts.transitions {
                if let Some(id) = trans.cipher_suite_id {
                    all_ciphers.insert(id);
                }
            }
        }

        let mut alignments = Vec::new();

        for &cipher_id in &all_ciphers {
            let cipher_name = iana_cipher_name(cipher_id);
            let mut behaviors = BTreeMap::new();

            for (lib_name, lts) in &self.library_lts {
                let transitions_with_cipher: Vec<&LtsTransition> = lts
                    .transitions
                    .iter()
                    .filter(|t| t.cipher_suite_id == Some(cipher_id))
                    .collect();

                let supported = !transitions_with_cipher.is_empty();
                let has_downgrade = transitions_with_cipher.iter().any(|t| t.is_downgrade);

                let outcome = if supported {
                    if has_downgrade {
                        NegotiationOutcome::FallbackToWeaker(cipher_id)
                    } else {
                        NegotiationOutcome::Accepted
                    }
                } else {
                    NegotiationOutcome::Rejected
                };

                let rank = if supported {
                    let accepting_targets: Vec<u32> = transitions_with_cipher
                        .iter()
                        .filter_map(|t| {
                            lts.get_state(t.target)
                                .filter(|s| s.is_accepting)
                                .map(|s| s.id)
                        })
                        .collect();
                    if accepting_targets.is_empty() {
                        None
                    } else {
                        Some(accepting_targets[0])
                    }
                } else {
                    None
                };

                behaviors.insert(
                    lib_name.clone(),
                    CipherBehavior {
                        supported,
                        offered_by_default: supported,
                        priority_rank: rank,
                        requires_extensions: vec![],
                        negotiation_outcome: outcome,
                    },
                );
            }

            alignments.push(WireProtocolAlignment {
                cipher_id,
                cipher_name,
                library_behaviors: behaviors,
            });
        }

        alignments
    }

    fn find_deviations(&self) -> Vec<BehavioralDeviation> {
        let mut deviations = Vec::new();

        deviations.extend(self.find_cipher_acceptance_deviations());
        deviations.extend(self.find_transition_ordering_deviations());
        deviations.extend(self.find_state_reachability_deviations());
        deviations.extend(self.find_fallback_deviations());
        deviations.extend(self.find_error_handling_deviations());

        deviations
    }

    fn find_cipher_acceptance_deviations(&self) -> Vec<BehavioralDeviation> {
        let mut devs = Vec::new();

        for alignment in &self.alignments {
            let libs: Vec<&String> = alignment.library_behaviors.keys().collect();

            for i in 0..libs.len() {
                for j in (i + 1)..libs.len() {
                    let beh_a = &alignment.library_behaviors[libs[i]];
                    let beh_b = &alignment.library_behaviors[libs[j]];

                    if beh_a.supported != beh_b.supported {
                        let accepting_lib = if beh_a.supported { libs[i] } else { libs[j] };
                        let rejecting_lib = if beh_a.supported { libs[j] } else { libs[i] };

                        devs.push(BehavioralDeviation {
                            id: Uuid::new_v4().to_string(),
                            library_a: accepting_lib.clone(),
                            library_b: rejecting_lib.clone(),
                            deviation_type: DeviationType::CipherSuiteAcceptance,
                            description: format!(
                                "{} accepts cipher {:#06x} ({}) but {} rejects it",
                                accepting_lib,
                                alignment.cipher_id,
                                alignment.cipher_name,
                                rejecting_lib
                            ),
                            cipher_suite_id: Some(alignment.cipher_id),
                            state_a: None,
                            state_b: None,
                            phase: Some(HandshakePhase::ServerHelloReceived),
                            security_impact: SecurityImpact::Medium,
                            confidence: 0.8,
                            evidence: vec![
                                format!(
                                    "{}: {:?}",
                                    accepting_lib, beh_a.negotiation_outcome
                                ),
                                format!(
                                    "{}: {:?}",
                                    rejecting_lib, beh_b.negotiation_outcome
                                ),
                            ],
                        });
                    }

                    if beh_a.negotiation_outcome != beh_b.negotiation_outcome
                        && beh_a.supported == beh_b.supported
                    {
                        devs.push(BehavioralDeviation {
                            id: Uuid::new_v4().to_string(),
                            library_a: libs[i].clone(),
                            library_b: libs[j].clone(),
                            deviation_type: DeviationType::FallbackBehavior,
                            description: format!(
                                "Different negotiation outcomes for cipher {:#06x}: {} vs {}",
                                alignment.cipher_id,
                                beh_a.negotiation_outcome,
                                beh_b.negotiation_outcome
                            ),
                            cipher_suite_id: Some(alignment.cipher_id),
                            state_a: None,
                            state_b: None,
                            phase: Some(HandshakePhase::ServerHelloReceived),
                            security_impact: SecurityImpact::Low,
                            confidence: 0.7,
                            evidence: vec![
                                format!("{}: {}", libs[i], beh_a.negotiation_outcome),
                                format!("{}: {}", libs[j], beh_b.negotiation_outcome),
                            ],
                        });
                    }
                }
            }
        }

        devs
    }

    fn find_transition_ordering_deviations(&self) -> Vec<BehavioralDeviation> {
        let mut devs = Vec::new();
        let libs: Vec<(&String, &Lts)> = self.library_lts.iter().collect();

        for i in 0..libs.len() {
            for j in (i + 1)..libs.len() {
                let (name_a, lts_a) = libs[i];
                let (name_b, lts_b) = libs[j];

                let phases_a = self.extract_phase_ordering(lts_a);
                let phases_b = self.extract_phase_ordering(lts_b);

                if phases_a != phases_b {
                    devs.push(BehavioralDeviation {
                        id: Uuid::new_v4().to_string(),
                        library_a: name_a.clone(),
                        library_b: name_b.clone(),
                        deviation_type: DeviationType::TransitionOrdering,
                        description: format!(
                            "Phase ordering differs: {} has {} phases, {} has {} phases",
                            name_a,
                            phases_a.len(),
                            name_b,
                            phases_b.len()
                        ),
                        cipher_suite_id: None,
                        state_a: None,
                        state_b: None,
                        phase: None,
                        security_impact: SecurityImpact::Low,
                        confidence: 0.6,
                        evidence: vec![
                            format!("{}: {:?}", name_a, phases_a),
                            format!("{}: {:?}", name_b, phases_b),
                        ],
                    });
                }
            }
        }

        devs
    }

    fn find_state_reachability_deviations(&self) -> Vec<BehavioralDeviation> {
        let mut devs = Vec::new();
        let libs: Vec<(&String, &Lts)> = self.library_lts.iter().collect();

        for i in 0..libs.len() {
            for j in (i + 1)..libs.len() {
                let (name_a, lts_a) = libs[i];
                let (name_b, lts_b) = libs[j];

                let reachable_a: BTreeSet<String> = lts_a
                    .reachable_states()
                    .iter()
                    .filter_map(|&id| lts_a.get_state(id))
                    .map(|s| format!("{:?}", s.phase))
                    .collect();

                let reachable_b: BTreeSet<String> = lts_b
                    .reachable_states()
                    .iter()
                    .filter_map(|&id| lts_b.get_state(id))
                    .map(|s| format!("{:?}", s.phase))
                    .collect();

                let only_a: BTreeSet<&String> = reachable_a.difference(&reachable_b).collect();
                let only_b: BTreeSet<&String> = reachable_b.difference(&reachable_a).collect();

                if !only_a.is_empty() || !only_b.is_empty() {
                    devs.push(BehavioralDeviation {
                        id: Uuid::new_v4().to_string(),
                        library_a: name_a.clone(),
                        library_b: name_b.clone(),
                        deviation_type: DeviationType::StateReachability,
                        description: format!(
                            "Reachable state difference: {} has {} unique phases, {} has {}",
                            name_a,
                            only_a.len(),
                            name_b,
                            only_b.len()
                        ),
                        cipher_suite_id: None,
                        state_a: None,
                        state_b: None,
                        phase: None,
                        security_impact: SecurityImpact::Low,
                        confidence: 0.5,
                        evidence: vec![
                            format!("Only in {}: {:?}", name_a, only_a),
                            format!("Only in {}: {:?}", name_b, only_b),
                        ],
                    });
                }
            }
        }

        devs
    }

    fn find_fallback_deviations(&self) -> Vec<BehavioralDeviation> {
        let mut devs = Vec::new();
        let libs: Vec<(&String, &Lts)> = self.library_lts.iter().collect();

        for i in 0..libs.len() {
            for j in (i + 1)..libs.len() {
                let (name_a, lts_a) = libs[i];
                let (name_b, lts_b) = libs[j];

                let downgrade_a: Vec<&LtsTransition> =
                    lts_a.transitions.iter().filter(|t| t.is_downgrade).collect();
                let downgrade_b: Vec<&LtsTransition> =
                    lts_b.transitions.iter().filter(|t| t.is_downgrade).collect();

                if downgrade_a.len() != downgrade_b.len() {
                    devs.push(BehavioralDeviation {
                        id: Uuid::new_v4().to_string(),
                        library_a: name_a.clone(),
                        library_b: name_b.clone(),
                        deviation_type: DeviationType::FallbackBehavior,
                        description: format!(
                            "Different downgrade transition counts: {} has {}, {} has {}",
                            name_a,
                            downgrade_a.len(),
                            name_b,
                            downgrade_b.len()
                        ),
                        cipher_suite_id: None,
                        state_a: None,
                        state_b: None,
                        phase: None,
                        security_impact: SecurityImpact::High,
                        confidence: 0.75,
                        evidence: vec![
                            format!("{} downgrade transitions: {}", name_a, downgrade_a.len()),
                            format!("{} downgrade transitions: {}", name_b, downgrade_b.len()),
                        ],
                    });
                }
            }
        }

        devs
    }

    fn find_error_handling_deviations(&self) -> Vec<BehavioralDeviation> {
        let mut devs = Vec::new();
        let libs: Vec<(&String, &Lts)> = self.library_lts.iter().collect();

        for i in 0..libs.len() {
            for j in (i + 1)..libs.len() {
                let (name_a, lts_a) = libs[i];
                let (name_b, lts_b) = libs[j];

                let errors_a = lts_a.error_states().len();
                let errors_b = lts_b.error_states().len();

                if errors_a != errors_b {
                    devs.push(BehavioralDeviation {
                        id: Uuid::new_v4().to_string(),
                        library_a: name_a.clone(),
                        library_b: name_b.clone(),
                        deviation_type: DeviationType::ErrorHandling,
                        description: format!(
                            "Error state count differs: {} has {}, {} has {}",
                            name_a, errors_a, name_b, errors_b
                        ),
                        cipher_suite_id: None,
                        state_a: None,
                        state_b: None,
                        phase: Some(HandshakePhase::Abort),
                        security_impact: SecurityImpact::Medium,
                        confidence: 0.65,
                        evidence: vec![
                            format!("{} error states: {}", name_a, errors_a),
                            format!("{} error states: {}", name_b, errors_b),
                        ],
                    });
                }
            }
        }

        devs
    }

    fn extract_phase_ordering(&self, lts: &Lts) -> Vec<HandshakePhase> {
        let reachable = lts.reachable_states();
        let mut phases: Vec<(u32, HandshakePhase)> = reachable
            .iter()
            .filter_map(|&id| lts.get_state(id).map(|s| (id, s.phase)))
            .collect();
        phases.sort_by_key(|(id, _)| *id);
        phases.dedup_by_key(|(_, phase)| *phase);
        phases.into_iter().map(|(_, p)| p).collect()
    }

    fn generate_certificate(
        &self,
        security_devs: usize,
        benign_devs: usize,
    ) -> CrossLibraryCertificate {
        let libraries: Vec<String> = self.library_lts.keys().cloned().collect();
        let total_ciphers: BTreeSet<u16> = self.alignments.iter().map(|a| a.cipher_id).collect();

        let total_devs = security_devs + benign_devs;
        let max_possible = self.alignments.len() * libraries.len().saturating_sub(1);
        let interop_score = if max_possible > 0 {
            1.0 - (total_devs as f64 / max_possible as f64).min(1.0)
        } else {
            1.0
        };

        let mut hasher = Sha256::new();
        for lib in &libraries {
            hasher.update(lib.as_bytes());
        }
        hasher.update(security_devs.to_le_bytes());
        hasher.update(benign_devs.to_le_bytes());
        let hash = hex::encode(hasher.finalize());

        CrossLibraryCertificate {
            id: Uuid::new_v4().to_string(),
            libraries,
            timestamp: Utc::now().to_rfc3339(),
            total_scenarios: self.alignments.len(),
            deviations_found: total_devs,
            security_deviations: security_devs,
            benign_deviations: benign_devs,
            cipher_suites_tested: total_ciphers.len(),
            versions_tested: 0,
            interop_score,
            hash,
        }
    }
}

impl Default for DifferentialAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of differential analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DifferentialResult {
    pub libraries: Vec<String>,
    pub alignments: Vec<WireProtocolAlignment>,
    pub deviations: Vec<BehavioralDeviation>,
    pub security_deviations: usize,
    pub benign_deviations: usize,
    pub certificate: CrossLibraryCertificate,
}

fn iana_cipher_name(id: u16) -> String {
    match id {
        0x0000 => "TLS_NULL_WITH_NULL_NULL".into(),
        0x0001 => "TLS_RSA_WITH_NULL_MD5".into(),
        0x0002 => "TLS_RSA_WITH_NULL_SHA".into(),
        0x0003 => "TLS_RSA_EXPORT_WITH_RC4_40_MD5".into(),
        0x0004 => "TLS_RSA_WITH_RC4_128_MD5".into(),
        0x0005 => "TLS_RSA_WITH_RC4_128_SHA".into(),
        0x000A => "TLS_RSA_WITH_3DES_EDE_CBC_SHA".into(),
        0x002F => "TLS_RSA_WITH_AES_128_CBC_SHA".into(),
        0x0035 => "TLS_RSA_WITH_AES_256_CBC_SHA".into(),
        0x009C => "TLS_RSA_WITH_AES_128_GCM_SHA256".into(),
        0x009D => "TLS_RSA_WITH_AES_256_GCM_SHA384".into(),
        0x009E => "TLS_DHE_RSA_WITH_AES_128_GCM_SHA256".into(),
        0x009F => "TLS_DHE_RSA_WITH_AES_256_GCM_SHA384".into(),
        0xC02B => "TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256".into(),
        0xC02F => "TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256".into(),
        0x1301 => "TLS_AES_128_GCM_SHA256".into(),
        0x1302 => "TLS_AES_256_GCM_SHA384".into(),
        0x1303 => "TLS_CHACHA20_POLY1305_SHA256".into(),
        _ => format!("UNKNOWN_{:#06x}", id),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_lts(name: &str, ciphers: &[(u16, bool)], extra_states: usize) -> Lts {
        let mut lts = Lts::new(name);
        let mut sid = 0u32;
        let mut tid = 0u32;

        lts.add_state(LtsState::new(sid, "initial", HandshakePhase::Init));
        let init_id = sid;
        sid += 1;

        let mut hello = LtsState::new(sid, "client_hello", HandshakePhase::ClientHelloSent);
        let hello_id = sid;
        lts.add_state(hello);
        sid += 1;

        lts.add_transition(LtsTransition::new(tid, init_id, hello_id, "init_to_hello"));
        tid += 1;

        for &(cipher_id, is_downgrade) in ciphers {
            let mut server_hello =
                LtsState::new(sid, &format!("server_{:#06x}", cipher_id), HandshakePhase::ServerHelloReceived);
            server_hello.is_accepting = !is_downgrade;
            let server_id = sid;
            lts.add_state(server_hello);
            sid += 1;

            let mut trans = LtsTransition::new(tid, hello_id, server_id, format!("select_{:#06x}", cipher_id));
            trans.cipher_suite_id = Some(cipher_id);
            trans.is_downgrade = is_downgrade;
            lts.add_transition(trans);
            tid += 1;
        }

        for i in 0..extra_states {
            let mut state = LtsState::new(sid, &format!("extra_{}", i), HandshakePhase::Negotiated);
            lts.add_state(state);
            sid += 1;
        }

        lts
    }

    #[test]
    fn test_differential_analyzer_basic() {
        let mut analyzer = DifferentialAnalyzer::new();
        let lts_a = make_lts("lib_a", &[(0x002F, false), (0x0035, false), (0x0003, true)], 0);
        let lts_b = make_lts("lib_b", &[(0x002F, false), (0x0035, false)], 0);

        analyzer.add_library("lib_a", lts_a);
        analyzer.add_library("lib_b", lts_b);
        assert_eq!(analyzer.library_count(), 2);

        let result = analyzer.analyze();
        assert_eq!(result.libraries.len(), 2);
        assert!(!result.deviations.is_empty());
    }

    #[test]
    fn test_cipher_acceptance_deviation() {
        let mut analyzer = DifferentialAnalyzer::new();
        analyzer.add_library("a", make_lts("a", &[(0x0003, true)], 0));
        analyzer.add_library("b", make_lts("b", &[(0x002F, false)], 0));

        let result = analyzer.analyze();
        let cipher_devs: Vec<&BehavioralDeviation> = result
            .deviations
            .iter()
            .filter(|d| d.deviation_type == DeviationType::CipherSuiteAcceptance)
            .collect();
        assert!(!cipher_devs.is_empty());
    }

    #[test]
    fn test_identical_libraries_no_cipher_deviations() {
        let mut analyzer = DifferentialAnalyzer::new();
        let ciphers = vec![(0x002F, false), (0x0035, false)];
        analyzer.add_library("a", make_lts("a", &ciphers, 0));
        analyzer.add_library("b", make_lts("b", &ciphers, 0));

        let result = analyzer.analyze();
        let cipher_devs: Vec<&BehavioralDeviation> = result
            .deviations
            .iter()
            .filter(|d| d.deviation_type == DeviationType::CipherSuiteAcceptance)
            .collect();
        assert!(cipher_devs.is_empty());
    }

    #[test]
    fn test_deviation_ranker_ordering() {
        let ranker = DeviationRanker::new();
        let mut devs = vec![
            BehavioralDeviation {
                id: "1".into(),
                library_a: "a".into(),
                library_b: "b".into(),
                deviation_type: DeviationType::TimingDifference,
                description: "minor timing".into(),
                cipher_suite_id: None,
                state_a: None,
                state_b: None,
                phase: None,
                security_impact: SecurityImpact::None,
                confidence: 0.5,
                evidence: vec![],
            },
            BehavioralDeviation {
                id: "2".into(),
                library_a: "a".into(),
                library_b: "b".into(),
                deviation_type: DeviationType::CipherSuiteAcceptance,
                description: "deprecated cipher".into(),
                cipher_suite_id: Some(0x0003),
                state_a: None,
                state_b: None,
                phase: None,
                security_impact: SecurityImpact::None,
                confidence: 0.5,
                evidence: vec!["ev1".into(), "ev2".into()],
            },
        ];

        ranker.rank(&mut devs);
        assert!(devs[0].security_impact > devs[1].security_impact);
    }

    #[test]
    fn test_security_impact_scoring() {
        assert!(SecurityImpact::Critical.score() > SecurityImpact::High.score());
        assert!(SecurityImpact::High.score() > SecurityImpact::Medium.score());
        assert!(SecurityImpact::Medium.score() > SecurityImpact::Low.score());
        assert!(SecurityImpact::Low.score() > SecurityImpact::None.score());
    }

    #[test]
    fn test_covering_design_pairwise() {
        let gen = CoveringDesignGenerator::new(
            2,
            vec![0x002F, 0x0035, 0x009C],
            vec![ProtocolVersion::tls12(), ProtocolVersion::tls13()],
            vec![0x0000, 0x000D],
        );
        let rows = gen.generate();
        assert!(rows.len() >= 6);

        let mut pairs_covered: HashSet<(u16, ProtocolVersion)> = HashSet::new();
        for row in &rows {
            if let (Some(c), Some(v)) = (row.cipher_suite, row.version) {
                pairs_covered.insert((c, v));
            }
        }
        assert!(pairs_covered.len() >= 6);
    }

    #[test]
    fn test_covering_design_boundary_cases() {
        let gen = CoveringDesignGenerator::new(1, vec![0x002F], vec![ProtocolVersion::tls12()], vec![]);
        let rows = gen.generate();
        assert!(rows.iter().any(|r| r.is_boundary));
    }

    #[test]
    fn test_wire_protocol_alignment() {
        let align = WireProtocolAlignment {
            cipher_id: 0x002F,
            cipher_name: "TLS_RSA_WITH_AES_128_CBC_SHA".into(),
            library_behaviors: BTreeMap::from([
                (
                    "openssl".into(),
                    CipherBehavior {
                        supported: true,
                        offered_by_default: true,
                        priority_rank: Some(1),
                        requires_extensions: vec![],
                        negotiation_outcome: NegotiationOutcome::Accepted,
                    },
                ),
                (
                    "gnutls".into(),
                    CipherBehavior {
                        supported: true,
                        offered_by_default: false,
                        priority_rank: Some(5),
                        requires_extensions: vec![],
                        negotiation_outcome: NegotiationOutcome::Accepted,
                    },
                ),
            ]),
        };
        assert_eq!(align.library_behaviors.len(), 2);
    }

    #[test]
    fn test_negotiation_outcome_display() {
        assert_eq!(format!("{}", NegotiationOutcome::Accepted), "ACCEPTED");
        assert_eq!(
            format!("{}", NegotiationOutcome::FallbackToWeaker(0x0003)),
            "FALLBACK(0x0003)"
        );
    }

    #[test]
    fn test_cross_library_certificate() {
        let mut analyzer = DifferentialAnalyzer::new();
        analyzer.add_library("a", make_lts("a", &[(0x002F, false)], 0));
        analyzer.add_library("b", make_lts("b", &[(0x002F, false)], 0));

        let result = analyzer.analyze();
        assert!(!result.certificate.hash.is_empty());
        assert_eq!(result.certificate.libraries.len(), 2);
    }

    #[test]
    fn test_iana_cipher_name() {
        assert_eq!(iana_cipher_name(0x002F), "TLS_RSA_WITH_AES_128_CBC_SHA");
        assert_eq!(iana_cipher_name(0x1301), "TLS_AES_128_GCM_SHA256");
        assert!(iana_cipher_name(0xFFFF).contains("UNKNOWN"));
    }

    #[test]
    fn test_three_library_comparison() {
        let mut analyzer = DifferentialAnalyzer::new();
        analyzer.add_library("a", make_lts("a", &[(0x002F, false), (0x0003, true)], 0));
        analyzer.add_library("b", make_lts("b", &[(0x002F, false)], 0));
        analyzer.add_library("c", make_lts("c", &[(0x002F, false), (0x0035, false)], 0));

        let result = analyzer.analyze();
        assert_eq!(result.libraries.len(), 3);
        assert!(!result.deviations.is_empty());
    }
}
