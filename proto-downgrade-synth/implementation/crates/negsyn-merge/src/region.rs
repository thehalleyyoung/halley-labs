//! Merge region identification, classification, and analysis.
//!
//! A *merge region* is a contiguous sequence of symbolic execution states that
//! share the same handshake phase and protocol parameters, making them
//! candidates for protocol-aware merging. This module identifies such regions,
//! classifies them by mergeability, and provides utilities for splitting and
//! joining them.

use std::collections::BTreeSet;
use std::fmt;

use serde::{Deserialize, Serialize};

use negsyn_types::{HandshakePhase, MergeConfig, ProtocolVersion, SymbolicState};

use crate::algebraic::{FallbackAction, PropertyChecker, PropertyCheckResult};

// ---------------------------------------------------------------------------
// Region classification
// ---------------------------------------------------------------------------

/// Classification of a merge region based on algebraic property satisfaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RegionClassification {
    /// All four axioms (A1–A4) hold — full protocol-aware merge is applicable.
    FullyMergeable,
    /// Some axioms hold — partial merge with fallback for violated properties.
    PartiallyMergeable,
    /// No axioms hold or hard requirements fail — cannot merge.
    NoMerge,
}

impl fmt::Display for RegionClassification {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FullyMergeable => write!(f, "FullyMergeable"),
            Self::PartiallyMergeable => write!(f, "PartiallyMergeable"),
            Self::NoMerge => write!(f, "NoMerge"),
        }
    }
}

// ---------------------------------------------------------------------------
// Region boundary
// ---------------------------------------------------------------------------

/// A boundary between two adjacent merge regions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionBoundary {
    /// Index of the last state in the preceding region.
    pub end_index: usize,
    /// Reason the boundary exists.
    pub reason: BoundaryReason,
}

/// Why a region boundary was placed here.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BoundaryReason {
    PhaseDifference {
        before: HandshakePhase,
        after: HandshakePhase,
    },
    VersionDifference {
        before: Option<ProtocolVersion>,
        after: Option<ProtocolVersion>,
    },
    CipherSetDifference,
    PropertyViolation,
}

impl fmt::Display for BoundaryReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PhaseDifference { before, after } => {
                write!(f, "phase change: {:?} -> {:?}", before, after)
            }
            Self::VersionDifference { before, after } => {
                write!(f, "version change: {:?} -> {:?}", before, after)
            }
            Self::CipherSetDifference => write!(f, "offered cipher set changed"),
            Self::PropertyViolation => write!(f, "algebraic property violation"),
        }
    }
}

// ---------------------------------------------------------------------------
// Merge region
// ---------------------------------------------------------------------------

/// A contiguous group of states that are candidates for merging.
#[derive(Debug, Clone)]
pub struct MergeRegion {
    /// Indices into the original state list.
    pub state_indices: Vec<usize>,
    /// Shared handshake phase of all states in the region.
    pub phase: HandshakePhase,
    /// Shared protocol version.
    pub version: Option<ProtocolVersion>,
    /// Union of offered cipher suites across the region.
    pub offered_ciphers: BTreeSet<u16>,
    /// Classification after property checking.
    pub classification: RegionClassification,
    /// Property check result (if available).
    pub property_result: Option<PropertyCheckResult>,
    /// Recommended fallback when classification is not FullyMergeable.
    pub fallback: Option<FallbackAction>,
}

impl MergeRegion {
    /// Number of states in the region.
    pub fn len(&self) -> usize {
        self.state_indices.len()
    }

    pub fn is_empty(&self) -> bool {
        self.state_indices.is_empty()
    }

    /// Can this region be protocol-aware merged?
    pub fn is_mergeable(&self) -> bool {
        matches!(
            self.classification,
            RegionClassification::FullyMergeable | RegionClassification::PartiallyMergeable
        )
    }
}

// ---------------------------------------------------------------------------
// Region classifier
// ---------------------------------------------------------------------------

/// Classifies a set of states into merge regions and checks properties.
pub struct RegionClassifier {
    config: MergeConfig,
}

impl RegionClassifier {
    pub fn new(config: MergeConfig) -> Self {
        Self { config }
    }

    /// Identify contiguous regions among an ordered slice of states.
    ///
    /// States are grouped when they share the same phase, version, and offered
    /// cipher set. Each group becomes a `MergeRegion`.
    pub fn identify_regions(&self, states: &[SymbolicState]) -> Vec<MergeRegion> {
        if states.is_empty() {
            return Vec::new();
        }

        let mut regions: Vec<MergeRegion> = Vec::new();
        let mut current_indices = vec![0usize];
        let mut current_phase = states[0].negotiation.phase;
        let mut current_version = states[0].negotiation.version;
        let mut current_ciphers: Vec<negsyn_types::CipherSuite> = states[0].negotiation.offered_ciphers.clone();

        for i in 1..states.len() {
            let s = &states[i];
            if s.negotiation.phase == current_phase
                && s.negotiation.version == current_version
                && s.negotiation.offered_ciphers == current_ciphers
            {
                current_indices.push(i);
            } else {
                regions.push(MergeRegion {
                    state_indices: current_indices.clone(),
                    phase: current_phase,
                    version: current_version,
                    offered_ciphers: current_ciphers.iter().map(|c| c.iana_id).collect(),
                    classification: RegionClassification::NoMerge,
                    property_result: None,
                    fallback: None,
                });
                current_indices = vec![i];
                current_phase = s.negotiation.phase;
                current_version = s.negotiation.version;
                current_ciphers = s.negotiation.offered_ciphers.clone();
            }
        }

        // Push the last region.
        regions.push(MergeRegion {
            state_indices: current_indices,
            phase: current_phase,
            version: current_version,
            offered_ciphers: current_ciphers.iter().map(|c| c.iana_id).collect(),
            classification: RegionClassification::NoMerge,
            property_result: None,
            fallback: None,
        });

        regions
    }

    /// Classify each region by running A1–A4 checks on its states.
    pub fn classify_regions(
        &self,
        states: &[SymbolicState],
        regions: &mut [MergeRegion],
    ) {
        let checker = PropertyChecker::new(self.config.clone());

        for region in regions.iter_mut() {
            if region.state_indices.len() < 2 {
                region.classification = RegionClassification::NoMerge;
                continue;
            }

            let refs: Vec<&SymbolicState> = region
                .state_indices
                .iter()
                .map(|&i| &states[i])
                .collect();

            let result = checker.check_all(&refs);

            region.classification = if result.all_satisfied() {
                RegionClassification::FullyMergeable
            } else if result.merge_eligible() {
                RegionClassification::PartiallyMergeable
            } else {
                RegionClassification::NoMerge
            };

            region.fallback = result.recommended_fallback;
            region.property_result = Some(result);
        }
    }

    /// Identify boundaries between adjacent states.
    pub fn find_boundaries(&self, states: &[SymbolicState]) -> Vec<RegionBoundary> {
        let mut boundaries = Vec::new();
        for i in 1..states.len() {
            let prev = &states[i - 1];
            let curr = &states[i];

            if prev.negotiation.phase != curr.negotiation.phase {
                boundaries.push(RegionBoundary {
                    end_index: i - 1,
                    reason: BoundaryReason::PhaseDifference {
                        before: prev.negotiation.phase,
                        after: curr.negotiation.phase,
                    },
                });
            } else if prev.negotiation.version != curr.negotiation.version {
                boundaries.push(RegionBoundary {
                    end_index: i - 1,
                    reason: BoundaryReason::VersionDifference {
                        before: prev.negotiation.version,
                        after: curr.negotiation.version,
                    },
                });
            } else if prev.negotiation.offered_ciphers != curr.negotiation.offered_ciphers {
                boundaries.push(RegionBoundary {
                    end_index: i - 1,
                    reason: BoundaryReason::CipherSetDifference,
                });
            }
        }
        boundaries
    }
}

// ---------------------------------------------------------------------------
// Region analysis
// ---------------------------------------------------------------------------

/// Summary statistics for a set of identified regions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionAnalysis {
    pub total_regions: usize,
    pub fully_mergeable: usize,
    pub partially_mergeable: usize,
    pub no_merge: usize,
    pub total_states: usize,
    pub mergeable_states: usize,
    pub boundary_count: usize,
}

impl RegionAnalysis {
    /// Build analysis from a set of classified regions and boundaries.
    pub fn from_regions(regions: &[MergeRegion], boundary_count: usize) -> Self {
        let mut fully = 0;
        let mut partial = 0;
        let mut none = 0;
        let mut mergeable_states = 0;
        let mut total_states = 0;

        for r in regions {
            total_states += r.state_indices.len();
            match r.classification {
                RegionClassification::FullyMergeable => {
                    fully += 1;
                    mergeable_states += r.state_indices.len();
                }
                RegionClassification::PartiallyMergeable => {
                    partial += 1;
                    mergeable_states += r.state_indices.len();
                }
                RegionClassification::NoMerge => {
                    none += 1;
                }
            }
        }

        Self {
            total_regions: regions.len(),
            fully_mergeable: fully,
            partially_mergeable: partial,
            no_merge: none,
            total_states,
            mergeable_states,
            boundary_count,
        }
    }

    /// Fraction of states that are in a mergeable region.
    pub fn merge_coverage(&self) -> f64 {
        if self.total_states == 0 {
            return 0.0;
        }
        self.mergeable_states as f64 / self.total_states as f64
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use negsyn_types::{HandshakePhase, MergeConfig, NegotiationState, ProtocolVersion};

    fn make_test_state(id: u64, phase: HandshakePhase, ciphers: &[u16]) -> SymbolicState {
        let mut neg = NegotiationState::new(phase, ProtocolVersion::Tls12);
        neg.offered_ciphers = ciphers.iter().copied().collect();
        SymbolicState::new(id, 0x1000, neg)
    }

    #[test]
    fn test_identify_single_region() {
        let config = MergeConfig::default();
        let classifier = RegionClassifier::new(config);
        let states = vec![
            make_test_state(1, HandshakePhase::ClientHello, &[0x002F]),
            make_test_state(2, HandshakePhase::ClientHello, &[0x002F]),
            make_test_state(3, HandshakePhase::ClientHello, &[0x002F]),
        ];
        let regions = classifier.identify_regions(&states);
        assert_eq!(regions.len(), 1);
        assert_eq!(regions[0].state_indices.len(), 3);
        assert_eq!(regions[0].phase, HandshakePhase::ClientHello);
    }

    #[test]
    fn test_identify_multiple_regions() {
        let config = MergeConfig::default();
        let classifier = RegionClassifier::new(config);
        let states = vec![
            make_test_state(1, HandshakePhase::ClientHello, &[0x002F]),
            make_test_state(2, HandshakePhase::ClientHello, &[0x002F]),
            make_test_state(3, HandshakePhase::ServerHello, &[0x002F]),
            make_test_state(4, HandshakePhase::ServerHello, &[0x002F]),
        ];
        let regions = classifier.identify_regions(&states);
        assert_eq!(regions.len(), 2);
        assert_eq!(regions[0].phase, HandshakePhase::ClientHello);
        assert_eq!(regions[1].phase, HandshakePhase::ServerHello);
    }

    #[test]
    fn test_identify_cipher_set_boundary() {
        let config = MergeConfig::default();
        let classifier = RegionClassifier::new(config);
        let states = vec![
            make_test_state(1, HandshakePhase::ClientHello, &[0x002F]),
            make_test_state(2, HandshakePhase::ClientHello, &[0xC02F]),
        ];
        let regions = classifier.identify_regions(&states);
        assert_eq!(regions.len(), 2);
    }

    #[test]
    fn test_identify_empty() {
        let config = MergeConfig::default();
        let classifier = RegionClassifier::new(config);
        let regions = classifier.identify_regions(&[]);
        assert!(regions.is_empty());
    }

    #[test]
    fn test_classify_regions() {
        let config = MergeConfig::default();
        let classifier = RegionClassifier::new(config);
        let states = vec![
            make_test_state(1, HandshakePhase::ClientHello, &[0x002F]),
            make_test_state(2, HandshakePhase::ClientHello, &[0x002F]),
        ];
        let mut regions = classifier.identify_regions(&states);
        classifier.classify_regions(&states, &mut regions);
        // With only 2 states and default config, should be at least partially mergeable
        assert_ne!(regions[0].classification, RegionClassification::NoMerge);
    }

    #[test]
    fn test_single_state_region_no_merge() {
        let config = MergeConfig::default();
        let classifier = RegionClassifier::new(config);
        let states = vec![make_test_state(1, HandshakePhase::ClientHello, &[0x002F])];
        let mut regions = classifier.identify_regions(&states);
        classifier.classify_regions(&states, &mut regions);
        assert_eq!(regions[0].classification, RegionClassification::NoMerge);
    }

    #[test]
    fn test_find_boundaries() {
        let config = MergeConfig::default();
        let classifier = RegionClassifier::new(config);
        let states = vec![
            make_test_state(1, HandshakePhase::ClientHello, &[0x002F]),
            make_test_state(2, HandshakePhase::ServerHello, &[0x002F]),
            make_test_state(3, HandshakePhase::ServerHello, &[0x002F]),
        ];
        let boundaries = classifier.find_boundaries(&states);
        assert_eq!(boundaries.len(), 1);
        assert_eq!(boundaries[0].end_index, 0);
    }

    #[test]
    fn test_region_analysis() {
        let regions = vec![
            MergeRegion {
                state_indices: vec![0, 1, 2],
                phase: HandshakePhase::ClientHello,
                version: ProtocolVersion::Tls12,
                offered_ciphers: [0x002F].iter().copied().collect(),
                classification: RegionClassification::FullyMergeable,
                property_result: None,
                fallback: None,
            },
            MergeRegion {
                state_indices: vec![3],
                phase: HandshakePhase::ServerHello,
                version: ProtocolVersion::Tls12,
                offered_ciphers: [0x002F].iter().copied().collect(),
                classification: RegionClassification::NoMerge,
                property_result: None,
                fallback: None,
            },
        ];
        let analysis = RegionAnalysis::from_regions(&regions, 1);
        assert_eq!(analysis.total_regions, 2);
        assert_eq!(analysis.fully_mergeable, 1);
        assert_eq!(analysis.no_merge, 1);
        assert_eq!(analysis.total_states, 4);
        assert_eq!(analysis.mergeable_states, 3);
        assert!((analysis.merge_coverage() - 0.75).abs() < f64::EPSILON);
    }

    #[test]
    fn test_region_classification_display() {
        assert_eq!(
            format!("{}", RegionClassification::FullyMergeable),
            "FullyMergeable"
        );
    }

    #[test]
    fn test_merge_region_is_mergeable() {
        let r = MergeRegion {
            state_indices: vec![0, 1],
            phase: HandshakePhase::ClientHello,
            version: ProtocolVersion::Tls12,
            offered_ciphers: BTreeSet::new(),
            classification: RegionClassification::FullyMergeable,
            property_result: None,
            fallback: None,
        };
        assert!(r.is_mergeable());
    }
}
