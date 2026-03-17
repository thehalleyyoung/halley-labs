//! Fallback strategy selection, region decomposition, and fallback statistics.
//!
//! When protocol-aware merge is not fully applicable (one or more algebraic
//! axioms are violated) the fallback module decides an alternative strategy,
//! optionally decomposes a region into sub-regions that can each be handled
//! independently, and tracks statistics on fallback usage.

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use negsyn_types::{HandshakePhase, MergeConfig, ProtocolVersion, SymbolicState};

use crate::algebraic::{FallbackAction, PropertyCheckResult, PropertyViolation};
use crate::region::{MergeRegion, RegionClassification, RegionClassifier};

// ---------------------------------------------------------------------------
// Fallback strategy
// ---------------------------------------------------------------------------

/// High-level strategy the engine should use for a region that cannot be
/// fully merged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FallbackStrategy {
    /// Apply generic veritesting: inline straight-line segments only.
    GenericVeritesting,
    /// Fork symbolic execution — keep states separate.
    ForkExecution,
    /// Decompose the region into smaller sub-regions and retry.
    RegionDecomposition,
    /// Produce an abstract summary (widening/abstraction).
    AbstractSummarization,
}

impl fmt::Display for FallbackStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::GenericVeritesting => write!(f, "GenericVeritesting"),
            Self::ForkExecution => write!(f, "ForkExecution"),
            Self::RegionDecomposition => write!(f, "RegionDecomposition"),
            Self::AbstractSummarization => write!(f, "AbstractSummarization"),
        }
    }
}

impl From<FallbackAction> for FallbackStrategy {
    fn from(action: FallbackAction) -> Self {
        match action {
            FallbackAction::GenericVeritesting => Self::GenericVeritesting,
            FallbackAction::Fork => Self::ForkExecution,
            FallbackAction::Decompose => Self::RegionDecomposition,
            FallbackAction::Summarize => Self::AbstractSummarization,
        }
    }
}

// ---------------------------------------------------------------------------
// Fallback decider
// ---------------------------------------------------------------------------

/// Selects a [`FallbackStrategy`] for a region whose merge was rejected or
/// only partially applicable.
pub struct FallbackDecider {
    config: MergeConfig,
}

impl FallbackDecider {
    pub fn new(config: MergeConfig) -> Self {
        Self { config }
    }

    /// Choose a strategy based on the property-check result.
    pub fn decide(&self, result: &PropertyCheckResult) -> FallbackStrategy {
        if result.all_satisfied() {
            // Should not normally be called for satisfied results, but handle
            // it gracefully.
            return FallbackStrategy::GenericVeritesting;
        }

        // If there is a recommendation from the algebraic checker, use it.
        if let Some(action) = result.recommended_fallback {
            return FallbackStrategy::from(action);
        }

        self.decide_from_violations(&result.violations)
    }

    /// Choose a strategy from the violation list directly.
    pub fn decide_from_violations(&self, violations: &[PropertyViolation]) -> FallbackStrategy {
        if violations.is_empty() {
            return FallbackStrategy::GenericVeritesting;
        }

        let has_lattice = violations
            .iter()
            .any(|v| matches!(v, PropertyViolation::LatticeViolation { .. }));
        let has_monotonicity = violations
            .iter()
            .any(|v| matches!(v, PropertyViolation::MonotonicityViolation { .. }));
        let has_determinism = violations
            .iter()
            .any(|v| matches!(v, PropertyViolation::DeterminismViolation { .. }));
        let has_finite_exceeded = violations
            .iter()
            .any(|v| matches!(v, PropertyViolation::FiniteOutcomeExceeded { .. }));

        // Priority-based selection:
        // Lattice violation → fork (incompatible security levels)
        if has_lattice {
            return FallbackStrategy::ForkExecution;
        }
        // Monotonicity or determinism → decompose into smaller pieces
        if has_monotonicity || has_determinism {
            return FallbackStrategy::RegionDecomposition;
        }
        // Finite outcomes exceeded → summarize to reduce explosion
        if has_finite_exceeded {
            return FallbackStrategy::AbstractSummarization;
        }

        FallbackStrategy::GenericVeritesting
    }

    /// Convenience: decide for a region and return the strategy.
    pub fn decide_for_region(&self, region: &MergeRegion) -> FallbackStrategy {
        match &region.property_result {
            Some(result) => self.decide(result),
            None => {
                // Unclassified region — use fallback if present.
                match region.fallback {
                    Some(action) => FallbackStrategy::from(action),
                    None => FallbackStrategy::ForkExecution,
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Region decomposer
// ---------------------------------------------------------------------------

/// Attempts to split a partially-mergeable or non-mergeable region into
/// smaller sub-regions that individually satisfy merge requirements.
pub struct RegionDecomposer {
    config: MergeConfig,
}

impl RegionDecomposer {
    pub fn new(config: MergeConfig) -> Self {
        Self { config }
    }

    /// Decompose a region by splitting at phase and cipher-set boundaries.
    ///
    /// Returns a vector of sub-regions. Each sub-region is re-classified using
    /// the property checker.
    pub fn decompose(
        &self,
        states: &[SymbolicState],
        region: &MergeRegion,
    ) -> Vec<MergeRegion> {
        if region.state_indices.len() <= 2 {
            // Cannot decompose further.
            return vec![region.clone()];
        }

        // Build local state slice from region indices.
        let local_states: Vec<SymbolicState> = region
            .state_indices
            .iter()
            .map(|&i| states[i].clone())
            .collect();

        let classifier = RegionClassifier::new(self.config.clone());
        let mut sub_regions = classifier.identify_regions(&local_states);
        classifier.classify_regions(&local_states, &mut sub_regions);

        // Remap indices back to global indices.
        for sub in &mut sub_regions {
            sub.state_indices = sub
                .state_indices
                .iter()
                .map(|&local_i| region.state_indices[local_i])
                .collect();
        }

        sub_regions
    }

    /// Try to split a region in half (binary split).
    pub fn binary_split(&self, region: &MergeRegion) -> (MergeRegion, MergeRegion) {
        let mid = region.state_indices.len() / 2;
        let (left_idx, right_idx) = region.state_indices.split_at(mid);

        let left = MergeRegion {
            state_indices: left_idx.to_vec(),
            phase: region.phase,
            version: region.version,
            offered_ciphers: region.offered_ciphers.clone(),
            classification: RegionClassification::NoMerge,
            property_result: None,
            fallback: None,
        };

        let right = MergeRegion {
            state_indices: right_idx.to_vec(),
            phase: region.phase,
            version: region.version,
            offered_ciphers: region.offered_ciphers.clone(),
            classification: RegionClassification::NoMerge,
            property_result: None,
            fallback: None,
        };

        (left, right)
    }
}

// ---------------------------------------------------------------------------
// Fallback statistics
// ---------------------------------------------------------------------------

/// Tracks how often each fallback strategy is selected and whether it
/// succeeded in producing a useful result.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FallbackStatistics {
    /// Number of times each strategy was used.
    pub usage_counts: BTreeMap<String, u64>,
    /// Number of times each strategy succeeded.
    pub success_counts: BTreeMap<String, u64>,
    /// Total fallback invocations.
    pub total_fallbacks: u64,
}

impl FallbackStatistics {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record that a strategy was selected.
    pub fn record_usage(&mut self, strategy: FallbackStrategy) {
        let key = strategy.to_string();
        *self.usage_counts.entry(key).or_insert(0) += 1;
        self.total_fallbacks += 1;
    }

    /// Record that a strategy succeeded (after recording usage).
    pub fn record_success(&mut self, strategy: FallbackStrategy) {
        let key = strategy.to_string();
        *self.success_counts.entry(key).or_insert(0) += 1;
    }

    /// Success rate for a given strategy (0.0 if never used).
    pub fn success_rate(&self, strategy: FallbackStrategy) -> f64 {
        let key = strategy.to_string();
        let used = self.usage_counts.get(&key).copied().unwrap_or(0);
        let ok = self.success_counts.get(&key).copied().unwrap_or(0);
        if used == 0 {
            0.0
        } else {
            ok as f64 / used as f64
        }
    }

    /// Overall success rate across all strategies.
    pub fn overall_success_rate(&self) -> f64 {
        let total_success: u64 = self.success_counts.values().sum();
        if self.total_fallbacks == 0 {
            0.0
        } else {
            total_success as f64 / self.total_fallbacks as f64
        }
    }

    /// Most frequently used strategy, if any.
    pub fn most_used_strategy(&self) -> Option<String> {
        self.usage_counts
            .iter()
            .max_by_key(|(_, &v)| v)
            .map(|(k, _)| k.clone())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    use negsyn_types::{MergeConfig, NegotiationState};

    use crate::algebraic::{FallbackAction, PropertyCheckResult, PropertyViolation};

    fn make_test_state(id: u64, phase: HandshakePhase, ciphers: &[u16]) -> SymbolicState {
        let mut neg = NegotiationState::new(phase, ProtocolVersion::Tls12);
        neg.offered_ciphers = ciphers.iter().copied().collect();
        SymbolicState::new(id, 0x1000, neg)
    }

    fn make_result(violations: Vec<PropertyViolation>, fallback: Option<FallbackAction>) -> PropertyCheckResult {
        PropertyCheckResult {
            violations,
            recommended_fallback: fallback,
        }
    }

    #[test]
    fn test_decide_all_satisfied() {
        let decider = FallbackDecider::new(MergeConfig::default());
        let result = make_result(vec![], None);
        assert_eq!(decider.decide(&result), FallbackStrategy::GenericVeritesting);
    }

    #[test]
    fn test_decide_recommended_fallback() {
        let decider = FallbackDecider::new(MergeConfig::default());
        let result = make_result(
            vec![PropertyViolation::LatticeViolation {
                description: "test".into(),
            }],
            Some(FallbackAction::Summarize),
        );
        // Should use the recommended fallback, not the violation-based one.
        assert_eq!(decider.decide(&result), FallbackStrategy::AbstractSummarization);
    }

    #[test]
    fn test_decide_lattice_violation() {
        let decider = FallbackDecider::new(MergeConfig::default());
        let violations = vec![PropertyViolation::LatticeViolation {
            description: "test".into(),
        }];
        assert_eq!(
            decider.decide_from_violations(&violations),
            FallbackStrategy::ForkExecution
        );
    }

    #[test]
    fn test_decide_monotonicity_violation() {
        let decider = FallbackDecider::new(MergeConfig::default());
        let violations = vec![PropertyViolation::MonotonicityViolation {
            description: "test".into(),
        }];
        assert_eq!(
            decider.decide_from_violations(&violations),
            FallbackStrategy::RegionDecomposition
        );
    }

    #[test]
    fn test_decide_finite_outcome_exceeded() {
        let decider = FallbackDecider::new(MergeConfig::default());
        let violations = vec![PropertyViolation::FiniteOutcomeExceeded {
            category: crate::algebraic::OutcomeCategory::CipherSuites,
            count: 100,
            limit: 50,
        }];
        assert_eq!(
            decider.decide_from_violations(&violations),
            FallbackStrategy::AbstractSummarization
        );
    }

    #[test]
    fn test_fallback_strategy_from_action() {
        assert_eq!(
            FallbackStrategy::from(FallbackAction::GenericVeritesting),
            FallbackStrategy::GenericVeritesting
        );
        assert_eq!(
            FallbackStrategy::from(FallbackAction::Fork),
            FallbackStrategy::ForkExecution
        );
        assert_eq!(
            FallbackStrategy::from(FallbackAction::Decompose),
            FallbackStrategy::RegionDecomposition
        );
        assert_eq!(
            FallbackStrategy::from(FallbackAction::Summarize),
            FallbackStrategy::AbstractSummarization
        );
    }

    #[test]
    fn test_fallback_statistics() {
        let mut stats = FallbackStatistics::new();
        assert_eq!(stats.total_fallbacks, 0);
        assert_eq!(stats.overall_success_rate(), 0.0);

        stats.record_usage(FallbackStrategy::ForkExecution);
        stats.record_usage(FallbackStrategy::ForkExecution);
        stats.record_success(FallbackStrategy::ForkExecution);

        stats.record_usage(FallbackStrategy::GenericVeritesting);
        stats.record_success(FallbackStrategy::GenericVeritesting);

        assert_eq!(stats.total_fallbacks, 3);
        assert_eq!(stats.success_rate(FallbackStrategy::ForkExecution), 0.5);
        assert_eq!(stats.success_rate(FallbackStrategy::GenericVeritesting), 1.0);
        assert!((stats.overall_success_rate() - 2.0 / 3.0).abs() < 1e-9);
        assert_eq!(
            stats.most_used_strategy(),
            Some("ForkExecution".to_string())
        );
    }

    #[test]
    fn test_region_decomposer_no_split_small() {
        let config = MergeConfig::default();
        let decomposer = RegionDecomposer::new(config);
        let states = vec![
            make_test_state(1, HandshakePhase::ClientHello, &[0x002F]),
            make_test_state(2, HandshakePhase::ClientHello, &[0x002F]),
        ];
        let region = MergeRegion {
            state_indices: vec![0, 1],
            phase: HandshakePhase::ClientHello,
            version: ProtocolVersion::Tls12,
            offered_ciphers: [0x002F].iter().copied().collect(),
            classification: RegionClassification::PartiallyMergeable,
            property_result: None,
            fallback: None,
        };
        let result = decomposer.decompose(&states, &region);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_binary_split() {
        let config = MergeConfig::default();
        let decomposer = RegionDecomposer::new(config);
        let region = MergeRegion {
            state_indices: vec![0, 1, 2, 3],
            phase: HandshakePhase::ClientHello,
            version: ProtocolVersion::Tls12,
            offered_ciphers: BTreeSet::new(),
            classification: RegionClassification::NoMerge,
            property_result: None,
            fallback: None,
        };
        let (left, right) = decomposer.binary_split(&region);
        assert_eq!(left.state_indices, vec![0, 1]);
        assert_eq!(right.state_indices, vec![2, 3]);
    }

    #[test]
    fn test_decide_for_region_with_result() {
        let decider = FallbackDecider::new(MergeConfig::default());
        let region = MergeRegion {
            state_indices: vec![0, 1],
            phase: HandshakePhase::ClientHello,
            version: ProtocolVersion::Tls12,
            offered_ciphers: BTreeSet::new(),
            classification: RegionClassification::NoMerge,
            property_result: Some(make_result(
                vec![PropertyViolation::DeterminismViolation {
                    description: "non-det".into(),
                }],
                None,
            )),
            fallback: None,
        };
        assert_eq!(
            decider.decide_for_region(&region),
            FallbackStrategy::RegionDecomposition
        );
    }

    #[test]
    fn test_decide_for_region_without_result() {
        let decider = FallbackDecider::new(MergeConfig::default());
        let region = MergeRegion {
            state_indices: vec![0, 1],
            phase: HandshakePhase::ClientHello,
            version: ProtocolVersion::Tls12,
            offered_ciphers: BTreeSet::new(),
            classification: RegionClassification::NoMerge,
            property_result: None,
            fallback: Some(FallbackAction::Summarize),
        };
        assert_eq!(
            decider.decide_for_region(&region),
            FallbackStrategy::AbstractSummarization
        );
    }
}
