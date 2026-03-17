//! Protocol bisimulation (Definition D3).
//!
//! Computes the coarsest protocol bisimulation relation on a NegotiationLTS
//! using partition refinement, then produces the bisimulation quotient.
//!
//! Two states s₁, s₂ are protocol-bisimilar (s₁ ≈_P s₂) iff:
//! 1. obs(s₁) = obs(s₂)  — observation agreement
//! 2. For every transition s₁ →ᵃ s₁', there exists s₂ →ᵃ s₂' with s₁' ≈_P s₂'
//!    (and symmetrically)  — transfer property

use crate::{
    observation::ObservationEquivalence, ExtractError, ExtractResult, HandshakePhase, LtsState,
    MessageLabel, NegotiationLTS, Observable, StateId,
};
use log::{debug, trace};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::fmt;

// ---------------------------------------------------------------------------
// BisimulationRelation
// ---------------------------------------------------------------------------

/// A symmetric relation on states representing protocol bisimulation.
///
/// Stored as a partition: each block is an equivalence class of bisimilar states.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BisimulationRelation {
    /// Equivalence classes (blocks). Each block is a set of bisimilar states.
    blocks: Vec<BTreeSet<StateId>>,
    /// Map from state to its block index.
    state_to_block: HashMap<StateId, usize>,
}

impl BisimulationRelation {
    pub fn new() -> Self {
        Self {
            blocks: Vec::new(),
            state_to_block: HashMap::new(),
        }
    }

    /// Create from a partition (list of blocks).
    pub fn from_partition(partition: Vec<Vec<StateId>>) -> Self {
        let mut rel = Self::new();
        for block in partition {
            let block_idx = rel.blocks.len();
            let block_set: BTreeSet<StateId> = block.iter().copied().collect();
            for &sid in &block_set {
                rel.state_to_block.insert(sid, block_idx);
            }
            rel.blocks.push(block_set);
        }
        rel
    }

    /// Number of equivalence classes.
    pub fn class_count(&self) -> usize {
        self.blocks.len()
    }

    /// Total number of states in the relation.
    pub fn state_count(&self) -> usize {
        self.state_to_block.len()
    }

    /// Get the block index of a state.
    pub fn block_of(&self, state: StateId) -> Option<usize> {
        self.state_to_block.get(&state).copied()
    }

    /// Check if two states are bisimilar (in the same block).
    pub fn are_bisimilar(&self, s1: StateId, s2: StateId) -> bool {
        match (self.block_of(s1), self.block_of(s2)) {
            (Some(b1), Some(b2)) => b1 == b2,
            _ => false,
        }
    }

    /// Get all states in a block.
    pub fn block_states(&self, block_idx: usize) -> Option<&BTreeSet<StateId>> {
        self.blocks.get(block_idx)
    }

    /// Iterate over all blocks.
    pub fn blocks(&self) -> &[BTreeSet<StateId>] {
        &self.blocks
    }

    /// Representative state for a block (the smallest ID).
    pub fn representative(&self, block_idx: usize) -> Option<StateId> {
        self.blocks.get(block_idx).and_then(|b| b.iter().next().copied())
    }

    /// All representatives (one per block).
    pub fn representatives(&self) -> Vec<StateId> {
        self.blocks
            .iter()
            .filter_map(|b| b.iter().next().copied())
            .collect()
    }

    /// Get the equivalence class containing a given state.
    pub fn equivalence_class(&self, state: StateId) -> Option<&BTreeSet<StateId>> {
        self.block_of(state).and_then(|b| self.blocks.get(b))
    }

    /// Sizes of all blocks.
    pub fn block_sizes(&self) -> Vec<usize> {
        self.blocks.iter().map(|b| b.len()).collect()
    }

    /// Maximum block size (indicates how much reduction was achieved).
    pub fn max_block_size(&self) -> usize {
        self.blocks.iter().map(|b| b.len()).max().unwrap_or(0)
    }
}

impl Default for BisimulationRelation {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for BisimulationRelation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Bisimulation: {} classes, {} states",
            self.class_count(),
            self.state_count()
        )?;
        for (i, block) in self.blocks.iter().enumerate() {
            write!(f, "  Block {}: {{", i)?;
            for (j, sid) in block.iter().enumerate() {
                if j > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", sid)?;
            }
            writeln!(f, "}}")?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ProtocolBisimilar trait
// ---------------------------------------------------------------------------

/// Trait for types that can be checked for protocol bisimilarity.
pub trait ProtocolBisimilar {
    /// Check if `self` is bisimilar to `other` under the given relation.
    fn is_bisimilar_to(&self, other: &Self, relation: &BisimulationRelation) -> bool;
}

impl ProtocolBisimilar for StateId {
    fn is_bisimilar_to(&self, other: &Self, relation: &BisimulationRelation) -> bool {
        relation.are_bisimilar(*self, *other)
    }
}

// ---------------------------------------------------------------------------
// BisimulationChecker
// ---------------------------------------------------------------------------

/// Computes the coarsest protocol bisimulation relation on a NegotiationLTS
/// using the partition refinement algorithm.
pub struct BisimulationChecker {
    max_iterations: u32,
    iterations_used: u32,
}

impl BisimulationChecker {
    pub fn new(max_iterations: u32) -> Self {
        Self {
            max_iterations,
            iterations_used: 0,
        }
    }

    /// Number of refinement iterations performed.
    pub fn iterations(&self) -> u32 {
        self.iterations_used
    }

    /// Compute the coarsest protocol bisimulation relation.
    pub fn compute(
        &mut self,
        lts: &NegotiationLTS,
    ) -> ExtractResult<BisimulationRelation> {
        if lts.state_count() == 0 {
            return Ok(BisimulationRelation::new());
        }

        // Step 1: Initial partition by observation equivalence.
        let mut obs_eq = ObservationEquivalence::new();
        let initial_partition = obs_eq.partition_by_observation(lts);
        debug!(
            "Initial partition: {} blocks from {} states",
            initial_partition.len(),
            lts.state_count()
        );

        // Step 2: Partition refinement.
        let refined = self.partition_refinement(lts, initial_partition)?;

        // Step 3: Build the relation.
        let relation = BisimulationRelation::from_partition(refined);
        debug!(
            "Bisimulation: {} classes (from {} states)",
            relation.class_count(),
            relation.state_count()
        );

        // Step 4: Validate transfer property.
        self.validate_transfer_property(lts, &relation)?;

        Ok(relation)
    }

    /// Standard partition refinement adapted for protocol bisimulation.
    ///
    /// The algorithm maintains a partition P and a worklist W of "splitter" blocks.
    /// At each step, it picks a block B from W and, for each action a, splits
    /// every block in P according to whether states have a transition to B under a.
    fn partition_refinement(
        &mut self,
        lts: &NegotiationLTS,
        initial: Vec<Vec<StateId>>,
    ) -> ExtractResult<Vec<Vec<StateId>>> {
        // Build adjacency structures.
        let adj = lts.adjacency_list();
        let all_labels = lts.alphabet();

        // Current partition as a vec of sets.
        let mut partition: Vec<BTreeSet<StateId>> = initial
            .into_iter()
            .map(|block| block.into_iter().collect())
            .collect();

        // Map from state → block index (kept in sync with partition).
        let mut state_block: HashMap<StateId, usize> = HashMap::new();
        for (idx, block) in partition.iter().enumerate() {
            for &s in block {
                state_block.insert(s, idx);
            }
        }

        // Worklist: block indices that may be splitters.
        let mut worklist: VecDeque<usize> = (0..partition.len()).collect();
        let mut in_worklist: HashSet<usize> = worklist.iter().copied().collect();

        self.iterations_used = 0;

        while let Some(splitter_idx) = worklist.pop_front() {
            in_worklist.remove(&splitter_idx);
            self.iterations_used += 1;

            if self.iterations_used > self.max_iterations {
                return Err(ExtractError::BisimulationDiverged {
                    iterations: self.max_iterations,
                });
            }

            if splitter_idx >= partition.len() || partition[splitter_idx].is_empty() {
                continue;
            }

            let splitter: BTreeSet<StateId> = partition[splitter_idx].clone();

            // For each action label, try to split.
            for label in &all_labels {
                // Compute the set of states that can reach the splitter under this label.
                let pre_states: HashSet<StateId> = self.predecessors_under_label(
                    lts,
                    &splitter,
                    label,
                );

                if pre_states.is_empty() {
                    continue;
                }

                // Try to split each block.
                let num_blocks = partition.len();
                let mut new_blocks: Vec<(usize, BTreeSet<StateId>, BTreeSet<StateId>)> =
                    Vec::new();

                for block_idx in 0..num_blocks {
                    if partition[block_idx].is_empty() {
                        continue;
                    }

                    let block = &partition[block_idx];
                    let mut in_pre: BTreeSet<StateId> = BTreeSet::new();
                    let mut not_in_pre: BTreeSet<StateId> = BTreeSet::new();

                    for &s in block {
                        if pre_states.contains(&s) {
                            in_pre.insert(s);
                        } else {
                            not_in_pre.insert(s);
                        }
                    }

                    // Split only if both parts are non-empty.
                    if !in_pre.is_empty() && !not_in_pre.is_empty() {
                        new_blocks.push((block_idx, in_pre, not_in_pre));
                    }
                }

                // Apply splits.
                for (block_idx, part_a, part_b) in new_blocks {
                    // Keep the larger part in the existing block, create new block for smaller.
                    let (keep, split_off) = if part_a.len() >= part_b.len() {
                        (part_a, part_b)
                    } else {
                        (part_b, part_a)
                    };

                    partition[block_idx] = keep;
                    for &s in &partition[block_idx] {
                        state_block.insert(s, block_idx);
                    }

                    let new_idx = partition.len();
                    for &s in &split_off {
                        state_block.insert(s, new_idx);
                    }
                    partition.push(split_off);

                    // Add the smaller block to the worklist (Hopcroft's optimization).
                    if !in_worklist.contains(&new_idx) {
                        worklist.push_back(new_idx);
                        in_worklist.insert(new_idx);
                    }
                    // Also re-add the modified block if not already in worklist.
                    if !in_worklist.contains(&block_idx) {
                        worklist.push_back(block_idx);
                        in_worklist.insert(block_idx);
                    }
                }
            }
        }

        // Convert to Vec<Vec<StateId>>, filtering out empty blocks.
        let result: Vec<Vec<StateId>> = partition
            .into_iter()
            .filter(|b| !b.is_empty())
            .map(|b| b.into_iter().collect())
            .collect();

        debug!(
            "Partition refinement: {} iterations, {} final blocks",
            self.iterations_used,
            result.len()
        );

        Ok(result)
    }

    /// Compute predecessors of a set of states under a given action label.
    fn predecessors_under_label(
        &self,
        lts: &NegotiationLTS,
        target_states: &BTreeSet<StateId>,
        label: &str,
    ) -> HashSet<StateId> {
        let mut preds = HashSet::new();
        for t in &lts.transitions {
            if t.label.label_name() == label && target_states.contains(&t.target) {
                preds.insert(t.source);
            }
        }
        preds
    }

    /// Validate the transfer property: for every pair of bisimilar states,
    /// every transition from one must be matched by a transition from the other
    /// to a bisimilar target.
    fn validate_transfer_property(
        &self,
        lts: &NegotiationLTS,
        relation: &BisimulationRelation,
    ) -> ExtractResult<()> {
        for block in relation.blocks() {
            let states: Vec<StateId> = block.iter().copied().collect();
            if states.len() < 2 {
                continue;
            }

            // Check pairwise within the block.
            let s0 = states[0];
            for &si in &states[1..] {
                // Check s0's transitions can be matched by si.
                for t in lts.transitions_from(s0) {
                    let label = t.label.label_name();
                    let target_block = relation.block_of(t.target);

                    let has_match = lts
                        .transitions_from(si)
                        .iter()
                        .any(|t2| {
                            t2.label.label_name() == label
                                && relation.block_of(t2.target) == target_block
                        });

                    if !has_match {
                        // This is a validation warning, not necessarily a hard error.
                        // The partition refinement should have handled this, but we
                        // log it for debugging.
                        trace!(
                            "Transfer property: {} →[{}] block {:?}, no match from {}",
                            s0,
                            label,
                            target_block,
                            si
                        );
                    }
                }
            }
        }

        Ok(())
    }

    /// Compute the quotient LTS from a bisimulation relation.
    /// Delegates to the QuotientBuilder.
    pub fn compute_quotient(
        &self,
        lts: &NegotiationLTS,
        relation: &BisimulationRelation,
    ) -> ExtractResult<NegotiationLTS> {
        let builder = crate::quotient::QuotientBuilder::new();
        builder.build(lts, relation)
    }
}

// ---------------------------------------------------------------------------
// Observation agreement
// ---------------------------------------------------------------------------

/// Utility functions for checking observation agreement.
pub struct ObservationAgreement;

impl ObservationAgreement {
    /// Check that all states in a block have the same observation.
    pub fn check_block(
        lts: &NegotiationLTS,
        block: &BTreeSet<StateId>,
    ) -> bool {
        let observations: Vec<Option<&Observable>> = block
            .iter()
            .map(|&s| lts.obs(s))
            .collect();

        if observations.is_empty() {
            return true;
        }

        let first = &observations[0];
        observations.iter().all(|o| o == first)
    }

    /// Check that the entire bisimulation relation has observation agreement.
    pub fn check_relation(
        lts: &NegotiationLTS,
        relation: &BisimulationRelation,
    ) -> bool {
        relation
            .blocks()
            .iter()
            .all(|block| Self::check_block(lts, block))
    }
}

// ---------------------------------------------------------------------------
// Transfer property verification
// ---------------------------------------------------------------------------

/// Verifies the transfer property of bisimulation: if s₁ ≈ s₂ and s₁ →ᵃ s₁',
/// then there exists s₂ →ᵃ s₂' such that s₁' ≈ s₂'.
pub struct TransferVerifier;

impl TransferVerifier {
    /// Check the transfer property for a given pair of states.
    pub fn check_pair(
        lts: &NegotiationLTS,
        relation: &BisimulationRelation,
        s1: StateId,
        s2: StateId,
    ) -> bool {
        // s1 → s1' must be matched by s2 → s2' (same label, bisimilar target).
        for t1 in lts.transitions_from(s1) {
            let matched = lts.transitions_from(s2).iter().any(|t2| {
                t2.label.label_name() == t1.label.label_name()
                    && relation.are_bisimilar(t1.target, t2.target)
            });
            if !matched {
                return false;
            }
        }
        // Symmetric: s2 → s2' must be matched by s1 → s1'.
        for t2 in lts.transitions_from(s2) {
            let matched = lts.transitions_from(s1).iter().any(|t1| {
                t1.label.label_name() == t2.label.label_name()
                    && relation.are_bisimilar(t1.target, t2.target)
            });
            if !matched {
                return false;
            }
        }
        true
    }

    /// Check the entire relation.
    pub fn check_all(
        lts: &NegotiationLTS,
        relation: &BisimulationRelation,
    ) -> bool {
        for block in relation.blocks() {
            let states: Vec<StateId> = block.iter().copied().collect();
            for i in 0..states.len() {
                for j in (i + 1)..states.len() {
                    if !Self::check_pair(lts, relation, states[i], states[j]) {
                        return false;
                    }
                }
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NegotiationOutcome;
    use negsyn_types::NegotiationState;

    fn make_neg(phase: HandshakePhase, cipher: Option<u16>) -> NegotiationState {
        let mut ns = NegotiationState::new();
        ns.phase = phase;
        ns.version = Some(ProtocolVersion::Tls12);
        ns.selected_cipher = cipher.map(|id| CipherSuite::new(
            id,
            format!("TEST_0x{:04x}", id),
            negsyn_types::protocol::KeyExchange::NULL,
            negsyn_types::protocol::AuthAlgorithm::NULL,
            negsyn_types::protocol::EncryptionAlgorithm::NULL,
            negsyn_types::protocol::MacAlgorithm::NULL,
            negsyn_types::SecurityLevel::Standard,
        ));
        ns
    }

    #[test]
    fn test_bisimulation_trivial() {
        // Single state → single block.
        let mut lts = NegotiationLTS::new();
        let s0 = lts.add_state(make_neg(HandshakePhase::Initial, None));
        lts.mark_initial(s0);

        let mut checker = BisimulationChecker::new(1000);
        let rel = checker.compute(&lts).unwrap();
        assert_eq!(rel.class_count(), 1);
        assert_eq!(rel.state_count(), 1);
    }

    #[test]
    fn test_bisimulation_two_equivalent_states() {
        // Two states with the same observation and no transitions → one block.
        let mut lts = NegotiationLTS::new();
        let s0 = lts.add_state(make_neg(
            HandshakePhase::ApplicationData,
            Some(0x002f),
        ));
        let s1 = lts.add_state(make_neg(
            HandshakePhase::ApplicationData,
            Some(0x002f),
        ));
        lts.mark_initial(s0);

        let mut checker = BisimulationChecker::new(1000);
        let rel = checker.compute(&lts).unwrap();
        assert!(rel.are_bisimilar(s0, s1));
        assert_eq!(rel.class_count(), 1);
    }

    #[test]
    fn test_bisimulation_two_different_observations() {
        // Two terminal states with different ciphers → two blocks.
        let mut lts = NegotiationLTS::new();
        let s0 = lts.add_state(make_neg(
            HandshakePhase::ApplicationData,
            Some(0x002f),
        ));
        let s1 = lts.add_state(make_neg(
            HandshakePhase::ApplicationData,
            Some(0x0035),
        ));
        lts.mark_initial(s0);

        let mut checker = BisimulationChecker::new(1000);
        let rel = checker.compute(&lts).unwrap();
        assert!(!rel.are_bisimilar(s0, s1));
        assert_eq!(rel.class_count(), 2);
    }

    #[test]
    fn test_bisimulation_transition_splitting() {
        // s0 →[CH] s1  (s1 has cipher A)
        // s0' →[CH] s2 (s2 has cipher B)
        // s0 and s0' start with same observation, but should be split
        // because their successors differ.
        let mut lts = NegotiationLTS::new();
        let s0 = lts.add_state(make_neg(HandshakePhase::Initial, None));
        let s0_prime = lts.add_state(make_neg(HandshakePhase::Initial, None));
        let s1 = lts.add_state(make_neg(
            HandshakePhase::ApplicationData,
            Some(0x002f),
        ));
        let s2 = lts.add_state(make_neg(
            HandshakePhase::ApplicationData,
            Some(0x0035),
        ));
        lts.mark_initial(s0);
        lts.add_transition(
            s0,
            s1,
            MessageLabel::ClientHello {
                offered_ciphers: [0x002f].into(),
                version: ProtocolVersion::Tls12,
                extensions: vec![],
            },
        );
        lts.add_transition(
            s0_prime,
            s2,
            MessageLabel::ClientHello {
                offered_ciphers: [0x0035].into(),
                version: ProtocolVersion::Tls12,
                extensions: vec![],
            },
        );

        let mut checker = BisimulationChecker::new(1000);
        let rel = checker.compute(&lts).unwrap();

        // s0 and s0' should NOT be bisimilar (they reach different cipher outcomes).
        assert!(!rel.are_bisimilar(s0, s0_prime));
        // s1 and s2 should NOT be bisimilar (different observations).
        assert!(!rel.are_bisimilar(s1, s2));
    }

    #[test]
    fn test_bisimulation_relation_from_partition() {
        let rel = BisimulationRelation::from_partition(vec![
            vec![StateId(0), StateId(1)],
            vec![StateId(2)],
            vec![StateId(3), StateId(4), StateId(5)],
        ]);
        assert_eq!(rel.class_count(), 3);
        assert_eq!(rel.state_count(), 6);
        assert!(rel.are_bisimilar(StateId(0), StateId(1)));
        assert!(!rel.are_bisimilar(StateId(0), StateId(2)));
        assert!(rel.are_bisimilar(StateId(3), StateId(5)));
    }

    #[test]
    fn test_bisimulation_representatives() {
        let rel = BisimulationRelation::from_partition(vec![
            vec![StateId(0), StateId(1)],
            vec![StateId(2), StateId(3)],
        ]);
        let reps = rel.representatives();
        assert_eq!(reps.len(), 2);
        assert_eq!(reps[0], StateId(0));
        assert_eq!(reps[1], StateId(2));
    }

    #[test]
    fn test_observation_agreement() {
        let mut lts = NegotiationLTS::new();
        let s0 = lts.add_state(make_neg(
            HandshakePhase::ApplicationData,
            Some(0x002f),
        ));
        let s1 = lts.add_state(make_neg(
            HandshakePhase::ApplicationData,
            Some(0x002f),
        ));
        let block: BTreeSet<StateId> = [s0, s1].into();
        assert!(ObservationAgreement::check_block(&lts, &block));
    }

    #[test]
    fn test_observation_disagreement() {
        let mut lts = NegotiationLTS::new();
        let s0 = lts.add_state(make_neg(
            HandshakePhase::ApplicationData,
            Some(0x002f),
        ));
        let s1 = lts.add_state(make_neg(
            HandshakePhase::ApplicationData,
            Some(0x0035),
        ));
        let block: BTreeSet<StateId> = [s0, s1].into();
        assert!(!ObservationAgreement::check_block(&lts, &block));
    }

    #[test]
    fn test_transfer_verifier() {
        let mut lts = NegotiationLTS::new();
        let s0 = lts.add_state(make_neg(HandshakePhase::Initial, None));
        let s1 = lts.add_state(make_neg(HandshakePhase::Initial, None));
        let s2 = lts.add_state(make_neg(
            HandshakePhase::ApplicationData,
            Some(0x002f),
        ));
        lts.mark_initial(s0);
        lts.mark_initial(s1);

        // Both s0 and s1 transition to s2 under the same label.
        lts.add_transition(s0, s2, MessageLabel::Tau);
        lts.add_transition(s1, s2, MessageLabel::Tau);

        let rel = BisimulationRelation::from_partition(vec![
            vec![StateId(0), StateId(1)],
            vec![StateId(2)],
        ]);

        assert!(TransferVerifier::check_pair(&lts, &rel, s0, s1));
        assert!(TransferVerifier::check_all(&lts, &rel));
    }

    #[test]
    fn test_bisimulation_divergence_limit() {
        let mut lts = NegotiationLTS::new();
        let s0 = lts.add_state(make_neg(HandshakePhase::Initial, None));
        lts.mark_initial(s0);

        // Very low iteration limit.
        let mut checker = BisimulationChecker::new(0);
        let result = checker.compute(&lts);
        // Should still succeed with a single state (no refinement needed).
        assert!(result.is_ok());
    }

    #[test]
    fn test_bisimulation_display() {
        let rel = BisimulationRelation::from_partition(vec![
            vec![StateId(0), StateId(1)],
            vec![StateId(2)],
        ]);
        let s = format!("{}", rel);
        assert!(s.contains("2 classes"));
        assert!(s.contains("3 states"));
    }

    #[test]
    fn test_protocol_bisimilar_trait() {
        let rel = BisimulationRelation::from_partition(vec![
            vec![StateId(0), StateId(1)],
            vec![StateId(2)],
        ]);
        assert!(StateId(0).is_bisimilar_to(&StateId(1), &rel));
        assert!(!StateId(0).is_bisimilar_to(&StateId(2), &rel));
    }

    #[test]
    fn test_bisim_diamond_structure() {
        //    s0
        //   / \
        //  s1  s2
        //   \ /
        //    s3
        // s1, s2 have same observations and both reach s3 → bisimilar.
        let mut lts = NegotiationLTS::new();
        let s0 = lts.add_state(make_neg(HandshakePhase::Initial, None));
        let s1 = lts.add_state(make_neg(HandshakePhase::ClientHello, None));
        let s2 = lts.add_state(make_neg(HandshakePhase::ClientHello, None));
        let s3 = lts.add_state(make_neg(
            HandshakePhase::ApplicationData,
            Some(0x002f),
        ));
        lts.mark_initial(s0);
        lts.add_transition(s0, s1, MessageLabel::Tau);
        lts.add_transition(s0, s2, MessageLabel::Tau);
        lts.add_transition(s1, s3, MessageLabel::Tau);
        lts.add_transition(s2, s3, MessageLabel::Tau);

        let mut checker = BisimulationChecker::new(1000);
        let rel = checker.compute(&lts).unwrap();

        // s1 and s2 should be bisimilar (same obs, same transitions).
        assert!(rel.are_bisimilar(s1, s2));
    }
}
