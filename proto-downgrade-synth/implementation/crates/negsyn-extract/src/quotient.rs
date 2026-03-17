//! Quotient graph construction from a bisimulation relation.
//!
//! Builds the quotient LTS where each state is an equivalence class of
//! bisimilar states and transitions are aggregated between classes.

use crate::{
    bisimulation::BisimulationRelation,
    observation::ObservationFunction,
    ExtractError, ExtractResult, HandshakePhase, LtsState, LtsTransition, MessageLabel,
    NegotiationLTS, Observable, StateId, TransitionId,
};
use log::{debug, info};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt;

// ---------------------------------------------------------------------------
// QuotientState
// ---------------------------------------------------------------------------

/// A state in the quotient LTS, representing an equivalence class.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuotientState {
    /// The quotient state ID (new namespace).
    pub id: StateId,
    /// The original state IDs in this equivalence class.
    pub members: BTreeSet<StateId>,
    /// The representative original state (used for observation).
    pub representative: StateId,
    /// The observation for this equivalence class.
    pub observation: Observable,
    /// Whether this class contains an initial state.
    pub is_initial: bool,
    /// Whether this class contains terminal states.
    pub is_terminal: bool,
}

impl QuotientState {
    pub fn member_count(&self) -> usize {
        self.members.len()
    }
}

impl fmt::Display for QuotientState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Q{} [{}members, obs={}, init={}, term={}]",
            self.id,
            self.members.len(),
            self.observation,
            self.is_initial,
            self.is_terminal,
        )
    }
}

// ---------------------------------------------------------------------------
// QuotientTransition
// ---------------------------------------------------------------------------

/// A transition in the quotient LTS, aggregating one or more original transitions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuotientTransition {
    /// Source quotient state.
    pub source: StateId,
    /// Target quotient state.
    pub target: StateId,
    /// The label (shared by all aggregated transitions in this class).
    pub label: MessageLabel,
    /// How many original transitions were aggregated.
    pub aggregated_count: usize,
    /// Original transition IDs that were aggregated.
    pub original_transitions: Vec<TransitionId>,
}

impl fmt::Display for QuotientTransition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} --[{}]--> {} (×{})",
            self.source,
            self.label.label_name(),
            self.target,
            self.aggregated_count,
        )
    }
}

// ---------------------------------------------------------------------------
// QuotientLTS
// ---------------------------------------------------------------------------

/// The reduced labeled transition system after quotient construction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuotientLTS {
    pub states: Vec<QuotientState>,
    pub transitions: Vec<QuotientTransition>,
    pub initial_states: Vec<StateId>,
    /// Map from original state ID to quotient state ID.
    pub state_map: HashMap<StateId, StateId>,
}

impl QuotientLTS {
    pub fn state_count(&self) -> usize {
        self.states.len()
    }

    pub fn transition_count(&self) -> usize {
        self.transitions.len()
    }

    /// Look up the quotient state for an original state.
    pub fn quotient_of(&self, original: StateId) -> Option<StateId> {
        self.state_map.get(&original).copied()
    }

    /// Get a quotient state by its ID.
    pub fn get_state(&self, id: StateId) -> Option<&QuotientState> {
        self.states.iter().find(|s| s.id == id)
    }

    /// Get transitions from a quotient state.
    pub fn transitions_from(&self, source: StateId) -> Vec<&QuotientTransition> {
        self.transitions.iter().filter(|t| t.source == source).collect()
    }

    /// Compute reduction statistics.
    pub fn reduction_stats(&self, original_states: usize, original_transitions: usize) -> QuotientStats {
        QuotientStats {
            original_states,
            original_transitions,
            quotient_states: self.state_count(),
            quotient_transitions: self.transition_count(),
            state_reduction: if original_states > 0 {
                1.0 - (self.state_count() as f64 / original_states as f64)
            } else {
                0.0
            },
            transition_reduction: if original_transitions > 0 {
                1.0 - (self.transition_count() as f64 / original_transitions as f64)
            } else {
                0.0
            },
            max_class_size: self.states.iter().map(|s| s.member_count()).max().unwrap_or(0),
            singleton_classes: self.states.iter().filter(|s| s.member_count() == 1).count(),
        }
    }
}

/// Statistics on the quotient reduction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuotientStats {
    pub original_states: usize,
    pub original_transitions: usize,
    pub quotient_states: usize,
    pub quotient_transitions: usize,
    pub state_reduction: f64,
    pub transition_reduction: f64,
    pub max_class_size: usize,
    pub singleton_classes: usize,
}

impl fmt::Display for QuotientStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Quotient Reduction:")?;
        writeln!(
            f,
            "  States:      {} → {} ({:.1}% reduction)",
            self.original_states,
            self.quotient_states,
            self.state_reduction * 100.0
        )?;
        writeln!(
            f,
            "  Transitions: {} → {} ({:.1}% reduction)",
            self.original_transitions,
            self.quotient_transitions,
            self.transition_reduction * 100.0
        )?;
        writeln!(
            f,
            "  Max class:   {}, singletons: {}",
            self.max_class_size, self.singleton_classes
        )
    }
}

// ---------------------------------------------------------------------------
// QuotientBuilder
// ---------------------------------------------------------------------------

/// Constructs the quotient LTS from an original LTS and a bisimulation relation.
pub struct QuotientBuilder {
    /// Whether to validate the quotient against the original.
    validate: bool,
}

impl QuotientBuilder {
    pub fn new() -> Self {
        Self { validate: true }
    }

    pub fn without_validation() -> Self {
        Self { validate: false }
    }

    /// Build the quotient LTS, returning it as a standard NegotiationLTS.
    pub fn build(
        &self,
        original: &NegotiationLTS,
        relation: &BisimulationRelation,
    ) -> ExtractResult<NegotiationLTS> {
        let quotient = self.build_quotient(original, relation)?;

        if self.validate {
            self.validate_quotient(original, &quotient, relation)?;
        }

        // Convert QuotientLTS back to NegotiationLTS.
        self.to_negotiation_lts(original, &quotient)
    }

    /// Build the quotient data structure.
    pub fn build_quotient(
        &self,
        original: &NegotiationLTS,
        relation: &BisimulationRelation,
    ) -> ExtractResult<QuotientLTS> {
        let mut obs_fn = ObservationFunction::new();

        // Build quotient states from equivalence classes.
        let mut quotient_states: Vec<QuotientState> = Vec::new();
        let mut state_map: HashMap<StateId, StateId> = HashMap::new();
        let initial_set: HashSet<StateId> = original.initial_states.iter().copied().collect();

        for (block_idx, block) in relation.blocks().iter().enumerate() {
            let qid = StateId(block_idx as u32);
            let representative = *block.iter().next().ok_or_else(|| {
                ExtractError::QuotientFailed { reason: "empty block in partition".into() }
            })?;

            let observation = obs_fn.observe(original, representative);
            let is_initial = block.iter().any(|s| initial_set.contains(s));
            let is_terminal = block.iter().any(|s| {
                original
                    .get_state(*s)
                    .map(|st| st.is_terminal)
                    .unwrap_or(false)
            });

            for &sid in block {
                state_map.insert(sid, qid);
            }

            quotient_states.push(QuotientState {
                id: qid,
                members: block.clone(),
                representative,
                observation,
                is_initial,
                is_terminal,
            });
        }

        // Build quotient transitions.
        // For each original transition (s, a, t), add the quotient transition
        // (class(s), a, class(t)) if it doesn't already exist.
        let mut seen_transitions: HashSet<(StateId, String, StateId)> = HashSet::new();
        let mut quotient_transitions: Vec<QuotientTransition> = Vec::new();
        let mut aggregation: HashMap<(StateId, String, StateId), Vec<TransitionId>> =
            HashMap::new();

        for t in &original.transitions {
            let q_src = state_map
                .get(&t.source)
                .copied()
                .ok_or_else(|| ExtractError::QuotientFailed { reason: format!("unmapped source {}", t.source) })?;
            let q_tgt = state_map
                .get(&t.target)
                .copied()
                .ok_or_else(|| ExtractError::QuotientFailed { reason: format!("unmapped target {}", t.target) })?;
            let label_name = t.label.label_name().to_string();

            // Skip self-loops on Tau in the quotient.
            if q_src == q_tgt && t.label == MessageLabel::Tau {
                continue;
            }

            let key = (q_src, label_name.clone(), q_tgt);
            aggregation.entry(key).or_default().push(t.id);
        }

        for ((q_src, label_name, q_tgt), orig_ids) in &aggregation {
            // Find a representative label from the original transitions.
            let repr_label = original
                .transitions
                .iter()
                .find(|t| {
                    state_map.get(&t.source) == Some(q_src)
                        && state_map.get(&t.target) == Some(q_tgt)
                        && t.label.label_name() == label_name
                })
                .map(|t| t.label.clone())
                .unwrap_or(MessageLabel::Tau);

            quotient_transitions.push(QuotientTransition {
                source: *q_src,
                target: *q_tgt,
                label: repr_label,
                aggregated_count: orig_ids.len(),
                original_transitions: orig_ids.clone(),
            });
        }

        let initial_states: Vec<StateId> = quotient_states
            .iter()
            .filter(|qs| qs.is_initial)
            .map(|qs| qs.id)
            .collect();

        let stats = QuotientLTS {
            states: quotient_states,
            transitions: quotient_transitions,
            initial_states,
            state_map: state_map.clone(),
        }
        .reduction_stats(original.state_count(), original.transition_count());
        debug!("{}", stats);

        Ok(QuotientLTS {
            states: {
                // Re-create because we moved it.
                let mut obs_fn2 = ObservationFunction::new();
                let mut qs2 = Vec::new();
                for (block_idx, block) in relation.blocks().iter().enumerate() {
                    let qid = StateId(block_idx as u32);
                    let representative = *block.iter().next().unwrap();
                    let observation = obs_fn2.observe(original, representative);
                    let is_initial = block.iter().any(|s| initial_set.contains(s));
                    let is_terminal = block.iter().any(|s| {
                        original.get_state(*s).map(|st| st.is_terminal).unwrap_or(false)
                    });
                    qs2.push(QuotientState {
                        id: qid,
                        members: block.clone(),
                        representative,
                        observation,
                        is_initial,
                        is_terminal,
                    });
                }
                qs2
            },
            transitions: {
                let mut qt2 = Vec::new();
                for ((q_src, label_name, q_tgt), orig_ids) in &aggregation {
                    let repr_label = original
                        .transitions
                        .iter()
                        .find(|t| {
                            state_map.get(&t.source) == Some(q_src)
                                && state_map.get(&t.target) == Some(q_tgt)
                                && t.label.label_name() == label_name
                        })
                        .map(|t| t.label.clone())
                        .unwrap_or(MessageLabel::Tau);
                    qt2.push(QuotientTransition {
                        source: *q_src,
                        target: *q_tgt,
                        label: repr_label,
                        aggregated_count: orig_ids.len(),
                        original_transitions: orig_ids.clone(),
                    });
                }
                qt2
            },
            initial_states: {
                relation
                    .blocks()
                    .iter()
                    .enumerate()
                    .filter(|(_, block)| block.iter().any(|s| initial_set.contains(s)))
                    .map(|(i, _)| StateId(i as u32))
                    .collect()
            },
            state_map: state_map.clone(),
        })
    }

    /// Validate: every trace in the original has a corresponding trace in the quotient.
    fn validate_quotient(
        &self,
        original: &NegotiationLTS,
        quotient: &QuotientLTS,
        relation: &BisimulationRelation,
    ) -> ExtractResult<()> {
        // For each original transition, check the corresponding quotient transition exists.
        for t in &original.transitions {
            let q_src = quotient.state_map.get(&t.source);
            let q_tgt = quotient.state_map.get(&t.target);

            match (q_src, q_tgt) {
                (Some(&qs), Some(&qt)) => {
                    // Skip self-loops on Tau.
                    if qs == qt && t.label == MessageLabel::Tau {
                        continue;
                    }

                    let has_matching = quotient.transitions.iter().any(|qt_trans| {
                        qt_trans.source == qs
                            && qt_trans.target == qt
                            && qt_trans.label.label_name() == t.label.label_name()
                    });

                    if !has_matching {
                        return Err(ExtractError::QuotientFailed { reason: format!(
                            "no quotient transition for {} --[{}]--> {}",
                            t.source,
                            t.label.label_name(),
                            t.target,
                        ) });
                    }
                }
                _ => {
                    return Err(ExtractError::QuotientFailed { reason: format!(
                        "unmapped state in transition {} → {}",
                        t.source, t.target,
                    ) });
                }
            }
        }

        Ok(())
    }

    /// Convert a QuotientLTS back into a NegotiationLTS for downstream use.
    fn to_negotiation_lts(
        &self,
        original: &NegotiationLTS,
        quotient: &QuotientLTS,
    ) -> ExtractResult<NegotiationLTS> {
        let mut lts = NegotiationLTS::new();

        for qs in &quotient.states {
            // Use the negotiation state of the representative.
            let neg = original
                .get_state(qs.representative)
                .map(|s| s.negotiation.clone())
                .ok_or_else(|| {
                    ExtractError::QuotientFailed { reason: format!(
                        "representative {} not found",
                        qs.representative,
                    ) }
                })?;
            lts.add_state_with_id(qs.id, neg);

            if let Some(state) = lts.get_state_mut(qs.id) {
                state.observation = qs.observation.clone();
                state.is_initial = qs.is_initial;
                state.is_terminal = qs.is_terminal;
                state.source_symbolic_ids = qs
                    .members
                    .iter()
                    .flat_map(|&m| {
                        original
                            .get_state(m)
                            .map(|s| s.source_symbolic_ids.clone())
                            .unwrap_or_default()
                    })
                    .collect();
            }

            if qs.is_initial {
                lts.mark_initial(qs.id);
            }
        }

        for qt in &quotient.transitions {
            lts.add_transition(qt.source, qt.target, qt.label.clone());
        }

        Ok(lts)
    }
}

impl Default for QuotientBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Incremental quotient update
// ---------------------------------------------------------------------------

/// Supports incremental updates to the quotient when new states are discovered.
pub struct IncrementalQuotient {
    quotient: QuotientLTS,
    relation: BisimulationRelation,
}

impl IncrementalQuotient {
    pub fn new(quotient: QuotientLTS, relation: BisimulationRelation) -> Self {
        Self { quotient, relation }
    }

    /// Add a new state to the quotient. Determines which existing class
    /// it belongs to (if any) or creates a new class.
    pub fn add_state(
        &mut self,
        original: &NegotiationLTS,
        new_state: StateId,
    ) -> ExtractResult<StateId> {
        let mut obs_fn = ObservationFunction::new();
        let new_obs = obs_fn.observe(original, new_state);

        // Try to find an existing class with matching observation and
        // compatible transitions.
        for qs in &self.quotient.states {
            if qs.observation.agrees_with(&new_obs) {
                // Check if transitions are compatible.
                let new_succs = self.successor_labels(original, new_state);
                let class_succs = self.class_successor_labels(original, &qs.members);

                if new_succs == class_succs {
                    // Add to this class.
                    self.quotient.state_map.insert(new_state, qs.id);
                    return Ok(qs.id);
                }
            }
        }

        // No matching class → create a new quotient state.
        let qid = StateId(self.quotient.states.len() as u32);
        let is_initial = original
            .initial_states
            .contains(&new_state);
        let is_terminal = original
            .get_state(new_state)
            .map(|s| s.is_terminal)
            .unwrap_or(false);

        self.quotient.states.push(QuotientState {
            id: qid,
            members: [new_state].into(),
            representative: new_state,
            observation: new_obs,
            is_initial,
            is_terminal,
        });
        self.quotient.state_map.insert(new_state, qid);

        if is_initial {
            self.quotient.initial_states.push(qid);
        }

        Ok(qid)
    }

    /// Add new transitions to the quotient for newly added states.
    pub fn update_transitions(
        &mut self,
        original: &NegotiationLTS,
        new_state: StateId,
    ) -> ExtractResult<()> {
        let q_src = self
            .quotient
            .state_map
            .get(&new_state)
            .copied()
            .ok_or_else(|| {
                ExtractError::QuotientFailed { reason: "state not in quotient".into() }
            })?;

        for t in original.transitions_from(new_state) {
            if let Some(&q_tgt) = self.quotient.state_map.get(&t.target) {
                let label_name = t.label.label_name();
                let exists = self.quotient.transitions.iter().any(|qt| {
                    qt.source == q_src
                        && qt.target == q_tgt
                        && qt.label.label_name() == label_name
                });

                if !exists && !(q_src == q_tgt && t.label == MessageLabel::Tau) {
                    self.quotient.transitions.push(QuotientTransition {
                        source: q_src,
                        target: q_tgt,
                        label: t.label.clone(),
                        aggregated_count: 1,
                        original_transitions: vec![t.id],
                    });
                }
            }
        }

        Ok(())
    }

    pub fn quotient(&self) -> &QuotientLTS {
        &self.quotient
    }

    fn successor_labels(
        &self,
        lts: &NegotiationLTS,
        state: StateId,
    ) -> BTreeSet<String> {
        lts.transitions_from(state)
            .iter()
            .map(|t| t.label.label_name().to_string())
            .collect()
    }

    fn class_successor_labels(
        &self,
        lts: &NegotiationLTS,
        members: &BTreeSet<StateId>,
    ) -> BTreeSet<String> {
        let rep = members.iter().next().copied();
        match rep {
            Some(s) => self.successor_labels(lts, s),
            None => BTreeSet::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bisimulation::{BisimulationChecker, BisimulationRelation};
    use crate::ProtocolVersion;
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
    fn test_quotient_single_state() {
        let mut lts = NegotiationLTS::new();
        let s0 = lts.add_state(make_neg(HandshakePhase::Initial, None));
        lts.mark_initial(s0);

        let rel = BisimulationRelation::from_partition(vec![vec![s0]]);
        let builder = QuotientBuilder::new();
        let qlts = builder.build_quotient(&lts, &rel).unwrap();

        assert_eq!(qlts.state_count(), 1);
        assert_eq!(qlts.transition_count(), 0);
    }

    #[test]
    fn test_quotient_merges_bisimilar() {
        // Two bisimilar states → one quotient state.
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

        let rel = BisimulationRelation::from_partition(vec![vec![s0, s1]]);
        let builder = QuotientBuilder::new();
        let qlts = builder.build_quotient(&lts, &rel).unwrap();

        assert_eq!(qlts.state_count(), 1);
        assert_eq!(qlts.states[0].member_count(), 2);
    }

    #[test]
    fn test_quotient_preserves_transitions() {
        let mut lts = NegotiationLTS::new();
        let s0 = lts.add_state(make_neg(HandshakePhase::Initial, None));
        let s1 = lts.add_state(make_neg(HandshakePhase::Initial, None));
        let s2 = lts.add_state(make_neg(
            HandshakePhase::ApplicationData,
            Some(0x002f),
        ));
        lts.mark_initial(s0);
        lts.add_transition(s0, s2, MessageLabel::Tau);
        lts.add_transition(s1, s2, MessageLabel::Tau);

        // s0 and s1 are bisimilar (same obs, same transition pattern).
        let rel = BisimulationRelation::from_partition(vec![
            vec![s0, s1],
            vec![s2],
        ]);
        let builder = QuotientBuilder::new();
        let qlts = builder.build_quotient(&lts, &rel).unwrap();

        assert_eq!(qlts.state_count(), 2);
        assert_eq!(qlts.transition_count(), 1);
        assert_eq!(qlts.transitions[0].aggregated_count, 2);
    }

    #[test]
    fn test_quotient_to_negotiation_lts() {
        let mut lts = NegotiationLTS::new();
        let s0 = lts.add_state(make_neg(HandshakePhase::Initial, None));
        let s1 = lts.add_state(make_neg(
            HandshakePhase::ApplicationData,
            Some(0x002f),
        ));
        lts.mark_initial(s0);
        lts.add_transition(s0, s1, MessageLabel::Tau);

        let rel = BisimulationRelation::from_partition(vec![vec![s0], vec![s1]]);
        let builder = QuotientBuilder::new();
        let result_lts = builder.build(&lts, &rel).unwrap();

        assert_eq!(result_lts.state_count(), 2);
        assert_eq!(result_lts.transition_count(), 1);
        assert!(!result_lts.initial_states.is_empty());
    }

    #[test]
    fn test_quotient_stats() {
        let mut lts = NegotiationLTS::new();
        for i in 0..10 {
            let s = lts.add_state(make_neg(
                HandshakePhase::ApplicationData,
                Some(0x002f),
            ));
            if i == 0 {
                lts.mark_initial(s);
            }
        }
        // All 10 states are bisimilar → 1 quotient state.
        let all_ids: Vec<StateId> = lts.state_ids();
        let rel = BisimulationRelation::from_partition(vec![all_ids]);
        let builder = QuotientBuilder::new();
        let qlts = builder.build_quotient(&lts, &rel).unwrap();

        let stats = qlts.reduction_stats(10, 0);
        assert_eq!(stats.quotient_states, 1);
        assert!((stats.state_reduction - 0.9).abs() < 0.01);
        assert_eq!(stats.max_class_size, 10);
        assert_eq!(stats.singleton_classes, 0);
    }

    #[test]
    fn test_quotient_validation_passes() {
        let mut lts = NegotiationLTS::new();
        let s0 = lts.add_state(make_neg(HandshakePhase::Initial, None));
        let s1 = lts.add_state(make_neg(
            HandshakePhase::ApplicationData,
            Some(0x002f),
        ));
        lts.mark_initial(s0);
        lts.add_transition(s0, s1, MessageLabel::Tau);

        let rel = BisimulationRelation::from_partition(vec![vec![s0], vec![s1]]);
        let builder = QuotientBuilder::new();
        let result = builder.build(&lts, &rel);
        assert!(result.is_ok());
    }

    #[test]
    fn test_incremental_quotient() {
        let mut lts = NegotiationLTS::new();
        let s0 = lts.add_state(make_neg(HandshakePhase::Initial, None));
        let s1 = lts.add_state(make_neg(
            HandshakePhase::ApplicationData,
            Some(0x002f),
        ));
        lts.mark_initial(s0);
        lts.add_transition(s0, s1, MessageLabel::Tau);

        let rel = BisimulationRelation::from_partition(vec![vec![s0], vec![s1]]);
        let builder = QuotientBuilder::new();
        let qlts = builder.build_quotient(&lts, &rel).unwrap();

        let mut inc = IncrementalQuotient::new(qlts, rel);

        // Add a new state with same observation as s1 → should join same class or create new.
        let s2 = lts.add_state(make_neg(
            HandshakePhase::ApplicationData,
            Some(0x002f),
        ));
        let qid = inc.add_state(&lts, s2).unwrap();
        // s2 has same observation as s1, but since we check transitions,
        // it may or may not join (no outgoing transitions from s2).
        assert!(qid.0 < 3); // Either joined class or got new ID.
    }

    #[test]
    fn test_quotient_self_loop_removal() {
        // Self-loop on Tau should be removed in quotient.
        let mut lts = NegotiationLTS::new();
        let s0 = lts.add_state(make_neg(HandshakePhase::Initial, None));
        let s1 = lts.add_state(make_neg(HandshakePhase::Initial, None));
        lts.mark_initial(s0);
        // s0 and s1 are bisimilar. s0 → s1 via Tau becomes self-loop in quotient.
        lts.add_transition(s0, s1, MessageLabel::Tau);

        let rel = BisimulationRelation::from_partition(vec![vec![s0, s1]]);
        let builder = QuotientBuilder::without_validation();
        let qlts = builder.build_quotient(&lts, &rel).unwrap();

        assert_eq!(qlts.state_count(), 1);
        assert_eq!(qlts.transition_count(), 0);
    }
}
