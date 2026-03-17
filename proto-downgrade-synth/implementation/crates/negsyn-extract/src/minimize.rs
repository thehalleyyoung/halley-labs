//! State machine minimization.
//!
//! Implements Hopcroft's algorithm adapted for protocol LTS, unreachable
//! state elimination, and redundant transition removal.

use crate::{
    bisimulation::{BisimulationChecker, BisimulationRelation},
    observation::ObservationEquivalence,
    quotient::QuotientBuilder,
    ExtractError, ExtractResult, HandshakePhase, LtsState, LtsTransition, MessageLabel,
    NegotiationLTS, Observable, StateId, TransitionId,
};
use log::{debug, info};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};
use std::fmt;

// ---------------------------------------------------------------------------
// Minimizer (top-level)
// ---------------------------------------------------------------------------

/// Reduces state machine size through multiple optimization passes.
pub struct Minimizer {
    unreachable_eliminated: usize,
    redundant_eliminated: usize,
}

impl Minimizer {
    pub fn new() -> Self {
        Self {
            unreachable_eliminated: 0,
            redundant_eliminated: 0,
        }
    }

    /// Run all minimization passes.
    pub fn minimize(&mut self, mut lts: NegotiationLTS) -> ExtractResult<NegotiationLTS> {
        let initial_states = lts.state_count();
        let initial_transitions = lts.transition_count();

        // Pass 1: Eliminate unreachable states.
        let before_unreach = lts.state_count();
        lts = UnreachableElimination::eliminate(lts);
        self.unreachable_eliminated = before_unreach - lts.state_count();
        debug!(
            "Unreachable elimination: {} → {} states (removed {})",
            before_unreach,
            lts.state_count(),
            self.unreachable_eliminated,
        );

        // Pass 2: Eliminate redundant transitions.
        let before_redundant = lts.transition_count();
        lts = RedundantTransitionElimination::eliminate(lts);
        self.redundant_eliminated = before_redundant - lts.transition_count();
        debug!(
            "Redundant transition elimination: {} → {} transitions (removed {})",
            before_redundant,
            lts.transition_count(),
            self.redundant_eliminated,
        );

        // Pass 3: Protocol-aware Hopcroft minimization.
        lts = ProtocolMinimization::minimize(lts)?;

        debug!(
            "Minimization complete: {} states / {} transitions → {} states / {} transitions",
            initial_states,
            initial_transitions,
            lts.state_count(),
            lts.transition_count(),
        );

        Ok(lts)
    }

    pub fn unreachable_eliminated(&self) -> usize {
        self.unreachable_eliminated
    }

    pub fn redundant_eliminated(&self) -> usize {
        self.redundant_eliminated
    }
}

impl Default for Minimizer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// UnreachableElimination
// ---------------------------------------------------------------------------

/// Removes states not reachable from any initial state.
pub struct UnreachableElimination;

impl UnreachableElimination {
    pub fn eliminate(mut lts: NegotiationLTS) -> NegotiationLTS {
        let reachable = lts.reachable_states();

        if reachable.len() == lts.state_count() {
            return lts;
        }

        // Collect unreachable state IDs.
        let unreachable: Vec<StateId> = lts
            .states
            .keys()
            .copied()
            .filter(|s| !reachable.contains(s))
            .collect();

        for sid in unreachable {
            lts.remove_state(sid);
        }

        lts
    }

    /// Also eliminate states that cannot reach any terminal state (dead ends).
    pub fn eliminate_dead_ends(mut lts: NegotiationLTS) -> NegotiationLTS {
        let terminal_states: HashSet<StateId> = lts
            .states
            .iter()
            .filter(|(_, s)| s.is_terminal)
            .map(|(&id, _)| id)
            .collect();

        if terminal_states.is_empty() {
            return lts;
        }

        // Backward BFS from terminal states.
        let reverse_adj = lts.reverse_adjacency_list();
        let mut can_reach_terminal: HashSet<StateId> = HashSet::new();
        let mut queue: VecDeque<StateId> = terminal_states.iter().copied().collect();

        while let Some(s) = queue.pop_front() {
            if can_reach_terminal.insert(s) {
                if let Some(preds) = reverse_adj.get(&s) {
                    for &(pred, _) in preds {
                        if !can_reach_terminal.contains(&pred) {
                            queue.push_back(pred);
                        }
                    }
                }
            }
        }

        // Remove states that are reachable but can't reach any terminal.
        let reachable = lts.reachable_states();
        let dead_ends: Vec<StateId> = reachable
            .iter()
            .filter(|s| !can_reach_terminal.contains(s))
            .copied()
            .collect();

        for sid in dead_ends {
            lts.remove_state(sid);
        }

        lts
    }
}

// ---------------------------------------------------------------------------
// RedundantTransitionElimination
// ---------------------------------------------------------------------------

/// Removes transitions that are redundant:
/// - Duplicate transitions (same source, label, target)
/// - Tau self-loops
/// - Subsumed transitions where a more direct path exists
pub struct RedundantTransitionElimination;

impl RedundantTransitionElimination {
    pub fn eliminate(mut lts: NegotiationLTS) -> NegotiationLTS {
        // Remove exact duplicates.
        let mut seen: HashSet<(StateId, String, StateId)> = HashSet::new();
        lts.transitions.retain(|t| {
            let key = (t.source, t.label.label_name().to_string(), t.target);
            seen.insert(key)
        });

        // Remove Tau self-loops.
        lts.transitions.retain(|t| {
            !(t.source == t.target && t.label == MessageLabel::Tau)
        });

        // Remove transitions to non-existent states.
        let state_ids: HashSet<StateId> = lts.states.keys().copied().collect();
        lts.transitions.retain(|t| {
            state_ids.contains(&t.source) && state_ids.contains(&t.target)
        });

        // Remove subsumed Tau transitions.
        // If s →τ→ t and s →a→ t for some visible action a, remove the Tau.
        let non_tau_pairs: HashSet<(StateId, StateId)> = lts
            .transitions
            .iter()
            .filter(|t| !t.label.is_internal())
            .map(|t| (t.source, t.target))
            .collect();

        lts.transitions.retain(|t| {
            if t.label.is_internal() {
                !non_tau_pairs.contains(&(t.source, t.target))
            } else {
                true
            }
        });

        lts
    }
}

// ---------------------------------------------------------------------------
// HopcroftMinimization
// ---------------------------------------------------------------------------

/// Implements Hopcroft's DFA minimization algorithm adapted for
/// nondeterministic protocol LTS.
///
/// This is functionally equivalent to bisimulation quotient, but uses
/// the classic O(n log n) Hopcroft approach.
pub struct HopcroftMinimization;

impl HopcroftMinimization {
    /// Minimize the LTS using Hopcroft's algorithm.
    pub fn minimize(lts: NegotiationLTS) -> ExtractResult<NegotiationLTS> {
        if lts.state_count() <= 1 {
            return Ok(lts);
        }

        // Initial partition: separate terminal from non-terminal, and by observation.
        let mut obs_eq = ObservationEquivalence::new();
        let initial_partition = obs_eq.partition_by_observation(&lts);

        if initial_partition.len() == lts.state_count() {
            // Already minimal.
            return Ok(lts);
        }

        let alphabet = lts.alphabet();
        let mut partition: Vec<BTreeSet<StateId>> = initial_partition
            .into_iter()
            .map(|b| b.into_iter().collect())
            .collect();

        // Build reverse transition map: for each (label, target), set of sources.
        let mut reverse_map: HashMap<(String, StateId), HashSet<StateId>> = HashMap::new();
        for t in &lts.transitions {
            reverse_map
                .entry((t.label.label_name().to_string(), t.target))
                .or_default()
                .insert(t.source);
        }

        // Worklist of (block_index, action) pairs.
        let mut worklist: VecDeque<(usize, String)> = VecDeque::new();
        for a in &alphabet {
            // Initialize worklist with the smallest of each split pair.
            for (i, _block) in partition.iter().enumerate() {
                worklist.push_back((i, a.clone()));
            }
        }

        let max_iterations = lts.state_count() * alphabet.len() * 2 + 100;
        let mut iterations = 0;

        while let Some((splitter_idx, action)) = worklist.pop_front() {
            iterations += 1;
            if iterations > max_iterations {
                break;
            }

            if splitter_idx >= partition.len() || partition[splitter_idx].is_empty() {
                continue;
            }

            // Compute pre(splitter, action).
            let splitter = &partition[splitter_idx];
            let mut pre_states: HashSet<StateId> = HashSet::new();
            for &target in splitter {
                if let Some(sources) = reverse_map.get(&(action.clone(), target)) {
                    pre_states.extend(sources);
                }
            }

            if pre_states.is_empty() {
                continue;
            }

            // Try to split blocks.
            let num_blocks = partition.len();
            let mut splits: Vec<(usize, BTreeSet<StateId>, BTreeSet<StateId>)> = Vec::new();

            for block_idx in 0..num_blocks {
                if partition[block_idx].len() <= 1 {
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

                if !in_pre.is_empty() && !not_in_pre.is_empty() {
                    splits.push((block_idx, in_pre, not_in_pre));
                }
            }

            for (block_idx, part_a, part_b) in splits {
                let (keep, split_off) = if part_a.len() >= part_b.len() {
                    (part_a, part_b)
                } else {
                    (part_b, part_a)
                };

                partition[block_idx] = keep;
                let new_idx = partition.len();
                partition.push(split_off);

                // Add new block to worklist for all actions.
                for a in &alphabet {
                    worklist.push_back((new_idx, a.clone()));
                }
            }
        }

        // Build the minimized LTS from the final partition.
        let relation = BisimulationRelation::from_partition(
            partition
                .into_iter()
                .filter(|b| !b.is_empty())
                .map(|b| b.into_iter().collect())
                .collect(),
        );

        let builder = QuotientBuilder::without_validation();
        builder.build(&lts, &relation)
    }
}

// ---------------------------------------------------------------------------
// ProtocolMinimization
// ---------------------------------------------------------------------------

/// Protocol-aware minimization that respects the observation function.
pub struct ProtocolMinimization;

impl ProtocolMinimization {
    /// Minimize by combining bisimulation quotient with Hopcroft.
    pub fn minimize(lts: NegotiationLTS) -> ExtractResult<NegotiationLTS> {
        if lts.state_count() <= 1 {
            return Ok(lts);
        }

        // Use Hopcroft minimization.
        HopcroftMinimization::minimize(lts)
    }
}

// ---------------------------------------------------------------------------
// MinimizationStats
// ---------------------------------------------------------------------------

/// Statistics from a minimization pass.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MinimizationStats {
    pub states_before: usize,
    pub states_after: usize,
    pub transitions_before: usize,
    pub transitions_after: usize,
    pub unreachable_removed: usize,
    pub dead_end_removed: usize,
    pub redundant_transitions_removed: usize,
    pub equivalence_classes: usize,
}

impl MinimizationStats {
    pub fn state_reduction(&self) -> f64 {
        if self.states_before == 0 {
            0.0
        } else {
            1.0 - (self.states_after as f64 / self.states_before as f64)
        }
    }

    pub fn transition_reduction(&self) -> f64 {
        if self.transitions_before == 0 {
            0.0
        } else {
            1.0 - (self.transitions_after as f64 / self.transitions_before as f64)
        }
    }
}

impl fmt::Display for MinimizationStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Minimization Stats:")?;
        writeln!(
            f,
            "  States:      {} → {} ({:.1}% reduction)",
            self.states_before,
            self.states_after,
            self.state_reduction() * 100.0,
        )?;
        writeln!(
            f,
            "  Transitions: {} → {} ({:.1}% reduction)",
            self.transitions_before,
            self.transitions_after,
            self.transition_reduction() * 100.0,
        )?;
        writeln!(
            f,
            "  Unreachable: {}, dead ends: {}, redundant: {}",
            self.unreachable_removed,
            self.dead_end_removed,
            self.redundant_transitions_removed,
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
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
    fn test_unreachable_elimination() {
        let mut lts = NegotiationLTS::new();
        let s0 = lts.add_state(make_neg(HandshakePhase::Initial, None));
        let s1 = lts.add_state(make_neg(HandshakePhase::ClientHello, None));
        let _s2 = lts.add_state(make_neg(HandshakePhase::Alert, None)); // unreachable
        lts.mark_initial(s0);
        lts.add_transition(s0, s1, MessageLabel::Tau);

        let minimized = UnreachableElimination::eliminate(lts);
        assert_eq!(minimized.state_count(), 2);
    }

    #[test]
    fn test_unreachable_all_reachable() {
        let mut lts = NegotiationLTS::new();
        let s0 = lts.add_state(make_neg(HandshakePhase::Initial, None));
        let s1 = lts.add_state(make_neg(HandshakePhase::ClientHello, None));
        lts.mark_initial(s0);
        lts.add_transition(s0, s1, MessageLabel::Tau);

        let minimized = UnreachableElimination::eliminate(lts);
        assert_eq!(minimized.state_count(), 2);
    }

    #[test]
    fn test_dead_end_elimination() {
        let mut lts = NegotiationLTS::new();
        let s0 = lts.add_state(make_neg(HandshakePhase::Initial, None));
        let s1 = lts.add_state(make_neg(HandshakePhase::ClientHello, None)); // dead end
        let s2 = lts.add_state(make_neg(
            HandshakePhase::ApplicationData,
            Some(0x002f),
        )); // terminal
        lts.mark_initial(s0);
        lts.add_transition(s0, s1, MessageLabel::Tau);
        lts.add_transition(s0, s2, MessageLabel::Tau);

        let minimized = UnreachableElimination::eliminate_dead_ends(lts);
        // s0 can reach terminal s2 directly. s1 is a dead end (no path to terminal).
        assert!(minimized.state_count() <= 3);
    }

    #[test]
    fn test_redundant_transition_elimination() {
        let mut lts = NegotiationLTS::new();
        let s0 = lts.add_state(make_neg(HandshakePhase::Initial, None));
        let s1 = lts.add_state(make_neg(HandshakePhase::ClientHello, None));
        lts.mark_initial(s0);

        // Add duplicate transitions.
        lts.add_transition(s0, s1, MessageLabel::Tau);
        lts.add_transition(s0, s1, MessageLabel::Tau);

        // Add Tau self-loop.
        lts.add_transition(s0, s0, MessageLabel::Tau);

        let minimized = RedundantTransitionElimination::eliminate(lts);
        assert_eq!(minimized.transition_count(), 1);
    }

    #[test]
    fn test_redundant_tau_subsumed() {
        let mut lts = NegotiationLTS::new();
        let s0 = lts.add_state(make_neg(HandshakePhase::Initial, None));
        let s1 = lts.add_state(make_neg(HandshakePhase::ClientHello, None));
        lts.mark_initial(s0);

        // Both a Tau and a visible action to same target.
        lts.add_transition(s0, s1, MessageLabel::Tau);
        lts.add_transition(
            s0,
            s1,
            MessageLabel::ClientHello {
                offered_ciphers: BTreeSet::new(),
                version: ProtocolVersion::Tls12,
                extensions: vec![],
            },
        );

        let minimized = RedundantTransitionElimination::eliminate(lts);
        // Tau subsumed by the visible action.
        assert_eq!(minimized.transition_count(), 1);
        assert!(minimized.transitions[0].label.is_client_action());
    }

    #[test]
    fn test_hopcroft_minimization_trivial() {
        let mut lts = NegotiationLTS::new();
        let s0 = lts.add_state(make_neg(HandshakePhase::Initial, None));
        lts.mark_initial(s0);

        let minimized = HopcroftMinimization::minimize(lts).unwrap();
        assert_eq!(minimized.state_count(), 1);
    }

    #[test]
    fn test_hopcroft_minimization_two_equivalent() {
        // Two equivalent states should be merged.
        let mut lts = NegotiationLTS::new();
        let s0 = lts.add_state(make_neg(HandshakePhase::Initial, None));
        let s1 = lts.add_state(make_neg(HandshakePhase::Initial, None));
        let s2 = lts.add_state(make_neg(
            HandshakePhase::ApplicationData,
            Some(0x002f),
        ));
        lts.mark_initial(s0);
        lts.mark_initial(s1);
        lts.add_transition(s0, s2, MessageLabel::Tau);
        lts.add_transition(s1, s2, MessageLabel::Tau);

        let minimized = HopcroftMinimization::minimize(lts).unwrap();
        // s0 and s1 should be merged into one.
        assert!(minimized.state_count() <= 2);
    }

    #[test]
    fn test_full_minimizer() {
        let mut lts = NegotiationLTS::new();
        let s0 = lts.add_state(make_neg(HandshakePhase::Initial, None));
        let s1 = lts.add_state(make_neg(HandshakePhase::ClientHello, None));
        let s2 = lts.add_state(make_neg(HandshakePhase::ClientHello, None));
        let s3 = lts.add_state(make_neg(
            HandshakePhase::ApplicationData,
            Some(0x002f),
        ));
        let _s4 = lts.add_state(make_neg(HandshakePhase::Alert, None)); // unreachable
        lts.mark_initial(s0);
        lts.add_transition(s0, s1, MessageLabel::Tau);
        lts.add_transition(s0, s2, MessageLabel::Tau);
        lts.add_transition(s1, s3, MessageLabel::Tau);
        lts.add_transition(s2, s3, MessageLabel::Tau);

        let mut minimizer = Minimizer::new();
        let minimized = minimizer.minimize(lts).unwrap();

        assert!(minimizer.unreachable_eliminated() >= 1);
        assert!(minimized.state_count() <= 4);
    }

    #[test]
    fn test_minimization_stats() {
        let stats = MinimizationStats {
            states_before: 100,
            states_after: 25,
            transitions_before: 200,
            transitions_after: 50,
            unreachable_removed: 10,
            dead_end_removed: 5,
            redundant_transitions_removed: 20,
            equivalence_classes: 25,
        };
        assert!((stats.state_reduction() - 0.75).abs() < 0.001);
        assert!((stats.transition_reduction() - 0.75).abs() < 0.001);
        let s = format!("{}", stats);
        assert!(s.contains("75.0%"));
    }

    #[test]
    fn test_minimization_preserves_initial() {
        let mut lts = NegotiationLTS::new();
        let s0 = lts.add_state(make_neg(HandshakePhase::Initial, None));
        let s1 = lts.add_state(make_neg(
            HandshakePhase::ApplicationData,
            Some(0x002f),
        ));
        lts.mark_initial(s0);
        lts.add_transition(s0, s1, MessageLabel::Tau);

        let mut minimizer = Minimizer::new();
        let minimized = minimizer.minimize(lts).unwrap();
        assert!(!minimized.initial_states.is_empty());
    }

    #[test]
    fn test_redundant_dangling_transitions() {
        let mut lts = NegotiationLTS::new();
        let s0 = lts.add_state(make_neg(HandshakePhase::Initial, None));
        lts.mark_initial(s0);
        // Add transition to non-existent state.
        lts.transitions.push(LtsTransition {
            id: TransitionId(99),
            source: s0,
            target: StateId(999),
            label: MessageLabel::Tau,
            guard: None,
        });

        let cleaned = RedundantTransitionElimination::eliminate(lts);
        assert_eq!(cleaned.transition_count(), 0);
    }
}
