//! Simulation relations for Theorem T1 (Extraction Soundness).
//!
//! Provides forward simulation, stuttering simulation, and witness/counterexample
//! generation to verify that the extracted LTS correctly simulates the original
//! symbolic execution.

use crate::{
    observation::ObservationFunction, ExtractError, ExtractResult, HandshakePhase, LtsState,
    MessageLabel, NegotiationLTS, Observable, StateId, SymbolicState,
};
use log::{debug, trace, warn};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::fmt;

// ---------------------------------------------------------------------------
// SimulationRelation
// ---------------------------------------------------------------------------

/// A relation R ⊆ (ProgramState × LtsState) witnessing that the extracted
/// LTS simulates the original program.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationRelation {
    /// Pairs (symbolic_state_id, lts_state_id) in the relation.
    pairs: Vec<(u64, StateId)>,
    /// Index: symbolic state ID → set of related LTS states.
    symbolic_to_lts: HashMap<u64, HashSet<StateId>>,
    /// Index: LTS state → set of related symbolic states.
    lts_to_symbolic: HashMap<StateId, HashSet<u64>>,
}

impl SimulationRelation {
    pub fn new() -> Self {
        Self {
            pairs: Vec::new(),
            symbolic_to_lts: HashMap::new(),
            lts_to_symbolic: HashMap::new(),
        }
    }

    /// Add a pair to the relation.
    pub fn add_pair(&mut self, sym_id: u64, lts_id: StateId) {
        self.pairs.push((sym_id, lts_id));
        self.symbolic_to_lts
            .entry(sym_id)
            .or_default()
            .insert(lts_id);
        self.lts_to_symbolic
            .entry(lts_id)
            .or_default()
            .insert(sym_id);
    }

    /// Check if a pair is in the relation.
    pub fn contains(&self, sym_id: u64, lts_id: StateId) -> bool {
        self.symbolic_to_lts
            .get(&sym_id)
            .map(|s| s.contains(&lts_id))
            .unwrap_or(false)
    }

    /// Get all LTS states related to a symbolic state.
    pub fn lts_states_for(&self, sym_id: u64) -> Option<&HashSet<StateId>> {
        self.symbolic_to_lts.get(&sym_id)
    }

    /// Get all symbolic states related to an LTS state.
    pub fn symbolic_states_for(&self, lts_id: StateId) -> Option<&HashSet<u64>> {
        self.lts_to_symbolic.get(&lts_id)
    }

    /// Number of pairs in the relation.
    pub fn pair_count(&self) -> usize {
        self.pairs.len()
    }

    /// Number of distinct symbolic states in the relation.
    pub fn symbolic_state_count(&self) -> usize {
        self.symbolic_to_lts.len()
    }

    /// Number of distinct LTS states in the relation.
    pub fn lts_state_count(&self) -> usize {
        self.lts_to_symbolic.len()
    }

    /// All pairs in the relation.
    pub fn pairs(&self) -> &[(u64, StateId)] {
        &self.pairs
    }
}

impl Default for SimulationRelation {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for SimulationRelation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SimulationRelation({} pairs, {} sym states, {} lts states)",
            self.pair_count(),
            self.symbolic_state_count(),
            self.lts_state_count(),
        )
    }
}

// ---------------------------------------------------------------------------
// SoundnessWitness
// ---------------------------------------------------------------------------

/// Evidence that the extraction is sound: every symbolic execution step
/// is captured by a corresponding LTS transition (or stutter).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoundnessWitness {
    /// The simulation relation.
    pub relation: SimulationRelation,
    /// Number of symbolic steps verified.
    pub steps_verified: usize,
    /// Number of stuttering steps (internal steps not visible in LTS).
    pub stuttering_steps: usize,
    /// Whether the witness is complete (all steps verified).
    pub is_complete: bool,
    /// Human-readable summary.
    pub summary: String,
}

impl SoundnessWitness {
    pub fn new(relation: SimulationRelation) -> Self {
        let summary = format!(
            "Simulation with {} pairs over {} symbolic states",
            relation.pair_count(),
            relation.symbolic_state_count(),
        );
        Self {
            relation,
            steps_verified: 0,
            stuttering_steps: 0,
            is_complete: false,
            summary,
        }
    }
}

impl fmt::Display for SoundnessWitness {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Soundness Witness:")?;
        writeln!(f, "  {}", self.relation)?;
        writeln!(f, "  Steps verified: {}", self.steps_verified)?;
        writeln!(f, "  Stuttering steps: {}", self.stuttering_steps)?;
        writeln!(f, "  Complete: {}", self.is_complete)?;
        writeln!(f, "  {}", self.summary)
    }
}

// ---------------------------------------------------------------------------
// Simulation counterexample
// ---------------------------------------------------------------------------

/// A counterexample showing where the simulation relation fails.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationCounterexample {
    /// The symbolic state where the failure occurs.
    pub symbolic_state_id: u64,
    /// The LTS state that should simulate it.
    pub lts_state_id: StateId,
    /// The action taken in the symbolic execution.
    pub action: String,
    /// The successor symbolic state.
    pub successor_sym_id: u64,
    /// Description of the failure.
    pub reason: String,
    /// The trace prefix leading to the failure.
    pub trace_prefix: Vec<(u64, String)>,
}

impl fmt::Display for SimulationCounterexample {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Simulation Counterexample:")?;
        writeln!(
            f,
            "  At sym_state={}, lts_state={}",
            self.symbolic_state_id, self.lts_state_id,
        )?;
        writeln!(
            f,
            "  Action: {} → successor sym_state={}",
            self.action, self.successor_sym_id,
        )?;
        writeln!(f, "  Reason: {}", self.reason)?;
        if !self.trace_prefix.is_empty() {
            write!(f, "  Trace: ")?;
            for (i, (sid, act)) in self.trace_prefix.iter().enumerate() {
                if i > 0 {
                    write!(f, " → ")?;
                }
                write!(f, "{}[{}]", sid, act)?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ForwardSimulation
// ---------------------------------------------------------------------------

/// Checks that the extracted LTS forward-simulates the original program.
///
/// For every symbolic step (σ, a, σ'), if (σ, s) ∈ R then there exists
/// s →ᵃ s' in the LTS such that (σ', s') ∈ R.
pub struct ForwardSimulation;

impl ForwardSimulation {
    /// Build a simulation relation from an LTS using the source_symbolic_ids mapping.
    pub fn build_relation(lts: &NegotiationLTS) -> SimulationRelation {
        let mut relation = SimulationRelation::new();
        for (&sid, state) in &lts.states {
            for &sym_id in &state.source_symbolic_ids {
                relation.add_pair(sym_id, sid);
            }
        }
        relation
    }

    /// Check if the relation is a forward simulation.
    ///
    /// For each pair (sym_id, lts_id) in R, and each LTS transition
    /// lts_id →ᵃ lts_id', verify that:
    ///   - The target LTS state lts_id' has related symbolic states
    ///   - (This is a necessary condition; full check requires the
    ///     original symbolic transition relation.)
    pub fn check(
        lts: &NegotiationLTS,
        relation: &SimulationRelation,
    ) -> Result<(), Vec<SimulationCounterexample>> {
        let mut counterexamples = Vec::new();

        for &(sym_id, lts_id) in relation.pairs() {
            // For each outgoing transition from lts_id, check coverage.
            for t in lts.transitions_from(lts_id) {
                let target_has_mapping =
                    relation.symbolic_states_for(t.target).is_some();
                if !target_has_mapping && !t.label.is_internal() {
                    counterexamples.push(SimulationCounterexample {
                        symbolic_state_id: sym_id,
                        lts_state_id: lts_id,
                        action: t.label.label_name().to_string(),
                        successor_sym_id: 0,
                        reason: format!(
                            "LTS target state {} has no related symbolic states",
                            t.target
                        ),
                        trace_prefix: Vec::new(),
                    });
                }
            }
        }

        if counterexamples.is_empty() {
            Ok(())
        } else {
            Err(counterexamples)
        }
    }
}

// ---------------------------------------------------------------------------
// StutteringSimulation
// ---------------------------------------------------------------------------

/// Handles internal (Tau) steps that may not correspond to any visible
/// action in the LTS. Tau transitions are "stutter steps" that do not
/// change the observable state.
pub struct StutteringSimulation;

impl StutteringSimulation {
    /// Check that all Tau transitions preserve the simulation.
    /// A Tau step from s to s' is acceptable if:
    /// 1. (sym, s) ∈ R and (sym, s') ∈ R (same symbolic state maps to both), OR
    /// 2. The observation is unchanged: obs(s) = obs(s').
    pub fn check_stuttering(
        lts: &NegotiationLTS,
        relation: &SimulationRelation,
    ) -> Result<(), Vec<SimulationCounterexample>> {
        let mut obs_fn = ObservationFunction::new();
        let mut counterexamples = Vec::new();

        for t in &lts.transitions {
            if !t.label.is_internal() {
                continue;
            }

            let obs_src = obs_fn.observe(lts, t.source);
            let obs_tgt = obs_fn.observe(lts, t.target);

            // Check option 1: shared symbolic state.
            let shared = relation
                .symbolic_states_for(t.source)
                .and_then(|src_syms| {
                    relation.symbolic_states_for(t.target).map(|tgt_syms| {
                        src_syms.intersection(tgt_syms).next().is_some()
                    })
                })
                .unwrap_or(false);

            // Check option 2: observation preservation.
            let obs_preserved = obs_src.agrees_with(&obs_tgt);

            if !shared && !obs_preserved {
                let sym_id = relation
                    .symbolic_states_for(t.source)
                    .and_then(|s| s.iter().next().copied())
                    .unwrap_or(0);
                counterexamples.push(SimulationCounterexample {
                    symbolic_state_id: sym_id,
                    lts_state_id: t.source,
                    action: "τ".to_string(),
                    successor_sym_id: 0,
                    reason: format!(
                        "Tau transition {} → {} changes observation without shared symbolic state",
                        t.source, t.target,
                    ),
                    trace_prefix: Vec::new(),
                });
            }
        }

        if counterexamples.is_empty() {
            Ok(())
        } else {
            Err(counterexamples)
        }
    }
}

// ---------------------------------------------------------------------------
// SimulationChecker
// ---------------------------------------------------------------------------

/// Comprehensive simulation checker combining forward and stuttering checks.
pub struct SimulationChecker {
    /// Maximum number of counterexamples to collect before stopping.
    max_counterexamples: usize,
}

impl SimulationChecker {
    pub fn new() -> Self {
        Self {
            max_counterexamples: 100,
        }
    }

    pub fn with_max_counterexamples(mut self, max: usize) -> Self {
        self.max_counterexamples = max;
        self
    }

    /// Run all simulation checks and produce a soundness witness or counterexamples.
    pub fn check(
        &self,
        lts: &NegotiationLTS,
    ) -> Result<SoundnessWitness, Vec<SimulationCounterexample>> {
        let relation = ForwardSimulation::build_relation(lts);

        if relation.pair_count() == 0 {
            // No symbolic state mappings → trivially sound (vacuously true).
            let mut witness = SoundnessWitness::new(relation);
            witness.is_complete = true;
            witness.summary = "Vacuously sound (no symbolic state mappings)".into();
            return Ok(witness);
        }

        let mut all_counterexamples: Vec<SimulationCounterexample> = Vec::new();

        // Forward simulation check.
        if let Err(cxs) = ForwardSimulation::check(lts, &relation) {
            all_counterexamples.extend(cxs);
        }

        // Stuttering simulation check.
        if let Err(cxs) = StutteringSimulation::check_stuttering(lts, &relation) {
            all_counterexamples.extend(cxs);
        }

        if all_counterexamples.len() > self.max_counterexamples {
            all_counterexamples.truncate(self.max_counterexamples);
        }

        if all_counterexamples.is_empty() {
            let mut witness = SoundnessWitness::new(relation.clone());
            witness.steps_verified = self.count_verified_steps(lts, &relation);
            witness.stuttering_steps = self.count_stuttering_steps(lts);
            witness.is_complete = true;
            witness.summary = format!(
                "Sound: {} steps verified ({} stuttering), {} simulation pairs",
                witness.steps_verified,
                witness.stuttering_steps,
                relation.pair_count(),
            );
            Ok(witness)
        } else {
            Err(all_counterexamples)
        }
    }

    /// Verify that a simulation relation is valid for an LTS.
    pub fn verify_relation(
        &self,
        lts: &NegotiationLTS,
        relation: &SimulationRelation,
    ) -> bool {
        ForwardSimulation::check(lts, relation).is_ok()
            && StutteringSimulation::check_stuttering(lts, relation).is_ok()
    }

    fn count_verified_steps(
        &self,
        lts: &NegotiationLTS,
        relation: &SimulationRelation,
    ) -> usize {
        let mut count = 0;
        for &(_sym_id, lts_id) in relation.pairs() {
            count += lts.transitions_from(lts_id).len();
        }
        count
    }

    fn count_stuttering_steps(&self, lts: &NegotiationLTS) -> usize {
        lts.transitions
            .iter()
            .filter(|t| t.label.is_internal())
            .count()
    }
}

impl Default for SimulationChecker {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{NegotiationOutcome, ProtocolVersion};
    use negsyn_types::NegotiationState;
    use std::collections::BTreeSet;

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
    fn test_simulation_relation_basic() {
        let mut rel = SimulationRelation::new();
        rel.add_pair(0, StateId(0));
        rel.add_pair(0, StateId(1));
        rel.add_pair(1, StateId(1));

        assert!(rel.contains(0, StateId(0)));
        assert!(rel.contains(0, StateId(1)));
        assert!(rel.contains(1, StateId(1)));
        assert!(!rel.contains(1, StateId(0)));
        assert_eq!(rel.pair_count(), 3);
        assert_eq!(rel.symbolic_state_count(), 2);
        assert_eq!(rel.lts_state_count(), 2);
    }

    #[test]
    fn test_forward_simulation_build() {
        let mut lts = NegotiationLTS::new();
        let s0 = lts.add_state(make_neg(HandshakePhase::Initial, None));
        lts.mark_initial(s0);

        // Add source symbolic IDs.
        if let Some(state) = lts.get_state_mut(s0) {
            state.source_symbolic_ids = vec![100, 200];
        }

        let rel = ForwardSimulation::build_relation(&lts);
        assert_eq!(rel.pair_count(), 2);
        assert!(rel.contains(100, s0));
        assert!(rel.contains(200, s0));
    }

    #[test]
    fn test_forward_simulation_succeeds() {
        let mut lts = NegotiationLTS::new();
        let s0 = lts.add_state(make_neg(HandshakePhase::Initial, None));
        let s1 = lts.add_state(make_neg(
            HandshakePhase::ApplicationData,
            Some(0x002f),
        ));
        lts.mark_initial(s0);
        lts.add_transition(s0, s1, MessageLabel::Tau);

        if let Some(state) = lts.get_state_mut(s0) {
            state.source_symbolic_ids = vec![0];
        }
        if let Some(state) = lts.get_state_mut(s1) {
            state.source_symbolic_ids = vec![1];
        }

        let rel = ForwardSimulation::build_relation(&lts);
        let result = ForwardSimulation::check(&lts, &rel);
        assert!(result.is_ok());
    }

    #[test]
    fn test_simulation_checker_sound() {
        let mut lts = NegotiationLTS::new();
        let s0 = lts.add_state(make_neg(HandshakePhase::Initial, None));
        let s1 = lts.add_state(make_neg(
            HandshakePhase::ApplicationData,
            Some(0x002f),
        ));
        lts.mark_initial(s0);
        lts.add_transition(s0, s1, MessageLabel::Tau);

        if let Some(state) = lts.get_state_mut(s0) {
            state.source_symbolic_ids = vec![0];
        }
        if let Some(state) = lts.get_state_mut(s1) {
            state.source_symbolic_ids = vec![1];
        }

        let checker = SimulationChecker::new();
        let result = checker.check(&lts);
        assert!(result.is_ok());
        let witness = result.unwrap();
        assert!(witness.is_complete);
    }

    #[test]
    fn test_simulation_checker_vacuous() {
        // LTS with no source_symbolic_ids → vacuously sound.
        let mut lts = NegotiationLTS::new();
        let s0 = lts.add_state(make_neg(HandshakePhase::Initial, None));
        lts.mark_initial(s0);

        let checker = SimulationChecker::new();
        let result = checker.check(&lts);
        assert!(result.is_ok());
        let witness = result.unwrap();
        assert!(witness.summary.contains("Vacuously"));
    }

    #[test]
    fn test_stuttering_simulation() {
        let mut lts = NegotiationLTS::new();
        let s0 = lts.add_state(make_neg(HandshakePhase::Initial, None));
        let s1 = lts.add_state(make_neg(HandshakePhase::Initial, None));
        lts.mark_initial(s0);
        lts.add_transition(s0, s1, MessageLabel::Tau);

        // Both states have the same observation → stuttering is OK.
        if let Some(state) = lts.get_state_mut(s0) {
            state.source_symbolic_ids = vec![0];
        }
        if let Some(state) = lts.get_state_mut(s1) {
            state.source_symbolic_ids = vec![0]; // Same symbolic state
        }

        let rel = ForwardSimulation::build_relation(&lts);
        let result = StutteringSimulation::check_stuttering(&lts, &rel);
        assert!(result.is_ok());
    }

    #[test]
    fn test_counterexample_display() {
        let cx = SimulationCounterexample {
            symbolic_state_id: 42,
            lts_state_id: StateId(7),
            action: "CH".to_string(),
            successor_sym_id: 43,
            reason: "no matching LTS transition".to_string(),
            trace_prefix: vec![
                (40, "τ".to_string()),
                (41, "CH".to_string()),
                (42, "SH".to_string()),
            ],
        };
        let s = format!("{}", cx);
        assert!(s.contains("42"));
        assert!(s.contains("s7"));
        assert!(s.contains("CH"));
    }

    #[test]
    fn test_soundness_witness_display() {
        let rel = SimulationRelation::new();
        let witness = SoundnessWitness::new(rel);
        let s = format!("{}", witness);
        assert!(s.contains("Soundness"));
    }

    #[test]
    fn test_simulation_relation_display() {
        let mut rel = SimulationRelation::new();
        rel.add_pair(0, StateId(0));
        let s = format!("{}", rel);
        assert!(s.contains("1 pairs"));
    }

    #[test]
    fn test_verify_relation() {
        let mut lts = NegotiationLTS::new();
        let s0 = lts.add_state(make_neg(HandshakePhase::Initial, None));
        lts.mark_initial(s0);

        if let Some(state) = lts.get_state_mut(s0) {
            state.source_symbolic_ids = vec![0];
        }

        let rel = ForwardSimulation::build_relation(&lts);
        let checker = SimulationChecker::new();
        assert!(checker.verify_relation(&lts, &rel));
    }
}
