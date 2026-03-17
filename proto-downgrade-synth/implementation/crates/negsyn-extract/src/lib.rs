//! negsyn-extract: State machine extractor with bisimulation quotient (ALG3: SMEXTRACT).
//!
//! Extracts a negotiation labeled transition system from symbolic execution traces,
//! computes the protocol bisimulation quotient to minimize the state space, and
//! provides simulation checking for extraction soundness (Theorem T1).
//!
//! This crate uses the `negsyn-types` API for all shared types:
//! - `StateGraph` / `StateId` / `TransitionId` from `negsyn_types::graph`
//! - `TransitionLabel` / `NegotiationState` / `HandshakePhase` from `negsyn_types::protocol`
//! - `SymbolicState` / `SymbolicValue` / `PathConstraint` from `negsyn_types::symbolic`
//!
//! Extraction-specific types (`Observable`, local `ExtractionConfig`, etc.) are
//! defined here since they are part of the extraction domain.

pub mod bisimulation;
pub mod extractor;
pub mod minimize;
pub mod observation;
pub mod quotient;
pub mod serialize;
pub mod simulation;
pub mod trace;

// ---------------------------------------------------------------------------
// Re-export core negsyn-types used throughout this crate
// ---------------------------------------------------------------------------
pub use negsyn_types::{
    // symbolic
    BinOp, ConcreteValue, MergeableState, PathConstraint, SymSort, SymbolicMemory,
    SymbolicState, SymbolicValue, UnOp,
    // protocol
    CipherSuite, Extension, HandshakePhase, NegotiationState, ProtocolFamily, ProtocolVersion,
    SecurityLevel, TransitionLabel,
    // graph
    StateData, StateGraph, StateId, TransitionId,
    // error
    NegSynthError, NegSynthResult,
};

/// Type alias: when sub-modules say `NegotiationOutcome`, they mean our
/// extraction-specific `OutcomeKind` (not `negsyn_types::NegotiationOutcome`).
pub type NegotiationOutcome = OutcomeKind;

/// Type alias: when sub-modules say `ExtractionConfig`, they mean our
/// extraction-pipeline `LocalExtractionConfig` (not `negsyn_types::ExtractionConfig`).
pub type ExtractionConfig = LocalExtractionConfig;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fmt;

// ---------------------------------------------------------------------------
// MessageLabel — extraction-level action labels
// ---------------------------------------------------------------------------

/// Action labels for the negotiation LTS.
///
/// These represent observable protocol events and adversary actions used as
/// transition labels in the extraction-level LTS.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MessageLabel {
    /// Internal (silent) step.
    Tau,
    /// Client sends ClientHello with offered cipher suites, version, and extensions.
    ClientHello {
        offered_ciphers: BTreeSet<u16>,
        version: ProtocolVersion,
        extensions: Vec<Extension>,
    },
    /// Server sends ServerHello with selected cipher, version, and extensions.
    ServerHello {
        selected_cipher: u16,
        version: ProtocolVersion,
        extensions: Vec<Extension>,
    },
    /// TLS Alert message.
    Alert { level: u8, description: u8 },
    /// Server Certificate message.
    ServerCertificate,
    /// Server Key Exchange message.
    ServerKeyExchange,
    /// Server Hello Done message.
    ServerHelloDone,
    /// Client Key Exchange message.
    ClientKeyExchange,
    /// Client ChangeCipherSpec message.
    ClientChangeCipherSpec,
    /// Server ChangeCipherSpec message.
    ServerChangeCipherSpec,
    /// Client Finished message.
    ClientFinished { verify_data_hash: u64 },
    /// Server Finished message.
    ServerFinished { verify_data_hash: u64 },
    /// Adversary drops a message.
    AdversaryDrop { message_index: u32 },
    /// Adversary modifies a message.
    AdversaryModify { message_index: u32 },
    /// Adversary injects a message.
    AdversaryInject { message_index: u32 },
    /// Adversary intercepts a message.
    AdversaryIntercept { message_index: u32 },
}

impl MessageLabel {
    /// Short name for the label (used as transition descriptions).
    pub fn label_name(&self) -> &str {
        match self {
            Self::Tau => "τ",
            Self::ClientHello { .. } => "CH",
            Self::ServerHello { .. } => "SH",
            Self::Alert { .. } => "Alert",
            Self::ServerCertificate => "SC",
            Self::ServerKeyExchange => "SKE",
            Self::ServerHelloDone => "SHD",
            Self::ClientKeyExchange => "CKE",
            Self::ClientChangeCipherSpec => "CCCS",
            Self::ServerChangeCipherSpec => "SCCS",
            Self::ClientFinished { .. } => "CF",
            Self::ServerFinished { .. } => "SF",
            Self::AdversaryDrop { .. } => "ADrop",
            Self::AdversaryModify { .. } => "AMod",
            Self::AdversaryInject { .. } => "AInj",
            Self::AdversaryIntercept { .. } => "AInt",
        }
    }

    /// Whether this label represents an internal (silent) action.
    pub fn is_internal(&self) -> bool {
        matches!(self, Self::Tau)
    }

    /// Whether this label represents an adversary action.
    pub fn is_adversary_action(&self) -> bool {
        matches!(
            self,
            Self::AdversaryDrop { .. }
                | Self::AdversaryModify { .. }
                | Self::AdversaryInject { .. }
                | Self::AdversaryIntercept { .. }
        )
    }

    /// Whether this label represents a client-initiated action.
    pub fn is_client_action(&self) -> bool {
        matches!(
            self,
            Self::ClientHello { .. }
                | Self::ClientKeyExchange
                | Self::ClientChangeCipherSpec
                | Self::ClientFinished { .. }
        )
    }
}

impl fmt::Display for MessageLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tau => write!(f, "τ"),
            Self::ClientHello {
                offered_ciphers,
                version,
                ..
            } => write!(f, "CH(ciphers={}, ver={})", offered_ciphers.len(), version),
            Self::ServerHello {
                selected_cipher,
                version,
                ..
            } => write!(f, "SH(cipher=0x{:04x}, ver={})", selected_cipher, version),
            Self::Alert { level, description } => write!(f, "Alert({},{})", level, description),
            Self::ServerCertificate => write!(f, "SC"),
            Self::ServerKeyExchange => write!(f, "SKE"),
            Self::ServerHelloDone => write!(f, "SHD"),
            Self::ClientKeyExchange => write!(f, "CKE"),
            Self::ClientChangeCipherSpec => write!(f, "CCCS"),
            Self::ServerChangeCipherSpec => write!(f, "SCCS"),
            Self::ClientFinished { verify_data_hash } => {
                write!(f, "CF(hash=0x{:x})", verify_data_hash)
            }
            Self::ServerFinished { verify_data_hash } => {
                write!(f, "SF(hash=0x{:x})", verify_data_hash)
            }
            Self::AdversaryDrop { message_index } => write!(f, "ADrop({})", message_index),
            Self::AdversaryModify { message_index } => write!(f, "AMod({})", message_index),
            Self::AdversaryInject { message_index } => write!(f, "AInj({})", message_index),
            Self::AdversaryIntercept { message_index } => write!(f, "AInt({})", message_index),
        }
    }
}

// ---------------------------------------------------------------------------
// LtsTransition — extraction-level transition
// ---------------------------------------------------------------------------

/// A transition in the extraction-level LTS.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LtsTransition {
    pub id: TransitionId,
    pub source: StateId,
    pub target: StateId,
    pub label: MessageLabel,
    pub guard: Option<SymbolicValue>,
}

// ---------------------------------------------------------------------------
// NegotiationLTS — extraction-level LTS (distinct from negsyn_types::NegotiationLTS)
// ---------------------------------------------------------------------------

/// The extraction-level Negotiation Labeled Transition System.
///
/// This is the LTS used internally by the extraction pipeline. It wraps
/// `LtsState` values indexed by `StateId`, with transitions carrying
/// `MessageLabel` labels and optional symbolic guards.
///
/// *Not* the same as `negsyn_types::NegotiationLTS`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NegotiationLTS {
    pub states: IndexMap<StateId, LtsState>,
    pub transitions: Vec<LtsTransition>,
    pub initial_states: Vec<StateId>,
    next_state_id: u32,
    next_transition_id: u32,
}

impl NegotiationLTS {
    pub fn new() -> Self {
        Self {
            states: IndexMap::new(),
            transitions: Vec::new(),
            initial_states: Vec::new(),
            next_state_id: 0,
            next_transition_id: 0,
        }
    }

    /// Add a state from a `NegotiationState`, returning its `StateId`.
    pub fn add_state(&mut self, negotiation: NegotiationState) -> StateId {
        let sid = StateId(self.next_state_id);
        self.next_state_id += 1;
        let lts_state = LtsState::new(sid, negotiation);
        self.states.insert(sid, lts_state);
        sid
    }

    /// Add a state with a specific `StateId`.
    pub fn add_state_with_id(&mut self, sid: StateId, negotiation: NegotiationState) {
        let lts_state = LtsState::new(sid, negotiation);
        self.states.insert(sid, lts_state);
        if sid.0 >= self.next_state_id {
            self.next_state_id = sid.0 + 1;
        }
    }

    /// Mark a state as initial.
    pub fn mark_initial(&mut self, id: StateId) {
        if let Some(state) = self.states.get_mut(&id) {
            state.is_initial = true;
        }
        if !self.initial_states.contains(&id) {
            self.initial_states.push(id);
        }
    }

    /// Add a transition.
    pub fn add_transition(
        &mut self,
        from: StateId,
        to: StateId,
        label: MessageLabel,
    ) {
        let tid = TransitionId(self.next_transition_id);
        self.next_transition_id += 1;
        self.transitions.push(LtsTransition {
            id: tid,
            source: from,
            target: to,
            label,
            guard: None,
        });
    }

    /// Add a transition with a symbolic guard.
    pub fn add_transition_with_guard(
        &mut self,
        from: StateId,
        to: StateId,
        label: MessageLabel,
        guard: SymbolicValue,
    ) {
        let tid = TransitionId(self.next_transition_id);
        self.next_transition_id += 1;
        self.transitions.push(LtsTransition {
            id: tid,
            source: from,
            target: to,
            label,
            guard: Some(guard),
        });
    }

    /// Get observation at a state.
    pub fn obs(&self, id: StateId) -> Option<&Observable> {
        self.states.get(&id).map(|s| &s.observation)
    }

    /// Get state by ID.
    pub fn get_state(&self, id: StateId) -> Option<&LtsState> {
        self.states.get(&id)
    }

    /// Get mutable state by ID.
    pub fn get_state_mut(&mut self, id: StateId) -> Option<&mut LtsState> {
        self.states.get_mut(&id)
    }

    /// Number of states.
    pub fn state_count(&self) -> usize {
        self.states.len()
    }

    /// Number of transitions.
    pub fn transition_count(&self) -> usize {
        self.transitions.len()
    }

    /// All state IDs.
    pub fn state_ids(&self) -> Vec<StateId> {
        self.states.keys().copied().collect()
    }

    /// Terminal states.
    pub fn terminal_states(&self) -> Vec<StateId> {
        self.states
            .iter()
            .filter(|(_, s)| s.is_terminal)
            .map(|(&id, _)| id)
            .collect()
    }

    /// Transitions originating from a given state.
    pub fn transitions_from(&self, id: StateId) -> Vec<&LtsTransition> {
        self.transitions.iter().filter(|t| t.source == id).collect()
    }

    /// Reachable states from initial states via BFS.
    pub fn reachable_states(&self) -> BTreeSet<StateId> {
        let mut visited = BTreeSet::new();
        let mut queue = std::collections::VecDeque::new();
        for &init in &self.initial_states {
            queue.push_back(init);
            visited.insert(init);
        }
        while let Some(s) = queue.pop_front() {
            for t in &self.transitions {
                if t.source == s && visited.insert(t.target) {
                    queue.push_back(t.target);
                }
            }
        }
        visited
    }

    /// Remove a state and all incident transitions.
    pub fn remove_state(&mut self, id: StateId) {
        self.states.swap_remove(&id);
        self.transitions
            .retain(|t| t.source != id && t.target != id);
        self.initial_states.retain(|s| *s != id);
    }

    /// Set of distinct transition label descriptions.
    pub fn alphabet(&self) -> BTreeSet<String> {
        self.transitions
            .iter()
            .map(|t| t.label.label_name().to_string())
            .collect()
    }

    /// Adjacency list: state → list of (label_name, target).
    pub fn adjacency_list(&self) -> HashMap<StateId, Vec<(String, StateId)>> {
        let mut adj: HashMap<StateId, Vec<(String, StateId)>> = HashMap::new();
        for t in &self.transitions {
            adj.entry(t.source)
                .or_default()
                .push((t.label.label_name().to_string(), t.target));
        }
        adj
    }

    /// Reverse adjacency list: state → list of (predecessor, label_name).
    pub fn reverse_adjacency_list(&self) -> HashMap<StateId, Vec<(StateId, String)>> {
        let mut radj: HashMap<StateId, Vec<(StateId, String)>> = HashMap::new();
        for t in &self.transitions {
            radj.entry(t.target)
                .or_default()
                .push((t.source, t.label.label_name().to_string()));
        }
        radj
    }
}

impl Default for NegotiationLTS {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Observable outcome (Definition D7) — extraction-specific
// ---------------------------------------------------------------------------

/// High-level negotiation outcome category.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OutcomeKind {
    /// Negotiation completed successfully.
    Completed,
    /// Negotiation was aborted (alert or error).
    Aborted,
    /// Negotiation is still in progress (non-terminal state).
    InProgress,
}

/// Observable negotiation outcome at LTS states (Definition D7: O, obs).
///
/// This is extraction-specific: it captures the observable aspects of a
/// negotiation state that matter for bisimulation equivalence.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Observable {
    /// Selected cipher suite IANA ID (if negotiation completed).
    pub selected_cipher: Option<u16>,
    /// Selected protocol version (if known).
    pub selected_version: Option<ProtocolVersion>,
    /// Active extension IDs at this state.
    pub active_extensions: BTreeSet<u16>,
    /// Outcome category.
    pub outcome: OutcomeKind,
    /// The handshake phase at observation time.
    pub phase: HandshakePhase,
}

impl Observable {
    /// Create an observation for a successfully completed negotiation.
    pub fn successful(
        cipher: u16,
        version: ProtocolVersion,
        extensions: BTreeSet<u16>,
    ) -> Self {
        Self {
            selected_cipher: Some(cipher),
            selected_version: Some(version),
            active_extensions: extensions,
            outcome: OutcomeKind::Completed,
            phase: HandshakePhase::Done,
        }
    }

    /// Create an observation for an aborted negotiation.
    pub fn aborted() -> Self {
        Self {
            selected_cipher: None,
            selected_version: None,
            active_extensions: BTreeSet::new(),
            outcome: OutcomeKind::Aborted,
            phase: HandshakePhase::Abort,
        }
    }

    /// Create an observation for an in-progress negotiation.
    pub fn in_progress(phase: HandshakePhase) -> Self {
        Self {
            selected_cipher: None,
            selected_version: None,
            active_extensions: BTreeSet::new(),
            outcome: OutcomeKind::InProgress,
            phase,
        }
    }

    /// Create from a `NegotiationState` from negsyn-types.
    pub fn from_negotiation_state(ns: &NegotiationState) -> Self {
        let phase = ns.phase.clone();
        if phase.is_terminal() {
            if phase.is_success() {
                let cipher_id = ns
                    .selected_cipher
                    .as_ref()
                    .map(|c| c.iana_id)
                    .unwrap_or(0);
                let version = ns.version.clone().unwrap_or_else(ProtocolVersion::tls12);
                let ext_ids: BTreeSet<u16> =
                    ns.extensions.iter().map(|e| e.id).collect();
                Self::successful(cipher_id, version, ext_ids)
            } else {
                Self::aborted()
            }
        } else {
            Self::in_progress(phase)
        }
    }

    /// Two observables agree if they are structurally equal (bisimulation check).
    pub fn agrees_with(&self, other: &Observable) -> bool {
        self == other
    }

    /// Coarser agreement: only check outcome and cipher/version, ignore extensions.
    pub fn coarse_agrees_with(&self, other: &Observable) -> bool {
        self.outcome == other.outcome
            && self.selected_cipher == other.selected_cipher
            && self.selected_version == other.selected_version
            && self.phase == other.phase
    }

    /// Whether this observation represents a downgrade from `baseline`.
    pub fn is_downgrade_from(&self, baseline: &Observable) -> bool {
        match (&self.outcome, &baseline.outcome) {
            (OutcomeKind::Completed, OutcomeKind::Completed) => {
                // Downgrade if cipher or version is weaker
                let cipher_downgrade = match (self.selected_cipher, baseline.selected_cipher) {
                    (Some(a), Some(b)) => a != b,
                    _ => false,
                };
                let version_downgrade = self.selected_version != baseline.selected_version;
                let ext_stripped = !baseline
                    .active_extensions
                    .is_subset(&self.active_extensions);
                cipher_downgrade || version_downgrade || ext_stripped
            }
            _ => false,
        }
    }
}

impl fmt::Display for Observable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.outcome {
            OutcomeKind::Completed => {
                write!(
                    f,
                    "OK(cipher=0x{:04x}, ver={}, ext={})",
                    self.selected_cipher.unwrap_or(0),
                    self.selected_version
                        .as_ref()
                        .map(|v| format!("{}.{}", v.major, v.minor))
                        .unwrap_or_else(|| "?".into()),
                    self.active_extensions.len()
                )
            }
            OutcomeKind::Aborted => write!(f, "ABORT"),
            OutcomeKind::InProgress => write!(f, "IN_PROGRESS({:?})", self.phase),
        }
    }
}

// ---------------------------------------------------------------------------
// Local LTS wrapper that extends StateGraph with observations
// ---------------------------------------------------------------------------

/// A state in the extraction LTS, combining a graph state with observation data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LtsState {
    /// The graph state ID from `StateGraph`.
    pub graph_id: StateId,
    /// The negotiation state that produced this LTS state.
    pub negotiation: NegotiationState,
    /// Computed observation at this state.
    pub observation: Observable,
    /// Whether this is an initial state.
    pub is_initial: bool,
    /// Whether this is a terminal state.
    pub is_terminal: bool,
    /// Arbitrary metadata (e.g., for serialization/debugging).
    pub metadata: HashMap<String, String>,
    /// Source symbolic state IDs that were mapped to this LTS state.
    pub source_symbolic_ids: Vec<u64>,
}

impl LtsState {
    /// Construct from a graph-level `StateId` and a `NegotiationState`.
    pub fn new(graph_id: StateId, negotiation: NegotiationState) -> Self {
        let is_terminal = negotiation.phase.is_terminal();
        let observation = Observable::from_negotiation_state(&negotiation);
        Self {
            graph_id,
            negotiation,
            observation,
            is_initial: false,
            is_terminal,
            metadata: HashMap::new(),
            source_symbolic_ids: Vec::new(),
        }
    }

    /// Set the observation explicitly (e.g., after refinement).
    pub fn set_observation(&mut self, obs: Observable) {
        self.observation = obs;
    }
}

/// The extraction-level Labeled Transition System.
///
/// Wraps a `StateGraph` (from negsyn-types) with per-state observation data,
/// negotiation metadata, and symbolic provenance tracking.
///
/// Definition D1: N = (S, S₀, Λ, δ, O, obs).
#[derive(Debug, Clone)]
pub struct ExtractionLTS {
    /// The underlying graph from negsyn-types.
    pub graph: StateGraph,
    /// Extended state data indexed by `StateId`.
    pub state_data: IndexMap<StateId, LtsState>,
    /// Initial state IDs.
    pub initial_states: Vec<StateId>,
    /// Mapping from symbolic state ID → graph StateId.
    pub symbolic_to_graph: HashMap<u64, StateId>,
}

impl ExtractionLTS {
    /// Create a new, empty extraction LTS.
    pub fn new() -> Self {
        Self {
            graph: StateGraph::new(),
            state_data: IndexMap::new(),
            initial_states: Vec::new(),
            symbolic_to_graph: HashMap::new(),
        }
    }

    /// Add a state from a `NegotiationState`, returning its `StateId`.
    pub fn add_state(&mut self, negotiation: NegotiationState) -> StateId {
        let phase = negotiation.phase.clone();
        let label = format!("{:?}", phase);
        let graph_id = self.graph.add_state(phase, label);
        let lts_state = LtsState::new(graph_id, negotiation);
        self.state_data.insert(graph_id, lts_state);
        graph_id
    }

    /// Add a state and record the originating symbolic state ID.
    pub fn add_state_from_symbolic(
        &mut self,
        negotiation: NegotiationState,
        symbolic_id: u64,
    ) -> StateId {
        let sid = self.add_state(negotiation);
        if let Some(data) = self.state_data.get_mut(&sid) {
            data.source_symbolic_ids.push(symbolic_id);
        }
        self.symbolic_to_graph.insert(symbolic_id, sid);
        sid
    }

    /// Mark a state as initial.
    pub fn mark_initial(&mut self, id: StateId) {
        self.graph.set_initial(id);
        if let Some(data) = self.state_data.get_mut(&id) {
            data.is_initial = true;
        }
        if !self.initial_states.contains(&id) {
            self.initial_states.push(id);
        }
    }

    /// Add a transition using a `TransitionLabel` from negsyn-types.
    pub fn add_transition(
        &mut self,
        from: StateId,
        to: StateId,
        label: TransitionLabel,
    ) -> Option<TransitionId> {
        self.graph.add_transition(from, to, label)
    }

    /// Get observation at a state.
    pub fn obs(&self, id: StateId) -> Option<&Observable> {
        self.state_data.get(&id).map(|s| &s.observation)
    }

    /// Get the extended state data.
    pub fn get_state(&self, id: StateId) -> Option<&LtsState> {
        self.state_data.get(&id)
    }

    /// Get mutable state data.
    pub fn get_state_mut(&mut self, id: StateId) -> Option<&mut LtsState> {
        self.state_data.get_mut(&id)
    }

    /// Number of states.
    pub fn state_count(&self) -> usize {
        self.graph.state_count()
    }

    /// Number of transitions.
    pub fn transition_count(&self) -> usize {
        self.graph.transition_count()
    }

    /// All state IDs.
    pub fn state_ids(&self) -> Vec<StateId> {
        self.graph.state_ids()
    }

    /// Terminal states.
    pub fn terminal_states(&self) -> Vec<StateId> {
        self.graph.terminal_states()
    }

    /// Successors from a given state.
    pub fn successors(&self, id: StateId) -> Vec<(TransitionLabel, StateId)> {
        self.graph.successors(id)
    }

    /// Predecessors of a given state.
    pub fn predecessors(&self, id: StateId) -> Vec<(TransitionLabel, StateId)> {
        self.graph.predecessors(id)
    }

    /// Reachable states from the initial state.
    pub fn reachable_states(&self) -> BTreeSet<StateId> {
        self.graph.reachable_from_initial()
    }

    /// The set of distinct transition label descriptions used.
    pub fn alphabet(&self) -> BTreeSet<String> {
        let mut labels = BTreeSet::new();
        for sid in self.graph.state_ids() {
            for (lbl, _) in self.graph.successors(sid) {
                labels.insert(format!("{:?}", lbl));
            }
        }
        labels
    }

    /// Remove a state and all incident transitions.
    pub fn remove_state(&mut self, id: StateId) {
        self.state_data.swap_remove(&id);
        self.initial_states.retain(|s| *s != id);
        // Note: StateGraph doesn't expose remove_state, so we track removal
        // in state_data and treat missing state_data entries as removed.
    }

    /// Check if a state ID is live (not removed).
    pub fn is_live(&self, id: StateId) -> bool {
        self.state_data.contains_key(&id)
    }

    /// Build a `NegotiationLTS` (from negsyn_types::protocol) from this extraction LTS.
    pub fn to_negotiation_lts(&self) -> negsyn_types::NegotiationLTS {
        let init_neg = self
            .initial_states
            .first()
            .and_then(|id| self.state_data.get(id))
            .map(|s| s.negotiation.clone())
            .unwrap_or_else(NegotiationState::initial);

        let mut nlts = negsyn_types::NegotiationLTS::new(init_neg);

        // Map our StateId → index in NegotiationLTS
        let mut id_to_idx: HashMap<StateId, usize> = HashMap::new();
        // The initial state is index 0 (added by ::new)
        if let Some(init_id) = self.initial_states.first() {
            id_to_idx.insert(*init_id, 0);
        }

        // Add remaining states
        for (sid, lts_state) in &self.state_data {
            if id_to_idx.contains_key(sid) {
                continue;
            }
            let idx = nlts.add_state(lts_state.negotiation.clone());
            id_to_idx.insert(*sid, idx);
        }

        // Add transitions
        for sid in self.graph.state_ids() {
            if let Some(&from_idx) = id_to_idx.get(&sid) {
                for (label, target) in self.graph.successors(sid) {
                    if let Some(&to_idx) = id_to_idx.get(&target) {
                        nlts.add_transition(from_idx, label, to_idx);
                    }
                }
            }
        }

        // Set observations
        for (sid, lts_state) in &self.state_data {
            if let (Some(&idx), OutcomeKind::Completed) =
                (id_to_idx.get(sid), &lts_state.observation.outcome)
            {
                let cipher = lts_state
                    .negotiation
                    .selected_cipher
                    .clone()
                    .unwrap_or_else(|| CipherSuite::new(
                        0,
                        "UNKNOWN",
                        negsyn_types::protocol::KeyExchange::NULL,
                        negsyn_types::protocol::AuthAlgorithm::NULL,
                        negsyn_types::protocol::EncryptionAlgorithm::NULL,
                        negsyn_types::protocol::MacAlgorithm::NULL,
                        SecurityLevel::Broken,
                    ));
                let version = lts_state
                    .negotiation
                    .version
                    .clone()
                    .unwrap_or_else(ProtocolVersion::tls12);
                let outcome = negsyn_types::NegotiationOutcome {
                    selected_cipher: cipher,
                    version,
                    extensions: lts_state.negotiation.extensions.clone(),
                    session_resumed: false,
                };
                nlts.set_observation(idx, outcome);
            }
        }

        nlts
    }
}

impl Default for ExtractionLTS {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Merge operator trait (compatible with negsyn-merge when available)
// ---------------------------------------------------------------------------

/// Protocol-aware merge operator interface.
///
/// Defined locally so negsyn-extract compiles independently of negsyn-merge.
pub trait MergeOperator: Send + Sync {
    /// Attempt to merge two symbolic states.
    fn merge_states(
        &self,
        s1: &SymbolicState,
        s2: &SymbolicState,
    ) -> Result<SymbolicState, ExtractError>;

    /// Quick check whether two states are eligible for merging.
    fn is_mergeable(&self, s1: &SymbolicState, s2: &SymbolicState) -> bool;
}

/// Default no-op merge operator that never merges.
#[derive(Debug, Clone)]
pub struct NoOpMergeOperator;

impl MergeOperator for NoOpMergeOperator {
    fn merge_states(
        &self,
        _s1: &SymbolicState,
        _s2: &SymbolicState,
    ) -> Result<SymbolicState, ExtractError> {
        Err(ExtractError::Internal(
            "NoOp merge operator: merging disabled".into(),
        ))
    }

    fn is_mergeable(&self, _s1: &SymbolicState, _s2: &SymbolicState) -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
// Extraction configuration (local, extraction-specific)
// ---------------------------------------------------------------------------

/// Configuration for the state machine extraction pipeline.
///
/// This is distinct from `negsyn_types::ExtractionConfig` which holds
/// pattern-matching config. Our config controls the extraction pipeline itself.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalExtractionConfig {
    /// Maximum number of states before aborting extraction.
    pub max_states: usize,
    /// Maximum number of transitions before aborting.
    pub max_transitions: usize,
    /// Maximum trace depth to consider.
    pub max_trace_depth: u32,
    /// Whether to run bisimulation quotient after extraction.
    pub enable_bisimulation: bool,
    /// Whether to run Hopcroft minimization after quotient.
    pub enable_minimization: bool,
    /// Whether to eliminate unreachable states.
    pub enable_unreachable_elimination: bool,
    /// Maximum partition refinement iterations.
    pub max_refinement_iterations: u32,
    /// Whether to compute simulation witness for soundness.
    pub compute_simulation_witness: bool,
    /// Timeout in milliseconds for the entire extraction.
    pub timeout_ms: u64,
    /// Enable trace normalization before extraction.
    pub normalize_traces: bool,
    /// Filter out non-negotiation trace steps.
    pub filter_non_negotiation: bool,
    /// The upstream extraction config for pattern matching.
    pub type_config: Option<negsyn_types::ExtractionConfig>,
    /// Upstream merge config for state merging parameters.
    pub merge_config: Option<negsyn_types::MergeConfig>,
    /// Confidence threshold for cipher/version extraction.
    pub confidence_threshold: f64,
}

impl Default for LocalExtractionConfig {
    fn default() -> Self {
        Self {
            max_states: 100_000,
            max_transitions: 500_000,
            max_trace_depth: 256,
            enable_bisimulation: true,
            enable_minimization: true,
            enable_unreachable_elimination: true,
            max_refinement_iterations: 10_000,
            compute_simulation_witness: false,
            timeout_ms: 120_000,
            normalize_traces: true,
            filter_non_negotiation: true,
            type_config: None,
            merge_config: None,
            confidence_threshold: 0.8,
        }
    }
}

impl LocalExtractionConfig {
    /// Create from an upstream `negsyn_types::ExtractionConfig`.
    pub fn from_upstream(upstream: &negsyn_types::ExtractionConfig) -> Self {
        let mut cfg = Self::default();
        cfg.confidence_threshold = upstream.confidence_threshold;
        cfg.type_config = Some(upstream.clone());
        cfg
    }

    /// Create from both upstream extraction and merge configs.
    pub fn from_upstream_with_merge(
        extraction: &negsyn_types::ExtractionConfig,
        merge: &negsyn_types::MergeConfig,
    ) -> Self {
        let mut cfg = Self::from_upstream(extraction);
        cfg.max_states = merge.max_states as usize;
        cfg.merge_config = Some(merge.clone());
        cfg
    }
}

// ---------------------------------------------------------------------------
// Extraction metrics
// ---------------------------------------------------------------------------

/// Metrics collected during the extraction pipeline.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExtractionMetrics {
    pub traces_processed: u64,
    pub trace_steps_total: u64,
    pub states_extracted: u64,
    pub transitions_extracted: u64,
    pub states_after_dedup: u64,
    pub transitions_after_dedup: u64,
    pub bisimulation_classes: u64,
    pub states_after_quotient: u64,
    pub transitions_after_quotient: u64,
    pub states_after_minimization: u64,
    pub transitions_after_minimization: u64,
    pub unreachable_eliminated: u64,
    pub redundant_transitions_eliminated: u64,
    pub refinement_iterations: u64,
    pub extraction_time_us: u64,
    pub bisimulation_time_us: u64,
    pub minimization_time_us: u64,
    pub total_time_us: u64,
}

impl ExtractionMetrics {
    /// Compute the overall state reduction ratio (0.0 = no reduction, 1.0 = fully reduced).
    pub fn reduction_ratio(&self) -> f64 {
        if self.states_extracted == 0 {
            return 0.0;
        }
        let final_states = if self.states_after_minimization > 0 {
            self.states_after_minimization
        } else if self.states_after_quotient > 0 {
            self.states_after_quotient
        } else {
            self.states_after_dedup
        };
        1.0 - (final_states as f64 / self.states_extracted as f64)
    }

    /// Compute transition reduction ratio.
    pub fn transition_reduction_ratio(&self) -> f64 {
        if self.transitions_extracted == 0 {
            return 0.0;
        }
        let final_trans = if self.transitions_after_minimization > 0 {
            self.transitions_after_minimization
        } else if self.transitions_after_quotient > 0 {
            self.transitions_after_quotient
        } else {
            self.transitions_after_dedup
        };
        1.0 - (final_trans as f64 / self.transitions_extracted as f64)
    }
}

impl fmt::Display for ExtractionMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Extraction Metrics:")?;
        writeln!(f, "  Traces processed:    {}", self.traces_processed)?;
        writeln!(f, "  States extracted:    {}", self.states_extracted)?;
        writeln!(f, "  After dedup:         {}", self.states_after_dedup)?;
        writeln!(f, "  After quotient:      {}", self.states_after_quotient)?;
        writeln!(f, "  After minimization:  {}", self.states_after_minimization)?;
        writeln!(
            f,
            "  State reduction:     {:.1}%",
            self.reduction_ratio() * 100.0
        )?;
        writeln!(
            f,
            "  Transition reduction:{:.1}%",
            self.transition_reduction_ratio() * 100.0
        )?;
        writeln!(f, "  Total time:          {}μs", self.total_time_us)
    }
}

// ---------------------------------------------------------------------------
// Extraction error
// ---------------------------------------------------------------------------

/// Errors specific to the extraction crate.
#[derive(Debug, thiserror::Error)]
pub enum ExtractError {
    #[error("no traces provided")]
    NoTraces,

    #[error("state limit exceeded: {count} > {limit}")]
    StateLimitExceeded { count: usize, limit: usize },

    #[error("transition limit exceeded: {count} > {limit}")]
    TransitionLimitExceeded { count: usize, limit: usize },

    #[error("bisimulation did not converge after {iterations} iterations")]
    BisimulationDiverged { iterations: u32 },

    #[error("simulation check failed: {reason}")]
    SimulationFailed { reason: String },

    #[error("no initial state found")]
    NoInitialState,

    #[error("quotient construction failed: {reason}")]
    QuotientFailed { reason: String },

    #[error("serialization error: {0}")]
    Serialization(String),

    #[error("timeout after {ms}ms")]
    Timeout { ms: u64 },

    #[error("type inference failed: {reason}")]
    TypeInferenceFailed { reason: String },

    #[error("pattern extraction failed: {pattern}")]
    PatternExtractionFailed { pattern: String },

    #[error("upstream error: {0}")]
    Upstream(#[from] NegSynthError),

    #[error("internal: {0}")]
    Internal(String),
}

/// Result type alias for extraction operations.
pub type ExtractResult<T> = std::result::Result<T, ExtractError>;

// ---------------------------------------------------------------------------
// Re-export sub-module items for convenience
// ---------------------------------------------------------------------------

pub use bisimulation::{BisimulationChecker, BisimulationRelation};
pub use extractor::{ExtractionContext, StateMachineExtractor};
pub use minimize::Minimizer;
pub use observation::{ObservationDomain, ObservationFunction};
pub use quotient::{QuotientBuilder, QuotientLTS, QuotientState, QuotientTransition};
pub use serialize::LtsSerializer;
pub use simulation::{SimulationChecker, SimulationRelation};
pub use trace::{SymbolicTrace, SymbolicTraceStep, TraceCollector};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_negotiation_state(phase: HandshakePhase) -> NegotiationState {
        NegotiationState {
            phase,
            offered_ciphers: Vec::new(),
            selected_cipher: None,
            version: Some(ProtocolVersion::tls12()),
            extensions: Vec::new(),
            random_client: None,
            random_server: None,
            session_id: None,
            is_resumption: false,
        }
    }

    fn make_completed_negotiation(cipher_id: u16) -> NegotiationState {
        let cipher = CipherSuite::new(
            cipher_id,
            format!("TEST_CIPHER_0x{:04x}", cipher_id),
            negsyn_types::protocol::KeyExchange::ECDHE,
            negsyn_types::protocol::AuthAlgorithm::RSA,
            negsyn_types::protocol::EncryptionAlgorithm::AES128GCM,
            negsyn_types::protocol::MacAlgorithm::AEAD,
            SecurityLevel::Standard,
        );
        NegotiationState {
            phase: HandshakePhase::Done,
            offered_ciphers: vec![cipher.clone()],
            selected_cipher: Some(cipher),
            version: Some(ProtocolVersion::tls12()),
            extensions: Vec::new(),
            random_client: None,
            random_server: None,
            session_id: None,
            is_resumption: false,
        }
    }

    #[test]
    fn test_extraction_lts_basic_operations() {
        let mut lts = ExtractionLTS::new();
        let s0 = lts.add_state(make_negotiation_state(HandshakePhase::Init));
        let s1 = lts.add_state(make_negotiation_state(HandshakePhase::ClientHelloSent));
        lts.mark_initial(s0);
        let label = TransitionLabel::ClientAction(
            negsyn_types::protocol::ClientActionKind::SendClientHello {
                ciphers: vec![0x002f, 0x0035],
                version: 0x0303,
            },
        );
        lts.add_transition(s0, s1, label);
        assert_eq!(lts.state_count(), 2);
        assert_eq!(lts.transition_count(), 1);
        assert_eq!(lts.initial_states.len(), 1);
        assert_eq!(lts.successors(s0).len(), 1);
    }

    #[test]
    fn test_observable_from_negotiation() {
        let ns = make_completed_negotiation(0x002f);
        let obs = Observable::from_negotiation_state(&ns);
        assert_eq!(obs.outcome, OutcomeKind::Completed);
        assert_eq!(obs.selected_cipher, Some(0x002f));

        let ns_abort = make_negotiation_state(HandshakePhase::Abort);
        let obs_abort = Observable::from_negotiation_state(&ns_abort);
        assert_eq!(obs_abort.outcome, OutcomeKind::Aborted);

        let ns_prog = make_negotiation_state(HandshakePhase::ClientHelloSent);
        let obs_prog = Observable::from_negotiation_state(&ns_prog);
        assert_eq!(obs_prog.outcome, OutcomeKind::InProgress);
    }

    #[test]
    fn test_observable_agreement() {
        let o1 = Observable::successful(0x002f, ProtocolVersion::tls12(), BTreeSet::new());
        let o2 = Observable::successful(0x002f, ProtocolVersion::tls12(), BTreeSet::new());
        let o3 = Observable::successful(0x0035, ProtocolVersion::tls12(), BTreeSet::new());
        assert!(o1.agrees_with(&o2));
        assert!(!o1.agrees_with(&o3));
    }

    #[test]
    fn test_observable_downgrade() {
        let baseline =
            Observable::successful(0x002f, ProtocolVersion::tls12(), BTreeSet::new());
        let downgraded =
            Observable::successful(0x0035, ProtocolVersion::tls12(), BTreeSet::new());
        assert!(downgraded.is_downgrade_from(&baseline));

        let same =
            Observable::successful(0x002f, ProtocolVersion::tls12(), BTreeSet::new());
        assert!(!same.is_downgrade_from(&baseline));
    }

    #[test]
    fn test_extraction_config_default() {
        let cfg = LocalExtractionConfig::default();
        assert!(cfg.enable_bisimulation);
        assert!(cfg.enable_minimization);
        assert!(cfg.max_states > 0);
        assert!(cfg.confidence_threshold > 0.0);
    }

    #[test]
    fn test_extraction_config_from_upstream() {
        let upstream = negsyn_types::ExtractionConfig {
            cipher_patterns: vec!["TLS_*".into()],
            version_patterns: vec![],
            extract_extensions: true,
            type_inference: true,
            confidence_threshold: 0.95,
        };
        let cfg = LocalExtractionConfig::from_upstream(&upstream);
        assert!((cfg.confidence_threshold - 0.95).abs() < 0.001);
        assert!(cfg.type_config.is_some());
    }

    #[test]
    fn test_extraction_metrics_reduction() {
        let mut m = ExtractionMetrics::default();
        m.states_extracted = 100;
        m.states_after_minimization = 10;
        m.transitions_extracted = 200;
        m.transitions_after_minimization = 20;
        assert!((m.reduction_ratio() - 0.9).abs() < 0.001);
        assert!((m.transition_reduction_ratio() - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_extraction_lts_to_negotiation_lts() {
        let mut lts = ExtractionLTS::new();
        let s0 = lts.add_state(make_negotiation_state(HandshakePhase::Init));
        let s1 = lts.add_state(make_completed_negotiation(0x002f));
        lts.mark_initial(s0);
        lts.add_transition(s0, s1, TransitionLabel::Tau);
        let nlts = lts.to_negotiation_lts();
        let (ns, nt) = nlts.size();
        assert_eq!(ns, 2);
        assert_eq!(nt, 1);
    }

    #[test]
    fn test_no_op_merge_operator() {
        let op = NoOpMergeOperator;
        let s1 = SymbolicState::new(1, 0);
        let s2 = SymbolicState::new(2, 0);
        assert!(!op.is_mergeable(&s1, &s2));
        assert!(op.merge_states(&s1, &s2).is_err());
    }

    #[test]
    fn test_lts_state_from_abort() {
        let ns = make_negotiation_state(HandshakePhase::Abort);
        let sid = StateId(0);
        let lts = LtsState::new(sid, ns);
        assert!(lts.is_terminal);
        assert_eq!(lts.observation.outcome, OutcomeKind::Aborted);
    }

    #[test]
    fn test_extract_error_display() {
        let e = ExtractError::StateLimitExceeded {
            count: 200,
            limit: 100,
        };
        let s = format!("{}", e);
        assert!(s.contains("200"));
        assert!(s.contains("100"));
    }

    #[test]
    fn test_outcome_kind_equality() {
        assert_eq!(OutcomeKind::Completed, OutcomeKind::Completed);
        assert_ne!(OutcomeKind::Completed, OutcomeKind::Aborted);
        assert_ne!(OutcomeKind::Aborted, OutcomeKind::InProgress);
    }
}
