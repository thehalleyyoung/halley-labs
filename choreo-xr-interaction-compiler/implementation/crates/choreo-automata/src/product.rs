//! Product automaton composition: parallel, synchronous, on-the-fly,
//! and symbolic product construction for spatial event automata.

use crate::automaton::{
    SpatialEventAutomaton, State, Transition,
};
use crate::{
    AutomataError, EventKind, Guard, Result,
    SpatialPredicateId, StateId,
};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};use std::fmt;

// ---------------------------------------------------------------------------
// ProductState
// ---------------------------------------------------------------------------

/// A composite state in a product automaton, tracking which states each
/// component automaton is in.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProductState {
    pub components: Vec<StateId>,
}

impl ProductState {
    pub fn new(components: Vec<StateId>) -> Self {
        Self { components }
    }

    pub fn pair(a: StateId, b: StateId) -> Self {
        Self {
            components: vec![a, b],
        }
    }
}

impl fmt::Display for ProductState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let parts: Vec<String> = self.components.iter().map(|s| format!("{}", s)).collect();
        write!(f, "({})", parts.join(", "))
    }
}

// ---------------------------------------------------------------------------
// ProductComposer
// ---------------------------------------------------------------------------

/// Composes two or more spatial event automata into a product automaton.
pub struct ProductComposer {
    /// Maximum number of product states before aborting (guards against blow-up).
    pub max_states: usize,
    /// Whether to apply spatial-independence optimisation.
    pub use_independence_opt: bool,
}

impl ProductComposer {
    pub fn new() -> Self {
        Self {
            max_states: 100_000,
            use_independence_opt: true,
        }
    }

    pub fn with_max_states(mut self, max: usize) -> Self {
        self.max_states = max;
        self
    }

    pub fn with_independence_opt(mut self, enabled: bool) -> Self {
        self.use_independence_opt = enabled;
        self
    }

    // -----------------------------------------------------------------------
    // Parallel composition (full interleaving)
    // -----------------------------------------------------------------------

    /// Parallel (interleaving) composition of two automata.
    ///
    /// Every transition of automaton A is interleaved with every transition of
    /// automaton B.  Shared events synchronise; independent events interleave.
    pub fn parallel_composition(
        &self,
        a: &SpatialEventAutomaton,
        b: &SpatialEventAutomaton,
    ) -> Result<SpatialEventAutomaton> {
        self.synchronous_composition(a, b, &HashSet::new())
    }

    // -----------------------------------------------------------------------
    // Synchronous composition (shared events must fire together)
    // -----------------------------------------------------------------------

    /// Synchronous composition where `sync_events` must be taken
    /// simultaneously by both automata; other events interleave freely.
    pub fn synchronous_composition(
        &self,
        a: &SpatialEventAutomaton,
        b: &SpatialEventAutomaton,
        sync_events: &HashSet<EventKind>,
    ) -> Result<SpatialEventAutomaton> {
        let init_a = a
            .initial_state
            .ok_or_else(|| AutomataError::CompositionError("Automaton A has no initial state".into()))?;
        let init_b = b
            .initial_state
            .ok_or_else(|| AutomataError::CompositionError("Automaton B has no initial state".into()))?;

        // Spatial independence check
        if self.use_independence_opt {
            let analysis = SharedPredicateAnalysis::analyze(a, b);
            if analysis.shared_predicates.is_empty() && sync_events.is_empty() {
                log::debug!(
                    "Automata share no predicates and no sync events – independent composition"
                );
            }
        }

        let mut product = SpatialEventAutomaton::new(format!(
            "{}×{}",
            a.metadata.name, b.metadata.name
        ));
        let mut state_map: HashMap<(StateId, StateId), StateId> = HashMap::new();
        let mut worklist: VecDeque<(StateId, StateId)> = VecDeque::new();

        // Initial product state
        let init_pid = product.fresh_state_id();
        let is_acc_a = a.accepting_states.contains(&init_a);
        let is_acc_b = b.accepting_states.contains(&init_b);
        let mut init_state = State::new(init_pid, format!("({},{})", init_a, init_b));
        init_state.is_initial = true;
        init_state.is_accepting = is_acc_a && is_acc_b;
        product.add_state(init_state);
        state_map.insert((init_a, init_b), init_pid);
        worklist.push_back((init_a, init_b));

        while let Some((sa, sb)) = worklist.pop_front() {
            if product.state_count() > self.max_states {
                return Err(AutomataError::CompositionError(format!(
                    "Product state space exceeded limit of {}",
                    self.max_states
                )));
            }
            let src_pid = state_map[&(sa, sb)];

            // Transitions from A only (events not in sync_events)
            for ta_id in a.outgoing(sa) {
                if let Some(ta) = a.transition(ta_id) {
                    let evt = extract_event(&ta.guard);
                    if evt.as_ref().map_or(true, |e| !sync_events.contains(e)) {
                        let tgt = get_or_create_product_state(
                            &mut product,
                            &mut state_map,
                            &mut worklist,
                            ta.target,
                            sb,
                            a,
                            b,
                            self.max_states,
                        )?;
                        let tid = product.fresh_transition_id();
                        let guard = ta.guard.clone();
                        let actions = ta.actions.clone();
                        product.add_transition(Transition::new(tid, src_pid, tgt, guard, actions));
                    }
                }
            }

            // Transitions from B only (events not in sync_events)
            for tb_id in b.outgoing(sb) {
                if let Some(tb) = b.transition(tb_id) {
                    let evt = extract_event(&tb.guard);
                    if evt.as_ref().map_or(true, |e| !sync_events.contains(e)) {
                        let tgt = get_or_create_product_state(
                            &mut product,
                            &mut state_map,
                            &mut worklist,
                            sa,
                            tb.target,
                            a,
                            b,
                            self.max_states,
                        )?;
                        let tid = product.fresh_transition_id();
                        let guard = tb.guard.clone();
                        let actions = tb.actions.clone();
                        product.add_transition(Transition::new(tid, src_pid, tgt, guard, actions));
                    }
                }
            }

            // Synchronous transitions (shared events)
            for ta_id in a.outgoing(sa) {
                if let Some(ta) = a.transition(ta_id) {
                    let evt_a = extract_event(&ta.guard);
                    if let Some(ref ea) = evt_a {
                        if sync_events.contains(ea) {
                            for tb_id in b.outgoing(sb) {
                                if let Some(tb) = b.transition(tb_id) {
                                    let evt_b = extract_event(&tb.guard);
                                    if evt_b.as_ref() == Some(ea) {
                                        let tgt = get_or_create_product_state(
                                            &mut product,
                                            &mut state_map,
                                            &mut worklist,
                                            ta.target,
                                            tb.target,
                                            a,
                                            b,
                                            self.max_states,
                                        )?;
                                        let tid = product.fresh_transition_id();
                                        let guard = Guard::And(vec![
                                            ta.guard.clone(),
                                            tb.guard.clone(),
                                        ]);
                                        let mut actions = ta.actions.clone();
                                        actions.extend(tb.actions.clone());
                                        product.add_transition(Transition::new(
                                            tid, src_pid, tgt, guard, actions,
                                        ));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        product.recompute_statistics();
        Ok(product)
    }

    // -----------------------------------------------------------------------
    // Compose multiple automata
    // -----------------------------------------------------------------------

    /// Compose an arbitrary number of automata pairwise.
    pub fn compose_multiple(
        &self,
        automata: &[SpatialEventAutomaton],
    ) -> Result<SpatialEventAutomaton> {
        if automata.is_empty() {
            return Err(AutomataError::CompositionError(
                "No automata to compose".into(),
            ));
        }
        if automata.len() == 1 {
            return Ok(automata[0].clone());
        }

        // Optional independence grouping
        if self.use_independence_opt && automata.len() > 2 {
            let groups = partition_independent(automata);
            if groups.len() > 1 {
                log::debug!(
                    "Partitioned {} automata into {} independent groups",
                    automata.len(),
                    groups.len()
                );
                let mut group_results = Vec::new();
                for group in &groups {
                    let group_autos: Vec<&SpatialEventAutomaton> =
                        group.iter().map(|&i| &automata[i]).collect();
                    let composed = self.compose_group(&group_autos)?;
                    group_results.push(composed);
                }
                return self.compose_group(
                    &group_results.iter().collect::<Vec<_>>(),
                );
            }
        }

        let refs: Vec<&SpatialEventAutomaton> = automata.iter().collect();
        self.compose_group(&refs)
    }

    fn compose_group(
        &self,
        automata: &[&SpatialEventAutomaton],
    ) -> Result<SpatialEventAutomaton> {
        if automata.is_empty() {
            return Err(AutomataError::CompositionError(
                "Empty group".into(),
            ));
        }
        let mut result = automata[0].clone();
        for i in 1..automata.len() {
            result = self.parallel_composition(&result, automata[i])?;
        }
        Ok(result)
    }
}

impl Default for ProductComposer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// On-the-fly product construction
// ---------------------------------------------------------------------------

/// Result of on-the-fly product exploration.
#[derive(Debug, Clone)]
pub struct ProductExploration {
    pub explored_states: Vec<ProductState>,
    pub explored_transitions: Vec<(ProductState, ProductState, Guard)>,
    pub frontier: Vec<ProductState>,
    pub is_complete: bool,
}

/// On-the-fly product construction that lazily materialises only reachable
/// product states.
pub struct OnTheFlyProduct<'a> {
    pub automata: Vec<&'a SpatialEventAutomaton>,
    pub explored: HashMap<Vec<StateId>, bool>,
    pub max_explore: usize,
}

impl<'a> OnTheFlyProduct<'a> {
    pub fn new(automata: Vec<&'a SpatialEventAutomaton>) -> Self {
        Self {
            automata,
            explored: HashMap::new(),
            max_explore: 50_000,
        }
    }

    pub fn with_max(mut self, max: usize) -> Self {
        self.max_explore = max;
        self
    }

    /// Explore the product state space from the initial product state.
    pub fn explore_from_initial(&mut self) -> Result<ProductExploration> {
        let initial: Vec<StateId> = self
            .automata
            .iter()
            .filter_map(|a| a.initial_state)
            .collect();
        if initial.len() != self.automata.len() {
            return Err(AutomataError::CompositionError(
                "Not all automata have initial states".into(),
            ));
        }
        self.explore_from(&ProductState::new(initial))
    }

    /// Explore from a given product state.
    pub fn explore_from(&mut self, initial: &ProductState) -> Result<ProductExploration> {
        let mut explored_states = Vec::new();
        let mut explored_transitions = Vec::new();
        let mut frontier = Vec::new();
        let mut worklist: VecDeque<Vec<StateId>> = VecDeque::new();
        let mut visited: HashSet<Vec<StateId>> = HashSet::new();

        worklist.push_back(initial.components.clone());
        visited.insert(initial.components.clone());

        while let Some(current) = worklist.pop_front() {
            if explored_states.len() >= self.max_explore {
                // Return partial exploration
                for remaining in worklist {
                    frontier.push(ProductState::new(remaining));
                }
                return Ok(ProductExploration {
                    explored_states,
                    explored_transitions,
                    frontier,
                    is_complete: false,
                });
            }

            explored_states.push(ProductState::new(current.clone()));
            self.explored.insert(current.clone(), true);

            // Generate successors by interleaving
            for (i, auto) in self.automata.iter().enumerate() {
                let si = current[i];
                for tid in auto.outgoing(si) {
                    if let Some(t) = auto.transition(tid) {
                        let mut next = current.clone();
                        next[i] = t.target;
                        explored_transitions.push((
                            ProductState::new(current.clone()),
                            ProductState::new(next.clone()),
                            t.guard.clone(),
                        ));
                        if visited.insert(next.clone()) {
                            worklist.push_back(next);
                        }
                    }
                }
            }
        }

        Ok(ProductExploration {
            explored_states,
            explored_transitions,
            frontier,
            is_complete: true,
        })
    }

    /// Check whether a product state has been explored.
    pub fn is_explored(&self, state: &ProductState) -> bool {
        self.explored.contains_key(&state.components)
    }
}

// ---------------------------------------------------------------------------
// Symbolic product state representation (BDD-like encoding)
// ---------------------------------------------------------------------------

/// Bitvector-based symbolic product state.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SymbolicProductState {
    /// Each entry encodes the state index of one automaton component using
    /// `bits_per_component` bits.
    pub bitvector: Vec<u64>,
    pub bits_per_component: usize,
    pub num_components: usize,
}

impl SymbolicProductState {
    /// Create from a list of state indices and bits-per-component.
    pub fn encode(state_indices: &[u32], bits_per_component: usize) -> Self {
        let total_bits = state_indices.len() * bits_per_component;
        let num_words = (total_bits + 63) / 64;
        let mut bitvector = vec![0u64; num_words];

        for (i, &idx) in state_indices.iter().enumerate() {
            let bit_offset = i * bits_per_component;
            let word_idx = bit_offset / 64;
            let bit_pos = bit_offset % 64;
            bitvector[word_idx] |= (idx as u64) << bit_pos;
            // Handle overflow into next word
            if bit_pos + bits_per_component > 64 && word_idx + 1 < num_words {
                let overflow = bit_pos + bits_per_component - 64;
                bitvector[word_idx + 1] |= (idx as u64) >> (bits_per_component - overflow);
            }
        }

        Self {
            bitvector,
            bits_per_component,
            num_components: state_indices.len(),
        }
    }

    /// Decode back to state indices.
    pub fn decode(&self) -> Vec<u32> {
        let mask = (1u64 << self.bits_per_component) - 1;
        let mut result = Vec::with_capacity(self.num_components);
        for i in 0..self.num_components {
            let bit_offset = i * self.bits_per_component;
            let word_idx = bit_offset / 64;
            let bit_pos = bit_offset % 64;
            let mut val = (self.bitvector[word_idx] >> bit_pos) & mask;
            // Handle cross-word boundary
            if bit_pos + self.bits_per_component > 64 && word_idx + 1 < self.bitvector.len() {
                let high_bits = self.bits_per_component - (64 - bit_pos);
                let high_mask = (1u64 << high_bits) - 1;
                val |= (self.bitvector[word_idx + 1] & high_mask) << (64 - bit_pos);
            }
            result.push(val as u32);
        }
        result
    }

    /// Get the state index for a specific component.
    pub fn component(&self, idx: usize) -> u32 {
        let bit_offset = idx * self.bits_per_component;
        let word_idx = bit_offset / 64;
        let bit_pos = bit_offset % 64;
        let mask = (1u64 << self.bits_per_component) - 1;
        let mut val = (self.bitvector[word_idx] >> bit_pos) & mask;
        if bit_pos + self.bits_per_component > 64 && word_idx + 1 < self.bitvector.len() {
            let high_bits = self.bits_per_component - (64 - bit_pos);
            let high_mask = (1u64 << high_bits) - 1;
            val |= (self.bitvector[word_idx + 1] & high_mask) << (64 - bit_pos);
        }
        val as u32
    }

    /// Set the state index for a specific component, returning a new state.
    pub fn with_component(&self, idx: usize, value: u32) -> Self {
        let mut indices = self.decode();
        indices[idx] = value;
        Self::encode(&indices, self.bits_per_component)
    }
}

// ---------------------------------------------------------------------------
// SymbolicTransitionRelation
// ---------------------------------------------------------------------------

/// Symbolic transition relation for efficient product composition.
/// Stores transitions as bitvector pairs.
#[derive(Debug, Clone)]
pub struct SymbolicTransitionRelation {
    pub bits_per_component: usize,
    pub num_components: usize,
    /// (source_symbolic, target_symbolic, guard, component_index)
    pub transitions: Vec<(SymbolicProductState, SymbolicProductState, Guard, usize)>,
}

impl SymbolicTransitionRelation {
    pub fn new(bits_per_component: usize, num_components: usize) -> Self {
        Self {
            bits_per_component,
            num_components,
            transitions: Vec::new(),
        }
    }

    /// Build from a set of component automata.
    pub fn from_automata(automata: &[&SpatialEventAutomaton]) -> Self {
        let max_states = automata
            .iter()
            .map(|a| a.state_count())
            .max()
            .unwrap_or(1);
        let bits = (max_states as f64).log2().ceil().max(1.0) as usize;
        let num = automata.len();

        let mut rel = Self::new(bits, num);

        // Build initial product state indices
        let initial_indices: Vec<u32> = automata
            .iter()
            .map(|a| a.initial_state.map_or(0, |s| s.0))
            .collect();

        // For each component, add its transitions
        for (comp_idx, auto) in automata.iter().enumerate() {
            for t in auto.transitions.values() {
                let mut src_indices = initial_indices.clone();
                src_indices[comp_idx] = t.source.0;
                let src = SymbolicProductState::encode(&src_indices, bits);

                let mut tgt_indices = initial_indices.clone();
                tgt_indices[comp_idx] = t.target.0;
                let tgt = SymbolicProductState::encode(&tgt_indices, bits);

                rel.transitions
                    .push((src, tgt, t.guard.clone(), comp_idx));
            }
        }

        rel
    }

    /// Compute the image (set of successors) of a symbolic state.
    pub fn image(&self, state: &SymbolicProductState) -> Vec<(SymbolicProductState, Guard)> {
        let mut successors = Vec::new();
        let current = state.decode();

        for (src, tgt, guard, comp_idx) in &self.transitions {
            if src.component(*comp_idx) == current[*comp_idx] {
                let mut next = current.clone();
                next[*comp_idx] = tgt.component(*comp_idx);
                successors.push((
                    SymbolicProductState::encode(&next, self.bits_per_component),
                    guard.clone(),
                ));
            }
        }
        successors
    }

    /// Compute the pre-image (set of predecessors) of a symbolic state.
    pub fn pre_image(&self, state: &SymbolicProductState) -> Vec<(SymbolicProductState, Guard)> {
        let mut predecessors = Vec::new();
        let current = state.decode();

        for (src, tgt, guard, comp_idx) in &self.transitions {
            if tgt.component(*comp_idx) == current[*comp_idx] {
                let mut prev = current.clone();
                prev[*comp_idx] = src.component(*comp_idx);
                predecessors.push((
                    SymbolicProductState::encode(&prev, self.bits_per_component),
                    guard.clone(),
                ));
            }
        }
        predecessors
    }
}

// ---------------------------------------------------------------------------
// SharedPredicateAnalysis
// ---------------------------------------------------------------------------

/// Analyses which spatial predicates are shared between two automata.
#[derive(Debug, Clone)]
pub struct SharedPredicateAnalysis {
    pub shared_predicates: HashSet<SpatialPredicateId>,
    pub a_only_predicates: HashSet<SpatialPredicateId>,
    pub b_only_predicates: HashSet<SpatialPredicateId>,
    pub shared_events: HashSet<EventKind>,
    pub independence_score: f64,
}

impl SharedPredicateAnalysis {
    /// Analyse two automata for shared predicates and events.
    pub fn analyze(
        a: &SpatialEventAutomaton,
        b: &SpatialEventAutomaton,
    ) -> Self {
        let preds_a = collect_predicate_ids(a);
        let preds_b = collect_predicate_ids(b);
        let events_a = a.alphabet();
        let events_b = b.alphabet();

        let shared_preds: HashSet<SpatialPredicateId> =
            preds_a.intersection(&preds_b).cloned().collect();
        let a_only: HashSet<SpatialPredicateId> =
            preds_a.difference(&preds_b).cloned().collect();
        let b_only: HashSet<SpatialPredicateId> =
            preds_b.difference(&preds_a).cloned().collect();
        let shared_evts: HashSet<EventKind> =
            events_a.intersection(&events_b).cloned().collect();

        let total = (preds_a.len() + preds_b.len()).max(1) as f64;
        let shared_count = shared_preds.len() as f64;
        let independence = 1.0 - (shared_count / total);

        Self {
            shared_predicates: shared_preds,
            a_only_predicates: a_only,
            b_only_predicates: b_only,
            shared_events: shared_evts,
            independence_score: independence,
        }
    }
}

// ---------------------------------------------------------------------------
// ProductStateSpace
// ---------------------------------------------------------------------------

/// Efficient representation of the product state space, allowing enumeration
/// and membership queries.
#[derive(Debug, Clone)]
pub struct ProductStateSpace {
    pub states: HashSet<Vec<StateId>>,
    pub transitions: Vec<(Vec<StateId>, Vec<StateId>, Guard)>,
    pub initial: Option<Vec<StateId>>,
    pub accepting: HashSet<Vec<StateId>>,
}

impl ProductStateSpace {
    pub fn new() -> Self {
        Self {
            states: HashSet::new(),
            transitions: Vec::new(),
            initial: None,
            accepting: HashSet::new(),
        }
    }

    /// Build from a set of component automata via explicit enumeration.
    pub fn from_automata(automata: &[&SpatialEventAutomaton], max_states: usize) -> Result<Self> {
        let mut space = Self::new();
        let initial: Vec<StateId> = automata
            .iter()
            .filter_map(|a| a.initial_state)
            .collect();
        if initial.len() != automata.len() {
            return Err(AutomataError::CompositionError(
                "Not all automata have initial states".into(),
            ));
        }

        space.initial = Some(initial.clone());
        let mut worklist: VecDeque<Vec<StateId>> = VecDeque::new();
        worklist.push_back(initial.clone());
        space.states.insert(initial.clone());

        // Check accepting
        let is_accepting = |state: &[StateId]| -> bool {
            state
                .iter()
                .zip(automata.iter())
                .all(|(s, a)| a.accepting_states.contains(s))
        };

        if is_accepting(&initial) {
            space.accepting.insert(initial);
        }

        while let Some(current) = worklist.pop_front() {
            if space.states.len() > max_states {
                return Err(AutomataError::CompositionError(format!(
                    "Product state space exceeded {}",
                    max_states,
                )));
            }

            for (comp_idx, auto) in automata.iter().enumerate() {
                let si = current[comp_idx];
                for tid in auto.outgoing(si) {
                    if let Some(t) = auto.transition(tid) {
                        let mut next = current.clone();
                        next[comp_idx] = t.target;
                        space.transitions.push((
                            current.clone(),
                            next.clone(),
                            t.guard.clone(),
                        ));
                        if space.states.insert(next.clone()) {
                            if is_accepting(&next) {
                                space.accepting.insert(next.clone());
                            }
                            worklist.push_back(next);
                        }
                    }
                }
            }
        }

        Ok(space)
    }

    pub fn state_count(&self) -> usize {
        self.states.len()
    }

    pub fn transition_count(&self) -> usize {
        self.transitions.len()
    }

    pub fn contains(&self, state: &[StateId]) -> bool {
        self.states.contains(state)
    }
}

impl Default for ProductStateSpace {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract the event kind from a guard (if it contains an Event guard).
fn extract_event(guard: &Guard) -> Option<EventKind> {
    match guard {
        Guard::Event(ek) => Some(ek.clone()),
        Guard::And(gs) => gs.iter().find_map(|g| extract_event(g)),
        Guard::Or(gs) => gs.iter().find_map(|g| extract_event(g)),
        _ => None,
    }
}

/// Get or create a product state in the product automaton.
fn get_or_create_product_state(
    product: &mut SpatialEventAutomaton,
    state_map: &mut HashMap<(StateId, StateId), StateId>,
    worklist: &mut VecDeque<(StateId, StateId)>,
    sa: StateId,
    sb: StateId,
    a: &SpatialEventAutomaton,
    b: &SpatialEventAutomaton,
    max_states: usize,
) -> Result<StateId> {
    if let Some(&existing) = state_map.get(&(sa, sb)) {
        return Ok(existing);
    }
    if state_map.len() >= max_states {
        return Err(AutomataError::CompositionError(format!(
            "Product state limit {} exceeded",
            max_states
        )));
    }
    let pid = product.fresh_state_id();
    let is_acc = a.accepting_states.contains(&sa) && b.accepting_states.contains(&sb);
    let mut ps = State::new(pid, format!("({},{})", sa, sb));
    ps.is_accepting = is_acc;
    product.add_state(ps);
    state_map.insert((sa, sb), pid);
    worklist.push_back((sa, sb));
    Ok(pid)
}

/// Collect all spatial predicate ids referenced by an automaton's guards.
fn collect_predicate_ids(auto: &SpatialEventAutomaton) -> HashSet<SpatialPredicateId> {
    let mut ids = HashSet::new();
    for t in auto.transitions.values() {
        for pid in t.guard.spatial_predicate_ids() {
            ids.insert(pid);
        }
    }
    ids
}

/// Partition automata into independent groups (those sharing no predicates or events).
fn partition_independent(automata: &[SpatialEventAutomaton]) -> Vec<Vec<usize>> {
    let n = automata.len();
    // Build predicate and event sets per automaton
    let pred_sets: Vec<HashSet<SpatialPredicateId>> =
        automata.iter().map(collect_predicate_ids).collect();
    let event_sets: Vec<HashSet<EventKind>> =
        automata.iter().map(|a| a.alphabet()).collect();

    // Union-Find
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut Vec<usize>, x: usize) -> usize {
        if parent[x] != x {
            parent[x] = find(parent, parent[x]);
        }
        parent[x]
    }
    fn union(parent: &mut Vec<usize>, a: usize, b: usize) {
        let ra = find(parent, a);
        let rb = find(parent, b);
        if ra != rb {
            parent[ra] = rb;
        }
    }

    for i in 0..n {
        for j in (i + 1)..n {
            let shared_preds = pred_sets[i].intersection(&pred_sets[j]).count();
            let shared_events = event_sets[i].intersection(&event_sets[j]).count();
            if shared_preds > 0 || shared_events > 0 {
                union(&mut parent, i, j);
            }
        }
    }

    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..n {
        let root = find(&mut parent, i);
        groups.entry(root).or_default().push(i);
    }
    groups.into_values().collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automaton::{State, Transition};
    use crate::builder::{AutomatonBuilder, StateConfig};
    use crate::*;

    fn make_ab_automaton(name: &str, events: (EventKind, EventKind)) -> SpatialEventAutomaton {
        let mut b = AutomatonBuilder::new(name);
        let s0 = b.add_state(StateConfig::new("s0").initial());
        let s1 = b.add_state(StateConfig::new("s1").accepting());
        b.add_transition(s0, s1, Guard::Event(events.0), vec![]);
        b.add_transition(s1, s0, Guard::Event(events.1), vec![]);
        b.build().unwrap()
    }

    #[test]
    fn test_parallel_composition() {
        let a = make_ab_automaton("a", (EventKind::GrabStart, EventKind::GrabEnd));
        let b = make_ab_automaton("b", (EventKind::TouchStart, EventKind::TouchEnd));
        let composer = ProductComposer::new();
        let product = composer.parallel_composition(&a, &b).unwrap();
        assert!(product.state_count() >= 4);
        assert!(product.transition_count() >= 4);
    }

    #[test]
    fn test_synchronous_composition() {
        let a = make_ab_automaton("a", (EventKind::GrabStart, EventKind::GrabEnd));
        let b = make_ab_automaton("b", (EventKind::GrabStart, EventKind::GrabEnd));
        let sync = HashSet::from([EventKind::GrabStart, EventKind::GrabEnd]);
        let composer = ProductComposer::new();
        let product = composer.synchronous_composition(&a, &b, &sync).unwrap();
        assert!(product.state_count() >= 2);
    }

    #[test]
    fn test_compose_multiple() {
        let autos = vec![
            make_ab_automaton("a", (EventKind::GrabStart, EventKind::GrabEnd)),
            make_ab_automaton("b", (EventKind::TouchStart, EventKind::TouchEnd)),
            make_ab_automaton("c", (EventKind::GazeEnter, EventKind::GazeExit)),
        ];
        let composer = ProductComposer::new();
        let product = composer.compose_multiple(&autos).unwrap();
        assert!(product.state_count() >= 8);
    }

    #[test]
    fn test_product_state() {
        let ps = ProductState::pair(StateId(0), StateId(1));
        assert_eq!(ps.components.len(), 2);
        let display = format!("{}", ps);
        assert!(display.contains("s0"));
    }

    #[test]
    fn test_on_the_fly_product() {
        let a = make_ab_automaton("a", (EventKind::GrabStart, EventKind::GrabEnd));
        let b = make_ab_automaton("b", (EventKind::TouchStart, EventKind::TouchEnd));
        let mut otf = OnTheFlyProduct::new(vec![&a, &b]);
        let result = otf.explore_from_initial().unwrap();
        assert!(result.is_complete);
        assert!(result.explored_states.len() >= 4);
    }

    #[test]
    fn test_symbolic_product_state() {
        let indices = vec![3u32, 5, 7];
        let encoded = SymbolicProductState::encode(&indices, 4);
        let decoded = encoded.decode();
        assert_eq!(decoded, indices);
    }

    #[test]
    fn test_symbolic_state_component() {
        let encoded = SymbolicProductState::encode(&[2, 4, 6], 4);
        assert_eq!(encoded.component(0), 2);
        assert_eq!(encoded.component(1), 4);
        assert_eq!(encoded.component(2), 6);
    }

    #[test]
    fn test_symbolic_with_component() {
        let encoded = SymbolicProductState::encode(&[1, 2, 3], 4);
        let modified = encoded.with_component(1, 7);
        assert_eq!(modified.component(0), 1);
        assert_eq!(modified.component(1), 7);
        assert_eq!(modified.component(2), 3);
    }

    #[test]
    fn test_symbolic_transition_relation() {
        let a = make_ab_automaton("a", (EventKind::GrabStart, EventKind::GrabEnd));
        let b = make_ab_automaton("b", (EventKind::TouchStart, EventKind::TouchEnd));
        let rel = SymbolicTransitionRelation::from_automata(&[&a, &b]);
        assert!(!rel.transitions.is_empty());
    }

    #[test]
    fn test_shared_predicate_analysis() {
        let a = make_ab_automaton("a", (EventKind::GrabStart, EventKind::GrabEnd));
        let b = make_ab_automaton("b", (EventKind::TouchStart, EventKind::TouchEnd));
        let analysis = SharedPredicateAnalysis::analyze(&a, &b);
        assert!(analysis.shared_predicates.is_empty());
        assert!(analysis.independence_score >= 0.0);
    }

    #[test]
    fn test_product_state_space() {
        let a = make_ab_automaton("a", (EventKind::GrabStart, EventKind::GrabEnd));
        let b = make_ab_automaton("b", (EventKind::TouchStart, EventKind::TouchEnd));
        let space = ProductStateSpace::from_automata(&[&a, &b], 1000).unwrap();
        assert_eq!(space.state_count(), 4);
        assert!(space.transition_count() >= 4);
    }

    #[test]
    fn test_product_state_space_overflow() {
        let a = make_ab_automaton("a", (EventKind::GrabStart, EventKind::GrabEnd));
        let b = make_ab_automaton("b", (EventKind::TouchStart, EventKind::TouchEnd));
        let result = ProductStateSpace::from_automata(&[&a, &b], 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_compose_multiple_single() {
        let a = make_ab_automaton("a", (EventKind::GrabStart, EventKind::GrabEnd));
        let composer = ProductComposer::new();
        let result = composer.compose_multiple(&[a.clone()]).unwrap();
        assert_eq!(result.state_count(), a.state_count());
    }

    #[test]
    fn test_compose_empty() {
        let composer = ProductComposer::new();
        let result = composer.compose_multiple(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_partition_independent() {
        let a = make_ab_automaton("a", (EventKind::GrabStart, EventKind::GrabEnd));
        let b = make_ab_automaton("b", (EventKind::TouchStart, EventKind::TouchEnd));
        let c = make_ab_automaton("c", (EventKind::GazeEnter, EventKind::GazeExit));
        let groups = partition_independent(&[a, b, c]);
        // All have independent events, so should be 3 groups
        assert_eq!(groups.len(), 3);
    }

    #[test]
    fn test_partition_dependent() {
        let a = make_ab_automaton("a", (EventKind::GrabStart, EventKind::GrabEnd));
        let b = make_ab_automaton("b", (EventKind::GrabStart, EventKind::GrabEnd));
        let groups = partition_independent(&[a, b]);
        assert_eq!(groups.len(), 1);
    }

    #[test]
    fn test_symbolic_relation_image() {
        let a = make_ab_automaton("a", (EventKind::GrabStart, EventKind::GrabEnd));
        let b = make_ab_automaton("b", (EventKind::TouchStart, EventKind::TouchEnd));
        let rel = SymbolicTransitionRelation::from_automata(&[&a, &b]);
        let initial = SymbolicProductState::encode(
            &[
                a.initial_state.unwrap().0,
                b.initial_state.unwrap().0,
            ],
            rel.bits_per_component,
        );
        let succs = rel.image(&initial);
        assert!(!succs.is_empty());
    }
}
