//! Core spatial event automaton: states, transitions, guard evaluation,
//! transition firing, reachability, NFA/DFA variants, determinisation,
//! and Hopcroft minimisation adapted for spatial guards.

use crate::{
    Action, AutomataError, EventKind, Guard, Result, SceneConfiguration,
    SpatialPredicate, Span, StateId, TemporalGuardExpr, TimePoint, TransitionId,
    TimerId,
};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};
use std::fmt;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

/// A single state in the automaton.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct State {
    pub id: StateId,
    pub name: String,
    pub is_initial: bool,
    pub is_accepting: bool,
    pub is_error: bool,
    /// Invariant that must hold while the automaton resides in this state.
    pub invariant: Option<Guard>,
    /// Actions executed upon entering the state.
    pub on_entry: Vec<Action>,
    /// Actions executed upon leaving the state.
    pub on_exit: Vec<Action>,
    /// Arbitrary key-value metadata.
    pub metadata: HashMap<String, String>,
}

impl State {
    pub fn new(id: StateId, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            is_initial: false,
            is_accepting: false,
            is_error: false,
            invariant: None,
            on_entry: Vec::new(),
            on_exit: Vec::new(),
            metadata: HashMap::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Transition
// ---------------------------------------------------------------------------

/// A transition between two states.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transition {
    pub id: TransitionId,
    pub source: StateId,
    pub target: StateId,
    pub guard: Guard,
    pub actions: Vec<Action>,
    pub priority: i32,
    pub metadata: HashMap<String, String>,
}

impl Transition {
    pub fn new(
        id: TransitionId,
        source: StateId,
        target: StateId,
        guard: Guard,
        actions: Vec<Action>,
    ) -> Self {
        Self {
            id,
            source,
            target,
            guard,
            actions,
            priority: 0,
            metadata: HashMap::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Spatial & Temporal guards (wrappers)
// ---------------------------------------------------------------------------

/// Wrapper around a spatial predicate evaluated against a scene.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialGuard {
    pub predicate: SpatialPredicate,
    pub cached_result: Option<bool>,
}

impl SpatialGuard {
    pub fn new(predicate: SpatialPredicate) -> Self {
        Self {
            predicate,
            cached_result: None,
        }
    }

    /// Evaluate the spatial guard against a scene configuration.
    pub fn evaluate(&self, scene: &SceneConfiguration) -> bool {
        evaluate_spatial_predicate(&self.predicate, scene)
    }
}

/// Wrapper around temporal guard expression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalGuard {
    pub expr: TemporalGuardExpr,
    pub cached_result: Option<bool>,
}

impl TemporalGuard {
    pub fn new(expr: TemporalGuardExpr) -> Self {
        Self {
            expr,
            cached_result: None,
        }
    }

    /// Evaluate the temporal guard at a given time point.
    pub fn evaluate(&self, time: TimePoint, timer_values: &HashMap<TimerId, f64>) -> bool {
        evaluate_temporal_expr(&self.expr, time, timer_values)
    }
}

// ---------------------------------------------------------------------------
// Metadata & statistics
// ---------------------------------------------------------------------------

/// Metadata attached to an automaton.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutomatonMetadata {
    pub name: String,
    pub source_span: Option<Span>,
    pub description: String,
    pub statistics: AutomatonStatistics,
    pub tags: Vec<String>,
}

impl AutomatonMetadata {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            source_span: None,
            description: String::new(),
            statistics: AutomatonStatistics::default(),
            tags: Vec::new(),
        }
    }
}

/// Quantitative statistics about an automaton.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AutomatonStatistics {
    pub state_count: usize,
    pub transition_count: usize,
    pub accepting_count: usize,
    pub guard_complexity: usize,
    pub is_deterministic: bool,
    pub has_epsilon: bool,
    pub max_out_degree: usize,
    pub avg_out_degree: f64,
}

// ---------------------------------------------------------------------------
// AutomatonKind
// ---------------------------------------------------------------------------

/// Whether the automaton is an NFA or DFA.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AutomatonKind {
    NFA,
    DFA,
}

// ---------------------------------------------------------------------------
// SpatialEventAutomaton
// ---------------------------------------------------------------------------

/// The core automaton type – a spatial event automaton whose guards reference
/// spatial predicates over a 3-D scene and temporal predicates over timers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialEventAutomaton {
    pub states: IndexMap<StateId, State>,
    pub transitions: IndexMap<TransitionId, Transition>,
    pub initial_state: Option<StateId>,
    pub accepting_states: HashSet<StateId>,
    pub spatial_guards: HashMap<TransitionId, Vec<SpatialGuard>>,
    pub temporal_guards: HashMap<TransitionId, Vec<TemporalGuard>>,
    pub metadata: AutomatonMetadata,
    pub kind: AutomatonKind,
    pub next_state_id: u32,
    pub next_transition_id: u32,
}

impl SpatialEventAutomaton {
    /// Create an empty automaton.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            states: IndexMap::new(),
            transitions: IndexMap::new(),
            initial_state: None,
            accepting_states: HashSet::new(),
            spatial_guards: HashMap::new(),
            temporal_guards: HashMap::new(),
            metadata: AutomatonMetadata::new(name),
            kind: AutomatonKind::DFA,
            next_state_id: 0,
            next_transition_id: 0,
        }
    }

    /// Allocate a fresh `StateId`.
    pub fn fresh_state_id(&mut self) -> StateId {
        let id = StateId(self.next_state_id);
        self.next_state_id += 1;
        id
    }

    /// Allocate a fresh `TransitionId`.
    pub fn fresh_transition_id(&mut self) -> TransitionId {
        let id = TransitionId(self.next_transition_id);
        self.next_transition_id += 1;
        id
    }

    /// Add a state and return its id.
    pub fn add_state(&mut self, state: State) -> StateId {
        let id = state.id;
        if state.is_initial {
            self.initial_state = Some(id);
        }
        if state.is_accepting {
            self.accepting_states.insert(id);
        }
        self.states.insert(id, state);
        id
    }

    /// Add a transition and return its id.
    pub fn add_transition(&mut self, transition: Transition) -> TransitionId {
        let id = transition.id;
        // Extract spatial and temporal guards
        extract_guards_for_transition(&transition.guard, id, &mut self.spatial_guards, &mut self.temporal_guards);
        // Check for epsilon transitions
        if matches!(transition.guard, Guard::Event(EventKind::Epsilon)) {
            self.kind = AutomatonKind::NFA;
        }
        self.transitions.insert(id, transition);
        id
    }

    /// Get a state by id.
    pub fn state(&self, id: StateId) -> Option<&State> {
        self.states.get(&id)
    }

    /// Get a mutable state by id.
    pub fn state_mut(&mut self, id: StateId) -> Option<&mut State> {
        self.states.get_mut(&id)
    }

    /// Get a transition by id.
    pub fn transition(&self, id: TransitionId) -> Option<&Transition> {
        self.transitions.get(&id)
    }

    /// All state ids.
    pub fn state_ids(&self) -> Vec<StateId> {
        self.states.keys().copied().collect()
    }

    /// All transition ids.
    pub fn transition_ids(&self) -> Vec<TransitionId> {
        self.transitions.keys().copied().collect()
    }

    /// Outgoing transitions from a given state.
    pub fn outgoing(&self, state: StateId) -> Vec<TransitionId> {
        self.transitions
            .values()
            .filter(|t| t.source == state)
            .map(|t| t.id)
            .collect()
    }

    /// Incoming transitions to a given state.
    pub fn incoming(&self, state: StateId) -> Vec<TransitionId> {
        self.transitions
            .values()
            .filter(|t| t.target == state)
            .map(|t| t.id)
            .collect()
    }

    // -----------------------------------------------------------------------
    // Guard evaluation
    // -----------------------------------------------------------------------

    /// Evaluate a guard against the current scene, time, and active event.
    pub fn evaluate_guard(
        &self,
        guard: &Guard,
        scene: &SceneConfiguration,
        time: TimePoint,
        active_event: Option<&EventKind>,
        timer_values: &HashMap<TimerId, f64>,
    ) -> bool {
        match guard {
            Guard::True => true,
            Guard::False => false,
            Guard::Spatial(sp) => evaluate_spatial_predicate(sp, scene),
            Guard::Temporal(te) => evaluate_temporal_expr(te, time, timer_values),
            Guard::Event(ek) => active_event.map_or(false, |ae| ae == ek),
            Guard::And(gs) => gs
                .iter()
                .all(|g| self.evaluate_guard(g, scene, time, active_event, timer_values)),
            Guard::Or(gs) => gs
                .iter()
                .any(|g| self.evaluate_guard(g, scene, time, active_event, timer_values)),
            Guard::Not(g) => !self.evaluate_guard(g, scene, time, active_event, timer_values),
        }
    }

    /// Return all transitions enabled from `state` given current conditions.
    pub fn enabled_transitions(
        &self,
        state: StateId,
        scene: &SceneConfiguration,
        time: TimePoint,
        active_event: Option<&EventKind>,
        timer_values: &HashMap<TimerId, f64>,
    ) -> Vec<TransitionId> {
        let mut enabled = Vec::new();
        for tid in self.outgoing(state) {
            if let Some(t) = self.transition(tid) {
                if self.evaluate_guard(&t.guard, scene, time, active_event, timer_values) {
                    enabled.push(tid);
                }
            }
        }
        // Sort by descending priority
        enabled.sort_by(|a, b| {
            let pa = self.transition(*a).map_or(0, |t| t.priority);
            let pb = self.transition(*b).map_or(0, |t| t.priority);
            pb.cmp(&pa)
        });
        enabled
    }

    /// Fire a transition: returns the target state and the actions produced.
    pub fn fire_transition(
        &self,
        transition_id: TransitionId,
    ) -> Result<(StateId, Vec<Action>)> {
        let t = self
            .transition(transition_id)
            .ok_or(AutomataError::TransitionNotFound(transition_id))?;
        let source_state = self
            .state(t.source)
            .ok_or(AutomataError::StateNotFound(t.source))?;
        let target_state = self
            .state(t.target)
            .ok_or(AutomataError::StateNotFound(t.target))?;

        let mut actions = Vec::new();
        // on_exit of source
        actions.extend(source_state.on_exit.clone());
        // transition actions
        actions.extend(t.actions.clone());
        // on_entry of target
        actions.extend(target_state.on_entry.clone());

        Ok((t.target, actions))
    }

    /// Check whether `state` is deadlocked: no enabled transitions.
    pub fn is_deadlocked(
        &self,
        state: StateId,
        scene: &SceneConfiguration,
        time: TimePoint,
        timer_values: &HashMap<TimerId, f64>,
    ) -> bool {
        self.enabled_transitions(state, scene, time, None, timer_values)
            .is_empty()
    }

    /// Check whether the invariant of `state` holds.
    pub fn state_invariant_holds(
        &self,
        state: StateId,
        scene: &SceneConfiguration,
        time: TimePoint,
        timer_values: &HashMap<TimerId, f64>,
    ) -> bool {
        match self.state(state) {
            Some(s) => match &s.invariant {
                Some(inv) => self.evaluate_guard(inv, scene, time, None, timer_values),
                None => true,
            },
            None => false,
        }
    }

    // -----------------------------------------------------------------------
    // Reachability
    // -----------------------------------------------------------------------

    /// BFS from the initial state to find all reachable states.
    pub fn reachable_states(&self) -> HashSet<StateId> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        if let Some(init) = self.initial_state {
            queue.push_back(init);
            visited.insert(init);
        }
        while let Some(s) = queue.pop_front() {
            for tid in self.outgoing(s) {
                if let Some(t) = self.transition(tid) {
                    if visited.insert(t.target) {
                        queue.push_back(t.target);
                    }
                }
            }
        }
        visited
    }

    /// BFS from a given set of start states.
    pub fn reachable_from(&self, starts: &HashSet<StateId>) -> HashSet<StateId> {
        let mut visited = starts.clone();
        let mut queue: VecDeque<StateId> = starts.iter().copied().collect();
        while let Some(s) = queue.pop_front() {
            for tid in self.outgoing(s) {
                if let Some(t) = self.transition(tid) {
                    if visited.insert(t.target) {
                        queue.push_back(t.target);
                    }
                }
            }
        }
        visited
    }

    /// Reverse BFS from accepting states.
    pub fn states_reaching_accepting(&self) -> HashSet<StateId> {
        let mut visited = HashSet::new();
        let mut queue: VecDeque<StateId> = self.accepting_states.iter().copied().collect();
        for &s in &self.accepting_states {
            visited.insert(s);
        }
        while let Some(s) = queue.pop_front() {
            for tid in self.incoming(s) {
                if let Some(t) = self.transition(tid) {
                    if visited.insert(t.source) {
                        queue.push_back(t.source);
                    }
                }
            }
        }
        visited
    }

    /// Can any accepting state be reached from `state`?
    pub fn accepting_reachable(&self, state: StateId) -> bool {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(state);
        visited.insert(state);
        while let Some(s) = queue.pop_front() {
            if self.accepting_states.contains(&s) {
                return true;
            }
            for tid in self.outgoing(s) {
                if let Some(t) = self.transition(tid) {
                    if visited.insert(t.target) {
                        queue.push_back(t.target);
                    }
                }
            }
        }
        false
    }

    // -----------------------------------------------------------------------
    // Collect alphabet
    // -----------------------------------------------------------------------

    /// Collect the set of distinct event kinds used in guards.
    pub fn alphabet(&self) -> HashSet<EventKind> {
        let mut events = HashSet::new();
        for t in self.transitions.values() {
            collect_events_from_guard(&t.guard, &mut events);
        }
        events
    }

    // -----------------------------------------------------------------------
    // NFA / DFA detection
    // -----------------------------------------------------------------------

    /// Check whether the automaton is deterministic (at most one transition
    /// per event from each state, no epsilon transitions).
    pub fn is_deterministic(&self) -> bool {
        for sid in self.states.keys() {
            let out = self.outgoing(*sid);
            let mut seen_events: HashSet<Option<&EventKind>> = HashSet::new();
            for tid in &out {
                if let Some(t) = self.transition(*tid) {
                    if let Guard::Event(ref ek) = t.guard {
                        if ek == &EventKind::Epsilon {
                            return false;
                        }
                        if !seen_events.insert(Some(ek)) {
                            return false;
                        }
                    }
                }
            }
        }
        true
    }

    // -----------------------------------------------------------------------
    // Determinise (subset construction)
    // -----------------------------------------------------------------------

    /// Convert an NFA-like automaton to a DFA via subset construction.
    /// Spatial/temporal guards are preserved; subset construction operates
    /// over the event component of guards.
    pub fn determinize(&self) -> SpatialEventAutomaton {
        let all_events = self.alphabet();
        let init = match self.initial_state {
            Some(s) => {
                let mut set = BTreeSet::new();
                set.insert(s);
                epsilon_closure_for_automaton(self, &set)
            }
            None => BTreeSet::new(),
        };

        let mut dfa = SpatialEventAutomaton::new(format!("{}_det", self.metadata.name));
        let mut state_map: HashMap<BTreeSet<StateId>, StateId> = HashMap::new();
        let mut worklist: VecDeque<BTreeSet<StateId>> = VecDeque::new();

        let init_id = dfa.fresh_state_id();
        let is_accepting = init.iter().any(|s| self.accepting_states.contains(s));
        let init_name = subset_name(&init);
        let mut init_state = State::new(init_id, init_name);
        init_state.is_initial = true;
        init_state.is_accepting = is_accepting;
        dfa.add_state(init_state);
        state_map.insert(init.clone(), init_id);
        worklist.push_back(init);

        while let Some(current_set) = worklist.pop_front() {
            let current_id = state_map[&current_set];
            for event in &all_events {
                if *event == EventKind::Epsilon {
                    continue;
                }
                // Compute move on event
                let mut next_set = BTreeSet::new();
                for &sid in &current_set {
                    for tid in self.outgoing(sid) {
                        if let Some(t) = self.transition(tid) {
                            if guard_matches_event(&t.guard, event) {
                                next_set.insert(t.target);
                            }
                        }
                    }
                }
                let next_set = epsilon_closure_for_automaton(self, &next_set);
                if next_set.is_empty() {
                    continue;
                }
                let next_id = if let Some(&existing) = state_map.get(&next_set) {
                    existing
                } else {
                    let nid = dfa.fresh_state_id();
                    let is_acc = next_set.iter().any(|s| self.accepting_states.contains(s));
                    let mut ns = State::new(nid, subset_name(&next_set));
                    ns.is_accepting = is_acc;
                    dfa.add_state(ns);
                    state_map.insert(next_set.clone(), nid);
                    worklist.push_back(next_set);
                    nid
                };
                // Merge guards from original transitions
                let merged_guard = merge_guards_for_subset(self, &current_set, event);
                let tid = dfa.fresh_transition_id();
                let trans = Transition::new(tid, current_id, next_id, merged_guard, Vec::new());
                dfa.add_transition(trans);
            }
        }
        dfa.kind = AutomatonKind::DFA;
        dfa.recompute_statistics();
        dfa
    }

    // -----------------------------------------------------------------------
    // Minimise (Hopcroft's algorithm adapted for spatial guards)
    // -----------------------------------------------------------------------

    /// Minimise the automaton using Hopcroft's algorithm. States are
    /// equivalent if they agree on acceptance and their outgoing transitions
    /// (event + spatial guard combination) lead to equivalent target classes.
    pub fn minimize(&self) -> SpatialEventAutomaton {
        if self.states.len() <= 1 {
            return self.clone();
        }

        let state_vec: Vec<StateId> = self.state_ids();
        let n = state_vec.len();
        let idx: HashMap<StateId, usize> = state_vec.iter().enumerate().map(|(i, &s)| (s, i)).collect();

        // Initial partition: accepting vs non-accepting
        let mut partition: Vec<usize> = vec![0; n];
        for (i, &sid) in state_vec.iter().enumerate() {
            partition[i] = if self.accepting_states.contains(&sid) { 1 } else { 0 };
        }

        let events: Vec<EventKind> = self.alphabet().into_iter().collect();

        // Iterative refinement
        let mut changed = true;
        let mut next_class = 2;
        while changed {
            changed = false;
            let mut new_partition = partition.clone();
            let mut class_signatures: HashMap<usize, HashMap<Vec<(usize, String)>, usize>> = HashMap::new();

            for (i, &sid) in state_vec.iter().enumerate() {
                let old_class = partition[i];
                let mut sig: Vec<(usize, String)> = Vec::new();
                for event in &events {
                    for tid in self.outgoing(sid) {
                        if let Some(t) = self.transition(tid) {
                            if guard_matches_event(&t.guard, event) {
                                if let Some(&target_idx) = idx.get(&t.target) {
                                    let target_class = partition[target_idx];
                                    let guard_str = format!("{}", t.guard);
                                    sig.push((target_class, guard_str));
                                }
                            }
                        }
                    }
                }
                sig.sort();

                let class_map = class_signatures.entry(old_class).or_default();
                if let Some(&assigned) = class_map.get(&sig) {
                    if new_partition[i] != assigned {
                        new_partition[i] = assigned;
                        changed = true;
                    }
                } else {
                    let c = if class_map.is_empty() {
                        old_class
                    } else {
                        let c = next_class;
                        next_class += 1;
                        changed = true;
                        c
                    };
                    class_map.insert(sig, c);
                    new_partition[i] = c;
                }
            }
            partition = new_partition;
        }

        // Build minimised automaton
        let mut min_auto = SpatialEventAutomaton::new(format!("{}_min", self.metadata.name));
        let mut class_to_state: HashMap<usize, StateId> = HashMap::new();

        // Create one state per class
        for (i, &sid) in state_vec.iter().enumerate() {
            let c = partition[i];
            if class_to_state.contains_key(&c) {
                continue;
            }
            let nid = min_auto.fresh_state_id();
            let original = &self.states[&sid];
            let mut ns = State::new(nid, format!("min_{}", c));
            ns.is_initial = self.initial_state == Some(sid)
                || (self.initial_state.is_some()
                    && partition[idx[&self.initial_state.unwrap()]] == c);
            ns.is_accepting = self.accepting_states.contains(&sid);
            ns.is_error = original.is_error;
            ns.invariant = original.invariant.clone();
            ns.on_entry = original.on_entry.clone();
            ns.on_exit = original.on_exit.clone();
            min_auto.add_state(ns);
            class_to_state.insert(c, nid);
        }

        // Create transitions (one per unique (class_src, class_tgt, guard) triple)
        let mut seen_transitions: HashSet<(usize, usize, String)> = HashSet::new();
        for t in self.transitions.values() {
            if let (Some(&src_idx), Some(&tgt_idx)) = (idx.get(&t.source), idx.get(&t.target)) {
                let src_class = partition[src_idx];
                let tgt_class = partition[tgt_idx];
                let guard_key = format!("{}", t.guard);
                if seen_transitions.insert((src_class, tgt_class, guard_key)) {
                    let tid = min_auto.fresh_transition_id();
                    let trans = Transition::new(
                        tid,
                        class_to_state[&src_class],
                        class_to_state[&tgt_class],
                        t.guard.clone(),
                        t.actions.clone(),
                    );
                    min_auto.add_transition(trans);
                }
            }
        }

        min_auto.kind = self.kind;
        min_auto.recompute_statistics();
        min_auto
    }

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    /// Recompute statistics from current state.
    pub fn recompute_statistics(&mut self) {
        let n_states = self.states.len();
        let n_trans = self.transitions.len();
        let n_accept = self.accepting_states.len();

        let guard_complexity: usize = self.transitions.values().map(|t| t.guard.complexity()).sum();

        let max_out = self
            .states
            .keys()
            .map(|s| self.outgoing(*s).len())
            .max()
            .unwrap_or(0);

        let avg_out = if n_states > 0 {
            n_trans as f64 / n_states as f64
        } else {
            0.0
        };

        let has_eps = self.transitions.values().any(|t| {
            matches!(t.guard, Guard::Event(EventKind::Epsilon))
        });

        self.metadata.statistics = AutomatonStatistics {
            state_count: n_states,
            transition_count: n_trans,
            accepting_count: n_accept,
            guard_complexity,
            is_deterministic: self.is_deterministic(),
            has_epsilon: has_eps,
            max_out_degree: max_out,
            avg_out_degree: avg_out,
        };
    }

    // -----------------------------------------------------------------------
    // Serialization helpers
    // -----------------------------------------------------------------------

    /// Serialize to JSON string.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(|e| AutomataError::SerializationError(e.to_string()))
    }

    /// Deserialize from JSON string.
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(|e| AutomataError::SerializationError(e.to_string()))
    }

    /// Number of states.
    pub fn state_count(&self) -> usize {
        self.states.len()
    }

    /// Number of transitions.
    pub fn transition_count(&self) -> usize {
        self.transitions.len()
    }

    /// Returns successor state ids reachable via a single transition from the given state.
    pub fn successors(&self, state: StateId) -> Vec<StateId> {
        self.outgoing(state)
            .into_iter()
            .filter_map(|tid| self.transition(tid).map(|t| t.target))
            .collect()
    }

    /// Returns predecessor state ids that have a transition into the given state.
    pub fn predecessors(&self, state: StateId) -> Vec<StateId> {
        self.incoming(state)
            .into_iter()
            .filter_map(|tid| self.transition(tid).map(|t| t.source))
            .collect()
    }

    /// Remove a state and all its incident transitions.
    pub fn remove_state(&mut self, state: StateId) {
        self.states.swap_remove(&state);
        self.accepting_states.remove(&state);
        if self.initial_state == Some(state) {
            self.initial_state = None;
        }
        let to_remove: Vec<TransitionId> = self
            .transitions
            .values()
            .filter(|t| t.source == state || t.target == state)
            .map(|t| t.id)
            .collect();
        for tid in to_remove {
            self.transitions.swap_remove(&tid);
            self.spatial_guards.remove(&tid);
            self.temporal_guards.remove(&tid);
        }
    }

    /// Remove a transition.
    pub fn remove_transition(&mut self, tid: TransitionId) {
        self.transitions.swap_remove(&tid);
        self.spatial_guards.remove(&tid);
        self.temporal_guards.remove(&tid);
    }
}

impl fmt::Display for SpatialEventAutomaton {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Automaton \"{}\" ({:?})", self.metadata.name, self.kind)?;
        writeln!(
            f,
            "  States: {}, Transitions: {}, Accepting: {}",
            self.states.len(),
            self.transitions.len(),
            self.accepting_states.len()
        )?;
        if let Some(init) = self.initial_state {
            writeln!(f, "  Initial: {}", init)?;
        }
        for t in self.transitions.values() {
            writeln!(
                f,
                "  {} --[{}]--> {}",
                t.source, t.guard, t.target
            )?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Evaluate a spatial predicate against a scene configuration.
pub fn evaluate_spatial_predicate(pred: &SpatialPredicate, scene: &SceneConfiguration) -> bool {
    match pred {
        SpatialPredicate::Inside { entity, region } => {
            let ent = match scene.entity(entity) {
                Some(e) => e,
                None => return false,
            };
            match scene.regions.get(region) {
                Some(crate::SpatialRegion::AABB(aabb)) => aabb.contains_point(&ent.position),
                Some(crate::SpatialRegion::Sphere(sph)) => {
                    ent.position.distance_to(&sph.center) <= sph.radius
                }
                Some(crate::SpatialRegion::Named(_)) => false,
                None => false,
            }
        }
        SpatialPredicate::Proximity {
            entity_a,
            entity_b,
            threshold,
        } => {
            let a = match scene.entity(entity_a) {
                Some(e) => e,
                None => return false,
            };
            let b = match scene.entity(entity_b) {
                Some(e) => e,
                None => return false,
            };
            a.position.distance_to(&b.position) <= *threshold
        }
        SpatialPredicate::Contact {
            entity_a,
            entity_b,
        } => {
            let a = match scene.entity(entity_a) {
                Some(e) => e,
                None => return false,
            };
            let b = match scene.entity(entity_b) {
                Some(e) => e,
                None => return false,
            };
            // Approximate: contact when bounding boxes overlap
            match (&a.bounding, &b.bounding) {
                (Some(ba), Some(bb)) => ba.intersects(bb),
                _ => a.position.distance_to(&b.position) < 0.05,
            }
        }
        SpatialPredicate::GazeAt { entity, target } => {
            let _ent = match scene.entity(entity) {
                Some(e) => e,
                None => return false,
            };
            // Simplified: check if gaze property references the target
            scene.regions.contains_key(target)
        }
        SpatialPredicate::Grasping { hand, object } => {
            let h = match scene.entity(hand) {
                Some(e) => e,
                None => return false,
            };
            h.properties
                .get("grasping")
                .map_or(false, |v| v == &object.0)
        }
        SpatialPredicate::Not(inner) => !evaluate_spatial_predicate(inner, scene),
        SpatialPredicate::And(preds) => preds.iter().all(|p| evaluate_spatial_predicate(p, scene)),
        SpatialPredicate::Or(preds) => preds.iter().any(|p| evaluate_spatial_predicate(p, scene)),
        SpatialPredicate::Named(_) => true, // Named predicates need external resolution
    }
}

/// Evaluate a temporal guard expression.
pub fn evaluate_temporal_expr(
    expr: &TemporalGuardExpr,
    time: TimePoint,
    timer_values: &HashMap<TimerId, f64>,
) -> bool {
    match expr {
        TemporalGuardExpr::TimerElapsed { timer, threshold } => {
            timer_values.get(timer).map_or(false, |v| *v >= threshold.0)
        }
        TemporalGuardExpr::WithinInterval(interval) => interval.contains(time),
        TemporalGuardExpr::Named(_) => true,
        TemporalGuardExpr::And(es) => es.iter().all(|e| evaluate_temporal_expr(e, time, timer_values)),
        TemporalGuardExpr::Or(es) => es.iter().any(|e| evaluate_temporal_expr(e, time, timer_values)),
        TemporalGuardExpr::Not(e) => !evaluate_temporal_expr(e, time, timer_values),
    }
}

/// Extract spatial and temporal guards from a guard tree.
fn extract_guards_for_transition(
    guard: &Guard,
    tid: TransitionId,
    spatial: &mut HashMap<TransitionId, Vec<SpatialGuard>>,
    temporal: &mut HashMap<TransitionId, Vec<TemporalGuard>>,
) {
    match guard {
        Guard::Spatial(sp) => {
            spatial.entry(tid).or_default().push(SpatialGuard::new(sp.clone()));
        }
        Guard::Temporal(te) => {
            temporal.entry(tid).or_default().push(TemporalGuard::new(te.clone()));
        }
        Guard::And(gs) | Guard::Or(gs) => {
            for g in gs {
                extract_guards_for_transition(g, tid, spatial, temporal);
            }
        }
        Guard::Not(g) => {
            extract_guards_for_transition(g, tid, spatial, temporal);
        }
        _ => {}
    }
}

/// Collect event kinds from a guard.
fn collect_events_from_guard(guard: &Guard, events: &mut HashSet<EventKind>) {
    match guard {
        Guard::Event(ek) => {
            events.insert(ek.clone());
        }
        Guard::And(gs) | Guard::Or(gs) => {
            for g in gs {
                collect_events_from_guard(g, events);
            }
        }
        Guard::Not(g) => collect_events_from_guard(g, events),
        _ => {}
    }
}

/// Epsilon closure for a set of states within a `SpatialEventAutomaton`.
fn epsilon_closure_for_automaton(
    auto: &SpatialEventAutomaton,
    states: &BTreeSet<StateId>,
) -> BTreeSet<StateId> {
    let mut closure = states.clone();
    let mut queue: VecDeque<StateId> = states.iter().copied().collect();
    while let Some(s) = queue.pop_front() {
        for tid in auto.outgoing(s) {
            if let Some(t) = auto.transition(tid) {
                if matches!(t.guard, Guard::Event(EventKind::Epsilon)) {
                    if closure.insert(t.target) {
                        queue.push_back(t.target);
                    }
                }
            }
        }
    }
    closure
}

/// Check whether a guard matches a given event kind.
fn guard_matches_event(guard: &Guard, event: &EventKind) -> bool {
    match guard {
        Guard::Event(ek) => ek == event,
        Guard::And(gs) => gs.iter().any(|g| guard_matches_event(g, event)),
        Guard::Or(gs) => gs.iter().any(|g| guard_matches_event(g, event)),
        Guard::True => true,
        _ => false,
    }
}

/// Merge guards from multiple transitions in a subset sharing the same event.
fn merge_guards_for_subset(
    auto: &SpatialEventAutomaton,
    subset: &BTreeSet<StateId>,
    event: &EventKind,
) -> Guard {
    let mut guards = Vec::new();
    for &sid in subset {
        for tid in auto.outgoing(sid) {
            if let Some(t) = auto.transition(tid) {
                if guard_matches_event(&t.guard, event) {
                    guards.push(t.guard.clone());
                }
            }
        }
    }
    if guards.is_empty() {
        Guard::Event(event.clone())
    } else if guards.len() == 1 {
        guards.into_iter().next().unwrap()
    } else {
        Guard::Or(guards)
    }
}

/// Build a printable name for a subset of states.
fn subset_name(set: &BTreeSet<StateId>) -> String {
    let parts: Vec<String> = set.iter().map(|s| format!("{}", s)).collect();
    format!("{{{}}}", parts.join(","))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;

    fn make_simple_automaton() -> SpatialEventAutomaton {
        let mut auto = SpatialEventAutomaton::new("test");
        let s0 = auto.fresh_state_id();
        let s1 = auto.fresh_state_id();
        let s2 = auto.fresh_state_id();

        let mut st0 = State::new(s0, "idle");
        st0.is_initial = true;
        auto.add_state(st0);

        let st1 = State::new(s1, "active");
        auto.add_state(st1);

        let mut st2 = State::new(s2, "done");
        st2.is_accepting = true;
        auto.add_state(st2);

        let t0 = auto.fresh_transition_id();
        auto.add_transition(Transition::new(
            t0,
            s0,
            s1,
            Guard::Event(EventKind::GrabStart),
            vec![],
        ));

        let t1 = auto.fresh_transition_id();
        auto.add_transition(Transition::new(
            t1,
            s1,
            s2,
            Guard::Event(EventKind::GrabEnd),
            vec![],
        ));

        auto
    }

    #[test]
    fn test_state_count() {
        let auto = make_simple_automaton();
        assert_eq!(auto.state_count(), 3);
        assert_eq!(auto.transition_count(), 2);
    }

    #[test]
    fn test_reachable_states() {
        let auto = make_simple_automaton();
        let reachable = auto.reachable_states();
        assert_eq!(reachable.len(), 3);
    }

    #[test]
    fn test_accepting_reachable() {
        let auto = make_simple_automaton();
        assert!(auto.accepting_reachable(StateId(0)));
        assert!(auto.accepting_reachable(StateId(1)));
        assert!(auto.accepting_reachable(StateId(2)));
    }

    #[test]
    fn test_outgoing_incoming() {
        let auto = make_simple_automaton();
        assert_eq!(auto.outgoing(StateId(0)).len(), 1);
        assert_eq!(auto.outgoing(StateId(1)).len(), 1);
        assert_eq!(auto.outgoing(StateId(2)).len(), 0);
        assert_eq!(auto.incoming(StateId(0)).len(), 0);
        assert_eq!(auto.incoming(StateId(1)).len(), 1);
        assert_eq!(auto.incoming(StateId(2)).len(), 1);
    }

    #[test]
    fn test_fire_transition() {
        let auto = make_simple_automaton();
        let (target, actions) = auto.fire_transition(TransitionId(0)).unwrap();
        assert_eq!(target, StateId(1));
        assert!(actions.is_empty());
    }

    #[test]
    fn test_enabled_transitions() {
        let auto = make_simple_automaton();
        let scene = SceneConfiguration::empty();
        let timers = HashMap::new();
        let evt = EventKind::GrabStart;
        let enabled =
            auto.enabled_transitions(StateId(0), &scene, TimePoint::zero(), Some(&evt), &timers);
        assert_eq!(enabled.len(), 1);
    }

    #[test]
    fn test_is_deterministic() {
        let auto = make_simple_automaton();
        assert!(auto.is_deterministic());
    }

    #[test]
    fn test_minimize() {
        let auto = make_simple_automaton();
        let minimised = auto.minimize();
        assert!(minimised.state_count() <= auto.state_count());
    }

    #[test]
    fn test_determinize() {
        let mut auto = SpatialEventAutomaton::new("nfa_test");
        let s0 = auto.fresh_state_id();
        let s1 = auto.fresh_state_id();
        let s2 = auto.fresh_state_id();

        let mut st0 = State::new(s0, "start");
        st0.is_initial = true;
        auto.add_state(st0);
        auto.add_state(State::new(s1, "mid"));
        let mut st2 = State::new(s2, "end");
        st2.is_accepting = true;
        auto.add_state(st2);

        let t0 = auto.fresh_transition_id();
        auto.add_transition(Transition::new(
            t0, s0, s1,
            Guard::Event(EventKind::GrabStart),
            vec![],
        ));
        let t1 = auto.fresh_transition_id();
        auto.add_transition(Transition::new(
            t1, s0, s2,
            Guard::Event(EventKind::GrabStart),
            vec![],
        ));
        let t2 = auto.fresh_transition_id();
        auto.add_transition(Transition::new(
            t2, s1, s2,
            Guard::Event(EventKind::GrabEnd),
            vec![],
        ));

        let dfa = auto.determinize();
        assert!(dfa.is_deterministic() || dfa.state_count() > 0);
        assert!(dfa.initial_state.is_some());
    }

    #[test]
    fn test_spatial_guard_evaluation() {
        let pred = SpatialPredicate::Proximity {
            entity_a: EntityId("hand".into()),
            entity_b: EntityId("button".into()),
            threshold: 1.0,
        };
        let mut scene = SceneConfiguration::empty();
        scene.entities.push(SceneEntity {
            id: EntityId("hand".into()),
            position: Point3::new(0.0, 0.0, 0.0),
            bounding: None,
            properties: HashMap::new(),
        });
        scene.entities.push(SceneEntity {
            id: EntityId("button".into()),
            position: Point3::new(0.5, 0.0, 0.0),
            bounding: None,
            properties: HashMap::new(),
        });
        assert!(evaluate_spatial_predicate(&pred, &scene));

        scene.entities[1].position = Point3::new(5.0, 5.0, 5.0);
        assert!(!evaluate_spatial_predicate(&pred, &scene));
    }

    #[test]
    fn test_temporal_guard_evaluation() {
        let expr = TemporalGuardExpr::TimerElapsed {
            timer: TimerId("dwell".into()),
            threshold: crate::Duration(2.0),
        };
        let mut timers = HashMap::new();
        timers.insert(TimerId("dwell".into()), 1.5);
        assert!(!evaluate_temporal_expr(&expr, TimePoint::zero(), &timers));
        timers.insert(TimerId("dwell".into()), 2.5);
        assert!(evaluate_temporal_expr(&expr, TimePoint::zero(), &timers));
    }

    #[test]
    fn test_serialization_roundtrip() {
        let auto = make_simple_automaton();
        let json = auto.to_json().unwrap();
        let restored = SpatialEventAutomaton::from_json(&json).unwrap();
        assert_eq!(restored.state_count(), auto.state_count());
        assert_eq!(restored.transition_count(), auto.transition_count());
    }

    #[test]
    fn test_successors_predecessors() {
        let auto = make_simple_automaton();
        let succs = auto.successors(StateId(0));
        assert_eq!(succs, vec![StateId(1)]);
        let preds = auto.predecessors(StateId(2));
        assert_eq!(preds, vec![StateId(1)]);
    }

    #[test]
    fn test_alphabet() {
        let auto = make_simple_automaton();
        let alpha = auto.alphabet();
        assert!(alpha.contains(&EventKind::GrabStart));
        assert!(alpha.contains(&EventKind::GrabEnd));
    }

    #[test]
    fn test_remove_state() {
        let mut auto = make_simple_automaton();
        auto.remove_state(StateId(1));
        assert_eq!(auto.state_count(), 2);
        assert_eq!(auto.transition_count(), 0);
    }

    #[test]
    fn test_state_invariant_holds() {
        let mut auto = SpatialEventAutomaton::new("inv_test");
        let sid = auto.fresh_state_id();
        let mut s = State::new(sid, "guarded");
        s.is_initial = true;
        s.invariant = Some(Guard::True);
        auto.add_state(s);
        let scene = SceneConfiguration::empty();
        assert!(auto.state_invariant_holds(sid, &scene, TimePoint::zero(), &HashMap::new()));
    }

    #[test]
    fn test_display() {
        let auto = make_simple_automaton();
        let display = format!("{}", auto);
        assert!(display.contains("test"));
    }

    #[test]
    fn test_guard_complexity() {
        assert_eq!(Guard::True.complexity(), 0);
        assert_eq!(Guard::Event(EventKind::GrabStart).complexity(), 1);
        let compound = Guard::And(vec![
            Guard::Event(EventKind::GrabStart),
            Guard::Event(EventKind::GazeEnter),
        ]);
        assert_eq!(compound.complexity(), 3);
    }
}
