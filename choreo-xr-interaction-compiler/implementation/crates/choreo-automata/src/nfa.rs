//! NFA-specific operations: epsilon transitions, epsilon closure,
//! Thompson construction, subset construction (NFA→DFA), simulation,
//! and token-passing execution.

use crate::automaton::{
    AutomatonKind, SpatialEventAutomaton, State, Transition,
};
use crate::{
    Action, EventKind, Guard, StateId, TransitionId,
};
use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};
use std::fmt;

// ---------------------------------------------------------------------------
// NFA
// ---------------------------------------------------------------------------

/// A non-deterministic finite automaton with explicit epsilon transitions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NFA {
    pub states: IndexMap<StateId, NFAState>,
    pub transitions: IndexMap<TransitionId, NFATransition>,
    pub epsilon_transitions: Vec<(StateId, StateId)>,
    pub initial_state: Option<StateId>,
    pub accepting_states: HashSet<StateId>,
    pub name: String,
    next_state_id: u32,
    next_transition_id: u32,
}

/// State in the NFA.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NFAState {
    pub id: StateId,
    pub name: String,
    pub is_initial: bool,
    pub is_accepting: bool,
    pub on_entry: Vec<Action>,
    pub on_exit: Vec<Action>,
}

impl NFAState {
    pub fn new(id: StateId, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            is_initial: false,
            is_accepting: false,
            on_entry: Vec::new(),
            on_exit: Vec::new(),
        }
    }
}

/// Transition in the NFA (non-epsilon).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NFATransition {
    pub id: TransitionId,
    pub source: StateId,
    pub target: StateId,
    pub guard: Guard,
    pub actions: Vec<Action>,
}

impl NFA {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            states: IndexMap::new(),
            transitions: IndexMap::new(),
            epsilon_transitions: Vec::new(),
            initial_state: None,
            accepting_states: HashSet::new(),
            name: name.into(),
            next_state_id: 0,
            next_transition_id: 0,
        }
    }

    /// Allocate a fresh state id.
    pub fn fresh_state_id(&mut self) -> StateId {
        let id = StateId(self.next_state_id);
        self.next_state_id += 1;
        id
    }

    /// Allocate a fresh transition id.
    pub fn fresh_transition_id(&mut self) -> TransitionId {
        let id = TransitionId(self.next_transition_id);
        self.next_transition_id += 1;
        id
    }

    /// Add a state.
    pub fn add_state(&mut self, state: NFAState) -> StateId {
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

    /// Add a non-epsilon transition.
    pub fn add_transition(&mut self, source: StateId, target: StateId, guard: Guard, actions: Vec<Action>) -> TransitionId {
        let id = self.fresh_transition_id();
        self.transitions.insert(
            id,
            NFATransition {
                id,
                source,
                target,
                guard,
                actions,
            },
        );
        id
    }

    /// Add an epsilon transition.
    pub fn add_epsilon_transition(&mut self, source: StateId, target: StateId) {
        self.epsilon_transitions.push((source, target));
    }

    /// Compute the epsilon closure of a set of states.
    pub fn epsilon_closure(&self, states: &HashSet<StateId>) -> HashSet<StateId> {
        let mut closure = states.clone();
        let mut queue: VecDeque<StateId> = states.iter().copied().collect();
        while let Some(s) = queue.pop_front() {
            for &(src, tgt) in &self.epsilon_transitions {
                if src == s && closure.insert(tgt) {
                    queue.push_back(tgt);
                }
            }
        }
        closure
    }

    /// Epsilon closure with BTreeSet for deterministic ordering.
    fn epsilon_closure_ordered(&self, states: &BTreeSet<StateId>) -> BTreeSet<StateId> {
        let mut closure = states.clone();
        let mut queue: VecDeque<StateId> = states.iter().copied().collect();
        while let Some(s) = queue.pop_front() {
            for &(src, tgt) in &self.epsilon_transitions {
                if src == s && closure.insert(tgt) {
                    queue.push_back(tgt);
                }
            }
        }
        closure
    }

    /// Compute the set of states reachable from `states` on a given event.
    pub fn move_on_event(&self, states: &HashSet<StateId>, event: &EventKind) -> HashSet<StateId> {
        let mut result = HashSet::new();
        for &s in states {
            for t in self.transitions.values() {
                if t.source == s && guard_matches_event(&t.guard, event) {
                    result.insert(t.target);
                }
            }
        }
        result
    }

    /// Move on event with ordered sets.
    fn move_on_event_ordered(&self, states: &BTreeSet<StateId>, event: &EventKind) -> BTreeSet<StateId> {
        let mut result = BTreeSet::new();
        for &s in states {
            for t in self.transitions.values() {
                if t.source == s && guard_matches_event(&t.guard, event) {
                    result.insert(t.target);
                }
            }
        }
        result
    }

    /// Outgoing non-epsilon transitions from a state.
    pub fn outgoing(&self, state: StateId) -> Vec<TransitionId> {
        self.transitions
            .values()
            .filter(|t| t.source == state)
            .map(|t| t.id)
            .collect()
    }

    /// Collect the alphabet (set of event kinds used in transitions).
    pub fn alphabet(&self) -> HashSet<EventKind> {
        let mut events = HashSet::new();
        for t in self.transitions.values() {
            collect_events(&t.guard, &mut events);
        }
        events
    }

    /// Check whether the NFA accepts a trace of events.
    pub fn accepts(&self, trace: &[EventKind]) -> bool {
        let init = match self.initial_state {
            Some(s) => {
                let mut set = HashSet::new();
                set.insert(s);
                self.epsilon_closure(&set)
            }
            None => return false,
        };

        let mut current = init;
        for event in trace {
            let moved = self.move_on_event(&current, event);
            current = self.epsilon_closure(&moved);
            if current.is_empty() {
                return false;
            }
        }
        current.iter().any(|s| self.accepting_states.contains(s))
    }

    /// Number of states.
    pub fn state_count(&self) -> usize {
        self.states.len()
    }

    /// Number of non-epsilon transitions.
    pub fn transition_count(&self) -> usize {
        self.transitions.len()
    }

    /// Number of epsilon transitions.
    pub fn epsilon_count(&self) -> usize {
        self.epsilon_transitions.len()
    }

    // -----------------------------------------------------------------------
    // NFA → DFA subset construction
    // -----------------------------------------------------------------------

    /// Convert this NFA to a DFA via subset construction.
    pub fn to_dfa(&self) -> DFA {
        let events = self.alphabet();
        let init = match self.initial_state {
            Some(s) => {
                let mut set = BTreeSet::new();
                set.insert(s);
                self.epsilon_closure_ordered(&set)
            }
            None => BTreeSet::new(),
        };

        let mut dfa = DFA::new(format!("{}_dfa", self.name));
        let mut state_map: HashMap<BTreeSet<StateId>, StateId> = HashMap::new();
        let mut worklist: VecDeque<BTreeSet<StateId>> = VecDeque::new();

        let init_id = dfa.fresh_state_id();
        let is_acc = init.iter().any(|s| self.accepting_states.contains(s));
        let mut init_state = DFAState::new(init_id, subset_name(&init));
        init_state.is_initial = true;
        init_state.is_accepting = is_acc;
        init_state.nfa_states = init.clone();
        dfa.add_state(init_state);
        state_map.insert(init.clone(), init_id);
        worklist.push_back(init);

        while let Some(current_set) = worklist.pop_front() {
            let current_id = state_map[&current_set];
            for event in &events {
                if *event == EventKind::Epsilon {
                    continue;
                }
                let moved = self.move_on_event_ordered(&current_set, event);
                let next_set = self.epsilon_closure_ordered(&moved);
                if next_set.is_empty() {
                    continue;
                }
                let next_id = if let Some(&existing) = state_map.get(&next_set) {
                    existing
                } else {
                    let nid = dfa.fresh_state_id();
                    let is_acc = next_set.iter().any(|s| self.accepting_states.contains(s));
                    let mut ns = DFAState::new(nid, subset_name(&next_set));
                    ns.is_accepting = is_acc;
                    ns.nfa_states = next_set.clone();
                    dfa.add_state(ns);
                    state_map.insert(next_set.clone(), nid);
                    worklist.push_back(next_set);
                    nid
                };
                dfa.add_transition(current_id, next_id, Guard::Event(event.clone()));
            }
        }

        dfa
    }

    // -----------------------------------------------------------------------
    // NFA → SpatialEventAutomaton
    // -----------------------------------------------------------------------

    /// Convert to a `SpatialEventAutomaton`.
    pub fn to_spatial_automaton(&self) -> SpatialEventAutomaton {
        let mut auto = SpatialEventAutomaton::new(&self.name);
        auto.kind = AutomatonKind::NFA;

        for nfa_state in self.states.values() {
            let mut s = State::new(nfa_state.id, &nfa_state.name);
            s.is_initial = nfa_state.is_initial;
            s.is_accepting = nfa_state.is_accepting;
            s.on_entry = nfa_state.on_entry.clone();
            s.on_exit = nfa_state.on_exit.clone();
            auto.add_state(s);
        }

        // Non-epsilon transitions
        for t in self.transitions.values() {
            let tid = auto.fresh_transition_id();
            auto.add_transition(Transition::new(
                tid, t.source, t.target, t.guard.clone(), t.actions.clone(),
            ));
        }

        // Epsilon transitions
        for &(src, tgt) in &self.epsilon_transitions {
            let tid = auto.fresh_transition_id();
            auto.add_transition(Transition::new(
                tid, src, tgt, Guard::Event(EventKind::Epsilon), vec![],
            ));
        }

        auto.recompute_statistics();
        auto
    }

    // -----------------------------------------------------------------------
    // Simulation
    // -----------------------------------------------------------------------

    /// Simulate the NFA on a trace of events, returning the full simulation
    /// result including all active state sets at each step.
    pub fn simulate(&self, trace: &[EventKind]) -> SimulationResult {
        let init = match self.initial_state {
            Some(s) => {
                let mut set = HashSet::new();
                set.insert(s);
                self.epsilon_closure(&set)
            }
            None => HashSet::new(),
        };

        let mut steps = Vec::new();
        let mut current = init.clone();

        steps.push(SimulationStep {
            event: None,
            active_states: current.clone(),
            is_accepting: current.iter().any(|s| self.accepting_states.contains(s)),
        });

        for event in trace {
            let moved = self.move_on_event(&current, event);
            current = self.epsilon_closure(&moved);
            steps.push(SimulationStep {
                event: Some(event.clone()),
                active_states: current.clone(),
                is_accepting: current.iter().any(|s| self.accepting_states.contains(s)),
            });
        }

        let accepted = current.iter().any(|s| self.accepting_states.contains(s));
        SimulationResult {
            steps,
            accepted,
            final_states: current,
        }
    }
}

impl fmt::Display for NFA {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "NFA \"{}\"", self.name)?;
        writeln!(
            f,
            "  States: {}, Transitions: {}, Epsilon: {}",
            self.states.len(),
            self.transitions.len(),
            self.epsilon_transitions.len(),
        )?;
        if let Some(init) = self.initial_state {
            writeln!(f, "  Initial: {}", init)?;
        }
        writeln!(
            f,
            "  Accepting: {:?}",
            self.accepting_states.iter().collect::<Vec<_>>()
        )?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// DFA
// ---------------------------------------------------------------------------

/// A deterministic finite automaton (result of subset construction).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DFA {
    pub states: IndexMap<StateId, DFAState>,
    pub transitions: Vec<DFATransition>,
    pub initial_state: Option<StateId>,
    pub accepting_states: HashSet<StateId>,
    pub name: String,
    next_state_id: u32,
}

/// State in the DFA.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DFAState {
    pub id: StateId,
    pub name: String,
    pub is_initial: bool,
    pub is_accepting: bool,
    /// The set of NFA states this DFA state corresponds to.
    pub nfa_states: BTreeSet<StateId>,
}

impl DFAState {
    pub fn new(id: StateId, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            is_initial: false,
            is_accepting: false,
            nfa_states: BTreeSet::new(),
        }
    }
}

/// Transition in the DFA.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DFATransition {
    pub source: StateId,
    pub target: StateId,
    pub guard: Guard,
}

impl DFA {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            states: IndexMap::new(),
            transitions: Vec::new(),
            initial_state: None,
            accepting_states: HashSet::new(),
            name: name.into(),
            next_state_id: 0,
        }
    }

    pub fn fresh_state_id(&mut self) -> StateId {
        let id = StateId(self.next_state_id);
        self.next_state_id += 1;
        id
    }

    pub fn add_state(&mut self, state: DFAState) -> StateId {
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

    pub fn add_transition(&mut self, source: StateId, target: StateId, guard: Guard) {
        self.transitions.push(DFATransition {
            source,
            target,
            guard,
        });
    }

    pub fn state_count(&self) -> usize {
        self.states.len()
    }

    pub fn transition_count(&self) -> usize {
        self.transitions.len()
    }

    /// Run the DFA on a trace.
    pub fn accepts(&self, trace: &[EventKind]) -> bool {
        let mut current = match self.initial_state {
            Some(s) => s,
            None => return false,
        };
        for event in trace {
            let mut found = false;
            for t in &self.transitions {
                if t.source == current && guard_matches_event(&t.guard, event) {
                    current = t.target;
                    found = true;
                    break;
                }
            }
            if !found {
                return false;
            }
        }
        self.accepting_states.contains(&current)
    }

    /// Convert to a `SpatialEventAutomaton`.
    pub fn to_spatial_automaton(&self) -> SpatialEventAutomaton {
        let mut auto = SpatialEventAutomaton::new(&self.name);
        auto.kind = AutomatonKind::DFA;

        for dfa_state in self.states.values() {
            let mut s = State::new(dfa_state.id, &dfa_state.name);
            s.is_initial = dfa_state.is_initial;
            s.is_accepting = dfa_state.is_accepting;
            auto.add_state(s);
        }

        for t in &self.transitions {
            let tid = auto.fresh_transition_id();
            auto.add_transition(Transition::new(
                tid, t.source, t.target, t.guard.clone(), vec![],
            ));
        }

        auto.recompute_statistics();
        auto
    }
}

// ---------------------------------------------------------------------------
// Simulation result
// ---------------------------------------------------------------------------

/// Result of simulating an NFA on an event trace.
#[derive(Debug, Clone)]
pub struct SimulationResult {
    pub steps: Vec<SimulationStep>,
    pub accepted: bool,
    pub final_states: HashSet<StateId>,
}

/// A single step in the simulation.
#[derive(Debug, Clone)]
pub struct SimulationStep {
    pub event: Option<EventKind>,
    pub active_states: HashSet<StateId>,
    pub is_accepting: bool,
}

// ---------------------------------------------------------------------------
// Token-passing NFA execution
// ---------------------------------------------------------------------------

/// A token in the token-passing execution model.
#[derive(Debug, Clone)]
pub struct Token {
    pub id: u64,
    pub state: StateId,
    pub data: HashMap<String, String>,
    pub created_at: usize,
}

/// Token-passing NFA executor for runtime use.
#[derive(Debug)]
pub struct TokenPassingExecutor {
    nfa: NFA,
    tokens: Vec<Token>,
    next_token_id: u64,
    step_count: usize,
    max_tokens: usize,
}

impl TokenPassingExecutor {
    pub fn new(nfa: NFA) -> Self {
        Self {
            nfa,
            tokens: Vec::new(),
            next_token_id: 0,
            step_count: 0,
            max_tokens: 10_000,
        }
    }

    pub fn with_max_tokens(mut self, max: usize) -> Self {
        self.max_tokens = max;
        self
    }

    /// Initialise: place a token on the initial state and its epsilon closure.
    pub fn initialize(&mut self) {
        self.tokens.clear();
        self.step_count = 0;
        if let Some(init) = self.nfa.initial_state {
            let mut init_set = HashSet::new();
            init_set.insert(init);
            let closure = self.nfa.epsilon_closure(&init_set);
            for s in closure {
                self.spawn_token(s);
            }
        }
    }

    /// Process one event: advance all tokens, spawn new tokens via epsilon
    /// closure, remove dead tokens.
    pub fn process_event(&mut self, event: &EventKind) -> TokenStepResult {
        self.step_count += 1;
        let mut new_tokens = Vec::new();

        for token in &self.tokens {
            // Find matching transitions
            for t in self.nfa.transitions.values() {
                if t.source == token.state && guard_matches_event(&t.guard, event) {
                    // Spawn new token at target
                    let mut data = token.data.clone();
                    data.insert("last_event".into(), format!("{}", event));
                    new_tokens.push((t.target, data));
                }
            }
        }

        // Replace tokens
        self.tokens.clear();
        let mut placed_states = HashSet::new();
        for (target, data) in new_tokens {
            // Epsilon closure from target
            let mut tgt_set = HashSet::new();
            tgt_set.insert(target);
            let closure = self.nfa.epsilon_closure(&tgt_set);
            for s in closure {
                if placed_states.insert(s) && self.tokens.len() < self.max_tokens {
                    let mut tok = self.make_token(s);
                    tok.data = data.clone();
                    self.tokens.push(tok);
                }
            }
        }

        let accepting = self
            .tokens
            .iter()
            .any(|t| self.nfa.accepting_states.contains(&t.state));

        TokenStepResult {
            active_tokens: self.tokens.len(),
            accepting,
            active_states: self.tokens.iter().map(|t| t.state).collect(),
            step: self.step_count,
        }
    }

    /// Check whether any token is in an accepting state.
    pub fn is_accepting(&self) -> bool {
        self.tokens
            .iter()
            .any(|t| self.nfa.accepting_states.contains(&t.state))
    }

    /// Current active states.
    pub fn active_states(&self) -> HashSet<StateId> {
        self.tokens.iter().map(|t| t.state).collect()
    }

    /// Number of active tokens.
    pub fn token_count(&self) -> usize {
        self.tokens.len()
    }

    fn spawn_token(&mut self, state: StateId) {
        let token = self.make_token(state);
        self.tokens.push(token);
    }

    fn make_token(&mut self, state: StateId) -> Token {
        let id = self.next_token_id;
        self.next_token_id += 1;
        Token {
            id,
            state,
            data: HashMap::new(),
            created_at: self.step_count,
        }
    }
}

/// Result of a single token-passing step.
#[derive(Debug, Clone)]
pub struct TokenStepResult {
    pub active_tokens: usize,
    pub accepting: bool,
    pub active_states: HashSet<StateId>,
    pub step: usize,
}

// ---------------------------------------------------------------------------
// Thompson construction
// ---------------------------------------------------------------------------

/// Thompson construction: concatenation.
pub fn thompson_concat(a: &NFA, b: &NFA) -> NFA {
    let mut result = NFA::new(format!("{}·{}", a.name, b.name));
    let (a_map, _) = copy_nfa_into(&mut result, a);
    let (b_map, _) = copy_nfa_into(&mut result, b);

    // Set initial of result to initial of a
    if let Some(a_init) = a.initial_state {
        let mapped_init = a_map[&a_init];
        result.initial_state = Some(mapped_init);
        if let Some(s) = result.states.get_mut(&mapped_init) {
            s.is_initial = true;
        }
    }

    // Set accepting of result to accepting of b
    result.accepting_states.clear();
    for &acc in &b.accepting_states {
        let mapped = b_map[&acc];
        result.accepting_states.insert(mapped);
        if let Some(s) = result.states.get_mut(&mapped) {
            s.is_accepting = true;
        }
    }

    // Epsilon from each accepting state of a to initial state of b
    for &a_acc in &a.accepting_states {
        if let Some(b_init) = b.initial_state {
            result.add_epsilon_transition(a_map[&a_acc], b_map[&b_init]);
        }
    }

    // Remove accepting flag from a's accepting states in result
    for &a_acc in &a.accepting_states {
        let mapped = a_map[&a_acc];
        if let Some(s) = result.states.get_mut(&mapped) {
            s.is_accepting = false;
        }
    }

    result
}

/// Thompson construction: union (alternation).
pub fn thompson_union(a: &NFA, b: &NFA) -> NFA {
    let mut result = NFA::new(format!("{}|{}", a.name, b.name));

    let new_init = result.fresh_state_id();
    let mut init_state = NFAState::new(new_init, "union_init");
    init_state.is_initial = true;
    result.add_state(init_state);

    let new_accept = result.fresh_state_id();
    let mut accept_state = NFAState::new(new_accept, "union_accept");
    accept_state.is_accepting = true;
    result.add_state(accept_state);

    let (a_map, _) = copy_nfa_into(&mut result, a);
    let (b_map, _) = copy_nfa_into(&mut result, b);

    // Epsilon from new_init to both initial states
    if let Some(a_init) = a.initial_state {
        result.add_epsilon_transition(new_init, a_map[&a_init]);
    }
    if let Some(b_init) = b.initial_state {
        result.add_epsilon_transition(new_init, b_map[&b_init]);
    }

    // Epsilon from both accepting states to new_accept
    for &a_acc in &a.accepting_states {
        let mapped = a_map[&a_acc];
        result.add_epsilon_transition(mapped, new_accept);
        if let Some(s) = result.states.get_mut(&mapped) {
            s.is_accepting = false;
        }
    }
    for &b_acc in &b.accepting_states {
        let mapped = b_map[&b_acc];
        result.add_epsilon_transition(mapped, new_accept);
        if let Some(s) = result.states.get_mut(&mapped) {
            s.is_accepting = false;
        }
    }

    result
}

/// Thompson construction: Kleene star.
pub fn thompson_star(a: &NFA) -> NFA {
    let mut result = NFA::new(format!("({})*", a.name));

    let new_init = result.fresh_state_id();
    let mut init_state = NFAState::new(new_init, "star_init");
    init_state.is_initial = true;
    init_state.is_accepting = true; // Star accepts empty string
    result.add_state(init_state);

    let new_accept = result.fresh_state_id();
    let mut accept_state = NFAState::new(new_accept, "star_accept");
    accept_state.is_accepting = true;
    result.add_state(accept_state);

    let (a_map, _) = copy_nfa_into(&mut result, a);

    // Epsilon: new_init → a_init
    if let Some(a_init) = a.initial_state {
        result.add_epsilon_transition(new_init, a_map[&a_init]);
    }

    // Epsilon: new_init → new_accept (empty string)
    result.add_epsilon_transition(new_init, new_accept);

    // Epsilon: a_accept → new_accept and a_accept → a_init (loop)
    for &a_acc in &a.accepting_states {
        let mapped = a_map[&a_acc];
        result.add_epsilon_transition(mapped, new_accept);
        if let Some(a_init) = a.initial_state {
            result.add_epsilon_transition(mapped, a_map[&a_init]);
        }
        if let Some(s) = result.states.get_mut(&mapped) {
            s.is_accepting = false;
        }
    }

    result
}

/// Thompson construction: one or more (Kleene plus).
pub fn thompson_plus(a: &NFA) -> NFA {
    // a+ = a · a*
    let star = thompson_star(a);
    thompson_concat(a, &star)
}

/// Thompson construction: optional (zero or one).
pub fn thompson_optional(a: &NFA) -> NFA {
    // a? = a | ε
    let mut eps = NFA::new("ε");
    let s = eps.fresh_state_id();
    let mut state = NFAState::new(s, "eps");
    state.is_initial = true;
    state.is_accepting = true;
    eps.add_state(state);
    thompson_union(a, &eps)
}

// ---------------------------------------------------------------------------
// Pattern → NFA conversion
// ---------------------------------------------------------------------------

/// Interaction pattern that can be converted to an NFA.
#[derive(Debug, Clone)]
pub enum InteractionPattern {
    /// Single event.
    Event(EventKind),
    /// Sequence of patterns.
    Sequence(Vec<InteractionPattern>),
    /// Choice between patterns.
    Choice(Vec<InteractionPattern>),
    /// Repetition (Kleene star).
    Repeat(Box<InteractionPattern>),
    /// One or more repetition.
    RepeatPlus(Box<InteractionPattern>),
    /// Optional.
    Optional(Box<InteractionPattern>),
}

/// Convert an interaction pattern to an NFA.
pub fn from_pattern(pattern: &InteractionPattern) -> NFA {
    match pattern {
        InteractionPattern::Event(ek) => {
            let mut nfa = NFA::new(format!("{}", ek));
            let s0 = nfa.fresh_state_id();
            let s1 = nfa.fresh_state_id();
            let mut st0 = NFAState::new(s0, "start");
            st0.is_initial = true;
            nfa.add_state(st0);
            let mut st1 = NFAState::new(s1, "end");
            st1.is_accepting = true;
            nfa.add_state(st1);
            nfa.add_transition(s0, s1, Guard::Event(ek.clone()), vec![]);
            nfa
        }
        InteractionPattern::Sequence(pats) => {
            if pats.is_empty() {
                let mut nfa = NFA::new("empty_seq");
                let s = nfa.fresh_state_id();
                let mut st = NFAState::new(s, "only");
                st.is_initial = true;
                st.is_accepting = true;
                nfa.add_state(st);
                return nfa;
            }
            let mut result = from_pattern(&pats[0]);
            for p in &pats[1..] {
                let next = from_pattern(p);
                result = thompson_concat(&result, &next);
            }
            result
        }
        InteractionPattern::Choice(pats) => {
            if pats.is_empty() {
                let mut nfa = NFA::new("empty_choice");
                let s = nfa.fresh_state_id();
                let mut st = NFAState::new(s, "only");
                st.is_initial = true;
                nfa.add_state(st);
                return nfa;
            }
            let mut result = from_pattern(&pats[0]);
            for p in &pats[1..] {
                let next = from_pattern(p);
                result = thompson_union(&result, &next);
            }
            result
        }
        InteractionPattern::Repeat(inner) => {
            let nfa = from_pattern(inner);
            thompson_star(&nfa)
        }
        InteractionPattern::RepeatPlus(inner) => {
            let nfa = from_pattern(inner);
            thompson_plus(&nfa)
        }
        InteractionPattern::Optional(inner) => {
            let nfa = from_pattern(inner);
            thompson_optional(&nfa)
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Copy all states and transitions of `source` into `target`, remapping ids.
/// Returns the state id mapping and transition id mapping.
fn copy_nfa_into(
    target: &mut NFA,
    source: &NFA,
) -> (HashMap<StateId, StateId>, HashMap<TransitionId, TransitionId>) {
    let mut s_map = HashMap::new();
    let mut t_map = HashMap::new();

    for (old_id, old_state) in &source.states {
        let new_id = target.fresh_state_id();
        let mut new_state = NFAState::new(new_id, &old_state.name);
        // Don't propagate initial/accepting flags; caller decides
        new_state.on_entry = old_state.on_entry.clone();
        new_state.on_exit = old_state.on_exit.clone();
        target.states.insert(new_id, new_state);
        s_map.insert(*old_id, new_id);
    }

    for (old_tid, old_trans) in &source.transitions {
        let new_src = s_map[&old_trans.source];
        let new_tgt = s_map[&old_trans.target];
        let new_tid = target.add_transition(new_src, new_tgt, old_trans.guard.clone(), old_trans.actions.clone());
        t_map.insert(*old_tid, new_tid);
    }

    // Copy epsilon transitions
    for &(src, tgt) in &source.epsilon_transitions {
        target.add_epsilon_transition(s_map[&src], s_map[&tgt]);
    }

    (s_map, t_map)
}

/// Check whether a guard matches an event kind.
fn guard_matches_event(guard: &Guard, event: &EventKind) -> bool {
    match guard {
        Guard::Event(ek) => ek == event,
        Guard::And(gs) => gs.iter().any(|g| guard_matches_event(g, event)),
        Guard::Or(gs) => gs.iter().any(|g| guard_matches_event(g, event)),
        Guard::True => true,
        _ => false,
    }
}

/// Collect event kinds from a guard tree.
fn collect_events(guard: &Guard, events: &mut HashSet<EventKind>) {
    match guard {
        Guard::Event(ek) => {
            events.insert(ek.clone());
        }
        Guard::And(gs) | Guard::Or(gs) => {
            for g in gs {
                collect_events(g, events);
            }
        }
        Guard::Not(g) => collect_events(g, events),
        _ => {}
    }
}

/// Build a name for a subset of states.
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

    fn make_simple_nfa() -> NFA {
        let mut nfa = NFA::new("simple");
        let s0 = nfa.fresh_state_id();
        let s1 = nfa.fresh_state_id();
        let s2 = nfa.fresh_state_id();

        let mut st0 = NFAState::new(s0, "start");
        st0.is_initial = true;
        nfa.add_state(st0);
        nfa.add_state(NFAState::new(s1, "mid"));
        let mut st2 = NFAState::new(s2, "end");
        st2.is_accepting = true;
        nfa.add_state(st2);

        nfa.add_transition(s0, s1, Guard::Event(EventKind::GrabStart), vec![]);
        nfa.add_transition(s1, s2, Guard::Event(EventKind::GrabEnd), vec![]);

        nfa
    }

    #[test]
    fn test_nfa_accepts() {
        let nfa = make_simple_nfa();
        assert!(nfa.accepts(&[EventKind::GrabStart, EventKind::GrabEnd]));
        assert!(!nfa.accepts(&[EventKind::GrabStart]));
        assert!(!nfa.accepts(&[EventKind::GrabEnd]));
        assert!(!nfa.accepts(&[]));
    }

    #[test]
    fn test_epsilon_closure() {
        let mut nfa = NFA::new("eps_test");
        let s0 = nfa.fresh_state_id();
        let s1 = nfa.fresh_state_id();
        let s2 = nfa.fresh_state_id();

        let mut st0 = NFAState::new(s0, "a");
        st0.is_initial = true;
        nfa.add_state(st0);
        nfa.add_state(NFAState::new(s1, "b"));
        let mut st2 = NFAState::new(s2, "c");
        st2.is_accepting = true;
        nfa.add_state(st2);

        nfa.add_epsilon_transition(s0, s1);
        nfa.add_epsilon_transition(s1, s2);

        let init = HashSet::from([s0]);
        let closure = nfa.epsilon_closure(&init);
        assert_eq!(closure.len(), 3);
        assert!(closure.contains(&s0));
        assert!(closure.contains(&s1));
        assert!(closure.contains(&s2));
    }

    #[test]
    fn test_nfa_with_epsilon() {
        let mut nfa = NFA::new("eps_accept");
        let s0 = nfa.fresh_state_id();
        let s1 = nfa.fresh_state_id();
        let s2 = nfa.fresh_state_id();

        let mut st0 = NFAState::new(s0, "a");
        st0.is_initial = true;
        nfa.add_state(st0);
        nfa.add_state(NFAState::new(s1, "b"));
        let mut st2 = NFAState::new(s2, "c");
        st2.is_accepting = true;
        nfa.add_state(st2);

        nfa.add_transition(s0, s1, Guard::Event(EventKind::GrabStart), vec![]);
        nfa.add_epsilon_transition(s1, s2);

        assert!(nfa.accepts(&[EventKind::GrabStart]));
    }

    #[test]
    fn test_to_dfa() {
        let nfa = make_simple_nfa();
        let dfa = nfa.to_dfa();
        assert!(dfa.state_count() >= 2);
        assert!(dfa.accepts(&[EventKind::GrabStart, EventKind::GrabEnd]));
        assert!(!dfa.accepts(&[EventKind::GrabStart]));
    }

    #[test]
    fn test_thompson_concat() {
        let a = from_pattern(&InteractionPattern::Event(EventKind::GrabStart));
        let b = from_pattern(&InteractionPattern::Event(EventKind::GrabEnd));
        let concat = thompson_concat(&a, &b);
        assert!(concat.accepts(&[EventKind::GrabStart, EventKind::GrabEnd]));
        assert!(!concat.accepts(&[EventKind::GrabStart]));
    }

    #[test]
    fn test_thompson_union() {
        let a = from_pattern(&InteractionPattern::Event(EventKind::GrabStart));
        let b = from_pattern(&InteractionPattern::Event(EventKind::GrabEnd));
        let union = thompson_union(&a, &b);
        assert!(union.accepts(&[EventKind::GrabStart]));
        assert!(union.accepts(&[EventKind::GrabEnd]));
        assert!(!union.accepts(&[EventKind::GrabStart, EventKind::GrabEnd]));
    }

    #[test]
    fn test_thompson_star() {
        let a = from_pattern(&InteractionPattern::Event(EventKind::GrabStart));
        let star = thompson_star(&a);
        assert!(star.accepts(&[])); // empty accepted
        assert!(star.accepts(&[EventKind::GrabStart]));
        assert!(star.accepts(&[EventKind::GrabStart, EventKind::GrabStart]));
    }

    #[test]
    fn test_thompson_plus() {
        let a = from_pattern(&InteractionPattern::Event(EventKind::GrabStart));
        let plus = thompson_plus(&a);
        assert!(!plus.accepts(&[])); // empty NOT accepted
        assert!(plus.accepts(&[EventKind::GrabStart]));
        assert!(plus.accepts(&[EventKind::GrabStart, EventKind::GrabStart]));
    }

    #[test]
    fn test_thompson_optional() {
        let a = from_pattern(&InteractionPattern::Event(EventKind::GrabStart));
        let opt = thompson_optional(&a);
        assert!(opt.accepts(&[])); // empty accepted
        assert!(opt.accepts(&[EventKind::GrabStart]));
        assert!(!opt.accepts(&[EventKind::GrabStart, EventKind::GrabStart]));
    }

    #[test]
    fn test_from_pattern_sequence() {
        let pat = InteractionPattern::Sequence(vec![
            InteractionPattern::Event(EventKind::GrabStart),
            InteractionPattern::Event(EventKind::GrabEnd),
        ]);
        let nfa = from_pattern(&pat);
        assert!(nfa.accepts(&[EventKind::GrabStart, EventKind::GrabEnd]));
    }

    #[test]
    fn test_from_pattern_choice() {
        let pat = InteractionPattern::Choice(vec![
            InteractionPattern::Event(EventKind::GrabStart),
            InteractionPattern::Event(EventKind::TouchStart),
        ]);
        let nfa = from_pattern(&pat);
        assert!(nfa.accepts(&[EventKind::GrabStart]));
        assert!(nfa.accepts(&[EventKind::TouchStart]));
    }

    #[test]
    fn test_simulate() {
        let nfa = make_simple_nfa();
        let result = nfa.simulate(&[EventKind::GrabStart, EventKind::GrabEnd]);
        assert!(result.accepted);
        assert_eq!(result.steps.len(), 3); // initial + 2 events
    }

    #[test]
    fn test_token_passing_executor() {
        let nfa = make_simple_nfa();
        let mut exec = TokenPassingExecutor::new(nfa);
        exec.initialize();
        assert_eq!(exec.token_count(), 1);

        let r1 = exec.process_event(&EventKind::GrabStart);
        assert!(!r1.accepting);
        assert!(r1.active_tokens > 0);

        let r2 = exec.process_event(&EventKind::GrabEnd);
        assert!(r2.accepting);
    }

    #[test]
    fn test_to_spatial_automaton() {
        let nfa = make_simple_nfa();
        let auto = nfa.to_spatial_automaton();
        assert_eq!(auto.state_count(), 3);
        assert_eq!(auto.transition_count(), 2);
    }

    #[test]
    fn test_dfa_to_spatial_automaton() {
        let nfa = make_simple_nfa();
        let dfa = nfa.to_dfa();
        let auto = dfa.to_spatial_automaton();
        assert!(auto.state_count() >= 2);
    }

    #[test]
    fn test_nfa_display() {
        let nfa = make_simple_nfa();
        let display = format!("{}", nfa);
        assert!(display.contains("simple"));
    }

    #[test]
    fn test_move_on_event() {
        let nfa = make_simple_nfa();
        let start = HashSet::from([StateId(0)]);
        let moved = nfa.move_on_event(&start, &EventKind::GrabStart);
        assert_eq!(moved.len(), 1);
        assert!(moved.contains(&StateId(1)));
    }

    #[test]
    fn test_nfa_counts() {
        let nfa = make_simple_nfa();
        assert_eq!(nfa.state_count(), 3);
        assert_eq!(nfa.transition_count(), 2);
        assert_eq!(nfa.epsilon_count(), 0);
    }

    #[test]
    fn test_complex_pattern() {
        // (grab_start · grab_end)* · touch_start
        let pat = InteractionPattern::Sequence(vec![
            InteractionPattern::Repeat(Box::new(InteractionPattern::Sequence(vec![
                InteractionPattern::Event(EventKind::GrabStart),
                InteractionPattern::Event(EventKind::GrabEnd),
            ]))),
            InteractionPattern::Event(EventKind::TouchStart),
        ]);
        let nfa = from_pattern(&pat);
        assert!(nfa.accepts(&[EventKind::TouchStart])); // zero repeats
        assert!(nfa.accepts(&[
            EventKind::GrabStart,
            EventKind::GrabEnd,
            EventKind::TouchStart,
        ]));
        assert!(nfa.accepts(&[
            EventKind::GrabStart,
            EventKind::GrabEnd,
            EventKind::GrabStart,
            EventKind::GrabEnd,
            EventKind::TouchStart,
        ]));
    }

    #[test]
    fn test_token_passing_no_match() {
        let nfa = make_simple_nfa();
        let mut exec = TokenPassingExecutor::new(nfa);
        exec.initialize();
        let result = exec.process_event(&EventKind::TouchStart);
        assert_eq!(result.active_tokens, 0);
        assert!(!result.accepting);
    }
}
