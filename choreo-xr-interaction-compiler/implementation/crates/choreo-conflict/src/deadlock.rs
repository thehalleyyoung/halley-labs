//! Deadlock detection for spatial-event automata.
//!
//! Builds a wait-for graph from transitions, detects cycles via DFS
//! colouring, checks spatial feasibility of cycles, and classifies
//! deadlocks. Also includes livelock detection.

use choreo_automata::automaton::{SpatialEventAutomaton, Transition};
use choreo_automata::{Guard, SpatialPredicate, StateId, TransitionId};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Classification of a deadlock.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeadlockClassification {
    /// All states in cycle are waiting for spatial guards that require the
    /// *other* states to fire first — classic circular wait.
    CircularWait,
    /// A state has no outgoing transitions at all.
    TerminalTrap,
    /// All outgoing guards are permanently false (given scene constraints).
    GuardBlockage,
    /// The cycle is reachable but depends on a particular spatial
    /// configuration that may or may not occur.
    ConditionalDeadlock,
}

impl fmt::Display for DeadlockClassification {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CircularWait => write!(f, "circular-wait"),
            Self::TerminalTrap => write!(f, "terminal-trap"),
            Self::GuardBlockage => write!(f, "guard-blockage"),
            Self::ConditionalDeadlock => write!(f, "conditional-deadlock"),
        }
    }
}

/// A concrete deadlock finding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Deadlock {
    /// The set of state ids forming the deadlock.
    pub states: Vec<u32>,
    /// Human-readable descriptions of the spatial conditions involved.
    pub spatial_conditions: Vec<String>,
    /// Whether the deadlock is reachable from the initial state.
    pub is_reachable: bool,
    /// Witness trace (state ids) from the initial state to the deadlock.
    pub trace: Vec<u32>,
    /// Classification of the deadlock.
    pub classification: DeadlockClassification,
}

/// A witness execution trace demonstrating how a deadlock is reached.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeadlockWitness {
    /// Sequence of (state, transition_fired) leading to the deadlock.
    pub steps: Vec<(u32, Option<u32>)>,
    /// The final deadlocked state set.
    pub deadlocked_states: Vec<u32>,
}

/// Trait for measuring whether an automaton is making progress.
pub trait ProgressMetric: fmt::Debug {
    /// Return a progress value for the given state. Higher means more
    /// progress has been made.
    fn progress(&self, state_id: u32) -> f64;

    /// Name of this metric (for reporting).
    fn name(&self) -> &str;
}

/// Default progress metric based on state id ordering (simple but useful
/// for detecting trivial livelocks).
#[derive(Debug)]
pub struct OrdinalProgress;

impl ProgressMetric for OrdinalProgress {
    fn progress(&self, state_id: u32) -> f64 {
        state_id as f64
    }
    fn name(&self) -> &str {
        "ordinal"
    }
}

/// Accepting-distance metric: progress = 1/(1 + shortest path to accepting).
#[derive(Debug)]
pub struct AcceptingDistanceProgress {
    distances: HashMap<u32, usize>,
}

impl AcceptingDistanceProgress {
    pub fn from_automaton(aut: &SpatialEventAutomaton) -> Self {
        let mut distances = HashMap::new();
        // Reverse BFS from accepting states
        let mut queue: VecDeque<StateId> = VecDeque::new();
        for &sid in &aut.accepting_states {
            distances.insert(sid.0, 0usize);
            queue.push_back(sid);
        }
        while let Some(s) = queue.pop_front() {
            let d = distances[&s.0];
            for t in aut.transitions.values() {
                if t.target == s && !distances.contains_key(&t.source.0) {
                    distances.insert(t.source.0, d + 1);
                    queue.push_back(t.source);
                }
            }
        }
        Self { distances }
    }
}

impl ProgressMetric for AcceptingDistanceProgress {
    fn progress(&self, state_id: u32) -> f64 {
        match self.distances.get(&state_id) {
            Some(&d) => 1.0 / (1.0 + d as f64),
            None => 0.0,
        }
    }
    fn name(&self) -> &str {
        "accepting-distance"
    }
}

// ---------------------------------------------------------------------------
// Wait-for graph
// ---------------------------------------------------------------------------

/// A wait-for graph: nodes are states, an edge (A → B) means "state A is
/// waiting for a guard that depends on being in state B".
#[derive(Debug, Clone)]
pub struct WaitForGraph {
    /// Adjacency list: state → set of states it waits for.
    pub adj: HashMap<u32, HashSet<u32>>,
    /// Edge labels: (src, dst) → description of the guard condition.
    pub edge_labels: HashMap<(u32, u32), String>,
    pub nodes: HashSet<u32>,
}

impl WaitForGraph {
    /// Build a wait-for graph from an automaton's states and transitions.
    ///
    /// A state S *waits for* state T when:
    /// - S has an outgoing transition whose guard references T's
    ///   incoming transitions or spatial conditions that require another
    ///   entity to be in a state reachable only through T.
    /// - Simplified: S waits for T if all of S's outgoing transitions
    ///   target T or have guards referencing entities that overlap with
    ///   guards on T's outgoing transitions.
    pub fn build(
        states: &[StateId],
        transitions: &[&Transition],
    ) -> Self {
        let mut adj: HashMap<u32, HashSet<u32>> = HashMap::new();
        let mut edge_labels: HashMap<(u32, u32), String> = HashMap::new();
        let mut nodes = HashSet::new();

        for &s in states {
            nodes.insert(s.0);
            adj.entry(s.0).or_default();
        }

        // For each state, look at outgoing transitions. If a transition's
        // guard references a spatial predicate that is also referenced by
        // transitions *into* another state, we add a wait-for edge.
        let guard_preds: HashMap<u32, Vec<String>> = {
            let mut m: HashMap<u32, Vec<String>> = HashMap::new();
            for t in transitions {
                let preds = guard_predicate_names(&t.guard);
                m.entry(t.source.0).or_default().extend(preds);
            }
            m
        };

        // Build reverse map: predicate_name → states whose incoming transitions use it
        let mut pred_to_target_states: HashMap<String, HashSet<u32>> = HashMap::new();
        for t in transitions {
            for p in guard_predicate_names(&t.guard) {
                pred_to_target_states
                    .entry(p)
                    .or_default()
                    .insert(t.target.0);
            }
        }

        for (&src, preds) in &guard_preds {
            for p in preds {
                if let Some(targets) = pred_to_target_states.get(p) {
                    for &tgt in targets {
                        if tgt != src && nodes.contains(&tgt) {
                            adj.entry(src).or_default().insert(tgt);
                            edge_labels
                                .entry((src, tgt))
                                .or_insert_with(|| p.clone());
                        }
                    }
                }
            }
        }

        Self {
            adj,
            edge_labels,
            nodes,
        }
    }

    /// Find all cycles in the wait-for graph using DFS with colouring.
    pub fn find_cycles(&self) -> Vec<Vec<u32>> {
        #[derive(Clone, Copy, PartialEq)]
        enum Color {
            White,
            Gray,
            Black,
        }

        let mut color: HashMap<u32, Color> = self
            .nodes
            .iter()
            .map(|&n| (n, Color::White))
            .collect();
        let mut parent: HashMap<u32, Option<u32>> = HashMap::new();
        let mut cycles: Vec<Vec<u32>> = Vec::new();

        fn dfs(
            node: u32,
            adj: &HashMap<u32, HashSet<u32>>,
            color: &mut HashMap<u32, Color>,
            parent: &mut HashMap<u32, Option<u32>>,
            path: &mut Vec<u32>,
            cycles: &mut Vec<Vec<u32>>,
        ) {
            color.insert(node, Color::Gray);
            path.push(node);

            if let Some(neighbours) = adj.get(&node) {
                let mut sorted: Vec<u32> = neighbours.iter().copied().collect();
                sorted.sort();
                for &next in &sorted {
                    match color.get(&next).copied().unwrap_or(Color::White) {
                        Color::White => {
                            parent.insert(next, Some(node));
                            dfs(next, adj, color, parent, path, cycles);
                        }
                        Color::Gray => {
                            // Found a cycle — extract it
                            if let Some(pos) = path.iter().position(|&x| x == next) {
                                let cycle: Vec<u32> = path[pos..].to_vec();
                                if cycle.len() >= 2 {
                                    cycles.push(cycle);
                                }
                            }
                        }
                        Color::Black => {}
                    }
                }
            }

            path.pop();
            color.insert(node, Color::Black);
        }

        let mut sorted_nodes: Vec<u32> = self.nodes.iter().copied().collect();
        sorted_nodes.sort();

        for &node in &sorted_nodes {
            if color.get(&node) == Some(&Color::White) {
                parent.insert(node, None);
                let mut path = Vec::new();
                dfs(
                    node,
                    &self.adj,
                    &mut color,
                    &mut parent,
                    &mut path,
                    &mut cycles,
                );
            }
        }

        // Deduplicate cycles (normalise by rotating to smallest element first)
        let mut unique: Vec<Vec<u32>> = Vec::new();
        let mut seen: HashSet<Vec<u32>> = HashSet::new();
        for mut cycle in cycles {
            let min_pos = cycle
                .iter()
                .enumerate()
                .min_by_key(|&(_, v)| v)
                .map(|(i, _)| i)
                .unwrap_or(0);
            cycle.rotate_left(min_pos);
            if seen.insert(cycle.clone()) {
                unique.push(cycle);
            }
        }
        unique
    }
}

// ---------------------------------------------------------------------------
// Spatial feasibility
// ---------------------------------------------------------------------------

/// Check whether a deadlock cycle can actually occur given spatial constraints.
///
/// A cycle is *spatially infeasible* if the guards along the cycle edges
/// are contradictory (e.g., entity must be inside AND outside a region
/// simultaneously).
pub fn check_spatial_feasibility(
    cycle: &[u32],
    transitions: &[&Transition],
) -> bool {
    // Collect all spatial predicates referenced by transitions whose source
    // is in the cycle.
    let cycle_set: HashSet<u32> = cycle.iter().copied().collect();
    let mut positive_preds: HashSet<String> = HashSet::new();
    let mut negative_preds: HashSet<String> = HashSet::new();

    for t in transitions {
        if cycle_set.contains(&t.source.0) {
            collect_guard_polarity(&t.guard, true, &mut positive_preds, &mut negative_preds);
        }
    }

    // If any predicate is required both positively and negatively, infeasible
    let contradiction = positive_preds.intersection(&negative_preds).next().is_some();
    !contradiction // feasible iff no contradiction
}

fn collect_guard_polarity(
    guard: &Guard,
    positive: bool,
    pos: &mut HashSet<String>,
    neg: &mut HashSet<String>,
) {
    match guard {
        Guard::Spatial(sp) => {
            let name = format!("{:?}", sp);
            if positive {
                pos.insert(name);
            } else {
                neg.insert(name);
            }
        }
        Guard::Not(inner) => {
            collect_guard_polarity(inner, !positive, pos, neg);
        }
        Guard::And(gs) | Guard::Or(gs) => {
            for g in gs {
                collect_guard_polarity(g, positive, pos, neg);
            }
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// DeadlockDetector
// ---------------------------------------------------------------------------

/// Main deadlock detector.
#[derive(Debug)]
pub struct DeadlockDetector {
    include_unreachable: bool,
}

impl DeadlockDetector {
    pub fn new() -> Self {
        Self {
            include_unreachable: false,
        }
    }

    /// If set, also report deadlocks in unreachable portions of the automaton.
    pub fn include_unreachable(mut self, yes: bool) -> Self {
        self.include_unreachable = yes;
        self
    }

    /// Detect deadlocks in the given automaton.
    pub fn detect(&self, automaton: &SpatialEventAutomaton) -> Vec<Deadlock> {
        let states: Vec<StateId> = automaton.state_ids();
        let transitions: Vec<&Transition> = automaton.transitions.values().collect();
        self.detect_deadlocks(&states, &transitions, automaton)
    }

    /// Core detection: find deadlocks among the given states and transitions.
    pub fn detect_deadlocks(
        &self,
        states: &[StateId],
        transitions: &[&Transition],
        automaton: &SpatialEventAutomaton,
    ) -> Vec<Deadlock> {
        let mut deadlocks = Vec::new();

        // 1. Terminal traps: states with no outgoing transitions
        let outgoing_counts = {
            let mut m: HashMap<u32, usize> = HashMap::new();
            for &s in states {
                m.insert(s.0, 0);
            }
            for t in transitions {
                *m.entry(t.source.0).or_default() += 1;
            }
            m
        };

        let reachable = automaton.reachable_states();
        let reachable_set: HashSet<u32> = reachable.iter().map(|s| s.0).collect();

        for (&sid, &count) in &outgoing_counts {
            if count == 0 {
                let is_accepting = automaton
                    .accepting_states
                    .contains(&StateId(sid));
                if is_accepting {
                    continue; // accepting terminal states are fine
                }
                let is_reachable = reachable_set.contains(&sid);
                if !is_reachable && !self.include_unreachable {
                    continue;
                }
                let trace = if is_reachable {
                    self.find_trace_to(automaton, StateId(sid))
                } else {
                    Vec::new()
                };
                deadlocks.push(Deadlock {
                    states: vec![sid],
                    spatial_conditions: vec![],
                    is_reachable,
                    trace,
                    classification: DeadlockClassification::TerminalTrap,
                });
            }
        }

        // 2. Guard blockage: states where all guards are trivially false
        for &s in states {
            let out_trans: Vec<&&Transition> = transitions
                .iter()
                .filter(|t| t.source == s)
                .collect();
            if out_trans.is_empty() {
                continue; // already covered by terminal traps
            }
            let all_false = out_trans.iter().all(|t| t.guard.is_trivially_false());
            if all_false {
                let is_reachable = reachable_set.contains(&s.0);
                if !is_reachable && !self.include_unreachable {
                    continue;
                }
                let trace = if is_reachable {
                    self.find_trace_to(automaton, s)
                } else {
                    Vec::new()
                };
                deadlocks.push(Deadlock {
                    states: vec![s.0],
                    spatial_conditions: out_trans
                        .iter()
                        .map(|t| format!("{}", t.guard))
                        .collect(),
                    is_reachable,
                    trace,
                    classification: DeadlockClassification::GuardBlockage,
                });
            }
        }

        // 3. Circular wait: cycles in the wait-for graph
        let wfg = WaitForGraph::build(states, transitions);
        let cycles = wfg.find_cycles();

        for cycle in &cycles {
            let feasible = check_spatial_feasibility(cycle, transitions);
            let cycle_reachable = cycle.iter().any(|s| reachable_set.contains(s));
            if !cycle_reachable && !self.include_unreachable {
                continue;
            }

            let spatial_conds: Vec<String> = cycle
                .iter()
                .filter_map(|&s| {
                    cycle.iter().find_map(|&t| {
                        wfg.edge_labels.get(&(s, t)).cloned()
                    })
                })
                .collect();

            let trace = if cycle_reachable {
                self.find_trace_to(automaton, StateId(cycle[0]))
            } else {
                Vec::new()
            };

            let classification = if feasible {
                DeadlockClassification::CircularWait
            } else {
                DeadlockClassification::ConditionalDeadlock
            };

            deadlocks.push(Deadlock {
                states: cycle.clone(),
                spatial_conditions: spatial_conds,
                is_reachable: cycle_reachable,
                trace,
                classification,
            });
        }

        deadlocks
    }

    /// BFS to find a path from the initial state to `target`.
    fn find_trace_to(&self, automaton: &SpatialEventAutomaton, target: StateId) -> Vec<u32> {
        let init = match automaton.initial_state {
            Some(s) => s,
            None => return Vec::new(),
        };
        if init == target {
            return vec![init.0];
        }
        let mut visited: HashMap<u32, u32> = HashMap::new();
        let mut queue = VecDeque::new();
        queue.push_back(init);
        visited.insert(init.0, init.0);

        while let Some(s) = queue.pop_front() {
            for t in automaton.transitions.values() {
                if t.source == s && !visited.contains_key(&t.target.0) {
                    visited.insert(t.target.0, s.0);
                    if t.target == target {
                        // Reconstruct path
                        let mut path = vec![target.0];
                        let mut cur = target.0;
                        while cur != init.0 {
                            cur = visited[&cur];
                            path.push(cur);
                        }
                        path.reverse();
                        return path;
                    }
                    queue.push_back(t.target);
                }
            }
        }
        Vec::new()
    }
}

impl Default for DeadlockDetector {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// LivelockDetector
// ---------------------------------------------------------------------------

/// Detects infinite execution loops that never reach an accepting state
/// or never make progress according to a supplied metric.
#[derive(Debug)]
pub struct LivelockDetector {
    max_cycle_len: usize,
}

impl LivelockDetector {
    pub fn new() -> Self {
        Self { max_cycle_len: 100 }
    }

    pub fn with_max_cycle_len(mut self, n: usize) -> Self {
        self.max_cycle_len = n;
        self
    }

    /// Detect livelocks: cycles reachable from the initial state that
    /// do not contain any accepting state.
    pub fn detect_livelocks(&self, automaton: &SpatialEventAutomaton) -> Vec<Vec<u32>> {
        let reachable = automaton.reachable_states();
        let reaching_accept = automaton.states_reaching_accepting();

        // Find SCCs in the reachable subgraph
        let sccs = self.tarjan_scc(automaton, &reachable);
        let mut livelocks = Vec::new();

        for scc in &sccs {
            if scc.len() < 2 {
                // Check self-loop
                let s = scc[0];
                let has_self_loop = automaton
                    .transitions
                    .values()
                    .any(|t| t.source == s && t.target == s);
                if !has_self_loop {
                    continue;
                }
            }
            // SCC is a livelock if none of its states can reach an accepting state
            let any_can_accept = scc.iter().any(|s| reaching_accept.contains(s));
            if !any_can_accept {
                let ids: Vec<u32> = scc.iter().map(|s| s.0).collect();
                livelocks.push(ids);
            }
        }
        livelocks
    }

    /// Detect cycles where progress (according to the given metric) never
    /// increases.
    pub fn detect_no_progress_cycles(
        &self,
        automaton: &SpatialEventAutomaton,
        metric: &dyn ProgressMetric,
    ) -> Vec<Vec<u32>> {
        let reachable = automaton.reachable_states();
        let sccs = self.tarjan_scc(automaton, &reachable);
        let mut result = Vec::new();

        for scc in &sccs {
            if scc.len() < 2 {
                continue;
            }
            // Check if progress is monotonic along any traversal order
            let max_progress = scc
                .iter()
                .map(|s| ordered_float::OrderedFloat(metric.progress(s.0)))
                .max();
            let min_progress = scc
                .iter()
                .map(|s| ordered_float::OrderedFloat(metric.progress(s.0)))
                .min();
            if max_progress == min_progress {
                // All states have the same progress — no progress cycle
                let ids: Vec<u32> = scc.iter().map(|s| s.0).collect();
                result.push(ids);
            }
        }
        result
    }

    /// Tarjan's SCC algorithm.
    fn tarjan_scc(
        &self,
        automaton: &SpatialEventAutomaton,
        subset: &HashSet<StateId>,
    ) -> Vec<Vec<StateId>> {
        struct TarjanState {
            index_counter: usize,
            stack: Vec<StateId>,
            on_stack: HashSet<StateId>,
            index: HashMap<StateId, usize>,
            lowlink: HashMap<StateId, usize>,
            sccs: Vec<Vec<StateId>>,
        }

        fn strongconnect(
            v: StateId,
            automaton: &SpatialEventAutomaton,
            subset: &HashSet<StateId>,
            state: &mut TarjanState,
        ) {
            state.index.insert(v, state.index_counter);
            state.lowlink.insert(v, state.index_counter);
            state.index_counter += 1;
            state.stack.push(v);
            state.on_stack.insert(v);

            for t in automaton.transitions.values() {
                if t.source == v && subset.contains(&t.target) {
                    let w = t.target;
                    if !state.index.contains_key(&w) {
                        strongconnect(w, automaton, subset, state);
                        let low_w = state.lowlink[&w];
                        let low_v = state.lowlink[&v];
                        state.lowlink.insert(v, low_v.min(low_w));
                    } else if state.on_stack.contains(&w) {
                        let idx_w = state.index[&w];
                        let low_v = state.lowlink[&v];
                        state.lowlink.insert(v, low_v.min(idx_w));
                    }
                }
            }

            if state.lowlink[&v] == state.index[&v] {
                let mut scc = Vec::new();
                loop {
                    let w = state.stack.pop().unwrap();
                    state.on_stack.remove(&w);
                    scc.push(w);
                    if w == v {
                        break;
                    }
                }
                state.sccs.push(scc);
            }
        }

        let mut ts = TarjanState {
            index_counter: 0,
            stack: Vec::new(),
            on_stack: HashSet::new(),
            index: HashMap::new(),
            lowlink: HashMap::new(),
            sccs: Vec::new(),
        };

        let mut sorted: Vec<StateId> = subset.iter().copied().collect();
        sorted.sort();
        for s in sorted {
            if !ts.index.contains_key(&s) {
                strongconnect(s, automaton, subset, &mut ts);
            }
        }
        ts.sccs
    }
}

impl Default for LivelockDetector {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract human-readable predicate names from a guard.
fn guard_predicate_names(guard: &Guard) -> Vec<String> {
    let mut names = Vec::new();
    collect_predicate_names(guard, &mut names);
    names
}

fn collect_predicate_names(guard: &Guard, names: &mut Vec<String>) {
    match guard {
        Guard::Spatial(sp) => {
            names.push(format!("{:?}", sp));
            match sp {
                SpatialPredicate::And(preds) | SpatialPredicate::Or(preds) => {
                    for p in preds {
                        collect_predicate_names(&Guard::Spatial(p.clone()), names);
                    }
                }
                SpatialPredicate::Not(inner) => {
                    collect_predicate_names(&Guard::Spatial((**inner).clone()), names);
                }
                _ => {}
            }
        }
        Guard::And(gs) | Guard::Or(gs) => {
            for g in gs {
                collect_predicate_names(g, names);
            }
        }
        Guard::Not(g) => collect_predicate_names(g, names),
        Guard::Event(ek) => names.push(format!("{}", ek)),
        Guard::Temporal(te) => names.push(format!("{:?}", te)),
        Guard::True | Guard::False => {}
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use choreo_automata::automaton::{State, Transition};

    fn make_automaton(
        n_states: u32,
        edges: &[(u32, u32, Guard)],
        initial: u32,
        accepting: &[u32],
    ) -> SpatialEventAutomaton {
        let mut aut = SpatialEventAutomaton::new("test");
        for i in 0..n_states {
            let mut s = State::new(StateId(i), format!("s{}", i));
            if i == initial {
                s.is_initial = true;
            }
            if accepting.contains(&i) {
                s.is_accepting = true;
            }
            aut.add_state(s);
        }
        for (idx, (src, tgt, guard)) in edges.iter().enumerate() {
            let t = Transition::new(
                TransitionId(idx as u32),
                StateId(*src),
                StateId(*tgt),
                guard.clone(),
                vec![],
            );
            aut.add_transition(t);
        }
        aut
    }

    #[test]
    fn terminal_trap_detected() {
        // s0 -> s1, s1 has no outgoing and is not accepting
        let aut = make_automaton(
            2,
            &[(0, 1, Guard::Event(EventKind::GrabStart))],
            0,
            &[],
        );
        let detector = DeadlockDetector::new();
        let dls = detector.detect(&aut);
        assert!(dls.iter().any(|d| d.states == vec![1]
            && d.classification == DeadlockClassification::TerminalTrap));
    }

    #[test]
    fn accepting_terminal_not_deadlock() {
        let aut = make_automaton(
            2,
            &[(0, 1, Guard::Event(EventKind::GrabStart))],
            0,
            &[1],
        );
        let detector = DeadlockDetector::new();
        let dls = detector.detect(&aut);
        assert!(
            dls.iter()
                .all(|d| d.classification != DeadlockClassification::TerminalTrap),
        );
    }

    #[test]
    fn guard_blockage_detected() {
        // s0 has one transition with Guard::False
        let aut = make_automaton(2, &[(0, 1, Guard::False)], 0, &[1]);
        let detector = DeadlockDetector::new();
        let dls = detector.detect(&aut);
        assert!(dls.iter().any(|d| d.classification == DeadlockClassification::GuardBlockage));
    }

    #[test]
    fn wait_for_graph_cycle() {
        let states = vec![StateId(0), StateId(1), StateId(2)];
        let sp_a = SpatialPredicate::Inside {
            entity: choreo_automata::EntityId("e1".into()),
            region: choreo_automata::RegionId("r1".into()),
        };
        let sp_b = SpatialPredicate::Inside {
            entity: choreo_automata::EntityId("e2".into()),
            region: choreo_automata::RegionId("r2".into()),
        };
        let transitions = vec![
            Transition::new(
                TransitionId(0),
                StateId(0),
                StateId(1),
                Guard::Spatial(sp_a.clone()),
                vec![],
            ),
            Transition::new(
                TransitionId(1),
                StateId(1),
                StateId(2),
                Guard::Spatial(sp_b.clone()),
                vec![],
            ),
            Transition::new(
                TransitionId(2),
                StateId(2),
                StateId(0),
                Guard::Spatial(sp_a.clone()),
                vec![],
            ),
        ];
        let trefs: Vec<&Transition> = transitions.iter().collect();
        let wfg = WaitForGraph::build(&states, &trefs);
        let cycles = wfg.find_cycles();
        // There should be at least one cycle involving the three states
        assert!(!cycles.is_empty());
    }

    #[test]
    fn no_deadlock_in_simple_linear() {
        // s0 -> s1 -> s2 (accepting), all with true guards
        let aut = make_automaton(
            3,
            &[
                (0, 1, Guard::Event(EventKind::GrabStart)),
                (1, 2, Guard::Event(EventKind::GrabEnd)),
            ],
            0,
            &[2],
        );
        let detector = DeadlockDetector::new();
        let dls = detector.detect(&aut);
        // Only s2 is terminal but it's accepting, so no deadlocks
        let terminal_traps: Vec<_> = dls
            .iter()
            .filter(|d| d.classification == DeadlockClassification::TerminalTrap)
            .collect();
        assert!(terminal_traps.is_empty());
    }

    #[test]
    fn livelock_detected() {
        // s0 -> s1 -> s0 (cycle), s2 accepting but unreachable from cycle
        let aut = make_automaton(
            3,
            &[
                (0, 1, Guard::Event(EventKind::GrabStart)),
                (1, 0, Guard::Event(EventKind::GrabEnd)),
            ],
            0,
            &[2],
        );
        let detector = LivelockDetector::new();
        let livelocks = detector.detect_livelocks(&aut);
        assert!(!livelocks.is_empty());
    }

    #[test]
    fn no_livelock_when_cycle_contains_accepting() {
        // s0 -> s1 -> s0, s0 is accepting
        let aut = make_automaton(
            2,
            &[
                (0, 1, Guard::Event(EventKind::GrabStart)),
                (1, 0, Guard::Event(EventKind::GrabEnd)),
            ],
            0,
            &[0],
        );
        let detector = LivelockDetector::new();
        let livelocks = detector.detect_livelocks(&aut);
        assert!(livelocks.is_empty());
    }

    #[test]
    fn spatial_feasibility_contradiction() {
        let sp = SpatialPredicate::Inside {
            entity: choreo_automata::EntityId("e1".into()),
            region: choreo_automata::RegionId("r1".into()),
        };
        let transitions = vec![
            Transition::new(
                TransitionId(0),
                StateId(0),
                StateId(1),
                Guard::Spatial(sp.clone()),
                vec![],
            ),
            Transition::new(
                TransitionId(1),
                StateId(1),
                StateId(0),
                Guard::Not(Box::new(Guard::Spatial(sp.clone()))),
                vec![],
            ),
        ];
        let trefs: Vec<&Transition> = transitions.iter().collect();
        // The cycle [0,1] has contradictory guards
        let feasible = check_spatial_feasibility(&[0, 1], &trefs);
        assert!(!feasible);
    }

    #[test]
    fn progress_metric_accepting_distance() {
        let aut = make_automaton(
            4,
            &[
                (0, 1, Guard::Event(EventKind::GrabStart)),
                (1, 2, Guard::Event(EventKind::TouchStart)),
                (2, 3, Guard::Event(EventKind::GrabEnd)),
            ],
            0,
            &[3],
        );
        let metric = AcceptingDistanceProgress::from_automaton(&aut);
        assert!(metric.progress(3) > metric.progress(0));
        assert!(metric.progress(2) > metric.progress(1));
    }

    #[test]
    fn deadlock_witness_trace() {
        // s0 -> s1 -> s2 (trap)
        let aut = make_automaton(
            3,
            &[
                (0, 1, Guard::Event(EventKind::GrabStart)),
                (1, 2, Guard::Event(EventKind::GrabEnd)),
            ],
            0,
            &[],
        );
        let detector = DeadlockDetector::new();
        let dls = detector.detect(&aut);
        let trap = dls
            .iter()
            .find(|d| d.states == vec![2] && d.classification == DeadlockClassification::TerminalTrap);
        assert!(trap.is_some());
        let trace = &trap.unwrap().trace;
        assert_eq!(trace, &vec![0, 1, 2]);
    }
}
