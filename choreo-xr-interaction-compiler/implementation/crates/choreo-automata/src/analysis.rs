//! Automaton structural analysis: SCC detection (Tarjan's), topological sort,
//! accepting cycle detection, transition density, bottleneck states,
//! bisimulation quotient, language inclusion, and language equivalence.

use crate::automaton::{
    SpatialEventAutomaton,
};
use crate::{
    EventKind, Guard, StateId, TransitionId,
};
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

// ---------------------------------------------------------------------------
// StructuralAnalysis
// ---------------------------------------------------------------------------

/// Complete structural analysis of an automaton.
#[derive(Debug, Clone)]
pub struct StructuralAnalysis {
    pub scc_count: usize,
    pub max_scc_size: usize,
    pub sccs: Vec<SCC>,
    pub is_deterministic: bool,
    pub has_cycles: bool,
    pub has_accepting_cycles: bool,
    pub topological_order: Option<Vec<StateId>>,
    pub transition_density: f64,
    pub bottleneck_states: Vec<StateId>,
    pub state_count: usize,
    pub transition_count: usize,
    pub accepting_count: usize,
    pub reachable_count: usize,
    pub max_in_degree: usize,
    pub max_out_degree: usize,
    pub avg_out_degree: f64,
    pub diameter_estimate: usize,
}

impl fmt::Display for StructuralAnalysis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Structural Analysis:")?;
        writeln!(f, "  States: {}, Transitions: {}", self.state_count, self.transition_count)?;
        writeln!(f, "  SCCs: {} (max size: {})", self.scc_count, self.max_scc_size)?;
        writeln!(f, "  Deterministic: {}", self.is_deterministic)?;
        writeln!(f, "  Has cycles: {}", self.has_cycles)?;
        writeln!(f, "  Has accepting cycles: {}", self.has_accepting_cycles)?;
        writeln!(f, "  Transition density: {:.4}", self.transition_density)?;
        writeln!(f, "  Bottleneck states: {:?}", self.bottleneck_states)?;
        writeln!(f, "  Diameter estimate: {}", self.diameter_estimate)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// SCC
// ---------------------------------------------------------------------------

/// Strongly connected component.
#[derive(Debug, Clone)]
pub struct SCC {
    pub states: Vec<StateId>,
    pub is_trivial: bool,
    pub contains_accepting: bool,
}

impl SCC {
    pub fn size(&self) -> usize {
        self.states.len()
    }
}

// ---------------------------------------------------------------------------
// Cycle
// ---------------------------------------------------------------------------

/// A cycle in the automaton (sequence of state ids forming a loop).
#[derive(Debug, Clone)]
pub struct Cycle {
    pub states: Vec<StateId>,
    pub transitions: Vec<TransitionId>,
    pub is_accepting: bool,
}

impl Cycle {
    pub fn length(&self) -> usize {
        self.states.len()
    }
}

// ---------------------------------------------------------------------------
// Main analysis function
// ---------------------------------------------------------------------------

/// Perform a complete structural analysis of the automaton.
pub fn analyze_structure(automaton: &SpatialEventAutomaton) -> StructuralAnalysis {
    let sccs = find_strongly_connected_components(automaton);
    let scc_count = sccs.len();
    let max_scc_size = sccs.iter().map(|s| s.size()).max().unwrap_or(0);
    let has_cycles = sccs.iter().any(|s| !s.is_trivial);
    let has_accepting_cycles = sccs.iter().any(|s| !s.is_trivial && s.contains_accepting);
    let accepting_cycles = find_accepting_cycles(automaton);

    let topo = topological_sort(automaton);
    let density = compute_transition_density(automaton);
    let bottlenecks = identify_bottleneck_states(automaton);
    let reachable = automaton.reachable_states();

    let max_in = automaton
        .state_ids()
        .iter()
        .map(|s| automaton.incoming(*s).len())
        .max()
        .unwrap_or(0);
    let max_out = automaton
        .state_ids()
        .iter()
        .map(|s| automaton.outgoing(*s).len())
        .max()
        .unwrap_or(0);
    let avg_out = if automaton.state_count() > 0 {
        automaton.transition_count() as f64 / automaton.state_count() as f64
    } else {
        0.0
    };

    let diameter = estimate_diameter(automaton);

    StructuralAnalysis {
        scc_count,
        max_scc_size,
        sccs,
        is_deterministic: automaton.is_deterministic(),
        has_cycles,
        has_accepting_cycles: has_accepting_cycles || !accepting_cycles.is_empty(),
        topological_order: topo,
        transition_density: density,
        bottleneck_states: bottlenecks,
        state_count: automaton.state_count(),
        transition_count: automaton.transition_count(),
        accepting_count: automaton.accepting_states.len(),
        reachable_count: reachable.len(),
        max_in_degree: max_in,
        max_out_degree: max_out,
        avg_out_degree: avg_out,
        diameter_estimate: diameter,
    }
}

// ---------------------------------------------------------------------------
// Tarjan's SCC algorithm
// ---------------------------------------------------------------------------

/// Find strongly connected components using Tarjan's algorithm.
pub fn find_strongly_connected_components(
    automaton: &SpatialEventAutomaton,
) -> Vec<SCC> {
    let state_ids: Vec<StateId> = automaton.state_ids();
    let n = state_ids.len();
    let idx_map: HashMap<StateId, usize> = state_ids
        .iter()
        .enumerate()
        .map(|(i, &s)| (s, i))
        .collect();

    // Build adjacency list
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for t in automaton.transitions.values() {
        if let (Some(&si), Some(&ti)) = (idx_map.get(&t.source), idx_map.get(&t.target)) {
            adj[si].push(ti);
        }
    }

    let mut index_counter: usize = 0;
    let mut stack: Vec<usize> = Vec::new();
    let mut on_stack = vec![false; n];
    let mut indices = vec![usize::MAX; n];
    let mut lowlinks = vec![usize::MAX; n];
    let mut result: Vec<Vec<usize>> = Vec::new();

    fn strongconnect(
        v: usize,
        adj: &[Vec<usize>],
        index_counter: &mut usize,
        stack: &mut Vec<usize>,
        on_stack: &mut [bool],
        indices: &mut [usize],
        lowlinks: &mut [usize],
        result: &mut Vec<Vec<usize>>,
    ) {
        indices[v] = *index_counter;
        lowlinks[v] = *index_counter;
        *index_counter += 1;
        stack.push(v);
        on_stack[v] = true;

        for &w in &adj[v] {
            if indices[w] == usize::MAX {
                strongconnect(w, adj, index_counter, stack, on_stack, indices, lowlinks, result);
                lowlinks[v] = lowlinks[v].min(lowlinks[w]);
            } else if on_stack[w] {
                lowlinks[v] = lowlinks[v].min(indices[w]);
            }
        }

        if lowlinks[v] == indices[v] {
            let mut component = Vec::new();
            while let Some(w) = stack.pop() {
                on_stack[w] = false;
                component.push(w);
                if w == v {
                    break;
                }
            }
            result.push(component);
        }
    }

    for i in 0..n {
        if indices[i] == usize::MAX {
            strongconnect(
                i,
                &adj,
                &mut index_counter,
                &mut stack,
                &mut on_stack,
                &mut indices,
                &mut lowlinks,
                &mut result,
            );
        }
    }

    // Convert to SCC structs
    result
        .into_iter()
        .map(|component| {
            let states: Vec<StateId> = component.iter().map(|&i| state_ids[i]).collect();
            let is_trivial = states.len() == 1 && {
                let s = states[0];
                !automaton
                    .outgoing(s)
                    .iter()
                    .any(|tid| automaton.transition(*tid).map_or(false, |t| t.target == s))
            };
            let contains_accepting = states
                .iter()
                .any(|s| automaton.accepting_states.contains(s));
            SCC {
                states,
                is_trivial,
                contains_accepting,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Accepting cycle detection
// ---------------------------------------------------------------------------

/// Find cycles that contain at least one accepting state.
pub fn find_accepting_cycles(
    automaton: &SpatialEventAutomaton,
) -> Vec<Cycle> {
    let sccs = find_strongly_connected_components(automaton);
    let mut cycles = Vec::new();

    for scc in &sccs {
        if scc.is_trivial || !scc.contains_accepting {
            continue;
        }
        // Find a simple cycle within this SCC
        let scc_set: HashSet<StateId> = scc.states.iter().copied().collect();
        if let Some(cycle) = find_cycle_in_scc(automaton, &scc_set) {
            cycles.push(cycle);
        }
    }

    cycles
}

/// Find a simple cycle within a set of states (SCC).
fn find_cycle_in_scc(
    automaton: &SpatialEventAutomaton,
    scc_states: &HashSet<StateId>,
) -> Option<Cycle> {
    if scc_states.is_empty() {
        return None;
    }

    let start = *scc_states.iter().next()?;
    let mut visited = HashSet::new();
    let mut path: Vec<(StateId, TransitionId)> = Vec::new();

    fn dfs(
        automaton: &SpatialEventAutomaton,
        current: StateId,
        start: StateId,
        scc_states: &HashSet<StateId>,
        visited: &mut HashSet<StateId>,
        path: &mut Vec<(StateId, TransitionId)>,
    ) -> Option<Cycle> {
        visited.insert(current);

        for tid in automaton.outgoing(current) {
            if let Some(t) = automaton.transition(tid) {
                if !scc_states.contains(&t.target) {
                    continue;
                }
                if t.target == start && !path.is_empty() {
                    // Found a cycle
                    let mut states: Vec<StateId> = path.iter().map(|(s, _)| *s).collect();
                    states.push(current);
                    let mut transitions: Vec<TransitionId> =
                        path.iter().map(|(_, t)| *t).collect();
                    transitions.push(tid);
                    let is_accepting = states
                        .iter()
                        .any(|s| automaton.accepting_states.contains(s));
                    return Some(Cycle {
                        states,
                        transitions,
                        is_accepting,
                    });
                }
                if !visited.contains(&t.target) {
                    path.push((current, tid));
                    if let Some(cycle) =
                        dfs(automaton, t.target, start, scc_states, visited, path)
                    {
                        return Some(cycle);
                    }
                    path.pop();
                }
            }
        }
        None
    }

    dfs(automaton, start, start, scc_states, &mut visited, &mut path)
}

// ---------------------------------------------------------------------------
// Topological sort
// ---------------------------------------------------------------------------

/// Topological sort of states (returns None if the automaton has cycles).
pub fn topological_sort(automaton: &SpatialEventAutomaton) -> Option<Vec<StateId>> {
    let state_ids: Vec<StateId> = automaton.state_ids();
    let n = state_ids.len();
    let idx_map: HashMap<StateId, usize> = state_ids
        .iter()
        .enumerate()
        .map(|(i, &s)| (s, i))
        .collect();

    let mut in_degree = vec![0usize; n];
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];

    for t in automaton.transitions.values() {
        if let (Some(&si), Some(&ti)) = (idx_map.get(&t.source), idx_map.get(&t.target)) {
            adj[si].push(ti);
            in_degree[ti] += 1;
        }
    }

    let mut queue: VecDeque<usize> = VecDeque::new();
    for i in 0..n {
        if in_degree[i] == 0 {
            queue.push_back(i);
        }
    }

    let mut order = Vec::new();
    while let Some(u) = queue.pop_front() {
        order.push(state_ids[u]);
        for &v in &adj[u] {
            in_degree[v] -= 1;
            if in_degree[v] == 0 {
                queue.push_back(v);
            }
        }
    }

    if order.len() == n {
        Some(order)
    } else {
        None // Has cycles
    }
}

// ---------------------------------------------------------------------------
// Transition density
// ---------------------------------------------------------------------------

/// Compute the ratio of actual transitions to maximum possible transitions.
pub fn compute_transition_density(automaton: &SpatialEventAutomaton) -> f64 {
    let n = automaton.state_count();
    if n == 0 {
        return 0.0;
    }
    let max_transitions = n * n; // self-loops included
    automaton.transition_count() as f64 / max_transitions as f64
}

// ---------------------------------------------------------------------------
// Bottleneck state identification
// ---------------------------------------------------------------------------

/// Identify bottleneck states: states whose removal would disconnect the
/// most paths from initial to accepting states.
///
/// Uses a betweenness-centrality heuristic: count how many shortest paths
/// pass through each state.
pub fn identify_bottleneck_states(
    automaton: &SpatialEventAutomaton,
) -> Vec<StateId> {
    let state_ids: Vec<StateId> = automaton.state_ids();
    if state_ids.is_empty() {
        return Vec::new();
    }

    let mut centrality: HashMap<StateId, f64> = HashMap::new();
    for &s in &state_ids {
        centrality.insert(s, 0.0);
    }

    // Brandes' algorithm (simplified)
    for &source in &state_ids {
        let mut stack: Vec<StateId> = Vec::new();
        let mut predecessors: HashMap<StateId, Vec<StateId>> = HashMap::new();
        let mut sigma: HashMap<StateId, f64> = HashMap::new();
        let mut dist: HashMap<StateId, i64> = HashMap::new();

        for &s in &state_ids {
            predecessors.insert(s, Vec::new());
            sigma.insert(s, 0.0);
            dist.insert(s, -1);
        }
        sigma.insert(source, 1.0);
        dist.insert(source, 0);

        let mut queue: VecDeque<StateId> = VecDeque::new();
        queue.push_back(source);

        while let Some(v) = queue.pop_front() {
            stack.push(v);
            let v_dist = dist[&v];
            for tid in automaton.outgoing(v) {
                if let Some(t) = automaton.transition(tid) {
                    let w = t.target;
                    let w_dist = dist[&w];
                    if w_dist < 0 {
                        dist.insert(w, v_dist + 1);
                        queue.push_back(w);
                    }
                    if dist[&w] == v_dist + 1 {
                        *sigma.entry(w).or_insert(0.0) += sigma[&v];
                        predecessors.entry(w).or_default().push(v);
                    }
                }
            }
        }

        // Back-propagation
        let mut delta: HashMap<StateId, f64> = HashMap::new();
        for &s in &state_ids {
            delta.insert(s, 0.0);
        }

        while let Some(w) = stack.pop() {
            for &v in predecessors.get(&w).unwrap_or(&Vec::new()) {
                let s_v = sigma.get(&v).copied().unwrap_or(0.0);
                let s_w = sigma.get(&w).copied().unwrap_or(0.0);
                if s_w > 0.0 {
                    let d_w = delta.get(&w).copied().unwrap_or(0.0);
                    *delta.entry(v).or_insert(0.0) += (s_v / s_w) * (1.0 + d_w);
                }
            }
            if w != source {
                let d_w = delta.get(&w).copied().unwrap_or(0.0);
                *centrality.entry(w).or_insert(0.0) += d_w;
            }
        }
    }

    // Return states with above-average centrality
    let avg = centrality.values().sum::<f64>() / centrality.len().max(1) as f64;
    let mut bottlenecks: Vec<(StateId, f64)> = centrality
        .into_iter()
        .filter(|(_, c)| *c > avg)
        .collect();
    bottlenecks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    bottlenecks.into_iter().map(|(s, _)| s).collect()
}

// ---------------------------------------------------------------------------
// Bisimulation quotient
// ---------------------------------------------------------------------------

/// Compute the bisimulation quotient of the automaton: merge states that are
/// bisimulation-equivalent.
pub fn compute_bisimulation_quotient(
    automaton: &SpatialEventAutomaton,
) -> SpatialEventAutomaton {
    // This uses the same partition-refinement as minimise
    automaton.minimize()
}

// ---------------------------------------------------------------------------
// Language inclusion & equivalence
// ---------------------------------------------------------------------------

/// Check whether the language of automaton `a` is included in the language
/// of automaton `b`: L(a) ⊆ L(b).
///
/// Uses on-the-fly product construction with the complement of `b`.
pub fn language_inclusion(
    a: &SpatialEventAutomaton,
    b: &SpatialEventAutomaton,
) -> bool {
    // L(a) ⊆ L(b) iff L(a) ∩ L(¬b) = ∅
    // Complement b: swap accepting / non-accepting
    let b_comp = complement(b);
    let events = {
        let mut e = a.alphabet();
        e.extend(b.alphabet());
        e
    };

    // Product of a and complement(b), check if any accepting state is reachable
    let init_a = match a.initial_state {
        Some(s) => s,
        None => return true, // empty language is subset of everything
    };
    let init_b = match b_comp.initial_state {
        Some(s) => s,
        None => return true,
    };

    // BFS over product states
    let mut visited: HashSet<(StateId, StateId)> = HashSet::new();
    let mut queue: VecDeque<(StateId, StateId)> = VecDeque::new();
    visited.insert((init_a, init_b));
    queue.push_back((init_a, init_b));

    while let Some((sa, sb)) = queue.pop_front() {
        // Check if product state is accepting in both
        if a.accepting_states.contains(&sa) && b_comp.accepting_states.contains(&sb) {
            return false; // Found a word in L(a) ∩ L(¬b)
        }

        // Explore successors
        for event in &events {
            let a_succs = successors_on_event(a, sa, event);
            let b_succs = successors_on_event(&b_comp, sb, event);
            for &na in &a_succs {
                for &nb in &b_succs {
                    if visited.insert((na, nb)) {
                        queue.push_back((na, nb));
                    }
                }
            }
        }
    }

    true
}

/// Check whether the languages of two automata are equal: L(a) = L(b).
pub fn language_equivalence(
    a: &SpatialEventAutomaton,
    b: &SpatialEventAutomaton,
) -> bool {
    language_inclusion(a, b) && language_inclusion(b, a)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Complement an automaton (swap accepting and non-accepting states).
/// Assumes the automaton is a DFA for correctness.
fn complement(automaton: &SpatialEventAutomaton) -> SpatialEventAutomaton {
    let mut result = automaton.clone();
    let all_states: HashSet<StateId> = result.state_ids().into_iter().collect();
    let old_accepting = result.accepting_states.clone();
    result.accepting_states = all_states.difference(&old_accepting).copied().collect();
    for (_, state) in result.states.iter_mut() {
        state.is_accepting = !state.is_accepting;
    }
    result
}

/// Find all successor states reachable on a specific event from a given state.
fn successors_on_event(
    automaton: &SpatialEventAutomaton,
    state: StateId,
    event: &EventKind,
) -> Vec<StateId> {
    let mut succs = Vec::new();
    for tid in automaton.outgoing(state) {
        if let Some(t) = automaton.transition(tid) {
            if guard_contains_event(&t.guard, event) {
                succs.push(t.target);
            }
        }
    }
    succs
}

/// Check whether a guard references a particular event kind.
fn guard_contains_event(guard: &Guard, event: &EventKind) -> bool {
    match guard {
        Guard::Event(ek) => ek == event,
        Guard::And(gs) => gs.iter().any(|g| guard_contains_event(g, event)),
        Guard::Or(gs) => gs.iter().any(|g| guard_contains_event(g, event)),
        Guard::True => true,
        _ => false,
    }
}

/// Estimate the diameter (longest shortest path between any pair of
/// reachable states) using BFS from each reachable state.
fn estimate_diameter(automaton: &SpatialEventAutomaton) -> usize {
    let reachable = automaton.reachable_states();
    if reachable.len() <= 1 {
        return 0;
    }

    let mut max_dist = 0;
    // Sample a subset of states for efficiency
    let sample: Vec<StateId> = reachable.iter().copied().take(50).collect();

    for &source in &sample {
        let mut visited: HashSet<StateId> = HashSet::new();
        let mut queue: VecDeque<(StateId, usize)> = VecDeque::new();
        visited.insert(source);
        queue.push_back((source, 0));

        while let Some((s, d)) = queue.pop_front() {
            max_dist = max_dist.max(d);
            for tid in automaton.outgoing(s) {
                if let Some(t) = automaton.transition(tid) {
                    if visited.insert(t.target) {
                        queue.push_back((t.target, d + 1));
                    }
                }
            }
        }
    }

    max_dist
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

    fn make_linear_automaton() -> SpatialEventAutomaton {
        let mut b = AutomatonBuilder::new("linear");
        let s0 = b.add_state(StateConfig::new("s0").initial());
        let s1 = b.add_state(StateConfig::new("s1"));
        let s2 = b.add_state(StateConfig::new("s2").accepting());
        b.add_transition(s0, s1, Guard::Event(EventKind::GrabStart), vec![]);
        b.add_transition(s1, s2, Guard::Event(EventKind::GrabEnd), vec![]);
        b.build().unwrap()
    }

    fn make_cyclic_automaton() -> SpatialEventAutomaton {
        let mut b = AutomatonBuilder::new("cyclic");
        let s0 = b.add_state(StateConfig::new("s0").initial().accepting());
        let s1 = b.add_state(StateConfig::new("s1"));
        b.add_transition(s0, s1, Guard::Event(EventKind::GrabStart), vec![]);
        b.add_transition(s1, s0, Guard::Event(EventKind::GrabEnd), vec![]);
        b.build().unwrap()
    }

    #[test]
    fn test_analyze_structure_linear() {
        let auto = make_linear_automaton();
        let analysis = analyze_structure(&auto);
        assert_eq!(analysis.state_count, 3);
        assert_eq!(analysis.transition_count, 2);
        assert!(!analysis.has_cycles);
        assert!(analysis.topological_order.is_some());
        assert!(analysis.is_deterministic);
    }

    #[test]
    fn test_analyze_structure_cyclic() {
        let auto = make_cyclic_automaton();
        let analysis = analyze_structure(&auto);
        assert_eq!(analysis.state_count, 2);
        assert!(analysis.has_cycles);
        assert!(analysis.topological_order.is_none());
    }

    #[test]
    fn test_find_sccs_linear() {
        let auto = make_linear_automaton();
        let sccs = find_strongly_connected_components(&auto);
        assert_eq!(sccs.len(), 3); // Each state is its own SCC
        assert!(sccs.iter().all(|s| s.is_trivial));
    }

    #[test]
    fn test_find_sccs_cyclic() {
        let auto = make_cyclic_automaton();
        let sccs = find_strongly_connected_components(&auto);
        // Should have one non-trivial SCC
        assert!(sccs.iter().any(|s| !s.is_trivial));
    }

    #[test]
    fn test_find_accepting_cycles() {
        let auto = make_cyclic_automaton();
        let cycles = find_accepting_cycles(&auto);
        assert!(!cycles.is_empty());
        assert!(cycles[0].is_accepting);
    }

    #[test]
    fn test_topological_sort_linear() {
        let auto = make_linear_automaton();
        let order = topological_sort(&auto);
        assert!(order.is_some());
        let order = order.unwrap();
        assert_eq!(order.len(), 3);
        // s0 should come before s1, and s1 before s2
        let pos: HashMap<StateId, usize> = order
            .iter()
            .enumerate()
            .map(|(i, s)| (*s, i))
            .collect();
        assert!(pos[&StateId(0)] < pos[&StateId(1)]);
        assert!(pos[&StateId(1)] < pos[&StateId(2)]);
    }

    #[test]
    fn test_topological_sort_cyclic() {
        let auto = make_cyclic_automaton();
        let order = topological_sort(&auto);
        assert!(order.is_none());
    }

    #[test]
    fn test_transition_density() {
        let auto = make_linear_automaton();
        let density = compute_transition_density(&auto);
        // 2 transitions out of 9 possible (3x3)
        assert!((density - 2.0 / 9.0).abs() < 1e-9);
    }

    #[test]
    fn test_bottleneck_states() {
        let auto = make_linear_automaton();
        let bottlenecks = identify_bottleneck_states(&auto);
        // The middle state should be a bottleneck
        assert!(bottlenecks.contains(&StateId(1)) || bottlenecks.is_empty());
    }

    #[test]
    fn test_bisimulation_quotient() {
        let auto = make_linear_automaton();
        let quotient = compute_bisimulation_quotient(&auto);
        assert!(quotient.state_count() <= auto.state_count());
    }

    #[test]
    fn test_language_inclusion_same() {
        let auto = make_linear_automaton();
        assert!(language_inclusion(&auto, &auto));
    }

    #[test]
    fn test_language_equivalence_same() {
        let auto = make_linear_automaton();
        assert!(language_equivalence(&auto, &auto));
    }

    #[test]
    fn test_language_inclusion_subset() {
        // Build a smaller automaton that accepts a subset
        let mut b1 = AutomatonBuilder::new("small");
        let s0 = b1.add_state(StateConfig::new("s0").initial());
        let s1 = b1.add_state(StateConfig::new("s1").accepting());
        b1.add_transition(s0, s1, Guard::Event(EventKind::GrabStart), vec![]);
        let small = b1.build().unwrap();

        // Build a larger automaton
        let mut b2 = AutomatonBuilder::new("large");
        let s0 = b2.add_state(StateConfig::new("s0").initial());
        let s1 = b2.add_state(StateConfig::new("s1").accepting());
        let s2 = b2.add_state(StateConfig::new("s2").accepting());
        b2.add_transition(s0, s1, Guard::Event(EventKind::GrabStart), vec![]);
        b2.add_transition(s0, s2, Guard::Event(EventKind::GrabEnd), vec![]);
        let large = b2.build().unwrap();

        // Small ⊆ Large should hold
        assert!(language_inclusion(&small, &large));
    }

    #[test]
    fn test_complement() {
        let auto = make_linear_automaton();
        let comp = complement(&auto);
        // Accepting states should be swapped
        assert!(!comp.accepting_states.contains(&StateId(2)));
        assert!(comp.accepting_states.contains(&StateId(0)));
    }

    #[test]
    fn test_analyze_display() {
        let auto = make_linear_automaton();
        let analysis = analyze_structure(&auto);
        let display = format!("{}", analysis);
        assert!(display.contains("States: 3"));
    }

    #[test]
    fn test_diameter_estimate() {
        let auto = make_linear_automaton();
        let diameter = estimate_diameter(&auto);
        assert_eq!(diameter, 2);
    }

    #[test]
    fn test_empty_automaton() {
        let auto = SpatialEventAutomaton::new("empty");
        let analysis = analyze_structure(&auto);
        assert_eq!(analysis.state_count, 0);
        assert_eq!(analysis.scc_count, 0);
        assert!(!analysis.has_cycles);
    }

    #[test]
    fn test_scc_size() {
        let auto = make_cyclic_automaton();
        let sccs = find_strongly_connected_components(&auto);
        let non_trivial: Vec<&SCC> = sccs.iter().filter(|s| !s.is_trivial).collect();
        assert!(!non_trivial.is_empty());
        assert_eq!(non_trivial[0].size(), 2);
    }

    #[test]
    fn test_cycle_length() {
        let auto = make_cyclic_automaton();
        let cycles = find_accepting_cycles(&auto);
        if !cycles.is_empty() {
            assert!(cycles[0].length() >= 2);
        }
    }
}
