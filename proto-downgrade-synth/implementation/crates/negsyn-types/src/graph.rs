//! Graph types for protocol state machines.
//!
//! Wraps petgraph with protocol-specific operations including
//! bisimulation computation, quotient construction, SCC analysis,
//! and negotiation-loop detection.

use crate::protocol::{HandshakePhase, NegotiationState, TransitionLabel};
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use petgraph::Direction;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::fmt;

// ── Newtype Ids ──────────────────────────────────────────────────────────

/// Newtype wrapper for state identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct StateId(pub u32);

/// Newtype wrapper for transition identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TransitionId(pub u32);

impl fmt::Display for StateId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "S{}", self.0)
    }
}

impl fmt::Display for TransitionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "T{}", self.0)
    }
}

// ── State and Edge data ──────────────────────────────────────────────────

/// Data stored at each graph node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateData {
    pub id: StateId,
    pub phase: HandshakePhase,
    pub label: String,
    pub properties: HashMap<String, String>,
}

impl StateData {
    pub fn new(id: StateId, phase: HandshakePhase, label: impl Into<String>) -> Self {
        StateData {
            id,
            phase,
            label: label.into(),
            properties: HashMap::new(),
        }
    }

    pub fn with_property(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.properties.insert(key.into(), value.into());
        self
    }
}

/// Data stored on each graph edge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeData {
    pub id: TransitionId,
    pub label: TransitionLabel,
    pub guard: Option<String>,
    pub weight: f64,
}

impl EdgeData {
    pub fn new(id: TransitionId, label: TransitionLabel) -> Self {
        EdgeData {
            id,
            label,
            guard: None,
            weight: 1.0,
        }
    }

    pub fn with_guard(mut self, guard: impl Into<String>) -> Self {
        self.guard = Some(guard.into());
        self
    }

    pub fn with_weight(mut self, weight: f64) -> Self {
        self.weight = weight;
        self
    }
}

// ── StateGraph ───────────────────────────────────────────────────────────

/// Protocol state graph wrapping petgraph with domain-specific operations.
#[derive(Clone, Debug)]
pub struct StateGraph {
    graph: DiGraph<StateData, EdgeData>,
    initial: Option<NodeIndex>,
    state_to_node: HashMap<StateId, NodeIndex>,
    next_state_id: u32,
    next_transition_id: u32,
}

impl StateGraph {
    pub fn new() -> Self {
        StateGraph {
            graph: DiGraph::new(),
            initial: None,
            state_to_node: HashMap::new(),
            next_state_id: 0,
            next_transition_id: 0,
        }
    }

    /// Add a state to the graph.
    pub fn add_state(&mut self, phase: HandshakePhase, label: impl Into<String>) -> StateId {
        let id = StateId(self.next_state_id);
        self.next_state_id += 1;
        let data = StateData::new(id, phase, label);
        let node = self.graph.add_node(data);
        self.state_to_node.insert(id, node);
        if self.initial.is_none() {
            self.initial = Some(node);
        }
        id
    }

    /// Add a transition between states.
    pub fn add_transition(
        &mut self,
        from: StateId,
        to: StateId,
        label: TransitionLabel,
    ) -> Option<TransitionId> {
        let from_node = self.state_to_node.get(&from)?;
        let to_node = self.state_to_node.get(&to)?;
        let id = TransitionId(self.next_transition_id);
        self.next_transition_id += 1;
        let data = EdgeData::new(id, label);
        self.graph.add_edge(*from_node, *to_node, data);
        Some(id)
    }

    /// Set the initial state.
    pub fn set_initial(&mut self, id: StateId) -> bool {
        if let Some(node) = self.state_to_node.get(&id) {
            self.initial = Some(*node);
            true
        } else {
            false
        }
    }

    /// Get state data by ID.
    pub fn state(&self, id: StateId) -> Option<&StateData> {
        self.state_to_node
            .get(&id)
            .and_then(|n| self.graph.node_weight(*n))
    }

    /// Number of states.
    pub fn state_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Number of transitions.
    pub fn transition_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// All state IDs.
    pub fn state_ids(&self) -> Vec<StateId> {
        self.graph
            .node_indices()
            .filter_map(|n| self.graph.node_weight(n).map(|d| d.id))
            .collect()
    }

    /// Successor states of a given state.
    pub fn successors(&self, id: StateId) -> Vec<(TransitionLabel, StateId)> {
        if let Some(&node) = self.state_to_node.get(&id) {
            self.graph
                .edges_directed(node, Direction::Outgoing)
                .filter_map(|e| {
                    let target = e.target();
                    let edge_data = e.weight();
                    let target_data = self.graph.node_weight(target)?;
                    Some((edge_data.label.clone(), target_data.id))
                })
                .collect()
        } else {
            vec![]
        }
    }

    /// Predecessor states of a given state.
    pub fn predecessors(&self, id: StateId) -> Vec<(TransitionLabel, StateId)> {
        if let Some(&node) = self.state_to_node.get(&id) {
            self.graph
                .edges_directed(node, Direction::Incoming)
                .filter_map(|e| {
                    let source = e.source();
                    let edge_data = e.weight();
                    let source_data = self.graph.node_weight(source)?;
                    Some((edge_data.label.clone(), source_data.id))
                })
                .collect()
        } else {
            vec![]
        }
    }

    /// Terminal states (no outgoing transitions).
    pub fn terminal_states(&self) -> Vec<StateId> {
        self.graph
            .node_indices()
            .filter(|&n| {
                self.graph
                    .edges_directed(n, Direction::Outgoing)
                    .next()
                    .is_none()
            })
            .filter_map(|n| self.graph.node_weight(n).map(|d| d.id))
            .collect()
    }

    /// Reachable states from the initial state via BFS.
    pub fn reachable_from_initial(&self) -> BTreeSet<StateId> {
        let mut visited = BTreeSet::new();
        if let Some(initial) = self.initial {
            let mut queue = VecDeque::new();
            queue.push_back(initial);
            while let Some(node) = queue.pop_front() {
                if let Some(data) = self.graph.node_weight(node) {
                    if visited.insert(data.id) {
                        for edge in self.graph.edges_directed(node, Direction::Outgoing) {
                            queue.push_back(edge.target());
                        }
                    }
                }
            }
        }
        visited
    }

    /// Find shortest path between two states (BFS).
    pub fn shortest_path(&self, from: StateId, to: StateId) -> Option<Vec<StateId>> {
        let from_node = *self.state_to_node.get(&from)?;
        let to_node = *self.state_to_node.get(&to)?;

        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let mut parent: HashMap<NodeIndex, NodeIndex> = HashMap::new();
        queue.push_back(from_node);
        visited.insert(from_node);

        while let Some(current) = queue.pop_front() {
            if current == to_node {
                // Reconstruct path
                let mut path = vec![to_node];
                let mut cur = to_node;
                while let Some(&p) = parent.get(&cur) {
                    path.push(p);
                    cur = p;
                }
                path.reverse();
                return Some(
                    path.iter()
                        .filter_map(|n| self.graph.node_weight(*n).map(|d| d.id))
                        .collect(),
                );
            }
            for edge in self.graph.edges_directed(current, Direction::Outgoing) {
                let target = edge.target();
                if visited.insert(target) {
                    parent.insert(target, current);
                    queue.push_back(target);
                }
            }
        }
        None
    }

    /// Detect cycles using DFS. Returns true if any cycle exists.
    pub fn has_cycle(&self) -> bool {
        let mut visited = HashSet::new();
        let mut in_stack = HashSet::new();

        for node in self.graph.node_indices() {
            if !visited.contains(&node) {
                if self.dfs_has_cycle(node, &mut visited, &mut in_stack) {
                    return true;
                }
            }
        }
        false
    }

    fn dfs_has_cycle(
        &self,
        node: NodeIndex,
        visited: &mut HashSet<NodeIndex>,
        in_stack: &mut HashSet<NodeIndex>,
    ) -> bool {
        visited.insert(node);
        in_stack.insert(node);

        for edge in self.graph.edges_directed(node, Direction::Outgoing) {
            let target = edge.target();
            if !visited.contains(&target) {
                if self.dfs_has_cycle(target, visited, in_stack) {
                    return true;
                }
            } else if in_stack.contains(&target) {
                return true;
            }
        }

        in_stack.remove(&node);
        false
    }

    /// Compute strongly connected components (Tarjan's algorithm).
    pub fn strongly_connected_components(&self) -> Vec<Vec<StateId>> {
        let sccs = petgraph::algo::tarjan_scc(&self.graph);
        sccs.into_iter()
            .map(|scc| {
                scc.into_iter()
                    .filter_map(|n| self.graph.node_weight(n).map(|d| d.id))
                    .collect()
            })
            .collect()
    }

    /// Find negotiation loops (SCCs with > 1 node or self-loops).
    pub fn negotiation_loops(&self) -> Vec<Vec<StateId>> {
        self.strongly_connected_components()
            .into_iter()
            .filter(|scc| {
                if scc.len() > 1 {
                    return true;
                }
                // Check for self-loops
                if let Some(&id) = scc.first() {
                    return self.successors(id).iter().any(|(_, succ)| *succ == id);
                }
                false
            })
            .collect()
    }

    /// States where adversary transitions are available.
    pub fn adversary_states(&self) -> Vec<StateId> {
        self.graph
            .node_indices()
            .filter(|&n| {
                self.graph
                    .edges_directed(n, Direction::Outgoing)
                    .any(|e| e.weight().label.is_adversary())
            })
            .filter_map(|n| self.graph.node_weight(n).map(|d| d.id))
            .collect()
    }
}

impl Default for StateGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for StateGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "StateGraph({} states, {} transitions)",
            self.state_count(),
            self.transition_count()
        )
    }
}

// ── Bisimulation Relation (Definition D3) ────────────────────────────────

/// A bisimulation relation between two state graphs.
///
/// Two states are bisimilar if they have the same observable behavior:
/// same set of enabled actions and transitions lead to bisimilar states.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BisimulationRelation {
    /// Pairs of bisimilar states.
    pub pairs: BTreeSet<(StateId, StateId)>,
    /// Number of refinement iterations.
    pub iterations: u32,
    pub converged: bool,
}

impl BisimulationRelation {
    /// Compute bisimulation on a single graph (partition refinement).
    ///
    /// Groups states into equivalence classes based on their behavior.
    pub fn compute(graph: &StateGraph) -> Self {
        let state_ids = graph.state_ids();
        if state_ids.is_empty() {
            return BisimulationRelation {
                pairs: BTreeSet::new(),
                iterations: 0,
                converged: true,
            };
        }

        // Initial partition: group by handshake phase
        let mut partition: HashMap<StateId, u32> = HashMap::new();
        let mut phase_to_class: HashMap<String, u32> = HashMap::new();
        let mut next_class = 0u32;

        for &sid in &state_ids {
            if let Some(data) = graph.state(sid) {
                let phase_key = format!("{:?}", data.phase);
                let class = *phase_to_class.entry(phase_key).or_insert_with(|| {
                    let c = next_class;
                    next_class += 1;
                    c
                });
                partition.insert(sid, class);
            }
        }

        // Refinement loop
        let max_iter = 100u32;
        let mut iterations = 0;
        let mut converged = false;

        for _ in 0..max_iter {
            iterations += 1;
            let mut new_partition: HashMap<StateId, u32> = HashMap::new();
            let mut signature_to_class: HashMap<Vec<u32>, u32> = HashMap::new();
            next_class = 0;

            for &sid in &state_ids {
                let current_class = partition.get(&sid).copied().unwrap_or(0);
                let mut sig = vec![current_class];

                // Include successor classes sorted
                let mut succ_classes: Vec<u32> = graph
                    .successors(sid)
                    .iter()
                    .filter_map(|(_, succ_id)| partition.get(succ_id).copied())
                    .collect();
                succ_classes.sort();
                sig.extend(succ_classes);

                let class = *signature_to_class.entry(sig).or_insert_with(|| {
                    let c = next_class;
                    next_class += 1;
                    c
                });
                new_partition.insert(sid, class);
            }

            if new_partition == partition {
                converged = true;
                break;
            }
            partition = new_partition;
        }

        // Build bisimilar pairs from partition
        let mut class_members: HashMap<u32, Vec<StateId>> = HashMap::new();
        for (&sid, &class) in &partition {
            class_members.entry(class).or_default().push(sid);
        }

        let mut pairs = BTreeSet::new();
        for members in class_members.values() {
            for i in 0..members.len() {
                for j in i..members.len() {
                    let a = members[i].min(members[j]);
                    let b = members[i].max(members[j]);
                    pairs.insert((a, b));
                }
            }
        }

        BisimulationRelation {
            pairs,
            iterations,
            converged,
        }
    }

    /// Whether two states are bisimilar.
    pub fn are_bisimilar(&self, a: StateId, b: StateId) -> bool {
        let lo = a.min(b);
        let hi = a.max(b);
        self.pairs.contains(&(lo, hi))
    }

    /// Number of equivalence classes.
    pub fn class_count(&self) -> usize {
        let mut classes: HashSet<StateId> = HashSet::new();
        for &(a, _) in &self.pairs {
            classes.insert(a);
        }
        classes.len()
    }
}

impl fmt::Display for BisimulationRelation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Bisimulation({} pairs, {} iterations, converged={})",
            self.pairs.len(),
            self.iterations,
            self.converged
        )
    }
}

// ── Quotient Graph ───────────────────────────────────────────────────────

/// The quotient graph obtained from a bisimulation relation.
///
/// Each node represents an equivalence class of bisimilar states.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuotientGraph {
    /// Equivalence classes (class id → member state ids).
    pub classes: BTreeMap<u32, Vec<StateId>>,
    /// Transitions between classes.
    pub transitions: Vec<(u32, TransitionLabel, u32)>,
    pub initial_class: u32,
}

impl QuotientGraph {
    /// Construct the quotient graph from a state graph and bisimulation.
    pub fn from_bisimulation(graph: &StateGraph, bisim: &BisimulationRelation) -> Self {
        let state_ids = graph.state_ids();

        // Build equivalence classes using union-find approach
        let mut class_of: HashMap<StateId, u32> = HashMap::new();
        let mut classes: BTreeMap<u32, Vec<StateId>> = BTreeMap::new();
        let mut next_class = 0u32;

        for &sid in &state_ids {
            if class_of.contains_key(&sid) {
                continue;
            }
            let class = next_class;
            next_class += 1;
            class_of.insert(sid, class);
            classes.entry(class).or_default().push(sid);

            for &other in &state_ids {
                if !class_of.contains_key(&other) && bisim.are_bisimilar(sid, other) {
                    class_of.insert(other, class);
                    classes.entry(class).or_default().push(other);
                }
            }
        }

        // Build transitions between classes
        let mut transition_set: HashSet<(u32, String, u32)> = HashSet::new();
        let mut transitions = Vec::new();

        for &sid in &state_ids {
            let from_class = class_of[&sid];
            for (label, succ) in graph.successors(sid) {
                if let Some(&to_class) = class_of.get(&succ) {
                    let key = (from_class, format!("{:?}", label), to_class);
                    if transition_set.insert(key) {
                        transitions.push((from_class, label, to_class));
                    }
                }
            }
        }

        // Find the initial class
        let initial_class = state_ids
            .first()
            .and_then(|s| class_of.get(s).copied())
            .unwrap_or(0);

        QuotientGraph {
            classes,
            transitions,
            initial_class,
        }
    }

    /// Number of classes (states in the quotient).
    pub fn class_count(&self) -> usize {
        self.classes.len()
    }

    /// Number of transitions in the quotient.
    pub fn transition_count(&self) -> usize {
        self.transitions.len()
    }

    /// Compression ratio (original states / quotient states).
    pub fn compression_ratio(&self, original_state_count: usize) -> f64 {
        if self.classes.is_empty() {
            return 0.0;
        }
        original_state_count as f64 / self.classes.len() as f64
    }
}

impl fmt::Display for QuotientGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "QuotientGraph({} classes, {} transitions)",
            self.class_count(),
            self.transition_count()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{ClientActionKind, ServerActionKind, TransitionLabel};

    fn build_simple_graph() -> StateGraph {
        let mut g = StateGraph::new();
        let s0 = g.add_state(HandshakePhase::Init, "init");
        let s1 = g.add_state(HandshakePhase::ClientHelloSent, "client_hello");
        let s2 = g.add_state(HandshakePhase::ServerHelloReceived, "server_hello");
        let s3 = g.add_state(HandshakePhase::Done, "done");
        let s4 = g.add_state(HandshakePhase::Abort, "abort");

        g.add_transition(
            s0, s1,
            TransitionLabel::ClientAction(ClientActionKind::SendClientHello {
                ciphers: vec![0x1301],
                version: 0x0304,
            }),
        );
        g.add_transition(
            s1, s2,
            TransitionLabel::ServerAction(ServerActionKind::SendServerHello {
                cipher: 0x1301,
                version: 0x0304,
            }),
        );
        g.add_transition(s2, s3, TransitionLabel::Tau);
        g.add_transition(s1, s4, TransitionLabel::Tau);
        g
    }

    #[test]
    fn test_graph_construction() {
        let g = build_simple_graph();
        assert_eq!(g.state_count(), 5);
        assert_eq!(g.transition_count(), 4);
    }

    #[test]
    fn test_successors_predecessors() {
        let g = build_simple_graph();
        let succs = g.successors(StateId(0));
        assert_eq!(succs.len(), 1);

        let succs = g.successors(StateId(1));
        assert_eq!(succs.len(), 2);

        let preds = g.predecessors(StateId(1));
        assert_eq!(preds.len(), 1);
    }

    #[test]
    fn test_reachable_states() {
        let g = build_simple_graph();
        let reachable = g.reachable_from_initial();
        assert_eq!(reachable.len(), 5);
    }

    #[test]
    fn test_terminal_states() {
        let g = build_simple_graph();
        let terminals = g.terminal_states();
        assert_eq!(terminals.len(), 2); // done and abort
    }

    #[test]
    fn test_shortest_path() {
        let g = build_simple_graph();
        let path = g.shortest_path(StateId(0), StateId(3));
        assert!(path.is_some());
        let p = path.unwrap();
        assert_eq!(p.len(), 4); // init -> hello -> server_hello -> done
    }

    #[test]
    fn test_no_path() {
        let g = build_simple_graph();
        let path = g.shortest_path(StateId(3), StateId(0)); // done -> init (no back-edge)
        assert!(path.is_none());
    }

    #[test]
    fn test_no_cycle() {
        let g = build_simple_graph();
        assert!(!g.has_cycle());
    }

    #[test]
    fn test_with_cycle() {
        let mut g = StateGraph::new();
        let s0 = g.add_state(HandshakePhase::Init, "a");
        let s1 = g.add_state(HandshakePhase::ClientHelloSent, "b");
        g.add_transition(s0, s1, TransitionLabel::Tau);
        g.add_transition(s1, s0, TransitionLabel::Tau);
        assert!(g.has_cycle());
    }

    #[test]
    fn test_scc() {
        let mut g = StateGraph::new();
        let s0 = g.add_state(HandshakePhase::Init, "a");
        let s1 = g.add_state(HandshakePhase::ClientHelloSent, "b");
        let s2 = g.add_state(HandshakePhase::Done, "c");
        g.add_transition(s0, s1, TransitionLabel::Tau);
        g.add_transition(s1, s0, TransitionLabel::Tau);
        g.add_transition(s1, s2, TransitionLabel::Tau);

        let sccs = g.strongly_connected_components();
        let big_scc: Vec<_> = sccs.iter().filter(|scc| scc.len() > 1).collect();
        assert_eq!(big_scc.len(), 1);
    }

    #[test]
    fn test_negotiation_loops() {
        let mut g = StateGraph::new();
        let s0 = g.add_state(HandshakePhase::Init, "a");
        let s1 = g.add_state(HandshakePhase::ClientHelloSent, "b");
        let s2 = g.add_state(HandshakePhase::Done, "c");
        g.add_transition(s0, s1, TransitionLabel::Tau);
        g.add_transition(s1, s0, TransitionLabel::Tau);
        g.add_transition(s1, s2, TransitionLabel::Tau);

        let loops = g.negotiation_loops();
        assert_eq!(loops.len(), 1);
    }

    #[test]
    fn test_bisimulation() {
        let mut g = StateGraph::new();
        let s0 = g.add_state(HandshakePhase::Init, "init1");
        let s1 = g.add_state(HandshakePhase::Init, "init2");
        let s2 = g.add_state(HandshakePhase::Done, "done1");
        let s3 = g.add_state(HandshakePhase::Done, "done2");
        g.add_transition(s0, s2, TransitionLabel::Tau);
        g.add_transition(s1, s3, TransitionLabel::Tau);

        let bisim = BisimulationRelation::compute(&g);
        assert!(bisim.converged);
        assert!(bisim.are_bisimilar(StateId(0), StateId(1)));
        assert!(bisim.are_bisimilar(StateId(2), StateId(3)));
        assert!(!bisim.are_bisimilar(StateId(0), StateId(2)));
    }

    #[test]
    fn test_quotient_graph() {
        let mut g = StateGraph::new();
        let s0 = g.add_state(HandshakePhase::Init, "init1");
        let s1 = g.add_state(HandshakePhase::Init, "init2");
        let s2 = g.add_state(HandshakePhase::Done, "done1");
        let s3 = g.add_state(HandshakePhase::Done, "done2");
        g.add_transition(s0, s2, TransitionLabel::Tau);
        g.add_transition(s1, s3, TransitionLabel::Tau);

        let bisim = BisimulationRelation::compute(&g);
        let quotient = QuotientGraph::from_bisimulation(&g, &bisim);

        assert!(quotient.class_count() <= g.state_count());
        let ratio = quotient.compression_ratio(g.state_count());
        assert!(ratio >= 1.0);
    }

    #[test]
    fn test_empty_graph() {
        let g = StateGraph::new();
        assert_eq!(g.state_count(), 0);
        assert!(!g.has_cycle());
        let reachable = g.reachable_from_initial();
        assert!(reachable.is_empty());
    }

    #[test]
    fn test_state_data() {
        let data = StateData::new(StateId(0), HandshakePhase::Init, "test")
            .with_property("cipher", "AES_128_GCM");
        assert_eq!(data.properties.get("cipher").unwrap(), "AES_128_GCM");
    }

    #[test]
    fn test_adversary_states() {
        use crate::protocol::AdversaryActionKind;
        let mut g = StateGraph::new();
        let s0 = g.add_state(HandshakePhase::Init, "init");
        let s1 = g.add_state(HandshakePhase::ClientHelloSent, "hello");
        let s2 = g.add_state(HandshakePhase::Abort, "abort");
        g.add_transition(s0, s1, TransitionLabel::Tau);
        g.add_transition(
            s1, s2,
            TransitionLabel::AdversaryAction(AdversaryActionKind::Drop),
        );

        let adv = g.adversary_states();
        assert_eq!(adv.len(), 1);
        assert_eq!(adv[0], StateId(1));
    }

    #[test]
    fn test_display_ids() {
        assert_eq!(format!("{}", StateId(42)), "S42");
        assert_eq!(format!("{}", TransitionId(7)), "T7");
    }
}
