//! Interference analysis between automata components.
//!
//! Computes an interference graph whose nodes are automaton states (or zones)
//! and whose edges represent conflicts: spatial, temporal, or resource-based.
//! Also provides interference-guided decomposition for compositional verification.

use choreo_automata::automaton::{SpatialEventAutomaton, Transition};
use choreo_automata::{Guard, SpatialPredicate, StateId, TransitionId};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

// ---------------------------------------------------------------------------
// InterferenceKind
// ---------------------------------------------------------------------------

/// Kind of interference between two states/components.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InterferenceKind {
    /// Both states guard on overlapping spatial regions.
    SpatialConflict,
    /// Both states guard on overlapping time windows.
    TemporalConflict,
    /// Both states read/write to the same variables or entities.
    ResourceConflict,
    /// No interference detected.
    None,
}

impl fmt::Display for InterferenceKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SpatialConflict => write!(f, "spatial"),
            Self::TemporalConflict => write!(f, "temporal"),
            Self::ResourceConflict => write!(f, "resource"),
            Self::None => write!(f, "none"),
        }
    }
}

// ---------------------------------------------------------------------------
// InterferenceGraph
// ---------------------------------------------------------------------------

/// An interference graph over automaton states.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterferenceGraph {
    /// Node ids (state ids).
    pub nodes: Vec<u32>,
    /// Edges: (source, target, kind).
    pub edges: Vec<(u32, u32, InterferenceKind)>,
    /// Adjacency list for fast lookup.
    #[serde(skip)]
    adj: HashMap<u32, Vec<(u32, InterferenceKind)>>,
}

impl InterferenceGraph {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            adj: HashMap::new(),
        }
    }

    pub fn add_node(&mut self, id: u32) {
        if !self.adj.contains_key(&id) {
            self.nodes.push(id);
            self.adj.entry(id).or_default();
        }
    }

    pub fn add_edge(&mut self, a: u32, b: u32, kind: InterferenceKind) {
        if kind == InterferenceKind::None {
            return;
        }
        self.edges.push((a, b, kind));
        self.adj.entry(a).or_default().push((b, kind));
        self.adj.entry(b).or_default().push((a, kind));
    }

    pub fn neighbors(&self, node: u32) -> Vec<(u32, InterferenceKind)> {
        self.adj.get(&node).cloned().unwrap_or_default()
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Return edges of a specific kind.
    pub fn edges_of_kind(&self, kind: InterferenceKind) -> Vec<(u32, u32)> {
        self.edges
            .iter()
            .filter(|(_, _, k)| *k == kind)
            .map(|(a, b, _)| (*a, *b))
            .collect()
    }

    /// Connected components via BFS.
    pub fn connected_components(&self) -> Vec<Vec<u32>> {
        let mut visited: HashSet<u32> = HashSet::new();
        let mut components = Vec::new();

        for &n in &self.nodes {
            if visited.contains(&n) {
                continue;
            }
            let mut component = Vec::new();
            let mut queue = VecDeque::new();
            queue.push_back(n);
            visited.insert(n);
            while let Some(cur) = queue.pop_front() {
                component.push(cur);
                for (nb, _) in self.neighbors(cur) {
                    if visited.insert(nb) {
                        queue.push_back(nb);
                    }
                }
            }
            component.sort();
            components.push(component);
        }
        components
    }

    /// Graph density: |E| / (|V|*(|V|-1)/2).
    pub fn density(&self) -> f64 {
        let n = self.nodes.len();
        if n < 2 {
            return 0.0;
        }
        let max_edges = n * (n - 1) / 2;
        self.edges.len() as f64 / max_edges as f64
    }

    /// Maximum degree.
    pub fn max_degree(&self) -> usize {
        self.adj.values().map(|v| v.len()).max().unwrap_or(0)
    }
}

impl Default for InterferenceGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// InterferenceAnalyzer
// ---------------------------------------------------------------------------

/// Analyses an automaton (or a pair of automata) for interference.
#[derive(Debug)]
pub struct InterferenceAnalyzer {
    /// Entity ids that are considered shared resources.
    shared_entities: HashSet<String>,
}

impl InterferenceAnalyzer {
    pub fn new() -> Self {
        Self {
            shared_entities: HashSet::new(),
        }
    }

    /// Register an entity as a shared resource.
    pub fn add_shared_entity(&mut self, entity: impl Into<String>) {
        self.shared_entities.insert(entity.into());
    }

    /// Compute the interference graph for a single automaton.
    ///
    /// Two states interfere when their outgoing transitions reference
    /// overlapping spatial predicates, temporal windows, or shared entities.
    pub fn compute_interference(
        &self,
        automaton: &SpatialEventAutomaton,
    ) -> InterferenceGraph {
        let state_ids: Vec<u32> = automaton.state_ids().into_iter().map(|s| s.0).collect();
        let transitions: Vec<&Transition> = automaton.transitions.values().collect();
        self.compute_from_states_transitions(&state_ids, &transitions)
    }

    /// Compute interference from raw data.
    pub fn compute_from_states_transitions(
        &self,
        state_ids: &[u32],
        transitions: &[&Transition],
    ) -> InterferenceGraph {
        let mut graph = InterferenceGraph::new();
        for &s in state_ids {
            graph.add_node(s);
        }

        // Collect per-state predicate sets and entity sets
        let mut state_spatial: HashMap<u32, HashSet<String>> = HashMap::new();
        let mut state_temporal: HashMap<u32, bool> = HashMap::new();
        let mut state_entities: HashMap<u32, HashSet<String>> = HashMap::new();

        for t in transitions {
            let spatial_preds = collect_spatial_predicate_ids(&t.guard);
            state_spatial
                .entry(t.source.0)
                .or_default()
                .extend(spatial_preds.clone());

            let has_temporal = has_temporal_guard(&t.guard);
            if has_temporal {
                state_temporal.insert(t.source.0, true);
            }

            let entities = collect_entity_ids(&t.guard);
            state_entities
                .entry(t.source.0)
                .or_default()
                .extend(entities);

            // Also consider actions for resource conflicts
            for action in &t.actions {
                if let choreo_automata::Action::MoveEntity { entity, .. }
                | choreo_automata::Action::Highlight { entity, .. }
                | choreo_automata::Action::ClearHighlight(entity) = action
                {
                    state_entities
                        .entry(t.source.0)
                        .or_default()
                        .insert(entity.0.clone());
                }
            }
        }

        // Compare all pairs
        for i in 0..state_ids.len() {
            for j in (i + 1)..state_ids.len() {
                let a = state_ids[i];
                let b = state_ids[j];

                let kind = self.check_interference(
                    a,
                    b,
                    &state_spatial,
                    &state_temporal,
                    &state_entities,
                );
                if kind != InterferenceKind::None {
                    graph.add_edge(a, b, kind);
                }
            }
        }

        graph
    }

    fn check_interference(
        &self,
        a: u32,
        b: u32,
        spatial: &HashMap<u32, HashSet<String>>,
        temporal: &HashMap<u32, bool>,
        entities: &HashMap<u32, HashSet<String>>,
    ) -> InterferenceKind {
        // Check resource conflict (highest priority)
        let ents_a = entities.get(&a);
        let ents_b = entities.get(&b);
        if let (Some(ea), Some(eb)) = (ents_a, ents_b) {
            let shared: HashSet<_> = ea.intersection(eb).collect();
            let has_shared_resource = shared.iter().any(|e| self.shared_entities.contains(*e));
            // Even without explicit shared_entities, overlapping entities are a resource conflict
            if has_shared_resource || !shared.is_empty() {
                return InterferenceKind::ResourceConflict;
            }
        }

        // Check spatial conflict
        let sp_a = spatial.get(&a);
        let sp_b = spatial.get(&b);
        if let (Some(sa), Some(sb)) = (sp_a, sp_b) {
            if sa.intersection(sb).next().is_some() {
                return InterferenceKind::SpatialConflict;
            }
        }

        // Check temporal conflict
        let ta = temporal.get(&a).copied().unwrap_or(false);
        let tb = temporal.get(&b).copied().unwrap_or(false);
        if ta && tb {
            return InterferenceKind::TemporalConflict;
        }

        InterferenceKind::None
    }

    /// Decompose the automaton into independent groups based on interference.
    pub fn decompose(
        &self,
        automaton: &SpatialEventAutomaton,
    ) -> Vec<InterferenceComponent> {
        let graph = self.compute_interference(automaton);
        let components = graph.connected_components();

        components
            .into_iter()
            .map(|states| {
                let shared_preds = self.shared_predicates_in_component(&states, automaton);
                InterferenceComponent {
                    states,
                    shared_predicates: shared_preds,
                }
            })
            .collect()
    }

    fn shared_predicates_in_component(
        &self,
        states: &[u32],
        automaton: &SpatialEventAutomaton,
    ) -> Vec<String> {
        let mut all_preds: HashMap<String, usize> = HashMap::new();
        let state_set: HashSet<u32> = states.iter().copied().collect();

        for t in automaton.transitions.values() {
            if state_set.contains(&t.source.0) {
                let preds = collect_spatial_predicate_ids(&t.guard);
                for p in preds {
                    *all_preds.entry(p).or_default() += 1;
                }
            }
        }

        // A predicate is "shared" if referenced by more than one state
        all_preds
            .into_iter()
            .filter(|(_, count)| *count > 1)
            .map(|(name, _)| name)
            .collect()
    }
}

impl Default for InterferenceAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// InterferenceComponent
// ---------------------------------------------------------------------------

/// A connected component from interference-guided decomposition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterferenceComponent {
    pub states: Vec<u32>,
    pub shared_predicates: Vec<String>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn collect_spatial_predicate_ids(guard: &Guard) -> HashSet<String> {
    let mut ids = HashSet::new();
    collect_sp_ids_inner(guard, &mut ids);
    ids
}

fn collect_sp_ids_inner(guard: &Guard, ids: &mut HashSet<String>) {
    match guard {
        Guard::Spatial(sp) => {
            ids.insert(format!("{:?}", sp));
            match sp {
                SpatialPredicate::And(preds) | SpatialPredicate::Or(preds) => {
                    for p in preds {
                        collect_sp_ids_inner(&Guard::Spatial(p.clone()), ids);
                    }
                }
                SpatialPredicate::Not(inner) => {
                    collect_sp_ids_inner(&Guard::Spatial((**inner).clone()), ids);
                }
                _ => {}
            }
        }
        Guard::And(gs) | Guard::Or(gs) => {
            for g in gs {
                collect_sp_ids_inner(g, ids);
            }
        }
        Guard::Not(g) => collect_sp_ids_inner(g, ids),
        _ => {}
    }
}

fn has_temporal_guard(guard: &Guard) -> bool {
    match guard {
        Guard::Temporal(_) => true,
        Guard::And(gs) | Guard::Or(gs) => gs.iter().any(has_temporal_guard),
        Guard::Not(g) => has_temporal_guard(g),
        _ => false,
    }
}

fn collect_entity_ids(guard: &Guard) -> HashSet<String> {
    let mut ids = HashSet::new();
    collect_entities_inner(guard, &mut ids);
    ids
}

fn collect_entities_inner(guard: &Guard, ids: &mut HashSet<String>) {
    match guard {
        Guard::Spatial(sp) => {
            collect_entities_from_spatial(sp, ids);
        }
        Guard::And(gs) | Guard::Or(gs) => {
            for g in gs {
                collect_entities_inner(g, ids);
            }
        }
        Guard::Not(g) => collect_entities_inner(g, ids),
        _ => {}
    }
}

fn collect_entities_from_spatial(sp: &SpatialPredicate, ids: &mut HashSet<String>) {
    match sp {
        SpatialPredicate::Inside { entity, .. } => {
            ids.insert(entity.0.clone());
        }
        SpatialPredicate::Proximity {
            entity_a,
            entity_b,
            ..
        } => {
            ids.insert(entity_a.0.clone());
            ids.insert(entity_b.0.clone());
        }
        SpatialPredicate::GazeAt { entity, .. } => {
            ids.insert(entity.0.clone());
        }
        SpatialPredicate::Contact {
            entity_a,
            entity_b,
        } => {
            ids.insert(entity_a.0.clone());
            ids.insert(entity_b.0.clone());
        }
        SpatialPredicate::Grasping { hand, object } => {
            ids.insert(hand.0.clone());
            ids.insert(object.0.clone());
        }
        SpatialPredicate::And(preds) | SpatialPredicate::Or(preds) => {
            for p in preds {
                collect_entities_from_spatial(p, ids);
            }
        }
        SpatialPredicate::Not(inner) => {
            collect_entities_from_spatial(inner, ids);
        }
        SpatialPredicate::Named(_) => {}
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use choreo_automata::automaton::{State, Transition};
    use choreo_automata::{EntityId, EventKind, Guard, RegionId, StateId, TransitionId};

    fn make_automaton(
        n_states: u32,
        edges: &[(u32, u32, Guard)],
        initial: u32,
    ) -> SpatialEventAutomaton {
        let mut aut = SpatialEventAutomaton::new("test_interference");
        for i in 0..n_states {
            let mut s = State::new(StateId(i), format!("s{}", i));
            if i == initial {
                s.is_initial = true;
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
    fn no_interference_disjoint_predicates() {
        let sp_a = SpatialPredicate::Inside {
            entity: EntityId("e1".into()),
            region: RegionId("r1".into()),
        };
        let sp_b = SpatialPredicate::Inside {
            entity: EntityId("e2".into()),
            region: RegionId("r2".into()),
        };
        let aut = make_automaton(
            3,
            &[
                (0, 1, Guard::Spatial(sp_a)),
                (0, 2, Guard::Spatial(sp_b)),
            ],
            0,
        );
        let analyzer = InterferenceAnalyzer::new();
        let graph = analyzer.compute_interference(&aut);
        // States 1 and 2 don't interfere (different entities)
        let edges_12: Vec<_> = graph
            .edges
            .iter()
            .filter(|(a, b, _)| (*a == 1 && *b == 2) || (*a == 2 && *b == 1))
            .collect();
        assert!(edges_12.is_empty());
    }

    #[test]
    fn spatial_interference_shared_entity() {
        let sp_a = SpatialPredicate::Inside {
            entity: EntityId("e1".into()),
            region: RegionId("r1".into()),
        };
        let sp_b = SpatialPredicate::Inside {
            entity: EntityId("e1".into()),
            region: RegionId("r2".into()),
        };
        let aut = make_automaton(
            3,
            &[
                (0, 1, Guard::Spatial(sp_a)),
                (1, 2, Guard::Spatial(sp_b)),
            ],
            0,
        );
        let analyzer = InterferenceAnalyzer::new();
        let graph = analyzer.compute_interference(&aut);
        // s0 and s1 both reference e1
        let has_resource = graph
            .edges
            .iter()
            .any(|(_, _, k)| *k == InterferenceKind::ResourceConflict);
        assert!(has_resource);
    }

    #[test]
    fn connected_components_separate() {
        let mut graph = InterferenceGraph::new();
        graph.add_node(0);
        graph.add_node(1);
        graph.add_node(2);
        graph.add_node(3);
        graph.add_edge(0, 1, InterferenceKind::SpatialConflict);
        graph.add_edge(2, 3, InterferenceKind::TemporalConflict);
        let comps = graph.connected_components();
        assert_eq!(comps.len(), 2);
    }

    #[test]
    fn connected_components_single() {
        let mut graph = InterferenceGraph::new();
        graph.add_node(0);
        graph.add_node(1);
        graph.add_node(2);
        graph.add_edge(0, 1, InterferenceKind::SpatialConflict);
        graph.add_edge(1, 2, InterferenceKind::TemporalConflict);
        let comps = graph.connected_components();
        assert_eq!(comps.len(), 1);
        assert_eq!(comps[0].len(), 3);
    }

    #[test]
    fn density_calculation() {
        let mut graph = InterferenceGraph::new();
        graph.add_node(0);
        graph.add_node(1);
        graph.add_node(2);
        // Complete graph on 3 nodes → 3 edges → density = 1.0
        graph.add_edge(0, 1, InterferenceKind::SpatialConflict);
        graph.add_edge(0, 2, InterferenceKind::SpatialConflict);
        graph.add_edge(1, 2, InterferenceKind::SpatialConflict);
        assert!((graph.density() - 1.0).abs() < 0.01);
    }

    #[test]
    fn decompose_into_components() {
        let sp_a = SpatialPredicate::Inside {
            entity: EntityId("e_isolated".into()),
            region: RegionId("r_isolated".into()),
        };
        let aut = make_automaton(
            4,
            &[
                (0, 1, Guard::Event(EventKind::GrabStart)),
                (2, 3, Guard::Spatial(sp_a)),
            ],
            0,
        );
        let analyzer = InterferenceAnalyzer::new();
        let components = analyzer.decompose(&aut);
        // Should get multiple components since s0/s1 and s2/s3 don't share entities
        assert!(components.len() >= 2);
    }

    #[test]
    fn edges_of_kind_filter() {
        let mut graph = InterferenceGraph::new();
        graph.add_node(0);
        graph.add_node(1);
        graph.add_node(2);
        graph.add_edge(0, 1, InterferenceKind::SpatialConflict);
        graph.add_edge(1, 2, InterferenceKind::TemporalConflict);
        let spatial = graph.edges_of_kind(InterferenceKind::SpatialConflict);
        assert_eq!(spatial.len(), 1);
        assert_eq!(spatial[0], (0, 1));
    }

    #[test]
    fn max_degree() {
        let mut graph = InterferenceGraph::new();
        graph.add_node(0);
        graph.add_node(1);
        graph.add_node(2);
        graph.add_node(3);
        graph.add_edge(0, 1, InterferenceKind::SpatialConflict);
        graph.add_edge(0, 2, InterferenceKind::SpatialConflict);
        graph.add_edge(0, 3, InterferenceKind::SpatialConflict);
        assert_eq!(graph.max_degree(), 3);
    }
}
