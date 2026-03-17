//! Unreachable state and dead-transition detection.
//!
//! Uses BFS from the initial state to identify states that can never be
//! entered, transitions that can never fire, and guards that are statically
//! unsatisfiable.

use choreo_automata::automaton::{SpatialEventAutomaton, Transition};
use choreo_automata::{Guard, SpatialPredicate, StateId, TransitionId};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

// ---------------------------------------------------------------------------
// State classification
// ---------------------------------------------------------------------------

/// Classification of a state's role in the automaton.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StateClass {
    /// The state participates in at least one path from initial to accepting.
    Productive,
    /// The state is reachable but cannot reach any accepting state.
    Unproductive,
    /// The state is not reachable from the initial state.
    Dead,
    /// The state is reachable but has no outgoing transitions (and is not accepting).
    Trap,
}

impl fmt::Display for StateClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Productive => write!(f, "productive"),
            Self::Unproductive => write!(f, "unproductive"),
            Self::Dead => write!(f, "dead"),
            Self::Trap => write!(f, "trap"),
        }
    }
}

// ---------------------------------------------------------------------------
// UnreachableAnalyzer
// ---------------------------------------------------------------------------

/// Analyses an automaton for unreachable states, dead transitions, and dead guards.
#[derive(Debug)]
pub struct UnreachableAnalyzer {
    /// Extra "scene constraints" for guard feasibility checks.
    /// Each constraint is a predicate name that is known to always be false.
    permanently_false: HashSet<String>,
}

impl UnreachableAnalyzer {
    pub fn new() -> Self {
        Self {
            permanently_false: HashSet::new(),
        }
    }

    /// Register a spatial predicate name as permanently false.
    pub fn add_permanently_false(&mut self, pred_name: impl Into<String>) {
        self.permanently_false.insert(pred_name.into());
    }

    /// Run all analyses on the automaton and return a full report.
    pub fn analyze(&self, automaton: &SpatialEventAutomaton) -> UnreachableReport {
        let reachable = self.find_reachable(automaton);
        let all_states: HashSet<u32> = automaton.state_ids().into_iter().map(|s| s.0).collect();
        let unreachable: Vec<u32> = all_states.difference(&reachable).copied().collect();

        let transitions: Vec<&Transition> = automaton.transitions.values().collect();
        let dead_transitions = self.find_dead_transitions(&transitions, &reachable);
        let dead_guards = self.find_dead_guards(&transitions);
        let classification = self.classify_states(automaton);

        UnreachableReport {
            unreachable_states: unreachable,
            dead_transitions,
            dead_guard_transitions: dead_guards,
            state_classifications: classification,
        }
    }

    /// BFS from the initial state; returns set of reachable state ids.
    pub fn find_reachable(&self, automaton: &SpatialEventAutomaton) -> HashSet<u32> {
        let init = match automaton.initial_state {
            Some(s) => s,
            None => return HashSet::new(),
        };
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        visited.insert(init.0);
        queue.push_back(init);

        while let Some(s) = queue.pop_front() {
            for t in automaton.transitions.values() {
                if t.source == s && !visited.contains(&t.target.0) {
                    visited.insert(t.target.0);
                    queue.push_back(t.target);
                }
            }
        }
        visited
    }

    /// Convenience: returns unreachable state ids (complement of reachable).
    pub fn find_unreachable_states(
        &self,
        initial: StateId,
        transitions: &[&Transition],
        all_state_ids: &[u32],
    ) -> Vec<u32> {
        let reachable = bfs_from(initial.0, transitions);
        let all: HashSet<u32> = all_state_ids.iter().copied().collect();
        let mut unreachable: Vec<u32> = all.difference(&reachable).copied().collect();
        unreachable.sort();
        unreachable
    }

    /// Find transitions whose source state is unreachable.
    pub fn find_dead_transitions(
        &self,
        transitions: &[&Transition],
        reachable: &HashSet<u32>,
    ) -> Vec<u32> {
        let mut dead = Vec::new();
        for t in transitions {
            if !reachable.contains(&t.source.0) {
                dead.push(t.id.0);
            }
        }
        dead.sort();
        dead
    }

    /// Find transitions whose guards can never be true.
    ///
    /// A guard is "dead" if:
    /// - It is `Guard::False`.
    /// - It references a spatial predicate that is in the `permanently_false` set.
    /// - It is an `And` containing a dead sub-guard.
    pub fn find_dead_guards(&self, transitions: &[&Transition]) -> Vec<u32> {
        let mut dead = Vec::new();
        for t in transitions {
            if self.is_guard_dead(&t.guard) {
                dead.push(t.id.0);
            }
        }
        dead.sort();
        dead
    }

    /// Classify every state in the automaton.
    pub fn classify_states(
        &self,
        automaton: &SpatialEventAutomaton,
    ) -> HashMap<u32, StateClass> {
        let reachable = automaton.reachable_states();
        let reachable_ids: HashSet<u32> = reachable.iter().map(|s| s.0).collect();
        let reaching_accept = automaton.states_reaching_accepting();
        let reaching_ids: HashSet<u32> = reaching_accept.iter().map(|s| s.0).collect();

        let mut outgoing_count: HashMap<u32, usize> = HashMap::new();
        for sid in automaton.state_ids() {
            outgoing_count.insert(sid.0, 0);
        }
        for t in automaton.transitions.values() {
            *outgoing_count.entry(t.source.0).or_default() += 1;
        }

        let mut classification = HashMap::new();
        for sid in automaton.state_ids() {
            let s = sid.0;
            let is_reachable = reachable_ids.contains(&s);
            let can_reach_accept = reaching_ids.contains(&s);
            let is_accepting = automaton.accepting_states.contains(&sid);
            let has_outgoing = outgoing_count.get(&s).copied().unwrap_or(0) > 0;

            let class = if !is_reachable {
                StateClass::Dead
            } else if !has_outgoing && !is_accepting {
                StateClass::Trap
            } else if !can_reach_accept && !is_accepting {
                StateClass::Unproductive
            } else {
                StateClass::Productive
            };
            classification.insert(s, class);
        }
        classification
    }

    // -----------------------------------------------------------------------
    // Guard analysis
    // -----------------------------------------------------------------------

    fn is_guard_dead(&self, guard: &Guard) -> bool {
        match guard {
            Guard::False => true,
            Guard::True => false,
            Guard::Spatial(sp) => self.is_spatial_dead(sp),
            Guard::And(gs) => gs.iter().any(|g| self.is_guard_dead(g)),
            Guard::Or(gs) => gs.iter().all(|g| self.is_guard_dead(g)),
            Guard::Not(g) => self.is_guard_always_true(g),
            Guard::Event(_) | Guard::Temporal(_) => false,
        }
    }

    fn is_guard_always_true(&self, guard: &Guard) -> bool {
        match guard {
            Guard::True => true,
            Guard::False => false,
            Guard::And(gs) => gs.iter().all(|g| self.is_guard_always_true(g)),
            Guard::Or(gs) => gs.iter().any(|g| self.is_guard_always_true(g)),
            Guard::Not(g) => self.is_guard_dead(g),
            _ => false,
        }
    }

    fn is_spatial_dead(&self, sp: &SpatialPredicate) -> bool {
        match sp {
            SpatialPredicate::Named(id) => self.permanently_false.contains(&id.0),
            SpatialPredicate::And(preds) => preds.iter().any(|p| self.is_spatial_dead(p)),
            SpatialPredicate::Or(preds) => preds.iter().all(|p| self.is_spatial_dead(p)),
            SpatialPredicate::Not(inner) => self.is_spatial_always_true(inner),
            _ => false,
        }
    }

    fn is_spatial_always_true(&self, sp: &SpatialPredicate) -> bool {
        match sp {
            SpatialPredicate::Not(inner) => self.is_spatial_dead(inner),
            SpatialPredicate::And(preds) => preds.iter().all(|p| self.is_spatial_always_true(p)),
            SpatialPredicate::Or(preds) => preds.iter().any(|p| self.is_spatial_always_true(p)),
            _ => false,
        }
    }
}

impl Default for UnreachableAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// BFS utility (raw transition list)
// ---------------------------------------------------------------------------

fn bfs_from(start: u32, transitions: &[&Transition]) -> HashSet<u32> {
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    visited.insert(start);
    queue.push_back(start);

    let mut adj: HashMap<u32, Vec<u32>> = HashMap::new();
    for t in transitions {
        adj.entry(t.source.0).or_default().push(t.target.0);
    }

    while let Some(s) = queue.pop_front() {
        if let Some(nexts) = adj.get(&s) {
            for &n in nexts {
                if visited.insert(n) {
                    queue.push_back(n);
                }
            }
        }
    }
    visited
}

// ---------------------------------------------------------------------------
// UnreachableReport
// ---------------------------------------------------------------------------

/// Complete report from unreachable analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnreachableReport {
    pub unreachable_states: Vec<u32>,
    pub dead_transitions: Vec<u32>,
    pub dead_guard_transitions: Vec<u32>,
    pub state_classifications: HashMap<u32, StateClass>,
}

impl UnreachableReport {
    pub fn total_issues(&self) -> usize {
        self.unreachable_states.len()
            + self.dead_transitions.len()
            + self.dead_guard_transitions.len()
    }

    pub fn productive_states(&self) -> Vec<u32> {
        self.state_classifications
            .iter()
            .filter(|(_, c)| **c == StateClass::Productive)
            .map(|(&s, _)| s)
            .collect()
    }

    pub fn dead_states(&self) -> Vec<u32> {
        self.state_classifications
            .iter()
            .filter(|(_, c)| **c == StateClass::Dead)
            .map(|(&s, _)| s)
            .collect()
    }

    pub fn trap_states(&self) -> Vec<u32> {
        self.state_classifications
            .iter()
            .filter(|(_, c)| **c == StateClass::Trap)
            .map(|(&s, _)| s)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use choreo_automata::automaton::{State, Transition};
    use choreo_automata::{EventKind, Guard, StateId, TransitionId};

    fn make_automaton(
        n_states: u32,
        edges: &[(u32, u32, Guard)],
        initial: u32,
        accepting: &[u32],
    ) -> SpatialEventAutomaton {
        let mut aut = SpatialEventAutomaton::new("test_unreach");
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
    fn all_reachable_linear() {
        let aut = make_automaton(
            3,
            &[
                (0, 1, Guard::Event(EventKind::GrabStart)),
                (1, 2, Guard::Event(EventKind::GrabEnd)),
            ],
            0,
            &[2],
        );
        let analyzer = UnreachableAnalyzer::new();
        let reachable = analyzer.find_reachable(&aut);
        assert_eq!(reachable, [0, 1, 2].iter().copied().collect());
    }

    #[test]
    fn unreachable_island() {
        // s3 has no incoming edges
        let aut = make_automaton(
            4,
            &[
                (0, 1, Guard::Event(EventKind::GrabStart)),
                (1, 2, Guard::Event(EventKind::GrabEnd)),
            ],
            0,
            &[2],
        );
        let analyzer = UnreachableAnalyzer::new();
        let report = analyzer.analyze(&aut);
        assert!(report.unreachable_states.contains(&3));
    }

    #[test]
    fn dead_transition_from_unreachable() {
        // Transition from s3→s2 but s3 is unreachable
        let aut = make_automaton(
            4,
            &[
                (0, 1, Guard::Event(EventKind::GrabStart)),
                (1, 2, Guard::Event(EventKind::GrabEnd)),
                (3, 2, Guard::Event(EventKind::TouchStart)),
            ],
            0,
            &[2],
        );
        let analyzer = UnreachableAnalyzer::new();
        let report = analyzer.analyze(&aut);
        assert!(report.dead_transitions.contains(&2)); // transition idx 2
    }

    #[test]
    fn dead_guard_false() {
        let aut = make_automaton(
            3,
            &[
                (0, 1, Guard::Event(EventKind::GrabStart)),
                (1, 2, Guard::False),
            ],
            0,
            &[2],
        );
        let analyzer = UnreachableAnalyzer::new();
        let report = analyzer.analyze(&aut);
        assert!(report.dead_guard_transitions.contains(&1));
    }

    #[test]
    fn dead_guard_named_predicate() {
        let sp = SpatialPredicate::Named(choreo_automata::SpatialPredicateId(
            "impossible_pred".into(),
        ));
        let aut = make_automaton(
            3,
            &[
                (0, 1, Guard::Event(EventKind::GrabStart)),
                (1, 2, Guard::Spatial(sp)),
            ],
            0,
            &[2],
        );
        let mut analyzer = UnreachableAnalyzer::new();
        analyzer.add_permanently_false("impossible_pred");
        let report = analyzer.analyze(&aut);
        assert!(report.dead_guard_transitions.contains(&1));
    }

    #[test]
    fn classify_productive() {
        let aut = make_automaton(
            3,
            &[
                (0, 1, Guard::Event(EventKind::GrabStart)),
                (1, 2, Guard::Event(EventKind::GrabEnd)),
            ],
            0,
            &[2],
        );
        let analyzer = UnreachableAnalyzer::new();
        let classes = analyzer.classify_states(&aut);
        assert_eq!(classes[&0], StateClass::Productive);
        assert_eq!(classes[&1], StateClass::Productive);
        assert_eq!(classes[&2], StateClass::Productive);
    }

    #[test]
    fn classify_trap() {
        // s2 has no outgoing and is not accepting
        let aut = make_automaton(
            3,
            &[
                (0, 1, Guard::Event(EventKind::GrabStart)),
                (1, 2, Guard::Event(EventKind::GrabEnd)),
            ],
            0,
            &[],
        );
        let analyzer = UnreachableAnalyzer::new();
        let classes = analyzer.classify_states(&aut);
        assert_eq!(classes[&2], StateClass::Trap);
    }

    #[test]
    fn classify_unproductive() {
        // s1 → s2 → s1 cycle, but no accepting state reachable from them
        let aut = make_automaton(
            4,
            &[
                (0, 1, Guard::Event(EventKind::GrabStart)),
                (1, 2, Guard::Event(EventKind::GrabEnd)),
                (2, 1, Guard::Event(EventKind::TouchStart)),
            ],
            0,
            &[3], // s3 is accepting but unreachable
        );
        let analyzer = UnreachableAnalyzer::new();
        let classes = analyzer.classify_states(&aut);
        assert_eq!(classes[&1], StateClass::Unproductive);
        assert_eq!(classes[&2], StateClass::Unproductive);
        assert_eq!(classes[&3], StateClass::Dead);
    }

    #[test]
    fn classify_dead() {
        let aut = make_automaton(
            4,
            &[
                (0, 1, Guard::Event(EventKind::GrabStart)),
            ],
            0,
            &[1],
        );
        let analyzer = UnreachableAnalyzer::new();
        let classes = analyzer.classify_states(&aut);
        assert_eq!(classes[&2], StateClass::Dead);
        assert_eq!(classes[&3], StateClass::Dead);
    }

    #[test]
    fn find_unreachable_states_raw() {
        let transitions = vec![
            Transition::new(TransitionId(0), StateId(0), StateId(1), Guard::True, vec![]),
            Transition::new(TransitionId(1), StateId(1), StateId(2), Guard::True, vec![]),
        ];
        let trefs: Vec<&Transition> = transitions.iter().collect();
        let analyzer = UnreachableAnalyzer::new();
        let unreachable =
            analyzer.find_unreachable_states(StateId(0), &trefs, &[0, 1, 2, 3, 4]);
        assert_eq!(unreachable, vec![3, 4]);
    }

    #[test]
    fn report_total_issues() {
        let aut = make_automaton(
            5,
            &[
                (0, 1, Guard::Event(EventKind::GrabStart)),
                (1, 2, Guard::False),
            ],
            0,
            &[2],
        );
        let analyzer = UnreachableAnalyzer::new();
        let report = analyzer.analyze(&aut);
        assert!(report.total_issues() > 0);
    }
}
