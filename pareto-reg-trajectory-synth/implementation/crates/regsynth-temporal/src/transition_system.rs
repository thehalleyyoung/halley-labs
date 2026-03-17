use crate::{RegulatoryEvent, RegulatoryState, StateId, TemporalError};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};

/// A labelled transition between two regulatory states.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transition {
    pub from: StateId,
    pub to: StateId,
    pub event: RegulatoryEvent,
}

/// Labels that summarise the *kind* of regulatory event on an edge.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TransitionLabel {
    AmendmentAdopted,
    PhaseInMilestone,
    Sunset,
    NewFramework,
    Repeal,
    GracePeriod,
}

impl TransitionLabel {
    /// Derive a label from a concrete `RegulatoryEvent`.
    pub fn from_event(event: &RegulatoryEvent) -> Self {
        match event {
            RegulatoryEvent::Amendment { .. } => Self::AmendmentAdopted,
            RegulatoryEvent::PhaseIn { .. } => Self::PhaseInMilestone,
            RegulatoryEvent::Sunset { .. } => Self::Sunset,
            RegulatoryEvent::Repeal { .. } => Self::Repeal,
            RegulatoryEvent::GracePeriodStart { .. }
            | RegulatoryEvent::GracePeriodEnd { .. } => Self::GracePeriod,
        }
    }
}

/// A finite-state transition system whose states are sets of active
/// regulatory obligations and whose edges are labelled by regulatory events.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegulatoryTransitionSystem {
    pub states: HashMap<StateId, RegulatoryState>,
    pub transitions: Vec<Transition>,
    pub initial_state: Option<StateId>,
}

impl RegulatoryTransitionSystem {
    pub fn new() -> Self {
        Self {
            states: HashMap::new(),
            transitions: Vec::new(),
            initial_state: None,
        }
    }

    /// Add a state.  The first state added becomes the initial state.
    pub fn add_state(&mut self, state: RegulatoryState) {
        if self.initial_state.is_none() {
            self.initial_state = Some(state.id.clone());
        }
        self.states.insert(state.id.clone(), state);
    }

    /// Set the initial state explicitly (must already be in the system).
    pub fn set_initial_state(&mut self, id: &str) -> Result<(), TemporalError> {
        if !self.states.contains_key(id) {
            return Err(TemporalError::StateNotFound(id.to_string()));
        }
        self.initial_state = Some(id.to_string());
        Ok(())
    }

    /// Add a transition, validating that both endpoints exist.
    pub fn add_transition(&mut self, t: Transition) -> Result<(), TemporalError> {
        if !self.states.contains_key(&t.from) {
            return Err(TemporalError::StateNotFound(t.from.clone()));
        }
        if !self.states.contains_key(&t.to) {
            return Err(TemporalError::StateNotFound(t.to.clone()));
        }
        self.transitions.push(t);
        Ok(())
    }

    /// Add a transition without validation (for bulk construction).
    pub fn add_transition_unchecked(&mut self, t: Transition) {
        self.transitions.push(t);
    }

    /// Return all transitions departing from `state_id`.
    pub fn successors(&self, state_id: &str) -> Vec<&Transition> {
        self.transitions
            .iter()
            .filter(|t| t.from == state_id)
            .collect()
    }

    /// Return all transitions arriving at `state_id`.
    pub fn predecessors(&self, state_id: &str) -> Vec<&Transition> {
        self.transitions
            .iter()
            .filter(|t| t.to == state_id)
            .collect()
    }

    /// Return the set of active obligation IDs for a given state.
    pub fn active_obligations_at(
        &self,
        state_id: &str,
    ) -> Result<&BTreeSet<String>, TemporalError> {
        self.states
            .get(state_id)
            .map(|s| &s.obligations)
            .ok_or_else(|| TemporalError::StateNotFound(state_id.to_string()))
    }

    /// BFS from the initial state (or a given start) to compute all
    /// reachable state IDs.
    pub fn compute_reachable_states(&self) -> HashSet<StateId> {
        self.reachable_from(self.initial_state.as_deref().unwrap_or(""))
    }

    /// BFS reachability from an arbitrary state.
    pub fn reachable_from(&self, start: &str) -> HashSet<StateId> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        if self.states.contains_key(start) {
            queue.push_back(start.to_string());
            visited.insert(start.to_string());
        }
        while let Some(cur) = queue.pop_front() {
            for t in self.successors(&cur) {
                if visited.insert(t.to.clone()) {
                    queue.push_back(t.to.clone());
                }
            }
        }
        visited
    }

    /// Find a shortest path (sequence of transitions) from `start` to `goal`
    /// using BFS. Returns `None` if no path exists.
    pub fn find_path(&self, start: &str, goal: &str) -> Option<Vec<Transition>> {
        if start == goal {
            return Some(Vec::new());
        }
        if !self.states.contains_key(start) || !self.states.contains_key(goal) {
            return None;
        }

        // BFS storing parent edge for reconstruction
        let mut visited: HashSet<String> = HashSet::new();
        let mut parent: HashMap<String, (String, usize)> = HashMap::new();
        let mut queue = VecDeque::new();
        visited.insert(start.to_string());
        queue.push_back(start.to_string());

        while let Some(cur) = queue.pop_front() {
            for (idx, t) in self.transitions.iter().enumerate() {
                if t.from == cur && visited.insert(t.to.clone()) {
                    parent.insert(t.to.clone(), (cur.clone(), idx));
                    if t.to == goal {
                        // Reconstruct
                        let mut path = Vec::new();
                        let mut node = goal.to_string();
                        while let Some((prev, tidx)) = parent.get(&node) {
                            path.push(self.transitions[*tidx].clone());
                            node = prev.clone();
                        }
                        path.reverse();
                        return Some(path);
                    }
                    queue.push_back(t.to.clone());
                }
            }
        }
        None
    }

    /// Collect all distinct `TransitionLabel` kinds that appear in the system.
    pub fn transition_labels(&self) -> HashSet<TransitionLabel> {
        self.transitions
            .iter()
            .map(|t| TransitionLabel::from_event(&t.event))
            .collect()
    }

    /// Count the number of strongly-connected components (Kosaraju's algorithm).
    pub fn strongly_connected_components(&self) -> Vec<Vec<StateId>> {
        let ids: Vec<&str> = self.states.keys().map(|s| s.as_str()).collect();
        let idx_of: HashMap<&str, usize> = ids.iter().enumerate().map(|(i, s)| (*s, i)).collect();
        let n = ids.len();

        // Build adjacency lists
        let mut adj = vec![Vec::new(); n];
        let mut radj = vec![Vec::new(); n];
        for t in &self.transitions {
            if let (Some(&u), Some(&v)) = (idx_of.get(t.from.as_str()), idx_of.get(t.to.as_str()))
            {
                adj[u].push(v);
                radj[v].push(u);
            }
        }

        // Pass 1: DFS order
        let mut visited = vec![false; n];
        let mut order = Vec::with_capacity(n);
        for i in 0..n {
            if !visited[i] {
                let mut stack = vec![(i, 0usize)];
                visited[i] = true;
                while let Some((node, ei)) = stack.last_mut() {
                    if *ei < adj[*node].len() {
                        let next = adj[*node][*ei];
                        *ei += 1;
                        if !visited[next] {
                            visited[next] = true;
                            stack.push((next, 0));
                        }
                    } else {
                        order.push(*node);
                        stack.pop();
                    }
                }
            }
        }

        // Pass 2: reverse DFS in reverse order
        let mut comp_id = vec![usize::MAX; n];
        let mut components: Vec<Vec<StateId>> = Vec::new();
        for &node in order.iter().rev() {
            if comp_id[node] != usize::MAX {
                continue;
            }
            let cid = components.len();
            let mut comp = Vec::new();
            let mut stack = vec![node];
            comp_id[node] = cid;
            while let Some(u) = stack.pop() {
                comp.push(ids[u].to_string());
                for &v in &radj[u] {
                    if comp_id[v] == usize::MAX {
                        comp_id[v] = cid;
                        stack.push(v);
                    }
                }
            }
            components.push(comp);
        }
        components
    }

    /// Return the set of states that have no outgoing transitions (sinks).
    pub fn sink_states(&self) -> Vec<StateId> {
        let has_outgoing: HashSet<&str> = self.transitions.iter().map(|t| t.from.as_str()).collect();
        self.states
            .keys()
            .filter(|s| !has_outgoing.contains(s.as_str()))
            .cloned()
            .collect()
    }

    /// Return the number of states.
    pub fn state_count(&self) -> usize {
        self.states.len()
    }

    /// Return the number of transitions.
    pub fn transition_count(&self) -> usize {
        self.transitions.len()
    }

    /// Check whether the system is deterministic: for every state and
    /// transition-label pair there is at most one successor.
    pub fn is_deterministic(&self) -> bool {
        let mut seen: HashMap<(&str, TransitionLabel), usize> = HashMap::new();
        for t in &self.transitions {
            let key = (t.from.as_str(), TransitionLabel::from_event(&t.event));
            let count = seen.entry(key).or_insert(0);
            *count += 1;
            if *count > 1 {
                return false;
            }
        }
        true
    }
}

impl Default for RegulatoryTransitionSystem {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ymd;

    fn sample_system() -> RegulatoryTransitionSystem {
        let mut sys = RegulatoryTransitionSystem::new();
        let mut s0 = RegulatoryState::new("s0");
        s0.add_obligation("obl-A".into());
        let mut s1 = RegulatoryState::new("s1");
        s1.add_obligation("obl-A".into());
        s1.add_obligation("obl-B".into());
        let mut s2 = RegulatoryState::new("s2");
        s2.add_obligation("obl-B".into());
        s2.add_obligation("obl-C".into());
        let s3 = RegulatoryState::new("s3");

        sys.add_state(s0);
        sys.add_state(s1);
        sys.add_state(s2);
        sys.add_state(s3);

        sys.add_transition_unchecked(Transition {
            from: "s0".into(),
            to: "s1".into(),
            event: RegulatoryEvent::PhaseIn {
                milestone: "M1".into(),
                date: ymd(2025, 2, 2),
            },
        });
        sys.add_transition_unchecked(Transition {
            from: "s1".into(),
            to: "s2".into(),
            event: RegulatoryEvent::Amendment {
                description: "Amendment 1".into(),
                date: ymd(2025, 8, 1),
            },
        });
        sys.add_transition_unchecked(Transition {
            from: "s2".into(),
            to: "s3".into(),
            event: RegulatoryEvent::Sunset {
                description: "Sunset".into(),
                date: ymd(2027, 1, 1),
            },
        });
        sys
    }

    #[test]
    fn test_initial_state() {
        let sys = sample_system();
        assert_eq!(sys.initial_state.as_deref(), Some("s0"));
    }

    #[test]
    fn test_successors_and_predecessors() {
        let sys = sample_system();
        assert_eq!(sys.successors("s0").len(), 1);
        assert_eq!(sys.successors("s1").len(), 1);
        assert!(sys.successors("s3").is_empty());
        assert_eq!(sys.predecessors("s1").len(), 1);
    }

    #[test]
    fn test_active_obligations() {
        let sys = sample_system();
        let obls = sys.active_obligations_at("s1").unwrap();
        assert!(obls.contains("obl-A"));
        assert!(obls.contains("obl-B"));
        assert!(!obls.contains("obl-C"));
    }

    #[test]
    fn test_reachable_states() {
        let sys = sample_system();
        let reachable = sys.compute_reachable_states();
        assert_eq!(reachable.len(), 4);
        assert!(reachable.contains("s0"));
        assert!(reachable.contains("s3"));
    }

    #[test]
    fn test_find_path() {
        let sys = sample_system();
        let path = sys.find_path("s0", "s3").unwrap();
        assert_eq!(path.len(), 3);
        assert_eq!(path[0].from, "s0");
        assert_eq!(path[2].to, "s3");
    }

    #[test]
    fn test_find_path_no_route() {
        let sys = sample_system();
        assert!(sys.find_path("s3", "s0").is_none());
    }

    #[test]
    fn test_find_path_same_node() {
        let sys = sample_system();
        let path = sys.find_path("s1", "s1").unwrap();
        assert!(path.is_empty());
    }

    #[test]
    fn test_sink_states() {
        let sys = sample_system();
        let sinks = sys.sink_states();
        assert_eq!(sinks.len(), 1);
        assert_eq!(sinks[0], "s3");
    }

    #[test]
    fn test_scc_linear() {
        let sys = sample_system();
        let sccs = sys.strongly_connected_components();
        // Linear chain → each state is its own SCC
        assert_eq!(sccs.len(), 4);
    }

    #[test]
    fn test_scc_with_cycle() {
        let mut sys = RegulatoryTransitionSystem::new();
        sys.add_state(RegulatoryState::new("a"));
        sys.add_state(RegulatoryState::new("b"));
        sys.add_transition_unchecked(Transition {
            from: "a".into(),
            to: "b".into(),
            event: RegulatoryEvent::PhaseIn {
                milestone: "M".into(),
                date: ymd(2025, 1, 1),
            },
        });
        sys.add_transition_unchecked(Transition {
            from: "b".into(),
            to: "a".into(),
            event: RegulatoryEvent::Amendment {
                description: "loop".into(),
                date: ymd(2025, 2, 1),
            },
        });
        let sccs = sys.strongly_connected_components();
        let big: Vec<_> = sccs.iter().filter(|c| c.len() == 2).collect();
        assert_eq!(big.len(), 1);
    }

    #[test]
    fn test_transition_labels() {
        let sys = sample_system();
        let labels = sys.transition_labels();
        assert!(labels.contains(&TransitionLabel::PhaseInMilestone));
        assert!(labels.contains(&TransitionLabel::AmendmentAdopted));
        assert!(labels.contains(&TransitionLabel::Sunset));
    }

    #[test]
    fn test_is_deterministic() {
        let sys = sample_system();
        assert!(sys.is_deterministic());
    }

    #[test]
    fn test_add_transition_validation() {
        let mut sys = RegulatoryTransitionSystem::new();
        sys.add_state(RegulatoryState::new("s0"));
        let result = sys.add_transition(Transition {
            from: "s0".into(),
            to: "s_missing".into(),
            event: RegulatoryEvent::PhaseIn {
                milestone: "M".into(),
                date: ymd(2025, 1, 1),
            },
        });
        assert!(result.is_err());
    }

    #[test]
    fn test_set_initial_state() {
        let mut sys = sample_system();
        assert!(sys.set_initial_state("s2").is_ok());
        assert_eq!(sys.initial_state.as_deref(), Some("s2"));
        assert!(sys.set_initial_state("nonexistent").is_err());
    }

    #[test]
    fn test_counts() {
        let sys = sample_system();
        assert_eq!(sys.state_count(), 4);
        assert_eq!(sys.transition_count(), 3);
    }
}
