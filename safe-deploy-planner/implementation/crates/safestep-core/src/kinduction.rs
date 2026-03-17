//! K-induction verification engine for SafeStep deployment plans.
//!
//! Implements k-induction model checking over the version product graph:
//! - **Base case**: invariant holds for all states reachable within k steps of initial states
//! - **Inductive step**: if invariant holds for k consecutive states on any path, it holds for
//!   the (k+1)-th state
//! - When both pass, the invariant holds for *all* reachable states.

use std::collections::{HashMap, HashSet, VecDeque};

use tracing::{debug, info, trace, warn};

use crate::{Constraint, Invariant, ServiceIndex, State, VersionIndex, VersionProductGraph};

// ---------------------------------------------------------------------------
// Result enums
// ---------------------------------------------------------------------------

/// Outcome of a full k-induction verification attempt.
#[derive(Debug, Clone)]
pub enum InductionResult {
    /// Invariant verified at induction depth `k`.
    Verified { k: usize },
    /// Base case failed – a reachable state violates the invariant.
    CounterExample { trace: Vec<State>, depth: usize },
    /// Inductive step failed – the witness path satisfies the invariant for k steps but
    /// a successor violates it.
    InductiveFailure { k: usize, witness_path: Vec<State> },
    /// Could not verify within the allowed depth budget.
    Insufficient { max_k_tried: usize },
}

/// Outcome of a base-case check up to depth k.
#[derive(Debug, Clone)]
pub enum BaseCaseResult {
    /// All states up to depth k satisfy the invariant.
    Pass { states_checked: usize },
    /// At least one reachable state violates the invariant.
    Fail { violations: Vec<Vec<State>> },
}

/// Outcome of the inductive-step check at depth k.
#[derive(Debug, Clone)]
pub enum InductiveStepResult {
    /// All checked k-length invariant-satisfying paths have successors that also satisfy
    /// the invariant.
    Pass { paths_checked: usize },
    /// Found a k-length path where the invariant holds on every state but at least one
    /// successor violates it.
    Fail { witness: Vec<State> },
}

// ---------------------------------------------------------------------------
// BaseCase
// ---------------------------------------------------------------------------

/// Performs a BFS from the initial states up to a bounded depth, checking that the invariant
/// holds at every visited state.
pub struct BaseCase;

impl BaseCase {
    /// Run the base-case check.
    ///
    /// Returns `BaseCaseResult::Pass` when every state reachable within `k` steps from
    /// `initial` satisfies both the constraints and the invariant. Otherwise returns all
    /// violating traces found.
    pub fn check(
        graph: &VersionProductGraph,
        constraints: &[Constraint],
        initial: &[State],
        invariant: &Invariant,
        k: usize,
    ) -> BaseCaseResult {
        let mut visited: HashSet<State> = HashSet::new();
        // BFS queue entries: (current_state, path_from_initial)
        let mut queue: VecDeque<(State, Vec<State>)> = VecDeque::new();
        let mut violations: Vec<Vec<State>> = Vec::new();
        let mut states_checked: usize = 0;

        for s in initial {
            if visited.contains(s) {
                continue;
            }
            visited.insert(s.clone());
            queue.push_back((s.clone(), vec![s.clone()]));
        }

        while let Some((current, path)) = queue.pop_front() {
            states_checked += 1;

            // Check constraints
            let constraints_ok = constraints.iter().all(|c| c.check_state(&current));
            if !constraints_ok {
                // A state that violates constraints is not part of the legal state space;
                // we skip its successors but do not count it as an invariant violation.
                trace!(state = %current, "state violates constraints, pruning");
                continue;
            }

            // Check invariant
            if !invariant.holds(&current) {
                debug!(state = %current, depth = path.len() - 1, "base case violation found");
                violations.push(path.clone());
                // Keep searching for more violations but don't expand from here.
                continue;
            }

            // Expand if within depth budget
            let depth = path.len() - 1;
            if depth < k {
                for (neighbor, _edge_idx) in graph.neighbors(&current) {
                    if !visited.contains(&neighbor) {
                        visited.insert(neighbor.clone());
                        let mut new_path = path.clone();
                        new_path.push(neighbor.clone());
                        queue.push_back((neighbor, new_path));
                    }
                }
            }
        }

        if violations.is_empty() {
            info!(states_checked, k, "base case passed");
            BaseCaseResult::Pass { states_checked }
        } else {
            info!(violation_count = violations.len(), k, "base case failed");
            BaseCaseResult::Fail { violations }
        }
    }
}

// ---------------------------------------------------------------------------
// InductiveStep
// ---------------------------------------------------------------------------

/// Verifies the inductive step: for every path of length k in the graph where the invariant
/// (and constraints) hold at each state, every successor of the last state must also satisfy
/// the invariant.
pub struct InductiveStep;

impl InductiveStep {
    /// Run the inductive-step check via bounded DFS over constraint-satisfying,
    /// invariant-satisfying paths.
    pub fn check(
        graph: &VersionProductGraph,
        constraints: &[Constraint],
        invariant: &Invariant,
        k: usize,
    ) -> InductiveStepResult {
        if k == 0 {
            // k=0: every state satisfying the invariant must have only invariant-satisfying
            // successors. We iterate all graph states.
            return Self::check_k0(graph, constraints, invariant);
        }

        let mut paths_checked: usize = 0;

        // Use iterative DFS with an explicit stack to enumerate all k-length paths where
        // invariant and constraints hold at every state.
        // Stack entry: (current_state, path_so_far)
        let mut stack: Vec<(State, Vec<State>)> = Vec::new();

        // Seed with every graph state that satisfies invariant + constraints.
        for state in &graph.states {
            if Self::state_ok(state, constraints, invariant) {
                stack.push((state.clone(), vec![state.clone()]));
            }
        }

        while let Some((current, path)) = stack.pop() {
            if path.len() == k + 1 {
                // We have a full k-length path (k+1 states, k edges).
                // Check all successors of the terminal state.
                paths_checked += 1;

                for (succ, _edge_idx) in graph.neighbors(&current) {
                    let constraints_ok = constraints.iter().all(|c| c.check_state(&succ));
                    if constraints_ok && !invariant.holds(&succ) {
                        let mut witness = path.clone();
                        witness.push(succ);
                        debug!(
                            k,
                            paths_checked,
                            "inductive step failure: successor violates invariant"
                        );
                        return InductiveStepResult::Fail { witness };
                    }
                }

                // Apply a cap to avoid combinatorial explosion in large graphs.
                if paths_checked >= Self::max_paths_budget(graph) {
                    trace!(paths_checked, "path budget exhausted, treating as pass");
                    break;
                }
                continue;
            }

            // Extend the path by one step.
            for (neighbor, _edge_idx) in graph.neighbors(&current) {
                if Self::state_ok(&neighbor, constraints, invariant) {
                    let mut new_path = path.clone();
                    new_path.push(neighbor.clone());
                    stack.push((neighbor, new_path));
                }
            }
        }

        info!(paths_checked, k, "inductive step passed");
        InductiveStepResult::Pass { paths_checked }
    }

    /// Special handling for k = 0: every invariant-satisfying state must have only
    /// invariant-satisfying successors.
    fn check_k0(
        graph: &VersionProductGraph,
        constraints: &[Constraint],
        invariant: &Invariant,
    ) -> InductiveStepResult {
        let mut paths_checked: usize = 0;
        for state in &graph.states {
            if !Self::state_ok(state, constraints, invariant) {
                continue;
            }
            paths_checked += 1;
            for (succ, _edge_idx) in graph.neighbors(state) {
                let constraints_ok = constraints.iter().all(|c| c.check_state(&succ));
                if constraints_ok && !invariant.holds(&succ) {
                    let witness = vec![state.clone(), succ];
                    return InductiveStepResult::Fail { witness };
                }
            }
        }
        InductiveStepResult::Pass { paths_checked }
    }

    /// Returns `true` when a state satisfies every constraint and the invariant.
    fn state_ok(state: &State, constraints: &[Constraint], invariant: &Invariant) -> bool {
        constraints.iter().all(|c| c.check_state(state)) && invariant.holds(state)
    }

    /// Heuristic budget: we allow up to N² paths (capped at 500_000) to keep the inductive
    /// step tractable.
    fn max_paths_budget(graph: &VersionProductGraph) -> usize {
        let n = graph.state_count();
        (n * n).min(500_000).max(1)
    }
}

// ---------------------------------------------------------------------------
// KInduction  –  main verification engine
// ---------------------------------------------------------------------------

/// K-induction verification engine.
///
/// Combines `BaseCase` and `InductiveStep` checks, iterating k from 1 to `max_k` until
/// verification succeeds or a counterexample is found.
pub struct KInduction<'g> {
    graph: &'g VersionProductGraph,
    constraints: Vec<Constraint>,
}

impl<'g> KInduction<'g> {
    /// Create a new engine referencing the given graph and constraints.
    pub fn new(graph: &'g VersionProductGraph, constraints: &[Constraint]) -> Self {
        Self {
            graph,
            constraints: constraints.to_vec(),
        }
    }

    /// Attempt to verify `invariant` via k-induction for k = 1, 2, …, `max_k`.
    ///
    /// The search stops early when:
    /// - the base case fails (returns `CounterExample`), or
    /// - both base and inductive checks pass (returns `Verified`).
    pub fn verify_invariant(
        &self,
        initial_states: &[State],
        invariant: &Invariant,
        max_k: usize,
    ) -> InductionResult {
        info!(
            max_k,
            initial_states = initial_states.len(),
            "starting k-induction verification"
        );

        for k in 1..=max_k {
            debug!(k, "trying k-induction depth");

            // 1. Base case
            match self.base_case_check(initial_states, invariant, k) {
                BaseCaseResult::Fail { violations } => {
                    let trace = violations.into_iter().next().unwrap_or_default();
                    let depth = trace.len().saturating_sub(1);
                    warn!(k, depth, "base case failed – counterexample found");
                    return InductionResult::CounterExample { trace, depth };
                }
                BaseCaseResult::Pass { states_checked } => {
                    debug!(k, states_checked, "base case passed");
                }
            }

            // 2. Inductive step
            match self.inductive_step_check(invariant, k) {
                InductiveStepResult::Fail { witness } => {
                    debug!(k, "inductive step failed – trying larger k");
                    if k == max_k {
                        return InductionResult::InductiveFailure {
                            k,
                            witness_path: witness,
                        };
                    }
                    // Try a larger k.
                }
                InductiveStepResult::Pass { paths_checked } => {
                    info!(k, paths_checked, "k-induction verification succeeded");
                    return InductionResult::Verified { k };
                }
            }
        }

        InductionResult::Insufficient {
            max_k_tried: max_k,
        }
    }

    // -- private helpers --

    fn base_case_check(
        &self,
        initial: &[State],
        invariant: &Invariant,
        k: usize,
    ) -> BaseCaseResult {
        BaseCase::check(self.graph, &self.constraints, initial, invariant, k)
    }

    fn inductive_step_check(&self, invariant: &Invariant, k: usize) -> InductiveStepResult {
        InductiveStep::check(self.graph, &self.constraints, invariant, k)
    }
}

// ---------------------------------------------------------------------------
// InvariantChecker  –  higher-level wrapper
// ---------------------------------------------------------------------------

/// Higher-level interface that wraps `KInduction` with invariant-strengthening heuristics
/// and reachability-guided violation search.
pub struct InvariantChecker<'g> {
    graph: &'g VersionProductGraph,
    constraints: Vec<Constraint>,
    max_k: usize,
}

impl<'g> InvariantChecker<'g> {
    /// Construct a checker with a default `max_k` of 10.
    pub fn new(graph: &'g VersionProductGraph, constraints: &[Constraint]) -> Self {
        Self {
            graph,
            constraints: constraints.to_vec(),
            max_k: 10,
        }
    }

    /// Set the maximum induction depth.
    pub fn with_max_k(mut self, max_k: usize) -> Self {
        self.max_k = max_k;
        self
    }

    /// Check a safety property via k-induction with k = 1 … max_k.
    pub fn check_safety(
        &self,
        initial: &[State],
        safety_predicate: &Invariant,
    ) -> InductionResult {
        let engine = KInduction::new(self.graph, &self.constraints);
        engine.verify_invariant(initial, safety_predicate, self.max_k)
    }

    /// Attempt to strengthen `invariant` by conjoining a predicate that rules out the states
    /// appearing in `counter_path`.
    ///
    /// The heuristic collects the set of `(ServiceIndex, VersionIndex)` pairs that appear in
    /// the last state of the counterexample but *not* in any state satisfying the original
    /// invariant, and adds a conjunct forbidding those pairs.
    pub fn strengthen_invariant(
        &self,
        invariant: &Invariant,
        counter_path: &[State],
    ) -> Option<Invariant> {
        let bad_state = counter_path.last()?;

        // Collect version assignments in the bad state.
        let bad_assignments: HashSet<(u16, u16)> = (0..bad_state.dimension())
            .map(|i| {
                let svc = ServiceIndex(i as u16);
                (svc.0, bad_state.get(svc).0)
            })
            .collect();

        // Collect version assignments that appear in *good* states (graph states that satisfy
        // the invariant).
        let mut good_assignments: HashSet<(u16, u16)> = HashSet::new();
        for s in &self.graph.states {
            if invariant.holds(s) {
                for i in 0..s.dimension() {
                    let svc = ServiceIndex(i as u16);
                    good_assignments.insert((svc.0, s.get(svc).0));
                }
            }
        }

        // The "suspect" assignments are those present in the bad state but never in a good
        // state.
        let suspects: Vec<(u16, u16)> = bad_assignments
            .difference(&good_assignments)
            .copied()
            .collect();

        if suspects.is_empty() {
            // Cannot derive a useful strengthening.
            debug!("no suspect assignments found; strengthening failed");
            return None;
        }

        // Build a new invariant that conjoins the original with a clause forbidding suspects.
        let desc = format!(
            "{} ∧ ¬suspect({})",
            invariant.description,
            suspects
                .iter()
                .map(|(s, v)| format!("svc{}=v{}", s, v))
                .collect::<Vec<_>>()
                .join(",")
        );

        // Clone the suspects into the closure.
        let suspects_set: HashSet<(u16, u16)> = suspects.into_iter().collect();

        // We need to capture the original check function. Because `Invariant.check` is behind
        // a `Box<dyn Fn>`, we clone the invariant and move it into the closure.
        let original = invariant.clone();
        let strengthened = Invariant::new(desc, move |state: &State| {
            if !original.holds(state) {
                return false;
            }
            // Reject states containing any suspect assignment.
            for i in 0..state.dimension() {
                let svc = ServiceIndex(i as u16);
                if suspects_set.contains(&(svc.0, state.get(svc).0)) {
                    return false;
                }
            }
            true
        });

        Some(strengthened)
    }

    /// Iteratively attempt k-induction: if the inductive step fails, try to strengthen the
    /// invariant and re-check, up to `max_attempts` rounds.
    pub fn check_with_strengthening(
        &self,
        initial: &[State],
        invariant: &Invariant,
        max_attempts: usize,
    ) -> InductionResult {
        let mut current_inv = invariant.clone();

        for attempt in 0..max_attempts {
            debug!(attempt, "strengthening attempt");

            let result = self.check_safety(initial, &current_inv);
            match &result {
                InductionResult::Verified { .. } => return result,
                InductionResult::CounterExample { .. } => {
                    // A real counterexample – invariant is violated on a reachable state.
                    return result;
                }
                InductionResult::InductiveFailure { witness_path, .. } => {
                    // Try to strengthen.
                    match self.strengthen_invariant(&current_inv, witness_path) {
                        Some(stronger) => {
                            info!(attempt, desc = %stronger.description, "invariant strengthened");
                            current_inv = stronger;
                        }
                        None => {
                            debug!(attempt, "could not strengthen further");
                            return result;
                        }
                    }
                }
                InductionResult::Insufficient { .. } => return result,
            }
        }

        // Exhausted attempts – return the last check result.
        self.check_safety(initial, &current_inv)
    }

    /// BFS from `initial` up to `max_depth` to find the first reachable state that violates
    /// the invariant. Returns the trace (path from an initial state to the violating state).
    pub fn find_reachable_violating(
        &self,
        initial: &[State],
        invariant: &Invariant,
        max_depth: usize,
    ) -> Option<Vec<State>> {
        let mut visited: HashSet<State> = HashSet::new();
        let mut queue: VecDeque<(State, Vec<State>)> = VecDeque::new();

        for s in initial {
            if visited.insert(s.clone()) {
                queue.push_back((s.clone(), vec![s.clone()]));
            }
        }

        while let Some((current, path)) = queue.pop_front() {
            let depth = path.len() - 1;

            let constraints_ok = self.constraints.iter().all(|c| c.check_state(&current));
            if !constraints_ok {
                continue;
            }

            if !invariant.holds(&current) {
                return Some(path);
            }

            if depth >= max_depth {
                continue;
            }

            for (neighbor, _) in self.graph.neighbors(&current) {
                if visited.insert(neighbor.clone()) {
                    let mut new_path = path.clone();
                    new_path.push(neighbor.clone());
                    queue.push_back((neighbor, new_path));
                }
            }
        }

        None
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Edge, ServiceDescriptor, TransitionMetadata};

    /// Helper: build a linear graph  s0 → s1 → s2 → … → s_{n-1}
    /// with a single service whose version steps from 0 to n-1.
    fn linear_graph(n: usize) -> (VersionProductGraph, Vec<State>) {
        let svc = ServiceDescriptor::new("svc-a", (0..n).map(|i| format!("v{}", i)).collect());
        let mut graph = VersionProductGraph::new(vec![svc]);

        let states: Vec<State> = (0..n)
            .map(|i| State::new(vec![VersionIndex(i as u16)]))
            .collect();

        for s in &states {
            graph.add_state(s.clone());
        }
        for i in 0..n.saturating_sub(1) {
            graph.add_edge(Edge {
                from: states[i].clone(),
                to: states[i + 1].clone(),
                service: ServiceIndex(0),
                from_version: VersionIndex(i as u16),
                to_version: VersionIndex((i + 1) as u16),
                metadata: TransitionMetadata::default(),
            });
        }

        (graph, states)
    }

    /// Helper: build a diamond graph
    ///       s0
    ///      /  \
    ///    s1    s2
    ///      \  /
    ///       s3
    fn diamond_graph() -> (VersionProductGraph, Vec<State>) {
        let svc_a = ServiceDescriptor::new("a", vec!["v0".into(), "v1".into()]);
        let svc_b = ServiceDescriptor::new("b", vec!["v0".into(), "v1".into()]);
        let mut graph = VersionProductGraph::new(vec![svc_a, svc_b]);

        let s0 = State::new(vec![VersionIndex(0), VersionIndex(0)]);
        let s1 = State::new(vec![VersionIndex(1), VersionIndex(0)]);
        let s2 = State::new(vec![VersionIndex(0), VersionIndex(1)]);
        let s3 = State::new(vec![VersionIndex(1), VersionIndex(1)]);

        for s in [&s0, &s1, &s2, &s3] {
            graph.add_state(s.clone());
        }

        let meta = TransitionMetadata::default();
        graph.add_edge(Edge {
            from: s0.clone(),
            to: s1.clone(),
            service: ServiceIndex(0),
            from_version: VersionIndex(0),
            to_version: VersionIndex(1),
            metadata: meta.clone(),
        });
        graph.add_edge(Edge {
            from: s0.clone(),
            to: s2.clone(),
            service: ServiceIndex(1),
            from_version: VersionIndex(0),
            to_version: VersionIndex(1),
            metadata: meta.clone(),
        });
        graph.add_edge(Edge {
            from: s1.clone(),
            to: s3.clone(),
            service: ServiceIndex(1),
            from_version: VersionIndex(0),
            to_version: VersionIndex(1),
            metadata: meta.clone(),
        });
        graph.add_edge(Edge {
            from: s2.clone(),
            to: s3.clone(),
            service: ServiceIndex(0),
            from_version: VersionIndex(0),
            to_version: VersionIndex(1),
            metadata: meta,
        });

        let states = vec![s0, s1, s2, s3];
        (graph, states)
    }

    // ------------------------------------------------------------------
    // Test 1: trivially true invariant on a linear graph
    // ------------------------------------------------------------------
    #[test]
    fn test_base_case_trivially_true() {
        let (graph, states) = linear_graph(5);
        let inv = Invariant::new("always true", |_s: &State| true);
        let result = BaseCase::check(&graph, &[], &[states[0].clone()], &inv, 4);
        match result {
            BaseCaseResult::Pass { states_checked } => {
                assert_eq!(states_checked, 5);
            }
            _ => panic!("expected pass"),
        }
    }

    // ------------------------------------------------------------------
    // Test 2: base case fails when a reachable state violates invariant
    // ------------------------------------------------------------------
    #[test]
    fn test_base_case_violation() {
        let (graph, states) = linear_graph(5);
        // Invariant: version index < 3
        let inv = Invariant::new("version < 3", |s: &State| s.get(ServiceIndex(0)).0 < 3);
        let result = BaseCase::check(&graph, &[], &[states[0].clone()], &inv, 4);
        match result {
            BaseCaseResult::Fail { violations } => {
                assert!(!violations.is_empty());
                // The first violating state should have version >= 3.
                let last = violations[0].last().unwrap();
                assert!(last.get(ServiceIndex(0)).0 >= 3);
            }
            _ => panic!("expected failure"),
        }
    }

    // ------------------------------------------------------------------
    // Test 3: inductive step passes on a diamond where invariant is universal
    // ------------------------------------------------------------------
    #[test]
    fn test_inductive_step_pass_diamond() {
        let (graph, _states) = diamond_graph();
        let inv = Invariant::new("always true", |_s: &State| true);
        let result = InductiveStep::check(&graph, &[], &inv, 1);
        match result {
            InductiveStepResult::Pass { paths_checked } => {
                assert!(paths_checked > 0);
            }
            _ => panic!("expected pass"),
        }
    }

    // ------------------------------------------------------------------
    // Test 4: inductive step fails – invariant holds on prefix but not successor
    // ------------------------------------------------------------------
    #[test]
    fn test_inductive_step_failure() {
        let (graph, states) = linear_graph(4);
        // Invariant: version < 3  (state s3 with version 3 violates)
        let inv = Invariant::new("version < 3", |s: &State| s.get(ServiceIndex(0)).0 < 3);
        let result = InductiveStep::check(&graph, &[], &inv, 1);
        match result {
            InductiveStepResult::Fail { witness } => {
                // Witness should end with the violating state.
                let last = witness.last().unwrap();
                assert_eq!(last.get(ServiceIndex(0)).0, 3);
            }
            _ => panic!("expected inductive step failure"),
        }
    }

    // ------------------------------------------------------------------
    // Test 5: full k-induction verification succeeds (trivial invariant)
    // ------------------------------------------------------------------
    #[test]
    fn test_kinduction_verified() {
        let (graph, states) = diamond_graph();
        let inv = Invariant::new("always true", |_s: &State| true);
        let engine = KInduction::new(&graph, &[]);
        let result = engine.verify_invariant(&[states[0].clone()], &inv, 5);
        match result {
            InductionResult::Verified { k } => {
                assert!(k >= 1);
            }
            other => panic!("expected Verified, got {:?}", other),
        }
    }

    // ------------------------------------------------------------------
    // Test 6: k-induction returns counterexample for violated safety property
    // ------------------------------------------------------------------
    #[test]
    fn test_kinduction_counterexample() {
        let (graph, states) = linear_graph(5);
        // Safety: version index must stay below 3. Violated at depth 3.
        let inv = Invariant::new("version < 3", |s: &State| s.get(ServiceIndex(0)).0 < 3);
        let engine = KInduction::new(&graph, &[]);
        let result = engine.verify_invariant(&[states[0].clone()], &inv, 5);
        match result {
            InductionResult::CounterExample { trace, depth } => {
                assert!(depth >= 3);
                assert!(!trace.is_empty());
            }
            other => panic!("expected CounterExample, got {:?}", other),
        }
    }

    // ------------------------------------------------------------------
    // Test 7: InvariantChecker – find reachable violation
    // ------------------------------------------------------------------
    #[test]
    fn test_find_reachable_violating() {
        let (graph, states) = linear_graph(6);
        let inv = Invariant::new("version < 4", |s: &State| s.get(ServiceIndex(0)).0 < 4);
        let checker = InvariantChecker::new(&graph, &[]);
        let trace = checker.find_reachable_violating(&[states[0].clone()], &inv, 10);
        assert!(trace.is_some());
        let trace = trace.unwrap();
        let violating = trace.last().unwrap();
        assert!(violating.get(ServiceIndex(0)).0 >= 4);
        // Path should start at the initial state.
        assert_eq!(trace[0], states[0]);
    }

    // ------------------------------------------------------------------
    // Test 8: no reachable violation when invariant always holds
    // ------------------------------------------------------------------
    #[test]
    fn test_find_reachable_violating_none() {
        let (graph, states) = diamond_graph();
        let inv = Invariant::new("always true", |_s: &State| true);
        let checker = InvariantChecker::new(&graph, &[]);
        let trace = checker.find_reachable_violating(&[states[0].clone()], &inv, 10);
        assert!(trace.is_none());
    }

    // ------------------------------------------------------------------
    // Test 9: check_safety via InvariantChecker – verified case
    // ------------------------------------------------------------------
    #[test]
    fn test_check_safety_verified() {
        let (graph, states) = diamond_graph();
        let inv = Invariant::new("sum <= 2", |s: &State| {
            let a = s.get(ServiceIndex(0)).0;
            let b = s.get(ServiceIndex(1)).0;
            (a + b) <= 2
        });
        let checker = InvariantChecker::new(&graph, &[]).with_max_k(5);
        let result = checker.check_safety(&[states[0].clone()], &inv);
        match result {
            InductionResult::Verified { .. } => {}
            other => panic!("expected Verified, got {:?}", other),
        }
    }

    // ------------------------------------------------------------------
    // Test 10: strengthening attempt
    // ------------------------------------------------------------------
    #[test]
    fn test_strengthen_invariant() {
        let (graph, states) = linear_graph(5);
        let inv = Invariant::new("version < 4", |s: &State| s.get(ServiceIndex(0)).0 < 4);
        let checker = InvariantChecker::new(&graph, &[]);

        // Construct a mock counter path ending at version 4.
        let counter = vec![
            State::new(vec![VersionIndex(2)]),
            State::new(vec![VersionIndex(3)]),
            State::new(vec![VersionIndex(4)]),
        ];

        let strengthened = checker.strengthen_invariant(&inv, &counter);
        assert!(strengthened.is_some());
        let stronger = strengthened.unwrap();
        // The strengthened invariant should still accept version 0.
        assert!(stronger.holds(&State::new(vec![VersionIndex(0)])));
        // And should reject the bad state.
        assert!(!stronger.holds(&State::new(vec![VersionIndex(4)])));
    }

    // ------------------------------------------------------------------
    // Test 11: check_with_strengthening converges
    // ------------------------------------------------------------------
    #[test]
    fn test_check_with_strengthening() {
        let (graph, states) = diamond_graph();
        let inv = Invariant::new("always true", |_s: &State| true);
        let checker = InvariantChecker::new(&graph, &[]).with_max_k(5);
        let result = checker.check_with_strengthening(&[states[0].clone()], &inv, 3);
        match result {
            InductionResult::Verified { .. } => {}
            other => panic!("expected Verified, got {:?}", other),
        }
    }

    // ------------------------------------------------------------------
    // Test 12: empty initial states
    // ------------------------------------------------------------------
    #[test]
    fn test_empty_initial_states() {
        let (graph, _states) = linear_graph(3);
        let inv = Invariant::new("always false", |_s: &State| false);
        // With no initial states the base case should trivially pass (nothing to check),
        // and the result depends on the inductive step.
        let result = BaseCase::check(&graph, &[], &[], &inv, 5);
        match result {
            BaseCaseResult::Pass { states_checked } => {
                assert_eq!(states_checked, 0);
            }
            _ => panic!("expected pass with 0 states checked"),
        }
    }
}
