//! Abstract model checking: BFS/DFS explicit state, symbolic BDD, SAT-based BMC.
//!
//! Provides multiple model-checking backends for verifying properties over
//! abstract models produced by the CEGAR loop.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::fmt;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::abstraction::{
    AbstractBlock, AbstractBlockId, AbstractState, AbstractTransition, AbstractTransitionRelation,
    AbstractionState, SpatialPartition,
};
use crate::counterexample::Counterexample;
use crate::properties::Property;
use crate::{
    AutomatonDef, CegarError, Guard, PredicateValuation, SpatialConstraint, SpatialPredicateId,
    StateId, TransitionId, AABB,
};

// ---------------------------------------------------------------------------
// ModelChecker
// ---------------------------------------------------------------------------

/// The abstract model checker.
#[derive(Debug, Clone)]
pub struct ModelChecker {
    /// Maximum number of states to explore before giving up.
    pub max_states: usize,
    /// Whether to use symbolic exploration when possible.
    pub use_symbolic: bool,
    /// BMC bound (0 = disabled).
    pub bmc_bound: usize,
}

impl ModelChecker {
    pub fn new() -> Self {
        Self {
            max_states: 1_000_000,
            use_symbolic: false,
            bmc_bound: 0,
        }
    }

    pub fn with_limits(max_states: usize, bmc_bound: usize) -> Self {
        Self {
            max_states,
            use_symbolic: false,
            bmc_bound,
        }
    }

    /// Check reachability: can any target state be reached from initial states?
    pub fn check_reachability(
        &self,
        model: &AbstractionState,
        target_states: &HashSet<AbstractState>,
    ) -> ReachabilityResult {
        let adj = model.transition_relation.adjacency_list();
        let mut visited: HashSet<AbstractState> = HashSet::new();
        let mut parent: HashMap<AbstractState, AbstractState> = HashMap::new();
        let mut queue = VecDeque::new();

        for init in &model.initial_states {
            if target_states.contains(init) {
                return ReachabilityResult::Reachable {
                    trace: vec![*init],
                    steps: 0,
                };
            }
            visited.insert(*init);
            queue.push_back(*init);
        }

        let mut explored = 0usize;

        while let Some(current) = queue.pop_front() {
            explored += 1;
            if explored > self.max_states {
                return ReachabilityResult::Unknown {
                    explored_states: explored,
                    reason: "State limit reached".to_string(),
                };
            }

            if let Some(succs) = adj.get(&current) {
                for &next in succs {
                    if !visited.contains(&next) {
                        visited.insert(next);
                        parent.insert(next, current);

                        if target_states.contains(&next) {
                            let trace = reconstruct_path(&parent, &model.initial_states, next);
                            return ReachabilityResult::Reachable {
                                steps: trace.len() - 1,
                                trace,
                            };
                        }

                        queue.push_back(next);
                    }
                }
            }
        }

        ReachabilityResult::Unreachable {
            explored_states: explored,
        }
    }

    /// Check deadlock freedom: no reachable non-accepting state has zero outgoing transitions.
    pub fn check_deadlock_freedom(&self, model: &AbstractionState) -> DeadlockResult {
        let adj = model.transition_relation.adjacency_list();
        let mut visited: HashSet<AbstractState> = HashSet::new();
        let mut parent: HashMap<AbstractState, AbstractState> = HashMap::new();
        let mut queue = VecDeque::new();

        for init in &model.initial_states {
            visited.insert(*init);
            queue.push_back(*init);
        }

        let mut explored = 0usize;

        while let Some(current) = queue.pop_front() {
            explored += 1;
            if explored > self.max_states {
                return DeadlockResult::Unknown {
                    explored_states: explored,
                };
            }

            // Check for deadlock: no outgoing transitions and not accepting
            let has_successors = adj.get(&current).map_or(false, |s| !s.is_empty());
            if !has_successors && !model.automaton.is_accepting(current.automaton_state) {
                let trace = reconstruct_path(&parent, &model.initial_states, current);
                return DeadlockResult::DeadlockFound {
                    deadlock_state: current,
                    trace,
                };
            }

            if let Some(succs) = adj.get(&current) {
                for &next in succs {
                    if !visited.contains(&next) {
                        visited.insert(next);
                        parent.insert(next, current);
                        queue.push_back(next);
                    }
                }
            }
        }

        DeadlockResult::DeadlockFree {
            explored_states: explored,
        }
    }

    /// Check safety: no bad state is reachable.
    pub fn check_safety(
        &self,
        model: &AbstractionState,
        bad_states: &HashSet<AbstractState>,
    ) -> SafetyResult {
        let reach = self.check_reachability(model, bad_states);
        match reach {
            ReachabilityResult::Reachable { trace, steps } => SafetyResult::Unsafe {
                counterexample: trace,
                steps,
            },
            ReachabilityResult::Unreachable { explored_states } => SafetyResult::Safe {
                explored_states,
            },
            ReachabilityResult::Unknown {
                explored_states,
                reason,
            } => SafetyResult::Unknown {
                explored_states,
                reason,
            },
        }
    }

    /// Check liveness: every reachable cycle passes through a progress state.
    pub fn check_liveness(
        &self,
        model: &AbstractionState,
        progress_states: &HashSet<AbstractState>,
    ) -> LivenessResult {
        let adj = model.transition_relation.adjacency_list();

        // First, find all reachable states
        let reachable = self.compute_reachable_states(model);

        // Find SCCs using Tarjan's algorithm, then check each SCC
        let sccs = tarjan_scc(&reachable, &adj);

        for scc in &sccs {
            if scc.len() < 2 {
                // Single-state SCC: check for self-loop
                let state = scc[0];
                let has_self_loop = adj
                    .get(&state)
                    .map_or(false, |succs| succs.contains(&state));
                if !has_self_loop {
                    continue;
                }
            }

            // Check if SCC contains a progress state
            let has_progress = scc.iter().any(|s| progress_states.contains(s));
            if !has_progress {
                // Non-progress cycle found
                let trace = scc.clone();
                return LivenessResult::LivenessViolation {
                    cycle: trace,
                    cycle_length: scc.len(),
                };
            }
        }

        LivenessResult::Live {
            explored_states: reachable.len(),
            scc_count: sccs.len(),
        }
    }

    /// Compute all reachable states from initial states.
    fn compute_reachable_states(&self, model: &AbstractionState) -> HashSet<AbstractState> {
        let adj = model.transition_relation.adjacency_list();
        let mut visited: HashSet<AbstractState> = HashSet::new();
        let mut queue = VecDeque::new();

        for init in &model.initial_states {
            visited.insert(*init);
            queue.push_back(*init);
        }

        while let Some(current) = queue.pop_front() {
            if visited.len() > self.max_states {
                break;
            }
            if let Some(succs) = adj.get(&current) {
                for &next in succs {
                    if !visited.contains(&next) {
                        visited.insert(next);
                        queue.push_back(next);
                    }
                }
            }
        }

        visited
    }

    /// Symbolic reachability using BDD-based exploration.
    pub fn symbolic_reachability(
        &self,
        relation: &BDDTransitionRelation,
        initial: &BDDSet,
        target: &BDDSet,
    ) -> BDDResult {
        let mut reached = initial.clone();
        let mut frontier = initial.clone();
        let mut steps = 0u32;

        loop {
            // Compute image: states reachable in one step from frontier
            let image = relation.image(&frontier);
            let new_states = image.difference(&reached);

            if new_states.is_empty() {
                // Fixed point reached
                let intersection = reached.intersection(target);
                if intersection.is_empty() {
                    return BDDResult::Unreachable { steps };
                } else {
                    return BDDResult::Reachable { steps };
                }
            }

            // Check if we've reached the target
            let target_reached = new_states.intersection(target);
            if !target_reached.is_empty() {
                return BDDResult::Reachable { steps: steps + 1 };
            }

            reached = reached.union(&new_states);
            frontier = new_states;
            steps += 1;

            if steps > 10000 {
                return BDDResult::Unknown {
                    reason: "Step limit".to_string(),
                };
            }
        }
    }

    /// Symbolic bounded model checking.
    pub fn symbolic_bmc(
        &self,
        relation: &BDDTransitionRelation,
        property: &BDDSet,
        bound: u32,
    ) -> BMCResult {
        let initial = BDDSet::new(relation.variables.clone());
        let mut current = initial;

        for k in 0..bound {
            let image = relation.image(&current);
            let violation = image.intersection(property);
            if !violation.is_empty() {
                return BMCResult::CounterexampleFound { bound: k + 1 };
            }
            current = image;
        }

        BMCResult::NoBugFound { bound }
    }

    /// BMC using SAT encoding.
    pub fn bmc_check(
        &self,
        model: &AbstractionState,
        bad_states: &HashSet<AbstractState>,
        bound: usize,
    ) -> BMCResult {
        // Encode the bounded unrolling as a SAT formula
        let state_list: Vec<AbstractState> = model.abstract_states.clone();
        let state_to_idx: HashMap<AbstractState, usize> = state_list
            .iter()
            .enumerate()
            .map(|(i, s)| (*s, i))
            .collect();
        let n = state_list.len();

        if n == 0 {
            return BMCResult::NoBugFound { bound: bound as u32 };
        }

        // Build SAT formula
        let mut formula = SATFormula::new();

        // Variables: state_var(time_step, state_idx) = time_step * n + state_idx + 1
        let state_var = |t: usize, s: usize| -> i32 { (t * n + s + 1) as i32 };

        // Initial state constraint: at time 0, we must be in an initial state
        let mut initial_clause: Vec<i32> = Vec::new();
        for init in &model.initial_states {
            if let Some(&idx) = state_to_idx.get(init) {
                initial_clause.push(state_var(0, idx));
            }
        }
        if !initial_clause.is_empty() {
            formula.add_clause(initial_clause);
        }

        // At most one state per time step (pairwise exclusion)
        for t in 0..=bound {
            for i in 0..n {
                for j in (i + 1)..n {
                    formula.add_clause(vec![-state_var(t, i), -state_var(t, j)]);
                }
            }
            // At least one state per time step
            let at_least_one: Vec<i32> = (0..n).map(|i| state_var(t, i)).collect();
            formula.add_clause(at_least_one);
        }

        // Transition constraints: if in state s at time t, must be in successor at t+1
        let adj = model.transition_relation.adjacency_list();
        for t in 0..bound {
            for (i, state) in state_list.iter().enumerate() {
                let succs: Vec<usize> = adj
                    .get(state)
                    .map(|s| {
                        s.iter()
                            .filter_map(|next| state_to_idx.get(next).copied())
                            .collect()
                    })
                    .unwrap_or_default();

                if succs.is_empty() {
                    // If no successors, this state can't appear except at the last step
                    // Actually, it can appear at any step but the trace ends there
                    continue;
                }

                // If state i at time t, then some successor at time t+1
                let mut clause = vec![-state_var(t, i)];
                for &s in &succs {
                    clause.push(state_var(t + 1, s));
                }
                formula.add_clause(clause);
            }
        }

        // Bad state constraint: at some time step, we're in a bad state
        let mut bad_clause: Vec<i32> = Vec::new();
        for t in 0..=bound {
            for bad in bad_states {
                if let Some(&idx) = state_to_idx.get(bad) {
                    bad_clause.push(state_var(t, idx));
                }
            }
        }
        if bad_clause.is_empty() {
            return BMCResult::NoBugFound { bound: bound as u32 };
        }
        formula.add_clause(bad_clause);

        // Solve the SAT formula
        let solver = SATSolver::new();
        match solver.solve(&formula) {
            SATResult::Satisfiable(_assignment) => BMCResult::CounterexampleFound {
                bound: bound as u32,
            },
            SATResult::Unsatisfiable => BMCResult::NoBugFound { bound: bound as u32 },
            SATResult::Unknown => BMCResult::Unknown {
                reason: "SAT solver returned unknown".to_string(),
            },
        }
    }
}

impl Default for ModelChecker {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------

/// Result of a reachability check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReachabilityResult {
    Reachable {
        trace: Vec<AbstractState>,
        steps: usize,
    },
    Unreachable {
        explored_states: usize,
    },
    Unknown {
        explored_states: usize,
        reason: String,
    },
}

impl ReachabilityResult {
    pub fn is_reachable(&self) -> bool {
        matches!(self, ReachabilityResult::Reachable { .. })
    }
}

/// Result of a deadlock check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeadlockResult {
    DeadlockFree {
        explored_states: usize,
    },
    DeadlockFound {
        deadlock_state: AbstractState,
        trace: Vec<AbstractState>,
    },
    Unknown {
        explored_states: usize,
    },
}

impl DeadlockResult {
    pub fn is_deadlock_free(&self) -> bool {
        matches!(self, DeadlockResult::DeadlockFree { .. })
    }
}

/// Result of a safety check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SafetyResult {
    Safe {
        explored_states: usize,
    },
    Unsafe {
        counterexample: Vec<AbstractState>,
        steps: usize,
    },
    Unknown {
        explored_states: usize,
        reason: String,
    },
}

impl SafetyResult {
    pub fn is_safe(&self) -> bool {
        matches!(self, SafetyResult::Safe { .. })
    }
}

/// Result of a liveness check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LivenessResult {
    Live {
        explored_states: usize,
        scc_count: usize,
    },
    LivenessViolation {
        cycle: Vec<AbstractState>,
        cycle_length: usize,
    },
}

impl LivenessResult {
    pub fn is_live(&self) -> bool {
        matches!(self, LivenessResult::Live { .. })
    }
}

/// Result of BDD-based symbolic checking.
#[derive(Debug, Clone)]
pub enum BDDResult {
    Reachable { steps: u32 },
    Unreachable { steps: u32 },
    Unknown { reason: String },
}

/// Result of bounded model checking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BMCResult {
    CounterexampleFound { bound: u32 },
    NoBugFound { bound: u32 },
    Unknown { reason: String },
}

// ---------------------------------------------------------------------------
// Tarjan's SCC algorithm
// ---------------------------------------------------------------------------

/// Compute strongly connected components using Tarjan's algorithm.
fn tarjan_scc(
    states: &HashSet<AbstractState>,
    adj: &HashMap<AbstractState, Vec<AbstractState>>,
) -> Vec<Vec<AbstractState>> {
    struct TarjanState {
        index: usize,
        lowlink: usize,
        on_stack: bool,
    }

    let mut index_counter = 0usize;
    let mut stack: Vec<AbstractState> = Vec::new();
    let mut state_info: HashMap<AbstractState, TarjanState> = HashMap::new();
    let mut result: Vec<Vec<AbstractState>> = Vec::new();

    fn strongconnect(
        v: AbstractState,
        adj: &HashMap<AbstractState, Vec<AbstractState>>,
        index_counter: &mut usize,
        stack: &mut Vec<AbstractState>,
        state_info: &mut HashMap<AbstractState, TarjanState>,
        result: &mut Vec<Vec<AbstractState>>,
    ) {
        state_info.insert(
            v,
            TarjanState {
                index: *index_counter,
                lowlink: *index_counter,
                on_stack: true,
            },
        );
        *index_counter += 1;
        stack.push(v);

        if let Some(succs) = adj.get(&v) {
            for &w in succs {
                if !state_info.contains_key(&w) {
                    strongconnect(w, adj, index_counter, stack, state_info, result);
                    let w_lowlink = state_info[&w].lowlink;
                    let v_info = state_info.get_mut(&v).unwrap();
                    v_info.lowlink = v_info.lowlink.min(w_lowlink);
                } else if state_info[&w].on_stack {
                    let w_index = state_info[&w].index;
                    let v_info = state_info.get_mut(&v).unwrap();
                    v_info.lowlink = v_info.lowlink.min(w_index);
                }
            }
        }

        let v_info = &state_info[&v];
        if v_info.lowlink == v_info.index {
            let mut scc = Vec::new();
            loop {
                let w = stack.pop().unwrap();
                state_info.get_mut(&w).unwrap().on_stack = false;
                scc.push(w);
                if w == v {
                    break;
                }
            }
            result.push(scc);
        }
    }

    for &state in states {
        if !state_info.contains_key(&state) {
            strongconnect(
                state,
                adj,
                &mut index_counter,
                &mut stack,
                &mut state_info,
                &mut result,
            );
        }
    }

    result
}

// ---------------------------------------------------------------------------
// BDD-based symbolic exploration
// ---------------------------------------------------------------------------

/// A simplified BDD set representation using explicit state sets.
/// In a production system this would use a proper BDD library.
#[derive(Debug, Clone)]
pub struct BDDSet {
    pub elements: HashSet<u64>,
    pub variables: Vec<String>,
}

impl BDDSet {
    pub fn new(variables: Vec<String>) -> Self {
        Self {
            elements: HashSet::new(),
            variables,
        }
    }

    pub fn from_elements(elements: HashSet<u64>, variables: Vec<String>) -> Self {
        Self {
            elements,
            variables,
        }
    }

    pub fn insert(&mut self, elem: u64) {
        self.elements.insert(elem);
    }

    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    pub fn len(&self) -> usize {
        self.elements.len()
    }

    pub fn contains(&self, elem: u64) -> bool {
        self.elements.contains(&elem)
    }

    pub fn union(&self, other: &BDDSet) -> BDDSet {
        let combined: HashSet<u64> = self.elements.union(&other.elements).copied().collect();
        BDDSet::from_elements(combined, self.variables.clone())
    }

    pub fn intersection(&self, other: &BDDSet) -> BDDSet {
        let common: HashSet<u64> = self
            .elements
            .intersection(&other.elements)
            .copied()
            .collect();
        BDDSet::from_elements(common, self.variables.clone())
    }

    pub fn difference(&self, other: &BDDSet) -> BDDSet {
        let diff: HashSet<u64> = self.elements.difference(&other.elements).copied().collect();
        BDDSet::from_elements(diff, self.variables.clone())
    }

    pub fn complement(&self, universe: &BDDSet) -> BDDSet {
        universe.difference(self)
    }
}

/// A simplified BDD transition relation.
#[derive(Debug, Clone)]
pub struct BDDTransitionRelation {
    pub transitions: HashMap<u64, Vec<u64>>,
    pub variables: Vec<String>,
}

impl BDDTransitionRelation {
    pub fn new(variables: Vec<String>) -> Self {
        Self {
            transitions: HashMap::new(),
            variables,
        }
    }

    pub fn add_transition(&mut self, from: u64, to: u64) {
        self.transitions.entry(from).or_default().push(to);
    }

    /// Compute the image: set of states reachable in one step from the given set.
    pub fn image(&self, set: &BDDSet) -> BDDSet {
        let mut result = HashSet::new();
        for &elem in &set.elements {
            if let Some(succs) = self.transitions.get(&elem) {
                result.extend(succs);
            }
        }
        BDDSet::from_elements(result, self.variables.clone())
    }

    /// Compute the preimage: set of states that can reach the given set in one step.
    pub fn preimage(&self, set: &BDDSet) -> BDDSet {
        let mut result = HashSet::new();
        for (&from, tos) in &self.transitions {
            for &to in tos {
                if set.elements.contains(&to) {
                    result.insert(from);
                }
            }
        }
        BDDSet::from_elements(result, self.variables.clone())
    }
}

/// Build a BDD transition relation from an abstract model.
pub fn build_bdd_relation(model: &AbstractionState) -> BDDTransitionRelation {
    let state_list: Vec<AbstractState> = model.abstract_states.clone();
    let state_to_idx: HashMap<AbstractState, u64> = state_list
        .iter()
        .enumerate()
        .map(|(i, s)| (*s, i as u64))
        .collect();

    let variables: Vec<String> = state_list.iter().map(|s| format!("{}", s)).collect();
    let mut relation = BDDTransitionRelation::new(variables);

    for t in &model.transition_relation.transitions {
        if let (Some(&from), Some(&to)) = (state_to_idx.get(&t.source), state_to_idx.get(&t.target))
        {
            relation.add_transition(from, to);
        }
    }

    relation
}

// ---------------------------------------------------------------------------
// SAT solver (basic DPLL with CDCL)
// ---------------------------------------------------------------------------

/// A SAT formula in CNF (conjunctive normal form).
#[derive(Debug, Clone)]
pub struct SATFormula {
    pub clauses: Vec<Vec<i32>>,
    pub num_variables: usize,
}

impl SATFormula {
    pub fn new() -> Self {
        Self {
            clauses: Vec::new(),
            num_variables: 0,
        }
    }

    pub fn add_clause(&mut self, clause: Vec<i32>) {
        for &lit in &clause {
            let var = lit.unsigned_abs() as usize;
            self.num_variables = self.num_variables.max(var);
        }
        self.clauses.push(clause);
    }

    pub fn clause_count(&self) -> usize {
        self.clauses.len()
    }

    pub fn is_empty(&self) -> bool {
        self.clauses.is_empty()
    }
}

impl Default for SATFormula {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of SAT solving.
#[derive(Debug, Clone)]
pub enum SATResult {
    /// Formula is satisfiable with the given assignment.
    Satisfiable(Vec<bool>),
    /// Formula is unsatisfiable.
    Unsatisfiable,
    /// Solver could not determine satisfiability.
    Unknown,
}

/// A basic DPLL SAT solver with unit propagation and CDCL.
pub struct SATSolver {
    max_conflicts: usize,
    max_decisions: usize,
}

impl SATSolver {
    pub fn new() -> Self {
        Self {
            max_conflicts: 100_000,
            max_decisions: 1_000_000,
        }
    }

    pub fn with_limits(max_conflicts: usize, max_decisions: usize) -> Self {
        Self {
            max_conflicts,
            max_decisions,
        }
    }

    /// Solve a SAT formula using DPLL with unit propagation and basic CDCL.
    pub fn solve(&self, formula: &SATFormula) -> SATResult {
        if formula.clauses.is_empty() {
            return SATResult::Satisfiable(vec![]);
        }

        let n = formula.num_variables;
        if n == 0 {
            // Check if all clauses are empty
            for c in &formula.clauses {
                if c.is_empty() {
                    return SATResult::Unsatisfiable;
                }
            }
            return SATResult::Satisfiable(vec![]);
        }

        let mut assignment: Vec<Option<bool>> = vec![None; n + 1];
        let mut decision_level: Vec<usize> = vec![0; n + 1];
        let mut trail: Vec<i32> = Vec::new();
        let mut level = 0usize;
        let mut conflicts = 0usize;
        let mut decisions = 0usize;

        // Learned clauses
        let mut all_clauses: Vec<Vec<i32>> = formula.clauses.clone();

        loop {
            // Unit propagation
            let prop_result = self.unit_propagate(
                &all_clauses,
                &mut assignment,
                &mut trail,
                &mut decision_level,
                level,
            );

            match prop_result {
                PropagationResult::Ok => {}
                PropagationResult::Conflict(conflict_clause_idx) => {
                    conflicts += 1;
                    if conflicts > self.max_conflicts {
                        return SATResult::Unknown;
                    }

                    if level == 0 {
                        return SATResult::Unsatisfiable;
                    }

                    // Conflict analysis: learn a clause and backtrack
                    let (learned_clause, backtrack_level) = self.analyze_conflict(
                        &all_clauses,
                        conflict_clause_idx,
                        &assignment,
                        &decision_level,
                        &trail,
                        level,
                    );

                    // Add learned clause
                    all_clauses.push(learned_clause);

                    // Backtrack
                    self.backtrack(
                        &mut assignment,
                        &mut trail,
                        &mut decision_level,
                        backtrack_level,
                    );
                    level = backtrack_level;
                    continue;
                }
            }

            // Check if all variables are assigned
            if trail.len() == n {
                let result: Vec<bool> = (0..=n)
                    .map(|i| assignment[i].unwrap_or(false))
                    .collect();
                return SATResult::Satisfiable(result);
            }

            // Decision: pick an unassigned variable
            decisions += 1;
            if decisions > self.max_decisions {
                return SATResult::Unknown;
            }

            level += 1;
            let var = self.pick_variable(&assignment, &all_clauses, n);
            if var == 0 {
                // All assigned
                let result: Vec<bool> = (0..=n)
                    .map(|i| assignment[i].unwrap_or(false))
                    .collect();
                return SATResult::Satisfiable(result);
            }

            // Try assigning true first
            assignment[var] = Some(true);
            decision_level[var] = level;
            trail.push(var as i32);
        }
    }

    /// Unit propagation: repeatedly assign forced literals.
    fn unit_propagate(
        &self,
        clauses: &[Vec<i32>],
        assignment: &mut [Option<bool>],
        trail: &mut Vec<i32>,
        decision_level: &mut [usize],
        level: usize,
    ) -> PropagationResult {
        let mut changed = true;
        while changed {
            changed = false;
            for (ci, clause) in clauses.iter().enumerate() {
                let mut unassigned_lit: Option<i32> = None;
                let mut unassigned_count = 0;
                let mut satisfied = false;

                for &lit in clause {
                    let var = lit.unsigned_abs() as usize;
                    let polarity = lit > 0;
                    match assignment.get(var).and_then(|a| *a) {
                        Some(val) => {
                            if val == polarity {
                                satisfied = true;
                                break;
                            }
                        }
                        None => {
                            unassigned_count += 1;
                            unassigned_lit = Some(lit);
                        }
                    }
                }

                if satisfied {
                    continue;
                }

                if unassigned_count == 0 {
                    // Conflict: all literals are false
                    return PropagationResult::Conflict(ci);
                }

                if unassigned_count == 1 {
                    // Unit clause: force the assignment
                    if let Some(lit) = unassigned_lit {
                        let var = lit.unsigned_abs() as usize;
                        let val = lit > 0;
                        assignment[var] = Some(val);
                        decision_level[var] = level;
                        trail.push(lit);
                        changed = true;
                    }
                }
            }
        }
        PropagationResult::Ok
    }

    /// Pure literal elimination: assign pure literals.
    fn pure_literal_elimination(
        &self,
        clauses: &[Vec<i32>],
        assignment: &mut [Option<bool>],
        trail: &mut Vec<i32>,
        decision_level: &mut [usize],
        level: usize,
        n: usize,
    ) {
        let mut positive = vec![false; n + 1];
        let mut negative = vec![false; n + 1];

        for clause in clauses {
            let satisfied = clause.iter().any(|&lit| {
                let var = lit.unsigned_abs() as usize;
                let polarity = lit > 0;
                assignment.get(var).and_then(|a| *a) == Some(polarity)
            });
            if satisfied {
                continue;
            }

            for &lit in clause {
                let var = lit.unsigned_abs() as usize;
                if assignment.get(var).and_then(|a| *a).is_some() {
                    continue;
                }
                if lit > 0 {
                    positive[var] = true;
                } else {
                    negative[var] = true;
                }
            }
        }

        for var in 1..=n {
            if assignment[var].is_some() {
                continue;
            }
            if positive[var] && !negative[var] {
                assignment[var] = Some(true);
                decision_level[var] = level;
                trail.push(var as i32);
            } else if negative[var] && !positive[var] {
                assignment[var] = Some(false);
                decision_level[var] = level;
                trail.push(-(var as i32));
            }
        }
    }

    /// Analyze a conflict to produce a learned clause and backtrack level.
    fn analyze_conflict(
        &self,
        clauses: &[Vec<i32>],
        conflict_idx: usize,
        assignment: &[Option<bool>],
        decision_level: &[usize],
        trail: &[i32],
        current_level: usize,
    ) -> (Vec<i32>, usize) {
        // Simple 1-UIP conflict analysis
        if conflict_idx >= clauses.len() {
            return (vec![], 0);
        }

        let conflict_clause = &clauses[conflict_idx];
        let mut learned: Vec<i32> = Vec::new();
        let mut max_level = 0usize;

        for &lit in conflict_clause {
            let var = lit.unsigned_abs() as usize;
            let level = if var < decision_level.len() {
                decision_level[var]
            } else {
                0
            };
            learned.push(-lit);
            if level > 0 && level < current_level {
                max_level = max_level.max(level);
            }
        }

        if learned.is_empty() {
            learned.push(1); // Trivial clause to prevent infinite loop
        }

        let backtrack_level = if max_level > 0 {
            max_level
        } else if current_level > 0 {
            current_level - 1
        } else {
            0
        };

        (learned, backtrack_level)
    }

    /// Backtrack: undo assignments above the given level.
    fn backtrack(
        &self,
        assignment: &mut [Option<bool>],
        trail: &mut Vec<i32>,
        decision_level: &mut [usize],
        target_level: usize,
    ) {
        while let Some(&lit) = trail.last() {
            let var = lit.unsigned_abs() as usize;
            if var < decision_level.len() && decision_level[var] > target_level {
                assignment[var] = None;
                decision_level[var] = 0;
                trail.pop();
            } else {
                break;
            }
        }
    }

    /// Pick an unassigned variable using VSIDS-like heuristic.
    fn pick_variable(
        &self,
        assignment: &[Option<bool>],
        clauses: &[Vec<i32>],
        n: usize,
    ) -> usize {
        // Simple heuristic: pick the variable that appears in the most unsatisfied clauses
        let mut scores = vec![0usize; n + 1];

        for clause in clauses {
            let satisfied = clause.iter().any(|&lit| {
                let var = lit.unsigned_abs() as usize;
                let polarity = lit > 0;
                assignment.get(var).and_then(|a| *a) == Some(polarity)
            });
            if satisfied {
                continue;
            }
            for &lit in clause {
                let var = lit.unsigned_abs() as usize;
                if var <= n && assignment[var].is_none() {
                    scores[var] += 1;
                }
            }
        }

        let mut best_var = 0;
        let mut best_score = 0;
        for var in 1..=n {
            if assignment[var].is_none() && scores[var] > best_score {
                best_score = scores[var];
                best_var = var;
            }
        }

        // If no scored variable found, pick the first unassigned
        if best_var == 0 {
            for var in 1..=n {
                if assignment[var].is_none() {
                    return var;
                }
            }
        }

        best_var
    }
}

impl Default for SATSolver {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of unit propagation.
#[derive(Debug)]
enum PropagationResult {
    Ok,
    Conflict(usize), // index of the conflicting clause
}

/// Unroll the transition relation for bounded model checking.
pub fn unroll_transition_relation(
    model: &AbstractionState,
    k: usize,
) -> SATFormula {
    let state_list: Vec<AbstractState> = model.abstract_states.clone();
    let state_to_idx: HashMap<AbstractState, usize> = state_list
        .iter()
        .enumerate()
        .map(|(i, s)| (*s, i))
        .collect();
    let n = state_list.len();
    let mut formula = SATFormula::new();

    if n == 0 {
        return formula;
    }

    let state_var = |t: usize, s: usize| -> i32 { (t * n + s + 1) as i32 };

    // Initial states at time 0
    let mut init_clause: Vec<i32> = Vec::new();
    for init in &model.initial_states {
        if let Some(&idx) = state_to_idx.get(init) {
            init_clause.push(state_var(0, idx));
        }
    }
    if !init_clause.is_empty() {
        formula.add_clause(init_clause);
    }

    let adj = model.transition_relation.adjacency_list();

    for t in 0..k {
        // Exactly one state at each time step
        let at_least: Vec<i32> = (0..n).map(|i| state_var(t, i)).collect();
        formula.add_clause(at_least);
        for i in 0..n {
            for j in (i + 1)..n {
                formula.add_clause(vec![-state_var(t, i), -state_var(t, j)]);
            }
        }

        // Transition relation
        for (i, state) in state_list.iter().enumerate() {
            let succs: Vec<usize> = adj
                .get(state)
                .map(|s| {
                    s.iter()
                        .filter_map(|next| state_to_idx.get(next).copied())
                        .collect()
                })
                .unwrap_or_default();

            if !succs.is_empty() {
                let mut clause = vec![-state_var(t, i)];
                for &s in &succs {
                    clause.push(state_var(t + 1, s));
                }
                formula.add_clause(clause);
            }
        }
    }

    // Last time step: at least one state
    let last_clause: Vec<i32> = (0..n).map(|i| state_var(k, i)).collect();
    formula.add_clause(last_clause);

    formula
}

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

/// Reconstruct a path from the parent map.
fn reconstruct_path(
    parent: &HashMap<AbstractState, AbstractState>,
    initial_states: &[AbstractState],
    target: AbstractState,
) -> Vec<AbstractState> {
    let mut path = vec![target];
    let mut current = target;
    let init_set: HashSet<AbstractState> = initial_states.iter().copied().collect();

    while let Some(&p) = parent.get(&current) {
        path.push(p);
        current = p;
        if init_set.contains(&current) {
            break;
        }
    }
    path.reverse();
    path
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abstraction::{AbstractBlockId, GeometricAbstraction};
    use crate::{
        Action, EntityId, RegionId, SceneConfiguration, SceneEntity, SpatialPredicate, State,
        Transition,
    };

    fn make_test_scene() -> SceneConfiguration {
        SceneConfiguration {
            entities: vec![
                SceneEntity {
                    id: EntityId(0),
                    name: "a".into(),
                    position: crate::Point3::new(0.0, 0.0, 0.0),
                    bounding_box: crate::AABB::new(
                        crate::Point3::new(-1.0, -1.0, -1.0),
                        crate::Point3::new(1.0, 1.0, 1.0),
                    ),
                },
            ],
            regions: IndexMap::new(),
            predicate_defs: IndexMap::new(),
        }
    }

    fn make_test_model() -> AbstractionState {
        let automaton = AutomatonDef {
            states: vec![
                State {
                    id: StateId(0),
                    name: "s0".into(),
                    invariant: None,
                    is_accepting: false,
                },
                State {
                    id: StateId(1),
                    name: "s1".into(),
                    invariant: None,
                    is_accepting: true,
                },
            ],
            transitions: vec![
                Transition {
                    id: TransitionId(0),
                    source: StateId(0),
                    target: StateId(1),
                    guard: Guard::True,
                    action: Action::Noop,
                },
                Transition {
                    id: TransitionId(1),
                    source: StateId(1),
                    target: StateId(0),
                    guard: Guard::True,
                    action: Action::Noop,
                },
            ],
            initial: StateId(0),
            accepting: vec![StateId(1)],
            predicates: IndexMap::new(),
        };

        let scene = make_test_scene();
        let abs = GeometricAbstraction::initial_abstraction(&automaton, &scene);
        abs.build_abstract_model()
    }

    #[test]
    fn test_reachability_check() {
        let model = make_test_model();
        let checker = ModelChecker::new();

        // All states should be reachable from initial
        let targets: HashSet<AbstractState> = model
            .abstract_states
            .iter()
            .filter(|s| s.automaton_state == StateId(1))
            .copied()
            .collect();

        let result = checker.check_reachability(&model, &targets);
        assert!(result.is_reachable());
    }

    #[test]
    fn test_unreachable() {
        let model = make_test_model();
        let checker = ModelChecker::new();

        // Empty target set should be unreachable
        let empty: HashSet<AbstractState> = HashSet::new();
        let result = checker.check_reachability(&model, &empty);
        match result {
            ReachabilityResult::Unreachable { .. } => {}
            _ => panic!("Expected unreachable"),
        }
    }

    #[test]
    fn test_deadlock_freedom() {
        let model = make_test_model();
        let checker = ModelChecker::new();
        let result = checker.check_deadlock_freedom(&model);
        assert!(result.is_deadlock_free());
    }

    #[test]
    fn test_safety_check() {
        let model = make_test_model();
        let checker = ModelChecker::new();

        // Empty bad states → safe
        let bad: HashSet<AbstractState> = HashSet::new();
        let result = checker.check_safety(&model, &bad);
        assert!(result.is_safe());
    }

    #[test]
    fn test_liveness_check() {
        let model = make_test_model();
        let checker = ModelChecker::new();

        // All states are progress states → live
        let progress: HashSet<AbstractState> = model.abstract_states.iter().copied().collect();
        let result = checker.check_liveness(&model, &progress);
        assert!(result.is_live());
    }

    #[test]
    fn test_tarjan_scc() {
        let mut states = HashSet::new();
        let s0 = AbstractState {
            automaton_state: StateId(0),
            block_id: AbstractBlockId(0),
        };
        let s1 = AbstractState {
            automaton_state: StateId(1),
            block_id: AbstractBlockId(0),
        };
        states.insert(s0);
        states.insert(s1);

        let mut adj = HashMap::new();
        adj.insert(s0, vec![s1]);
        adj.insert(s1, vec![s0]);

        let sccs = tarjan_scc(&states, &adj);
        // s0 and s1 form one SCC
        assert_eq!(sccs.len(), 1);
        assert_eq!(sccs[0].len(), 2);
    }

    #[test]
    fn test_sat_formula() {
        let mut formula = SATFormula::new();
        formula.add_clause(vec![1, 2]);
        formula.add_clause(vec![-1, 3]);
        assert_eq!(formula.clause_count(), 2);
        assert_eq!(formula.num_variables, 3);
    }

    #[test]
    fn test_sat_solver_satisfiable() {
        let mut formula = SATFormula::new();
        // (x1 ∨ x2) ∧ (¬x1 ∨ x2)  => x2 = true
        formula.add_clause(vec![1, 2]);
        formula.add_clause(vec![-1, 2]);

        let solver = SATSolver::new();
        let result = solver.solve(&formula);
        match result {
            SATResult::Satisfiable(assignment) => {
                assert!(assignment[2]); // x2 must be true
            }
            _ => panic!("Expected satisfiable"),
        }
    }

    #[test]
    fn test_sat_solver_unsatisfiable() {
        let mut formula = SATFormula::new();
        // (x1) ∧ (¬x1)
        formula.add_clause(vec![1]);
        formula.add_clause(vec![-1]);

        let solver = SATSolver::new();
        let result = solver.solve(&formula);
        assert!(matches!(result, SATResult::Unsatisfiable));
    }

    #[test]
    fn test_sat_solver_empty() {
        let formula = SATFormula::new();
        let solver = SATSolver::new();
        let result = solver.solve(&formula);
        assert!(matches!(result, SATResult::Satisfiable(_)));
    }

    #[test]
    fn test_bdd_set_operations() {
        let vars = vec!["a".to_string(), "b".to_string()];
        let mut a = BDDSet::new(vars.clone());
        a.insert(1);
        a.insert(2);
        a.insert(3);

        let mut b = BDDSet::new(vars.clone());
        b.insert(2);
        b.insert(3);
        b.insert(4);

        let union = a.union(&b);
        assert_eq!(union.len(), 4);

        let inter = a.intersection(&b);
        assert_eq!(inter.len(), 2);

        let diff = a.difference(&b);
        assert_eq!(diff.len(), 1);
        assert!(diff.contains(1));
    }

    #[test]
    fn test_bdd_transition_relation() {
        let vars = vec!["s0".to_string(), "s1".to_string(), "s2".to_string()];
        let mut rel = BDDTransitionRelation::new(vars.clone());
        rel.add_transition(0, 1);
        rel.add_transition(1, 2);
        rel.add_transition(2, 0);

        let mut initial = BDDSet::new(vars.clone());
        initial.insert(0);

        let step1 = rel.image(&initial);
        assert!(step1.contains(1));
        assert!(!step1.contains(0));
    }

    #[test]
    fn test_unroll_transition_relation() {
        let model = make_test_model();
        let formula = unroll_transition_relation(&model, 3);
        assert!(formula.clause_count() > 0);
    }

    #[test]
    fn test_bmc_check() {
        let model = make_test_model();
        let checker = ModelChecker::new();

        // No bad states
        let bad: HashSet<AbstractState> = HashSet::new();
        let result = checker.bmc_check(&model, &bad, 5);
        match result {
            BMCResult::NoBugFound { bound } => assert_eq!(bound, 5),
            _ => panic!("Expected no bug found"),
        }
    }

    #[test]
    fn test_symbolic_reachability() {
        let checker = ModelChecker::new();
        let vars = vec!["s0".to_string(), "s1".to_string()];

        let mut rel = BDDTransitionRelation::new(vars.clone());
        rel.add_transition(0, 1);

        let mut initial = BDDSet::new(vars.clone());
        initial.insert(0);

        let mut target = BDDSet::new(vars.clone());
        target.insert(1);

        let result = checker.symbolic_reachability(&rel, &initial, &target);
        assert!(matches!(result, BDDResult::Reachable { steps: 1 }));
    }

    #[test]
    fn test_build_bdd_relation() {
        let model = make_test_model();
        let rel = build_bdd_relation(&model);
        assert!(!rel.transitions.is_empty());
    }
}
