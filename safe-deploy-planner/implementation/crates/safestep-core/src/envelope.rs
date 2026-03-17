//! Rollback safety envelope computation for the SafeStep deployment planner.
//!
//! The safety envelope is the set of states from which a deployment can both
//! reach the target *and* safely retreat to the start, subject to all
//! constraints. Points of No Return (PNR) are states where forward progress
//! is possible but retreat is not.
//!
//! # Architecture
//!
//! - [`SafeSubgraph`] — filters the version-product graph to only
//!   constraint-satisfying states and edges.
//! - [`ReachabilityChecker`] — bidirectional BFS over the safe subgraph.
//! - [`EnvelopeComputer`] — computes the full [`SafetyEnvelope`] for a plan.
//! - [`EnvelopeAnnotator`] — decorates plan steps with envelope membership.
//! - [`WitnessGenerator`] — produces [`StuckWitness`] proofs for stuck states.
//! - [`EnvelopeStats`] — aggregate statistics over a computed envelope.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

use serde::{Deserialize, Serialize};
use tracing::{debug, info, instrument, trace, warn};

use crate::{
    Constraint, CoreResult, DeploymentPlan, Edge, EnvelopeAnnotation, PlanStep, SafetyEnvelope,
    ServiceIndex, State, StuckWitness, TransitionMetadata, VersionIndex, VersionProductGraph,
};
use safestep_types::identifiers::{ConstraintId, Id};

// ---------------------------------------------------------------------------
// SafeSubgraph
// ---------------------------------------------------------------------------

/// A view of the [`VersionProductGraph`] restricted to states and edges that
/// satisfy every provided constraint.
///
/// The subgraph is computed eagerly: on construction we iterate over every
/// graph state once and cache the set of safe states so that later lookups
/// are O(1).
#[derive(Debug, Clone)]
pub struct SafeSubgraph<'g> {
    graph: &'g VersionProductGraph,
    constraints: Vec<Constraint>,
    safe_state_set: HashSet<State>,
}

impl<'g> SafeSubgraph<'g> {
    /// Build a safe subgraph by evaluating every constraint on every state.
    pub fn new(graph: &'g VersionProductGraph, constraints: &[Constraint]) -> Self {
        let constraints = constraints.to_vec();
        let safe_state_set: HashSet<State> = graph
            .states
            .iter()
            .filter(|s| constraints.iter().all(|c| c.check_state(s)))
            .cloned()
            .collect();
        debug!(
            safe_states = safe_state_set.len(),
            total_states = graph.state_count(),
            constraints = constraints.len(),
            "built safe subgraph"
        );
        Self {
            graph,
            constraints,
            safe_state_set,
        }
    }

    /// Returns `true` if `state` satisfies every constraint.
    pub fn is_safe_state(&self, state: &State) -> bool {
        self.safe_state_set.contains(state)
    }

    /// Returns the number of safe states.
    pub fn safe_state_count(&self) -> usize {
        self.safe_state_set.len()
    }

    /// Forward neighbors of `state` that are themselves safe.
    ///
    /// Each returned pair is `(neighbor_state, edge)` where the edge is the
    /// graph edge connecting `state` → `neighbor_state`.
    pub fn safe_neighbors(&self, state: &State) -> Vec<(State, Edge)> {
        if !self.is_safe_state(state) {
            return Vec::new();
        }
        self.graph
            .neighbors(state)
            .into_iter()
            .filter_map(|(neighbor, edge_idx)| {
                if self.is_safe_state(&neighbor) {
                    Some((neighbor, self.graph.edges[edge_idx].clone()))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Backward predecessors of `state` that are themselves safe.
    ///
    /// Each returned pair is `(predecessor_state, edge)` where the edge is
    /// `predecessor_state` → `state`.
    pub fn safe_predecessors(&self, state: &State) -> Vec<(State, Edge)> {
        if !self.is_safe_state(state) {
            return Vec::new();
        }
        self.graph
            .predecessors(state)
            .into_iter()
            .filter_map(|(pred, edge_idx)| {
                if self.is_safe_state(&pred) {
                    Some((pred, self.graph.edges[edge_idx].clone()))
                } else {
                    None
                }
            })
            .collect()
    }

    /// All safe states as a borrowed set.
    pub fn safe_states(&self) -> &HashSet<State> {
        &self.safe_state_set
    }

    /// Which constraints are violated by a given state?
    pub fn violated_constraints(&self, state: &State) -> Vec<ConstraintId> {
        self.constraints
            .iter()
            .filter(|c| !c.check_state(state))
            .map(|c| c.id().clone())
            .collect()
    }

    /// For a given state, find every single-step neighbor (safe or not) and
    /// report which constraints each neighbor violates.
    pub fn neighbor_violations(
        &self,
        state: &State,
    ) -> Vec<(State, Vec<ConstraintId>)> {
        self.graph
            .neighbors(state)
            .into_iter()
            .map(|(neighbor, _edge_idx)| {
                let violations = self.violated_constraints(&neighbor);
                (neighbor, violations)
            })
            .collect()
    }

    /// For a given state, find every single-step predecessor (safe or not) and
    /// report which constraints each predecessor violates.
    pub fn predecessor_violations(
        &self,
        state: &State,
    ) -> Vec<(State, Vec<ConstraintId>)> {
        self.graph
            .predecessors(state)
            .into_iter()
            .map(|(pred, _edge_idx)| {
                let violations = self.violated_constraints(&pred);
                (pred, violations)
            })
            .collect()
    }

    /// Returns a reference to the underlying graph.
    pub fn graph(&self) -> &'g VersionProductGraph {
        self.graph
    }

    /// Returns a reference to the constraints.
    pub fn constraints(&self) -> &[Constraint] {
        &self.constraints
    }
}

// ---------------------------------------------------------------------------
// ReachabilityChecker
// ---------------------------------------------------------------------------

/// Bidirectional BFS reachability over the constraint-satisfying subgraph.
///
/// All BFS traversals respect constraints: a state is only enqueued if it
/// satisfies every constraint (i.e., it lives in the [`SafeSubgraph`]).
#[derive(Debug, Clone)]
pub struct ReachabilityChecker<'g> {
    subgraph: SafeSubgraph<'g>,
}

impl<'g> ReachabilityChecker<'g> {
    /// Create a new checker backed by the given graph and constraints.
    pub fn new(graph: &'g VersionProductGraph, constraints: &[Constraint]) -> Self {
        Self {
            subgraph: SafeSubgraph::new(graph, constraints),
        }
    }

    /// Can we reach `to` from `from` using forward (outgoing) edges,
    /// visiting only constraint-satisfying states?
    #[instrument(skip(self), fields(from = %from, to = %to))]
    pub fn forward_reachable(&self, from: &State, to: &State) -> bool {
        if from == to {
            return self.subgraph.is_safe_state(from);
        }
        if !self.subgraph.is_safe_state(from) || !self.subgraph.is_safe_state(to) {
            return false;
        }

        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        visited.insert(from.clone());
        queue.push_back(from.clone());

        while let Some(current) = queue.pop_front() {
            for (neighbor, _edge) in self.subgraph.safe_neighbors(&current) {
                if neighbor == *to {
                    trace!("forward path found");
                    return true;
                }
                if visited.insert(neighbor.clone()) {
                    queue.push_back(neighbor);
                }
            }
        }
        trace!(visited = visited.len(), "forward path not found");
        false
    }

    /// Can we reach `to` from `from` using *reverse* edges (backwards)?
    ///
    /// Concretely we BFS from `from` along the reverse adjacency and check
    /// whether `to` is discovered. This answers: "starting at `from`, can I
    /// walk backwards along edges until I reach `to`?"
    #[instrument(skip(self), fields(from = %from, to = %to))]
    pub fn backward_reachable(&self, from: &State, to: &State) -> bool {
        if from == to {
            return self.subgraph.is_safe_state(from);
        }
        if !self.subgraph.is_safe_state(from) || !self.subgraph.is_safe_state(to) {
            return false;
        }

        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        visited.insert(from.clone());
        queue.push_back(from.clone());

        while let Some(current) = queue.pop_front() {
            for (pred, _edge) in self.subgraph.safe_predecessors(&current) {
                if pred == *to {
                    trace!("backward path found");
                    return true;
                }
                if visited.insert(pred.clone()) {
                    queue.push_back(pred);
                }
            }
        }
        trace!(visited = visited.len(), "backward path not found");
        false
    }

    /// Compute the full set of states reachable forward from `from`.
    pub fn forward_reachable_set(&self, from: &State) -> HashSet<State> {
        let mut visited = HashSet::new();
        if !self.subgraph.is_safe_state(from) {
            return visited;
        }
        let mut queue = VecDeque::new();
        visited.insert(from.clone());
        queue.push_back(from.clone());

        while let Some(current) = queue.pop_front() {
            for (neighbor, _edge) in self.subgraph.safe_neighbors(&current) {
                if visited.insert(neighbor.clone()) {
                    queue.push_back(neighbor);
                }
            }
        }
        visited
    }

    /// Compute the full set of states reachable backward from `from`.
    pub fn backward_reachable_set(&self, from: &State) -> HashSet<State> {
        let mut visited = HashSet::new();
        if !self.subgraph.is_safe_state(from) {
            return visited;
        }
        let mut queue = VecDeque::new();
        visited.insert(from.clone());
        queue.push_back(from.clone());

        while let Some(current) = queue.pop_front() {
            for (pred, _edge) in self.subgraph.safe_predecessors(&current) {
                if visited.insert(pred.clone()) {
                    queue.push_back(pred);
                }
            }
        }
        visited
    }

    /// Shortest path from `from` to `to` via forward edges, returning the
    /// sequence of states including both endpoints. Returns `None` if
    /// unreachable.
    pub fn shortest_path(&self, from: &State, to: &State) -> Option<Vec<State>> {
        if from == to && self.subgraph.is_safe_state(from) {
            return Some(vec![from.clone()]);
        }
        if !self.subgraph.is_safe_state(from) || !self.subgraph.is_safe_state(to) {
            return None;
        }

        let mut visited: HashMap<State, State> = HashMap::new();
        let mut queue = VecDeque::new();
        visited.insert(from.clone(), from.clone());
        queue.push_back(from.clone());

        while let Some(current) = queue.pop_front() {
            for (neighbor, _edge) in self.subgraph.safe_neighbors(&current) {
                if !visited.contains_key(&neighbor) {
                    visited.insert(neighbor.clone(), current.clone());
                    if neighbor == *to {
                        return Some(Self::reconstruct_path(&visited, from, to));
                    }
                    queue.push_back(neighbor);
                }
            }
        }
        None
    }

    /// Shortest *backward* path from `from` to `to` using reverse edges.
    pub fn shortest_backward_path(&self, from: &State, to: &State) -> Option<Vec<State>> {
        if from == to && self.subgraph.is_safe_state(from) {
            return Some(vec![from.clone()]);
        }
        if !self.subgraph.is_safe_state(from) || !self.subgraph.is_safe_state(to) {
            return None;
        }

        let mut visited: HashMap<State, State> = HashMap::new();
        let mut queue = VecDeque::new();
        visited.insert(from.clone(), from.clone());
        queue.push_back(from.clone());

        while let Some(current) = queue.pop_front() {
            for (pred, _edge) in self.subgraph.safe_predecessors(&current) {
                if !visited.contains_key(&pred) {
                    visited.insert(pred.clone(), current.clone());
                    if pred == *to {
                        return Some(Self::reconstruct_path(&visited, from, to));
                    }
                    queue.push_back(pred);
                }
            }
        }
        None
    }

    /// BFS distance from `from` to `to` via forward edges.
    pub fn forward_distance(&self, from: &State, to: &State) -> Option<usize> {
        self.shortest_path(from, to)
            .map(|path| path.len().saturating_sub(1))
    }

    /// BFS distance from `from` to `to` via backward edges.
    pub fn backward_distance(&self, from: &State, to: &State) -> Option<usize> {
        self.shortest_backward_path(from, to)
            .map(|path| path.len().saturating_sub(1))
    }

    /// Borrow the underlying safe subgraph.
    pub fn subgraph(&self) -> &SafeSubgraph<'g> {
        &self.subgraph
    }

    // -- private helpers --

    fn reconstruct_path(
        parent_map: &HashMap<State, State>,
        from: &State,
        to: &State,
    ) -> Vec<State> {
        let mut path = Vec::new();
        let mut current = to.clone();
        while current != *from {
            path.push(current.clone());
            current = parent_map[&current].clone();
        }
        path.push(from.clone());
        path.reverse();
        path
    }
}

// ---------------------------------------------------------------------------
// EnvelopeComputer
// ---------------------------------------------------------------------------

/// Computes the rollback safety envelope for a given deployment plan.
///
/// The algorithm:
/// 1. Extract intermediate states from the plan.
/// 2. Compute the forward-reachable set from the target (using reverse BFS)
///    and the backward-reachable set from the start (using forward BFS from
///    the start, then checking membership).
/// 3. For each plan state classify it as Inside / PNR / Outside.
#[derive(Debug, Clone)]
pub struct EnvelopeComputer<'g> {
    checker: ReachabilityChecker<'g>,
    graph: &'g VersionProductGraph,
    constraints: Vec<Constraint>,
}

impl<'g> EnvelopeComputer<'g> {
    /// Create a new envelope computer for the given graph and constraints.
    pub fn new(graph: &'g VersionProductGraph, constraints: &[Constraint]) -> Self {
        let checker = ReachabilityChecker::new(graph, constraints);
        Self {
            checker,
            graph,
            constraints: constraints.to_vec(),
        }
    }

    /// Compute the full safety envelope for `plan`.
    ///
    /// The returned [`SafetyEnvelope`] classifies every intermediate state on
    /// the plan path as safe (inside the envelope), a point of no return, or
    /// outside.
    #[instrument(skip(self, plan), fields(plan_id = %plan.id, steps = plan.step_count()))]
    pub fn compute(&self, plan: &DeploymentPlan) -> CoreResult<SafetyEnvelope> {
        info!("computing safety envelope");

        let plan_states = plan.intermediate_states();
        if plan_states.is_empty() {
            return Ok(SafetyEnvelope::new());
        }

        let start = &plan.start;
        let target = &plan.target;

        // Pre-compute the set of states from which the target is reachable.
        // We do this by BFS *backward* from the target: any state in that set
        // can reach the target going forward.
        let can_reach_target = self.checker.backward_reachable_set(target);
        debug!(can_reach_target = can_reach_target.len(), "target backward set");

        // Pre-compute the set of states reachable backward from the start.
        // A state S is in `can_retreat_to_start` if there is a backward path
        // from S to start, i.e., we can walk edges in reverse from S to start.
        // Equivalently: start can reach S going forward.
        let reachable_from_start = self.checker.forward_reachable_set(start);
        debug!(reachable_from_start = reachable_from_start.len(), "start forward set");

        let mut envelope = SafetyEnvelope::new();

        for (idx, state) in plan_states.iter().enumerate() {
            let state_is_safe = self.checker.subgraph().is_safe_state(state);
            let can_forward = state_is_safe && can_reach_target.contains(state);

            // A state can retreat to start if: the start can reach it forward
            // (equivalently, from this state we can walk backward to start).
            let can_backward = state_is_safe && reachable_from_start.contains(state);

            let annotation = match (can_forward, can_backward) {
                (true, true) => {
                    // Inside the envelope: both forward and backward ok.
                    envelope.safe_states.push(state.clone());
                    let risk = self.compute_state_risk(state, idx, &plan_states);
                    EnvelopeAnnotation::Inside { risk_score: risk }
                }
                (true, false) => {
                    // Point of no return: can finish but can't go back.
                    let blocking = self.find_blocking_constraints(state, start);
                    envelope.pnr_states.push(state.clone());
                    EnvelopeAnnotation::PointOfNoReturn {
                        blocking_constraints: blocking,
                    }
                }
                _ => {
                    // Outside: can't even reach the target.
                    let risk = self.compute_state_risk(state, idx, &plan_states);
                    EnvelopeAnnotation::Outside { risk_score: risk }
                }
            };

            trace!(idx, state = %state, ?annotation, "classified plan state");
            envelope.plan_annotations.push(annotation);
        }

        info!(
            safe = envelope.safe_count(),
            pnr = envelope.pnr_count(),
            total = plan_states.len(),
            "envelope computed"
        );
        Ok(envelope)
    }

    /// Compute the envelope and simultaneously generate witnesses for any
    /// states that are outside the envelope.
    pub fn compute_with_witnesses(
        &self,
        plan: &DeploymentPlan,
    ) -> CoreResult<(SafetyEnvelope, Vec<StuckWitness>)> {
        let envelope = self.compute(plan)?;
        let plan_states = plan.intermediate_states();

        let witness_gen = WitnessGenerator::new(self.graph, &self.constraints);
        let mut witnesses = Vec::new();

        for (idx, annotation) in envelope.plan_annotations.iter().enumerate() {
            match annotation {
                EnvelopeAnnotation::Outside { .. } => {
                    if let Some(witness) =
                        witness_gen.generate_witness(&plan_states[idx], &plan.start)
                    {
                        witnesses.push(witness);
                    }
                }
                EnvelopeAnnotation::PointOfNoReturn { .. } => {
                    if let Some(witness) =
                        witness_gen.generate_witness(&plan_states[idx], &plan.start)
                    {
                        witnesses.push(witness);
                    }
                }
                _ => {}
            }
        }

        Ok((envelope, witnesses))
    }

    /// Compute a risk score for a single state in the plan.
    ///
    /// The risk takes into account: distance from start, distance from target,
    /// number of constraint-violating neighbors, and the step's inherent risk.
    fn compute_state_risk(
        &self,
        state: &State,
        idx: usize,
        plan_states: &[State],
    ) -> u32 {
        let total_steps = plan_states.len().saturating_sub(1).max(1);
        let progress_risk = ((idx as f64 / total_steps as f64) * 50.0) as u32;

        let unsafe_neighbor_count = self
            .graph
            .neighbors(state)
            .iter()
            .filter(|(n, _)| !self.checker.subgraph().is_safe_state(n))
            .count();
        let neighbor_risk = (unsafe_neighbor_count as u32).min(50);

        progress_risk + neighbor_risk
    }

    /// Find which constraints block backward reachability from `state` to
    /// `start`.
    fn find_blocking_constraints(&self, state: &State, start: &State) -> Vec<ConstraintId> {
        let mut blocking = Vec::new();

        // Try removing each constraint one at a time and see if backward
        // reachability is restored.
        for (i, constraint) in self.constraints.iter().enumerate() {
            let reduced: Vec<Constraint> = self
                .constraints
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != i)
                .map(|(_, c)| c.clone())
                .collect();

            let reduced_checker = ReachabilityChecker::new(self.graph, &reduced);
            if reduced_checker.backward_reachable(state, start) {
                blocking.push(constraint.id().clone());
            }
        }
        blocking
    }

    /// Expose the internal reachability checker.
    pub fn checker(&self) -> &ReachabilityChecker<'g> {
        &self.checker
    }
}

// ---------------------------------------------------------------------------
// AnnotatedStep
// ---------------------------------------------------------------------------

/// A plan step decorated with envelope membership information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnotatedStep {
    /// Index of this step within the plan (0-based).
    pub step_index: usize,
    /// The system state *after* this step executes.
    pub state: State,
    /// Envelope classification for this state.
    pub annotation: EnvelopeAnnotation,
    /// Number of BFS steps to the nearest envelope boundary (Inside→Outside
    /// transition). 0 if already outside.
    pub distance_to_boundary: usize,
}

impl fmt::Display for AnnotatedStep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match &self.annotation {
            EnvelopeAnnotation::Inside { risk_score } => {
                format!("SAFE (risk={})", risk_score)
            }
            EnvelopeAnnotation::Outside { risk_score } => {
                format!("OUTSIDE (risk={})", risk_score)
            }
            EnvelopeAnnotation::PointOfNoReturn { .. } => "PNR".to_string(),
        };
        write!(
            f,
            "[step {}] {} — boundary_dist={} — {}",
            self.step_index, self.state, self.distance_to_boundary, label
        )
    }
}

// ---------------------------------------------------------------------------
// EnvelopeAnnotator
// ---------------------------------------------------------------------------

/// Annotates plan steps with envelope membership and related metrics.
pub struct EnvelopeAnnotator;

impl EnvelopeAnnotator {
    /// Produce an [`AnnotatedStep`] for every intermediate state in the plan.
    pub fn annotate(plan: &DeploymentPlan, envelope: &SafetyEnvelope) -> Vec<AnnotatedStep> {
        let plan_states = plan.intermediate_states();
        let annotations = &envelope.plan_annotations;

        // Build a quick set of safe-state indices so we can compute distance
        // to the boundary.
        let safe_indices: HashSet<usize> = annotations
            .iter()
            .enumerate()
            .filter_map(|(i, a)| match a {
                EnvelopeAnnotation::Inside { .. } => Some(i),
                _ => None,
            })
            .collect();

        plan_states
            .into_iter()
            .enumerate()
            .map(|(idx, state)| {
                let annotation = annotations
                    .get(idx)
                    .cloned()
                    .unwrap_or(EnvelopeAnnotation::Outside { risk_score: 100 });

                let distance_to_boundary = if safe_indices.contains(&idx) {
                    Self::distance_to_boundary_from(idx, &safe_indices, annotations.len())
                } else {
                    0
                };

                AnnotatedStep {
                    step_index: idx,
                    state,
                    annotation,
                    distance_to_boundary,
                }
            })
            .collect()
    }

    /// Compute a risk score for each plan state, derived from the annotation.
    pub fn compute_risk_scores(
        plan: &DeploymentPlan,
        envelope: &SafetyEnvelope,
    ) -> Vec<u32> {
        envelope
            .plan_annotations
            .iter()
            .enumerate()
            .map(|(idx, annotation)| {
                let base = match annotation {
                    EnvelopeAnnotation::Inside { risk_score } => *risk_score,
                    EnvelopeAnnotation::Outside { risk_score } => *risk_score + 50,
                    EnvelopeAnnotation::PointOfNoReturn {
                        blocking_constraints,
                    } => 30 + (blocking_constraints.len() as u32 * 10),
                };
                let step_risk = plan
                    .steps
                    .get(idx)
                    .map(|s| s.risk_score)
                    .unwrap_or(0);
                base.saturating_add(step_risk)
            })
            .collect()
    }

    /// Find indices where envelope membership transitions.
    ///
    /// A "critical transition" is a step index `i` such that the annotation at
    /// `i` differs in kind from the annotation at `i-1` — specifically where
    /// we go from Inside to PNR, or Inside to Outside.
    pub fn find_critical_transitions(
        plan: &DeploymentPlan,
        envelope: &SafetyEnvelope,
    ) -> Vec<usize> {
        let annotations = &envelope.plan_annotations;
        if annotations.len() < 2 {
            return Vec::new();
        }

        let mut critical = Vec::new();
        for i in 1..annotations.len() {
            let prev_inside = matches!(annotations[i - 1], EnvelopeAnnotation::Inside { .. });
            let curr_pnr = matches!(
                annotations[i],
                EnvelopeAnnotation::PointOfNoReturn { .. }
            );
            let curr_outside = matches!(annotations[i], EnvelopeAnnotation::Outside { .. });

            if prev_inside && (curr_pnr || curr_outside) {
                critical.push(i);
            }
        }
        critical
    }

    /// Find the first plan index that is a Point of No Return.
    pub fn first_pnr_index(envelope: &SafetyEnvelope) -> Option<usize> {
        envelope
            .plan_annotations
            .iter()
            .position(|a| matches!(a, EnvelopeAnnotation::PointOfNoReturn { .. }))
    }

    /// Find the last plan index that is inside the envelope.
    pub fn last_safe_index(envelope: &SafetyEnvelope) -> Option<usize> {
        envelope
            .plan_annotations
            .iter()
            .enumerate()
            .rev()
            .find(|(_, a)| matches!(a, EnvelopeAnnotation::Inside { .. }))
            .map(|(i, _)| i)
    }

    // -- private helpers --

    /// Distance (in plan-step indices) from `idx` to the nearest non-safe
    /// index. Returns the number of safe-state steps to the boundary.
    fn distance_to_boundary_from(
        idx: usize,
        safe_indices: &HashSet<usize>,
        total: usize,
    ) -> usize {
        let mut dist = 0usize;
        loop {
            dist += 1;
            let left_outside = idx
                .checked_sub(dist)
                .map(|j| !safe_indices.contains(&j))
                .unwrap_or(true);
            let right_outside = if idx + dist < total {
                !safe_indices.contains(&(idx + dist))
            } else {
                true
            };
            if left_outside || right_outside {
                return dist;
            }
            if dist > total {
                return total;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// WitnessGenerator
// ---------------------------------------------------------------------------

/// Generates [`StuckWitness`] certificates for states that cannot retreat to
/// the start.
///
/// For a stuck state, the witness enumerates every single-step retreat
/// (backward neighbor) and records which constraints each retreat violates,
/// producing a human-readable explanation.
#[derive(Debug, Clone)]
pub struct WitnessGenerator<'g> {
    subgraph: SafeSubgraph<'g>,
    graph: &'g VersionProductGraph,
    constraints: Vec<Constraint>,
}

impl<'g> WitnessGenerator<'g> {
    pub fn new(graph: &'g VersionProductGraph, constraints: &[Constraint]) -> Self {
        Self {
            subgraph: SafeSubgraph::new(graph, constraints),
            graph,
            constraints: constraints.to_vec(),
        }
    }

    /// Generate a stuck-configuration witness for `state`.
    ///
    /// Returns `Some(witness)` if the state is indeed stuck (cannot retreat
    /// to `start`), or `None` if retreat is actually possible.
    pub fn generate_witness(
        &self,
        state: &State,
        start: &State,
    ) -> Option<StuckWitness> {
        // First verify it's actually stuck.
        let checker = ReachabilityChecker::new(self.graph, &self.constraints);
        if checker.backward_reachable(state, start) {
            return None;
        }

        let mut attempted_retreats = Vec::new();
        let mut all_blocking: HashSet<ConstraintId> = HashSet::new();

        // Examine every single-step predecessor (regardless of safety).
        let predecessors = self.graph.predecessors(state);
        if predecessors.is_empty() {
            return Some(StuckWitness {
                stuck_state: state.clone(),
                blocking_constraints: Vec::new(),
                attempted_retreats: Vec::new(),
                explanation: format!(
                    "State {} has no predecessors in the graph — it is a source node \
                     and retreat is structurally impossible.",
                    state
                ),
            });
        }

        for (pred, _edge_idx) in &predecessors {
            attempted_retreats.push(pred.clone());

            // Collect violated constraints for this predecessor.
            let violations = self.subgraph.violated_constraints(pred);
            for v in &violations {
                all_blocking.insert(v.clone());
            }

            // Even if the predecessor itself is safe, check whether we can
            // continue backward from it to the start.
            if violations.is_empty() && !checker.backward_reachable(pred, start) {
                // The predecessor is safe but still can't reach start.
                // Deeper structural blockage.
            }
        }

        // If no constraint violations were found at the immediate
        // predecessors, it means the blockage is deeper — the predecessors
        // are safe but form a dead-end cluster. Try to find blocking
        // constraints by looking at the broader backward neighborhood.
        if all_blocking.is_empty() {
            all_blocking = self.deep_blocking_search(state, start, &checker);
        }

        let blocking_vec: Vec<ConstraintId> = all_blocking.into_iter().collect();

        let explanation = self.build_explanation(state, start, &attempted_retreats, &blocking_vec);

        Some(StuckWitness {
            stuck_state: state.clone(),
            blocking_constraints: blocking_vec,
            attempted_retreats,
            explanation,
        })
    }

    /// Search up to 3 hops backward for constraint violations.
    fn deep_blocking_search(
        &self,
        state: &State,
        start: &State,
        checker: &ReachabilityChecker<'_>,
    ) -> HashSet<ConstraintId> {
        let mut blocking = HashSet::new();
        let mut visited = HashSet::new();
        let mut frontier: Vec<State> = vec![state.clone()];
        visited.insert(state.clone());

        for _depth in 0..3 {
            let mut next_frontier = Vec::new();
            for s in &frontier {
                for (pred, _edge_idx) in self.graph.predecessors(s) {
                    if visited.insert(pred.clone()) {
                        let violations = self.subgraph.violated_constraints(&pred);
                        for v in violations {
                            blocking.insert(v);
                        }
                        next_frontier.push(pred);
                    }
                }
            }
            if next_frontier.is_empty() {
                break;
            }
            frontier = next_frontier;
        }
        blocking
    }

    /// Build a human-readable explanation of why retreat is blocked.
    fn build_explanation(
        &self,
        state: &State,
        start: &State,
        attempted: &[State],
        blocking: &[ConstraintId],
    ) -> String {
        let mut parts = Vec::new();
        parts.push(format!(
            "State {} cannot retreat to start state {}.",
            state, start
        ));
        parts.push(format!(
            "Examined {} possible retreat state(s).",
            attempted.len()
        ));

        if blocking.is_empty() {
            parts.push(
                "No single constraint was identified as blocking; the state is \
                 structurally unreachable from start in the reverse graph."
                    .to_string(),
            );
        } else {
            let ids: Vec<String> = blocking.iter().map(|c| c.to_string()).collect();
            parts.push(format!(
                "Blocking constraint(s): [{}].",
                ids.join(", ")
            ));
        }

        parts.join(" ")
    }

    /// Generate witnesses for all PNR states in an envelope.
    pub fn generate_all_witnesses(
        &self,
        envelope: &SafetyEnvelope,
        start: &State,
    ) -> Vec<StuckWitness> {
        envelope
            .pnr_states
            .iter()
            .filter_map(|s| self.generate_witness(s, start))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// EnvelopeStats
// ---------------------------------------------------------------------------

/// Aggregate statistics about a computed safety envelope.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvelopeStats {
    /// Number of plan states inside the envelope.
    pub envelope_size: usize,
    /// Number of PNR states.
    pub pnr_count: usize,
    /// Number of outside states.
    pub outside_count: usize,
    /// Ratio of safe states to total states.
    pub safe_ratio: f64,
    /// Plan indices where envelope membership transitions.
    pub critical_transitions: Vec<usize>,
    /// Maximum distance (in plan steps) one can retreat before exiting the
    /// envelope.
    pub max_retreat_distance: usize,
    /// Index of the first PNR state, if any.
    pub first_pnr_index: Option<usize>,
    /// Total number of plan states examined.
    pub total_states: usize,
}

impl EnvelopeStats {
    /// Compute statistics from an envelope and the plan it annotates.
    pub fn compute(envelope: &SafetyEnvelope, plan: &DeploymentPlan) -> Self {
        let annotations = &envelope.plan_annotations;
        let total_states = annotations.len();

        let envelope_size = annotations
            .iter()
            .filter(|a| matches!(a, EnvelopeAnnotation::Inside { .. }))
            .count();

        let pnr_count = annotations
            .iter()
            .filter(|a| matches!(a, EnvelopeAnnotation::PointOfNoReturn { .. }))
            .count();

        let outside_count = annotations
            .iter()
            .filter(|a| matches!(a, EnvelopeAnnotation::Outside { .. }))
            .count();

        let safe_ratio = if total_states > 0 {
            envelope_size as f64 / total_states as f64
        } else {
            0.0
        };

        let critical_transitions =
            EnvelopeAnnotator::find_critical_transitions(plan, envelope);

        let first_pnr_index = EnvelopeAnnotator::first_pnr_index(envelope);

        let max_retreat_distance = Self::compute_max_retreat_distance(annotations);

        Self {
            envelope_size,
            pnr_count,
            outside_count,
            safe_ratio,
            critical_transitions,
            max_retreat_distance,
            first_pnr_index,
            total_states,
        }
    }

    /// A textual summary of the envelope statistics.
    pub fn summary(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!("Envelope Statistics:"));
        lines.push(format!(
            "  Total plan states: {}",
            self.total_states
        ));
        lines.push(format!(
            "  Safe (inside envelope): {} ({:.1}%)",
            self.envelope_size,
            self.safe_ratio * 100.0
        ));
        lines.push(format!("  Points of No Return: {}", self.pnr_count));
        lines.push(format!("  Outside envelope: {}", self.outside_count));

        if let Some(pnr_idx) = self.first_pnr_index {
            lines.push(format!("  First PNR at step index: {}", pnr_idx));
        } else {
            lines.push("  No PNR states — full retreat possible from every step.".to_string());
        }

        lines.push(format!(
            "  Max retreat distance: {} step(s)",
            self.max_retreat_distance
        ));

        if !self.critical_transitions.is_empty() {
            let indices: Vec<String> = self
                .critical_transitions
                .iter()
                .map(|i| i.to_string())
                .collect();
            lines.push(format!(
                "  Critical transitions at: [{}]",
                indices.join(", ")
            ));
        }

        lines.join("\n")
    }

    /// Is the entire plan inside the envelope?
    pub fn is_fully_safe(&self) -> bool {
        self.pnr_count == 0 && self.outside_count == 0
    }

    /// The maximum contiguous run of safe states starting from index 0.
    fn compute_max_retreat_distance(annotations: &[EnvelopeAnnotation]) -> usize {
        let mut max_run = 0usize;
        let mut current_run = 0usize;

        for annotation in annotations {
            if matches!(annotation, EnvelopeAnnotation::Inside { .. }) {
                current_run += 1;
                if current_run > max_run {
                    max_run = current_run;
                }
            } else {
                current_run = 0;
            }
        }
        max_run
    }
}

impl fmt::Display for EnvelopeStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.summary())
    }
}

// ---------------------------------------------------------------------------
// Helper: envelope membership classification for arbitrary states
// ---------------------------------------------------------------------------

/// Classify a single state w.r.t. the plan envelope *without* building the
/// full envelope first. Useful for ad-hoc queries during plan exploration.
pub fn classify_state(
    state: &State,
    start: &State,
    target: &State,
    graph: &VersionProductGraph,
    constraints: &[Constraint],
) -> EnvelopeAnnotation {
    let checker = ReachabilityChecker::new(graph, constraints);
    let safe = checker.subgraph().is_safe_state(state);
    if !safe {
        let violations = checker.subgraph().violated_constraints(state);
        return EnvelopeAnnotation::Outside {
            risk_score: 100 + violations.len() as u32 * 10,
        };
    }

    let can_forward = checker.forward_reachable(state, target);
    let can_backward = checker.backward_reachable(state, start);

    match (can_forward, can_backward) {
        (true, true) => EnvelopeAnnotation::Inside { risk_score: 0 },
        (true, false) => {
            let blocking = find_all_blocking(state, start, graph, constraints);
            EnvelopeAnnotation::PointOfNoReturn {
                blocking_constraints: blocking,
            }
        }
        (false, _) => EnvelopeAnnotation::Outside { risk_score: 75 },
    }
}

/// Identify which constraints are responsible for blocking backward
/// reachability from `state` to `start`.
fn find_all_blocking(
    state: &State,
    start: &State,
    graph: &VersionProductGraph,
    constraints: &[Constraint],
) -> Vec<ConstraintId> {
    let mut result = Vec::new();
    for (i, c) in constraints.iter().enumerate() {
        let reduced: Vec<Constraint> = constraints
            .iter()
            .enumerate()
            .filter(|&(j, _)| j != i)
            .map(|(_, c)| c.clone())
            .collect();
        let checker = ReachabilityChecker::new(graph, &reduced);
        if checker.backward_reachable(state, start) {
            result.push(c.id().clone());
        }
    }
    result
}

/// Compute the "retreat frontier": the set of safe states reachable backward
/// from any plan state. Useful for visualization.
pub fn retreat_frontier(
    plan: &DeploymentPlan,
    graph: &VersionProductGraph,
    constraints: &[Constraint],
) -> HashSet<State> {
    let checker = ReachabilityChecker::new(graph, constraints);
    let mut frontier = HashSet::new();
    for state in plan.intermediate_states() {
        if checker.subgraph().is_safe_state(&state) {
            let backward = checker.backward_reachable_set(&state);
            for s in backward {
                frontier.insert(s);
            }
        }
    }
    frontier
}

/// Check whether a plan is "fully safe" — every intermediate state is inside
/// the envelope (can reach both start and target).
pub fn is_plan_fully_safe(
    plan: &DeploymentPlan,
    graph: &VersionProductGraph,
    constraints: &[Constraint],
) -> bool {
    let checker = ReachabilityChecker::new(graph, constraints);
    let start = &plan.start;
    let target = &plan.target;

    let can_reach_target = checker.backward_reachable_set(target);
    let reachable_from_start = checker.forward_reachable_set(start);

    for state in plan.intermediate_states() {
        if !checker.subgraph().is_safe_state(&state) {
            return false;
        }
        if !can_reach_target.contains(&state) {
            return false;
        }
        if !reachable_from_start.contains(&state) {
            return false;
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Constraint, DeploymentPlan, Edge, PlanStep, ServiceDescriptor, ServiceIndex, State,
        TransitionMetadata, VersionIndex, VersionProductGraph,
    };
    use safestep_types::identifiers::Id;

    // -- test helpers --

    /// Build a simple two-service, two-version graph:
    ///
    /// States: (0,0) (0,1) (1,0) (1,1)
    /// Edges:  forward and backward for each single-service step.
    fn make_simple_graph() -> VersionProductGraph {
        let svc_a = ServiceDescriptor::new("alpha", vec!["v0".into(), "v1".into()]);
        let svc_b = ServiceDescriptor::new("beta", vec!["v0".into(), "v1".into()]);
        let mut g = VersionProductGraph::new(vec![svc_a, svc_b]);

        let states: Vec<State> = vec![
            State::new(vec![VersionIndex(0), VersionIndex(0)]),
            State::new(vec![VersionIndex(0), VersionIndex(1)]),
            State::new(vec![VersionIndex(1), VersionIndex(0)]),
            State::new(vec![VersionIndex(1), VersionIndex(1)]),
        ];
        for s in &states {
            g.add_state(s.clone());
        }

        // Edges for service 0 transitions (both directions):
        for b in 0u16..=1 {
            let from = State::new(vec![VersionIndex(0), VersionIndex(b)]);
            let to = State::new(vec![VersionIndex(1), VersionIndex(b)]);
            g.add_edge(Edge {
                from: from.clone(),
                to: to.clone(),
                service: ServiceIndex(0),
                from_version: VersionIndex(0),
                to_version: VersionIndex(1),
                metadata: TransitionMetadata::default(),
            });
            g.add_edge(Edge {
                from: to,
                to: from,
                service: ServiceIndex(0),
                from_version: VersionIndex(1),
                to_version: VersionIndex(0),
                metadata: TransitionMetadata {
                    is_upgrade: false,
                    ..Default::default()
                },
            });
        }

        // Edges for service 1 transitions (both directions):
        for a in 0u16..=1 {
            let from = State::new(vec![VersionIndex(a), VersionIndex(0)]);
            let to = State::new(vec![VersionIndex(a), VersionIndex(1)]);
            g.add_edge(Edge {
                from: from.clone(),
                to: to.clone(),
                service: ServiceIndex(1),
                from_version: VersionIndex(0),
                to_version: VersionIndex(1),
                metadata: TransitionMetadata::default(),
            });
            g.add_edge(Edge {
                from: to,
                to: from,
                service: ServiceIndex(1),
                from_version: VersionIndex(1),
                to_version: VersionIndex(0),
                metadata: TransitionMetadata {
                    is_upgrade: false,
                    ..Default::default()
                },
            });
        }

        g
    }

    /// Build a simple plan: (0,0) → (1,0) → (1,1).
    fn make_simple_plan() -> DeploymentPlan {
        let start = State::new(vec![VersionIndex(0), VersionIndex(0)]);
        let target = State::new(vec![VersionIndex(1), VersionIndex(1)]);
        let steps = vec![
            PlanStep::new(ServiceIndex(0), VersionIndex(0), VersionIndex(1)),
            PlanStep::new(ServiceIndex(1), VersionIndex(0), VersionIndex(1)),
        ];
        DeploymentPlan::new(start, target, steps)
    }

    /// A compatibility constraint that forbids (1,0) — service 0 at v1 while
    /// service 1 is still at v0. This creates a PNR.
    fn make_blocking_constraint() -> Constraint {
        Constraint::Compatibility {
            id: Id::from_name("compat-block"),
            service_a: ServiceIndex(0),
            service_b: ServiceIndex(1),
            compatible_pairs: vec![
                (VersionIndex(0), VersionIndex(0)),
                (VersionIndex(0), VersionIndex(1)),
                // (VersionIndex(1), VersionIndex(0)) deliberately missing!
                (VersionIndex(1), VersionIndex(1)),
            ],
        }
    }

    // -- actual tests --

    #[test]
    fn test_safe_subgraph_no_constraints() {
        let g = make_simple_graph();
        let sub = SafeSubgraph::new(&g, &[]);
        assert_eq!(sub.safe_state_count(), 4);
        assert!(sub.is_safe_state(&State::new(vec![VersionIndex(0), VersionIndex(0)])));
        assert!(sub.is_safe_state(&State::new(vec![VersionIndex(1), VersionIndex(0)])));
    }

    #[test]
    fn test_safe_subgraph_with_forbidden() {
        let g = make_simple_graph();
        let c = Constraint::Forbidden {
            id: Id::from_name("no-1-0"),
            service: ServiceIndex(0),
            version: VersionIndex(1),
        };
        let sub = SafeSubgraph::new(&g, &[c]);
        assert_eq!(sub.safe_state_count(), 2);
        assert!(sub.is_safe_state(&State::new(vec![VersionIndex(0), VersionIndex(0)])));
        assert!(sub.is_safe_state(&State::new(vec![VersionIndex(0), VersionIndex(1)])));
        assert!(!sub.is_safe_state(&State::new(vec![VersionIndex(1), VersionIndex(0)])));
        assert!(!sub.is_safe_state(&State::new(vec![VersionIndex(1), VersionIndex(1)])));
    }

    #[test]
    fn test_safe_subgraph_neighbors() {
        let g = make_simple_graph();
        let c = make_blocking_constraint();
        let sub = SafeSubgraph::new(&g, &[c]);
        // (1,0) is unsafe so neighbors from (0,0) should not include it.
        let s00 = State::new(vec![VersionIndex(0), VersionIndex(0)]);
        let neighbors: Vec<State> = sub.safe_neighbors(&s00).into_iter().map(|(s, _)| s).collect();
        assert!(!neighbors.contains(&State::new(vec![VersionIndex(1), VersionIndex(0)])));
        assert!(neighbors.contains(&State::new(vec![VersionIndex(0), VersionIndex(1)])));
    }

    #[test]
    fn test_forward_reachability_unconstrained() {
        let g = make_simple_graph();
        let checker = ReachabilityChecker::new(&g, &[]);
        let s00 = State::new(vec![VersionIndex(0), VersionIndex(0)]);
        let s11 = State::new(vec![VersionIndex(1), VersionIndex(1)]);
        assert!(checker.forward_reachable(&s00, &s11));
        assert!(checker.forward_reachable(&s11, &s00));
    }

    #[test]
    fn test_backward_reachability_unconstrained() {
        let g = make_simple_graph();
        let checker = ReachabilityChecker::new(&g, &[]);
        let s00 = State::new(vec![VersionIndex(0), VersionIndex(0)]);
        let s11 = State::new(vec![VersionIndex(1), VersionIndex(1)]);
        assert!(checker.backward_reachable(&s11, &s00));
    }

    #[test]
    fn test_forward_reachable_set() {
        let g = make_simple_graph();
        let checker = ReachabilityChecker::new(&g, &[]);
        let s00 = State::new(vec![VersionIndex(0), VersionIndex(0)]);
        let reachable = checker.forward_reachable_set(&s00);
        assert_eq!(reachable.len(), 4);
    }

    #[test]
    fn test_shortest_path() {
        let g = make_simple_graph();
        let checker = ReachabilityChecker::new(&g, &[]);
        let s00 = State::new(vec![VersionIndex(0), VersionIndex(0)]);
        let s11 = State::new(vec![VersionIndex(1), VersionIndex(1)]);
        let path = checker.shortest_path(&s00, &s11).unwrap();
        // Should be length 3 (start, intermediate, end) — two hops.
        assert_eq!(path.len(), 3);
        assert_eq!(path[0], s00);
        assert_eq!(path[2], s11);
    }

    #[test]
    fn test_envelope_fully_safe() {
        let g = make_simple_graph();
        let plan = make_simple_plan();
        let computer = EnvelopeComputer::new(&g, &[]);
        let envelope = computer.compute(&plan).unwrap();
        // Without constraints, every state is reachable in both directions.
        assert_eq!(envelope.safe_states.len(), 3); // (0,0), (1,0), (1,1)
        assert_eq!(envelope.pnr_states.len(), 0);
        for annotation in &envelope.plan_annotations {
            assert!(matches!(annotation, EnvelopeAnnotation::Inside { .. }));
        }
    }

    #[test]
    fn test_envelope_with_pnr() {
        // The blocking constraint forbids (1,0), so the plan path
        // (0,0) → (1,0) → (1,1) has (1,0) as an unsafe state.
        // Since (1,0) is not in the safe subgraph at all, it becomes Outside.
        let g = make_simple_graph();
        let c = make_blocking_constraint();
        let plan = make_simple_plan();
        let computer = EnvelopeComputer::new(&g, &[c]);
        let envelope = computer.compute(&plan).unwrap();

        // (0,0) can reach target via (0,1)→(1,1) and can retreat to itself.
        // (1,0) violates the compatibility constraint → Outside.
        // (1,1) can reach target (is target) but can it retreat to (0,0)?
        //   (1,1) backward: predecessors are (0,1) and (1,0).
        //   (1,0) is unsafe so skip. (0,1) is safe.
        //   From (0,1) backward: predecessors include (0,0). So yes.
        //   Also check forward from start: (0,0) → (0,1) → (1,1). Yes.
        //   So (1,1) should be Inside.
        let state_10 = State::new(vec![VersionIndex(1), VersionIndex(0)]);
        assert!(!envelope.is_safe(&state_10));

        // The plan path is [(0,0), (1,0), (1,1)].
        // (0,0) is Inside, (1,0) is Outside, (1,1) is Inside.
        assert!(matches!(
            envelope.plan_annotations[0],
            EnvelopeAnnotation::Inside { .. }
        ));
        assert!(matches!(
            envelope.plan_annotations[1],
            EnvelopeAnnotation::Outside { .. }
        ));
        assert!(matches!(
            envelope.plan_annotations[2],
            EnvelopeAnnotation::Inside { .. }
        ));
    }

    #[test]
    fn test_envelope_annotator() {
        let g = make_simple_graph();
        let plan = make_simple_plan();
        let computer = EnvelopeComputer::new(&g, &[]);
        let envelope = computer.compute(&plan).unwrap();
        let annotated = EnvelopeAnnotator::annotate(&plan, &envelope);
        assert_eq!(annotated.len(), 3);
        for step in &annotated {
            assert!(matches!(
                step.annotation,
                EnvelopeAnnotation::Inside { .. }
            ));
        }
    }

    #[test]
    fn test_envelope_risk_scores() {
        let g = make_simple_graph();
        let c = make_blocking_constraint();
        let plan = make_simple_plan();
        let computer = EnvelopeComputer::new(&g, &[c]);
        let envelope = computer.compute(&plan).unwrap();
        let scores = EnvelopeAnnotator::compute_risk_scores(&plan, &envelope);
        assert_eq!(scores.len(), 3);
        // The Outside state (index 1) should have a higher score.
        assert!(scores[1] > scores[0]);
    }

    #[test]
    fn test_critical_transitions() {
        let g = make_simple_graph();
        let c = make_blocking_constraint();
        let plan = make_simple_plan();
        let computer = EnvelopeComputer::new(&g, &[c]);
        let envelope = computer.compute(&plan).unwrap();
        let critical = EnvelopeAnnotator::find_critical_transitions(&plan, &envelope);
        // Transition at index 1: Inside → Outside.
        assert!(critical.contains(&1));
    }

    #[test]
    fn test_witness_generator_no_retreat() {
        let g = make_simple_graph();
        let c = make_blocking_constraint();
        let start = State::new(vec![VersionIndex(0), VersionIndex(0)]);
        let stuck = State::new(vec![VersionIndex(1), VersionIndex(0)]);
        let gen = WitnessGenerator::new(&g, &[c]);
        let witness = gen.generate_witness(&stuck, &start);
        // (1,0) violates the constraint so it's unsafe.
        // The witness should exist since the state is unsafe and BFS can't
        // reach start from an unsafe starting point.
        assert!(witness.is_some());
        let w = witness.unwrap();
        assert_eq!(w.stuck_state, stuck);
        assert!(!w.explanation.is_empty());
    }

    #[test]
    fn test_witness_generator_reachable() {
        let g = make_simple_graph();
        let start = State::new(vec![VersionIndex(0), VersionIndex(0)]);
        let gen = WitnessGenerator::new(&g, &[]);
        let s11 = State::new(vec![VersionIndex(1), VersionIndex(1)]);
        // Without constraints, retreat is possible → no witness.
        let witness = gen.generate_witness(&s11, &start);
        assert!(witness.is_none());
    }

    #[test]
    fn test_envelope_stats() {
        let g = make_simple_graph();
        let c = make_blocking_constraint();
        let plan = make_simple_plan();
        let computer = EnvelopeComputer::new(&g, &[c]);
        let envelope = computer.compute(&plan).unwrap();
        let stats = EnvelopeStats::compute(&envelope, &plan);

        assert_eq!(stats.total_states, 3);
        assert_eq!(stats.envelope_size, 2); // (0,0) and (1,1)
        assert!(!stats.is_fully_safe());
        assert!(stats.safe_ratio > 0.0 && stats.safe_ratio < 1.0);

        let summary = stats.summary();
        assert!(summary.contains("Envelope Statistics"));
        assert!(summary.contains("Safe (inside envelope): 2"));
    }

    #[test]
    fn test_classify_state_standalone() {
        let g = make_simple_graph();
        let start = State::new(vec![VersionIndex(0), VersionIndex(0)]);
        let target = State::new(vec![VersionIndex(1), VersionIndex(1)]);
        let annotation = classify_state(&start, &start, &target, &g, &[]);
        assert!(matches!(annotation, EnvelopeAnnotation::Inside { .. }));
    }

    #[test]
    fn test_is_plan_fully_safe() {
        let g = make_simple_graph();
        let plan = make_simple_plan();
        assert!(is_plan_fully_safe(&plan, &g, &[]));
        let c = make_blocking_constraint();
        assert!(!is_plan_fully_safe(&plan, &g, &[c]));
    }

    #[test]
    fn test_reachability_identity() {
        let g = make_simple_graph();
        let checker = ReachabilityChecker::new(&g, &[]);
        let s = State::new(vec![VersionIndex(0), VersionIndex(0)]);
        assert!(checker.forward_reachable(&s, &s));
        assert!(checker.backward_reachable(&s, &s));
        let path = checker.shortest_path(&s, &s).unwrap();
        assert_eq!(path.len(), 1);
    }

    #[test]
    fn test_reachability_unsafe_source() {
        let g = make_simple_graph();
        let c = Constraint::Forbidden {
            id: Id::from_name("ban-00"),
            service: ServiceIndex(0),
            version: VersionIndex(0),
        };
        let checker = ReachabilityChecker::new(&g, &[c]);
        let s00 = State::new(vec![VersionIndex(0), VersionIndex(0)]);
        let s11 = State::new(vec![VersionIndex(1), VersionIndex(1)]);
        // s00 is unsafe, so forward reachability from it is false.
        assert!(!checker.forward_reachable(&s00, &s11));
        assert!(checker.forward_reachable_set(&s00).is_empty());
    }

    #[test]
    fn test_forward_and_backward_distance() {
        let g = make_simple_graph();
        let checker = ReachabilityChecker::new(&g, &[]);
        let s00 = State::new(vec![VersionIndex(0), VersionIndex(0)]);
        let s11 = State::new(vec![VersionIndex(1), VersionIndex(1)]);
        let fwd = checker.forward_distance(&s00, &s11);
        assert_eq!(fwd, Some(2));
        let bwd = checker.backward_distance(&s11, &s00);
        assert_eq!(bwd, Some(2));
    }

    #[test]
    fn test_safe_subgraph_violated_constraints() {
        let g = make_simple_graph();
        let c = make_blocking_constraint();
        let sub = SafeSubgraph::new(&g, &[c]);
        let s10 = State::new(vec![VersionIndex(1), VersionIndex(0)]);
        let violations = sub.violated_constraints(&s10);
        assert_eq!(violations.len(), 1);
    }

    #[test]
    fn test_envelope_compute_with_witnesses() {
        let g = make_simple_graph();
        let c = make_blocking_constraint();
        let plan = make_simple_plan();
        let computer = EnvelopeComputer::new(&g, &[c]);
        let (envelope, witnesses) = computer.compute_with_witnesses(&plan).unwrap();
        // (1,0) is Outside → should produce a witness.
        assert!(!witnesses.is_empty());
        assert!(matches!(
            envelope.plan_annotations[1],
            EnvelopeAnnotation::Outside { .. }
        ));
    }

    #[test]
    fn test_retreat_frontier() {
        let g = make_simple_graph();
        let plan = make_simple_plan();
        let frontier = retreat_frontier(&plan, &g, &[]);
        // All 4 states should be in the frontier since everything is reachable.
        assert_eq!(frontier.len(), 4);
    }

    #[test]
    fn test_annotated_step_display() {
        let step = AnnotatedStep {
            step_index: 0,
            state: State::new(vec![VersionIndex(0), VersionIndex(0)]),
            annotation: EnvelopeAnnotation::Inside { risk_score: 5 },
            distance_to_boundary: 2,
        };
        let s = format!("{}", step);
        assert!(s.contains("step 0"));
        assert!(s.contains("SAFE"));
        assert!(s.contains("boundary_dist=2"));
    }

    #[test]
    fn test_envelope_stats_fully_safe() {
        let g = make_simple_graph();
        let plan = make_simple_plan();
        let computer = EnvelopeComputer::new(&g, &[]);
        let envelope = computer.compute(&plan).unwrap();
        let stats = EnvelopeStats::compute(&envelope, &plan);
        assert!(stats.is_fully_safe());
        assert_eq!(stats.safe_ratio, 1.0);
        assert!(stats.first_pnr_index.is_none());
        assert!(stats.critical_transitions.is_empty());
    }

    #[test]
    fn test_envelope_stats_display() {
        let g = make_simple_graph();
        let plan = make_simple_plan();
        let computer = EnvelopeComputer::new(&g, &[]);
        let envelope = computer.compute(&plan).unwrap();
        let stats = EnvelopeStats::compute(&envelope, &plan);
        let display = format!("{}", stats);
        assert!(display.contains("Envelope Statistics"));
    }
}
