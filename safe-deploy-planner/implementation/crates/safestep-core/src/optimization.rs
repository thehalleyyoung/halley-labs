//! Plan optimization module for the SafeStep deployment planner.
//!
//! Provides several optimization strategies for deployment plans:
//! - **Step minimisation** — remove redundant transitions and shorten paths via BFS.
//! - **Risk minimisation** — reorder steps to reduce exposure to unsafe / PNR states.
//! - **Pareto optimisation** — compute the Pareto-optimal frontier across multiple
//!   objectives (steps, risk, duration, resource cost, PNR exposure).
//! - **Step merging** — detect independent steps that can execute in parallel.
//! - **Configurable cost model** — weighted combination of per-objective costs.

use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::cmp::Ordering as CmpOrdering;

use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};
use tracing::{debug, instrument, trace, warn};

use crate::{
    Constraint, CoreResult, DeploymentPlan, PlanStep, SafetyEnvelope,
    ServiceIndex, State, VersionIndex, VersionProductGraph,
};
use safestep_types::identifiers::{Id, PlanId, StepId};

// ---------------------------------------------------------------------------
// OptimizationObjective
// ---------------------------------------------------------------------------

/// An objective that may be optimised independently or as part of a
/// multi-objective Pareto analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OptimizationObjective {
    /// Minimise the number of deployment steps.
    Steps,
    /// Minimise the cumulative risk score.
    Risk,
    /// Minimise total estimated duration in seconds.
    Duration,
    /// Minimise aggregate resource cost.
    ResourceCost,
    /// Minimise the number of steps whose intermediate state is a PNR.
    PnrExposure,
}

impl OptimizationObjective {
    /// Evaluate this objective on a plan, optionally using a safety envelope
    /// (required for [`PnrExposure`]).
    pub fn evaluate(&self, plan: &DeploymentPlan, envelope: Option<&SafetyEnvelope>) -> f64 {
        match self {
            Self::Steps => plan.step_count() as f64,
            Self::Risk => plan.total_risk as f64,
            Self::Duration => plan.total_duration_secs as f64,
            Self::ResourceCost => {
                plan.steps
                    .iter()
                    .map(|s| s.estimated_duration_secs as f64 * (1.0 + s.risk_score as f64 / 100.0))
                    .sum()
            }
            Self::PnrExposure => {
                let Some(env) = envelope else { return 0.0 };
                let intermediates = plan.intermediate_states();
                intermediates
                    .iter()
                    .filter(|s| env.is_pnr(s))
                    .count() as f64
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CostModel
// ---------------------------------------------------------------------------

/// A configurable, weighted cost model that produces scalar or per-objective
/// cost vectors from a deployment plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostModel {
    pub step_cost: f64,
    pub risk_weight: f64,
    pub duration_weight: f64,
    pub resource_weight: f64,
}

impl CostModel {
    pub fn new() -> Self {
        Self {
            step_cost: 1.0,
            risk_weight: 1.0,
            duration_weight: 1.0,
            resource_weight: 0.5,
        }
    }

    pub fn with_step_cost(mut self, cost: f64) -> Self {
        self.step_cost = cost;
        self
    }

    pub fn with_risk_weight(mut self, weight: f64) -> Self {
        self.risk_weight = weight;
        self
    }

    pub fn with_duration_weight(mut self, weight: f64) -> Self {
        self.duration_weight = weight;
        self
    }

    pub fn with_resource_weight(mut self, weight: f64) -> Self {
        self.resource_weight = weight;
        self
    }

    /// Weighted scalar cost.
    pub fn evaluate(&self, plan: &DeploymentPlan) -> f64 {
        let step_term = plan.step_count() as f64 * self.step_cost;
        let risk_term = plan.total_risk as f64 * self.risk_weight;
        let duration_term = plan.total_duration_secs as f64 * self.duration_weight;
        let resource_term: f64 = plan
            .steps
            .iter()
            .map(|s| s.estimated_duration_secs as f64 * (1.0 + s.risk_score as f64 / 100.0))
            .sum::<f64>()
            * self.resource_weight;
        step_term + risk_term + duration_term + resource_term
    }

    /// Per-objective cost vector in order: [Steps, Risk, Duration, ResourceCost].
    pub fn evaluate_multi(&self, plan: &DeploymentPlan) -> Vec<f64> {
        vec![
            plan.step_count() as f64 * self.step_cost,
            plan.total_risk as f64 * self.risk_weight,
            plan.total_duration_secs as f64 * self.duration_weight,
            plan.steps
                .iter()
                .map(|s| s.estimated_duration_secs as f64 * (1.0 + s.risk_score as f64 / 100.0))
                .sum::<f64>()
                * self.resource_weight,
        ]
    }
}

impl Default for CostModel {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ParetoFrontier
// ---------------------------------------------------------------------------

/// Maintains a set of Pareto-non-dominated plans.
///
/// A cost vector **a** dominates **b** iff every component of **a** ≤ the
/// corresponding component of **b**, and at least one is strictly less.
#[derive(Debug, Clone)]
pub struct ParetoFrontier {
    entries: Vec<(DeploymentPlan, Vec<f64>)>,
}

impl ParetoFrontier {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Returns `true` if **a** dominates **b**.
    pub fn dominates(a: &[f64], b: &[f64]) -> bool {
        if a.len() != b.len() || a.is_empty() {
            return false;
        }
        let mut at_least_one_strict = false;
        for (ai, bi) in a.iter().zip(b.iter()) {
            if ai > bi {
                return false;
            }
            if ai < bi {
                at_least_one_strict = true;
            }
        }
        at_least_one_strict
    }

    /// Returns `true` if any entry in the current frontier dominates `costs`.
    pub fn is_dominated(&self, costs: &[f64]) -> bool {
        self.entries.iter().any(|(_, c)| Self::dominates(c, costs))
    }

    /// Attempt to add `plan` with the given `costs` to the frontier.
    ///
    /// Returns `true` if the plan was added (i.e. it is non-dominated).
    /// Any existing entries that the new plan dominates are removed.
    pub fn add(&mut self, plan: DeploymentPlan, costs: Vec<f64>) -> bool {
        if self.is_dominated(&costs) {
            return false;
        }

        // Remove entries dominated by the new plan.
        self.entries
            .retain(|(_, existing)| !Self::dominates(&costs, existing));

        self.entries.push((plan, costs));
        true
    }

    /// View of the current frontier.
    pub fn frontier(&self) -> Vec<(&DeploymentPlan, &Vec<f64>)> {
        self.entries.iter().map(|(p, c)| (p, c)).collect()
    }

    pub fn size(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Consume the frontier and return owned plans + costs.
    pub fn into_plans(self) -> Vec<DeploymentPlan> {
        self.entries.into_iter().map(|(p, _)| p).collect()
    }
}

impl Default for ParetoFrontier {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// StepMerger
// ---------------------------------------------------------------------------

/// Identifies independent steps within a deployment plan that can safely
/// execute in parallel (they touch disjoint services with no ordering
/// dependency).
#[derive(Debug, Clone)]
pub struct StepMerger {
    _priv: (),
}

impl StepMerger {
    pub fn new() -> Self {
        Self { _priv: () }
    }

    /// Partition plan steps into groups of indices that can execute in
    /// parallel. Groups are produced in topological order: all steps in
    /// group *k* must complete before any step in group *k+1* begins.
    pub fn find_independent_steps(&self, plan: &DeploymentPlan) -> Vec<Vec<usize>> {
        let steps = &plan.steps;
        if steps.is_empty() {
            return Vec::new();
        }

        // Build a dependency graph over step indices.
        // Step j depends on step i if:
        //   - They operate on the same service AND i < j, OR
        //   - Step j reads from_version that is produced by step i on the same service.
        let n = steps.len();
        let mut deps: Vec<HashSet<usize>> = vec![HashSet::new(); n];

        for j in 0..n {
            for i in 0..j {
                if steps_depend(steps, i, j) {
                    deps[j].insert(i);
                }
            }
        }

        // Greedy topological layering (Kahn-like).
        let mut remaining_deps: Vec<usize> = deps.iter().map(|d| d.len()).collect();
        let mut scheduled = vec![false; n];
        let mut groups: Vec<Vec<usize>> = Vec::new();

        loop {
            let ready: Vec<usize> = (0..n)
                .filter(|&i| !scheduled[i] && remaining_deps[i] == 0)
                .collect();
            if ready.is_empty() {
                break;
            }
            for &idx in &ready {
                scheduled[idx] = true;
                for j in 0..n {
                    if deps[j].contains(&idx) {
                        remaining_deps[j] = remaining_deps[j].saturating_sub(1);
                    }
                }
            }
            groups.push(ready);
        }

        groups
    }

    /// Return groups of [`PlanStep`]s that can execute in parallel, preserving
    /// the topological ordering between groups.
    pub fn merge(&self, plan: &DeploymentPlan) -> Vec<Vec<PlanStep>> {
        self.find_independent_steps(plan)
            .into_iter()
            .map(|group| group.into_iter().map(|i| plan.steps[i].clone()).collect())
            .collect()
    }
}

impl Default for StepMerger {
    fn default() -> Self {
        Self::new()
    }
}

/// Two steps are dependent iff they touch the same service.
fn steps_depend(steps: &[PlanStep], i: usize, j: usize) -> bool {
    steps[i].service == steps[j].service
}

// ---------------------------------------------------------------------------
// PathShortener
// ---------------------------------------------------------------------------

/// Uses BFS over the version-product graph to find shorter step sequences
/// between two states while respecting all constraints.
#[derive(Debug)]
pub struct PathShortener<'a> {
    graph: &'a VersionProductGraph,
    constraints: &'a [Constraint],
}

impl<'a> PathShortener<'a> {
    pub fn new(graph: &'a VersionProductGraph, constraints: &'a [Constraint]) -> Self {
        Self { graph, constraints }
    }

    /// BFS from `start` to `target`, returning the shortest plan-step
    /// sequence that satisfies every constraint at each intermediate state.
    #[instrument(skip(self), fields(start = %start, target = %target))]
    pub fn shorten(&self, start: &State, target: &State) -> Option<Vec<PlanStep>> {
        if start == target {
            return Some(Vec::new());
        }

        let mut visited: HashSet<State> = HashSet::new();
        // Map from state → (predecessor state, edge index in graph).
        let mut came_from: HashMap<State, (State, usize)> = HashMap::new();
        let mut queue: VecDeque<State> = VecDeque::new();

        visited.insert(start.clone());
        queue.push_back(start.clone());

        while let Some(current) = queue.pop_front() {
            for (neighbor, edge_idx) in self.graph.neighbors(&current) {
                if visited.contains(&neighbor) {
                    continue;
                }
                // Check all state-level constraints on the neighbor.
                if !self.satisfies_constraints(&neighbor) {
                    continue;
                }
                came_from.insert(neighbor.clone(), (current.clone(), edge_idx));
                if neighbor == *target {
                    return Some(self.reconstruct(start, target, &came_from));
                }
                visited.insert(neighbor.clone());
                queue.push_back(neighbor);
            }
        }

        trace!("BFS: no path found from {} to {}", start, target);
        None
    }

    /// Try to shorten an existing segment of steps between `from` and `to`.
    /// Returns `Some(shorter)` only if the BFS path is strictly shorter than
    /// `current_steps`.
    pub fn shorten_segment(
        &self,
        from: &State,
        to: &State,
        current_steps: &[PlanStep],
    ) -> Option<Vec<PlanStep>> {
        let candidate = self.shorten(from, to)?;
        if candidate.len() < current_steps.len() {
            Some(candidate)
        } else {
            None
        }
    }

    // -- helpers --

    fn satisfies_constraints(&self, state: &State) -> bool {
        self.constraints.iter().all(|c| c.check_state(state))
    }

    fn reconstruct(
        &self,
        start: &State,
        target: &State,
        came_from: &HashMap<State, (State, usize)>,
    ) -> Vec<PlanStep> {
        let mut path_edges: Vec<usize> = Vec::new();
        let mut cursor = target.clone();
        while cursor != *start {
            let (prev, edge_idx) = came_from.get(&cursor).expect("BFS path broken");
            path_edges.push(*edge_idx);
            cursor = prev.clone();
        }
        path_edges.reverse();

        path_edges
            .into_iter()
            .map(|eidx| {
                let edge = &self.graph.edges[eidx];
                PlanStep::new(edge.service, edge.from_version, edge.to_version)
                    .with_risk(edge.metadata.risk_score)
                    .with_duration(edge.metadata.estimated_duration_secs)
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// PlanOptimizer
// ---------------------------------------------------------------------------

/// The main optimiser entry-point. Wraps a graph and a set of constraints and
/// exposes several optimisation strategies that can be composed.
#[derive(Debug)]
pub struct PlanOptimizer<'a> {
    graph: &'a VersionProductGraph,
    constraints: &'a [Constraint],
}

impl<'a> PlanOptimizer<'a> {
    pub fn new(graph: &'a VersionProductGraph, constraints: &'a [Constraint]) -> Self {
        Self { graph, constraints }
    }

    // -----------------------------------------------------------------------
    // minimize_steps
    // -----------------------------------------------------------------------

    /// Remove redundant steps (round-trips where a service version changes
    /// back) and attempt to shorten the remaining path via BFS.
    #[instrument(skip(self, plan), fields(plan_id = %plan.id, original_steps = plan.step_count()))]
    pub fn minimize_steps(&self, plan: &DeploymentPlan) -> CoreResult<DeploymentPlan> {
        debug!("minimize_steps: starting with {} steps", plan.step_count());

        // Phase 1: remove round-trip no-ops.
        let compacted = self.remove_roundtrips(plan);

        // Phase 2: try to BFS-shorten the whole plan.
        let shortener = PathShortener::new(self.graph, self.constraints);
        let bfs_path = shortener.shorten(&plan.start, &plan.target);

        let best_steps = match bfs_path {
            Some(ref bfs) if bfs.len() < compacted.len() => bfs.clone(),
            _ => compacted,
        };

        // Phase 3: try to BFS-shorten segments between consecutive
        // intermediate states of the compacted plan.
        let segment_optimised = self.optimise_segments(&plan.start, &best_steps);

        let result = DeploymentPlan::new(
            plan.start.clone(),
            plan.target.clone(),
            segment_optimised,
        );

        debug!(
            "minimize_steps: reduced from {} to {} steps",
            plan.step_count(),
            result.step_count()
        );
        Ok(result)
    }

    // -----------------------------------------------------------------------
    // minimize_risk
    // -----------------------------------------------------------------------

    /// Reorder steps to minimise time spent in PNR or unsafe states.
    ///
    /// Strategy: generate permutations of the plan that are consistent (each
    /// step's `from_version` matches the current state) and pick the one with
    /// the lowest sum of risk at PNR/outside-envelope states. For plans with
    /// many steps the search is bounded to avoid combinatorial explosion.
    #[instrument(skip(self, plan, envelope), fields(plan_id = %plan.id))]
    pub fn minimize_risk(
        &self,
        plan: &DeploymentPlan,
        envelope: &SafetyEnvelope,
    ) -> CoreResult<DeploymentPlan> {
        debug!("minimize_risk: starting with {} steps", plan.step_count());

        if plan.step_count() <= 1 {
            return Ok(plan.clone());
        }

        // Collect services that need changing and the required version deltas.
        let required_changes: Vec<(ServiceIndex, VersionIndex, VersionIndex)> = plan
            .steps
            .iter()
            .map(|s| (s.service, s.from_version, s.to_version))
            .collect();

        // Group changes by service; keep the final from → to per service.
        let mut net_changes: HashMap<ServiceIndex, (VersionIndex, VersionIndex)> = HashMap::new();
        let mut current = plan.start.clone();
        for step in &plan.steps {
            let entry = net_changes
                .entry(step.service)
                .or_insert((step.from_version, step.to_version));
            entry.1 = step.to_version;
            current.set(step.service, step.to_version);
        }

        // Build candidate orderings via a bounded DFS of valid permutations.
        let services_to_move: Vec<ServiceIndex> = net_changes.keys().copied().collect();
        let mut best_steps = plan.steps.clone();
        let mut best_risk_exposure = self.compute_risk_exposure(&plan.start, &plan.steps, envelope);

        let step_map = self.build_step_lookup(&plan.steps);

        let mut stack: Vec<(State, Vec<PlanStep>, HashSet<ServiceIndex>)> = Vec::new();
        stack.push((plan.start.clone(), Vec::new(), HashSet::new()));

        let max_candidates = 5000usize;
        let mut candidates_evaluated = 0usize;

        while let Some((state, path, visited_svcs)) = stack.pop() {
            if candidates_evaluated >= max_candidates {
                break;
            }

            if visited_svcs.len() == services_to_move.len() {
                let exposure = self.compute_risk_exposure(&plan.start, &path, envelope);
                if exposure < best_risk_exposure {
                    best_risk_exposure = exposure;
                    best_steps = path.clone();
                }
                candidates_evaluated += 1;
                continue;
            }

            for &svc in &services_to_move {
                if visited_svcs.contains(&svc) {
                    continue;
                }

                // Collect all steps for this service in order.
                let svc_steps: Vec<&PlanStep> = step_map
                    .get(&svc)
                    .map(|v| v.iter().collect())
                    .unwrap_or_default();

                // Try appending this service's steps.
                let mut next_state = state.clone();
                let mut next_path = path.clone();
                let mut valid = true;

                for step in &svc_steps {
                    if next_state.get(step.service) != step.from_version {
                        valid = false;
                        break;
                    }
                    next_state.set(step.service, step.to_version);
                    next_path.push((*step).clone());
                }

                if !valid {
                    continue;
                }

                // Check constraints on each new intermediate state.
                if !self.all_intermediates_valid(&state, &svc_steps) {
                    continue;
                }

                let mut next_visited = visited_svcs.clone();
                next_visited.insert(svc);
                stack.push((next_state, next_path, next_visited));
            }
        }

        let result = DeploymentPlan::new(plan.start.clone(), plan.target.clone(), best_steps);
        debug!(
            "minimize_risk: risk exposure {} -> {}",
            self.compute_risk_exposure(&plan.start, &plan.steps, envelope),
            best_risk_exposure
        );
        Ok(result)
    }

    // -----------------------------------------------------------------------
    // pareto_optimize
    // -----------------------------------------------------------------------

    /// Compute the Pareto-optimal frontier over the given objectives.
    ///
    /// Generates candidate plans via step-minimisation, risk-minimisation (if
    /// an empty envelope is acceptable), different orderings, and BFS paths,
    /// then keeps only the non-dominated set.
    #[instrument(skip(self, plan, objectives), fields(plan_id = %plan.id))]
    pub fn pareto_optimize(
        &self,
        plan: &DeploymentPlan,
        objectives: &[OptimizationObjective],
    ) -> Vec<DeploymentPlan> {
        let mut frontier = ParetoFrontier::new();

        // Evaluate and add the original plan.
        let orig_costs = self.evaluate_objectives(plan, objectives, None);
        frontier.add(plan.clone(), orig_costs);

        // Candidate 1: step-minimised.
        if let Ok(minimised) = self.minimize_steps(plan) {
            let costs = self.evaluate_objectives(&minimised, objectives, None);
            frontier.add(minimised, costs);
        }

        // Candidate 2: compacted (no-op removal only).
        if let Ok(compacted) = self.compact(plan) {
            let costs = self.evaluate_objectives(&compacted, objectives, None);
            frontier.add(compacted, costs);
        }

        // Candidate 3: risk-minimised with an empty envelope (no PNR info).
        let empty_env = SafetyEnvelope::new();
        if let Ok(risk_min) = self.minimize_risk(plan, &empty_env) {
            let costs = self.evaluate_objectives(&risk_min, objectives, None);
            frontier.add(risk_min, costs);
        }

        // Candidate 4: generate several random-order valid permutations.
        let permutation_candidates = self.generate_ordering_candidates(plan, 20);
        for candidate in permutation_candidates {
            let costs = self.evaluate_objectives(&candidate, objectives, None);
            frontier.add(candidate, costs);
        }

        // Candidate 5: BFS shortest path.
        let shortener = PathShortener::new(self.graph, self.constraints);
        if let Some(bfs_steps) = shortener.shorten(&plan.start, &plan.target) {
            let bfs_plan = DeploymentPlan::new(
                plan.start.clone(),
                plan.target.clone(),
                bfs_steps,
            );
            let costs = self.evaluate_objectives(&bfs_plan, objectives, None);
            frontier.add(bfs_plan, costs);
        }

        debug!("pareto_optimize: frontier size = {}", frontier.size());
        frontier.into_plans()
    }

    // -----------------------------------------------------------------------
    // compact
    // -----------------------------------------------------------------------

    /// Remove no-op steps (where `from_version == to_version`).
    pub fn compact(&self, plan: &DeploymentPlan) -> CoreResult<DeploymentPlan> {
        let kept: Vec<PlanStep> = plan
            .steps
            .iter()
            .filter(|s| s.from_version != s.to_version)
            .cloned()
            .collect();

        Ok(DeploymentPlan::new(
            plan.start.clone(),
            plan.target.clone(),
            kept,
        ))
    }

    // -----------------------------------------------------------------------
    // internal helpers
    // -----------------------------------------------------------------------

    /// Remove round-trip pairs where a service's version changes and later
    /// reverts. Returns the compacted step list.
    fn remove_roundtrips(&self, plan: &DeploymentPlan) -> Vec<PlanStep> {
        let mut steps = plan.steps.clone();
        let mut changed = true;

        while changed {
            changed = false;
            let mut i = 0;
            while i + 1 < steps.len() {
                if steps[i].service == steps[i + 1].service
                    && steps[i].from_version == steps[i + 1].to_version
                    && steps[i].to_version == steps[i + 1].from_version
                {
                    // Adjacent round-trip: remove both.
                    steps.remove(i + 1);
                    steps.remove(i);
                    changed = true;
                    continue;
                }
                i += 1;
            }
        }

        // Second pass: remove steps that are net no-ops after the whole plan
        // (service ends at the same version it started).
        let mut net_version: HashMap<ServiceIndex, VersionIndex> = HashMap::new();
        let mut current = plan.start.clone();
        for s in &steps {
            current.set(s.service, s.to_version);
        }
        for (i, v) in plan.start.versions.iter().enumerate() {
            let svc = ServiceIndex(i as u16);
            net_version.insert(svc, *v);
        }

        // Track which services actually change.
        let diff = plan.start.diff_services(&plan.target);
        let changing_set: HashSet<ServiceIndex> = diff.into_iter().collect();

        // Remove steps for services that are not in the diff set and that
        // create round-trips within the plan.
        let kept: Vec<PlanStep> = steps
            .into_iter()
            .filter(|s| {
                if changing_set.contains(&s.service) {
                    return true;
                }
                // If the service doesn't ultimately change, skip
                // intermediate steps for it.
                false
            })
            .collect();

        // Verify consistency; fall back to the original if something broke.
        let test_plan = DeploymentPlan::new(plan.start.clone(), plan.target.clone(), kept.clone());
        if test_plan.validate_consistency() {
            kept
        } else {
            plan.steps.clone()
        }
    }

    /// Try to BFS-shorten consecutive segments of the step list.
    fn optimise_segments(&self, start: &State, steps: &[PlanStep]) -> Vec<PlanStep> {
        if steps.len() <= 2 {
            return steps.to_vec();
        }

        let shortener = PathShortener::new(self.graph, self.constraints);

        // Build intermediate states.
        let mut intermediates = vec![start.clone()];
        let mut cur = start.clone();
        for s in steps {
            cur.set(s.service, s.to_version);
            intermediates.push(cur.clone());
        }

        // Try pairwise segment shortening with a sliding window.
        let mut result = steps.to_vec();
        let window_sizes = [4, 3, 2];

        for &window in &window_sizes {
            if result.len() < window {
                continue;
            }
            let mut new_result = Vec::new();
            let mut idx = 0;
            let mut cur_state = start.clone();

            while idx < result.len() {
                let end = (idx + window).min(result.len());
                let segment = &result[idx..end];

                let mut seg_end_state = cur_state.clone();
                for s in segment {
                    seg_end_state.set(s.service, s.to_version);
                }

                if let Some(shorter) =
                    shortener.shorten_segment(&cur_state, &seg_end_state, segment)
                {
                    for s in &shorter {
                        cur_state.set(s.service, s.to_version);
                    }
                    new_result.extend(shorter);
                    idx = end;
                } else {
                    let s = result[idx].clone();
                    cur_state.set(s.service, s.to_version);
                    new_result.push(s);
                    idx += 1;
                }
            }

            result = new_result;
        }

        result
    }

    /// Score how many intermediate states are PNR or outside the envelope.
    fn compute_risk_exposure(
        &self,
        start: &State,
        steps: &[PlanStep],
        envelope: &SafetyEnvelope,
    ) -> f64 {
        let mut current = start.clone();
        let mut exposure = 0.0;

        for step in steps {
            current.set(step.service, step.to_version);
            if envelope.is_pnr(&current) {
                exposure += 3.0 * step.risk_score as f64;
            } else if !envelope.is_safe(&current) {
                exposure += step.risk_score as f64;
            }
        }

        exposure
    }

    /// Group plan steps by service index.
    fn build_step_lookup(&self, steps: &[PlanStep]) -> HashMap<ServiceIndex, Vec<PlanStep>> {
        let mut map: HashMap<ServiceIndex, Vec<PlanStep>> = HashMap::new();
        for s in steps {
            map.entry(s.service).or_default().push(s.clone());
        }
        map
    }

    /// Check that appending a set of steps to the current state produces only
    /// valid (constraint-satisfying) intermediate states.
    fn all_intermediates_valid(&self, start: &State, steps: &[&PlanStep]) -> bool {
        let mut current = start.clone();
        for step in steps {
            current.set(step.service, step.to_version);
            if !self.constraints.iter().all(|c| c.check_state(&current)) {
                return false;
            }
        }
        true
    }

    /// Evaluate a plan over a set of objectives, returning a cost vector.
    fn evaluate_objectives(
        &self,
        plan: &DeploymentPlan,
        objectives: &[OptimizationObjective],
        envelope: Option<&SafetyEnvelope>,
    ) -> Vec<f64> {
        objectives
            .iter()
            .map(|obj| obj.evaluate(plan, envelope))
            .collect()
    }

    /// Generate up to `max` valid ordering candidates of the plan's steps.
    ///
    /// We try every permutation of the *per-service groups* rather than
    /// individual steps, which keeps the search bounded.
    fn generate_ordering_candidates(
        &self,
        plan: &DeploymentPlan,
        max: usize,
    ) -> Vec<DeploymentPlan> {
        let step_groups = self.build_step_lookup(&plan.steps);
        let service_order: Vec<ServiceIndex> = step_groups.keys().copied().collect();

        if service_order.is_empty() {
            return Vec::new();
        }

        let mut candidates = Vec::new();
        let mut permutation_buf: Vec<Vec<ServiceIndex>> = Vec::new();
        generate_permutations(&service_order, &mut permutation_buf, max);

        for perm in permutation_buf {
            let mut steps = Vec::new();
            let mut state = plan.start.clone();
            let mut valid = true;

            for svc in &perm {
                if let Some(svc_steps) = step_groups.get(svc) {
                    for step in svc_steps {
                        if state.get(step.service) != step.from_version {
                            valid = false;
                            break;
                        }
                        state.set(step.service, step.to_version);
                        steps.push(step.clone());
                    }
                }
                if !valid {
                    break;
                }
            }

            if valid && state == plan.target {
                let candidate =
                    DeploymentPlan::new(plan.start.clone(), plan.target.clone(), steps);
                if candidate.validate_consistency() {
                    candidates.push(candidate);
                }
            }

            if candidates.len() >= max {
                break;
            }
        }

        candidates
    }
}

// ---------------------------------------------------------------------------
// Utility: bounded permutation generation
// ---------------------------------------------------------------------------

/// Generate up to `max` permutations of `items` using iterative
/// Heap's algorithm, collecting results into `out`.
fn generate_permutations<T: Clone>(items: &[T], out: &mut Vec<Vec<T>>, max: usize) {
    let n = items.len();
    if n == 0 {
        return;
    }

    let mut arr: Vec<T> = items.to_vec();
    let mut c = vec![0usize; n];
    out.push(arr.clone());

    let mut i = 0;
    while i < n && out.len() < max {
        if c[i] < i {
            if i % 2 == 0 {
                arr.swap(0, i);
            } else {
                arr.swap(c[i], i);
            }
            out.push(arr.clone());
            c[i] += 1;
            i = 0;
        } else {
            c[i] = 0;
            i += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Weighted A* node for risk-aware shortest path (used internally)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct AStarNode {
    state: State,
    g_cost: OrderedFloat<f64>,
    f_cost: OrderedFloat<f64>,
}

impl Eq for AStarNode {}

impl PartialEq for AStarNode {
    fn eq(&self, other: &Self) -> bool {
        self.f_cost == other.f_cost
    }
}

impl Ord for AStarNode {
    fn cmp(&self, other: &Self) -> CmpOrdering {
        // Reverse order for min-heap behaviour via BinaryHeap.
        other.f_cost.cmp(&self.f_cost)
    }
}

impl PartialOrd for AStarNode {
    fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
        Some(self.cmp(other))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Edge, ServiceDescriptor, TransitionMetadata, VersionProductGraph,
    };

    /// Build a simple 2-service, 2-version graph:
    ///
    /// Service 0: v0 ↔ v1
    /// Service 1: v0 ↔ v1
    ///
    /// States: (0,0), (0,1), (1,0), (1,1) — fully connected through
    /// single-service edges.
    fn make_test_graph() -> VersionProductGraph {
        let svc0 = ServiceDescriptor::new("alpha", vec!["0.1".into(), "0.2".into()]);
        let svc1 = ServiceDescriptor::new("beta", vec!["1.0".into(), "2.0".into()]);

        let mut graph = VersionProductGraph::new(vec![svc0, svc1]);

        let states: Vec<State> = vec![
            State::new(vec![VersionIndex(0), VersionIndex(0)]),
            State::new(vec![VersionIndex(0), VersionIndex(1)]),
            State::new(vec![VersionIndex(1), VersionIndex(0)]),
            State::new(vec![VersionIndex(1), VersionIndex(1)]),
        ];

        for s in &states {
            graph.add_state(s.clone());
        }

        // Service 0 transitions in each state-pair.
        let meta_low = TransitionMetadata {
            is_upgrade: true,
            risk_score: 10,
            estimated_duration_secs: 30,
            requires_downtime: false,
        };
        let meta_high = TransitionMetadata {
            is_upgrade: true,
            risk_score: 50,
            estimated_duration_secs: 120,
            requires_downtime: true,
        };

        // (0,0) → (1,0)
        graph.add_edge(Edge {
            from: states[0].clone(),
            to: states[2].clone(),
            service: ServiceIndex(0),
            from_version: VersionIndex(0),
            to_version: VersionIndex(1),
            metadata: meta_low.clone(),
        });
        // (1,0) → (0,0)
        graph.add_edge(Edge {
            from: states[2].clone(),
            to: states[0].clone(),
            service: ServiceIndex(0),
            from_version: VersionIndex(1),
            to_version: VersionIndex(0),
            metadata: meta_low.clone(),
        });
        // (0,1) → (1,1)
        graph.add_edge(Edge {
            from: states[1].clone(),
            to: states[3].clone(),
            service: ServiceIndex(0),
            from_version: VersionIndex(0),
            to_version: VersionIndex(1),
            metadata: meta_low.clone(),
        });
        // (1,1) → (0,1)
        graph.add_edge(Edge {
            from: states[3].clone(),
            to: states[1].clone(),
            service: ServiceIndex(0),
            from_version: VersionIndex(1),
            to_version: VersionIndex(0),
            metadata: meta_low.clone(),
        });

        // Service 1 transitions.
        // (0,0) → (0,1)
        graph.add_edge(Edge {
            from: states[0].clone(),
            to: states[1].clone(),
            service: ServiceIndex(1),
            from_version: VersionIndex(0),
            to_version: VersionIndex(1),
            metadata: meta_high.clone(),
        });
        // (0,1) → (0,0)
        graph.add_edge(Edge {
            from: states[1].clone(),
            to: states[0].clone(),
            service: ServiceIndex(1),
            from_version: VersionIndex(1),
            to_version: VersionIndex(0),
            metadata: meta_high.clone(),
        });
        // (1,0) → (1,1)
        graph.add_edge(Edge {
            from: states[2].clone(),
            to: states[3].clone(),
            service: ServiceIndex(1),
            from_version: VersionIndex(0),
            to_version: VersionIndex(1),
            metadata: meta_high.clone(),
        });
        // (1,1) → (1,0)
        graph.add_edge(Edge {
            from: states[3].clone(),
            to: states[2].clone(),
            service: ServiceIndex(1),
            from_version: VersionIndex(1),
            to_version: VersionIndex(0),
            metadata: meta_high.clone(),
        });

        graph
    }

    fn make_constraints() -> Vec<Constraint> {
        Vec::new()
    }

    fn make_simple_plan() -> DeploymentPlan {
        let start = State::new(vec![VersionIndex(0), VersionIndex(0)]);
        let target = State::new(vec![VersionIndex(1), VersionIndex(1)]);
        let steps = vec![
            PlanStep::new(ServiceIndex(0), VersionIndex(0), VersionIndex(1))
                .with_risk(10)
                .with_duration(30),
            PlanStep::new(ServiceIndex(1), VersionIndex(0), VersionIndex(1))
                .with_risk(50)
                .with_duration(120),
        ];
        DeploymentPlan::new(start, target, steps)
    }

    // -- tests --

    #[test]
    fn test_pareto_dominates() {
        assert!(ParetoFrontier::dominates(&[1.0, 2.0], &[2.0, 3.0]));
        assert!(ParetoFrontier::dominates(&[1.0, 2.0], &[1.0, 3.0]));
        assert!(!ParetoFrontier::dominates(&[1.0, 2.0], &[1.0, 2.0]));
        assert!(!ParetoFrontier::dominates(&[1.0, 3.0], &[2.0, 2.0]));
        assert!(!ParetoFrontier::dominates(&[], &[]));
    }

    #[test]
    fn test_pareto_frontier_add_and_prune() {
        let mut frontier = ParetoFrontier::new();
        let plan_a = make_simple_plan();
        let plan_b = make_simple_plan();
        let plan_c = make_simple_plan();

        assert!(frontier.add(plan_a, vec![3.0, 5.0]));
        assert_eq!(frontier.size(), 1);

        // plan_b dominates plan_a.
        assert!(frontier.add(plan_b, vec![2.0, 4.0]));
        assert_eq!(frontier.size(), 1);

        // plan_c is non-dominated (trade-off).
        assert!(frontier.add(plan_c, vec![4.0, 1.0]));
        assert_eq!(frontier.size(), 2);
    }

    #[test]
    fn test_pareto_frontier_reject_dominated() {
        let mut frontier = ParetoFrontier::new();
        let plan_a = make_simple_plan();
        let plan_b = make_simple_plan();

        frontier.add(plan_a, vec![1.0, 1.0]);
        // plan_b is dominated by plan_a.
        assert!(!frontier.add(plan_b, vec![2.0, 3.0]));
        assert_eq!(frontier.size(), 1);
    }

    #[test]
    fn test_cost_model_evaluate() {
        let plan = make_simple_plan();
        let model = CostModel::new()
            .with_step_cost(10.0)
            .with_risk_weight(2.0)
            .with_duration_weight(0.5);

        let cost = model.evaluate(&plan);
        // step_term = 2 * 10 = 20
        // risk_term = 60 * 2 = 120
        // duration_term = 150 * 0.5 = 75
        // resource_term computed from per-step formula
        assert!(cost > 0.0);

        let multi = model.evaluate_multi(&plan);
        assert_eq!(multi.len(), 4);
        assert!((multi[0] - 20.0).abs() < 1e-9);
    }

    #[test]
    fn test_step_merger_independent() {
        let plan = make_simple_plan();
        let merger = StepMerger::new();
        let groups = merger.find_independent_steps(&plan);

        // Two steps on different services → one parallel group.
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].len(), 2);
    }

    #[test]
    fn test_step_merger_dependent() {
        let start = State::new(vec![VersionIndex(0), VersionIndex(0)]);
        let target = State::new(vec![VersionIndex(1), VersionIndex(0)]);
        let steps = vec![
            PlanStep::new(ServiceIndex(0), VersionIndex(0), VersionIndex(1))
                .with_risk(10)
                .with_duration(30),
            PlanStep::new(ServiceIndex(0), VersionIndex(1), VersionIndex(0))
                .with_risk(10)
                .with_duration(30),
        ];
        let plan = DeploymentPlan::new(start, target, steps);
        let merger = StepMerger::new();
        let groups = merger.find_independent_steps(&plan);

        // Both steps on the same service → two sequential groups.
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].len(), 1);
        assert_eq!(groups[1].len(), 1);
    }

    #[test]
    fn test_path_shortener_identity() {
        let graph = make_test_graph();
        let constraints = make_constraints();
        let shortener = PathShortener::new(&graph, &constraints);

        let state = State::new(vec![VersionIndex(0), VersionIndex(0)]);
        let path = shortener.shorten(&state, &state);
        assert!(path.is_some());
        assert!(path.unwrap().is_empty());
    }

    #[test]
    fn test_path_shortener_finds_path() {
        let graph = make_test_graph();
        let constraints = make_constraints();
        let shortener = PathShortener::new(&graph, &constraints);

        let start = State::new(vec![VersionIndex(0), VersionIndex(0)]);
        let target = State::new(vec![VersionIndex(1), VersionIndex(1)]);
        let path = shortener.shorten(&start, &target);

        assert!(path.is_some());
        let steps = path.unwrap();
        // Shortest is 2 steps (one per service).
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn test_minimize_steps_removes_roundtrip() {
        let graph = make_test_graph();
        let constraints = make_constraints();
        let optimizer = PlanOptimizer::new(&graph, &constraints);

        let start = State::new(vec![VersionIndex(0), VersionIndex(0)]);
        let target = State::new(vec![VersionIndex(1), VersionIndex(1)]);
        // Intentionally include a round-trip on service 0.
        let steps = vec![
            PlanStep::new(ServiceIndex(0), VersionIndex(0), VersionIndex(1))
                .with_risk(10)
                .with_duration(30),
            PlanStep::new(ServiceIndex(0), VersionIndex(1), VersionIndex(0))
                .with_risk(10)
                .with_duration(30),
            PlanStep::new(ServiceIndex(0), VersionIndex(0), VersionIndex(1))
                .with_risk(10)
                .with_duration(30),
            PlanStep::new(ServiceIndex(1), VersionIndex(0), VersionIndex(1))
                .with_risk(50)
                .with_duration(120),
        ];
        let plan = DeploymentPlan::new(start, target, steps);
        assert_eq!(plan.step_count(), 4);

        let optimised = optimizer.minimize_steps(&plan).unwrap();
        assert!(optimised.step_count() <= 2);
        assert!(optimised.validate_consistency());
    }

    #[test]
    fn test_compact_removes_noops() {
        let graph = make_test_graph();
        let constraints = make_constraints();
        let optimizer = PlanOptimizer::new(&graph, &constraints);

        let start = State::new(vec![VersionIndex(0), VersionIndex(0)]);
        let target = State::new(vec![VersionIndex(1), VersionIndex(0)]);
        let steps = vec![
            PlanStep::new(ServiceIndex(0), VersionIndex(0), VersionIndex(1))
                .with_risk(10)
                .with_duration(30),
            // No-op: same from and to.
            PlanStep::new(ServiceIndex(1), VersionIndex(0), VersionIndex(0))
                .with_risk(0)
                .with_duration(0),
        ];
        let plan = DeploymentPlan::new(start, target, steps);
        let compacted = optimizer.compact(&plan).unwrap();

        assert_eq!(compacted.step_count(), 1);
        assert_eq!(compacted.steps[0].service, ServiceIndex(0));
    }

    #[test]
    fn test_pareto_optimize_returns_nonempty() {
        let graph = make_test_graph();
        let constraints = make_constraints();
        let optimizer = PlanOptimizer::new(&graph, &constraints);
        let plan = make_simple_plan();

        let objectives = vec![OptimizationObjective::Steps, OptimizationObjective::Risk];
        let frontier = optimizer.pareto_optimize(&plan, &objectives);

        assert!(!frontier.is_empty());
        for p in &frontier {
            assert!(p.validate_consistency());
        }
    }

    #[test]
    fn test_optimization_objective_evaluate() {
        let plan = make_simple_plan();

        let step_cost = OptimizationObjective::Steps.evaluate(&plan, None);
        assert!((step_cost - 2.0).abs() < 1e-9);

        let risk_cost = OptimizationObjective::Risk.evaluate(&plan, None);
        assert!((risk_cost - 60.0).abs() < 1e-9);

        let dur_cost = OptimizationObjective::Duration.evaluate(&plan, None);
        assert!((dur_cost - 150.0).abs() < 1e-9);
    }

    #[test]
    fn test_minimize_risk_basic() {
        let graph = make_test_graph();
        let constraints = make_constraints();
        let optimizer = PlanOptimizer::new(&graph, &constraints);
        let plan = make_simple_plan();

        let mut envelope = SafetyEnvelope::new();
        // Mark all states as safe.
        envelope.safe_states.push(State::new(vec![VersionIndex(0), VersionIndex(0)]));
        envelope.safe_states.push(State::new(vec![VersionIndex(1), VersionIndex(0)]));
        envelope.safe_states.push(State::new(vec![VersionIndex(0), VersionIndex(1)]));
        envelope.safe_states.push(State::new(vec![VersionIndex(1), VersionIndex(1)]));

        let result = optimizer.minimize_risk(&plan, &envelope).unwrap();
        assert!(result.validate_consistency());
        assert_eq!(result.step_count(), 2);
    }
}
