//! Parallel execution planning for SafeStep deployment plans.
//!
//! Converts sequential deployment plans into parallel plans by analyzing
//! inter-step dependencies, building a dependency DAG, and scheduling
//! independent steps for concurrent execution.

use std::collections::{HashMap, HashSet, VecDeque};

use serde::{Deserialize, Serialize};
use tracing::{debug, info, instrument, warn};

use crate::{
    Constraint, CoreResult, DeploymentPlan, PlanStep, ServiceIndex, State, VersionIndex,
};
use safestep_types::error::SafeStepError;
use safestep_types::identifiers::{PlanId, StepId, Id};

// ---------------------------------------------------------------------------
// ParallelPlanStep — a group of steps that execute concurrently
// ---------------------------------------------------------------------------

/// A group of deployment steps that can safely execute in parallel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelPlanStep {
    pub steps: Vec<PlanStep>,
    /// Maximum risk score among all steps in this group.
    pub group_risk: u32,
    /// Duration of the group equals the slowest (bottleneck) step.
    pub group_duration: u64,
}

impl ParallelPlanStep {
    /// Build a new parallel group, computing aggregate risk and duration.
    pub fn new(steps: Vec<PlanStep>) -> Self {
        let group_risk = steps.iter().map(|s| s.risk_score).max().unwrap_or(0);
        let group_duration = steps
            .iter()
            .map(|s| s.estimated_duration_secs)
            .max()
            .unwrap_or(0);
        Self {
            steps,
            group_risk,
            group_duration,
        }
    }

    /// Number of steps executing concurrently in this group.
    pub fn step_count(&self) -> usize {
        self.steps.len()
    }

    /// Distinct services touched by this group.
    pub fn services(&self) -> Vec<ServiceIndex> {
        let mut seen = HashSet::new();
        self.steps
            .iter()
            .filter_map(|s| {
                if seen.insert(s.service) {
                    Some(s.service)
                } else {
                    None
                }
            })
            .collect()
    }

    /// True when the group contains exactly one step (no actual parallelism).
    pub fn is_single(&self) -> bool {
        self.steps.len() == 1
    }
}

// ---------------------------------------------------------------------------
// ParallelPlan — a deployment plan organized into sequential groups
// ---------------------------------------------------------------------------

/// A deployment plan whose steps have been partitioned into sequential groups,
/// where steps within each group execute concurrently.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelPlan {
    pub groups: Vec<ParallelPlanStep>,
    pub start: State,
    pub target: State,
}

impl ParallelPlan {
    pub fn new(groups: Vec<ParallelPlanStep>, start: State, target: State) -> Self {
        Self {
            groups,
            start,
            target,
        }
    }

    /// Number of sequential groups (the critical-path length).
    pub fn critical_path_length(&self) -> usize {
        self.groups.len()
    }

    /// Total number of individual steps across all groups.
    pub fn total_steps(&self) -> usize {
        self.groups.iter().map(|g| g.step_count()).sum()
    }

    /// Size of the largest parallel group.
    pub fn max_parallelism(&self) -> usize {
        self.groups.iter().map(|g| g.step_count()).max().unwrap_or(0)
    }

    /// Total wall-clock duration (sum of sequential group durations).
    pub fn total_duration(&self) -> u64 {
        self.groups.iter().map(|g| g.group_duration).sum()
    }

    /// Ratio of sequential duration over parallel duration.
    /// Returns 1.0 when no speedup is possible.
    pub fn speedup_over_sequential(&self) -> f64 {
        let parallel_dur = self.total_duration();
        if parallel_dur == 0 {
            return 1.0;
        }
        let sequential_dur: u64 = self
            .groups
            .iter()
            .flat_map(|g| g.steps.iter())
            .map(|s| s.estimated_duration_secs)
            .sum();
        sequential_dur as f64 / parallel_dur as f64
    }

    /// Convert a sequential `DeploymentPlan` into a `ParallelPlan` by
    /// analysing dependencies and grouping independent steps.
    #[instrument(skip_all, fields(plan_id = %plan.id, step_count = plan.step_count()))]
    pub fn parallelize(plan: &DeploymentPlan, constraints: &[Constraint]) -> CoreResult<Self> {
        if plan.steps.is_empty() {
            return Ok(Self::new(Vec::new(), plan.start.clone(), plan.target.clone()));
        }

        let analyzer = DependencyAnalyzer::new();
        let dep_graph = analyzer.compute_dependencies(plan, constraints);

        if !dep_graph.is_acyclic() {
            return Err(SafeStepError::plan_validation(
                "dependency graph contains a cycle",
            ));
        }

        let scheduler = ParallelScheduler::new(usize::MAX);
        let schedule = scheduler.schedule(&dep_graph, &plan.steps);
        let parallel_plan = schedule.to_parallel_plan(plan);

        info!(
            groups = parallel_plan.groups.len(),
            max_par = parallel_plan.max_parallelism(),
            speedup = %format!("{:.2}x", parallel_plan.speedup_over_sequential()),
            "parallelized deployment plan"
        );

        Ok(parallel_plan)
    }
}

// ---------------------------------------------------------------------------
// DependencyGraph — DAG of step-level ordering constraints
// ---------------------------------------------------------------------------

/// Directed acyclic graph encoding which steps must precede which.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyGraph {
    /// Directed edges `(from, to)` meaning step `from` must complete before `to`.
    pub edges: Vec<(usize, usize)>,
    pub step_count: usize,
    /// Forward adjacency list (successors).
    #[serde(skip)]
    fwd: Vec<Vec<usize>>,
    /// Reverse adjacency list (predecessors).
    #[serde(skip)]
    rev: Vec<Vec<usize>>,
}

impl DependencyGraph {
    pub fn new(step_count: usize) -> Self {
        Self {
            edges: Vec::new(),
            step_count,
            fwd: vec![Vec::new(); step_count],
            rev: vec![Vec::new(); step_count],
        }
    }

    /// Record that step `from` must finish before step `to` starts.
    pub fn add_dependency(&mut self, from: usize, to: usize) {
        if from >= self.step_count || to >= self.step_count {
            return;
        }
        if self.has_dependency(from, to) {
            return;
        }
        self.edges.push((from, to));
        self.fwd[from].push(to);
        self.rev[to].push(from);
    }

    /// Check whether a direct edge `from -> to` exists.
    pub fn has_dependency(&self, from: usize, to: usize) -> bool {
        self.fwd
            .get(from)
            .map(|succs| succs.contains(&to))
            .unwrap_or(false)
    }

    /// All steps that must complete before `step`.
    pub fn predecessors(&self, step: usize) -> Vec<usize> {
        self.rev.get(step).cloned().unwrap_or_default()
    }

    /// All steps that depend on `step`.
    pub fn successors(&self, step: usize) -> Vec<usize> {
        self.fwd.get(step).cloned().unwrap_or_default()
    }

    /// Kahn's algorithm: topological sort. Returns `Err` when a cycle exists.
    pub fn topological_sort(&self) -> CoreResult<Vec<usize>> {
        let n = self.step_count;
        let mut in_degree = vec![0u32; n];
        for &(_, to) in &self.edges {
            in_degree[to] += 1;
        }

        let mut queue: VecDeque<usize> = (0..n)
            .filter(|&i| in_degree[i] == 0)
            .collect();

        let mut order = Vec::with_capacity(n);
        while let Some(node) = queue.pop_front() {
            order.push(node);
            for &succ in &self.fwd[node] {
                in_degree[succ] -= 1;
                if in_degree[succ] == 0 {
                    queue.push_back(succ);
                }
            }
        }

        if order.len() == n {
            Ok(order)
        } else {
            Err(SafeStepError::plan_validation(
                "dependency graph contains a cycle",
            ))
        }
    }

    /// Length of the longest path through the DAG (critical-path length).
    /// Uses DP over topological order.
    pub fn longest_path(&self) -> usize {
        let n = self.step_count;
        if n == 0 {
            return 0;
        }

        let order = match self.topological_sort() {
            Ok(o) => o,
            Err(_) => return 0,
        };

        let mut dist = vec![0usize; n];
        for &node in &order {
            for &succ in &self.fwd[node] {
                let candidate = dist[node] + 1;
                if candidate > dist[succ] {
                    dist[succ] = candidate;
                }
            }
        }

        dist.into_iter().max().unwrap_or(0) + 1
    }

    /// True when the graph has no directed cycles (i.e. is a DAG).
    pub fn is_acyclic(&self) -> bool {
        self.topological_sort().is_ok()
    }

    /// Compute the transitive reduction: remove edges implied by longer paths.
    pub fn transitive_reduction(&self) -> Self {
        let n = self.step_count;
        // Compute transitive reachability (excluding direct edge) via BFS.
        let mut reachable = vec![HashSet::<usize>::new(); n];
        let order = match self.topological_sort() {
            Ok(o) => o,
            Err(_) => return self.clone(),
        };

        // Process in reverse topological order so successors are already done.
        for &node in order.iter().rev() {
            for &succ in &self.fwd[node] {
                // `succ` and everything reachable from `succ` are reachable from `node`.
                let downstream: HashSet<usize> = reachable[succ].clone();
                reachable[node].insert(succ);
                reachable[node].extend(downstream);
            }
        }

        let mut reduced = DependencyGraph::new(n);
        for &(from, to) in &self.edges {
            // Keep edge only if `to` is NOT reachable from `from` via other paths.
            let reachable_via_others: bool = self.fwd[from]
                .iter()
                .any(|&mid| mid != to && reachable[mid].contains(&to));
            if !reachable_via_others {
                reduced.add_dependency(from, to);
            }
        }
        reduced
    }
}

// ---------------------------------------------------------------------------
// DependencyAnalyzer — determines which steps depend on which
// ---------------------------------------------------------------------------

/// Analyses a sequential deployment plan to extract inter-step dependencies.
#[derive(Debug, Clone)]
pub struct DependencyAnalyzer {
    /// When true, only ordering & constraint-implied deps are added;
    /// otherwise same-service deps are always added.
    _lenient: bool,
}

impl DependencyAnalyzer {
    pub fn new() -> Self {
        Self { _lenient: false }
    }

    /// Build a `DependencyGraph` for every step in `plan`.
    ///
    /// Rules:
    /// 1. Same-service sequencing: if two steps touch the same service, the
    ///    earlier one must precede the later one.
    /// 2. Ordering constraints: explicit `Constraint::Ordering`.
    /// 3. Compatibility constraints: if swapping the execution order of two
    ///    steps would create an intermediate state that violates a
    ///    `Constraint::Compatibility`, add a dependency to keep original order.
    #[instrument(skip_all, fields(steps = plan.steps.len(), constraints = constraints.len()))]
    pub fn compute_dependencies(
        &self,
        plan: &DeploymentPlan,
        constraints: &[Constraint],
    ) -> DependencyGraph {
        let n = plan.steps.len();
        let mut graph = DependencyGraph::new(n);

        // Rule 1: same-service ordering.
        for i in 0..n {
            for j in (i + 1)..n {
                if plan.steps[i].service == plan.steps[j].service {
                    graph.add_dependency(i, j);
                    debug!(from = i, to = j, svc = %plan.steps[i].service, "same-service dep");
                }
            }
        }

        // Rule 2: explicit ordering constraints.
        for constraint in constraints {
            if let Constraint::Ordering { before, after, .. } = constraint {
                // Find earliest step that upgrades `before` and latest that upgrades `after`.
                let before_idxs: Vec<usize> = (0..n)
                    .filter(|&i| plan.steps[i].service == *before)
                    .collect();
                let after_idxs: Vec<usize> = (0..n)
                    .filter(|&i| plan.steps[i].service == *after)
                    .collect();
                for &bi in &before_idxs {
                    for &ai in &after_idxs {
                        if bi != ai {
                            graph.add_dependency(bi, ai);
                            debug!(from = bi, to = ai, "ordering constraint dep");
                        }
                    }
                }
            }
        }

        // Rule 3: constraint-implied ordering for otherwise-independent steps.
        for i in 0..n {
            for j in (i + 1)..n {
                if graph.has_dependency(i, j) || graph.has_dependency(j, i) {
                    continue;
                }
                if !self.is_independent(&plan.steps[i], &plan.steps[j], constraints) {
                    // Keep the original order from the sequential plan.
                    graph.add_dependency(i, j);
                    debug!(from = i, to = j, "constraint-implied dep");
                }
            }
        }

        graph
    }

    /// Two steps are independent iff:
    /// 1. They touch different services.
    /// 2. Applying them in either order does not violate any state constraint.
    /// 3. No ordering constraint links their services.
    pub fn is_independent(
        &self,
        step_a: &PlanStep,
        step_b: &PlanStep,
        constraints: &[Constraint],
    ) -> bool {
        // Different services is necessary for independence.
        if step_a.service == step_b.service {
            return false;
        }

        // Check ordering constraints.
        for c in constraints {
            if let Constraint::Ordering { before, after, .. } = c {
                if (*before == step_a.service && *after == step_b.service)
                    || (*before == step_b.service && *after == step_a.service)
                {
                    return false;
                }
            }
        }

        // Check compatibility constraints: build the two intermediate states
        // (A-then-B vs B-then-A) and make sure both pass all state constraints.
        // We construct minimal synthetic states for the involved services.
        let dim = std::cmp::max(step_a.service.0, step_b.service.0) as usize + 1;

        // State after A-then-B.
        let mut state_ab = State::new(vec![VersionIndex(0); dim]);
        state_ab.set(step_a.service, step_a.from_version);
        state_ab.set(step_b.service, step_b.from_version);

        // Intermediate: A done, B not yet.
        let mut mid_ab = state_ab.clone();
        mid_ab.set(step_a.service, step_a.to_version);

        // State after B-then-A.
        let mut mid_ba = state_ab.clone();
        mid_ba.set(step_b.service, step_b.to_version);

        // Both intermediates and the final state must satisfy all constraints.
        // We only check constraints that reference these two services.
        for c in constraints {
            let relevant = match c {
                Constraint::Compatibility {
                    service_a,
                    service_b,
                    ..
                } => {
                    let pair = [step_a.service, step_b.service];
                    pair.contains(service_a) && pair.contains(service_b)
                }
                Constraint::Forbidden { service, .. } => {
                    *service == step_a.service || *service == step_b.service
                }
                Constraint::Resource { .. } => true,
                Constraint::Custom { .. } => {
                    // Custom predicates may depend on any services; assume relevant
                    // only when state dimension is sufficient.
                    mid_ab.dimension() >= dim && mid_ba.dimension() >= dim
                }
                Constraint::Ordering { .. } => false,
            };
            if !relevant {
                continue;
            }

            // A-then-B intermediate must be valid.
            if mid_ab.dimension() >= dim && !c.check_state(&mid_ab) {
                return false;
            }
            // B-then-A intermediate must be valid.
            if mid_ba.dimension() >= dim && !c.check_state(&mid_ba) {
                return false;
            }
        }

        true
    }
}

// ---------------------------------------------------------------------------
// Schedule — the result of scheduling steps into time slots
// ---------------------------------------------------------------------------

/// A concrete schedule: each slot holds the indices of steps that execute
/// concurrently during that time unit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Schedule {
    /// Each entry is a list of step indices that execute in that time slot.
    pub slots: Vec<Vec<usize>>,
}

impl Schedule {
    /// Number of sequential time slots required.
    pub fn makespan(&self) -> usize {
        self.slots.len()
    }

    /// Maximum number of steps in any single time slot.
    pub fn max_concurrency(&self) -> usize {
        self.slots.iter().map(|s| s.len()).max().unwrap_or(0)
    }

    /// Utilization = total_steps / (makespan × max_concurrency).
    /// Indicates how efficiently resources are used across time slots.
    pub fn utilization(&self) -> f64 {
        let makespan = self.makespan();
        let max_c = self.max_concurrency();
        if makespan == 0 || max_c == 0 {
            return 0.0;
        }
        let total_steps: usize = self.slots.iter().map(|s| s.len()).sum();
        total_steps as f64 / (makespan * max_c) as f64
    }

    /// Convert this schedule back into a `ParallelPlan`.
    pub fn to_parallel_plan(&self, plan: &DeploymentPlan) -> ParallelPlan {
        let groups: Vec<ParallelPlanStep> = self
            .slots
            .iter()
            .map(|slot| {
                let steps: Vec<PlanStep> = slot
                    .iter()
                    .map(|&idx| plan.steps[idx].clone())
                    .collect();
                ParallelPlanStep::new(steps)
            })
            .collect();

        ParallelPlan::new(groups, plan.start.clone(), plan.target.clone())
    }
}

// ---------------------------------------------------------------------------
// ParallelScheduler — list-scheduling with optional resource limits
// ---------------------------------------------------------------------------

/// Schedules steps into time slots respecting a dependency DAG and an
/// optional concurrency cap.
#[derive(Debug, Clone)]
pub struct ParallelScheduler {
    max_concurrency: usize,
}

impl ParallelScheduler {
    pub fn new(max_concurrency: usize) -> Self {
        Self { max_concurrency }
    }

    /// Classic list-scheduling algorithm.
    ///
    /// At each time slot pick all *ready* steps (predecessors finished) up to
    /// `max_concurrency`, prioritising steps with longer estimated duration
    /// (longest-processing-time-first heuristic, which tends to minimise
    /// makespan).
    #[instrument(skip_all, fields(steps = steps.len(), max_c = self.max_concurrency))]
    pub fn schedule(&self, dep_graph: &DependencyGraph, steps: &[PlanStep]) -> Schedule {
        let n = dep_graph.step_count;
        if n == 0 {
            return Schedule { slots: Vec::new() };
        }

        let mut in_degree = vec![0u32; n];
        for &(_, to) in &dep_graph.edges {
            in_degree[to] += 1;
        }

        let mut finished = vec![false; n];
        let mut slots: Vec<Vec<usize>> = Vec::new();

        let mut remaining = n;
        while remaining > 0 {
            // Gather ready steps: in-degree zero and not yet finished.
            let mut ready: Vec<usize> = (0..n)
                .filter(|&i| !finished[i] && in_degree[i] == 0)
                .collect();

            if ready.is_empty() {
                warn!("scheduler stall — no ready steps but {} remain", remaining);
                break;
            }

            // LPT heuristic: schedule longest jobs first.
            ready.sort_by(|&a, &b| {
                steps
                    .get(b)
                    .map(|s| s.estimated_duration_secs)
                    .unwrap_or(0)
                    .cmp(
                        &steps
                            .get(a)
                            .map(|s| s.estimated_duration_secs)
                            .unwrap_or(0),
                    )
            });

            let take = ready.len().min(self.max_concurrency);
            let slot: Vec<usize> = ready[..take].to_vec();

            for &idx in &slot {
                finished[idx] = true;
                remaining -= 1;
                for &succ in &dep_graph.fwd[idx] {
                    in_degree[succ] -= 1;
                }
            }

            debug!(slot_num = slots.len(), width = slot.len(), "scheduled slot");
            slots.push(slot);
        }

        Schedule { slots }
    }

    /// Schedule with per-resource capacity limits.
    ///
    /// Each step may consume a fraction of named resources. A step is only
    /// scheduled in a slot if adding it would not exceed any resource limit.
    /// Resource usage is looked up from `resource_limits` keyed by
    /// `"svc:<service_index>"`.
    #[instrument(skip_all, fields(steps = steps.len(), resources = resource_limits.len()))]
    pub fn schedule_with_resources(
        &self,
        dep_graph: &DependencyGraph,
        steps: &[PlanStep],
        resource_limits: &HashMap<String, f64>,
    ) -> Schedule {
        let n = dep_graph.step_count;
        if n == 0 {
            return Schedule { slots: Vec::new() };
        }

        let mut in_degree = vec![0u32; n];
        for &(_, to) in &dep_graph.edges {
            in_degree[to] += 1;
        }

        let mut finished = vec![false; n];
        let mut slots: Vec<Vec<usize>> = Vec::new();

        let step_resource_cost = |idx: usize| -> HashMap<String, f64> {
            let mut costs = HashMap::new();
            if let Some(step) = steps.get(idx) {
                let key = format!("svc:{}", step.service.0);
                costs.insert(key, 1.0);
                // Each step also consumes one unit of generic "concurrency".
                costs.insert("concurrency".to_string(), 1.0);
            }
            costs
        };

        let mut remaining = n;
        while remaining > 0 {
            let mut ready: Vec<usize> = (0..n)
                .filter(|&i| !finished[i] && in_degree[i] == 0)
                .collect();

            if ready.is_empty() {
                warn!(
                    "resource scheduler stall — no ready steps but {} remain",
                    remaining
                );
                break;
            }

            // LPT heuristic.
            ready.sort_by(|&a, &b| {
                steps
                    .get(b)
                    .map(|s| s.estimated_duration_secs)
                    .unwrap_or(0)
                    .cmp(
                        &steps
                            .get(a)
                            .map(|s| s.estimated_duration_secs)
                            .unwrap_or(0),
                    )
            });

            let mut slot: Vec<usize> = Vec::new();
            let mut slot_usage: HashMap<String, f64> = HashMap::new();

            for &idx in &ready {
                if slot.len() >= self.max_concurrency {
                    break;
                }

                let cost = step_resource_cost(idx);
                let fits = cost.iter().all(|(res, &amount)| {
                    let limit = resource_limits.get(res).copied().unwrap_or(f64::INFINITY);
                    let current = slot_usage.get(res).copied().unwrap_or(0.0);
                    current + amount <= limit + f64::EPSILON
                });

                if fits {
                    for (res, amount) in &cost {
                        *slot_usage.entry(res.clone()).or_insert(0.0) += amount;
                    }
                    slot.push(idx);
                }
            }

            if slot.is_empty() {
                // If nothing fits due to resource limits, force-schedule the
                // first ready step to guarantee progress.
                let forced = ready[0];
                slot.push(forced);
                warn!(step = forced, "force-scheduled step due to resource limits");
            }

            for &idx in &slot {
                finished[idx] = true;
                remaining -= 1;
                for &succ in &dep_graph.fwd[idx] {
                    in_degree[succ] -= 1;
                }
            }

            slots.push(slot);
        }

        Schedule { slots }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a plan step for service `svc` from version `from` to `to`.
    fn step(svc: u16, from: u16, to: u16) -> PlanStep {
        PlanStep::new(
            ServiceIndex(svc),
            VersionIndex(from),
            VersionIndex(to),
        )
    }

    /// Helper: build a plan step with explicit risk and duration.
    fn step_full(svc: u16, from: u16, to: u16, risk: u32, dur: u64) -> PlanStep {
        PlanStep::new(ServiceIndex(svc), VersionIndex(from), VersionIndex(to))
            .with_risk(risk)
            .with_duration(dur)
    }

    /// Helper: build a `DeploymentPlan` from raw steps, deriving start/target.
    fn make_plan(start: Vec<u16>, target: Vec<u16>, steps: Vec<PlanStep>) -> DeploymentPlan {
        let s = State::new(start.into_iter().map(VersionIndex).collect());
        let t = State::new(target.into_iter().map(VersionIndex).collect());
        DeploymentPlan::new(s, t, steps)
    }

    // -- DependencyGraph basic operations --

    #[test]
    fn test_dependency_graph_basics() {
        let mut g = DependencyGraph::new(4);
        g.add_dependency(0, 1);
        g.add_dependency(0, 2);
        g.add_dependency(1, 3);
        g.add_dependency(2, 3);

        assert!(g.has_dependency(0, 1));
        assert!(!g.has_dependency(1, 0));
        assert_eq!(g.predecessors(3), vec![1, 2]);
        assert_eq!(g.successors(0), vec![1, 2]);
        assert!(g.is_acyclic());
    }

    #[test]
    fn test_topological_sort_diamond() {
        let mut g = DependencyGraph::new(4);
        g.add_dependency(0, 1);
        g.add_dependency(0, 2);
        g.add_dependency(1, 3);
        g.add_dependency(2, 3);

        let order = g.topological_sort().unwrap();
        assert_eq!(order.len(), 4);
        // 0 must come before 1, 2; both before 3.
        let pos: HashMap<usize, usize> =
            order.iter().enumerate().map(|(i, &v)| (v, i)).collect();
        assert!(pos[&0] < pos[&1]);
        assert!(pos[&0] < pos[&2]);
        assert!(pos[&1] < pos[&3]);
        assert!(pos[&2] < pos[&3]);
    }

    #[test]
    fn test_cycle_detection() {
        let mut g = DependencyGraph::new(3);
        g.add_dependency(0, 1);
        g.add_dependency(1, 2);
        g.add_dependency(2, 0);

        assert!(!g.is_acyclic());
        assert!(g.topological_sort().is_err());
    }

    #[test]
    fn test_longest_path() {
        // Chain: 0 -> 1 -> 2 -> 3 (length 4)
        let mut g = DependencyGraph::new(5);
        g.add_dependency(0, 1);
        g.add_dependency(1, 2);
        g.add_dependency(2, 3);
        // 4 is independent
        assert_eq!(g.longest_path(), 4);
    }

    // -- DependencyAnalyzer --

    #[test]
    fn test_independent_steps_different_services() {
        let analyzer = DependencyAnalyzer::new();
        let a = step(0, 0, 1);
        let b = step(1, 0, 1);
        assert!(analyzer.is_independent(&a, &b, &[]));
    }

    #[test]
    fn test_same_service_not_independent() {
        let analyzer = DependencyAnalyzer::new();
        let a = step(0, 0, 1);
        let b = step(0, 1, 2);
        assert!(!analyzer.is_independent(&a, &b, &[]));
    }

    // -- ParallelScheduler & Schedule --

    #[test]
    fn test_schedule_fully_independent() {
        // 3 independent steps -> should all land in one slot.
        let steps = vec![
            step_full(0, 0, 1, 5, 10),
            step_full(1, 0, 1, 3, 20),
            step_full(2, 0, 1, 1, 15),
        ];
        let graph = DependencyGraph::new(3); // no edges
        let sched = ParallelScheduler::new(10).schedule(&graph, &steps);

        assert_eq!(sched.makespan(), 1);
        assert_eq!(sched.max_concurrency(), 3);
        assert!((sched.utilization() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_schedule_chain() {
        // Chain 0 -> 1 -> 2: must be 3 sequential slots.
        let steps = vec![step(0, 0, 1), step(1, 0, 1), step(2, 0, 1)];
        let mut graph = DependencyGraph::new(3);
        graph.add_dependency(0, 1);
        graph.add_dependency(1, 2);

        let sched = ParallelScheduler::new(10).schedule(&graph, &steps);
        assert_eq!(sched.makespan(), 3);
        assert_eq!(sched.max_concurrency(), 1);
    }

    // -- ParallelPlan end-to-end --

    #[test]
    fn test_parallelize_two_independent_services() {
        let steps = vec![
            step_full(0, 0, 1, 2, 30),
            step_full(1, 0, 1, 4, 60),
        ];
        let plan = make_plan(vec![0, 0], vec![1, 1], steps);

        let pp = ParallelPlan::parallelize(&plan, &[]).unwrap();
        assert_eq!(pp.critical_path_length(), 1);
        assert_eq!(pp.total_steps(), 2);
        assert_eq!(pp.max_parallelism(), 2);
        // Parallel duration = max(30, 60) = 60.
        assert_eq!(pp.total_duration(), 60);
        // Sequential would be 90, speedup = 90/60 = 1.5.
        assert!((pp.speedup_over_sequential() - 1.5).abs() < 0.01);
    }

    #[test]
    fn test_parallelize_with_ordering_constraint() {
        let steps = vec![
            step_full(0, 0, 1, 2, 30),
            step_full(1, 0, 1, 4, 60),
        ];
        let plan = make_plan(vec![0, 0], vec![1, 1], steps);
        let constraints = vec![Constraint::Ordering {
            id: Id::from_name("ord-0-before-1"),
            before: ServiceIndex(0),
            after: ServiceIndex(1),
        }];

        let pp = ParallelPlan::parallelize(&plan, &constraints).unwrap();
        // Ordering constraint forces sequential execution.
        assert_eq!(pp.critical_path_length(), 2);
        assert_eq!(pp.max_parallelism(), 1);
    }

    // -- Schedule resource limits --

    #[test]
    fn test_schedule_with_resource_limits() {
        // 3 independent steps but concurrency limited to 2.
        let steps = vec![
            step_full(0, 0, 1, 1, 10),
            step_full(1, 0, 1, 1, 10),
            step_full(2, 0, 1, 1, 10),
        ];
        let graph = DependencyGraph::new(3);
        let mut limits = HashMap::new();
        limits.insert("concurrency".to_string(), 2.0);

        let sched =
            ParallelScheduler::new(3).schedule_with_resources(&graph, &steps, &limits);

        // With concurrency limit 2, need 2 slots for 3 steps.
        assert_eq!(sched.makespan(), 2);
        assert!(sched.max_concurrency() <= 2);
    }

    // -- Transitive reduction --

    #[test]
    fn test_transitive_reduction() {
        // 0 -> 1 -> 2, plus shortcut 0 -> 2. Reduction removes 0 -> 2.
        let mut g = DependencyGraph::new(3);
        g.add_dependency(0, 1);
        g.add_dependency(1, 2);
        g.add_dependency(0, 2);

        let reduced = g.transitive_reduction();
        assert!(reduced.has_dependency(0, 1));
        assert!(reduced.has_dependency(1, 2));
        assert!(!reduced.has_dependency(0, 2));
    }

    // -- ParallelPlanStep --

    #[test]
    fn test_parallel_plan_step_aggregation() {
        let steps = vec![
            step_full(0, 0, 1, 10, 30),
            step_full(1, 0, 1, 5, 60),
            step_full(2, 0, 1, 8, 45),
        ];
        let group = ParallelPlanStep::new(steps);
        assert_eq!(group.step_count(), 3);
        assert_eq!(group.group_risk, 10);
        assert_eq!(group.group_duration, 60);
        assert!(!group.is_single());
        assert_eq!(group.services().len(), 3);
    }
}
