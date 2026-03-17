//! Deployment planner — main planning engine with BFS, greedy, and optimal strategies.
//!
//! This module provides the core planning algorithms for computing safe deployment
//! plans. The planner coordinates graph search over the version-product graph,
//! respecting compatibility constraints, ordering constraints, and optional
//! monotonicity requirements. Three strategies are available:
//!
//! - **Greedy**: topological-order service-by-service upgrades with backtracking
//! - **BFS**: breadth-first search over safe states (shortest plan)
//! - **A\***: heuristic search using Hamming distance to the target
//!
//! The planner also computes completeness bounds, checks downward closure of
//! the safe region, and collects detailed statistics for diagnostics.

use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::cmp::{Ordering, Reverse};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use safestep_types::identifiers::{Id, PlanId, ConstraintId};
use safestep_types::error::SafeStepError;

use crate::{
    CoreResult, Constraint, DeploymentPlan, Edge, PlanStep, PlannerConfig,
    ServiceDescriptor, ServiceIndex, State, StuckWitness, TransitionMetadata,
    VersionIndex, VersionProductGraph,
};

// ---------------------------------------------------------------------------
// OptGoal — optimisation objective for plan search
// ---------------------------------------------------------------------------

/// Optimisation objective guiding the planner's search strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OptGoal {
    /// Minimise the number of deployment steps.
    MinSteps,
    /// Minimise total risk score across all steps.
    MinRisk,
    /// Minimise total estimated deployment duration.
    MinDuration,
    /// Balance all three objectives equally.
    Balanced,
}

impl Default for OptGoal {
    fn default() -> Self {
        OptGoal::MinSteps
    }
}

// ---------------------------------------------------------------------------
// PlanResult
// ---------------------------------------------------------------------------

/// Outcome of a planning attempt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PlanResult {
    Plan(DeploymentPlan),
    Infeasible(InfeasibilityWitness),
    Timeout,
    Error(String),
}

impl PlanResult {
    pub fn is_plan(&self) -> bool {
        matches!(self, PlanResult::Plan(_))
    }

    pub fn into_plan(self) -> Option<DeploymentPlan> {
        match self {
            PlanResult::Plan(p) => Some(p),
            _ => None,
        }
    }

    pub fn is_infeasible(&self) -> bool {
        matches!(self, PlanResult::Infeasible(_))
    }

    pub fn is_timeout(&self) -> bool {
        matches!(self, PlanResult::Timeout)
    }

    pub fn is_error(&self) -> bool {
        matches!(self, PlanResult::Error(_))
    }
}

// ---------------------------------------------------------------------------
// InfeasibilityWitness
// ---------------------------------------------------------------------------

/// Witness proving that no feasible plan exists.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InfeasibilityWitness {
    pub reason: String,
    pub blocking_constraints: Vec<ConstraintId>,
    pub start: State,
    pub target: State,
    pub explored_states: usize,
}

impl InfeasibilityWitness {
    pub fn new(reason: impl Into<String>, start: State, target: State) -> Self {
        Self {
            reason: reason.into(),
            blocking_constraints: Vec::new(),
            start,
            target,
            explored_states: 0,
        }
    }

    pub fn with_explored(mut self, count: usize) -> Self {
        self.explored_states = count;
        self
    }

    pub fn with_blocking(mut self, ids: Vec<ConstraintId>) -> Self {
        self.blocking_constraints = ids;
        self
    }
}

// ---------------------------------------------------------------------------
// PlanStepInfo — lightweight step descriptor for internal diagnostics
// ---------------------------------------------------------------------------

/// Lightweight description of one step in a plan, used for internal diagnostics.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanStepInfo {
    pub service_idx: usize,
    pub from_ver: usize,
    pub to_ver: usize,
    pub step_num: usize,
}

impl PlanStepInfo {
    pub fn new(service_idx: usize, from_ver: usize, to_ver: usize, step_num: usize) -> Self {
        Self { service_idx, from_ver, to_ver, step_num }
    }

    pub fn is_upgrade(&self) -> bool {
        self.to_ver > self.from_ver
    }

    pub fn to_plan_step(&self) -> PlanStep {
        PlanStep::new(
            ServiceIndex(self.service_idx as u16),
            VersionIndex(self.from_ver as u16),
            VersionIndex(self.to_ver as u16),
        )
    }
}

/// Summary of a partial plan found before timeout.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartialInfo {
    pub best_depth: usize,
    pub partial_steps: Vec<PlanStepInfo>,
    pub explored: usize,
}

/// Detailed info about an infeasibility result (richer than InfeasibilityWitness).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InfeasibleInfo {
    pub reason: String,
    pub blocking_constraints: Vec<(usize, usize, String)>,
    pub depth_reached: usize,
}

// ---------------------------------------------------------------------------
// PlannerStats
// ---------------------------------------------------------------------------

/// Aggregate statistics collected during a planning run.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PlannerStats {
    pub states_explored: u64,
    pub states_pruned: u64,
    pub solve_time_ms: u64,
    pub max_depth_reached: usize,
    pub plan_length: usize,
    pub greedy_attempts: usize,
    pub bfs_expansions: u64,
    pub astar_expansions: u64,
}

// ---------------------------------------------------------------------------
// PlannerState (internal bookkeeping)
// ---------------------------------------------------------------------------

/// Internal bookkeeping for the planner.
#[derive(Debug, Clone)]
pub struct PlannerState {
    pub current_depth: usize,
    pub max_depth: usize,
    pub nodes_explored: u64,
    pub solver_calls: u64,
    pub start_time: Option<Instant>,
}

impl PlannerState {
    pub fn new(max_depth: usize) -> Self {
        Self {
            current_depth: 0,
            max_depth,
            nodes_explored: 0,
            solver_calls: 0,
            start_time: None,
        }
    }

    pub fn elapsed(&self) -> Duration {
        self.start_time.map(|t| t.elapsed()).unwrap_or(Duration::ZERO)
    }

    pub fn timed_out(&self, timeout: Duration) -> bool {
        self.elapsed() > timeout
    }
}

// ---------------------------------------------------------------------------
// TreewidthEstimator
// ---------------------------------------------------------------------------

/// Estimates the treewidth of the constraint-interaction graph using
/// min-degree elimination heuristic.
pub struct TreewidthEstimator;

impl TreewidthEstimator {
    /// Estimate treewidth of the constraint-interaction graph.
    ///
    /// Build an interaction graph: one node per service, one edge when two
    /// services share a constraint. Run greedy min-degree elimination and
    /// return the maximum clique size minus one encountered.
    pub fn estimate(graph: &VersionProductGraph, constraints: &[Constraint]) -> usize {
        let n = graph.service_count();
        if n == 0 {
            return 0;
        }
        let mut adj: Vec<HashSet<usize>> = vec![HashSet::new(); n];
        for c in constraints {
            match c {
                Constraint::Compatibility { service_a, service_b, .. } => {
                    let a = service_a.0 as usize;
                    let b = service_b.0 as usize;
                    if a < n && b < n && a != b {
                        adj[a].insert(b);
                        adj[b].insert(a);
                    }
                }
                Constraint::Ordering { before, after, .. } => {
                    let a = before.0 as usize;
                    let b = after.0 as usize;
                    if a < n && b < n && a != b {
                        adj[a].insert(b);
                        adj[b].insert(a);
                    }
                }
                _ => {}
            }
        }
        let mut eliminated = vec![false; n];
        let mut tw = 0usize;
        for _ in 0..n {
            let mut best = None;
            let mut best_deg = usize::MAX;
            for v in 0..n {
                if eliminated[v] {
                    continue;
                }
                let deg = adj[v].iter().filter(|&&u| !eliminated[u]).count();
                if deg < best_deg {
                    best_deg = deg;
                    best = Some(v);
                }
            }
            let v = match best {
                Some(v) => v,
                None => break,
            };
            tw = tw.max(best_deg);
            let neighbors: Vec<usize> = adj[v]
                .iter()
                .copied()
                .filter(|&u| !eliminated[u])
                .collect();
            for i in 0..neighbors.len() {
                for j in (i + 1)..neighbors.len() {
                    let a = neighbors[i];
                    let b = neighbors[j];
                    adj[a].insert(b);
                    adj[b].insert(a);
                }
            }
            eliminated[v] = true;
        }
        tw
    }
}

// ---------------------------------------------------------------------------
// Helpers: constraint checking and path reconstruction
// ---------------------------------------------------------------------------

fn all_constraints_satisfied(state: &State, constraints: &[Constraint]) -> bool {
    constraints.iter().all(|c| c.check_state(state))
}

fn all_plan_constraints_satisfied(steps: &[PlanStep], constraints: &[Constraint]) -> bool {
    for c in constraints {
        for (i, _step) in steps.iter().enumerate() {
            if !c.check_transition(i, steps) {
                return false;
            }
        }
    }
    true
}

/// Build PlanStep list from a sequence of states.
fn states_to_steps(path: &[State]) -> Vec<PlanStep> {
    let mut steps = Vec::new();
    for i in 0..path.len().saturating_sub(1) {
        let from = &path[i];
        let to = &path[i + 1];
        for (s, (fv, tv)) in from.versions.iter().zip(to.versions.iter()).enumerate() {
            if fv != tv {
                let mut step = PlanStep::new(ServiceIndex(s as u16), *fv, *tv);
                step.risk_score = if tv.0 > fv.0 { 1 } else { 2 };
                steps.push(step);
            }
        }
    }
    steps
}

/// Build PlanStepInfo list from a state path, for diagnostics.
fn states_to_step_infos(path: &[State]) -> Vec<PlanStepInfo> {
    let mut infos = Vec::new();
    let mut step_num = 0usize;
    for i in 0..path.len().saturating_sub(1) {
        let from = &path[i];
        let to = &path[i + 1];
        for (s, (fv, tv)) in from.versions.iter().zip(to.versions.iter()).enumerate() {
            if fv != tv {
                infos.push(PlanStepInfo::new(s, fv.0 as usize, tv.0 as usize, step_num));
                step_num += 1;
            }
        }
    }
    infos
}

// ---------------------------------------------------------------------------
// DeploymentPlanner
// ---------------------------------------------------------------------------

/// Main planning engine.
///
/// Orchestrates greedy, BFS, and A* strategies according to the planner
/// configuration. Estimates treewidth to pick an appropriate algorithm.
pub struct DeploymentPlanner {
    config: PlannerConfig,
    graph: VersionProductGraph,
    constraints: Vec<Constraint>,
    state: PlannerState,
    opt_goal: OptGoal,
    stats: PlannerStats,
}

impl DeploymentPlanner {
    /// Create a new planner.
    pub fn new(
        config: PlannerConfig,
        graph: VersionProductGraph,
        constraints: Vec<Constraint>,
    ) -> Self {
        let max_depth = config.max_depth;
        Self {
            config,
            graph,
            constraints,
            state: PlannerState::new(max_depth),
            opt_goal: OptGoal::default(),
            stats: PlannerStats::default(),
        }
    }

    /// Set the optimisation objective.
    pub fn with_opt_goal(mut self, goal: OptGoal) -> Self {
        self.opt_goal = goal;
        self
    }

    /// Produce a deployment plan (or explain why none exists).
    pub fn plan(&mut self, start: &State, target: &State) -> PlanResult {
        self.plan_with_timeout(start, target, self.config.timeout)
    }

    /// Plan with an explicit timeout.
    pub fn plan_with_timeout(
        &mut self,
        start: &State,
        target: &State,
        timeout: Duration,
    ) -> PlanResult {
        self.state = PlannerState::new(self.config.max_depth);
        self.state.start_time = Some(Instant::now());
        self.stats = PlannerStats::default();

        if start == target {
            return PlanResult::Plan(DeploymentPlan::new(
                start.clone(),
                target.clone(),
                Vec::new(),
            ));
        }

        if !all_constraints_satisfied(start, &self.constraints) {
            return PlanResult::Infeasible(InfeasibilityWitness::new(
                "Start state violates constraints",
                start.clone(),
                target.clone(),
            ));
        }
        if !all_constraints_satisfied(target, &self.constraints) {
            return PlanResult::Infeasible(InfeasibilityWitness::new(
                "Target state violates constraints",
                start.clone(),
                target.clone(),
            ));
        }

        let completeness_bound = self.compute_completeness_bound(start, target);
        let effective_depth = self.config.max_depth.min(completeness_bound);

        let dc = self.check_downward_closure();

        // Try greedy first for a fast (possibly suboptimal) plan.
        if let Some(plan) = self.plan_greedy(start, target) {
            self.stats.greedy_attempts = 1;
            if self.opt_goal == OptGoal::MinSteps {
                // Greedy is not guaranteed shortest — run BFS anyway
                let bfs_result = self.plan_bfs(start, target, effective_depth, timeout);
                if let PlanResult::Plan(better) = &bfs_result {
                    if better.step_count() < plan.step_count() {
                        return bfs_result;
                    }
                }
            }
            return PlanResult::Plan(plan);
        }

        let tw = TreewidthEstimator::estimate(&self.graph, &self.constraints);
        if tw <= self.config.treewidth_threshold && self.graph.state_count() < 5000 {
            self.plan_bfs(start, target, effective_depth, timeout)
        } else {
            self.astar_plan(start, target, timeout)
        }
    }

    /// Greedy plan: upgrade one service at a time in topological order.
    pub fn plan_greedy(&self, start: &State, target: &State) -> Option<DeploymentPlan> {
        if start == target {
            return Some(DeploymentPlan::new(start.clone(), target.clone(), Vec::new()));
        }
        let greedy = GreedyPlanner::new(&self.graph, &self.constraints);
        greedy.plan(start, target)
    }

    /// BFS plan over safe states.
    pub fn plan_bfs(
        &mut self,
        start: &State,
        target: &State,
        max_depth: usize,
        timeout: Duration,
    ) -> PlanResult {
        let mut visited: HashMap<State, Option<State>> = HashMap::new();
        let mut queue: VecDeque<(State, usize)> = VecDeque::new();

        visited.insert(start.clone(), None);
        queue.push_back((start.clone(), 0));

        while let Some((current, depth)) = queue.pop_front() {
            self.state.nodes_explored += 1;
            self.stats.bfs_expansions += 1;

            if self.state.timed_out(timeout) {
                return PlanResult::Timeout;
            }

            if current == *target {
                let path = self.reconstruct_path(&visited, target);
                let steps = states_to_steps(&path);
                self.stats.plan_length = steps.len();
                return PlanResult::Plan(DeploymentPlan::new(
                    start.clone(),
                    target.clone(),
                    steps,
                ));
            }

            if depth >= max_depth {
                self.stats.states_pruned += 1;
                continue;
            }

            if visited.len() > max_depth * 2000 {
                break;
            }

            let neighbors = self.generate_successors(&current);
            for next in neighbors {
                if !visited.contains_key(&next) {
                    visited.insert(next.clone(), Some(current.clone()));
                    queue.push_back((next, depth + 1));
                }
            }
        }

        self.stats.max_depth_reached = max_depth;
        PlanResult::Infeasible(InfeasibilityWitness {
            reason: "BFS exhausted reachable states without finding target".into(),
            blocking_constraints: Vec::new(),
            start: start.clone(),
            target: target.clone(),
            explored_states: self.state.nodes_explored as usize,
        })
    }

    /// Check whether a state satisfies all pairwise compatibility constraints.
    pub fn is_safe_state(&self, state: &State) -> bool {
        all_constraints_satisfied(state, &self.constraints)
    }

    /// Generate neighbor states from a given state.
    ///
    /// Returns `(new_state, service_idx, old_version, new_version)` tuples.
    /// If `monotone` is true, only version increases are generated.
    pub fn neighbors(
        &self,
        state: &State,
        monotone: bool,
    ) -> Vec<(State, ServiceIndex, VersionIndex, VersionIndex)> {
        let mut out = Vec::new();
        for (s_idx, svc) in self.graph.services.iter().enumerate() {
            let si = ServiceIndex(s_idx as u16);
            let cur = state.get(si);
            for v in 0..svc.versions.len() as u16 {
                let vi = VersionIndex(v);
                if vi == cur {
                    continue;
                }
                if monotone && vi.0 < cur.0 {
                    continue;
                }
                let mut next = state.clone();
                next.set(si, vi);
                if all_constraints_satisfied(&next, &self.constraints) {
                    out.push((next, si, cur, vi));
                }
            }
        }
        out
    }

    /// Upper bound on the number of steps needed for any plan between two states.
    ///
    /// The completeness bound is the product of the version spans across all
    /// services multiplied by the service count (a conservative over-estimate).
    pub fn compute_completeness_bound(&self, start: &State, target: &State) -> usize {
        let n = self.graph.service_count();
        if n == 0 {
            return 0;
        }
        let total_versions: usize = self
            .graph
            .services
            .iter()
            .map(|s| s.versions.len())
            .product::<usize>();
        let hamming = start.distance(target);
        let span_sum: usize = start
            .versions
            .iter()
            .zip(target.versions.iter())
            .map(|(a, b)| {
                let diff = (a.0 as isize - b.0 as isize).unsigned_abs();
                diff.max(1)
            })
            .sum();
        // Completeness bound: cannot exceed total reachable states.
        total_versions.min(span_sum * n).max(hamming)
    }

    /// Check whether the set of safe states satisfies downward closure.
    ///
    /// Downward closure means: if state `s` is safe and `s'` is component-wise
    /// ≤ `s`, then `s'` is also safe. This property enables monotone planning.
    pub fn check_downward_closure(&self) -> bool {
        let n = self.graph.service_count();
        if n == 0 {
            return true;
        }
        let version_counts: Vec<u16> = self
            .graph
            .services
            .iter()
            .map(|s| s.versions.len() as u16)
            .collect();

        let all_states = enumerate_states_bounded(&version_counts, 5);

        for state in &all_states {
            if !all_constraints_satisfied(state, &self.constraints) {
                continue;
            }
            for (s_idx, &max_v) in version_counts.iter().enumerate() {
                let si = ServiceIndex(s_idx as u16);
                let cur = state.get(si);
                if cur.0 > 0 {
                    let mut lower = state.clone();
                    lower.set(si, VersionIndex(cur.0 - 1));
                    if !all_constraints_satisfied(&lower, &self.constraints) {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// A* plan with Hamming distance heuristic.
    fn astar_plan(&mut self, start: &State, target: &State, timeout: Duration) -> PlanResult {
        #[derive(Clone, Eq, PartialEq)]
        struct AStarNode {
            state: State,
            g: usize,
            f: usize,
        }

        impl Ord for AStarNode {
            fn cmp(&self, other: &Self) -> Ordering {
                other.f.cmp(&self.f).then_with(|| other.g.cmp(&self.g))
            }
        }

        impl PartialOrd for AStarNode {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }

        let h = |s: &State| -> usize { s.distance(target) };

        let mut open: BinaryHeap<AStarNode> = BinaryHeap::new();
        let mut g_scores: HashMap<State, usize> = HashMap::new();
        let mut came_from: HashMap<State, State> = HashMap::new();

        g_scores.insert(start.clone(), 0);
        open.push(AStarNode {
            state: start.clone(),
            g: 0,
            f: h(start),
        });

        while let Some(AStarNode { state: current, g, .. }) = open.pop() {
            self.state.nodes_explored += 1;
            self.stats.astar_expansions += 1;

            if self.state.timed_out(timeout) {
                return PlanResult::Timeout;
            }

            if current == *target {
                let mut path = vec![current.clone()];
                let mut cur = &current;
                while let Some(prev) = came_from.get(cur) {
                    path.push(prev.clone());
                    cur = prev;
                }
                path.reverse();
                let steps = states_to_steps(&path);
                self.stats.plan_length = steps.len();
                return PlanResult::Plan(DeploymentPlan::new(
                    start.clone(),
                    target.clone(),
                    steps,
                ));
            }

            if g > *g_scores.get(&current).unwrap_or(&usize::MAX) {
                continue;
            }

            if g >= self.config.max_depth {
                self.stats.states_pruned += 1;
                continue;
            }

            let neighbors = self.generate_successors(&current);
            for next in neighbors {
                let tentative_g = g + 1;
                if tentative_g < *g_scores.get(&next).unwrap_or(&usize::MAX) {
                    g_scores.insert(next.clone(), tentative_g);
                    came_from.insert(next.clone(), current.clone());
                    open.push(AStarNode {
                        state: next.clone(),
                        g: tentative_g,
                        f: tentative_g + h(&next),
                    });
                }
            }
        }

        PlanResult::Infeasible(InfeasibilityWitness {
            reason: "A* exhausted search space without finding target".into(),
            blocking_constraints: Vec::new(),
            start: start.clone(),
            target: target.clone(),
            explored_states: self.state.nodes_explored as usize,
        })
    }

    /// Generate all valid successor states (single-service version changes that
    /// satisfy constraints).
    fn generate_successors(&self, state: &State) -> Vec<State> {
        if !self.graph.adjacency.is_empty() && self.graph.adjacency.contains_key(state) {
            return self
                .graph
                .neighbors(state)
                .into_iter()
                .filter(|(s, _)| all_constraints_satisfied(s, &self.constraints))
                .map(|(s, _)| s)
                .collect();
        }
        let mut successors = Vec::new();
        for (s_idx, svc) in self.graph.services.iter().enumerate() {
            let si = ServiceIndex(s_idx as u16);
            let current_ver = state.get(si);
            for v in 0..svc.versions.len() as u16 {
                let vi = VersionIndex(v);
                if vi == current_ver {
                    continue;
                }
                let mut next = state.clone();
                next.set(si, vi);
                if all_constraints_satisfied(&next, &self.constraints) {
                    successors.push(next);
                }
            }
        }
        successors
    }

    /// Reconstruct BFS path from predecessor map.
    fn reconstruct_path(
        &self,
        came_from: &HashMap<State, Option<State>>,
        target: &State,
    ) -> Vec<State> {
        let mut path = vec![target.clone()];
        let mut current = target.clone();
        while let Some(Some(prev)) = came_from.get(&current) {
            path.push(prev.clone());
            current = prev.clone();
        }
        path.reverse();
        path
    }

    /// Get the current planner state.
    pub fn planner_state(&self) -> &PlannerState {
        &self.state
    }

    /// Get accumulated statistics.
    pub fn stats(&self) -> &PlannerStats {
        &self.stats
    }

    /// Get config.
    pub fn config(&self) -> &PlannerConfig {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// GreedyPlanner
// ---------------------------------------------------------------------------

/// Fast heuristic planner using greedy service-by-service upgrades.
///
/// Strategy: compute a topological ordering over service dependencies, then
/// for each service (in order) change it from start version to target version.
/// If a direct jump isn't safe, try intermediate versions. If that fails,
/// backtrack and try alternative orderings.
pub struct GreedyPlanner<'a> {
    graph: &'a VersionProductGraph,
    constraints: &'a [Constraint],
}

impl<'a> GreedyPlanner<'a> {
    pub fn new(graph: &'a VersionProductGraph, constraints: &'a [Constraint]) -> Self {
        Self { graph, constraints }
    }

    /// Attempt a greedy plan.
    pub fn plan(&self, start: &State, target: &State) -> Option<DeploymentPlan> {
        if start == target {
            return Some(DeploymentPlan::new(start.clone(), target.clone(), Vec::new()));
        }

        let n = self.graph.service_count();
        let order = self.topological_order(n);

        // First try the straightforward topological order.
        if let Some(steps) = self.try_order(start, target, &order) {
            return Some(DeploymentPlan::new(start.clone(), target.clone(), steps));
        }

        // If that fails, try the reverse order.
        let mut rev_order = order.clone();
        rev_order.reverse();
        if let Some(steps) = self.try_order(start, target, &rev_order) {
            return Some(DeploymentPlan::new(start.clone(), target.clone(), steps));
        }

        // Try with backtracking (limited depth).
        self.plan_with_backtrack(start, target, n * 3)
    }

    /// Try upgrading services in the given order.
    fn try_order(
        &self,
        start: &State,
        target: &State,
        order: &[usize],
    ) -> Option<Vec<PlanStep>> {
        let mut current = start.clone();
        let mut steps = Vec::new();

        for &svc_idx in order {
            let si = ServiceIndex(svc_idx as u16);
            let from_ver = current.get(si);
            let to_ver = target.get(si);
            if from_ver == to_ver {
                continue;
            }

            // Try direct jump.
            let mut next = current.clone();
            next.set(si, to_ver);
            if all_constraints_satisfied(&next, self.constraints) {
                steps.push(PlanStep::new(si, from_ver, to_ver));
                current = next;
                continue;
            }

            // Try intermediate versions.
            let svc = &self.graph.services[svc_idx];
            let range: Vec<u16> = if to_ver.0 > from_ver.0 {
                (from_ver.0 + 1..=to_ver.0).collect()
            } else {
                (to_ver.0..from_ver.0).rev().collect()
            };

            let mut ok = true;
            for v in range {
                let vi = VersionIndex(v);
                let prev_ver = current.get(si);
                let mut next = current.clone();
                next.set(si, vi);
                if all_constraints_satisfied(&next, self.constraints) {
                    steps.push(PlanStep::new(si, prev_ver, vi));
                    current = next;
                } else {
                    ok = false;
                    break;
                }
            }
            if !ok {
                return None;
            }
        }

        if current != *target {
            return None;
        }
        Some(steps)
    }

    /// Greedy plan with limited backtracking.
    fn plan_with_backtrack(
        &self,
        start: &State,
        target: &State,
        max_depth: usize,
    ) -> Option<DeploymentPlan> {
        let mut visited: HashSet<State> = HashSet::new();
        let mut path: Vec<State> = vec![start.clone()];
        visited.insert(start.clone());

        if self.backtrack_dfs(target, max_depth, &mut visited, &mut path) {
            let steps = states_to_steps(&path);
            Some(DeploymentPlan::new(start.clone(), target.clone(), steps))
        } else {
            None
        }
    }

    fn backtrack_dfs(
        &self,
        target: &State,
        max_depth: usize,
        visited: &mut HashSet<State>,
        path: &mut Vec<State>,
    ) -> bool {
        let current = path.last().unwrap().clone();
        if current == *target {
            return true;
        }
        if path.len() > max_depth {
            return false;
        }

        let n = self.graph.service_count();
        let mut candidates: Vec<State> = Vec::new();

        for s_idx in 0..n {
            let si = ServiceIndex(s_idx as u16);
            let cur_v = current.get(si);
            let tgt_v = target.get(si);
            if cur_v == tgt_v {
                continue;
            }
            // Try moving toward target.
            let next_v = if tgt_v.0 > cur_v.0 {
                VersionIndex(cur_v.0 + 1)
            } else {
                VersionIndex(cur_v.0 - 1)
            };
            let mut next = current.clone();
            next.set(si, next_v);
            if all_constraints_satisfied(&next, self.constraints) && !visited.contains(&next) {
                candidates.push(next);
            }
        }

        // Sort candidates: prefer states closer to target.
        candidates.sort_by_key(|s| s.distance(target));

        for next in candidates {
            visited.insert(next.clone());
            path.push(next.clone());
            if self.backtrack_dfs(target, max_depth, visited, path) {
                return true;
            }
            path.pop();
        }
        false
    }

    /// Compute topological order of services based on ordering constraints.
    fn topological_order(&self, n: usize) -> Vec<usize> {
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut in_degree = vec![0usize; n];

        for c in self.constraints {
            if let Constraint::Ordering { before, after, .. } = c {
                let b = before.0 as usize;
                let a = after.0 as usize;
                if b < n && a < n {
                    adj[b].push(a);
                    in_degree[a] += 1;
                }
            }
        }

        let mut queue: VecDeque<usize> = VecDeque::new();
        for i in 0..n {
            if in_degree[i] == 0 {
                queue.push_back(i);
            }
        }

        let mut order = Vec::with_capacity(n);
        while let Some(v) = queue.pop_front() {
            order.push(v);
            for &u in &adj[v] {
                in_degree[u] -= 1;
                if in_degree[u] == 0 {
                    queue.push_back(u);
                }
            }
        }

        if order.len() < n {
            for i in 0..n {
                if !order.contains(&i) {
                    order.push(i);
                }
            }
        }
        order
    }
}

// ---------------------------------------------------------------------------
// OptimalPlanner
// ---------------------------------------------------------------------------

/// Finds minimum-step (and optionally multi-objective) deployment plans.
pub struct OptimalPlanner<'a> {
    graph: &'a VersionProductGraph,
    constraints: &'a [Constraint],
}

impl<'a> OptimalPlanner<'a> {
    pub fn new(graph: &'a VersionProductGraph, constraints: &'a [Constraint]) -> Self {
        Self { graph, constraints }
    }

    /// Find shortest (minimum-step) feasible plan.
    pub fn find_optimal(&self, start: &State, target: &State, max_depth: usize) -> PlanResult {
        self.find_optimal_with_timeout(start, target, max_depth, Duration::from_secs(300))
    }

    /// Find optimal with timeout.
    pub fn find_optimal_with_timeout(
        &self,
        start: &State,
        target: &State,
        max_depth: usize,
        timeout: Duration,
    ) -> PlanResult {
        if start == target {
            return PlanResult::Plan(DeploymentPlan::new(
                start.clone(),
                target.clone(),
                Vec::new(),
            ));
        }
        let started = Instant::now();

        let mut visited: HashMap<State, Option<State>> = HashMap::new();
        let mut queue: VecDeque<(State, usize)> = VecDeque::new();

        visited.insert(start.clone(), None);
        queue.push_back((start.clone(), 0));

        while let Some((current, depth)) = queue.pop_front() {
            if started.elapsed() > timeout {
                return PlanResult::Timeout;
            }
            if current == *target {
                let path = Self::reconstruct(&visited, target);
                let steps = states_to_steps(&path);
                return PlanResult::Plan(DeploymentPlan::new(
                    start.clone(),
                    target.clone(),
                    steps,
                ));
            }
            if depth >= max_depth {
                continue;
            }

            let successors = self.generate_successors(&current);
            for next in successors {
                if !visited.contains_key(&next) {
                    visited.insert(next.clone(), Some(current.clone()));
                    queue.push_back((next, depth + 1));
                }
            }
        }

        PlanResult::Infeasible(InfeasibilityWitness {
            reason: "No feasible plan found within depth bound".into(),
            blocking_constraints: Vec::new(),
            start: start.clone(),
            target: target.clone(),
            explored_states: visited.len(),
        })
    }

    /// Find a set of Pareto-optimal plans (steps vs. risk).
    pub fn find_pareto(
        &self,
        start: &State,
        target: &State,
        max_depth: usize,
    ) -> Vec<DeploymentPlan> {
        let mut plans = Vec::new();

        if let PlanResult::Plan(p) = self.find_optimal(start, target, max_depth) {
            plans.push(p);
        }

        if let Some(p) = self.find_low_risk_plan(start, target, max_depth) {
            let dominated = plans.iter().any(|existing: &DeploymentPlan| {
                existing.step_count() == p.step_count() && existing.total_risk == p.total_risk
            });
            if !dominated {
                plans.push(p);
            }
        }

        let mut pareto = Vec::new();
        for plan in &plans {
            let is_dominated = plans.iter().any(|other| {
                other.step_count() <= plan.step_count()
                    && other.total_risk <= plan.total_risk
                    && (other.step_count() < plan.step_count()
                        || other.total_risk < plan.total_risk)
            });
            if !is_dominated {
                pareto.push(plan.clone());
            }
        }

        if pareto.is_empty() && !plans.is_empty() {
            pareto.push(plans.into_iter().next().unwrap());
        }
        pareto
    }

    /// Find a plan that prioritizes upgrades (typically lower risk).
    fn find_low_risk_plan(
        &self,
        start: &State,
        target: &State,
        max_depth: usize,
    ) -> Option<DeploymentPlan> {
        let mut visited: HashSet<State> = HashSet::new();
        let mut path = vec![start.clone()];
        visited.insert(start.clone());

        if self.dfs_low_risk(target, max_depth, &mut visited, &mut path) {
            let steps = states_to_steps(&path);
            Some(DeploymentPlan::new(start.clone(), target.clone(), steps))
        } else {
            None
        }
    }

    fn dfs_low_risk(
        &self,
        target: &State,
        max_depth: usize,
        visited: &mut HashSet<State>,
        path: &mut Vec<State>,
    ) -> bool {
        let current = path.last().unwrap().clone();
        if current == *target {
            return true;
        }
        if path.len() > max_depth {
            return false;
        }

        let mut successors = self.generate_successors(&current);
        successors.sort_by_key(|s| {
            let dist = s.distance(target);
            let upgrade_bonus: usize = s
                .versions
                .iter()
                .zip(current.versions.iter())
                .filter(|(new, old)| new.0 > old.0)
                .count();
            (dist, Reverse(upgrade_bonus))
        });

        for next in successors {
            if visited.contains(&next) {
                continue;
            }
            visited.insert(next.clone());
            path.push(next.clone());
            if self.dfs_low_risk(target, max_depth, visited, path) {
                return true;
            }
            path.pop();
        }
        false
    }

    fn generate_successors(&self, state: &State) -> Vec<State> {
        if !self.graph.adjacency.is_empty() && self.graph.adjacency.contains_key(state) {
            return self
                .graph
                .neighbors(state)
                .into_iter()
                .filter(|(s, _)| all_constraints_satisfied(s, self.constraints))
                .map(|(s, _)| s)
                .collect();
        }

        let mut out = Vec::new();
        for (s_idx, svc) in self.graph.services.iter().enumerate() {
            let si = ServiceIndex(s_idx as u16);
            let cur = state.get(si);
            for v in 0..svc.versions.len() as u16 {
                let vi = VersionIndex(v);
                if vi == cur {
                    continue;
                }
                let mut next = state.clone();
                next.set(si, vi);
                if all_constraints_satisfied(&next, self.constraints) {
                    out.push(next);
                }
            }
        }
        out
    }

    fn reconstruct(came_from: &HashMap<State, Option<State>>, target: &State) -> Vec<State> {
        let mut path = vec![target.clone()];
        let mut current = target.clone();
        while let Some(Some(prev)) = came_from.get(&current) {
            path.push(prev.clone());
            current = prev.clone();
        }
        path.reverse();
        path
    }
}

// ---------------------------------------------------------------------------
// Helper: enumerate states up to a per-service cap for downward closure check
// ---------------------------------------------------------------------------

fn enumerate_states_bounded(version_counts: &[u16], per_service_cap: u16) -> Vec<State> {
    let n = version_counts.len();
    if n == 0 {
        return vec![State::new(vec![])];
    }
    let caps: Vec<u16> = version_counts.iter().map(|&c| c.min(per_service_cap)).collect();
    let total: usize = caps.iter().map(|&c| c as usize).product();
    if total > 100_000 {
        return Vec::new();
    }
    let mut states = Vec::with_capacity(total);
    let mut indices = vec![0u16; n];
    loop {
        states.push(State::new(indices.iter().map(|&i| VersionIndex(i)).collect()));
        let mut carry = true;
        for dim in (0..n).rev() {
            if carry {
                indices[dim] += 1;
                if indices[dim] >= caps[dim] {
                    indices[dim] = 0;
                } else {
                    carry = false;
                }
            }
        }
        if carry {
            break;
        }
    }
    states
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_simple_graph() -> (VersionProductGraph, Vec<Constraint>) {
        let svc_a = ServiceDescriptor::new("svc-a", vec!["v1".into(), "v2".into()]);
        let svc_b = ServiceDescriptor::new("svc-b", vec!["v1".into(), "v2".into()]);
        let graph = VersionProductGraph::new(vec![svc_a, svc_b]);
        let constraints = vec![];
        (graph, constraints)
    }

    fn make_constrained_graph() -> (VersionProductGraph, Vec<Constraint>) {
        let svc_a = ServiceDescriptor::new("svc-a", vec!["v1".into(), "v2".into()]);
        let svc_b = ServiceDescriptor::new("svc-b", vec!["v1".into(), "v2".into()]);
        let graph = VersionProductGraph::new(vec![svc_a, svc_b]);
        let constraints = vec![Constraint::Compatibility {
            id: Id::from_name("compat-ab"),
            service_a: ServiceIndex(0),
            service_b: ServiceIndex(1),
            compatible_pairs: vec![
                (VersionIndex(0), VersionIndex(0)),
                (VersionIndex(0), VersionIndex(1)),
                (VersionIndex(1), VersionIndex(1)),
            ],
        }];
        (graph, constraints)
    }

    fn make_three_service_graph() -> (VersionProductGraph, Vec<Constraint>) {
        let svc_a = ServiceDescriptor::new("a", vec!["v1".into(), "v2".into(), "v3".into()]);
        let svc_b = ServiceDescriptor::new("b", vec!["v1".into(), "v2".into(), "v3".into()]);
        let svc_c = ServiceDescriptor::new("c", vec!["v1".into(), "v2".into()]);
        let graph = VersionProductGraph::new(vec![svc_a, svc_b, svc_c]);
        let constraints = vec![];
        (graph, constraints)
    }

    #[test]
    fn test_trivial_plan() {
        let (graph, constraints) = make_simple_graph();
        let state = State::new(vec![VersionIndex(0), VersionIndex(0)]);
        let mut planner = DeploymentPlanner::new(PlannerConfig::default(), graph, constraints);
        let result = planner.plan(&state, &state);
        assert!(result.is_plan());
        let plan = result.into_plan().unwrap();
        assert_eq!(plan.step_count(), 0);
    }

    #[test]
    fn test_single_step_plan() {
        let (graph, constraints) = make_simple_graph();
        let start = State::new(vec![VersionIndex(0), VersionIndex(0)]);
        let target = State::new(vec![VersionIndex(1), VersionIndex(0)]);
        let mut planner = DeploymentPlanner::new(PlannerConfig::default(), graph, constraints);
        let result = planner.plan(&start, &target);
        assert!(result.is_plan());
        let plan = result.into_plan().unwrap();
        assert_eq!(plan.step_count(), 1);
    }

    #[test]
    fn test_multi_step_plan() {
        let (graph, constraints) = make_simple_graph();
        let start = State::new(vec![VersionIndex(0), VersionIndex(0)]);
        let target = State::new(vec![VersionIndex(1), VersionIndex(1)]);
        let mut planner = DeploymentPlanner::new(PlannerConfig::default(), graph, constraints);
        let result = planner.plan(&start, &target);
        assert!(result.is_plan());
        let plan = result.into_plan().unwrap();
        assert!(plan.step_count() >= 2);
        assert!(plan.validate_consistency());
    }

    #[test]
    fn test_constrained_plan_avoids_unsafe() {
        let (graph, constraints) = make_constrained_graph();
        let start = State::new(vec![VersionIndex(0), VersionIndex(0)]);
        let target = State::new(vec![VersionIndex(1), VersionIndex(1)]);
        let mut planner = DeploymentPlanner::new(PlannerConfig::default(), graph, constraints);
        let result = planner.plan(&start, &target);
        assert!(result.is_plan());
        let plan = result.into_plan().unwrap();
        assert!(plan.validate_consistency());
        let intermediates = plan.intermediate_states();
        let bad = State::new(vec![VersionIndex(1), VersionIndex(0)]);
        assert!(!intermediates.contains(&bad));
    }

    #[test]
    fn test_infeasible_forbidden_start() {
        let svc = ServiceDescriptor::new("svc-a", vec!["v1".into()]);
        let graph = VersionProductGraph::new(vec![svc]);
        let start = State::new(vec![VersionIndex(0)]);
        let target = State::new(vec![VersionIndex(0)]);
        let constraints = vec![Constraint::Forbidden {
            id: Id::from_name("forbid"),
            service: ServiceIndex(0),
            version: VersionIndex(0),
        }];
        let mut planner = DeploymentPlanner::new(PlannerConfig::default(), graph, constraints);
        let result = planner.plan(&start, &target);
        assert!(result.is_infeasible());
    }

    #[test]
    fn test_plan_timeout_succeeds_quickly() {
        let (graph, constraints) = make_simple_graph();
        let start = State::new(vec![VersionIndex(0), VersionIndex(0)]);
        let target = State::new(vec![VersionIndex(1), VersionIndex(1)]);
        let mut planner = DeploymentPlanner::new(PlannerConfig::default(), graph, constraints);
        let result = planner.plan_with_timeout(&start, &target, Duration::from_secs(10));
        assert!(result.is_plan());
    }

    #[test]
    fn test_greedy_planner_simple() {
        let (graph, constraints) = make_simple_graph();
        let greedy = GreedyPlanner::new(&graph, &constraints);
        let start = State::new(vec![VersionIndex(0), VersionIndex(0)]);
        let target = State::new(vec![VersionIndex(1), VersionIndex(1)]);
        let plan = greedy.plan(&start, &target);
        assert!(plan.is_some());
        assert!(plan.unwrap().validate_consistency());
    }

    #[test]
    fn test_greedy_planner_constrained() {
        let (graph, constraints) = make_constrained_graph();
        let greedy = GreedyPlanner::new(&graph, &constraints);
        let start = State::new(vec![VersionIndex(0), VersionIndex(0)]);
        let target = State::new(vec![VersionIndex(1), VersionIndex(1)]);
        let result = greedy.plan(&start, &target);
        if let Some(plan) = result {
            assert!(plan.validate_consistency());
        }
    }

    #[test]
    fn test_optimal_planner_shortest() {
        let (graph, constraints) = make_simple_graph();
        let optimal = OptimalPlanner::new(&graph, &constraints);
        let start = State::new(vec![VersionIndex(0), VersionIndex(0)]);
        let target = State::new(vec![VersionIndex(1), VersionIndex(1)]);
        let result = optimal.find_optimal(&start, &target, 10);
        assert!(result.is_plan());
        let plan = result.into_plan().unwrap();
        assert_eq!(plan.step_count(), 2);
    }

    #[test]
    fn test_optimal_planner_pareto() {
        let (graph, constraints) = make_simple_graph();
        let optimal = OptimalPlanner::new(&graph, &constraints);
        let start = State::new(vec![VersionIndex(0), VersionIndex(0)]);
        let target = State::new(vec![VersionIndex(1), VersionIndex(1)]);
        let pareto = optimal.find_pareto(&start, &target, 10);
        assert!(!pareto.is_empty());
        for plan in &pareto {
            assert!(plan.validate_consistency());
        }
    }

    #[test]
    fn test_treewidth_estimator_empty() {
        let (graph, _) = make_simple_graph();
        let tw = TreewidthEstimator::estimate(&graph, &[]);
        assert_eq!(tw, 0);
    }

    #[test]
    fn test_treewidth_estimator_clique() {
        let svc_a = ServiceDescriptor::new("a", vec!["v1".into(), "v2".into()]);
        let svc_b = ServiceDescriptor::new("b", vec!["v1".into(), "v2".into()]);
        let svc_c = ServiceDescriptor::new("c", vec!["v1".into(), "v2".into()]);
        let graph = VersionProductGraph::new(vec![svc_a, svc_b, svc_c]);
        let constraints = vec![
            Constraint::Compatibility {
                id: Id::from_name("ab"),
                service_a: ServiceIndex(0),
                service_b: ServiceIndex(1),
                compatible_pairs: vec![(VersionIndex(0), VersionIndex(0))],
            },
            Constraint::Compatibility {
                id: Id::from_name("bc"),
                service_a: ServiceIndex(1),
                service_b: ServiceIndex(2),
                compatible_pairs: vec![(VersionIndex(0), VersionIndex(0))],
            },
            Constraint::Compatibility {
                id: Id::from_name("ac"),
                service_a: ServiceIndex(0),
                service_b: ServiceIndex(2),
                compatible_pairs: vec![(VersionIndex(0), VersionIndex(0))],
            },
        ];
        let tw = TreewidthEstimator::estimate(&graph, &constraints);
        assert_eq!(tw, 2);
    }

    #[test]
    fn test_planner_three_services() {
        let (graph, constraints) = make_three_service_graph();
        let start = State::new(vec![VersionIndex(0), VersionIndex(0), VersionIndex(0)]);
        let target = State::new(vec![VersionIndex(2), VersionIndex(2), VersionIndex(1)]);
        let mut planner = DeploymentPlanner::new(PlannerConfig::default(), graph, constraints);
        let result = planner.plan(&start, &target);
        assert!(result.is_plan());
        let plan = result.into_plan().unwrap();
        assert!(plan.validate_consistency());
    }

    #[test]
    fn test_plan_with_ordering_constraint() {
        let svc_a = ServiceDescriptor::new("a", vec!["v1".into(), "v2".into()]);
        let svc_b = ServiceDescriptor::new("b", vec!["v1".into(), "v2".into()]);
        let graph = VersionProductGraph::new(vec![svc_a, svc_b]);
        let constraints = vec![Constraint::Ordering {
            id: Id::from_name("order-ab"),
            before: ServiceIndex(0),
            after: ServiceIndex(1),
        }];
        let start = State::new(vec![VersionIndex(0), VersionIndex(0)]);
        let target = State::new(vec![VersionIndex(1), VersionIndex(1)]);
        let greedy = GreedyPlanner::new(&graph, &constraints);
        let plan = greedy.plan(&start, &target);
        assert!(plan.is_some());
        let plan = plan.unwrap();
        let svc0_idx = plan.steps.iter().position(|s| s.service == ServiceIndex(0));
        let svc1_idx = plan.steps.iter().position(|s| s.service == ServiceIndex(1));
        if let (Some(a), Some(b)) = (svc0_idx, svc1_idx) {
            assert!(a < b);
        }
    }

    #[test]
    fn test_is_safe_state() {
        let (graph, constraints) = make_constrained_graph();
        let planner = DeploymentPlanner::new(PlannerConfig::default(), graph, constraints);
        let safe = State::new(vec![VersionIndex(0), VersionIndex(1)]);
        let unsafe_state = State::new(vec![VersionIndex(1), VersionIndex(0)]);
        assert!(planner.is_safe_state(&safe));
        assert!(!planner.is_safe_state(&unsafe_state));
    }

    #[test]
    fn test_neighbors_monotone() {
        let (graph, constraints) = make_simple_graph();
        let planner = DeploymentPlanner::new(PlannerConfig::default(), graph, constraints);
        let state = State::new(vec![VersionIndex(1), VersionIndex(0)]);
        let mono_neighbors = planner.neighbors(&state, true);
        for (_, _, old_v, new_v) in &mono_neighbors {
            assert!(new_v.0 >= old_v.0);
        }
        let all_neighbors = planner.neighbors(&state, false);
        assert!(all_neighbors.len() >= mono_neighbors.len());
    }

    #[test]
    fn test_completeness_bound() {
        let (graph, constraints) = make_simple_graph();
        let planner = DeploymentPlanner::new(PlannerConfig::default(), graph, constraints);
        let start = State::new(vec![VersionIndex(0), VersionIndex(0)]);
        let target = State::new(vec![VersionIndex(1), VersionIndex(1)]);
        let bound = planner.compute_completeness_bound(&start, &target);
        assert!(bound >= 2);
    }

    #[test]
    fn test_downward_closure_unconstrained() {
        let (graph, constraints) = make_simple_graph();
        let planner = DeploymentPlanner::new(PlannerConfig::default(), graph, constraints);
        assert!(planner.check_downward_closure());
    }

    #[test]
    fn test_plan_step_info() {
        let info = PlanStepInfo::new(0, 1, 3, 0);
        assert!(info.is_upgrade());
        let step = info.to_plan_step();
        assert_eq!(step.service, ServiceIndex(0));
        assert_eq!(step.from_version, VersionIndex(1));
        assert_eq!(step.to_version, VersionIndex(3));
    }

    #[test]
    fn test_plan_result_methods() {
        let result = PlanResult::Timeout;
        assert!(!result.is_plan());
        assert!(result.is_timeout());
        assert!(result.into_plan().is_none());

        let plan = DeploymentPlan::new(
            State::new(vec![VersionIndex(0)]),
            State::new(vec![VersionIndex(0)]),
            vec![],
        );
        let result = PlanResult::Plan(plan);
        assert!(result.is_plan());
        assert!(!result.is_infeasible());
    }

    #[test]
    fn test_planner_state() {
        let mut state = PlannerState::new(50);
        assert_eq!(state.current_depth, 0);
        assert_eq!(state.max_depth, 50);
        state.start_time = Some(Instant::now());
        assert!(!state.timed_out(Duration::from_secs(10)));
    }

    #[test]
    fn test_opt_goal_default() {
        assert_eq!(OptGoal::default(), OptGoal::MinSteps);
    }

    #[test]
    fn test_bfs_plan_direct() {
        let (graph, constraints) = make_simple_graph();
        let start = State::new(vec![VersionIndex(0), VersionIndex(0)]);
        let target = State::new(vec![VersionIndex(1), VersionIndex(1)]);
        let mut planner = DeploymentPlanner::new(PlannerConfig::default(), graph, constraints);
        let result = planner.plan_bfs(&start, &target, 10, Duration::from_secs(5));
        assert!(result.is_plan());
        let plan = result.into_plan().unwrap();
        assert_eq!(plan.step_count(), 2);
    }

    #[test]
    fn test_states_to_step_infos() {
        let s0 = State::new(vec![VersionIndex(0), VersionIndex(0)]);
        let s1 = State::new(vec![VersionIndex(1), VersionIndex(0)]);
        let s2 = State::new(vec![VersionIndex(1), VersionIndex(1)]);
        let infos = states_to_step_infos(&[s0, s1, s2]);
        assert_eq!(infos.len(), 2);
        assert_eq!(infos[0].service_idx, 0);
        assert_eq!(infos[1].service_idx, 1);
    }

    #[test]
    fn test_enumerate_states_bounded() {
        let counts = vec![2u16, 3u16];
        let states = enumerate_states_bounded(&counts, 5);
        assert_eq!(states.len(), 6);
    }

    #[test]
    fn test_planner_stats_default() {
        let stats = PlannerStats::default();
        assert_eq!(stats.states_explored, 0);
        assert_eq!(stats.plan_length, 0);
    }

    #[test]
    fn test_infeasibility_witness_builder() {
        let start = State::new(vec![VersionIndex(0)]);
        let target = State::new(vec![VersionIndex(1)]);
        let w = InfeasibilityWitness::new("blocked", start, target)
            .with_explored(42)
            .with_blocking(vec![Id::from_name("c1")]);
        assert_eq!(w.explored_states, 42);
        assert_eq!(w.blocking_constraints.len(), 1);
    }

    #[test]
    fn test_greedy_trivial() {
        let (graph, constraints) = make_simple_graph();
        let greedy = GreedyPlanner::new(&graph, &constraints);
        let state = State::new(vec![VersionIndex(0), VersionIndex(0)]);
        let plan = greedy.plan(&state, &state);
        assert!(plan.is_some());
        assert_eq!(plan.unwrap().step_count(), 0);
    }

    #[test]
    fn test_constrained_three_services() {
        let svc_a = ServiceDescriptor::new("a", vec!["v1".into(), "v2".into()]);
        let svc_b = ServiceDescriptor::new("b", vec!["v1".into(), "v2".into()]);
        let svc_c = ServiceDescriptor::new("c", vec!["v1".into(), "v2".into()]);
        let graph = VersionProductGraph::new(vec![svc_a, svc_b, svc_c]);
        let constraints = vec![
            Constraint::Compatibility {
                id: Id::from_name("ab"),
                service_a: ServiceIndex(0),
                service_b: ServiceIndex(1),
                compatible_pairs: vec![
                    (VersionIndex(0), VersionIndex(0)),
                    (VersionIndex(0), VersionIndex(1)),
                    (VersionIndex(1), VersionIndex(1)),
                ],
            },
            Constraint::Compatibility {
                id: Id::from_name("bc"),
                service_a: ServiceIndex(1),
                service_b: ServiceIndex(2),
                compatible_pairs: vec![
                    (VersionIndex(0), VersionIndex(0)),
                    (VersionIndex(1), VersionIndex(0)),
                    (VersionIndex(1), VersionIndex(1)),
                ],
            },
        ];
        let start = State::new(vec![VersionIndex(0), VersionIndex(0), VersionIndex(0)]);
        let target = State::new(vec![VersionIndex(1), VersionIndex(1), VersionIndex(1)]);
        let mut planner = DeploymentPlanner::new(PlannerConfig::default(), graph, constraints);
        let result = planner.plan(&start, &target);
        assert!(result.is_plan());
        let plan = result.into_plan().unwrap();
        assert!(plan.validate_consistency());
        assert!(plan.step_count() >= 3);
    }
}
