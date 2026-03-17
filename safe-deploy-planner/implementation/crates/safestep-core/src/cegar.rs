//! CEGAR (Counterexample-Guided Abstraction Refinement) engine for SafeStep.
//!
//! Splits the deployment planning problem into:
//! - An **abstract** layer handling Boolean/structural constraints (compatibility,
//!   ordering, forbidden states) solved via BFS on the version-product graph.
//! - A **concrete** checker verifying resource/numeric constraints on candidate plans.
//!
//! The CEGAR loop iterates: abstract-solve → concrete-check → refine, until a
//! real plan is found, infeasibility is proven, or a budget is exhausted.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tracing::{debug, info, trace, warn};

use safestep_types::identifiers::{ConstraintId, Id};

use crate::{
    Constraint, CoreResult, DeploymentPlan, Edge, PlanStep, PlannerConfig,
    ServiceIndex, State, TransitionMetadata, VersionIndex, VersionProductGraph,
};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration knobs for the CEGAR engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CegarConfig {
    /// Maximum refinement iterations before giving up.
    pub max_iterations: usize,
    /// Wall-clock timeout for the whole CEGAR loop.
    pub timeout: Duration,
    /// Maximum BFS depth when searching for abstract plans.
    pub max_depth: usize,
    /// When true, log every refinement step at DEBUG level.
    pub verbose: bool,
    /// Maximum number of blocked plans to retain (older ones are dropped).
    pub max_blocked_plans: usize,
}

impl Default for CegarConfig {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            timeout: Duration::from_secs(300),
            max_depth: 100,
            verbose: false,
            max_blocked_plans: 10_000,
        }
    }
}

impl CegarConfig {
    /// Build a `CegarConfig` from the generic `PlannerConfig`.
    pub fn from_planner_config(pc: &PlannerConfig) -> Self {
        Self {
            max_iterations: pc.max_cegar_iterations,
            timeout: pc.timeout,
            max_depth: pc.max_depth,
            verbose: false,
            max_blocked_plans: 10_000,
        }
    }
}

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Outcome of a CEGAR run.
#[derive(Debug, Clone)]
pub enum CegarResult {
    /// A concrete (fully-checked) deployment plan.
    RealPlan(DeploymentPlan),
    /// The abstract problem itself is infeasible — no plan exists even ignoring
    /// resource constraints.
    Infeasible(String),
    /// Wall-clock timeout was hit before convergence.
    Timeout,
    /// Exhausted the iteration budget without finding a plan or proving infeasibility.
    MaxIterations(usize),
}

impl fmt::Display for CegarResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CegarResult::RealPlan(p) => write!(f, "RealPlan({} steps)", p.step_count()),
            CegarResult::Infeasible(reason) => write!(f, "Infeasible: {reason}"),
            CegarResult::Timeout => write!(f, "Timeout"),
            CegarResult::MaxIterations(n) => write!(f, "MaxIterations({n})"),
        }
    }
}

// ---------------------------------------------------------------------------
// Abstract constraint representation
// ---------------------------------------------------------------------------

/// A simplified Boolean constraint used in the abstract layer.
///
/// Resource constraints are deliberately *not* represented here — they are
/// deferred to the concrete checker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AbstractConstraint {
    /// Two services must be at one of the listed compatible version pairs.
    CompatPair {
        svc_a: ServiceIndex,
        svc_b: ServiceIndex,
        allowed: Vec<(VersionIndex, VersionIndex)>,
    },
    /// A specific (service, version) combination is forbidden.
    ForbiddenState {
        svc: ServiceIndex,
        ver: VersionIndex,
    },
    /// Service `before` must reach its target version before `after` begins
    /// to change.
    Order {
        before: ServiceIndex,
        after: ServiceIndex,
    },
    /// Custom predicate — wrapped as an opaque function pointer.
    CustomPredicate {
        id: String,
        #[serde(skip, default = "default_abstract_predicate")]
        check: fn(&State) -> bool,
    },
}

fn default_abstract_predicate() -> fn(&State) -> bool {
    |_| false
}

impl AbstractConstraint {
    /// Check whether a *single state* satisfies this abstract constraint.
    /// Ordering constraints are plan-level and always pass at the state level.
    pub fn check_state(&self, state: &State) -> bool {
        match self {
            AbstractConstraint::CompatPair { svc_a, svc_b, allowed } => {
                let va = state.get(*svc_a);
                let vb = state.get(*svc_b);
                allowed.contains(&(va, vb))
            }
            AbstractConstraint::ForbiddenState { svc, ver } => {
                state.get(*svc) != *ver
            }
            AbstractConstraint::Order { .. } => true,
            AbstractConstraint::CustomPredicate { check, .. } => check(state),
        }
    }

    /// Check an ordering constraint against a full plan.
    /// Returns `true` if the constraint is satisfied or is not an ordering constraint.
    pub fn check_plan_order(&self, steps: &[PlanStep]) -> bool {
        match self {
            AbstractConstraint::Order { before, after } => {
                let before_pos = steps.iter().position(|s| s.service == *before);
                let after_pos = steps.iter().position(|s| s.service == *after);
                match (before_pos, after_pos) {
                    (Some(b), Some(a)) => b < a,
                    _ => true,
                }
            }
            _ => true,
        }
    }
}

// ---------------------------------------------------------------------------
// BlockedPlan
// ---------------------------------------------------------------------------

/// Records a state-sequence that has been proven spurious and must not be
/// reused as a contiguous sub-path in future abstract plans.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlockedPlan {
    /// The contiguous sub-sequence of states that is blocked.
    pub states: Vec<State>,
}

impl BlockedPlan {
    pub fn new(states: Vec<State>) -> Self {
        Self { states }
    }

    /// Returns `true` if `candidate_states` contains `self.states` as a
    /// contiguous sub-sequence.
    pub fn is_blocked(&self, candidate_states: &[State]) -> bool {
        if self.states.is_empty() || candidate_states.len() < self.states.len() {
            return false;
        }
        candidate_states
            .windows(self.states.len())
            .any(|window| window == self.states.as_slice())
    }

    /// Length of the blocked sub-path.
    pub fn len(&self) -> usize {
        self.states.len()
    }

    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }
}

impl fmt::Display for BlockedPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Blocked[")?;
        for (i, s) in self.states.iter().enumerate() {
            if i > 0 {
                write!(f, " -> ")?;
            }
            write!(f, "{s}")?;
        }
        write!(f, "]")
    }
}

// ---------------------------------------------------------------------------
// Abstraction
// ---------------------------------------------------------------------------

/// The abstract problem: a reference to the graph plus the Boolean constraints.
///
/// The abstraction deliberately *ignores* resource constraints so that the BFS
/// solver only needs to reason about discrete feasibility.
#[derive(Debug, Clone)]
pub struct Abstraction {
    /// Adjacency list representation (state → list of (neighbour, edge_index)).
    adjacency: HashMap<State, Vec<(State, usize)>>,
    /// Abstract constraints extracted from the full constraint set.
    constraints: Vec<AbstractConstraint>,
    /// All states in the graph.
    states: Vec<State>,
    /// Edge list (index-aligned with VersionProductGraph.edges).
    edges: Vec<Edge>,
    /// Number of services.
    service_count: usize,
}

impl Abstraction {
    /// Build an abstraction from the full graph and constraint set.
    pub fn new(graph: &VersionProductGraph, constraints: &[Constraint]) -> Self {
        let mut adjacency: HashMap<State, Vec<(State, usize)>> = HashMap::new();
        for state in &graph.states {
            adjacency.entry(state.clone()).or_default();
        }
        for (idx, edge) in graph.edges.iter().enumerate() {
            adjacency
                .entry(edge.from.clone())
                .or_default()
                .push((edge.to.clone(), idx));
        }

        let abstract_constraints = Self::abstract_constraints(constraints);

        Self {
            adjacency,
            constraints: abstract_constraints,
            states: graph.states.clone(),
            edges: graph.edges.clone(),
            service_count: graph.service_count(),
        }
    }

    /// Extract the Boolean/structural subset of constraints, discarding
    /// resource constraints (which are handled by the concrete checker).
    pub fn abstract_constraints(constraints: &[Constraint]) -> Vec<AbstractConstraint> {
        let mut result = Vec::new();
        for c in constraints {
            match c {
                Constraint::Compatibility {
                    service_a,
                    service_b,
                    compatible_pairs,
                    ..
                } => {
                    result.push(AbstractConstraint::CompatPair {
                        svc_a: *service_a,
                        svc_b: *service_b,
                        allowed: compatible_pairs.clone(),
                    });
                }
                Constraint::Forbidden { service, version, .. } => {
                    result.push(AbstractConstraint::ForbiddenState {
                        svc: *service,
                        ver: *version,
                    });
                }
                Constraint::Ordering { before, after, .. } => {
                    result.push(AbstractConstraint::Order {
                        before: *before,
                        after: *after,
                    });
                }
                Constraint::Custom { id, check, .. } => {
                    result.push(AbstractConstraint::CustomPredicate {
                        id: id.as_str().to_owned(),
                        check: *check,
                    });
                }
                Constraint::Resource { .. } => {
                    // deliberately omitted from the abstraction
                }
            }
        }
        result
    }

    /// Check whether a single state satisfies *all* abstract constraints.
    pub fn is_abstract_feasible(&self, state: &State) -> bool {
        self.constraints.iter().all(|c| c.check_state(state))
    }

    /// BFS on the abstract graph, returning a state-path from `start` to
    /// `target` that satisfies all abstract (non-resource) constraints and
    /// avoids every blocked sub-path.
    ///
    /// Returns `None` if no such path exists within `max_depth` steps.
    pub fn find_abstract_plan(
        &self,
        start: &State,
        target: &State,
        blocked: &[BlockedPlan],
        max_depth: usize,
    ) -> Option<Vec<State>> {
        if start == target {
            if self.is_abstract_feasible(start) {
                return Some(vec![start.clone()]);
            } else {
                return None;
            }
        }

        // BFS: queue entries are (current_state, path_so_far)
        let mut queue: VecDeque<(State, Vec<State>)> = VecDeque::new();
        let mut visited: HashSet<State> = HashSet::new();

        if !self.is_abstract_feasible(start) {
            return None;
        }

        queue.push_back((start.clone(), vec![start.clone()]));
        visited.insert(start.clone());

        while let Some((current, path)) = queue.pop_front() {
            if path.len() > max_depth + 1 {
                continue;
            }

            let neighbors = self.adjacency.get(&current).cloned().unwrap_or_default();
            for (next_state, _edge_idx) in &neighbors {
                if visited.contains(next_state) {
                    continue;
                }

                if !self.is_abstract_feasible(next_state) {
                    continue;
                }

                let mut new_path = path.clone();
                new_path.push(next_state.clone());

                // Check that the new path doesn't contain any blocked sub-sequence.
                let dominated = blocked.iter().any(|bp| bp.is_blocked(&new_path));
                if dominated {
                    continue;
                }

                if next_state == target {
                    // Verify ordering constraints on the derived steps.
                    let steps = self.path_to_steps(&new_path);
                    let order_ok = self.constraints.iter().all(|c| c.check_plan_order(&steps));
                    if order_ok {
                        return Some(new_path);
                    }
                    // ordering violated — keep searching
                    continue;
                }

                visited.insert(next_state.clone());
                queue.push_back((next_state.clone(), new_path));
            }
        }

        None
    }

    /// Convert a path of states into a sequence of `PlanStep`s.
    fn path_to_steps(&self, path: &[State]) -> Vec<PlanStep> {
        let mut steps = Vec::new();
        for window in path.windows(2) {
            let from_state = &window[0];
            let to_state = &window[1];
            let diffs = from_state.diff_services(to_state);
            for svc in diffs {
                let from_ver = from_state.get(svc);
                let to_ver = to_state.get(svc);
                // Look up risk from edge metadata if available.
                let mut step = PlanStep::new(svc, from_ver, to_ver);
                if let Some(edge) = self.find_edge(from_state, to_state) {
                    step = step.with_risk(edge.metadata.risk_score);
                    step = step.with_duration(edge.metadata.estimated_duration_secs);
                    step.requires_downtime = edge.metadata.requires_downtime;
                }
                steps.push(step);
            }
        }
        steps
    }

    /// Find the edge connecting two states (if one exists).
    fn find_edge(&self, from: &State, to: &State) -> Option<&Edge> {
        if let Some(neighbors) = self.adjacency.get(from) {
            for (nb, edge_idx) in neighbors {
                if nb == to {
                    return Some(&self.edges[*edge_idx]);
                }
            }
        }
        None
    }

    /// Total number of abstract constraints.
    pub fn constraint_count(&self) -> usize {
        self.constraints.len()
    }

    /// How many states in the abstract graph.
    pub fn state_count(&self) -> usize {
        self.states.len()
    }
}

// ---------------------------------------------------------------------------
// ConcreteChecker
// ---------------------------------------------------------------------------

/// Result of a concrete feasibility check.
#[derive(Debug, Clone)]
pub enum ConcreteCheckResult {
    /// Every state along the plan satisfies all resource constraints.
    Feasible,
    /// At least one state violates a resource constraint.
    Infeasible {
        step_index: usize,
        constraint_id: ConstraintId,
        violation_detail: String,
    },
}

impl ConcreteCheckResult {
    pub fn is_feasible(&self) -> bool {
        matches!(self, ConcreteCheckResult::Feasible)
    }
}

impl fmt::Display for ConcreteCheckResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConcreteCheckResult::Feasible => write!(f, "Feasible"),
            ConcreteCheckResult::Infeasible {
                step_index,
                constraint_id,
                violation_detail,
            } => write!(
                f,
                "Infeasible at step {step_index} (constraint {constraint_id}): {violation_detail}",
                constraint_id = constraint_id.as_str()
            ),
        }
    }
}

/// Checks a candidate plan against the *full* constraint set, including
/// resource/numeric constraints that the abstract layer ignores.
#[derive(Debug, Clone)]
pub struct ConcreteChecker {
    /// All constraints (we only *need* the resource ones, but we re-check
    /// everything for safety).
    constraints: Vec<Constraint>,
    /// Indices into `constraints` that are resource constraints (for fast
    /// identification of the failing class).
    resource_indices: Vec<usize>,
}

impl ConcreteChecker {
    pub fn new(constraints: &[Constraint]) -> Self {
        let resource_indices = constraints
            .iter()
            .enumerate()
            .filter(|(_, c)| matches!(c, Constraint::Resource { .. }))
            .map(|(i, _)| i)
            .collect();
        Self {
            constraints: constraints.to_vec(),
            resource_indices,
        }
    }

    /// Run every constraint against every state in the plan path.
    ///
    /// Returns `Feasible` if all states pass, or `Infeasible` with the first
    /// violation found.
    pub fn check(&self, plan_states: &[State]) -> ConcreteCheckResult {
        for (step_idx, state) in plan_states.iter().enumerate() {
            for constraint in &self.constraints {
                if !constraint.check_state(state) {
                    let detail = self.describe_violation(constraint, state);
                    return ConcreteCheckResult::Infeasible {
                        step_index: step_idx,
                        constraint_id: constraint.id().clone(),
                        violation_detail: detail,
                    };
                }
            }
        }
        ConcreteCheckResult::Feasible
    }

    /// Scan for the first state that violates a resource constraint specifically.
    /// Returns `(step_index, constraint_id)` or `None` if all resource
    /// constraints pass.
    pub fn find_violating_state(
        &self,
        plan_states: &[State],
    ) -> Option<(usize, ConstraintId)> {
        for (step_idx, state) in plan_states.iter().enumerate() {
            for &ci in &self.resource_indices {
                let c = &self.constraints[ci];
                if !c.check_state(state) {
                    return Some((step_idx, c.id().clone()));
                }
            }
        }
        None
    }

    /// Check all constraints against a single state, returning the ids of all
    /// violated constraints.
    pub fn violated_constraints(&self, state: &State) -> Vec<ConstraintId> {
        self.constraints
            .iter()
            .filter(|c| !c.check_state(state))
            .map(|c| c.id().clone())
            .collect()
    }

    /// Produce a human-readable description of why `constraint` is violated
    /// by `state`.
    fn describe_violation(&self, constraint: &Constraint, state: &State) -> String {
        match constraint {
            Constraint::Compatibility {
                service_a,
                service_b,
                ..
            } => {
                let va = state.get(*service_a);
                let vb = state.get(*service_b);
                format!(
                    "Incompatible versions: {service_a}={va}, {service_b}={vb}"
                )
            }
            Constraint::Resource {
                resource_name,
                max_budget,
                per_service_cost,
                ..
            } => {
                let total: f64 = state
                    .versions
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| {
                        per_service_cost
                            .get(&(ServiceIndex(i as u16), v))
                            .copied()
                            .unwrap_or(0.0)
                    })
                    .sum();
                format!(
                    "Resource '{resource_name}' exceeded: {total:.2} > {max_budget:.2}"
                )
            }
            Constraint::Forbidden { service, version, .. } => {
                format!("Forbidden version: {service}={version}")
            }
            Constraint::Ordering { before, after, .. } => {
                format!("Ordering violated: {before} must precede {after}")
            }
            Constraint::Custom { description, .. } => {
                format!("Custom constraint failed: {description}")
            }
        }
    }

    /// Number of resource constraints being tracked.
    pub fn resource_constraint_count(&self) -> usize {
        self.resource_indices.len()
    }
}

// ---------------------------------------------------------------------------
// Refinement
// ---------------------------------------------------------------------------

/// A refinement generated from a spurious counterexample.
///
/// When the abstract solver proposes a plan that fails the concrete check, we
/// create a `Refinement` that encodes *why* it failed and produces a
/// `BlockedPlan` so that the abstract solver never proposes the same (or a
/// closely-related) abstract path again.
#[derive(Debug, Clone)]
pub struct Refinement {
    /// The full state-sequence that was spurious.
    spurious_path: Vec<State>,
    /// Index of the step where the concrete violation was detected.
    failure_step: usize,
    /// Which constraint caused the failure.
    constraint_id: ConstraintId,
    /// Human-readable description.
    description: String,
}

impl Refinement {
    /// Create a refinement from a spurious counterexample.
    ///
    /// We identify the *minimal* sub-path around the failure that should be
    /// blocked: the state at `failure_step` and its immediate predecessor
    /// (if any), since the violation is typically caused by the transition
    /// into the offending state.
    pub fn from_spurious(
        plan_states: &[State],
        failure_step: usize,
        constraint_id: &ConstraintId,
    ) -> Self {
        let desc = format!(
            "Spurious at step {failure_step}: constraint {} violated on state {}",
            constraint_id.as_str(),
            plan_states
                .get(failure_step)
                .map(|s| s.to_string())
                .unwrap_or_else(|| "?".into()),
        );

        Self {
            spurious_path: plan_states.to_vec(),
            failure_step,
            constraint_id: constraint_id.clone(),
            description: desc,
        }
    }

    /// Produce the `BlockedPlan` that the abstract solver should respect.
    ///
    /// Strategy: block the sub-path of length min(3, path_len) centred on the
    /// failure step.  This is narrow enough to preserve reachability for
    /// unrelated paths while preventing the exact spurious counterexample from
    /// recurring.
    pub fn blocking_clause(&self) -> BlockedPlan {
        let path = &self.spurious_path;
        if path.is_empty() {
            return BlockedPlan::new(Vec::new());
        }

        // Determine a window around the failure step.
        let window_radius: usize = 1;
        let lo = if self.failure_step > window_radius {
            self.failure_step - window_radius
        } else {
            0
        };
        let hi = (self.failure_step + window_radius + 1).min(path.len());

        let sub = path[lo..hi].to_vec();
        BlockedPlan::new(sub)
    }

    pub fn description(&self) -> &str {
        &self.description
    }

    pub fn failure_step(&self) -> usize {
        self.failure_step
    }

    pub fn constraint_id(&self) -> &ConstraintId {
        &self.constraint_id
    }

    /// Returns the state that was the direct cause of the violation.
    pub fn violating_state(&self) -> Option<&State> {
        self.spurious_path.get(self.failure_step)
    }
}

impl fmt::Display for Refinement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Refinement(step={}, constraint={})",
            self.failure_step,
            self.constraint_id.as_str()
        )
    }
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Cumulative statistics for a CEGAR run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CegarStats {
    pub iterations: usize,
    pub refinements: usize,
    pub abstract_solve_time: Duration,
    pub concrete_check_time: Duration,
    pub total_time: Duration,
    pub blocked_plans_count: usize,
    pub abstract_plans_found: usize,
    pub concrete_failures: usize,
}

impl Default for CegarStats {
    fn default() -> Self {
        Self {
            iterations: 0,
            refinements: 0,
            abstract_solve_time: Duration::ZERO,
            concrete_check_time: Duration::ZERO,
            total_time: Duration::ZERO,
            blocked_plans_count: 0,
            abstract_plans_found: 0,
            concrete_failures: 0,
        }
    }
}

impl CegarStats {
    pub fn summary(&self) -> String {
        format!(
            "CEGAR: {} iterations, {} refinements, {:.1?} abstract, {:.1?} concrete, {:.1?} total | {} blocked, {} found, {} failed",
            self.iterations,
            self.refinements,
            self.abstract_solve_time,
            self.concrete_check_time,
            self.total_time,
            self.blocked_plans_count,
            self.abstract_plans_found,
            self.concrete_failures,
        )
    }

    /// Average abstract solve time per iteration (returns zero if none).
    pub fn avg_abstract_time(&self) -> Duration {
        if self.iterations == 0 {
            Duration::ZERO
        } else {
            self.abstract_solve_time / self.iterations as u32
        }
    }
}

impl fmt::Display for CegarStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.summary())
    }
}

// ---------------------------------------------------------------------------
// CegarEngine — main driver
// ---------------------------------------------------------------------------

/// The main CEGAR abstraction-refinement engine.
///
/// Usage:
/// ```ignore
/// let mut engine = CegarEngine::new(config, graph, constraints);
/// let result = engine.cegar_loop(&start_state, &target_state);
/// ```
pub struct CegarEngine {
    config: CegarConfig,
    abstraction: Abstraction,
    concrete_checker: ConcreteChecker,
    blocked_plans: Vec<BlockedPlan>,
    refinements: Vec<Refinement>,
    stats: CegarStats,
    /// Full constraint set (kept for plan construction).
    all_constraints: Vec<Constraint>,
    /// The graph (kept for step metadata lookup).
    graph: VersionProductGraph,
}

impl CegarEngine {
    /// Create a new engine from a config, the version-product graph, and the
    /// full set of constraints.
    pub fn new(
        config: CegarConfig,
        graph: VersionProductGraph,
        constraints: Vec<Constraint>,
    ) -> Self {
        let abstraction = Abstraction::new(&graph, &constraints);
        let concrete_checker = ConcreteChecker::new(&constraints);
        Self {
            config,
            abstraction,
            concrete_checker,
            blocked_plans: Vec::new(),
            refinements: Vec::new(),
            stats: CegarStats::default(),
            all_constraints: constraints,
            graph,
        }
    }

    // ----- public API -----

    /// Run the CEGAR loop from `start` to `target`.
    pub fn cegar_loop(&mut self, start: &State, target: &State) -> CegarResult {
        let global_start = Instant::now();
        info!(
            "CEGAR loop start: {} → {}, max_iter={}, timeout={:?}",
            start, target, self.config.max_iterations, self.config.timeout
        );

        // Quick check: if start == target and it's feasible, return trivially.
        if start == target {
            let states = vec![start.clone()];
            let check = self.concrete_checker.check(&states);
            if check.is_feasible() {
                let plan = DeploymentPlan::new(start.clone(), target.clone(), vec![]);
                self.stats.total_time = global_start.elapsed();
                return CegarResult::RealPlan(plan);
            }
        }

        for iteration in 0..self.config.max_iterations {
            self.stats.iterations = iteration + 1;

            // Timeout guard.
            if global_start.elapsed() >= self.config.timeout {
                self.stats.total_time = global_start.elapsed();
                warn!("CEGAR timeout after {} iterations", iteration);
                return CegarResult::Timeout;
            }

            debug!("CEGAR iteration {}: {} blocked plans", iteration, self.blocked_plans.len());

            // ---- Step 1: Abstract solve ----
            let abs_start = Instant::now();
            let abstract_plan = self.abstract_solve(start, target);
            self.stats.abstract_solve_time += abs_start.elapsed();

            let plan_states = match abstract_plan {
                Some(ps) => {
                    self.stats.abstract_plans_found += 1;
                    trace!("Abstract plan found: {} states", ps.len());
                    ps
                }
                None => {
                    // The abstract problem has no more solutions → truly infeasible.
                    self.stats.total_time = global_start.elapsed();
                    info!("CEGAR: abstract problem infeasible after {} iterations", iteration);
                    return CegarResult::Infeasible(format!(
                        "No abstract plan exists after {iteration} refinement(s) \
                         ({} plans blocked)",
                        self.blocked_plans.len()
                    ));
                }
            };

            // ---- Step 2: Concrete check ----
            let conc_start = Instant::now();
            let concrete_result = self.concrete_check(&plan_states);
            self.stats.concrete_check_time += conc_start.elapsed();

            match concrete_result {
                ConcreteCheckResult::Feasible => {
                    // Build the real plan and return.
                    let plan = self.build_plan(start, target, &plan_states);
                    self.stats.total_time = global_start.elapsed();
                    info!("CEGAR: real plan found in {} iterations, {} steps",
                          iteration + 1, plan.step_count());
                    return CegarResult::RealPlan(plan);
                }
                ConcreteCheckResult::Infeasible {
                    step_index,
                    constraint_id,
                    violation_detail,
                } => {
                    self.stats.concrete_failures += 1;
                    debug!(
                        "Spurious plan at step {step_index}: {violation_detail}"
                    );

                    // ---- Step 3: Refine ----
                    self.refine(&plan_states, step_index, &constraint_id);
                }
            }
        }

        self.stats.total_time = global_start.elapsed();
        warn!(
            "CEGAR: max iterations ({}) reached",
            self.config.max_iterations
        );
        CegarResult::MaxIterations(self.config.max_iterations)
    }

    /// Provide read access to statistics.
    pub fn stats(&self) -> &CegarStats {
        &self.stats
    }

    /// Blocked plans accumulated so far.
    pub fn blocked_plans(&self) -> &[BlockedPlan] {
        &self.blocked_plans
    }

    /// All refinements generated.
    pub fn refinements(&self) -> &[Refinement] {
        &self.refinements
    }

    /// Reset the engine state, keeping config and graph.
    pub fn reset(&mut self) {
        self.blocked_plans.clear();
        self.refinements.clear();
        self.stats = CegarStats::default();
    }

    // ----- internal steps -----

    /// Step 1 — solve the abstract problem (BFS on the graph with only
    /// Boolean/structural constraints and the current set of blocked plans).
    fn abstract_solve(&self, start: &State, target: &State) -> Option<Vec<State>> {
        self.abstraction
            .find_abstract_plan(start, target, &self.blocked_plans, self.config.max_depth)
    }

    /// Step 2 — check a candidate plan against *all* constraints including
    /// resource/numeric ones.
    fn concrete_check(&self, plan_states: &[State]) -> ConcreteCheckResult {
        self.concrete_checker.check(plan_states)
    }

    /// Step 3 — generate a refinement from a spurious counterexample and
    /// record the blocking clause.
    fn refine(
        &mut self,
        plan_states: &[State],
        failure_step: usize,
        constraint_id: &ConstraintId,
    ) {
        let refinement = Refinement::from_spurious(plan_states, failure_step, constraint_id);
        debug!("Refinement: {}", refinement.description());

        let blocked = refinement.blocking_clause();
        trace!("Blocked: {blocked}");

        // Evict oldest blocks if we exceed the cap.
        if self.blocked_plans.len() >= self.config.max_blocked_plans {
            let to_remove = self.blocked_plans.len() - self.config.max_blocked_plans + 1;
            self.blocked_plans.drain(0..to_remove);
        }

        self.blocked_plans.push(blocked);
        self.stats.blocked_plans_count = self.blocked_plans.len();
        self.stats.refinements += 1;
        self.refinements.push(refinement);
    }

    /// Convert a state-path into a `DeploymentPlan`.
    fn build_plan(
        &self,
        start: &State,
        target: &State,
        plan_states: &[State],
    ) -> DeploymentPlan {
        let mut steps: Vec<PlanStep> = Vec::new();
        for window in plan_states.windows(2) {
            let from = &window[0];
            let to = &window[1];
            let diffs = from.diff_services(to);
            for svc in diffs {
                let from_ver = from.get(svc);
                let to_ver = to.get(svc);
                let mut step = PlanStep::new(svc, from_ver, to_ver);
                // Annotate with edge metadata when available.
                if let Some(neighbors) = self.graph.adjacency.get(from) {
                    for &edge_idx in neighbors {
                        let edge = &self.graph.edges[edge_idx];
                        if edge.to == *to && edge.service == svc {
                            step = step
                                .with_risk(edge.metadata.risk_score)
                                .with_duration(edge.metadata.estimated_duration_secs);
                            step.requires_downtime = edge.metadata.requires_downtime;
                            break;
                        }
                    }
                }
                steps.push(step);
            }
        }
        DeploymentPlan::new(start.clone(), target.clone(), steps)
    }
}

impl fmt::Debug for CegarEngine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CegarEngine")
            .field("config", &self.config)
            .field("blocked_plans", &self.blocked_plans.len())
            .field("refinements", &self.refinements.len())
            .field("stats", &self.stats)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Helper: build a small test graph
// ---------------------------------------------------------------------------

/// Build a fully-connected version-product graph for services with given
/// version counts.  Each service can transition between adjacent versions
/// (i.e. v_i → v_{i±1}).  Useful for testing and small instances.
pub fn build_full_graph(version_counts: &[usize]) -> VersionProductGraph {
    use crate::ServiceDescriptor;

    let services: Vec<ServiceDescriptor> = version_counts
        .iter()
        .enumerate()
        .map(|(i, &vc)| {
            let versions: Vec<String> = (0..vc).map(|v| format!("v{v}")).collect();
            ServiceDescriptor::new(format!("svc-{i}"), versions)
        })
        .collect();

    let mut graph = VersionProductGraph::new(services);

    // Enumerate all states.
    let mut all_states: Vec<Vec<u16>> = vec![vec![]];
    for &vc in version_counts {
        let mut next = Vec::new();
        for partial in &all_states {
            for v in 0..vc as u16 {
                let mut ext = partial.clone();
                ext.push(v);
                next.push(ext);
            }
        }
        all_states = next;
    }

    for raw in &all_states {
        let state = State::new(raw.iter().map(|&v| VersionIndex(v)).collect());
        graph.add_state(state);
    }

    // Add edges: change exactly one service by ±1.
    for raw in &all_states {
        let from = State::new(raw.iter().map(|&v| VersionIndex(v)).collect());
        for (svc_idx, &vc) in version_counts.iter().enumerate() {
            let current = raw[svc_idx];
            let mut targets = Vec::new();
            if current > 0 {
                targets.push(current - 1);
            }
            if (current as usize) < vc - 1 {
                targets.push(current + 1);
            }
            for t in targets {
                let mut to_raw = raw.clone();
                to_raw[svc_idx] = t;
                let to = State::new(to_raw.iter().map(|&v| VersionIndex(v)).collect());
                let is_upgrade = t > current;
                let edge = Edge {
                    from: from.clone(),
                    to: to.clone(),
                    service: ServiceIndex(svc_idx as u16),
                    from_version: VersionIndex(current),
                    to_version: VersionIndex(t),
                    metadata: TransitionMetadata {
                        is_upgrade,
                        risk_score: if is_upgrade { 1 } else { 2 },
                        estimated_duration_secs: 30,
                        requires_downtime: false,
                    },
                };
                graph.add_edge(edge);
            }
        }
    }

    graph
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use safestep_types::identifiers::Id;

    // -- helpers --

    fn make_graph_2x2() -> VersionProductGraph {
        build_full_graph(&[2, 2])
    }

    fn make_graph_3x3() -> VersionProductGraph {
        build_full_graph(&[3, 3])
    }

    fn s(vals: &[u16]) -> State {
        State::new(vals.iter().map(|&v| VersionIndex(v)).collect())
    }

    // ---- Tests ----

    #[test]
    fn test_cegar_trivial_same_state() {
        let graph = make_graph_2x2();
        let config = CegarConfig::default();
        let constraints: Vec<Constraint> = vec![];
        let mut engine = CegarEngine::new(config, graph, constraints);
        let start = s(&[0, 0]);
        let result = engine.cegar_loop(&start, &start);
        match result {
            CegarResult::RealPlan(plan) => {
                assert_eq!(plan.step_count(), 0);
                assert!(plan.validate_consistency());
            }
            other => panic!("Expected RealPlan, got {other}"),
        }
    }

    #[test]
    fn test_cegar_simple_plan_no_constraints() {
        let graph = make_graph_2x2();
        let config = CegarConfig::default();
        let constraints: Vec<Constraint> = vec![];
        let mut engine = CegarEngine::new(config, graph, constraints);
        let start = s(&[0, 0]);
        let target = s(&[1, 1]);
        let result = engine.cegar_loop(&start, &target);
        match result {
            CegarResult::RealPlan(plan) => {
                assert!(plan.validate_consistency());
                assert!(plan.step_count() <= 3);
            }
            other => panic!("Expected RealPlan, got {other}"),
        }
    }

    #[test]
    fn test_cegar_with_compatibility_constraint() {
        // Only (0,0) and (1,1) are compatible — forces upgrading both together
        // in a specific order.
        let graph = make_graph_2x2();
        let config = CegarConfig::default();
        let constraints = vec![Constraint::Compatibility {
            id: Id::from_name("compat-01"),
            service_a: ServiceIndex(0),
            service_b: ServiceIndex(1),
            compatible_pairs: vec![
                (VersionIndex(0), VersionIndex(0)),
                (VersionIndex(1), VersionIndex(1)),
                // Allow the transition (1,0) or (0,1) — at least one path must exist
                (VersionIndex(1), VersionIndex(0)),
            ],
        }];
        let mut engine = CegarEngine::new(config, graph, constraints);
        let start = s(&[0, 0]);
        let target = s(&[1, 1]);
        let result = engine.cegar_loop(&start, &target);
        match result {
            CegarResult::RealPlan(plan) => {
                assert!(plan.validate_consistency());
                // Should go (0,0) → (1,0) → (1,1)
                let states = plan.intermediate_states();
                for state in &states {
                    // all intermediate states are (0,0), (1,0) or (1,1)
                    let va = state.get(ServiceIndex(0));
                    let vb = state.get(ServiceIndex(1));
                    assert!(
                        [(0, 0), (1, 0), (1, 1)].contains(&(va.0, vb.0)),
                        "Unexpected intermediate state: {state}"
                    );
                }
            }
            other => panic!("Expected RealPlan, got {other}"),
        }
    }

    #[test]
    fn test_cegar_with_forbidden_state() {
        // Forbid service 0 at version 1 — makes (0,0) → (1,*) impossible.
        let graph = make_graph_2x2();
        let config = CegarConfig::default();
        let constraints = vec![Constraint::Forbidden {
            id: Id::from_name("forbid-svc0-v1"),
            service: ServiceIndex(0),
            version: VersionIndex(1),
        }];
        let mut engine = CegarEngine::new(config, graph, constraints);
        let start = s(&[0, 0]);
        let target = s(&[1, 1]);
        // Target itself is forbidden, so no plan is possible.
        let result = engine.cegar_loop(&start, &target);
        match result {
            CegarResult::Infeasible(_) => { /* expected */ }
            other => panic!("Expected Infeasible, got {other}"),
        }
    }

    #[test]
    fn test_cegar_resource_constraint_causes_refinement() {
        // Build a 3×3 graph. Add a resource constraint that forbids the state
        // (1,1) even though it's abstractly feasible.
        let graph = make_graph_3x3();
        let config = CegarConfig { max_iterations: 20, ..CegarConfig::default() };

        let mut costs = HashMap::new();
        costs.insert((ServiceIndex(0), VersionIndex(1)), 6.0);
        costs.insert((ServiceIndex(1), VersionIndex(1)), 6.0);
        // (1,1) costs 12 > 10 → violates
        // (2,1) costs 6 (only svc1), (1,2) costs 6 (only svc0), (2,2) costs 0
        let constraints = vec![Constraint::Resource {
            id: Id::from_name("cpu-budget"),
            resource_name: "cpu".into(),
            max_budget: 10.0,
            per_service_cost: costs,
        }];
        let mut engine = CegarEngine::new(config, graph, constraints);
        let start = s(&[0, 0]);
        let target = s(&[2, 2]);
        let result = engine.cegar_loop(&start, &target);
        match result {
            CegarResult::RealPlan(plan) => {
                assert!(plan.validate_consistency());
                // The plan must avoid (1,1). Verify:
                let states = plan.intermediate_states();
                for st in &states {
                    assert!(
                        !(st.get(ServiceIndex(0)) == VersionIndex(1)
                            && st.get(ServiceIndex(1)) == VersionIndex(1)),
                        "Plan should not pass through (1,1) but did"
                    );
                }
                // At least one refinement should have occurred.
                assert!(engine.stats().refinements > 0 || engine.stats().iterations == 1);
            }
            other => panic!("Expected RealPlan, got {other}"),
        }
    }

    #[test]
    fn test_cegar_max_iterations() {
        // Create a situation where the abstract solver always finds a plan but
        // the concrete checker always rejects it, and we hit max iterations.
        let graph = make_graph_2x2();
        let config = CegarConfig {
            max_iterations: 3,
            timeout: Duration::from_secs(10),
            max_depth: 10,
            ..CegarConfig::default()
        };
        // Resource constraint that forbids *every* non-start state.
        let mut costs = HashMap::new();
        costs.insert((ServiceIndex(0), VersionIndex(1)), 100.0);
        costs.insert((ServiceIndex(1), VersionIndex(1)), 100.0);
        let constraints = vec![Constraint::Resource {
            id: Id::from_name("impossible"),
            resource_name: "mem".into(),
            max_budget: 1.0,
            per_service_cost: costs,
        }];
        let mut engine = CegarEngine::new(config, graph, constraints);
        let start = s(&[0, 0]);
        let target = s(&[1, 1]);
        let result = engine.cegar_loop(&start, &target);
        match result {
            CegarResult::MaxIterations(n) | CegarResult::Infeasible(_) => {
                // Either is acceptable — we either exhaust iterations or
                // block all paths and prove infeasible.
                if let CegarResult::MaxIterations(n) = result {
                    assert_eq!(n, 3);
                }
            }
            other => panic!("Expected MaxIterations or Infeasible, got {other}"),
        }
    }

    #[test]
    fn test_abstraction_extract_constraints() {
        let constraints = vec![
            Constraint::Compatibility {
                id: Id::from_name("c1"),
                service_a: ServiceIndex(0),
                service_b: ServiceIndex(1),
                compatible_pairs: vec![(VersionIndex(0), VersionIndex(0))],
            },
            Constraint::Resource {
                id: Id::from_name("r1"),
                resource_name: "cpu".into(),
                max_budget: 10.0,
                per_service_cost: HashMap::new(),
            },
            Constraint::Forbidden {
                id: Id::from_name("f1"),
                service: ServiceIndex(0),
                version: VersionIndex(2),
            },
        ];
        let abs = Abstraction::abstract_constraints(&constraints);
        // Resource should be excluded → 2 abstract constraints.
        assert_eq!(abs.len(), 2);
        assert!(matches!(abs[0], AbstractConstraint::CompatPair { .. }));
        assert!(matches!(abs[1], AbstractConstraint::ForbiddenState { .. }));
    }

    #[test]
    fn test_blocked_plan_matching() {
        let blocked = BlockedPlan::new(vec![s(&[0, 0]), s(&[1, 0])]);
        let path_a = vec![s(&[0, 0]), s(&[1, 0]), s(&[1, 1])];
        assert!(blocked.is_blocked(&path_a));

        let path_b = vec![s(&[0, 1]), s(&[1, 1])];
        assert!(!blocked.is_blocked(&path_b));

        // Sub-sequence must be contiguous.
        let path_c = vec![s(&[0, 0]), s(&[0, 1]), s(&[1, 0])];
        assert!(!blocked.is_blocked(&path_c));

        assert!(!blocked.is_blocked(&[]));
        assert!(!blocked.is_blocked(&[s(&[0, 0])]));
    }

    #[test]
    fn test_concrete_checker_all_pass() {
        let constraints = vec![Constraint::Compatibility {
            id: Id::from_name("c"),
            service_a: ServiceIndex(0),
            service_b: ServiceIndex(1),
            compatible_pairs: vec![
                (VersionIndex(0), VersionIndex(0)),
                (VersionIndex(1), VersionIndex(1)),
            ],
        }];
        let checker = ConcreteChecker::new(&constraints);
        let states = vec![s(&[0, 0]), s(&[1, 1])];
        assert!(checker.check(&states).is_feasible());
    }

    #[test]
    fn test_concrete_checker_violation() {
        let mut costs = HashMap::new();
        costs.insert((ServiceIndex(0), VersionIndex(1)), 8.0);
        costs.insert((ServiceIndex(1), VersionIndex(1)), 8.0);
        let constraints = vec![Constraint::Resource {
            id: Id::from_name("res"),
            resource_name: "cpu".into(),
            max_budget: 10.0,
            per_service_cost: costs,
        }];
        let checker = ConcreteChecker::new(&constraints);

        let good = vec![s(&[0, 0])];
        assert!(checker.check(&good).is_feasible());

        let bad = vec![s(&[0, 0]), s(&[1, 1])];
        let result = checker.check(&bad);
        assert!(!result.is_feasible());
        if let ConcreteCheckResult::Infeasible { step_index, .. } = result {
            assert_eq!(step_index, 1);
        }
    }

    #[test]
    fn test_refinement_blocking_clause() {
        let path = vec![s(&[0, 0]), s(&[1, 0]), s(&[1, 1]), s(&[2, 1])];
        let cid = Id::from_name("constraint-x");
        let refinement = Refinement::from_spurious(&path, 2, &cid);
        assert_eq!(refinement.failure_step(), 2);
        let blocked = refinement.blocking_clause();
        // Window around step 2 ⇒ states [1..4) = [s(1,0), s(1,1), s(2,1)]
        assert_eq!(blocked.len(), 3);
        assert!(blocked.is_blocked(&path));
    }

    #[test]
    fn test_cegar_stats_summary() {
        let mut stats = CegarStats::default();
        stats.iterations = 5;
        stats.refinements = 3;
        stats.total_time = Duration::from_millis(1200);
        let summary = stats.summary();
        assert!(summary.contains("5 iterations"));
        assert!(summary.contains("3 refinements"));
    }

    #[test]
    fn test_abstraction_feasibility_check() {
        let graph = make_graph_2x2();
        let constraints = vec![Constraint::Forbidden {
            id: Id::from_name("f"),
            service: ServiceIndex(0),
            version: VersionIndex(1),
        }];
        let abs = Abstraction::new(&graph, &constraints);
        assert!(abs.is_abstract_feasible(&s(&[0, 0])));
        assert!(!abs.is_abstract_feasible(&s(&[1, 0])));
        assert!(!abs.is_abstract_feasible(&s(&[1, 1])));
        assert!(abs.is_abstract_feasible(&s(&[0, 1])));
    }

    #[test]
    fn test_cegar_ordering_constraint() {
        // Service 0 must upgrade before service 1.
        let graph = make_graph_2x2();
        let config = CegarConfig::default();
        let constraints = vec![Constraint::Ordering {
            id: Id::from_name("order-01"),
            before: ServiceIndex(0),
            after: ServiceIndex(1),
        }];
        let mut engine = CegarEngine::new(config, graph, constraints);
        let start = s(&[0, 0]);
        let target = s(&[1, 1]);
        let result = engine.cegar_loop(&start, &target);
        match result {
            CegarResult::RealPlan(plan) => {
                assert!(plan.validate_consistency());
                // The first step should upgrade service 0.
                assert_eq!(plan.steps[0].service, ServiceIndex(0));
            }
            other => panic!("Expected RealPlan, got {other}"),
        }
    }

    #[test]
    fn test_build_full_graph_sizes() {
        let g1 = build_full_graph(&[2]);
        assert_eq!(g1.state_count(), 2);
        assert_eq!(g1.edge_count(), 2); // 0→1 and 1→0

        let g2 = build_full_graph(&[2, 3]);
        assert_eq!(g2.state_count(), 6);
        // svc0: 2 transitions × 3 versions of svc1 = 6
        // svc1: per state of svc0 (2), (0→1, 1→0, 1→2, 2→1) = 4 × 2 = 8
        assert_eq!(g2.edge_count(), 14);
    }

    #[test]
    fn test_engine_reset() {
        let graph = make_graph_2x2();
        let config = CegarConfig::default();
        let mut engine = CegarEngine::new(config, graph, vec![]);
        let start = s(&[0, 0]);
        let target = s(&[1, 1]);
        let _ = engine.cegar_loop(&start, &target);
        assert!(engine.stats().iterations > 0);
        engine.reset();
        assert_eq!(engine.stats().iterations, 0);
        assert!(engine.blocked_plans().is_empty());
    }

    #[test]
    fn test_cegar_larger_graph() {
        // 3 services, each with 3 versions → 27 states.
        let graph = build_full_graph(&[3, 3, 3]);
        let config = CegarConfig {
            max_iterations: 100,
            max_depth: 20,
            ..CegarConfig::default()
        };
        let constraints: Vec<Constraint> = vec![];
        let mut engine = CegarEngine::new(config, graph, constraints);
        let start = s(&[0, 0, 0]);
        let target = s(&[2, 2, 2]);
        let result = engine.cegar_loop(&start, &target);
        match result {
            CegarResult::RealPlan(plan) => {
                assert!(plan.validate_consistency());
                // Minimum 6 steps (each service must move +2, one step at a time).
                assert!(plan.step_count() >= 6);
            }
            other => panic!("Expected RealPlan, got {other}"),
        }
    }
}
