//! SafeStep Core — the main planning engine for synthesizing verified deployment
//! plans with rollback safety envelopes.
//!
//! This crate implements:
//! - Version-product graph construction from service configurations
//! - Bounded Model Checking (BMC) plan finding
//! - CEGAR abstraction-refinement for mixed SAT/SMT solving
//! - Rollback safety envelope computation via bidirectional reachability
//! - Points of No Return (PNR) identification and stuck-configuration witnesses
//! - k-robustness checking against oracle uncertainty

pub mod graph_builder;
pub mod planner;
// pub mod bmc_engine; // module file does not exist
pub mod cegar;
pub mod envelope;
pub mod robustness;
pub mod oracle;
pub mod optimization;
pub mod kinduction;
pub mod parallel;
pub mod witness_validator;

use std::collections::HashMap;
use std::fmt;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use safestep_types::identifiers::{
    ServiceId, ConstraintId, PlanId, StepId, StateId, Id, IdSet, IdMap,
    ServiceTag, ConstraintTag, PlanTag, StepTag, StateTag,
};
use safestep_types::error::SafeStepError;

pub use graph_builder::{GraphBuilder, StateSpace, GraphStatistics, DownwardClosureChecker};
pub use planner::{DeploymentPlanner, PlanResult, GreedyPlanner, OptimalPlanner};
// pub use bmc_engine::{BmcEngine, BmcResult, UnrollingManager, PlanExtractor};
pub use cegar::{CegarEngine, CegarResult, Abstraction, ConcreteChecker, Refinement, CegarStats};
pub use envelope::{EnvelopeComputer, ReachabilityChecker, EnvelopeAnnotator, WitnessGenerator, EnvelopeStats};
pub use robustness::{RobustnessChecker, RobustnessResult, UncertaintyModel, AdversaryBudget};
pub use oracle::{CompatibilityOracle, CompatResult, OracleCache, CompositeOracle, OracleValidator};
pub use optimization::{PlanOptimizer, OptimizationObjective, ParetoFrontier, StepMerger, CostModel};
pub use kinduction::{KInduction, InductionResult, InvariantChecker};
pub use parallel::{ParallelPlanStep, ParallelPlan, DependencyAnalyzer, ParallelScheduler};
pub use witness_validator::{
    WitnessValidator, WitnessVerdict, MonotoneChecker, MonotoneCheckResult,
    CegarBoundTracker, CegarBoundSummary, CegarRunRecord,
    DownwardClosureValidator, ClosureValidationResult, ClosureViolation,
};

/// Type alias for core results.
pub type CoreResult<T> = std::result::Result<T, SafeStepError>;

// ---------------------------------------------------------------------------
// Common domain types shared across modules
// ---------------------------------------------------------------------------

/// Index of a service within the version-product graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ServiceIndex(pub u16);

impl fmt::Display for ServiceIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "svc:{}", self.0)
    }
}

/// Index of a version within a single service's version list.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct VersionIndex(pub u16);

impl fmt::Display for VersionIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "v{}", self.0)
    }
}

/// A state in the version-product graph — one version per service.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct State {
    pub versions: Vec<VersionIndex>,
}

impl State {
    pub fn new(versions: Vec<VersionIndex>) -> Self {
        Self { versions }
    }

    pub fn dimension(&self) -> usize {
        self.versions.len()
    }

    pub fn get(&self, svc: ServiceIndex) -> VersionIndex {
        self.versions[svc.0 as usize]
    }

    pub fn set(&mut self, svc: ServiceIndex, ver: VersionIndex) {
        self.versions[svc.0 as usize] = ver;
    }

    /// Hamming distance: number of services that differ.
    pub fn distance(&self, other: &State) -> usize {
        self.versions
            .iter()
            .zip(other.versions.iter())
            .filter(|(a, b)| a != b)
            .count()
    }

    /// Services that differ between two states.
    pub fn diff_services(&self, other: &State) -> Vec<ServiceIndex> {
        self.versions
            .iter()
            .zip(other.versions.iter())
            .enumerate()
            .filter(|(_, (a, b))| a != b)
            .map(|(i, _)| ServiceIndex(i as u16))
            .collect()
    }
}

impl fmt::Display for State {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(")?;
        for (i, v) in self.versions.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", v)?;
        }
        write!(f, ")")
    }
}

/// An edge in the version-product graph: a single-service upgrade/downgrade.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Edge {
    pub from: State,
    pub to: State,
    pub service: ServiceIndex,
    pub from_version: VersionIndex,
    pub to_version: VersionIndex,
    pub metadata: TransitionMetadata,
}

/// Metadata attached to a graph edge (transition).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TransitionMetadata {
    pub is_upgrade: bool,
    pub risk_score: u32,
    pub estimated_duration_secs: u64,
    pub requires_downtime: bool,
}

impl Default for TransitionMetadata {
    fn default() -> Self {
        Self {
            is_upgrade: true,
            risk_score: 0,
            estimated_duration_secs: 60,
            requires_downtime: false,
        }
    }
}

/// Descriptor for a single service in the deployment fleet.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceDescriptor {
    pub id: ServiceId,
    pub name: String,
    pub versions: Vec<String>,
    pub dependencies: Vec<ServiceId>,
    pub resource_requirements: HashMap<String, f64>,
    pub upgrade_risk: HashMap<(usize, usize), u32>,
    pub downtime_required: Vec<bool>,
}

impl ServiceDescriptor {
    pub fn new(name: impl Into<String>, versions: Vec<String>) -> Self {
        Self {
            id: Id::from_name(&format!("svc-{}", versions.len())),
            name: name.into(),
            versions,
            dependencies: Vec::new(),
            resource_requirements: HashMap::new(),
            upgrade_risk: HashMap::new(),
            downtime_required: Vec::new(),
        }
    }

    pub fn version_count(&self) -> usize {
        self.versions.len()
    }
}

/// The version-product graph: states are tuples of version indices,
/// edges are single-service version transitions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionProductGraph {
    pub services: Vec<ServiceDescriptor>,
    pub states: Vec<State>,
    pub edges: Vec<Edge>,
    pub adjacency: HashMap<State, Vec<usize>>,
    pub reverse_adjacency: HashMap<State, Vec<usize>>,
}

impl VersionProductGraph {
    pub fn new(services: Vec<ServiceDescriptor>) -> Self {
        Self {
            services,
            states: Vec::new(),
            edges: Vec::new(),
            adjacency: HashMap::new(),
            reverse_adjacency: HashMap::new(),
        }
    }

    pub fn service_count(&self) -> usize {
        self.services.len()
    }

    pub fn state_count(&self) -> usize {
        self.states.len()
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    pub fn neighbors(&self, state: &State) -> Vec<(State, usize)> {
        self.adjacency
            .get(state)
            .map(|idxs| {
                idxs.iter()
                    .map(|&i| (self.edges[i].to.clone(), i))
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn predecessors(&self, state: &State) -> Vec<(State, usize)> {
        self.reverse_adjacency
            .get(state)
            .map(|idxs| {
                idxs.iter()
                    .map(|&i| (self.edges[i].from.clone(), i))
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn has_state(&self, state: &State) -> bool {
        self.adjacency.contains_key(state) || self.states.contains(state)
    }

    pub fn add_state(&mut self, state: State) {
        if !self.has_state(&state) {
            self.adjacency.entry(state.clone()).or_default();
            self.reverse_adjacency.entry(state.clone()).or_default();
            self.states.push(state);
        }
    }

    pub fn add_edge(&mut self, edge: Edge) {
        let idx = self.edges.len();
        self.adjacency
            .entry(edge.from.clone())
            .or_default()
            .push(idx);
        self.reverse_adjacency
            .entry(edge.to.clone())
            .or_default()
            .push(idx);
        self.edges.push(edge);
    }
}

/// A deployment plan: sequence of steps that move from start to target.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeploymentPlan {
    pub id: PlanId,
    pub steps: Vec<PlanStep>,
    pub start: State,
    pub target: State,
    pub total_risk: u32,
    pub total_duration_secs: u64,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

impl DeploymentPlan {
    pub fn new(start: State, target: State, steps: Vec<PlanStep>) -> Self {
        let total_risk: u32 = steps.iter().map(|s| s.risk_score).sum();
        let total_duration_secs: u64 = steps.iter().map(|s| s.estimated_duration_secs).sum();
        Self {
            id: Id::generate(),
            steps,
            start,
            target,
            total_risk,
            total_duration_secs,
            created_at: chrono::Utc::now(),
        }
    }

    pub fn step_count(&self) -> usize {
        self.steps.len()
    }

    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Walk through the plan producing intermediate states.
    pub fn intermediate_states(&self) -> Vec<State> {
        let mut states = vec![self.start.clone()];
        let mut current = self.start.clone();
        for step in &self.steps {
            current.set(step.service, step.to_version);
            states.push(current.clone());
        }
        states
    }

    /// Validate that consecutive steps are consistent.
    pub fn validate_consistency(&self) -> bool {
        let mut current = self.start.clone();
        for step in &self.steps {
            if current.get(step.service) != step.from_version {
                return false;
            }
            current.set(step.service, step.to_version);
        }
        current == self.target
    }
}

/// A single step in a deployment plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanStep {
    pub id: StepId,
    pub service: ServiceIndex,
    pub from_version: VersionIndex,
    pub to_version: VersionIndex,
    pub risk_score: u32,
    pub estimated_duration_secs: u64,
    pub requires_downtime: bool,
}

impl PlanStep {
    pub fn new(
        service: ServiceIndex,
        from_version: VersionIndex,
        to_version: VersionIndex,
    ) -> Self {
        Self {
            id: Id::generate(),
            service,
            from_version,
            to_version,
            risk_score: 0,
            estimated_duration_secs: 60,
            requires_downtime: false,
        }
    }

    pub fn with_risk(mut self, risk: u32) -> Self {
        self.risk_score = risk;
        self
    }

    pub fn with_duration(mut self, secs: u64) -> Self {
        self.estimated_duration_secs = secs;
        self
    }

    pub fn is_upgrade(&self) -> bool {
        self.to_version.0 > self.from_version.0
    }
}

/// Constraint on the deployment (compatibility, resource, ordering, etc.).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Constraint {
    /// Two services must be at compatible versions simultaneously.
    Compatibility {
        id: ConstraintId,
        service_a: ServiceIndex,
        service_b: ServiceIndex,
        compatible_pairs: Vec<(VersionIndex, VersionIndex)>,
    },
    /// Resource budget must not be exceeded at any state.
    Resource {
        id: ConstraintId,
        resource_name: String,
        max_budget: f64,
        per_service_cost: HashMap<(ServiceIndex, VersionIndex), f64>,
    },
    /// Service A must be upgraded before service B.
    Ordering {
        id: ConstraintId,
        before: ServiceIndex,
        after: ServiceIndex,
    },
    /// A service version must not be deployed.
    Forbidden {
        id: ConstraintId,
        service: ServiceIndex,
        version: VersionIndex,
    },
    /// Custom Boolean predicate over state (encoded as a clause).
    Custom {
        id: ConstraintId,
        description: String,
        #[serde(skip, default = "default_constraint_predicate")]
        check: fn(&State) -> bool,
    },
}

fn default_constraint_predicate() -> fn(&State) -> bool {
    |_| false
}

impl Constraint {
    pub fn id(&self) -> &ConstraintId {
        match self {
            Constraint::Compatibility { id, .. } => id,
            Constraint::Resource { id, .. } => id,
            Constraint::Ordering { id, .. } => id,
            Constraint::Forbidden { id, .. } => id,
            Constraint::Custom { id, .. } => id,
        }
    }

    /// Check if a single state satisfies this constraint.
    pub fn check_state(&self, state: &State) -> bool {
        match self {
            Constraint::Compatibility {
                service_a,
                service_b,
                compatible_pairs,
                ..
            } => {
                let va = state.get(*service_a);
                let vb = state.get(*service_b);
                compatible_pairs.contains(&(va, vb))
            }
            Constraint::Resource {
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
                total <= *max_budget
            }
            Constraint::Forbidden {
                service, version, ..
            } => state.get(*service) != *version,
            Constraint::Custom { check, .. } => check(state),
            Constraint::Ordering { .. } => true, // ordering is a plan-level constraint
        }
    }

    /// Check if a transition (plan step) satisfies ordering constraints.
    pub fn check_transition(&self, step_idx: usize, plan: &[PlanStep]) -> bool {
        match self {
            Constraint::Ordering { before, after, .. } => {
                let before_idx = plan.iter().position(|s| s.service == *before);
                let after_idx = plan.iter().position(|s| s.service == *after);
                match (before_idx, after_idx) {
                    (Some(b), Some(a)) => b < a,
                    _ => true,
                }
            }
            _ => true,
        }
    }
}

/// Configuration for the planning engine.
#[derive(Clone, Serialize, Deserialize)]
pub struct PlannerConfig {
    pub max_depth: usize,
    pub timeout: Duration,
    pub use_cegar: bool,
    pub use_parallel: bool,
    pub max_cegar_iterations: usize,
    pub completeness_check: bool,
    pub treewidth_threshold: usize,
    pub optimization_objectives: Vec<String>,
}

impl Default for PlannerConfig {
    fn default() -> Self {
        Self {
            max_depth: 100,
            timeout: Duration::from_secs(300),
            use_cegar: true,
            use_parallel: true,
            max_cegar_iterations: 50,
            completeness_check: true,
            treewidth_threshold: 4,
            optimization_objectives: vec!["steps".into()],
        }
    }
}

/// A clause in our internal SAT representation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Clause {
    pub literals: Vec<i32>,
}

impl Clause {
    pub fn new(literals: Vec<i32>) -> Self {
        Self { literals }
    }

    pub fn unit(lit: i32) -> Self {
        Self {
            literals: vec![lit],
        }
    }

    pub fn binary(a: i32, b: i32) -> Self {
        Self {
            literals: vec![a, b],
        }
    }

    pub fn len(&self) -> usize {
        self.literals.len()
    }

    pub fn is_empty(&self) -> bool {
        self.literals.is_empty()
    }

    pub fn is_unit(&self) -> bool {
        self.literals.len() == 1
    }

    /// Evaluate under a partial assignment (variable -> bool).
    pub fn evaluate(&self, assignment: &HashMap<u32, bool>) -> Option<bool> {
        let mut all_false = true;
        for &lit in &self.literals {
            let var = lit.unsigned_abs();
            let polarity = lit > 0;
            match assignment.get(&var) {
                Some(&val) => {
                    if val == polarity {
                        return Some(true);
                    }
                }
                None => {
                    all_false = false;
                }
            }
        }
        if all_false {
            Some(false)
        } else {
            None
        }
    }
}

/// The safety envelope: the set of states from which the deployment can still
/// safely reach the target *and* safely retreat to the start.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyEnvelope {
    pub safe_states: Vec<State>,
    pub pnr_states: Vec<State>,
    pub plan_annotations: Vec<EnvelopeAnnotation>,
}

impl SafetyEnvelope {
    pub fn new() -> Self {
        Self {
            safe_states: Vec::new(),
            pnr_states: Vec::new(),
            plan_annotations: Vec::new(),
        }
    }

    pub fn is_safe(&self, state: &State) -> bool {
        self.safe_states.contains(state)
    }

    pub fn is_pnr(&self, state: &State) -> bool {
        self.pnr_states.contains(state)
    }

    pub fn safe_count(&self) -> usize {
        self.safe_states.len()
    }

    pub fn pnr_count(&self) -> usize {
        self.pnr_states.len()
    }
}

impl Default for SafetyEnvelope {
    fn default() -> Self {
        Self::new()
    }
}

/// Annotation for a single state on the plan w.r.t. the envelope.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EnvelopeAnnotation {
    Inside { risk_score: u32 },
    Outside { risk_score: u32 },
    PointOfNoReturn { blocking_constraints: Vec<ConstraintId> },
}

/// A witness that a state is stuck (cannot retreat).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StuckWitness {
    pub stuck_state: State,
    pub blocking_constraints: Vec<ConstraintId>,
    pub attempted_retreats: Vec<State>,
    pub explanation: String,
}

/// Invariant predicate over states (used in k-induction).
pub struct Invariant {
    pub description: String,
    pub check: Box<dyn Fn(&State) -> bool + Send + Sync>,
}

impl fmt::Debug for Invariant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Invariant")
            .field("description", &self.description)
            .field("check", &"<fn>")
            .finish()
    }
}

impl Clone for Invariant {
    fn clone(&self) -> Self {
        Self {
            description: self.description.clone(),
            check: Box::new(|_| false),
        }
    }
}

impl Invariant {
    pub fn new(description: impl Into<String>, check: impl Fn(&State) -> bool + Send + Sync + 'static) -> Self {
        Self {
            description: description.into(),
            check: Box::new(check),
        }
    }

    pub fn holds(&self, state: &State) -> bool {
        (self.check)(state)
    }
}

impl fmt::Debug for PlannerConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PlannerConfig")
            .field("max_depth", &self.max_depth)
            .field("timeout", &self.timeout)
            .field("use_cegar", &self.use_cegar)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_basics() {
        let s = State::new(vec![VersionIndex(0), VersionIndex(1), VersionIndex(2)]);
        assert_eq!(s.dimension(), 3);
        assert_eq!(s.get(ServiceIndex(0)), VersionIndex(0));
        assert_eq!(s.get(ServiceIndex(1)), VersionIndex(1));
    }

    #[test]
    fn test_state_distance() {
        let a = State::new(vec![VersionIndex(0), VersionIndex(1)]);
        let b = State::new(vec![VersionIndex(1), VersionIndex(1)]);
        assert_eq!(a.distance(&b), 1);

        let c = State::new(vec![VersionIndex(1), VersionIndex(2)]);
        assert_eq!(a.distance(&c), 2);
    }

    #[test]
    fn test_state_diff_services() {
        let a = State::new(vec![VersionIndex(0), VersionIndex(1), VersionIndex(0)]);
        let b = State::new(vec![VersionIndex(0), VersionIndex(2), VersionIndex(1)]);
        let diff = a.diff_services(&b);
        assert_eq!(diff, vec![ServiceIndex(1), ServiceIndex(2)]);
    }

    #[test]
    fn test_clause_evaluate() {
        let clause = Clause::new(vec![1, -2, 3]);
        let mut asgn = HashMap::new();
        asgn.insert(1, false);
        asgn.insert(2, true);
        asgn.insert(3, false);
        assert_eq!(clause.evaluate(&asgn), Some(false));

        asgn.insert(1, true);
        assert_eq!(clause.evaluate(&asgn), Some(true));
    }

    #[test]
    fn test_plan_step() {
        let step = PlanStep::new(ServiceIndex(0), VersionIndex(0), VersionIndex(1));
        assert!(step.is_upgrade());
        let step2 = PlanStep::new(ServiceIndex(0), VersionIndex(2), VersionIndex(1));
        assert!(!step2.is_upgrade());
    }

    #[test]
    fn test_deployment_plan_consistency() {
        let start = State::new(vec![VersionIndex(0), VersionIndex(0)]);
        let target = State::new(vec![VersionIndex(1), VersionIndex(1)]);
        let steps = vec![
            PlanStep::new(ServiceIndex(0), VersionIndex(0), VersionIndex(1)),
            PlanStep::new(ServiceIndex(1), VersionIndex(0), VersionIndex(1)),
        ];
        let plan = DeploymentPlan::new(start, target, steps);
        assert!(plan.validate_consistency());
        assert_eq!(plan.step_count(), 2);
    }

    #[test]
    fn test_deployment_plan_intermediate_states() {
        let start = State::new(vec![VersionIndex(0), VersionIndex(0)]);
        let target = State::new(vec![VersionIndex(1), VersionIndex(1)]);
        let steps = vec![
            PlanStep::new(ServiceIndex(0), VersionIndex(0), VersionIndex(1)),
            PlanStep::new(ServiceIndex(1), VersionIndex(0), VersionIndex(1)),
        ];
        let plan = DeploymentPlan::new(start.clone(), target.clone(), steps);
        let intermediates = plan.intermediate_states();
        assert_eq!(intermediates.len(), 3);
        assert_eq!(intermediates[0], start);
        assert_eq!(intermediates[2], target);
    }

    #[test]
    fn test_constraint_compatibility() {
        let c = Constraint::Compatibility {
            id: Id::from_name("test-compat"),
            service_a: ServiceIndex(0),
            service_b: ServiceIndex(1),
            compatible_pairs: vec![
                (VersionIndex(0), VersionIndex(0)),
                (VersionIndex(1), VersionIndex(1)),
            ],
        };
        let good = State::new(vec![VersionIndex(0), VersionIndex(0)]);
        assert!(c.check_state(&good));
        let bad = State::new(vec![VersionIndex(0), VersionIndex(1)]);
        assert!(!c.check_state(&bad));
    }

    #[test]
    fn test_constraint_resource() {
        let mut costs = HashMap::new();
        costs.insert((ServiceIndex(0), VersionIndex(1)), 5.0);
        costs.insert((ServiceIndex(1), VersionIndex(1)), 6.0);
        let c = Constraint::Resource {
            id: Id::from_name("res"),
            resource_name: "cpu".into(),
            max_budget: 10.0,
            per_service_cost: costs,
        };
        let s1 = State::new(vec![VersionIndex(0), VersionIndex(0)]);
        assert!(c.check_state(&s1));
        let s2 = State::new(vec![VersionIndex(1), VersionIndex(1)]);
        assert!(!c.check_state(&s2)); // 5+6 > 10
    }

    #[test]
    fn test_graph_basics() {
        let svc = ServiceDescriptor::new("api", vec!["v1".into(), "v2".into()]);
        let mut graph = VersionProductGraph::new(vec![svc]);
        let s0 = State::new(vec![VersionIndex(0)]);
        let s1 = State::new(vec![VersionIndex(1)]);
        graph.add_state(s0.clone());
        graph.add_state(s1.clone());
        let edge = Edge {
            from: s0.clone(),
            to: s1.clone(),
            service: ServiceIndex(0),
            from_version: VersionIndex(0),
            to_version: VersionIndex(1),
            metadata: TransitionMetadata::default(),
        };
        graph.add_edge(edge);
        assert_eq!(graph.state_count(), 2);
        assert_eq!(graph.edge_count(), 1);
        assert_eq!(graph.neighbors(&s0).len(), 1);
        assert_eq!(graph.predecessors(&s1).len(), 1);
    }

    #[test]
    fn test_safety_envelope() {
        let mut env = SafetyEnvelope::new();
        let s1 = State::new(vec![VersionIndex(0)]);
        let s2 = State::new(vec![VersionIndex(1)]);
        env.safe_states.push(s1.clone());
        env.pnr_states.push(s2.clone());
        assert!(env.is_safe(&s1));
        assert!(!env.is_safe(&s2));
        assert!(env.is_pnr(&s2));
        assert_eq!(env.safe_count(), 1);
        assert_eq!(env.pnr_count(), 1);
    }
}
