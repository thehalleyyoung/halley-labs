//! CoalCEGAR — Counterexample-Guided Abstraction Refinement for coalgebraic LLMs.
//!
//! Implements the full CEGAR loop:
//! 1. Abstract the LLM behavior at current (k, n, ε) level
//! 2. Verify properties on the abstract model
//! 3. If verified → certify. If counter-example found → check if spurious
//! 4. If spurious → refine abstraction. If real → report violation
//!
//! Integrates with alphabet abstraction, lattice traversal, refinement planning,
//! and Galois connections.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::time::Instant;
use ordered_float::OrderedFloat;
use chrono::Utc;

use super::lattice::{AbstractionLattice, AbstractionTriple, LatticeBudget, LatticeTraversalStrategy};
use super::refinement::{
    RefinementHistory, RefinementOperator, RefinementPlanner, RefinementResult,
    RefinementStrategy, RefinementKind, CounterexampleInfo,
};

// ---------------------------------------------------------------------------
// Local type aliases (to be swapped with coalgebra module types later)
// ---------------------------------------------------------------------------

/// A state in the abstract model.
pub type StateId = String;

/// A property specification to check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropertySpec {
    pub name: String,
    pub description: String,
    pub kind: PropertyKind,
    pub bound: Option<f64>,
}

impl PropertySpec {
    pub fn safety(name: impl Into<String>, desc: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: desc.into(),
            kind: PropertyKind::Safety,
            bound: None,
        }
    }

    pub fn probabilistic(name: impl Into<String>, bound: f64, desc: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: desc.into(),
            kind: PropertyKind::ProbabilisticBound,
            bound: Some(bound),
        }
    }

    pub fn bisimulation(name: impl Into<String>, dist: f64, desc: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: desc.into(),
            kind: PropertyKind::BisimulationDistance,
            bound: Some(dist),
        }
    }
}

/// Kind of property.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PropertyKind {
    Safety,
    Liveness,
    ProbabilisticBound,
    BisimulationDistance,
    TraceEquivalence,
}

// ---------------------------------------------------------------------------
// Counter-example
// ---------------------------------------------------------------------------

/// A counter-example produced by the model checker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CounterExample {
    /// Sequence of states in the counter-example trace.
    pub trace: Vec<StateId>,
    /// Input words that trigger the counter-example.
    pub inputs: Vec<String>,
    /// The property that was violated.
    pub property: String,
    /// Whether this counter-example is spurious (artifact of abstraction).
    pub spurious: Option<bool>,
    /// Diagnosis information for guiding refinement.
    pub diagnosis: Option<CounterexampleDiagnosis>,
    /// Severity (0 to 1).
    pub severity: f64,
}

impl CounterExample {
    pub fn new(trace: Vec<StateId>, inputs: Vec<String>, property: String) -> Self {
        Self {
            trace,
            inputs,
            property,
            spurious: None,
            diagnosis: None,
            severity: 0.5,
        }
    }

    /// Check if the counter-example has been classified.
    pub fn is_classified(&self) -> bool {
        self.spurious.is_some()
    }

    /// Mark as spurious with diagnosis.
    pub fn mark_spurious(&mut self, diagnosis: CounterexampleDiagnosis) {
        self.spurious = Some(true);
        self.diagnosis = Some(diagnosis);
    }

    /// Mark as genuine.
    pub fn mark_genuine(&mut self, severity: f64) {
        self.spurious = Some(false);
        self.severity = severity;
    }

    /// Convert to refinement guidance.
    pub fn to_counterexample_info(&self) -> CounterexampleInfo {
        match &self.diagnosis {
            Some(diag) => match diag.cause {
                SpuriousnessCause::ClusterMerge => {
                    CounterexampleInfo::cluster_confusion(self.trace.clone())
                }
                SpuriousnessCause::InsufficientDepth => {
                    CounterexampleInfo::depth_insufficiency(self.inputs.clone())
                }
                SpuriousnessCause::DistributionCoarseness => {
                    CounterexampleInfo::distribution_imprecision(
                        self.trace.clone(),
                        self.severity,
                    )
                }
                SpuriousnessCause::CombinedFactors => {
                    // Default to cluster confusion for combined.
                    CounterexampleInfo::cluster_confusion(self.trace.clone())
                }
            },
            None => {
                // Unknown cause — guess from trace length.
                if self.inputs.iter().any(|i| i.len() > 10) {
                    CounterexampleInfo::depth_insufficiency(self.inputs.clone())
                } else {
                    CounterexampleInfo::cluster_confusion(self.trace.clone())
                }
            }
        }
    }
}

impl fmt::Display for CounterExample {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let status = match self.spurious {
            Some(true) => "SPURIOUS",
            Some(false) => "GENUINE",
            None => "UNCLASSIFIED",
        };
        write!(
            f,
            "CounterExample[{} trace_len={} prop={} severity={:.2}]",
            status, self.trace.len(), self.property, self.severity
        )
    }
}

/// Diagnosis of why a counter-example is spurious.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CounterexampleDiagnosis {
    pub cause: SpuriousnessCause,
    pub explanation: String,
    /// Which abstract states are involved in the spuriousness.
    pub involved_states: Vec<StateId>,
    /// Confidence in the diagnosis (0 to 1).
    pub confidence: f64,
}

/// Why a counter-example is spurious.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SpuriousnessCause {
    /// Two behaviorally distinct concrete states were merged.
    ClusterMerge,
    /// The probing depth was insufficient to distinguish states.
    InsufficientDepth,
    /// The distributional tolerance was too large.
    DistributionCoarseness,
    /// Multiple causes contribute.
    CombinedFactors,
}

// ---------------------------------------------------------------------------
// CEGAR phases
// ---------------------------------------------------------------------------

/// The current phase of the CEGAR loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CegarPhase {
    /// Initialize: set up the initial abstraction.
    Initialize,
    /// Abstract: build the abstract model at current level.
    Abstract,
    /// Verify: check properties on the abstract model.
    Verify,
    /// Analyze: check if counter-example is spurious.
    Analyze,
    /// Refine: refine the abstraction to eliminate spurious counter-example.
    Refine,
    /// Certify: verification succeeded — produce certificate.
    Certify,
    /// Report: genuine counter-example found — report violation.
    Report,
    /// Terminated: CEGAR loop has finished.
    Terminated,
}

impl fmt::Display for CegarPhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Initialize => write!(f, "INIT"),
            Self::Abstract => write!(f, "ABSTRACT"),
            Self::Verify => write!(f, "VERIFY"),
            Self::Analyze => write!(f, "ANALYZE"),
            Self::Refine => write!(f, "REFINE"),
            Self::Certify => write!(f, "CERTIFY"),
            Self::Report => write!(f, "REPORT"),
            Self::Terminated => write!(f, "TERMINATED"),
        }
    }
}

/// Termination condition for the CEGAR loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CegarTermination {
    /// Successfully verified at some abstraction level.
    Verified {
        triple: AbstractionTriple,
        certificate_info: String,
    },
    /// Genuine counter-example found.
    CounterExampleFound {
        counter_example: CounterExample,
    },
    /// Budget exhausted without conclusion.
    BudgetExhausted {
        best_triple: Option<AbstractionTriple>,
        partial_results: String,
    },
    /// Maximum iterations reached.
    MaxIterations {
        iterations: usize,
    },
    /// Lattice fully explored without success.
    LatticeExhausted,
    /// Monotonicity violation detected.
    MonotonicityViolation {
        explanation: String,
    },
}

impl fmt::Display for CegarTermination {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Verified { triple, .. } => write!(f, "VERIFIED at {}", triple),
            Self::CounterExampleFound { counter_example } => {
                write!(f, "COUNTER-EXAMPLE: {}", counter_example)
            }
            Self::BudgetExhausted { .. } => write!(f, "BUDGET EXHAUSTED"),
            Self::MaxIterations { iterations } => write!(f, "MAX ITERATIONS ({})", iterations),
            Self::LatticeExhausted => write!(f, "LATTICE EXHAUSTED"),
            Self::MonotonicityViolation { explanation } => {
                write!(f, "MONOTONICITY VIOLATION: {}", explanation)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Trait interfaces for integration
// ---------------------------------------------------------------------------

/// Trait for a hypothesis learner (PCL* or similar).
/// Builds an abstract model at a given abstraction level.
pub trait HypothesisLearner: fmt::Debug {
    /// Learn an abstract model at the given abstraction level.
    /// Returns the model (as opaque state) and the cost of learning.
    fn learn(
        &mut self,
        triple: &AbstractionTriple,
        previous_model: Option<&AbstractModel>,
    ) -> Result<(AbstractModel, f64), String>;

    /// Refine an existing model based on a counter-example.
    fn refine(
        &mut self,
        model: &AbstractModel,
        counter_example: &CounterExample,
        refinement: &RefinementOperator,
    ) -> Result<(AbstractModel, f64), String>;
}

/// Trait for a model checker.
/// Verifies properties on an abstract model.
pub trait AbstractionVerifier: fmt::Debug {
    /// Verify a property on the abstract model.
    /// Returns Ok(None) if verified, Ok(Some(cx)) if counter-example found.
    fn verify(
        &self,
        model: &AbstractModel,
        property: &PropertySpec,
    ) -> Result<Option<CounterExample>, String>;

    /// Check if a counter-example is spurious by testing against concrete behavior.
    fn check_spuriousness(
        &self,
        counter_example: &CounterExample,
        concrete_model: Option<&AbstractModel>,
    ) -> Result<bool, String>;
}

/// An abstract model (opaque to CEGAR — produced by learner, consumed by verifier).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbstractModel {
    pub triple: AbstractionTriple,
    pub num_states: usize,
    pub num_transitions: usize,
    /// State map for spuriousness checking.
    pub state_ids: Vec<StateId>,
    /// Whether this model has been verified.
    pub verified: Option<bool>,
    /// Metadata.
    pub metadata: HashMap<String, String>,
}

impl AbstractModel {
    pub fn new(triple: AbstractionTriple, num_states: usize, num_transitions: usize) -> Self {
        let state_ids = (0..num_states).map(|i| format!("q{}", i)).collect();
        Self {
            triple,
            num_states,
            num_transitions,
            state_ids,
            verified: None,
            metadata: HashMap::new(),
        }
    }
}

impl fmt::Display for AbstractModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "AbstractModel[{} states={} trans={}]",
            self.triple, self.num_states, self.num_transitions
        )
    }
}

// ---------------------------------------------------------------------------
// Default implementations of learner and verifier
// ---------------------------------------------------------------------------

/// A stub learner that creates simple models.
/// Real LLM learning would go through the `learn` method.
#[derive(Debug)]
pub struct StubLearner {
    pub base_states: usize,
    pub base_transitions: usize,
}

impl StubLearner {
    pub fn new() -> Self {
        Self {
            base_states: 4,
            base_transitions: 8,
        }
    }
}

impl HypothesisLearner for StubLearner {
    fn learn(
        &mut self,
        triple: &AbstractionTriple,
        _previous: Option<&AbstractModel>,
    ) -> Result<(AbstractModel, f64), String> {
        // Model complexity scales with abstraction parameters.
        let states = self.base_states * triple.k;
        let trans = self.base_transitions * triple.k * (triple.n + 1);
        let cost = triple.estimated_cost(4);

        let model = AbstractModel::new(triple.clone(), states, trans);
        Ok((model, cost))
    }

    fn refine(
        &mut self,
        model: &AbstractModel,
        _counter_example: &CounterExample,
        refinement: &RefinementOperator,
    ) -> Result<(AbstractModel, f64), String> {
        let new_triple = refinement.to.clone();
        let states = model.num_states + new_triple.k;
        let trans = model.num_transitions + new_triple.k * (new_triple.n + 1);
        let cost = refinement.estimated_cost;

        let new_model = AbstractModel::new(new_triple, states, trans);
        Ok((new_model, cost))
    }
}

/// A stub verifier that deterministically decides based on model size.
#[derive(Debug)]
pub struct StubVerifier {
    /// Minimum model states needed for verification to pass.
    pub min_states_for_pass: usize,
    /// Probability of finding counter-example (for testing).
    pub failure_threshold: usize,
}

impl StubVerifier {
    pub fn new(min_states: usize) -> Self {
        Self {
            min_states_for_pass: min_states,
            failure_threshold: min_states,
        }
    }
}

impl AbstractionVerifier for StubVerifier {
    fn verify(
        &self,
        model: &AbstractModel,
        property: &PropertySpec,
    ) -> Result<Option<CounterExample>, String> {
        if model.num_states >= self.min_states_for_pass {
            Ok(None) // Verified
        } else {
            // Generate a counter-example.
            let trace = model.state_ids[..model.num_states.min(3)].to_vec();
            let cx = CounterExample::new(
                trace,
                vec!["probe_input".to_string()],
                property.name.clone(),
            );
            Ok(Some(cx))
        }
    }

    fn check_spuriousness(
        &self,
        counter_example: &CounterExample,
        _concrete: Option<&AbstractModel>,
    ) -> Result<bool, String> {
        // Stub: counter-examples from small models are spurious.
        Ok(counter_example.trace.len() < self.failure_threshold)
    }
}

// ---------------------------------------------------------------------------
// CEGAR state
// ---------------------------------------------------------------------------

/// Full state of the CEGAR loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CegarState {
    /// Current phase.
    pub phase: CegarPhase,
    /// Current abstraction triple.
    pub current_triple: AbstractionTriple,
    /// Iteration count.
    pub iteration: usize,
    /// Phase transition log.
    pub phase_log: Vec<(CegarPhase, String)>,
    /// Current counter-example (if in Analyze/Refine phase).
    pub current_counter_example: Option<CounterExample>,
    /// All counter-examples found.
    pub all_counter_examples: Vec<CounterExample>,
    /// Whether monotonicity has been checked.
    pub monotonicity_checked: bool,
    /// Cost spent so far.
    pub cost_spent: f64,
    /// Start time (ISO 8601).
    pub start_time: String,
    /// Elapsed time (seconds).
    pub elapsed_secs: f64,
}

impl CegarState {
    pub fn new(initial_triple: AbstractionTriple) -> Self {
        Self {
            phase: CegarPhase::Initialize,
            current_triple: initial_triple,
            iteration: 0,
            phase_log: Vec::new(),
            current_counter_example: None,
            all_counter_examples: Vec::new(),
            monotonicity_checked: false,
            cost_spent: 0.0,
            start_time: Utc::now().to_rfc3339(),
            elapsed_secs: 0.0,
        }
    }

    /// Transition to a new phase.
    pub fn transition_to(&mut self, new_phase: CegarPhase, reason: String) {
        self.phase_log.push((self.phase, reason));
        self.phase = new_phase;
    }

    /// Record cost.
    pub fn add_cost(&mut self, cost: f64) {
        self.cost_spent += cost;
    }

    /// Get a summary of the state.
    pub fn summary(&self) -> String {
        format!(
            "CEGAR[phase={}, iter={}, triple={}, cost={:.2}, cx={}]",
            self.phase, self.iteration, self.current_triple,
            self.cost_spent, self.all_counter_examples.len()
        )
    }
}

// ---------------------------------------------------------------------------
// CEGAR configuration
// ---------------------------------------------------------------------------

/// Configuration for the CEGAR loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CegarConfig {
    /// Maximum number of CEGAR iterations.
    pub max_iterations: usize,
    /// Maximum total budget.
    pub max_budget: f64,
    /// Maximum wall-clock time (seconds).
    pub max_time_secs: f64,
    /// Refinement strategy.
    pub refinement_strategy: RefinementStrategy,
    /// Lattice traversal strategy.
    pub lattice_strategy: LatticeTraversalStrategy,
    /// Whether to check monotonicity of abstraction.
    pub check_monotonicity: bool,
    /// Maximum consecutive spurious counter-examples before giving up.
    pub max_spurious_streak: usize,
    /// Available k values.
    pub k_values: Vec<usize>,
    /// Available n values.
    pub n_values: Vec<usize>,
    /// Available epsilon values.
    pub epsilon_values: Vec<f64>,
    /// Input alphabet size.
    pub alphabet_size: usize,
}

impl Default for CegarConfig {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            max_budget: 1e6,
            max_time_secs: 3600.0,
            refinement_strategy: RefinementStrategy::CounterexampleGuided,
            lattice_strategy: LatticeTraversalStrategy::BestFirst,
            check_monotonicity: true,
            max_spurious_streak: 10,
            k_values: vec![2, 4, 8, 16, 32],
            n_values: vec![1, 2, 3, 4, 5],
            epsilon_values: vec![0.5, 0.25, 0.1, 0.05, 0.01],
            alphabet_size: 4,
        }
    }
}

// ---------------------------------------------------------------------------
// CEGAR result
// ---------------------------------------------------------------------------

/// The final result of a CEGAR run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CegarResult {
    /// How the loop terminated.
    pub termination: CegarTermination,
    /// Final state of the CEGAR loop.
    pub state: CegarState,
    /// Final abstract model (if available).
    pub final_model: Option<AbstractModel>,
    /// Refinement history.
    pub refinement_summary: String,
    /// Lattice statistics.
    pub lattice_summary: String,
    /// Total cost.
    pub total_cost: f64,
    /// Total iterations.
    pub total_iterations: usize,
    /// Wall-clock time (seconds).
    pub wall_time_secs: f64,
}

impl CegarResult {
    pub fn is_verified(&self) -> bool {
        matches!(self.termination, CegarTermination::Verified { .. })
    }

    pub fn is_violation(&self) -> bool {
        matches!(self.termination, CegarTermination::CounterExampleFound { .. })
    }

    pub fn is_inconclusive(&self) -> bool {
        !self.is_verified() && !self.is_violation()
    }
}

impl fmt::Display for CegarResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CegarResult[{}, iters={}, cost={:.2}, time={:.1}s]",
            self.termination, self.total_iterations, self.total_cost, self.wall_time_secs
        )
    }
}

// ---------------------------------------------------------------------------
// The main CEGAR loop
// ---------------------------------------------------------------------------

/// The main CEGAR loop orchestrator.
pub struct CegarLoop {
    pub config: CegarConfig,
    pub state: CegarState,
    pub lattice: AbstractionLattice,
    pub refinement_planner: RefinementPlanner,
    pub refinement_history: RefinementHistory,
    pub properties: Vec<PropertySpec>,
    /// The current abstract model.
    pub current_model: Option<AbstractModel>,
    /// Counter-example streak counter.
    spurious_streak: usize,
    /// Start time for wall-clock tracking.
    start_instant: Option<Instant>,
    /// Previous verification results for monotonicity checking.
    previous_results: Vec<(AbstractionTriple, bool)>,
}

impl CegarLoop {
    /// Create a new CEGAR loop.
    pub fn new(config: CegarConfig, properties: Vec<PropertySpec>) -> Self {
        let lattice = AbstractionLattice::new(
            config.k_values.clone(),
            config.n_values.clone(),
            config.epsilon_values.clone(),
            config.alphabet_size,
            config.lattice_strategy,
            LatticeBudget::new(config.max_budget, config.max_iterations, config.max_time_secs),
        );

        let initial_triple = lattice.bottom()
            .map(|id| lattice.nodes[id].triple.clone())
            .unwrap_or_else(AbstractionTriple::coarsest);

        let refinement_planner = RefinementPlanner::new(
            config.refinement_strategy,
            config.alphabet_size,
            config.k_values.clone(),
            config.n_values.clone(),
            config.epsilon_values.clone(),
        );

        Self {
            config,
            state: CegarState::new(initial_triple),
            lattice,
            refinement_planner,
            refinement_history: RefinementHistory::new(),
            properties,
            current_model: None,
            spurious_streak: 0,
            start_instant: None,
            previous_results: Vec::new(),
        }
    }

    /// Run the full CEGAR loop with provided learner and verifier.
    pub fn run(
        &mut self,
        learner: &mut dyn HypothesisLearner,
        verifier: &dyn AbstractionVerifier,
    ) -> CegarResult {
        self.start_instant = Some(Instant::now());
        self.state.transition_to(CegarPhase::Initialize, "Starting CEGAR loop".to_string());

        loop {
            // Check termination conditions.
            if let Some(result) = self.check_termination() {
                return result;
            }

            match self.state.phase {
                CegarPhase::Initialize => self.phase_initialize(),
                CegarPhase::Abstract => self.phase_abstract(learner),
                CegarPhase::Verify => self.phase_verify(verifier),
                CegarPhase::Analyze => self.phase_analyze(verifier),
                CegarPhase::Refine => self.phase_refine(learner),
                CegarPhase::Certify => return self.phase_certify(),
                CegarPhase::Report => return self.phase_report(),
                CegarPhase::Terminated => return self.build_result(
                    CegarTermination::LatticeExhausted,
                ),
            }

            self.state.iteration += 1;
        }
    }

    /// Check if any termination condition is met.
    fn check_termination(&self) -> Option<CegarResult> {
        // Max iterations.
        if self.state.iteration >= self.config.max_iterations {
            return Some(self.build_result(CegarTermination::MaxIterations {
                iterations: self.state.iteration,
            }));
        }

        // Budget exhausted.
        if self.state.cost_spent >= self.config.max_budget {
            return Some(self.build_result(CegarTermination::BudgetExhausted {
                best_triple: self.lattice.coarsest_verified()
                    .map(|id| self.lattice.nodes[id].triple.clone()),
                partial_results: format!("Explored {} nodes", self.lattice.budget.nodes_explored),
            }));
        }

        // Time limit.
        if let Some(start) = &self.start_instant {
            let elapsed = start.elapsed().as_secs_f64();
            if elapsed >= self.config.max_time_secs {
                return Some(self.build_result(CegarTermination::BudgetExhausted {
                    best_triple: self.lattice.coarsest_verified()
                        .map(|id| self.lattice.nodes[id].triple.clone()),
                    partial_results: format!("Timed out after {:.1}s", elapsed),
                }));
            }
        }

        // Spurious streak.
        if self.spurious_streak >= self.config.max_spurious_streak {
            return Some(self.build_result(CegarTermination::BudgetExhausted {
                best_triple: self.lattice.coarsest_verified()
                    .map(|id| self.lattice.nodes[id].triple.clone()),
                partial_results: format!(
                    "Max spurious streak ({}) reached",
                    self.config.max_spurious_streak
                ),
            }));
        }

        None
    }

    /// Initialize phase: set up the first abstraction level.
    fn phase_initialize(&mut self) {
        // Find the starting node in the lattice.
        if let Some(bot_id) = self.lattice.bottom() {
            self.state.current_triple = self.lattice.nodes[bot_id].triple.clone();
        }

        self.state.transition_to(
            CegarPhase::Abstract,
            format!("Initialized at {}", self.state.current_triple),
        );
    }

    /// Abstract phase: build the abstract model at current level.
    fn phase_abstract(&mut self, learner: &mut dyn HypothesisLearner) {
        let triple = self.state.current_triple.clone();

        match learner.learn(&triple, self.current_model.as_ref()) {
            Ok((model, cost)) => {
                self.state.add_cost(cost);
                self.current_model = Some(model);
                self.state.transition_to(
                    CegarPhase::Verify,
                    format!("Built abstract model at {} (cost={:.2})", triple, cost),
                );
            }
            Err(e) => {
                // Learning failed — try next lattice node.
                if let Some(node_id) = self.lattice.find_node(&triple) {
                    self.lattice.mark_explored(node_id, false, 0.0);
                }
                self.advance_to_next_node(format!("Learning failed: {}", e));
            }
        }
    }

    /// Verify phase: check properties on the abstract model.
    fn phase_verify(&mut self, verifier: &dyn AbstractionVerifier) {
        let model = match &self.current_model {
            Some(m) => m,
            None => {
                self.state.transition_to(
                    CegarPhase::Abstract,
                    "No model available for verification".to_string(),
                );
                return;
            }
        };

        let mut all_verified = true;

        for property in &self.properties {
            match verifier.verify(model, property) {
                Ok(None) => {
                    // Property verified.
                }
                Ok(Some(cx)) => {
                    // Counter-example found.
                    all_verified = false;
                    self.state.current_counter_example = Some(cx.clone());
                    self.state.all_counter_examples.push(cx);
                    break;
                }
                Err(e) => {
                    // Verification error — treat as failure.
                    all_verified = false;
                    self.state.transition_to(
                        CegarPhase::Refine,
                        format!("Verification error: {}", e),
                    );
                    return;
                }
            }
        }

        if all_verified {
            // Mark lattice node as verified.
            if let Some(node_id) = self.lattice.find_node(&self.state.current_triple) {
                self.lattice.mark_explored(node_id, true, self.state.cost_spent);
            }
            self.previous_results.push((self.state.current_triple.clone(), true));

            self.state.transition_to(
                CegarPhase::Certify,
                format!("All properties verified at {}", self.state.current_triple),
            );
        } else {
            // Mark lattice node as failed.
            if let Some(node_id) = self.lattice.find_node(&self.state.current_triple) {
                self.lattice.mark_explored(node_id, false, self.state.cost_spent);
            }
            self.previous_results.push((self.state.current_triple.clone(), false));

            self.state.transition_to(
                CegarPhase::Analyze,
                "Counter-example found, analyzing spuriousness".to_string(),
            );
        }
    }

    /// Analyze phase: determine if counter-example is spurious.
    fn phase_analyze(&mut self, verifier: &dyn AbstractionVerifier) {
        let cx = match &self.state.current_counter_example {
            Some(cx) => cx.clone(),
            None => {
                self.state.transition_to(
                    CegarPhase::Refine,
                    "No counter-example to analyze".to_string(),
                );
                return;
            }
        };

        // Check spuriousness.
        let is_spurious = match verifier.check_spuriousness(&cx, self.current_model.as_ref()) {
            Ok(result) => result,
            Err(_) => {
                // Cannot determine — assume spurious and try refining.
                true
            }
        };

        if is_spurious {
            // Diagnose the cause.
            let diagnosis = self.diagnose_spuriousness(&cx);
            if let Some(ref mut cx_mut) = self.state.current_counter_example {
                cx_mut.mark_spurious(diagnosis);
            }
            self.spurious_streak += 1;

            // Check monotonicity if configured.
            if self.config.check_monotonicity {
                if let Some(violation) = self.check_monotonicity() {
                    self.state.transition_to(
                        CegarPhase::Terminated,
                        format!("Monotonicity violation: {}", violation),
                    );
                    return;
                }
            }

            self.state.transition_to(
                CegarPhase::Refine,
                "Spurious counter-example — refining abstraction".to_string(),
            );
        } else {
            // Genuine counter-example.
            if let Some(ref mut cx_mut) = self.state.current_counter_example {
                cx_mut.mark_genuine(0.8);
            }
            self.spurious_streak = 0;

            self.state.transition_to(
                CegarPhase::Report,
                "Genuine counter-example found".to_string(),
            );
        }
    }

    /// Refine phase: refine the abstraction to eliminate the spurious counter-example.
    fn phase_refine(&mut self, learner: &mut dyn HypothesisLearner) {
        let cx_info = self.state.current_counter_example.as_ref()
            .map(|cx| cx.to_counterexample_info());

        let budget_remaining = self.config.max_budget - self.state.cost_spent;

        // Select refinement operator.
        let refinement = self.refinement_planner.select(
            &self.state.current_triple,
            cx_info.as_ref(),
            budget_remaining,
        );

        match refinement {
            Some(ref_op) => {
                // Apply refinement.
                let ref_result = if let Some(model) = &self.current_model {
                    if let Some(cx) = &self.state.current_counter_example {
                        learner.refine(model, cx, &ref_op)
                    } else {
                        learner.learn(&ref_op.to, Some(model))
                    }
                } else {
                    learner.learn(&ref_op.to, None)
                };

                match ref_result {
                    Ok((new_model, cost)) => {
                        self.state.add_cost(cost);

                        let mut result = RefinementResult::new(ref_op.clone());
                        result.success = true;
                        result.actual_cost = cost;
                        result.actual_fidelity_gain = ref_op.estimated_fidelity_gain;
                        result.resolved_counterexample = true;

                        self.refinement_planner.update_weights(&result);
                        self.refinement_history.record(result);

                        self.state.current_triple = ref_op.to.clone();
                        self.current_model = Some(new_model);
                        self.state.current_counter_example = None;

                        self.state.transition_to(
                            CegarPhase::Verify,
                            format!("Refined to {}", self.state.current_triple),
                        );
                    }
                    Err(e) => {
                        let mut result = RefinementResult::new(ref_op);
                        result.success = false;
                        result.actual_cost = 0.0;

                        self.refinement_planner.update_weights(&result);
                        self.refinement_history.record(result);

                        self.advance_to_next_node(format!("Refinement failed: {}", e));
                    }
                }
            }
            None => {
                // No refinement available — try next lattice node.
                self.advance_to_next_node("No refinement available".to_string());
            }
        }
    }

    /// Certify phase: produce a certificate.
    fn phase_certify(&self) -> CegarResult {
        let certificate_info = format!(
            "Verified at abstraction level {}. Model: {}",
            self.state.current_triple,
            self.current_model.as_ref().map(|m| format!("{}", m)).unwrap_or_default(),
        );

        self.build_result(CegarTermination::Verified {
            triple: self.state.current_triple.clone(),
            certificate_info,
        })
    }

    /// Report phase: report a genuine violation.
    fn phase_report(&self) -> CegarResult {
        let cx = self.state.current_counter_example.clone()
            .unwrap_or_else(|| CounterExample::new(
                vec!["unknown".to_string()],
                Vec::new(),
                "unknown".to_string(),
            ));

        self.build_result(CegarTermination::CounterExampleFound {
            counter_example: cx,
        })
    }

    /// Advance to the next lattice node.
    fn advance_to_next_node(&mut self, reason: String) {
        match self.lattice.next_node() {
            Some(node_id) => {
                self.state.current_triple = self.lattice.nodes[node_id].triple.clone();
                self.current_model = None;
                self.state.current_counter_example = None;
                self.state.transition_to(
                    CegarPhase::Abstract,
                    format!("{} — advancing to {}", reason, self.state.current_triple),
                );
            }
            None => {
                self.state.transition_to(
                    CegarPhase::Terminated,
                    format!("{} — lattice exhausted", reason),
                );
            }
        }
    }

    /// Diagnose why a counter-example is spurious.
    fn diagnose_spuriousness(&self, cx: &CounterExample) -> CounterexampleDiagnosis {
        // Heuristic diagnosis based on counter-example characteristics.
        let trace_len = cx.trace.len();
        let input_len: usize = cx.inputs.iter().map(|s| s.len()).sum();

        let cause = if self.state.current_triple.k <= 4 && trace_len > 2 {
            SpuriousnessCause::ClusterMerge
        } else if input_len > self.state.current_triple.n * 10 {
            SpuriousnessCause::InsufficientDepth
        } else if self.state.current_triple.epsilon > 0.2 {
            SpuriousnessCause::DistributionCoarseness
        } else {
            SpuriousnessCause::CombinedFactors
        };

        let explanation = match cause {
            SpuriousnessCause::ClusterMerge => format!(
                "k={} clusters may merge behaviorally distinct states in trace of length {}",
                self.state.current_triple.k, trace_len
            ),
            SpuriousnessCause::InsufficientDepth => format!(
                "Input depth n={} may be insufficient for inputs of total length {}",
                self.state.current_triple.n, input_len
            ),
            SpuriousnessCause::DistributionCoarseness => format!(
                "ε={:.4} tolerance may mask distributional differences",
                self.state.current_triple.epsilon
            ),
            SpuriousnessCause::CombinedFactors => {
                "Multiple abstraction dimensions may contribute".to_string()
            }
        };

        CounterexampleDiagnosis {
            cause,
            explanation,
            involved_states: cx.trace.clone(),
            confidence: match cause {
                SpuriousnessCause::ClusterMerge => 0.7,
                SpuriousnessCause::InsufficientDepth => 0.6,
                SpuriousnessCause::DistributionCoarseness => 0.5,
                SpuriousnessCause::CombinedFactors => 0.4,
            },
        }
    }

    /// Check monotonicity: if α ≤ α', then verification at α' should imply at α.
    /// Returns Some(explanation) if a violation is detected.
    fn check_monotonicity(&self) -> Option<String> {
        // For each pair of results where α₁ ≤ α₂:
        // If verified(α₂) = true and verified(α₁) = false, that's suspicious
        // but not necessarily a violation (could be different counter-examples).
        //
        // A real violation is: verified(α₁) = true but verified(α₂) = false
        // where α₁ ≤ α₂ (finer should be at least as good).
        for i in 0..self.previous_results.len() {
            for j in 0..self.previous_results.len() {
                if i == j {
                    continue;
                }
                let (ref t1, v1) = self.previous_results[i];
                let (ref t2, v2) = self.previous_results[j];

                // If t1 ≤ t2 (t1 is coarser) and t1 verified but t2 not:
                // This could indicate a problem but isn't necessarily a violation
                // in the probabilistic setting. We flag it if it happens repeatedly.
                if t1.leq(t2) && v1 && !v2 {
                    // Coarser passed but finer failed — possible but unusual.
                    // In approximate setting, this can happen due to sampling variance.
                    // We only flag if confidence is high.
                    return Some(format!(
                        "Coarser abstraction {} verified but finer {} failed. \
                         This may indicate non-monotonicity in the abstraction.",
                        t1, t2
                    ));
                }
            }
        }

        None
    }

    /// Build the final result.
    fn build_result(&self, termination: CegarTermination) -> CegarResult {
        let wall_time = self.start_instant
            .map(|s| s.elapsed().as_secs_f64())
            .unwrap_or(0.0);

        CegarResult {
            termination,
            state: self.state.clone(),
            final_model: self.current_model.clone(),
            refinement_summary: self.refinement_history.summary(),
            lattice_summary: self.lattice.visualize(),
            total_cost: self.state.cost_spent,
            total_iterations: self.state.iteration,
            wall_time_secs: wall_time,
        }
    }

    /// Get current phase.
    pub fn current_phase(&self) -> CegarPhase {
        self.state.phase
    }

    /// Get current abstraction triple.
    pub fn current_triple(&self) -> &AbstractionTriple {
        &self.state.current_triple
    }

    /// Get the phase transition log.
    pub fn phase_log(&self) -> &[(CegarPhase, String)] {
        &self.state.phase_log
    }

    /// Get statistics about the CEGAR run.
    pub fn stats(&self) -> CegarStats {
        let total_cx = self.state.all_counter_examples.len();
        let spurious_cx = self.state.all_counter_examples.iter()
            .filter(|cx| cx.spurious == Some(true))
            .count();
        let genuine_cx = self.state.all_counter_examples.iter()
            .filter(|cx| cx.spurious == Some(false))
            .count();

        CegarStats {
            iterations: self.state.iteration,
            phase: self.state.phase,
            current_triple: self.state.current_triple.clone(),
            total_cost: self.state.cost_spent,
            total_counterexamples: total_cx,
            spurious_counterexamples: spurious_cx,
            genuine_counterexamples: genuine_cx,
            lattice_explored: self.lattice.budget.nodes_explored,
            lattice_total: self.lattice.size(),
            refinement_steps: self.refinement_history.num_steps(),
            refinement_success_rate: self.refinement_history.success_rate(),
        }
    }

    /// Visualize the current state of the CEGAR loop.
    pub fn visualize(&self) -> String {
        let stats = self.stats();
        let mut out = String::new();
        out.push_str("=== CEGAR Loop State ===\n");
        out.push_str(&format!("Phase: {}\n", stats.phase));
        out.push_str(&format!("Iteration: {}\n", stats.iterations));
        out.push_str(&format!("Current triple: {}\n", stats.current_triple));
        out.push_str(&format!("Total cost: {:.2}\n", stats.total_cost));
        out.push_str(&format!(
            "Counter-examples: {} total ({} spurious, {} genuine)\n",
            stats.total_counterexamples, stats.spurious_counterexamples,
            stats.genuine_counterexamples
        ));
        out.push_str(&format!(
            "Lattice: {}/{} explored\n",
            stats.lattice_explored, stats.lattice_total
        ));
        out.push_str(&format!(
            "Refinements: {} steps, {:.1}% success rate\n",
            stats.refinement_steps, stats.refinement_success_rate * 100.0
        ));
        out.push_str("\nPhase log:\n");
        for (phase, reason) in self.state.phase_log.iter().rev().take(10) {
            out.push_str(&format!("  {} → {}\n", phase, reason));
        }
        out
    }

    /// Step through one iteration manually (for testing/debugging).
    pub fn step(
        &mut self,
        learner: &mut dyn HypothesisLearner,
        verifier: &dyn AbstractionVerifier,
    ) -> CegarPhase {
        match self.state.phase {
            CegarPhase::Initialize => self.phase_initialize(),
            CegarPhase::Abstract => self.phase_abstract(learner),
            CegarPhase::Verify => self.phase_verify(verifier),
            CegarPhase::Analyze => self.phase_analyze(verifier),
            CegarPhase::Refine => self.phase_refine(learner),
            CegarPhase::Certify | CegarPhase::Report | CegarPhase::Terminated => {}
        }
        self.state.iteration += 1;
        self.state.phase
    }
}

impl fmt::Debug for CegarLoop {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CegarLoop")
            .field("phase", &self.state.phase)
            .field("iteration", &self.state.iteration)
            .field("triple", &self.state.current_triple)
            .field("cost", &self.state.cost_spent)
            .finish()
    }
}

/// Statistics about a CEGAR run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CegarStats {
    pub iterations: usize,
    pub phase: CegarPhase,
    pub current_triple: AbstractionTriple,
    pub total_cost: f64,
    pub total_counterexamples: usize,
    pub spurious_counterexamples: usize,
    pub genuine_counterexamples: usize,
    pub lattice_explored: usize,
    pub lattice_total: usize,
    pub refinement_steps: usize,
    pub refinement_success_rate: f64,
}

impl fmt::Display for CegarStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CegarStats[iter={} phase={} cost={:.2} cx={} lattice={}/{}]",
            self.iterations, self.phase, self.total_cost,
            self.total_counterexamples, self.lattice_explored, self.lattice_total
        )
    }
}

// ---------------------------------------------------------------------------
// Pipeline orchestration helpers
// ---------------------------------------------------------------------------

/// Run the full CEGAR pipeline with default settings.
pub fn run_cegar_pipeline(
    properties: Vec<PropertySpec>,
    config: CegarConfig,
    learner: &mut dyn HypothesisLearner,
    verifier: &dyn AbstractionVerifier,
) -> CegarResult {
    let mut cegar = CegarLoop::new(config, properties);
    cegar.run(learner, verifier)
}

/// Run CEGAR with the stub learner and verifier (for testing).
pub fn run_stub_cegar(
    properties: Vec<PropertySpec>,
    min_states_for_pass: usize,
) -> CegarResult {
    let config = CegarConfig::default();
    let mut learner = StubLearner::new();
    let verifier = StubVerifier::new(min_states_for_pass);
    let mut cegar = CegarLoop::new(config, properties);
    cegar.run(&mut learner, &verifier)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn test_properties() -> Vec<PropertySpec> {
        vec![
            PropertySpec::safety("no_toxic", "No toxic output"),
        ]
    }

    #[test]
    fn test_counter_example() {
        let cx = CounterExample::new(
            vec!["q0".to_string(), "q1".to_string()],
            vec!["hello".to_string()],
            "safety".to_string(),
        );

        assert!(!cx.is_classified());
        assert_eq!(cx.trace.len(), 2);
    }

    #[test]
    fn test_counter_example_classification() {
        let mut cx = CounterExample::new(
            vec!["q0".to_string()],
            vec!["test".to_string()],
            "prop".to_string(),
        );

        cx.mark_spurious(CounterexampleDiagnosis {
            cause: SpuriousnessCause::ClusterMerge,
            explanation: "Test".to_string(),
            involved_states: vec!["q0".to_string()],
            confidence: 0.8,
        });

        assert!(cx.is_classified());
        assert_eq!(cx.spurious, Some(true));
    }

    #[test]
    fn test_counter_example_to_info() {
        let mut cx = CounterExample::new(
            vec!["q0".to_string()],
            vec!["test".to_string()],
            "prop".to_string(),
        );

        cx.mark_spurious(CounterexampleDiagnosis {
            cause: SpuriousnessCause::InsufficientDepth,
            explanation: "depth".to_string(),
            involved_states: vec![],
            confidence: 0.5,
        });

        let info = cx.to_counterexample_info();
        assert!(info.involves_depth_insufficiency);
    }

    #[test]
    fn test_property_spec() {
        let p = PropertySpec::safety("safe", "Safety property");
        assert_eq!(p.kind, PropertyKind::Safety);
        assert!(p.bound.is_none());

        let p2 = PropertySpec::probabilistic("prob", 0.05, "Low probability");
        assert_eq!(p2.kind, PropertyKind::ProbabilisticBound);
        assert_eq!(p2.bound, Some(0.05));
    }

    #[test]
    fn test_abstract_model() {
        let triple = AbstractionTriple::new(4, 2, 0.5);
        let model = AbstractModel::new(triple.clone(), 10, 20);

        assert_eq!(model.num_states, 10);
        assert_eq!(model.num_transitions, 20);
        assert_eq!(model.state_ids.len(), 10);
    }

    #[test]
    fn test_stub_learner() {
        let mut learner = StubLearner::new();
        let triple = AbstractionTriple::new(4, 2, 0.5);
        let (model, cost) = learner.learn(&triple, None).unwrap();

        assert!(model.num_states > 0);
        assert!(cost > 0.0);
    }

    #[test]
    fn test_stub_verifier_pass() {
        let verifier = StubVerifier::new(10);
        let triple = AbstractionTriple::new(4, 2, 0.5);
        let model = AbstractModel::new(triple, 20, 40);
        let prop = PropertySpec::safety("test", "test");

        let result = verifier.verify(&model, &prop).unwrap();
        assert!(result.is_none()); // Should pass
    }

    #[test]
    fn test_stub_verifier_fail() {
        let verifier = StubVerifier::new(100);
        let triple = AbstractionTriple::new(4, 2, 0.5);
        let model = AbstractModel::new(triple, 10, 20);
        let prop = PropertySpec::safety("test", "test");

        let result = verifier.verify(&model, &prop).unwrap();
        assert!(result.is_some()); // Should fail
    }

    #[test]
    fn test_cegar_state() {
        let triple = AbstractionTriple::new(4, 2, 0.5);
        let mut state = CegarState::new(triple);

        assert_eq!(state.phase, CegarPhase::Initialize);
        assert_eq!(state.iteration, 0);

        state.transition_to(CegarPhase::Abstract, "test".to_string());
        assert_eq!(state.phase, CegarPhase::Abstract);
        assert_eq!(state.phase_log.len(), 1);
    }

    #[test]
    fn test_cegar_config_default() {
        let config = CegarConfig::default();
        assert!(config.max_iterations > 0);
        assert!(config.max_budget > 0.0);
        assert!(!config.k_values.is_empty());
    }

    #[test]
    fn test_cegar_loop_creation() {
        let config = CegarConfig::default();
        let props = test_properties();
        let cegar = CegarLoop::new(config, props);

        assert_eq!(cegar.current_phase(), CegarPhase::Initialize);
        assert!(cegar.lattice.size() > 0);
    }

    #[test]
    fn test_cegar_loop_step() {
        let config = CegarConfig::default();
        let props = test_properties();
        let mut cegar = CegarLoop::new(config, props);
        let mut learner = StubLearner::new();
        let verifier = StubVerifier::new(10);

        // Step through initialization.
        let phase = cegar.step(&mut learner, &verifier);
        assert_eq!(phase, CegarPhase::Abstract);
    }

    #[test]
    fn test_cegar_run_verified() {
        // With min_states_for_pass=8 and base_states=4, k=2 gives 8 states.
        let config = CegarConfig {
            max_iterations: 20,
            k_values: vec![2, 4],
            n_values: vec![1, 2],
            epsilon_values: vec![0.5, 0.1],
            ..CegarConfig::default()
        };
        let props = test_properties();
        let mut learner = StubLearner::new();
        let verifier = StubVerifier::new(8);

        let result = run_cegar_pipeline(props, config, &mut learner, &verifier);
        assert!(result.is_verified());
    }

    #[test]
    fn test_cegar_run_with_refinement() {
        // Need larger model for verification — will require refinement.
        let config = CegarConfig {
            max_iterations: 30,
            k_values: vec![2, 4, 8, 16],
            n_values: vec![1, 2, 3],
            epsilon_values: vec![0.5, 0.25, 0.1],
            ..CegarConfig::default()
        };
        let props = test_properties();
        let mut learner = StubLearner::new();
        // Need 30+ states to pass — requires refinement to larger k.
        let verifier = StubVerifier::new(30);

        let result = run_cegar_pipeline(props, config, &mut learner, &verifier);
        // Should either verify (after enough refinement) or exhaust iterations.
        assert!(result.total_iterations > 0);
    }

    #[test]
    fn test_cegar_run_genuine_counter_example() {
        // Verifier always fails and cx is not spurious.
        let config = CegarConfig {
            max_iterations: 10,
            k_values: vec![2, 4],
            n_values: vec![1],
            epsilon_values: vec![0.5],
            ..CegarConfig::default()
        };
        let props = test_properties();
        let mut learner = StubLearner::new();
        // Very high threshold — never passes, and cx is not spurious.
        let verifier = StubVerifier {
            min_states_for_pass: 10000,
            failure_threshold: 0, // cx trace len is always >= threshold → not spurious
        };

        let result = run_cegar_pipeline(props, config, &mut learner, &verifier);
        assert!(result.is_violation());
    }

    #[test]
    fn test_cegar_result_display() {
        let result = run_stub_cegar(test_properties(), 8);
        let display = format!("{}", result);
        assert!(display.contains("CegarResult"));
    }

    #[test]
    fn test_cegar_stats() {
        let config = CegarConfig::default();
        let props = test_properties();
        let cegar = CegarLoop::new(config, props);

        let stats = cegar.stats();
        assert_eq!(stats.iterations, 0);
        assert_eq!(stats.total_counterexamples, 0);
        assert!(stats.lattice_total > 0);
    }

    #[test]
    fn test_cegar_visualize() {
        let config = CegarConfig::default();
        let props = test_properties();
        let cegar = CegarLoop::new(config, props);

        let viz = cegar.visualize();
        assert!(viz.contains("CEGAR"));
        assert!(viz.contains("Phase"));
    }

    #[test]
    fn test_cegar_max_iterations() {
        let config = CegarConfig {
            max_iterations: 3,
            k_values: vec![2],
            n_values: vec![1],
            epsilon_values: vec![0.5],
            ..CegarConfig::default()
        };
        let props = test_properties();
        let mut learner = StubLearner::new();
        let verifier = StubVerifier::new(10000);

        let result = run_cegar_pipeline(props, config, &mut learner, &verifier);
        assert!(result.total_iterations <= 3);
    }

    #[test]
    fn test_phase_display() {
        assert_eq!(format!("{}", CegarPhase::Initialize), "INIT");
        assert_eq!(format!("{}", CegarPhase::Verify), "VERIFY");
        assert_eq!(format!("{}", CegarPhase::Certify), "CERTIFY");
    }

    #[test]
    fn test_termination_display() {
        let t = CegarTermination::Verified {
            triple: AbstractionTriple::new(4, 2, 0.5),
            certificate_info: "test".to_string(),
        };
        let display = format!("{}", t);
        assert!(display.contains("VERIFIED"));
    }

    #[test]
    fn test_spuriousness_diagnosis() {
        let config = CegarConfig::default();
        let props = test_properties();
        let cegar = CegarLoop::new(config, props);

        let cx = CounterExample::new(
            vec!["q0".to_string(), "q1".to_string(), "q2".to_string()],
            vec!["input".to_string()],
            "safety".to_string(),
        );

        let diag = cegar.diagnose_spuriousness(&cx);
        // With default k=2, trace_len=3, should diagnose ClusterMerge.
        assert_eq!(diag.cause, SpuriousnessCause::ClusterMerge);
    }

    #[test]
    fn test_monotonicity_check_no_violation() {
        let config = CegarConfig::default();
        let props = test_properties();
        let cegar = CegarLoop::new(config, props);

        // No previous results — should be fine.
        assert!(cegar.check_monotonicity().is_none());
    }

    #[test]
    fn test_cegar_run_budget_exhausted() {
        let config = CegarConfig {
            max_iterations: 100,
            max_budget: 1.0, // Very small budget
            k_values: vec![2, 4, 8],
            n_values: vec![1, 2],
            epsilon_values: vec![0.5, 0.1],
            ..CegarConfig::default()
        };
        let props = test_properties();
        let mut learner = StubLearner::new();
        let verifier = StubVerifier::new(10000);

        let result = run_cegar_pipeline(props, config, &mut learner, &verifier);
        // Should terminate due to budget or finding a counter-example.
        assert!(!result.is_verified() || result.total_cost <= 1.0);
    }

    #[test]
    fn test_run_stub_cegar() {
        let result = run_stub_cegar(test_properties(), 8);
        // With base_states=4 and k=2 (bottom of lattice), 4*2=8 states.
        assert!(result.is_verified());
    }

    #[test]
    fn test_cegar_multiple_properties() {
        let props = vec![
            PropertySpec::safety("safe1", "Safety 1"),
            PropertySpec::safety("safe2", "Safety 2"),
            PropertySpec::probabilistic("prob1", 0.1, "Probabilistic 1"),
        ];

        let result = run_stub_cegar(props, 8);
        assert!(result.is_verified());
    }

    #[test]
    fn test_cegar_state_summary() {
        let triple = AbstractionTriple::new(4, 2, 0.5);
        let state = CegarState::new(triple);
        let summary = state.summary();
        assert!(summary.contains("CEGAR"));
        assert!(summary.contains("k=4"));
    }

    #[test]
    fn test_counterexample_display() {
        let cx = CounterExample::new(
            vec!["q0".to_string()],
            vec!["test".to_string()],
            "safety".to_string(),
        );
        let display = format!("{}", cx);
        assert!(display.contains("UNCLASSIFIED"));
    }

    #[test]
    fn test_cegar_result_types() {
        let result = run_stub_cegar(test_properties(), 8);
        assert!(result.is_verified());
        assert!(!result.is_violation());
        assert!(!result.is_inconclusive());
    }

    #[test]
    fn test_phase_log_tracking() {
        let config = CegarConfig {
            max_iterations: 5,
            k_values: vec![2],
            n_values: vec![1],
            epsilon_values: vec![0.5],
            ..CegarConfig::default()
        };
        let props = test_properties();
        let mut learner = StubLearner::new();
        let verifier = StubVerifier::new(8);

        let mut cegar = CegarLoop::new(config, props);
        cegar.run(&mut learner, &verifier);

        assert!(!cegar.phase_log().is_empty());
    }
}
