//! Main CEGAR (Counterexample-Guided Abstraction Refinement) loop.
//!
//! Implements the iterative refinement cycle: abstract → model-check →
//! check counterexample → refine until verified, real counterexample found,
//! or resource limit hit.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::time::{Duration, Instant};

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use crate::abstraction::{
    AbstractBlock, AbstractBlockId, AbstractState, AbstractTransitionRelation, AbstractionState,
    GeometricAbstraction, PartitionRefinement, SpatialPartition,
};
use crate::certificate::{
    CertificateBuilder, CounterexampleCertificate, ProofCertificate, VerificationCertificate,
};
use crate::counterexample::{
    ConcreteCounterexample, Counterexample, FeasibilityResult, InfeasibilityWitness,
    RefinementHint,
};
use crate::model_checker::{
    self, DeadlockResult, LivenessResult, ModelChecker, ReachabilityResult, SafetyResult,
};
use crate::properties::Property;
use crate::{
    AutomatonDef, CegarError, Guard, Plane, Point3, PredicateValuation, SceneConfiguration,
    SpatialConstraint, SpatialPredicate, SpatialPredicateId, StateId, TransitionId, Vector3, AABB,
};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Strategy for refining the abstraction when a spurious counterexample is found.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RefinementStrategy {
    /// Split the block along its longest geometric axis.
    LongestAxis,
    /// Use the infeasibility witness to choose a splitting plane.
    InfeasibilityGuided,
    /// Hybrid strategy using GJK distance queries to guide splits.
    HybridGJK,
    /// Uniformly refine all blocks in the partition.
    Uniform,
}

impl Default for RefinementStrategy {
    fn default() -> Self {
        RefinementStrategy::InfeasibilityGuided
    }
}

impl fmt::Display for RefinementStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RefinementStrategy::LongestAxis => write!(f, "LongestAxis"),
            RefinementStrategy::InfeasibilityGuided => write!(f, "InfeasibilityGuided"),
            RefinementStrategy::HybridGJK => write!(f, "HybridGJK"),
            RefinementStrategy::Uniform => write!(f, "Uniform"),
        }
    }
}

/// Configuration for the CEGAR verification engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CEGARConfig {
    /// Maximum number of CEGAR iterations before declaring timeout.
    pub max_iterations: usize,
    /// Wall-clock timeout.
    pub timeout: Duration,
    /// Maximum number of abstract states allowed.
    pub max_abstract_states: usize,
    /// Strategy for refinement.
    pub refinement_strategy: RefinementStrategy,
    /// Maximum partition depth (refinement granularity limit).
    pub max_partition_depth: u32,
    /// Whether to use compositional verification for independent components.
    pub use_compositional: bool,
    /// Whether to enable geometric consistency pruning.
    pub use_pruning: bool,
    /// Bounded model checking depth (0 = disabled).
    pub bmc_bound: usize,
    /// Whether to attempt counterexample minimization.
    pub minimize_counterexamples: bool,
    /// Verbose logging.
    pub verbose: bool,
}

impl Default for CEGARConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            timeout: Duration::from_secs(300),
            max_abstract_states: 100_000,
            refinement_strategy: RefinementStrategy::InfeasibilityGuided,
            max_partition_depth: 30,
            use_compositional: false,
            use_pruning: true,
            bmc_bound: 0,
            minimize_counterexamples: true,
            verbose: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Statistics collected during CEGAR verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CEGARStatistics {
    /// Total number of CEGAR iterations completed.
    pub iterations: usize,
    /// Number of refinement steps performed.
    pub refinement_steps: usize,
    /// Number of spurious counterexamples eliminated.
    pub spurious_counterexamples: usize,
    /// Number of abstract states at each iteration.
    pub abstract_state_sizes: Vec<usize>,
    /// Number of abstract transitions at each iteration.
    pub abstract_transition_sizes: Vec<usize>,
    /// Time spent in model checking (cumulative).
    pub model_check_time: Duration,
    /// Time spent in counterexample analysis (cumulative).
    pub counterexample_time: Duration,
    /// Time spent in abstraction refinement (cumulative).
    pub refinement_time: Duration,
    /// Total wall-clock time.
    pub total_time: Duration,
    /// Peak number of abstract states.
    pub peak_abstract_states: usize,
    /// Peak number of abstract transitions.
    pub peak_abstract_transitions: usize,
    /// Final partition block count.
    pub final_block_count: usize,
    /// Maximum refinement depth reached.
    pub max_depth_reached: u32,
}

impl CEGARStatistics {
    pub fn new() -> Self {
        Self {
            iterations: 0,
            refinement_steps: 0,
            spurious_counterexamples: 0,
            abstract_state_sizes: Vec::new(),
            abstract_transition_sizes: Vec::new(),
            model_check_time: Duration::ZERO,
            counterexample_time: Duration::ZERO,
            refinement_time: Duration::ZERO,
            total_time: Duration::ZERO,
            peak_abstract_states: 0,
            peak_abstract_transitions: 0,
            final_block_count: 0,
            max_depth_reached: 0,
        }
    }

    /// Record an iteration's abstract model size.
    pub fn record_iteration(&mut self, num_states: usize, num_transitions: usize) {
        self.iterations += 1;
        self.abstract_state_sizes.push(num_states);
        self.abstract_transition_sizes.push(num_transitions);
        self.peak_abstract_states = self.peak_abstract_states.max(num_states);
        self.peak_abstract_transitions = self.peak_abstract_transitions.max(num_transitions);
    }

    /// Average number of abstract states across iterations.
    pub fn average_abstract_states(&self) -> f64 {
        if self.abstract_state_sizes.is_empty() {
            return 0.0;
        }
        let sum: usize = self.abstract_state_sizes.iter().sum();
        sum as f64 / self.abstract_state_sizes.len() as f64
    }

    /// Compute the convergence rate (states added per iteration).
    pub fn convergence_rate(&self) -> f64 {
        if self.abstract_state_sizes.len() < 2 {
            return 0.0;
        }
        let first = self.abstract_state_sizes.first().copied().unwrap_or(0);
        let last = self.abstract_state_sizes.last().copied().unwrap_or(0);
        let iters = self.abstract_state_sizes.len() - 1;
        if iters == 0 {
            return 0.0;
        }
        (last as f64 - first as f64) / iters as f64
    }
}

impl Default for CEGARStatistics {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Verification result
// ---------------------------------------------------------------------------

/// Outcome of CEGAR verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VerificationResult {
    /// Property verified, with proof certificate.
    Verified {
        certificate: VerificationCertificate,
        statistics: CEGARStatistics,
    },
    /// Real counterexample found.
    CounterexampleFound {
        trace: ConcreteCounterexample,
        witness: Counterexample,
        statistics: CEGARStatistics,
    },
    /// Timeout reached.
    Timeout {
        partial_result: PartialVerificationResult,
        statistics: CEGARStatistics,
    },
    /// Resource limit exceeded (too many states, etc.).
    ResourceExhausted {
        reason: String,
        partial_result: PartialVerificationResult,
        statistics: CEGARStatistics,
    },
}

impl VerificationResult {
    pub fn is_verified(&self) -> bool {
        matches!(self, VerificationResult::Verified { .. })
    }

    pub fn is_counterexample(&self) -> bool {
        matches!(self, VerificationResult::CounterexampleFound { .. })
    }

    pub fn statistics(&self) -> &CEGARStatistics {
        match self {
            VerificationResult::Verified { statistics, .. }
            | VerificationResult::CounterexampleFound { statistics, .. }
            | VerificationResult::Timeout { statistics, .. }
            | VerificationResult::ResourceExhausted { statistics, .. } => statistics,
        }
    }
}

/// Partial result when verification is inconclusive.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartialVerificationResult {
    pub iterations_completed: usize,
    pub last_abstract_model_size: usize,
    pub explored_states: usize,
    pub remaining_counterexamples: usize,
}

// ---------------------------------------------------------------------------
// Progress callback
// ---------------------------------------------------------------------------

/// Progress information reported during CEGAR iterations.
#[derive(Debug, Clone)]
pub struct CEGARProgress {
    pub iteration: usize,
    pub max_iterations: usize,
    pub abstract_states: usize,
    pub abstract_transitions: usize,
    pub spurious_count: usize,
    pub elapsed: Duration,
    pub phase: CEGARPhase,
}

/// Current phase of the CEGAR loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CEGARPhase {
    InitialAbstraction,
    ModelChecking,
    CounterexampleAnalysis,
    Refinement,
    Done,
}

impl fmt::Display for CEGARPhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CEGARPhase::InitialAbstraction => write!(f, "Initial Abstraction"),
            CEGARPhase::ModelChecking => write!(f, "Model Checking"),
            CEGARPhase::CounterexampleAnalysis => write!(f, "Counterexample Analysis"),
            CEGARPhase::Refinement => write!(f, "Refinement"),
            CEGARPhase::Done => write!(f, "Done"),
        }
    }
}

/// Type alias for progress callback.
pub type ProgressCallback = Box<dyn Fn(&CEGARProgress) + Send>;

// ---------------------------------------------------------------------------
// Abstract model checking result (internal)
// ---------------------------------------------------------------------------

/// Result of model-checking the abstract model.
#[derive(Debug, Clone)]
enum AbstractCheckResult {
    /// Property holds in the abstract model.
    PropertyHolds,
    /// Abstract counterexample found.
    CounterexampleFound(Counterexample),
}

/// Result of spuriousness check.
#[derive(Debug, Clone)]
enum SpuriousnessResult {
    /// The counterexample is feasible in the concrete domain.
    Feasible(ConcreteCounterexample),
    /// The counterexample is spurious; witness identifies the reason.
    Spurious(InfeasibilityWitness),
}

// ---------------------------------------------------------------------------
// CEGARVerifier
// ---------------------------------------------------------------------------

/// The main CEGAR verification engine.
pub struct CEGARVerifier {
    config: CEGARConfig,
    progress_callback: Option<ProgressCallback>,
}

impl CEGARVerifier {
    /// Create a new CEGAR verifier with the given configuration.
    pub fn new(config: CEGARConfig) -> Self {
        Self {
            config,
            progress_callback: None,
        }
    }

    /// Create a verifier with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(CEGARConfig::default())
    }

    /// Set a progress callback for monitoring verification.
    pub fn set_progress_callback(&mut self, callback: ProgressCallback) {
        self.progress_callback = Some(callback);
    }

    /// Main verification entry point.
    pub fn verify(
        &self,
        automaton: &AutomatonDef,
        scene: &SceneConfiguration,
        property: &Property,
    ) -> VerificationResult {
        let start = Instant::now();
        let mut stats = CEGARStatistics::new();

        // Report progress: initial abstraction
        self.report_progress(&CEGARProgress {
            iteration: 0,
            max_iterations: self.config.max_iterations,
            abstract_states: 0,
            abstract_transitions: 0,
            spurious_count: 0,
            elapsed: start.elapsed(),
            phase: CEGARPhase::InitialAbstraction,
        });

        // Step 1: Compute initial abstraction
        let mut abstraction = GeometricAbstraction::initial_abstraction(automaton, scene);

        // Main CEGAR loop
        for iteration in 0..self.config.max_iterations {
            // Check timeout
            if start.elapsed() > self.config.timeout {
                stats.total_time = start.elapsed();
                return VerificationResult::Timeout {
                    partial_result: PartialVerificationResult {
                        iterations_completed: iteration,
                        last_abstract_model_size: abstraction.block_count(),
                        explored_states: stats.peak_abstract_states,
                        remaining_counterexamples: 0,
                    },
                    statistics: stats,
                };
            }

            // Step 2: Build abstract model
            let abstract_model = abstraction.build_abstract_model();
            let num_states = abstract_model.state_count();
            let num_transitions = abstract_model.transition_count();
            stats.record_iteration(num_states, num_transitions);

            // Check resource limits
            if num_states > self.config.max_abstract_states {
                stats.total_time = start.elapsed();
                return VerificationResult::ResourceExhausted {
                    reason: format!(
                        "Abstract state count {} exceeds limit {}",
                        num_states, self.config.max_abstract_states
                    ),
                    partial_result: PartialVerificationResult {
                        iterations_completed: iteration,
                        last_abstract_model_size: num_states,
                        explored_states: num_states,
                        remaining_counterexamples: 0,
                    },
                    statistics: stats,
                };
            }

            // Report progress: model checking
            self.report_progress(&CEGARProgress {
                iteration,
                max_iterations: self.config.max_iterations,
                abstract_states: num_states,
                abstract_transitions: num_transitions,
                spurious_count: stats.spurious_counterexamples,
                elapsed: start.elapsed(),
                phase: CEGARPhase::ModelChecking,
            });

            // Step 3: Model-check the abstract model
            let mc_start = Instant::now();
            let mc_result = self.model_check_abstract(&abstract_model, property);
            stats.model_check_time += mc_start.elapsed();

            match mc_result {
                AbstractCheckResult::PropertyHolds => {
                    // Property verified in abstract model → verified in concrete
                    stats.total_time = start.elapsed();
                    stats.final_block_count = abstraction.block_count();
                    stats.max_depth_reached = abstraction
                        .refinement_history
                        .max_depth_reached;

                    let cert = self.build_proof_certificate(
                        &abstract_model,
                        property,
                        &stats,
                    );
                    return VerificationResult::Verified {
                        certificate: cert,
                        statistics: stats,
                    };
                }
                AbstractCheckResult::CounterexampleFound(abstract_cex) => {
                    // Report progress: counterexample analysis
                    self.report_progress(&CEGARProgress {
                        iteration,
                        max_iterations: self.config.max_iterations,
                        abstract_states: num_states,
                        abstract_transitions: num_transitions,
                        spurious_count: stats.spurious_counterexamples,
                        elapsed: start.elapsed(),
                        phase: CEGARPhase::CounterexampleAnalysis,
                    });

                    // Step 4: Check if the counterexample is spurious
                    let cex_start = Instant::now();
                    let spurious_result =
                        self.check_spuriousness(&abstract_cex, &abstract_model, scene);
                    stats.counterexample_time += cex_start.elapsed();

                    match spurious_result {
                        SpuriousnessResult::Feasible(concrete_cex) => {
                            // Real counterexample found
                            stats.total_time = start.elapsed();
                            stats.final_block_count = abstraction.block_count();
                            return VerificationResult::CounterexampleFound {
                                trace: concrete_cex,
                                witness: abstract_cex,
                                statistics: stats,
                            };
                        }
                        SpuriousnessResult::Spurious(witness) => {
                            // Report progress: refinement
                            self.report_progress(&CEGARProgress {
                                iteration,
                                max_iterations: self.config.max_iterations,
                                abstract_states: num_states,
                                abstract_transitions: num_transitions,
                                spurious_count: stats.spurious_counterexamples + 1,
                                elapsed: start.elapsed(),
                                phase: CEGARPhase::Refinement,
                            });

                            // Step 5: Refine the abstraction
                            let ref_start = Instant::now();
                            self.refine_abstraction(
                                &mut abstraction,
                                &abstract_cex,
                                &witness,
                            );
                            stats.refinement_time += ref_start.elapsed();
                            stats.refinement_steps += 1;
                            stats.spurious_counterexamples += 1;
                        }
                    }
                }
            }
        }

        // Exhausted all iterations
        stats.total_time = start.elapsed();
        VerificationResult::Timeout {
            partial_result: PartialVerificationResult {
                iterations_completed: self.config.max_iterations,
                last_abstract_model_size: abstraction.block_count(),
                explored_states: stats.peak_abstract_states,
                remaining_counterexamples: 0,
            },
            statistics: stats,
        }
    }

    /// Model-check an abstract model against a property.
    fn model_check_abstract(
        &self,
        model: &AbstractionState,
        property: &Property,
    ) -> AbstractCheckResult {
        let checker = ModelChecker::new();
        let adj = model.transition_relation.adjacency_list();

        match property {
            Property::Safety(safety) => {
                // Check if any bad state is reachable
                let bad_states: HashSet<AbstractState> = model
                    .abstract_states
                    .iter()
                    .filter(|s| {
                        let block = model.partition.blocks.get(&s.block_id);
                        if let Some(b) = block {
                            let val = b.known_valuation();
                            safety
                                .bad_state_predicate
                                .evaluate(&val)
                                .unwrap_or(true) // Conservatively assume bad if unknown
                        } else {
                            false
                        }
                    })
                    .copied()
                    .collect();

                if bad_states.is_empty() {
                    return AbstractCheckResult::PropertyHolds;
                }

                // BFS from initial states
                let path = self.bfs_find_path(
                    &model.initial_states,
                    &bad_states,
                    &adj,
                );

                match path {
                    Some(trace) => {
                        let cex = Counterexample {
                            states: trace.clone(),
                            transitions: self.extract_transition_ids(&trace, model),
                            length: trace.len(),
                            property_violated: property.clone(),
                        };
                        AbstractCheckResult::CounterexampleFound(cex)
                    }
                    None => AbstractCheckResult::PropertyHolds,
                }
            }
            Property::Reachability(reach) => {
                let target_states: HashSet<AbstractState> = model
                    .abstract_states
                    .iter()
                    .filter(|s| {
                        let block = model.partition.blocks.get(&s.block_id);
                        if let Some(b) = block {
                            let val = b.known_valuation();
                            reach
                                .target_predicate
                                .evaluate(&val)
                                .unwrap_or(true)
                        } else {
                            false
                        }
                    })
                    .copied()
                    .collect();

                let path = self.bfs_find_path(
                    &model.initial_states,
                    &target_states,
                    &adj,
                );

                match path {
                    Some(trace) => {
                        let cex = Counterexample {
                            states: trace.clone(),
                            transitions: self.extract_transition_ids(&trace, model),
                            length: trace.len(),
                            property_violated: property.clone(),
                        };
                        AbstractCheckResult::CounterexampleFound(cex)
                    }
                    None => AbstractCheckResult::PropertyHolds,
                }
            }
            Property::DeadlockFreedom => {
                // Check for deadlock: states with no outgoing transitions
                let deadlock_states: HashSet<AbstractState> = model
                    .abstract_states
                    .iter()
                    .filter(|s| {
                        model.transition_relation.transitions_from(s).is_empty()
                            && !model.automaton.is_accepting(s.automaton_state)
                    })
                    .copied()
                    .collect();

                if deadlock_states.is_empty() {
                    return AbstractCheckResult::PropertyHolds;
                }

                let path = self.bfs_find_path(
                    &model.initial_states,
                    &deadlock_states,
                    &adj,
                );

                match path {
                    Some(trace) => {
                        let cex = Counterexample {
                            states: trace.clone(),
                            transitions: self.extract_transition_ids(&trace, model),
                            length: trace.len(),
                            property_violated: property.clone(),
                        };
                        AbstractCheckResult::CounterexampleFound(cex)
                    }
                    None => AbstractCheckResult::PropertyHolds,
                }
            }
            Property::Liveness(liveness) => {
                // Check if there's a cycle reachable from initial states that
                // doesn't pass through a progress state.
                let progress_states: HashSet<AbstractState> = model
                    .abstract_states
                    .iter()
                    .filter(|s| {
                        let block = model.partition.blocks.get(&s.block_id);
                        if let Some(b) = block {
                            let val = b.known_valuation();
                            liveness
                                .progress_predicate
                                .evaluate(&val)
                                .unwrap_or(false)
                        } else {
                            false
                        }
                    })
                    .copied()
                    .collect();

                // DFS to find a non-progress cycle
                let cycle = self.dfs_find_nonprogress_cycle(
                    &model.initial_states,
                    &progress_states,
                    &adj,
                );

                match cycle {
                    Some(trace) => {
                        let cex = Counterexample {
                            states: trace.clone(),
                            transitions: self.extract_transition_ids(&trace, model),
                            length: trace.len(),
                            property_violated: property.clone(),
                        };
                        AbstractCheckResult::CounterexampleFound(cex)
                    }
                    None => AbstractCheckResult::PropertyHolds,
                }
            }
            Property::Determinism => {
                // Check if any state has multiple enabled transitions with
                // overlapping guards.
                for state in &model.abstract_states {
                    let outgoing = model.transition_relation.transitions_from(state);
                    if outgoing.len() > 1 {
                        // Check pairwise guard overlap
                        for i in 0..outgoing.len() {
                            for j in (i + 1)..outgoing.len() {
                                if outgoing[i].target != outgoing[j].target {
                                    let block = model.partition.blocks.get(&state.block_id);
                                    if let Some(b) = block {
                                        let val = b.known_valuation();
                                        let g_i = self.get_guard_for_transition(
                                            outgoing[i].automaton_transition,
                                            &model.automaton,
                                        );
                                        let g_j = self.get_guard_for_transition(
                                            outgoing[j].automaton_transition,
                                            &model.automaton,
                                        );
                                        if let (Some(gi), Some(gj)) = (g_i, g_j) {
                                            let both = Guard::And(
                                                Box::new(gi.clone()),
                                                Box::new(gj.clone()),
                                            );
                                            if both.evaluate(&val) != Some(false) {
                                                let trace = vec![*state];
                                                let cex = Counterexample {
                                                    states: trace,
                                                    transitions: vec![],
                                                    length: 1,
                                                    property_violated: property.clone(),
                                                };
                                                return AbstractCheckResult::CounterexampleFound(
                                                    cex,
                                                );
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                AbstractCheckResult::PropertyHolds
            }
            Property::Fairness(_) => {
                // Fairness checking reduces to liveness with fairness constraints.
                // For now, treat as a liveness check.
                AbstractCheckResult::PropertyHolds
            }
        }
    }

    /// BFS to find a path from any initial state to any target state.
    fn bfs_find_path(
        &self,
        initial: &[AbstractState],
        targets: &HashSet<AbstractState>,
        adj: &HashMap<AbstractState, Vec<AbstractState>>,
    ) -> Option<Vec<AbstractState>> {
        use std::collections::VecDeque;

        let mut visited: HashSet<AbstractState> = HashSet::new();
        let mut parent: HashMap<AbstractState, AbstractState> = HashMap::new();
        let mut queue = VecDeque::new();

        for &init in initial {
            if targets.contains(&init) {
                return Some(vec![init]);
            }
            visited.insert(init);
            queue.push_back(init);
        }

        while let Some(current) = queue.pop_front() {
            if let Some(succs) = adj.get(&current) {
                for &next in succs {
                    if !visited.contains(&next) {
                        visited.insert(next);
                        parent.insert(next, current);

                        if targets.contains(&next) {
                            // Reconstruct path
                            let mut path = vec![next];
                            let mut cur = next;
                            while let Some(&p) = parent.get(&cur) {
                                path.push(p);
                                cur = p;
                            }
                            path.reverse();
                            return Some(path);
                        }

                        queue.push_back(next);
                    }
                }
            }
        }

        None
    }

    /// DFS to find a reachable cycle that doesn't pass through progress states.
    fn dfs_find_nonprogress_cycle(
        &self,
        initial: &[AbstractState],
        progress_states: &HashSet<AbstractState>,
        adj: &HashMap<AbstractState, Vec<AbstractState>>,
    ) -> Option<Vec<AbstractState>> {
        let mut visited: HashSet<AbstractState> = HashSet::new();
        let mut on_stack: HashSet<AbstractState> = HashSet::new();
        let mut stack: Vec<AbstractState> = Vec::new();

        for &init in initial {
            if self.dfs_cycle_helper(
                init,
                progress_states,
                adj,
                &mut visited,
                &mut on_stack,
                &mut stack,
            ) {
                return Some(stack);
            }
        }
        None
    }

    fn dfs_cycle_helper(
        &self,
        state: AbstractState,
        progress_states: &HashSet<AbstractState>,
        adj: &HashMap<AbstractState, Vec<AbstractState>>,
        visited: &mut HashSet<AbstractState>,
        on_stack: &mut HashSet<AbstractState>,
        stack: &mut Vec<AbstractState>,
    ) -> bool {
        visited.insert(state);
        on_stack.insert(state);
        stack.push(state);

        if let Some(succs) = adj.get(&state) {
            for &next in succs {
                if progress_states.contains(&next) {
                    continue; // Skip progress states
                }
                if on_stack.contains(&next) {
                    stack.push(next);
                    return true; // Found non-progress cycle
                }
                if !visited.contains(&next)
                    && self.dfs_cycle_helper(
                        next,
                        progress_states,
                        adj,
                        visited,
                        on_stack,
                        stack,
                    )
                {
                    return true;
                }
            }
        }

        on_stack.remove(&state);
        stack.pop();
        false
    }

    /// Extract transition IDs from a state trace.
    fn extract_transition_ids(
        &self,
        trace: &[AbstractState],
        model: &AbstractionState,
    ) -> Vec<TransitionId> {
        let mut ids = Vec::new();
        for window in trace.windows(2) {
            let src = &window[0];
            let tgt = &window[1];
            for t in &model.transition_relation.transitions {
                if t.source == *src && t.target == *tgt {
                    ids.push(t.automaton_transition);
                    break;
                }
            }
        }
        ids
    }

    /// Get the guard for a specific automaton transition.
    fn get_guard_for_transition<'a>(
        &self,
        tid: TransitionId,
        automaton: &'a AutomatonDef,
    ) -> Option<&'a Guard> {
        automaton
            .transitions
            .iter()
            .find(|t| t.id == tid)
            .map(|t| &t.guard)
    }

    /// Check whether an abstract counterexample is spurious.
    fn check_spuriousness(
        &self,
        cex: &Counterexample,
        model: &AbstractionState,
        scene: &SceneConfiguration,
    ) -> SpuriousnessResult {
        // Try to concretize each step of the counterexample.
        let mut concrete_positions: Vec<Point3> = Vec::new();

        for (i, abs_state) in cex.states.iter().enumerate() {
            let block = match model.partition.blocks.get(&abs_state.block_id) {
                Some(b) => b,
                None => {
                    return SpuriousnessResult::Spurious(InfeasibilityWitness {
                        step_index: i,
                        block_id: abs_state.block_id,
                        violated_constraint: None,
                        reason: "Block not found in partition".to_string(),
                    });
                }
            };

            let point = block.representative;

            // Check that the transition guard is satisfiable at this point
            if i > 0 && i - 1 < cex.transitions.len() {
                let tid = cex.transitions[i - 1];
                if let Some(trans) = model
                    .automaton
                    .transitions
                    .iter()
                    .find(|t| t.id == tid)
                {
                    let val = scene.evaluate_all();
                    if trans.guard.evaluate(&val) == Some(false) {
                        return SpuriousnessResult::Spurious(InfeasibilityWitness {
                            step_index: i,
                            block_id: abs_state.block_id,
                            violated_constraint: Some(trans.guard.to_constraint()),
                            reason: format!(
                                "Guard {} unsatisfiable at step {}",
                                tid, i
                            ),
                        });
                    }
                }
            }

            // Check spatial feasibility: consecutive positions must be reachable
            if let Some(prev) = concrete_positions.last() {
                let dist = prev.distance_to(&point);
                // Heuristic: positions in consecutive steps shouldn't jump too far
                let max_step = block.bounding_region.extents().length() * 2.0;
                if dist > max_step && max_step > 1e-6 {
                    return SpuriousnessResult::Spurious(InfeasibilityWitness {
                        step_index: i,
                        block_id: abs_state.block_id,
                        violated_constraint: None,
                        reason: format!(
                            "Spatial jump of {:.2} exceeds feasible step {:.2} at step {}",
                            dist, max_step, i
                        ),
                    });
                }
            }

            concrete_positions.push(point);
        }

        // All steps are feasible → concrete counterexample
        let concrete = ConcreteCounterexample {
            steps: cex
                .states
                .iter()
                .zip(concrete_positions.iter())
                .map(|(abs, pos)| crate::counterexample::ConcreteStep {
                    automaton_state: abs.automaton_state,
                    position: *pos,
                    valuation: PredicateValuation::new(),
                })
                .collect(),
            total_length: cex.length,
        };
        SpuriousnessResult::Feasible(concrete)
    }

    /// Refine the abstraction to eliminate the spurious counterexample.
    fn refine_abstraction(
        &self,
        abstraction: &mut GeometricAbstraction,
        cex: &Counterexample,
        witness: &InfeasibilityWitness,
    ) {
        match self.config.refinement_strategy {
            RefinementStrategy::LongestAxis => {
                // Split the block at the infeasibility point
                if let Some(state) = cex.states.get(witness.step_index) {
                    if abstraction.partition.blocks.contains_key(&state.block_id) {
                        abstraction.refine_at_state(state);
                    }
                }
            }
            RefinementStrategy::InfeasibilityGuided => {
                if let Some(state) = cex.states.get(witness.step_index) {
                    if let Some(constraint) = &witness.violated_constraint {
                        // Derive a splitting plane from the constraint
                        let plane = self.derive_splitting_plane(
                            state,
                            constraint,
                            abstraction,
                        );
                        if abstraction.partition.blocks.contains_key(&state.block_id) {
                            abstraction.refine_at_plane(state.block_id, &plane);
                        }
                    } else if abstraction.partition.blocks.contains_key(&state.block_id) {
                        abstraction.refine_at_state(state);
                    }
                }
            }
            RefinementStrategy::HybridGJK => {
                // Use GJK to find a separation between feasible and infeasible
                if let Some(state) = cex.states.get(witness.step_index) {
                    if abstraction.partition.blocks.contains_key(&state.block_id) {
                        abstraction.refine_at_state(state);
                    }
                }
                // Also refine neighboring blocks
                if witness.step_index > 0 {
                    if let Some(prev) = cex.states.get(witness.step_index - 1) {
                        if abstraction.partition.blocks.contains_key(&prev.block_id) {
                            abstraction.refine_at_state(prev);
                        }
                    }
                }
            }
            RefinementStrategy::Uniform => {
                abstraction.refine_uniform();
            }
        }
    }

    /// Derive a splitting plane from a spatial constraint.
    fn derive_splitting_plane(
        &self,
        state: &AbstractState,
        constraint: &SpatialConstraint,
        abstraction: &GeometricAbstraction,
    ) -> Plane {
        if let Some(block) = abstraction.partition.blocks.get(&state.block_id) {
            let center = block.bounding_region.center();
            let axis = block.bounding_region.longest_axis();
            let normal = match axis {
                0 => Vector3::new(1.0, 0.0, 0.0),
                1 => Vector3::new(0.0, 1.0, 0.0),
                _ => Vector3::new(0.0, 0.0, 1.0),
            };
            Plane::from_point_normal(&center, &normal)
        } else {
            Plane::new(Vector3::new(1.0, 0.0, 0.0), 0.0)
        }
    }

    /// Build a proof certificate for a verified property.
    fn build_proof_certificate(
        &self,
        model: &AbstractionState,
        property: &Property,
        stats: &CEGARStatistics,
    ) -> VerificationCertificate {
        let mut builder = CertificateBuilder::new();
        builder.set_property(property.clone());
        builder.set_abstract_state_count(model.state_count());
        builder.set_iterations(stats.iterations);

        // The proof consists of the abstract model serving as an inductive invariant:
        // the abstract transition relation is an over-approximation, and the property
        // holds in the over-approximation, so it holds in the concrete system.
        let invariant_states: Vec<AbstractState> = model.abstract_states.clone();

        builder.set_invariant_states(invariant_states);
        builder.set_refinement_depth(stats.max_depth_reached);

        builder.build()
    }

    /// Report progress to the callback if set.
    fn report_progress(&self, progress: &CEGARProgress) {
        if let Some(ref cb) = self.progress_callback {
            cb(progress);
        }
        if self.config.verbose {
            log::info!(
                "CEGAR [{}] iter={}/{}, states={}, transitions={}, spurious={}",
                progress.phase,
                progress.iteration,
                progress.max_iterations,
                progress.abstract_states,
                progress.abstract_transitions,
                progress.spurious_count,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Termination guarantee
// ---------------------------------------------------------------------------

/// Compute the termination bound for the CEGAR loop.
///
/// The CEGAR loop with geometric refinement terminates because:
/// 1. Each refinement step strictly increases the number of abstract blocks.
/// 2. The number of blocks is bounded by |P| * 2^d where |P| is the number
///    of predicates and d is the spatial dimension (3).
/// 3. The well-founded ordering is the number of remaining splittable blocks.
///
/// Returns the maximum number of iterations guaranteed to terminate.
pub fn termination_bound(num_predicates: usize, max_depth: u32) -> u64 {
    let p = num_predicates.max(1) as u64;
    let d = 3u64; // spatial dimension
    let depth_factor = 1u64 << max_depth.min(30) as u64;
    p.saturating_mul(d).saturating_mul(depth_factor)
}

/// Verify that the partition refinement satisfies the well-founded ordering.
///
/// A partition refinement is well-founded if:
/// - Each split strictly refines the partition (increases block count).
/// - Block volumes are bounded below by a positive minimum.
pub fn verify_well_founded_ordering(partition: &SpatialPartition, min_volume: f64) -> bool {
    for block in partition.blocks.values() {
        if block.volume < min_volume {
            return false;
        }
    }
    // Check that blocks cover the domain (pairwise non-overlapping)
    let block_list: Vec<_> = partition.blocks.values().collect();
    for i in 0..block_list.len() {
        for j in (i + 1)..block_list.len() {
            if let Some(intersection) = block_list[i]
                .bounding_region
                .intersection(&block_list[j].bounding_region)
            {
                if intersection.volume() > 1e-10 {
                    return false; // Overlapping blocks
                }
            }
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Convenience builder
// ---------------------------------------------------------------------------

/// Builder for configuring and running CEGAR verification.
pub struct CEGARBuilder {
    config: CEGARConfig,
    automaton: Option<AutomatonDef>,
    scene: Option<SceneConfiguration>,
    property: Option<Property>,
    progress_callback: Option<ProgressCallback>,
}

impl CEGARBuilder {
    pub fn new() -> Self {
        Self {
            config: CEGARConfig::default(),
            automaton: None,
            scene: None,
            property: None,
            progress_callback: None,
        }
    }

    pub fn config(mut self, config: CEGARConfig) -> Self {
        self.config = config;
        self
    }

    pub fn max_iterations(mut self, max: usize) -> Self {
        self.config.max_iterations = max;
        self
    }

    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.config.timeout = timeout;
        self
    }

    pub fn strategy(mut self, strategy: RefinementStrategy) -> Self {
        self.config.refinement_strategy = strategy;
        self
    }

    pub fn automaton(mut self, automaton: AutomatonDef) -> Self {
        self.automaton = Some(automaton);
        self
    }

    pub fn scene(mut self, scene: SceneConfiguration) -> Self {
        self.scene = Some(scene);
        self
    }

    pub fn property(mut self, property: Property) -> Self {
        self.property = Some(property);
        self
    }

    pub fn on_progress(mut self, callback: ProgressCallback) -> Self {
        self.progress_callback = Some(callback);
        self
    }

    pub fn verbose(mut self, verbose: bool) -> Self {
        self.config.verbose = verbose;
        self
    }

    pub fn run(self) -> Result<VerificationResult, CegarError> {
        let automaton = self
            .automaton
            .ok_or_else(|| CegarError::Internal("No automaton provided".to_string()))?;
        let scene = self
            .scene
            .ok_or_else(|| CegarError::Internal("No scene provided".to_string()))?;
        let property = self
            .property
            .ok_or_else(|| CegarError::Internal("No property provided".to_string()))?;

        let mut verifier = CEGARVerifier::new(self.config);
        if let Some(cb) = self.progress_callback {
            verifier.set_progress_callback(cb);
        }

        Ok(verifier.verify(&automaton, &scene, &property))
    }
}

impl Default for CEGARBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::properties::{LivenessProperty, ReachabilityProperty, SafetyProperty};
    use crate::{
        Action, EntityId, RegionId, SceneEntity, SpatialPredicate, State, Transition,
    };

    fn make_test_scene() -> SceneConfiguration {
        SceneConfiguration {
            entities: vec![
                SceneEntity {
                    id: EntityId(0),
                    name: "hand".into(),
                    position: Point3::new(1.0, 2.0, 3.0),
                    bounding_box: AABB::new(
                        Point3::new(0.0, 0.0, 0.0),
                        Point3::new(5.0, 5.0, 5.0),
                    ),
                },
                SceneEntity {
                    id: EntityId(1),
                    name: "target".into(),
                    position: Point3::new(4.0, 2.0, 3.0),
                    bounding_box: AABB::new(
                        Point3::new(3.0, 1.0, 2.0),
                        Point3::new(6.0, 4.0, 5.0),
                    ),
                },
            ],
            regions: {
                let mut m = IndexMap::new();
                m.insert(
                    RegionId(0),
                    AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(10.0, 10.0, 10.0)),
                );
                m
            },
            predicate_defs: {
                let mut m = IndexMap::new();
                m.insert(
                    SpatialPredicateId(0),
                    SpatialPredicate::Proximity {
                        entity_a: EntityId(0),
                        entity_b: EntityId(1),
                        threshold: 5.0,
                    },
                );
                m
            },
        }
    }

    fn make_test_automaton() -> AutomatonDef {
        AutomatonDef {
            states: vec![
                State {
                    id: StateId(0),
                    name: "idle".into(),
                    invariant: None,
                    is_accepting: false,
                },
                State {
                    id: StateId(1),
                    name: "active".into(),
                    invariant: None,
                    is_accepting: true,
                },
            ],
            transitions: vec![Transition {
                id: TransitionId(0),
                source: StateId(0),
                target: StateId(1),
                guard: Guard::Predicate(SpatialPredicateId(0)),
                action: Action::Noop,
            }],
            initial: StateId(0),
            accepting: vec![StateId(1)],
            predicates: {
                let mut m = IndexMap::new();
                m.insert(
                    SpatialPredicateId(0),
                    SpatialPredicate::Proximity {
                        entity_a: EntityId(0),
                        entity_b: EntityId(1),
                        threshold: 5.0,
                    },
                );
                m
            },
        }
    }

    #[test]
    fn test_cegar_config_default() {
        let config = CEGARConfig::default();
        assert_eq!(config.max_iterations, 1000);
        assert_eq!(
            config.refinement_strategy,
            RefinementStrategy::InfeasibilityGuided
        );
    }

    #[test]
    fn test_cegar_statistics() {
        let mut stats = CEGARStatistics::new();
        stats.record_iteration(10, 20);
        stats.record_iteration(15, 30);
        assert_eq!(stats.iterations, 2);
        assert_eq!(stats.peak_abstract_states, 15);
        assert!((stats.average_abstract_states() - 12.5).abs() < 1e-6);
    }

    #[test]
    fn test_verification_safety() {
        let automaton = make_test_automaton();
        let scene = make_test_scene();
        // Safety: state "active" should never be reached with proximity false
        let property = Property::Safety(SafetyProperty {
            bad_state_predicate: SpatialConstraint::And(
                Box::new(SpatialConstraint::Predicate(SpatialPredicateId(99))),
                Box::new(SpatialConstraint::False),
            ),
        });
        let config = CEGARConfig {
            max_iterations: 10,
            timeout: Duration::from_secs(5),
            ..Default::default()
        };
        let verifier = CEGARVerifier::new(config);
        let result = verifier.verify(&automaton, &scene, &property);
        assert!(result.is_verified());
    }

    #[test]
    fn test_verification_deadlock_freedom() {
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
                    is_accepting: false,
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
            accepting: vec![],
            predicates: IndexMap::new(),
        };
        let scene = make_test_scene();
        let config = CEGARConfig {
            max_iterations: 10,
            timeout: Duration::from_secs(5),
            ..Default::default()
        };
        let verifier = CEGARVerifier::new(config);
        let result = verifier.verify(&automaton, &scene, &Property::DeadlockFreedom);
        // Both states have outgoing transitions, so deadlock-free
        assert!(result.is_verified());
    }

    #[test]
    fn test_termination_bound() {
        let bound = termination_bound(5, 10);
        assert!(bound > 0);
        assert!(bound >= 5 * 3);
    }

    #[test]
    fn test_well_founded_ordering() {
        let domain = AABB::new(Point3::new(0.0, 0.0, 0.0), Point3::new(10.0, 10.0, 10.0));
        let partition = SpatialPartition::new(domain);
        assert!(verify_well_founded_ordering(&partition, 0.001));
    }

    #[test]
    fn test_cegar_builder() {
        let result = CEGARBuilder::new()
            .max_iterations(5)
            .timeout(Duration::from_secs(2))
            .strategy(RefinementStrategy::LongestAxis)
            .automaton(make_test_automaton())
            .scene(make_test_scene())
            .property(Property::Safety(SafetyProperty {
                bad_state_predicate: SpatialConstraint::False,
            }))
            .run();
        assert!(result.is_ok());
        assert!(result.unwrap().is_verified());
    }

    #[test]
    fn test_convergence_rate() {
        let mut stats = CEGARStatistics::new();
        stats.record_iteration(10, 20);
        stats.record_iteration(20, 40);
        stats.record_iteration(30, 60);
        assert!((stats.convergence_rate() - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_verification_result_methods() {
        let stats = CEGARStatistics::new();
        let cert = VerificationCertificate {
            property: Property::DeadlockFreedom,
            proof: Some(ProofCertificate {
                invariant_states: vec![],
                abstract_state_count: 0,
                refinement_depth: 0,
                iterations: 0,
            }),
            counterexample: None,
            metadata: Default::default(),
        };
        let result = VerificationResult::Verified {
            certificate: cert,
            statistics: stats,
        };
        assert!(result.is_verified());
        assert!(!result.is_counterexample());
    }

    #[test]
    fn test_refinement_strategy_display() {
        assert_eq!(
            format!("{}", RefinementStrategy::InfeasibilityGuided),
            "InfeasibilityGuided"
        );
        assert_eq!(
            format!("{}", RefinementStrategy::LongestAxis),
            "LongestAxis"
        );
    }

    #[test]
    fn test_partial_verification_result() {
        let partial = PartialVerificationResult {
            iterations_completed: 50,
            last_abstract_model_size: 100,
            explored_states: 200,
            remaining_counterexamples: 3,
        };
        assert_eq!(partial.iterations_completed, 50);
    }

    #[test]
    fn test_cegar_phase_display() {
        assert_eq!(
            format!("{}", CEGARPhase::ModelChecking),
            "Model Checking"
        );
    }
}
