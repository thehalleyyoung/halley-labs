//! # Lattice-Walk Contract Synthesis (Algorithm A2)
//!
//! Implements the novel lattice-walk synthesis algorithm for Tier 1 contract
//! generation. The algorithm walks the specification lattice from Top (the
//! weakest specification, `True`) towards Bottom, progressively strengthening
//! the contract by conjoining negated error predicates from killed mutants.
//!
//! ## Algorithm Overview
//!
//! Given a set of killed mutants M_kill with error predicates {E(m) | m ∈ M_kill}:
//!
//! 1. **Initialise** the current specification to Top (⊤ = True).
//! 2. **Order** the mutants using a dominator-based heuristic so that the most
//!    discriminating (i.e. most informative) error predicates are processed first.
//! 3. **Walk**: for each mutant m in order, compute the candidate spec
//!    `candidate = current ∧ ¬E(m)`.  If the candidate is consistent (not ⊥)
//!    and does not exceed the complexity budget, accept it as the new current
//!    specification.  Otherwise, skip the mutant or attempt simplification.
//! 4. **Post-process**: simplify the final specification and record provenance
//!    (which mutants contributed to which clauses).
//!
//! ## Dominator Ordering
//!
//! The walk order is determined by a priority score for each mutant.  Mutants
//! whose error predicates have *fewer free variables* and *smaller AST size*
//! are processed first because their negations impose tighter, more general
//! constraints.  This corresponds to processing dominator nodes before
//! dominated nodes in the CFG interpretation of the specification lattice.
//!
//! ## Soundness
//!
//! Every accepted step preserves soundness: the candidate ¬E(m) was derived
//! from the weakest-precondition computation over the mutated program, so the
//! conjunction of accepted negated error predicates is a valid postcondition
//! of the original program.

use std::collections::{BTreeSet, HashMap};
use std::fmt;
use std::time::Instant;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};

use shared_types::{
    Contract, ContractClause, ContractProvenance, ContractStrength, Formula, MutantId, Predicate,
    Relation, SynthesisTier, Term, MutationOperator,
};

use crate::lattice::{DiscriminationLattice, EntailmentResult, LatticeElement, SpecLattice};

// ---------------------------------------------------------------------------
// WalkConfig
// ---------------------------------------------------------------------------

/// Configuration parameters for the lattice-walk synthesizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalkConfig {
    /// Maximum AST node count for the synthesized formula.
    /// Steps that would push the formula beyond this limit are skipped.
    pub max_formula_size: usize,

    /// Maximum nesting depth for the synthesized formula.
    pub max_formula_depth: usize,

    /// Maximum number of walk steps before the algorithm halts.
    /// This prevents runaway synthesis on programs with many mutants.
    pub max_steps: usize,

    /// Maximum wall-clock time (milliseconds) for the entire walk.
    pub timeout_ms: u64,

    /// If true, apply syntactic simplification after every accepted step.
    pub simplify_eagerly: bool,

    /// Minimum improvement in mutant coverage required to accept a step.
    /// A value of 0 means any new coverage is accepted.
    pub min_coverage_gain: usize,

    /// If true, attempt to decompose large conjunctions into separate
    /// contract clauses (one per error predicate) rather than a single
    /// monolithic ensures clause.
    pub decompose_clauses: bool,

    /// Weight for formula size in the ordering heuristic (higher = prefer
    /// smaller error predicates first).
    pub size_weight: f64,

    /// Weight for free-variable count in the ordering heuristic.
    pub var_weight: f64,

    /// Weight for formula depth in the ordering heuristic.
    pub depth_weight: f64,

    /// Whether to use the subsumption check: skip mutants whose negated
    /// error predicate is already entailed by the current specification.
    pub enable_subsumption: bool,

    /// Whether to attempt recovery when a step produces an inconsistent
    /// (Bottom) result by trying subsets of conjuncts.
    pub enable_recovery: bool,
}

impl Default for WalkConfig {
    fn default() -> Self {
        Self {
            max_formula_size: 200,
            max_formula_depth: 15,
            max_steps: 1000,
            timeout_ms: 30_000,
            simplify_eagerly: true,
            min_coverage_gain: 0,
            decompose_clauses: true,
            size_weight: 1.0,
            var_weight: 2.0,
            depth_weight: 1.5,
            enable_subsumption: true,
            enable_recovery: false,
        }
    }
}

impl WalkConfig {
    /// Create a config tuned for fast synthesis (smaller limits).
    pub fn fast() -> Self {
        Self {
            max_formula_size: 80,
            max_formula_depth: 8,
            max_steps: 200,
            timeout_ms: 5_000,
            simplify_eagerly: true,
            min_coverage_gain: 0,
            decompose_clauses: true,
            size_weight: 1.0,
            var_weight: 2.0,
            depth_weight: 1.5,
            enable_subsumption: true,
            enable_recovery: false,
        }
    }

    /// Create a config tuned for thorough synthesis (larger limits).
    pub fn thorough() -> Self {
        Self {
            max_formula_size: 500,
            max_formula_depth: 25,
            max_steps: 5000,
            timeout_ms: 120_000,
            simplify_eagerly: true,
            min_coverage_gain: 0,
            decompose_clauses: true,
            size_weight: 1.0,
            var_weight: 2.0,
            depth_weight: 1.5,
            enable_subsumption: true,
            enable_recovery: true,
        }
    }
}

// ---------------------------------------------------------------------------
// WalkStep
// ---------------------------------------------------------------------------

/// Outcome of a single lattice-walk step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WalkStep {
    /// The step was accepted: the negated error predicate was conjoined.
    Accepted {
        /// Mutant whose error predicate was negated and added.
        mutant_id: MutantId,
        /// New formula size after the step.
        new_size: usize,
        /// Number of mutants now covered by the specification.
        coverage: usize,
    },

    /// The step was skipped because the candidate exceeded complexity limits.
    SkippedComplexity {
        mutant_id: MutantId,
        /// Size that would have resulted.
        would_be_size: usize,
    },

    /// The step was skipped because the mutant was already subsumed.
    SkippedSubsumed {
        mutant_id: MutantId,
    },

    /// The step was skipped because it would make the spec inconsistent (⊥).
    SkippedInconsistent {
        mutant_id: MutantId,
    },

    /// The step was skipped because the timeout was reached.
    SkippedTimeout {
        mutant_id: MutantId,
    },

    /// The step was skipped because the coverage gain was insufficient.
    SkippedInsufficientGain {
        mutant_id: MutantId,
        gain: usize,
    },
}

impl WalkStep {
    /// Returns the mutant ID involved in this step.
    pub fn mutant_id(&self) -> &MutantId {
        match self {
            WalkStep::Accepted { mutant_id, .. }
            | WalkStep::SkippedComplexity { mutant_id, .. }
            | WalkStep::SkippedSubsumed { mutant_id }
            | WalkStep::SkippedInconsistent { mutant_id }
            | WalkStep::SkippedTimeout { mutant_id }
            | WalkStep::SkippedInsufficientGain { mutant_id, .. } => mutant_id,
        }
    }

    /// Returns `true` if this step was accepted (contributed to the spec).
    pub fn is_accepted(&self) -> bool {
        matches!(self, WalkStep::Accepted { .. })
    }

    /// Returns `true` if this step was skipped for any reason.
    pub fn is_skipped(&self) -> bool {
        !self.is_accepted()
    }

    /// Short human-readable description of this step.
    pub fn description(&self) -> String {
        match self {
            WalkStep::Accepted { mutant_id, new_size, coverage } => {
                format!(
                    "accepted ¬E({}) → size={}, coverage={}",
                    mutant_id.short(),
                    new_size,
                    coverage
                )
            }
            WalkStep::SkippedComplexity { mutant_id, would_be_size } => {
                format!(
                    "skipped {} (complexity: would be {} nodes)",
                    mutant_id.short(),
                    would_be_size
                )
            }
            WalkStep::SkippedSubsumed { mutant_id } => {
                format!("skipped {} (already subsumed)", mutant_id.short())
            }
            WalkStep::SkippedInconsistent { mutant_id } => {
                format!("skipped {} (would be inconsistent)", mutant_id.short())
            }
            WalkStep::SkippedTimeout { mutant_id } => {
                format!("skipped {} (timeout)", mutant_id.short())
            }
            WalkStep::SkippedInsufficientGain { mutant_id, gain } => {
                format!(
                    "skipped {} (gain {} < threshold)",
                    mutant_id.short(),
                    gain
                )
            }
        }
    }
}

impl fmt::Display for WalkStep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.description())
    }
}

// ---------------------------------------------------------------------------
// WalkState
// ---------------------------------------------------------------------------

/// Snapshot of the synthesizer's state at any point during the walk.
///
/// The walk state captures the current lattice element (accumulated spec),
/// step history, and coverage information. It can be serialized for
/// debugging or checkpoint/restore.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalkState {
    /// The current accumulated specification (lattice element).
    current_element: LatticeElement,

    /// Steps taken so far (both accepted and skipped).
    steps: Vec<WalkStep>,

    /// Mutants that have been processed (whether accepted or skipped).
    processed: BTreeSet<MutantId>,

    /// Mutants whose negated error predicates are part of the current spec.
    accepted_mutants: BTreeSet<MutantId>,

    /// Current coverage: number of mutants killed by the current spec.
    coverage: usize,

    /// Total mutants available for processing.
    total_mutants: usize,

    /// Whether the walk has completed (all mutants processed or halted).
    completed: bool,

    /// If the walk was halted, the reason.
    halt_reason: Option<HaltReason>,
}

/// Reason the walk was halted before processing all mutants.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HaltReason {
    /// Reached the maximum number of steps.
    MaxSteps,
    /// Exceeded the wall-clock timeout.
    Timeout,
    /// Reached maximum formula complexity.
    ComplexityBound,
    /// All remaining mutants are subsumed.
    AllSubsumed,
    /// User-requested halt.
    UserHalt,
}

impl fmt::Display for HaltReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HaltReason::MaxSteps => write!(f, "max steps reached"),
            HaltReason::Timeout => write!(f, "timeout"),
            HaltReason::ComplexityBound => write!(f, "complexity bound"),
            HaltReason::AllSubsumed => write!(f, "all subsumed"),
            HaltReason::UserHalt => write!(f, "user halt"),
        }
    }
}

impl WalkState {
    /// Create the initial walk state (starting at Top).
    fn initial(total_mutants: usize) -> Self {
        Self {
            current_element: LatticeElement::top(),
            steps: Vec::new(),
            processed: BTreeSet::new(),
            accepted_mutants: BTreeSet::new(),
            coverage: 0,
            total_mutants,
            completed: false,
            halt_reason: None,
        }
    }

    /// The current specification as a lattice element.
    pub fn current_element(&self) -> &LatticeElement {
        &self.current_element
    }

    /// The current specification formula.
    pub fn current_formula(&self) -> &Formula {
        self.current_element.formula()
    }

    /// All steps taken so far.
    pub fn steps(&self) -> &[WalkStep] {
        &self.steps
    }

    /// Number of steps taken.
    pub fn step_count(&self) -> usize {
        self.steps.len()
    }

    /// Number of accepted steps.
    pub fn accepted_count(&self) -> usize {
        self.steps.iter().filter(|s| s.is_accepted()).count()
    }

    /// Number of skipped steps.
    pub fn skipped_count(&self) -> usize {
        self.steps.iter().filter(|s| s.is_skipped()).count()
    }

    /// Mutant IDs that contributed to the current specification.
    pub fn accepted_mutants(&self) -> &BTreeSet<MutantId> {
        &self.accepted_mutants
    }

    /// Current mutant coverage count.
    pub fn coverage(&self) -> usize {
        self.coverage
    }

    /// Coverage as a fraction of total mutants.
    pub fn coverage_ratio(&self) -> f64 {
        if self.total_mutants == 0 {
            0.0
        } else {
            self.coverage as f64 / self.total_mutants as f64
        }
    }

    /// Whether the walk has completed.
    pub fn is_completed(&self) -> bool {
        self.completed
    }

    /// The halt reason, if the walk was halted early.
    pub fn halt_reason(&self) -> Option<&HaltReason> {
        self.halt_reason.as_ref()
    }
}

impl fmt::Display for WalkState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "WalkState: {} steps ({} accepted, {} skipped), coverage {}/{} ({:.1}%)",
            self.step_count(),
            self.accepted_count(),
            self.skipped_count(),
            self.coverage,
            self.total_mutants,
            self.coverage_ratio() * 100.0,
        )?;
        if let Some(reason) = &self.halt_reason {
            write!(f, " [halted: {}]", reason)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// WalkStatistics
// ---------------------------------------------------------------------------

/// Aggregate statistics from a completed lattice walk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalkStatistics {
    /// Total wall-clock time for the synthesis (milliseconds).
    pub total_time_ms: f64,

    /// Number of walk steps attempted.
    pub steps_attempted: usize,

    /// Number of walk steps accepted.
    pub steps_accepted: usize,

    /// Number of walk steps skipped due to complexity.
    pub steps_skipped_complexity: usize,

    /// Number of walk steps skipped due to subsumption.
    pub steps_skipped_subsumption: usize,

    /// Number of walk steps skipped due to inconsistency.
    pub steps_skipped_inconsistency: usize,

    /// Number of walk steps skipped due to timeout.
    pub steps_skipped_timeout: usize,

    /// Number of walk steps skipped due to insufficient gain.
    pub steps_skipped_insufficient_gain: usize,

    /// Final formula size (AST node count).
    pub final_formula_size: usize,

    /// Final formula depth.
    pub final_formula_depth: usize,

    /// Number of mutants covered by the final specification.
    pub mutants_covered: usize,

    /// Total number of killed mutants available.
    pub mutants_total: usize,

    /// Number of entailment checks performed.
    pub entailment_checks: u64,

    /// Number of simplification passes applied.
    pub simplification_passes: usize,

    /// Whether the walk completed normally or was halted.
    pub halt_reason: Option<HaltReason>,

    /// Number of contract clauses in the final contract.
    pub clause_count: usize,

    /// Coverage ratio as a percentage.
    pub coverage_percent: f64,

    /// The synthesis tier used.
    pub tier: SynthesisTier,
}

impl WalkStatistics {
    /// Compute the acceptance rate (fraction of steps accepted).
    pub fn acceptance_rate(&self) -> f64 {
        if self.steps_attempted == 0 {
            0.0
        } else {
            self.steps_accepted as f64 / self.steps_attempted as f64
        }
    }

    /// Compute the contract strength from coverage.
    pub fn contract_strength(&self) -> ContractStrength {
        if self.mutants_total == 0 {
            return ContractStrength::Trivial;
        }
        ContractStrength::from_kill_ratio(
            self.mutants_covered as f64 / self.mutants_total as f64,
        )
    }
}

impl fmt::Display for WalkStatistics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Lattice Walk Statistics:")?;
        writeln!(f, "  Time:        {:.1} ms", self.total_time_ms)?;
        writeln!(
            f,
            "  Steps:       {} attempted, {} accepted ({:.1}% acceptance)",
            self.steps_attempted,
            self.steps_accepted,
            self.acceptance_rate() * 100.0,
        )?;
        writeln!(
            f,
            "  Coverage:    {}/{} ({:.1}%)",
            self.mutants_covered, self.mutants_total, self.coverage_percent,
        )?;
        writeln!(
            f,
            "  Formula:     {} nodes, depth {}",
            self.final_formula_size, self.final_formula_depth,
        )?;
        writeln!(f, "  Clauses:     {}", self.clause_count)?;
        if let Some(reason) = &self.halt_reason {
            writeln!(f, "  Halted:      {}", reason)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// MutantPriority (internal)
// ---------------------------------------------------------------------------

/// Internal struct for ordering mutants by their priority score.
#[derive(Debug, Clone)]
struct MutantPriority {
    id: MutantId,
    /// Lower score = higher priority (processed first).
    score: f64,
    /// Error predicate AST size.
    ep_size: usize,
    /// Number of free variables in the error predicate.
    ep_vars: usize,
    /// Error predicate depth.
    ep_depth: usize,
}

impl MutantPriority {
    fn compute(id: MutantId, error_predicate: &Formula, config: &WalkConfig) -> Self {
        let ep_size = error_predicate.size();
        let ep_vars = error_predicate.free_vars().len();
        let ep_depth = error_predicate.depth();
        let score = config.size_weight * ep_size as f64
            + config.var_weight * ep_vars as f64
            + config.depth_weight * ep_depth as f64;
        Self {
            id,
            score,
            ep_size,
            ep_vars,
            ep_depth,
        }
    }
}

// ---------------------------------------------------------------------------
// LatticeWalkSynthesizer
// ---------------------------------------------------------------------------

/// The Tier 1 lattice-walk contract synthesizer (Algorithm A2).
///
/// Walks the specification lattice from Top towards Bottom by iteratively
/// conjoining negated error predicates from killed mutants, ordered by a
/// dominator-based heuristic. Produces the strongest sound postcondition
/// that can be efficiently synthesized within the configured complexity
/// and time budgets.
///
/// # Example
///
/// ```ignore
/// use contract_synth::lattice_walk::{LatticeWalkSynthesizer, WalkConfig};
/// use contract_synth::lattice::DiscriminationLattice;
///
/// let mut disc = DiscriminationLattice::new();
/// // ... register mutants and their error predicates ...
///
/// let config = WalkConfig::default();
/// let mut synth = LatticeWalkSynthesizer::new(config);
/// let contract = synth.synthesize(&mut disc, "my_function");
/// ```
pub struct LatticeWalkSynthesizer {
    /// Configuration for the walk.
    config: WalkConfig,

    /// The walk state (populated during synthesis).
    state: Option<WalkState>,

    /// Statistics from the most recent synthesis run.
    statistics: Option<WalkStatistics>,

    /// Counter for simplification passes applied.
    simplification_count: usize,
}

impl LatticeWalkSynthesizer {
    /// Create a new synthesizer with the given configuration.
    pub fn new(config: WalkConfig) -> Self {
        Self {
            config,
            state: None,
            statistics: None,
            simplification_count: 0,
        }
    }

    /// Create a synthesizer with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(WalkConfig::default())
    }

    /// Access the current configuration.
    pub fn config(&self) -> &WalkConfig {
        &self.config
    }

    /// Mutable access to the configuration (before synthesis).
    pub fn config_mut(&mut self) -> &mut WalkConfig {
        &mut self.config
    }

    /// Access the walk state (available after synthesis starts).
    pub fn state(&self) -> Option<&WalkState> {
        self.state.as_ref()
    }

    /// Access the statistics (available after synthesis completes).
    pub fn statistics(&self) -> Option<&WalkStatistics> {
        self.statistics.as_ref()
    }

    // -- main synthesis entry point -----------------------------------------

    /// Run the lattice-walk synthesis algorithm.
    ///
    /// Produces a [`Contract`] for the given function, using the error
    /// predicates registered in the [`DiscriminationLattice`].
    pub fn synthesize(
        &mut self,
        disc: &mut DiscriminationLattice,
        function_name: &str,
    ) -> Contract {
        let start = Instant::now();
        let mutant_ids = disc.mutant_ids();
        let total_mutants = mutant_ids.len();

        // Initialise state.
        let mut walk_state = WalkState::initial(total_mutants);

        if total_mutants == 0 {
            walk_state.completed = true;
            self.state = Some(walk_state);
            self.statistics = Some(self.build_statistics(&walk_state, start.elapsed().as_secs_f64() * 1000.0, 0));
            return Contract::new(function_name.to_string());
        }

        // 1. Compute ordering.
        let ordered = self.compute_walk_order(disc, &mutant_ids);

        // 2. Walk the lattice.
        let mut entailment_checks: u64 = 0;
        let deadline_ms = self.config.timeout_ms;

        for priority in &ordered {
            // Check termination conditions.
            let elapsed_ms = start.elapsed().as_millis() as u64;
            if elapsed_ms >= deadline_ms {
                walk_state.halt_reason = Some(HaltReason::Timeout);
                walk_state.completed = true;
                break;
            }

            if walk_state.steps.len() >= self.config.max_steps {
                walk_state.halt_reason = Some(HaltReason::MaxSteps);
                walk_state.completed = true;
                break;
            }

            if walk_state.current_element.size_hint() >= self.config.max_formula_size {
                walk_state.halt_reason = Some(HaltReason::ComplexityBound);
                walk_state.completed = true;
                break;
            }

            let step = self.walk_step(disc, &mut walk_state, &priority.id, &mut entailment_checks);
            walk_state.steps.push(step);
            walk_state.processed.insert(priority.id.clone());
        }

        if walk_state.halt_reason.is_none() {
            walk_state.completed = true;
        }

        // 3. Build the contract.
        let contract = self.build_contract(disc, &walk_state, function_name);

        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        self.statistics = Some(self.build_statistics(&walk_state, elapsed_ms, entailment_checks));
        self.state = Some(walk_state);

        contract
    }

    // -- walk step ----------------------------------------------------------

    /// Execute a single walk step: attempt to conjoin ¬E(m) to the current spec.
    fn walk_step(
        &mut self,
        disc: &mut DiscriminationLattice,
        state: &mut WalkState,
        mutant_id: &MutantId,
        entailment_checks: &mut u64,
    ) -> WalkStep {
        // Compute ¬E(m) for this mutant.
        let neg_ep = disc.sigma_single(mutant_id);

        // Subsumption check: is ¬E(m) already entailed by the current spec?
        if self.config.enable_subsumption {
            let result = disc.entails(&state.current_element, &neg_ep);
            *entailment_checks += 1;
            if result.is_entailed() {
                return WalkStep::SkippedSubsumed {
                    mutant_id: mutant_id.clone(),
                };
            }
        }

        // Compute candidate = current ∧ ¬E(m).
        let candidate = state.current_element.meet(&neg_ep);

        // Check for inconsistency.
        if candidate.is_bottom() {
            return WalkStep::SkippedInconsistent {
                mutant_id: mutant_id.clone(),
            };
        }

        // Simplify if configured.
        let candidate = if self.config.simplify_eagerly {
            self.simplification_count += 1;
            candidate.simplify()
        } else {
            candidate
        };

        // Check complexity bounds.
        let candidate_size = candidate.size_hint();
        if candidate_size > self.config.max_formula_size {
            return WalkStep::SkippedComplexity {
                mutant_id: mutant_id.clone(),
                would_be_size: candidate_size,
            };
        }

        if candidate.depth_hint() > self.config.max_formula_depth {
            return WalkStep::SkippedComplexity {
                mutant_id: mutant_id.clone(),
                would_be_size: candidate_size,
            };
        }

        // Check minimum coverage gain.
        if self.config.min_coverage_gain > 0 {
            let current_covered = disc.covered_mutants(&state.current_element);
            let candidate_covered = disc.covered_mutants(&candidate);
            let gain = candidate_covered.len().saturating_sub(current_covered.len());
            if gain < self.config.min_coverage_gain {
                return WalkStep::SkippedInsufficientGain {
                    mutant_id: mutant_id.clone(),
                    gain,
                };
            }
        }

        // Accept the step.
        let coverage = disc.covered_mutants(&candidate).len();
        state.current_element = candidate;
        state.accepted_mutants.insert(mutant_id.clone());
        state.coverage = coverage;

        WalkStep::Accepted {
            mutant_id: mutant_id.clone(),
            new_size: candidate_size,
            coverage,
        }
    }

    // -- ordering -----------------------------------------------------------

    /// Compute the dominator-based walk order for mutants.
    ///
    /// Mutants are ordered by increasing priority score (smaller formulas
    /// and fewer variables go first).
    fn compute_walk_order(
        &self,
        disc: &DiscriminationLattice,
        mutant_ids: &[MutantId],
    ) -> Vec<MutantPriority> {
        let mut priorities: Vec<MutantPriority> = mutant_ids
            .iter()
            .filter_map(|id| {
                disc.error_predicate(id).map(|ep| {
                    MutantPriority::compute(id.clone(), ep, &self.config)
                })
            })
            .collect();

        // Sort by score (ascending = highest priority first).
        priorities.sort_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        priorities
    }

    // -- contract building --------------------------------------------------

    /// Build a [`Contract`] from the completed walk state.
    fn build_contract(
        &self,
        disc: &mut DiscriminationLattice,
        state: &WalkState,
        function_name: &str,
    ) -> Contract {
        let clauses = if self.config.decompose_clauses {
            self.decompose_into_clauses(&state.current_element)
        } else {
            vec![state.current_element.to_ensures_clause()]
        };

        let targeted: Vec<MutantId> = state.accepted_mutants.iter().cloned().collect();
        let prov = ContractProvenance {
            targeted_mutants: targeted,
            tier: SynthesisTier::Tier1LatticeWalk,
            solver_queries: 0,
            synthesis_time_ms: 0.0,
        };

        let strength = if state.total_mutants == 0 {
            ContractStrength::Trivial
        } else {
            ContractStrength::from_kill_ratio(state.coverage_ratio())
        };

        Contract {
            function_name: function_name.to_string(),
            clauses,
            provenance: vec![prov],
            strength,
            verified: false,
        }
    }

    /// Decompose a conjunction into separate ensures clauses.
    ///
    /// If the current formula is `A ∧ B ∧ C`, produce three separate
    /// `Ensures` clauses rather than one monolithic one.
    fn decompose_into_clauses(&self, element: &LatticeElement) -> Vec<ContractClause> {
        match element.formula() {
            Formula::And(conjuncts) => conjuncts
                .iter()
                .map(|c| ContractClause::Ensures(c.clone()))
                .collect(),
            Formula::True => Vec::new(),
            other => vec![ContractClause::Ensures(other.clone())],
        }
    }

    // -- statistics ---------------------------------------------------------

    /// Build statistics from the walk state.
    fn build_statistics(
        &self,
        state: &WalkState,
        total_time_ms: f64,
        entailment_checks: u64,
    ) -> WalkStatistics {
        let mut skipped_complexity = 0;
        let mut skipped_subsumption = 0;
        let mut skipped_inconsistency = 0;
        let mut skipped_timeout = 0;
        let mut skipped_insufficient_gain = 0;

        for step in &state.steps {
            match step {
                WalkStep::SkippedComplexity { .. } => skipped_complexity += 1,
                WalkStep::SkippedSubsumed { .. } => skipped_subsumption += 1,
                WalkStep::SkippedInconsistent { .. } => skipped_inconsistency += 1,
                WalkStep::SkippedTimeout { .. } => skipped_timeout += 1,
                WalkStep::SkippedInsufficientGain { .. } => skipped_insufficient_gain += 1,
                WalkStep::Accepted { .. } => {}
            }
        }

        let final_size = state.current_element.size_hint();
        let final_depth = state.current_element.depth_hint();
        let clause_count = match state.current_element.formula() {
            Formula::And(conjuncts) => conjuncts.len(),
            Formula::True => 0,
            _ => 1,
        };

        WalkStatistics {
            total_time_ms,
            steps_attempted: state.steps.len(),
            steps_accepted: state.accepted_count(),
            steps_skipped_complexity: skipped_complexity,
            steps_skipped_subsumption: skipped_subsumption,
            steps_skipped_inconsistency: skipped_inconsistency,
            steps_skipped_timeout: skipped_timeout,
            steps_skipped_insufficient_gain: skipped_insufficient_gain,
            final_formula_size: final_size,
            final_formula_depth: final_depth,
            mutants_covered: state.coverage,
            mutants_total: state.total_mutants,
            entailment_checks,
            simplification_passes: self.simplification_count,
            halt_reason: state.halt_reason.clone(),
            clause_count,
            coverage_percent: state.coverage_ratio() * 100.0,
            tier: SynthesisTier::Tier1LatticeWalk,
        }
    }

    // -- incremental API ----------------------------------------------------

    /// Resume a previously paused walk from a saved state.
    ///
    /// This allows the synthesizer to be used incrementally: run for a while,
    /// save the state, then resume later with more time or updated configuration.
    pub fn resume(
        &mut self,
        disc: &mut DiscriminationLattice,
        saved_state: WalkState,
        function_name: &str,
    ) -> Contract {
        let start = Instant::now();
        let mutant_ids = disc.mutant_ids();
        let mut walk_state = saved_state;
        walk_state.completed = false;
        walk_state.halt_reason = None;

        let ordered = self.compute_walk_order(disc, &mutant_ids);
        let mut entailment_checks: u64 = 0;
        let deadline_ms = self.config.timeout_ms;

        for priority in &ordered {
            // Skip already processed mutants.
            if walk_state.processed.contains(&priority.id) {
                continue;
            }

            let elapsed_ms = start.elapsed().as_millis() as u64;
            if elapsed_ms >= deadline_ms {
                walk_state.halt_reason = Some(HaltReason::Timeout);
                break;
            }

            if walk_state.steps.len() >= self.config.max_steps {
                walk_state.halt_reason = Some(HaltReason::MaxSteps);
                break;
            }

            let step = self.walk_step(disc, &mut walk_state, &priority.id, &mut entailment_checks);
            walk_state.steps.push(step);
            walk_state.processed.insert(priority.id.clone());
        }

        walk_state.completed = true;

        let contract = self.build_contract(disc, &walk_state, function_name);
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        self.statistics = Some(self.build_statistics(&walk_state, elapsed_ms, entailment_checks));
        self.state = Some(walk_state);

        contract
    }

    /// Synthesize a contract using only a specified subset of mutants.
    pub fn synthesize_with_subset(
        &mut self,
        disc: &mut DiscriminationLattice,
        mutant_subset: &[MutantId],
        function_name: &str,
    ) -> Contract {
        let start = Instant::now();
        let total_mutants = mutant_subset.len();
        let mut walk_state = WalkState::initial(total_mutants);

        if total_mutants == 0 {
            walk_state.completed = true;
            self.state = Some(walk_state);
            self.statistics = Some(self.build_statistics(&walk_state, 0.0, 0));
            return Contract::new(function_name.to_string());
        }

        let ordered = self.compute_walk_order(disc, mutant_subset);
        let mut entailment_checks: u64 = 0;
        let deadline_ms = self.config.timeout_ms;

        for priority in &ordered {
            let elapsed_ms = start.elapsed().as_millis() as u64;
            if elapsed_ms >= deadline_ms {
                walk_state.halt_reason = Some(HaltReason::Timeout);
                break;
            }

            if walk_state.steps.len() >= self.config.max_steps {
                walk_state.halt_reason = Some(HaltReason::MaxSteps);
                break;
            }

            let step = self.walk_step(disc, &mut walk_state, &priority.id, &mut entailment_checks);
            walk_state.steps.push(step);
            walk_state.processed.insert(priority.id.clone());
        }

        walk_state.completed = true;

        let contract = self.build_contract(disc, &walk_state, function_name);
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        self.statistics = Some(self.build_statistics(&walk_state, elapsed_ms, entailment_checks));
        self.state = Some(walk_state);

        contract
    }
}

impl fmt::Debug for LatticeWalkSynthesizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LatticeWalkSynthesizer")
            .field("config", &self.config)
            .field("has_state", &self.state.is_some())
            .field("has_statistics", &self.statistics.is_some())
            .finish()
    }
}

impl fmt::Display for LatticeWalkSynthesizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LatticeWalkSynthesizer")?;
        if let Some(stats) = &self.statistics {
            write!(
                f,
                " (last run: {:.1}ms, {}/{} mutants covered)",
                stats.total_time_ms, stats.mutants_covered, stats.mutants_total,
            )?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice::{var_eq, var_ge, var_lt, var_rel_const};

    fn make_id() -> MutantId {
        MutantId::new()
    }

    #[test]
    fn test_walk_config_default() {
        let cfg = WalkConfig::default();
        assert_eq!(cfg.max_formula_size, 200);
        assert!(cfg.simplify_eagerly);
        assert!(cfg.enable_subsumption);
    }

    #[test]
    fn test_walk_config_fast() {
        let cfg = WalkConfig::fast();
        assert_eq!(cfg.max_formula_size, 80);
        assert_eq!(cfg.timeout_ms, 5_000);
    }

    #[test]
    fn test_walk_config_thorough() {
        let cfg = WalkConfig::thorough();
        assert!(cfg.max_formula_size > 200);
        assert!(cfg.enable_recovery);
    }

    #[test]
    fn test_walk_step_display() {
        let id = make_id();
        let step = WalkStep::Accepted {
            mutant_id: id.clone(),
            new_size: 5,
            coverage: 3,
        };
        assert!(step.is_accepted());
        assert!(!step.is_skipped());
        let desc = step.description();
        assert!(desc.contains("accepted"));
    }

    #[test]
    fn test_walk_state_initial() {
        let state = WalkState::initial(10);
        assert_eq!(state.coverage(), 0);
        assert_eq!(state.step_count(), 0);
        assert!(state.current_element().is_top());
        assert!(!state.is_completed());
    }

    #[test]
    fn test_walk_statistics_acceptance_rate() {
        let stats = WalkStatistics {
            total_time_ms: 100.0,
            steps_attempted: 10,
            steps_accepted: 7,
            steps_skipped_complexity: 1,
            steps_skipped_subsumption: 1,
            steps_skipped_inconsistency: 1,
            steps_skipped_timeout: 0,
            steps_skipped_insufficient_gain: 0,
            final_formula_size: 20,
            final_formula_depth: 4,
            mutants_covered: 7,
            mutants_total: 10,
            entailment_checks: 15,
            simplification_passes: 7,
            halt_reason: None,
            clause_count: 3,
            coverage_percent: 70.0,
            tier: SynthesisTier::Tier1LatticeWalk,
        };
        assert!((stats.acceptance_rate() - 0.7).abs() < 1e-10);
        assert!(stats.contract_strength().is_adequate_or_better());
    }

    #[test]
    fn test_synthesize_empty() {
        let mut disc = DiscriminationLattice::new();
        let mut synth = LatticeWalkSynthesizer::with_defaults();
        let contract = synth.synthesize(&mut disc, "foo");
        assert_eq!(contract.function_name, "foo");
        assert!(contract.clauses.is_empty() || contract.strength == ContractStrength::Trivial);
    }

    #[test]
    fn test_synthesize_single_mutant() {
        let mut disc = DiscriminationLattice::new();
        let id = make_id();
        disc.register_mutant(id.clone(), var_ge("x", 0));
        let mut synth = LatticeWalkSynthesizer::with_defaults();
        let contract = synth.synthesize(&mut disc, "bar");
        assert_eq!(contract.function_name, "bar");
        assert!(!contract.clauses.is_empty());
    }

    #[test]
    fn test_synthesize_multiple_mutants() {
        let mut disc = DiscriminationLattice::new();
        let id1 = make_id();
        let id2 = make_id();
        let id3 = make_id();
        disc.register_mutant(id1.clone(), var_ge("x", 0));
        disc.register_mutant(id2.clone(), var_eq("y", 1));
        disc.register_mutant(id3.clone(), var_lt("z", 100));

        let mut synth = LatticeWalkSynthesizer::new(WalkConfig::default());
        let contract = synth.synthesize(&mut disc, "baz");
        assert_eq!(contract.function_name, "baz");
        assert!(synth.statistics().is_some());

        let stats = synth.statistics().unwrap();
        assert_eq!(stats.steps_attempted, 3);
    }

    #[test]
    fn test_halt_reason_display() {
        assert_eq!(HaltReason::Timeout.to_string(), "timeout");
        assert_eq!(HaltReason::MaxSteps.to_string(), "max steps reached");
    }

    #[test]
    fn test_mutant_priority_ordering() {
        let id1 = make_id();
        let id2 = make_id();
        let config = WalkConfig::default();

        // Smaller formula should get lower score (higher priority).
        let p1 = MutantPriority::compute(id1, &var_eq("x", 0), &config);
        let big_formula = Formula::and(vec![
            var_eq("x", 0),
            var_ge("y", 1),
            var_lt("z", 10),
        ]);
        let p2 = MutantPriority::compute(id2, &big_formula, &config);
        assert!(p1.score < p2.score);
    }

    #[test]
    fn test_decompose_conjunction() {
        let synth = LatticeWalkSynthesizer::with_defaults();
        let conj = Formula::and(vec![var_eq("x", 0), var_ge("y", 1)]);
        let elem = LatticeElement::new(conj);
        let clauses = synth.decompose_into_clauses(&elem);
        assert_eq!(clauses.len(), 2);
    }

    #[test]
    fn test_decompose_single() {
        let synth = LatticeWalkSynthesizer::with_defaults();
        let elem = LatticeElement::new(var_eq("x", 0));
        let clauses = synth.decompose_into_clauses(&elem);
        assert_eq!(clauses.len(), 1);
    }

    #[test]
    fn test_decompose_true() {
        let synth = LatticeWalkSynthesizer::with_defaults();
        let elem = LatticeElement::top();
        let clauses = synth.decompose_into_clauses(&elem);
        assert!(clauses.is_empty());
    }

    #[test]
    fn test_walk_state_display() {
        let state = WalkState::initial(5);
        let display = format!("{}", state);
        assert!(display.contains("WalkState"));
        assert!(display.contains("0/5"));
    }

    #[test]
    fn test_walk_statistics_display() {
        let stats = WalkStatistics {
            total_time_ms: 42.5,
            steps_attempted: 3,
            steps_accepted: 2,
            steps_skipped_complexity: 0,
            steps_skipped_subsumption: 1,
            steps_skipped_inconsistency: 0,
            steps_skipped_timeout: 0,
            steps_skipped_insufficient_gain: 0,
            final_formula_size: 10,
            final_formula_depth: 3,
            mutants_covered: 2,
            mutants_total: 3,
            entailment_checks: 5,
            simplification_passes: 2,
            halt_reason: None,
            clause_count: 2,
            coverage_percent: 66.7,
            tier: SynthesisTier::Tier1LatticeWalk,
        };
        let display = format!("{}", stats);
        assert!(display.contains("42.5 ms"));
    }
}
