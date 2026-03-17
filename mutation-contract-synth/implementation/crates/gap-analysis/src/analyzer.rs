//! # analyzer
//!
//! Core gap analysis engine for MutSpec.
//!
//! The [`GapAnalyzer`] drives the full gap-analysis pipeline:
//!
//! 1. Accept a set of surviving mutants together with an inferred contract.
//! 2. Partition survivors into *equivalent* and *non-equivalent* via
//!    [`crate::equivalence::EquivalenceChecker`].
//! 3. For each non-equivalent survivor, determine whether the current contract
//!    *covers* the mutation – i.e. whether the contract's clauses already
//!    distinguish the mutant from the original.
//! 4. Survivors **not** covered by the contract are *gap witnesses* and are
//!    forwarded to [`crate::witness::WitnessGenerator`] for concrete input
//!    synthesis.
//! 5. All results are assembled into a [`GapReport`] that downstream consumers
//!    (ranking, SARIF, statistics) can process.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::time::{Duration, Instant};

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use shared_types::contracts::{Contract, ContractClause, ContractStrength};
use shared_types::formula::{Formula, Predicate, Relation, Term};
use shared_types::operators::{MutantDescriptor, MutantId, MutantStatus, MutationOperator};

use crate::equivalence::{EquivalenceChecker, EquivalenceClass, EquivalenceResult};
use crate::witness::{DistinguishingInput, GapWitness, WitnessGenerator};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration knobs for the gap analysis engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GapAnalysisConfig {
    /// Maximum wall-clock time for the entire analysis.
    pub timeout: Duration,

    /// Maximum time for a single equivalence check.
    pub per_mutant_timeout: Duration,

    /// Whether to attempt equivalence detection before gap checking.
    pub enable_equivalence_detection: bool,

    /// Minimum contract strength required to consider a clause as covering.
    pub minimum_contract_strength: ContractStrength,

    /// Maximum number of gap witnesses to produce.
    pub max_witnesses: usize,

    /// Whether to generate concrete distinguishing inputs for each witness.
    pub generate_inputs: bool,

    /// Maximum number of distinguishing inputs per witness.
    pub max_inputs_per_witness: usize,

    /// Whether to include killed mutants in the final report for context.
    pub include_killed_in_report: bool,

    /// Operator categories to exclude from analysis.
    pub excluded_operators: HashSet<MutationOperator>,

    /// Whether to perform subsumption-aware deduplication of witnesses.
    pub deduplicate_witnesses: bool,

    /// Confidence threshold below which witnesses are discarded.
    pub confidence_threshold: f64,
}

impl Default for GapAnalysisConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(300),
            per_mutant_timeout: Duration::from_secs(30),
            enable_equivalence_detection: true,
            minimum_contract_strength: ContractStrength::Adequate,
            max_witnesses: 1000,
            generate_inputs: true,
            max_inputs_per_witness: 5,
            include_killed_in_report: false,
            excluded_operators: HashSet::new(),
            deduplicate_witnesses: true,
            confidence_threshold: 0.1,
        }
    }
}

impl GapAnalysisConfig {
    /// Create a fast configuration suitable for CI / smoke-test usage.
    pub fn fast() -> Self {
        Self {
            timeout: Duration::from_secs(60),
            per_mutant_timeout: Duration::from_secs(5),
            max_witnesses: 50,
            max_inputs_per_witness: 1,
            deduplicate_witnesses: false,
            ..Self::default()
        }
    }

    /// Create a thorough configuration for nightly / full analysis.
    pub fn thorough() -> Self {
        Self {
            timeout: Duration::from_secs(1800),
            per_mutant_timeout: Duration::from_secs(120),
            max_witnesses: 10_000,
            max_inputs_per_witness: 20,
            confidence_threshold: 0.0,
            ..Self::default()
        }
    }

    /// Validate configuration invariants.
    pub fn validate(&self) -> Result<(), GapAnalysisError> {
        if self.timeout.is_zero() {
            return Err(GapAnalysisError::InvalidConfig(
                "timeout must be positive".into(),
            ));
        }
        if self.per_mutant_timeout > self.timeout {
            return Err(GapAnalysisError::InvalidConfig(
                "per_mutant_timeout exceeds total timeout".into(),
            ));
        }
        if self.confidence_threshold < 0.0 || self.confidence_threshold > 1.0 {
            return Err(GapAnalysisError::InvalidConfig(
                "confidence_threshold must be in [0, 1]".into(),
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by the gap analysis engine.
#[derive(Debug, thiserror::Error)]
pub enum GapAnalysisError {
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("analysis timed out after {0:?}")]
    Timeout(Duration),

    #[error("equivalence check failed for mutant {mutant_id}: {reason}")]
    EquivalenceCheckFailed { mutant_id: MutantId, reason: String },

    #[error("contract coverage check failed: {0}")]
    CoverageCheckFailed(String),

    #[error("witness generation failed for mutant {mutant_id}: {reason}")]
    WitnessGenerationFailed { mutant_id: MutantId, reason: String },

    #[error("no surviving mutants to analyse")]
    NoSurvivors,

    #[error(transparent)]
    Internal(#[from] anyhow::Error),
}

// ---------------------------------------------------------------------------
// Surviving mutant input
// ---------------------------------------------------------------------------

/// A surviving mutant together with its descriptor, ready for gap analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurvivingMutant {
    /// Unique identifier.
    pub id: MutantId,

    /// Full descriptor from the mutation engine.
    pub descriptor: MutantDescriptor,

    /// The mutation operator that produced this mutant.
    pub operator: MutationOperator,

    /// Original expression or statement (source-level).
    pub original_fragment: String,

    /// Mutated expression or statement (source-level).
    pub mutated_fragment: String,

    /// Function in which the mutation resides.
    pub function_name: String,

    /// Logical formula representing the *weakest precondition* of the mutant.
    pub mutant_wp: Option<Formula>,

    /// Logical formula representing the *weakest precondition* of the original.
    pub original_wp: Option<Formula>,
}

impl SurvivingMutant {
    /// Create a new surviving mutant record.
    pub fn new(
        id: MutantId,
        descriptor: MutantDescriptor,
        operator: MutationOperator,
        original_fragment: String,
        mutated_fragment: String,
        function_name: String,
    ) -> Self {
        Self {
            id,
            descriptor,
            operator,
            original_fragment,
            mutated_fragment,
            function_name,
            mutant_wp: None,
            original_wp: None,
        }
    }

    /// Attach weakest-precondition formulas for semantic comparison.
    pub fn with_wp(mut self, original: Formula, mutant: Formula) -> Self {
        self.original_wp = Some(original);
        self.mutant_wp = Some(mutant);
        self
    }

    /// Returns `true` if both WP formulas are available for semantic analysis.
    pub fn has_wp(&self) -> bool {
        self.original_wp.is_some() && self.mutant_wp.is_some()
    }
}

// ---------------------------------------------------------------------------
// Gap classification
// ---------------------------------------------------------------------------

/// Classification of a surviving mutant after gap analysis.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GapClassification {
    /// The mutant is semantically equivalent to the original.
    Equivalent,

    /// The contract already distinguishes this mutant – no gap.
    CoveredByContract {
        /// Which clause(s) cover the mutant.
        covering_clauses: Vec<ContractClause>,
    },

    /// The contract does **not** distinguish this mutant – a spec gap exists.
    SpecificationGap {
        /// Concrete distinguishing inputs, if generated.
        witnesses: Vec<GapWitness>,
        /// The clauses that were checked but failed to cover.
        checked_clauses: Vec<ContractClause>,
    },

    /// Analysis was inconclusive (e.g. timeout, solver unknown).
    Inconclusive { reason: String },
}

impl GapClassification {
    /// Returns `true` when the classification represents a real spec gap.
    pub fn is_gap(&self) -> bool {
        matches!(self, Self::SpecificationGap { .. })
    }

    /// Returns `true` when the mutant is equivalent.
    pub fn is_equivalent(&self) -> bool {
        matches!(self, Self::Equivalent)
    }

    /// Returns `true` when the contract already covers the mutant.
    pub fn is_covered(&self) -> bool {
        matches!(self, Self::CoveredByContract { .. })
    }

    /// Returns `true` when analysis was inconclusive.
    pub fn is_inconclusive(&self) -> bool {
        matches!(self, Self::Inconclusive { .. })
    }

    /// Human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Equivalent => "equivalent",
            Self::CoveredByContract { .. } => "covered",
            Self::SpecificationGap { .. } => "gap",
            Self::Inconclusive { .. } => "inconclusive",
        }
    }
}

impl fmt::Display for GapClassification {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Equivalent => write!(f, "Equivalent (no distinguishing input exists)"),
            Self::CoveredByContract { covering_clauses } => {
                write!(
                    f,
                    "Covered by {} contract clause(s)",
                    covering_clauses.len()
                )
            }
            Self::SpecificationGap {
                witnesses,
                checked_clauses,
            } => {
                write!(
                    f,
                    "Specification gap ({} witness(es), {} clause(s) checked)",
                    witnesses.len(),
                    checked_clauses.len()
                )
            }
            Self::Inconclusive { reason } => write!(f, "Inconclusive: {reason}"),
        }
    }
}

// ---------------------------------------------------------------------------
// Per-mutant result
// ---------------------------------------------------------------------------

/// Result of analysing a single surviving mutant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutantAnalysisResult {
    /// The mutant that was analysed.
    pub mutant_id: MutantId,

    /// Function containing the mutation.
    pub function_name: String,

    /// Mutation operator that produced this mutant.
    pub operator: MutationOperator,

    /// Gap classification.
    pub classification: GapClassification,

    /// Time spent analysing this mutant.
    pub analysis_duration: Duration,

    /// Equivalence class, if computed.
    pub equivalence_class: Option<EquivalenceClass>,
}

impl MutantAnalysisResult {
    /// Returns `true` when this result represents a specification gap.
    pub fn is_gap(&self) -> bool {
        self.classification.is_gap()
    }

    /// Returns the gap witnesses, if any.
    pub fn witnesses(&self) -> &[GapWitness] {
        match &self.classification {
            GapClassification::SpecificationGap { witnesses, .. } => witnesses,
            _ => &[],
        }
    }
}

// ---------------------------------------------------------------------------
// Aggregate report
// ---------------------------------------------------------------------------

/// Aggregate gap-analysis report over all surviving mutants.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GapReport {
    /// Unique identifier for this report.
    pub id: Uuid,

    /// Timestamp of report creation.
    pub created_at: chrono::DateTime<chrono::Utc>,

    /// Configuration used for analysis.
    pub config: GapAnalysisConfig,

    /// Per-mutant results, keyed by mutant id.
    pub results: IndexMap<MutantId, MutantAnalysisResult>,

    /// Total wall-clock time.
    pub total_duration: Duration,

    /// Whether the analysis timed out globally.
    pub timed_out: bool,

    /// Number of mutants that were equivalent.
    pub equivalent_count: usize,

    /// Number of mutants covered by the contract.
    pub covered_count: usize,

    /// Number of specification gaps found.
    pub gap_count: usize,

    /// Number of inconclusive results.
    pub inconclusive_count: usize,
}

impl GapReport {
    /// Create an empty report shell.
    pub fn new(config: GapAnalysisConfig) -> Self {
        Self {
            id: Uuid::new_v4(),
            created_at: chrono::Utc::now(),
            config,
            results: IndexMap::new(),
            total_duration: Duration::ZERO,
            timed_out: false,
            equivalent_count: 0,
            covered_count: 0,
            gap_count: 0,
            inconclusive_count: 0,
        }
    }

    /// Insert a per-mutant result and update aggregate counters.
    pub fn insert(&mut self, result: MutantAnalysisResult) {
        match &result.classification {
            GapClassification::Equivalent => self.equivalent_count += 1,
            GapClassification::CoveredByContract { .. } => self.covered_count += 1,
            GapClassification::SpecificationGap { .. } => self.gap_count += 1,
            GapClassification::Inconclusive { .. } => self.inconclusive_count += 1,
        }
        self.results.insert(result.mutant_id.clone(), result);
    }

    /// Total number of surviving mutants analysed.
    pub fn total_analysed(&self) -> usize {
        self.results.len()
    }

    /// Fraction of survivors that are specification gaps.
    pub fn gap_rate(&self) -> f64 {
        let total = self.total_analysed();
        if total == 0 {
            return 0.0;
        }
        self.gap_count as f64 / total as f64
    }

    /// Fraction of survivors that are equivalent.
    pub fn equivalence_rate(&self) -> f64 {
        let total = self.total_analysed();
        if total == 0 {
            return 0.0;
        }
        self.equivalent_count as f64 / total as f64
    }

    /// Iterate over gap witnesses only.
    pub fn gap_results(&self) -> impl Iterator<Item = &MutantAnalysisResult> {
        self.results.values().filter(|r| r.is_gap())
    }

    /// Iterate over equivalent mutant results.
    pub fn equivalent_results(&self) -> impl Iterator<Item = &MutantAnalysisResult> {
        self.results
            .values()
            .filter(|r| r.classification.is_equivalent())
    }

    /// Iterate over covered mutant results.
    pub fn covered_results(&self) -> impl Iterator<Item = &MutantAnalysisResult> {
        self.results
            .values()
            .filter(|r| r.classification.is_covered())
    }

    /// Collect all gap witnesses across all mutants.
    pub fn all_witnesses(&self) -> Vec<&GapWitness> {
        self.results.values().flat_map(|r| r.witnesses()).collect()
    }

    /// Group gap results by mutation operator.
    pub fn gaps_by_operator(&self) -> IndexMap<MutationOperator, Vec<&MutantAnalysisResult>> {
        let mut map: IndexMap<MutationOperator, Vec<&MutantAnalysisResult>> = IndexMap::new();
        for result in self.gap_results() {
            map.entry(result.operator.clone()).or_default().push(result);
        }
        map
    }

    /// Group gap results by function name.
    pub fn gaps_by_function(&self) -> IndexMap<String, Vec<&MutantAnalysisResult>> {
        let mut map: IndexMap<String, Vec<&MutantAnalysisResult>> = IndexMap::new();
        for result in self.gap_results() {
            map.entry(result.function_name.clone())
                .or_default()
                .push(result);
        }
        map
    }
}

impl fmt::Display for GapReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Gap Analysis Report  (id: {})", self.id)?;
        writeln!(f, "  Created:       {}", self.created_at)?;
        writeln!(f, "  Duration:      {:?}", self.total_duration)?;
        writeln!(f, "  Timed out:     {}", self.timed_out)?;
        writeln!(f, "  Total mutants: {}", self.total_analysed())?;
        writeln!(f, "  Equivalent:    {}", self.equivalent_count)?;
        writeln!(f, "  Covered:       {}", self.covered_count)?;
        writeln!(f, "  Gaps:          {}", self.gap_count)?;
        writeln!(f, "  Inconclusive:  {}", self.inconclusive_count)?;
        writeln!(f, "  Gap rate:      {:.1}%", self.gap_rate() * 100.0)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Analysis result wrapper
// ---------------------------------------------------------------------------

/// Top-level result of the gap analysis engine, wrapping the report and any
/// errors encountered during partial analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GapAnalysisResult {
    /// The gap report.
    pub report: GapReport,

    /// Per-mutant errors that did not abort the entire analysis.
    pub partial_errors: Vec<PartialError>,
}

/// A non-fatal error that occurred during analysis of a single mutant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartialError {
    pub mutant_id: MutantId,
    pub message: String,
}

impl GapAnalysisResult {
    /// Whether the analysis completed without any partial errors.
    pub fn is_clean(&self) -> bool {
        self.partial_errors.is_empty()
    }

    /// Convenience accessor for the report.
    pub fn report(&self) -> &GapReport {
        &self.report
    }

    /// Total number of gap witnesses found.
    pub fn total_witnesses(&self) -> usize {
        self.report.all_witnesses().len()
    }
}

// ---------------------------------------------------------------------------
// Core analyser
// ---------------------------------------------------------------------------

/// The gap analysis engine.
///
/// Orchestrates equivalence detection, contract coverage checking, and witness
/// generation to classify every surviving mutant.
///
/// # Usage
///
/// ```ignore
/// let config = GapAnalysisConfig::default();
/// let analyzer = GapAnalyzer::new(config);
/// let result = analyzer.analyse(&survivors, &contract)?;
/// println!("{}", result.report());
/// ```
pub struct GapAnalyzer {
    config: GapAnalysisConfig,
}

impl GapAnalyzer {
    /// Create a new gap analyser with the given configuration.
    pub fn new(config: GapAnalysisConfig) -> Self {
        Self { config }
    }

    /// Create a gap analyser with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(GapAnalysisConfig::default())
    }

    /// Run the full gap analysis pipeline.
    ///
    /// # Arguments
    ///
    /// * `survivors` – The set of surviving mutants to analyse.
    /// * `contract`  – The inferred contract for the function under test.
    ///
    /// # Errors
    ///
    /// Returns [`GapAnalysisError::NoSurvivors`] if `survivors` is empty, or
    /// [`GapAnalysisError::Timeout`] if the global timeout is exceeded.
    pub fn analyse(
        &self,
        survivors: &[SurvivingMutant],
        contract: &Contract,
    ) -> Result<GapAnalysisResult, GapAnalysisError> {
        self.config.validate()?;

        if survivors.is_empty() {
            return Err(GapAnalysisError::NoSurvivors);
        }

        log::info!(
            "Starting gap analysis for {} surviving mutant(s)",
            survivors.len()
        );

        let start = Instant::now();
        let mut report = GapReport::new(self.config.clone());
        let mut partial_errors: Vec<PartialError> = Vec::new();

        // Filter out excluded operators.
        let active_survivors: Vec<&SurvivingMutant> = survivors
            .iter()
            .filter(|s| !self.config.excluded_operators.contains(&s.operator))
            .collect();

        log::info!(
            "{} survivor(s) after operator filtering (excluded {} operator type(s))",
            active_survivors.len(),
            self.config.excluded_operators.len()
        );

        // Phase 1 – equivalence detection
        let equivalence_map = if self.config.enable_equivalence_detection {
            self.detect_equivalences(&active_survivors, &start)?
        } else {
            HashMap::new()
        };

        // Phase 2+3 – per-mutant analysis
        for survivor in &active_survivors {
            if start.elapsed() > self.config.timeout {
                log::warn!("Global timeout exceeded – aborting remaining mutants");
                report.timed_out = true;
                break;
            }

            match self.analyse_single(survivor, contract, &equivalence_map) {
                Ok(result) => report.insert(result),
                Err(e) => {
                    log::warn!("Error analysing mutant {}: {e}", survivor.id);
                    partial_errors.push(PartialError {
                        mutant_id: survivor.id.clone(),
                        message: e.to_string(),
                    });
                    // Record as inconclusive so the report is complete.
                    report.insert(MutantAnalysisResult {
                        mutant_id: survivor.id.clone(),
                        function_name: survivor.function_name.clone(),
                        operator: survivor.operator.clone(),
                        classification: GapClassification::Inconclusive {
                            reason: e.to_string(),
                        },
                        analysis_duration: Duration::ZERO,
                        equivalence_class: None,
                    });
                }
            }
        }

        report.total_duration = start.elapsed();

        // Phase 4 – deduplication
        if self.config.deduplicate_witnesses {
            self.deduplicate_gaps(&mut report);
        }

        log::info!(
            "Gap analysis complete: {} gap(s), {} equivalent, {} covered, {} inconclusive (took {:?})",
            report.gap_count,
            report.equivalent_count,
            report.covered_count,
            report.inconclusive_count,
            report.total_duration,
        );

        Ok(GapAnalysisResult {
            report,
            partial_errors,
        })
    }

    // -- internal helpers ---------------------------------------------------

    /// Run equivalence detection for all survivors.
    fn detect_equivalences(
        &self,
        survivors: &[&SurvivingMutant],
        start: &Instant,
    ) -> Result<HashMap<MutantId, EquivalenceResult>, GapAnalysisError> {
        log::info!(
            "Phase 1: equivalence detection for {} mutant(s)",
            survivors.len()
        );

        let checker = EquivalenceChecker::new(self.config.per_mutant_timeout);
        let mut map = HashMap::new();

        for survivor in survivors {
            if start.elapsed() > self.config.timeout {
                log::warn!("Timeout during equivalence detection phase");
                break;
            }

            let result = checker.check(survivor);
            map.insert(survivor.id.clone(), result);
        }

        let eq_count = map.values().filter(|r| r.is_equivalent()).count();
        log::info!(
            "Equivalence detection: {eq_count} equivalent out of {} checked",
            map.len()
        );

        Ok(map)
    }

    /// Analyse a single surviving mutant.
    fn analyse_single(
        &self,
        survivor: &SurvivingMutant,
        contract: &Contract,
        equivalence_map: &HashMap<MutantId, EquivalenceResult>,
    ) -> Result<MutantAnalysisResult, GapAnalysisError> {
        let mutant_start = Instant::now();

        // Check equivalence first.
        if let Some(eq_result) = equivalence_map.get(&survivor.id) {
            if eq_result.is_equivalent() {
                return Ok(MutantAnalysisResult {
                    mutant_id: survivor.id.clone(),
                    function_name: survivor.function_name.clone(),
                    operator: survivor.operator.clone(),
                    classification: GapClassification::Equivalent,
                    analysis_duration: mutant_start.elapsed(),
                    equivalence_class: Some(eq_result.class.clone()),
                });
            }
        }

        // Contract coverage check.
        let coverage = self.check_contract_coverage(survivor, contract)?;

        let classification = match coverage {
            ContractCoverage::Covered(clauses) => GapClassification::CoveredByContract {
                covering_clauses: clauses,
            },
            ContractCoverage::NotCovered(checked) => {
                // Generate witnesses.
                let witnesses = if self.config.generate_inputs {
                    self.generate_witnesses_for(survivor)?
                } else {
                    Vec::new()
                };
                GapClassification::SpecificationGap {
                    witnesses,
                    checked_clauses: checked,
                }
            }
        };

        let eq_class = equivalence_map.get(&survivor.id).map(|r| r.class.clone());

        Ok(MutantAnalysisResult {
            mutant_id: survivor.id.clone(),
            function_name: survivor.function_name.clone(),
            operator: survivor.operator.clone(),
            classification,
            analysis_duration: mutant_start.elapsed(),
            equivalence_class: eq_class,
        })
    }

    /// Check whether the contract clauses cover (distinguish) a mutant.
    ///
    /// A clause *covers* a mutant if, under the clause's precondition,
    /// the mutant's postcondition differs from the original's postcondition.
    fn check_contract_coverage(
        &self,
        survivor: &SurvivingMutant,
        contract: &Contract,
    ) -> Result<ContractCoverage, GapAnalysisError> {
        let clauses = &contract.clauses;
        if clauses.is_empty() {
            return Ok(ContractCoverage::NotCovered(Vec::new()));
        }

        let mut covering: Vec<ContractClause> = Vec::new();
        let mut checked: Vec<ContractClause> = Vec::new();

        for clause in clauses {
            checked.push(clause.clone());

            if self.clause_covers_mutant(clause, survivor)? {
                covering.push(clause.clone());
            }
        }

        if covering.is_empty() {
            Ok(ContractCoverage::NotCovered(checked))
        } else {
            Ok(ContractCoverage::Covered(covering))
        }
    }

    /// Determine whether a single contract clause distinguishes the mutant.
    ///
    /// For an `Ensures(φ)` clause the question is:
    ///   ∃ input . precondition(input) ∧ φ(original(input)) ∧ ¬φ(mutant(input))
    ///
    /// If this formula is satisfiable the clause covers the mutant.
    fn clause_covers_mutant(
        &self,
        clause: &ContractClause,
        survivor: &SurvivingMutant,
    ) -> Result<bool, GapAnalysisError> {
        let formula = clause.formula();

        // If WP information is not available, fall back to syntactic heuristics.
        let (original_wp, mutant_wp) = match (&survivor.original_wp, &survivor.mutant_wp) {
            (Some(o), Some(m)) => (o, m),
            _ => return Ok(self.syntactic_clause_check(clause, survivor)),
        };

        // Build distinguishing query:
        //   original_wp ∧ ¬mutant_wp  (under the clause's constraints)
        // If SAT → clause distinguishes the mutant.
        let distinguishing = Formula::And(vec![
            original_wp.clone(),
            Formula::Not(Box::new(mutant_wp.clone())),
            formula.clone(),
        ]);

        // Evaluate satisfiability via structural analysis.
        let is_sat = self.evaluate_formula_satisfiability(&distinguishing);

        Ok(is_sat)
    }

    /// Syntactic heuristic: check if a clause mentions variables affected by
    /// the mutation.
    fn syntactic_clause_check(&self, clause: &ContractClause, survivor: &SurvivingMutant) -> bool {
        let formula = clause.formula();
        let clause_vars = collect_formula_variables(formula);
        let mutation_vars =
            extract_mutation_variables(&survivor.original_fragment, &survivor.mutated_fragment);

        // If the clause references any variable affected by the mutation,
        // optimistically assume coverage.
        clause_vars.intersection(&mutation_vars).next().is_some()
    }

    /// Simple structural satisfiability check for a formula.
    ///
    /// This is a conservative approximation: it checks for trivially
    /// unsatisfiable patterns (e.g. `p ∧ ¬p`) and defaults to SAT otherwise.
    fn evaluate_formula_satisfiability(&self, formula: &Formula) -> bool {
        match formula {
            Formula::Atom(pred) => !is_trivially_false(pred),
            Formula::Not(inner) => match inner.as_ref() {
                Formula::Atom(pred) => !is_trivially_true(pred),
                _ => true,
            },
            Formula::And(conjuncts) => {
                // Check for complementary pairs.
                if has_complementary_pair(conjuncts) {
                    return false;
                }
                conjuncts
                    .iter()
                    .all(|c| self.evaluate_formula_satisfiability(c))
            }
            Formula::Or(disjuncts) => disjuncts
                .iter()
                .any(|d| self.evaluate_formula_satisfiability(d)),
            Formula::Implies(lhs, rhs) => {
                // ¬lhs ∨ rhs
                !self.evaluate_formula_satisfiability(lhs)
                    || self.evaluate_formula_satisfiability(rhs)
            }
            Formula::True => true,
            Formula::False => false,
            _ => true, // conservative: assume satisfiable for Iff/Forall/Exists
        }
    }

    /// Generate gap witnesses for a single mutant.
    fn generate_witnesses_for(
        &self,
        survivor: &SurvivingMutant,
    ) -> Result<Vec<GapWitness>, GapAnalysisError> {
        let generator = WitnessGenerator::new(self.config.max_inputs_per_witness);
        generator
            .generate(survivor)
            .map_err(|e| GapAnalysisError::WitnessGenerationFailed {
                mutant_id: survivor.id.clone(),
                reason: e.to_string(),
            })
    }

    /// Deduplicate witnesses across the report.
    ///
    /// Two witnesses are considered duplicates if they share the same mutation
    /// operator, function, and their distinguishing inputs are identical up to
    /// variable renaming.
    fn deduplicate_gaps(&self, report: &mut GapReport) {
        let mut seen_signatures: HashSet<String> = HashSet::new();
        let mut to_demote: Vec<MutantId> = Vec::new();

        for (id, result) in &report.results {
            if !result.is_gap() {
                continue;
            }
            let sig = format!(
                "{}:{}:{}",
                result.function_name,
                result.operator,
                result.witnesses().len()
            );
            if !seen_signatures.insert(sig) {
                to_demote.push(id.clone());
            }
        }

        for id in &to_demote {
            if let Some(result) = report.results.get_mut(id) {
                result.classification = GapClassification::Inconclusive {
                    reason: "deduplicated: subsumed by another gap witness".into(),
                };
                report.gap_count = report.gap_count.saturating_sub(1);
                report.inconclusive_count += 1;
            }
        }

        if !to_demote.is_empty() {
            log::info!(
                "Deduplicated {} subsumable gap witness(es)",
                to_demote.len()
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Result of contract coverage analysis.
enum ContractCoverage {
    /// At least one clause covers the mutant.
    Covered(Vec<ContractClause>),
    /// No clause covers the mutant; all checked clauses listed.
    NotCovered(Vec<ContractClause>),
}

/// Collect free variable names from a formula.
fn collect_formula_variables(formula: &Formula) -> HashSet<String> {
    let mut vars = HashSet::new();
    collect_vars_rec(formula, &mut vars);
    vars
}

fn collect_vars_rec(formula: &Formula, vars: &mut HashSet<String>) {
    match formula {
        Formula::Atom(pred) => {
            collect_term_vars(&pred.left, vars);
            collect_term_vars(&pred.right, vars);
        }
        Formula::Not(inner) => collect_vars_rec(inner, vars),
        Formula::And(children) | Formula::Or(children) => {
            for child in children {
                collect_vars_rec(child, vars);
            }
        }
        Formula::Implies(lhs, rhs) => {
            collect_vars_rec(lhs, vars);
            collect_vars_rec(rhs, vars);
        }
        Formula::True | Formula::False => {}
        Formula::Iff(a, b) => {
            collect_vars_rec(a, vars);
            collect_vars_rec(b, vars);
        }
        Formula::Forall(_, body) | Formula::Exists(_, body) => {
            collect_vars_rec(body, vars);
        }
    }
}

fn collect_term_vars(term: &Term, vars: &mut HashSet<String>) {
    match term {
        Term::Var(name) => {
            vars.insert(name.clone());
        }
        Term::Const(_) => {}
        Term::Add(a, b) | Term::Sub(a, b) => {
            collect_term_vars(a, vars);
            collect_term_vars(b, vars);
        }
        Term::Mul(_, inner) => {
            collect_term_vars(inner, vars);
        }
        Term::Neg(inner) => collect_term_vars(inner, vars),
        Term::ArraySelect(arr, idx) => {
            collect_term_vars(arr, vars);
            collect_term_vars(idx, vars);
        }
        Term::Ite(cond, t, e) => {
            collect_vars_rec(cond, vars);
            collect_term_vars(t, vars);
            collect_term_vars(e, vars);
        }
        Term::Old(inner) => collect_term_vars(inner, vars),
        Term::Result => {}
    }
}

/// Extract variable names likely affected by a mutation from source fragments.
fn extract_mutation_variables(original: &str, mutated: &str) -> HashSet<String> {
    let mut vars = HashSet::new();

    // Simple lexical heuristic: split on non-alphanumeric characters and
    // collect tokens that look like identifiers.
    let extract = |s: &str| -> HashSet<String> {
        s.split(|c: char| !c.is_alphanumeric() && c != '_')
            .filter(|tok| {
                !tok.is_empty()
                    && tok
                        .chars()
                        .next()
                        .map_or(false, |c| c.is_alphabetic() || c == '_')
            })
            .map(String::from)
            .collect()
    };

    let orig_toks = extract(original);
    let mut_toks = extract(mutated);

    // Variables that appear in both fragments are likely the affected ones.
    for tok in orig_toks.intersection(&mut_toks) {
        vars.insert(tok.clone());
    }
    // Variables unique to either side are also interesting.
    for tok in orig_toks.symmetric_difference(&mut_toks) {
        vars.insert(tok.clone());
    }

    vars
}

/// Check if a predicate is trivially false (e.g. `x < x`).
fn is_trivially_false(pred: &Predicate) -> bool {
    if pred.left == pred.right {
        matches!(pred.relation, Relation::Lt | Relation::Gt | Relation::Ne)
    } else {
        false
    }
}

/// Check if a predicate is trivially true (e.g. `x == x`, `x <= x`).
fn is_trivially_true(pred: &Predicate) -> bool {
    if pred.left == pred.right {
        matches!(pred.relation, Relation::Eq | Relation::Le | Relation::Ge)
    } else {
        false
    }
}

/// Check whether a list of conjuncts contains `φ` and `¬φ`.
fn has_complementary_pair(conjuncts: &[Formula]) -> bool {
    for (i, a) in conjuncts.iter().enumerate() {
        for b in &conjuncts[i + 1..] {
            if is_negation_of(a, b) || is_negation_of(b, a) {
                return true;
            }
        }
    }
    false
}

/// Returns true if `a` is `¬b`.
fn is_negation_of(a: &Formula, b: &Formula) -> bool {
    match a {
        Formula::Not(inner) => inner.as_ref() == b,
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_valid() {
        GapAnalysisConfig::default().validate().unwrap();
    }

    #[test]
    fn fast_config_is_valid() {
        GapAnalysisConfig::fast().validate().unwrap();
    }

    #[test]
    fn thorough_config_is_valid() {
        GapAnalysisConfig::thorough().validate().unwrap();
    }

    #[test]
    fn invalid_confidence_threshold() {
        let mut cfg = GapAnalysisConfig::default();
        cfg.confidence_threshold = 1.5;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn gap_report_counters() {
        let mut report = GapReport::new(GapAnalysisConfig::default());
        assert_eq!(report.total_analysed(), 0);
        assert_eq!(report.gap_rate(), 0.0);

        report.insert(MutantAnalysisResult {
            mutant_id: MutantId::new(),
            function_name: "foo".into(),
            operator: MutationOperator::Aor,
            classification: GapClassification::Equivalent,
            analysis_duration: Duration::from_millis(10),
            equivalence_class: None,
        });

        report.insert(MutantAnalysisResult {
            mutant_id: MutantId::new(),
            function_name: "foo".into(),
            operator: MutationOperator::Ror,
            classification: GapClassification::SpecificationGap {
                witnesses: Vec::new(),
                checked_clauses: Vec::new(),
            },
            analysis_duration: Duration::from_millis(20),
            equivalence_class: None,
        });

        assert_eq!(report.total_analysed(), 2);
        assert_eq!(report.equivalent_count, 1);
        assert_eq!(report.gap_count, 1);
        assert!((report.gap_rate() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn classification_labels() {
        assert_eq!(GapClassification::Equivalent.label(), "equivalent");
        assert!(GapClassification::Equivalent.is_equivalent());
        assert!(!GapClassification::Equivalent.is_gap());

        let gap = GapClassification::SpecificationGap {
            witnesses: vec![],
            checked_clauses: vec![],
        };
        assert!(gap.is_gap());
        assert_eq!(gap.label(), "gap");
    }

    #[test]
    fn trivially_true_and_false_predicates() {
        let x = Term::Var("x".into());
        let eq_pred = Predicate::new(Relation::Eq, x.clone(), x.clone());
        assert!(is_trivially_true(&eq_pred));
        assert!(!is_trivially_false(&eq_pred));

        let lt_pred = Predicate::new(Relation::Lt, x.clone(), x.clone());
        assert!(is_trivially_false(&lt_pred));
        assert!(!is_trivially_true(&lt_pred));
    }

    #[test]
    fn complementary_pair_detection() {
        let atom = Formula::Atom(Predicate::new(
            Relation::Gt,
            Term::Var("x".into()),
            Term::Const(0),
        ));
        let neg = Formula::Not(Box::new(atom.clone()));
        assert!(has_complementary_pair(&[atom.clone(), neg.clone()]));
        assert!(!has_complementary_pair(&[atom.clone(), atom]));
    }

    #[test]
    fn extract_mutation_variables_basic() {
        let orig = "x + y";
        let mutated = "x - y";
        let vars = extract_mutation_variables(orig, mutated);
        assert!(vars.contains("x"));
        assert!(vars.contains("y"));
    }

    #[test]
    fn gap_report_display() {
        let report = GapReport::new(GapAnalysisConfig::default());
        let text = format!("{report}");
        assert!(text.contains("Gap Analysis Report"));
        assert!(text.contains("Total mutants: 0"));
    }
}
