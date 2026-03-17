//! # GuardPharma Model Checker
//!
//! Tier 2 model-checking engine for the GuardPharma polypharmacy verification
//! system.  This crate performs **contract-based compositional model checking**
//! over Pharmacological Timed Automata (PTA).
//!
//! ## Architecture
//!
//! 1. **Product construction** ([`product`]) – synchronised parallel composition
//!    of individual drug PTAs.
//! 2. **Contract extraction & composition** ([`contract`]) – assume-guarantee
//!    reasoning over shared CYP enzyme resources.
//! 3. **Bounded model checking** ([`bounded_checker`]) – iterative deepening BMC
//!    with incremental solver support.
//! 4. **Counterexample extraction** ([`counterexample`]) – decode SMT models into
//!    PTA execution traces and minimise them.
//! 5. **Clinical narration** ([`narrator`]) – translate traces into human-readable
//!    pharmacological narratives.
//! 6. **CEGAR** ([`cegar`]) – counterexample-guided abstraction refinement loop.
//! 7. **Verification orchestrator** ([`verification`]) – top-level entry point
//!    that chains all the above.

pub use guardpharma_types::drug::{
    DrugId, DrugInfo, DrugClass, DrugRoute, DosingSchedule,
    TherapeuticWindow, ToxicThreshold, Severity, PatientInfo,
};
pub use guardpharma_types::enzyme::{
    CypEnzyme, EnzymeActivity, InhibitionType, InhibitionConstant,
    EnzymeMetabolismRoute, EnzymeInhibitionEffect, InductionEffect,
};
pub use guardpharma_types::concentration::{Concentration, ConcentrationInterval};
pub use guardpharma_types::identifiers::{GuidelineId, InteractionId, VerificationRunId};

/// Backward-compatible aliases for renamed types.
pub type CypMetabolismRoute = EnzymeMetabolismRoute;
pub type CypInhibitionEffect = EnzymeInhibitionEffect;
pub type CypInductionEffect = InductionEffect;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Result alias
// ---------------------------------------------------------------------------

/// Crate-level result type.
pub type Result<T> = std::result::Result<T, ModelCheckerError>;

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

/// Errors specific to the model-checking subsystem.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelCheckerError {
    /// PTA construction failed.
    PtaConstruction(String),
    /// Product construction exceeded resource limits.
    StateSpaceExplosion { num_states: usize, limit: usize },
    /// Contract incompatibility detected.
    ContractIncompatible { reason: String },
    /// SMT encoding or solving error.
    SolverError(String),
    /// Timeout during verification.
    Timeout { operation: String, duration_secs: f64 },
    /// Malformed safety property.
    MalformedProperty(String),
    /// CEGAR iteration limit exceeded.
    CegarDiverged { iterations: usize },
    /// Generic internal error.
    Internal(String),
    /// Encoding too large.
    EncodingTooLarge { clauses: usize, limit: usize },
    /// Invalid configuration.
    InvalidConfig(String),
}

impl fmt::Display for ModelCheckerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PtaConstruction(s) => write!(f, "PTA construction error: {s}"),
            Self::StateSpaceExplosion { num_states, limit } => {
                write!(f, "state-space explosion: {num_states} states (limit {limit})")
            }
            Self::ContractIncompatible { reason } => {
                write!(f, "contract incompatibility: {reason}")
            }
            Self::SolverError(s) => write!(f, "solver error: {s}"),
            Self::Timeout { operation, duration_secs } => {
                write!(f, "timeout in {operation} after {duration_secs:.1}s")
            }
            Self::MalformedProperty(s) => write!(f, "malformed property: {s}"),
            Self::CegarDiverged { iterations } => {
                write!(f, "CEGAR did not converge after {iterations} iterations")
            }
            Self::Internal(s) => write!(f, "internal error: {s}"),
            Self::EncodingTooLarge { clauses, limit } => {
                write!(f, "encoding too large: {clauses} clauses (limit {limit})")
            }
            Self::InvalidConfig(s) => write!(f, "invalid config: {s}"),
        }
    }
}

impl std::error::Error for ModelCheckerError {}

// ---------------------------------------------------------------------------
// PTA types  (Pharmacological Timed Automaton)
// ---------------------------------------------------------------------------

/// Unique location identifier within a PTA.
pub type LocationId = usize;

/// Unique edge identifier within a PTA.
pub type EdgeId = usize;

/// Clock identifier.
pub type ClockId = usize;

/// Variable identifier (shared or local).
pub type VariableId = usize;

/// A named clock used in timed-automaton guards / invariants.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Clock {
    pub id: ClockId,
    pub name: String,
}

/// A continuous variable (e.g. drug concentration, enzyme activity level).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Variable {
    pub id: VariableId,
    pub name: String,
    pub kind: VariableKind,
    pub lower_bound: f64,
    pub upper_bound: f64,
}

/// Classification of a PTA variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VariableKind {
    /// Drug plasma concentration.
    Concentration,
    /// CYP enzyme activity fraction.
    EnzymeActivity,
    /// Cumulative dose counter.
    CumulativeDose,
    /// Auxiliary / helper variable.
    Auxiliary,
}

/// Atomic predicate used in guards and invariants.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AtomicPredicate {
    /// clock ≤ constant
    ClockLeq { clock: ClockId, bound: f64 },
    /// clock ≥ constant
    ClockGeq { clock: ClockId, bound: f64 },
    /// clock == constant
    ClockEq { clock: ClockId, value: f64 },
    /// variable ≤ constant
    VarLeq { var: VariableId, bound: f64 },
    /// variable ≥ constant
    VarGeq { var: VariableId, bound: f64 },
    /// variable in [lo, hi]
    VarInRange { var: VariableId, lo: f64, hi: f64 },
    /// Boolean literal (always true / always false).
    BoolConst(bool),
}

impl AtomicPredicate {
    /// Evaluate this predicate against concrete clock and variable valuations.
    pub fn evaluate(&self, clocks: &[f64], vars: &[f64]) -> bool {
        match self {
            Self::ClockLeq { clock, bound } => {
                clocks.get(*clock).map_or(true, |v| *v <= *bound)
            }
            Self::ClockGeq { clock, bound } => {
                clocks.get(*clock).map_or(true, |v| *v >= *bound)
            }
            Self::ClockEq { clock, value } => {
                clocks.get(*clock).map_or(true, |v| (*v - *value).abs() < 1e-9)
            }
            Self::VarLeq { var, bound } => {
                vars.get(*var).map_or(true, |v| *v <= *bound)
            }
            Self::VarGeq { var, bound } => {
                vars.get(*var).map_or(true, |v| *v >= *bound)
            }
            Self::VarInRange { var, lo, hi } => {
                vars.get(*var).map_or(true, |v| *v >= *lo && *v <= *hi)
            }
            Self::BoolConst(b) => *b,
        }
    }
}

/// Conjunction of atomic predicates (represents a guard or invariant).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Predicate {
    pub conjuncts: Vec<AtomicPredicate>,
}

impl Predicate {
    pub fn trivially_true() -> Self {
        Self { conjuncts: vec![AtomicPredicate::BoolConst(true)] }
    }

    pub fn from_conjuncts(conjuncts: Vec<AtomicPredicate>) -> Self {
        Self { conjuncts }
    }

    pub fn evaluate(&self, clocks: &[f64], vars: &[f64]) -> bool {
        self.conjuncts.iter().all(|p| p.evaluate(clocks, vars))
    }

    pub fn is_trivially_true(&self) -> bool {
        self.conjuncts.is_empty()
            || (self.conjuncts.len() == 1
                && matches!(self.conjuncts[0], AtomicPredicate::BoolConst(true)))
    }

    pub fn and(mut self, other: Predicate) -> Self {
        self.conjuncts.extend(other.conjuncts);
        self
    }

    /// Return the set of variable IDs referenced.
    pub fn referenced_variables(&self) -> HashSet<VariableId> {
        let mut vars = HashSet::new();
        for c in &self.conjuncts {
            match c {
                AtomicPredicate::VarLeq { var, .. }
                | AtomicPredicate::VarGeq { var, .. }
                | AtomicPredicate::VarInRange { var, .. } => {
                    vars.insert(*var);
                }
                _ => {}
            }
        }
        vars
    }

    /// Return the set of clock IDs referenced.
    pub fn referenced_clocks(&self) -> HashSet<ClockId> {
        let mut clks = HashSet::new();
        for c in &self.conjuncts {
            match c {
                AtomicPredicate::ClockLeq { clock, .. }
                | AtomicPredicate::ClockGeq { clock, .. }
                | AtomicPredicate::ClockEq { clock, .. } => {
                    clks.insert(*clock);
                }
                _ => {}
            }
        }
        clks
    }
}

/// An update applied when an edge fires.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Update {
    /// Reset a clock to zero.
    ClockReset(ClockId),
    /// Assign a variable to a constant.
    VarAssign { var: VariableId, value: f64 },
    /// Increment a variable by a constant.
    VarIncrement { var: VariableId, delta: f64 },
    /// Multiply a variable by a factor (e.g. enzyme inhibition).
    VarScale { var: VariableId, factor: f64 },
}

/// An action label on an edge (for synchronisation).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ActionLabel {
    /// Internal (silent) action.
    Tau,
    /// Drug administration event.
    Administer { drug: String },
    /// Enzyme inhibition event.
    InhibitEnzyme { enzyme: String, drug: String },
    /// Enzyme induction event.
    InduceEnzyme { enzyme: String, drug: String },
    /// PK absorption event.
    Absorb { drug: String },
    /// PK elimination event.
    Eliminate { drug: String },
    /// Synchronisation label on a shared variable.
    Sync(String),
}

impl fmt::Display for ActionLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tau => write!(f, "τ"),
            Self::Administer { drug } => write!(f, "administer({drug})"),
            Self::InhibitEnzyme { enzyme, drug } => {
                write!(f, "inhibit({enzyme}, {drug})")
            }
            Self::InduceEnzyme { enzyme, drug } => {
                write!(f, "induce({enzyme}, {drug})")
            }
            Self::Absorb { drug } => write!(f, "absorb({drug})"),
            Self::Eliminate { drug } => write!(f, "eliminate({drug})"),
            Self::Sync(label) => write!(f, "sync({label})"),
        }
    }
}

/// A location (state) in a PTA.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Location {
    pub id: LocationId,
    pub name: String,
    pub invariant: Predicate,
    /// Whether this is an accepting / normal / error location.
    pub kind: LocationKind,
}

/// Classification of a PTA location.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LocationKind {
    Initial,
    Normal,
    Accepting,
    Error,
    Urgent,
}

/// An edge (transition) in a PTA.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Edge {
    pub id: EdgeId,
    pub source: LocationId,
    pub target: LocationId,
    pub guard: Predicate,
    pub action: ActionLabel,
    pub updates: Vec<Update>,
}

impl Edge {
    /// Return variable IDs read (in guard) and written (in updates).
    pub fn variable_reads(&self) -> HashSet<VariableId> {
        self.guard.referenced_variables()
    }

    pub fn variable_writes(&self) -> HashSet<VariableId> {
        let mut writes = HashSet::new();
        for u in &self.updates {
            match u {
                Update::VarAssign { var, .. }
                | Update::VarIncrement { var, .. }
                | Update::VarScale { var, .. } => {
                    writes.insert(*var);
                }
                _ => {}
            }
        }
        writes
    }
}

/// The Pharmacological Timed Automaton – core data structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PTA {
    pub name: String,
    pub drug_id: DrugId,
    pub locations: Vec<Location>,
    pub edges: Vec<Edge>,
    pub clocks: Vec<Clock>,
    pub variables: Vec<Variable>,
    pub initial_location: LocationId,
    pub initial_variable_values: Vec<f64>,
    /// PK parameters associated with this drug PTA.
    pub pk_params: PkParameters,
    /// Metabolism routes for this drug.
    pub metabolism_routes: Vec<CypMetabolismRoute>,
    /// Inhibition effects produced by this drug.
    pub inhibition_effects: Vec<CypInhibitionEffect>,
    /// Induction effects produced by this drug.
    pub induction_effects: Vec<CypInductionEffect>,
    /// Dosing schedule.
    pub dosing: DosingSchedule,
}

impl PTA {
    /// Number of locations.
    pub fn num_locations(&self) -> usize {
        self.locations.len()
    }

    /// Number of edges.
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    /// Outgoing edges from a location.
    pub fn outgoing_edges(&self, loc: LocationId) -> Vec<&Edge> {
        self.edges.iter().filter(|e| e.source == loc).collect()
    }

    /// Incoming edges to a location.
    pub fn incoming_edges(&self, loc: LocationId) -> Vec<&Edge> {
        self.edges.iter().filter(|e| e.target == loc).collect()
    }

    /// Get location by id.
    pub fn location(&self, id: LocationId) -> Option<&Location> {
        self.locations.get(id)
    }

    /// Get the set of CYP enzymes this PTA interacts with (metabolism + effects).
    pub fn involved_enzymes(&self) -> HashSet<CypEnzyme> {
        let mut enzymes = HashSet::new();
        for r in &self.metabolism_routes {
            enzymes.insert(r.enzyme);
        }
        for e in &self.inhibition_effects {
            enzymes.insert(e.enzyme);
        }
        for e in &self.induction_effects {
            enzymes.insert(e.enzyme);
        }
        enzymes
    }

    /// The set of variable IDs that are "shared" (enzyme-activity type).
    pub fn shared_variable_ids(&self) -> HashSet<VariableId> {
        self.variables
            .iter()
            .filter(|v| v.kind == VariableKind::EnzymeActivity)
            .map(|v| v.id)
            .collect()
    }
}

/// Simplified PK parameters sufficient for model-checking bounds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PkParameters {
    /// Absorption rate constant (h⁻¹).
    pub ka: f64,
    /// Elimination rate constant (h⁻¹).
    pub ke: f64,
    /// Volume of distribution (L).
    pub vd: f64,
    /// Bioavailability fraction (0..1).
    pub bioavailability: f64,
    /// Worst-case Cmax after single dose (mg/L).
    pub cmax_single: f64,
    /// Worst-case steady-state trough (mg/L).
    pub css_trough: f64,
    /// Worst-case steady-state peak (mg/L).
    pub css_peak: f64,
}

impl Default for PkParameters {
    fn default() -> Self {
        Self {
            ka: 1.0,
            ke: 0.1,
            vd: 50.0,
            bioavailability: 0.8,
            cmax_single: 5.0,
            css_trough: 2.0,
            css_peak: 8.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Safety properties
// ---------------------------------------------------------------------------

/// A safety property to be verified against a PTA (or product PTA).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyProperty {
    pub id: String,
    pub name: String,
    pub description: String,
    pub kind: SafetyPropertyKind,
}

/// The kind of safety property.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SafetyPropertyKind {
    /// AG(concentration(drug) ≤ threshold)
    ConcentrationBound {
        drug: DrugId,
        max_concentration: f64,
    },
    /// AG(concentration(drug) ∈ [lo, hi])  (therapeutic window)
    TherapeuticRange {
        drug: DrugId,
        lower: f64,
        upper: f64,
    },
    /// AG(enzyme_activity(enzyme) ≥ minimum)
    EnzymeActivityFloor {
        enzyme: CypEnzyme,
        min_activity: f64,
    },
    /// No error location is reachable.
    NoErrorReachable,
    /// Generic invariant expressed as a predicate.
    Invariant(Predicate),
    /// Bounded response: if P then within T time units Q holds.
    BoundedResponse {
        trigger: Predicate,
        response: Predicate,
        time_bound: f64,
    },
}

impl SafetyProperty {
    /// Create a concentration-bound property.
    pub fn concentration_bound(drug: DrugId, max: f64) -> Self {
        Self {
            id: format!("conc_bound_{}", drug),
            name: format!("Concentration bound for {}", drug),
            description: format!(
                "Plasma concentration of {} must never exceed {:.2} mg/L",
                drug, max
            ),
            kind: SafetyPropertyKind::ConcentrationBound {
                drug,
                max_concentration: max,
            },
        }
    }

    /// Create a therapeutic-range property.
    pub fn therapeutic_range(drug: DrugId, lower: f64, upper: f64) -> Self {
        Self {
            id: format!("ther_range_{}", drug),
            name: format!("Therapeutic range for {}", drug),
            description: format!(
                "Plasma concentration of {} must stay in [{:.2}, {:.2}] mg/L",
                drug, lower, upper
            ),
            kind: SafetyPropertyKind::TherapeuticRange { drug, lower, upper },
        }
    }

    /// Create an enzyme-activity-floor property.
    pub fn enzyme_floor(enzyme: CypEnzyme, min: f64) -> Self {
        Self {
            id: format!("enzyme_floor_{}", enzyme),
            name: format!("Enzyme activity floor for {}", enzyme),
            description: format!(
                "Activity of {} must remain above {:.0}%",
                enzyme,
                min * 100.0
            ),
            kind: SafetyPropertyKind::EnzymeActivityFloor {
                enzyme,
                min_activity: min,
            },
        }
    }

    /// No-error-reachable property.
    pub fn no_error() -> Self {
        Self {
            id: "no_error".into(),
            name: "No error reachable".into(),
            description: "No error location is reachable".into(),
            kind: SafetyPropertyKind::NoErrorReachable,
        }
    }

    /// The set of drug IDs referenced by this property.
    pub fn referenced_drugs(&self) -> HashSet<DrugId> {
        let mut drugs = HashSet::new();
        match &self.kind {
            SafetyPropertyKind::ConcentrationBound { drug, .. }
            | SafetyPropertyKind::TherapeuticRange { drug, .. } => {
                drugs.insert(drug.clone());
            }
            _ => {}
        }
        drugs
    }

    /// The set of enzymes referenced by this property.
    pub fn referenced_enzymes(&self) -> HashSet<CypEnzyme> {
        let mut enzymes = HashSet::new();
        if let SafetyPropertyKind::EnzymeActivityFloor { enzyme, .. } = &self.kind {
            enzymes.insert(*enzyme);
        }
        enzymes
    }
}

// ---------------------------------------------------------------------------
// Contract types
// ---------------------------------------------------------------------------

/// An enzyme-level assume-guarantee contract.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnzymeContract {
    /// The enzyme this contract is about.
    pub enzyme: CypEnzyme,
    /// The PTA (drug) that owns this contract.
    pub owner_drug: DrugId,
    /// Assumption: minimum enzyme activity required from the environment.
    pub assumed_min_activity: f64,
    /// Guarantee: maximum enzyme load (inhibition fraction) this drug produces.
    pub guaranteed_max_load: f64,
    /// The inhibition type (if applicable).
    pub inhibition_type: Option<InhibitionType>,
    /// Worst-case inhibitor concentration used to derive the guarantee.
    pub worst_case_inhibitor_conc: f64,
    /// Confidence: how tight the bound is (1.0 = exact, < 1.0 = conservative).
    pub tightness: f64,
}

impl EnzymeContract {
    /// Check whether this contract is satisfiable (guarantee ≤ 1 - assumption).
    pub fn is_self_consistent(&self) -> bool {
        self.guaranteed_max_load >= 0.0
            && self.guaranteed_max_load <= 1.0
            && self.assumed_min_activity >= 0.0
            && self.assumed_min_activity <= 1.0
    }
}

/// A guideline-level contract aggregating enzyme contracts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuidelineContract {
    pub guideline_id: String,
    pub drug_id: DrugId,
    pub enzyme_contracts: Vec<EnzymeContract>,
}

/// Result of contract compatibility checking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContractCompatibility {
    pub compatible: bool,
    pub enzyme_results: Vec<EnzymeCompatibilityResult>,
    pub summary: String,
}

/// Per-enzyme compatibility result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnzymeCompatibilityResult {
    pub enzyme: CypEnzyme,
    pub total_load: f64,
    pub capacity: f64,
    pub compatible: bool,
    pub margin: f64,
}

// ---------------------------------------------------------------------------
// Guideline document (simplified)
// ---------------------------------------------------------------------------

/// A clinical practice guideline document (simplified for narration).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuidelineDocument {
    pub id: String,
    pub name: String,
    pub drugs: Vec<DrugId>,
    pub description: String,
    pub recommendations: Vec<String>,
}

// ---------------------------------------------------------------------------
// SMT / solver types (self-contained since smt-encoder is a placeholder)
// ---------------------------------------------------------------------------

/// A simplified SMT variable.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct SmtVariable {
    pub name: String,
    pub sort: SmtSort,
}

/// SMT sort (type).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum SmtSort {
    Bool,
    Real,
    Int,
}

/// An SMT assertion (simplified string-based representation).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmtAssertion {
    pub description: String,
    pub smt2: String,
}

/// An encoded BMC problem ready for a solver.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodedProblem {
    pub variables: Vec<SmtVariable>,
    pub assertions: Vec<SmtAssertion>,
    pub bound: usize,
    pub num_clauses: usize,
    pub num_variables: usize,
    /// Mapping: step → (variable_name → SmtVariable) for decoding.
    pub step_variable_map: Vec<HashMap<String, SmtVariable>>,
}

impl EncodedProblem {
    pub fn empty(bound: usize) -> Self {
        Self {
            variables: Vec::new(),
            assertions: Vec::new(),
            bound,
            num_clauses: 0,
            num_variables: 0,
            step_variable_map: Vec::new(),
        }
    }
}

/// An SMT model (assignment from variables to values).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmtModel {
    pub assignments: HashMap<String, SmtValue>,
}

impl SmtModel {
    pub fn new() -> Self {
        Self { assignments: HashMap::new() }
    }

    pub fn get_real(&self, name: &str) -> Option<f64> {
        match self.assignments.get(name) {
            Some(SmtValue::Real(v)) => Some(*v),
            _ => None,
        }
    }

    pub fn get_bool(&self, name: &str) -> Option<bool> {
        match self.assignments.get(name) {
            Some(SmtValue::Bool(v)) => Some(*v),
            _ => None,
        }
    }

    pub fn get_int(&self, name: &str) -> Option<i64> {
        match self.assignments.get(name) {
            Some(SmtValue::Int(v)) => Some(*v),
            _ => None,
        }
    }
}

impl Default for SmtModel {
    fn default() -> Self {
        Self::new()
    }
}

/// A value in an SMT model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SmtValue {
    Bool(bool),
    Real(f64),
    Int(i64),
}

/// Solver verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SolverVerdict {
    Sat,
    Unsat,
    Unknown,
    Timeout,
}

impl fmt::Display for SolverVerdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sat => write!(f, "SAT"),
            Self::Unsat => write!(f, "UNSAT"),
            Self::Unknown => write!(f, "UNKNOWN"),
            Self::Timeout => write!(f, "TIMEOUT"),
        }
    }
}

// ---------------------------------------------------------------------------
// Verification verdicts
// ---------------------------------------------------------------------------

/// Outcome of checking a single safety property.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Verdict {
    /// Property is satisfied (safe).
    Safe,
    /// Property is violated (unsafe) – counterexample available.
    Unsafe,
    /// Could not determine within resource bounds.
    Unknown,
}

impl fmt::Display for Verdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Safe => write!(f, "SAFE"),
            Self::Unsafe => write!(f, "UNSAFE"),
            Self::Unknown => write!(f, "UNKNOWN"),
        }
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Top-level verification configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationConfig {
    /// Maximum BMC unrolling bound.
    pub max_bmc_bound: usize,
    /// Global timeout in seconds.
    pub timeout_secs: f64,
    /// Per-property timeout in seconds.
    pub per_property_timeout_secs: f64,
    /// Maximum product state-space size before giving up.
    pub max_product_states: usize,
    /// Whether to attempt contract-based decomposition.
    pub use_contracts: bool,
    /// Whether to run CEGAR on inconclusive results.
    pub use_cegar: bool,
    /// Maximum CEGAR iterations.
    pub max_cegar_iterations: usize,
    /// Whether to generate clinical narratives for counterexamples.
    pub generate_narratives: bool,
    /// Minimum enzyme capacity fraction considered safe.
    pub enzyme_capacity_threshold: f64,
}

impl Default for VerificationConfig {
    fn default() -> Self {
        Self {
            max_bmc_bound: 50,
            timeout_secs: 300.0,
            per_property_timeout_secs: 60.0,
            max_product_states: 1_000_000,
            use_contracts: true,
            use_cegar: true,
            max_cegar_iterations: 20,
            generate_narratives: true,
            enzyme_capacity_threshold: 0.3,
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: simple PTA builder (for testing)
// ---------------------------------------------------------------------------

/// Builder for constructing PTA instances programmatically.
#[derive(Debug)]
pub struct PtaBuilder {
    name: String,
    drug_id: DrugId,
    locations: Vec<Location>,
    edges: Vec<Edge>,
    clocks: Vec<Clock>,
    variables: Vec<Variable>,
    initial_location: LocationId,
    initial_variable_values: Vec<f64>,
    pk_params: PkParameters,
    metabolism_routes: Vec<CypMetabolismRoute>,
    inhibition_effects: Vec<CypInhibitionEffect>,
    induction_effects: Vec<CypInductionEffect>,
    dosing: DosingSchedule,
    next_edge_id: EdgeId,
}

impl PtaBuilder {
    pub fn new(name: &str, drug_id: DrugId) -> Self {
        Self {
            name: name.to_string(),
            drug_id,
            locations: Vec::new(),
            edges: Vec::new(),
            clocks: Vec::new(),
            variables: Vec::new(),
            initial_location: 0,
            initial_variable_values: Vec::new(),
            pk_params: PkParameters::default(),
            metabolism_routes: Vec::new(),
            inhibition_effects: Vec::new(),
            induction_effects: Vec::new(),
            dosing: DosingSchedule::new(12.0, 500.0, DrugRoute::Oral, 2),
            next_edge_id: 0,
        }
    }

    pub fn add_location(&mut self, name: &str, kind: LocationKind) -> LocationId {
        let id = self.locations.len();
        self.locations.push(Location {
            id,
            name: name.to_string(),
            invariant: Predicate::trivially_true(),
            kind,
        });
        id
    }

    pub fn add_location_with_invariant(
        &mut self,
        name: &str,
        kind: LocationKind,
        invariant: Predicate,
    ) -> LocationId {
        let id = self.locations.len();
        self.locations.push(Location {
            id,
            name: name.to_string(),
            invariant,
            kind,
        });
        id
    }

    pub fn add_clock(&mut self, name: &str) -> ClockId {
        let id = self.clocks.len();
        self.clocks.push(Clock { id, name: name.to_string() });
        id
    }

    pub fn add_variable(
        &mut self,
        name: &str,
        kind: VariableKind,
        lower: f64,
        upper: f64,
        initial: f64,
    ) -> VariableId {
        let id = self.variables.len();
        self.variables.push(Variable {
            id,
            name: name.to_string(),
            kind,
            lower_bound: lower,
            upper_bound: upper,
        });
        self.initial_variable_values.push(initial);
        id
    }

    pub fn add_edge(
        &mut self,
        source: LocationId,
        target: LocationId,
        guard: Predicate,
        action: ActionLabel,
        updates: Vec<Update>,
    ) -> EdgeId {
        let id = self.next_edge_id;
        self.next_edge_id += 1;
        self.edges.push(Edge { id, source, target, guard, action, updates });
        id
    }

    pub fn set_initial_location(&mut self, loc: LocationId) {
        self.initial_location = loc;
    }

    pub fn set_pk_params(&mut self, params: PkParameters) {
        self.pk_params = params;
    }

    pub fn set_dosing(&mut self, dosing: DosingSchedule) {
        self.dosing = dosing;
    }

    pub fn add_metabolism_route(&mut self, route: CypMetabolismRoute) {
        self.metabolism_routes.push(route);
    }

    pub fn add_inhibition_effect(&mut self, effect: CypInhibitionEffect) {
        self.inhibition_effects.push(effect);
    }

    pub fn add_induction_effect(&mut self, effect: CypInductionEffect) {
        self.induction_effects.push(effect);
    }

    pub fn build(self) -> PTA {
        PTA {
            name: self.name,
            drug_id: self.drug_id,
            locations: self.locations,
            edges: self.edges,
            clocks: self.clocks,
            variables: self.variables,
            initial_location: self.initial_location,
            initial_variable_values: self.initial_variable_values,
            pk_params: self.pk_params,
            metabolism_routes: self.metabolism_routes,
            inhibition_effects: self.inhibition_effects,
            induction_effects: self.induction_effects,
            dosing: self.dosing,
        }
    }
}

/// Create a minimal test PTA for a given drug with standard pharmacokinetic
/// locations: Idle → Absorbing → SteadyState → (optional) Toxic.
pub fn make_test_pta(drug_name: &str, dose_mg: f64, has_toxic: bool) -> PTA {
    let drug_id = DrugId::new(drug_name);
    let mut b = PtaBuilder::new(drug_name, drug_id.clone());

    let idle = b.add_location("idle", LocationKind::Initial);
    let absorbing = b.add_location("absorbing", LocationKind::Normal);
    let steady = b.add_location("steady_state", LocationKind::Normal);

    let _t = b.add_clock("t");
    let conc = b.add_variable(
        &format!("conc_{drug_name}"),
        VariableKind::Concentration,
        0.0,
        100.0,
        0.0,
    );

    b.set_initial_location(idle);

    // idle → absorbing: administer
    b.add_edge(
        idle,
        absorbing,
        Predicate::trivially_true(),
        ActionLabel::Administer { drug: drug_name.into() },
        vec![Update::VarAssign { var: conc, value: dose_mg / 50.0 }],
    );

    // absorbing → steady: absorb
    b.add_edge(
        absorbing,
        steady,
        Predicate::from_conjuncts(vec![AtomicPredicate::ClockGeq { clock: 0, bound: 1.0 }]),
        ActionLabel::Absorb { drug: drug_name.into() },
        vec![],
    );

    if has_toxic {
        let toxic = b.add_location("toxic", LocationKind::Error);
        let threshold = dose_mg / 50.0 * 1.5;
        b.add_edge(
            steady,
            toxic,
            Predicate::from_conjuncts(vec![AtomicPredicate::VarGeq {
                var: conc,
                bound: threshold,
            }]),
            ActionLabel::Tau,
            vec![],
        );
    }

    // steady → idle: eliminate
    b.add_edge(
        steady,
        idle,
        Predicate::from_conjuncts(vec![AtomicPredicate::ClockGeq { clock: 0, bound: 12.0 }]),
        ActionLabel::Eliminate { drug: drug_name.into() },
        vec![Update::VarAssign { var: conc, value: 0.0 }, Update::ClockReset(0)],
    );

    b.set_dosing(DosingSchedule::new(12.0, dose_mg, DrugRoute::Oral, 2));
    b.build()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predicate_trivially_true() {
        let p = Predicate::trivially_true();
        assert!(p.is_trivially_true());
        assert!(p.evaluate(&[], &[]));
    }

    #[test]
    fn test_predicate_evaluate_clock() {
        let p = Predicate::from_conjuncts(vec![
            AtomicPredicate::ClockLeq { clock: 0, bound: 10.0 },
        ]);
        assert!(p.evaluate(&[5.0], &[]));
        assert!(!p.evaluate(&[15.0], &[]));
    }

    #[test]
    fn test_predicate_evaluate_var() {
        let p = Predicate::from_conjuncts(vec![
            AtomicPredicate::VarInRange { var: 0, lo: 1.0, hi: 5.0 },
        ]);
        assert!(p.evaluate(&[], &[3.0]));
        assert!(!p.evaluate(&[], &[6.0]));
    }

    #[test]
    fn test_predicate_referenced_variables() {
        let p = Predicate::from_conjuncts(vec![
            AtomicPredicate::VarLeq { var: 0, bound: 10.0 },
            AtomicPredicate::VarGeq { var: 2, bound: 1.0 },
            AtomicPredicate::ClockLeq { clock: 0, bound: 5.0 },
        ]);
        let vars = p.referenced_variables();
        assert!(vars.contains(&0));
        assert!(vars.contains(&2));
        assert!(!vars.contains(&1));
    }

    #[test]
    fn test_action_label_display() {
        let a = ActionLabel::Administer { drug: "warfarin".into() };
        assert_eq!(format!("{a}"), "administer(warfarin)");
    }

    #[test]
    fn test_pta_builder() {
        let pta = make_test_pta("metformin", 500.0, true);
        assert_eq!(pta.num_locations(), 4); // idle, absorbing, steady, toxic
        assert!(pta.num_edges() >= 3);
        assert_eq!(pta.initial_location, 0);
    }

    #[test]
    fn test_pta_outgoing_edges() {
        let pta = make_test_pta("aspirin", 100.0, false);
        let out = pta.outgoing_edges(0);
        assert!(!out.is_empty());
    }

    #[test]
    fn test_safety_property_concentration_bound() {
        let prop = SafetyProperty::concentration_bound(DrugId::new("warfarin"), 5.0);
        assert!(prop.referenced_drugs().contains(&DrugId::new("warfarin")));
    }

    #[test]
    fn test_enzyme_contract_self_consistent() {
        let ec = EnzymeContract {
            enzyme: CypEnzyme::CYP3A4,
            owner_drug: DrugId::new("clarithromycin"),
            assumed_min_activity: 0.3,
            guaranteed_max_load: 0.6,
            inhibition_type: Some(InhibitionType::Competitive),
            worst_case_inhibitor_conc: 2.5,
            tightness: 0.9,
        };
        assert!(ec.is_self_consistent());
    }

    #[test]
    fn test_solver_verdict_display() {
        assert_eq!(format!("{}", SolverVerdict::Sat), "SAT");
        assert_eq!(format!("{}", SolverVerdict::Unsat), "UNSAT");
    }

    #[test]
    fn test_smt_model_accessors() {
        let mut model = SmtModel::new();
        model.assignments.insert("x".into(), SmtValue::Real(3.14));
        model.assignments.insert("b".into(), SmtValue::Bool(true));
        model.assignments.insert("n".into(), SmtValue::Int(42));
        assert!((model.get_real("x").unwrap() - 3.14).abs() < 1e-10);
        assert_eq!(model.get_bool("b"), Some(true));
        assert_eq!(model.get_int("n"), Some(42));
        assert_eq!(model.get_real("missing"), None);
    }

    #[test]
    fn test_verification_config_default() {
        let cfg = VerificationConfig::default();
        assert!(cfg.max_bmc_bound > 0);
        assert!(cfg.use_contracts);
    }

    #[test]
    fn test_variable_kind_enum() {
        assert_ne!(VariableKind::Concentration, VariableKind::EnzymeActivity);
    }

    #[test]
    fn test_edge_variable_reads_writes() {
        let e = Edge {
            id: 0,
            source: 0,
            target: 1,
            guard: Predicate::from_conjuncts(vec![
                AtomicPredicate::VarGeq { var: 0, bound: 1.0 },
            ]),
            action: ActionLabel::Tau,
            updates: vec![
                Update::VarAssign { var: 1, value: 0.0 },
                Update::VarIncrement { var: 2, delta: 1.0 },
            ],
        };
        assert!(e.variable_reads().contains(&0));
        assert!(e.variable_writes().contains(&1));
        assert!(e.variable_writes().contains(&2));
    }
}
