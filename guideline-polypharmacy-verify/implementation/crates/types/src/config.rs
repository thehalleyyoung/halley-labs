//! Configuration types for the GuardPharma polypharmacy verification engine.
//!
//! This module defines every knob and toggle exposed to callers of the
//! verification pipeline.  The top-level [`VerificationConfig`] struct
//! aggregates sub-configurations for each analysis tier, pharmacokinetic
//! simulation, output formatting, and clinical rule-sets.
//!
//! All types derive `Debug`, `Clone`, `Serialize`, and `Deserialize` so they
//! can be round-tripped through JSON / YAML configuration files.

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Abstract domain & join strategy (Tier 1 – abstract interpretation)
// ---------------------------------------------------------------------------

/// The lattice domain used for abstract interpretation in Tier 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AbstractDomain {
    /// Classic interval domain – fast but imprecise for relational properties.
    Intervals,
    /// Octagon domain – tracks constraints of the form ±x ± y ≤ c.
    Octagons,
    /// Convex polyhedra – most precise, most expensive.
    Polyhedra,
    /// Difference-bound matrices (zones) – ±(x − y) ≤ c.
    Zones,
}

impl AbstractDomain {
    /// Whether the domain has a well-defined widening operator.
    pub fn supports_widening(&self) -> bool {
        match self {
            Self::Intervals => true,
            Self::Octagons => true,
            Self::Polyhedra => true,
            Self::Zones => true,
        }
    }

    /// Relative computational cost on a 1–4 scale (1 = cheapest).
    pub fn computational_cost_rank(&self) -> u32 {
        match self {
            Self::Intervals => 1,
            Self::Zones => 2,
            Self::Octagons => 3,
            Self::Polyhedra => 4,
        }
    }
}

impl fmt::Display for AbstractDomain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Intervals => write!(f, "intervals"),
            Self::Octagons => write!(f, "octagons"),
            Self::Polyhedra => write!(f, "polyhedra"),
            Self::Zones => write!(f, "zones"),
        }
    }
}

impl FromStr for AbstractDomain {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "intervals" => Ok(Self::Intervals),
            "octagons" => Ok(Self::Octagons),
            "polyhedra" => Ok(Self::Polyhedra),
            "zones" => Ok(Self::Zones),
            other => Err(format!("unknown abstract domain: `{other}`")),
        }
    }
}

/// Strategy for joining abstract states at control-flow merge points.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JoinStrategy {
    /// Plain least-upper-bound at every join.
    PointwiseJoin,
    /// Apply widening once the iteration count exceeds a threshold.
    WideningAfterThreshold,
    /// Delay widening for a configurable number of iterations, then widen.
    DelayedWidening,
}

impl fmt::Display for JoinStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PointwiseJoin => write!(f, "pointwise_join"),
            Self::WideningAfterThreshold => write!(f, "widening_after_threshold"),
            Self::DelayedWidening => write!(f, "delayed_widening"),
        }
    }
}

// ---------------------------------------------------------------------------
// Tier 1 configuration
// ---------------------------------------------------------------------------

/// Configuration for **Tier 1** analysis (abstract interpretation).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tier1Config {
    /// Maximum number of fixed-point iterations before giving up.
    pub max_iterations: usize,
    /// Iteration count at which widening kicks in.
    pub widening_threshold: usize,
    /// Number of iterations to delay before applying widening.
    pub widening_delay: usize,
    /// Number of narrowing iterations after a post-fixpoint is reached.
    pub narrowing_iterations: usize,
    /// Floating-point precision (epsilon) for convergence checks.
    pub precision: f64,
    /// Whether widening is enabled at all.
    pub use_widening: bool,
    /// Whether narrowing should be applied after reaching a post-fixpoint.
    pub use_narrowing: bool,
    /// The abstract domain to use.
    pub abstract_domain: AbstractDomain,
    /// Join strategy at merge points.
    pub join_strategy: JoinStrategy,
}

impl Default for Tier1Config {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            widening_threshold: 5,
            widening_delay: 3,
            narrowing_iterations: 4,
            precision: 1e-8,
            use_widening: true,
            use_narrowing: true,
            abstract_domain: AbstractDomain::Octagons,
            join_strategy: JoinStrategy::WideningAfterThreshold,
        }
    }
}

impl Tier1Config {
    /// Validate internal consistency, returning a list of problems (if any).
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors: Vec<String> = Vec::new();

        if self.max_iterations == 0 {
            errors.push("tier1.max_iterations must be > 0".into());
        }
        if self.precision <= 0.0 || self.precision >= 1.0 {
            errors.push("tier1.precision must be in (0, 1)".into());
        }
        if self.use_widening && self.widening_threshold == 0 {
            errors.push("tier1.widening_threshold must be > 0 when widening is enabled".into());
        }
        if self.use_widening && self.widening_delay >= self.widening_threshold {
            errors.push(
                "tier1.widening_delay must be < widening_threshold".into(),
            );
        }
        if self.use_narrowing && self.narrowing_iterations == 0 {
            errors.push(
                "tier1.narrowing_iterations must be > 0 when narrowing is enabled".into(),
            );
        }
        if !self.use_widening && self.join_strategy != JoinStrategy::PointwiseJoin {
            errors.push(
                "tier1.join_strategy must be PointwiseJoin when widening is disabled".into(),
            );
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

// ---------------------------------------------------------------------------
// SAT / SMT backend (Tier 2 – model checking)
// ---------------------------------------------------------------------------

/// Back-end solver used by the Tier 2 model checker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SatBackend {
    /// Built-in BDD-based engine (no external dependency).
    InternalBdd,
    /// Calls out to a Z3 binary via SMT-LIB2.
    ExternalZ3,
    /// Calls out to a CVC5 binary via SMT-LIB2.
    ExternalCvc5,
    /// Dumps SMT-LIB2 files to disk (offline mode).
    SmtLib2File,
}

impl SatBackend {
    /// Whether the backend requires an external binary.
    pub fn is_external(&self) -> bool {
        matches!(self, Self::ExternalZ3 | Self::ExternalCvc5)
    }

    /// Whether the backend needs a configured binary path to function.
    pub fn requires_binary_path(&self) -> bool {
        matches!(self, Self::ExternalZ3 | Self::ExternalCvc5)
    }
}

impl fmt::Display for SatBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InternalBdd => write!(f, "internal_bdd"),
            Self::ExternalZ3 => write!(f, "external_z3"),
            Self::ExternalCvc5 => write!(f, "external_cvc5"),
            Self::SmtLib2File => write!(f, "smtlib2_file"),
        }
    }
}

impl FromStr for SatBackend {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().replace('-', "_").as_str() {
            "internal_bdd" | "bdd" => Ok(Self::InternalBdd),
            "external_z3" | "z3" => Ok(Self::ExternalZ3),
            "external_cvc5" | "cvc5" => Ok(Self::ExternalCvc5),
            "smtlib2_file" | "smtlib2" | "smt" => Ok(Self::SmtLib2File),
            other => Err(format!("unknown SAT backend: `{other}`")),
        }
    }
}

// ---------------------------------------------------------------------------
// Tier 2 configuration
// ---------------------------------------------------------------------------

/// Configuration for **Tier 2** analysis (bounded model checking / SAT).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tier2Config {
    /// Maximum search depth (number of transition steps to unroll).
    pub max_depth: usize,
    /// Maximum number of states explored before aborting.
    pub max_states: usize,
    /// Per-property timeout in seconds.
    pub timeout_secs: u64,
    /// Use contract-based reasoning to decompose verification tasks.
    pub use_contracts: bool,
    /// Enable compositional (modular) verification.
    pub compositional: bool,
    /// Back-end solver to use.
    pub sat_backend: SatBackend,
    /// Maximum depth for counterexample generation.
    pub counterexample_depth: usize,
    /// Exploit state symmetries to prune the search space.
    pub symmetry_reduction: bool,
    /// Use partial-order reduction for concurrent models.
    pub partial_order_reduction: bool,
}

impl Default for Tier2Config {
    fn default() -> Self {
        Self {
            max_depth: 50,
            max_states: 1_000_000,
            timeout_secs: 300,
            use_contracts: true,
            compositional: false,
            sat_backend: SatBackend::InternalBdd,
            counterexample_depth: 20,
            symmetry_reduction: true,
            partial_order_reduction: true,
        }
    }
}

impl Tier2Config {
    /// Validate internal consistency.
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors: Vec<String> = Vec::new();

        if self.max_depth == 0 {
            errors.push("tier2.max_depth must be > 0".into());
        }
        if self.max_states == 0 {
            errors.push("tier2.max_states must be > 0".into());
        }
        if self.timeout_secs == 0 {
            errors.push("tier2.timeout_secs must be > 0".into());
        }
        if self.counterexample_depth == 0 {
            errors.push("tier2.counterexample_depth must be > 0".into());
        }
        if self.counterexample_depth > self.max_depth {
            errors.push(
                "tier2.counterexample_depth must be <= max_depth".into(),
            );
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

// ---------------------------------------------------------------------------
// PK sub-models
// ---------------------------------------------------------------------------

/// Pharmacokinetic compartment model topology.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompartmentModel {
    /// Single-compartment (plasma only).
    OneCompartment,
    /// Two-compartment (plasma + peripheral tissue).
    TwoCompartment,
    /// Three-compartment (plasma + shallow + deep tissue).
    ThreeCompartment,
    /// Physiologically-based pharmacokinetic model.
    #[serde(rename = "pbpk")]
    PBPK,
}

impl CompartmentModel {
    /// Number of distinct compartments in the model.
    pub fn num_compartments(&self) -> usize {
        match self {
            Self::OneCompartment => 1,
            Self::TwoCompartment => 2,
            Self::ThreeCompartment => 3,
            Self::PBPK => 14, // typical whole-body PBPK: ~14 tissue compartments
        }
    }

    /// Whether the model requires tissue-level physiological data.
    pub fn requires_tissue_data(&self) -> bool {
        matches!(self, Self::PBPK)
    }
}

impl fmt::Display for CompartmentModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OneCompartment => write!(f, "one_compartment"),
            Self::TwoCompartment => write!(f, "two_compartment"),
            Self::ThreeCompartment => write!(f, "three_compartment"),
            Self::PBPK => write!(f, "pbpk"),
        }
    }
}

impl FromStr for CompartmentModel {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().replace('-', "_").as_str() {
            "one_compartment" | "1cmt" | "1" => Ok(Self::OneCompartment),
            "two_compartment" | "2cmt" | "2" => Ok(Self::TwoCompartment),
            "three_compartment" | "3cmt" | "3" => Ok(Self::ThreeCompartment),
            "pbpk" => Ok(Self::PBPK),
            other => Err(format!("unknown compartment model: `{other}`")),
        }
    }
}

/// Drug absorption model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AbsorptionModel {
    /// Bolus / IV – drug appears instantly in the central compartment.
    Instantaneous,
    /// First-order absorption with a rate constant ka.
    FirstOrder,
    /// Zero-order (constant-rate) absorption (e.g., controlled-release).
    ZeroOrder,
    /// Saturable (Michaelis–Menten) absorption kinetics.
    MichaelisMenten,
    /// Transit-compartment absorption chain.
    Transit,
}

impl fmt::Display for AbsorptionModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Instantaneous => write!(f, "instantaneous"),
            Self::FirstOrder => write!(f, "first_order"),
            Self::ZeroOrder => write!(f, "zero_order"),
            Self::MichaelisMenten => write!(f, "michaelis_menten"),
            Self::Transit => write!(f, "transit"),
        }
    }
}

/// Drug distribution model within a compartment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DistributionModel {
    /// Instantaneous equilibrium between plasma and tissue.
    Instantaneous,
    /// Blood-flow–limited distribution (well-stirred model).
    PerfusionLimited,
    /// Membrane-permeability–limited distribution.
    PermeabilityLimited,
}

impl fmt::Display for DistributionModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Instantaneous => write!(f, "instantaneous"),
            Self::PerfusionLimited => write!(f, "perfusion_limited"),
            Self::PermeabilityLimited => write!(f, "permeability_limited"),
        }
    }
}

// ---------------------------------------------------------------------------
// PK configuration
// ---------------------------------------------------------------------------

/// Configuration for pharmacokinetic modelling and simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PkConfig {
    /// Which compartment topology to use.
    pub compartment_model: CompartmentModel,
    /// Integration time step in hours.
    pub time_step_hours: f64,
    /// Total simulation window in hours (168 h = 1 week).
    pub simulation_horizon_hours: f64,
    /// Relative change threshold for declaring steady state.
    pub steady_state_threshold: f64,
    /// Maximum number of ODE solver steps per simulation.
    pub max_ode_steps: usize,
    /// Absorption sub-model.
    pub absorption_model: AbsorptionModel,
    /// Distribution sub-model.
    pub distribution_model: DistributionModel,
    /// Account for plasma-protein binding.
    pub enable_protein_binding: bool,
    /// Model renal clearance explicitly.
    pub enable_renal_clearance: bool,
    /// Model hepatic (liver) clearance explicitly.
    pub enable_hepatic_clearance: bool,
}

impl Default for PkConfig {
    fn default() -> Self {
        Self {
            compartment_model: CompartmentModel::TwoCompartment,
            time_step_hours: 0.1,
            simulation_horizon_hours: 168.0, // 1 week
            steady_state_threshold: 0.01,
            max_ode_steps: 100_000,
            absorption_model: AbsorptionModel::FirstOrder,
            distribution_model: DistributionModel::PerfusionLimited,
            enable_protein_binding: true,
            enable_renal_clearance: true,
            enable_hepatic_clearance: true,
        }
    }
}

impl PkConfig {
    /// Validate internal consistency.
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors: Vec<String> = Vec::new();

        if self.time_step_hours <= 0.0 {
            errors.push("pk.time_step_hours must be > 0".into());
        }
        if self.simulation_horizon_hours <= 0.0 {
            errors.push("pk.simulation_horizon_hours must be > 0".into());
        }
        if self.time_step_hours >= self.simulation_horizon_hours {
            errors.push(
                "pk.time_step_hours must be < simulation_horizon_hours".into(),
            );
        }
        if self.steady_state_threshold <= 0.0 || self.steady_state_threshold >= 1.0 {
            errors.push("pk.steady_state_threshold must be in (0, 1)".into());
        }
        if self.max_ode_steps == 0 {
            errors.push("pk.max_ode_steps must be > 0".into());
        }
        if self.compartment_model == CompartmentModel::PBPK
            && self.distribution_model == DistributionModel::Instantaneous
        {
            errors.push(
                "pk: PBPK compartment model is incompatible with instantaneous distribution"
                    .into(),
            );
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

// ---------------------------------------------------------------------------
// Output configuration
// ---------------------------------------------------------------------------

/// Supported output formats for verification reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OutputFormat {
    Json,
    Yaml,
    Html,
    Markdown,
    PlainText,
}

impl fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Json => write!(f, "json"),
            Self::Yaml => write!(f, "yaml"),
            Self::Html => write!(f, "html"),
            Self::Markdown => write!(f, "markdown"),
            Self::PlainText => write!(f, "plain_text"),
        }
    }
}

impl FromStr for OutputFormat {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "json" => Ok(Self::Json),
            "yaml" | "yml" => Ok(Self::Yaml),
            "html" => Ok(Self::Html),
            "markdown" | "md" => Ok(Self::Markdown),
            "plain_text" | "plaintext" | "text" | "txt" => Ok(Self::PlainText),
            other => Err(format!("unknown output format: `{other}`")),
        }
    }
}

/// Verbosity level for reports and console output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verbosity {
    /// Suppress all non-error output.
    Quiet = 0,
    /// Standard output.
    Normal = 1,
    /// Extra diagnostic information.
    Verbose = 2,
    /// Full debug traces.
    Debug = 3,
}

impl Verbosity {
    /// Returns `true` if `self` is at least as verbose as `level`.
    pub fn is_at_least(&self, level: &Verbosity) -> bool {
        self >= level
    }
}

impl fmt::Display for Verbosity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Quiet => write!(f, "quiet"),
            Self::Normal => write!(f, "normal"),
            Self::Verbose => write!(f, "verbose"),
            Self::Debug => write!(f, "debug"),
        }
    }
}

impl FromStr for Verbosity {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "quiet" | "q" => Ok(Self::Quiet),
            "normal" | "n" => Ok(Self::Normal),
            "verbose" | "v" => Ok(Self::Verbose),
            "debug" | "d" => Ok(Self::Debug),
            other => Err(format!("unknown verbosity level: `{other}`")),
        }
    }
}

/// Output-related settings for verification results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    /// Report file format.
    pub format: OutputFormat,
    /// How much detail to include.
    pub verbosity: Verbosity,
    /// Include full execution / proof traces.
    pub include_traces: bool,
    /// Generate human-readable clinical narratives.
    pub include_narratives: bool,
    /// Attach PK concentration-time curves to the report.
    pub include_pk_curves: bool,
    /// Maximum number of counterexamples per property.
    pub max_counterexamples: usize,
    /// Directory for output artefacts (None = current directory).
    pub output_dir: Option<String>,
    /// Title that appears at the top of generated reports.
    pub report_title: Option<String>,
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            format: OutputFormat::Json,
            verbosity: Verbosity::Normal,
            include_traces: false,
            include_narratives: true,
            include_pk_curves: true,
            max_counterexamples: 5,
            output_dir: None,
            report_title: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Log level
// ---------------------------------------------------------------------------

/// Application-level log severity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LogLevel {
    Off = 0,
    Error = 1,
    Warn = 2,
    Info = 3,
    Debug = 4,
    Trace = 5,
}

impl fmt::Display for LogLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Off => write!(f, "off"),
            Self::Error => write!(f, "error"),
            Self::Warn => write!(f, "warn"),
            Self::Info => write!(f, "info"),
            Self::Debug => write!(f, "debug"),
            Self::Trace => write!(f, "trace"),
        }
    }
}

impl FromStr for LogLevel {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "off" => Ok(Self::Off),
            "error" => Ok(Self::Error),
            "warn" | "warning" => Ok(Self::Warn),
            "info" => Ok(Self::Info),
            "debug" => Ok(Self::Debug),
            "trace" => Ok(Self::Trace),
            other => Err(format!("unknown log level: `{other}`")),
        }
    }
}

// ---------------------------------------------------------------------------
// Clinical configuration
// ---------------------------------------------------------------------------

/// Configuration for clinical rule evaluation and database look-ups.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClinicalConfig {
    /// Minimum conflict severity that triggers a report entry.
    /// Encoded as a string so it can reference the `ConflictSeverity` enum
    /// from the core crate without creating a circular dependency.
    /// Expected values: "info", "low", "moderate", "high", "critical".
    pub severity_threshold: String,
    /// Include Beers Criteria checks (inappropriate medications in elderly).
    pub include_beers_criteria: bool,
    /// Include STOPP/START criteria (screening tool for older persons).
    pub include_stopp_start: bool,
    /// Pull adverse-event data from the FDA FAERS database.
    pub include_faers_data: bool,
    /// Maximum patient age (years) for paediatric dose-adjustment rules.
    pub max_patient_age_years: u32,
    /// eGFR threshold (mL/min/1.73 m²) below which renal dose adjustment
    /// rules are activated.
    pub min_egfr_for_dose_adjustment: f64,
    /// Factor in hepatic impairment for dose adjustment.
    pub consider_hepatic_impairment: bool,
    /// Factor in renal impairment for dose adjustment.
    pub consider_renal_impairment: bool,
    /// Enable pharmacogenomic (PGx) variant-based dosing.
    pub pharmacogenomic_enabled: bool,
    /// File-system path to the drug monograph database.
    pub drug_database_path: Option<String>,
    /// File-system path to the drug–drug interaction database.
    pub interaction_database_path: Option<String>,
}

impl Default for ClinicalConfig {
    fn default() -> Self {
        Self {
            severity_threshold: "moderate".to_string(),
            include_beers_criteria: true,
            include_stopp_start: true,
            include_faers_data: false,
            max_patient_age_years: 120,
            min_egfr_for_dose_adjustment: 60.0, // CKD stage ≥ 3
            consider_hepatic_impairment: true,
            consider_renal_impairment: true,
            pharmacogenomic_enabled: false,
            drug_database_path: None,
            interaction_database_path: None,
        }
    }
}

impl ClinicalConfig {
    /// Recognised severity level strings (case-insensitive).
    const VALID_SEVERITIES: &'static [&'static str] =
        &["info", "low", "moderate", "high", "critical"];

    /// Validate internal consistency.
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors: Vec<String> = Vec::new();

        let normalised = self.severity_threshold.to_ascii_lowercase();
        if !Self::VALID_SEVERITIES.contains(&normalised.as_str()) {
            errors.push(format!(
                "clinical.severity_threshold `{}` is not one of {:?}",
                self.severity_threshold,
                Self::VALID_SEVERITIES,
            ));
        }

        if self.min_egfr_for_dose_adjustment <= 0.0 {
            errors.push("clinical.min_egfr_for_dose_adjustment must be > 0".into());
        }
        if self.min_egfr_for_dose_adjustment > 200.0 {
            errors.push(
                "clinical.min_egfr_for_dose_adjustment > 200 is physiologically implausible"
                    .into(),
            );
        }
        if self.max_patient_age_years == 0 {
            errors.push("clinical.max_patient_age_years must be > 0".into());
        }

        if let Some(ref path) = self.drug_database_path {
            if path.trim().is_empty() {
                errors.push("clinical.drug_database_path must not be empty if set".into());
            }
        }
        if let Some(ref path) = self.interaction_database_path {
            if path.trim().is_empty() {
                errors.push(
                    "clinical.interaction_database_path must not be empty if set".into(),
                );
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

// ---------------------------------------------------------------------------
// Top-level verification configuration
// ---------------------------------------------------------------------------

/// Top-level configuration for the polypharmacy verification engine.
///
/// Aggregates sub-configurations for each analysis tier, PK modelling,
/// output formatting, and clinical rule-sets.  Use the builder methods
/// ([`with_tier1`](Self::with_tier1), [`with_tier2`](Self::with_tier2), …)
/// for ergonomic construction, then call [`validate`](Self::validate)
/// before passing the config into the engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationConfig {
    /// Tier 1 (abstract interpretation) settings.
    pub tier1: Tier1Config,
    /// Tier 2 (model checking / SAT) settings.
    pub tier2: Tier2Config,
    /// Pharmacokinetic simulation settings.
    pub pk: PkConfig,
    /// Output / reporting settings.
    pub output: OutputConfig,
    /// Clinical rule-set and database settings.
    pub clinical: ClinicalConfig,
    /// Maximum number of verification tasks to run in parallel.
    pub max_concurrent_verifications: usize,
    /// Global wall-clock timeout in seconds for the entire run.
    pub global_timeout_secs: u64,
    /// Cache intermediate analysis results to speed up re-runs.
    pub enable_caching: bool,
    /// Application log level.
    pub log_level: LogLevel,
}

impl Default for VerificationConfig {
    fn default() -> Self {
        Self {
            tier1: Tier1Config::default(),
            tier2: Tier2Config::default(),
            pk: PkConfig::default(),
            output: OutputConfig::default(),
            clinical: ClinicalConfig::default(),
            max_concurrent_verifications: 4,
            global_timeout_secs: 3600,
            enable_caching: true,
            log_level: LogLevel::Info,
        }
    }
}

impl VerificationConfig {
    /// Create a new configuration with all defaults.
    pub fn new() -> Self {
        Self::default()
    }

    // -- Builder-pattern helpers ------------------------------------------

    /// Replace the Tier 1 sub-configuration.
    pub fn with_tier1(mut self, tier1: Tier1Config) -> Self {
        self.tier1 = tier1;
        self
    }

    /// Replace the Tier 2 sub-configuration.
    pub fn with_tier2(mut self, tier2: Tier2Config) -> Self {
        self.tier2 = tier2;
        self
    }

    /// Replace the PK sub-configuration.
    pub fn with_pk(mut self, pk: PkConfig) -> Self {
        self.pk = pk;
        self
    }

    /// Replace the output sub-configuration.
    pub fn with_output(mut self, output: OutputConfig) -> Self {
        self.output = output;
        self
    }

    /// Replace the clinical sub-configuration.
    pub fn with_clinical(mut self, clinical: ClinicalConfig) -> Self {
        self.clinical = clinical;
        self
    }

    /// Validate the entire configuration tree.
    ///
    /// Checks each sub-config for internal consistency and then verifies
    /// cross-cutting invariants between sub-configs.  Returns `Ok(())`
    /// when everything is consistent, or `Err(errors)` with a complete
    /// list of problems.
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors: Vec<String> = Vec::new();

        // Collect errors from sub-configs.
        if let Err(mut e) = self.tier1.validate() {
            errors.append(&mut e);
        }
        if let Err(mut e) = self.tier2.validate() {
            errors.append(&mut e);
        }
        if let Err(mut e) = self.pk.validate() {
            errors.append(&mut e);
        }
        if let Err(mut e) = self.clinical.validate() {
            errors.append(&mut e);
        }

        // Cross-cutting checks.
        if self.max_concurrent_verifications == 0 {
            errors.push("max_concurrent_verifications must be > 0".into());
        }
        if self.global_timeout_secs == 0 {
            errors.push("global_timeout_secs must be > 0".into());
        }
        if self.tier2.timeout_secs > self.global_timeout_secs {
            errors.push(
                "tier2.timeout_secs exceeds global_timeout_secs".into(),
            );
        }

        // Ensure counter-example depth doesn't exceed Tier 2 search depth.
        if self.output.max_counterexamples > 0
            && self.tier2.counterexample_depth > self.tier2.max_depth
        {
            errors.push(
                "tier2.counterexample_depth exceeds tier2.max_depth while counterexamples are requested".into(),
            );
        }

        // PK steady-state check: horizon must allow at least a few steps.
        let min_steps =
            (self.pk.simulation_horizon_hours / self.pk.time_step_hours).ceil() as usize;
        if min_steps > self.pk.max_ode_steps {
            errors.push(format!(
                "pk: horizon/step requires {} steps but max_ode_steps is {}",
                min_steps, self.pk.max_ode_steps,
            ));
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // 1. Default config validates successfully.
    #[test]
    fn default_config_validates() {
        let cfg = VerificationConfig::default();
        assert!(cfg.validate().is_ok(), "default config should be valid");
    }

    // 2. Default sub-configs have expected values.
    #[test]
    fn default_tier1_values() {
        let t1 = Tier1Config::default();
        assert_eq!(t1.max_iterations, 100);
        assert_eq!(t1.widening_threshold, 5);
        assert!(t1.use_widening);
        assert_eq!(t1.abstract_domain, AbstractDomain::Octagons);
    }

    // 3. Tier1 validation catches bad precision.
    #[test]
    fn tier1_invalid_precision() {
        let mut t1 = Tier1Config::default();
        t1.precision = -0.5;
        let errs = t1.validate().unwrap_err();
        assert!(errs.iter().any(|e| e.contains("precision")));
    }

    // 4. Tier2 validation catches counterexample > max_depth.
    #[test]
    fn tier2_counterexample_exceeds_depth() {
        let mut t2 = Tier2Config::default();
        t2.counterexample_depth = t2.max_depth + 1;
        let errs = t2.validate().unwrap_err();
        assert!(errs.iter().any(|e| e.contains("counterexample_depth")));
    }

    // 5. PK validation catches PBPK + instantaneous distribution.
    #[test]
    fn pk_pbpk_incompatible_distribution() {
        let mut pk = PkConfig::default();
        pk.compartment_model = CompartmentModel::PBPK;
        pk.distribution_model = DistributionModel::Instantaneous;
        let errs = pk.validate().unwrap_err();
        assert!(errs.iter().any(|e| e.contains("PBPK")));
    }

    // 6. Clinical validation rejects unknown severity.
    #[test]
    fn clinical_invalid_severity() {
        let mut cl = ClinicalConfig::default();
        cl.severity_threshold = "catastrophic".into();
        let errs = cl.validate().unwrap_err();
        assert!(errs.iter().any(|e| e.contains("severity_threshold")));
    }

    // 7. Builder pattern works.
    #[test]
    fn builder_pattern() {
        let cfg = VerificationConfig::new()
            .with_tier1(Tier1Config {
                max_iterations: 200,
                ..Tier1Config::default()
            })
            .with_output(OutputConfig {
                format: OutputFormat::Markdown,
                ..OutputConfig::default()
            });
        assert_eq!(cfg.tier1.max_iterations, 200);
        assert_eq!(cfg.output.format, OutputFormat::Markdown);
        assert!(cfg.validate().is_ok());
    }

    // 8. JSON serialization round-trip.
    #[test]
    fn json_roundtrip() {
        let cfg = VerificationConfig::default();
        let json = serde_json::to_string_pretty(&cfg).expect("serialize");
        let cfg2: VerificationConfig =
            serde_json::from_str(&json).expect("deserialize");
        assert_eq!(cfg2.tier1.max_iterations, cfg.tier1.max_iterations);
        assert_eq!(cfg2.pk.time_step_hours, cfg.pk.time_step_hours);
        assert_eq!(cfg2.log_level, cfg.log_level);
    }

    // 9. Display / FromStr round-trip for enums.
    #[test]
    fn enum_display_fromstr() {
        // AbstractDomain
        for domain in &[
            AbstractDomain::Intervals,
            AbstractDomain::Octagons,
            AbstractDomain::Polyhedra,
            AbstractDomain::Zones,
        ] {
            let s = domain.to_string();
            let parsed: AbstractDomain = s.parse().unwrap();
            assert_eq!(*domain, parsed);
        }

        // SatBackend
        for backend in &[
            SatBackend::InternalBdd,
            SatBackend::ExternalZ3,
            SatBackend::ExternalCvc5,
            SatBackend::SmtLib2File,
        ] {
            let s = backend.to_string();
            let parsed: SatBackend = s.parse().unwrap();
            assert_eq!(*backend, parsed);
        }

        // OutputFormat
        for fmt in &[
            OutputFormat::Json,
            OutputFormat::Yaml,
            OutputFormat::Html,
            OutputFormat::Markdown,
            OutputFormat::PlainText,
        ] {
            let s = fmt.to_string();
            let parsed: OutputFormat = s.parse().unwrap();
            assert_eq!(*fmt, parsed);
        }

        // LogLevel
        for lvl in &[
            LogLevel::Off,
            LogLevel::Error,
            LogLevel::Warn,
            LogLevel::Info,
            LogLevel::Debug,
            LogLevel::Trace,
        ] {
            let s = lvl.to_string();
            let parsed: LogLevel = s.parse().unwrap();
            assert_eq!(*lvl, parsed);
        }
    }

    // 10. Verbosity ordering.
    #[test]
    fn verbosity_ordering() {
        assert!(Verbosity::Debug > Verbosity::Verbose);
        assert!(Verbosity::Verbose > Verbosity::Normal);
        assert!(Verbosity::Normal > Verbosity::Quiet);

        assert!(Verbosity::Debug.is_at_least(&Verbosity::Quiet));
        assert!(Verbosity::Normal.is_at_least(&Verbosity::Normal));
        assert!(!Verbosity::Quiet.is_at_least(&Verbosity::Normal));
    }

    // 11. Cross-cutting validation: tier2 timeout > global timeout.
    #[test]
    fn cross_cutting_timeout_validation() {
        let mut cfg = VerificationConfig::default();
        cfg.tier2.timeout_secs = cfg.global_timeout_secs + 1;
        let errs = cfg.validate().unwrap_err();
        assert!(errs.iter().any(|e| e.contains("global_timeout_secs")));
    }

    // 12. CompartmentModel helpers.
    #[test]
    fn compartment_model_helpers() {
        assert_eq!(CompartmentModel::OneCompartment.num_compartments(), 1);
        assert_eq!(CompartmentModel::TwoCompartment.num_compartments(), 2);
        assert_eq!(CompartmentModel::PBPK.num_compartments(), 14);
        assert!(CompartmentModel::PBPK.requires_tissue_data());
        assert!(!CompartmentModel::TwoCompartment.requires_tissue_data());
    }

    // 13. SatBackend external checks.
    #[test]
    fn sat_backend_external() {
        assert!(!SatBackend::InternalBdd.is_external());
        assert!(SatBackend::ExternalZ3.is_external());
        assert!(SatBackend::ExternalCvc5.requires_binary_path());
        assert!(!SatBackend::SmtLib2File.requires_binary_path());
    }

    // 14. AbstractDomain cost ranking.
    #[test]
    fn abstract_domain_cost_ranking() {
        assert!(
            AbstractDomain::Intervals.computational_cost_rank()
                < AbstractDomain::Polyhedra.computational_cost_rank()
        );
        assert!(AbstractDomain::Zones.supports_widening());
    }
}
