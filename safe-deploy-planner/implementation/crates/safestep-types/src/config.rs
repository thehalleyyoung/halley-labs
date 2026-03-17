// Configuration types for the SafeStep deployment planner.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::error::{Result, SafeStepError};

// ─── Top-level configuration ────────────────────────────────────────────

/// Top-level configuration for the SafeStep deployment planner.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafeStepConfig {
    pub solver: SolverConfig,
    pub encoding: EncodingConfig,
    pub planner: PlannerConfig,
    pub output: OutputConfig,
    pub k8s: K8sConfig,
    pub schema: SchemaConfig,
}

impl SafeStepConfig {
    pub fn default_config() -> Self {
        Self {
            solver: SolverConfig::default(),
            encoding: EncodingConfig::default(),
            planner: PlannerConfig::default(),
            output: OutputConfig::default(),
            k8s: K8sConfig::default(),
            schema: SchemaConfig::default(),
        }
    }

    /// Load from a JSON string.
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(|e| {
            SafeStepError::config(format!("Failed to parse JSON config: {}", e))
        })
    }

    /// Serialize to JSON string.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(|e| {
            SafeStepError::config(format!("Failed to serialize config: {}", e))
        })
    }

    /// Load from environment variables with the given prefix.
    pub fn from_env(prefix: &str) -> Self {
        let mut config = Self::default_config();

        if let Ok(val) = std::env::var(format!("{}_SOLVER_TIMEOUT_MS", prefix)) {
            if let Ok(ms) = val.parse::<u64>() {
                config.solver.timeout_ms = ms;
            }
        }
        if let Ok(val) = std::env::var(format!("{}_SOLVER_MAX_CLAUSES", prefix)) {
            if let Ok(n) = val.parse::<u64>() {
                config.solver.max_clauses = n;
            }
        }
        if let Ok(val) = std::env::var(format!("{}_PLANNER_BMC_DEPTH", prefix)) {
            if let Ok(d) = val.parse::<usize>() {
                config.planner.bmc_depth = d;
            }
        }
        if let Ok(val) = std::env::var(format!("{}_PLANNER_MAX_STATES", prefix)) {
            if let Ok(n) = val.parse::<usize>() {
                config.planner.max_states = n;
            }
        }
        if let Ok(val) = std::env::var(format!("{}_OUTPUT_FORMAT", prefix)) {
            config.output.format = OutputFormat::from_str(&val);
        }
        if let Ok(val) = std::env::var(format!("{}_OUTPUT_VERBOSITY", prefix)) {
            if let Ok(v) = val.parse::<u8>() {
                config.output.verbosity = Verbosity::from_level(v);
            }
        }
        if let Ok(val) = std::env::var(format!("{}_K8S_NAMESPACE", prefix)) {
            config.k8s.namespace = Some(val);
        }
        if let Ok(val) = std::env::var(format!("{}_K8S_KUBECONFIG", prefix)) {
            config.k8s.kubeconfig_path = Some(val);
        }

        config
    }

    /// Merge with another config (other takes precedence for non-default fields).
    pub fn merge_with(&mut self, other: &SafeStepConfig) {
        if other.solver.timeout_ms != SolverConfig::default().timeout_ms {
            self.solver.timeout_ms = other.solver.timeout_ms;
        }
        if other.solver.max_clauses != SolverConfig::default().max_clauses {
            self.solver.max_clauses = other.solver.max_clauses;
        }
        if other.planner.bmc_depth != PlannerConfig::default().bmc_depth {
            self.planner.bmc_depth = other.planner.bmc_depth;
        }
    }

    /// Validate the entire configuration for consistency.
    pub fn validate(&self) -> Result<()> {
        self.solver.validate()?;
        self.encoding.validate()?;
        self.planner.validate()?;
        self.output.validate()?;
        Ok(())
    }
}

impl Default for SafeStepConfig {
    fn default() -> Self {
        Self::default_config()
    }
}

// ─── Solver configuration ───────────────────────────────────────────────

/// Configuration for the SAT/SMT solver.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverConfig {
    pub timeout_ms: u64,
    pub max_clauses: u64,
    pub max_variables: u64,
    pub strategy: SolverStrategy,
    pub incremental: bool,
    pub random_seed: Option<u64>,
    pub restart_interval: u32,
    pub clause_decay: f64,
    pub variable_decay: f64,
    pub phase_saving: bool,
    pub luby_restart: bool,
    pub preprocessing: bool,
    pub subsumption: bool,
}

impl SolverConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn fast() -> Self {
        Self {
            timeout_ms: 5_000,
            max_clauses: 100_000,
            max_variables: 10_000,
            strategy: SolverStrategy::DPLL,
            preprocessing: false,
            ..Default::default()
        }
    }

    pub fn thorough() -> Self {
        Self {
            timeout_ms: 300_000,
            max_clauses: 10_000_000,
            max_variables: 1_000_000,
            strategy: SolverStrategy::CDCL,
            preprocessing: true,
            subsumption: true,
            ..Default::default()
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.timeout_ms == 0 {
            return Err(SafeStepError::config("solver timeout must be > 0"));
        }
        if self.clause_decay <= 0.0 || self.clause_decay >= 1.0 {
            return Err(SafeStepError::config(
                "clause_decay must be in (0, 1)",
            ));
        }
        if self.variable_decay <= 0.0 || self.variable_decay >= 1.0 {
            return Err(SafeStepError::config(
                "variable_decay must be in (0, 1)",
            ));
        }
        Ok(())
    }
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            timeout_ms: 30_000,
            max_clauses: 1_000_000,
            max_variables: 100_000,
            strategy: SolverStrategy::CDCL,
            incremental: true,
            random_seed: None,
            restart_interval: 100,
            clause_decay: 0.999,
            variable_decay: 0.95,
            phase_saving: true,
            luby_restart: true,
            preprocessing: true,
            subsumption: false,
        }
    }
}

/// SAT solver strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SolverStrategy {
    DPLL,
    CDCL,
    LocalSearch,
    Portfolio,
}

impl fmt::Display for SolverStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DPLL => write!(f, "DPLL"),
            Self::CDCL => write!(f, "CDCL"),
            Self::LocalSearch => write!(f, "LocalSearch"),
            Self::Portfolio => write!(f, "Portfolio"),
        }
    }
}

// ─── Encoding configuration ─────────────────────────────────────────────

/// Configuration for how constraints are encoded into SAT/SMT formulas.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodingConfig {
    pub encoding_type: EncodingType,
    pub use_interval_optimization: bool,
    pub use_bdd_compression: bool,
    pub treewidth_threshold: usize,
    pub max_bdd_nodes: u64,
    pub symmetry_breaking: bool,
    pub cardinality_encoding: CardinalityEncoding,
    pub pb_encoding: PBEncoding,
    pub use_tseitin: bool,
    pub variable_ordering: VariableOrdering,
}

impl EncodingConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn validate(&self) -> Result<()> {
        if self.treewidth_threshold == 0 {
            return Err(SafeStepError::config(
                "treewidth_threshold must be > 0",
            ));
        }
        if self.max_bdd_nodes == 0 {
            return Err(SafeStepError::config("max_bdd_nodes must be > 0"));
        }
        Ok(())
    }
}

impl Default for EncodingConfig {
    fn default() -> Self {
        Self {
            encoding_type: EncodingType::Interval,
            use_interval_optimization: true,
            use_bdd_compression: false,
            treewidth_threshold: 4,
            max_bdd_nodes: 1_000_000,
            symmetry_breaking: true,
            cardinality_encoding: CardinalityEncoding::Totalizer,
            pb_encoding: PBEncoding::BDD,
            use_tseitin: true,
            variable_ordering: VariableOrdering::Natural,
        }
    }
}

/// Type of SAT encoding for version constraints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EncodingType {
    /// Direct encoding: one boolean per (service, version) pair.
    Direct,
    /// Log encoding: binary representation of version indices.
    Log,
    /// Interval-based encoding exploiting interval structure.
    Interval,
    /// BDD-based encoding.
    BDD,
    /// Hybrid: choose per-service based on version set size.
    Hybrid,
}

impl fmt::Display for EncodingType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Direct => write!(f, "direct"),
            Self::Log => write!(f, "log"),
            Self::Interval => write!(f, "interval"),
            Self::BDD => write!(f, "BDD"),
            Self::Hybrid => write!(f, "hybrid"),
        }
    }
}

/// Cardinality constraint encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CardinalityEncoding {
    Sequential,
    Parallel,
    Totalizer,
    Modular,
}

/// Pseudo-boolean constraint encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PBEncoding {
    BDD,
    Sorter,
    Adder,
    Binary,
}

/// Variable ordering strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VariableOrdering {
    Natural,
    VSIDS,
    Random,
    MinDegree,
}

// ─── Planner configuration ──────────────────────────────────────────────

/// Configuration for the planning algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlannerConfig {
    pub bmc_depth: usize,
    pub max_states: usize,
    pub max_cegar_iterations: usize,
    pub monotone_only: bool,
    pub single_step: bool,
    pub optimization_goal: OptimizationGoal,
    pub pareto_front: bool,
    pub max_plans: usize,
    pub exploration_strategy: ExplorationStrategy,
    pub envelope_computation: bool,
    pub pnr_analysis: bool,
    pub symmetry_reduction: bool,
}

impl PlannerConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn fast() -> Self {
        Self {
            bmc_depth: 20,
            max_states: 10_000,
            max_cegar_iterations: 5,
            monotone_only: true,
            single_step: true,
            optimization_goal: OptimizationGoal::MinSteps,
            pareto_front: false,
            max_plans: 1,
            exploration_strategy: ExplorationStrategy::BFS,
            envelope_computation: false,
            pnr_analysis: false,
            symmetry_reduction: false,
        }
    }

    pub fn exhaustive() -> Self {
        Self {
            bmc_depth: 100,
            max_states: 1_000_000,
            max_cegar_iterations: 50,
            monotone_only: false,
            single_step: false,
            optimization_goal: OptimizationGoal::Pareto,
            pareto_front: true,
            max_plans: 10,
            exploration_strategy: ExplorationStrategy::AStar,
            envelope_computation: true,
            pnr_analysis: true,
            symmetry_reduction: true,
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.bmc_depth == 0 {
            return Err(SafeStepError::config("bmc_depth must be > 0"));
        }
        if self.max_states == 0 {
            return Err(SafeStepError::config("max_states must be > 0"));
        }
        if self.max_plans == 0 {
            return Err(SafeStepError::config("max_plans must be > 0"));
        }
        Ok(())
    }
}

impl Default for PlannerConfig {
    fn default() -> Self {
        Self {
            bmc_depth: 50,
            max_states: 100_000,
            max_cegar_iterations: 20,
            monotone_only: true,
            single_step: true,
            optimization_goal: OptimizationGoal::MinSteps,
            pareto_front: false,
            max_plans: 3,
            exploration_strategy: ExplorationStrategy::BFS,
            envelope_computation: true,
            pnr_analysis: true,
            symmetry_reduction: true,
        }
    }
}

/// Optimization goal for planning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OptimizationGoal {
    MinSteps,
    MinRisk,
    MinTime,
    MinPNR,
    Pareto,
    Custom,
}

impl fmt::Display for OptimizationGoal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MinSteps => write!(f, "min-steps"),
            Self::MinRisk => write!(f, "min-risk"),
            Self::MinTime => write!(f, "min-time"),
            Self::MinPNR => write!(f, "min-pnr"),
            Self::Pareto => write!(f, "pareto"),
            Self::Custom => write!(f, "custom"),
        }
    }
}

/// Graph exploration strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExplorationStrategy {
    BFS,
    DFS,
    AStar,
    BeamSearch,
    CEGAR,
}

impl fmt::Display for ExplorationStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BFS => write!(f, "BFS"),
            Self::DFS => write!(f, "DFS"),
            Self::AStar => write!(f, "A*"),
            Self::BeamSearch => write!(f, "beam-search"),
            Self::CEGAR => write!(f, "CEGAR"),
        }
    }
}

// ─── Output configuration ───────────────────────────────────────────────

/// Output format and verbosity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    pub format: OutputFormat,
    pub verbosity: Verbosity,
    pub output_path: Option<String>,
    pub include_graph_viz: bool,
    pub include_envelope_viz: bool,
    pub include_plan_trace: bool,
    pub max_display_states: usize,
    pub color_output: bool,
}

impl OutputConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn validate(&self) -> Result<()> {
        Ok(())
    }
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            format: OutputFormat::Json,
            verbosity: Verbosity::Normal,
            output_path: None,
            include_graph_viz: false,
            include_envelope_viz: false,
            include_plan_trace: true,
            max_display_states: 100,
            color_output: true,
        }
    }
}

/// Output format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OutputFormat {
    Json,
    Yaml,
    Text,
    Dot,
    Csv,
}

impl OutputFormat {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "json" => Self::Json,
            "yaml" | "yml" => Self::Yaml,
            "text" | "txt" => Self::Text,
            "dot" | "graphviz" => Self::Dot,
            "csv" => Self::Csv,
            _ => Self::Json,
        }
    }

    pub fn file_extension(&self) -> &str {
        match self {
            Self::Json => "json",
            Self::Yaml => "yaml",
            Self::Text => "txt",
            Self::Dot => "dot",
            Self::Csv => "csv",
        }
    }
}

impl fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.file_extension())
    }
}

/// Verbosity level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Verbosity {
    Quiet,
    Normal,
    Verbose,
    Debug,
    Trace,
}

impl Verbosity {
    pub fn from_level(level: u8) -> Self {
        match level {
            0 => Self::Quiet,
            1 => Self::Normal,
            2 => Self::Verbose,
            3 => Self::Debug,
            _ => Self::Trace,
        }
    }

    pub fn level(&self) -> u8 {
        match self {
            Self::Quiet => 0,
            Self::Normal => 1,
            Self::Verbose => 2,
            Self::Debug => 3,
            Self::Trace => 4,
        }
    }

    pub fn is_at_least(&self, other: Verbosity) -> bool {
        self.level() >= other.level()
    }
}

impl fmt::Display for Verbosity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Quiet => write!(f, "quiet"),
            Self::Normal => write!(f, "normal"),
            Self::Verbose => write!(f, "verbose"),
            Self::Debug => write!(f, "debug"),
            Self::Trace => write!(f, "trace"),
        }
    }
}

// ─── Kubernetes configuration ───────────────────────────────────────────

/// Kubernetes connection and filtering settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct K8sConfig {
    pub kubeconfig_path: Option<String>,
    pub context: Option<String>,
    pub namespace: Option<String>,
    pub namespace_filter: Vec<String>,
    pub label_selector: Option<String>,
    pub field_selector: Option<String>,
    pub timeout_secs: u64,
    pub batch_size: usize,
    pub dry_run: bool,
    pub force: bool,
}

impl K8sConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn in_cluster() -> Self {
        Self {
            kubeconfig_path: None,
            context: None,
            ..Default::default()
        }
    }

    pub fn with_namespace(mut self, namespace: impl Into<String>) -> Self {
        self.namespace = Some(namespace.into());
        self
    }

    pub fn with_kubeconfig(mut self, path: impl Into<String>) -> Self {
        self.kubeconfig_path = Some(path.into());
        self
    }

    pub fn with_dry_run(mut self) -> Self {
        self.dry_run = true;
        self
    }

    /// Effective namespace, defaulting to "default".
    pub fn effective_namespace(&self) -> &str {
        self.namespace.as_deref().unwrap_or("default")
    }
}

impl Default for K8sConfig {
    fn default() -> Self {
        Self {
            kubeconfig_path: None,
            context: None,
            namespace: None,
            namespace_filter: Vec::new(),
            label_selector: None,
            field_selector: None,
            timeout_secs: 30,
            batch_size: 50,
            dry_run: false,
            force: false,
        }
    }
}

// ─── Schema configuration ───────────────────────────────────────────────

/// Configuration for API schema analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaConfig {
    pub schema_source: SchemaSource,
    pub compatibility_mode: CompatibilityMode,
    pub cache_schemas: bool,
    pub cache_path: Option<String>,
    pub max_schema_size_bytes: usize,
    pub analyze_breaking_changes: bool,
    pub analyze_deprecations: bool,
}

impl SchemaConfig {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Default for SchemaConfig {
    fn default() -> Self {
        Self {
            schema_source: SchemaSource::Registry,
            compatibility_mode: CompatibilityMode::Backward,
            cache_schemas: true,
            cache_path: None,
            max_schema_size_bytes: 10 * 1024 * 1024,
            analyze_breaking_changes: true,
            analyze_deprecations: true,
        }
    }
}

/// Source of API schemas.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SchemaSource {
    Registry,
    Git,
    File,
    Inline,
}

impl fmt::Display for SchemaSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Registry => write!(f, "registry"),
            Self::Git => write!(f, "git"),
            Self::File => write!(f, "file"),
            Self::Inline => write!(f, "inline"),
        }
    }
}

/// API compatibility checking mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CompatibilityMode {
    /// New version must accept all old requests.
    Backward,
    /// Old version must accept all new requests.
    Forward,
    /// Both backward and forward.
    Full,
    /// No compatibility check.
    None,
}

impl fmt::Display for CompatibilityMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Backward => write!(f, "backward"),
            Self::Forward => write!(f, "forward"),
            Self::Full => write!(f, "full"),
            Self::None => write!(f, "none"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = SafeStepConfig::default_config();
        assert_eq!(config.solver.timeout_ms, 30_000);
        assert_eq!(config.planner.bmc_depth, 50);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_json_roundtrip() {
        let config = SafeStepConfig::default_config();
        let json = config.to_json().unwrap();
        let parsed = SafeStepConfig::from_json(&json).unwrap();
        assert_eq!(parsed.solver.timeout_ms, config.solver.timeout_ms);
    }

    #[test]
    fn test_solver_config_fast() {
        let cfg = SolverConfig::fast();
        assert_eq!(cfg.timeout_ms, 5_000);
        assert_eq!(cfg.strategy, SolverStrategy::DPLL);
    }

    #[test]
    fn test_solver_config_thorough() {
        let cfg = SolverConfig::thorough();
        assert_eq!(cfg.timeout_ms, 300_000);
        assert!(cfg.preprocessing);
    }

    #[test]
    fn test_solver_config_validate() {
        let cfg = SolverConfig::default();
        assert!(cfg.validate().is_ok());

        let bad = SolverConfig {
            timeout_ms: 0,
            ..Default::default()
        };
        assert!(bad.validate().is_err());

        let bad2 = SolverConfig {
            clause_decay: 1.5,
            ..Default::default()
        };
        assert!(bad2.validate().is_err());
    }

    #[test]
    fn test_encoding_config() {
        let cfg = EncodingConfig::default();
        assert_eq!(cfg.encoding_type, EncodingType::Interval);
        assert!(cfg.use_interval_optimization);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_encoding_config_validate() {
        let bad = EncodingConfig {
            treewidth_threshold: 0,
            ..Default::default()
        };
        assert!(bad.validate().is_err());
    }

    #[test]
    fn test_planner_config() {
        let cfg = PlannerConfig::default();
        assert_eq!(cfg.bmc_depth, 50);
        assert!(cfg.monotone_only);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_planner_config_fast() {
        let cfg = PlannerConfig::fast();
        assert_eq!(cfg.bmc_depth, 20);
        assert!(!cfg.envelope_computation);
    }

    #[test]
    fn test_planner_config_exhaustive() {
        let cfg = PlannerConfig::exhaustive();
        assert_eq!(cfg.bmc_depth, 100);
        assert!(cfg.pareto_front);
        assert!(cfg.pnr_analysis);
    }

    #[test]
    fn test_planner_config_validate() {
        let bad = PlannerConfig {
            bmc_depth: 0,
            ..Default::default()
        };
        assert!(bad.validate().is_err());
    }

    #[test]
    fn test_output_format() {
        assert_eq!(OutputFormat::from_str("json"), OutputFormat::Json);
        assert_eq!(OutputFormat::from_str("yaml"), OutputFormat::Yaml);
        assert_eq!(OutputFormat::from_str("yml"), OutputFormat::Yaml);
        assert_eq!(OutputFormat::from_str("txt"), OutputFormat::Text);
        assert_eq!(OutputFormat::from_str("dot"), OutputFormat::Dot);
        assert_eq!(OutputFormat::from_str("csv"), OutputFormat::Csv);
        assert_eq!(OutputFormat::from_str("unknown"), OutputFormat::Json);
    }

    #[test]
    fn test_output_format_extension() {
        assert_eq!(OutputFormat::Json.file_extension(), "json");
        assert_eq!(OutputFormat::Yaml.file_extension(), "yaml");
    }

    #[test]
    fn test_verbosity() {
        assert_eq!(Verbosity::from_level(0), Verbosity::Quiet);
        assert_eq!(Verbosity::from_level(1), Verbosity::Normal);
        assert_eq!(Verbosity::from_level(4), Verbosity::Trace);
        assert_eq!(Verbosity::from_level(99), Verbosity::Trace);

        assert!(Verbosity::Debug.is_at_least(Verbosity::Normal));
        assert!(!Verbosity::Normal.is_at_least(Verbosity::Debug));
    }

    #[test]
    fn test_k8s_config() {
        let cfg = K8sConfig::new()
            .with_namespace("production")
            .with_kubeconfig("/home/user/.kube/config")
            .with_dry_run();
        assert_eq!(cfg.effective_namespace(), "production");
        assert!(cfg.dry_run);
    }

    #[test]
    fn test_k8s_config_default_namespace() {
        let cfg = K8sConfig::default();
        assert_eq!(cfg.effective_namespace(), "default");
    }

    #[test]
    fn test_schema_config() {
        let cfg = SchemaConfig::default();
        assert_eq!(cfg.schema_source, SchemaSource::Registry);
        assert_eq!(cfg.compatibility_mode, CompatibilityMode::Backward);
        assert!(cfg.cache_schemas);
    }

    #[test]
    fn test_encoding_type_display() {
        assert_eq!(EncodingType::Interval.to_string(), "interval");
        assert_eq!(EncodingType::BDD.to_string(), "BDD");
    }

    #[test]
    fn test_solver_strategy_display() {
        assert_eq!(SolverStrategy::CDCL.to_string(), "CDCL");
        assert_eq!(SolverStrategy::Portfolio.to_string(), "Portfolio");
    }

    #[test]
    fn test_optimization_goal_display() {
        assert_eq!(OptimizationGoal::MinSteps.to_string(), "min-steps");
        assert_eq!(OptimizationGoal::Pareto.to_string(), "pareto");
    }

    #[test]
    fn test_exploration_strategy_display() {
        assert_eq!(ExplorationStrategy::BFS.to_string(), "BFS");
        assert_eq!(ExplorationStrategy::AStar.to_string(), "A*");
    }

    #[test]
    fn test_compatibility_mode_display() {
        assert_eq!(CompatibilityMode::Backward.to_string(), "backward");
        assert_eq!(CompatibilityMode::Full.to_string(), "full");
    }

    #[test]
    fn test_config_merge() {
        let mut base = SafeStepConfig::default_config();
        let override_cfg = SafeStepConfig {
            solver: SolverConfig {
                timeout_ms: 60_000,
                ..Default::default()
            },
            planner: PlannerConfig {
                bmc_depth: 75,
                ..Default::default()
            },
            ..Default::default()
        };
        base.merge_with(&override_cfg);
        assert_eq!(base.solver.timeout_ms, 60_000);
        assert_eq!(base.planner.bmc_depth, 75);
    }

    #[test]
    fn test_k8s_in_cluster() {
        let cfg = K8sConfig::in_cluster();
        assert!(cfg.kubeconfig_path.is_none());
        assert!(cfg.context.is_none());
    }

    #[test]
    fn test_full_config_validate() {
        let config = SafeStepConfig::default_config();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_from_invalid_json() {
        let result = SafeStepConfig::from_json("not json");
        assert!(result.is_err());
    }
}
