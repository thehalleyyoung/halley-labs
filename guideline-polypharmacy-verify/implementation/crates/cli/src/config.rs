//! Configuration management for GuardPharma CLI.
//!
//! Loads configuration from TOML files, environment variables, and CLI overrides.
//! Provides validation, default generation, and serialization.

use anyhow::{bail, Context, Result};
use log::{debug, info, warn};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::CliArgs;

// ─────────────────────────────── Error Types ─────────────────────────────

/// Errors specific to configuration loading and validation.
#[derive(Debug, Clone, thiserror::Error)]
pub enum ConfigError {
    #[error("Configuration file not found: {path}")]
    FileNotFound { path: String },

    #[error("Failed to parse configuration: {message}")]
    ParseError { message: String },

    #[error("Invalid configuration value for '{key}': {reason}")]
    InvalidValue { key: String, reason: String },

    #[error("Missing required configuration key: {key}")]
    MissingKey { key: String },

    #[error("Conflicting configuration: {message}")]
    Conflict { message: String },
}

// ────────────────────────────── App Configuration ────────────────────────

/// Top-level application configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    /// General application settings.
    #[serde(default)]
    pub general: GeneralConfig,

    /// Verification pipeline settings.
    #[serde(default)]
    pub verification: VerificationConfig,

    /// Pharmacokinetic model settings.
    #[serde(default)]
    pub pk_model: PkModelConfig,

    /// Clinical significance filter settings.
    #[serde(default)]
    pub significance: SignificanceConfig,

    /// SMT solver settings.
    #[serde(default)]
    pub smt: SmtConfig,

    /// Model checker settings.
    #[serde(default)]
    pub model_checker: ModelCheckerConfig,

    /// Output and reporting settings.
    #[serde(default)]
    pub output: OutputConfig,

    /// Drug database settings.
    #[serde(default)]
    pub database: DatabaseConfig,

    /// Evaluation benchmark settings.
    #[serde(default)]
    pub evaluation: EvaluationConfig,

    /// Logging settings.
    #[serde(default)]
    pub logging: LoggingConfig,

    /// Extra key-value overrides.
    #[serde(default)]
    pub extra: HashMap<String, String>,
}

impl Default for AppConfig {
    fn default() -> Self {
        AppConfig {
            general: GeneralConfig::default(),
            verification: VerificationConfig::default(),
            pk_model: PkModelConfig::default(),
            significance: SignificanceConfig::default(),
            smt: SmtConfig::default(),
            model_checker: ModelCheckerConfig::default(),
            output: OutputConfig::default(),
            database: DatabaseConfig::default(),
            evaluation: EvaluationConfig::default(),
            logging: LoggingConfig::default(),
            extra: HashMap::new(),
        }
    }
}

// ─────────────────────────── Section Configs ─────────────────────────────

/// General application settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneralConfig {
    /// Global timeout for any single operation (seconds).
    pub timeout_secs: u64,

    /// Maximum number of concurrent verification threads.
    pub max_threads: usize,

    /// Enable progress reporting.
    pub show_progress: bool,

    /// Enable colored terminal output.
    pub color: bool,

    /// Temporary directory for intermediate files.
    pub temp_dir: Option<PathBuf>,

    /// Working directory override.
    pub work_dir: Option<PathBuf>,
}

impl Default for GeneralConfig {
    fn default() -> Self {
        GeneralConfig {
            timeout_secs: 300,
            max_threads: 4,
            show_progress: true,
            color: true,
            temp_dir: None,
            work_dir: None,
        }
    }
}

/// Verification pipeline settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationConfig {
    /// Enable Tier 1 abstract interpretation.
    pub enable_tier1: bool,

    /// Enable Tier 2 model checking.
    pub enable_tier2: bool,

    /// Maximum number of drug pairs to check simultaneously.
    pub max_drug_pairs: usize,

    /// Tier 1 timeout per drug pair (seconds).
    pub tier1_timeout_secs: u64,

    /// Tier 2 timeout per drug pair (seconds).
    pub tier2_timeout_secs: u64,

    /// Whether to continue on individual pair failure.
    pub continue_on_error: bool,

    /// Minimum confidence threshold for reported conflicts.
    pub confidence_threshold: f64,

    /// Generate counterexample traces.
    pub generate_traces: bool,

    /// Maximum depth for state space exploration.
    pub max_exploration_depth: usize,

    /// Enable monotonicity optimization.
    pub use_monotonicity: bool,

    /// Enable symmetry reduction.
    pub use_symmetry_reduction: bool,

    /// Enable partial order reduction.
    pub use_partial_order_reduction: bool,
}

impl Default for VerificationConfig {
    fn default() -> Self {
        VerificationConfig {
            enable_tier1: true,
            enable_tier2: true,
            max_drug_pairs: 100,
            tier1_timeout_secs: 30,
            tier2_timeout_secs: 120,
            continue_on_error: true,
            confidence_threshold: 0.7,
            generate_traces: true,
            max_exploration_depth: 1000,
            use_monotonicity: true,
            use_symmetry_reduction: true,
            use_partial_order_reduction: true,
        }
    }
}

/// Pharmacokinetic model settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PkModelConfig {
    /// Default compartment model type (1, 2, or 3).
    pub default_compartments: u8,

    /// ODE solver step size (hours).
    pub ode_step_size: f64,

    /// ODE solver error tolerance.
    pub ode_tolerance: f64,

    /// Maximum simulation time (hours).
    pub max_simulation_hours: f64,

    /// Number of steady-state doses to simulate.
    pub steady_state_doses: usize,

    /// Enable population PK variability.
    pub enable_population_pk: bool,

    /// Number of Monte Carlo samples for population PK.
    pub monte_carlo_samples: usize,

    /// Interval arithmetic widening factor.
    pub widening_factor: f64,

    /// Enable CYP enzyme inhibition modeling.
    pub enable_cyp_modeling: bool,

    /// Enable protein binding displacement modeling.
    pub enable_protein_binding: bool,

    /// Precision for interval arithmetic (digits).
    pub interval_precision: u32,
}

impl Default for PkModelConfig {
    fn default() -> Self {
        PkModelConfig {
            default_compartments: 1,
            ode_step_size: 0.1,
            ode_tolerance: 1e-6,
            max_simulation_hours: 168.0,
            steady_state_doses: 10,
            enable_population_pk: false,
            monte_carlo_samples: 1000,
            widening_factor: 1.1,
            enable_cyp_modeling: true,
            enable_protein_binding: true,
            interval_precision: 6,
        }
    }
}

/// Clinical significance filter settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignificanceConfig {
    /// Weight for DrugBank evidence.
    pub drugbank_weight: f64,

    /// Weight for Beers Criteria.
    pub beers_weight: f64,

    /// Weight for FAERS signal data.
    pub faers_weight: f64,

    /// Weight for comorbidity scoring.
    pub comorbidity_weight: f64,

    /// Minimum severity threshold (0.0 – 1.0).
    pub min_severity_threshold: f64,

    /// Include Beers Criteria checking.
    pub include_beers: bool,

    /// Include FAERS signal analysis.
    pub include_faers: bool,

    /// Include comorbidity-weighted scoring.
    pub include_comorbidity: bool,

    /// Age threshold for elderly classification.
    pub elderly_age_threshold: f64,

    /// Minimum evidence level required (1 = highest, 5 = lowest).
    pub min_evidence_level: u8,
}

impl Default for SignificanceConfig {
    fn default() -> Self {
        SignificanceConfig {
            drugbank_weight: 0.35,
            beers_weight: 0.25,
            faers_weight: 0.20,
            comorbidity_weight: 0.20,
            min_severity_threshold: 0.3,
            include_beers: true,
            include_faers: true,
            include_comorbidity: true,
            elderly_age_threshold: 65.0,
            min_evidence_level: 3,
        }
    }
}

/// SMT solver settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmtConfig {
    /// SMT solver backend.
    pub solver: SmtSolverBackend,

    /// Solver timeout per query (seconds).
    pub query_timeout_secs: u64,

    /// Enable incremental solving.
    pub incremental: bool,

    /// Enable quantifier elimination.
    pub quantifier_elimination: bool,

    /// Logic to use (QF_NRA, QF_LRA, etc.).
    pub logic: String,

    /// Maximum number of refinement iterations.
    pub max_refinements: usize,

    /// Enable proof generation.
    pub generate_proofs: bool,

    /// Random seed for the solver.
    pub random_seed: Option<u64>,
}

impl Default for SmtConfig {
    fn default() -> Self {
        SmtConfig {
            solver: SmtSolverBackend::Internal,
            query_timeout_secs: 60,
            incremental: true,
            quantifier_elimination: false,
            logic: "QF_NRA".to_string(),
            max_refinements: 10,
            generate_proofs: false,
            random_seed: None,
        }
    }
}

/// SMT solver backend selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SmtSolverBackend {
    /// Built-in constraint solver.
    Internal,
    /// External Z3 solver.
    Z3,
    /// External CVC5 solver.
    Cvc5,
}

/// Model checker settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCheckerConfig {
    /// Maximum number of states to explore.
    pub max_states: usize,

    /// Enable on-the-fly verification.
    pub on_the_fly: bool,

    /// Compositional verification strategy.
    pub compositional: bool,

    /// Contract checking mode.
    pub contract_mode: ContractMode,

    /// Bounded model checking depth.
    pub bmc_depth: usize,

    /// Enable abstraction refinement (CEGAR).
    pub cegar: bool,

    /// Maximum CEGAR iterations.
    pub max_cegar_iterations: usize,

    /// Hash table size for state storage (entries).
    pub hash_table_size: usize,
}

impl Default for ModelCheckerConfig {
    fn default() -> Self {
        ModelCheckerConfig {
            max_states: 1_000_000,
            on_the_fly: true,
            compositional: true,
            contract_mode: ContractMode::AssumeGuarantee,
            bmc_depth: 50,
            cegar: false,
            max_cegar_iterations: 20,
            hash_table_size: 1 << 20,
        }
    }
}

/// Contract checking mode for compositional verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContractMode {
    /// Assume-guarantee reasoning.
    AssumeGuarantee,
    /// Rely-guarantee reasoning.
    RelyGuarantee,
    /// Direct product composition.
    DirectProduct,
}

/// Output and reporting settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    /// Include timing information in output.
    pub include_timing: bool,

    /// Include verification traces in output.
    pub include_traces: bool,

    /// Maximum number of conflicts to display.
    pub max_conflicts_shown: usize,

    /// Include enzyme pathway details.
    pub show_enzyme_details: bool,

    /// Include PK concentration curves.
    pub show_pk_curves: bool,

    /// Decimal precision for numeric output.
    pub decimal_places: usize,

    /// Include guideline rule references.
    pub include_rule_refs: bool,

    /// Certificate output format.
    pub certificate_format: CertificateFormat,

    /// Page width for text formatting.
    pub page_width: usize,
}

impl Default for OutputConfig {
    fn default() -> Self {
        OutputConfig {
            include_timing: true,
            include_traces: false,
            max_conflicts_shown: 50,
            show_enzyme_details: false,
            show_pk_curves: false,
            decimal_places: 4,
            include_rule_refs: true,
            certificate_format: CertificateFormat::Text,
            page_width: 80,
        }
    }
}

/// Safety certificate output format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CertificateFormat {
    Text,
    Json,
    Pdf,
}

/// Drug database settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    /// Path to external drug database file.
    pub drug_database_path: Option<PathBuf>,

    /// Path to external interaction database file.
    pub interaction_database_path: Option<PathBuf>,

    /// Use built-in drug database.
    pub use_builtin: bool,

    /// Path to custom Beers Criteria data.
    pub beers_criteria_path: Option<PathBuf>,

    /// Path to custom FAERS data.
    pub faers_data_path: Option<PathBuf>,

    /// Cache directory for downloaded data.
    pub cache_dir: Option<PathBuf>,

    /// Enable automatic database updates.
    pub auto_update: bool,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        DatabaseConfig {
            drug_database_path: None,
            interaction_database_path: None,
            use_builtin: true,
            beers_criteria_path: None,
            faers_data_path: None,
            cache_dir: None,
            auto_update: false,
        }
    }
}

/// Evaluation benchmark settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationConfig {
    /// Default benchmark dataset path.
    pub dataset_path: Option<PathBuf>,

    /// Number of parallel evaluation workers.
    pub workers: usize,

    /// Enable statistical analysis of results.
    pub statistical_analysis: bool,

    /// Confidence interval percentage.
    pub confidence_interval: f64,

    /// Generate comparison reports.
    pub comparison_reports: bool,

    /// Maximum test cases per suite.
    pub max_cases_per_suite: usize,

    /// Timeout per individual benchmark case (seconds).
    pub case_timeout_secs: u64,

    /// Output detailed per-case results.
    pub detailed_results: bool,
}

impl Default for EvaluationConfig {
    fn default() -> Self {
        EvaluationConfig {
            dataset_path: None,
            workers: 1,
            statistical_analysis: true,
            confidence_interval: 95.0,
            comparison_reports: false,
            max_cases_per_suite: 1000,
            case_timeout_secs: 60,
            detailed_results: false,
        }
    }
}

/// Logging configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level (error, warn, info, debug, trace).
    pub level: String,

    /// Log to file.
    pub log_file: Option<PathBuf>,

    /// Enable structured JSON logging.
    pub json_logging: bool,

    /// Include timestamps in log messages.
    pub timestamps: bool,

    /// Include module paths in log messages.
    pub module_paths: bool,

    /// Maximum log file size in MB before rotation.
    pub max_log_size_mb: usize,

    /// Number of rotated log files to keep.
    pub max_log_files: usize,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        LoggingConfig {
            level: "warn".to_string(),
            log_file: None,
            json_logging: false,
            timestamps: true,
            module_paths: false,
            max_log_size_mb: 50,
            max_log_files: 3,
        }
    }
}

// ──────────────────────────── Config Loading ─────────────────────────────

/// Configuration loader with environment variable and file support.
pub struct ConfigLoader {
    search_paths: Vec<PathBuf>,
    env_prefix: String,
}

impl ConfigLoader {
    /// Create a new config loader with default search paths.
    pub fn new() -> Self {
        let mut search_paths = Vec::new();

        // Current directory
        search_paths.push(PathBuf::from("guardpharma.toml"));
        search_paths.push(PathBuf::from(".guardpharma.toml"));

        // Home directory
        if let Ok(home) = std::env::var("HOME") {
            search_paths.push(PathBuf::from(home.clone()).join(".config/guardpharma/config.toml"));
            search_paths.push(PathBuf::from(home).join(".guardpharma.toml"));
        }

        // XDG config
        if let Ok(xdg) = std::env::var("XDG_CONFIG_HOME") {
            search_paths.push(PathBuf::from(xdg).join("guardpharma/config.toml"));
        }

        // System-wide
        search_paths.push(PathBuf::from("/etc/guardpharma/config.toml"));

        ConfigLoader {
            search_paths,
            env_prefix: "GUARDPHARMA".to_string(),
        }
    }

    /// Create a config loader with a custom environment prefix.
    pub fn with_prefix(prefix: &str) -> Self {
        let mut loader = ConfigLoader::new();
        loader.env_prefix = prefix.to_string();
        loader
    }

    /// Find the first existing config file from search paths.
    pub fn find_config_file(&self) -> Option<PathBuf> {
        for path in &self.search_paths {
            if path.exists() {
                debug!("Found config file at: {}", path.display());
                return Some(path.clone());
            }
        }
        None
    }

    /// Load configuration from the specified path.
    pub fn load_from_file(&self, path: &Path) -> Result<AppConfig> {
        if !path.exists() {
            bail!(ConfigError::FileNotFound {
                path: path.display().to_string(),
            });
        }

        let content = fs::read_to_string(path)
            .with_context(|| format!("Failed to read config file: {}", path.display()))?;

        let config: AppConfig = toml::from_str(&content).map_err(|e| {
            ConfigError::ParseError {
                message: format!("TOML parse error in {}: {}", path.display(), e),
            }
        })?;

        info!("Loaded configuration from {}", path.display());
        Ok(config)
    }

    /// Apply environment variable overrides to the configuration.
    pub fn apply_env_overrides(&self, config: &mut AppConfig) {
        let prefix = &self.env_prefix;

        if let Ok(val) = std::env::var(format!("{}_TIMEOUT", prefix)) {
            if let Ok(secs) = val.parse::<u64>() {
                debug!("Env override: timeout_secs = {}", secs);
                config.general.timeout_secs = secs;
            }
        }

        if let Ok(val) = std::env::var(format!("{}_LOG_LEVEL", prefix)) {
            debug!("Env override: log level = {}", val);
            config.logging.level = val;
        }

        if let Ok(val) = std::env::var(format!("{}_MAX_THREADS", prefix)) {
            if let Ok(n) = val.parse::<usize>() {
                debug!("Env override: max_threads = {}", n);
                config.general.max_threads = n;
            }
        }

        if let Ok(val) = std::env::var(format!("{}_COLOR", prefix)) {
            let enabled = matches!(val.to_lowercase().as_str(), "1" | "true" | "yes");
            debug!("Env override: color = {}", enabled);
            config.general.color = enabled;
        }

        if let Ok(val) = std::env::var(format!("{}_TIER1_TIMEOUT", prefix)) {
            if let Ok(secs) = val.parse::<u64>() {
                config.verification.tier1_timeout_secs = secs;
            }
        }

        if let Ok(val) = std::env::var(format!("{}_TIER2_TIMEOUT", prefix)) {
            if let Ok(secs) = val.parse::<u64>() {
                config.verification.tier2_timeout_secs = secs;
            }
        }

        if let Ok(val) = std::env::var(format!("{}_DRUG_DB", prefix)) {
            config.database.drug_database_path = Some(PathBuf::from(val));
        }

        if let Ok(val) = std::env::var(format!("{}_INTERACTION_DB", prefix)) {
            config.database.interaction_database_path = Some(PathBuf::from(val));
        }

        if let Ok(val) = std::env::var(format!("{}_SMT_SOLVER", prefix)) {
            match val.to_lowercase().as_str() {
                "z3" => config.smt.solver = SmtSolverBackend::Z3,
                "cvc5" => config.smt.solver = SmtSolverBackend::Cvc5,
                "internal" => config.smt.solver = SmtSolverBackend::Internal,
                _ => warn!("Unknown SMT solver backend: {}", val),
            }
        }

        if let Ok(val) = std::env::var(format!("{}_WORKERS", prefix)) {
            if let Ok(n) = val.parse::<usize>() {
                config.evaluation.workers = n;
            }
        }

        if let Ok(val) = std::env::var(format!("{}_CONFIDENCE_THRESHOLD", prefix)) {
            if let Ok(t) = val.parse::<f64>() {
                config.verification.confidence_threshold = t;
            }
        }

        if let Ok(val) = std::env::var(format!("{}_ODE_STEP", prefix)) {
            if let Ok(s) = val.parse::<f64>() {
                config.pk_model.ode_step_size = s;
            }
        }

        if let Ok(val) = std::env::var(format!("{}_MAX_STATES", prefix)) {
            if let Ok(n) = val.parse::<usize>() {
                config.model_checker.max_states = n;
            }
        }

        if let Ok(val) = std::env::var(format!("{}_TEMP_DIR", prefix)) {
            config.general.temp_dir = Some(PathBuf::from(val));
        }

        if let Ok(val) = std::env::var(format!("{}_CACHE_DIR", prefix)) {
            config.database.cache_dir = Some(PathBuf::from(val));
        }

        if let Ok(val) = std::env::var(format!("{}_PAGE_WIDTH", prefix)) {
            if let Ok(w) = val.parse::<usize>() {
                config.output.page_width = w;
            }
        }
    }

    /// Load config from auto-discovered file, or return defaults.
    pub fn load_auto(&self) -> Result<AppConfig> {
        let mut config = match self.find_config_file() {
            Some(path) => self.load_from_file(&path)?,
            None => {
                debug!("No config file found, using defaults");
                AppConfig::default()
            }
        };
        self.apply_env_overrides(&mut config);
        Ok(config)
    }
}

impl Default for ConfigLoader {
    fn default() -> Self {
        Self::new()
    }
}

// ──────────────────────────── Public API ─────────────────────────────────

/// Load configuration from the given path, or auto-discover.
pub fn load_config(path: Option<&Path>) -> Result<AppConfig> {
    let loader = ConfigLoader::new();

    match path {
        Some(p) => {
            let mut config = loader.load_from_file(p)?;
            loader.apply_env_overrides(&mut config);
            validate_config(&config)?;
            Ok(config)
        }
        None => {
            let config = loader.load_auto()?;
            validate_config(&config)?;
            Ok(config)
        }
    }
}

/// Merge CLI arguments into an existing configuration.
pub fn merge_with_cli(config: &mut AppConfig, cli: &CliArgs) {
    if let Some(timeout) = cli.timeout {
        config.general.timeout_secs = timeout;
        config.verification.tier1_timeout_secs = timeout.min(config.verification.tier1_timeout_secs);
        config.verification.tier2_timeout_secs = timeout;
    }

    if cli.no_color {
        config.general.color = false;
    }

    if let Some(ref level) = cli.log_level {
        config.logging.level = level.clone();
    }

    match cli.verbose {
        0 => {}
        1 => config.logging.level = "info".to_string(),
        2 => config.logging.level = "debug".to_string(),
        _ => config.logging.level = "trace".to_string(),
    }

    if cli.quiet {
        config.logging.level = "error".to_string();
        config.general.show_progress = false;
    }
}

/// Validate configuration values for correctness.
pub fn validate_config(config: &AppConfig) -> Result<()> {
    // Timeouts must be positive
    if config.general.timeout_secs == 0 {
        bail!(ConfigError::InvalidValue {
            key: "general.timeout_secs".to_string(),
            reason: "Timeout must be greater than 0".to_string(),
        });
    }

    if config.general.max_threads == 0 {
        bail!(ConfigError::InvalidValue {
            key: "general.max_threads".to_string(),
            reason: "Max threads must be greater than 0".to_string(),
        });
    }

    // PK model validation
    if config.pk_model.ode_step_size <= 0.0 {
        bail!(ConfigError::InvalidValue {
            key: "pk_model.ode_step_size".to_string(),
            reason: "ODE step size must be positive".to_string(),
        });
    }

    if config.pk_model.ode_tolerance <= 0.0 {
        bail!(ConfigError::InvalidValue {
            key: "pk_model.ode_tolerance".to_string(),
            reason: "ODE tolerance must be positive".to_string(),
        });
    }

    if config.pk_model.max_simulation_hours <= 0.0 {
        bail!(ConfigError::InvalidValue {
            key: "pk_model.max_simulation_hours".to_string(),
            reason: "Max simulation hours must be positive".to_string(),
        });
    }

    if !(1..=3).contains(&config.pk_model.default_compartments) {
        bail!(ConfigError::InvalidValue {
            key: "pk_model.default_compartments".to_string(),
            reason: "Compartment count must be 1, 2, or 3".to_string(),
        });
    }

    // Significance weights must sum approximately to 1.0
    let total_weight = config.significance.drugbank_weight
        + config.significance.beers_weight
        + config.significance.faers_weight
        + config.significance.comorbidity_weight;
    if (total_weight - 1.0).abs() > 0.01 {
        warn!(
            "Significance weights sum to {:.4} (expected 1.0); results may be skewed",
            total_weight
        );
    }

    if config.significance.min_severity_threshold < 0.0
        || config.significance.min_severity_threshold > 1.0
    {
        bail!(ConfigError::InvalidValue {
            key: "significance.min_severity_threshold".to_string(),
            reason: "Must be between 0.0 and 1.0".to_string(),
        });
    }

    // Confidence threshold
    if config.verification.confidence_threshold < 0.0
        || config.verification.confidence_threshold > 1.0
    {
        bail!(ConfigError::InvalidValue {
            key: "verification.confidence_threshold".to_string(),
            reason: "Must be between 0.0 and 1.0".to_string(),
        });
    }

    // SMT logic
    let valid_logics = ["QF_NRA", "QF_LRA", "QF_NIA", "QF_LIA", "QF_NIRA", "ALL"];
    if !valid_logics.contains(&config.smt.logic.as_str()) {
        warn!("Non-standard SMT logic: {}; this may not be supported", config.smt.logic);
    }

    // Evaluation
    if config.evaluation.confidence_interval < 50.0
        || config.evaluation.confidence_interval > 99.99
    {
        bail!(ConfigError::InvalidValue {
            key: "evaluation.confidence_interval".to_string(),
            reason: "Must be between 50.0 and 99.99".to_string(),
        });
    }

    // Output
    if config.output.page_width < 40 {
        bail!(ConfigError::InvalidValue {
            key: "output.page_width".to_string(),
            reason: "Page width must be at least 40".to_string(),
        });
    }

    if config.output.decimal_places > 15 {
        bail!(ConfigError::InvalidValue {
            key: "output.decimal_places".to_string(),
            reason: "Decimal places must be at most 15".to_string(),
        });
    }

    // Model checker
    if config.model_checker.max_states == 0 {
        bail!(ConfigError::InvalidValue {
            key: "model_checker.max_states".to_string(),
            reason: "Must be greater than 0".to_string(),
        });
    }

    Ok(())
}

/// Generate default configuration as a TOML string.
pub fn default_config_toml() -> String {
    let config = AppConfig::default();
    let mut output = String::with_capacity(4096);

    output.push_str("# GuardPharma Configuration File\n");
    output.push_str("# Generated by guardpharma init\n\n");

    output.push_str("[general]\n");
    output.push_str(&format!("timeout_secs = {}\n", config.general.timeout_secs));
    output.push_str(&format!("max_threads = {}\n", config.general.max_threads));
    output.push_str(&format!("show_progress = {}\n", config.general.show_progress));
    output.push_str(&format!("color = {}\n", config.general.color));
    output.push_str("# temp_dir = \"/tmp/guardpharma\"\n");
    output.push_str("# work_dir = \".\"\n\n");

    output.push_str("[verification]\n");
    output.push_str(&format!("enable_tier1 = {}\n", config.verification.enable_tier1));
    output.push_str(&format!("enable_tier2 = {}\n", config.verification.enable_tier2));
    output.push_str(&format!("max_drug_pairs = {}\n", config.verification.max_drug_pairs));
    output.push_str(&format!("tier1_timeout_secs = {}\n", config.verification.tier1_timeout_secs));
    output.push_str(&format!("tier2_timeout_secs = {}\n", config.verification.tier2_timeout_secs));
    output.push_str(&format!("continue_on_error = {}\n", config.verification.continue_on_error));
    output.push_str(&format!("confidence_threshold = {}\n", config.verification.confidence_threshold));
    output.push_str(&format!("generate_traces = {}\n", config.verification.generate_traces));
    output.push_str(&format!("max_exploration_depth = {}\n", config.verification.max_exploration_depth));
    output.push_str(&format!("use_monotonicity = {}\n", config.verification.use_monotonicity));
    output.push_str(&format!("use_symmetry_reduction = {}\n", config.verification.use_symmetry_reduction));
    output.push_str(&format!("use_partial_order_reduction = {}\n", config.verification.use_partial_order_reduction));
    output.push('\n');

    output.push_str("[pk_model]\n");
    output.push_str(&format!("default_compartments = {}\n", config.pk_model.default_compartments));
    output.push_str(&format!("ode_step_size = {}\n", config.pk_model.ode_step_size));
    output.push_str(&format!("ode_tolerance = {}\n", config.pk_model.ode_tolerance));
    output.push_str(&format!("max_simulation_hours = {}\n", config.pk_model.max_simulation_hours));
    output.push_str(&format!("steady_state_doses = {}\n", config.pk_model.steady_state_doses));
    output.push_str(&format!("enable_population_pk = {}\n", config.pk_model.enable_population_pk));
    output.push_str(&format!("monte_carlo_samples = {}\n", config.pk_model.monte_carlo_samples));
    output.push_str(&format!("widening_factor = {}\n", config.pk_model.widening_factor));
    output.push_str(&format!("enable_cyp_modeling = {}\n", config.pk_model.enable_cyp_modeling));
    output.push_str(&format!("enable_protein_binding = {}\n", config.pk_model.enable_protein_binding));
    output.push_str(&format!("interval_precision = {}\n", config.pk_model.interval_precision));
    output.push('\n');

    output.push_str("[significance]\n");
    output.push_str(&format!("drugbank_weight = {}\n", config.significance.drugbank_weight));
    output.push_str(&format!("beers_weight = {}\n", config.significance.beers_weight));
    output.push_str(&format!("faers_weight = {}\n", config.significance.faers_weight));
    output.push_str(&format!("comorbidity_weight = {}\n", config.significance.comorbidity_weight));
    output.push_str(&format!("min_severity_threshold = {}\n", config.significance.min_severity_threshold));
    output.push_str(&format!("include_beers = {}\n", config.significance.include_beers));
    output.push_str(&format!("include_faers = {}\n", config.significance.include_faers));
    output.push_str(&format!("include_comorbidity = {}\n", config.significance.include_comorbidity));
    output.push_str(&format!("elderly_age_threshold = {}\n", config.significance.elderly_age_threshold));
    output.push_str(&format!("min_evidence_level = {}\n", config.significance.min_evidence_level));
    output.push('\n');

    output.push_str("[smt]\n");
    output.push_str("solver = \"Internal\"\n");
    output.push_str(&format!("query_timeout_secs = {}\n", config.smt.query_timeout_secs));
    output.push_str(&format!("incremental = {}\n", config.smt.incremental));
    output.push_str(&format!("quantifier_elimination = {}\n", config.smt.quantifier_elimination));
    output.push_str(&format!("logic = \"{}\"\n", config.smt.logic));
    output.push_str(&format!("max_refinements = {}\n", config.smt.max_refinements));
    output.push_str(&format!("generate_proofs = {}\n", config.smt.generate_proofs));
    output.push_str("# random_seed = 42\n\n");

    output.push_str("[model_checker]\n");
    output.push_str(&format!("max_states = {}\n", config.model_checker.max_states));
    output.push_str(&format!("on_the_fly = {}\n", config.model_checker.on_the_fly));
    output.push_str(&format!("compositional = {}\n", config.model_checker.compositional));
    output.push_str("contract_mode = \"AssumeGuarantee\"\n");
    output.push_str(&format!("bmc_depth = {}\n", config.model_checker.bmc_depth));
    output.push_str(&format!("cegar = {}\n", config.model_checker.cegar));
    output.push_str(&format!("max_cegar_iterations = {}\n", config.model_checker.max_cegar_iterations));
    output.push_str(&format!("hash_table_size = {}\n", config.model_checker.hash_table_size));
    output.push('\n');

    output.push_str("[output]\n");
    output.push_str(&format!("include_timing = {}\n", config.output.include_timing));
    output.push_str(&format!("include_traces = {}\n", config.output.include_traces));
    output.push_str(&format!("max_conflicts_shown = {}\n", config.output.max_conflicts_shown));
    output.push_str(&format!("show_enzyme_details = {}\n", config.output.show_enzyme_details));
    output.push_str(&format!("show_pk_curves = {}\n", config.output.show_pk_curves));
    output.push_str(&format!("decimal_places = {}\n", config.output.decimal_places));
    output.push_str(&format!("include_rule_refs = {}\n", config.output.include_rule_refs));
    output.push_str("certificate_format = \"Text\"\n");
    output.push_str(&format!("page_width = {}\n", config.output.page_width));
    output.push('\n');

    output.push_str("[database]\n");
    output.push_str(&format!("use_builtin = {}\n", config.database.use_builtin));
    output.push_str("# drug_database_path = \"drugs.json\"\n");
    output.push_str("# interaction_database_path = \"interactions.json\"\n");
    output.push_str("# beers_criteria_path = \"beers.json\"\n");
    output.push_str("# faers_data_path = \"faers.csv\"\n");
    output.push_str("# cache_dir = \"~/.cache/guardpharma\"\n");
    output.push_str(&format!("auto_update = {}\n", config.database.auto_update));
    output.push('\n');

    output.push_str("[evaluation]\n");
    output.push_str("# dataset_path = \"benchmarks/\"\n");
    output.push_str(&format!("workers = {}\n", config.evaluation.workers));
    output.push_str(&format!("statistical_analysis = {}\n", config.evaluation.statistical_analysis));
    output.push_str(&format!("confidence_interval = {}\n", config.evaluation.confidence_interval));
    output.push_str(&format!("comparison_reports = {}\n", config.evaluation.comparison_reports));
    output.push_str(&format!("max_cases_per_suite = {}\n", config.evaluation.max_cases_per_suite));
    output.push_str(&format!("case_timeout_secs = {}\n", config.evaluation.case_timeout_secs));
    output.push_str(&format!("detailed_results = {}\n", config.evaluation.detailed_results));
    output.push('\n');

    output.push_str("[logging]\n");
    output.push_str(&format!("level = \"{}\"\n", config.logging.level));
    output.push_str("# log_file = \"guardpharma.log\"\n");
    output.push_str(&format!("json_logging = {}\n", config.logging.json_logging));
    output.push_str(&format!("timestamps = {}\n", config.logging.timestamps));
    output.push_str(&format!("module_paths = {}\n", config.logging.module_paths));
    output.push_str(&format!("max_log_size_mb = {}\n", config.logging.max_log_size_mb));
    output.push_str(&format!("max_log_files = {}\n", config.logging.max_log_files));

    output
}

/// Write a default configuration file to disk.
pub fn write_default_config(path: &Path, force: bool) -> Result<()> {
    if path.exists() && !force {
        bail!(
            "Configuration file already exists at {}. Use --force to overwrite.",
            path.display()
        );
    }

    let content = default_config_toml();
    fs::write(path, &content)
        .with_context(|| format!("Failed to write config to {}", path.display()))?;

    info!("Wrote default configuration to {}", path.display());
    Ok(())
}

/// Map of environment variables to configuration keys for documentation.
pub fn env_var_mapping() -> Vec<(&'static str, &'static str)> {
    vec![
        ("GUARDPHARMA_TIMEOUT", "general.timeout_secs"),
        ("GUARDPHARMA_LOG_LEVEL", "logging.level"),
        ("GUARDPHARMA_MAX_THREADS", "general.max_threads"),
        ("GUARDPHARMA_COLOR", "general.color"),
        ("GUARDPHARMA_TIER1_TIMEOUT", "verification.tier1_timeout_secs"),
        ("GUARDPHARMA_TIER2_TIMEOUT", "verification.tier2_timeout_secs"),
        ("GUARDPHARMA_DRUG_DB", "database.drug_database_path"),
        ("GUARDPHARMA_INTERACTION_DB", "database.interaction_database_path"),
        ("GUARDPHARMA_SMT_SOLVER", "smt.solver"),
        ("GUARDPHARMA_WORKERS", "evaluation.workers"),
        ("GUARDPHARMA_CONFIDENCE_THRESHOLD", "verification.confidence_threshold"),
        ("GUARDPHARMA_ODE_STEP", "pk_model.ode_step_size"),
        ("GUARDPHARMA_MAX_STATES", "model_checker.max_states"),
        ("GUARDPHARMA_TEMP_DIR", "general.temp_dir"),
        ("GUARDPHARMA_CACHE_DIR", "database.cache_dir"),
        ("GUARDPHARMA_PAGE_WIDTH", "output.page_width"),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_default_config_valid() {
        let config = AppConfig::default();
        assert!(validate_config(&config).is_ok());
    }

    #[test]
    fn test_default_config_roundtrip() {
        let config = AppConfig::default();
        let toml_str = toml::to_string_pretty(&config).unwrap();
        let parsed: AppConfig = toml::from_str(&toml_str).unwrap();
        assert_eq!(parsed.general.timeout_secs, config.general.timeout_secs);
        assert_eq!(parsed.verification.enable_tier1, config.verification.enable_tier1);
        assert_eq!(parsed.pk_model.default_compartments, config.pk_model.default_compartments);
    }

    #[test]
    fn test_invalid_timeout_zero() {
        let mut config = AppConfig::default();
        config.general.timeout_secs = 0;
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn test_invalid_max_threads_zero() {
        let mut config = AppConfig::default();
        config.general.max_threads = 0;
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn test_invalid_ode_step_negative() {
        let mut config = AppConfig::default();
        config.pk_model.ode_step_size = -0.1;
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn test_invalid_compartments() {
        let mut config = AppConfig::default();
        config.pk_model.default_compartments = 5;
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn test_invalid_severity_threshold() {
        let mut config = AppConfig::default();
        config.significance.min_severity_threshold = 1.5;
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn test_invalid_confidence_threshold() {
        let mut config = AppConfig::default();
        config.verification.confidence_threshold = -0.1;
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn test_invalid_page_width() {
        let mut config = AppConfig::default();
        config.output.page_width = 10;
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn test_invalid_decimal_places() {
        let mut config = AppConfig::default();
        config.output.decimal_places = 20;
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn test_invalid_confidence_interval() {
        let mut config = AppConfig::default();
        config.evaluation.confidence_interval = 30.0;
        assert!(validate_config(&config).is_err());
    }

    #[test]
    fn test_load_from_toml_string() {
        let toml_str = r#"
[general]
timeout_secs = 600
max_threads = 8
show_progress = false
color = true

[verification]
enable_tier1 = true
enable_tier2 = false
max_drug_pairs = 50
"#;
        let config: AppConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.general.timeout_secs, 600);
        assert_eq!(config.general.max_threads, 8);
        assert!(!config.general.show_progress);
        assert!(!config.verification.enable_tier2);
        assert_eq!(config.verification.max_drug_pairs, 50);
    }

    #[test]
    fn test_partial_toml_uses_defaults() {
        let toml_str = r#"
[general]
timeout_secs = 100
"#;
        let config: AppConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.general.timeout_secs, 100);
        // Other sections should use defaults
        assert!(config.verification.enable_tier1);
        assert!(config.verification.enable_tier2);
        assert_eq!(config.pk_model.default_compartments, 1);
    }

    #[test]
    fn test_default_config_toml_generation() {
        let toml_str = default_config_toml();
        assert!(toml_str.contains("[general]"));
        assert!(toml_str.contains("[verification]"));
        assert!(toml_str.contains("[pk_model]"));
        assert!(toml_str.contains("[significance]"));
        assert!(toml_str.contains("[smt]"));
        assert!(toml_str.contains("[model_checker]"));
        assert!(toml_str.contains("[output]"));
        assert!(toml_str.contains("[database]"));
        assert!(toml_str.contains("[evaluation]"));
        assert!(toml_str.contains("[logging]"));
    }

    #[test]
    fn test_write_default_config() {
        let dir = std::env::temp_dir().join("guardpharma_test_config");
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("test_config.toml");
        let _ = fs::remove_file(&path);

        write_default_config(&path, false).unwrap();
        assert!(path.exists());

        // Should fail without force
        assert!(write_default_config(&path, false).is_err());

        // Should succeed with force
        write_default_config(&path, true).unwrap();

        let _ = fs::remove_file(&path);
        let _ = fs::remove_dir(&dir);
    }

    #[test]
    fn test_config_loader_new() {
        let loader = ConfigLoader::new();
        assert!(!loader.search_paths.is_empty());
        assert_eq!(loader.env_prefix, "GUARDPHARMA");
    }

    #[test]
    fn test_config_loader_with_prefix() {
        let loader = ConfigLoader::with_prefix("GP_TEST");
        assert_eq!(loader.env_prefix, "GP_TEST");
    }

    #[test]
    fn test_load_from_file() {
        let dir = std::env::temp_dir().join("guardpharma_test_load");
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("test_load.toml");

        let mut file = fs::File::create(&path).unwrap();
        writeln!(file, "[general]\ntimeout_secs = 999").unwrap();

        let loader = ConfigLoader::new();
        let config = loader.load_from_file(&path).unwrap();
        assert_eq!(config.general.timeout_secs, 999);

        let _ = fs::remove_file(&path);
        let _ = fs::remove_dir(&dir);
    }

    #[test]
    fn test_load_nonexistent_file() {
        let loader = ConfigLoader::new();
        assert!(loader.load_from_file(Path::new("/nonexistent/config.toml")).is_err());
    }

    #[test]
    fn test_env_var_mapping_not_empty() {
        let mapping = env_var_mapping();
        assert!(!mapping.is_empty());
        for (env, key) in &mapping {
            assert!(env.starts_with("GUARDPHARMA_"));
            assert!(key.contains('.'));
        }
    }

    #[test]
    fn test_smt_solver_backend_serialization() {
        let config = SmtConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("Internal"));
    }

    #[test]
    fn test_contract_mode_serialization() {
        let config = ModelCheckerConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("AssumeGuarantee"));
    }

    #[test]
    fn test_significance_weights_sum() {
        let config = SignificanceConfig::default();
        let total = config.drugbank_weight
            + config.beers_weight
            + config.faers_weight
            + config.comorbidity_weight;
        assert!((total - 1.0).abs() < 1e-10);
    }
}
