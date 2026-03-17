//! CLI command definitions for the CollusionProof system.
//!
//! Uses clap derive macros for ergonomic argument parsing with full help text,
//! defaults, and value validation for every subcommand.

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand, ValueEnum};

// ── Top-level CLI ───────────────────────────────────────────────────────────

/// CollusionProof: Algorithmic Collusion Certification System
///
/// A formally-grounded framework for detecting, certifying, and evaluating
/// algorithmic collusion in repeated games. Supports tiered oracle access
/// (Layer 0/1/2), statistical hypothesis testing with FWER control,
/// counterfactual analysis, and proof-carrying certificates.
#[derive(Debug, Parser)]
#[command(
    name = "collusion-proof",
    version,
    author = "CollusionProof Team",
    about = "Algorithmic Collusion Certification System",
    long_about = "A formally-grounded framework for detecting, certifying, and evaluating \
                  algorithmic collusion in repeated games.\n\n\
                  Supports tiered oracle access (Layer 0/1/2), statistical hypothesis testing \
                  with FWER control, counterfactual analysis, and proof-carrying certificates.",
    propagate_version = true,
    arg_required_else_help = true,
)]
pub struct CollusionProofCli {
    #[command(subcommand)]
    pub command: Command,

    /// Suppress all output except errors
    #[arg(long, short = 'q', global = true)]
    pub quiet: bool,

    /// Increase verbosity (-v for verbose, -vv for debug)
    #[arg(long, short = 'v', global = true, action = clap::ArgAction::Count)]
    pub verbose: u8,

    /// Override config file path
    #[arg(long, global = true)]
    pub config: Option<PathBuf>,

    /// Disable colored output
    #[arg(long, global = true)]
    pub no_color: bool,

    /// Output format
    #[arg(long, global = true, default_value = "text", value_enum)]
    pub format: OutputFormat,
}

/// Top-level subcommands.
#[derive(Debug, Subcommand)]
pub enum Command {
    /// Run the full certification pipeline on a scenario
    Run(RunArgs),

    /// Run market simulation only
    Simulate(SimulateArgs),

    /// Run statistical analysis on trajectory data
    Analyze(AnalyzeArgs),

    /// Generate a certificate from analysis results
    Certify(CertifyArgs),

    /// Verify an existing certificate
    Verify(VerifyArgs),

    /// Run the evaluation benchmark suite
    Evaluate(EvaluateArgs),

    /// List available scenarios
    Scenarios(ScenariosArgs),

    /// Show or generate configuration
    Config(ConfigArgs),
}

// ── Output format ───────────────────────────────────────────────────────────

/// Output format for CLI results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum OutputFormat {
    /// Human-readable text
    Text,
    /// JSON output
    Json,
    /// Tabular output
    Table,
}

impl Default for OutputFormat {
    fn default() -> Self {
        OutputFormat::Text
    }
}

// ── Oracle level arg ────────────────────────────────────────────────────────

/// Oracle access level for the detection pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum OracleLevelArg {
    /// Layer 0: Price trajectories only (black-box)
    #[value(alias = "0")]
    Layer0,
    /// Layer 1: Price trajectories + unilateral deviation results
    #[value(alias = "1")]
    Layer1,
    /// Layer 2: Full oracle access including punishment detection
    #[value(alias = "2")]
    Layer2,
}

impl Default for OracleLevelArg {
    fn default() -> Self {
        OracleLevelArg::Layer0
    }
}

impl std::fmt::Display for OracleLevelArg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OracleLevelArg::Layer0 => write!(f, "Layer0"),
            OracleLevelArg::Layer1 => write!(f, "Layer1"),
            OracleLevelArg::Layer2 => write!(f, "Layer2"),
        }
    }
}

// ── Market type arg ─────────────────────────────────────────────────────────

/// Market competition type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum MarketTypeArg {
    /// Bertrand (price competition)
    Bertrand,
    /// Cournot (quantity competition)
    Cournot,
}

impl Default for MarketTypeArg {
    fn default() -> Self {
        MarketTypeArg::Bertrand
    }
}

impl std::fmt::Display for MarketTypeArg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MarketTypeArg::Bertrand => write!(f, "Bertrand"),
            MarketTypeArg::Cournot => write!(f, "Cournot"),
        }
    }
}

// ── Demand system arg ───────────────────────────────────────────────────────

/// Demand model specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum DemandModelArg {
    /// Linear demand: q = a - b*p + c*p_avg
    Linear,
    /// Logit demand: multinomial logit choice model
    Logit,
    /// CES demand: constant elasticity of substitution
    Ces,
}

impl Default for DemandModelArg {
    fn default() -> Self {
        DemandModelArg::Linear
    }
}

impl std::fmt::Display for DemandModelArg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DemandModelArg::Linear => write!(f, "Linear"),
            DemandModelArg::Logit => write!(f, "Logit"),
            DemandModelArg::Ces => write!(f, "CES"),
        }
    }
}

// ── Algorithm type arg ──────────────────────────────────────────────────────

/// Pricing algorithm specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum AlgorithmArg {
    /// Q-Learning agent
    QLearning,
    /// Deep Q-Network agent
    Dqn,
    /// Grim Trigger strategy
    GrimTrigger,
    /// Tit-for-Tat strategy
    TitForTat,
    /// Multi-armed bandit
    Bandit,
    /// Nash equilibrium (competitive baseline)
    Nash,
    /// Myopic best response (competitive baseline)
    Myopic,
}

impl Default for AlgorithmArg {
    fn default() -> Self {
        AlgorithmArg::QLearning
    }
}

impl std::fmt::Display for AlgorithmArg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AlgorithmArg::QLearning => write!(f, "Q-Learning"),
            AlgorithmArg::Dqn => write!(f, "DQN"),
            AlgorithmArg::GrimTrigger => write!(f, "Grim Trigger"),
            AlgorithmArg::TitForTat => write!(f, "Tit-for-Tat"),
            AlgorithmArg::Bandit => write!(f, "Bandit"),
            AlgorithmArg::Nash => write!(f, "Nash Equilibrium"),
            AlgorithmArg::Myopic => write!(f, "Myopic Best Response"),
        }
    }
}

// ── Evaluation mode arg ─────────────────────────────────────────────────────

/// Evaluation suite mode controlling scope and thoroughness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum EvalModeArg {
    /// Quick smoke test: 5 scenarios, 2 seeds (~30s)
    Smoke,
    /// Standard evaluation: 15 scenarios, 5 seeds (~5min)
    Standard,
    /// Full evaluation: 30 scenarios, 10 seeds (~30min)
    Full,
}

impl Default for EvalModeArg {
    fn default() -> Self {
        EvalModeArg::Standard
    }
}

impl std::fmt::Display for EvalModeArg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EvalModeArg::Smoke => write!(f, "Smoke"),
            EvalModeArg::Standard => write!(f, "Standard"),
            EvalModeArg::Full => write!(f, "Full"),
        }
    }
}

// ── Null tier arg ───────────────────────────────────────────────────────────

/// Null hypothesis tier for statistical testing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum NullTierArg {
    /// Simple independence null
    Simple,
    /// Composite tiered null with multiple levels
    Tiered,
    /// Calibrated competitive null from simulation
    Calibrated,
}

impl Default for NullTierArg {
    fn default() -> Self {
        NullTierArg::Tiered
    }
}

impl std::fmt::Display for NullTierArg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NullTierArg::Simple => write!(f, "Simple"),
            NullTierArg::Tiered => write!(f, "Tiered"),
            NullTierArg::Calibrated => write!(f, "Calibrated"),
        }
    }
}

// ── Config action ───────────────────────────────────────────────────────────

/// Configuration management subcommand action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum ConfigAction {
    /// Show the current effective configuration
    Show,
    /// Generate a default configuration file
    Generate,
    /// Validate configuration
    Validate,
}

impl Default for ConfigAction {
    fn default() -> Self {
        ConfigAction::Show
    }
}

// ── Run subcommand ──────────────────────────────────────────────────────────

/// Run the full certification pipeline on a scenario.
///
/// Executes the complete pipeline: simulation → segmentation → statistical
/// testing → deviation analysis → punishment detection → collusion premium →
/// certificate generation → verification.
#[derive(Debug, Args)]
pub struct RunArgs {
    /// Scenario identifier (e.g., "bertrand_qlearning_2p")
    #[arg(
        long,
        short = 's',
        help = "Scenario to run (use 'scenarios' command to list available)",
        required_unless_present = "all"
    )]
    pub scenario: Option<String>,

    /// Run all available scenarios
    #[arg(long, conflicts_with = "scenario")]
    pub all: bool,

    /// Oracle access level
    #[arg(long, short = 'l', default_value = "layer0", value_enum)]
    pub oracle_level: OracleLevelArg,

    /// Significance level (alpha) for hypothesis tests
    #[arg(
        long,
        short = 'a',
        default_value = "0.05",
        value_parser = validate_alpha,
        help = "Significance level (0 < alpha < 1)"
    )]
    pub alpha: f64,

    /// Output directory for results and certificates
    #[arg(long, short = 'o', default_value = "./output")]
    pub output_dir: PathBuf,

    /// Random seed for reproducibility
    #[arg(long, default_value = "42")]
    pub seed: u64,

    /// Number of simulation rounds
    #[arg(
        long,
        short = 'r',
        default_value = "1000",
        value_parser = validate_rounds,
        help = "Number of rounds (10-1000000)"
    )]
    pub rounds: usize,

    /// Evaluation mode for data splitting
    #[arg(long, default_value = "testing", value_enum)]
    pub evaluation_mode: EvalModeDataArg,

    /// Number of bootstrap resamples
    #[arg(long, default_value = "1000")]
    pub bootstrap_resamples: usize,

    /// Number of parallel workers (0 = auto-detect)
    #[arg(long, short = 'j', default_value = "0")]
    pub jobs: usize,

    /// Enable checkpointing for resume support
    #[arg(long)]
    pub checkpoint: bool,

    /// Checkpoint directory
    #[arg(long, default_value = "./.checkpoints")]
    pub checkpoint_dir: PathBuf,

    /// Skip certificate verification step
    #[arg(long)]
    pub skip_verify: bool,

    /// Save intermediate results
    #[arg(long)]
    pub save_intermediates: bool,
}

/// Evaluation mode for data splitting during pipeline runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum EvalModeDataArg {
    /// Training set
    Training,
    /// Testing set
    Testing,
    /// Validation set
    Validation,
    /// Holdout set
    Holdout,
}

impl Default for EvalModeDataArg {
    fn default() -> Self {
        EvalModeDataArg::Testing
    }
}

// ── Simulate subcommand ─────────────────────────────────────────────────────

/// Run a market simulation and produce a price trajectory.
///
/// Generates a trajectory by running the specified pricing algorithm(s)
/// in the given market environment. Output can be used as input to the
/// 'analyze' command.
#[derive(Debug, Args)]
pub struct SimulateArgs {
    /// Market competition type
    #[arg(long, short = 'm', default_value = "bertrand", value_enum)]
    pub market_type: MarketTypeArg,

    /// Demand model
    #[arg(long, short = 'd', default_value = "linear", value_enum)]
    pub demand: DemandModelArg,

    /// Demand intercept (for linear model)
    #[arg(long, default_value = "10.0")]
    pub demand_intercept: f64,

    /// Demand slope (for linear model)
    #[arg(long, default_value = "1.0")]
    pub demand_slope: f64,

    /// Cross-price elasticity slope (for linear model)
    #[arg(long, default_value = "0.5")]
    pub cross_slope: f64,

    /// Logit mu parameter
    #[arg(long, default_value = "0.25")]
    pub logit_mu: f64,

    /// CES sigma parameter
    #[arg(long, default_value = "2.0")]
    pub ces_sigma: f64,

    /// Number of players
    #[arg(
        long,
        short = 'n',
        default_value = "2",
        value_parser = validate_players,
        help = "Number of players (2-20)"
    )]
    pub num_players: usize,

    /// Pricing algorithm
    #[arg(long, short = 'A', default_value = "q-learning", value_enum)]
    pub algorithm: AlgorithmArg,

    /// Number of simulation rounds
    #[arg(
        long,
        short = 'r',
        default_value = "1000",
        value_parser = validate_rounds,
    )]
    pub rounds: usize,

    /// Output file for trajectory data
    #[arg(long, short = 'o', default_value = "trajectory.json")]
    pub output: PathBuf,

    /// Random seed
    #[arg(long, default_value = "42")]
    pub seed: u64,

    /// Marginal cost per unit
    #[arg(long, default_value = "1.0")]
    pub marginal_cost: f64,

    /// Lower price bound
    #[arg(long, default_value = "0.0")]
    pub price_min: f64,

    /// Upper price bound
    #[arg(long, default_value = "20.0")]
    pub price_max: f64,
}

// ── Analyze subcommand ──────────────────────────────────────────────────────

/// Run statistical analysis on a price trajectory.
///
/// Performs hypothesis testing at the specified null tier and oracle level.
/// Can process trajectory data from a file or from a previous simulation.
#[derive(Debug, Args)]
pub struct AnalyzeArgs {
    /// Input trajectory file (JSON)
    #[arg(long, short = 'i', required = true)]
    pub input: PathBuf,

    /// Null hypothesis tier
    #[arg(long, short = 'n', default_value = "tiered", value_enum)]
    pub null_tier: NullTierArg,

    /// Significance level (alpha)
    #[arg(
        long,
        short = 'a',
        default_value = "0.05",
        value_parser = validate_alpha,
    )]
    pub alpha: f64,

    /// Oracle access level
    #[arg(long, short = 'l', default_value = "layer0", value_enum)]
    pub oracle_level: OracleLevelArg,

    /// Output file for analysis results
    #[arg(long, short = 'o', default_value = "analysis.json")]
    pub output: PathBuf,

    /// Number of bootstrap resamples
    #[arg(long, default_value = "1000")]
    pub bootstrap_resamples: usize,

    /// Number of permutations for permutation tests
    #[arg(long, default_value = "10000")]
    pub num_permutations: usize,

    /// Apply multiple testing correction
    #[arg(long, default_value = "holm")]
    pub correction: String,

    /// Random seed
    #[arg(long, default_value = "42")]
    pub seed: u64,
}

// ── Certify subcommand ──────────────────────────────────────────────────────

/// Generate a formal certificate from analysis results.
///
/// Constructs a proof-carrying certificate including the detection result,
/// statistical evidence, and a verifiable proof chain.
#[derive(Debug, Args)]
pub struct CertifyArgs {
    /// Analysis results file (JSON)
    #[arg(long, short = 'i', required = true)]
    pub analysis_results: PathBuf,

    /// Output certificate file
    #[arg(long, short = 'o', default_value = "certificate.json")]
    pub output: PathBuf,

    /// Include full evidence bundle in certificate
    #[arg(long)]
    pub include_evidence: bool,

    /// Certificate format
    #[arg(long, default_value = "json")]
    pub cert_format: String,

    /// Sign certificate with scenario hash
    #[arg(long)]
    pub sign: bool,
}

// ── Verify subcommand ───────────────────────────────────────────────────────

/// Verify an existing certificate for correctness and consistency.
///
/// Checks the certificate hash, statistical test consistency,
/// evidence sufficiency, and optionally cross-references with
/// an evidence bundle.
#[derive(Debug, Args)]
pub struct VerifyArgs {
    /// Certificate file to verify
    #[arg(long, short = 'c', required = true)]
    pub certificate: PathBuf,

    /// Evidence bundle file for cross-verification
    #[arg(long, short = 'e')]
    pub evidence_bundle: Option<PathBuf>,

    /// Strict mode: reject on any warning
    #[arg(long)]
    pub strict: bool,

    /// Print detailed verification report
    #[arg(long)]
    pub detailed: bool,
}

// ── Evaluate subcommand ─────────────────────────────────────────────────────

/// Run the evaluation benchmark suite.
///
/// Executes a configurable set of scenarios with multiple seeds,
/// computes classification metrics (precision, recall, F1),
/// performs ROC analysis, and compares against baselines.
#[derive(Debug, Args)]
pub struct EvaluateArgs {
    /// Evaluation mode
    #[arg(long, short = 'm', default_value = "standard", value_enum)]
    pub mode: EvalModeArg,

    /// Specific scenarios to evaluate (comma-separated)
    #[arg(long, short = 's', value_delimiter = ',')]
    pub scenarios: Option<Vec<String>>,

    /// Output directory for evaluation results
    #[arg(long, short = 'o', default_value = "./evaluation_output")]
    pub output_dir: PathBuf,

    /// Oracle access level for evaluation
    #[arg(long, short = 'l', default_value = "layer0", value_enum)]
    pub oracle_level: OracleLevelArg,

    /// Significance level for all tests
    #[arg(
        long,
        short = 'a',
        default_value = "0.05",
        value_parser = validate_alpha,
    )]
    pub alpha: f64,

    /// Number of parallel workers (0 = auto)
    #[arg(long, short = 'j', default_value = "0")]
    pub jobs: usize,

    /// Include baseline comparisons
    #[arg(long)]
    pub baselines: bool,

    /// Include ROC analysis
    #[arg(long)]
    pub roc: bool,

    /// Random seed
    #[arg(long, default_value = "42")]
    pub seed: u64,
}

// ── Scenarios subcommand ────────────────────────────────────────────────────

/// List available evaluation scenarios.
///
/// Shows scenario identifiers, descriptions, difficulty levels,
/// ground truth labels, and market configurations.
#[derive(Debug, Args)]
pub struct ScenariosArgs {
    /// Filter by difficulty
    #[arg(long, value_enum)]
    pub difficulty: Option<DifficultyArg>,

    /// Filter by market type
    #[arg(long, value_enum)]
    pub market_type: Option<MarketTypeArg>,

    /// Filter by ground truth
    #[arg(long, value_enum)]
    pub ground_truth: Option<GroundTruthArg>,

    /// Show detailed information for each scenario
    #[arg(long)]
    pub detailed: bool,
}

/// Difficulty filter for scenario listing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum DifficultyArg {
    Easy,
    Medium,
    Hard,
}

impl std::fmt::Display for DifficultyArg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DifficultyArg::Easy => write!(f, "Easy"),
            DifficultyArg::Medium => write!(f, "Medium"),
            DifficultyArg::Hard => write!(f, "Hard"),
        }
    }
}

/// Ground truth filter for scenario listing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum GroundTruthArg {
    Collusive,
    Competitive,
    Inconclusive,
}

impl std::fmt::Display for GroundTruthArg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GroundTruthArg::Collusive => write!(f, "Collusive"),
            GroundTruthArg::Competitive => write!(f, "Competitive"),
            GroundTruthArg::Inconclusive => write!(f, "Inconclusive"),
        }
    }
}

// ── Config subcommand ───────────────────────────────────────────────────────

/// Manage configuration settings.
///
/// Show the effective configuration, generate a default config file,
/// or validate an existing config.
#[derive(Debug, Args)]
pub struct ConfigArgs {
    /// Configuration action
    #[arg(value_enum)]
    pub action: ConfigAction,

    /// Output file (for generate action)
    #[arg(long, short = 'o', default_value = "collusion-proof.toml")]
    pub output: PathBuf,

    /// Config profile to use for generation
    #[arg(long, value_enum)]
    pub profile: Option<ConfigProfileArg>,
}

/// Configuration profile for preset parameter values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum ConfigProfileArg {
    /// Quick testing profile (minimal rounds/bootstraps)
    Smoke,
    /// Standard evaluation profile
    Standard,
    /// Full high-fidelity profile
    Full,
}

impl Default for ConfigProfileArg {
    fn default() -> Self {
        ConfigProfileArg::Standard
    }
}

impl std::fmt::Display for ConfigProfileArg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigProfileArg::Smoke => write!(f, "Smoke"),
            ConfigProfileArg::Standard => write!(f, "Standard"),
            ConfigProfileArg::Full => write!(f, "Full"),
        }
    }
}

// ── Validation functions ────────────────────────────────────────────────────

/// Validate that alpha is in (0, 1).
fn validate_alpha(s: &str) -> Result<f64, String> {
    let val: f64 = s.parse().map_err(|_| format!("'{}' is not a valid number", s))?;
    if val <= 0.0 || val >= 1.0 {
        Err(format!("alpha must be in (0, 1), got {}", val))
    } else {
        Ok(val)
    }
}

/// Validate that rounds are in a reasonable range.
fn validate_rounds(s: &str) -> Result<usize, String> {
    let val: usize = s.parse().map_err(|_| format!("'{}' is not a valid integer", s))?;
    if val < 10 {
        Err(format!("rounds must be >= 10, got {}", val))
    } else if val > 1_000_000 {
        Err(format!("rounds must be <= 1000000, got {}", val))
    } else {
        Ok(val)
    }
}

/// Validate number of players.
fn validate_players(s: &str) -> Result<usize, String> {
    let val: usize = s.parse().map_err(|_| format!("'{}' is not a valid integer", s))?;
    if val < 2 {
        Err(format!("num_players must be >= 2, got {}", val))
    } else if val > 20 {
        Err(format!("num_players must be <= 20, got {}", val))
    } else {
        Ok(val)
    }
}

// ── Conversion helpers ──────────────────────────────────────────────────────

impl OracleLevelArg {
    /// Convert to the shared-types OracleAccessLevel.
    pub fn to_oracle_level(self) -> shared_types::OracleAccessLevel {
        match self {
            OracleLevelArg::Layer0 => shared_types::OracleAccessLevel::Layer0,
            OracleLevelArg::Layer1 => shared_types::OracleAccessLevel::Layer1,
            OracleLevelArg::Layer2 => shared_types::OracleAccessLevel::Layer2,
        }
    }
}

impl MarketTypeArg {
    /// Convert to the shared-types MarketType.
    pub fn to_market_type(self) -> shared_types::MarketType {
        match self {
            MarketTypeArg::Bertrand => shared_types::MarketType::Bertrand,
            MarketTypeArg::Cournot => shared_types::MarketType::Cournot,
        }
    }
}

impl AlgorithmArg {
    /// Convert to the shared-types AlgorithmType.
    pub fn to_algorithm_type(self) -> shared_types::AlgorithmType {
        match self {
            AlgorithmArg::QLearning => shared_types::AlgorithmType::QLearning,
            AlgorithmArg::Dqn => shared_types::AlgorithmType::DQN,
            AlgorithmArg::GrimTrigger => shared_types::AlgorithmType::GrimTrigger,
            AlgorithmArg::TitForTat => shared_types::AlgorithmType::TitForTat,
            AlgorithmArg::Bandit => shared_types::AlgorithmType::Bandit,
            AlgorithmArg::Nash => shared_types::AlgorithmType::NashEquilibrium,
            AlgorithmArg::Myopic => shared_types::AlgorithmType::MyopicBestResponse,
        }
    }
}

impl DemandModelArg {
    /// Build a DemandSystem from this arg and associated parameters.
    pub fn to_demand_system(
        self,
        intercept: f64,
        slope: f64,
        _cross_slope: f64,
        logit_mu: f64,
        ces_sigma: f64,
    ) -> shared_types::DemandSystem {
        match self {
            DemandModelArg::Linear => shared_types::DemandSystem::Linear {
                max_quantity: intercept,
                slope,
            },
            DemandModelArg::Logit => shared_types::DemandSystem::Logit {
                temperature: logit_mu,
                outside_option_value: 0.0,
                market_size: 1.0,
            },
            DemandModelArg::Ces => shared_types::DemandSystem::CES {
                elasticity_of_substitution: ces_sigma,
                market_size: 1.0,
                quality_indices: vec![],
            },
        }
    }
}

impl EvalModeDataArg {
    /// Convert to shared-types EvaluationMode.
    /// Data splitting phases map to the Standard evaluation mode.
    pub fn to_evaluation_mode(self) -> shared_types::EvaluationMode {
        match self {
            EvalModeDataArg::Training => shared_types::EvaluationMode::Standard,
            EvalModeDataArg::Testing => shared_types::EvaluationMode::Standard,
            EvalModeDataArg::Validation => shared_types::EvaluationMode::Standard,
            EvalModeDataArg::Holdout => shared_types::EvaluationMode::Standard,
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn test_validate_alpha_valid() {
        assert!(validate_alpha("0.05").is_ok());
        assert!(validate_alpha("0.01").is_ok());
        assert!(validate_alpha("0.1").is_ok());
        assert!(validate_alpha("0.99").is_ok());
    }

    #[test]
    fn test_validate_alpha_invalid() {
        assert!(validate_alpha("0.0").is_err());
        assert!(validate_alpha("1.0").is_err());
        assert!(validate_alpha("-0.1").is_err());
        assert!(validate_alpha("abc").is_err());
    }

    #[test]
    fn test_validate_rounds_valid() {
        assert!(validate_rounds("10").is_ok());
        assert!(validate_rounds("1000").is_ok());
        assert!(validate_rounds("1000000").is_ok());
    }

    #[test]
    fn test_validate_rounds_invalid() {
        assert!(validate_rounds("5").is_err());
        assert!(validate_rounds("2000000").is_err());
        assert!(validate_rounds("abc").is_err());
    }

    #[test]
    fn test_validate_players_valid() {
        assert!(validate_players("2").is_ok());
        assert!(validate_players("10").is_ok());
        assert!(validate_players("20").is_ok());
    }

    #[test]
    fn test_validate_players_invalid() {
        assert!(validate_players("1").is_err());
        assert!(validate_players("25").is_err());
        assert!(validate_players("abc").is_err());
    }

    #[test]
    fn test_oracle_level_conversion() {
        assert_eq!(
            OracleLevelArg::Layer0.to_oracle_level(),
            shared_types::OracleAccessLevel::Layer0
        );
        assert_eq!(
            OracleLevelArg::Layer1.to_oracle_level(),
            shared_types::OracleAccessLevel::Layer1
        );
        assert_eq!(
            OracleLevelArg::Layer2.to_oracle_level(),
            shared_types::OracleAccessLevel::Layer2
        );
    }

    #[test]
    fn test_market_type_conversion() {
        assert_eq!(
            MarketTypeArg::Bertrand.to_market_type(),
            shared_types::MarketType::Bertrand
        );
        assert_eq!(
            MarketTypeArg::Cournot.to_market_type(),
            shared_types::MarketType::Cournot
        );
    }

    #[test]
    fn test_algorithm_conversion() {
        assert_eq!(
            AlgorithmArg::QLearning.to_algorithm_type(),
            shared_types::AlgorithmType::QLearning
        );
        assert_eq!(
            AlgorithmArg::Nash.to_algorithm_type(),
            shared_types::AlgorithmType::NashEquilibrium
        );
    }

    #[test]
    fn test_demand_model_linear() {
        let ds = DemandModelArg::Linear.to_demand_system(10.0, 1.0, 0.5, 0.25, 2.0);
        matches!(ds, shared_types::DemandSystem::Linear { .. });
    }

    #[test]
    fn test_demand_model_logit() {
        let ds = DemandModelArg::Logit.to_demand_system(10.0, 1.0, 0.5, 0.25, 2.0);
        matches!(ds, shared_types::DemandSystem::Logit { .. });
    }

    #[test]
    fn test_demand_model_ces() {
        let ds = DemandModelArg::Ces.to_demand_system(10.0, 1.0, 0.5, 0.25, 2.0);
        matches!(ds, shared_types::DemandSystem::CES { .. });
    }

    #[test]
    fn test_output_format_default() {
        assert_eq!(OutputFormat::default(), OutputFormat::Text);
    }

    #[test]
    fn test_oracle_level_display() {
        assert_eq!(OracleLevelArg::Layer0.to_string(), "Layer0");
        assert_eq!(OracleLevelArg::Layer2.to_string(), "Layer2");
    }

    #[test]
    fn test_eval_mode_display() {
        assert_eq!(EvalModeArg::Smoke.to_string(), "Smoke");
        assert_eq!(EvalModeArg::Standard.to_string(), "Standard");
        assert_eq!(EvalModeArg::Full.to_string(), "Full");
    }

    #[test]
    fn test_parse_run_command() {
        let cli = CollusionProofCli::try_parse_from([
            "collusion-proof",
            "run",
            "--scenario",
            "bertrand_qlearning_2p",
            "--alpha",
            "0.01",
            "--rounds",
            "500",
        ]);
        assert!(cli.is_ok());
        if let Ok(cli) = cli {
            match cli.command {
                Command::Run(args) => {
                    assert_eq!(args.scenario.as_deref(), Some("bertrand_qlearning_2p"));
                    assert!((args.alpha - 0.01).abs() < 1e-9);
                    assert_eq!(args.rounds, 500);
                }
                _ => panic!("Expected Run command"),
            }
        }
    }

    #[test]
    fn test_parse_simulate_command() {
        let cli = CollusionProofCli::try_parse_from([
            "collusion-proof",
            "simulate",
            "--market-type",
            "bertrand",
            "--num-players",
            "3",
            "--algorithm",
            "q-learning",
        ]);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_parse_evaluate_command() {
        let cli = CollusionProofCli::try_parse_from([
            "collusion-proof",
            "evaluate",
            "--mode",
            "smoke",
            "--baselines",
            "--roc",
        ]);
        assert!(cli.is_ok());
        if let Ok(cli) = cli {
            match cli.command {
                Command::Evaluate(args) => {
                    assert_eq!(args.mode, EvalModeArg::Smoke);
                    assert!(args.baselines);
                    assert!(args.roc);
                }
                _ => panic!("Expected Evaluate command"),
            }
        }
    }

    #[test]
    fn test_parse_config_show() {
        let cli = CollusionProofCli::try_parse_from([
            "collusion-proof",
            "config",
            "show",
        ]);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_parse_verify_command() {
        let cli = CollusionProofCli::try_parse_from([
            "collusion-proof",
            "verify",
            "--certificate",
            "cert.json",
            "--strict",
        ]);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_parse_global_flags() {
        let cli = CollusionProofCli::try_parse_from([
            "collusion-proof",
            "--quiet",
            "--format",
            "json",
            "config",
            "show",
        ]);
        assert!(cli.is_ok());
        if let Ok(cli) = cli {
            assert!(cli.quiet);
            assert_eq!(cli.format, OutputFormat::Json);
        }
    }

    #[test]
    fn test_parse_scenarios_command() {
        let cli = CollusionProofCli::try_parse_from([
            "collusion-proof",
            "scenarios",
            "--difficulty",
            "easy",
            "--detailed",
        ]);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_verbosity_flags() {
        let cli = CollusionProofCli::try_parse_from([
            "collusion-proof",
            "-vv",
            "config",
            "show",
        ]);
        assert!(cli.is_ok());
        if let Ok(cli) = cli {
            assert_eq!(cli.verbose, 2);
        }
    }
}
