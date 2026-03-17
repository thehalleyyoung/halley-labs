//! CLI argument parsing for SafeStep using clap derive macros.

use std::path::PathBuf;
use clap::{Parser, Subcommand, ValueEnum};

/// SafeStep verified deployment planner with rollback safety envelopes.
#[derive(Parser, Debug, Clone)]
#[command(
    name = "safestep",
    version,
    about = "Verified deployment planner with rollback safety envelopes",
    long_about = "SafeStep synthesizes deployment plans that are formally verified \
                  to satisfy compatibility constraints, with computed rollback \
                  safety envelopes and point-of-no-return identification."
)]
pub struct Cli {
    /// Path to configuration file (JSON or YAML).
    #[arg(long, short = 'c', global = true, env = "SAFESTEP_CONFIG")]
    pub config: Option<PathBuf>,

    /// Increase verbosity (-v, -vv, -vvv).
    #[arg(long, short = 'v', global = true, action = clap::ArgAction::Count)]
    pub verbose: u8,

    /// Suppress all output except errors.
    #[arg(long, short = 'q', global = true)]
    pub quiet: bool,

    /// Output format for results.
    #[arg(long, global = true, default_value = "text", env = "SAFESTEP_FORMAT")]
    pub output_format: OutputFormat,

    /// Color output control.
    #[arg(long, global = true, default_value = "auto", env = "SAFESTEP_COLOR")]
    pub color: ColorOption,

    /// Write output to file instead of stdout.
    #[arg(long, short = 'o', global = true)]
    pub output: Option<PathBuf>,

    #[command(subcommand)]
    pub command: Command,
}

/// Supported output formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum OutputFormat {
    Text,
    Json,
    Yaml,
    Markdown,
    Html,
}

impl std::fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Text => write!(f, "text"),
            Self::Json => write!(f, "json"),
            Self::Yaml => write!(f, "yaml"),
            Self::Markdown => write!(f, "markdown"),
            Self::Html => write!(f, "html"),
        }
    }
}

/// Color output control.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum ColorOption {
    Auto,
    Always,
    Never,
}

impl ColorOption {
    /// Resolve whether to use color based on env, terminal, and user choice.
    pub fn should_use_color(&self) -> bool {
        match self {
            Self::Always => true,
            Self::Never => false,
            Self::Auto => {
                if std::env::var("NO_COLOR").is_ok() {
                    return false;
                }
                if std::env::var("FORCE_COLOR").is_ok() {
                    return true;
                }
                is_stdout_tty()
            }
        }
    }
}

fn is_stdout_tty() -> bool {
    #[cfg(unix)]
    {
        extern "C" {
            fn isatty(fd: i32) -> i32;
        }
        unsafe { isatty(1) != 0 }
    }
    #[cfg(not(unix))]
    {
        true
    }
}

/// Top-level subcommands.
#[derive(Subcommand, Debug, Clone)]
pub enum Command {
    /// Generate a verified deployment plan.
    Plan(PlanArgs),
    /// Verify an existing deployment plan against constraints.
    Verify(VerifyArgs),
    /// Compute rollback safety envelope for a plan.
    Envelope(EnvelopeArgs),
    /// Analyze service schema compatibility.
    Analyze(AnalyzeArgs),
    /// Diff two plans or schemas.
    Diff(DiffArgs),
    /// Export plan to GitOps format (ArgoCD/Flux).
    Export(ExportArgs),
    /// Validate manifests and configuration files.
    Validate(ValidateArgs),
    /// Run performance benchmarks.
    Benchmark(BenchmarkArgs),
}

/// Arguments for the `plan` subcommand.
#[derive(Parser, Debug, Clone)]
pub struct PlanArgs {
    /// Starting state as comma-separated version indices (e.g. "0,0,1").
    #[arg(long, short = 's')]
    pub start_state: String,

    /// Target state as comma-separated version indices (e.g. "1,2,1").
    #[arg(long, short = 't')]
    pub target_state: String,

    /// Directory containing service manifests.
    #[arg(long, short = 'm', default_value = ".")]
    pub manifest_dir: PathBuf,

    /// Maximum search depth for the planner.
    #[arg(long, default_value = "100")]
    pub max_depth: usize,

    /// Planner timeout in seconds.
    #[arg(long, default_value = "300")]
    pub timeout: u64,

    /// Optimization objective (steps, risk, duration).
    #[arg(long, default_value = "steps")]
    pub optimize: OptimizeObjective,

    /// Constraints file path.
    #[arg(long)]
    pub constraints_file: Option<PathBuf>,

    /// Use CEGAR refinement loop.
    #[arg(long, default_value = "true")]
    pub cegar: bool,

    /// Save plan to file.
    #[arg(long)]
    pub save: Option<PathBuf>,
}

/// Optimization objectives for planning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum OptimizeObjective {
    Steps,
    Risk,
    Duration,
}

impl std::fmt::Display for OptimizeObjective {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Steps => write!(f, "steps"),
            Self::Risk => write!(f, "risk"),
            Self::Duration => write!(f, "duration"),
        }
    }
}

/// Arguments for the `verify` subcommand.
#[derive(Parser, Debug, Clone)]
pub struct VerifyArgs {
    /// Path to the plan file (JSON).
    #[arg(long, short = 'p')]
    pub plan_file: PathBuf,

    /// Path to the constraints file (JSON).
    #[arg(long)]
    pub constraints_file: Option<PathBuf>,

    /// Check plan monotonicity (no service revisits).
    #[arg(long, default_value = "true")]
    pub check_monotonicity: bool,

    /// Check plan completeness (start -> target).
    #[arg(long, default_value = "true")]
    pub check_completeness: bool,

    /// Show all violations (not just the first).
    #[arg(long)]
    pub show_all: bool,
}

/// Arguments for the `envelope` subcommand.
#[derive(Parser, Debug, Clone)]
pub struct EnvelopeArgs {
    /// Path to the plan file (JSON).
    #[arg(long, short = 'p')]
    pub plan_file: PathBuf,

    /// Show detailed annotations.
    #[arg(long)]
    pub detailed: bool,

    /// Include robustness analysis.
    #[arg(long)]
    pub robustness: bool,

    /// Adversary budget for robustness (k value).
    #[arg(long, default_value = "1")]
    pub adversary_budget: usize,

    /// Enable point-of-no-return detection.
    #[arg(long, default_value = "true")]
    pub detect_pnr: bool,

    /// Minimum robustness score threshold.
    #[arg(long, default_value = "0.5")]
    pub min_robustness: f64,
}

/// Arguments for the `analyze` subcommand.
#[derive(Parser, Debug, Clone)]
pub struct AnalyzeArgs {
    /// Directory containing API schemas.
    #[arg(long, short = 'd')]
    pub schema_dir: PathBuf,

    /// Schema format to parse.
    #[arg(long, short = 'f', default_value = "openapi")]
    pub format: SchemaFormat,

    /// Show only breaking changes.
    #[arg(long)]
    pub breaking_only: bool,

    /// Export compatibility predicates to file.
    #[arg(long)]
    pub export_predicates: Option<PathBuf>,

    /// Minimum confidence score for classifications (0.0-1.0).
    #[arg(long, default_value = "0.8")]
    pub min_confidence: f64,

    /// Baseline version to filter changes against.
    #[arg(long)]
    pub baseline_version: Option<String>,
}

/// Supported schema formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum SchemaFormat {
    Openapi,
    Protobuf,
    Graphql,
    Avro,
}

impl std::fmt::Display for SchemaFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Openapi => write!(f, "openapi"),
            Self::Protobuf => write!(f, "protobuf"),
            Self::Graphql => write!(f, "graphql"),
            Self::Avro => write!(f, "avro"),
        }
    }
}

/// Arguments for the `diff` subcommand.
#[derive(Parser, Debug, Clone)]
pub struct DiffArgs {
    /// Path to the old plan or schema.
    #[arg(long)]
    pub old: PathBuf,

    /// Path to the new plan or schema.
    #[arg(long)]
    pub new: PathBuf,

    /// Highlight safety-relevant changes.
    #[arg(long, default_value = "true")]
    pub highlight_safety: bool,

    /// Show context lines around changes.
    #[arg(long, default_value = "3")]
    pub context_lines: usize,
}

/// Arguments for the `export` subcommand.
#[derive(Parser, Debug, Clone)]
pub struct ExportArgs {
    /// Path to the plan file (JSON).
    #[arg(long, short = 'p')]
    pub plan_file: PathBuf,

    /// GitOps format to export.
    #[arg(long, short = 'f', default_value = "argocd")]
    pub format: GitOpsFormat,

    /// Output directory for exported resources.
    #[arg(long, short = 'O', default_value = "./gitops-output")]
    pub output_dir: PathBuf,

    /// Namespace for resources.
    #[arg(long, default_value = "default")]
    pub namespace: String,

    /// Validate generated resources.
    #[arg(long, default_value = "true")]
    pub validate_output: bool,
}

/// Supported GitOps export formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum GitOpsFormat {
    Argocd,
    Flux,
}

impl std::fmt::Display for GitOpsFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Argocd => write!(f, "argocd"),
            Self::Flux => write!(f, "flux"),
        }
    }
}

/// Arguments for the `validate` subcommand.
#[derive(Parser, Debug, Clone)]
pub struct ValidateArgs {
    /// Directory containing manifests to validate.
    #[arg(long, short = 'm', default_value = ".")]
    pub manifest_dir: PathBuf,

    /// Strict mode: treat warnings as errors.
    #[arg(long)]
    pub strict: bool,

    /// Validate only files matching this glob pattern.
    #[arg(long)]
    pub pattern: Option<String>,

    /// Maximum number of errors before stopping.
    #[arg(long, default_value = "100")]
    pub max_errors: usize,
}

/// Arguments for the `benchmark` subcommand.
#[derive(Parser, Debug, Clone)]
pub struct BenchmarkArgs {
    /// Number of services in the benchmark.
    #[arg(long, default_value = "10")]
    pub services: usize,

    /// Number of versions per service.
    #[arg(long, default_value = "5")]
    pub versions: usize,

    /// Topology type for the service graph.
    #[arg(long, default_value = "mesh")]
    pub topology: TopologyType,

    /// Number of benchmark iterations.
    #[arg(long, default_value = "5")]
    pub iterations: usize,

    /// Compare against baseline file.
    #[arg(long)]
    pub baseline: Option<PathBuf>,

    /// Save results as new baseline.
    #[arg(long)]
    pub save_baseline: Option<PathBuf>,
}

/// Topology types for benchmark service graphs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum TopologyType {
    Mesh,
    HubSpoke,
    Hierarchical,
    Chain,
    Random,
}

impl std::fmt::Display for TopologyType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Mesh => write!(f, "mesh"),
            Self::HubSpoke => write!(f, "hub-spoke"),
            Self::Hierarchical => write!(f, "hierarchical"),
            Self::Chain => write!(f, "chain"),
            Self::Random => write!(f, "random"),
        }
    }
}

/// Parse a comma-separated list of version indices into a vector.
pub fn parse_state(s: &str) -> anyhow::Result<Vec<u16>> {
    s.split(',')
        .map(|part| {
            part.trim()
                .parse::<u16>()
                .map_err(|e| anyhow::anyhow!("invalid version index '{}': {}", part.trim(), e))
        })
        .collect()
}

/// Resolve the effective log level from verbose/quiet flags.
pub fn resolve_log_level(verbose: u8, quiet: bool) -> tracing::Level {
    if quiet {
        return tracing::Level::ERROR;
    }
    match verbose {
        0 => tracing::Level::WARN,
        1 => tracing::Level::INFO,
        2 => tracing::Level::DEBUG,
        _ => tracing::Level::TRACE,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn test_cli_parses() {
        Cli::command().debug_assert();
    }

    #[test]
    fn test_parse_state_valid() {
        assert_eq!(parse_state("0,1,2").unwrap(), vec![0, 1, 2]);
    }

    #[test]
    fn test_parse_state_single() {
        assert_eq!(parse_state("5").unwrap(), vec![5]);
    }

    #[test]
    fn test_parse_state_with_spaces() {
        assert_eq!(parse_state("0, 1, 2").unwrap(), vec![0, 1, 2]);
    }

    #[test]
    fn test_parse_state_invalid() {
        assert!(parse_state("0,abc,2").is_err());
    }

    #[test]
    fn test_resolve_log_level_quiet() {
        assert_eq!(resolve_log_level(0, true), tracing::Level::ERROR);
    }

    #[test]
    fn test_resolve_log_level_verbose() {
        assert_eq!(resolve_log_level(0, false), tracing::Level::WARN);
        assert_eq!(resolve_log_level(1, false), tracing::Level::INFO);
        assert_eq!(resolve_log_level(2, false), tracing::Level::DEBUG);
        assert_eq!(resolve_log_level(3, false), tracing::Level::TRACE);
    }

    #[test]
    fn test_output_format_display() {
        assert_eq!(OutputFormat::Text.to_string(), "text");
        assert_eq!(OutputFormat::Json.to_string(), "json");
        assert_eq!(OutputFormat::Yaml.to_string(), "yaml");
        assert_eq!(OutputFormat::Markdown.to_string(), "markdown");
        assert_eq!(OutputFormat::Html.to_string(), "html");
    }

    #[test]
    fn test_color_option_never() {
        assert!(!ColorOption::Never.should_use_color());
    }

    #[test]
    fn test_color_option_always() {
        assert!(ColorOption::Always.should_use_color());
    }

    #[test]
    fn test_plan_subcommand_parses() {
        let cli = Cli::try_parse_from([
            "safestep", "plan",
            "--start-state", "0,0", "--target-state", "1,1",
            "--manifest-dir", "/tmp/manifests", "--max-depth", "50", "--timeout", "120",
        ]).unwrap();
        if let Command::Plan(args) = &cli.command {
            assert_eq!(args.start_state, "0,0");
            assert_eq!(args.max_depth, 50);
        } else {
            panic!("expected Plan");
        }
    }

    #[test]
    fn test_verify_subcommand_parses() {
        assert!(Cli::try_parse_from(["safestep", "verify", "--plan-file", "/tmp/p.json"]).is_ok());
    }

    #[test]
    fn test_envelope_subcommand_parses() {
        assert!(Cli::try_parse_from(["safestep", "envelope", "--plan-file", "/tmp/p.json", "--detailed"]).is_ok());
    }

    #[test]
    fn test_analyze_subcommand_parses() {
        assert!(Cli::try_parse_from(["safestep", "analyze", "--schema-dir", "/tmp/s", "--format", "protobuf"]).is_ok());
    }

    #[test]
    fn test_diff_subcommand_parses() {
        assert!(Cli::try_parse_from(["safestep", "diff", "--old", "/tmp/o", "--new", "/tmp/n"]).is_ok());
    }

    #[test]
    fn test_export_subcommand_parses() {
        assert!(Cli::try_parse_from(["safestep", "export", "--plan-file", "/tmp/p.json", "--format", "flux", "--output-dir", "/tmp/o"]).is_ok());
    }

    #[test]
    fn test_validate_subcommand_parses() {
        assert!(Cli::try_parse_from(["safestep", "validate", "--manifest-dir", "/tmp/m", "--strict"]).is_ok());
    }

    #[test]
    fn test_benchmark_subcommand_parses() {
        let cli = Cli::try_parse_from([
            "safestep", "benchmark", "--services", "20", "--versions", "8", "--topology", "hub-spoke",
        ]).unwrap();
        if let Command::Benchmark(args) = &cli.command {
            assert_eq!(args.services, 20);
            assert_eq!(args.topology, TopologyType::HubSpoke);
        } else {
            panic!("expected Benchmark");
        }
    }

    #[test]
    fn test_global_options() {
        let cli = Cli::try_parse_from([
            "safestep", "-vv", "--output-format", "json", "--color", "never",
            "validate", "--manifest-dir", ".",
        ]).unwrap();
        assert_eq!(cli.verbose, 2);
        assert_eq!(cli.output_format, OutputFormat::Json);
        assert_eq!(cli.color, ColorOption::Never);
    }
}
