//! CLI command definitions and argument parsing via [`clap`] derive macros.
//!
//! Every subcommand is represented as a struct that owns its arguments.  Shared
//! argument groups ([`OutputArgs`], [`ModeArgs`], [`PolicyArgs`]) are flattened
//! into the subcommands that need them.

use clap::{Parser, Subcommand, ValueEnum};
use serde::{Deserialize, Serialize};

// ── Top-level CLI ──────────────────────────────────────────────────────────

/// CascadeVerify – static verification of retry-amplification cascades
/// in microservice configuration files.
#[derive(Parser, Debug)]
#[command(
    name = "cascade-verify",
    version,
    about = "Detect and repair retry-amplification cascades in microservice configs",
    long_about = "CascadeVerify statically analyses Kubernetes / Istio / Envoy / Helm \
                  configuration files to detect retry-amplification cascades, timeout \
                  propagation issues, and circuit-breaker storms.  It can also synthesise \
                  minimal configuration repairs.",
    propagate_version = true
)]
pub struct CascadeCli {
    /// Increase log verbosity (-v for DEBUG, -vv for TRACE).
    #[arg(short, long, global = true, action = clap::ArgAction::Count)]
    pub verbose: u8,

    /// Suppress all output except errors.
    #[arg(short, long, global = true)]
    pub quiet: bool,

    /// Override the configuration file location.
    #[arg(long, global = true, env = "CASCADE_CONFIG")]
    pub config: Option<String>,

    /// Disable coloured output.
    #[arg(long, global = true, env = "NO_COLOR")]
    pub no_color: bool,

    #[command(subcommand)]
    pub command: Commands,
}

impl CascadeCli {
    /// Return `true` when the user requested verbose logging.
    pub fn verbose(&self) -> bool {
        self.verbose > 0
    }

    /// Perform semantic validation that cannot be expressed with clap attrs.
    pub fn validate(&self) -> Result<(), String> {
        if self.quiet && self.verbose > 0 {
            return Err("--quiet and --verbose are mutually exclusive".into());
        }
        match &self.command {
            Commands::Verify(a) => a.validate(),
            Commands::Repair(a) => a.validate(),
            Commands::Check(a) => a.validate(),
            Commands::Analyze(a) => a.validate(),
            Commands::Diff(a) => a.validate(),
            Commands::Report(a) => a.validate(),
            Commands::Benchmark(a) => a.validate(),
        }
    }
}

// ── Subcommands ────────────────────────────────────────────────────────────

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Analyse configuration files for cascade risks.
    Verify(VerifyArgs),

    /// Synthesise repairs for detected cascades.
    Repair(RepairArgs),

    /// Quick CI/CD gate check (Tier 1 only – fast).
    Check(CheckArgs),

    /// Deep analysis with bounded model checking (Tier 2).
    Analyze(AnalyzeArgs),

    /// Analyse only the delta between two config sets.
    Diff(DiffArgs),

    /// Generate a report from cached analysis results.
    Report(ReportArgs),

    /// Run benchmarks on synthetic topologies.
    Benchmark(BenchmarkArgs),
}

// ── Shared argument groups ─────────────────────────────────────────────────

/// Controls the output destination and format.
#[derive(Parser, Debug, Clone, Serialize, Deserialize)]
pub struct OutputArgs {
    /// Output format.
    #[arg(long, default_value = "table", env = "CASCADE_OUTPUT")]
    pub format: String,

    /// Write output to a file instead of stdout.
    #[arg(short, long)]
    pub output_file: Option<String>,

    /// Show detailed evidence for each finding.
    #[arg(long)]
    pub verbose: bool,
}

impl Default for OutputArgs {
    fn default() -> Self {
        Self {
            format: "table".into(),
            verbose: false,
            output_file: None,
        }
    }
}

impl OutputArgs {
    /// Parse the user-supplied format string into an [`OutputFormat`].
    pub fn parsed_format(&self) -> Result<OutputFormat, String> {
        OutputFormat::from_str_loose(&self.format)
    }
}

/// Controls which analysis tier and caching behaviour to use.
#[derive(Parser, Debug, Clone, Serialize, Deserialize)]
pub struct ModeArgs {
    /// Analysis tier: `tier1` (fast graph), `tier2` (BMC), `auto`.
    #[arg(long, default_value = "auto", env = "CASCADE_MODE")]
    pub tier: String,

    /// Enable incremental analysis (only re-analyse changed services).
    #[arg(long)]
    pub incremental: bool,

    /// Enable / disable result caching.
    #[arg(long, default_value_t = true)]
    pub cache: bool,
}

impl Default for ModeArgs {
    fn default() -> Self {
        Self {
            tier: "auto".into(),
            incremental: false,
            cache: true,
        }
    }
}

impl ModeArgs {
    /// Parse the tier string into the internal enum.
    pub fn parsed_tier(&self) -> Result<AnalysisTier, String> {
        match self.tier.to_lowercase().as_str() {
            "tier1" | "1" | "fast" => Ok(AnalysisTier::Tier1),
            "tier2" | "2" | "deep" => Ok(AnalysisTier::Tier2),
            "auto" => Ok(AnalysisTier::Auto),
            other => Err(format!(
                "unknown analysis tier '{other}': expected tier1, tier2, or auto"
            )),
        }
    }
}

/// Severity-based policy enforcement for CI gating.
#[derive(Parser, Debug, Clone, Serialize, Deserialize)]
pub struct PolicyArgs {
    /// Fail the build if any CRITICAL finding is detected.
    #[arg(long, default_value_t = true)]
    pub fail_on_critical: bool,

    /// Fail the build if any HIGH finding is detected.
    #[arg(long)]
    pub fail_on_high: bool,

    /// Maximum allowed MEDIUM-severity findings before the build fails.
    #[arg(long, default_value_t = 10)]
    pub max_medium: usize,
}

impl Default for PolicyArgs {
    fn default() -> Self {
        Self {
            fail_on_critical: true,
            fail_on_high: false,
            max_medium: 10,
        }
    }
}

impl PolicyArgs {
    /// Evaluate the policy against a set of findings and return `true` if the
    /// build should fail.
    pub fn should_fail(
        &self,
        critical: usize,
        high: usize,
        medium: usize,
    ) -> bool {
        if self.fail_on_critical && critical > 0 {
            return true;
        }
        if self.fail_on_high && high > 0 {
            return true;
        }
        medium > self.max_medium
    }
}

// ── OutputFormat ────────────────────────────────────────────────────────────

/// Supported output formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, ValueEnum)]
pub enum OutputFormat {
    Table,
    Json,
    Yaml,
    Sarif,
    JUnit,
    Markdown,
}

impl OutputFormat {
    /// Lenient parsing from a user string.
    pub fn from_str_loose(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "table" | "text" | "human" => Ok(Self::Table),
            "json" => Ok(Self::Json),
            "yaml" | "yml" => Ok(Self::Yaml),
            "sarif" => Ok(Self::Sarif),
            "junit" | "xml" => Ok(Self::JUnit),
            "markdown" | "md" => Ok(Self::Markdown),
            other => Err(format!(
                "unknown output format '{other}': expected table, json, yaml, sarif, junit, or markdown"
            )),
        }
    }

    /// File extension appropriate for this format.
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Table => "txt",
            Self::Json => "json",
            Self::Yaml => "yaml",
            Self::Sarif => "sarif.json",
            Self::JUnit => "xml",
            Self::Markdown => "md",
        }
    }
}

impl std::fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let label = match self {
            Self::Table => "table",
            Self::Json => "json",
            Self::Yaml => "yaml",
            Self::Sarif => "sarif",
            Self::JUnit => "junit",
            Self::Markdown => "markdown",
        };
        f.write_str(label)
    }
}

// ── AnalysisTier ───────────────────────────────────────────────────────────

/// Which analysis tier to execute.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnalysisTier {
    /// Tier 1 – fast graph-based analysis (SCC, fan-in, path enumeration).
    Tier1,
    /// Tier 2 – bounded model checking with MaxSAT repair synthesis.
    Tier2,
    /// Auto-select: start with Tier 1, promote uncertain findings to Tier 2.
    Auto,
}

impl std::fmt::Display for AnalysisTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Tier1 => f.write_str("tier1"),
            Self::Tier2 => f.write_str("tier2"),
            Self::Auto => f.write_str("auto"),
        }
    }
}

// ── Per-subcommand args ────────────────────────────────────────────────────

/// Arguments for the `verify` subcommand.
#[derive(Parser, Debug, Clone)]
pub struct VerifyArgs {
    /// One or more configuration file paths or directories to analyse.
    #[arg(required = true)]
    pub paths: Vec<String>,

    #[command(flatten)]
    pub output: OutputArgs,

    #[command(flatten)]
    pub mode: ModeArgs,

    #[command(flatten)]
    pub policy: PolicyArgs,
}

impl VerifyArgs {
    pub fn validate(&self) -> Result<(), String> {
        if self.paths.is_empty() {
            return Err("at least one config path is required".into());
        }
        self.output.parsed_format()?;
        self.mode.parsed_tier()?;
        Ok(())
    }
}

/// Arguments for the `repair` subcommand.
#[derive(Parser, Debug, Clone)]
pub struct RepairArgs {
    /// Configuration file paths to analyse and repair.
    #[arg(required = true)]
    pub paths: Vec<String>,

    /// Repair strategy: `greedy`, `optimal`, `pareto`.
    #[arg(long, default_value = "greedy")]
    pub strategy: String,

    /// Maximum number of parameter changes allowed per repair plan.
    #[arg(long, default_value_t = 10)]
    pub max_changes: usize,

    /// Only display the top-N repair plans.
    #[arg(long, default_value_t = 5)]
    pub top_n: usize,

    /// Generate unified diffs for the repairs.
    #[arg(long)]
    pub emit_diff: bool,

    #[command(flatten)]
    pub output: OutputArgs,
}

impl RepairArgs {
    pub fn validate(&self) -> Result<(), String> {
        if self.paths.is_empty() {
            return Err("at least one config path is required".into());
        }
        match self.strategy.as_str() {
            "greedy" | "optimal" | "pareto" => {}
            other => {
                return Err(format!(
                    "unknown repair strategy '{other}': expected greedy, optimal, or pareto"
                ));
            }
        }
        if self.max_changes == 0 {
            return Err("--max-changes must be at least 1".into());
        }
        self.output.parsed_format()?;
        Ok(())
    }

    /// Parse the strategy string into the internal enum.
    pub fn parsed_strategy(&self) -> RepairStrategy {
        match self.strategy.as_str() {
            "optimal" => RepairStrategy::Optimal,
            "pareto" => RepairStrategy::Pareto,
            _ => RepairStrategy::Greedy,
        }
    }
}

/// Repair strategies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepairStrategy {
    Greedy,
    Optimal,
    Pareto,
}

/// Arguments for the `check` subcommand (CI/CD gate).
#[derive(Parser, Debug, Clone)]
pub struct CheckArgs {
    /// Configuration file paths.
    #[arg(required = true)]
    pub paths: Vec<String>,

    /// Minimum severity to trigger a non-zero exit: `critical`, `high`, `medium`, `low`.
    #[arg(long, default_value = "critical")]
    pub fail_on: String,

    /// Output GitHub Actions annotations.
    #[arg(long)]
    pub annotations: bool,

    /// Write a step summary (GitHub Actions).
    #[arg(long)]
    pub step_summary: bool,

    #[command(flatten)]
    pub output: OutputArgs,
}

impl CheckArgs {
    pub fn validate(&self) -> Result<(), String> {
        if self.paths.is_empty() {
            return Err("at least one config path is required".into());
        }
        Self::parse_fail_on(&self.fail_on)?;
        Ok(())
    }

    /// Parse the `--fail-on` value.
    pub fn parse_fail_on(s: &str) -> Result<FailOnLevel, String> {
        match s.to_lowercase().as_str() {
            "critical" => Ok(FailOnLevel::Critical),
            "high" => Ok(FailOnLevel::High),
            "medium" => Ok(FailOnLevel::Medium),
            "low" => Ok(FailOnLevel::Low),
            other => Err(format!(
                "unknown --fail-on level '{other}': expected critical, high, medium, or low"
            )),
        }
    }

    pub fn parsed_fail_on(&self) -> FailOnLevel {
        Self::parse_fail_on(&self.fail_on).unwrap_or(FailOnLevel::Critical)
    }
}

/// Minimum severity that triggers a non-zero exit code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum FailOnLevel {
    Low = 0,
    Medium = 1,
    High = 2,
    Critical = 3,
}

impl FailOnLevel {
    /// Convert a types-layer [`cascade_types::Severity`] to the CLI severity
    /// scale so we can compare.
    pub fn from_types_severity(sev: &cascade_types::Severity) -> Self {
        match sev {
            cascade_types::Severity::Info => Self::Low,
            cascade_types::Severity::Low => Self::Low,
            cascade_types::Severity::Medium => Self::Medium,
            cascade_types::Severity::High => Self::High,
            cascade_types::Severity::Critical => Self::Critical,
        }
    }

    /// Return `true` when a finding at this severity should cause a failure
    /// given the policy threshold.
    pub fn exceeds_threshold(&self, threshold: FailOnLevel) -> bool {
        (*self as u8) >= (threshold as u8)
    }
}

/// Arguments for the `analyze` subcommand (deep Tier 2).
#[derive(Parser, Debug, Clone)]
pub struct AnalyzeArgs {
    /// Configuration file paths.
    #[arg(required = true)]
    pub paths: Vec<String>,

    /// Maximum simultaneous failure budget for BMC.
    #[arg(long, default_value_t = 3)]
    pub max_failures: usize,

    /// Timeout in seconds per BMC query.
    #[arg(long, default_value_t = 60)]
    pub timeout: u64,

    /// Show full propagation traces.
    #[arg(long)]
    pub traces: bool,

    #[command(flatten)]
    pub output: OutputArgs,
}

impl AnalyzeArgs {
    pub fn validate(&self) -> Result<(), String> {
        if self.paths.is_empty() {
            return Err("at least one config path is required".into());
        }
        if self.max_failures == 0 {
            return Err("--max-failures must be at least 1".into());
        }
        if self.timeout == 0 {
            return Err("--timeout must be at least 1 second".into());
        }
        self.output.parsed_format()?;
        Ok(())
    }
}

/// Arguments for the `diff` subcommand.
#[derive(Parser, Debug, Clone)]
pub struct DiffArgs {
    /// Base (before) configuration paths.
    #[arg(long = "base", required = true)]
    pub base_paths: Vec<String>,

    /// Changed (after) configuration paths.
    #[arg(long = "changed", required = true)]
    pub changed_paths: Vec<String>,

    /// Only report findings that are *new* compared to the base.
    #[arg(long)]
    pub new_only: bool,

    #[command(flatten)]
    pub output: OutputArgs,
}

impl DiffArgs {
    pub fn validate(&self) -> Result<(), String> {
        if self.base_paths.is_empty() {
            return Err("--base requires at least one path".into());
        }
        if self.changed_paths.is_empty() {
            return Err("--changed requires at least one path".into());
        }
        self.output.parsed_format()?;
        Ok(())
    }
}

/// Arguments for the `report` subcommand.
#[derive(Parser, Debug, Clone)]
pub struct ReportArgs {
    /// Directory containing cached analysis results.
    #[arg(long, default_value = ".cascade-cache")]
    pub cache_dir: String,

    /// Output format for the generated report.
    #[arg(long, default_value = "markdown")]
    pub format: String,

    /// Only include findings at or above this severity.
    #[arg(long, default_value = "low")]
    pub min_severity: String,

    /// Title for the report.
    #[arg(long, default_value = "CascadeVerify Report")]
    pub title: String,

    #[command(flatten)]
    pub output: OutputArgs,
}

impl ReportArgs {
    pub fn validate(&self) -> Result<(), String> {
        OutputFormat::from_str_loose(&self.format)?;
        Ok(())
    }

    pub fn parsed_format(&self) -> OutputFormat {
        OutputFormat::from_str_loose(&self.format).unwrap_or(OutputFormat::Markdown)
    }
}

/// Arguments for the `benchmark` subcommand.
#[derive(Parser, Debug, Clone)]
pub struct BenchmarkArgs {
    /// Topology shape: `chain`, `fanout`, `mesh`, `star`, `random`.
    #[arg(long, default_value = "chain")]
    pub topology: String,

    /// Service-count sizes to benchmark.
    #[arg(long, value_delimiter = ',', default_values_t = vec![10, 50, 100, 500])]
    pub sizes: Vec<usize>,

    /// Number of iterations per size for statistical stability.
    #[arg(long, default_value_t = 3)]
    pub iterations: usize,

    /// Include Tier 2 (BMC) in the benchmark (slow).
    #[arg(long)]
    pub include_tier2: bool,

    #[command(flatten)]
    pub output: OutputArgs,
}

impl BenchmarkArgs {
    pub fn validate(&self) -> Result<(), String> {
        match self.topology.as_str() {
            "chain" | "fanout" | "mesh" | "star" | "random" => {}
            other => {
                return Err(format!(
                    "unknown topology '{other}': expected chain, fanout, mesh, star, or random"
                ));
            }
        }
        if self.sizes.is_empty() {
            return Err("--sizes must contain at least one value".into());
        }
        for &s in &self.sizes {
            if s == 0 {
                return Err("each --sizes value must be > 0".into());
            }
        }
        if self.iterations == 0 {
            return Err("--iterations must be at least 1".into());
        }
        self.output.parsed_format()?;
        Ok(())
    }

    /// Parse the topology string.
    pub fn parsed_topology(&self) -> BenchmarkTopology {
        match self.topology.as_str() {
            "fanout" => BenchmarkTopology::FanOut,
            "mesh" => BenchmarkTopology::Mesh,
            "star" => BenchmarkTopology::Star,
            "random" => BenchmarkTopology::Random,
            _ => BenchmarkTopology::Chain,
        }
    }
}

/// Synthetic topology shapes for benchmarking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BenchmarkTopology {
    Chain,
    FanOut,
    Mesh,
    Star,
    Random,
}

impl std::fmt::Display for BenchmarkTopology {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Chain => f.write_str("chain"),
            Self::FanOut => f.write_str("fanout"),
            Self::Mesh => f.write_str("mesh"),
            Self::Star => f.write_str("star"),
            Self::Random => f.write_str("random"),
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<CascadeCli, clap::Error> {
        CascadeCli::try_parse_from(args)
    }

    // -- OutputFormat --

    #[test]
    fn output_format_from_str_table() {
        assert_eq!(OutputFormat::from_str_loose("table").unwrap(), OutputFormat::Table);
        assert_eq!(OutputFormat::from_str_loose("human").unwrap(), OutputFormat::Table);
        assert_eq!(OutputFormat::from_str_loose("text").unwrap(), OutputFormat::Table);
    }

    #[test]
    fn output_format_from_str_json() {
        assert_eq!(OutputFormat::from_str_loose("json").unwrap(), OutputFormat::Json);
    }

    #[test]
    fn output_format_from_str_yaml_aliases() {
        assert_eq!(OutputFormat::from_str_loose("yaml").unwrap(), OutputFormat::Yaml);
        assert_eq!(OutputFormat::from_str_loose("yml").unwrap(), OutputFormat::Yaml);
    }

    #[test]
    fn output_format_unknown_returns_err() {
        assert!(OutputFormat::from_str_loose("foobar").is_err());
    }

    #[test]
    fn output_format_extension() {
        assert_eq!(OutputFormat::Table.extension(), "txt");
        assert_eq!(OutputFormat::Json.extension(), "json");
        assert_eq!(OutputFormat::Sarif.extension(), "sarif.json");
        assert_eq!(OutputFormat::JUnit.extension(), "xml");
        assert_eq!(OutputFormat::Markdown.extension(), "md");
    }

    #[test]
    fn output_format_display() {
        assert_eq!(format!("{}", OutputFormat::Table), "table");
        assert_eq!(format!("{}", OutputFormat::Json), "json");
    }

    // -- AnalysisTier --

    #[test]
    fn mode_args_tier_parsing() {
        let m = ModeArgs { tier: "tier1".into(), incremental: false, cache: true };
        assert_eq!(m.parsed_tier().unwrap(), AnalysisTier::Tier1);

        let m2 = ModeArgs { tier: "2".into(), incremental: false, cache: true };
        assert_eq!(m2.parsed_tier().unwrap(), AnalysisTier::Tier2);

        let m3 = ModeArgs { tier: "auto".into(), incremental: false, cache: true };
        assert_eq!(m3.parsed_tier().unwrap(), AnalysisTier::Auto);
    }

    #[test]
    fn mode_args_unknown_tier_errors() {
        let m = ModeArgs { tier: "tier99".into(), incremental: false, cache: true };
        assert!(m.parsed_tier().is_err());
    }

    // -- PolicyArgs --

    #[test]
    fn policy_should_fail_on_critical() {
        let p = PolicyArgs { fail_on_critical: true, fail_on_high: false, max_medium: 10 };
        assert!(p.should_fail(1, 0, 0));
        assert!(!p.should_fail(0, 5, 0));
    }

    #[test]
    fn policy_should_fail_on_high() {
        let p = PolicyArgs { fail_on_critical: false, fail_on_high: true, max_medium: 10 };
        assert!(!p.should_fail(0, 0, 5));
        assert!(p.should_fail(0, 1, 0));
    }

    #[test]
    fn policy_should_fail_on_medium_threshold() {
        let p = PolicyArgs { fail_on_critical: false, fail_on_high: false, max_medium: 5 };
        assert!(!p.should_fail(0, 0, 5));
        assert!(p.should_fail(0, 0, 6));
    }

    // -- VerifyArgs validation --

    #[test]
    fn verify_args_validate_ok() {
        let a = VerifyArgs {
            paths: vec!["a.yaml".into()],
            output: OutputArgs::default(),
            mode: ModeArgs::default(),
            policy: PolicyArgs::default(),
        };
        assert!(a.validate().is_ok());
    }

    // -- RepairArgs --

    #[test]
    fn repair_strategy_parsing() {
        let r = RepairArgs {
            paths: vec!["a.yaml".into()],
            strategy: "optimal".into(),
            max_changes: 5,
            top_n: 5,
            emit_diff: false,
            output: OutputArgs::default(),
        };
        assert_eq!(r.parsed_strategy(), RepairStrategy::Optimal);
    }

    #[test]
    fn repair_validation_rejects_zero_changes() {
        let r = RepairArgs {
            paths: vec!["a.yaml".into()],
            strategy: "greedy".into(),
            max_changes: 0,
            top_n: 5,
            emit_diff: false,
            output: OutputArgs::default(),
        };
        assert!(r.validate().is_err());
    }

    #[test]
    fn repair_validation_rejects_unknown_strategy() {
        let r = RepairArgs {
            paths: vec!["a.yaml".into()],
            strategy: "magic".into(),
            max_changes: 5,
            top_n: 5,
            emit_diff: false,
            output: OutputArgs::default(),
        };
        assert!(r.validate().is_err());
    }

    // -- CheckArgs --

    #[test]
    fn check_parse_fail_on() {
        assert_eq!(CheckArgs::parse_fail_on("critical").unwrap(), FailOnLevel::Critical);
        assert_eq!(CheckArgs::parse_fail_on("high").unwrap(), FailOnLevel::High);
        assert_eq!(CheckArgs::parse_fail_on("medium").unwrap(), FailOnLevel::Medium);
        assert_eq!(CheckArgs::parse_fail_on("low").unwrap(), FailOnLevel::Low);
    }

    #[test]
    fn check_parse_fail_on_unknown_errors() {
        assert!(CheckArgs::parse_fail_on("fatal").is_err());
    }

    #[test]
    fn fail_on_level_exceeds_threshold() {
        assert!(FailOnLevel::Critical.exceeds_threshold(FailOnLevel::High));
        assert!(FailOnLevel::High.exceeds_threshold(FailOnLevel::High));
        assert!(!FailOnLevel::Medium.exceeds_threshold(FailOnLevel::High));
    }

    // -- BenchmarkArgs --

    #[test]
    fn benchmark_topology_parsing() {
        let b = BenchmarkArgs {
            topology: "mesh".into(),
            sizes: vec![10, 50],
            iterations: 3,
            include_tier2: false,
            output: OutputArgs::default(),
        };
        assert_eq!(b.parsed_topology(), BenchmarkTopology::Mesh);
    }

    #[test]
    fn benchmark_validation_rejects_empty_sizes() {
        let b = BenchmarkArgs {
            topology: "chain".into(),
            sizes: vec![],
            iterations: 3,
            include_tier2: false,
            output: OutputArgs::default(),
        };
        assert!(b.validate().is_err());
    }

    #[test]
    fn benchmark_validation_rejects_zero_size() {
        let b = BenchmarkArgs {
            topology: "chain".into(),
            sizes: vec![0],
            iterations: 3,
            include_tier2: false,
            output: OutputArgs::default(),
        };
        assert!(b.validate().is_err());
    }

    #[test]
    fn benchmark_validation_rejects_zero_iterations() {
        let b = BenchmarkArgs {
            topology: "chain".into(),
            sizes: vec![10],
            iterations: 0,
            include_tier2: false,
            output: OutputArgs::default(),
        };
        assert!(b.validate().is_err());
    }

    // -- AnalysisTier display --

    #[test]
    fn analysis_tier_display() {
        assert_eq!(format!("{}", AnalysisTier::Tier1), "tier1");
        assert_eq!(format!("{}", AnalysisTier::Tier2), "tier2");
        assert_eq!(format!("{}", AnalysisTier::Auto), "auto");
    }

    // -- Quiet + verbose mutually exclusive --

    #[test]
    fn quiet_and_verbose_are_exclusive() {
        let cli = CascadeCli {
            verbose: 1,
            quiet: true,
            config: None,
            no_color: false,
            command: Commands::Check(CheckArgs {
                paths: vec!["a.yaml".into()],
                fail_on: "critical".into(),
                annotations: false,
                step_summary: false,
                output: OutputArgs::default(),
            }),
        };
        assert!(cli.validate().is_err());
    }

    // -- Full round-trip parse tests --

    #[test]
    fn full_verify_parse() {
        let cli = parse(&[
            "cascade-verify", "verify",
            "--format", "json",
            "--tier", "tier2",
            "--fail-on-critical",
            "a.yaml", "b.yaml",
        ]).unwrap();
        if let Commands::Verify(args) = cli.command {
            assert_eq!(args.paths, vec!["a.yaml", "b.yaml"]);
            assert_eq!(args.output.format, "json");
            assert_eq!(args.mode.tier, "tier2");
            assert!(args.policy.fail_on_critical);
        } else {
            panic!("expected Verify");
        }
    }

    #[test]
    fn full_repair_parse() {
        let cli = parse(&[
            "cascade-verify", "repair",
            "--strategy", "pareto",
            "--max-changes", "7",
            "--emit-diff",
            "svc.yaml",
        ]).unwrap();
        if let Commands::Repair(args) = cli.command {
            assert_eq!(args.strategy, "pareto");
            assert_eq!(args.max_changes, 7);
            assert!(args.emit_diff);
        } else {
            panic!("expected Repair");
        }
    }

    #[test]
    fn full_benchmark_parse() {
        let cli = parse(&[
            "cascade-verify", "benchmark",
            "--topology", "star",
            "--sizes", "10,20,30",
            "--iterations", "5",
        ]).unwrap();
        if let Commands::Benchmark(args) = cli.command {
            assert_eq!(args.topology, "star");
            assert_eq!(args.sizes, vec![10, 20, 30]);
            assert_eq!(args.iterations, 5);
        } else {
            panic!("expected Benchmark");
        }
    }

    #[test]
    fn benchmark_topology_display() {
        assert_eq!(format!("{}", BenchmarkTopology::Chain), "chain");
        assert_eq!(format!("{}", BenchmarkTopology::FanOut), "fanout");
        assert_eq!(format!("{}", BenchmarkTopology::Mesh), "mesh");
        assert_eq!(format!("{}", BenchmarkTopology::Star), "star");
        assert_eq!(format!("{}", BenchmarkTopology::Random), "random");
    }
}
