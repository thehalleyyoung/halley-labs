//! IsoSpec CLI - command-line interface for transaction isolation analysis.

use std::path::PathBuf;
use std::process;

use clap::{Parser, Subcommand, ValueEnum};

mod commands;
mod format;
mod input;
mod output;

// ---------------------------------------------------------------------------
// Top-level CLI definition
// ---------------------------------------------------------------------------

/// IsoSpec: Transaction isolation specification verifier.
///
/// Analyzes workloads for anomalies under different database engine isolation
/// levels, checks portability, generates witness schedules, and runs benchmarks.
#[derive(Parser, Debug)]
#[command(name = "isospec", version, about, long_about = None)]
pub struct Cli {
    /// Output verbosity (-v, -vv, -vvv).
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    verbose: u8,

    /// Suppress all non-error output.
    #[arg(short, long, global = true, default_value_t = false)]
    quiet: bool,

    /// Override output format.
    #[arg(long, global = true)]
    format: Option<OutputFormatArg>,

    /// Subcommand to execute.
    #[command(subcommand)]
    command: Command,
}

impl Cli {
    /// Resolved verbosity level (0 = quiet, 1 = normal, 2+ = verbose).
    pub fn verbosity(&self) -> u8 {
        if self.quiet { 0 } else { self.verbose.saturating_add(1) }
    }

    pub fn output_format(&self) -> OutputFormatArg {
        self.format.unwrap_or(OutputFormatArg::Text)
    }
}

// ---------------------------------------------------------------------------
// Subcommands
// ---------------------------------------------------------------------------

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Run anomaly analysis on a workload.
    Analyze(AnalyzeArgs),
    /// Check portability between database engines.
    Portability(PortabilityArgs),
    /// Generate a witness schedule demonstrating an anomaly.
    Witness(WitnessArgs),
    /// Run the benchmark suite.
    Benchmark(BenchmarkArgs),
    /// Validate a witness against a real engine model.
    Validate(ValidateArgs),
    /// Check refinement relations between isolation levels.
    Refinement(RefinementArgs),
}

// ---------------------------------------------------------------------------
// Per-command arguments
// ---------------------------------------------------------------------------

/// Arguments for the `analyze` subcommand.
#[derive(Parser, Debug)]
pub struct AnalyzeArgs {
    /// Path to the workload file (JSON).
    #[arg(short, long)]
    pub workload: PathBuf,

    /// Target database engine.
    #[arg(short, long, default_value = "postgresql")]
    pub engine: EngineArg,

    /// Isolation level to analyse.
    #[arg(short, long, default_value = "serializable")]
    pub isolation: IsolationArg,

    /// Maximum transactions to consider.
    #[arg(long, default_value_t = 10)]
    pub max_txns: usize,

    /// SMT solver timeout in seconds.
    #[arg(long, default_value_t = 60)]
    pub timeout: u64,

    /// Enable predicate-level analysis.
    #[arg(long, default_value_t = false)]
    pub predicates: bool,

    /// Output file (default: stdout).
    #[arg(short, long)]
    pub output: Option<PathBuf>,
}

/// Arguments for the `portability` subcommand.
#[derive(Parser, Debug)]
pub struct PortabilityArgs {
    /// Path to the workload file.
    #[arg(short, long)]
    pub workload: PathBuf,

    /// Source database engine.
    #[arg(long, default_value = "postgresql")]
    pub source_engine: EngineArg,

    /// Source isolation level.
    #[arg(long, default_value = "serializable")]
    pub source_isolation: IsolationArg,

    /// Target database engine.
    #[arg(long, default_value = "mysql")]
    pub target_engine: EngineArg,

    /// Target isolation level.
    #[arg(long, default_value = "repeatable-read")]
    pub target_isolation: IsolationArg,

    /// Generate witness schedules for portability violations.
    #[arg(long, default_value_t = false)]
    pub witnesses: bool,

    /// Maximum number of witnesses to generate.
    #[arg(long, default_value_t = 5)]
    pub max_witnesses: usize,

    /// Output file.
    #[arg(short, long)]
    pub output: Option<PathBuf>,
}

/// Arguments for the `witness` subcommand.
#[derive(Parser, Debug)]
pub struct WitnessArgs {
    /// Path to the workload file.
    #[arg(short, long)]
    pub workload: PathBuf,

    /// Target engine.
    #[arg(short, long, default_value = "postgresql")]
    pub engine: EngineArg,

    /// Isolation level.
    #[arg(short, long, default_value = "read-committed")]
    pub isolation: IsolationArg,

    /// Anomaly class to witness.
    #[arg(short, long)]
    pub anomaly: AnomalyArg,

    /// Maximum witnesses to generate.
    #[arg(long, default_value_t = 1)]
    pub count: usize,

    /// Output file.
    #[arg(short, long)]
    pub output: Option<PathBuf>,
}

/// Arguments for the `benchmark` subcommand.
#[derive(Parser, Debug)]
pub struct BenchmarkArgs {
    /// Benchmark suite to run.
    #[arg(short, long, default_value = "standard")]
    pub suite: BenchmarkSuiteArg,

    /// Number of warm-up iterations.
    #[arg(long, default_value_t = 3)]
    pub warmup: usize,

    /// Number of measurement iterations.
    #[arg(long, default_value_t = 10)]
    pub iterations: usize,

    /// Output directory for reports.
    #[arg(short, long)]
    pub output_dir: Option<PathBuf>,

    /// Output format for the report.
    #[arg(long, default_value = "text")]
    pub report_format: OutputFormatArg,
}

/// Arguments for the `validate` subcommand.
#[derive(Parser, Debug)]
pub struct ValidateArgs {
    /// Path to a witness schedule file (JSON).
    #[arg(short, long)]
    pub witness: PathBuf,

    /// Target engine for validation.
    #[arg(short, long, default_value = "postgresql")]
    pub engine: EngineArg,

    /// Isolation level.
    #[arg(short, long, default_value = "serializable")]
    pub isolation: IsolationArg,

    /// Output file.
    #[arg(short, long)]
    pub output: Option<PathBuf>,
}

/// Arguments for the `refinement` subcommand.
#[derive(Parser, Debug)]
pub struct RefinementArgs {
    /// First engine.
    #[arg(long, default_value = "postgresql")]
    pub engine_a: EngineArg,

    /// Isolation level of first engine.
    #[arg(long, default_value = "serializable")]
    pub level_a: IsolationArg,

    /// Second engine.
    #[arg(long, default_value = "mysql")]
    pub engine_b: EngineArg,

    /// Isolation level of second engine.
    #[arg(long, default_value = "repeatable-read")]
    pub level_b: IsolationArg,

    /// Optional workload for bounded refinement check.
    #[arg(short, long)]
    pub workload: Option<PathBuf>,

    /// Output file.
    #[arg(short, long)]
    pub output: Option<PathBuf>,
}

// ---------------------------------------------------------------------------
// Shared enums
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum EngineArg {
    Postgresql,
    Mysql,
    Sqlserver,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum IsolationArg {
    ReadUncommitted,
    ReadCommitted,
    RepeatableRead,
    Serializable,
    Snapshot,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum AnomalyArg {
    G0,
    G1a,
    G1b,
    G1c,
    G2Item,
    G2,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum BenchmarkSuiteArg {
    Standard,
    Tpcc,
    Tpce,
    Scaling,
    Adversarial,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum OutputFormatArg {
    Text,
    Json,
    Csv,
    Dot,
}

// ---------------------------------------------------------------------------
// Conversion helpers
// ---------------------------------------------------------------------------

impl EngineArg {
    pub fn to_engine_kind(self) -> isospec_types::config::EngineKind {
        match self {
            EngineArg::Postgresql => isospec_types::config::EngineKind::PostgreSQL,
            EngineArg::Mysql => isospec_types::config::EngineKind::MySQL,
            EngineArg::Sqlserver => isospec_types::config::EngineKind::SqlServer,
        }
    }
}

impl IsolationArg {
    pub fn to_isolation_level(self) -> isospec_types::isolation::IsolationLevel {
        match self {
            IsolationArg::ReadUncommitted => isospec_types::isolation::IsolationLevel::ReadUncommitted,
            IsolationArg::ReadCommitted => isospec_types::isolation::IsolationLevel::ReadCommitted,
            IsolationArg::RepeatableRead => isospec_types::isolation::IsolationLevel::RepeatableRead,
            IsolationArg::Serializable => isospec_types::isolation::IsolationLevel::Serializable,
            IsolationArg::Snapshot => isospec_types::isolation::IsolationLevel::Snapshot,
        }
    }
}

impl AnomalyArg {
    pub fn to_anomaly_class(self) -> isospec_types::isolation::AnomalyClass {
        match self {
            AnomalyArg::G0 => isospec_types::isolation::AnomalyClass::G0,
            AnomalyArg::G1a => isospec_types::isolation::AnomalyClass::G1a,
            AnomalyArg::G1b => isospec_types::isolation::AnomalyClass::G1b,
            AnomalyArg::G1c => isospec_types::isolation::AnomalyClass::G1c,
            AnomalyArg::G2Item => isospec_types::isolation::AnomalyClass::G2Item,
            AnomalyArg::G2 => isospec_types::isolation::AnomalyClass::G2,
        }
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let cli = Cli::parse();

    let result = match &cli.command {
        Command::Analyze(args) => commands::execute_analyze(&cli, args),
        Command::Portability(args) => commands::execute_portability(&cli, args),
        Command::Witness(args) => commands::execute_witness(&cli, args),
        Command::Benchmark(args) => commands::execute_benchmark(&cli, args),
        Command::Validate(args) => commands::execute_validate(&cli, args),
        Command::Refinement(args) => commands::execute_refinement(&cli, args),
    };

    if let Err(e) = result {
        eprintln!("Error: {}", e);
        process::exit(1);
    }
}
