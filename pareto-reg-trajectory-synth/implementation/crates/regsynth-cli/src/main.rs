use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

mod commands;
mod config;
mod output;
mod pipeline;

/// RegSynth — Regulatory Compliance Analysis Pipeline
///
/// Parses regulatory DSL files, encodes obligations into constraints,
/// solves for feasibility, computes Pareto-optimal compliance strategies,
/// generates remediation roadmaps, and produces verifiable certificates.
#[derive(Parser, Debug)]
#[command(name = "regsynth", version, about, long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,

    /// Path to TOML configuration file
    #[arg(long, short = 'c', global = true)]
    pub config: Option<PathBuf>,

    /// Enable verbose output (repeat for more: -v, -vv, -vvv)
    #[arg(long, short = 'v', global = true, action = clap::ArgAction::Count)]
    pub verbose: u8,

    /// Output format
    #[arg(long, global = true, default_value = "text")]
    pub output_format: OutputFormat,

    /// Solver backend to use
    #[arg(long, global = true)]
    pub solver: Option<SolverBackend>,

    /// Solver timeout in seconds
    #[arg(long, global = true)]
    pub timeout: Option<u64>,

    /// Output file path (defaults to stdout)
    #[arg(long, short = 'o', global = true)]
    pub output: Option<PathBuf>,
}

#[derive(Debug, Clone, ValueEnum)]
pub enum OutputFormat {
    Text,
    Json,
    Csv,
}

#[derive(Debug, Clone, ValueEnum)]
pub enum SolverBackend {
    Sat,
    Smt,
    MaxSmt,
    Ilp,
}

impl std::fmt::Display for SolverBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Sat => write!(f, "SAT"),
            Self::Smt => write!(f, "SMT"),
            Self::MaxSmt => write!(f, "MaxSMT"),
            Self::Ilp => write!(f, "ILP"),
        }
    }
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Run the full analysis pipeline on regulatory DSL files
    Analyze {
        /// Input DSL file(s)
        #[arg(required = true)]
        files: Vec<PathBuf>,

        /// Skip certificate generation
        #[arg(long)]
        no_certify: bool,

        /// Maximum Pareto iterations
        #[arg(long, default_value = "100")]
        max_iterations: usize,

        /// Epsilon tolerance for Pareto dominance
        #[arg(long, default_value = "0.001")]
        epsilon: f64,
    },

    /// Type-check regulatory DSL files without running the solver
    Check {
        /// Input DSL file(s)
        #[arg(required = true)]
        files: Vec<PathBuf>,

        /// Report warnings in addition to errors
        #[arg(long)]
        warnings: bool,
    },

    /// Encode obligations to SMT/ILP constraints
    Encode {
        /// Input DSL file(s)
        #[arg(required = true)]
        files: Vec<PathBuf>,

        /// Encoding target format
        #[arg(long, default_value = "smt")]
        target: EncodingTarget,

        /// Include soft constraints for MaxSMT
        #[arg(long)]
        soft_constraints: bool,
    },

    /// Run the solver on encoded constraints
    Solve {
        /// Input constraint file (JSON)
        #[arg(required = true)]
        input: PathBuf,

        /// Extract conflict core on UNSAT
        #[arg(long)]
        extract_conflicts: bool,

        /// Maximum solver iterations
        #[arg(long, default_value = "10000")]
        max_iterations: u64,
    },

    /// Compute the Pareto frontier of compliance strategies
    Pareto {
        /// Input solver results file (JSON)
        #[arg(required = true)]
        input: PathBuf,

        /// Epsilon tolerance for Pareto dominance
        #[arg(long, default_value = "0.001")]
        epsilon: f64,

        /// Maximum number of Pareto iterations
        #[arg(long, default_value = "100")]
        max_iterations: usize,

        /// Objective names (comma-separated)
        #[arg(long, default_value = "cost,compliance,risk")]
        objectives: String,
    },

    /// Generate a remediation roadmap from a Pareto strategy
    Plan {
        /// Input Pareto frontier file (JSON)
        #[arg(required = true)]
        input: PathBuf,

        /// Index of the strategy to use (0-based)
        #[arg(long, default_value = "0")]
        strategy_index: usize,

        /// Maximum parallel tasks in schedule
        #[arg(long, default_value = "4")]
        max_parallel: usize,

        /// Planning start date (YYYY-MM-DD)
        #[arg(long)]
        start_date: Option<String>,
    },

    /// Generate compliance/infeasibility/Pareto certificates
    Certify {
        /// Input analysis results file (JSON)
        #[arg(required = true)]
        input: PathBuf,

        /// Certificate type
        #[arg(long, default_value = "compliance")]
        cert_type: CertificateType,

        /// Certificate subject description
        #[arg(long, default_value = "Regulatory compliance analysis")]
        subject: String,
    },

    /// Verify a previously generated certificate
    Verify {
        /// Input certificate file (JSON)
        #[arg(required = true)]
        certificate: PathBuf,

        /// Perform deep proof chain verification
        #[arg(long)]
        deep: bool,
    },

    /// Run benchmarks with synthetic regulatory problems
    Benchmark {
        /// Number of obligations to generate
        #[arg(long, default_value = "50")]
        num_obligations: usize,

        /// Number of jurisdictions
        #[arg(long, default_value = "3")]
        num_jurisdictions: usize,

        /// Number of benchmark iterations
        #[arg(long, default_value = "5")]
        iterations: usize,

        /// Conflict density (0.0 to 1.0)
        #[arg(long, default_value = "0.3")]
        conflict_density: f64,

        /// Seed for random generation
        #[arg(long)]
        seed: Option<u64>,
    },

    /// Export analysis results to a file
    Export {
        /// Input results file (JSON)
        #[arg(required = true)]
        input: PathBuf,

        /// Export format (overrides global --output-format)
        #[arg(long)]
        format: Option<OutputFormat>,

        /// Include detailed solver output
        #[arg(long)]
        detailed: bool,
    },
}

#[derive(Debug, Clone, ValueEnum)]
pub enum EncodingTarget {
    Smt,
    Ilp,
    Both,
}

#[derive(Debug, Clone, ValueEnum)]
pub enum CertificateType {
    Compliance,
    Infeasibility,
    Pareto,
}

fn init_logging(verbosity: u8) {
    let level = match verbosity {
        0 => log::LevelFilter::Warn,
        1 => log::LevelFilter::Info,
        2 => log::LevelFilter::Debug,
        _ => log::LevelFilter::Trace,
    };
    env_logger::Builder::new()
        .filter_level(level)
        .format_timestamp_millis()
        .init();
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    init_logging(cli.verbose);

    log::info!("RegSynth v{}", env!("CARGO_PKG_VERSION"));

    let cfg = config::AppConfig::load(cli.config.as_deref(), &cli)?;
    log::debug!("Configuration loaded: {:?}", cfg);

    let formatter = output::OutputFormatter::new(
        cli.output_format.clone(),
        cli.output.clone(),
    );

    match cli.command {
        Command::Analyze {
            files,
            no_certify,
            max_iterations,
            epsilon,
        } => commands::analyze::run(&cfg, &formatter, &files, no_certify, max_iterations, epsilon),

        Command::Check { files, warnings } => {
            commands::check::run(&cfg, &formatter, &files, warnings)
        }

        Command::Encode {
            files,
            target,
            soft_constraints,
        } => commands::encode::run(&cfg, &formatter, &files, &target, soft_constraints),

        Command::Solve {
            input,
            extract_conflicts,
            max_iterations,
        } => commands::solve::run(&cfg, &formatter, &input, extract_conflicts, max_iterations),

        Command::Pareto {
            input,
            epsilon,
            max_iterations,
            objectives,
        } => commands::pareto::run(&cfg, &formatter, &input, epsilon, max_iterations, &objectives),

        Command::Plan {
            input,
            strategy_index,
            max_parallel,
            start_date,
        } => commands::plan::run(
            &cfg,
            &formatter,
            &input,
            strategy_index,
            max_parallel,
            start_date.as_deref(),
        ),

        Command::Certify {
            input,
            cert_type,
            subject,
        } => commands::certify::run(&cfg, &formatter, &input, &cert_type, &subject),

        Command::Verify { certificate, deep } => {
            commands::verify::run(&cfg, &formatter, &certificate, deep)
        }

        Command::Benchmark {
            num_obligations,
            num_jurisdictions,
            iterations,
            conflict_density,
            seed,
        } => commands::benchmark::run(
            &cfg,
            &formatter,
            num_obligations,
            num_jurisdictions,
            iterations,
            conflict_density,
            seed,
        ),

        Command::Export {
            input,
            format,
            detailed,
        } => {
            let fmt = format.as_ref().unwrap_or(&cli.output_format);
            let export_formatter = output::OutputFormatter::new(fmt.clone(), cli.output.clone());
            commands::export::run(&cfg, &export_formatter, &input, detailed)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn verify_cli_structure() {
        Cli::command().debug_assert();
    }

    #[test]
    fn test_solver_backend_display() {
        assert_eq!(SolverBackend::Sat.to_string(), "SAT");
        assert_eq!(SolverBackend::MaxSmt.to_string(), "MaxSMT");
    }

    #[test]
    fn test_init_logging_levels() {
        let levels = [
            (0u8, log::LevelFilter::Warn),
            (1, log::LevelFilter::Info),
            (2, log::LevelFilter::Debug),
            (3, log::LevelFilter::Trace),
        ];
        for (v, expected) in levels {
            let level = match v {
                0 => log::LevelFilter::Warn,
                1 => log::LevelFilter::Info,
                2 => log::LevelFilter::Debug,
                _ => log::LevelFilter::Trace,
            };
            assert_eq!(level, expected);
        }
    }
}
