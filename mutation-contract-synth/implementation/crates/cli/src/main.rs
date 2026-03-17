//! MutSpec CLI – command-line interface for mutation-contract synthesis.
//!
//! This binary (`mutspec`) is the primary user entry-point for the MutSpec
//! tool-chain.  It wires together every workspace crate and exposes the full
//! analysis pipeline through a set of composable sub-commands.

mod commands;
mod config;
mod output;

use std::path::PathBuf;
use std::process;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use log::{debug, error, info, LevelFilter};

use crate::commands::analyze::AnalyzeArgs;
use crate::commands::config::ConfigArgs;
use crate::commands::mutate::MutateArgs;
use crate::commands::report::ReportArgs;
use crate::commands::synthesize::SynthesizeArgs;
use crate::commands::verify::VerifyArgs;
use crate::config::CliConfig;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const VERSION: &str = env!("CARGO_PKG_VERSION");
const PKG_NAME: &str = env!("CARGO_PKG_NAME");

const LONG_ABOUT: &str = "\
MutSpec – Mutation-Specification Duality Engine

MutSpec discovers latent specification gaps in loop-free, first-order
imperative programs over QF-LIA (quantifier-free linear integer arithmetic)
by exploiting the formal duality between mutations and contracts.

The tool pipeline:
  1. Parse source into an AST
  2. Generate mutants via configurable mutation operators
  3. Build a kill matrix (which contract kills which mutant)
  4. Synthesize missing contracts to close specification gaps
  5. Verify synthesized contracts via SMT
  6. Produce human-readable and machine-readable reports

EXAMPLES:
  mutspec analyze program.ms
  mutspec mutate program.ms --operators AOR,ROR --format json
  mutspec synthesize mutants.json --tier 1 -o contracts.json
  mutspec verify contracts.json
  mutspec report --input results.json --format sarif -o report.sarif
  mutspec config init
  mutspec config show";

// ---------------------------------------------------------------------------
// CLI top-level
// ---------------------------------------------------------------------------

/// MutSpec: mutation-contract synthesis for specification-gap detection.
#[derive(Debug, Parser)]
#[command(
    name = "mutspec",
    version = VERSION,
    about = "Mutation-contract synthesis for specification-gap detection",
    long_about = LONG_ABOUT,
    propagate_version = true,
    arg_required_else_help = true,
)]
pub struct MutspecCli {
    /// Verbosity level (-v, -vv, -vvv).
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    pub verbose: u8,

    /// Suppress all output except errors.
    #[arg(short, long, global = true, conflicts_with = "verbose")]
    pub quiet: bool,

    /// Path to configuration file.
    #[arg(short, long, global = true, value_name = "FILE")]
    pub config: Option<PathBuf>,

    /// Number of parallel workers (0 = auto-detect).
    #[arg(long, global = true, default_value_t = 0)]
    pub parallelism: usize,

    /// Disable colored output.
    #[arg(long, global = true)]
    pub no_color: bool,

    #[command(subcommand)]
    pub command: Command,
}

/// Available sub-commands.
#[derive(Debug, Subcommand)]
pub enum Command {
    /// Run the full analysis pipeline (parse → mutate → synthesize → report).
    #[command(alias = "a")]
    Analyze(AnalyzeArgs),

    /// Generate mutants from a source program.
    #[command(alias = "m")]
    Mutate(MutateArgs),

    /// Synthesize contracts from a kill matrix.
    #[command(alias = "s")]
    Synthesize(SynthesizeArgs),

    /// Verify synthesized contracts via SMT.
    #[command(alias = "v")]
    Verify(VerifyArgs),

    /// Generate reports in various formats.
    #[command(alias = "r")]
    Report(ReportArgs),

    /// Manage configuration files.
    Config(ConfigArgs),

    /// Print version and build information.
    Version,
}

// ---------------------------------------------------------------------------
// Logging
// ---------------------------------------------------------------------------

fn init_logging(verbosity: u8, quiet: bool) {
    let filter = if quiet {
        LevelFilter::Error
    } else {
        match verbosity {
            0 => LevelFilter::Warn,
            1 => LevelFilter::Info,
            2 => LevelFilter::Debug,
            _ => LevelFilter::Trace,
        }
    };

    env_logger::Builder::new()
        .filter_level(filter)
        .format_timestamp_millis()
        .format_module_path(verbosity >= 2)
        .format_target(false)
        .init();

    debug!("Log level set to {filter}");
}

// ---------------------------------------------------------------------------
// Parallelism
// ---------------------------------------------------------------------------

fn configure_parallelism(requested: usize) -> usize {
    let cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    let effective = if requested == 0 {
        cpus
    } else {
        requested.min(cpus * 2)
    };

    rayon::ThreadPoolBuilder::new()
        .num_threads(effective)
        .build_global()
        .ok();

    info!("Using {effective} worker thread(s) ({cpus} CPU(s) detected)");
    effective
}

// ---------------------------------------------------------------------------
// Output destination
// ---------------------------------------------------------------------------

#[allow(dead_code)]
fn open_output(path: &Option<PathBuf>) -> Result<Box<dyn std::io::Write>> {
    match path {
        Some(p) => {
            let file = std::fs::File::create(p)
                .with_context(|| format!("Cannot create output file: {}", p.display()))?;
            Ok(Box::new(std::io::BufWriter::new(file)))
        }
        None => Ok(Box::new(std::io::stdout().lock())),
    }
}

// ---------------------------------------------------------------------------
// Version
// ---------------------------------------------------------------------------

fn print_version_info() {
    println!("mutspec {VERSION}");
    println!();
    println!("  Package:       {PKG_NAME}");
    println!("  Version:       {VERSION}");
    println!("  Rust edition:  2021");
    println!("  License:       MIT OR Apache-2.0");
    println!("  Authors:       MutSpec Team");
    println!();
    println!("Workspace crates:");
    println!("  shared-types   – Core type definitions");
    println!("  mutation-core  – Mutation generation engine");
    println!("  program-analysis – Parsing, CFG, SSA, WP");
    println!("  smt-solver     – SMT solver interface (Z3)");
    println!("  contract-synth – Contract synthesis");
    println!("  coverage       – Mutation coverage analysis");
    println!("  test-gen       – Test-case generation");
    println!();
    println!("Build info:");
    println!("  Target:   {}", std::env::consts::ARCH);
    println!("  OS:       {}", std::env::consts::OS);
    println!("  Family:   {}", std::env::consts::FAMILY);

    if let Ok(cpus) = std::thread::available_parallelism() {
        println!("  CPUs:     {}", cpus.get());
    }
}

// ---------------------------------------------------------------------------
// Run
// ---------------------------------------------------------------------------

fn run(cli: MutspecCli) -> Result<()> {
    init_logging(cli.verbose, cli.quiet);

    let started = Instant::now();

    let cfg = CliConfig::load(cli.config.as_deref(), cli.verbose > 0)?;
    let _parallelism = configure_parallelism(cli.parallelism);

    let result = match cli.command {
        Command::Analyze(ref args) => commands::analyze::run(args, &cfg),
        Command::Mutate(ref args) => commands::mutate::run(args, &cfg),
        Command::Synthesize(ref args) => commands::synthesize::run(args, &cfg),
        Command::Verify(ref args) => commands::verify::run(args, &cfg),
        Command::Report(ref args) => commands::report::run(args, &cfg),
        Command::Config(ref args) => commands::config::run(args, &cfg),
        Command::Version => {
            print_version_info();
            Ok(())
        }
    };

    let elapsed = started.elapsed();
    info!("Completed in {:.2}s", elapsed.as_secs_f64());

    result
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() {
    let cli = MutspecCli::parse();

    if let Err(err) = run(cli) {
        error!("{err:#}");

        let chain: Vec<_> = err.chain().skip(1).collect();
        if !chain.is_empty() {
            eprintln!();
            eprintln!("Error: {err}");
            for (i, cause) in chain.iter().enumerate() {
                eprintln!("  {}: {cause}", i + 1);
            }
        } else {
            eprintln!("Error: {err}");
        }

        process::exit(1);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn verify_cli_structure() {
        MutspecCli::command().debug_assert();
    }

    #[test]
    fn test_version_string() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn test_open_output_stdout() {
        let w = open_output(&None);
        assert!(w.is_ok());
    }

    #[test]
    fn test_configure_parallelism_auto() {
        let n = configure_parallelism(0);
        assert!(n >= 1);
    }

    #[test]
    fn test_long_about_not_empty() {
        assert!(!LONG_ABOUT.is_empty());
    }
}
