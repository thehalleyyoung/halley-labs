//! CLI entry point for the Certified Leakage Contracts framework.

use std::path::PathBuf;

use clap::{Parser, Subcommand};

use leak_cli::{CliError, CliResult};

/// Certified Leakage Contracts — static analysis for cache side-channel leakage.
#[derive(Debug, Parser)]
#[command(name = "leakage-contracts", version, about, long_about = None)]
struct Cli {
    /// Enable verbose logging output.
    #[arg(short, long, global = true)]
    verbose: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Run leakage analysis on a binary.
    Analyze {
        /// Path to the input binary to analyze.
        #[arg(short, long)]
        input: PathBuf,

        /// Path to write analysis results.
        #[arg(short, long)]
        output: PathBuf,

        /// Optional path to a configuration file.
        #[arg(short, long)]
        config: Option<PathBuf>,
    },

    /// Compose leakage contracts from multiple sources.
    Compose {
        /// Paths to contract files to compose.
        #[arg(required = true)]
        contracts: Vec<PathBuf>,
    },

    /// Generate or verify leakage certificates.
    Certify {
        /// Path to the input contract or binary.
        #[arg(short, long)]
        input: PathBuf,

        /// Verify an existing certificate instead of generating one.
        #[arg(long)]
        verify: bool,
    },

    /// Check for regressions between contract versions.
    Regression {
        /// Path to the baseline contract.
        #[arg(short, long)]
        baseline: PathBuf,

        /// Path to the current contract to compare against baseline.
        #[arg(short, long)]
        current: PathBuf,
    },
}

fn main() -> CliResult<()> {
    let cli = Cli::parse();

    if cli.verbose {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug")).init();
    } else {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    }

    match cli.command {
        Commands::Analyze {
            input,
            output,
            config,
        } => cmd_analyze(input, output, config),
        Commands::Compose { contracts } => cmd_compose(contracts),
        Commands::Certify { input, verify } => cmd_certify(input, verify),
        Commands::Regression { baseline, current } => cmd_regression(baseline, current),
    }
}

fn cmd_analyze(input: PathBuf, output: PathBuf, config: Option<PathBuf>) -> CliResult<()> {
    log::info!("Analyzing binary: {}", input.display());
    if let Some(cfg) = &config {
        log::info!("Using configuration: {}", cfg.display());
    }
    println!(
        "Analysis not yet implemented (input={}, output={})",
        input.display(),
        output.display()
    );
    Ok(())
}

fn cmd_compose(contracts: Vec<PathBuf>) -> CliResult<()> {
    log::info!("Composing {} contracts", contracts.len());
    println!(
        "Composition not yet implemented ({} contract files)",
        contracts.len()
    );
    Ok(())
}

fn cmd_certify(input: PathBuf, verify: bool) -> CliResult<()> {
    if verify {
        log::info!("Verifying certificate: {}", input.display());
        println!(
            "Certificate verification not yet implemented (input={})",
            input.display()
        );
    } else {
        log::info!("Generating certificate for: {}", input.display());
        println!(
            "Certificate generation not yet implemented (input={})",
            input.display()
        );
    }
    Ok(())
}

fn cmd_regression(baseline: PathBuf, current: PathBuf) -> CliResult<()> {
    log::info!(
        "Regression check: baseline={}, current={}",
        baseline.display(),
        current.display()
    );
    println!(
        "Regression check not yet implemented (baseline={}, current={})",
        baseline.display(),
        current.display()
    );
    Ok(())
}
