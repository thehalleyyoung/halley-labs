//! CLI entry point for the NLP metamorphic fault localizer.

mod commands;
mod config;
mod output;

use clap::{Parser, Subcommand};

/// NLP Metamorphic Fault Localizer – pinpoint faulty pipeline stages
/// using differential testing and causal analysis.
#[derive(Parser)]
#[command(
    name = "nlp-localizer",
    version,
    about = "NLP metamorphic fault localization CLI",
    long_about = "Localize faults in NLP pipelines using metamorphic testing, \
                  differential analysis, and causal decomposition."
)]
struct Cli {
    /// Path to a TOML configuration file.
    #[arg(short, long, global = true)]
    config: Option<String>,

    /// Verbosity level (repeat for more: -v, -vv, -vvv).
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    verbose: u8,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run fault localization on a pipeline.
    Localize(commands::LocalizeCommand),

    /// Run calibration to establish baseline differentials.
    Calibrate(commands::CalibrateCommand),

    /// Shrink a counterexample to a minimal violation-inducing input.
    Shrink(commands::ShrinkCommand),

    /// Generate a report from localization results.
    Report(commands::ReportCommand),

    /// Validate grammar of a sentence or transformation output.
    Validate(commands::ValidateCommand),

    /// Generate a behavioral atlas from results.
    Atlas(commands::AtlasCommand),

    /// Run a localization benchmark.
    Benchmark(commands::BenchmarkCommand),
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    let log_level = match cli.verbose {
        0 => "warn",
        1 => "info",
        2 => "debug",
        _ => "trace",
    };
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(log_level))
        .init();

    log::info!("NLP Metamorphic Fault Localizer starting");

    let cfg = if let Some(ref path) = cli.config {
        config::CliConfig::load_from_file(path)?
    } else {
        config::CliConfig::default()
    };

    match cli.command {
        Commands::Localize(cmd) => commands::run_localize(cmd, &cfg)?,
        Commands::Calibrate(cmd) => commands::run_calibrate(cmd, &cfg)?,
        Commands::Shrink(cmd) => commands::run_shrink(cmd, &cfg)?,
        Commands::Report(cmd) => commands::run_report(cmd, &cfg)?,
        Commands::Validate(cmd) => commands::run_validate(cmd, &cfg)?,
        Commands::Atlas(cmd) => commands::run_atlas(cmd, &cfg)?,
        Commands::Benchmark(cmd) => commands::run_benchmark(cmd, &cfg)?,
    }

    Ok(())
}
