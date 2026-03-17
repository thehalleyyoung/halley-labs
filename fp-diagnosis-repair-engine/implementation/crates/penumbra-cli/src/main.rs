//! Penumbra — Floating-point diagnosis and repair engine.
//!
//! This binary provides the command-line interface for the Penumbra tool suite,
//! which traces, diagnoses, repairs, certifies, and optimizes floating-point
//! computations.  Run `penumbra --help` for usage information.

use anyhow::{Context, Result};
use clap::Parser;
use log::{debug, error, info};
use std::time::Instant;

use penumbra_cli::args::{Command, OutputFormat, PenumbraCli};
use penumbra_cli::commands;

// ────────────────────────────────────────────────────────────────────────────
// Entry point
// ────────────────────────────────────────────────────────────────────────────

fn main() {
    let exit_code = match run() {
        Ok(()) => 0,
        Err(e) => {
            let msg = format_error_chain(&e);
            error!("{}", msg);
            eprintln!("{}", msg);
            1
        }
    };
    std::process::exit(exit_code);
}

fn run() -> Result<()> {
    let cli = PenumbraCli::parse();

    init_logging(&cli)?;

    if !cli.quiet {
        print_banner();
    }

    validate_global_args(&cli)?;

    let config = load_or_default_config(&cli)?;

    if let Some(ref out_dir) = cli.output_dir {
        std::fs::create_dir_all(out_dir)
            .with_context(|| format!("Cannot create output directory: {}", out_dir.display()))?;
        debug!("Output directory: {}", out_dir.display());
    }

    let global = penumbra_cli::args::GlobalOpts {
        verbose: cli.verbose,
        quiet: cli.quiet,
        output_format: cli.output_format.clone(),
        output_dir: cli.output_dir.clone(),
    };

    let start = Instant::now();
    let result = dispatch(&cli.command, &global, &config);
    let elapsed = start.elapsed();

    if !cli.quiet {
        info!("Completed in {:.3}s", elapsed.as_secs_f64());
    }

    result
}

// ────────────────────────────────────────────────────────────────────────────
// Logging
// ────────────────────────────────────────────────────────────────────────────

fn init_logging(cli: &PenumbraCli) -> Result<()> {
    let level = if cli.verbose {
        "debug"
    } else if cli.quiet {
        "error"
    } else {
        "info"
    };

    env_logger::Builder::new()
        .parse_filters(level)
        .format_timestamp_millis()
        .format_module_path(false)
        .format_target(false)
        .try_init()
        .ok(); // ignore double-init

    Ok(())
}

// ────────────────────────────────────────────────────────────────────────────
// Banner
// ────────────────────────────────────────────────────────────────────────────

fn print_banner() {
    eprintln!(
        "╔══════════════════════════════════════════════╗\n\
         ║  Penumbra — FP Diagnosis & Repair Engine     ║\n\
         ║  Version {}                              ║\n\
         ╚══════════════════════════════════════════════╝",
        env!("CARGO_PKG_VERSION"),
    );
}

// ────────────────────────────────────────────────────────────────────────────
// Configuration
// ────────────────────────────────────────────────────────────────────────────

fn load_or_default_config(cli: &PenumbraCli) -> Result<penumbra_cli::config::PenumbraConfig> {
    if let Some(ref path) = cli.config_file {
        let s = path
            .to_str()
            .context("Config path is not valid UTF-8")?;
        let cfg = penumbra_cli::config::load_config(s)?;
        info!("Loaded config from {}", s);
        Ok(cfg)
    } else {
        Ok(penumbra_cli::config::PenumbraConfig::default())
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Argument validation
// ────────────────────────────────────────────────────────────────────────────

fn validate_global_args(cli: &PenumbraCli) -> Result<()> {
    if cli.verbose && cli.quiet {
        anyhow::bail!("--verbose and --quiet cannot be used together");
    }
    if let Some(ref cfg_path) = cli.config_file {
        if !cfg_path.exists() {
            anyhow::bail!("Config file not found: {}", cfg_path.display());
        }
    }
    Ok(())
}

// ────────────────────────────────────────────────────────────────────────────
// Command dispatch
// ────────────────────────────────────────────────────────────────────────────

fn dispatch(
    command: &Command,
    global: &penumbra_cli::args::GlobalOpts,
    config: &penumbra_cli::config::PenumbraConfig,
) -> Result<()> {
    match command {
        Command::Trace(args) => {
            info!("Running shadow execution trace…");
            commands::trace::execute_trace(args, global, config)
        }
        Command::Diagnose(args) => {
            info!("Diagnosing floating-point errors…");
            commands::diagnose::execute_diagnose(args, global, config)
        }
        Command::Repair(args) => {
            info!("Synthesizing repairs…");
            commands::repair::execute_repair(args, global, config)
        }
        Command::Certify(args) => {
            info!("Certifying repairs…");
            commands::certify::execute_certify(args, global, config)
        }
        Command::Analyze(args) => {
            info!("Running full analysis pipeline…");
            commands::analyze::execute_analyze(args, global, config)
        }
        Command::Optimize(args) => {
            info!("Optimizing expression…");
            commands::optimize::execute_optimize(args, global, config)
        }
        Command::Inspect(args) => {
            commands::inspect::execute_inspect(args, global)
        }
        Command::Report(args) => {
            info!("Generating report…");
            commands::report::execute_report(args, global, config)
        }
        Command::Benchmark(args) => {
            info!("Running benchmarks…");
            commands::benchmark::execute_benchmark(args, global, config)
        }
        Command::Config(args) => {
            commands::config_cmd::execute_config(args, global, config)
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Error formatting
// ────────────────────────────────────────────────────────────────────────────

fn format_error_chain(err: &anyhow::Error) -> String {
    let mut buf = String::new();
    buf.push_str(&format!("error: {}", err));
    for cause in err.chain().skip(1) {
        buf.push_str(&format!("\n  caused by: {}", cause));
    }
    buf
}

// ────────────────────────────────────────────────────────────────────────────
// Output-format helper used by multiple dispatch arms
// ────────────────────────────────────────────────────────────────────────────

/// Convenience: resolve the effective output format from global opts,
/// falling back to `Text` if nothing is specified.
#[allow(dead_code)]
fn effective_format(global: &penumbra_cli::args::GlobalOpts) -> OutputFormat {
    global.output_format.clone()
}
