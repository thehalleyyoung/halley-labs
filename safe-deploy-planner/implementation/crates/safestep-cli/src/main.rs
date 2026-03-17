//! SafeStep CLI — verified deployment planner.
//!
//! Entry point: parse CLI args, setup logging, dispatch to commands.

pub mod cli;
pub mod commands;
pub mod config_loader;
pub mod output;

use std::process::ExitCode;

use anyhow::{Context, Result};
use clap::Parser;
use tracing::info;
use tracing_subscriber::EnvFilter;

use cli::{Cli, ColorOption, Command};
use commands::plan::PlanCommand;
use commands::verify::VerifyCommand;
use commands::envelope::EnvelopeCommand;
use commands::analyze::AnalyzeCommand;
use commands::diff::DiffCommand;
use commands::export::ExportCommand;
use commands::validate::ValidateCommand;
use commands::benchmark::BenchmarkCommand;
use commands::CommandExecutor;
use config_loader::{discover_config_file, ConfigLoader};
use output::create_output_manager;

/// Run the CLI: parse args, configure logging, load config, dispatch command.
fn run() -> Result<()> {
    let cli = Cli::parse();

    // --- tracing setup ---
    let level = cli::resolve_log_level(cli.verbose, cli.quiet);
    let env_filter = EnvFilter::builder()
        .with_default_directive(level.into())
        .from_env_lossy();

    tracing_subscriber::fmt()
        .with_env_filter(env_filter)
        .with_target(false)
        .with_writer(std::io::stderr)
        .init();

    info!(version = env!("CARGO_PKG_VERSION"), "safestep starting");

    // --- config loading ---
    let mut loader = ConfigLoader::new();

    if let Some(ref path) = cli.config {
        loader
            .load_file(path)
            .with_context(|| format!("failed to load config: {}", path.display()))?;
    } else if let Some(discovered) = discover_config_file() {
        // Best-effort: don't fail if auto-discovered file can't be loaded.
        if let Err(e) = loader.load_file(&discovered) {
            tracing::warn!(
                path = %discovered.display(),
                error = %e,
                "ignoring unreadable auto-discovered config"
            );
        }
    }

    loader.load_env();

    // Forward relevant CLI flags as config overrides.
    if cli.verbose > 0 {
        loader.set_cli_override("output.format", cli.output_format.to_string());
    }

    let config = loader.build();

    // --- output manager ---
    let color_enabled = cli.color.should_use_color();
    let mut output = create_output_manager(cli.output_format, color_enabled, cli.output.clone());

    // --- command dispatch ---
    match cli.command {
        Command::Plan(args) => {
            PlanCommand::new(args, config).execute(&mut output)?;
        }
        Command::Verify(args) => {
            VerifyCommand::new(args).execute(&mut output)?;
        }
        Command::Envelope(args) => {
            EnvelopeCommand::new(args, config).execute(&mut output)?;
        }
        Command::Analyze(args) => {
            AnalyzeCommand::new(args, config).execute(&mut output)?;
        }
        Command::Diff(args) => {
            DiffCommand::new(args).execute(&mut output)?;
        }
        Command::Export(args) => {
            ExportCommand::new(args).execute(&mut output)?;
        }
        Command::Validate(args) => {
            ValidateCommand::new(args).execute(&mut output)?;
        }
        Command::Benchmark(args) => {
            BenchmarkCommand::new(args).execute(&mut output)?;
        }
    }

    output.flush().context("failed to flush output")?;
    Ok(())
}

/// Program entry point.
///
/// Converts `anyhow::Error` into a user-facing error message with cause chain
/// and returns an appropriate exit code.
fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            let use_color = std::env::var("NO_COLOR").is_err();
            let prefix = if use_color {
                "\x1b[1;31merror:\x1b[0m"
            } else {
                "error:"
            };
            eprintln!("{} {}", prefix, err);

            // Print the cause chain so the user can diagnose the root problem.
            let mut source = err.source();
            while let Some(cause) = source {
                let caused_prefix = if use_color {
                    "\x1b[1;31m  caused by:\x1b[0m"
                } else {
                    "  caused by:"
                };
                eprintln!("{} {}", caused_prefix, cause);
                source = std::error::Error::source(cause);
            }

            ExitCode::FAILURE
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    use crate::cli::{Cli, ColorOption, Command};
    use crate::config_loader::ConfigLoader;
    use crate::output::{create_output_manager, OutputManager};

    #[test]
    fn test_run_help_exits() {
        // `--help` causes clap to return an error (successful early exit).
        let result = Cli::try_parse_from(["safestep", "--help"]);
        assert!(result.is_err(), "clap --help should produce early exit error");
        let err = result.unwrap_err();
        assert_eq!(err.kind(), clap::error::ErrorKind::DisplayHelp);
    }

    #[test]
    fn test_output_manager_creation() {
        let mgr = create_output_manager(
            crate::cli::OutputFormat::Json,
            false,
            None,
        );
        assert_eq!(mgr.format(), crate::cli::OutputFormat::Json);
        assert!(!mgr.colors().enabled);

        let mgr_color = create_output_manager(
            crate::cli::OutputFormat::Text,
            true,
            None,
        );
        assert!(mgr_color.colors().enabled);
    }

    #[test]
    fn test_config_loader_integration() {
        let mut loader = ConfigLoader::new();
        loader.set_cli_override("planner.max_depth", "42".to_string());
        loader.set_cli_override("output.format", "json".to_string());
        let config = loader.build();
        assert_eq!(config.planner.max_depth, 42);
        assert_eq!(config.output.format, "json");
    }

    #[test]
    fn test_resolve_log_level() {
        use crate::cli::resolve_log_level;

        assert_eq!(resolve_log_level(0, false), tracing::Level::WARN);
        assert_eq!(resolve_log_level(1, false), tracing::Level::INFO);
        assert_eq!(resolve_log_level(2, false), tracing::Level::DEBUG);
        assert_eq!(resolve_log_level(3, false), tracing::Level::TRACE);
        assert_eq!(resolve_log_level(10, false), tracing::Level::TRACE);

        // quiet overrides verbose
        assert_eq!(resolve_log_level(0, true), tracing::Level::ERROR);
        assert_eq!(resolve_log_level(3, true), tracing::Level::ERROR);
    }

    #[test]
    fn test_color_never_no_color() {
        assert!(!ColorOption::Never.should_use_color());
    }

    #[test]
    fn test_all_subcommands_parse() {
        let cases: Vec<Vec<&str>> = vec![
            vec!["safestep", "plan", "-s", "0,0", "-t", "1,1"],
            vec!["safestep", "verify", "--plan-file", "/tmp/p.json"],
            vec!["safestep", "envelope", "--plan-file", "/tmp/p.json"],
            vec!["safestep", "analyze", "--schema-dir", "/tmp/schemas"],
            vec!["safestep", "diff", "--old", "/tmp/a", "--new", "/tmp/b"],
            vec![
                "safestep", "export", "--plan-file", "/tmp/p.json",
                "--format", "flux", "--output-dir", "/tmp/out",
            ],
            vec!["safestep", "validate", "--manifest-dir", "/tmp/m"],
            vec!["safestep", "benchmark", "--services", "5", "--versions", "3"],
        ];

        for argv in &cases {
            let result = Cli::try_parse_from(argv.iter());
            assert!(
                result.is_ok(),
                "failed to parse subcommand {:?}: {}",
                argv,
                result.unwrap_err()
            );
        }

        // Verify the parsed variants are correct.
        let plan = Cli::try_parse_from(&cases[0]).unwrap();
        assert!(matches!(plan.command, Command::Plan(_)));
        let verify = Cli::try_parse_from(&cases[1]).unwrap();
        assert!(matches!(verify.command, Command::Verify(_)));
        let envelope = Cli::try_parse_from(&cases[2]).unwrap();
        assert!(matches!(envelope.command, Command::Envelope(_)));
        let analyze = Cli::try_parse_from(&cases[3]).unwrap();
        assert!(matches!(analyze.command, Command::Analyze(_)));
        let diff = Cli::try_parse_from(&cases[4]).unwrap();
        assert!(matches!(diff.command, Command::Diff(_)));
        let export = Cli::try_parse_from(&cases[5]).unwrap();
        assert!(matches!(export.command, Command::Export(_)));
        let validate = Cli::try_parse_from(&cases[6]).unwrap();
        assert!(matches!(validate.command, Command::Validate(_)));
        let benchmark = Cli::try_parse_from(&cases[7]).unwrap();
        assert!(matches!(benchmark.command, Command::Benchmark(_)));
    }

    #[test]
    fn test_color_always() {
        assert!(ColorOption::Always.should_use_color());
    }
}
