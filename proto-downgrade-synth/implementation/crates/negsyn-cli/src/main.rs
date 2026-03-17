//! NegSynth CLI — Protocol Downgrade Attack Synthesis Tool.
//!
//! Entry point for the `negsyn` binary.  Parses CLI arguments with clap,
//! initialises logging via env_logger, and dispatches to the appropriate
//! subcommand handler.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::process;

mod commands;
mod config;
mod logging;
mod output;

use commands::{
    analyze::AnalyzeCommand,
    benchmark::BenchmarkCommand,
    diff::DiffCommand,
    inspect::InspectCommand,
    replay::ReplayCommand,
    verify::VerifyCommand,
};
use config::CliConfig;
use logging::LogConfig;
use output::OutputFormat;

// ---------------------------------------------------------------------------
// CLI definition
// ---------------------------------------------------------------------------

/// NegSynth — Protocol Downgrade Attack Synthesis & Verification.
///
/// Synthesises, verifies, and replays protocol downgrade attacks against
/// TLS and SSH library implementations using symbolic execution with
/// Dolev–Yao adversary modelling.
#[derive(Parser, Debug)]
#[command(
    name = "negsyn",
    version,
    about = "Protocol downgrade attack synthesis tool",
    long_about = "NegSynth synthesises, verifies, and replays protocol downgrade attacks\n\
                  against TLS and SSH implementations using symbolic execution with\n\
                  Dolev–Yao adversary modelling.",
    propagate_version = true,
    arg_required_else_help = true
)]
pub struct Cli {
    /// Verbosity level (-v, -vv, -vvv).
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    verbose: u8,

    /// Suppress all output except errors.
    #[arg(short, long, global = true)]
    quiet: bool,

    /// Configuration file path.
    #[arg(short, long, global = true, value_name = "FILE")]
    config: Option<PathBuf>,

    /// Log output file (receives all messages regardless of verbosity).
    #[arg(long, global = true, value_name = "FILE")]
    log_file: Option<PathBuf>,

    /// Output format override (applies to all subcommands).
    #[arg(long, global = true, value_enum)]
    format: Option<OutputFormat>,

    /// Disable coloured output.
    #[arg(long, global = true)]
    no_color: bool,

    #[command(subcommand)]
    command: Command,
}

/// Available subcommands.
#[derive(Subcommand, Debug)]
pub enum Command {
    /// Run full downgrade attack synthesis analysis.
    Analyze(AnalyzeCommand),
    /// Verify an analysis certificate.
    Verify(VerifyCommand),
    /// Run differential analysis across multiple libraries.
    Diff(DiffCommand),
    /// Replay an attack trace.
    Replay(ReplayCommand),
    /// Run benchmark suite.
    Benchmark(BenchmarkCommand),
    /// Inspect a state machine.
    Inspect(InspectCommand),
    /// Generate a default configuration file.
    Init {
        /// Output path for the configuration file.
        #[arg(short, long, default_value = "negsyn.toml")]
        output: PathBuf,
    },
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() {
    let cli = Cli::parse();

    // Build logging config from CLI flags.
    let log_config = LogConfig {
        verbosity: if cli.quiet { 0 } else { cli.verbose + 1 },
        log_file: cli.log_file.clone(),
        no_color: cli.no_color,
    };

    if let Err(e) = logging::init(&log_config) {
        eprintln!("negsyn: failed to initialise logging: {e}");
        process::exit(2);
    }

    log::debug!(
        "NegSynth CLI v{} (pid {})",
        env!("CARGO_PKG_VERSION"),
        std::process::id()
    );
    log::trace!("CLI args: {cli:?}");

    let exit_code = match run(cli) {
        Ok(code) => code,
        Err(e) => {
            log::error!("{e:#}");
            for cause in e.chain().skip(1) {
                log::error!("  caused by: {cause}");
            }
            1
        }
    };

    process::exit(exit_code);
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

fn run(cli: Cli) -> Result<i32> {
    let cfg = load_config(&cli)?;

    let format = cli.format.unwrap_or(cfg.default_format);
    let no_color = cli.no_color || cfg.no_color;

    match cli.command {
        Command::Analyze(cmd) => {
            cmd.execute(&cfg, format, no_color)
                .context("analysis failed")?;
            Ok(0)
        }
        Command::Verify(cmd) => {
            let valid = cmd
                .execute(&cfg, format, no_color)
                .context("verification failed")?;
            Ok(if valid { 0 } else { 1 })
        }
        Command::Diff(cmd) => {
            cmd.execute(&cfg, format, no_color)
                .context("differential analysis failed")?;
            Ok(0)
        }
        Command::Replay(cmd) => {
            let success = cmd
                .execute(&cfg, format, no_color)
                .context("replay failed")?;
            Ok(if success { 0 } else { 1 })
        }
        Command::Benchmark(cmd) => {
            cmd.execute(&cfg, format, no_color)
                .context("benchmark failed")?;
            Ok(0)
        }
        Command::Inspect(cmd) => {
            cmd.execute(&cfg, format, no_color)
                .context("inspection failed")?;
            Ok(0)
        }
        Command::Init { output } => {
            config::generate_default(&output)
                .context("failed to write default configuration")?;
            eprintln!("Wrote default configuration to {}", output.display());
            Ok(0)
        }
    }
}

// ---------------------------------------------------------------------------
// Config loading
// ---------------------------------------------------------------------------

/// Resolve and load configuration in priority order:
/// 1. Explicit `--config` flag.
/// 2. `negsyn.toml` in the current directory.
/// 3. `~/.config/negsyn/negsyn.toml`.
/// 4. Compiled-in defaults.
fn load_config(cli: &Cli) -> Result<CliConfig> {
    let mut cfg = if let Some(ref path) = cli.config {
        CliConfig::from_file(path)
            .with_context(|| format!("loading config from {}", path.display()))?
    } else {
        let candidates = [local_config_path(), user_config_path()];
        let mut loaded = None;
        for p in &candidates {
            if p.exists() {
                log::debug!("Loading config from {}", p.display());
                loaded = Some(CliConfig::from_file(p)?);
                break;
            }
        }
        loaded.unwrap_or_default()
    };

    cfg.apply_env_overrides();
    cfg.validate().context("invalid configuration")?;

    Ok(cfg)
}

/// `negsyn.toml` in the current working directory.
fn local_config_path() -> PathBuf {
    PathBuf::from("negsyn.toml")
}

/// `~/.config/negsyn/negsyn.toml`.
fn user_config_path() -> PathBuf {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_else(|_| ".".into());
    PathBuf::from(home)
        .join(".config")
        .join("negsyn")
        .join("negsyn.toml")
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
        Cli::command().debug_assert();
    }

    #[test]
    fn cli_parses_analyze() {
        let args = vec![
            "negsyn",
            "analyze",
            "input.bc",
            "--library",
            "openssl",
            "--protocol",
            "tls",
        ];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok(), "{:?}", cli.err());
    }

    #[test]
    fn cli_parses_verify() {
        let args = vec!["negsyn", "verify", "cert.json"];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn cli_parses_diff() {
        let args = vec![
            "negsyn", "diff", "a.bc", "b.bc", "--names", "openssl,mbedtls",
        ];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok(), "{:?}", cli.err());
    }

    #[test]
    fn cli_parses_replay() {
        let args = vec!["negsyn", "replay", "trace.json"];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn cli_parses_benchmark() {
        let args = vec!["negsyn", "benchmark", "all"];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn cli_parses_inspect() {
        let args = vec!["negsyn", "inspect", "sm.json"];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn cli_parses_init() {
        let args = vec!["negsyn", "init"];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn cli_global_flags() {
        let args = vec![
            "negsyn",
            "-vvv",
            "--no-color",
            "--format",
            "json",
            "benchmark",
        ];
        let cli = Cli::try_parse_from(args).unwrap();
        assert_eq!(cli.verbose, 3);
        assert!(cli.no_color);
        assert_eq!(cli.format, Some(OutputFormat::Json));
    }

    #[test]
    fn cli_quiet_flag() {
        let args = vec!["negsyn", "-q", "benchmark"];
        let cli = Cli::try_parse_from(args).unwrap();
        assert!(cli.quiet);
    }

    #[test]
    fn default_config_loads_and_validates() {
        let cfg = CliConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn local_config_path_is_relative() {
        let p = local_config_path();
        assert!(!p.is_absolute());
        assert!(p.to_string_lossy().contains("negsyn.toml"));
    }

    #[test]
    fn user_config_path_contains_negsyn() {
        let p = user_config_path();
        assert!(p.to_string_lossy().contains("negsyn"));
    }

    #[test]
    fn cli_help_text_available() {
        let mut cmd = Cli::command();
        let help = cmd.render_help();
        let text = help.to_string();
        assert!(text.contains("NegSynth") || text.contains("negsyn") || text.contains("downgrade"));
    }

    #[test]
    fn cli_version_info() {
        let version = env!("CARGO_PKG_VERSION");
        assert!(!version.is_empty());
    }
}
