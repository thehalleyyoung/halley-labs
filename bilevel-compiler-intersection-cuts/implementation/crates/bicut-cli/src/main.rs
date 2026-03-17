//! BiCut CLI — command-line interface for the BiCut bilevel optimization compiler.
//!
//! Provides subcommands for compiling, solving, analysing, benchmarking,
//! verifying, and generating bilevel optimisation problems.

mod commands;
mod config_file;
mod interactive;
mod output;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use log::{debug, error, info, LevelFilter};
use std::path::PathBuf;
use std::process;

use crate::config_file::BiCutConfig;

// ── Version info ───────────────────────────────────────────────────

/// Compile-time version string from Cargo.
const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Build profile (debug / release).
const BUILD_PROFILE: &str = if cfg!(debug_assertions) {
    "debug"
} else {
    "release"
};

/// Combined version banner used by `--version`.
fn version_long() -> String {
    format!(
        "{VERSION} ({BUILD_PROFILE} build, bicut-types {types_v})",
        types_v = VERSION,
    )
}

// ── CLI definition ─────────────────────────────────────────────────

/// BiCut — a bilevel optimization compiler with intersection cuts.
#[derive(Parser, Debug)]
#[command(
    name = "bicut",
    version = VERSION,
    about = "Bilevel optimization compiler with intersection cuts",
    long_about = "BiCut compiles bilevel optimisation problems into mixed-integer \
                  linear programs, solves them with branch-and-cut, and generates \
                  verifiable optimality certificates.",
    after_help = "Use `bicut <command> --help` for more information on each command."
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,

    /// Path to configuration file (default: ~/.bicut/config.toml).
    #[arg(long, short = 'c', global = true)]
    pub config: Option<PathBuf>,

    /// Set log verbosity level.
    #[arg(long, short, global = true, default_value = "warn")]
    pub log_level: LogLevel,

    /// Output format.
    #[arg(long, short = 'f', global = true, default_value = "human")]
    pub format: OutputFormat,

    /// Disable coloured output.
    #[arg(long, global = true, default_value_t = false)]
    pub no_color: bool,

    /// Write output to file instead of stdout.
    #[arg(long, short = 'o', global = true)]
    pub output: Option<PathBuf>,

    /// Quiet mode — suppress informational messages.
    #[arg(long, short = 'q', global = true, default_value_t = false)]
    pub quiet: bool,
}

/// Available log verbosity levels.
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum LogLevel {
    Off,
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

impl LogLevel {
    fn to_level_filter(self) -> LevelFilter {
        match self {
            LogLevel::Off => LevelFilter::Off,
            LogLevel::Error => LevelFilter::Error,
            LogLevel::Warn => LevelFilter::Warn,
            LogLevel::Info => LevelFilter::Info,
            LogLevel::Debug => LevelFilter::Debug,
            LogLevel::Trace => LevelFilter::Trace,
        }
    }
}

/// Available output formats.
#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
pub enum OutputFormat {
    /// Human-readable pretty-printed text.
    Human,
    /// Machine-readable JSON.
    Json,
    /// Compact single-line summaries.
    Compact,
}

/// Top-level subcommands.
#[derive(Subcommand, Debug)]
pub enum Command {
    /// Compile a bilevel problem to a single-level MILP reformulation.
    Compile(commands::CompileArgs),

    /// Compile and solve a bilevel problem.
    Solve(commands::SolveArgs),

    /// Perform structural analysis on a bilevel problem.
    Analyze(commands::AnalyzeArgs),

    /// Run benchmarks on a suite of problem instances.
    Benchmark(commands::BenchmarkArgs),

    /// Verify an optimality certificate for a bilevel solution.
    Verify(commands::VerifyArgs),

    /// Generate random bilevel problem instances.
    Generate(commands::GenerateArgs),

    /// Enter interactive exploration mode.
    Interactive(commands::InteractiveArgs),

    /// Print a default configuration file to stdout.
    InitConfig,

    /// Print detailed version information.
    VersionInfo,
}

// ── Logging ────────────────────────────────────────────────────────

/// Initialise `env_logger` respecting both `--log-level` and `RUST_LOG`.
fn init_logging(level: LogLevel) {
    let filter = level.to_level_filter();
    let mut builder = env_logger::Builder::new();
    if std::env::var("RUST_LOG").is_ok() {
        builder.parse_default_env();
    } else {
        builder.filter_level(filter);
    }
    builder.format_timestamp_millis();
    builder.init();
}

// ── Run context ────────────────────────────────────────────────────

/// Shared state passed to every subcommand.
pub struct RunContext {
    pub config: BiCutConfig,
    pub format: OutputFormat,
    pub no_color: bool,
    pub output_path: Option<PathBuf>,
    pub quiet: bool,
}

impl RunContext {
    fn from_cli(cli: &Cli) -> Result<Self> {
        let config = match &cli.config {
            Some(path) => {
                let text = std::fs::read_to_string(path)
                    .with_context(|| format!("reading config {}", path.display()))?;
                let mut cfg: BiCutConfig = toml::from_str(&text)
                    .with_context(|| format!("parsing config {}", path.display()))?;
                cfg.validate()?;
                cfg
            }
            None => {
                let default_path = config_file::default_config_path();
                if default_path.exists() {
                    debug!("Loading default config from {}", default_path.display());
                    let text = std::fs::read_to_string(&default_path).with_context(|| {
                        format!("reading default config {}", default_path.display())
                    })?;
                    let mut cfg: BiCutConfig = toml::from_str(&text).unwrap_or_default();
                    cfg.validate().ok();
                    cfg
                } else {
                    BiCutConfig::default()
                }
            }
        };
        Ok(RunContext {
            config,
            format: cli.format,
            no_color: cli.no_color,
            output_path: cli.output.clone(),
            quiet: cli.quiet,
        })
    }

    /// Write `content` to the output destination (stdout or file).
    pub fn write_output(&self, content: &str) -> Result<()> {
        match &self.output_path {
            Some(p) => {
                std::fs::write(p, content)
                    .with_context(|| format!("writing output to {}", p.display()))?;
                if !self.quiet {
                    eprintln!("Output written to {}", p.display());
                }
            }
            None => print!("{content}"),
        }
        Ok(())
    }
}

// ── Dispatch ───────────────────────────────────────────────────────

fn dispatch(cli: Cli) -> Result<()> {
    let ctx = RunContext::from_cli(&cli)?;

    match cli.command {
        Command::Compile(args) => commands::run_compile(args, &ctx),
        Command::Solve(args) => commands::run_solve(args, &ctx),
        Command::Analyze(args) => commands::run_analyze(args, &ctx),
        Command::Benchmark(args) => commands::run_benchmark(args, &ctx),
        Command::Verify(args) => commands::run_verify(args, &ctx),
        Command::Generate(args) => commands::run_generate(args, &ctx),
        Command::Interactive(args) => interactive::run_interactive(args, &ctx),
        Command::InitConfig => {
            let default_cfg = BiCutConfig::default();
            let toml_str =
                toml::to_string_pretty(&default_cfg).context("serialising default config")?;
            ctx.write_output(&toml_str)?;
            Ok(())
        }
        Command::VersionInfo => {
            print_version_info(&ctx);
            Ok(())
        }
    }
}

fn print_version_info(ctx: &RunContext) {
    let info = serde_json::json!({
        "version": VERSION,
        "profile": BUILD_PROFILE,
        "target": std::env::consts::ARCH,
        "os": std::env::consts::OS,
        "features": {
            "lp_solver": "simplex",
            "output_formats": ["human", "json", "compact"],
        }
    });
    match ctx.format {
        OutputFormat::Json => {
            let text = serde_json::to_string_pretty(&info).unwrap_or_default();
            let _ = ctx.write_output(&text);
        }
        _ => {
            let mut buf = String::new();
            buf.push_str(&format!("bicut {VERSION} ({BUILD_PROFILE} build)\n"));
            buf.push_str(&format!(
                "  target : {}/{}\n",
                std::env::consts::ARCH,
                std::env::consts::OS
            ));
            buf.push_str("  solver : simplex (built-in)\n");
            buf.push_str("  formats: human, json, compact\n");
            let _ = ctx.write_output(&buf);
        }
    }
}

// ── Entrypoint ─────────────────────────────────────────────────────

fn main() {
    let cli = Cli::parse();
    init_logging(cli.log_level);

    info!("bicut {} starting", VERSION);

    if let Err(e) = dispatch(cli) {
        error!("{:#}", e);
        eprintln!("Error: {e:#}");
        process::exit(1);
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn test_cli_parses_compile() {
        let args = vec!["bicut", "compile", "--input", "problem.json"];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok(), "compile subcommand should parse");
    }

    #[test]
    fn test_cli_parses_solve() {
        let args = vec!["bicut", "solve", "--input", "problem.json"];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok(), "solve subcommand should parse");
    }

    #[test]
    fn test_cli_parses_generate() {
        let args = vec![
            "bicut",
            "generate",
            "--num-upper",
            "3",
            "--num-lower",
            "4",
            "--num-constraints",
            "5",
        ];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok(), "generate subcommand should parse");
    }

    #[test]
    fn test_cli_global_options() {
        let args = vec![
            "bicut",
            "--log-level",
            "debug",
            "--format",
            "json",
            "--no-color",
            "version-info",
        ];
        let cli = Cli::try_parse_from(args).expect("should parse");
        assert!(matches!(cli.log_level, LogLevel::Debug));
        assert_eq!(cli.format, OutputFormat::Json);
        assert!(cli.no_color);
    }

    #[test]
    fn test_version_string_not_empty() {
        assert!(!VERSION.is_empty());
        assert!(!version_long().is_empty());
    }

    #[test]
    fn test_log_level_conversion() {
        assert_eq!(LogLevel::Off.to_level_filter(), LevelFilter::Off);
        assert_eq!(LogLevel::Error.to_level_filter(), LevelFilter::Error);
        assert_eq!(LogLevel::Trace.to_level_filter(), LevelFilter::Trace);
    }

    #[test]
    fn test_cli_verify_app() {
        Cli::command().debug_assert();
    }

    #[test]
    fn test_build_profile_is_debug_in_test() {
        assert_eq!(BUILD_PROFILE, "debug");
    }

    #[test]
    fn test_init_config_command_parses() {
        let args = vec!["bicut", "init-config"];
        let cli = Cli::try_parse_from(args);
        assert!(cli.is_ok());
    }

    #[test]
    fn test_output_format_values() {
        let human = OutputFormat::Human;
        let json = OutputFormat::Json;
        let compact = OutputFormat::Compact;
        assert_ne!(human, json);
        assert_ne!(json, compact);
    }
}
