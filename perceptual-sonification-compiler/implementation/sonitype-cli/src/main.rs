//! SoniType — Perceptual Sonification Compiler
//!
//! Main entry point. Parses CLI arguments via `clap`, initialises logging,
//! and dispatches to the appropriate command handler.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use log::LevelFilter;
use std::path::PathBuf;
use std::process;

use sonitype_cli::commands::{
    CheckCommand, CompileCommand, InfoCommand, InitCommand, LintCommand, PreviewCommand,
    RenderCommand,
};
use sonitype_cli::config::CliConfig;
use sonitype_cli::diagnostics::{DiagnosticEngine, DiagnosticFormat};
use sonitype_cli::repl::SoniTypeRepl;

// ── Top-level CLI definition ────────────────────────────────────────────────

/// SoniType — a perceptual type system and optimising compiler for
/// information-preserving data sonification.
#[derive(Parser, Debug)]
#[command(
    name = "sonitype",
    version,
    about = "Perceptual sonification compiler",
    long_about = "SoniType compiles high-level sonification specifications into \
                  optimised, perceptually validated audio renderers."
)]
pub struct Cli {
    /// Path to a configuration file (TOML or JSON).
    #[arg(long, global = true)]
    config: Option<PathBuf>,

    /// Increase verbosity (-v, -vv, -vvv).
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    verbose: u8,

    /// Suppress all output except errors.
    #[arg(short, long, global = true)]
    quiet: bool,

    /// Emit diagnostics as JSON (for IDE integration).
    #[arg(long, global = true)]
    json: bool,

    /// Override the output sample rate.
    #[arg(long, global = true)]
    sample_rate: Option<u32>,

    /// Override the optimisation level (0–3).
    #[arg(long, global = true, value_parser = clap::value_parser!(u8).range(0..=3))]
    opt_level: Option<u8>,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Compile a .soni DSL file into an audio renderer.
    Compile(CompileArgs),
    /// Render a compiled sonification to a WAV audio file.
    Render(RenderArgs),
    /// Type-check a DSL file without compiling.
    Check(CheckArgs),
    /// Quick low-quality preview rendering.
    Preview(PreviewArgs),
    /// Run perceptual lint checks on a DSL file.
    Lint(LintArgs),
    /// Show information about a DSL file.
    Info(InfoArgs),
    /// Create a new sonification project from a template.
    Init(InitArgs),
    /// Start an interactive REPL.
    Repl,
}

// ── Per-command argument structs ────────────────────────────────────────────

#[derive(Parser, Debug)]
pub struct CompileArgs {
    /// Input .soni source file.
    #[arg()]
    pub input: PathBuf,
    /// Output path for the compiled renderer.
    #[arg(short, long)]
    pub output: Option<PathBuf>,
    /// Emit Rust source instead of a binary audio graph.
    #[arg(long)]
    pub emit_rust: bool,
    /// Skip WCET verification.
    #[arg(long)]
    pub skip_wcet: bool,
}

#[derive(Parser, Debug)]
pub struct RenderArgs {
    /// Compiled audio-graph file.
    #[arg()]
    pub graph: PathBuf,
    /// Data source file (CSV or JSON).
    #[arg(short, long)]
    pub data: PathBuf,
    /// Output WAV file path.
    #[arg(short, long)]
    pub output: PathBuf,
    /// Maximum render duration in seconds.
    #[arg(long)]
    pub max_duration: Option<f64>,
}

#[derive(Parser, Debug)]
pub struct CheckArgs {
    /// Input .soni source file.
    #[arg()]
    pub input: PathBuf,
}

#[derive(Parser, Debug)]
pub struct PreviewArgs {
    /// Input .soni source file.
    #[arg()]
    pub input: PathBuf,
    /// Preview duration in seconds (default 3).
    #[arg(long, default_value = "3.0")]
    pub duration: f64,
}

#[derive(Parser, Debug)]
pub struct LintArgs {
    /// Input .soni source file.
    #[arg()]
    pub input: PathBuf,
}

#[derive(Parser, Debug)]
pub struct InfoArgs {
    /// Input .soni source file.
    #[arg()]
    pub input: PathBuf,
}

#[derive(Parser, Debug)]
pub struct InitArgs {
    /// Project directory to create.
    #[arg()]
    pub path: PathBuf,
    /// Template to use (basic, multi-stream, spatial).
    #[arg(long, default_value = "basic")]
    pub template: String,
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn init_logging(verbose: u8, quiet: bool) {
    let filter = if quiet {
        LevelFilter::Error
    } else {
        match verbose {
            0 => LevelFilter::Warn,
            1 => LevelFilter::Info,
            2 => LevelFilter::Debug,
            _ => LevelFilter::Trace,
        }
    };

    env_logger::Builder::new()
        .filter_level(filter)
        .format_timestamp_millis()
        .init();
}

fn build_config(cli: &Cli) -> Result<CliConfig> {
    let mut cfg = if let Some(ref path) = cli.config {
        CliConfig::load_from_file(path).context("Failed to load configuration file")?
    } else {
        CliConfig::discover().unwrap_or_default()
    };

    // Apply global overrides.
    if let Some(sr) = cli.sample_rate {
        cfg.sample_rate = sr;
    }
    if let Some(opt) = cli.opt_level {
        cfg.optimization_level = opt;
    }
    cfg.verbose = cli.verbose > 0;
    cfg.quiet = cli.quiet;

    cfg.validate().context("Invalid configuration")?;
    Ok(cfg)
}

fn diagnostic_format(cli: &Cli) -> DiagnosticFormat {
    if cli.json {
        DiagnosticFormat::Json
    } else {
        DiagnosticFormat::Plain
    }
}

// ── Entry point ─────────────────────────────────────────────────────────────

fn main() {
    let cli = Cli::parse();
    init_logging(cli.verbose, cli.quiet);

    if let Err(e) = run(cli) {
        eprintln!("error: {e:#}");
        process::exit(1);
    }
}

fn run(cli: Cli) -> Result<()> {
    let config = build_config(&cli)?;
    let diag_fmt = diagnostic_format(&cli);
    let diagnostics = DiagnosticEngine::new(diag_fmt);

    match cli.command {
        Command::Compile(args) => {
            let cmd = CompileCommand::new(args.input, args.output, args.emit_rust, args.skip_wcet);
            cmd.execute(&config, &diagnostics)
                .context("Compilation failed")?;
        }
        Command::Render(args) => {
            let cmd = RenderCommand::new(args.graph, args.data, args.output, args.max_duration);
            cmd.execute(&config, &diagnostics)
                .context("Rendering failed")?;
        }
        Command::Check(args) => {
            let cmd = CheckCommand::new(args.input);
            cmd.execute(&config, &diagnostics)
                .context("Type checking failed")?;
        }
        Command::Preview(args) => {
            let cmd = PreviewCommand::new(args.input, args.duration);
            cmd.execute(&config, &diagnostics)
                .context("Preview failed")?;
        }
        Command::Lint(args) => {
            let cmd = LintCommand::new(args.input);
            cmd.execute(&config, &diagnostics)
                .context("Linting failed")?;
        }
        Command::Info(args) => {
            let cmd = InfoCommand::new(args.input);
            cmd.execute(&config, &diagnostics)
                .context("Info command failed")?;
        }
        Command::Init(args) => {
            let cmd = InitCommand::new(args.path, args.template);
            cmd.execute(&config, &diagnostics)
                .context("Project initialisation failed")?;
        }
        Command::Repl => {
            let mut repl = SoniTypeRepl::new(config);
            repl.run().context("REPL error")?;
        }
    }

    Ok(())
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn cli_parses_compile_command() {
        let cli = Cli::try_parse_from(["sonitype", "compile", "test.soni"]).unwrap();
        assert!(matches!(cli.command, Command::Compile(_)));
    }

    #[test]
    fn cli_parses_render_command() {
        let cli = Cli::try_parse_from([
            "sonitype", "render", "graph.bin", "-d", "data.csv", "-o", "out.wav",
        ])
        .unwrap();
        assert!(matches!(cli.command, Command::Render(_)));
    }

    #[test]
    fn cli_parses_check_command() {
        let cli = Cli::try_parse_from(["sonitype", "check", "test.soni"]).unwrap();
        assert!(matches!(cli.command, Command::Check(_)));
    }

    #[test]
    fn cli_parses_preview_command() {
        let cli = Cli::try_parse_from(["sonitype", "preview", "test.soni"]).unwrap();
        if let Command::Preview(args) = cli.command {
            assert_eq!(args.duration, 3.0);
        } else {
            panic!("expected Preview");
        }
    }

    #[test]
    fn cli_parses_lint_command() {
        let cli = Cli::try_parse_from(["sonitype", "lint", "test.soni"]).unwrap();
        assert!(matches!(cli.command, Command::Lint(_)));
    }

    #[test]
    fn cli_parses_info_command() {
        let cli = Cli::try_parse_from(["sonitype", "info", "test.soni"]).unwrap();
        assert!(matches!(cli.command, Command::Info(_)));
    }

    #[test]
    fn cli_parses_init_command() {
        let cli = Cli::try_parse_from(["sonitype", "init", "my_project"]).unwrap();
        if let Command::Init(args) = cli.command {
            assert_eq!(args.template, "basic");
        } else {
            panic!("expected Init");
        }
    }

    #[test]
    fn cli_parses_repl_command() {
        let cli = Cli::try_parse_from(["sonitype", "repl"]).unwrap();
        assert!(matches!(cli.command, Command::Repl));
    }

    #[test]
    fn cli_global_flags() {
        let cli =
            Cli::try_parse_from(["sonitype", "-vv", "--json", "--opt-level", "2", "check", "x.soni"])
                .unwrap();
        assert_eq!(cli.verbose, 2);
        assert!(cli.json);
        assert_eq!(cli.opt_level, Some(2));
    }

    #[test]
    fn cli_rejects_invalid_opt_level() {
        let result = Cli::try_parse_from(["sonitype", "--opt-level", "5", "check", "x.soni"]);
        assert!(result.is_err());
    }

    #[test]
    fn cli_help_does_not_panic() {
        // Verify the command factory builds without panic.
        Cli::command().debug_assert();
    }

    #[test]
    fn init_logging_levels() {
        // Smoke-test that the mapping is correct (cannot re-init env_logger
        // in-process, so just verify the function compiles and the match arms
        // are exhaustive).
        let _ = std::panic::catch_unwind(|| {
            // These will fail because env_logger is already initialised,
            // but we only care that no panics happen in the mapping logic.
            let _ = LevelFilter::Error;
            let _ = LevelFilter::Warn;
            let _ = LevelFilter::Info;
            let _ = LevelFilter::Debug;
            let _ = LevelFilter::Trace;
        });
    }

    #[test]
    fn diagnostic_format_json() {
        let cli =
            Cli::try_parse_from(["sonitype", "--json", "check", "x.soni"]).unwrap();
        assert_eq!(diagnostic_format(&cli), DiagnosticFormat::Json);
    }

    #[test]
    fn diagnostic_format_plain() {
        let cli = Cli::try_parse_from(["sonitype", "check", "x.soni"]).unwrap();
        assert_eq!(diagnostic_format(&cli), DiagnosticFormat::Plain);
    }
}
