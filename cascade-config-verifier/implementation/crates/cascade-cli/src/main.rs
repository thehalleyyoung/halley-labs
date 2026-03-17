//! Main entry point for the `cascade-verify` CLI binary.
//!
//! Responsibilities:
//! - Parse command-line arguments via [`clap`].
//! - Initialise logging / tracing.
//! - Dispatch to the appropriate command handler.
//! - Translate handler results into process exit codes.

use std::process;

use anyhow::Context;
use clap::Parser;
use tracing::Level;
use tracing_subscriber::fmt::format::FmtSpan;
use tracing_subscriber::EnvFilter;

use cascade_cli::commands::CascadeCli;
use cascade_cli::handlers;

// ── Exit-code constants ────────────────────────────────────────────────────

/// Success – no findings above the configured threshold.
const EXIT_OK: i32 = 0;
/// At least one finding exceeded the configured severity threshold.
const EXIT_FINDINGS: i32 = 1;
/// A runtime error occurred (bad input, I/O failure, etc.).
const EXIT_ERROR: i32 = 2;

// ── Logging / tracing bootstrap ────────────────────────────────────────────

/// Initialise the tracing subscriber.
///
/// The verbosity is controlled via:
/// 1. `--verbose` / `-v` flag on the CLI → forces DEBUG.
/// 2. `RUST_LOG` environment variable  → arbitrary filter expression.
/// 3. Fallback                         → WARN for libs, INFO for cascade crates.
fn init_tracing(verbose: bool) {
    let default_filter = if verbose {
        "cascade_cli=debug,cascade_types=debug,cascade_graph=debug,cascade_config=debug,info"
            .to_string()
    } else {
        std::env::var("RUST_LOG")
            .unwrap_or_else(|_| "cascade_cli=info,warn".to_string())
    };

    let filter = EnvFilter::try_new(&default_filter).unwrap_or_else(|_| {
        eprintln!(
            "warning: invalid RUST_LOG filter '{}', falling back to defaults",
            default_filter
        );
        EnvFilter::new("cascade_cli=info,warn")
    });

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_span_events(FmtSpan::CLOSE)
        .with_target(false)
        .with_thread_ids(false)
        .with_file(false)
        .with_line_number(false)
        .init();
}

// ── Pretty-print user-facing errors ────────────────────────────────────────

/// Walk the `anyhow` error chain and emit each cause on its own line to stderr.
fn print_error_chain(err: &anyhow::Error) {
    eprintln!("error: {err}");
    for cause in err.chain().skip(1) {
        eprintln!("  caused by: {cause}");
    }
}

/// When the user passes completely invalid arguments clap already exits, but
/// for *semantic* validation failures we produce a friendly message.
fn user_error(msg: &str) -> ! {
    eprintln!("error: {msg}");
    eprintln!("Hint: run `cascade-verify --help` for usage information.");
    process::exit(EXIT_ERROR);
}

// ── Dispatching ────────────────────────────────────────────────────────────

/// Dispatch the parsed CLI to the correct handler.
fn dispatch(cli: CascadeCli) -> anyhow::Result<i32> {
    match cli.command {
        cascade_cli::commands::Commands::Verify(args) => {
            handlers::handle_verify(&args).context("verify command failed")
        }
        cascade_cli::commands::Commands::Repair(args) => {
            handlers::handle_repair(&args).context("repair command failed")
        }
        cascade_cli::commands::Commands::Check(args) => {
            handlers::handle_check(&args).context("check command failed")
        }
        cascade_cli::commands::Commands::Analyze(args) => {
            handlers::handle_analyze(&args).context("analyze command failed")
        }
        cascade_cli::commands::Commands::Diff(args) => {
            handlers::handle_diff(&args).context("diff command failed")
        }
        cascade_cli::commands::Commands::Report(args) => {
            handlers::handle_report(&args).context("report command failed")
        }
        cascade_cli::commands::Commands::Benchmark(args) => {
            handlers::handle_benchmark(&args).context("benchmark command failed")
        }
    }
}

// ── Entry point ────────────────────────────────────────────────────────────

fn main() {
    // 1. Parse CLI arguments (clap will handle --help / --version / errors).
    let cli = CascadeCli::parse();

    // 2. Determine verbosity *before* consuming the struct so we can bootstrap
    //    tracing early.
    let verbose = cli.verbose();

    // 3. Initialise tracing / logging.
    init_tracing(verbose);

    tracing::debug!("cascade-verify starting");

    // 4. Validate semantic constraints that clap cannot express.
    if let Err(msg) = cli.validate() {
        user_error(&msg);
    }

    // 5. Dispatch.
    match dispatch(cli) {
        Ok(code) => {
            tracing::debug!("exiting with code {code}");
            process::exit(code);
        }
        Err(err) => {
            print_error_chain(&err);
            process::exit(EXIT_ERROR);
        }
    }
}

// ── Unit tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use cascade_cli::commands::CascadeCli;
    use clap::Parser;

    // Helper: parse a command line into CascadeCli (returns Err on bad args).
    fn parse(args: &[&str]) -> Result<CascadeCli, clap::Error> {
        CascadeCli::try_parse_from(args)
    }

    #[test]
    fn test_verify_subcommand_parses() {
        let cli = parse(&["cascade-verify", "verify", "path/to/config.yaml"]).unwrap();
        assert!(matches!(
            cli.command,
            cascade_cli::commands::Commands::Verify(_)
        ));
    }

    #[test]
    fn test_repair_subcommand_parses() {
        let cli = parse(&["cascade-verify", "repair", "path/a.yaml"]).unwrap();
        assert!(matches!(
            cli.command,
            cascade_cli::commands::Commands::Repair(_)
        ));
    }

    #[test]
    fn test_check_subcommand_parses() {
        let cli = parse(&["cascade-verify", "check", "some.yaml"]).unwrap();
        assert!(matches!(
            cli.command,
            cascade_cli::commands::Commands::Check(_)
        ));
    }

    #[test]
    fn test_analyze_subcommand_parses() {
        let cli = parse(&["cascade-verify", "analyze", "a.yaml"]).unwrap();
        assert!(matches!(
            cli.command,
            cascade_cli::commands::Commands::Analyze(_)
        ));
    }

    #[test]
    fn test_diff_subcommand_parses() {
        let cli = parse(&[
            "cascade-verify",
            "diff",
            "--base",
            "old.yaml",
            "--changed",
            "new.yaml",
        ])
        .unwrap();
        assert!(matches!(
            cli.command,
            cascade_cli::commands::Commands::Diff(_)
        ));
    }

    #[test]
    fn test_report_subcommand_parses() {
        let cli = parse(&["cascade-verify", "report", "--cache-dir", "/tmp/cache"]).unwrap();
        assert!(matches!(
            cli.command,
            cascade_cli::commands::Commands::Report(_)
        ));
    }

    #[test]
    fn test_benchmark_subcommand_parses() {
        let cli = parse(&["cascade-verify", "benchmark"]).unwrap();
        assert!(matches!(
            cli.command,
            cascade_cli::commands::Commands::Benchmark(_)
        ));
    }

    #[test]
    fn test_verbose_flag_propagated() {
        let cli = parse(&["cascade-verify", "-v", "check", "a.yaml"]).unwrap();
        assert!(cli.verbose());
    }

    #[test]
    fn test_missing_subcommand_errors() {
        let result = parse(&["cascade-verify"]);
        assert!(result.is_err());
    }

    #[test]
    fn test_exit_code_constants() {
        assert_eq!(EXIT_OK, 0);
        assert_eq!(EXIT_FINDINGS, 1);
        assert_eq!(EXIT_ERROR, 2);
    }

    #[test]
    fn test_unknown_subcommand_errors() {
        let result = parse(&["cascade-verify", "foobar"]);
        assert!(result.is_err());
    }

    #[test]
    fn test_verify_with_output_format() {
        let cli = parse(&[
            "cascade-verify",
            "verify",
            "--format",
            "json",
            "path.yaml",
        ])
        .unwrap();
        if let cascade_cli::commands::Commands::Verify(args) = cli.command {
            assert_eq!(args.output.format, "json");
        } else {
            panic!("expected Verify");
        }
    }

    #[test]
    fn test_validate_passes_for_good_args() {
        let cli = parse(&["cascade-verify", "check", "a.yaml"]).unwrap();
        assert!(cli.validate().is_ok());
    }

    #[test]
    fn test_verify_multiple_paths() {
        let cli =
            parse(&["cascade-verify", "verify", "a.yaml", "b.yaml", "c.yaml"]).unwrap();
        if let cascade_cli::commands::Commands::Verify(args) = cli.command {
            assert_eq!(args.paths.len(), 3);
        } else {
            panic!("expected Verify");
        }
    }
}
