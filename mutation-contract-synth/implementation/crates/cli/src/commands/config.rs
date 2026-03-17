//! `mutspec config` — configuration management command.
//!
//! Subcommands: init, show, validate, path, env.

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Args, Subcommand};
use log::{info, warn};

use crate::config::{
    annotated_config_template, default_config_template, CliConfig, ConfigDiagnostic, ConfigSeverity,
};
use crate::output::{write_json, AlignedTable, CliOutputFormat, Colour};

// ---------------------------------------------------------------------------
// CLI arguments
// ---------------------------------------------------------------------------

#[derive(Debug, Args)]
pub struct ConfigArgs {
    #[command(subcommand)]
    pub action: ConfigAction,

    /// Disable colour output.
    #[arg(long)]
    pub no_color: bool,
}

#[derive(Debug, Subcommand)]
pub enum ConfigAction {
    /// Create a new mutspec.toml configuration file.
    Init {
        /// Output path (default: mutspec.toml in current directory).
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Create an annotated template with comments.
        #[arg(long)]
        annotated: bool,

        /// Overwrite an existing file.
        #[arg(long)]
        force: bool,
    },

    /// Display the effective configuration.
    Show {
        /// Output format.
        #[arg(short = 'f', long, value_enum, default_value = "text")]
        format: CliOutputFormat,

        /// Show only a specific section (mutation, synthesis, smt, analysis, coverage, output).
        #[arg(long)]
        section: Option<String>,
    },

    /// Validate the configuration and report issues.
    Validate,

    /// Print the path to the loaded configuration file.
    Path,

    /// Show active environment variable overrides.
    Env,
}

// ---------------------------------------------------------------------------
// Run
// ---------------------------------------------------------------------------

pub fn run(args: &ConfigArgs, cfg: &CliConfig) -> Result<()> {
    let colour = Colour::new(args.no_color);

    match &args.action {
        ConfigAction::Init {
            output,
            annotated,
            force,
        } => run_init(output.as_deref(), *annotated, *force, &colour),
        ConfigAction::Show { format, section } => {
            run_show(cfg, *format, section.as_deref(), &colour)
        }
        ConfigAction::Validate => run_validate(cfg, &colour),
        ConfigAction::Path => run_path(cfg, &colour),
        ConfigAction::Env => run_env(cfg, &colour),
    }
}

// ---------------------------------------------------------------------------
// Subcommand implementations
// ---------------------------------------------------------------------------

fn run_init(
    output: Option<&std::path::Path>,
    annotated: bool,
    force: bool,
    colour: &Colour,
) -> Result<()> {
    let path = output
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("mutspec.toml"));

    if path.exists() && !force {
        anyhow::bail!(
            "Config file already exists at {}. Use --force to overwrite.",
            path.display()
        );
    }

    let template = if annotated {
        annotated_config_template()
    } else {
        default_config_template()
    };

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&path, &template)
        .with_context(|| format!("Failed to write {}", path.display()))?;

    println!(
        "{} Configuration file created at {}",
        colour.green("✓"),
        colour.bold(&path.display().to_string())
    );
    if annotated {
        println!("  (annotated template with comments)");
    }

    Ok(())
}

fn run_show(
    cfg: &CliConfig,
    format: CliOutputFormat,
    section: Option<&str>,
    colour: &Colour,
) -> Result<()> {
    match format {
        CliOutputFormat::Json => {
            if let Some(section) = section {
                let json = section_to_json(cfg, section)?;
                println!("{json}");
            } else {
                let json = serde_json::to_string_pretty(&cfg.inner)
                    .context("Failed to serialize config")?;
                println!("{json}");
            }
        }
        CliOutputFormat::Text | CliOutputFormat::Markdown | CliOutputFormat::Sarif => {
            if let Some(section) = section {
                print_section(cfg, section, colour)?;
            } else {
                print_full_config(cfg, colour)?;
            }
        }
    }
    Ok(())
}

fn print_full_config(cfg: &CliConfig, colour: &Colour) -> Result<()> {
    println!("{}", colour.bold("MutSpec Configuration"));
    if let Some(ref path) = cfg.config_path {
        println!("  Source: {}", path.display());
    } else {
        println!("  Source: {} (defaults)", colour.dim("none"));
    }
    println!();

    // Mutation section
    println!("{}", colour.bold("[mutation]"));
    let ops_str = cfg.mutation().operators.join(", ");
    println!("  operators              = [{}]", ops_str);
    println!(
        "  max_mutants_per_site   = {}",
        cfg.mutation().max_mutants_per_site
    );
    println!(
        "  generation_timeout     = {}s",
        cfg.mutation().generation_timeout_secs
    );
    println!();

    // Synthesis section
    println!("{}", colour.bold("[synthesis]"));
    println!("  enabled_tiers    = {:?}", cfg.synthesis().enabled_tiers);
    println!(
        "  tier_timeout     = {}s",
        cfg.synthesis().tier_timeout_secs
    );
    println!(
        "  minimise         = {}",
        cfg.synthesis().minimise_contracts
    );
    println!();

    // SMT section
    println!("{}", colour.bold("[smt]"));
    println!("  solver_path  = {}", cfg.smt().solver_path.display());
    println!("  timeout_secs = {}", cfg.smt().timeout_secs);
    println!("  logic        = {}", cfg.smt().logic);
    println!();

    // Analysis section
    println!("{}", colour.bold("[analysis]"));
    println!("  max_expr_depth = {}", cfg.analysis().max_expr_depth);
    println!("  use_ssa        = {}", cfg.analysis().use_ssa);
    println!();

    // Coverage section
    println!("{}", colour.bold("[coverage]"));
    println!("  subsumption            = {}", cfg.coverage().subsumption);
    println!(
        "  adequate_score_thresh  = {}",
        cfg.coverage().adequate_score_threshold
    );
    println!();

    // Output section
    println!("{}", colour.bold("[output]"));
    println!("  format    = {:?}", cfg.inner.output.format);
    println!("  output_dir = {}", cfg.inner.output.output_dir.display());
    println!("  verbosity = {}", cfg.inner.output.verbosity);
    println!();

    // TOML representation
    println!("{}", colour.dim("# TOML representation:"));
    let toml = cfg.to_toml_string()?;
    for line in toml.lines() {
        println!("{}", colour.dim(line));
    }

    Ok(())
}

fn print_section(cfg: &CliConfig, section: &str, colour: &Colour) -> Result<()> {
    match section {
        "mutation" => {
            println!("{}", colour.bold("[mutation]"));
            let ops = cfg.mutation().operators.join(", ");
            println!("  operators              = [{}]", ops);
            println!(
                "  max_mutants_per_site   = {}",
                cfg.mutation().max_mutants_per_site
            );
            println!(
                "  generation_timeout     = {}s",
                cfg.mutation().generation_timeout_secs
            );
        }
        "synthesis" => {
            println!("{}", colour.bold("[synthesis]"));
            println!("  enabled_tiers = {:?}", cfg.synthesis().enabled_tiers);
            println!("  tier_timeout  = {}s", cfg.synthesis().tier_timeout_secs);
            println!("  minimise      = {}", cfg.synthesis().minimise_contracts);
        }
        "smt" => {
            println!("{}", colour.bold("[smt]"));
            println!("  solver_path  = {}", cfg.smt().solver_path.display());
            println!("  timeout_secs = {}", cfg.smt().timeout_secs);
            println!("  logic        = {}", cfg.smt().logic);
        }
        "analysis" => {
            println!("{}", colour.bold("[analysis]"));
            println!("  max_expr_depth = {}", cfg.analysis().max_expr_depth);
            println!("  use_ssa        = {}", cfg.analysis().use_ssa);
        }
        "coverage" => {
            println!("{}", colour.bold("[coverage]"));
            println!("  subsumption           = {}", cfg.coverage().subsumption);
            println!(
                "  adequate_score_thresh = {}",
                cfg.coverage().adequate_score_threshold
            );
        }
        "output" => {
            println!("{}", colour.bold("[output]"));
            println!("  format     = {:?}", cfg.inner.output.format);
            println!("  output_dir = {}", cfg.inner.output.output_dir.display());
            println!("  verbosity  = {}", cfg.inner.output.verbosity);
        }
        other => {
            anyhow::bail!(
                "Unknown config section: '{}'. Valid sections: mutation, synthesis, smt, analysis, coverage, output",
                other
            );
        }
    }
    Ok(())
}

fn section_to_json(cfg: &CliConfig, section: &str) -> Result<String> {
    let json = match section {
        "mutation" => serde_json::to_string_pretty(&cfg.inner.mutation)?,
        "synthesis" => serde_json::to_string_pretty(&cfg.inner.synthesis)?,
        "smt" => serde_json::to_string_pretty(&cfg.inner.smt)?,
        "analysis" => serde_json::to_string_pretty(&cfg.inner.analysis)?,
        "coverage" => serde_json::to_string_pretty(&cfg.inner.coverage)?,
        "output" => serde_json::to_string_pretty(&cfg.inner.output)?,
        other => {
            anyhow::bail!(
                "Unknown config section: '{}'. Valid: mutation, synthesis, smt, analysis, coverage, output",
                other
            );
        }
    };
    Ok(json)
}

fn run_validate(cfg: &CliConfig, colour: &Colour) -> Result<()> {
    let diags = cfg.validate();

    if diags.is_empty() {
        println!("{} Configuration is valid", colour.green("✓"));
        return Ok(());
    }

    let mut tbl = AlignedTable::new(vec!["Severity".into(), "Field".into(), "Message".into()]);

    let mut has_errors = false;
    for d in &diags {
        let sev = match d.severity {
            ConfigSeverity::Error => {
                has_errors = true;
                colour.red("ERROR")
            }
            ConfigSeverity::Warning => colour.yellow("WARN"),
            ConfigSeverity::Info => colour.cyan("INFO"),
        };
        tbl.add_row(vec![sev, d.field.clone(), d.message.clone()]);
    }

    println!("{tbl}");

    if has_errors {
        println!(
            "\n{} Configuration has errors that must be fixed",
            colour.red("✗")
        );
    } else {
        println!(
            "\n{} Configuration is valid with {} warning(s)",
            colour.yellow("⚠"),
            diags.len()
        );
    }

    Ok(())
}

fn run_path(cfg: &CliConfig, colour: &Colour) -> Result<()> {
    match &cfg.config_path {
        Some(path) => {
            println!("{}", path.display());
        }
        None => {
            eprintln!(
                "{} No configuration file loaded (using defaults)",
                colour.dim("–")
            );
            // Print search paths
            eprintln!("Searched for:");
            let cwd = std::env::current_dir().unwrap_or_default();
            for name in &[
                "mutspec.toml",
                ".mutspec.toml",
                ".mutspec/config.toml",
                "mutspec/config.toml",
            ] {
                let path = cwd.join(name);
                eprintln!("  {}", path.display());
            }
        }
    }
    Ok(())
}

fn run_env(cfg: &CliConfig, colour: &Colour) -> Result<()> {
    let overrides = cfg.active_env_overrides();

    if overrides.is_empty() {
        println!(
            "{} No environment variable overrides active",
            colour.dim("–")
        );
        println!();
        println!("Available environment variables:");
        print_env_help(colour);
        return Ok(());
    }

    println!("{}", colour.bold("Active Environment Overrides:"));
    println!();

    let mut tbl = AlignedTable::new(vec!["Variable".into(), "Value".into()]);
    for (key, val) in &overrides {
        tbl.add_row(vec![colour.cyan(key), val.clone()]);
    }
    println!("{tbl}");
    println!();
    println!("All available variables:");
    print_env_help(colour);

    Ok(())
}

fn print_env_help(colour: &Colour) {
    let vars = [
        ("MUTSPEC_TIMEOUT_SECS", "SMT solver timeout in seconds"),
        ("MUTSPEC_MAX_MUTANTS", "Maximum mutants per function"),
        ("MUTSPEC_OPERATORS", "Comma-separated operator mnemonics"),
        (
            "MUTSPEC_OUTPUT_FORMAT",
            "Output format (text, json, sarif, jml)",
        ),
        ("MUTSPEC_SMT_SOLVER", "Path to SMT solver binary"),
        ("MUTSPEC_SYNTHESIS_TIER", "Maximum synthesis tier (1-3)"),
        ("MUTSPEC_PARALLEL", "Number of parallel threads"),
    ];

    let mut tbl = AlignedTable::new(vec!["Variable".into(), "Description".into()]);
    for (var, desc) in &vars {
        tbl.add_row(vec![colour.cyan(var), desc.to_string()]);
    }
    println!("{tbl}");
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_validate_default() {
        let cfg = CliConfig::default_config();
        let colour = Colour::new(true);
        // Should not panic
        run_validate(&cfg, &colour).unwrap();
    }

    #[test]
    fn test_section_to_json_mutation() {
        let cfg = CliConfig::default_config();
        let json = section_to_json(&cfg, "mutation").unwrap();
        assert!(json.contains("operators"));
    }

    #[test]
    fn test_section_to_json_invalid() {
        let cfg = CliConfig::default_config();
        assert!(section_to_json(&cfg, "nonexistent").is_err());
    }

    #[test]
    fn test_default_config_template() {
        let t = default_config_template();
        assert!(!t.is_empty());
    }

    #[test]
    fn test_annotated_template_has_comments() {
        let t = annotated_config_template();
        assert!(t.contains('#'));
        assert!(t.contains("[mutation]"));
    }
}
