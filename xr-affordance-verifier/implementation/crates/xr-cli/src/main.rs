//! XR Affordance Verifier CLI – main entry point.
//!
//! Provides the `xr-verify` binary with subcommands for linting,
//! verification, certification, inspection, reporting, and configuration.

mod commands;
mod config;
mod demo;
mod output;
mod pipeline;
mod scene_loader;
mod showcase;
mod webapp;

use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;
use tracing::Level;

/// XR Affordance Verifier – verify interactable element accessibility
/// across the human body-parameter population.
#[derive(Parser, Debug)]
#[command(
    name = "xr-verify",
    version,
    about = "XR Affordance Verifier: formal accessibility verification for XR scenes",
    long_about = "Verify that interactable elements in XR scenes are accessible \
                  across the target human body-parameter population (5th–95th percentile). \
                  Supports Tier 1 fast linting and Tier 2 formal SMT-based verification."
)]
pub struct Cli {
    /// Subcommand to execute.
    #[command(subcommand)]
    pub command: Command,

    /// Output format.
    #[arg(long, global = true, default_value = "text")]
    pub format: OutputFormat,

    /// Verbosity level (0=error, 1=warn, 2=info, 3=debug, 4=trace).
    #[arg(short, long, global = true, default_value_t = 2)]
    pub verbose: u8,

    /// Disable colored output.
    #[arg(long, global = true)]
    pub no_color: bool,

    /// Path to configuration file.
    #[arg(short, long, global = true)]
    pub config: Option<PathBuf>,
}

/// Output format for results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum OutputFormat {
    /// Human-readable text output.
    Text,
    /// Machine-readable JSON output.
    Json,
    /// Compact single-line output.
    Compact,
}

/// Available subcommands.
#[derive(Subcommand, Debug)]
pub enum Command {
    /// Run Tier 1 linting on a scene file.
    Lint {
        /// Path to the scene file.
        #[arg(value_name = "SCENE_FILE")]
        scene: PathBuf,

        /// Minimum element height threshold (meters).
        #[arg(long, default_value_t = 0.4)]
        min_height: f64,

        /// Maximum element height threshold (meters).
        #[arg(long, default_value_t = 2.2)]
        max_height: f64,

        /// Disable specific lint rules (comma-separated codes).
        #[arg(long, value_delimiter = ',')]
        disable: Vec<String>,

        /// Write report to file instead of stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Run full Tier 1 + Tier 2 verification pipeline.
    Verify {
        /// Path to the scene file.
        #[arg(value_name = "SCENE_FILE")]
        scene: PathBuf,

        /// Number of population samples.
        #[arg(short = 'n', long, default_value_t = 200)]
        samples: usize,

        /// SMT solver timeout in seconds.
        #[arg(long, default_value_t = 30.0)]
        smt_timeout: f64,

        /// Skip Tier 2 formal verification (sampling only).
        #[arg(long)]
        skip_tier2: bool,

        /// Stop on first failure.
        #[arg(long)]
        fail_fast: bool,

        /// Target coverage kappa.
        #[arg(long, default_value_t = 0.95)]
        target_kappa: f64,

        /// Write results to file.
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Generate a coverage certificate from verification results.
    Certify {
        /// Path to the scene file.
        #[arg(value_name = "SCENE_FILE")]
        scene: PathBuf,

        /// Number of population samples.
        #[arg(short = 'n', long, default_value_t = 500)]
        samples: usize,

        /// Target confidence level.
        #[arg(long, default_value_t = 0.95)]
        confidence: f64,

        /// Output certificate path (JSON).
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Also generate SVG diagram.
        #[arg(long)]
        svg: bool,
    },

    /// Inspect a scene file and display information.
    Inspect {
        /// Path to the scene file.
        #[arg(value_name = "SCENE_FILE")]
        scene: PathBuf,

        /// Show detailed element information.
        #[arg(long)]
        elements: bool,

        /// Show dependency graph.
        #[arg(long)]
        deps: bool,

        /// Show device configurations.
        #[arg(long)]
        devices: bool,

        /// Show all details.
        #[arg(long)]
        all: bool,
    },

    /// Generate a report from verification results.
    Report {
        /// Path to certificate JSON file.
        #[arg(value_name = "CERTIFICATE_FILE")]
        certificate: PathBuf,

        /// Report format (text, json, svg, html).
        #[arg(long, default_value = "text")]
        report_format: String,

        /// Output path for the report.
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Generate a self-contained interactive HTML dashboard.
    Webapp {
        /// Path to the scene file.
        #[arg(value_name = "SCENE_FILE")]
        scene: PathBuf,

        /// Optional existing certificate JSON file.
        #[arg(long)]
        certificate: Option<PathBuf>,

        /// Number of samples to use when generating a certificate on the fly.
        #[arg(short = 'n', long, default_value_t = 500)]
        samples: usize,

        /// Target confidence level when generating a certificate on the fly.
        #[arg(long, default_value_t = 0.95)]
        confidence: f64,

        /// Output HTML file path.
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Optional dashboard title override.
        #[arg(long)]
        title: Option<String>,
    },

    /// Generate a polished before/after showcase bundle for live demos.
    Showcase {
        /// Showcase scenario.
        #[arg(value_name = "SCENARIO", default_value = "accessibility-remediation")]
        scenario: ShowcaseScenario,

        /// Output directory for the generated bundle.
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Number of samples to use during certificate generation.
        #[arg(short = 'n', long, default_value_t = 500)]
        samples: usize,

        /// Target confidence level when generating certificates.
        #[arg(long, default_value_t = 0.95)]
        confidence: f64,

        /// Optional title override for the landing page.
        #[arg(long)]
        title: Option<String>,
    },

    /// Manage verifier configuration.
    Config {
        #[command(subcommand)]
        action: ConfigAction,
    },

    /// Generate a demo scene for testing.
    Demo {
        /// Demo scene type.
        #[arg(value_name = "SCENE_TYPE")]
        scene_type: DemoScene,

        /// Output path for the generated scene.
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
}

/// Config subcommand actions.
#[derive(Subcommand, Debug)]
pub enum ConfigAction {
    /// Show current effective configuration.
    Show,
    /// Generate a default configuration template.
    Init {
        /// Output path (default: ./xr-verify.json).
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// Validate a configuration file.
    Validate {
        /// Path to the configuration file.
        #[arg(value_name = "CONFIG_FILE")]
        path: PathBuf,
    },
    /// Show the path search order for configuration files.
    Path,
}

/// Available demo scene types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum DemoScene {
    /// Simple button panel with 5 buttons at different heights.
    ButtonPanel,
    /// VR control room with 20 interactable elements.
    ControlRoom,
    /// Manufacturing training scenario with multi-step interactions.
    Manufacturing,
    /// Accessibility showcase triggering various lint rules.
    Accessibility,
}

/// Available showcase bundle scenarios.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum ShowcaseScenario {
    /// Broken-vs-fixed accessibility remediation storyline.
    AccessibilityRemediation,
}

fn init_tracing(verbosity: u8) {
    let level = match verbosity {
        0 => Level::ERROR,
        1 => Level::WARN,
        2 => Level::INFO,
        3 => Level::DEBUG,
        _ => Level::TRACE,
    };

    tracing_subscriber::fmt()
        .with_max_level(level)
        .with_target(false)
        .with_thread_ids(false)
        .compact()
        .init();
}

fn main() {
    let cli = Cli::parse();

    init_tracing(cli.verbose);

    tracing::info!(
        "XR Affordance Verifier v{}",
        env!("CARGO_PKG_VERSION")
    );

    let cli_config = config::CliConfig::load_with_overrides(
        cli.config.as_deref(),
        cli.format,
        cli.verbose,
        cli.no_color,
    );

    let result = match cli.command {
        Command::Lint {
            scene,
            min_height,
            max_height,
            disable,
            output,
        } => commands::LintCommand {
            scene_path: scene,
            min_height,
            max_height,
            disabled_rules: disable,
            output_path: output,
            cli_config: &cli_config,
        }
        .execute(),

        Command::Verify {
            scene,
            samples,
            smt_timeout,
            skip_tier2,
            fail_fast,
            target_kappa,
            output,
        } => commands::VerifyCommand {
            scene_path: scene,
            num_samples: samples,
            smt_timeout,
            skip_tier2,
            fail_fast,
            target_kappa,
            output_path: output,
            cli_config: &cli_config,
        }
        .execute(),

        Command::Certify {
            scene,
            samples,
            confidence,
            output,
            svg,
        } => commands::CertifyCommand {
            scene_path: scene,
            num_samples: samples,
            confidence,
            output_path: output,
            generate_svg: svg,
            cli_config: &cli_config,
        }
        .execute(),

        Command::Inspect {
            scene,
            elements,
            deps,
            devices,
            all,
        } => commands::InspectCommand {
            scene_path: scene,
            show_elements: elements || all,
            show_deps: deps || all,
            show_devices: devices || all,
            cli_config: &cli_config,
        }
        .execute(),

        Command::Report {
            certificate,
            report_format,
            output,
        } => commands::ReportCommand {
            certificate_path: certificate,
            report_format,
            output_path: output,
            cli_config: &cli_config,
        }
        .execute(),

        Command::Webapp {
            scene,
            certificate,
            samples,
            confidence,
            output,
            title,
        } => commands::WebAppCommand {
            scene_path: scene,
            certificate_path: certificate,
            num_samples: samples,
            confidence,
            output_path: output,
            title,
            cli_config: &cli_config,
        }
        .execute(),

        Command::Showcase {
            scenario,
            output,
            samples,
            confidence,
            title,
        } => commands::ShowcaseCommand {
            scenario,
            output_dir: output,
            num_samples: samples,
            confidence,
            title,
            cli_config: &cli_config,
        }
        .execute(),

        Command::Config { action } => commands::ConfigCommand {
            action,
            cli_config: &cli_config,
        }
        .execute(),

        Command::Demo {
            scene_type,
            output,
        } => commands::DemoCommand {
            scene_type,
            output_path: output,
            cli_config: &cli_config,
        }
        .execute(),
    };

    match result {
        Ok(()) => {
            tracing::info!("Done.");
            std::process::exit(0);
        }
        Err(e) => {
            if cli_config.format == OutputFormat::Json {
                let err_json = serde_json::json!({
                    "error": true,
                    "message": e.to_string(),
                });
                eprintln!("{}", serde_json::to_string_pretty(&err_json).unwrap_or_default());
            } else {
                eprintln!("\x1b[1;31merror\x1b[0m: {e}");
            }
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn test_cli_parses() {
        Cli::command().debug_assert();
    }

    #[test]
    fn test_output_format_values() {
        assert_ne!(OutputFormat::Text, OutputFormat::Json);
        assert_ne!(OutputFormat::Json, OutputFormat::Compact);
    }

    #[test]
    fn test_demo_scene_values() {
        assert_ne!(DemoScene::ButtonPanel, DemoScene::ControlRoom);
        assert_ne!(DemoScene::Manufacturing, DemoScene::Accessibility);
    }
}
