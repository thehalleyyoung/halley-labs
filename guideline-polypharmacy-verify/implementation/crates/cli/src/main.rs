//! GuardPharma CLI — two-tier polypharmacy verification pipeline.
//!
//! Subcommands:
//!   verify    Full two-tier verification pipeline
//!   screen    Tier 1 screening only (fast abstract interpretation)
//!   analyze   Detailed conflict analysis
//!   benchmark Run benchmarks on polypharmacy scenarios
//!   convert   Convert between FHIR JSON and internal format

mod config;
mod input;
mod output;
mod pipeline;

use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use log::info;

use crate::config::{load_config, merge_with_cli, AppConfig};
use crate::input::{
    ActiveMedication, ClinicalCondition, GuidelineDocument, GuidelineMetadata, GuidelineRule,
    PatientProfile,
};
use crate::output::{write_output, JsonFormatter, OutputFormatter, TableFormatter, TextFormatter};
use crate::pipeline::PipelineOrchestrator;

use guardpharma_types::{CypEnzyme, DrugRoute, PatientInfo, Severity, Sex};

// ──────────────────────────── CLI Definition ─────────────────────────────

/// GuardPharma: formally verified polypharmacy safety checking.
#[derive(Parser, Debug)]
#[command(name = "guardpharma", version, about, long_about = None)]
pub struct CliArgs {
    #[command(subcommand)]
    pub command: Option<Commands>,

    /// Configuration file path
    #[arg(short, long, global = true)]
    pub config: Option<PathBuf>,

    /// Increase verbosity (-v info, -vv debug, -vvv trace)
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    pub verbose: u8,

    /// Suppress all non-error output
    #[arg(short, long, global = true)]
    pub quiet: bool,

    /// Disable colored output
    #[arg(long, global = true)]
    pub no_color: bool,

    /// Override log level (error, warn, info, debug, trace)
    #[arg(long, global = true)]
    pub log_level: Option<String>,

    /// Global timeout in seconds
    #[arg(long, global = true)]
    pub timeout: Option<u64>,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Full two-tier verification pipeline
    Verify {
        /// Input patient file (JSON or YAML)
        #[arg(short, long)]
        input: Option<PathBuf>,
        /// Output file path (default: stdout)
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Output format
        #[arg(short, long, default_value = "text")]
        format: OutputFormat,
        /// Run built-in demo scenario instead of reading a file
        #[arg(long)]
        demo: bool,
    },
    /// Tier 1 screening only (fast abstract interpretation)
    Screen {
        /// Input patient file (JSON or YAML)
        #[arg(short, long)]
        input: Option<PathBuf>,
        /// Output file path (default: stdout)
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Output format
        #[arg(short, long, default_value = "text")]
        format: OutputFormat,
        /// Run built-in demo scenario
        #[arg(long)]
        demo: bool,
    },
    /// Detailed conflict analysis
    Analyze {
        /// Input patient file (JSON or YAML)
        #[arg(short, long)]
        input: Option<PathBuf>,
        /// Output file path (default: stdout)
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Output format
        #[arg(short, long, default_value = "text")]
        format: OutputFormat,
        /// Show enzyme pathway details
        #[arg(long)]
        enzymes: bool,
        /// Show PK concentration traces
        #[arg(long)]
        pk_traces: bool,
        /// Run built-in demo scenario
        #[arg(long)]
        demo: bool,
    },
    /// Run benchmarks on polypharmacy scenarios
    Benchmark {
        /// Output file path (default: stdout)
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Output format
        #[arg(short, long, default_value = "table")]
        format: OutputFormat,
        /// Maximum number of drugs per scenario
        #[arg(long, default_value = "20")]
        max_drugs: usize,
    },
    /// Convert between FHIR JSON and internal format
    Convert {
        /// Input file (FHIR JSON or internal format)
        #[arg(short, long)]
        input: PathBuf,
        /// Output file path (default: stdout)
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// Target format
        #[arg(short, long, default_value = "json")]
        format: OutputFormat,
    },
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum OutputFormat {
    Json,
    Text,
    Table,
}

// ──────────────────────────── Entry Point ────────────────────────────────

fn main() -> Result<()> {
    let cli = CliArgs::parse();

    // Initialize logging
    let log_level = match cli.verbose {
        0 if cli.quiet => "error",
        0 => "warn",
        1 => "info",
        2 => "debug",
        _ => "trace",
    };
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(log_level))
        .format_timestamp_secs()
        .init();

    info!("GuardPharma CLI starting");

    // Load configuration
    let mut app_config = load_config(cli.config.as_deref()).unwrap_or_else(|e| {
        log::warn!("Using default config ({})", e);
        AppConfig::default()
    });
    merge_with_cli(&mut app_config, &cli);

    match cli.command {
        Some(Commands::Verify {
            input,
            output,
            format,
            demo,
        }) => cmd_verify(&app_config, input, output, format, demo),
        Some(Commands::Screen {
            input,
            output,
            format,
            demo,
        }) => cmd_screen(&app_config, input, output, format, demo),
        Some(Commands::Analyze {
            input,
            output,
            format,
            enzymes,
            pk_traces,
            demo,
        }) => cmd_analyze(&app_config, input, output, format, enzymes, pk_traces, demo),
        Some(Commands::Benchmark {
            output,
            format,
            max_drugs,
        }) => cmd_benchmark(&app_config, output, format, max_drugs),
        Some(Commands::Convert {
            input,
            output,
            format,
        }) => cmd_convert(input, output, format),
        None => {
            // No subcommand: run the demo verification
            println!("GuardPharma v{}", env!("CARGO_PKG_VERSION"));
            println!("Use --help for usage information, or try `guardpharma verify --demo`\n");
            cmd_verify(&app_config, None, None, OutputFormat::Text, true)
        }
    }
}

// ──────────────────────── Subcommand Handlers ───────────────────────────

fn cmd_verify(
    config: &AppConfig,
    input: Option<PathBuf>,
    output: Option<PathBuf>,
    format: OutputFormat,
    demo: bool,
) -> Result<()> {
    let (patient, guidelines) = load_or_demo(input, demo)?;

    let orchestrator = PipelineOrchestrator::new(config);
    let result = orchestrator.run_full_pipeline(&guidelines, &patient)?;

    let content = match format {
        OutputFormat::Text => TextFormatter::new(config.general.color).format_verification(&result),
        OutputFormat::Json => JsonFormatter::new().format_verification(&result),
        OutputFormat::Table => {
            TableFormatter::new(config.general.color).format_verification(&result)
        }
    };

    write_output(&content, output.as_deref())
}

fn cmd_screen(
    config: &AppConfig,
    input: Option<PathBuf>,
    output: Option<PathBuf>,
    format: OutputFormat,
    demo: bool,
) -> Result<()> {
    let (patient, guidelines) = load_or_demo(input, demo)?;

    let orchestrator = PipelineOrchestrator::new(config);
    let result = orchestrator.run_screening_only(&guidelines, &patient)?;

    let content = match format {
        OutputFormat::Text => TextFormatter::new(config.general.color).format_screening(&result),
        OutputFormat::Json => JsonFormatter::new().format_screening(&result),
        OutputFormat::Table => TableFormatter::new(config.general.color).format_screening(&result),
    };

    write_output(&content, output.as_deref())
}

fn cmd_analyze(
    config: &AppConfig,
    input: Option<PathBuf>,
    output: Option<PathBuf>,
    format: OutputFormat,
    enzymes: bool,
    pk_traces: bool,
    demo: bool,
) -> Result<()> {
    let (patient, guidelines) = load_or_demo(input, demo)?;

    let orchestrator = PipelineOrchestrator::new(config);
    let result = orchestrator.run_analysis_only(&guidelines, &patient, enzymes, pk_traces)?;

    let content = match format {
        OutputFormat::Text => TextFormatter::new(config.general.color).format_conflicts(&result),
        OutputFormat::Json => JsonFormatter::new().format_conflicts(&result),
        OutputFormat::Table => TableFormatter::new(config.general.color).format_conflicts(&result),
    };

    write_output(&content, output.as_deref())
}

fn cmd_benchmark(
    config: &AppConfig,
    output: Option<PathBuf>,
    format: OutputFormat,
    max_drugs: usize,
) -> Result<()> {
    use std::time::Instant;

    let drug_counts = [2, 5, 10, 15, 20].iter().copied().filter(|&n| n <= max_drugs);
    let mut rows: Vec<(usize, usize, f64, f64, String)> = Vec::new();

    for n_drugs in drug_counts {
        let patient = build_scaled_patient(n_drugs);
        let guidelines = build_demo_guidelines();
        let n_pairs = n_drugs * (n_drugs - 1) / 2;

        let orchestrator = PipelineOrchestrator::new(config);

        let start = Instant::now();
        let result = orchestrator.run_full_pipeline(&guidelines, &patient)?;
        let elapsed = start.elapsed();

        let verdict = match &result.verdict {
            pipeline::VerificationVerdict::Safe => "Safe".to_string(),
            pipeline::VerificationVerdict::ConflictsFound { count } => {
                format!("{} conflicts", count)
            }
            pipeline::VerificationVerdict::Inconclusive { reason } => {
                format!("Inconclusive: {}", reason)
            }
            pipeline::VerificationVerdict::Error { message } => format!("Error: {}", message),
        };

        rows.push((n_drugs, n_pairs, elapsed.as_secs_f64() * 1000.0, elapsed.as_secs_f64() * 1000.0 / n_pairs as f64, verdict));
    }

    let content = match format {
        OutputFormat::Table | OutputFormat::Text => {
            let mut out = String::new();
            out.push_str("GuardPharma Polypharmacy Scaling Benchmark\n");
            out.push_str(&"═".repeat(72));
            out.push('\n');
            out.push_str(&format!(
                "{:>6} {:>8} {:>12} {:>12}  {}\n",
                "Drugs", "Pairs", "Total (ms)", "Per-pair", "Verdict"
            ));
            out.push_str(&"─".repeat(72));
            out.push('\n');
            for (n, pairs, total, per_pair, verdict) in &rows {
                out.push_str(&format!(
                    "{:>6} {:>8} {:>12.2} {:>12.4}  {}\n",
                    n, pairs, total, per_pair, verdict
                ));
            }
            out.push_str(&"═".repeat(72));
            out.push('\n');
            out
        }
        OutputFormat::Json => serde_json::to_string_pretty(
            &rows
                .iter()
                .map(|(n, pairs, total, per_pair, verdict)| {
                    serde_json::json!({
                        "drugs": n,
                        "pairs": pairs,
                        "total_ms": total,
                        "per_pair_ms": per_pair,
                        "verdict": verdict,
                    })
                })
                .collect::<Vec<_>>(),
        )?,
    };

    write_output(&content, output.as_deref())
}

fn cmd_convert(input: PathBuf, output: Option<PathBuf>, format: OutputFormat) -> Result<()> {
    let content = std::fs::read_to_string(&input)
        .map_err(|e| anyhow::anyhow!("Cannot read {}: {}", input.display(), e))?;

    // Try parsing as PatientProfile (internal format)
    let patient: PatientProfile = if input.extension().map(|e| e == "yaml" || e == "yml").unwrap_or(false) {
        serde_yaml::from_str(&content)?
    } else {
        serde_json::from_str(&content)?
    };

    let result = match format {
        OutputFormat::Json => serde_json::to_string_pretty(&patient)?,
        OutputFormat::Text => format!("{:#?}", patient),
        OutputFormat::Table => format!("{:#?}", patient),
    };

    write_output(&result, output.as_deref())
}

// ──────────────────────── Scenario Builders ──────────────────────────────

fn load_or_demo(
    input: Option<PathBuf>,
    demo: bool,
) -> Result<(PatientProfile, Vec<GuidelineDocument>)> {
    if let Some(path) = input {
        let content = std::fs::read_to_string(&path)?;
        let patient: PatientProfile =
            if path.extension().map(|e| e == "yaml" || e == "yml").unwrap_or(false) {
                serde_yaml::from_str(&content)?
            } else {
                serde_json::from_str(&content)?
            };
        let guidelines = build_demo_guidelines();
        Ok((patient, guidelines))
    } else if demo {
        Ok((build_demo_patient(), build_demo_guidelines()))
    } else {
        // Default to demo when no input provided
        info!("No input file specified, running demo scenario");
        Ok((build_demo_patient(), build_demo_guidelines()))
    }
}

/// Build a demo patient: 72-year-old with diabetes, hypertension, AFib.
pub fn build_demo_patient() -> PatientProfile {
    let info = PatientInfo {
        age_years: 72.0,
        weight_kg: 82.0,
        height_cm: 175.0,
        sex: Sex::Male,
        serum_creatinine: 1.4,
        ..Default::default()
    };

    PatientProfile::new(info)
        .with_conditions(vec![
            ClinicalCondition::new("E11", "Type 2 Diabetes Mellitus"),
            ClinicalCondition::new("I10", "Essential Hypertension"),
            ClinicalCondition::new("I48", "Atrial Fibrillation"),
        ])
        .with_medications(vec![
            ActiveMedication::new("Metformin", 1000.0)
                .with_frequency(12.0)
                .with_class("Biguanide")
                .with_indication("Type 2 Diabetes"),
            ActiveMedication::new("Lisinopril", 20.0)
                .with_frequency(24.0)
                .with_class("ACE Inhibitor")
                .with_indication("Hypertension"),
            ActiveMedication::new("Amlodipine", 5.0)
                .with_frequency(24.0)
                .with_class("Calcium Channel Blocker")
                .with_indication("Hypertension"),
            ActiveMedication::new("Warfarin", 5.0)
                .with_frequency(24.0)
                .with_class("Anticoagulant")
                .with_indication("Atrial Fibrillation"),
            ActiveMedication::new("Metoprolol", 50.0)
                .with_frequency(12.0)
                .with_class("Beta Blocker")
                .with_indication("Rate control / Hypertension"),
        ])
        .with_egfr(52.0)
}

/// Build demo guidelines with clinically relevant rules.
pub fn build_demo_guidelines() -> Vec<GuidelineDocument> {
    let mut ada = GuidelineDocument::new("ADA Diabetes Standards of Care");
    ada.version = "2024".to_string();
    ada.source = "American Diabetes Association".to_string();
    ada.description = "Standards of Medical Care in Diabetes".to_string();
    ada.rules.push({
        let mut r = GuidelineRule::new(
            "ADA-6.2",
            "Metformin dose adjustment required when eGFR < 45",
            Severity::Major,
        );
        r.affected_drugs = vec!["metformin".to_string()];
        r.evidence_level = "A".to_string();
        r.dose_adjustment = Some("Reduce to 500mg BID if eGFR 30-45; discontinue if eGFR < 30".to_string());
        r
    });

    let mut acc = GuidelineDocument::new("ACC/AHA Hypertension Guideline");
    acc.version = "2023".to_string();
    acc.source = "American College of Cardiology / American Heart Association".to_string();
    acc.rules.push({
        let mut r = GuidelineRule::new(
            "ACC-BP-7.1",
            "ACE inhibitor + calcium channel blocker preferred combination",
            Severity::Minor,
        );
        r.affected_drugs = vec!["lisinopril".to_string(), "amlodipine".to_string()];
        r.evidence_level = "A".to_string();
        r
    });

    let mut chest = GuidelineDocument::new("CHEST Antithrombotic Guideline");
    chest.version = "2022".to_string();
    chest.source = "American College of Chest Physicians".to_string();
    chest.rules.push({
        let mut r = GuidelineRule::new(
            "CHEST-AT-3.2",
            "Warfarin + metoprolol: CYP2C9 competitive metabolism increases warfarin exposure",
            Severity::Moderate,
        );
        r.affected_drugs = vec!["warfarin".to_string(), "metoprolol".to_string()];
        r.affected_conditions = vec!["I48".to_string()];
        r.monitoring_required = true;
        r.evidence_level = "B".to_string();
        r
    });
    chest.rules.push({
        let mut r = GuidelineRule::new(
            "CHEST-AT-5.1",
            "Anticoagulant + beta-blocker: monitor for enhanced bleeding risk in elderly",
            Severity::Moderate,
        );
        r.affected_drugs = vec!["warfarin".to_string(), "metoprolol".to_string()];
        r.monitoring_required = true;
        r.evidence_level = "C".to_string();
        r
    });

    vec![ada, acc, chest]
}

/// Build a scaled patient with N drugs for benchmarking.
fn build_scaled_patient(n_drugs: usize) -> PatientProfile {
    let drug_pool = vec![
        ("Metformin", 1000.0, 12.0, "Biguanide"),
        ("Lisinopril", 20.0, 24.0, "ACE Inhibitor"),
        ("Amlodipine", 5.0, 24.0, "Calcium Channel Blocker"),
        ("Warfarin", 5.0, 24.0, "Anticoagulant"),
        ("Metoprolol", 50.0, 12.0, "Beta Blocker"),
        ("Atorvastatin", 40.0, 24.0, "Statin"),
        ("Omeprazole", 20.0, 24.0, "Proton Pump Inhibitor"),
        ("Aspirin", 81.0, 24.0, "Antiplatelet"),
        ("Furosemide", 40.0, 12.0, "Loop Diuretic"),
        ("Gabapentin", 300.0, 8.0, "Anticonvulsant"),
        ("Sertraline", 50.0, 24.0, "SSRI"),
        ("Allopurinol", 300.0, 24.0, "Xanthine Oxidase Inhibitor"),
        ("Levothyroxine", 0.1, 24.0, "Thyroid Hormone"),
        ("Tamsulosin", 0.4, 24.0, "Alpha Blocker"),
        ("Clopidogrel", 75.0, 24.0, "Antiplatelet"),
        ("Amiodarone", 200.0, 24.0, "Antiarrhythmic"),
        ("Digoxin", 0.125, 24.0, "Cardiac Glycoside"),
        ("Prednisone", 10.0, 24.0, "Corticosteroid"),
        ("Fluconazole", 200.0, 24.0, "Antifungal"),
        ("Carbamazepine", 200.0, 12.0, "Anticonvulsant"),
    ];

    let meds: Vec<ActiveMedication> = drug_pool
        .iter()
        .take(n_drugs)
        .map(|(name, dose, freq, class)| {
            ActiveMedication::new(name, *dose)
                .with_frequency(*freq)
                .with_class(class)
        })
        .collect();

    let info = PatientInfo {
        age_years: 72.0,
        weight_kg: 82.0,
        height_cm: 175.0,
        sex: Sex::Male,
        serum_creatinine: 1.4,
        ..Default::default()
    };

    PatientProfile::new(info)
        .with_conditions(vec![
            ClinicalCondition::new("E11", "Type 2 Diabetes"),
            ClinicalCondition::new("I10", "Hypertension"),
            ClinicalCondition::new("I48", "Atrial Fibrillation"),
        ])
        .with_medications(meds)
        .with_egfr(52.0)
}
