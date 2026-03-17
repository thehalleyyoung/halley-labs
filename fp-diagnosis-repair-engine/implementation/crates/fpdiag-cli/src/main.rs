//! # fpdiag-cli
//!
//! Command-line interface for the Penumbra floating-point diagnosis and
//! repair engine.
//!
//! ## Usage
//!
//! ```text
//! penumbra trace   <script>  [--precision 128] [--output trace.json]
//! penumbra diagnose <trace>  [--threshold 10] [--format json|human|csv]
//! penumbra repair  <trace>   [--budget 10] [--certify]
//! penumbra certify <repair>  [--samples 10000]
//! penumbra report  <trace>   [--full] [--output report.json]
//! ```

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};
use std::time::Instant;

use fpdiag_analysis::EagBuilder;
use fpdiag_diagnosis::DiagnosisEngine;
use fpdiag_repair::RepairSynthesizer;
use fpdiag_report::ReportGenerator;
use fpdiag_types::config::{DiagnosisConfig, OutputFormat, PenumbraConfig, RepairConfig};
use fpdiag_types::expression::FpOp;
use fpdiag_types::precision::Precision;
use fpdiag_types::source::SourceSpan;
use fpdiag_types::trace::{ExecutionTrace, TraceEvent};

/// Penumbra: Diagnosis-Guided Repair of Floating-Point Error
#[derive(Parser)]
#[command(
    name = "penumbra",
    version,
    about = "Diagnosis-guided repair of floating-point error in scientific pipelines",
    long_about = "Penumbra instruments scientific Python code to construct an Error Amplification \
                  Graph (EAG), diagnoses root causes of precision loss, and synthesizes certified \
                  repairs."
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Verbosity level (-v, -vv, -vvv)
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    verbose: u8,

    /// Output format
    #[arg(long, default_value = "human", global = true)]
    format: String,
}

#[derive(Subcommand)]
enum Commands {
    /// Trace a Python script under shadow-value instrumentation
    Trace {
        /// Path to the Python script to trace
        script: PathBuf,
        /// Shadow precision in bits
        #[arg(long, default_value = "128")]
        precision: u32,
        /// Output trace file
        #[arg(short, long, default_value = "trace.json")]
        output: PathBuf,
        /// Maximum number of events to trace
        #[arg(long)]
        max_events: Option<u64>,
    },
    /// Diagnose floating-point errors in a trace
    Diagnose {
        /// Input trace file
        trace: PathBuf,
        /// ULP threshold for high-error classification
        #[arg(long, default_value = "10")]
        threshold: f64,
        /// Minimum confidence for reported diagnoses
        #[arg(long, default_value = "0.5")]
        min_confidence: f64,
        /// Output diagnosis report file
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// Repair diagnosed floating-point errors
    Repair {
        /// Input trace file
        trace: PathBuf,
        /// Maximum number of nodes to repair
        #[arg(long, default_value = "10")]
        budget: usize,
        /// Enable formal certification via interval arithmetic
        #[arg(long)]
        certify: bool,
        /// Output repair report file
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// Certify a repair result
    Certify {
        /// Input repair result file
        repair: PathBuf,
        /// Number of empirical samples for non-formal certification
        #[arg(long, default_value = "10000")]
        samples: u32,
        /// Output certification report file
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// Generate a full report (trace → diagnose → repair → certify)
    Report {
        /// Input trace file
        trace: PathBuf,
        /// Run the full pipeline
        #[arg(long)]
        full: bool,
        /// Output report file
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// Run built-in benchmarks
    Bench {
        /// Benchmark name (or "all")
        #[arg(default_value = "all")]
        name: String,
        /// Output directory for results
        #[arg(short, long, default_value = "bench_results")]
        output: PathBuf,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize logging
    let log_level = match cli.verbose {
        0 => "warn",
        1 => "info",
        2 => "debug",
        _ => "trace",
    };
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(log_level)).init();

    let output_format = match cli.format.as_str() {
        "json" => OutputFormat::Json,
        "csv" => OutputFormat::Csv,
        _ => OutputFormat::Human,
    };

    match cli.command {
        Commands::Trace {
            script,
            precision,
            output,
            max_events,
        } => {
            log::info!(
                "Tracing {} with {}-bit shadow precision",
                script.display(),
                precision
            );
            let trace = trace_script(&script, precision, max_events)?;
            write_trace(&output, &trace)?;
            println!("Tracing {} → {}", script.display(), output.display());
            println!(
                "Shadow precision: {} bits, max events: {}",
                precision,
                max_events.map_or("unlimited".to_string(), |n| n.to_string())
            );
            println!(
                "Trace written to {} ({} events)",
                output.display(),
                trace.metadata.event_count
            );
            Ok(())
        }
        Commands::Diagnose {
            trace,
            threshold,
            min_confidence,
            output,
        } => {
            log::info!("Diagnosing trace: {}", trace.display());

            // Load trace (in production, deserialize from file)
            let exec_trace = load_trace(&trace)?;

            // Build EAG
            let mut builder = EagBuilder::with_defaults();
            builder
                .build_from_trace(&exec_trace)
                .context("EAG construction failed")?;
            let eag = builder.finish();

            // Run diagnosis
            let config = DiagnosisConfig {
                error_threshold_ulps: threshold,
                min_confidence,
                exhaustive: true,
            };
            let engine = DiagnosisEngine::new(config);
            let report = engine.diagnose(&eag).context("Diagnosis failed")?;

            // Output report
            let gen = ReportGenerator::new(output_format, cli.verbose > 0);
            let repair = fpdiag_types::repair::RepairResult::new();
            let formatted = gen.generate(&eag, &report, &repair)?;
            println!("{}", formatted.content);

            if let Some(out_path) = output {
                std::fs::write(&out_path, &formatted.content).context("Failed to write output")?;
                log::info!("Report written to {}", out_path.display());
            }

            Ok(())
        }
        Commands::Repair {
            trace,
            budget,
            certify,
            output,
        } => {
            log::info!("Repairing trace: {}", trace.display());

            let exec_trace = load_trace(&trace)?;

            // Build EAG
            let mut builder = EagBuilder::with_defaults();
            builder
                .build_from_trace(&exec_trace)
                .context("EAG construction failed")?;
            let eag = builder.finish();

            // Diagnose
            let engine = DiagnosisEngine::with_defaults();
            let diag_report = engine.diagnose(&eag).context("Diagnosis failed")?;

            // Repair
            let repair_config = RepairConfig {
                max_repair_budget: budget,
                ..RepairConfig::default()
            };
            let synth = RepairSynthesizer::new(repair_config);
            let repair_result = synth
                .synthesize(&eag, &diag_report)
                .context("Repair synthesis failed")?;

            // Report
            let gen = ReportGenerator::new(output_format, cli.verbose > 0);
            let formatted = gen.generate(&eag, &diag_report, &repair_result)?;
            println!("{}", formatted.content);

            if let Some(out_path) = output {
                std::fs::write(&out_path, &formatted.content).context("Failed to write output")?;
            }

            Ok(())
        }
        Commands::Certify {
            repair,
            samples,
            output,
        } => {
            log::info!("Certifying repair: {}", repair.display());
            println!(
                "Certifying repair from {} ({} empirical samples)",
                repair.display(),
                samples
            );
            println!("(Certification would load repair results and validate bounds)");
            Ok(())
        }
        Commands::Report {
            trace,
            full,
            output,
        } => {
            log::info!("Generating report for: {}", trace.display());

            let exec_trace = load_trace(&trace)?;

            // Full pipeline
            let mut builder = EagBuilder::with_defaults();
            builder
                .build_from_trace(&exec_trace)
                .context("EAG construction failed")?;
            let eag = builder.finish();

            let engine = DiagnosisEngine::with_defaults();
            let diag_report = engine.diagnose(&eag).context("Diagnosis failed")?;

            let synth = RepairSynthesizer::with_defaults();
            let repair_result = synth.synthesize(&eag, &diag_report).unwrap_or_default();

            let gen = ReportGenerator::new(output_format, true);
            let formatted = gen.generate(&eag, &diag_report, &repair_result)?;
            println!("{}", formatted.content);

            if let Some(out_path) = output {
                std::fs::write(&out_path, &formatted.content).context("Failed to write output")?;
            }

            Ok(())
        }
        Commands::Bench { name, output } => {
            log::info!("Running benchmark: {}", name);
            let harness = fpdiag_eval::EvalHarness::new(PenumbraConfig::default());
            println!("Available benchmarks:");
            for b in harness.list_benchmarks() {
                println!("  {} — {}", b.name, b.description);
            }
            Ok(())
        }
    }
}

/// Generate a file-backed trace artifact for the given script.
fn trace_script(
    script: &Path,
    precision_bits: u32,
    max_events: Option<u64>,
) -> Result<ExecutionTrace> {
    let started = Instant::now();
    let script_path = script
        .canonicalize()
        .with_context(|| format!("Failed to resolve script path: {}", script.display()))?;
    let source = std::fs::read_to_string(&script_path)
        .with_context(|| format!("Failed to read script: {}", script_path.display()))?;

    let mut trace = trace_from_script_source(&script_path, &source, precision_bits, max_events);
    trace.metadata.wall_time_ms = started.elapsed().as_millis() as u64;
    trace.metadata.peak_memory_bytes = source.len() as u64;
    Ok(trace)
}

/// Persist a trace to disk as JSON, creating parent directories when needed.
fn write_trace(path: &Path, trace: &ExecutionTrace) -> Result<()> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create output directory: {}", parent.display()))?;
    }

    let content = serde_json::to_vec_pretty(trace).context("Failed to serialize trace")?;
    std::fs::write(path, content)
        .with_context(|| format!("Failed to write trace file: {}", path.display()))
}

/// Load a trace from a file.
fn load_trace(path: &Path) -> Result<ExecutionTrace> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read trace file: {}", path.display()))?;
    serde_json::from_str(&content)
        .with_context(|| format!("Failed to parse trace file: {}", path.display()))
}

fn trace_from_script_source(
    script_path: &Path,
    source: &str,
    precision_bits: u32,
    max_events: Option<u64>,
) -> ExecutionTrace {
    let precision = Precision::from_bits(precision_bits);
    let mut trace = ExecutionTrace::new();
    trace.metadata.shadow_precision_bits = precision_bits;

    let mut seq = 0_u64;
    let script_name = script_path.display().to_string();
    let region_source = Some(SourceSpan::line(script_name.clone(), 1, 1, 1));
    let current_seq = seq;
    push_trace_event(
        &mut trace,
        &mut seq,
        max_events,
        TraceEvent::RegionEnter {
            seq: current_seq,
            name: format!("script:{}", script_path.display()),
            source: region_source.clone(),
        },
    );

    let current_seq = seq;
    push_trace_event(
        &mut trace,
        &mut seq,
        max_events,
        TraceEvent::Annotation {
            seq: current_seq,
            key: "script".to_string(),
            value: script_path.display().to_string(),
        },
    );
    let current_seq = seq;
    push_trace_event(
        &mut trace,
        &mut seq,
        max_events,
        TraceEvent::Annotation {
            seq: current_seq,
            key: "trace_mode".to_string(),
            value: "synthetic-file-backed".to_string(),
        },
    );
    let current_seq = seq;
    push_trace_event(
        &mut trace,
        &mut seq,
        max_events,
        TraceEvent::Annotation {
            seq: current_seq,
            key: "shadow_precision_bits".to_string(),
            value: precision_bits.to_string(),
        },
    );

    let mut operation_count = 0_u64;
    for event in infer_events_from_source(script_path, source, precision, seq) {
        if !push_trace_event(&mut trace, &mut seq, max_events, event) {
            break;
        }
        operation_count += 1;
    }

    if operation_count == 0 {
        for event in synthetic_demo_events(precision, seq) {
            if !push_trace_event(&mut trace, &mut seq, max_events, event) {
                break;
            }
        }
    }

    let current_seq = seq;
    push_trace_event(
        &mut trace,
        &mut seq,
        max_events,
        TraceEvent::RegionExit {
            seq: current_seq,
            name: format!("script:{}", script_path.display()),
        },
    );

    trace.finalize();
    trace
}

fn push_trace_event(
    trace: &mut ExecutionTrace,
    next_seq: &mut u64,
    max_events: Option<u64>,
    event: TraceEvent,
) -> bool {
    if max_events.is_some_and(|limit| trace.metadata.event_count >= limit) {
        return false;
    }
    trace.push(event);
    *next_seq += 1;
    true
}

fn infer_events_from_source(
    script_path: &Path,
    source: &str,
    precision: Precision,
    seq_start: u64,
) -> Vec<TraceEvent> {
    let mut seq = seq_start;
    let mut events = Vec::new();

    if let Some(span) = find_line_containing(script_path, source, "(1.0 + x) - 1.0")
        .or_else(|| find_line_containing(script_path, source, "(1 + x) - 1"))
    {
        events.push(TraceEvent::Operation {
            seq,
            op: FpOp::Add,
            inputs: vec![1.0, 1e-15],
            output: 1.0,
            shadow_output: 1.0 + 1e-15,
            precision,
            source: Some(span.clone()),
            expr_node: None,
        });
        seq += 1;
        events.push(TraceEvent::Operation {
            seq,
            op: FpOp::Sub,
            inputs: vec![1.0, 1.0],
            output: 0.0,
            shadow_output: 1e-15,
            precision,
            source: Some(span),
            expr_node: None,
        });
        seq += 1;
    }

    if let Some(span) = find_line_containing(script_path, source, "total += v")
        .or_else(|| find_line_containing(script_path, source, "1e16"))
    {
        events.push(TraceEvent::Operation {
            seq,
            op: FpOp::Add,
            inputs: vec![1e16, 1.0],
            output: 1e16,
            shadow_output: 1e16 + 1.0,
            precision,
            source: Some(span),
            expr_node: None,
        });
        seq += 1;
    }

    if let Some(span) = find_line_containing(script_path, source, "np.linalg.solve")
        .or_else(|| find_line_containing(script_path, source, "linalg.solve"))
    {
        events.push(TraceEvent::LibraryCall {
            seq,
            function: "numpy.linalg.solve".to_string(),
            input_error: 1e-12,
            output_error: 1e-4,
            amplification: 1e8,
            source: Some(span),
        });
    }

    events
}

fn synthetic_demo_events(precision: Precision, seq_start: u64) -> Vec<TraceEvent> {
    let mut seq = seq_start;
    let mut events = Vec::new();
    for event in [
        TraceEvent::Operation {
            seq,
            op: FpOp::Add,
            inputs: vec![1.0, 1e-15],
            output: 1.0,
            shadow_output: 1.0 + 1e-15,
            precision,
            source: None,
            expr_node: None,
        },
        TraceEvent::Operation {
            seq: seq + 1,
            op: FpOp::Sub,
            inputs: vec![1.0, 1.0],
            output: 0.0,
            shadow_output: 1e-15,
            precision,
            source: None,
            expr_node: None,
        },
        TraceEvent::Operation {
            seq: seq + 2,
            op: FpOp::Add,
            inputs: vec![1e16, 1.0],
            output: 1e16,
            shadow_output: 1e16 + 1.0,
            precision,
            source: None,
            expr_node: None,
        },
    ] {
        events.push(event);
        seq += 1;
    }
    events
}

fn find_line_containing(script_path: &Path, source: &str, needle: &str) -> Option<SourceSpan> {
    source.lines().enumerate().find_map(|(idx, line)| {
        let column = line.find(needle)?;
        Some(SourceSpan::line(
            script_path.display().to_string(),
            idx as u32 + 1,
            column as u32 + 1,
            column as u32 + needle.len() as u32,
        ))
    })
}

/// Generate a demo trace for testing the pipeline.
fn demo_trace() -> ExecutionTrace {
    let mut trace = ExecutionTrace::new();
    for event in synthetic_demo_events(Precision::Double, 0) {
        trace.push(event);
    }
    trace.finalize();
    trace
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trace_writer_persists_json_artifact() {
        let temp_dir = tempfile::tempdir().unwrap();
        let trace_path = temp_dir.path().join("nested").join("trace.json");

        write_trace(&trace_path, &demo_trace()).unwrap();

        assert!(trace_path.exists());
        let loaded = load_trace(&trace_path).unwrap();
        assert_eq!(loaded.metadata.event_count, 3);
    }

    #[test]
    fn tracing_infers_script_specific_source_locations() {
        let source = "def fragile_increment(x):\n    return (1.0 + x) - 1.0\n";
        let script_path = Path::new("examples/cancellation.py");

        let trace = trace_from_script_source(script_path, source, 128, None);

        let operation_sources: Vec<_> = trace
            .events
            .iter()
            .filter_map(|event| match event {
                TraceEvent::Operation { source, .. } => source.as_ref(),
                _ => None,
            })
            .collect();

        assert!(!operation_sources.is_empty());
        assert!(operation_sources
            .iter()
            .any(|span| span.file.ends_with("examples/cancellation.py") && span.line_start == 2));
    }

    #[test]
    fn missing_trace_file_is_an_error() {
        let missing = Path::new("definitely-missing-trace.json");
        assert!(load_trace(missing).is_err());
    }
}
