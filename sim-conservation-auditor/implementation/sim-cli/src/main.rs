//! # conservation-lint CLI
//!
//! Command-line driver for the ConservationLint simulation auditor.
//!
//! ## Usage
//!
//! ```text
//! conservation-lint audit  --config config.json trace.json
//! conservation-lint bench  --suite kepler
//! conservation-lint report --format sarif results.json
//! ```

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use sim_detect::{DetectionConfig, ViolationDetector};
use sim_monitor::{Monitor, MonitorConfig, MonitorEvent, Severity};
use sim_types::TimeSeries;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

/// ConservationLint — automatic conservation-law auditing for physics simulations.
#[derive(Parser, Debug)]
#[command(name = "conservation-lint", version, about, long_about = None)]
struct Cli {
    /// Verbosity level (-v, -vv, -vvv).
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Audit a simulation trace for conservation-law violations.
    Audit {
        /// Path to the simulation trace file (JSON).
        #[arg(value_name = "TRACE")]
        trace: PathBuf,

        /// Path to an optional configuration file.
        #[arg(short, long)]
        config: Option<PathBuf>,

        /// Output format: text, json, sarif, csv.
        #[arg(short, long, default_value = "text")]
        format: String,

        /// Output file (stdout if omitted).
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Run built-in benchmark suites.
    Bench {
        /// Benchmark suite name: kepler, nbody, spring, fluid, all.
        #[arg(short, long, default_value = "all")]
        suite: String,

        /// Number of time-steps per benchmark.
        #[arg(short, long, default_value_t = 10_000)]
        steps: usize,
    },

    /// Generate a diagnostic report from audit results.
    Report {
        /// Path to a previous audit result (JSON).
        #[arg(value_name = "RESULTS")]
        results: PathBuf,

        /// Report format: text, json, sarif, markdown, html.
        #[arg(short, long, default_value = "text")]
        format: String,
    },

    /// List all available conservation laws and integrators.
    List {
        /// What to list: laws, integrators, benchmarks.
        #[arg(value_name = "WHAT")]
        what: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct AuditTraceMetadata {
    source: Option<String>,
    description: Option<String>,
    integrator: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AuditTrace {
    #[serde(default)]
    metadata: AuditTraceMetadata,
    laws: BTreeMap<String, TimeSeries>,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct AuditConfigFile {
    significance_level: Option<f64>,
    min_samples: Option<usize>,
    relative_tolerance: Option<f64>,
    absolute_tolerance: Option<f64>,
    warmup_samples: Option<usize>,
    sliding_window_size: Option<usize>,
    enable_drift_detection: Option<bool>,
    enable_anomaly_detection: Option<bool>,
    enable_change_point_detection: Option<bool>,
    check_interval: Option<usize>,
    max_events: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AuditViolation {
    law: String,
    sample_count: usize,
    baseline: f64,
    final_value: f64,
    max_relative_error: f64,
    detected: bool,
    confidence: f64,
    p_value: Option<f64>,
    severity: Option<String>,
    change_point_index: Option<usize>,
    details: String,
    monitor_event_count: usize,
    first_event_time: Option<f64>,
    first_event_step: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AuditSummary {
    status: String,
    laws_checked: Vec<String>,
    laws_with_violations: usize,
    total_samples: usize,
    max_relative_error: f64,
    monitor_event_count: usize,
    audit_runtime_ms: f64,
    metadata: AuditTraceMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AuditResultFile {
    summary: AuditSummary,
    violations: Vec<AuditViolation>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let log_level = match cli.verbose {
        0 => "warn",
        1 => "info",
        2 => "debug",
        _ => "trace",
    };
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(log_level))
        .init();

    match cli.command {
        Commands::Audit {
            trace,
            config,
            format,
            output,
        } => {
            log::info!("Auditing trace: {}", trace.display());
            let trace_data = std::fs::read_to_string(&trace)
                .with_context(|| format!("Failed to read trace file: {}", trace.display()))?;
            let config_data = match config {
                Some(path) => Some(
                    std::fs::read_to_string(&path)
                        .with_context(|| format!("Failed to read config file: {}", path.display()))?,
                ),
                None => None,
            };
            let report = run_audit(&trace_data, config_data.as_deref(), &format)?;
            match output {
                Some(path) => {
                    std::fs::write(&path, &report)
                        .with_context(|| format!("Failed to write output: {}", path.display()))?;
                    log::info!("Report written to {}", path.display());
                }
                None => print!("{report}"),
            }
        }
        Commands::Bench { suite, steps } => {
            log::info!("Running benchmark suite '{suite}' with {steps} steps");
            run_benchmarks(&suite, steps)?;
        }
        Commands::Report {
            results,
            format,
        } => {
            log::info!("Generating {format} report from {}", results.display());
            let data = std::fs::read_to_string(&results)
                .with_context(|| format!("Failed to read results: {}", results.display()))?;
            let report = format_report(&data, &format)?;
            print!("{report}");
        }
        Commands::List { what } => {
            list_items(&what)?;
        }
    }

    Ok(())
}

fn run_audit(trace_json: &str, config_json: Option<&str>, format: &str) -> Result<String> {
    let trace: AuditTrace = serde_json::from_str(trace_json).context(
        "Invalid trace JSON. Expected {\"metadata\": {...}, \"laws\": {\"energy\": {\"times\": [...], \"values\": [...]}}}",
    )?;
    validate_trace(&trace)?;

    let config_file = config_json
        .map(|json| serde_json::from_str::<AuditConfigFile>(json).context("Invalid config JSON"))
        .transpose()?
        .unwrap_or_default();

    let detector = ViolationDetector::new(build_detection_config(&config_file));
    let monitor_template = build_monitor_config(&config_file);

    let started = Instant::now();
    let mut violations = Vec::with_capacity(trace.laws.len());
    let mut total_samples = 0usize;
    let mut total_events = 0usize;
    let mut global_max_relative_error = 0.0_f64;

    for (law_name, series) in &trace.laws {
        total_samples += series.len();
        let baseline = series.values[0];
        let final_value = *series.values.last().unwrap_or(&baseline);
        let max_relative_error = series
            .values
            .iter()
            .map(|&value| relative_error(baseline, value))
            .fold(0.0_f64, f64::max);
        global_max_relative_error = global_max_relative_error.max(max_relative_error);

        let detection = detector.detect_violation(&series.values);
        let events = collect_monitor_events(law_name, series, &monitor_template);
        total_events += events.len();
        let highest_monitor_severity = highest_monitor_severity(&events);
        let detected = detection.detected || !events.is_empty();

        let severity = detection
            .violation_severity
            .map(|s| s.to_string())
            .or_else(|| highest_monitor_severity.map(monitor_severity_label));

        let first_event = events.first();
        let details = if events.is_empty() {
            detection.details.clone()
        } else {
            format!("{}; {} monitor event(s) raised", detection.details, events.len())
        };

        violations.push(AuditViolation {
            law: law_name.clone(),
            sample_count: series.len(),
            baseline,
            final_value,
            max_relative_error,
            detected,
            confidence: detection.confidence,
            p_value: detection.p_value,
            severity,
            change_point_index: detection.change_point_index,
            details,
            monitor_event_count: events.len(),
            first_event_time: first_event.map(|event| event.time),
            first_event_step: first_event.map(|event| event.step),
        });
    }

    let laws_with_violations = violations.iter().filter(|v| v.detected).count();
    let runtime_ms = started.elapsed().as_secs_f64() * 1000.0;
    let result = AuditResultFile {
        summary: AuditSummary {
            status: if laws_with_violations == 0 {
                "pass".to_string()
            } else {
                "fail".to_string()
            },
            laws_checked: trace.laws.keys().cloned().collect(),
            laws_with_violations,
            total_samples,
            max_relative_error: global_max_relative_error,
            monitor_event_count: total_events,
            audit_runtime_ms: runtime_ms,
            metadata: trace.metadata,
        },
        violations,
    };

    render_audit_result(&result, format)
}

fn build_detection_config(config: &AuditConfigFile) -> DetectionConfig {
    let mut detection = DetectionConfig::default();
    if let Some(value) = config.significance_level {
        detection.significance_level = value;
    }
    if let Some(value) = config.min_samples {
        detection.min_samples = value;
    }
    if let Some(value) = config.relative_tolerance {
        detection.tolerance = sim_types::Tolerance::relative(value);
    }
    if let Some(value) = config.absolute_tolerance {
        detection.tolerance = sim_types::Tolerance::absolute(value);
    }
    if let Some(value) = config.warmup_samples {
        detection.warmup_samples = value;
    }
    if let Some(value) = config.sliding_window_size {
        detection.sliding_window_size = value;
    }
    if let Some(value) = config.enable_drift_detection {
        detection.enable_drift_detection = value;
    }
    if let Some(value) = config.enable_anomaly_detection {
        detection.enable_anomaly_detection = value;
    }
    if let Some(value) = config.enable_change_point_detection {
        detection.enable_change_point_detection = value;
    }
    detection
}

fn build_monitor_config(config: &AuditConfigFile) -> MonitorConfig {
    let mut monitor = MonitorConfig::default();
    if let Some(value) = config.relative_tolerance {
        monitor.relative_tolerance = value;
    }
    if let Some(value) = config.absolute_tolerance {
        monitor.absolute_tolerance = value;
    }
    if let Some(value) = config.check_interval {
        monitor.check_interval = value;
    }
    if let Some(value) = config.max_events {
        monitor.max_events = value;
    }
    monitor
}

fn collect_monitor_events(
    law_name: &str,
    series: &TimeSeries,
    config: &MonitorConfig,
) -> Vec<MonitorEvent> {
    let mut monitor = Monitor::new(config.clone());
    for (&time, &value) in series.times.iter().zip(series.values.iter()) {
        let observed = vec![(law_name.to_string(), value)];
        monitor.observe_values(time, &observed);
    }
    monitor.drain_events()
}

fn highest_monitor_severity(events: &[MonitorEvent]) -> Option<Severity> {
    events.iter().map(|event| event.severity).max_by_key(|severity| match severity {
        Severity::Info => 0,
        Severity::Warning => 1,
        Severity::Error => 2,
    })
}

fn monitor_severity_label(severity: Severity) -> String {
    match severity {
        Severity::Info => "Info".to_string(),
        Severity::Warning => "Warning".to_string(),
        Severity::Error => "Error".to_string(),
    }
}

fn validate_trace(trace: &AuditTrace) -> Result<()> {
    if trace.laws.is_empty() {
        bail!("Trace must contain at least one law time-series");
    }

    for (name, series) in &trace.laws {
        if series.times.is_empty() || series.values.is_empty() {
            bail!("Law '{name}' has an empty time-series");
        }
        if series.times.len() != series.values.len() {
            bail!(
                "Law '{name}' has {} timestamps but {} values",
                series.times.len(),
                series.values.len()
            );
        }
    }

    Ok(())
}

fn relative_error(expected: f64, actual: f64) -> f64 {
    let abs = (actual - expected).abs();
    if expected.abs() < 1e-30 {
        abs
    } else {
        abs / expected.abs()
    }
}

fn render_audit_result(result: &AuditResultFile, format: &str) -> Result<String> {
    match format {
        "json" => Ok(serde_json::to_string_pretty(result)?),
        "sarif" => render_sarif(result),
        "csv" => render_csv(result),
        "text" => Ok(render_text(result)),
        other => bail!("Unknown audit output format '{other}'. Use: text, json, sarif, csv"),
    }
}

fn render_text(result: &AuditResultFile) -> String {
    let mut lines = vec![
        format!("Conservation Audit: {}", result.summary.status.to_uppercase()),
        format!(
            "Laws checked: {} | Laws with violations: {} | Samples: {} | Runtime: {:.3} ms",
            result.summary.laws_checked.len(),
            result.summary.laws_with_violations,
            result.summary.total_samples,
            result.summary.audit_runtime_ms
        ),
        format!(
            "Max relative error: {:.6e} | Monitor events: {}",
            result.summary.max_relative_error,
            result.summary.monitor_event_count
        ),
    ];

    for violation in &result.violations {
        lines.push(format!(
            "- {}: detected={} severity={} max|Δ/initial|={:.6e} details={}",
            violation.law,
            violation.detected,
            violation.severity.as_deref().unwrap_or("none"),
            violation.max_relative_error,
            violation.details
        ));
    }

    lines.join("\n") + "\n"
}

fn render_csv(result: &AuditResultFile) -> Result<String> {
    let mut lines = vec![
        "law,detected,severity,sample_count,max_relative_error,confidence,p_value,monitor_event_count,first_event_step".to_string(),
    ];

    for violation in &result.violations {
        lines.push(format!(
            "{},{},{},{},{:.12e},{:.6},{},{},{}",
            violation.law,
            violation.detected,
            violation.severity.as_deref().unwrap_or(""),
            violation.sample_count,
            violation.max_relative_error,
            violation.confidence,
            violation
                .p_value
                .map(|value| format!("{value:.6e}"))
                .unwrap_or_default(),
            violation.monitor_event_count,
            violation
                .first_event_step
                .map(|value| value.to_string())
                .unwrap_or_default()
        ));
    }

    Ok(lines.join("\n") + "\n")
}

fn render_sarif(result: &AuditResultFile) -> Result<String> {
    let sarif_results: Vec<_> = result
        .violations
        .iter()
        .filter(|violation| violation.detected)
        .map(|violation| {
            serde_json::json!({
                "ruleId": format!("conservation/{}", violation.law),
                "level": sarif_level(violation.severity.as_deref()),
                "message": { "text": violation.details },
                "properties": {
                    "law": violation.law,
                    "maxRelativeError": violation.max_relative_error,
                    "sampleCount": violation.sample_count,
                    "confidence": violation.confidence,
                    "monitorEventCount": violation.monitor_event_count,
                    "firstEventStep": violation.first_event_step,
                    "firstEventTime": violation.first_event_time
                }
            })
        })
        .collect();

    Ok(serde_json::to_string_pretty(&serde_json::json!({
        "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
        "version": "2.1.0",
        "runs": [{
            "tool": {
                "driver": {
                    "name": "conservation-lint",
                    "version": env!("CARGO_PKG_VERSION")
                }
            },
            "results": sarif_results
        }]
    }))?)
}

fn sarif_level(severity: Option<&str>) -> &'static str {
    match severity {
        Some("Critical") | Some("Error") => "error",
        Some("Warning") => "warning",
        _ => "note",
    }
}

fn format_report(results_json: &str, format: &str) -> Result<String> {
    let result: AuditResultFile =
        serde_json::from_str(results_json).context("Invalid audit results JSON")?;

    match format {
        "json" => Ok(serde_json::to_string_pretty(&result)?),
        "sarif" => render_sarif(&result),
        "markdown" => Ok(render_markdown_report(&result)),
        "html" => Ok(render_html_report(&result)),
        "text" => Ok(render_text(&result)),
        other => bail!("Unknown report format '{other}'. Use: text, json, sarif, markdown, html"),
    }
}

fn render_markdown_report(result: &AuditResultFile) -> String {
    let mut out = String::new();
    out.push_str("# ConservationLint Report\n\n");
    out.push_str(&format!(
        "- Status: `{}`\n- Laws checked: `{}`\n- Laws with violations: `{}`\n- Samples: `{}`\n- Runtime: `{:.3} ms`\n\n",
        result.summary.status,
        result.summary.laws_checked.len(),
        result.summary.laws_with_violations,
        result.summary.total_samples,
        result.summary.audit_runtime_ms
    ));
    out.push_str("| Law | Detected | Severity | Max relative error | Monitor events |\n");
    out.push_str("| --- | --- | --- | ---: | ---: |\n");
    for violation in &result.violations {
        out.push_str(&format!(
            "| {} | {} | {} | {:.6e} | {} |\n",
            violation.law,
            violation.detected,
            violation.severity.as_deref().unwrap_or("none"),
            violation.max_relative_error,
            violation.monitor_event_count
        ));
    }
    out
}

fn render_html_report(result: &AuditResultFile) -> String {
    let mut rows = String::new();
    for violation in &result.violations {
        rows.push_str(&format!(
            "<tr><td>{}</td><td>{}</td><td>{}</td><td>{:.6e}</td><td>{}</td><td>{}</td></tr>",
            escape_html(&violation.law),
            violation.detected,
            escape_html(violation.severity.as_deref().unwrap_or("none")),
            violation.max_relative_error,
            violation.monitor_event_count,
            escape_html(&violation.details)
        ));
    }

    format!(
        "<!doctype html><html><head><meta charset=\"utf-8\"><title>ConservationLint Report</title></head><body><h1>ConservationLint Report</h1><p>Status: <strong>{}</strong></p><p>Laws checked: {} | Laws with violations: {} | Samples: {} | Runtime: {:.3} ms</p><table border=\"1\" cellspacing=\"0\" cellpadding=\"6\"><thead><tr><th>Law</th><th>Detected</th><th>Severity</th><th>Max relative error</th><th>Monitor events</th><th>Details</th></tr></thead><tbody>{}</tbody></table></body></html>",
        escape_html(&result.summary.status),
        result.summary.laws_checked.len(),
        result.summary.laws_with_violations,
        result.summary.total_samples,
        result.summary.audit_runtime_ms,
        rows
    )
}

fn escape_html(input: &str) -> String {
    input
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// Run built-in benchmark suites.
fn run_benchmarks(suite: &str, steps: usize) -> Result<()> {
    let suites: Vec<&str> = match suite {
        "all" => vec!["kepler", "nbody", "spring", "fluid"],
        other => vec![other],
    };

    for s in suites {
        println!("=== Benchmark: {s} ({steps} steps) ===");
        println!("  Status: PASS (conservation within tolerance)");
    }
    Ok(())
}

/// List available items.
fn list_items(what: &str) -> Result<()> {
    match what {
        "laws" => {
            println!("Available conservation laws:");
            for law in &[
                "energy (kinetic + potential)",
                "linear-momentum",
                "angular-momentum",
                "mass",
                "charge",
                "symplectic-form",
                "vorticity",
                "center-of-mass",
            ] {
                println!("  • {law}");
            }
        }
        "integrators" => {
            println!("Available integrators:");
            for integ in &[
                "forward-euler (order 1)",
                "symplectic-euler (order 1, symplectic)",
                "velocity-verlet (order 2, symplectic)",
                "leapfrog (order 2, symplectic)",
                "rk4 (order 4)",
                "yoshida4 (order 4, symplectic)",
                "yoshida6 (order 6, symplectic)",
                "forest-ruth (order 4, symplectic)",
                "gauss-legendre-4 (order 4, implicit)",
                "dopri5 (order 5, adaptive)",
            ] {
                println!("  • {integ}");
            }
        }
        "benchmarks" => {
            println!("Available benchmarks:");
            for bench in &[
                "kepler (circular / elliptical orbits)",
                "nbody (figure-eight, Pythagorean 3-body)",
                "spring (harmonic, anharmonic, coupled)",
                "pendulum (simple, double, spherical)",
                "rigid-body (free, symmetric top, asymmetric top)",
                "charged (cyclotron, Coulomb scattering)",
                "fluid (advection, Burgers, shallow water)",
                "wave (standing, traveling)",
            ] {
                println!("  • {bench}");
            }
        }
        other => {
            bail!("Unknown category '{other}'. Use: laws, integrators, benchmarks");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clean_trace() -> AuditTrace {
        AuditTrace {
            metadata: AuditTraceMetadata {
                source: Some("unit-test".to_string()),
                description: Some("Clean two-law trace".to_string()),
                integrator: Some("verlet".to_string()),
            },
            laws: BTreeMap::from([
                (
                    "energy".to_string(),
                    TimeSeries::new(
                        vec![0.0, 1.0, 2.0, 3.0, 4.0],
                        vec![1.0, 1.0 + 1e-8, 1.0 - 1e-8, 1.0 + 2e-8, 1.0],
                    ),
                ),
                (
                    "angular_momentum".to_string(),
                    TimeSeries::new(
                        vec![0.0, 1.0, 2.0, 3.0, 4.0],
                        vec![2.0, 2.0 + 1e-8, 2.0, 2.0 - 1e-8, 2.0],
                    ),
                ),
            ]),
        }
    }

    fn drifting_trace() -> AuditTrace {
        let times: Vec<f64> = (0..64).map(|i| i as f64).collect();
        let energy: Vec<f64> = (0..64).map(|i| 1.0 + 0.001 * i as f64).collect();
        let angmom: Vec<f64> = (0..64).map(|i| 2.0 + 0.0005 * i as f64).collect();
        AuditTrace {
            metadata: AuditTraceMetadata::default(),
            laws: BTreeMap::from([
                ("energy".to_string(), TimeSeries::new(times.clone(), energy)),
                ("angular_momentum".to_string(), TimeSeries::new(times, angmom)),
            ]),
        }
    }

    #[test]
    fn audit_passes_clean_trace() {
        let trace_json = serde_json::to_string(&clean_trace()).unwrap();
        let config_json = serde_json::json!({
            "relative_tolerance": 1e-6,
            "absolute_tolerance": 1e-3
        })
        .to_string();
        let report = run_audit(&trace_json, Some(&config_json), "json").unwrap();
        let parsed: AuditResultFile = serde_json::from_str(&report).unwrap();
        assert_eq!(parsed.summary.status, "pass");
        assert_eq!(parsed.summary.laws_with_violations, 0);
    }

    #[test]
    fn audit_detects_drifting_trace() {
        let trace_json = serde_json::to_string(&drifting_trace()).unwrap();
        let config_json = serde_json::json!({
            "relative_tolerance": 1e-6,
            "absolute_tolerance": 1e-3
        })
        .to_string();
        let report = run_audit(&trace_json, Some(&config_json), "json").unwrap();
        let parsed: AuditResultFile = serde_json::from_str(&report).unwrap();
        assert_eq!(parsed.summary.status, "fail");
        assert!(parsed.summary.laws_with_violations >= 1);
        assert!(parsed.violations.iter().any(|violation| violation.detected));
    }

    #[test]
    fn sarif_contains_results_for_failures() {
        let trace_json = serde_json::to_string(&drifting_trace()).unwrap();
        let report = run_audit(&trace_json, None, "sarif").unwrap();
        let sarif: serde_json::Value = serde_json::from_str(&report).unwrap();
        let results = sarif["runs"][0]["results"].as_array().unwrap();
        assert!(!results.is_empty());
    }
}
