//! Output formatting for CLI results.
//!
//! Provides text, JSON, and DOT formatters for all command result types.
//! Includes optional ANSI color support for terminal output.

use std::fmt::Write;

use crate::commands::{
    AnalyzeResults, AnomalyReport, BenchmarkExperiment, BenchmarkRunResult, PortabilityResult,
    RefinementResult, ValidationResult, WitnessResult,
};
use crate::{Cli, OutputFormatArg};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Formatting configuration derived from CLI flags.
#[derive(Debug, Clone)]
pub struct FormatConfig {
    pub color: bool,
    pub verbose: bool,
}

impl FormatConfig {
    pub fn from_cli(cli: &Cli) -> Self {
        Self {
            color: atty_stdout(),
            verbose: cli.verbosity() >= 2,
        }
    }
}

/// Heuristic check for terminal (no external crate needed).
fn atty_stdout() -> bool {
    // In a real build we'd use atty or is-terminal; here we assume true
    // when the TERM env var is set and NO_COLOR is absent.
    std::env::var("NO_COLOR").is_err() && std::env::var("TERM").is_ok()
}

// ---------------------------------------------------------------------------
// ANSI helpers
// ---------------------------------------------------------------------------

const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";
const RED: &str = "\x1b[31m";
const GREEN: &str = "\x1b[32m";
const YELLOW: &str = "\x1b[33m";
const CYAN: &str = "\x1b[36m";
const DIM: &str = "\x1b[2m";

fn color(s: &str, code: &str, enabled: bool) -> String {
    if enabled {
        format!("{}{}{}", code, s, RESET)
    } else {
        s.to_string()
    }
}

fn bold(s: &str, enabled: bool) -> String { color(s, BOLD, enabled) }
fn red(s: &str, enabled: bool) -> String { color(s, RED, enabled) }
fn green(s: &str, enabled: bool) -> String { color(s, GREEN, enabled) }
fn yellow(s: &str, enabled: bool) -> String { color(s, YELLOW, enabled) }
fn cyan(s: &str, enabled: bool) -> String { color(s, CYAN, enabled) }
fn dim(s: &str, enabled: bool) -> String { color(s, DIM, enabled) }

// ---------------------------------------------------------------------------
// Analyze formatters
// ---------------------------------------------------------------------------

pub fn format_analyze_text(result: &AnalyzeResults, verbosity: u8) -> String {
    let use_color = atty_stdout();
    let mut out = String::with_capacity(2048);

    let _ = writeln!(out, "{}", bold("═══ Anomaly Analysis ═══", use_color));
    let _ = writeln!(out, "Workload:     {}", cyan(&result.workload_name, use_color));
    let _ = writeln!(out, "Engine:       {}", result.engine);
    let _ = writeln!(out, "Isolation:    {}", result.isolation);
    let _ = writeln!(out, "Transactions: {}", result.transaction_count);
    let _ = writeln!(out, "Operations:   {}", result.operation_count);
    let _ = writeln!(out, "Tables:       {}", result.tables_accessed);
    let _ = writeln!(out);

    if result.anomalies.is_empty() {
        let _ = writeln!(out, "{}", green("✓ No anomalies detected.", use_color));
    } else {
        let _ = writeln!(out, "{}", red(&format!("✗ {} anomaly class(es) detected:", result.anomalies.len()), use_color));
        let _ = writeln!(out);
        for a in &result.anomalies {
            let sev_color = match a.severity.as_str() {
                "Critical" => RED,
                "High" => RED,
                "Medium" => YELLOW,
                _ => DIM,
            };
            let _ = writeln!(out, "  {} [{}]", bold(&a.class, use_color), color(&a.severity, sev_color, use_color));
            let _ = writeln!(out, "    {}", a.description);
            if verbosity >= 2 {
                let _ = writeln!(out, "    Transactions: {}", a.involved_transactions.join(", "));
            }
        }
    }

    let _ = writeln!(out);
    let _ = writeln!(out, "{}", dim(&format!("Analysis completed in {:.2}ms ({} SMT calls)", result.duration_ms, result.smt_calls), use_color));
    out
}

pub fn format_analyze_json(result: &AnalyzeResults) -> String {
    let mut out = String::with_capacity(2048);
    out.push_str("{\n");
    let _ = write!(out, "  \"workload\": \"{}\",\n", json_esc(&result.workload_name));
    let _ = write!(out, "  \"engine\": \"{}\",\n", json_esc(&result.engine));
    let _ = write!(out, "  \"isolation\": \"{}\",\n", json_esc(&result.isolation));
    let _ = write!(out, "  \"transaction_count\": {},\n", result.transaction_count);
    let _ = write!(out, "  \"operation_count\": {},\n", result.operation_count);
    let _ = write!(out, "  \"duration_ms\": {:.3},\n", result.duration_ms);
    let _ = write!(out, "  \"smt_calls\": {},\n", result.smt_calls);
    out.push_str("  \"anomalies\": [\n");
    for (i, a) in result.anomalies.iter().enumerate() {
        out.push_str("    {\n");
        let _ = write!(out, "      \"class\": \"{}\",\n", json_esc(&a.class));
        let _ = write!(out, "      \"severity\": \"{}\",\n", json_esc(&a.severity));
        let _ = write!(out, "      \"description\": \"{}\",\n", json_esc(&a.description));
        let txns: Vec<String> = a.involved_transactions.iter().map(|t| format!("\"{}\"", json_esc(t))).collect();
        let _ = write!(out, "      \"transactions\": [{}]\n", txns.join(", "));
        let comma = if i + 1 < result.anomalies.len() { "," } else { "" };
        let _ = writeln!(out, "    }}{}", comma);
    }
    out.push_str("  ]\n");
    out.push_str("}\n");
    out
}

pub fn format_analyze_csv(result: &AnalyzeResults) -> String {
    let mut out = String::new();
    out.push_str("anomaly_class,severity,description\n");
    for a in &result.anomalies {
        let _ = writeln!(out, "{},{},{}", csv_esc(&a.class), csv_esc(&a.severity), csv_esc(&a.description));
    }
    out
}

// ---------------------------------------------------------------------------
// Portability formatters
// ---------------------------------------------------------------------------

pub fn format_portability_text(result: &PortabilityResult, verbosity: u8) -> String {
    let c = atty_stdout();
    let mut out = String::with_capacity(1024);
    let _ = writeln!(out, "{}", bold("═══ Portability Check ═══", c));
    let _ = writeln!(out, "Workload: {}", cyan(&result.workload_name, c));
    let _ = writeln!(out, "Source:   {}/{}", result.source_engine, result.source_isolation);
    let _ = writeln!(out, "Target:   {}/{}", result.target_engine, result.target_isolation);
    let _ = writeln!(out);

    if result.is_portable {
        let _ = writeln!(out, "{}", green("✓ Workload is portable — no new anomalies on target.", c));
    } else {
        let _ = writeln!(out, "{}", red(&format!("✗ {} portability violation(s) found:", result.violations.len()), c));
        for v in &result.violations {
            let _ = writeln!(out, "  • {} [{}]: {}", bold(&v.anomaly_class, c), yellow(&v.severity, c), v.message);
        }
    }

    let _ = writeln!(out);
    let _ = writeln!(out, "{}", dim(&format!("Completed in {:.2}ms", result.duration_ms), c));
    out
}

pub fn format_portability_json(result: &PortabilityResult) -> String {
    let mut out = String::with_capacity(1024);
    out.push_str("{\n");
    let _ = write!(out, "  \"workload\": \"{}\",\n", json_esc(&result.workload_name));
    let _ = write!(out, "  \"source\": \"{}/{}\",\n", json_esc(&result.source_engine), json_esc(&result.source_isolation));
    let _ = write!(out, "  \"target\": \"{}/{}\",\n", json_esc(&result.target_engine), json_esc(&result.target_isolation));
    let _ = write!(out, "  \"is_portable\": {},\n", result.is_portable);
    let _ = write!(out, "  \"duration_ms\": {:.3},\n", result.duration_ms);
    out.push_str("  \"violations\": [\n");
    for (i, v) in result.violations.iter().enumerate() {
        let comma = if i + 1 < result.violations.len() { "," } else { "" };
        let _ = writeln!(out, "    {{\"anomaly\": \"{}\", \"severity\": \"{}\", \"message\": \"{}\"}}{}", json_esc(&v.anomaly_class), json_esc(&v.severity), json_esc(&v.message), comma);
    }
    out.push_str("  ]\n");
    out.push_str("}\n");
    out
}

// ---------------------------------------------------------------------------
// Witness formatters
// ---------------------------------------------------------------------------

pub fn format_witness_text(result: &WitnessResult, verbosity: u8) -> String {
    let c = atty_stdout();
    let mut out = String::with_capacity(2048);
    let _ = writeln!(out, "{}", bold("═══ Witness Generation ═══", c));
    let _ = writeln!(out, "Workload: {}", cyan(&result.workload_name, c));
    let _ = writeln!(out, "Engine:   {}", result.engine);
    let _ = writeln!(out, "Isolation:{}", result.isolation);
    let _ = writeln!(out, "Anomaly:  {}", yellow(&result.anomaly, c));
    let _ = writeln!(out);

    if result.witnesses.is_empty() {
        let _ = writeln!(out, "{}", dim("No witness could be generated for this configuration.", c));
    } else {
        let _ = writeln!(out, "Generated {} witness(es):", result.witnesses.len());
        for w in &result.witnesses {
            let _ = writeln!(out);
            let _ = writeln!(out, "{}",
                bold(&format!("── Witness #{} ({}, {} txns) ──", w.id, w.anomaly_class, w.transaction_count), c));
            for (si, step) in w.steps.iter().enumerate() {
                let _ = writeln!(out, "  {:3}. {}", si + 1, step);
            }
        }
    }

    let _ = writeln!(out);
    let _ = writeln!(out, "{}", dim(&format!("Completed in {:.2}ms", result.duration_ms), c));
    out
}

pub fn format_witness_json(result: &WitnessResult) -> String {
    let mut out = String::with_capacity(2048);
    out.push_str("{\n");
    let _ = write!(out, "  \"workload\": \"{}\",\n", json_esc(&result.workload_name));
    let _ = write!(out, "  \"engine\": \"{}\",\n", json_esc(&result.engine));
    let _ = write!(out, "  \"isolation\": \"{}\",\n", json_esc(&result.isolation));
    let _ = write!(out, "  \"anomaly\": \"{}\",\n", json_esc(&result.anomaly));
    let _ = write!(out, "  \"duration_ms\": {:.3},\n", result.duration_ms);
    out.push_str("  \"witnesses\": [\n");
    for (wi, w) in result.witnesses.iter().enumerate() {
        out.push_str("    {\n");
        let _ = write!(out, "      \"id\": {},\n", w.id);
        let _ = write!(out, "      \"anomaly_class\": \"{}\",\n", json_esc(&w.anomaly_class));
        let _ = write!(out, "      \"transaction_count\": {},\n", w.transaction_count);
        let steps_json: Vec<String> = w.steps.iter().map(|s| format!("\"{}\"", json_esc(s))).collect();
        let _ = write!(out, "      \"steps\": [{}]\n", steps_json.join(", "));
        let comma = if wi + 1 < result.witnesses.len() { "," } else { "" };
        let _ = writeln!(out, "    }}{}", comma);
    }
    out.push_str("  ]\n");
    out.push_str("}\n");
    out
}

// ---------------------------------------------------------------------------
// Benchmark formatters
// ---------------------------------------------------------------------------

pub fn format_benchmark_text(result: &BenchmarkRunResult, verbosity: u8) -> String {
    let c = atty_stdout();
    let mut out = String::with_capacity(2048);
    let _ = writeln!(out, "{}", bold("═══ Benchmark Results ═══", c));
    let _ = writeln!(out, "Suite:      {}", cyan(&result.suite_name, c));
    let _ = writeln!(out, "Warmup:     {}", result.warmup_iterations);
    let _ = writeln!(out, "Iterations: {}", result.measurement_iterations);
    let _ = writeln!(out, "Total time: {:.2}ms", result.total_duration_ms);
    let _ = writeln!(out);

    // Table header
    let _ = writeln!(out, "{:<25} {:>10} {:>10} {:>10} {:>10} {:>10}",
        bold("Experiment", c), "Mean(ms)", "Median", "StdDev", "Min", "Max");
    let _ = writeln!(out, "{}", "─".repeat(77));

    for exp in &result.experiments {
        let _ = writeln!(out, "{:<25} {:>10.2} {:>10.2} {:>10.2} {:>10.2} {:>10.2}",
            exp.name, exp.mean_ms, exp.median_ms, exp.std_dev_ms, exp.min_ms, exp.max_ms);
    }

    let _ = writeln!(out);
    let _ = writeln!(out, "{}", dim(&format!("Completed in {:.2}ms", result.total_duration_ms), c));
    out
}

pub fn format_benchmark_json(result: &BenchmarkRunResult) -> String {
    let mut out = String::with_capacity(2048);
    out.push_str("{\n");
    let _ = write!(out, "  \"suite\": \"{}\",\n", json_esc(&result.suite_name));
    let _ = write!(out, "  \"warmup\": {},\n", result.warmup_iterations);
    let _ = write!(out, "  \"iterations\": {},\n", result.measurement_iterations);
    let _ = write!(out, "  \"total_duration_ms\": {:.3},\n", result.total_duration_ms);
    out.push_str("  \"experiments\": [\n");
    for (i, exp) in result.experiments.iter().enumerate() {
        out.push_str("    {\n");
        let _ = write!(out, "      \"name\": \"{}\",\n", json_esc(&exp.name));
        let _ = write!(out, "      \"iterations\": {},\n", exp.iterations);
        let _ = write!(out, "      \"mean_ms\": {:.3},\n", exp.mean_ms);
        let _ = write!(out, "      \"median_ms\": {:.3},\n", exp.median_ms);
        let _ = write!(out, "      \"std_dev_ms\": {:.3},\n", exp.std_dev_ms);
        let _ = write!(out, "      \"min_ms\": {:.3},\n", exp.min_ms);
        let _ = write!(out, "      \"max_ms\": {:.3}\n", exp.max_ms);
        let comma = if i + 1 < result.experiments.len() { "," } else { "" };
        let _ = writeln!(out, "    }}{}", comma);
    }
    out.push_str("  ]\n");
    out.push_str("}\n");
    out
}

pub fn format_benchmark_csv(result: &BenchmarkRunResult) -> String {
    let mut out = String::new();
    out.push_str("name,iterations,mean_ms,median_ms,std_dev_ms,min_ms,max_ms\n");
    for exp in &result.experiments {
        let _ = writeln!(out, "{},{},{:.3},{:.3},{:.3},{:.3},{:.3}",
            csv_esc(&exp.name), exp.iterations, exp.mean_ms, exp.median_ms,
            exp.std_dev_ms, exp.min_ms, exp.max_ms);
    }
    out
}

// ---------------------------------------------------------------------------
// Validation formatters
// ---------------------------------------------------------------------------

pub fn format_validation_text(result: &ValidationResult, _verbosity: u8) -> String {
    let c = atty_stdout();
    let mut out = String::with_capacity(512);
    let _ = writeln!(out, "{}", bold("═══ Witness Validation ═══", c));
    let _ = writeln!(out, "File:      {}", result.witness_file);
    let _ = writeln!(out, "Engine:    {}", result.engine);
    let _ = writeln!(out, "Isolation: {}", result.isolation);
    let _ = writeln!(out, "Steps:     {}", result.step_count);
    let _ = writeln!(out);

    if result.is_valid {
        let _ = writeln!(out, "{}", green("✓ Witness is valid.", c));
    } else {
        let _ = writeln!(out, "{}", red("✗ Witness is NOT valid:", c));
        for v in &result.violations {
            let _ = writeln!(out, "  • {}", v);
        }
    }

    let _ = writeln!(out);
    let _ = writeln!(out, "{}", dim(&format!("Validated in {:.2}ms", result.duration_ms), c));
    out
}

pub fn format_validation_json(result: &ValidationResult) -> String {
    let mut out = String::with_capacity(512);
    out.push_str("{\n");
    let _ = write!(out, "  \"witness_file\": \"{}\",\n", json_esc(&result.witness_file));
    let _ = write!(out, "  \"engine\": \"{}\",\n", json_esc(&result.engine));
    let _ = write!(out, "  \"isolation\": \"{}\",\n", json_esc(&result.isolation));
    let _ = write!(out, "  \"is_valid\": {},\n", result.is_valid);
    let _ = write!(out, "  \"step_count\": {},\n", result.step_count);
    let viols: Vec<String> = result.violations.iter().map(|v| format!("\"{}\"", json_esc(v))).collect();
    let _ = write!(out, "  \"violations\": [{}],\n", viols.join(", "));
    let _ = write!(out, "  \"duration_ms\": {:.3}\n", result.duration_ms);
    out.push_str("}\n");
    out
}

// ---------------------------------------------------------------------------
// Refinement formatters
// ---------------------------------------------------------------------------

pub fn format_refinement_text(result: &RefinementResult, verbosity: u8) -> String {
    let c = atty_stdout();
    let mut out = String::with_capacity(1024);
    let _ = writeln!(out, "{}", bold("═══ Refinement Check ═══", c));
    let _ = writeln!(out, "A: {}/{}", result.engine_a, result.level_a);
    let _ = writeln!(out, "B: {}/{}", result.engine_b, result.level_b);
    let _ = writeln!(out);

    let check = |holds: bool, msg: &str| -> String {
        if holds {
            format!("{} {}", green("✓", c), msg)
        } else {
            format!("{} {}", red("✗", c), msg)
        }
    };

    let _ = writeln!(out, "{}", check(result.a_refines_b, &format!("{}/{} ⊑ {}/{}", result.engine_a, result.level_a, result.engine_b, result.level_b)));
    let _ = writeln!(out, "{}", check(result.b_refines_a, &format!("{}/{} ⊑ {}/{}", result.engine_b, result.level_b, result.engine_a, result.level_a)));

    if !result.witness_anomalies.is_empty() && verbosity >= 1 {
        let _ = writeln!(out);
        let _ = writeln!(out, "Witness anomalies (present in B but not A):");
        for a in &result.witness_anomalies {
            let _ = writeln!(out, "  • {}", yellow(a, c));
        }
    }

    if let Some(ref bounded) = result.bounded {
        let _ = writeln!(out);
        let _ = writeln!(out, "Bounded check on workload '{}':", cyan(&bounded.workload_name, c));
        let _ = writeln!(out, "  Transactions: {}, Operations: {}", bounded.transaction_count, bounded.operation_count);
        let _ = writeln!(out, "  {}",
            if bounded.refinement_holds {
                green("Refinement holds for this workload", c)
            } else {
                red("Refinement does NOT hold for this workload", c)
            }
        );
    }

    let _ = writeln!(out);
    let _ = writeln!(out, "{}", dim(&format!("Completed in {:.2}ms", result.duration_ms), c));
    out
}

pub fn format_refinement_json(result: &RefinementResult) -> String {
    let mut out = String::with_capacity(1024);
    out.push_str("{\n");
    let _ = write!(out, "  \"engine_a\": \"{}\",\n", json_esc(&result.engine_a));
    let _ = write!(out, "  \"level_a\": \"{}\",\n", json_esc(&result.level_a));
    let _ = write!(out, "  \"engine_b\": \"{}\",\n", json_esc(&result.engine_b));
    let _ = write!(out, "  \"level_b\": \"{}\",\n", json_esc(&result.level_b));
    let _ = write!(out, "  \"a_refines_b\": {},\n", result.a_refines_b);
    let _ = write!(out, "  \"b_refines_a\": {},\n", result.b_refines_a);
    let wa: Vec<String> = result.witness_anomalies.iter().map(|a| format!("\"{}\"", json_esc(a))).collect();
    let _ = write!(out, "  \"witness_anomalies\": [{}],\n", wa.join(", "));

    if let Some(ref bounded) = result.bounded {
        out.push_str("  \"bounded\": {\n");
        let _ = write!(out, "    \"workload\": \"{}\",\n", json_esc(&bounded.workload_name));
        let _ = write!(out, "    \"transaction_count\": {},\n", bounded.transaction_count);
        let _ = write!(out, "    \"operation_count\": {},\n", bounded.operation_count);
        let _ = write!(out, "    \"refinement_holds\": {}\n", bounded.refinement_holds);
        out.push_str("  },\n");
    } else {
        out.push_str("  \"bounded\": null,\n");
    }

    let _ = write!(out, "  \"duration_ms\": {:.3}\n", result.duration_ms);
    out.push_str("}\n");
    out
}

// ---------------------------------------------------------------------------
// Escaping helpers
// ---------------------------------------------------------------------------

fn json_esc(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

fn csv_esc(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}
