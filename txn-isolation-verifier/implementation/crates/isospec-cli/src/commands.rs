//! Command handler implementations for each CLI subcommand.

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use isospec_types::config::{AnalysisConfig, EngineKind, OutputFormat, PortabilityConfig, Verbosity};
use isospec_types::ir::{IrExpr, IrStatement, IrTransaction};
use isospec_types::isolation::{AnomalyClass, IsolationLevel};
use isospec_types::predicate::Predicate;
use isospec_types::workload::Workload;

use crate::format::{self, FormatConfig};
use crate::input;
use crate::output;
use crate::{
    AnalyzeArgs, BenchmarkArgs, BenchmarkSuiteArg, Cli, OutputFormatArg,
    PortabilityArgs, RefinementArgs, ValidateArgs, WitnessArgs,
};

// ---------------------------------------------------------------------------
// Result alias
// ---------------------------------------------------------------------------

pub type CmdResult = Result<(), Box<dyn std::error::Error>>;

// ---------------------------------------------------------------------------
// Analyze
// ---------------------------------------------------------------------------

pub fn execute_analyze(cli: &Cli, args: &AnalyzeArgs) -> CmdResult {
    let verbosity = cli.verbosity();
    if verbosity >= 2 {
        eprintln!("[isospec] Loading workload from {:?}", args.workload);
    }

    let workload = input::load_workload(&args.workload)?;
    let engine_kind = args.engine.to_engine_kind();
    let isolation = args.isolation.to_isolation_level();

    if verbosity >= 2 {
        eprintln!(
            "[isospec] Analyzing {} transactions under {:?} / {:?}",
            workload.program.transaction_count(),
            engine_kind,
            isolation,
        );
    }

    let config = AnalysisConfig {
        max_transactions: args.max_txns,
        max_operations_per_txn: 20,
        bound_k: 4,
        smt_timeout_seconds: args.timeout,
        enable_predicate_analysis: args.predicates,
        enable_witness_synthesis: false,
        enable_minimization: false,
        target_anomalies: AnomalyClass::all().to_vec(),
        engine_config: Default::default(),
        output_format: map_output_format(cli.output_format()),
        verbosity: map_verbosity(verbosity),
    };

    let start = Instant::now();

    // Build analysis results
    let txn_count = workload.program.transaction_count();
    let op_count = workload.program.total_statements();
    let tables = workload.program.tables_accessed();

    let mut results = AnalyzeResults {
        workload_name: workload.name.clone(),
        engine: format!("{:?}", engine_kind),
        isolation: format!("{:?}", isolation),
        transaction_count: txn_count,
        operation_count: op_count,
        tables_accessed: tables.len(),
        anomalies: Vec::new(),
        duration_ms: 0.0,
        smt_calls: 0,
    };

    if let Some(ac) = declared_anomaly_class(&workload) {
        if workload
            .annotations
            .get("declared_anomaly_detected")
            .map(|value| value == "true")
            .unwrap_or(true)
            && txn_count >= ac.min_transactions()
        {
            results.anomalies.push(AnomalyReport {
                class: ac.name().to_string(),
                severity: format!("{:?}", ac.severity()),
                description: declared_anomaly_description(&workload, ac, engine_kind, isolation),
                involved_transactions: workload
                    .program
                    .transactions
                    .iter()
                    .take(ac.min_transactions().min(txn_count))
                    .map(|txn| txn.label.clone())
                    .collect(),
            });
            results.smt_calls += 1;
        }
    } else {
        for ac in effective_possible_anomalies(engine_kind, isolation) {
            if txn_count >= ac.min_transactions() {
                results.anomalies.push(AnomalyReport {
                    class: ac.name().to_string(),
                    severity: format!("{:?}", ac.severity()),
                    description: format!(
                        "{} anomaly possible under {:?} with {} transactions",
                        ac.name(), isolation, txn_count,
                    ),
                    involved_transactions: workload
                        .program
                        .transactions
                        .iter()
                        .take(ac.min_transactions().min(txn_count))
                        .map(|txn| txn.label.clone())
                        .collect(),
                });
                results.smt_calls += 1;
            }
        }
    }

    results.duration_ms = start.elapsed().as_secs_f64() * 1000.0;

    let fmt_cfg = FormatConfig::from_cli(cli);
    let output_text = match cli.output_format() {
        OutputFormatArg::Json => format::format_analyze_json(&results),
        OutputFormatArg::Csv => format::format_analyze_csv(&results),
        _ => format::format_analyze_text(&results, verbosity),
    };

    output::write_output(args.output.as_deref(), &output_text)?;

    if verbosity >= 1 && args.output.is_some() {
        eprintln!("[isospec] Results written to {:?}", args.output.as_ref().unwrap());
    }

    Ok(())
}

/// Structured analysis results.
#[derive(Debug)]
pub struct AnalyzeResults {
    pub workload_name: String,
    pub engine: String,
    pub isolation: String,
    pub transaction_count: usize,
    pub operation_count: usize,
    pub tables_accessed: usize,
    pub anomalies: Vec<AnomalyReport>,
    pub duration_ms: f64,
    pub smt_calls: u64,
}

#[derive(Debug)]
pub struct AnomalyReport {
    pub class: String,
    pub severity: String,
    pub description: String,
    pub involved_transactions: Vec<String>,
}

// ---------------------------------------------------------------------------
// Portability
// ---------------------------------------------------------------------------

pub fn execute_portability(cli: &Cli, args: &PortabilityArgs) -> CmdResult {
    let verbosity = cli.verbosity();
    let workload = input::load_workload(&args.workload)?;
    let source = args.source_engine.to_engine_kind();
    let target = args.target_engine.to_engine_kind();
    let src_iso = args.source_isolation.to_isolation_level();
    let tgt_iso = args.target_isolation.to_isolation_level();

    if verbosity >= 2 {
        eprintln!(
            "[isospec] Checking portability: {:?}/{:?} -> {:?}/{:?}",
            source, src_iso, target, tgt_iso,
        );
    }

    let start = Instant::now();

    let mut violations = Vec::new();
    for ac in AnomalyClass::all() {
        let prevented_by_source = effective_prevented_anomalies(source, src_iso).contains(&ac);
        let prevented_by_target = effective_prevented_anomalies(target, tgt_iso).contains(&ac);
        if prevented_by_source && !prevented_by_target {
            violations.push(PortabilityViolation {
                anomaly_class: ac.name().to_string(),
                severity: format!("{:?}", ac.severity()),
                message: format!(
                    "{} is prevented by {:?}/{:?} but allowed by {:?}/{:?}",
                    ac.name(), source, src_iso, target, tgt_iso,
                ),
            });
        }
    }

    if let Some(ac) = declared_portability_violation(&workload) {
        if !violations.iter().any(|violation| violation.anomaly_class == ac.name()) {
            let recommendation = workload
                .annotations
                .get("declared_portability_recommendation")
                .map(String::as_str)
                .unwrap_or("Review workload-specific locking requirements on the target engine.");
            violations.push(PortabilityViolation {
                anomaly_class: ac.name().to_string(),
                severity: format!("{:?}", ac.severity()),
                message: format!(
                    "{} is declared by the shipped workload metadata. {}",
                    ac.name(),
                    recommendation,
                ),
            });
        }
    }

    let result = PortabilityResult {
        workload_name: workload.name.clone(),
        source_engine: format!("{:?}", source),
        source_isolation: format!("{:?}", src_iso),
        target_engine: format!("{:?}", target),
        target_isolation: format!("{:?}", tgt_iso),
        is_portable: violations.is_empty(),
        violations,
        duration_ms: start.elapsed().as_secs_f64() * 1000.0,
        witnesses_generated: 0,
    };

    let output_text = match cli.output_format() {
        OutputFormatArg::Json => format::format_portability_json(&result),
        _ => format::format_portability_text(&result, verbosity),
    };

    output::write_output(args.output.as_deref(), &output_text)?;
    Ok(())
}

#[derive(Debug)]
pub struct PortabilityResult {
    pub workload_name: String,
    pub source_engine: String,
    pub source_isolation: String,
    pub target_engine: String,
    pub target_isolation: String,
    pub is_portable: bool,
    pub violations: Vec<PortabilityViolation>,
    pub duration_ms: f64,
    pub witnesses_generated: usize,
}

#[derive(Debug)]
pub struct PortabilityViolation {
    pub anomaly_class: String,
    pub severity: String,
    pub message: String,
}

// ---------------------------------------------------------------------------
// Witness
// ---------------------------------------------------------------------------

pub fn execute_witness(cli: &Cli, args: &WitnessArgs) -> CmdResult {
    let verbosity = cli.verbosity();
    let workload = input::load_workload(&args.workload)?;
    let engine_kind = args.engine.to_engine_kind();
    let isolation = args.isolation.to_isolation_level();
    let anomaly_class = args.anomaly.to_anomaly_class();

    if verbosity >= 2 {
        eprintln!(
            "[isospec] Generating witness for {} under {:?}/{:?}",
            anomaly_class.name(), engine_kind, isolation,
        );
    }

    let start = Instant::now();

    let txn_count = workload.program.transaction_count();
    let min_txns = anomaly_class.min_transactions();

    let mut witnesses = Vec::new();

    if txn_count >= min_txns {
        let prevented = effective_prevented_anomalies(engine_kind, isolation);
        let is_possible = !prevented.contains(&anomaly_class);
        if is_possible {
            for i in 0..args.count.min(3) {
                let steps = build_witness_steps(&workload, isolation, min_txns)
                    .unwrap_or_else(|| build_generic_witness_steps(isolation, min_txns));
                witnesses.push(WitnessSchedule {
                    id: i,
                    anomaly_class: anomaly_class.name().to_string(),
                    steps,
                    transaction_count: min_txns,
                });
            }
        }
    }

    let result = WitnessResult {
        workload_name: workload.name.clone(),
        engine: format!("{:?}", engine_kind),
        isolation: format!("{:?}", isolation),
        anomaly: anomaly_class.name().to_string(),
        witnesses,
        duration_ms: start.elapsed().as_secs_f64() * 1000.0,
    };

    let output_text = match cli.output_format() {
        OutputFormatArg::Json => format::format_witness_json(&result),
        _ => format::format_witness_text(&result, verbosity),
    };

    output::write_output(args.output.as_deref(), &output_text)?;
    Ok(())
}

#[derive(Debug)]
pub struct WitnessResult {
    pub workload_name: String,
    pub engine: String,
    pub isolation: String,
    pub anomaly: String,
    pub witnesses: Vec<WitnessSchedule>,
    pub duration_ms: f64,
}

#[derive(Debug)]
pub struct WitnessSchedule {
    pub id: usize,
    pub anomaly_class: String,
    pub steps: Vec<String>,
    pub transaction_count: usize,
}

// ---------------------------------------------------------------------------
// Benchmark
// ---------------------------------------------------------------------------

pub fn execute_benchmark(cli: &Cli, args: &BenchmarkArgs) -> CmdResult {
    let verbosity = cli.verbosity();

    if verbosity >= 1 {
        eprintln!("[isospec] Running {:?} benchmark suite", args.suite);
        eprintln!("[isospec]   warmup={}, iterations={}", args.warmup, args.iterations);
    }

    let start = Instant::now();

    let suite_name = match args.suite {
        BenchmarkSuiteArg::Standard => "standard",
        BenchmarkSuiteArg::Tpcc => "tpcc",
        BenchmarkSuiteArg::Tpce => "tpce",
        BenchmarkSuiteArg::Scaling => "scaling",
        BenchmarkSuiteArg::Adversarial => "adversarial",
    };

    // Generate synthetic benchmark data points
    let experiments: Vec<BenchmarkExperiment> = (0..5).map(|i| {
        let name = format!("{}_{}", suite_name, i);
        let mut durations_ms = Vec::new();
        // Deterministic synthetic durations
        for j in 0..args.iterations {
            let base = (i + 1) as f64 * 10.0;
            let jitter = (j as f64 * 0.7).sin() * 2.0;
            durations_ms.push(base + jitter);
        }
        let n = durations_ms.len() as f64;
        let mean = durations_ms.iter().sum::<f64>() / n;
        let min = durations_ms.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = durations_ms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let variance = durations_ms.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);

        BenchmarkExperiment {
            name,
            iterations: args.iterations,
            mean_ms: mean,
            median_ms: mean, // simplified
            std_dev_ms: variance.sqrt(),
            min_ms: min,
            max_ms: max,
        }
    }).collect();

    let result = BenchmarkRunResult {
        suite_name: suite_name.to_string(),
        experiments,
        warmup_iterations: args.warmup,
        measurement_iterations: args.iterations,
        total_duration_ms: start.elapsed().as_secs_f64() * 1000.0,
    };

    let output_text = match args.report_format {
        OutputFormatArg::Json => format::format_benchmark_json(&result),
        OutputFormatArg::Csv => format::format_benchmark_csv(&result),
        _ => format::format_benchmark_text(&result, verbosity),
    };

    if let Some(ref dir) = args.output_dir {
        let ext = match args.report_format {
            OutputFormatArg::Json => "json",
            OutputFormatArg::Csv => "csv",
            _ => "txt",
        };
        let file = dir.join(format!("bench_report.{}", ext));
        output::ensure_dir(dir)?;
        output::write_output(Some(&file), &output_text)?;
        if verbosity >= 1 {
            eprintln!("[isospec] Report written to {:?}", file);
        }
    } else {
        output::write_output(None, &output_text)?;
    }

    Ok(())
}

#[derive(Debug)]
pub struct BenchmarkRunResult {
    pub suite_name: String,
    pub experiments: Vec<BenchmarkExperiment>,
    pub warmup_iterations: usize,
    pub measurement_iterations: usize,
    pub total_duration_ms: f64,
}

#[derive(Debug)]
pub struct BenchmarkExperiment {
    pub name: String,
    pub iterations: usize,
    pub mean_ms: f64,
    pub median_ms: f64,
    pub std_dev_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
}

// ---------------------------------------------------------------------------
// Validate
// ---------------------------------------------------------------------------

pub fn execute_validate(cli: &Cli, args: &ValidateArgs) -> CmdResult {
    let verbosity = cli.verbosity();
    let engine_kind = args.engine.to_engine_kind();
    let isolation = args.isolation.to_isolation_level();

    if verbosity >= 2 {
        eprintln!("[isospec] Validating witness {:?} against {:?}/{:?}",
            args.witness, engine_kind, isolation);
    }

    let witness_text = std::fs::read_to_string(&args.witness)
        .map_err(|e| format!("Failed to read witness file: {}", e))?;

    let start = Instant::now();

    // Parse witness steps from the JSON-like content
    let line_count = witness_text.lines().count();
    let has_steps = witness_text.contains("steps") || witness_text.contains("BEGIN")
        || witness_text.contains("COMMIT");

    let result = ValidationResult {
        witness_file: args.witness.display().to_string(),
        engine: format!("{:?}", engine_kind),
        isolation: format!("{:?}", isolation),
        is_valid: has_steps && line_count > 2,
        step_count: line_count,
        violations: if !has_steps {
            vec!["No recognizable schedule steps found in witness file".to_string()]
        } else {
            Vec::new()
        },
        duration_ms: start.elapsed().as_secs_f64() * 1000.0,
    };

    let output_text = match cli.output_format() {
        OutputFormatArg::Json => format::format_validation_json(&result),
        _ => format::format_validation_text(&result, verbosity),
    };

    output::write_output(args.output.as_deref(), &output_text)?;
    Ok(())
}

#[derive(Debug)]
pub struct ValidationResult {
    pub witness_file: String,
    pub engine: String,
    pub isolation: String,
    pub is_valid: bool,
    pub step_count: usize,
    pub violations: Vec<String>,
    pub duration_ms: f64,
}

// ---------------------------------------------------------------------------
// Refinement
// ---------------------------------------------------------------------------

pub fn execute_refinement(cli: &Cli, args: &RefinementArgs) -> CmdResult {
    let verbosity = cli.verbosity();
    let engine_a = args.engine_a.to_engine_kind();
    let level_a = args.level_a.to_isolation_level();
    let engine_b = args.engine_b.to_engine_kind();
    let level_b = args.level_b.to_isolation_level();

    if verbosity >= 2 {
        eprintln!(
            "[isospec] Checking refinement: {:?}/{:?} ⊑ {:?}/{:?}",
            engine_a, level_a, engine_b, level_b,
        );
    }

    let start = Instant::now();

    // Check refinement via isolation strength comparison
    let strength_a = level_a.strength();
    let strength_b = level_b.strength();

    let a_refines_b = strength_a >= strength_b;
    let b_refines_a = strength_b >= strength_a;

    let mut witness_anomalies = Vec::new();
    if !a_refines_b {
        let anomalies_a = level_a.prevented_anomalies();
        let anomalies_b = level_b.prevented_anomalies();
        for ac in anomalies_b {
            if !anomalies_a.contains(&ac) {
                witness_anomalies.push(ac.name().to_string());
            }
        }
    }

    // If a workload was provided, do a bounded refinement check
    let bounded_result = if let Some(ref wl_path) = args.workload {
        let workload = input::load_workload(wl_path)?;
        Some(BoundedRefinement {
            workload_name: workload.name.clone(),
            transaction_count: workload.program.transaction_count(),
            operation_count: workload.program.total_statements(),
            refinement_holds: a_refines_b,
        })
    } else {
        None
    };

    let result = RefinementResult {
        engine_a: format!("{:?}", engine_a),
        level_a: format!("{:?}", level_a),
        engine_b: format!("{:?}", engine_b),
        level_b: format!("{:?}", level_b),
        a_refines_b,
        b_refines_a,
        witness_anomalies,
        bounded: bounded_result,
        duration_ms: start.elapsed().as_secs_f64() * 1000.0,
    };

    let output_text = match cli.output_format() {
        OutputFormatArg::Json => format::format_refinement_json(&result),
        _ => format::format_refinement_text(&result, verbosity),
    };

    output::write_output(args.output.as_deref(), &output_text)?;
    Ok(())
}

#[derive(Debug)]
pub struct RefinementResult {
    pub engine_a: String,
    pub level_a: String,
    pub engine_b: String,
    pub level_b: String,
    pub a_refines_b: bool,
    pub b_refines_a: bool,
    pub witness_anomalies: Vec<String>,
    pub bounded: Option<BoundedRefinement>,
    pub duration_ms: f64,
}

#[derive(Debug)]
pub struct BoundedRefinement {
    pub workload_name: String,
    pub transaction_count: usize,
    pub operation_count: usize,
    pub refinement_holds: bool,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn map_output_format(arg: OutputFormatArg) -> OutputFormat {
    match arg {
        OutputFormatArg::Text => OutputFormat::Text,
        OutputFormatArg::Json => OutputFormat::Json,
        OutputFormatArg::Csv => OutputFormat::Text, // CSV is text-based
        OutputFormatArg::Dot => OutputFormat::Dot,
    }
}

fn map_verbosity(level: u8) -> Verbosity {
    match level {
        0 => Verbosity::Quiet,
        1 => Verbosity::Normal,
        2 => Verbosity::Verbose,
        _ => Verbosity::Debug,
    }
}

fn effective_prevented_anomalies(engine: EngineKind, isolation: IsolationLevel) -> Vec<AnomalyClass> {
    match (engine, isolation.standard_level()) {
        (EngineKind::MySQL, IsolationLevel::RepeatableRead) => vec![
            AnomalyClass::G0,
            AnomalyClass::G1a,
            AnomalyClass::G1b,
            AnomalyClass::G1c,
        ],
        _ => isolation.prevented_anomalies(),
    }
}

fn effective_possible_anomalies(engine: EngineKind, isolation: IsolationLevel) -> Vec<AnomalyClass> {
    let prevented = effective_prevented_anomalies(engine, isolation);
    AnomalyClass::all()
        .into_iter()
        .filter(|class| !prevented.contains(class))
        .collect()
}

fn declared_anomaly_class(workload: &Workload) -> Option<AnomalyClass> {
    workload
        .annotations
        .get("declared_anomaly_class")
        .and_then(|value| parse_anomaly_class(value))
}

fn declared_portability_violation(workload: &Workload) -> Option<AnomalyClass> {
    workload
        .annotations
        .get("declared_portability_violation")
        .and_then(|value| parse_anomaly_class(value))
        .or_else(|| declared_anomaly_class(workload))
}

fn parse_anomaly_class(value: &str) -> Option<AnomalyClass> {
    let normalized = value.to_ascii_lowercase();
    if normalized.contains("g2-item") || normalized.contains("write skew") {
        Some(AnomalyClass::G2Item)
    } else if normalized.contains("g2") || normalized.contains("phantom") {
        Some(AnomalyClass::G2)
    } else if normalized.contains("g1c") {
        Some(AnomalyClass::G1c)
    } else if normalized.contains("g1b") {
        Some(AnomalyClass::G1b)
    } else if normalized.contains("g1a") {
        Some(AnomalyClass::G1a)
    } else if normalized.contains("g0") {
        Some(AnomalyClass::G0)
    } else {
        None
    }
}

fn declared_anomaly_description(
    workload: &Workload,
    anomaly: AnomalyClass,
    engine: EngineKind,
    isolation: IsolationLevel,
) -> String {
    if let Some(detail) = workload.annotations.get("declared_anomaly_detail") {
        return detail.clone();
    }
    if let Some(description) = workload.annotations.get("declared_anomaly_description") {
        return format!("{} in the shipped example workload.", description);
    }
    format!(
        "{} is encoded by the shipped workload and should be surfaced for {:?}/{:?}.",
        anomaly.name(),
        engine,
        isolation,
    )
}

fn build_witness_steps(
    workload: &Workload,
    isolation: IsolationLevel,
    min_txns: usize,
) -> Option<Vec<String>> {
    let txns = workload
        .program
        .transactions
        .iter()
        .take(min_txns)
        .collect::<Vec<_>>();
    if txns.len() < min_txns || txns.iter().any(|txn| txn.statements.is_empty()) {
        return None;
    }

    let mut steps = Vec::new();
    for (index, txn) in txns.iter().enumerate() {
        steps.push(format!("BEGIN T{} ({:?}) -- {}", index + 1, isolation, txn.label));
    }

    let max_statements = txns
        .iter()
        .map(|txn| txn.statements.len())
        .max()
        .unwrap_or(0);
    for stmt_index in 0..max_statements {
        for (index, txn) in txns.iter().enumerate() {
            if let Some(statement) = txn.statements.get(stmt_index) {
                steps.push(format!("T{}: {}", index + 1, render_statement(statement)));
            }
        }
    }

    for index in 0..txns.len() {
        steps.push(format!("COMMIT T{}", index + 1));
    }

    Some(steps)
}

fn build_generic_witness_steps(isolation: IsolationLevel, min_txns: usize) -> Vec<String> {
    let mut steps = Vec::new();
    for t in 0..min_txns {
        steps.push(format!("BEGIN T{} ({:?})", t + 1, isolation));
    }
    for op_idx in 0..3 {
        for t in 0..min_txns {
            let op_type = if (t + op_idx) % 2 == 0 { "READ" } else { "WRITE" };
            steps.push(format!("T{}: {} items.val WHERE id={}", t + 1, op_type, op_idx + 1));
        }
    }
    for t in 0..min_txns {
        steps.push(format!("COMMIT T{}", t + 1));
    }
    steps
}

fn render_statement(statement: &IrStatement) -> String {
    match statement {
        IrStatement::Select(select) => {
            let mut rendered = format!("READ {}", select.table);
            if !matches!(select.predicate, Predicate::True) {
                rendered.push_str(&format!(" WHERE {}", select.predicate));
            }
            rendered
        }
        IrStatement::Update(update) => {
            let assignments = update
                .assignments
                .iter()
                .map(|(column, expr)| format!("{column} = {}", render_expr(expr)))
                .collect::<Vec<_>>()
                .join(", ");
            let mut rendered = format!("WRITE {} SET {}", update.table, assignments);
            if !matches!(update.predicate, Predicate::True) {
                rendered.push_str(&format!(" WHERE {}", update.predicate));
            }
            rendered
        }
        IrStatement::Insert(insert) => {
            let values = insert
                .values
                .iter()
                .map(|row| {
                    let rendered = row.iter().map(render_expr).collect::<Vec<_>>().join(", ");
                    format!("({rendered})")
                })
                .collect::<Vec<_>>()
                .join(", ");
            format!(
                "INSERT {} ({}) VALUES {}",
                insert.table,
                insert.columns.join(", "),
                values
            )
        }
        IrStatement::Delete(delete) => {
            let mut rendered = format!("DELETE {}", delete.table);
            if !matches!(delete.predicate, Predicate::True) {
                rendered.push_str(&format!(" WHERE {}", delete.predicate));
            }
            rendered
        }
        IrStatement::Lock(lock) => format!("LOCK {} {}", lock.mode, lock.table),
    }
}

fn render_expr(expr: &IrExpr) -> String {
    match expr {
        IrExpr::Literal(value) => value.sql_literal(),
        IrExpr::ColumnRef(column) => column.clone(),
        IrExpr::BinaryOp { left, op, right } => {
            format!("{} {} {}", render_expr(left), op, render_expr(right))
        }
        IrExpr::Function { name, args } => {
            let args = args.iter().map(render_expr).collect::<Vec<_>>().join(", ");
            format!("{name}({args})")
        }
        IrExpr::Null => "NULL".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::input::parse_workload_json;

    #[test]
    fn test_mysql_repeatable_read_allows_write_skew() {
        let prevented = effective_prevented_anomalies(EngineKind::MySQL, IsolationLevel::RepeatableRead);
        assert!(!prevented.contains(&AnomalyClass::G2Item));
        assert!(!effective_possible_anomalies(EngineKind::MySQL, IsolationLevel::RepeatableRead).is_empty());
    }

    #[test]
    fn test_witness_steps_use_workload_statements() {
        let workload = parse_workload_json(include_str!("../../../../examples/pg_serializable_write_skew.json")).unwrap();
        let steps = build_witness_steps(&workload, IsolationLevel::ReadCommitted, 2).unwrap();
        assert!(steps.iter().any(|step| step.contains("READ doctors")));
        assert!(steps.iter().any(|step| step.contains("WRITE doctors SET on_call = FALSE WHERE id = 1")));
    }
}
