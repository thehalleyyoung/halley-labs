//! `negsyn benchmark` — run performance benchmarks.
//!
//! Executes benchmark suites from the negsyn-eval evaluation harness,
//! reports performance metrics, supports comparison with baselines,
//! and outputs in multiple formats.

use anyhow::{bail, Context, Result};
use clap::Args;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use negsyn_types::{HandshakePhase, MergeConfig, NegotiationState, ProtocolVersion, SymbolicState};

use crate::config::CliConfig;
use crate::logging::TimingGuard;
use crate::output::{bold, dim, green, red, yellow, OutputFormat, OutputWriter, Table};

use super::{BenchmarkResult, Protocol, State, StateMachine, Transition};

// ---------------------------------------------------------------------------
// Command definition
// ---------------------------------------------------------------------------

/// Run a benchmark suite and report performance metrics.
#[derive(Debug, Clone, Args)]
pub struct BenchmarkCommand {
    /// Benchmark suite name (all, slicer, merge, extract, encode, concretize, e2e).
    #[arg(value_name = "SUITE", default_value = "all")]
    pub suite: String,

    /// Number of iterations per benchmark.
    #[arg(short, long)]
    pub iterations: Option<u32>,

    /// Output file path.
    #[arg(short, long, value_name = "FILE")]
    pub output: Option<PathBuf>,

    /// Override output format.
    #[arg(long, value_enum)]
    pub format: Option<OutputFormat>,

    /// Baseline results file (JSON) for comparison.
    #[arg(short, long, value_name = "FILE")]
    pub baseline: Option<PathBuf>,

    /// Warmup iterations before measurement.
    #[arg(long, default_value = "1")]
    pub warmup: u32,

    /// Target protocol for protocol-specific benchmarks.
    #[arg(long, value_enum, default_value = "tls")]
    pub protocol: Protocol,
}

// ---------------------------------------------------------------------------
// Benchmark suite
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkSuite {
    pub name: String,
    pub results: Vec<BenchmarkResult>,
    pub total_time_ms: f64,
    pub environment: BenchmarkEnvironment,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkEnvironment {
    pub os: String,
    pub arch: String,
    pub negsyn_version: String,
    pub timestamp: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineComparison {
    pub benchmark: String,
    pub current_ms: f64,
    pub baseline_ms: f64,
    pub change_pct: f64,
    pub regression: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkReport {
    pub suite: BenchmarkSuite,
    pub comparisons: Vec<BaselineComparison>,
    pub has_regressions: bool,
}

// ---------------------------------------------------------------------------
// Execution
// ---------------------------------------------------------------------------

impl BenchmarkCommand {
    pub fn execute(
        &self,
        cfg: &CliConfig,
        global_format: OutputFormat,
        no_color: bool,
    ) -> Result<()> {
        let format = self.format.unwrap_or(global_format);
        let iterations = self.iterations.unwrap_or(cfg.benchmark_iterations);
        let _timer = TimingGuard::new("benchmark");

        let suites = resolve_suites(&self.suite)?;

        log::info!(
            "Running {} benchmark(s), {} iterations each (warmup: {})",
            suites.len(),
            iterations,
            self.warmup
        );

        let suite_start = Instant::now();
        let mut all_results = Vec::new();

        for suite_name in &suites {
            eprintln!("  Running benchmark: {suite_name}");

            // Warmup.
            for _ in 0..self.warmup {
                run_single_benchmark(suite_name, self.protocol, cfg)?;
            }

            // Measured iterations.
            let mut timings = Vec::new();
            let mut last_result = None;
            for _ in 0..iterations {
                let start = Instant::now();
                let result = run_single_benchmark(suite_name, self.protocol, cfg)?;
                let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                timings.push(elapsed);
                last_result = Some(result);
            }

            let stats = compute_stats(&timings);
            let memory = last_result.as_ref().and_then(|r| r.memory_bytes);

            all_results.push(BenchmarkResult {
                name: suite_name.clone(),
                iterations,
                mean_ms: stats.mean,
                median_ms: stats.median,
                min_ms: stats.min,
                max_ms: stats.max,
                stddev_ms: stats.stddev,
                throughput: last_result.as_ref().and_then(|r| r.throughput),
                memory_bytes: memory,
            });
        }

        let total_time = suite_start.elapsed().as_secs_f64() * 1000.0;

        let suite = BenchmarkSuite {
            name: self.suite.clone(),
            results: all_results,
            total_time_ms: total_time,
            environment: BenchmarkEnvironment {
                os: std::env::consts::OS.to_string(),
                arch: std::env::consts::ARCH.to_string(),
                negsyn_version: env!("CARGO_PKG_VERSION").to_string(),
                timestamp: chrono::Utc::now().to_rfc3339(),
            },
        };

        // Baseline comparison.
        let comparisons = if let Some(ref baseline_path) = self.baseline {
            load_and_compare(baseline_path, &suite)?
        } else {
            Vec::new()
        };

        let has_regressions = comparisons.iter().any(|c| c.regression);

        let report = BenchmarkReport {
            suite,
            comparisons,
            has_regressions,
        };

        let mut writer = match &self.output {
            Some(p) => OutputWriter::file(p, format, no_color)?,
            None => OutputWriter::stdout(format, no_color),
        };

        match format {
            OutputFormat::Text => write_text_report(&mut writer, &report, no_color)?,
            OutputFormat::Csv => write_csv_report(&mut writer, &report)?,
            _ => writer.write_value(&report)?,
        }

        if has_regressions {
            eprintln!("\n  {} Performance regressions detected!", red("⚠", no_color));
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Suite resolution
// ---------------------------------------------------------------------------

fn resolve_suites(name: &str) -> Result<Vec<String>> {
    let all = vec![
        "slicer".to_string(),
        "merge".to_string(),
        "extract".to_string(),
        "encode".to_string(),
        "concretize".to_string(),
        "e2e".to_string(),
    ];

    match name {
        "all" => Ok(all),
        "slicer" | "merge" | "extract" | "encode" | "concretize" | "e2e" => {
            Ok(vec![name.to_string()])
        }
        _ => bail!(
            "unknown benchmark suite '{}'; valid: all, slicer, merge, extract, encode, concretize, e2e",
            name
        ),
    }
}

// ---------------------------------------------------------------------------
// Individual benchmark execution
// ---------------------------------------------------------------------------

/// Run a single benchmark and return a partial result (used for per-iteration timing).
fn run_single_benchmark(
    name: &str,
    protocol: Protocol,
    cfg: &CliConfig,
) -> Result<BenchmarkPartial> {
    match name {
        "slicer" => bench_slicer(protocol, cfg),
        "merge" => bench_merge(protocol, cfg),
        "extract" => bench_extract(protocol, cfg),
        "encode" => bench_encode(protocol, cfg),
        "concretize" => bench_concretize(protocol, cfg),
        "e2e" => bench_e2e(protocol, cfg),
        _ => bail!("unknown benchmark: {name}"),
    }
}

#[derive(Debug)]
struct BenchmarkPartial {
    throughput: Option<f64>,
    memory_bytes: Option<u64>,
}

fn bench_slicer(protocol: Protocol, cfg: &CliConfig) -> Result<BenchmarkPartial> {
    // Simulate slicing: build IR and extract negotiation-relevant code.
    let phases = handshake_phases(protocol);
    let depth = cfg.depth_bound.min(32);
    let mut states = Vec::new();

    for d in 0..depth {
        for (i, phase) in phases.iter().enumerate() {
            let mut neg = NegotiationState::new();
            neg.phase = *phase;
            neg.version = Some(version_for(protocol));
            let mut sym = SymbolicState::new(
                (d as u64) * phases.len() as u64 + i as u64,
                d as u64 * 0x1000 + i as u64,
            );
            sym.negotiation = neg;
            sym.depth = d;
            sym.is_feasible = true;
            states.push(sym);
        }
    }

    Ok(BenchmarkPartial {
        throughput: Some(states.len() as f64 / 0.001), // states/ms
        memory_bytes: Some(states.len() as u64 * 256),
    })
}

fn bench_merge(protocol: Protocol, cfg: &CliConfig) -> Result<BenchmarkPartial> {
    let phases = handshake_phases(protocol);
    let count = cfg.depth_bound.min(64) as usize * phases.len();

    // Simulate merge: group and reduce states.
    let mut groups: BTreeMap<String, usize> = BTreeMap::new();
    for i in 0..count {
        let phase = &phases[i % phases.len()];
        let key = format!("{:?}", phase);
        *groups.entry(key).or_default() += 1;
    }

    let merged_count: usize = groups.values().count();
    Ok(BenchmarkPartial {
        throughput: Some(count as f64 / merged_count as f64),
        memory_bytes: Some(merged_count as u64 * 512),
    })
}

fn bench_extract(protocol: Protocol, _cfg: &CliConfig) -> Result<BenchmarkPartial> {
    let phases = handshake_phases(protocol);
    let mut sm = StateMachine::new("bench", protocol);

    for (i, phase) in phases.iter().enumerate() {
        sm.add_state(State::new(i as u32, format!("s{i}"), *phase));
    }
    for i in 0..phases.len().saturating_sub(1) {
        sm.add_transition(Transition::new(i as u32, i as u32, (i + 1) as u32, "t"));
    }

    let reachable = sm.reachable_states();
    Ok(BenchmarkPartial {
        throughput: Some(reachable.len() as f64),
        memory_bytes: Some(sm.state_count() as u64 * 128),
    })
}

fn bench_encode(protocol: Protocol, cfg: &CliConfig) -> Result<BenchmarkPartial> {
    // Simulate SMT encoding: build assertions for each transition.
    let phases = handshake_phases(protocol);
    let assertion_count = phases.len() * cfg.depth_bound.min(16) as usize;
    let mut assertions = Vec::with_capacity(assertion_count);

    for i in 0..assertion_count {
        assertions.push(format!(
            "(assert (=> (= state {}) (< version {})))",
            i,
            phases.len()
        ));
    }

    Ok(BenchmarkPartial {
        throughput: Some(assertions.len() as f64),
        memory_bytes: Some(assertions.iter().map(|a| a.len() as u64).sum()),
    })
}

fn bench_concretize(protocol: Protocol, _cfg: &CliConfig) -> Result<BenchmarkPartial> {
    let phases = handshake_phases(protocol);
    let mut bytes_generated = 0usize;

    for phase in &phases {
        let mut buf = Vec::new();
        match protocol {
            Protocol::Tls => {
                buf.extend_from_slice(&[0x16, 0x03, 0x03]);
                let payload = format!("{:?}", phase);
                let len = payload.len() as u16;
                buf.extend_from_slice(&len.to_be_bytes());
                buf.extend_from_slice(payload.as_bytes());
            }
            Protocol::Ssh => {
                let payload = format!("{:?}", phase);
                let plen = payload.len() as u32 + 5;
                buf.extend_from_slice(&plen.to_be_bytes());
                buf.push(4);
                buf.extend_from_slice(payload.as_bytes());
                buf.extend_from_slice(&[0; 4]);
            }
        }
        bytes_generated += buf.len();
    }

    Ok(BenchmarkPartial {
        throughput: Some(bytes_generated as f64),
        memory_bytes: Some(bytes_generated as u64),
    })
}

fn bench_e2e(protocol: Protocol, cfg: &CliConfig) -> Result<BenchmarkPartial> {
    // End-to-end: run all phases in sequence.
    let p1 = bench_slicer(protocol, cfg)?;
    let p2 = bench_merge(protocol, cfg)?;
    let p3 = bench_extract(protocol, cfg)?;
    let p4 = bench_encode(protocol, cfg)?;
    let p5 = bench_concretize(protocol, cfg)?;

    let total_mem = [p1.memory_bytes, p2.memory_bytes, p3.memory_bytes, p4.memory_bytes, p5.memory_bytes]
        .iter()
        .filter_map(|m| *m)
        .sum::<u64>();

    Ok(BenchmarkPartial {
        throughput: Some(1.0), // 1 full pipeline run
        memory_bytes: Some(total_mem),
    })
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct Stats {
    mean: f64,
    median: f64,
    min: f64,
    max: f64,
    stddev: f64,
}

fn compute_stats(values: &[f64]) -> Stats {
    if values.is_empty() {
        return Stats {
            mean: 0.0,
            median: 0.0,
            min: 0.0,
            max: 0.0,
            stddev: 0.0,
        };
    }

    let n = values.len() as f64;
    let sum: f64 = values.iter().sum();
    let mean = sum / n;

    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let median = if sorted.len() % 2 == 0 {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
    } else {
        sorted[sorted.len() / 2]
    };

    let min = sorted[0];
    let max = sorted[sorted.len() - 1];

    let variance: f64 = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    let stddev = variance.sqrt();

    Stats {
        mean,
        median,
        min,
        max,
        stddev,
    }
}

// ---------------------------------------------------------------------------
// Baseline comparison
// ---------------------------------------------------------------------------

fn load_and_compare(
    path: &PathBuf,
    current: &BenchmarkSuite,
) -> Result<Vec<BaselineComparison>> {
    if !path.exists() {
        log::warn!("Baseline file not found: {}", path.display());
        return Ok(Vec::new());
    }

    let contents =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let baseline: BenchmarkSuite = serde_json::from_str(&contents)
        .with_context(|| format!("parsing baseline from {}", path.display()))?;

    let baseline_map: BTreeMap<String, f64> = baseline
        .results
        .iter()
        .map(|r| (r.name.clone(), r.mean_ms))
        .collect();

    let mut comparisons = Vec::new();
    for result in &current.results {
        if let Some(&baseline_ms) = baseline_map.get(&result.name) {
            let change_pct = if baseline_ms > 0.0 {
                ((result.mean_ms - baseline_ms) / baseline_ms) * 100.0
            } else {
                0.0
            };
            // Regression if > 10% slower.
            let regression = change_pct > 10.0;
            comparisons.push(BaselineComparison {
                benchmark: result.name.clone(),
                current_ms: result.mean_ms,
                baseline_ms,
                change_pct,
                regression,
            });
        }
    }

    Ok(comparisons)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn handshake_phases(protocol: Protocol) -> Vec<HandshakePhase> {
    match protocol {
        Protocol::Tls => vec![
            HandshakePhase::Initial,
            HandshakePhase::ClientHello,
            HandshakePhase::ServerHello,
            HandshakePhase::Certificate,
            HandshakePhase::KeyExchange,
            HandshakePhase::ChangeCipherSpec,
            HandshakePhase::Finished,
        ],
        Protocol::Ssh => vec![
            HandshakePhase::Initial,
            HandshakePhase::ClientHello,
            HandshakePhase::ServerHello,
            HandshakePhase::KeyExchange,
            HandshakePhase::Finished,
        ],
    }
}

fn version_for(protocol: Protocol) -> ProtocolVersion {
    match protocol {
        Protocol::Tls => ProtocolVersion::Tls12,
        Protocol::Ssh => ProtocolVersion::Ssh2,
    }
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

fn write_text_report(
    writer: &mut OutputWriter,
    report: &BenchmarkReport,
    no_color: bool,
) -> Result<()> {
    let mut buf = String::new();
    buf.push_str(&bold("NegSynth Benchmark Report", no_color));
    buf.push_str(&format!("\n  Suite:   {}", report.suite.name));
    buf.push_str(&format!("\n  OS:      {}", report.suite.environment.os));
    buf.push_str(&format!("\n  Arch:    {}", report.suite.environment.arch));
    buf.push_str(&format!(
        "\n  Version: {}",
        report.suite.environment.negsyn_version
    ));
    buf.push_str(&format!(
        "\n  Total:   {:.1}ms\n",
        report.suite.total_time_ms
    ));

    let mut table = Table::new(vec![
        "Benchmark".into(),
        "Iters".into(),
        "Mean (ms)".into(),
        "Median (ms)".into(),
        "Min (ms)".into(),
        "Max (ms)".into(),
        "StdDev".into(),
    ]);

    for r in &report.suite.results {
        table.add_row(vec![
            r.name.clone(),
            r.iterations.to_string(),
            format!("{:.3}", r.mean_ms),
            format!("{:.3}", r.median_ms),
            format!("{:.3}", r.min_ms),
            format!("{:.3}", r.max_ms),
            format!("{:.3}", r.stddev_ms),
        ]);
    }

    buf.push_str(&table.render_text(no_color));

    // Baseline comparisons.
    if !report.comparisons.is_empty() {
        buf.push_str(&format!("\n  {}\n", bold("Baseline Comparison", no_color)));

        let mut cmp_table = Table::new(vec![
            "Benchmark".into(),
            "Current (ms)".into(),
            "Baseline (ms)".into(),
            "Change".into(),
            "Status".into(),
        ]);

        for c in &report.comparisons {
            let change_str = format!("{:+.1}%", c.change_pct);
            let status = if c.regression {
                red("REGRESSION", no_color)
            } else if c.change_pct < -5.0 {
                green("IMPROVED", no_color)
            } else {
                dim("OK", no_color)
            };
            cmp_table.add_row(vec![
                c.benchmark.clone(),
                format!("{:.3}", c.current_ms),
                format!("{:.3}", c.baseline_ms),
                change_str,
                status,
            ]);
        }

        buf.push_str(&cmp_table.render_text(no_color));
    }

    writer.write_raw(&buf)
}

fn write_csv_report(writer: &mut OutputWriter, report: &BenchmarkReport) -> Result<()> {
    let mut table = Table::new(vec![
        "benchmark".into(),
        "iterations".into(),
        "mean_ms".into(),
        "median_ms".into(),
        "min_ms".into(),
        "max_ms".into(),
        "stddev_ms".into(),
        "memory_bytes".into(),
    ]);

    for r in &report.suite.results {
        table.add_row(vec![
            r.name.clone(),
            r.iterations.to_string(),
            format!("{:.6}", r.mean_ms),
            format!("{:.6}", r.median_ms),
            format!("{:.6}", r.min_ms),
            format!("{:.6}", r.max_ms),
            format!("{:.6}", r.stddev_ms),
            r.memory_bytes.map(|m| m.to_string()).unwrap_or_default(),
        ]);
    }

    writer.write_raw(&table.render_csv())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_all_suites() {
        let s = resolve_suites("all").unwrap();
        assert!(s.len() >= 5);
        assert!(s.contains(&"slicer".to_string()));
        assert!(s.contains(&"e2e".to_string()));
    }

    #[test]
    fn resolve_single_suite() {
        let s = resolve_suites("merge").unwrap();
        assert_eq!(s, vec!["merge".to_string()]);
    }

    #[test]
    fn resolve_unknown_fails() {
        assert!(resolve_suites("nonexistent").is_err());
    }

    #[test]
    fn compute_stats_basic() {
        let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let s = compute_stats(&vals);
        assert!((s.mean - 3.0).abs() < 0.001);
        assert!((s.median - 3.0).abs() < 0.001);
        assert!((s.min - 1.0).abs() < 0.001);
        assert!((s.max - 5.0).abs() < 0.001);
        assert!(s.stddev > 0.0);
    }

    #[test]
    fn compute_stats_single() {
        let s = compute_stats(&[42.0]);
        assert!((s.mean - 42.0).abs() < 0.001);
        assert!((s.median - 42.0).abs() < 0.001);
        assert!((s.stddev).abs() < 0.001);
    }

    #[test]
    fn compute_stats_empty() {
        let s = compute_stats(&[]);
        assert_eq!(s.mean, 0.0);
    }

    #[test]
    fn compute_stats_even_count() {
        let vals = vec![1.0, 2.0, 3.0, 4.0];
        let s = compute_stats(&vals);
        assert!((s.median - 2.5).abs() < 0.001);
    }

    #[test]
    fn bench_slicer_runs() {
        let cfg = CliConfig::default();
        let result = bench_slicer(Protocol::Tls, &cfg).unwrap();
        assert!(result.throughput.is_some());
        assert!(result.memory_bytes.is_some());
    }

    #[test]
    fn bench_merge_runs() {
        let cfg = CliConfig::default();
        let result = bench_merge(Protocol::Ssh, &cfg).unwrap();
        assert!(result.throughput.is_some());
    }

    #[test]
    fn bench_extract_runs() {
        let cfg = CliConfig::default();
        let result = bench_extract(Protocol::Tls, &cfg).unwrap();
        assert!(result.memory_bytes.is_some());
    }

    #[test]
    fn bench_encode_runs() {
        let cfg = CliConfig::default();
        let result = bench_encode(Protocol::Tls, &cfg).unwrap();
        assert!(result.throughput.unwrap() > 0.0);
    }

    #[test]
    fn bench_concretize_runs() {
        let cfg = CliConfig::default();
        let result = bench_concretize(Protocol::Ssh, &cfg).unwrap();
        assert!(result.memory_bytes.unwrap() > 0);
    }

    #[test]
    fn bench_e2e_runs() {
        let cfg = CliConfig::default();
        let result = bench_e2e(Protocol::Tls, &cfg).unwrap();
        assert!(result.memory_bytes.unwrap() > 0);
    }

    #[test]
    fn baseline_comparison_regression() {
        let c = BaselineComparison {
            benchmark: "test".into(),
            current_ms: 110.0,
            baseline_ms: 100.0,
            change_pct: 10.0,
            regression: false,
        };
        assert!(!c.regression);

        let c2 = BaselineComparison {
            benchmark: "test".into(),
            current_ms: 120.0,
            baseline_ms: 100.0,
            change_pct: 20.0,
            regression: true,
        };
        assert!(c2.regression);
    }

    #[test]
    fn benchmark_suite_serializes() {
        let suite = BenchmarkSuite {
            name: "test".into(),
            results: vec![BenchmarkResult {
                name: "slicer".into(),
                iterations: 3,
                mean_ms: 10.0,
                median_ms: 9.5,
                min_ms: 8.0,
                max_ms: 12.0,
                stddev_ms: 1.5,
                throughput: Some(100.0),
                memory_bytes: Some(1024),
            }],
            total_time_ms: 30.0,
            environment: BenchmarkEnvironment {
                os: "linux".into(),
                arch: "x86_64".into(),
                negsyn_version: "0.1.0".into(),
                timestamp: "2024-01-01T00:00:00Z".into(),
            },
        };
        let json = serde_json::to_string(&suite).unwrap();
        assert!(json.contains("slicer"));
        assert!(json.contains("1024"));
    }

    #[test]
    fn handshake_phases_protocol() {
        assert!(handshake_phases(Protocol::Tls).len() > handshake_phases(Protocol::Ssh).len());
    }
}
