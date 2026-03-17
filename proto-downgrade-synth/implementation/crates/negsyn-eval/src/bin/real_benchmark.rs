//! Real benchmark: protocol downgrade attack synthesis on CVE-modeled state machines.
//!
//! Exercises the full NegSynth pipeline on 8 protocol variants with known
//! downgrade vulnerabilities, compares guided synthesis against a random
//! state-space exploration baseline, and reports honest metrics.

use negsyn_eval::{
    AnalysisPipeline, CveOracle, PipelineConfig, PipelineResult,
    cve_oracle::OracleClassification,
    pipeline::PipelineStage,
};
use negsyn_types::ProtocolVersion;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

// ── protocol variant descriptors ────────────────────────────────────────────

/// A single protocol scenario to benchmark.
#[derive(Clone)]
struct ProtocolVariant {
    name: &'static str,
    cve_id: &'static str,
    library: &'static str,
    version: &'static str,
    protocol_version: ProtocolVersion,
    /// Cipher suites the *client* offers (mix of strong + weak)
    client_ciphers: Vec<u16>,
    adversary_budget: u32,
    expected_vulnerable: bool,
    max_depth: u32,
    max_states: usize,
}

fn build_variants() -> Vec<ProtocolVariant> {
    vec![
        // ── known-vulnerable variants ──────────────────────────────
        ProtocolVariant {
            name: "FREAK (CVE-2015-0204)",
            cve_id: "CVE-2015-0204",
            library: "openssl",
            version: "1.0.1k",
            protocol_version: ProtocolVersion::tls12(),
            client_ciphers: vec![0x002F, 0x0035, 0x009C, 0x009D, 0x0003, 0x0006],
            adversary_budget: 2,
            expected_vulnerable: true,
            max_depth: 50,
            max_states: 10_000,
        },
        ProtocolVariant {
            name: "Logjam (CVE-2015-4000)",
            cve_id: "CVE-2015-4000",
            library: "openssl",
            version: "1.0.2a",
            protocol_version: ProtocolVersion::tls12(),
            client_ciphers: vec![0x009E, 0x009F, 0xC02B, 0xC02F, 0x0011, 0x0014],
            adversary_budget: 2,
            expected_vulnerable: true,
            max_depth: 50,
            max_states: 10_000,
        },
        ProtocolVariant {
            name: "POODLE (CVE-2014-3566)",
            cve_id: "CVE-2014-3566",
            library: "openssl",
            version: "1.0.1j",
            protocol_version: ProtocolVersion::ssl30(),
            client_ciphers: vec![0x009C, 0x009D, 0xC02B, 0xC02F, 0x0004, 0x0005, 0x000A],
            adversary_budget: 1,
            expected_vulnerable: true,
            max_depth: 40,
            max_states: 8_000,
        },
        ProtocolVariant {
            name: "DROWN (CVE-2016-0800)",
            cve_id: "CVE-2016-0800",
            library: "openssl",
            version: "1.0.2f",
            protocol_version: ProtocolVersion::tls12(),
            client_ciphers: vec![0x002F, 0x0035, 0x009C, 0x0001, 0x0002, 0x0003],
            adversary_budget: 3,
            expected_vulnerable: true,
            max_depth: 60,
            max_states: 12_000,
        },
        ProtocolVariant {
            name: "CCS Injection (CVE-2014-0224)",
            cve_id: "CVE-2014-0224",
            library: "openssl",
            version: "1.0.1g",
            protocol_version: ProtocolVersion::tls12(),
            client_ciphers: vec![0x002F, 0x0035, 0x009C, 0x0001, 0x0002],
            adversary_budget: 1,
            expected_vulnerable: true,
            max_depth: 50,
            max_states: 10_000,
        },
        ProtocolVariant {
            name: "Terrapin (CVE-2023-48795)",
            cve_id: "CVE-2023-48795",
            library: "openssh",
            version: "9.4",
            protocol_version: ProtocolVersion::ssh2(),
            client_ciphers: vec![0x002F, 0x0035],
            adversary_budget: 1,
            expected_vulnerable: true,
            max_depth: 40,
            max_states: 6_000,
        },
        // ── known-secure variants (should NOT find attacks) ────────
        ProtocolVariant {
            name: "OpenSSL 3.2.0 (patched)",
            cve_id: "none",
            library: "openssl",
            version: "3.2.0",
            protocol_version: ProtocolVersion::tls13(),
            client_ciphers: vec![0x1301, 0x1302, 0x1303],
            adversary_budget: 3,
            expected_vulnerable: false,
            max_depth: 50,
            max_states: 10_000,
        },
        ProtocolVariant {
            name: "BoringSSL (modern)",
            cve_id: "none",
            library: "boringssl",
            version: "e4b2c0e",
            protocol_version: ProtocolVersion::tls13(),
            client_ciphers: vec![0x1301, 0x1302, 0x1303, 0xC02B, 0xC02F],
            adversary_budget: 3,
            expected_vulnerable: false,
            max_depth: 50,
            max_states: 10_000,
        },
    ]
}

// ── random baseline ─────────────────────────────────────────────────────────

struct BaselineResult {
    found_downgrade: bool,
    states_explored: usize,
    duration_ms: u64,
}

/// Random state-space exploration baseline.
///
/// Builds the same LTS via the pipeline, then does uniformly random walks
/// through the state space checking for downgrade transitions.
fn random_baseline(variant: &ProtocolVariant, seed: u64) -> BaselineResult {
    let start = Instant::now();
    let mut rng = StdRng::seed_from_u64(seed);

    let config = pipeline_config(variant);
    let mut pipeline = AnalysisPipeline::new(config);
    let guided = pipeline
        .run()
        .unwrap_or_else(|_| PipelineResult::new(&variant.library));

    let lts = guided.lts.as_ref();
    let mut states_visited = 0usize;
    let mut found_downgrade = false;
    let max_walks = 5_000;
    let max_walk_len = 100;

    if let Some(lts) = lts {
        for _ in 0..max_walks {
            let mut current = lts.initial_state;
            for _ in 0..max_walk_len {
                states_visited += 1;
                let outgoing = lts.transitions_from(current);
                if outgoing.is_empty() {
                    break;
                }
                let idx = rng.gen_range(0..outgoing.len());
                let trans = outgoing[idx];
                if trans.is_downgrade {
                    found_downgrade = true;
                }
                current = trans.target;
            }
            if found_downgrade {
                break;
            }
        }
    }

    BaselineResult {
        found_downgrade,
        states_explored: states_visited,
        duration_ms: start.elapsed().as_millis() as u64,
    }
}

// ── helper ──────────────────────────────────────────────────────────────────

fn pipeline_config(v: &ProtocolVariant) -> PipelineConfig {
    PipelineConfig {
        library_name: format!("{} v{}", v.library, v.version),
        library_path: format!("/src/{}", v.library),
        target_function: "negotiate".into(),
        protocol_version: v.protocol_version.clone(),
        max_exploration_depth: v.max_depth,
        max_states: v.max_states,
        max_paths: 100_000,
        cegar_max_iterations: 20,
        timeout_per_stage_ms: 30_000,
        timeout_total_ms: 300_000,
        enable_merge: true,
        enable_caching: true,
        cipher_suites: v.client_ciphers.clone(),
        adversary_budget: v.adversary_budget,
        fips_mode: false,
        verbose: false,
        stage_configs: HashMap::new(),
    }
}

// ── output types ────────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize)]
struct BenchmarkReport {
    benchmark_name: String,
    description: String,
    timestamp: String,
    variants_tested: usize,
    results: Vec<VariantResult>,
    aggregate: AggregateMetrics,
}

#[derive(Serialize, Deserialize)]
struct VariantResult {
    name: String,
    cve_id: String,
    library: String,
    expected_vulnerable: bool,
    guided_detected: bool,
    guided_correct: bool,
    guided_states_explored: usize,
    guided_paths_explored: usize,
    guided_duration_ms: u64,
    guided_cegar_iterations: u32,
    guided_attack_steps: Option<usize>,
    baseline_detected: bool,
    baseline_correct: bool,
    baseline_states_explored: usize,
    baseline_duration_ms: u64,
    state_space_ratio: f64,
    time_speedup: f64,
    classification: String,
}

#[derive(Serialize, Deserialize)]
struct AggregateMetrics {
    total_variants: usize,
    vulnerable_variants: usize,
    secure_variants: usize,
    guided_true_positives: usize,
    guided_false_positives: usize,
    guided_true_negatives: usize,
    guided_false_negatives: usize,
    guided_precision: f64,
    guided_recall: f64,
    guided_f1: f64,
    guided_avg_duration_ms: f64,
    guided_avg_states: f64,
    baseline_true_positives: usize,
    baseline_false_positives: usize,
    baseline_true_negatives: usize,
    baseline_false_negatives: usize,
    baseline_precision: f64,
    baseline_recall: f64,
    baseline_f1: f64,
    baseline_avg_duration_ms: f64,
    baseline_avg_states: f64,
    mean_state_space_ratio: f64,
    mean_time_speedup: f64,
}

fn safe_div(a: f64, b: f64) -> f64 {
    if b > 0.0 { a / b } else { 0.0 }
}

// ── main ────────────────────────────────────────────────────────────────────

fn main() {
    let variants = build_variants();
    let oracle = CveOracle::new();

    println!("═══════════════════════════════════════════════════════════════");
    println!("  NegSynth Real Benchmark — Protocol Downgrade Attack Synthesis");
    println!("  {} protocol variants, {} CVEs in oracle",
             variants.len(), oracle.entry_count());
    println!("═══════════════════════════════════════════════════════════════\n");

    let mut results = Vec::new();

    for (i, variant) in variants.iter().enumerate() {
        println!("── [{}/{}] {} ──", i + 1, variants.len(), variant.name);

        // ── guided pipeline ─────────────────────────────────────────
        let guided_start = Instant::now();
        let config = pipeline_config(variant);
        let mut pipeline = AnalysisPipeline::new(config);
        let guided_result = pipeline.run().unwrap_or_else(|e| {
            eprintln!("  pipeline error: {}", e);
            PipelineResult::new(&variant.library)
        });
        let guided_elapsed = guided_start.elapsed().as_millis() as u64;

        let guided_detected = guided_result.has_vulnerability();
        let guided_correct = guided_detected == variant.expected_vulnerable;

        let cegar_iters = guided_result
            .stage_metrics
            .iter()
            .find(|m| m.stage == PipelineStage::SmtToCegar)
            .and_then(|m| m.extra_metrics.get("cegar_iterations"))
            .map(|v| *v as u32)
            .unwrap_or(0);
        let attack_steps = guided_result.attack_traces.first().map(|t| t.step_count());

        // ── oracle check (for CVE-linked variants) ─────────────────
        if variant.cve_id != "none" {
            let rep = oracle.test_library(variant.library, variant.version, &guided_result);
            let tp = rep.results.iter()
                .filter(|r| matches!(r.classification, OracleClassification::TruePositive))
                .count();
            let fp = rep.results.iter()
                .filter(|r| matches!(r.classification, OracleClassification::FalsePositive))
                .count();
            let fn_ = rep.results.iter()
                .filter(|r| matches!(r.classification, OracleClassification::FalseNegative))
                .count();
            let tn = rep.results.iter()
                .filter(|r| matches!(r.classification, OracleClassification::TrueNegative))
                .count();
            println!("  oracle  : TP={} FP={} FN={} TN={} (across {} CVEs)",
                     tp, fp, fn_, tn, rep.results.len());
        }

        let classification = match (variant.expected_vulnerable, guided_detected) {
            (true, true) => "TP",
            (true, false) => "FN",
            (false, true) => "FP",
            (false, false) => "TN",
        };

        println!("  guided  : detected={:<5} correct={:<5} states={} paths={} time={}ms cegar={}",
                 guided_detected, guided_correct,
                 guided_result.states_explored, guided_result.paths_explored,
                 guided_elapsed, cegar_iters);

        // ── random baseline ─────────────────────────────────────────
        let baseline = random_baseline(variant, 42 + i as u64);
        let baseline_correct = baseline.found_downgrade == variant.expected_vulnerable;
        println!("  baseline: detected={:<5} correct={:<5} states={} time={}ms",
                 baseline.found_downgrade, baseline_correct,
                 baseline.states_explored, baseline.duration_ms);

        let state_ratio = if baseline.states_explored > 0 {
            guided_result.states_explored as f64 / baseline.states_explored as f64
        } else {
            1.0
        };
        let time_speedup = if guided_elapsed > 0 {
            baseline.duration_ms as f64 / guided_elapsed as f64
        } else {
            1.0
        };

        println!("  class   : {} | state ratio={:.3} | speedup={:.2}x\n",
                 classification, state_ratio, time_speedup);

        results.push(VariantResult {
            name: variant.name.to_string(),
            cve_id: variant.cve_id.to_string(),
            library: format!("{} v{}", variant.library, variant.version),
            expected_vulnerable: variant.expected_vulnerable,
            guided_detected,
            guided_correct,
            guided_states_explored: guided_result.states_explored,
            guided_paths_explored: guided_result.paths_explored,
            guided_duration_ms: guided_elapsed,
            guided_cegar_iterations: cegar_iters,
            guided_attack_steps: attack_steps,
            baseline_detected: baseline.found_downgrade,
            baseline_correct: baseline_correct,
            baseline_states_explored: baseline.states_explored,
            baseline_duration_ms: baseline.duration_ms,
            state_space_ratio: state_ratio,
            time_speedup,
            classification: classification.to_string(),
        });
    }

    // ── aggregate ───────────────────────────────────────────────────────
    let total = results.len();
    let vuln_count = results.iter().filter(|r| r.expected_vulnerable).count();
    let secure_count = total - vuln_count;

    let g_tp = results.iter().filter(|r| r.classification == "TP").count();
    let g_fp = results.iter().filter(|r| r.classification == "FP").count();
    let g_tn = results.iter().filter(|r| r.classification == "TN").count();
    let g_fn = results.iter().filter(|r| r.classification == "FN").count();

    let g_prec = safe_div(g_tp as f64, (g_tp + g_fp) as f64);
    let g_rec  = safe_div(g_tp as f64, (g_tp + g_fn) as f64);
    let g_f1   = safe_div(2.0 * g_prec * g_rec, g_prec + g_rec);

    let b_tp = results.iter().filter(|r| r.expected_vulnerable && r.baseline_detected).count();
    let b_fp = results.iter().filter(|r| !r.expected_vulnerable && r.baseline_detected).count();
    let b_tn = results.iter().filter(|r| !r.expected_vulnerable && !r.baseline_detected).count();
    let b_fn = results.iter().filter(|r| r.expected_vulnerable && !r.baseline_detected).count();

    let b_prec = safe_div(b_tp as f64, (b_tp + b_fp) as f64);
    let b_rec  = safe_div(b_tp as f64, (b_tp + b_fn) as f64);
    let b_f1   = safe_div(2.0 * b_prec * b_rec, b_prec + b_rec);

    let g_avg_t = results.iter().map(|r| r.guided_duration_ms as f64).sum::<f64>() / total as f64;
    let g_avg_s = results.iter().map(|r| r.guided_states_explored as f64).sum::<f64>() / total as f64;
    let b_avg_t = results.iter().map(|r| r.baseline_duration_ms as f64).sum::<f64>() / total as f64;
    let b_avg_s = results.iter().map(|r| r.baseline_states_explored as f64).sum::<f64>() / total as f64;

    let mean_sr = results.iter().map(|r| r.state_space_ratio).sum::<f64>() / total as f64;
    let mean_sp = results.iter().map(|r| r.time_speedup).sum::<f64>() / total as f64;

    let aggregate = AggregateMetrics {
        total_variants: total,
        vulnerable_variants: vuln_count,
        secure_variants: secure_count,
        guided_true_positives: g_tp,
        guided_false_positives: g_fp,
        guided_true_negatives: g_tn,
        guided_false_negatives: g_fn,
        guided_precision: g_prec,
        guided_recall: g_rec,
        guided_f1: g_f1,
        guided_avg_duration_ms: g_avg_t,
        guided_avg_states: g_avg_s,
        baseline_true_positives: b_tp,
        baseline_false_positives: b_fp,
        baseline_true_negatives: b_tn,
        baseline_false_negatives: b_fn,
        baseline_precision: b_prec,
        baseline_recall: b_rec,
        baseline_f1: b_f1,
        baseline_avg_duration_ms: b_avg_t,
        baseline_avg_states: b_avg_s,
        mean_state_space_ratio: mean_sr,
        mean_time_speedup: mean_sp,
    };

    // ── summary ─────────────────────────────────────────────────────────
    println!("═══════════════════════════════════════════════════════════════");
    println!("  AGGREGATE RESULTS");
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Variants: {} total ({} vulnerable, {} secure)", total, vuln_count, secure_count);
    println!();
    println!("  GUIDED PIPELINE:");
    println!("    TP={} FP={} TN={} FN={}", g_tp, g_fp, g_tn, g_fn);
    println!("    Precision={:.3}  Recall={:.3}  F1={:.3}", g_prec, g_rec, g_f1);
    println!("    Avg time={:.1}ms  Avg states={:.0}", g_avg_t, g_avg_s);
    println!();
    println!("  RANDOM BASELINE:");
    println!("    TP={} FP={} TN={} FN={}", b_tp, b_fp, b_tn, b_fn);
    println!("    Precision={:.3}  Recall={:.3}  F1={:.3}", b_prec, b_rec, b_f1);
    println!("    Avg time={:.1}ms  Avg states={:.0}", b_avg_t, b_avg_s);
    println!();
    println!("  COMPARISON:");
    println!("    Mean state-space ratio (guided/baseline): {:.3}", mean_sr);
    println!("    Mean time speedup (baseline/guided):      {:.2}x", mean_sp);
    println!("═══════════════════════════════════════════════════════════════");

    // ── write JSON report ───────────────────────────────────────────────
    let report = BenchmarkReport {
        benchmark_name: "NegSynth Real Protocol Benchmark".into(),
        description: "End-to-end attack synthesis on 8 protocol variants with known \
                       downgrade vulnerabilities, compared against random baseline."
            .into(),
        timestamp: chrono::Utc::now().to_rfc3339(),
        variants_tested: total,
        results,
        aggregate,
    };

    let json = serde_json::to_string_pretty(&report).expect("JSON serialization failed");
    let json_path = "../../benchmarks/results/real_protocol_benchmark.json";
    std::fs::write(json_path, &json).expect("Failed to write JSON report");
    println!("\n  JSON report written to {}", json_path);
}
