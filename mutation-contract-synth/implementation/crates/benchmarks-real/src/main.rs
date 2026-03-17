//! Real benchmarks for MutSpec: mutation-guided contract synthesis via
//! discrimination lattice walk.
//!
//! Tests 8 method specifications against known ground-truth contracts.
//! Measures precision, recall, F1, mutations generated, lattice walk steps,
//! and wall-clock time.  Compares the lattice-walk algorithm against two
//! baselines: (1) random mutation sampling and (2) spec mining from tests.

// We inline a minimal version of the lattice + lattice-walk code here because
// the contract-synth crate has an unresolvable dependency on the broken
// program-analysis crate.  All logic is faithfully ported from
// crates/contract-synth/src/{lattice.rs, lattice_walk.rs}.

mod baselines;
mod lattice;
mod methods;
mod metrics;

use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::baselines::{random_mutation_baseline, spec_mining_baseline};
use crate::lattice::{DiscriminationLattice, LatticeWalkSynthesizer, WalkConfig};
use crate::methods::{build_method_specs, MethodSpec};
use crate::metrics::{evaluate_contract, BenchmarkResult, ContractMetrics, SuiteResult};

fn main() {
    let specs = build_method_specs();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║  MutSpec Real Benchmark: Mutation-Guided Contract Synthesis     ║");
    println!("║  via Discrimination Lattice Walk                                ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();
    println!("Methods under test: {}", specs.len());
    println!();

    let mut results: Vec<BenchmarkResult> = Vec::new();

    for spec in &specs {
        println!("━━━ {} ━━━", spec.name);
        println!("  Signature: {}", spec.signature);
        println!(
            "  Ground-truth postconditions: {}",
            spec.ground_truth_post.len()
        );
        println!(
            "  Ground-truth preconditions:  {}",
            spec.ground_truth_pre.len()
        );
        println!("  Mutation operators: {:?}", spec.mutation_ops);
        println!();

        // --- Lattice Walk (our algorithm) ---
        let lw_start = Instant::now();
        let (lw_contract, lw_stats) = run_lattice_walk(spec);
        let lw_time = lw_start.elapsed();
        let lw_metrics = evaluate_contract(
            &lw_contract,
            &spec.ground_truth_pre,
            &spec.ground_truth_post,
        );

        // --- Baseline 1: Random Mutation Sampling ---
        let rm_start = Instant::now();
        let (rm_contract, rm_stats) = random_mutation_baseline(spec);
        let rm_time = rm_start.elapsed();
        let rm_metrics = evaluate_contract(
            &rm_contract,
            &spec.ground_truth_pre,
            &spec.ground_truth_post,
        );

        // --- Baseline 2: Spec Mining from Tests ---
        let sm_start = Instant::now();
        let (sm_contract, sm_stats) = spec_mining_baseline(spec);
        let sm_time = sm_start.elapsed();
        let sm_metrics = evaluate_contract(
            &sm_contract,
            &spec.ground_truth_pre,
            &spec.ground_truth_post,
        );

        println!("  ┌─────────────────────┬────────────┬────────────┬────────────┐");
        println!("  │ Metric              │ LatticeWlk │ RandomMut  │ SpecMine   │");
        println!("  ├─────────────────────┼────────────┼────────────┼────────────┤");
        println!(
            "  │ Precision           │ {:>10.3} │ {:>10.3} │ {:>10.3} │",
            lw_metrics.precision, rm_metrics.precision, sm_metrics.precision
        );
        println!(
            "  │ Recall              │ {:>10.3} │ {:>10.3} │ {:>10.3} │",
            lw_metrics.recall, rm_metrics.recall, sm_metrics.recall
        );
        println!(
            "  │ F1                  │ {:>10.3} │ {:>10.3} │ {:>10.3} │",
            lw_metrics.f1, rm_metrics.f1, sm_metrics.f1
        );
        println!(
            "  │ Clauses synth.      │ {:>10} │ {:>10} │ {:>10} │",
            lw_metrics.clauses_synthesized,
            rm_metrics.clauses_synthesized,
            sm_metrics.clauses_synthesized
        );
        println!(
            "  │ Mutations used      │ {:>10} │ {:>10} │ {:>10} │",
            lw_stats.mutations_used, rm_stats.mutations_used, sm_stats.mutations_used
        );
        println!(
            "  │ Lattice steps       │ {:>10} │ {:>10} │ {:>10} │",
            lw_stats.lattice_steps, rm_stats.lattice_steps, sm_stats.lattice_steps
        );
        println!(
            "  │ Runtime (µs)        │ {:>10} │ {:>10} │ {:>10} │",
            lw_time.as_micros(),
            rm_time.as_micros(),
            sm_time.as_micros()
        );
        println!("  └─────────────────────┴────────────┴────────────┴────────────┘");
        println!();

        results.push(BenchmarkResult {
            method: spec.name.clone(),
            signature: spec.signature.clone(),
            ground_truth_pre: spec.ground_truth_pre.len(),
            ground_truth_post: spec.ground_truth_post.len(),
            lattice_walk: lw_metrics.clone(),
            lattice_walk_time_us: lw_time.as_micros() as u64,
            lattice_walk_stats: lw_stats,
            random_mutation: rm_metrics.clone(),
            random_mutation_time_us: rm_time.as_micros() as u64,
            random_mutation_stats: rm_stats,
            spec_mining: sm_metrics.clone(),
            spec_mining_time_us: sm_time.as_micros() as u64,
            spec_mining_stats: sm_stats,
        });
    }

    // --- Aggregate ---
    let suite = compute_suite_result(&results);
    print_summary(&suite);

    // --- Write JSON ---
    let json = serde_json::to_string_pretty(&suite).expect("JSON serialization");
    let out_path = "benchmark_results.json";
    std::fs::write(out_path, &json).expect("write JSON");
    println!("Results written to {}", out_path);
}

fn run_lattice_walk(spec: &MethodSpec) -> (Vec<SynthesizedClause>, AlgorithmStats) {
    let mut disc = DiscriminationLattice::new();

    // Register error predicates from mutations
    for (mid, error_pred) in &spec.error_predicates {
        disc.register_mutant(mid.clone(), error_pred.clone());
    }

    let config = WalkConfig::default();
    let mut synth = LatticeWalkSynthesizer::new(config);
    let contract = synth.synthesize(&mut disc, &spec.name);

    let stats = synth.statistics().cloned().unwrap_or_default();
    let clauses: Vec<SynthesizedClause> = contract_to_clauses(&contract);

    let algo_stats = AlgorithmStats {
        mutations_used: stats.accepted_steps as usize,
        lattice_steps: stats.total_steps as usize,
        entailment_checks: stats.entailment_checks as usize,
    };

    (clauses, algo_stats)
}

fn contract_to_clauses(contract: &shared_types::Contract) -> Vec<SynthesizedClause> {
    let mut out = Vec::new();
    for f in contract.preconditions() {
        out.push(SynthesizedClause {
            kind: ClauseKind::Pre,
            formula_str: format!("{}", f),
        });
    }
    for f in contract.postconditions() {
        out.push(SynthesizedClause {
            kind: ClauseKind::Post,
            formula_str: format!("{}", f),
        });
    }
    out
}

fn compute_suite_result(results: &[BenchmarkResult]) -> SuiteResult {
    let n = results.len() as f64;
    let avg = |f: fn(&BenchmarkResult) -> f64| results.iter().map(f).sum::<f64>() / n;
    let avg_u =
        |f: fn(&BenchmarkResult) -> u64| (results.iter().map(f).sum::<u64>() as f64 / n) as u64;

    SuiteResult {
        methods_tested: results.len(),
        lattice_walk_avg_precision: avg(|r| r.lattice_walk.precision),
        lattice_walk_avg_recall: avg(|r| r.lattice_walk.recall),
        lattice_walk_avg_f1: avg(|r| r.lattice_walk.f1),
        lattice_walk_avg_time_us: avg_u(|r| r.lattice_walk_time_us),
        random_mutation_avg_precision: avg(|r| r.random_mutation.precision),
        random_mutation_avg_recall: avg(|r| r.random_mutation.recall),
        random_mutation_avg_f1: avg(|r| r.random_mutation.f1),
        random_mutation_avg_time_us: avg_u(|r| r.random_mutation_time_us),
        spec_mining_avg_precision: avg(|r| r.spec_mining.precision),
        spec_mining_avg_recall: avg(|r| r.spec_mining.recall),
        spec_mining_avg_f1: avg(|r| r.spec_mining.f1),
        spec_mining_avg_time_us: avg_u(|r| r.spec_mining_time_us),
        per_method: results.to_vec(),
    }
}

fn print_summary(suite: &SuiteResult) {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!(
        "║  AGGREGATE RESULTS ({} methods)                              ║",
        suite.methods_tested
    );
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║                        LatticeWlk   RandomMut   SpecMine       ║");
    println!(
        "║  Avg Precision     {:>10.3}   {:>10.3}  {:>10.3}       ║",
        suite.lattice_walk_avg_precision,
        suite.random_mutation_avg_precision,
        suite.spec_mining_avg_precision
    );
    println!(
        "║  Avg Recall        {:>10.3}   {:>10.3}  {:>10.3}       ║",
        suite.lattice_walk_avg_recall,
        suite.random_mutation_avg_recall,
        suite.spec_mining_avg_recall
    );
    println!(
        "║  Avg F1            {:>10.3}   {:>10.3}  {:>10.3}       ║",
        suite.lattice_walk_avg_f1, suite.random_mutation_avg_f1, suite.spec_mining_avg_f1
    );
    println!(
        "║  Avg Time (µs)     {:>10}   {:>10}  {:>10}       ║",
        suite.lattice_walk_avg_time_us,
        suite.random_mutation_avg_time_us,
        suite.spec_mining_avg_time_us
    );
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();
}

// --- Small shared types for the benchmark ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesizedClause {
    pub kind: ClauseKind,
    pub formula_str: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClauseKind {
    Pre,
    Post,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AlgorithmStats {
    pub mutations_used: usize,
    pub lattice_steps: usize,
    pub entailment_checks: usize,
}
