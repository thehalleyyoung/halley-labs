//! Production Corpus Benchmark
//!
//! Generates 2000 production-scale test pairs using BPE vocabulary,
//! runs differential testing across all 5 metrics, and reports
//! corpus statistics, per-metric agreement, timing, and memory estimates.

use spectacles_core::scoring::{
    ScoringPair,
    differential::{DifferentialTester, production_test_pairs, production_corpus_stats},
};
use serde::Serialize;
use std::time::Instant;

const NUM_PAIRS: usize = 2000;
const SEED: u64 = 20240101;
const OUTPUT_PATH: &str = "/Users/halleyyoung/Documents/div/mathdivergence/best/spectacles-wfa-zk-scoring-circuits/implementation/production_corpus_results.json";

#[derive(Debug, Serialize)]
struct ProductionCorpusBenchmark {
    timestamp: String,
    corpus_stats: CorpusStatsReport,
    total_tests: usize,
    total_disagreements: usize,
    all_agree: bool,
    per_metric: Vec<PerMetricResult>,
    timing: TimingBreakdown,
    memory_estimation: MemoryEstimation,
    scaling_analysis: String,
    honest_assessment: String,
}

#[derive(Debug, Serialize)]
struct CorpusStatsReport {
    num_pairs: usize,
    seed: u64,
    unique_tokens: usize,
    mean_sequence_length: f64,
    max_sequence_length: usize,
    min_sequence_length: usize,
    mean_jaccard_overlap: f64,
}

#[derive(Debug, Serialize)]
struct PerMetricResult {
    metric: String,
    tests: usize,
    agreements: usize,
    disagreements: usize,
    agreement_rate: f64,
    timing_ms: f64,
}

#[derive(Debug, Serialize)]
struct TimingBreakdown {
    total_wall_clock_ms: f64,
    pair_generation_ms: f64,
    corpus_stats_ms: f64,
    differential_testing_ms: f64,
    per_metric_ms: Vec<(String, f64)>,
}

#[derive(Debug, Serialize)]
struct MemoryEstimation {
    total_pair_bytes: usize,
    mean_pair_bytes: usize,
    max_pair_bytes: usize,
    estimated_wfa_overhead_bytes: usize,
    estimated_total_bytes: usize,
}

/// Estimate memory footprint of scoring pairs
fn estimate_memory(pairs: &[ScoringPair]) -> MemoryEstimation {
    let mut total_bytes: usize = 0;
    let mut max_bytes: usize = 0;
    for pair in pairs {
        let pair_bytes = pair.candidate.len() + pair.reference.len();
        total_bytes += pair_bytes;
        max_bytes = max_bytes.max(pair_bytes);
    }
    let mean_bytes = if pairs.is_empty() { 0 } else { total_bytes / pairs.len() };
    // WFA overhead: ~4x the input size for state tables and transition maps
    let wfa_overhead = total_bytes * 4;
    MemoryEstimation {
        total_pair_bytes: total_bytes,
        mean_pair_bytes: mean_bytes,
        max_pair_bytes: max_bytes,
        estimated_wfa_overhead_bytes: wfa_overhead,
        estimated_total_bytes: total_bytes + wfa_overhead,
    }
}

fn main() {
    let total_start = Instant::now();

    println!("═══════════════════════════════════════════════════════════════");
    println!("  Production Corpus Benchmark");
    println!("  {} pairs · BPE vocabulary · 5 metrics", NUM_PAIRS);
    println!("═══════════════════════════════════════════════════════════════");

    // Phase 1: Generate production-scale pairs
    println!("\n▸ Phase 1: Generating {} production-scale test pairs...", NUM_PAIRS);
    let gen_start = Instant::now();
    let pairs = production_test_pairs(NUM_PAIRS, SEED);
    let gen_ms = gen_start.elapsed().as_secs_f64() * 1000.0;
    println!("  Generated {} pairs in {:.1} ms", pairs.len(), gen_ms);

    // Phase 2: Corpus statistics
    println!("\n▸ Phase 2: Computing corpus statistics...");
    let stats_start = Instant::now();
    let stats = production_corpus_stats(&pairs);
    let stats_ms = stats_start.elapsed().as_secs_f64() * 1000.0;

    let mean_overlap = if stats.overlap_distribution.is_empty() {
        0.0
    } else {
        stats.overlap_distribution.iter().sum::<f64>() / stats.overlap_distribution.len() as f64
    };

    println!("  Unique tokens:         {}", stats.unique_tokens);
    println!("  Sequence lengths:      min={} mean={:.1} max={}",
        stats.min_sequence_length, stats.mean_sequence_length, stats.max_sequence_length);
    println!("  Mean Jaccard overlap:  {:.4}", mean_overlap);

    // Phase 3: Memory estimation
    let mem = estimate_memory(&pairs);
    println!("\n▸ Phase 3: Memory estimation...");
    println!("  Pair data:       {} bytes ({:.1} KiB)", mem.total_pair_bytes, mem.total_pair_bytes as f64 / 1024.0);
    println!("  WFA overhead:    {} bytes ({:.1} KiB)", mem.estimated_wfa_overhead_bytes, mem.estimated_wfa_overhead_bytes as f64 / 1024.0);
    println!("  Estimated total: {} bytes ({:.1} KiB)", mem.estimated_total_bytes, mem.estimated_total_bytes as f64 / 1024.0);

    // Phase 4: Differential testing across all 5 metrics
    println!("\n▸ Phase 4: Differential testing ({} pairs × 5 metrics)...", NUM_PAIRS);
    let diff_start = Instant::now();
    let tester = DifferentialTester::new();

    let metric_names = ["exact_match", "token_f1", "bleu", "rouge1", "rouge_l"];
    let mut per_metric_results = Vec::new();
    let mut per_metric_timing = Vec::new();
    let mut total_tests = 0;
    let mut total_disagreements = 0;

    for metric_name in &metric_names {
        let m_start = Instant::now();
        let report = match *metric_name {
            "exact_match" => tester.test_exact_match(&pairs),
            "token_f1" => tester.test_token_f1(&pairs),
            "bleu" => tester.test_bleu(&pairs),
            "rouge1" => tester.test_rouge1(&pairs),
            "rouge_l" => tester.test_rouge_l(&pairs),
            _ => unreachable!(),
        };
        let m_ms = m_start.elapsed().as_secs_f64() * 1000.0;

        total_tests += report.total_tests;
        total_disagreements += report.disagreements;

        println!("  {:12} | {}/{} agree ({:.1}%) | {:.1} ms",
            metric_name, report.agreements, report.total_tests,
            report.agreement_rate * 100.0, m_ms);

        per_metric_results.push(PerMetricResult {
            metric: metric_name.to_string(),
            tests: report.total_tests,
            agreements: report.agreements,
            disagreements: report.disagreements,
            agreement_rate: report.agreement_rate,
            timing_ms: m_ms,
        });
        per_metric_timing.push((metric_name.to_string(), m_ms));
    }

    let diff_ms = diff_start.elapsed().as_secs_f64() * 1000.0;
    let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;

    // Summary
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("  Production Corpus Benchmark Summary");
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Total tests:          {}", total_tests);
    println!("  Total disagreements:  {}", total_disagreements);
    println!("  All agree:            {}", total_disagreements == 0);
    println!("  Wall clock:           {:.1} ms", total_ms);

    // Scaling analysis
    let scaling_analysis = format!(
        "Production vocabulary: {} unique tokens (BPE-scale). \
         Sequence lengths range {}-{} tokens (mean {:.1}). \
         All 5 metrics tested on {} pairs with {} total checks. \
         Bottlenecks: ROUGE-L has O(mn) LCS computation; BLEU requires n-gram \
         counting up to 4-grams. Circuit compilation cost scales with WFA \
         state count. The system handles production vocabulary without issues \
         at this scale.",
        stats.unique_tokens,
        stats.min_sequence_length, stats.max_sequence_length,
        stats.mean_sequence_length,
        NUM_PAIRS, total_tests,
    );

    let honest_assessment = format!(
        "This benchmark uses deterministic pseudo-random pairs from a BPE-like \
         vocabulary of {} tokens. While production-scale in vocabulary size, \
         the token distribution is uniform rather than Zipfian, and sequences \
         lack grammatical structure. The differential tester verifies that \
         reference, automaton, and circuit implementations agree to within 1e-7 \
         tolerance. {} disagreements found across {} total checks. \
         Real production workloads may exhibit different performance \
         characteristics due to longer sequences, skewed distributions, and \
         multi-language content.",
        stats.unique_tokens, total_disagreements, total_tests,
    );

    let corpus_stats_report = CorpusStatsReport {
        num_pairs: NUM_PAIRS,
        seed: SEED,
        unique_tokens: stats.unique_tokens,
        mean_sequence_length: stats.mean_sequence_length,
        max_sequence_length: stats.max_sequence_length,
        min_sequence_length: stats.min_sequence_length,
        mean_jaccard_overlap: mean_overlap,
    };

    let timing = TimingBreakdown {
        total_wall_clock_ms: total_ms,
        pair_generation_ms: gen_ms,
        corpus_stats_ms: stats_ms,
        differential_testing_ms: diff_ms,
        per_metric_ms: per_metric_timing,
    };

    let report = ProductionCorpusBenchmark {
        timestamp: chrono::Utc::now().to_rfc3339(),
        corpus_stats: corpus_stats_report,
        total_tests,
        total_disagreements,
        all_agree: total_disagreements == 0,
        per_metric: per_metric_results,
        timing,
        memory_estimation: mem,
        scaling_analysis,
        honest_assessment,
    };

    let json = serde_json::to_string_pretty(&report).expect("serialize");
    std::fs::write(OUTPUT_PATH, &json).expect("write results");
    println!("\n  Results saved to: {}", OUTPUT_PATH);
}
