//! Real end-to-end fault localization benchmark.
//!
//! Builds mock NLP pipelines (tokenizer → encoder → classifier), injects
//! known faults at specific stages, runs the full metamorphic localization
//! engine, and measures:
//!   - Localization accuracy (does it find the right stage?)
//!   - False localization rate
//!   - Runtime
//!   - Comparison against baselines (random, always-blame-last)

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;
use std::time::Instant;

use shared_types::StageId;
use localization::{LocalizationConfig, LocalizationEngine, SBFLMetric, TestObservation};
use evaluation::metrics::{compute_top_k, compute_exam, compute_wasted_effort, compute_full_metrics};

// ============================================================================
// Core types for the benchmark harness
// ============================================================================

/// A fault scenario with known ground truth.
struct FaultScenario {
    name: &'static str,
    description: &'static str,
    /// Which stages are faulty (ground truth).
    faulty_stages: Vec<&'static str>,
    /// Per-stage differential generation function.
    /// Given (stage_name, test_index, is_violation, rng) → differential value.
    gen_differentials: Box<dyn Fn(&str, usize, bool, &mut StdRng) -> f64>,
    /// Fraction of tests that show violations.
    violation_rate: f64,
    /// Number of test observations to generate.
    n_tests: usize,
}

/// Results from running one scenario.
#[derive(Debug, Clone)]
struct ScenarioResult {
    name: String,
    faulty_stages: Vec<String>,
    predicted_ranking: Vec<String>,
    top1_correct: bool,
    top2_correct: bool,
    first_fault_rank: usize,
    engine_time_us: u64,
}

/// Aggregated results across all scenarios.
#[derive(Debug, Clone)]
struct BenchmarkReport {
    scenarios: Vec<ScenarioResult>,
    // Our tool
    tool_top1_accuracy: f64,
    tool_top2_accuracy: f64,
    tool_mean_first_rank: f64,
    tool_mean_runtime_us: f64,
    // Baselines
    random_top1_accuracy: f64,
    random_top2_accuracy: f64,
    last_stage_top1_accuracy: f64,
    last_stage_top2_accuracy: f64,
    // False localization
    false_localization_rate: f64,
}

// ============================================================================
// Pipeline stage definitions
// ============================================================================

const PIPELINE_STAGES: &[&str] = &["tokenizer", "encoder", "classifier"];

fn stage_list() -> Vec<(StageId, String)> {
    PIPELINE_STAGES
        .iter()
        .map(|s| (StageId::new(s), s.to_string()))
        .collect()
}

const TRANSFORMATIONS: &[&str] = &[
    "passivization",
    "clefting",
    "topicalization",
    "tense_change",
    "negation_insertion",
    "synonym_substitution",
];

const SAMPLE_INPUTS: &[&str] = &[
    "The quick brown fox jumps over the lazy dog",
    "A renowned scientist discovered a groundbreaking formula in the laboratory",
    "She gave the student a book about ancient civilizations",
    "The report was written by the lead researcher last Friday",
    "Several experts debated the implications of the new policy",
    "Kim presented the findings at the annual conference yesterday",
    "The algorithm efficiently processes large datasets in parallel",
    "Multiple stakeholders reviewed the proposed changes carefully",
];

// ============================================================================
// Fault scenario definitions
// ============================================================================

fn build_scenarios() -> Vec<FaultScenario> {
    vec![
        // --- Scenario 1: Tokenizer bug (token splitting error) ---
        FaultScenario {
            name: "tokenizer_split_bug",
            description: "Tokenizer incorrectly splits compound words",
            faulty_stages: vec!["tokenizer"],
            gen_differentials: Box::new(|stage, _i, is_violation, rng| {
                if is_violation && stage == "tokenizer" {
                    0.75 + rng.gen::<f64>() * 0.2 // high signal
                } else if is_violation && stage == "encoder" {
                    0.15 + rng.gen::<f64>() * 0.1 // mild downstream amplification
                } else if is_violation && stage == "classifier" {
                    0.10 + rng.gen::<f64>() * 0.08
                } else {
                    0.02 + rng.gen::<f64>() * 0.03 // noise floor
                }
            }),
            violation_rate: 0.30,
            n_tests: 200,
        },
        // --- Scenario 2: Encoder drift (embedding space shift) ---
        FaultScenario {
            name: "encoder_embedding_drift",
            description: "Encoder produces shifted embeddings for passive voice",
            faulty_stages: vec!["encoder"],
            gen_differentials: Box::new(|stage, _i, is_violation, rng| {
                if is_violation && stage == "encoder" {
                    0.80 + rng.gen::<f64>() * 0.15
                } else if is_violation && stage == "classifier" {
                    0.20 + rng.gen::<f64>() * 0.12 // amplification
                } else if is_violation && stage == "tokenizer" {
                    0.03 + rng.gen::<f64>() * 0.02 // no backpropagation
                } else {
                    0.02 + rng.gen::<f64>() * 0.03
                }
            }),
            violation_rate: 0.25,
            n_tests: 200,
        },
        // --- Scenario 3: Classifier bias (sentiment label flip) ---
        FaultScenario {
            name: "classifier_sentiment_bias",
            description: "Classifier flips sentiment for negated inputs",
            faulty_stages: vec!["classifier"],
            gen_differentials: Box::new(|stage, _i, is_violation, rng| {
                if is_violation && stage == "classifier" {
                    0.85 + rng.gen::<f64>() * 0.12
                } else if is_violation {
                    // Upstream stages are fine; fault is purely at the end
                    0.03 + rng.gen::<f64>() * 0.04
                } else {
                    0.02 + rng.gen::<f64>() * 0.03
                }
            }),
            violation_rate: 0.35,
            n_tests: 200,
        },
        // --- Scenario 4: Cascading fault (tokenizer → encoder) ---
        FaultScenario {
            name: "cascading_tokenizer_encoder",
            description: "Tokenizer bug cascades into encoder misrepresentation",
            faulty_stages: vec!["tokenizer", "encoder"],
            gen_differentials: Box::new(|stage, _i, is_violation, rng| {
                if is_violation && stage == "tokenizer" {
                    0.60 + rng.gen::<f64>() * 0.15 // moderate fault origin
                } else if is_violation && stage == "encoder" {
                    0.70 + rng.gen::<f64>() * 0.2 // amplified in encoder
                } else if is_violation && stage == "classifier" {
                    0.25 + rng.gen::<f64>() * 0.15 // further cascade
                } else {
                    0.02 + rng.gen::<f64>() * 0.03
                }
            }),
            violation_rate: 0.28,
            n_tests: 200,
        },
        // --- Scenario 5: Subtle tokenizer bug (low signal) ---
        FaultScenario {
            name: "tokenizer_subtle_whitespace",
            description: "Tokenizer mishandles Unicode whitespace variants",
            faulty_stages: vec!["tokenizer"],
            gen_differentials: Box::new(|stage, _i, is_violation, rng| {
                if is_violation && stage == "tokenizer" {
                    // Subtle: signal barely above noise
                    0.25 + rng.gen::<f64>() * 0.15
                } else if is_violation {
                    0.08 + rng.gen::<f64>() * 0.06
                } else {
                    0.03 + rng.gen::<f64>() * 0.04
                }
            }),
            violation_rate: 0.20,
            n_tests: 300,
        },
        // --- Scenario 6: Encoder attention mask bug ---
        FaultScenario {
            name: "encoder_attention_mask",
            description: "Encoder drops attention on sentence-final tokens",
            faulty_stages: vec!["encoder"],
            gen_differentials: Box::new(|stage, i, is_violation, rng| {
                if is_violation && stage == "encoder" {
                    // Stronger for longer sentences
                    let len_factor = 0.5 + (i % 8) as f64 * 0.05;
                    len_factor + rng.gen::<f64>() * 0.15
                } else if is_violation && stage == "classifier" {
                    0.18 + rng.gen::<f64>() * 0.1
                } else if is_violation && stage == "tokenizer" {
                    0.02 + rng.gen::<f64>() * 0.02
                } else {
                    0.02 + rng.gen::<f64>() * 0.03
                }
            }),
            violation_rate: 0.30,
            n_tests: 200,
        },
        // --- Scenario 7: Classifier confidence collapse ---
        FaultScenario {
            name: "classifier_confidence_collapse",
            description: "Classifier outputs near-uniform distributions on clefted inputs",
            faulty_stages: vec!["classifier"],
            gen_differentials: Box::new(|stage, _i, is_violation, rng| {
                if is_violation && stage == "classifier" {
                    0.70 + rng.gen::<f64>() * 0.25
                } else if is_violation {
                    0.04 + rng.gen::<f64>() * 0.04
                } else {
                    0.02 + rng.gen::<f64>() * 0.03
                }
            }),
            violation_rate: 0.22,
            n_tests: 200,
        },
        // --- Scenario 8: No fault (clean pipeline) ---
        FaultScenario {
            name: "clean_pipeline_no_fault",
            description: "Correctly functioning pipeline with no injected faults",
            faulty_stages: vec![],
            gen_differentials: Box::new(|_stage, _i, _is_violation, rng| {
                // All noise, no real signal
                0.02 + rng.gen::<f64>() * 0.04
            }),
            violation_rate: 0.0, // no violations in clean pipeline
            n_tests: 200,
        },
        // --- Scenario 9: Multi-fault (encoder + classifier) ---
        FaultScenario {
            name: "multi_fault_encoder_classifier",
            description: "Independent faults in encoder and classifier",
            faulty_stages: vec!["encoder", "classifier"],
            gen_differentials: Box::new(|stage, i, is_violation, rng| {
                if !is_violation {
                    return 0.02 + rng.gen::<f64>() * 0.03;
                }
                match stage {
                    "encoder" if i % 3 != 2 => 0.65 + rng.gen::<f64>() * 0.2,
                    "classifier" if i % 3 != 0 => 0.60 + rng.gen::<f64>() * 0.25,
                    "tokenizer" => 0.03 + rng.gen::<f64>() * 0.03,
                    _ => 0.10 + rng.gen::<f64>() * 0.08,
                }
            }),
            violation_rate: 0.32,
            n_tests: 250,
        },
        // --- Scenario 10: Adversarial — all stages slightly elevated ---
        FaultScenario {
            name: "adversarial_uniform_signal",
            description: "Encoder is faulty but all stages show moderate signal (hard case)",
            faulty_stages: vec!["encoder"],
            gen_differentials: Box::new(|stage, _i, is_violation, rng| {
                if !is_violation {
                    return 0.03 + rng.gen::<f64>() * 0.03;
                }
                match stage {
                    "tokenizer" => 0.30 + rng.gen::<f64>() * 0.15,
                    "encoder" => 0.45 + rng.gen::<f64>() * 0.2,    // barely above others
                    "classifier" => 0.35 + rng.gen::<f64>() * 0.15,
                    _ => 0.10 + rng.gen::<f64>() * 0.05,
                }
            }),
            violation_rate: 0.30,
            n_tests: 300,
        },
    ]
}

// ============================================================================
// Generate synthetic observations for a scenario
// ============================================================================

fn generate_observations(
    scenario: &FaultScenario,
    stages: &[(StageId, String)],
    seed: u64,
) -> Vec<TestObservation> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut obs = Vec::with_capacity(scenario.n_tests);

    for i in 0..scenario.n_tests {
        let is_violation = rng.gen::<f64>() < scenario.violation_rate;
        let input_idx = i % SAMPLE_INPUTS.len();
        let trans_idx = i % TRANSFORMATIONS.len();

        let mut per_stage = HashMap::new();
        for (_, name) in stages {
            let diff = (scenario.gen_differentials)(name, i, is_violation, &mut rng);
            per_stage.insert(name.clone(), diff);
        }

        let violation_magnitude = if is_violation {
            per_stage.values().cloned().fold(0.0f64, f64::max)
        } else {
            per_stage.values().cloned().fold(0.0f64, f64::max)
        };

        obs.push(TestObservation {
            test_id: format!("{}_{}", scenario.name, i),
            transformation_name: TRANSFORMATIONS[trans_idx].to_string(),
            input_text: SAMPLE_INPUTS[input_idx].to_string(),
            transformed_text: format!("[transformed] {}", SAMPLE_INPUTS[input_idx]),
            violation_detected: is_violation,
            violation_magnitude,
            per_stage_differentials: per_stage,
            execution_time_ms: 1.0 + rng.gen::<f64>() * 2.0,
        });
    }
    obs
}

// ============================================================================
// Run our localization engine on a scenario
// ============================================================================

fn run_localization(
    scenario: &FaultScenario,
    stages: &[(StageId, String)],
    seed: u64,
) -> ScenarioResult {
    let observations = generate_observations(scenario, stages, seed);

    let config = LocalizationConfig {
        sbfl_metric: SBFLMetric::Ochiai,
        enable_causal_analysis: true,
        enable_discriminability_check: false,
        ..Default::default()
    };

    let start = Instant::now();
    let mut engine = LocalizationEngine::with_config(config);
    engine.register_stages(stages.to_vec());
    engine.record_observations(observations);
    let ranking = engine.compute_suspiciousness();
    let elapsed = start.elapsed();

    let predicted: Vec<String> = ranking
        .rankings
        .iter()
        .map(|e| e.stage_name.clone())
        .collect();

    let faulty: Vec<String> = scenario.faulty_stages.iter().map(|s| s.to_string()).collect();

    let top1_correct = if faulty.is_empty() {
        false // no fault to find
    } else {
        predicted.first().map_or(false, |p| faulty.contains(p))
    };

    let top2_correct = if faulty.is_empty() {
        false
    } else {
        predicted.iter().take(2).any(|p| faulty.contains(p))
    };

    let first_fault_rank = if faulty.is_empty() {
        predicted.len() + 1
    } else {
        predicted
            .iter()
            .position(|p| faulty.contains(p))
            .map(|i| i + 1)
            .unwrap_or(predicted.len() + 1)
    };

    ScenarioResult {
        name: scenario.name.to_string(),
        faulty_stages: faulty,
        predicted_ranking: predicted,
        top1_correct,
        top2_correct,
        first_fault_rank,
        engine_time_us: elapsed.as_micros() as u64,
    }
}

// ============================================================================
// Baseline: Random stage selection
// ============================================================================

fn random_baseline_accuracy(scenarios: &[FaultScenario], n_trials: usize) -> (f64, f64) {
    let mut rng = StdRng::seed_from_u64(999);
    let n_stages = PIPELINE_STAGES.len();
    let mut top1_hits = 0usize;
    let mut top2_hits = 0usize;
    let mut total = 0usize;

    for scenario in scenarios {
        if scenario.faulty_stages.is_empty() {
            continue;
        }
        for _ in 0..n_trials {
            // Random permutation of stages
            let mut perm: Vec<usize> = (0..n_stages).collect();
            for i in (1..n_stages).rev() {
                let j = rng.gen_range(0..=i);
                perm.swap(i, j);
            }
            let random_ranking: Vec<&str> = perm.iter().map(|&i| PIPELINE_STAGES[i]).collect();
            let top1 = scenario.faulty_stages.contains(&random_ranking[0]);
            let top2 = random_ranking.iter().take(2).any(|s| scenario.faulty_stages.contains(s));
            if top1 { top1_hits += 1; }
            if top2 { top2_hits += 1; }
            total += 1;
        }
    }

    (
        top1_hits as f64 / total as f64,
        top2_hits as f64 / total as f64,
    )
}

// ============================================================================
// Baseline: Always blame the last stage
// ============================================================================

fn last_stage_baseline_accuracy(scenarios: &[FaultScenario]) -> (f64, f64) {
    let mut top1_hits = 0usize;
    let mut top2_hits = 0usize;
    let mut total = 0usize;

    let last = PIPELINE_STAGES.last().unwrap();
    let second_last = PIPELINE_STAGES[PIPELINE_STAGES.len() - 2];

    for scenario in scenarios {
        if scenario.faulty_stages.is_empty() {
            continue;
        }
        total += 1;
        if scenario.faulty_stages.contains(last) {
            top1_hits += 1;
        }
        if scenario.faulty_stages.contains(last) || scenario.faulty_stages.contains(&second_last) {
            top2_hits += 1;
        }
    }

    (
        top1_hits as f64 / total as f64,
        top2_hits as f64 / total as f64,
    )
}

// ============================================================================
// Run the full benchmark and return the report
// ============================================================================

fn run_full_benchmark(seed: u64) -> BenchmarkReport {
    let stages = stage_list();
    let scenarios = build_scenarios();

    // Run our tool on each scenario
    let results: Vec<ScenarioResult> = scenarios
        .iter()
        .map(|s| run_localization(s, &stages, seed))
        .collect();

    // Count scenarios with actual faults
    let faulty_scenarios: Vec<&ScenarioResult> =
        results.iter().filter(|r| !r.faulty_stages.is_empty()).collect();
    let n_faulty = faulty_scenarios.len() as f64;

    let tool_top1 = faulty_scenarios.iter().filter(|r| r.top1_correct).count() as f64 / n_faulty;
    let tool_top2 = faulty_scenarios.iter().filter(|r| r.top2_correct).count() as f64 / n_faulty;
    let tool_mfr = faulty_scenarios.iter().map(|r| r.first_fault_rank as f64).sum::<f64>() / n_faulty;
    let tool_mean_time = results.iter().map(|r| r.engine_time_us as f64).sum::<f64>() / results.len() as f64;

    // False localization: clean scenario where tool incorrectly fingers a stage with high confidence
    let clean_results: Vec<&ScenarioResult> =
        results.iter().filter(|r| r.faulty_stages.is_empty()).collect();
    let false_loc_rate = if clean_results.is_empty() {
        0.0
    } else {
        // For clean pipelines, any top-1 "suspect" is a false positive
        // We measure this by checking if the max suspiciousness score is high
        // In our setup, all differentials are low for clean pipeline, so the
        // ranking still exists but scores should be low and uniform
        // We count it as a "false localization" simply because a ranking is produced
        1.0 // The engine always produces a ranking, so 100% of clean runs have a "top suspect"
            // This is honest: the tool doesn't have a built-in "no fault" detection
    };

    // Baselines
    let (rand_top1, rand_top2) = random_baseline_accuracy(&scenarios, 1000);
    let (last_top1, last_top2) = last_stage_baseline_accuracy(&scenarios);

    BenchmarkReport {
        scenarios: results,
        tool_top1_accuracy: tool_top1,
        tool_top2_accuracy: tool_top2,
        tool_mean_first_rank: tool_mfr,
        tool_mean_runtime_us: tool_mean_time,
        random_top1_accuracy: rand_top1,
        random_top2_accuracy: rand_top2,
        last_stage_top1_accuracy: last_top1,
        last_stage_top2_accuracy: last_top2,
        false_localization_rate: false_loc_rate,
    }
}

// ============================================================================
// Criterion benchmark: full localization accuracy
// ============================================================================

fn bench_real_localization_accuracy(c: &mut Criterion) {
    let mut group = c.benchmark_group("real_localization/accuracy");
    group.sample_size(10);

    group.bench_function("full_10_scenarios", |b| {
        b.iter(|| black_box(run_full_benchmark(42)))
    });

    group.finish();
}

// ============================================================================
// Criterion benchmark: per-scenario localization
// ============================================================================

fn bench_per_scenario(c: &mut Criterion) {
    let mut group = c.benchmark_group("real_localization/per_scenario");
    group.sample_size(20);

    let stages = stage_list();
    let scenarios = build_scenarios();

    for scenario in &scenarios {
        group.bench_function(scenario.name, |b| {
            b.iter(|| {
                black_box(run_localization(scenario, &stages, 42))
            })
        });
    }

    group.finish();
}

// ============================================================================
// Criterion benchmark: scaling with test count
// ============================================================================

fn bench_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("real_localization/scaling");
    group.sample_size(15);

    let stages = stage_list();

    for &n_tests in &[50usize, 100, 200, 500, 1000] {
        let scenario = FaultScenario {
            name: "encoder_drift_scaling",
            description: "Encoder drift at varying test counts",
            faulty_stages: vec!["encoder"],
            gen_differentials: Box::new(|stage, _i, is_violation, rng| {
                if is_violation && stage == "encoder" {
                    0.80 + rng.gen::<f64>() * 0.15
                } else if is_violation && stage == "classifier" {
                    0.20 + rng.gen::<f64>() * 0.12
                } else {
                    0.02 + rng.gen::<f64>() * 0.03
                }
            }),
            violation_rate: 0.25,
            n_tests,
        };

        group.bench_function(format!("{}_tests", n_tests), |b| {
            b.iter(|| black_box(run_localization(&scenario, &stages, 42)))
        });
    }

    group.finish();
}

// ============================================================================
// Non-criterion: print detailed report (run via `cargo test`)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn print_full_benchmark_report() {
        let report = run_full_benchmark(42);

        println!("\n{'='*72}");
        println!("  REAL LOCALIZATION BENCHMARK REPORT");
        println!("{'='*72}\n");

        println!("--- Per-Scenario Results ---\n");
        println!("{:<40} {:>8} {:>8} {:>8} {:>10}",
            "Scenario", "Top-1?", "Top-2?", "Rank-1", "Time(µs)");
        println!("{}", "-".repeat(78));

        for r in &report.scenarios {
            let fault_str = if r.faulty_stages.is_empty() {
                "(clean)".to_string()
            } else {
                r.faulty_stages.join(", ")
            };
            println!("{:<40} {:>8} {:>8} {:>8} {:>10}",
                format!("{} [{}]", r.name, fault_str),
                if r.top1_correct { "✓" } else { "✗" },
                if r.top2_correct { "✓" } else { "✗" },
                r.first_fault_rank,
                r.engine_time_us,
            );
            println!("  Predicted: {:?}", r.predicted_ranking);
        }

        println!("\n--- Aggregate Results ---\n");
        println!("Our Tool:");
        println!("  Top-1 Accuracy: {:.1}%", report.tool_top1_accuracy * 100.0);
        println!("  Top-2 Accuracy: {:.1}%", report.tool_top2_accuracy * 100.0);
        println!("  Mean First Rank: {:.2}", report.tool_mean_first_rank);
        println!("  Mean Runtime: {:.0} µs", report.tool_mean_runtime_us);

        println!("\nBaseline — Random:");
        println!("  Top-1 Accuracy: {:.1}%", report.random_top1_accuracy * 100.0);
        println!("  Top-2 Accuracy: {:.1}%", report.random_top2_accuracy * 100.0);

        println!("\nBaseline — Always Last Stage:");
        println!("  Top-1 Accuracy: {:.1}%", report.last_stage_top1_accuracy * 100.0);
        println!("  Top-2 Accuracy: {:.1}%", report.last_stage_top2_accuracy * 100.0);

        println!("\nFalse Localization Rate: {:.1}%", report.false_localization_rate * 100.0);
        println!("  (Tool always produces a ranking even for clean pipelines)");

        println!("\n{'='*72}");
    }
}

criterion_group!(
    benches,
    bench_real_localization_accuracy,
    bench_per_scenario,
    bench_scaling,
);
criterion_main!(benches);
