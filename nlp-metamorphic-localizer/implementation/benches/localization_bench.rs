use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use std::collections::HashMap;

use shared_types::StageId;
use localization::{
    LocalizationConfig, LocalizationEngine, SBFLMetric, TestObservation,
};
use statistical_oracle::{
    DifferentialMatrix, OchiaiMetric, TarantulaMetric, DStarMetric, BarinelMetric,
    ViolationVector,
};
use evaluation::metrics::{
    compute_top_k, compute_exam, compute_wasted_effort, compute_full_metrics,
};

// ---------------------------------------------------------------------------
// Helper: register N stages in an engine
// ---------------------------------------------------------------------------
fn stages_for(n: usize) -> Vec<(StageId, String)> {
    let names = [
        "tokenizer",
        "pos_tagger",
        "dep_parser",
        "ner",
        "sentiment",
        "embedder",
        "classifier",
    ];
    (0..n)
        .map(|i| {
            let name = names[i % names.len()].to_string();
            (StageId::new(&name), name)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Helper: generate synthetic observations
// ---------------------------------------------------------------------------
fn synthetic_observations(
    n_tests: usize,
    stages: &[(StageId, String)],
) -> Vec<TestObservation> {
    let mut obs = Vec::with_capacity(n_tests);
    for i in 0..n_tests {
        let violation = i % 4 == 0;
        let mut per_stage = HashMap::new();
        for (_, name) in stages {
            let diff = if violation && name == "dep_parser" {
                0.85 + (i as f64 % 7.0) * 0.01
            } else {
                0.02 + (i as f64 % 5.0) * 0.005
            };
            per_stage.insert(name.clone(), diff);
        }
        obs.push(TestObservation {
            test_id: format!("test_{i}"),
            transformation_name: ["passivization", "clefting", "topicalization"][i % 3]
                .to_string(),
            input_text: format!("The quick brown fox jumps over the lazy dog {i}"),
            transformed_text: format!("The lazy dog was jumped over by the quick brown fox {i}"),
            violation_detected: violation,
            violation_magnitude: if violation { 0.9 } else { 0.03 },
            per_stage_differentials: per_stage,
            execution_time_ms: 1.5,
        });
    }
    obs
}

// ---------------------------------------------------------------------------
// Helper: build a differential matrix for SBFL metrics
// ---------------------------------------------------------------------------
fn build_matrix(n_tests: usize, n_stages: usize) -> (DifferentialMatrix, ViolationVector) {
    let stage_names: Vec<String> = (0..n_stages).map(|k| format!("stage_{k}")).collect();
    let mut data = Vec::with_capacity(n_tests);
    let mut violations = Vec::with_capacity(n_tests);
    for i in 0..n_tests {
        let violation = i % 4 == 0;
        violations.push(violation);
        let mut row = Vec::with_capacity(n_stages);
        for k in 0..n_stages {
            let val = if violation && k == 2 {
                0.85 + (i as f64 % 10.0) * 0.01
            } else {
                0.03 + (k as f64) * 0.005
            };
            row.push(val);
        }
        data.push(row);
    }
    (
        DifferentialMatrix::new(data, stage_names).unwrap(),
        ViolationVector::new(violations),
    )
}

// =========================================================================
// Benchmark group 1: Fault localization on varying pipeline sizes / corpus
// =========================================================================
fn bench_localization_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("localization/pipeline");
    group.sample_size(20);

    for &n_stages in &[3usize, 5, 7] {
        for &n_tests in &[100usize, 500, 1000] {
            let stages = stages_for(n_stages);
            let observations = synthetic_observations(n_tests, &stages);

            group.throughput(Throughput::Elements(n_tests as u64));
            group.bench_with_input(
                BenchmarkId::new(
                    format!("{n_stages}_stages"),
                    n_tests,
                ),
                &(stages.clone(), observations),
                |b, (stages, obs)| {
                    b.iter(|| {
                        let config = LocalizationConfig {
                            sbfl_metric: SBFLMetric::Ochiai,
                            enable_causal_analysis: false,
                            enable_discriminability_check: false,
                            ..Default::default()
                        };
                        let mut engine = LocalizationEngine::with_config(config);
                        engine.register_stages(stages.clone());
                        engine.record_observations(obs.clone());
                        black_box(engine.compute_suspiciousness())
                    })
                },
            );
        }
    }
    group.finish();
}

// =========================================================================
// Benchmark group 2: SBFL metric computation
// =========================================================================
fn bench_sbfl_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("localization/sbfl_metrics");
    group.sample_size(30);

    for &(n_tests, n_stages) in &[(100, 5), (500, 7), (1000, 10)] {
        let (matrix, violations) = build_matrix(n_tests, n_stages);

        group.bench_with_input(
            BenchmarkId::new("ochiai", format!("{n_tests}x{n_stages}")),
            &(matrix.clone(), violations.clone()),
            |b, (m, v)| {
                b.iter(|| {
                    let metric = OchiaiMetric::new();
                    black_box(metric.compute_suspiciousness(m, v).unwrap())
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("tarantula", format!("{n_tests}x{n_stages}")),
            &(matrix.clone(), violations.clone()),
            |b, (m, v)| {
                b.iter(|| {
                    let metric = TarantulaMetric::new();
                    black_box(metric.compute_suspiciousness(m, v).unwrap())
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("dstar", format!("{n_tests}x{n_stages}")),
            &(matrix.clone(), violations.clone()),
            |b, (m, v)| {
                b.iter(|| {
                    let metric = DStarMetric::new();
                    black_box(metric.compute_suspiciousness(m, v).unwrap())
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("barinel", format!("{n_tests}x{n_stages}")),
            &(matrix.clone(), violations.clone()),
            |b, (m, v)| {
                b.iter(|| {
                    let metric = BarinelMetric::new();
                    black_box(metric.compute_posteriors(m, v).unwrap())
                })
            },
        );
    }
    group.finish();
}

// =========================================================================
// Benchmark group 3: Causal intervention analysis
// =========================================================================
fn bench_causal_intervention(c: &mut Criterion) {
    let mut group = c.benchmark_group("localization/causal_intervention");
    group.sample_size(30);

    let n_stages = 5;
    let stages = stages_for(n_stages);
    let observations = synthetic_observations(500, &stages);

    let config = LocalizationConfig {
        enable_causal_analysis: true,
        enable_discriminability_check: false,
        ..Default::default()
    };
    let mut engine = LocalizationEngine::with_config(config);
    engine.register_stages(stages.clone());
    engine.record_observations(observations);

    group.bench_function("single_intervention", |b| {
        b.iter(|| {
            black_box(engine.analyze_causal_intervention("dep_parser", 0.92, 0.15))
        })
    });

    let intervention_results: Vec<(String, f64)> = stages
        .iter()
        .map(|(_, name)| {
            let post = if name == "dep_parser" { 0.12 } else { 0.85 };
            (name.clone(), post)
        })
        .collect();

    group.bench_function("iterative_peeling", |b| {
        b.iter(|| {
            black_box(engine.iterative_peeling(0.92, &intervention_results))
        })
    });

    group.finish();
}

// =========================================================================
// Benchmark group 4: Stage discriminability matrix computation
// =========================================================================
fn bench_discriminability(c: &mut Criterion) {
    let mut group = c.benchmark_group("localization/discriminability");
    group.sample_size(20);

    for &n_stages in &[3usize, 5, 7] {
        let stages = stages_for(n_stages);
        let observations = synthetic_observations(500, &stages);

        let config = LocalizationConfig {
            enable_causal_analysis: false,
            enable_discriminability_check: true,
            ..Default::default()
        };
        let mut engine = LocalizationEngine::with_config(config);
        engine.register_stages(stages.clone());
        engine.record_observations(observations);

        // Pre-compute spectra then benchmark discriminability
        let _ = engine.compute_suspiciousness();
        let spectra = engine.get_spectra().clone();
        let stage_order: Vec<String> = stages.iter().map(|(_, n)| n.clone()).collect();
        let transform_order = vec![
            "passivization".to_string(),
            "clefting".to_string(),
            "topicalization".to_string(),
        ];

        group.bench_with_input(
            BenchmarkId::from_parameter(n_stages),
            &(spectra, stage_order, transform_order),
            |b, (spec, so, to)| {
                b.iter(|| {
                    black_box(localization::compute_discriminability(spec, so, to))
                })
            },
        );
    }
    group.finish();
}

// =========================================================================
// Benchmark group 5: Evaluation metric computation
// =========================================================================
fn bench_evaluation_metrics(c: &mut Criterion) {
    let mut group = c.benchmark_group("localization/eval_metrics");
    group.sample_size(50);

    let predicted: Vec<String> = (0..10).map(|i| format!("stage_{i}")).collect();
    let actual = vec!["stage_2".to_string(), "stage_5".to_string()];
    let k_values = vec![1, 2, 3, 5];

    group.bench_function("top_k", |b| {
        b.iter(|| black_box(compute_top_k(&predicted, &actual, &k_values)))
    });

    group.bench_function("exam_score", |b| {
        b.iter(|| black_box(compute_exam(&predicted, &actual)))
    });

    group.bench_function("wasted_effort", |b| {
        b.iter(|| black_box(compute_wasted_effort(&predicted, &actual)))
    });

    group.bench_function("full_metrics", |b| {
        b.iter(|| black_box(compute_full_metrics(&predicted, &actual)))
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_localization_pipeline,
    bench_sbfl_metrics,
    bench_causal_intervention,
    bench_discriminability,
    bench_evaluation_metrics,
);
criterion_main!(benches);
