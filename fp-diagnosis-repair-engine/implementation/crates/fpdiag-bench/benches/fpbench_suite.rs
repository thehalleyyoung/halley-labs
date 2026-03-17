//! Benchmark: FPBench standard suite — Penumbra vs. Herbie reference.
//!
//! Measures Penumbra's diagnosis and repair quality on the standard
//! FPBench expressions and compares against published Herbie results.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use fpdiag_analysis::EagBuilder;
use fpdiag_bench::suite::{build_trace_for_case, fpbench_standard_suite, herbie_reference_results};
use fpdiag_diagnosis::DiagnosisEngine;
use fpdiag_repair::RepairSynthesizer;

fn bench_fpbench_suite(c: &mut Criterion) {
    let mut group = c.benchmark_group("fpbench_suite");
    let suite = fpbench_standard_suite();

    for case in &suite {
        group.bench_function(case.name, |b| {
            let trace = build_trace_for_case(case);

            b.iter(|| {
                // Build EAG
                let mut builder = EagBuilder::with_defaults();
                builder.build_from_trace(black_box(&trace)).unwrap();
                let eag = builder.finish();

                // Diagnose
                let engine = DiagnosisEngine::with_defaults();
                let report = engine.diagnose(&eag).unwrap();

                // Repair
                let synth = RepairSynthesizer::with_defaults();
                let _repair = synth.synthesize(&eag, &report);

                black_box(report)
            });
        });
    }

    group.finish();
}

fn bench_herbie_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("herbie_comparison");
    let herbie_refs = herbie_reference_results();
    let suite = fpbench_standard_suite();

    for href in &herbie_refs {
        if let Some(case) = suite.iter().find(|c| c.name == href.benchmark) {
            group.bench_function(&format!("{}_penumbra_pipeline", href.benchmark), |b| {
                let trace = build_trace_for_case(case);
                b.iter(|| {
                    let mut builder = EagBuilder::with_defaults();
                    builder.build_from_trace(black_box(&trace)).unwrap();
                    let eag = builder.finish();
                    let engine = DiagnosisEngine::with_defaults();
                    let report = engine.diagnose(&eag).unwrap();
                    let synth = RepairSynthesizer::with_defaults();
                    let _ = synth.synthesize(&eag, &report);
                    black_box(report)
                });
            });
        }
    }

    group.finish();
}

criterion_group!(benches, bench_fpbench_suite, bench_herbie_comparison);
criterion_main!(benches);
