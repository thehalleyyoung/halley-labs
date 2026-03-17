//! Benchmark: Diagnosis engine performance.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use fpdiag_analysis::EagBuilder;
use fpdiag_diagnosis::DiagnosisEngine;
use fpdiag_types::{
    expression::FpOp,
    precision::Precision,
    trace::{ExecutionTrace, TraceEvent},
};

fn build_cancellation_trace(n: usize) -> ExecutionTrace {
    let mut trace = ExecutionTrace::new();
    for i in 0..n {
        let a = 1.0 + (i as f64) * 1e-15;
        let b = 1.0;
        trace.push(TraceEvent::Operation {
            seq: i as u64,
            op: FpOp::Sub,
            inputs: vec![a, b],
            output: a - b,
            shadow_output: (i as f64) * 1e-15,
            precision: Precision::Double,
            source: None,
            expr_node: None,
        });
    }
    trace.finalize();
    trace
}

fn build_absorption_trace(n: usize) -> ExecutionTrace {
    let mut trace = ExecutionTrace::new();
    let mut acc = 1e16_f64;
    for i in 0..n {
        let old = acc;
        acc += 1.0;
        trace.push(TraceEvent::Operation {
            seq: i as u64,
            op: FpOp::Add,
            inputs: vec![old, 1.0],
            output: acc,
            shadow_output: 1e16 + (i + 1) as f64,
            precision: Precision::Double,
            source: None,
            expr_node: None,
        });
    }
    trace.finalize();
    trace
}

fn bench_diagnosis(c: &mut Criterion) {
    let mut group = c.benchmark_group("diagnosis_engine");

    for size in [10, 50, 100, 500] {
        group.bench_with_input(BenchmarkId::new("cancellation", size), &size, |b, &size| {
            let trace = build_cancellation_trace(size);
            let mut builder = EagBuilder::with_defaults();
            builder.build_from_trace(&trace).unwrap();
            let eag = builder.finish();
            let engine = DiagnosisEngine::with_defaults();

            b.iter(|| black_box(engine.diagnose(black_box(&eag)).unwrap()));
        });

        group.bench_with_input(BenchmarkId::new("absorption", size), &size, |b, &size| {
            let trace = build_absorption_trace(size);
            let mut builder = EagBuilder::with_defaults();
            builder.build_from_trace(&trace).unwrap();
            let eag = builder.finish();
            let engine = DiagnosisEngine::with_defaults();

            b.iter(|| black_box(engine.diagnose(black_box(&eag)).unwrap()));
        });
    }

    group.finish();
}

criterion_group!(benches, bench_diagnosis);
criterion_main!(benches);
