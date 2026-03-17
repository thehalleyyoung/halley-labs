//! Benchmark: EAG construction performance.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use fpdiag_analysis::EagBuilder;
use fpdiag_types::{
    expression::FpOp,
    precision::Precision,
    trace::{ExecutionTrace, TraceEvent},
};

fn build_linear_trace(n: usize) -> ExecutionTrace {
    let mut trace = ExecutionTrace::new();
    let mut acc = 1.0_f64;
    for i in 0..n {
        let old = acc;
        acc += 1.0;
        trace.push(TraceEvent::Operation {
            seq: i as u64,
            op: FpOp::Add,
            inputs: vec![old, 1.0],
            output: acc,
            shadow_output: (i + 2) as f64,
            precision: Precision::Double,
            source: None,
            expr_node: None,
        });
    }
    trace.finalize();
    trace
}

fn build_mixed_trace(n: usize) -> ExecutionTrace {
    let mut trace = ExecutionTrace::new();
    let ops = [FpOp::Add, FpOp::Sub, FpOp::Mul, FpOp::Div];
    for i in 0..n {
        let op = ops[i % ops.len()];
        let a = (i as f64) + 1.0;
        let b = (i as f64) * 0.1 + 0.5;
        let output = match op {
            FpOp::Add => a + b,
            FpOp::Sub => a - b,
            FpOp::Mul => a * b,
            FpOp::Div => a / b,
            _ => a,
        };
        trace.push(TraceEvent::Operation {
            seq: i as u64,
            op,
            inputs: vec![a, b],
            output,
            shadow_output: output * 1.0000001,
            precision: Precision::Double,
            source: None,
            expr_node: None,
        });
    }
    trace.finalize();
    trace
}

fn bench_eag_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("eag_construction");

    for size in [10, 50, 100, 500, 1000] {
        group.bench_with_input(BenchmarkId::new("linear_chain", size), &size, |b, &size| {
            let trace = build_linear_trace(size);
            b.iter(|| {
                let mut builder = EagBuilder::with_defaults();
                builder.build_from_trace(black_box(&trace)).unwrap();
                black_box(builder.finish())
            });
        });

        group.bench_with_input(BenchmarkId::new("mixed_ops", size), &size, |b, &size| {
            let trace = build_mixed_trace(size);
            b.iter(|| {
                let mut builder = EagBuilder::with_defaults();
                builder.build_from_trace(black_box(&trace)).unwrap();
                black_box(builder.finish())
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_eag_construction);
criterion_main!(benches);
