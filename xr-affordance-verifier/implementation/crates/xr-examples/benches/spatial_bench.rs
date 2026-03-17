//! Spatial and interval arithmetic benchmarks.
//!
//! Measures raw interval and affine-form throughput, spatial index
//! construction, and zone subdivision strategies.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use xr_spatial::affine::AffineForm;
use xr_spatial::interval::{Interval, IntervalVector};
use xr_spatial::zone::{Zone, ZoneClassification, ZonePartition};

// ── Interval arithmetic ─────────────────────────────────────────────────────

fn bench_interval_arithmetic(c: &mut Criterion) {
    let mut group = c.benchmark_group("interval_arithmetic");

    let a = Interval::new(0.5, 1.5);
    let b = Interval::new(0.2, 0.8);

    group.bench_function("add", |bench| {
        bench.iter(|| black_box(black_box(a) + black_box(b)));
    });

    group.bench_function("sub", |bench| {
        bench.iter(|| black_box(black_box(a) - black_box(b)));
    });

    group.bench_function("mul", |bench| {
        bench.iter(|| black_box(black_box(a) * black_box(b)));
    });

    group.bench_function("div", |bench| {
        bench.iter(|| black_box(black_box(a) / black_box(b)));
    });

    group.bench_function("sin", |bench| {
        bench.iter(|| black_box(black_box(a).sin()));
    });

    group.bench_function("cos", |bench| {
        bench.iter(|| black_box(black_box(a).cos()));
    });

    group.bench_function("sqr", |bench| {
        bench.iter(|| black_box(black_box(a).sqr()));
    });

    group.bench_function("sqrt", |bench| {
        bench.iter(|| black_box(black_box(a).sqrt()));
    });

    group.bench_function("bisect", |bench| {
        bench.iter(|| black_box(black_box(a).bisect()));
    });

    group.finish();
}

// ── Affine form operations ──────────────────────────────────────────────────

fn bench_affine_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("affine_form");

    let iv_a = Interval::new(0.8, 1.2);
    let iv_b = Interval::new(0.3, 0.5);

    group.bench_function("from_interval", |bench| {
        bench.iter(|| black_box(AffineForm::from_interval(black_box(&iv_a))));
    });

    let af_a = AffineForm::from_interval(&iv_a);
    let af_b = AffineForm::from_interval(&iv_b);

    group.bench_function("add", |bench| {
        bench.iter(|| black_box(black_box(&af_a).add(black_box(&af_b))));
    });

    group.bench_function("sub", |bench| {
        bench.iter(|| black_box(black_box(&af_a).sub(black_box(&af_b))));
    });

    group.bench_function("mul", |bench| {
        bench.iter(|| black_box(black_box(&af_a).mul(black_box(&af_b))));
    });

    group.bench_function("scale", |bench| {
        bench.iter(|| black_box(black_box(&af_a).scale(black_box(2.5))));
    });

    group.bench_function("to_interval", |bench| {
        bench.iter(|| black_box(black_box(&af_a).to_interval()));
    });

    group.finish();
}

// ── IntervalVector operations ───────────────────────────────────────────────

fn bench_interval_vector(c: &mut Criterion) {
    let mut group = c.benchmark_group("interval_vector");

    for &dim in &[3, 5, 10, 50] {
        let ranges: Vec<(f64, f64)> = (0..dim).map(|i| (i as f64 * 0.1, i as f64 * 0.1 + 0.2)).collect();
        let iv = IntervalVector::from_ranges(&ranges);
        let point: Vec<f64> = (0..dim).map(|i| i as f64 * 0.1 + 0.1).collect();

        group.bench_with_input(BenchmarkId::new("from_ranges", dim), &ranges, |b, ranges| {
            b.iter(|| black_box(IntervalVector::from_ranges(black_box(ranges))));
        });

        group.bench_with_input(BenchmarkId::new("contains_point", dim), &(&iv, &point), |b, &(iv, pt)| {
            b.iter(|| black_box(black_box(iv).contains_point(black_box(pt))));
        });

        group.bench_with_input(BenchmarkId::new("bisect_widest", dim), &iv, |b, iv| {
            b.iter(|| black_box(black_box(iv).bisect_widest()));
        });

        group.bench_with_input(BenchmarkId::new("volume", dim), &iv, |b, iv| {
            b.iter(|| black_box(black_box(iv).volume()));
        });
    }

    group.finish();
}

// ── Zone construction and queries ───────────────────────────────────────────

fn bench_zone_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("zone_construction");

    group.bench_function("standard_zones", |bench| {
        bench.iter(|| black_box(Zone::standard_zones(black_box(1.4), black_box(0.7))));
    });

    for &n in &[10, 100, 500] {
        group.bench_with_input(BenchmarkId::new("partition_build", n), &n, |b, &n| {
            b.iter(|| {
                let mut partition = ZonePartition::new();
                for i in 0..n {
                    let t = i as f64 / n as f64;
                    let region = IntervalVector::from_ranges(&[
                        (1.5 + t * 0.3, 1.55 + t * 0.3),
                        (0.29 + t * 0.1, 0.31 + t * 0.1),
                        (0.37 + t * 0.1, 0.39 + t * 0.1),
                        (0.22 + t * 0.05, 0.24 + t * 0.05),
                        (0.16 + t * 0.04, 0.18 + t * 0.04),
                    ]);
                    let class = match i % 3 {
                        0 => ZoneClassification::Accessible,
                        1 => ZoneClassification::Inaccessible,
                        _ => ZoneClassification::Uncertain,
                    };
                    partition.add_zone(region, class);
                }
                black_box(&partition);
            });
        });
    }

    group.finish();
}

// ── Zone subdivision ────────────────────────────────────────────────────────

fn bench_zone_subdivision(c: &mut Criterion) {
    let mut group = c.benchmark_group("zone_subdivision");

    for &depth in &[4, 8, 12] {
        group.bench_with_input(
            BenchmarkId::new("bisect_chain", depth),
            &depth,
            |b, &depth| {
                let root = IntervalVector::from_ranges(&[
                    (1.5, 1.9),
                    (0.29, 0.40),
                    (0.37, 0.53),
                    (0.22, 0.30),
                    (0.16, 0.21),
                ]);
                b.iter(|| {
                    let mut current = root.clone();
                    for _ in 0..depth {
                        let (left, _right) = current.bisect_widest();
                        current = left;
                    }
                    black_box(&current);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_interval_arithmetic,
    bench_affine_ops,
    bench_interval_vector,
    bench_zone_construction,
    bench_zone_subdivision,
);
criterion_main!(benches);
