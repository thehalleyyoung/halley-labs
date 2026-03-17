//! Benchmarks for XR accessibility verification pipeline.
//!
//! Measures Tier 1 verification throughput on scenes of increasing complexity,
//! zone partition construction, and interval arithmetic.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use xr_examples::generate_scene;
use xr_types::kinematic::BodyParameterRange;

use xr_lint::tier1_engine::Tier1Engine;
use xr_lint::{LintConfig, SceneLinter};

use xr_spatial::interval::{Interval, IntervalVector};
use xr_spatial::zone::{ZoneClassification, ZonePartition};

// ── Tier 1 scene verification ───────────────────────────────────────────────

fn bench_tier1_scene(c: &mut Criterion) {
    let range = BodyParameterRange::default();
    let engine = Tier1Engine::new(&range);

    let mut group = c.benchmark_group("tier1_scene_check");
    for &n in &[10, 50, 100, 500, 1000] {
        let scene = generate_scene(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &scene, |b, scene| {
            b.iter(|| {
                let results = engine.full_scene_check(black_box(scene));
                black_box(results);
            });
        });
    }
    group.finish();
}

// ── Lint pass ───────────────────────────────────────────────────────────────

fn bench_lint(c: &mut Criterion) {
    let linter = SceneLinter::with_config(LintConfig {
        require_labels: true,
        require_feedback: true,
        ..LintConfig::default()
    });

    let mut group = c.benchmark_group("lint_scene");
    for &n in &[10, 50, 100, 500] {
        let scene = generate_scene(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &scene, |b, scene| {
            b.iter(|| {
                let report = linter.lint(black_box(scene));
                black_box(report);
            });
        });
    }
    group.finish();
}

// ── Tier 1 single-element quick_check ───────────────────────────────────────

fn bench_quick_check(c: &mut Criterion) {
    let range = BodyParameterRange::default();
    let engine = Tier1Engine::new(&range);
    let scene = generate_scene(1);
    let element = &scene.elements[0];

    c.bench_function("tier1_quick_check_single", |b| {
        b.iter(|| {
            let result = engine.quick_check(black_box(element));
            black_box(result);
        });
    });
}

// ── Zone partition construction ─────────────────────────────────────────────

fn bench_zone_partition(c: &mut Criterion) {
    let mut group = c.benchmark_group("zone_partition_build");
    for &n in &[10, 50, 200, 1000] {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
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
                    let class = if i % 3 == 0 {
                        ZoneClassification::Accessible
                    } else if i % 3 == 1 {
                        ZoneClassification::Inaccessible
                    } else {
                        ZoneClassification::Uncertain
                    };
                    partition.add_zone(region, class);
                }
                black_box(&partition);
            });
        });
    }
    group.finish();
}

// ── Interval arithmetic operations ──────────────────────────────────────────

fn bench_interval_ops(c: &mut Criterion) {
    let a = Interval::new(0.8, 1.2);
    let b = Interval::new(0.3, 0.5);

    c.bench_function("interval_add", |bench| {
        bench.iter(|| black_box(black_box(a) + black_box(b)));
    });

    c.bench_function("interval_mul", |bench| {
        bench.iter(|| black_box(black_box(a) * black_box(b)));
    });

    c.bench_function("interval_sin", |bench| {
        bench.iter(|| black_box(black_box(a).sin()));
    });

    c.bench_function("interval_cos", |bench| {
        bench.iter(|| black_box(black_box(a).cos()));
    });

    c.bench_function("interval_sqr", |bench| {
        bench.iter(|| black_box(black_box(a).sqr()));
    });

    c.bench_function("interval_sqrt", |bench| {
        bench.iter(|| black_box(black_box(a).sqrt()));
    });
}

criterion_group!(
    benches,
    bench_tier1_scene,
    bench_lint,
    bench_quick_check,
    bench_zone_partition,
    bench_interval_ops,
);
criterion_main!(benches);
