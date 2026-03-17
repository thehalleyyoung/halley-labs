//! Benchmark: Pareto frontier synthesis at varying scales.
//!
//! Measures construction time and quality metrics for Pareto frontiers
//! built from regulatory compliance cost vectors at different problem sizes.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use regsynth_pareto::{
    CostVector, ParetoFrontier, filter_dominated, fast_non_dominated_sort,
    metrics::hypervolume_indicator,
};
use rand::Rng;
use rand::SeedableRng;

/// Generate `n` random 4-D regulatory cost vectors.
fn random_cost_vectors(n: usize, seed: u64) -> Vec<CostVector> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            CostVector::regulatory(
                rng.gen_range(10_000.0..1_000_000.0),
                rng.gen_range(1.0..24.0),
                rng.gen_range(0.01..1.0),
                rng.gen_range(1.0..100.0),
            )
        })
        .collect()
}

/// Benchmark inserting points into a ParetoFrontier one at a time.
fn bench_frontier_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("pareto_frontier_construction");
    for &size in &[50, 200, 1000, 5000] {
        let vectors = random_cost_vectors(size, 42);
        group.bench_with_input(
            BenchmarkId::new("insert", size),
            &vectors,
            |b, vecs| {
                b.iter(|| {
                    let mut frontier: ParetoFrontier<usize> = ParetoFrontier::new(4);
                    for (i, cv) in vecs.iter().enumerate() {
                        frontier.add_point(i, cv.clone());
                    }
                    frontier
                });
            },
        );
    }
    group.finish();
}

/// Benchmark epsilon-dominance frontier construction.
fn bench_epsilon_frontier(c: &mut Criterion) {
    let mut group = c.benchmark_group("pareto_epsilon_frontier");
    let vectors = random_cost_vectors(1000, 99);
    for &eps in &[0.01, 0.05, 0.1, 0.2] {
        group.bench_with_input(
            BenchmarkId::new("eps", format!("{:.2}", eps)),
            &eps,
            |b, &epsilon| {
                b.iter(|| {
                    let mut frontier: ParetoFrontier<usize> =
                        ParetoFrontier::with_epsilon(4, epsilon);
                    for (i, cv) in vectors.iter().enumerate() {
                        frontier.add_point(i, cv.clone());
                    }
                    frontier
                });
            },
        );
    }
    group.finish();
}

/// Benchmark NSGA-II non-dominated sorting.
fn bench_non_dominated_sort(c: &mut Criterion) {
    let mut group = c.benchmark_group("pareto_nsga2_sort");
    for &size in &[100, 500, 2000] {
        let vectors = random_cost_vectors(size, 123);
        group.bench_with_input(
            BenchmarkId::new("sort", size),
            &vectors,
            |b, vecs| {
                b.iter(|| fast_non_dominated_sort(vecs));
            },
        );
    }
    group.finish();
}

/// Benchmark dominance filtering.
fn bench_filter_dominated(c: &mut Criterion) {
    let mut group = c.benchmark_group("pareto_filter_dominated");
    for &size in &[100, 500, 2000, 10000] {
        let vectors = random_cost_vectors(size, 77);
        group.bench_with_input(
            BenchmarkId::new("filter", size),
            &vectors,
            |b, vecs| {
                b.iter(|| filter_dominated(vecs));
            },
        );
    }
    group.finish();
}

/// Benchmark hypervolume computation.
fn bench_hypervolume(c: &mut Criterion) {
    let mut group = c.benchmark_group("pareto_hypervolume");
    let reference = CostVector::regulatory(1_000_000.0, 24.0, 1.0, 100.0);
    for &size in &[10, 50, 200] {
        let vectors = random_cost_vectors(size, 55);
        let non_dominated = filter_dominated(&vectors);
        group.bench_with_input(
            BenchmarkId::new("hv", non_dominated.len()),
            &non_dominated,
            |b, pts| {
                b.iter(|| hypervolume_indicator(pts, &reference));
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_frontier_construction,
    bench_epsilon_frontier,
    bench_non_dominated_sort,
    bench_filter_dominated,
    bench_hypervolume,
);
criterion_main!(benches);
