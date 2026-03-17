//! Benchmarks for RTIG construction and Tier 1 analysis at varying topology
//! sizes. Uses Criterion for statistically rigorous measurement.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::collections::HashMap;

use cascade_graph::rtig::RtigGraph;
use cascade_types::policy::{ResiliencePolicy, RetryPolicy, TimeoutPolicy};
use cascade_types::service::ServiceId;

use cascade_analysis::tier1::{Tier1Analyzer, Tier1Config};

// ---------------------------------------------------------------------------
// Topology generators
// ---------------------------------------------------------------------------

/// Chain topology: svc_0 → svc_1 → … → svc_{n-1}.
/// Each edge has 3 retries and a 5 s per-try timeout.
fn build_chain(n: usize) -> RtigGraph {
    let mut g = RtigGraph::new();
    let ids: Vec<ServiceId> = (0..n).map(|i| ServiceId::new(format!("svc_{i}"))).collect();
    for id in &ids {
        g.add_service(id.clone());
    }
    let policy = ResiliencePolicy::empty()
        .with_retry(RetryPolicy::new(3).with_per_try_timeout(5000))
        .with_timeout(TimeoutPolicy::new(20_000));
    for w in ids.windows(2) {
        g.add_dependency(&w[0], &w[1], policy.clone());
    }
    g
}

/// Star topology: hub → spoke_0, hub → spoke_1, …, hub → spoke_{n-2}.
/// The hub is `svc_0`; spokes are `svc_1` … `svc_{n-1}`.
fn build_star(n: usize) -> RtigGraph {
    let mut g = RtigGraph::new();
    let hub = ServiceId::new("svc_0");
    g.add_service(hub.clone());
    let policy = ResiliencePolicy::empty()
        .with_retry(RetryPolicy::new(2).with_per_try_timeout(3000))
        .with_timeout(TimeoutPolicy::new(15_000));
    for i in 1..n {
        let spoke = ServiceId::new(format!("svc_{i}"));
        g.add_service(spoke.clone());
        g.add_dependency(&hub, &spoke, policy.clone());
    }
    g
}

/// Mesh topology: `n` services, each with ~3 outgoing edges to random
/// successors (deterministic via simple modular arithmetic, no `rand` needed).
fn build_mesh(n: usize) -> RtigGraph {
    let mut g = RtigGraph::new();
    let ids: Vec<ServiceId> = (0..n).map(|i| ServiceId::new(format!("svc_{i}"))).collect();
    for id in &ids {
        g.add_service(id.clone());
    }
    let policy = ResiliencePolicy::empty()
        .with_retry(RetryPolicy::new(2).with_per_try_timeout(4000))
        .with_timeout(TimeoutPolicy::new(15_000));
    // Deterministic pseudo-random edges: ~3 per node using a simple hash.
    for i in 0..n {
        let edges_per_node = 3usize.min(n - 1);
        for k in 1..=edges_per_node {
            let j = (i.wrapping_mul(7) + k.wrapping_mul(13) + 3) % n;
            if j != i {
                g.add_dependency(&ids[i], &ids[j], policy.clone());
            }
        }
    }
    g
}

// ---------------------------------------------------------------------------
// Adjacency helpers (converts an RtigGraph into Tier 1 input tuples)
// ---------------------------------------------------------------------------

fn graph_to_adjacency(g: &RtigGraph) -> Vec<(String, String, u32, u64, u64)> {
    g.to_adjacency_list()
        .into_iter()
        .map(|(src, tgt, pol)| {
            let retries = pol.retry.as_ref().map_or(0, |r| r.max_retries);
            let timeout = pol
                .timeout
                .as_ref()
                .map_or(10_000, |t| t.request_timeout_ms);
            (src.as_str().to_owned(), tgt.as_str().to_owned(), retries, timeout, 1u64)
        })
        .collect()
}

fn make_capacities(g: &RtigGraph) -> HashMap<String, u64> {
    g.services()
        .into_iter()
        .map(|s| (s.as_str().to_owned(), 1000))
        .collect()
}

fn make_deadlines(g: &RtigGraph) -> HashMap<String, u64> {
    g.services()
        .into_iter()
        .map(|s| (s.as_str().to_owned(), 60_000))
        .collect()
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

static SIZES: &[usize] = &[10, 50, 100, 500];

fn bench_rtig_construction_chain(c: &mut Criterion) {
    let mut group = c.benchmark_group("rtig_construction_chain");
    for &n in SIZES {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter(|| build_chain(black_box(n)));
        });
    }
    group.finish();
}

fn bench_rtig_construction_star(c: &mut Criterion) {
    let mut group = c.benchmark_group("rtig_construction_star");
    for &n in SIZES {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter(|| build_star(black_box(n)));
        });
    }
    group.finish();
}

fn bench_rtig_construction_mesh(c: &mut Criterion) {
    let mut group = c.benchmark_group("rtig_construction_mesh");
    for &n in SIZES {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter(|| build_mesh(black_box(n)));
        });
    }
    group.finish();
}

fn bench_tier1_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("tier1_analysis");
    let config = Tier1Config::default();
    let analyzer = Tier1Analyzer::new();

    for &n in SIZES {
        let g = build_chain(n);
        let adj = graph_to_adjacency(&g);
        let caps = make_capacities(&g);
        let deadlines = make_deadlines(&g);

        group.bench_with_input(BenchmarkId::new("chain", n), &n, |b, _| {
            b.iter(|| {
                analyzer.analyze(
                    black_box(&adj),
                    black_box(&caps),
                    black_box(&deadlines),
                    black_box(&config),
                )
            });
        });
    }
    group.finish();
}

fn bench_path_amplification(c: &mut Criterion) {
    let mut group = c.benchmark_group("path_amplification");
    let config = Tier1Config::default();
    let analyzer = Tier1Analyzer::new();

    for &n in SIZES {
        let g = build_chain(n);
        let adj = graph_to_adjacency(&g);
        let caps = make_capacities(&g);

        group.bench_with_input(BenchmarkId::new("chain", n), &n, |b, _| {
            b.iter(|| {
                analyzer.find_amplification_risks(
                    black_box(&adj),
                    black_box(&caps),
                    black_box(&config),
                )
            });
        });
    }
    group.finish();
}

fn bench_topology_building(c: &mut Criterion) {
    let mut group = c.benchmark_group("topology_building");

    for &n in SIZES {
        group.bench_with_input(BenchmarkId::new("chain", n), &n, |b, &n| {
            b.iter(|| {
                let g = build_chain(black_box(n));
                black_box(g.graph_stats())
            });
        });
        group.bench_with_input(BenchmarkId::new("star", n), &n, |b, &n| {
            b.iter(|| {
                let g = build_star(black_box(n));
                black_box(g.graph_stats())
            });
        });
        group.bench_with_input(BenchmarkId::new("mesh", n), &n, |b, &n| {
            b.iter(|| {
                let g = build_mesh(black_box(n));
                black_box(g.graph_stats())
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_rtig_construction_chain,
    bench_rtig_construction_star,
    bench_rtig_construction_mesh,
    bench_tier1_analysis,
    bench_path_amplification,
    bench_topology_building,
);
criterion_main!(benches);
