//! Criterion benchmarks for SafeStep planning pipeline stages.
//!
//! Measures performance of:
//! - Version-product graph construction
//! - BMC encoding (interval vs naive clause counts)
//! - Envelope computation (forward/backward reachability)
//! - Treewidth decomposition heuristics
//! - Prefilter effectiveness (arc consistency, bitmap operations)
//!
//! Run with: `cargo bench --bench planning_benchmarks`

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use std::collections::HashMap;

use safestep_encoding::bmc::{BmcEncoder, BmcUnrolling, CompletenessChecker, StepEncoder};
use safestep_encoding::interval::{
    BinaryEncoding, Comparator, IntervalCompressor, IntervalEncoder, IntervalPredicate,
};
use safestep_encoding::prefilter::{ArcConsistency, CompatibilityBitmap, PairwisePrefilter};
use safestep_encoding::treewidth::{TreewidthComputer, TreeDecomposition};

// ---------------------------------------------------------------------------
// Helpers: synthetic cluster generation
// ---------------------------------------------------------------------------

/// Describes a synthetic deployment scenario used for benchmarking.
struct BenchCluster {
    /// Human-readable label for the scenario.
    label: &'static str,
    /// Number of micro-services in the cluster.
    num_services: usize,
    /// Number of version slots per service.
    versions_per_service: usize,
}

impl BenchCluster {
    const fn new(label: &'static str, num_services: usize, versions_per_service: usize) -> Self {
        Self {
            label,
            num_services,
            versions_per_service,
        }
    }

    /// Starting state: every service at version 0.
    fn start_state(&self) -> Vec<usize> {
        vec![0; self.num_services]
    }

    /// Target state: every service at its highest version.
    fn target_state(&self) -> Vec<usize> {
        vec![self.versions_per_service - 1; self.num_services]
    }

    /// Version counts per service (uniform).
    fn version_counts(&self) -> Vec<usize> {
        vec![self.versions_per_service; self.num_services]
    }

    /// Build a synthetic adjacency list representing service dependencies.
    ///
    /// Layout: linear chain  `0 -> 1 -> 2 -> ... -> n-1` plus a few
    /// cross-edges to keep the graph realistic.
    fn dependency_adjacency(&self) -> Vec<Vec<usize>> {
        let n = self.num_services;
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for i in 0..n.saturating_sub(1) {
            adj[i].push(i + 1);
            adj[i + 1].push(i);
        }
        // Add cross-edges every 5 services for realism.
        for i in (0..n).step_by(5) {
            let j = (i + 3) % n;
            if i != j && !adj[i].contains(&j) {
                adj[i].push(j);
                adj[j].push(i);
            }
        }
        adj
    }

    /// Build an `IntervalPredicate` where service `i` is compatible with
    /// service `i+1` when their version indices differ by at most `delta`.
    fn interval_predicate(&self, delta: usize) -> IntervalPredicate {
        let v = self.versions_per_service;
        let n = self.num_services;
        let mut pred = IntervalPredicate::new(v, v);
        for a in 0..v {
            let lo = a.saturating_sub(delta);
            let hi = (a + delta).min(v - 1);
            pred.set_range(a, lo, hi);
        }
        let _ = n; // used only for sizing elsewhere
        pred
    }

    /// Generate a dense boolean compatibility matrix for the *naive*
    /// encoding path (all-pairs enumeration).
    fn compatibility_matrix(&self, delta: usize) -> Vec<Vec<bool>> {
        let v = self.versions_per_service;
        let mut mat = vec![vec![false; v]; v];
        for a in 0..v {
            for b in 0..v {
                mat[a][b] = (a as isize - b as isize).unsigned_abs() <= delta;
            }
        }
        mat
    }

    /// Create a `BmcEncoder` suitable for this cluster.
    fn bmc_encoder(&self) -> BmcEncoder {
        BmcEncoder::new(self.num_services, self.version_counts())
    }

    /// Create a `PairwisePrefilter` with interval-compatible constraints.
    fn prefilter(&self, delta: usize) -> PairwisePrefilter {
        let n = self.num_services;
        let vps = self.version_counts();
        let mut pf = PairwisePrefilter::new(n, &vps);
        for i in 0..n.saturating_sub(1) {
            let mut bm = CompatibilityBitmap::new(vps[i], vps[i + 1]);
            for a in 0..vps[i] {
                for b in 0..vps[i + 1] {
                    bm.set(
                        a,
                        b,
                        (a as isize - b as isize).unsigned_abs() <= delta,
                    );
                }
            }
            pf.add_compatibility(i, i + 1, bm);
        }
        pf
    }
}

const SMALL: BenchCluster = BenchCluster::new("small_5svc_3ver", 5, 3);
const MEDIUM: BenchCluster = BenchCluster::new("medium_20svc_10ver", 20, 10);
const LARGE: BenchCluster = BenchCluster::new("large_50svc_20ver", 50, 20);

// ---------------------------------------------------------------------------
// Benchmark group 1: Graph construction
// ---------------------------------------------------------------------------

fn bench_graph_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_construction");
    group.sample_size(50);

    for cluster in &[&SMALL, &MEDIUM, &LARGE] {
        let n = cluster.num_services;
        let v = cluster.versions_per_service;
        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(
            BenchmarkId::new("bmc_encoder_init", cluster.label),
            &(n, v),
            |b, &(n, v)| {
                let vps = vec![v; n];
                b.iter(|| {
                    let enc = BmcEncoder::new(black_box(n), black_box(vps.clone()));
                    black_box(enc.total_variables(n));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("adjacency_build", cluster.label),
            cluster,
            |b, cl| {
                b.iter(|| {
                    black_box(cl.dependency_adjacency());
                });
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark group 2: BMC encoding
// ---------------------------------------------------------------------------

fn bench_encoding(c: &mut Criterion) {
    let mut group = c.benchmark_group("bmc_encoding");
    group.sample_size(30);

    for cluster in &[&SMALL, &MEDIUM, &LARGE] {
        let encoder = cluster.bmc_encoder();
        let start = cluster.start_state();
        let target = cluster.target_state();
        let depth = CompletenessChecker::completeness_bound(&start, &target);

        group.bench_with_input(
            BenchmarkId::new("encode_initial", cluster.label),
            &(&encoder, &start),
            |b, &(enc, st)| {
                b.iter(|| {
                    black_box(enc.encode_initial_state(black_box(st)));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("encode_target", cluster.label),
            &(&encoder, &target, depth),
            |b, &(enc, tgt, d)| {
                b.iter(|| {
                    black_box(enc.encode_target_state(black_box(tgt), black_box(d)));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("encode_transition", cluster.label),
            &(&encoder, depth),
            |b, &(enc, d)| {
                b.iter(|| {
                    for step in 0..d {
                        black_box(enc.encode_transition(black_box(step)));
                    }
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("step_encoder_amo", cluster.label),
            &(&encoder, depth),
            |b, &(enc, d)| {
                b.iter(|| {
                    for step in 0..d {
                        black_box(StepEncoder::encode_best(
                            black_box(cluster.num_services),
                            black_box(step),
                            black_box(enc),
                        ));
                    }
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("unroll_full", cluster.label),
            &(&encoder, depth),
            |b, &(enc, d)| {
                b.iter(|| {
                    let enc_clone = BmcEncoder::new(
                        cluster.num_services,
                        cluster.version_counts(),
                    );
                    let mut unroll = BmcUnrolling::new(enc_clone);
                    unroll.unroll_to_depth(black_box(d));
                    black_box(unroll.all_clauses().len());
                });
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark group 3: Interval encoding vs naive clause count
// ---------------------------------------------------------------------------

fn bench_interval_vs_naive(c: &mut Criterion) {
    let mut group = c.benchmark_group("interval_vs_naive");
    group.sample_size(50);

    for cluster in &[&SMALL, &MEDIUM, &LARGE] {
        let v = cluster.versions_per_service;
        let delta = v / 3 + 1; // roughly 1/3 window

        // Naive: enumerate all compatible pairs.
        group.bench_with_input(
            BenchmarkId::new("naive_enumeration", cluster.label),
            &(v, delta),
            |b, &(v, delta)| {
                b.iter(|| {
                    let mut count = 0usize;
                    for a in 0..v {
                        for bb in 0..v {
                            if (a as isize - bb as isize).unsigned_abs() <= delta {
                                count += 1;
                            }
                        }
                    }
                    black_box(count)
                });
            },
        );

        // Interval: binary encoding + range clauses.
        group.bench_with_input(
            BenchmarkId::new("interval_encoding", cluster.label),
            &(v, delta),
            |b, &(v, delta)| {
                b.iter(|| {
                    let pred = cluster.interval_predicate(delta);
                    let compressed = pred.compress();
                    black_box(compressed)
                });
            },
        );

        // Clause-count comparison via IntervalCompressor.
        group.bench_with_input(
            BenchmarkId::new("compressor_analysis", cluster.label),
            &(v, delta),
            |b, &(v, delta)| {
                b.iter(|| {
                    let mat = cluster.compatibility_matrix(delta);
                    let mut compressor = IntervalCompressor::new();
                    let analysis = compressor.analyze(&mat);
                    black_box(analysis)
                });
            },
        );

        // Binary encoding with comparator clauses.
        group.bench_with_input(
            BenchmarkId::new("binary_comparator", cluster.label),
            &v,
            |b, &v| {
                b.iter(|| {
                    let enc_a = BinaryEncoding::new(black_box(v), 1);
                    let enc_b = BinaryEncoding::new(black_box(v), 100);
                    let mut cmp = Comparator::new(200);
                    let clauses = cmp.less_than_or_equal(&enc_a, &enc_b);
                    black_box(clauses.len())
                });
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark group 4: Envelope computation
// ---------------------------------------------------------------------------

fn bench_envelope(c: &mut Criterion) {
    let mut group = c.benchmark_group("envelope_computation");
    group.sample_size(20);

    // Envelope benchmarks test the state-space analysis routines that
    // determine rollback coverage and points-of-no-return.
    //
    // We benchmark the *encoding-side* of envelope computation: building
    // the reachability structures that feed into the safety envelope.

    for cluster in &[&SMALL, &MEDIUM] {
        let n = cluster.num_services;
        let vps = cluster.version_counts();
        let start = cluster.start_state();
        let target = cluster.target_state();
        let depth = CompletenessChecker::completeness_bound(&start, &target);

        // Forward reachability: encode BMC unrolling up to completeness bound.
        group.bench_with_input(
            BenchmarkId::new("forward_reachability_encoding", cluster.label),
            &(n, &vps, &start, &target, depth),
            |b, &(n, vps, start, target, depth)| {
                b.iter(|| {
                    let enc = BmcEncoder::new(n, vps.clone());
                    let init_clauses = enc.encode_initial_state(start);
                    let mut all_clauses = init_clauses;
                    for step in 0..depth {
                        let trans = enc.encode_transition(step);
                        all_clauses.extend(trans);
                    }
                    let target_clauses = enc.encode_target_state(target, depth);
                    all_clauses.extend(target_clauses);
                    black_box(all_clauses.len())
                });
            },
        );

        // Backward reachability: encode from target backwards.
        group.bench_with_input(
            BenchmarkId::new("backward_reachability_encoding", cluster.label),
            &(n, &vps, &start, &target, depth),
            |b, &(n, vps, start, target, depth)| {
                b.iter(|| {
                    let enc = BmcEncoder::new(n, vps.clone());
                    // Encode from target as "initial" and start as "target"
                    // to compute backward reachability.
                    let init_clauses = enc.encode_initial_state(target);
                    let mut all_clauses = init_clauses;
                    for step in 0..depth {
                        let trans = enc.encode_transition(step);
                        all_clauses.extend(trans);
                    }
                    let tgt_clauses = enc.encode_target_state(start, depth);
                    all_clauses.extend(tgt_clauses);
                    black_box(all_clauses.len())
                });
            },
        );

        // Completeness bound computation (very cheap, but included for
        // thoroughness).
        group.bench_with_input(
            BenchmarkId::new("completeness_bound", cluster.label),
            &(&start, &target),
            |b, &(s, t)| {
                b.iter(|| {
                    black_box(CompletenessChecker::completeness_bound(
                        black_box(s),
                        black_box(t),
                    ))
                });
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark group 5: Treewidth decomposition
// ---------------------------------------------------------------------------

fn bench_treewidth(c: &mut Criterion) {
    let mut group = c.benchmark_group("treewidth_decomposition");
    group.sample_size(30);

    for cluster in &[&SMALL, &MEDIUM, &LARGE] {
        let adj = cluster.dependency_adjacency();

        group.bench_with_input(
            BenchmarkId::new("upper_bound", cluster.label),
            &adj,
            |b, adj| {
                b.iter(|| {
                    black_box(TreewidthComputer::upper_bound(black_box(adj)));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("lower_bound", cluster.label),
            &adj,
            |b, adj| {
                b.iter(|| {
                    black_box(TreewidthComputer::lower_bound(black_box(adj)));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("min_degree_ordering", cluster.label),
            &adj,
            |b, adj| {
                b.iter(|| {
                    black_box(TreewidthComputer::min_degree_ordering(black_box(adj)));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("min_fill_ordering", cluster.label),
            &adj,
            |b, adj| {
                b.iter(|| {
                    black_box(TreewidthComputer::min_fill_ordering(black_box(adj)));
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("full_decomposition", cluster.label),
            &adj,
            |b, adj| {
                b.iter(|| {
                    let td = TreewidthComputer::compute_decomposition(black_box(adj));
                    black_box(td.bag_count());
                });
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark group 6: Prefilter effectiveness
// ---------------------------------------------------------------------------

fn bench_prefilter(c: &mut Criterion) {
    let mut group = c.benchmark_group("prefilter");
    group.sample_size(50);

    for cluster in &[&SMALL, &MEDIUM, &LARGE] {
        let n = cluster.num_services;
        let v = cluster.versions_per_service;
        let delta = v / 3 + 1;

        // Bitmap construction and query.
        group.bench_with_input(
            BenchmarkId::new("bitmap_construct", cluster.label),
            &(v, delta),
            |b, &(v, delta)| {
                b.iter(|| {
                    let mut bm = CompatibilityBitmap::new(v, v);
                    for a in 0..v {
                        for bb in 0..v {
                            bm.set(
                                a,
                                bb,
                                (a as isize - bb as isize).unsigned_abs() <= delta,
                            );
                        }
                    }
                    black_box(bm.density())
                });
            },
        );

        // Bitmap query: compatible_with.
        group.bench_with_input(
            BenchmarkId::new("bitmap_query", cluster.label),
            &(v, delta),
            |b, &(v, delta)| {
                let mut bm = CompatibilityBitmap::new(v, v);
                for a in 0..v {
                    for bb in 0..v {
                        bm.set(a, bb, (a as isize - bb as isize).unsigned_abs() <= delta);
                    }
                }
                b.iter(|| {
                    for a in 0..v {
                        black_box(bm.compatible_with(a));
                    }
                });
            },
        );

        // PairwisePrefilter feasibility check.
        group.bench_with_input(
            BenchmarkId::new("pairwise_feasibility", cluster.label),
            &(n, v, delta),
            |b, &(n, v, delta)| {
                let pf = cluster.prefilter(delta);
                // Test a mix of feasible and infeasible states.
                let states: Vec<Vec<usize>> = (0..50)
                    .map(|i| {
                        (0..n).map(|s| (s + i) % v).collect()
                    })
                    .collect();
                b.iter(|| {
                    let mut feasible_count = 0usize;
                    for st in &states {
                        if pf.is_potentially_feasible(black_box(st)) {
                            feasible_count += 1;
                        }
                    }
                    black_box(feasible_count)
                });
            },
        );

        // Arc consistency propagation.
        group.bench_with_input(
            BenchmarkId::new("arc_consistency", cluster.label),
            &(n, v, delta),
            |b, &(n, v, delta)| {
                let vps: Vec<usize> = vec![v; n];
                let mut constraints: Vec<(usize, usize, Vec<(usize, usize)>)> = Vec::new();
                for i in 0..n.saturating_sub(1) {
                    let mut pairs = Vec::new();
                    for a in 0..v {
                        for bb in 0..v {
                            if (a as isize - bb as isize).unsigned_abs() <= delta {
                                pairs.push((a, bb));
                            }
                        }
                    }
                    constraints.push((i, i + 1, pairs));
                }

                b.iter(|| {
                    let mut ac = ArcConsistency::new(n, &vps, &constraints);
                    let result = ac.propagate();
                    black_box(result)
                });
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark group 7: Full pipeline (encoding phase)
// ---------------------------------------------------------------------------

fn bench_full_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_pipeline");
    group.sample_size(10);

    for cluster in &[&SMALL, &MEDIUM] {
        let n = cluster.num_services;
        let v = cluster.versions_per_service;
        let delta = v / 3 + 1;

        group.bench_with_input(
            BenchmarkId::new("encode_and_prefilter", cluster.label),
            &(n, v, delta),
            |b, &(n, v, delta)| {
                b.iter(|| {
                    // Phase 1: Build prefilter.
                    let pf = cluster.prefilter(delta);

                    // Phase 2: Build BMC encoding.
                    let vps = vec![v; n];
                    let enc = BmcEncoder::new(n, vps);
                    let start = cluster.start_state();
                    let target = cluster.target_state();
                    let depth = CompletenessChecker::completeness_bound(&start, &target);

                    // Phase 3: Encode full BMC unrolling.
                    let mut unroll = BmcUnrolling::new(enc);
                    unroll.unroll_to_depth(depth);
                    let clause_count = unroll.all_clauses().len();

                    // Phase 4: Build adjacency and compute treewidth.
                    let adj = cluster.dependency_adjacency();
                    let tw = TreewidthComputer::upper_bound(&adj);

                    // Phase 5: Check feasibility of start and target.
                    let start_ok = pf.is_potentially_feasible(&start);
                    let target_ok = pf.is_potentially_feasible(&target);

                    black_box((clause_count, tw, start_ok, target_ok))
                });
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Register all benchmark groups
// ---------------------------------------------------------------------------

criterion_group!(
    benches,
    bench_graph_construction,
    bench_encoding,
    bench_interval_vs_naive,
    bench_envelope,
    bench_treewidth,
    bench_prefilter,
    bench_full_pipeline,
);
criterion_main!(benches);
