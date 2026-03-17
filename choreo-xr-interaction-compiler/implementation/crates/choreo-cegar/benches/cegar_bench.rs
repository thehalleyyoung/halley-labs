//! Criterion benchmarks for the choreo-cegar verification engine.
//!
//! Covers BDD operations, spatial partition refinement, geometric pruning,
//! product-automaton construction, and end-to-end CEGAR iterations.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use choreo_cegar::{
    AbstractBlock, AbstractBlockId, CEGARConfig, CEGARStatistics,
    GeometricPruner, ModelChecker, SpatialPartition,
};
use choreo_cegar::model_checker::{BDDSet, BDDTransitionRelation};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Lightweight AABB stand-in for benchmarks (mirrors choreo_types::geometry::AABB).
#[derive(Clone, Debug)]
struct AABB {
    min: [f64; 3],
    max: [f64; 3],
}

impl AABB {
    fn new(cx: f64, cy: f64, cz: f64, half: f64) -> Self {
        Self {
            min: [cx - half, cy - half, cz - half],
            max: [cx + half, cy + half, cz + half],
        }
    }

    fn overlaps(&self, other: &AABB) -> bool {
        self.min[0] <= other.max[0]
            && self.max[0] >= other.min[0]
            && self.min[1] <= other.max[1]
            && self.max[1] >= other.min[1]
            && self.min[2] <= other.max[2]
            && self.max[2] >= other.min[2]
    }

    fn volume(&self) -> f64 {
        (self.max[0] - self.min[0])
            * (self.max[1] - self.min[1])
            * (self.max[2] - self.min[2])
    }
}

// ---------------------------------------------------------------------------
// BDD benchmarks
// ---------------------------------------------------------------------------

fn bench_bdd_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("bdd_construction");
    for n in [10u64, 50, 200, 1000] {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter(|| {
                let mut bdd = BDDSet::new();
                for i in 0..n {
                    bdd.insert(i);
                }
                black_box(&bdd);
            });
        });
    }
    group.finish();
}

fn bench_bdd_set_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("bdd_set_operations");

    for n in [100u64, 500, 2000] {
        let a = BDDSet::from_elements((0..n).collect());
        let b = BDDSet::from_elements((n / 2..n + n / 2).collect());

        group.bench_with_input(BenchmarkId::new("union", n), &(&a, &b), |bench, (a, b)| {
            bench.iter(|| black_box(a.union(b)));
        });

        group.bench_with_input(
            BenchmarkId::new("intersection", n),
            &(&a, &b),
            |bench, (a, b)| {
                bench.iter(|| black_box(a.intersection(b)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("difference", n),
            &(&a, &b),
            |bench, (a, b)| {
                bench.iter(|| black_box(a.difference(b)));
            },
        );
    }
    group.finish();
}

fn bench_bdd_transition_image(c: &mut Criterion) {
    let mut group = c.benchmark_group("bdd_transition_image");

    for n in [50u64, 200, 1000] {
        let mut rel = BDDTransitionRelation::new();
        for i in 0..n {
            rel.add_transition(i, (i + 1) % n);
            if i + 3 < n {
                rel.add_transition(i, i + 3);
            }
        }
        let frontier = BDDSet::from_elements((0..n / 4).collect());

        group.bench_with_input(BenchmarkId::from_parameter(n), &(&rel, &frontier), |b, (rel, frontier)| {
            b.iter(|| black_box(rel.image(frontier)));
        });
    }
    group.finish();
}

fn bench_bdd_transition_preimage(c: &mut Criterion) {
    let mut group = c.benchmark_group("bdd_transition_preimage");

    for n in [50u64, 200, 1000] {
        let mut rel = BDDTransitionRelation::new();
        for i in 0..n {
            rel.add_transition(i, (i + 1) % n);
        }
        let target = BDDSet::from_elements(vec![0, n / 2]);

        group.bench_with_input(BenchmarkId::from_parameter(n), &(&rel, &target), |b, (rel, target)| {
            b.iter(|| black_box(rel.preimage(target)));
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Spatial partition benchmarks
// ---------------------------------------------------------------------------

fn bench_partition_refinement(c: &mut Criterion) {
    let mut group = c.benchmark_group("partition_refinement");

    for depth in [4u32, 6, 8] {
        group.bench_with_input(BenchmarkId::from_parameter(depth), &depth, |b, &depth| {
            b.iter(|| {
                let mut partition = SpatialPartition::new();
                let mut next_id = 1u64;
                // Simulate iterative octree-style splitting
                for _ in 0..depth {
                    let ids: Vec<_> = partition.block_ids().collect();
                    for id in ids {
                        partition.refine_block(id);
                    }
                }
                black_box(partition.block_count());
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Geometric pruning simulation benchmark
// ---------------------------------------------------------------------------

fn bench_aabb_overlap_pruning(c: &mut Criterion) {
    let mut group = c.benchmark_group("aabb_overlap_pruning");

    for n in [100usize, 500, 2000] {
        let boxes: Vec<AABB> = (0..n)
            .map(|i| {
                let x = (i as f64) * 0.3;
                AABB::new(x, 0.0, 0.0, 0.5)
            })
            .collect();
        let query = AABB::new(5.0, 0.0, 0.0, 3.0);

        group.bench_with_input(BenchmarkId::from_parameter(n), &(&boxes, &query), |b, (boxes, query)| {
            b.iter(|| {
                let count = boxes.iter().filter(|b| b.overlaps(query)).count();
                black_box(count);
            });
        });
    }
    group.finish();
}

fn bench_volume_pruning_ratio(c: &mut Criterion) {
    let mut group = c.benchmark_group("volume_pruning_ratio");

    for n in [100usize, 500, 2000] {
        let boxes: Vec<AABB> = (0..n)
            .map(|i| {
                let x = (i as f64) * 0.3;
                AABB::new(x, 0.0, 0.0, 0.5)
            })
            .collect();
        let region = AABB::new(5.0, 0.0, 0.0, 3.0);

        group.bench_with_input(BenchmarkId::from_parameter(n), &(&boxes, &region), |b, (boxes, region)| {
            b.iter(|| {
                let total_vol: f64 = boxes.iter().map(|b| b.volume()).sum();
                let pruned_vol: f64 = boxes
                    .iter()
                    .filter(|b| !b.overlaps(region))
                    .map(|b| b.volume())
                    .sum();
                black_box(pruned_vol / total_vol);
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// State-machine stepping simulation
// ---------------------------------------------------------------------------

fn bench_automaton_stepping(c: &mut Criterion) {
    let mut group = c.benchmark_group("automaton_stepping");

    for n_states in [10u64, 50, 200] {
        // Build a simple ring automaton + skip transitions
        let mut rel = BDDTransitionRelation::new();
        for s in 0..n_states {
            rel.add_transition(s, (s + 1) % n_states);
            rel.add_transition(s, (s + 2) % n_states);
        }

        group.bench_with_input(
            BenchmarkId::new("reachability_fixpoint", n_states),
            &(n_states, &rel),
            |b, &(n_states, ref rel)| {
                b.iter(|| {
                    let mut frontier = BDDSet::from_elements(vec![0]);
                    let mut visited = BDDSet::new();
                    let mut steps = 0u32;
                    loop {
                        let new_states = frontier.difference(&visited);
                        if new_states.is_empty() {
                            break;
                        }
                        visited = visited.union(&new_states);
                        frontier = rel.image(&new_states);
                        steps += 1;
                    }
                    black_box((visited.len(), steps));
                });
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// CEGAR statistics tracking overhead
// ---------------------------------------------------------------------------

fn bench_statistics_recording(c: &mut Criterion) {
    c.bench_function("cegar_statistics_100_iterations", |b| {
        b.iter(|| {
            let mut stats = CEGARStatistics::new();
            for i in 0..100 {
                stats.record_iteration(i as u64 + 10, i as u64 * 2);
            }
            black_box(stats.average_abstract_states());
            black_box(stats.convergence_rate());
        });
    });
}

// ---------------------------------------------------------------------------
// Groups
// ---------------------------------------------------------------------------

criterion_group!(
    cegar_benches,
    bench_bdd_construction,
    bench_bdd_set_operations,
    bench_bdd_transition_image,
    bench_bdd_transition_preimage,
    bench_partition_refinement,
    bench_aabb_overlap_pruning,
    bench_volume_pruning_ratio,
    bench_automaton_stepping,
    bench_statistics_recording,
);
criterion_main!(cegar_benches);
