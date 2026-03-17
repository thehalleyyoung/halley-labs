//! Benchmark: Multi-jurisdictional regulatory compliance trajectory synthesis.
//!
//! Exercises the full pipeline: create regulatory obligation sets across
//! 2–5 jurisdictions (10–50 obligations each), run Pareto synthesis,
//! detect temporal conflicts, and compare against a greedy sequential baseline.
//!
//! Metrics:
//! - Synthesis time (Pareto frontier construction)
//! - Number of Pareto-optimal trajectories found
//! - Conflict detection count and cycle-detection time
//! - Greedy baseline: satisfy one jurisdiction at a time (faster but misses optima)

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use regsynth_pareto::{
    cost_model::{CostModel, ObligationCostEstimate},
    dominance::dominates,
    frontier::ParetoFrontier,
    metrics::hypervolume_indicator,
    strategy_repr::{greedy_strategy, StrategyBitVec},
    trajectory::{ComplianceTrajectory, TrajectoryConfig, TrajectoryOptimizer},
    CostVector,
};
use regsynth_temporal::conflict_detector::{
    TcgEdgeKind, TemporalConflictDetector, TemporalConstraintGraph,
};

// ---------------------------------------------------------------------------
// Jurisdiction descriptions for realistic obligation generation
// ---------------------------------------------------------------------------

struct JurisdictionSpec {
    name: &'static str,
    prefix: &'static str,
    cost_ranges: (f64, f64, f64, f64, f64, f64, f64, f64, f64, f64),
}

const JURISDICTIONS: &[JurisdictionSpec] = &[
    JurisdictionSpec {
        name: "EU-AI-Act",
        prefix: "eu-aia",
        cost_ranges: (30_000.0, 500_000.0, 2.0, 18.0, 0.1, 0.7, 15.0, 90.0, 100_000.0, 2_000_000.0),
    },
    JurisdictionSpec {
        name: "GDPR",
        prefix: "gdpr",
        cost_ranges: (20_000.0, 300_000.0, 1.0, 12.0, 0.1, 0.6, 10.0, 70.0, 50_000.0, 1_500_000.0),
    },
    JurisdictionSpec {
        name: "NIST-AI-RMF",
        prefix: "nist",
        cost_ranges: (10_000.0, 200_000.0, 1.0, 10.0, 0.05, 0.4, 5.0, 60.0, 20_000.0, 500_000.0),
    },
    JurisdictionSpec {
        name: "China-GenAI",
        prefix: "cn-gai",
        cost_ranges: (15_000.0, 250_000.0, 2.0, 14.0, 0.1, 0.5, 10.0, 80.0, 80_000.0, 1_000_000.0),
    },
    JurisdictionSpec {
        name: "ISO-42001",
        prefix: "iso42",
        cost_ranges: (25_000.0, 400_000.0, 3.0, 16.0, 0.05, 0.3, 20.0, 85.0, 30_000.0, 800_000.0),
    },
];

// ---------------------------------------------------------------------------
// Test data generation
// ---------------------------------------------------------------------------

fn build_multi_jurisdiction_model(
    n_jurisdictions: usize,
    obls_per_jurisdiction: usize,
    seed: u64,
) -> (CostModel, Vec<(usize, usize)>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut model = CostModel::new();
    let mut jurisdiction_ranges = Vec::new();

    for j in 0..n_jurisdictions {
        let spec = &JURISDICTIONS[j % JURISDICTIONS.len()];
        let start = j * obls_per_jurisdiction;
        let end = start + obls_per_jurisdiction;

        for i in 0..obls_per_jurisdiction {
            let (fc_min, fc_max, t_min, t_max, r_min, r_max, cx_min, cx_max, p_min, p_max) =
                spec.cost_ranges;
            model.add_obligation(
                ObligationCostEstimate::new(format!("{}-art{}", spec.prefix, i + 1))
                    .with_financial_cost(rng.gen_range(fc_min..fc_max))
                    .with_time(rng.gen_range(t_min..t_max))
                    .with_risk(rng.gen_range(r_min..r_max))
                    .with_complexity(rng.gen_range(cx_min..cx_max))
                    .with_penalty(rng.gen_range(p_min..p_max)),
            );
        }

        jurisdiction_ranges.push((start, end));
    }

    let model = model.with_parallelism_factor(0.7);
    (model, jurisdiction_ranges)
}

fn build_conflict_graph(
    n_jurisdictions: usize,
    obls_per_jurisdiction: usize,
    n_timesteps: usize,
    conflict_density: f64,
    seed: u64,
) -> TemporalConstraintGraph {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut tcg = TemporalConstraintGraph::new();

    let edge_kinds = [
        TcgEdgeKind::ComplianceForces,
        TcgEdgeKind::Prerequisite,
        TcgEdgeKind::ResourceConflict,
    ];

    for j1 in 0..n_jurisdictions {
        let spec1 = &JURISDICTIONS[j1 % JURISDICTIONS.len()];
        for j2 in (j1 + 1)..n_jurisdictions {
            let spec2 = &JURISDICTIONS[j2 % JURISDICTIONS.len()];

            for i1 in 0..obls_per_jurisdiction {
                for i2 in 0..obls_per_jurisdiction {
                    if rng.gen::<f64>() < conflict_density {
                        let t_from = rng.gen_range(0..n_timesteps);
                        let t_to = rng.gen_range(0..n_timesteps);
                        let kind = edge_kinds[rng.gen_range(0..3)].clone();
                        let cost = rng.gen_range(1.0..10.0);

                        tcg.add_dependency(
                            format!("{}-art{}", spec1.prefix, i1 + 1),
                            t_from,
                            format!("{}-art{}", spec2.prefix, i2 + 1),
                            t_to,
                            kind,
                            format!(
                                "{} Art.{} t={} vs {} Art.{} t={}",
                                spec1.name, i1 + 1, t_from, spec2.name, i2 + 1, t_to
                            ),
                            cost,
                        );
                    }
                }
            }
        }
    }

    tcg
}

fn greedy_sequential_compliance(
    _model: &CostModel,
    jurisdiction_ranges: &[(usize, usize)],
    n_obligations: usize,
) -> StrategyBitVec {
    let mut strategy = StrategyBitVec::new(n_obligations);
    for &(start, end) in jurisdiction_ranges {
        for i in start..end {
            strategy.set(i, true);
        }
    }
    strategy
}

fn greedy_weighted_baseline(model: &CostModel, n_obligations: usize) -> StrategyBitVec {
    let weights = vec![0.3, 0.2, 0.3, 0.2];
    greedy_strategy(n_obligations, &weights, |s| model.evaluate(s))
}

// ---------------------------------------------------------------------------
// Scenario configuration
// ---------------------------------------------------------------------------

struct ScenarioConfig {
    label: &'static str,
    n_jurisdictions: usize,
    obls_per_jurisdiction: usize,
    n_timesteps: usize,
    conflict_density: f64,
}

const SCENARIOS: &[ScenarioConfig] = &[
    ScenarioConfig {
        label: "small_2j_10o",
        n_jurisdictions: 2,
        obls_per_jurisdiction: 10,
        n_timesteps: 3,
        conflict_density: 0.10,
    },
    ScenarioConfig {
        label: "medium_3j_20o",
        n_jurisdictions: 3,
        obls_per_jurisdiction: 20,
        n_timesteps: 4,
        conflict_density: 0.08,
    },
    ScenarioConfig {
        label: "large_5j_30o",
        n_jurisdictions: 5,
        obls_per_jurisdiction: 30,
        n_timesteps: 4,
        conflict_density: 0.05,
    },
    ScenarioConfig {
        label: "xlarge_5j_50o",
        n_jurisdictions: 5,
        obls_per_jurisdiction: 50,
        n_timesteps: 4,
        conflict_density: 0.03,
    },
];

// ---------------------------------------------------------------------------
// Benchmark: Pareto Synthesis
// ---------------------------------------------------------------------------

fn bench_pareto_synthesis(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_jurisdiction_pareto_synthesis");
    group.sample_size(10);

    for scenario in SCENARIOS {
        let n_obls = scenario.n_jurisdictions * scenario.obls_per_jurisdiction;
        let (model, _ranges) = build_multi_jurisdiction_model(
            scenario.n_jurisdictions,
            scenario.obls_per_jurisdiction,
            42,
        );

        let mut rng = StdRng::seed_from_u64(123);
        let n_candidates = (n_obls * 10).min(2000);
        let candidates: Vec<StrategyBitVec> = (0..n_candidates)
            .map(|_| {
                let bits: Vec<bool> = (0..n_obls).map(|_| rng.gen_bool(0.5)).collect();
                StrategyBitVec::from_bits(bits)
            })
            .collect();

        let input = (model, candidates);
        group.bench_with_input(
            BenchmarkId::new("frontier", scenario.label),
            &input,
            |b, input| {
                let (ref mdl, ref cands) = *input;
                b.iter(|| {
                    let mut frontier: ParetoFrontier<usize> = ParetoFrontier::new(4);
                    for (i, strategy) in cands.iter().enumerate() {
                        let cost = mdl.evaluate(strategy);
                        frontier.add_point(i, cost);
                    }
                    frontier.size()
                });
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Temporal Conflict Detection
// ---------------------------------------------------------------------------

fn bench_temporal_conflict_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_jurisdiction_conflict_detection");
    group.sample_size(10);

    for scenario in SCENARIOS {
        let tcg = build_conflict_graph(
            scenario.n_jurisdictions,
            scenario.obls_per_jurisdiction,
            scenario.n_timesteps,
            scenario.conflict_density,
            42,
        );

        group.bench_with_input(
            BenchmarkId::new("detect_cycles", scenario.label),
            &tcg,
            |b, graph| {
                b.iter(|| {
                    let detector = TemporalConflictDetector::new(graph.clone())
                        .with_max_cycle_length(8);
                    let certs = detector.detect_conflicts();
                    certs.len()
                });
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Trajectory Optimization (Pareto vs Greedy)
// ---------------------------------------------------------------------------

fn bench_trajectory_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_jurisdiction_trajectory_optimization");
    group.sample_size(10);

    let traj_scenarios = &[
        ScenarioConfig {
            label: "traj_2j_5o",
            n_jurisdictions: 2,
            obls_per_jurisdiction: 5,
            n_timesteps: 3,
            conflict_density: 0.10,
        },
        ScenarioConfig {
            label: "traj_2j_8o",
            n_jurisdictions: 2,
            obls_per_jurisdiction: 8,
            n_timesteps: 3,
            conflict_density: 0.08,
        },
        ScenarioConfig {
            label: "traj_3j_4o",
            n_jurisdictions: 3,
            obls_per_jurisdiction: 4,
            n_timesteps: 4,
            conflict_density: 0.10,
        },
    ];

    for scenario in traj_scenarios {
        let n_obls = scenario.n_jurisdictions * scenario.obls_per_jurisdiction;
        let timestep_models: Vec<CostModel> = (0..scenario.n_timesteps)
            .map(|t| {
                let (model, _) = build_multi_jurisdiction_model(
                    scenario.n_jurisdictions,
                    scenario.obls_per_jurisdiction,
                    42 + t as u64,
                );
                model
            })
            .collect();

        // Pareto trajectory optimization
        let pareto_input = (timestep_models.clone(), n_obls);
        group.bench_with_input(
            BenchmarkId::new("pareto_trajectory", scenario.label),
            &pareto_input,
            |b, input| {
                let (ref models, n) = *input;
                b.iter(|| {
                    let optimizer = TrajectoryOptimizer::new(TrajectoryConfig {
                        max_weight_vectors: 20,
                        transition_budget: 4,
                        epsilon: 0.05,
                        ..TrajectoryConfig::default()
                    });
                    let frontier = optimizer.optimize_trajectory(
                        models,
                        |_t, _s| true,
                        n,
                    );
                    frontier.size()
                });
            },
        );

        // Greedy per-timestep baseline
        let greedy_input = (timestep_models.clone(), n_obls);
        group.bench_with_input(
            BenchmarkId::new("greedy_per_timestep", scenario.label),
            &greedy_input,
            |b, input| {
                let (ref models, n) = *input;
                b.iter(|| {
                    let weights = vec![0.3, 0.2, 0.3, 0.2];
                    let strategies: Vec<StrategyBitVec> = models
                        .iter()
                        .map(|m| greedy_strategy(n, &weights, |s| m.evaluate(s)))
                        .collect();
                    let costs: Vec<CostVector> = strategies
                        .iter()
                        .zip(models.iter())
                        .map(|(s, m)| m.evaluate(s))
                        .collect();
                    let agg = costs.iter().skip(1).fold(costs[0].clone(), |a, c| a.add(c));
                    ComplianceTrajectory::new(strategies, costs, agg)
                });
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Full Pipeline
// ---------------------------------------------------------------------------

fn bench_full_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_jurisdiction_full_pipeline");
    group.sample_size(10);

    for scenario in SCENARIOS {
        let n_obls = scenario.n_jurisdictions * scenario.obls_per_jurisdiction;
        let (model, ranges) = build_multi_jurisdiction_model(
            scenario.n_jurisdictions,
            scenario.obls_per_jurisdiction,
            42,
        );
        let tcg = build_conflict_graph(
            scenario.n_jurisdictions,
            scenario.obls_per_jurisdiction,
            scenario.n_timesteps,
            scenario.conflict_density,
            42,
        );

        let input = (model, tcg, ranges, n_obls);
        group.bench_with_input(
            BenchmarkId::new("pipeline", scenario.label),
            &input,
            |b, input| {
                let (ref mdl, ref graph, ref rngs, n) = *input;
                b.iter(|| {
                    // 1. Pareto frontier from sampled strategies
                    let mut rng = StdRng::seed_from_u64(77);
                    let mut frontier: ParetoFrontier<StrategyBitVec> = ParetoFrontier::new(4);
                    let n_samples = (n * 10).min(2000);
                    for _ in 0..n_samples {
                        let bits: Vec<bool> = (0..n).map(|_| rng.gen_bool(0.5)).collect();
                        let s = StrategyBitVec::from_bits(bits);
                        let cost = mdl.evaluate(&s);
                        frontier.add_point(s, cost);
                    }

                    // 2. Temporal conflict detection
                    let detector = TemporalConflictDetector::new(graph.clone())
                        .with_max_cycle_length(8);
                    let conflicts = detector.detect_conflicts();

                    // 3. Greedy baseline for comparison
                    let greedy = greedy_sequential_compliance(mdl, rngs, n);
                    let greedy_cost = mdl.evaluate(&greedy);

                    (frontier.size(), conflicts.len(), greedy_cost)
                });
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Greedy vs Pareto quality comparison
// ---------------------------------------------------------------------------

fn bench_greedy_vs_pareto_quality(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_jurisdiction_quality_comparison");
    group.sample_size(10);

    for scenario in SCENARIOS {
        let n_obls = scenario.n_jurisdictions * scenario.obls_per_jurisdiction;
        let (model, ranges) = build_multi_jurisdiction_model(
            scenario.n_jurisdictions,
            scenario.obls_per_jurisdiction,
            42,
        );

        let input = (model, ranges, n_obls);
        group.bench_with_input(
            BenchmarkId::new("quality_compare", scenario.label),
            &input,
            |b, input| {
                let (ref mdl, ref rngs, n) = *input;
                b.iter(|| {
                    let reference = CostVector::regulatory(2_000_000.0, 24.0, 1.0, 100.0);

                    // Pareto frontier
                    let mut rng = StdRng::seed_from_u64(77);
                    let mut frontier: ParetoFrontier<usize> = ParetoFrontier::new(4);
                    let n_samples = (n * 10).min(2000);
                    for i in 0..n_samples {
                        let bits: Vec<bool> = (0..n).map(|_| rng.gen_bool(0.5)).collect();
                        let s = StrategyBitVec::from_bits(bits);
                        let cost = mdl.evaluate(&s);
                        frontier.add_point(i, cost);
                    }

                    let pareto_costs: Vec<CostVector> =
                        frontier.entries().iter().map(|e| e.cost.clone()).collect();
                    let pareto_hv = hypervolume_indicator(&pareto_costs, &reference);

                    // Greedy sequential baseline
                    let greedy_seq = greedy_sequential_compliance(mdl, rngs, n);
                    let greedy_cost = mdl.evaluate(&greedy_seq);
                    let greedy_hv = hypervolume_indicator(&[greedy_cost.clone()], &reference);

                    // Greedy weighted baseline
                    let greedy_w = greedy_weighted_baseline(mdl, n);
                    let greedy_w_cost = mdl.evaluate(&greedy_w);
                    let greedy_w_hv = hypervolume_indicator(&[greedy_w_cost], &reference);

                    // Check if greedy is dominated by any Pareto point
                    let greedy_dominated = pareto_costs
                        .iter()
                        .any(|p| dominates(p, &greedy_cost));

                    (frontier.size(), pareto_hv, greedy_hv, greedy_w_hv, greedy_dominated)
                });
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark: Scaling analysis
// ---------------------------------------------------------------------------

fn bench_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_jurisdiction_scaling");
    group.sample_size(10);

    for &obls_per in &[10usize, 20, 30, 50] {
        let n_jurisdictions = 3usize;
        let n_obls = n_jurisdictions * obls_per;
        let (model, _) = build_multi_jurisdiction_model(n_jurisdictions, obls_per, 42);

        let input = (model, n_obls);
        group.bench_with_input(
            BenchmarkId::new("scale_obligations", format!("3j_{}o", obls_per)),
            &input,
            |b, input| {
                let (ref mdl, n) = *input;
                b.iter(|| {
                    let mut rng = StdRng::seed_from_u64(77);
                    let mut frontier: ParetoFrontier<usize> = ParetoFrontier::new(4);
                    for i in 0..(n * 10).min(2000) {
                        let bits: Vec<bool> = (0..n).map(|_| rng.gen_bool(0.5)).collect();
                        let s = StrategyBitVec::from_bits(bits);
                        let cost = mdl.evaluate(&s);
                        frontier.add_point(i, cost);
                    }
                    frontier.size()
                });
            },
        );
    }

    for &n_j in &[2usize, 3, 4, 5] {
        let obls_per = 15usize;
        let n_obls = n_j * obls_per;
        let (model, _) = build_multi_jurisdiction_model(n_j, obls_per, 42);

        let input = (model, n_obls);
        group.bench_with_input(
            BenchmarkId::new("scale_jurisdictions", format!("{}j_15o", n_j)),
            &input,
            |b, input| {
                let (ref mdl, n) = *input;
                b.iter(|| {
                    let mut rng = StdRng::seed_from_u64(77);
                    let mut frontier: ParetoFrontier<usize> = ParetoFrontier::new(4);
                    for i in 0..(n * 10).min(2000) {
                        let bits: Vec<bool> = (0..n).map(|_| rng.gen_bool(0.5)).collect();
                        let s = StrategyBitVec::from_bits(bits);
                        let cost = mdl.evaluate(&s);
                        frontier.add_point(i, cost);
                    }
                    frontier.size()
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_pareto_synthesis,
    bench_temporal_conflict_detection,
    bench_trajectory_optimization,
    bench_full_pipeline,
    bench_greedy_vs_pareto_quality,
    bench_scaling,
);
criterion_main!(benches);
