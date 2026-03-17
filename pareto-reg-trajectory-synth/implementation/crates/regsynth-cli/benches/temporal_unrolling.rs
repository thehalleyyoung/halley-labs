//! Benchmark: Temporal constraint unrolling at varying time horizons.
//!
//! Measures the performance of encoding temporal regulatory obligations
//! across multiple time steps, as used in trajectory optimization.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use regsynth_temporal::{
    eu_ai_act_schedule, PhaseInSchedule, Milestone,
    RegulatoryTransitionSystem, RegulatoryState, RegulatoryEvent, Transition,
};
use regsynth_pareto::{
    cost_model::{CostModel, ObligationCostEstimate},
    strategy_repr::StrategyBitVec,
};

use chrono::NaiveDate;
use std::collections::BTreeSet;
use rand::Rng;
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Build a phase-in schedule with `n_milestones` milestones,
/// each adding `obls_per_milestone` obligations.
fn build_synthetic_schedule(n_milestones: usize, obls_per_milestone: usize) -> PhaseInSchedule {
    let mut schedule = PhaseInSchedule::new("Synthetic-Framework");
    let base_date = NaiveDate::from_ymd_opt(2025, 1, 1).unwrap();

    let mut obl_counter = 0;
    for i in 0..n_milestones {
        let date = base_date + chrono::Duration::days((i as i64 + 1) * 180);
        let obligations: BTreeSet<String> = (0..obls_per_milestone)
            .map(|_| {
                obl_counter += 1;
                format!("obl-{:04}", obl_counter)
            })
            .collect();
        schedule.add_milestone(Milestone::new(
            date,
            format!("Milestone-{}", i + 1),
            obligations,
        ));
    }

    schedule
}

/// Build a transition system from a phase-in schedule.
fn build_transition_system(schedule: &PhaseInSchedule) -> RegulatoryTransitionSystem {
    let mut rts = RegulatoryTransitionSystem::new();

    let s0 = RegulatoryState::new("s0");
    rts.add_state(s0);

    for (i, ms) in schedule.milestones().iter().enumerate() {
        let sid = format!("s{}", i + 1);
        let state = RegulatoryState::with_obligations(
            sid.clone(),
            schedule.obligations_active_at(&ms.date),
        ).with_timestamp(ms.date);
        rts.add_state(state);

        let from = if i == 0 { "s0".to_string() } else { format!("s{}", i) };
        let _ = rts.add_transition(Transition {
            from,
            to: sid,
            event: RegulatoryEvent::PhaseIn {
                milestone: ms.label.clone(),
                date: ms.date,
            },
        });
    }

    rts
}

/// Build a cost model with `n` obligations.
fn build_cost_model(n: usize, seed: u64) -> CostModel {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut model = CostModel::new();
    for i in 0..n {
        model.add_obligation(
            ObligationCostEstimate::new(format!("obl-{:04}", i + 1))
                .with_financial_cost(rng.gen_range(10_000.0..500_000.0))
                .with_time(rng.gen_range(1.0..12.0))
                .with_risk(rng.gen_range(0.1..0.9))
                .with_complexity(rng.gen_range(5.0..95.0))
                .with_penalty(rng.gen_range(50_000.0..500_000.0)),
        );
    }
    model
}

/// Benchmark building phase-in schedules at different scales.
fn bench_schedule_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("temporal_schedule_construction");
    for &(milestones, obls_per) in &[(4, 5), (8, 10), (16, 20), (32, 40)] {
        group.bench_function(
            BenchmarkId::new("schedule", format!("{}m_{}o", milestones, obls_per)),
            |b| {
                b.iter(|| build_synthetic_schedule(milestones, obls_per));
            },
        );
    }
    group.finish();
}

/// Benchmark transition system construction.
fn bench_transition_system_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("temporal_transition_system");
    for &(milestones, obls_per) in &[(4, 5), (8, 10), (16, 20)] {
        let schedule = build_synthetic_schedule(milestones, obls_per);
        group.bench_with_input(
            BenchmarkId::new("rts", format!("{}m_{}o", milestones, obls_per)),
            &schedule,
            |b, sched| {
                b.iter(|| build_transition_system(sched));
            },
        );
    }
    group.finish();
}

/// Benchmark obligation activation queries over time.
fn bench_obligation_activation(c: &mut Criterion) {
    let mut group = c.benchmark_group("temporal_activation_query");
    for &(milestones, obls_per) in &[(4, 5), (16, 20), (64, 50)] {
        let schedule = build_synthetic_schedule(milestones, obls_per);
        let base_date = NaiveDate::from_ymd_opt(2025, 1, 1).unwrap();
        let query_dates: Vec<NaiveDate> = (0..100)
            .map(|i| base_date + chrono::Duration::days(i * 30))
            .collect();

        group.bench_with_input(
            BenchmarkId::new("query", format!("{}m_{}o", milestones, obls_per)),
            &(schedule, query_dates),
            |b, (sched, dates)| {
                b.iter(|| {
                    let mut total = 0usize;
                    for date in dates {
                        total += sched.obligations_active_at(date).len();
                    }
                    total
                });
            },
        );
    }
    group.finish();
}

/// Benchmark cost model evaluation at different strategy sizes.
fn bench_cost_model_evaluation(c: &mut Criterion) {
    let mut group = c.benchmark_group("temporal_cost_evaluation");
    for &n_obls in &[10, 50, 200, 1000] {
        let model = build_cost_model(n_obls, 42);
        let strategy = StrategyBitVec::from_active(
            n_obls,
            &(0..n_obls / 2).collect::<Vec<_>>(),
        );

        group.bench_with_input(
            BenchmarkId::new("evaluate", n_obls),
            &(model, strategy),
            |b, (m, s)| {
                b.iter(|| m.evaluate(s));
            },
        );
    }
    group.finish();
}

/// Benchmark the EU AI Act schedule specifically.
fn bench_eu_ai_act_schedule(c: &mut Criterion) {
    let mut group = c.benchmark_group("temporal_eu_ai_act");

    group.bench_function("schedule_build", |b| {
        b.iter(|| eu_ai_act_schedule());
    });

    let schedule = eu_ai_act_schedule();
    group.bench_function("full_activation_check", |b| {
        b.iter(|| {
            let base = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
            let mut total = 0usize;
            for day in 0..1500 {
                let date = base + chrono::Duration::days(day);
                total += schedule.obligations_active_at(&date).len();
            }
            total
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_schedule_construction,
    bench_transition_system_construction,
    bench_obligation_activation,
    bench_cost_model_evaluation,
    bench_eu_ai_act_schedule,
);
criterion_main!(benches);
