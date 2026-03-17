//! # Temporal Trajectory Optimization
//!
//! This example demonstrates computing Pareto-optimal compliance trajectories
//! over the EU AI Act's phased enforcement timeline.
//!
//! ## Regulatory Context
//!
//! The EU AI Act (Regulation 2024/1689) enters into force in phases:
//!
//! | Date       | Milestone                          | Key Articles     |
//! |------------|-------------------------------------|------------------|
//! | 2025-02-02 | Prohibited AI practices banned      | Art. 5           |
//! | 2025-08-02 | GPAI model obligations apply        | Art. 51-55       |
//! | 2026-08-02 | High-risk AI obligations apply      | Art. 6-49        |
//! | 2027-08-02 | Full enforcement and penalties      | Art. 71, 99      |
//!
//! ## Optimization Problem
//!
//! A company must decide *when* to implement each compliance measure.
//! Early implementation reduces regulatory risk but increases cost;
//! deferred implementation saves money short-term but incurs transition
//! penalties and risk accumulation.
//!
//! The trajectory optimizer uses dynamic programming over the phase-in
//! schedule to find Pareto-optimal sequences of compliance decisions,
//! considering transition costs between strategies at each milestone.
//!
//! ## Demonstrated Features
//!
//! - EU AI Act phase-in schedule construction
//! - Per-timestep cost model evaluation
//! - Trajectory optimization with transition budgets
//! - Pareto frontier over trajectories

use regsynth_types::{
    ArticleRef, FormalizabilityGrade, Jurisdiction, ObligationKind, RiskLevel,
    TemporalInterval,
};
use regsynth_temporal::{
    eu_ai_act_schedule, Obligation, PhaseInSchedule, Milestone,
    RegulatoryTransitionSystem, RegulatoryState, RegulatoryEvent, Transition,
};
use regsynth_pareto::{
    CostVector, ParetoFrontier, dominates,
    cost_model::{CostModel, ObligationCostEstimate},
    trajectory::{TrajectoryOptimizer, TrajectoryConfig, ComplianceTrajectory},
    strategy_repr::StrategyBitVec,
    metrics::hypervolume_indicator,
};

use chrono::NaiveDate;
use std::collections::BTreeSet;

/// Build cost estimates for each obligation in the EU AI Act schedule.
fn build_cost_model_for_phase(phase: usize) -> CostModel {
    let mut model = CostModel::new();

    let all_obligations = [
        // Phase 1: Prohibited practices
        ("prohibited-subliminal", 20_000.0, 1.0, 0.9, 10.0, 500_000.0),
        ("prohibited-social-scoring", 15_000.0, 1.0, 0.95, 8.0, 500_000.0),
        ("prohibited-biometric-categorization", 30_000.0, 2.0, 0.85, 15.0, 500_000.0),
        // Phase 2: GPAI
        ("gpai-transparency", 80_000.0, 4.0, 0.6, 30.0, 200_000.0),
        ("gpai-copyright", 50_000.0, 3.0, 0.5, 25.0, 150_000.0),
        ("gpai-systemic-risk", 120_000.0, 6.0, 0.7, 50.0, 300_000.0),
        // Phase 3: High-risk
        ("high-risk-conformity", 200_000.0, 10.0, 0.8, 70.0, 400_000.0),
        ("high-risk-risk-management", 150_000.0, 8.0, 0.75, 60.0, 350_000.0),
        ("high-risk-data-governance", 180_000.0, 9.0, 0.7, 65.0, 300_000.0),
        ("high-risk-transparency-users", 100_000.0, 5.0, 0.5, 40.0, 200_000.0),
        ("high-risk-human-oversight", 130_000.0, 7.0, 0.65, 55.0, 250_000.0),
        // Phase 4: Full enforcement
        ("full-market-surveillance", 90_000.0, 4.0, 0.4, 35.0, 150_000.0),
        ("full-penalties", 10_000.0, 1.0, 0.3, 5.0, 750_000.0),
        ("full-reporting", 60_000.0, 3.0, 0.35, 20.0, 100_000.0),
    ];

    // Include obligations up to the given phase
    let phase_counts = [3, 6, 11, 14]; // cumulative obligation count per phase
    let limit = if phase < phase_counts.len() { phase_counts[phase] } else { all_obligations.len() };

    for &(id, cost, time, risk, complexity, penalty) in all_obligations.iter().take(limit) {
        model.add_obligation(
            ObligationCostEstimate::new(id)
                .with_financial_cost(cost)
                .with_time(time)
                .with_risk(risk)
                .with_complexity(complexity)
                .with_penalty(penalty),
        );
    }

    model
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  RegSynth — Temporal Trajectory Optimization               ║");
    println!("║  EU AI Act Phase-In Schedule (2025–2027)                   ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // 1. Display the phase-in schedule
    let schedule = eu_ai_act_schedule();
    println!("📅 EU AI Act Phase-In Schedule:\n");
    for (i, ms) in schedule.milestones().iter().enumerate() {
        println!("  Phase {} — {} ({})", i + 1, ms.label, ms.date);
        for obl_id in &ms.obligations {
            println!("    • {}", obl_id);
        }
    }
    println!();

    // 2. Build regulatory transition system
    let mut rts = RegulatoryTransitionSystem::new();

    let s0 = RegulatoryState::new("s0-pre-enforcement");
    rts.add_state(s0);

    for (i, ms) in schedule.milestones().iter().enumerate() {
        let sid = format!("s{}-{}", i + 1, ms.label.to_lowercase().replace(' ', "-"));
        let state = RegulatoryState::with_obligations(
            sid.clone(),
            schedule.obligations_active_at(&ms.date),
        ).with_timestamp(ms.date);
        rts.add_state(state);

        let from = if i == 0 { "s0-pre-enforcement".to_string() }
                   else {
                       let prev = &schedule.milestones()[i - 1];
                       format!("s{}-{}", i, prev.label.to_lowercase().replace(' ', "-"))
                   };
        let _ = rts.add_transition(Transition {
            from,
            to: sid,
            event: RegulatoryEvent::PhaseIn {
                milestone: ms.label.clone(),
                date: ms.date,
            },
        });
    }

    println!("🔄 Regulatory Transition System:");
    println!("  States: {}", rts.states.len());
    println!("  Transitions: {}", rts.transitions.len());
    for t in &rts.transitions {
        println!("    {} →[{}]→ {}", t.from, t.event, t.to);
    }
    println!();

    // 3. Per-phase cost models
    println!("💰 Per-Phase Cost Model Summary:\n");
    for phase in 0..4 {
        let model = build_cost_model_for_phase(phase);
        let n = model.obligation_count();
        let full_strategy = StrategyBitVec::from_active(n, &(0..n).collect::<Vec<_>>());
        let full_cost = model.evaluate(&full_strategy);
        println!(
            "  Phase {} ({} obligations): full compliance cost = {}",
            phase + 1, n, full_cost
        );
    }
    println!();

    // 4. Trajectory optimization
    println!("🚀 Trajectory Optimization (4 timesteps):\n");

    let timestep_models: Vec<CostModel> = (0..4)
        .map(|p| build_cost_model_for_phase(p))
        .collect();

    let config = TrajectoryConfig {
        transition_budget: 5,
        discount_factor: 0.95,
        epsilon: 0.02,
        max_weight_vectors: 20,
        ..Default::default()
    };

    let optimizer = TrajectoryOptimizer::new(config);
    let obligation_count = 14; // total across all phases

    let trajectory_frontier = optimizer.optimize_trajectory(
        &timestep_models,
        |_t, _strategy| true, // all strategies feasible for demo
        obligation_count,
    );

    println!("  Pareto-optimal trajectories found: {}", trajectory_frontier.size());

    if !trajectory_frontier.is_empty() {
        let costs: Vec<CostVector> = trajectory_frontier.entries()
            .iter()
            .map(|e| e.cost.clone())
            .collect();
        let ref_point = CostVector::regulatory(2_000_000.0, 48.0, 1.0, 100.0);
        let hv = hypervolume_indicator(&costs, &ref_point);
        println!("  Hypervolume indicator: {:.6}", hv);

        println!("\n  Sample trajectories:");
        for (i, entry) in trajectory_frontier.entries().iter().take(5).enumerate() {
            let traj = &entry.point;
            println!(
                "    T{}: {} timesteps, {} transitions, aggregate cost = {}",
                i + 1, traj.timesteps(), traj.transition_count, traj.aggregate_cost
            );
        }
    }

    // 5. Activation timeline analysis
    println!("\n📊 Obligation Activation Over Time:\n");
    let check_dates = [
        ("Pre-enforcement", NaiveDate::from_ymd_opt(2025, 1, 1).unwrap()),
        ("After Phase 1",   NaiveDate::from_ymd_opt(2025, 3, 1).unwrap()),
        ("After Phase 2",   NaiveDate::from_ymd_opt(2025, 9, 1).unwrap()),
        ("After Phase 3",   NaiveDate::from_ymd_opt(2026, 9, 1).unwrap()),
        ("After Phase 4",   NaiveDate::from_ymd_opt(2027, 9, 1).unwrap()),
    ];

    for (label, date) in &check_dates {
        let active = schedule.obligations_active_at(date);
        let bar = "█".repeat(active.len() * 2);
        println!("  {:20} {} ({:>2} obligations) {}", label, date, active.len(), bar);
    }

    println!("\n✅ Temporal trajectory optimization complete.");
}
