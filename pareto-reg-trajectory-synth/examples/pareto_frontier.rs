//! # Pareto Frontier Computation
//!
//! This example demonstrates computing a Pareto frontier over multi-objective
//! compliance strategies for a company subject to the EU AI Act.
//!
//! ## Multi-Objective Optimization Problem
//!
//! Each compliance strategy has a 4-dimensional cost vector:
//! 1. **Financial cost** ($): implementation budget
//! 2. **Time to compliance** (months): schedule impact
//! 3. **Regulatory risk** (0–1): residual non-compliance probability
//! 4. **Implementation complexity** (0–100): engineering effort
//!
//! The Pareto frontier represents the set of strategies where no other
//! strategy is better in ALL dimensions simultaneously. Points on the
//! frontier represent genuine trade-offs the decision-maker must resolve.
//!
//! ## Demonstrated Features
//!
//! - Dominance checking (strict, weak, epsilon)
//! - Pareto frontier construction with automatic filtering
//! - Quality metrics: hypervolume, spread, spacing
//! - Scalarization methods for single-objective approximation
//! - Non-dominated sorting (NSGA-II style)

use regsynth_pareto::{
    CostVector, ParetoFrontier, ParetoMetrics,
    dominates, epsilon_dominates, pareto_compare, ParetoOrdering,
    filter_dominated, fast_non_dominated_sort,
    WeightedSumScalarizer, ChebyshevScalarizer,
    ComplianceStrategy, ObligationEntry,
    metrics::{hypervolume_indicator, spread_metric, spacing_metric},
};
use regsynth_types::{Cost, Id};

/// Generate a realistic set of compliance strategies with varying trade-offs.
fn generate_strategies() -> Vec<(&'static str, CostVector)> {
    vec![
        ("S1: Minimal (documentation only)",
         CostVector::regulatory(80_000.0, 3.0, 0.45, 15.0)),
        ("S2: Risk-focused (testing + monitoring)",
         CostVector::regulatory(250_000.0, 8.0, 0.12, 45.0)),
        ("S3: Balanced (all core obligations)",
         CostVector::regulatory(400_000.0, 12.0, 0.08, 60.0)),
        ("S4: Premium (automated compliance)",
         CostVector::regulatory(700_000.0, 6.0, 0.03, 85.0)),
        ("S5: Fast-track (external consultants)",
         CostVector::regulatory(550_000.0, 4.0, 0.10, 35.0)),
        ("S6: Budget (student interns)",
         CostVector::regulatory(120_000.0, 18.0, 0.30, 20.0)),
        ("S7: Gold-plated (exceeds requirements)",
         CostVector::regulatory(900_000.0, 14.0, 0.01, 95.0)),
        ("S8: Outsourced (managed compliance)",
         CostVector::regulatory(500_000.0, 5.0, 0.06, 25.0)),
        // Dominated strategies (should be filtered)
        ("S9: Worse S3 (dominated)",
         CostVector::regulatory(450_000.0, 14.0, 0.10, 65.0)),
        ("S10: Worse S8 (dominated)",
         CostVector::regulatory(600_000.0, 7.0, 0.08, 30.0)),
    ]
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  RegSynth — Pareto Frontier Computation                    ║");
    println!("║  4-Objective Compliance Strategy Optimization              ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let strategies = generate_strategies();

    // 1. Pairwise dominance analysis
    println!("🔍 Pairwise Dominance Analysis:\n");
    let mut dominance_count = 0;
    for i in 0..strategies.len() {
        for j in 0..strategies.len() {
            if i != j && dominates(&strategies[i].1, &strategies[j].1) {
                println!("  {} ≻ {}", strategies[i].0, strategies[j].0);
                dominance_count += 1;
            }
        }
    }
    if dominance_count == 0 {
        println!("  No strict dominance relationships found (all strategies are Pareto-incomparable).");
    }
    println!("  Total dominance pairs: {}\n", dominance_count);

    // 2. Build Pareto frontier
    println!("📈 Building Pareto Frontier:\n");
    let mut frontier: ParetoFrontier<String> = ParetoFrontier::new(4);
    for (name, cv) in &strategies {
        let added = frontier.add_point(name.to_string(), cv.clone());
        let symbol = if added { "✅" } else { "❌" };
        println!("  {} {} — cost={}", symbol, name, cv);
    }
    println!("\n  Frontier size: {} / {} strategies", frontier.size(), strategies.len());

    // 3. Epsilon-dominance frontier
    println!("\n📐 Epsilon-Dominance Frontier (ε=0.05):\n");
    let mut eps_frontier: ParetoFrontier<String> = ParetoFrontier::with_epsilon(4, 0.05);
    for (name, cv) in &strategies {
        eps_frontier.add_point(name.to_string(), cv.clone());
    }
    println!("  ε-frontier size: {} strategies (vs {} exact)",
        eps_frontier.size(), frontier.size());

    // 4. Non-dominated sorting
    println!("\n🏷  Non-Dominated Sorting (NSGA-II):\n");
    let all_costs: Vec<CostVector> = strategies.iter().map(|(_, cv)| cv.clone()).collect();
    let fronts = fast_non_dominated_sort(&all_costs);
    for (rank, front) in fronts.iter().enumerate() {
        let names: Vec<_> = front.iter().map(|&i| strategies[i].0).collect();
        println!("  Rank {}: {} strategies — {:?}", rank, front.len(), names);
    }

    // 5. Quality metrics
    println!("\n📊 Pareto Quality Metrics:\n");
    let frontier_costs: Vec<CostVector> = frontier.entries()
        .iter()
        .map(|e| e.cost.clone())
        .collect();
    let reference = CostVector::regulatory(1_000_000.0, 24.0, 1.0, 100.0);
    let metrics = ParetoMetrics::compute_self(&frontier_costs, &reference);
    println!("  Hypervolume indicator: {:.6}", metrics.hypervolume);
    println!("  Spread:               {:.6}", metrics.spread);
    println!("  Spacing:              {:.6}", metrics.spacing);
    println!("  Number of points:     {}", metrics.num_points);

    // 6. Scalarization analysis
    println!("\n⚖️  Scalarization Analysis:\n");
    let weight_sets = vec![
        ("Cost-focused",   vec![0.6, 0.1, 0.2, 0.1]),
        ("Time-focused",   vec![0.1, 0.6, 0.2, 0.1]),
        ("Risk-focused",   vec![0.1, 0.1, 0.7, 0.1]),
        ("Balanced",       vec![0.25, 0.25, 0.25, 0.25]),
    ];

    for (label, weights) in &weight_sets {
        let scalarizer = WeightedSumScalarizer::new(weights.clone());
        let mut best_idx = 0;
        let mut best_val = f64::INFINITY;
        for (i, (_, cv)) in strategies.iter().enumerate() {
            let val = cv.weighted_sum(weights);
            if val < best_val {
                best_val = val;
                best_idx = i;
            }
        }
        println!("  {:14} → {} (score: {:.2})",
            label, strategies[best_idx].0, best_val);
    }

    // 7. Pareto comparison matrix
    println!("\n🔢 Pareto Comparison Matrix (first 5):\n");
    println!("  {:>5} {:>5} {:>5} {:>5} {:>5}", "S1", "S2", "S3", "S4", "S5");
    for i in 0..5.min(strategies.len()) {
        print!("  ");
        for j in 0..5.min(strategies.len()) {
            let symbol = if i == j {
                " — "
            } else {
                match pareto_compare(&strategies[i].1, &strategies[j].1) {
                    ParetoOrdering::Dominates => " ≻ ",
                    ParetoOrdering::Dominated => " ≺ ",
                    ParetoOrdering::Incomparable => " ~ ",
                    ParetoOrdering::Equal => " = ",
                }
            };
            print!("{:>5}", symbol);
        }
        println!("  {}", strategies[i].0.split(':').next().unwrap_or(""));
    }

    println!("\n✅ Pareto analysis complete. {} Pareto-optimal strategies identified.",
        frontier.size());
}
