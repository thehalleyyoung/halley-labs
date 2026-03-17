//! Honest experiments for BiCut tool paper.
//!
//! Tests:
//! 1. Structural analysis timing at various sizes
//! 2. Branch-and-cut solver with/without cuts comparison  
//! 3. Instance generation variety and structural properties
//!
//! Run: cargo run --example run_experiments_v2 --release

use bicut_bench::generator::{
    GeneratorConfig, InstanceGenerator, KnapsackInterdictionConfig, NetworkInterdictionConfig,
};
use bicut_branch_cut::{
    BranchAndCutSolver, BranchingStrategyType, BuiltinLpSolver, CompiledBilevelModel,
    NodeSelectionType, SolutionStatus, SolverConfig,
};
use bicut_core::StructuralAnalysis;
use bicut_types::*;
use std::time::Instant;

fn geometric_mean(vals: &[f64]) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    let shifted: Vec<f64> = vals.iter().map(|v| v.max(1e-9)).collect();
    let log_sum: f64 = shifted.iter().map(|v| v.ln()).sum();
    (log_sum / shifted.len() as f64).exp()
}

fn make_config(cut_rounds: usize, cuts_per_round: usize) -> SolverConfig {
    SolverConfig {
        time_limit_secs: 60.0,
        node_limit: 50_000,
        gap_tolerance: 1e-6,
        int_tolerance: 1e-6,
        cut_rounds_per_node: cut_rounds,
        max_cuts_per_round: cuts_per_round,
        enable_heuristics: true,
        enable_preprocessing: true,
        branching_strategy: BranchingStrategyType::MostFractional,
        node_selection: NodeSelectionType::BestFirst,
        verbosity: 0,
        strong_branching_candidates: 5,
        reliability_threshold: 8,
        heuristic_frequency: 5,
        diving_max_depth: 10,
        feasibility_pump_iterations: 20,
    }
}

/// Experiment 1: Structural Analysis Timing
fn experiment_1_structural_analysis() {
    println!("\n================================================================");
    println!("  Experiment 1: Structural Analysis Timing");
    println!("================================================================\n");

    let sizes: Vec<(usize, &str)> = vec![
        (5, "tiny"),
        (10, "small"),
        (20, "medium"),
        (50, "large"),
        (100, "xlarge"),
    ];
    let n_seeds: u64 = 10;

    println!(
        "{:<14} {:>12} {:>12} {:>12}",
        "Size", "Mean(us)", "Min(us)", "Max(us)"
    );
    println!("{}", "-".repeat(54));

    let mut latex_rows = Vec::new();

    for &(n, label) in &sizes {
        let mut times = Vec::new();

        for seed in 0..n_seeds {
            let mut gen = InstanceGenerator::new(seed * 7 + 13);
            let config = KnapsackInterdictionConfig {
                num_items: n,
                capacity_ratio: 0.5,
                budget_ratio: 0.3,
                profit_range: (1, 100),
                weight_range: (1, 50),
                seed: seed * 7 + 13,
            };
            let inst =
                gen.generate_knapsack_interdiction(&format!("knap-{}-s{}", n, seed), &config);

            let start = Instant::now();
            let _report = StructuralAnalysis::analyze(&inst.problem);
            let elapsed_us = start.elapsed().as_micros() as f64;
            times.push(elapsed_us);
        }

        let gm = geometric_mean(&times);
        let min_t = times.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_t = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        println!(
            "{:<14} {:>12.0} {:>12.0} {:>12.0}",
            format!("{}({})", n, label),
            gm,
            min_t,
            max_t
        );
        latex_rows.push(format!(
            "{} & {:.0} & {:.0} & {:.0} \\\\",
            n, gm, min_t, max_t
        ));
    }

    println!("\n% LaTeX table:");
    println!("\\begin{{tabular}}{{lrrr}}");
    println!("\\toprule");
    println!("$n$ & Geo.~mean ($\\mu$s) & Min ($\\mu$s) & Max ($\\mu$s) \\\\");
    println!("\\midrule");
    for row in &latex_rows {
        println!("{}", row);
    }
    println!("\\bottomrule");
    println!("\\end{{tabular}}");
}

/// Experiment 2: Branch-and-Cut with/without Cuts
fn experiment_2_branch_cut() {
    println!("\n================================================================");
    println!("  Experiment 2: Branch-and-Cut (No Cuts vs With Cuts)");
    println!("================================================================");

    println!("\n--- Knapsack Interdiction ---\n");
    let sizes: Vec<(usize, f64, &str)> = vec![
        (5, 0.3, "tiny"),
        (10, 0.3, "small"),
        (15, 0.3, "med-s"),
        (20, 0.3, "medium"),
        (30, 0.3, "large"),
        (50, 0.3, "xlarge"),
    ];
    let n_seeds: u64 = 5;

    println!(
        "{:<10} | {:>8} {:>6} {:>5} | {:>8} {:>6} {:>5} {:>5}",
        "Size", "NC_ms", "Nodes", "OK", "WC_ms", "Nodes", "Cuts", "OK"
    );
    println!("{}", "-".repeat(72));

    let mut latex_rows = Vec::new();

    for &(n, budget_ratio, label) in &sizes {
        let mut nc_times = Vec::new();
        let mut nc_nodes = Vec::new();
        let mut nc_solved = 0u32;
        let mut wc_times = Vec::new();
        let mut wc_nodes = Vec::new();
        let mut wc_cuts = Vec::new();
        let mut wc_solved = 0u32;

        for seed in 0..n_seeds {
            let s = seed * 137 + 42;
            let mut gen = InstanceGenerator::new(s);
            let config = KnapsackInterdictionConfig {
                num_items: n,
                capacity_ratio: 0.5,
                budget_ratio,
                profit_range: (1, 20),
                weight_range: (1, 15),
                seed: s,
            };
            let inst = gen.generate_knapsack_interdiction(&format!("bc-{}-s{}", n, seed), &config);

            // Without cuts
            {
                let model = CompiledBilevelModel::new(inst.problem.clone());
                let lp_solver = BuiltinLpSolver::new();
                let sc = make_config(0, 0);
                let start = Instant::now();
                let mut solver = BranchAndCutSolver::new(sc);
                let sol = solver.solve(&model, &lp_solver);
                let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                nc_times.push(elapsed);
                nc_nodes.push(solver.statistics.nodes_processed as f64);
                if matches!(
                    sol.status,
                    SolutionStatus::Optimal | SolutionStatus::Feasible
                ) {
                    nc_solved += 1;
                }
            }

            // With cuts (5 rounds, 10 per round)
            {
                let model = CompiledBilevelModel::new(inst.problem.clone());
                let lp_solver = BuiltinLpSolver::new();
                let sc = make_config(5, 10);
                let start = Instant::now();
                let mut solver = BranchAndCutSolver::new(sc);
                let sol = solver.solve(&model, &lp_solver);
                let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                wc_times.push(elapsed);
                wc_nodes.push(solver.statistics.nodes_processed as f64);
                wc_cuts.push(solver.statistics.total_cuts_generated() as f64);
                if matches!(
                    sol.status,
                    SolutionStatus::Optimal | SolutionStatus::Feasible
                ) {
                    wc_solved += 1;
                }
            }
        }

        let nc_gm_t = geometric_mean(&nc_times);
        let nc_gm_n = geometric_mean(&nc_nodes);
        let wc_gm_t = geometric_mean(&wc_times);
        let wc_gm_n = geometric_mean(&wc_nodes);
        let wc_avg_c = wc_cuts.iter().sum::<f64>() / n_seeds as f64;

        println!(
            "{:<10} | {:>7.1} {:>6.0} {:>4}/{} | {:>7.1} {:>6.0} {:>5.1} {:>4}/{}",
            format!("{}({})", n, label),
            nc_gm_t,
            nc_gm_n,
            nc_solved,
            n_seeds,
            wc_gm_t,
            wc_gm_n,
            wc_avg_c,
            wc_solved,
            n_seeds
        );

        latex_rows.push(format!(
            "{} & {:.1} & {:.0} & {}/{} & {:.1} & {:.0} & {:.1} & {}/{} \\\\",
            n, nc_gm_t, nc_gm_n, nc_solved, n_seeds, wc_gm_t, wc_gm_n, wc_avg_c, wc_solved, n_seeds
        ));
    }

    // Network interdiction instances
    println!("\n--- Network Interdiction ---\n");
    println!(
        "{:<12} | {:>8} {:>6} {:>5} | {:>8} {:>6} {:>5} {:>5}",
        "Nodes", "NC_ms", "Nodes", "OK", "WC_ms", "Nodes", "Cuts", "OK"
    );
    println!("{}", "-".repeat(72));

    for &n_nodes in &[8usize, 12, 15, 20] {
        let mut nc_times = Vec::new();
        let mut nc_nodes_v = Vec::new();
        let mut nc_solved = 0u32;
        let mut wc_times = Vec::new();
        let mut wc_nodes_v = Vec::new();
        let mut wc_cuts = Vec::new();
        let mut wc_solved = 0u32;

        for seed in 0..n_seeds {
            let s = seed * 73 + 11;
            let mut gen = InstanceGenerator::new(s);
            let config = NetworkInterdictionConfig {
                num_nodes: n_nodes,
                edge_probability: 0.4,
                interdiction_budget: (n_nodes / 3).max(2),
                capacity_range: (1.0, 10.0),
                seed: s,
            };
            let inst =
                gen.generate_network_interdiction(&format!("net-{}-s{}", n_nodes, seed), &config);

            // Without cuts
            {
                let model = CompiledBilevelModel::new(inst.problem.clone());
                let lp_solver = BuiltinLpSolver::new();
                let sc = make_config(0, 0);
                let start = Instant::now();
                let mut solver = BranchAndCutSolver::new(sc);
                let sol = solver.solve(&model, &lp_solver);
                let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                nc_times.push(elapsed);
                nc_nodes_v.push(solver.statistics.nodes_processed as f64);
                if matches!(
                    sol.status,
                    SolutionStatus::Optimal | SolutionStatus::Feasible
                ) {
                    nc_solved += 1;
                }
            }

            // With cuts
            {
                let model = CompiledBilevelModel::new(inst.problem.clone());
                let lp_solver = BuiltinLpSolver::new();
                let sc = make_config(5, 10);
                let start = Instant::now();
                let mut solver = BranchAndCutSolver::new(sc);
                let sol = solver.solve(&model, &lp_solver);
                let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                wc_times.push(elapsed);
                wc_nodes_v.push(solver.statistics.nodes_processed as f64);
                wc_cuts.push(solver.statistics.total_cuts_generated() as f64);
                if matches!(
                    sol.status,
                    SolutionStatus::Optimal | SolutionStatus::Feasible
                ) {
                    wc_solved += 1;
                }
            }
        }

        let nc_gm_t = geometric_mean(&nc_times);
        let nc_gm_n = geometric_mean(&nc_nodes_v);
        let wc_gm_t = geometric_mean(&wc_times);
        let wc_gm_n = geometric_mean(&wc_nodes_v);
        let wc_avg_c = wc_cuts.iter().sum::<f64>() / n_seeds as f64;

        println!(
            "{:<12} | {:>7.1} {:>6.0} {:>4}/{} | {:>7.1} {:>6.0} {:>5.1} {:>4}/{}",
            format!("net-{}", n_nodes),
            nc_gm_t,
            nc_gm_n,
            nc_solved,
            n_seeds,
            wc_gm_t,
            wc_gm_n,
            wc_avg_c,
            wc_solved,
            n_seeds
        );

        latex_rows.push(format!(
            "Net-{} & {:.1} & {:.0} & {}/{} & {:.1} & {:.0} & {:.1} & {}/{} \\\\",
            n_nodes,
            nc_gm_t,
            nc_gm_n,
            nc_solved,
            n_seeds,
            wc_gm_t,
            wc_gm_n,
            wc_avg_c,
            wc_solved,
            n_seeds
        ));
    }

    println!("\n% LaTeX combined table:");
    println!("\\begin{{tabular}}{{lrrrrrrr}}");
    println!("\\toprule");
    println!("& \\multicolumn{{3}}{{c}}{{No cuts}} & \\multicolumn{{4}}{{c}}{{With intersection cuts}} \\\\");
    println!("\\cmidrule(lr){{2-4}} \\cmidrule(lr){{5-8}}");
    println!("Instance & Time (ms) & Nodes & Solved & Time (ms) & Nodes & Cuts & Solved \\\\");
    println!("\\midrule");
    for row in &latex_rows {
        println!("{}", row);
    }
    println!("\\bottomrule");
    println!("\\end{{tabular}}");
}

/// Experiment 3: Instance Properties
fn experiment_3_instance_properties() {
    println!("\n================================================================");
    println!("  Experiment 3: Instance Structural Properties");
    println!("================================================================\n");

    println!(
        "{:<16} {:>5} {:>5} {:>5} {:>18} {:>18}",
        "Type", "n_x", "n_y", "m", "Lower type", "Coupling"
    );
    println!("{}", "-".repeat(75));

    // Knapsack interdiction
    for &n in &[10usize, 20, 50] {
        let mut gen = InstanceGenerator::new(42);
        let config = KnapsackInterdictionConfig {
            num_items: n,
            capacity_ratio: 0.5,
            budget_ratio: 0.3,
            profit_range: (1, 100),
            weight_range: (1, 50),
            seed: 42,
        };
        let inst = gen.generate_knapsack_interdiction(&format!("knap-{}", n), &config);
        let report = StructuralAnalysis::analyze(&inst.problem);
        println!(
            "{:<16} {:>5} {:>5} {:>5} {:>18?} {:>18?}",
            format!("Knap-{}", n),
            inst.problem.num_upper_vars,
            inst.problem.num_lower_vars,
            inst.problem.num_lower_constraints,
            report.lower_level_type,
            report.coupling_type
        );
    }

    // Network interdiction
    for &n_nodes in &[8usize, 15, 30] {
        let mut gen = InstanceGenerator::new(42);
        let config = NetworkInterdictionConfig {
            num_nodes: n_nodes,
            edge_probability: 0.3,
            interdiction_budget: (n_nodes / 3).max(1),
            capacity_range: (1.0, 10.0),
            seed: 42,
        };
        let inst = gen.generate_network_interdiction(&format!("net-{}", n_nodes), &config);
        let report = StructuralAnalysis::analyze(&inst.problem);
        println!(
            "{:<16} {:>5} {:>5} {:>5} {:>18?} {:>18?}",
            format!("Net-{}", n_nodes),
            inst.problem.num_upper_vars,
            inst.problem.num_lower_vars,
            inst.problem.num_lower_constraints,
            report.lower_level_type,
            report.coupling_type
        );
    }

    // Random bilevel
    for &(nu, nl) in &[(5usize, 10usize), (10, 20), (20, 50)] {
        let mut gen = InstanceGenerator::new(42);
        let config = GeneratorConfig::default().with_dimensions(nu, nl, nu, nl + 5);
        let inst = gen.generate(&format!("rand-{}-{}", nu, nl), &config);
        let report = StructuralAnalysis::analyze(&inst.problem);
        println!(
            "{:<16} {:>5} {:>5} {:>5} {:>18?} {:>18?}",
            format!("Rand-{}/{}", nu, nl),
            inst.problem.num_upper_vars,
            inst.problem.num_lower_vars,
            inst.problem.num_lower_constraints,
            report.lower_level_type,
            report.coupling_type
        );
    }
}

fn main() {
    println!("================================================================");
    println!("  BiCut Honest Experiments");
    println!(
        "  Platform: {} {}",
        std::env::consts::OS,
        std::env::consts::ARCH
    );
    println!("================================================================");

    experiment_1_structural_analysis();
    experiment_2_branch_cut();
    experiment_3_instance_properties();

    println!("\n================================================================");
    println!("  All experiments complete.");
    println!("================================================================");
}
