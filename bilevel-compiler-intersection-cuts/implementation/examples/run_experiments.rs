//! Real experiments for the BiCut tool paper.
//!
//! Generates bilevel knapsack interdiction instances at various sizes,
//! runs the actual branch-and-cut solver with and without intersection cuts,
//! and reports honest timing/node-count data.
//!
//! Run: `cargo run --example run_experiments --release`

use bicut_bench::generator::{GeneratorConfig, InstanceGenerator, KnapsackInterdictionConfig};
use bicut_bench::instance::{BenchmarkInstance, DifficultyClass, InstanceMetadata, InstanceType};
use bicut_branch_cut::{
    BranchAndCutSolver, BranchingStrategyType, BuiltinLpSolver, CompiledBilevelModel,
    LpSolverInterface, NodeSelectionType, SolutionStatus, SolverConfig,
};
use bicut_core::StructuralAnalysis;
use bicut_lp::SimplexSolver;
use bicut_types::*;
use std::time::Instant;

/// A single experiment result row.
#[derive(Debug, Clone)]
struct ExperimentResult {
    instance_name: String,
    n_items: usize,
    n_leader: usize,
    n_follower: usize,
    config_label: String, // "no_cuts" or "with_cuts"
    status: String,
    wall_time_ms: f64,
    node_count: u64,
    gap: f64,
    objective: f64,
    lp_time_ms: f64,
    cut_time_ms: f64,
    cuts_added: u64,
}

fn make_solver_config(cut_rounds: usize, time_limit: f64) -> SolverConfig {
    SolverConfig {
        time_limit_secs: time_limit,
        node_limit: 50_000,
        gap_tolerance: 1e-4,
        int_tolerance: 1e-6,
        cut_rounds_per_node: cut_rounds,
        max_cuts_per_round: 10,
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

fn run_instance(
    problem: &BilevelProblem,
    name: &str,
    config_label: &str,
    cut_rounds: usize,
) -> ExperimentResult {
    let start = Instant::now();

    let model = CompiledBilevelModel::new(problem.clone());
    let lp_solver = BuiltinLpSolver::new();
    let solver_config = make_solver_config(cut_rounds, 30.0); // 30s time limit per instance

    let mut solver = BranchAndCutSolver::new(solver_config);
    let solution = solver.solve(&model, &lp_solver);

    let elapsed = start.elapsed();
    let stats = &solver.statistics;

    ExperimentResult {
        instance_name: name.to_string(),
        n_items: problem.num_upper_vars,
        n_leader: problem.num_upper_vars,
        n_follower: problem.num_lower_vars,
        config_label: config_label.to_string(),
        status: format!("{}", solution.status),
        wall_time_ms: elapsed.as_secs_f64() * 1000.0,
        node_count: stats.nodes_processed,
        gap: solution.gap,
        objective: solution.objective,
        lp_time_ms: solver.timers.lp.elapsed() * 1000.0,
        cut_time_ms: solver.timers.cuts.elapsed() * 1000.0,
        cuts_added: stats.total_cuts_generated(),
    }
}

fn geometric_mean(vals: &[f64]) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    let log_sum: f64 = vals.iter().map(|v| (v.max(1e-6)).ln()).sum();
    (log_sum / vals.len() as f64).exp()
}

fn main() {
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  BiCut Real Experiments: Knapsack Interdiction Benchmark");
    println!("═══════════════════════════════════════════════════════════════════\n");

    // Instance sizes to test
    let sizes = vec![
        (6, 2, "tiny"),
        (10, 3, "small"),
        (15, 4, "medium-small"),
        (20, 5, "medium"),
        (30, 7, "medium-large"),
        (50, 10, "large"),
    ];
    let seeds_per_size = 5;

    let mut all_results: Vec<ExperimentResult> = Vec::new();

    for &(n_items, budget, size_label) in &sizes {
        println!(
            "── Size: {} ({} items, budget {}) ──",
            size_label, n_items, budget
        );

        for seed in 0..seeds_per_size {
            let instance_name = format!("knap-{}-{}-s{}", n_items, budget, seed);

            // Generate a knapsack interdiction instance
            let mut gen = InstanceGenerator::new(seed as u64 * 137 + 42);
            let config = KnapsackInterdictionConfig {
                num_items: n_items,
                capacity_ratio: 0.5,
                budget_ratio: budget as f64 / n_items as f64,
                profit_range: (1, 20),
                weight_range: (1, 15),
                seed: seed as u64 * 137 + 42,
            };
            let instance =
                gen.generate_knapsack_interdiction(&format!("knap-{}-{}", n_items, seed), &config);
            let problem = &instance.problem;

            // Run without cuts
            let r_no_cuts = run_instance(problem, &instance_name, "no_cuts", 0);

            // Run with cuts
            let r_with_cuts = run_instance(problem, &instance_name, "with_cuts", 5);

            print!(
                "  {} | no_cuts: {:.1}ms, {} nodes | with_cuts: {:.1}ms, {} nodes",
                instance_name,
                r_no_cuts.wall_time_ms,
                r_no_cuts.node_count,
                r_with_cuts.wall_time_ms,
                r_with_cuts.node_count
            );

            if r_with_cuts.node_count < r_no_cuts.node_count && r_no_cuts.node_count > 0 {
                let reduction = 1.0 - (r_with_cuts.node_count as f64 / r_no_cuts.node_count as f64);
                print!(" (node reduction: {:.0}%)", reduction * 100.0);
            }
            println!();

            all_results.push(r_no_cuts);
            all_results.push(r_with_cuts);
        }
        println!();
    }

    // Print aggregate results as LaTeX table
    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("  LaTeX Table: Aggregate Results by Instance Size");
    println!("═══════════════════════════════════════════════════════════════════\n");

    println!("\\begin{{table}}[t]");
    println!("\\centering");
    println!("\\caption{{Knapsack interdiction benchmark results. Times in milliseconds");
    println!(
        "  (geometric mean over {} seeds per size). Cuts = bilevel intersection",
        seeds_per_size
    );
    println!("  cut rounds per B\\&B node.}}");
    println!("\\label{{tab:knapsack_results}}");
    println!("\\begin{{tabular}}{{llrrrrrr}}");
    println!("\\toprule");
    println!("Size & Config & Time (ms) & Nodes & Cuts & LP (ms) & Cut (ms) & Solved \\\\");
    println!("\\midrule");

    for &(n_items, budget, size_label) in &sizes {
        let no_cuts: Vec<&ExperimentResult> = all_results
            .iter()
            .filter(|r| r.n_items == n_items && r.config_label == "no_cuts")
            .collect();
        let with_cuts: Vec<&ExperimentResult> = all_results
            .iter()
            .filter(|r| r.n_items == n_items && r.config_label == "with_cuts")
            .collect();

        let nc_times: Vec<f64> = no_cuts.iter().map(|r| r.wall_time_ms).collect();
        let nc_nodes: Vec<f64> = no_cuts.iter().map(|r| r.node_count as f64).collect();
        let nc_solved = no_cuts
            .iter()
            .filter(|r| r.status == "Optimal" || r.status == "Feasible")
            .count();

        let wc_times: Vec<f64> = with_cuts.iter().map(|r| r.wall_time_ms).collect();
        let wc_nodes: Vec<f64> = with_cuts.iter().map(|r| r.node_count as f64).collect();
        let wc_cuts: Vec<f64> = with_cuts.iter().map(|r| r.cuts_added as f64).collect();
        let wc_lp: Vec<f64> = with_cuts.iter().map(|r| r.lp_time_ms).collect();
        let wc_cut_t: Vec<f64> = with_cuts.iter().map(|r| r.cut_time_ms).collect();
        let wc_solved = with_cuts
            .iter()
            .filter(|r| r.status == "Optimal" || r.status == "Feasible")
            .count();

        let nc_lp: Vec<f64> = no_cuts.iter().map(|r| r.lp_time_ms).collect();

        println!(
            "\\multicolumn{{8}}{{l}}{{\\emph{{{} ($n={}$, $k={}$)}}}} \\\\",
            size_label, n_items, budget
        );
        println!(
            "  & No cuts & {:.1} & {:.0} & --- & {:.1} & --- & {}/{} \\\\",
            geometric_mean(&nc_times),
            geometric_mean(&nc_nodes),
            geometric_mean(&nc_lp),
            nc_solved,
            seeds_per_size
        );
        println!(
            "  & With cuts & {:.1} & {:.0} & {:.0} & {:.1} & {:.1} & {}/{} \\\\",
            geometric_mean(&wc_times),
            geometric_mean(&wc_nodes),
            geometric_mean(&wc_cuts),
            geometric_mean(&wc_lp),
            geometric_mean(&wc_cut_t),
            wc_solved,
            seeds_per_size
        );

        if geometric_mean(&nc_times) > 0.0 {
            let speedup = geometric_mean(&nc_times) / geometric_mean(&wc_times).max(0.001);
            let node_ratio = geometric_mean(&nc_nodes) / geometric_mean(&wc_nodes).max(0.001);
            println!(
                "  & \\emph{{Speedup}} & {:.2}$\\times$ & {:.2}$\\times$ & & & & \\\\",
                speedup, node_ratio
            );
        }
        println!("\\midrule");
    }

    println!("\\bottomrule");
    println!("\\end{{tabular}}");
    println!("\\end{{table}}");

    // Also print raw CSV for reference
    println!("\n\n═══════════════════════════════════════════════════════════════════");
    println!("  Raw CSV Data");
    println!("═══════════════════════════════════════════════════════════════════\n");
    println!("instance,n_items,leader,follower,config,status,time_ms,nodes,gap,objective,lp_ms,cut_ms,cuts");
    for r in &all_results {
        println!(
            "{},{},{},{},{},{},{:.2},{},{:.6},{:.6},{:.2},{:.2},{}",
            r.instance_name,
            r.n_items,
            r.n_leader,
            r.n_follower,
            r.config_label,
            r.status,
            r.wall_time_ms,
            r.node_count,
            r.gap,
            r.objective,
            r.lp_time_ms,
            r.cut_time_ms,
            r.cuts_added
        );
    }

    // Summary statistics
    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("  Summary");
    println!("═══════════════════════════════════════════════════════════════════\n");

    let total = all_results.len() / 2;
    let nc_optimal = all_results
        .iter()
        .filter(|r| {
            r.config_label == "no_cuts" && (r.status == "Optimal" || r.status == "Feasible")
        })
        .count();
    let wc_optimal = all_results
        .iter()
        .filter(|r| {
            r.config_label == "with_cuts" && (r.status == "Optimal" || r.status == "Feasible")
        })
        .count();

    println!("Total instances: {}", total);
    println!("Solved (no cuts): {}/{}", nc_optimal, total);
    println!("Solved (with cuts): {}/{}", wc_optimal, total);

    let nc_all_times: Vec<f64> = all_results
        .iter()
        .filter(|r| r.config_label == "no_cuts")
        .map(|r| r.wall_time_ms)
        .collect();
    let wc_all_times: Vec<f64> = all_results
        .iter()
        .filter(|r| r.config_label == "with_cuts")
        .map(|r| r.wall_time_ms)
        .collect();

    println!(
        "Geometric mean time (no cuts): {:.2} ms",
        geometric_mean(&nc_all_times)
    );
    println!(
        "Geometric mean time (with cuts): {:.2} ms",
        geometric_mean(&wc_all_times)
    );

    let nc_all_nodes: Vec<f64> = all_results
        .iter()
        .filter(|r| r.config_label == "no_cuts")
        .map(|r| r.node_count as f64)
        .collect();
    let wc_all_nodes: Vec<f64> = all_results
        .iter()
        .filter(|r| r.config_label == "with_cuts")
        .map(|r| r.node_count as f64)
        .collect();

    println!(
        "Geometric mean nodes (no cuts): {:.1}",
        geometric_mean(&nc_all_nodes)
    );
    println!(
        "Geometric mean nodes (with cuts): {:.1}",
        geometric_mean(&wc_all_nodes)
    );
}
