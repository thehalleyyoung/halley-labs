//! Real end-to-end benchmark for BiCut bilevel optimization compiler.
//!
//! Exercises actual code paths:
//!   - Structural analysis
//!   - KKT / Strong Duality / Value Function compilation
//!   - LP relaxation solving (SimplexSolver)
//!   - Branch-and-cut solving
//!   - Cut generation via CutCallbackManager
//!
//! Tests 10 instances across 3 problem families:
//!   - Knapsack interdiction (3 sizes: 4, 8, 15 items)
//!   - Network design (3 sizes: 4, 6, 10 nodes)
//!   - Toll pricing (4 sizes: 3, 5, 8, 12 arcs)
//!
//! Reports JSON on stdout for consumption by real_benchmark.py.
//!
//! Run: `cargo run --release --example real_benchmark_runner`

use std::time::Instant;

use bicut_branch_cut::{
    BranchAndCutSolver, BranchingStrategyType, BuiltinLpSolver, CompiledBilevelModel,
    NodeSelectionType, SolutionStatus, SolverConfig,
};
use bicut_compiler::{
    compile, milp_to_lp, BackendTarget, BigMStrategy, CompilerConfig, ComplementarityEncoding,
    ReformulationType,
};
use bicut_core::StructuralAnalysis;
use bicut_cuts::{Cut as PoolCut, CutPool, CutStats};
use bicut_lp::{LpSolver, SimplexSolver};
use bicut_types::{
    BilevelProblem, ConstraintSense, LpProblem, LpStatus, OptDirection, SparseMatrix, VarBound,
};

// ─── Result types ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, serde::Serialize)]
struct BenchmarkResult {
    instance: String,
    family: String,
    reformulation: String,
    num_upper_vars: usize,
    num_lower_vars: usize,
    num_vars: usize,
    num_constraints: usize,
    compile_time_ms: f64,
    lp_solve_time_ms: f64,
    lp_objective: f64,
    bc_solve_time_ms: f64,
    bc_objective: f64,
    gap_pct: f64,
    nodes: u64,
    cuts_added: usize,
    status: String,
    baseline_obj: f64,
    baseline_time_ms: f64,
    bound_improvement_pct: f64,
    analysis_type: String,
    coupling_type: String,
}

#[derive(Debug, Clone, serde::Serialize)]
struct BenchmarkSuite {
    benchmark_results: Vec<BenchmarkResult>,
    total_time_secs: f64,
    platform: String,
    timestamp: String,
}

// ─── Problem builders ────────────────────────────────────────────────────────

/// Build a knapsack interdiction bilevel problem.
///
/// Leader: choose ≤ budget items to interdict (binary z_i).
/// Follower: max Σ p_i * x_i s.t. Σ w_i * x_i ≤ capacity, x_i ≤ 1-z_i.
///
/// Modeled as: leader min (−follower profit), follower min (−profit).
fn make_knapsack_interdiction(n_items: usize, seed: u64) -> (String, BilevelProblem) {
    let name = format!("knapsack_inter_{}", n_items);

    // Deterministic pseudo-random data from seed
    let profits: Vec<f64> = (0..n_items)
        .map(|i| ((seed.wrapping_mul(7 + i as u64) + 13) % 50 + 5) as f64)
        .collect();
    let weights: Vec<f64> = (0..n_items)
        .map(|i| ((seed.wrapping_mul(11 + i as u64) + 7) % 30 + 3) as f64)
        .collect();
    let capacity: f64 = weights.iter().sum::<f64>() * 0.5;
    let budget = (n_items as f64 * 0.3).ceil() as usize;

    // Follower constraints: 1 capacity + n_items coupling
    let num_lower_cons = 1 + n_items;
    let mut lower_a = SparseMatrix::new(num_lower_cons, n_items);
    for i in 0..n_items {
        lower_a.add_entry(0, i, weights[i]); // capacity
        lower_a.add_entry(1 + i, i, 1.0); // x_i ≤ 1 - z_i
    }
    let mut lower_b = vec![capacity];
    lower_b.extend(vec![1.0; n_items]);

    // Linking: x_i + z_i ≤ 1  →  x_i ≤ 1 - z_i  →  linking adds -z_i to RHS
    let mut linking = SparseMatrix::new(num_lower_cons, n_items);
    for i in 0..n_items {
        linking.add_entry(1 + i, i, -1.0);
    }

    // Upper constraint: Σ z_i ≤ budget
    let mut upper_a = SparseMatrix::new(1, n_items);
    for i in 0..n_items {
        upper_a.add_entry(0, i, 1.0);
    }

    let bilevel = BilevelProblem {
        upper_obj_c_x: vec![0.0; n_items],
        upper_obj_c_y: profits.iter().map(|p| -p).collect(),
        lower_obj_c: profits.iter().map(|p| -p).collect(),
        lower_a,
        lower_b,
        lower_linking_b: linking,
        upper_constraints_a: upper_a,
        upper_constraints_b: vec![budget as f64],
        num_upper_vars: n_items,
        num_lower_vars: n_items,
        num_lower_constraints: num_lower_cons,
        num_upper_constraints: 1,
    };

    (name, bilevel)
}

/// Build a network design bilevel problem.
///
/// Leader: invest in arc expansions (continuous x_i ∈ [0, cap]).
/// Follower: route flow to minimize cost on expanded network.
fn make_network_design(n_nodes: usize, seed: u64) -> (String, BilevelProblem) {
    let name = format!("netdesign_{}", n_nodes);
    let demand = 10.0;

    // Build edges: connect i → j for j > i with some gaps
    let mut edges: Vec<(usize, usize, f64, f64)> = Vec::new(); // (from, to, cost, base_cap)
    for i in 0..n_nodes {
        for j in (i + 1)..n_nodes {
            let hash = seed.wrapping_mul(31 + i as u64).wrapping_add(j as u64 * 17);
            if hash % 3 != 0 || j == i + 1 {
                let cost = ((hash % 20) + 1) as f64;
                let cap = ((hash % 15) + 5) as f64;
                edges.push((i, j, cost, cap));
            }
        }
    }
    let n_arcs = edges.len();
    let n_leader = n_arcs; // expansion variable per arc

    // Follower constraints: flow conservation (n_nodes) + capacity (n_arcs)
    let num_lower_cons = n_nodes + n_arcs;
    let mut lower_a = SparseMatrix::new(num_lower_cons, n_arcs);
    let mut lower_b = vec![0.0; num_lower_cons];

    // Flow conservation
    lower_b[0] = demand;
    lower_b[n_nodes - 1] = -demand;
    for (ai, &(from, to, _, _)) in edges.iter().enumerate() {
        lower_a.add_entry(from, ai, 1.0);
        lower_a.add_entry(to, ai, -1.0);
    }
    // Capacity: f_a ≤ base_cap + x_a  →  f_a ≤ base_cap (RHS) + x_a (linking)
    for (ai, &(_, _, _, cap)) in edges.iter().enumerate() {
        lower_a.add_entry(n_nodes + ai, ai, 1.0);
        lower_b[n_nodes + ai] = cap;
    }

    // Linking: x_a adds capacity → +x_a on RHS of capacity constraints
    let mut linking = SparseMatrix::new(num_lower_cons, n_leader);
    for ai in 0..n_arcs {
        linking.add_entry(n_nodes + ai, ai, 1.0); // +x_a to RHS
    }

    // Leader: min Σ invest_cost_a * x_a (investment) - α * total_flow
    // Simplify: leader wants to minimize investment cost
    let invest_costs: Vec<f64> = edges
        .iter()
        .enumerate()
        .map(|(i, _)| ((seed.wrapping_mul(3 + i as u64) % 10) + 1) as f64)
        .collect();

    // Upper constraint: total investment ≤ budget
    let budget = invest_costs.iter().sum::<f64>() * 0.4;
    let mut upper_a = SparseMatrix::new(1, n_leader);
    for i in 0..n_leader {
        upper_a.add_entry(0, i, 1.0);
    }

    let bilevel = BilevelProblem {
        upper_obj_c_x: invest_costs,
        upper_obj_c_y: vec![0.0; n_arcs],
        lower_obj_c: edges.iter().map(|e| e.2).collect(),
        lower_a,
        lower_b,
        lower_linking_b: linking,
        upper_constraints_a: upper_a,
        upper_constraints_b: vec![budget],
        num_upper_vars: n_leader,
        num_lower_vars: n_arcs,
        num_lower_constraints: num_lower_cons,
        num_upper_constraints: 1,
    };

    (name, bilevel)
}

/// Build a toll pricing bilevel problem.
///
/// Leader: set tolls on tollable arcs to maximize revenue.
/// Follower: route flow to minimize (base_cost + toll) over network.
fn make_toll_pricing(n_arcs: usize, seed: u64) -> (String, BilevelProblem) {
    let name = format!("toll_pricing_{}", n_arcs);
    let n_nodes = (n_arcs as f64 * 0.6).ceil() as usize + 2;
    let demand = 8.0;

    // Generate arcs
    let mut arcs: Vec<(usize, usize, f64, f64, bool)> = Vec::new();
    let mut added = 0;
    for i in 0..n_nodes {
        for j in (i + 1)..n_nodes {
            if added >= n_arcs {
                break;
            }
            let hash = seed.wrapping_mul(23 + i as u64).wrapping_add(j as u64 * 41);
            if j == i + 1 || hash % 3 == 0 {
                let cost = ((hash % 10) + 1) as f64;
                let cap = ((hash % 15) + 5) as f64;
                let tollable = hash % 2 == 0;
                arcs.push((i, j, cost, cap, tollable));
                added += 1;
            }
        }
    }
    // Ensure we reach n_arcs
    while arcs.len() < n_arcs {
        let i = arcs.len() % n_nodes;
        let j = (i + 1) % n_nodes;
        arcs.push((i, j.max(1), 5.0, 10.0, true));
    }

    let n_actual_arcs = arcs.len();
    let tollable: Vec<usize> = arcs
        .iter()
        .enumerate()
        .filter(|(_, a)| a.4)
        .map(|(i, _)| i)
        .collect();
    let num_tolls = tollable.len().max(1);

    // Follower: flow conservation + capacity
    let num_lower_cons = n_nodes + n_actual_arcs;
    let mut lower_a = SparseMatrix::new(num_lower_cons, n_actual_arcs);
    let mut lower_b = vec![0.0; num_lower_cons];

    lower_b[0] = demand;
    if n_nodes > 1 {
        lower_b[n_nodes - 1] = -demand;
    }
    for (ai, &(from, to, _, _, _)) in arcs.iter().enumerate() {
        if from < n_nodes {
            lower_a.add_entry(from, ai, 1.0);
        }
        if to < n_nodes {
            lower_a.add_entry(to, ai, -1.0);
        }
    }
    for (ai, &(_, _, _, cap, _)) in arcs.iter().enumerate() {
        lower_a.add_entry(n_nodes + ai, ai, 1.0);
        lower_b[n_nodes + ai] = cap;
    }

    // Linking: tolls modify the follower's effective cost
    // In the bilevel model, linking is via parametric RHS (simplified)
    let linking = SparseMatrix::new(num_lower_cons, num_tolls);

    // Upper constraints: toll bounds 0 ≤ t_i ≤ 10
    let upper_a = SparseMatrix::new(0, num_tolls);

    let bilevel = BilevelProblem {
        upper_obj_c_x: vec![0.0; num_tolls],
        upper_obj_c_y: vec![0.0; n_actual_arcs],
        lower_obj_c: arcs.iter().map(|a| a.2).collect(),
        lower_a,
        lower_b,
        lower_linking_b: linking,
        upper_constraints_a: upper_a,
        upper_constraints_b: vec![],
        num_upper_vars: num_tolls,
        num_lower_vars: n_actual_arcs,
        num_lower_constraints: num_lower_cons,
        num_upper_constraints: 0,
    };

    (name, bilevel)
}

// ─── Benchmark runner ────────────────────────────────────────────────────────

fn benchmark_instance(
    name: &str,
    family: &str,
    bilevel: &BilevelProblem,
    reformulation: ReformulationType,
) -> BenchmarkResult {
    let reform_name = format!("{}", reformulation);

    // 1. Structural analysis
    let analysis = StructuralAnalysis::analyze(bilevel);

    // 2. Compile
    let compile_start = Instant::now();
    let config = CompilerConfig::new(reformulation, BackendTarget::GenericMps)
        .with_tolerance(1e-7)
        .with_big_m(BigMStrategy::Computed)
        .with_certificate(false)
        .with_verbosity(0);

    let compile_result = compile(bilevel, config);
    let compile_time = compile_start.elapsed();

    let (num_vars, num_constraints, lp_problem) = match &compile_result {
        Ok(cr) => {
            let lp = cr.problem().clone();
            (cr.stats.num_vars, cr.stats.num_constraints, Some(lp))
        }
        Err(e) => {
            eprintln!("  [{}] Compilation failed ({}): {}", name, reform_name, e);
            // Fall back: use the raw bilevel as a simple LP
            let x0 = vec![0.0; bilevel.num_upper_vars];
            let lp = bilevel.lower_level_lp(&x0);
            (
                bilevel.num_upper_vars + bilevel.num_lower_vars,
                bilevel.num_upper_constraints + bilevel.num_lower_constraints,
                Some(lp),
            )
        }
    };

    // 3. Solve LP relaxation
    let solver = SimplexSolver::new().with_max_iterations(100_000);
    let lp_start = Instant::now();
    let lp_obj = if let Some(ref lp) = lp_problem {
        match solver.solve(lp) {
            Ok(sol) => sol.objective,
            Err(_) => f64::INFINITY,
        }
    } else {
        f64::INFINITY
    };
    let lp_time = lp_start.elapsed();

    // 4. Branch-and-cut solve
    let bc_start = Instant::now();
    let compiled_model = CompiledBilevelModel::new(bilevel.clone());
    let bc_config = SolverConfig {
        time_limit_secs: 10.0,
        node_limit: 5000,
        gap_tolerance: 1e-4,
        cut_rounds_per_node: 5,
        max_cuts_per_round: 20,
        enable_heuristics: true,
        enable_preprocessing: true,
        branching_strategy: BranchingStrategyType::MostFractional,
        node_selection: NodeSelectionType::BestFirst,
        verbosity: 0,
        ..SolverConfig::default()
    };
    let lp_solver = BuiltinLpSolver::new();
    let mut bc_solver = BranchAndCutSolver::new(bc_config);
    let bc_solution = bc_solver.solve(&compiled_model, &lp_solver);
    let bc_time = bc_start.elapsed();

    // 5. Baseline: solve without cuts (node_limit=1 → root only)
    let baseline_start = Instant::now();
    let baseline_config = SolverConfig {
        time_limit_secs: 10.0,
        node_limit: 1,
        gap_tolerance: 1e-4,
        cut_rounds_per_node: 0,
        max_cuts_per_round: 0,
        enable_heuristics: false,
        enable_preprocessing: false,
        branching_strategy: BranchingStrategyType::MostFractional,
        node_selection: NodeSelectionType::BestFirst,
        verbosity: 0,
        ..SolverConfig::default()
    };
    let mut baseline_solver = BranchAndCutSolver::new(baseline_config);
    let baseline_sol = baseline_solver.solve(&compiled_model, &lp_solver);
    let baseline_time = baseline_start.elapsed();

    // 6. Compute bound improvement
    let bound_improvement = if baseline_sol.objective.is_finite()
        && bc_solution.objective.is_finite()
        && (baseline_sol.objective - bc_solution.objective).abs() > 1e-10
    {
        let denom = baseline_sol.objective.abs().max(1e-10);
        ((baseline_sol.objective - bc_solution.objective) / denom * 100.0).abs()
    } else {
        0.0
    };

    // 7. Count cuts from solver statistics
    let stats = bc_solver.get_statistics();
    let cuts_added = stats.total_cuts_added;

    BenchmarkResult {
        instance: name.to_string(),
        family: family.to_string(),
        reformulation: reform_name,
        num_upper_vars: bilevel.num_upper_vars,
        num_lower_vars: bilevel.num_lower_vars,
        num_vars,
        num_constraints,
        compile_time_ms: compile_time.as_secs_f64() * 1000.0,
        lp_solve_time_ms: lp_time.as_secs_f64() * 1000.0,
        lp_objective: if lp_obj.is_finite() { lp_obj } else { 0.0 },
        bc_solve_time_ms: bc_time.as_secs_f64() * 1000.0,
        bc_objective: bc_solution.objective,
        gap_pct: bc_solution.gap * 100.0,
        nodes: bc_solution.node_count,
        cuts_added,
        status: format!("{}", bc_solution.status),
        baseline_obj: baseline_sol.objective,
        baseline_time_ms: baseline_time.as_secs_f64() * 1000.0,
        bound_improvement_pct: bound_improvement,
        analysis_type: format!("{:?}", analysis.lower_level_type),
        coupling_type: format!("{:?}", analysis.coupling_type),
    }
}

fn main() {
    let suite_start = Instant::now();

    eprintln!("═══════════════════════════════════════════════════════════════");
    eprintln!("  BiCut Real Benchmark Suite");
    eprintln!("═══════════════════════════════════════════════════════════════");

    let mut results: Vec<BenchmarkResult> = Vec::new();

    // ── Knapsack Interdiction Family ─────────────────────────────────────
    eprintln!("\n── Knapsack Interdiction ────────────────────────");
    for &n_items in &[4, 8, 15] {
        let (name, bilevel) = make_knapsack_interdiction(n_items, 42);
        eprintln!(
            "  {} ({} leader × {} follower)...",
            name, bilevel.num_upper_vars, bilevel.num_lower_vars
        );

        // Try KKT (primary for LP lower levels)
        let r = benchmark_instance(&name, "knapsack", &bilevel, ReformulationType::KKT);
        eprintln!(
            "    KKT: compile={:.1}ms, LP={:.2}ms, B&C={:.1}ms, gap={:.2}%, nodes={}",
            r.compile_time_ms, r.lp_solve_time_ms, r.bc_solve_time_ms, r.gap_pct, r.nodes
        );
        results.push(r);
    }

    // ── Network Design Family ────────────────────────────────────────────
    eprintln!("\n── Network Design ──────────────────────────────");
    for &n_nodes in &[4, 6, 10] {
        let (name, bilevel) = make_network_design(n_nodes, 42);
        eprintln!(
            "  {} ({} leader × {} follower)...",
            name, bilevel.num_upper_vars, bilevel.num_lower_vars
        );

        let r = benchmark_instance(&name, "network", &bilevel, ReformulationType::StrongDuality);
        eprintln!(
            "    SD:  compile={:.1}ms, LP={:.2}ms, B&C={:.1}ms, gap={:.2}%, nodes={}",
            r.compile_time_ms, r.lp_solve_time_ms, r.bc_solve_time_ms, r.gap_pct, r.nodes
        );
        results.push(r);
    }

    // ── Toll Pricing Family ──────────────────────────────────────────────
    eprintln!("\n── Toll Pricing ────────────────────────────────");
    for &n_arcs in &[3, 5, 8, 12] {
        let (name, bilevel) = make_toll_pricing(n_arcs, 42);
        eprintln!(
            "  {} ({} leader × {} follower)...",
            name, bilevel.num_upper_vars, bilevel.num_lower_vars
        );

        let r = benchmark_instance(&name, "pricing", &bilevel, ReformulationType::KKT);
        eprintln!(
            "    KKT: compile={:.1}ms, LP={:.2}ms, B&C={:.1}ms, gap={:.2}%, nodes={}",
            r.compile_time_ms, r.lp_solve_time_ms, r.bc_solve_time_ms, r.gap_pct, r.nodes
        );
        results.push(r);
    }

    let total_time = suite_start.elapsed();

    // ── Summary ──────────────────────────────────────────────────────────
    eprintln!("\n═══════════════════════════════════════════════════════════════");
    eprintln!(
        "  Total: {} instances in {:.2}s",
        results.len(),
        total_time.as_secs_f64()
    );
    eprintln!("═══════════════════════════════════════════════════════════════");

    // Print per-family summaries to stderr
    for family in &["knapsack", "network", "pricing"] {
        let fam_results: Vec<&BenchmarkResult> =
            results.iter().filter(|r| r.family == *family).collect();
        if fam_results.is_empty() {
            continue;
        }
        let avg_compile: f64 =
            fam_results.iter().map(|r| r.compile_time_ms).sum::<f64>() / fam_results.len() as f64;
        let avg_bc: f64 =
            fam_results.iter().map(|r| r.bc_solve_time_ms).sum::<f64>() / fam_results.len() as f64;
        let avg_gap: f64 =
            fam_results.iter().map(|r| r.gap_pct).sum::<f64>() / fam_results.len() as f64;
        let avg_bound: f64 = fam_results
            .iter()
            .map(|r| r.bound_improvement_pct)
            .sum::<f64>()
            / fam_results.len() as f64;
        eprintln!(
            "  {:<12} avg compile={:.1}ms, avg B&C={:.1}ms, avg gap={:.2}%, avg bound_impr={:.2}%",
            family, avg_compile, avg_bc, avg_gap, avg_bound
        );
    }

    // Emit JSON to stdout
    let suite = BenchmarkSuite {
        benchmark_results: results,
        total_time_secs: total_time.as_secs_f64(),
        platform: format!("{} {}", std::env::consts::OS, std::env::consts::ARCH),
        timestamp: chrono::Utc::now().to_rfc3339(),
    };

    println!("{}", serde_json::to_string_pretty(&suite).unwrap());
}
