//! # Knapsack Interdiction via Bilevel Optimization
//!
//! A classic bilevel problem: the leader (interdictor) removes items to minimize
//! the follower's (packer's) knapsack profit. Demonstrates BiCut's structural
//! analysis and LP relaxation solving.
//!
//! Run: `cargo run --example knapsack_interdiction`

use bicut_core::StructuralAnalysis;
use bicut_lp::{LpSolver, SimplexSolver};
use bicut_types::{BilevelProblem, SparseMatrix};

fn main() {
    println!("═══════════════════════════════════════════════════");
    println!("  BiCut Example: Knapsack Interdiction Problem");
    println!("═══════════════════════════════════════════════════\n");

    let n = 6;
    let profits = [10.0, 8.0, 15.0, 4.0, 12.0, 7.0];
    let weights = [5.0, 4.0, 8.0, 3.0, 7.0, 2.0];
    let capacity = 15.0;
    let budget = 2;

    println!("Problem Setup:");
    println!("  Items: {n}, Capacity: {capacity}, Budget: {budget}");
    for i in 0..n {
        println!(
            "  Item {i}: profit={:.0}, weight={:.0}, p/w={:.2}",
            profits[i],
            weights[i],
            profits[i] / weights[i]
        );
    }
    println!();

    // Build bilevel problem in BiCut's flat representation
    // Leader vars: z_0..z_5 (binary interdiction)
    // Follower vars: x_0..x_5 (binary packing)
    //
    // Leader: min -Σ p_i·x_i   (minimize follower profit)
    // s.t.   Σ z_i ≤ k
    //
    // Follower: max Σ p_i·x_i
    // s.t.   Σ w_i·x_i ≤ W
    //        x_i ≤ 1 - z_i  (i.e., x_i + z_i ≤ 1)

    let num_follower_cons = 1 + n; // capacity + n interdiction coupling
    let mut lower_a = SparseMatrix::new(num_follower_cons, n);
    for i in 0..n {
        lower_a.add_entry(0, i, weights[i]); // capacity
    }
    for i in 0..n {
        lower_a.add_entry(1 + i, i, 1.0); // x_i ≤ 1 - z_i
    }
    let mut lower_b = vec![capacity];
    lower_b.extend(vec![1.0; n]); // RHS for coupling: x_i ≤ 1 (before linking)

    // Linking matrix B: for coupling constraints x_i + z_i ≤ 1
    // We encode z_i on RHS: lower_b[1+i] - z_i (linking moves z to RHS)
    let mut linking = SparseMatrix::new(num_follower_cons, n);
    for i in 0..n {
        linking.add_entry(1 + i, i, -1.0); // -z_i added to RHS
    }

    let upper_a = SparseMatrix::new(1, n);
    // Budget constraint handled separately
    let upper_b = vec![budget as f64];

    let bilevel = BilevelProblem {
        upper_obj_c_x: vec![0.0; n],                         // leader obj on z
        upper_obj_c_y: profits.iter().map(|p| -p).collect(), // leader obj on x (minimize -profit)
        lower_obj_c: profits.iter().map(|p| -p).collect(),   // follower min (-profit) = max profit
        lower_a,
        lower_b,
        lower_linking_b: linking,
        upper_constraints_a: upper_a,
        upper_constraints_b: upper_b,
        num_upper_vars: n,
        num_lower_vars: n,
        num_lower_constraints: num_follower_cons,
        num_upper_constraints: 1,
    };

    println!("── Structural Analysis ──────────────────────────");
    let report = StructuralAnalysis::analyze(&bilevel);
    println!("  Lower-level type: {:?}", report.lower_level_type);
    println!("  Coupling type:    {:?}", report.coupling_type);
    println!(
        "  Dimensions:       {} leader × {} follower",
        report.dimensions.num_leader_vars, report.dimensions.num_follower_vars
    );
    println!(
        "  Lower constraints: {}",
        report.dimensions.num_lower_constraints
    );
    println!();

    // Solve follower LP relaxation (no interdiction, relaxed integrality)
    println!("── Follower LP Relaxation (no interdiction) ─────");
    let x_none = vec![0.0; n]; // no interdiction
    let lp = bilevel.lower_level_lp(&x_none);
    let solver = SimplexSolver::new().with_max_iterations(50_000);
    match solver.solve(&lp) {
        Ok(sol) => {
            let obj = -sol.objective; // negate: we minimized -profit
            println!("  LP relaxation profit: {obj:.2}");
            for (i, &val) in sol.primal.iter().enumerate() {
                if val > 1e-6 {
                    println!(
                        "    x_{i} = {val:.4} (p={:.0}, w={:.0})",
                        profits[i], weights[i]
                    );
                }
            }
        }
        Err(e) => println!("  LP failed: {e}"),
    }
    println!();

    // Brute-force bilevel enumeration
    println!("── Bilevel Enumeration (brute-force) ────────────");
    let mut best_obj = f64::INFINITY;
    let mut best_z = vec![0u8; n];

    for mask in 0u32..(1 << n) {
        let z: Vec<u8> = (0..n).map(|i| ((mask >> i) & 1) as u8).collect();
        let cnt: u8 = z.iter().sum();
        if cnt as usize > budget {
            continue;
        }

        // Greedy integer knapsack
        let mut profit = 0.0;
        let mut rem = capacity;
        let mut items: Vec<(usize, f64)> = (0..n)
            .filter(|&i| z[i] == 0)
            .map(|i| (i, profits[i] / weights[i]))
            .collect();
        items.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        for (i, _) in &items {
            if weights[*i] <= rem {
                profit += profits[*i];
                rem -= weights[*i];
            }
        }
        if profit < best_obj {
            best_obj = profit;
            best_z = z;
        }
    }

    let interdicted: Vec<usize> = best_z
        .iter()
        .enumerate()
        .filter(|(_, &z)| z == 1)
        .map(|(i, _)| i)
        .collect();
    println!("  Optimal interdiction: items {:?}", interdicted);
    for &i in &interdicted {
        println!(
            "    Remove item {i} (p={:.0}, w={:.0})",
            profits[i], weights[i]
        );
    }
    let remaining: Vec<usize> = (0..n).filter(|i| best_z[*i] == 0).collect();
    println!("  Remaining items: {:?}", remaining);
    println!("  Follower's best profit: {best_obj:.1}");
    println!();

    // Solve follower LP with optimal interdiction
    println!("── Follower LP with Optimal Interdiction ────────");
    let x_opt: Vec<f64> = best_z.iter().map(|&z| z as f64).collect();
    let lp_opt = bilevel.lower_level_lp(&x_opt);
    match solver.solve(&lp_opt) {
        Ok(sol) => {
            println!("  Follower LP bound: {:.2}", -sol.objective);
        }
        Err(e) => println!("  LP failed: {e}"),
    }
    println!();

    println!("═══════════════════════════════════════════════════");
    println!("  BiCut compiles this via value-function reform-");
    println!("  ulation (integer lower level), then applies");
    println!("  bilevel intersection cuts for LP tightening.");
    println!("═══════════════════════════════════════════════════");
}
