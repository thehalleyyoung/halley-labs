//! Example: Solver backend comparison
//!
//! Demonstrates using different solver backends (internal, SCIP emulation,
//! HiGHS emulation) to solve an LP relaxation.
//!
//! Run: cargo run --example solver_comparison

use optimization::solver_interface::{SolverConfig, SolverType, create_solver};
use optimization::lp::{LpProblem, ConstraintType};

fn main() {
    println!("=== SpectralOracle — Solver Backend Comparison ===\n");

    // Build a small LP: min -2x1 - 3x2 s.t. x1 + x2 <= 4, x1 <= 3, x2 <= 3
    let lp = build_test_lp();
    println!("Problem: min -2x₁ - 3x₂");
    println!("  s.t.  x₁ + x₂ ≤ 4");
    println!("        x₁ ≤ 3,  x₂ ≤ 3");
    println!("        x₁, x₂ ≥ 0\n");

    let backends = vec![
        ("Internal Simplex", SolverType::InternalSimplex),
        ("Internal Interior Point", SolverType::InternalInteriorPoint),
        ("SCIP Emulation", SolverType::ScipEmulation),
        ("HiGHS Emulation", SolverType::HiGHS),
    ];

    println!("{:<28} {:>12} {:>10} {:>10}",
        "Backend", "Objective", "x₁", "x₂");
    println!("{}", "-".repeat(62));

    for (name, solver_type) in backends {
        let config = SolverConfig::default()
            .with_type(solver_type)
            .with_time_limit(60.0);
        let mut solver = create_solver(config);

        match solver.solve_lp(&lp) {
            Ok(solution) => {
                let x1 = solution.primal_values.get(0).copied().unwrap_or(f64::NAN);
                let x2 = solution.primal_values.get(1).copied().unwrap_or(f64::NAN);
                println!("{:<28} {:>12.6} {:>10.4} {:>10.4}",
                    name, solution.objective_value, x1, x2);
            }
            Err(e) => {
                println!("{:<28} ERROR: {}", name, e);
            }
        }
    }

    println!("\nOptimal: z* = -11 at x₁=1, x₂=3  (or nearby)\n");

    // Demonstrate feature flags
    println!("Available backends:");
    println!("  ✓ Internal Simplex (always available)");
    println!("  ✓ Internal Interior Point (always available)");
    println!("  ✓ SCIP Emulation (always available)");
    println!("  ✓ HiGHS Emulation (always available)");
    if cfg!(feature = "scip") {
        println!("  ✓ SCIP (via russcip, enabled)");
    } else {
        println!("  ○ SCIP (via russcip, enable with --features scip)");
    }
    if cfg!(feature = "highs") {
        println!("  ✓ HiGHS (via highs crate, enabled)");
    } else {
        println!("  ○ HiGHS (via highs crate, enable with --features highs)");
    }

    println!("\n=== Done ===");
}

fn build_test_lp() -> LpProblem {
    let mut lp = LpProblem::new(false); // minimize
    lp.add_variable(-2.0, 0.0, f64::INFINITY, None); // x1
    lp.add_variable(-3.0, 0.0, f64::INFINITY, None); // x2

    lp.add_constraint(&[0, 1], &[1.0, 1.0], ConstraintType::Le, 4.0).unwrap();
    lp.add_constraint(&[0], &[1.0], ConstraintType::Le, 3.0).unwrap();
    lp.add_constraint(&[1], &[1.0], ConstraintType::Le, 3.0).unwrap();
    lp
}
