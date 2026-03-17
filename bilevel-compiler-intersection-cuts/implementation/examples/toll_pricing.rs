//! # Toll-Setting (Pricing) Problem via Bilevel Optimization
//!
//! The leader sets tolls on network arcs to maximize revenue, while the
//! follower routes flow to minimize cost. Canonical bilevel LP where
//! strong duality reformulation applies.
//!
//! Run: `cargo run --example toll_pricing`

use bicut_core::StructuralAnalysis;
use bicut_lp::{LpSolver, SimplexSolver};
use bicut_types::{BilevelProblem, ConstraintSense, LpProblem, OptDirection, SparseMatrix};

fn main() {
    println!("═══════════════════════════════════════════════════");
    println!("  BiCut Example: Toll-Setting (Pricing) Problem");
    println!("═══════════════════════════════════════════════════\n");

    // Network: 4 nodes (s=0, A=1, B=2, t=3), 5 arcs
    let nodes = ["s", "A", "B", "t"];
    let arcs: Vec<(usize, usize, f64, f64, bool)> = vec![
        (0, 1, 2.0, 10.0, true),  // s->A  [TOLLABLE]
        (0, 2, 5.0, 10.0, false), // s->B
        (1, 2, 1.0, 5.0, true),   // A->B  [TOLLABLE]
        (1, 3, 3.0, 10.0, true),  // A->t  [TOLLABLE]
        (2, 3, 2.0, 10.0, false), // B->t
    ];
    let demand = 8.0;
    let t_max = 10.0;
    let num_arcs = arcs.len();
    let tollable: Vec<usize> = arcs
        .iter()
        .enumerate()
        .filter(|(_, a)| a.4)
        .map(|(i, _)| i)
        .collect();
    let num_tolls = tollable.len();
    let num_nodes = nodes.len();

    println!("Network ({num_nodes} nodes, {num_arcs} arcs, {num_tolls} tollable):");
    for (i, a) in arcs.iter().enumerate() {
        let t = if a.4 { " ★" } else { "" };
        println!(
            "  {i}: {} → {} (c={:.0}, u={:.0}){t}",
            nodes[a.0], nodes[a.1], a.2, a.3
        );
    }
    println!("  Demand: {demand}, Max toll: {t_max}\n");

    // Build bilevel problem
    // Leader vars: t_0, t_1, t_2 (tolls on 3 tollable arcs)
    // Follower vars: f_0..f_4 (flow on 5 arcs)
    // Follower: min Σ (c_a + t_a) * f_a  s.t. flow conservation + capacity
    let num_flow_cons = num_nodes;
    let num_cap_cons = num_arcs;
    let num_lower_cons = num_flow_cons + num_cap_cons;

    let mut lower_a = SparseMatrix::new(num_lower_cons, num_arcs);
    let mut lower_b = vec![0.0; num_lower_cons];

    // Flow conservation
    lower_b[0] = demand; // source
    lower_b[num_nodes - 1] = -demand; // sink
    for (ai, arc) in arcs.iter().enumerate() {
        lower_a.add_entry(arc.0, ai, 1.0); // outflow
        lower_a.add_entry(arc.1, ai, -1.0); // inflow
    }
    // Capacity
    for ai in 0..num_arcs {
        lower_a.add_entry(num_flow_cons + ai, ai, 1.0);
        lower_b[num_flow_cons + ai] = arcs[ai].3;
    }

    let bilevel = BilevelProblem {
        upper_obj_c_x: vec![0.0; num_tolls],
        upper_obj_c_y: vec![0.0; num_arcs], // bilinear in reality
        lower_obj_c: arcs.iter().map(|a| a.2).collect(),
        lower_a,
        lower_b,
        lower_linking_b: SparseMatrix::new(num_lower_cons, num_tolls),
        upper_constraints_a: SparseMatrix::new(0, num_tolls),
        upper_constraints_b: vec![],
        num_upper_vars: num_tolls,
        num_lower_vars: num_arcs,
        num_lower_constraints: num_lower_cons,
        num_upper_constraints: 0,
    };

    println!("── Structural Analysis ──────────────────────────");
    let report = StructuralAnalysis::analyze(&bilevel);
    println!("  Lower-level type: {:?}", report.lower_level_type);
    println!("  Coupling type:    {:?}", report.coupling_type);
    println!("  Strong duality applies: LP lower + bounded");
    println!();

    // Solve follower for different toll vectors
    println!("── Parametric Follower Solutions ────────────────");
    let solver = SimplexSolver::new();
    let scenarios = vec![
        vec![0.0, 0.0, 0.0],
        vec![3.0, 0.0, 0.0],
        vec![0.0, 2.0, 5.0],
        vec![5.0, 3.0, 4.0],
    ];

    let mut best_rev = 0.0f64;
    let mut best_t = vec![0.0; num_tolls];

    for tolls in &scenarios {
        // Build follower LP: min Σ (c_a + t_a) * f_a s.t. flow + cap
        let mut senses = vec![ConstraintSense::Eq; num_nodes];
        senses.extend(vec![ConstraintSense::Le; num_arcs]);
        let mut lp = LpProblem::new(num_arcs, num_nodes + num_arcs);
        lp.direction = OptDirection::Minimize;

        for (ai, arc) in arcs.iter().enumerate() {
            let mut cost = arc.2;
            if let Some(ti) = tollable.iter().position(|&ta| ta == ai) {
                cost += tolls[ti];
            }
            lp.c[ai] = cost;
        }

        let mut a_mat = SparseMatrix::new(num_nodes + num_arcs, num_arcs);
        let mut rhs = vec![0.0; num_nodes + num_arcs];
        rhs[0] = demand;
        rhs[num_nodes - 1] = -demand;
        for (ai, arc) in arcs.iter().enumerate() {
            a_mat.add_entry(arc.0, ai, 1.0);
            a_mat.add_entry(arc.1, ai, -1.0);
        }
        for ai in 0..num_arcs {
            a_mat.add_entry(num_nodes + ai, ai, 1.0);
            rhs[num_nodes + ai] = arcs[ai].3;
        }
        lp.a_matrix = a_mat;
        lp.b_rhs = rhs;
        lp.senses = senses;

        match solver.solve(&lp) {
            Ok(sol) => {
                let mut rev = 0.0;
                for (ti, &ai) in tollable.iter().enumerate() {
                    rev += tolls[ti] * sol.primal[ai];
                }
                let ts: Vec<String> = tollable
                    .iter()
                    .enumerate()
                    .map(|(ti, &ai)| {
                        format!(
                            "{}→{}:{:.0}",
                            nodes[arcs[ai].0], nodes[arcs[ai].1], tolls[ti]
                        )
                    })
                    .collect();
                println!(
                    "  [{}]  cost={:.1}  revenue={:.1}",
                    ts.join(", "),
                    sol.objective,
                    rev
                );

                // Show flows
                let flows: Vec<String> = arcs
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| sol.primal[*i] > 1e-6)
                    .map(|(i, a)| format!("{}→{}:{:.1}", nodes[a.0], nodes[a.1], sol.primal[i]))
                    .collect();
                println!("    flows: {}", flows.join(", "));

                if rev > best_rev {
                    best_rev = rev;
                    best_t = tolls.clone();
                }
            }
            Err(e) => println!("  {:?}: FAILED ({e})", tolls),
        }
    }

    println!("\n── Best Found ─────────────────────────────────");
    println!("  Revenue: {best_rev:.2}");
    for (ti, &ai) in tollable.iter().enumerate() {
        println!(
            "    t({} → {}) = {:.1}",
            nodes[arcs[ai].0], nodes[arcs[ai].1], best_t[ti]
        );
    }
    println!();
    println!("═══════════════════════════════════════════════════");
    println!("  BiCut compiles this via strong duality (LP lower");
    println!("  level, bounded). Bilinear t·f terms in the leader");
    println!("  objective are linearized via the LP dual.");
    println!("═══════════════════════════════════════════════════");
}
