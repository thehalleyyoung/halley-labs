//! Benchmark: Solver comparison across SAT, SMT, MaxSMT, and ILP backends.
//!
//! Generates regulatory compliance problems of varying sizes and benchmarks
//! each solver backend on the same problem instances.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use regsynth_solver::{
    DpllSolver, MaxSmtSolver, IlpSolver, SoftClause,
    solve_cnf,
    SolverConfig, Clause, Literal, Variable,
};
use regsynth_encoding::{
    IlpModel, IlpVariable, IlpConstraint, IlpConstraintType, IlpObjective, ObjectiveSense,
};
use rand::Rng;
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Make a literal from a variable and polarity.
fn make_literal(var: Variable, positive: bool) -> Literal {
    if positive { var as Literal } else { -(var as Literal) }
}

/// Generate a random 3-SAT instance with `n` variables and `m` clauses.
fn random_3sat(n: u32, m: usize, seed: u64) -> (u32, Vec<Clause>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let clauses = (0..m)
        .map(|_| {
            (0..3)
                .map(|_| {
                    let var = rng.gen_range(1..=n);
                    let positive = rng.gen_bool(0.5);
                    make_literal(var, positive)
                })
                .collect::<Clause>()
        })
        .collect();
    (n, clauses)
}

/// Generate weighted soft clauses from a clause set.
fn to_soft_clauses(clauses: &[Clause], seed: u64) -> Vec<SoftClause> {
    let mut rng = StdRng::seed_from_u64(seed);
    clauses
        .iter()
        .enumerate()
        .map(|(i, c)| SoftClause {
            lits: c.clone(),
            weight: rng.gen_range(1.0..100.0),
            id: i,
        })
        .collect()
}

/// Build a random ILP model.
fn random_ilp_model(n_vars: usize, n_constraints: usize, seed: u64) -> IlpModel {
    let mut rng = StdRng::seed_from_u64(seed);
    let variables: Vec<IlpVariable> = (0..n_vars)
        .map(|i| IlpVariable {
            name: format!("x_{}", i),
            lower_bound: 0.0,
            upper_bound: 1.0,
            is_integer: true,
            is_binary: true,
        })
        .collect();

    let constraints: Vec<IlpConstraint> = (0..n_constraints)
        .map(|j| {
            let coeffs: Vec<(String, f64)> = (0..3)
                .map(|_| {
                    let var_idx = rng.gen_range(0..n_vars);
                    (format!("x_{}", var_idx), rng.gen_range(-10.0..10.0))
                })
                .collect();
            IlpConstraint {
                id: format!("c_{}", j),
                coefficients: coeffs,
                constraint_type: IlpConstraintType::Le,
                rhs: rng.gen_range(0.0..20.0),
                provenance: None,
            }
        })
        .collect();

    let obj_coeffs: Vec<(String, f64)> = (0..n_vars)
        .map(|i| (format!("x_{}", i), rng.gen_range(1.0..100.0)))
        .collect();

    IlpModel {
        variables,
        constraints,
        objective: IlpObjective {
            sense: ObjectiveSense::Minimize,
            coefficients: obj_coeffs,
            constant: 0.0,
        },
    }
}

/// Benchmark the CDCL SAT solver at different problem scales.
fn bench_sat_solver(c: &mut Criterion) {
    let mut group = c.benchmark_group("solver_sat_cdcl");
    let ratios: &[(u32, usize)] = &[(50, 200), (100, 430), (200, 860)];
    for &(n, m) in ratios {
        let (num_vars, clauses) = random_3sat(n, m, 42);
        group.bench_with_input(
            BenchmarkId::new("3sat", format!("{}v_{}c", n, m)),
            &(num_vars, clauses),
            |b, (nv, cls)| {
                b.iter(|| {
                    let mut solver = DpllSolver::new(*nv, SolverConfig::default());
                    for clause in cls {
                        solver.add_clause(clause.clone());
                    }
                    solver.solve()
                });
            },
        );
    }
    group.finish();
}

/// Benchmark the convenience `solve_cnf` function.
fn bench_solve_cnf(c: &mut Criterion) {
    let mut group = c.benchmark_group("solver_solve_cnf");
    for &(n, m) in &[(50u32, 200usize), (100, 430)] {
        let (num_vars, clauses) = random_3sat(n, m, 77);
        group.bench_with_input(
            BenchmarkId::new("cnf", format!("{}v_{}c", n, m)),
            &(num_vars, clauses),
            |b, (nv, cls)| {
                b.iter(|| solve_cnf(*nv, cls));
            },
        );
    }
    group.finish();
}

/// Benchmark the MaxSMT solver (weighted partial MaxSAT).
fn bench_maxsmt_solver(c: &mut Criterion) {
    let mut group = c.benchmark_group("solver_maxsmt");
    for &(n, m) in &[(30u32, 120usize), (50, 200), (100, 430)] {
        let (_num_vars, clauses) = random_3sat(n, m, 88);
        let hard_count = (clauses.len() * 3) / 10;
        let hard: Vec<Clause> = clauses[..hard_count].to_vec();
        let soft = to_soft_clauses(&clauses[hard_count..], 99);

        group.bench_with_input(
            BenchmarkId::new("maxsmt", format!("{}v_{}h_{}s", n, hard.len(), soft.len())),
            &(hard.clone(), soft.clone()),
            |b, (h, s)| {
                b.iter(|| {
                    let mut solver = MaxSmtSolver::new(SolverConfig::default());
                    solver.solve(h, s)
                });
            },
        );
    }
    group.finish();
}

/// Benchmark the ILP solver on small-to-medium regulatory problems.
fn bench_ilp_solver(c: &mut Criterion) {
    let mut group = c.benchmark_group("solver_ilp");
    for &n_vars in &[10usize, 25, 50] {
        let n_constraints = n_vars * 2;
        let model = random_ilp_model(n_vars, n_constraints, 42);
        group.bench_with_input(
            BenchmarkId::new("ilp", format!("{}v_{}c", n_vars, n_constraints)),
            &model,
            |b, m| {
                b.iter(|| {
                    let mut solver = IlpSolver::new(SolverConfig::default());
                    solver.solve(m)
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_sat_solver,
    bench_solve_cnf,
    bench_maxsmt_solver,
    bench_ilp_solver,
);
criterion_main!(benches);
