//! Benchmarks for BiCut's core operations: LP solving, structural analysis.
//!
//! Run: `cargo bench` from the implementation/ directory.

use bicut_core::StructuralAnalysis;
use bicut_lp::{LpSolver, SimplexSolver};
use bicut_types::{BilevelProblem, ConstraintSense, LpProblem, OptDirection, SparseMatrix};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

fn make_bilevel(n_upper: usize, n_lower: usize, n_cons: usize) -> BilevelProblem {
    let mut lower_a = SparseMatrix::new(n_cons, n_lower);
    for i in 0..n_cons {
        for j in 0..n_lower {
            let val = ((i * 7 + j * 13) % 20) as f64 / 10.0;
            if val > 0.5 {
                lower_a.add_entry(i, j, val);
            }
        }
    }
    let linking = SparseMatrix::new(n_cons, n_upper);
    BilevelProblem {
        upper_obj_c_x: vec![1.0; n_upper],
        upper_obj_c_y: vec![-2.0; n_lower],
        lower_obj_c: (0..n_lower).map(|j| (j as f64 + 1.0) * 0.5).collect(),
        lower_a,
        lower_b: (0..n_cons).map(|i| (i as f64 + 1.0) * 5.0).collect(),
        lower_linking_b: linking,
        upper_constraints_a: SparseMatrix::new(1, n_upper),
        upper_constraints_b: vec![n_upper as f64 * 5.0],
        num_upper_vars: n_upper,
        num_lower_vars: n_lower,
        num_lower_constraints: n_cons,
        num_upper_constraints: 1,
    }
}

fn make_lp(n: usize, m: usize) -> LpProblem {
    let mut lp = LpProblem::new(n, m);
    lp.direction = OptDirection::Minimize;
    lp.c = (0..n).map(|j| (j as f64 + 1.0)).collect();
    let mut a = SparseMatrix::new(m, n);
    for i in 0..m {
        for j in 0..n {
            let val = ((i + 1) * (j + 1)) as f64 % 7.0;
            if val > 2.0 {
                a.add_entry(i, j, val);
            }
        }
    }
    lp.a_matrix = a;
    lp.b_rhs = (0..m).map(|i| (i as f64 + 1.0) * 10.0).collect();
    lp.senses = vec![ConstraintSense::Le; m];
    lp
}

fn bench_simplex(c: &mut Criterion) {
    let mut group = c.benchmark_group("simplex_solve");
    let solver = SimplexSolver::new().with_max_iterations(100_000);
    for &(n, m) in &[(5, 3), (10, 8), (20, 15), (50, 30)] {
        let lp = make_lp(n, m);
        group.bench_with_input(
            BenchmarkId::new("simplex", format!("{n}x{m}")),
            &lp,
            |b, lp| b.iter(|| solver.solve(black_box(lp))),
        );
    }
    group.finish();
}

fn bench_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("structural_analysis");
    for &(nu, nl, nc) in &[(5, 5, 5), (10, 10, 10), (20, 30, 20)] {
        let prob = make_bilevel(nu, nl, nc);
        group.bench_with_input(
            BenchmarkId::new("analyze", format!("{nu}x{nl}x{nc}")),
            &prob,
            |b, p| b.iter(|| StructuralAnalysis::analyze(black_box(p))),
        );
    }
    group.finish();
}

fn bench_sparse(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_matrix");
    for &size in &[10, 50, 100] {
        group.bench_with_input(
            BenchmarkId::new("construct_to_dense", size),
            &size,
            |b, &s| {
                b.iter(|| {
                    let mut m = SparseMatrix::new(s, s);
                    for i in 0..s {
                        for j in 0..s {
                            if (i + j) % 3 == 0 {
                                m.add_entry(i, j, (i * j) as f64);
                            }
                        }
                    }
                    black_box(m.to_dense())
                })
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_simplex, bench_analysis, bench_sparse);
criterion_main!(benches);
