//! Criterion benchmarks for the spectral decomposition oracle.
//!
//! Run with: cargo bench
//!
//! Benchmarks cover the critical path: matrix construction, eigensolve,
//! feature extraction, and structure detection.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use spectral_types::sparse::CooMatrix;
use spectral_types::mip::{MipInstance, VariableType, ConstraintSense};

/// Generate a synthetic block-diagonal MIP instance.
fn make_block_diagonal(n_blocks: usize, block_size: usize) -> MipInstance {
    let n_rows = n_blocks * block_size;
    let n_cols = n_blocks * block_size * 2;
    let mut inst = MipInstance::new("bench_block_diagonal", n_cols, n_rows);

    let mut coo = CooMatrix::new(n_rows, n_cols);
    for b in 0..n_blocks {
        let r0 = b * block_size;
        let c0 = b * block_size * 2;
        for i in 0..block_size {
            for off in 0..3.min(block_size * 2) {
                let j = (i * 2 + off) % (block_size * 2);
                coo.push(r0 + i, c0 + j, 1.0 + (i as f64) * 0.01);
            }
        }
        // Weak cross-block coupling
        if b + 1 < n_blocks {
            coo.push(r0, (b + 1) * block_size * 2, 0.001);
        }
    }

    inst.constraint_matrix = coo.to_csr();
    inst.objective = vec![1.0; n_cols];
    inst.rhs = vec![10.0; n_rows];
    inst.senses = vec![ConstraintSense::Le; n_rows];
    inst.var_types = vec![VariableType::Continuous; n_cols];
    inst.lower_bounds = vec![0.0; n_cols];
    inst.upper_bounds = vec![f64::INFINITY; n_cols];
    inst.var_names = (0..n_cols).map(|i| format!("x{}", i)).collect();
    inst.con_names = (0..n_rows).map(|i| format!("c{}", i)).collect();
    inst
}

fn bench_hypergraph_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("hypergraph_construction");

    for &n_blocks in &[2, 4, 8, 16] {
        let block_size = 25;
        let inst = make_block_diagonal(n_blocks, block_size);
        let label = format!("{}x{}", n_blocks, block_size);

        group.bench_with_input(
            BenchmarkId::new("build", &label),
            &inst,
            |b, inst| {
                b.iter(|| {
                    let _ = black_box(
                        spectral_core::build_constraint_hypergraph(inst)
                    );
                });
            },
        );
    }
    group.finish();
}

fn bench_eigensolve(c: &mut Criterion) {
    let mut group = c.benchmark_group("eigensolve");
    group.sample_size(10);

    for &size in &[50, 100, 200] {
        let inst = make_block_diagonal(4, size / 4);
        let hg = spectral_core::build_constraint_hypergraph(&inst).unwrap();
        let config = spectral_core::hypergraph::LaplacianConfig::default();
        let laplacian = spectral_core::build_normalized_laplacian(
            &hg.hypergraph, &config,
        ).unwrap();

        let eigen_config = spectral_core::eigensolve::EigenConfig {
            num_eigenvalues: 8,
            tolerance: 1e-8,
            max_iter: 500,
            ..Default::default()
        };

        group.bench_with_input(
            BenchmarkId::new("lanczos_k8", size),
            &laplacian,
            |b, lap| {
                let solver = spectral_core::EigenSolver::new(eigen_config.clone());
                b.iter(|| {
                    let _ = black_box(solver.solve(lap));
                });
            },
        );
    }
    group.finish();
}

fn bench_spectral_features(c: &mut Criterion) {
    use spectral_core::features::spectral_features::*;

    let mut group = c.benchmark_group("spectral_features");

    let inst = make_block_diagonal(4, 50);
    let hg = spectral_core::build_constraint_hypergraph(&inst).unwrap();
    let config = spectral_core::hypergraph::LaplacianConfig::default();
    let lap = spectral_core::build_normalized_laplacian(&hg.hypergraph, &config).unwrap();
    let eigen_config = spectral_core::eigensolve::EigenConfig {
        num_eigenvalues: 8, ..Default::default()
    };
    let solver = spectral_core::EigenSolver::new(eigen_config);
    let result = solver.solve(&lap).unwrap();

    group.bench_function("spectral_gap", |b| {
        b.iter(|| black_box(compute_spectral_gap(&result.eigenvalues)));
    });

    group.bench_function("eigenvalue_decay", |b| {
        b.iter(|| black_box(compute_eigenvalue_decay_rate(&result.eigenvalues)));
    });

    group.bench_function("fiedler_entropy", |b| {
        b.iter(|| black_box(compute_fiedler_entropy(&result.eigenvectors)));
    });

    group.bench_function("effective_dimension", |b| {
        b.iter(|| black_box(compute_effective_dimension(&result.eigenvalues)));
    });

    group.finish();
}

fn bench_mps_parsing(c: &mut Criterion) {
    let mut group = c.benchmark_group("mps_parsing");

    // Generate a small MPS string for benchmarking the parser
    let mps_content = generate_small_mps(50, 100);

    group.bench_function("parse_50x100", |b| {
        b.iter(|| {
            let _ = black_box(spectral_types::mip::read_mps(&mps_content));
        });
    });

    let mps_large = generate_small_mps(200, 400);
    group.bench_function("parse_200x400", |b| {
        b.iter(|| {
            let _ = black_box(spectral_types::mip::read_mps(&mps_large));
        });
    });

    group.finish();
}

fn generate_small_mps(n_rows: usize, n_cols: usize) -> String {
    let mut s = String::new();
    s.push_str("NAME          bench\n");
    s.push_str("ROWS\n");
    s.push_str(" N  obj\n");
    for i in 0..n_rows {
        s.push_str(&format!(" L  c{}\n", i));
    }
    s.push_str("COLUMNS\n");
    for j in 0..n_cols {
        s.push_str(&format!("    x{:<8}obj           {:.1}\n", j, 1.0));
        // Each variable appears in ~3 constraints
        for off in 0..3 {
            let row = (j * 3 + off) % n_rows;
            s.push_str(&format!("    x{:<8}c{:<12}{:.1}\n", j, row, 1.0 + off as f64 * 0.1));
        }
    }
    s.push_str("RHS\n");
    for i in 0..n_rows {
        s.push_str(&format!("    rhs       c{:<12}{:.1}\n", i, 10.0));
    }
    s.push_str("BOUNDS\n");
    for j in 0..n_cols {
        s.push_str(&format!(" UP bound     x{:<12}{:.1}\n", j, 100.0));
    }
    s.push_str("ENDATA\n");
    s
}

criterion_group!(
    benches,
    bench_hypergraph_construction,
    bench_eigensolve,
    bench_spectral_features,
    bench_mps_parsing,
);
criterion_main!(benches);
