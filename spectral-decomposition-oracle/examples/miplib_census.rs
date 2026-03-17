//! Example: MIPLIB 2017 census demonstration
//!
//! Shows how to run a mini decomposition census on MIPLIB instances.
//! In practice, use the CLI: `spectral-oracle census --tier pilot`
//!
//! Run: cargo run --example miplib_census

use spectral_types::mip::{read_mps, MipInstance};

fn main() {
    println!("=== SpectralOracle — MIPLIB 2017 Census Demo ===\n");

    // Simulate census on synthetic instances (since MIPLIB files aren't bundled)
    let instances = generate_synthetic_census_instances();

    println!("Census: {} instances\n", instances.len());
    println!("{:<20} {:>6} {:>6} {:>8} {:>14} {:>12}",
        "Instance", "Rows", "Cols", "NNZ", "Spectral Gap", "Recommend");
    println!("{}", "-".repeat(72));

    for inst in &instances {
        // Build hypergraph and extract spectral features
        match spectral_core::build_constraint_hypergraph(inst) {
            Ok(hg_result) => {
                let config = spectral_core::hypergraph::LaplacianConfig::default();
                match spectral_core::build_normalized_laplacian(&hg_result.hypergraph, &config) {
                    Ok(laplacian) => {
                        let eigen_config = spectral_core::eigensolve::EigenConfig {
                            num_eigenvalues: 8,
                            tolerance: 1e-8,
                            max_iter: 500,
                            ..Default::default()
                        };
                        let solver = spectral_core::EigenSolver::new(eigen_config);
                        match solver.solve(&laplacian) {
                            Ok(eigen) => {
                                use spectral_core::features::spectral_features::compute_spectral_gap;
                                let gap = compute_spectral_gap(&eigen.eigenvalues);
                                let method = if gap > 0.5 {
                                    "None"
                                } else if gap > 0.1 {
                                    "Lagrangian"
                                } else if gap > 0.01 {
                                    "DantzigWolfe"
                                } else {
                                    "Benders"
                                };
                                println!("{:<20} {:>6} {:>6} {:>8} {:>14.6e} {:>12}",
                                    inst.name, inst.num_constraints, inst.num_variables,
                                    inst.nnz(), gap, method);
                            }
                            Err(_) => {
                                println!("{:<20} {:>6} {:>6} {:>8} {:>14} {:>12}",
                                    inst.name, inst.num_constraints, inst.num_variables,
                                    inst.nnz(), "FAIL", "-");
                            }
                        }
                    }
                    Err(_) => {
                        println!("{:<20} Laplacian construction failed", inst.name);
                    }
                }
            }
            Err(_) => {
                println!("{:<20} Hypergraph construction failed", inst.name);
            }
        }
    }

    println!("\n--- Census Summary ---");
    println!("To run the full MIPLIB 2017 census:");
    println!("  1. Download instances: https://miplib.zib.de/tag_benchmark.html");
    println!("  2. Place .mps files in miplib2017/");
    println!("  3. Run: spectral-oracle census --tier pilot");
    println!("\n=== Done ===");
}

/// Generate synthetic MIP instances mimicking different MIPLIB structures.
fn generate_synthetic_census_instances() -> Vec<MipInstance> {
    use spectral_types::sparse::CooMatrix;
    use spectral_types::mip::{VariableType, ConstraintSense};

    let configs = vec![
        ("block_angular_4", 4, 25, 0.001),
        ("block_angular_8", 8, 15, 0.001),
        ("tight_coupling", 2, 50, 0.5),
        ("staircase", 6, 20, 0.01),
        ("dense_small", 1, 100, 1.0),
    ];

    configs.into_iter().map(|(name, n_blocks, block_size, coupling)| {
        let n_rows = n_blocks * block_size;
        let n_cols = n_blocks * block_size * 2;
        let mut inst = MipInstance::new(name, n_cols, n_rows);
        let mut coo = CooMatrix::new(n_rows, n_cols);

        for b in 0..n_blocks {
            let r0 = b * block_size;
            let c0 = b * block_size * 2;
            for i in 0..block_size {
                for off in 0..4.min(block_size * 2) {
                    let j = (i * 2 + off) % (block_size * 2);
                    coo.push(r0 + i, c0 + j, 1.0 + (i as f64) * 0.01);
                }
            }
            if b + 1 < n_blocks {
                for k in 0..2 {
                    coo.push(r0 + k, (b + 1) * block_size * 2 + k, coupling);
                }
            }
        }

        inst.constraint_matrix = coo.to_csr();
        inst.objective = vec![1.0; n_cols];
        inst.rhs = vec![10.0; n_rows];
        inst.senses = vec![ConstraintSense::Le; n_rows];
        inst.var_types = (0..n_cols).map(|i| {
            if i % 3 == 0 { VariableType::Binary } else { VariableType::Continuous }
        }).collect();
        inst.lower_bounds = vec![0.0; n_cols];
        inst.upper_bounds = vec![f64::INFINITY; n_cols];
        inst.var_names = (0..n_cols).map(|i| format!("x{}", i)).collect();
        inst.con_names = (0..n_rows).map(|i| format!("c{}", i)).collect();
        inst
    }).collect()
}
