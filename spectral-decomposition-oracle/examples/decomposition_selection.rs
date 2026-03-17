//! Example: Full oracle pipeline for decomposition method selection
//!
//! Demonstrates the end-to-end workflow:
//! 1. Load a constraint matrix
//! 2. Extract spectral + syntactic features
//! 3. Run the oracle classifier to select a decomposition method
//! 4. Check futility prediction
//! 5. Execute the selected decomposition
//! 6. Generate a certificate report
//!
//! Run: cargo run --example decomposition_selection

use spectral_types::sparse::CooMatrix;
use spectral_types::config::NumericalConfig;
use spectral_core::hypergraph::ConstraintHypergraph;
use spectral_core::laplacian::NormalizedLaplacian;
use spectral_core::features::SpectralFeatures;
use spectral_core::eigensolve::{EigensolveConfig, EigensolveFallback};
use oracle::classifier::EnsembleClassifier;
use oracle::features::SyntacticFeatures;
use oracle::futility::FutilityPredictor;
use optimization::decomposition::{DecompositionMethod, DecompositionResult};
use optimization::benders::BendersDecomposition;
use optimization::dw::DantzigWolfeDecomposition;
use optimization::bundle::LagrangianBundle;
use optimization::lp::LpSolver;
use certificate::verification::DualChecker;
use certificate::spectral_bound::ScalingLaw;
use certificate::report::CertificateReport;

/// Decomposition method label returned by the oracle.
#[derive(Debug, Clone, Copy)]
enum MethodChoice {
    Benders,
    DantzigWolfe,
    Lagrangian,
}

fn main() {
    println!("=== Spectral Decomposition Oracle — Full Pipeline ===\n");

    let config = NumericalConfig::default();

    // ---------------------------------------------------------------
    // Step 1: Build a synthetic constraint matrix
    // ---------------------------------------------------------------
    let (n_rows, n_cols) = (200, 400);
    let mut coo = CooMatrix::new(n_rows, n_cols);
    populate_block_angular(&mut coo, n_rows, n_cols, 4);

    println!("Constraint matrix: {}×{}, nnz={}", n_rows, n_cols, coo.nnz());

    let csr = coo.to_csr();

    // ---------------------------------------------------------------
    // Step 2: Extract spectral features
    // ---------------------------------------------------------------
    let hypergraph = ConstraintHypergraph::from_constraint_matrix(&csr);
    let laplacian = NormalizedLaplacian::auto(&hypergraph);

    let eigen_config = EigensolveConfig {
        num_eigenvalues: 8,
        tolerance: config.eigenvalue_tol,
        max_iterations: 1000,
        ..Default::default()
    };
    let eigensolver = EigensolveFallback::new(eigen_config);
    let eigen_result = eigensolver.solve(&laplacian).expect("Eigensolve failed");

    let spectral = SpectralFeatures::extract(&eigen_result);
    println!("Spectral gap: {:.4e}", spectral.spectral_gap);
    println!("Cheeger est:  {:.4e}", spectral.cheeger_estimate);

    // ---------------------------------------------------------------
    // Step 3: Extract syntactic features
    // ---------------------------------------------------------------
    let syntactic = SyntacticFeatures::from_matrix(&csr);
    println!(
        "Syntactic: rows={}, cols={}, density={:.4e}",
        syntactic.num_rows, syntactic.num_cols, syntactic.density
    );

    // ---------------------------------------------------------------
    // Step 4: Oracle classification
    // ---------------------------------------------------------------
    let classifier = EnsembleClassifier::load_pretrained()
        .expect("Failed to load pretrained classifier");

    let feature_vec = classifier.combine_features(&spectral, &syntactic);
    let prediction = classifier.predict(&feature_vec);

    let method = match prediction.class_label {
        0 => MethodChoice::Benders,
        1 => MethodChoice::DantzigWolfe,
        2 => MethodChoice::Lagrangian,
        _ => panic!("Unknown class label"),
    };

    println!(
        "\nOracle recommendation: {:?} (confidence: {:.1}%)",
        method,
        prediction.confidence * 100.0
    );
    println!("Class probabilities: {:?}", prediction.class_probabilities);

    // ---------------------------------------------------------------
    // Step 5: Futility check
    // ---------------------------------------------------------------
    let futility = FutilityPredictor::load_pretrained()
        .expect("Failed to load futility model");

    let futility_prob = futility.predict_futility(&feature_vec);
    println!("Futility probability: {:.1}%", futility_prob * 100.0);

    if futility_prob > 0.8 {
        println!("WARNING: High futility probability — decomposition unlikely to help.");
        println!("Consider solving the LP directly instead.");
        // In production, you might bail out here.
    }

    // ---------------------------------------------------------------
    // Step 6: Run the selected decomposition
    // ---------------------------------------------------------------
    println!("\n--- Running {:?} decomposition ---", method);

    // First solve the LP relaxation for reference
    let lp_result = LpSolver::auto(&csr, &config)
        .solve()
        .expect("LP solve failed");
    println!("LP relaxation: z_LP = {:.6}", lp_result.objective);

    let decomp_result: DecompositionResult = match method {
        MethodChoice::Benders => {
            BendersDecomposition::new(&csr, &spectral, &config)
                .solve()
                .expect("Benders failed")
        }
        MethodChoice::DantzigWolfe => {
            DantzigWolfeDecomposition::new(&csr, &spectral, &config)
                .solve()
                .expect("DW failed")
        }
        MethodChoice::Lagrangian => {
            LagrangianBundle::new(&csr, &spectral, &config)
                .solve()
                .expect("Lagrangian failed")
        }
    };

    let gap = lp_result.objective - decomp_result.dual_bound;
    println!("Dual bound:    z_D  = {:.6}", decomp_result.dual_bound);
    println!("LP-Dual gap:   {:.6e}", gap);
    println!("Iterations:    {}", decomp_result.iterations);

    // ---------------------------------------------------------------
    // Step 7: Verify and certify
    // ---------------------------------------------------------------
    println!("\n--- Certificate verification ---");

    // Check dual feasibility
    let dual_check = DualChecker::new(&config);
    let feasibility = dual_check
        .check(&csr, &decomp_result)
        .expect("Dual check failed");
    println!(
        "Dual feasibility: {} (max violation: {:.2e})",
        if feasibility.is_feasible { "PASS" } else { "FAIL" },
        feasibility.max_violation
    );

    // Compute theoretical bound via Proposition T2
    let scaling = ScalingLaw::from_eigen_result(&eigen_result, &config);
    let theoretical_gap = scaling.bound();
    println!(
        "Proposition T2 bound: z_LP - z_D ≤ {:.6e}",
        theoretical_gap
    );
    println!(
        "Actual gap within bound: {}",
        if gap <= theoretical_gap * 1.01 { "YES" } else { "NO (numerical)" }
    );

    // ---------------------------------------------------------------
    // Step 8: Generate report
    // ---------------------------------------------------------------
    let report = CertificateReport::builder()
        .matrix_info(n_rows, n_cols, coo.nnz())
        .spectral_features(&spectral)
        .eigenvalues(&eigen_result.eigenvalues)
        .oracle_prediction(&prediction)
        .decomposition_result(&decomp_result)
        .dual_feasibility(&feasibility)
        .scaling_law_bound(theoretical_gap)
        .build();

    let report_path = "examples/output/certificate_report.json";
    report
        .write_json(report_path)
        .expect("Failed to write report");
    println!("\nCertificate report written to {}", report_path);

    println!("\n=== Pipeline complete ===");
}

/// Populate a block-angular constraint matrix with `num_blocks` blocks
/// and weak inter-block coupling.
fn populate_block_angular(
    coo: &mut CooMatrix,
    n_rows: usize,
    n_cols: usize,
    num_blocks: usize,
) {
    let block_rows = n_rows / num_blocks;
    let block_cols = n_cols / num_blocks;

    for b in 0..num_blocks {
        let row_start = b * block_rows;
        let col_start = b * block_cols;

        // Dense-ish block
        for i in 0..block_rows {
            for offset in 0..6.min(block_cols) {
                let j = (i * 3 + offset) % block_cols;
                let val = 1.0 + (i as f64 + j as f64) * 0.01;
                coo.push(row_start + i, col_start + j, val);
            }
        }

        // Weak coupling to next block
        if b + 1 < num_blocks {
            let next_col_start = (b + 1) * block_cols;
            for k in 0..3 {
                coo.push(row_start + k, next_col_start + k, 0.001);
            }
        }
    }
}
