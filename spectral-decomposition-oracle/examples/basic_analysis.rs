//! Example: Basic spectral analysis of a constraint matrix
//!
//! Demonstrates:
//! - Creating a sparse matrix in COO format
//! - Converting to CSR for efficient operations
//! - Building a constraint hypergraph
//! - Computing the normalized Laplacian
//! - Extracting 8 spectral features
//!
//! Run: cargo run --example basic_analysis

use spectral_types::sparse::{CooMatrix, CsrMatrix};
use spectral_core::hypergraph::ConstraintHypergraph;
use spectral_core::laplacian::NormalizedLaplacian;
use spectral_core::features::SpectralFeatures;
use spectral_core::eigensolve::{EigensolveConfig, EigensolveFallback};

fn main() {
    println!("=== Spectral Decomposition Oracle — Basic Analysis ===\n");

    // ---------------------------------------------------------------
    // Step 1: Build a sparse constraint matrix in COO format
    // ---------------------------------------------------------------
    // This represents a small LP constraint matrix A where A x <= b.
    // Rows = constraints, Columns = variables.
    let n_constraints = 50;
    let n_variables = 100;

    let mut coo = CooMatrix::new(n_constraints, n_variables);

    // Populate with a block-diagonal-ish structure (two loosely coupled blocks)
    // Block 1: constraints 0..25 couple variables 0..50
    for i in 0..25 {
        for offset in 0..4 {
            let j = (i * 2 + offset) % 50;
            coo.push(i, j, 1.0 + (i as f64) * 0.1);
        }
    }
    // Block 2: constraints 25..50 couple variables 50..100
    for i in 25..50 {
        for offset in 0..4 {
            let j = 50 + ((i - 25) * 2 + offset) % 50;
            coo.push(i, j, 1.0 + (i as f64) * 0.1);
        }
    }
    // Weak coupling between blocks
    coo.push(12, 75, 0.01);
    coo.push(37, 25, 0.01);

    println!(
        "Constraint matrix: {}×{}, nnz={}",
        coo.rows(),
        coo.cols(),
        coo.nnz()
    );

    // ---------------------------------------------------------------
    // Step 2: Convert to CSR for efficient row-based operations
    // ---------------------------------------------------------------
    let csr: CsrMatrix = coo.into();
    println!("Converted to CSR format");

    // ---------------------------------------------------------------
    // Step 3: Build the constraint hypergraph
    // ---------------------------------------------------------------
    // Each constraint (row) becomes a hyperedge connecting its variables.
    let hypergraph = ConstraintHypergraph::from_constraint_matrix(&csr);
    println!(
        "Hypergraph: {} vertices, {} hyperedges, max cardinality={}",
        hypergraph.num_vertices(),
        hypergraph.num_hyperedges(),
        hypergraph.max_cardinality()
    );

    // ---------------------------------------------------------------
    // Step 4: Compute the normalized Laplacian
    // ---------------------------------------------------------------
    // Method selection: clique expansion (d_max <= 200) vs Bolla incidence
    let laplacian = if hypergraph.max_cardinality() <= 200 {
        println!("Using clique expansion (d_max <= 200)");
        NormalizedLaplacian::from_clique_expansion(&hypergraph)
    } else {
        println!("Using Bolla incidence Laplacian (d_max > 200)");
        NormalizedLaplacian::from_incidence(&hypergraph)
    };
    println!(
        "Laplacian: {}×{}, nnz={}",
        laplacian.dim(),
        laplacian.dim(),
        laplacian.nnz()
    );

    // ---------------------------------------------------------------
    // Step 5: Compute eigenvalues via ARPACK + LOBPCG fallback
    // ---------------------------------------------------------------
    let eigen_config = EigensolveConfig {
        num_eigenvalues: 8,
        tolerance: 1e-10,
        max_iterations: 1000,
        ..Default::default()
    };

    let eigensolver = EigensolveFallback::new(eigen_config);
    let eigen_result = eigensolver
        .solve(&laplacian)
        .expect("Eigensolve failed");

    println!("\nEigenvalues (smallest 8):");
    for (i, val) in eigen_result.eigenvalues.iter().enumerate() {
        println!("  λ_{} = {:.6e}", i + 1, val);
    }

    // ---------------------------------------------------------------
    // Step 6: Extract spectral features
    // ---------------------------------------------------------------
    let features = SpectralFeatures::extract(&eigen_result);

    println!("\nSpectral Features:");
    println!("  Spectral gap (γ₂):         {:.6e}", features.spectral_gap);
    println!("  Algebraic connectivity:     {:.6e}", features.algebraic_connectivity);
    println!("  Fiedler vector entropy:     {:.4f}", features.fiedler_entropy);
    println!("  Spectral radius:            {:.6e}", features.spectral_radius);
    println!("  Spectral width:             {:.6e}", features.spectral_width);
    println!("  Normalized cut estimate:    {:.6e}", features.normalized_cut_estimate);
    println!("  Cheeger constant estimate:  {:.6e}", features.cheeger_estimate);
    println!("  Eigenvalue decay rate:      {:.6e}", features.eigenvalue_decay_rate);

    // ---------------------------------------------------------------
    // Interpretation
    // ---------------------------------------------------------------
    println!("\n--- Interpretation ---");
    if features.spectral_gap < 0.01 {
        println!("Small spectral gap → constraint matrix has near-decomposable structure.");
        println!("Decomposition methods (Benders/DW/Lagrangian) are likely to be effective.");
    } else if features.spectral_gap < 0.1 {
        println!("Moderate spectral gap → partial decomposability.");
        println!("Decomposition may help but with limited gap closure.");
    } else {
        println!("Large spectral gap → tightly coupled constraints.");
        println!("Decomposition is unlikely to be beneficial (consider direct solve).");
    }

    println!("\n=== Done ===");
}
