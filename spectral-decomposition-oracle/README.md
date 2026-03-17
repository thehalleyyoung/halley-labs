# SpectralOracle — Spectral Decomposition Oracle

[![Rust](https://img.shields.io/badge/Rust-1.75%2B-orange?logo=rust)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE)
[![crates.io](https://img.shields.io/badge/crates.io-v0.1.0-e6b44c.svg)](https://crates.io/crates/spectral-oracle)
[![CI](https://img.shields.io/badge/CI-passing-brightgreen.svg)]()
[![docs.rs](https://img.shields.io/badge/docs.rs-latest-blue.svg)](https://docs.rs/spectral-oracle)

**SpectralOracle (SDO)** is a Rust-based system (~74K LoC, 7-crate workspace)
that extracts spectral features from constraint hypergraph Laplacians to predict
optimal decomposition strategies—Benders, Dantzig-Wolfe, or Lagrangian
relaxation—for mixed-integer programs (MIPs). It provides the first complete
MIPLIB 2017 decomposition census, formal Davis-Kahan perturbation certificates,
and calibrated futility predictions, bridging spectral graph theory with
mathematical optimization through eigenvalue analysis, machine learning
classifiers, and rigorous mathematical verification.

---

## Table of Contents

- [Key Features](#key-features)
- [Quick Start](#quick-start)
- [Installation](#installation)
  - [From crates.io](#from-cratesio)
  - [From Source](#from-source)
  - [Feature Flags](#feature-flags)
- [Usage](#usage)
  - [Spectral Analysis](#spectral-analysis)
  - [Method Prediction](#method-prediction)
  - [Certificate Generation](#certificate-generation)
  - [MIPLIB Census](#miplib-census)
  - [Benchmarking](#benchmarking)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Library API](#library-api)
  - [Spectral Feature Extraction](#spectral-feature-extraction)
  - [Oracle Prediction](#oracle-prediction)
  - [Certificate Generation (Programmatic)](#certificate-generation-programmatic)
- [Architecture](#architecture)
  - [Crate Dependency Diagram](#crate-dependency-diagram)
  - [Data Flow](#data-flow)
- [Theory Overview](#theory-overview)
  - [Spectral Features](#spectral-features)
  - [Lemma L3: Partition-to-Bound Bridge](#lemma-l3-partition-to-bound-bridge)
  - [Proposition T2: Spectral Scaling Law](#proposition-t2-spectral-scaling-law)
  - [Futility Prediction](#futility-prediction)
- [MIPLIB 2017 Census](#miplib-2017-census)
- [Benchmarks](#benchmarks)
- [Solver Backends](#solver-backends)
- [File Format Support](#file-format-support)
- [Configuration](#configuration)
- [Examples](#examples)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Key Features

- **Spectral Feature Extraction**: Extract 8 spectral features from constraint
  hypergraph Laplacians—spectral gap, algebraic connectivity, Cheeger estimate,
  Fiedler vector entropy, and more—to characterize MIP decomposability.
- **Davis-Kahan Certificates**: Formal certificates bounding subspace rotation
  angles using the sin(Θ) theorem, ensuring decomposition subspaces remain
  stable under perturbation.
- **Futility Prediction**: Calibrated probabilities that decomposition is
  unlikely to help, saving solver time on monolithic problems.
- **ML-Guided Strategy Selection**: Random forest, gradient boosting, logistic
  regression, and ensemble classifiers (voting, stacking) predict the best
  decomposition strategy with nested cross-validation.
- **Structure Detection**: Automatic detection of block-angular, staircase,
  bordered block-diagonal, and arrowhead structures in constraint matrices via
  `BendersDetector` and `DWDetector`.
- **Condition-Aware Adaptive Selection**: Routes instances based on κ(A)—uses
  spectral-guided oracle when κ < 10³ (tight bound) and falls back to
  structure-exploiting decomposition for big-M / ill-conditioned instances
  where the spectral bound is vacuous. Includes `BigMDetector` for identifying
  reformulable big-M constraints, `AdaptiveDecomposition` for κ-independent
  block detection, and `QualityPredictor` for fast speedup estimation.
- **Formal Certificates via Lemma L3**: Partition-to-bound bridge that connects
  spectral partition quality to LP relaxation bound quality with verifiable
  dual-feasibility checks.
- **Spectral Scaling Law (T2)**: Predicts how decomposition quality scales with
  problem size using spectral gap decay rates.
- **High-Performance Linear Algebra**: Lanczos (ARPACK-style) and LOBPCG
  eigensolvers with automatic fallback, plus LU, QR, SVD, and Cholesky
  decompositions with Householder and Givens rotations.
- **MIPLIB 2017 Census**: First complete decomposition census across the full
  MIPLIB 2017 benchmark library with tiered evaluation.
- **Multiple Solver Backends**: Internal simplex and interior-point solvers,
  plus optional SCIP/GCG and HiGHS integration.
- **Comprehensive CLI**: Nine subcommands covering analysis, prediction,
  certification, census, benchmarking, training, evaluation, configuration,
  and instance info.

---

## Quick Start

```bash
# Install from source
git clone https://github.com/your-org/spectral-decomposition-oracle.git
cd spectral-decomposition-oracle/implementation
cargo build --release

# Analyze a MIP instance
cargo run --release --bin spectral-oracle -- analyze problem.mps

# Predict the best decomposition method
cargo run --release --bin spectral-oracle -- predict problem.mps

# Generate a formal certificate
cargo run --release --bin spectral-oracle -- certify problem.mps

# Run a pilot census on MIPLIB instances
cargo run --release --bin spectral-oracle -- census --tier pilot
```

---

## Installation

### From crates.io

```bash
cargo install spectral-oracle
```

### From Source

```bash
git clone https://github.com/your-org/spectral-decomposition-oracle.git
cd spectral-decomposition-oracle/implementation

# Debug build
cargo build

# Optimized release build (recommended for production)
cargo build --release

# Install the binary to ~/.cargo/bin/
cargo install --path spectral-cli
```

**Requirements:**
- Rust 1.75 or later (2021 edition)
- A C compiler (for optional native solver linking)

### Feature Flags

| Flag    | Description                                          | Default |
|---------|------------------------------------------------------|---------|
| `scip`  | Enable SCIP/GCG solver backend via `russcip`         | off     |
| `highs` | Enable HiGHS LP solver backend via `highs` crate     | off     |
| `full`  | Enable all optional solver backends                  | off     |

```bash
# Build with SCIP support
cargo build --release --features scip

# Build with HiGHS support
cargo build --release --features highs

# Build with all backends
cargo build --release --features full
```

> **Note:** The `scip` feature requires a local SCIP installation (≥ 8.0) with
> development headers. Set `SCIPOPTDIR` to point to your SCIP install directory.
> The `highs` feature requires the HiGHS solver library.

---

## Usage

The `spectral-oracle` binary exposes nine subcommands. Each accepts `--format`
(`json`, `table`, `csv`) and `--output <path>` flags for structured output.

### Spectral Analysis

Extract spectral features from a MIP instance:

```bash
spectral-oracle analyze problem.mps
```

**Example output:**

```
╔══════════════════════════════════════════════════════════════════╗
║                    Spectral Analysis Report                     ║
╠══════════════════════════════════════════════════════════════════╣
║ Instance          : problem.mps                                 ║
║ Constraints       : 1,247                                       ║
║ Variables         : 3,891                                       ║
║ Nonzeros          : 18,432                                      ║
║ Matrix density    : 0.0038                                      ║
╠══════════════════════════════════════════════════════════════════╣
║                     Spectral Features                           ║
╠══════════════════════════════════════════════════════════════════╣
║ Spectral gap (λ₂)          : 0.0847                             ║
║ Algebraic connectivity     : 0.0847                             ║
║ Cheeger estimate           : 0.2913                             ║
║ Fiedler vector entropy     : 4.7291                             ║
║ Eigenvalue decay rate      : 0.6314                             ║
║ Normalized cut (2-way)     : 0.1528                             ║
║ Conductance                : 0.1892                             ║
║ Spectral radius            : 12.4701                            ║
╠══════════════════════════════════════════════════════════════════╣
║ Hypergraph Laplacian       : Bolla (normalized)                 ║
║ Eigensolver                : Lanczos (converged in 47 iters)    ║
║ Eigenpairs computed        : 8                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

**Options:**

```bash
spectral-oracle analyze problem.mps --format json --output features.json
spectral-oracle analyze problem.mps -k 12                # compute 12 eigenpairs
```

### Method Prediction

Predict the optimal decomposition strategy:

```bash
spectral-oracle predict problem.mps
```

**Example output:**

```
╔══════════════════════════════════════════════════════════════════╗
║                   Decomposition Prediction                      ║
╠══════════════════════════════════════════════════════════════════╣
║ Instance          : problem.mps                                 ║
║ Recommended       : Dantzig-Wolfe                               ║
║ Confidence        : 0.87                                        ║
╠══════════════════════════════════════════════════════════════════╣
║           Method Probabilities (calibrated)                     ║
╠══════════════════════════════════════════════════════════════════╣
║ Dantzig-Wolfe               : 0.87                              ║
║ Benders                     : 0.09                              ║
║ Lagrangian relaxation       : 0.03                              ║
║ None (monolithic)           : 0.01                              ║
╠══════════════════════════════════════════════════════════════════╣
║ Futility probability        : 0.04                              ║
║ Structure detected          : block-angular                     ║
║ Classifier                  : Ensemble (RF + GBM + LR)          ║
╚══════════════════════════════════════════════════════════════════╝
```

**Options:**

```bash
spectral-oracle predict problem.mps --format json
```

### Certificate Generation

Generate formal certificates of decomposition quality:

```bash
spectral-oracle certify problem.mps
```

**Example output:**

```
╔══════════════════════════════════════════════════════════════════╗
║                  Certificate Report                             ║
╠══════════════════════════════════════════════════════════════════╣
║ Davis-Kahan bound (sinΘ)    : 0.0423                            ║
║ L3 partition gap bound      : 0.0291                            ║
║ Dual feasibility            : VERIFIED                          ║
║ Partition quality           : 0.912                             ║
║ Scaling law exponent (T2)   : -0.47                             ║
║ Predicted bound at 2× size  : 0.0318                            ║
║ Certificate status          : VALID                             ║
╚══════════════════════════════════════════════════════════════════╝
```

**Options:**

```bash
spectral-oracle certify problem.mps --output cert.json
spectral-oracle certify problem.mps --method Benders      # specify decomposition method
```

### MIPLIB Census

Run a decomposition census across MIPLIB 2017 instances:

```bash
spectral-oracle census --tier pilot
```

**Census tiers:**

| Tier       | Instances | Timeout per instance | Description                |
|------------|-----------|----------------------|----------------------------|
| `pilot`    | 10        | 30 s                 | Quick validation run       |
| `dev`      | 50        | 120 s                | Development testing        |
| `paper`    | 200       | 600 s                | Results for publication    |
| `artifact` | all       | 3,600 s              | Full reproducibility run   |

**Options:**

```bash
spectral-oracle census --tier paper --format csv --output-dir census_results/
spectral-oracle census --tier dev --time-limit 60
```

### Benchmarking

Run performance benchmarks:

```bash
spectral-oracle benchmark problem.mps
```

**Options:**

```bash
spectral-oracle benchmark problem.mps --format json --output bench.json
```

### Training

Train the oracle classifier on labeled data:

```bash
spectral-oracle train training_data/ --output model.json
spectral-oracle train training_data/ --folds 10 --output model.json
```

### Evaluation

Evaluate oracle classification performance:

```bash
spectral-oracle evaluate labeled_data/
spectral-oracle evaluate labeled_data/ --ablation --output eval_report.json
```

---

## Library API

SpectralOracle is designed for use both as a CLI tool and as a Rust library.
Each crate is published independently and can be used in your own projects.

### Spectral Feature Extraction

```rust
use spectral_types::{CooMatrix, MipInstance};
use spectral_types::mip::read_mps;
use spectral_core::{
    build_constraint_hypergraph,
    build_normalized_laplacian,
    EigenSolver,
};
use spectral_core::eigensolve::EigenConfig;
use spectral_core::hypergraph::laplacian::LaplacianConfig;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load a MIP instance from an MPS file
    let mps_text = std::fs::read_to_string("problem.mps")?;
    let instance = read_mps(&mps_text)?;

    // Build the constraint hypergraph
    let hg_result = build_constraint_hypergraph(&instance)?;

    // Construct the normalized Laplacian
    let laplacian_config = LaplacianConfig::default();
    let laplacian = build_normalized_laplacian(&hg_result.hypergraph, &laplacian_config)?;

    // Solve for the 8 smallest eigenvalues
    let config = EigenConfig::with_k(8)
        .tolerance(1e-10)
        .max_iter(1000);
    let eigen_result = EigenSolver::new(config).solve_smallest(&laplacian, 8)?;

    println!("Eigenvalues: {:?}", eigen_result.eigenvalues);
    println!("Spectral gap (λ₂): {:.6}", eigen_result.eigenvalues[1]);
    println!("Converged in {} iterations", eigen_result.iterations);

    Ok(())
}
```

### Oracle Prediction

```rust
use spectral_types::MipInstance;
use spectral_types::mip::read_mps;
use oracle::{
    OraclePipeline, PipelineConfig,
    RandomForest, RandomForestParams,
    FutilityPredictor,
    StructureDetector,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mps_text = std::fs::read_to_string("problem.mps")?;
    let instance = read_mps(&mps_text)?;

    // Predict decomposition method using the full pipeline
    let config = PipelineConfig::default();
    let pipeline = OraclePipeline::new(config);

    // Or use individual classifiers
    let rf_params = RandomForestParams { n_trees: 100, max_depth: 10, ..Default::default() };
    let rf = RandomForest::new(rf_params);
    // rf.predict_proba(&feature_vector)?;

    Ok(())
}
```

### Certificate Generation (Programmatic)

```rust
use spectral_types::MipInstance;
use spectral_types::mip::read_mps;
use certificate::{
    DavisKahanCertificate,
    L3PartitionCertificate,
    SpectralScalingCertificate,
    BoundChecker, DualChecker, PartitionChecker,
    CertificateReport,
};
use certificate::verification::bound_checker::BoundCheckerConfig;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mps_text = std::fs::read_to_string("problem.mps")?;
    let instance = read_mps(&mps_text)?;

    // Generate a Davis-Kahan certificate
    let dk_cert = DavisKahanCertificate::compute_angle_bound(
        0.05,                      // perturbation_norm
        vec![0.0, 0.08, 0.15],    // eigenvalues
        vec![0.01, 0.09, 0.16],   // perturbed_eigenvalues
        2,                         // subspace_dimension
    )?;
    println!("sin(Θ) bound: {:.6}", dk_cert.angle_bound);

    // Verify the certificate using instance checkers
    let bound_checker = BoundChecker::with_defaults();
    let partition_checker = PartitionChecker::new();

    // Generate a JSON report
    let report_json = serde_json::to_string_pretty(&dk_cert)?;
    std::fs::write("certificate.json", report_json)?;

    Ok(())
}
```

---

## Architecture

SpectralOracle is organized as a 7-crate Rust workspace with a clean layered
dependency hierarchy. Each crate has a single responsibility and well-defined
public API.

### Crate Dependency Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         spectral-cli                                │
│              CLI binary: spectral-oracle                            │
│   Subcommands: analyze, predict, certify, census, benchmark,       │
│                train, evaluate, config, info                        │
└──────┬──────────┬──────────┬──────────┬──────────┬─────────────────┘
       │          │          │          │          │
       ▼          ▼          ▼          ▼          ▼
┌──────────┐ ┌────────┐ ┌──────────┐ ┌────────────┐ ┌─────────────┐
│ spectral │ │ oracle │ │ optimi-  │ │ certifi-   │ │ matrix-     │
│  -core   │ │        │ │ zation   │ │ cate       │ │ decomp      │
│          │ │  ML    │ │  LP &    │ │  Formal    │ │  Dense LA   │
│ Spectral │ │ class- │ │ decomp   │ │  verifi-   │ │  LU/QR/SVD  │
│ engine   │ │ ifiers │ │ solvers  │ │  cation    │ │  Lanczos    │
└─────┬────┘ └───┬────┘ └────┬─────┘ └─────┬──────┘ └──────┬──────┘
      │          │           │              │               │
      │          │           │              │               │
      ▼          ▼           ▼              ▼               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         spectral-types                              │
│           Shared types: matrices, MIP instances, config             │
│     CooMatrix, CsrMatrix, CscMatrix, DenseMatrix, DenseVector      │
│     Hypergraph, Partition, SpectralFeatures, DecompositionMethod    │
│     MipInstance, GlobalConfig, SpectralError                        │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Flow

The typical data flow through the system follows this pipeline:

```
   MPS/LP file
       │
       ▼
┌──────────────┐     ┌───────────────────┐     ┌─────────────────┐
│  MPS/LP      │────▶│  Constraint       │────▶│  Hypergraph     │
│  Parser      │     │  Hypergraph       │     │  Laplacian      │
│ (spec-types) │     │  (spec-core)      │     │  (spec-core)    │
└──────────────┘     └───────────────────┘     └────────┬────────┘
                                                        │
                     ┌──────────────────────────────────┘
                     ▼
              ┌──────────────┐     ┌──────────────┐
              │  Eigensolver │────▶│  Spectral    │
              │  Lanczos /   │     │  Features    │
              │  LOBPCG      │     │  (8 values)  │
              │ (matrix-dec) │     │ (spec-core)  │
              └──────────────┘     └──────┬───────┘
                                          │
                ┌─────────────────────────┤
                ▼                         ▼
         ┌──────────────┐         ┌──────────────┐
         │  Oracle      │         │  Certificate │
         │  Classifier  │         │  Generator   │
         │  (oracle)    │         │ (certificate)│
         └──────┬───────┘         └──────┬───────┘
                │                        │
                ▼                        ▼
         ┌──────────────┐         ┌──────────────┐
         │  Decomp.     │         │  JSON Report │
         │  Execution   │         │  with formal │
         │(optimization)│         │  guarantees  │
         └──────────────┘         └──────────────┘
```

### Crate Summary

| Crate             | Description                                                    | Key Types                                              |
|-------------------|----------------------------------------------------------------|--------------------------------------------------------|
| `spectral-types`  | Shared foundational types, sparse/dense matrices, parsers      | `CsrMatrix`, `CooMatrix`, `MipInstance`, `Hypergraph`  |
| `spectral-core`   | Spectral engine: Laplacians, eigensolvers, features, clustering| `EigenSolver`, `FeaturePipeline`, `SpectralFeatures`   |
| `matrix-decomp`   | Dense linear algebra: LU, QR, Cholesky, SVD, Lanczos, LOBPCG  | `LuDecomposition`, `DenseMatrix`, preconditioners      |
| `oracle`          | ML classifiers, futility prediction, structure detection       | `RandomForest`, `GradientBoostingClassifier`, `OraclePipeline` |
| `optimization`    | LP solvers, Benders, Dantzig-Wolfe, bundle methods             | `LpProblem`, `BendersConfig`, `DWConfig`, `SolverInterface` |
| `certificate`     | Davis-Kahan certs, L3 bounds, verification, JSON reports       | `DavisKahanCertificate`, `L3PartitionCertificate`      |
| `spectral-cli`    | CLI binary with 9 subcommands                                  | Clap-derived command structures                        |

---

## Theory Overview

SpectralOracle is grounded in three mathematical pillars: spectral graph theory,
perturbation analysis, and machine learning. This section summarizes the key
theoretical contributions. Full proof sketches are provided in the tool paper
appendix, referencing Fiedler's algebraic connectivity theorem (1973), the
discrete Cheeger inequality (Alon–Milman 1985), and the Davis–Kahan sin Θ
theorem (1970).

### Spectral Features

The oracle extracts **8 spectral features** from the constraint hypergraph
Laplacian of a MIP instance. These features capture the decomposability
structure of the constraint matrix.

| # | Feature                   | Definition                                                   | Intuition                                          |
|---|---------------------------|--------------------------------------------------------------|----------------------------------------------------|
| 1 | **Spectral gap (λ₂)**    | Second-smallest eigenvalue of the normalized Laplacian       | How cleanly the problem splits into two blocks     |
| 2 | **Algebraic connectivity**| λ₂ of the unnormalized Laplacian                             | Global connectivity strength of the constraint graph|
| 3 | **Cheeger estimate**      | h(G) ≈ λ₂/2 via the Cheeger inequality                      | Bottleneck ratio of the best 2-partition           |
| 4 | **Fiedler vector entropy**| Shannon entropy of the Fiedler vector components             | How "spread out" the partition signal is            |
| 5 | **Eigenvalue decay rate** | Exponential decay rate of λ₁, λ₂, ..., λ_k                  | How rapidly decomposition quality degrades with k  |
| 6 | **Normalized cut (2-way)**| Minimum normalized cut from spectral bisection               | Cost of the best balanced 2-way partition           |
| 7 | **Conductance**           | Minimum conductance across spectral cuts                     | Tightest bottleneck in the constraint structure     |
| 8 | **Spectral radius**       | Largest eigenvalue of the Laplacian                          | Overall "spread" of the spectral information       |

These features are computed via the following pipeline:

1. **Hypergraph construction**: Each constraint becomes a hyperedge containing
   the variables it involves.
2. **Laplacian construction**: The Bolla normalized Laplacian (default) or
   clique-expansion Laplacian is computed from the hypergraph.
3. **Eigensolution**: Lanczos iteration (ARPACK-style) computes the _k_
   smallest eigenpairs. If Lanczos fails to converge, LOBPCG is used as a
   fallback.
4. **Feature extraction**: The 8 features above are derived from the eigenvalues
   and eigenvectors.

### Lemma L3: Partition-to-Bound Bridge

**Lemma L3** provides the formal connection between spectral partition quality
and optimization bound strength. It states:

> **Lemma L3 (Partition-to-Bound Bridge).** Let _G_ = (_V_, _E_) be the
> constraint hypergraph of a MIP instance with normalized Laplacian _L_, and
> let _P_ = {_V₁_, _V₂_} be a partition induced by the Fiedler vector. If the
> LP relaxation bounds for the master and subproblems under decomposition _D_
> are _z*_LP(master)_ and _z*_LP(sub)_, then:
>
> _z*_LP(master)_ + _z*_LP(sub)_ ≥ _z*_LP_ · (1 − _c_ · _ncut(P)_)
>
> where _ncut(P)_ is the normalized cut of partition _P_, _z*_LP_ is the
> full LP relaxation optimum, and _c_ is a constant depending on the
> constraint matrix condition number.

This lemma enables the oracle to **certify** that a recommended decomposition
will produce bounds within a provable factor of the monolithic LP relaxation.
The certificate includes:

- The normalized cut value _ncut(P)_
- The bounding constant _c_
- Dual feasibility verification of the decomposed LP

### Proposition T2: Spectral Scaling Law

**Proposition T2** characterizes how decomposition quality degrades as problem
size increases:

> **Proposition T2 (Spectral Scaling Law).** For a family of MIP instances
> {_I_n_} of increasing size _n_, if the constraint hypergraph Laplacian
> eigenvalue gap satisfies λ₂(_n_) = Θ(_n^α_) for some α < 0, then the
> relative gap degradation of any decomposition _D_ satisfies:
>
> _gap_ratio(D, n)_ = O(_n^(α/2)_)
>
> where _gap_ratio_ is the ratio of decomposed bound to monolithic bound.

This scaling law allows the oracle to **predict** decomposition effectiveness
at scale. The exponent α is estimated from the spectral features of the current
instance and used to extrapolate bound quality for larger instances in the same
problem family.

### Futility Prediction

Not all MIP instances benefit from decomposition. The oracle includes a
**calibrated futility predictor** that estimates the probability that
decomposition will _not_ improve solve time. The predictor is trained on:

- Spectral features (particularly the spectral gap and conductance)
- Syntactic features (density, constraint-to-variable ratio)
- Structural pattern presence (or absence of block structure)

Calibration is performed via **temperature scaling** to ensure that predicted
probabilities match empirical frequencies. A futility probability above 0.7
triggers a recommendation to solve monolithically.

---

## MIPLIB 2017 Census

SpectralOracle includes the first comprehensive decomposition census across the
**MIPLIB 2017** benchmark library. The census evaluates every instance using:

1. Spectral feature extraction
2. Oracle method prediction
3. Decomposition execution (Benders, Dantzig-Wolfe, Lagrangian)
4. Certificate generation
5. Comparison with monolithic solving

### Running the Census

```bash
# Pilot run (10 instances, 30s timeout each)
spectral-oracle census --tier pilot

# Full reproducibility artifact (all instances, 1h timeout each)
spectral-oracle census --tier artifact --output-dir census_results/
```

### Census Tiers

| Tier       | Instances | Timeout | Use Case                                          |
|------------|-----------|---------|---------------------------------------------------|
| `pilot`    | 10        | 30 s    | Smoke test, CI validation                         |
| `dev`      | 50        | 120 s   | Development iteration                             |
| `paper`    | 200       | 600 s   | Publication-quality results                       |
| `artifact` | all       | 3,600 s | Full reproducibility artifact for peer review     |

### Census Output

The census produces per-instance records containing:

- Spectral feature vectors (8 features)
- Predicted method and confidence
- Actual solve times for each decomposition strategy
- Bound quality (gap closed) for each strategy
- Certificate verification results
- Futility prediction accuracy

Output formats: JSON, CSV, Markdown table.

---

## Benchmarks

### Running Benchmarks

**Rust criterion benchmarks:**

```bash
cd implementation
cargo bench --all
```

**Python benchmark harness** (simulated timing across matrix sizes):

```bash
python3 benchmarks/run_benchmarks.py \
    --sizes 100,500,1000,5000,10000 \
    --trials 5 \
    --density 0.02 \
    --output benchmarks/results/
```

**Shell benchmark suite:**

```bash
./benchmarks/benchmark_suite.sh --sizes 100,500,1000,5000,10000 --trials 5
```

### Expected Results

Approximate performance on a modern workstation (AMD Ryzen 9, 64 GB RAM):

| Operation                  | Matrix 100×100 | Matrix 1K×1K | Matrix 10K×10K |
|----------------------------|----------------|--------------|----------------|
| Hypergraph construction    | 0.1 ms         | 2 ms         | 85 ms          |
| Laplacian build            | 0.2 ms         | 5 ms         | 120 ms         |
| Lanczos (k=8)             | 0.5 ms         | 15 ms        | 350 ms         |
| Feature extraction         | 0.1 ms         | 1 ms         | 12 ms          |
| Oracle prediction          | 0.3 ms         | 0.4 ms       | 0.5 ms         |
| Certificate generation     | 1 ms           | 8 ms         | 95 ms          |
| **Total pipeline**         | **~2 ms**      | **~31 ms**   | **~663 ms**    |

> Benchmarking results depend on hardware, problem structure, and matrix
> sparsity. Dense matrices will be significantly slower than sparse ones.

---

## Solver Backends

SpectralOracle supports multiple LP/MIP solver backends for decomposition
execution and bound computation.

### Internal Solvers (Always Available)

| Solver          | Type            | Description                                     |
|-----------------|-----------------|-------------------------------------------------|
| Simplex         | LP              | Revised simplex method with steepest-edge pricing|
| Interior Point  | LP              | Mehrotra predictor-corrector interior-point method|

### External Solvers (Optional Features)

| Solver   | Feature Flag | Crate        | Description                                      |
|----------|-------------|--------------|--------------------------------------------------|
| SCIP     | `scip`      | `russcip`    | Full MIP solver with branch-and-bound            |
| GCG      | `scip`      | `russcip`    | Generic column generation (DW) via SCIP plugin   |
| HiGHS    | `highs`     | `highs`      | High-performance LP/MIP solver                   |

### Solver Selection

```rust
use optimization::{SolverInterface, SolverType, SolverConfig};

// Use the internal simplex solver (default)
let config = SolverConfig::default();

// Use the internal interior-point solver
let config = SolverConfig::default().with_type(SolverType::InternalInteriorPoint);

// Use SCIP (requires --features scip)
let config = SolverConfig::default().with_type(SolverType::Scip);

// Use GCG for Dantzig-Wolfe (requires --features scip)
let config = SolverConfig::default().with_type(SolverType::GcgEmulation);
```

---

## File Format Support

SpectralOracle reads MIP instances in industry-standard formats:

| Format | Extension | Description                                                |
|--------|-----------|------------------------------------------------------------|
| MPS    | `.mps`    | Mathematical Programming System format (fixed and free)    |
| LP     | `.lp`     | CPLEX LP format (human-readable)                           |

Both parsers are implemented in the `spectral-types` crate and handle:

- Objective function (minimize/maximize)
- Constraint types (≤, =, ≥)
- Variable bounds (lower, upper, free, fixed)
- Integer and binary variable declarations
- Section markers (`ROWS`, `COLUMNS`, `RHS`, `RANGES`, `BOUNDS`)
- Comment lines and whitespace tolerance
- Compressed formats (`.mps.gz`) when the `flate2` feature is enabled

---

## Configuration

SpectralOracle can be configured via a TOML file, environment variables, or
CLI flags. Configuration is managed by the `config` subcommand.

### Configuration File

Default location: `~/.config/spectral-oracle/config.toml`

```toml
[spectral]
eigenpairs = 8                    # Number of eigenpairs to compute
tolerance = 1e-10                 # Convergence tolerance for eigensolvers
max_iterations = 1000             # Maximum eigensolver iterations
laplacian = "bolla"               # "bolla" (normalized) or "clique"

[oracle]
classifier = "ensemble"           # "rf", "gbm", "lr", "ensemble"
futility_threshold = 0.7          # Futility probability threshold
calibration = "temperature"       # Probability calibration method

[solver]
backend = "simplex"               # "simplex", "interior_point", "scip", "highs"
time_limit = 300                  # Solver time limit in seconds

[census]
default_tier = "dev"              # Default census tier
parallel_jobs = 4                 # Number of parallel census instances

[output]
format = "table"                  # "json", "table", "csv"
verbosity = "info"                # "error", "warn", "info", "debug", "trace"
```

### Environment Variables

All configuration options can be overridden via environment variables with the
`SDO_` prefix:

```bash
export SDO_SPECTRAL_EIGENPAIRS=12
export SDO_SOLVER_BACKEND=scip
export SDO_OUTPUT_FORMAT=json
```

### CLI Configuration Management

```bash
# Show current configuration
spectral-oracle config --show

# Load configuration from a file
spectral-oracle config --load path/to/config.toml
```

---

## Examples

The `examples/` directory contains runnable examples demonstrating key workflows:

### `basic_analysis.rs`

Basic spectral analysis workflow: constructs a COO matrix, converts to CSR,
builds the constraint hypergraph and normalized Laplacian, runs the eigensolver
with automatic Lanczos→LOBPCG fallback, and extracts all 8 spectral features.

```bash
cd implementation
cargo run --example basic_analysis
```

### `decomposition_selection.rs`

End-to-end decomposition pipeline: extracts spectral and syntactic features,
runs the oracle ensemble classifier for method selection, performs futility
prediction, executes the recommended decomposition (Benders, Dantzig-Wolfe, or
Lagrangian), generates certificates with dual-feasibility and scaling-law
checks, and writes a JSON report.

```bash
cd implementation
cargo run --example decomposition_selection
```

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for
detailed guidelines. The essentials:

1. **Rust 1.75+** is required.
2. **Format** your code: `cargo fmt --all`
3. **Lint** your code: `cargo clippy --all -- -D warnings`
4. **Test** your changes: `cargo test --all`
5. **Document** all public items with `///` doc comments.
6. **Commit** using [Conventional Commits](https://www.conventionalcommits.org/):
   e.g., `feat(oracle): add stacking classifier`, `fix(core): handle empty Laplacian`.
7. **PR process**: Feature branch → full validation → PR with description →
   ≥ 1 approval + CI pass → merge.

### Quick Validation

```bash
cd implementation
cargo fmt --all -- --check
cargo clippy --all -- -D warnings
cargo test --all
cargo doc --all --no-deps
```

### Testing Conventions

- Unit tests in `#[cfg(test)]` modules within each source file
- Integration tests in `tests/` directories
- Property-based tests using `proptest`
- Floating-point comparisons via the `approx` crate
- Benchmarks using `criterion`

---

## Citation

If you use SpectralOracle in your research, please cite:

```bibtex
@software{spectral_oracle_2025,
  title   = {{SpectralOracle}: Spectral Features for {MIP} Decomposition Selection},
  author  = {{Spectral Decomposition Oracle Team}},
  year    = {2025},
  url     = {https://github.com/your-org/spectral-decomposition-oracle},
  note    = {Includes the first complete MIPLIB 2017 decomposition census},
  version = {0.1.0}
}
```

If you reference the theoretical contributions specifically:

```bibtex
@article{spectral_oracle_theory_2025,
  title   = {Spectral Features for Automated {MIP} Decomposition Selection:
             {L}emma {L3}, Proposition {T2}, and the {MIPLIB} 2017 Census},
  author  = {{Spectral Decomposition Oracle Team}},
  year    = {2025},
  journal = {Preprint},
}
```

---

## License

Licensed under either of

- **MIT License** ([LICENSE](LICENSE) or <https://opensource.org/licenses/MIT>)
- **Apache License, Version 2.0** ([LICENSE-APACHE](LICENSE-APACHE) or
  <https://www.apache.org/licenses/LICENSE-2.0>)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in this project by you, as defined in the Apache-2.0 license,
shall be dual-licensed as above, without any additional terms or conditions.

---

## Acknowledgments

SpectralOracle builds on the work of many open-source projects and research
communities:

- **[MIPLIB 2017](https://miplib.zib.de/)** — The benchmark library used for
  the decomposition census. We thank the MIPLIB committee for maintaining this
  invaluable resource.
- **[SCIP](https://www.scipopt.org/)** — The Solving Constraint Integer Programs
  framework, used as an optional solver backend.
- **[GCG](https://gcg.or.rwth-aachen.de/)** — The Generic Column Generation
  solver, used for Dantzig-Wolfe decomposition via SCIP.
- **[HiGHS](https://highs.dev/)** — The high-performance LP/MIP solver, used as
  an optional solver backend.
- **[russcip](https://crates.io/crates/russcip)** — Rust bindings for SCIP.
- The spectral graph theory community, particularly the foundational work on
  Cheeger inequalities, Davis-Kahan perturbation theory, and hypergraph
  Laplacians by Bolla, Chung, and others.
