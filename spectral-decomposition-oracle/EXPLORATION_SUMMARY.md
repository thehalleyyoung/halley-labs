# Spectral Decomposition Oracle - Complete Exploration Summary

**Date:** March 8, 2025
**Project Status:** Theory complete; implementation phase initialized with empty crates
**Primary Finding:** All 7 Rust crates exist with Cargo.toml configurations but NO source code (.rs files) exists

---

## 1. PROJECT OVERVIEW

### What Is It?
A lightweight preprocessing layer that extracts **spectral features** from the **constraint hypergraph Laplacian** of mixed-integer programs (MIPs) to predict which decomposition method (Benders, Dantzig-Wolfe, Lagrangian relaxation, or none) will yield the strongest dual bounds.

### Why?
Decomposition is a powerful technique to solve large MIPs, but selecting the RIGHT decomposition method is a reformulation selection problem — harder than algorithm selection because it changes the mathematical structure the solver sees. This project provides the first **cross-method decomposition oracle**.

### Scale
- **Target benchmark:** MIPLIB 2017 (1,065 instances)
- **Target venue:** INFORMS Journal on Computing
- **Estimated total code:** 25,000-30,000 lines (mostly Python; Rust interface layer TBD)

---

## 2. PROJECT STRUCTURE

```
/Users/halleyyoung/Documents/div/mathdivergence/
  pipeline_staging/
    spectral-decomposition-oracle/
      ├── implementation/                    # RUST WORKSPACE (7 empty crates)
      │   ├── spectral-types/               # ← PRIORITY 1: Shared types (EMPTY)
      │   ├── spectral-core/                # ← PRIORITY 2: Core algorithms
      │   ├── matrix-decomp/                # ← PRIORITY 3: Decomposition algorithms
      │   ├── oracle/                       # Classification & prediction
      │   ├── optimization/                 # Optimization utilities
      │   ├── certificate/                  # Proof certificates
      │   └── spectral-cli/                 # CLI interface
      ├── theory/                           # Formal mathematical specifications
      │   ├── algorithms.md                 # Complete algorithm specifications (1151 lines)
      │   ├── approach.json                 # Mathematical approach & foundations
      │   ├── verification_framework.md     # Theory quality gates
      │   ├── evaluation_strategy.md        # Evaluation metrics
      │   ├── verification_report.md        # Theory verification
      │   └── red_team_report.md           # Critical analysis
      ├── ideation/                         # Design documents
      │   ├── final_approach.md             # Final approved approach (binding)
      │   ├── approaches.md
      │   ├── theory_eval_*.md
      │   └── ...
      ├── proposals/                        # Archived design proposals
      ├── problem_statement.md              # Original problem specification
      ├── depth_check.md                    # Feasibility analysis
      ├── State.json                        # Project phase tracking
      ├── IMPLEMENTATION_SPECIFICATION.md   # ← GENERATED: Complete type definitions
      └── EXPLORATION_SUMMARY.md            # ← THIS FILE
```

---

## 3. CURRENT IMPLEMENTATION STATUS

| Crate | Status | Purpose |
|-------|--------|---------|
| `spectral-types` | ✅ PLANNED ← YOU START HERE | Shared types: CsrMatrix, CscMatrix, DenseMatrix, DenseVector, error types, decomposition result types |
| `spectral-core` | 🔵 BLOCKED by spectral-types | Laplacian construction, eigensolve pipeline, feature extraction, partition recovery |
| `matrix-decomp` | 🔵 BLOCKED by spectral-types | LU, QR, SVD, eigendecomposition algorithms |
| `oracle` | 🔵 BLOCKED by all above | Decomposition classifiers, futility predictor, L3 bound computation |
| `optimization` | 🔵 BLOCKED by spectral-types | Optimization utilities |
| `certificate` | 🔵 BLOCKED by spectral-types | Proof certificate generation |
| `spectral-cli` | 🔵 BLOCKED by all above | Command-line interface |

**Blocker:** All crates depend on `spectral-types`, so that MUST be the first implementation.

---

## 4. WHAT YOU NEED TO BUILD

### IMMEDIATE PRIORITY: `spectral-types/src/lib.rs`

This single file must define ALL the following (detailed specifications in `IMPLEMENTATION_SPECIFICATION.md`):

#### **Matrix Types**
- **CsrMatrix** - Compressed Sparse Row format (values, col_indices, row_pointers)
- **CscMatrix** - Compressed Sparse Column format
- **CooMatrix** - Coordinate format (for construction before CSR/CSC conversion)
- **DenseMatrix** - Full dense matrix (row-major storage)

#### **Vector Types**
- **DenseVector** - Dense vector (used for eigenvectors, feature vectors)

#### **Trait Abstractions**
- **MatrixLike** - Trait for matrix operations (matvec, adjoint, norm, etc.)
- **VectorLike** - Trait for vector operations (norm, dot, scale, axpy, etc.)

#### **Error Types**
- **SpectralError** enum with variants:
  - LaplacianConstructionError
  - EigensolveFailed
  - EigensolveLackOfConvergence
  - ArpackError, LobpcgError
  - DimensionMismatch
  - SingularMatrix, NotSquareMatrix
  - PartitionValidationFailed
  - NumericalInstability
  - Serialization/IO errors
- **Result<T>** type alias using thiserror

#### **Decomposition Result Types**
- **Partition** - k-block partition of variables (blocks: Vec<Vec<usize>>)
- **EigendecompositionResult** - Eigenvalues, eigenvectors, convergence status, residuals
- **SpectralFeatureVector** - All 8 spectral features (SF1-SF8)
- **CrossingWeightResult** - Crossing weight for L3 bound
- **LaplacianMetadata** - Info about which Laplacian variant was used

#### **Preprocessing Types**
- **EquilibrationMethod** enum (Ruiz, Geometric, ScipNative)
- **EquilibrationResult** - Scaled matrix + scaling factors
- **LaplacianMetadata** - Construction metadata

#### **Constants**
- Numerical tolerances (EPSILON_SMALL=1e-12, EPSILON_GAP=1e-12, etc.)
- Algorithm parameters (ARPACK_TOLERANCE=1e-8, D_MAX_CLIQUE_THRESHOLD=200, etc.)
- Validation thresholds

---

## 5. DEPENDENCIES ALREADY CONFIGURED

All workspace dependencies are pre-configured in `Cargo.toml`:

```toml
serde + serde_json           # Serialization
thiserror + anyhow           # Error handling (← USE thiserror for SpectralError)
num-traits + num-complex     # Numerical traits
ordered-float v4.0           # For sorting floats
log v0.4                     # Logging
rand + rand_chacha           # Randomness
rayon v1.8                   # Parallelism (for matrix ops)
crossbeam v0.8               # Concurrency primitives
indexmap v2.0                # Ordered maps
smallvec v1.11               # Small-vector optimization
bitflags v2.4                # Bit flag definitions
uuid + chrono                # IDs and timestamps
```

You do NOT need to add any new dependencies for spectral-types.

---

## 6. ALGORITHM SPECIFICATIONS (FROM theory/algorithms.md)

### Algorithm 1: Hypergraph Laplacian Construction
- **Input:** Raw sparse constraint matrix A ∈ ℝ^{m×n}
- **Output:** Sparse symmetric positive semidefinite Laplacian L_H ∈ ℝ^{n×n}
- **Steps:**
  1. Equilibrate A (Ruiz, geometric-mean, or SCIP-native scaling)
  2. Compute hyperedge weights: w_i = ‖A_{i,:}‖₂² / d_i
  3. If d_max ≤ 200: Clique expansion Laplacian (exact)
  4. If d_max > 200: Incidence-matrix Laplacian (Bolla 1993 to avoid quadratic blowup)
- **Complexity:** O(nnz · d_max) typical case

### Algorithm 2: Robust Eigensolve
- **Input:** Laplacian L_H, desired number of blocks k
- **Output:** Bottom k+1 eigenvalues and eigenvectors
- **Primary method:** ARPACK shift-invert with σ = -1e-6
- **Fallback chain:**
  1. LOBPCG with Jacobi preconditioner
  2. Explicit shift σ = 1e-4 then re-solve
  3. Return NaN if all fail (for downstream handling)
- **Complexity:** O(T_Lanczos · (nnz + n)) where T_Lanczos ≈ 50-300

### Algorithm 3: Spectral Partition Recovery
- **Input:** Bottom k eigenvectors V_k ∈ ℝ^{n×k}
- **Output:** k-block partition P = {B_1, ..., B_k}
- **Method:** k-means clustering on row-normalized eigenvectors
- **Seeding:** 10 random restarts, Lloyd's algorithm, max 300 iterations

### Algorithm 4: Crossing Weight Computation (L3 Bound)
- **Input:** Partition P, constraint matrix A, LP dual y*
- **Output:** Upper bound on z_LP - z_D = ∑_{e ∈ E_cross} |y*_e| · (n_e - 1)
- **Purpose:** Quality metric for any partition (not just spectral ones)

---

## 7. THE 8 SPECTRAL FEATURES (MUST BE COMPUTABLE)

These are defined in `SpectralFeatureVector` and extracted from eigendecomposition:

| # | Name | Definition | Purpose |
|---|------|-----------|---------|
| 1 | Spectral gap γ₂ | λ₂ of normalized Laplacian | Algebraic connectivity; large gap → well-separated blocks |
| 2 | Spectral gap ratio δ²/γ² | Coupling energy / gap² | T2's main predictor (robust to scaling) |
| 3 | Eigenvalue decay rate β | Slope of log(λ_i) fit | Fast decay → few blocks; slow decay → distributed coupling |
| 4 | Fiedler entropy H | -∑ p_j log(p_j) where p = v²/‖v‖² | Localized eigenvector → clear block structure |
| 5 | Algebraic connectivity ratio | γ₂ / γ_k | Near 1 → clean k-way partition; near 0 → hierarchical |
| 6 | Coupling energy δ² | ‖L_H - L_block‖_F² | Direct measure of inter-block coupling |
| 7 | Separability index | Silhouette of k-means on V_k | Quality of spectral partition |
| 8 | Effective spectral dimension | Count of λ_i < λ_median/10 | Number of exploitable block components |

---

## 8. KEY DESIGN DECISIONS

1. **All numerical types are f64 (double precision)** - Standard for scientific computing
2. **Row-major storage for dense matrices** - Standard for linear algebra
3. **CSR primary sparse format; CSC for transposes** - Efficient matvec and cache locality
4. **COO format used only during construction** - Converted to CSR/CSC before use
5. **Error handling via thiserror** - Ergonomic error definitions with `#[error(...)]`
6. **Traits for algorithm abstraction** - Eigensolvers work with any MatrixLike
7. **Serialization via serde** - For persistence, logging, and data pipelines
8. **Immutable-by-default API** - Mutating operations marked with `&mut`

---

## 9. DOCUMENTATION & REFERENCE FILES

All generated during this exploration:

| File | Location | Purpose |
|------|----------|---------|
| **IMPLEMENTATION_SPECIFICATION.md** | `spectral-decomposition-oracle/` | Complete type definitions, trait specifications, and Rust code templates for spectral-types |
| **EXPLORATION_SUMMARY.md** | `spectral-decomposition-oracle/` | This file |
| **theory/algorithms.md** | `spectral-decomposition-oracle/theory/` | Complete algorithm specifications (1151 lines) |
| **theory/approach.json** | `spectral-decomposition-oracle/theory/` | Mathematical approach and theoretical foundations |
| **ideation/final_approach.md** | `spectral-decomposition-oracle/ideation/` | Approved final approach (binding specification) |

---

## 10. NEXT STEPS

### Step 1: Implement spectral-types/src/lib.rs
Use `IMPLEMENTATION_SPECIFICATION.md` as the template. Key modules to create:
- error.rs (SpectralError, Result)
- constants.rs (numerical tolerances, algorithm parameters)
- matrix.rs (CsrMatrix, CscMatrix, CooMatrix, DenseMatrix)
- vector.rs (DenseVector)
- traits.rs (MatrixLike, VectorLike)
- decomp.rs (Partition, EigendecompositionResult, SpectralFeatureVector, etc.)
- preprocess.rs (EquilibrationMethod, EquilibrationResult, etc.)

### Step 2: Implement spectral-core
- Algorithm 1: Laplacian construction (equilibration + clique/incidence variants)
- Algorithm 2: Eigensolve pipeline (ARPACK + LOBPCG fallback)
- Algorithm 2.2: Feature extraction (compute all 8 spectral features)
- Algorithm 3: Partition recovery (k-means on eigenvectors)

### Step 3: Implement matrix-decomp
- LU, QR, SVD, eigendecomposition wrappers

### Step 4: Implement oracle
- Decomposition method classifiers
- Futility predictor
- L3 bound computation (crossing weight)

---

## 11. VALIDATION CHECKPOINTS

From `theory/verification_framework.md`:

- **D1-D5 (Definitions):** All types must be precisely defined with no ambiguity
- **P1-P5 (Proof rigor):** All algorithms must have clear complexity analysis
- **A1-A5 (Algorithm quality):** Pseudocode → Rust must be faithful to specification
- **G1, G3 (Empirical gates):** Feature ablation must show ≥5pp improvement from spectral features

---

## 12. KEY INVARIANTS

1. All matrices are double precision (f64)
2. Laplacian matrices are always symmetric and positive semidefinite
3. Smallest eigenvalue λ₁ ≈ 0 (validated with threshold 1e-4)
4. Eigenvectors returned in ascending eigenvalue order
5. Partitions are disjoint and exhaustive (cover all variables)
6. All matrix-vector operations validated for dimension compatibility
7. Eigensolve failure returns NaN (handled downstream by classifiers)

---

## CONCLUSION

**What exists:**
- ✅ Complete theory (algorithms.md, approach.json, verification_framework.md)
- ✅ 7 configured Rust crates with Cargo.toml and workspace setup
- ✅ Complete specification of all required types and traits

**What is missing (YOUR WORK):**
- ❌ All source code (.rs files)
- ❌ Matrix types (CsrMatrix, CscMatrix, CooMatrix, DenseMatrix, DenseVector)
- ❌ Error handling (SpectralError)
- ❌ Algorithms (Laplacian construction, eigensolve, feature extraction, partition recovery)

**Start here:** `IMPLEMENTATION_SPECIFICATION.md` provides ready-to-code specifications for every type and trait needed.

