# Spectral Decomposition Oracle - Implementation Specification

## Project Status & Context

**Location:** `/Users/halleyyoung/Documents/div/mathdivergence/pipeline_staging/spectral-decomposition-oracle`

**Current Phase:** Implementation - Theory complete, code generation beginning

**Status:** All 7 crates exist with Cargo.toml files configured, but ALL source code (*.rs) is missing. 

**Primary Task:** Implement `spectral-types/src/lib.rs` with all shared type definitions, which all other crates depend on.

---

## 1. DIRECTORY STRUCTURE

```
implementation/
├── Cargo.toml                         # Workspace with 7 members
├── spectral-types/                   # PRIORITY 1: SHARED TYPES (EMPTY)
│   ├── Cargo.toml
│   └── src/                          # ← CREATE lib.rs HERE
├── spectral-core/                    # PRIORITY 2: CORE ALGORITHMS
│   ├── Cargo.toml
│   └── src/                          # ← lib.rs: Laplacian, eigensolve, features
├── matrix-decomp/                    # PRIORITY 3: DECOMPOSITION ALGORITHMS
│   ├── Cargo.toml
│   └── src/                          # ← lib.rs: LU, QR, SVD, eigendecomp
├── oracle/                           # Classification & prediction layer
│   ├── Cargo.toml
│   └── src/
├── optimization/                     # Optimization utilities
│   ├── Cargo.toml
│   └── src/
├── certificate/                      # Proof certificates
│   ├── Cargo.toml
│   └── src/
└── spectral-cli/                     # Command-line interface
    ├── Cargo.toml
    └── src/
```

---

## 2. SHARED DEPENDENCIES (From Cargo.toml)

All crates share these dependencies via workspace:

```toml
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
thiserror = "1.0"
anyhow = "1.0"
num-traits = "0.2"
num-complex = "0.4"
ordered-float = { version = "4.0", features = ["serde"] }
log = "0.4"
rand = "0.8"
rand_chacha = "0.3"
rayon = "1.8"
crossbeam = "0.8"
indexmap = { version = "2.0", features = ["serde"] }
parking_lot = "0.12"
smallvec = { version = "1.11", features = ["serde"] }
bitflags = "2.4"
uuid = { version = "1.0", features = ["v4", "serde"] }
chrono = { version = "0.4", features = ["serde"] }
env_logger = "0.10"
clap = { version = "4.0", features = ["derive"] }
```

**Additional for specific crates:**
- `matrix-decomp`: also includes `rand_chacha` (already in workspace)
- `spectral-core`: no additional deps beyond workspace

---

## 3. SPECTRAL-TYPES CRATE: COMPLETE SPECIFICATION

### 3.1 Module Structure (`src/lib.rs`)

```rust
// spectral-types/src/lib.rs

pub mod error;
pub mod constants;
pub mod matrix;
pub mod vector;
pub mod traits;
pub mod decomp;
pub mod preprocess;

// Public re-exports
pub use error::{SpectralError, Result};
pub use constants::*;
pub use matrix::{CsrMatrix, CscMatrix, CooMatrix, DenseMatrix};
pub use vector::DenseVector;
pub use traits::{MatrixLike, VectorLike};
pub use decomp::*;
pub use preprocess::*;

// Type aliases
pub type Scalar = f64;
```

### 3.2 Error Types (`error.rs`)

```rust
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SpectralError {
    #[error("Laplacian construction failed: {0}")]
    LaplacianConstructionError(String),
    
    #[error("Eigensolve failed: {0}")]
    EigensolveFailed(String),
    
    #[error("Eigensolve did not converge: achieved residual {residual}, target {target}")]
    EigensolveLackOfConvergence { residual: f64, target: f64 },
    
    #[error("ARPACK error: {0}")]
    ArpackError(String),
    
    #[error("LOBPCG error: {0}")]
    LobpcgError(String),
    
    #[error("Invalid matrix dimensions: expected {expected}, got {actual}")]
    DimensionMismatch { expected: String, actual: String },
    
    #[error("Singular matrix: cannot invert")]
    SingularMatrix,
    
    #[error("Matrix is not square: {nrows}x{ncols}")]
    NotSquareMatrix { nrows: usize, ncols: usize },
    
    #[error("Sparse matrix format error: {0}")]
    SparseFormatError(String),
    
    #[error("Partition validation failed: {0}")]
    PartitionValidationFailed(String),
    
    #[error("Zero spectral gap")]
    ZeroSpectralGap,
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
    
    #[error("Numerical instability: {0}")]
    NumericalInstability(String),
    
    #[error("Configuration error: {0}")]
    ConfigError(String),
}

pub type Result<T> = std::result::Result<T, SpectralError>;
```

### 3.3 Constants (`constants.rs`)

```rust
// Numerical tolerances
pub const EPSILON_SMALL: f64 = 1e-12;       // Regularization for zero-degree vertices
pub const EPSILON_GAP: f64 = 1e-12;         // Guard for spectral gap validity
pub const EPSILON_FLOOR: f64 = 1e-15;       // Floor for eigenvalues (log(ε_floor))
pub const EPSILON_ENTROPY: f64 = 1e-15;     // Floor for entropy computation

// Eigensolve parameters
pub const ARPACK_TOLERANCE: f64 = 1e-8;
pub const ARPACK_MAX_ITERATIONS: usize = 3000;
pub const ARPACK_SHIFT: f64 = -1e-6;

// Laplacian construction
pub const D_MAX_CLIQUE_THRESHOLD: usize = 200;  // Use clique expansion if ≤ 200
pub const D_MAX_EXPLICIT_THRESHOLD: usize = 50000; // Use explicit matrix if n < 50k

// Presolve equilibration
pub const EQUILIBRATE_MAX_ITERATIONS: usize = 20;
pub const EQUILIBRATE_TOLERANCE: f64 = 1e-6;

// k-means clustering
pub const KMEANS_RESTARTS: usize = 10;
pub const KMEANS_MAX_ITERATIONS: usize = 300;

// Validation checks
pub const EIGENVALUE_ZERO_THRESHOLD: f64 = 1e-4;  // For checking λ_1 ≈ 0
pub const RESIDUAL_VALIDATION_THRESHOLD: f64 = 1e-6;
```

### 3.4 Matrix Types (`matrix.rs`)

#### **CsrMatrix** - Compressed Sparse Row
```rust
use serde::{Deserialize, Serialize};
use crate::error::Result;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CsrMatrix {
    pub values: Vec<f64>,
    pub col_indices: Vec<usize>,
    pub row_pointers: Vec<usize>,
    pub nrows: usize,
    pub ncols: usize,
}

impl CsrMatrix {
    pub fn new(
        nrows: usize,
        ncols: usize,
        values: Vec<f64>,
        col_indices: Vec<usize>,
        row_pointers: Vec<usize>,
    ) -> Result<Self> {
        if row_pointers.len() != nrows + 1 {
            return Err(SpectralError::SparseFormatError(
                format!("Invalid row pointers length: expected {}, got {}", 
                    nrows + 1, row_pointers.len())
            ));
        }
        if values.len() != col_indices.len() {
            return Err(SpectralError::SparseFormatError(
                "Mismatch between values and col_indices lengths".to_string()
            ));
        }
        Ok(Self { values, col_indices, row_pointers, nrows, ncols })
    }
    
    pub fn nnz(&self) -> usize {
        self.values.len()
    }
    
    pub fn get(&self, i: usize, j: usize) -> f64 {
        if i >= self.nrows || j >= self.ncols { return 0.0; }
        let row_start = self.row_pointers[i];
        let row_end = self.row_pointers[i + 1];
        for idx in row_start..row_end {
            if self.col_indices[idx] == j {
                return self.values[idx];
            }
        }
        0.0
    }
    
    pub fn row_iter(&self, i: usize) -> impl Iterator<Item = (usize, f64)> + '_ {
        let row_start = self.row_pointers[i];
        let row_end = self.row_pointers.get(i + 1).copied().unwrap_or(0);
        (row_start..row_end).map(move |idx| {
            (self.col_indices[idx], self.values[idx])
        })
    }
    
    // Matrix-vector multiplication: y = A * x
    pub fn matvec(&self, x: &DenseVector) -> Result<DenseVector> {
        if x.len() != self.ncols {
            return Err(SpectralError::DimensionMismatch {
                expected: format!("vector length {}", self.ncols),
                actual: format!("{}", x.len()),
            });
        }
        let mut y = vec![0.0; self.nrows];
        for i in 0..self.nrows {
            for (j, val) in self.row_iter(i) {
                y[i] += val * x.get(j);
            }
        }
        Ok(DenseVector::from_vec(y))
    }
    
    // Transpose to CSC
    pub fn transpose_to_csc(&self) -> CscMatrix {
        // ... implementation
    }
    
    // Frobenius norm
    pub fn frobenius_norm(&self) -> f64 {
        self.values.iter().map(|v| v * v).sum::<f64>().sqrt()
    }
    
    // Convert to dense
    pub fn to_dense(&self) -> DenseMatrix {
        let mut data = vec![0.0; self.nrows * self.ncols];
        for i in 0..self.nrows {
            for (j, val) in self.row_iter(i) {
                data[i * self.ncols + j] = val;
            }
        }
        DenseMatrix::from_vec(self.nrows, self.ncols, data)
    }
}
```

#### **CscMatrix** - Compressed Sparse Column
```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CscMatrix {
    pub values: Vec<f64>,
    pub row_indices: Vec<usize>,
    pub col_pointers: Vec<usize>,
    pub nrows: usize,
    pub ncols: usize,
}

impl CscMatrix {
    pub fn new(
        nrows: usize,
        ncols: usize,
        values: Vec<f64>,
        row_indices: Vec<usize>,
        col_pointers: Vec<usize>,
    ) -> Result<Self> {
        if col_pointers.len() != ncols + 1 {
            return Err(SpectralError::SparseFormatError(
                format!("Invalid col pointers length: expected {}, got {}", 
                    ncols + 1, col_pointers.len())
            ));
        }
        if values.len() != row_indices.len() {
            return Err(SpectralError::SparseFormatError(
                "Mismatch between values and row_indices lengths".to_string()
            ));
        }
        Ok(Self { values, row_indices, col_pointers, nrows, ncols })
    }
    
    pub fn nnz(&self) -> usize {
        self.values.len()
    }
    
    pub fn transpose_to_csr(&self) -> CsrMatrix {
        // ... implementation
    }
}
```

#### **CooMatrix** - Coordinate Format
```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CooMatrix {
    pub row_indices: Vec<usize>,
    pub col_indices: Vec<usize>,
    pub values: Vec<f64>,
    pub nrows: usize,
    pub ncols: usize,
}

impl CooMatrix {
    pub fn new(nrows: usize, ncols: usize) -> Self {
        Self {
            row_indices: Vec::new(),
            col_indices: Vec::new(),
            values: Vec::new(),
            nrows,
            ncols,
        }
    }
    
    pub fn push(&mut self, row: usize, col: usize, val: f64) -> Result<()> {
        if row >= self.nrows || col >= self.ncols {
            return Err(SpectralError::DimensionMismatch {
                expected: format!("({}, {})", self.nrows, self.ncols),
                actual: format!("({}, {})", row, col),
            });
        }
        self.row_indices.push(row);
        self.col_indices.push(col);
        self.values.push(val);
        Ok(())
    }
    
    pub fn nnz(&self) -> usize {
        self.values.len()
    }
    
    pub fn to_csr(&self) -> Result<CsrMatrix> {
        // Accumulate by row, sort each row by column
        // ... implementation
    }
    
    pub fn to_csc(&self) -> Result<CscMatrix> {
        // ... implementation
    }
}
```

#### **DenseMatrix** - Full Dense Matrix
```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DenseMatrix {
    pub data: Vec<f64>,
    pub nrows: usize,
    pub ncols: usize,
}

impl DenseMatrix {
    pub fn new(nrows: usize, ncols: usize) -> Self {
        Self {
            data: vec![0.0; nrows * ncols],
            nrows,
            ncols,
        }
    }
    
    pub fn from_vec(nrows: usize, ncols: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), nrows * ncols);
        Self { data, nrows, ncols }
    }
    
    pub fn from_slice(nrows: usize, ncols: usize, data: &[f64]) -> Self {
        Self {
            data: data.to_vec(),
            nrows,
            ncols,
        }
    }
    
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.ncols + j]
    }
    
    #[inline]
    pub fn set(&mut self, i: usize, j: usize, val: f64) {
        self.data[i * self.ncols + j] = val;
    }
    
    pub fn row(&self, i: usize) -> &[f64] {
        &self.data[i * self.ncols..(i + 1) * self.ncols]
    }
    
    pub fn col(&self, j: usize) -> Vec<f64> {
        (0..self.nrows)
            .map(|i| self.get(i, j))
            .collect()
    }
    
    pub fn column(&self, j: usize) -> Vec<f64> {
        self.col(j)
    }
    
    pub fn row_norm_2(&self, i: usize) -> f64 {
        self.row(i).iter().map(|v| v * v).sum::<f64>().sqrt()
    }
    
    pub fn normalize_row(&mut self, i: usize) {
        let norm = self.row_norm_2(i);
        if norm > 0.0 {
            for j in 0..self.ncols {
                let idx = i * self.ncols + j;
                self.data[idx] /= norm;
            }
        }
    }
    
    pub fn matvec(&self, x: &DenseVector) -> Result<DenseVector> {
        if x.len() != self.ncols {
            return Err(SpectralError::DimensionMismatch {
                expected: format!("vector length {}", self.ncols),
                actual: format!("{}", x.len()),
            });
        }
        let mut y = vec![0.0; self.nrows];
        for i in 0..self.nrows {
            for j in 0..self.ncols {
                y[i] += self.get(i, j) * x.get(j);
            }
        }
        Ok(DenseVector::from_vec(y))
    }
    
    pub fn frobenius_norm(&self) -> f64 {
        self.data.iter().map(|v| v * v).sum::<f64>().sqrt()
    }
    
    pub fn transpose(&self) -> DenseMatrix {
        let mut new_data = vec![0.0; self.data.len()];
        for i in 0..self.nrows {
            for j in 0..self.ncols {
                new_data[j * self.nrows + i] = self.get(i, j);
            }
        }
        DenseMatrix::from_vec(self.ncols, self.nrows, new_data)
    }
}
```

### 3.5 Vector Types (`vector.rs`)

```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DenseVector {
    pub data: Vec<f64>,
}

impl DenseVector {
    pub fn new(len: usize) -> Self {
        Self {
            data: vec![0.0; len],
        }
    }
    
    pub fn from_vec(data: Vec<f64>) -> Self {
        Self { data }
    }
    
    pub fn from_slice(data: &[f64]) -> Self {
        Self {
            data: data.to_vec(),
        }
    }
    
    #[inline]
    pub fn get(&self, i: usize) -> f64 {
        self.data[i]
    }
    
    #[inline]
    pub fn set(&mut self, i: usize, val: f64) {
        self.data[i] = val;
    }
    
    pub fn len(&self) -> usize {
        self.data.len()
    }
    
    pub fn norm_2(&self) -> f64 {
        self.data.iter().map(|v| v * v).sum::<f64>().sqrt()
    }
    
    pub fn norm_inf(&self) -> f64 {
        self.data.iter().map(|v| v.abs()).fold(0.0, f64::max)
    }
    
    pub fn dot(&self, other: &DenseVector) -> f64 {
        self.data.iter().zip(&other.data).map(|(a, b)| a * b).sum()
    }
    
    pub fn scale(&mut self, alpha: f64) {
        for val in &mut self.data {
            *val *= alpha;
        }
    }
    
    // y := a*x + y
    pub fn axpy(&mut self, alpha: f64, x: &DenseVector) {
        for (y, x_val) in self.data.iter_mut().zip(&x.data) {
            *y += alpha * x_val;
        }
    }
    
    pub fn iter(&self) -> impl Iterator<Item = f64> + '_ {
        self.data.iter().copied()
    }
    
    // Element-wise square
    pub fn elementwise_square(&self) -> DenseVector {
        DenseVector::from_vec(self.data.iter().map(|v| v * v).collect())
    }
}
```

### 3.6 Traits (`traits.rs`)

```rust
pub trait MatrixLike: Send + Sync {
    type VectorType: VectorLike;
    
    fn nrows(&self) -> usize;
    fn ncols(&self) -> usize;
    fn nnz(&self) -> usize;
    
    fn matvec(&self, x: &Self::VectorType) -> crate::Result<Self::VectorType>;
    fn matvec_adjoint(&self, x: &Self::VectorType) -> crate::Result<Self::VectorType>;
    fn diagonal(&self) -> Self::VectorType;
    fn frobenius_norm(&self) -> f64;
}

pub trait VectorLike: Clone + Send + Sync {
    fn len(&self) -> usize;
    fn norm_2(&self) -> f64;
    fn norm_inf(&self) -> f64;
    fn dot(&self, other: &Self) -> f64;
    fn scale(&mut self, alpha: f64);
    fn axpy(&mut self, alpha: f64, x: &Self);
    fn iter(&self) -> Box<dyn Iterator<Item = f64> + '_>;
}

impl MatrixLike for CsrMatrix {
    type VectorType = DenseVector;
    
    fn nrows(&self) -> usize { self.nrows }
    fn ncols(&self) -> usize { self.ncols }
    fn nnz(&self) -> usize { self.nnz() }
    
    fn matvec(&self, x: &DenseVector) -> crate::Result<DenseVector> {
        self.matvec(x)
    }
    
    fn matvec_adjoint(&self, x: &DenseVector) -> crate::Result<DenseVector> {
        self.transpose_to_csc().matvec(x)
    }
    
    fn diagonal(&self) -> DenseVector {
        let mut diag = DenseVector::new(self.nrows.min(self.ncols));
        for i in 0..diag.len() {
            diag.set(i, self.get(i, i));
        }
        diag
    }
    
    fn frobenius_norm(&self) -> f64 {
        self.frobenius_norm()
    }
}

impl MatrixLike for DenseMatrix {
    type VectorType = DenseVector;
    
    fn nrows(&self) -> usize { self.nrows }
    fn ncols(&self) -> usize { self.ncols }
    fn nnz(&self) -> usize { self.data.len() }
    
    fn matvec(&self, x: &DenseVector) -> crate::Result<DenseVector> {
        self.matvec(x)
    }
    
    fn matvec_adjoint(&self, x: &DenseVector) -> crate::Result<DenseVector> {
        self.transpose().matvec(x)
    }
    
    fn diagonal(&self) -> DenseVector {
        let mut diag = DenseVector::new(self.nrows.min(self.ncols));
        for i in 0..diag.len() {
            diag.set(i, self.get(i, i));
        }
        diag
    }
    
    fn frobenius_norm(&self) -> f64 {
        self.frobenius_norm()
    }
}

impl VectorLike for DenseVector {
    fn len(&self) -> usize { self.len() }
    fn norm_2(&self) -> f64 { self.norm_2() }
    fn norm_inf(&self) -> f64 { self.norm_inf() }
    fn dot(&self, other: &Self) -> f64 { self.dot(other) }
    fn scale(&mut self, alpha: f64) { self.scale(alpha) }
    fn axpy(&mut self, alpha: f64, x: &Self) { self.axpy(alpha, x) }
    fn iter(&self) -> Box<dyn Iterator<Item = f64> + '_> {
        Box::new(self.iter())
    }
}
```

### 3.7 Decomposition Types (`decomp.rs`)

```rust
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Partition {
    pub blocks: Vec<Vec<usize>>,        // Variable indices in each block
    pub k: usize,                       // Number of blocks
    pub quality_metric: Option<f64>,    // Silhouette score
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EigendecompositionResult {
    pub eigenvalues: DenseVector,       // λ_1 ≤ λ_2 ≤ ... ≤ λ_{k+1}
    pub eigenvectors: DenseMatrix,      // V ∈ ℝ^{n×(k+1)}, row-major
    pub k: usize,
    pub converged: bool,
    pub residuals: Option<Vec<f64>>,    // ‖L·v_i - λ_i·v_i‖ / |λ_i|
    pub iterations: usize,
    pub method: String,                 // "ARPACK", "LOBPCG", "shift-invert"
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SpectralFeatureVector {
    pub gamma_2: f64,                   // SF1: Spectral gap
    pub spectral_ratio: f64,            // SF2: δ²/γ²
    pub decay_rate: f64,                // SF3: Eigenvalue decay
    pub fiedler_entropy: f64,           // SF4: Localization entropy
    pub algebraic_connectivity_ratio: f64, // SF5: γ_2/γ_k
    pub coupling_energy: f64,           // SF6: δ²
    pub separability_index: f64,        // SF7: Silhouette
    pub effective_spectral_dimension: usize, // SF8
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CrossingWeightResult {
    pub total_crossing_weight: f64,
    pub crossing_hyperedges: Vec<usize>,
    pub dual_weights: Vec<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LaplacianVariant {
    pub variant: String,                // "clique" or "incidence"
    pub d_max: usize,
    pub n: usize,
    pub m: usize,
    pub nnz: usize,
}
```

### 3.8 Preprocessing Types (`preprocess.rs`)

```rust
use crate::{DenseVector, CsrMatrix, Scalar};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum EquilibrationMethod {
    Ruiz,           // Ruiz iterative scaling
    Geometric,      // Geometric-mean scaling
    ScipNative,     // Use SCIP's internal scaling
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EquilibrationResult {
    pub equilibrated_matrix: CsrMatrix,
    pub row_scaling: DenseVector,
    pub col_scaling: DenseVector,
    pub condition_number_estimate: f64,
    pub method: EquilibrationMethod,
    pub iterations: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LaplacianMetadata {
    pub variant: String,                // "clique" or "incidence"
    pub d_max: usize,
    pub nnz: usize,
    pub n: usize,
    pub m: usize,
    pub kappa_estimate_row: f64,
    pub kappa_estimate_col: f64,
}
```

---

## 4. NEXT STEPS: IMPLEMENTATION ORDER

1. **spectral-types/src/lib.rs** (⚠️ BLOCKING - all others depend on it)
   - Create lib.rs with all modules listed above
   - Implement error types (short, straightforward)
   - Implement constants
   - Implement CsrMatrix, CscMatrix, CooMatrix (matrix construction & basic ops)
   - Implement DenseMatrix, DenseVector (dense operations)
   - Implement traits MatrixLike, VectorLike
   - Implement decomposition and preprocessing type definitions

2. **spectral-core** (depends on spectral-types)
   - Laplacian construction (Algorithm 1)
   - Eigensolve pipeline (Algorithm 2)
   - Feature extraction (Algorithm 2.2)
   - Partition recovery (Algorithm 3)

3. **matrix-decomp** (depends on spectral-types)
   - LU, QR, SVD decompositions
   - Eigendecomposition wrappers

4. **oracle** (depends on all three)
   - Decomposition classifiers
   - Futility predictor
   - L3 bound computation

---

## 5. KEY DESIGN DECISIONS

1. **All numerical types are f64 (double precision)**
2. **Row-major storage for dense matrices (standard for linear algebra)**
3. **CSR is primary sparse format; CSC used for transposes**
4. **COO used only during construction, converted to CSR/CSC**
5. **All public matrices/vectors are immutable by default (mutating methods mark &mut)**
6. **Error handling via thiserror + anyhow for ergonomic Result types**
7. **Traits are used for algorithm abstraction (ARPACK can work with any MatrixLike)**
8. **Serialization via serde for data persistence/logging**

---

## 6. THEORETICAL GROUNDING

All type definitions follow Algorithm 1-4 specifications from `theory/algorithms.md`:

| Algorithm | Input | Output | Type |
|-----------|-------|--------|------|
| 1: Laplacian Construction | CsrMatrix (A) | CsrMatrix (L_H) | SpectralCore |
| 2: Robust Eigensolve | CsrMatrix (L_H) | EigendecompositionResult | SpectralCore |
| 2.2: Feature Extraction | EigendecompositionResult | SpectralFeatureVector | SpectralCore |
| 3: Spectral Partition | DenseMatrix (V_k) | Partition | SpectralCore |
| L3: Crossing Weight | Partition + CsrMatrix | CrossingWeightResult | Oracle |

All 8 spectral features are defined in `SpectralFeatureVector`.

---

## FILES CREATED IN THIS EXPLORATION

1. **IMPLEMENTATION_SPECIFICATION.md** (this file)
   - Location: spectral-decomposition-oracle/IMPLEMENTATION_SPECIFICATION.md

---

## KEY FILES FOR REFERENCE

- **theory/algorithms.md** (1151 lines) - Complete algorithm specifications
- **theory/approach.json** - Mathematical foundations and approach
- **ideation/final_approach.md** - Approved final approach (binding spec)
- **theory/verification_framework.md** - Quality assurance criteria

