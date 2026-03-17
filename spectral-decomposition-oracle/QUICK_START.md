# Spectral Decomposition Oracle - Quick Start Guide

## TL;DR

**Status:** Rust workspace fully configured, ALL 7 crates are EMPTY and waiting for implementation
**Start with:** `spectral-types/src/lib.rs` - all other crates depend on it

---

## Files Generated For You

1. **EXPLORATION_SUMMARY.md** (294 lines)
   - High-level overview of project, current status, next steps
   - Start here if you want the big picture

2. **IMPLEMENTATION_SPECIFICATION.md** (818 lines) ⭐ PRIMARY REFERENCE
   - Complete type definitions ready to implement
   - Rust code templates for all matrix, vector, trait types
   - Error handling patterns
   - Trait implementations

3. **theory/algorithms.md** (1151 lines) - Mathematical Foundation
   - Algorithm 1: Hypergraph Laplacian Construction
   - Algorithm 2: Robust Eigensolve  
   - Algorithm 3: Spectral Partition Recovery
   - Algorithm 4: Crossing Weight Computation
   - Complete pseudocode with complexity analysis

---

## What Needs To Be Built

### Priority 1: spectral-types/src/lib.rs (BLOCKING)

**What to implement:**
- Error types: `SpectralError`, `Result<T>`
- Constants: Numerical tolerances, algorithm parameters
- Matrix types: `CsrMatrix`, `CscMatrix`, `CooMatrix`, `DenseMatrix`
- Vector type: `DenseVector`
- Traits: `MatrixLike`, `VectorLike`
- Decomposition types: `Partition`, `EigendecompositionResult`, `SpectralFeatureVector`
- Preprocessing types: `EquilibrationResult`, `LaplacianMetadata`

**Template:** See IMPLEMENTATION_SPECIFICATION.md § 3 (Matrix Types) and § 4 (Error Types)

### Priority 2: spectral-core (depends on spectral-types)

Implements Algorithms 1-3 from theory/algorithms.md:
- Laplacian construction (clique expansion + incidence-matrix variants)
- Eigensolve pipeline (ARPACK + LOBPCG fallback)
- Feature extraction (compute 8 spectral features)
- Partition recovery (k-means on eigenvectors)

### Priority 3: matrix-decomp (depends on spectral-types)

Matrix decomposition algorithms (LU, QR, SVD, eigendecomposition)

### Priority 4-7: oracle, optimization, certificate, spectral-cli

Depend on spectral-core and matrix-decomp

---

## The 8 Spectral Features

All must be computable in `SpectralFeatureVector`:

1. **Spectral gap γ₂** = λ₂ of Laplacian
2. **Spectral ratio δ²/γ²** = coupling energy / gap²
3. **Eigenvalue decay rate β** = exponential decay of eigenvalues
4. **Fiedler entropy H** = localization of Fiedler vector
5. **Connectivity ratio γ₂/γₖ** = 2nd vs k-th eigenvalue
6. **Coupling energy δ²** = ‖L_H - L_block‖_F²
7. **Separability index** = silhouette of k-means partition
8. **Spectral dimension** = count of near-zero eigenvalues

---

## Key Design Principles

- ✅ All numbers are `f64` (double precision)
- ✅ Row-major storage for dense matrices
- ✅ CSR primary sparse format, CSC for transposes
- ✅ Traits for algorithm abstraction
- ✅ Error handling via `thiserror`
- ✅ Serialization via `serde`

---

## Dependencies Already Configured

No additional crates needed beyond workspace dependencies:
- `serde` + `serde_json` for serialization
- `thiserror` for error definitions
- `num-traits`, `num-complex` for numerics
- `rayon` for parallelism

---

## Testing Files

From `theory/verification_framework.md`:

- Definitions (D1-D5): Types precisely defined, no ambiguity
- Proofs (P1-P5): Algorithms have clear justifications
- Algorithms (A1-A5): Pseudocode → Rust must be faithful
- Gates (G1, G3): Empirical validation on MIPLIB 2017

---

## Where To Find Information

| Topic | File | Location |
|-------|------|----------|
| High-level overview | EXPLORATION_SUMMARY.md | Root |
| Type specifications | IMPLEMENTATION_SPECIFICATION.md | Root |
| Algorithm details | theory/algorithms.md | theory/ |
| Mathematical approach | theory/approach.json | theory/ |
| Approved final approach | ideation/final_approach.md | ideation/ |
| Quality framework | theory/verification_framework.md | theory/ |

---

## Success Criteria

✅ spectral-types compiles with all types defined
✅ spectral-core compiles and implements Algorithms 1-4
✅ matrix-decomp compiles and implements LU/QR/SVD
✅ oracle compiles with decomposition classifiers
✅ All 8 spectral features computable from eigendecomposition
✅ Error handling covers all edge cases (NaN propagation, dimension mismatches, convergence failures)

---

## Quick Command Reference

```bash
cd /Users/halleyyoung/Documents/div/mathdivergence/pipeline_staging/spectral-decomposition-oracle/implementation

# Check workspace structure
cargo metadata --format-version 1 | jq '.workspace_members'

# Build spectral-types first
cargo build -p spectral-types

# Then build dependent crates
cargo build -p spectral-core
cargo build -p matrix-decomp
cargo build -p oracle

# Run tests
cargo test -p spectral-types
```

---

## Notes

- All matrices use row-major dense storage or CSR sparse storage
- All vectors are column vectors (n×1)
- Eigenvalues/eigenvectors returned in ascending eigenvalue order
- Partitions are always disjoint and exhaustive
- Eigensolve failure → NaN (handled by downstream ML classifiers)

---

**You are here:** Ready to implement spectral-types with complete specifications provided.
**Next step:** Read IMPLEMENTATION_SPECIFICATION.md § 3 and start coding spectral-types/src/lib.rs
