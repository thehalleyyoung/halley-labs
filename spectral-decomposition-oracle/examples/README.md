# Examples

Runnable examples demonstrating the Spectral Decomposition Oracle.

## Examples

| File | Description |
|------|-------------|
| `basic_analysis.rs` | Build a sparse matrix, construct the constraint hypergraph, compute the normalized Laplacian, and extract spectral features. Good starting point. |
| `decomposition_selection.rs` | Full end-to-end pipeline: spectral + syntactic features → oracle classification → futility check → decomposition → certificate. |
| `mps_parsing.rs` | Parse MPS (Mathematical Programming System) and LP (CPLEX LP) file formats into `MipInstance`. |
| `solver_comparison.rs` | Compare solver backends: internal simplex, interior point, SCIP emulation, HiGHS emulation. |
| `miplib_census.rs` | Run a mini MIPLIB 2017 decomposition census on synthetic instances. |

## Running

```bash
# From the implementation/ directory:
cargo run --example basic_analysis
cargo run --example decomposition_selection
cargo run --example mps_parsing
cargo run --example solver_comparison
cargo run --example miplib_census
```

## What They Demonstrate

### `basic_analysis`

1. **Sparse matrix creation** — COO format construction and CSR conversion
2. **Hypergraph construction** — Constraints as hyperedges over variable vertices
3. **Laplacian selection** — Clique expansion (d_max ≤ 200) vs Bolla incidence (d_max > 200)
4. **Eigensolve** — ARPACK shift-invert with LOBPCG fallback
5. **Feature extraction** — 8 spectral features (spectral gap, Cheeger estimate, etc.)

### `decomposition_selection`

1. Everything in `basic_analysis`, plus:
2. **Syntactic features** — Matrix dimensions, density, structure statistics
3. **Oracle classifier** — Ensemble (random forest + gradient boosting) method selection
4. **Futility prediction** — Calibrated probability that decomposition is futile
5. **Decomposition execution** — Benders, Dantzig-Wolfe, or Lagrangian relaxation
6. **Certificate** — Dual feasibility check, Proposition T2 bound, JSON report

### `mps_parsing`

1. **MPS format** — Parse the standard MIP file format including ROWS, COLUMNS, RHS, BOUNDS, and INTEGER markers
2. **LP format** — Parse CPLEX LP format with objectives, constraints, and bounds
3. **Instance statistics** — Variable counts, constraint types, coefficient ranges

### `solver_comparison`

1. **Backend selection** — Switch between internal simplex, interior point, SCIP emulation, and HiGHS emulation
2. **Feature flags** — Enable real SCIP/HiGHS with `--features scip` or `--features highs`
3. **Performance comparison** — Same LP solved by different backends

### `miplib_census`

1. **Census pipeline** — Iterate over instances, extract features, classify structure
2. **Structure detection** — Identify block-angular, bordered-block-diagonal, staircase, network patterns
3. **Method recommendation** — Spectral-gap-based heuristic for decomposition selection
