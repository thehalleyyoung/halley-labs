# Benchmarks

Benchmark suite for the Spectral Decomposition Oracle.

## MIPLIB 2017 Benchmark Suite

Compares three decomposition approaches on MIPLIB 2017 instances:

| Configuration | Description |
|---|---|
| **Default SCIP** | Monolithic solve, no decomposition |
| **GCG auto-decomposition** | Dantzig-Wolfe via hMETIS partitioning |
| **SpectralOracle** | Spectral-guided decomposition selection |

### Metrics

- **Solve time** (seconds): wall-clock time to optimality or time limit
- **Gap closed** (%): `(z_D - z_LP_root) / (z* - z_LP_root)`
- **Decomposition quality score**: spectral gap, crossing weight
- **Prediction accuracy**: oracle recommendation vs best method

### Quick Start

```bash
# Install spectral-oracle
cd implementation && cargo install --path spectral-cli

# Download MIPLIB 2017 benchmark instances
mkdir -p miplib2017
# Get instances from https://miplib.zib.de/tag_benchmark.html

# Run pilot benchmark (10 instances)
python3 benchmarks/run_miplib_benchmark.py --tier pilot

# Run paper-grade benchmark (200 instances)
python3 benchmarks/run_miplib_benchmark.py --tier paper --time-limit 600

# Full census (1,065 instances)
python3 benchmarks/run_miplib_benchmark.py --tier artifact
```

### Benchmark Tiers

| Tier | Instances | Per-Instance Limit | Total Time (est.) |
|---|---|---|---|
| pilot | 10 | 60s | ~10 min |
| dev | 50 | 300s | ~4 hours |
| paper | 200 | 600s | ~33 hours |
| artifact | 1,065 | 3,600s | ~12 days (4 cores) |

## Contents

| File | Description |
|------|-------------|
| `run_miplib_benchmark.py` | MIPLIB 2017 comparison: SCIP vs GCG vs SpectralOracle |
| `run_benchmarks.py` | Python simulation harness — synthetic sparse matrices |
| `benchmark_suite.sh` | Shell driver — builds, benches, and reports |
| `results/` | Output directory (created at runtime) |

## Criterion Microbenchmarks

```bash
cd implementation
cargo bench -p spectral-cli
```

Benchmarks:
- `hypergraph_construction`: Building constraint hypergraphs (2–16 blocks)
- `eigensolve`: Lanczos/LOBPCG eigenvalue computation (50–200 dim)
- `spectral_features`: Feature extraction (gap, entropy, decay rate)
- `mps_parsing`: MPS file parser throughput (50×100 to 200×400)

## Simulation Benchmarks

### `run_benchmarks.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--sizes` | `100,500,1000,5000,10000` | Comma-separated matrix sizes |
| `--trials` | `5` | Trials per matrix size |
| `--density` | `0.02` | Sparse matrix fill density |
| `--output` | `benchmarks/results` | Output directory |

### `run_miplib_benchmark.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--instances-dir` | `miplib2017` | Directory with .mps files |
| `--tier` | `pilot` | Benchmark tier (pilot/dev/paper/artifact) |
| `--time-limit` | tier-dependent | Per-instance time limit |
| `--skip-scip` | false | Skip SCIP baseline |
| `--skip-gcg` | false | Skip GCG comparison |

## Requirements

- Rust 1.75+ and `cargo`
- Python 3.8+
- Optional: SCIP on PATH for baseline comparison
- Optional: GCG on PATH for DW comparison
- MIPLIB 2017 instances from https://miplib.zib.de/
