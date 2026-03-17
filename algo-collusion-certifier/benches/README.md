# Benchmarks

## Quick Start

### Standalone benchmark examples (recommended)

```bash
# Run both benchmarks in release mode
./benchmarks/run_benchmarks.sh

# Run individually
cargo run --example bench_detection --release
cargo run --example bench_certificate_gen --release

# Or just one via the script
./benchmarks/run_benchmarks.sh detection
./benchmarks/run_benchmarks.sh certificate
```

Results are saved to `benchmarks/results/`.

### Criterion benchmarks (`cargo bench`)

The `benches/` directory contains Criterion-based microbenchmarks. To run them:

```bash
# Run all Criterion benchmarks
cargo bench

# Run a specific benchmark group
cargo bench --bench detection_bench
cargo bench --bench certificate_bench

# Generate an HTML report (requires gnuplot)
cargo bench -- --output-format bencher
```

Criterion automatically tracks regressions across runs and stores results in
`target/criterion/`.

## Benchmark Descriptions

| Benchmark | What it measures |
|-----------|-----------------|
| `bench_detection` | M1 composite test, Harrington PCM screening, and correlation-based detection on synthetic Bertrand trajectories (1K / 10K / 100K rounds) |
| `bench_certificate_gen` | Layer 0 certificate construction, proof checker verification, and Merkle tree construction at varying sizes |
| `detection_bench` (Criterion) | Microbenchmark for composite test execution |
| `certificate_bench` (Criterion) | Microbenchmark for certificate build + verify cycle |

## Output Formats

Both standalone benchmarks emit:
- **CSV** — machine-parseable, pipe to downstream analysis
- **JSON** — for dashboard integration
- **Table** — human-readable comparison
