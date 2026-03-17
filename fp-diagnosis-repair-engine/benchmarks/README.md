# Benchmarks

This directory contains benchmark suites for evaluating Penumbra against
state-of-the-art floating-point tools.

## Directory Structure

```
benchmarks/
├── fpbench/                    # FPBench-format benchmark expressions
│   └── standard_suite.fpcore   # Standard FPBench + Herbie benchmarks
├── results/                    # Benchmark output (generated)
│   └── comparison.json         # Penumbra vs Herbie comparison
└── README.md
```

## Running Benchmarks

### Criterion microbenchmarks (Rust)

```bash
cd implementation
cargo bench -p fpdiag-bench
```

Three benchmark groups are included:

| Group | What it measures |
|-------|-----------------|
| `eag_construction` | EAG build time for linear chains and mixed-op traces (10–1000 ops) |
| `diagnosis_engine` | Diagnosis throughput for cancellation and absorption patterns |
| `fpbench_suite` | Full pipeline (EAG → diagnose → repair) on FPBench expressions |

### FPBench comparison

The `fpbench_suite` benchmark compares Penumbra's diagnosis + repair against
published Herbie results on standard benchmarks:

| Benchmark | Herbie bits | Penumbra diagnosis | Penumbra repair |
|-----------|------------|-------------------|----------------|
| expm1 | 52 | cancellation | expm1 substitution |
| log1p | 52 | cancellation | log1p substitution |
| NMSE-3.1 | 51 | cancellation | algebraic rationalization |
| NMSE-3.3 | 50 | cancellation | algebraic simplification |
| quadratic | 50 | cancellation | stable quadratic formula |
| hypot | 53 | overflow | hypot substitution |

### Running via CLI

```bash
cargo run -p fpdiag-cli -- bench all --output benchmarks/results/
```

## FPBench Format

Benchmarks use the [FPBench](https://fpbench.org) standard format (FPCore 2.0).
Each `.fpcore` file contains S-expression benchmark definitions that Penumbra
can parse directly via `fpdiag_types::fpbench::parse_fpbench_file`.

## Comparison Methodology

Penumbra and Herbie address complementary aspects of FP error:

- **Herbie**: Expression-level rewriting via e-graph equality saturation.
  Optimizes single expressions for numerical accuracy.
- **Penumbra**: Pipeline-level diagnosis and repair via Error Amplification
  Graphs. Diagnoses *why* error accumulated and prescribes targeted repairs.

On single-expression benchmarks, both tools achieve similar accuracy.
Penumbra's advantage is on multi-stage pipelines where error propagates
across function boundaries — a scenario Herbie cannot address.
