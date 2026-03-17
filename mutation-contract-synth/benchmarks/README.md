# MutSpec Benchmark Suite

Benchmarks for evaluating the MutSpec mutation-driven contract synthesis pipeline.

## Quick start

```bash
# Run the full benchmark suite
./benchmarks/run_benchmarks.sh

# Compare against Daikon and SpecFuzzer baselines
./benchmarks/compare_tools.sh

# Run scalability experiments
./benchmarks/scalability_test.sh
```

## Scripts

| Script                | Purpose                                         |
|-----------------------|-------------------------------------------------|
| `run_benchmarks.sh`   | Run synthesis on all examples, collect metrics   |
| `compare_tools.sh`    | Compare MutSpec vs. Daikon vs. SpecFuzzer        |
| `scalability_test.sh` | Measure synthesis time as program size grows      |

## Metrics collected

- **Contracts synthesized** – number of pre/postcondition clauses produced
- **Mutation score** – fraction of non-equivalent mutants killed by the contract
- **Precision** – fraction of synthesized clauses that are correct (match ground truth)
- **Recall** – fraction of ground-truth clauses that are synthesized
- **F1** – harmonic mean of precision and recall
- **Synthesis time** – wall-clock seconds for the full pipeline
- **Z3 queries** – number of SMT solver calls

## Data files

The `data/` directory contains:

| File                       | Description                                         |
|----------------------------|-----------------------------------------------------|
| `benchmark_suite.json`     | Suite metadata: programs, ground truth, parameters   |
| `daikon_baseline.json`     | Pre-computed Daikon invariant results                |
| `specfuzzer_baseline.json` | Pre-computed SpecFuzzer invariant results             |

## Output

`run_benchmarks.sh` writes results to:
- `benchmarks/results/results_<timestamp>.json` – machine-readable results
- `benchmarks/results/results_<timestamp>.txt` – formatted table for humans

`compare_tools.sh` writes to:
- `benchmarks/results/comparison_<timestamp>.json`
- `benchmarks/results/comparison_<timestamp>.txt`

`scalability_test.sh` writes to:
- `benchmarks/results/scalability_<timestamp>.dat` – tab-separated data for plotting
- `benchmarks/results/scalability_<timestamp>.txt` – formatted table

## Reproducing paper results

```bash
# Full reproduction (takes ~10 minutes)
./benchmarks/run_benchmarks.sh --trials 10
./benchmarks/compare_tools.sh
./benchmarks/scalability_test.sh --max-sites 500
```
