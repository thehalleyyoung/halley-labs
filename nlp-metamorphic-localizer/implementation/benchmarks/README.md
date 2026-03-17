# NLP Metamorphic Localizer — Benchmarks

## Quick Start

```bash
# Run all benchmarks
cargo bench

# Run a specific benchmark suite
cargo bench --bench localization_bench
cargo bench --bench transformation_bench
cargo bench --bench shrinking_bench
cargo bench --bench pipeline_bench

# Run a single benchmark group
cargo bench --bench localization_bench -- "localization/sbfl_metrics"

# Run with a filter pattern
cargo bench --bench transformation_bench -- "individual/passivization"
```

## Benchmark Suites

### 1. `localization_bench` — Fault Localization

| Group | What it measures |
|-------|-----------------|
| `localization/pipeline` | End-to-end localization on 3/5/7-stage pipelines with 100–1000 test cases |
| `localization/sbfl_metrics` | Ochiai, Tarantula, D\*, Barinel suspiciousness on differential matrices |
| `localization/causal_intervention` | Single-stage intervention and iterative peeling |
| `localization/discriminability` | Stage discriminability matrix rank and condition number |
| `localization/eval_metrics` | Top-k accuracy, EXAM score, wasted effort, full metric suites |

### 2. `transformation_bench` — Metamorphic Transformations

| Group | What it measures |
|-------|-----------------|
| `transformations/individual` | Throughput of each of the 15 transformations |
| `transformations/composition` | Pair and triple composition latency |
| `transformations/registry` | `get_applicable`, `apply_all_applicable`, coverage analysis |
| `transformations/mr_check` | MR checking (semantic equivalence, entity/sentiment preservation, syntactic consistency) |

### 3. `shrinking_bench` — Input Shrinking (GCHDD)

| Group | What it measures |
|-------|-----------------|
| `shrinking/gchdd` | GCHDD on 10/20/40/80-word sentences |
| `shrinking/gchdd_binary` | GCHDD with binary search enabled |
| `shrinking/always_accept` | Baseline with oracle that accepts everything |
| `shrinking/grammar_check` | Grammar validity checker throughput |
| `shrinking/unification` | Feature unification and constraint checking |
| `shrinking/tree_construction` | `ShrinkableTree::from_sentence` overhead |

### 4. `pipeline_bench` — NLP Pipeline & Differential Analysis

| Group | What it measures |
|-------|-----------------|
| `pipeline/execution` | SpaCy / HuggingFace / Stanza adapter throughput |
| `pipeline/prefix` | Partial pipeline execution (prefix up to stage k) |
| `pipeline/ir_capture` | IR snapshot capture overhead per stage |
| `pipeline/distance` | Stage differential (Δ_k) computation |
| `pipeline/fragility_index` | Behavioral Fragility Index computation |
| `pipeline/cumulative_deltas` | Cumulative delta, fragility index, max-jump detection |

## Viewing Results

Criterion generates HTML reports in `target/criterion/`. Open the index:

```bash
open target/criterion/report/index.html   # macOS
xdg-open target/criterion/report/index.html  # Linux
```

## Comparing Against Baseline

After an initial run, Criterion automatically stores baselines. Subsequent runs
show percentage regressions/improvements. To save a named baseline:

```bash
cargo bench -- --save-baseline v0.1.0
cargo bench -- --baseline v0.1.0   # compare current against v0.1.0
```

## Baseline Reference Numbers

See [`baseline_results.json`](baseline_results.json) for expected performance on
a modern laptop CPU (Apple M2 / Intel i7-12700H class). Key numbers:

| Metric | Value |
|--------|-------|
| Localization top-1 accuracy | 87 % |
| Localization top-2 accuracy | 96 % |
| Mean shrinking ratio | 5.2× |
| Transformation throughput | 8 000–45 000 /sec |
| False positive rate | < 8 % |

## CI Integration

Add to your CI pipeline:

```yaml
- name: Run benchmarks (compile check)
  run: cargo bench --no-run

- name: Run benchmarks (short)
  run: cargo bench -- --warm-up-time 1 --measurement-time 3
```
