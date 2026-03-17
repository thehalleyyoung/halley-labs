# IsoSpec Benchmark Results

Benchmarks run with `cargo bench -p isospec-bench --bench isolation_verification` using
[Criterion.rs](https://bheisler.github.io/criterion.rs/book/). Results are from a single
machine and should be interpreted relatively, not absolutely.

## Environment

- **Crate version:** 0.1.0
- **Rust edition:** 2021
- **Benchmark framework:** Criterion 0.5

## History Construction

| Scenario | Median | Notes |
|----------|--------|-------|
| 10 txn × 10 ops | ~15 µs | Baseline: small workload |
| 50 txn × 20 ops | ~108 µs | Moderate OLTP batch |
| 100 txn × 10 ops | ~142 µs | Many short transactions |
| 100 txn × 50 ops | ~550 µs | Large analytical workload |

**Scaling:** Roughly linear in total operations (txn × ops).

## History Queries

| Query | Median | Notes |
|-------|--------|-------|
| `committed_transactions()` | ~740 ns | O(n) scan over transaction map |
| `event_count()` | ~0.5 ns | O(1) cached length |
| `events_for_txn()` | ~218 ns | Single transaction lookup |
| `items_read_by()` | ~614 ns | Read-set extraction |
| `items_written_by()` | ~763 ns | Write-set extraction |
| `referenced_tables()` | ~15 µs | Full event scan |
| `is_well_formed()` | ~14 µs | Validation pass |

## Anomaly Detection Scenarios

### Write Skew (G2-item)

| Transaction Pairs | Median | Notes |
|-------------------|--------|-------|
| 2 pairs (4 txn) | ~1.8 µs | Minimal write-skew |
| 5 pairs (10 txn) | ~5.3 µs | Small workload |
| 10 pairs (20 txn) | ~8.1 µs | Medium workload |
| 20 pairs (40 txn) | ~17 µs | Linear scaling |

### Phantom Read (G2)

| Transactions | Median | Notes |
|-------------|--------|-------|
| 5 txn | ~3.4 µs | Small range-scan workload |
| 10 txn | ~7.1 µs | |
| 20 txn | ~16 µs | |
| 50 txn | ~40 µs | Linear scaling confirmed |

## Item Scaling (10 transactions, variable items)

| Items/txn | Median | Throughput |
|-----------|--------|------------|
| 10 | ~14 µs | ~7.1 Mops/s |
| 50 | ~67 µs | ~7.5 Mops/s |
| 100 | ~113 µs | ~8.8 Mops/s |
| 500 | ~484 µs | ~10.3 Mops/s |
| 1000 | ~1.5 ms | ~6.7 Mops/s |

## Isolation Classification

| Operation | Median | Notes |
|-----------|--------|-------|
| All levels (anomaly enum) | ~877 ns | 8 levels × prevented + possible |
| Pairwise comparison | ~49 ns | 8×8 = 64 comparisons |

## Running Benchmarks

```bash
cd implementation
cargo bench -p isospec-bench --bench isolation_verification

# Quick test (no measurement)
cargo bench -p isospec-bench --bench isolation_verification -- --test

# Specific benchmark group
cargo bench -p isospec-bench --bench isolation_verification -- "write_skew"

# HTML reports
# After running, open target/criterion/report/index.html
```
