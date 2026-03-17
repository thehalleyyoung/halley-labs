# Benchmark Suite

Performance benchmarks for the NegSynth protocol downgrade synthesis pipeline.

## Running benchmarks

All benchmarks use [Criterion.rs](https://github.com/bhaxx87/criterion.rs) and can be run from the `implementation/` directory:

```bash
cd implementation

# Run all benchmarks across all crates
cargo bench

# Run benchmarks for a single crate
cargo bench -p negsyn-merge
cargo bench -p negsyn-slicer
cargo bench -p negsyn-encode

# Run a specific benchmark by name
cargo bench -p negsyn-merge -- merge_cipher_suites
cargo bench -p negsyn-slicer -- slicer_scaling
cargo bench -p negsyn-encode -- dy_encoding
```

## Benchmark descriptions

### negsyn-merge — `merge_operator`

| Benchmark | What it measures |
|-----------|-----------------|
| `merge_cipher_suites/N` | Protocol-aware merge of states with *N* cipher suites (16, 32, 64, 128). Measures how merge cost scales with negotiation breadth. |
| `merge_vs_naive` | Compares the protocol-aware merge operator against a baseline naive (union) merge. Quantifies the overhead of lattice checking and monotonicity verification. |
| `merge_tls_negotiation` | End-to-end merge of realistic TLS 1.2/1.3 negotiation states including cipher suites, extensions, and version information. |

### negsyn-slicer — `slicer_performance`

| Benchmark | What it measures |
|-----------|-----------------|
| `slicer_scaling/N` | Program slicing on synthetic program dependency graphs with *N* nodes (64, 256, 1024, 4096). Captures PDG construction + backward slice. |
| `callgraph_construction` | Call-graph construction from a synthetic module with indirect calls and vtable resolution. |
| `cfg_dominator_tree` | Dominator-tree computation on increasingly complex control-flow graphs. |

### negsyn-encode — `smt_encoding`

| Benchmark | What it measures |
|-----------|-----------------|
| `dy_encoding/depth_N` | Dolev-Yao adversary encoding for protocols unrolled to depth *N* (4, 8, 16). Measures formula size and constraint generation time. |
| `bitvector_constraints` | Bitvector encoding of cipher-suite selection constraints. |
| `property_encoding` | Encoding of downgrade-freedom and secrecy properties over a fixed LTS. |

## Interpreting results

Criterion generates HTML reports in `implementation/target/criterion/`. Open
`report/index.html` in a browser to see historical comparisons, violin plots,
and regression detection.

A typical run prints something like:

```
merge_cipher_suites/64  time:   [1.234 ms 1.256 ms 1.278 ms]
```

where the three values are the lower bound, estimate, and upper bound of the
mean execution time at 95% confidence.
