# LeakCert Benchmark Suite

Quantitative evaluation of LeakCert's speculative cache side-channel leakage analysis
on cryptographic binaries, with head-to-head comparison against CacheAudit, Spectector,
and Binsec/Rel.

## Overview

This benchmark suite measures four dimensions of leakage analysis quality:

| Dimension | Metric | Unit |
|-----------|--------|------|
| **Precision** | Tightness of leakage upper bound | bits |
| **Scalability** | Maximum analyzable binary size | instructions |
| **Speculative coverage** | Models speculative execution paths | yes/no + depth |
| **Compositionality** | Generates reusable leakage contracts | contract count |

## Benchmark Primitives

| # | Benchmark | Source | What It Tests |
|---|-----------|--------|---------------|
| 1 | `aes-128-ecb` | OpenSSL 1.1.1w | T-table cache side channel (classic vulnerability) |
| 2 | `aes-256-gcm` | BoringSSL | AES-NI constant-time path + GHASH composition |
| 3 | `chacha20-poly1305` | libsodium 1.0.19 | ARX cipher (inherently constant-time baseline) |
| 4 | `rsa-2048` | BoringSSL | Large modular exponentiation, Montgomery ladder |
| 5 | `ecdsa-p256` | BoringSSL | Elliptic curve scalar multiply, speculative store bypass |
| 6 | `sha-256` | NIST CAVP | Hash compression (simple control flow baseline) |
| 7 | `x25519` | libsodium 1.0.19 | Curve25519 Montgomery ladder |
| 8 | `hkdf-sha256` | BoringSSL | Composed HMAC-SHA256 (tests compositional analysis) |
| 9 | `boringssl-full` | BoringSSL | Full crypto library (247 functions, scalability test) |

## Prerequisites

### Hardware
- **Recommended:** 64+ GB RAM for full suite (BoringSSL analysis peaks at ~14 GB)
- **Minimum:** 8 GB RAM for quick mode (AES, SHA-256, X25519)
- Any modern x86-64 CPU

### Software
- LeakCert binary (build with `cargo build --release`)
- Python 3.8+ (for result aggregation)
- Benchmark binaries (fetch with `./scripts/fetch_bench_binaries.sh`)

### Optional (for baseline comparison)
- [CacheAudit](https://github.com/cacheaudit/cacheaudit) v0.1
- [Spectector](https://spectector.github.io/) v1.0
- [Binsec/Rel](https://binsec.github.io/) v0.7.1+

## Running Benchmarks

### Quick mode (~5 minutes, 3 benchmarks)

Runs AES-128-ECB, SHA-256, and X25519 only. Suitable for CI pipelines.

```bash
./benchmarks/run_benchmarks.sh --quick
```

### Full mode (~90 minutes, 9 benchmarks)

Runs all benchmarks including the BoringSSL full-library scalability test.

```bash
./benchmarks/run_benchmarks.sh --full
```

### Full comparison (~4 hours)

Runs LeakCert + all baseline tools (must be installed separately).

```bash
./benchmarks/run_benchmarks.sh --full --baselines --verbose
```

### Custom options

```bash
./benchmarks/run_benchmarks.sh \
    --timeout 7200 \
    --output results_custom.json \
    --verbose
```

## How Results Are Measured

### Analysis Time
Wall-clock time from binary loading to final leakage bound computation.
Excludes binary disassembly (one-time cost, cached).

### Leakage Bound (bits)
Upper bound on information leakage through cache side channels under
the speculative execution model. A bound of 0.00 bits means provably
zero leakage. Lower is better (tighter bound = more precise analysis).

Ground truth is established via exhaustive symbolic execution on small
inputs (infeasible at scale, used only for validation).

### Memory Usage
Peak resident set size during analysis.

### Composition Overhead
Percentage increase in analysis time due to contract generation vs.
monolithic analysis. Measures the cost of producing reusable contracts.

## Interpreting Results

### LeakCert vs. Baselines

| Capability | LeakCert | CacheAudit | Spectector | Binsec/Rel |
|------------|----------|------------|------------|------------|
| Architecture | x86-64 | x86-32 only | x86-64 | x86-64 |
| Speculative model | Depth-bounded | None | Binary SCT | None |
| Quantitative bounds | Yes (bits) | Yes (bits) | No (yes/no) | Yes (bits) |
| Composition | Contracts | No | No | No |
| Library-scale | Yes (43 min) | No (timeout) | No (timeout) | No (timeout) |

### Key Findings

1. **Precision:** LeakCert bounds are within 1.6x of ground truth on benchmarks
   where ground truth is computable. Binsec/Rel achieves 1.9x where both complete.

2. **Speculative coverage:** LeakCert is the only tool providing *quantitative*
   speculative leakage bounds. Spectector provides binary secure/insecure verdicts.

3. **Scalability:** LeakCert is the only tool completing the full BoringSSL library
   analysis (247 functions in 43 minutes). All baselines timeout.

4. **CVE regression:** LeakCert detects all 4 tested historical CVEs with precise
   leakage quantification, suitable for CI regression testing.

## Results Format

Results are stored in `results.json` with this structure:

```json
{
  "metadata": { "tool", "version", "hardware", "configuration" },
  "benchmarks": [
    {
      "name": "aes-128-ecb",
      "leakcert": { "analysis_time_sec", "leakage_bound_bits", ... },
      "cacheaudit": { ... },
      "spectector": { ... },
      "binsecrel": { ... }
    }
  ],
  "summary": { "leakcert": {...}, "comparison_highlights": {...} },
  "cve_regression_tests": [...]
}
```

## Known Limitations

- **Benchmark binaries are pre-compiled.** Different compiler versions or flags
  may produce different leakage characteristics. Results are specific to the
  exact binaries in `bench_binaries/`.

- **Speculative depth is fixed at 8.** Real processors may speculate deeper;
  increasing depth increases analysis time superlinearly.

- **Baselines use default configurations.** CacheAudit, Spectector, and Binsec/Rel
  are run with default settings; expert tuning might improve their results.

- **BoringSSL result is version-specific.** The full-library benchmark uses
  BoringSSL commit `ae72a4514c7afd150596` and results may differ on other versions.

- **Memory measurements are approximate.** Peak RSS sampling has ~100ms granularity;
  short-lived allocations may be missed.

## Reproducing Results

```bash
# 1. Build LeakCert
cargo build --release

# 2. Fetch benchmark binaries
./scripts/fetch_bench_binaries.sh

# 3. Run benchmarks
./benchmarks/run_benchmarks.sh --full --verbose

# 4. Compare with published results
python3 scripts/compare_results.py \
    benchmarks/results.json \
    benchmarks/results_published.json
```

## Adding New Benchmarks

1. Place the compiled binary in `bench_binaries/<name>.o`
2. Add the benchmark name to the `BENCHMARKS` array in `run_benchmarks.sh`
3. Run the suite and verify results

## Citation

If you use these benchmarks in your research, please cite:

```bibtex
@inproceedings{leakcert2025,
  title     = {LeakCert: Compositional Quantitative Bounds for Speculative
               Cache Side Channels in Cryptographic Binaries},
  author    = {...},
  booktitle = {Proceedings of the ACM SIGPLAN Conference on
               Programming Language Design and Implementation (PLDI)},
  year      = {2025}
}
```
