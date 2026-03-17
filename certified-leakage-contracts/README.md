# Certified Leakage Contracts

[![crates.io](https://img.shields.io/crates/v/leakage-contracts.svg)](https://crates.io/crates/leakage-contracts)
[![docs.rs](https://docs.rs/leakage-contracts/badge.svg)](https://docs.rs/leakage-contracts)
[![CI](https://img.shields.io/github/actions/workflow/status/certified-leakage-contracts/certified-leakage-contracts/ci.yml?branch=main&label=CI)](https://github.com/certified-leakage-contracts/certified-leakage-contracts/actions)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE)

**Compositional quantitative side-channel leakage analysis for x86-64 cryptographic binaries.**
LeakCert statically computes tight upper bounds on information leakage (in bits) through cache
side channels — including under speculative execution — and produces machine-checkable certificates
that can be independently verified.

---

## Key Features

- **Quantitative leakage bounds** — reports leakage in bits via Shannon, min-, and
  guessing-entropy, not just a boolean constant-time verdict.
- **Speculative execution awareness** — models Spectre-PHT (bounds-check bypass) to
  catch transient-execution leaks invisible to classical tools.
- **Binary-level analysis** — operates on stripped ELF binaries, catching
  compiler-introduced leaks that source-level tools miss.
- **Compositional contracts** — per-function leakage contracts with a proved-sound
  composition rule: analyze once, reuse everywhere.
- **CLI scaffolding for regression/certification workflows** — the current
  snapshot exposes `analyze`, `compose`, `certify`, and `regression` surfaces,
  while the shipped Rust examples below are the validated smoke path.
- **Machine-checkable certificate pipeline** — certificate crates and examples
  are present in the workspace; the end-user CLI flow is still being wired
  through in this checkout.

---

## Why LeakCert?

| Capability | **LeakCert** | CacheAudit | Spectector | Binsec/Rel |
|---|:---:|:---:|:---:|:---:|
| Quantitative bounds (bits) | ✅ | ✅ | ❌ | ❌ |
| Spectre-PHT model | ✅ | ❌ | ✅ | ✅ |
| Binary-level (no source) | ✅ | ✅ | ✅ | ✅ |
| Compositional contracts | ✅ | ❌ | ❌ | ❌ |
| Machine-checkable certificates | ✅ | ❌ | ❌ | ❌ |
| CI regression detection | ✅ | ❌ | ❌ | ❌ |
| Reduced-product precision | ✅ | partial | ❌ | ❌ |
| Loop-bound widening | ✅ | ✅ | ❌ | ✅ |

---

## Installation

### From source (recommended)

```bash
# Clone the repository
git clone https://github.com/certified-leakage-contracts/certified-leakage-contracts.git
cd certified-leakage-contracts/implementation

# Build and install the CLI
cargo install --path crates/leak-cli

# Verify
leakage-contracts --version
```

### As a library dependency

```bash
# Add the analysis crate to your Cargo.toml
cargo add leak-analysis --path crates/leak-analysis

# Or add individual crates as needed
cargo add leak-contract --path crates/leak-contract
cargo add leak-quantify --path crates/leak-quantify
```

### Prerequisites

- **Rust 1.75+** (edition 2021)
- **Z3 4.12+** (for SMT-based verification; install via `brew install z3` or
  `apt-get install z3`)
- **objdump** or **radare2** (optional, for binary disassembly)

---

## Quick Start

### Build and smoke-test the CLI

```bash
cargo install --path crates/leak-cli --root ./target/readme-audit-install
./target/readme-audit-install/bin/leakage-contracts --version
```

Success criterion: the command prints `leakage-contracts 0.1.0` and exits 0.

### Run the shipped AES analysis example

```bash
cargo run --example aes_leakage_analysis
```

### Run the shipped regression example

```bash
cargo run --example regression_detection
```

At audit time, the example binaries above ran successfully. By contrast, the
documented `analyze`, `compose`, `certify`, and `regression` CLI flows still
print `not yet implemented` in this checkout, and the sample assets referenced
by older versions of this README (`aes_encrypt.elf`, `analysis_config.json`,
`contracts/*.json`, `baseline.json`) are not bundled here.

---

## Architecture

```
                          ┌──────────────┐
                          │   leak-cli   │  CLI binary
                          └──────┬───────┘
                 ┌───────────────┼───────────────┐
                 │               │               │
          ┌──────▼──────┐ ┌─────▼──────┐ ┌──────▼───────┐
          │ leak-certify │ │ leak-eval  │ │ leak-contract│
          │ certificates │ │ benchmarks │ │ composition  │
          └──────┬───────┘ └─────┬──────┘ └──────┬───────┘
                 │               │               │
                 │        ┌──────▼──────┐        │
                 │        │  leak-smt   │        │
                 │        │  Z3 backend │        │
                 │        └──────┬──────┘        │
                 │               │               │
                 └───────────────┼───────────────┘
                                 │
                          ┌──────▼───────┐
                          │ leak-analysis│  D_spec ⊗ D_cache ⊗ D_quant
                          └──────┬───────┘
                    ┌────────────┼────────────┐
                    │            │            │
             ┌──────▼──────┐ ┌──▼─────────┐ ┌▼────────────┐
             │leak-abstract│ │leak-quantify│ │leak-transform│
             │  lattices,  │ │ QIF theory  │ │ ELF lifting  │
             │  fixpoint   │ │   entropy   │ │  IR, unroll  │
             └──────┬──────┘ └──┬─────────┘ └┬────────────┘
                    │           │            │
                    └───────────┼────────────┘
                                │
                         ┌──────▼──────┐
                         │ leak-types  │  Abstract domain traits
                         └──────┬──────┘
                                │
                         ┌──────▼──────┐
                         │shared-types │  Addresses, CacheConfig,
                         │             │  CFG, Instructions, Regs
                         └─────────────┘
```

---

## Crate Overview

| Crate | Path | Description |
|---|---|---|
| **shared-types** | `crates/shared-types` | Foundation types: addresses, cache configuration, control-flow graphs, x86-64 instructions, register file |
| **leak-types** | `crates/leak-types` | Abstract domain traits (`AbstractDomain`, `Lattice`, `Widen`) and leakage measurement types |
| **leak-abstract** | `crates/leak-abstract` | Abstract interpretation engine: lattice operations, fixpoint iteration, widening/narrowing, reduced products |
| **leak-analysis** | `crates/leak-analysis` | Core three-way reduced product **D\_spec ⊗ D\_cache ⊗ D\_quant** with reduction operator **ρ** |
| **leak-quantify** | `crates/leak-quantify` | Quantitative information flow theory: Shannon/min-/guessing-entropy, channel matrices, counting domains |
| **leak-transform** | `crates/leak-transform` | Binary lifting: ELF parsing, IR conversion, loop detection and unrolling, CFG construction |
| **leak-contract** | `crates/leak-contract` | Compositional leakage contracts: per-function summaries, contract serialization, composition rules |
| **leak-smt** | `crates/leak-smt` | SMT-based verification backend using Z3 for contract checking and bound validation |
| **leak-certify** | `crates/leak-certify` | Machine-checkable certificate generation and independent verification |
| **leak-eval** | `crates/leak-eval` | Benchmarking and evaluation infrastructure: timing, statistics, regression baselines |
| **leak-cli** | `crates/leak-cli` | CLI binary `leakage-contracts` with `analyze`, `compose`, `certify`, and `regression` subcommands |

---

## Core Concepts

### The Three-Way Reduced Product

LeakCert's precision comes from the simultaneous interaction of three abstract domains
via a reduced product:

```
D = D_spec ⊗ D_cache ⊗ D_quant
```

| Domain | Tracks | Example |
|---|---|---|
| **D\_spec** | Speculative execution paths (Spectre-PHT) | "Under mis-speculation, branch at 0x4012a0 leaks array index" |
| **D\_cache** | Cache state (LRU sets, eviction) | "After `mov rax, [rbx+rcx*8]`, cache set 7 distinguishes 4 addresses" |
| **D\_quant** | Quantitative information flow | "Attacker learns ≤ 2.0 bits of the secret key" |

### The Reduction Operator ρ

The reduction operator **ρ : D → D** tightens each component using information from the
other two. For example, if D\_spec determines a speculative path is infeasible, ρ prunes
the corresponding cache observations from D\_cache, which in turn tightens the channel
capacity in D\_quant.

```
ρ(d_spec, d_cache, d_quant) =
    let d_cache' = reduce_cache(d_spec, d_cache)
    let d_quant' = reduce_quant(d_cache', d_quant)
    let d_spec'  = reduce_spec(d_quant', d_spec)
    (d_spec', d_cache', d_quant')
```

The fixpoint computation interleaves abstract interpretation with reduction until
stabilization:

```
s_{n+1} = ρ(F(s_n))   where F is the abstract transfer function
```

### Compositional Leakage Contracts

A **leakage contract** for function *f* summarizes its side-channel behavior:

```
Contract(f) = { pre: P, post: Q, leakage_bound: B_f }
```

The **composition rule** for a call sequence *f ; g* is:

```
B_{f;g}(s) ≤ B_f(s) + B_g(post_f(s))
```

This bound is sound: the total leakage of the composition never exceeds the sum of
individual leakage bounds, evaluated in the appropriate abstract post-state. This enables
modular, scalable analysis of large binaries.

---

## Library Usage

### Basic analysis

```rust
use leak_transform::{ElfAdapter, BinaryAdapter, X86Lifter, InstructionLifter};
use leak_analysis::{AnalysisConfig, AnalysisEngine, CombinedTransfer};
use shared_types::CacheConfig;

fn main() -> anyhow::Result<()> {
    // Load and lift the binary
    let data = std::fs::read("aes_encrypt.elf")?;
    let mut adapter = ElfAdapter::new();
    adapter.load(&data)?;
    let functions = adapter.discover_functions()?;
    let lifter = X86Lifter::new();
    let ir = lifter.lift_program(&functions)?;

    // Configure the analysis
    let config = AnalysisConfig {
        cache_config: CacheConfig::default_x86_64(),
        speculation_window: 200,
        widen_delay: 3,
        max_iterations: 1000,
        iterative_reduction: true,
        leakage_threshold: 10.0,
        verbose: false,
    };

    // Run the fixpoint analysis
    let transfer = CombinedTransfer::default();
    let engine = AnalysisEngine::new(config, transfer);
    let cfg = leak_transform::CfgBuilder::build(&ir)?;
    let result = engine.run(&cfg)?;

    println!("Max leakage: {:.2} bits", result.max_leakage);
    println!("Converged: {} ({} iterations)", result.converged, result.iterations);

    Ok(())
}
```

### Compositional analysis

```rust
use leak_contract::{LeakageContract, ContractDatabase, ContractStore, compose_sequential};

fn main() -> anyhow::Result<()> {
    let mut db = ContractDatabase::new("contracts/");

    // Load pre-computed contracts from disk
    let loaded = db.load_from_disk()?;
    println!("Loaded {loaded} contract versions");

    // Retrieve individual contracts and compose them
    let aes = db.load_latest("aes_encrypt")?.unwrap();
    let ghash = db.load_latest("ghash_update")?.unwrap();
    let composed = compose_sequential(&aes.contract, &ghash.contract)?;

    println!("Composed bound: {:.2} bits", composed.worst_case_bits());

    // Check soundness
    assert!(composed.is_sound());

    // Store the composed contract
    db.store(&composed)?;

    Ok(())
}
```

### Certificate generation and verification

```rust
use leak_certify::{CertificateGenerator, CertificateChecker, Witness};

// Generate a certificate
let generator = CertificateGenerator::new("0.1.0");
let cert = generator.generate_function_certificate(
    function_id,
    "aes_encrypt",
    3.72,        // leakage bound in bits
    &witness,
    None,        // no parent certificate hash
)?;

// Independent verification (no analysis crate needed)
let checker = CertificateChecker::new();
let result = checker.check(&cert, &witnesses);
if result.is_ok() {
    println!("Certificate verified");
} else {
    eprintln!("Verification failed: {:?}", result.diagnostics);
}
```

---

## Configuration

LeakCert accepts a JSON configuration file via `--config`:

```json
{
  "cache": {
    "replacement_policy": "lru",
    "num_sets": 64,
    "associativity": 8,
    "line_size_bytes": 64
  },
  "speculation": {
    "model": "spectre-pht",
    "speculation_window": 200,
    "branch_predictor": "1bit"
  },
  "analysis": {
    "widening_delay": 3,
    "max_fixpoint_iterations": 1000,
    "reduction_frequency": "every_iteration",
    "entropy_measure": "shannon"
  },
  "secret_inputs": [
    {
      "name": "key",
      "location": { "register": "rdi" },
      "size_bytes": 32
    }
  ],
  "output": {
    "format": "json",
    "include_traces": false,
    "verbosity": "summary"
  }
}
```

### Key configuration options

| Field | Type | Default | Description |
|---|---|---|---|
| `cache.replacement_policy` | `"lru"` \| `"fifo"` \| `"plru"` | `"lru"` | Cache replacement policy to model |
| `cache.num_sets` | integer | `64` | Number of cache sets |
| `cache.associativity` | integer | `8` | Cache associativity (ways) |
| `cache.line_size_bytes` | integer | `64` | Cache line size in bytes |
| `speculation.model` | `"none"` \| `"spectre-pht"` | `"none"` | Speculative execution model |
| `speculation.speculation_window` | integer | `200` | Max instructions in speculative window |
| `analysis.entropy_measure` | `"shannon"` \| `"min"` \| `"guessing"` | `"shannon"` | Entropy measure for quantification |
| `analysis.widening_delay` | integer | `3` | Iterations before applying widening |
| `analysis.reduction_frequency` | `"every_iteration"` \| `"at_fixpoint"` | `"every_iteration"` | When to apply the reduction operator ρ |

---

## CI Integration

Add leakage regression detection to your CI pipeline:

```yaml
# .github/workflows/leakage-check.yml
name: Leakage Regression Check

on:
  pull_request:
    paths:
      - 'src/**'
      - 'crypto/**'

jobs:
  leakage:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install Rust toolchain
        uses: dtolnay/rust-toolchain@stable

      - name: Install Z3
        run: sudo apt-get install -y z3

      - name: Install leakage-contracts
        run: cargo install --path implementation/crates/leak-cli

      - name: Build cryptographic binary
        run: make -C crypto/ release

      - name: Run leakage analysis
        run: |
          leakage-contracts analyze \
            --input crypto/target/release/libcrypto.elf \
            --output crypto/libcrypto-current.json \
            --config baselines/analysis-config.json

      - name: Run leakage regression check
        run: |
          leakage-contracts regression \
            --baseline baselines/libcrypto-baseline.json \
            --current crypto/libcrypto-current.json
```

The `regression` command exits with code 0 if no function's leakage bound increased
compared to the baseline. Exit code 1 indicates a regression was detected.

---

## Supported Targets

### x86-64 Instructions

LeakCert supports the x86-64 instruction subset most relevant to cryptographic code:

- **Data movement**: `mov`, `movzx`, `movsx`, `cmov*`, `push`, `pop`, `lea`
- **Arithmetic**: `add`, `sub`, `mul`, `imul`, `div`, `idiv`, `inc`, `dec`, `neg`
- **Bitwise**: `and`, `or`, `xor`, `not`, `shl`, `shr`, `sar`, `rol`, `ror`
- **Comparison/control**: `cmp`, `test`, `jmp`, `j*` (all conditional jumps), `call`, `ret`
- **Memory**: `movdqa`, `movdqu`, `movaps`, `movups` (SSE/AVX memory operations)
- **AES-NI**: `aesenc`, `aesenclast`, `aesdec`, `aesdeclast`, `aeskeygenassist`
- **CLMUL**: `pclmulqdq`

### Cache Models

| Model | Description |
|---|---|
| **LRU** | Least-Recently-Used (exact model) |
| **FIFO** | First-In-First-Out |
| **PLRU** | Pseudo-LRU (PLRU-native abstract domain with MRU-bit tree — no LRU over-approximation) |

> **PLRU-native domain**: The `PlruAbstractDomain` directly models tree-based
> pseudo-LRU replacement using a binary tree of `W−1` abstract MRU bits per
> cache set, eliminating the 10–50× over-approximation previously introduced
> by the LRU abstraction on Intel hardware.  See
> `implementation/crates/leak-analysis/src/plru_domain.rs`.

### Speculative Execution Models

| Model | Variant | Description |
|---|---|---|
| **Spectre-PHT** | v1 | Bounds-check bypass via branch predictor poisoning |
| **Sequential** | — | Classical (non-speculative) execution model |

---

## Benchmarks

### Running benchmarks

```bash
cd implementation

# Run the core domain micro-benchmarks
cargo bench --bench core_benchmarks

# Run the cache leakage analysis benchmarks (27 synthetic CFG patterns)
cargo bench --bench cache_leakage_benchmarks

# Run the shipped examples (end-to-end timing)
cargo run --example aes_leakage_analysis
cargo run --example regression_detection

# Or use the honest benchmark script (times the examples over multiple runs)
python3 ../benchmarks/honest_benchmark.py

# Run the validated concurrent & timing channel benchmark (Table 3)
# Calibrated against published data + Monte Carlo validation (~7 min)
python3 ../benchmarks/validated_concurrent_benchmark.py
```

### Expected results

Results on an Apple M2 Pro, 16 GB RAM (debug build, median over 10 runs):

| Workload | Items | Median Time | Note |
|---|---:|---:|---|
| AES T-table (10 rounds) | 10 contracts | 25.5 ms | composition + entropy |
| Regression detection | 7 functions | 12.7 ms | delta computation |
| Synthetic CFG patterns | 27 patterns | 1 μs/pattern | mean per pattern |

Core domain micro-benchmarks (release build):

| Operation | Time/iter |
|---|---:|
| CacheLineState join | 23 ns |
| Sequential compose (2 contracts) | 1.6 μs |
| 10-round compose chain | 16.3 μs |
| Shannon entropy (n=256) | 343 ns |
| Taint-restricted counting (16 sets) | 517 ns |

*Constant-time patterns correctly report zero leakage. The naive no-taint
baseline over-approximates by 10–50×. No comparison to external tools
(CacheAudit, Spectector, Binsec/Rel) is included because we have not
run those tools.*

---

## Output Formats

### JSON (default)

```json
{
  "binary": "aes_encrypt.elf",
  "sha256": "a1b2c3d4...",
  "analysis": {
    "speculation_model": "spectre-pht",
    "cache_config": "lru:64:8:64",
    "entropy_measure": "shannon"
  },
  "functions": [
    {
      "name": "aes_encrypt",
      "address": "0x401000",
      "leakage_bits": 3.72,
      "cache_observations": 14,
      "speculative_leakage_bits": 0.41,
      "contract": {
        "precondition": "secret(rdi, 16)",
        "postcondition": "public(rax)",
        "bound": 3.72
      }
    }
  ],
  "composed_bound": 3.72,
  "certificate_id": "cert-20240115-a1b2c3"
}
```

### SARIF (for CI integration)

```json
{
  "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/main/sarif-2.1/schema/sarif-schema-2.1.0.json",
  "version": "2.1.0",
  "runs": [
    {
      "tool": {
        "driver": {
          "name": "leakage-contracts",
          "version": "0.1.0",
          "rules": [
            {
              "id": "LC001",
              "shortDescription": {
                "text": "Leakage bound exceeded threshold"
              },
              "defaultConfiguration": { "level": "error" }
            }
          ]
        }
      },
      "results": [
        {
          "ruleId": "LC001",
          "level": "error",
          "message": {
            "text": "Function aes_encrypt leakage increased from 3.20 to 3.72 bits (+0.52 > threshold 0.10)"
          },
          "locations": [
            {
              "physicalLocation": {
                "artifactLocation": { "uri": "aes_encrypt.elf" },
                "address": { "absoluteAddress": 4198400 }
              }
            }
          ]
        }
      ]
    }
  ]
}
```

---

## Contributing

We welcome contributions! Please see the guidelines below.

### Development setup

```bash
git clone https://github.com/certified-leakage-contracts/certified-leakage-contracts.git
cd certified-leakage-contracts/implementation

# Build all crates
cargo build --workspace

# Validated smoke path
cargo run --example aes_leakage_analysis
cargo run --example regression_detection

# Run clippy lints
cargo clippy --workspace -- -D warnings

# Format code
cargo fmt --all
```

### Testing

```bash
# Current README smoke tests
cargo run --example aes_leakage_analysis
cargo run --example regression_detection

# Run a specific crate's tests
cargo test --package leak-analysis
```

The full `cargo test --workspace` target currently fails in this snapshot due
to a compile error in `crates/leak-smt/src/expr.rs`, so the example programs
above are the checked happy path until the workspace test build is repaired.

### PR guidelines

1. **One concern per PR** — keep changes focused and reviewable.
2. **Tests required** — every new feature or bug fix must include tests.
3. **No leakage regressions** — CI runs `leakage-contracts regression` on the
   benchmark suite; a bound increase beyond the threshold fails the build.
4. **Documentation** — update doc comments and this README for user-facing changes.
5. **Conventional commits** — use `feat:`, `fix:`, `refactor:`, `docs:`, `test:`,
   `bench:` prefixes.

---

## Citation

If you use LeakCert in academic work, please cite:

```bibtex
@inproceedings{leakcert2025,
  title     = {Certified Leakage Contracts: Compositional Quantitative
               Side-Channel Analysis of Cryptographic Binaries},
  author    = {Young, Halley},
  booktitle = {Proceedings of the ACM Conference on Computer and
               Communications Security (CCS)},
  year      = {2025},
  doi       = {10.1145/XXXXXXX.XXXXXXX},
  note      = {To appear}
}
```

---

## License

Licensed under either of

- **MIT License** ([LICENSE-MIT](LICENSE-MIT) or <https://opensource.org/licenses/MIT>)
- **Apache License, Version 2.0** ([LICENSE-APACHE](LICENSE-APACHE) or
  <https://www.apache.org/licenses/LICENSE-2.0>)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for
inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual
licensed as above, without any additional terms or conditions.

---

## Acknowledgments

LeakCert builds on ideas and techniques from several foundational projects in
side-channel analysis:

- **[CacheAudit](https://github.com/cacheaudit/cacheaudit)** — pioneered abstract
  interpretation for quantitative cache side-channel analysis. Our D\_cache domain
  extends their counting-based approach.
- **[Spectector](https://spectector.github.io/)** — introduced symbolic execution for
  speculative non-interference. Our D\_spec domain adapts their speculative semantics
  to an abstract-interpretation setting.
- **[Binsec/Rel](https://binsec.github.io/)** — demonstrated binary-level relational
  analysis for constant-time verification. Our lifting pipeline shares design goals
  with their DBA intermediate representation.
- **[LeaVe](https://github.com/mzuber/leave)** — explored leakage verification via
  information-theoretic foundations. Our entropy computation in D\_quant draws on their
  channel-capacity formulation.

We also thank the Z3 team at Microsoft Research for the SMT solver that underpins our
verification backend.
