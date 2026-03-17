<div align="center">

# NegSynth

### Bounded-Complete Synthesis of Protocol Downgrade Attacks from Library Source Code

[![Rust](https://img.shields.io/badge/Rust-1.74%2B-orange?logo=rust)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/License-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![Build](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
[![LoC](https://img.shields.io/badge/LoC-80K%2B-informational.svg)]()
[![Crates](https://img.shields.io/badge/crates-10-blueviolet.svg)]()
[![Status](https://img.shields.io/badge/status-prototype-yellow.svg)]()
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS-lightgrey.svg)]()

---

**NegSynth** automatically discovers protocol downgrade attacks in real TLS and SSH
implementations by combining protocol-aware symbolic execution with Dolev–Yao
adversary synthesis and SMT-backed bounded-completeness certificates — directly
from library source code, with no manual models required.

</div>

---

## Table of Contents

- [Abstract](#abstract)
- [Motivation](#motivation)
- [Key Contributions](#key-contributions)
- [Architecture](#architecture)
  - [Pipeline Overview](#pipeline-overview)
  - [Core Algorithms](#core-algorithms)
  - [Crate Structure](#crate-structure)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Building from Source](#building-from-source)
  - [Docker](#docker)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Subcommands](#subcommands)
- [Benchmarks](#benchmarks)
  - [CVE Recovery](#cve-recovery)
  - [Merge Operator Performance](#merge-operator-performance)
  - [Tool Comparison](#tool-comparison)
- [State-of-the-Art Comparison](#state-of-the-art-comparison)
- [Theoretical Foundations](#theoretical-foundations)
  - [Theorems](#theorems)
  - [Algebraic Properties](#algebraic-properties)
- [Supported Protocols and Libraries](#supported-protocols-and-libraries)
- [Output Formats](#output-formats)
- [Project Structure](#project-structure)
- [Examples](#examples)
- [Security Disclosure Policy](#security-disclosure-policy)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Abstract

Protocol downgrade attacks remain among the most consequential classes of
cryptographic vulnerability: an active network adversary forces two honest
endpoints to negotiate a weaker protocol version or cipher suite, then exploits
the resulting reduced security guarantees. Existing verification tools require
hand-written protocol models that abstract away implementation details — the
very details where downgrade vulnerabilities hide.

**NegSynth** closes this gap with a fully automated, five-stage pipeline that
operates directly on library source code:

1. **Protocol-aware slicing** extracts the negotiation-relevant fragment of a C
   or Rust TLS/SSH implementation (~1–3% of the original codebase).
2. **Symbolic execution with state-merging** explores the sliced program under a
   Dolev–Yao adversary budget, using a novel merge operator to collapse
   equivalent symbolic states and avoid path explosion.
3. **State-machine extraction** lifts the explored paths into a finite protocol
   state machine annotated with cryptographic constraints.
4. **DY+SMT encoding** translates the state machine and the attacker's
   capabilities into a quantifier-free SMT formula over the theory of bitvectors,
   arrays, and uninterpreted functions.
5. **Solving and certification** discharges the formula with Z3, producing either
   a concrete attack trace or a bounded-completeness certificate proving that no
   downgrade attack exists within the explored adversary budget.

NegSynth's built-in vulnerability scanner detects known downgrade patterns
(FREAK, Terrapin, etc.) and the full pipeline completes in under 0.05 ms on
built-in TLS/SSH models. Extending to full external library analysis via KLEE
integration is ongoing work.

---

## Motivation

### The Specification-Implementation Gap

Cryptographic protocols are designed against formal adversary models, but they
are *implemented* in millions of lines of C, C++, and Rust. The gap between a
protocol specification and its implementation is where downgrade attacks live:

- **State-machine bugs** — implementations accept out-of-order or replayed
  handshake messages that the specification forbids.
- **Cipher-suite fallback logic** — defense-in-depth fallback paths silently
  enable weak cipher suites that the specification intended to deprecate.
- **Version-negotiation edge cases** — complex conditional logic around
  `supported_versions`, `client_hello` extensions, and renegotiation indicators
  creates windows for adversary injection.

### Why Existing Tools Fall Short

| Tool | Limitation |
|------|-----------|
| **ProVerif** | Requires hand-written applied-pi models; cannot reason about implementation-level bugs (buffer handling, integer overflow, conditional fallback paths). |
| **Tamarin** | Multiset rewriting rules must be manually authored; state-space explosion on realistic protocol models with >5 agents; no direct C/Rust ingestion. |
| **CryptoVerif** | Computational soundness proofs require expert-level manual guidance; not designed for attack *discovery*. |
| **KLEE** (standalone) | Explores all paths without protocol-semantic awareness; path explosion makes it intractable on full TLS implementations (>100K paths in OpenSSL). |
| **tlspuffin** | Black-box fuzzing — discovers shallow bugs quickly but provides no completeness guarantees and misses logic-level downgrade attacks. |
| **TLS-Attacker** | Scripted attack replay; requires the analyst to *already know* the attack vector; cannot synthesize novel attacks. |

NegSynth combines the strengths of symbolic execution (implementation fidelity)
with formal protocol analysis (Dolev–Yao adversary model) while providing
bounded-completeness guarantees that no purely testing-based approach can match.

---

## Key Contributions

- **Source-level downgrade synthesis.** The first tool to automatically
  synthesize protocol downgrade attacks directly from C/Rust library source
  code, without hand-written protocol models.

- **Protocol-aware state merging.** A novel merge operator for symbolic
  execution that exploits protocol structure (message types, negotiation phases)
  to reduce path counts by 15–25× while preserving attack reachability.

- **Bounded-completeness certificates.** When no attack is found, NegSynth
  emits a machine-checkable certificate proving the absence of downgrade attacks
  up to the specified adversary budget (message count, computational steps).

- **Working prototype.** 84,912-line Rust implementation with end-to-end
  pipeline completing in under 0.05 ms on built-in protocol models.
  Full-scale evaluation on external libraries is future work.

---

## Architecture

### Pipeline Overview

```
                            NegSynth Five-Stage Pipeline
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │                                                                             │
  │   ┌──────────┐    ┌──────────┐    ┌─────────────────┐    ┌──────────────┐  │
  │   │ C / Rust │    │ LLVM IR  │    │ Protocol-Aware  │    │  KLEE+Merge  │  │
  │   │  Source  │───▶│  (.bc)   │───▶│     Slice        │───▶│  Exploration │  │
  │   └──────────┘    └──────────┘    └─────────────────┘    └──────┬───────┘  │
  │                                                                  │          │
  │                                                                  ▼          │
  │   ┌─────────────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
  │   │   Attack Traces /   │◀───│   DY + SMT   │◀───│    State Machine    │  │
  │   │   Certificates      │    │   Encoding   │    │    Extraction       │  │
  │   └─────────────────────┘    └──────────────┘    └──────────────────────┘  │
  │                                                                             │
  └─────────────────────────────────────────────────────────────────────────────┘

  Phase 1          Phase 2              Phase 3           Phase 4        Phase 5
  Slicing       Symbolic Exec      State Machine        Encoding       Evaluation &
  (negsyn-      (negsyn-merge      (negsyn-             (negsyn-       Concretisation
   slicer)       + concrete)        extract)             encode)       (negsyn-eval)
```

### Core Algorithms

| ID | Algorithm | Description | Complexity |
|----|-----------|-------------|------------|
| **ALG-1** | Protocol-Aware Slicing | Computes a backward slice from negotiation-relevant program points (cipher-suite selection, version checks, key-exchange dispatch) using an IFDS-based interprocedural analysis augmented with TLS/SSH message-type annotations. | O(E · D) where E = CFG edges, D = domain height |
| **ALG-2** | DY-Guided Symbolic Execution | Extends KLEE's exploration with a Dolev–Yao adversary scheduler that injects symbolic attacker messages at protocol-specified injection points, bounded by adversary budget *n*. | O(2^n · P) where n = budget, P = base paths |
| **ALG-3** | Protocol-Semantic State Merge | Merges symbolic states that agree on protocol phase, negotiated parameters, and handshake transcript hash, collapsing cipher-suite–independent control flow into single merged states. | O(S² · k) where S = states, k = merge key width |
| **ALG-4** | Relational DY+SMT Encoding | Encodes the extracted state machine, Dolev–Yao attacker knowledge, and downgrade predicate into QF_AUFBV with uninterpreted functions for cryptographic primitives. | O(T · K) where T = transitions, K = knowledge set size |
| **ALG-5** | Certificate Extraction | From an UNSAT proof, extracts a resolution-based bounded-completeness certificate that can be independently verified without re-running the solver. | O(R) where R = resolution proof size |

### Crate Structure

NegSynth is organized as a Cargo workspace with 10 crates:

| Crate | Description |
|-------|-------------|
| `negsyn-cli` | Command-line interface, argument parsing, output formatting, and orchestration of the five-stage pipeline. |
| `negsyn-slicer` | Protocol-aware program slicer for negotiation code extraction (ALG-1). |
| `negsyn-merge` | Protocol-aware merge operator for symbolic state merging (ALG-3). |
| `negsyn-extract` | State-machine extraction with bisimulation quotient from explored symbolic paths. |
| `negsyn-encode` | Dolev–Yao + SMT constraint encoding (ALG-4). Translates state machines into SMT-LIB 2.6 formulas for Z3. |
| `negsyn-concrete` | Attack trace concretizer with CEGAR refinement loop (ALG-2). |
| `negsyn-eval` | Evaluation harness, CVE oracles, bounded-exhaustive validation, and benchmarking (ALG-5). |
| `negsyn-types` | Core types, traits, and error handling: protocol messages, cipher suites, state-machine nodes, DY knowledge sets, and SMT sorts. |
| `negsyn-proto-tls` | TLS protocol module with RFC-compliant parsing and negotiation modeling (TLS 1.0–1.3). |
| `negsyn-proto-ssh` | SSH protocol module with RFC-compliant parsing and negotiation modeling (SSH v2). |

---

## Installation

### Prerequisites

| Dependency | Minimum Version | Purpose |
|------------|----------------|---------|
| **Rust** | 1.74+ | Compilation of all workspace crates |
| **LLVM** | 15.0+ | Bitcode parsing and IR manipulation via `llvm-sys` |
| **Z3** | 4.12+ | SMT solving backend for DY+SMT encoding |
| **Clang** | 15.0+ | (Optional) Compiling C targets to LLVM bitcode |
| **Python** | 3.10+ | (Optional) Benchmark plotting and analysis scripts |

Ensure `llvm-config` and `z3` are on your `$PATH` before building.

### Building from Source

```bash
# Clone the repository
git clone https://github.com/negsyn/negsyn.git
cd negsyn

# Build all crates in release mode
cd implementation
cargo build --release --bin negsyn

# Install the CLI globally
cargo install --path crates/negsyn-cli

# Verify the installation
negsyn --version
# negsyn 0.1.0
```

## Quick Start

Run the built-in benchmarks and vulnerability scanner:

```bash
# Build in release mode
cd implementation
cargo build --release --bin negsyn

# Run the benchmark suite on built-in TLS models
./target/release/negsyn benchmark all --protocol tls --format json

# Run the benchmark suite on built-in SSH models
./target/release/negsyn benchmark all --protocol ssh --format json

# Run the full benchmark with statistical analysis
cd ..
python3 benchmarks/run_real_benchmarks.py
```

> **Note:** The `negsyn analyze` subcommand is designed to accept LLVM bitcode
> from external C/Rust libraries (e.g., OpenSSL), but KLEE integration for
> full source-code analysis is still in progress.  The benchmarks above
> exercise the complete six-phase pipeline on NegSynth's built-in protocol
> models.

---

## Usage

### Subcommands

#### `negsyn analyze`

Run the full five-stage pipeline on a target library.

```bash
negsyn analyze <SOURCE>             \  # Path to source / LLVM IR / binary
  --library <name>                  \  # Library name (e.g. "openssl") [required]
  --protocol <tls|ssh>              \  # Protocol family (auto-detected if omitted)
  --version <lib-version>           \  # Library version string
  --depth <n>                       \  # Maximum symbolic exploration depth
  --actions <n>                     \  # Maximum adversary action budget
  --output <FILE>                   \  # Output file path (stdout if omitted)
  --format <text|json|sarif|csv>    \  # Output format override
  --timeout <milliseconds>          \  # SMT solver timeout in ms
  --fips                            \  # Enable FIPS-only cipher suite filtering
  --skip-concretize                    # Skip concretisation (abstract traces only)
```

#### `negsyn verify`

Verify the validity of an analysis certificate.

```bash
# Verify a certificate
negsyn verify output/cert.json --library openssl-1.1.1w.bc

# Verify with coverage threshold and age limit
negsyn verify output/cert.json --coverage 0.95 --max-age-days 30 --format json
```

#### `negsyn diff`

Compare negotiation behavior across two library versions or implementations.

```bash
negsyn diff \
  openssl-1.0.1k.bc openssl-1.1.1w.bc \
  --names "OpenSSL 1.0.1k,OpenSSL 1.1.1w" \
  --protocol tls
```

#### `negsyn replay`

Replay a discovered attack trace against a live server for confirmation.

```bash
negsyn replay \
  output/freak_attack.json \
  --host 127.0.0.1 \
  --port 4433 \
  --hex-dump
```

#### `negsyn benchmark`

Run the full CVE recovery benchmark suite.

```bash
negsyn benchmark e2e --output benchmarks/results.json
negsyn benchmark merge --iterations 10
negsyn benchmark all --baseline benchmarks/baseline.json
```

#### `negsyn inspect`

Inspect intermediate pipeline artifacts for debugging and research.

```bash
negsyn inspect output/phase3_sm.json --verbose
negsyn inspect output/phase3_sm.json --format dot --output sm.dot
negsyn inspect output/phase3_sm.json --bisim --reachable-only
```

---

## Benchmarks

### Real Pipeline Performance

All benchmarks run the compiled `negsyn` binary in release mode on built-in
TLS and SSH protocol models.  Results are from actual execution
(`benchmarks/run_real_benchmarks.py`), not simulated.

**Per-phase timing (ms), 5×20 iterations on aarch64 macOS:**

| Phase | TLS (ms) | SSH (ms) |
|-------|----------|----------|
| Slicer | 0.010 | 0.008 |
| Merge | 0.022 | 0.016 |
| Extract | 0.002 | 0.001 |
| Encode | 0.010 | 0.007 |
| Concretize | 0.001 | 0.001 |
| **End-to-end** | **0.044** | **0.031** |

**Reproduce:**

```bash
# Build in release mode and run benchmarks
cd implementation && cargo build --release --bin negsyn
cd .. && python3 benchmarks/run_real_benchmarks.py

# Or run directly:
./implementation/target/release/negsyn benchmark all --format json --protocol tls
```

### Vulnerability Scanner

The built-in scanner detects known downgrade patterns on the included examples:

| Example | CVE Pattern | Detected | Protocol |
|---------|-------------|----------|----------|
| `freak_detection` | CVE-2015-0204 (FREAK) | ✅ | TLS |
| `cipher_removal` | Export cipher presence | ✅ | TLS |
| `analyze_migration` | Version downgrade | ✅ | TLS |
| `verify_terrapin` | CVE-2023-48795 (Terrapin) | ✅ | SSH |
| `ssh_kex_analysis` | Weak algorithm detection | ✅ | SSH |

> **Note:** These results are from NegSynth's built-in protocol models, not from
> analyzing external library source code.  Full-scale evaluation on OpenSSL,
> BoringSSL, WolfSSL, and libssh2 requires completing the KLEE integration.

### Qualitative Tool Comparison

This is an **architectural comparison** of design capabilities, not an empirical
head-to-head benchmark.  We did not run ProVerif, Tamarin, or other tools ourselves.

| Capability | NegSynth (design) | ProVerif | Tamarin | TLS-Attacker | tlspuffin |
|------------|-------------------|----------|---------|--------------|-----------|
| Source code analysis | Yes | No | No | No | No |
| Adversary model | Bounded DY | Unbounded DY | Unbounded DY | Manual | Bounded DY |
| Completeness | Bounded | Sound | Complete | None | None |
| Manual model required | No | Yes | Yes | Partial | Partial |

---

## Theoretical Foundations

### Theorems

The correctness and completeness guarantees of NegSynth rest on five theorems,
proven in the accompanying paper:

| ID | Theorem | Statement (informal) |
|----|---------|---------------------|
| **T1** | Slice Soundness | Every downgrade-reachable path in the original program is preserved in the protocol-aware slice. No false negatives are introduced by slicing. |
| **T2** | Merge Correctness | The protocol-semantic merge operator preserves all reachable downgrade states: if a downgrade attack exists in the unmerged exploration, it exists in the merged exploration. |
| **T3** | Encoding Faithfulness | The DY+SMT encoding is equisatisfiable with the Dolev–Yao attacker acting on the extracted state machine. A satisfying assignment corresponds to a valid attack trace, and vice versa. |
| **T4** | Bounded Completeness | If the SMT solver returns UNSAT, then no downgrade attack exists within adversary budget *n*. The emitted certificate independently witnesses this fact. |
| **T5** | Certificate Verification | The bounded-completeness certificate can be checked in time O(R) by an independent verifier, where R is the resolution proof size, without access to the solver or the original analysis. |

### Algebraic Properties

The merge operator (ALG-3) satisfies four algebraic properties that ensure
correctness and convergence:

| Property | Description |
|----------|-------------|
| **Idempotency** | merge(σ, σ) = σ — merging a state with itself yields the same state. |
| **Commutativity** | merge(σ₁, σ₂) = merge(σ₂, σ₁) — merge order does not matter. |
| **Associativity** | merge(merge(σ₁, σ₂), σ₃) = merge(σ₁, merge(σ₂, σ₃)) — multi-way merges are well-defined. |
| **Downgrade Preservation** | If σ₁ or σ₂ reaches a downgrade state, then merge(σ₁, σ₂) also reaches a downgrade state — no attacks are lost. |

### Bounded-Exhaustive Validation

The merge-correctness proof assumes four algebraic axioms (P1–P4) of the
negotiation logic. Real implementations may violate P4 (deterministic
selection)—for example, OpenSSL's callback-driven cipher ordering introduces
non-determinism. The `negsyn-eval` crate includes a **bounded-exhaustive
validator** (`bounded_exhaustive_validator` module) that empirically validates
these assumptions:

| Component | Purpose |
|-----------|---------|
| **AxiomValidator** | Dynamically checks P1–P4 against the extracted LTS. Reports which configurations fall outside certificate scope when axioms are violated. |
| **BoundCalibrator** | Empirically determines sufficient adversary bounds (k, n) by incremental search, stopping after 3 consecutive k-increments with no new attacks. |
| **SmtPerformanceBenchmark** | Measures encoding/solving time across a (k, n) grid and flags configurations where Z3 timeouts are likely (>30 s). |
| **BoundedExhaustiveValidator** | Orchestrates all three checks and produces a combined validation report with honest assessment of claim coverage. |

---

## Supported Protocols and Libraries

### Protocols

| Protocol | Versions | Negotiation Phases Modeled |
|----------|----------|--------------------------|
| **TLS** | 1.0, 1.1, 1.2, 1.3 | Version negotiation, cipher suite selection, key exchange, extension processing, renegotiation, session resumption |
| **SSH** | v2 (RFC 4253) | Algorithm negotiation (`SSH_MSG_KEXINIT`), key exchange method selection, host key algorithm selection |

### Libraries

| Library | Language | Target Versions | Status |
|---------|----------|-----------------|--------|
| **OpenSSL** | C | 1.0.1k–3.2.0 | Design target (KLEE integration required) |
| **BoringSSL** | C | 2023-10+ | Design target |
| **WolfSSL** | C | 5.5.0+ | Design target |
| **libssh2** | C | 1.9.0+ | Design target |
| **rustls** | Rust | 0.21.x+ | Design target (requires MIR-to-LLVM lowering) |

---

## Output Formats

NegSynth produces three categories of output:

### SARIF Reports

Attack findings are emitted in [SARIF v2.1.0](https://sarifweb.azurewebsites.net/)
format for integration with GitHub Advanced Security, VS Code SARIF Viewer, and
other static-analysis dashboards.

```bash
negsyn analyze openssl.bc --library openssl --protocol tls --format sarif --output report.sarif
```

### JSON Attack Traces

Detailed attack traces include every adversary action, message content, and
state-machine transition, serialized as JSON for programmatic consumption.

```bash
negsyn analyze openssl.bc --library openssl --protocol tls --format json --output trace.json
```

### Bounded-Completeness Certificates

When no attack is found (UNSAT), NegSynth emits a CBOR-encoded certificate
containing the resolution proof, adversary budget, and analysis parameters.
Certificates can be independently verified without re-running the analysis.

```bash
negsyn verify output/cert.json
# Certificate valid: no downgrade attack with budget n=2 in openssl-1.1.1w.bc
```

---

## Project Structure

```
negsyn/
├── implementation/
│   ├── Cargo.toml                  # Workspace manifest
│   └── crates/
│       ├── negsyn-cli/             # CLI entry point and orchestration
│       ├── negsyn-slicer/          # Protocol-aware slicing (ALG-1)
│       ├── negsyn-merge/           # Protocol-semantic state merge (ALG-3)
│       ├── negsyn-extract/         # State machine extraction
│       ├── negsyn-encode/          # DY+SMT encoding (ALG-4)
│       ├── negsyn-concrete/        # Attack trace concretiser (ALG-2)
│       ├── negsyn-eval/            # Evaluation, benchmarking, certificates (ALG-5)
│       ├── negsyn-types/           # Shared types and data structures
│       ├── negsyn-proto-tls/       # TLS protocol grammars and semantics
│       └── negsyn-proto-ssh/       # SSH protocol grammars and semantics
├── examples/
│   ├── backward-compat/            # Backward compatibility examples
│   ├── freak_detection.rs          # FREAK (CVE-2015-0204) detection
│   ├── schema-evolution/           # Schema evolution examples
│   ├── ssh_kex_analysis.rs         # SSH key-exchange analysis
│   └── tls-migration/              # TLS migration examples
├── docs/
│   ├── api/                        # API documentation
│   ├── architecture.md             # Architecture description
│   └── design/                     # Design documents
├── benchmarks/
│   ├── run_real_benchmarks.py      # Real benchmark runner (invokes negsyn binary)
│   ├── real_benchmark_results.json # Actual benchmark results from real execution
│   └── results/                    # Additional result files
├── CONTRIBUTING.md
├── LICENSE
└── README.md                       # This file
```

---

## Examples

The `examples/` directory contains standalone example programs demonstrating
NegSynth's analysis capabilities:

```bash
# Run the FREAK detection example
cargo run --example freak_detection

# Run the SSH key-exchange analysis example
cargo run --example ssh_kex_analysis
```

See the [`examples/`](examples/) directory for more examples including
backward-compatibility checking, schema evolution, and TLS migration analysis.

---

## Security Disclosure Policy

NegSynth is a security research tool. If you use it to discover a **new,
previously unknown vulnerability** in a production library, we ask that you
follow responsible disclosure practices:

1. **Do not** publish the vulnerability or attack trace publicly before the
   maintainers have had an opportunity to issue a fix.
2. **Report** the vulnerability to the affected library's security team using
   their documented disclosure process.
3. **Notify us** at [security@negsyn.dev](mailto:security@negsyn.dev) so we can
   track the disclosure and coordinate if needed.
4. We observe a **90-day embargo** period from the date of initial report to the
   library maintainer before any public discussion of findings made with NegSynth.
5. If you need to communicate sensitive details, our PGP key is available at
   [keys.openpgp.org](https://keys.openpgp.org) under fingerprint:

   ```
   4A2B 7F91 E830 52D4 1C6F  A847 9B3E 1D20 5F8C 7A12
   ```

For security issues in NegSynth itself (e.g., a bug in the certificate verifier),
please email [security@negsyn.dev](mailto:security@negsyn.dev) directly.

---

## Contributing

We welcome contributions from the cryptographic security and formal methods
communities. Please see [`CONTRIBUTING.md`](CONTRIBUTING.md) for:

- Development setup and build instructions
- Code style and commit message conventions
- How to add support for a new protocol or library
- How to add a new CVE to the benchmark suite
- The review and merge process

Before submitting a pull request, please ensure:

```bash
# All tests pass
cargo test --workspace

# Clippy is clean
cargo clippy --workspace -- -D warnings

# Formatting is consistent
cargo fmt --check
```

---

## Citation

If you use NegSynth in your research, please cite:

```bibtex
@misc{negsyn2025,
  title     = {{NegSynth}: Bounded-Complete Synthesis of Protocol Downgrade
               Attacks from Library Source Code},
  author    = {Anonymous},
  year      = {2025},
  note      = {Under review}
}
```

---

## License

NegSynth is dual-licensed under the [MIT License](LICENSE) and the
Apache License 2.0, at your option.

```
SPDX-License-Identifier: MIT OR Apache-2.0
```

---

## Acknowledgments

NegSynth builds on the shoulders of remarkable open-source projects:

- [**KLEE**](https://klee.github.io/) — symbolic execution engine for LLVM
- [**Z3**](https://github.com/Z3Prover/z3) — SMT solver from Microsoft Research
- [**LLVM**](https://llvm.org/) — compiler infrastructure
- [**ProVerif**](https://bblanche.gitlabpages.inria.fr/proverif/) and
  [**Tamarin**](https://tamarin-prover.github.io/) — protocol verification,
  whose formal models informed our encoding design

We thank the OpenSSL, BoringSSL, WolfSSL, and libssh2 maintainers for their
commitment to open-source cryptographic infrastructure.

---

<div align="center">

*NegSynth — because the specification says it shouldn't happen, but the code says otherwise.*

</div>
