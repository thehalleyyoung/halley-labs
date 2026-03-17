# Contributing to NegSynth

Thank you for your interest in contributing to NegSynth! This project aims to advance the
state of the art in automated protocol downgrade attack synthesis, and we welcome contributions
from the security, formal methods, and systems research communities.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Architecture Overview](#architecture-overview)
- [Contributing Guidelines](#contributing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Reporting Security Vulnerabilities](#reporting-security-vulnerabilities)

## Code of Conduct

This project follows the [Rust Code of Conduct](https://www.rust-lang.org/policies/code-of-conduct).
Please be respectful and constructive in all interactions.

## Getting Started

1. **Fork the repository** and clone your fork locally.
2. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**, ensuring they follow the project's coding style.
4. **Run the test suite** to verify nothing is broken:
   ```bash
   cd implementation && cargo test
   ```
5. **Submit a pull request** with a clear description of your changes.

## Development Setup

### Prerequisites

- **Rust** 1.74+ (install via [rustup](https://rustup.rs/))
- **Z3** 4.12+ (SMT solver)
- **LLVM** 15+ (for bitcode analysis)
- **Clang** 15+ (for compiling target libraries)

### Building

```bash
cd implementation
cargo build
cargo test
cargo clippy -- -D warnings
```

### Running Checks

```bash
# Type checking
cargo check

# Linting
cargo clippy -- -D warnings

# Formatting
cargo fmt -- --check

# Full test suite
cargo test --workspace
```

## Architecture Overview

NegSynth is organized as a Rust workspace with 10 crates:

| Crate | Role | Key Types |
|-------|------|-----------|
| `negsyn-types` | Core type definitions | `SymbolicState`, `NegotiationLTS`, `SmtExpr` |
| `negsyn-slicer` | Protocol-aware program slicing | `Slicer`, `TaintTracker` |
| `negsyn-merge` | Protocol-aware merge operator | `MergeOperator`, `AlgebraicMerge` |
| `negsyn-extract` | State machine extraction | `Extractor`, `BisimulationQuotient` |
| `negsyn-encode` | DY+SMT constraint encoding | `Encoder`, `DolevYaoModel` |
| `negsyn-concrete` | CEGAR concretization | `CegarLoop`, `Concretizer` |
| `negsyn-proto-tls` | TLS protocol models | `TlsHandshake`, `CipherSuiteNegotiation` |
| `negsyn-proto-ssh` | SSH protocol models | `SshKex`, `AlgorithmNegotiation` |
| `negsyn-eval` | Evaluation harness | `CveOracle`, `Benchmark` |
| `negsyn-cli` | CLI interface | `Commands`, `OutputFormat` |

### Dependency Flow

```
negsyn-types (foundation)
    ├── negsyn-slicer
    ├── negsyn-merge
    ├── negsyn-extract
    ├── negsyn-encode
    ├── negsyn-concrete
    ├── negsyn-proto-tls
    ├── negsyn-proto-ssh
    ├── negsyn-eval (depends on most crates)
    └── negsyn-cli (depends on all crates)
```

## Contributing Guidelines

### Code Style

- Follow standard Rust conventions (`rustfmt`, `clippy`)
- All public items must have rustdoc comments
- Use `thiserror` for error types
- Prefer `Result<T, E>` over panics in library code
- Add unit tests for new functionality

### Commit Messages

Use conventional commit format:
```
feat(merge): add algebraic property detection for cipher suites
fix(slicer): handle indirect calls through SSL_METHOD vtables
docs(readme): add SSH analysis example
test(encode): add property tests for DY encoding correctness
```

### Areas for Contribution

We especially welcome contributions in these areas:

1. **Protocol modules**: Adding support for new protocols (QUIC, DTLS, WireGuard)
2. **SMT encoding**: Optimizations for constraint generation and solving
3. **Test cases**: Additional CVE test cases and regression tests
4. **Documentation**: Tutorials, examples, and API documentation
5. **Benchmarks**: Performance benchmarks and comparisons
6. **CI/CD**: Improving the build and test pipeline

## Pull Request Process

1. Ensure your changes compile (`cargo check --workspace`)
2. Run the full test suite (`cargo test --workspace`)
3. Update documentation if needed
4. Write clear commit messages
5. Reference any related issues in your PR description
6. Request review from at least one maintainer

### Review Criteria

PRs will be evaluated on:
- **Correctness**: Does the change correctly implement the intended behavior?
- **Soundness**: For formal methods code, does it preserve existing soundness guarantees?
- **Performance**: Does it maintain or improve analysis performance?
- **Testing**: Are there adequate tests for the new code?
- **Documentation**: Is the change well-documented?

## Reporting Security Vulnerabilities

If you discover a security vulnerability in NegSynth itself (as opposed to vulnerabilities
NegSynth finds in target libraries), please report it responsibly:

1. **Do not** open a public GitHub issue
2. Email the maintainers at negsyn-security@example.com
3. Include a detailed description and reproduction steps
4. We will acknowledge receipt within 48 hours

For vulnerabilities discovered by NegSynth in target libraries, follow the responsible
disclosure policy of the affected library.

## License

By contributing to NegSynth, you agree that your contributions will be licensed under
the MIT OR Apache-2.0 dual license.
