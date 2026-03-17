# Contributing to GuardPharma

Thank you for your interest in contributing to GuardPharma! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Areas of Interest](#areas-of-interest)

## Code of Conduct

This project adheres to a code of conduct adapted from the [Contributor Covenant v2.1](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## Getting Started

1. **Fork** the repository on GitHub.
2. **Clone** your fork locally:
   ```bash
   git clone https://github.com/<your-username>/guardpharma.git
   cd guardpharma/implementation
   ```
3. **Build** the project:
   ```bash
   cargo build
   ```
4. **Run tests**:
   ```bash
   cargo test
   ```

## Development Setup

### Prerequisites

- **Rust** 1.75 or later (install via [rustup](https://rustup.rs/))
- **Z3** 4.12+ (optional, for SMT-based bounded model checking)
  - macOS: `brew install z3`
  - Ubuntu: `apt install libz3-dev`
- **pdflatex** (optional, for building the tool paper)

### Workspace Structure

GuardPharma is organized as a Cargo workspace with the following crates:

| Crate | Purpose |
|-------|---------|
| `types` | Shared domain types (drugs, enzymes, concentrations) |
| `pk-model` | Pharmacokinetic ODE models and CYP interaction dynamics |
| `clinical` | Clinical state space, patient profiles, lab values |
| `abstract-interp` | Tier 1: PK-aware abstract interpretation engine |
| `smt-encoder` | SMT-LIB2 encoding for PTA verification |
| `model-checker` | Tier 2: Contract-based compositional model checking |
| `guideline-parser` | Guideline document parsing and PTA compilation |
| `conflict-detect` | Two-tier pipeline orchestration and conflict analysis |
| `significance` | Clinical significance filtering (Beers, FAERS, DrugBank) |
| `recommendation` | Schedule optimization and alternative suggestions |
| `evaluation` | Benchmark harness and evaluation metrics |
| `cli` | Command-line interface |

## Making Changes

1. Create a **feature branch** from `main`:
   ```bash
   git checkout -b feature/my-improvement
   ```
2. Make your changes, ensuring each commit is **atomic** and has a clear message.
3. Add or update **tests** for your changes.
4. Run the full test suite:
   ```bash
   cargo test --workspace
   ```
5. Run `cargo clippy` and address any warnings:
   ```bash
   cargo clippy --workspace -- -D warnings
   ```
6. Format your code:
   ```bash
   cargo fmt --all
   ```

## Pull Request Process

1. Update the **CHANGELOG.md** with a note about your changes.
2. Ensure your PR description clearly explains:
   - **What** the change does
   - **Why** it is needed
   - **How** it was tested
3. Link to any relevant issues.
4. A maintainer will review your PR. Please respond to feedback promptly.

## Coding Standards

- Follow [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/).
- All public items must have **rustdoc comments** with examples where appropriate.
- Use `thiserror` for error types; avoid `unwrap()` in library code.
- Prefer strong typing over stringly-typed APIs (e.g., `DrugId` over `String`).
- Clinical domain constants (thresholds, PK parameters) must cite their source.

## Testing

### Unit Tests

Each module should have a `#[cfg(test)]` section with unit tests. Focus on:

- **Edge cases** in PK calculations (zero concentrations, boundary PK parameters)
- **Roundtrip** tests for serialization/deserialization
- **Invariant checks** for abstract domain operations (join commutativity, widening soundness)

### Integration Tests

Integration tests in `tests/` exercise the full pipeline:

```bash
cargo test --test integration
```

### Benchmark Tests

Performance benchmarks use `criterion`:

```bash
cargo bench
```

## Documentation

- Update rustdoc when changing public APIs.
- Update `README.md` for user-visible changes.
- Add entries to `docs/` for new features or architectural decisions.

## Areas of Interest

We especially welcome contributions in:

- **Pharmacokinetic models**: Adding population PK parameters for underrepresented drugs
- **Guideline encoding**: Translating clinical guidelines into GuardPharma's JSON format
- **SMT encoding**: Optimizing the Z3 encoding for faster verification
- **Clinical validation**: Comparing GuardPharma output against pharmacist expert review
- **Performance**: Improving abstract interpretation convergence speed
- **Visualization**: PK trajectory plotting and counterexample visualization

## Questions?

Open a [GitHub Discussion](https://github.com/guardpharma/guardpharma/discussions) or file an issue. We're happy to help!
