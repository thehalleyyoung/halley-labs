# Contributing to Spectral Decomposition Oracle

Thank you for your interest in contributing to the Spectral Decomposition Oracle!
This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)

---

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).
By participating, you are expected to uphold this code. Please report
unacceptable behavior to the project maintainers.

We are committed to providing a welcoming and inclusive experience for everyone.
We expect all participants to:

- Be respectful and considerate in all interactions
- Refrain from demeaning, discriminatory, or harassing behavior
- Be mindful of fellow participants and alert maintainers if you witness violations
- Act professionally and constructively

---

## Development Setup

### Prerequisites

- **Rust 1.75+** — install via [rustup](https://rustup.rs/)
- **Git** — for version control
- **LAPACK/BLAS** (optional) — for accelerated linear algebra benchmarks

### Getting Started

```bash
# 1. Fork the repository on GitHub

# 2. Clone your fork
git clone https://github.com/<your-username>/spectral-decomposition-oracle.git
cd spectral-decomposition-oracle/implementation

# 3. Add the upstream remote
git remote add upstream https://github.com/spectral-decomp/spectral-decomposition-oracle.git

# 4. Install Rust development tools
rustup component add rustfmt clippy

# 5. Build the project
cargo build --all

# 6. Run the test suite to verify everything works
cargo test --all
```

### Project Structure

The workspace contains 7 crates:

| Crate            | Path                       | Description                        |
|------------------|----------------------------|------------------------------------|
| `spectral-types` | `crates/spectral-types/`   | Shared type definitions            |
| `spectral-core`  | `crates/spectral-core/`    | Core spectral algorithms           |
| `matrix-decomp`  | `crates/matrix-decomp/`    | Matrix decomposition routines      |
| `oracle`         | `crates/oracle/`           | ML classification layer            |
| `optimization`   | `crates/optimization/`     | Optimization solvers               |
| `certificate`    | `crates/certificate/`      | Formal certificate generation      |
| `spectral-cli`   | `crates/spectral-cli/`     | Command-line interface             |

---

## Code Style

We use standard Rust tooling to enforce a consistent code style.

### Formatting

All code must be formatted with `rustfmt`:

```bash
# Check formatting (CI will fail if this reports issues)
cargo fmt --all -- --check

# Auto-format all code
cargo fmt --all
```

We use the default `rustfmt` configuration. Do not add a custom `rustfmt.toml`
unless discussed and approved by maintainers.

### Linting

All code must pass `clippy` with no warnings:

```bash
# Run clippy on all crates
cargo clippy --all -- -D warnings

# Run clippy including tests
cargo clippy --all --tests -- -D warnings
```

### Style Guidelines

- **Documentation**: All public items (functions, structs, enums, traits, modules)
  must have doc comments (`///` or `//!`).
- **Error handling**: Use `Result<T, SpectralError>` for fallible operations.
  Avoid `unwrap()` and `expect()` in library code; they are acceptable in tests
  and examples.
- **Naming**: Follow Rust naming conventions — `snake_case` for functions and
  variables, `CamelCase` for types, `SCREAMING_SNAKE_CASE` for constants.
- **Unsafe code**: Minimize `unsafe` blocks. Every `unsafe` block must have a
  `// SAFETY:` comment explaining why it is sound.
- **Dependencies**: Minimize external dependencies. Discuss new dependencies in
  the PR before adding them.

---

## Testing

### Running Tests

```bash
# Run all tests across the workspace
cargo test --all

# Run tests for a specific crate
cargo test -p spectral-core

# Run a specific test by name
cargo test -p certificate test_davis_kahan_certificate

# Run tests with output
cargo test --all -- --nocapture

# Run ignored (slow) tests
cargo test --all -- --ignored
```

### Writing Tests

- **Unit tests**: Place in the same file as the code being tested, inside a
  `#[cfg(test)] mod tests { ... }` block.
- **Integration tests**: Place in the `tests/` directory of the relevant crate.
- **Property-based tests**: We use `proptest` for property-based testing of
  numerical algorithms. Add property tests for any new numerical code.
- **Test naming**: Use descriptive names that indicate what is being tested:
  `test_lanczos_converges_for_symmetric_tridiagonal`.
- **Numerical tolerances**: Use appropriate tolerances (typically `1e-10` to
  `1e-12`) for floating-point comparisons. Use the `approx` crate's
  `assert_relative_eq!` macro.

### Benchmarks

Performance benchmarks use `criterion`:

```bash
# Run all benchmarks
cargo bench --all

# Run benchmarks for a specific crate
cargo bench -p matrix-decomp
```

When making performance-sensitive changes, include benchmark results in your PR.

---

## Pull Request Process

### Before Submitting

1. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** with clear, atomic commits.

3. **Write or update tests** to cover your changes.

4. **Run the full validation suite**:
   ```bash
   cargo fmt --all -- --check
   cargo clippy --all -- -D warnings
   cargo test --all
   cargo doc --all --no-deps
   ```

5. **Update documentation** if your changes affect the public API or user-facing
   behavior.

### Submitting

1. Push your branch to your fork.
2. Open a Pull Request against `main` on the upstream repository.
3. Fill out the PR template with:
   - A clear description of what the PR does
   - Motivation and context
   - Any breaking changes
   - Test plan and results
4. Request review from at least one maintainer.

### Review Process

- All PRs require at least one approving review.
- CI must pass (formatting, linting, tests, documentation).
- Maintainers may request changes — please address feedback promptly.
- Once approved, a maintainer will merge the PR.

### Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) format:

```
feat(certificate): add Weyl inequality certificate generation

Add WeylCertifier that computes eigenvalue perturbation bounds
using Weyl's inequalities. Includes unit tests and documentation.

Closes #42
```

Prefixes: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`.

---

## Reporting Issues

### Bug Reports

When filing a bug report, please include:

- **Rust version** (`rustc --version`)
- **Operating system** and version
- **Steps to reproduce** the issue
- **Expected behavior** vs. **actual behavior**
- **Minimal reproducing example** if possible
- **Relevant log output** or error messages

### Feature Requests

Feature requests are welcome. Please describe:

- The **problem** you are trying to solve
- Your **proposed solution**
- Any **alternatives** you have considered
- How the feature fits into the project's goals

---

Thank you for contributing to the Spectral Decomposition Oracle! Your efforts
help make certified mathematical optimization more accessible to everyone.
