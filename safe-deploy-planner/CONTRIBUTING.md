# Contributing to SafeStep

Thank you for your interest in contributing to SafeStep! This document provides
guidelines and information for contributors.

## Table of Contents

- [Reporting Bugs](#reporting-bugs)
- [Requesting Features](#requesting-features)
- [Development Setup](#development-setup)
- [Building from Source](#building-from-source)
- [Running Tests](#running-tests)
- [Code Style](#code-style)
- [Pull Request Process](#pull-request-process)
- [Code of Conduct](#code-of-conduct)

## Reporting Bugs

Before filing a bug report, please check the existing issues to avoid
duplicates. When filing a bug, include:

1. **SafeStep version** (`safestep --version`)
2. **Operating system and architecture**
3. **Rust toolchain version** (`rustc --version`)
4. **Minimal reproduction case** — the smallest input configuration that
   triggers the bug
5. **Expected behaviour** vs. **actual behaviour**
6. **Full error output** (use `RUST_LOG=debug safestep ...` for verbose logs)

Open a new issue with the **Bug Report** template and fill in every section.

## Requesting Features

Feature requests are welcome. Please open an issue with the **Feature Request**
template and describe:

1. The problem the feature would solve
2. Your proposed solution (if any)
3. Alternatives you have considered
4. Whether you are willing to implement it yourself

We will triage feature requests and label them accordingly.

## Development Setup

### Prerequisites

| Tool      | Minimum Version | Notes                              |
|-----------|-----------------|------------------------------------|
| Rust      | 1.75+           | Install via [rustup](https://rustup.rs) |
| CaDiCaL   | 1.9+            | SAT solver backend                 |
| Z3        | 4.12+           | SMT solver for envelope analysis   |
| Protobuf  | 3.x             | Only for schema-oracle protobuf support |

### Clone and configure

```bash
git clone https://github.com/your-org/safe-deploy-planner.git
cd safe-deploy-planner

# Install Rust nightly for certain dev tools (stable is used for building)
rustup toolchain install nightly --component rustfmt clippy

# Verify solver availability
z3 --version
cadical --version
```

## Building from Source

```bash
# Debug build (fast compile, slow runtime)
cargo build --workspace

# Release build (optimised)
cargo build --workspace --release

# The CLI binary is at:
#   target/release/safestep
```

If CaDiCaL or Z3 are installed in non-standard paths, set the appropriate
environment variables:

```bash
export CADICAL_LIB_DIR=/opt/cadical/lib
export Z3_SYS_Z3_HEADER=/opt/z3/include/z3.h
```

## Running Tests

```bash
# Run all tests
cargo test --workspace

# Run tests for a specific crate
cargo test -p safestep-bmc

# Run integration tests only
cargo test --workspace --test '*'

# Run benchmarks
cargo bench --workspace
```

All tests must pass before a pull request will be reviewed.

## Code Style

We enforce consistent style through automated tooling:

- **rustfmt** — format all code before committing:
  ```bash
  cargo +nightly fmt --all
  ```
- **clippy** — lint for common mistakes:
  ```bash
  cargo clippy --workspace --all-targets -- -D warnings
  ```

Additional conventions:

- Prefer `thiserror` for library error types and `anyhow` in the CLI binary.
- Every public function and type must have a doc comment (`///`).
- Use `#[must_use]` on functions that return values that should not be ignored.
- Keep individual functions under ~60 lines where practical.

## Pull Request Process

1. **Fork** the repository and create a feature branch from `main`.
2. Make your changes in small, focused commits with clear messages.
3. Ensure `cargo test --workspace`, `cargo clippy ...`, and `cargo fmt --check`
   all pass.
4. Add or update tests that cover your changes.
5. Update documentation (including `CHANGELOG.md` under `[Unreleased]`).
6. Open a pull request against `main` with a clear description of:
   - What the PR does
   - Why the change is needed
   - Any breaking changes
7. At least one maintainer approval is required before merging.
8. Squash-merge is preferred for single-feature PRs.

### Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/)
specification:

```
feat(bmc): add k-robustness checking to planner
fix(schema): handle missing OpenAPI version field
docs: update architecture diagram
```

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).
By participating, you agree to uphold this code. Please report unacceptable
behaviour to the project maintainers.

---

Thank you for helping make SafeStep better!
