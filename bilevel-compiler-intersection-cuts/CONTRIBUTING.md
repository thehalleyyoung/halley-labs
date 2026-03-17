# Contributing to BiCut

Thank you for your interest in contributing to BiCut! This document provides
guidelines and information for contributors.

## Getting Started

### Prerequisites

- **Rust** ≥ 1.70 (install via [rustup](https://rustup.rs/))
- **Git** for version control
- Familiarity with bilevel optimization concepts is helpful but not required

### Development Setup

```bash
# Clone the repository
git clone https://github.com/bicut-project/bicut.git
cd bicut/implementation

# Build the project
cargo build

# Run all tests
cargo test

# Run clippy lints
cargo clippy -- -D warnings

# Format code
cargo fmt
```

## How to Contribute

### Reporting Issues

- Use GitHub Issues for bug reports and feature requests
- Include a minimal reproducible example when reporting bugs
- Specify your Rust version, OS, and solver versions

### Pull Requests

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes with clear commit messages
4. Ensure all tests pass: `cargo test`
5. Ensure no clippy warnings: `cargo clippy -- -D warnings`
6. Ensure code is formatted: `cargo fmt --check`
7. Open a pull request against `main`

### Code Style

- Follow standard Rust conventions (rustfmt defaults)
- Use `///` doc comments for all public items
- Include `#[cfg(test)]` modules for unit tests
- Error types should use `thiserror` derive macros
- Prefer `anyhow::Result` for application-level errors

### Commit Messages

Use conventional commits:
```
feat: add CPLEX backend emission
fix: correct big-M computation for degenerate LPs
docs: update CLI reference for cut-loop command
test: add roundtrip verification for KKT pass
refactor: extract common emission logic
```

## Architecture Overview

BiCut is a Rust workspace with nine crates:

| Crate | Purpose |
|-------|---------|
| `bicut-types` | Core types: IR, problem definitions, certificates |
| `bicut-core` | Structural analysis, CQ verification, reformulation selection |
| `bicut-lp` | LP solver (simplex, interior point, tableau) |
| `bicut-cuts` | Intersection cuts, separation oracle, cut pool |
| `bicut-value-function` | Parametric LP, value function oracle |
| `bicut-compiler` | Reformulation passes, solver emission |
| `bicut-branch-cut` | Branch-and-cut solver framework |
| `bicut-bench` | BOBILib benchmark harness |
| `bicut-cli` | Command-line interface |

### Adding a New Cut Family

1. Create a new module in `bicut-cuts/src/`
2. Implement the cut generation logic
3. Register with the cut manager in `bicut-cuts/src/manager.rs`
4. Add tests with known bilevel instances
5. Update the CLI to expose the new cut type

### Adding a New Solver Backend

1. Create `backend_<solver>.rs` in `bicut-compiler/src/`
2. Implement MPS/LP emission with solver-specific optimizations
3. Add the backend to `BackendTarget` enum
4. Add integration tests with the solver

## Testing

```bash
# Run all tests
cargo test

# Run tests for a specific crate
cargo test -p bicut-cuts

# Run a specific test
cargo test -p bicut-cuts test_intersection_cut

# Run with output
cargo test -- --nocapture
```

## Questions?

Open a GitHub Discussion or reach out to the maintainers.

## License

By contributing, you agree that your contributions will be licensed under the
MIT / Apache-2.0 dual license.
