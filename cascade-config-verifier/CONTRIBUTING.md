# Contributing to CascadeVerify

Thank you for your interest in contributing to CascadeVerify! This document
provides guidelines and information for contributors.

## Code of Conduct

This project follows the [Contributor Covenant](https://www.contributor-covenant.org/)
code of conduct. Please be respectful and constructive in all interactions.

## Getting Started

### Prerequisites

- **Rust 1.75+** with `cargo`
- **Git** for version control
- Optional: `pdflatex` for building the paper

### Development Setup

```bash
git clone https://github.com/cascade-verify/cascade-verify.git
cd cascade-verify/implementation

# Build the workspace
cargo build

# Run all tests
cargo test

# Run clippy lints
cargo clippy -- -D warnings

# Format code
cargo fmt --check
```

## How to Contribute

### Reporting Bugs

- Use the GitHub issue tracker
- Include: Rust version, OS, configuration files (anonymised), error output
- Label with `bug`

### Suggesting Features

- Open a GitHub issue with the `enhancement` label
- Describe the use case, not just the solution
- Reference relevant post-mortems or documentation if applicable

### Submitting Code

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes (see style guide below)
4. Add tests for new functionality
5. Run `cargo test && cargo clippy -- -D warnings && cargo fmt --check`
6. Commit with a descriptive message
7. Open a pull request against `main`

## Code Style

### Rust

- Follow standard `rustfmt` formatting (run `cargo fmt`)
- Use `thiserror` for library error types
- Use `anyhow` for application-level errors (CLI only)
- Prefer builder patterns for types with many fields
- Document all public APIs with `///` rustdoc comments
- Use `#[must_use]` on functions that return values callers should not ignore

### Testing

- **Unit tests:** In the same file, within `#[cfg(test)] mod tests { ... }`
- **Integration tests:** In `tests/` directories within each crate
- **Property tests:** Use `proptest` for invariant checking
- Aim for >90% line coverage on new code

### Commit Messages

- Use imperative mood: "Add timeout chain detection" not "Added..."
- First line: 50 chars max, capitalised, no period
- Body: wrap at 72 chars, explain *why* not just *what*

## Architecture Overview

```
cascade-cli          ← User-facing binary
  └─ cascade-verify  ← Pipeline orchestration, SARIF/JUnit output
       ├─ cascade-analysis  ← Two-tier analysis (Tier 1 + Tier 2)
       │    ├─ cascade-bmc     ← BMC encoding, MARCO MUS enumeration
       │    └─ cascade-service ← Service mesh modelling
       └─ cascade-repair    ← MaxSAT repair synthesis
            └─ cascade-maxsat  ← MaxSAT formula & solving
```

All crates depend on `cascade-types` (foundation types) and `cascade-graph`
(RTIG construction and algorithms).

## Areas for Contribution

### Good First Issues

- Add more unit tests for edge cases in `cascade-config` parsers
- Improve error messages with more context
- Add `--format mermaid` output option for `cascade-verify graph`

### Medium Complexity

- Support for Linkerd service mesh configuration parsing
- Add `proptest` property-based tests for the RTIG builder
- Implement `--watch` mode for continuous verification during development

### Research Contributions

- **Circuit-breaker extension:** Non-monotone model handling (significant research)
- **Compositional verification:** Decompose large topologies at cut vertices
- **Runtime integration:** Closed-loop with chaos testing results

## Release Process

1. Update version in all `Cargo.toml` files
2. Update `CHANGELOG.md`
3. Run full test suite: `cargo test --workspace`
4. Tag release: `git tag v0.x.y`
5. Build release binaries: `cargo build --release`

## License

By contributing, you agree that your contributions will be licensed under the
same MIT / Apache-2.0 dual license as the project.
