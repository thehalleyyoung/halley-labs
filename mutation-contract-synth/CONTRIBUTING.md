# Contributing to MutSpec

Thank you for your interest in contributing to MutSpec! This document provides
guidelines for contributing to the mutation-contract synthesis engine.

## Code of Conduct

This project follows the [Contributor Covenant](https://www.contributor-covenant.org/)
code of conduct. By participating, you are expected to uphold this code.

## Getting Started

### Prerequisites

- **Rust 1.75+** with `cargo`
- **Z3 4.12+** (optional, for SMT verification)
- **pdflatex** (optional, for building documentation)

### Building from Source

```bash
cd implementation
cargo build --release
```

### Running Tests

```bash
cd implementation
cargo test
```

### Running Clippy

```bash
cd implementation
cargo clippy --all-targets --all-features -- -D warnings
```

## Development Workflow

1. **Fork** the repository and create a feature branch from `main`.
2. **Write tests** for any new functionality.
3. **Run `cargo check`** and `cargo clippy` before submitting.
4. **Open a pull request** with a clear description of your changes.

## Project Structure

```
implementation/
└── crates/
    ├── shared-types/      # Core AST, IR, formula, error types
    ├── mutation-core/     # Mutation operators and kill matrix
    ├── program-analysis/  # Parser, CFG, SSA, WP engine
    ├── smt-solver/        # SMT-LIB interface and incremental solving
    ├── contract-synth/    # Lattice-walk contract synthesis
    ├── coverage/          # Subsumption, dominators, scoring
    ├── gap-analysis/      # Bug detection via specification gaps
    ├── pit-integration/   # PIT XML report parsing
    ├── test-gen/          # Test generation types
    └── cli/               # Command-line interface
```

## Coding Conventions

- **Rustfmt**: All code must pass `cargo fmt --check`.
- **Clippy**: No warnings allowed (CI enforces `-D warnings`).
- **Documentation**: All public items must have rustdoc comments.
- **Error handling**: Use `thiserror` for error types, `anyhow` in the CLI.
  Library crates return `shared_types::Result<T>`.
- **Naming**: Follow Rust API guidelines. Prefer `snake_case` for functions,
  `CamelCase` for types, `SCREAMING_SNAKE_CASE` for constants.

## Adding a Mutation Operator

1. Create a new file in `crates/mutation-core/src/operators/`.
2. Implement the `MutationOperatorImpl` trait.
3. Register the operator in `operators/mod.rs`.
4. Add the operator variant to `MutationOperator` in `shared-types`.
5. Write tests covering all applicable expression/statement types.

## Reporting Issues

- Use GitHub Issues for bug reports and feature requests.
- Include a minimal reproducing example when reporting bugs.
- Tag issues with appropriate labels (`bug`, `enhancement`, `documentation`).

## License

By contributing, you agree that your contributions will be licensed under the
MIT License (see [LICENSE](LICENSE)).
