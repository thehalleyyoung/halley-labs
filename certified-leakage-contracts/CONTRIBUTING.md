# Contributing to Certified Leakage Contracts

Thank you for your interest in contributing! This document provides guidelines
and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Code Style](#code-style)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)

## Code of Conduct

This project follows the [Rust Code of Conduct](https://www.rust-lang.org/policies/code-of-conduct).
Please be respectful and constructive in all interactions.

## Getting Started

### Prerequisites

- Rust 1.75 or later (`rustup update stable`)
- Z3 4.12+ (optional, for SMT verification tests)
- Git

### Setting Up the Development Environment

```bash
# Clone your fork
git clone https://github.com/<your-username>/certified-leakage-contracts.git
cd certified-leakage-contracts/implementation

# Build the project
cargo build

# Run all tests
cargo test --all

# Run clippy
cargo clippy --all -- -D warnings

# Format code
cargo fmt --all
```

## Development Workflow

1. **Fork** the repository on GitHub.

2. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feat/my-feature main
   ```

3. **Make your changes.** Follow the [code style](#code-style) guidelines.

4. **Write or update tests** for your changes.

5. **Run the full check suite:**
   ```bash
   cargo fmt --all -- --check
   cargo clippy --all -- -D warnings
   cargo test --all
   ```

6. **Commit** with a descriptive message following
   [Conventional Commits](https://www.conventionalcommits.org/):
   ```
   feat(leak-analysis): add PLRU replacement policy support
   fix(leak-contract): correct composition bound for recursive calls
   docs(readme): update installation instructions
   test(leak-certify): add certificate chain validation tests
   ```

7. **Push** your branch and open a **Pull Request** against `main`.

## Code Style

### Formatting

- Use `rustfmt` with default settings. Run `cargo fmt --all` before every commit.
- Line width: 100 characters (rustfmt default).

### Linting

- Zero warnings from `cargo clippy --all -- -D warnings`.
- Address all clippy suggestions or add targeted `#[allow(...)]` with a comment
  explaining why.

### Documentation

- All `pub` items must have doc comments (`///` or `//!`).
- Include examples in doc comments for non-trivial APIs.
- Use `# Examples` sections in doc comments where helpful.

### Naming Conventions

- Types and traits: `PascalCase`
- Functions and methods: `snake_case`
- Constants: `SCREAMING_SNAKE_CASE`
- Modules: `snake_case`
- Type parameters: single uppercase letter or descriptive `PascalCase`

### Error Handling

- Use `thiserror` for library error types.
- Return `Result<T, E>` rather than panicking.
- Provide context in error messages.

### Dependencies

- Minimize new external dependencies.
- Prefer well-maintained, widely-used crates.
- Discuss new dependencies in the PR description.

## Testing

### Running Tests

```bash
# All tests
cargo test --all

# Specific crate
cargo test -p leak-analysis

# Specific test
cargo test -p leak-analysis -- test_reduction_operator

# With output
cargo test --all -- --nocapture
```

### Writing Tests

- Place unit tests in a `#[cfg(test)] mod tests` block within the source file.
- Place integration tests in `tests/` directories within each crate.
- Name test functions descriptively: `test_<what>_<condition>_<expected>`.
- Test both success and failure paths.
- Use property-based testing (proptest) for domain operations where applicable.

### Test Coverage

- New features should have >80% line coverage.
- Critical paths (composition, reduction, certificate checking) should have >90%.

## Pull Request Process

1. **Title**: Use Conventional Commits format.
2. **Description**: Explain what the PR does and why.
3. **Tests**: All tests must pass. Add new tests for new functionality.
4. **Review**: At least one maintainer approval is required.
5. **CI**: All CI checks must pass (fmt, clippy, test).
6. **Merge**: Squash-and-merge is preferred for feature branches.

### PR Checklist

- [ ] Code compiles without warnings
- [ ] All tests pass (`cargo test --all`)
- [ ] Code is formatted (`cargo fmt --all -- --check`)
- [ ] No clippy warnings (`cargo clippy --all -- -D warnings`)
- [ ] Documentation is updated if needed
- [ ] CHANGELOG.md is updated for user-facing changes

## Reporting Issues

- Use GitHub Issues for bug reports and feature requests.
- Include reproduction steps for bugs.
- For security vulnerabilities, please email the maintainers directly instead
  of opening a public issue.

## Architecture

See [docs/architecture.md](docs/architecture.md) for a detailed overview of the
system design and crate responsibilities.

## Questions?

Open a GitHub Discussion or reach out to the maintainers. We're happy to help!
