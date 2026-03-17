# Contributing to Penumbra

Thank you for your interest in contributing to Penumbra! This document provides
guidelines for contributing to the project.

## Getting Started

1. **Fork** the repository
2. **Clone** your fork locally
3. **Build** the project: `cd implementation && cargo build`
4. **Test** the project: `cargo test`
5. **Check** the project: `cargo check && cargo clippy`

## Development Workflow

### Branch Naming

- `feature/short-description` for new features
- `fix/short-description` for bug fixes
- `docs/short-description` for documentation changes
- `refactor/short-description` for refactoring

### Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(diagnosis): add smearing classifier confidence scoring
fix(eag): handle zero-weight edges in path decomposition
docs(readme): add ill-conditioning example
test(repair): add property test for Kahan summation
```

### Code Style

- Follow `rustfmt` defaults (`cargo fmt`)
- Use `cargo clippy` with no warnings
- Write rustdoc for all public items
- Prefer explicit error types over `anyhow` in library crates

### Testing

- Unit tests: in the same file as the code (`#[cfg(test)]` module)
- Integration tests: in `tests/` directory of each crate
- Property-based tests: use `proptest` or `quickcheck` for numerical properties
- All PRs must pass `cargo test --workspace`

## Areas for Contribution

### High Priority

- **Python instrumentation layer** (PyO3): Implement the `__array_ufunc__` and
  `__array_function__` interceptors
- **MPFR integration**: Connect to `rug` or `gmp-mpfr-sys` for high-precision
  shadow values
- **Additional repair patterns**: Implement more algebraic rewrites from the
  numerical analysis literature

### Medium Priority

- **Benchmarks**: Add real-world scientific computing benchmarks from SciPy,
  scikit-learn, and Astropy issue trackers
- **Visualization**: EAG visualization using DOT/Graphviz or interactive web UI
- **Streaming trace format**: Binary format with LZ4/Zstd compression

### Lower Priority

- **SMT integration**: Connect to Z3 for formal repair validation
- **IDE integration**: VS Code extension for inline error annotations
- **CI/CD**: GitHub Actions for automated testing and benchmarking

## Reporting Issues

- Use GitHub Issues
- Include: Rust version, OS, minimal reproduction, expected vs. actual behavior
- For numerical bugs: include input values, expected output, actual output, and
  the error metric (ULPs, relative error, etc.)

## Code of Conduct

Be respectful, constructive, and inclusive. We follow the
[Rust Code of Conduct](https://www.rust-lang.org/policies/code-of-conduct).
