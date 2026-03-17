# Contributing to IsoSpec

Thank you for your interest in contributing to IsoSpec! This document provides
guidelines for contributing to the project.

## Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork locally
3. **Create a branch** for your feature or fix
4. **Make your changes** following the guidelines below
5. **Test** your changes
6. **Submit a pull request**

## Development Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/isospec.git
cd isospec/implementation

# Build
cargo build

# Run tests
cargo test

# Run linter
cargo clippy -- -D warnings

# Format code
cargo fmt
```

## Code Guidelines

### Rust Style

- Follow standard Rust formatting (`cargo fmt`)
- All public items must have rustdoc documentation comments (`///`)
- Use `thiserror` for library error types
- Use `anyhow` for application-level errors
- Prefer `Result<T, IsoSpecError>` over panicking
- No `unwrap()` in library code; use `expect()` with descriptive messages
  only in tests
- Use `#[must_use]` on functions that return values that should not be ignored

### Commit Messages

Use conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Examples:
- `feat(engines): add Oracle 23c MVRC model`
- `fix(smt): handle timeout in incremental solver`
- `docs(readme): update installation instructions`

### Testing

- Write unit tests for all new public functions
- Add integration tests for cross-crate interactions
- Include property-based tests for critical algorithms (cycle detection,
  predicate conflict)
- Test edge cases: empty histories, single-transaction workloads, NULL values

### Engine Models

When adding or modifying engine models:

1. Document the source of behavioral specifications (manual pages, source code
   references, academic papers)
2. Include version numbers for the specific engine release modeled
3. Write adequacy tests against live database instances where possible
4. Document any over-approximations and their soundness justification

## Reporting Issues

- Use GitHub Issues for bug reports and feature requests
- Include: IsoSpec version, Rust version, OS, reproduction steps
- For anomaly detection issues, include the input history JSON

## Pull Request Process

1. Ensure all tests pass (`cargo test`)
2. Ensure no clippy warnings (`cargo clippy -- -D warnings`)
3. Ensure code is formatted (`cargo fmt --check`)
4. Update documentation for any API changes
5. Add a CHANGELOG entry for user-visible changes
6. Request review from at least one maintainer

## License

By contributing, you agree that your contributions will be licensed under the
MIT/Apache-2.0 dual license.
