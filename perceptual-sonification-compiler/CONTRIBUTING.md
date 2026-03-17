# Contributing to SoniType

Thank you for your interest in contributing to SoniType! This document provides
guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Architecture Overview](#architecture-overview)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Psychoacoustic Model Contributions](#psychoacoustic-model-contributions)

## Code of Conduct

This project follows the [Rust Code of Conduct](https://www.rust-lang.org/policies/code-of-conduct).
We are committed to providing a friendly, safe, and welcoming environment for all.

## Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork locally
3. **Create a branch** for your feature or fix
4. **Make changes** following our coding standards
5. **Test** your changes thoroughly
6. **Submit** a pull request

## Development Setup

### Prerequisites

- Rust 1.70+ (install via [rustup](https://rustup.rs/))
- Cargo (comes with Rust)
- A C compiler (for audio dependencies)

### Building

```bash
cd implementation
cargo build
```

### Running Tests

```bash
cd implementation
cargo test
```

### Running the CLI

```bash
cd implementation
cargo run --bin sonitype-cli -- --help
```

## Architecture Overview

SoniType is a Rust workspace with 11 crates:

| Crate | Purpose |
|-------|---------|
| `sonitype-core` | Shared types, traits, errors, audio primitives |
| `sonitype-dsl` | Lexer, parser, AST, type inference |
| `sonitype-psychoacoustic` | Masking, JND, segregation, pitch, timbre models |
| `sonitype-ir` | Intermediate representation, graph, analysis passes |
| `sonitype-optimizer` | Branch-and-bound search, constraint propagation |
| `sonitype-codegen` | Code generation, WCET analysis, scheduling |
| `sonitype-renderer` | Audio synthesis, effects, mixing |
| `sonitype-stdlib` | Scales, timbres, mappings, presets |
| `sonitype-accessibility` | Hearing profiles, adaptation, cognitive support |
| `sonitype-streaming` | Real-time streaming, buffering, transport |
| `sonitype-cli` | Command-line interface, REPL, diagnostics |

### Compilation Pipeline

```
DSL Source → Lexer → Parser → AST → Type Check → IR → Optimize → Codegen → Renderer
```

## Making Changes

### Branch Naming

- `feature/description` for new features
- `fix/description` for bug fixes
- `docs/description` for documentation
- `perf/description` for performance improvements

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(optimizer): add Bark-band decomposition pruning
fix(psychoacoustic): correct spreading function normalization
docs(readme): add multivariate sonification example
perf(codegen): optimize buffer allocation for real-time path
```

## Testing

### Unit Tests

Each crate should have comprehensive unit tests:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bark_to_hz_roundtrip() {
        for freq in [100.0, 440.0, 1000.0, 4000.0, 10000.0] {
            let bark = hz_to_bark(freq);
            let recovered = bark_to_hz(bark);
            assert!((recovered - freq).abs() < 1.0);
        }
    }
}
```

### Integration Tests

Place integration tests in `tests/` directories within each crate.

### Psychoacoustic Model Validation

When modifying psychoacoustic models, validate against published human data:
- JND thresholds: Moore (2012) reference values
- Masking: Schroeder spreading function baselines
- Segregation: Bregman (1990) streaming criteria

## Pull Request Process

1. Update documentation for any API changes
2. Add tests for new functionality
3. Ensure `cargo check` passes with no errors
4. Ensure `cargo test` passes
5. Run `cargo clippy` and address warnings
6. Update CHANGELOG.md with your changes
7. Request review from at least one maintainer

## Coding Standards

### Rust Style

- Follow the [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- Use `rustfmt` for formatting
- Use `clippy` for linting
- Document all public items with rustdoc

### Documentation

- Every public function, struct, and module needs a doc comment
- Include examples in doc comments where appropriate
- Use `///` for item docs and `//!` for module docs

### Error Handling

- Use `thiserror` for library error types
- Use `anyhow` for application-level errors (CLI only)
- Prefer `Result` over panics
- Provide meaningful error messages with context

### Performance

- Profile before optimizing
- Document complexity of algorithms
- Use `#[inline]` judiciously
- Prefer stack allocation over heap when feasible

## Psychoacoustic Model Contributions

If you're contributing psychoacoustic models or parameters:

1. **Cite sources**: Every psychoacoustic parameter must have a literature citation
2. **Validate**: Compare model outputs against published human data
3. **Document assumptions**: Clearly state what the model assumes
4. **Bound errors**: Quantify the model's accuracy range

### Key References

- Moore, B.C.J. (2012). *An Introduction to the Psychology of Hearing*
- Zwicker & Fastl (2013). *Psychoacoustics: Facts and Models*
- Bregman, A.S. (1990). *Auditory Scene Analysis*
- Kramer et al. (2010). *Sonification Report: Status of the Field*

## Questions?

Open an issue with the `question` label or start a discussion in the
Discussions tab.

Thank you for contributing to SoniType!
