# Changelog

All notable changes to the SoniType project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive README with architecture diagrams, examples, and theory overview
- Tool paper (tool_paper.tex) with SOTA comparisons against SonifYR, TwoTone,
  WebAudioXR, Sonification Sandbox, and manual sonification design
- groundings.json with 25 grounded claims and literature citations
- CONTRIBUTING.md with development guidelines
- LICENSE (MIT/Apache-2.0 dual license)
- .gitignore for Rust, LaTeX, and IDE artifacts
- Example sonification specifications (time series, multivariate, accessibility)
- Documentation directory with architecture and psychoacoustic model docs

### Fixed
- Compilation errors: added `#[derive(Debug, Clone)]` to `TimbreSpace`,
  `TimbreDistance`, and `PitchModel` structs in sonitype-psychoacoustic
- Float type ambiguity in `compute_feasible_region()` in sonitype-optimizer
- Borrow checker error in `RendererExecutor::process_block()` in sonitype-codegen
  by cloning processing order and wiring data before mutable iteration
- Borrow checker error in `Chorus::tick()` in sonitype-renderer by inlining
  `read_interp` to avoid overlapping immutable/mutable borrows
- Type inference ambiguity for `.into()` calls in sonitype-stdlib validation
  by replacing with explicit `.to_string()`

## [0.1.0] - 2025-01-15

### Added
- Initial implementation of the SoniType perceptual sonification compiler
- 11-crate Rust workspace architecture
- SoniType DSL with lexer, parser, AST, and type inference (sonitype-dsl)
- Perceptual type system with psychoacoustic constraint checking
- Psychoacoustic models: critical-band masking, JND validation, stream
  segregation, pitch perception, timbre analysis (sonitype-psychoacoustic)
- Intermediate representation with graph-based audio processing pipeline
  (sonitype-ir)
- Branch-and-bound optimizer with Bark-band decomposition and constraint
  propagation (sonitype-optimizer)
- Code generation with WCET analysis and buffer scheduling (sonitype-codegen)
- Audio renderer with oscillators, filters, effects, and mixing
  (sonitype-renderer)
- Standard library: scales, timbres, mappings, presets (sonitype-stdlib)
- Accessibility module: hearing profiles, adaptation, cognitive support
  (sonitype-accessibility)
- Real-time streaming with ring buffers and transport control
  (sonitype-streaming)
- CLI with compile, check, render, lint, and REPL commands (sonitype-cli)
- Core types, traits, error handling, and unit conversions (sonitype-core)
