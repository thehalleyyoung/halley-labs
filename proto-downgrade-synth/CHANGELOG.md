# Changelog

All notable changes to NegSynth will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Complete Rust workspace with 10 crates covering the full NegSynth pipeline
- Protocol-aware merge operator with algebraic property exploitation (`negsyn-merge`)
- Protocol-aware program slicer with vtable and callback tracking (`negsyn-slicer`)
- Bisimulation-quotient state machine extractor (`negsyn-extract`)
- Dolev-Yao + SMT constraint encoder with CEGAR loop (`negsyn-encode`, `negsyn-concrete`)
- TLS 1.0–1.3 protocol models including cipher suite negotiation (`negsyn-proto-tls`)
- SSH v2 protocol models including key exchange and algorithm negotiation (`negsyn-proto-ssh`)
- Evaluation harness with CVE oracle and benchmark framework (`negsyn-eval`)
- CLI with analyze, verify, diff, replay, benchmark, and inspect commands (`negsyn-cli`)
- Comprehensive type system with 80+ core types (`negsyn-types`)
- SARIF output format for CI/CD integration
- JSON and DOT graph output formats
- Bounded-completeness certificate generation
- Cross-library differential analysis support

### Security
- Replaced panic-based error handling with proper Result types in production code
- Added input validation for all CLI parameters
- Certificate chain validation for bounded-completeness proofs

## [0.1.0] - 2026-03-08

### Added
- Initial project structure and workspace layout
- Core type definitions for symbolic states, negotiation protocols, and SMT expressions
- Foundational error handling hierarchy with context tracking
- Configuration system with builder pattern and validation
- Protocol version ordering and cipher suite registry
- Adversary model based on Dolev-Yao message algebra
- Graph types for state machine representation with bisimulation relations
- Metrics collection and phase timing infrastructure

### Changed
- N/A (initial release)

### Deprecated
- N/A (initial release)

### Removed
- N/A (initial release)

### Fixed
- N/A (initial release)

[Unreleased]: https://github.com/negsyn/negsyn/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/negsyn/negsyn/releases/tag/v0.1.0
