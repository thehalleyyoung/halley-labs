# Changelog

All notable changes to CascadeVerify will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Two-tier analysis architecture (Tier 1 graph-algebraic, Tier 2 BMC)
- Retry-Timeout Interaction Graph (RTIG) construction from K8s/Istio/Envoy manifests
- Bounded model checking with QF_LIA encoding
- Monotonicity-aware antichain pruning for MUS enumeration (MARCO)
- MaxSAT repair synthesis with Pareto frontier enumeration
- Symmetry-breaking constraint generation for BMC pruning
- Cone-of-influence reduction for targeted analysis
- Incremental solving with clause reuse
- SARIF and JUnit XML output formats
- Configuration diff mode for CI/CD integration
- Helm template expansion and Kustomize overlay merging
- Comprehensive test suite (1,300+ tests)

### Architecture
- 10-crate Rust workspace (~60,000 lines of code)
- Foundation types with BitVec-based failure sets
- Builder patterns for ergonomic API construction
- Parallel analysis support via Rayon
- Full serialisation support via Serde

## [0.1.0] - 2026-03-09

### Added
- Initial release
- Core RTIG construction and analysis
- BMC encoding with monotonicity pruning
- MaxSAT repair synthesis
- CLI with check, repair, diff, graph, and benchmark commands
- Support for Kubernetes, Istio, Envoy, Helm, and Kustomize formats
