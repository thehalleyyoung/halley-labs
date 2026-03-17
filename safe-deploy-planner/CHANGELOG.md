# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Nothing yet.

### Changed

- Nothing yet.

### Fixed

- Nothing yet.

## [0.1.0] — 2024-01-15

### Added

- **Core BMC planning engine** — bounded model-checking formulation for
  multi-service deployment sequencing with configurable horizon depth.
- **Interval-compressed constraint encoding** — compact SAT/SMT encoding that
  merges contiguous compatible time-steps, reducing clause count by up to 40×
  on real-world service graphs.
- **Rollback safety envelope computation** — automatic derivation of the maximal
  set of deployment prefixes from which every reachable state can be safely
  rolled back to the previous known-good configuration.
- **Schema compatibility oracle** with pluggable backends:
  - OpenAPI 3.x structural diff and backward-compatibility checking.
  - Protobuf wire-compatibility analysis (field numbering, type coercion).
- **Kubernetes integration**:
  - Helm chart version extraction and value-override generation.
  - Kustomize overlay patching for staged rollouts.
- **CLI** (`safestep`) with four primary commands:
  - `plan` — compute a safe deployment ordering.
  - `verify` — check an existing ordering against all constraints.
  - `envelope` — emit the rollback safety envelope as JSON.
  - `analyze` — run static analysis on a service dependency graph.
- **GitOps export formats**:
  - ArgoCD `ApplicationSet` manifest generation.
  - Flux `HelmRelease` / `Kustomization` manifest generation.
- **k-robustness checking** — verify that the computed plan tolerates up to *k*
  simultaneous service failures without leaving the safety envelope.
- **Structured logging** via `tracing` with JSON output support.
- **Machine-readable output** — all commands support `--format json` for CI
  integration.
- Comprehensive unit and integration test suite (~92 % line coverage).
- `examples/` directory with sample input configurations.

### Dependencies

- Rust 1.75+
- CaDiCaL ≥ 1.9 (SAT backend)
- Z3 ≥ 4.12 (SMT backend, optional for envelope analysis)

### Notes

- This is the initial public release.  The API surface is not yet stable;
  breaking changes may occur in 0.x releases.

---

[Unreleased]: https://github.com/your-org/safe-deploy-planner/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/your-org/safe-deploy-planner/releases/tag/v0.1.0
