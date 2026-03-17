# Changelog

All notable changes to Penumbra will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-07-07

### Added

- **fpdiag-types**: Core type definitions for the entire workspace
  - IEEE 754 bit-level types and classification (`ieee754`)
  - Precision descriptors and cost modelling (`precision`)
  - Rounding modes including stochastic rounding (`rounding`)
  - Arena-based expression trees (`expression`)
  - Error metrics, intervals, and summary statistics (`error_bounds`)
  - Error Amplification Graph types with path analysis (`eag`)
  - Execution trace events and metadata (`trace`)
  - Five-category diagnosis taxonomy (`diagnosis`)
  - Repair strategies and certification types (`repair`)
  - Source location types (`source`)
  - Engine configuration structs (`config`)
  - ULP computation utilities (`ulp`)
  - Double-double extended-precision arithmetic (`double_double`)
  - Extended floating-point classification (`fpclass`)

- **fpdiag-analysis**: EAG construction and analysis
  - Streaming EAG builder from execution traces
  - Condition number estimation for common operations
  - Edge weight computation via finite differencing
  - T1 error bound computation
  - Error path decomposition
  - Treewidth estimation (min-fill heuristic)

- **fpdiag-symbolic**: Pattern matching on expression trees
  - Detection of `exp(x) - 1`, `log(1 + x)`, `sqrt(a² + b²)` patterns
  - Extensible pattern matcher framework

- **fpdiag-diagnosis**: Taxonomic diagnosis engine
  - Five classifiers: cancellation, absorption, smearing, amplified rounding,
    ill-conditioned subproblem
  - Confidence scoring and severity classification
  - Error contribution attribution via EAG paths

- **fpdiag-repair**: Repair synthesis and certification
  - Pattern library with 14+ repair strategies
  - T4-optimal greedy repair ordering
  - Interval arithmetic certification
  - Monotonicity and submodularity analysis

- **fpdiag-smt**: SMT solver integration
  - SMT-LIB2 formula encoding for FP expressions
  - Error reduction verification queries

- **fpdiag-transform**: Expression rewriting passes
  - Tree-to-tree transformation framework
  - Pattern-based rewrite application

- **fpdiag-eval**: Evaluation and benchmarking harness
  - Five built-in benchmark families
  - Metrics collection and CSV export

- **fpdiag-report**: Report generation
  - Human-readable reports with Unicode formatting
  - JSON output for programmatic consumption
  - CSV export for statistical analysis

- **fpdiag-cli**: Command-line interface
  - `penumbra trace` — Shadow-value instrumentation
  - `penumbra diagnose` — Error diagnosis
  - `penumbra repair` — Repair synthesis
  - `penumbra certify` — Repair certification
  - `penumbra report` — Full pipeline report
  - `penumbra bench` — Benchmark runner

### Infrastructure

- Rust workspace with 10 crates
- MIT/Apache-2.0 dual license
- README with 500+ lines of documentation
- Tool comparison paper (tool_paper.tex)
- 20 grounded claims (groundings.json)
- Contributing guidelines
- Example scripts

[0.1.0]: https://github.com/penumbra-fp/penumbra/releases/tag/v0.1.0
