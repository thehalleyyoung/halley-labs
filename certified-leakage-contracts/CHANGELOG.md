# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-01-15

### Added

#### Core Framework
- Initial implementation of the reduced product domain D_spec ⊗ D_cache ⊗ D_quant
- Speculative reachability domain (D_spec) with bounded Spectre-PHT model
- Tainted abstract LRU cache domain (D_cache) with secret-dependence annotations
- Quantitative channel capacity domain (D_quant) with min-entropy bounds
- Reduction operator ρ for inter-domain information exchange
- Fixpoint engine with configurable widening (standard, delayed, threshold)
- Narrowing passes for improved precision

#### Abstract Interpretation Engine (`leak-abstract`)
- Complete lattice framework with Galois connections
- Worklist-based fixpoint algorithm
- Widening and narrowing operators
- Forward and backward transfer functions
- Generic reduced product combinator
- Abstract trace semantics

#### Leakage Contracts (`leak-contract`)
- Per-function leakage contract data structures
- Sequential composition rule: B_{f;g}(s) = B_f(s) + B_g(τ_f(s))
- Parallel and conditional composition rules
- Loop composition with widening
- Contract validation (soundness, monotonicity, independence checks)
- Contract storage and versioning
- Library-level bound computation
- Contract reporting with heatmaps and tables

#### Quantitative Information Flow (`leak-quantify`)
- Shannon, min-entropy, and guessing entropy computations
- Discrete memoryless channel abstraction
- Taint-restricted counting domain
- Cache timing and access pattern leakage models
- Speculative leakage model
- Per-function and compositional bound computation
- Vulnerability score and guessing advantage metrics

#### Binary Transformation (`leak-transform`)
- Analysis IR for x86-64 cryptographic instructions
- Instruction lifter (x86-64 → IR)
- Security annotation and taint tracking
- IR normalization: constant folding, dead code elimination, copy propagation
- Instruction canonicalization
- Loop unrolling with configurable bounds
- CFG construction
- ELF binary adapter with symbol table and function discovery

#### SMT Verification (`leak-smt`)
- Expression AST and sort system
- SMT-LIB2 script generation
- Z3 and CVC5 solver backends
- Cache theory encoding
- Leakage bound encoding
- Contract verification discharge

#### Certificate System (`leak-certify`)
- Hash-linked certificate chains
- Fixpoint witnesses
- Counting witnesses
- Reduction witnesses
- Composition witnesses
- Independent certificate checker
- JSON/CBOR certificate formats
- Audit trail and logging

#### Evaluation (`leak-eval`)
- Benchmark runner for crypto libraries
- Tool comparison framework (vs. CacheAudit, Spectector, Binsec)
- Synthetic test generator
- Precision, recall, false positive rate metrics
- Scalability profiling
- Statistical analysis (confidence intervals, effect size)
- CSV, JSON, and LaTeX report generation

#### CLI (`leak-cli`)
- `analyze` subcommand for per-function analysis
- `compose` subcommand for contract composition
- `certify` subcommand for certificate generation
- `regress` subcommand for CI regression detection
- `eval` subcommand for benchmarking
- TOML configuration file support
- JSON output mode for CI integration
- Structured exit codes

#### Infrastructure
- Workspace-level Cargo configuration with 11 crates
- Shared dependency management
- MIT OR Apache-2.0 dual licensing
- Comprehensive README with architecture diagrams
- Contributing guidelines
- Example scripts for common workflows

### Known Limitations
- Only LRU replacement policy supported in v0.1 (PLRU planned for v0.2)
- Speculation model limited to Spectre-PHT (Spectre-STL planned for v0.2)
- Single cache level (L1) analysis only
- Binary lifting covers ~150 crypto-critical x86-64 instructions

[Unreleased]: https://github.com/certified-leakage-contracts/certified-leakage-contracts/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/certified-leakage-contracts/certified-leakage-contracts/releases/tag/v0.1.0
