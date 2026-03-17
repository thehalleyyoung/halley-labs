# Changelog

All notable changes to MutSpec will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Complete Rust workspace with 10 crates covering the full MutSpec pipeline.
- Shared type system: AST, IR (SSA-based), formulas, contracts, error types.
- Mutation operators: AOR, ROR, LCR, UOI, ABS, COR, SDL, RVR, CRC, AIR, OSW, BCN.
- Kill matrix construction and mutant status tracking.
- Program analysis: parser, CFG builder, SSA transform, weakest-precondition engine.
- SMT solver interface with incremental solving and S-expression parsing.
- Contract synthesis via lattice-walk algorithm over discrimination lattice.
- Coverage analysis: subsumption hierarchies, dominator sets, adequacy scoring.
- Gap analysis: specification gap detection, witness generation, bug ranking.
- PIT integration: XML report parsing, bytecode-to-source mapping.
- CLI with subcommands: `mutate`, `analyze`, `synthesize`, `verify`, `report`, `config`.
- SARIF 2.1.0 report output for CI/CD integration.
- Configurable pipeline via TOML configuration files.
- Comprehensive README with architecture diagram, examples, and theory overview.
- Tool paper (tool_paper.tex) with SOTA comparison against Daikon, EvoSuite, Randoop, CodeContracts, and JML tools.
- Grounded claims document (groundings.json) with 30 evidence-backed claims.
- Example programs with expected contracts for testing.
- MIT license and contribution guidelines.

## [0.1.0] - 2025-07-18

### Added
- Initial project structure and ideation documents.
- Seed idea: mutation-guided contract synthesis via SyGuS.
- Crystallized problem statement and final approach design.
- Theory specification with 5 core theorems.
- Prior art audit covering 30+ related works.
