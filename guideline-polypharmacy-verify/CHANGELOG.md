# Changelog

All notable changes to GuardPharma will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Guideline validation module (`guideline-parser::validation`) for static checking of guideline documents before PTA compilation.
- Batch conflict detection module (`conflict-detect::batch`) for pairwise and N-wise analysis.
- Differential analysis module (`conflict-detect::differential`) for incremental re-verification on medication changes.
- Multi-format reporting module (`conflict-detect::reporting`) with plain-text, JSON, Markdown, and HTML output.
- Thread-safe `DelayedWidening` operator using `AtomicUsize` instead of `Cell<usize>`.
- Comprehensive project documentation: `CONTRIBUTING.md`, `CHANGELOG.md`, `LICENSE`, `.gitignore`.
- Example polypharmacy scenarios in `examples/`.
- Tool paper (`tool_paper.tex`) with SOTA comparison.
- Grounded claims documentation (`groundings.json`).

### Fixed
- Compilation error: missing `validation` module in `guideline-parser` crate.
- Compilation error: `Cell<usize>` not `Sync` in `DelayedWidening` — replaced with `AtomicUsize`.
- Type mismatch between `guardpharma_types::DrugId` and `guardpharma_conflict_detect::DrugId` in schedule optimizer.
- Missing field references (`drug_a`/`drug_b` → `drugs` vector) in recommendation crate.
- `OneCompartmentModel::from_pk_entry` replaced with direct field access to `DrugPkEntry`.
- Lifetime error in `AlternativeFinder::find_alternatives` for class-based lookup.
- Ambiguous float type in `patient_risk_factor` computation.

### Changed
- `ClassifiedConflict` field access now goes through `base.severity` instead of `severity`.

## [0.1.0] - 2025-01-15

### Added
- Initial implementation of the GuardPharma polypharmacy verification engine.
- 12-crate Rust workspace: types, pk-model, clinical, abstract-interp, smt-encoder, model-checker, guideline-parser, conflict-detect, significance, recommendation, evaluation, cli.
- Two-tier verification architecture: abstract interpretation screening + compositional model checking.
- Pharmacokinetic ODE models (1/2/3-compartment) with Metzler matrix representation.
- CYP enzyme inhibition modeling (competitive, noncompetitive, mixed).
- PK-aware widening operators for abstract interpretation convergence.
- Contract-based compositional safety verification for N-guideline composition.
- SMT-LIB2 encoding for bounded model checking via Z3.
- Clinical significance filtering using Beers Criteria, DrugBank severity, and FAERS disproportionality signals.
- Counterexample generation with clinical narrative translation.
- Schedule optimization for conflict resolution via temporal separation.
- Drug alternative recommendation engine.
