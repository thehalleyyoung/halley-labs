//! # gap-analysis
//!
//! Gap analysis engine for the MutSpec mutation-contract synthesis pipeline.
//!
//! This crate implements the **Gap Theorem**: every surviving mutant is either
//! *equivalent* (semantically identical to the original program) or
//! *non-equivalent*.  Non-equivalent survivors that violate an inferred contract
//! are formal **bug witnesses** with concrete distinguishing inputs.
//!
//! ## Architecture
//!
//! The analysis proceeds in four phases:
//!
//! 1. **Equivalence Detection** ([`equivalence`]) – Classify each surviving
//!    mutant as equivalent or non-equivalent via SMT-backed semantic comparison.
//! 2. **Gap Analysis** ([`analyzer`]) – For every non-equivalent survivor, check
//!    whether the inferred contract distinguishes it from the original.  If not,
//!    the survivor witnesses a *specification gap*.
//! 3. **Witness Generation** ([`witness`]) – Produce concrete distinguishing
//!    inputs for each gap witness, suitable for automated test generation.
//! 4. **Reporting** ([`ranking`], [`sarif`], [`statistics`]) – Rank witnesses by
//!    severity and confidence, emit SARIF reports, and compute aggregate metrics.
//!
//! ## Modules
//!
//! - [`analyzer`]     – Core gap analyzer that processes surviving mutants
//!                      against inferred contracts.
//! - [`equivalence`]  – Equivalence detection for surviving mutants using SMT
//!                      queries.
//! - [`witness`]      – Gap witness generation: produces distinguishing inputs
//!                      for non-equivalent survivors.
//! - [`ranking`]      – Bug report ranking: prioritise gap witnesses by severity
//!                      and confidence.
//! - [`sarif`]        – SARIF report generation: produce Static Analysis Results
//!                      Interchange Format output.
//! - [`statistics`]   – Gap analysis statistics and metrics.

pub mod analyzer;
pub mod equivalence;
pub mod latent_bug_discriminator;
pub mod ranking;
pub mod sarif;
pub mod statistics;
pub mod witness;

// ---- Re-exports for ergonomic imports ------------------------------------

pub use analyzer::{GapAnalysisConfig, GapAnalysisResult, GapAnalyzer, GapReport};
pub use equivalence::{EquivalenceChecker, EquivalenceClass, EquivalenceResult};
pub use latent_bug_discriminator::{
    BoundaryWitness, BoundaryWitnessGenerator, BoundaryWitnessSeverity, DiscriminationPower,
    DiscriminatorConfig, DiscriminatorResult, LatentBugDiscriminator, LatticeBoundary,
};
pub use ranking::{Confidence, RankedWitness, RankingEngine, Severity};
pub use sarif::{SarifConfig, SarifEmitter, SarifReport};
pub use statistics::{GapMetrics, GapStatistics};
pub use witness::{DistinguishingInput, GapWitness, WitnessGenerator};
