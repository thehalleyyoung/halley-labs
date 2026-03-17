//! # contract-synth
//!
//! Contract synthesis engine for the MutSpec project.
//!
//! Implements the mutation-specification duality: killed mutants define what tests
//! enforce (error predicates, negated and conjoined), survived mutants define what
//! tests permit. The contract synthesis makes this duality constructive.
//!
//! ## Tiers
//!
//! - **Tier 1 (Lattice Walk)**: Novel algorithm A2 for loop-free QF-LIA programs.
//!   Walks the specification lattice using dominator-ordered error predicates.
//! - **Tier 2 (Template)**: Template-based synthesis for programs with some
//!   non-QF-LIA features but where templates are applicable.
//! - **Tier 3 (Fallback)**: Daikon-quality dynamic invariant detection.
//!
//! ## Key Insight
//!
//! The sigma function σ: P(M_kill) → Spec sends a subset S of killed mutants
//! to ∧_{m ∈ S} ¬E(m), where E(m) is the error predicate of mutant m.
//! This is a lattice homomorphism from the powerset lattice to the specification
//! lattice ordered by logical implication.

pub mod lattice;
pub mod lattice_walk;
pub mod template_synth;
pub mod fallback;
pub mod simplification;
pub mod tier;
pub mod verification;
pub mod provenance;

pub use lattice::{SpecLattice, LatticeElement, DiscriminationLattice};
pub use lattice_walk::{LatticeWalkSynthesizer, WalkConfig, WalkState, WalkStep, WalkStatistics};
pub use template_synth::{TemplateSynthesizer, Template, TemplateInstance, TemplateConfig};
pub use fallback::{FallbackSynthesizer, FallbackConfig, DynamicInvariant, InvariantPattern};
pub use simplification::{ContractSimplifier, SimplificationConfig, SimplificationStats};
pub use tier::{TierClassifier, TierResult, TierConfig};
pub use verification::{ContractVerifier, VerificationResult, VerificationReport, VerifierConfig};
pub use provenance::{Provenance, ClauseOrigin, ProvenanceReport};
