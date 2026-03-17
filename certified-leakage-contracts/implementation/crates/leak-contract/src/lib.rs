//! # Certified Leakage Contracts
//!
//! This crate implements **compositional leakage contracts** for speculative
//! side-channel analysis of x86-64 cryptographic binaries.
//!
//! A leakage contract for function *f* has the form:
//!
//! ```text
//! Contract_f = (τ_f : CacheState → CacheState,  B_f : CacheState → ℝ≥0)
//! ```
//!
//! where `τ_f` is a **cache transformer** describing how *f* changes the
//! abstract cache state, and `B_f` is a **leakage bound** mapping the initial
//! abstract cache state to a worst-case information leakage in bits.
//!
//! ## Composition rules
//!
//! | Pattern | Bound |
//! |---------|-------|
//! | Sequential `f ; g` | `B_{f;g}(s) = B_f(s) + B_g(τ_f(s))` |
//! | Parallel `f ‖ g` | `B_{f‖g}(s) = B_f(s) + B_g(s)` (independent) |
//! | Conditional `if c then f else g` | `B(s) = 1 + max(B_f(s), B_g(s))` |
//! | Loop `for i in 0..n { body }` | `Σ_{i=0}^{n-1} B_body(τ_body^i(s))` |
//!
//! ## Modules
//!
//! - [`contract`] – Core contract types and arithmetic
//! - [`composition`] – Sequential, parallel, conditional, and loop composition
//! - [`signature`] – Human-readable contract signatures
//! - [`regression`] – Regression detection across contract versions
//! - [`storage`] – Persistence and serialization
//! - [`validation`] – Well-formedness and soundness checks
//! - [`library`] – Library-level contract management
//! - [`display`] – Rich display and reporting

pub mod contract;
pub mod composition;
pub mod signature;
pub mod regression;
pub mod storage;
pub mod validation;
pub mod library;
pub mod display;

pub use contract::{
    LeakageContract, CacheTransformer, LeakageBound, ContractPrecondition,
    ContractPostcondition, ContractStrength, ContractMetadata,
    AbstractCacheState, CacheSetState, CacheLineState,
};
pub use composition::{
    compose_sequential, compose_parallel, compose_conditional, compose_loop,
    IndependenceChecker, CompositionError, WholeLibraryBound,
};
pub use signature::{
    ContractSignature, SignatureParser, SignatureFormatter, FunctionSignature,
};
pub use regression::{
    RegressionAnalyzer, ContractDelta, RegressionReport, RegressionSeverity, CIReport,
};
pub use storage::{
    ContractStore, ContractDatabase, ContractVersion,
};
pub use validation::{
    ContractValidator, SoundnessCheck, IndependenceVerifier, MonotonicityCheck,
    ValidationReport, ValidationResult, ValidationSeverity,
};
pub use library::{
    ContractLibrary, LibraryBound, DependencyGraph, ContractSummary,
    CryptoLibraryProfile,
};
pub use display::{
    ContractReport, ContractTable, LeakageHeatmap,
};
