//! # LeakCert — Certified Leakage Contracts
//!
//! Compositional quantitative bounds for speculative cache side channels
//! in cryptographic binaries.
//!
//! This is the top-level crate that re-exports the framework's components
//! for convenient library usage.  For the CLI binary, see `leak-cli`.
//!
//! ## Quick start
//!
//! ```rust,no_run
//! use leakcert::contract::{LeakageContract, CacheTransformer, LeakageBound};
//! use leakcert::quantify::{Distribution, ShannonEntropy, MinEntropy};
//! use leakcert::types::CacheGeometry;
//! ```

/// Foundation types: addresses, cache geometry, CFG, instructions, registers.
pub use shared_types as types;

/// Abstract domain traits and leakage measurement types.
pub use leak_types as domain;

/// Abstract interpretation engine: lattices, fixpoint, widening.
pub use leak_abstract as interpret;

/// Core three-way reduced product analysis: D_spec ⊗ D_cache ⊗ D_quant.
pub use leak_analysis as analysis;

/// Compositional leakage contracts and composition rules.
pub use leak_contract as contract;

/// Machine-checkable certificate generation and verification.
pub use leak_certify as certify;

/// Quantitative information flow: entropy, channels, counting.
pub use leak_quantify as quantify;

/// Binary lifting, IR, and loop unrolling.
pub use leak_transform as transform;

/// SMT-based contract verification.
pub use leak_smt as smt;

/// Benchmarking and evaluation infrastructure.
pub use leak_eval as eval;
