//! Bisimulation module — compositional bisimulation engine.
//!
//! Provides exact bisimulation via partition refinement, quantitative bisimulation
//! distances via Kantorovich lifting, optimal transport computations, and
//! witness/counterexample generation for bisimilarity proofs.

pub mod exact;
pub mod quantitative;
pub mod lifting;
pub mod witness_gen;
