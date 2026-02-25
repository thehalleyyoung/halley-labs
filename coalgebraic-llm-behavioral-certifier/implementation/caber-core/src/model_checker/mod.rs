//! Model checker module — QCTL_F model checking on finite probabilistic coalgebras.
//!
//! Provides the main model checking engine with support for CTL temporal operators,
//! quantitative (graded) satisfaction, fixpoint computation, and witness/counterexample generation.

pub mod checker;
pub mod fixpoint;
pub mod witness;
pub mod graded;
