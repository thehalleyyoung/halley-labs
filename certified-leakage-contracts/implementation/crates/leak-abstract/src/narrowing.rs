//! Narrowing operators for recovering precision after widening.
//!
//! After an ascending chain has stabilised (possibly with widening-induced
//! over-approximation), narrowing iterates a descending chain to tighten the
//! result while preserving soundness.

use crate::lattice::Lattice;

/// A narrowing operator on abstract domain `V`.
///
/// Given the current (widened) fixpoint `current` and a tighter candidate
/// `candidate`, `narrow` must return an element `n` such that:
///
/// 1. `candidate ⊑ n ⊑ current`  (the result is between the two inputs).
/// 2. Every descending chain `v₀ △ v₁ △ v₂ …` eventually stabilises.
pub trait NarrowingOperator<V: Lattice> {
    /// Apply narrowing: `current △ candidate`.
    fn narrow(&self, current: &V, candidate: &V) -> V;

    /// Maximum number of narrowing steps to perform.
    /// Returning `None` means "use the engine default".
    fn max_steps(&self) -> Option<usize> {
        None
    }

    /// Human-readable name for diagnostics.
    fn name(&self) -> &str {
        "narrowing"
    }
}
