//! Widening operators for accelerating fixpoint convergence.
//!
//! Widening ensures termination of ascending chains in infinite-height
//! lattices by over-approximating the limit in finitely many steps.

use crate::lattice::Lattice;

/// A widening operator on abstract domain `V`.
///
/// Given the previous iterate `prev` and the next candidate `next`,
/// `widen` must return an element `w` such that:
///
/// 1. `next ⊑ w`  (soundness – the result over-approximates).
/// 2. Every ascending chain `v₀ ▽ v₁ ▽ v₂ …` eventually stabilises.
pub trait WideningOperator<V: Lattice> {
    /// Apply widening: `prev ▽ next`.
    fn widen(&self, prev: &V, next: &V) -> V;

    /// Optional: number of iterations to delay before engaging widening.
    /// Returning `None` means "use the engine default".
    fn delay(&self) -> Option<usize> {
        None
    }

    /// Human-readable name for diagnostics.
    fn name(&self) -> &str {
        "widening"
    }
}
