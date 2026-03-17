//! Reduced product of abstract domains.
//!
//! Combines two or three independent abstract domains via a *reduction
//! operator* that exchanges information between components to tighten the
//! overall approximation.  This is the standard mechanism for running
//! speculation, cache, and quantitative analyses simultaneously.

use std::fmt;

use serde::{Serialize, Deserialize};

use crate::lattice::{Lattice, BoundedLattice};

// ---------------------------------------------------------------------------
// ReductionOperator
// ---------------------------------------------------------------------------

/// A reduction operator that tightens a product element by exploiting
/// inter-domain relationships.
pub trait ReductionOperator<A: Lattice, B: Lattice> {
    /// Reduce the pair `(a, b)` by transferring information between the
    /// two components.  The result must satisfy:
    ///   `reduce(a, b) ⊑ (a, b)`  in the product ordering.
    fn reduce(&self, a: &A, b: &B) -> (A, B);

    /// Human-readable name for diagnostics.
    fn name(&self) -> &str {
        "reduction"
    }
}

// ---------------------------------------------------------------------------
// ReducedProduct
// ---------------------------------------------------------------------------

/// The reduced product of two abstract domains `A` and `B`.
///
/// Each lattice operation first computes the component-wise result, then
/// applies the reduction to tighten the outcome.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReducedProduct<A: Lattice, B: Lattice> {
    /// First component.
    pub first: A,
    /// Second component.
    pub second: B,
}

impl<A: Lattice, B: Lattice> ReducedProduct<A, B> {
    /// Construct a new product element.
    pub fn new(first: A, second: B) -> Self {
        Self { first, second }
    }

    /// Apply a reduction operator to tighten this product element in-place.
    pub fn reduce<R: ReductionOperator<A, B>>(&mut self, op: &R) {
        let (a, b) = op.reduce(&self.first, &self.second);
        self.first = a;
        self.second = b;
    }
}

impl<A: Lattice, B: Lattice> Lattice for ReducedProduct<A, B> {
    fn join(&self, other: &Self) -> Self {
        Self {
            first: self.first.join(&other.first),
            second: self.second.join(&other.second),
        }
    }

    fn meet(&self, other: &Self) -> Self {
        Self {
            first: self.first.meet(&other.first),
            second: self.second.meet(&other.second),
        }
    }

    fn less_equal(&self, other: &Self) -> bool {
        self.first.less_equal(&other.first) && self.second.less_equal(&other.second)
    }
}

impl<A: BoundedLattice, B: BoundedLattice> BoundedLattice for ReducedProduct<A, B> {
    fn bottom() -> Self {
        Self {
            first: A::bottom(),
            second: B::bottom(),
        }
    }

    fn top() -> Self {
        Self {
            first: A::top(),
            second: B::top(),
        }
    }
}

impl<A: Lattice + fmt::Display, B: Lattice + fmt::Display> fmt::Display for ReducedProduct<A, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.first, self.second)
    }
}

// ---------------------------------------------------------------------------
// ThreeWayProduct
// ---------------------------------------------------------------------------

/// The reduced product of *three* abstract domains.
///
/// Designed for the core leakage analysis triple:
/// `(Speculation × Cache × Quantitative)`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThreeWayProduct<A: Lattice, B: Lattice, C: Lattice> {
    /// First component (e.g., speculation state).
    pub first: A,
    /// Second component (e.g., cache abstract state).
    pub second: B,
    /// Third component (e.g., quantitative leakage bound).
    pub third: C,
}

impl<A: Lattice, B: Lattice, C: Lattice> ThreeWayProduct<A, B, C> {
    /// Construct a new three-way product element.
    pub fn new(first: A, second: B, third: C) -> Self {
        Self { first, second, third }
    }
}

impl<A: Lattice, B: Lattice, C: Lattice> Lattice for ThreeWayProduct<A, B, C> {
    fn join(&self, other: &Self) -> Self {
        Self {
            first: self.first.join(&other.first),
            second: self.second.join(&other.second),
            third: self.third.join(&other.third),
        }
    }

    fn meet(&self, other: &Self) -> Self {
        Self {
            first: self.first.meet(&other.first),
            second: self.second.meet(&other.second),
            third: self.third.meet(&other.third),
        }
    }

    fn less_equal(&self, other: &Self) -> bool {
        self.first.less_equal(&other.first)
            && self.second.less_equal(&other.second)
            && self.third.less_equal(&other.third)
    }
}

impl<A: BoundedLattice, B: BoundedLattice, C: BoundedLattice> BoundedLattice
    for ThreeWayProduct<A, B, C>
{
    fn bottom() -> Self {
        Self {
            first: A::bottom(),
            second: B::bottom(),
            third: C::bottom(),
        }
    }

    fn top() -> Self {
        Self {
            first: A::top(),
            second: B::top(),
            third: C::top(),
        }
    }
}

impl<A: Lattice + fmt::Display, B: Lattice + fmt::Display, C: Lattice + fmt::Display> fmt::Display
    for ThreeWayProduct<A, B, C>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {}, {})", self.first, self.second, self.third)
    }
}
