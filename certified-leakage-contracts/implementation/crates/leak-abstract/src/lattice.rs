//! Lattice-theoretic foundations for abstract interpretation.
//!
//! Provides the algebraic structures (join-semilattices, bounded lattices,
//! complete lattices) that underpin every abstract domain in the analysis.

use std::fmt::Debug;

use serde::{Serialize, Deserialize};

// ---------------------------------------------------------------------------
// Core lattice traits
// ---------------------------------------------------------------------------

/// A join-semilattice with a partial order.
///
/// Implementors must satisfy:
/// - `join` is commutative, associative, and idempotent.
/// - The partial order is consistent: `a ⊑ b  ⟺  a ⊔ b = b`.
pub trait Lattice: Clone + Debug + PartialEq {
    /// Least upper bound (join / ⊔).
    fn join(&self, other: &Self) -> Self;

    /// Greatest lower bound (meet / ⊓), when it exists.
    fn meet(&self, other: &Self) -> Self;

    /// Partial-order test: returns `true` when `self ⊑ other`.
    fn less_equal(&self, other: &Self) -> bool;

    /// Widening-safe equality: returns `true` when two elements are
    /// indistinguishable for fixpoint purposes.
    fn equivalent(&self, other: &Self) -> bool {
        self.less_equal(other) && other.less_equal(self)
    }
}

/// A lattice with distinguished ⊥ (bottom) and ⊤ (top) elements.
pub trait BoundedLattice: Lattice {
    /// The least element ⊥.
    fn bottom() -> Self;

    /// The greatest element ⊤.
    fn top() -> Self;

    /// Returns `true` if `self` is ⊥.
    fn is_bottom(&self) -> bool {
        self == &Self::bottom()
    }

    /// Returns `true` if `self` is ⊤.
    fn is_top(&self) -> bool {
        self == &Self::top()
    }
}

/// A complete lattice: every subset has both a least upper bound and a
/// greatest lower bound.
pub trait CompleteLattice: BoundedLattice {
    /// Compute the join of an arbitrary (possibly empty) collection.
    fn join_all<I: IntoIterator<Item = Self>>(iter: I) -> Self {
        iter.into_iter().fold(Self::bottom(), |acc, x| acc.join(&x))
    }

    /// Compute the meet of an arbitrary (possibly empty) collection.
    fn meet_all<I: IntoIterator<Item = Self>>(iter: I) -> Self {
        iter.into_iter().fold(Self::top(), |acc, x| acc.meet(&x))
    }
}

// ---------------------------------------------------------------------------
// FlatLattice – the canonical three-element lattice {⊥, v, ⊤}
// ---------------------------------------------------------------------------

/// A *flat* lattice that lifts any value type `T` into `{⊥, Value(T), ⊤}`.
///
/// Two distinct concrete values are incomparable, so their join is ⊤.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FlatLattice<T: Clone + Debug + PartialEq> {
    /// The least element – "no information".
    Bottom,
    /// A precise concrete value.
    Value(T),
    /// The greatest element – "any value possible".
    Top,
}

impl<T: Clone + Debug + PartialEq> Lattice for FlatLattice<T> {
    fn join(&self, other: &Self) -> Self {
        match (self, other) {
            (Self::Bottom, x) | (x, Self::Bottom) => x.clone(),
            (Self::Top, _) | (_, Self::Top) => Self::Top,
            (Self::Value(a), Self::Value(b)) if a == b => Self::Value(a.clone()),
            _ => Self::Top,
        }
    }

    fn meet(&self, other: &Self) -> Self {
        match (self, other) {
            (Self::Top, x) | (x, Self::Top) => x.clone(),
            (Self::Bottom, _) | (_, Self::Bottom) => Self::Bottom,
            (Self::Value(a), Self::Value(b)) if a == b => Self::Value(a.clone()),
            _ => Self::Bottom,
        }
    }

    fn less_equal(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Bottom, _) => true,
            (_, Self::Top) => true,
            (Self::Value(a), Self::Value(b)) => a == b,
            _ => false,
        }
    }
}

impl<T: Clone + Debug + PartialEq> BoundedLattice for FlatLattice<T> {
    fn bottom() -> Self {
        Self::Bottom
    }

    fn top() -> Self {
        Self::Top
    }
}

impl<T: Clone + Debug + PartialEq> CompleteLattice for FlatLattice<T> {}

impl<T: Clone + Debug + PartialEq> Default for FlatLattice<T> {
    fn default() -> Self {
        Self::Bottom
    }
}
