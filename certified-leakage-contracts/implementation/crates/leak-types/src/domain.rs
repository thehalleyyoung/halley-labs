//! Abstract domain trait hierarchy for static analysis.
//!
//! This module defines the core abstract interpretation framework used throughout
//! the certified leakage analysis. All abstract domains (cache states, taint maps,
//! leakage bounds) conform to these traits, enabling compositional fixpoint computation.
//!
//! The lattice-theoretic foundations follow the standard Cousot & Cousot framework,
//! with extensions for widening/narrowing to ensure termination on infinite-height domains.

use std::fmt;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Core abstract domain trait
// ---------------------------------------------------------------------------

/// Core trait for abstract domains in the analysis lattice.
///
/// Every abstract state must form a partial order with join (⊔) and meet (⊓).
/// Widening (∇) and narrowing (△) ensure convergence of fixpoint iteration.
pub trait AbstractDomain: Clone + fmt::Debug + PartialEq {
    /// Least upper bound (join / ⊔). Over-approximates the union of
    /// concrete states represented by `self` and `other`.
    fn join(&self, other: &Self) -> Self;

    /// Greatest lower bound (meet / ⊓). Under-approximates the intersection.
    fn meet(&self, other: &Self) -> Self;

    /// Widening operator (∇). Guarantees eventual stabilization of ascending
    /// chains by "jumping ahead" in the lattice.
    fn widen(&self, other: &Self) -> Self;

    /// Narrowing operator (△). Improves precision after widening by descending
    /// towards the least fixpoint.
    fn narrow(&self, other: &Self) -> Self;

    /// Returns `true` if this is the bottom element (⊥), representing the
    /// empty set of concrete states (unreachable).
    fn is_bottom(&self) -> bool;

    /// Returns `true` if this is the top element (⊤), representing all
    /// possible concrete states (no information).
    fn is_top(&self) -> bool;

    /// Partial order test: `self ⊑ other`.
    fn partial_order(&self, other: &Self) -> bool;

    /// Returns the bottom element of this domain.
    fn bottom() -> Self;

    /// Returns the top element of this domain.
    fn top() -> Self;

    /// Join-assign: `self = self ⊔ other`. Returns whether `self` changed.
    fn join_assign(&mut self, other: &Self) -> bool {
        let joined = self.join(other);
        if joined != *self {
            *self = joined;
            true
        } else {
            false
        }
    }

    /// Meet-assign: `self = self ⊓ other`. Returns whether `self` changed.
    fn meet_assign(&mut self, other: &Self) -> bool {
        let met = self.meet(other);
        if met != *self {
            *self = met;
            true
        } else {
            false
        }
    }

    /// Widen-assign: `self = self ∇ other`. Returns whether `self` changed.
    fn widen_assign(&mut self, other: &Self) -> bool {
        let widened = self.widen(other);
        if widened != *self {
            *self = widened;
            true
        } else {
            false
        }
    }

    /// Narrow-assign: `self = self △ other`. Returns whether `self` changed.
    fn narrow_assign(&mut self, other: &Self) -> bool {
        let narrowed = self.narrow(other);
        if narrowed != *self {
            *self = narrowed;
            true
        } else {
            false
        }
    }
}

// ---------------------------------------------------------------------------
// Finite-height lattice
// ---------------------------------------------------------------------------

/// Marker trait for domains with finite height.
///
/// Finite-height lattices guarantee termination of ascending chain iteration
/// without widening. The `height()` method returns the length of the longest
/// ascending chain from ⊥ to ⊤.
pub trait FiniteDomain: AbstractDomain {
    /// Maximum length of any ascending chain from ⊥ to ⊤.
    fn height() -> u64;

    /// Rank of the current element in the lattice (distance from ⊥).
    fn rank(&self) -> u64;
}

// ---------------------------------------------------------------------------
// Product domain
// ---------------------------------------------------------------------------

/// A product domain combining two abstract domains component-wise.
///
/// Operations are applied independently to each component:
///   `(a₁, b₁) ⊔ (a₂, b₂) = (a₁ ⊔ a₂, b₁ ⊔ b₂)`
///
/// This is the standard reduced product construction. For truly reduced
/// products with inter-component reduction, override the default methods.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProductDomain<A, B> {
    /// First component.
    pub first: A,
    /// Second component.
    pub second: B,
}

impl<A, B> ProductDomain<A, B> {
    /// Create a new product domain from two components.
    pub fn new(first: A, second: B) -> Self {
        Self { first, second }
    }

    /// Destructure into components.
    pub fn into_parts(self) -> (A, B) {
        (self.first, self.second)
    }
}

impl<A: AbstractDomain, B: AbstractDomain> AbstractDomain for ProductDomain<A, B>
where
    A: Serialize + for<'de> Deserialize<'de>,
    B: Serialize + for<'de> Deserialize<'de>,
{
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

    fn widen(&self, other: &Self) -> Self {
        Self {
            first: self.first.widen(&other.first),
            second: self.second.widen(&other.second),
        }
    }

    fn narrow(&self, other: &Self) -> Self {
        Self {
            first: self.first.narrow(&other.first),
            second: self.second.narrow(&other.second),
        }
    }

    fn is_bottom(&self) -> bool {
        self.first.is_bottom() || self.second.is_bottom()
    }

    fn is_top(&self) -> bool {
        self.first.is_top() && self.second.is_top()
    }

    fn partial_order(&self, other: &Self) -> bool {
        self.first.partial_order(&other.first) && self.second.partial_order(&other.second)
    }

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

impl<A: FiniteDomain, B: FiniteDomain> FiniteDomain for ProductDomain<A, B>
where
    A: Serialize + for<'de> Deserialize<'de>,
    B: Serialize + for<'de> Deserialize<'de>,
{
    fn height() -> u64 {
        A::height() + B::height()
    }

    fn rank(&self) -> u64 {
        self.first.rank() + self.second.rank()
    }
}

impl<A: fmt::Display, B: fmt::Display> fmt::Display for ProductDomain<A, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.first, self.second)
    }
}

// ---------------------------------------------------------------------------
// Powerset domain
// ---------------------------------------------------------------------------

/// A powerset domain over a base set of elements.
///
/// Represents sets of concrete values with join = union, meet = intersection.
/// Used for may/must analyses where we track sets of possible cache lines,
/// tainted addresses, etc.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PowersetDomain<T: Ord + Clone + fmt::Debug> {
    /// Elements in the set. Kept sorted for deterministic iteration.
    elements: Vec<T>,
    /// Whether this represents the universal set (⊤).
    is_universe: bool,
}

impl<T: Ord + Clone + fmt::Debug> PowersetDomain<T> {
    /// Create the empty powerset (⊥).
    pub fn empty() -> Self {
        Self {
            elements: Vec::new(),
            is_universe: false,
        }
    }

    /// Create a powerset with a single element.
    pub fn singleton(elem: T) -> Self {
        Self {
            elements: vec![elem],
            is_universe: false,
        }
    }

    /// Create a powerset from an iterator of elements.
    pub fn from_iter(iter: impl IntoIterator<Item = T>) -> Self {
        let mut elements: Vec<T> = iter.into_iter().collect();
        elements.sort();
        elements.dedup();
        Self {
            elements,
            is_universe: false,
        }
    }

    /// Create the universal set (⊤).
    pub fn universe() -> Self {
        Self {
            elements: Vec::new(),
            is_universe: true,
        }
    }

    /// Insert an element.
    pub fn insert(&mut self, elem: T) {
        if self.is_universe {
            return;
        }
        match self.elements.binary_search(&elem) {
            Ok(_) => {}
            Err(pos) => self.elements.insert(pos, elem),
        }
    }

    /// Remove an element.
    pub fn remove(&mut self, elem: &T) {
        if self.is_universe {
            return;
        }
        if let Ok(pos) = self.elements.binary_search(elem) {
            self.elements.remove(pos);
        }
    }

    /// Check containment of an element.
    pub fn contains(&self, elem: &T) -> bool {
        if self.is_universe {
            return true;
        }
        self.elements.binary_search(elem).is_ok()
    }

    /// Number of elements (returns None for universe).
    pub fn len(&self) -> Option<usize> {
        if self.is_universe {
            None
        } else {
            Some(self.elements.len())
        }
    }

    /// Whether the concrete set is empty.
    pub fn is_empty(&self) -> bool {
        !self.is_universe && self.elements.is_empty()
    }

    /// Iterate over elements (panics if universe).
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        assert!(
            !self.is_universe,
            "cannot iterate over universal powerset"
        );
        self.elements.iter()
    }

    /// Number of elements for a finite set, or 0 for universe.
    pub fn cardinality(&self) -> usize {
        if self.is_universe {
            0
        } else {
            self.elements.len()
        }
    }

    /// Set union.
    pub fn union(&self, other: &Self) -> Self {
        if self.is_universe || other.is_universe {
            return Self::universe();
        }
        let mut result = self.elements.clone();
        for e in &other.elements {
            if let Err(pos) = result.binary_search(e) {
                result.insert(pos, e.clone());
            }
        }
        Self {
            elements: result,
            is_universe: false,
        }
    }

    /// Set intersection.
    pub fn intersection(&self, other: &Self) -> Self {
        if self.is_universe {
            return other.clone();
        }
        if other.is_universe {
            return self.clone();
        }
        let mut result = Vec::new();
        let (mut i, mut j) = (0, 0);
        while i < self.elements.len() && j < other.elements.len() {
            match self.elements[i].cmp(&other.elements[j]) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    result.push(self.elements[i].clone());
                    i += 1;
                    j += 1;
                }
            }
        }
        Self {
            elements: result,
            is_universe: false,
        }
    }

    /// Set difference: `self \ other`.
    pub fn difference(&self, other: &Self) -> Self {
        if self.is_universe || other.is_universe {
            return Self::empty();
        }
        let elements = self
            .elements
            .iter()
            .filter(|e| !other.contains(e))
            .cloned()
            .collect();
        Self {
            elements,
            is_universe: false,
        }
    }

    /// Subset test: `self ⊆ other`.
    pub fn is_subset_of(&self, other: &Self) -> bool {
        if other.is_universe {
            return true;
        }
        if self.is_universe {
            return false;
        }
        self.elements.iter().all(|e| other.contains(e))
    }
}

impl<T: Ord + Clone + fmt::Debug + Serialize + for<'de> Deserialize<'de>> AbstractDomain
    for PowersetDomain<T>
{
    fn join(&self, other: &Self) -> Self {
        self.union(other)
    }

    fn meet(&self, other: &Self) -> Self {
        self.intersection(other)
    }

    fn widen(&self, other: &Self) -> Self {
        // For finite powersets widening = join (termination is guaranteed).
        self.join(other)
    }

    fn narrow(&self, other: &Self) -> Self {
        // Narrowing = meet for finite powersets.
        self.meet(other)
    }

    fn is_bottom(&self) -> bool {
        self.is_empty()
    }

    fn is_top(&self) -> bool {
        self.is_universe
    }

    fn partial_order(&self, other: &Self) -> bool {
        self.is_subset_of(other)
    }

    fn bottom() -> Self {
        Self::empty()
    }

    fn top() -> Self {
        Self::universe()
    }
}

impl<T: Ord + Clone + fmt::Debug + fmt::Display> fmt::Display for PowersetDomain<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_universe {
            write!(f, "⊤")
        } else if self.elements.is_empty() {
            write!(f, "∅")
        } else {
            write!(f, "{{")?;
            for (i, e) in self.elements.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{e}")?;
            }
            write!(f, "}}")
        }
    }
}

// ---------------------------------------------------------------------------
// Flat lattice: ⊥ < concrete values < ⊤
// ---------------------------------------------------------------------------

/// A flat lattice lifting a base type with bottom and top.
///
/// The ordering is: ⊥ ⊑ Concrete(v) ⊑ ⊤ for all v,
/// and `Concrete(v₁)` and `Concrete(v₂)` are incomparable when v₁ ≠ v₂.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FlatLattice<T> {
    /// Bottom element – unreachable / no information.
    Bottom,
    /// A concrete value.
    Value(T),
    /// Top element – any value possible.
    Top,
}

impl<T: Clone + PartialEq + fmt::Debug> FlatLattice<T> {
    /// Create a concrete value.
    pub fn value(v: T) -> Self {
        FlatLattice::Value(v)
    }

    /// Extract the concrete value if present.
    pub fn as_value(&self) -> Option<&T> {
        match self {
            FlatLattice::Value(v) => Some(v),
            _ => None,
        }
    }

    /// Map over the concrete value.
    pub fn map<U: Clone + PartialEq + fmt::Debug>(
        &self,
        f: impl FnOnce(&T) -> U,
    ) -> FlatLattice<U> {
        match self {
            FlatLattice::Bottom => FlatLattice::Bottom,
            FlatLattice::Value(v) => FlatLattice::Value(f(v)),
            FlatLattice::Top => FlatLattice::Top,
        }
    }
}

impl<T: Clone + PartialEq + fmt::Debug + Serialize + for<'de> Deserialize<'de>> AbstractDomain
    for FlatLattice<T>
{
    fn join(&self, other: &Self) -> Self {
        match (self, other) {
            (FlatLattice::Bottom, x) | (x, FlatLattice::Bottom) => x.clone(),
            (FlatLattice::Top, _) | (_, FlatLattice::Top) => FlatLattice::Top,
            (FlatLattice::Value(a), FlatLattice::Value(b)) => {
                if a == b {
                    self.clone()
                } else {
                    FlatLattice::Top
                }
            }
        }
    }

    fn meet(&self, other: &Self) -> Self {
        match (self, other) {
            (FlatLattice::Top, x) | (x, FlatLattice::Top) => x.clone(),
            (FlatLattice::Bottom, _) | (_, FlatLattice::Bottom) => FlatLattice::Bottom,
            (FlatLattice::Value(a), FlatLattice::Value(b)) => {
                if a == b {
                    self.clone()
                } else {
                    FlatLattice::Bottom
                }
            }
        }
    }

    fn widen(&self, other: &Self) -> Self {
        self.join(other)
    }

    fn narrow(&self, other: &Self) -> Self {
        self.meet(other)
    }

    fn is_bottom(&self) -> bool {
        matches!(self, FlatLattice::Bottom)
    }

    fn is_top(&self) -> bool {
        matches!(self, FlatLattice::Top)
    }

    fn partial_order(&self, other: &Self) -> bool {
        match (self, other) {
            (FlatLattice::Bottom, _) => true,
            (_, FlatLattice::Top) => true,
            (FlatLattice::Value(a), FlatLattice::Value(b)) => a == b,
            _ => false,
        }
    }

    fn bottom() -> Self {
        FlatLattice::Bottom
    }

    fn top() -> Self {
        FlatLattice::Top
    }
}

impl<T: Clone + PartialEq + fmt::Debug + Serialize + for<'de> Deserialize<'de>> FiniteDomain
    for FlatLattice<T>
{
    fn height() -> u64 {
        2
    }

    fn rank(&self) -> u64 {
        match self {
            FlatLattice::Bottom => 0,
            FlatLattice::Value(_) => 1,
            FlatLattice::Top => 2,
        }
    }
}

impl<T: fmt::Display> fmt::Display for FlatLattice<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FlatLattice::Bottom => write!(f, "⊥"),
            FlatLattice::Value(v) => write!(f, "{v}"),
            FlatLattice::Top => write!(f, "⊤"),
        }
    }
}

// ---------------------------------------------------------------------------
// Two-element boolean domain
// ---------------------------------------------------------------------------

/// A two-element abstract domain {⊥, ⊤} isomorphic to `bool`.
///
/// Used where only reachability matters (e.g., "is this path feasible?").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BoolDomain {
    /// `false` = ⊥ (unreachable), `true` = ⊤ (reachable).
    pub value: bool,
}

impl BoolDomain {
    /// Create a reachable state.
    pub fn reachable() -> Self {
        Self { value: true }
    }

    /// Create an unreachable state.
    pub fn unreachable() -> Self {
        Self { value: false }
    }
}

impl AbstractDomain for BoolDomain {
    fn join(&self, other: &Self) -> Self {
        Self {
            value: self.value || other.value,
        }
    }

    fn meet(&self, other: &Self) -> Self {
        Self {
            value: self.value && other.value,
        }
    }

    fn widen(&self, other: &Self) -> Self {
        self.join(other)
    }

    fn narrow(&self, other: &Self) -> Self {
        self.meet(other)
    }

    fn is_bottom(&self) -> bool {
        !self.value
    }

    fn is_top(&self) -> bool {
        self.value
    }

    fn partial_order(&self, other: &Self) -> bool {
        !self.value || other.value
    }

    fn bottom() -> Self {
        Self::unreachable()
    }

    fn top() -> Self {
        Self::reachable()
    }
}

impl FiniteDomain for BoolDomain {
    fn height() -> u64 {
        1
    }

    fn rank(&self) -> u64 {
        u64::from(self.value)
    }
}

impl fmt::Display for BoolDomain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.value {
            write!(f, "⊤")
        } else {
            write!(f, "⊥")
        }
    }
}

// ---------------------------------------------------------------------------
// Domain validation helpers
// ---------------------------------------------------------------------------

/// Result of a lattice property check.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LatticeCheck {
    /// Name of the property that was checked.
    pub property: String,
    /// Whether the property holds.
    pub holds: bool,
    /// Optional counter-example or explanation.
    pub message: Option<String>,
}

impl LatticeCheck {
    /// A passing check.
    pub fn pass(property: impl Into<String>) -> Self {
        Self {
            property: property.into(),
            holds: true,
            message: None,
        }
    }

    /// A failing check with an explanation.
    pub fn fail(property: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            property: property.into(),
            holds: false,
            message: Some(message.into()),
        }
    }
}

impl fmt::Display for LatticeCheck {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let status = if self.holds { "✓" } else { "✗" };
        write!(f, "[{status}] {}", self.property)?;
        if let Some(msg) = &self.message {
            write!(f, ": {msg}")?;
        }
        Ok(())
    }
}

/// Verify that `join` is commutative for two elements.
pub fn check_join_commutative<D: AbstractDomain>(a: &D, b: &D) -> LatticeCheck {
    let ab = a.join(b);
    let ba = b.join(a);
    if ab == ba {
        LatticeCheck::pass("join_commutative")
    } else {
        LatticeCheck::fail("join_commutative", "a ⊔ b ≠ b ⊔ a")
    }
}

/// Verify that `join` is associative for three elements.
pub fn check_join_associative<D: AbstractDomain>(a: &D, b: &D, c: &D) -> LatticeCheck {
    let ab_c = a.join(b).join(c);
    let a_bc = a.join(&b.join(c));
    if ab_c == a_bc {
        LatticeCheck::pass("join_associative")
    } else {
        LatticeCheck::fail("join_associative", "(a ⊔ b) ⊔ c ≠ a ⊔ (b ⊔ c)")
    }
}

/// Verify that `join` is idempotent.
pub fn check_join_idempotent<D: AbstractDomain>(a: &D) -> LatticeCheck {
    let aa = a.join(a);
    if aa == *a {
        LatticeCheck::pass("join_idempotent")
    } else {
        LatticeCheck::fail("join_idempotent", "a ⊔ a ≠ a")
    }
}

/// Verify that bottom is the identity for join.
pub fn check_bottom_identity<D: AbstractDomain>(a: &D) -> LatticeCheck {
    let bot = D::bottom();
    let result = a.join(&bot);
    if result == *a {
        LatticeCheck::pass("bottom_identity")
    } else {
        LatticeCheck::fail("bottom_identity", "a ⊔ ⊥ ≠ a")
    }
}

/// Verify that top absorbs join.
pub fn check_top_absorbing<D: AbstractDomain>(a: &D) -> LatticeCheck {
    let top = D::top();
    let result = a.join(&top);
    if result == top {
        LatticeCheck::pass("top_absorbing")
    } else {
        LatticeCheck::fail("top_absorbing", "a ⊔ ⊤ ≠ ⊤")
    }
}

/// Verify that partial_order is consistent with join.
pub fn check_order_join_consistent<D: AbstractDomain>(a: &D, b: &D) -> LatticeCheck {
    let joined = a.join(b);
    let a_le = a.partial_order(&joined);
    let b_le = b.partial_order(&joined);
    if a_le && b_le {
        LatticeCheck::pass("order_join_consistent")
    } else {
        LatticeCheck::fail(
            "order_join_consistent",
            "a ⊑ (a ⊔ b) or b ⊑ (a ⊔ b) violated",
        )
    }
}

/// Run all lattice axiom checks for two elements.
pub fn check_lattice_axioms<D: AbstractDomain>(a: &D, b: &D) -> Vec<LatticeCheck> {
    vec![
        check_join_commutative(a, b),
        check_join_idempotent(a),
        check_join_idempotent(b),
        check_bottom_identity(a),
        check_bottom_identity(b),
        check_top_absorbing(a),
        check_top_absorbing(b),
        check_order_join_consistent(a, b),
    ]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bool_domain_lattice_properties() {
        let bot = BoolDomain::unreachable();
        let top = BoolDomain::reachable();
        assert!(bot.is_bottom());
        assert!(top.is_top());
        assert_eq!(bot.join(&top), top);
        assert_eq!(top.meet(&bot), bot);
        assert!(bot.partial_order(&top));
        assert!(!top.partial_order(&bot));
    }

    #[test]
    fn bool_domain_finite() {
        assert_eq!(BoolDomain::height(), 1);
        assert_eq!(BoolDomain::unreachable().rank(), 0);
        assert_eq!(BoolDomain::reachable().rank(), 1);
    }

    #[test]
    fn flat_lattice_join() {
        let a: FlatLattice<i32> = FlatLattice::value(42);
        let b: FlatLattice<i32> = FlatLattice::value(42);
        let c: FlatLattice<i32> = FlatLattice::value(99);
        assert_eq!(a.join(&b), FlatLattice::value(42));
        assert_eq!(a.join(&c), FlatLattice::Top);
    }

    #[test]
    fn flat_lattice_bottom_top() {
        let bot: FlatLattice<i32> = FlatLattice::Bottom;
        let top: FlatLattice<i32> = FlatLattice::Top;
        let v: FlatLattice<i32> = FlatLattice::value(1);
        assert_eq!(bot.join(&v), v);
        assert_eq!(top.meet(&v), v);
        assert!(bot.partial_order(&v));
        assert!(v.partial_order(&top));
    }

    #[test]
    fn powerset_operations() {
        let a = PowersetDomain::from_iter(vec![1, 2, 3]);
        let b = PowersetDomain::from_iter(vec![2, 3, 4]);
        let u = a.union(&b);
        let i = a.intersection(&b);
        assert_eq!(u.len(), Some(4));
        assert_eq!(i.len(), Some(2));
        assert!(i.contains(&2));
        assert!(i.contains(&3));
        assert!(!i.contains(&1));
    }

    #[test]
    fn powerset_lattice_checks() {
        let a = PowersetDomain::from_iter(vec![1, 2]);
        let b = PowersetDomain::from_iter(vec![2, 3]);
        let checks = check_lattice_axioms(&a, &b);
        for c in &checks {
            assert!(c.holds, "Failed: {c}");
        }
    }

    #[test]
    fn product_domain_join() {
        let a = ProductDomain::new(BoolDomain::unreachable(), BoolDomain::reachable());
        let b = ProductDomain::new(BoolDomain::reachable(), BoolDomain::unreachable());
        let j = a.join(&b);
        assert!(j.first.is_top());
        assert!(j.second.is_top());
    }

    #[test]
    fn product_domain_bottom_top() {
        let bot: ProductDomain<BoolDomain, BoolDomain> = ProductDomain::bottom();
        let top: ProductDomain<BoolDomain, BoolDomain> = ProductDomain::top();
        assert!(bot.is_bottom());
        assert!(top.is_top());
        assert!(bot.partial_order(&top));
    }

    #[test]
    fn join_assign_returns_changed() {
        let mut a = BoolDomain::unreachable();
        assert!(a.join_assign(&BoolDomain::reachable()));
        assert!(!a.join_assign(&BoolDomain::reachable()));
    }

    #[test]
    fn lattice_check_display() {
        let pass = LatticeCheck::pass("test_prop");
        assert!(pass.to_string().contains('✓'));
        let fail = LatticeCheck::fail("test_prop", "oops");
        assert!(fail.to_string().contains('✗'));
    }

    #[test]
    fn powerset_universe() {
        let u: PowersetDomain<i32> = PowersetDomain::universe();
        assert!(u.is_top());
        assert!(u.contains(&42));
        let e: PowersetDomain<i32> = PowersetDomain::empty();
        assert!(e.is_bottom());
        assert!(e.partial_order(&u));
    }

    #[test]
    fn powerset_difference() {
        let a = PowersetDomain::from_iter(vec![1, 2, 3, 4]);
        let b = PowersetDomain::from_iter(vec![2, 4]);
        let d = a.difference(&b);
        assert_eq!(d.len(), Some(2));
        assert!(d.contains(&1));
        assert!(d.contains(&3));
    }

    #[test]
    fn flat_lattice_map() {
        let v: FlatLattice<i32> = FlatLattice::value(10);
        let doubled = v.map(|x| x * 2);
        assert_eq!(doubled.as_value(), Some(&20));
    }
}
