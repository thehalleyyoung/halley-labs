//! Generic fixpoint computation engine for CABER.
//!
//! Implements Kleene iteration, widening/narrowing, Tarski fixpoints,
//! and symbolic BDD-based fixpoint computation over arbitrary lattices.

use std::collections::HashMap;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Lattice trait
// ---------------------------------------------------------------------------

/// A complete lattice with distance metric for convergence detection.
pub trait Lattice {
    type Element: Clone + PartialEq;

    /// Bottom element ⊥.
    fn bottom(&self) -> Self::Element;

    /// Top element ⊤.
    fn top(&self) -> Self::Element;

    /// Least upper bound (join, ⊔).
    fn join(&self, a: &Self::Element, b: &Self::Element) -> Self::Element;

    /// Greatest lower bound (meet, ⊓).
    fn meet(&self, a: &Self::Element, b: &Self::Element) -> Self::Element;

    /// Partial order: a ≤ b.
    fn leq(&self, a: &Self::Element, b: &Self::Element) -> bool;

    /// Distance metric for convergence detection.
    fn distance(&self, a: &Self::Element, b: &Self::Element) -> f64;
}

// ---------------------------------------------------------------------------
// FixpointConfig
// ---------------------------------------------------------------------------

/// Configuration for fixpoint computations.
#[derive(Debug, Clone)]
pub struct FixpointConfig {
    /// Maximum number of Kleene iterations before giving up.
    pub max_iterations: usize,
    /// Convergence threshold – iteration stops when distance < epsilon.
    pub epsilon: f64,
    /// Whether to record every iterate in `history`.
    pub record_history: bool,
    /// Apply widening only after this many plain Kleene steps.
    pub widening_delay: usize,
    /// Number of narrowing refinement steps after widening converges.
    pub narrowing_iterations: usize,
}

impl Default for FixpointConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10_000,
            epsilon: 1e-10,
            record_history: false,
            widening_delay: 3,
            narrowing_iterations: 5,
        }
    }
}

// ---------------------------------------------------------------------------
// FixpointResult
// ---------------------------------------------------------------------------

/// Result of a fixpoint computation.
#[derive(Debug, Clone)]
pub struct FixpointResult<E> {
    /// The computed (approximate) fixpoint value.
    pub value: E,
    /// Whether the iteration converged within the configured tolerance.
    pub converged: bool,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Distance between the last two iterates.
    pub final_distance: f64,
    /// Wall-clock time spent in the computation (milliseconds).
    pub computation_time_ms: f64,
}

impl<E> FixpointResult<E> {
    /// Returns `true` when `final_distance` is below `threshold`.
    pub fn is_exact(&self, threshold: f64) -> bool {
        self.final_distance < threshold
    }
}

// ---------------------------------------------------------------------------
// FixpointEngine
// ---------------------------------------------------------------------------

/// Generic Kleene-style fixpoint engine parameterised over a lattice.
pub struct FixpointEngine<L: Lattice> {
    config: FixpointConfig,
    history: Vec<L::Element>,
    converged: bool,
    iterations: usize,
}

impl<L: Lattice> FixpointEngine<L> {
    pub fn new(config: FixpointConfig) -> Self {
        Self {
            config,
            history: Vec::new(),
            converged: false,
            iterations: 0,
        }
    }

    // -- accessors ----------------------------------------------------------

    pub fn iterations(&self) -> usize {
        self.iterations
    }

    pub fn converged(&self) -> bool {
        self.converged
    }

    pub fn history(&self) -> &[L::Element] {
        &self.history
    }

    // -- helpers ------------------------------------------------------------

    fn reset(&mut self) {
        self.history.clear();
        self.converged = false;
        self.iterations = 0;
    }

    fn record(&mut self, elem: &L::Element) {
        if self.config.record_history {
            self.history.push(elem.clone());
        }
    }

    // -- least fixpoint (Kleene from ⊥) ------------------------------------

    /// Compute the least fixpoint by Kleene iteration starting from ⊥.
    ///
    /// x₀ = ⊥,  x_{n+1} = f(x_n),  stop when distance(x_n, x_{n+1}) < ε.
    pub fn least_fixpoint<F>(&mut self, lattice: &L, f: F) -> FixpointResult<L::Element>
    where
        F: Fn(&L::Element) -> L::Element,
    {
        self.reset();
        let start = Instant::now();

        let mut current = lattice.bottom();
        self.record(&current);

        let mut dist = f64::INFINITY;

        for _i in 0..self.config.max_iterations {
            let next = f(&current);
            dist = lattice.distance(&current, &next);
            self.iterations += 1;
            self.record(&next);

            if dist < self.config.epsilon {
                self.converged = true;
                current = next;
                break;
            }
            current = next;
        }

        let elapsed = start.elapsed().as_secs_f64() * 1000.0;

        FixpointResult {
            value: current,
            converged: self.converged,
            iterations: self.iterations,
            final_distance: dist,
            computation_time_ms: elapsed,
        }
    }

    // -- greatest fixpoint (Kleene from ⊤) ---------------------------------

    /// Compute the greatest fixpoint by Kleene iteration starting from ⊤.
    pub fn greatest_fixpoint<F>(&mut self, lattice: &L, f: F) -> FixpointResult<L::Element>
    where
        F: Fn(&L::Element) -> L::Element,
    {
        self.reset();
        let start = Instant::now();

        let mut current = lattice.top();
        self.record(&current);

        let mut dist = f64::INFINITY;

        for _i in 0..self.config.max_iterations {
            let next = f(&current);
            dist = lattice.distance(&current, &next);
            self.iterations += 1;
            self.record(&next);

            if dist < self.config.epsilon {
                self.converged = true;
                current = next;
                break;
            }
            current = next;
        }

        let elapsed = start.elapsed().as_secs_f64() * 1000.0;

        FixpointResult {
            value: current,
            converged: self.converged,
            iterations: self.iterations,
            final_distance: dist,
            computation_time_ms: elapsed,
        }
    }

    // -- least fixpoint with widening --------------------------------------

    /// Least fixpoint with widening to accelerate convergence on infinite
    /// ascending chains.
    ///
    /// For the first `widening_delay` steps, plain Kleene iteration is used.
    /// After that, `x_{n+1} = widen(x_n, f(x_n))`.
    pub fn least_fixpoint_with_widening<F, W>(
        &mut self,
        lattice: &L,
        f: F,
        widen: W,
    ) -> FixpointResult<L::Element>
    where
        F: Fn(&L::Element) -> L::Element,
        W: Fn(&L::Element, &L::Element) -> L::Element,
    {
        self.reset();
        let start = Instant::now();

        let mut current = lattice.bottom();
        self.record(&current);

        let mut dist = f64::INFINITY;

        for i in 0..self.config.max_iterations {
            let fx = f(&current);
            let next = if i < self.config.widening_delay {
                fx
            } else {
                widen(&current, &fx)
            };
            dist = lattice.distance(&current, &next);
            self.iterations += 1;
            self.record(&next);

            if dist < self.config.epsilon {
                self.converged = true;
                current = next;
                break;
            }
            current = next;
        }

        let elapsed = start.elapsed().as_secs_f64() * 1000.0;

        FixpointResult {
            value: current,
            converged: self.converged,
            iterations: self.iterations,
            final_distance: dist,
            computation_time_ms: elapsed,
        }
    }

    // -- greatest fixpoint with narrowing ----------------------------------

    /// Greatest fixpoint with narrowing for precision recovery.
    ///
    /// First computes an over-approximation via Kleene from ⊤, then
    /// applies up to `narrowing_iterations` narrowing steps.
    pub fn greatest_fixpoint_with_narrowing<F, N>(
        &mut self,
        lattice: &L,
        f: F,
        narrow: N,
    ) -> FixpointResult<L::Element>
    where
        F: Fn(&L::Element) -> L::Element,
        N: Fn(&L::Element, &L::Element) -> L::Element,
    {
        self.reset();
        let start = Instant::now();

        // Phase 1: descending Kleene from ⊤
        let mut current = lattice.top();
        self.record(&current);
        let mut dist = f64::INFINITY;

        for _i in 0..self.config.max_iterations {
            let next = f(&current);
            dist = lattice.distance(&current, &next);
            self.iterations += 1;
            self.record(&next);

            if dist < self.config.epsilon {
                self.converged = true;
                current = next;
                break;
            }
            current = next;
        }

        // Phase 2: narrowing refinement
        for _i in 0..self.config.narrowing_iterations {
            let fx = f(&current);
            let next = narrow(&current, &fx);
            dist = lattice.distance(&current, &next);
            self.iterations += 1;
            self.record(&next);

            if dist < self.config.epsilon {
                break;
            }
            current = next;
        }

        let elapsed = start.elapsed().as_secs_f64() * 1000.0;

        FixpointResult {
            value: current,
            converged: self.converged,
            iterations: self.iterations,
            final_distance: dist,
            computation_time_ms: elapsed,
        }
    }
}

// ===========================================================================
// Concrete lattice implementations
// ===========================================================================

// ---------------------------------------------------------------------------
// 3a. BooleanLattice  –  {false ≤ true}
// ---------------------------------------------------------------------------

/// Two-element Boolean lattice: false (⊥) ≤ true (⊤).
#[derive(Debug, Clone)]
pub struct BooleanLattice;

impl Lattice for BooleanLattice {
    type Element = bool;

    fn bottom(&self) -> bool {
        false
    }
    fn top(&self) -> bool {
        true
    }
    fn join(&self, a: &bool, b: &bool) -> bool {
        *a || *b
    }
    fn meet(&self, a: &bool, b: &bool) -> bool {
        *a && *b
    }
    fn leq(&self, a: &bool, b: &bool) -> bool {
        // false ≤ false, false ≤ true, true ≤ true
        !a || *b
    }
    fn distance(&self, a: &bool, b: &bool) -> f64 {
        if a == b {
            0.0
        } else {
            1.0
        }
    }
}

// ---------------------------------------------------------------------------
// 3b. UnitIntervalLattice  –  [0, 1]
// ---------------------------------------------------------------------------

/// The real interval [0, 1] with standard ordering, join = max, meet = min.
#[derive(Debug, Clone)]
pub struct UnitIntervalLattice;

impl Lattice for UnitIntervalLattice {
    type Element = f64;

    fn bottom(&self) -> f64 {
        0.0
    }
    fn top(&self) -> f64 {
        1.0
    }
    fn join(&self, a: &f64, b: &f64) -> f64 {
        a.max(*b)
    }
    fn meet(&self, a: &f64, b: &f64) -> f64 {
        a.min(*b)
    }
    fn leq(&self, a: &f64, b: &f64) -> bool {
        *a <= *b + f64::EPSILON
    }
    fn distance(&self, a: &f64, b: &f64) -> f64 {
        (a - b).abs()
    }
}

// ---------------------------------------------------------------------------
// 3c. PowerSetLattice  –  2^{0..n-1}  represented as Vec<bool>
// ---------------------------------------------------------------------------

/// Power-set lattice over {0, …, n−1} with ⊆ ordering.
/// Elements are represented as characteristic vectors (`Vec<bool>`).
#[derive(Debug, Clone)]
pub struct PowerSetLattice {
    pub size: usize,
}

impl PowerSetLattice {
    pub fn new(size: usize) -> Self {
        Self { size }
    }
}

impl Lattice for PowerSetLattice {
    type Element = Vec<bool>;

    fn bottom(&self) -> Vec<bool> {
        vec![false; self.size]
    }

    fn top(&self) -> Vec<bool> {
        vec![true; self.size]
    }

    fn join(&self, a: &Vec<bool>, b: &Vec<bool>) -> Vec<bool> {
        a.iter().zip(b.iter()).map(|(x, y)| *x || *y).collect()
    }

    fn meet(&self, a: &Vec<bool>, b: &Vec<bool>) -> Vec<bool> {
        a.iter().zip(b.iter()).map(|(x, y)| *x && *y).collect()
    }

    fn leq(&self, a: &Vec<bool>, b: &Vec<bool>) -> bool {
        a.iter().zip(b.iter()).all(|(x, y)| !x || *y)
    }

    fn distance(&self, a: &Vec<bool>, b: &Vec<bool>) -> f64 {
        a.iter()
            .zip(b.iter())
            .filter(|(x, y)| x != y)
            .count() as f64
    }
}

// ---------------------------------------------------------------------------
// 3d. ProductLattice<L1, L2>
// ---------------------------------------------------------------------------

/// Product lattice of two lattices with component-wise operations.
#[derive(Debug, Clone)]
pub struct ProductLattice<L1, L2> {
    pub left: L1,
    pub right: L2,
}

impl<L1, L2> ProductLattice<L1, L2> {
    pub fn new(left: L1, right: L2) -> Self {
        Self { left, right }
    }
}

impl<L1: Lattice, L2: Lattice> Lattice for ProductLattice<L1, L2> {
    type Element = (L1::Element, L2::Element);

    fn bottom(&self) -> Self::Element {
        (self.left.bottom(), self.right.bottom())
    }

    fn top(&self) -> Self::Element {
        (self.left.top(), self.right.top())
    }

    fn join(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        (
            self.left.join(&a.0, &b.0),
            self.right.join(&a.1, &b.1),
        )
    }

    fn meet(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        (
            self.left.meet(&a.0, &b.0),
            self.right.meet(&a.1, &b.1),
        )
    }

    fn leq(&self, a: &Self::Element, b: &Self::Element) -> bool {
        self.left.leq(&a.0, &b.0) && self.right.leq(&a.1, &b.1)
    }

    fn distance(&self, a: &Self::Element, b: &Self::Element) -> f64 {
        let d1 = self.left.distance(&a.0, &b.0);
        let d2 = self.right.distance(&a.1, &b.1);
        d1.max(d2)
    }
}

// ---------------------------------------------------------------------------
// 3e. VectorLattice  –  [0,1]^n  component-wise
// ---------------------------------------------------------------------------

/// n-dimensional vector lattice over [0, 1]^n with component-wise max/min.
#[derive(Debug, Clone)]
pub struct VectorLattice {
    pub dim: usize,
}

impl VectorLattice {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl Lattice for VectorLattice {
    type Element = Vec<f64>;

    fn bottom(&self) -> Vec<f64> {
        vec![0.0; self.dim]
    }

    fn top(&self) -> Vec<f64> {
        vec![1.0; self.dim]
    }

    fn join(&self, a: &Vec<f64>, b: &Vec<f64>) -> Vec<f64> {
        a.iter().zip(b.iter()).map(|(x, y)| x.max(*y)).collect()
    }

    fn meet(&self, a: &Vec<f64>, b: &Vec<f64>) -> Vec<f64> {
        a.iter().zip(b.iter()).map(|(x, y)| x.min(*y)).collect()
    }

    fn leq(&self, a: &Vec<f64>, b: &Vec<f64>) -> bool {
        a.iter()
            .zip(b.iter())
            .all(|(x, y)| *x <= *y + f64::EPSILON)
    }

    fn distance(&self, a: &Vec<f64>, b: &Vec<f64>) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f64, f64::max)
    }
}

// ---------------------------------------------------------------------------
// 3f. FlatLattice<T>  –  {⊥} ∪ T ∪ {⊤}
// ---------------------------------------------------------------------------

/// Flat lattice: all elements of T are incomparable; ⊥ < every t ∈ T < ⊤.
#[derive(Debug, Clone, PartialEq)]
pub enum FlatElement<T: Clone + PartialEq> {
    Bottom,
    Value(T),
    Top,
}

/// A flat lattice over a finite set of values.
/// `values` lists all elements of T (used for enumeration).
#[derive(Debug, Clone)]
pub struct FlatLattice<T: Clone + PartialEq> {
    pub values: Vec<T>,
}

impl<T: Clone + PartialEq> FlatLattice<T> {
    pub fn new(values: Vec<T>) -> Self {
        Self { values }
    }
}

impl<T: Clone + PartialEq> Lattice for FlatLattice<T> {
    type Element = FlatElement<T>;

    fn bottom(&self) -> FlatElement<T> {
        FlatElement::Bottom
    }

    fn top(&self) -> FlatElement<T> {
        FlatElement::Top
    }

    fn join(&self, a: &FlatElement<T>, b: &FlatElement<T>) -> FlatElement<T> {
        match (a, b) {
            (FlatElement::Bottom, x) | (x, FlatElement::Bottom) => x.clone(),
            (FlatElement::Top, _) | (_, FlatElement::Top) => FlatElement::Top,
            (FlatElement::Value(x), FlatElement::Value(y)) => {
                if x == y {
                    FlatElement::Value(x.clone())
                } else {
                    FlatElement::Top
                }
            }
        }
    }

    fn meet(&self, a: &FlatElement<T>, b: &FlatElement<T>) -> FlatElement<T> {
        match (a, b) {
            (FlatElement::Top, x) | (x, FlatElement::Top) => x.clone(),
            (FlatElement::Bottom, _) | (_, FlatElement::Bottom) => FlatElement::Bottom,
            (FlatElement::Value(x), FlatElement::Value(y)) => {
                if x == y {
                    FlatElement::Value(x.clone())
                } else {
                    FlatElement::Bottom
                }
            }
        }
    }

    fn leq(&self, a: &FlatElement<T>, b: &FlatElement<T>) -> bool {
        match (a, b) {
            (FlatElement::Bottom, _) => true,
            (_, FlatElement::Top) => true,
            (FlatElement::Value(x), FlatElement::Value(y)) => x == y,
            _ => false,
        }
    }

    fn distance(&self, a: &FlatElement<T>, b: &FlatElement<T>) -> f64 {
        if a == b {
            0.0
        } else {
            1.0
        }
    }
}

// ---------------------------------------------------------------------------
// 3g. IntervalLattice  –  intervals [a,b] ⊆ [0,1], ordered by ⊇
// ---------------------------------------------------------------------------

/// Closed sub-intervals of [0, 1] ordered by *reverse inclusion* (⊇): a
/// more-precise (narrower) interval is *higher* in the lattice.
///
/// ⊥ = [0, 1]  (least information),  ⊤ = empty interval (inconsistency).
/// We represent the empty interval with `lo > hi`.
#[derive(Debug, Clone, PartialEq)]
pub struct Interval {
    pub lo: f64,
    pub hi: f64,
}

impl Interval {
    pub fn new(lo: f64, hi: f64) -> Self {
        Self { lo, hi }
    }

    pub fn is_empty(&self) -> bool {
        self.lo > self.hi
    }

    pub fn width(&self) -> f64 {
        if self.is_empty() {
            0.0
        } else {
            self.hi - self.lo
        }
    }
}

/// Lattice of intervals [a, b] ⊆ [0, 1] ordered by reverse inclusion.
#[derive(Debug, Clone)]
pub struct IntervalLattice;

impl Lattice for IntervalLattice {
    type Element = Interval;

    /// ⊥ = [0, 1] – widest interval, least precise.
    fn bottom(&self) -> Interval {
        Interval::new(0.0, 1.0)
    }

    /// ⊤ = empty interval (represented by lo > hi).
    fn top(&self) -> Interval {
        Interval::new(1.0, 0.0) // empty
    }

    /// Join in reverse-inclusion order = intersection (more precise).
    fn join(&self, a: &Interval, b: &Interval) -> Interval {
        if a.is_empty() || b.is_empty() {
            return self.top();
        }
        let lo = a.lo.max(b.lo);
        let hi = a.hi.min(b.hi);
        Interval::new(lo, hi) // may be empty
    }

    /// Meet in reverse-inclusion order = hull / union of intervals.
    fn meet(&self, a: &Interval, b: &Interval) -> Interval {
        if a.is_empty() {
            return b.clone();
        }
        if b.is_empty() {
            return a.clone();
        }
        Interval::new(a.lo.min(b.lo), a.hi.max(b.hi))
    }

    /// a ≤ b  iff  b ⊆ a  (reverse inclusion).
    fn leq(&self, a: &Interval, b: &Interval) -> bool {
        if b.is_empty() {
            return true; // everything ≤ ⊤
        }
        if a.is_empty() {
            return false; // ⊤ ≤ non-⊤ only if b is also ⊤
        }
        // b ⊆ a  ⟺  a.lo ≤ b.lo ∧ b.hi ≤ a.hi
        a.lo <= b.lo + f64::EPSILON && b.hi <= a.hi + f64::EPSILON
    }

    fn distance(&self, a: &Interval, b: &Interval) -> f64 {
        if a.is_empty() && b.is_empty() {
            return 0.0;
        }
        if a.is_empty() || b.is_empty() {
            return 1.0;
        }
        (a.lo - b.lo).abs().max((a.hi - b.hi).abs())
    }
}

// ===========================================================================
// 6. TarskiFixpoint
// ===========================================================================

/// Computes fixpoints via Tarski's theorem.
///
/// For *small* lattices (≤ 20 Boolean variables in a power-set) the least
/// fixpoint is obtained by enumerating all pre-fixpoints {x | f(x) ≤ x} and
/// taking their meet.  For large lattices we fall back to Kleene iteration.
pub struct TarskiFixpoint;

impl TarskiFixpoint {
    /// Compute the least fixpoint of `f` on `lattice`.
    pub fn compute<L: Lattice, F>(lattice: &L, f: F) -> L::Element
    where
        F: Fn(&L::Element) -> L::Element,
    {
        // Fall back to Kleene iteration with generous budget.
        let mut current = lattice.bottom();
        for _ in 0..100_000 {
            let next = f(&current);
            if lattice.distance(&current, &next) < 1e-12 {
                return next;
            }
            current = next;
        }
        current
    }
}

/// Specialised Tarski computation for `PowerSetLattice` that enumerates
/// all pre-fixpoints when the universe is small enough (≤ 20).
pub struct TarskiPowerSet;

impl TarskiPowerSet {
    /// Enumerate all subsets, keep those that are pre-fixpoints, return their
    /// intersection (= meet in the power-set lattice).
    pub fn least_fixpoint<F>(lattice: &PowerSetLattice, f: F) -> Vec<bool>
    where
        F: Fn(&Vec<bool>) -> Vec<bool>,
    {
        let n = lattice.size;
        if n > 20 {
            // Too large to enumerate – fall back to Kleene.
            return TarskiFixpoint::compute(lattice, f);
        }

        let total = 1u64 << n;
        let mut result = lattice.top(); // start with everything true

        for bits in 0..total {
            let subset: Vec<bool> = (0..n).map(|i| (bits >> i) & 1 == 1).collect();
            let image = f(&subset);
            // Check pre-fixpoint: f(x) ⊆ x
            let is_pre = lattice.leq(&image, &subset);
            if is_pre {
                result = lattice.meet(&result, &subset);
            }
        }

        result
    }
}

// ===========================================================================
// 7–8. BDDSet & BDDFixpoint  (simplified symbolic representation)
// ===========================================================================

/// Simplified BDD representation: a set of Boolean valuations (rows of a
/// truth table).  Each valuation is a `Vec<bool>` of length `num_vars`.
#[derive(Debug, Clone, PartialEq)]
pub struct BDDSet {
    pub elements: Vec<Vec<bool>>,
    pub num_vars: usize,
}

impl BDDSet {
    /// Empty set.
    pub fn empty(n: usize) -> Self {
        Self {
            elements: Vec::new(),
            num_vars: n,
        }
    }

    /// Full set containing all 2^n valuations.
    pub fn full(n: usize) -> Self {
        let total = 1u64 << n;
        let elements: Vec<Vec<bool>> = (0..total)
            .map(|bits| (0..n).map(|i| (bits >> i) & 1 == 1).collect())
            .collect();
        Self {
            elements,
            num_vars: n,
        }
    }

    /// Set union.
    pub fn union(&self, other: &BDDSet) -> BDDSet {
        let mut elems = self.elements.clone();
        for e in &other.elements {
            if !elems.contains(e) {
                elems.push(e.clone());
            }
        }
        elems.sort();
        BDDSet {
            elements: elems,
            num_vars: self.num_vars,
        }
    }

    /// Set intersection.
    pub fn intersection(&self, other: &BDDSet) -> BDDSet {
        let elems: Vec<Vec<bool>> = self
            .elements
            .iter()
            .filter(|e| other.elements.contains(e))
            .cloned()
            .collect();
        BDDSet {
            elements: elems,
            num_vars: self.num_vars,
        }
    }

    /// Set complement (relative to the full set of valuations).
    pub fn complement(&self) -> BDDSet {
        let full = BDDSet::full(self.num_vars);
        let elems: Vec<Vec<bool>> = full
            .elements
            .into_iter()
            .filter(|e| !self.elements.contains(e))
            .collect();
        BDDSet {
            elements: elems,
            num_vars: self.num_vars,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    pub fn contains(&self, elem: &Vec<bool>) -> bool {
        self.elements.contains(elem)
    }

    pub fn size(&self) -> usize {
        self.elements.len()
    }
}

/// Symbolic fixpoint computation using simplified BDD (set-of-valuations)
/// representation.
pub struct BDDFixpoint {
    pub variables: usize,
}

impl BDDFixpoint {
    pub fn new(num_variables: usize) -> Self {
        Self {
            variables: num_variables,
        }
    }

    /// Least fixpoint: iterate from ∅ applying f until stable.
    pub fn least_fixpoint<F>(&self, f: F) -> BDDSet
    where
        F: Fn(&BDDSet) -> BDDSet,
    {
        let mut current = BDDSet::empty(self.variables);
        loop {
            let next = f(&current);
            let merged = current.union(&next);
            if merged.size() == current.size() && merged == current {
                return current;
            }
            current = merged;
        }
    }

    /// Greatest fixpoint: iterate from the full set applying f until stable.
    pub fn greatest_fixpoint<F>(&self, f: F) -> BDDSet
    where
        F: Fn(&BDDSet) -> BDDSet,
    {
        let mut current = BDDSet::full(self.variables);
        loop {
            let next = f(&current);
            let narrowed = current.intersection(&next);
            if narrowed.size() == current.size() && narrowed == current {
                return current;
            }
            current = narrowed;
        }
    }
}

// ===========================================================================
// 9. FixpointCache
// ===========================================================================

/// Simple memoisation cache keyed by formula/expression strings.
pub struct FixpointCache {
    cache: HashMap<String, Vec<f64>>,
}

impl FixpointCache {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    pub fn get(&self, key: &str) -> Option<&Vec<f64>> {
        self.cache.get(key)
    }

    pub fn insert(&mut self, key: &str, value: Vec<f64>) {
        self.cache.insert(key.to_string(), value);
    }

    pub fn has(&self, key: &str) -> bool {
        self.cache.contains_key(key)
    }

    pub fn clear(&mut self) {
        self.cache.clear();
    }

    pub fn size(&self) -> usize {
        self.cache.len()
    }
}

impl Default for FixpointCache {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// 10. Widening operators
// ===========================================================================

/// Interval widening: if a bound moved outward, push it to ±∞ (here ±1e308).
pub fn interval_widening(old: &(f64, f64), new: &(f64, f64)) -> (f64, f64) {
    let lo = if new.0 < old.0 { f64::NEG_INFINITY } else { old.0 };
    let hi = if new.1 > old.1 { f64::INFINITY } else { old.1 };
    (lo, hi)
}

/// Threshold widening: if `new > old`, jump to the next threshold ≥ new.
pub fn threshold_widening(old: f64, new: f64, thresholds: &[f64]) -> f64 {
    if new <= old {
        return old;
    }
    // Find the smallest threshold ≥ new.
    let mut best: Option<f64> = None;
    for &t in thresholds {
        if t >= new {
            match best {
                None => best = Some(t),
                Some(b) if t < b => best = Some(t),
                _ => {}
            }
        }
    }
    best.unwrap_or(f64::INFINITY)
}

/// Component-wise widening for vectors.
pub fn vector_widening(old: &[f64], new: &[f64]) -> Vec<f64> {
    old.iter()
        .zip(new.iter())
        .map(|(&o, &n)| {
            if n > o {
                f64::INFINITY
            } else if n < o {
                f64::NEG_INFINITY
            } else {
                o
            }
        })
        .collect()
}

// ===========================================================================
// 11. Narrowing operators
// ===========================================================================

/// Interval narrowing: if a bound was at ±∞, replace it with the precise bound.
pub fn interval_narrowing(wide: &(f64, f64), precise: &(f64, f64)) -> (f64, f64) {
    let lo = if wide.0 == f64::NEG_INFINITY {
        precise.0
    } else {
        wide.0
    };
    let hi = if wide.1 == f64::INFINITY {
        precise.1
    } else {
        wide.1
    };
    (lo, hi)
}

/// Component-wise narrowing for vectors.
pub fn vector_narrowing(wide: &[f64], precise: &[f64]) -> Vec<f64> {
    wide.iter()
        .zip(precise.iter())
        .map(|(&w, &p)| {
            if w == f64::INFINITY || w == f64::NEG_INFINITY {
                p
            } else {
                w
            }
        })
        .collect()
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // 1. Least fixpoint of x -> x/2 + 0.5 on [0,1] converges to 1.0
    #[test]
    fn test_lfp_unit_interval_converges_to_one() {
        let lattice = UnitIntervalLattice;
        let config = FixpointConfig::default();
        let mut engine = FixpointEngine::new(config);
        let result = engine.least_fixpoint(&lattice, |x| x / 2.0 + 0.5);
        assert!(result.converged);
        assert!((result.value - 1.0).abs() < 1e-6);
    }

    // 2. Greatest fixpoint of x -> x/2 on [0,1] converges to 0.0
    #[test]
    fn test_gfp_unit_interval_converges_to_zero() {
        let lattice = UnitIntervalLattice;
        let config = FixpointConfig::default();
        let mut engine = FixpointEngine::new(config);
        let result = engine.greatest_fixpoint(&lattice, |x| x / 2.0);
        assert!(result.converged);
        assert!(result.value.abs() < 1e-6);
    }

    // 3. Least fixpoint on PowerSetLattice (reachability)
    #[test]
    fn test_lfp_powerset_reachability() {
        // Graph: 0 -> 1, 1 -> 2.  Start = {0}.  Reachable = {0,1,2}.
        let lattice = PowerSetLattice::new(4);
        let config = FixpointConfig::default();
        let mut engine = FixpointEngine::new(config);
        let result = engine.least_fixpoint(&lattice, |s| {
            let mut next = s.clone();
            // initial set contains 0
            next[0] = true;
            // edge 0 -> 1
            if s[0] {
                next[1] = true;
            }
            // edge 1 -> 2
            if s[1] {
                next[2] = true;
            }
            next
        });
        assert!(result.converged);
        assert_eq!(result.value, vec![true, true, true, false]);
    }

    // 4. Boolean lattice fixpoints
    #[test]
    fn test_boolean_lfp_identity() {
        let lattice = BooleanLattice;
        let mut engine = FixpointEngine::new(FixpointConfig::default());
        // f(x) = x has lfp = false (bottom)
        let result = engine.least_fixpoint(&lattice, |x| *x);
        assert!(result.converged);
        assert!(!result.value);
    }

    #[test]
    fn test_boolean_gfp_identity() {
        let lattice = BooleanLattice;
        let mut engine = FixpointEngine::new(FixpointConfig::default());
        // f(x) = x has gfp = true (top)
        let result = engine.greatest_fixpoint(&lattice, |x| *x);
        assert!(result.converged);
        assert!(result.value);
    }

    // 5. Product lattice operations
    #[test]
    fn test_product_lattice_basic() {
        let lat = ProductLattice::new(BooleanLattice, UnitIntervalLattice);
        let bot = lat.bottom();
        assert_eq!(bot, (false, 0.0));
        let top = lat.top();
        assert_eq!(top, (true, 1.0));
        let j = lat.join(&(false, 0.3), &(true, 0.7));
        assert_eq!(j, (true, 0.7));
        let m = lat.meet(&(true, 0.3), &(true, 0.7));
        assert_eq!(m, (true, 0.3));
        assert!(lat.leq(&bot, &top));
    }

    // 6. Vector lattice fixpoints (simultaneous equations)
    #[test]
    fn test_vector_lattice_simultaneous() {
        // Solve x1 = 0.5*x2 + 0.25, x2 = 0.5*x1 + 0.25
        // Solution: x1 = x2 = 0.5
        let lattice = VectorLattice::new(2);
        let mut engine = FixpointEngine::new(FixpointConfig::default());
        let result = engine.least_fixpoint(&lattice, |v| {
            vec![
                (0.5 * v[1] + 0.25).min(1.0),
                (0.5 * v[0] + 0.25).min(1.0),
            ]
        });
        assert!(result.converged);
        assert!((result.value[0] - 0.5).abs() < 1e-6);
        assert!((result.value[1] - 0.5).abs() < 1e-6);
    }

    // 7. Widening acceleration
    #[test]
    fn test_widening_interval() {
        let old = (0.0, 5.0);
        let new = (-1.0, 7.0);
        let w = interval_widening(&old, &new);
        assert_eq!(w.0, f64::NEG_INFINITY);
        assert_eq!(w.1, f64::INFINITY);

        // No change when bounds don't grow.
        let w2 = interval_widening(&old, &(0.0, 3.0));
        assert_eq!(w2, (0.0, 5.0));
    }

    #[test]
    fn test_threshold_widening() {
        let thresholds = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        assert_eq!(threshold_widening(0.0, 0.3, &thresholds), 0.5);
        assert_eq!(threshold_widening(0.5, 0.6, &thresholds), 0.75);
        assert_eq!(threshold_widening(0.5, 0.3, &thresholds), 0.5); // no change
    }

    // 8. Narrowing precision recovery
    #[test]
    fn test_narrowing_interval() {
        let wide = (f64::NEG_INFINITY, f64::INFINITY);
        let precise = (1.0, 5.0);
        let n = interval_narrowing(&wide, &precise);
        assert_eq!(n, (1.0, 5.0));

        let partially_wide = (0.0, f64::INFINITY);
        let n2 = interval_narrowing(&partially_wide, &precise);
        assert_eq!(n2, (0.0, 5.0));
    }

    #[test]
    fn test_vector_narrowing() {
        let wide = vec![f64::NEG_INFINITY, 3.0, f64::INFINITY];
        let precise = vec![1.0, 2.0, 5.0];
        let n = vector_narrowing(&wide, &precise);
        assert_eq!(n, vec![1.0, 3.0, 5.0]); // finite components kept from wide
    }

    // 9. Tarski fixpoint on small lattice
    #[test]
    fn test_tarski_powerset() {
        // f(S) = {0} ∪ {i+1 | i ∈ S, i+1 < n}. Fixpoint = {0,1,2}.
        let lattice = PowerSetLattice::new(3);
        let result = TarskiPowerSet::least_fixpoint(&lattice, |s| {
            let mut next = vec![false; 3];
            next[0] = true;
            for i in 0..2 {
                if s[i] {
                    next[i + 1] = true;
                }
            }
            next
        });
        assert_eq!(result, vec![true, true, true]);
    }

    #[test]
    fn test_tarski_unit_interval() {
        // f(x) = x/2 + 0.5 on [0,1].  Fixpoint = 1.0.
        let lattice = UnitIntervalLattice;
        let result = TarskiFixpoint::compute(&lattice, |x| x / 2.0 + 0.5);
        assert!((result - 1.0).abs() < 1e-6);
    }

    // 10. BDD fixpoint operations
    #[test]
    fn test_bdd_set_operations() {
        let a = BDDSet {
            elements: vec![vec![false, false], vec![true, false]],
            num_vars: 2,
        };
        let b = BDDSet {
            elements: vec![vec![true, false], vec![true, true]],
            num_vars: 2,
        };
        let u = a.union(&b);
        assert_eq!(u.size(), 3);
        let i = a.intersection(&b);
        assert_eq!(i.size(), 1);
        assert!(i.contains(&vec![true, false]));

        let c = a.complement();
        assert_eq!(c.size(), 2); // full has 4, a has 2
        assert!(!c.contains(&vec![false, false]));
    }

    #[test]
    fn test_bdd_least_fixpoint() {
        let bdd = BDDFixpoint::new(2);
        // f(S) = S ∪ {(false,false)} – lfp is { (false,false) }
        let result = bdd.least_fixpoint(|s| {
            let singleton = BDDSet {
                elements: vec![vec![false, false]],
                num_vars: 2,
            };
            s.union(&singleton)
        });
        assert!(result.contains(&vec![false, false]));
        assert_eq!(result.size(), 1);
    }

    #[test]
    fn test_bdd_greatest_fixpoint() {
        let bdd = BDDFixpoint::new(2);
        // f(S) = S ∩ { valuations where var0 is true }
        let result = bdd.greatest_fixpoint(|s| {
            let with_var0 = BDDSet {
                elements: vec![vec![true, false], vec![true, true]],
                num_vars: 2,
            };
            s.intersection(&with_var0)
        });
        assert_eq!(result.size(), 2);
        assert!(result.contains(&vec![true, false]));
        assert!(result.contains(&vec![true, true]));
    }

    // 11. Fixpoint cache
    #[test]
    fn test_fixpoint_cache() {
        let mut cache = FixpointCache::new();
        assert_eq!(cache.size(), 0);
        assert!(!cache.has("mu X. phi"));

        cache.insert("mu X. phi", vec![0.5, 0.7]);
        assert!(cache.has("mu X. phi"));
        assert_eq!(cache.get("mu X. phi"), Some(&vec![0.5, 0.7]));
        assert_eq!(cache.size(), 1);

        cache.clear();
        assert_eq!(cache.size(), 0);
    }

    // 12. FlatLattice operations
    #[test]
    fn test_flat_lattice() {
        let lat = FlatLattice::new(vec![1, 2, 3]);
        assert_eq!(lat.bottom(), FlatElement::Bottom);
        assert_eq!(lat.top(), FlatElement::Top);
        assert!(lat.leq(&FlatElement::Bottom, &FlatElement::Value(1)));
        assert!(lat.leq(&FlatElement::Value(2), &FlatElement::Top));
        assert!(lat.leq(&FlatElement::Value(2), &FlatElement::Value(2)));
        assert!(!lat.leq(&FlatElement::Value(1), &FlatElement::Value(2)));

        // join of incomparable values = ⊤
        assert_eq!(
            lat.join(&FlatElement::Value(1), &FlatElement::Value(2)),
            FlatElement::Top
        );
        // meet of incomparable values = ⊥
        assert_eq!(
            lat.meet(&FlatElement::Value(1), &FlatElement::Value(2)),
            FlatElement::Bottom
        );
        // join of identical values
        assert_eq!(
            lat.join(&FlatElement::Value(1), &FlatElement::Value(1)),
            FlatElement::Value(1)
        );
    }

    // 13. IntervalLattice operations
    #[test]
    fn test_interval_lattice_operations() {
        let lat = IntervalLattice;
        let bot = lat.bottom(); // [0,1]
        let top = lat.top(); // empty

        assert!(lat.leq(&bot, &top)); // [0,1] ≤ ⊤
        assert!(lat.leq(&bot, &Interval::new(0.2, 0.8))); // [0,1] ≤ [0.2,0.8]
        assert!(!lat.leq(&Interval::new(0.2, 0.8), &bot)); // [0.2,0.8] ≰ [0,1]

        // join = intersection
        let j = lat.join(&Interval::new(0.0, 0.7), &Interval::new(0.3, 1.0));
        assert!((j.lo - 0.3).abs() < 1e-10);
        assert!((j.hi - 0.7).abs() < 1e-10);

        // meet = hull
        let m = lat.meet(&Interval::new(0.2, 0.4), &Interval::new(0.6, 0.8));
        assert!((m.lo - 0.2).abs() < 1e-10);
        assert!((m.hi - 0.8).abs() < 1e-10);
    }

    // 14. Convergence detection
    #[test]
    fn test_convergence_detection() {
        let lattice = UnitIntervalLattice;
        let config = FixpointConfig {
            max_iterations: 5,
            epsilon: 1e-10,
            ..FixpointConfig::default()
        };
        let mut engine = FixpointEngine::new(config);
        // f(x) = 0.9*x + 0.1 has fixpoint 1.0 but needs many iterations
        let result = engine.least_fixpoint(&lattice, |x| 0.9 * x + 0.1);
        // 5 iterations is not enough for 1e-10 tolerance
        assert!(!result.converged);
        assert_eq!(result.iterations, 5);
    }

    // 15. History recording
    #[test]
    fn test_history_recording() {
        let lattice = BooleanLattice;
        let config = FixpointConfig {
            record_history: true,
            ..FixpointConfig::default()
        };
        let mut engine = FixpointEngine::new(config);
        let _ = engine.least_fixpoint(&lattice, |_| true);
        let history = engine.history();
        // history should contain at least the initial value and final value
        assert!(history.len() >= 2);
        assert!(!history[0]); // starts from bottom = false
    }

    // 16. Widening on unit interval engine
    #[test]
    fn test_engine_widening() {
        let lattice = UnitIntervalLattice;
        let config = FixpointConfig {
            widening_delay: 0,
            ..FixpointConfig::default()
        };
        let mut engine = FixpointEngine::new(config);
        // f(x) = min(x + 0.1, 1.0) – widening should accelerate.
        let result = engine.least_fixpoint_with_widening(
            &lattice,
            |x| (x + 0.1).min(1.0),
            |old, new| {
                if *new > *old {
                    1.0 // jump straight to top
                } else {
                    *old
                }
            },
        );
        assert!(result.converged);
        assert!((result.value - 1.0).abs() < 1e-6);
        // Widening should converge faster than plain Kleene
        assert!(result.iterations <= 3);
    }

    // 17. Greatest fixpoint with narrowing
    #[test]
    fn test_engine_narrowing() {
        let lattice = UnitIntervalLattice;
        let config = FixpointConfig {
            narrowing_iterations: 10,
            ..FixpointConfig::default()
        };
        let mut engine = FixpointEngine::new(config);
        // f(x) = x * 0.5; gfp = 0. Narrowing should not hurt.
        let result = engine.greatest_fixpoint_with_narrowing(
            &lattice,
            |x| x * 0.5,
            |_wide, precise| *precise,
        );
        assert!(result.value.abs() < 1e-6);
    }

    // 18. FixpointResult::is_exact
    #[test]
    fn test_fixpoint_result_is_exact() {
        let r = FixpointResult {
            value: 1.0,
            converged: true,
            iterations: 10,
            final_distance: 1e-12,
            computation_time_ms: 0.5,
        };
        assert!(r.is_exact(1e-10));
        assert!(!r.is_exact(1e-13));
    }

    // 19. Vector widening operator
    #[test]
    fn test_vector_widening() {
        let old = vec![1.0, 2.0, 3.0];
        let new = vec![1.0, 3.0, 2.0];
        let w = vector_widening(&old, &new);
        assert_eq!(w[0], 1.0); // unchanged
        assert_eq!(w[1], f64::INFINITY); // grew
        assert_eq!(w[2], f64::NEG_INFINITY); // shrank
    }

    // 20. Interval lattice distance
    #[test]
    fn test_interval_distance() {
        let lat = IntervalLattice;
        assert_eq!(
            lat.distance(&Interval::new(0.0, 1.0), &Interval::new(0.0, 1.0)),
            0.0
        );
        assert!(
            (lat.distance(&Interval::new(0.0, 1.0), &Interval::new(0.1, 0.9)) - 0.1).abs()
                < 1e-10
        );
        // Distance involving empty interval
        assert_eq!(
            lat.distance(&lat.top(), &Interval::new(0.0, 1.0)),
            1.0
        );
    }
}
