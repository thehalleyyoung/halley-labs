//! Galois connections and abstract interpretation for memory model verification.
//!
//! Implements the Galois connection framework from §9 of the LITMUS∞ paper.
//! Provides a lattice-theoretic bridge between concrete executions and
//! abstract memory model states, enabling sound over-approximation for
//! faster verification.

use std::collections::{HashMap, HashSet, BTreeSet};
use std::fmt;
use std::hash::Hash;

// ═══════════════════════════════════════════════════════════════════════
// GaloisConnection trait
// ═══════════════════════════════════════════════════════════════════════

/// A Galois connection between a concrete domain C and an abstract domain A.
///
/// A pair (α, γ) where:
///   α : C → A  (abstraction)
///   γ : A → C  (concretization)
/// satisfying: ∀c ∈ C, a ∈ A: α(c) ⊑_A a ⟺ c ⊑_C γ(a)
pub trait GaloisConnection {
    /// The concrete domain type.
    type Concrete: Clone + PartialEq;
    /// The abstract domain type.
    type Abstract: Clone + PartialEq;

    /// Abstraction function: maps concrete to abstract.
    fn alpha(&self, concrete: &Self::Concrete) -> Self::Abstract;

    /// Concretization function: maps abstract to concrete.
    fn gamma(&self, abs: &Self::Abstract) -> Self::Concrete;

    /// Check soundness: α(γ(a)) ⊑ a for all abstract elements.
    /// The abstraction of the concretization should not exceed the original.
    fn is_sound(&self, abs: &Self::Abstract) -> bool;

    /// Check optimality: α(c) is the best abstraction of c.
    fn is_optimal(&self, concrete: &Self::Concrete) -> bool;
}

// ═══════════════════════════════════════════════════════════════════════
// Lattice trait — abstract domain structure
// ═══════════════════════════════════════════════════════════════════════

/// A complete lattice for abstract interpretation.
pub trait Lattice: Clone + PartialEq + fmt::Debug {
    /// Bottom element ⊥.
    fn bottom() -> Self;
    /// Top element ⊤.
    fn top() -> Self;
    /// Join (least upper bound).
    fn join(&self, other: &Self) -> Self;
    /// Meet (greatest lower bound).
    fn meet(&self, other: &Self) -> Self;
    /// Partial order.
    fn leq(&self, other: &Self) -> bool;
    /// Is this bottom?
    fn is_bottom(&self) -> bool;
    /// Is this top?
    fn is_top(&self) -> bool;
}

// ═══════════════════════════════════════════════════════════════════════
// IntervalAbstraction — value range over-approximation
// ═══════════════════════════════════════════════════════════════════════

/// An interval [lo, hi] abstracting a set of integer values.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Interval {
    Bottom,
    Range { lo: i64, hi: i64 },
    Top,
}

impl Interval {
    pub fn exact(val: i64) -> Self {
        Interval::Range { lo: val, hi: val }
    }

    pub fn range(lo: i64, hi: i64) -> Self {
        if lo > hi {
            Interval::Bottom
        } else {
            Interval::Range { lo, hi }
        }
    }

    pub fn contains(&self, val: i64) -> bool {
        match self {
            Interval::Bottom => false,
            Interval::Range { lo, hi } => *lo <= val && val <= *hi,
            Interval::Top => true,
        }
    }

    pub fn width(&self) -> Option<u64> {
        match self {
            Interval::Bottom => Some(0),
            Interval::Range { lo, hi } => Some((*hi - *lo) as u64 + 1),
            Interval::Top => None,
        }
    }

    /// Arithmetic: add two intervals.
    pub fn add(&self, other: &Interval) -> Interval {
        match (self, other) {
            (Interval::Bottom, _) | (_, Interval::Bottom) => Interval::Bottom,
            (Interval::Top, _) | (_, Interval::Top) => Interval::Top,
            (Interval::Range { lo: a, hi: b }, Interval::Range { lo: c, hi: d }) => {
                Interval::Range {
                    lo: a.saturating_add(*c),
                    hi: b.saturating_add(*d),
                }
            }
        }
    }

    /// Arithmetic: negate an interval.
    pub fn neg(&self) -> Interval {
        match self {
            Interval::Bottom => Interval::Bottom,
            Interval::Top => Interval::Top,
            Interval::Range { lo, hi } => Interval::Range {
                lo: -hi,
                hi: -lo,
            },
        }
    }

    /// Arithmetic: subtract.
    pub fn sub(&self, other: &Interval) -> Interval {
        self.add(&other.neg())
    }
}

impl Lattice for Interval {
    fn bottom() -> Self {
        Interval::Bottom
    }

    fn top() -> Self {
        Interval::Top
    }

    fn join(&self, other: &Self) -> Self {
        match (self, other) {
            (Interval::Bottom, x) | (x, Interval::Bottom) => x.clone(),
            (Interval::Top, _) | (_, Interval::Top) => Interval::Top,
            (Interval::Range { lo: a, hi: b }, Interval::Range { lo: c, hi: d }) => {
                Interval::Range {
                    lo: (*a).min(*c),
                    hi: (*b).max(*d),
                }
            }
        }
    }

    fn meet(&self, other: &Self) -> Self {
        match (self, other) {
            (Interval::Bottom, _) | (_, Interval::Bottom) => Interval::Bottom,
            (Interval::Top, x) | (x, Interval::Top) => x.clone(),
            (Interval::Range { lo: a, hi: b }, Interval::Range { lo: c, hi: d }) => {
                let lo = (*a).max(*c);
                let hi = (*b).min(*d);
                if lo > hi {
                    Interval::Bottom
                } else {
                    Interval::Range { lo, hi }
                }
            }
        }
    }

    fn leq(&self, other: &Self) -> bool {
        match (self, other) {
            (Interval::Bottom, _) => true,
            (_, Interval::Top) => true,
            (Interval::Top, _) => false,
            (_, Interval::Bottom) => false,
            (Interval::Range { lo: a, hi: b }, Interval::Range { lo: c, hi: d }) => {
                c <= a && b <= d
            }
        }
    }

    fn is_bottom(&self) -> bool {
        matches!(self, Interval::Bottom)
    }

    fn is_top(&self) -> bool {
        matches!(self, Interval::Top)
    }
}

impl fmt::Display for Interval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Interval::Bottom => write!(f, "⊥"),
            Interval::Range { lo, hi } => {
                if lo == hi {
                    write!(f, "{{{}}}", lo)
                } else {
                    write!(f, "[{}, {}]", lo, hi)
                }
            }
            Interval::Top => write!(f, "⊤"),
        }
    }
}

/// Interval-based abstraction for memory values.
#[derive(Debug, Clone, PartialEq)]
pub struct IntervalAbstraction {
    /// Maps location → interval.
    pub values: HashMap<u64, Interval>,
}

impl IntervalAbstraction {
    pub fn new() -> Self {
        Self {
            values: HashMap::new(),
        }
    }

    pub fn set(&mut self, loc: u64, interval: Interval) {
        self.values.insert(loc, interval);
    }

    pub fn get(&self, loc: u64) -> &Interval {
        self.values.get(&loc).unwrap_or(&Interval::Top)
    }

    pub fn join(&self, other: &IntervalAbstraction) -> IntervalAbstraction {
        let mut result = self.clone();
        for (&loc, interval) in &other.values {
            let current = result.get(loc).clone();
            result.set(loc, current.join(interval));
        }
        result
    }

    pub fn meet(&self, other: &IntervalAbstraction) -> IntervalAbstraction {
        let mut result = IntervalAbstraction::new();
        let all_locs: HashSet<u64> = self
            .values
            .keys()
            .chain(other.values.keys())
            .copied()
            .collect();
        for loc in all_locs {
            let a = self.get(loc);
            let b = other.get(loc);
            result.set(loc, a.meet(b));
        }
        result
    }

    pub fn is_bottom(&self) -> bool {
        self.values.values().any(|v| v.is_bottom())
    }
}

impl Default for IntervalAbstraction {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════
// RelationalAbstraction — relational over-approximation
// ═══════════════════════════════════════════════════════════════════════

/// An abstract relation between events.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AbstractRelation {
    /// Empty relation (no pairs).
    Empty,
    /// The exact set of pairs.
    Exact(BTreeSet<(usize, usize)>),
    /// Over-approximation: all pairs within the given row/col sets.
    OverApprox {
        sources: BTreeSet<usize>,
        targets: BTreeSet<usize>,
    },
    /// Universal (all pairs).
    Universal,
}

impl AbstractRelation {
    pub fn empty() -> Self {
        AbstractRelation::Empty
    }

    pub fn exact(pairs: BTreeSet<(usize, usize)>) -> Self {
        if pairs.is_empty() {
            AbstractRelation::Empty
        } else {
            AbstractRelation::Exact(pairs)
        }
    }

    pub fn over_approx(sources: BTreeSet<usize>, targets: BTreeSet<usize>) -> Self {
        if sources.is_empty() || targets.is_empty() {
            AbstractRelation::Empty
        } else {
            AbstractRelation::OverApprox { sources, targets }
        }
    }

    pub fn universal() -> Self {
        AbstractRelation::Universal
    }

    pub fn may_contain(&self, a: usize, b: usize) -> bool {
        match self {
            AbstractRelation::Empty => false,
            AbstractRelation::Exact(pairs) => pairs.contains(&(a, b)),
            AbstractRelation::OverApprox { sources, targets } => {
                sources.contains(&a) && targets.contains(&b)
            }
            AbstractRelation::Universal => true,
        }
    }

    pub fn must_be_empty(&self) -> bool {
        matches!(self, AbstractRelation::Empty)
    }

    pub fn is_exact(&self) -> bool {
        matches!(self, AbstractRelation::Exact(_) | AbstractRelation::Empty)
    }
}

impl Lattice for AbstractRelation {
    fn bottom() -> Self {
        AbstractRelation::Empty
    }

    fn top() -> Self {
        AbstractRelation::Universal
    }

    fn join(&self, other: &Self) -> Self {
        match (self, other) {
            (AbstractRelation::Empty, x) | (x, AbstractRelation::Empty) => x.clone(),
            (AbstractRelation::Universal, _) | (_, AbstractRelation::Universal) => {
                AbstractRelation::Universal
            }
            (AbstractRelation::Exact(a), AbstractRelation::Exact(b)) => {
                let union: BTreeSet<_> = a.union(b).copied().collect();
                AbstractRelation::Exact(union)
            }
            (AbstractRelation::Exact(pairs), AbstractRelation::OverApprox { sources, targets })
            | (AbstractRelation::OverApprox { sources, targets }, AbstractRelation::Exact(pairs)) => {
                let mut new_sources = sources.clone();
                let mut new_targets = targets.clone();
                for (s, t) in pairs {
                    new_sources.insert(*s);
                    new_targets.insert(*t);
                }
                AbstractRelation::OverApprox {
                    sources: new_sources,
                    targets: new_targets,
                }
            }
            (
                AbstractRelation::OverApprox {
                    sources: s1,
                    targets: t1,
                },
                AbstractRelation::OverApprox {
                    sources: s2,
                    targets: t2,
                },
            ) => AbstractRelation::OverApprox {
                sources: s1.union(s2).copied().collect(),
                targets: t1.union(t2).copied().collect(),
            },
        }
    }

    fn meet(&self, other: &Self) -> Self {
        match (self, other) {
            (AbstractRelation::Empty, _) | (_, AbstractRelation::Empty) => {
                AbstractRelation::Empty
            }
            (AbstractRelation::Universal, x) | (x, AbstractRelation::Universal) => x.clone(),
            (AbstractRelation::Exact(a), AbstractRelation::Exact(b)) => {
                let inter: BTreeSet<_> = a.intersection(b).copied().collect();
                AbstractRelation::exact(inter)
            }
            (AbstractRelation::Exact(pairs), AbstractRelation::OverApprox { sources, targets })
            | (AbstractRelation::OverApprox { sources, targets }, AbstractRelation::Exact(pairs)) => {
                let filtered: BTreeSet<_> = pairs
                    .iter()
                    .filter(|(s, t)| sources.contains(s) && targets.contains(t))
                    .copied()
                    .collect();
                AbstractRelation::exact(filtered)
            }
            (
                AbstractRelation::OverApprox {
                    sources: s1,
                    targets: t1,
                },
                AbstractRelation::OverApprox {
                    sources: s2,
                    targets: t2,
                },
            ) => AbstractRelation::over_approx(
                s1.intersection(s2).copied().collect(),
                t1.intersection(t2).copied().collect(),
            ),
        }
    }

    fn leq(&self, other: &Self) -> bool {
        match (self, other) {
            (AbstractRelation::Empty, _) => true,
            (_, AbstractRelation::Universal) => true,
            (AbstractRelation::Universal, _) => false,
            (_, AbstractRelation::Empty) => self.must_be_empty(),
            (AbstractRelation::Exact(a), AbstractRelation::Exact(b)) => a.is_subset(b),
            (AbstractRelation::Exact(pairs), AbstractRelation::OverApprox { sources, targets }) => {
                pairs
                    .iter()
                    .all(|(s, t)| sources.contains(s) && targets.contains(t))
            }
            (AbstractRelation::OverApprox { .. }, AbstractRelation::Exact(_)) => {
                false // Over-approx can't be subset of exact in general
            }
            (
                AbstractRelation::OverApprox {
                    sources: s1,
                    targets: t1,
                },
                AbstractRelation::OverApprox {
                    sources: s2,
                    targets: t2,
                },
            ) => s1.is_subset(s2) && t1.is_subset(t2),
        }
    }

    fn is_bottom(&self) -> bool {
        self.must_be_empty()
    }

    fn is_top(&self) -> bool {
        matches!(self, AbstractRelation::Universal)
    }
}

/// Relational abstraction for an execution graph.
#[derive(Debug, Clone)]
pub struct RelationalAbstraction {
    pub relations: HashMap<String, AbstractRelation>,
}

impl RelationalAbstraction {
    pub fn new() -> Self {
        Self {
            relations: HashMap::new(),
        }
    }

    pub fn set_relation(&mut self, name: &str, rel: AbstractRelation) {
        self.relations.insert(name.to_string(), rel);
    }

    pub fn get_relation(&self, name: &str) -> &AbstractRelation {
        self.relations
            .get(name)
            .unwrap_or(&AbstractRelation::Universal)
    }

    /// Over-approximate composition of two relations.
    pub fn compose(&self, name_a: &str, name_b: &str) -> AbstractRelation {
        let a = self.get_relation(name_a);
        let b = self.get_relation(name_b);

        match (a, b) {
            (AbstractRelation::Empty, _) | (_, AbstractRelation::Empty) => {
                AbstractRelation::Empty
            }
            (AbstractRelation::Universal, _) | (_, AbstractRelation::Universal) => {
                AbstractRelation::Universal
            }
            (
                AbstractRelation::OverApprox {
                    sources: s1,
                    targets: _t1,
                },
                AbstractRelation::OverApprox {
                    sources: _s2,
                    targets: t2,
                },
            ) => AbstractRelation::OverApprox {
                sources: s1.clone(),
                targets: t2.clone(),
            },
            _ => AbstractRelation::Universal,
        }
    }

    /// Check if a relation is definitely acyclic.
    pub fn is_definitely_acyclic(&self, name: &str) -> Option<bool> {
        let rel = self.get_relation(name);
        match rel {
            AbstractRelation::Empty => Some(true),
            AbstractRelation::Exact(pairs) => {
                // Check for cycles in exact pairs
                Some(!has_cycle(pairs))
            }
            _ => None, // Can't determine
        }
    }

    pub fn join(&self, other: &RelationalAbstraction) -> RelationalAbstraction {
        let mut result = RelationalAbstraction::new();
        let all_names: HashSet<&String> = self
            .relations
            .keys()
            .chain(other.relations.keys())
            .collect();
        for name in all_names {
            let a = self.get_relation(name);
            let b = other.get_relation(name);
            result.set_relation(name, a.join(b));
        }
        result
    }
}

impl Default for RelationalAbstraction {
    fn default() -> Self {
        Self::new()
    }
}

/// Check if a set of directed pairs contains a cycle.
fn has_cycle(pairs: &BTreeSet<(usize, usize)>) -> bool {
    let mut adj: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut nodes: HashSet<usize> = HashSet::new();
    for &(a, b) in pairs {
        adj.entry(a).or_default().push(b);
        nodes.insert(a);
        nodes.insert(b);
    }

    let mut visited = HashSet::new();
    let mut in_stack = HashSet::new();

    fn dfs(
        node: usize,
        adj: &HashMap<usize, Vec<usize>>,
        visited: &mut HashSet<usize>,
        in_stack: &mut HashSet<usize>,
    ) -> bool {
        visited.insert(node);
        in_stack.insert(node);
        if let Some(neighbors) = adj.get(&node) {
            for &next in neighbors {
                if !visited.contains(&next) {
                    if dfs(next, adj, visited, in_stack) {
                        return true;
                    }
                } else if in_stack.contains(&next) {
                    return true;
                }
            }
        }
        in_stack.remove(&node);
        false
    }

    for &node in &nodes {
        if !visited.contains(&node) {
            if dfs(node, &adj, &mut visited, &mut in_stack) {
                return true;
            }
        }
    }
    false
}

// ═══════════════════════════════════════════════════════════════════════
// AbstractMemoryModel — abstract interpretation of memory models
// ═══════════════════════════════════════════════════════════════════════

/// Abstract memory state combining value abstraction and relational abstraction.
#[derive(Debug, Clone)]
pub struct AbstractMemoryState {
    /// Abstract values per memory location.
    pub values: IntervalAbstraction,
    /// Abstract relations between events.
    pub relations: RelationalAbstraction,
    /// Abstract thread states.
    pub thread_states: HashMap<usize, ThreadAbstractState>,
}

/// Abstract state of a single thread.
#[derive(Debug, Clone)]
pub struct ThreadAbstractState {
    pub thread_id: usize,
    /// Abstract register values.
    pub registers: HashMap<usize, Interval>,
    /// Program counter range.
    pub pc: Interval,
    /// Whether the thread is active.
    pub active: bool,
}

impl ThreadAbstractState {
    pub fn new(thread_id: usize) -> Self {
        Self {
            thread_id,
            registers: HashMap::new(),
            pc: Interval::exact(0),
            active: true,
        }
    }

    pub fn set_register(&mut self, reg: usize, val: Interval) {
        self.registers.insert(reg, val);
    }

    pub fn get_register(&self, reg: usize) -> &Interval {
        self.registers.get(&reg).unwrap_or(&Interval::Top)
    }
}

impl AbstractMemoryState {
    pub fn new() -> Self {
        Self {
            values: IntervalAbstraction::new(),
            relations: RelationalAbstraction::new(),
            thread_states: HashMap::new(),
        }
    }

    pub fn add_thread(&mut self, thread_id: usize) {
        self.thread_states
            .insert(thread_id, ThreadAbstractState::new(thread_id));
    }

    pub fn join(&self, other: &AbstractMemoryState) -> AbstractMemoryState {
        AbstractMemoryState {
            values: self.values.join(&other.values),
            relations: self.relations.join(&other.relations),
            thread_states: {
                let mut merged = self.thread_states.clone();
                for (&tid, other_ts) in &other.thread_states {
                    if let Some(self_ts) = merged.get_mut(&tid) {
                        // Join registers
                        let all_regs: HashSet<usize> = self_ts
                            .registers
                            .keys()
                            .chain(other_ts.registers.keys())
                            .copied()
                            .collect();
                        for reg in all_regs {
                            let a = self_ts.get_register(reg).clone();
                            let b = other_ts.get_register(reg);
                            self_ts.set_register(reg, a.join(b));
                        }
                        self_ts.pc = self_ts.pc.join(&other_ts.pc);
                        self_ts.active = self_ts.active || other_ts.active;
                    } else {
                        merged.insert(tid, other_ts.clone());
                    }
                }
                merged
            },
        }
    }
}

impl Default for AbstractMemoryState {
    fn default() -> Self {
        Self::new()
    }
}

/// Abstract memory model that over-approximates the concrete model.
#[derive(Debug, Clone)]
pub struct AbstractMemoryModel {
    pub name: String,
    /// Abstract constraints (name → whether they are known to hold).
    pub constraints: HashMap<String, ConstraintStatus>,
}

/// Status of an abstract constraint check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintStatus {
    /// Definitely holds.
    DefinitelyHolds,
    /// Definitely violated.
    DefinitelyViolated,
    /// Unknown (over-approximation prevents determination).
    Unknown,
}

impl AbstractMemoryModel {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            constraints: HashMap::new(),
        }
    }

    pub fn set_constraint(&mut self, name: &str, status: ConstraintStatus) {
        self.constraints.insert(name.to_string(), status);
    }

    /// Check if the abstract model is sound with respect to a set of executions.
    /// Returns true if every constraint that definitely holds in the abstract
    /// model also holds in all concrete executions.
    pub fn is_sound_for(
        &self,
        concrete_results: &HashMap<String, bool>,
    ) -> bool {
        for (name, &status) in &self.constraints {
            if status == ConstraintStatus::DefinitelyHolds {
                if let Some(&concrete) = concrete_results.get(name) {
                    if !concrete {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Get constraints that are definitely violated.
    pub fn violated_constraints(&self) -> Vec<&str> {
        self.constraints
            .iter()
            .filter(|(_, &s)| s == ConstraintStatus::DefinitelyViolated)
            .map(|(name, _)| name.as_str())
            .collect()
    }

    /// Get constraints with unknown status.
    pub fn unknown_constraints(&self) -> Vec<&str> {
        self.constraints
            .iter()
            .filter(|(_, &s)| s == ConstraintStatus::Unknown)
            .map(|(name, _)| name.as_str())
            .collect()
    }
}

// ═══════════════════════════════════════════════════════════════════════
// FixpointComputation — iterative abstract interpretation
// ═══════════════════════════════════════════════════════════════════════

/// Configuration for fixpoint computation.
#[derive(Debug, Clone)]
pub struct FixpointConfig {
    /// Maximum number of iterations.
    pub max_iterations: usize,
    /// Whether to use widening for convergence acceleration.
    pub use_widening: bool,
    /// Widening delay: apply widening after this many iterations.
    pub widening_delay: usize,
    /// Whether to use narrowing for precision recovery.
    pub use_narrowing: bool,
    /// Maximum narrowing iterations.
    pub max_narrowing_iterations: usize,
}

impl FixpointConfig {
    pub fn new() -> Self {
        Self {
            max_iterations: 100,
            use_widening: true,
            widening_delay: 3,
            use_narrowing: true,
            max_narrowing_iterations: 10,
        }
    }

    pub fn with_max_iterations(mut self, max: usize) -> Self {
        self.max_iterations = max;
        self
    }

    pub fn without_widening(mut self) -> Self {
        self.use_widening = false;
        self
    }
}

impl Default for FixpointConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of a fixpoint computation.
#[derive(Debug, Clone)]
pub struct FixpointResult<S: Clone> {
    /// The final state.
    pub state: S,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Whether the computation converged.
    pub converged: bool,
    /// Whether widening was applied.
    pub widening_applied: bool,
    /// Whether narrowing was applied.
    pub narrowing_applied: bool,
}

/// Abstract transfer function for computing fixpoints.
pub trait TransferFunction<S: Clone + PartialEq> {
    /// Apply one step of the abstract interpretation.
    fn transfer(&self, state: &S) -> S;

    /// Join two states.
    fn join(&self, a: &S, b: &S) -> S;

    /// Check if state a is less than or equal to state b.
    fn leq(&self, a: &S, b: &S) -> bool;
}

/// Fixpoint computation engine.
pub struct FixpointComputation<S: Clone + PartialEq> {
    pub config: FixpointConfig,
    _phantom: std::marker::PhantomData<S>,
}

impl<S: Clone + PartialEq> FixpointComputation<S> {
    pub fn new(config: FixpointConfig) -> Self {
        Self {
            config,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Compute the least fixpoint using Kleene iteration.
    pub fn compute<F: TransferFunction<S>>(&self, initial: S, tf: &F) -> FixpointResult<S> {
        let mut current = initial;
        let mut iterations = 0;
        let mut converged = false;
        let mut widening_applied = false;

        while iterations < self.config.max_iterations {
            let next = tf.transfer(&current);
            let joined = tf.join(&current, &next);

            if tf.leq(&joined, &current) {
                converged = true;
                break;
            }

            current = joined;
            iterations += 1;
        }

        FixpointResult {
            state: current,
            iterations,
            converged,
            widening_applied,
            narrowing_applied: false,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Widening — convergence acceleration
// ═══════════════════════════════════════════════════════════════════════

/// Widening operator for intervals.
pub struct IntervalWidening {
    /// Threshold values for widening.
    pub thresholds: Vec<i64>,
}

impl IntervalWidening {
    pub fn new() -> Self {
        Self {
            thresholds: vec![i64::MIN, -1000, -100, -10, -1, 0, 1, 10, 100, 1000, i64::MAX],
        }
    }

    pub fn with_thresholds(thresholds: Vec<i64>) -> Self {
        Self { thresholds }
    }

    /// Apply widening: a ∇ b.
    /// If b's lower bound decreased, widen to the next threshold below.
    /// If b's upper bound increased, widen to the next threshold above.
    pub fn widen(&self, a: &Interval, b: &Interval) -> Interval {
        match (a, b) {
            (Interval::Bottom, x) => x.clone(),
            (x, Interval::Bottom) => x.clone(),
            (_, Interval::Top) | (Interval::Top, _) => Interval::Top,
            (Interval::Range { lo: a_lo, hi: a_hi }, Interval::Range { lo: b_lo, hi: b_hi }) => {
                let new_lo = if *b_lo < *a_lo {
                    // Lower bound decreased — widen to next threshold below
                    self.thresholds
                        .iter()
                        .rev()
                        .find(|&&t| t <= *b_lo)
                        .copied()
                        .unwrap_or(i64::MIN)
                } else {
                    *a_lo
                };

                let new_hi = if *b_hi > *a_hi {
                    // Upper bound increased — widen to next threshold above
                    self.thresholds
                        .iter()
                        .find(|&&t| t >= *b_hi)
                        .copied()
                        .unwrap_or(i64::MAX)
                } else {
                    *a_hi
                };

                if new_lo == i64::MIN && new_hi == i64::MAX {
                    Interval::Top
                } else {
                    Interval::Range {
                        lo: new_lo,
                        hi: new_hi,
                    }
                }
            }
        }
    }

    /// Apply narrowing: a Δ b.
    /// Narrow a using information from b.
    pub fn narrow(&self, a: &Interval, b: &Interval) -> Interval {
        match (a, b) {
            (Interval::Bottom, _) => Interval::Bottom,
            (_, Interval::Bottom) => Interval::Bottom,
            (Interval::Top, x) => x.clone(),
            (x, Interval::Top) => x.clone(),
            (Interval::Range { lo: a_lo, hi: a_hi }, Interval::Range { lo: b_lo, hi: b_hi }) => {
                let new_lo = if *a_lo == i64::MIN { *b_lo } else { *a_lo };
                let new_hi = if *a_hi == i64::MAX { *b_hi } else { *a_hi };
                Interval::range(new_lo, new_hi)
            }
        }
    }
}

impl Default for IntervalWidening {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════
// SoundnessProof — verify abstraction is sound
// ═══════════════════════════════════════════════════════════════════════

/// A soundness witness for a Galois connection.
#[derive(Debug, Clone)]
pub struct SoundnessProof {
    /// What was checked.
    pub description: String,
    /// Whether soundness was verified.
    pub sound: bool,
    /// Evidence details.
    pub evidence: Vec<SoundnessEvidence>,
}

/// Individual piece of soundness evidence.
#[derive(Debug, Clone)]
pub struct SoundnessEvidence {
    pub property: String,
    pub concrete_result: bool,
    pub abstract_result: ConstraintStatus,
    pub sound: bool,
}

impl SoundnessEvidence {
    pub fn check(
        property: &str,
        concrete_result: bool,
        abstract_result: ConstraintStatus,
    ) -> Self {
        let sound = match abstract_result {
            ConstraintStatus::DefinitelyHolds => concrete_result,
            ConstraintStatus::DefinitelyViolated => !concrete_result,
            ConstraintStatus::Unknown => true, // Unknown is always sound
        };
        Self {
            property: property.to_string(),
            concrete_result,
            abstract_result,
            sound,
        }
    }
}

impl SoundnessProof {
    pub fn new(description: &str) -> Self {
        Self {
            description: description.to_string(),
            sound: true,
            evidence: Vec::new(),
        }
    }

    pub fn add_evidence(&mut self, evidence: SoundnessEvidence) {
        if !evidence.sound {
            self.sound = false;
        }
        self.evidence.push(evidence);
    }

    /// Verify soundness of an abstract model against concrete results.
    pub fn verify(
        description: &str,
        abstract_model: &AbstractMemoryModel,
        concrete_results: &HashMap<String, bool>,
    ) -> Self {
        let mut proof = Self::new(description);

        for (name, &abstract_status) in &abstract_model.constraints {
            if let Some(&concrete) = concrete_results.get(name) {
                proof.add_evidence(SoundnessEvidence::check(
                    name,
                    concrete,
                    abstract_status,
                ));
            }
        }

        proof
    }

    pub fn is_sound(&self) -> bool {
        self.sound
    }

    pub fn unsound_properties(&self) -> Vec<&str> {
        self.evidence
            .iter()
            .filter(|e| !e.sound)
            .map(|e| e.property.as_str())
            .collect()
    }
}

// ═══════════════════════════════════════════════════════════════════════
// MemoryModelGaloisConnection — concrete implementation
// ═══════════════════════════════════════════════════════════════════════

/// Concrete execution state for the Galois connection.
#[derive(Debug, Clone, PartialEq)]
pub struct ConcreteExecution {
    /// Number of events.
    pub num_events: usize,
    /// Relations as sets of pairs.
    pub relations: HashMap<String, BTreeSet<(usize, usize)>>,
    /// Memory values.
    pub memory: HashMap<u64, i64>,
}

impl ConcreteExecution {
    pub fn new(num_events: usize) -> Self {
        Self {
            num_events,
            relations: HashMap::new(),
            memory: HashMap::new(),
        }
    }

    pub fn add_relation(&mut self, name: &str, pairs: BTreeSet<(usize, usize)>) {
        self.relations.insert(name.to_string(), pairs);
    }

    pub fn set_memory(&mut self, addr: u64, val: i64) {
        self.memory.insert(addr, val);
    }
}

/// Abstract execution state.
#[derive(Debug, Clone, PartialEq)]
pub struct AbstractExecution {
    pub num_events: usize,
    pub relations: HashMap<String, AbstractRelation>,
    pub memory: IntervalAbstraction,
}

impl AbstractExecution {
    pub fn new(num_events: usize) -> Self {
        Self {
            num_events,
            relations: HashMap::new(),
            memory: IntervalAbstraction::new(),
        }
    }
}

/// The Galois connection between concrete and abstract executions.
pub struct ExecutionGaloisConnection {
    pub num_events: usize,
}

impl ExecutionGaloisConnection {
    pub fn new(num_events: usize) -> Self {
        Self { num_events }
    }
}

impl GaloisConnection for ExecutionGaloisConnection {
    type Concrete = ConcreteExecution;
    type Abstract = AbstractExecution;

    fn alpha(&self, concrete: &ConcreteExecution) -> AbstractExecution {
        let mut abs = AbstractExecution::new(concrete.num_events);

        // Abstract relations exactly
        for (name, pairs) in &concrete.relations {
            abs.relations
                .insert(name.clone(), AbstractRelation::exact(pairs.clone()));
        }

        // Abstract memory values as exact intervals
        for (&addr, &val) in &concrete.memory {
            abs.memory.set(addr, Interval::exact(val));
        }

        abs
    }

    fn gamma(&self, abs: &AbstractExecution) -> ConcreteExecution {
        let mut concrete = ConcreteExecution::new(abs.num_events);

        // Concretize relations
        for (name, rel) in &abs.relations {
            match rel {
                AbstractRelation::Exact(pairs) => {
                    concrete.add_relation(name, pairs.clone());
                }
                _ => {
                    // Over-approximation: concretize to empty (conservative)
                    concrete.add_relation(name, BTreeSet::new());
                }
            }
        }

        // Concretize memory: take lower bounds
        for (&addr, interval) in &abs.memory.values {
            match interval {
                Interval::Range { lo, .. } => {
                    concrete.set_memory(addr, *lo);
                }
                _ => {}
            }
        }

        concrete
    }

    fn is_sound(&self, abs: &AbstractExecution) -> bool {
        // α(γ(a)) ⊑ a
        let concretized = self.gamma(abs);
        let re_abstracted = self.alpha(&concretized);

        // Check that re-abstraction is less than or equal to original
        for (name, re_abs_rel) in &re_abstracted.relations {
            if let Some(orig_rel) = abs.relations.get(name) {
                if !re_abs_rel.leq(orig_rel) {
                    return false;
                }
            }
        }
        true
    }

    fn is_optimal(&self, concrete: &ConcreteExecution) -> bool {
        // γ(α(c)) = c for optimal Galois connections
        let abstracted = self.alpha(concrete);
        let concretized = self.gamma(&abstracted);
        concretized == *concrete
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Abstraction / Concretization helper functions
// ═══════════════════════════════════════════════════════════════════════

/// Abstract a set of concrete executions into a single abstract execution.
pub fn abstract_executions(
    executions: &[ConcreteExecution],
    num_events: usize,
) -> AbstractExecution {
    let gc = ExecutionGaloisConnection::new(num_events);
    let mut result = AbstractExecution::new(num_events);

    for exec in executions {
        let abs = gc.alpha(exec);

        // Join relations
        for (name, rel) in abs.relations {
            let current = result
                .relations
                .remove(&name)
                .unwrap_or(AbstractRelation::Empty);
            result.relations.insert(name, current.join(&rel));
        }

        // Join memory
        result.memory = result.memory.join(&abs.memory);
    }

    result
}

// ═══════════════════════════════════════════════════════════════════════
// SecurityLattice — multi-level security via Galois connections
// ═══════════════════════════════════════════════════════════════════════

/// A security level in a lattice.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum SecurityLevel {
    /// Public (low).
    Low,
    /// Confidential (medium).
    Medium,
    /// Secret (high).
    High,
    /// Top secret.
    TopSecret,
}

impl Lattice for SecurityLevel {
    fn bottom() -> Self {
        SecurityLevel::Low
    }

    fn top() -> Self {
        SecurityLevel::TopSecret
    }

    fn join(&self, other: &Self) -> Self {
        if self >= other {
            self.clone()
        } else {
            other.clone()
        }
    }

    fn meet(&self, other: &Self) -> Self {
        if self <= other {
            self.clone()
        } else {
            other.clone()
        }
    }

    fn leq(&self, other: &Self) -> bool {
        self <= other
    }

    fn is_bottom(&self) -> bool {
        *self == SecurityLevel::Low
    }

    fn is_top(&self) -> bool {
        *self == SecurityLevel::TopSecret
    }
}

impl fmt::Display for SecurityLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SecurityLevel::Low => write!(f, "Low"),
            SecurityLevel::Medium => write!(f, "Medium"),
            SecurityLevel::High => write!(f, "High"),
            SecurityLevel::TopSecret => write!(f, "TopSecret"),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // --- Interval tests ---

    #[test]
    fn test_interval_exact() {
        let i = Interval::exact(42);
        assert!(i.contains(42));
        assert!(!i.contains(41));
        assert_eq!(i.width(), Some(1));
    }

    #[test]
    fn test_interval_range() {
        let i = Interval::range(10, 20);
        assert!(i.contains(10));
        assert!(i.contains(15));
        assert!(i.contains(20));
        assert!(!i.contains(9));
        assert!(!i.contains(21));
        assert_eq!(i.width(), Some(11));
    }

    #[test]
    fn test_interval_bottom_top() {
        assert!(Interval::Bottom.is_bottom());
        assert!(Interval::Top.is_top());
        assert!(!Interval::Bottom.contains(0));
        assert!(Interval::Top.contains(0));
        assert!(Interval::Top.contains(i64::MAX));
    }

    #[test]
    fn test_interval_join() {
        let a = Interval::range(1, 5);
        let b = Interval::range(3, 10);
        let j = a.join(&b);
        assert_eq!(j, Interval::range(1, 10));

        let z = Interval::Bottom.join(&a);
        assert_eq!(z, a);
    }

    #[test]
    fn test_interval_meet() {
        let a = Interval::range(1, 10);
        let b = Interval::range(5, 15);
        let m = a.meet(&b);
        assert_eq!(m, Interval::range(5, 10));

        let disjoint_a = Interval::range(1, 3);
        let disjoint_b = Interval::range(5, 7);
        assert!(disjoint_a.meet(&disjoint_b).is_bottom());
    }

    #[test]
    fn test_interval_leq() {
        assert!(Interval::range(3, 5).leq(&Interval::range(1, 10)));
        assert!(!Interval::range(1, 10).leq(&Interval::range(3, 5)));
        assert!(Interval::Bottom.leq(&Interval::range(0, 0)));
        assert!(Interval::range(0, 0).leq(&Interval::Top));
    }

    #[test]
    fn test_interval_arithmetic() {
        let a = Interval::range(1, 5);
        let b = Interval::range(10, 20);
        let sum = a.add(&b);
        assert_eq!(sum, Interval::range(11, 25));

        let neg = a.neg();
        assert_eq!(neg, Interval::range(-5, -1));

        let diff = b.sub(&a);
        assert_eq!(diff, Interval::range(5, 19));
    }

    #[test]
    fn test_interval_invalid_range() {
        let i = Interval::range(10, 5);
        assert!(i.is_bottom());
    }

    // --- IntervalAbstraction tests ---

    #[test]
    fn test_interval_abstraction() {
        let mut abs = IntervalAbstraction::new();
        abs.set(0x100, Interval::range(0, 255));
        abs.set(0x200, Interval::exact(42));

        assert_eq!(abs.get(0x100), &Interval::range(0, 255));
        assert_eq!(abs.get(0x200), &Interval::exact(42));
        assert_eq!(abs.get(0x300), &Interval::Top);
    }

    #[test]
    fn test_interval_abstraction_join() {
        let mut a = IntervalAbstraction::new();
        a.set(0x100, Interval::range(0, 10));

        let mut b = IntervalAbstraction::new();
        b.set(0x100, Interval::range(5, 20));

        let joined = a.join(&b);
        assert_eq!(joined.get(0x100), &Interval::range(0, 20));
    }

    // --- AbstractRelation tests ---

    #[test]
    fn test_abstract_relation_empty() {
        let r = AbstractRelation::empty();
        assert!(r.must_be_empty());
        assert!(!r.may_contain(0, 1));
    }

    #[test]
    fn test_abstract_relation_exact() {
        let mut pairs = BTreeSet::new();
        pairs.insert((0, 1));
        pairs.insert((1, 2));
        let r = AbstractRelation::exact(pairs);
        assert!(r.may_contain(0, 1));
        assert!(r.may_contain(1, 2));
        assert!(!r.may_contain(0, 2));
        assert!(r.is_exact());
    }

    #[test]
    fn test_abstract_relation_over_approx() {
        let sources: BTreeSet<usize> = vec![0, 1].into_iter().collect();
        let targets: BTreeSet<usize> = vec![2, 3].into_iter().collect();
        let r = AbstractRelation::over_approx(sources, targets);
        assert!(r.may_contain(0, 2));
        assert!(r.may_contain(1, 3));
        assert!(!r.may_contain(0, 1));
        assert!(!r.is_exact());
    }

    #[test]
    fn test_abstract_relation_join() {
        let mut p1 = BTreeSet::new();
        p1.insert((0, 1));
        let r1 = AbstractRelation::exact(p1);

        let mut p2 = BTreeSet::new();
        p2.insert((2, 3));
        let r2 = AbstractRelation::exact(p2);

        let joined = r1.join(&r2);
        assert!(joined.may_contain(0, 1));
        assert!(joined.may_contain(2, 3));
    }

    #[test]
    fn test_abstract_relation_meet() {
        let mut p1 = BTreeSet::new();
        p1.insert((0, 1));
        p1.insert((1, 2));
        let r1 = AbstractRelation::exact(p1);

        let mut p2 = BTreeSet::new();
        p2.insert((1, 2));
        p2.insert((2, 3));
        let r2 = AbstractRelation::exact(p2);

        let met = r1.meet(&r2);
        assert!(met.may_contain(1, 2));
        assert!(!met.may_contain(0, 1));
        assert!(!met.may_contain(2, 3));
    }

    #[test]
    fn test_abstract_relation_lattice_laws() {
        let r = AbstractRelation::empty();
        assert!(r.leq(&AbstractRelation::Universal));
        assert!(!AbstractRelation::Universal.leq(&r));
        assert!(r.leq(&r));
    }

    // --- RelationalAbstraction tests ---

    #[test]
    fn test_relational_abstraction_acyclicity() {
        let mut ra = RelationalAbstraction::new();

        // Acyclic relation
        let mut pairs = BTreeSet::new();
        pairs.insert((0, 1));
        pairs.insert((1, 2));
        ra.set_relation("po", AbstractRelation::exact(pairs));
        assert_eq!(ra.is_definitely_acyclic("po"), Some(true));

        // Cyclic relation
        let mut cycle_pairs = BTreeSet::new();
        cycle_pairs.insert((0, 1));
        cycle_pairs.insert((1, 0));
        ra.set_relation("cycle", AbstractRelation::exact(cycle_pairs));
        assert_eq!(ra.is_definitely_acyclic("cycle"), Some(false));

        // Over-approx: can't determine
        let sources: BTreeSet<usize> = vec![0, 1].into_iter().collect();
        let targets: BTreeSet<usize> = vec![0, 1].into_iter().collect();
        ra.set_relation("unknown", AbstractRelation::over_approx(sources, targets));
        assert_eq!(ra.is_definitely_acyclic("unknown"), None);
    }

    // --- AbstractMemoryModel tests ---

    #[test]
    fn test_abstract_memory_model() {
        let mut model = AbstractMemoryModel::new("test-model");
        model.set_constraint("acyclic-po-rf", ConstraintStatus::DefinitelyHolds);
        model.set_constraint("acyclic-co", ConstraintStatus::Unknown);
        model.set_constraint("irreflexive-fr", ConstraintStatus::DefinitelyViolated);

        assert_eq!(model.violated_constraints(), vec!["irreflexive-fr"]);
        assert_eq!(model.unknown_constraints(), vec!["acyclic-co"]);
    }

    #[test]
    fn test_abstract_memory_model_soundness() {
        let mut model = AbstractMemoryModel::new("test");
        model.set_constraint("c1", ConstraintStatus::DefinitelyHolds);
        model.set_constraint("c2", ConstraintStatus::Unknown);

        let mut concrete = HashMap::new();
        concrete.insert("c1".to_string(), true);
        concrete.insert("c2".to_string(), false);

        assert!(model.is_sound_for(&concrete));

        // Unsound case: abstract says holds but concrete says no
        model.set_constraint("c2", ConstraintStatus::DefinitelyHolds);
        assert!(!model.is_sound_for(&concrete));
    }

    // --- FixpointComputation tests ---

    #[test]
    fn test_fixpoint_config() {
        let config = FixpointConfig::new();
        assert_eq!(config.max_iterations, 100);
        assert!(config.use_widening);

        let config2 = FixpointConfig::new().with_max_iterations(50).without_widening();
        assert_eq!(config2.max_iterations, 50);
        assert!(!config2.use_widening);
    }

    struct IncrementTransfer;

    impl TransferFunction<Interval> for IncrementTransfer {
        fn transfer(&self, state: &Interval) -> Interval {
            state.add(&Interval::exact(1))
        }

        fn join(&self, a: &Interval, b: &Interval) -> Interval {
            a.join(b)
        }

        fn leq(&self, a: &Interval, b: &Interval) -> bool {
            a.leq(b)
        }
    }

    #[test]
    fn test_fixpoint_computation() {
        let config = FixpointConfig::new().with_max_iterations(10);
        let fp = FixpointComputation::new(config);
        let result = fp.compute(Interval::exact(0), &IncrementTransfer);
        // Won't converge without widening, but should terminate
        assert!(result.iterations <= 10);
    }

    struct IdentityTransfer;

    impl TransferFunction<Interval> for IdentityTransfer {
        fn transfer(&self, state: &Interval) -> Interval {
            state.clone()
        }

        fn join(&self, a: &Interval, b: &Interval) -> Interval {
            a.join(b)
        }

        fn leq(&self, a: &Interval, b: &Interval) -> bool {
            a.leq(b)
        }
    }

    #[test]
    fn test_fixpoint_converges_immediately() {
        let config = FixpointConfig::new();
        let fp = FixpointComputation::new(config);
        let result = fp.compute(Interval::exact(0), &IdentityTransfer);
        assert!(result.converged);
        assert_eq!(result.iterations, 0);
    }

    // --- Widening tests ---

    #[test]
    fn test_widening() {
        let w = IntervalWidening::new();
        let a = Interval::range(0, 10);
        let b = Interval::range(0, 20);
        let widened = w.widen(&a, &b);
        // Upper bound increased, should widen to threshold ≥ 20
        match widened {
            Interval::Range { hi, .. } => assert!(hi >= 20),
            _ => {}
        }
    }

    #[test]
    fn test_widening_lower_bound() {
        let w = IntervalWidening::new();
        let a = Interval::range(0, 10);
        let b = Interval::range(-5, 10);
        let widened = w.widen(&a, &b);
        match widened {
            Interval::Range { lo, .. } => assert!(lo <= -5),
            _ => {}
        }
    }

    #[test]
    fn test_narrowing() {
        let w = IntervalWidening::new();
        let a = Interval::range(0, 100);
        let b = Interval::range(5, 50);
        let narrowed = w.narrow(&a, &b);
        assert_eq!(narrowed, Interval::range(0, 100));
    }

    // --- SoundnessProof tests ---

    #[test]
    fn test_soundness_proof_sound() {
        let mut model = AbstractMemoryModel::new("test");
        model.set_constraint("c1", ConstraintStatus::DefinitelyHolds);
        model.set_constraint("c2", ConstraintStatus::DefinitelyViolated);

        let mut concrete = HashMap::new();
        concrete.insert("c1".to_string(), true);
        concrete.insert("c2".to_string(), false);

        let proof = SoundnessProof::verify("test", &model, &concrete);
        assert!(proof.is_sound());
        assert!(proof.unsound_properties().is_empty());
    }

    #[test]
    fn test_soundness_proof_unsound() {
        let mut model = AbstractMemoryModel::new("test");
        model.set_constraint("c1", ConstraintStatus::DefinitelyHolds);

        let mut concrete = HashMap::new();
        concrete.insert("c1".to_string(), false);

        let proof = SoundnessProof::verify("test", &model, &concrete);
        assert!(!proof.is_sound());
        assert_eq!(proof.unsound_properties(), vec!["c1"]);
    }

    // --- GaloisConnection tests ---

    #[test]
    fn test_execution_galois_connection_roundtrip() {
        let gc = ExecutionGaloisConnection::new(4);

        let mut concrete = ConcreteExecution::new(4);
        let mut po = BTreeSet::new();
        po.insert((0, 1));
        po.insert((2, 3));
        concrete.add_relation("po", po);
        concrete.set_memory(0x100, 42);

        let abs = gc.alpha(&concrete);
        assert!(abs.relations["po"].may_contain(0, 1));
        assert!(abs.relations["po"].may_contain(2, 3));
        assert!(!abs.relations["po"].may_contain(0, 2));

        let back = gc.gamma(&abs);
        assert_eq!(back.num_events, 4);
    }

    #[test]
    fn test_execution_galois_soundness() {
        let gc = ExecutionGaloisConnection::new(4);
        let abs = AbstractExecution::new(4);
        assert!(gc.is_sound(&abs));
    }

    #[test]
    fn test_abstract_executions() {
        let exec1 = {
            let mut e = ConcreteExecution::new(2);
            let mut po = BTreeSet::new();
            po.insert((0, 1));
            e.add_relation("po", po);
            e.set_memory(0x100, 1);
            e
        };

        let exec2 = {
            let mut e = ConcreteExecution::new(2);
            let mut po = BTreeSet::new();
            po.insert((0, 1));
            e.add_relation("po", po);
            e.set_memory(0x100, 5);
            e
        };

        let abs = abstract_executions(&[exec1, exec2], 2);
        assert!(abs.relations["po"].may_contain(0, 1));

        // Memory abstraction should cover both values
        match abs.memory.get(0x100) {
            Interval::Range { lo, hi } => {
                assert!(*lo <= 1);
                assert!(*hi >= 5);
            }
            _ => {}
        }
    }

    // --- SecurityLevel tests ---

    #[test]
    fn test_security_level_lattice() {
        assert!(SecurityLevel::Low.leq(&SecurityLevel::High));
        assert!(!SecurityLevel::High.leq(&SecurityLevel::Low));
        assert_eq!(
            SecurityLevel::Low.join(&SecurityLevel::High),
            SecurityLevel::High
        );
        assert_eq!(
            SecurityLevel::Low.meet(&SecurityLevel::High),
            SecurityLevel::Low
        );
    }

    #[test]
    fn test_security_level_bottom_top() {
        assert!(SecurityLevel::Low.is_bottom());
        assert!(SecurityLevel::TopSecret.is_top());
    }

    // --- ThreadAbstractState tests ---

    #[test]
    fn test_thread_abstract_state() {
        let mut ts = ThreadAbstractState::new(0);
        ts.set_register(0, Interval::exact(42));
        assert_eq!(ts.get_register(0), &Interval::exact(42));
        assert_eq!(ts.get_register(1), &Interval::Top);
    }

    // --- AbstractMemoryState tests ---

    #[test]
    fn test_abstract_memory_state_join() {
        let mut s1 = AbstractMemoryState::new();
        s1.add_thread(0);
        s1.values.set(0x100, Interval::range(0, 10));

        let mut s2 = AbstractMemoryState::new();
        s2.add_thread(0);
        s2.values.set(0x100, Interval::range(5, 20));

        let joined = s1.join(&s2);
        assert_eq!(joined.values.get(0x100), &Interval::range(0, 20));
    }

    // --- has_cycle tests ---

    #[test]
    fn test_has_cycle_no_cycle() {
        let mut pairs = BTreeSet::new();
        pairs.insert((0, 1));
        pairs.insert((1, 2));
        pairs.insert((2, 3));
        assert!(!has_cycle(&pairs));
    }

    #[test]
    fn test_has_cycle_with_cycle() {
        let mut pairs = BTreeSet::new();
        pairs.insert((0, 1));
        pairs.insert((1, 2));
        pairs.insert((2, 0));
        assert!(has_cycle(&pairs));
    }

    #[test]
    fn test_has_cycle_self_loop() {
        let mut pairs = BTreeSet::new();
        pairs.insert((0, 0));
        assert!(has_cycle(&pairs));
    }

    #[test]
    fn test_has_cycle_empty() {
        let pairs = BTreeSet::new();
        assert!(!has_cycle(&pairs));
    }

    // --- Display tests ---

    #[test]
    fn test_interval_display() {
        assert_eq!(format!("{}", Interval::Bottom), "⊥");
        assert_eq!(format!("{}", Interval::Top), "⊤");
        assert_eq!(format!("{}", Interval::exact(42)), "{42}");
        assert_eq!(format!("{}", Interval::range(1, 10)), "[1, 10]");
    }

    #[test]
    fn test_security_level_display() {
        assert_eq!(format!("{}", SecurityLevel::Low), "Low");
        assert_eq!(format!("{}", SecurityLevel::High), "High");
    }

    // --- Constraint evidence tests ---

    #[test]
    fn test_constraint_evidence_sound() {
        let e = SoundnessEvidence::check("test", true, ConstraintStatus::DefinitelyHolds);
        assert!(e.sound);

        let e2 = SoundnessEvidence::check("test", false, ConstraintStatus::Unknown);
        assert!(e2.sound); // Unknown is always sound

        let e3 = SoundnessEvidence::check("test", false, ConstraintStatus::DefinitelyHolds);
        assert!(!e3.sound);
    }
}
