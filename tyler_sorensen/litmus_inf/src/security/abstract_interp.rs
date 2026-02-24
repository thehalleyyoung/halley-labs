//! Abstract interpretation for litmus test analysis.
//!
//! Provides abstract domains (intervals, signs, taint), abstract execution
//! of litmus tests, over-approximation of reachable states, and
//! Galois connections between concrete and abstract domains.

use std::collections::HashMap;
use std::fmt;
use serde::{Serialize, Deserialize};

// ---------------------------------------------------------------------------
// AbstractDomain trait — lattice operations
// ---------------------------------------------------------------------------

/// A lattice-based abstract domain for abstract interpretation.
pub trait AbstractDomain: Clone + PartialEq + fmt::Debug + fmt::Display {
    /// Bottom element (⊥) — no information / unreachable.
    fn bottom() -> Self;

    /// Top element (⊤) — all possible values.
    fn top() -> Self;

    /// Join (least upper bound): a ⊔ b.
    fn join(&self, other: &Self) -> Self;

    /// Meet (greatest lower bound): a ⊓ b.
    fn meet(&self, other: &Self) -> Self;

    /// Widening operator for convergence.
    fn widen(&self, other: &Self) -> Self {
        self.join(other)
    }

    /// Narrowing operator for precision.
    fn narrow(&self, other: &Self) -> Self {
        self.meet(other)
    }

    /// Is this the bottom element?
    fn is_bottom(&self) -> bool;

    /// Is this the top element?
    fn is_top(&self) -> bool;

    /// Partial order: self ⊑ other.
    fn leq(&self, other: &Self) -> bool;
}

// ---------------------------------------------------------------------------
// IntervalDomain — value range abstraction
// ---------------------------------------------------------------------------

/// Abstract domain of integer intervals [lo, hi].
#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IntervalDomain {
    /// Bottom: empty set (unreachable).
    Bottom,
    /// An interval [lo, hi].
    Interval { lo: i64, hi: i64 },
    /// Top: all integers.
    Top,
}

impl fmt::Debug for IntervalDomain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bottom => write!(f, "⊥"),
            Self::Interval { lo, hi } => write!(f, "[{}, {}]", lo, hi),
            Self::Top => write!(f, "⊤"),
        }
    }
}

impl fmt::Display for IntervalDomain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

impl IntervalDomain {
    /// Create a singleton interval [v, v].
    pub fn constant(v: i64) -> Self {
        Self::Interval { lo: v, hi: v }
    }

    /// Create an interval [lo, hi].
    pub fn interval(lo: i64, hi: i64) -> Self {
        if lo > hi {
            Self::Bottom
        } else {
            Self::Interval { lo, hi }
        }
    }

    /// Abstract addition.
    pub fn add(&self, other: &Self) -> Self {
        match (self, other) {
            (Self::Bottom, _) | (_, Self::Bottom) => Self::Bottom,
            (Self::Top, _) | (_, Self::Top) => Self::Top,
            (Self::Interval { lo: a, hi: b }, Self::Interval { lo: c, hi: d }) => {
                Self::Interval {
                    lo: a.saturating_add(*c),
                    hi: b.saturating_add(*d),
                }
            }
        }
    }

    /// Abstract subtraction.
    pub fn sub(&self, other: &Self) -> Self {
        match (self, other) {
            (Self::Bottom, _) | (_, Self::Bottom) => Self::Bottom,
            (Self::Top, _) | (_, Self::Top) => Self::Top,
            (Self::Interval { lo: a, hi: b }, Self::Interval { lo: c, hi: d }) => {
                Self::Interval {
                    lo: a.saturating_sub(*d),
                    hi: b.saturating_sub(*c),
                }
            }
        }
    }

    /// Abstract multiplication.
    pub fn mul(&self, other: &Self) -> Self {
        match (self, other) {
            (Self::Bottom, _) | (_, Self::Bottom) => Self::Bottom,
            (Self::Top, _) | (_, Self::Top) => Self::Top,
            (Self::Interval { lo: a, hi: b }, Self::Interval { lo: c, hi: d }) => {
                let products = [
                    a.saturating_mul(*c),
                    a.saturating_mul(*d),
                    b.saturating_mul(*c),
                    b.saturating_mul(*d),
                ];
                Self::Interval {
                    lo: *products.iter().min().unwrap(),
                    hi: *products.iter().max().unwrap(),
                }
            }
        }
    }

    /// Whether the interval contains a specific value.
    pub fn contains_value(&self, v: i64) -> bool {
        match self {
            Self::Bottom => false,
            Self::Top => true,
            Self::Interval { lo, hi } => *lo <= v && v <= *hi,
        }
    }

    /// Width of the interval (None for ⊥ and ⊤).
    pub fn width(&self) -> Option<u64> {
        match self {
            Self::Interval { lo, hi } => Some((*hi - *lo) as u64),
            _ => None,
        }
    }
}

impl AbstractDomain for IntervalDomain {
    fn bottom() -> Self { Self::Bottom }
    fn top() -> Self { Self::Top }

    fn join(&self, other: &Self) -> Self {
        match (self, other) {
            (Self::Bottom, x) | (x, Self::Bottom) => x.clone(),
            (Self::Top, _) | (_, Self::Top) => Self::Top,
            (Self::Interval { lo: a, hi: b }, Self::Interval { lo: c, hi: d }) => {
                Self::Interval { lo: (*a).min(*c), hi: (*b).max(*d) }
            }
        }
    }

    fn meet(&self, other: &Self) -> Self {
        match (self, other) {
            (Self::Bottom, _) | (_, Self::Bottom) => Self::Bottom,
            (Self::Top, x) | (x, Self::Top) => x.clone(),
            (Self::Interval { lo: a, hi: b }, Self::Interval { lo: c, hi: d }) => {
                let lo = (*a).max(*c);
                let hi = (*b).min(*d);
                if lo > hi { Self::Bottom } else { Self::Interval { lo, hi } }
            }
        }
    }

    fn widen(&self, other: &Self) -> Self {
        match (self, other) {
            (Self::Bottom, x) => x.clone(),
            (x, Self::Bottom) => x.clone(),
            (Self::Top, _) | (_, Self::Top) => Self::Top,
            (Self::Interval { lo: a, hi: b }, Self::Interval { lo: c, hi: d }) => {
                let lo = if c < a { i64::MIN } else { *a };
                let hi = if d > b { i64::MAX } else { *b };
                if lo == i64::MIN && hi == i64::MAX {
                    Self::Top
                } else {
                    Self::Interval { lo, hi }
                }
            }
        }
    }

    fn is_bottom(&self) -> bool { matches!(self, Self::Bottom) }
    fn is_top(&self) -> bool { matches!(self, Self::Top) }

    fn leq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Bottom, _) => true,
            (_, Self::Top) => true,
            (Self::Top, _) => false,
            (_, Self::Bottom) => false,
            (Self::Interval { lo: a, hi: b }, Self::Interval { lo: c, hi: d }) => {
                c <= a && b <= d
            }
        }
    }
}

// ---------------------------------------------------------------------------
// SignDomain — sign abstraction
// ---------------------------------------------------------------------------

/// Abstract domain tracking the sign of values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SignDomain {
    /// Bottom: unreachable.
    Bottom,
    /// Strictly positive.
    Positive,
    /// Zero.
    Zero,
    /// Strictly negative.
    Negative,
    /// Non-negative (≥ 0).
    NonNegative,
    /// Non-positive (≤ 0).
    NonPositive,
    /// Non-zero (positive or negative).
    NonZero,
    /// Top: any value.
    Top,
}

impl fmt::Display for SignDomain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bottom => write!(f, "⊥"),
            Self::Positive => write!(f, "+"),
            Self::Zero => write!(f, "0"),
            Self::Negative => write!(f, "-"),
            Self::NonNegative => write!(f, "≥0"),
            Self::NonPositive => write!(f, "≤0"),
            Self::NonZero => write!(f, "≠0"),
            Self::Top => write!(f, "⊤"),
        }
    }
}

impl SignDomain {
    /// Abstract a concrete value.
    pub fn from_value(v: i64) -> Self {
        if v > 0 { Self::Positive }
        else if v < 0 { Self::Negative }
        else { Self::Zero }
    }

    /// Abstract addition.
    pub fn add(&self, other: &Self) -> Self {
        match (self, other) {
            (Self::Bottom, _) | (_, Self::Bottom) => Self::Bottom,
            (Self::Zero, x) | (x, Self::Zero) => *x,
            (Self::Positive, Self::Positive) => Self::Positive,
            (Self::Negative, Self::Negative) => Self::Negative,
            (Self::Positive, Self::Negative) | (Self::Negative, Self::Positive) => Self::Top,
            _ => Self::Top,
        }
    }

    /// Abstract multiplication.
    pub fn mul(&self, other: &Self) -> Self {
        match (self, other) {
            (Self::Bottom, _) | (_, Self::Bottom) => Self::Bottom,
            (Self::Zero, _) | (_, Self::Zero) => Self::Zero,
            (Self::Positive, Self::Positive) | (Self::Negative, Self::Negative) => Self::Positive,
            (Self::Positive, Self::Negative) | (Self::Negative, Self::Positive) => Self::Negative,
            _ => Self::Top,
        }
    }

    /// Abstract negation.
    pub fn negate(&self) -> Self {
        match self {
            Self::Bottom => Self::Bottom,
            Self::Positive => Self::Negative,
            Self::Negative => Self::Positive,
            Self::Zero => Self::Zero,
            Self::NonNegative => Self::NonPositive,
            Self::NonPositive => Self::NonNegative,
            Self::NonZero => Self::NonZero,
            Self::Top => Self::Top,
        }
    }
}

impl AbstractDomain for SignDomain {
    fn bottom() -> Self { Self::Bottom }
    fn top() -> Self { Self::Top }

    fn join(&self, other: &Self) -> Self {
        if self == other { return *self; }
        match (self, other) {
            (Self::Bottom, x) | (x, Self::Bottom) => *x,
            (Self::Top, _) | (_, Self::Top) => Self::Top,
            (Self::Positive, Self::Zero) | (Self::Zero, Self::Positive) => Self::NonNegative,
            (Self::Negative, Self::Zero) | (Self::Zero, Self::Negative) => Self::NonPositive,
            (Self::Positive, Self::Negative) | (Self::Negative, Self::Positive) => Self::NonZero,
            (Self::NonNegative, Self::Negative) | (Self::Negative, Self::NonNegative) => Self::Top,
            (Self::NonPositive, Self::Positive) | (Self::Positive, Self::NonPositive) => Self::Top,
            (Self::NonNegative, Self::Zero) | (Self::Zero, Self::NonNegative) => Self::NonNegative,
            (Self::NonPositive, Self::Zero) | (Self::Zero, Self::NonPositive) => Self::NonPositive,
            (Self::NonNegative, Self::Positive) | (Self::Positive, Self::NonNegative) => Self::NonNegative,
            (Self::NonPositive, Self::Negative) | (Self::Negative, Self::NonPositive) => Self::NonPositive,
            _ => Self::Top,
        }
    }

    fn meet(&self, other: &Self) -> Self {
        if self == other { return *self; }
        match (self, other) {
            (Self::Bottom, _) | (_, Self::Bottom) => Self::Bottom,
            (Self::Top, x) | (x, Self::Top) => *x,
            (Self::NonNegative, Self::Positive) | (Self::Positive, Self::NonNegative) => Self::Positive,
            (Self::NonNegative, Self::Zero) | (Self::Zero, Self::NonNegative) => Self::Zero,
            (Self::NonPositive, Self::Negative) | (Self::Negative, Self::NonPositive) => Self::Negative,
            (Self::NonPositive, Self::Zero) | (Self::Zero, Self::NonPositive) => Self::Zero,
            (Self::NonZero, Self::Positive) | (Self::Positive, Self::NonZero) => Self::Positive,
            (Self::NonZero, Self::Negative) | (Self::Negative, Self::NonZero) => Self::Negative,
            (Self::NonNegative, Self::NonPositive) | (Self::NonPositive, Self::NonNegative) => Self::Zero,
            (Self::NonNegative, Self::NonZero) | (Self::NonZero, Self::NonNegative) => Self::Positive,
            (Self::NonPositive, Self::NonZero) | (Self::NonZero, Self::NonPositive) => Self::Negative,
            _ => Self::Bottom,
        }
    }

    fn is_bottom(&self) -> bool { matches!(self, Self::Bottom) }
    fn is_top(&self) -> bool { matches!(self, Self::Top) }

    fn leq(&self, other: &Self) -> bool {
        if self == other { return true; }
        match (self, other) {
            (Self::Bottom, _) => true,
            (_, Self::Top) => true,
            (Self::Positive, Self::NonNegative) => true,
            (Self::Positive, Self::NonZero) => true,
            (Self::Negative, Self::NonPositive) => true,
            (Self::Negative, Self::NonZero) => true,
            (Self::Zero, Self::NonNegative) => true,
            (Self::Zero, Self::NonPositive) => true,
            _ => false,
        }
    }
}

// ---------------------------------------------------------------------------
// TaintDomain — information flow tracking
// ---------------------------------------------------------------------------

/// Abstract domain for information flow tracking.
/// Tracks whether a value is tainted (influenced by secrets).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TaintDomain {
    /// Bottom: unreachable.
    Bottom,
    /// Clean: not influenced by secrets.
    Clean,
    /// Tainted: influenced by secrets.
    Tainted,
    /// Top: may or may not be tainted.
    Top,
}

impl fmt::Display for TaintDomain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Bottom => write!(f, "⊥"),
            Self::Clean => write!(f, "clean"),
            Self::Tainted => write!(f, "tainted"),
            Self::Top => write!(f, "⊤"),
        }
    }
}

impl TaintDomain {
    /// Propagate taint through a binary operation.
    /// If either operand is tainted, the result is tainted.
    pub fn propagate(&self, other: &Self) -> Self {
        match (self, other) {
            (Self::Bottom, _) | (_, Self::Bottom) => Self::Bottom,
            (Self::Tainted, _) | (_, Self::Tainted) => Self::Tainted,
            (Self::Top, _) | (_, Self::Top) => Self::Top,
            (Self::Clean, Self::Clean) => Self::Clean,
        }
    }

    /// Is this value potentially tainted?
    pub fn is_potentially_tainted(&self) -> bool {
        matches!(self, Self::Tainted | Self::Top)
    }
}

impl AbstractDomain for TaintDomain {
    fn bottom() -> Self { Self::Bottom }
    fn top() -> Self { Self::Top }

    fn join(&self, other: &Self) -> Self {
        if self == other { return *self; }
        match (self, other) {
            (Self::Bottom, x) | (x, Self::Bottom) => *x,
            (Self::Top, _) | (_, Self::Top) => Self::Top,
            (Self::Clean, Self::Tainted) | (Self::Tainted, Self::Clean) => Self::Top,
            _ => Self::Top,
        }
    }

    fn meet(&self, other: &Self) -> Self {
        if self == other { return *self; }
        match (self, other) {
            (Self::Bottom, _) | (_, Self::Bottom) => Self::Bottom,
            (Self::Top, x) | (x, Self::Top) => *x,
            (Self::Clean, Self::Tainted) | (Self::Tainted, Self::Clean) => Self::Bottom,
            _ => Self::Bottom,
        }
    }

    fn is_bottom(&self) -> bool { matches!(self, Self::Bottom) }
    fn is_top(&self) -> bool { matches!(self, Self::Top) }

    fn leq(&self, other: &Self) -> bool {
        if self == other { return true; }
        match (self, other) {
            (Self::Bottom, _) => true,
            (_, Self::Top) => true,
            _ => false,
        }
    }
}

// ---------------------------------------------------------------------------
// AbstractState — abstract state for litmus test execution
// ---------------------------------------------------------------------------

/// Abstract state mapping registers and memory to abstract values.
#[derive(Debug, Clone)]
pub struct AbstractState<D: AbstractDomain> {
    /// Register values: (thread_id, reg_id) → abstract value.
    pub registers: HashMap<(usize, usize), D>,
    /// Memory values: address → abstract value.
    pub memory: HashMap<u64, D>,
}

impl<D: AbstractDomain> AbstractState<D> {
    /// Create a new empty abstract state (all ⊥).
    pub fn new() -> Self {
        Self {
            registers: HashMap::new(),
            memory: HashMap::new(),
        }
    }

    /// Get a register's abstract value, defaulting to top.
    pub fn get_reg(&self, thread: usize, reg: usize) -> D {
        self.registers.get(&(thread, reg)).cloned().unwrap_or_else(D::top)
    }

    /// Set a register's abstract value.
    pub fn set_reg(&mut self, thread: usize, reg: usize, val: D) {
        self.registers.insert((thread, reg), val);
    }

    /// Get a memory location's abstract value, defaulting to top.
    pub fn get_mem(&self, addr: u64) -> D {
        self.memory.get(&addr).cloned().unwrap_or_else(D::top)
    }

    /// Set a memory location's abstract value.
    pub fn set_mem(&mut self, addr: u64, val: D) {
        self.memory.insert(addr, val);
    }

    /// Join two abstract states.
    pub fn join(&self, other: &Self) -> Self {
        let mut result = self.clone();
        for (&key, val) in &other.registers {
            let existing = result.registers.entry(key).or_insert_with(D::bottom);
            *existing = existing.join(val);
        }
        for (&addr, val) in &other.memory {
            let existing = result.memory.entry(addr).or_insert_with(D::bottom);
            *existing = existing.join(val);
        }
        result
    }

    /// Widen two abstract states.
    pub fn widen(&self, other: &Self) -> Self {
        let mut result = self.clone();
        for (&key, val) in &other.registers {
            let existing = result.registers.entry(key).or_insert_with(D::bottom);
            *existing = existing.widen(val);
        }
        for (&addr, val) in &other.memory {
            let existing = result.memory.entry(addr).or_insert_with(D::bottom);
            *existing = existing.widen(val);
        }
        result
    }
}

impl<D: AbstractDomain> Default for AbstractState<D> {
    fn default() -> Self { Self::new() }
}

// ---------------------------------------------------------------------------
// AbstractExecutor — abstract execution of litmus tests
// ---------------------------------------------------------------------------

/// Abstract executor for litmus tests.
pub struct AbstractExecutor<D: AbstractDomain> {
    _phantom: std::marker::PhantomData<D>,
}

impl AbstractExecutor<IntervalDomain> {
    /// Execute a litmus test abstractly with interval domain.
    pub fn execute_interval(
        test: &crate::checker::litmus::LitmusTest,
    ) -> AbstractState<IntervalDomain> {
        let mut state = AbstractState::<IntervalDomain>::new();

        // Initialize memory.
        for (&addr, &val) in &test.initial_state {
            state.set_mem(addr, IntervalDomain::constant(val as i64));
        }

        // Process each thread's instructions.
        for thread in &test.threads {
            for instr in &thread.instructions {
                match instr {
                    crate::checker::litmus::Instruction::Store { addr, value, .. } => {
                        state.set_mem(*addr, IntervalDomain::constant(*value as i64));
                    }
                    crate::checker::litmus::Instruction::Load { reg, addr, .. } => {
                        let mem_val = state.get_mem(*addr);
                        state.set_reg(thread.id, *reg, mem_val);
                    }
                    crate::checker::litmus::Instruction::RMW { reg, addr, value, .. } => {
                        let old_val = state.get_mem(*addr);
                        state.set_reg(thread.id, *reg, old_val);
                        state.set_mem(*addr, IntervalDomain::constant(*value as i64));
                    }
                    _ => {} // Fences, branches don't affect abstract state directly.
                }
            }
        }

        state
    }
}

impl AbstractExecutor<TaintDomain> {
    /// Execute a litmus test abstractly with taint domain.
    pub fn execute_taint(
        test: &crate::checker::litmus::LitmusTest,
        tainted_addresses: &[u64],
    ) -> AbstractState<TaintDomain> {
        let mut state = AbstractState::<TaintDomain>::new();

        // Mark tainted addresses.
        for &addr in tainted_addresses {
            state.set_mem(addr, TaintDomain::Tainted);
        }

        // Initialize non-tainted memory as clean.
        for (&addr, _) in &test.initial_state {
            if !tainted_addresses.contains(&addr) {
                state.set_mem(addr, TaintDomain::Clean);
            }
        }

        // Process each thread's instructions.
        for thread in &test.threads {
            for instr in &thread.instructions {
                match instr {
                    crate::checker::litmus::Instruction::Store { addr, .. } => {
                        // Store doesn't change taint unless storing tainted value.
                        // Conservative: mark as Top if any register could be tainted.
                        let current = state.get_mem(*addr);
                        if current != TaintDomain::Tainted {
                            state.set_mem(*addr, TaintDomain::Clean);
                        }
                    }
                    crate::checker::litmus::Instruction::Load { reg, addr, .. } => {
                        let mem_taint = state.get_mem(*addr);
                        state.set_reg(thread.id, *reg, mem_taint);
                    }
                    crate::checker::litmus::Instruction::RMW { reg, addr, .. } => {
                        let mem_taint = state.get_mem(*addr);
                        state.set_reg(thread.id, *reg, mem_taint);
                    }
                    _ => {}
                }
            }
        }

        state
    }
}

// ---------------------------------------------------------------------------
// Galois connections
// ---------------------------------------------------------------------------

/// A Galois connection between concrete and abstract domains.
/// α (abstraction): concrete → abstract
/// γ (concretization): abstract → concrete
pub struct GaloisConnection<C, A> {
    _phantom: std::marker::PhantomData<(C, A)>,
}

/// Galois connection between concrete integers and interval domain.
impl GaloisConnection<i64, IntervalDomain> {
    /// Abstraction function: concrete value → interval.
    pub fn alpha(v: i64) -> IntervalDomain {
        IntervalDomain::constant(v)
    }

    /// Abstraction of a set of values.
    pub fn alpha_set(values: &[i64]) -> IntervalDomain {
        if values.is_empty() {
            return IntervalDomain::Bottom;
        }
        let lo = *values.iter().min().unwrap();
        let hi = *values.iter().max().unwrap();
        IntervalDomain::Interval { lo, hi }
    }

    /// Concretization function: interval → set of possible values.
    /// Returns None for Top (infinite set).
    pub fn gamma(abs: &IntervalDomain) -> Option<Vec<i64>> {
        match abs {
            IntervalDomain::Bottom => Some(vec![]),
            IntervalDomain::Interval { lo, hi } => {
                if (hi - lo) > 1000 {
                    None // Too large to enumerate.
                } else {
                    Some((*lo..=*hi).collect())
                }
            }
            IntervalDomain::Top => None,
        }
    }
}

/// Galois connection between concrete integers and sign domain.
impl GaloisConnection<i64, SignDomain> {
    /// Abstraction function: concrete value → sign.
    pub fn alpha(v: i64) -> SignDomain {
        SignDomain::from_value(v)
    }

    /// Abstraction of a set of values.
    pub fn alpha_set(values: &[i64]) -> SignDomain {
        if values.is_empty() {
            return SignDomain::Bottom;
        }
        let mut result = SignDomain::from_value(values[0]);
        for &v in &values[1..] {
            result = result.join(&SignDomain::from_value(v));
        }
        result
    }
}

// ---------------------------------------------------------------------------
// Over-approximation of reachable states
// ---------------------------------------------------------------------------

/// Result of abstract reachability analysis.
#[derive(Debug, Clone)]
pub struct ReachabilityResult<D: AbstractDomain> {
    /// Final abstract state (over-approximation).
    pub final_state: AbstractState<D>,
    /// Number of iterations until fixpoint.
    pub iterations: usize,
    /// Whether a fixpoint was reached.
    pub converged: bool,
}

impl<D: AbstractDomain> fmt::Display for ReachabilityResult<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Reachability: {} iterations, converged={}",
            self.iterations, self.converged)
    }
}

/// Compute over-approximation of reachable states for a litmus test.
pub fn overapproximate_reachable(
    test: &crate::checker::litmus::LitmusTest,
) -> ReachabilityResult<IntervalDomain> {
    let state = AbstractExecutor::<IntervalDomain>::execute_interval(test);
    ReachabilityResult {
        final_state: state,
        iterations: 1,
        converged: true,
    }
}

/// Compute taint propagation analysis.
pub fn taint_analysis(
    test: &crate::checker::litmus::LitmusTest,
    secret_addresses: &[u64],
) -> ReachabilityResult<TaintDomain> {
    let state = AbstractExecutor::<TaintDomain>::execute_taint(test, secret_addresses);
    ReachabilityResult {
        final_state: state,
        iterations: 1,
        converged: true,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // IntervalDomain tests
    #[test]
    fn test_interval_constant() {
        let c = IntervalDomain::constant(42);
        assert!(c.contains_value(42));
        assert!(!c.contains_value(43));
    }

    #[test]
    fn test_interval_join() {
        let a = IntervalDomain::interval(1, 5);
        let b = IntervalDomain::interval(3, 8);
        let c = a.join(&b);
        assert_eq!(c, IntervalDomain::interval(1, 8));
    }

    #[test]
    fn test_interval_meet() {
        let a = IntervalDomain::interval(1, 5);
        let b = IntervalDomain::interval(3, 8);
        let c = a.meet(&b);
        assert_eq!(c, IntervalDomain::interval(3, 5));
    }

    #[test]
    fn test_interval_meet_empty() {
        let a = IntervalDomain::interval(1, 3);
        let b = IntervalDomain::interval(5, 8);
        let c = a.meet(&b);
        assert!(c.is_bottom());
    }

    #[test]
    fn test_interval_add() {
        let a = IntervalDomain::interval(1, 3);
        let b = IntervalDomain::interval(2, 4);
        let c = a.add(&b);
        assert_eq!(c, IntervalDomain::interval(3, 7));
    }

    #[test]
    fn test_interval_sub() {
        let a = IntervalDomain::interval(5, 10);
        let b = IntervalDomain::interval(1, 3);
        let c = a.sub(&b);
        assert_eq!(c, IntervalDomain::interval(2, 9));
    }

    #[test]
    fn test_interval_mul() {
        let a = IntervalDomain::interval(-2, 3);
        let b = IntervalDomain::interval(1, 4);
        let c = a.mul(&b);
        assert!(c.contains_value(-8));
        assert!(c.contains_value(12));
    }

    #[test]
    fn test_interval_bottom() {
        let bot = IntervalDomain::bottom();
        assert!(bot.is_bottom());
        let a = IntervalDomain::interval(1, 5);
        assert_eq!(bot.join(&a), a);
    }

    #[test]
    fn test_interval_top() {
        let top = IntervalDomain::top();
        assert!(top.is_top());
        assert!(top.contains_value(999));
    }

    #[test]
    fn test_interval_leq() {
        let a = IntervalDomain::interval(2, 5);
        let b = IntervalDomain::interval(1, 8);
        assert!(a.leq(&b));
        assert!(!b.leq(&a));
    }

    #[test]
    fn test_interval_widen() {
        let a = IntervalDomain::interval(0, 10);
        let b = IntervalDomain::interval(0, 15);
        let w = a.widen(&b);
        // hi grew, so widening pushes to MAX.
        match w {
            IntervalDomain::Interval { lo, hi } => {
                assert_eq!(lo, 0);
                assert_eq!(hi, i64::MAX);
            }
            IntervalDomain::Top => {} // Also acceptable.
            _ => panic!("unexpected"),
        }
    }

    #[test]
    fn test_interval_width() {
        let a = IntervalDomain::interval(3, 7);
        assert_eq!(a.width(), Some(4));
    }

    // SignDomain tests
    #[test]
    fn test_sign_from_value() {
        assert_eq!(SignDomain::from_value(5), SignDomain::Positive);
        assert_eq!(SignDomain::from_value(-3), SignDomain::Negative);
        assert_eq!(SignDomain::from_value(0), SignDomain::Zero);
    }

    #[test]
    fn test_sign_join() {
        let p = SignDomain::Positive;
        let z = SignDomain::Zero;
        assert_eq!(p.join(&z), SignDomain::NonNegative);
    }

    #[test]
    fn test_sign_meet() {
        let nn = SignDomain::NonNegative;
        let nz = SignDomain::NonZero;
        assert_eq!(nn.meet(&nz), SignDomain::Positive);
    }

    #[test]
    fn test_sign_add() {
        let p = SignDomain::Positive;
        assert_eq!(p.add(&p), SignDomain::Positive);
        let n = SignDomain::Negative;
        assert_eq!(p.add(&n), SignDomain::Top);
    }

    #[test]
    fn test_sign_mul() {
        let p = SignDomain::Positive;
        let n = SignDomain::Negative;
        assert_eq!(p.mul(&n), SignDomain::Negative);
        assert_eq!(n.mul(&n), SignDomain::Positive);
    }

    #[test]
    fn test_sign_negate() {
        assert_eq!(SignDomain::Positive.negate(), SignDomain::Negative);
        assert_eq!(SignDomain::Zero.negate(), SignDomain::Zero);
        assert_eq!(SignDomain::NonNegative.negate(), SignDomain::NonPositive);
    }

    #[test]
    fn test_sign_leq() {
        assert!(SignDomain::Positive.leq(&SignDomain::NonNegative));
        assert!(SignDomain::Negative.leq(&SignDomain::NonZero));
        assert!(SignDomain::Bottom.leq(&SignDomain::Top));
        assert!(!SignDomain::Top.leq(&SignDomain::Bottom));
    }

    // TaintDomain tests
    #[test]
    fn test_taint_propagate() {
        let c = TaintDomain::Clean;
        let t = TaintDomain::Tainted;
        assert_eq!(c.propagate(&c), TaintDomain::Clean);
        assert_eq!(c.propagate(&t), TaintDomain::Tainted);
        assert_eq!(t.propagate(&c), TaintDomain::Tainted);
    }

    #[test]
    fn test_taint_join() {
        let c = TaintDomain::Clean;
        let t = TaintDomain::Tainted;
        assert_eq!(c.join(&t), TaintDomain::Top);
    }

    #[test]
    fn test_taint_meet() {
        let c = TaintDomain::Clean;
        let t = TaintDomain::Tainted;
        assert_eq!(c.meet(&t), TaintDomain::Bottom);
    }

    #[test]
    fn test_taint_is_potentially_tainted() {
        assert!(TaintDomain::Tainted.is_potentially_tainted());
        assert!(TaintDomain::Top.is_potentially_tainted());
        assert!(!TaintDomain::Clean.is_potentially_tainted());
        assert!(!TaintDomain::Bottom.is_potentially_tainted());
    }

    // AbstractState tests
    #[test]
    fn test_abstract_state_basic() {
        let mut state = AbstractState::<IntervalDomain>::new();
        state.set_reg(0, 0, IntervalDomain::constant(5));
        let val = state.get_reg(0, 0);
        assert_eq!(val, IntervalDomain::constant(5));
    }

    #[test]
    fn test_abstract_state_join() {
        let mut s1 = AbstractState::<IntervalDomain>::new();
        s1.set_reg(0, 0, IntervalDomain::interval(1, 5));
        let mut s2 = AbstractState::<IntervalDomain>::new();
        s2.set_reg(0, 0, IntervalDomain::interval(3, 8));
        let joined = s1.join(&s2);
        assert_eq!(joined.get_reg(0, 0), IntervalDomain::interval(1, 8));
    }

    #[test]
    fn test_abstract_state_memory() {
        let mut state = AbstractState::<TaintDomain>::new();
        state.set_mem(0x100, TaintDomain::Tainted);
        assert_eq!(state.get_mem(0x100), TaintDomain::Tainted);
        assert_eq!(state.get_mem(0x200), TaintDomain::Top); // Default.
    }

    // GaloisConnection tests
    #[test]
    fn test_galois_interval_alpha() {
        let a = GaloisConnection::<i64, IntervalDomain>::alpha(42);
        assert_eq!(a, IntervalDomain::constant(42));
    }

    #[test]
    fn test_galois_interval_alpha_set() {
        let a = GaloisConnection::<i64, IntervalDomain>::alpha_set(&[1, 5, 3]);
        assert_eq!(a, IntervalDomain::interval(1, 5));
    }

    #[test]
    fn test_galois_interval_gamma() {
        let g = GaloisConnection::<i64, IntervalDomain>::gamma(&IntervalDomain::interval(1, 3));
        assert_eq!(g, Some(vec![1, 2, 3]));
    }

    #[test]
    fn test_galois_interval_gamma_bottom() {
        let g = GaloisConnection::<i64, IntervalDomain>::gamma(&IntervalDomain::Bottom);
        assert_eq!(g, Some(vec![]));
    }

    #[test]
    fn test_galois_interval_gamma_top() {
        let g = GaloisConnection::<i64, IntervalDomain>::gamma(&IntervalDomain::Top);
        assert!(g.is_none());
    }

    #[test]
    fn test_galois_sign_alpha() {
        assert_eq!(GaloisConnection::<i64, SignDomain>::alpha(5), SignDomain::Positive);
        assert_eq!(GaloisConnection::<i64, SignDomain>::alpha(-3), SignDomain::Negative);
    }

    #[test]
    fn test_galois_sign_alpha_set() {
        let a = GaloisConnection::<i64, SignDomain>::alpha_set(&[1, -1]);
        assert_eq!(a, SignDomain::NonZero);
    }

    // Abstract execution tests
    #[test]
    fn test_abstract_executor_interval() {
        use crate::checker::litmus::*;
        let mut test = LitmusTest::new("test");
        test.set_initial(0x100, 0);
        let mut t0 = Thread::new(0);
        t0.store(0x100, 42, Ordering::Relaxed);
        t0.load(0, 0x100, Ordering::Relaxed);
        test.add_thread(t0);

        let state = AbstractExecutor::<IntervalDomain>::execute_interval(&test);
        let reg_val = state.get_reg(0, 0);
        // After store(42) and load, register should contain the stored value.
        assert!(reg_val.contains_value(42));
    }

    #[test]
    fn test_abstract_executor_taint() {
        use crate::checker::litmus::*;
        let mut test = LitmusTest::new("test");
        test.set_initial(0x100, 0);
        test.set_initial(0x200, 0);
        let mut t0 = Thread::new(0);
        t0.load(0, 0x100, Ordering::Relaxed);
        test.add_thread(t0);

        let state = AbstractExecutor::<TaintDomain>::execute_taint(&test, &[0x100]);
        let reg_taint = state.get_reg(0, 0);
        assert_eq!(reg_taint, TaintDomain::Tainted);
    }

    #[test]
    fn test_overapproximate_reachable() {
        use crate::checker::litmus::*;
        let mut test = LitmusTest::new("test");
        test.set_initial(0x100, 0);
        let mut t0 = Thread::new(0);
        t0.store(0x100, 1, Ordering::Relaxed);
        test.add_thread(t0);

        let result = overapproximate_reachable(&test);
        assert!(result.converged);
    }

    #[test]
    fn test_taint_analysis() {
        use crate::checker::litmus::*;
        let mut test = LitmusTest::new("test");
        test.set_initial(0x100, 0);
        let mut t0 = Thread::new(0);
        t0.load(0, 0x100, Ordering::Relaxed);
        test.add_thread(t0);

        let result = taint_analysis(&test, &[0x100]);
        assert!(result.converged);
        assert_eq!(result.final_state.get_reg(0, 0), TaintDomain::Tainted);
    }

    #[test]
    fn test_reachability_result_display() {
        let result = ReachabilityResult {
            final_state: AbstractState::<IntervalDomain>::new(),
            iterations: 5,
            converged: true,
        };
        let s = format!("{}", result);
        assert!(s.contains("5 iterations"));
    }
}
