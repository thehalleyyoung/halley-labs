//! Leakage bound computation and composition.
//!
//! Provides per-function, compositional, and whole-library leakage bounds
//! that can be soundly combined for modular verification.

use std::collections::BTreeMap;
use std::fmt;

use num_rational::Rational64;
use serde::{Deserialize, Serialize};
use shared_types::FunctionId;

use crate::counting::CountBound;
use crate::entropy::EntropyBound;
use crate::metrics::BitsLeaked;
use crate::{QuantifyError, QuantifyResult};

// ---------------------------------------------------------------------------
// Leakage Bound
// ---------------------------------------------------------------------------

/// A quantified upper bound on information leakage for a program component.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeakageBound {
    /// Upper bound on leakage in bits.
    pub bits: f64,
    /// Exact rational bound when available (for compositional soundness).
    #[serde(skip)]
    pub exact_bits: Option<Rational64>,
    /// How this bound was derived.
    pub derivation: BoundDerivation,
    /// Confidence: 1.0 = provably sound, < 1.0 = heuristic.
    pub confidence: f64,
    /// Human-readable justification or proof sketch.
    pub justification: String,
}

/// How a leakage bound was derived.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BoundDerivation {
    /// Derived from entropy computation.
    Entropy,
    /// Derived from counting distinguishable states.
    Counting,
    /// Derived from channel capacity computation.
    ChannelCapacity,
    /// Derived from compositional reasoning over sub-bounds.
    Compositional,
    /// Manual annotation.
    Annotation,
    /// Combined from multiple derivation strategies.
    Combined(Vec<BoundDerivation>),
}

impl LeakageBound {
    /// Create a new leakage bound.
    pub fn new(bits: f64, derivation: BoundDerivation, justification: impl Into<String>) -> Self {
        Self {
            bits,
            exact_bits: None,
            derivation,
            confidence: 1.0,
            justification: justification.into(),
        }
    }

    /// Create from an entropy bound.
    pub fn from_entropy(bound: &EntropyBound) -> Self {
        Self {
            bits: bound.bits,
            exact_bits: None,
            derivation: BoundDerivation::Entropy,
            confidence: bound.confidence,
            justification: bound.justification.clone(),
        }
    }

    /// Create from a count bound.
    pub fn from_count(bound: &CountBound) -> Self {
        Self {
            bits: bound.leakage_bits,
            exact_bits: bound.exact,
            derivation: BoundDerivation::Counting,
            confidence: 1.0,
            justification: bound.justification.clone(),
        }
    }

    /// Set exact rational bits.
    pub fn with_exact(mut self, exact: Rational64) -> Self {
        self.exact_bits = Some(exact);
        self
    }

    /// Set confidence level.
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// A zero-leakage bound (constant-time).
    pub fn zero() -> Self {
        Self::new(0.0, BoundDerivation::Annotation, "constant-time")
    }

    /// Tighten with another bound (take the minimum).
    pub fn tighten(&self, other: &LeakageBound) -> LeakageBound {
        if self.bits <= other.bits {
            self.clone()
        } else {
            other.clone()
        }
    }

    /// Widen: take the maximum of two bounds (sound over-approximation).
    pub fn widen(&self, other: &LeakageBound) -> LeakageBound {
        if self.bits >= other.bits {
            self.clone()
        } else {
            other.clone()
        }
    }

    /// Add two bounds (for sequential composition).
    pub fn add(&self, other: &LeakageBound) -> LeakageBound {
        let bits = self.bits + other.bits;
        let exact = match (self.exact_bits, other.exact_bits) {
            (Some(a), Some(b)) => Some(a + b),
            _ => None,
        };
        LeakageBound {
            bits,
            exact_bits: exact,
            derivation: BoundDerivation::Compositional,
            confidence: self.confidence.min(other.confidence),
            justification: format!(
                "sum of ({}) and ({})",
                self.justification, other.justification
            ),
        }
    }

    /// Whether this bound certifies zero leakage (constant-time).
    pub fn is_zero(&self) -> bool {
        self.bits <= 1e-15
    }

    /// The bound value in bits.
    pub fn value(&self) -> f64 {
        self.bits
    }
}

impl fmt::Display for LeakageBound {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "≤ {:.4} bits", self.bits)?;
        if self.confidence < 1.0 {
            write!(f, " (conf={:.2})", self.confidence)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Bound Computation
// ---------------------------------------------------------------------------

/// Orchestrates leakage bound computation using multiple strategies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundComputation {
    /// Strategies attempted, in order.
    pub strategies: Vec<BoundStrategy>,
    /// Best (tightest) bound found, if any.
    pub best_bound: Option<LeakageBound>,
    /// Computation time in milliseconds.
    pub elapsed_ms: u64,
}

/// A strategy used during bound computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundStrategy {
    /// Strategy name.
    pub name: String,
    /// Whether this strategy succeeded.
    pub succeeded: bool,
    /// The bound produced, if successful.
    pub bound: Option<LeakageBound>,
    /// Error message if the strategy failed.
    pub error: Option<String>,
}

impl BoundComputation {
    /// Create a new empty computation.
    pub fn new() -> Self {
        Self {
            strategies: Vec::new(),
            best_bound: None,
            elapsed_ms: 0,
        }
    }

    /// Record a successful strategy result.
    pub fn record_success(&mut self, name: impl Into<String>, bound: LeakageBound) {
        let name = name.into();
        self.strategies.push(BoundStrategy {
            name,
            succeeded: true,
            bound: Some(bound.clone()),
            error: None,
        });
        self.best_bound = Some(match &self.best_bound {
            Some(existing) => existing.tighten(&bound),
            None => bound,
        });
    }

    /// Record a failed strategy.
    pub fn record_failure(&mut self, name: impl Into<String>, error: impl Into<String>) {
        self.strategies.push(BoundStrategy {
            name: name.into(),
            succeeded: false,
            bound: None,
            error: Some(error.into()),
        });
    }

    /// The best bound found, or an error if none succeeded.
    pub fn result(&self) -> QuantifyResult<&LeakageBound> {
        self.best_bound.as_ref().ok_or_else(|| {
            QuantifyError::BoundComputationFailed("no strategy produced a bound".into())
        })
    }

    /// Number of strategies attempted.
    pub fn num_strategies(&self) -> usize {
        self.strategies.len()
    }

    /// Number of strategies that succeeded.
    pub fn num_succeeded(&self) -> usize {
        self.strategies.iter().filter(|s| s.succeeded).count()
    }
}

impl Default for BoundComputation {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for BoundComputation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.best_bound {
            Some(b) => write!(
                f,
                "BoundComputation({}/{} ok, best={})",
                self.num_succeeded(),
                self.num_strategies(),
                b,
            ),
            None => write!(
                f,
                "BoundComputation({} strategies, no bound)",
                self.num_strategies()
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// Per-Function Bound
// ---------------------------------------------------------------------------

/// Leakage bound for a single function.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerFunctionBound {
    /// Function identifier.
    pub function_id: String,
    /// Function name (human-readable).
    pub function_name: String,
    /// The computed leakage bound.
    pub bound: LeakageBound,
    /// Number of secret-dependent branches in this function.
    pub secret_branches: usize,
    /// Number of secret-dependent memory accesses.
    pub secret_accesses: usize,
    /// Whether this function is certified constant-time.
    pub is_constant_time: bool,
}

impl PerFunctionBound {
    /// Create a new per-function bound.
    pub fn new(
        function_id: impl Into<String>,
        function_name: impl Into<String>,
        bound: LeakageBound,
    ) -> Self {
        let is_ct = bound.is_zero();
        Self {
            function_id: function_id.into(),
            function_name: function_name.into(),
            bound,
            secret_branches: 0,
            secret_accesses: 0,
            is_constant_time: is_ct,
        }
    }

    /// Set the number of secret-dependent branches.
    pub fn with_secret_branches(mut self, n: usize) -> Self {
        self.secret_branches = n;
        self.is_constant_time = self.bound.is_zero() && n == 0 && self.secret_accesses == 0;
        self
    }

    /// Set the number of secret-dependent memory accesses.
    pub fn with_secret_accesses(mut self, n: usize) -> Self {
        self.secret_accesses = n;
        self.is_constant_time = self.bound.is_zero() && self.secret_branches == 0 && n == 0;
        self
    }

    /// The leakage bound in bits.
    pub fn bits(&self) -> f64 {
        self.bound.bits
    }
}

impl fmt::Display for PerFunctionBound {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_constant_time {
            write!(f, "{}: constant-time ✓", self.function_name)
        } else {
            write!(f, "{}: {}", self.function_name, self.bound)
        }
    }
}

// ---------------------------------------------------------------------------
// Compositional Bound
// ---------------------------------------------------------------------------

/// Compositional bound over a call graph.
///
/// Combines per-function bounds using the call graph structure to derive a
/// sound bound for a function and all its callees.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositionalBound {
    /// Root function of the composition.
    pub root_function: String,
    /// Per-function bounds for all functions in the call subtree.
    pub function_bounds: BTreeMap<String, PerFunctionBound>,
    /// The composed bound for the root.
    pub composed_bound: LeakageBound,
    /// Call depth of the composition.
    pub call_depth: usize,
}

impl CompositionalBound {
    /// Create a new compositional bound.
    pub fn new(root_function: impl Into<String>) -> Self {
        Self {
            root_function: root_function.into(),
            function_bounds: BTreeMap::new(),
            composed_bound: LeakageBound::zero(),
            call_depth: 0,
        }
    }

    /// Add a per-function bound.
    pub fn add_function_bound(&mut self, bound: PerFunctionBound) {
        self.function_bounds
            .insert(bound.function_id.clone(), bound);
    }

    /// Compose all function bounds by summing (conservative).
    pub fn compose_sum(&mut self) {
        let total: f64 = self.function_bounds.values().map(|fb| fb.bits()).sum();
        let min_confidence = self
            .function_bounds
            .values()
            .map(|fb| fb.bound.confidence)
            .fold(1.0_f64, f64::min);
        self.composed_bound = LeakageBound {
            bits: total,
            exact_bits: None,
            derivation: BoundDerivation::Compositional,
            confidence: min_confidence,
            justification: format!(
                "sum of {} function bounds",
                self.function_bounds.len()
            ),
        };
    }

    /// The composed bound in bits.
    pub fn bits(&self) -> f64 {
        self.composed_bound.bits
    }

    /// Number of functions in the composition.
    pub fn num_functions(&self) -> usize {
        self.function_bounds.len()
    }

    /// Whether all functions in the composition are constant-time.
    pub fn all_constant_time(&self) -> bool {
        self.function_bounds.values().all(|fb| fb.is_constant_time)
    }
}

impl fmt::Display for CompositionalBound {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CompositionalBound({}, {} funcs, {})",
            self.root_function,
            self.num_functions(),
            self.composed_bound,
        )
    }
}

// ---------------------------------------------------------------------------
// Whole Library Bound
// ---------------------------------------------------------------------------

/// Whole-library leakage bound aggregating all entry-point compositions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WholeLibraryBound {
    /// Library or crate name.
    pub library_name: String,
    /// Per-entry-point compositional bounds.
    pub entry_bounds: BTreeMap<String, CompositionalBound>,
    /// Overall worst-case bound across all entry points.
    pub worst_case_bound: LeakageBound,
    /// Total number of functions analyzed.
    pub total_functions: usize,
    /// Number of functions certified constant-time.
    pub constant_time_functions: usize,
}

impl WholeLibraryBound {
    /// Create a new whole-library bound.
    pub fn new(library_name: impl Into<String>) -> Self {
        Self {
            library_name: library_name.into(),
            entry_bounds: BTreeMap::new(),
            worst_case_bound: LeakageBound::zero(),
            total_functions: 0,
            constant_time_functions: 0,
        }
    }

    /// Add a compositional bound for an entry point.
    pub fn add_entry_bound(&mut self, entry_name: impl Into<String>, bound: CompositionalBound) {
        let name = entry_name.into();
        self.entry_bounds.insert(name, bound);
        self.recompute();
    }

    /// Recompute aggregate statistics.
    fn recompute(&mut self) {
        let mut worst = 0.0_f64;
        let mut total_funcs = 0usize;
        let mut ct_funcs = 0usize;

        for cb in self.entry_bounds.values() {
            worst = worst.max(cb.bits());
            for fb in cb.function_bounds.values() {
                total_funcs += 1;
                if fb.is_constant_time {
                    ct_funcs += 1;
                }
            }
        }

        self.worst_case_bound = if worst <= 1e-15 {
            LeakageBound::zero()
        } else {
            LeakageBound::new(
                worst,
                BoundDerivation::Compositional,
                "worst-case across entry points",
            )
        };
        self.total_functions = total_funcs;
        self.constant_time_functions = ct_funcs;
    }

    /// The worst-case leakage across all entry points, in bits.
    pub fn worst_case_bits(&self) -> f64 {
        self.worst_case_bound.bits
    }

    /// Number of entry points analyzed.
    pub fn num_entries(&self) -> usize {
        self.entry_bounds.len()
    }

    /// Fraction of functions that are constant-time.
    pub fn constant_time_fraction(&self) -> f64 {
        if self.total_functions == 0 {
            return 1.0;
        }
        self.constant_time_functions as f64 / self.total_functions as f64
    }

    /// Whether the entire library is certified constant-time.
    pub fn is_constant_time(&self) -> bool {
        self.worst_case_bound.is_zero()
    }
}

impl fmt::Display for WholeLibraryBound {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "WholeLibraryBound({}: {} entries, worst={}, CT={}/{})",
            self.library_name,
            self.num_entries(),
            self.worst_case_bound,
            self.constant_time_functions,
            self.total_functions,
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_leakage_bound_zero() {
        let b = LeakageBound::zero();
        assert!(b.is_zero());
        assert!((b.value() - 0.0).abs() < 1e-15);
    }

    #[test]
    fn test_leakage_bound_tighten() {
        let b1 = LeakageBound::new(3.0, BoundDerivation::Entropy, "entropy");
        let b2 = LeakageBound::new(2.5, BoundDerivation::Counting, "counting");
        let tight = b1.tighten(&b2);
        assert!((tight.value() - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_leakage_bound_add() {
        let b1 = LeakageBound::new(1.0, BoundDerivation::Entropy, "a");
        let b2 = LeakageBound::new(2.0, BoundDerivation::Entropy, "b");
        let sum = b1.add(&b2);
        assert!((sum.value() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_bound_computation() {
        let mut bc = BoundComputation::new();
        bc.record_failure("strategy_a", "not applicable");
        bc.record_success(
            "strategy_b",
            LeakageBound::new(4.0, BoundDerivation::Counting, "counting"),
        );
        bc.record_success(
            "strategy_c",
            LeakageBound::new(3.0, BoundDerivation::Entropy, "entropy"),
        );
        assert_eq!(bc.num_strategies(), 3);
        assert_eq!(bc.num_succeeded(), 2);
        assert!((bc.result().unwrap().value() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_per_function_bound() {
        let fb = PerFunctionBound::new(
            "fn_1",
            "aes_encrypt",
            LeakageBound::zero(),
        );
        assert!(fb.is_constant_time);
    }

    #[test]
    fn test_compositional_bound() {
        let mut cb = CompositionalBound::new("main");
        cb.add_function_bound(PerFunctionBound::new(
            "f1",
            "func_a",
            LeakageBound::new(1.0, BoundDerivation::Entropy, "a"),
        ));
        cb.add_function_bound(PerFunctionBound::new(
            "f2",
            "func_b",
            LeakageBound::new(2.0, BoundDerivation::Counting, "b"),
        ));
        cb.compose_sum();
        assert!((cb.bits() - 3.0).abs() < 1e-10);
        assert!(!cb.all_constant_time());
    }

    #[test]
    fn test_whole_library_bound() {
        let mut wlb = WholeLibraryBound::new("libcrypto");
        let mut cb = CompositionalBound::new("entry1");
        cb.add_function_bound(PerFunctionBound::new(
            "f1",
            "encrypt",
            LeakageBound::new(0.5, BoundDerivation::Entropy, "e"),
        ));
        cb.compose_sum();
        wlb.add_entry_bound("entry1", cb);
        assert!((wlb.worst_case_bits() - 0.5).abs() < 1e-10);
        assert!(!wlb.is_constant_time());
    }
}
