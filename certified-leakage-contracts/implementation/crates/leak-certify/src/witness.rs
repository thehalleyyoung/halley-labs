//! Checkable evidence: fixpoint traces, counting arguments, reductions, and compositions.
//!
//! A [`Witness`] is a machine-checkable proof artifact that a [`Claim`](crate::Claim)
//! holds.  The [`WitnessChecker`] re-checks a witness without re-running the full
//! analysis, enabling lightweight independent verification.

use serde::{Deserialize, Serialize};

use shared_types::FunctionId;

use crate::certificate::CertificateHash;

// ---------------------------------------------------------------------------
// Witness (enum)
// ---------------------------------------------------------------------------

/// Top-level witness discriminator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Witness {
    /// Evidence from abstract-interpretation fixpoint computation.
    Fixpoint(FixpointWitness),
    /// Evidence via a counting / combinatorial argument.
    Counting(CountingWitness),
    /// Evidence via reduction to a known-hard problem.
    Reduction(ReductionWitness),
    /// Evidence via composing sub-witnesses.
    Composition(CompositionWitness),
}

impl Witness {
    /// Short human-readable label for the variant.
    pub fn kind_name(&self) -> &'static str {
        match self {
            Witness::Fixpoint(_) => "fixpoint",
            Witness::Counting(_) => "counting",
            Witness::Reduction(_) => "reduction",
            Witness::Composition(_) => "composition",
        }
    }
}

// ---------------------------------------------------------------------------
// FixpointWitness
// ---------------------------------------------------------------------------

/// A witness recording successive abstract states converging to a fixpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixpointWitness {
    /// The function this fixpoint was computed for.
    pub function_id: FunctionId,
    /// Number of iterations until convergence.
    pub iterations: usize,
    /// Abstract state at each widening point, serialised as JSON.
    pub widening_states: Vec<serde_json::Value>,
    /// The final fixpoint state.
    pub fixpoint_state: serde_json::Value,
    /// Whether the fixpoint was reached (vs. hitting iteration limit).
    pub converged: bool,
}

impl FixpointWitness {
    /// Quick plausibility check: converged and at least one iteration.
    pub fn is_plausible(&self) -> bool {
        self.converged && self.iterations > 0
    }
}

// ---------------------------------------------------------------------------
// CountingWitness
// ---------------------------------------------------------------------------

/// A witness using a counting / combinatorial bound on observable states.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CountingWitness {
    /// The function being bounded.
    pub function_id: FunctionId,
    /// Number of distinct observable cache-state equivalence classes.
    pub equivalence_classes: u64,
    /// Resulting leakage bound in bits: log2(equivalence_classes).
    pub leakage_bits: f64,
    /// Free-form justification of the counting argument.
    pub justification: String,
}

impl CountingWitness {
    /// Re-derive the leakage bound from the class count.
    pub fn recompute_bound(&self) -> f64 {
        (self.equivalence_classes as f64).log2()
    }

    /// Check that the stored bound is consistent with the class count.
    pub fn is_consistent(&self) -> bool {
        (self.leakage_bits - self.recompute_bound()).abs() < 1e-9
    }
}

// ---------------------------------------------------------------------------
// ReductionWitness
// ---------------------------------------------------------------------------

/// A witness via reduction: if the system leaks more than claimed, a hard
/// problem can be solved.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReductionWitness {
    /// The function being bounded.
    pub function_id: FunctionId,
    /// Name of the hard problem used in the reduction (e.g. "DDH", "LWE").
    pub hard_problem: String,
    /// Security parameter λ.
    pub security_parameter: u32,
    /// Serialised reduction mapping.
    pub reduction_mapping: serde_json::Value,
    /// Reference to a published proof or paper.
    pub reference: Option<String>,
}

impl ReductionWitness {
    /// Sanity check that essential fields are populated.
    pub fn is_well_formed(&self) -> bool {
        !self.hard_problem.is_empty() && self.security_parameter > 0
    }
}

// ---------------------------------------------------------------------------
// CompositionWitness
// ---------------------------------------------------------------------------

/// A witness built by composing sub-witnesses for individual components.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositionWitness {
    /// Hashes of the sub-certificates being composed.
    pub component_hashes: Vec<CertificateHash>,
    /// Composition rule applied (e.g. "sequential", "parallel", "hybrid").
    pub composition_rule: String,
    /// Resulting aggregate leakage bound in bits.
    pub aggregate_leakage_bits: f64,
    /// Per-component leakage contributions (same order as `component_hashes`).
    pub component_leakage_bits: Vec<f64>,
}

impl CompositionWitness {
    /// Verify that the aggregate bound is at least the sum of component bounds.
    pub fn is_consistent(&self) -> bool {
        let sum: f64 = self.component_leakage_bits.iter().sum();
        self.aggregate_leakage_bits >= sum - 1e-9
    }

    /// Number of components in this composition.
    pub fn component_count(&self) -> usize {
        self.component_hashes.len()
    }
}

// ---------------------------------------------------------------------------
// WitnessChecker
// ---------------------------------------------------------------------------

/// Result of checking a single witness.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WitnessCheckResult {
    /// Whether the witness passed all checks.
    pub passed: bool,
    /// Diagnostic messages (may be non-empty even on success).
    pub messages: Vec<String>,
}

/// Lightweight independent re-checker for [`Witness`] artifacts.
#[derive(Debug, Clone)]
pub struct WitnessChecker {
    /// Maximum iterations to allow before considering a fixpoint witness suspect.
    pub max_fixpoint_iterations: usize,
}

impl Default for WitnessChecker {
    fn default() -> Self {
        Self {
            max_fixpoint_iterations: 100_000,
        }
    }
}

impl WitnessChecker {
    /// Create a new checker with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Check a witness and return a diagnostic result.
    pub fn check(&self, witness: &Witness) -> WitnessCheckResult {
        match witness {
            Witness::Fixpoint(w) => self.check_fixpoint(w),
            Witness::Counting(w) => self.check_counting(w),
            Witness::Reduction(w) => self.check_reduction(w),
            Witness::Composition(w) => self.check_composition(w),
        }
    }

    fn check_fixpoint(&self, w: &FixpointWitness) -> WitnessCheckResult {
        let mut messages = Vec::new();
        let mut passed = true;

        if !w.converged {
            messages.push("fixpoint did not converge".into());
            passed = false;
        }
        if w.iterations > self.max_fixpoint_iterations {
            messages.push(format!(
                "iteration count {} exceeds maximum {}",
                w.iterations, self.max_fixpoint_iterations
            ));
            passed = false;
        }

        WitnessCheckResult { passed, messages }
    }

    fn check_counting(&self, w: &CountingWitness) -> WitnessCheckResult {
        let mut messages = Vec::new();
        let passed = w.is_consistent();
        if !passed {
            messages.push(format!(
                "stored bound {:.6} does not match recomputed {:.6}",
                w.leakage_bits,
                w.recompute_bound()
            ));
        }
        WitnessCheckResult { passed, messages }
    }

    fn check_reduction(&self, w: &ReductionWitness) -> WitnessCheckResult {
        let mut messages = Vec::new();
        let passed = w.is_well_formed();
        if !passed {
            messages.push("reduction witness missing hard_problem or security_parameter".into());
        }
        WitnessCheckResult { passed, messages }
    }

    fn check_composition(&self, w: &CompositionWitness) -> WitnessCheckResult {
        let mut messages = Vec::new();
        let mut passed = true;

        if w.component_hashes.len() != w.component_leakage_bits.len() {
            messages.push("component_hashes and component_leakage_bits length mismatch".into());
            passed = false;
        }
        if !w.is_consistent() {
            messages.push("aggregate bound is less than sum of components".into());
            passed = false;
        }

        WitnessCheckResult { passed, messages }
    }
}
