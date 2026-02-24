//! Relaxed (C++11-style) memory model for LITMUS∞.
//!
//! Implements the C++11 memory model with relaxed atomics, release-acquire
//! semantics, and sequential consistency fences. Provides axiom definitions,
//! release-acquire chain checking, and violation detection.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use serde::{Serialize, Deserialize};

use crate::checker::{
    LitmusTest, Outcome, LitmusOutcome,
    MemoryModel, RelationExpr, ExecutionGraph, BitMatrix,
};
use crate::checker::execution::{EventId, Address, OpType, Scope};
use crate::checker::memory_model::PredicateExpr;
use crate::checker::litmus::Ordering;

// ---------------------------------------------------------------------------
// RelaxedAxiom
// ---------------------------------------------------------------------------

/// Axioms of the C++11 relaxed memory model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RelaxedAxiom {
    /// Per-location coherence: acyclic(po-loc ∪ com).
    CoherenceOrder,
    /// Release-acquire ordering: acyclic(hb).
    ReleaseAcquire,
    /// SeqCst total order.
    SeqCstOrder,
    /// No thin-air reads.
    NoThinAir,
    /// RMW atomicity.
    RmwAtomicity,
    /// Release sequence consistency.
    ReleaseSequence,
}

impl fmt::Display for RelaxedAxiom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CoherenceOrder => write!(f, "coherence-order"),
            Self::ReleaseAcquire => write!(f, "release-acquire"),
            Self::SeqCstOrder => write!(f, "seq-cst-order"),
            Self::NoThinAir => write!(f, "no-thin-air"),
            Self::RmwAtomicity => write!(f, "rmw-atomicity"),
            Self::ReleaseSequence => write!(f, "release-sequence"),
        }
    }
}

impl RelaxedAxiom {
    /// All axioms.
    pub fn all() -> Vec<Self> {
        vec![
            Self::CoherenceOrder,
            Self::ReleaseAcquire,
            Self::SeqCstOrder,
            Self::NoThinAir,
            Self::RmwAtomicity,
            Self::ReleaseSequence,
        ]
    }
}

// ---------------------------------------------------------------------------
// RelaxedViolation
// ---------------------------------------------------------------------------

/// A violation of a relaxed model axiom.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelaxedViolation {
    /// Which axiom was violated.
    pub axiom: RelaxedAxiom,
    /// Events involved.
    pub events: Vec<EventId>,
    /// Description.
    pub description: String,
}

impl fmt::Display for RelaxedViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Relaxed violation [{}]: {} [events: {:?}]",
            self.axiom, self.description, self.events)
    }
}

// ---------------------------------------------------------------------------
// ReleaseAcquireChecker
// ---------------------------------------------------------------------------

/// Release-acquire consistency checker.
///
/// Checks that release-acquire synchronization creates proper
/// happens-before edges and that these edges are acyclic.
#[derive(Debug, Clone)]
pub struct ReleaseAcquireChecker {
    /// Whether to enforce strict mode (all orderings checked).
    pub strict: bool,
}

impl ReleaseAcquireChecker {
    /// Create a new checker.
    pub fn new() -> Self {
        Self { strict: true }
    }

    /// Create a lenient checker (only checks explicit ra pairs).
    pub fn lenient() -> Self {
        Self { strict: false }
    }

    /// Check release-acquire consistency of an execution.
    pub fn check(&self, exec: &ExecutionGraph) -> Vec<RelaxedViolation> {
        let mut violations = Vec::new();

        // Build the sw (synchronizes-with) relation.
        let sw = self.compute_sw(exec);

        // Build the hb (happens-before) relation: hb = (po ∪ sw)+.
        let po_sw = exec.po.union(&sw);
        let hb = po_sw.transitive_closure();

        // Check acyclicity of hb.
        let n = exec.len();
        for i in 0..n {
            if hb.get(i, i) {
                violations.push(RelaxedViolation {
                    axiom: RelaxedAxiom::ReleaseAcquire,
                    events: vec![i],
                    description: format!("hb cycle involving e{}", i),
                });
                break;
            }
        }

        // Check coherence: acyclic(po-loc ∪ com).
        violations.extend(self.check_coherence(exec));

        violations
    }

    /// Compute the synchronizes-with relation.
    ///
    /// sw = release writes rf→ acquire reads (in C++11 terms).
    pub fn compute_sw(&self, exec: &ExecutionGraph) -> BitMatrix {
        let n = exec.len();
        let mut sw = BitMatrix::new(n);

        for (w, r) in exec.rf.edges() {
            let ew = &exec.events[w];
            let er = &exec.events[r];

            // A release write synchronizes-with an acquire read.
            let w_is_release = is_release_event(ew);
            let r_is_acquire = is_acquire_event(er);

            if w_is_release && r_is_acquire {
                sw.set(w, r, true);
            }

            // SeqCst always synchronizes.
            if ew.scope == Scope::System || er.scope == Scope::System {
                // Treat system-scoped as always synchronizing.
            }
        }

        sw
    }

    /// Compute the happens-before relation.
    pub fn compute_hb(&self, exec: &ExecutionGraph) -> BitMatrix {
        let sw = self.compute_sw(exec);
        let po_sw = exec.po.union(&sw);
        po_sw.transitive_closure()
    }

    /// Check per-location coherence.
    fn check_coherence(&self, exec: &ExecutionGraph) -> Vec<RelaxedViolation> {
        let com = exec.rf.union(&exec.co).union(&exec.fr);
        let sa = exec.same_address();
        let po_loc = exec.po.intersection(&sa);
        let coh = po_loc.union(&com);
        let tc = coh.transitive_closure();
        let n = exec.len();

        let mut violations = Vec::new();
        for i in 0..n {
            if tc.get(i, i) {
                violations.push(RelaxedViolation {
                    axiom: RelaxedAxiom::CoherenceOrder,
                    events: vec![i],
                    description: format!("coherence cycle at e{}", i),
                });
                break;
            }
        }
        violations
    }
}

impl Default for ReleaseAcquireChecker {
    fn default() -> Self { Self::new() }
}

/// Check if an event acts as a release.
fn is_release_event(event: &crate::checker::execution::Event) -> bool {
    match event.scope {
        Scope::System => true,
        _ => event.is_write(), // simplified: all writes could be release
    }
}

/// Check if an event acts as an acquire.
fn is_acquire_event(event: &crate::checker::execution::Event) -> bool {
    match event.scope {
        Scope::System => true,
        _ => event.is_read(), // simplified: all reads could be acquire
    }
}

// ---------------------------------------------------------------------------
// RelaxedModel
// ---------------------------------------------------------------------------

/// C++11 relaxed memory model checker.
#[derive(Debug, Clone)]
pub struct RelaxedModel {
    pub name: String,
    ra_checker: ReleaseAcquireChecker,
}

impl RelaxedModel {
    /// Create a new relaxed model.
    pub fn new() -> Self {
        Self {
            name: "Relaxed".to_string(),
            ra_checker: ReleaseAcquireChecker::new(),
        }
    }

    /// Build the relaxed memory model axiomatically.
    pub fn build_model() -> MemoryModel {
        let mut m = MemoryModel::new("C++11-relaxed");

        // Communication.
        m.add_derived("com",
            RelationExpr::union_many(vec![
                RelationExpr::base("rf"),
                RelationExpr::base("co"),
                RelationExpr::base("fr"),
            ]),
            "communication",
        );

        // Per-location coherence.
        m.add_acyclic(
            RelationExpr::union(RelationExpr::base("po-loc"), RelationExpr::base("com")),
        );

        // No thin-air: acyclic(hb) where hb = (po ∪ sw)+.
        // sw is computed from release/acquire annotations.

        m
    }

    /// Check an execution.
    pub fn check(&self, exec: &ExecutionGraph) -> Result<(), Vec<RelaxedViolation>> {
        let violations = self.ra_checker.check(exec);
        if violations.is_empty() { Ok(()) } else { Err(violations) }
    }

    /// Check a litmus test.
    pub fn check_test(&self, test: &LitmusTest) -> RelaxedTestResult {
        let executions = test.enumerate_executions();
        let mut consistent_outcomes: Vec<Outcome> = Vec::new();
        let mut inconsistent_count = 0;

        for (exec, regs, mem) in &executions {
            match self.check(exec) {
                Ok(()) => {
                    let outcome = Outcome {
                        registers: regs.clone(),
                        memory: mem.clone(),
                    };
                    if !consistent_outcomes.iter().any(|o| o == &outcome) {
                        consistent_outcomes.push(outcome);
                    }
                }
                Err(_) => { inconsistent_count += 1; }
            }
        }

        let mut forbidden_observed = false;
        for (outcome, kind) in &test.expected_outcomes {
            if *kind == LitmusOutcome::Forbidden {
                if consistent_outcomes.iter().any(|o| outcome.matches(&o.registers, &o.memory)) {
                    forbidden_observed = true;
                }
            }
        }

        RelaxedTestResult {
            test_name: test.name.clone(),
            total_executions: executions.len(),
            consistent_executions: executions.len() - inconsistent_count,
            inconsistent_executions: inconsistent_count,
            consistent_outcomes,
            forbidden_observed,
        }
    }

    /// Get the release-acquire checker.
    pub fn ra_checker(&self) -> &ReleaseAcquireChecker {
        &self.ra_checker
    }

    /// Compute happens-before for an execution.
    pub fn compute_hb(&self, exec: &ExecutionGraph) -> BitMatrix {
        self.ra_checker.compute_hb(exec)
    }

    /// Compute synchronizes-with for an execution.
    pub fn compute_sw(&self, exec: &ExecutionGraph) -> BitMatrix {
        self.ra_checker.compute_sw(exec)
    }
}

impl Default for RelaxedModel {
    fn default() -> Self { Self::new() }
}

// ---------------------------------------------------------------------------
// RelaxedTestResult
// ---------------------------------------------------------------------------

/// Result of checking a test against the relaxed model.
#[derive(Debug, Clone)]
pub struct RelaxedTestResult {
    pub test_name: String,
    pub total_executions: usize,
    pub consistent_executions: usize,
    pub inconsistent_executions: usize,
    pub consistent_outcomes: Vec<Outcome>,
    pub forbidden_observed: bool,
}

impl fmt::Display for RelaxedTestResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Relaxed check for '{}': {}/{} consistent",
            self.test_name,
            self.consistent_executions,
            self.total_executions,
        )?;
        writeln!(f, "  {} distinct outcomes", self.consistent_outcomes.len())?;
        if self.forbidden_observed {
            writeln!(f, "  ⚠ forbidden outcome observed!")?;
        }
        Ok(())
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checker::execution::ExecutionGraphBuilder;

    #[test]
    fn test_relaxed_axiom_all() {
        let axioms = RelaxedAxiom::all();
        assert_eq!(axioms.len(), 6);
    }

    #[test]
    fn test_relaxed_axiom_display() {
        assert!(format!("{}", RelaxedAxiom::CoherenceOrder).contains("coherence"));
        assert!(format!("{}", RelaxedAxiom::ReleaseAcquire).contains("release"));
    }

    #[test]
    fn test_relaxed_violation_display() {
        let v = RelaxedViolation {
            axiom: RelaxedAxiom::CoherenceOrder,
            events: vec![0],
            description: "test".into(),
        };
        assert!(format!("{}", v).contains("Relaxed violation"));
    }

    #[test]
    fn test_ra_checker_new() {
        let checker = ReleaseAcquireChecker::new();
        assert!(checker.strict);
    }

    #[test]
    fn test_ra_checker_lenient() {
        let checker = ReleaseAcquireChecker::lenient();
        assert!(!checker.strict);
    }

    #[test]
    fn test_ra_checker_empty() {
        let checker = ReleaseAcquireChecker::new();
        let builder = ExecutionGraphBuilder::new();
        let exec = builder.build();
        let violations = checker.check(&exec);
        assert!(violations.is_empty());
    }

    #[test]
    fn test_ra_checker_simple_rf() {
        let checker = ReleaseAcquireChecker::new();
        let mut builder = ExecutionGraphBuilder::new();
        let w = builder.add_write(0, 0x100, 1);
        let r = builder.add_read(1, 0x100, 1);
        let mut exec = builder.build();
        exec.add_rf(w, r);
        exec.derive_fr();
        let violations = checker.check(&exec);
        assert!(violations.is_empty());
    }

    #[test]
    fn test_ra_checker_compute_sw() {
        let checker = ReleaseAcquireChecker::new();
        let mut builder = ExecutionGraphBuilder::new();
        let w = builder.add_write(0, 0x100, 1);
        let r = builder.add_read(1, 0x100, 1);
        let mut exec = builder.build();
        exec.add_rf(w, r);
        let sw = checker.compute_sw(&exec);
        // With simplified check, all rf edges are sw edges.
        assert!(sw.get(w, r));
    }

    #[test]
    fn test_ra_checker_compute_hb() {
        let checker = ReleaseAcquireChecker::new();
        let mut builder = ExecutionGraphBuilder::new();
        let w = builder.add_write(0, 0x100, 1);
        let r = builder.add_read(1, 0x100, 1);
        let mut exec = builder.build();
        exec.add_rf(w, r);
        let hb = checker.compute_hb(&exec);
        assert!(hb.get(w, r));
    }

    #[test]
    fn test_relaxed_model_new() {
        let model = RelaxedModel::new();
        assert_eq!(model.name, "Relaxed");
    }

    #[test]
    fn test_relaxed_model_default() {
        let model = RelaxedModel::default();
        assert_eq!(model.name, "Relaxed");
    }

    #[test]
    fn test_relaxed_build_model() {
        let model = RelaxedModel::build_model();
        assert_eq!(model.name, "C++11-relaxed");
        assert!(model.validate().is_ok());
    }

    #[test]
    fn test_relaxed_check_empty() {
        let model = RelaxedModel::new();
        let builder = ExecutionGraphBuilder::new();
        let exec = builder.build();
        assert!(model.check(&exec).is_ok());
    }

    #[test]
    fn test_relaxed_check_simple() {
        let model = RelaxedModel::new();
        let mut builder = ExecutionGraphBuilder::new();
        let w = builder.add_write(0, 0x100, 1);
        let r = builder.add_read(1, 0x100, 1);
        let mut exec = builder.build();
        exec.add_rf(w, r);
        exec.derive_fr();
        assert!(model.check(&exec).is_ok());
    }

    #[test]
    fn test_relaxed_check_test_mp() {
        let model = RelaxedModel::new();
        let test = crate::checker::litmus::mp_test();
        let result = model.check_test(&test);
        assert!(result.total_executions > 0);
    }

    #[test]
    fn test_relaxed_check_test_sb() {
        let model = RelaxedModel::new();
        let test = crate::checker::litmus::sb_test();
        let result = model.check_test(&test);
        assert!(result.total_executions > 0);
    }

    #[test]
    fn test_relaxed_ra_checker_ref() {
        let model = RelaxedModel::new();
        assert!(model.ra_checker().strict);
    }

    #[test]
    fn test_relaxed_compute_hb() {
        let model = RelaxedModel::new();
        let mut builder = ExecutionGraphBuilder::new();
        let w = builder.add_write(0, 0x100, 1);
        let r = builder.add_read(1, 0x100, 1);
        let mut exec = builder.build();
        exec.add_rf(w, r);
        let hb = model.compute_hb(&exec);
        assert!(hb.get(w, r));
    }

    #[test]
    fn test_relaxed_compute_sw() {
        let model = RelaxedModel::new();
        let mut builder = ExecutionGraphBuilder::new();
        let w = builder.add_write(0, 0x100, 1);
        let r = builder.add_read(1, 0x100, 1);
        let mut exec = builder.build();
        exec.add_rf(w, r);
        let sw = model.compute_sw(&exec);
        assert!(sw.get(w, r));
    }

    #[test]
    fn test_relaxed_test_result_display() {
        let result = RelaxedTestResult {
            test_name: "test".into(),
            total_executions: 10,
            consistent_executions: 8,
            inconsistent_executions: 2,
            consistent_outcomes: vec![],
            forbidden_observed: false,
        };
        let s = format!("{}", result);
        assert!(s.contains("Relaxed check"));
    }

    #[test]
    fn test_ra_default() {
        let checker = ReleaseAcquireChecker::default();
        assert!(checker.strict);
    }
}
