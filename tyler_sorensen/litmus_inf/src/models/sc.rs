//! Sequential Consistency (SC) memory model for LITMUS∞.
//!
//! SC requires all memory accesses to appear in a single total order
//! consistent with each thread's program order. This module provides
//! SC axiom definitions, violation detection, and test utilities.

use std::collections::HashMap;
use std::fmt;
use serde::{Serialize, Deserialize};

use crate::checker::{
    LitmusTest, Outcome, LitmusOutcome,
    MemoryModel, RelationExpr, BuiltinModel,
    ExecutionGraph, BitMatrix,
};
use crate::checker::execution::{EventId, OpType};

// ---------------------------------------------------------------------------
// ScAxiom
// ---------------------------------------------------------------------------

/// Axioms of Sequential Consistency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ScAxiom {
    /// All operations (po ∪ com) must be acyclic.
    Acyclicity,
    /// Reads-from is well-formed: each read reads from at most one write.
    RfWellFormed,
    /// Coherence order is a total order per location.
    CoTotal,
    /// From-reads consistency: fr = rf⁻¹ ; co.
    FrConsistency,
    /// Atomicity of RMW operations.
    RmwAtomicity,
}

impl fmt::Display for ScAxiom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Acyclicity => write!(f, "acyclicity(po ∪ com)"),
            Self::RfWellFormed => write!(f, "rf-well-formed"),
            Self::CoTotal => write!(f, "co-total-per-loc"),
            Self::FrConsistency => write!(f, "fr = rf⁻¹;co"),
            Self::RmwAtomicity => write!(f, "rmw-atomicity"),
        }
    }
}

impl ScAxiom {
    /// All SC axioms.
    pub fn all() -> Vec<Self> {
        vec![
            Self::Acyclicity,
            Self::RfWellFormed,
            Self::CoTotal,
            Self::FrConsistency,
            Self::RmwAtomicity,
        ]
    }
}

// ---------------------------------------------------------------------------
// ScViolation
// ---------------------------------------------------------------------------

/// A violation of an SC axiom.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScViolation {
    /// Which axiom was violated.
    pub axiom: ScAxiom,
    /// Events involved in the violation.
    pub events: Vec<EventId>,
    /// Description of the violation.
    pub description: String,
}

impl fmt::Display for ScViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SC violation [{}]: {} [events: {:?}]",
            self.axiom, self.description, self.events)
    }
}

// ---------------------------------------------------------------------------
// ScModel
// ---------------------------------------------------------------------------

/// Sequential Consistency model checker.
#[derive(Debug, Clone)]
pub struct ScModel {
    pub name: String,
    model: MemoryModel,
}

impl ScModel {
    /// Create a new SC model.
    pub fn new() -> Self {
        Self {
            name: "SC".to_string(),
            model: BuiltinModel::SC.build(),
        }
    }

    /// Get the underlying memory model.
    pub fn model(&self) -> &MemoryModel {
        &self.model
    }

    /// Build the SC memory model from scratch.
    pub fn build_model() -> MemoryModel {
        let mut m = MemoryModel::new("SC");
        m.add_derived("com",
            RelationExpr::union_many(vec![
                RelationExpr::base("rf"),
                RelationExpr::base("co"),
                RelationExpr::base("fr"),
            ]),
            "communication = rf ∪ co ∪ fr",
        );
        m.add_acyclic(
            RelationExpr::union(RelationExpr::base("po"), RelationExpr::base("com")),
        );
        m
    }

    /// Check whether an execution is SC-consistent.
    pub fn check(&self, exec: &ExecutionGraph) -> Result<(), Vec<ScViolation>> {
        let mut violations = Vec::new();
        violations.extend(self.check_acyclicity(exec));
        violations.extend(self.check_rf_well_formed(exec));
        violations.extend(self.check_co_total(exec));
        violations.extend(self.check_fr_consistency(exec));
        violations.extend(self.check_rmw_atomicity(exec));
        if violations.is_empty() { Ok(()) } else { Err(violations) }
    }

    /// Check acyclicity of po ∪ com.
    pub fn check_acyclicity(&self, exec: &ExecutionGraph) -> Vec<ScViolation> {
        let com = exec.rf.union(&exec.co).union(&exec.fr);
        let po_com = exec.po.union(&com);
        let tc = po_com.transitive_closure();
        let n = exec.len();
        let mut violations = Vec::new();
        for i in 0..n {
            if tc.get(i, i) {
                violations.push(ScViolation {
                    axiom: ScAxiom::Acyclicity,
                    events: vec![i],
                    description: format!("cycle found involving event e{}", i),
                });
                break;
            }
        }
        violations
    }

    /// Check that rf is well-formed (each read has at most one writer).
    pub fn check_rf_well_formed(&self, exec: &ExecutionGraph) -> Vec<ScViolation> {
        let mut violations = Vec::new();
        let n = exec.len();
        for r in 0..n {
            if !exec.events[r].is_read() { continue; }
            let writers: Vec<usize> = (0..n)
                .filter(|&w| exec.rf.get(w, r))
                .collect();
            if writers.len() > 1 {
                violations.push(ScViolation {
                    axiom: ScAxiom::RfWellFormed,
                    events: writers,
                    description: format!("read e{} has multiple rf sources", r),
                });
            }
        }
        violations
    }

    /// Check that co is a total order per location.
    pub fn check_co_total(&self, exec: &ExecutionGraph) -> Vec<ScViolation> {
        let mut violations = Vec::new();
        for addr in exec.addresses() {
            let writes: Vec<EventId> = exec.addr_events(addr).iter()
                .filter(|&&eid| exec.events[eid].is_write() && !exec.events[eid].is_read())
                .copied()
                .collect();
            for i in 0..writes.len() {
                for j in i + 1..writes.len() {
                    let w1 = writes[i];
                    let w2 = writes[j];
                    if !exec.co.get(w1, w2) && !exec.co.get(w2, w1) {
                        violations.push(ScViolation {
                            axiom: ScAxiom::CoTotal,
                            events: vec![w1, w2],
                            description: format!(
                                "writes e{} and e{} to {:#x} not co-ordered",
                                w1, w2, addr
                            ),
                        });
                    }
                }
            }
        }
        violations
    }

    /// Check from-reads consistency.
    pub fn check_fr_consistency(&self, exec: &ExecutionGraph) -> Vec<ScViolation> {
        let expected_fr = exec.rf.inverse().compose(&exec.co);
        let mut violations = Vec::new();
        let n = exec.len();
        for i in 0..n {
            for j in 0..n {
                if exec.fr.get(i, j) != expected_fr.get(i, j) {
                    violations.push(ScViolation {
                        axiom: ScAxiom::FrConsistency,
                        events: vec![i, j],
                        description: format!(
                            "fr({},{}) = {} but rf⁻¹;co({},{}) = {}",
                            i, j, exec.fr.get(i, j),
                            i, j, expected_fr.get(i, j),
                        ),
                    });
                    return violations;
                }
            }
        }
        violations
    }

    /// Check RMW atomicity.
    pub fn check_rmw_atomicity(&self, exec: &ExecutionGraph) -> Vec<ScViolation> {
        let mut violations = Vec::new();
        let n = exec.len();
        for i in 0..n {
            if !exec.events[i].is_rmw() { continue; }
            let rf_sources: Vec<EventId> = (0..n)
                .filter(|&w| exec.rf.get(w, i))
                .collect();
            for &src in &rf_sources {
                for w in 0..n {
                    if w == src || w == i { continue; }
                    if exec.events[w].address != exec.events[i].address { continue; }
                    if !exec.events[w].is_write() { continue; }
                    if exec.co.get(src, w) && exec.co.get(w, i) {
                        violations.push(ScViolation {
                            axiom: ScAxiom::RmwAtomicity,
                            events: vec![src, w, i],
                            description: format!(
                                "write e{} intervenes between rf-source e{} and RMW e{}",
                                w, src, i
                            ),
                        });
                    }
                }
            }
        }
        violations
    }

    /// Check a litmus test against SC model.
    pub fn check_test(&self, test: &LitmusTest) -> ScTestResult {
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
                Err(_) => {
                    inconsistent_count += 1;
                }
            }
        }

        let mut forbidden_observed = false;
        for (outcome, kind) in &test.expected_outcomes {
            if *kind == LitmusOutcome::Forbidden {
                if consistent_outcomes.iter().any(|o| {
                    outcome.matches(&o.registers, &o.memory)
                }) {
                    forbidden_observed = true;
                }
            }
        }

        ScTestResult {
            test_name: test.name.clone(),
            total_executions: executions.len(),
            consistent_executions: executions.len() - inconsistent_count,
            inconsistent_executions: inconsistent_count,
            consistent_outcomes,
            forbidden_observed,
        }
    }

    /// Check whether a specific outcome is SC-allowed.
    pub fn is_allowed(&self, test: &LitmusTest, outcome: &Outcome) -> bool {
        let executions = test.enumerate_executions();
        for (exec, regs, mem) in &executions {
            if outcome.matches(&regs, &mem) && self.check(&exec).is_ok() {
                return true;
            }
        }
        false
    }

    /// Enumerate all SC-consistent outcomes.
    pub fn allowed_outcomes(&self, test: &LitmusTest) -> Vec<Outcome> {
        self.check_test(test).consistent_outcomes
    }
}

impl Default for ScModel {
    fn default() -> Self { Self::new() }
}

// ---------------------------------------------------------------------------
// ScTestResult
// ---------------------------------------------------------------------------

/// Result of checking a test against SC.
#[derive(Debug, Clone)]
pub struct ScTestResult {
    pub test_name: String,
    pub total_executions: usize,
    pub consistent_executions: usize,
    pub inconsistent_executions: usize,
    pub consistent_outcomes: Vec<Outcome>,
    pub forbidden_observed: bool,
}

impl fmt::Display for ScTestResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "SC check for '{}': {}/{} consistent",
            self.test_name,
            self.consistent_executions,
            self.total_executions,
        )?;
        writeln!(f, "  {} distinct outcomes", self.consistent_outcomes.len())?;
        for (i, o) in self.consistent_outcomes.iter().enumerate() {
            writeln!(f, "    [{}] {}", i, o)?;
        }
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
    fn test_sc_axiom_all() {
        let axioms = ScAxiom::all();
        assert_eq!(axioms.len(), 5);
    }

    #[test]
    fn test_sc_axiom_display() {
        assert!(format!("{}", ScAxiom::Acyclicity).contains("acyclicity"));
        assert!(format!("{}", ScAxiom::RfWellFormed).contains("rf"));
    }

    #[test]
    fn test_sc_model_creation() {
        let model = ScModel::new();
        assert_eq!(model.name, "SC");
    }

    #[test]
    fn test_sc_build_model() {
        let model = ScModel::build_model();
        assert_eq!(model.name, "SC");
        assert!(!model.constraints.is_empty());
    }

    #[test]
    fn test_sc_model_validate() {
        let model = ScModel::build_model();
        assert!(model.validate().is_ok());
    }

    #[test]
    fn test_sc_violation_display() {
        let v = ScViolation {
            axiom: ScAxiom::Acyclicity,
            events: vec![0, 1],
            description: "test".into(),
        };
        let s = format!("{}", v);
        assert!(s.contains("SC violation"));
    }

    #[test]
    fn test_sc_check_empty_execution() {
        let model = ScModel::new();
        let builder = ExecutionGraphBuilder::new();
        let exec = builder.build();
        assert!(model.check(&exec).is_ok());
    }

    #[test]
    fn test_sc_check_single_write() {
        let model = ScModel::new();
        let mut builder = ExecutionGraphBuilder::new();
        builder.add_write(0, 0x100, 1);
        let exec = builder.build();
        assert!(model.check(&exec).is_ok());
    }

    #[test]
    fn test_sc_check_simple_rf() {
        let model = ScModel::new();
        let mut builder = ExecutionGraphBuilder::new();
        let w = builder.add_write(0, 0x100, 1);
        let r = builder.add_read(1, 0x100, 1);
        let mut exec = builder.build();
        exec.add_rf(w, r);
        exec.derive_fr();
        assert!(model.check(&exec).is_ok());
    }

    #[test]
    fn test_sc_check_test_mp() {
        let model = ScModel::new();
        let test = crate::checker::litmus::mp_test();
        let result = model.check_test(&test);
        assert!(result.total_executions > 0);
        assert!(!result.forbidden_observed);
    }

    #[test]
    fn test_sc_check_test_sb() {
        let model = ScModel::new();
        let test = crate::checker::litmus::sb_test();
        let result = model.check_test(&test);
        assert!(result.total_executions > 0);
    }

    #[test]
    fn test_sc_allowed_outcomes() {
        let model = ScModel::new();
        let test = crate::checker::litmus::mp_test();
        let _outcomes = model.allowed_outcomes(&test);
    }

    #[test]
    fn test_sc_test_result_display() {
        let result = ScTestResult {
            test_name: "test".into(),
            total_executions: 10,
            consistent_executions: 8,
            inconsistent_executions: 2,
            consistent_outcomes: vec![],
            forbidden_observed: false,
        };
        let s = format!("{}", result);
        assert!(s.contains("SC check"));
    }

    #[test]
    fn test_sc_rf_well_formed() {
        let model = ScModel::new();
        let mut builder = ExecutionGraphBuilder::new();
        let w = builder.add_write(0, 0x100, 1);
        let r = builder.add_read(1, 0x100, 1);
        let mut exec = builder.build();
        exec.add_rf(w, r);
        let violations = model.check_rf_well_formed(&exec);
        assert!(violations.is_empty());
    }

    #[test]
    fn test_sc_default() {
        let model = ScModel::default();
        assert_eq!(model.name, "SC");
    }

    #[test]
    fn test_sc_model_ref() {
        let model = ScModel::new();
        assert_eq!(model.model().name, "SC");
    }
}
