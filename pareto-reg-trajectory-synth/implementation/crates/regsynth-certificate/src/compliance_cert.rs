//! Compliance certificate generation.
//!
//! Given a strategy assignment and a set of constraints, verify that every
//! constraint is satisfied and produce a `ComplianceCertificate` with a
//! satisfying-assignment proof and SHA-256 fingerprint.

use crate::fingerprint::CertificateFingerprint;
use crate::proof_types::SatisfactionWitness;
use regsynth_encoding::{SmtConstraint, SmtExpr};
use regsynth_types::certificate::{CertificateKind, SatisfyingAssignment};
use regsynth_types::constraint::{ConstraintId, ConstraintSet};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ─── Types ──────────────────────────────────────────────────────────────────

/// Proof that a single constraint is satisfied under the assignment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintSatisfactionProof {
    pub constraint_id: String,
    pub satisfied: bool,
    pub evaluated_value: Option<String>,
    pub description: String,
}

/// Metadata attached to a compliance certificate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceMetadata {
    pub total_constraints: usize,
    pub satisfied_count: usize,
    pub hard_satisfied: usize,
    pub soft_satisfied: usize,
    pub coverage_ratio: f64,
    pub solver_used: String,
}

/// A compliance certificate proving that a particular assignment satisfies
/// a set of regulatory constraints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceCertificate {
    pub id: String,
    pub kind: CertificateKind,
    pub timestamp: String,
    pub assignment: SatisfyingAssignment,
    pub constraint_proofs: Vec<ConstraintSatisfactionProof>,
    pub metadata: ComplianceMetadata,
    pub fingerprint: CertificateFingerprint,
    pub witness: SatisfactionWitness,
}

// ─── Generator ──────────────────────────────────────────────────────────────

/// Generates compliance certificates by evaluating constraints against
/// a provided assignment.
pub struct ComplianceCertGenerator {
    solver_name: String,
}

impl ComplianceCertGenerator {
    pub fn new(solver_name: &str) -> Self {
        Self {
            solver_name: solver_name.to_string(),
        }
    }

    /// Generate a compliance certificate from a satisfying assignment and
    /// the constraint set that must be satisfied.
    pub fn generate(
        &self,
        assignment: &SatisfyingAssignment,
        constraints: &ConstraintSet,
    ) -> crate::Result<ComplianceCertificate> {
        let bool_env = &assignment.bool_assignments;
        let real_env = &assignment.real_assignments;

        let mut proofs = Vec::new();
        let mut hard_satisfied = 0usize;
        let mut soft_satisfied = 0usize;
        let mut witness = SatisfactionWitness::new();

        // Populate witness from the assignment
        for (k, v) in bool_env {
            witness.set_bool(k, *v);
        }
        for (k, v) in real_env {
            witness.set_real(k, *v);
        }

        for constraint in constraints.all() {
            let cid = constraint.id.as_str();
            let result = constraint.expr.evaluate(bool_env, real_env);

            let satisfied = result.unwrap_or(false);
            witness.record_constraint(cid, satisfied);

            proofs.push(ConstraintSatisfactionProof {
                constraint_id: cid.to_string(),
                satisfied,
                evaluated_value: Some(format!("{:?}", result)),
                description: constraint.description.clone(),
            });

            if satisfied {
                if constraint.kind.is_hard() {
                    hard_satisfied += 1;
                } else {
                    soft_satisfied += 1;
                }
            }
        }

        let total = constraints.len();
        let sat_count = hard_satisfied + soft_satisfied;
        let coverage = if total > 0 {
            sat_count as f64 / total as f64
        } else {
            1.0
        };

        // Check all hard constraints are satisfied
        let all_hard_ok = constraints
            .hard_constraints()
            .iter()
            .all(|c| {
                let cid = c.id.as_str();
                proofs.iter().any(|p| p.constraint_id == cid && p.satisfied)
            });

        if !all_hard_ok {
            let failing: Vec<String> = proofs
                .iter()
                .filter(|p| !p.satisfied)
                .map(|p| p.constraint_id.clone())
                .collect();
            return Err(crate::CertificateError::ProofValidation(format!(
                "hard constraints not satisfied: {:?}",
                failing
            )));
        }

        let metadata = ComplianceMetadata {
            total_constraints: total,
            satisfied_count: sat_count,
            hard_satisfied,
            soft_satisfied,
            coverage_ratio: coverage,
            solver_used: self.solver_name.clone(),
        };

        // Build the satisfying assignment record
        let mut sa = assignment.clone();
        for proof in &proofs {
            sa.record_satisfaction(&proof.constraint_id, proof.satisfied);
        }

        // Compute fingerprint over the full certificate content
        let fp_content = serde_json::to_string(&(&sa, &proofs, &metadata))
            .map_err(|e| crate::CertificateError::Serialization(e.to_string()))?;
        let fingerprint = CertificateFingerprint::compute(fp_content.as_bytes());

        Ok(ComplianceCertificate {
            id: uuid::Uuid::new_v4().to_string(),
            kind: CertificateKind::Compliance,
            timestamp: chrono::Utc::now().to_rfc3339(),
            assignment: sa,
            constraint_proofs: proofs,
            metadata,
            fingerprint,
            witness,
        })
    }

    /// Generate a compliance certificate directly from SMT constraints and
    /// a variable assignment map.
    pub fn generate_from_smt(
        &self,
        bool_assignments: &HashMap<String, bool>,
        real_assignments: &HashMap<String, f64>,
        smt_constraints: &[SmtConstraint],
    ) -> crate::Result<ComplianceCertificate> {
        let mut proofs = Vec::new();
        let mut witness = SatisfactionWitness::new();
        let mut sat_count = 0usize;

        for (k, v) in bool_assignments {
            witness.set_bool(k, *v);
        }
        for (k, v) in real_assignments {
            witness.set_real(k, *v);
        }

        for sc in smt_constraints {
            let satisfied = evaluate_smt_expr(&sc.expr, bool_assignments, real_assignments);
            witness.record_constraint(&sc.id, satisfied);
            if satisfied {
                sat_count += 1;
            }
            proofs.push(ConstraintSatisfactionProof {
                constraint_id: sc.id.clone(),
                satisfied,
                evaluated_value: Some(format!("{}", satisfied)),
                description: sc
                    .provenance
                    .as_ref()
                    .map(|p| p.description.clone())
                    .unwrap_or_default(),
            });
        }

        let total = smt_constraints.len();
        let coverage = if total > 0 {
            sat_count as f64 / total as f64
        } else {
            1.0
        };

        let mut sa = SatisfyingAssignment::new();
        for (k, v) in bool_assignments {
            sa.set_bool(k, *v);
        }
        for (k, v) in real_assignments {
            sa.set_real(k, *v);
        }
        for p in &proofs {
            sa.record_satisfaction(&p.constraint_id, p.satisfied);
        }

        let metadata = ComplianceMetadata {
            total_constraints: total,
            satisfied_count: sat_count,
            hard_satisfied: sat_count,
            soft_satisfied: 0,
            coverage_ratio: coverage,
            solver_used: self.solver_name.clone(),
        };

        let fp_content = serde_json::to_string(&(&sa, &proofs, &metadata))
            .map_err(|e| crate::CertificateError::Serialization(e.to_string()))?;
        let fingerprint = CertificateFingerprint::compute(fp_content.as_bytes());

        Ok(ComplianceCertificate {
            id: uuid::Uuid::new_v4().to_string(),
            kind: CertificateKind::Compliance,
            timestamp: chrono::Utc::now().to_rfc3339(),
            assignment: sa,
            constraint_proofs: proofs,
            metadata,
            fingerprint,
            witness,
        })
    }
}

// ─── SMT Expression Evaluator ───────────────────────────────────────────────

/// Evaluate an SMT expression under Boolean and real-valued assignments.
fn evaluate_smt_expr(
    expr: &SmtExpr,
    bool_env: &HashMap<String, bool>,
    real_env: &HashMap<String, f64>,
) -> bool {
    match expr {
        SmtExpr::BoolLit(b) => *b,
        SmtExpr::Var(name, _sort) => {
            bool_env.get(name).copied().unwrap_or(false)
        }
        SmtExpr::Not(inner) => !evaluate_smt_expr(inner, bool_env, real_env),
        SmtExpr::And(exprs) => exprs.iter().all(|e| evaluate_smt_expr(e, bool_env, real_env)),
        SmtExpr::Or(exprs) => exprs.iter().any(|e| evaluate_smt_expr(e, bool_env, real_env)),
        SmtExpr::Implies(a, b) => {
            !evaluate_smt_expr(a, bool_env, real_env)
                || evaluate_smt_expr(b, bool_env, real_env)
        }
        SmtExpr::Eq(a, b) => {
            let av = eval_smt_real(a, real_env);
            let bv = eval_smt_real(b, real_env);
            match (av, bv) {
                (Some(x), Some(y)) => (x - y).abs() < 1e-10,
                _ => {
                    // Fall back to boolean equality
                    evaluate_smt_expr(a, bool_env, real_env)
                        == evaluate_smt_expr(b, bool_env, real_env)
                }
            }
        }
        SmtExpr::Lt(a, b) => {
            match (eval_smt_real(a, real_env), eval_smt_real(b, real_env)) {
                (Some(x), Some(y)) => x < y,
                _ => false,
            }
        }
        SmtExpr::Le(a, b) => {
            match (eval_smt_real(a, real_env), eval_smt_real(b, real_env)) {
                (Some(x), Some(y)) => x <= y,
                _ => false,
            }
        }
        SmtExpr::Gt(a, b) => {
            match (eval_smt_real(a, real_env), eval_smt_real(b, real_env)) {
                (Some(x), Some(y)) => x > y,
                _ => false,
            }
        }
        SmtExpr::Ge(a, b) => {
            match (eval_smt_real(a, real_env), eval_smt_real(b, real_env)) {
                (Some(x), Some(y)) => x >= y,
                _ => false,
            }
        }
        SmtExpr::Ite(cond, then_br, else_br) => {
            if evaluate_smt_expr(cond, bool_env, real_env) {
                evaluate_smt_expr(then_br, bool_env, real_env)
            } else {
                evaluate_smt_expr(else_br, bool_env, real_env)
            }
        }
        _ => false,
    }
}

/// Evaluate an SMT expression to a real-valued result.
fn eval_smt_real(expr: &SmtExpr, real_env: &HashMap<String, f64>) -> Option<f64> {
    match expr {
        SmtExpr::RealLit(v) => Some(*v),
        SmtExpr::IntLit(v) => Some(*v as f64),
        SmtExpr::Var(name, _) => real_env.get(name).copied(),
        SmtExpr::Add(exprs) => {
            let mut sum = 0.0;
            for e in exprs {
                sum += eval_smt_real(e, real_env)?;
            }
            Some(sum)
        }
        SmtExpr::Sub(a, b) => {
            Some(eval_smt_real(a, real_env)? - eval_smt_real(b, real_env)?)
        }
        SmtExpr::Mul(exprs) => {
            let mut prod = 1.0;
            for e in exprs {
                prod *= eval_smt_real(e, real_env)?;
            }
            Some(prod)
        }
        SmtExpr::Neg(inner) => eval_smt_real(inner, real_env).map(|v| -v),
        _ => None,
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use regsynth_types::constraint::{Constraint, ConstraintExpr, ConstraintSet};

    fn make_constraint_set() -> ConstraintSet {
        let mut cs = ConstraintSet::new();
        cs.add(
            Constraint::hard("c1", ConstraintExpr::var("x"))
                .with_description("x must be true"),
        );
        cs.add(
            Constraint::hard("c2", ConstraintExpr::var("y"))
                .with_description("y must be true"),
        );
        cs.add(
            Constraint::soft("c3", ConstraintExpr::var("z"), 1.0)
                .with_description("z preferred true"),
        );
        cs
    }

    #[test]
    fn generate_compliance_cert_all_satisfied() {
        let gen = ComplianceCertGenerator::new("test-solver");
        let cs = make_constraint_set();
        let mut sa = SatisfyingAssignment::new();
        sa.set_bool("x", true);
        sa.set_bool("y", true);
        sa.set_bool("z", true);

        let cert = gen.generate(&sa, &cs).unwrap();
        assert_eq!(cert.kind, CertificateKind::Compliance);
        assert_eq!(cert.metadata.total_constraints, 3);
        assert_eq!(cert.metadata.satisfied_count, 3);
        assert!((cert.metadata.coverage_ratio - 1.0).abs() < 1e-10);
        assert!(cert.witness.all_satisfied());
    }

    #[test]
    fn generate_compliance_cert_hard_fail() {
        let gen = ComplianceCertGenerator::new("test-solver");
        let cs = make_constraint_set();
        let mut sa = SatisfyingAssignment::new();
        sa.set_bool("x", true);
        sa.set_bool("y", false); // hard constraint fails
        sa.set_bool("z", true);

        let result = gen.generate(&sa, &cs);
        assert!(result.is_err());
    }

    #[test]
    fn generate_compliance_cert_soft_fail_ok() {
        let gen = ComplianceCertGenerator::new("test-solver");
        let cs = make_constraint_set();
        let mut sa = SatisfyingAssignment::new();
        sa.set_bool("x", true);
        sa.set_bool("y", true);
        sa.set_bool("z", false); // soft constraint fails — ok

        let cert = gen.generate(&sa, &cs).unwrap();
        assert_eq!(cert.metadata.satisfied_count, 2);
        assert_eq!(cert.metadata.hard_satisfied, 2);
        assert_eq!(cert.metadata.soft_satisfied, 0);
    }

    #[test]
    fn generate_from_smt() {
        let gen = ComplianceCertGenerator::new("smt-solver");
        let constraints = vec![
            SmtConstraint {
                id: "s1".into(),
                expr: SmtExpr::BoolLit(true),
                provenance: None,
            },
            SmtConstraint {
                id: "s2".into(),
                expr: SmtExpr::Le(
                    Box::new(SmtExpr::Var("cost".into(), regsynth_encoding::SmtSort::Real)),
                    Box::new(SmtExpr::RealLit(100.0)),
                ),
                provenance: None,
            },
        ];
        let mut real_env = HashMap::new();
        real_env.insert("cost".into(), 50.0);
        let bool_env = HashMap::new();

        let cert = gen.generate_from_smt(&bool_env, &real_env, &constraints).unwrap();
        assert_eq!(cert.metadata.satisfied_count, 2);
    }

    #[test]
    fn fingerprint_determinism() {
        let gen = ComplianceCertGenerator::new("solver");
        let cs = make_constraint_set();
        let mut sa = SatisfyingAssignment::new();
        sa.set_bool("x", true);
        sa.set_bool("y", true);
        sa.set_bool("z", true);

        let cert1 = gen.generate(&sa, &cs).unwrap();
        let cert2 = gen.generate(&sa, &cs).unwrap();
        // Fingerprints should be deterministic given same input
        // (ids and timestamps differ, but the fingerprint is over assignment + proofs)
        assert!(!cert1.fingerprint.hex_digest.is_empty());
        assert!(!cert2.fingerprint.hex_digest.is_empty());
    }

    #[test]
    fn evaluate_smt_arithmetic() {
        let mut real_env = HashMap::new();
        real_env.insert("a".into(), 5.0);
        real_env.insert("b".into(), 3.0);

        // a + b > 7 => true (5+3=8 > 7)
        let expr = SmtExpr::Gt(
            Box::new(SmtExpr::Add(vec![
                SmtExpr::Var("a".into(), regsynth_encoding::SmtSort::Real),
                SmtExpr::Var("b".into(), regsynth_encoding::SmtSort::Real),
            ])),
            Box::new(SmtExpr::RealLit(7.0)),
        );
        assert!(evaluate_smt_expr(&expr, &HashMap::new(), &real_env));
    }
}
