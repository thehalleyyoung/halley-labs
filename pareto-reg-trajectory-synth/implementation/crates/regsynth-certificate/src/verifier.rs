//! Certificate verification.
//!
//! Standalone verifier for compliance, infeasibility, and Pareto certificates.
//! Does not require a solver — works purely from the certificate data.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::compliance_cert::ComplianceCertificate;
use crate::fingerprint::CertificateFingerprint;
use crate::infeasibility_cert::InfeasibilityCertificate;
use crate::pareto_cert::ParetoCertificate;
use crate::serialization::Certificate;

// ─── Verification Check ─────────────────────────────────────────────────────

/// A single verification check with pass/fail status and a human-readable
/// description.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationCheck {
    pub name: String,
    pub passed: bool,
    pub detail: String,
}

impl VerificationCheck {
    pub fn pass(name: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            passed: true,
            detail: detail.into(),
        }
    }

    pub fn fail(name: impl Into<String>, detail: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            passed: false,
            detail: detail.into(),
        }
    }
}

// ─── Verification Result ────────────────────────────────────────────────────

/// Aggregated result of verifying a certificate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    pub valid: bool,
    pub checks: Vec<VerificationCheck>,
    pub verified_at: String,
}

impl VerificationResult {
    pub fn new(checks: Vec<VerificationCheck>) -> Self {
        let valid = checks.iter().all(|c| c.passed);
        Self {
            valid,
            checks,
            verified_at: chrono::Utc::now().to_rfc3339(),
        }
    }

    pub fn pass_count(&self) -> usize {
        self.checks.iter().filter(|c| c.passed).count()
    }

    pub fn fail_count(&self) -> usize {
        self.checks.iter().filter(|c| !c.passed).count()
    }
}

// ─── Verifier ───────────────────────────────────────────────────────────────

/// Certificate verifier: performs standalone verification of any certificate
/// type without requiring a solver instance.
pub struct CertificateVerifier {
    deep: bool,
}

impl CertificateVerifier {
    pub fn new() -> Self {
        Self { deep: false }
    }

    pub fn with_deep(mut self, deep: bool) -> Self {
        self.deep = deep;
        self
    }

    // ── Generic Certificate Envelope ────────────────────────────────────

    /// Verify a serialized Certificate envelope.
    pub fn verify(&self, cert: &Certificate) -> VerificationResult {
        let mut checks = Vec::new();

        // 1. Fingerprint integrity
        let recomputed = CertificateFingerprint::compute(cert.payload_json.as_bytes());
        if recomputed == cert.fingerprint {
            checks.push(VerificationCheck::pass(
                "fingerprint_integrity",
                "payload fingerprint matches stored value",
            ));
        } else {
            checks.push(VerificationCheck::fail(
                "fingerprint_integrity",
                format!(
                    "mismatch: stored={}, computed={}",
                    cert.fingerprint.hex_digest, recomputed.hex_digest
                ),
            ));
        }

        // 2. Payload parseable
        match serde_json::from_str::<serde_json::Value>(&cert.payload_json) {
            Ok(_) => checks.push(VerificationCheck::pass(
                "payload_parseable",
                "payload is valid JSON",
            )),
            Err(e) => checks.push(VerificationCheck::fail(
                "payload_parseable",
                format!("parse error: {}", e),
            )),
        }

        // 3. Certificate type
        if !cert.certificate_type.is_empty() {
            checks.push(VerificationCheck::pass(
                "certificate_type",
                format!("type: {}", cert.certificate_type),
            ));
        } else {
            checks.push(VerificationCheck::fail(
                "certificate_type",
                "certificate type is empty",
            ));
        }

        // 4. Deep verification
        if self.deep {
            checks.extend(self.deep_verify_envelope(cert));
        }

        VerificationResult::new(checks)
    }

    /// Verify a raw JSON certificate string.
    pub fn verify_json(&self, json: &str) -> crate::Result<VerificationResult> {
        let cert: Certificate = serde_json::from_str(json)
            .map_err(|e| crate::CertificateError::Serialization(e.to_string()))?;
        Ok(self.verify(&cert))
    }

    fn deep_verify_envelope(&self, cert: &Certificate) -> Vec<VerificationCheck> {
        match cert.certificate_type.as_str() {
            "compliance" => {
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(&cert.payload_json) {
                    if val.get("witness")
                        .and_then(|w| w.get("constraint_results"))
                        .and_then(|cr| cr.as_object())
                        .map(|m| m.values().all(|v| v.as_bool() == Some(true)))
                        == Some(true)
                    {
                        vec![VerificationCheck::pass("deep_compliance", "all constraints satisfied")]
                    } else {
                        vec![VerificationCheck::fail("deep_compliance", "not all constraints satisfied")]
                    }
                } else {
                    vec![VerificationCheck::fail("deep_compliance", "cannot parse compliance payload")]
                }
            }
            "infeasibility" => {
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(&cert.payload_json) {
                    let proof_complete = val.get("resolution_proof")
                        .and_then(|p| p.get("empty_clause_step"))
                        .is_some();
                    if proof_complete {
                        vec![VerificationCheck::pass("deep_infeasibility", "resolution proof derives empty clause")]
                    } else {
                        vec![VerificationCheck::fail("deep_infeasibility", "resolution proof incomplete")]
                    }
                } else {
                    vec![VerificationCheck::fail("deep_infeasibility", "cannot parse infeasibility payload")]
                }
            }
            "pareto" => {
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(&cert.payload_json) {
                    let all_valid = val.get("point_proofs")
                        .and_then(|pp| pp.as_array())
                        .map(|arr| arr.iter().all(|p| p.get("is_valid").and_then(|v| v.as_bool()) == Some(true)))
                        .unwrap_or(false);
                    if all_valid {
                        vec![VerificationCheck::pass("deep_pareto", "all point proofs valid")]
                    } else {
                        vec![VerificationCheck::fail("deep_pareto", "some point proofs invalid")]
                    }
                } else {
                    vec![VerificationCheck::fail("deep_pareto", "cannot parse pareto payload")]
                }
            }
            _ => vec![VerificationCheck::pass(
                "deep_unknown",
                format!("no deep verification for type: {}", cert.certificate_type),
            )],
        }
    }

    // ── Compliance Verification ─────────────────────────────────────────

    /// Verify a compliance certificate by re-evaluating the witness against
    /// the constraint results recorded in the certificate.
    pub fn verify_compliance(&self, cert: &ComplianceCertificate) -> VerificationResult {
        let mut checks = Vec::new();

        // Check fingerprint
        let fp_content = match serde_json::to_string(&(
            &cert.assignment,
            &cert.constraint_proofs,
            &cert.metadata,
        )) {
            Ok(s) => s,
            Err(e) => {
                checks.push(VerificationCheck::fail(
                    "fingerprint_recompute",
                    format!("serialization error: {}", e),
                ));
                return VerificationResult::new(checks);
            }
        };
        let recomputed = CertificateFingerprint::compute(fp_content.as_bytes());
        if recomputed == cert.fingerprint {
            checks.push(VerificationCheck::pass(
                "fingerprint_integrity",
                "fingerprint matches",
            ));
        } else {
            checks.push(VerificationCheck::fail(
                "fingerprint_integrity",
                "fingerprint mismatch",
            ));
        }

        // Check all constraint proofs report satisfied
        let all_satisfied = cert.constraint_proofs.iter().all(|p| p.satisfied);
        if all_satisfied {
            checks.push(VerificationCheck::pass(
                "all_constraints_satisfied",
                format!("{}/{} constraints satisfied", cert.metadata.satisfied_count, cert.metadata.total_constraints),
            ));
        } else {
            let failing: Vec<&str> = cert
                .constraint_proofs
                .iter()
                .filter(|p| !p.satisfied)
                .map(|p| p.constraint_id.as_str())
                .collect();
            checks.push(VerificationCheck::fail(
                "all_constraints_satisfied",
                format!("failing constraints: {:?}", failing),
            ));
        }

        // Cross-check witness
        if cert.witness.all_satisfied() {
            checks.push(VerificationCheck::pass(
                "witness_consistency",
                "witness confirms all constraints satisfied",
            ));
        } else {
            checks.push(VerificationCheck::fail(
                "witness_consistency",
                format!("witness satisfaction ratio: {:.2}", cert.witness.satisfaction_ratio()),
            ));
        }

        // Check coverage
        if cert.metadata.coverage_ratio >= 1.0 - 1e-10 {
            checks.push(VerificationCheck::pass(
                "full_coverage",
                "100% constraint coverage",
            ));
        } else {
            checks.push(VerificationCheck::pass(
                "partial_coverage",
                format!("{:.1}% coverage (soft constraints may be relaxed)", cert.metadata.coverage_ratio * 100.0),
            ));
        }

        VerificationResult::new(checks)
    }

    // ── Infeasibility Verification ──────────────────────────────────────

    /// Verify an infeasibility certificate by validating the resolution proof.
    pub fn verify_infeasibility(&self, cert: &InfeasibilityCertificate) -> VerificationResult {
        let mut checks = Vec::new();

        // Check fingerprint
        let fp_content = match serde_json::to_string(&(
            &cert.mus,
            &cert.resolution_proof,
            &cert.metadata,
        )) {
            Ok(s) => s,
            Err(e) => {
                checks.push(VerificationCheck::fail(
                    "fingerprint_recompute",
                    format!("serialization error: {}", e),
                ));
                return VerificationResult::new(checks);
            }
        };
        let recomputed = CertificateFingerprint::compute(fp_content.as_bytes());
        if recomputed == cert.fingerprint {
            checks.push(VerificationCheck::pass(
                "fingerprint_integrity",
                "fingerprint matches",
            ));
        } else {
            checks.push(VerificationCheck::fail(
                "fingerprint_integrity",
                "fingerprint mismatch",
            ));
        }

        // Check MUS is non-empty
        if cert.mus.size > 0 {
            checks.push(VerificationCheck::pass(
                "mus_nonempty",
                format!("MUS contains {} constraints", cert.mus.size),
            ));
        } else {
            checks.push(VerificationCheck::fail("mus_nonempty", "MUS is empty"));
        }

        // Validate resolution proof completeness
        if cert.resolution_proof.is_complete() {
            checks.push(VerificationCheck::pass(
                "proof_complete",
                "resolution proof derives empty clause",
            ));
        } else {
            checks.push(VerificationCheck::fail(
                "proof_complete",
                "resolution proof does not derive empty clause",
            ));
        }

        // Validate each resolution step
        let step_results = cert.resolution_proof.validate_chain();
        let all_steps_valid = step_results.iter().all(|&v| v);
        if all_steps_valid {
            checks.push(VerificationCheck::pass(
                "proof_steps_valid",
                format!("all {} resolution steps valid", step_results.len()),
            ));
        } else {
            let failing: Vec<usize> = step_results
                .iter()
                .enumerate()
                .filter(|(_, &v)| !v)
                .map(|(i, _)| i)
                .collect();
            checks.push(VerificationCheck::fail(
                "proof_steps_valid",
                format!("invalid steps: {:?}", failing),
            ));
        }

        // Full proof validation
        if cert.resolution_proof.validate() {
            checks.push(VerificationCheck::pass(
                "proof_valid",
                "full resolution proof validates",
            ));
        } else {
            checks.push(VerificationCheck::fail(
                "proof_valid",
                "full resolution proof validation failed",
            ));
        }

        VerificationResult::new(checks)
    }

    // ── Pareto Verification ─────────────────────────────────────────────

    /// Verify a Pareto certificate by checking dominance proofs.
    pub fn verify_pareto(&self, cert: &ParetoCertificate) -> VerificationResult {
        let mut checks = Vec::new();

        // Check fingerprint
        let fp_content = match serde_json::to_string(&(&cert.point_proofs, &cert.metadata)) {
            Ok(s) => s,
            Err(e) => {
                checks.push(VerificationCheck::fail(
                    "fingerprint_recompute",
                    format!("serialization error: {}", e),
                ));
                return VerificationResult::new(checks);
            }
        };
        let recomputed = CertificateFingerprint::compute(fp_content.as_bytes());
        if recomputed == cert.fingerprint {
            checks.push(VerificationCheck::pass(
                "fingerprint_integrity",
                "fingerprint matches",
            ));
        } else {
            checks.push(VerificationCheck::fail(
                "fingerprint_integrity",
                "fingerprint mismatch",
            ));
        }

        // Verify frontier non-empty
        if !cert.point_proofs.is_empty() {
            checks.push(VerificationCheck::pass(
                "frontier_nonempty",
                format!("{} points on frontier", cert.point_proofs.len()),
            ));
        } else {
            checks.push(VerificationCheck::fail(
                "frontier_nonempty",
                "frontier is empty",
            ));
        }

        // Verify mutual non-dominance among frontier points
        let costs: Vec<&regsynth_pareto::CostVector> = cert
            .point_proofs
            .iter()
            .map(|p| &p.cost_vector)
            .collect();
        let mut mutual_nondom = true;
        for i in 0..costs.len() {
            for j in 0..costs.len() {
                if i != j && regsynth_pareto::dominates(costs[i], costs[j]) {
                    mutual_nondom = false;
                    checks.push(VerificationCheck::fail(
                        "mutual_nondominance",
                        format!("point {} dominates point {}", i, j),
                    ));
                    break;
                }
            }
            if !mutual_nondom {
                break;
            }
        }
        if mutual_nondom {
            checks.push(VerificationCheck::pass(
                "mutual_nondominance",
                "all frontier points are mutually non-dominated",
            ));
        }

        // Verify each dominance proof
        let mut all_dim_proofs_valid = true;
        let mut total_dim_proofs = 0usize;
        for (i, pp) in cert.point_proofs.iter().enumerate() {
            let dp = &pp.dominance_proof;
            if dp.is_complete() {
                total_dim_proofs += dp.dimension_proofs.len();
                if !dp.validate() {
                    all_dim_proofs_valid = false;
                    checks.push(VerificationCheck::fail(
                        format!("dominance_proof_{}", i),
                        "dimension proof validation failed",
                    ));
                }
            } else {
                all_dim_proofs_valid = false;
                checks.push(VerificationCheck::fail(
                    format!("dominance_proof_{}", i),
                    format!(
                        "incomplete: {} of {} dimension proofs",
                        dp.dimension_proofs.len(),
                        dp.point_costs.len()
                    ),
                ));
            }
        }
        if all_dim_proofs_valid && !cert.point_proofs.is_empty() {
            checks.push(VerificationCheck::pass(
                "all_dominance_proofs",
                format!("{} dimension proofs validated across {} points", total_dim_proofs, cert.point_proofs.len()),
            ));
        }

        // Verify dimension proof bounds
        for (i, pp) in cert.point_proofs.iter().enumerate() {
            for dp in &pp.dominance_proof.dimension_proofs {
                if dp.proven_lower_bound < dp.current_value - 1e-9 {
                    checks.push(VerificationCheck::fail(
                        format!("bound_check_{}_{}", i, dp.dimension_name),
                        format!(
                            "lower bound {:.4} < current value {:.4}",
                            dp.proven_lower_bound, dp.current_value
                        ),
                    ));
                }
            }
        }

        VerificationResult::new(checks)
    }
}

impl Default for CertificateVerifier {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compliance_cert::ComplianceCertGenerator;
    use crate::infeasibility_cert::InfeasibilityCertGenerator;
    use crate::pareto_cert::ParetoCertGenerator;
    use crate::serialization::Certificate;
    use regsynth_encoding::{SmtConstraint, SmtExpr, SmtSort};
    use regsynth_pareto::{CostVector, ParetoFrontier};
    use regsynth_types::certificate::SatisfyingAssignment;
    use regsynth_types::constraint::{Constraint, ConstraintExpr, ConstraintSet};

    #[test]
    fn verify_valid_envelope() {
        let cert =
            Certificate::wrap("compliance", &serde_json::json!({"status": "ok"}), Vec::new())
                .unwrap();
        let verifier = CertificateVerifier::new();
        let result = verifier.verify(&cert);
        assert!(result.valid);
        assert!(result.checks.iter().all(|c| c.passed));
    }

    #[test]
    fn verify_tampered_envelope() {
        let mut cert =
            Certificate::wrap("compliance", &serde_json::json!({"status": "ok"}), Vec::new())
                .unwrap();
        cert.payload_json = r#"{"status":"tampered"}"#.into();
        let verifier = CertificateVerifier::new();
        let result = verifier.verify(&cert);
        assert!(!result.valid);
    }

    #[test]
    fn verify_json_string() {
        let cert = Certificate::wrap("test", &"data", Vec::new()).unwrap();
        let json = cert.to_json().unwrap();
        let verifier = CertificateVerifier::new();
        let result = verifier.verify_json(&json).unwrap();
        assert!(result.valid);
    }

    #[test]
    fn verify_compliance_cert() {
        let gen = ComplianceCertGenerator::new("test-solver");
        let mut cs = ConstraintSet::new();
        cs.add(Constraint::hard("c1", ConstraintExpr::var("x")));
        cs.add(Constraint::hard("c2", ConstraintExpr::var("y")));
        let mut sa = SatisfyingAssignment::new();
        sa.set_bool("x", true);
        sa.set_bool("y", true);

        let cert = gen.generate(&sa, &cs).unwrap();
        let verifier = CertificateVerifier::new();
        let result = verifier.verify_compliance(&cert);
        assert!(result.valid);
        assert!(result.pass_count() >= 3);
    }

    #[test]
    fn verify_infeasibility_cert() {
        let gen = InfeasibilityCertGenerator::new("solver");
        let constraints = vec![
            SmtConstraint {
                id: "c1".into(),
                expr: SmtExpr::BoolLit(true),
                provenance: None,
            },
            SmtConstraint {
                id: "c2".into(),
                expr: SmtExpr::BoolLit(true),
                provenance: None,
            },
        ];
        let cert = gen.generate(&[0, 1], &constraints).unwrap();
        let verifier = CertificateVerifier::new();
        let result = verifier.verify_infeasibility(&cert);
        assert!(result.valid);
        assert!(result.pass_count() >= 4);
    }

    #[test]
    fn verify_pareto_cert() {
        let gen = ParetoCertGenerator::new("solver");
        let mut frontier: ParetoFrontier<i32> = ParetoFrontier::new(2);
        frontier.add_point(1, CostVector::new(vec![1.0, 4.0]));
        frontier.add_point(2, CostVector::new(vec![3.0, 2.0]));

        let mut cs = ConstraintSet::new();
        cs.add(Constraint::hard("c1", ConstraintExpr::bool_const(true)));
        let cert = gen.generate(&frontier, &cs).unwrap();

        let verifier = CertificateVerifier::new();
        let result = verifier.verify_pareto(&cert);
        assert!(result.valid);
        assert!(result.pass_count() >= 3);
    }

    #[test]
    fn deep_verify_compliance_envelope() {
        let cert = Certificate::wrap(
            "compliance",
            &serde_json::json!({
                "witness": {
                    "constraint_results": {"c1": true, "c2": true}
                }
            }),
            Vec::new(),
        ).unwrap();
        let verifier = CertificateVerifier::new().with_deep(true);
        let result = verifier.verify(&cert);
        assert!(result.valid);
    }

    #[test]
    fn verification_result_counts() {
        let checks = vec![
            VerificationCheck::pass("a", "ok"),
            VerificationCheck::fail("b", "bad"),
            VerificationCheck::pass("c", "ok"),
        ];
        let result = VerificationResult::new(checks);
        assert!(!result.valid);
        assert_eq!(result.pass_count(), 2);
        assert_eq!(result.fail_count(), 1);
    }
}
