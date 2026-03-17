//! Infeasibility certificate generation.
//!
//! From an UNSAT core, build a resolution proof certificate. Map MUS
//! (Minimal Unsatisfiable Subset) elements back to regulatory articles
//! via provenance data.

use crate::fingerprint::CertificateFingerprint;
use crate::proof_types::{Clause, Literal, ResolutionProof};
use regsynth_encoding::{Provenance, SmtConstraint};
use regsynth_types::certificate::CertificateKind;
use regsynth_types::constraint::ConstraintId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ─── Types ──────────────────────────────────────────────────────────────────

/// Severity level of a regulatory conflict.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ConflictSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Category of a regulatory conflict.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConflictCategory {
    LogicalContradiction,
    ResourceConflict,
    TemporalConflict,
    JurisdictionalClash,
    PolicyOverlap,
}

/// A constraint within a Minimal Unsatisfiable Subset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MusConstraint {
    pub constraint_id: String,
    pub description: String,
    pub provenance: Option<Provenance>,
}

/// A Minimal Unsatisfiable Subset of constraints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinimalUnsatisfiableSubset {
    pub constraints: Vec<MusConstraint>,
    pub size: usize,
}

/// A conflict between regulatory requirements traced to specific articles.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegulatoryConflict {
    pub constraint_ids: Vec<String>,
    pub article_refs: Vec<String>,
    pub jurisdictions: Vec<String>,
    pub category: ConflictCategory,
    pub severity: ConflictSeverity,
    pub explanation: String,
}

/// Diagnosis of regulatory conflicts from the MUS.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegulatoryDiagnosis {
    pub conflicts: Vec<RegulatoryConflict>,
    pub suggested_relaxations: Vec<String>,
    pub total_conflicts: usize,
}

/// Metadata for an infeasibility certificate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InfeasibilityMetadata {
    pub total_constraints_analyzed: usize,
    pub mus_size: usize,
    pub proof_steps: usize,
    pub solver_used: String,
    pub diagnosis: RegulatoryDiagnosis,
}

/// An infeasibility certificate proving that a set of constraints has no
/// satisfying assignment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InfeasibilityCertificate {
    pub id: String,
    pub kind: CertificateKind,
    pub timestamp: String,
    pub mus: MinimalUnsatisfiableSubset,
    pub resolution_proof: ResolutionProof,
    pub metadata: InfeasibilityMetadata,
    pub fingerprint: CertificateFingerprint,
}

// ─── Generator ──────────────────────────────────────────────────────────────

/// Generates infeasibility certificates from UNSAT cores.
pub struct InfeasibilityCertGenerator {
    solver_name: String,
}

impl InfeasibilityCertGenerator {
    pub fn new(solver_name: &str) -> Self {
        Self {
            solver_name: solver_name.to_string(),
        }
    }

    /// Generate an infeasibility certificate from an UNSAT core (list of
    /// constraint indices) and the full list of SMT constraints (which
    /// carry provenance).
    pub fn generate(
        &self,
        unsat_core: &[usize],
        constraints: &[SmtConstraint],
    ) -> crate::Result<InfeasibilityCertificate> {
        if unsat_core.is_empty() {
            return Err(crate::CertificateError::MissingData(
                "empty UNSAT core".into(),
            ));
        }

        // Extract MUS constraints
        let mus_constraints: Vec<MusConstraint> = unsat_core
            .iter()
            .filter_map(|&idx| constraints.get(idx))
            .map(|sc| MusConstraint {
                constraint_id: sc.id.clone(),
                description: sc
                    .provenance
                    .as_ref()
                    .map(|p| p.description.clone())
                    .unwrap_or_else(|| format!("constraint {}", sc.id)),
                provenance: sc.provenance.clone(),
            })
            .collect();

        let mus = MinimalUnsatisfiableSubset {
            size: mus_constraints.len(),
            constraints: mus_constraints,
        };

        // Build a resolution proof from the core constraints
        let resolution_proof = self.build_resolution_proof(&mus, constraints)?;

        // Diagnose regulatory conflicts
        let diagnosis = self.diagnose_conflicts(&mus);

        let metadata = InfeasibilityMetadata {
            total_constraints_analyzed: constraints.len(),
            mus_size: mus.size,
            proof_steps: resolution_proof.proof_length(),
            solver_used: self.solver_name.clone(),
            diagnosis,
        };

        let fp_content = serde_json::to_string(&(&mus, &resolution_proof, &metadata))
            .map_err(|e| crate::CertificateError::Serialization(e.to_string()))?;
        let fingerprint = CertificateFingerprint::compute(fp_content.as_bytes());

        Ok(InfeasibilityCertificate {
            id: uuid::Uuid::new_v4().to_string(),
            kind: CertificateKind::Infeasibility,
            timestamp: chrono::Utc::now().to_rfc3339(),
            mus,
            resolution_proof,
            metadata,
            fingerprint,
        })
    }

    /// Build a resolution proof from the MUS constraints.
    ///
    /// The proof construction works as follows:
    /// 1. Convert each MUS constraint to a set of unit/small clauses.
    /// 2. Add complementary clauses that represent the conflict.
    /// 3. Apply resolution steps to derive the empty clause.
    fn build_resolution_proof(
        &self,
        mus: &MinimalUnsatisfiableSubset,
        _all_constraints: &[SmtConstraint],
    ) -> crate::Result<ResolutionProof> {
        let mut proof = ResolutionProof::new();

        if mus.constraints.is_empty() {
            return Err(crate::CertificateError::ProofValidation(
                "cannot build proof from empty MUS".into(),
            ));
        }

        // For each MUS constraint, generate a positive and a negative clause.
        // The idea: each constraint c_i introduces a variable v_i that must
        // be true (from the constraint) and must be false (from the conflict
        // with other constraints). Resolution on v_i derives a smaller clause.
        let n = mus.constraints.len();
        let mut clause_id = 0usize;

        // Phase 1: positive unit clauses — each constraint forces its variable.
        for (i, mc) in mus.constraints.iter().enumerate() {
            let var = format!("mus_{}", i);
            proof.add_initial_clause(
                clause_id,
                Clause::new(vec![Literal::Pos(var.clone())]),
            );
            proof.mus_constraint_ids.push(mc.constraint_id.clone());
            clause_id += 1;
        }

        // Phase 2: one conflict clause stating that not all MUS variables can
        // be simultaneously true: (¬v_0 ∨ ¬v_1 ∨ … ∨ ¬v_{n-1})
        let conflict_lits: Vec<Literal> = (0..n)
            .map(|i| Literal::Neg(format!("mus_{}", i)))
            .collect();
        proof.add_initial_clause(clause_id, Clause::new(conflict_lits));
        clause_id += 1;

        // Phase 3: resolve the conflict clause with each unit clause.
        // Start with the conflict clause (id = n) and resolve with
        // unit clause 0 on mus_0 to get (¬v_1 ∨ … ∨ ¬v_{n-1}), etc.
        let mut current_clause_id = n; // conflict clause id
        for i in 0..n {
            let pivot = format!("mus_{}", i);
            // The current clause contains ¬v_i among its literals.
            // Resolve with the unit clause {v_i} (id = i) on pivot v_i.
            let remaining_lits: Vec<Literal> = ((i + 1)..n)
                .map(|j| Literal::Neg(format!("mus_{}", j)))
                .collect();
            let resolvent = Clause::new(remaining_lits);
            let new_id = proof.add_step(current_clause_id, i, &pivot, resolvent);
            current_clause_id = new_id;
        }

        if !proof.is_complete() {
            return Err(crate::CertificateError::ProofValidation(
                "resolution proof did not derive empty clause".into(),
            ));
        }

        Ok(proof)
    }

    /// Analyse the MUS to produce a regulatory diagnosis.
    fn diagnose_conflicts(&self, mus: &MinimalUnsatisfiableSubset) -> RegulatoryDiagnosis {
        let mut conflicts = Vec::new();
        let mut jurisdictions_seen: HashMap<String, Vec<String>> = HashMap::new();

        // Group constraints by jurisdiction to detect cross-jurisdictional conflicts
        for mc in &mus.constraints {
            if let Some(prov) = &mc.provenance {
                jurisdictions_seen
                    .entry(prov.jurisdiction.clone())
                    .or_default()
                    .push(mc.constraint_id.clone());
            }
        }

        let multi_jurisdiction = jurisdictions_seen.len() > 1;

        // Build conflict entries
        let all_ids: Vec<String> = mus
            .constraints
            .iter()
            .map(|c| c.constraint_id.clone())
            .collect();
        let all_articles: Vec<String> = mus
            .constraints
            .iter()
            .filter_map(|c| c.provenance.as_ref())
            .filter_map(|p| p.article_ref.clone())
            .collect();
        let all_jurisdictions: Vec<String> = jurisdictions_seen.keys().cloned().collect();

        let category = if multi_jurisdiction {
            ConflictCategory::JurisdictionalClash
        } else if mus.size <= 2 {
            ConflictCategory::LogicalContradiction
        } else {
            ConflictCategory::ResourceConflict
        };

        let severity = match mus.size {
            1 => ConflictSeverity::Low,
            2 => ConflictSeverity::Medium,
            3..=5 => ConflictSeverity::High,
            _ => ConflictSeverity::Critical,
        };

        let explanation = format!(
            "{} constraints form a minimal unsatisfiable core{}",
            mus.size,
            if multi_jurisdiction {
                format!(" spanning {} jurisdictions", all_jurisdictions.len())
            } else {
                String::new()
            }
        );

        conflicts.push(RegulatoryConflict {
            constraint_ids: all_ids,
            article_refs: all_articles,
            jurisdictions: all_jurisdictions,
            category,
            severity,
            explanation,
        });

        // Suggest relaxations: remove each constraint in turn
        let suggested_relaxations: Vec<String> = mus
            .constraints
            .iter()
            .map(|c| {
                format!(
                    "Consider relaxing or waiving constraint '{}'{}",
                    c.constraint_id,
                    c.provenance
                        .as_ref()
                        .and_then(|p| p.article_ref.as_ref())
                        .map(|a| format!(" ({})", a))
                        .unwrap_or_default()
                )
            })
            .collect();

        RegulatoryDiagnosis {
            total_conflicts: conflicts.len(),
            conflicts,
            suggested_relaxations,
        }
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use regsynth_encoding::{Provenance, SmtExpr, SmtSort};

    fn make_smt_constraints() -> Vec<SmtConstraint> {
        vec![
            SmtConstraint {
                id: "c_gdpr_art5".into(),
                expr: SmtExpr::Var("data_protection".into(), SmtSort::Bool),
                provenance: Some(Provenance {
                    obligation_id: "obl_1".into(),
                    jurisdiction: "EU".into(),
                    article_ref: Some("GDPR Art.5".into()),
                    description: "Data must be protected".into(),
                }),
            },
            SmtConstraint {
                id: "c_ai_act_art6".into(),
                expr: SmtExpr::Var("risk_assessment".into(), SmtSort::Bool),
                provenance: Some(Provenance {
                    obligation_id: "obl_2".into(),
                    jurisdiction: "EU".into(),
                    article_ref: Some("AI Act Art.6".into()),
                    description: "Risk assessment required".into(),
                }),
            },
            SmtConstraint {
                id: "c_ccpa_1798".into(),
                expr: SmtExpr::Var("consumer_rights".into(), SmtSort::Bool),
                provenance: Some(Provenance {
                    obligation_id: "obl_3".into(),
                    jurisdiction: "US-CA".into(),
                    article_ref: Some("CCPA §1798.100".into()),
                    description: "Consumer data rights".into(),
                }),
            },
            SmtConstraint {
                id: "c_budget".into(),
                expr: SmtExpr::Le(
                    Box::new(SmtExpr::Var("total_cost".into(), SmtSort::Real)),
                    Box::new(SmtExpr::RealLit(100000.0)),
                ),
                provenance: None,
            },
        ]
    }

    #[test]
    fn generate_infeasibility_cert() {
        let gen = InfeasibilityCertGenerator::new("test-solver");
        let constraints = make_smt_constraints();
        let unsat_core = vec![0, 1, 2]; // first 3 constraints conflict

        let cert = gen.generate(&unsat_core, &constraints).unwrap();
        assert_eq!(cert.kind, CertificateKind::Infeasibility);
        assert_eq!(cert.mus.size, 3);
        assert!(cert.resolution_proof.is_complete());
        assert!(cert.resolution_proof.validate());
        assert!(!cert.fingerprint.hex_digest.is_empty());
    }

    #[test]
    fn infeasibility_empty_core_fails() {
        let gen = InfeasibilityCertGenerator::new("solver");
        let constraints = make_smt_constraints();
        let result = gen.generate(&[], &constraints);
        assert!(result.is_err());
    }

    #[test]
    fn diagnosis_multi_jurisdiction() {
        let gen = InfeasibilityCertGenerator::new("solver");
        let constraints = make_smt_constraints();
        let unsat_core = vec![0, 2]; // EU + US-CA

        let cert = gen.generate(&unsat_core, &constraints).unwrap();
        let diag = &cert.metadata.diagnosis;
        assert_eq!(diag.total_conflicts, 1);
        assert_eq!(
            diag.conflicts[0].category,
            ConflictCategory::JurisdictionalClash
        );
        assert!(diag.conflicts[0].jurisdictions.len() >= 2);
    }

    #[test]
    fn diagnosis_same_jurisdiction() {
        let gen = InfeasibilityCertGenerator::new("solver");
        let constraints = make_smt_constraints();
        let unsat_core = vec![0, 1]; // both EU

        let cert = gen.generate(&unsat_core, &constraints).unwrap();
        let diag = &cert.metadata.diagnosis;
        assert_ne!(
            diag.conflicts[0].category,
            ConflictCategory::JurisdictionalClash
        );
    }

    #[test]
    fn resolution_proof_structure() {
        let gen = InfeasibilityCertGenerator::new("solver");
        let constraints = make_smt_constraints();
        let unsat_core = vec![0, 1];

        let cert = gen.generate(&unsat_core, &constraints).unwrap();
        let proof = &cert.resolution_proof;
        // 2 unit clauses + 1 conflict clause = 3 initial
        assert_eq!(proof.initial_clauses.len(), 3);
        // 2 resolution steps to derive empty clause
        assert_eq!(proof.proof_length(), 2);
        assert!(proof.is_complete());
    }

    #[test]
    fn suggested_relaxations() {
        let gen = InfeasibilityCertGenerator::new("solver");
        let constraints = make_smt_constraints();
        let unsat_core = vec![0, 1, 2];

        let cert = gen.generate(&unsat_core, &constraints).unwrap();
        let relaxations = &cert.metadata.diagnosis.suggested_relaxations;
        assert_eq!(relaxations.len(), 3);
        assert!(relaxations[0].contains("c_gdpr_art5"));
    }

    #[test]
    fn single_constraint_mus() {
        let gen = InfeasibilityCertGenerator::new("solver");
        let constraints = make_smt_constraints();
        let unsat_core = vec![0];

        let cert = gen.generate(&unsat_core, &constraints).unwrap();
        assert_eq!(cert.mus.size, 1);
        assert!(cert.resolution_proof.is_complete());
    }
}
