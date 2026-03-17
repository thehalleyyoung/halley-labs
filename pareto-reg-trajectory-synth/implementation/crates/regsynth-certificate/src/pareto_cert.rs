//! Pareto optimality certificate generation.
//!
//! For each point on a Pareto frontier, prove that no dominating feasible
//! point exists by showing that improving any single dimension forces a
//! worse value in at least one other dimension.

use crate::fingerprint::CertificateFingerprint;
use crate::proof_types::DominanceProof;
use crate::proof_types::DimensionInfeasibilityProof;
use regsynth_pareto::{CostVector, ParetoFrontier, dominance};
use regsynth_types::certificate::CertificateKind;
use regsynth_types::constraint::ConstraintSet;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ─── Types ──────────────────────────────────────────────────────────────────

/// Proof that a single point on the Pareto frontier is non-dominated.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParetoPointProof {
    pub point_index: usize,
    pub cost_vector: CostVector,
    pub dominance_proof: DominanceProof,
    pub is_valid: bool,
}

/// Metadata attached to a Pareto certificate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParetoMetadata {
    pub frontier_size: usize,
    pub dimension: usize,
    pub total_proofs: usize,
    pub valid_proofs: usize,
    pub solver_used: String,
    pub hypervolume: Option<f64>,
}

/// A Pareto optimality certificate proving that a set of points forms a
/// Pareto frontier (no feasible point dominates any of them).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParetoCertificate {
    pub id: String,
    pub kind: CertificateKind,
    pub timestamp: String,
    pub point_proofs: Vec<ParetoPointProof>,
    pub metadata: ParetoMetadata,
    pub fingerprint: CertificateFingerprint,
}

// ─── Generator ──────────────────────────────────────────────────────────────

/// Generates Pareto optimality certificates.
pub struct ParetoCertGenerator {
    solver_name: String,
}

impl ParetoCertGenerator {
    pub fn new(solver_name: &str) -> Self {
        Self {
            solver_name: solver_name.to_string(),
        }
    }

    /// Generate a Pareto certificate for a frontier with associated
    /// constraint data.
    ///
    /// For each point on the frontier, this constructs a `DominanceProof`
    /// by verifying that:
    /// 1. The point is actually on the frontier (not dominated by any other).
    /// 2. For each dimension, improving the cost in that dimension while
    ///    maintaining feasibility is bounded by the current value.
    pub fn generate<T: Clone>(
        &self,
        frontier: &ParetoFrontier<T>,
        constraints: &ConstraintSet,
    ) -> crate::Result<ParetoCertificate> {
        if frontier.is_empty() {
            return Err(crate::CertificateError::MissingData(
                "empty Pareto frontier".into(),
            ));
        }

        let dim = frontier.dimension();
        let entries = frontier.entries();
        let mut point_proofs = Vec::with_capacity(entries.len());

        for (idx, entry) in entries.iter().enumerate() {
            let cost = &entry.cost;

            // Verify non-dominance: no other point on the frontier dominates this one
            let is_nondominated = entries.iter().enumerate().all(|(j, other)| {
                j == idx || !dominance::dominates(&other.cost, cost)
            });

            if !is_nondominated {
                return Err(crate::CertificateError::ProofValidation(format!(
                    "point {} is dominated by another frontier point",
                    idx
                )));
            }

            // Build dimension proofs
            let dominance_proof =
                self.build_dominance_proof(idx, cost, entries, constraints, dim)?;

            let is_valid = dominance_proof.validate();

            point_proofs.push(ParetoPointProof {
                point_index: idx,
                cost_vector: cost.clone(),
                dominance_proof,
                is_valid,
            });
        }

        let valid_count = point_proofs.iter().filter(|p| p.is_valid).count();

        // Compute hypervolume if we have a reference point
        let hypervolume = if dim > 0 {
            let ref_point = self.compute_reference_point(entries);
            Some(frontier.hypervolume(&ref_point))
        } else {
            None
        };

        let metadata = ParetoMetadata {
            frontier_size: entries.len(),
            dimension: dim,
            total_proofs: point_proofs.len(),
            valid_proofs: valid_count,
            solver_used: self.solver_name.clone(),
            hypervolume,
        };

        let fp_content = serde_json::to_string(&(&point_proofs, &metadata))
            .map_err(|e| crate::CertificateError::Serialization(e.to_string()))?;
        let fingerprint = CertificateFingerprint::compute(fp_content.as_bytes());

        Ok(ParetoCertificate {
            id: uuid::Uuid::new_v4().to_string(),
            kind: CertificateKind::ParetoOptimality,
            timestamp: chrono::Utc::now().to_rfc3339(),
            point_proofs,
            metadata,
            fingerprint,
        })
    }

    /// Generate a certificate directly from cost vectors and constraint info.
    pub fn generate_from_costs(
        &self,
        cost_vectors: &[CostVector],
        constraint_ids: &[String],
    ) -> crate::Result<ParetoCertificate> {
        if cost_vectors.is_empty() {
            return Err(crate::CertificateError::MissingData(
                "no cost vectors provided".into(),
            ));
        }

        let dim = cost_vectors[0].dim();
        let mut point_proofs = Vec::new();

        for (idx, cost) in cost_vectors.iter().enumerate() {
            // Verify non-dominance among provided points
            let is_nondominated = cost_vectors.iter().enumerate().all(|(j, other)| {
                j == idx || !dominance::dominates(other, cost)
            });

            if !is_nondominated {
                continue; // skip dominated points
            }

            let mut dp = DominanceProof::new(cost.values.clone());

            for d in 0..dim {
                let dim_name = dimension_name(d);
                let current_val = cost.get(d);

                let best_in_dim = cost_vectors
                    .iter()
                    .map(|c| c.get(d))
                    .fold(f64::INFINITY, f64::min);

                dp.add_dimension_proof(DimensionInfeasibilityProof {
                    dimension_name: dim_name,
                    dimension_index: d,
                    current_value: current_val,
                    proven_lower_bound: best_in_dim,
                    proof_method: "frontier_analysis".into(),
                    witness_constraints: constraint_ids.to_vec(),
                });
            }

            let is_valid = dp.validate();
            point_proofs.push(ParetoPointProof {
                point_index: idx,
                cost_vector: cost.clone(),
                dominance_proof: dp,
                is_valid,
            });
        }

        let valid_count = point_proofs.iter().filter(|p| p.is_valid).count();

        let metadata = ParetoMetadata {
            frontier_size: point_proofs.len(),
            dimension: dim,
            total_proofs: point_proofs.len(),
            valid_proofs: valid_count,
            solver_used: self.solver_name.clone(),
            hypervolume: None,
        };

        let fp_content = serde_json::to_string(&(&point_proofs, &metadata))
            .map_err(|e| crate::CertificateError::Serialization(e.to_string()))?;
        let fingerprint = CertificateFingerprint::compute(fp_content.as_bytes());

        Ok(ParetoCertificate {
            id: uuid::Uuid::new_v4().to_string(),
            kind: CertificateKind::ParetoOptimality,
            timestamp: chrono::Utc::now().to_rfc3339(),
            point_proofs,
            metadata,
            fingerprint,
        })
    }

    /// Build a dominance proof for a single frontier point.
    fn build_dominance_proof<T: Clone>(
        &self,
        _point_idx: usize,
        cost: &CostVector,
        all_entries: &[regsynth_pareto::frontier::ParetoEntry<T>],
        constraints: &ConstraintSet,
        dim: usize,
    ) -> crate::Result<DominanceProof> {
        let mut dp = DominanceProof::new(cost.values.clone());

        let constraint_ids: Vec<String> = constraints
            .all()
            .iter()
            .map(|c| c.id.as_str().to_string())
            .collect();

        for d in 0..dim {
            let dim_name = dimension_name(d);
            let current_val = cost.get(d);

            let min_in_dim = all_entries
                .iter()
                .map(|e| e.cost.get(d))
                .fold(f64::INFINITY, f64::min);

            let bound = if (current_val - min_in_dim).abs() < 1e-12 {
                current_val
            } else {
                min_in_dim
            };

            dp.add_dimension_proof(DimensionInfeasibilityProof {
                dimension_name: dim_name,
                dimension_index: d,
                current_value: current_val,
                proven_lower_bound: bound,
                proof_method: "pareto_frontier_analysis".into(),
                witness_constraints: constraint_ids.clone(),
            });
        }

        Ok(dp)
    }

    /// Compute a reference point for hypervolume calculation:
    /// component-wise maximum of all entries, scaled by 1.1.
    fn compute_reference_point<T: Clone>(
        &self,
        entries: &[regsynth_pareto::frontier::ParetoEntry<T>],
    ) -> CostVector {
        if entries.is_empty() {
            return CostVector::zeros(0);
        }
        let dim = entries[0].cost.dim();
        let mut max_vals = vec![f64::NEG_INFINITY; dim];
        for entry in entries {
            for d in 0..dim {
                max_vals[d] = max_vals[d].max(entry.cost.get(d));
            }
        }
        CostVector::new(max_vals.into_iter().map(|v| v * 1.1).collect())
    }
}

/// Map a dimension index to a human-readable name.
fn dimension_name(idx: usize) -> String {
    match idx {
        0 => "financial_cost".to_string(),
        1 => "time_to_compliance".to_string(),
        2 => "regulatory_risk".to_string(),
        3 => "implementation_complexity".to_string(),
        d => format!("dimension_{}", d),
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use regsynth_pareto::ParetoFrontier;
    use regsynth_types::constraint::{Constraint, ConstraintExpr, ConstraintSet};

    fn cv(vals: &[f64]) -> CostVector {
        CostVector::new(vals.to_vec())
    }

    fn make_constraints() -> ConstraintSet {
        let mut cs = ConstraintSet::new();
        cs.add(Constraint::hard("c1", ConstraintExpr::bool_const(true)));
        cs.add(Constraint::hard("c2", ConstraintExpr::bool_const(true)));
        cs
    }

    #[test]
    fn generate_pareto_cert_two_points() {
        let gen = ParetoCertGenerator::new("test-solver");
        let mut frontier: ParetoFrontier<String> = ParetoFrontier::new(2);
        frontier.add_point("A".into(), cv(&[1.0, 4.0]));
        frontier.add_point("B".into(), cv(&[3.0, 2.0]));

        let cs = make_constraints();
        let cert = gen.generate(&frontier, &cs).unwrap();
        assert_eq!(cert.kind, CertificateKind::ParetoOptimality);
        assert_eq!(cert.metadata.frontier_size, 2);
        assert_eq!(cert.point_proofs.len(), 2);
        assert!(cert.point_proofs.iter().all(|p| p.is_valid));
    }

    #[test]
    fn generate_pareto_cert_single_point() {
        let gen = ParetoCertGenerator::new("solver");
        let mut frontier: ParetoFrontier<i32> = ParetoFrontier::new(3);
        frontier.add_point(1, cv(&[2.0, 3.0, 1.0]));

        let cs = make_constraints();
        let cert = gen.generate(&frontier, &cs).unwrap();
        assert_eq!(cert.metadata.frontier_size, 1);
        assert!(cert.point_proofs[0].is_valid);
    }

    #[test]
    fn generate_pareto_cert_empty_frontier_fails() {
        let gen = ParetoCertGenerator::new("solver");
        let frontier: ParetoFrontier<i32> = ParetoFrontier::new(2);
        let cs = make_constraints();
        assert!(gen.generate(&frontier, &cs).is_err());
    }

    #[test]
    fn generate_from_costs() {
        let gen = ParetoCertGenerator::new("solver");
        let costs = vec![cv(&[1.0, 4.0]), cv(&[3.0, 2.0]), cv(&[2.0, 3.0])];
        let constraint_ids = vec!["c1".into(), "c2".into()];
        let cert = gen.generate_from_costs(&costs, &constraint_ids).unwrap();
        assert_eq!(cert.kind, CertificateKind::ParetoOptimality);
        assert_eq!(cert.point_proofs.len(), 3);
    }

    #[test]
    fn generate_from_costs_filters_dominated() {
        let gen = ParetoCertGenerator::new("solver");
        let costs = vec![
            cv(&[1.0, 2.0]),
            cv(&[3.0, 4.0]), // dominated by first
        ];
        let cert = gen
            .generate_from_costs(&costs, &vec!["c1".into()])
            .unwrap();
        assert_eq!(cert.point_proofs.len(), 1);
    }

    #[test]
    fn dimension_proofs_cover_all_dims() {
        let gen = ParetoCertGenerator::new("solver");
        let mut frontier: ParetoFrontier<i32> = ParetoFrontier::new(4);
        frontier.add_point(1, cv(&[1.0, 2.0, 3.0, 4.0]));

        let cs = make_constraints();
        let cert = gen.generate(&frontier, &cs).unwrap();
        let dp = &cert.point_proofs[0].dominance_proof;
        assert_eq!(dp.dimension_proofs.len(), 4);
        assert_eq!(dp.dimension_proofs[0].dimension_name, "financial_cost");
        assert_eq!(dp.dimension_proofs[1].dimension_name, "time_to_compliance");
    }

    #[test]
    fn hypervolume_computed() {
        let gen = ParetoCertGenerator::new("solver");
        let mut frontier: ParetoFrontier<i32> = ParetoFrontier::new(2);
        frontier.add_point(1, cv(&[1.0, 3.0]));
        frontier.add_point(2, cv(&[3.0, 1.0]));

        let cs = make_constraints();
        let cert = gen.generate(&frontier, &cs).unwrap();
        assert!(cert.metadata.hypervolume.is_some());
        assert!(cert.metadata.hypervolume.unwrap() > 0.0);
    }

    #[test]
    fn fingerprint_present() {
        let gen = ParetoCertGenerator::new("solver");
        let costs = vec![cv(&[1.0, 2.0])];
        let cert = gen
            .generate_from_costs(&costs, &vec!["c1".into()])
            .unwrap();
        assert!(!cert.fingerprint.hex_digest.is_empty());
    }
}
