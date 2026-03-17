use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use sha2::{Sha256, Digest};
use crate::cost::CostVector;
use crate::constraint::{ConstraintId, Clause};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CertificateKind {
    Compliance,
    Infeasibility,
    ParetoOptimality,
}

impl fmt::Display for CertificateKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CertificateKind::Compliance => write!(f, "Compliance"),
            CertificateKind::Infeasibility => write!(f, "Infeasibility"),
            CertificateKind::ParetoOptimality => write!(f, "Pareto Optimality"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Certificate {
    pub id: String,
    pub kind: CertificateKind,
    pub timestamp: String,
    pub solver_used: String,
    pub formalizability_coverage: f64,
    pub proof_witness: ProofWitness,
    pub metadata: HashMap<String, String>,
    pub fingerprint: String,
}

impl Certificate {
    pub fn new(kind: CertificateKind, solver: &str, witness: ProofWitness) -> Self {
        let id = uuid::Uuid::new_v4().to_string();
        let timestamp = chrono::Utc::now().to_rfc3339();
        let mut cert = Certificate {
            id, kind, timestamp, solver_used: solver.to_string(),
            formalizability_coverage: 1.0, proof_witness: witness,
            metadata: HashMap::new(), fingerprint: String::new(),
        };
        cert.fingerprint = cert.compute_fingerprint();
        cert
    }

    pub fn compute_fingerprint(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.id.as_bytes());
        hasher.update(format!("{:?}", self.kind).as_bytes());
        hasher.update(self.timestamp.as_bytes());
        hasher.update(self.solver_used.as_bytes());
        hasher.update(format!("{:?}", self.proof_witness).as_bytes());
        hex::encode(hasher.finalize())
    }

    pub fn verify_integrity(&self) -> bool {
        self.fingerprint == self.compute_fingerprint()
    }

    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self.fingerprint = self.compute_fingerprint();
        self
    }
}

impl fmt::Display for Certificate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Certificate({}, {}, fp={}...)", self.kind, self.id, &self.fingerprint[..16])
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProofWitness {
    SatisfyingAssignment(SatisfyingAssignment),
    ResolutionProof(ResolutionProof),
    DominanceProof(DominanceProof),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SatisfyingAssignment {
    pub bool_assignments: HashMap<String, bool>,
    pub real_assignments: HashMap<String, f64>,
    pub constraint_satisfaction: HashMap<String, bool>,
}

impl SatisfyingAssignment {
    pub fn new() -> Self {
        SatisfyingAssignment {
            bool_assignments: HashMap::new(),
            real_assignments: HashMap::new(),
            constraint_satisfaction: HashMap::new(),
        }
    }

    pub fn set_bool(&mut self, var: &str, val: bool) {
        self.bool_assignments.insert(var.to_string(), val);
    }

    pub fn set_real(&mut self, var: &str, val: f64) {
        self.real_assignments.insert(var.to_string(), val);
    }

    pub fn record_satisfaction(&mut self, constraint_id: &str, satisfied: bool) {
        self.constraint_satisfaction.insert(constraint_id.to_string(), satisfied);
    }

    pub fn all_satisfied(&self) -> bool {
        self.constraint_satisfaction.values().all(|&v| v)
    }

    pub fn satisfaction_count(&self) -> usize {
        self.constraint_satisfaction.values().filter(|&&v| v).count()
    }

    pub fn total_constraints(&self) -> usize {
        self.constraint_satisfaction.len()
    }
}

impl Default for SatisfyingAssignment {
    fn default() -> Self { Self::new() }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolutionStep {
    pub step_id: usize,
    pub parent1: usize,
    pub parent2: usize,
    pub pivot_variable: String,
    pub resolvent: Clause,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolutionProof {
    pub initial_clauses: Vec<(usize, Clause)>,
    pub steps: Vec<ResolutionStep>,
    pub empty_clause_step: Option<usize>,
    pub mus_constraint_ids: Vec<ConstraintId>,
}

impl ResolutionProof {
    pub fn new() -> Self {
        ResolutionProof {
            initial_clauses: Vec::new(), steps: Vec::new(),
            empty_clause_step: None, mus_constraint_ids: Vec::new(),
        }
    }

    pub fn add_initial_clause(&mut self, id: usize, clause: Clause) {
        self.initial_clauses.push((id, clause));
    }

    pub fn add_step(&mut self, parent1: usize, parent2: usize, pivot: &str, resolvent: Clause) -> usize {
        let step_id = self.initial_clauses.len() + self.steps.len();
        let is_empty = resolvent.is_empty();
        self.steps.push(ResolutionStep {
            step_id, parent1, parent2,
            pivot_variable: pivot.to_string(), resolvent,
        });
        if is_empty { self.empty_clause_step = Some(step_id); }
        step_id
    }

    pub fn is_complete(&self) -> bool { self.empty_clause_step.is_some() }

    pub fn proof_length(&self) -> usize { self.steps.len() }

    pub fn validate(&self) -> bool {
        for step in &self.steps {
            if step.parent1 >= step.step_id || step.parent2 >= step.step_id {
                return false;
            }
        }
        self.is_complete()
    }
}

impl Default for ResolutionProof {
    fn default() -> Self { Self::new() }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DominanceProof {
    pub pareto_point: CostVector,
    pub dimension_proofs: Vec<DimensionInfeasibility>,
    pub strategy_assignment: HashMap<String, bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DimensionInfeasibility {
    pub dimension: String,
    pub bound: f64,
    pub proof_type: DimensionProofType,
    pub witness_constraints: Vec<ConstraintId>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DimensionProofType {
    MaxSmtOptimum,
    IlpBound,
    InfeasibilityCore,
}

impl DominanceProof {
    pub fn new(point: CostVector) -> Self {
        DominanceProof {
            pareto_point: point, dimension_proofs: Vec::new(), strategy_assignment: HashMap::new(),
        }
    }

    pub fn add_dimension_proof(&mut self, dim: &str, bound: f64, proof_type: DimensionProofType) {
        self.dimension_proofs.push(DimensionInfeasibility {
            dimension: dim.to_string(), bound, proof_type, witness_constraints: Vec::new(),
        });
    }

    pub fn is_complete(&self) -> bool {
        self.dimension_proofs.len() == 4
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerificationStatus {
    Valid,
    Invalid,
    Unknown,
    Error,
}

impl fmt::Display for VerificationStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VerificationStatus::Valid => write!(f, "✓ Valid"),
            VerificationStatus::Invalid => write!(f, "✗ Invalid"),
            VerificationStatus::Unknown => write!(f, "? Unknown"),
            VerificationStatus::Error => write!(f, "! Error"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    pub status: VerificationStatus,
    pub details: Vec<String>,
    pub checked_at: String,
}

impl VerificationResult {
    pub fn valid() -> Self {
        VerificationResult {
            status: VerificationStatus::Valid,
            details: Vec::new(),
            checked_at: chrono::Utc::now().to_rfc3339(),
        }
    }

    pub fn invalid(reason: &str) -> Self {
        VerificationResult {
            status: VerificationStatus::Invalid,
            details: vec![reason.to_string()],
            checked_at: chrono::Utc::now().to_rfc3339(),
        }
    }

    pub fn is_valid(&self) -> bool { self.status == VerificationStatus::Valid }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_certificate_fingerprint() {
        let witness = ProofWitness::SatisfyingAssignment(SatisfyingAssignment::new());
        let cert = Certificate::new(CertificateKind::Compliance, "test_solver", witness);
        assert!(cert.verify_integrity());
        assert!(!cert.fingerprint.is_empty());
    }

    #[test]
    fn test_resolution_proof() {
        let mut proof = ResolutionProof::new();
        proof.add_initial_clause(0, Clause::new(vec![]));
        assert!(!proof.is_complete());
    }

    #[test]
    fn test_verification_result() {
        let valid = VerificationResult::valid();
        assert!(valid.is_valid());
        let invalid = VerificationResult::invalid("bad proof");
        assert!(!invalid.is_valid());
    }
}
