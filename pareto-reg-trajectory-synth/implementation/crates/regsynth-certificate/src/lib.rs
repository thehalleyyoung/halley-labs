//! # regsynth-certificate
//!
//! Certificate generation and verification for RegSynth.
//!
//! Provides three kinds of certificates:
//! - **Compliance**: proves a strategy assignment satisfies all constraints
//! - **Infeasibility**: proves no satisfying assignment exists (via resolution)
//! - **Pareto optimality**: proves no feasible point dominates a frontier point
//!
//! Each certificate carries a SHA-256 fingerprint and can be serialized for audit.

pub mod compliance_cert;
pub mod fingerprint;
pub mod infeasibility_cert;
pub mod pareto_cert;
pub mod proof_types;
pub mod serialization;
pub mod verifier;

// ─── Public Exports ───
pub use compliance_cert::{
    ComplianceCertGenerator, ComplianceCertificate, ComplianceMetadata,
    ConstraintSatisfactionProof,
};
pub use fingerprint::{
    CertificateChain, CertificateFingerprint, CertificateSignature, ChainEntry,
};
pub use infeasibility_cert::{
    ConflictCategory, ConflictSeverity, InfeasibilityCertGenerator, InfeasibilityCertificate,
    InfeasibilityMetadata, MinimalUnsatisfiableSubset, MusConstraint, RegulatoryConflict,
    RegulatoryDiagnosis,
};
pub use pareto_cert::{ParetoCertGenerator, ParetoCertificate, ParetoMetadata, ParetoPointProof};
pub use proof_types::{
    Clause, ClauseId, DimensionInfeasibilityProof, DominanceProof, Literal, ProofNode,
    ResolutionProof, ResolutionStep, SatisfactionWitness, WitnessValue,
};
pub use serialization::{AuditPackage, Certificate, CertificateFormat, SourceReference};
pub use verifier::{CertificateVerifier, VerificationCheck, VerificationResult};

// ─── Error Type ───
#[derive(Debug, thiserror::Error)]
pub enum CertificateError {
    #[error("proof validation failed: {0}")]
    ProofValidation(String),
    #[error("missing data: {0}")]
    MissingData(String),
    #[error("serialization error: {0}")]
    Serialization(String),
    #[error("fingerprint mismatch: expected {expected}, got {actual}")]
    FingerprintMismatch { expected: String, actual: String },
    #[error("invalid certificate: {0}")]
    InvalidCertificate(String),
    #[error("chain verification failed at index {index}: {reason}")]
    ChainVerification { index: usize, reason: String },
}

pub type Result<T> = std::result::Result<T, CertificateError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display() {
        let err = CertificateError::ProofValidation("step 3 invalid".into());
        assert!(format!("{}", err).contains("step 3"));
    }

    #[test]
    fn error_fingerprint_mismatch() {
        let err = CertificateError::FingerprintMismatch {
            expected: "aabb".into(),
            actual: "ccdd".into(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("aabb"));
        assert!(msg.contains("ccdd"));
    }

    #[test]
    fn error_chain_verification() {
        let err = CertificateError::ChainVerification {
            index: 2,
            reason: "hash break".into(),
        };
        assert!(format!("{}", err).contains("index 2"));
    }

    #[test]
    fn result_alias_works() {
        let ok: Result<i32> = Ok(42);
        assert!(ok.is_ok());
        let err: Result<i32> = Err(CertificateError::MissingData("x".into()));
        assert!(err.is_err());
    }
}
