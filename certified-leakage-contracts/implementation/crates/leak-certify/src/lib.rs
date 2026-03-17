//! Certificate generation and checking for the Certified Leakage Contracts framework.
//!
//! This crate provides independently-verifiable proofs that information leakage bounds
//! hold for analyzed cryptographic binaries. Certificates form hash-linked chains that
//! connect function-level leakage proofs through compositional reasoning up to
//! whole-library guarantees.
//!
//! # Architecture
//!
//! - [`certificate`]: Core certificate types with hash-chain integrity
//! - [`chain`]: Certificate chains linking function → composition → library
//! - [`witness`]: Checkable evidence (fixpoint traces, counting arguments, etc.)
//! - [`checker`]: Independent verification without re-running analysis
//! - [`generator`]: Certificate production from analysis results
//! - [`format`]: Serialization and export in multiple formats
//! - [`audit`]: Audit trail for compliance-ready documentation

pub mod certificate;
pub mod chain;
pub mod witness;
pub mod checker;
pub mod generator;
pub mod format;
pub mod audit;

pub use certificate::{
    Certificate, CertificateId, CertificateKind, Claim, Evidence,
    CertificateSubject, CertificateHash, ClaimProperty,
};
pub use chain::{CertificateChain, ChainValidator, ChainBuilder};
pub use witness::{
    Witness, FixpointWitness, CountingWitness, ReductionWitness,
    CompositionWitness, WitnessChecker,
};
pub use checker::{CertificateChecker, CheckResult, CheckStatus};
pub use generator::CertificateGenerator;
pub use format::{CertificateFormat, CertificateExporter, CertificateImporter};
pub use audit::{AuditLog, AuditEntry, AuditTrail, AuditReport};
