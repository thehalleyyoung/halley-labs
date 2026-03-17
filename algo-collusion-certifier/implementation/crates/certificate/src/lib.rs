//! Certificate crate for the CollusionProof system.
//!
//! This crate provides the proof-carrying code infrastructure:
//! - Certificate DSL (abstract syntax tree)
//! - Proof term language
//! - Trusted proof checker kernel (≤2500 LoC)
//! - Rational arithmetic re-verification
//! - Certificate builder and optimizer
//! - Merkle tree evidence integrity
//! - Evidence bundle pipeline
//! - Certificate parser

pub mod ast;
pub mod proof_term;
pub mod checker;
pub mod rational_verifier;
pub mod certificate_builder;
pub mod merkle;
pub mod evidence_bundle;
pub mod parser;

pub use ast::{
    BinaryOp, BoolLiteral, CertificateAST, CertificateBody, CertificateHeader,
    ComparisonOp, Expression, IntervalLiteral, ProofStep, RationalLiteral, TypeAnnotation,
    UnaryOp,
};
pub use proof_term::{AxiomSchema, InferenceRule, ProofTerm};
pub use checker::{
    AlphaBudgetChecker, CheckerContext, ProofChecker, SegmentIsolationChecker, StepResult,
    VerificationError, VerificationReport, VerificationResult,
};
pub use rational_verifier::{
    ComparisonResult, DualPathComputation, F64ToRationalConverter, IntervalRationalBridge,
    OrderingVerification, RationalVerifier,
};
pub use certificate_builder::{
    AlphaBudgetPlanner, CertificateBuilder, CertificateOptimizer, CertificateSerializer,
    ProofConstructor, SegmentAllocator,
};
pub use merkle::{
    CertMerkleNode, CertMerkleProof, CertMerkleTree, EvidenceIntegrity, IncrementalMerkleTree,
    MerkleForest,
};
pub use evidence_bundle::{
    BundleBuilder, BundleMetadata, BundleSerializer, BundleVerifier, CertEvidenceBundle,
    StandaloneVerifier,
};
pub use parser::{parse_certificate, CertificateTokenizer, ParseError, RecursiveDescentParser, Token};
