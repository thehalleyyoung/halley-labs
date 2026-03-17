//! BiCut Core: structural analysis, reformulation selection, problem classification,
//! and constraint qualification verification for bilevel optimization programs.
//!
//! This crate provides the analytical backbone of the BiCut compiler.  Given a
//! `BilevelProblem` from `bicut_types`, the modules here determine:
//!
//! - **What** kind of bilevel problem it is (classification, analysis)
//! - **Whether** it satisfies required regularity conditions (cq_verify)
//! - **How** to best reformulate it (reformulation)
//! - **Whether** the lower level is well-posed (boundedness)
//! - **How** leader/follower interact (coupling)
//! - **What** structure the dependency graph reveals (graph)
//! - **What** simplifications are available (preprocess)
//! - **Whether** the input is well-formed (validate)

pub mod analysis;
pub mod boundedness;
pub mod classification;
pub mod coupling;
pub mod cq_verify;
pub mod graph;
pub mod preprocess;
pub mod reformulation;
pub mod validate;

// Re-export primary entry points
pub use analysis::{StructuralAnalysis, StructuralReport};
pub use boundedness::{BoundednessAnalyzer, BoundednessResult, BoundednessStatus};
pub use classification::{ClassificationReport, ProblemClassifier};
pub use coupling::{CouplingAnalyzer, CouplingReport, CouplingStrength};
pub use cq_verify::{CqVerificationReport, CqVerifier, VerificationTier};
pub use graph::{BlockDecomposition, DependencyGraph};
pub use preprocess::{PreprocessReport, Preprocessor};
pub use reformulation::{ReformulationSelector, ReformulationStrategy, StrategyCost};
pub use validate::{ProblemValidator, ValidationIssue, ValidationReport};
