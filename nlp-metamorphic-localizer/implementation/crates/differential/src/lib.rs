//! Per-stage differential analysis for NLP metamorphic fault localization.
//!
//! This crate implements the core differential analysis framework:
//! - Per-stage differential computation (Δ_k)
//! - Stage Discriminability Matrix (N1)
//! - Interventional replay for causal refinement
//! - DCE / IE computation for causal fault localization
//! - Token-to-tree alignment utilities

pub mod alignment;
pub mod dce_ie;
pub mod discriminability;
pub mod intervention;
pub mod matrix;
pub mod stage_differential;

pub use alignment::{
    AlignmentFailure, AlignmentMap, AlignmentQuality, LemmaAligner,
    TransformationSpecificAligner, TreeAlignment,
};
pub use dce_ie::{CausalAnalyzer, CausalChain, CausalDecomposition, CausalEffect, FaultType};
pub use discriminability::{
    DiscriminabilityMatrix, DiscriminabilityReport, GaussianElimination,
    SingularValueDecomposition,
};
pub use intervention::{
    Intervention, InterventionCache, InterventionEngine, InterventionPlan, InterventionResult,
};
pub use matrix::{DifferentialMatrix, MatrixSlice};
pub use stage_differential::{
    DifferentialComputer, DifferentialTimeSeries, NormalizationParams, StageDifferential,
};
