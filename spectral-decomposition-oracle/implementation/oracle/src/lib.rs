//! Decomposition selection oracle for MIP instances.
//!
//! This crate implements the full prediction pipeline for recommending
//! decomposition methods (Benders, Dantzig-Wolfe, Lagrangian, or None)
//! for mixed-integer programming instances based on spectral and
//! structural features.
//!
//! # Architecture
//!
//! - [`condition_aware`] — Condition-number-aware method selector that routes
//!   big-M / ill-conditioned instances to structure-exploiting decomposition.
//! - [`classifier`] — Ensemble classifiers (Random Forest, Gradient Boosting,
//!   Logistic Regression, Voting, and Stacking).
//! - [`futility`] — Futility prediction to skip unpromising decompositions.
//! - [`structure`] — Structural detectors for Benders/Dantzig-Wolfe patterns.
//! - [`pipeline`] — End-to-end oracle pipeline combining features → classification → recommendation.
//! - [`evaluation`] — Nested cross-validation, ablation studies, and hypothesis testing.
//! - [`model`] — Model persistence, metadata tracking, and AutoML model selection.

pub mod error;
pub mod classifier;
pub mod condition_aware;
pub mod futility;
pub mod structure;
pub mod pipeline;
pub mod evaluation;
pub mod model;

// Re-exports for convenient access.
pub use error::{OracleError, OracleResult};
pub use classifier::traits::{
    Classifier, ClassificationMetrics, DecompositionMethod, FeatureVector,
};
pub use classifier::{
    RandomForest, RandomForestParams,
    GradientBoostingClassifier, GradientBoostingParams,
    LogisticRegression, LogisticRegressionParams,
    VotingClassifier, StackingClassifier, VotingStrategy,
    Dataset,
};
pub use futility::{FutilityPredictor, FutilityPrediction, FutilityFeatures};
pub use structure::{
    StructureDetector, StructureType, BendersDetector, DWDetector,
};
pub use pipeline::{
    OraclePipeline, PipelineConfig, PipelineResult,
    CensusPipeline, CensusTier,
    GroundTruthLabeler, TimeCutoff,
};
pub use evaluation::{
    NestedCV, StratifiedKFold, CVResults,
    AblationStudy, AblationConfig,
    HypothesisHarness, HypothesisResult,
};
pub use model::{ModelStore, ModelMetadata, ModelSelector, AutoMLConfig};
pub use condition_aware::{
    ConditionAwareSelector, SelectionResult, DecompositionRoute, BoundTightness,
    BigMDetector, BigMReport, BigMConstraint,
    AdaptiveDecomposition, AdaptiveDecompositionResult, Block,
    QualityPredictor, QualityPrediction,
};
