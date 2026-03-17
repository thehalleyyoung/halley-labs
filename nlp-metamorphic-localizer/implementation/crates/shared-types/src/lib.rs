//! Shared types, traits, and error handling for the NLP metamorphic fault localizer.
//!
//! This crate provides foundational types used across all workspace crates,
//! including pipeline stage representations, intermediate representations,
//! distance metrics, configuration, and statistical utilities.

pub mod config;
pub mod distance;
pub mod error;
pub mod ir;
pub mod statistics;
pub mod traits;
pub mod types;

pub use config::{
    CalibrationConfig, LocalizerConfig, OutputFormat, PipelineConfig, ReportConfig, SBFLMetric,
    ShrinkingConfig, StatisticalConfig, TransformationConfig,
};
pub use distance::{
    ComponentDistance, DistanceConfig, DistanceMetric, DistanceValue, StageDistance,
};
pub use error::{LocalizerError, Result};
pub use ir::{IRSequence, IRSnapshot, IRType, IntermediateRepresentation};
pub use statistics::{ConfidenceInterval, DescriptiveStats, Histogram, RunningStats};
pub use traits::{
    DistanceComputer, MRCheckDetail, MetamorphicRelation, PipelineStage, Transformation,
    ValidityOracle,
};
pub use types::{
    DependencyEdge, DependencyRelation, EntityLabel, EntitySpan, Mood, ParseNode, ParseTree,
    PipelineId, PosTag, Sentence, SentenceFeatures, StageId, Tense, TestCaseId, Token,
    TransformationId, Voice,
};
