//! Fault localization engine combining SBFL metrics, causal analysis,
//! and discriminability testing to rank suspicious pipeline stages.
//!
//! # Architecture
//!
//! - [`engine`] — Core localization engine with SBFL scoring and causal analysis.
//! - [`runner`] — Test suite runner orchestrating end-to-end localization.
//! - [`classification`] — Fault classification and causal verdict types.

pub mod classification;
pub mod engine;
pub mod runner;

pub use classification::{CausalVerdict, FaultClassification, InterventionResult};
pub use engine::{
    compute_discriminability, DiscriminabilityReport, LocalizationConfig, LocalizationEngine,
    PeelingRound, PeelingState, SBFLMetric, StageIntervention, StageSpectrum, TestObservation,
};
pub use runner::{
    CoverageReport, MetamorphicTestCase, PipelineAdapter, RunnerConfig, SuiteRunResult,
    SuiteRunner, TestCaseBuilder, TestCaseResult, TestSuite, TransformationAdapter,
};

use serde::{Deserialize, Serialize};
use shared_types::StageId;
use std::collections::HashMap;

/// Complete result of a localization analysis run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalizationResult {
    pub pipeline_name: String,
    pub stage_results: Vec<StageLocalizationResult>,
    pub test_count: usize,
    pub violation_count: usize,
    pub transformations_used: Vec<String>,
    pub metadata: HashMap<String, String>,
}

/// Per-stage localization data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageLocalizationResult {
    pub stage_name: String,
    pub stage_id: StageId,
    pub suspiciousness: f64,
    pub rank: usize,
    pub fault_type: Option<String>,
    pub evidence: Vec<String>,
    pub differential_data: Vec<f64>,
    pub per_transformation: HashMap<String, TransformationStageData>,
}

/// Per-transformation differential data at a stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformationStageData {
    pub transformation_name: String,
    pub mean_differential: f64,
    pub sample_count: usize,
    pub violation_count: usize,
}

/// Ordered suspiciousness ranking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuspiciousnessRanking {
    pub rankings: Vec<SuspiciousnessEntry>,
}

/// A single entry in the suspiciousness ranking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuspiciousnessEntry {
    pub stage_name: String,
    pub score: f64,
    pub rank: usize,
}
