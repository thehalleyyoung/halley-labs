//! # test-gen: Gap Analysis, Witness Generation, and Test Synthesis
//!
//! This crate implements the core theorem of the MutSpec project:
//!
//! > **Every surviving non-equivalent mutant that violates the inferred contract
//! > is a concrete witness to a latent defect or test-suite gap.**
//!
//! ## Architecture
//!
//! The crate is organized into the following modules:
//!
//! - [`gap_analysis`]: Algorithm A3 – classify surviving mutants into equivalents,
//!   test gaps, contract gaps, and benign survivors.
//! - [`witness`]: Distinguishing-input computation via SMT, minimal witness
//!   extraction, and witness validation.
//! - [`sarif`]: SARIF 2.1.0 compliant report generation for IDE integration.
//! - [`ranking`]: Multi-criteria bug-report ranking with configurable weights.
//! - [`test_synthesis`]: Synthesize executable test cases from witnesses.
//! - [`report`]: Human-readable report generation (Markdown, plain text, JSON).
//! - [`pipeline`]: End-to-end analysis pipeline with checkpointing.
//!
//! ## Dependency Type Re-exports
//!
//! Since the companion crates (`shared-types`, `mutation-core`, `smt-solver`,
//! `contract-synth`, `coverage`, `program-analysis`) are under parallel
//! development, this crate defines compatible type stubs in [`types`] so that
//! it can be built and tested independently. Once the companion crates are
//! complete these stubs will be replaced by proper re-exports.

pub mod types;
pub mod gap_analysis;
pub mod witness;
pub mod sarif;
pub mod ranking;
pub mod test_synthesis;
pub mod report;
pub mod pipeline;

pub use gap_analysis::{
    GapAnalyzer, GapAnalyzerConfig, GapResult, GapClassification,
    GapWitness, GapStatistics, BatchGapResult,
};
pub use witness::{
    WitnessGenerator, WitnessGeneratorConfig, WitnessInfo,
    WitnessExplanation, WitnessValidationResult,
};
pub use sarif::{
    SarifReport, SarifRun, SarifResult, SarifLocation, SarifMessage,
    SarifArtifact, SarifRule, SarifReportBuilder,
};
pub use ranking::{
    BugRanker, RankerConfig, RankedBugReport, RankingCriteria,
    ScoreBreakdown, DeduplicationStrategy,
};
pub use test_synthesis::{
    TestSynthesizer, TestSynthConfig, SynthesizedTest, TestSuite,
    TestTemplate, AssertionKind,
};
pub use report::{
    ReportGenerator, ReportConfig, ReportFormat, ReportSection,
    FormattedReport,
};
pub use pipeline::{
    AnalysisPipeline, PipelineConfig, PipelineStage, PipelineState,
    PipelineResult, PipelineStatistics,
};

/// Crate-level error type.
#[derive(Debug, thiserror::Error)]
pub enum TestGenError {
    #[error("gap analysis failed: {0}")]
    GapAnalysis(String),
    #[error("witness generation failed: {0}")]
    WitnessGeneration(String),
    #[error("SARIF generation failed: {0}")]
    Sarif(String),
    #[error("ranking failed: {0}")]
    Ranking(String),
    #[error("test synthesis failed: {0}")]
    TestSynthesis(String),
    #[error("report generation failed: {0}")]
    Report(String),
    #[error("pipeline error: {0}")]
    Pipeline(String),
    #[error("SMT solver error: {0}")]
    Smt(String),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, TestGenError>;

/// Prelude module for convenient imports.
pub mod prelude {
    pub use crate::types::*;
    pub use crate::{
        GapAnalyzer, GapAnalyzerConfig, GapResult, GapClassification,
        GapWitness, GapStatistics,
        WitnessGenerator, WitnessGeneratorConfig, WitnessInfo,
        BugRanker, RankerConfig, RankedBugReport,
        TestSynthesizer, TestSynthConfig, SynthesizedTest, TestSuite,
        ReportGenerator, ReportConfig, ReportFormat,
        AnalysisPipeline, PipelineConfig,
        SarifReport, SarifReportBuilder,
        TestGenError, Result,
    };
}
