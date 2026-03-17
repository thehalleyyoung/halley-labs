//! Report generation, behavioral atlas, and counterexample management.
//!
//! This crate provides:
//! - [`atlas`]: Behavioral atlas generation with BFI computation
//! - [`bfi`]: Behavioral Fragility Index (M7) computation
//! - [`counterexample`]: Counterexample database and regression test export
//! - [`summary`]: Full localization report generation
//! - [`export`]: Multi-format export (JSON, CSV, Markdown, HTML)

pub mod atlas;
pub mod bfi;
pub mod counterexample;
pub mod export;
pub mod summary;

pub use atlas::{
    AtlasRenderer, BehavioralAtlas, InteractionEntry, JsonAtlasRenderer, MarkdownAtlasRenderer,
    PlainTextAtlasRenderer, StageCoverage, StageAtlasEntry, TransformationAtlasEntry,
};
pub use bfi::{BFIComputer, BFIInterpretation, BFIProfile, BFIResult, BFITrend};
pub use counterexample::{
    CounterexampleDB, CounterexampleEntry, QueryBuilder, RegressionTest, RegressionTestSuite,
};
pub use export::{
    CsvExporter, ExportBundle, ExportConfig, Exporter, HtmlExporter, JsonExporter,
    MarkdownExporter, TemplateEngine,
};
pub use summary::{
    Appendix, Evidence, Finding, LocalizationReport, MethodologySection, Recommendation,
    ReportFormat, ReportGenerator, ReportHeader, StatisticalSection,
};
