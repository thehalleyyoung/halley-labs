//! Evaluation and benchmarking infrastructure for the Certified Leakage Contracts framework.
//!
//! This crate provides tools for evaluating the precision, performance, and correctness
//! of leakage analysis on x86-64 cryptographic binaries. It includes benchmark suites,
//! evaluation metrics, reporting, synthetic test generation, tool comparison, and
//! statistical analysis.

pub mod benchmark;
pub mod comparator;
pub mod generator;
pub mod metrics;
pub mod reporter;
pub mod stats;
pub mod csv_export;

pub use benchmark::{
    Benchmark, BenchmarkCategory, BenchmarkResult, BenchmarkRunner, BenchmarkSuite,
};
pub use comparator::{BaselineResult, ComparisonReport, ToolComparator, ToolProfile};
pub use generator::SyntheticGenerator;
pub use metrics::{
    FalsePositiveRate, MetricAggregator, OverheadMetrics, Precision, Recall, ScalabilityProfile,
    TightnessRatio,
};
pub use reporter::{
    BenchmarkReport, ComparisonTable, CsvReport, JsonReport, LatexTable, ReportConfig,
};
pub use stats::{ConfidenceInterval, CorrelationAnalysis, DescriptiveStats, EffectSize};
