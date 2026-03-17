//! Benchmarking infrastructure for the BiCut bilevel optimization compiler.
//!
//! This crate provides:
//! - BOBILib instance loading and classification
//! - Benchmark instance management and filtering
//! - Performance measurement and metrics collection
//! - Benchmark runner with parallel execution
//! - Result comparison and statistical analysis
//! - Report generation (CSV, LaTeX, summary tables)
//! - Random instance generation for testing
//! - Performance profiling and time breakdowns

pub mod bobilib;
pub mod comparison;
pub mod generator;
pub mod instance;
pub mod metrics;
pub mod profile;
pub mod report;
pub mod runner;

// Re-export primary types from each module.
pub use bobilib::{BobilibCatalog, BobilibEntry, BobilibParser, ProblemClassification};
pub use comparison::{
    ComparisonReport, ConfigComparison, DomainAnalysis, PerformanceProfileData, VirtualBestSolver,
    WilcoxonResult,
};
pub use generator::{
    DensityProfile, GeneratorConfig, InstanceGenerator, KnapsackInterdictionConfig,
    NetworkInterdictionConfig,
};
pub use instance::{
    BenchmarkInstance, DifficultyClass, InstanceFilter, InstanceMetadata, InstanceSet, InstanceType,
};
pub use metrics::{
    AggregateMetrics, BenchmarkMetrics, CutStats, MetricsAggregator, PerformanceProfile,
};
pub use profile::{FlamegraphEntry, MemorySnapshot, PhaseProfile, PhaseTimer, ProfilingSession};
pub use report::{
    CsvReporter, LatexReporter, ReportConfig, ReportFormat, SummaryReporter, TimingBreakdown,
};
pub use runner::{
    BenchmarkConfig, BenchmarkRunner, CutConfig, ProgressCallback, ReformulationMethod, RunResult,
    RunStatus, SolverConfig,
};

use thiserror::Error;

/// Errors in benchmark operations.
#[derive(Error, Debug)]
pub enum BenchError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Parse error: {0}")]
    Parse(String),

    #[error("Instance not found: {0}")]
    InstanceNotFound(String),

    #[error("Timeout after {0} seconds")]
    Timeout(f64),

    #[error("Solver error: {0}")]
    SolverError(String),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Statistics error: {0}")]
    StatsError(String),
}

/// Result type for benchmark operations.
pub type BenchResult<T> = Result<T, BenchError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let e = BenchError::Timeout(60.0);
        assert!(e.to_string().contains("60"));
    }

    #[test]
    fn test_error_from_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "missing");
        let e = BenchError::from(io_err);
        assert!(e.to_string().contains("missing"));
    }

    #[test]
    fn test_parse_error() {
        let e = BenchError::Parse("bad format".into());
        assert!(e.to_string().contains("bad format"));
    }

    #[test]
    fn test_instance_not_found() {
        let e = BenchError::InstanceNotFound("foo.bos".into());
        assert!(e.to_string().contains("foo.bos"));
    }

    #[test]
    fn test_solver_error() {
        let e = BenchError::SolverError("numerical issue".into());
        assert!(e.to_string().contains("numerical"));
    }

    #[test]
    fn test_bench_result_ok() {
        let r: BenchResult<i32> = Ok(42);
        assert_eq!(r.unwrap(), 42);
    }
}
