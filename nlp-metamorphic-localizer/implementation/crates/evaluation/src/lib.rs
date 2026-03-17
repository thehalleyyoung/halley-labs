//! Evaluation framework for measuring fault localization effectiveness.
//!
//! Provides metrics, benchmarking, fault injection, and ground truth comparison
//! to assess how well the localizer identifies faulty pipeline stages.

pub mod benchmarks;
pub mod fault_injection;
pub mod metrics;
pub mod ground_truth;
pub mod harness;

pub use benchmarks::{Benchmark, BenchmarkConfig, BenchmarkResult, BenchmarkSuite};
pub use fault_injection::{FaultInjector, InjectedFault, InjectionSite, FaultProfile};
pub use metrics::{
    AccuracyMetrics, LocalizationAccuracy, RankingMetrics, TopKAccuracy,
    WastedEffort, EXAM, MeanFirstRank,
};
pub use ground_truth::{GroundTruth, GroundTruthEntry, GroundTruthBuilder};
pub use harness::{EvaluationHarness, EvaluationConfig, EvaluationReport};
