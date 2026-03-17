//! # fpdiag-eval
//!
//! Evaluation and benchmarking harness for Penumbra.
//!
//! Provides infrastructure for running the full Penumbra pipeline
//! (trace → EAG → diagnose → repair → certify) on benchmark programs
//! and collecting metrics.

use fpdiag_types::{
    config::PenumbraConfig, diagnosis::DiagnosisReport, eag::ErrorAmplificationGraph,
    repair::RepairResult, trace::ExecutionTrace,
};
use serde::{Deserialize, Serialize};
use std::time::Instant;
use thiserror::Error;

/// Errors from the evaluation module.
#[derive(Debug, Error)]
pub enum EvalError {
    #[error("benchmark not found: {0}")]
    BenchmarkNotFound(String),
    #[error("pipeline stage failed: {stage}: {reason}")]
    PipelineFailed { stage: String, reason: String },
    #[error("ground truth not available for benchmark: {0}")]
    NoGroundTruth(String),
}

/// Metrics collected from a single benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkMetrics {
    /// Benchmark name.
    pub name: String,
    /// Number of traced operations.
    pub trace_events: u64,
    /// Number of EAG nodes.
    pub eag_nodes: usize,
    /// Number of EAG edges.
    pub eag_edges: usize,
    /// Estimated treewidth of the EAG.
    pub treewidth: Option<usize>,
    /// Number of high-error nodes diagnosed.
    pub diagnosed_nodes: usize,
    /// Diagnosis breakdown by category.
    pub diagnosis_categories: Vec<(String, usize)>,
    /// Number of repairs applied.
    pub repairs_applied: usize,
    /// Overall error reduction factor.
    pub error_reduction: f64,
    /// Whether all repairs are formally certified.
    pub fully_certified: bool,
    /// Instrumentation coverage fraction.
    pub coverage: f64,
    /// Tracing overhead (wall-clock ms).
    pub trace_time_ms: u64,
    /// EAG construction time (ms).
    pub eag_time_ms: u64,
    /// Diagnosis time (ms).
    pub diagnosis_time_ms: u64,
    /// Repair + certification time (ms).
    pub repair_time_ms: u64,
    /// Total pipeline time (ms).
    pub total_time_ms: u64,
}

impl BenchmarkMetrics {
    /// Create empty metrics for a named benchmark.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            trace_events: 0,
            eag_nodes: 0,
            eag_edges: 0,
            treewidth: None,
            diagnosed_nodes: 0,
            diagnosis_categories: Vec::new(),
            repairs_applied: 0,
            error_reduction: 1.0,
            fully_certified: false,
            coverage: 0.0,
            trace_time_ms: 0,
            eag_time_ms: 0,
            diagnosis_time_ms: 0,
            repair_time_ms: 0,
            total_time_ms: 0,
        }
    }
}

/// A benchmark definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Benchmark {
    /// Human-readable name.
    pub name: String,
    /// Description of the benchmark.
    pub description: String,
    /// Category (e.g., "cancellation", "absorption", "ill-conditioning").
    pub category: String,
    /// Expected diagnosis category.
    pub expected_category: Option<String>,
    /// Known error reduction achievable.
    pub known_reduction: Option<f64>,
}

/// The evaluation harness.
pub struct EvalHarness {
    config: PenumbraConfig,
    benchmarks: Vec<Benchmark>,
}

impl EvalHarness {
    /// Create a new harness.
    pub fn new(config: PenumbraConfig) -> Self {
        Self {
            config,
            benchmarks: Self::builtin_benchmarks(),
        }
    }

    /// Built-in benchmark suite.
    fn builtin_benchmarks() -> Vec<Benchmark> {
        vec![
            Benchmark {
                name: "catastrophic_cancellation_quadratic".to_string(),
                description: "Quadratic formula with near-zero discriminant".to_string(),
                category: "cancellation".to_string(),
                expected_category: Some("CatastrophicCancellation".to_string()),
                known_reduction: Some(1000.0),
            },
            Benchmark {
                name: "absorption_naive_sum".to_string(),
                description: "Naive summation of 10^6 terms with varying magnitude".to_string(),
                category: "absorption".to_string(),
                expected_category: Some("Absorption".to_string()),
                known_reduction: Some(100.0),
            },
            Benchmark {
                name: "smearing_alternating_series".to_string(),
                description: "Alternating harmonic series partial sum".to_string(),
                category: "smearing".to_string(),
                expected_category: Some("Smearing".to_string()),
                known_reduction: Some(50.0),
            },
            Benchmark {
                name: "ill_conditioned_hilbert".to_string(),
                description: "Solve Hilbert matrix linear system".to_string(),
                category: "ill-conditioning".to_string(),
                expected_category: Some("IllConditionedSubproblem".to_string()),
                known_reduction: Some(10.0),
            },
            Benchmark {
                name: "log_sum_exp_overflow".to_string(),
                description: "Log-sum-exp with large exponents (overflow risk)".to_string(),
                category: "cancellation".to_string(),
                expected_category: Some("CatastrophicCancellation".to_string()),
                known_reduction: Some(1e6),
            },
        ]
    }

    /// Run a single benchmark and collect metrics.
    pub fn run_benchmark(
        &self,
        trace: &ExecutionTrace,
        eag: &ErrorAmplificationGraph,
        diagnosis: &DiagnosisReport,
        repair: &RepairResult,
        benchmark_name: &str,
    ) -> BenchmarkMetrics {
        let mut metrics = BenchmarkMetrics::new(benchmark_name);
        metrics.trace_events = trace.metadata.event_count;
        metrics.coverage = trace.metadata.coverage;
        metrics.eag_nodes = eag.node_count();
        metrics.eag_edges = eag.edge_count();
        metrics.diagnosed_nodes = diagnosis.high_error_nodes;
        metrics.diagnosis_categories = diagnosis
            .category_counts
            .iter()
            .map(|(cat, count)| (cat.code().to_string(), *count))
            .collect();
        metrics.repairs_applied = repair.applied_repairs.len();
        metrics.error_reduction = repair.overall_reduction;
        metrics.fully_certified = repair.fully_certified;
        metrics.diagnosis_time_ms = diagnosis.diagnosis_time_ms;
        metrics.repair_time_ms = repair.repair_time_ms;
        metrics
    }

    /// List available benchmarks.
    pub fn list_benchmarks(&self) -> &[Benchmark] {
        &self.benchmarks
    }

    /// Generate a comparison table (CSV format) from multiple metrics.
    pub fn comparison_csv(metrics: &[BenchmarkMetrics]) -> String {
        let mut csv = String::new();
        csv.push_str(
            "benchmark,eag_nodes,eag_edges,diagnosed,repairs,reduction,certified,total_ms\n",
        );
        for m in metrics {
            csv.push_str(&format!(
                "{},{},{},{},{},{:.1},{},{}\n",
                m.name,
                m.eag_nodes,
                m.eag_edges,
                m.diagnosed_nodes,
                m.repairs_applied,
                m.error_reduction,
                m.fully_certified,
                m.total_time_ms,
            ));
        }
        csv
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_benchmarks_exist() {
        let harness = EvalHarness::new(PenumbraConfig::default());
        assert!(!harness.list_benchmarks().is_empty());
    }
}
