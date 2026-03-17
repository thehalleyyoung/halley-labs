//! Tool comparison infrastructure for evaluating multiple leakage analysis tools.
//!
//! Provides types for defining tool profiles, collecting baseline results,
//! and generating comparison reports across different analysis approaches.

use std::collections::HashMap;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::benchmark::{BenchmarkCategory, BenchmarkResult};

/// Profile describing a leakage analysis tool being evaluated.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolProfile {
    /// Name of the tool (e.g., "CacheAudit", "Abacus", "CacheQL").
    pub name: String,
    /// Version string.
    pub version: String,
    /// Brief description of the tool's approach.
    pub approach: String,
    /// Whether the tool provides sound over-approximations.
    pub is_sound: bool,
    /// Whether the tool supports speculative execution modeling.
    pub supports_speculation: bool,
    /// Supported cache replacement policies.
    pub supported_policies: Vec<String>,
}

impl ToolProfile {
    /// Create a new tool profile.
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            approach: String::new(),
            is_sound: false,
            supports_speculation: false,
            supported_policies: Vec::new(),
        }
    }

    /// Set the approach description.
    pub fn with_approach(mut self, approach: impl Into<String>) -> Self {
        self.approach = approach.into();
        self
    }

    /// Mark whether the tool is sound.
    pub fn with_soundness(mut self, sound: bool) -> Self {
        self.is_sound = sound;
        self
    }

    /// Mark whether the tool supports speculation.
    pub fn with_speculation_support(mut self, supported: bool) -> Self {
        self.supports_speculation = supported;
        self
    }
}

/// Baseline result from a reference tool or ground truth.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineResult {
    /// Name of the benchmark.
    pub benchmark_name: String,
    /// Name of the baseline tool (or "ground-truth").
    pub tool_name: String,
    /// Number of leaking cache sets identified.
    pub leaking_sets: u32,
    /// Total leakage bound in bits.
    pub leakage_bits: f64,
    /// Analysis time.
    pub elapsed: Duration,
    /// Whether the result is considered ground truth.
    pub is_ground_truth: bool,
}

impl BaselineResult {
    /// Create a ground-truth baseline.
    pub fn ground_truth(benchmark_name: impl Into<String>, leaking_sets: u32, leakage_bits: f64) -> Self {
        Self {
            benchmark_name: benchmark_name.into(),
            tool_name: "ground-truth".into(),
            leaking_sets,
            leakage_bits,
            elapsed: Duration::ZERO,
            is_ground_truth: true,
        }
    }

    /// Create a baseline from another tool's result.
    pub fn from_tool(tool_name: impl Into<String>, result: &BenchmarkResult) -> Self {
        Self {
            benchmark_name: result.benchmark_name.clone(),
            tool_name: tool_name.into(),
            leaking_sets: result.reported_leaking_sets,
            leakage_bits: result.reported_leakage_bits,
            elapsed: result.elapsed,
            is_ground_truth: false,
        }
    }
}

/// A comparison report across multiple tools and benchmarks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonReport {
    /// Tool profiles being compared.
    pub tools: Vec<ToolProfile>,
    /// Per-tool, per-benchmark results keyed by (tool_name, benchmark_name).
    pub results: HashMap<String, Vec<BenchmarkResult>>,
    /// Baseline results for comparison.
    pub baselines: Vec<BaselineResult>,
    /// Summary statistics per tool.
    pub summaries: HashMap<String, ToolSummary>,
}

/// Aggregate statistics for a single tool across all benchmarks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSummary {
    /// Total benchmarks attempted.
    pub total: usize,
    /// Benchmarks completed successfully.
    pub successes: usize,
    /// Benchmarks that timed out.
    pub timeouts: usize,
    /// Benchmarks that failed with errors.
    pub failures: usize,
    /// Mean analysis time across successful runs.
    pub mean_time: Duration,
    /// Mean leakage bits reported.
    pub mean_leakage_bits: f64,
}

impl ComparisonReport {
    /// Create an empty comparison report.
    pub fn new() -> Self {
        Self {
            tools: Vec::new(),
            results: HashMap::new(),
            baselines: Vec::new(),
            summaries: HashMap::new(),
        }
    }

    /// Number of tools being compared.
    pub fn tool_count(&self) -> usize {
        self.tools.len()
    }

    /// Retrieve results for a specific tool.
    pub fn results_for(&self, tool_name: &str) -> Option<&[BenchmarkResult]> {
        self.results.get(tool_name).map(|v| v.as_slice())
    }
}

impl Default for ComparisonReport {
    fn default() -> Self {
        Self::new()
    }
}

/// Engine for comparing multiple leakage analysis tools on the same benchmark suite.
#[derive(Debug)]
pub struct ToolComparator {
    /// Registered tool profiles.
    tools: Vec<ToolProfile>,
    /// Collected results per tool.
    results: HashMap<String, Vec<BenchmarkResult>>,
    /// Baseline (ground-truth) results.
    baselines: Vec<BaselineResult>,
}

impl ToolComparator {
    /// Create a new comparator.
    pub fn new() -> Self {
        Self {
            tools: Vec::new(),
            results: HashMap::new(),
            baselines: Vec::new(),
        }
    }

    /// Register a tool for comparison.
    pub fn add_tool(&mut self, profile: ToolProfile) {
        self.results.entry(profile.name.clone()).or_default();
        self.tools.push(profile);
    }

    /// Add a baseline result for comparison.
    pub fn add_baseline(&mut self, baseline: BaselineResult) {
        self.baselines.push(baseline);
    }

    /// Record benchmark results for a specific tool.
    pub fn record_results(&mut self, tool_name: &str, results: Vec<BenchmarkResult>) {
        self.results
            .entry(tool_name.to_string())
            .or_default()
            .extend(results);
    }

    /// Generate a comparison report from all collected data.
    pub fn generate_report(&self) -> ComparisonReport {
        let mut summaries = HashMap::new();
        for (tool_name, results) in &self.results {
            let total = results.len();
            let successes = results.iter().filter(|r| r.success).count();
            let timeouts = results.iter().filter(|r| r.timed_out).count();
            let failures = total - successes;

            let successful: Vec<&BenchmarkResult> = results.iter().filter(|r| r.success).collect();
            let mean_time = if successful.is_empty() {
                Duration::ZERO
            } else {
                let total_nanos: u128 = successful.iter().map(|r| r.elapsed.as_nanos()).sum();
                Duration::from_nanos((total_nanos / successful.len() as u128) as u64)
            };
            let mean_leakage_bits = if successful.is_empty() {
                0.0
            } else {
                successful.iter().map(|r| r.reported_leakage_bits).sum::<f64>() / successful.len() as f64
            };

            summaries.insert(
                tool_name.clone(),
                ToolSummary {
                    total,
                    successes,
                    timeouts,
                    failures,
                    mean_time,
                    mean_leakage_bits,
                },
            );
        }

        ComparisonReport {
            tools: self.tools.clone(),
            results: self.results.clone(),
            baselines: self.baselines.clone(),
            summaries,
        }
    }
}

impl Default for ToolComparator {
    fn default() -> Self {
        Self::new()
    }
}
