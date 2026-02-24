//! Verification statistics for LITMUS∞ frontend.
//!
//! Provides verification-level statistics, execution tracking,
//! model coverage, benchmarking, hypothesis testing, and reporting.

use std::collections::{HashMap, BTreeMap};
use std::fmt;
use std::time::{Duration, Instant};
use serde::{Serialize, Deserialize};

// ---------------------------------------------------------------------------
// Test Result and Model Result
// ---------------------------------------------------------------------------

/// Result of a single test verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    pub name: String,
    pub model: String,
    pub pass: bool,
    pub executions: usize,
    pub consistent: usize,
    pub inconsistent: usize,
    pub elapsed_ms: u64,
    pub outcomes_observed: usize,
    pub forbidden_count: usize,
}

impl TestResult {
    pub fn new(name: &str, model: &str) -> Self {
        Self {
            name: name.to_string(),
            model: model.to_string(),
            pass: false,
            executions: 0,
            consistent: 0,
            inconsistent: 0,
            elapsed_ms: 0,
            outcomes_observed: 0,
            forbidden_count: 0,
        }
    }

    pub fn consistency_rate(&self) -> f64 {
        if self.executions == 0 { return 0.0; }
        self.consistent as f64 / self.executions as f64
    }
}

impl fmt::Display for TestResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({}) [{}] {}/{} consistent ({}ms)",
            self.name, self.model,
            if self.pass { "PASS" } else { "FAIL" },
            self.consistent, self.executions, self.elapsed_ms)
    }
}

/// Aggregate result for a memory model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelResult {
    pub model_name: String,
    pub tests_run: usize,
    pub tests_passed: usize,
    pub total_executions: usize,
    pub total_elapsed_ms: u64,
}

impl ModelResult {
    pub fn new(name: &str) -> Self {
        Self {
            model_name: name.to_string(),
            tests_run: 0,
            tests_passed: 0,
            total_executions: 0,
            total_elapsed_ms: 0,
        }
    }

    pub fn pass_rate(&self) -> f64 {
        if self.tests_run == 0 { return 0.0; }
        self.tests_passed as f64 / self.tests_run as f64
    }

    pub fn add_test(&mut self, result: &TestResult) {
        self.tests_run += 1;
        if result.pass { self.tests_passed += 1; }
        self.total_executions += result.executions;
        self.total_elapsed_ms += result.elapsed_ms;
    }
}

impl fmt::Display for ModelResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}/{} passed ({:.1}%, {}ms)",
            self.model_name, self.tests_passed, self.tests_run,
            self.pass_rate() * 100.0, self.total_elapsed_ms)
    }
}

// ---------------------------------------------------------------------------
// Verification Statistics
// ---------------------------------------------------------------------------

/// Aggregate verification statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationStatistics {
    pub per_test_results: BTreeMap<String, TestResult>,
    pub per_model_results: BTreeMap<String, ModelResult>,
    pub start_time_ms: u64,
    pub end_time_ms: u64,
}

impl VerificationStatistics {
    pub fn new() -> Self {
        Self {
            per_test_results: BTreeMap::new(),
            per_model_results: BTreeMap::new(),
            start_time_ms: 0,
            end_time_ms: 0,
        }
    }

    /// Add a test result.
    pub fn add_result(&mut self, result: TestResult) {
        let key = format!("{}@{}", result.name, result.model);
        let model = result.model.clone();

        self.per_model_results
            .entry(model.clone())
            .or_insert_with(|| ModelResult::new(&model))
            .add_test(&result);

        self.per_test_results.insert(key, result);
    }

    /// Aggregate pass rate across all tests.
    pub fn aggregate_pass_rate(&self) -> f64 {
        let total = self.per_test_results.len();
        if total == 0 { return 0.0; }
        let passed = self.per_test_results.values().filter(|r| r.pass).count();
        passed as f64 / total as f64
    }

    /// Total number of tests.
    pub fn total_tests(&self) -> usize {
        self.per_test_results.len()
    }

    /// Total number of models.
    pub fn total_models(&self) -> usize {
        self.per_model_results.len()
    }

    /// Total executions across all tests.
    pub fn total_executions(&self) -> usize {
        self.per_test_results.values().map(|r| r.executions).sum()
    }

    /// Total elapsed time.
    pub fn total_elapsed_ms(&self) -> u64 {
        self.per_test_results.values().map(|r| r.elapsed_ms).sum()
    }

    /// Get failing tests.
    pub fn failing_tests(&self) -> Vec<&TestResult> {
        self.per_test_results.values().filter(|r| !r.pass).collect()
    }

    /// Get passing tests.
    pub fn passing_tests(&self) -> Vec<&TestResult> {
        self.per_test_results.values().filter(|r| r.pass).collect()
    }
}

impl Default for VerificationStatistics {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for VerificationStatistics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Verification Statistics ===")?;
        writeln!(f, "Tests: {} ({} passed, {:.1}% pass rate)",
            self.total_tests(),
            self.passing_tests().len(),
            self.aggregate_pass_rate() * 100.0)?;
        writeln!(f, "Models: {}", self.total_models())?;
        writeln!(f, "Total executions: {}", self.total_executions())?;
        writeln!(f, "Total time: {}ms", self.total_elapsed_ms())?;
        for mr in self.per_model_results.values() {
            writeln!(f, "  {}", mr)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Execution Counter
// ---------------------------------------------------------------------------

/// Tracks execution counts and outcome frequencies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionCounter {
    pub total: usize,
    pub consistent: usize,
    pub inconsistent: usize,
    pub per_outcome: HashMap<String, usize>,
}

impl ExecutionCounter {
    pub fn new() -> Self {
        Self {
            total: 0,
            consistent: 0,
            inconsistent: 0,
            per_outcome: HashMap::new(),
        }
    }

    /// Record an execution.
    pub fn add_execution(&mut self, outcome: &str, is_consistent: bool) {
        self.total += 1;
        if is_consistent {
            self.consistent += 1;
        } else {
            self.inconsistent += 1;
        }
        *self.per_outcome.entry(outcome.to_string()).or_insert(0) += 1;
    }

    /// Merge another counter into this one.
    pub fn merge(&mut self, other: &ExecutionCounter) {
        self.total += other.total;
        self.consistent += other.consistent;
        self.inconsistent += other.inconsistent;
        for (k, v) in &other.per_outcome {
            *self.per_outcome.entry(k.clone()).or_insert(0) += v;
        }
    }

    /// Reset counters.
    pub fn reset(&mut self) {
        self.total = 0;
        self.consistent = 0;
        self.inconsistent = 0;
        self.per_outcome.clear();
    }

    /// Consistency rate.
    pub fn consistency_rate(&self) -> f64 {
        if self.total == 0 { return 0.0; }
        self.consistent as f64 / self.total as f64
    }

    /// Number of unique outcomes.
    pub fn unique_outcomes(&self) -> usize {
        self.per_outcome.len()
    }

    /// Most common outcome.
    pub fn most_common_outcome(&self) -> Option<(&str, usize)> {
        self.per_outcome.iter()
            .max_by_key(|(_, &v)| v)
            .map(|(k, &v)| (k.as_str(), v))
    }
}

impl Default for ExecutionCounter {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for ExecutionCounter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Executions: {} total ({} consistent, {} inconsistent, {:.1}%)",
            self.total, self.consistent, self.inconsistent,
            self.consistency_rate() * 100.0)
    }
}

// ---------------------------------------------------------------------------
// Batch Execution Tracker
// ---------------------------------------------------------------------------

/// Result of a single run in a batch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunResult {
    pub run_id: usize,
    pub test_count: usize,
    pub pass_count: usize,
    pub fail_count: usize,
    pub elapsed_ms: u64,
}

impl RunResult {
    pub fn pass_rate(&self) -> f64 {
        if self.test_count == 0 { return 0.0; }
        self.pass_count as f64 / self.test_count as f64
    }
}

impl fmt::Display for RunResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Run {}: {}/{} passed ({}ms)",
            self.run_id, self.pass_count, self.test_count, self.elapsed_ms)
    }
}

/// Tracks multiple verification runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchExecutionTracker {
    pub runs: Vec<RunResult>,
}

impl BatchExecutionTracker {
    pub fn new() -> Self {
        Self { runs: Vec::new() }
    }

    pub fn add_run(&mut self, result: RunResult) {
        self.runs.push(result);
    }

    pub fn total_runs(&self) -> usize {
        self.runs.len()
    }

    pub fn total_tests(&self) -> usize {
        self.runs.iter().map(|r| r.test_count).sum()
    }

    pub fn total_passes(&self) -> usize {
        self.runs.iter().map(|r| r.pass_count).sum()
    }

    pub fn average_pass_rate(&self) -> f64 {
        if self.runs.is_empty() { return 0.0; }
        let sum: f64 = self.runs.iter().map(|r| r.pass_rate()).sum();
        sum / self.runs.len() as f64
    }

    pub fn total_elapsed_ms(&self) -> u64 {
        self.runs.iter().map(|r| r.elapsed_ms).sum()
    }

    pub fn summary(&self) -> String {
        format!("{} runs, {} total tests, {:.1}% avg pass rate, {}ms total",
            self.total_runs(), self.total_tests(),
            self.average_pass_rate() * 100.0, self.total_elapsed_ms())
    }
}

impl Default for BatchExecutionTracker {
    fn default() -> Self { Self::new() }
}

// ---------------------------------------------------------------------------
// Model Coverage Statistics
// ---------------------------------------------------------------------------

/// Coverage statistics for memory model verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCoverageStats {
    pub axioms_exercised: HashMap<String, usize>,
    pub orderings_tested: HashMap<String, usize>,
    pub scopes_covered: HashMap<String, usize>,
    pub total_axioms_available: usize,
    pub total_orderings_available: usize,
}

impl ModelCoverageStats {
    pub fn new() -> Self {
        Self {
            axioms_exercised: HashMap::new(),
            orderings_tested: HashMap::new(),
            scopes_covered: HashMap::new(),
            total_axioms_available: 0,
            total_orderings_available: 0,
        }
    }

    /// Record an axiom being exercised.
    pub fn add_axiom(&mut self, axiom: &str) {
        *self.axioms_exercised.entry(axiom.to_string()).or_insert(0) += 1;
    }

    /// Record an ordering being tested.
    pub fn add_ordering(&mut self, ordering: &str) {
        *self.orderings_tested.entry(ordering.to_string()).or_insert(0) += 1;
    }

    /// Record a scope being covered.
    pub fn add_scope(&mut self, scope: &str) {
        *self.scopes_covered.entry(scope.to_string()).or_insert(0) += 1;
    }

    /// Axiom coverage percentage.
    pub fn axiom_coverage(&self) -> f64 {
        if self.total_axioms_available == 0 { return 0.0; }
        self.axioms_exercised.len() as f64 / self.total_axioms_available as f64
    }

    /// Ordering coverage percentage.
    pub fn ordering_coverage(&self) -> f64 {
        if self.total_orderings_available == 0 { return 0.0; }
        self.orderings_tested.len() as f64 / self.total_orderings_available as f64
    }
}

impl Default for ModelCoverageStats {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for ModelCoverageStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Coverage Statistics:")?;
        writeln!(f, "  Axioms: {}/{} ({:.1}%)",
            self.axioms_exercised.len(), self.total_axioms_available,
            self.axiom_coverage() * 100.0)?;
        writeln!(f, "  Orderings: {}/{} ({:.1}%)",
            self.orderings_tested.len(), self.total_orderings_available,
            self.ordering_coverage() * 100.0)?;
        writeln!(f, "  Scopes: {}", self.scopes_covered.len())?;
        Ok(())
    }
}

/// Coverage matrix: model × test → covered.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageMatrix {
    /// Rows are models, columns are tests.
    pub models: Vec<String>,
    pub tests: Vec<String>,
    pub covered: Vec<Vec<bool>>,
}

impl CoverageMatrix {
    pub fn new(models: Vec<String>, tests: Vec<String>) -> Self {
        let rows = models.len();
        let cols = tests.len();
        Self {
            models,
            tests,
            covered: vec![vec![false; cols]; rows],
        }
    }

    pub fn set_covered(&mut self, model_idx: usize, test_idx: usize) {
        if model_idx < self.covered.len() && test_idx < self.covered[model_idx].len() {
            self.covered[model_idx][test_idx] = true;
        }
    }

    pub fn is_covered(&self, model_idx: usize, test_idx: usize) -> bool {
        self.covered.get(model_idx)
            .and_then(|row| row.get(test_idx))
            .copied()
            .unwrap_or(false)
    }

    pub fn coverage_percentage(&self) -> f64 {
        let total = self.models.len() * self.tests.len();
        if total == 0 { return 0.0; }
        let covered: usize = self.covered.iter().flat_map(|r| r.iter()).filter(|&&c| c).count();
        covered as f64 / total as f64
    }

    pub fn model_coverage(&self, model_idx: usize) -> f64 {
        if model_idx >= self.covered.len() || self.tests.is_empty() { return 0.0; }
        let covered = self.covered[model_idx].iter().filter(|&&c| c).count();
        covered as f64 / self.tests.len() as f64
    }
}

impl fmt::Display for CoverageMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Coverage Matrix ({:.1}% total)", self.coverage_percentage() * 100.0)?;
        for (i, model) in self.models.iter().enumerate() {
            writeln!(f, "  {}: {:.1}% coverage", model, self.model_coverage(i) * 100.0)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Benchmarking
// ---------------------------------------------------------------------------

/// Result of a benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub name: String,
    pub iterations: usize,
    pub total_time_ms: u64,
    pub mean_time_ms: f64,
    pub std_dev_ms: f64,
    pub min_time_ms: f64,
    pub max_time_ms: f64,
    pub throughput_ops_sec: f64,
    pub peak_memory_bytes: usize,
}

impl BenchmarkResult {
    pub fn from_times(name: &str, times_ms: &[f64]) -> Self {
        let n = times_ms.len();
        let sum: f64 = times_ms.iter().sum();
        let mean = if n > 0 { sum / n as f64 } else { 0.0 };
        let variance = if n > 1 {
            times_ms.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / (n - 1) as f64
        } else { 0.0 };
        let min = times_ms.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = times_ms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let throughput = if mean > 0.0 { 1000.0 / mean } else { 0.0 };

        Self {
            name: name.to_string(),
            iterations: n,
            total_time_ms: sum as u64,
            mean_time_ms: mean,
            std_dev_ms: variance.sqrt(),
            min_time_ms: if min.is_finite() { min } else { 0.0 },
            max_time_ms: if max.is_finite() { max } else { 0.0 },
            throughput_ops_sec: throughput,
            peak_memory_bytes: 0,
        }
    }
}

impl fmt::Display for BenchmarkResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {:.2}ms ± {:.2}ms ({} iterations, {:.1} ops/sec)",
            self.name, self.mean_time_ms, self.std_dev_ms,
            self.iterations, self.throughput_ops_sec)
    }
}

/// Suite of benchmark results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkSuite {
    pub results: Vec<BenchmarkResult>,
}

impl BenchmarkSuite {
    pub fn new() -> Self {
        Self { results: Vec::new() }
    }

    pub fn add_result(&mut self, result: BenchmarkResult) {
        self.results.push(result);
    }

    pub fn best(&self) -> Option<&BenchmarkResult> {
        self.results.iter().min_by(|a, b|
            a.mean_time_ms.partial_cmp(&b.mean_time_ms).unwrap_or(std::cmp::Ordering::Equal)
        )
    }

    pub fn worst(&self) -> Option<&BenchmarkResult> {
        self.results.iter().max_by(|a, b|
            a.mean_time_ms.partial_cmp(&b.mean_time_ms).unwrap_or(std::cmp::Ordering::Equal)
        )
    }

    pub fn summary(&self) -> String {
        let mut s = format!("Benchmark Suite ({} benchmarks)\n", self.results.len());
        for r in &self.results {
            s.push_str(&format!("  {}\n", r));
        }
        s
    }
}

impl Default for BenchmarkSuite {
    fn default() -> Self { Self::new() }
}

/// Performance regression detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRegression {
    pub baseline: BenchmarkResult,
    pub current: BenchmarkResult,
    pub threshold: f64,
}

impl PerformanceRegression {
    pub fn new(baseline: BenchmarkResult, current: BenchmarkResult, threshold: f64) -> Self {
        Self { baseline, current, threshold }
    }

    /// Relative slowdown (positive = slower).
    pub fn relative_change(&self) -> f64 {
        if self.baseline.mean_time_ms <= 0.0 { return 0.0; }
        (self.current.mean_time_ms - self.baseline.mean_time_ms) / self.baseline.mean_time_ms
    }

    /// Whether this constitutes a regression.
    pub fn is_regression(&self) -> bool {
        self.relative_change() > self.threshold
    }

    /// Whether this is an improvement.
    pub fn is_improvement(&self) -> bool {
        self.relative_change() < -self.threshold
    }
}

impl fmt::Display for PerformanceRegression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let change = self.relative_change() * 100.0;
        let status = if self.is_regression() { "REGRESSION" }
            else if self.is_improvement() { "IMPROVEMENT" }
            else { "STABLE" };
        write!(f, "{}: {:.2}ms -> {:.2}ms ({:+.1}%, {})",
            self.current.name, self.baseline.mean_time_ms,
            self.current.mean_time_ms, change, status)
    }
}

// ---------------------------------------------------------------------------
// Hypothesis Testing
// ---------------------------------------------------------------------------

/// Chi-squared test for outcome distributions.
#[derive(Debug, Clone)]
pub struct ChiSquaredTest {
    pub observed: Vec<f64>,
    pub expected: Vec<f64>,
}

impl ChiSquaredTest {
    pub fn new(observed: Vec<f64>, expected: Vec<f64>) -> Self {
        assert_eq!(observed.len(), expected.len());
        Self { observed, expected }
    }

    /// Compute the chi-squared statistic.
    pub fn statistic(&self) -> f64 {
        self.observed.iter().zip(&self.expected)
            .map(|(o, e)| {
                if *e == 0.0 { return 0.0; }
                (o - e).powi(2) / e
            })
            .sum()
    }

    /// Degrees of freedom.
    pub fn degrees_of_freedom(&self) -> usize {
        self.observed.len().saturating_sub(1)
    }

    /// Approximate p-value using chi-squared distribution.
    pub fn p_value(&self) -> f64 {
        let chi2 = self.statistic();
        let df = self.degrees_of_freedom() as f64;
        if df <= 0.0 { return 1.0; }
        // Rough approximation using normal approximation for large df.
        let z = (2.0 * chi2).sqrt() - (2.0 * df - 1.0).sqrt();
        // Simple approximate CDF of standard normal.
        0.5 * (1.0 - erf_approx(z / std::f64::consts::SQRT_2))
    }

    /// Check significance at the given alpha level.
    pub fn is_significant(&self, alpha: f64) -> bool {
        self.p_value() < alpha
    }
}

/// Simple t-test for comparing two sample means.
#[derive(Debug, Clone)]
pub struct TTest {
    pub sample1: Vec<f64>,
    pub sample2: Vec<f64>,
}

impl TTest {
    pub fn new(sample1: Vec<f64>, sample2: Vec<f64>) -> Self {
        Self { sample1, sample2 }
    }

    fn mean(data: &[f64]) -> f64 {
        if data.is_empty() { return 0.0; }
        data.iter().sum::<f64>() / data.len() as f64
    }

    fn variance(data: &[f64]) -> f64 {
        if data.len() < 2 { return 0.0; }
        let m = Self::mean(data);
        data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (data.len() - 1) as f64
    }

    /// Compute the t-statistic (Welch's t-test).
    pub fn t_statistic(&self) -> f64 {
        let m1 = Self::mean(&self.sample1);
        let m2 = Self::mean(&self.sample2);
        let v1 = Self::variance(&self.sample1);
        let v2 = Self::variance(&self.sample2);
        let n1 = self.sample1.len() as f64;
        let n2 = self.sample2.len() as f64;

        let se = (v1 / n1 + v2 / n2).sqrt();
        if se == 0.0 { return 0.0; }
        (m1 - m2) / se
    }

    /// Approximate p-value (two-tailed).
    pub fn p_value_approx(&self) -> f64 {
        let t = self.t_statistic().abs();
        // Rough approximation using standard normal.
        2.0 * 0.5 * (1.0 - erf_approx(t / std::f64::consts::SQRT_2))
    }

    /// Check significance.
    pub fn is_significant(&self, alpha: f64) -> bool {
        self.p_value_approx() < alpha
    }
}

/// Confidence interval.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceInterval {
    pub mean: f64,
    pub margin: f64,
    pub confidence_level: f64,
}

impl ConfidenceInterval {
    /// Construct a 95% confidence interval from sample data.
    pub fn from_sample(data: &[f64]) -> Self {
        let n = data.len() as f64;
        if n < 2.0 {
            return Self { mean: TTest::mean(data), margin: 0.0, confidence_level: 0.95 };
        }
        let mean = TTest::mean(data);
        let std_dev = TTest::variance(data).sqrt();
        let margin = 1.96 * std_dev / n.sqrt(); // z=1.96 for 95% CI.
        Self { mean, margin, confidence_level: 0.95 }
    }

    pub fn lower(&self) -> f64 { self.mean - self.margin }
    pub fn upper(&self) -> f64 { self.mean + self.margin }

    pub fn contains(&self, value: f64) -> bool {
        value >= self.lower() && value <= self.upper()
    }
}

impl fmt::Display for ConfidenceInterval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.4} ± {:.4} ({:.0}% CI: [{:.4}, {:.4}])",
            self.mean, self.margin, self.confidence_level * 100.0,
            self.lower(), self.upper())
    }
}

/// Approximate error function for p-value computation.
fn erf_approx(x: f64) -> f64 {
    // Abramowitz and Stegun approximation.
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

// ---------------------------------------------------------------------------
// Statistics Report
// ---------------------------------------------------------------------------

/// Generates formatted statistics reports.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticsReport {
    pub title: String,
    pub sections: Vec<ReportSection>,
}

/// A section within a report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportSection {
    pub heading: String,
    pub content: String,
}

impl StatisticsReport {
    pub fn new(title: &str) -> Self {
        Self {
            title: title.to_string(),
            sections: Vec::new(),
        }
    }

    pub fn add_section(&mut self, heading: &str, content: &str) {
        self.sections.push(ReportSection {
            heading: heading.to_string(),
            content: content.to_string(),
        });
    }

    /// Generate from verification statistics.
    pub fn from_verification_stats(stats: &VerificationStatistics) -> Self {
        let mut report = Self::new("Verification Report");
        report.add_section("Summary", &format!(
            "Tests: {}, Pass rate: {:.1}%, Executions: {}, Time: {}ms",
            stats.total_tests(),
            stats.aggregate_pass_rate() * 100.0,
            stats.total_executions(),
            stats.total_elapsed_ms(),
        ));

        let mut models_text = String::new();
        for mr in stats.per_model_results.values() {
            models_text.push_str(&format!("  {}\n", mr));
        }
        report.add_section("Models", &models_text);

        let failing = stats.failing_tests();
        if !failing.is_empty() {
            let mut fail_text = String::new();
            for tr in &failing {
                fail_text.push_str(&format!("  {}\n", tr));
            }
            report.add_section("Failing Tests", &fail_text);
        }

        report
    }

    /// Generate from benchmarks.
    pub fn from_benchmarks(suite: &BenchmarkSuite) -> Self {
        let mut report = Self::new("Benchmark Report");
        report.add_section("Summary", &suite.summary());
        if let Some(best) = suite.best() {
            report.add_section("Best", &format!("{}", best));
        }
        if let Some(worst) = suite.worst() {
            report.add_section("Worst", &format!("{}", worst));
        }
        report
    }

    /// Render as text.
    pub fn to_text(&self) -> String {
        let mut s = format!("=== {} ===\n", self.title);
        for section in &self.sections {
            s.push_str(&format!("\n--- {} ---\n{}\n", section.heading, section.content));
        }
        s
    }

    /// Render as JSON string.
    pub fn to_json_string(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }
}

// ---------------------------------------------------------------------------
// Metrics Collector
// ---------------------------------------------------------------------------

/// Type of metric.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MetricType {
    Counter,
    Gauge,
    HistogramMetric,
    TimerMetric,
}

/// A metric value.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricValue {
    Counter(u64),
    Gauge(f64),
    Histogram(Vec<f64>),
    Timer(Vec<u64>),
}

impl MetricValue {
    pub fn as_counter(&self) -> Option<u64> {
        if let Self::Counter(v) = self { Some(*v) } else { None }
    }

    pub fn as_gauge(&self) -> Option<f64> {
        if let Self::Gauge(v) = self { Some(*v) } else { None }
    }
}

/// Registry of named metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsRegistry {
    pub metrics: HashMap<String, MetricValue>,
}

impl MetricsRegistry {
    pub fn new() -> Self {
        Self { metrics: HashMap::new() }
    }

    /// Increment a counter.
    pub fn record_counter(&mut self, name: &str, delta: u64) {
        let entry = self.metrics.entry(name.to_string())
            .or_insert(MetricValue::Counter(0));
        if let MetricValue::Counter(v) = entry {
            *v += delta;
        }
    }

    /// Set a gauge value.
    pub fn record_gauge(&mut self, name: &str, value: f64) {
        self.metrics.insert(name.to_string(), MetricValue::Gauge(value));
    }

    /// Record a duration.
    pub fn record_duration(&mut self, name: &str, duration_ms: u64) {
        let entry = self.metrics.entry(name.to_string())
            .or_insert(MetricValue::Timer(Vec::new()));
        if let MetricValue::Timer(v) = entry {
            v.push(duration_ms);
        }
    }

    /// Record a histogram value.
    pub fn record_histogram(&mut self, name: &str, value: f64) {
        let entry = self.metrics.entry(name.to_string())
            .or_insert(MetricValue::Histogram(Vec::new()));
        if let MetricValue::Histogram(v) = entry {
            v.push(value);
        }
    }

    /// Get a metric value.
    pub fn get(&self, name: &str) -> Option<&MetricValue> {
        self.metrics.get(name)
    }

    /// Number of registered metrics.
    pub fn count(&self) -> usize {
        self.metrics.len()
    }

    /// Clear all metrics.
    pub fn clear(&mut self) {
        self.metrics.clear();
    }
}

impl Default for MetricsRegistry {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for MetricsRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Metrics ({} registered):", self.metrics.len())?;
        for (name, value) in &self.metrics {
            match value {
                MetricValue::Counter(v) => writeln!(f, "  {} (counter): {}", name, v)?,
                MetricValue::Gauge(v) => writeln!(f, "  {} (gauge): {:.4}", name, v)?,
                MetricValue::Histogram(v) => writeln!(f, "  {} (histogram): {} values", name, v.len())?,
                MetricValue::Timer(v) => writeln!(f, "  {} (timer): {} recordings", name, v.len())?,
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- TestResult tests --

    #[test]
    fn test_test_result() {
        let mut tr = TestResult::new("SB", "TSO");
        tr.executions = 100;
        tr.consistent = 80;
        tr.pass = true;
        assert!((tr.consistency_rate() - 0.8).abs() < 1e-10);
    }

    // -- ModelResult tests --

    #[test]
    fn test_model_result() {
        let mut mr = ModelResult::new("TSO");
        let mut tr = TestResult::new("SB", "TSO");
        tr.pass = true;
        tr.executions = 100;
        mr.add_test(&tr);
        assert_eq!(mr.tests_run, 1);
        assert_eq!(mr.tests_passed, 1);
        assert_eq!(mr.pass_rate(), 1.0);
    }

    // -- VerificationStatistics tests --

    #[test]
    fn test_verification_stats() {
        let mut stats = VerificationStatistics::new();
        let mut tr = TestResult::new("SB", "TSO");
        tr.pass = true;
        tr.executions = 50;
        stats.add_result(tr);

        let mut tr2 = TestResult::new("MP", "TSO");
        tr2.pass = false;
        tr2.executions = 30;
        stats.add_result(tr2);

        assert_eq!(stats.total_tests(), 2);
        assert_eq!(stats.total_models(), 1);
        assert!((stats.aggregate_pass_rate() - 0.5).abs() < 1e-10);
        assert_eq!(stats.total_executions(), 80);
    }

    #[test]
    fn test_verification_stats_failing() {
        let mut stats = VerificationStatistics::new();
        let mut tr = TestResult::new("SB", "TSO");
        tr.pass = false;
        stats.add_result(tr);
        assert_eq!(stats.failing_tests().len(), 1);
    }

    // -- ExecutionCounter tests --

    #[test]
    fn test_execution_counter() {
        let mut counter = ExecutionCounter::new();
        counter.add_execution("outcome1", true);
        counter.add_execution("outcome2", false);
        counter.add_execution("outcome1", true);
        assert_eq!(counter.total, 3);
        assert_eq!(counter.consistent, 2);
        assert_eq!(counter.inconsistent, 1);
        assert_eq!(counter.unique_outcomes(), 2);
    }

    #[test]
    fn test_execution_counter_merge() {
        let mut a = ExecutionCounter::new();
        a.add_execution("x", true);
        let mut b = ExecutionCounter::new();
        b.add_execution("y", false);
        a.merge(&b);
        assert_eq!(a.total, 2);
    }

    #[test]
    fn test_execution_counter_most_common() {
        let mut counter = ExecutionCounter::new();
        counter.add_execution("a", true);
        counter.add_execution("b", true);
        counter.add_execution("a", true);
        let (name, count) = counter.most_common_outcome().unwrap();
        assert_eq!(name, "a");
        assert_eq!(count, 2);
    }

    // -- BatchExecutionTracker tests --

    #[test]
    fn test_batch_tracker() {
        let mut tracker = BatchExecutionTracker::new();
        tracker.add_run(RunResult { run_id: 0, test_count: 10, pass_count: 8, fail_count: 2, elapsed_ms: 100 });
        tracker.add_run(RunResult { run_id: 1, test_count: 10, pass_count: 10, fail_count: 0, elapsed_ms: 90 });
        assert_eq!(tracker.total_runs(), 2);
        assert_eq!(tracker.total_tests(), 20);
        assert_eq!(tracker.total_passes(), 18);
        assert!((tracker.average_pass_rate() - 0.9).abs() < 1e-10);
    }

    // -- ModelCoverageStats tests --

    #[test]
    fn test_coverage_stats() {
        let mut cov = ModelCoverageStats::new();
        cov.total_axioms_available = 5;
        cov.add_axiom("acyclicity");
        cov.add_axiom("irreflexivity");
        assert_eq!(cov.axiom_coverage(), 0.4);
    }

    #[test]
    fn test_coverage_matrix() {
        let mut matrix = CoverageMatrix::new(
            vec!["SC".to_string(), "TSO".to_string()],
            vec!["SB".to_string(), "MP".to_string()],
        );
        matrix.set_covered(0, 0);
        matrix.set_covered(0, 1);
        matrix.set_covered(1, 0);
        assert!(matrix.is_covered(0, 0));
        assert!(!matrix.is_covered(1, 1));
        assert_eq!(matrix.model_coverage(0), 1.0);
        assert_eq!(matrix.model_coverage(1), 0.5);
        assert!((matrix.coverage_percentage() - 0.75).abs() < 1e-10);
    }

    // -- BenchmarkResult tests --

    #[test]
    fn test_benchmark_result() {
        let result = BenchmarkResult::from_times("test", &[10.0, 12.0, 11.0, 13.0, 14.0]);
        assert_eq!(result.iterations, 5);
        assert!((result.mean_time_ms - 12.0).abs() < 1e-10);
        assert!(result.min_time_ms == 10.0);
        assert!(result.max_time_ms == 14.0);
    }

    // -- BenchmarkSuite tests --

    #[test]
    fn test_benchmark_suite() {
        let mut suite = BenchmarkSuite::new();
        suite.add_result(BenchmarkResult::from_times("fast", &[1.0, 1.5]));
        suite.add_result(BenchmarkResult::from_times("slow", &[10.0, 12.0]));
        assert_eq!(suite.best().unwrap().name, "fast");
        assert_eq!(suite.worst().unwrap().name, "slow");
    }

    // -- PerformanceRegression tests --

    #[test]
    fn test_regression_detection() {
        let baseline = BenchmarkResult::from_times("test", &[10.0, 10.0]);
        let current = BenchmarkResult::from_times("test", &[15.0, 15.0]);
        let reg = PerformanceRegression::new(baseline, current, 0.1);
        assert!(reg.is_regression());
        assert!((reg.relative_change() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_improvement_detection() {
        let baseline = BenchmarkResult::from_times("test", &[10.0, 10.0]);
        let current = BenchmarkResult::from_times("test", &[5.0, 5.0]);
        let reg = PerformanceRegression::new(baseline, current, 0.1);
        assert!(reg.is_improvement());
    }

    // -- ChiSquaredTest tests --

    #[test]
    fn test_chi_squared() {
        let test = ChiSquaredTest::new(
            vec![10.0, 20.0, 30.0],
            vec![20.0, 20.0, 20.0],
        );
        assert!(test.statistic() > 0.0);
        assert_eq!(test.degrees_of_freedom(), 2);
    }

    #[test]
    fn test_chi_squared_perfect() {
        let test = ChiSquaredTest::new(
            vec![10.0, 10.0, 10.0],
            vec![10.0, 10.0, 10.0],
        );
        assert_eq!(test.statistic(), 0.0);
    }

    // -- TTest tests --

    #[test]
    fn test_ttest() {
        let test = TTest::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![2.0, 3.0, 4.0, 5.0, 6.0],
        );
        // Means differ by 1.0.
        let t = test.t_statistic();
        assert!(t < 0.0); // Sample1 < Sample2.
    }

    #[test]
    fn test_ttest_equal() {
        let test = TTest::new(
            vec![5.0, 5.0, 5.0],
            vec![5.0, 5.0, 5.0],
        );
        assert_eq!(test.t_statistic(), 0.0);
    }

    // -- ConfidenceInterval tests --

    #[test]
    fn test_confidence_interval() {
        let ci = ConfidenceInterval::from_sample(&[10.0, 11.0, 12.0, 9.0, 10.0]);
        assert!(ci.contains(ci.mean));
        assert!(ci.lower() < ci.upper());
    }

    // -- StatisticsReport tests --

    #[test]
    fn test_report_text() {
        let mut report = StatisticsReport::new("Test Report");
        report.add_section("Summary", "All tests passed");
        let text = report.to_text();
        assert!(text.contains("Test Report"));
        assert!(text.contains("Summary"));
    }

    #[test]
    fn test_report_json() {
        let mut report = StatisticsReport::new("Test");
        report.add_section("S", "Content");
        let json = report.to_json_string();
        assert!(json.contains("Test"));
    }

    // -- MetricsRegistry tests --

    #[test]
    fn test_metrics_counter() {
        let mut reg = MetricsRegistry::new();
        reg.record_counter("requests", 1);
        reg.record_counter("requests", 1);
        assert_eq!(reg.get("requests").unwrap().as_counter(), Some(2));
    }

    #[test]
    fn test_metrics_gauge() {
        let mut reg = MetricsRegistry::new();
        reg.record_gauge("temperature", 36.5);
        assert_eq!(reg.get("temperature").unwrap().as_gauge(), Some(36.5));
    }

    #[test]
    fn test_metrics_count() {
        let mut reg = MetricsRegistry::new();
        reg.record_counter("a", 1);
        reg.record_gauge("b", 1.0);
        assert_eq!(reg.count(), 2);
    }
}
