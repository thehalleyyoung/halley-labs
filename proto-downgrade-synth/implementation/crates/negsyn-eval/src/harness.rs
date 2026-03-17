//! Test harness infrastructure for managing test execution.

use crate::pipeline::{AnalysisPipeline, PipelineConfig, PipelineResult, PipelineStage};

use chrono::Utc;
use log::{debug, error, info, warn};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Configuration for a single test case.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestCase {
    pub id: String,
    pub name: String,
    pub description: String,
    pub pipeline_config: PipelineConfig,
    pub expected_output: ExpectedOutput,
    pub timeout_ms: u64,
    pub tags: Vec<String>,
    pub priority: TestPriority,
}

impl TestCase {
    pub fn new(id: impl Into<String>, name: impl Into<String>, config: PipelineConfig) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            description: String::new(),
            pipeline_config: config,
            expected_output: ExpectedOutput::default(),
            timeout_ms: 60_000,
            tags: Vec::new(),
            priority: TestPriority::Normal,
        }
    }

    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    pub fn with_timeout(mut self, ms: u64) -> Self {
        self.timeout_ms = ms;
        self
    }

    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    pub fn with_expected(mut self, expected: ExpectedOutput) -> Self {
        self.expected_output = expected;
        self
    }

    pub fn with_priority(mut self, priority: TestPriority) -> Self {
        self.priority = priority;
        self
    }
}

/// Expected output specification for a test case.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectedOutput {
    pub should_succeed: bool,
    pub min_states: Option<usize>,
    pub max_states: Option<usize>,
    pub min_paths: Option<usize>,
    pub expect_vulnerability: Option<bool>,
    pub expected_vuln_types: Vec<String>,
    pub min_coverage: Option<f64>,
    pub max_duration_ms: Option<u64>,
    pub required_stages: Vec<PipelineStage>,
    pub tolerance: ResultTolerance,
}

impl Default for ExpectedOutput {
    fn default() -> Self {
        Self {
            should_succeed: true,
            min_states: None,
            max_states: None,
            min_paths: None,
            expect_vulnerability: None,
            expected_vuln_types: Vec::new(),
            min_coverage: None,
            max_duration_ms: None,
            required_stages: Vec::new(),
            tolerance: ResultTolerance::default(),
        }
    }
}

/// Tolerance settings for comparing expected vs. actual results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultTolerance {
    pub state_count_pct: f64,
    pub path_count_pct: f64,
    pub timing_pct: f64,
    pub coverage_abs: f64,
}

impl Default for ResultTolerance {
    fn default() -> Self {
        Self {
            state_count_pct: 0.10,
            path_count_pct: 0.10,
            timing_pct: 0.50,
            coverage_abs: 0.02,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum TestPriority {
    Critical,
    High,
    Normal,
    Low,
}

/// Result of a single test case execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestCaseResult {
    pub test_id: String,
    pub test_name: String,
    pub outcome: TestOutcome,
    pub duration_ms: u64,
    pub pipeline_result: Option<PipelineResult>,
    pub comparison: ResultComparison,
    pub error_message: Option<String>,
}

impl TestCaseResult {
    pub fn passed(&self) -> bool {
        self.outcome == TestOutcome::Pass
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TestOutcome {
    Pass,
    Fail,
    Error,
    Timeout,
    Skip,
}

impl std::fmt::Display for TestOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TestOutcome::Pass => write!(f, "PASS"),
            TestOutcome::Fail => write!(f, "FAIL"),
            TestOutcome::Error => write!(f, "ERROR"),
            TestOutcome::Timeout => write!(f, "TIMEOUT"),
            TestOutcome::Skip => write!(f, "SKIP"),
        }
    }
}

/// Detailed comparison between expected and actual results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultComparison {
    pub checks: Vec<ComparisonCheck>,
    pub all_passed: bool,
}

impl ResultComparison {
    pub fn new() -> Self {
        Self {
            checks: Vec::new(),
            all_passed: true,
        }
    }

    pub fn add_check(&mut self, check: ComparisonCheck) {
        if !check.passed {
            self.all_passed = false;
        }
        self.checks.push(check);
    }

    pub fn failed_checks(&self) -> Vec<&ComparisonCheck> {
        self.checks.iter().filter(|c| !c.passed).collect()
    }
}

impl Default for ResultComparison {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonCheck {
    pub name: String,
    pub expected: String,
    pub actual: String,
    pub passed: bool,
    pub message: String,
}

/// The test runner that executes test cases.
pub struct TestRunner {
    max_parallel: usize,
    global_timeout_ms: u64,
    fail_fast: bool,
}

impl TestRunner {
    pub fn new() -> Self {
        Self {
            max_parallel: 1,
            global_timeout_ms: 600_000,
            fail_fast: false,
        }
    }

    pub fn with_parallel(mut self, n: usize) -> Self {
        self.max_parallel = n.max(1);
        self
    }

    pub fn with_global_timeout(mut self, ms: u64) -> Self {
        self.global_timeout_ms = ms;
        self
    }

    pub fn with_fail_fast(mut self, ff: bool) -> Self {
        self.fail_fast = ff;
        self
    }

    /// Run a single test case.
    pub fn run_test(&self, test: &TestCase) -> TestCaseResult {
        let start = Instant::now();
        info!("Running test: {} ({})", test.name, test.id);

        let timeout = Duration::from_millis(test.timeout_ms);

        let pipeline_result = self.execute_with_timeout(test, timeout);
        let duration = start.elapsed().as_millis() as u64;

        match pipeline_result {
            ExecutionResult::Success(result) => {
                let comparison = self.compare_result(&result, &test.expected_output);
                let outcome = if comparison.all_passed {
                    TestOutcome::Pass
                } else {
                    TestOutcome::Fail
                };

                let error_msg = if outcome == TestOutcome::Fail {
                    let failed: Vec<String> = comparison
                        .failed_checks()
                        .iter()
                        .map(|c| format!("{}: expected {}, got {}", c.name, c.expected, c.actual))
                        .collect();
                    Some(failed.join("; "))
                } else {
                    None
                };

                TestCaseResult {
                    test_id: test.id.clone(),
                    test_name: test.name.clone(),
                    outcome,
                    duration_ms: duration,
                    pipeline_result: Some(result),
                    comparison,
                    error_message: error_msg,
                }
            }
            ExecutionResult::Timeout => TestCaseResult {
                test_id: test.id.clone(),
                test_name: test.name.clone(),
                outcome: TestOutcome::Timeout,
                duration_ms: duration,
                pipeline_result: None,
                comparison: ResultComparison::new(),
                error_message: Some(format!("Test timed out after {}ms", test.timeout_ms)),
            },
            ExecutionResult::Error(e) => TestCaseResult {
                test_id: test.id.clone(),
                test_name: test.name.clone(),
                outcome: TestOutcome::Error,
                duration_ms: duration,
                pipeline_result: None,
                comparison: ResultComparison::new(),
                error_message: Some(e),
            },
        }
    }

    /// Run multiple test cases, coordinating parallel execution.
    pub fn run_all(&self, tests: &[TestCase]) -> Vec<TestCaseResult> {
        let mut sorted_tests: Vec<&TestCase> = tests.iter().collect();
        sorted_tests.sort_by(|a, b| a.priority.cmp(&b.priority));

        let global_start = Instant::now();
        let global_timeout = Duration::from_millis(self.global_timeout_ms);
        let results = Arc::new(Mutex::new(Vec::new()));
        let should_stop = Arc::new(Mutex::new(false));

        if self.max_parallel <= 1 {
            for test in &sorted_tests {
                if global_start.elapsed() > global_timeout {
                    warn!("Global timeout reached, skipping remaining tests");
                    break;
                }

                {
                    let stop = should_stop.lock().unwrap();
                    if *stop {
                        break;
                    }
                }

                let result = self.run_test(test);

                if self.fail_fast && result.outcome == TestOutcome::Fail {
                    let mut stop = should_stop.lock().unwrap();
                    *stop = true;
                }

                results.lock().unwrap().push(result);
            }
        } else {
            let chunk_size = self.max_parallel;
            for chunk in sorted_tests.chunks(chunk_size) {
                if global_start.elapsed() > global_timeout {
                    break;
                }

                {
                    let stop = should_stop.lock().unwrap();
                    if *stop {
                        break;
                    }
                }

                let chunk_results: Vec<TestCaseResult> =
                    chunk.iter().map(|test| self.run_test(test)).collect();

                for result in chunk_results {
                    if self.fail_fast && result.outcome == TestOutcome::Fail {
                        let mut stop = should_stop.lock().unwrap();
                        *stop = true;
                    }
                    results.lock().unwrap().push(result);
                }
            }
        }

        Arc::try_unwrap(results)
            .map(|mutex| mutex.into_inner().unwrap_or_default())
            .unwrap_or_else(|arc| arc.lock().unwrap().clone())
    }

    fn execute_with_timeout(
        &self,
        test: &TestCase,
        timeout: Duration,
    ) -> ExecutionResult {
        let start = Instant::now();
        let mut pipeline = AnalysisPipeline::new(test.pipeline_config.clone());

        match pipeline.run() {
            Ok(result) => {
                if start.elapsed() > timeout {
                    ExecutionResult::Timeout
                } else {
                    ExecutionResult::Success(result)
                }
            }
            Err(e) => ExecutionResult::Error(format!("{}", e)),
        }
    }

    fn compare_result(
        &self,
        result: &PipelineResult,
        expected: &ExpectedOutput,
    ) -> ResultComparison {
        let mut comparison = ResultComparison::new();

        comparison.add_check(ComparisonCheck {
            name: "pipeline_success".into(),
            expected: format!("{}", expected.should_succeed),
            actual: format!("{}", result.success),
            passed: result.success == expected.should_succeed,
            message: if result.success == expected.should_succeed {
                "Pipeline success matches expectation".into()
            } else {
                format!(
                    "Expected success={}, got success={}",
                    expected.should_succeed, result.success
                )
            },
        });

        if let Some(min) = expected.min_states {
            let tolerance_margin = (min as f64 * expected.tolerance.state_count_pct) as usize;
            let effective_min = min.saturating_sub(tolerance_margin);
            comparison.add_check(ComparisonCheck {
                name: "min_states".into(),
                expected: format!(">= {} (with tolerance: {})", min, effective_min),
                actual: format!("{}", result.states_explored),
                passed: result.states_explored >= effective_min,
                message: format!(
                    "State count {} vs minimum {}",
                    result.states_explored, effective_min
                ),
            });
        }

        if let Some(max) = expected.max_states {
            let tolerance_margin = (max as f64 * expected.tolerance.state_count_pct) as usize;
            let effective_max = max + tolerance_margin;
            comparison.add_check(ComparisonCheck {
                name: "max_states".into(),
                expected: format!("<= {} (with tolerance: {})", max, effective_max),
                actual: format!("{}", result.states_explored),
                passed: result.states_explored <= effective_max,
                message: format!(
                    "State count {} vs maximum {}",
                    result.states_explored, effective_max
                ),
            });
        }

        if let Some(min_paths) = expected.min_paths {
            let tolerance_margin = (min_paths as f64 * expected.tolerance.path_count_pct) as usize;
            let effective_min = min_paths.saturating_sub(tolerance_margin);
            comparison.add_check(ComparisonCheck {
                name: "min_paths".into(),
                expected: format!(">= {}", effective_min),
                actual: format!("{}", result.paths_explored),
                passed: result.paths_explored >= effective_min,
                message: format!(
                    "Path count {} vs minimum {}",
                    result.paths_explored, effective_min
                ),
            });
        }

        if let Some(expect_vuln) = expected.expect_vulnerability {
            let has_vuln = result.has_vulnerability();
            comparison.add_check(ComparisonCheck {
                name: "vulnerability_detection".into(),
                expected: format!("{}", expect_vuln),
                actual: format!("{}", has_vuln),
                passed: has_vuln == expect_vuln,
                message: if has_vuln == expect_vuln {
                    "Vulnerability detection matches".into()
                } else {
                    format!(
                        "Expected vulnerability={}, got={}",
                        expect_vuln, has_vuln
                    )
                },
            });
        }

        if let Some(max_dur) = expected.max_duration_ms {
            let tolerance_margin = (max_dur as f64 * expected.tolerance.timing_pct) as u64;
            let effective_max = max_dur + tolerance_margin;
            comparison.add_check(ComparisonCheck {
                name: "max_duration".into(),
                expected: format!("<= {}ms (with tolerance: {}ms)", max_dur, effective_max),
                actual: format!("{}ms", result.total_duration_ms),
                passed: result.total_duration_ms <= effective_max,
                message: format!(
                    "Duration {}ms vs limit {}ms",
                    result.total_duration_ms, effective_max
                ),
            });
        }

        for stage in &expected.required_stages {
            let stage_ok = result
                .stage_metrics
                .iter()
                .any(|m| m.stage == *stage && m.success);
            comparison.add_check(ComparisonCheck {
                name: format!("required_stage_{}", stage.name()),
                expected: "completed".into(),
                actual: if stage_ok {
                    "completed".into()
                } else {
                    "missing/failed".into()
                },
                passed: stage_ok,
                message: format!("Stage {} completion", stage.name()),
            });
        }

        comparison
    }
}

impl Default for TestRunner {
    fn default() -> Self {
        Self::new()
    }
}

enum ExecutionResult {
    Success(PipelineResult),
    Timeout,
    Error(String),
}

/// Test report aggregating results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestReport {
    pub results: Vec<TestCaseResult>,
    pub total_tests: usize,
    pub passed: usize,
    pub failed: usize,
    pub errors: usize,
    pub timeouts: usize,
    pub skipped: usize,
    pub total_duration_ms: u64,
    pub pass_rate: f64,
    pub timestamp: String,
}

impl TestReport {
    pub fn from_results(results: Vec<TestCaseResult>) -> Self {
        let total = results.len();
        let passed = results
            .iter()
            .filter(|r| r.outcome == TestOutcome::Pass)
            .count();
        let failed = results
            .iter()
            .filter(|r| r.outcome == TestOutcome::Fail)
            .count();
        let errors = results
            .iter()
            .filter(|r| r.outcome == TestOutcome::Error)
            .count();
        let timeouts = results
            .iter()
            .filter(|r| r.outcome == TestOutcome::Timeout)
            .count();
        let skipped = results
            .iter()
            .filter(|r| r.outcome == TestOutcome::Skip)
            .count();
        let total_duration: u64 = results.iter().map(|r| r.duration_ms).sum();
        let pass_rate = if total > 0 {
            passed as f64 / total as f64
        } else {
            0.0
        };

        Self {
            results,
            total_tests: total,
            passed,
            failed,
            errors,
            timeouts,
            skipped,
            total_duration_ms: total_duration,
            pass_rate,
            timestamp: Utc::now().to_rfc3339(),
        }
    }

    pub fn all_passed(&self) -> bool {
        self.failed == 0 && self.errors == 0 && self.timeouts == 0
    }

    pub fn summary(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!(
            "Test Report: {} total, {} passed, {} failed, {} errors, {} timeouts, {} skipped",
            self.total_tests, self.passed, self.failed, self.errors, self.timeouts, self.skipped
        ));
        lines.push(format!(
            "Pass rate: {:.1}%, Total duration: {}ms",
            self.pass_rate * 100.0,
            self.total_duration_ms
        ));

        for result in &self.results {
            let status = format!("[{}]", result.outcome);
            let detail = result
                .error_message
                .as_ref()
                .map(|e| format!(" - {}", e))
                .unwrap_or_default();
            lines.push(format!(
                "  {} {} ({}ms){}",
                status, result.test_name, result.duration_ms, detail
            ));
        }

        lines.join("\n")
    }

    /// Get results filtered by outcome.
    pub fn by_outcome(&self, outcome: TestOutcome) -> Vec<&TestCaseResult> {
        self.results.iter().filter(|r| r.outcome == outcome).collect()
    }
}

/// The top-level test harness managing everything.
pub struct TestHarness {
    runner: TestRunner,
    test_cases: Vec<TestCase>,
}

impl TestHarness {
    pub fn new() -> Self {
        Self {
            runner: TestRunner::new(),
            test_cases: Vec::new(),
        }
    }

    pub fn with_runner(mut self, runner: TestRunner) -> Self {
        self.runner = runner;
        self
    }

    pub fn add_test(&mut self, test: TestCase) {
        self.test_cases.push(test);
    }

    pub fn add_tests(&mut self, tests: impl IntoIterator<Item = TestCase>) {
        self.test_cases.extend(tests);
    }

    pub fn test_count(&self) -> usize {
        self.test_cases.len()
    }

    pub fn tests_by_tag(&self, tag: &str) -> Vec<&TestCase> {
        self.test_cases
            .iter()
            .filter(|t| t.tags.iter().any(|tg| tg == tag))
            .collect()
    }

    /// Run all test cases and produce a report.
    pub fn run(&self) -> TestReport {
        info!("Running test harness with {} tests", self.test_cases.len());
        let results = self.runner.run_all(&self.test_cases);
        TestReport::from_results(results)
    }

    /// Run only tests matching a specific tag.
    pub fn run_tagged(&self, tag: &str) -> TestReport {
        let tests: Vec<TestCase> = self
            .test_cases
            .iter()
            .filter(|t| t.tags.iter().any(|tg| tg == tag))
            .cloned()
            .collect();
        info!("Running {} tests with tag '{}'", tests.len(), tag);
        let results = self.runner.run_all(&tests);
        TestReport::from_results(results)
    }

    /// Generate standard test cases for common scenarios.
    pub fn generate_standard_tests(&mut self) {
        let basic_config = PipelineConfig::default();
        let basic = TestCase::new("basic-pipeline", "Basic Pipeline Test", basic_config.clone())
            .with_description("Verify basic pipeline completes successfully")
            .with_tag("smoke")
            .with_expected(ExpectedOutput {
                should_succeed: true,
                min_states: Some(1),
                ..Default::default()
            });
        self.add_test(basic);

        let mut small_config = PipelineConfig::default();
        small_config.library_name = "small-test".into();
        small_config.cipher_suites = vec![0x002F, 0x0035];
        let small = TestCase::new("small-cipher", "Small Cipher Set", small_config)
            .with_description("Test with minimal cipher suite set")
            .with_tag("regression")
            .with_expected(ExpectedOutput {
                should_succeed: true,
                ..Default::default()
            });
        self.add_test(small);

        let mut large_config = PipelineConfig::default();
        large_config.library_name = "large-test".into();
        large_config.cipher_suites = (0x0030..0x0050).collect();
        large_config.max_paths = 200_000;
        let large = TestCase::new("large-cipher", "Large Cipher Set", large_config)
            .with_description("Test scalability with many cipher suites")
            .with_tag("scalability")
            .with_timeout(120_000)
            .with_expected(ExpectedOutput {
                should_succeed: true,
                min_states: Some(5),
                ..Default::default()
            });
        self.add_test(large);

        let mut timeout_config = PipelineConfig::default();
        timeout_config.library_name = "timeout-test".into();
        timeout_config.timeout_total_ms = 1;
        let timeout_test = TestCase::new("timeout-handling", "Timeout Test", timeout_config)
            .with_description("Verify graceful timeout handling")
            .with_tag("error-handling")
            .with_timeout(5_000)
            .with_expected(ExpectedOutput {
                should_succeed: false,
                ..Default::default()
            });
        self.add_test(timeout_test);
    }
}

impl Default for TestHarness {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_test_case_builder() {
        let config = PipelineConfig::default();
        let tc = TestCase::new("t1", "Test One", config)
            .with_description("A test")
            .with_timeout(5000)
            .with_tag("smoke")
            .with_priority(TestPriority::High);

        assert_eq!(tc.id, "t1");
        assert_eq!(tc.name, "Test One");
        assert_eq!(tc.timeout_ms, 5000);
        assert_eq!(tc.tags, vec!["smoke"]);
        assert_eq!(tc.priority, TestPriority::High);
    }

    #[test]
    fn test_runner_basic() {
        let runner = TestRunner::new();
        let config = PipelineConfig::default();
        let tc = TestCase::new("basic", "Basic", config).with_expected(ExpectedOutput {
            should_succeed: true,
            ..Default::default()
        });

        let result = runner.run_test(&tc);
        assert_eq!(result.outcome, TestOutcome::Pass);
        assert!(result.passed());
    }

    #[test]
    fn test_runner_multiple() {
        let runner = TestRunner::new();
        let tests: Vec<TestCase> = (0..3)
            .map(|i| {
                let mut config = PipelineConfig::default();
                config.library_name = format!("lib_{}", i);
                TestCase::new(format!("t{}", i), format!("Test {}", i), config)
                    .with_expected(ExpectedOutput {
                        should_succeed: true,
                        ..Default::default()
                    })
            })
            .collect();

        let results = runner.run_all(&tests);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_comparison_checks() {
        let mut comp = ResultComparison::new();
        comp.add_check(ComparisonCheck {
            name: "check1".into(),
            expected: "true".into(),
            actual: "true".into(),
            passed: true,
            message: "ok".into(),
        });
        assert!(comp.all_passed);

        comp.add_check(ComparisonCheck {
            name: "check2".into(),
            expected: "10".into(),
            actual: "5".into(),
            passed: false,
            message: "too few".into(),
        });
        assert!(!comp.all_passed);
        assert_eq!(comp.failed_checks().len(), 1);
    }

    #[test]
    fn test_report_from_results() {
        let results = vec![
            TestCaseResult {
                test_id: "t1".into(),
                test_name: "Test 1".into(),
                outcome: TestOutcome::Pass,
                duration_ms: 100,
                pipeline_result: None,
                comparison: ResultComparison::new(),
                error_message: None,
            },
            TestCaseResult {
                test_id: "t2".into(),
                test_name: "Test 2".into(),
                outcome: TestOutcome::Fail,
                duration_ms: 200,
                pipeline_result: None,
                comparison: ResultComparison::new(),
                error_message: Some("mismatch".into()),
            },
        ];

        let report = TestReport::from_results(results);
        assert_eq!(report.total_tests, 2);
        assert_eq!(report.passed, 1);
        assert_eq!(report.failed, 1);
        assert!((report.pass_rate - 0.5).abs() < 0.01);
        assert!(!report.all_passed());
    }

    #[test]
    fn test_report_summary_output() {
        let results = vec![TestCaseResult {
            test_id: "t1".into(),
            test_name: "Test One".into(),
            outcome: TestOutcome::Pass,
            duration_ms: 50,
            pipeline_result: None,
            comparison: ResultComparison::new(),
            error_message: None,
        }];
        let report = TestReport::from_results(results);
        let summary = report.summary();
        assert!(summary.contains("1 passed"));
        assert!(summary.contains("Test One"));
    }

    #[test]
    fn test_report_by_outcome() {
        let results = vec![
            TestCaseResult {
                test_id: "t1".into(),
                test_name: "P".into(),
                outcome: TestOutcome::Pass,
                duration_ms: 10,
                pipeline_result: None,
                comparison: ResultComparison::new(),
                error_message: None,
            },
            TestCaseResult {
                test_id: "t2".into(),
                test_name: "F".into(),
                outcome: TestOutcome::Fail,
                duration_ms: 10,
                pipeline_result: None,
                comparison: ResultComparison::new(),
                error_message: None,
            },
        ];
        let report = TestReport::from_results(results);
        assert_eq!(report.by_outcome(TestOutcome::Pass).len(), 1);
        assert_eq!(report.by_outcome(TestOutcome::Fail).len(), 1);
        assert_eq!(report.by_outcome(TestOutcome::Error).len(), 0);
    }

    #[test]
    fn test_harness_standard_tests() {
        let mut harness = TestHarness::new();
        harness.generate_standard_tests();
        assert!(harness.test_count() >= 4);
    }

    #[test]
    fn test_harness_tagged_run() {
        let mut harness = TestHarness::new();
        let config = PipelineConfig::default();
        harness.add_test(
            TestCase::new("t1", "Smoke Test", config.clone())
                .with_tag("smoke")
                .with_expected(ExpectedOutput {
                    should_succeed: true,
                    ..Default::default()
                }),
        );
        harness.add_test(
            TestCase::new("t2", "Regression", config)
                .with_tag("regression")
                .with_expected(ExpectedOutput {
                    should_succeed: true,
                    ..Default::default()
                }),
        );

        let smoke_tests = harness.tests_by_tag("smoke");
        assert_eq!(smoke_tests.len(), 1);
    }

    #[test]
    fn test_test_outcome_display() {
        assert_eq!(format!("{}", TestOutcome::Pass), "PASS");
        assert_eq!(format!("{}", TestOutcome::Timeout), "TIMEOUT");
    }

    #[test]
    fn test_priority_ordering() {
        assert!(TestPriority::Critical < TestPriority::High);
        assert!(TestPriority::High < TestPriority::Normal);
        assert!(TestPriority::Normal < TestPriority::Low);
    }

    #[test]
    fn test_harness_run() {
        let mut harness = TestHarness::new();
        let config = PipelineConfig::default();
        harness.add_test(
            TestCase::new("run-test", "Run Test", config).with_expected(ExpectedOutput {
                should_succeed: true,
                ..Default::default()
            }),
        );
        let report = harness.run();
        assert_eq!(report.total_tests, 1);
        assert!(report.all_passed());
    }
}
