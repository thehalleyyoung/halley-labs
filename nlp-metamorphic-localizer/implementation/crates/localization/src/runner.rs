//! Test suite runner that orchestrates the complete fault localization workflow.
//!
//! Manages the end-to-end flow: input generation → transformation application →
//! pipeline execution → differential computation → localization → reporting.

use crate::engine::{
    LocalizationConfig, LocalizationEngine, TestObservation,
};
use crate::{LocalizationResult, FaultClassification, CausalVerdict};
use shared_types::error::{LocalizerError, Result};
use shared_types::ir::IntermediateRepresentation;
use shared_types::types::StageId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

// ── Test case definition ────────────────────────────────────────────────────

/// A metamorphic test case to be executed against a pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetamorphicTestCase {
    pub id: String,
    pub input_text: String,
    pub transformation_name: String,
    pub expected_relation: String,
    pub seed_source: Option<String>,
    pub applicable_transformations: Vec<String>,
    pub tags: Vec<String>,
}

/// A batch of test cases for execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestSuite {
    pub name: String,
    pub description: String,
    pub test_cases: Vec<MetamorphicTestCase>,
    pub pipeline_name: String,
    pub stage_names: Vec<String>,
    pub created_at: String,
}

impl TestSuite {
    pub fn new(name: impl Into<String>, pipeline: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            test_cases: Vec::new(),
            pipeline_name: pipeline.into(),
            stage_names: Vec::new(),
            created_at: chrono::Utc::now().to_rfc3339(),
        }
    }

    pub fn add_case(&mut self, case: MetamorphicTestCase) {
        self.test_cases.push(case);
    }

    pub fn with_stages(mut self, stages: Vec<String>) -> Self {
        self.stage_names = stages;
        self
    }

    pub fn case_count(&self) -> usize {
        self.test_cases.len()
    }

    /// Filter test cases by transformation name.
    pub fn filter_by_transformation(&self, name: &str) -> Vec<&MetamorphicTestCase> {
        self.test_cases
            .iter()
            .filter(|tc| tc.transformation_name == name)
            .collect()
    }

    /// Filter test cases by tag.
    pub fn filter_by_tag(&self, tag: &str) -> Vec<&MetamorphicTestCase> {
        self.test_cases
            .iter()
            .filter(|tc| tc.tags.contains(&tag.to_string()))
            .collect()
    }
}

// ── Pipeline adapter trait ──────────────────────────────────────────────────

/// Abstract pipeline adapter that allows running pipeline stages and capturing IRs.
pub trait PipelineAdapter: Send + Sync {
    /// Get the names of all pipeline stages.
    fn get_stages(&self) -> Vec<String>;

    /// Run the full pipeline on input text and return per-stage IRs.
    fn run_full(&self, input: &str) -> Result<Vec<(String, IntermediateRepresentation)>>;

    /// Run the pipeline prefix through stage k and return the IR.
    fn run_prefix(&self, input: &str, stage_index: usize) -> Result<IntermediateRepresentation>;

    /// Run the pipeline from stage k onward with the given IR.
    fn run_from(
        &self,
        ir: &IntermediateRepresentation,
        stage_index: usize,
    ) -> Result<Vec<(String, IntermediateRepresentation)>>;

    /// Compute distance between two IRs at a given stage.
    fn compute_distance(
        &self,
        ir1: &IntermediateRepresentation,
        ir2: &IntermediateRepresentation,
        stage_name: &str,
    ) -> Result<f64>;
}

/// Transformation adapter trait for applying transformations.
pub trait TransformationAdapter: Send + Sync {
    /// Apply a named transformation to input text.
    fn apply(&self, transformation_name: &str, input: &str) -> Result<String>;

    /// Check if a transformation is applicable to the given input.
    fn is_applicable(&self, transformation_name: &str, input: &str) -> bool;

    /// Get list of available transformations.
    fn available_transformations(&self) -> Vec<String>;
}

// ── Suite runner ────────────────────────────────────────────────────────────

/// Configuration for the suite runner.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunnerConfig {
    pub localization_config: LocalizationConfig,
    pub max_concurrent_tests: usize,
    pub timeout_per_test: Duration,
    pub continue_on_error: bool,
    pub collect_irs: bool,
    pub verbose: bool,
}

impl Default for RunnerConfig {
    fn default() -> Self {
        Self {
            localization_config: LocalizationConfig::default(),
            max_concurrent_tests: 4,
            timeout_per_test: Duration::from_secs(10),
            continue_on_error: true,
            collect_irs: false,
            verbose: false,
        }
    }
}

/// Progress callback for suite execution.
pub type ProgressCallback = Box<dyn Fn(usize, usize, &str) + Send + Sync>;

/// Result of running a single test case.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestCaseResult {
    pub test_id: String,
    pub transformation_name: String,
    pub input_text: String,
    pub transformed_text: String,
    pub violation_detected: bool,
    pub violation_magnitude: f64,
    pub per_stage_differentials: HashMap<String, f64>,
    pub execution_time: Duration,
    pub error: Option<String>,
}

impl TestCaseResult {
    /// Convert to a TestObservation for the localization engine.
    pub fn to_observation(&self) -> TestObservation {
        TestObservation {
            test_id: self.test_id.clone(),
            transformation_name: self.transformation_name.clone(),
            input_text: self.input_text.clone(),
            transformed_text: self.transformed_text.clone(),
            violation_detected: self.violation_detected,
            violation_magnitude: self.violation_magnitude,
            per_stage_differentials: self.per_stage_differentials.clone(),
            execution_time_ms: self.execution_time.as_secs_f64() * 1000.0,
        }
    }
}

/// Complete results from running an entire test suite.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuiteRunResult {
    pub suite_name: String,
    pub test_results: Vec<TestCaseResult>,
    pub total_tests: usize,
    pub passed_tests: usize,
    pub failed_tests: usize,
    pub error_tests: usize,
    pub total_time: Duration,
    pub localization: Option<LocalizationResult>,
}

impl SuiteRunResult {
    /// Get the violation rate.
    pub fn violation_rate(&self) -> f64 {
        if self.total_tests == 0 {
            return 0.0;
        }
        self.failed_tests as f64 / self.total_tests as f64
    }

    /// Get test cases that detected violations.
    pub fn violations(&self) -> Vec<&TestCaseResult> {
        self.test_results
            .iter()
            .filter(|r| r.violation_detected)
            .collect()
    }

    /// Get test cases that errored.
    pub fn errors(&self) -> Vec<&TestCaseResult> {
        self.test_results
            .iter()
            .filter(|r| r.error.is_some())
            .collect()
    }

    /// Group results by transformation.
    pub fn by_transformation(&self) -> HashMap<String, Vec<&TestCaseResult>> {
        let mut groups: HashMap<String, Vec<&TestCaseResult>> = HashMap::new();
        for result in &self.test_results {
            groups
                .entry(result.transformation_name.clone())
                .or_default()
                .push(result);
        }
        groups
    }
}

/// The main suite runner that orchestrates test execution and localization.
pub struct SuiteRunner<P: PipelineAdapter, T: TransformationAdapter> {
    pipeline: P,
    transformer: T,
    config: RunnerConfig,
}

impl<P: PipelineAdapter, T: TransformationAdapter> SuiteRunner<P, T> {
    pub fn new(pipeline: P, transformer: T) -> Self {
        Self {
            pipeline,
            transformer,
            config: RunnerConfig::default(),
        }
    }

    pub fn with_config(pipeline: P, transformer: T, config: RunnerConfig) -> Self {
        Self {
            pipeline,
            transformer,
            config,
        }
    }

    /// Run a complete test suite and produce localization results.
    pub fn run_suite(&self, suite: &TestSuite) -> Result<SuiteRunResult> {
        let start = Instant::now();
        let stages = self.pipeline.get_stages();

        let mut engine = LocalizationEngine::with_config(self.config.localization_config.clone());
        engine.register_stages(
            stages
                .iter()
                .map(|name| (StageId::new(name), name.clone()))
                .collect(),
        );

        let mut test_results = Vec::with_capacity(suite.test_cases.len());
        let mut passed = 0usize;
        let mut failed = 0usize;
        let mut errored = 0usize;

        for (i, test_case) in suite.test_cases.iter().enumerate() {
            let result = self.run_single_test(test_case, &stages);

            match &result {
                Ok(res) => {
                    if res.violation_detected {
                        failed += 1;
                    } else {
                        passed += 1;
                    }
                    engine.record_observation(res.to_observation());
                    test_results.push(res.clone());
                }
                Err(e) => {
                    if self.config.continue_on_error {
                        errored += 1;
                        test_results.push(TestCaseResult {
                            test_id: test_case.id.clone(),
                            transformation_name: test_case.transformation_name.clone(),
                            input_text: test_case.input_text.clone(),
                            transformed_text: String::new(),
                            violation_detected: false,
                            violation_magnitude: 0.0,
                            per_stage_differentials: HashMap::new(),
                            execution_time: Duration::ZERO,
                            error: Some(e.to_string()),
                        });
                    } else {
                        return Err(LocalizerError::pipeline(
                            "suite_runner",
                            format!("test case {} failed: {}", test_case.id, e),
                        ));
                    }
                }
            }
        }

        let localization = if failed > 0 {
            Some(engine.run_analysis()?)
        } else {
            None
        };

        Ok(SuiteRunResult {
            suite_name: suite.name.clone(),
            test_results,
            total_tests: suite.test_cases.len(),
            passed_tests: passed,
            failed_tests: failed,
            error_tests: errored,
            total_time: start.elapsed(),
            localization,
        })
    }

    /// Run a single metamorphic test case.
    fn run_single_test(
        &self,
        test_case: &MetamorphicTestCase,
        stages: &[String],
    ) -> Result<TestCaseResult> {
        let start = Instant::now();

        // Apply transformation.
        let transformed = self
            .transformer
            .apply(&test_case.transformation_name, &test_case.input_text)?;

        // Run both original and transformed through the pipeline.
        let original_irs = self.pipeline.run_full(&test_case.input_text)?;
        let transformed_irs = self.pipeline.run_full(&transformed)?;

        // Compute per-stage differentials.
        let mut per_stage_diffs = HashMap::new();
        let mut max_diff = 0.0f64;

        for (stage_name, orig_ir) in &original_irs {
            if let Some((_, trans_ir)) = transformed_irs.iter().find(|(n, _)| n == stage_name) {
                let diff = self
                    .pipeline
                    .compute_distance(orig_ir, trans_ir, stage_name)?;
                per_stage_diffs.insert(stage_name.clone(), diff);
                max_diff = max_diff.max(diff);
            }
        }

        // Determine if there's a metamorphic violation.
        // A violation occurs when the differential exceeds what's expected
        // for a meaning-preserving transformation.
        let violation_detected = max_diff > 0.5;
        let violation_magnitude = if violation_detected { max_diff } else { 0.0 };

        Ok(TestCaseResult {
            test_id: test_case.id.clone(),
            transformation_name: test_case.transformation_name.clone(),
            input_text: test_case.input_text.clone(),
            transformed_text: transformed,
            violation_detected,
            violation_magnitude,
            per_stage_differentials: per_stage_diffs,
            execution_time: start.elapsed(),
            error: None,
        })
    }
}

// ── Coverage analysis ───────────────────────────────────────────────────────

/// Analyze transformation coverage across test cases.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageReport {
    pub total_test_cases: usize,
    pub transformation_counts: HashMap<String, usize>,
    pub tag_counts: HashMap<String, usize>,
    pub missing_transformations: Vec<String>,
    pub coverage_ratio: f64,
}

/// Compute a coverage report for a test suite.
pub fn compute_coverage(
    suite: &TestSuite,
    all_transformations: &[String],
) -> CoverageReport {
    let mut transform_counts: HashMap<String, usize> = HashMap::new();
    let mut tag_counts: HashMap<String, usize> = HashMap::new();

    for tc in &suite.test_cases {
        *transform_counts
            .entry(tc.transformation_name.clone())
            .or_insert(0) += 1;
        for tag in &tc.tags {
            *tag_counts.entry(tag.clone()).or_insert(0) += 1;
        }
    }

    let covered: usize = all_transformations
        .iter()
        .filter(|t| transform_counts.contains_key(*t))
        .count();

    let missing: Vec<String> = all_transformations
        .iter()
        .filter(|t| !transform_counts.contains_key(*t))
        .cloned()
        .collect();

    let coverage_ratio = if all_transformations.is_empty() {
        1.0
    } else {
        covered as f64 / all_transformations.len() as f64
    };

    CoverageReport {
        total_test_cases: suite.test_cases.len(),
        transformation_counts: transform_counts,
        tag_counts,
        missing_transformations: missing,
        coverage_ratio,
    }
}

// ── Test case builder ───────────────────────────────────────────────────────

/// Builder for creating MetamorphicTestCase instances conveniently.
pub struct TestCaseBuilder {
    id_counter: usize,
    prefix: String,
}

impl TestCaseBuilder {
    pub fn new(prefix: impl Into<String>) -> Self {
        Self {
            id_counter: 0,
            prefix: prefix.into(),
        }
    }

    /// Create a test case with the next sequential ID.
    pub fn build(
        &mut self,
        input: impl Into<String>,
        transformation: impl Into<String>,
    ) -> MetamorphicTestCase {
        self.id_counter += 1;
        MetamorphicTestCase {
            id: format!("{}_{:04}", self.prefix, self.id_counter),
            input_text: input.into(),
            transformation_name: transformation.into(),
            expected_relation: "semantic_equivalence".to_string(),
            seed_source: None,
            applicable_transformations: Vec::new(),
            tags: Vec::new(),
        }
    }

    /// Create a tagged test case.
    pub fn build_tagged(
        &mut self,
        input: impl Into<String>,
        transformation: impl Into<String>,
        tags: Vec<String>,
    ) -> MetamorphicTestCase {
        let mut tc = self.build(input, transformation);
        tc.tags = tags;
        tc
    }

    /// Create a batch of test cases from input-transformation pairs.
    pub fn build_batch(
        &mut self,
        pairs: Vec<(String, String)>,
    ) -> Vec<MetamorphicTestCase> {
        pairs
            .into_iter()
            .map(|(input, transform)| self.build(input, transform))
            .collect()
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_suite_creation() {
        let mut suite = TestSuite::new("basic_suite", "spacy_pipeline");
        suite.stage_names = vec!["tokenizer".into(), "tagger".into(), "parser".into()];

        let mut builder = TestCaseBuilder::new("basic");
        suite.add_case(builder.build("The cat sat on the mat.", "passivization"));
        suite.add_case(builder.build("John gave Mary the book.", "dative_alternation"));

        assert_eq!(suite.case_count(), 2);
        assert_eq!(suite.filter_by_transformation("passivization").len(), 1);
    }

    #[test]
    fn test_coverage_report() {
        let mut suite = TestSuite::new("test", "pipe");
        let mut builder = TestCaseBuilder::new("cov");
        suite.add_case(builder.build("x", "passivization"));
        suite.add_case(builder.build("y", "passivization"));
        suite.add_case(builder.build("z", "clefting"));

        let all = vec![
            "passivization".to_string(),
            "clefting".to_string(),
            "topicalization".to_string(),
        ];
        let report = compute_coverage(&suite, &all);

        assert_eq!(report.total_test_cases, 3);
        assert_eq!(*report.transformation_counts.get("passivization").unwrap(), 2);
        assert_eq!(report.missing_transformations, vec!["topicalization"]);
        assert!((report.coverage_ratio - 2.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_builder_batch() {
        let mut builder = TestCaseBuilder::new("batch");
        let cases = builder.build_batch(vec![
            ("a".to_string(), "passivization".to_string()),
            ("b".to_string(), "clefting".to_string()),
            ("c".to_string(), "topicalization".to_string()),
        ]);
        assert_eq!(cases.len(), 3);
        assert_eq!(cases[0].id, "batch_0001");
        assert_eq!(cases[2].id, "batch_0003");
    }

    #[test]
    fn test_suite_filter_by_tag() {
        let mut suite = TestSuite::new("tag_test", "pipe");
        let mut builder = TestCaseBuilder::new("tag");
        suite.add_case(builder.build_tagged("a", "pass", vec!["rare".to_string()]));
        suite.add_case(builder.build_tagged("b", "cleft", vec!["common".to_string()]));
        suite.add_case(builder.build_tagged(
            "c",
            "pass",
            vec!["rare".to_string(), "complex".to_string()],
        ));

        assert_eq!(suite.filter_by_tag("rare").len(), 2);
        assert_eq!(suite.filter_by_tag("complex").len(), 1);
        assert_eq!(suite.filter_by_tag("missing").len(), 0);
    }

    #[test]
    fn test_suite_run_result_statistics() {
        let result = SuiteRunResult {
            suite_name: "test".to_string(),
            test_results: vec![
                TestCaseResult {
                    test_id: "t1".to_string(),
                    transformation_name: "pass".to_string(),
                    input_text: "a".to_string(),
                    transformed_text: "b".to_string(),
                    violation_detected: true,
                    violation_magnitude: 0.8,
                    per_stage_differentials: HashMap::new(),
                    execution_time: Duration::from_millis(100),
                    error: None,
                },
                TestCaseResult {
                    test_id: "t2".to_string(),
                    transformation_name: "cleft".to_string(),
                    input_text: "c".to_string(),
                    transformed_text: "d".to_string(),
                    violation_detected: false,
                    violation_magnitude: 0.0,
                    per_stage_differentials: HashMap::new(),
                    execution_time: Duration::from_millis(50),
                    error: None,
                },
                TestCaseResult {
                    test_id: "t3".to_string(),
                    transformation_name: "pass".to_string(),
                    input_text: "e".to_string(),
                    transformed_text: "f".to_string(),
                    violation_detected: true,
                    violation_magnitude: 0.6,
                    per_stage_differentials: HashMap::new(),
                    execution_time: Duration::from_millis(80),
                    error: None,
                },
            ],
            total_tests: 3,
            passed_tests: 1,
            failed_tests: 2,
            error_tests: 0,
            total_time: Duration::from_millis(230),
            localization: None,
        };

        assert!((result.violation_rate() - 2.0 / 3.0).abs() < 0.01);
        assert_eq!(result.violations().len(), 2);
        assert_eq!(result.by_transformation().len(), 2);
        assert_eq!(result.by_transformation().get("pass").unwrap().len(), 2);
    }
}
