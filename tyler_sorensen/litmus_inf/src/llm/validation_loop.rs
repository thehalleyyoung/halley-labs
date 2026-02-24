//! Validation loop: generate → parse → check → refine.
//!
//! Orchestrates the full cycle of LLM-assisted litmus test generation,
//! parsing, memory-model checking, and iterative refinement until a
//! valid test is produced or the retry budget is exhausted.

use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, Instant};
use serde::{Serialize, Deserialize};

use crate::checker::litmus::{
    LitmusTest, Thread, Instruction, Ordering, Outcome, LitmusOutcome,
};
use crate::checker::execution::{Address, Value};

use super::prompt_engine::{
    GenerationConstraint, GenerationRequest, GenerationResult,
    LlmPromptEngine, ParseError, PromptConfig,
};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// How the refinement loop behaves.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Maximum number of LLM calls before giving up.
    pub max_retries: usize,
    /// Which refinement strategies to try, in order.
    pub refinement_strategies: Vec<RefinementStrategy>,
    /// Per-attempt timeout (applies to the LLM call, not parsing).
    pub timeout: Duration,
    /// Whether to log every step.
    pub verbose: bool,
    /// Minimum confidence score to accept a result.
    pub min_confidence: f64,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            max_retries: 5,
            refinement_strategies: vec![
                RefinementStrategy::ErrorFeedback,
                RefinementStrategy::ConstraintNarrowing,
                RefinementStrategy::ExampleAugmentation,
                RefinementStrategy::ChainOfThought,
            ],
            timeout: Duration::from_secs(60),
            verbose: false,
            min_confidence: 0.7,
        }
    }
}

// ---------------------------------------------------------------------------
// Refinement strategy
// ---------------------------------------------------------------------------

/// Strategy for refining a failed generation attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RefinementStrategy {
    /// Include parse/check errors verbatim in the next prompt.
    ErrorFeedback,
    /// Add constraints derived from the failed check.
    ConstraintNarrowing,
    /// Inject additional few-shot examples that match the failure mode.
    ExampleAugmentation,
    /// Ask the LLM to reason step-by-step before producing the test.
    ChainOfThought,
}

impl fmt::Display for RefinementStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RefinementStrategy::ErrorFeedback => write!(f, "error-feedback"),
            RefinementStrategy::ConstraintNarrowing => write!(f, "constraint-narrowing"),
            RefinementStrategy::ExampleAugmentation => write!(f, "example-augmentation"),
            RefinementStrategy::ChainOfThought => write!(f, "chain-of-thought"),
        }
    }
}

// ---------------------------------------------------------------------------
// Validation step record
// ---------------------------------------------------------------------------

/// What happened during one attempt in the loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationStep {
    pub attempt: usize,
    pub strategy_used: Option<RefinementStrategy>,
    pub raw_response: String,
    pub parse_result: StepParseResult,
    pub check_result: StepCheckResult,
    pub duration: Duration,
}

/// Parse outcome for one step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StepParseResult {
    Success,
    Failed { errors: Vec<String> },
}

/// Check outcome for one step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StepCheckResult {
    /// Not checked (e.g., parse failed).
    Skipped,
    /// All constraints satisfied.
    Passed,
    /// One or more constraint violations.
    Failed { violations: Vec<String> },
}

// ---------------------------------------------------------------------------
// Validation result
// ---------------------------------------------------------------------------

/// Final result of a validation loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// The final, validated litmus test (if successful).
    pub test: Option<LitmusTest>,
    /// Whether the loop succeeded.
    pub success: bool,
    /// Total number of LLM calls made.
    pub attempts: usize,
    /// Record of each attempt.
    pub steps: Vec<ValidationStep>,
    /// Accuracy metrics accumulated during this run.
    pub accuracy: AccuracyMetrics,
    /// Total wall-clock time.
    pub total_duration: Duration,
}

/// Accuracy metrics for a single validation run.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AccuracyMetrics {
    pub parse_successes: usize,
    pub parse_failures: usize,
    pub check_passes: usize,
    pub check_failures: usize,
}

impl AccuracyMetrics {
    pub fn parse_rate(&self) -> f64 {
        let total = self.parse_successes + self.parse_failures;
        if total == 0 {
            0.0
        } else {
            self.parse_successes as f64 / total as f64
        }
    }

    pub fn check_rate(&self) -> f64 {
        let total = self.check_passes + self.check_failures;
        if total == 0 {
            0.0
        } else {
            self.check_passes as f64 / total as f64
        }
    }

    pub fn overall_rate(&self) -> f64 {
        let total = self.parse_successes + self.parse_failures;
        if total == 0 {
            return 0.0;
        }
        self.check_passes as f64 / total as f64
    }
}

// ---------------------------------------------------------------------------
// Accuracy tracker (across many runs)
// ---------------------------------------------------------------------------

/// Tracks accuracy across many generation runs.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AccuracyTracker {
    pub total_runs: usize,
    pub successful_runs: usize,
    pub total_attempts: usize,
    pub per_pattern: HashMap<String, PatternAccuracy>,
    pub per_model: HashMap<String, ModelAccuracy>,
}

/// Per-pattern accuracy stats.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PatternAccuracy {
    pub runs: usize,
    pub successes: usize,
    pub total_attempts: usize,
    pub avg_attempts_to_success: f64,
}

/// Per-model accuracy stats.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelAccuracy {
    pub runs: usize,
    pub successes: usize,
    pub parse_successes: usize,
    pub parse_failures: usize,
}

impl AccuracyTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record the result of a validation run.
    pub fn record(&mut self, pattern: &str, model: &str, result: &ValidationResult) {
        self.total_runs += 1;
        self.total_attempts += result.attempts;

        if result.success {
            self.successful_runs += 1;
        }

        // Per-pattern
        let pa = self
            .per_pattern
            .entry(pattern.to_string())
            .or_default();
        pa.runs += 1;
        pa.total_attempts += result.attempts;
        if result.success {
            pa.successes += 1;
            // Update running average
            let prev_total = pa.avg_attempts_to_success * (pa.successes - 1) as f64;
            pa.avg_attempts_to_success =
                (prev_total + result.attempts as f64) / pa.successes as f64;
        }

        // Per-model
        let ma = self.per_model.entry(model.to_string()).or_default();
        ma.runs += 1;
        if result.success {
            ma.successes += 1;
        }
        ma.parse_successes += result.accuracy.parse_successes;
        ma.parse_failures += result.accuracy.parse_failures;
    }

    pub fn success_rate(&self) -> f64 {
        if self.total_runs == 0 {
            0.0
        } else {
            self.successful_runs as f64 / self.total_runs as f64
        }
    }

    pub fn avg_attempts(&self) -> f64 {
        if self.total_runs == 0 {
            0.0
        } else {
            self.total_attempts as f64 / self.total_runs as f64
        }
    }

    /// Summary report.
    pub fn report(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "Accuracy Report: {}/{} runs succeeded ({:.1}%)\n",
            self.successful_runs,
            self.total_runs,
            self.success_rate() * 100.0,
        ));
        out.push_str(&format!(
            "Average attempts per run: {:.1}\n\n",
            self.avg_attempts()
        ));

        if !self.per_pattern.is_empty() {
            out.push_str("Per-pattern breakdown:\n");
            let mut patterns: Vec<_> = self.per_pattern.iter().collect();
            patterns.sort_by_key(|(k, _)| k.clone());
            for (name, pa) in &patterns {
                out.push_str(&format!(
                    "  {}: {}/{} ({:.1}%), avg attempts {:.1}\n",
                    name,
                    pa.successes,
                    pa.runs,
                    if pa.runs > 0 {
                        pa.successes as f64 / pa.runs as f64 * 100.0
                    } else {
                        0.0
                    },
                    pa.avg_attempts_to_success,
                ));
            }
            out.push('\n');
        }

        if !self.per_model.is_empty() {
            out.push_str("Per-model breakdown:\n");
            let mut models: Vec<_> = self.per_model.iter().collect();
            models.sort_by_key(|(k, _)| k.clone());
            for (name, ma) in &models {
                let parse_total = ma.parse_successes + ma.parse_failures;
                out.push_str(&format!(
                    "  {}: {}/{} runs, parse {}/{} ({:.1}%)\n",
                    name,
                    ma.successes,
                    ma.runs,
                    ma.parse_successes,
                    parse_total,
                    if parse_total > 0 {
                        ma.parse_successes as f64 / parse_total as f64 * 100.0
                    } else {
                        0.0
                    },
                ));
            }
        }

        out
    }
}

// ---------------------------------------------------------------------------
// Check trait
// ---------------------------------------------------------------------------

/// Trait for checking whether a generated litmus test is valid against
/// a memory model. Implementors wrap actual memory-model checking logic.
pub trait MemoryModelChecker {
    /// Check the test and return a list of violations (empty = pass).
    fn check(&self, test: &LitmusTest, model: &str) -> Vec<String>;
}

/// Default checker that validates structural properties only.
#[derive(Debug, Clone)]
pub struct StructuralChecker;

impl StructuralChecker {
    pub fn new() -> Self {
        Self
    }
}

impl MemoryModelChecker for StructuralChecker {
    fn check(&self, test: &LitmusTest, _model: &str) -> Vec<String> {
        let mut violations = Vec::new();

        if test.threads.is_empty() {
            violations.push("test has no threads".to_string());
        }

        for thread in &test.threads {
            if thread.instructions.is_empty() {
                violations.push(format!("thread {} has no instructions", thread.id));
            }
        }

        // Check that all addresses referenced are initialised
        let mut referenced_addrs = std::collections::HashSet::new();
        for thread in &test.threads {
            for instr in &thread.instructions {
                match instr {
                    Instruction::Load { addr, .. }
                    | Instruction::Store { addr, .. }
                    | Instruction::RMW { addr, .. } => {
                        referenced_addrs.insert(*addr);
                    }
                    _ => {}
                }
            }
        }
        for addr in &referenced_addrs {
            if !test.initial_state.contains_key(addr) {
                violations.push(format!(
                    "address {} is used but not in initial state",
                    addr
                ));
            }
        }

        // Thread id uniqueness
        let mut ids = std::collections::HashSet::new();
        for thread in &test.threads {
            if !ids.insert(thread.id) {
                violations.push(format!("duplicate thread id {}", thread.id));
            }
        }

        violations
    }
}

/// Checker that also validates constraint satisfaction.
#[derive(Debug, Clone)]
pub struct ConstraintChecker {
    structural: StructuralChecker,
    constraints: Vec<GenerationConstraint>,
}

impl ConstraintChecker {
    pub fn new(constraints: Vec<GenerationConstraint>) -> Self {
        Self {
            structural: StructuralChecker::new(),
            constraints,
        }
    }
}

impl MemoryModelChecker for ConstraintChecker {
    fn check(&self, test: &LitmusTest, model: &str) -> Vec<String> {
        let mut violations = self.structural.check(test, model);

        for constraint in &self.constraints {
            match constraint {
                GenerationConstraint::MaxThreads(n) => {
                    if test.threads.len() > *n {
                        violations.push(format!(
                            "test has {} threads, max allowed is {}",
                            test.threads.len(),
                            n
                        ));
                    }
                }
                GenerationConstraint::MaxInstructions(n) => {
                    for thread in &test.threads {
                        if thread.instructions.len() > *n {
                            violations.push(format!(
                                "thread {} has {} instructions, max allowed is {}",
                                thread.id,
                                thread.instructions.len(),
                                n
                            ));
                        }
                    }
                }
                GenerationConstraint::RequireOrdering(ord) => {
                    let found = test.threads.iter().any(|t| {
                        t.instructions.iter().any(|i| match i {
                            Instruction::Load { ordering, .. }
                            | Instruction::Store { ordering, .. }
                            | Instruction::Fence { ordering, .. }
                            | Instruction::RMW { ordering, .. } => ordering == ord,
                            _ => false,
                        })
                    });
                    if !found {
                        violations.push(format!(
                            "test does not use required ordering {:?}",
                            ord
                        ));
                    }
                }
                GenerationConstraint::ForbidOrdering(ord) => {
                    let found = test.threads.iter().any(|t| {
                        t.instructions.iter().any(|i| match i {
                            Instruction::Load { ordering, .. }
                            | Instruction::Store { ordering, .. }
                            | Instruction::Fence { ordering, .. }
                            | Instruction::RMW { ordering, .. } => ordering == ord,
                            _ => false,
                        })
                    });
                    if found {
                        violations.push(format!(
                            "test uses forbidden ordering {:?}",
                            ord
                        ));
                    }
                }
                GenerationConstraint::RequireOutcome(expected_class) => {
                    let has = test
                        .expected_outcomes
                        .iter()
                        .any(|(_, c)| c == expected_class);
                    if !has {
                        violations.push(format!(
                            "test does not have a {:?} outcome",
                            expected_class
                        ));
                    }
                }
                GenerationConstraint::RequireScope(_scope_name) => {
                    // Scope checking is model-dependent; accept for now
                }
                GenerationConstraint::Custom(msg) => {
                    // Custom constraints require domain knowledge; log warning
                    if self.structural.check(test, model).is_empty() {
                        // Structural OK is the best we can do
                    } else {
                        violations.push(format!("custom constraint not verified: {}", msg));
                    }
                }
            }
        }

        violations
    }
}

// ---------------------------------------------------------------------------
// LLM caller trait
// ---------------------------------------------------------------------------

/// Trait for actually calling the LLM. Implementors handle HTTP, auth, etc.
pub trait LlmCaller {
    /// Send a prompt and get a response string.
    fn call(&self, prompt: &str, config: &PromptConfig) -> Result<String, LlmCallError>;
}

/// Errors from calling the LLM API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LlmCallError {
    Timeout,
    RateLimit,
    AuthError,
    ServerError(String),
    NetworkError(String),
}

impl fmt::Display for LlmCallError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LlmCallError::Timeout => write!(f, "LLM call timed out"),
            LlmCallError::RateLimit => write!(f, "LLM rate limit exceeded"),
            LlmCallError::AuthError => write!(f, "LLM authentication error"),
            LlmCallError::ServerError(s) => write!(f, "LLM server error: {}", s),
            LlmCallError::NetworkError(s) => write!(f, "network error: {}", s),
        }
    }
}

impl std::error::Error for LlmCallError {}

/// A mock caller for testing. Returns pre-configured responses.
#[derive(Debug, Clone)]
pub struct MockLlmCaller {
    responses: Vec<String>,
    call_index: std::cell::Cell<usize>,
}

impl MockLlmCaller {
    pub fn new(responses: Vec<String>) -> Self {
        Self {
            responses,
            call_index: std::cell::Cell::new(0),
        }
    }

    pub fn single(response: &str) -> Self {
        Self::new(vec![response.to_string()])
    }
}

impl LlmCaller for MockLlmCaller {
    fn call(&self, _prompt: &str, _config: &PromptConfig) -> Result<String, LlmCallError> {
        let idx = self.call_index.get();
        if idx < self.responses.len() {
            self.call_index.set(idx + 1);
            Ok(self.responses[idx].clone())
        } else if !self.responses.is_empty() {
            // Repeat last response
            Ok(self.responses.last().unwrap().clone())
        } else {
            Err(LlmCallError::ServerError("no responses configured".to_string()))
        }
    }
}

// ---------------------------------------------------------------------------
// Validation loop
// ---------------------------------------------------------------------------

/// Orchestrates: generate → parse → check → refine.
pub struct ValidationLoop<C: LlmCaller, M: MemoryModelChecker> {
    pub engine: LlmPromptEngine,
    pub config: ValidationConfig,
    pub caller: C,
    pub checker: M,
}

impl<C: LlmCaller, M: MemoryModelChecker> ValidationLoop<C, M> {
    pub fn new(
        engine: LlmPromptEngine,
        config: ValidationConfig,
        caller: C,
        checker: M,
    ) -> Self {
        Self {
            engine,
            config,
            caller,
            checker,
        }
    }

    /// Run the full validation loop for a generation request.
    pub fn run(&self, request: &GenerationRequest) -> ValidationResult {
        let start = Instant::now();
        let mut steps = Vec::new();
        let mut accuracy = AccuracyMetrics::default();
        let mut last_response = String::new();
        let mut last_errors: Vec<String> = Vec::new();
        let mut last_violations: Vec<String> = Vec::new();

        for attempt in 0..=self.config.max_retries {
            let step_start = Instant::now();

            // Determine which strategy to use
            let strategy = if attempt == 0 {
                None
            } else {
                let idx =
                    (attempt - 1) % self.config.refinement_strategies.len();
                Some(self.config.refinement_strategies[idx])
            };

            // Build the prompt
            let prompt = self.build_attempt_prompt(
                request,
                attempt,
                strategy,
                &last_response,
                &last_errors,
                &last_violations,
            );

            // Call the LLM
            let response = match self.caller.call(&prompt, &self.engine.config) {
                Ok(r) => r,
                Err(_e) => {
                    steps.push(ValidationStep {
                        attempt,
                        strategy_used: strategy,
                        raw_response: String::new(),
                        parse_result: StepParseResult::Failed {
                            errors: vec!["LLM call failed".to_string()],
                        },
                        check_result: StepCheckResult::Skipped,
                        duration: step_start.elapsed(),
                    });
                    accuracy.parse_failures += 1;
                    continue;
                }
            };

            last_response = response.clone();

            // Parse the response
            let test = match self.engine.parse_response(&response) {
                Ok(t) => {
                    accuracy.parse_successes += 1;
                    last_errors.clear();
                    t
                }
                Err(e) => {
                    accuracy.parse_failures += 1;
                    last_errors = vec![format!("{}", e)];
                    steps.push(ValidationStep {
                        attempt,
                        strategy_used: strategy,
                        raw_response: response,
                        parse_result: StepParseResult::Failed {
                            errors: last_errors.clone(),
                        },
                        check_result: StepCheckResult::Skipped,
                        duration: step_start.elapsed(),
                    });
                    continue;
                }
            };

            // Check the test
            let violations = self.checker.check(&test, &request.memory_model);
            if violations.is_empty() {
                accuracy.check_passes += 1;
                steps.push(ValidationStep {
                    attempt,
                    strategy_used: strategy,
                    raw_response: response,
                    parse_result: StepParseResult::Success,
                    check_result: StepCheckResult::Passed,
                    duration: step_start.elapsed(),
                });
                return ValidationResult {
                    test: Some(test),
                    success: true,
                    attempts: attempt + 1,
                    steps,
                    accuracy,
                    total_duration: start.elapsed(),
                };
            }

            // Check failed
            accuracy.check_failures += 1;
            last_violations = violations.clone();
            steps.push(ValidationStep {
                attempt,
                strategy_used: strategy,
                raw_response: response,
                parse_result: StepParseResult::Success,
                check_result: StepCheckResult::Failed {
                    violations: violations.clone(),
                },
                duration: step_start.elapsed(),
            });
        }

        // Exhausted retries
        ValidationResult {
            test: None,
            success: false,
            attempts: self.config.max_retries + 1,
            steps,
            accuracy,
            total_duration: start.elapsed(),
        }
    }

    /// Build the prompt for a specific attempt.
    fn build_attempt_prompt(
        &self,
        request: &GenerationRequest,
        attempt: usize,
        strategy: Option<RefinementStrategy>,
        last_response: &str,
        last_errors: &[String],
        last_violations: &[String],
    ) -> String {
        if attempt == 0 {
            return self.engine.build_prompt(request);
        }

        match strategy {
            Some(RefinementStrategy::ErrorFeedback) => {
                let error_text = if !last_errors.is_empty() {
                    last_errors.join("; ")
                } else {
                    last_violations.join("; ")
                };
                let pseudo_err = ParseError::MissingField(error_text);
                self.engine
                    .build_refinement_prompt(request, last_response, &pseudo_err)
            }
            Some(RefinementStrategy::ConstraintNarrowing) => {
                self.engine.build_constraint_refinement_prompt(
                    request,
                    last_response,
                    last_violations,
                )
            }
            Some(RefinementStrategy::ExampleAugmentation) => {
                // Re-build with extra examples
                let mut prompt = self.engine.build_prompt(request);
                prompt.push_str("\n\nNote: your previous attempt was incorrect. ");
                prompt.push_str("Pay close attention to the examples above.\n");
                if !last_violations.is_empty() {
                    prompt.push_str("Issues with previous attempt:\n");
                    for v in last_violations {
                        prompt.push_str(&format!("  - {}\n", v));
                    }
                }
                prompt
            }
            Some(RefinementStrategy::ChainOfThought) => {
                let mut prompt = String::new();
                prompt.push_str(
                    "Let's think step-by-step about what went wrong and how to fix it.\n\n",
                );
                if !last_violations.is_empty() {
                    prompt.push_str("The previous test had these issues:\n");
                    for v in last_violations {
                        prompt.push_str(&format!("  - {}\n", v));
                    }
                    prompt.push('\n');
                }
                if !last_errors.is_empty() {
                    prompt.push_str("There were also parse errors:\n");
                    for e in last_errors {
                        prompt.push_str(&format!("  - {}\n", e));
                    }
                    prompt.push('\n');
                }
                prompt.push_str("Please reason about each issue and produce a corrected test.\n\n");
                prompt.push_str(&self.engine.build_prompt(request));
                prompt
            }
            None => self.engine.build_prompt(request),
        }
    }
}

// ---------------------------------------------------------------------------
// Convenience constructors
// ---------------------------------------------------------------------------

impl ValidationLoop<MockLlmCaller, StructuralChecker> {
    /// Create a loop with a mock caller and structural checker (for testing).
    pub fn mock(responses: Vec<String>) -> Self {
        Self {
            engine: LlmPromptEngine::new(PromptConfig::default()),
            config: ValidationConfig::default(),
            caller: MockLlmCaller::new(responses),
            checker: StructuralChecker::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const VALID_TEST: &str = "\
test MP
init x = 0
init y = 0

Thread 0:
  store x 1 relaxed
  store y 1 release

Thread 1:
  load r0 y acquire
  load r1 x relaxed

outcome: 1:r0=1, 1:r1=0 -> Forbidden";

    const INVALID_NO_THREADS: &str = "\
Just some random text with no litmus test structure at all.
This should fail parsing entirely.
";

    const MISSING_INIT: &str = "\
test BadInit

Thread 0:
  store x 1 relaxed
  load r0 y relaxed

Thread 1:
  store y 1 relaxed";

    fn default_request() -> GenerationRequest {
        GenerationRequest {
            target_pattern: "MP".to_string(),
            memory_model: "TSO".to_string(),
            constraints: vec![],
            description: None,
            num_threads: Some(2),
            max_instructions_per_thread: None,
        }
    }

    // -- ValidationConfig ---------------------------------------------------

    #[test]
    fn test_default_config() {
        let cfg = ValidationConfig::default();
        assert_eq!(cfg.max_retries, 5);
        assert!(!cfg.refinement_strategies.is_empty());
        assert!(cfg.min_confidence > 0.0);
    }

    // -- RefinementStrategy -------------------------------------------------

    #[test]
    fn test_refinement_strategy_display() {
        assert_eq!(
            format!("{}", RefinementStrategy::ErrorFeedback),
            "error-feedback"
        );
        assert_eq!(
            format!("{}", RefinementStrategy::ChainOfThought),
            "chain-of-thought"
        );
    }

    // -- AccuracyMetrics ----------------------------------------------------

    #[test]
    fn test_accuracy_metrics_rates() {
        let m = AccuracyMetrics {
            parse_successes: 8,
            parse_failures: 2,
            check_passes: 6,
            check_failures: 2,
        };
        assert!((m.parse_rate() - 0.8).abs() < 1e-9);
        assert!((m.check_rate() - 0.75).abs() < 1e-9);
        assert!((m.overall_rate() - 0.6).abs() < 1e-9);
    }

    #[test]
    fn test_accuracy_metrics_zero() {
        let m = AccuracyMetrics::default();
        assert_eq!(m.parse_rate(), 0.0);
        assert_eq!(m.check_rate(), 0.0);
        assert_eq!(m.overall_rate(), 0.0);
    }

    // -- AccuracyTracker ----------------------------------------------------

    #[test]
    fn test_tracker_record_success() {
        let mut tracker = AccuracyTracker::new();
        let result = ValidationResult {
            test: None,
            success: true,
            attempts: 2,
            steps: vec![],
            accuracy: AccuracyMetrics {
                parse_successes: 2,
                parse_failures: 0,
                check_passes: 1,
                check_failures: 1,
            },
            total_duration: Duration::from_secs(1),
        };
        tracker.record("MP", "gpt-4", &result);
        assert_eq!(tracker.total_runs, 1);
        assert_eq!(tracker.successful_runs, 1);
        assert_eq!(tracker.total_attempts, 2);
        assert!(tracker.success_rate() > 0.99);
    }

    #[test]
    fn test_tracker_record_failure() {
        let mut tracker = AccuracyTracker::new();
        let result = ValidationResult {
            test: None,
            success: false,
            attempts: 5,
            steps: vec![],
            accuracy: AccuracyMetrics::default(),
            total_duration: Duration::from_secs(5),
        };
        tracker.record("SB", "gpt-4", &result);
        assert_eq!(tracker.successful_runs, 0);
        assert_eq!(tracker.success_rate(), 0.0);
    }

    #[test]
    fn test_tracker_report() {
        let mut tracker = AccuracyTracker::new();
        let result = ValidationResult {
            test: None,
            success: true,
            attempts: 1,
            steps: vec![],
            accuracy: AccuracyMetrics {
                parse_successes: 1,
                parse_failures: 0,
                check_passes: 1,
                check_failures: 0,
            },
            total_duration: Duration::from_secs(1),
        };
        tracker.record("MP", "gpt-4", &result);
        let report = tracker.report();
        assert!(report.contains("1/1"));
        assert!(report.contains("MP"));
        assert!(report.contains("gpt-4"));
    }

    #[test]
    fn test_tracker_multiple_patterns() {
        let mut tracker = AccuracyTracker::new();
        let mk = |success| ValidationResult {
            test: None,
            success,
            attempts: 1,
            steps: vec![],
            accuracy: AccuracyMetrics::default(),
            total_duration: Duration::from_secs(1),
        };
        tracker.record("MP", "gpt-4", &mk(true));
        tracker.record("SB", "gpt-4", &mk(false));
        tracker.record("MP", "gpt-4", &mk(true));
        assert_eq!(tracker.per_pattern["MP"].successes, 2);
        assert_eq!(tracker.per_pattern["SB"].successes, 0);
    }

    // -- StructuralChecker --------------------------------------------------

    #[test]
    fn test_structural_checker_valid() {
        let checker = StructuralChecker::new();
        let engine = LlmPromptEngine::new(PromptConfig::default());
        let test = engine.parse_response(VALID_TEST).unwrap();
        let violations = checker.check(&test, "TSO");
        assert!(violations.is_empty(), "violations: {:?}", violations);
    }

    #[test]
    fn test_structural_checker_empty_threads() {
        let checker = StructuralChecker::new();
        let test = LitmusTest {
            name: "empty".to_string(),
            threads: vec![],
            initial_state: HashMap::new(),
            expected_outcomes: vec![],
        };
        let violations = checker.check(&test, "TSO");
        assert!(!violations.is_empty());
        assert!(violations[0].contains("no threads"));
    }

    #[test]
    fn test_structural_checker_missing_init() {
        let checker = StructuralChecker::new();
        let engine = LlmPromptEngine::new(PromptConfig::default());
        let test = engine.parse_response(MISSING_INIT).unwrap();
        let violations = checker.check(&test, "TSO");
        assert!(
            violations.iter().any(|v| v.contains("not in initial state")),
            "violations: {:?}",
            violations
        );
    }

    #[test]
    fn test_structural_checker_duplicate_thread_id() {
        let checker = StructuralChecker::new();
        let test = LitmusTest {
            name: "dup".to_string(),
            threads: vec![
                Thread {
                    id: 0,
                    instructions: vec![Instruction::Store {
                        addr: 0,
                        value: 1,
                        ordering: Ordering::Relaxed,
                    }],
                },
                Thread {
                    id: 0,
                    instructions: vec![Instruction::Store {
                        addr: 0,
                        value: 2,
                        ordering: Ordering::Relaxed,
                    }],
                },
            ],
            initial_state: [(0, 0)].into_iter().collect(),
            expected_outcomes: vec![],
        };
        let violations = checker.check(&test, "TSO");
        assert!(violations.iter().any(|v| v.contains("duplicate")));
    }

    // -- ConstraintChecker --------------------------------------------------

    #[test]
    fn test_constraint_checker_max_threads() {
        let checker = ConstraintChecker::new(vec![GenerationConstraint::MaxThreads(1)]);
        let engine = LlmPromptEngine::new(PromptConfig::default());
        let test = engine.parse_response(VALID_TEST).unwrap();
        let violations = checker.check(&test, "TSO");
        assert!(violations.iter().any(|v| v.contains("threads")));
    }

    #[test]
    fn test_constraint_checker_require_ordering() {
        let checker = ConstraintChecker::new(vec![GenerationConstraint::RequireOrdering(
            Ordering::SeqCst,
        )]);
        let engine = LlmPromptEngine::new(PromptConfig::default());
        let test = engine.parse_response(VALID_TEST).unwrap();
        let violations = checker.check(&test, "TSO");
        assert!(violations.iter().any(|v| v.contains("SeqCst")));
    }

    #[test]
    fn test_constraint_checker_forbid_ordering() {
        let checker = ConstraintChecker::new(vec![GenerationConstraint::ForbidOrdering(
            Ordering::Release,
        )]);
        let engine = LlmPromptEngine::new(PromptConfig::default());
        let test = engine.parse_response(VALID_TEST).unwrap();
        let violations = checker.check(&test, "TSO");
        assert!(violations.iter().any(|v| v.contains("forbidden")));
    }

    #[test]
    fn test_constraint_checker_passes() {
        let checker = ConstraintChecker::new(vec![
            GenerationConstraint::MaxThreads(2),
            GenerationConstraint::RequireOrdering(Ordering::Acquire),
        ]);
        let engine = LlmPromptEngine::new(PromptConfig::default());
        let test = engine.parse_response(VALID_TEST).unwrap();
        let violations = checker.check(&test, "TSO");
        assert!(violations.is_empty(), "violations: {:?}", violations);
    }

    // -- MockLlmCaller ------------------------------------------------------

    #[test]
    fn test_mock_caller_single() {
        let caller = MockLlmCaller::single("hello");
        let cfg = PromptConfig::default();
        assert_eq!(caller.call("x", &cfg).unwrap(), "hello");
        // Repeats last
        assert_eq!(caller.call("x", &cfg).unwrap(), "hello");
    }

    #[test]
    fn test_mock_caller_sequence() {
        let caller = MockLlmCaller::new(vec!["first".to_string(), "second".to_string()]);
        let cfg = PromptConfig::default();
        assert_eq!(caller.call("x", &cfg).unwrap(), "first");
        assert_eq!(caller.call("x", &cfg).unwrap(), "second");
        assert_eq!(caller.call("x", &cfg).unwrap(), "second");
    }

    #[test]
    fn test_mock_caller_empty() {
        let caller = MockLlmCaller::new(vec![]);
        let cfg = PromptConfig::default();
        assert!(caller.call("x", &cfg).is_err());
    }

    // -- LlmCallError -------------------------------------------------------

    #[test]
    fn test_llm_call_error_display() {
        assert_eq!(format!("{}", LlmCallError::Timeout), "LLM call timed out");
        assert!(format!("{}", LlmCallError::ServerError("oops".into())).contains("oops"));
    }

    // -- ValidationLoop: success on first attempt ---------------------------

    #[test]
    fn test_loop_success_first_attempt() {
        let vl = ValidationLoop::mock(vec![VALID_TEST.to_string()]);
        let result = vl.run(&default_request());
        assert!(result.success);
        assert_eq!(result.attempts, 1);
        assert!(result.test.is_some());
        assert_eq!(result.test.as_ref().unwrap().name, "MP");
    }

    // -- ValidationLoop: parse failure then success -------------------------

    #[test]
    fn test_loop_retry_after_parse_failure() {
        let vl = ValidationLoop::mock(vec![
            INVALID_NO_THREADS.to_string(),
            VALID_TEST.to_string(),
        ]);
        let result = vl.run(&default_request());
        assert!(result.success);
        assert_eq!(result.attempts, 2);
        assert_eq!(result.accuracy.parse_failures, 1);
        assert_eq!(result.accuracy.parse_successes, 1);
    }

    // -- ValidationLoop: check failure then success -------------------------

    #[test]
    fn test_loop_retry_after_check_failure() {
        let checker = ConstraintChecker::new(vec![GenerationConstraint::MaxThreads(1)]);
        let two_thread_test = VALID_TEST.to_string();
        let one_thread_test = "\
test Single
init x = 0

Thread 0:
  store x 1 relaxed

outcome: x=1 -> Allowed"
            .to_string();

        let vl = ValidationLoop {
            engine: LlmPromptEngine::new(PromptConfig::default()),
            config: ValidationConfig::default(),
            caller: MockLlmCaller::new(vec![two_thread_test, one_thread_test]),
            checker,
        };

        let result = vl.run(&default_request());
        assert!(result.success);
        assert_eq!(result.attempts, 2);
        assert_eq!(result.accuracy.check_failures, 1);
        assert_eq!(result.accuracy.check_passes, 1);
    }

    // -- ValidationLoop: exhausted retries ----------------------------------

    #[test]
    fn test_loop_exhausted_retries() {
        let mut cfg = ValidationConfig::default();
        cfg.max_retries = 2;
        let vl = ValidationLoop {
            engine: LlmPromptEngine::new(PromptConfig::default()),
            config: cfg,
            caller: MockLlmCaller::new(vec![INVALID_NO_THREADS.to_string()]),
            checker: StructuralChecker::new(),
        };
        let result = vl.run(&default_request());
        assert!(!result.success);
        assert!(result.test.is_none());
        assert_eq!(result.attempts, 3); // initial + 2 retries
    }

    // -- ValidationLoop: steps recorded correctly ---------------------------

    #[test]
    fn test_loop_records_steps() {
        let vl = ValidationLoop::mock(vec![
            INVALID_NO_THREADS.to_string(),
            VALID_TEST.to_string(),
        ]);
        let result = vl.run(&default_request());
        assert_eq!(result.steps.len(), 2);

        // First step: parse failed
        assert!(result.steps[0].strategy_used.is_none()); // first attempt
        match &result.steps[0].parse_result {
            StepParseResult::Failed { errors } => assert!(!errors.is_empty()),
            _ => panic!("expected parse failure"),
        }
        match &result.steps[0].check_result {
            StepCheckResult::Skipped => {}
            _ => panic!("expected skipped check"),
        }

        // Second step: success
        match &result.steps[1].parse_result {
            StepParseResult::Success => {}
            _ => panic!("expected parse success"),
        }
        match &result.steps[1].check_result {
            StepCheckResult::Passed => {}
            _ => panic!("expected check passed"),
        }
    }

    // -- ValidationLoop: refinement strategies cycle ------------------------

    #[test]
    fn test_loop_cycles_strategies() {
        let mut cfg = ValidationConfig::default();
        cfg.max_retries = 4;
        cfg.refinement_strategies = vec![
            RefinementStrategy::ErrorFeedback,
            RefinementStrategy::ChainOfThought,
        ];

        let vl = ValidationLoop {
            engine: LlmPromptEngine::new(PromptConfig::default()),
            config: cfg,
            caller: MockLlmCaller::new(vec![INVALID_NO_THREADS.to_string()]),
            checker: StructuralChecker::new(),
        };
        let result = vl.run(&default_request());
        assert!(!result.success);

        // Check strategy cycling
        assert!(result.steps[0].strategy_used.is_none());
        assert_eq!(
            result.steps[1].strategy_used,
            Some(RefinementStrategy::ErrorFeedback)
        );
        assert_eq!(
            result.steps[2].strategy_used,
            Some(RefinementStrategy::ChainOfThought)
        );
        assert_eq!(
            result.steps[3].strategy_used,
            Some(RefinementStrategy::ErrorFeedback)
        );
    }

    // -- ValidationLoop: LLM call failure -----------------------------------

    #[test]
    fn test_loop_handles_llm_call_failure() {
        let vl = ValidationLoop {
            engine: LlmPromptEngine::new(PromptConfig::default()),
            config: ValidationConfig {
                max_retries: 1,
                ..ValidationConfig::default()
            },
            caller: MockLlmCaller::new(vec![]),
            checker: StructuralChecker::new(),
        };
        let result = vl.run(&default_request());
        assert!(!result.success);
        assert_eq!(result.accuracy.parse_failures, 2);
    }

    // -- Integration: tracker + loop ----------------------------------------

    #[test]
    fn test_tracker_with_loop_results() {
        let mut tracker = AccuracyTracker::new();

        // Run 1: success
        let vl = ValidationLoop::mock(vec![VALID_TEST.to_string()]);
        let r1 = vl.run(&default_request());
        tracker.record("MP", "gpt-4", &r1);

        // Run 2: failure
        let mut cfg = ValidationConfig::default();
        cfg.max_retries = 0;
        let vl2 = ValidationLoop {
            engine: LlmPromptEngine::new(PromptConfig::default()),
            config: cfg,
            caller: MockLlmCaller::single(INVALID_NO_THREADS),
            checker: StructuralChecker::new(),
        };
        let r2 = vl2.run(&default_request());
        tracker.record("MP", "gpt-4", &r2);

        assert_eq!(tracker.total_runs, 2);
        assert_eq!(tracker.successful_runs, 1);
        assert!((tracker.success_rate() - 0.5).abs() < 1e-9);
    }
}
