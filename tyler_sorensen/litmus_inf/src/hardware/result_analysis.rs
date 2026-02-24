//! Analyze hardware litmus-test results against model predictions.
//!
//! Provides statistical tests, consistency checks, and validation reports
//! comparing observed GPU behaviour with axiomatic model outcomes.

use std::collections::HashMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::checker::litmus::{LitmusOutcome, LitmusTest, Outcome};

use super::litmus_runner::{
    GpuBackend, HardwareResult, ObservedOutcome, OutcomeClassification, TestOutcome,
};

// ---------------------------------------------------------------------------
// Outcome histogram
// ---------------------------------------------------------------------------

/// Frequency counts of each observed outcome with optional labels.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutcomeHistogram {
    /// Outcome → count.
    pub counts: HashMap<String, u64>,
    /// Total observations.
    pub total: u64,
}

impl OutcomeHistogram {
    pub fn new() -> Self {
        Self {
            counts: HashMap::new(),
            total: 0,
        }
    }

    /// Build from a `TestOutcome`.
    pub fn from_test_outcome(outcome: &TestOutcome) -> Self {
        let mut hist = Self::new();
        for (obs, &count) in &outcome.histogram {
            let key = obs.display_key();
            *hist.counts.entry(key).or_insert(0) += count;
            hist.total += count;
        }
        hist
    }

    /// Record a single observation.
    pub fn record(&mut self, key: impl Into<String>, count: u64) {
        *self.counts.entry(key.into()).or_insert(0) += count;
        self.total += count;
    }

    /// Fraction for a given key.
    pub fn fraction(&self, key: &str) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        self.counts.get(key).copied().unwrap_or(0) as f64 / self.total as f64
    }

    /// Number of distinct outcomes.
    pub fn distinct(&self) -> usize {
        self.counts.len()
    }

    /// Sorted entries (descending by count).
    pub fn sorted(&self) -> Vec<(&str, u64)> {
        let mut v: Vec<_> = self.counts.iter().map(|(k, &c)| (k.as_str(), c)).collect();
        v.sort_by(|a, b| b.1.cmp(&a.1));
        v
    }

    /// Shannon entropy of the distribution (bits).
    pub fn entropy(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        let mut h = 0.0f64;
        for &count in self.counts.values() {
            if count > 0 {
                let p = count as f64 / self.total as f64;
                h -= p * p.log2();
            }
        }
        h
    }

    /// Merge another histogram.
    pub fn merge(&mut self, other: &OutcomeHistogram) {
        for (key, &count) in &other.counts {
            *self.counts.entry(key.clone()).or_insert(0) += count;
        }
        self.total += other.total;
    }
}

impl Default for OutcomeHistogram {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Consistency check
// ---------------------------------------------------------------------------

/// Result of checking whether hardware behaviour is a subset of the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyCheck {
    /// Name of the test.
    pub test_name: String,
    /// Whether hardware is consistent with the model.
    pub is_consistent: bool,
    /// Outcomes that are forbidden but were observed.
    pub forbidden_observed: Vec<(String, u64)>,
    /// Required outcomes that were never observed.
    pub required_missing: Vec<String>,
    /// Allowed outcomes that were observed.
    pub allowed_observed: Vec<(String, u64)>,
    /// Total observations.
    pub total_observations: u64,
}

impl ConsistencyCheck {
    /// Perform a consistency check.
    pub fn check(test: &LitmusTest, observed: &TestOutcome) -> Self {
        let mut forbidden_observed = Vec::new();
        let mut required_missing = Vec::new();
        let mut allowed_observed = Vec::new();

        // Check each observed outcome against the model.
        for (obs, &count) in &observed.histogram {
            let mut classified = false;
            for (expected, classification) in &test.expected_outcomes {
                if obs.matches(expected) {
                    classified = true;
                    match classification {
                        LitmusOutcome::Forbidden => {
                            forbidden_observed.push((obs.display_key(), count));
                        }
                        LitmusOutcome::Allowed | LitmusOutcome::Required => {
                            allowed_observed.push((obs.display_key(), count));
                        }
                    }
                    break;
                }
            }
            if !classified {
                // Unknown outcome — treat as forbidden.
                forbidden_observed.push((obs.display_key(), count));
            }
        }

        // Check for required outcomes never seen.
        for (expected, classification) in &test.expected_outcomes {
            if *classification == LitmusOutcome::Required {
                let seen = observed.histogram.keys().any(|o| o.matches(expected));
                if !seen {
                    let key = format_expected_outcome(expected);
                    required_missing.push(key);
                }
            }
        }

        let is_consistent = forbidden_observed.is_empty() && required_missing.is_empty();

        Self {
            test_name: test.name.clone(),
            is_consistent,
            forbidden_observed,
            required_missing,
            allowed_observed,
            total_observations: observed.total_iterations,
        }
    }

    /// Human-readable summary.
    pub fn summary(&self) -> String {
        if self.is_consistent {
            format!(
                "{}: CONSISTENT ({} observations, {} distinct allowed)",
                self.test_name,
                self.total_observations,
                self.allowed_observed.len()
            )
        } else {
            let mut parts = Vec::new();
            if !self.forbidden_observed.is_empty() {
                let total: u64 = self.forbidden_observed.iter().map(|(_, c)| c).sum();
                parts.push(format!("{} forbidden observed ({} times)", self.forbidden_observed.len(), total));
            }
            if !self.required_missing.is_empty() {
                parts.push(format!("{} required missing", self.required_missing.len()));
            }
            format!(
                "{}: INCONSISTENT — {}",
                self.test_name,
                parts.join(", ")
            )
        }
    }
}

// ---------------------------------------------------------------------------
// Statistical test (chi-square)
// ---------------------------------------------------------------------------

/// Chi-square test comparing observed vs expected outcome distributions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalTest {
    /// Test name.
    pub test_name: String,
    /// Chi-square statistic.
    pub chi_square: f64,
    /// Degrees of freedom.
    pub degrees_of_freedom: usize,
    /// p-value (approximate).
    pub p_value: f64,
    /// Whether the test passes at the given significance level.
    pub passes: bool,
    /// Significance level used.
    pub significance_level: f64,
    /// Per-category contributions.
    pub contributions: Vec<(String, f64, f64, f64)>, // (outcome, observed, expected, contribution)
}

impl StatisticalTest {
    /// Run a chi-square goodness-of-fit test.
    ///
    /// `expected_fractions` maps outcome keys to their expected probability.
    /// Outcomes not in the map are treated as having expected fraction 0.
    pub fn chi_square_test(
        test_name: &str,
        observed: &OutcomeHistogram,
        expected_fractions: &HashMap<String, f64>,
        significance_level: f64,
    ) -> Self {
        let n = observed.total as f64;
        let mut chi2 = 0.0;
        let mut contributions = Vec::new();

        // For each expected category.
        for (key, &exp_frac) in expected_fractions {
            let obs_count = observed.counts.get(key).copied().unwrap_or(0) as f64;
            let exp_count = exp_frac * n;
            if exp_count > 0.0 {
                let contrib = (obs_count - exp_count).powi(2) / exp_count;
                chi2 += contrib;
                contributions.push((key.clone(), obs_count, exp_count, contrib));
            }
        }

        // For observed outcomes not in expected.
        for (key, &count) in &observed.counts {
            if !expected_fractions.contains_key(key) && count > 0 {
                // Unexpected outcome — contributes heavily.
                let contrib = count as f64;
                chi2 += contrib;
                contributions.push((key.clone(), count as f64, 0.0, contrib));
            }
        }

        let df = if expected_fractions.len() > 1 {
            expected_fractions.len() - 1
        } else {
            1
        };

        // Approximate p-value using Wilson-Hilferty approximation.
        let p_value = approximate_chi2_p_value(chi2, df);
        let passes = p_value >= significance_level;

        Self {
            test_name: test_name.to_string(),
            chi_square: chi2,
            degrees_of_freedom: df,
            p_value,
            passes,
            significance_level,
            contributions,
        }
    }

    /// Summary.
    pub fn summary(&self) -> String {
        format!(
            "{}: χ²={:.2}, df={}, p={:.4} — {}",
            self.test_name,
            self.chi_square,
            self.degrees_of_freedom,
            self.p_value,
            if self.passes { "PASS" } else { "FAIL" }
        )
    }
}

/// Approximate p-value for chi-square distribution using the Wilson-Hilferty
/// normal approximation.
fn approximate_chi2_p_value(chi2: f64, df: usize) -> f64 {
    if df == 0 {
        return 1.0;
    }
    let k = df as f64;
    // Wilson-Hilferty: Z ≈ ((χ²/k)^(1/3) - (1 - 2/(9k))) / sqrt(2/(9k))
    let term = 2.0 / (9.0 * k);
    let z = ((chi2 / k).powf(1.0 / 3.0) - (1.0 - term)) / term.sqrt();

    // Approximate P(Z > z) using logistic approximation.
    let p = 1.0 / (1.0 + (1.7 * z).exp());
    p.max(0.0).min(1.0)
}

// ---------------------------------------------------------------------------
// Axiom validation
// ---------------------------------------------------------------------------

/// Validation status for a single axiom.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AxiomStatus {
    Pass,
    Fail,
    NotTested,
}

impl fmt::Display for AxiomStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AxiomStatus::Pass => write!(f, "PASS"),
            AxiomStatus::Fail => write!(f, "FAIL"),
            AxiomStatus::NotTested => write!(f, "NOT_TESTED"),
        }
    }
}

/// Per-axiom validation result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AxiomValidation {
    pub axiom_name: String,
    pub status: AxiomStatus,
    pub relevant_tests: Vec<String>,
    pub notes: String,
}

// ---------------------------------------------------------------------------
// Model validation report
// ---------------------------------------------------------------------------

/// Complete validation report comparing hardware against model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelValidationReport {
    /// Report title / identifier.
    pub title: String,
    /// GPU backend tested.
    pub backend: GpuBackend,
    /// Per-test consistency checks.
    pub consistency_checks: Vec<ConsistencyCheck>,
    /// Per-axiom validation.
    pub axiom_validations: Vec<AxiomValidation>,
    /// Overall pass/fail.
    pub overall_pass: bool,
    /// Total tests run.
    pub total_tests: usize,
    /// Tests that passed.
    pub passed_tests: usize,
    /// Tests that failed.
    pub failed_tests: usize,
    /// Statistical tests.
    pub statistical_tests: Vec<StatisticalTest>,
    /// Free-form notes.
    pub notes: Vec<String>,
}

impl ModelValidationReport {
    pub fn new(title: impl Into<String>, backend: GpuBackend) -> Self {
        Self {
            title: title.into(),
            backend,
            consistency_checks: Vec::new(),
            axiom_validations: Vec::new(),
            overall_pass: true,
            total_tests: 0,
            passed_tests: 0,
            failed_tests: 0,
            statistical_tests: Vec::new(),
            notes: Vec::new(),
        }
    }

    /// Add a consistency check result.
    pub fn add_consistency_check(&mut self, check: ConsistencyCheck) {
        if !check.is_consistent {
            self.overall_pass = false;
            self.failed_tests += 1;
        } else {
            self.passed_tests += 1;
        }
        self.total_tests += 1;
        self.consistency_checks.push(check);
    }

    /// Add an axiom validation.
    pub fn add_axiom_validation(&mut self, validation: AxiomValidation) {
        if validation.status == AxiomStatus::Fail {
            self.overall_pass = false;
        }
        self.axiom_validations.push(validation);
    }

    /// Add a statistical test.
    pub fn add_statistical_test(&mut self, test: StatisticalTest) {
        if !test.passes {
            self.notes.push(format!(
                "Statistical test failed for {}: p={:.4}",
                test.test_name, test.p_value
            ));
        }
        self.statistical_tests.push(test);
    }

    /// Generate a human-readable text report.
    pub fn generate_text_report(&self) -> String {
        let mut report = String::with_capacity(4096);

        report.push_str(&format!("=== Model Validation Report: {} ===\n", self.title));
        report.push_str(&format!("Backend: {}\n", self.backend));
        report.push_str(&format!(
            "Overall: {} ({}/{} passed)\n\n",
            if self.overall_pass { "PASS" } else { "FAIL" },
            self.passed_tests,
            self.total_tests,
        ));

        // Consistency checks
        report.push_str("--- Consistency Checks ---\n");
        for check in &self.consistency_checks {
            report.push_str(&check.summary());
            report.push('\n');
        }
        report.push('\n');

        // Axiom validations
        if !self.axiom_validations.is_empty() {
            report.push_str("--- Axiom Validations ---\n");
            for av in &self.axiom_validations {
                report.push_str(&format!(
                    "{}: {} (tests: {})\n",
                    av.axiom_name,
                    av.status,
                    av.relevant_tests.join(", ")
                ));
                if !av.notes.is_empty() {
                    report.push_str(&format!("  Notes: {}\n", av.notes));
                }
            }
            report.push('\n');
        }

        // Statistical tests
        if !self.statistical_tests.is_empty() {
            report.push_str("--- Statistical Tests ---\n");
            for st in &self.statistical_tests {
                report.push_str(&st.summary());
                report.push('\n');
            }
            report.push('\n');
        }

        // Notes
        if !self.notes.is_empty() {
            report.push_str("--- Notes ---\n");
            for note in &self.notes {
                report.push_str(&format!("• {}\n", note));
            }
        }

        report
    }

    /// Serialise to JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

// ---------------------------------------------------------------------------
// ResultAnalyzer
// ---------------------------------------------------------------------------

/// High-level analyser that compares observed results against model predictions.
#[derive(Debug)]
pub struct ResultAnalyzer {
    backend: GpuBackend,
}

impl ResultAnalyzer {
    pub fn new(backend: GpuBackend) -> Self {
        Self { backend }
    }

    /// Analyse a single test result.
    pub fn analyze_test(
        &self,
        test: &LitmusTest,
        observed: &TestOutcome,
    ) -> ConsistencyCheck {
        ConsistencyCheck::check(test, observed)
    }

    /// Analyse a batch of results and produce a validation report.
    pub fn analyze_batch(
        &self,
        title: &str,
        tests: &[LitmusTest],
        results: &[HardwareResult],
    ) -> ModelValidationReport {
        let mut report = ModelValidationReport::new(title, self.backend);

        for result in results {
            // Find matching test.
            if let Some(test) = tests.iter().find(|t| t.name == result.test_name) {
                let check = ConsistencyCheck::check(test, &result.observed);
                report.add_consistency_check(check);
            }
        }

        report
    }

    /// Run a chi-square test on observed outcomes.
    pub fn run_statistical_test(
        &self,
        test_name: &str,
        observed: &TestOutcome,
        expected_fractions: &HashMap<String, f64>,
        significance_level: f64,
    ) -> StatisticalTest {
        let hist = OutcomeHistogram::from_test_outcome(observed);
        StatisticalTest::chi_square_test(test_name, &hist, expected_fractions, significance_level)
    }

    /// Compare two sets of hardware results (e.g., different backends).
    pub fn compare_backends(
        &self,
        title: &str,
        results_a: &[HardwareResult],
        label_a: &str,
        results_b: &[HardwareResult],
        label_b: &str,
    ) -> BackendComparison {
        let mut comparison = BackendComparison {
            title: title.to_string(),
            label_a: label_a.to_string(),
            label_b: label_b.to_string(),
            test_comparisons: Vec::new(),
        };

        for ra in results_a {
            if let Some(rb) = results_b.iter().find(|r| r.test_name == ra.test_name) {
                let tc = TestComparison {
                    test_name: ra.test_name.clone(),
                    consistent_a: ra.consistent,
                    consistent_b: rb.consistent,
                    outcomes_a: ra.observed.distinct_outcomes(),
                    outcomes_b: rb.observed.distinct_outcomes(),
                    agree: ra.consistent == rb.consistent,
                };
                comparison.test_comparisons.push(tc);
            }
        }

        comparison
    }
}

// ---------------------------------------------------------------------------
// Backend comparison
// ---------------------------------------------------------------------------

/// Comparison of results across two backends.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendComparison {
    pub title: String,
    pub label_a: String,
    pub label_b: String,
    pub test_comparisons: Vec<TestComparison>,
}

/// Per-test comparison between two backends.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestComparison {
    pub test_name: String,
    pub consistent_a: bool,
    pub consistent_b: bool,
    pub outcomes_a: usize,
    pub outcomes_b: usize,
    pub agree: bool,
}

impl BackendComparison {
    /// How many tests agree between backends.
    pub fn agreement_count(&self) -> usize {
        self.test_comparisons.iter().filter(|tc| tc.agree).count()
    }

    /// Fraction of tests that agree.
    pub fn agreement_fraction(&self) -> f64 {
        if self.test_comparisons.is_empty() {
            return 1.0;
        }
        self.agreement_count() as f64 / self.test_comparisons.len() as f64
    }

    pub fn summary(&self) -> String {
        format!(
            "{}: {}/{} tests agree between {} and {} ({:.1}%)",
            self.title,
            self.agreement_count(),
            self.test_comparisons.len(),
            self.label_a,
            self.label_b,
            self.agreement_fraction() * 100.0,
        )
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Format a model `Outcome` as a display string.
fn format_expected_outcome(outcome: &Outcome) -> String {
    let mut parts = Vec::new();
    let mut regs: Vec<_> = outcome.registers.iter().collect();
    regs.sort_by_key(|&(&(t, r), _)| (t, r));
    for (&(t, r), &v) in &regs {
        parts.push(format!("T{}:r{}={}", t, r, v));
    }
    let mut mems: Vec<_> = outcome.memory.iter().collect();
    mems.sort_by_key(|&(&a, _)| a);
    for (&a, &v) in &mems {
        parts.push(format!("[{}]={}", a, v));
    }
    parts.join(", ")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checker::litmus::{
        Instruction, LitmusOutcome, Ordering, Thread,
    };

    fn make_test() -> LitmusTest {
        let t0 = Thread {
            id: 0,
            instructions: vec![Instruction::Store {
                addr: 0,
                value: 1,
                ordering: Ordering::Relaxed,
            }],
        };
        let t1 = Thread {
            id: 1,
            instructions: vec![Instruction::Load {
                reg: 0,
                addr: 0,
                ordering: Ordering::Relaxed,
            }],
        };

        let allowed = Outcome {
            registers: {
                let mut m = HashMap::new();
                m.insert((1, 0), 0);
                m
            },
            memory: HashMap::new(),
        };
        let forbidden = Outcome {
            registers: {
                let mut m = HashMap::new();
                m.insert((1, 0), 99);
                m
            },
            memory: HashMap::new(),
        };

        LitmusTest {
            name: "SL".into(),
            threads: vec![t0, t1],
            initial_state: {
                let mut m = HashMap::new();
                m.insert(0u64, 0u64);
                m
            },
            expected_outcomes: vec![
                (allowed, LitmusOutcome::Allowed),
                (forbidden, LitmusOutcome::Forbidden),
            ],
        }
    }

    fn make_observed(vals: &[(usize, usize, u64, u64)]) -> TestOutcome {
        let mut to = TestOutcome::new("SL", GpuBackend::Cuda);
        for &(tid, reg, val, count) in vals {
            let obs = ObservedOutcome::new().with_register(tid, reg, val);
            *to.histogram.entry(obs).or_insert(0) += count;
            to.total_iterations += count;
        }
        to
    }

    // -- OutcomeHistogram tests --

    #[test]
    fn test_histogram_basic() {
        let mut hist = OutcomeHistogram::new();
        hist.record("a", 10);
        hist.record("b", 20);
        hist.record("a", 5);
        assert_eq!(hist.total, 35);
        assert_eq!(hist.distinct(), 2);
        assert!((hist.fraction("a") - 15.0 / 35.0).abs() < 1e-9);
    }

    #[test]
    fn test_histogram_sorted() {
        let mut hist = OutcomeHistogram::new();
        hist.record("rare", 1);
        hist.record("common", 100);
        hist.record("mid", 50);
        let sorted = hist.sorted();
        assert_eq!(sorted[0].0, "common");
        assert_eq!(sorted[1].0, "mid");
        assert_eq!(sorted[2].0, "rare");
    }

    #[test]
    fn test_histogram_entropy() {
        let mut hist = OutcomeHistogram::new();
        hist.record("a", 50);
        hist.record("b", 50);
        let e = hist.entropy();
        assert!((e - 1.0).abs() < 0.01); // 1 bit for uniform binary
    }

    #[test]
    fn test_histogram_merge() {
        let mut h1 = OutcomeHistogram::new();
        h1.record("a", 10);
        let mut h2 = OutcomeHistogram::new();
        h2.record("a", 5);
        h2.record("b", 3);
        h1.merge(&h2);
        assert_eq!(h1.total, 18);
        assert_eq!(h1.counts["a"], 15);
    }

    #[test]
    fn test_histogram_from_test_outcome() {
        let observed = make_observed(&[(1, 0, 0, 80), (1, 0, 1, 20)]);
        let hist = OutcomeHistogram::from_test_outcome(&observed);
        assert_eq!(hist.total, 100);
        assert_eq!(hist.distinct(), 2);
    }

    // -- ConsistencyCheck tests --

    #[test]
    fn test_consistency_pass() {
        let test = make_test();
        let observed = make_observed(&[(1, 0, 0, 100)]);
        let check = ConsistencyCheck::check(&test, &observed);
        assert!(check.is_consistent);
        assert!(check.forbidden_observed.is_empty());
    }

    #[test]
    fn test_consistency_fail_forbidden() {
        let test = make_test();
        let observed = make_observed(&[(1, 0, 99, 5), (1, 0, 0, 95)]);
        let check = ConsistencyCheck::check(&test, &observed);
        assert!(!check.is_consistent);
        assert_eq!(check.forbidden_observed.len(), 1);
    }

    #[test]
    fn test_consistency_fail_unknown() {
        let test = make_test();
        let observed = make_observed(&[(1, 0, 42, 10)]);
        let check = ConsistencyCheck::check(&test, &observed);
        assert!(!check.is_consistent);
    }

    #[test]
    fn test_consistency_summary() {
        let test = make_test();
        let observed = make_observed(&[(1, 0, 0, 100)]);
        let check = ConsistencyCheck::check(&test, &observed);
        let s = check.summary();
        assert!(s.contains("CONSISTENT"));
    }

    // -- StatisticalTest tests --

    #[test]
    fn test_chi_square_uniform() {
        let mut hist = OutcomeHistogram::new();
        hist.record("a", 50);
        hist.record("b", 50);

        let mut expected = HashMap::new();
        expected.insert("a".to_string(), 0.5);
        expected.insert("b".to_string(), 0.5);

        let result = StatisticalTest::chi_square_test("test", &hist, &expected, 0.05);
        assert!(result.chi_square < 1.0);
        assert!(result.passes);
    }

    #[test]
    fn test_chi_square_skewed() {
        let mut hist = OutcomeHistogram::new();
        hist.record("a", 990);
        hist.record("b", 10);

        let mut expected = HashMap::new();
        expected.insert("a".to_string(), 0.5);
        expected.insert("b".to_string(), 0.5);

        let result = StatisticalTest::chi_square_test("test", &hist, &expected, 0.05);
        assert!(result.chi_square > 10.0);
    }

    #[test]
    fn test_chi_square_summary() {
        let mut hist = OutcomeHistogram::new();
        hist.record("a", 50);
        hist.record("b", 50);
        let mut expected = HashMap::new();
        expected.insert("a".to_string(), 0.5);
        expected.insert("b".to_string(), 0.5);
        let result = StatisticalTest::chi_square_test("test", &hist, &expected, 0.05);
        let s = result.summary();
        assert!(s.contains("χ²"));
    }

    // -- AxiomStatus tests --

    #[test]
    fn test_axiom_status_display() {
        assert_eq!(format!("{}", AxiomStatus::Pass), "PASS");
        assert_eq!(format!("{}", AxiomStatus::Fail), "FAIL");
        assert_eq!(format!("{}", AxiomStatus::NotTested), "NOT_TESTED");
    }

    // -- ModelValidationReport tests --

    #[test]
    fn test_report_basic() {
        let mut report = ModelValidationReport::new("test_report", GpuBackend::Cuda);
        let test = make_test();
        let observed = make_observed(&[(1, 0, 0, 100)]);
        let check = ConsistencyCheck::check(&test, &observed);
        report.add_consistency_check(check);

        assert!(report.overall_pass);
        assert_eq!(report.total_tests, 1);
        assert_eq!(report.passed_tests, 1);
    }

    #[test]
    fn test_report_with_failure() {
        let mut report = ModelValidationReport::new("test_report", GpuBackend::Vulkan);
        let test = make_test();
        let observed = make_observed(&[(1, 0, 99, 5)]);
        let check = ConsistencyCheck::check(&test, &observed);
        report.add_consistency_check(check);

        assert!(!report.overall_pass);
        assert_eq!(report.failed_tests, 1);
    }

    #[test]
    fn test_report_text_generation() {
        let mut report = ModelValidationReport::new("test", GpuBackend::Cuda);
        let test = make_test();
        let observed = make_observed(&[(1, 0, 0, 100)]);
        let check = ConsistencyCheck::check(&test, &observed);
        report.add_consistency_check(check);

        let text = report.generate_text_report();
        assert!(text.contains("Model Validation Report"));
        assert!(text.contains("PASS"));
    }

    #[test]
    fn test_report_json() {
        let report = ModelValidationReport::new("test", GpuBackend::Cuda);
        let json = report.to_json().unwrap();
        assert!(json.contains("\"title\""));
    }

    // -- ResultAnalyzer tests --

    #[test]
    fn test_analyzer_single() {
        let analyzer = ResultAnalyzer::new(GpuBackend::Cuda);
        let test = make_test();
        let observed = make_observed(&[(1, 0, 0, 100)]);
        let check = analyzer.analyze_test(&test, &observed);
        assert!(check.is_consistent);
    }

    #[test]
    fn test_analyzer_batch() {
        let analyzer = ResultAnalyzer::new(GpuBackend::Cuda);
        let test = make_test();
        let mut observed = TestOutcome::new("SL", GpuBackend::Cuda);
        let obs = ObservedOutcome::new().with_register(1, 0, 0);
        observed.record(obs);

        let hr = HardwareResult::from_test(&test, observed);
        let report = analyzer.analyze_batch("batch", &[test], &[hr]);
        assert!(report.overall_pass);
    }

    // -- BackendComparison tests --

    #[test]
    fn test_backend_comparison() {
        let analyzer = ResultAnalyzer::new(GpuBackend::Cuda);
        let test = make_test();

        let mut obs_a = TestOutcome::new("SL", GpuBackend::Cuda);
        obs_a.record(ObservedOutcome::new().with_register(1, 0, 0));
        let hr_a = HardwareResult::from_test(&test, obs_a);

        let mut obs_b = TestOutcome::new("SL", GpuBackend::Vulkan);
        obs_b.record(ObservedOutcome::new().with_register(1, 0, 0));
        let hr_b = HardwareResult::from_test(&test, obs_b);

        let cmp = analyzer.compare_backends("cmp", &[hr_a], "CUDA", &[hr_b], "Vulkan");
        assert_eq!(cmp.agreement_count(), 1);
        assert!((cmp.agreement_fraction() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_format_expected_outcome() {
        let outcome = Outcome {
            registers: {
                let mut m = HashMap::new();
                m.insert((0, 0), 1);
                m
            },
            memory: {
                let mut m = HashMap::new();
                m.insert(100u64, 42u64);
                m
            },
        };
        let s = format_expected_outcome(&outcome);
        assert!(s.contains("T0:r0=1"));
        assert!(s.contains("[100]=42"));
    }

    #[test]
    fn test_approximate_chi2_p_value() {
        // Small chi-square with many df should give p close to 1.
        let p = approximate_chi2_p_value(1.0, 10);
        assert!(p > 0.5);
        // Very large chi-square should give p close to 0.
        let p2 = approximate_chi2_p_value(1000.0, 5);
        assert!(p2 < 0.01);
    }
}
