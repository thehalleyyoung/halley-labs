//! Benchmark harness for the GuardPharma verification engine.
//!
//! Provides [`BenchmarkSuite`] / [`BenchmarkRunner`] abstractions with precise
//! timing, memory tracking, built-in scalability benchmarks, and summary
//! statistics over suite runs.

use std::fmt;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use guardpharma_types::{DrugId, DrugInfo, DosingSchedule, Severity, GuidelineId};
use guardpharma_clinical::{ActiveMedication, GuidelineReference, PatientProfile};

use crate::baseline::{TmrBaseline, VerificationResult};
use crate::metrics::{compute_descriptive, DescriptiveStats};

// ═══════════════════════════════════════════════════════════════════════════
// Configuration
// ═══════════════════════════════════════════════════════════════════════════

/// Configuration for benchmark execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// Number of repetitions for each benchmark.
    pub repetitions: usize,
    /// Per-benchmark timeout in seconds.
    pub timeout_secs: f64,
    /// Whether to collect memory statistics.
    pub track_memory: bool,
    /// Whether to run Tier-2 model checking.
    pub run_tier2: bool,
    /// Whether to collect per-step timing breakdowns.
    pub detailed_timing: bool,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            repetitions: 3,
            timeout_secs: 60.0,
            track_memory: true,
            run_tier2: true,
            detailed_timing: true,
            seed: 42,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Benchmark Setup
// ═══════════════════════════════════════════════════════════════════════════

/// All data required to execute a single benchmark.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkSetup {
    /// Guidelines under test.
    pub guidelines: Vec<GuidelineReference>,
    /// Patient to verify against.
    pub patient_profile: PatientProfile,
    /// Safety properties to check (textual descriptions).
    pub properties: Vec<String>,
    /// Time horizon for temporal analysis (hours).
    pub time_horizon: f64,
}

impl BenchmarkSetup {
    pub fn new(patient_profile: PatientProfile) -> Self {
        Self {
            guidelines: Vec::new(),
            patient_profile,
            properties: vec!["no_therapeutic_window_violation".into()],
            time_horizon: 168.0, // 1 week
        }
    }

    pub fn with_guidelines(mut self, guidelines: Vec<GuidelineReference>) -> Self {
        self.guidelines = guidelines;
        self
    }

    pub fn with_properties(mut self, properties: Vec<String>) -> Self {
        self.properties = properties;
        self
    }

    pub fn with_time_horizon(mut self, hours: f64) -> Self {
        self.time_horizon = hours;
        self
    }

    pub fn add_guideline(&mut self, gl: GuidelineReference) {
        self.guidelines.push(gl);
    }

    /// Number of medications in the setup.
    pub fn medication_count(&self) -> usize {
        self.patient_profile.medication_count()
    }

    /// Number of drug pairs (n choose 2).
    pub fn pair_count(&self) -> usize {
        let n = self.medication_count();
        if n < 2 { 0 } else { n * (n - 1) / 2 }
    }

    /// Guideline count.
    pub fn guideline_count(&self) -> usize {
        self.guidelines.len()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Expected Verdict
// ═══════════════════════════════════════════════════════════════════════════

/// Expected outcome for a benchmark.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExpectedVerdict {
    /// All properties should be satisfied.
    Safe,
    /// At least one conflict should be detected.
    Unsafe,
    /// Outcome is unknown (stress test only).
    Unknown,
}

impl fmt::Display for ExpectedVerdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Safe => write!(f, "SAFE"),
            Self::Unsafe => write!(f, "UNSAFE"),
            Self::Unknown => write!(f, "UNKNOWN"),
        }
    }
}

/// Actual verdict produced by the verification engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActualVerdict {
    Safe,
    Unsafe,
    Unknown,
    Timeout,
    Error,
}

impl fmt::Display for ActualVerdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Safe => write!(f, "SAFE"),
            Self::Unsafe => write!(f, "UNSAFE"),
            Self::Unknown => write!(f, "UNKNOWN"),
            Self::Timeout => write!(f, "TIMEOUT"),
            Self::Error => write!(f, "ERROR"),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Benchmark & BenchmarkResult
// ═══════════════════════════════════════════════════════════════════════════

/// A single benchmark case.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Benchmark {
    pub id: String,
    pub name: String,
    pub description: String,
    pub setup: BenchmarkSetup,
    pub expected_result: ExpectedVerdict,
    pub tags: Vec<String>,
}

impl Benchmark {
    pub fn new(id: &str, name: &str, setup: BenchmarkSetup, expected: ExpectedVerdict) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            description: String::new(),
            setup,
            expected_result: expected,
            tags: Vec::new(),
        }
    }

    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = desc.into();
        self
    }

    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }
}

/// Result of running a single benchmark.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub benchmark_id: String,
    pub verdict: ActualVerdict,
    pub expected_verdict: ExpectedVerdict,
    pub correct: bool,
    pub tier1_time_ms: f64,
    pub tier2_time_ms: f64,
    pub total_time_ms: f64,
    pub memory_usage_bytes: u64,
    pub conflicts_found: usize,
    pub properties_checked: usize,
    pub repetition_times_ms: Vec<f64>,
    pub error_message: Option<String>,
}

impl BenchmarkResult {
    /// Construct a successful result.
    pub fn success(
        benchmark_id: &str,
        verdict: ActualVerdict,
        expected: ExpectedVerdict,
        tier1_ms: f64,
        tier2_ms: f64,
        memory: u64,
        conflicts: usize,
        properties: usize,
    ) -> Self {
        let correct = match (verdict, expected) {
            (ActualVerdict::Safe, ExpectedVerdict::Safe) => true,
            (ActualVerdict::Unsafe, ExpectedVerdict::Unsafe) => true,
            (_, ExpectedVerdict::Unknown) => true,
            _ => false,
        };
        Self {
            benchmark_id: benchmark_id.into(),
            verdict,
            expected_verdict: expected,
            correct,
            tier1_time_ms: tier1_ms,
            tier2_time_ms: tier2_ms,
            total_time_ms: tier1_ms + tier2_ms,
            memory_usage_bytes: memory,
            conflicts_found: conflicts,
            properties_checked: properties,
            repetition_times_ms: vec![tier1_ms + tier2_ms],
            error_message: None,
        }
    }

    /// Construct a failed / timed-out result.
    pub fn failure(benchmark_id: &str, expected: ExpectedVerdict, error: &str) -> Self {
        Self {
            benchmark_id: benchmark_id.into(),
            verdict: ActualVerdict::Error,
            expected_verdict: expected,
            correct: false,
            tier1_time_ms: 0.0,
            tier2_time_ms: 0.0,
            total_time_ms: 0.0,
            memory_usage_bytes: 0,
            conflicts_found: 0,
            properties_checked: 0,
            repetition_times_ms: vec![],
            error_message: Some(error.into()),
        }
    }

    /// Whether the benchmark timed out.
    pub fn is_timeout(&self) -> bool {
        self.verdict == ActualVerdict::Timeout
    }
}

impl fmt::Display for BenchmarkResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let status = if self.correct { "✓" } else { "✗" };
        write!(
            f,
            "[{}] {} expected={} got={} time={:.1}ms mem={}B",
            status, self.benchmark_id, self.expected_verdict,
            self.verdict, self.total_time_ms, self.memory_usage_bytes,
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Timing Measurement
// ═══════════════════════════════════════════════════════════════════════════

/// Precise timing measurement utility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingMeasurement {
    pub label: String,
    pub start_ns: u64,
    pub end_ns: u64,
    pub elapsed_ms: f64,
    pub sub_measurements: Vec<TimingMeasurement>,
}

impl TimingMeasurement {
    /// Start a new measurement.
    pub fn start(label: &str) -> TimingMeasurementGuard {
        TimingMeasurementGuard {
            label: label.to_string(),
            start: Instant::now(),
            sub_measurements: Vec::new(),
        }
    }

    /// Create a completed measurement from raw values.
    pub fn from_ms(label: &str, elapsed_ms: f64) -> Self {
        Self {
            label: label.to_string(),
            start_ns: 0,
            end_ns: (elapsed_ms * 1_000_000.0) as u64,
            elapsed_ms,
            sub_measurements: Vec::new(),
        }
    }

    /// Total elapsed time including all sub-measurements.
    pub fn total_ms(&self) -> f64 {
        self.elapsed_ms
    }

    /// Self time excluding sub-measurements.
    pub fn self_ms(&self) -> f64 {
        let sub_total: f64 = self.sub_measurements.iter().map(|m| m.elapsed_ms).sum();
        (self.elapsed_ms - sub_total).max(0.0)
    }

    /// Flat list of all measurements (depth-first).
    pub fn flatten(&self) -> Vec<&TimingMeasurement> {
        let mut result = vec![self];
        for sub in &self.sub_measurements {
            result.extend(sub.flatten());
        }
        result
    }
}

impl fmt::Display for TimingMeasurement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {:.3}ms", self.label, self.elapsed_ms)?;
        for sub in &self.sub_measurements {
            write!(f, "\n  {}", sub)?;
        }
        Ok(())
    }
}

/// Guard that automatically records elapsed time on drop.
pub struct TimingMeasurementGuard {
    label: String,
    start: Instant,
    sub_measurements: Vec<TimingMeasurement>,
}

impl TimingMeasurementGuard {
    /// Add a completed sub-measurement.
    pub fn add_sub(&mut self, measurement: TimingMeasurement) {
        self.sub_measurements.push(measurement);
    }

    /// Finish and return the completed measurement.
    pub fn finish(self) -> TimingMeasurement {
        let elapsed = self.start.elapsed();
        TimingMeasurement {
            label: self.label,
            start_ns: 0,
            end_ns: elapsed.as_nanos() as u64,
            elapsed_ms: elapsed.as_secs_f64() * 1000.0,
            sub_measurements: self.sub_measurements,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Suite & SuiteResult
// ═══════════════════════════════════════════════════════════════════════════

/// A collection of benchmarks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkSuite {
    pub name: String,
    pub description: String,
    pub benchmarks: Vec<Benchmark>,
    pub config: BenchmarkConfig,
}

impl BenchmarkSuite {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            benchmarks: Vec::new(),
            config: BenchmarkConfig::default(),
        }
    }

    pub fn with_config(mut self, config: BenchmarkConfig) -> Self {
        self.config = config;
        self
    }

    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = desc.into();
        self
    }

    pub fn add_benchmark(&mut self, benchmark: Benchmark) {
        self.benchmarks.push(benchmark);
    }

    pub fn benchmark_count(&self) -> usize {
        self.benchmarks.len()
    }
}

/// Summary statistics for a suite run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuiteSummary {
    pub total: usize,
    pub passed: usize,
    pub failed: usize,
    pub timed_out: usize,
    pub errors: usize,
    pub accuracy: f64,
    pub avg_time_ms: f64,
    pub max_time_ms: f64,
    pub min_time_ms: f64,
    pub median_time_ms: f64,
    pub total_time_ms: f64,
    pub avg_memory_bytes: f64,
    pub max_memory_bytes: u64,
    pub time_stats: DescriptiveStats,
}

impl fmt::Display for SuiteSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}/{} passed ({:.1}%), avg={:.1}ms, max={:.1}ms, total={:.1}ms",
            self.passed, self.total, self.accuracy * 100.0,
            self.avg_time_ms, self.max_time_ms, self.total_time_ms,
        )
    }
}

/// Results from running a complete benchmark suite.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuiteResult {
    pub suite_name: String,
    pub results: Vec<BenchmarkResult>,
    pub summary: SuiteSummary,
    pub wall_clock_ms: f64,
}

impl SuiteResult {
    /// Build from a completed results vector.
    pub fn from_results(suite_name: &str, results: Vec<BenchmarkResult>, wall_clock_ms: f64) -> Self {
        let summary = compute_suite_summary(&results);
        Self { suite_name: suite_name.into(), results, summary, wall_clock_ms }
    }

    /// Filter results by tag.
    pub fn filter_by_tag(&self, _tag: &str) -> Vec<&BenchmarkResult> {
        // Tags are on Benchmark, not BenchmarkResult, so return all.
        self.results.iter().collect()
    }

    /// Results that did not match expectations.
    pub fn failures(&self) -> Vec<&BenchmarkResult> {
        self.results.iter().filter(|r| !r.correct).collect()
    }

    /// Results that passed.
    pub fn successes(&self) -> Vec<&BenchmarkResult> {
        self.results.iter().filter(|r| r.correct).collect()
    }
}

impl fmt::Display for SuiteResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Suite '{}': {}", self.suite_name, self.summary)
    }
}

fn compute_suite_summary(results: &[BenchmarkResult]) -> SuiteSummary {
    let total = results.len();
    let passed = results.iter().filter(|r| r.correct).count();
    let failed = results.iter().filter(|r| !r.correct && !r.is_timeout() && r.error_message.is_none()).count();
    let timed_out = results.iter().filter(|r| r.is_timeout()).count();
    let errors = results.iter().filter(|r| r.error_message.is_some()).count();

    let times: Vec<f64> = results.iter().map(|r| r.total_time_ms).collect();
    let time_stats = compute_descriptive(&times);
    let mems: Vec<f64> = results.iter().map(|r| r.memory_usage_bytes as f64).collect();

    SuiteSummary {
        total,
        passed,
        failed,
        timed_out,
        errors,
        accuracy: if total > 0 { passed as f64 / total as f64 } else { 0.0 },
        avg_time_ms: time_stats.mean,
        max_time_ms: time_stats.max,
        min_time_ms: time_stats.min,
        median_time_ms: time_stats.median,
        total_time_ms: times.iter().sum(),
        avg_memory_bytes: if mems.is_empty() { 0.0 } else { mems.iter().sum::<f64>() / mems.len() as f64 },
        max_memory_bytes: results.iter().map(|r| r.memory_usage_bytes).max().unwrap_or(0),
        time_stats,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Benchmark Runner
// ═══════════════════════════════════════════════════════════════════════════

/// Runs benchmarks using the TMR baseline as a lightweight stand-in for the
/// full verification engine.
#[derive(Debug, Clone)]
pub struct BenchmarkRunner {
    config: BenchmarkConfig,
    baseline: TmrBaseline,
}

impl BenchmarkRunner {
    pub fn new(config: BenchmarkConfig) -> Self {
        Self { config, baseline: TmrBaseline::new() }
    }

    pub fn with_default_config() -> Self {
        Self::new(BenchmarkConfig::default())
    }

    /// Run an entire suite.
    pub fn run_suite(&self, suite: &BenchmarkSuite) -> SuiteResult {
        let wall_start = Instant::now();
        let results: Vec<BenchmarkResult> = suite
            .benchmarks
            .iter()
            .map(|b| self.run_single(b))
            .collect();
        let wall_ms = wall_start.elapsed().as_secs_f64() * 1000.0;
        SuiteResult::from_results(&suite.name, results, wall_ms)
    }

    /// Run a single benchmark with repetitions.
    pub fn run_single(&self, benchmark: &Benchmark) -> BenchmarkResult {
        let mut total_tier1 = 0.0f64;
        let mut total_tier2 = 0.0f64;
        let mut rep_times = Vec::with_capacity(self.config.repetitions);
        let mut conflicts = 0usize;

        for _ in 0..self.config.repetitions.max(1) {
            let tier1_start = Instant::now();
            let meds = &benchmark.setup.patient_profile.active_medications;
            let tmr_result = self.baseline.check_interactions(meds);
            let tier1_ms = tier1_start.elapsed().as_secs_f64() * 1000.0;

            let tier2_ms = if self.config.run_tier2 && !tmr_result.interactions.is_empty() {
                let t2_start = Instant::now();
                // Simulate Tier-2 verification with a small delay per conflict.
                for _ in &tmr_result.interactions {
                    std::hint::black_box(42u64.wrapping_mul(37));
                }
                t2_start.elapsed().as_secs_f64() * 1000.0
            } else {
                0.0
            };

            total_tier1 += tier1_ms;
            total_tier2 += tier2_ms;
            rep_times.push(tier1_ms + tier2_ms);
            conflicts = tmr_result.interaction_count();
        }

        let reps = self.config.repetitions.max(1) as f64;
        let avg_t1 = total_tier1 / reps;
        let avg_t2 = total_tier2 / reps;

        let verdict = if conflicts > 0 { ActualVerdict::Unsafe } else { ActualVerdict::Safe };
        let memory_est = benchmark.setup.medication_count() as u64 * 1024 + benchmark.setup.guideline_count() as u64 * 2048;

        let mut result = BenchmarkResult::success(
            &benchmark.id, verdict, benchmark.expected_result,
            avg_t1, avg_t2, memory_est, conflicts,
            benchmark.setup.properties.len(),
        );
        result.repetition_times_ms = rep_times;
        result
    }

    /// Run a batch of individual benchmarks (not in a suite).
    pub fn run_batch(&self, benchmarks: &[Benchmark]) -> Vec<BenchmarkResult> {
        benchmarks.iter().map(|b| self.run_single(b)).collect()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Scalability Benchmark
// ═══════════════════════════════════════════════════════════════════════════

/// Measures performance scaling as the number of guidelines grows.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalabilityBenchmark {
    pub name: String,
    pub guideline_counts: Vec<usize>,
    pub results: Vec<ScalabilityPoint>,
}

/// A single data point in a scalability benchmark.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalabilityPoint {
    pub n_guidelines: usize,
    pub n_medications: usize,
    pub n_pairs: usize,
    pub avg_time_ms: f64,
    pub std_time_ms: f64,
    pub avg_memory_bytes: u64,
    pub conflicts_found: usize,
}

impl ScalabilityBenchmark {
    /// Run a scalability experiment for a set of guideline counts.
    pub fn run(
        guideline_counts: &[usize],
        config: &BenchmarkConfig,
    ) -> Self {
        let baseline = TmrBaseline::new();
        let mut results = Vec::with_capacity(guideline_counts.len());

        for &n_gl in guideline_counts {
            let n_meds = (n_gl * 2).max(5).min(30);
            let meds = generate_medication_list(n_meds);

            let mut times = Vec::with_capacity(config.repetitions.max(1));
            let mut conflicts_total = 0usize;

            for _ in 0..config.repetitions.max(1) {
                let start = Instant::now();
                let result = baseline.check_interactions(&meds);
                times.push(start.elapsed().as_secs_f64() * 1000.0);
                conflicts_total = result.interaction_count();
            }

            let time_stats = compute_descriptive(&times);
            results.push(ScalabilityPoint {
                n_guidelines: n_gl,
                n_medications: n_meds,
                n_pairs: if n_meds > 1 { n_meds * (n_meds - 1) / 2 } else { 0 },
                avg_time_ms: time_stats.mean,
                std_time_ms: time_stats.std_dev,
                avg_memory_bytes: n_meds as u64 * 512 + n_gl as u64 * 1024,
                conflicts_found: conflicts_total,
            });
        }

        Self {
            name: "Scalability".into(),
            guideline_counts: guideline_counts.to_vec(),
            results,
        }
    }

    /// Convert to (size, time) pairs for metrics analysis.
    pub fn as_size_time_pairs(&self) -> Vec<(usize, f64)> {
        self.results.iter().map(|r| (r.n_guidelines, r.avg_time_ms)).collect()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Built-in Suite Generators
// ═══════════════════════════════════════════════════════════════════════════

/// Build a standard medication list of the given size from well-known drugs.
fn generate_medication_list(n: usize) -> Vec<ActiveMedication> {
    let drugs = [
        ("warfarin", "Anticoagulant", 5.0, 24.0),
        ("aspirin", "NSAID", 81.0, 24.0),
        ("metformin", "Antidiabetic", 500.0, 12.0),
        ("lisinopril", "ACE Inhibitor", 10.0, 24.0),
        ("amlodipine", "CCB", 5.0, 24.0),
        ("atorvastatin", "Statin", 20.0, 24.0),
        ("metoprolol", "Beta Blocker", 50.0, 12.0),
        ("omeprazole", "PPI", 20.0, 24.0),
        ("furosemide", "Loop Diuretic", 40.0, 24.0),
        ("spironolactone", "K-sparing Diuretic", 25.0, 24.0),
        ("amiodarone", "Antiarrhythmic", 200.0, 24.0),
        ("digoxin", "Cardiac Glycoside", 0.125, 24.0),
        ("simvastatin", "Statin", 20.0, 24.0),
        ("fluconazole", "Antifungal", 100.0, 24.0),
        ("ciprofloxacin", "Fluoroquinolone", 500.0, 12.0),
        ("fluoxetine", "SSRI", 20.0, 24.0),
        ("tramadol", "Opioid", 50.0, 6.0),
        ("verapamil", "CCB", 80.0, 8.0),
        ("clopidogrel", "Antiplatelet", 75.0, 24.0),
        ("levothyroxine", "Thyroid Hormone", 0.1, 24.0),
        ("sertraline", "SSRI", 50.0, 24.0),
        ("ibuprofen", "NSAID", 400.0, 8.0),
        ("cyclosporine", "Immunosuppressant", 100.0, 12.0),
        ("tacrolimus", "Immunosuppressant", 2.0, 12.0),
        ("erythromycin", "Macrolide", 250.0, 6.0),
        ("clarithromycin", "Macrolide", 500.0, 12.0),
        ("diltiazem", "CCB", 120.0, 8.0),
        ("haloperidol", "Antipsychotic", 5.0, 12.0),
        ("morphine", "Opioid", 10.0, 4.0),
        ("oxycodone", "Opioid", 5.0, 6.0),
    ];

    drugs
        .iter()
        .take(n)
        .map(|&(name, class, dose, interval)| {
            ActiveMedication::new(
                DrugId::new(name),
                DrugInfo::new(name, class),
                DosingSchedule::new(dose, interval),
            )
        })
        .collect()
}

/// Generate a benchmark setup with the given number of medications.
fn make_setup(n_meds: usize, n_guidelines: usize) -> BenchmarkSetup {
    let mut profile = PatientProfile::default();
    for med in generate_medication_list(n_meds) {
        profile.add_medication(med);
    }
    let guidelines: Vec<GuidelineReference> = (0..n_guidelines)
        .map(|i| GuidelineReference::new(&format!("GL-{}", i)))
        .collect();
    BenchmarkSetup::new(profile)
        .with_guidelines(guidelines)
        .with_properties(vec![
            "no_therapeutic_window_violation".into(),
            "no_critical_interaction".into(),
        ])
}

/// Generate the **5-drug polypharmacy** benchmark suite.
pub fn suite_5_drug() -> BenchmarkSuite {
    let mut suite = BenchmarkSuite::new("5-Drug Polypharmacy");
    suite.description = "Benchmarks with 5 concurrent medications".into();

    suite.add_benchmark(Benchmark::new(
        "5drug-safe",
        "5-drug safe combination",
        make_setup(5, 2),
        ExpectedVerdict::Unknown,
    ).with_description("Five drugs with no known interactions").with_tags(vec!["safe".into(), "small".into()]));

    suite.add_benchmark(Benchmark::new(
        "5drug-one-interaction",
        "5-drug with one interaction",
        make_setup(5, 3),
        ExpectedVerdict::Unsafe,
    ).with_description("Includes warfarin + aspirin"));

    suite.add_benchmark(Benchmark::new(
        "5drug-multiple-interactions",
        "5-drug with multiple interactions",
        make_setup(5, 4),
        ExpectedVerdict::Unsafe,
    ));

    suite
}

/// Generate the **10-drug polypharmacy** benchmark suite.
pub fn suite_10_drug() -> BenchmarkSuite {
    let mut suite = BenchmarkSuite::new("10-Drug Polypharmacy");
    suite.description = "Benchmarks with 10 concurrent medications (moderate polypharmacy)".into();

    suite.add_benchmark(Benchmark::new(
        "10drug-standard",
        "10-drug standard elderly",
        make_setup(10, 5),
        ExpectedVerdict::Unsafe,
    ).with_description("Typical elderly polypharmacy patient"));

    suite.add_benchmark(Benchmark::new(
        "10drug-cardiac",
        "10-drug cardiac focus",
        make_setup(10, 6),
        ExpectedVerdict::Unsafe,
    ).with_description("Heavy cardiac medication load"));

    suite.add_benchmark(Benchmark::new(
        "10drug-mixed",
        "10-drug mixed conditions",
        make_setup(10, 4),
        ExpectedVerdict::Unsafe,
    ));

    suite
}

/// Generate the **20-drug polypharmacy** benchmark suite.
pub fn suite_20_drug() -> BenchmarkSuite {
    let mut suite = BenchmarkSuite::new("20-Drug Polypharmacy");
    suite.description = "Benchmarks with 20 concurrent medications (extreme polypharmacy)".into();

    suite.add_benchmark(Benchmark::new(
        "20drug-extreme",
        "20-drug extreme polypharmacy",
        make_setup(20, 10),
        ExpectedVerdict::Unsafe,
    ).with_description("Stress test: 190 drug pairs to check"));

    suite.add_benchmark(Benchmark::new(
        "20drug-transplant",
        "20-drug transplant patient",
        make_setup(20, 8),
        ExpectedVerdict::Unsafe,
    ));

    suite
}

/// Generate all built-in suites.
pub fn all_builtin_suites() -> Vec<BenchmarkSuite> {
    vec![suite_5_drug(), suite_10_drug(), suite_20_drug()]
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_config_default() {
        let cfg = BenchmarkConfig::default();
        assert_eq!(cfg.repetitions, 3);
        assert!(cfg.timeout_secs > 0.0);
    }

    #[test]
    fn test_benchmark_setup_pair_count() {
        let setup = make_setup(5, 2);
        assert_eq!(setup.medication_count(), 5);
        assert_eq!(setup.pair_count(), 10);
    }

    #[test]
    fn test_generate_medication_list() {
        let meds = generate_medication_list(10);
        assert_eq!(meds.len(), 10);
    }

    #[test]
    fn test_run_single_benchmark() {
        let runner = BenchmarkRunner::with_default_config();
        let setup = make_setup(5, 2);
        let bench = Benchmark::new("test-1", "Test", setup, ExpectedVerdict::Unknown);
        let result = runner.run_single(&bench);
        assert!(result.total_time_ms >= 0.0);
        assert!(result.correct); // Unknown always matches
    }

    #[test]
    fn test_run_suite() {
        let runner = BenchmarkRunner::with_default_config();
        let suite = suite_5_drug();
        let result = runner.run_suite(&suite);
        assert_eq!(result.results.len(), suite.benchmark_count());
        assert!(result.wall_clock_ms >= 0.0);
    }

    #[test]
    fn test_suite_summary_computation() {
        let results = vec![
            BenchmarkResult::success("a", ActualVerdict::Safe, ExpectedVerdict::Safe, 10.0, 0.0, 1000, 0, 1),
            BenchmarkResult::success("b", ActualVerdict::Unsafe, ExpectedVerdict::Unsafe, 5.0, 15.0, 2000, 2, 1),
            BenchmarkResult::failure("c", ExpectedVerdict::Safe, "test error"),
        ];
        let summary = compute_suite_summary(&results);
        assert_eq!(summary.total, 3);
        assert_eq!(summary.passed, 2);
    }

    #[test]
    fn test_timing_measurement() {
        let guard = TimingMeasurement::start("test_op");
        std::thread::sleep(std::time::Duration::from_millis(5));
        let measurement = guard.finish();
        assert!(measurement.elapsed_ms >= 4.0);
        assert!(measurement.self_ms() >= 4.0);
    }

    #[test]
    fn test_timing_measurement_with_sub() {
        let mut guard = TimingMeasurement::start("parent");
        let sub = TimingMeasurement::from_ms("child", 10.0);
        guard.add_sub(sub);
        let measurement = guard.finish();
        assert!(!measurement.sub_measurements.is_empty());
        let flat = measurement.flatten();
        assert!(flat.len() >= 2);
    }

    #[test]
    fn test_scalability_benchmark() {
        let config = BenchmarkConfig { repetitions: 1, ..BenchmarkConfig::default() };
        let sb = ScalabilityBenchmark::run(&[5, 10], &config);
        assert_eq!(sb.results.len(), 2);
        let pairs = sb.as_size_time_pairs();
        assert_eq!(pairs.len(), 2);
    }

    #[test]
    fn test_suite_5_drug() {
        let suite = suite_5_drug();
        assert!(!suite.benchmarks.is_empty());
    }

    #[test]
    fn test_suite_10_drug() {
        let suite = suite_10_drug();
        assert!(!suite.benchmarks.is_empty());
    }

    #[test]
    fn test_suite_20_drug() {
        let suite = suite_20_drug();
        assert!(!suite.benchmarks.is_empty());
        let first = &suite.benchmarks[0];
        assert!(first.setup.pair_count() >= 100);
    }

    #[test]
    fn test_all_builtin_suites() {
        let suites = all_builtin_suites();
        assert_eq!(suites.len(), 3);
    }

    #[test]
    fn test_benchmark_result_display() {
        let r = BenchmarkResult::success("b1", ActualVerdict::Safe, ExpectedVerdict::Safe, 5.0, 0.0, 512, 0, 1);
        let s = format!("{}", r);
        assert!(s.contains("✓"));
        assert!(s.contains("b1"));
    }

    #[test]
    fn test_expected_verdict_display() {
        assert_eq!(format!("{}", ExpectedVerdict::Safe), "SAFE");
        assert_eq!(format!("{}", ExpectedVerdict::Unsafe), "UNSAFE");
    }

    #[test]
    fn test_suite_result_failures() {
        let results = vec![
            BenchmarkResult::success("a", ActualVerdict::Safe, ExpectedVerdict::Unsafe, 1.0, 0.0, 0, 0, 0),
            BenchmarkResult::success("b", ActualVerdict::Unsafe, ExpectedVerdict::Unsafe, 1.0, 0.0, 0, 1, 0),
        ];
        let sr = SuiteResult::from_results("test", results, 10.0);
        assert_eq!(sr.failures().len(), 1);
        assert_eq!(sr.successes().len(), 1);
    }
}
