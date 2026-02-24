//! Run litmus tests on actual GPU hardware.
//!
//! Generates GPU kernel code, compiles it, runs tests, and collects results
//! across multiple backends (CUDA, OpenCL, Vulkan, Metal).

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::checker::litmus::{
    Instruction, LitmusOutcome, LitmusTest, Ordering, Outcome, Thread,
};
use crate::checker::execution::{Address, Value};

use super::kernel_gen::{
    CudaKernelGenerator, KernelGenerator, MetalKernelGenerator,
    OpenClKernelGenerator, VulkanShaderGenerator,
};

// ---------------------------------------------------------------------------
// GPU backend enumeration
// ---------------------------------------------------------------------------

/// Supported GPU compute backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GpuBackend {
    Cuda,
    OpenCL,
    Vulkan,
    Metal,
}

impl fmt::Display for GpuBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuBackend::Cuda => write!(f, "CUDA"),
            GpuBackend::OpenCL => write!(f, "OpenCL"),
            GpuBackend::Vulkan => write!(f, "Vulkan"),
            GpuBackend::Metal => write!(f, "Metal"),
        }
    }
}

impl GpuBackend {
    /// File extension for kernel source files.
    pub fn source_extension(&self) -> &'static str {
        match self {
            GpuBackend::Cuda => "cu",
            GpuBackend::OpenCL => "cl",
            GpuBackend::Vulkan => "comp",
            GpuBackend::Metal => "metal",
        }
    }

    /// Compiler command for this backend.
    pub fn compiler(&self) -> &'static str {
        match self {
            GpuBackend::Cuda => "nvcc",
            GpuBackend::OpenCL => "gcc",
            GpuBackend::Vulkan => "glslangValidator",
            GpuBackend::Metal => "xcrun",
        }
    }

    /// Default compiler flags.
    pub fn default_flags(&self) -> Vec<&'static str> {
        match self {
            GpuBackend::Cuda => vec!["-O2", "--gpu-architecture=sm_70"],
            GpuBackend::OpenCL => vec!["-lOpenCL", "-O2"],
            GpuBackend::Vulkan => vec!["--target-env", "vulkan1.2", "-V"],
            GpuBackend::Metal => vec!["-sdk", "macosx", "metal"],
        }
    }
}

// ---------------------------------------------------------------------------
// Stress mode
// ---------------------------------------------------------------------------

/// Stress-testing mode applied during hardware runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StressMode {
    /// No stress — plain execution.
    None,
    /// Light stress: mild memory pressure and scheduling perturbation.
    Light,
    /// Medium stress: bank conflicts, padding, and cache pressure.
    Medium,
    /// Heavy stress: all techniques combined.
    Heavy,
    /// Custom stress profile (index into user-defined table).
    Custom(u32),
}

impl Default for StressMode {
    fn default() -> Self {
        StressMode::None
    }
}

// ---------------------------------------------------------------------------
// Hardware configuration
// ---------------------------------------------------------------------------

/// Configuration for a hardware litmus-test run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareConfig {
    /// Number of iterations per test.
    pub iterations: u64,
    /// Threads per workgroup / block.
    pub thread_count: u32,
    /// Number of workgroups / blocks.
    pub workgroup_count: u32,
    /// GPU backend to use.
    pub gpu_backend: GpuBackend,
    /// Stress-testing mode.
    pub stress_mode: StressMode,
    /// Per-test timeout.
    pub timeout: Duration,
    /// Working directory for generated files.
    pub work_dir: PathBuf,
    /// Extra compiler flags.
    pub extra_flags: Vec<String>,
    /// Whether to keep intermediate files after the run.
    pub keep_intermediates: bool,
    /// Shuffle thread assignments between iterations.
    pub shuffle_threads: bool,
    /// Random seed (0 = use system entropy).
    pub seed: u64,
}

impl Default for HardwareConfig {
    fn default() -> Self {
        Self {
            iterations: 100_000,
            thread_count: 256,
            workgroup_count: 1,
            gpu_backend: GpuBackend::Cuda,
            stress_mode: StressMode::None,
            timeout: Duration::from_secs(60),
            work_dir: PathBuf::from("/tmp/litmus_hw"),
            extra_flags: Vec::new(),
            keep_intermediates: false,
            shuffle_threads: true,
            seed: 0,
        }
    }
}

impl HardwareConfig {
    pub fn new(backend: GpuBackend) -> Self {
        Self {
            gpu_backend: backend,
            ..Default::default()
        }
    }

    pub fn with_iterations(mut self, n: u64) -> Self {
        self.iterations = n;
        self
    }

    pub fn with_thread_count(mut self, n: u32) -> Self {
        self.thread_count = n;
        self
    }

    pub fn with_workgroup_count(mut self, n: u32) -> Self {
        self.workgroup_count = n;
        self
    }

    pub fn with_stress_mode(mut self, mode: StressMode) -> Self {
        self.stress_mode = mode;
        self
    }

    pub fn with_timeout(mut self, t: Duration) -> Self {
        self.timeout = t;
        self
    }

    pub fn with_work_dir(mut self, p: impl Into<PathBuf>) -> Self {
        self.work_dir = p.into();
        self
    }

    pub fn with_seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }

    /// Total threads launched across all workgroups.
    pub fn total_threads(&self) -> u64 {
        self.thread_count as u64 * self.workgroup_count as u64
    }

    /// Validate the configuration, returning errors.
    pub fn validate(&self) -> Result<(), RunnerError> {
        if self.iterations == 0 {
            return Err(RunnerError::InvalidConfig("iterations must be > 0".into()));
        }
        if self.thread_count == 0 {
            return Err(RunnerError::InvalidConfig("thread_count must be > 0".into()));
        }
        if self.workgroup_count == 0 {
            return Err(RunnerError::InvalidConfig(
                "workgroup_count must be > 0".into(),
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Observed outcome
// ---------------------------------------------------------------------------

/// A single observed outcome from one iteration of a litmus test.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ObservedOutcome {
    /// Register values: (thread_id, reg_id) -> value
    pub registers: BTreeMap<(usize, usize), Value>,
    /// Final memory state: address -> value
    pub memory: BTreeMap<Address, Value>,
}

impl ObservedOutcome {
    pub fn new() -> Self {
        Self {
            registers: BTreeMap::new(),
            memory: BTreeMap::new(),
        }
    }

    pub fn with_register(mut self, thread: usize, reg: usize, val: Value) -> Self {
        self.registers.insert((thread, reg), val);
        self
    }

    pub fn with_memory(mut self, addr: Address, val: Value) -> Self {
        self.memory.insert(addr, val);
        self
    }

    /// Check whether this outcome matches a litmus-test `Outcome`.
    pub fn matches(&self, expected: &Outcome) -> bool {
        for (&(tid, rid), &val) in &expected.registers {
            match self.registers.get(&(tid, rid)) {
                Some(&v) if v == val => {}
                _ => return false,
            }
        }
        for (&addr, &val) in &expected.memory {
            match self.memory.get(&addr) {
                Some(&v) if v == val => {}
                _ => return false,
            }
        }
        true
    }

    /// Compact display string.
    pub fn display_key(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        let mut regs: Vec<_> = self.registers.iter().collect();
        regs.sort_by_key(|&(&(t, r), _)| (t, r));
        for (&(t, r), &v) in &regs {
            parts.push(format!("T{}:r{}={}", t, r, v));
        }
        let mut mems: Vec<_> = self.memory.iter().collect();
        mems.sort_by_key(|&(&a, _)| a);
        for (&a, &v) in &mems {
            parts.push(format!("[{}]={}", a, v));
        }
        parts.join(", ")
    }
}

impl Default for ObservedOutcome {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Test outcome histogram
// ---------------------------------------------------------------------------

/// Histogram of observed outcomes across all iterations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestOutcome {
    /// Name of the test.
    pub test_name: String,
    /// Per-outcome frequency counts.
    pub histogram: HashMap<ObservedOutcome, u64>,
    /// Total iterations executed.
    pub total_iterations: u64,
    /// Wall-clock duration of the run.
    pub duration: Duration,
    /// Backend used.
    pub backend: GpuBackend,
}

impl TestOutcome {
    pub fn new(name: impl Into<String>, backend: GpuBackend) -> Self {
        Self {
            test_name: name.into(),
            histogram: HashMap::new(),
            total_iterations: 0,
            duration: Duration::ZERO,
            backend,
        }
    }

    /// Record one observed outcome.
    pub fn record(&mut self, outcome: ObservedOutcome) {
        *self.histogram.entry(outcome).or_insert(0) += 1;
        self.total_iterations += 1;
    }

    /// Number of distinct outcomes observed.
    pub fn distinct_outcomes(&self) -> usize {
        self.histogram.len()
    }

    /// Frequency of a particular outcome, or 0 if never observed.
    pub fn frequency(&self, outcome: &ObservedOutcome) -> u64 {
        self.histogram.get(outcome).copied().unwrap_or(0)
    }

    /// Fraction of iterations that produced `outcome`.
    pub fn fraction(&self, outcome: &ObservedOutcome) -> f64 {
        if self.total_iterations == 0 {
            return 0.0;
        }
        self.frequency(outcome) as f64 / self.total_iterations as f64
    }

    /// Outcomes sorted by decreasing frequency.
    pub fn sorted_outcomes(&self) -> Vec<(&ObservedOutcome, u64)> {
        let mut v: Vec<_> = self.histogram.iter().map(|(o, &c)| (o, c)).collect();
        v.sort_by(|a, b| b.1.cmp(&a.1));
        v
    }

    /// Merge another `TestOutcome` into this one.
    pub fn merge(&mut self, other: &TestOutcome) {
        for (outcome, &count) in &other.histogram {
            *self.histogram.entry(outcome.clone()).or_insert(0) += count;
        }
        self.total_iterations += other.total_iterations;
        self.duration += other.duration;
    }
}

// ---------------------------------------------------------------------------
// Outcome classification against model
// ---------------------------------------------------------------------------

/// Classification of an observed outcome with respect to the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutcomeClassification {
    /// Outcome is allowed by the model and was observed.
    AllowedObserved,
    /// Outcome is allowed by the model but was never observed.
    AllowedNotObserved,
    /// Outcome is forbidden by the model but was observed — a bug.
    ForbiddenObserved,
    /// Outcome is forbidden by the model and was not observed.
    ForbiddenNotObserved,
    /// Outcome was required by the model and observed.
    RequiredObserved,
    /// Outcome was required by the model but not observed.
    RequiredNotObserved,
}

impl OutcomeClassification {
    /// True if this classification represents a model violation.
    pub fn is_violation(&self) -> bool {
        matches!(
            self,
            OutcomeClassification::ForbiddenObserved
                | OutcomeClassification::RequiredNotObserved
        )
    }
}

// ---------------------------------------------------------------------------
// Hardware result (observed vs model-predicted comparison)
// ---------------------------------------------------------------------------

/// Comparison of observed hardware behaviour with model predictions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareResult {
    /// The litmus test that was run.
    pub test_name: String,
    /// Observed outcomes.
    pub observed: TestOutcome,
    /// Per-outcome classifications.
    pub classifications: Vec<(ObservedOutcome, OutcomeClassification, u64)>,
    /// Whether hardware behaviour is consistent with the model.
    pub consistent: bool,
    /// Human-readable notes.
    pub notes: Vec<String>,
}

impl HardwareResult {
    /// Build a `HardwareResult` by comparing observed outcomes against model.
    pub fn from_test(
        test: &LitmusTest,
        observed: TestOutcome,
    ) -> Self {
        let mut classifications = Vec::new();
        let mut consistent = true;
        let mut notes = Vec::new();

        // Classify each observed outcome.
        for (outcome, &count) in &observed.histogram {
            let mut matched = false;
            for (expected, classification) in &test.expected_outcomes {
                if outcome.matches(expected) {
                    matched = true;
                    let cls = match classification {
                        LitmusOutcome::Allowed => OutcomeClassification::AllowedObserved,
                        LitmusOutcome::Forbidden => {
                            consistent = false;
                            notes.push(format!(
                                "VIOLATION: forbidden outcome observed {} times: {}",
                                count,
                                outcome.display_key()
                            ));
                            OutcomeClassification::ForbiddenObserved
                        }
                        LitmusOutcome::Required => OutcomeClassification::RequiredObserved,
                    };
                    classifications.push((outcome.clone(), cls, count));
                    break;
                }
            }
            if !matched {
                // Outcome not mentioned in the test — treat as unexpected.
                notes.push(format!(
                    "Unexpected outcome observed {} times: {}",
                    count,
                    outcome.display_key()
                ));
                classifications.push((
                    outcome.clone(),
                    OutcomeClassification::ForbiddenObserved,
                    count,
                ));
                consistent = false;
            }
        }

        // Check for required outcomes that were never observed.
        for (expected, classification) in &test.expected_outcomes {
            if *classification == LitmusOutcome::Required {
                let found = observed.histogram.keys().any(|o| o.matches(expected));
                if !found {
                    notes.push("Required outcome was never observed".into());
                    let oo = ObservedOutcome {
                        registers: expected.registers.iter().map(|(&k, &v)| (k, v)).collect(),
                        memory: expected.memory.iter().map(|(&k, &v)| (k, v)).collect(),
                    };
                    classifications.push((
                        oo,
                        OutcomeClassification::RequiredNotObserved,
                        0,
                    ));
                    consistent = false;
                }
            }
        }

        Self {
            test_name: test.name.clone(),
            observed,
            classifications,
            consistent,
            notes,
        }
    }

    /// Number of model violations.
    pub fn violation_count(&self) -> usize {
        self.classifications
            .iter()
            .filter(|(_, c, _)| c.is_violation())
            .count()
    }

    /// Total observations of forbidden outcomes.
    pub fn forbidden_observation_count(&self) -> u64 {
        self.classifications
            .iter()
            .filter(|(_, c, _)| *c == OutcomeClassification::ForbiddenObserved)
            .map(|(_, _, n)| n)
            .sum()
    }

    /// Summary string.
    pub fn summary(&self) -> String {
        if self.consistent {
            format!(
                "{}: PASS ({} iterations, {} distinct outcomes)",
                self.test_name,
                self.observed.total_iterations,
                self.observed.distinct_outcomes()
            )
        } else {
            format!(
                "{}: FAIL ({} violations, {} iterations)",
                self.test_name,
                self.violation_count(),
                self.observed.total_iterations,
            )
        }
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors that can occur during hardware litmus-test execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RunnerError {
    /// Invalid configuration.
    InvalidConfig(String),
    /// Kernel generation failed.
    KernelGeneration(String),
    /// Kernel compilation failed.
    CompilationFailed { stderr: String, exit_code: Option<i32> },
    /// Test execution failed.
    ExecutionFailed(String),
    /// Timeout waiting for test completion.
    Timeout { elapsed: Duration },
    /// IO error (serialized message).
    Io(String),
    /// Backend not available on this system.
    BackendUnavailable(GpuBackend),
    /// Result parsing error.
    ResultParsing(String),
}

impl fmt::Display for RunnerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RunnerError::InvalidConfig(msg) => write!(f, "invalid config: {}", msg),
            RunnerError::KernelGeneration(msg) => write!(f, "kernel generation: {}", msg),
            RunnerError::CompilationFailed { stderr, exit_code } => {
                write!(f, "compilation failed (exit {:?}): {}", exit_code, stderr)
            }
            RunnerError::ExecutionFailed(msg) => write!(f, "execution failed: {}", msg),
            RunnerError::Timeout { elapsed } => {
                write!(f, "timeout after {:.1}s", elapsed.as_secs_f64())
            }
            RunnerError::Io(msg) => write!(f, "IO error: {}", msg),
            RunnerError::BackendUnavailable(b) => write!(f, "backend unavailable: {}", b),
            RunnerError::ResultParsing(msg) => write!(f, "result parsing: {}", msg),
        }
    }
}

impl From<std::io::Error> for RunnerError {
    fn from(e: std::io::Error) -> Self {
        RunnerError::Io(e.to_string())
    }
}

// ---------------------------------------------------------------------------
// Compilation result
// ---------------------------------------------------------------------------

/// Result of compiling a generated kernel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompilationResult {
    /// Path to the compiled binary / shader object.
    pub binary_path: PathBuf,
    /// Compiler stdout.
    pub stdout: String,
    /// Compiler stderr.
    pub stderr: String,
    /// Compilation duration.
    pub duration: Duration,
}

// ---------------------------------------------------------------------------
// LitmusRunner
// ---------------------------------------------------------------------------

/// Top-level runner for hardware litmus tests.
#[derive(Debug, Clone)]
pub struct LitmusRunner {
    config: HardwareConfig,
}

impl LitmusRunner {
    // ----- construction -----------------------------------------------------

    pub fn new(config: HardwareConfig) -> Result<Self, RunnerError> {
        config.validate()?;
        Ok(Self { config })
    }

    pub fn config(&self) -> &HardwareConfig {
        &self.config
    }

    // ----- kernel generation ------------------------------------------------

    /// Generate GPU kernel source code for `test`.
    pub fn generate_kernel(&self, test: &LitmusTest) -> Result<String, RunnerError> {
        let source = match self.config.gpu_backend {
            GpuBackend::Cuda => {
                let gen = CudaKernelGenerator::new(
                    self.config.thread_count,
                    self.config.workgroup_count,
                    self.config.iterations,
                );
                gen.generate_source(test)
            }
            GpuBackend::OpenCL => {
                let gen = OpenClKernelGenerator::new(
                    self.config.thread_count,
                    self.config.workgroup_count,
                    self.config.iterations,
                );
                gen.generate_source(test)
            }
            GpuBackend::Vulkan => {
                let gen = VulkanShaderGenerator::new(
                    self.config.thread_count,
                    self.config.workgroup_count,
                    self.config.iterations,
                );
                gen.generate_source(test)
            }
            GpuBackend::Metal => {
                let gen = MetalKernelGenerator::new(
                    self.config.thread_count,
                    self.config.workgroup_count,
                    self.config.iterations,
                );
                gen.generate_source(test)
            }
        };
        Ok(source)
    }

    /// Write kernel source to a file under `work_dir` and return its path.
    pub fn write_kernel(
        &self,
        test: &LitmusTest,
    ) -> Result<PathBuf, RunnerError> {
        let source = self.generate_kernel(test)?;
        std::fs::create_dir_all(&self.config.work_dir)?;

        let filename = format!(
            "{}.{}",
            sanitize_name(&test.name),
            self.config.gpu_backend.source_extension()
        );
        let path = self.config.work_dir.join(&filename);
        std::fs::write(&path, &source)?;
        Ok(path)
    }

    // ----- compilation ------------------------------------------------------

    /// Compile a previously-written kernel source file.
    pub fn compile_kernel(&self, source_path: &Path) -> Result<CompilationResult, RunnerError> {
        let output_path = source_path.with_extension("bin");
        let compiler = self.config.gpu_backend.compiler();

        let mut cmd = Command::new(compiler);
        for flag in self.config.gpu_backend.default_flags() {
            cmd.arg(flag);
        }
        for flag in &self.config.extra_flags {
            cmd.arg(flag);
        }

        match self.config.gpu_backend {
            GpuBackend::Cuda => {
                cmd.arg("-o").arg(&output_path).arg(source_path);
            }
            GpuBackend::OpenCL => {
                cmd.arg("-o").arg(&output_path).arg(source_path);
            }
            GpuBackend::Vulkan => {
                cmd.arg("-o").arg(&output_path).arg(source_path);
            }
            GpuBackend::Metal => {
                cmd.arg("-o").arg(&output_path).arg(source_path);
            }
        }

        let start = Instant::now();
        let output = cmd.output().map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                RunnerError::BackendUnavailable(self.config.gpu_backend)
            } else {
                RunnerError::Io(e.to_string())
            }
        })?;
        let duration = start.elapsed();

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        if !output.status.success() {
            return Err(RunnerError::CompilationFailed {
                stderr,
                exit_code: output.status.code(),
            });
        }

        Ok(CompilationResult {
            binary_path: output_path,
            stdout,
            stderr,
            duration,
        })
    }

    // ----- execution --------------------------------------------------------

    /// Run a compiled test binary and collect raw stdout.
    pub fn run_binary(&self, binary_path: &Path) -> Result<String, RunnerError> {
        let mut cmd = Command::new(binary_path);
        cmd.arg("--iterations")
            .arg(self.config.iterations.to_string())
            .arg("--threads")
            .arg(self.config.thread_count.to_string())
            .arg("--workgroups")
            .arg(self.config.workgroup_count.to_string());

        if self.config.seed != 0 {
            cmd.arg("--seed").arg(self.config.seed.to_string());
        }

        let start = Instant::now();
        let child = cmd
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()?;

        // Wait with timeout.
        let output = wait_with_timeout(child, self.config.timeout)?;
        let elapsed = start.elapsed();

        if elapsed > self.config.timeout {
            return Err(RunnerError::Timeout { elapsed });
        }

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            return Err(RunnerError::ExecutionFailed(stderr));
        }

        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

    // ----- result collection ------------------------------------------------

    /// Parse the raw stdout of a test run into a `TestOutcome`.
    pub fn collect_results(
        &self,
        test_name: &str,
        raw_output: &str,
    ) -> Result<TestOutcome, RunnerError> {
        let mut outcome = TestOutcome::new(test_name, self.config.gpu_backend);
        parse_output_into_histogram(raw_output, &mut outcome)?;
        Ok(outcome)
    }

    // ----- end-to-end -------------------------------------------------------

    /// Generate, compile, run, and collect results for a single litmus test.
    pub fn run_test(&self, test: &LitmusTest) -> Result<TestOutcome, RunnerError> {
        let source_path = self.write_kernel(test)?;
        let compiled = self.compile_kernel(&source_path)?;
        let raw = self.run_binary(&compiled.binary_path)?;
        let mut result = self.collect_results(&test.name, &raw)?;

        if !self.config.keep_intermediates {
            let _ = std::fs::remove_file(&source_path);
            let _ = std::fs::remove_file(&compiled.binary_path);
        }

        Ok(result)
    }

    /// Run a test and compare against model predictions.
    pub fn run_and_validate(
        &self,
        test: &LitmusTest,
    ) -> Result<HardwareResult, RunnerError> {
        let observed = self.run_test(test)?;
        Ok(HardwareResult::from_test(test, observed))
    }

    /// Run multiple tests, returning results for each.
    pub fn run_batch(
        &self,
        tests: &[LitmusTest],
    ) -> Vec<Result<HardwareResult, RunnerError>> {
        tests.iter().map(|t| self.run_and_validate(t)).collect()
    }

    /// Check whether the configured backend is available.
    pub fn check_backend_available(&self) -> Result<(), RunnerError> {
        let compiler = self.config.gpu_backend.compiler();
        let result = Command::new("which").arg(compiler).output();
        match result {
            Ok(o) if o.status.success() => Ok(()),
            _ => Err(RunnerError::BackendUnavailable(self.config.gpu_backend)),
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Wait for a child process with a timeout.
fn wait_with_timeout(
    mut child: std::process::Child,
    timeout: Duration,
) -> Result<std::process::Output, RunnerError> {
    let start = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let stdout = child
                    .stdout
                    .take()
                    .map(|mut s| {
                        let mut buf = Vec::new();
                        std::io::Read::read_to_end(&mut s, &mut buf).ok();
                        buf
                    })
                    .unwrap_or_default();
                let stderr = child
                    .stderr
                    .take()
                    .map(|mut s| {
                        let mut buf = Vec::new();
                        std::io::Read::read_to_end(&mut s, &mut buf).ok();
                        buf
                    })
                    .unwrap_or_default();
                return Ok(std::process::Output {
                    status,
                    stdout,
                    stderr,
                });
            }
            Ok(None) => {
                if start.elapsed() > timeout {
                    let _ = child.kill();
                    return Err(RunnerError::Timeout {
                        elapsed: start.elapsed(),
                    });
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(e) => return Err(RunnerError::Io(e.to_string())),
        }
    }
}

/// Sanitize a test name for use as a filename.
fn sanitize_name(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_alphanumeric() || c == '_' { c } else { '_' })
        .collect()
}

/// Parse structured test output into a histogram.
///
/// Expected format per line:
/// ```text
/// OUTCOME T0:r0=1,T1:r0=0,[0]=1,[1]=0 COUNT 42
/// ```
fn parse_output_into_histogram(
    raw: &str,
    outcome: &mut TestOutcome,
) -> Result<(), RunnerError> {
    for line in raw.lines() {
        let line = line.trim();
        if !line.starts_with("OUTCOME") {
            continue;
        }

        let parts: Vec<&str> = line.splitn(4, ' ').collect();
        if parts.len() < 4 {
            return Err(RunnerError::ResultParsing(format!(
                "malformed outcome line: {}",
                line
            )));
        }

        let outcome_str = parts[1];
        let count: u64 = parts[3].parse().map_err(|_| {
            RunnerError::ResultParsing(format!("invalid count in: {}", line))
        })?;

        let observed = parse_outcome_string(outcome_str)?;
        *outcome.histogram.entry(observed).or_insert(0) += count;
        outcome.total_iterations += count;
    }
    Ok(())
}

/// Parse an outcome string like `T0:r0=1,T1:r0=0,[0]=1`.
fn parse_outcome_string(s: &str) -> Result<ObservedOutcome, RunnerError> {
    let mut obs = ObservedOutcome::new();
    for part in s.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if part.starts_with('T') {
            // Thread register: T0:r0=1
            let rest = &part[1..];
            let colon = rest.find(':').ok_or_else(|| {
                RunnerError::ResultParsing(format!("missing ':' in {}", part))
            })?;
            let tid: usize = rest[..colon].parse().map_err(|_| {
                RunnerError::ResultParsing(format!("bad thread id in {}", part))
            })?;
            let reg_val = &rest[colon + 1..];
            if !reg_val.starts_with('r') {
                return Err(RunnerError::ResultParsing(format!(
                    "expected 'r' in {}",
                    part
                )));
            }
            let eq = reg_val.find('=').ok_or_else(|| {
                RunnerError::ResultParsing(format!("missing '=' in {}", part))
            })?;
            let rid: usize = reg_val[1..eq].parse().map_err(|_| {
                RunnerError::ResultParsing(format!("bad reg id in {}", part))
            })?;
            let val: Value = reg_val[eq + 1..].parse().map_err(|_| {
                RunnerError::ResultParsing(format!("bad value in {}", part))
            })?;
            obs.registers.insert((tid, rid), val);
        } else if part.starts_with('[') {
            // Memory: [0]=1
            let bracket = part.find(']').ok_or_else(|| {
                RunnerError::ResultParsing(format!("missing ']' in {}", part))
            })?;
            let addr: Address = part[1..bracket].parse().map_err(|_| {
                RunnerError::ResultParsing(format!("bad address in {}", part))
            })?;
            let eq = part.find('=').ok_or_else(|| {
                RunnerError::ResultParsing(format!("missing '=' in {}", part))
            })?;
            let val: Value = part[eq + 1..].parse().map_err(|_| {
                RunnerError::ResultParsing(format!("bad value in {}", part))
            })?;
            obs.memory.insert(addr, val);
        } else {
            return Err(RunnerError::ResultParsing(format!(
                "unexpected token: {}",
                part
            )));
        }
    }
    Ok(obs)
}

// ---------------------------------------------------------------------------
// Batch runner with retries
// ---------------------------------------------------------------------------

/// Options for batch test execution with retry logic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchRunConfig {
    /// Maximum retries per test on transient failures.
    pub max_retries: u32,
    /// Delay between retries.
    pub retry_delay: Duration,
    /// Whether to stop on first failure.
    pub fail_fast: bool,
}

impl Default for BatchRunConfig {
    fn default() -> Self {
        Self {
            max_retries: 2,
            retry_delay: Duration::from_secs(1),
            fail_fast: false,
        }
    }
}

/// Run a batch of tests with retry logic.
pub fn run_batch_with_retries(
    runner: &LitmusRunner,
    tests: &[LitmusTest],
    batch_config: &BatchRunConfig,
) -> Vec<Result<HardwareResult, RunnerError>> {
    let mut results = Vec::with_capacity(tests.len());

    for test in tests {
        let mut last_err = None;
        let mut succeeded = false;

        for attempt in 0..=batch_config.max_retries {
            if attempt > 0 {
                std::thread::sleep(batch_config.retry_delay);
            }
            match runner.run_and_validate(test) {
                Ok(r) => {
                    results.push(Ok(r));
                    succeeded = true;
                    break;
                }
                Err(e) => {
                    last_err = Some(e);
                }
            }
        }

        if !succeeded {
            let err = last_err.unwrap();
            if batch_config.fail_fast {
                results.push(Err(err));
                return results;
            }
            results.push(Err(err));
        }
    }

    results
}

// ---------------------------------------------------------------------------
// Summary reporting
// ---------------------------------------------------------------------------

/// Summary statistics for a batch run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchSummary {
    pub total_tests: usize,
    pub passed: usize,
    pub failed: usize,
    pub errors: usize,
    pub total_iterations: u64,
    pub total_violations: usize,
    pub total_duration: Duration,
}

impl BatchSummary {
    pub fn from_results(results: &[Result<HardwareResult, RunnerError>]) -> Self {
        let mut summary = Self {
            total_tests: results.len(),
            passed: 0,
            failed: 0,
            errors: 0,
            total_iterations: 0,
            total_violations: 0,
            total_duration: Duration::ZERO,
        };

        for r in results {
            match r {
                Ok(hr) => {
                    summary.total_iterations += hr.observed.total_iterations;
                    summary.total_duration += hr.observed.duration;
                    if hr.consistent {
                        summary.passed += 1;
                    } else {
                        summary.failed += 1;
                        summary.total_violations += hr.violation_count();
                    }
                }
                Err(_) => {
                    summary.errors += 1;
                }
            }
        }

        summary
    }

    pub fn display(&self) -> String {
        format!(
            "Tests: {} total, {} passed, {} failed, {} errors | \
             Iterations: {} | Violations: {} | Duration: {:.1}s",
            self.total_tests,
            self.passed,
            self.failed,
            self.errors,
            self.total_iterations,
            self.total_violations,
            self.total_duration.as_secs_f64(),
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_simple_test() -> LitmusTest {
        let t0 = Thread {
            id: 0,
            instructions: vec![
                Instruction::Store {
                    addr: 0,
                    value: 1,
                    ordering: Ordering::Relaxed,
                },
            ],
        };
        let t1 = Thread {
            id: 1,
            instructions: vec![
                Instruction::Load {
                    reg: 0,
                    addr: 0,
                    ordering: Ordering::Relaxed,
                },
            ],
        };

        let mut initial = HashMap::new();
        initial.insert(0u64, 0u64);

        let allowed = Outcome {
            registers: {
                let mut m = HashMap::new();
                m.insert((1, 0), 0);
                m
            },
            memory: HashMap::new(),
        };
        let also_allowed = Outcome {
            registers: {
                let mut m = HashMap::new();
                m.insert((1, 0), 1);
                m
            },
            memory: HashMap::new(),
        };

        LitmusTest {
            name: "store_load".into(),
            threads: vec![t0, t1],
            initial_state: initial,
            expected_outcomes: vec![
                (allowed, LitmusOutcome::Allowed),
                (also_allowed, LitmusOutcome::Allowed),
            ],
        }
    }

    #[test]
    fn test_hardware_config_default() {
        let cfg = HardwareConfig::default();
        assert_eq!(cfg.iterations, 100_000);
        assert_eq!(cfg.gpu_backend, GpuBackend::Cuda);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_hardware_config_validation() {
        let cfg = HardwareConfig {
            iterations: 0,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_hardware_config_builder() {
        let cfg = HardwareConfig::new(GpuBackend::Vulkan)
            .with_iterations(50_000)
            .with_thread_count(128)
            .with_workgroup_count(4)
            .with_stress_mode(StressMode::Heavy)
            .with_seed(42);
        assert_eq!(cfg.gpu_backend, GpuBackend::Vulkan);
        assert_eq!(cfg.iterations, 50_000);
        assert_eq!(cfg.total_threads(), 512);
    }

    #[test]
    fn test_observed_outcome_matches() {
        let obs = ObservedOutcome::new()
            .with_register(0, 0, 1)
            .with_memory(100, 42);

        let expected = Outcome {
            registers: {
                let mut m = HashMap::new();
                m.insert((0, 0), 1);
                m
            },
            memory: {
                let mut m = HashMap::new();
                m.insert(100, 42);
                m
            },
        };
        assert!(obs.matches(&expected));
    }

    #[test]
    fn test_observed_outcome_no_match() {
        let obs = ObservedOutcome::new().with_register(0, 0, 1);
        let expected = Outcome {
            registers: {
                let mut m = HashMap::new();
                m.insert((0, 0), 2);
                m
            },
            memory: HashMap::new(),
        };
        assert!(!obs.matches(&expected));
    }

    #[test]
    fn test_outcome_display_key() {
        let obs = ObservedOutcome::new()
            .with_register(0, 0, 1)
            .with_register(1, 0, 0)
            .with_memory(0, 1);
        let key = obs.display_key();
        assert!(key.contains("T0:r0=1"));
        assert!(key.contains("T1:r0=0"));
        assert!(key.contains("[0]=1"));
    }

    #[test]
    fn test_test_outcome_histogram() {
        let mut to = TestOutcome::new("test", GpuBackend::Cuda);
        let o1 = ObservedOutcome::new().with_register(0, 0, 1);
        let o2 = ObservedOutcome::new().with_register(0, 0, 0);

        to.record(o1.clone());
        to.record(o1.clone());
        to.record(o2.clone());

        assert_eq!(to.total_iterations, 3);
        assert_eq!(to.distinct_outcomes(), 2);
        assert_eq!(to.frequency(&o1), 2);
        assert_eq!(to.frequency(&o2), 1);
        assert!((to.fraction(&o1) - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_test_outcome_merge() {
        let mut to1 = TestOutcome::new("t", GpuBackend::Cuda);
        let mut to2 = TestOutcome::new("t", GpuBackend::Cuda);
        let o = ObservedOutcome::new().with_register(0, 0, 1);

        to1.record(o.clone());
        to2.record(o.clone());
        to2.record(o.clone());

        to1.merge(&to2);
        assert_eq!(to1.total_iterations, 3);
        assert_eq!(to1.frequency(&o), 3);
    }

    #[test]
    fn test_hardware_result_consistent() {
        let test = make_simple_test();
        let mut observed = TestOutcome::new("store_load", GpuBackend::Cuda);
        let o = ObservedOutcome::new().with_register(1, 0, 0);
        observed.record(o);

        let result = HardwareResult::from_test(&test, observed);
        assert!(result.consistent);
        assert_eq!(result.violation_count(), 0);
    }

    #[test]
    fn test_hardware_result_violation() {
        let test = make_simple_test();
        let mut observed = TestOutcome::new("store_load", GpuBackend::Cuda);
        // Unexpected outcome not in allowed list.
        let o = ObservedOutcome::new().with_register(1, 0, 99);
        observed.record(o);

        let result = HardwareResult::from_test(&test, observed);
        assert!(!result.consistent);
        assert!(result.violation_count() > 0);
    }

    #[test]
    fn test_parse_outcome_string() {
        let obs = parse_outcome_string("T0:r0=1,T1:r0=0,[0]=1").unwrap();
        assert_eq!(obs.registers.get(&(0, 0)), Some(&1));
        assert_eq!(obs.registers.get(&(1, 0)), Some(&0));
        assert_eq!(obs.memory.get(&0), Some(&1));
    }

    #[test]
    fn test_parse_output_histogram() {
        let raw = "\
OUTCOME T0:r0=1,T1:r0=0 COUNT 50
OUTCOME T0:r0=0,T1:r0=0 COUNT 30
# comment line
OUTCOME T0:r0=1,T1:r0=1 COUNT 20
";
        let mut to = TestOutcome::new("t", GpuBackend::Cuda);
        parse_output_into_histogram(raw, &mut to).unwrap();
        assert_eq!(to.total_iterations, 100);
        assert_eq!(to.distinct_outcomes(), 3);
    }

    #[test]
    fn test_sanitize_name() {
        assert_eq!(sanitize_name("MP+pos"), "MP_pos");
        assert_eq!(sanitize_name("test 1.2"), "test_1_2");
    }

    #[test]
    fn test_gpu_backend_display() {
        assert_eq!(format!("{}", GpuBackend::Cuda), "CUDA");
        assert_eq!(GpuBackend::OpenCL.source_extension(), "cl");
        assert_eq!(GpuBackend::Vulkan.compiler(), "glslangValidator");
    }

    #[test]
    fn test_outcome_classification_violation() {
        assert!(OutcomeClassification::ForbiddenObserved.is_violation());
        assert!(OutcomeClassification::RequiredNotObserved.is_violation());
        assert!(!OutcomeClassification::AllowedObserved.is_violation());
        assert!(!OutcomeClassification::ForbiddenNotObserved.is_violation());
    }

    #[test]
    fn test_runner_creation() {
        let cfg = HardwareConfig::default();
        let runner = LitmusRunner::new(cfg);
        assert!(runner.is_ok());
    }

    #[test]
    fn test_runner_generate_kernel() {
        let cfg = HardwareConfig::new(GpuBackend::Cuda).with_iterations(1000);
        let runner = LitmusRunner::new(cfg).unwrap();
        let test = make_simple_test();
        let source = runner.generate_kernel(&test).unwrap();
        assert!(source.contains("__global__"));
    }

    #[test]
    fn test_batch_summary() {
        let test = make_simple_test();
        let mut obs = TestOutcome::new("t", GpuBackend::Cuda);
        obs.record(ObservedOutcome::new().with_register(1, 0, 0));
        let hr = HardwareResult::from_test(&test, obs);

        let results: Vec<Result<HardwareResult, RunnerError>> = vec![
            Ok(hr),
            Err(RunnerError::ExecutionFailed("boom".into())),
        ];
        let summary = BatchSummary::from_results(&results);
        assert_eq!(summary.total_tests, 2);
        assert_eq!(summary.passed, 1);
        assert_eq!(summary.errors, 1);
    }

    #[test]
    fn test_stress_mode_default() {
        let m = StressMode::default();
        assert_eq!(m, StressMode::None);
    }

    #[test]
    fn test_batch_run_config_default() {
        let bc = BatchRunConfig::default();
        assert_eq!(bc.max_retries, 2);
        assert!(!bc.fail_fast);
    }
}
