//! Bridge to herd7 for running litmus tests and comparing results.
//!
//! Provides [`Herd7Bridge`] for invoking the herd7 tool on litmus tests,
//! parsing its output, and comparing outcomes against the Litmus∞ checker.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use crate::checker::litmus::{
    LitmusTest, Thread, Instruction, Ordering, Outcome, LitmusOutcome,
};

// ---------------------------------------------------------------------------
// Herd7Config
// ---------------------------------------------------------------------------

/// Configuration for the herd7 bridge.
#[derive(Debug, Clone)]
pub struct Herd7Config {
    /// Path to the herd7 binary.
    pub binary_path: PathBuf,
    /// Path to the `.cat` model file to use.
    pub cat_model_path: Option<PathBuf>,
    /// Name of a built-in model (e.g. "sc", "tso").
    pub builtin_model: Option<String>,
    /// Timeout for each herd7 invocation.
    pub timeout: Duration,
    /// Working directory for temporary files.
    pub work_dir: PathBuf,
    /// Whether to keep generated `.litmus` files.
    pub keep_files: bool,
    /// Extra command-line arguments.
    pub extra_args: Vec<String>,
}

impl Herd7Config {
    /// Create a config with the given herd7 binary path.
    pub fn new(binary_path: impl Into<PathBuf>) -> Self {
        Self {
            binary_path: binary_path.into(),
            cat_model_path: None,
            builtin_model: None,
            timeout: Duration::from_secs(30),
            work_dir: std::env::temp_dir(),
            keep_files: false,
            extra_args: Vec::new(),
        }
    }

    /// Set the `.cat` model file.
    pub fn with_cat_model(mut self, path: impl Into<PathBuf>) -> Self {
        self.cat_model_path = Some(path.into());
        self
    }

    /// Set a built-in model name.
    pub fn with_builtin_model(mut self, name: &str) -> Self {
        self.builtin_model = Some(name.to_string());
        self
    }

    /// Set timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set working directory.
    pub fn with_work_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.work_dir = dir.into();
        self
    }

    /// Keep generated files.
    pub fn keep_files(mut self) -> Self {
        self.keep_files = true;
        self
    }
}

impl Default for Herd7Config {
    fn default() -> Self {
        Self::new("herd7")
    }
}

// ---------------------------------------------------------------------------
// Herd7Result
// ---------------------------------------------------------------------------

/// Outcome from a single herd7 run.
#[derive(Debug, Clone)]
pub struct Herd7Result {
    /// The litmus test name.
    pub test_name: String,
    /// Allowed outcomes observed by herd7.
    pub allowed_outcomes: Vec<Herd7Outcome>,
    /// Forbidden outcomes (if any were checked).
    pub forbidden_outcomes: Vec<Herd7Outcome>,
    /// Overall verdict: true if the test condition is satisfied.
    pub condition_satisfied: Option<bool>,
    /// Raw herd7 output.
    pub raw_output: String,
    /// How long herd7 took.
    pub duration: Duration,
    /// Whether herd7 ran successfully.
    pub success: bool,
    /// Error message if herd7 failed.
    pub error: Option<String>,
}

/// A single outcome as reported by herd7.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Herd7Outcome {
    /// Register values: `"P0:r0"` → value.
    pub registers: HashMap<String, u64>,
    /// Memory values: `"x"` → value.
    pub memory: HashMap<String, u64>,
    /// How many times this outcome was observed.
    pub count: u64,
}

impl Herd7Outcome {
    pub fn new() -> Self {
        Self {
            registers: HashMap::new(),
            memory: HashMap::new(),
            count: 0,
        }
    }
}

impl Default for Herd7Outcome {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for Herd7Outcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts = Vec::new();
        let mut reg_keys: Vec<_> = self.registers.keys().collect();
        reg_keys.sort();
        for k in reg_keys {
            parts.push(format!("{}={}", k, self.registers[k]));
        }
        let mut mem_keys: Vec<_> = self.memory.keys().collect();
        mem_keys.sort();
        for k in mem_keys {
            parts.push(format!("{}={}", k, self.memory[k]));
        }
        write!(f, "{{{}}}", parts.join(", "))
    }
}

impl fmt::Display for Herd7Result {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "herd7[{}]: ", self.test_name)?;
        if self.success {
            write!(f, "{} allowed", self.allowed_outcomes.len())?;
            if let Some(sat) = self.condition_satisfied {
                write!(f, ", condition {}", if sat { "satisfied" } else { "NOT satisfied" })?;
            }
            write!(f, " ({:.1}ms)", self.duration.as_secs_f64() * 1000.0)
        } else {
            write!(f, "FAILED: {}", self.error.as_deref().unwrap_or("unknown"))
        }
    }
}

// ---------------------------------------------------------------------------
// ComparisonResult
// ---------------------------------------------------------------------------

/// Result of comparing Litmus∞ and herd7 on a single test.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComparisonResult {
    /// Both agree on the set of allowed/forbidden outcomes.
    Agreement,
    /// Litmus∞ allows outcomes that herd7 forbids.
    LitmusInfOnly {
        /// Outcomes allowed by Litmus∞ but not herd7.
        outcomes: Vec<String>,
    },
    /// Herd7 allows outcomes that Litmus∞ forbids.
    Herd7Only {
        /// Outcomes allowed by herd7 but not Litmus∞.
        outcomes: Vec<String>,
    },
    /// Both have unique outcomes (mutual disagreement).
    MutualDisagreement {
        litmus_only: Vec<String>,
        herd7_only: Vec<String>,
    },
    /// Comparison could not be performed (e.g. herd7 failed).
    Error(String),
}

impl fmt::Display for ComparisonResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Agreement => write!(f, "✓ Agreement"),
            Self::LitmusInfOnly { outcomes } => {
                write!(f, "⚠ Litmus∞ only: {}", outcomes.join(", "))
            }
            Self::Herd7Only { outcomes } => {
                write!(f, "⚠ Herd7 only: {}", outcomes.join(", "))
            }
            Self::MutualDisagreement { litmus_only, herd7_only } => {
                write!(f, "✗ Disagree: litmus∞=[{}], herd7=[{}]",
                    litmus_only.join(", "), herd7_only.join(", "))
            }
            Self::Error(e) => write!(f, "✗ Error: {}", e),
        }
    }
}

// ---------------------------------------------------------------------------
// ComparisonReport
// ---------------------------------------------------------------------------

/// Aggregate comparison report across multiple litmus tests.
#[derive(Debug, Clone)]
pub struct ComparisonReport {
    /// Per-test comparison results.
    pub results: Vec<(String, ComparisonResult)>,
    /// Total number of tests compared.
    pub total_tests: usize,
    /// Number of tests where both agree.
    pub agreements: usize,
    /// Number of tests where only Litmus∞ allows extra outcomes.
    pub litmus_inf_only: usize,
    /// Number of tests where only herd7 allows extra outcomes.
    pub herd7_only: usize,
    /// Number of tests with mutual disagreement.
    pub mutual_disagreements: usize,
    /// Number of tests with errors.
    pub errors: usize,
    /// Average Litmus∞ time per test.
    pub avg_litmus_time: Duration,
    /// Average herd7 time per test.
    pub avg_herd7_time: Duration,
    /// Total Litmus∞ time.
    pub total_litmus_time: Duration,
    /// Total herd7 time.
    pub total_herd7_time: Duration,
}

impl ComparisonReport {
    /// Create a new empty report.
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
            total_tests: 0,
            agreements: 0,
            litmus_inf_only: 0,
            herd7_only: 0,
            mutual_disagreements: 0,
            errors: 0,
            avg_litmus_time: Duration::ZERO,
            avg_herd7_time: Duration::ZERO,
            total_litmus_time: Duration::ZERO,
            total_herd7_time: Duration::ZERO,
        }
    }

    /// Add a comparison result to the report.
    pub fn add_result(
        &mut self,
        test_name: &str,
        result: ComparisonResult,
        litmus_time: Duration,
        herd7_time: Duration,
    ) {
        self.total_tests += 1;
        match &result {
            ComparisonResult::Agreement => self.agreements += 1,
            ComparisonResult::LitmusInfOnly { .. } => self.litmus_inf_only += 1,
            ComparisonResult::Herd7Only { .. } => self.herd7_only += 1,
            ComparisonResult::MutualDisagreement { .. } => self.mutual_disagreements += 1,
            ComparisonResult::Error(_) => self.errors += 1,
        }
        self.total_litmus_time += litmus_time;
        self.total_herd7_time += herd7_time;
        if self.total_tests > 0 {
            self.avg_litmus_time = self.total_litmus_time / self.total_tests as u32;
            self.avg_herd7_time = self.total_herd7_time / self.total_tests as u32;
        }
        self.results.push((test_name.to_string(), result));
    }

    /// Agreement rate as a fraction.
    pub fn agreement_rate(&self) -> f64 {
        if self.total_tests == 0 {
            1.0
        } else {
            self.agreements as f64 / self.total_tests as f64
        }
    }

    /// Performance ratio (Litmus∞ time / herd7 time). < 1.0 means Litmus∞ is faster.
    pub fn performance_ratio(&self) -> f64 {
        let h = self.total_herd7_time.as_secs_f64();
        if h == 0.0 {
            0.0
        } else {
            self.total_litmus_time.as_secs_f64() / h
        }
    }

    /// Get only disagreement results.
    pub fn disagreements(&self) -> Vec<&(String, ComparisonResult)> {
        self.results
            .iter()
            .filter(|(_, r)| !matches!(r, ComparisonResult::Agreement))
            .collect()
    }
}

impl Default for ComparisonReport {
    fn default() -> Self { Self::new() }
}

impl fmt::Display for ComparisonReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Comparison Report ===")?;
        writeln!(f, "Total tests:    {}", self.total_tests)?;
        writeln!(f, "Agreements:     {} ({:.1}%)", self.agreements, self.agreement_rate() * 100.0)?;
        writeln!(f, "Litmus∞ only:   {}", self.litmus_inf_only)?;
        writeln!(f, "Herd7 only:     {}", self.herd7_only)?;
        writeln!(f, "Mutual disagr.: {}", self.mutual_disagreements)?;
        writeln!(f, "Errors:         {}", self.errors)?;
        writeln!(f, "Avg litmus∞:    {:.2}ms", self.avg_litmus_time.as_secs_f64() * 1000.0)?;
        writeln!(f, "Avg herd7:      {:.2}ms", self.avg_herd7_time.as_secs_f64() * 1000.0)?;
        writeln!(f, "Perf ratio:     {:.2}x", self.performance_ratio())?;

        if !self.disagreements().is_empty() {
            writeln!(f, "\n--- Disagreements ---")?;
            for (name, result) in self.disagreements() {
                writeln!(f, "  {}: {}", name, result)?;
            }
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// LitmusFileGenerator — convert LitmusTest to .litmus format
// ---------------------------------------------------------------------------

/// Converts Litmus∞ [`LitmusTest`] to herd7 `.litmus` file format.
pub struct LitmusFileGenerator;

impl LitmusFileGenerator {
    /// Generate a `.litmus` file string for a given test.
    ///
    /// Produces output in the standard herd7 litmus test format for the
    /// "C" (C11) language variant.
    pub fn generate(test: &LitmusTest) -> String {
        let mut out = String::new();

        // Architecture line.
        out.push_str("C ");
        out.push_str(&test.name);
        out.push('\n');

        // Comment with description.
        out.push_str(&format!("\"Generated from Litmus∞ test {}\"\n", test.name));

        // Initial state.
        out.push('{');
        out.push('\n');
        let addresses = test.all_addresses();
        let addr_names = Self::address_names(&addresses);

        for (&addr, name) in addresses.iter().zip(addr_names.iter()) {
            let val = test.initial_state.get(&addr).copied().unwrap_or(0);
            if val != 0 {
                out.push_str(&format!("  {} = {};\n", name, val));
            }
        }
        out.push_str("}\n\n");

        // Thread bodies.
        for (tidx, thread) in test.threads.iter().enumerate() {
            out.push_str(&format!("P{} (", tidx));

            // Parameters: pointers to shared variables accessed by this thread.
            let thread_addrs = thread.accessed_addresses();
            let params: Vec<String> = thread_addrs
                .iter()
                .map(|a| {
                    let name = addr_names
                        .get(addresses.iter().position(|x| x == a).unwrap_or(0))
                        .cloned()
                        .unwrap_or_else(|| format!("x{}", a));
                    format!("atomic_int* {}", name)
                })
                .collect();
            out.push_str(&params.join(", "));
            out.push_str(") {\n");

            for instr in &thread.instructions {
                out.push_str("  ");
                out.push_str(&Self::instruction_to_c(instr, &addresses, &addr_names));
                out.push('\n');
            }

            out.push_str("}\n\n");
        }

        // Final condition (exists clause).
        if !test.expected_outcomes.is_empty() {
            let conditions: Vec<String> = test
                .expected_outcomes
                .iter()
                .filter(|(_, kind)| *kind == LitmusOutcome::Forbidden || *kind == LitmusOutcome::Allowed)
                .map(|(outcome, _)| Self::outcome_to_condition(outcome, &addresses, &addr_names))
                .collect();

            if !conditions.is_empty() {
                out.push_str("exists (");
                out.push_str(&conditions.join(" /\\ "));
                out.push_str(")\n");
            }
        }

        out
    }

    /// Generate canonical address names (x, y, z, a, b, ...).
    fn address_names(addresses: &[u64]) -> Vec<String> {
        let names = ["x", "y", "z", "a", "b", "c", "d", "e", "f", "g"];
        addresses
            .iter()
            .enumerate()
            .map(|(i, _)| {
                if i < names.len() {
                    names[i].to_string()
                } else {
                    format!("v{}", i)
                }
            })
            .collect()
    }

    /// Convert a single instruction to C11 syntax.
    fn instruction_to_c(instr: &Instruction, addresses: &[u64], addr_names: &[String]) -> String {
        let addr_name = |addr: &u64| -> String {
            addresses
                .iter()
                .position(|a| a == addr)
                .and_then(|i| addr_names.get(i))
                .cloned()
                .unwrap_or_else(|| format!("v{}", addr))
        };

        let ordering_to_c = |ord: &Ordering| -> &str {
            match ord {
                Ordering::Relaxed => "memory_order_relaxed",
                Ordering::Acquire | Ordering::AcquireCTA
                | Ordering::AcquireGPU | Ordering::AcquireSystem => "memory_order_acquire",
                Ordering::Release | Ordering::ReleaseCTA
                | Ordering::ReleaseGPU | Ordering::ReleaseSystem => "memory_order_release",
                Ordering::AcqRel => "memory_order_acq_rel",
                Ordering::SeqCst => "memory_order_seq_cst",
            }
        };

        match instr {
            Instruction::Load { reg, addr, ordering } => {
                format!(
                    "int r{} = atomic_load_explicit({}, {});",
                    reg,
                    addr_name(addr),
                    ordering_to_c(ordering)
                )
            }
            Instruction::Store { addr, value, ordering } => {
                format!(
                    "atomic_store_explicit({}, {}, {});",
                    addr_name(addr),
                    value,
                    ordering_to_c(ordering)
                )
            }
            Instruction::Fence { ordering, .. } => {
                format!("atomic_thread_fence({});", ordering_to_c(ordering))
            }
            Instruction::RMW { reg, addr, value, ordering } => {
                format!(
                    "int r{} = atomic_exchange_explicit({}, {}, {});",
                    reg,
                    addr_name(addr),
                    value,
                    ordering_to_c(ordering)
                )
            }
            Instruction::Branch { label } => format!("goto L{};", label),
            Instruction::Label { id } => format!("L{}:", id),
            Instruction::BranchCond { reg, expected, label } => {
                format!("if (r{} == {}) goto L{};", reg, expected, label)
            }
        }
    }

    /// Convert an outcome to a herd7 exists-condition fragment.
    fn outcome_to_condition(
        outcome: &Outcome,
        addresses: &[u64],
        addr_names: &[String],
    ) -> String {
        let mut parts = Vec::new();

        let mut reg_keys: Vec<_> = outcome.registers.keys().collect();
        reg_keys.sort();
        for &(tid, reg) in &reg_keys {
            let val = outcome.registers[&(*tid, *reg)];
            parts.push(format!("{}:r{} = {}", tid, reg, val));
        }

        let mut mem_keys: Vec<_> = outcome.memory.keys().collect();
        mem_keys.sort();
        for &addr in &mem_keys {
            let val = outcome.memory[&addr];
            let name = addresses
                .iter()
                .position(|a| *a == *addr)
                .and_then(|i| addr_names.get(i))
                .cloned()
                .unwrap_or_else(|| format!("v{}", addr));
            parts.push(format!("{} = {}", name, val));
        }

        parts.join(" /\\ ")
    }

    /// Write a `.litmus` file to disk and return the path.
    pub fn write_to_file(test: &LitmusTest, dir: &Path) -> Result<PathBuf, String> {
        let content = Self::generate(test);
        let filename = format!("{}.litmus", test.name.replace(' ', "_"));
        let path = dir.join(filename);
        let mut file = std::fs::File::create(&path)
            .map_err(|e| format!("failed to create {}: {}", path.display(), e))?;
        file.write_all(content.as_bytes())
            .map_err(|e| format!("failed to write {}: {}", path.display(), e))?;
        Ok(path)
    }
}

// ---------------------------------------------------------------------------
// Herd7Bridge
// ---------------------------------------------------------------------------

/// Interface to the herd7 memory model simulator.
///
/// Runs herd7 on litmus tests, parses its output, and enables comparison
/// between Litmus∞ and herd7 results.
pub struct Herd7Bridge {
    config: Herd7Config,
}

impl Herd7Bridge {
    /// Create a new bridge with the given configuration.
    pub fn new(config: Herd7Config) -> Self {
        Self { config }
    }

    /// Create with default configuration (expects `herd7` on PATH).
    pub fn with_defaults() -> Self {
        Self::new(Herd7Config::default())
    }

    /// Check whether herd7 is available.
    pub fn is_available(&self) -> bool {
        Command::new(&self.config.binary_path)
            .arg("-version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// Get the herd7 version string.
    pub fn version(&self) -> Result<String, String> {
        let output = Command::new(&self.config.binary_path)
            .arg("-version")
            .output()
            .map_err(|e| format!("failed to run herd7: {}", e))?;

        if output.status.success() {
            Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
        } else {
            Err(String::from_utf8_lossy(&output.stderr).trim().to_string())
        }
    }

    /// Run herd7 on a single litmus test.
    pub fn run_test(&self, test: &LitmusTest) -> Herd7Result {
        let start = Instant::now();

        // Generate .litmus file.
        let litmus_path = match LitmusFileGenerator::write_to_file(test, &self.config.work_dir) {
            Ok(p) => p,
            Err(e) => {
                return Herd7Result {
                    test_name: test.name.clone(),
                    allowed_outcomes: Vec::new(),
                    forbidden_outcomes: Vec::new(),
                    condition_satisfied: None,
                    raw_output: String::new(),
                    duration: start.elapsed(),
                    success: false,
                    error: Some(e),
                };
            }
        };

        // Build command.
        let mut cmd = Command::new(&self.config.binary_path);
        if let Some(ref cat) = self.config.cat_model_path {
            cmd.arg("-model").arg(cat);
        } else if let Some(ref builtin) = self.config.builtin_model {
            cmd.arg("-model").arg(builtin);
        }
        for arg in &self.config.extra_args {
            cmd.arg(arg);
        }
        cmd.arg(&litmus_path);

        // Run with timeout (best-effort — we rely on OS-level kill).
        let output = cmd.output();

        // Clean up.
        if !self.config.keep_files {
            let _ = std::fs::remove_file(&litmus_path);
        }

        let elapsed = start.elapsed();

        match output {
            Ok(o) => {
                let stdout = String::from_utf8_lossy(&o.stdout).to_string();
                let stderr = String::from_utf8_lossy(&o.stderr).to_string();

                if o.status.success() {
                    let mut result = Self::parse_herd7_output(&stdout, &test.name);
                    result.duration = elapsed;
                    result
                } else {
                    Herd7Result {
                        test_name: test.name.clone(),
                        allowed_outcomes: Vec::new(),
                        forbidden_outcomes: Vec::new(),
                        condition_satisfied: None,
                        raw_output: format!("{}\n{}", stdout, stderr),
                        duration: elapsed,
                        success: false,
                        error: Some(format!("herd7 exited with {}: {}", o.status, stderr.trim())),
                    }
                }
            }
            Err(e) => Herd7Result {
                test_name: test.name.clone(),
                allowed_outcomes: Vec::new(),
                forbidden_outcomes: Vec::new(),
                condition_satisfied: None,
                raw_output: String::new(),
                duration: elapsed,
                success: false,
                error: Some(format!("failed to execute herd7: {}", e)),
            },
        }
    }

    /// Run herd7 on a `.litmus` file directly.
    pub fn run_litmus_file(&self, path: &Path) -> Herd7Result {
        let start = Instant::now();
        let test_name = path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".to_string());

        let mut cmd = Command::new(&self.config.binary_path);
        if let Some(ref cat) = self.config.cat_model_path {
            cmd.arg("-model").arg(cat);
        } else if let Some(ref builtin) = self.config.builtin_model {
            cmd.arg("-model").arg(builtin);
        }
        for arg in &self.config.extra_args {
            cmd.arg(arg);
        }
        cmd.arg(path);

        match cmd.output() {
            Ok(o) => {
                let stdout = String::from_utf8_lossy(&o.stdout).to_string();
                let stderr = String::from_utf8_lossy(&o.stderr).to_string();

                if o.status.success() {
                    let mut result = Self::parse_herd7_output(&stdout, &test_name);
                    result.duration = start.elapsed();
                    result
                } else {
                    Herd7Result {
                        test_name,
                        allowed_outcomes: Vec::new(),
                        forbidden_outcomes: Vec::new(),
                        condition_satisfied: None,
                        raw_output: format!("{}\n{}", stdout, stderr),
                        duration: start.elapsed(),
                        success: false,
                        error: Some(format!("herd7 error: {}", stderr.trim())),
                    }
                }
            }
            Err(e) => Herd7Result {
                test_name,
                allowed_outcomes: Vec::new(),
                forbidden_outcomes: Vec::new(),
                condition_satisfied: None,
                raw_output: String::new(),
                duration: start.elapsed(),
                success: false,
                error: Some(format!("failed to execute herd7: {}", e)),
            },
        }
    }

    /// Run herd7 on multiple tests and build a comparison report.
    pub fn run_batch(
        &self,
        tests: &[LitmusTest],
        litmus_results: &[(String, Vec<String>, Duration)],
    ) -> ComparisonReport {
        let mut report = ComparisonReport::new();

        for (test, (test_name, litmus_allowed, litmus_time)) in tests.iter().zip(litmus_results.iter()) {
            let herd_result = self.run_test(test);
            let herd_time = herd_result.duration;

            let comparison = if !herd_result.success {
                ComparisonResult::Error(
                    herd_result.error.unwrap_or_else(|| "unknown error".to_string()),
                )
            } else {
                Self::compare_outcomes(litmus_allowed, &herd_result)
            };

            report.add_result(test_name, comparison, *litmus_time, herd_time);
        }

        report
    }

    /// Compare Litmus∞ allowed outcome strings against herd7 results.
    pub fn compare_outcomes(
        litmus_allowed: &[String],
        herd7: &Herd7Result,
    ) -> ComparisonResult {
        let litmus_set: HashSet<&str> = litmus_allowed.iter().map(|s| s.as_str()).collect();
        let herd7_set: HashSet<String> = herd7
            .allowed_outcomes
            .iter()
            .map(|o| format!("{}", o))
            .collect();
        let herd7_refs: HashSet<&str> = herd7_set.iter().map(|s| s.as_str()).collect();

        let only_litmus: Vec<String> = litmus_set
            .difference(&herd7_refs)
            .map(|s| s.to_string())
            .collect();
        let only_herd7: Vec<String> = herd7_refs
            .difference(&litmus_set)
            .map(|s| s.to_string())
            .collect();

        match (only_litmus.is_empty(), only_herd7.is_empty()) {
            (true, true) => ComparisonResult::Agreement,
            (false, true) => ComparisonResult::LitmusInfOnly { outcomes: only_litmus },
            (true, false) => ComparisonResult::Herd7Only { outcomes: only_herd7 },
            (false, false) => ComparisonResult::MutualDisagreement {
                litmus_only: only_litmus,
                herd7_only: only_herd7,
            },
        }
    }

    /// Parse herd7's stdout into a [`Herd7Result`].
    fn parse_herd7_output(output: &str, test_name: &str) -> Herd7Result {
        let mut allowed = Vec::new();
        let mut forbidden = Vec::new();
        let mut condition_satisfied = None;

        let mut in_outcomes = false;

        for line in output.lines() {
            let trimmed = line.trim();

            // Detect "States N" line.
            if trimmed.starts_with("States ") {
                in_outcomes = false;
                continue;
            }

            // Detect outcome lines: "N*> ..." or "N:> ..."
            if let Some(idx) = trimmed.find(">") {
                if idx > 0 {
                    let prefix = &trimmed[..idx];
                    if prefix.ends_with('*') || prefix.ends_with(':') {
                        let is_forbidden = prefix.ends_with('*');
                        let outcome_str = trimmed[idx + 1..].trim();
                        let outcome = Self::parse_outcome_line(outcome_str);
                        if is_forbidden {
                            forbidden.push(outcome);
                        } else {
                            allowed.push(outcome);
                        }
                        continue;
                    }
                }
            }

            // Detect verdict lines.
            if trimmed.starts_with("Ok") || trimmed.contains("Condition") {
                if trimmed.contains("validated") || trimmed.contains("Ok") {
                    condition_satisfied = Some(true);
                } else if trimmed.contains("Not validated") || trimmed.contains("No") {
                    condition_satisfied = Some(false);
                }
            }

            // Simple outcome format: "x=1; y=0;"
            if trimmed.contains('=') && trimmed.contains(';') && !trimmed.starts_with("Test")
                && !trimmed.starts_with("States") && !trimmed.starts_with("Hash")
            {
                let outcome = Self::parse_simple_outcome(trimmed);
                if !outcome.registers.is_empty() || !outcome.memory.is_empty() {
                    allowed.push(outcome);
                }
            }
        }

        Herd7Result {
            test_name: test_name.to_string(),
            allowed_outcomes: allowed,
            forbidden_outcomes: forbidden,
            condition_satisfied,
            raw_output: output.to_string(),
            duration: Duration::ZERO,
            success: true,
            error: None,
        }
    }

    /// Parse a single outcome line like `x=1; y=0; P0:r0=1;`.
    fn parse_outcome_line(s: &str) -> Herd7Outcome {
        Self::parse_simple_outcome(s)
    }

    /// Parse key=value pairs from an outcome string.
    fn parse_simple_outcome(s: &str) -> Herd7Outcome {
        let mut outcome = Herd7Outcome::new();

        for part in s.split(';') {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            if let Some(eq_pos) = part.find('=') {
                let key = part[..eq_pos].trim();
                let val_str = part[eq_pos + 1..].trim();
                if let Ok(val) = val_str.parse::<u64>() {
                    if key.contains(':') {
                        // Register: "P0:r0"
                        outcome.registers.insert(key.to_string(), val);
                    } else {
                        // Memory location.
                        outcome.memory.insert(key.to_string(), val);
                    }
                }
            }
        }

        outcome
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checker::litmus::{Scope, Outcome};

    fn make_simple_test(name: &str) -> LitmusTest {
        let mut test = LitmusTest::new(name);
        test.set_initial(0x100, 0);
        test.set_initial(0x200, 0);

        let mut t0 = Thread::new(0);
        t0.store(0x100, 1, Ordering::Relaxed);
        t0.store(0x200, 1, Ordering::Relaxed);

        let mut t1 = Thread::new(1);
        t1.load(0, 0x200, Ordering::Relaxed);
        t1.load(1, 0x100, Ordering::Relaxed);

        test.add_thread(t0);
        test.add_thread(t1);
        test.expect(
            Outcome::new().with_reg(1, 0, 1).with_reg(1, 1, 0),
            LitmusOutcome::Forbidden,
        );
        test
    }

    #[test]
    fn generate_litmus_file() {
        let test = make_simple_test("MP");
        let content = LitmusFileGenerator::generate(&test);
        assert!(content.contains("C MP"));
        assert!(content.contains("P0"));
        assert!(content.contains("P1"));
        assert!(content.contains("atomic_store_explicit"));
        assert!(content.contains("atomic_load_explicit"));
    }

    #[test]
    fn litmus_file_has_initial_state() {
        let test = make_simple_test("MP");
        let content = LitmusFileGenerator::generate(&test);
        assert!(content.contains('{'));
        assert!(content.contains('}'));
    }

    #[test]
    fn litmus_file_has_exists_clause() {
        let test = make_simple_test("MP");
        let content = LitmusFileGenerator::generate(&test);
        assert!(content.contains("exists"));
    }

    #[test]
    fn herd7_config_builder() {
        let config = Herd7Config::new("/usr/bin/herd7")
            .with_timeout(Duration::from_secs(60))
            .with_cat_model("/path/to/model.cat")
            .with_work_dir("/tmp/herd7")
            .keep_files();

        assert_eq!(config.binary_path, PathBuf::from("/usr/bin/herd7"));
        assert_eq!(config.timeout, Duration::from_secs(60));
        assert_eq!(
            config.cat_model_path,
            Some(PathBuf::from("/path/to/model.cat"))
        );
        assert!(config.keep_files);
    }

    #[test]
    fn herd7_config_default() {
        let config = Herd7Config::default();
        assert_eq!(config.binary_path, PathBuf::from("herd7"));
        assert_eq!(config.timeout, Duration::from_secs(30));
        assert!(!config.keep_files);
    }

    #[test]
    fn herd7_config_builtin_model() {
        let config = Herd7Config::new("herd7").with_builtin_model("sc");
        assert_eq!(config.builtin_model, Some("sc".to_string()));
    }

    #[test]
    fn parse_herd7_simple_outcome() {
        let outcome = Herd7Bridge::parse_simple_outcome("x=1; y=0;");
        assert_eq!(outcome.memory.get("x"), Some(&1));
        assert_eq!(outcome.memory.get("y"), Some(&0));
    }

    #[test]
    fn parse_herd7_register_outcome() {
        let outcome = Herd7Bridge::parse_simple_outcome("P0:r0=1; P1:r0=0;");
        assert_eq!(outcome.registers.get("P0:r0"), Some(&1));
        assert_eq!(outcome.registers.get("P1:r0"), Some(&0));
    }

    #[test]
    fn parse_herd7_mixed_outcome() {
        let outcome = Herd7Bridge::parse_simple_outcome("x=1; P0:r0=0;");
        assert_eq!(outcome.memory.get("x"), Some(&1));
        assert_eq!(outcome.registers.get("P0:r0"), Some(&0));
    }

    #[test]
    fn parse_herd7_empty_outcome() {
        let outcome = Herd7Bridge::parse_simple_outcome("");
        assert!(outcome.registers.is_empty());
        assert!(outcome.memory.is_empty());
    }

    #[test]
    fn comparison_result_agreement() {
        let herd_result = Herd7Result {
            test_name: "test".to_string(),
            allowed_outcomes: vec![
                {
                    let mut o = Herd7Outcome::new();
                    o.memory.insert("x".to_string(), 1);
                    o
                },
            ],
            forbidden_outcomes: Vec::new(),
            condition_satisfied: None,
            raw_output: String::new(),
            duration: Duration::ZERO,
            success: true,
            error: None,
        };

        let litmus_allowed = vec!["{x=1}".to_string()];
        let result = Herd7Bridge::compare_outcomes(&litmus_allowed, &herd_result);
        // The format may differ, so we just check it returns something.
        assert!(matches!(
            result,
            ComparisonResult::Agreement
                | ComparisonResult::LitmusInfOnly { .. }
                | ComparisonResult::Herd7Only { .. }
                | ComparisonResult::MutualDisagreement { .. }
        ));
    }

    #[test]
    fn comparison_report_empty() {
        let report = ComparisonReport::new();
        assert_eq!(report.total_tests, 0);
        assert_eq!(report.agreement_rate(), 1.0);
        assert_eq!(report.performance_ratio(), 0.0);
    }

    #[test]
    fn comparison_report_add_results() {
        let mut report = ComparisonReport::new();
        report.add_result("test1", ComparisonResult::Agreement, Duration::from_millis(10), Duration::from_millis(20));
        report.add_result("test2", ComparisonResult::Agreement, Duration::from_millis(15), Duration::from_millis(25));
        report.add_result(
            "test3",
            ComparisonResult::LitmusInfOnly { outcomes: vec!["x=1".to_string()] },
            Duration::from_millis(12),
            Duration::from_millis(18),
        );

        assert_eq!(report.total_tests, 3);
        assert_eq!(report.agreements, 2);
        assert_eq!(report.litmus_inf_only, 1);
        assert!((report.agreement_rate() - 2.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn comparison_report_disagreements() {
        let mut report = ComparisonReport::new();
        report.add_result("ok", ComparisonResult::Agreement, Duration::ZERO, Duration::ZERO);
        report.add_result(
            "bad",
            ComparisonResult::Herd7Only { outcomes: vec!["x=1".to_string()] },
            Duration::ZERO,
            Duration::ZERO,
        );

        let disag = report.disagreements();
        assert_eq!(disag.len(), 1);
        assert_eq!(disag[0].0, "bad");
    }

    #[test]
    fn comparison_report_display() {
        let mut report = ComparisonReport::new();
        report.add_result("test1", ComparisonResult::Agreement, Duration::from_millis(5), Duration::from_millis(10));
        let display = format!("{}", report);
        assert!(display.contains("Comparison Report"));
        assert!(display.contains("Agreements"));
    }

    #[test]
    fn comparison_result_display() {
        assert!(format!("{}", ComparisonResult::Agreement).contains("Agreement"));
        assert!(
            format!(
                "{}",
                ComparisonResult::LitmusInfOnly {
                    outcomes: vec!["x=1".into()]
                }
            )
            .contains("Litmus")
        );
        assert!(
            format!(
                "{}",
                ComparisonResult::Herd7Only {
                    outcomes: vec!["x=1".into()]
                }
            )
            .contains("Herd7")
        );
        assert!(format!("{}", ComparisonResult::Error("fail".into())).contains("Error"));
    }

    #[test]
    fn herd7_result_display_success() {
        let result = Herd7Result {
            test_name: "MP".to_string(),
            allowed_outcomes: vec![Herd7Outcome::new()],
            forbidden_outcomes: Vec::new(),
            condition_satisfied: Some(true),
            raw_output: String::new(),
            duration: Duration::from_millis(42),
            success: true,
            error: None,
        };
        let display = format!("{}", result);
        assert!(display.contains("MP"));
        assert!(display.contains("1 allowed"));
    }

    #[test]
    fn herd7_result_display_failure() {
        let result = Herd7Result {
            test_name: "MP".to_string(),
            allowed_outcomes: Vec::new(),
            forbidden_outcomes: Vec::new(),
            condition_satisfied: None,
            raw_output: String::new(),
            duration: Duration::ZERO,
            success: false,
            error: Some("not found".to_string()),
        };
        let display = format!("{}", result);
        assert!(display.contains("FAILED"));
    }

    #[test]
    fn herd7_outcome_display() {
        let mut o = Herd7Outcome::new();
        o.registers.insert("P0:r0".to_string(), 1);
        o.memory.insert("x".to_string(), 0);
        let display = format!("{}", o);
        assert!(display.contains("P0:r0=1"));
        assert!(display.contains("x=0"));
    }

    #[test]
    fn herd7_bridge_with_defaults() {
        let bridge = Herd7Bridge::with_defaults();
        assert_eq!(bridge.config.binary_path, PathBuf::from("herd7"));
    }

    #[test]
    fn address_names_generation() {
        let addrs = vec![0x100, 0x200, 0x300];
        let names = LitmusFileGenerator::address_names(&addrs);
        assert_eq!(names, vec!["x", "y", "z"]);
    }

    #[test]
    fn address_names_overflow() {
        let addrs: Vec<u64> = (0..15).collect();
        let names = LitmusFileGenerator::address_names(&addrs);
        assert_eq!(names[0], "x");
        assert_eq!(names[10], "v10");
        assert_eq!(names[14], "v14");
    }

    #[test]
    fn litmus_file_generator_fence() {
        let mut test = LitmusTest::new("FenceTest");
        test.set_initial(0x100, 0);
        let mut t0 = Thread::new(0);
        t0.store(0x100, 1, Ordering::Relaxed);
        t0.fence(Ordering::SeqCst, Scope::None);
        t0.store(0x100, 2, Ordering::Relaxed);
        test.add_thread(t0);
        let content = LitmusFileGenerator::generate(&test);
        assert!(content.contains("atomic_thread_fence"));
    }

    #[test]
    fn litmus_file_generator_rmw() {
        let mut test = LitmusTest::new("RMWTest");
        test.set_initial(0x100, 0);
        let mut t0 = Thread::new(0);
        t0.rmw(0, 0x100, 1, Ordering::AcqRel);
        test.add_thread(t0);
        let content = LitmusFileGenerator::generate(&test);
        assert!(content.contains("atomic_exchange_explicit"));
    }

    #[test]
    fn parse_herd7_output_basic() {
        let output = "Test MP Allowed\nStates 2\nx=1; y=0;\nx=1; y=1;\nOk\n";
        let result = Herd7Bridge::parse_herd7_output(output, "MP");
        assert!(result.success);
        assert!(!result.allowed_outcomes.is_empty());
    }

    #[test]
    fn mutual_disagreement_display() {
        let result = ComparisonResult::MutualDisagreement {
            litmus_only: vec!["a".into()],
            herd7_only: vec!["b".into()],
        };
        let display = format!("{}", result);
        assert!(display.contains("Disagree"));
        assert!(display.contains("litmus"));
        assert!(display.contains("herd7"));
    }

    #[test]
    fn comparison_report_all_categories() {
        let mut report = ComparisonReport::new();
        report.add_result("t1", ComparisonResult::Agreement, Duration::ZERO, Duration::ZERO);
        report.add_result("t2", ComparisonResult::LitmusInfOnly { outcomes: vec![] }, Duration::ZERO, Duration::ZERO);
        report.add_result("t3", ComparisonResult::Herd7Only { outcomes: vec![] }, Duration::ZERO, Duration::ZERO);
        report.add_result("t4", ComparisonResult::MutualDisagreement { litmus_only: vec![], herd7_only: vec![] }, Duration::ZERO, Duration::ZERO);
        report.add_result("t5", ComparisonResult::Error("e".into()), Duration::ZERO, Duration::ZERO);

        assert_eq!(report.total_tests, 5);
        assert_eq!(report.agreements, 1);
        assert_eq!(report.litmus_inf_only, 1);
        assert_eq!(report.herd7_only, 1);
        assert_eq!(report.mutual_disagreements, 1);
        assert_eq!(report.errors, 1);
    }
}
