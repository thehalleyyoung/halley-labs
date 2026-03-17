//! Benchmark suite and runner infrastructure for evaluating leakage analysis tools.
//!
//! Provides types for defining benchmarks, organizing them into suites,
//! running them, and collecting results with timing and correctness data.

use std::collections::HashMap;
use std::fmt;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

/// Category of a benchmark, indicating what aspect of analysis it stresses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BenchmarkCategory {
    /// AES and related block cipher implementations.
    BlockCipher,
    /// Public-key cryptography (RSA, ECC).
    PublicKey,
    /// Hash functions (SHA, BLAKE).
    HashFunction,
    /// Stream ciphers (ChaCha, Salsa).
    StreamCipher,
    /// Key derivation functions (PBKDF2, Argon2).
    KeyDerivation,
    /// Synthetic micro-benchmarks targeting specific patterns.
    Synthetic,
    /// Real-world application code.
    Application,
}

impl fmt::Display for BenchmarkCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BlockCipher => write!(f, "Block Cipher"),
            Self::PublicKey => write!(f, "Public Key"),
            Self::HashFunction => write!(f, "Hash Function"),
            Self::StreamCipher => write!(f, "Stream Cipher"),
            Self::KeyDerivation => write!(f, "Key Derivation"),
            Self::Synthetic => write!(f, "Synthetic"),
            Self::Application => write!(f, "Application"),
        }
    }
}

/// A single benchmark definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Benchmark {
    /// Unique identifier for this benchmark.
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// Category for grouping.
    pub category: BenchmarkCategory,
    /// Path to the binary under test.
    pub binary_path: PathBuf,
    /// Expected number of leaking cache sets (ground truth), if known.
    pub expected_leaking_sets: Option<u32>,
    /// Expected total leakage bound in bits (ground truth), if known.
    pub expected_leakage_bits: Option<f64>,
    /// Maximum allowed analysis time before timeout.
    pub timeout: Duration,
}

impl Benchmark {
    /// Create a new benchmark with the given name, category, and binary path.
    pub fn new(name: impl Into<String>, category: BenchmarkCategory, binary_path: impl Into<PathBuf>) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            category,
            binary_path: binary_path.into(),
            expected_leaking_sets: None,
            expected_leakage_bits: None,
            timeout: Duration::from_secs(300),
        }
    }

    /// Set a human-readable description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Set the expected ground-truth leaking sets.
    pub fn with_expected_leaking_sets(mut self, count: u32) -> Self {
        self.expected_leaking_sets = Some(count);
        self
    }

    /// Set the expected ground-truth leakage bits.
    pub fn with_expected_leakage_bits(mut self, bits: f64) -> Self {
        self.expected_leakage_bits = Some(bits);
        self
    }

    /// Set the analysis timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Whether ground-truth data is available for correctness evaluation.
    pub fn has_ground_truth(&self) -> bool {
        self.expected_leaking_sets.is_some() || self.expected_leakage_bits.is_some()
    }
}

/// Result of running a single benchmark.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Name of the benchmark that produced this result.
    pub benchmark_name: String,
    /// Category of the benchmark.
    pub category: BenchmarkCategory,
    /// Wall-clock time taken by the analysis.
    pub elapsed: Duration,
    /// Peak memory usage in bytes, if measured.
    pub peak_memory_bytes: Option<u64>,
    /// Number of leaking cache sets reported by the tool.
    pub reported_leaking_sets: u32,
    /// Total leakage bound in bits reported by the tool.
    pub reported_leakage_bits: f64,
    /// Whether the analysis completed without error.
    pub success: bool,
    /// Error message if the analysis failed.
    pub error_message: Option<String>,
    /// Whether the analysis timed out.
    pub timed_out: bool,
}

impl BenchmarkResult {
    /// Create a successful result.
    pub fn success(
        benchmark_name: impl Into<String>,
        category: BenchmarkCategory,
        elapsed: Duration,
        leaking_sets: u32,
        leakage_bits: f64,
    ) -> Self {
        Self {
            benchmark_name: benchmark_name.into(),
            category,
            elapsed,
            peak_memory_bytes: None,
            reported_leaking_sets: leaking_sets,
            reported_leakage_bits: leakage_bits,
            success: true,
            error_message: None,
            timed_out: false,
        }
    }

    /// Create a failure result.
    pub fn failure(
        benchmark_name: impl Into<String>,
        category: BenchmarkCategory,
        elapsed: Duration,
        error: impl Into<String>,
    ) -> Self {
        Self {
            benchmark_name: benchmark_name.into(),
            category,
            elapsed,
            peak_memory_bytes: None,
            reported_leaking_sets: 0,
            reported_leakage_bits: 0.0,
            success: false,
            error_message: Some(error.into()),
            timed_out: false,
        }
    }

    /// Create a timeout result.
    pub fn timeout(benchmark_name: impl Into<String>, category: BenchmarkCategory, timeout: Duration) -> Self {
        Self {
            benchmark_name: benchmark_name.into(),
            category,
            elapsed: timeout,
            peak_memory_bytes: None,
            reported_leaking_sets: 0,
            reported_leakage_bits: 0.0,
            success: false,
            error_message: Some("Analysis timed out".into()),
            timed_out: true,
        }
    }
}

/// A collection of related benchmarks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkSuite {
    /// Name of this suite.
    pub name: String,
    /// Description of the suite.
    pub description: String,
    /// The benchmarks in this suite.
    pub benchmarks: Vec<Benchmark>,
}

impl BenchmarkSuite {
    /// Create a new empty suite.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            benchmarks: Vec::new(),
        }
    }

    /// Add a benchmark to the suite.
    pub fn add(&mut self, benchmark: Benchmark) {
        self.benchmarks.push(benchmark);
    }

    /// Number of benchmarks in the suite.
    pub fn len(&self) -> usize {
        self.benchmarks.len()
    }

    /// Whether the suite is empty.
    pub fn is_empty(&self) -> bool {
        self.benchmarks.is_empty()
    }

    /// Iterate over benchmarks filtered by category.
    pub fn by_category(&self, category: BenchmarkCategory) -> impl Iterator<Item = &Benchmark> {
        self.benchmarks.iter().filter(move |b| b.category == category)
    }
}

/// Configuration and execution engine for running benchmark suites.
#[derive(Debug)]
pub struct BenchmarkRunner {
    /// Number of times to repeat each benchmark for statistical significance.
    pub repetitions: u32,
    /// Whether to run benchmarks in parallel.
    pub parallel: bool,
    /// Maximum number of concurrent benchmarks.
    pub max_concurrency: usize,
    /// Collected results from completed runs.
    results: Vec<BenchmarkResult>,
}

impl BenchmarkRunner {
    /// Create a new runner with default settings.
    pub fn new() -> Self {
        Self {
            repetitions: 1,
            parallel: false,
            max_concurrency: 4,
            results: Vec::new(),
        }
    }

    /// Set the number of repetitions per benchmark.
    pub fn with_repetitions(mut self, n: u32) -> Self {
        self.repetitions = n;
        self
    }

    /// Enable parallel execution.
    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    /// Run all benchmarks in a suite and collect results.
    ///
    /// Currently returns placeholder results; actual analysis integration is pending.
    pub fn run_suite(&mut self, suite: &BenchmarkSuite) -> Vec<BenchmarkResult> {
        let mut results = Vec::new();
        for benchmark in &suite.benchmarks {
            for _ in 0..self.repetitions {
                let start = Instant::now();
                // Placeholder: actual analysis invocation goes here.
                let elapsed = start.elapsed();
                let result = BenchmarkResult::success(
                    &benchmark.name,
                    benchmark.category,
                    elapsed,
                    0,
                    0.0,
                );
                results.push(result);
            }
        }
        self.results.extend(results.clone());
        results
    }

    /// Retrieve all collected results.
    pub fn results(&self) -> &[BenchmarkResult] {
        &self.results
    }

    /// Group results by category.
    pub fn results_by_category(&self) -> HashMap<BenchmarkCategory, Vec<&BenchmarkResult>> {
        let mut map: HashMap<BenchmarkCategory, Vec<&BenchmarkResult>> = HashMap::new();
        for r in &self.results {
            map.entry(r.category).or_default().push(r);
        }
        map
    }

    /// Clear all collected results.
    pub fn clear(&mut self) {
        self.results.clear();
    }
}

impl Default for BenchmarkRunner {
    fn default() -> Self {
        Self::new()
    }
}
