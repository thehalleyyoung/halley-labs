//! Synthetic test case generator for leakage analysis benchmarks.
//!
//! Generates controlled test binaries and program patterns with known leakage
//! characteristics, enabling ground-truth evaluation of analysis precision.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::benchmark::{Benchmark, BenchmarkCategory, BenchmarkSuite};

/// Pattern of cache access to generate in a synthetic test.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AccessPattern {
    /// Secret-independent constant-time access.
    ConstantTime,
    /// Secret-dependent table lookup (e.g., AES T-table).
    TableLookup,
    /// Secret-dependent branch (if/else on secret bit).
    SecretBranch,
    /// Secret-dependent loop bound.
    SecretLoop,
    /// Diamond pattern: converging branches with shared access.
    Diamond,
    /// Nested secret-dependent accesses.
    Nested,
}

/// Configuration for a single synthetic test case.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyntheticConfig {
    /// Pattern of cache access.
    pub pattern: AccessPattern,
    /// Number of secret bits influencing the access.
    pub secret_bits: u32,
    /// Number of cache sets involved.
    pub cache_sets: u32,
    /// Expected leakage in bits (ground truth).
    pub expected_leakage_bits: f64,
    /// Expected number of leaking cache sets.
    pub expected_leaking_sets: u32,
    /// Optional label for the test.
    pub label: Option<String>,
}

impl SyntheticConfig {
    /// Create a constant-time test case with zero expected leakage.
    pub fn constant_time(cache_sets: u32) -> Self {
        Self {
            pattern: AccessPattern::ConstantTime,
            secret_bits: 0,
            cache_sets,
            expected_leakage_bits: 0.0,
            expected_leaking_sets: 0,
            label: None,
        }
    }

    /// Create a table-lookup test case.
    pub fn table_lookup(secret_bits: u32, cache_sets: u32) -> Self {
        let expected_leakage_bits = secret_bits as f64;
        Self {
            pattern: AccessPattern::TableLookup,
            secret_bits,
            cache_sets,
            expected_leakage_bits,
            expected_leaking_sets: cache_sets.min(1 << secret_bits),
            label: None,
        }
    }

    /// Create a secret-branch test case.
    pub fn secret_branch(cache_sets: u32) -> Self {
        Self {
            pattern: AccessPattern::SecretBranch,
            secret_bits: 1,
            cache_sets,
            expected_leakage_bits: 1.0,
            expected_leaking_sets: 2,
            label: None,
        }
    }
}

/// Generator that produces synthetic benchmark suites with known ground truth.
#[derive(Debug)]
pub struct SyntheticGenerator {
    /// Output directory for generated test binaries.
    pub output_dir: PathBuf,
    /// Configurations to generate.
    configs: Vec<SyntheticConfig>,
}

impl SyntheticGenerator {
    /// Create a new generator writing to the given output directory.
    pub fn new(output_dir: impl Into<PathBuf>) -> Self {
        Self {
            output_dir: output_dir.into(),
            configs: Vec::new(),
        }
    }

    /// Add a synthetic test configuration.
    pub fn add_config(&mut self, config: SyntheticConfig) {
        self.configs.push(config);
    }

    /// Add the standard suite of synthetic patterns covering common cases.
    pub fn add_standard_suite(&mut self) {
        self.configs.push(SyntheticConfig::constant_time(64));
        self.configs.push(SyntheticConfig::table_lookup(8, 16));
        self.configs.push(SyntheticConfig::secret_branch(64));
        self.configs.push(SyntheticConfig {
            pattern: AccessPattern::SecretLoop,
            secret_bits: 4,
            cache_sets: 64,
            expected_leakage_bits: 4.0,
            expected_leaking_sets: 16,
            label: Some("secret-loop-4bit".into()),
        });
        self.configs.push(SyntheticConfig {
            pattern: AccessPattern::Diamond,
            secret_bits: 1,
            cache_sets: 64,
            expected_leakage_bits: 1.0,
            expected_leaking_sets: 2,
            label: Some("diamond-1bit".into()),
        });
    }

    /// Generate a benchmark suite from the registered configurations.
    ///
    /// Binary generation is not yet implemented; benchmarks point to placeholder paths.
    pub fn generate_suite(&self, suite_name: impl Into<String>) -> BenchmarkSuite {
        let mut suite = BenchmarkSuite::new(suite_name);
        suite.description = "Synthetically generated benchmarks with known ground truth".into();

        for (i, config) in self.configs.iter().enumerate() {
            let name = config
                .label
                .clone()
                .unwrap_or_else(|| format!("synthetic-{}-{}", format!("{:?}", config.pattern).to_lowercase(), i));

            let binary_path = self.output_dir.join(format!("{}.bin", name));

            let benchmark = Benchmark::new(&name, BenchmarkCategory::Synthetic, binary_path)
                .with_description(format!(
                    "{:?} pattern with {} secret bits across {} cache sets",
                    config.pattern, config.secret_bits, config.cache_sets
                ))
                .with_expected_leaking_sets(config.expected_leaking_sets)
                .with_expected_leakage_bits(config.expected_leakage_bits);

            suite.add(benchmark);
        }

        suite
    }

    /// Number of configured test cases.
    pub fn config_count(&self) -> usize {
        self.configs.len()
    }
}
