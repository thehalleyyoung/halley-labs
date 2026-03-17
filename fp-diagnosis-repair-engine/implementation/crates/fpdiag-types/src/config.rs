//! Engine configuration structs.
//!
//! Central configuration for all Penumbra subsystems: tracing, EAG
//! construction, diagnosis, repair, and certification.

use crate::precision::Precision;
use crate::rounding::RoundingMode;
use serde::{Deserialize, Serialize};

/// Top-level Penumbra configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PenumbraConfig {
    /// Tracing configuration.
    pub trace: TraceConfig,
    /// EAG construction configuration.
    pub eag: EagConfig,
    /// Diagnosis configuration.
    pub diagnosis: DiagnosisConfig,
    /// Repair configuration.
    pub repair: RepairConfig,
    /// Certification configuration.
    pub certification: CertificationConfig,
    /// Output configuration.
    pub output: OutputConfig,
}

impl Default for PenumbraConfig {
    fn default() -> Self {
        Self {
            trace: TraceConfig::default(),
            eag: EagConfig::default(),
            diagnosis: DiagnosisConfig::default(),
            repair: RepairConfig::default(),
            certification: CertificationConfig::default(),
            output: OutputConfig::default(),
        }
    }
}

/// Shadow-value tracing configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceConfig {
    /// Shadow precision in significand bits.
    pub shadow_precision_bits: u32,
    /// Maximum trace events before truncation (0 = unlimited).
    pub max_events: u64,
    /// Whether to trace Tier-2 (black-box) library calls.
    pub trace_library_calls: bool,
    /// Rounding mode for the primary computation.
    pub rounding_mode: RoundingMode,
    /// Whether to enable streaming trace output.
    pub streaming: bool,
    /// Compression algorithm for trace storage.
    pub compression: CompressionAlgorithm,
}

impl Default for TraceConfig {
    fn default() -> Self {
        Self {
            shadow_precision_bits: 128,
            max_events: 0,
            trace_library_calls: true,
            rounding_mode: RoundingMode::NearestEven,
            streaming: true,
            compression: CompressionAlgorithm::Lz4,
        }
    }
}

/// Compression algorithm for trace storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    None,
    Lz4,
    Zstd,
}

/// EAG construction configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EagConfig {
    /// Finite-difference step for sensitivity computation.
    /// Should be in [ε_mach, √ε_mach].
    pub finite_diff_step: f64,
    /// Minimum edge weight to retain (sparsification threshold).
    pub min_edge_weight: f64,
    /// Aggregation method for array elements.
    pub aggregation: AggregationMethod,
    /// Whether to compute treewidth (can be expensive for large graphs).
    pub compute_treewidth: bool,
}

impl Default for EagConfig {
    fn default() -> Self {
        Self {
            finite_diff_step: 1e-8, // ~√ε_mach for f64
            min_edge_weight: 1e-12,
            aggregation: AggregationMethod::WorstCase,
            compute_treewidth: false,
        }
    }
}

/// Aggregation method for error over array elements.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AggregationMethod {
    /// Use the worst-case (maximum) error.
    WorstCase,
    /// Use the mean error.
    Mean,
    /// Use a specific percentile (stored as u8, 0–100).
    Percentile(u8),
}

/// Diagnosis engine configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosisConfig {
    /// ULP threshold for classifying a node as "high error".
    pub error_threshold_ulps: f64,
    /// Minimum confidence to report a diagnosis.
    pub min_confidence: f64,
    /// Whether to run all classifiers or stop at first match.
    pub exhaustive: bool,
}

impl Default for DiagnosisConfig {
    fn default() -> Self {
        Self {
            error_threshold_ulps: 10.0,
            min_confidence: 0.5,
            exhaustive: true,
        }
    }
}

/// Repair synthesis configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepairConfig {
    /// Maximum number of repair candidates to generate per node.
    pub max_candidates_per_node: usize,
    /// Maximum total repair budget (number of nodes to repair).
    pub max_repair_budget: usize,
    /// Whether to allow mixed-precision promotion as a fallback.
    pub allow_precision_promotion: bool,
    /// Target precision for promotion.
    pub promotion_precision: Precision,
}

impl Default for RepairConfig {
    fn default() -> Self {
        Self {
            max_candidates_per_node: 3,
            max_repair_budget: 10,
            allow_precision_promotion: true,
            promotion_precision: Precision::Quad,
        }
    }
}

/// Certification configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertificationConfig {
    /// Whether to use interval arithmetic for formal certification.
    pub use_interval_arithmetic: bool,
    /// Number of random samples for empirical certification.
    pub empirical_samples: u32,
    /// Minimum Tier-1 coverage to attempt formal certification.
    pub min_tier1_coverage: f64,
}

impl Default for CertificationConfig {
    fn default() -> Self {
        Self {
            use_interval_arithmetic: true,
            empirical_samples: 10_000,
            min_tier1_coverage: 0.5,
        }
    }
}

/// Output and reporting configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    /// Output format.
    pub format: OutputFormat,
    /// Whether to include source snippets in reports.
    pub include_source: bool,
    /// Verbosity level (0 = quiet, 1 = normal, 2 = verbose, 3 = debug).
    pub verbosity: u8,
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            format: OutputFormat::Human,
            include_source: true,
            verbosity: 1,
        }
    }
}

/// Output format for reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputFormat {
    Human,
    Json,
    Csv,
}
