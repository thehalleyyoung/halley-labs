//! Configuration types for the localizer and its sub-systems.

use serde::{Deserialize, Serialize};

/// Which SBFL metric to use for fault localization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SBFLMetric {
    Ochiai,
    DStar { star: f64 },
    Tarantula,
    Barinel,
    Adaptive,
}

impl Default for SBFLMetric {
    fn default() -> Self {
        Self::Ochiai
    }
}

/// Output format for reports.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutputFormat {
    Json,
    Csv,
    Table,
    Html,
}

impl Default for OutputFormat {
    fn default() -> Self {
        Self::Json
    }
}

/// Statistical test configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalConfig {
    pub confidence_level: f64,
    pub bootstrap_resamples: usize,
    pub min_effect_size: f64,
    pub multiple_testing_correction: String,
}

impl Default for StatisticalConfig {
    fn default() -> Self {
        Self {
            confidence_level: 0.95,
            bootstrap_resamples: 1000,
            min_effect_size: 0.2,
            multiple_testing_correction: "bonferroni".into(),
        }
    }
}

/// Configuration for the calibration phase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationConfig {
    pub n_calibration_runs: usize,
    pub warmup_runs: usize,
    pub outlier_threshold: f64,
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            n_calibration_runs: 30,
            warmup_runs: 5,
            outlier_threshold: 3.0,
        }
    }
}

/// Top-level localizer configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalizerConfig {
    pub pipeline: PipelineConfig,
    pub transformations: TransformationConfig,
    pub statistical: StatisticalConfig,
    pub calibration: CalibrationConfig,
    pub shrinking: ShrinkingConfig,
    pub report: ReportConfig,
}

/// Pipeline-specific configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    pub stage_names: Vec<String>,
    pub distance_threshold: f64,
    pub top_k_suspects: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            stage_names: Vec::new(),
            distance_threshold: 0.1,
            top_k_suspects: 3,
        }
    }
}

/// Transformation selection configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformationConfig {
    pub enabled_transformations: Vec<String>,
    pub max_per_input: usize,
}

impl Default for TransformationConfig {
    fn default() -> Self {
        Self {
            enabled_transformations: Vec::new(),
            max_per_input: 5,
        }
    }
}

/// Input-shrinking configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShrinkingConfig {
    pub max_iterations: u32,
    pub min_tokens: usize,
    pub strategies: Vec<String>,
}

impl Default for ShrinkingConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            min_tokens: 2,
            strategies: vec!["token_removal".into(), "clause_removal".into()],
        }
    }
}

/// Report generation configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportConfig {
    pub format: OutputFormat,
    pub include_details: bool,
    pub max_suspects: usize,
}

impl Default for ReportConfig {
    fn default() -> Self {
        Self {
            format: OutputFormat::Json,
            include_details: true,
            max_suspects: 5,
        }
    }
}
