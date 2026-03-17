//! Spectrum-based fault localization (SBFL) metrics adapted for NLP pipeline stages.
//!
//! Traditional SBFL uses binary coverage (statement executed / not executed).
//! Our variant uses **continuous-valued per-stage differentials** as the
//! coverage signal, adapting suspiciousness scoring accordingly.

pub mod adaptive;
pub mod barinel;
pub mod confidence;
pub mod dstar;
pub mod hypothesis;
pub mod ochiai;
pub mod tarantula;

// ── Re-exports ──────────────────────────────────────────────────────────────

pub use adaptive::{AdaptiveLocator, AdaptiveResult, BetaDistribution};
pub use barinel::{BarinelMetric, BarinelResult, FaultCandidate};
pub use confidence::{BootstrapCI, ClopperPearsonCI, WilsonCI};
pub use dstar::{DStarMetric, DStarResult};
pub use hypothesis::{HypothesisFramework, HypothesisResult, LocalizationHypothesis};
pub use ochiai::{OchiaiMetric, OchiaiResult, SuspiciousnessScore};
pub use tarantula::{SuspiciousnessColor, TarantulaMetric, TarantulaResult};

use serde::{Deserialize, Serialize};
use shared_types::{LocalizerError, Result};

// ── Core data types shared by all metrics ───────────────────────────────────

/// N tests × n stages differential matrix.
///
/// Each entry `D[i][k]` is the continuous-valued differential observed at
/// stage `k` for test case `i`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DifferentialMatrix {
    pub data: Vec<Vec<f64>>,
    pub n_tests: usize,
    pub n_stages: usize,
    pub stage_names: Vec<String>,
}

impl DifferentialMatrix {
    pub fn new(data: Vec<Vec<f64>>, stage_names: Vec<String>) -> Result<Self> {
        let n_tests = data.len();
        let n_stages = stage_names.len();
        for (i, row) in data.iter().enumerate() {
            if row.len() != n_stages {
                return Err(LocalizerError::matrix(
                    format!("row {i} has {} cols, expected {n_stages}", row.len()),
                    n_tests,
                    n_stages,
                ));
            }
        }
        Ok(Self {
            data,
            n_tests,
            n_stages,
            stage_names,
        })
    }

    /// Extract the column vector for stage `k`.
    pub fn column(&self, k: usize) -> Vec<f64> {
        self.data.iter().map(|row| row[k]).collect()
    }

    /// Reference to a single row.
    pub fn row(&self, i: usize) -> &[f64] {
        &self.data[i]
    }

    /// Column sum.
    pub fn column_sum(&self, k: usize) -> f64 {
        self.data.iter().map(|row| row[k]).sum()
    }

    /// Column sum restricted to rows where `mask[i]` is true.
    pub fn masked_column_sum(&self, k: usize, mask: &[bool]) -> f64 {
        self.data
            .iter()
            .zip(mask.iter())
            .filter(|(_, &m)| m)
            .map(|(row, _)| row[k])
            .sum()
    }
}

/// Boolean violation vector (true ⟹ the metamorphic relation was violated).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViolationVector {
    pub violations: Vec<bool>,
}

impl ViolationVector {
    pub fn new(violations: Vec<bool>) -> Self {
        Self { violations }
    }

    pub fn n_violations(&self) -> usize {
        self.violations.iter().filter(|&&v| v).count()
    }

    pub fn n_passing(&self) -> usize {
        self.violations.iter().filter(|&&v| !v).count()
    }

    pub fn len(&self) -> usize {
        self.violations.len()
    }

    pub fn is_empty(&self) -> bool {
        self.violations.is_empty()
    }
}

// ── Calibration data types ─────────────────────────────────────────────────

/// Baseline statistics for a single pipeline stage, computed during
/// calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageBaseline {
    pub stage_name: String,
    pub mean: f64,
    pub std_dev: f64,
    pub threshold: f64,
    pub sample_count: usize,
}

/// Aggregated calibration data for all stages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationData {
    pub stage_baselines: std::collections::HashMap<String, StageBaseline>,
    pub sample_count: usize,
    pub calibration_quality: f64,
}
