//! # Regression Detection
//!
//! Compares two [`AnalysisResult`]s (e.g., before and after a code change)
//! and reports any regressions — blocks whose leakage increased or new
//! leakage sites that appeared.

use serde::{Serialize, Deserialize};
use thiserror::Error;

use shared_types::BlockId;

use crate::fixpoint::AnalysisResult;
use crate::quant_domain::LeakageBits;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors from regression detection.
#[derive(Debug, Error)]
pub enum RegressionError {
    #[error("baseline analysis did not converge")]
    BaselineNotConverged,

    #[error("current analysis did not converge")]
    CurrentNotConverged,
}

// ---------------------------------------------------------------------------
// ContractDelta
// ---------------------------------------------------------------------------

/// Describes the change in leakage contract for a single block.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContractDelta {
    /// The basic block that changed.
    pub block: BlockId,
    /// Leakage in the baseline (old) analysis.
    pub baseline_leakage: LeakageBits,
    /// Leakage in the current (new) analysis.
    pub current_leakage: LeakageBits,
    /// Absolute change (current − baseline) in bits (as f64).
    pub delta_bits: f64,
    /// Whether this constitutes a regression (leakage increased).
    pub is_regression: bool,
}

impl ContractDelta {
    /// Compute the delta between a baseline and current leakage for a block.
    pub fn compute(block: BlockId, baseline: &LeakageBits, current: &LeakageBits) -> Self {
        let b = baseline.to_f64();
        let c = current.to_f64();
        Self {
            block,
            baseline_leakage: baseline.clone(),
            current_leakage: current.clone(),
            delta_bits: c - b,
            is_regression: c > b,
        }
    }
}

// ---------------------------------------------------------------------------
// RegressionReport
// ---------------------------------------------------------------------------

/// Summary report produced by the [`RegressionDetector`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionReport {
    /// Per-block deltas that constitute regressions.
    pub regressions: Vec<ContractDelta>,
    /// Per-block deltas that constitute improvements (leakage decreased).
    pub improvements: Vec<ContractDelta>,
    /// Blocks present in the current analysis but absent from the baseline.
    pub new_blocks: Vec<BlockId>,
    /// Blocks present in the baseline but absent from the current analysis.
    pub removed_blocks: Vec<BlockId>,
    /// Maximum leakage in the baseline.
    pub baseline_max: LeakageBits,
    /// Maximum leakage in the current analysis.
    pub current_max: LeakageBits,
    /// Overall verdict: `true` if any regression was detected.
    pub has_regressions: bool,
}

impl RegressionReport {
    /// Returns the number of regressions found.
    pub fn regression_count(&self) -> usize {
        self.regressions.len()
    }

    /// Returns the number of improvements found.
    pub fn improvement_count(&self) -> usize {
        self.improvements.len()
    }

    /// Produce a human-readable summary string.
    pub fn summary(&self) -> String {
        if self.has_regressions {
            format!(
                "REGRESSION: {} block(s) regressed, max leakage {} → {}",
                self.regressions.len(),
                self.baseline_max,
                self.current_max,
            )
        } else {
            format!(
                "OK: no regressions ({} improvement(s)), max leakage {} → {}",
                self.improvements.len(),
                self.baseline_max,
                self.current_max,
            )
        }
    }
}

// ---------------------------------------------------------------------------
// RegressionDetector
// ---------------------------------------------------------------------------

/// Compares a baseline [`AnalysisResult`] against a current one and
/// produces a [`RegressionReport`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionDetector {
    /// Minimum leakage increase (in bits) to count as a regression.
    pub threshold: f64,
}

impl RegressionDetector {
    /// Create a new detector with the given regression threshold (bits).
    pub fn new(threshold: f64) -> Self {
        Self { threshold }
    }

    /// Compare the `baseline` and `current` analysis results.
    pub fn compare(
        &self,
        baseline: &AnalysisResult,
        current: &AnalysisResult,
    ) -> Result<RegressionReport, RegressionError> {
        if !baseline.converged {
            return Err(RegressionError::BaselineNotConverged);
        }
        if !current.converged {
            return Err(RegressionError::CurrentNotConverged);
        }

        let mut regressions = Vec::new();
        let mut improvements = Vec::new();
        let mut new_blocks = Vec::new();
        let removed_blocks: Vec<BlockId> = baseline
            .block_leakage
            .keys()
            .filter(|b| !current.block_leakage.contains_key(*b))
            .copied()
            .collect();

        for (block, cur_leak) in &current.block_leakage {
            match baseline.block_leakage.get(block) {
                Some(base_leak) => {
                    let delta = ContractDelta::compute(*block, base_leak, cur_leak);
                    if delta.is_regression && delta.delta_bits > self.threshold {
                        regressions.push(delta);
                    } else if delta.delta_bits < -self.threshold {
                        improvements.push(delta);
                    }
                }
                None => {
                    if !cur_leak.is_zero() {
                        new_blocks.push(*block);
                    }
                }
            }
        }

        let has_regressions = !regressions.is_empty() || !new_blocks.is_empty();

        Ok(RegressionReport {
            regressions,
            improvements,
            new_blocks,
            removed_blocks,
            baseline_max: baseline.max_leakage.clone(),
            current_max: current.max_leakage.clone(),
            has_regressions,
        })
    }
}

impl Default for RegressionDetector {
    fn default() -> Self {
        Self::new(0.0)
    }
}
