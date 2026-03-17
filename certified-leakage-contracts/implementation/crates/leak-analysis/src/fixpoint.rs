//! # Worklist-Based Fixpoint Engine
//!
//! Drives the reduced-product analysis over a [`ControlFlowGraph`] until
//! the abstract state stabilises at every program point.

use std::collections::VecDeque;

use serde::{Serialize, Deserialize};
use thiserror::Error;

use shared_types::{BlockId, CacheConfig, ControlFlowGraph};

use crate::product::ReducedProductState;
use crate::spec_domain::{SpecDomain, SpecWindow};
use crate::cache_domain::CacheDomain;
use crate::quant_domain::{QuantDomain, LeakageBits};
use crate::reduction::{ReductionOperator, SinglePassReduction, IterativeReduction};
use crate::transfer::CombinedTransfer;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors from the fixpoint engine.
#[derive(Debug, Error)]
pub enum FixpointError {
    #[error("fixpoint did not converge after {iterations} iterations")]
    NonConvergence { iterations: u32 },

    #[error("empty control-flow graph")]
    EmptyCfg,

    #[error("entry block not found in CFG")]
    MissingEntry,
}

// ---------------------------------------------------------------------------
// AnalysisConfig
// ---------------------------------------------------------------------------

/// Configuration knobs for the fixpoint engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisConfig {
    /// Maximum number of fixpoint iterations before giving up.
    pub max_iterations: u32,
    /// Speculation window depth for D\_spec.
    pub speculation_window: u32,
    /// Whether to use iterative reduction (vs. single-pass).
    pub iterative_reduction: bool,
    /// Maximum tolerable leakage in bits.
    pub leakage_threshold: f64,
    /// Cache configuration for D\_cache.
    pub cache_config: CacheConfig,
    /// Enable widening after this many iterations per block.
    pub widen_delay: u32,
    /// Emit trace-level diagnostics during the solve.
    pub verbose: bool,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            speculation_window: 224,
            iterative_reduction: false,
            leakage_threshold: 0.0,
            cache_config: CacheConfig {
                l1d: shared_types::CacheLevel {
                    geometry: shared_types::CacheGeometry {
                        line_size_bits: 6,    // 64-byte lines
                        set_index_bits: 6,    // 64 sets
                        num_ways: 8,
                        num_sets: 64,
                        total_size: 64 * 8 * 64,
                    },
                    replacement: shared_types::ReplacementPolicy::LRU,
                    latency_cycles: 4,
                    is_inclusive: false,
                    is_shared: false,
                },
                l1i: None,
                l2: None,
                l3: None,
                line_size: 64,
                speculation_window: 224,
                prefetch_enabled: false,
            },
            widen_delay: 3,
            verbose: false,
        }
    }
}

// ---------------------------------------------------------------------------
// AnalysisResult
// ---------------------------------------------------------------------------

/// The final output of a completed fixpoint analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResult {
    /// The stabilised reduced-product state.
    pub state: ReducedProductState,
    /// Number of fixpoint iterations executed.
    pub iterations: u32,
    /// Whether the fixpoint converged within the budget.
    pub converged: bool,
    /// Maximum leakage observed at any program point.
    pub max_leakage: LeakageBits,
    /// Per-block leakage map (block → bits).
    pub block_leakage: indexmap::IndexMap<BlockId, LeakageBits>,
}

impl AnalysisResult {
    /// Returns `true` when the analysis found zero leakage.
    pub fn is_leak_free(&self) -> bool {
        self.max_leakage.is_zero()
    }

    /// Returns `true` when the leakage exceeds the configured threshold.
    pub fn exceeds_threshold(&self, threshold: &LeakageBits) -> bool {
        self.max_leakage.cmp_value(threshold) == std::cmp::Ordering::Greater
    }
}

// ---------------------------------------------------------------------------
// AnalysisEngine
// ---------------------------------------------------------------------------

/// Worklist-based fixpoint engine.
///
/// Iterates over the [`ControlFlowGraph`], applying the [`CombinedTransfer`]
/// at each block and joining results until the abstract state converges.
#[derive(Debug)]
pub struct AnalysisEngine {
    /// Analysis configuration.
    pub config: AnalysisConfig,
    /// The combined transfer function.
    pub transfer: CombinedTransfer,
}

impl AnalysisEngine {
    /// Create a new engine from a configuration and a combined transfer.
    pub fn new(config: AnalysisConfig, transfer: CombinedTransfer) -> Self {
        Self { config, transfer }
    }

    /// Run the fixpoint analysis on the given CFG.
    ///
    /// Returns an [`AnalysisResult`] on success, or a [`FixpointError`] if
    /// the analysis cannot complete.
    pub fn run(&self, cfg: &ControlFlowGraph) -> Result<AnalysisResult, FixpointError> {
        let entry = cfg.entry.ok_or(FixpointError::MissingEntry)?;
        if cfg.blocks.is_empty() {
            return Err(FixpointError::EmptyCfg);
        }

        // Initialise the product state.
        let window = SpecWindow::new(self.config.speculation_window);
        let spec = SpecDomain::new(window);
        let cache = CacheDomain::new(self.config.cache_config.clone());
        let quant = QuantDomain::new(LeakageBits::from_bits(
            self.config.leakage_threshold as u64,
        ));
        let mut state = ReducedProductState::new(spec, cache, quant);

        // Worklist initialisation.
        let mut worklist: VecDeque<BlockId> = VecDeque::new();
        worklist.push_back(entry);
        let mut iterations: u32 = 0;

        while let Some(block) = worklist.pop_front() {
            iterations += 1;
            if iterations > self.config.max_iterations {
                return Ok(AnalysisResult {
                    state: state.clone(),
                    iterations,
                    converged: false,
                    max_leakage: state.max_leakage(),
                    block_leakage: self.collect_block_leakage(&state),
                });
            }

            let prev = state.clone();

            // Apply transfer for each instruction in the block.
            if let Some(bb) = cfg.blocks.get(&block) {
                for instr in &bb.instructions {
                    self.transfer.apply(instr, block, &mut state);
                }
            }

            // Check stability — if the state changed, re-add successors.
            if !state.is_stable(&prev) {
                for edge in &cfg.edges {
                    if edge.source == block {
                        worklist.push_back(edge.target);
                    }
                }
            }
        }

        Ok(AnalysisResult {
            state: state.clone(),
            iterations,
            converged: true,
            max_leakage: state.max_leakage(),
            block_leakage: self.collect_block_leakage(&state),
        })
    }

    /// Collect per-block leakage from the quantitative domain.
    fn collect_block_leakage(
        &self,
        state: &ReducedProductState,
    ) -> indexmap::IndexMap<BlockId, LeakageBits> {
        state
            .quant
            .states
            .iter()
            .map(|(block, qs)| (*block, qs.total.clone()))
            .collect()
    }
}
