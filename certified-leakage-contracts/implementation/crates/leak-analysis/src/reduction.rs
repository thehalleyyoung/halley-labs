//! # Reduction Operator (ρ)
//!
//! The reduction operator sharpens the three-way reduced product by
//! exploiting cross-domain information:
//!
//! - Unreachable speculative paths (from D\_spec) remove cache taint
//!   (in D\_cache).
//! - Untainted cache sets (in D\_cache) zero their leakage contribution
//!   (in D\_quant).
//!
//! Two strategies are provided:
//! - [`SinglePassReduction`]: applies ρ once.
//! - [`IterativeReduction`]: applies ρ until a fixpoint is reached.

use serde::{Serialize, Deserialize};
use thiserror::Error;

use shared_types::BlockId;

use crate::spec_domain::SpecDomain;
use crate::cache_domain::CacheDomain;
use crate::quant_domain::{QuantDomain, LeakageBits};

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors from the reduction operator.
#[derive(Debug, Error)]
pub enum ReductionError {
    #[error("reduction did not converge after {iterations} iterations")]
    NonConvergence { iterations: u32 },
}

// ---------------------------------------------------------------------------
// ReductionResult
// ---------------------------------------------------------------------------

/// The result of applying the reduction operator, carrying the sharpened
/// domains and diagnostic metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReductionResult {
    /// Number of iterations the reduction required (1 for single-pass).
    pub iterations: u32,
    /// Number of taint annotations removed during reduction.
    pub taints_removed: usize,
    /// Number of set-leakage entries zeroed during reduction.
    pub sets_zeroed: usize,
    /// Whether the reduction reached a fixpoint.
    pub converged: bool,
}

impl ReductionResult {
    /// A trivial result indicating no reduction was needed.
    pub fn trivial() -> Self {
        Self {
            iterations: 0,
            taints_removed: 0,
            sets_zeroed: 0,
            converged: true,
        }
    }
}

// ---------------------------------------------------------------------------
// ReductionOperator trait
// ---------------------------------------------------------------------------

/// Trait for reduction operators that sharpen the three-way reduced product.
pub trait ReductionOperator: std::fmt::Debug + Send + Sync {
    /// Apply the reduction to the three domains **in place**, returning
    /// diagnostic metadata.
    fn reduce(
        &self,
        spec: &mut SpecDomain,
        cache: &mut CacheDomain,
        quant: &mut QuantDomain,
    ) -> ReductionResult;

    /// Human-readable name of this reduction strategy.
    fn name(&self) -> &str;
}

// ---------------------------------------------------------------------------
// SinglePassReduction
// ---------------------------------------------------------------------------

/// Applies the reduction operator exactly once.
///
/// 1. For every block unreachable under speculation, remove taint from
///    the corresponding cache lines.
/// 2. For every untainted cache set, zero its leakage contribution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SinglePassReduction {
    /// When `true`, log each taint removal at `trace` level.
    pub verbose: bool,
}

impl SinglePassReduction {
    pub fn new() -> Self {
        Self { verbose: false }
    }

    /// Perform one reduction pass.
    fn reduce_once(
        &self,
        _spec: &mut SpecDomain,
        cache: &mut CacheDomain,
        quant: &mut QuantDomain,
    ) -> (usize, usize) {
        // Phase 1: remove taint from unreachable speculative paths.
        // (Stub — full implementation depends on the spec→cache mapping.)
        let taints_removed: usize = 0;

        // Phase 2: zero leakage for untainted sets.
        let mut sets_zeroed: usize = 0;
        let untainted: Vec<_> = cache
            .sets
            .iter()
            .filter(|(_, s)| s.is_untainted())
            .map(|(idx, _)| *idx)
            .collect();

        for state in quant.states.values_mut() {
            for set_idx in &untainted {
                if let Some(sl) = state.per_set.get_mut(set_idx) {
                    if !sl.leakage.is_zero() {
                        sl.leakage = LeakageBits::zero();
                        sl.distinguishable_configs = 1;
                        sets_zeroed += 1;
                    }
                }
            }
            state.recompute_total();
        }

        (taints_removed, sets_zeroed)
    }
}

impl Default for SinglePassReduction {
    fn default() -> Self {
        Self::new()
    }
}

impl ReductionOperator for SinglePassReduction {
    fn reduce(
        &self,
        spec: &mut SpecDomain,
        cache: &mut CacheDomain,
        quant: &mut QuantDomain,
    ) -> ReductionResult {
        let (taints_removed, sets_zeroed) = self.reduce_once(spec, cache, quant);
        ReductionResult {
            iterations: 1,
            taints_removed,
            sets_zeroed,
            converged: true,
        }
    }

    fn name(&self) -> &str {
        "single-pass"
    }
}

// ---------------------------------------------------------------------------
// IterativeReduction
// ---------------------------------------------------------------------------

/// Applies the reduction operator iteratively until the result stabilises
/// (i.e., a fixpoint is reached) or the iteration budget is exhausted.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IterativeReduction {
    /// Maximum number of reduction iterations before giving up.
    pub max_iterations: u32,
    /// When `true`, log per-iteration statistics.
    pub verbose: bool,
}

impl IterativeReduction {
    pub fn new(max_iterations: u32) -> Self {
        Self { max_iterations, verbose: false }
    }
}

impl Default for IterativeReduction {
    fn default() -> Self {
        Self::new(100)
    }
}

impl ReductionOperator for IterativeReduction {
    fn reduce(
        &self,
        spec: &mut SpecDomain,
        cache: &mut CacheDomain,
        quant: &mut QuantDomain,
    ) -> ReductionResult {
        let inner = SinglePassReduction { verbose: self.verbose };
        let mut total_taints = 0usize;
        let mut total_sets = 0usize;

        for i in 1..=self.max_iterations {
            let (t, s) = inner.reduce_once(spec, cache, quant);
            total_taints += t;
            total_sets += s;
            if t == 0 && s == 0 {
                return ReductionResult {
                    iterations: i,
                    taints_removed: total_taints,
                    sets_zeroed: total_sets,
                    converged: true,
                };
            }
        }

        ReductionResult {
            iterations: self.max_iterations,
            taints_removed: total_taints,
            sets_zeroed: total_sets,
            converged: false,
        }
    }

    fn name(&self) -> &str {
        "iterative"
    }
}
