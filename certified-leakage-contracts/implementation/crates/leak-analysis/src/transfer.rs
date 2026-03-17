//! # Transfer Functions
//!
//! Instruction-level abstract transfer functions for each of the three
//! domains (D\_spec, D\_cache, D\_quant) and the combined transfer that
//! updates the full reduced-product state in one step.

use serde::{Serialize, Deserialize};
use thiserror::Error;

use shared_types::{
    BlockId, CacheConfig, CacheSet, CacheTag, Instruction, VirtualAddress,
    SecurityLevel,
};

use crate::spec_domain::{SpecDomain, SpecState, SpecTag, MisspecKind, SpecWindow};
use crate::cache_domain::{
    CacheDomain, TaintAnnotation, TaintSource,
};
use crate::quant_domain::{QuantDomain, QuantState, SetLeakage};
use crate::product::ReducedProductState;
use crate::reduction::{ReductionOperator, SinglePassReduction};

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors from transfer functions.
#[derive(Debug, Error)]
pub enum TransferError {
    #[error("unknown instruction at {0}")]
    UnknownInstruction(VirtualAddress),

    #[error("cache configuration mismatch")]
    CacheConfigMismatch,
}

// ---------------------------------------------------------------------------
// SpecTransfer
// ---------------------------------------------------------------------------

/// Transfer function for the speculative reachability domain (D\_spec).
///
/// Decides, per instruction, whether a new speculative path is spawned
/// and advances the speculation window.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecTransfer {
    /// Default speculation window for new transient paths.
    pub default_window: SpecWindow,
}

impl SpecTransfer {
    pub fn new(default_window: SpecWindow) -> Self {
        Self { default_window }
    }

    /// Apply the speculative transfer to a single instruction.
    ///
    /// Returns the updated [`SpecState`] at the successor block(s).
    pub fn apply(
        &self,
        instruction: &Instruction,
        block: BlockId,
        state: &SpecState,
    ) -> Vec<(BlockId, SpecState)> {
        let mut successors = Vec::new();

        // Advance the window for the architectural successor.
        let mut arch_state = state.clone();
        arch_state.window.advance(1);
        // Placeholder: the architectural fall-through successor would be
        // determined by the CFG.
        successors.push((block, arch_state));

        successors
    }

    /// Spawn a speculative path from a conditional/indirect branch.
    pub fn spawn_speculative(
        &self,
        origin: BlockId,
        target: BlockId,
        kind: MisspecKind,
        depth: u32,
        base_state: &SpecState,
    ) -> SpecState {
        let mut new_state = base_state.clone();
        new_state.active_tags.insert(SpecTag::new(origin, kind, depth));
        new_state.window = self.default_window;
        new_state
    }
}

impl Default for SpecTransfer {
    fn default() -> Self {
        Self::new(SpecWindow::default())
    }
}

// ---------------------------------------------------------------------------
// CacheTransfer
// ---------------------------------------------------------------------------

/// Transfer function for the tainted abstract cache-state domain (D\_cache).
///
/// Models the effect of load/store instructions on the abstract cache state,
/// assigning taint annotations based on the security level of the accessed
/// data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheTransfer {
    /// Cache configuration (geometry, replacement policy).
    pub config: CacheConfig,
}

impl CacheTransfer {
    pub fn new(config: CacheConfig) -> Self {
        Self { config }
    }

    /// Apply the cache transfer for a memory-accessing instruction.
    pub fn apply(
        &self,
        instruction: &Instruction,
        block: BlockId,
        cache: &mut CacheDomain,
        set: CacheSet,
        tag: CacheTag,
        security: SecurityLevel,
    ) {
        let taint = match security {
            SecurityLevel::Secret => TaintAnnotation::tainted(TaintSource::new(
                block,
                instruction.address,
                security,
            )),
            SecurityLevel::Public => TaintAnnotation::clean(),
        };
        let associativity = self.config.l1d.geometry.num_ways;
        cache.access(set, tag, taint, associativity);
    }
}

// ---------------------------------------------------------------------------
// QuantTransfer
// ---------------------------------------------------------------------------

/// Transfer function for the quantitative channel-capacity domain (D\_quant).
///
/// After the cache domain has been updated, re-computes the per-set
/// leakage contributions based on the number of distinguishable
/// configurations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantTransfer {
    /// Maximum leakage threshold (bits) — used for early termination.
    pub threshold: f64,
}

impl QuantTransfer {
    pub fn new(threshold: f64) -> Self {
        Self { threshold }
    }

    /// Recompute the quantitative state from the current cache domain.
    pub fn apply(
        &self,
        block: BlockId,
        cache: &CacheDomain,
        quant: &mut QuantDomain,
    ) {
        let mut qstate = QuantState::empty();
        for (set_idx, abs_set) in &cache.sets {
            let tainted_count = abs_set
                .ways
                .iter()
                .filter(|w| w.taint.is_tainted)
                .count() as u64;
            // Distinguishable configs: 2^tainted_count (each tainted line
            // can independently be hit or miss).
            let configs = if tainted_count == 0 { 1 } else { 1u64 << tainted_count.min(63) };
            qstate.record_set(SetLeakage::from_config_count(*set_idx, configs));
        }
        quant.update(block, qstate);
    }
}

impl Default for QuantTransfer {
    fn default() -> Self {
        Self::new(0.0)
    }
}

// ---------------------------------------------------------------------------
// CombinedTransfer
// ---------------------------------------------------------------------------

/// Combined transfer function that updates all three domains and then
/// applies reduction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CombinedTransfer {
    /// Transfer for D\_spec.
    pub spec_transfer: SpecTransfer,
    /// Transfer for D\_cache.
    pub cache_transfer: CacheTransfer,
    /// Transfer for D\_quant.
    pub quant_transfer: QuantTransfer,
}

impl CombinedTransfer {
    pub fn new(
        spec_transfer: SpecTransfer,
        cache_transfer: CacheTransfer,
        quant_transfer: QuantTransfer,
    ) -> Self {
        Self { spec_transfer, cache_transfer, quant_transfer }
    }

    /// Apply the combined transfer for a single instruction on the full
    /// reduced-product state.
    pub fn apply(
        &self,
        instruction: &Instruction,
        block: BlockId,
        state: &mut ReducedProductState,
    ) {
        // 1. Speculative transfer — update SpecDomain.
        if let Some(spec_state) = state.spec.state_for(block).cloned() {
            let successors = self.spec_transfer.apply(instruction, block, &spec_state);
            for (target, succ_state) in successors {
                state.spec.update(target, succ_state);
            }
        }

        // 2. Cache transfer — handled externally when memory access info is
        //    available (set, tag, security level).

        // 3. Quantitative transfer — recompute from cache.
        self.quant_transfer.apply(block, &state.cache, &mut state.quant);

        // 4. Reduction.
        let reducer = SinglePassReduction::new();
        reducer.reduce(&mut state.spec, &mut state.cache, &mut state.quant);
    }
}
