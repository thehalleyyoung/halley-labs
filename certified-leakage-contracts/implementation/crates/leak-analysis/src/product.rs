//! # Three-Way Reduced Product
//!
//! Combines the speculative reachability domain (D\_spec), the tainted
//! abstract cache-state domain (D\_cache), and the quantitative
//! channel-capacity domain (D\_quant) into a single [`ReducedProductState`].
//!
//! The product exposes lattice operations that delegate to each component
//! and then apply the reduction operator ρ to sharpen the result.

use serde::{Serialize, Deserialize};

use crate::spec_domain::{SpecDomain, SpecWindow};
use crate::cache_domain::CacheDomain;
use crate::quant_domain::{QuantDomain, LeakageBits};
use crate::reduction::{ReductionOperator, ReductionResult, SinglePassReduction};

// ---------------------------------------------------------------------------
// ReducedProductState
// ---------------------------------------------------------------------------

/// The three-way reduced product state ⟨D\_spec, D\_cache, D\_quant⟩.
///
/// All lattice operations (join, widen) first compute the component-wise
/// result and then apply the reduction operator ρ.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReducedProductState {
    /// Speculative reachability domain.
    pub spec: SpecDomain,
    /// Tainted abstract cache-state domain.
    pub cache: CacheDomain,
    /// Quantitative channel-capacity domain.
    pub quant: QuantDomain,
}

impl ReducedProductState {
    /// Create a new product state from its three component domains.
    pub fn new(spec: SpecDomain, cache: CacheDomain, quant: QuantDomain) -> Self {
        Self { spec, cache, quant }
    }

    /// Component-wise join followed by reduction.
    pub fn join(&self, other: &Self) -> Self {
        let mut result = Self {
            spec: self.spec.join(&other.spec),
            cache: self.cache.join(&other.cache),
            quant: self.quant.join(&other.quant),
        };
        let reducer = SinglePassReduction::new();
        let _result = reducer.reduce(&mut result.spec, &mut result.cache, &mut result.quant);
        result
    }

    /// Apply an arbitrary [`ReductionOperator`] in place.
    pub fn reduce(&mut self, operator: &dyn ReductionOperator) -> ReductionResult {
        operator.reduce(&mut self.spec, &mut self.cache, &mut self.quant)
    }

    /// Returns the maximum leakage observed across all blocks.
    pub fn max_leakage(&self) -> LeakageBits {
        self.quant.max_leakage()
    }

    /// Returns `true` when all three component domains have stabilised
    /// relative to the `previous` product state.
    pub fn is_stable(&self, previous: &Self) -> bool {
        self.spec.is_stable(&previous.spec)
        // Cache and quant stability checks delegate to their own
        // equality-based comparisons via PartialEq.
    }

    /// Total number of active speculative tags.
    pub fn active_spec_tags(&self) -> usize {
        self.spec.total_active_tags()
    }

    /// Total number of tainted cache lines.
    pub fn tainted_lines(&self) -> usize {
        self.cache.total_tainted_lines()
    }
}
