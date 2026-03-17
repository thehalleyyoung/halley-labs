//! # Quantitative Channel-Capacity Domain (D\_quant)
//!
//! Bounds the information leakage (in bits) through cache side channels by
//! counting the number of distinguishable cache configurations restricted to
//! tainted lines. The domain tracks per-set leakage and aggregates it into
//! a whole-program leakage bound.

use std::fmt;

use serde::{Serialize, Deserialize};
use thiserror::Error;

use shared_types::{BlockId, CacheSet};

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors from the quantitative domain.
#[derive(Debug, Error)]
pub enum QuantDomainError {
    #[error("leakage overflow for set {set:?}: {bits} bits exceeds bound")]
    LeakageOverflow { set: CacheSet, bits: f64 },
}

// ---------------------------------------------------------------------------
// LeakageBits
// ---------------------------------------------------------------------------

/// A non-negative measure of information leakage expressed in bits.
///
/// Stored as a pair of `u64` (numerator, denominator) for exact rational
/// arithmetic; converted to `f64` on demand.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LeakageBits {
    /// Numerator of the rational value.
    pub numer: u64,
    /// Denominator of the rational value (always ≥ 1).
    pub denom: u64,
}

impl LeakageBits {
    /// Zero leakage.
    pub fn zero() -> Self {
        Self { numer: 0, denom: 1 }
    }

    /// Construct from an integer number of bits.
    pub fn from_bits(bits: u64) -> Self {
        Self { numer: bits, denom: 1 }
    }

    /// Construct from a rational number of bits.
    pub fn from_rational(numer: u64, denom: u64) -> Self {
        assert!(denom > 0, "denominator must be positive");
        Self { numer, denom }
    }

    /// Convert to a floating-point approximation.
    pub fn to_f64(&self) -> f64 {
        self.numer as f64 / self.denom as f64
    }

    /// Lattice join: take the maximum.
    pub fn join(&self, other: &Self) -> Self {
        // Cross-multiply to compare without floating-point.
        if self.numer * other.denom >= other.numer * self.denom {
            self.clone()
        } else {
            other.clone()
        }
    }

    /// Add two leakage values.
    pub fn add(&self, other: &Self) -> Self {
        Self {
            numer: self.numer * other.denom + other.numer * self.denom,
            denom: self.denom * other.denom,
        }
    }

    /// Returns `true` when there is zero leakage.
    pub fn is_zero(&self) -> bool {
        self.numer == 0
    }

    /// Ordering helper for comparisons.
    pub fn cmp_value(&self, other: &Self) -> std::cmp::Ordering {
        (self.numer * other.denom).cmp(&(other.numer * self.denom))
    }
}

impl Default for LeakageBits {
    fn default() -> Self {
        Self::zero()
    }
}

impl fmt::Display for LeakageBits {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.4} bits", self.to_f64())
    }
}

// ---------------------------------------------------------------------------
// SetLeakage
// ---------------------------------------------------------------------------

/// Leakage contribution of a single cache set.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SetLeakage {
    /// Which cache set this measurement belongs to.
    pub set: CacheSet,
    /// Number of distinguishable configurations for this set.
    pub distinguishable_configs: u64,
    /// Leakage in bits: ⌈log₂(distinguishable_configs)⌉.
    pub leakage: LeakageBits,
}

impl SetLeakage {
    /// Create a [`SetLeakage`] from the count of distinguishable
    /// configurations.
    pub fn from_config_count(set: CacheSet, count: u64) -> Self {
        let bits = if count <= 1 {
            LeakageBits::zero()
        } else {
            // ⌈log₂(count)⌉ as an integer.
            let log2 = 64 - (count - 1).leading_zeros() as u64;
            LeakageBits::from_bits(log2)
        };
        Self {
            set,
            distinguishable_configs: count,
            leakage: bits,
        }
    }

    /// Join two set-leakage values (take the maximum).
    pub fn join(&self, other: &Self) -> Self {
        Self {
            set: self.set,
            distinguishable_configs: self.distinguishable_configs.max(other.distinguishable_configs),
            leakage: self.leakage.join(&other.leakage),
        }
    }
}

// ---------------------------------------------------------------------------
// QuantState
// ---------------------------------------------------------------------------

/// Per-program-point quantitative state: per-set leakage contributions and
/// the accumulated total.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuantState {
    /// Leakage contributions keyed by cache-set index.
    pub per_set: indexmap::IndexMap<CacheSet, SetLeakage>,
    /// Total leakage bound (sum over all sets).
    pub total: LeakageBits,
}

impl QuantState {
    /// Create an empty state (zero leakage everywhere).
    pub fn empty() -> Self {
        Self {
            per_set: indexmap::IndexMap::new(),
            total: LeakageBits::zero(),
        }
    }

    /// Record the leakage for a single cache set.
    pub fn record_set(&mut self, set_leakage: SetLeakage) {
        self.per_set.insert(set_leakage.set, set_leakage);
        self.recompute_total();
    }

    /// Recompute the total leakage as the sum of per-set contributions.
    pub fn recompute_total(&mut self) {
        self.total = self
            .per_set
            .values()
            .fold(LeakageBits::zero(), |acc, sl| acc.add(&sl.leakage));
    }

    /// Lattice join across all per-set entries.
    pub fn join(&self, other: &Self) -> Self {
        let mut result = self.clone();
        for (set, other_sl) in &other.per_set {
            let entry = result
                .per_set
                .entry(*set)
                .or_insert_with(|| SetLeakage::from_config_count(*set, 1));
            *entry = entry.join(other_sl);
        }
        result.recompute_total();
        result
    }
}

impl Default for QuantState {
    fn default() -> Self {
        Self::empty()
    }
}

// ---------------------------------------------------------------------------
// QuantDomain
// ---------------------------------------------------------------------------

/// The quantitative channel-capacity abstract domain (D\_quant).
///
/// Maintains per-block [`QuantState`] and provides lattice operations for
/// fixpoint computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantDomain {
    /// Per-block quantitative states.
    pub states: indexmap::IndexMap<BlockId, QuantState>,
    /// Maximum tolerable leakage (in bits) before raising an alarm.
    pub threshold: LeakageBits,
}

impl QuantDomain {
    /// Create a new quantitative domain with the given alarm threshold.
    pub fn new(threshold: LeakageBits) -> Self {
        Self {
            states: indexmap::IndexMap::new(),
            threshold,
        }
    }

    /// Retrieve the quantitative state for a given block.
    pub fn state_for(&self, block: BlockId) -> Option<&QuantState> {
        self.states.get(&block)
    }

    /// Update the state for a block.
    pub fn update(&mut self, block: BlockId, state: QuantState) -> Option<QuantState> {
        self.states.insert(block, state)
    }

    /// Point-wise join.
    pub fn join(&self, other: &Self) -> Self {
        let mut result = self.clone();
        for (block, other_state) in &other.states {
            let entry = result
                .states
                .entry(*block)
                .or_insert_with(QuantState::empty);
            *entry = entry.join(other_state);
        }
        result
    }

    /// The maximum leakage observed across all blocks.
    pub fn max_leakage(&self) -> LeakageBits {
        self.states
            .values()
            .map(|s| &s.total)
            .fold(LeakageBits::zero(), |acc, l| acc.join(l))
    }

    /// Returns `true` when the maximum leakage exceeds the threshold.
    pub fn exceeds_threshold(&self) -> bool {
        self.max_leakage().cmp_value(&self.threshold) == std::cmp::Ordering::Greater
    }
}

impl Default for QuantDomain {
    fn default() -> Self {
        Self::new(LeakageBits::zero())
    }
}
