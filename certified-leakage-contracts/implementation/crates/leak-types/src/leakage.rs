//! Core leakage measurement types for quantitative information flow analysis.
//!
//! This module provides the mathematical types for measuring and bounding
//! information leakage through speculative cache side-channels. Leakage is
//! quantified in bits using information-theoretic metrics (Shannon entropy,
//! min-entropy, guessing entropy, max-leakage).
//!
//! The key types are:
//! - [`LeakageBound`]: a single bound on bits leaked
//! - [`LeakageVector`]: per-cache-set leakage bounds
//! - [`LeakageMetric`]: which entropy measure to use
//! - [`ChannelCapacity`]: capacity of the cache side-channel
//! - [`LeakageClassification`]: qualitative classification of leakage severity

use std::fmt;
use std::ops::{Add, Mul};

use serde::{Deserialize, Serialize};
use shared_types::CacheSet;

use crate::domain::AbstractDomain;

// ---------------------------------------------------------------------------
// LeakageBound
// ---------------------------------------------------------------------------

/// An upper bound on information leakage measured in bits.
///
/// Wraps an `f64` value representing the maximum number of bits an attacker
/// can learn about secret data through a cache side-channel observation.
/// A bound of 0.0 means the observation is independent of secrets.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct LeakageBound {
    /// Bits of leakage (≥ 0.0). `f64::INFINITY` represents unbounded leakage.
    bits: f64,
}

impl LeakageBound {
    /// Zero leakage – the observation reveals nothing about secrets.
    pub const ZERO: Self = Self { bits: 0.0 };

    /// Unbounded leakage – worst case, all secret bits may leak.
    pub const UNBOUNDED: Self = Self {
        bits: f64::INFINITY,
    };

    /// Create a new leakage bound from a bit count.
    ///
    /// # Panics
    /// Panics if `bits` is negative or NaN.
    pub fn new(bits: f64) -> Self {
        assert!(
            bits >= 0.0 && !bits.is_nan(),
            "leakage bound must be non-negative, got {bits}"
        );
        Self { bits }
    }

    /// Create from an integer number of bits.
    pub fn from_bits(n: u32) -> Self {
        Self { bits: f64::from(n) }
    }

    /// Create from the log₂ of the number of distinguishable observations.
    pub fn from_observations(count: u64) -> Self {
        if count <= 1 {
            Self::ZERO
        } else {
            Self {
                bits: (count as f64).log2(),
            }
        }
    }

    /// Raw bit count.
    pub fn bits(&self) -> f64 {
        self.bits
    }

    /// Whether this represents zero leakage.
    pub fn is_zero(&self) -> bool {
        self.bits == 0.0
    }

    /// Whether this is unbounded.
    pub fn is_unbounded(&self) -> bool {
        self.bits.is_infinite()
    }

    /// Maximum of two bounds (pessimistic choice).
    pub fn max(self, other: Self) -> Self {
        Self {
            bits: self.bits.max(other.bits),
        }
    }

    /// Minimum of two bounds (optimistic choice).
    pub fn min(self, other: Self) -> Self {
        Self {
            bits: self.bits.min(other.bits),
        }
    }

    /// Bound is at most `limit` bits. Clamps to [0, limit].
    pub fn clamp(self, limit: f64) -> Self {
        Self {
            bits: self.bits.clamp(0.0, limit),
        }
    }

    /// Scale the leakage bound by a constant factor.
    pub fn scale(self, factor: f64) -> Self {
        assert!(factor >= 0.0, "scale factor must be non-negative");
        Self {
            bits: self.bits * factor,
        }
    }

    /// Convert to the number of distinguishable observations: 2^bits.
    pub fn observation_count(&self) -> f64 {
        if self.is_zero() {
            1.0
        } else {
            (2.0_f64).powf(self.bits)
        }
    }

    /// Leakage as a fraction of a total secret size (in bits).
    pub fn fraction_of(&self, total_secret_bits: u32) -> f64 {
        if total_secret_bits == 0 {
            return 0.0;
        }
        (self.bits / f64::from(total_secret_bits)).min(1.0)
    }
}

impl PartialEq for LeakageBound {
    fn eq(&self, other: &Self) -> bool {
        self.bits == other.bits
    }
}

impl Eq for LeakageBound {}

impl PartialOrd for LeakageBound {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for LeakageBound {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.bits.total_cmp(&other.bits)
    }
}

impl std::hash::Hash for LeakageBound {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.bits.to_bits().hash(state);
    }
}

impl Add for LeakageBound {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            bits: self.bits + rhs.bits,
        }
    }
}

impl Mul<f64> for LeakageBound {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        self.scale(rhs)
    }
}

impl Default for LeakageBound {
    fn default() -> Self {
        Self::ZERO
    }
}

impl fmt::Display for LeakageBound {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            write!(f, "0 bits")
        } else if self.is_unbounded() {
            write!(f, "∞ bits")
        } else {
            write!(f, "{:.4} bits", self.bits)
        }
    }
}

// ---------------------------------------------------------------------------
// LeakageVector
// ---------------------------------------------------------------------------

/// Per-cache-set leakage bounds.
///
/// Each entry records the maximum information (in bits) an attacker can learn
/// from observing a specific cache set. This captures the spatial distribution
/// of leakage across the L1 data cache.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LeakageVector {
    /// Leakage bound for each cache set, indexed by set number.
    per_set: Vec<LeakageBound>,
    /// Total accumulated leakage across all sets.
    total: LeakageBound,
}

impl LeakageVector {
    /// Create a zero-leakage vector for the given number of sets.
    pub fn zero(num_sets: u32) -> Self {
        Self {
            per_set: vec![LeakageBound::ZERO; num_sets as usize],
            total: LeakageBound::ZERO,
        }
    }

    /// Create from a vector of per-set bounds.
    pub fn from_bounds(bounds: Vec<LeakageBound>) -> Self {
        let total_bits: f64 = bounds.iter().map(|b| b.bits()).sum();
        Self {
            per_set: bounds,
            total: LeakageBound::new(total_bits),
        }
    }

    /// Number of cache sets.
    pub fn num_sets(&self) -> u32 {
        self.per_set.len() as u32
    }

    /// Get the leakage bound for a specific cache set.
    pub fn get(&self, set: CacheSet) -> LeakageBound {
        self.per_set
            .get(set.as_usize())
            .copied()
            .unwrap_or(LeakageBound::ZERO)
    }

    /// Set the leakage bound for a specific cache set.
    pub fn set(&mut self, cache_set: CacheSet, bound: LeakageBound) {
        if let Some(entry) = self.per_set.get_mut(cache_set.as_usize()) {
            *entry = bound;
        }
        self.recompute_total();
    }

    /// Total leakage across all sets.
    pub fn total(&self) -> LeakageBound {
        self.total
    }

    /// Maximum per-set leakage.
    pub fn max_per_set(&self) -> LeakageBound {
        self.per_set
            .iter()
            .copied()
            .max()
            .unwrap_or(LeakageBound::ZERO)
    }

    /// Number of sets with non-zero leakage.
    pub fn leaking_set_count(&self) -> u32 {
        self.per_set.iter().filter(|b| !b.is_zero()).count() as u32
    }

    /// Indices of sets with non-zero leakage.
    pub fn leaking_sets(&self) -> Vec<CacheSet> {
        self.per_set
            .iter()
            .enumerate()
            .filter(|(_, b)| !b.is_zero())
            .map(|(i, _)| CacheSet::new(i as u32))
            .collect()
    }

    /// Point-wise join (max) of two vectors.
    pub fn join(&self, other: &Self) -> Self {
        let len = self.per_set.len().max(other.per_set.len());
        let mut bounds = Vec::with_capacity(len);
        for i in 0..len {
            let a = self
                .per_set
                .get(i)
                .copied()
                .unwrap_or(LeakageBound::ZERO);
            let b = other
                .per_set
                .get(i)
                .copied()
                .unwrap_or(LeakageBound::ZERO);
            bounds.push(a.max(b));
        }
        Self::from_bounds(bounds)
    }

    /// Point-wise meet (min) of two vectors.
    pub fn meet(&self, other: &Self) -> Self {
        let len = self.per_set.len().min(other.per_set.len());
        let mut bounds = Vec::with_capacity(len);
        for i in 0..len {
            bounds.push(self.per_set[i].min(other.per_set[i]));
        }
        Self::from_bounds(bounds)
    }

    /// Point-wise addition (sequential composition).
    pub fn add(&self, other: &Self) -> Self {
        let len = self.per_set.len().max(other.per_set.len());
        let mut bounds = Vec::with_capacity(len);
        for i in 0..len {
            let a = self.per_set.get(i).copied().unwrap_or(LeakageBound::ZERO);
            let b = other.per_set.get(i).copied().unwrap_or(LeakageBound::ZERO);
            bounds.push(a + b);
        }
        Self::from_bounds(bounds)
    }

    /// Iterate over (set_index, bound) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (CacheSet, LeakageBound)> + '_ {
        self.per_set
            .iter()
            .enumerate()
            .map(|(i, &b)| (CacheSet::new(i as u32), b))
    }

    fn recompute_total(&mut self) {
        let sum: f64 = self.per_set.iter().map(|b| b.bits()).sum();
        self.total = LeakageBound::new(sum);
    }
}

impl fmt::Display for LeakageVector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LeakageVector(total={}, sets={})", self.total, self.num_sets())
    }
}

// ---------------------------------------------------------------------------
// LeakageMetric
// ---------------------------------------------------------------------------

/// Information-theoretic metric used to quantify leakage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LeakageMetric {
    /// Shannon entropy: H(S|O) where S = secret, O = observation.
    /// Measures average-case leakage.
    Shannon,
    /// Min-entropy: H∞(S|O). Measures worst-case single-guess success.
    /// Tighter bound than Shannon for security analysis.
    MinEntropy,
    /// Guessing entropy: G(S|O). Average number of guesses to find secret.
    Guessing,
    /// Max-leakage: maxₒ log(Pr[S=s|O=o] / Pr[S=s]).
    /// Operational measure of one-try success amplification.
    MaxLeakage,
}

impl LeakageMetric {
    /// Human-readable name of the metric.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Shannon => "Shannon entropy",
            Self::MinEntropy => "min-entropy",
            Self::Guessing => "guessing entropy",
            Self::MaxLeakage => "max-leakage",
        }
    }

    /// Whether the metric provides a sound upper bound on operationally
    /// meaningful leakage.
    pub fn is_sound_bound(&self) -> bool {
        matches!(self, Self::MinEntropy | Self::MaxLeakage)
    }

    /// Compute leakage from a probability distribution over observations.
    ///
    /// `probs` is a slice of P(O=oᵢ|S=s) for each distinguishable observation oᵢ.
    /// All entries must be non-negative and sum to 1.
    pub fn compute(&self, probs: &[f64]) -> f64 {
        if probs.is_empty() {
            return 0.0;
        }
        match self {
            Self::Shannon => {
                let mut h = 0.0_f64;
                for &p in probs {
                    if p > 0.0 {
                        h -= p * p.log2();
                    }
                }
                h
            }
            Self::MinEntropy => {
                let max_p = probs.iter().cloned().fold(0.0_f64, f64::max);
                if max_p > 0.0 {
                    -max_p.log2()
                } else {
                    0.0
                }
            }
            Self::Guessing => {
                let mut sorted: Vec<f64> = probs.to_vec();
                sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                let mut g = 0.0_f64;
                for (i, &p) in sorted.iter().enumerate() {
                    g += (i as f64 + 1.0) * p;
                }
                g
            }
            Self::MaxLeakage => {
                let n = probs.len() as f64;
                if n <= 1.0 {
                    0.0
                } else {
                    n.log2()
                }
            }
        }
    }
}

impl fmt::Display for LeakageMetric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ---------------------------------------------------------------------------
// ChannelCapacity
// ---------------------------------------------------------------------------

/// Capacity of a cache side-channel.
///
/// The channel capacity is the maximum mutual information between secret
/// inputs and cache observations, taken over all possible input distributions.
/// It provides an absolute upper bound on leakage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelCapacity {
    /// Capacity in bits.
    pub capacity_bits: f64,
    /// Metric used to compute this capacity.
    pub metric: LeakageMetric,
    /// Number of cache sets contributing to the channel.
    pub active_sets: u32,
    /// Cache associativity (ways) – bounds per-set capacity.
    pub associativity: u32,
    /// Whether the capacity was computed exactly or over-approximated.
    pub is_exact: bool,
}

impl ChannelCapacity {
    /// Compute channel capacity from cache parameters.
    ///
    /// For an LRU cache with `w` ways and `s` active sets, the capacity is
    /// bounded by `s × log₂(w+1)` bits (each set can be in w+1 observable states).
    pub fn from_cache_params(active_sets: u32, associativity: u32, metric: LeakageMetric) -> Self {
        let per_set_states = f64::from(associativity + 1);
        let per_set_bits = per_set_states.log2();
        let capacity = f64::from(active_sets) * per_set_bits;
        Self {
            capacity_bits: capacity,
            metric,
            active_sets,
            associativity,
            is_exact: false,
        }
    }

    /// Create with an exact capacity value.
    pub fn exact(bits: f64, metric: LeakageMetric) -> Self {
        Self {
            capacity_bits: bits,
            metric,
            active_sets: 0,
            associativity: 0,
            is_exact: true,
        }
    }

    /// Whether the bound is tight enough to be useful.
    pub fn is_useful(&self) -> bool {
        self.capacity_bits < 128.0 && self.capacity_bits >= 0.0
    }

    /// The number of distinguishable observations: 2^capacity.
    pub fn observation_count(&self) -> f64 {
        (2.0_f64).powf(self.capacity_bits)
    }

    /// Whether the leakage bound exceeds the channel capacity (unsound).
    pub fn exceeds(&self, bound: &LeakageBound) -> bool {
        bound.bits() > self.capacity_bits
    }
}

impl fmt::Display for ChannelCapacity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "C={:.4} bits ({}, {} sets, {} ways{})",
            self.capacity_bits,
            self.metric,
            self.active_sets,
            self.associativity,
            if self.is_exact { ", exact" } else { "" }
        )
    }
}

// ---------------------------------------------------------------------------
// LeakageClassification
// ---------------------------------------------------------------------------

/// Qualitative classification of leakage severity.
///
/// Provides a coarse-grained summary of leakage that can be computed cheaply
/// and used for quick triage before full quantitative analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum LeakageClassification {
    /// No leakage detected. All cache accesses are secret-independent.
    NoLeak,
    /// Constant-time: the observation is a fixed function of public inputs.
    /// This includes programs that always access the same cache sets.
    Constant,
    /// Some cache accesses depend on secrets, but leakage is bounded.
    SecretDependent,
    /// Full secret leakage – the entire secret can be recovered from observations.
    FullLeak,
}

impl LeakageClassification {
    /// Whether the code is safe to deploy (no secret-dependent leakage).
    pub fn is_safe(&self) -> bool {
        matches!(self, Self::NoLeak | Self::Constant)
    }

    /// Whether any secret information may leak.
    pub fn has_leakage(&self) -> bool {
        matches!(self, Self::SecretDependent | Self::FullLeak)
    }

    /// Classify from a quantitative leakage bound and total secret size.
    pub fn from_bound(bound: LeakageBound, total_secret_bits: u32) -> Self {
        if bound.is_zero() {
            Self::NoLeak
        } else if bound.bits() < 0.01 {
            Self::Constant
        } else if bound.bits() >= f64::from(total_secret_bits) - 0.01 {
            Self::FullLeak
        } else {
            Self::SecretDependent
        }
    }

    /// Join two classifications (take the worse case).
    pub fn join(self, other: Self) -> Self {
        std::cmp::max(self, other)
    }

    /// Security recommendation string.
    pub fn recommendation(&self) -> &'static str {
        match self {
            Self::NoLeak => "Code is safe: no secret-dependent cache behaviour detected.",
            Self::Constant => "Code is constant-time with respect to cache observations.",
            Self::SecretDependent => {
                "WARNING: Secret-dependent cache accesses detected. Consider applying countermeasures."
            }
            Self::FullLeak => {
                "CRITICAL: Full secret leakage possible. Code must not be deployed."
            }
        }
    }

    /// Severity level for reporting.
    pub fn severity(&self) -> &'static str {
        match self {
            Self::NoLeak => "info",
            Self::Constant => "info",
            Self::SecretDependent => "warning",
            Self::FullLeak => "critical",
        }
    }
}

impl fmt::Display for LeakageClassification {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoLeak => write!(f, "no-leak"),
            Self::Constant => write!(f, "constant-time"),
            Self::SecretDependent => write!(f, "secret-dependent"),
            Self::FullLeak => write!(f, "full-leak"),
        }
    }
}

// ---------------------------------------------------------------------------
// Leakage observation model
// ---------------------------------------------------------------------------

/// A single observable event in the cache side-channel.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Observation {
    /// Cache set that was accessed.
    pub cache_set: CacheSet,
    /// Whether this was a cache hit or miss (observable via timing).
    pub hit: bool,
    /// The address that was accessed (may be partially known).
    pub address_bits: u64,
    /// Number of address bits observable by the attacker.
    pub observable_bits: u32,
}

impl Observation {
    /// Create a new observation.
    pub fn new(cache_set: CacheSet, hit: bool, address_bits: u64, observable_bits: u32) -> Self {
        Self {
            cache_set,
            hit,
            address_bits,
            observable_bits,
        }
    }

    /// Leakage from this single observation.
    pub fn leakage_bits(&self) -> f64 {
        f64::from(self.observable_bits)
    }
}

impl fmt::Display for Observation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let hm = if self.hit { "hit" } else { "miss" };
        write!(
            f,
            "obs(set={}, {}, bits={}/{})",
            self.cache_set, hm, self.observable_bits, self.address_bits
        )
    }
}

/// A trace of observations collected during execution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObservationTrace {
    /// Ordered sequence of observations.
    pub observations: Vec<Observation>,
    /// Whether this trace includes speculative observations.
    pub includes_speculative: bool,
    /// Maximum speculation depth in this trace.
    pub max_speculation_depth: u32,
}

impl ObservationTrace {
    /// Create an empty trace.
    pub fn new() -> Self {
        Self {
            observations: Vec::new(),
            includes_speculative: false,
            max_speculation_depth: 0,
        }
    }

    /// Append an observation.
    pub fn push(&mut self, obs: Observation) {
        self.observations.push(obs);
    }

    /// Mark this trace as including speculative observations.
    pub fn set_speculative(&mut self, depth: u32) {
        self.includes_speculative = true;
        self.max_speculation_depth = self.max_speculation_depth.max(depth);
    }

    /// Total number of observations.
    pub fn len(&self) -> usize {
        self.observations.len()
    }

    /// Whether the trace is empty.
    pub fn is_empty(&self) -> bool {
        self.observations.is_empty()
    }

    /// Number of distinct cache sets observed.
    pub fn distinct_sets(&self) -> usize {
        let mut sets: Vec<_> = self.observations.iter().map(|o| o.cache_set).collect();
        sets.sort();
        sets.dedup();
        sets.len()
    }

    /// Total leakage bound from all observations.
    pub fn total_leakage(&self) -> LeakageBound {
        let bits: f64 = self.observations.iter().map(|o| o.leakage_bits()).sum();
        LeakageBound::new(bits)
    }

    /// Merge two traces (sequential composition).
    pub fn merge(&self, other: &Self) -> Self {
        let mut merged = self.clone();
        merged.observations.extend(other.observations.iter().cloned());
        merged.includes_speculative =
            self.includes_speculative || other.includes_speculative;
        merged.max_speculation_depth =
            self.max_speculation_depth.max(other.max_speculation_depth);
        merged
    }
}

impl Default for ObservationTrace {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for ObservationTrace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ObsTrace(len={}, sets={}, spec={})",
            self.len(),
            self.distinct_sets(),
            self.includes_speculative
        )
    }
}

// ---------------------------------------------------------------------------
// LeakageBound as an AbstractDomain
// ---------------------------------------------------------------------------

impl AbstractDomain for LeakageBound {
    fn join(&self, other: &Self) -> Self {
        *self.max(other)
    }

    fn meet(&self, other: &Self) -> Self {
        *self.min(other)
    }

    fn widen(&self, other: &Self) -> Self {
        if other.bits() > self.bits() {
            Self::UNBOUNDED
        } else {
            *self
        }
    }

    fn narrow(&self, other: &Self) -> Self {
        if self.is_unbounded() {
            *other
        } else {
            *self
        }
    }

    fn is_bottom(&self) -> bool {
        self.is_zero()
    }

    fn is_top(&self) -> bool {
        self.is_unbounded()
    }

    fn partial_order(&self, other: &Self) -> bool {
        self.bits() <= other.bits()
    }

    fn bottom() -> Self {
        Self::ZERO
    }

    fn top() -> Self {
        Self::UNBOUNDED
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leakage_bound_basic() {
        let zero = LeakageBound::ZERO;
        assert!(zero.is_zero());
        assert!(!zero.is_unbounded());
        assert_eq!(zero.bits(), 0.0);

        let unb = LeakageBound::UNBOUNDED;
        assert!(unb.is_unbounded());
        assert!(!unb.is_zero());
    }

    #[test]
    fn leakage_bound_from_observations() {
        let b = LeakageBound::from_observations(256);
        assert!((b.bits() - 8.0).abs() < 1e-10);
        assert_eq!(LeakageBound::from_observations(1).bits(), 0.0);
    }

    #[test]
    fn leakage_bound_arithmetic() {
        let a = LeakageBound::from_bits(3);
        let b = LeakageBound::from_bits(5);
        assert_eq!((a + b).bits(), 8.0);
        assert_eq!(a.max(b).bits(), 5.0);
        assert_eq!(a.min(b).bits(), 3.0);
    }

    #[test]
    fn leakage_bound_fraction() {
        let b = LeakageBound::from_bits(4);
        assert!((b.fraction_of(16) - 0.25).abs() < 1e-10);
        assert!((b.fraction_of(2) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn leakage_bound_scale() {
        let b = LeakageBound::from_bits(4);
        let scaled = b.scale(2.0);
        assert_eq!(scaled.bits(), 8.0);
    }

    #[test]
    fn leakage_vector_basic() {
        let mut v = LeakageVector::zero(64);
        assert_eq!(v.num_sets(), 64);
        assert!(v.total().is_zero());
        v.set(CacheSet::new(10), LeakageBound::from_bits(3));
        assert_eq!(v.get(CacheSet::new(10)).bits(), 3.0);
        assert_eq!(v.total().bits(), 3.0);
        assert_eq!(v.leaking_set_count(), 1);
    }

    #[test]
    fn leakage_vector_join() {
        let a = LeakageVector::from_bounds(vec![
            LeakageBound::from_bits(1),
            LeakageBound::from_bits(3),
        ]);
        let b = LeakageVector::from_bounds(vec![
            LeakageBound::from_bits(2),
            LeakageBound::from_bits(1),
        ]);
        let j = a.join(&b);
        assert_eq!(j.get(CacheSet::new(0)).bits(), 2.0);
        assert_eq!(j.get(CacheSet::new(1)).bits(), 3.0);
    }

    #[test]
    fn leakage_metric_shannon() {
        let probs = [0.5, 0.5];
        let h = LeakageMetric::Shannon.compute(&probs);
        assert!((h - 1.0).abs() < 1e-10);
    }

    #[test]
    fn leakage_metric_min_entropy() {
        let probs = [0.5, 0.25, 0.25];
        let h = LeakageMetric::MinEntropy.compute(&probs);
        assert!((h - 1.0).abs() < 1e-10);
    }

    #[test]
    fn leakage_metric_max_leakage() {
        let probs = [0.25, 0.25, 0.25, 0.25];
        let h = LeakageMetric::MaxLeakage.compute(&probs);
        assert!((h - 2.0).abs() < 1e-10);
    }

    #[test]
    fn channel_capacity_from_cache() {
        let cap = ChannelCapacity::from_cache_params(64, 8, LeakageMetric::Shannon);
        assert!(cap.capacity_bits > 0.0);
        assert!(cap.is_useful());
    }

    #[test]
    fn classification_ordering() {
        assert!(LeakageClassification::NoLeak < LeakageClassification::FullLeak);
        assert!(LeakageClassification::Constant < LeakageClassification::SecretDependent);
        assert!(LeakageClassification::NoLeak.is_safe());
        assert!(!LeakageClassification::SecretDependent.is_safe());
    }

    #[test]
    fn classification_from_bound() {
        assert_eq!(
            LeakageClassification::from_bound(LeakageBound::ZERO, 128),
            LeakageClassification::NoLeak
        );
        assert_eq!(
            LeakageClassification::from_bound(LeakageBound::from_bits(4), 128),
            LeakageClassification::SecretDependent
        );
        assert_eq!(
            LeakageClassification::from_bound(LeakageBound::from_bits(128), 128),
            LeakageClassification::FullLeak
        );
    }

    #[test]
    fn observation_trace_merge() {
        let mut t1 = ObservationTrace::new();
        t1.push(Observation::new(CacheSet::new(0), true, 0, 6));
        let mut t2 = ObservationTrace::new();
        t2.push(Observation::new(CacheSet::new(1), false, 0, 6));
        t2.set_speculative(2);
        let merged = t1.merge(&t2);
        assert_eq!(merged.len(), 2);
        assert!(merged.includes_speculative);
        assert_eq!(merged.max_speculation_depth, 2);
    }

    #[test]
    fn leakage_bound_domain() {
        let a = LeakageBound::from_bits(3);
        let b = LeakageBound::from_bits(5);
        assert_eq!(a.join(&b), LeakageBound::from_bits(5));
        assert_eq!(a.meet(&b), LeakageBound::from_bits(3));
        assert!(a.partial_order(&b));
        assert!(!b.partial_order(&a));
        assert!(LeakageBound::ZERO.is_bottom());
        assert!(LeakageBound::UNBOUNDED.is_top());
    }

    #[test]
    fn leakage_bound_display() {
        assert_eq!(format!("{}", LeakageBound::ZERO), "0 bits");
        assert_eq!(format!("{}", LeakageBound::UNBOUNDED), "∞ bits");
        assert!(format!("{}", LeakageBound::from_bits(3)).contains("3.0"));
    }
}
