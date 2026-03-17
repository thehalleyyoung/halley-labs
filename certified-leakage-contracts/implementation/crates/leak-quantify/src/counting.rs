//! Counting abstractions for cache side-channel analysis.
//!
//! Provides domain-aware counting of distinguishable cache states, used to
//! derive upper bounds on information leakage through cache observations.

use std::collections::BTreeSet;
use std::fmt;

use num_rational::Rational64;
use serde::{Deserialize, Serialize};
use shared_types::{CacheLine, CacheSet, FunctionId, VirtualAddress};

use crate::{QuantifyError, QuantifyResult};

// ---------------------------------------------------------------------------
// Counting Domain
// ---------------------------------------------------------------------------

/// Defines the domain over which distinguishable states are counted.
///
/// A counting domain specifies which cache sets and address ranges are relevant
/// for a given leakage analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CountingDomain {
    /// Set of cache set indices included in the domain.
    pub cache_sets: BTreeSet<u32>,
    /// Address ranges contributing to the domain.
    pub address_ranges: Vec<AddressRange>,
    /// Optional function scope restriction.
    pub function_scope: Option<String>,
    /// Cache line size in bytes (for address-to-set mapping).
    pub line_size: usize,
    /// Number of cache ways (associativity).
    pub associativity: usize,
}

/// An address range `[start, end)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct AddressRange {
    pub start: u64,
    pub end: u64,
}

impl CountingDomain {
    /// Create a new counting domain.
    pub fn new(line_size: usize, associativity: usize) -> Self {
        Self {
            cache_sets: BTreeSet::new(),
            address_ranges: Vec::new(),
            function_scope: None,
            line_size,
            associativity,
        }
    }

    /// Add a cache set to the domain.
    pub fn add_set(&mut self, set_idx: u32) {
        self.cache_sets.insert(set_idx);
    }

    /// Add an address range.
    pub fn add_address_range(&mut self, start: u64, end: u64) {
        self.address_ranges.push(AddressRange { start, end });
    }

    /// Restrict to a specific function.
    pub fn with_function(mut self, func: impl Into<String>) -> Self {
        self.function_scope = Some(func.into());
        self
    }

    /// Number of cache sets in the domain.
    pub fn num_sets(&self) -> usize {
        self.cache_sets.len()
    }

    /// Total number of cache lines across all address ranges.
    pub fn total_cache_lines(&self) -> u64 {
        if self.line_size == 0 {
            return 0;
        }
        self.address_ranges
            .iter()
            .map(|r| (r.end.saturating_sub(r.start)) / self.line_size as u64)
            .sum()
    }

    /// Compute the maximum number of distinguishable states for this domain.
    pub fn max_distinguishable_states(&self) -> DistinguishableStates {
        let sets = self.num_sets();
        let ways = self.associativity;
        // Each set can have up to (ways + 1) distinguishable states
        // (which lines are present, considering LRU ordering).
        // Upper bound: (ways+1)^sets
        let per_set = (ways + 1) as u64;
        let total = per_set.saturating_pow(sets as u32);
        DistinguishableStates {
            count: total,
            log2_count: (total as f64).log2(),
            domain_sets: sets,
            domain_ways: ways,
        }
    }
}

impl fmt::Display for CountingDomain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CountingDomain({} sets, {} ranges, ways={})",
            self.num_sets(),
            self.address_ranges.len(),
            self.associativity
        )
    }
}

// ---------------------------------------------------------------------------
// Distinguishable States
// ---------------------------------------------------------------------------

/// The number of cache states an attacker can distinguish through observations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistinguishableStates {
    /// Raw count of distinguishable states.
    pub count: u64,
    /// log₂ of the count (the leakage upper bound in bits).
    pub log2_count: f64,
    /// Number of cache sets considered.
    pub domain_sets: usize,
    /// Cache associativity.
    pub domain_ways: usize,
}

impl DistinguishableStates {
    /// Create from a raw count.
    pub fn from_count(count: u64) -> Self {
        Self {
            count,
            log2_count: if count > 0 {
                (count as f64).log2()
            } else {
                0.0
            },
            domain_sets: 0,
            domain_ways: 0,
        }
    }

    /// Leakage upper bound in bits: log₂(count).
    pub fn leakage_bits(&self) -> f64 {
        self.log2_count
    }
}

impl fmt::Display for DistinguishableStates {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} states ({:.2} bits)",
            self.count, self.log2_count
        )
    }
}

// ---------------------------------------------------------------------------
// Taint-Restricted Counting
// ---------------------------------------------------------------------------

/// Counting restricted to cache lines reachable from tainted (secret-dependent)
/// data flows.
///
/// Only counts states for cache sets that can be influenced by secret data,
/// yielding tighter bounds than whole-cache counting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaintRestrictedCounting {
    /// The underlying counting domain (restricted to tainted sets).
    pub domain: CountingDomain,
    /// Cache sets that are tainted (influenced by secret data).
    pub tainted_sets: BTreeSet<u32>,
    /// Addresses confirmed as secret-dependent.
    pub tainted_addresses: Vec<u64>,
}

impl TaintRestrictedCounting {
    /// Create a new taint-restricted counter.
    pub fn new(domain: CountingDomain) -> Self {
        Self {
            tainted_sets: BTreeSet::new(),
            tainted_addresses: Vec::new(),
            domain,
        }
    }

    /// Mark a cache set as tainted.
    pub fn taint_set(&mut self, set_idx: u32) {
        self.tainted_sets.insert(set_idx);
        self.domain.add_set(set_idx);
    }

    /// Mark an address as tainted and add its cache set.
    pub fn taint_address(&mut self, addr: u64) {
        self.tainted_addresses.push(addr);
        if self.domain.line_size > 0 {
            let line_bits = (self.domain.line_size as f64).log2() as u32;
            // Simplified set mapping
            let set_idx = ((addr >> line_bits) % self.domain.cache_sets.len().max(1) as u64) as u32;
            self.tainted_sets.insert(set_idx);
        }
    }

    /// Number of tainted cache sets.
    pub fn num_tainted_sets(&self) -> usize {
        self.tainted_sets.len()
    }

    /// Compute the distinguishable states considering only tainted sets.
    pub fn distinguishable_states(&self) -> DistinguishableStates {
        let sets = self.tainted_sets.len();
        let ways = self.domain.associativity;
        let per_set = (ways + 1) as u64;
        let count = per_set.saturating_pow(sets as u32);
        DistinguishableStates {
            count,
            log2_count: if count > 0 {
                (count as f64).log2()
            } else {
                0.0
            },
            domain_sets: sets,
            domain_ways: ways,
        }
    }

    /// Upper bound on leakage in bits.
    pub fn leakage_bound_bits(&self) -> f64 {
        self.distinguishable_states().log2_count
    }
}

impl fmt::Display for TaintRestrictedCounting {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TaintRestricted({} tainted sets, {:.2} bits)",
            self.num_tainted_sets(),
            self.leakage_bound_bits()
        )
    }
}

// ---------------------------------------------------------------------------
// Set Counting
// ---------------------------------------------------------------------------

/// Per-set counting: counts distinguishable states within each individual
/// cache set, then combines.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SetCounting {
    /// Per-set state counts: maps set index → number of distinguishable states.
    pub per_set_counts: Vec<(u32, u64)>,
    /// Cache associativity used.
    pub associativity: usize,
}

impl SetCounting {
    /// Create a new set counter.
    pub fn new(associativity: usize) -> Self {
        Self {
            per_set_counts: Vec::new(),
            associativity,
        }
    }

    /// Record the number of distinguishable states for a cache set.
    pub fn add_set_count(&mut self, set_idx: u32, count: u64) {
        self.per_set_counts.push((set_idx, count));
    }

    /// Compute from a counting domain using default per-set bound.
    pub fn from_domain(domain: &CountingDomain) -> Self {
        let per_set = (domain.associativity + 1) as u64;
        let counts = domain
            .cache_sets
            .iter()
            .map(|&s| (s, per_set))
            .collect();
        Self {
            per_set_counts: counts,
            associativity: domain.associativity,
        }
    }

    /// Total product of per-set counts = total distinguishable states.
    pub fn total_states(&self) -> u64 {
        self.per_set_counts
            .iter()
            .map(|(_, c)| *c)
            .fold(1u64, u64::saturating_mul)
    }

    /// Total leakage bound in bits = Σ log₂(count_i).
    pub fn total_leakage_bits(&self) -> f64 {
        self.per_set_counts
            .iter()
            .map(|(_, c)| (*c as f64).log2())
            .sum()
    }

    /// Number of sets tracked.
    pub fn num_sets(&self) -> usize {
        self.per_set_counts.len()
    }

    /// Exact count as a rational number (for precise compositional reasoning).
    pub fn total_states_rational(&self) -> Rational64 {
        self.per_set_counts
            .iter()
            .map(|(_, c)| Rational64::from_integer(*c as i64))
            .fold(Rational64::from_integer(1), |a, b| a * b)
    }
}

impl fmt::Display for SetCounting {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SetCounting({} sets, {:.2} bits total)",
            self.num_sets(),
            self.total_leakage_bits()
        )
    }
}

// ---------------------------------------------------------------------------
// Count Bound
// ---------------------------------------------------------------------------

/// A provable bound on the number of distinguishable states, with justification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CountBound {
    /// The bound on the count.
    pub count: u64,
    /// Leakage implied by this count bound (log₂ count).
    pub leakage_bits: f64,
    /// Exact rational representation, if available.
    #[serde(skip)]
    pub exact: Option<Rational64>,
    /// How the bound was derived.
    pub derivation: CountBoundDerivation,
    /// Human-readable justification.
    pub justification: String,
}

/// How a count bound was derived.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CountBoundDerivation {
    /// Direct enumeration of reachable states.
    Enumeration,
    /// Taint-restricted counting.
    TaintRestriction,
    /// Per-set product bound.
    SetProduct,
    /// Compositional bound from sub-function bounds.
    Compositional,
    /// Manual annotation or assumption.
    Annotation,
}

impl CountBound {
    /// Create a new count bound.
    pub fn new(count: u64, derivation: CountBoundDerivation, justification: impl Into<String>) -> Self {
        Self {
            count,
            leakage_bits: if count > 0 {
                (count as f64).log2()
            } else {
                0.0
            },
            exact: Some(Rational64::from_integer(count as i64)),
            derivation,
            justification: justification.into(),
        }
    }

    /// Create from a `DistinguishableStates` result.
    pub fn from_distinguishable(
        states: &DistinguishableStates,
        derivation: CountBoundDerivation,
    ) -> Self {
        Self {
            count: states.count,
            leakage_bits: states.log2_count,
            exact: Some(Rational64::from_integer(states.count as i64)),
            derivation,
            justification: format!("{} distinguishable states", states.count),
        }
    }

    /// Tighten two bounds by taking the minimum.
    pub fn tighten(&self, other: &CountBound) -> CountBound {
        if self.count <= other.count {
            self.clone()
        } else {
            other.clone()
        }
    }

    /// The leakage bound in bits.
    pub fn bits(&self) -> f64 {
        self.leakage_bits
    }
}

impl fmt::Display for CountBound {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CountBound({} states, {:.4} bits, {:?})",
            self.count, self.leakage_bits, self.derivation
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_counting_domain_basic() {
        let mut dom = CountingDomain::new(64, 8);
        dom.add_set(0);
        dom.add_set(1);
        dom.add_set(2);
        assert_eq!(dom.num_sets(), 3);
        let states = dom.max_distinguishable_states();
        // (8+1)^3 = 729
        assert_eq!(states.count, 729);
    }

    #[test]
    fn test_distinguishable_from_count() {
        let ds = DistinguishableStates::from_count(256);
        assert!((ds.leakage_bits() - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_taint_restricted_empty() {
        let dom = CountingDomain::new(64, 4);
        let trc = TaintRestrictedCounting::new(dom);
        assert_eq!(trc.num_tainted_sets(), 0);
    }

    #[test]
    fn test_set_counting_product() {
        let mut sc = SetCounting::new(4);
        sc.add_set_count(0, 5);
        sc.add_set_count(1, 5);
        assert_eq!(sc.total_states(), 25);
        assert!((sc.total_leakage_bits() - (25.0_f64).log2()).abs() < 1e-10);
    }

    #[test]
    fn test_count_bound_tighten() {
        let b1 = CountBound::new(100, CountBoundDerivation::Enumeration, "enum");
        let b2 = CountBound::new(64, CountBoundDerivation::TaintRestriction, "taint");
        let tight = b1.tighten(&b2);
        assert_eq!(tight.count, 64);
    }
}
