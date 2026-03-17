//! # Tainted Abstract Cache-State Domain (D\_cache)
//!
//! Models LRU cache state with per-line taint annotations that track
//! dependence on secret data. Each abstract cache set maintains a sequence
//! of [`AbstractCacheWay`] entries whose ages are represented by the
//! [`AbstractAge`] lattice, and whose taint is given by [`TaintAnnotation`].

use std::fmt;

use serde::{Serialize, Deserialize};
use smallvec::SmallVec;
use thiserror::Error;

use shared_types::{
    BlockId, CacheConfig, CacheSet, CacheTag, SecurityLevel, VirtualAddress,
};

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors from the abstract cache domain.
#[derive(Debug, Error)]
pub enum CacheDomainError {
    #[error("cache set index {index} out of range (max {max})")]
    SetOutOfRange { index: u32, max: u32 },

    #[error("associativity mismatch: expected {expected}, found {found}")]
    AssociativityMismatch { expected: u32, found: u32 },
}

// ---------------------------------------------------------------------------
// AbstractAge
// ---------------------------------------------------------------------------

/// Abstract age of a cache line within its set's LRU stack.
///
/// Ages form a bounded lattice `[0, associativity]` where 0 is the
/// most-recently-used position and `associativity` means "evicted".
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct AbstractAge {
    /// Lower bound on the LRU age (inclusive).
    pub min: u32,
    /// Upper bound on the LRU age (inclusive).
    pub max: u32,
}

impl AbstractAge {
    /// Exact age (single concrete value).
    pub fn exact(age: u32) -> Self {
        Self { min: age, max: age }
    }

    /// Full range `[lo, hi]`.
    pub fn range(min: u32, max: u32) -> Self {
        debug_assert!(min <= max);
        Self { min, max }
    }

    /// Lattice join: widen the interval.
    pub fn join(&self, other: &Self) -> Self {
        Self {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    /// Returns `true` when the line is definitely evicted (age ≥ associativity).
    pub fn is_definitely_evicted(&self, associativity: u32) -> bool {
        self.min >= associativity
    }

    /// Returns `true` when the line is definitely cached (age < associativity).
    pub fn is_definitely_cached(&self, associativity: u32) -> bool {
        self.max < associativity
    }
}

impl Default for AbstractAge {
    fn default() -> Self {
        Self::exact(0)
    }
}

// ---------------------------------------------------------------------------
// TaintSource
// ---------------------------------------------------------------------------

/// Describes where taint (secret-dependence) originates.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TaintSource {
    /// Block in which the taint-introducing instruction resides.
    pub block: BlockId,
    /// Address of the taint-introducing instruction.
    pub address: VirtualAddress,
    /// The security level that induced the taint.
    pub level: SecurityLevel,
}

impl TaintSource {
    pub fn new(block: BlockId, address: VirtualAddress, level: SecurityLevel) -> Self {
        Self { block, address, level }
    }
}

// ---------------------------------------------------------------------------
// TaintAnnotation
// ---------------------------------------------------------------------------

/// Per-cache-line taint annotation recording whether the line's presence
/// or eviction may leak information about secret data.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TaintAnnotation {
    /// Whether this cache line is tainted.
    pub is_tainted: bool,
    /// Sources of the taint (empty when untainted).
    pub sources: SmallVec<[TaintSource; 2]>,
}

impl TaintAnnotation {
    /// Untainted annotation.
    pub fn clean() -> Self {
        Self { is_tainted: false, sources: SmallVec::new() }
    }

    /// Create a tainted annotation from a single source.
    pub fn tainted(source: TaintSource) -> Self {
        Self { is_tainted: true, sources: SmallVec::from_elem(source, 1) }
    }

    /// Join two taint annotations.
    pub fn join(&self, other: &Self) -> Self {
        if !self.is_tainted && !other.is_tainted {
            return Self::clean();
        }
        let mut sources = self.sources.clone();
        for s in &other.sources {
            if !sources.contains(s) {
                sources.push(s.clone());
            }
        }
        Self { is_tainted: true, sources }
    }
}

impl Default for TaintAnnotation {
    fn default() -> Self {
        Self::clean()
    }
}

// ---------------------------------------------------------------------------
// CacheLineState
// ---------------------------------------------------------------------------

/// Abstract state of a single cache line (presence / coherence).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CacheLineState {
    /// The line is not present in the cache.
    Invalid,
    /// The line is present and unmodified.
    Clean,
    /// The line is present and dirty.
    Dirty,
    /// Unknown — may or may not be present.
    Unknown,
}

impl Default for CacheLineState {
    fn default() -> Self {
        Self::Invalid
    }
}

// ---------------------------------------------------------------------------
// AbstractCacheWay
// ---------------------------------------------------------------------------

/// One way of an abstract cache set: tag + state + age + taint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AbstractCacheWay {
    /// Cache tag identifying the line.
    pub tag: CacheTag,
    /// Abstract coherence state.
    pub state: CacheLineState,
    /// Abstract LRU age.
    pub age: AbstractAge,
    /// Taint annotation.
    pub taint: TaintAnnotation,
}

impl AbstractCacheWay {
    /// Create a new way with the given tag, defaulting to clean / age-0 / untainted.
    pub fn new(tag: CacheTag) -> Self {
        Self {
            tag,
            state: CacheLineState::Clean,
            age: AbstractAge::exact(0),
            taint: TaintAnnotation::clean(),
        }
    }

    /// Join two ways element-wise.
    pub fn join(&self, other: &Self) -> Self {
        Self {
            tag: self.tag, // tags must match for a meaningful join
            state: match (self.state, other.state) {
                (a, b) if a == b => a,
                _ => CacheLineState::Unknown,
            },
            age: self.age.join(&other.age),
            taint: self.taint.join(&other.taint),
        }
    }
}

// ---------------------------------------------------------------------------
// AbstractCacheSet
// ---------------------------------------------------------------------------

/// Abstract model of a single cache set (all ways).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AbstractCacheSet {
    /// The cache-set index.
    pub set_index: CacheSet,
    /// Associativity of this set.
    pub associativity: u32,
    /// Abstract ways currently modelled.
    pub ways: SmallVec<[AbstractCacheWay; 8]>,
}

impl AbstractCacheSet {
    /// Create an empty abstract cache set.
    pub fn new(set_index: CacheSet, associativity: u32) -> Self {
        Self {
            set_index,
            associativity,
            ways: SmallVec::new(),
        }
    }

    /// Record an access to the given `tag`, updating ages and taint.
    pub fn access(&mut self, tag: CacheTag, taint: TaintAnnotation) {
        // Age all existing ways.
        for way in self.ways.iter_mut() {
            way.age = AbstractAge::range(
                way.age.min.saturating_add(1),
                way.age.max.saturating_add(1),
            );
        }
        // Evict ways that are definitely beyond associativity.
        self.ways.retain(|w| !w.age.is_definitely_evicted(self.associativity));
        // Insert the new way at MRU.
        self.ways.push(AbstractCacheWay {
            tag,
            state: CacheLineState::Clean,
            age: AbstractAge::exact(0),
            taint,
        });
    }

    /// Join two abstract cache sets.
    pub fn join(&self, other: &Self) -> Self {
        let mut result = self.clone();
        for other_way in &other.ways {
            if let Some(existing) = result.ways.iter_mut().find(|w| w.tag == other_way.tag) {
                *existing = existing.join(other_way);
            } else {
                result.ways.push(other_way.clone());
            }
        }
        result
    }

    /// Returns `true` when no way in this set is tainted.
    pub fn is_untainted(&self) -> bool {
        self.ways.iter().all(|w| !w.taint.is_tainted)
    }
}

// ---------------------------------------------------------------------------
// CacheDomain
// ---------------------------------------------------------------------------

/// The tainted abstract cache-state domain (D\_cache).
///
/// Maintains per-set abstract cache states for the configured cache
/// hierarchy, with taint annotations tracking secret-dependent accesses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheDomain {
    /// Abstract state per cache set, keyed by [`CacheSet`] index.
    pub sets: indexmap::IndexMap<CacheSet, AbstractCacheSet>,
    /// The cache configuration governing geometry and replacement policy.
    pub config: CacheConfig,
}

impl CacheDomain {
    /// Create a new, empty cache domain for the given configuration.
    pub fn new(config: CacheConfig) -> Self {
        Self {
            sets: indexmap::IndexMap::new(),
            config,
        }
    }

    /// Retrieve the abstract state for a specific cache set.
    pub fn set_state(&self, set: CacheSet) -> Option<&AbstractCacheSet> {
        self.sets.get(&set)
    }

    /// Record a memory access to the given cache set and tag.
    pub fn access(
        &mut self,
        set: CacheSet,
        tag: CacheTag,
        taint: TaintAnnotation,
        associativity: u32,
    ) {
        let abs_set = self
            .sets
            .entry(set)
            .or_insert_with(|| AbstractCacheSet::new(set, associativity));
        abs_set.access(tag, taint);
    }

    /// Join two cache domains point-wise across all sets.
    pub fn join(&self, other: &Self) -> Self {
        let mut result = self.clone();
        for (set_idx, other_set) in &other.sets {
            let entry = result
                .sets
                .entry(*set_idx)
                .or_insert_with(|| AbstractCacheSet::new(*set_idx, other_set.associativity));
            *entry = entry.join(other_set);
        }
        result
    }

    /// Collect all cache sets that contain at least one tainted line.
    pub fn tainted_sets(&self) -> Vec<CacheSet> {
        self.sets
            .iter()
            .filter(|(_, s)| !s.is_untainted())
            .map(|(idx, _)| *idx)
            .collect()
    }

    /// Returns the total number of tainted cache lines across all sets.
    pub fn total_tainted_lines(&self) -> usize {
        self.sets
            .values()
            .flat_map(|s| s.ways.iter())
            .filter(|w| w.taint.is_tainted)
            .count()
    }
}
