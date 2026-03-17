//! Merge result caching with LRU eviction.
//!
//! Caches the output of successful merge operations keyed by a hash of the two
//! input states, avoiding redundant re-computation when the same pair is
//! encountered again.

use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::Instant;

use serde::{Deserialize, Serialize};

use negsyn_types::SymbolicState;

use crate::operator::MergeOutput;

// ---------------------------------------------------------------------------
// State signature
// ---------------------------------------------------------------------------

/// A compact, hashable fingerprint of a `SymbolicState`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StateSignature {
    pub id: u64,
    pub pc: u64,
    pub constraint_hash: u64,
    pub register_hash: u64,
    pub cipher_set_hash: u64,
}

impl StateSignature {
    /// Build a signature from a symbolic state.
    pub fn from_state(state: &SymbolicState) -> Self {
        let mut h = DefaultHasher::new();
        for c in &state.constraints {
            format!("{:?}", c).hash(&mut h);
        }
        let constraint_hash = h.finish();

        let mut h = DefaultHasher::new();
        for (k, v) in &state.registers {
            k.hash(&mut h);
            format!("{:?}", v).hash(&mut h);
        }
        let register_hash = h.finish();

        let mut h = DefaultHasher::new();
        for c in &state.negotiation.offered_ciphers {
            c.hash(&mut h);
        }
        let cipher_set_hash = h.finish();

        Self {
            id: state.id,
            pc: state.program_counter,
            constraint_hash,
            register_hash,
            cipher_set_hash,
        }
    }
}

// ---------------------------------------------------------------------------
// Cache key
// ---------------------------------------------------------------------------

/// Key for looking up a cached merge result.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CacheKey {
    pub left: StateSignature,
    pub right: StateSignature,
}

impl CacheKey {
    /// Build a cache key from two states (order-independent).
    pub fn from_states(left: &SymbolicState, right: &SymbolicState) -> Self {
        let ls = StateSignature::from_state(left);
        let rs = StateSignature::from_state(right);
        // Canonical ordering so (a,b) and (b,a) hit the same entry.
        if ls.id <= rs.id {
            Self { left: ls, right: rs }
        } else {
            Self { left: rs, right: ls }
        }
    }

    pub fn new(left: StateSignature, right: StateSignature) -> Self {
        Self { left, right }
    }
}

// ---------------------------------------------------------------------------
// Cache entry (internal)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct CacheEntry {
    output: MergeOutput,
    last_access_seq: u64,
    hit_count: u64,
}

// ---------------------------------------------------------------------------
// Merge cache
// ---------------------------------------------------------------------------

/// LRU-style cache for merge results.
pub struct MergeCache {
    capacity: usize,
    entries: HashMap<CacheKey, CacheEntry>,
    sequence: u64,
    total_hits: u64,
    total_misses: u64,
}

impl MergeCache {
    /// Create a new cache with the given maximum capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            entries: HashMap::with_capacity(capacity),
            sequence: 0,
            total_hits: 0,
            total_misses: 0,
        }
    }

    /// Look up a cached merge result.
    pub fn get(&self, key: &CacheKey) -> Option<&MergeOutput> {
        // NOTE: we take `&self` (not `&mut self`) to match operator.rs usage
        // where the cache is behind an immutable reference inside `if let`.
        // Hit-count tracking happens separately via `record_hit`.
        self.entries.get(key).map(|e| &e.output)
    }

    /// Record a cache hit for bookkeeping (call after a successful `get`).
    pub fn record_hit(&mut self, key: &CacheKey) {
        self.sequence += 1;
        self.total_hits += 1;
        if let Some(entry) = self.entries.get_mut(key) {
            entry.last_access_seq = self.sequence;
            entry.hit_count += 1;
        }
    }

    /// Insert a merge result into the cache, evicting the LRU entry if full.
    pub fn put(&mut self, key: CacheKey, output: MergeOutput) {
        self.sequence += 1;

        if self.entries.len() >= self.capacity && !self.entries.contains_key(&key) {
            self.evict_lru();
        }

        self.entries.insert(
            key,
            CacheEntry {
                output,
                last_access_seq: self.sequence,
                hit_count: 0,
            },
        );
    }

    /// Number of entries currently in the cache.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn hit_count(&self) -> u64 {
        self.total_hits
    }

    pub fn miss_count(&self) -> u64 {
        self.total_misses
    }

    pub fn record_miss(&mut self) {
        self.total_misses += 1;
    }

    /// Hit rate over total lookups.
    pub fn hit_rate(&self) -> f64 {
        let total = self.total_hits + self.total_misses;
        if total == 0 {
            return 0.0;
        }
        self.total_hits as f64 / total as f64
    }

    /// Remove all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.sequence = 0;
    }

    // Evict the least-recently-used entry.
    fn evict_lru(&mut self) {
        if let Some(lru_key) = self
            .entries
            .iter()
            .min_by_key(|(_, e)| e.last_access_seq)
            .map(|(k, _)| k.clone())
        {
            self.entries.remove(&lru_key);
        }
    }
}

impl std::fmt::Debug for MergeCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MergeCache")
            .field("capacity", &self.capacity)
            .field("len", &self.entries.len())
            .field("hits", &self.total_hits)
            .field("misses", &self.total_misses)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cost::MergeCost;
    use crate::operator::{CipherSelectionMode, MergeMetadata, MergeOutput};
    use negsyn_types::{HandshakePhase, NegotiationState, ProtocolVersion, SymbolicState};

    fn make_test_state(id: u64, phase: HandshakePhase, ciphers: &[u16]) -> SymbolicState {
        let mut neg = NegotiationState::new(phase, ProtocolVersion::Tls12);
        neg.offered_ciphers = ciphers.iter().copied().collect();
        SymbolicState::new(id, 0x1000, neg)
    }

    fn make_test_output(left: &SymbolicState, right: &SymbolicState) -> MergeOutput {
        let merged = make_test_state(999, HandshakePhase::ClientHello, &[0x002F]);
        MergeOutput {
            merged_state: merged,
            left_id: left.id,
            right_id: right.id,
            cost: MergeCost::zero(),
            metadata: MergeMetadata {
                constraint_count_before: 0,
                constraint_count_after: 1,
                ite_nodes_created: 0,
                memory_regions_merged: 0,
                registers_merged: 0,
                extension_conflicts: 0,
                cipher_selection_mode: CipherSelectionMode::Identical,
                merge_time_us: 0,
                was_cached: false,
            },
        }
    }

    #[test]
    fn test_state_signature_deterministic() {
        let s = make_test_state(1, HandshakePhase::ClientHello, &[0x002F]);
        let sig1 = StateSignature::from_state(&s);
        let sig2 = StateSignature::from_state(&s);
        assert_eq!(sig1, sig2);
    }

    #[test]
    fn test_state_signature_differs_by_id() {
        let s1 = make_test_state(1, HandshakePhase::ClientHello, &[0x002F]);
        let s2 = make_test_state(2, HandshakePhase::ClientHello, &[0x002F]);
        let sig1 = StateSignature::from_state(&s1);
        let sig2 = StateSignature::from_state(&s2);
        assert_ne!(sig1, sig2);
    }

    #[test]
    fn test_cache_key_order_independent() {
        let s1 = make_test_state(1, HandshakePhase::ClientHello, &[0x002F]);
        let s2 = make_test_state(2, HandshakePhase::ClientHello, &[0x002F]);
        let k1 = CacheKey::from_states(&s1, &s2);
        let k2 = CacheKey::from_states(&s2, &s1);
        assert_eq!(k1, k2);
    }

    #[test]
    fn test_cache_put_and_get() {
        let mut cache = MergeCache::new(16);
        let s1 = make_test_state(1, HandshakePhase::ClientHello, &[0x002F]);
        let s2 = make_test_state(2, HandshakePhase::ClientHello, &[0x002F]);
        let key = CacheKey::from_states(&s1, &s2);
        let output = make_test_output(&s1, &s2);

        cache.put(key.clone(), output.clone());
        assert_eq!(cache.len(), 1);

        let got = cache.get(&key);
        assert!(got.is_some());
        assert_eq!(got.unwrap().left_id, 1);
    }

    #[test]
    fn test_cache_miss() {
        let cache = MergeCache::new(16);
        let s1 = make_test_state(1, HandshakePhase::ClientHello, &[0x002F]);
        let s2 = make_test_state(2, HandshakePhase::ClientHello, &[0x002F]);
        let key = CacheKey::from_states(&s1, &s2);
        assert!(cache.get(&key).is_none());
    }

    #[test]
    fn test_cache_eviction() {
        let mut cache = MergeCache::new(2);
        let s1 = make_test_state(1, HandshakePhase::ClientHello, &[0x002F]);
        let s2 = make_test_state(2, HandshakePhase::ClientHello, &[0x002F]);
        let s3 = make_test_state(3, HandshakePhase::ClientHello, &[0x002F]);
        let s4 = make_test_state(4, HandshakePhase::ClientHello, &[0x002F]);

        cache.put(CacheKey::from_states(&s1, &s2), make_test_output(&s1, &s2));
        cache.put(CacheKey::from_states(&s3, &s4), make_test_output(&s3, &s4));
        assert_eq!(cache.len(), 2);

        // This should evict the oldest entry (s1, s2).
        let s5 = make_test_state(5, HandshakePhase::ClientHello, &[0x002F]);
        cache.put(CacheKey::from_states(&s1, &s5), make_test_output(&s1, &s5));
        assert_eq!(cache.len(), 2);

        // The evicted entry should be gone.
        assert!(cache.get(&CacheKey::from_states(&s1, &s2)).is_none());
    }

    #[test]
    fn test_cache_clear() {
        let mut cache = MergeCache::new(16);
        let s1 = make_test_state(1, HandshakePhase::ClientHello, &[0x002F]);
        let s2 = make_test_state(2, HandshakePhase::ClientHello, &[0x002F]);
        cache.put(CacheKey::from_states(&s1, &s2), make_test_output(&s1, &s2));
        assert!(!cache.is_empty());
        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_cache_hit_rate() {
        let mut cache = MergeCache::new(16);
        let s1 = make_test_state(1, HandshakePhase::ClientHello, &[0x002F]);
        let s2 = make_test_state(2, HandshakePhase::ClientHello, &[0x002F]);
        let key = CacheKey::from_states(&s1, &s2);
        cache.put(key.clone(), make_test_output(&s1, &s2));

        cache.record_hit(&key);
        cache.record_miss();
        assert!((cache.hit_rate() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cache_debug() {
        let cache = MergeCache::new(8);
        let dbg = format!("{:?}", cache);
        assert!(dbg.contains("MergeCache"));
    }
}
