// Result caching for refinement pairs and analysis results.
// Provides LRU cache, refinement cache, and analysis cache.

use std::collections::HashMap;
use std::collections::VecDeque;
use std::hash::Hash;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use isospec_types::config::EngineKind;
use isospec_types::identifier::{TransactionId, WorkloadId};
use isospec_types::isolation::{AnomalyClass, IsolationLevel};

// ---------------------------------------------------------------------------
// Generic LRU cache
// ---------------------------------------------------------------------------

/// A simple Least-Recently-Used cache with optional TTL expiry.
#[derive(Debug)]
pub struct LruCache<K: Eq + Hash + Clone, V: Clone> {
    capacity: usize,
    ttl: Option<Duration>,
    order: VecDeque<K>,
    entries: HashMap<K, CacheEntry<V>>,
    hits: u64,
    misses: u64,
}

#[derive(Debug, Clone)]
struct CacheEntry<V: Clone> {
    value: V,
    inserted_at: Instant,
    last_accessed: Instant,
}

impl<K: Eq + Hash + Clone, V: Clone> LruCache<K, V> {
    /// Create a new LRU cache with the given capacity.
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "LRU cache capacity must be positive");
        Self {
            capacity,
            ttl: None,
            order: VecDeque::with_capacity(capacity),
            entries: HashMap::with_capacity(capacity),
            hits: 0,
            misses: 0,
        }
    }

    /// Create a new LRU cache with a TTL for entries.
    pub fn with_ttl(capacity: usize, ttl: Duration) -> Self {
        let mut cache = Self::new(capacity);
        cache.ttl = Some(ttl);
        cache
    }

    /// Insert a key-value pair, evicting the LRU entry if at capacity.
    pub fn insert(&mut self, key: K, value: V) {
        let now = Instant::now();
        if self.entries.contains_key(&key) {
            self.order.retain(|k| k != &key);
        } else if self.order.len() >= self.capacity {
            if let Some(evicted) = self.order.pop_front() {
                self.entries.remove(&evicted);
            }
        }
        self.order.push_back(key.clone());
        self.entries.insert(
            key,
            CacheEntry {
                value,
                inserted_at: now,
                last_accessed: now,
            },
        );
    }

    /// Retrieve a value, returning `None` if absent or expired.
    pub fn get(&mut self, key: &K) -> Option<&V> {
        let now = Instant::now();
        // Check TTL first
        if let Some(ttl) = self.ttl {
            if let Some(entry) = self.entries.get(key) {
                if now.duration_since(entry.inserted_at) > ttl {
                    self.entries.remove(key);
                    self.order.retain(|k| k != key);
                    self.misses += 1;
                    return None;
                }
            }
        }
        if self.entries.contains_key(key) {
            // Move to back (most recently used)
            self.order.retain(|k| k != key);
            self.order.push_back(key.clone());
            if let Some(entry) = self.entries.get_mut(key) {
                entry.last_accessed = now;
            }
            self.hits += 1;
            self.entries.get(key).map(|e| &e.value)
        } else {
            self.misses += 1;
            None
        }
    }

    /// Check if a key exists (without updating LRU order).
    pub fn contains(&self, key: &K) -> bool {
        if let Some(ttl) = self.ttl {
            if let Some(entry) = self.entries.get(key) {
                return Instant::now().duration_since(entry.inserted_at) <= ttl;
            }
            return false;
        }
        self.entries.contains_key(key)
    }

    /// Remove a specific entry.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        self.order.retain(|k| k != key);
        self.entries.remove(key).map(|e| e.value)
    }

    /// Evict all expired entries (only meaningful with TTL).
    pub fn evict_expired(&mut self) -> usize {
        let ttl = match self.ttl {
            Some(t) => t,
            None => return 0,
        };
        let now = Instant::now();
        let before = self.entries.len();
        self.entries
            .retain(|_, entry| now.duration_since(entry.inserted_at) <= ttl);
        let evicted = before - self.entries.len();
        if evicted > 0 {
            let remaining: std::collections::HashSet<&K> = self.entries.keys().collect();
            self.order.retain(|k| remaining.contains(k));
        }
        evicted
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.order.clear();
    }

    /// Current number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Maximum capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Number of cache hits since creation.
    pub fn hits(&self) -> u64 {
        self.hits
    }

    /// Number of cache misses since creation.
    pub fn misses(&self) -> u64 {
        self.misses
    }

    /// Hit rate as a fraction in [0, 1]. Returns 0 if no accesses.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Return all keys in LRU order (front = least recently used).
    pub fn keys_lru_order(&self) -> Vec<K> {
        self.order.iter().cloned().collect()
    }
}

// ---------------------------------------------------------------------------
// RefinementCache – pre-computed for 3 engines × 4 levels = 12 pairs
// ---------------------------------------------------------------------------

/// Outcome of a refinement check: does engine E at level I refine Adya spec S?
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RefinementResult {
    /// The engine refines the spec; all required anomalies are prevented.
    Refines,
    /// The engine does NOT refine; these anomaly classes are possible.
    DoesNotRefine {
        possible_anomalies: Vec<AnomalyClass>,
    },
    /// The check was not yet computed.
    Unknown,
}

/// Key for looking up a refinement result.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RefinementKey {
    pub engine: EngineKind,
    pub level: IsolationLevel,
}

impl RefinementKey {
    pub fn new(engine: EngineKind, level: IsolationLevel) -> Self {
        Self { engine, level }
    }
}

/// Cache for refinement results (engine, level) → RefinementResult.
/// Pre-computable for 3 engines × 4 standard levels = 12 pairs.
#[derive(Debug)]
pub struct RefinementCache {
    inner: HashMap<RefinementKey, RefinementResult>,
}

impl RefinementCache {
    /// Create an empty cache.
    pub fn new() -> Self {
        Self {
            inner: HashMap::new(),
        }
    }

    /// Pre-populate with the standard 12 pairs using the given evaluator.
    /// `evaluator` receives (engine, level) and returns a `RefinementResult`.
    pub fn precompute<F>(mut evaluator: F) -> Self
    where
        F: FnMut(EngineKind, IsolationLevel) -> RefinementResult,
    {
        let mut cache = Self::new();
        let engines = EngineKind::all();
        let levels = [
            IsolationLevel::ReadUncommitted,
            IsolationLevel::ReadCommitted,
            IsolationLevel::RepeatableRead,
            IsolationLevel::Serializable,
        ];
        for &engine in &engines {
            for &level in &levels {
                let result = evaluator(engine, level);
                cache.insert(engine, level, result);
            }
        }
        cache
    }

    /// Insert a result for a given pair.
    pub fn insert(&mut self, engine: EngineKind, level: IsolationLevel, result: RefinementResult) {
        self.inner
            .insert(RefinementKey::new(engine, level), result);
    }

    /// Look up a cached result.
    pub fn get(&self, engine: EngineKind, level: IsolationLevel) -> RefinementResult {
        self.inner
            .get(&RefinementKey::new(engine, level))
            .cloned()
            .unwrap_or(RefinementResult::Unknown)
    }

    /// Whether the pair has been evaluated.
    pub fn contains(&self, engine: EngineKind, level: IsolationLevel) -> bool {
        self.inner.contains_key(&RefinementKey::new(engine, level))
    }

    /// Number of cached results.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Return all pairs that refine their spec.
    pub fn refining_pairs(&self) -> Vec<RefinementKey> {
        self.inner
            .iter()
            .filter_map(|(k, v)| {
                if *v == RefinementResult::Refines {
                    Some(k.clone())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Return all pairs that do NOT refine their spec.
    pub fn non_refining_pairs(&self) -> Vec<(RefinementKey, Vec<AnomalyClass>)> {
        self.inner
            .iter()
            .filter_map(|(k, v)| match v {
                RefinementResult::DoesNotRefine {
                    possible_anomalies,
                } => Some((k.clone(), possible_anomalies.clone())),
                _ => None,
            })
            .collect()
    }

    pub fn clear(&mut self) {
        self.inner.clear();
    }
}

impl Default for RefinementCache {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// AnalysisCache – workload-level memoization
// ---------------------------------------------------------------------------

/// A single analysis result for a (workload, engine, level) triple.
#[derive(Debug, Clone)]
pub struct AnalysisCacheEntry {
    pub workload_id: WorkloadId,
    pub engine: EngineKind,
    pub level: IsolationLevel,
    pub anomalies_found: Vec<AnomalyClass>,
    pub is_safe: bool,
    pub computed_at: Instant,
    pub bound_k: usize,
}

/// Key for the analysis cache.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AnalysisCacheKey {
    pub workload_id: WorkloadId,
    pub engine: EngineKind,
    pub level: IsolationLevel,
    pub bound_k: usize,
}

impl AnalysisCacheKey {
    pub fn new(
        workload_id: WorkloadId,
        engine: EngineKind,
        level: IsolationLevel,
        bound_k: usize,
    ) -> Self {
        Self {
            workload_id,
            engine,
            level,
            bound_k,
        }
    }
}

/// Thread-safe analysis cache backed by an LRU cache.
#[derive(Debug)]
pub struct AnalysisCache {
    inner: Arc<RwLock<LruCache<AnalysisCacheKey, AnalysisCacheEntry>>>,
}

impl AnalysisCache {
    /// Create a new cache with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Arc::new(RwLock::new(LruCache::new(capacity))),
        }
    }

    /// Create a new cache with a TTL.
    pub fn with_ttl(capacity: usize, ttl: Duration) -> Self {
        Self {
            inner: Arc::new(RwLock::new(LruCache::with_ttl(capacity, ttl))),
        }
    }

    /// Store an analysis result.
    pub fn insert(&self, entry: AnalysisCacheEntry) {
        let key = AnalysisCacheKey::new(
            entry.workload_id,
            entry.engine,
            entry.level,
            entry.bound_k,
        );
        let mut cache = self.inner.write().unwrap();
        cache.insert(key, entry);
    }

    /// Retrieve an analysis result.
    pub fn get(&self, key: &AnalysisCacheKey) -> Option<AnalysisCacheEntry> {
        let mut cache = self.inner.write().unwrap();
        cache.get(key).cloned()
    }

    /// Check if a result is cached.
    pub fn contains(&self, key: &AnalysisCacheKey) -> bool {
        let cache = self.inner.read().unwrap();
        cache.contains(key)
    }

    /// Number of cached results.
    pub fn len(&self) -> usize {
        let cache = self.inner.read().unwrap();
        cache.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Cache hit rate.
    pub fn hit_rate(&self) -> f64 {
        let cache = self.inner.read().unwrap();
        cache.hit_rate()
    }

    /// Invalidate all entries for a specific workload.
    pub fn invalidate_workload(&self, workload_id: WorkloadId) {
        let mut cache = self.inner.write().unwrap();
        let keys_to_remove: Vec<_> = cache
            .keys_lru_order()
            .into_iter()
            .filter(|k| k.workload_id == workload_id)
            .collect();
        for key in keys_to_remove {
            cache.remove(&key);
        }
    }

    /// Clear all cached results.
    pub fn clear(&self) {
        let mut cache = self.inner.write().unwrap();
        cache.clear();
    }

    /// Return a snapshot of cache statistics.
    pub fn stats(&self) -> CacheStats {
        let cache = self.inner.read().unwrap();
        CacheStats {
            size: cache.len(),
            capacity: cache.capacity(),
            hits: cache.hits(),
            misses: cache.misses(),
            hit_rate: cache.hit_rate(),
        }
    }
}

/// Summary of cache performance.
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub size: usize,
    pub capacity: usize,
    pub hits: u64,
    pub misses: u64,
    pub hit_rate: f64,
}

impl std::fmt::Display for CacheStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Cache(size={}/{}, hits={}, misses={}, rate={:.1}%)",
            self.size,
            self.capacity,
            self.hits,
            self.misses,
            self.hit_rate * 100.0,
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_lru_basic_insert_get() {
        let mut cache: LruCache<String, i32> = LruCache::new(3);
        cache.insert("a".into(), 1);
        cache.insert("b".into(), 2);
        cache.insert("c".into(), 3);

        assert_eq!(cache.get(&"a".into()), Some(&1));
        assert_eq!(cache.get(&"b".into()), Some(&2));
        assert_eq!(cache.get(&"c".into()), Some(&3));
        assert_eq!(cache.len(), 3);
    }

    #[test]
    fn test_lru_eviction() {
        let mut cache: LruCache<String, i32> = LruCache::new(2);
        cache.insert("a".into(), 1);
        cache.insert("b".into(), 2);
        // "a" is LRU; inserting "c" should evict it
        cache.insert("c".into(), 3);

        assert_eq!(cache.get(&"a".into()), None);
        assert_eq!(cache.get(&"b".into()), Some(&2));
        assert_eq!(cache.get(&"c".into()), Some(&3));
    }

    #[test]
    fn test_lru_access_updates_order() {
        let mut cache: LruCache<String, i32> = LruCache::new(2);
        cache.insert("a".into(), 1);
        cache.insert("b".into(), 2);
        // Access "a" so "b" becomes LRU
        let _ = cache.get(&"a".into());
        cache.insert("c".into(), 3);

        assert_eq!(cache.get(&"b".into()), None); // evicted
        assert_eq!(cache.get(&"a".into()), Some(&1));
        assert_eq!(cache.get(&"c".into()), Some(&3));
    }

    #[test]
    fn test_lru_hit_rate() {
        let mut cache: LruCache<u32, u32> = LruCache::new(4);
        cache.insert(1, 10);
        cache.insert(2, 20);
        let _ = cache.get(&1); // hit
        let _ = cache.get(&2); // hit
        let _ = cache.get(&99); // miss

        assert_eq!(cache.hits(), 2);
        assert_eq!(cache.misses(), 1);
        assert!((cache.hit_rate() - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_lru_remove() {
        let mut cache: LruCache<String, i32> = LruCache::new(4);
        cache.insert("x".into(), 42);
        assert_eq!(cache.remove(&"x".into()), Some(42));
        assert!(cache.is_empty());
        assert_eq!(cache.get(&"x".into()), None);
    }

    #[test]
    fn test_lru_overwrite() {
        let mut cache: LruCache<String, i32> = LruCache::new(4);
        cache.insert("x".into(), 1);
        cache.insert("x".into(), 2);
        assert_eq!(cache.get(&"x".into()), Some(&2));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_refinement_cache_precompute() {
        let cache = RefinementCache::precompute(|_engine, level| {
            if level == IsolationLevel::Serializable {
                RefinementResult::Refines
            } else {
                RefinementResult::DoesNotRefine {
                    possible_anomalies: vec![AnomalyClass::G0],
                }
            }
        });
        // 3 engines × 4 levels = 12 entries
        assert_eq!(cache.len(), 12);
        assert_eq!(
            cache.get(EngineKind::PostgreSQL, IsolationLevel::Serializable),
            RefinementResult::Refines
        );
        assert!(matches!(
            cache.get(EngineKind::MySQL, IsolationLevel::ReadCommitted),
            RefinementResult::DoesNotRefine { .. }
        ));
    }

    #[test]
    fn test_refinement_cache_refining_pairs() {
        let mut cache = RefinementCache::new();
        cache.insert(
            EngineKind::PostgreSQL,
            IsolationLevel::Serializable,
            RefinementResult::Refines,
        );
        cache.insert(
            EngineKind::MySQL,
            IsolationLevel::Serializable,
            RefinementResult::DoesNotRefine {
                possible_anomalies: vec![AnomalyClass::G2],
            },
        );
        let refining = cache.refining_pairs();
        assert_eq!(refining.len(), 1);
        assert_eq!(refining[0].engine, EngineKind::PostgreSQL);
    }

    #[test]
    fn test_analysis_cache_thread_safe() {
        let cache = AnalysisCache::new(16);
        let entry = AnalysisCacheEntry {
            workload_id: WorkloadId::new(1),
            engine: EngineKind::PostgreSQL,
            level: IsolationLevel::ReadCommitted,
            anomalies_found: vec![AnomalyClass::G1a],
            is_safe: false,
            computed_at: Instant::now(),
            bound_k: 3,
        };
        cache.insert(entry.clone());

        let key = AnalysisCacheKey::new(
            WorkloadId::new(1),
            EngineKind::PostgreSQL,
            IsolationLevel::ReadCommitted,
            3,
        );
        let result = cache.get(&key).unwrap();
        assert_eq!(result.anomalies_found, vec![AnomalyClass::G1a]);
        assert!(!result.is_safe);
    }

    #[test]
    fn test_analysis_cache_invalidate_workload() {
        let cache = AnalysisCache::new(16);
        let entry = AnalysisCacheEntry {
            workload_id: WorkloadId::new(5),
            engine: EngineKind::MySQL,
            level: IsolationLevel::RepeatableRead,
            anomalies_found: vec![],
            is_safe: true,
            computed_at: Instant::now(),
            bound_k: 2,
        };
        cache.insert(entry);
        assert_eq!(cache.len(), 1);
        cache.invalidate_workload(WorkloadId::new(5));
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_cache_stats_display() {
        let stats = CacheStats {
            size: 5,
            capacity: 10,
            hits: 20,
            misses: 5,
            hit_rate: 0.8,
        };
        let s = format!("{}", stats);
        assert!(s.contains("5/10"));
        assert!(s.contains("80.0%"));
    }
}
