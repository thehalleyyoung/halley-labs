//! Analysis result caching with in-memory LRU and filesystem persistence.
//!
//! Caching is keyed on topology hash + config hash + analysis mode so that
//! identical configurations yield instant results on repeat runs.  The cache
//! supports partial invalidation when individual services change.

use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::time::{Duration, Instant, SystemTime};

use serde::{Deserialize, Serialize};

use cascade_graph::rtig::RtigGraph;
use cascade_types::config::ConfigManifest;
use cascade_types::report::{AnalysisReport, Finding, Location, Severity};
use cascade_types::service::ServiceId;

// ---------------------------------------------------------------------------
// Cache key / entry
// ---------------------------------------------------------------------------

/// Composite key that uniquely identifies an analysis run.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CacheKey {
    pub topology_hash: u64,
    pub config_hash: u64,
    pub analysis_mode: String,
}

impl CacheKey {
    pub fn new(topology_hash: u64, config_hash: u64, analysis_mode: &str) -> Self {
        Self {
            topology_hash,
            config_hash,
            analysis_mode: analysis_mode.to_string(),
        }
    }

    /// Derive a compact string representation suitable for filesystem paths.
    pub fn as_path_key(&self) -> String {
        format!(
            "{:016x}_{:016x}_{}",
            self.topology_hash, self.config_hash, self.analysis_mode
        )
    }
}

/// Cached analysis result with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    pub key: CacheKey,
    pub result: CachedResult,
    pub created_at_epoch_ms: u64,
    pub ttl_seconds: u64,
    pub access_count: u64,
}

impl CacheEntry {
    /// True if the entry has exceeded its TTL.
    pub fn is_expired(&self) -> bool {
        let now_ms = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        let age_sec = (now_ms.saturating_sub(self.created_at_epoch_ms)) / 1000;
        age_sec > self.ttl_seconds
    }

    /// Remaining TTL in seconds (0 if expired).
    pub fn remaining_ttl_sec(&self) -> u64 {
        let now_ms = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        let age_sec = (now_ms.saturating_sub(self.created_at_epoch_ms)) / 1000;
        self.ttl_seconds.saturating_sub(age_sec)
    }
}

/// Subset of analysis results that we persist in the cache.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedResult {
    pub findings: Vec<Finding>,
    pub summary: String,
    pub exit_code: i32,
    pub service_ids: Vec<String>,
}

// ---------------------------------------------------------------------------
// Cache configuration
// ---------------------------------------------------------------------------

/// Configuration for the analysis cache.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Maximum number of entries in the in-memory cache.
    pub max_entries: usize,
    /// Default time-to-live for cache entries (seconds).
    pub ttl_seconds: u64,
    /// Directory for filesystem-backed cache (optional).
    pub cache_dir: Option<String>,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 128,
            ttl_seconds: 3600,
            cache_dir: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Cache statistics
// ---------------------------------------------------------------------------

/// Counters for cache hits, misses, evictions.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub stores: u64,
    pub invalidations: u64,
    pub size_bytes: u64,
    pub entry_count: usize,
}

impl CacheStats {
    /// Hit rate as a fraction in [0.0, 1.0].
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

// ---------------------------------------------------------------------------
// AnalysisCache (in-memory LRU)
// ---------------------------------------------------------------------------

/// In-memory LRU cache for analysis results.
pub struct AnalysisCache {
    config: CacheConfig,
    entries: HashMap<CacheKey, CacheEntry>,
    order: VecDeque<CacheKey>,
    stats: CacheStats,
}

impl AnalysisCache {
    /// Create a new cache with the given configuration.
    pub fn new(config: CacheConfig) -> Self {
        Self {
            config,
            entries: HashMap::new(),
            order: VecDeque::new(),
            stats: CacheStats::default(),
        }
    }

    /// Create a cache with default settings.
    pub fn with_defaults() -> Self {
        Self::new(CacheConfig::default())
    }

    /// Store a result in the cache under the given key.
    pub fn store(&mut self, key: CacheKey, result: &CachedResult) {
        // If at capacity, evict the least-recently-used entry.
        while self.entries.len() >= self.config.max_entries {
            self.evict_lru();
        }

        let now_ms = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let size_estimate = Self::estimate_size(result);

        let entry = CacheEntry {
            key: key.clone(),
            result: result.clone(),
            created_at_epoch_ms: now_ms,
            ttl_seconds: self.config.ttl_seconds,
            access_count: 0,
        };

        self.entries.insert(key.clone(), entry);
        self.order.retain(|k| k != &key);
        self.order.push_back(key);

        self.stats.stores += 1;
        self.stats.size_bytes += size_estimate;
        self.stats.entry_count = self.entries.len();
    }

    /// Look up a cached result.  Returns `None` on miss or if the entry has
    /// expired.
    pub fn lookup(&mut self, key: &CacheKey) -> Option<&CacheEntry> {
        // Check if entry exists and is not expired.
        let is_valid = self
            .entries
            .get(key)
            .map(|e| !e.is_expired())
            .unwrap_or(false);

        if is_valid {
            // Touch: move to back of LRU order.
            self.order.retain(|k| k != key);
            self.order.push_back(key.clone());

            if let Some(entry) = self.entries.get_mut(key) {
                entry.access_count += 1;
            }
            self.stats.hits += 1;
            self.entries.get(key)
        } else {
            // Remove expired entry if present.
            if self.entries.contains_key(key) {
                self.entries.remove(key);
                self.order.retain(|k| k != key);
                self.stats.entry_count = self.entries.len();
            }
            self.stats.misses += 1;
            None
        }
    }

    /// Explicitly invalidate a specific key.
    pub fn invalidate(&mut self, key: &CacheKey) {
        if self.entries.remove(key).is_some() {
            self.order.retain(|k| k != key);
            self.stats.invalidations += 1;
            self.stats.entry_count = self.entries.len();
        }
    }

    /// Invalidate all cache entries that reference any of the given service
    /// IDs.  This is used when incremental mode detects that specific services
    /// have changed.
    pub fn invalidate_affected(&mut self, changed_services: &[ServiceId]) {
        let changed: HashSet<&str> = changed_services.iter().map(|s| s.as_str()).collect();
        let keys_to_remove: Vec<CacheKey> = self
            .entries
            .iter()
            .filter(|(_, entry)| {
                entry
                    .result
                    .service_ids
                    .iter()
                    .any(|s| changed.contains(s.as_str()))
            })
            .map(|(k, _)| k.clone())
            .collect();

        for key in &keys_to_remove {
            self.entries.remove(key);
            self.stats.invalidations += 1;
        }
        self.order.retain(|k| !keys_to_remove.contains(k));
        self.stats.entry_count = self.entries.len();
    }

    /// Remove all entries.
    pub fn clear(&mut self) {
        let count = self.entries.len() as u64;
        self.entries.clear();
        self.order.clear();
        self.stats.evictions += count;
        self.stats.size_bytes = 0;
        self.stats.entry_count = 0;
    }

    /// Current number of cached entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True if the cache contains no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Snapshot of cache statistics.
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Remove expired entries proactively.
    pub fn gc(&mut self) {
        let expired: Vec<CacheKey> = self
            .entries
            .iter()
            .filter(|(_, e)| e.is_expired())
            .map(|(k, _)| k.clone())
            .collect();
        for key in &expired {
            self.entries.remove(key);
            self.stats.evictions += 1;
        }
        self.order.retain(|k| !expired.contains(k));
        self.stats.entry_count = self.entries.len();
    }

    // ---- Private helpers ----

    fn evict_lru(&mut self) {
        if let Some(oldest_key) = self.order.pop_front() {
            self.entries.remove(&oldest_key);
            self.stats.evictions += 1;
            self.stats.entry_count = self.entries.len();
        }
    }

    fn estimate_size(result: &CachedResult) -> u64 {
        let findings_size: u64 = result
            .findings
            .iter()
            .map(|f| f.description.len() as u64 + f.id.len() as u64 + 64)
            .sum();
        findings_size + result.summary.len() as u64 + 128
    }
}

use std::collections::HashSet;

// ---------------------------------------------------------------------------
// Hashing helpers
// ---------------------------------------------------------------------------

/// Compute a deterministic hash of the graph topology.
///
/// Considers service names, edge structure, retry counts, and timeouts but
/// ignores runtime state (load, health).
pub fn compute_topology_hash(graph: &RtigGraph) -> u64 {
    let mut hasher = SimpleHasher::new();

    let mut service_ids: Vec<&str> = graph.service_ids();
    service_ids.sort();
    for id in &service_ids {
        hasher.write_str(id);
    }

    for edge in graph.edges() {
        hasher.write_str(edge.source.as_str());
        hasher.write_str(edge.target.as_str());
        hasher.write_u32(edge.retry_count);
        hasher.write_u64(edge.timeout_ms);
    }

    hasher.finish()
}

/// Compute a deterministic hash of a config manifest.
///
/// Considers source paths and format strings (not file contents since those
/// are captured by the topology hash).
pub fn compute_config_hash(manifest: &ConfigManifest) -> u64 {
    let mut hasher = SimpleHasher::new();
    let mut paths: Vec<&str> = manifest.file_paths.iter().map(|s| s.as_str()).collect();
    paths.sort();
    for p in &paths {
        hasher.write_str(p);
    }
    for source in &manifest.sources {
        hasher.write_str(&source.to_string());
    }
    hasher.finish()
}

/// Minimal deterministic hasher (FNV-1a 64-bit).
struct SimpleHasher {
    state: u64,
}

impl SimpleHasher {
    const FNV_OFFSET: u64 = 14695981039346656037;
    const FNV_PRIME: u64 = 1099511628211;

    fn new() -> Self {
        Self {
            state: Self::FNV_OFFSET,
        }
    }

    fn write_bytes(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.state ^= b as u64;
            self.state = self.state.wrapping_mul(Self::FNV_PRIME);
        }
    }

    fn write_str(&mut self, s: &str) {
        self.write_bytes(s.as_bytes());
        self.write_bytes(&[0xFF]); // delimiter
    }

    fn write_u32(&mut self, v: u32) {
        self.write_bytes(&v.to_le_bytes());
    }

    fn write_u64(&mut self, v: u64) {
        self.write_bytes(&v.to_le_bytes());
    }

    fn finish(&self) -> u64 {
        self.state
    }
}

// ---------------------------------------------------------------------------
// Filesystem cache (optional persistence)
// ---------------------------------------------------------------------------

/// Filesystem-backed cache that serialises entries as JSON files in a directory.
pub struct FilesystemCache {
    cache_dir: String,
}

impl FilesystemCache {
    /// Create a new filesystem cache.  The directory is created if it does not
    /// exist.
    pub fn new(cache_dir: &str) -> std::io::Result<Self> {
        std::fs::create_dir_all(cache_dir)?;
        Ok(Self {
            cache_dir: cache_dir.to_string(),
        })
    }

    /// Write a cache entry to disk.
    pub fn store(&self, key: &CacheKey, entry: &CacheEntry) -> std::io::Result<()> {
        let path = self.entry_path(key);
        let json = serde_json::to_string(entry)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(&path, json)?;
        Ok(())
    }

    /// Load a cache entry from disk.
    pub fn lookup(&self, key: &CacheKey) -> Option<CacheEntry> {
        let path = self.entry_path(key);
        let data = std::fs::read_to_string(&path).ok()?;
        let entry: CacheEntry = serde_json::from_str(&data).ok()?;
        if entry.is_expired() {
            let _ = std::fs::remove_file(&path);
            return None;
        }
        Some(entry)
    }

    /// Remove a specific entry from disk.
    pub fn invalidate(&self, key: &CacheKey) {
        let path = self.entry_path(key);
        let _ = std::fs::remove_file(&path);
    }

    /// Remove all cached files.
    pub fn clear(&self) -> std::io::Result<()> {
        for entry in std::fs::read_dir(&self.cache_dir)? {
            let entry = entry?;
            if entry.path().extension().map(|e| e == "json").unwrap_or(false) {
                let _ = std::fs::remove_file(entry.path());
            }
        }
        Ok(())
    }

    /// Count the number of cached entries on disk.
    pub fn count(&self) -> usize {
        std::fs::read_dir(&self.cache_dir)
            .map(|entries| {
                entries
                    .filter_map(|e| e.ok())
                    .filter(|e| {
                        e.path()
                            .extension()
                            .map(|ext| ext == "json")
                            .unwrap_or(false)
                    })
                    .count()
            })
            .unwrap_or(0)
    }

    /// Total size in bytes of all cached entries.
    pub fn total_size_bytes(&self) -> u64 {
        std::fs::read_dir(&self.cache_dir)
            .map(|entries| {
                entries
                    .filter_map(|e| e.ok())
                    .filter_map(|e| e.metadata().ok())
                    .map(|m| m.len())
                    .sum()
            })
            .unwrap_or(0)
    }

    fn entry_path(&self, key: &CacheKey) -> String {
        format!("{}/{}.json", self.cache_dir, key.as_path_key())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use cascade_graph::rtig::{DependencyEdgeInfo, ServiceNode};
    use cascade_types::config::ConfigSource;

    fn sample_key() -> CacheKey {
        CacheKey::new(12345, 67890, "standard")
    }

    fn sample_result() -> CachedResult {
        CachedResult {
            findings: vec![Finding {
                id: "F1".into(),
                severity: Severity::Medium,
                title: "test finding".into(),
                description: "test finding".into(),
                evidence: vec![],
                location: Location {
                    file: None,
                    service: None,
                    edge: None,
                    line: None,
                    column: None,
                },
                code_flow: None,
                remediation: None,
            }],
            summary: "1 warning".into(),
            exit_code: 0,
            service_ids: vec!["gateway".into(), "api".into()],
        }
    }

    fn sample_graph() -> RtigGraph {
        let mut g = RtigGraph::new();
        g.add_service(ServiceNode::new("a", 1000));
        g.add_service(ServiceNode::new("b", 500));
        g.add_edge(DependencyEdgeInfo::new("a", "b").with_retry_count(3));
        g
    }

    #[test]
    fn test_store_and_lookup() {
        let mut cache = AnalysisCache::with_defaults();
        let key = sample_key();
        cache.store(key.clone(), &sample_result());
        let entry = cache.lookup(&key);
        assert!(entry.is_some());
        assert_eq!(entry.unwrap().result.exit_code, 0);
    }

    #[test]
    fn test_lookup_miss() {
        let mut cache = AnalysisCache::with_defaults();
        let key = CacheKey::new(999, 999, "quick");
        assert!(cache.lookup(&key).is_none());
    }

    #[test]
    fn test_invalidate() {
        let mut cache = AnalysisCache::with_defaults();
        let key = sample_key();
        cache.store(key.clone(), &sample_result());
        cache.invalidate(&key);
        assert!(cache.lookup(&key).is_none());
    }

    #[test]
    fn test_invalidate_affected() {
        let mut cache = AnalysisCache::with_defaults();
        let key = sample_key();
        cache.store(key.clone(), &sample_result());

        cache.invalidate_affected(&[ServiceId::new("gateway")]);
        assert!(cache.lookup(&key).is_none());
    }

    #[test]
    fn test_invalidate_affected_no_match() {
        let mut cache = AnalysisCache::with_defaults();
        let key = sample_key();
        cache.store(key.clone(), &sample_result());

        cache.invalidate_affected(&[ServiceId::new("unrelated")]);
        assert!(cache.lookup(&key).is_some());
    }

    #[test]
    fn test_lru_eviction() {
        let config = CacheConfig {
            max_entries: 2,
            ttl_seconds: 3600,
            cache_dir: None,
        };
        let mut cache = AnalysisCache::new(config);

        let k1 = CacheKey::new(1, 1, "s");
        let k2 = CacheKey::new(2, 2, "s");
        let k3 = CacheKey::new(3, 3, "s");

        cache.store(k1.clone(), &sample_result());
        cache.store(k2.clone(), &sample_result());
        // This should evict k1.
        cache.store(k3.clone(), &sample_result());

        assert!(cache.lookup(&k1).is_none());
        assert!(cache.lookup(&k2).is_some());
        assert!(cache.lookup(&k3).is_some());
    }

    #[test]
    fn test_lru_touch_on_access() {
        let config = CacheConfig {
            max_entries: 2,
            ttl_seconds: 3600,
            cache_dir: None,
        };
        let mut cache = AnalysisCache::new(config);

        let k1 = CacheKey::new(1, 1, "s");
        let k2 = CacheKey::new(2, 2, "s");
        let k3 = CacheKey::new(3, 3, "s");

        cache.store(k1.clone(), &sample_result());
        cache.store(k2.clone(), &sample_result());
        // Touch k1 so k2 becomes the oldest.
        let _ = cache.lookup(&k1);
        // Insert k3 – should evict k2 (oldest untouched).
        cache.store(k3.clone(), &sample_result());

        assert!(cache.lookup(&k1).is_some());
        assert!(cache.lookup(&k2).is_none());
    }

    #[test]
    fn test_clear() {
        let mut cache = AnalysisCache::with_defaults();
        cache.store(sample_key(), &sample_result());
        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_stats_hits_misses() {
        let mut cache = AnalysisCache::with_defaults();
        let key = sample_key();
        cache.store(key.clone(), &sample_result());

        let _ = cache.lookup(&key); // hit
        let _ = cache.lookup(&CacheKey::new(0, 0, "x")); // miss

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.stores, 1);
        assert!((stats.hit_rate() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_cache_key_path() {
        let key = CacheKey::new(0xABCD, 0x1234, "deep");
        let path = key.as_path_key();
        assert!(path.contains("deep"));
        assert!(path.contains("abcd"));
    }

    #[test]
    fn test_topology_hash_deterministic() {
        let g = sample_graph();
        let h1 = compute_topology_hash(&g);
        let h2 = compute_topology_hash(&g);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_topology_hash_differs() {
        let g1 = sample_graph();
        let mut g2 = RtigGraph::new();
        g2.add_service(ServiceNode::new("x", 100));
        assert_ne!(compute_topology_hash(&g1), compute_topology_hash(&g2));
    }

    #[test]
    fn test_config_hash_deterministic() {
        let manifest = ConfigManifest {
            sources: vec![ConfigSource::Raw {
                format: "yaml".into(),
                content: "test".into(),
            }],
            file_paths: vec!["a.yaml".into()],
        };
        let h1 = compute_config_hash(&manifest);
        let h2 = compute_config_hash(&manifest);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_config_hash_differs() {
        let m1 = ConfigManifest {
            sources: vec![ConfigSource::Raw {
                format: "yaml".into(),
                content: "test".into(),
            }],
            file_paths: vec!["a.yaml".into()],
        };
        let m2 = ConfigManifest {
            sources: vec![ConfigSource::Raw {
                format: "yaml".into(),
                content: "test".into(),
            }],
            file_paths: vec!["b.yaml".into()],
        };
        assert_ne!(compute_config_hash(&m1), compute_config_hash(&m2));
    }

    #[test]
    fn test_gc_removes_expired() {
        let config = CacheConfig {
            max_entries: 10,
            ttl_seconds: 0, // expire immediately
            cache_dir: None,
        };
        let mut cache = AnalysisCache::new(config);
        cache.store(sample_key(), &sample_result());
        // All entries have TTL=0 so should be expired.
        cache.gc();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_filesystem_cache_roundtrip() {
        let dir = std::env::temp_dir().join("cascade_cache_test");
        let _ = std::fs::remove_dir_all(&dir);
        let fs_cache = FilesystemCache::new(dir.to_str().unwrap()).unwrap();

        let key = sample_key();
        let entry = CacheEntry {
            key: key.clone(),
            result: sample_result(),
            created_at_epoch_ms: SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            ttl_seconds: 3600,
            access_count: 0,
        };

        fs_cache.store(&key, &entry).unwrap();
        let loaded = fs_cache.lookup(&key);
        assert!(loaded.is_some());
        assert_eq!(loaded.unwrap().result.exit_code, 0);

        fs_cache.invalidate(&key);
        assert!(fs_cache.lookup(&key).is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_cache_entry_remaining_ttl() {
        let now_ms = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        let entry = CacheEntry {
            key: sample_key(),
            result: sample_result(),
            created_at_epoch_ms: now_ms,
            ttl_seconds: 3600,
            access_count: 0,
        };
        assert!(entry.remaining_ttl_sec() > 3500);
        assert!(!entry.is_expired());
    }

    #[test]
    fn test_access_count_increments() {
        let mut cache = AnalysisCache::with_defaults();
        let key = sample_key();
        cache.store(key.clone(), &sample_result());
        let _ = cache.lookup(&key);
        let _ = cache.lookup(&key);
        let entry = cache.lookup(&key).unwrap();
        assert_eq!(entry.access_count, 3);
    }
}
