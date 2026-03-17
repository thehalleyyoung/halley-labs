//! Eigendecomposition caching with LRU eviction and persistence.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use indexmap::IndexMap;
use spectral_types::sparse::CsrMatrix;

use crate::eigensolve::EigenResult;
use crate::error::{Result, SpectralCoreError};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// LRU cache for eigendecomposition results keyed by matrix fingerprint.
pub struct EigenCache {
    entries: IndexMap<u64, CacheEntry>,
    max_entries: usize,
    hits: usize,
    misses: usize,
}

struct CacheEntry {
    result: EigenResult,
    fingerprint: MatrixFingerprint,
    #[allow(dead_code)]
    created_at: std::time::Instant,
    access_count: usize,
}

/// Compact fingerprint of a sparse matrix for cache lookup.
#[derive(Debug, Clone, Hash, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MatrixFingerprint {
    pub rows: usize,
    pub cols: usize,
    pub nnz: usize,
    pub pattern_hash: u64,
    pub value_hash: u64,
}

/// Summary statistics for the cache.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CacheStatistics {
    pub entries: usize,
    pub max_entries: usize,
    pub hits: usize,
    pub misses: usize,
    pub hit_rate: f64,
}

// ---------------------------------------------------------------------------
// Fingerprint computation
// ---------------------------------------------------------------------------

/// FNV-1a style hash combining helper.
fn fnv_mix(state: u64, value: u64) -> u64 {
    const FNV_PRIME: u64 = 0x100000001b3;
    (state ^ value).wrapping_mul(FNV_PRIME)
}

impl MatrixFingerprint {
    /// Hash the sparsity pattern (row_ptr + col_ind) of a CSR matrix.
    fn hash_pattern(matrix: &CsrMatrix<f64>) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325; // FNV offset basis
        for &v in &matrix.row_ptr {
            h = fnv_mix(h, v as u64);
        }
        for &v in &matrix.col_ind {
            h = fnv_mix(h, v as u64);
        }
        h
    }

    /// Hash the nonzero values of a CSR matrix.
    fn hash_values(matrix: &CsrMatrix<f64>) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        for &v in &matrix.values {
            h = fnv_mix(h, v.to_bits());
        }
        h
    }
}

impl EigenCache {
    /// Create an empty cache with the given maximum capacity.
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: IndexMap::with_capacity(max_entries.min(256)),
            max_entries: max_entries.max(1),
            hits: 0,
            misses: 0,
        }
    }

    /// Compute a fingerprint for the given matrix.
    pub fn compute_fingerprint(matrix: &CsrMatrix<f64>) -> MatrixFingerprint {
        MatrixFingerprint {
            rows: matrix.rows,
            cols: matrix.cols,
            nnz: matrix.nnz(),
            pattern_hash: MatrixFingerprint::hash_pattern(matrix),
            value_hash: MatrixFingerprint::hash_values(matrix),
        }
    }

    /// Derive a u64 key from a fingerprint (used as the IndexMap key).
    fn key_of(fp: &MatrixFingerprint) -> u64 {
        let mut h = DefaultHasher::new();
        fp.hash(&mut h);
        h.finish()
    }

    /// Look up a cached result.  On hit the entry is moved to the back
    /// (most-recently-used position).
    pub fn get(&mut self, fingerprint: &MatrixFingerprint) -> Option<&EigenResult> {
        let key = Self::key_of(fingerprint);
        if let Some(entry) = self.entries.get_mut(&key) {
            if entry.fingerprint == *fingerprint {
                entry.access_count += 1;
                self.hits += 1;
                // Move to back for LRU
                let idx = self.entries.get_index_of(&key).unwrap();
                self.entries.move_index(idx, self.entries.len() - 1);
                return Some(&self.entries[&key].result);
            }
        }
        self.misses += 1;
        None
    }

    /// Insert a result into the cache, evicting the least-recently-used entry
    /// if the cache is full.
    pub fn insert(&mut self, fingerprint: MatrixFingerprint, result: EigenResult) {
        let key = Self::key_of(&fingerprint);

        // Evict LRU (front of IndexMap) if at capacity and key is new
        if !self.entries.contains_key(&key) && self.entries.len() >= self.max_entries {
            self.entries.shift_remove_index(0);
        }

        self.entries.insert(
            key,
            CacheEntry {
                result,
                fingerprint,
                created_at: std::time::Instant::now(),
                access_count: 0,
            },
        );
    }

    /// Remove all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.hits = 0;
        self.misses = 0;
    }

    /// Number of entries currently cached.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Hit rate (0.0–1.0).  Returns 0.0 when no queries have been made.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Aggregate cache statistics.
    pub fn statistics(&self) -> CacheStatistics {
        CacheStatistics {
            entries: self.entries.len(),
            max_entries: self.max_entries,
            hits: self.hits,
            misses: self.misses,
            hit_rate: self.hit_rate(),
        }
    }

    /// Serialize the cache contents to a JSON file.
    pub fn save_to_file(&self, path: &std::path::Path) -> Result<()> {
        let items: Vec<(&MatrixFingerprint, &EigenResult)> = self
            .entries
            .values()
            .map(|e| (&e.fingerprint, &e.result))
            .collect();

        let json = serde_json::to_string_pretty(&items).map_err(SpectralCoreError::from)?;
        std::fs::write(path, json).map_err(SpectralCoreError::from)?;
        log::info!("EigenCache: saved {} entries to {:?}", self.len(), path);
        Ok(())
    }

    /// Load cache entries from a JSON file previously written by [`save_to_file`].
    pub fn load_from_file(path: &std::path::Path) -> Result<Self> {
        let data = std::fs::read_to_string(path).map_err(SpectralCoreError::from)?;
        let items: Vec<(MatrixFingerprint, EigenResult)> =
            serde_json::from_str(&data).map_err(SpectralCoreError::from)?;

        let max_entries = items.len().max(64);
        let mut cache = Self::new(max_entries);
        for (fp, result) in items {
            cache.insert(fp, result);
        }

        log::info!("EigenCache: loaded {} entries from {:?}", cache.len(), path);
        Ok(cache)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use spectral_types::dense::DenseMatrix;

    fn dummy_result(tag: &str) -> EigenResult {
        EigenResult {
            eigenvalues: vec![1.0, 2.0],
            eigenvectors: DenseMatrix::zeros(3, 2),
            residuals: vec![1e-10, 2e-10],
            iterations: 42,
            converged: true,
            method_used: tag.to_string(),
            time_ms: 1.0,
        }
    }

    fn small_csr() -> CsrMatrix<f64> {
        CsrMatrix::identity(3)
    }

    fn different_csr() -> CsrMatrix<f64> {
        CsrMatrix::new(3, 3, vec![0, 1, 2, 3], vec![0, 1, 2], vec![2.0, 3.0, 4.0]).unwrap()
    }

    #[test]
    fn test_new_cache() {
        let cache = EigenCache::new(10);
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
        assert_eq!(cache.hit_rate(), 0.0);
    }

    #[test]
    fn test_insert_and_get() {
        let mut cache = EigenCache::new(10);
        let fp = EigenCache::compute_fingerprint(&small_csr());
        cache.insert(fp.clone(), dummy_result("lanczos"));
        assert_eq!(cache.len(), 1);

        let hit = cache.get(&fp);
        assert!(hit.is_some());
        assert_eq!(hit.unwrap().method_used, "lanczos");
    }

    #[test]
    fn test_cache_miss() {
        let mut cache = EigenCache::new(10);
        let fp1 = EigenCache::compute_fingerprint(&small_csr());
        let fp2 = EigenCache::compute_fingerprint(&different_csr());

        cache.insert(fp1, dummy_result("a"));
        assert!(cache.get(&fp2).is_none());
        assert_eq!(cache.misses, 1);
    }

    #[test]
    fn test_lru_eviction() {
        let mut cache = EigenCache::new(2);
        let fp1 = EigenCache::compute_fingerprint(&small_csr());
        let fp2 = EigenCache::compute_fingerprint(&different_csr());

        cache.insert(fp1.clone(), dummy_result("first"));
        cache.insert(fp2.clone(), dummy_result("second"));
        assert_eq!(cache.len(), 2);

        let mat3 =
            CsrMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![9.0, 9.0]).unwrap();
        let fp3 = EigenCache::compute_fingerprint(&mat3);
        cache.insert(fp3, dummy_result("third"));

        assert_eq!(cache.len(), 2);
        assert!(cache.get(&fp1).is_none());
        assert!(cache.get(&fp2).is_some());
    }

    #[test]
    fn test_hit_rate() {
        let mut cache = EigenCache::new(10);
        let fp = EigenCache::compute_fingerprint(&small_csr());
        cache.insert(fp.clone(), dummy_result("x"));

        cache.get(&fp); // hit
        cache.get(&fp); // hit

        let fp2 = EigenCache::compute_fingerprint(&different_csr());
        cache.get(&fp2); // miss

        assert!((cache.hit_rate() - 2.0 / 3.0).abs() < 1e-14);
    }

    #[test]
    fn test_clear() {
        let mut cache = EigenCache::new(10);
        let fp = EigenCache::compute_fingerprint(&small_csr());
        cache.insert(fp, dummy_result("x"));
        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.hit_rate(), 0.0);
    }

    #[test]
    fn test_statistics() {
        let mut cache = EigenCache::new(5);
        let fp = EigenCache::compute_fingerprint(&small_csr());
        cache.insert(fp.clone(), dummy_result("x"));
        cache.get(&fp);

        let stats = cache.statistics();
        assert_eq!(stats.entries, 1);
        assert_eq!(stats.max_entries, 5);
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 0);
        assert!((stats.hit_rate - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_fingerprint_deterministic() {
        let m = small_csr();
        let fp1 = EigenCache::compute_fingerprint(&m);
        let fp2 = EigenCache::compute_fingerprint(&m);
        assert_eq!(fp1, fp2);
    }

    #[test]
    fn test_fingerprint_different_matrices() {
        let fp1 = EigenCache::compute_fingerprint(&small_csr());
        let fp2 = EigenCache::compute_fingerprint(&different_csr());
        assert_ne!(fp1, fp2);
    }

    #[test]
    fn test_save_and_load() {
        let dir = std::env::temp_dir().join("eigen_cache_test");
        std::fs::create_dir_all(&dir).ok();
        let path = dir.join("cache.json");

        {
            let mut cache = EigenCache::new(10);
            let fp = EigenCache::compute_fingerprint(&small_csr());
            cache.insert(fp, dummy_result("saved"));
            cache.save_to_file(&path).unwrap();
        }

        let loaded = EigenCache::load_from_file(&path).unwrap();
        assert_eq!(loaded.len(), 1);

        std::fs::remove_file(&path).ok();
        std::fs::remove_dir(&dir).ok();
    }

    #[test]
    fn test_fingerprint_serialize_roundtrip() {
        let fp = EigenCache::compute_fingerprint(&small_csr());
        let json = serde_json::to_string(&fp).unwrap();
        let fp2: MatrixFingerprint = serde_json::from_str(&json).unwrap();
        assert_eq!(fp, fp2);
    }

    #[test]
    fn test_cache_statistics_serialize() {
        let stats = CacheStatistics {
            entries: 5,
            max_entries: 100,
            hits: 10,
            misses: 3,
            hit_rate: 10.0 / 13.0,
        };
        let json = serde_json::to_string(&stats).unwrap();
        let stats2: CacheStatistics = serde_json::from_str(&json).unwrap();
        assert_eq!(stats2.entries, 5);
        assert_eq!(stats2.hits, 10);
    }
}

