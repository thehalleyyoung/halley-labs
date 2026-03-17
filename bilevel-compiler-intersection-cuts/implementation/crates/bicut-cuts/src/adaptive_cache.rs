//! Adaptive cut cache with hit-rate monitoring and oracle performance tracking.
//!
//! Wraps the separation oracle with an LRU cache that automatically resizes
//! based on observed hit rates. Provides latency histograms, cache-miss
//! analysis, and runtime metrics that make oracle performance observable
//! rather than assumed.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::Cut;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the adaptive cut cache.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveCacheConfig {
    /// Initial cache capacity (number of entries).
    pub initial_capacity: usize,
    /// Minimum cache capacity (floor for downsizing).
    pub min_capacity: usize,
    /// Maximum cache capacity (ceiling for growth).
    pub max_capacity: usize,
    /// Target hit-rate window size (number of recent lookups to consider).
    pub hit_rate_window: usize,
    /// If hit rate falls below this, grow the cache.
    pub grow_threshold: f64,
    /// If hit rate exceeds this, consider shrinking.
    pub shrink_threshold: f64,
    /// Growth factor when expanding.
    pub grow_factor: f64,
    /// Shrink factor when contracting.
    pub shrink_factor: f64,
    /// Interval (in lookups) between resize evaluations.
    pub resize_interval: usize,
    /// Discretization precision for cache keys (decimal places).
    pub key_precision: u32,
    /// Number of latency histogram buckets.
    pub latency_buckets: usize,
    /// Maximum latency tracked (microseconds).
    pub max_latency_us: u64,
}

impl Default for AdaptiveCacheConfig {
    fn default() -> Self {
        Self {
            initial_capacity: 4096,
            min_capacity: 256,
            max_capacity: 65536,
            hit_rate_window: 500,
            grow_threshold: 0.60,
            shrink_threshold: 0.95,
            grow_factor: 1.5,
            shrink_factor: 0.75,
            resize_interval: 200,
            key_precision: 8,
            latency_buckets: 16,
            max_latency_us: 100_000,
        }
    }
}

// ---------------------------------------------------------------------------
// Latency histogram
// ---------------------------------------------------------------------------

/// Fixed-bucket histogram for oracle call latencies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyHistogram {
    /// Bucket boundaries in microseconds (len = num_buckets + 1).
    boundaries: Vec<u64>,
    /// Counts per bucket.
    counts: Vec<u64>,
    /// Running sum for mean computation (microseconds).
    total_us: u64,
    /// Total number of recorded samples.
    total_samples: u64,
    /// Minimum observed latency (microseconds).
    min_us: u64,
    /// Maximum observed latency (microseconds).
    max_us: u64,
}

impl LatencyHistogram {
    pub fn new(num_buckets: usize, max_us: u64) -> Self {
        let step = max_us / num_buckets as u64;
        let boundaries: Vec<u64> = (0..=num_buckets).map(|i| i as u64 * step).collect();
        Self {
            counts: vec![0; num_buckets + 1], // +1 for overflow bucket
            boundaries,
            total_us: 0,
            total_samples: 0,
            min_us: u64::MAX,
            max_us: 0,
        }
    }

    pub fn record(&mut self, latency_us: u64) {
        self.total_us += latency_us;
        self.total_samples += 1;
        self.min_us = self.min_us.min(latency_us);
        self.max_us = self.max_us.max(latency_us);

        let bucket = self
            .boundaries
            .windows(2)
            .position(|w| latency_us >= w[0] && latency_us < w[1])
            .unwrap_or(self.counts.len() - 1);
        self.counts[bucket] += 1;
    }

    pub fn mean_us(&self) -> f64 {
        if self.total_samples == 0 {
            return 0.0;
        }
        self.total_us as f64 / self.total_samples as f64
    }

    /// Approximate percentile (linear interpolation within buckets).
    pub fn percentile(&self, p: f64) -> u64 {
        if self.total_samples == 0 {
            return 0;
        }
        let target = (p * self.total_samples as f64).ceil() as u64;
        let mut cumulative = 0u64;
        for (i, &count) in self.counts.iter().enumerate() {
            cumulative += count;
            if cumulative >= target {
                return if i < self.boundaries.len() - 1 {
                    self.boundaries[i]
                } else {
                    self.max_us
                };
            }
        }
        self.max_us
    }

    pub fn min_us(&self) -> u64 {
        if self.total_samples == 0 {
            0
        } else {
            self.min_us
        }
    }

    pub fn max_us(&self) -> u64 {
        self.max_us
    }

    pub fn sample_count(&self) -> u64 {
        self.total_samples
    }

    pub fn reset(&mut self) {
        self.counts.iter_mut().for_each(|c| *c = 0);
        self.total_us = 0;
        self.total_samples = 0;
        self.min_us = u64::MAX;
        self.max_us = 0;
    }
}

// ---------------------------------------------------------------------------
// Hit-rate tracker (rolling window)
// ---------------------------------------------------------------------------

/// Tracks hit/miss events in a rolling window for accurate recent hit-rate.
#[derive(Debug, Clone)]
struct HitRateTracker {
    /// Ring buffer: true = hit, false = miss.
    window: Vec<bool>,
    /// Current write position.
    pos: usize,
    /// Number of entries written (capped at window size).
    count: usize,
    /// Running hit count within the window.
    hits_in_window: usize,
    /// Lifetime counters.
    total_hits: u64,
    total_lookups: u64,
}

impl HitRateTracker {
    fn new(window_size: usize) -> Self {
        Self {
            window: vec![false; window_size],
            pos: 0,
            count: 0,
            hits_in_window: 0,
            total_hits: 0,
            total_lookups: 0,
        }
    }

    fn record(&mut self, hit: bool) {
        self.total_lookups += 1;
        if hit {
            self.total_hits += 1;
        }

        // If buffer is full, remove the oldest entry's contribution.
        if self.count == self.window.len() {
            if self.window[self.pos] {
                self.hits_in_window -= 1;
            }
        } else {
            self.count += 1;
        }

        self.window[self.pos] = hit;
        if hit {
            self.hits_in_window += 1;
        }
        self.pos = (self.pos + 1) % self.window.len();
    }

    fn windowed_hit_rate(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        self.hits_in_window as f64 / self.count as f64
    }

    fn lifetime_hit_rate(&self) -> f64 {
        if self.total_lookups == 0 {
            return 0.0;
        }
        self.total_hits as f64 / self.total_lookups as f64
    }

    fn total_lookups(&self) -> u64 {
        self.total_lookups
    }

    fn total_hits(&self) -> u64 {
        self.total_hits
    }
}

// ---------------------------------------------------------------------------
// Cache entry and key
// ---------------------------------------------------------------------------

/// A cached separation result: the cut(s) generated for a given point.
#[derive(Debug, Clone)]
struct CachedSeparationResult {
    cuts: Vec<Cut>,
    phi_x: f64,
    last_access: u64,
    access_count: u64,
    creation_time: Instant,
}

fn discretize_point(point: &[f64], precision: u32) -> Vec<i64> {
    let scale = 10f64.powi(precision as i32);
    point.iter().map(|&v| (v * scale).round() as i64).collect()
}

// ---------------------------------------------------------------------------
// Adaptive cut cache
// ---------------------------------------------------------------------------

/// Runtime metrics snapshot for external consumption.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheMetrics {
    /// Current cache capacity.
    pub capacity: usize,
    /// Current number of entries stored.
    pub size: usize,
    /// Rolling-window hit rate.
    pub windowed_hit_rate: f64,
    /// Lifetime hit rate.
    pub lifetime_hit_rate: f64,
    /// Total lookups (lifetime).
    pub total_lookups: u64,
    /// Total hits (lifetime).
    pub total_hits: u64,
    /// Total misses (lifetime).
    pub total_misses: u64,
    /// Number of cache resizes performed.
    pub resize_count: u32,
    /// Number of evictions performed.
    pub eviction_count: u64,
    /// Oracle call latency: mean (µs).
    pub oracle_latency_mean_us: f64,
    /// Oracle call latency: p50 (µs).
    pub oracle_latency_p50_us: u64,
    /// Oracle call latency: p95 (µs).
    pub oracle_latency_p95_us: u64,
    /// Oracle call latency: p99 (µs).
    pub oracle_latency_p99_us: u64,
    /// Oracle call latency: max (µs).
    pub oracle_latency_max_us: u64,
    /// Total oracle calls (cache misses that invoked the oracle).
    pub oracle_calls: u64,
    /// Estimated oracle calls saved by caching.
    pub oracle_calls_saved: u64,
    /// Cache reduction ratio: oracle_calls_saved / total_lookups.
    pub oracle_reduction_ratio: f64,
}

/// Adaptive cut cache with LRU eviction and hit-rate-driven resizing.
///
/// Wraps oracle-generated separation results. On each lookup, records
/// hit/miss and oracle latency. Periodically evaluates the rolling
/// hit rate and grows or shrinks the cache to meet target performance.
pub struct AdaptiveCutCache {
    config: AdaptiveCacheConfig,
    /// LRU map: discretized point → cached result.
    entries: HashMap<Vec<i64>, CachedSeparationResult>,
    /// Current capacity.
    capacity: usize,
    /// Monotonic access counter for LRU ordering.
    access_counter: u64,
    /// Rolling hit-rate tracker.
    hit_tracker: HitRateTracker,
    /// Oracle call latency histogram.
    oracle_latency: LatencyHistogram,
    /// Number of resize operations performed.
    resize_count: u32,
    /// Total evictions.
    eviction_count: u64,
    /// Lookups since last resize evaluation.
    lookups_since_resize: usize,
}

impl AdaptiveCutCache {
    pub fn new(config: AdaptiveCacheConfig) -> Self {
        let capacity = config.initial_capacity;
        let window = config.hit_rate_window;
        let buckets = config.latency_buckets;
        let max_lat = config.max_latency_us;
        Self {
            config,
            entries: HashMap::with_capacity(capacity),
            capacity,
            access_counter: 0,
            hit_tracker: HitRateTracker::new(window),
            oracle_latency: LatencyHistogram::new(buckets, max_lat),
            resize_count: 0,
            eviction_count: 0,
            lookups_since_resize: 0,
        }
    }

    /// Look up cached cuts for a given point. Returns `None` on cache miss.
    pub fn lookup(&mut self, point: &[f64]) -> Option<Vec<Cut>> {
        let key = discretize_point(point, self.config.key_precision);
        self.access_counter += 1;
        let counter = self.access_counter;

        let result = if let Some(entry) = self.entries.get_mut(&key) {
            entry.last_access = counter;
            entry.access_count += 1;
            self.hit_tracker.record(true);
            Some(entry.cuts.clone())
        } else {
            self.hit_tracker.record(false);
            None
        };
        self.maybe_resize();
        result
    }

    /// Insert a separation result after an oracle call.
    /// `oracle_duration` is the wall-clock time of the oracle call that
    /// produced these cuts (used for latency tracking).
    pub fn insert(&mut self, point: &[f64], cuts: Vec<Cut>, phi_x: f64, oracle_duration: Duration) {
        let key = discretize_point(point, self.config.key_precision);

        self.oracle_latency
            .record(oracle_duration.as_micros() as u64);

        // Evict if at capacity.
        while self.entries.len() >= self.capacity {
            self.evict_lru();
        }

        self.access_counter += 1;
        self.entries.insert(
            key,
            CachedSeparationResult {
                cuts,
                phi_x,
                last_access: self.access_counter,
                access_count: 1,
                creation_time: Instant::now(),
            },
        );
    }

    /// Evict the least-recently-used entry.
    fn evict_lru(&mut self) {
        let lru_key = self
            .entries
            .iter()
            .min_by_key(|(_, v)| v.last_access)
            .map(|(k, _)| k.clone());
        if let Some(key) = lru_key {
            self.entries.remove(&key);
            self.eviction_count += 1;
        }
    }

    /// Check whether the cache should be resized and do so if needed.
    fn maybe_resize(&mut self) {
        self.lookups_since_resize += 1;
        if self.lookups_since_resize < self.config.resize_interval {
            return;
        }
        self.lookups_since_resize = 0;

        // Need enough data in the window to make a decision.
        if self.hit_tracker.total_lookups() < self.config.hit_rate_window as u64 / 2 {
            return;
        }

        let rate = self.hit_tracker.windowed_hit_rate();

        if rate < self.config.grow_threshold && self.capacity < self.config.max_capacity {
            let new_cap = ((self.capacity as f64 * self.config.grow_factor).ceil() as usize)
                .min(self.config.max_capacity);
            if new_cap > self.capacity {
                log::info!(
                    "AdaptiveCutCache: growing {} → {} (hit rate {:.1}% < {:.1}%)",
                    self.capacity,
                    new_cap,
                    rate * 100.0,
                    self.config.grow_threshold * 100.0,
                );
                self.capacity = new_cap;
                self.resize_count += 1;
            }
        } else if rate > self.config.shrink_threshold && self.capacity > self.config.min_capacity {
            let new_cap = ((self.capacity as f64 * self.config.shrink_factor).floor() as usize)
                .max(self.config.min_capacity);
            if new_cap < self.capacity {
                log::info!(
                    "AdaptiveCutCache: shrinking {} → {} (hit rate {:.1}% > {:.1}%)",
                    self.capacity,
                    new_cap,
                    rate * 100.0,
                    self.config.shrink_threshold * 100.0,
                );
                self.capacity = new_cap;
                // Evict excess entries.
                while self.entries.len() > self.capacity {
                    self.evict_lru();
                }
                self.resize_count += 1;
            }
        }
    }

    /// Snapshot of all runtime metrics.
    pub fn metrics(&self) -> CacheMetrics {
        let total_lookups = self.hit_tracker.total_lookups();
        let total_hits = self.hit_tracker.total_hits();
        let total_misses = total_lookups - total_hits;
        let oracle_calls = self.oracle_latency.sample_count();
        let oracle_calls_saved = total_hits;
        let reduction = if total_lookups > 0 {
            oracle_calls_saved as f64 / total_lookups as f64
        } else {
            0.0
        };

        CacheMetrics {
            capacity: self.capacity,
            size: self.entries.len(),
            windowed_hit_rate: self.hit_tracker.windowed_hit_rate(),
            lifetime_hit_rate: self.hit_tracker.lifetime_hit_rate(),
            total_lookups,
            total_hits,
            total_misses,
            resize_count: self.resize_count,
            eviction_count: self.eviction_count,
            oracle_latency_mean_us: self.oracle_latency.mean_us(),
            oracle_latency_p50_us: self.oracle_latency.percentile(0.50),
            oracle_latency_p95_us: self.oracle_latency.percentile(0.95),
            oracle_latency_p99_us: self.oracle_latency.percentile(0.99),
            oracle_latency_max_us: self.oracle_latency.max_us(),
            oracle_calls,
            oracle_calls_saved,
            oracle_reduction_ratio: reduction,
        }
    }

    /// Current capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Current number of cached entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Clear all cached entries and reset metrics.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.access_counter = 0;
        self.hit_tracker = HitRateTracker::new(self.config.hit_rate_window);
        self.oracle_latency.reset();
        self.resize_count = 0;
        self.eviction_count = 0;
        self.lookups_since_resize = 0;
    }

    /// Log a human-readable summary of current metrics.
    pub fn log_summary(&self) {
        let m = self.metrics();
        log::info!(
            "AdaptiveCutCache: size={}/{}, hit_rate={:.1}% (window) / {:.1}% (lifetime), \
             oracle_calls={}, saved={}, reduction={:.1}×, resizes={}, \
             latency p50={}µs p95={}µs p99={}µs",
            m.size,
            m.capacity,
            m.windowed_hit_rate * 100.0,
            m.lifetime_hit_rate * 100.0,
            m.oracle_calls,
            m.oracle_calls_saved,
            if m.oracle_calls > 0 {
                (m.oracle_calls + m.oracle_calls_saved) as f64 / m.oracle_calls as f64
            } else {
                f64::INFINITY
            },
            m.resize_count,
            m.oracle_latency_p50_us,
            m.oracle_latency_p95_us,
            m.oracle_latency_p99_us,
        );
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use bicut_types::ConstraintSense;

    fn make_cut(idx: usize, coeff: f64) -> Cut {
        Cut::new(
            vec![0.0; idx + 1]
                .into_iter()
                .enumerate()
                .map(|(i, _)| if i == idx { coeff } else { 0.0 })
                .collect(),
            1.0,
            ConstraintSense::Ge,
        )
    }

    fn default_cache() -> AdaptiveCutCache {
        AdaptiveCutCache::new(AdaptiveCacheConfig {
            initial_capacity: 8,
            min_capacity: 4,
            max_capacity: 64,
            hit_rate_window: 10,
            grow_threshold: 0.40,
            shrink_threshold: 0.95,
            resize_interval: 5,
            ..AdaptiveCacheConfig::default()
        })
    }

    #[test]
    fn test_empty_cache() {
        let cache = default_cache();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.capacity(), 8);
    }

    #[test]
    fn test_insert_and_lookup() {
        let mut cache = default_cache();
        let point = vec![1.0, 2.0, 3.0];
        let cuts = vec![make_cut(0, 1.0)];
        cache.insert(&point, cuts.clone(), 5.0, Duration::from_micros(100));

        let result = cache.lookup(&point);
        assert!(result.is_some());
        assert_eq!(result.unwrap().len(), 1);
    }

    #[test]
    fn test_cache_miss() {
        let mut cache = default_cache();
        let point = vec![1.0, 2.0, 3.0];
        let result = cache.lookup(&point);
        assert!(result.is_none());
    }

    #[test]
    fn test_hit_rate_tracking() {
        let mut cache = default_cache();
        let p1 = vec![1.0, 2.0];
        let p2 = vec![3.0, 4.0];
        cache.insert(&p1, vec![make_cut(0, 1.0)], 1.0, Duration::from_micros(50));

        // Miss
        cache.lookup(&p2);
        // Hit
        cache.lookup(&p1);
        // Hit
        cache.lookup(&p1);

        let m = cache.metrics();
        assert_eq!(m.total_lookups, 3);
        assert_eq!(m.total_hits, 2);
        assert!((m.lifetime_hit_rate - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_lru_eviction() {
        let mut cache = AdaptiveCutCache::new(AdaptiveCacheConfig {
            initial_capacity: 2,
            min_capacity: 2,
            max_capacity: 4,
            hit_rate_window: 100,
            resize_interval: 1000, // prevent auto-resize
            ..AdaptiveCacheConfig::default()
        });

        let p1 = vec![1.0];
        let p2 = vec![2.0];
        let p3 = vec![3.0];
        let dur = Duration::from_micros(10);

        cache.insert(&p1, vec![make_cut(0, 1.0)], 1.0, dur);
        cache.insert(&p2, vec![make_cut(1, 2.0)], 2.0, dur);
        // This should evict p1 (LRU).
        cache.insert(&p3, vec![make_cut(2, 3.0)], 3.0, dur);

        assert!(cache.lookup(&p1).is_none());
        assert!(cache.lookup(&p2).is_some());
        assert!(cache.lookup(&p3).is_some());
    }

    #[test]
    fn test_latency_histogram() {
        let mut hist = LatencyHistogram::new(10, 1000);
        hist.record(50);
        hist.record(100);
        hist.record(950);

        assert_eq!(hist.sample_count(), 3);
        assert_eq!(hist.min_us(), 50);
        assert_eq!(hist.max_us(), 950);
        assert!((hist.mean_us() - 366.666).abs() < 1.0);
    }

    #[test]
    fn test_auto_grow() {
        let mut cache = AdaptiveCutCache::new(AdaptiveCacheConfig {
            initial_capacity: 4,
            min_capacity: 2,
            max_capacity: 64,
            hit_rate_window: 10,
            grow_threshold: 0.50,
            shrink_threshold: 0.98,
            grow_factor: 2.0,
            shrink_factor: 0.5,
            resize_interval: 5,
            ..AdaptiveCacheConfig::default()
        });

        // Generate all misses to trigger growth.
        for i in 0..20 {
            let point = vec![i as f64 * 100.0];
            cache.lookup(&point);
        }

        // With 0% hit rate, cache should have grown.
        assert!(cache.capacity() > 4);
    }

    #[test]
    fn test_metrics_snapshot() {
        let mut cache = default_cache();
        let point = vec![1.0, 2.0];
        cache.insert(
            &point,
            vec![make_cut(0, 1.0)],
            1.0,
            Duration::from_micros(200),
        );
        cache.lookup(&point);

        let m = cache.metrics();
        assert_eq!(m.size, 1);
        assert_eq!(m.oracle_calls, 1);
        assert_eq!(m.oracle_latency_mean_us, 200.0);
        assert!(m.oracle_reduction_ratio > 0.0);
    }

    #[test]
    fn test_clear() {
        let mut cache = default_cache();
        cache.insert(
            &[1.0],
            vec![make_cut(0, 1.0)],
            1.0,
            Duration::from_micros(10),
        );
        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.metrics().total_lookups, 0);
    }

    #[test]
    fn test_oracle_reduction_ratio() {
        let mut cache = default_cache();
        let p = vec![1.0];
        cache.insert(&p, vec![make_cut(0, 1.0)], 1.0, Duration::from_micros(50));

        // 4 lookups: all hits
        for _ in 0..4 {
            cache.lookup(&p);
        }

        let m = cache.metrics();
        // 1 oracle call, 4 hits → saved 4 calls → reduction = 4/4 = 1.0
        assert_eq!(m.oracle_calls, 1);
        assert_eq!(m.oracle_calls_saved, 4);
        assert!((m.oracle_reduction_ratio - 1.0).abs() < 1e-10);
    }
}
