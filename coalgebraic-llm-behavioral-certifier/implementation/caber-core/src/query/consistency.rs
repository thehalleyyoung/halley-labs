//! Consistency monitoring module for CABER.
//!
//! Implements re-query consistency checks, drift detection via KL divergence,
//! total-variation distance, and Jensen-Shannon divergence, plus validity-window
//! management and abort/reaudit recommendations.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Configuration knobs for the consistency monitor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyConfig {
    /// Fraction of queries that will be re-issued (0.0–1.0).
    pub requery_rate: f64,
    /// Per-query drift score above which the query is flagged.
    pub drift_threshold: f64,
    /// Minimum number of response samples before we compare distributions.
    pub min_samples_for_comparison: usize,
    /// KL-divergence ceiling.
    pub kl_divergence_threshold: f64,
    /// Total-variation distance ceiling.
    pub tv_distance_threshold: f64,
    /// If true, flag drift even when only a single metric exceeds threshold.
    pub alert_on_any_drift: bool,
    /// Whether to truncate the validity window when drift is detected.
    pub window_truncation_enabled: bool,
    /// After this many drift events the monitor recommends abort.
    pub max_drift_events_before_abort: usize,
}

impl Default for ConsistencyConfig {
    fn default() -> Self {
        Self {
            requery_rate: 0.05,
            drift_threshold: 0.1,
            min_samples_for_comparison: 5,
            kl_divergence_threshold: 0.1,
            tv_distance_threshold: 0.15,
            alert_on_any_drift: false,
            window_truncation_enabled: true,
            max_drift_events_before_abort: 10,
        }
    }
}

/// Outcome of a consistency check across all monitored queries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyResult {
    pub overall_consistent: bool,
    pub num_queries_checked: usize,
    pub num_drifted: usize,
    pub max_drift_score: f64,
    pub average_drift_score: f64,
    pub drifted_queries: Vec<DriftedQuery>,
    pub recommendation: ConsistencyRecommendation,
}

/// Detail record for a single query that drifted.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftedQuery {
    pub query_key: String,
    pub drift_score: f64,
    pub baseline_distribution: Vec<(String, f64)>,
    pub requery_distribution: Vec<(String, f64)>,
}

/// A single drift event recorded by the monitor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftEvent {
    pub query_key: String,
    pub drift_score: f64,
    pub detected_at: String,
    pub drift_type: DriftType,
}

/// Classification of drift.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum DriftType {
    /// The response distribution shifted.
    Distribution,
    /// A previously-seen response is now absent.
    Absence,
    /// A response appeared that was never in the baseline.
    NewResponse,
    /// The magnitude of probability mass moved significantly.
    Magnitude,
}

/// Action recommendation after a consistency check.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ConsistencyRecommendation {
    Continue,
    ReduceWindow,
    Abort,
    Reaudit,
}

/// Describes the time interval during which results remain valid.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidityWindow {
    pub start: String,
    pub end: Option<String>,
    pub truncated: bool,
    pub reason: Option<String>,
}

/// Aggregate statistics about consistency monitoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyStats {
    pub total_baseline_queries: usize,
    pub total_requery_queries: usize,
    pub total_drift_detected: usize,
    pub max_observed_drift: f64,
    pub average_drift: f64,
    pub drift_rate: f64,
}

impl Default for ConsistencyStats {
    fn default() -> Self {
        Self {
            total_baseline_queries: 0,
            total_requery_queries: 0,
            total_drift_detected: 0,
            max_observed_drift: 0.0,
            average_drift: 0.0,
            drift_rate: 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Helper / standalone functions
// ---------------------------------------------------------------------------

/// Deterministic 64-bit FNV-1a hash of a string.
pub fn hash_string(s: &str) -> u64 {
    const FNV_OFFSET: u64 = 14695981039346656037;
    const FNV_PRIME: u64 = 1099511628211;
    let mut h = FNV_OFFSET;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// Build a normalised probability distribution from a bag of responses.
///
/// Counts occurrences, sorts alphabetically, then divides by total.
pub fn responses_to_distribution(responses: &[String]) -> Vec<(String, f64)> {
    if responses.is_empty() {
        return Vec::new();
    }
    let mut counts: HashMap<String, usize> = HashMap::new();
    for r in responses {
        *counts.entry(r.clone()).or_insert(0) += 1;
    }
    let total = responses.len() as f64;
    let mut dist: Vec<(String, f64)> = counts
        .into_iter()
        .map(|(k, c)| (k, c as f64 / total))
        .collect();
    dist.sort_by(|a, b| a.0.cmp(&b.0));
    dist
}

/// Merge the support of two distributions, applying additive smoothing so that
/// no probability is exactly zero.  Returns `(p_smoothed, q_smoothed)` aligned
/// on the same support.
fn align_and_smooth(
    p: &[(String, f64)],
    q: &[(String, f64)],
    epsilon: f64,
) -> (Vec<(String, f64)>, Vec<(String, f64)>) {
    let mut support: Vec<String> = Vec::new();
    let p_map: HashMap<&str, f64> = p.iter().map(|(k, v)| (k.as_str(), *v)).collect();
    let q_map: HashMap<&str, f64> = q.iter().map(|(k, v)| (k.as_str(), *v)).collect();

    for (k, _) in p.iter() {
        if !support.contains(k) {
            support.push(k.clone());
        }
    }
    for (k, _) in q.iter() {
        if !support.contains(k) {
            support.push(k.clone());
        }
    }
    support.sort();

    let n = support.len() as f64;
    let smooth = |map: &HashMap<&str, f64>| -> Vec<(String, f64)> {
        let raw: Vec<f64> = support
            .iter()
            .map(|k| *map.get(k.as_str()).unwrap_or(&0.0) + epsilon)
            .collect();
        let z: f64 = raw.iter().sum();
        support
            .iter()
            .zip(raw.iter())
            .map(|(k, &v)| (k.clone(), v / z))
            .collect()
    };

    let _ = n; // suppress warning; epsilon handles smoothing
    (smooth(&p_map), smooth(&q_map))
}

/// KL divergence D_KL(P || Q) with additive smoothing (epsilon = 1e-10).
pub fn kl_divergence_discrete(p: &[(String, f64)], q: &[(String, f64)]) -> f64 {
    if p.is_empty() || q.is_empty() {
        return 0.0;
    }
    let eps = 1e-10;
    let (ps, qs) = align_and_smooth(p, q, eps);
    let mut kl = 0.0_f64;
    for (i, (_, pi)) in ps.iter().enumerate() {
        let qi = qs[i].1;
        if *pi > 0.0 {
            kl += pi * (pi / qi).ln();
        }
    }
    kl.max(0.0)
}

/// Total-variation distance  TV(P, Q) = 0.5 * Σ |p_i - q_i|.
pub fn total_variation_discrete(p: &[(String, f64)], q: &[(String, f64)]) -> f64 {
    if p.is_empty() && q.is_empty() {
        return 0.0;
    }
    let eps = 1e-10;
    let (ps, qs) = align_and_smooth(p, q, eps);
    let mut tv = 0.0_f64;
    for (i, (_, pi)) in ps.iter().enumerate() {
        let qi = qs[i].1;
        tv += (pi - qi).abs();
    }
    tv * 0.5
}

/// Jensen-Shannon divergence  JSD(P, Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
/// where M = 0.5*(P+Q).
pub fn jensen_shannon_divergence(p: &[(String, f64)], q: &[(String, f64)]) -> f64 {
    if p.is_empty() || q.is_empty() {
        return 0.0;
    }
    let eps = 1e-10;
    let (ps, qs) = align_and_smooth(p, q, eps);

    // Build M = (P+Q)/2
    let m: Vec<(String, f64)> = ps
        .iter()
        .zip(qs.iter())
        .map(|((k, pi), (_, qi))| (k.clone(), (pi + qi) / 2.0))
        .collect();

    0.5 * kl_divergence_raw(&ps, &m) + 0.5 * kl_divergence_raw(&qs, &m)
}

/// Internal KL divergence on already-aligned, already-smoothed vectors.
fn kl_divergence_raw(p: &[(String, f64)], q: &[(String, f64)]) -> f64 {
    let mut kl = 0.0_f64;
    for (i, (_, pi)) in p.iter().enumerate() {
        let qi = q[i].1;
        if *pi > 0.0 && qi > 0.0 {
            kl += pi * (pi / qi).ln();
        }
    }
    kl.max(0.0)
}

/// Simple ISO-8601 timestamp string (UTC-ish, no TZ suffix – good enough for
/// ordering inside a single process).
fn now_iso() -> String {
    // We deliberately avoid pulling in `chrono` – just use a monotonic counter
    // seeded from the hash of the current drift-events length.  In production
    // you would use real wall-clock time; here we produce a fixed-format string
    // that sorts correctly within a session.
    //
    // For deterministic tests we simply return a fixed string; the caller can
    // override via `DriftEvent.detected_at` if needed.
    "2025-01-01T00:00:00Z".to_string()
}

/// Minimal xorshift64 PRNG.
struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Returns a float in [0, 1).
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }
}

// ---------------------------------------------------------------------------
// ConsistencyMonitor
// ---------------------------------------------------------------------------

/// Central monitor that records baseline and re-query responses, detects drift,
/// and manages validity windows.
pub struct ConsistencyMonitor {
    pub config: ConsistencyConfig,
    pub baseline_responses: HashMap<String, Vec<String>>,
    pub requery_responses: HashMap<String, Vec<String>>,
    pub drift_events: Vec<DriftEvent>,
    pub stats: ConsistencyStats,
    pub rng_seed: u64,
    /// Internal monotonic counter used for generating unique timestamps.
    event_counter: u64,
    /// ISO string marking the start of the current validity window.
    window_start: String,
    /// Set to `Some(...)` when the window is truncated due to drift.
    window_end: Option<String>,
    /// Reason the window was truncated, if applicable.
    window_reason: Option<String>,
}

impl ConsistencyMonitor {
    // -- construction -------------------------------------------------------

    pub fn new(config: ConsistencyConfig) -> Self {
        Self {
            config,
            baseline_responses: HashMap::new(),
            requery_responses: HashMap::new(),
            drift_events: Vec::new(),
            stats: ConsistencyStats::default(),
            rng_seed: 42,
            event_counter: 0,
            window_start: now_iso(),
            window_end: None,
            window_reason: None,
        }
    }

    // -- recording ----------------------------------------------------------

    /// Store a baseline (first-pass) response for a query key.
    pub fn record_baseline(&mut self, query_key: &str, response: &str) {
        self.baseline_responses
            .entry(query_key.to_string())
            .or_insert_with(Vec::new)
            .push(response.to_string());
        self.stats.total_baseline_queries = self.baseline_responses.len();
    }

    /// Store a re-query response for an already-seen query key.
    pub fn record_requery(&mut self, query_key: &str, response: &str) {
        self.requery_responses
            .entry(query_key.to_string())
            .or_insert_with(Vec::new)
            .push(response.to_string());
        self.stats.total_requery_queries = self.requery_responses.len();
    }

    // -- candidate selection ------------------------------------------------

    /// Deterministically select approximately `requery_rate * total_queries`
    /// query keys for re-querying.  Uses the monitor's RNG seed so results are
    /// reproducible.
    pub fn select_requery_candidates(&self, total_queries: usize) -> Vec<String> {
        let mut keys: Vec<String> = self.baseline_responses.keys().cloned().collect();
        keys.sort(); // deterministic order

        let target = ((total_queries as f64) * self.config.requery_rate).ceil() as usize;
        let target = target.min(keys.len());

        if target == 0 {
            return Vec::new();
        }

        let mut rng = Xorshift64::new(self.rng_seed);
        // Fisher-Yates partial shuffle to pick `target` elements.
        let mut indices: Vec<usize> = (0..keys.len()).collect();
        let mut selected: Vec<String> = Vec::with_capacity(target);
        for i in 0..target {
            let j = i + (rng.next_u64() as usize) % (indices.len() - i);
            indices.swap(i, j);
            selected.push(keys[indices[i]].clone());
        }
        selected.sort();
        selected
    }

    // -- consistency checking -----------------------------------------------

    /// Run the full consistency check across all query keys that have both
    /// baseline and re-query data.
    pub fn check_consistency(&mut self) -> ConsistencyResult {
        let mut drifted_queries: Vec<DriftedQuery> = Vec::new();
        let mut scores: Vec<f64> = Vec::new();
        let mut checked = 0usize;

        // Collect keys to check (must have both baseline + requery data).
        let keys: Vec<String> = self
            .requery_responses
            .keys()
            .filter(|k| self.baseline_responses.contains_key(*k))
            .cloned()
            .collect();

        for key in &keys {
            let baseline = match self.baseline_responses.get(key) {
                Some(v) => v,
                None => continue,
            };
            let requery = match self.requery_responses.get(key) {
                Some(v) => v,
                None => continue,
            };

            // Skip if we don't have enough samples.
            if baseline.len() < self.config.min_samples_for_comparison
                && requery.len() < self.config.min_samples_for_comparison
            {
                continue;
            }

            checked += 1;

            let score = self.compute_drift_score_internal(baseline, requery);
            scores.push(score);

            if score > self.config.drift_threshold {
                let b_dist = responses_to_distribution(baseline);
                let r_dist = responses_to_distribution(requery);

                // Classify drift type.
                let drift_type = classify_drift(&b_dist, &r_dist, score, self.config.drift_threshold);

                drifted_queries.push(DriftedQuery {
                    query_key: key.clone(),
                    drift_score: score,
                    baseline_distribution: b_dist,
                    requery_distribution: r_dist,
                });

                self.event_counter += 1;
                self.drift_events.push(DriftEvent {
                    query_key: key.clone(),
                    drift_score: score,
                    detected_at: format!(
                        "2025-01-01T00:00:{:02}Z",
                        self.event_counter.min(59)
                    ),
                    drift_type,
                });
            }
        }

        let num_drifted = drifted_queries.len();
        let max_drift = scores.iter().cloned().fold(0.0_f64, f64::max);
        let avg_drift = if scores.is_empty() {
            0.0
        } else {
            scores.iter().sum::<f64>() / scores.len() as f64
        };

        // Update stats.
        self.stats.total_drift_detected += num_drifted;
        if max_drift > self.stats.max_observed_drift {
            self.stats.max_observed_drift = max_drift;
        }
        self.stats.average_drift = avg_drift;
        self.stats.drift_rate = if checked > 0 {
            num_drifted as f64 / checked as f64
        } else {
            0.0
        };

        let overall_consistent = if self.config.alert_on_any_drift {
            num_drifted == 0
        } else {
            num_drifted == 0 || max_drift <= self.config.drift_threshold
        };

        let recommendation = self.recommend(num_drifted, max_drift);

        // Possibly truncate validity window.
        if !overall_consistent && self.config.window_truncation_enabled && self.window_end.is_none()
        {
            self.window_end = Some(format!(
                "2025-01-01T00:00:{:02}Z",
                self.event_counter.min(59)
            ));
            self.window_reason = Some(format!(
                "Drift detected in {} queries (max score {:.4})",
                num_drifted, max_drift
            ));
        }

        ConsistencyResult {
            overall_consistent,
            num_queries_checked: checked,
            num_drifted,
            max_drift_score: max_drift,
            average_drift_score: avg_drift,
            drifted_queries,
            recommendation,
        }
    }

    /// Compute the drift score for a single query key.
    ///
    /// The score is the maximum of the KL divergence and the total-variation
    /// distance between the baseline and re-query distributions. Returns 0.0
    /// if either side has no data.
    pub fn compute_drift_score(&self, query_key: &str) -> f64 {
        let baseline = match self.baseline_responses.get(query_key) {
            Some(v) => v,
            None => return 0.0,
        };
        let requery = match self.requery_responses.get(query_key) {
            Some(v) => v,
            None => return 0.0,
        };
        self.compute_drift_score_internal(baseline, requery)
    }

    /// Internal drift score computation from two response vectors.
    fn compute_drift_score_internal(&self, baseline: &[String], requery: &[String]) -> f64 {
        let p = responses_to_distribution(baseline);
        let q = responses_to_distribution(requery);

        let kl = kl_divergence_discrete(&p, &q);
        let tv = total_variation_discrete(&p, &q);
        let jsd = jensen_shannon_divergence(&p, &q);

        // Composite score: max of the three normalised metrics.
        // KL and JSD are unbounded above, but in practice for discrete
        // distributions with smoothing they stay moderate.  We cap to 1.0 for
        // TV (which is naturally in [0,1]), and leave KL/JSD raw.
        let score = kl.max(tv).max(jsd);
        score
    }

    /// Returns `true` if the overall drift rate exceeds the configured
    /// threshold or the number of drift events exceeds the abort ceiling.
    pub fn is_drifting(&self) -> bool {
        if self.drift_events.len() >= self.config.max_drift_events_before_abort {
            return true;
        }
        self.stats.drift_rate > self.config.drift_threshold
    }

    /// Immutable view of all drift events.
    pub fn drift_events(&self) -> &[DriftEvent] {
        &self.drift_events
    }

    /// Current validity window.
    pub fn validity_window(&self) -> ValidityWindow {
        ValidityWindow {
            start: self.window_start.clone(),
            end: self.window_end.clone(),
            truncated: self.window_end.is_some(),
            reason: self.window_reason.clone(),
        }
    }

    /// Immutable view of aggregate statistics.
    pub fn stats(&self) -> &ConsistencyStats {
        &self.stats
    }

    /// Reset all recorded data, drift events, and statistics.  Config is kept.
    pub fn reset(&mut self) {
        self.baseline_responses.clear();
        self.requery_responses.clear();
        self.drift_events.clear();
        self.stats = ConsistencyStats::default();
        self.event_counter = 0;
        self.window_start = now_iso();
        self.window_end = None;
        self.window_reason = None;
    }

    // -- private helpers ----------------------------------------------------

    fn recommend(
        &self,
        num_drifted: usize,
        max_drift: f64,
    ) -> ConsistencyRecommendation {
        let total_events = self.drift_events.len() + num_drifted;
        if total_events >= self.config.max_drift_events_before_abort {
            return ConsistencyRecommendation::Abort;
        }
        if max_drift > self.config.kl_divergence_threshold * 5.0 {
            return ConsistencyRecommendation::Reaudit;
        }
        if num_drifted > 0 {
            return ConsistencyRecommendation::ReduceWindow;
        }
        ConsistencyRecommendation::Continue
    }
}

/// Classify the dominant drift type based on the two distributions and score.
fn classify_drift(
    baseline: &[(String, f64)],
    requery: &[(String, f64)],
    score: f64,
    threshold: f64,
) -> DriftType {
    let b_keys: Vec<&str> = baseline.iter().map(|(k, _)| k.as_str()).collect();
    let r_keys: Vec<&str> = requery.iter().map(|(k, _)| k.as_str()).collect();

    // Check for entirely new responses in the requery set.
    let has_new = r_keys.iter().any(|k| !b_keys.contains(k));
    // Check for absent responses (baseline response missing from requery).
    let has_absent = b_keys.iter().any(|k| !r_keys.contains(k));

    if has_new && !has_absent {
        return DriftType::NewResponse;
    }
    if has_absent && !has_new {
        return DriftType::Absence;
    }
    if score > threshold * 3.0 {
        return DriftType::Magnitude;
    }
    DriftType::Distribution
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- helper utilities ---------------------------------------------------

    fn dist(pairs: &[(&str, f64)]) -> Vec<(String, f64)> {
        pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
    }

    fn default_monitor() -> ConsistencyMonitor {
        ConsistencyMonitor::new(ConsistencyConfig::default())
    }

    // -- hash_string --------------------------------------------------------

    #[test]
    fn test_hash_string_deterministic() {
        let h1 = hash_string("hello");
        let h2 = hash_string("hello");
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_hash_string_different_inputs() {
        assert_ne!(hash_string("hello"), hash_string("world"));
    }

    // -- responses_to_distribution ------------------------------------------

    #[test]
    fn test_responses_to_distribution_empty() {
        let d = responses_to_distribution(&[]);
        assert!(d.is_empty());
    }

    #[test]
    fn test_responses_to_distribution_uniform() {
        let responses: Vec<String> = vec!["a", "b", "c"]
            .into_iter()
            .map(String::from)
            .collect();
        let d = responses_to_distribution(&responses);
        assert_eq!(d.len(), 3);
        for (_, p) in &d {
            assert!((p - 1.0 / 3.0).abs() < 1e-9);
        }
    }

    #[test]
    fn test_responses_to_distribution_skewed() {
        let responses: Vec<String> = vec!["a", "a", "a", "b"]
            .into_iter()
            .map(String::from)
            .collect();
        let d = responses_to_distribution(&responses);
        let a_prob = d.iter().find(|(k, _)| k == "a").unwrap().1;
        let b_prob = d.iter().find(|(k, _)| k == "b").unwrap().1;
        assert!((a_prob - 0.75).abs() < 1e-9);
        assert!((b_prob - 0.25).abs() < 1e-9);
    }

    // -- divergence measures ------------------------------------------------

    #[test]
    fn test_kl_divergence_identical() {
        let p = dist(&[("a", 0.5), ("b", 0.5)]);
        let kl = kl_divergence_discrete(&p, &p);
        assert!(kl < 1e-6, "KL of identical distributions should be ~0, got {}", kl);
    }

    #[test]
    fn test_kl_divergence_different() {
        let p = dist(&[("a", 0.9), ("b", 0.1)]);
        let q = dist(&[("a", 0.1), ("b", 0.9)]);
        let kl = kl_divergence_discrete(&p, &q);
        assert!(kl > 0.5, "KL should be large for very different dists, got {}", kl);
    }

    #[test]
    fn test_total_variation_identical() {
        let p = dist(&[("a", 0.5), ("b", 0.5)]);
        let tv = total_variation_discrete(&p, &p);
        assert!(tv < 1e-6, "TV of identical distributions should be ~0, got {}", tv);
    }

    #[test]
    fn test_total_variation_disjoint() {
        let p = dist(&[("a", 1.0)]);
        let q = dist(&[("b", 1.0)]);
        let tv = total_variation_discrete(&p, &q);
        // With smoothing TV won't be exactly 1.0, but should be close.
        assert!(tv > 0.9, "TV of disjoint distributions should be ~1, got {}", tv);
    }

    #[test]
    fn test_jensen_shannon_symmetry() {
        let p = dist(&[("a", 0.7), ("b", 0.3)]);
        let q = dist(&[("a", 0.3), ("b", 0.7)]);
        let jsd_pq = jensen_shannon_divergence(&p, &q);
        let jsd_qp = jensen_shannon_divergence(&q, &p);
        assert!(
            (jsd_pq - jsd_qp).abs() < 1e-12,
            "JSD should be symmetric: {} vs {}",
            jsd_pq,
            jsd_qp
        );
    }

    #[test]
    fn test_jensen_shannon_identical() {
        let p = dist(&[("a", 0.5), ("b", 0.5)]);
        let jsd = jensen_shannon_divergence(&p, &p);
        assert!(jsd < 1e-6, "JSD of identical dists should be ~0, got {}", jsd);
    }

    // -- ConsistencyMonitor -------------------------------------------------

    #[test]
    fn test_monitor_record_and_stats() {
        let mut m = default_monitor();
        m.record_baseline("q1", "yes");
        m.record_baseline("q1", "yes");
        m.record_baseline("q2", "no");
        assert_eq!(m.stats().total_baseline_queries, 2);

        m.record_requery("q1", "yes");
        assert_eq!(m.stats().total_requery_queries, 1);
    }

    #[test]
    fn test_monitor_no_drift_identical_responses() {
        let mut m = ConsistencyMonitor::new(ConsistencyConfig {
            min_samples_for_comparison: 1,
            ..Default::default()
        });
        for _ in 0..10 {
            m.record_baseline("q1", "answer_a");
        }
        for _ in 0..10 {
            m.record_requery("q1", "answer_a");
        }
        let result = m.check_consistency();
        assert!(result.overall_consistent);
        assert_eq!(result.num_drifted, 0);
        assert_eq!(result.recommendation, ConsistencyRecommendation::Continue);
    }

    #[test]
    fn test_monitor_drift_detected() {
        let mut m = ConsistencyMonitor::new(ConsistencyConfig {
            min_samples_for_comparison: 1,
            drift_threshold: 0.01, // very sensitive
            ..Default::default()
        });
        // Baseline: always "yes"
        for _ in 0..20 {
            m.record_baseline("q1", "yes");
        }
        // Requery: always "no" → massive drift
        for _ in 0..20 {
            m.record_requery("q1", "no");
        }
        let result = m.check_consistency();
        assert!(!result.overall_consistent);
        assert!(result.num_drifted > 0);
        assert!(result.max_drift_score > 0.01);
        assert!(!result.drifted_queries.is_empty());
    }

    #[test]
    fn test_monitor_select_requery_candidates() {
        let mut m = default_monitor();
        for i in 0..100 {
            m.record_baseline(&format!("q{}", i), "x");
        }
        let candidates = m.select_requery_candidates(100);
        // ~5% of 100 = 5
        assert_eq!(candidates.len(), 5);
        // Deterministic: same seed → same result.
        let candidates2 = m.select_requery_candidates(100);
        assert_eq!(candidates, candidates2);
    }

    #[test]
    fn test_monitor_validity_window_truncated_on_drift() {
        let mut m = ConsistencyMonitor::new(ConsistencyConfig {
            min_samples_for_comparison: 1,
            drift_threshold: 0.001,
            window_truncation_enabled: true,
            ..Default::default()
        });
        for _ in 0..10 {
            m.record_baseline("q1", "a");
        }
        for _ in 0..10 {
            m.record_requery("q1", "b");
        }
        let _ = m.check_consistency();
        let w = m.validity_window();
        assert!(w.truncated);
        assert!(w.end.is_some());
        assert!(w.reason.is_some());
    }

    #[test]
    fn test_monitor_reset() {
        let mut m = default_monitor();
        m.record_baseline("q1", "a");
        m.record_requery("q1", "b");
        m.reset();
        assert!(m.baseline_responses.is_empty());
        assert!(m.requery_responses.is_empty());
        assert!(m.drift_events.is_empty());
        assert_eq!(m.stats().total_baseline_queries, 0);
        let w = m.validity_window();
        assert!(!w.truncated);
    }

    #[test]
    fn test_monitor_is_drifting_by_event_count() {
        let mut m = ConsistencyMonitor::new(ConsistencyConfig {
            max_drift_events_before_abort: 3,
            ..Default::default()
        });
        assert!(!m.is_drifting());
        for i in 0..3 {
            m.drift_events.push(DriftEvent {
                query_key: format!("q{}", i),
                drift_score: 0.5,
                detected_at: now_iso(),
                drift_type: DriftType::Distribution,
            });
        }
        assert!(m.is_drifting());
    }

    #[test]
    fn test_monitor_recommendation_abort() {
        let mut m = ConsistencyMonitor::new(ConsistencyConfig {
            min_samples_for_comparison: 1,
            drift_threshold: 0.001,
            max_drift_events_before_abort: 2,
            ..Default::default()
        });
        // Pre-load one event so the next check pushes us to the limit.
        m.drift_events.push(DriftEvent {
            query_key: "pre".into(),
            drift_score: 0.5,
            detected_at: now_iso(),
            drift_type: DriftType::Distribution,
        });
        for _ in 0..10 {
            m.record_baseline("q1", "a");
        }
        for _ in 0..10 {
            m.record_requery("q1", "z");
        }
        let result = m.check_consistency();
        assert_eq!(result.recommendation, ConsistencyRecommendation::Abort);
    }

    #[test]
    fn test_compute_drift_score_no_data() {
        let m = default_monitor();
        assert_eq!(m.compute_drift_score("nonexistent"), 0.0);
    }

    #[test]
    fn test_classify_drift_new_response() {
        let baseline = dist(&[("a", 0.5), ("b", 0.5)]);
        let requery = dist(&[("a", 0.5), ("b", 0.3), ("c", 0.2)]);
        let dt = classify_drift(&baseline, &requery, 0.3, 0.1);
        assert_eq!(dt, DriftType::NewResponse);
    }

    #[test]
    fn test_classify_drift_absence() {
        let baseline = dist(&[("a", 0.5), ("b", 0.5)]);
        let requery = dist(&[("a", 1.0)]);
        let dt = classify_drift(&baseline, &requery, 0.3, 0.1);
        assert_eq!(dt, DriftType::Absence);
    }

    #[test]
    fn test_classify_drift_magnitude() {
        let baseline = dist(&[("a", 0.5), ("b", 0.5)]);
        let requery = dist(&[("a", 0.9), ("b", 0.1)]);
        // score = 0.35 > 0.1 * 3 = 0.3 → Magnitude
        let dt = classify_drift(&baseline, &requery, 0.35, 0.1);
        assert_eq!(dt, DriftType::Magnitude);
    }

    #[test]
    fn test_config_defaults() {
        let c = ConsistencyConfig::default();
        assert!((c.requery_rate - 0.05).abs() < 1e-9);
        assert!((c.drift_threshold - 0.1).abs() < 1e-9);
        assert_eq!(c.min_samples_for_comparison, 5);
        assert!((c.kl_divergence_threshold - 0.1).abs() < 1e-9);
        assert!((c.tv_distance_threshold - 0.15).abs() < 1e-9);
        assert!(!c.alert_on_any_drift);
        assert!(c.window_truncation_enabled);
        assert_eq!(c.max_drift_events_before_abort, 10);
    }

    #[test]
    fn test_alert_on_any_drift_mode() {
        let mut m = ConsistencyMonitor::new(ConsistencyConfig {
            min_samples_for_comparison: 1,
            drift_threshold: 0.001,
            alert_on_any_drift: true,
            ..Default::default()
        });
        for _ in 0..10 {
            m.record_baseline("q1", "a");
        }
        // Slightly different requery distribution.
        for _ in 0..9 {
            m.record_requery("q1", "a");
        }
        m.record_requery("q1", "b");
        let result = m.check_consistency();
        // In alert_on_any_drift mode, even tiny drift flags inconsistency.
        if result.num_drifted > 0 {
            assert!(!result.overall_consistent);
        }
    }

    #[test]
    fn test_end_to_end_workflow() {
        let mut m = ConsistencyMonitor::new(ConsistencyConfig {
            min_samples_for_comparison: 2,
            requery_rate: 0.5, // 50 % for easier testing
            drift_threshold: 0.05,
            ..Default::default()
        });

        // Phase 1: record baselines.
        for i in 0..10 {
            let key = format!("q{}", i);
            m.record_baseline(&key, "consistent_answer");
            m.record_baseline(&key, "consistent_answer");
        }

        // Phase 2: select candidates and record re-queries.
        let candidates = m.select_requery_candidates(10);
        assert!(!candidates.is_empty());

        for key in &candidates {
            m.record_requery(key, "consistent_answer");
            m.record_requery(key, "consistent_answer");
        }

        // Phase 3: check consistency – should be fine.
        let result = m.check_consistency();
        assert!(result.overall_consistent);
        assert_eq!(result.recommendation, ConsistencyRecommendation::Continue);

        // Phase 4: introduce drift on one query.
        m.record_requery("q0", "totally_different");
        m.record_requery("q0", "totally_different");
        m.record_requery("q0", "totally_different");
        m.record_requery("q0", "totally_different");

        let result2 = m.check_consistency();
        // q0 should now be flagged.
        let q0_drifted = result2.drifted_queries.iter().any(|dq| dq.query_key == "q0");
        assert!(q0_drifted, "q0 should be flagged as drifted");
    }
}
