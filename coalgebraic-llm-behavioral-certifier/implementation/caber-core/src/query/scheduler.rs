//! Query scheduling module for the CABER project.
//!
//! Provides priority-based query scheduling with deduplication, budget management,
//! adaptive concurrency control, exponential backoff, and comprehensive statistics.

use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::collections::{BinaryHeap, HashMap};
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};

// ---------------------------------------------------------------------------
// SchedulerError
// ---------------------------------------------------------------------------

/// Errors that can occur during query scheduling and execution.
#[derive(Debug, Clone)]
pub enum SchedulerError {
    RateLimited,
    Timeout,
    ServerError(String),
    BudgetExhausted,
    MaxRetriesExceeded,
}

impl std::fmt::Display for SchedulerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SchedulerError::RateLimited => write!(f, "Rate limited"),
            SchedulerError::Timeout => write!(f, "Request timed out"),
            SchedulerError::ServerError(msg) => write!(f, "Server error: {}", msg),
            SchedulerError::BudgetExhausted => write!(f, "Query budget exhausted"),
            SchedulerError::MaxRetriesExceeded => write!(f, "Maximum retries exceeded"),
        }
    }
}

impl std::error::Error for SchedulerError {}

// ---------------------------------------------------------------------------
// QueryPriority
// ---------------------------------------------------------------------------

/// Priority levels for scheduled queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QueryPriority {
    Background = 0,
    Low = 1,
    Normal = 2,
    High = 3,
    Critical = 4,
}

impl QueryPriority {
    fn numeric(&self) -> u8 {
        match self {
            QueryPriority::Background => 0,
            QueryPriority::Low => 1,
            QueryPriority::Normal => 2,
            QueryPriority::High => 3,
            QueryPriority::Critical => 4,
        }
    }
}

impl Ord for QueryPriority {
    fn cmp(&self, other: &Self) -> Ordering {
        self.numeric().cmp(&other.numeric())
    }
}

impl PartialOrd for QueryPriority {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// ---------------------------------------------------------------------------
// ScheduledQuery
// ---------------------------------------------------------------------------

/// A query that has been submitted to the scheduler.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduledQuery {
    pub id: String,
    pub content: String,
    pub query_type: String,
    pub priority: QueryPriority,
    pub attempt: usize,
    pub max_retries: usize,
    pub created_at: String,
    pub content_hash: u64,
}

impl ScheduledQuery {
    /// Create a new `ScheduledQuery` with computed content hash.
    pub fn new(
        id: String,
        content: String,
        query_type: String,
        priority: QueryPriority,
        max_retries: usize,
    ) -> Self {
        let content_hash = hash_query_content(&content);
        Self {
            id,
            content,
            query_type,
            priority,
            attempt: 0,
            max_retries,
            created_at: chrono::Utc::now().to_rfc3339(),
            content_hash,
        }
    }
}

// ---------------------------------------------------------------------------
// PrioritizedQuery  (wrapper for BinaryHeap ordering)
// ---------------------------------------------------------------------------

/// Wrapper around `ScheduledQuery` providing ordering for `BinaryHeap`.
///
/// Higher priority queries come first.  Among equal priorities the query that
/// was created earlier (smaller `sequence`) is dequeued first (FIFO within the
/// same priority band).
#[derive(Debug, Clone)]
pub struct PrioritizedQuery {
    pub query: ScheduledQuery,
    /// Monotonically increasing insertion counter – lower means earlier.
    pub sequence: u64,
}

impl PartialEq for PrioritizedQuery {
    fn eq(&self, other: &Self) -> bool {
        self.query.priority == other.query.priority && self.sequence == other.sequence
    }
}

impl Eq for PrioritizedQuery {}

impl Ord for PrioritizedQuery {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher priority first; ties broken by lower sequence (earlier insertion).
        self.query
            .priority
            .cmp(&other.query.priority)
            .then_with(|| other.sequence.cmp(&self.sequence))
    }
}

impl PartialOrd for PrioritizedQuery {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// ---------------------------------------------------------------------------
// SubmitResult
// ---------------------------------------------------------------------------

/// Result of submitting a query to the scheduler.
#[derive(Debug, Clone, PartialEq)]
pub enum SubmitResult {
    /// Query was added to the queue.
    Queued { position: usize },
    /// Query was deduplicated against an earlier submission.
    Deduplicated { original_id: usize },
    /// No more budget available.
    BudgetExhausted,
}

// ---------------------------------------------------------------------------
// CompletedQuery
// ---------------------------------------------------------------------------

/// Record of a completed (or failed) query execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletedQuery {
    pub query_id: String,
    pub success: bool,
    pub latency_ms: Option<f64>,
    pub error: Option<String>,
    pub completed_at: String,
}

// ---------------------------------------------------------------------------
// QueryBudget
// ---------------------------------------------------------------------------

/// Tracks the total number of queries allowed versus used.
#[derive(Debug, Clone)]
pub struct QueryBudget {
    total: usize,
    used: usize,
    reserved: usize,
}

impl QueryBudget {
    pub fn new(total: usize) -> Self {
        Self {
            total,
            used: 0,
            reserved: 0,
        }
    }

    /// Try to consume one query from the budget.  Returns `true` on success.
    pub fn try_consume(&mut self) -> bool {
        if self.used + self.reserved < self.total {
            self.used += 1;
            true
        } else {
            false
        }
    }

    /// Number of queries still available (accounting for reservations).
    pub fn remaining(&self) -> usize {
        self.total.saturating_sub(self.used + self.reserved)
    }

    /// Fraction of the budget that has been consumed (0.0 – 1.0).
    pub fn utilization(&self) -> f64 {
        if self.total == 0 {
            1.0
        } else {
            self.used as f64 / self.total as f64
        }
    }

    /// Returns `true` when no more queries can be dispatched.
    pub fn is_exhausted(&self) -> bool {
        self.remaining() == 0
    }
}

// ---------------------------------------------------------------------------
// BackoffState
// ---------------------------------------------------------------------------

/// Tracks exponential-backoff state for retries and rate-limit handling.
#[derive(Debug, Clone)]
pub struct BackoffState {
    initial_delay_ms: u64,
    current_delay_ms: u64,
    max_delay_ms: u64,
    consecutive_failures: usize,
    last_rate_limit: Option<String>,
    jitter_factor: f64,
}

impl BackoffState {
    pub fn new(initial_ms: u64, max_ms: u64) -> Self {
        Self {
            initial_delay_ms: initial_ms,
            current_delay_ms: initial_ms,
            max_delay_ms: max_ms,
            consecutive_failures: 0,
            last_rate_limit: None,
            jitter_factor: 0.1,
        }
    }

    /// Set the jitter factor used when computing the next delay.
    pub fn with_jitter(mut self, factor: f64) -> Self {
        self.jitter_factor = factor;
        self
    }

    /// Compute the next delay (in ms), applying exponential backoff with jitter.
    ///
    /// The formula is `min(current * 2, max) ± jitter`.  The internal
    /// `current_delay_ms` is updated so successive calls keep doubling.
    pub fn next_delay(&mut self) -> u64 {
        let base = self.current_delay_ms;

        // Double for next time, capped at max.
        self.current_delay_ms = (base.saturating_mul(2)).min(self.max_delay_ms);

        // Apply jitter: uniform in [base*(1-j), base*(1+j)].
        let jitter_range = (base as f64) * self.jitter_factor;
        // Deterministic-friendly: use a simple hash-based pseudo-random offset
        // so that tests remain reproducible while still spreading retries.
        let seed = self.consecutive_failures as u64 ^ base;
        let pseudo_random = ((seed.wrapping_mul(6364136223846793005).wrapping_add(1)) >> 33) as f64
            / (u32::MAX as f64);
        let jitter = jitter_range * (2.0 * pseudo_random - 1.0);

        let delay = (base as f64 + jitter).max(0.0) as u64;
        delay.min(self.max_delay_ms)
    }

    /// Reset backoff state after a successful operation.
    pub fn reset(&mut self) {
        self.current_delay_ms = self.initial_delay_ms;
        self.consecutive_failures = 0;
    }

    /// Record a failure, bumping the consecutive-failure counter.
    pub fn record_failure(&mut self) {
        self.consecutive_failures += 1;
    }

    /// Returns `true` if we are in a backoff period (i.e. there have been
    /// recent consecutive failures).
    pub fn is_backing_off(&self) -> bool {
        self.consecutive_failures > 0
    }
}

// ---------------------------------------------------------------------------
// SchedulerStats
// ---------------------------------------------------------------------------

/// Aggregate statistics for the scheduler.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerStats {
    pub total_submitted: usize,
    pub total_completed: usize,
    pub total_failed: usize,
    pub total_deduplicated: usize,
    pub total_retries: usize,
    pub total_rate_limits: usize,
    pub average_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub queries_per_second: f64,
    pub error_rate: f64,
    #[serde(skip)]
    latencies: Vec<f64>,
    #[serde(skip)]
    start_time: Option<String>,
}

impl SchedulerStats {
    pub fn new() -> Self {
        Self {
            total_submitted: 0,
            total_completed: 0,
            total_failed: 0,
            total_deduplicated: 0,
            total_retries: 0,
            total_rate_limits: 0,
            average_latency_ms: 0.0,
            p99_latency_ms: 0.0,
            queries_per_second: 0.0,
            error_rate: 0.0,
            latencies: Vec::new(),
            start_time: Some(chrono::Utc::now().to_rfc3339()),
        }
    }

    /// Record an observed latency value.
    pub fn record_latency(&mut self, ms: f64) {
        self.latencies.push(ms);
    }

    /// Recompute derived statistics from the raw latency samples.
    pub fn recompute(&mut self) {
        // Average latency
        if self.latencies.is_empty() {
            self.average_latency_ms = 0.0;
            self.p99_latency_ms = 0.0;
        } else {
            let sum: f64 = self.latencies.iter().sum();
            self.average_latency_ms = sum / self.latencies.len() as f64;

            // p99
            let mut sorted = self.latencies.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
            let idx = ((sorted.len() as f64) * 0.99).ceil() as usize;
            let idx = idx.min(sorted.len()).saturating_sub(1);
            self.p99_latency_ms = sorted[idx];
        }

        // Error rate
        let total_resolved = self.total_completed + self.total_failed;
        self.error_rate = if total_resolved == 0 {
            0.0
        } else {
            self.total_failed as f64 / total_resolved as f64
        };

        // QPS – based on elapsed wall-clock time since start.
        if let Some(ref start_str) = self.start_time {
            if let Ok(start) = chrono::DateTime::parse_from_rfc3339(start_str) {
                let elapsed = chrono::Utc::now()
                    .signed_duration_since(start)
                    .num_milliseconds()
                    .max(1) as f64
                    / 1000.0;
                self.queries_per_second = self.total_completed as f64 / elapsed;
            }
        }
    }

    /// Return a human-readable summary of the stats.
    pub fn summary(&self) -> String {
        format!(
            "Scheduler Stats: submitted={}, completed={}, failed={}, dedup={}, \
             retries={}, rate_limits={}, avg_lat={:.2}ms, p99_lat={:.2}ms, \
             qps={:.2}, err_rate={:.4}",
            self.total_submitted,
            self.total_completed,
            self.total_failed,
            self.total_deduplicated,
            self.total_retries,
            self.total_rate_limits,
            self.average_latency_ms,
            self.p99_latency_ms,
            self.queries_per_second,
            self.error_rate,
        )
    }
}

impl Default for SchedulerStats {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// SchedulerConfig
// ---------------------------------------------------------------------------

/// Configuration for the `QueryScheduler`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    /// Maximum number of concurrent in-flight queries.
    pub max_concurrent: usize,
    /// Maximum number of retries per query.
    pub max_retries: usize,
    /// Initial backoff delay in milliseconds.
    pub initial_backoff_ms: u64,
    /// Maximum backoff delay in milliseconds.
    pub max_backoff_ms: u64,
    /// Jitter factor applied to backoff delays (0.0 – 1.0).
    pub jitter_factor: f64,
    /// Total number of queries allowed to be dispatched.
    pub query_budget: usize,
    /// Whether content-based deduplication is enabled.
    pub dedup_enabled: bool,
    /// Whether adaptive concurrency adjustment is enabled.
    pub adaptive_concurrency: bool,
    /// Error-rate threshold above which concurrency is reduced.
    pub error_rate_threshold: f64,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_concurrent: 10,
            max_retries: 3,
            initial_backoff_ms: 1000,
            max_backoff_ms: 60_000,
            jitter_factor: 0.1,
            query_budget: 1000,
            dedup_enabled: true,
            adaptive_concurrency: true,
            error_rate_threshold: 0.1,
        }
    }
}

// ---------------------------------------------------------------------------
// hash_query_content
// ---------------------------------------------------------------------------

/// Compute a 64-bit hash of query content for deduplication.
pub fn hash_query_content(content: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    content.hash(&mut hasher);
    hasher.finish()
}

// ---------------------------------------------------------------------------
// QueryScheduler
// ---------------------------------------------------------------------------

/// Priority-aware query scheduler with deduplication, budget management,
/// adaptive concurrency, and exponential backoff.
pub struct QueryScheduler {
    config: SchedulerConfig,
    query_queue: BinaryHeap<PrioritizedQuery>,
    completed: Vec<CompletedQuery>,
    budget: QueryBudget,
    stats: SchedulerStats,
    dedup_cache: HashMap<u64, usize>,
    backoff_state: BackoffState,
    /// Monotonically increasing counter to break priority ties (FIFO).
    sequence_counter: u64,
    /// Current effective concurrency (may differ from config when adaptive).
    effective_concurrency: usize,
    /// Map of query-id to attempt count, for retry tracking.
    retry_tracker: HashMap<String, usize>,
    /// Map of query-id to max_retries for that query.
    retry_limits: HashMap<String, usize>,
}

impl QueryScheduler {
    // ------------------------------------------------------------------
    // Construction
    // ------------------------------------------------------------------

    /// Create a new `QueryScheduler` from the given configuration.
    pub fn new(config: SchedulerConfig) -> Self {
        let budget = QueryBudget::new(config.query_budget);
        let backoff = BackoffState::new(config.initial_backoff_ms, config.max_backoff_ms)
            .with_jitter(config.jitter_factor);
        let effective_concurrency = config.max_concurrent;
        Self {
            config,
            query_queue: BinaryHeap::new(),
            completed: Vec::new(),
            budget,
            stats: SchedulerStats::new(),
            dedup_cache: HashMap::new(),
            backoff_state: backoff,
            sequence_counter: 0,
            effective_concurrency,
            retry_tracker: HashMap::new(),
            retry_limits: HashMap::new(),
        }
    }

    // ------------------------------------------------------------------
    // Submission
    // ------------------------------------------------------------------

    /// Submit a query to the scheduler.
    ///
    /// If deduplication is enabled and an identical query (by content hash) has
    /// already been submitted, returns `SubmitResult::Deduplicated`.
    /// If the budget is exhausted, returns `SubmitResult::BudgetExhausted`.
    /// Otherwise the query is enqueued and `SubmitResult::Queued` is returned.
    pub fn submit(&mut self, query: ScheduledQuery, priority: QueryPriority) -> SubmitResult {
        // Budget check
        if self.budget.is_exhausted() {
            return SubmitResult::BudgetExhausted;
        }

        // Dedup check
        if self.config.dedup_enabled {
            if let Some(&original_idx) = self.dedup_cache.get(&query.content_hash) {
                self.stats.total_deduplicated += 1;
                return SubmitResult::Deduplicated {
                    original_id: original_idx,
                };
            }
        }

        // Consume budget
        if !self.budget.try_consume() {
            return SubmitResult::BudgetExhausted;
        }

        // Record in dedup cache
        let position = self.stats.total_submitted;
        if self.config.dedup_enabled {
            self.dedup_cache.insert(query.content_hash, position);
        }

        // Track retry limits
        self.retry_limits
            .insert(query.id.clone(), query.max_retries);

        // Enqueue
        let mut q = query;
        q.priority = priority;
        let seq = self.sequence_counter;
        self.sequence_counter += 1;
        self.query_queue.push(PrioritizedQuery {
            query: q,
            sequence: seq,
        });

        self.stats.total_submitted += 1;

        SubmitResult::Queued { position }
    }

    // ------------------------------------------------------------------
    // Batch retrieval
    // ------------------------------------------------------------------

    /// Pull up to `max_batch_size` queries from the priority queue.
    ///
    /// Queries are returned highest-priority first.  The effective batch size
    /// is also bounded by the current concurrency limit.
    pub fn next_batch(&mut self, max_batch_size: usize) -> Vec<ScheduledQuery> {
        let limit = max_batch_size.min(self.effective_concurrency);
        let mut batch = Vec::with_capacity(limit);
        for _ in 0..limit {
            if let Some(pq) = self.query_queue.pop() {
                batch.push(pq.query);
            } else {
                break;
            }
        }
        batch
    }

    // ------------------------------------------------------------------
    // Result recording
    // ------------------------------------------------------------------

    /// Record the successful completion of a query.
    pub fn record_success(&mut self, query_id: &str, latency_ms: f64) {
        self.stats.total_completed += 1;
        self.stats.record_latency(latency_ms);
        self.stats.recompute();

        self.backoff_state.reset();

        self.completed.push(CompletedQuery {
            query_id: query_id.to_string(),
            success: true,
            latency_ms: Some(latency_ms),
            error: None,
            completed_at: chrono::Utc::now().to_rfc3339(),
        });

        // Clear retry tracker for this query on success
        self.retry_tracker.remove(query_id);

        if self.config.adaptive_concurrency {
            self.adjust_concurrency();
        }
    }

    /// Record a query failure and update backoff state.
    pub fn record_failure(&mut self, query_id: &str, error: SchedulerError) {
        self.stats.total_failed += 1;
        self.backoff_state.record_failure();
        let _ = self.backoff_state.next_delay();
        self.stats.recompute();

        // Increment retry tracker
        let attempts = self.retry_tracker.entry(query_id.to_string()).or_insert(0);
        *attempts += 1;
        self.stats.total_retries += 1;

        self.completed.push(CompletedQuery {
            query_id: query_id.to_string(),
            success: false,
            latency_ms: None,
            error: Some(error.to_string()),
            completed_at: chrono::Utc::now().to_rfc3339(),
        });

        if self.config.adaptive_concurrency {
            self.adjust_concurrency();
        }
    }

    /// Record a rate-limit response and enter backoff.
    pub fn record_rate_limit(&mut self, retry_after_ms: u64) {
        self.stats.total_rate_limits += 1;
        self.backoff_state.consecutive_failures += 1;
        self.backoff_state.current_delay_ms = retry_after_ms;
        self.backoff_state.last_rate_limit = Some(chrono::Utc::now().to_rfc3339());
        self.stats.recompute();

        if self.config.adaptive_concurrency {
            self.adjust_concurrency();
        }
    }

    // ------------------------------------------------------------------
    // Retry policy
    // ------------------------------------------------------------------

    /// Check whether a query should be retried based on its attempt count and
    /// the configured (or per-query) retry limit.
    pub fn should_retry(&self, query_id: &str) -> bool {
        let attempts = self.retry_tracker.get(query_id).copied().unwrap_or(0);
        let max = self
            .retry_limits
            .get(query_id)
            .copied()
            .unwrap_or(self.config.max_retries);
        attempts < max
    }

    // ------------------------------------------------------------------
    // Budget helpers
    // ------------------------------------------------------------------

    /// Number of queries that can still be dispatched.
    pub fn budget_remaining(&self) -> usize {
        self.budget.remaining()
    }

    /// Fraction of the budget consumed so far (0.0 – 1.0).
    pub fn budget_utilization(&self) -> f64 {
        self.budget.utilization()
    }

    /// Returns `true` when the query budget is fully consumed.
    pub fn is_budget_exhausted(&self) -> bool {
        self.budget.is_exhausted()
    }

    // ------------------------------------------------------------------
    // Stats / counts
    // ------------------------------------------------------------------

    /// Borrow the current scheduler statistics.
    pub fn stats(&self) -> &SchedulerStats {
        &self.stats
    }

    /// Number of queries waiting in the queue.
    pub fn pending_count(&self) -> usize {
        self.query_queue.len()
    }

    /// Number of queries that have been completed (success + failure).
    pub fn completed_count(&self) -> usize {
        self.completed.len()
    }

    // ------------------------------------------------------------------
    // Adaptive concurrency
    // ------------------------------------------------------------------

    /// Adjust the effective concurrency based on the observed error rate.
    ///
    /// If the error rate exceeds the configured threshold the concurrency is
    /// halved (minimum 1).  If the error rate is well below the threshold and
    /// we are currently below the configured maximum, concurrency is increased
    /// by 1.
    pub fn adjust_concurrency(&mut self) {
        if !self.config.adaptive_concurrency {
            return;
        }

        let total_resolved = self.stats.total_completed + self.stats.total_failed;
        if total_resolved < 5 {
            // Not enough data to make a decision.
            return;
        }

        let error_rate = self.stats.total_failed as f64 / total_resolved as f64;

        if error_rate > self.config.error_rate_threshold {
            // Too many errors – reduce concurrency.
            self.effective_concurrency = (self.effective_concurrency / 2).max(1);
        } else if error_rate < self.config.error_rate_threshold / 2.0 {
            // Error rate is low – try increasing concurrency.
            if self.effective_concurrency < self.config.max_concurrent {
                self.effective_concurrency += 1;
            }
        }
    }

    /// Return the current effective concurrency level.
    pub fn current_concurrency(&self) -> usize {
        self.effective_concurrency
    }

    // ------------------------------------------------------------------
    // Drain
    // ------------------------------------------------------------------

    /// Remove and return all remaining queries from the queue.
    pub fn drain_queue(&mut self) -> Vec<ScheduledQuery> {
        let mut all = Vec::with_capacity(self.query_queue.len());
        while let Some(pq) = self.query_queue.pop() {
            all.push(pq.query);
        }
        all
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- helpers ----

    fn default_config() -> SchedulerConfig {
        SchedulerConfig::default()
    }

    fn small_budget_config(budget: usize) -> SchedulerConfig {
        SchedulerConfig {
            query_budget: budget,
            ..Default::default()
        }
    }

    fn make_query(id: &str, content: &str) -> ScheduledQuery {
        ScheduledQuery::new(
            id.to_string(),
            content.to_string(),
            "test".to_string(),
            QueryPriority::Normal,
            3,
        )
    }

    fn make_query_with_priority(id: &str, content: &str, priority: QueryPriority) -> ScheduledQuery {
        ScheduledQuery::new(
            id.to_string(),
            content.to_string(),
            "test".to_string(),
            priority,
            3,
        )
    }

    // ---- tests ----

    #[test]
    fn test_submit_and_pending_count() {
        let mut sched = QueryScheduler::new(default_config());
        let q = make_query("q1", "What is 2+2?");
        let result = sched.submit(q, QueryPriority::Normal);
        assert!(matches!(result, SubmitResult::Queued { position: 0 }));
        assert_eq!(sched.pending_count(), 1);
    }

    #[test]
    fn test_dedup_identical_content() {
        let mut sched = QueryScheduler::new(default_config());
        let q1 = make_query("q1", "duplicate content");
        let q2 = make_query("q2", "duplicate content");
        let r1 = sched.submit(q1, QueryPriority::Normal);
        let r2 = sched.submit(q2, QueryPriority::Normal);
        assert!(matches!(r1, SubmitResult::Queued { .. }));
        assert!(matches!(r2, SubmitResult::Deduplicated { .. }));
        assert_eq!(sched.stats().total_deduplicated, 1);
        // Only one should be in the queue
        assert_eq!(sched.pending_count(), 1);
    }

    #[test]
    fn test_dedup_disabled() {
        let cfg = SchedulerConfig {
            dedup_enabled: false,
            ..Default::default()
        };
        let mut sched = QueryScheduler::new(cfg);
        let q1 = make_query("q1", "same thing");
        let q2 = make_query("q2", "same thing");
        let r1 = sched.submit(q1, QueryPriority::Normal);
        let r2 = sched.submit(q2, QueryPriority::Normal);
        assert!(matches!(r1, SubmitResult::Queued { .. }));
        assert!(matches!(r2, SubmitResult::Queued { .. }));
        assert_eq!(sched.pending_count(), 2);
    }

    #[test]
    fn test_budget_exhausted() {
        let mut sched = QueryScheduler::new(small_budget_config(2));
        let r1 = sched.submit(make_query("q1", "a"), QueryPriority::Normal);
        let r2 = sched.submit(make_query("q2", "b"), QueryPriority::Normal);
        let r3 = sched.submit(make_query("q3", "c"), QueryPriority::Normal);
        assert!(matches!(r1, SubmitResult::Queued { .. }));
        assert!(matches!(r2, SubmitResult::Queued { .. }));
        assert_eq!(r3, SubmitResult::BudgetExhausted);
        assert!(sched.is_budget_exhausted());
    }

    #[test]
    fn test_budget_utilization() {
        let mut sched = QueryScheduler::new(small_budget_config(4));
        sched.submit(make_query("q1", "x"), QueryPriority::Normal);
        sched.submit(make_query("q2", "y"), QueryPriority::Normal);
        assert!((sched.budget_utilization() - 0.5).abs() < 1e-9);
        assert_eq!(sched.budget_remaining(), 2);
    }

    #[test]
    fn test_next_batch_respects_size() {
        let mut sched = QueryScheduler::new(default_config());
        for i in 0..5 {
            sched.submit(
                make_query(&format!("q{}", i), &format!("content {}", i)),
                QueryPriority::Normal,
            );
        }
        let batch = sched.next_batch(3);
        assert_eq!(batch.len(), 3);
        assert_eq!(sched.pending_count(), 2);
    }

    #[test]
    fn test_next_batch_returns_fewer_if_queue_small() {
        let mut sched = QueryScheduler::new(default_config());
        sched.submit(make_query("q1", "only one"), QueryPriority::Normal);
        let batch = sched.next_batch(10);
        assert_eq!(batch.len(), 1);
        assert_eq!(sched.pending_count(), 0);
    }

    #[test]
    fn test_priority_ordering() {
        let mut sched = QueryScheduler::new(default_config());
        sched.submit(make_query("low", "lo"), QueryPriority::Low);
        sched.submit(make_query("crit", "cr"), QueryPriority::Critical);
        sched.submit(make_query("norm", "no"), QueryPriority::Normal);

        let batch = sched.next_batch(3);
        assert_eq!(batch[0].id, "crit");
        assert_eq!(batch[1].id, "norm");
        assert_eq!(batch[2].id, "low");
    }

    #[test]
    fn test_fifo_within_same_priority() {
        let mut sched = QueryScheduler::new(default_config());
        sched.submit(make_query("first", "aaa"), QueryPriority::High);
        sched.submit(make_query("second", "bbb"), QueryPriority::High);
        sched.submit(make_query("third", "ccc"), QueryPriority::High);

        let batch = sched.next_batch(3);
        assert_eq!(batch[0].id, "first");
        assert_eq!(batch[1].id, "second");
        assert_eq!(batch[2].id, "third");
    }

    #[test]
    fn test_record_success_updates_stats() {
        let mut sched = QueryScheduler::new(default_config());
        sched.submit(make_query("q1", "a"), QueryPriority::Normal);
        sched.record_success("q1", 42.0);
        assert_eq!(sched.stats().total_completed, 1);
        assert!((sched.stats().average_latency_ms - 42.0).abs() < 1e-9);
        assert_eq!(sched.completed_count(), 1);
    }

    #[test]
    fn test_record_failure_updates_stats() {
        let mut sched = QueryScheduler::new(default_config());
        sched.submit(make_query("q1", "a"), QueryPriority::Normal);
        sched.record_failure("q1", SchedulerError::Timeout);
        assert_eq!(sched.stats().total_failed, 1);
        assert_eq!(sched.completed_count(), 1);
    }

    #[test]
    fn test_should_retry() {
        let mut sched = QueryScheduler::new(default_config());
        sched.submit(make_query("q1", "a"), QueryPriority::Normal);
        // Before any failure
        assert!(sched.should_retry("q1"));
        // After 3 failures (max_retries = 3)
        sched.record_failure("q1", SchedulerError::Timeout);
        sched.record_failure("q1", SchedulerError::Timeout);
        sched.record_failure("q1", SchedulerError::Timeout);
        assert!(!sched.should_retry("q1"));
    }

    #[test]
    fn test_backoff_state() {
        let mut bs = BackoffState::new(100, 10_000);
        assert!(!bs.is_backing_off());
        bs.record_failure();
        assert!(bs.is_backing_off());
        let d1 = bs.next_delay();
        assert!(d1 >= 90 && d1 <= 110); // ~100 ± jitter
        let d2 = bs.next_delay();
        assert!(d2 >= 150); // should have doubled to ~200
        bs.reset();
        assert!(!bs.is_backing_off());
    }

    #[test]
    fn test_backoff_caps_at_max() {
        let mut bs = BackoffState::new(1000, 5000).with_jitter(0.0);
        // Force many doublings
        for _ in 0..20 {
            bs.record_failure();
            let d = bs.next_delay();
            assert!(d <= 5000);
        }
    }

    #[test]
    fn test_rate_limit_recording() {
        let mut sched = QueryScheduler::new(default_config());
        sched.record_rate_limit(5000);
        assert_eq!(sched.stats().total_rate_limits, 1);
        assert!(sched.backoff_state.is_backing_off());
    }

    #[test]
    fn test_adaptive_concurrency_reduces_on_errors() {
        let cfg = SchedulerConfig {
            max_concurrent: 10,
            adaptive_concurrency: true,
            error_rate_threshold: 0.1,
            query_budget: 100,
            ..Default::default()
        };
        let mut sched = QueryScheduler::new(cfg);

        // Submit enough queries
        for i in 0..20 {
            sched.submit(
                make_query(&format!("q{}", i), &format!("c{}", i)),
                QueryPriority::Normal,
            );
        }

        // Record 1 success and 9 failures -> 90% error rate
        sched.record_success("q0", 10.0);
        for i in 1..10 {
            sched.record_failure(
                &format!("q{}", i),
                SchedulerError::ServerError("fail".into()),
            );
        }

        assert!(sched.current_concurrency() < 10);
    }

    #[test]
    fn test_adaptive_concurrency_increases_on_low_errors() {
        let cfg = SchedulerConfig {
            max_concurrent: 20,
            adaptive_concurrency: true,
            error_rate_threshold: 0.5,
            query_budget: 100,
            ..Default::default()
        };
        let mut sched = QueryScheduler::new(cfg);

        // Artificially lower effective concurrency
        sched.effective_concurrency = 5;

        for i in 0..20 {
            sched.submit(
                make_query(&format!("q{}", i), &format!("c{}", i)),
                QueryPriority::Normal,
            );
        }

        // Record 10 successes, 0 failures -> 0% error rate
        for i in 0..10 {
            sched.record_success(&format!("q{}", i), 10.0);
        }

        // Concurrency should have increased from 5
        assert!(sched.current_concurrency() > 5);
    }

    #[test]
    fn test_drain_queue() {
        let mut sched = QueryScheduler::new(default_config());
        for i in 0..4 {
            sched.submit(
                make_query(&format!("q{}", i), &format!("d{}", i)),
                QueryPriority::Normal,
            );
        }
        let drained = sched.drain_queue();
        assert_eq!(drained.len(), 4);
        assert_eq!(sched.pending_count(), 0);
    }

    #[test]
    fn test_hash_deterministic() {
        let h1 = hash_query_content("hello world");
        let h2 = hash_query_content("hello world");
        let h3 = hash_query_content("different");
        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_stats_summary_not_empty() {
        let mut stats = SchedulerStats::new();
        stats.total_submitted = 5;
        stats.total_completed = 3;
        stats.total_failed = 1;
        stats.record_latency(10.0);
        stats.record_latency(20.0);
        stats.recompute();
        let s = stats.summary();
        assert!(s.contains("submitted=5"));
        assert!(s.contains("completed=3"));
        assert!(s.contains("failed=1"));
    }

    #[test]
    fn test_stats_p99_single_value() {
        let mut stats = SchedulerStats::new();
        stats.record_latency(42.0);
        stats.recompute();
        assert!((stats.p99_latency_ms - 42.0).abs() < 1e-9);
    }

    #[test]
    fn test_query_budget_standalone() {
        let mut budget = QueryBudget::new(3);
        assert_eq!(budget.remaining(), 3);
        assert!(!budget.is_exhausted());
        assert!(budget.try_consume());
        assert!(budget.try_consume());
        assert!(budget.try_consume());
        assert!(!budget.try_consume());
        assert!(budget.is_exhausted());
        assert!((budget.utilization() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_config_serialization() {
        let cfg = SchedulerConfig::default();
        let json = serde_json::to_string(&cfg).expect("serialize");
        let cfg2: SchedulerConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(cfg2.max_concurrent, 10);
        assert_eq!(cfg2.max_retries, 3);
        assert!((cfg2.jitter_factor - 0.1).abs() < 1e-9);
    }

    #[test]
    fn test_scheduled_query_serialization() {
        let q = make_query("sq1", "serialize me");
        let json = serde_json::to_string(&q).expect("serialize");
        let q2: ScheduledQuery = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(q2.id, "sq1");
        assert_eq!(q2.content, "serialize me");
        assert_eq!(q2.content_hash, hash_query_content("serialize me"));
    }

    #[test]
    fn test_scheduler_error_display() {
        let e = SchedulerError::ServerError("oops".into());
        assert_eq!(format!("{}", e), "Server error: oops");
        let e2 = SchedulerError::RateLimited;
        assert_eq!(format!("{}", e2), "Rate limited");
    }

    #[test]
    fn test_next_batch_respects_concurrency_limit() {
        let cfg = SchedulerConfig {
            max_concurrent: 2,
            query_budget: 100,
            ..Default::default()
        };
        let mut sched = QueryScheduler::new(cfg);
        for i in 0..10 {
            sched.submit(
                make_query(&format!("q{}", i), &format!("content{}", i)),
                QueryPriority::Normal,
            );
        }
        // Ask for 10 but concurrency is 2
        let batch = sched.next_batch(10);
        assert_eq!(batch.len(), 2);
    }
}
