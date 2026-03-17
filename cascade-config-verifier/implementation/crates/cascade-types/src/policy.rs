use serde::{Deserialize, Serialize};

use std::fmt;

// ---------------------------------------------------------------------------
// RetryCondition
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RetryCondition {
    #[serde(rename = "5xx")]
    ServerError,
    GatewayError,
    Reset,
    ConnectFailure,
    #[serde(rename = "retriable-4xx")]
    Retriable4xx,
    EnvoyRatelimited,
    RefusedStream,
    Cancelled,
    DeadlineExceeded,
    ResourceExhausted,
    Unavailable,
}

impl RetryCondition {
    pub fn is_always_safe(self) -> bool {
        matches!(
            self,
            RetryCondition::ConnectFailure | RetryCondition::Reset | RetryCondition::RefusedStream
        )
    }

    pub fn is_status_based(self) -> bool {
        matches!(
            self,
            RetryCondition::ServerError
                | RetryCondition::GatewayError
                | RetryCondition::Retriable4xx
                | RetryCondition::EnvoyRatelimited
        )
    }
}

impl fmt::Display for RetryCondition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RetryCondition::ServerError => write!(f, "5xx"),
            RetryCondition::GatewayError => write!(f, "gateway-error"),
            RetryCondition::Reset => write!(f, "reset"),
            RetryCondition::ConnectFailure => write!(f, "connect-failure"),
            RetryCondition::Retriable4xx => write!(f, "retriable-4xx"),
            RetryCondition::EnvoyRatelimited => write!(f, "envoy-ratelimited"),
            RetryCondition::RefusedStream => write!(f, "refused-stream"),
            RetryCondition::Cancelled => write!(f, "cancelled"),
            RetryCondition::DeadlineExceeded => write!(f, "deadline-exceeded"),
            RetryCondition::ResourceExhausted => write!(f, "resource-exhausted"),
            RetryCondition::Unavailable => write!(f, "unavailable"),
        }
    }
}

// ---------------------------------------------------------------------------
// BackoffStrategy
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BackoffStrategy {
    Fixed {
        delay_ms: u64,
    },
    Exponential {
        base_ms: u64,
        max_ms: u64,
    },
    Linear {
        increment_ms: u64,
    },
}

impl BackoffStrategy {
    /// Compute the delay for the `attempt`-th retry (0-based).
    pub fn delay_for_attempt(&self, attempt: u32) -> u64 {
        match self {
            BackoffStrategy::Fixed { delay_ms } => *delay_ms,
            BackoffStrategy::Exponential { base_ms, max_ms } => {
                let exp = base_ms.saturating_mul(1u64.checked_shl(attempt).unwrap_or(u64::MAX));
                exp.min(*max_ms)
            }
            BackoffStrategy::Linear { increment_ms } => {
                increment_ms.saturating_mul(attempt as u64 + 1)
            }
        }
    }

    /// Total delay across all retries (0..max_retries-1).
    pub fn total_delay(&self, max_retries: u32) -> u64 {
        (0..max_retries).map(|i| self.delay_for_attempt(i)).sum()
    }
}

impl Default for BackoffStrategy {
    fn default() -> Self {
        BackoffStrategy::Exponential {
            base_ms: 25,
            max_ms: 1000,
        }
    }
}

impl fmt::Display for BackoffStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BackoffStrategy::Fixed { delay_ms } => write!(f, "fixed({delay_ms}ms)"),
            BackoffStrategy::Exponential { base_ms, max_ms } => {
                write!(f, "exponential(base={base_ms}ms, max={max_ms}ms)")
            }
            BackoffStrategy::Linear { increment_ms } => {
                write!(f, "linear(+{increment_ms}ms)")
            }
        }
    }
}

// ---------------------------------------------------------------------------
// RetryPolicy
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RetryPolicy {
    pub max_retries: u32,
    pub per_try_timeout_ms: u64,
    pub retry_on: Vec<RetryCondition>,
    pub retry_budget: Option<f64>,
    pub backoff: BackoffStrategy,
}

impl RetryPolicy {
    pub fn new(max_retries: u32) -> Self {
        Self {
            max_retries,
            per_try_timeout_ms: 2000,
            retry_on: vec![RetryCondition::ServerError, RetryCondition::ConnectFailure],
            retry_budget: None,
            backoff: BackoffStrategy::default(),
        }
    }

    pub fn with_per_try_timeout(mut self, ms: u64) -> Self {
        self.per_try_timeout_ms = ms;
        self
    }

    pub fn with_retry_on(mut self, conditions: Vec<RetryCondition>) -> Self {
        self.retry_on = conditions;
        self
    }

    pub fn with_retry_budget(mut self, budget: f64) -> Self {
        self.retry_budget = Some(budget);
        self
    }

    pub fn with_backoff(mut self, backoff: BackoffStrategy) -> Self {
        self.backoff = backoff;
        self
    }

    /// Total worst-case time for all retries including backoff.
    pub fn worst_case_latency_ms(&self) -> u64 {
        let attempts = self.max_retries as u64 + 1; // original + retries
        let attempt_time = attempts * self.per_try_timeout_ms;
        let backoff_time = self.backoff.total_delay(self.max_retries);
        attempt_time + backoff_time
    }

    /// Amplification factor = 1 + max_retries.
    pub fn amplification_factor(&self) -> u32 {
        1 + self.max_retries
    }

    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if self.max_retries > 25 {
            errors.push(format!(
                "max_retries {} is very high; consider ≤ 5",
                self.max_retries
            ));
        }
        if self.per_try_timeout_ms == 0 {
            errors.push("per_try_timeout_ms must be > 0".into());
        }
        if let Some(budget) = self.retry_budget {
            if !(0.0..=1.0).contains(&budget) {
                errors.push(format!("retry_budget {budget} must be in [0, 1]"));
            }
        }
        if self.retry_on.is_empty() {
            errors.push("retry_on conditions list is empty".into());
        }
        errors
    }

    pub fn is_noop(&self) -> bool {
        self.max_retries == 0
    }
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self::new(3)
    }
}

impl fmt::Display for RetryPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RetryPolicy(retries={}, per_try={}ms, backoff={})",
            self.max_retries, self.per_try_timeout_ms, self.backoff
        )
    }
}

// ---------------------------------------------------------------------------
// TimeoutPolicy
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimeoutPolicy {
    pub request_timeout_ms: u64,
    pub idle_timeout_ms: Option<u64>,
    pub per_try_timeout_ms: Option<u64>,
    pub upstream_deadline_ms: Option<u64>,
}

impl TimeoutPolicy {
    pub fn new(request_timeout_ms: u64) -> Self {
        Self {
            request_timeout_ms,
            idle_timeout_ms: None,
            per_try_timeout_ms: None,
            upstream_deadline_ms: None,
        }
    }

    pub fn with_idle_timeout(mut self, ms: u64) -> Self {
        self.idle_timeout_ms = Some(ms);
        self
    }

    pub fn with_per_try_timeout(mut self, ms: u64) -> Self {
        self.per_try_timeout_ms = Some(ms);
        self
    }

    pub fn with_upstream_deadline(mut self, ms: u64) -> Self {
        self.upstream_deadline_ms = Some(ms);
        self
    }

    /// Check that per-try timeout ≤ request timeout.
    pub fn is_consistent(&self) -> bool {
        if let Some(pt) = self.per_try_timeout_ms {
            if pt > self.request_timeout_ms {
                return false;
            }
        }
        true
    }

    /// Effective per-try timeout: explicit or falls back to request_timeout.
    pub fn effective_per_try_timeout_ms(&self) -> u64 {
        self.per_try_timeout_ms.unwrap_or(self.request_timeout_ms)
    }

    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if self.request_timeout_ms == 0 {
            errors.push("request_timeout_ms must be > 0".into());
        }
        if !self.is_consistent() {
            errors.push(format!(
                "per_try_timeout ({}ms) exceeds request_timeout ({}ms)",
                self.per_try_timeout_ms.unwrap_or(0),
                self.request_timeout_ms
            ));
        }
        if let Some(idle) = self.idle_timeout_ms {
            if idle == 0 {
                errors.push("idle_timeout_ms must be > 0".into());
            }
        }
        errors
    }
}

impl Default for TimeoutPolicy {
    fn default() -> Self {
        Self::new(30_000)
    }
}

impl fmt::Display for TimeoutPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TimeoutPolicy(request={}ms", self.request_timeout_ms)?;
        if let Some(pt) = self.per_try_timeout_ms {
            write!(f, ", per_try={pt}ms")?;
        }
        if let Some(idle) = self.idle_timeout_ms {
            write!(f, ", idle={idle}ms")?;
        }
        write!(f, ")")
    }
}

// ---------------------------------------------------------------------------
// CircuitBreakerPolicy
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CircuitBreakerPolicy {
    pub max_connections: u32,
    pub max_pending: u32,
    pub max_requests: u32,
    pub max_retries: u32,
    pub consecutive_errors: u32,
    pub interval_ms: u64,
    pub base_ejection_time_ms: u64,
    pub max_ejection_percent: f64,
}

impl CircuitBreakerPolicy {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_max_connections(mut self, n: u32) -> Self {
        self.max_connections = n;
        self
    }

    pub fn with_max_pending(mut self, n: u32) -> Self {
        self.max_pending = n;
        self
    }

    pub fn with_max_requests(mut self, n: u32) -> Self {
        self.max_requests = n;
        self
    }

    pub fn with_max_retries(mut self, n: u32) -> Self {
        self.max_retries = n;
        self
    }

    pub fn with_consecutive_errors(mut self, n: u32) -> Self {
        self.consecutive_errors = n;
        self
    }

    pub fn with_interval(mut self, ms: u64) -> Self {
        self.interval_ms = ms;
        self
    }

    pub fn with_base_ejection_time(mut self, ms: u64) -> Self {
        self.base_ejection_time_ms = ms;
        self
    }

    pub fn with_max_ejection_percent(mut self, pct: f64) -> Self {
        self.max_ejection_percent = pct;
        self
    }

    /// Whether this circuit breaker is strict (low thresholds).
    pub fn is_strict(&self) -> bool {
        self.consecutive_errors <= 3 && self.max_ejection_percent >= 50.0
    }

    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if self.max_connections == 0 {
            errors.push("max_connections must be > 0".into());
        }
        if self.max_ejection_percent < 0.0 || self.max_ejection_percent > 100.0 {
            errors.push(format!(
                "max_ejection_percent {} out of [0, 100]",
                self.max_ejection_percent
            ));
        }
        if self.consecutive_errors == 0 {
            errors.push("consecutive_errors must be > 0".into());
        }
        if self.base_ejection_time_ms == 0 {
            errors.push("base_ejection_time_ms must be > 0".into());
        }
        errors
    }
}

impl Default for CircuitBreakerPolicy {
    fn default() -> Self {
        Self {
            max_connections: 1024,
            max_pending: 1024,
            max_requests: 1024,
            max_retries: 3,
            consecutive_errors: 5,
            interval_ms: 10_000,
            base_ejection_time_ms: 30_000,
            max_ejection_percent: 10.0,
        }
    }
}

impl fmt::Display for CircuitBreakerPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CircuitBreaker(errs={}, eject={}ms, max_eject={}%)",
            self.consecutive_errors, self.base_ejection_time_ms, self.max_ejection_percent
        )
    }
}

// ---------------------------------------------------------------------------
// RateLimitPolicy
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RateLimitPolicy {
    pub requests_per_second: f64,
    pub burst_size: u32,
}

impl RateLimitPolicy {
    pub fn new(rps: f64, burst: u32) -> Self {
        Self {
            requests_per_second: rps,
            burst_size: burst,
        }
    }

    /// Minimum inter-request interval in ms.
    pub fn min_interval_ms(&self) -> f64 {
        if self.requests_per_second <= 0.0 {
            f64::INFINITY
        } else {
            1000.0 / self.requests_per_second
        }
    }

    /// Whether a given load (rps) is within the rate limit.
    pub fn admits(&self, load_rps: f64) -> bool {
        load_rps <= self.requests_per_second + self.burst_size as f64
    }

    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if self.requests_per_second <= 0.0 {
            errors.push("requests_per_second must be > 0".into());
        }
        if self.burst_size == 0 {
            errors.push("burst_size must be > 0".into());
        }
        errors
    }
}

impl fmt::Display for RateLimitPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RateLimit({} rps, burst={})",
            self.requests_per_second, self.burst_size
        )
    }
}

// ---------------------------------------------------------------------------
// BulkheadPolicy
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BulkheadPolicy {
    pub max_concurrent: u32,
    pub max_wait_ms: u64,
}

impl BulkheadPolicy {
    pub fn new(max_concurrent: u32, max_wait_ms: u64) -> Self {
        Self {
            max_concurrent,
            max_wait_ms,
        }
    }

    /// Whether the bulkhead would reject under `current` concurrent requests.
    pub fn would_reject(&self, current: u32) -> bool {
        current >= self.max_concurrent
    }

    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if self.max_concurrent == 0 {
            errors.push("max_concurrent must be > 0".into());
        }
        errors
    }
}

impl Default for BulkheadPolicy {
    fn default() -> Self {
        Self::new(100, 1000)
    }
}

impl fmt::Display for BulkheadPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Bulkhead(max={}, wait={}ms)",
            self.max_concurrent, self.max_wait_ms
        )
    }
}

// ---------------------------------------------------------------------------
// ResiliencePolicy – combines all sub-policies
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResiliencePolicy {
    pub retry: Option<RetryPolicy>,
    pub timeout: Option<TimeoutPolicy>,
    pub circuit_breaker: Option<CircuitBreakerPolicy>,
    pub rate_limit: Option<RateLimitPolicy>,
    pub bulkhead: Option<BulkheadPolicy>,
}

impl ResiliencePolicy {
    pub fn empty() -> Self {
        Self {
            retry: None,
            timeout: None,
            circuit_breaker: None,
            rate_limit: None,
            bulkhead: None,
        }
    }

    pub fn with_retry(mut self, r: RetryPolicy) -> Self {
        self.retry = Some(r);
        self
    }

    pub fn with_timeout(mut self, t: TimeoutPolicy) -> Self {
        self.timeout = Some(t);
        self
    }

    pub fn with_circuit_breaker(mut self, cb: CircuitBreakerPolicy) -> Self {
        self.circuit_breaker = Some(cb);
        self
    }

    pub fn with_rate_limit(mut self, rl: RateLimitPolicy) -> Self {
        self.rate_limit = Some(rl);
        self
    }

    pub fn with_bulkhead(mut self, bh: BulkheadPolicy) -> Self {
        self.bulkhead = Some(bh);
        self
    }

    /// Amplification factor from retry policy (1 if no retries).
    pub fn amplification_factor(&self) -> u32 {
        self.retry.as_ref().map_or(1, |r| r.amplification_factor())
    }

    /// Worst-case latency accounting for retries and timeouts.
    pub fn worst_case_latency_ms(&self) -> u64 {
        let timeout = self
            .timeout
            .as_ref()
            .map_or(u64::MAX, |t| t.request_timeout_ms);
        let retry_latency = self
            .retry
            .as_ref()
            .map_or(0, |r| r.worst_case_latency_ms());
        timeout.min(retry_latency.max(1))
    }

    /// Whether any resilience policy is configured.
    pub fn has_any(&self) -> bool {
        self.retry.is_some()
            || self.timeout.is_some()
            || self.circuit_breaker.is_some()
            || self.rate_limit.is_some()
            || self.bulkhead.is_some()
    }

    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if let Some(ref r) = self.retry {
            errors.extend(r.validate().into_iter().map(|e| format!("retry: {e}")));
        }
        if let Some(ref t) = self.timeout {
            errors.extend(t.validate().into_iter().map(|e| format!("timeout: {e}")));
        }
        if let Some(ref cb) = self.circuit_breaker {
            errors.extend(
                cb.validate()
                    .into_iter()
                    .map(|e| format!("circuit_breaker: {e}")),
            );
        }
        if let Some(ref rl) = self.rate_limit {
            errors.extend(
                rl.validate()
                    .into_iter()
                    .map(|e| format!("rate_limit: {e}")),
            );
        }
        if let Some(ref bh) = self.bulkhead {
            errors.extend(bh.validate().into_iter().map(|e| format!("bulkhead: {e}")));
        }
        // Cross-policy consistency: if retry and timeout both exist, per-try fits in overall
        if let (Some(ref r), Some(ref t)) = (&self.retry, &self.timeout) {
            if r.per_try_timeout_ms > t.request_timeout_ms {
                errors.push(format!(
                    "retry per_try_timeout ({}ms) > request_timeout ({}ms)",
                    r.per_try_timeout_ms, t.request_timeout_ms
                ));
            }
            if r.worst_case_latency_ms() > t.request_timeout_ms * 2 {
                errors.push(
                    "retry worst-case latency far exceeds request timeout; retries may be wasted"
                        .into(),
                );
            }
        }
        errors
    }
}

impl Default for ResiliencePolicy {
    fn default() -> Self {
        Self::empty()
    }
}

// ---------------------------------------------------------------------------
// PolicySource
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PolicySource {
    Istio,
    Envoy,
    Kubernetes,
    Custom,
}

impl PolicySource {
    pub fn precedence(self) -> u8 {
        match self {
            PolicySource::Custom => 0,     // highest priority
            PolicySource::Istio => 1,
            PolicySource::Envoy => 2,
            PolicySource::Kubernetes => 3, // lowest
        }
    }
}

impl fmt::Display for PolicySource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PolicySource::Istio => write!(f, "Istio"),
            PolicySource::Envoy => write!(f, "Envoy"),
            PolicySource::Kubernetes => write!(f, "Kubernetes"),
            PolicySource::Custom => write!(f, "Custom"),
        }
    }
}

impl Ord for PolicySource {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.precedence().cmp(&other.precedence())
    }
}

impl PartialOrd for PolicySource {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

// ---------------------------------------------------------------------------
// PolicyPrecedence
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PolicyPrecedence {
    pub source: PolicySource,
    pub specificity: u32,
    pub timestamp: Option<chrono::DateTime<chrono::Utc>>,
}

impl PolicyPrecedence {
    pub fn new(source: PolicySource, specificity: u32) -> Self {
        Self {
            source,
            specificity,
            timestamp: None,
        }
    }

    pub fn with_timestamp(mut self, ts: chrono::DateTime<chrono::Utc>) -> Self {
        self.timestamp = Some(ts);
        self
    }

    /// Returns true if self should override `other`.
    pub fn overrides(&self, other: &PolicyPrecedence) -> bool {
        if self.source.precedence() != other.source.precedence() {
            return self.source.precedence() < other.source.precedence();
        }
        if self.specificity != other.specificity {
            return self.specificity > other.specificity;
        }
        // tie-break by timestamp: newer wins
        match (&self.timestamp, &other.timestamp) {
            (Some(a), Some(b)) => a > b,
            (Some(_), None) => true,
            _ => false,
        }
    }
}

impl Eq for PolicyPrecedence {}

impl Ord for PolicyPrecedence {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if self.overrides(other) {
            std::cmp::Ordering::Less // "less" = higher priority in sort
        } else if other.overrides(self) {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    }
}

impl PartialOrd for PolicyPrecedence {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

// ---------------------------------------------------------------------------
// SourcedPolicy – policy with provenance
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourcedPolicy {
    pub policy: ResiliencePolicy,
    pub precedence: PolicyPrecedence,
}

// ---------------------------------------------------------------------------
// PolicyMerger
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
pub struct PolicyMerger {
    sources: Vec<SourcedPolicy>,
}

impl PolicyMerger {
    pub fn new() -> Self {
        Self {
            sources: Vec::new(),
        }
    }

    pub fn add(&mut self, policy: ResiliencePolicy, precedence: PolicyPrecedence) {
        self.sources.push(SourcedPolicy { policy, precedence });
    }

    pub fn source_count(&self) -> usize {
        self.sources.len()
    }

    /// Merge all sources, using the highest-precedence value for each policy
    /// dimension independently.
    pub fn merge(&self) -> ResiliencePolicy {
        if self.sources.is_empty() {
            return ResiliencePolicy::empty();
        }

        let mut sorted = self.sources.clone();
        sorted.sort_by(|a, b| a.precedence.cmp(&b.precedence));

        let mut result = ResiliencePolicy::empty();
        for sp in &sorted {
            if result.retry.is_none() {
                result.retry = sp.policy.retry.clone();
            }
            if result.timeout.is_none() {
                result.timeout = sp.policy.timeout.clone();
            }
            if result.circuit_breaker.is_none() {
                result.circuit_breaker = sp.policy.circuit_breaker.clone();
            }
            if result.rate_limit.is_none() {
                result.rate_limit = sp.policy.rate_limit.clone();
            }
            if result.bulkhead.is_none() {
                result.bulkhead = sp.policy.bulkhead.clone();
            }
        }
        result
    }

    /// Return conflicts where multiple sources define the same policy dimension
    /// with different values.
    pub fn conflicts(&self) -> Vec<String> {
        let mut conflicts = Vec::new();
        let retries: Vec<_> = self
            .sources
            .iter()
            .filter(|s| s.policy.retry.is_some())
            .collect();
        if retries.len() > 1 {
            conflicts.push(format!(
                "retry policy defined by {} sources: {}",
                retries.len(),
                retries
                    .iter()
                    .map(|s| s.precedence.source.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        let timeouts: Vec<_> = self
            .sources
            .iter()
            .filter(|s| s.policy.timeout.is_some())
            .collect();
        if timeouts.len() > 1 {
            conflicts.push(format!(
                "timeout policy defined by {} sources: {}",
                timeouts.len(),
                timeouts
                    .iter()
                    .map(|s| s.precedence.source.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        let cbs: Vec<_> = self
            .sources
            .iter()
            .filter(|s| s.policy.circuit_breaker.is_some())
            .collect();
        if cbs.len() > 1 {
            conflicts.push(format!(
                "circuit_breaker defined by {} sources: {}",
                cbs.len(),
                cbs.iter()
                    .map(|s| s.precedence.source.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        conflicts
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backoff_fixed() {
        let b = BackoffStrategy::Fixed { delay_ms: 100 };
        assert_eq!(b.delay_for_attempt(0), 100);
        assert_eq!(b.delay_for_attempt(5), 100);
        assert_eq!(b.total_delay(3), 300);
    }

    #[test]
    fn test_backoff_exponential() {
        let b = BackoffStrategy::Exponential {
            base_ms: 100,
            max_ms: 5000,
        };
        assert_eq!(b.delay_for_attempt(0), 100);
        assert_eq!(b.delay_for_attempt(1), 200);
        assert_eq!(b.delay_for_attempt(2), 400);
        assert_eq!(b.delay_for_attempt(10), 5000); // capped
    }

    #[test]
    fn test_backoff_linear() {
        let b = BackoffStrategy::Linear { increment_ms: 50 };
        assert_eq!(b.delay_for_attempt(0), 50);
        assert_eq!(b.delay_for_attempt(1), 100);
        assert_eq!(b.delay_for_attempt(2), 150);
    }

    #[test]
    fn test_retry_policy_worst_case() {
        let r = RetryPolicy::new(3).with_per_try_timeout(1000);
        // 4 attempts * 1000ms + backoff(0..2)
        assert!(r.worst_case_latency_ms() > 4000);
    }

    #[test]
    fn test_retry_policy_amplification() {
        let r = RetryPolicy::new(5);
        assert_eq!(r.amplification_factor(), 6);
    }

    #[test]
    fn test_retry_policy_validate_ok() {
        let r = RetryPolicy::new(3);
        assert!(r.validate().is_empty());
    }

    #[test]
    fn test_retry_policy_validate_high_retries() {
        let r = RetryPolicy::new(30);
        assert!(!r.validate().is_empty());
    }

    #[test]
    fn test_retry_policy_noop() {
        let r = RetryPolicy::new(0);
        assert!(r.is_noop());
    }

    #[test]
    fn test_timeout_policy_consistency() {
        let t = TimeoutPolicy::new(5000).with_per_try_timeout(2000);
        assert!(t.is_consistent());

        let bad = TimeoutPolicy::new(1000).with_per_try_timeout(2000);
        assert!(!bad.is_consistent());
    }

    #[test]
    fn test_timeout_effective_per_try() {
        let t = TimeoutPolicy::new(5000);
        assert_eq!(t.effective_per_try_timeout_ms(), 5000);
        let t2 = t.with_per_try_timeout(1000);
        assert_eq!(t2.effective_per_try_timeout_ms(), 1000);
    }

    #[test]
    fn test_circuit_breaker_strict() {
        let cb = CircuitBreakerPolicy::new()
            .with_consecutive_errors(2)
            .with_max_ejection_percent(60.0);
        assert!(cb.is_strict());
    }

    #[test]
    fn test_rate_limit_admits() {
        let rl = RateLimitPolicy::new(100.0, 20);
        assert!(rl.admits(110.0));
        assert!(!rl.admits(200.0));
    }

    #[test]
    fn test_bulkhead_would_reject() {
        let bh = BulkheadPolicy::new(10, 500);
        assert!(!bh.would_reject(5));
        assert!(bh.would_reject(10));
    }

    #[test]
    fn test_resilience_policy_validate_cross_policy() {
        let rp = ResiliencePolicy::empty()
            .with_retry(RetryPolicy::new(3).with_per_try_timeout(10000))
            .with_timeout(TimeoutPolicy::new(5000));
        let errs = rp.validate();
        assert!(errs.iter().any(|e| e.contains("per_try_timeout")));
    }

    #[test]
    fn test_policy_source_precedence() {
        assert!(PolicySource::Custom.precedence() < PolicySource::Istio.precedence());
        assert!(PolicySource::Istio.precedence() < PolicySource::Kubernetes.precedence());
    }

    #[test]
    fn test_policy_precedence_overrides() {
        let a = PolicyPrecedence::new(PolicySource::Custom, 1);
        let b = PolicyPrecedence::new(PolicySource::Istio, 1);
        assert!(a.overrides(&b));
        assert!(!b.overrides(&a));
    }

    #[test]
    fn test_policy_precedence_specificity_tiebreak() {
        let a = PolicyPrecedence::new(PolicySource::Istio, 5);
        let b = PolicyPrecedence::new(PolicySource::Istio, 3);
        assert!(a.overrides(&b));
    }

    #[test]
    fn test_policy_merger_basic() {
        let mut merger = PolicyMerger::new();
        let istio_policy = ResiliencePolicy::empty().with_retry(RetryPolicy::new(3));
        let k8s_policy = ResiliencePolicy::empty()
            .with_retry(RetryPolicy::new(5))
            .with_timeout(TimeoutPolicy::new(10_000));

        merger.add(istio_policy, PolicyPrecedence::new(PolicySource::Istio, 1));
        merger.add(
            k8s_policy,
            PolicyPrecedence::new(PolicySource::Kubernetes, 1),
        );

        let merged = merger.merge();
        // Istio has higher precedence, so its retry wins
        assert_eq!(merged.retry.as_ref().unwrap().max_retries, 3);
        // timeout only from K8s
        assert!(merged.timeout.is_some());
    }

    #[test]
    fn test_policy_merger_conflicts() {
        let mut merger = PolicyMerger::new();
        merger.add(
            ResiliencePolicy::empty().with_retry(RetryPolicy::new(3)),
            PolicyPrecedence::new(PolicySource::Istio, 1),
        );
        merger.add(
            ResiliencePolicy::empty().with_retry(RetryPolicy::new(5)),
            PolicyPrecedence::new(PolicySource::Envoy, 1),
        );
        let conflicts = merger.conflicts();
        assert!(!conflicts.is_empty());
    }

    #[test]
    fn test_retry_condition_display() {
        assert_eq!(RetryCondition::ServerError.to_string(), "5xx");
        assert_eq!(RetryCondition::ConnectFailure.to_string(), "connect-failure");
    }

    #[test]
    fn test_retry_condition_safe() {
        assert!(RetryCondition::ConnectFailure.is_always_safe());
        assert!(!RetryCondition::ServerError.is_always_safe());
    }

    #[test]
    fn test_resilience_policy_has_any() {
        assert!(!ResiliencePolicy::empty().has_any());
        assert!(ResiliencePolicy::empty()
            .with_retry(RetryPolicy::new(1))
            .has_any());
    }

    #[test]
    fn test_serialization_roundtrip() {
        let p = ResiliencePolicy::empty()
            .with_retry(RetryPolicy::new(3))
            .with_timeout(TimeoutPolicy::new(5000))
            .with_circuit_breaker(CircuitBreakerPolicy::new());
        let json = serde_json::to_string(&p).unwrap();
        let back: ResiliencePolicy = serde_json::from_str(&json).unwrap();
        assert_eq!(p, back);
    }
}
