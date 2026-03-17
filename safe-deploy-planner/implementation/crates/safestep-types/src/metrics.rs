// Metrics collection types for the SafeStep deployment planner.

use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::time::Instant;

use serde::{Deserialize, Serialize};

// ─── Counter ────────────────────────────────────────────────────────────

/// A monotonically increasing counter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Counter {
    value: u64,
    name: String,
}

impl Counter {
    /// Create a new counter with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self { value: 0, name: name.into() }
    }

    /// Increment the counter by 1.
    pub fn increment(&mut self) {
        self.value = self.value.saturating_add(1);
    }

    /// Increment by a specific amount.
    pub fn increment_by(&mut self, amount: u64) {
        self.value = self.value.saturating_add(amount);
    }

    /// Get the current value.
    pub fn get(&self) -> u64 {
        self.value
    }

    /// Reset to zero.
    pub fn reset(&mut self) {
        self.value = 0;
    }

    /// Get the counter name.
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl fmt::Display for Counter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}={}", self.name, self.value)
    }
}

// ─── AtomicCounter ──────────────────────────────────────────────────────

/// A thread-safe atomic counter.
#[derive(Debug)]
pub struct AtomicCounter {
    value: AtomicU64,
    name: String,
}

impl AtomicCounter {
    /// Create a new atomic counter.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            value: AtomicU64::new(0),
            name: name.into(),
        }
    }

    /// Increment by 1 and return the previous value.
    pub fn increment(&self) -> u64 {
        self.value.fetch_add(1, AtomicOrdering::Relaxed)
    }

    /// Get the current value.
    pub fn get(&self) -> u64 {
        self.value.load(AtomicOrdering::Relaxed)
    }

    /// Reset to zero.
    pub fn reset(&self) {
        self.value.store(0, AtomicOrdering::Relaxed);
    }

    /// Name of this counter.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Take a snapshot as a regular Counter.
    pub fn snapshot(&self) -> Counter {
        Counter {
            value: self.get(),
            name: self.name.clone(),
        }
    }
}

impl fmt::Display for AtomicCounter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}={}", self.name, self.get())
    }
}

// ─── Gauge ──────────────────────────────────────────────────────────────

/// A value that can go up and down.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gauge {
    value: f64,
    name: String,
    min_seen: f64,
    max_seen: f64,
}

impl Gauge {
    /// Create a new gauge starting at zero.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            value: 0.0,
            name: name.into(),
            min_seen: 0.0,
            max_seen: 0.0,
        }
    }

    /// Set the value.
    pub fn set(&mut self, value: f64) {
        self.value = value;
        if value < self.min_seen { self.min_seen = value; }
        if value > self.max_seen { self.max_seen = value; }
    }

    /// Get the current value.
    pub fn get(&self) -> f64 {
        self.value
    }

    /// Increment by a delta.
    pub fn increment(&mut self, delta: f64) {
        self.set(self.value + delta);
    }

    /// Decrement by a delta.
    pub fn decrement(&mut self, delta: f64) {
        self.set(self.value - delta);
    }

    /// Reset to zero and clear min/max tracking.
    pub fn reset(&mut self) {
        self.value = 0.0;
        self.min_seen = 0.0;
        self.max_seen = 0.0;
    }

    /// Get the minimum value seen since creation/reset.
    pub fn min_seen(&self) -> f64 {
        self.min_seen
    }

    /// Get the maximum value seen since creation/reset.
    pub fn max_seen(&self) -> f64 {
        self.max_seen
    }

    /// Get the gauge name.
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl fmt::Display for Gauge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}={:.2} [min={:.2}, max={:.2}]", self.name, self.value, self.min_seen, self.max_seen)
    }
}

// ─── Histogram ──────────────────────────────────────────────────────────

/// Records a distribution of values and computes percentiles.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Histogram {
    values: Vec<f64>,
    name: String,
    sorted: bool,
}

impl Histogram {
    /// Create a new empty histogram.
    pub fn new(name: impl Into<String>) -> Self {
        Self { values: Vec::new(), name: name.into(), sorted: true }
    }

    /// Record a value.
    pub fn record(&mut self, value: f64) {
        self.values.push(value);
        self.sorted = false;
    }

    /// Number of recorded values.
    pub fn count(&self) -> usize {
        self.values.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    fn ensure_sorted(&mut self) {
        if !self.sorted {
            self.values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            self.sorted = true;
        }
    }

    /// Minimum value.
    pub fn min(&mut self) -> Option<f64> {
        self.ensure_sorted();
        self.values.first().copied()
    }

    /// Maximum value.
    pub fn max(&mut self) -> Option<f64> {
        self.ensure_sorted();
        self.values.last().copied()
    }

    /// Sum of all values.
    pub fn sum(&self) -> f64 {
        self.values.iter().sum()
    }

    /// Arithmetic mean.
    pub fn mean(&self) -> Option<f64> {
        if self.values.is_empty() {
            return None;
        }
        Some(self.sum() / self.values.len() as f64)
    }

    /// Median value (50th percentile).
    pub fn median(&mut self) -> Option<f64> {
        self.percentile(50.0)
    }

    /// 95th percentile.
    pub fn p95(&mut self) -> Option<f64> {
        self.percentile(95.0)
    }

    /// 99th percentile.
    pub fn p99(&mut self) -> Option<f64> {
        self.percentile(99.0)
    }

    /// Compute a given percentile (0-100).
    pub fn percentile(&mut self, p: f64) -> Option<f64> {
        if self.values.is_empty() {
            return None;
        }
        self.ensure_sorted();
        let idx = (p / 100.0 * (self.values.len() - 1) as f64).round() as usize;
        let idx = idx.min(self.values.len() - 1);
        Some(self.values[idx])
    }

    /// Variance of the distribution.
    pub fn variance(&self) -> Option<f64> {
        let mean = self.mean()?;
        let n = self.values.len() as f64;
        let sum_sq: f64 = self.values.iter().map(|v| (v - mean).powi(2)).sum();
        Some(sum_sq / n)
    }

    /// Standard deviation.
    pub fn stddev(&self) -> Option<f64> {
        self.variance().map(|v| v.sqrt())
    }

    /// Reset the histogram.
    pub fn reset(&mut self) {
        self.values.clear();
        self.sorted = true;
    }

    /// Get the histogram name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get a copy of all recorded values (sorted).
    pub fn sorted_values(&mut self) -> Vec<f64> {
        self.ensure_sorted();
        self.values.clone()
    }
}

impl fmt::Display for Histogram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}(n={}", self.name, self.values.len())?;
        if let Some(mean) = self.mean() {
            write!(f, ", mean={:.3}", mean)?;
        }
        write!(f, ")")
    }
}

// ─── Timer ──────────────────────────────────────────────────────────────

/// A timer for measuring elapsed time.
#[derive(Debug, Clone)]
pub struct Timer {
    start: Option<Instant>,
    accumulated: std::time::Duration,
    name: String,
    running: bool,
}

impl Timer {
    /// Create a new timer (not started).
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            start: None,
            accumulated: std::time::Duration::ZERO,
            name: name.into(),
            running: false,
        }
    }

    /// Start the timer.
    pub fn start(&mut self) {
        if !self.running {
            self.start = Some(Instant::now());
            self.running = true;
        }
    }

    /// Stop the timer and return elapsed since start.
    pub fn stop(&mut self) -> std::time::Duration {
        if self.running {
            if let Some(start) = self.start.take() {
                let elapsed = start.elapsed();
                self.accumulated += elapsed;
                self.running = false;
                return elapsed;
            }
        }
        std::time::Duration::ZERO
    }

    /// Return total accumulated time.
    pub fn elapsed(&self) -> std::time::Duration {
        let running_extra = if self.running {
            self.start.map(|s| s.elapsed()).unwrap_or_default()
        } else {
            std::time::Duration::ZERO
        };
        self.accumulated + running_extra
    }

    /// Is the timer currently running?
    pub fn is_running(&self) -> bool {
        self.running
    }

    /// Reset the timer.
    pub fn reset(&mut self) {
        self.start = None;
        self.accumulated = std::time::Duration::ZERO;
        self.running = false;
    }

    /// Get the timer name.
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl fmt::Display for Timer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let elapsed = self.elapsed();
        write!(f, "{}={:.3}s", self.name, elapsed.as_secs_f64())
    }
}

// ─── MetricValue ────────────────────────────────────────────────────────

/// A tagged union of metric types for the registry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricValue {
    Counter(Counter),
    Gauge(Gauge),
    Histogram(Histogram),
}

impl MetricValue {
    /// Get the name of the metric.
    pub fn name(&self) -> &str {
        match self {
            MetricValue::Counter(c) => c.name(),
            MetricValue::Gauge(g) => g.name(),
            MetricValue::Histogram(h) => h.name(),
        }
    }
}

impl fmt::Display for MetricValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MetricValue::Counter(c) => write!(f, "{}", c),
            MetricValue::Gauge(g) => write!(f, "{}", g),
            MetricValue::Histogram(h) => write!(f, "{}", h),
        }
    }
}

// ─── MetricsRegistry ────────────────────────────────────────────────────

/// Central registry for named metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsRegistry {
    metrics: HashMap<String, MetricValue>,
}

impl MetricsRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self { metrics: HashMap::new() }
    }

    /// Register a counter. Returns false if the name is already taken.
    pub fn register_counter(&mut self, name: impl Into<String>) -> bool {
        let name = name.into();
        if self.metrics.contains_key(&name) {
            return false;
        }
        self.metrics.insert(name.clone(), MetricValue::Counter(Counter::new(name)));
        true
    }

    /// Register a gauge.
    pub fn register_gauge(&mut self, name: impl Into<String>) -> bool {
        let name = name.into();
        if self.metrics.contains_key(&name) {
            return false;
        }
        self.metrics.insert(name.clone(), MetricValue::Gauge(Gauge::new(name)));
        true
    }

    /// Register a histogram.
    pub fn register_histogram(&mut self, name: impl Into<String>) -> bool {
        let name = name.into();
        if self.metrics.contains_key(&name) {
            return false;
        }
        self.metrics.insert(name.clone(), MetricValue::Histogram(Histogram::new(name)));
        true
    }

    /// Get a counter by name.
    pub fn get_counter(&self, name: &str) -> Option<&Counter> {
        match self.metrics.get(name) {
            Some(MetricValue::Counter(c)) => Some(c),
            _ => None,
        }
    }

    /// Get a mutable counter by name.
    pub fn get_counter_mut(&mut self, name: &str) -> Option<&mut Counter> {
        match self.metrics.get_mut(name) {
            Some(MetricValue::Counter(c)) => Some(c),
            _ => None,
        }
    }

    /// Get a gauge by name.
    pub fn get_gauge(&self, name: &str) -> Option<&Gauge> {
        match self.metrics.get(name) {
            Some(MetricValue::Gauge(g)) => Some(g),
            _ => None,
        }
    }

    /// Get a mutable gauge by name.
    pub fn get_gauge_mut(&mut self, name: &str) -> Option<&mut Gauge> {
        match self.metrics.get_mut(name) {
            Some(MetricValue::Gauge(g)) => Some(g),
            _ => None,
        }
    }

    /// Get a histogram by name.
    pub fn get_histogram(&self, name: &str) -> Option<&Histogram> {
        match self.metrics.get(name) {
            Some(MetricValue::Histogram(h)) => Some(h),
            _ => None,
        }
    }

    /// Get a mutable histogram by name.
    pub fn get_histogram_mut(&mut self, name: &str) -> Option<&mut Histogram> {
        match self.metrics.get_mut(name) {
            Some(MetricValue::Histogram(h)) => Some(h),
            _ => None,
        }
    }

    /// Number of registered metrics.
    pub fn len(&self) -> usize {
        self.metrics.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.metrics.is_empty()
    }

    /// List all metric names.
    pub fn names(&self) -> Vec<&String> {
        self.metrics.keys().collect()
    }

    /// Take a snapshot of all metrics.
    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            metrics: self.metrics.clone(),
            timestamp_epoch_ms: chrono::Utc::now().timestamp_millis(),
        }
    }

    /// Reset all metrics.
    pub fn reset_all(&mut self) {
        for v in self.metrics.values_mut() {
            match v {
                MetricValue::Counter(c) => c.reset(),
                MetricValue::Gauge(g) => g.reset(),
                MetricValue::Histogram(h) => h.reset(),
            }
        }
    }

    /// Remove a metric by name.
    pub fn remove(&mut self, name: &str) -> bool {
        self.metrics.remove(name).is_some()
    }
}

impl Default for MetricsRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for MetricsRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MetricsRegistry({} metrics)", self.metrics.len())
    }
}

// ─── MetricsSnapshot ────────────────────────────────────────────────────

/// A serializable snapshot of all metrics at a point in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSnapshot {
    pub metrics: HashMap<String, MetricValue>,
    pub timestamp_epoch_ms: i64,
}

impl MetricsSnapshot {
    /// Get a counter value from the snapshot.
    pub fn counter_value(&self, name: &str) -> Option<u64> {
        match self.metrics.get(name) {
            Some(MetricValue::Counter(c)) => Some(c.get()),
            _ => None,
        }
    }

    /// Get a gauge value from the snapshot.
    pub fn gauge_value(&self, name: &str) -> Option<f64> {
        match self.metrics.get(name) {
            Some(MetricValue::Gauge(g)) => Some(g.get()),
            _ => None,
        }
    }

    /// Number of metrics in the snapshot.
    pub fn len(&self) -> usize {
        self.metrics.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.metrics.is_empty()
    }
}

impl fmt::Display for MetricsSnapshot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MetricsSnapshot({} metrics at {})", self.metrics.len(), self.timestamp_epoch_ms)
    }
}

// ─── SolverMetrics ──────────────────────────────────────────────────────

/// Metrics collected during SAT/SMT solving.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SolverMetrics {
    pub decisions: u64,
    pub propagations: u64,
    pub conflicts: u64,
    pub restarts: u64,
    pub learned_clauses: u64,
    pub deleted_clauses: u64,
    pub solving_time_ms: u64,
    pub peak_memory_bytes: u64,
}

impl SolverMetrics {
    /// Create new empty solver metrics.
    pub fn new() -> Self {
        Self::default()
    }

    /// Decisions per second.
    pub fn decisions_per_sec(&self) -> f64 {
        if self.solving_time_ms == 0 { return 0.0; }
        self.decisions as f64 / (self.solving_time_ms as f64 / 1000.0)
    }

    /// Conflict rate (conflicts per decision).
    pub fn conflict_rate(&self) -> f64 {
        if self.decisions == 0 { return 0.0; }
        self.conflicts as f64 / self.decisions as f64
    }

    /// Propagation ratio (propagations per decision).
    pub fn propagation_ratio(&self) -> f64 {
        if self.decisions == 0 { return 0.0; }
        self.propagations as f64 / self.decisions as f64
    }

    /// Average learned clause size (approximation: learned per conflict).
    pub fn learn_rate(&self) -> f64 {
        if self.conflicts == 0 { return 0.0; }
        self.learned_clauses as f64 / self.conflicts as f64
    }

    /// Merge with another set of solver metrics (add counters).
    pub fn merge(&mut self, other: &SolverMetrics) {
        self.decisions += other.decisions;
        self.propagations += other.propagations;
        self.conflicts += other.conflicts;
        self.restarts += other.restarts;
        self.learned_clauses += other.learned_clauses;
        self.deleted_clauses += other.deleted_clauses;
        self.solving_time_ms += other.solving_time_ms;
        self.peak_memory_bytes = self.peak_memory_bytes.max(other.peak_memory_bytes);
    }
}

impl fmt::Display for SolverMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SolverMetrics(decisions={}, conflicts={}, restarts={}, time={:.1}s)",
            self.decisions,
            self.conflicts,
            self.restarts,
            self.solving_time_ms as f64 / 1000.0
        )
    }
}

// ─── PlannerMetrics ─────────────────────────────────────────────────────

/// Metrics collected during plan search.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PlannerMetrics {
    pub plans_found: u64,
    pub plans_rejected: u64,
    pub plans_optimized: u64,
    pub depth_reached: u32,
    pub cegar_iterations: u32,
    pub states_explored: u64,
    pub total_time_ms: u64,
}

impl PlannerMetrics {
    /// Create new empty planner metrics.
    pub fn new() -> Self {
        Self::default()
    }

    /// Plans evaluated per second.
    pub fn plans_per_sec(&self) -> f64 {
        if self.total_time_ms == 0 { return 0.0; }
        (self.plans_found + self.plans_rejected) as f64 / (self.total_time_ms as f64 / 1000.0)
    }

    /// Acceptance rate.
    pub fn acceptance_rate(&self) -> f64 {
        let total = self.plans_found + self.plans_rejected;
        if total == 0 { return 0.0; }
        self.plans_found as f64 / total as f64
    }

    /// States explored per second.
    pub fn exploration_rate(&self) -> f64 {
        if self.total_time_ms == 0 { return 0.0; }
        self.states_explored as f64 / (self.total_time_ms as f64 / 1000.0)
    }

    /// Merge with another set of planner metrics.
    pub fn merge(&mut self, other: &PlannerMetrics) {
        self.plans_found += other.plans_found;
        self.plans_rejected += other.plans_rejected;
        self.plans_optimized += other.plans_optimized;
        self.depth_reached = self.depth_reached.max(other.depth_reached);
        self.cegar_iterations += other.cegar_iterations;
        self.states_explored += other.states_explored;
        self.total_time_ms += other.total_time_ms;
    }
}

impl fmt::Display for PlannerMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PlannerMetrics(found={}, rejected={}, depth={}, cegar={}, time={:.1}s)",
            self.plans_found,
            self.plans_rejected,
            self.depth_reached,
            self.cegar_iterations,
            self.total_time_ms as f64 / 1000.0
        )
    }
}

// ─── EncodingMetrics ────────────────────────────────────────────────────

/// Metrics collected during constraint encoding.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EncodingMetrics {
    pub variables: u64,
    pub clauses: u64,
    pub interval_constraints: u64,
    pub bdd_nodes: u64,
    pub unit_propagations: u64,
    pub encoding_time_ms: u64,
    pub compression_ratio: f64,
}

impl EncodingMetrics {
    /// Create new empty encoding metrics.
    pub fn new() -> Self {
        Self::default()
    }

    /// Variables per clause ratio.
    pub fn variables_per_clause(&self) -> f64 {
        if self.clauses == 0 { return 0.0; }
        self.variables as f64 / self.clauses as f64
    }

    /// Encoding speed (clauses per second).
    pub fn clauses_per_sec(&self) -> f64 {
        if self.encoding_time_ms == 0 { return 0.0; }
        self.clauses as f64 / (self.encoding_time_ms as f64 / 1000.0)
    }

    /// Merge with another set of encoding metrics.
    pub fn merge(&mut self, other: &EncodingMetrics) {
        self.variables += other.variables;
        self.clauses += other.clauses;
        self.interval_constraints += other.interval_constraints;
        self.bdd_nodes += other.bdd_nodes;
        self.unit_propagations += other.unit_propagations;
        self.encoding_time_ms += other.encoding_time_ms;
        if self.compression_ratio == 0.0 {
            self.compression_ratio = other.compression_ratio;
        } else if other.compression_ratio > 0.0 {
            self.compression_ratio = (self.compression_ratio + other.compression_ratio) / 2.0;
        }
    }
}

impl fmt::Display for EncodingMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "EncodingMetrics(vars={}, clauses={}, intervals={}, bdds={}, time={:.1}s)",
            self.variables,
            self.clauses,
            self.interval_constraints,
            self.bdd_nodes,
            self.encoding_time_ms as f64 / 1000.0
        )
    }
}

// ─── CombinedMetrics ────────────────────────────────────────────────────

/// All metrics from a complete planning run.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CombinedMetrics {
    pub solver: SolverMetrics,
    pub planner: PlannerMetrics,
    pub encoding: EncodingMetrics,
    pub total_time_ms: u64,
}

impl CombinedMetrics {
    /// Create new empty combined metrics.
    pub fn new() -> Self {
        Self::default()
    }

    /// Total wall-clock time in seconds.
    pub fn total_time_secs(&self) -> f64 {
        self.total_time_ms as f64 / 1000.0
    }

    /// Fraction of time spent solving.
    pub fn solver_time_fraction(&self) -> f64 {
        if self.total_time_ms == 0 { return 0.0; }
        self.solver.solving_time_ms as f64 / self.total_time_ms as f64
    }

    /// Fraction of time spent encoding.
    pub fn encoding_time_fraction(&self) -> f64 {
        if self.total_time_ms == 0 { return 0.0; }
        self.encoding.encoding_time_ms as f64 / self.total_time_ms as f64
    }

    /// Populate into a metrics registry.
    pub fn to_registry(&self) -> MetricsRegistry {
        let mut reg = MetricsRegistry::new();
        reg.register_counter("solver.decisions");
        if let Some(c) = reg.get_counter_mut("solver.decisions") {
            c.increment_by(self.solver.decisions);
        }
        reg.register_counter("solver.conflicts");
        if let Some(c) = reg.get_counter_mut("solver.conflicts") {
            c.increment_by(self.solver.conflicts);
        }
        reg.register_counter("solver.restarts");
        if let Some(c) = reg.get_counter_mut("solver.restarts") {
            c.increment_by(self.solver.restarts);
        }
        reg.register_counter("planner.plans_found");
        if let Some(c) = reg.get_counter_mut("planner.plans_found") {
            c.increment_by(self.planner.plans_found);
        }
        reg.register_gauge("encoding.variables");
        if let Some(g) = reg.get_gauge_mut("encoding.variables") {
            g.set(self.encoding.variables as f64);
        }
        reg.register_gauge("encoding.clauses");
        if let Some(g) = reg.get_gauge_mut("encoding.clauses") {
            g.set(self.encoding.clauses as f64);
        }
        reg
    }
}

impl fmt::Display for CombinedMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CombinedMetrics(total={:.1}s, solver={:.0}%, encoding={:.0}%)",
            self.total_time_secs(),
            self.solver_time_fraction() * 100.0,
            self.encoding_time_fraction() * 100.0
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_counter_basic() {
        let mut c = Counter::new("test");
        assert_eq!(c.get(), 0);
        c.increment();
        assert_eq!(c.get(), 1);
        c.increment_by(5);
        assert_eq!(c.get(), 6);
        c.reset();
        assert_eq!(c.get(), 0);
    }

    #[test]
    fn test_counter_display() {
        let mut c = Counter::new("ops");
        c.increment_by(42);
        assert_eq!(format!("{}", c), "ops=42");
    }

    #[test]
    fn test_counter_name() {
        let c = Counter::new("my.counter");
        assert_eq!(c.name(), "my.counter");
    }

    #[test]
    fn test_atomic_counter() {
        let c = AtomicCounter::new("atomic_test");
        assert_eq!(c.get(), 0);
        c.increment();
        c.increment();
        assert_eq!(c.get(), 2);
        c.reset();
        assert_eq!(c.get(), 0);
    }

    #[test]
    fn test_atomic_counter_snapshot() {
        let c = AtomicCounter::new("snap");
        c.increment();
        c.increment();
        let snap = c.snapshot();
        assert_eq!(snap.get(), 2);
        assert_eq!(snap.name(), "snap");
    }

    #[test]
    fn test_gauge_basic() {
        let mut g = Gauge::new("temp");
        assert_eq!(g.get(), 0.0);
        g.set(42.5);
        assert_eq!(g.get(), 42.5);
        g.increment(1.5);
        assert_eq!(g.get(), 44.0);
        g.decrement(4.0);
        assert_eq!(g.get(), 40.0);
    }

    #[test]
    fn test_gauge_min_max() {
        let mut g = Gauge::new("level");
        g.set(10.0);
        g.set(5.0);
        g.set(20.0);
        g.set(15.0);
        assert_eq!(g.min_seen(), 0.0); // initial zero
        assert_eq!(g.max_seen(), 20.0);
    }

    #[test]
    fn test_gauge_reset() {
        let mut g = Gauge::new("x");
        g.set(100.0);
        g.reset();
        assert_eq!(g.get(), 0.0);
        assert_eq!(g.min_seen(), 0.0);
        assert_eq!(g.max_seen(), 0.0);
    }

    #[test]
    fn test_histogram_basic() {
        let mut h = Histogram::new("latency");
        assert!(h.is_empty());
        h.record(1.0);
        h.record(2.0);
        h.record(3.0);
        h.record(4.0);
        h.record(5.0);
        assert_eq!(h.count(), 5);
        assert_eq!(h.sum(), 15.0);
        assert!((h.mean().unwrap() - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_histogram_min_max() {
        let mut h = Histogram::new("vals");
        h.record(10.0);
        h.record(3.0);
        h.record(7.0);
        assert_eq!(h.min().unwrap(), 3.0);
        assert_eq!(h.max().unwrap(), 10.0);
    }

    #[test]
    fn test_histogram_median() {
        let mut h = Histogram::new("m");
        for v in &[1.0, 2.0, 3.0, 4.0, 5.0] {
            h.record(*v);
        }
        assert_eq!(h.median().unwrap(), 3.0);
    }

    #[test]
    fn test_histogram_percentiles() {
        let mut h = Histogram::new("p");
        for i in 1..=100 {
            h.record(i as f64);
        }
        let p95 = h.p95().unwrap();
        assert!(p95 >= 94.0 && p95 <= 96.0);
        let p99 = h.p99().unwrap();
        assert!(p99 >= 98.0 && p99 <= 100.0);
    }

    #[test]
    fn test_histogram_empty() {
        let mut h = Histogram::new("empty");
        assert!(h.mean().is_none());
        assert!(h.median().is_none());
        assert!(h.min().is_none());
        assert!(h.max().is_none());
        assert!(h.p95().is_none());
        assert!(h.variance().is_none());
    }

    #[test]
    fn test_histogram_variance_stddev() {
        let mut h = Histogram::new("v");
        h.record(2.0);
        h.record(4.0);
        h.record(4.0);
        h.record(4.0);
        h.record(5.0);
        h.record(5.0);
        h.record(7.0);
        h.record(9.0);
        let var = h.variance().unwrap();
        assert!((var - 4.0).abs() < 0.01);
        let sd = h.stddev().unwrap();
        assert!((sd - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_histogram_reset() {
        let mut h = Histogram::new("r");
        h.record(1.0);
        h.record(2.0);
        h.reset();
        assert!(h.is_empty());
    }

    #[test]
    fn test_timer_basic() {
        let mut t = Timer::new("op");
        assert!(!t.is_running());
        t.start();
        assert!(t.is_running());
        std::thread::sleep(std::time::Duration::from_millis(10));
        let d = t.stop();
        assert!(!t.is_running());
        assert!(d.as_millis() >= 5); // some tolerance
    }

    #[test]
    fn test_timer_accumulated() {
        let mut t = Timer::new("multi");
        t.start();
        std::thread::sleep(std::time::Duration::from_millis(10));
        t.stop();
        t.start();
        std::thread::sleep(std::time::Duration::from_millis(10));
        t.stop();
        assert!(t.elapsed().as_millis() >= 10);
    }

    #[test]
    fn test_timer_reset() {
        let mut t = Timer::new("r");
        t.start();
        std::thread::sleep(std::time::Duration::from_millis(10));
        t.stop();
        t.reset();
        assert_eq!(t.elapsed().as_millis(), 0);
        assert!(!t.is_running());
    }

    #[test]
    fn test_registry_basic() {
        let mut reg = MetricsRegistry::new();
        assert!(reg.is_empty());
        assert!(reg.register_counter("c1"));
        assert!(reg.register_gauge("g1"));
        assert!(reg.register_histogram("h1"));
        assert_eq!(reg.len(), 3);
    }

    #[test]
    fn test_registry_duplicate_name() {
        let mut reg = MetricsRegistry::new();
        assert!(reg.register_counter("dup"));
        assert!(!reg.register_counter("dup"));
        assert!(!reg.register_gauge("dup"));
    }

    #[test]
    fn test_registry_counter_ops() {
        let mut reg = MetricsRegistry::new();
        reg.register_counter("ops");
        reg.get_counter_mut("ops").unwrap().increment();
        reg.get_counter_mut("ops").unwrap().increment();
        assert_eq!(reg.get_counter("ops").unwrap().get(), 2);
    }

    #[test]
    fn test_registry_gauge_ops() {
        let mut reg = MetricsRegistry::new();
        reg.register_gauge("mem");
        reg.get_gauge_mut("mem").unwrap().set(1024.0);
        assert_eq!(reg.get_gauge("mem").unwrap().get(), 1024.0);
    }

    #[test]
    fn test_registry_histogram_ops() {
        let mut reg = MetricsRegistry::new();
        reg.register_histogram("lat");
        reg.get_histogram_mut("lat").unwrap().record(1.0);
        reg.get_histogram_mut("lat").unwrap().record(2.0);
        assert_eq!(reg.get_histogram("lat").unwrap().count(), 2);
    }

    #[test]
    fn test_registry_snapshot() {
        let mut reg = MetricsRegistry::new();
        reg.register_counter("c");
        reg.get_counter_mut("c").unwrap().increment_by(10);
        let snap = reg.snapshot();
        assert_eq!(snap.counter_value("c"), Some(10));
        assert_eq!(snap.len(), 1);
    }

    #[test]
    fn test_registry_reset_all() {
        let mut reg = MetricsRegistry::new();
        reg.register_counter("c");
        reg.register_gauge("g");
        reg.get_counter_mut("c").unwrap().increment_by(5);
        reg.get_gauge_mut("g").unwrap().set(10.0);
        reg.reset_all();
        assert_eq!(reg.get_counter("c").unwrap().get(), 0);
        assert_eq!(reg.get_gauge("g").unwrap().get(), 0.0);
    }

    #[test]
    fn test_registry_remove() {
        let mut reg = MetricsRegistry::new();
        reg.register_counter("temp");
        assert!(reg.remove("temp"));
        assert!(!reg.remove("temp"));
        assert!(reg.is_empty());
    }

    #[test]
    fn test_solver_metrics() {
        let mut m = SolverMetrics::new();
        m.decisions = 1000;
        m.conflicts = 100;
        m.solving_time_ms = 2000;
        assert!((m.conflict_rate() - 0.1).abs() < 0.01);
        assert!((m.decisions_per_sec() - 500.0).abs() < 0.01);
    }

    #[test]
    fn test_solver_metrics_merge() {
        let mut a = SolverMetrics::new();
        a.decisions = 100;
        a.conflicts = 10;
        let mut b = SolverMetrics::new();
        b.decisions = 200;
        b.conflicts = 20;
        a.merge(&b);
        assert_eq!(a.decisions, 300);
        assert_eq!(a.conflicts, 30);
    }

    #[test]
    fn test_solver_metrics_zero_division() {
        let m = SolverMetrics::new();
        assert_eq!(m.decisions_per_sec(), 0.0);
        assert_eq!(m.conflict_rate(), 0.0);
        assert_eq!(m.propagation_ratio(), 0.0);
    }

    #[test]
    fn test_planner_metrics() {
        let mut m = PlannerMetrics::new();
        m.plans_found = 5;
        m.plans_rejected = 15;
        m.total_time_ms = 1000;
        assert!((m.acceptance_rate() - 0.25).abs() < 0.01);
        assert!((m.plans_per_sec() - 20.0).abs() < 0.01);
    }

    #[test]
    fn test_planner_metrics_merge() {
        let mut a = PlannerMetrics::new();
        a.plans_found = 5;
        a.depth_reached = 10;
        let mut b = PlannerMetrics::new();
        b.plans_found = 3;
        b.depth_reached = 15;
        a.merge(&b);
        assert_eq!(a.plans_found, 8);
        assert_eq!(a.depth_reached, 15);
    }

    #[test]
    fn test_encoding_metrics() {
        let mut m = EncodingMetrics::new();
        m.variables = 100;
        m.clauses = 500;
        m.encoding_time_ms = 1000;
        assert!((m.variables_per_clause() - 0.2).abs() < 0.01);
        assert!((m.clauses_per_sec() - 500.0).abs() < 0.01);
    }

    #[test]
    fn test_encoding_metrics_merge() {
        let mut a = EncodingMetrics::new();
        a.variables = 50;
        a.clauses = 100;
        let mut b = EncodingMetrics::new();
        b.variables = 30;
        b.clauses = 60;
        a.merge(&b);
        assert_eq!(a.variables, 80);
        assert_eq!(a.clauses, 160);
    }

    #[test]
    fn test_combined_metrics() {
        let mut cm = CombinedMetrics::new();
        cm.total_time_ms = 10000;
        cm.solver.solving_time_ms = 7000;
        cm.encoding.encoding_time_ms = 2000;
        assert!((cm.total_time_secs() - 10.0).abs() < 0.01);
        assert!((cm.solver_time_fraction() - 0.7).abs() < 0.01);
        assert!((cm.encoding_time_fraction() - 0.2).abs() < 0.01);
    }

    #[test]
    fn test_combined_metrics_to_registry() {
        let mut cm = CombinedMetrics::new();
        cm.solver.decisions = 100;
        cm.solver.conflicts = 10;
        cm.planner.plans_found = 5;
        cm.encoding.variables = 200;
        let reg = cm.to_registry();
        assert_eq!(reg.get_counter("solver.decisions").unwrap().get(), 100);
        assert_eq!(reg.get_counter("solver.conflicts").unwrap().get(), 10);
        assert_eq!(reg.get_counter("planner.plans_found").unwrap().get(), 5);
        assert_eq!(reg.get_gauge("encoding.variables").unwrap().get(), 200.0);
    }

    #[test]
    fn test_serde_counter() {
        let mut c = Counter::new("x");
        c.increment_by(42);
        let json = serde_json::to_string(&c).unwrap();
        let c2: Counter = serde_json::from_str(&json).unwrap();
        assert_eq!(c2.get(), 42);
    }

    #[test]
    fn test_serde_gauge() {
        let mut g = Gauge::new("y");
        g.set(3.14);
        let json = serde_json::to_string(&g).unwrap();
        let g2: Gauge = serde_json::from_str(&json).unwrap();
        assert!((g2.get() - 3.14).abs() < 0.001);
    }

    #[test]
    fn test_serde_histogram() {
        let mut h = Histogram::new("z");
        h.record(1.0);
        h.record(2.0);
        let json = serde_json::to_string(&h).unwrap();
        let h2: Histogram = serde_json::from_str(&json).unwrap();
        assert_eq!(h2.count(), 2);
    }

    #[test]
    fn test_serde_solver_metrics() {
        let mut m = SolverMetrics::new();
        m.decisions = 42;
        let json = serde_json::to_string(&m).unwrap();
        let m2: SolverMetrics = serde_json::from_str(&json).unwrap();
        assert_eq!(m2.decisions, 42);
    }

    #[test]
    fn test_serde_snapshot() {
        let mut reg = MetricsRegistry::new();
        reg.register_counter("count");
        reg.get_counter_mut("count").unwrap().increment_by(7);
        let snap = reg.snapshot();
        let json = serde_json::to_string(&snap).unwrap();
        let snap2: MetricsSnapshot = serde_json::from_str(&json).unwrap();
        assert_eq!(snap2.counter_value("count"), Some(7));
    }

    #[test]
    fn test_solver_metrics_learn_rate() {
        let mut m = SolverMetrics::new();
        m.conflicts = 10;
        m.learned_clauses = 30;
        assert!((m.learn_rate() - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_planner_metrics_exploration_rate() {
        let mut m = PlannerMetrics::new();
        m.states_explored = 5000;
        m.total_time_ms = 2000;
        assert!((m.exploration_rate() - 2500.0).abs() < 0.1);
    }

    #[test]
    fn test_snapshot_gauge_value() {
        let mut reg = MetricsRegistry::new();
        reg.register_gauge("load");
        reg.get_gauge_mut("load").unwrap().set(0.75);
        let snap = reg.snapshot();
        assert!((snap.gauge_value("load").unwrap() - 0.75).abs() < 0.01);
    }

    #[test]
    fn test_snapshot_missing() {
        let reg = MetricsRegistry::new();
        let snap = reg.snapshot();
        assert!(snap.counter_value("missing").is_none());
        assert!(snap.gauge_value("missing").is_none());
        assert!(snap.is_empty());
    }

    #[test]
    fn test_metric_value_name() {
        let mv = MetricValue::Counter(Counter::new("test_name"));
        assert_eq!(mv.name(), "test_name");
    }
}
