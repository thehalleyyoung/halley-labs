//! Performance profiling: time breakdown by compiler phase, memory usage tracking,
//! cut generation time vs solve time, cache hit rates, and flamegraph-compatible output.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Phase timer
// ---------------------------------------------------------------------------

/// A named timer that accumulates time across multiple start/stop pairs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseTimer {
    /// Phase name.
    pub name: String,
    /// Total elapsed duration in seconds.
    pub total_secs: f64,
    /// Number of times this phase was entered.
    pub call_count: u64,
    /// Minimum single-call duration in seconds.
    pub min_secs: f64,
    /// Maximum single-call duration in seconds.
    pub max_secs: f64,
    /// Whether the timer is currently running (transient, not serialized).
    #[serde(skip)]
    start_instant: Option<Instant>,
}

impl PhaseTimer {
    /// Create a new timer for a named phase.
    pub fn new(name: &str) -> Self {
        PhaseTimer {
            name: name.to_string(),
            total_secs: 0.0,
            call_count: 0,
            min_secs: f64::INFINITY,
            max_secs: 0.0,
            start_instant: None,
        }
    }

    /// Start the timer.
    pub fn start(&mut self) {
        self.start_instant = Some(Instant::now());
    }

    /// Stop the timer and accumulate elapsed time. Returns elapsed seconds.
    pub fn stop(&mut self) -> f64 {
        if let Some(start) = self.start_instant.take() {
            let elapsed = start.elapsed().as_secs_f64();
            self.total_secs += elapsed;
            self.call_count += 1;
            if elapsed < self.min_secs {
                self.min_secs = elapsed;
            }
            if elapsed > self.max_secs {
                self.max_secs = elapsed;
            }
            elapsed
        } else {
            0.0
        }
    }

    /// Whether the timer is currently running.
    pub fn is_running(&self) -> bool {
        self.start_instant.is_some()
    }

    /// Average time per call.
    pub fn avg_secs(&self) -> f64 {
        if self.call_count == 0 {
            0.0
        } else {
            self.total_secs / self.call_count as f64
        }
    }

    /// Reset the timer.
    pub fn reset(&mut self) {
        self.total_secs = 0.0;
        self.call_count = 0;
        self.min_secs = f64::INFINITY;
        self.max_secs = 0.0;
        self.start_instant = None;
    }
}

impl std::fmt::Display for PhaseTimer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}: {:.4}s total ({} calls, avg={:.4}s, min={:.4}s, max={:.4}s)",
            self.name,
            self.total_secs,
            self.call_count,
            self.avg_secs(),
            if self.min_secs.is_infinite() {
                0.0
            } else {
                self.min_secs
            },
            self.max_secs,
        )
    }
}

// ---------------------------------------------------------------------------
// Phase profile (collection of timers)
// ---------------------------------------------------------------------------

/// A collection of phase timers providing a complete profile of an execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseProfile {
    /// Timers by phase name.
    pub timers: HashMap<String, PhaseTimer>,
    /// Wall-clock time for the entire profiled operation.
    pub wall_time_secs: f64,
    /// Phase ordering (insertion order).
    pub phase_order: Vec<String>,
    /// Start time (transient).
    #[serde(skip)]
    start_instant: Option<Instant>,
}

impl PhaseProfile {
    /// Create a new empty profile.
    pub fn new() -> Self {
        PhaseProfile {
            timers: HashMap::new(),
            wall_time_secs: 0.0,
            phase_order: Vec::new(),
            start_instant: None,
        }
    }

    /// Start wall-clock timing.
    pub fn start(&mut self) {
        self.start_instant = Some(Instant::now());
    }

    /// Stop wall-clock timing.
    pub fn stop(&mut self) {
        if let Some(start) = self.start_instant.take() {
            self.wall_time_secs = start.elapsed().as_secs_f64();
        }
    }

    /// Get or create a timer for a phase.
    pub fn timer(&mut self, name: &str) -> &mut PhaseTimer {
        if !self.timers.contains_key(name) {
            self.phase_order.push(name.to_string());
            self.timers.insert(name.to_string(), PhaseTimer::new(name));
        }
        self.timers.get_mut(name).unwrap()
    }

    /// Start a named phase.
    pub fn start_phase(&mut self, name: &str) {
        if !self.timers.contains_key(name) {
            self.phase_order.push(name.to_string());
            self.timers.insert(name.to_string(), PhaseTimer::new(name));
        }
        self.timers.get_mut(name).unwrap().start();
    }

    /// Stop a named phase.
    pub fn stop_phase(&mut self, name: &str) -> f64 {
        if let Some(timer) = self.timers.get_mut(name) {
            timer.stop()
        } else {
            0.0
        }
    }

    /// Get total time for a phase.
    pub fn phase_time(&self, name: &str) -> f64 {
        self.timers.get(name).map(|t| t.total_secs).unwrap_or(0.0)
    }

    /// Get all phase names in order.
    pub fn phases(&self) -> &[String] {
        &self.phase_order
    }

    /// Total time accounted for by all phases.
    pub fn accounted_time(&self) -> f64 {
        self.timers.values().map(|t| t.total_secs).sum()
    }

    /// Unaccounted time (wall time - accounted time).
    pub fn unaccounted_time(&self) -> f64 {
        (self.wall_time_secs - self.accounted_time()).max(0.0)
    }

    /// Percentage of wall time accounted for.
    pub fn coverage_percent(&self) -> f64 {
        if self.wall_time_secs > 0.0 {
            (self.accounted_time() / self.wall_time_secs * 100.0).min(100.0)
        } else {
            0.0
        }
    }

    /// Per-phase percentages of wall time.
    pub fn phase_percentages(&self) -> HashMap<String, f64> {
        let wall = self.wall_time_secs.max(1e-12);
        self.timers
            .iter()
            .map(|(name, timer)| (name.clone(), (timer.total_secs / wall) * 100.0))
            .collect()
    }

    /// Merge another profile into this one (add times).
    pub fn merge(&mut self, other: &PhaseProfile) {
        self.wall_time_secs += other.wall_time_secs;
        for (name, timer) in &other.timers {
            let entry = self
                .timers
                .entry(name.clone())
                .or_insert_with(|| PhaseTimer::new(name));
            entry.total_secs += timer.total_secs;
            entry.call_count += timer.call_count;
            if timer.min_secs < entry.min_secs {
                entry.min_secs = timer.min_secs;
            }
            if timer.max_secs > entry.max_secs {
                entry.max_secs = timer.max_secs;
            }
            if !self.phase_order.contains(name) {
                self.phase_order.push(name.clone());
            }
        }
    }

    /// Reset all timers.
    pub fn reset(&mut self) {
        for timer in self.timers.values_mut() {
            timer.reset();
        }
        self.wall_time_secs = 0.0;
        self.start_instant = None;
    }
}

impl Default for PhaseProfile {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for PhaseProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Phase Profile (wall time: {:.4}s)", self.wall_time_secs)?;
        let pcts = self.phase_percentages();
        for name in &self.phase_order {
            if let Some(timer) = self.timers.get(name) {
                let pct = pcts.get(name).unwrap_or(&0.0);
                writeln!(f, "  {} ({:.1}%)", timer, pct)?;
            }
        }
        writeln!(
            f,
            "  Coverage: {:.1}% (unaccounted: {:.4}s)",
            self.coverage_percent(),
            self.unaccounted_time()
        )
    }
}

// ---------------------------------------------------------------------------
// Memory snapshot
// ---------------------------------------------------------------------------

/// A snapshot of memory usage at a point in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySnapshot {
    /// Label for this snapshot.
    pub label: String,
    /// Estimated memory usage in bytes.
    pub bytes: usize,
    /// Timestamp relative to profiling start (in seconds).
    pub timestamp_secs: f64,
}

impl MemorySnapshot {
    /// Create a snapshot.
    pub fn new(label: &str, bytes: usize, timestamp_secs: f64) -> Self {
        MemorySnapshot {
            label: label.to_string(),
            bytes,
            timestamp_secs,
        }
    }

    /// Memory in megabytes.
    pub fn megabytes(&self) -> f64 {
        self.bytes as f64 / (1024.0 * 1024.0)
    }

    /// Memory in kilobytes.
    pub fn kilobytes(&self) -> f64 {
        self.bytes as f64 / 1024.0
    }
}

/// Estimate memory usage of a bilevel problem.
pub fn estimate_problem_memory(problem: &bicut_types::BilevelProblem) -> usize {
    let f64_size = std::mem::size_of::<f64>();
    let entry_size = std::mem::size_of::<bicut_types::SparseEntry>();
    let mut total = 0;
    total += problem.upper_obj_c_x.len() * f64_size;
    total += problem.upper_obj_c_y.len() * f64_size;
    total += problem.lower_obj_c.len() * f64_size;
    total += problem.lower_b.len() * f64_size;
    total += problem.upper_constraints_b.len() * f64_size;
    total += problem.lower_a.entries.len() * entry_size;
    total += problem.lower_linking_b.entries.len() * entry_size;
    total += problem.upper_constraints_a.entries.len() * entry_size;
    total += std::mem::size_of::<bicut_types::BilevelProblem>();
    total
}

// ---------------------------------------------------------------------------
// Profiling session
// ---------------------------------------------------------------------------

/// A complete profiling session that tracks phases, memory, and cache hits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilingSession {
    /// Name of this session.
    pub name: String,
    /// Phase profile.
    pub phases: PhaseProfile,
    /// Memory snapshots over time.
    pub memory_snapshots: Vec<MemorySnapshot>,
    /// Cache hit/miss counters.
    pub cache_hits: u64,
    pub cache_misses: u64,
    /// Custom counters.
    pub counters: HashMap<String, u64>,
    /// Session start time (transient).
    #[serde(skip)]
    start_instant: Option<Instant>,
}

impl ProfilingSession {
    /// Create a new session.
    pub fn new(name: &str) -> Self {
        ProfilingSession {
            name: name.to_string(),
            phases: PhaseProfile::new(),
            memory_snapshots: Vec::new(),
            cache_hits: 0,
            cache_misses: 0,
            counters: HashMap::new(),
            start_instant: None,
        }
    }

    /// Start the profiling session.
    pub fn start(&mut self) {
        self.start_instant = Some(Instant::now());
        self.phases.start();
    }

    /// Stop the profiling session.
    pub fn stop(&mut self) {
        self.phases.stop();
    }

    /// Record a cache hit.
    pub fn record_cache_hit(&mut self) {
        self.cache_hits += 1;
    }

    /// Record a cache miss.
    pub fn record_cache_miss(&mut self) {
        self.cache_misses += 1;
    }

    /// Cache hit rate as a percentage.
    pub fn cache_hit_rate(&self) -> f64 {
        let total = self.cache_hits + self.cache_misses;
        if total == 0 {
            0.0
        } else {
            (self.cache_hits as f64 / total as f64) * 100.0
        }
    }

    /// Increment a named counter.
    pub fn increment_counter(&mut self, name: &str, amount: u64) {
        *self.counters.entry(name.to_string()).or_default() += amount;
    }

    /// Get a counter value.
    pub fn counter(&self, name: &str) -> u64 {
        self.counters.get(name).copied().unwrap_or(0)
    }

    /// Take a memory snapshot.
    pub fn snapshot_memory(&mut self, label: &str, bytes: usize) {
        let timestamp = self
            .start_instant
            .map(|s| s.elapsed().as_secs_f64())
            .unwrap_or(0.0);
        self.memory_snapshots
            .push(MemorySnapshot::new(label, bytes, timestamp));
    }

    /// Peak memory observed across snapshots.
    pub fn peak_memory_bytes(&self) -> usize {
        self.memory_snapshots
            .iter()
            .map(|s| s.bytes)
            .max()
            .unwrap_or(0)
    }

    /// Ratio of cut generation time to total LP solve time.
    pub fn cut_vs_solve_ratio(&self) -> f64 {
        let cut_time = self.phases.phase_time("cut_generation");
        let solve_time = self.phases.phase_time("lp_solve");
        if solve_time > 0.0 {
            cut_time / solve_time
        } else {
            0.0
        }
    }

    /// Generate flamegraph-compatible output entries.
    pub fn to_flamegraph_entries(&self) -> Vec<FlamegraphEntry> {
        let mut entries = Vec::new();
        for name in &self.phases.phase_order {
            if let Some(timer) = self.phases.timers.get(name) {
                let micros = (timer.total_secs * 1_000_000.0) as u64;
                entries.push(FlamegraphEntry {
                    stack: vec![self.name.clone(), name.clone()],
                    value: micros,
                });
            }
        }
        // Add unaccounted.
        let unaccounted_micros = (self.phases.unaccounted_time() * 1_000_000.0) as u64;
        if unaccounted_micros > 0 {
            entries.push(FlamegraphEntry {
                stack: vec![self.name.clone(), "unaccounted".to_string()],
                value: unaccounted_micros,
            });
        }
        entries
    }

    /// Generate flamegraph-compatible folded stack format string.
    pub fn to_folded_stacks(&self) -> String {
        let entries = self.to_flamegraph_entries();
        let mut lines = Vec::with_capacity(entries.len());
        for entry in &entries {
            lines.push(format!("{} {}", entry.stack.join(";"), entry.value));
        }
        lines.join("\n")
    }
}

impl std::fmt::Display for ProfilingSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== Profiling Session: {} ===", self.name)?;
        write!(f, "{}", self.phases)?;
        writeln!(
            f,
            "Cache: {} hits, {} misses ({:.1}% hit rate)",
            self.cache_hits,
            self.cache_misses,
            self.cache_hit_rate()
        )?;
        if !self.memory_snapshots.is_empty() {
            writeln!(
                f,
                "Memory: {} snapshots, peak={:.2} MB",
                self.memory_snapshots.len(),
                self.peak_memory_bytes() as f64 / (1024.0 * 1024.0)
            )?;
        }
        if !self.counters.is_empty() {
            writeln!(f, "Counters:")?;
            let mut keys: Vec<_> = self.counters.keys().collect();
            keys.sort();
            for key in keys {
                writeln!(f, "  {}: {}", key, self.counters[key])?;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Flamegraph entry
// ---------------------------------------------------------------------------

/// A single entry for flamegraph-compatible output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlamegraphEntry {
    /// Call stack (outermost first).
    pub stack: Vec<String>,
    /// Value (e.g., microseconds).
    pub value: u64,
}

impl FlamegraphEntry {
    /// Create a new entry.
    pub fn new(stack: Vec<String>, value: u64) -> Self {
        FlamegraphEntry { stack, value }
    }

    /// Folded stack format string.
    pub fn to_folded(&self) -> String {
        format!("{} {}", self.stack.join(";"), self.value)
    }
}

/// Write flamegraph entries to a folded stack file.
pub fn write_folded_stacks(
    entries: &[FlamegraphEntry],
    path: &std::path::Path,
) -> Result<(), crate::BenchError> {
    let content: String = entries
        .iter()
        .map(|e| e.to_folded())
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(path, content).map_err(crate::BenchError::Io)
}

/// Merge multiple profiling sessions into an aggregate.
pub fn merge_sessions(sessions: &[ProfilingSession]) -> ProfilingSession {
    let mut merged = ProfilingSession::new("merged");
    for session in sessions {
        merged.phases.merge(&session.phases);
        merged.cache_hits += session.cache_hits;
        merged.cache_misses += session.cache_misses;
        merged
            .memory_snapshots
            .extend(session.memory_snapshots.clone());
        for (key, &val) in &session.counters {
            *merged.counters.entry(key.clone()).or_default() += val;
        }
    }
    merged
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phase_timer_basic() {
        let mut timer = PhaseTimer::new("test");
        timer.start();
        std::thread::sleep(Duration::from_millis(10));
        let elapsed = timer.stop();
        assert!(elapsed > 0.0);
        assert_eq!(timer.call_count, 1);
        assert!(timer.total_secs > 0.0);
    }

    #[test]
    fn test_phase_timer_multiple_calls() {
        let mut timer = PhaseTimer::new("multi");
        for _ in 0..3 {
            timer.start();
            timer.stop();
        }
        assert_eq!(timer.call_count, 3);
    }

    #[test]
    fn test_phase_profile() {
        let mut profile = PhaseProfile::new();
        profile.start();
        profile.start_phase("phase_a");
        std::thread::sleep(Duration::from_millis(5));
        profile.stop_phase("phase_a");
        profile.start_phase("phase_b");
        std::thread::sleep(Duration::from_millis(5));
        profile.stop_phase("phase_b");
        profile.stop();

        assert!(profile.wall_time_secs > 0.0);
        assert!(profile.phase_time("phase_a") > 0.0);
        assert!(profile.phase_time("phase_b") > 0.0);
        assert_eq!(profile.phases().len(), 2);
    }

    #[test]
    fn test_phase_profile_percentages() {
        let mut profile = PhaseProfile::new();
        profile.wall_time_secs = 10.0;
        let timer_a = profile.timer("a");
        timer_a.total_secs = 3.0;
        timer_a.call_count = 1;
        let timer_b = profile.timer("b");
        timer_b.total_secs = 7.0;
        timer_b.call_count = 1;

        let pcts = profile.phase_percentages();
        assert!((pcts["a"] - 30.0).abs() < 1e-10);
        assert!((pcts["b"] - 70.0).abs() < 1e-10);
        assert!((profile.coverage_percent() - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_memory_snapshot() {
        let snap = MemorySnapshot::new("test", 1024 * 1024, 1.5);
        assert!((snap.megabytes() - 1.0).abs() < 1e-10);
        assert!((snap.kilobytes() - 1024.0).abs() < 1e-10);
    }

    #[test]
    fn test_profiling_session() {
        let mut session = ProfilingSession::new("test_session");
        session.start();
        session.phases.start_phase("work");
        std::thread::sleep(Duration::from_millis(5));
        session.phases.stop_phase("work");
        session.record_cache_hit();
        session.record_cache_hit();
        session.record_cache_miss();
        session.snapshot_memory("after_work", 4096);
        session.increment_counter("cuts", 10);
        session.stop();

        assert!((session.cache_hit_rate() - 66.666).abs() < 1.0);
        assert_eq!(session.peak_memory_bytes(), 4096);
        assert_eq!(session.counter("cuts"), 10);
    }

    #[test]
    fn test_flamegraph_entries() {
        let mut session = ProfilingSession::new("bench");
        session.phases.wall_time_secs = 1.0;
        let timer = session.phases.timer("lp_solve");
        timer.total_secs = 0.6;
        timer.call_count = 1;
        let timer2 = session.phases.timer("cut_gen");
        timer2.total_secs = 0.3;
        timer2.call_count = 1;

        let entries = session.to_flamegraph_entries();
        assert!(!entries.is_empty());
        let folded = session.to_folded_stacks();
        assert!(folded.contains("bench;lp_solve"));
    }

    #[test]
    fn test_merge_sessions() {
        let mut s1 = ProfilingSession::new("s1");
        s1.phases.wall_time_secs = 1.0;
        s1.cache_hits = 5;
        s1.cache_misses = 2;
        s1.increment_counter("cuts", 10);
        let timer = s1.phases.timer("work");
        timer.total_secs = 0.5;
        timer.call_count = 1;

        let mut s2 = ProfilingSession::new("s2");
        s2.phases.wall_time_secs = 2.0;
        s2.cache_hits = 3;
        s2.increment_counter("cuts", 20);
        let timer2 = s2.phases.timer("work");
        timer2.total_secs = 1.0;
        timer2.call_count = 2;

        let merged = merge_sessions(&[s1, s2]);
        assert_eq!(merged.cache_hits, 8);
        assert_eq!(merged.counter("cuts"), 30);
        assert!((merged.phases.phase_time("work") - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_estimate_problem_memory() {
        let problem = bicut_types::BilevelProblem {
            upper_obj_c_x: vec![1.0; 5],
            upper_obj_c_y: vec![1.0; 10],
            lower_obj_c: vec![1.0; 10],
            lower_a: bicut_types::SparseMatrix::new(8, 10),
            lower_b: vec![1.0; 8],
            lower_linking_b: bicut_types::SparseMatrix::new(8, 5),
            upper_constraints_a: bicut_types::SparseMatrix::new(3, 15),
            upper_constraints_b: vec![1.0; 3],
            num_upper_vars: 5,
            num_lower_vars: 10,
            num_lower_constraints: 8,
            num_upper_constraints: 3,
        };
        let mem = estimate_problem_memory(&problem);
        assert!(mem > 0);
    }
}
