//! Metrics and statistics for the NegSynth analysis pipeline.
//!
//! Tracks states explored, paths merged, coverage metrics, timing,
//! and memory usage across all pipeline phases.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, Instant};

// ── Analysis Metrics ─────────────────────────────────────────────────────

/// Top-level metrics for a complete analysis run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisMetrics {
    pub states_explored: u64,
    pub transitions_explored: u64,
    pub paths_explored: u64,
    pub paths_merged: u64,
    pub paths_pruned: u64,
    pub solver_calls: u64,
    pub solver_sat: u64,
    pub solver_unsat: u64,
    pub solver_unknown: u64,
    pub attacks_found: u32,
    pub phase_metrics: HashMap<String, PhaseMetrics>,
}

/// Metrics for a single pipeline phase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseMetrics {
    pub name: String,
    pub duration_ms: u64,
    pub states_in: u64,
    pub states_out: u64,
    pub peak_memory_kb: u64,
    pub errors: u32,
}

impl AnalysisMetrics {
    pub fn new() -> Self {
        AnalysisMetrics {
            states_explored: 0,
            transitions_explored: 0,
            paths_explored: 0,
            paths_merged: 0,
            paths_pruned: 0,
            solver_calls: 0,
            solver_sat: 0,
            solver_unsat: 0,
            solver_unknown: 0,
            attacks_found: 0,
            phase_metrics: HashMap::new(),
        }
    }

    pub fn record_state_explored(&mut self) {
        self.states_explored += 1;
    }

    pub fn record_transition_explored(&mut self) {
        self.transitions_explored += 1;
    }

    pub fn record_path_explored(&mut self) {
        self.paths_explored += 1;
    }

    pub fn record_path_merged(&mut self) {
        self.paths_merged += 1;
    }

    pub fn record_path_pruned(&mut self) {
        self.paths_pruned += 1;
    }

    pub fn record_solver_call(&mut self, sat: bool, unknown: bool) {
        self.solver_calls += 1;
        if unknown {
            self.solver_unknown += 1;
        } else if sat {
            self.solver_sat += 1;
        } else {
            self.solver_unsat += 1;
        }
    }

    pub fn record_attack_found(&mut self) {
        self.attacks_found += 1;
    }

    pub fn add_phase(&mut self, metrics: PhaseMetrics) {
        self.phase_metrics.insert(metrics.name.clone(), metrics);
    }

    /// Total analysis time across all phases.
    pub fn total_duration_ms(&self) -> u64 {
        self.phase_metrics.values().map(|p| p.duration_ms).sum()
    }

    /// Merge ratio: merged paths / total paths explored.
    pub fn merge_ratio(&self) -> f64 {
        if self.paths_explored == 0 {
            return 0.0;
        }
        self.paths_merged as f64 / self.paths_explored as f64
    }

    /// Solver success rate: (sat + unsat) / total calls.
    pub fn solver_success_rate(&self) -> f64 {
        if self.solver_calls == 0 {
            return 0.0;
        }
        (self.solver_sat + self.solver_unsat) as f64 / self.solver_calls as f64
    }

    /// Aggregate all phase errors.
    pub fn total_errors(&self) -> u32 {
        self.phase_metrics.values().map(|p| p.errors).sum()
    }

    /// Serialize to JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

impl Default for AnalysisMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for AnalysisMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Analysis Metrics:")?;
        writeln!(f, "  States explored:   {}", self.states_explored)?;
        writeln!(f, "  Transitions:       {}", self.transitions_explored)?;
        writeln!(f, "  Paths explored:    {}", self.paths_explored)?;
        writeln!(f, "  Paths merged:      {}", self.paths_merged)?;
        writeln!(f, "  Paths pruned:      {}", self.paths_pruned)?;
        writeln!(f, "  Solver calls:      {}", self.solver_calls)?;
        writeln!(f, "    SAT:   {}", self.solver_sat)?;
        writeln!(f, "    UNSAT: {}", self.solver_unsat)?;
        writeln!(f, "    UNK:   {}", self.solver_unknown)?;
        writeln!(f, "  Attacks found:     {}", self.attacks_found)?;
        writeln!(f, "  Total time:        {}ms", self.total_duration_ms())?;
        writeln!(f, "  Merge ratio:       {:.2}", self.merge_ratio())?;
        Ok(())
    }
}

impl PhaseMetrics {
    pub fn new(name: impl Into<String>) -> Self {
        PhaseMetrics {
            name: name.into(),
            duration_ms: 0,
            states_in: 0,
            states_out: 0,
            peak_memory_kb: 0,
            errors: 0,
        }
    }

    /// State reduction ratio (states_out / states_in).
    pub fn reduction_ratio(&self) -> f64 {
        if self.states_in == 0 {
            return 0.0;
        }
        self.states_out as f64 / self.states_in as f64
    }
}

impl fmt::Display for PhaseMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {}ms, {}->{} states, {}KB peak",
            self.name, self.duration_ms, self.states_in, self.states_out, self.peak_memory_kb
        )
    }
}

// ── Coverage Metrics ─────────────────────────────────────────────────────

/// Coverage metrics for the analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageMetrics {
    pub total_states: u64,
    pub explored_states: u64,
    pub total_transitions: u64,
    pub explored_transitions: u64,
    pub total_paths: u64,
    pub explored_paths: u64,
    pub cipher_suites_covered: u32,
    pub cipher_suites_total: u32,
    pub versions_covered: u32,
    pub versions_total: u32,
}

impl CoverageMetrics {
    pub fn new() -> Self {
        CoverageMetrics {
            total_states: 0,
            explored_states: 0,
            total_transitions: 0,
            explored_transitions: 0,
            total_paths: 0,
            explored_paths: 0,
            cipher_suites_covered: 0,
            cipher_suites_total: 0,
            versions_covered: 0,
            versions_total: 0,
        }
    }

    /// State coverage as a percentage.
    pub fn state_coverage_pct(&self) -> f64 {
        if self.total_states == 0 {
            return 0.0;
        }
        (self.explored_states as f64 / self.total_states as f64) * 100.0
    }

    /// Transition coverage as a percentage.
    pub fn transition_coverage_pct(&self) -> f64 {
        if self.total_transitions == 0 {
            return 0.0;
        }
        (self.explored_transitions as f64 / self.total_transitions as f64) * 100.0
    }

    /// Path coverage as a percentage.
    pub fn path_coverage_pct(&self) -> f64 {
        if self.total_paths == 0 {
            return 0.0;
        }
        (self.explored_paths as f64 / self.total_paths as f64) * 100.0
    }

    /// Cipher suite coverage as a percentage.
    pub fn cipher_coverage_pct(&self) -> f64 {
        if self.cipher_suites_total == 0 {
            return 0.0;
        }
        (self.cipher_suites_covered as f64 / self.cipher_suites_total as f64) * 100.0
    }

    /// Whether full state and path coverage was achieved.
    pub fn is_complete(&self) -> bool {
        self.state_coverage_pct() >= 100.0 && self.path_coverage_pct() >= 100.0
    }

    /// Overall coverage score (average of sub-coverages).
    pub fn overall_score(&self) -> f64 {
        let scores = [
            self.state_coverage_pct(),
            self.transition_coverage_pct(),
            self.path_coverage_pct(),
        ];
        let non_zero: Vec<f64> = scores.iter().filter(|&&s| s > 0.0).copied().collect();
        if non_zero.is_empty() {
            0.0
        } else {
            non_zero.iter().sum::<f64>() / non_zero.len() as f64
        }
    }
}

impl Default for CoverageMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for CoverageMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Coverage Metrics:")?;
        writeln!(f, "  State:      {}/{} ({:.1}%)", self.explored_states, self.total_states, self.state_coverage_pct())?;
        writeln!(f, "  Transition: {}/{} ({:.1}%)", self.explored_transitions, self.total_transitions, self.transition_coverage_pct())?;
        writeln!(f, "  Path:       {}/{} ({:.1}%)", self.explored_paths, self.total_paths, self.path_coverage_pct())?;
        writeln!(f, "  Ciphers:    {}/{} ({:.1}%)", self.cipher_suites_covered, self.cipher_suites_total, self.cipher_coverage_pct())?;
        writeln!(f, "  Overall:    {:.1}%", self.overall_score())?;
        Ok(())
    }
}

// ── Performance Metrics ──────────────────────────────────────────────────

/// Performance metrics including timing and resource usage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub wall_time_ms: u64,
    pub cpu_time_ms: u64,
    pub peak_memory_kb: u64,
    pub current_memory_kb: u64,
    pub disk_io_bytes: u64,
    pub phase_timings: HashMap<String, u64>,
}

impl PerformanceMetrics {
    pub fn new() -> Self {
        PerformanceMetrics {
            wall_time_ms: 0,
            cpu_time_ms: 0,
            peak_memory_kb: 0,
            current_memory_kb: 0,
            disk_io_bytes: 0,
            phase_timings: HashMap::new(),
        }
    }

    pub fn record_phase_time(&mut self, phase: impl Into<String>, duration_ms: u64) {
        self.phase_timings.insert(phase.into(), duration_ms);
    }

    pub fn update_memory(&mut self, current_kb: u64) {
        self.current_memory_kb = current_kb;
        if current_kb > self.peak_memory_kb {
            self.peak_memory_kb = current_kb;
        }
    }

    /// Throughput in states per second.
    pub fn states_per_second(&self, states: u64) -> f64 {
        if self.wall_time_ms == 0 {
            return 0.0;
        }
        states as f64 / (self.wall_time_ms as f64 / 1000.0)
    }

    /// Slowest phase.
    pub fn bottleneck_phase(&self) -> Option<(&str, u64)> {
        self.phase_timings
            .iter()
            .max_by_key(|(_, &v)| v)
            .map(|(k, &v)| (k.as_str(), v))
    }

    /// Total phase time.
    pub fn total_phase_time_ms(&self) -> u64 {
        self.phase_timings.values().sum()
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for PerformanceMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Performance Metrics:")?;
        writeln!(f, "  Wall time:    {}ms", self.wall_time_ms)?;
        writeln!(f, "  CPU time:     {}ms", self.cpu_time_ms)?;
        writeln!(f, "  Peak memory:  {}KB", self.peak_memory_kb)?;
        writeln!(f, "  Disk I/O:     {} bytes", self.disk_io_bytes)?;
        if let Some((phase, ms)) = self.bottleneck_phase() {
            writeln!(f, "  Bottleneck:   {} ({}ms)", phase, ms)?;
        }
        Ok(())
    }
}

// ── Merge Statistics ─────────────────────────────────────────────────────

/// Statistics specifically for the merge phase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeStatistics {
    pub merge_attempts: u64,
    pub merge_successes: u64,
    pub merge_failures: u64,
    pub merge_rejections: u64,
    pub widening_applications: u64,
    pub states_before_merge: u64,
    pub states_after_merge: u64,
    pub merge_point_stats: HashMap<String, MergePointStats>,
}

/// Statistics for a single merge point (program location).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergePointStats {
    pub address: u64,
    pub attempts: u32,
    pub successes: u32,
    pub max_states_before: u32,
    pub max_states_after: u32,
    pub avg_similarity: f64,
}

impl MergeStatistics {
    pub fn new() -> Self {
        MergeStatistics {
            merge_attempts: 0,
            merge_successes: 0,
            merge_failures: 0,
            merge_rejections: 0,
            widening_applications: 0,
            states_before_merge: 0,
            states_after_merge: 0,
            merge_point_stats: HashMap::new(),
        }
    }

    pub fn record_attempt(&mut self, success: bool) {
        self.merge_attempts += 1;
        if success {
            self.merge_successes += 1;
        } else {
            self.merge_failures += 1;
        }
    }

    pub fn record_rejection(&mut self) {
        self.merge_rejections += 1;
    }

    pub fn record_widening(&mut self) {
        self.widening_applications += 1;
    }

    /// Merge success rate.
    pub fn success_rate(&self) -> f64 {
        if self.merge_attempts == 0 {
            return 0.0;
        }
        self.merge_successes as f64 / self.merge_attempts as f64
    }

    /// State reduction ratio.
    pub fn reduction_ratio(&self) -> f64 {
        if self.states_before_merge == 0 {
            return 1.0;
        }
        self.states_after_merge as f64 / self.states_before_merge as f64
    }

    /// State reduction percentage (how much was eliminated).
    pub fn reduction_pct(&self) -> f64 {
        (1.0 - self.reduction_ratio()) * 100.0
    }

    pub fn record_merge_point(&mut self, addr: u64, success: bool, similarity: f64) {
        let key = format!("{:#x}", addr);
        let stats = self.merge_point_stats.entry(key).or_insert(MergePointStats {
            address: addr,
            attempts: 0,
            successes: 0,
            max_states_before: 0,
            max_states_after: 0,
            avg_similarity: 0.0,
        });
        let n = stats.attempts as f64;
        stats.attempts += 1;
        if success {
            stats.successes += 1;
        }
        // Running average
        stats.avg_similarity = (stats.avg_similarity * n + similarity) / (n + 1.0);
    }

    /// Most active merge point (by number of attempts).
    pub fn hottest_merge_point(&self) -> Option<(&str, &MergePointStats)> {
        self.merge_point_stats
            .iter()
            .max_by_key(|(_, s)| s.attempts)
            .map(|(k, v)| (k.as_str(), v))
    }
}

impl Default for MergeStatistics {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for MergeStatistics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Merge Statistics:")?;
        writeln!(f, "  Attempts:   {}", self.merge_attempts)?;
        writeln!(f, "  Successes:  {} ({:.1}%)", self.merge_successes, self.success_rate() * 100.0)?;
        writeln!(f, "  Failures:   {}", self.merge_failures)?;
        writeln!(f, "  Rejections: {}", self.merge_rejections)?;
        writeln!(f, "  Widenings:  {}", self.widening_applications)?;
        writeln!(f, "  Reduction:  {} → {} ({:.1}% eliminated)", self.states_before_merge, self.states_after_merge, self.reduction_pct())?;
        Ok(())
    }
}

// ── Metric Aggregation ───────────────────────────────────────────────────

/// Aggregate report combining all metric categories.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricReport {
    pub analysis: AnalysisMetrics,
    pub coverage: CoverageMetrics,
    pub performance: PerformanceMetrics,
    pub merge: MergeStatistics,
    pub timestamp: String,
}

impl MetricReport {
    pub fn new(
        analysis: AnalysisMetrics,
        coverage: CoverageMetrics,
        performance: PerformanceMetrics,
        merge: MergeStatistics,
    ) -> Self {
        MetricReport {
            analysis,
            coverage,
            performance,
            merge,
            timestamp: chrono::Utc::now().to_rfc3339(),
        }
    }

    /// Overall success: complete coverage and no errors.
    pub fn is_successful(&self) -> bool {
        self.analysis.total_errors() == 0 && self.coverage.overall_score() > 0.0
    }

    /// Serialize to JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize from JSON.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

impl fmt::Display for MetricReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== NegSynth Metric Report ===")?;
        writeln!(f, "Timestamp: {}", self.timestamp)?;
        writeln!(f)?;
        write!(f, "{}", self.analysis)?;
        writeln!(f)?;
        write!(f, "{}", self.coverage)?;
        writeln!(f)?;
        write!(f, "{}", self.performance)?;
        writeln!(f)?;
        write!(f, "{}", self.merge)?;
        Ok(())
    }
}

// ── Timer Utility ────────────────────────────────────────────────────────

/// A simple scoped timer for measuring phase durations.
pub struct PhaseTimer {
    name: String,
    start: Instant,
}

impl PhaseTimer {
    pub fn start(name: impl Into<String>) -> Self {
        PhaseTimer {
            name: name.into(),
            start: Instant::now(),
        }
    }

    pub fn elapsed_ms(&self) -> u64 {
        self.start.elapsed().as_millis() as u64
    }

    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    pub fn finish(self) -> PhaseMetrics {
        let elapsed = self.start.elapsed().as_millis() as u64;
        let name = self.name;
        let mut metrics = PhaseMetrics::new(name);
        metrics.duration_ms = elapsed;
        metrics
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analysis_metrics() {
        let mut m = AnalysisMetrics::new();
        m.record_state_explored();
        m.record_state_explored();
        m.record_path_explored();
        m.record_path_merged();
        m.record_solver_call(true, false);
        m.record_solver_call(false, false);
        m.record_solver_call(false, true);

        assert_eq!(m.states_explored, 2);
        assert_eq!(m.paths_explored, 1);
        assert_eq!(m.solver_calls, 3);
        assert_eq!(m.solver_sat, 1);
        assert_eq!(m.solver_unsat, 1);
        assert_eq!(m.solver_unknown, 1);
    }

    #[test]
    fn test_merge_ratio() {
        let mut m = AnalysisMetrics::new();
        for _ in 0..10 {
            m.record_path_explored();
        }
        for _ in 0..3 {
            m.record_path_merged();
        }
        assert!((m.merge_ratio() - 0.3).abs() < 0.001);
    }

    #[test]
    fn test_solver_success_rate() {
        let mut m = AnalysisMetrics::new();
        m.record_solver_call(true, false);
        m.record_solver_call(false, false);
        m.record_solver_call(false, true);
        assert!((m.solver_success_rate() - 2.0 / 3.0).abs() < 0.001);
    }

    #[test]
    fn test_coverage_metrics() {
        let mut c = CoverageMetrics::new();
        c.total_states = 100;
        c.explored_states = 75;
        c.total_transitions = 200;
        c.explored_transitions = 150;
        c.total_paths = 50;
        c.explored_paths = 50;

        assert!((c.state_coverage_pct() - 75.0).abs() < 0.001);
        assert!((c.transition_coverage_pct() - 75.0).abs() < 0.001);
        assert!((c.path_coverage_pct() - 100.0).abs() < 0.001);
        assert!(!c.is_complete());
    }

    #[test]
    fn test_complete_coverage() {
        let mut c = CoverageMetrics::new();
        c.total_states = 10;
        c.explored_states = 10;
        c.total_paths = 5;
        c.explored_paths = 5;
        assert!(c.is_complete());
    }

    #[test]
    fn test_performance_metrics() {
        let mut p = PerformanceMetrics::new();
        p.wall_time_ms = 5000;
        p.record_phase_time("slicer", 1000);
        p.record_phase_time("merge", 3000);
        p.record_phase_time("encoding", 1000);

        let (phase, _) = p.bottleneck_phase().unwrap();
        assert_eq!(phase, "merge");
        assert_eq!(p.total_phase_time_ms(), 5000);
    }

    #[test]
    fn test_performance_throughput() {
        let mut p = PerformanceMetrics::new();
        p.wall_time_ms = 2000;
        let throughput = p.states_per_second(1000);
        assert!((throughput - 500.0).abs() < 0.001);
    }

    #[test]
    fn test_memory_tracking() {
        let mut p = PerformanceMetrics::new();
        p.update_memory(100);
        p.update_memory(200);
        p.update_memory(150);
        assert_eq!(p.peak_memory_kb, 200);
        assert_eq!(p.current_memory_kb, 150);
    }

    #[test]
    fn test_merge_statistics() {
        let mut ms = MergeStatistics::new();
        ms.record_attempt(true);
        ms.record_attempt(true);
        ms.record_attempt(false);
        ms.record_rejection();

        assert_eq!(ms.merge_attempts, 3);
        assert_eq!(ms.merge_successes, 2);
        assert!((ms.success_rate() - 2.0 / 3.0).abs() < 0.001);
    }

    #[test]
    fn test_merge_reduction() {
        let mut ms = MergeStatistics::new();
        ms.states_before_merge = 1000;
        ms.states_after_merge = 250;
        assert!((ms.reduction_pct() - 75.0).abs() < 0.001);
    }

    #[test]
    fn test_merge_point_stats() {
        let mut ms = MergeStatistics::new();
        ms.record_merge_point(0x1000, true, 0.9);
        ms.record_merge_point(0x1000, true, 0.8);
        ms.record_merge_point(0x2000, false, 0.3);

        let (hot, stats) = ms.hottest_merge_point().unwrap();
        assert_eq!(hot, "0x1000");
        assert_eq!(stats.attempts, 2);
        assert!((stats.avg_similarity - 0.85).abs() < 0.001);
    }

    #[test]
    fn test_phase_metrics() {
        let pm = PhaseMetrics::new("slicer");
        assert_eq!(pm.name, "slicer");
        assert_eq!(pm.reduction_ratio(), 0.0);
    }

    #[test]
    fn test_metric_report() {
        let report = MetricReport::new(
            AnalysisMetrics::new(),
            CoverageMetrics::new(),
            PerformanceMetrics::new(),
            MergeStatistics::new(),
        );
        assert!(!report.is_successful()); // no coverage
        let json = report.to_json().unwrap();
        let deserialized = MetricReport::from_json(&json).unwrap();
        assert_eq!(deserialized.analysis.states_explored, 0);
    }

    #[test]
    fn test_phase_timer() {
        let timer = PhaseTimer::start("test");
        assert_eq!(timer.name(), "test");
        std::thread::sleep(std::time::Duration::from_millis(10));
        let metrics = timer.finish();
        assert!(metrics.duration_ms >= 5); // allow some tolerance
    }

    #[test]
    fn test_analysis_metrics_serialization() {
        let mut m = AnalysisMetrics::new();
        m.states_explored = 42;
        let json = m.to_json().unwrap();
        assert!(json.contains("42"));
    }

    #[test]
    fn test_display_implementations() {
        let m = AnalysisMetrics::new();
        let s = format!("{}", m);
        assert!(s.contains("Analysis Metrics"));

        let c = CoverageMetrics::new();
        let s = format!("{}", c);
        assert!(s.contains("Coverage Metrics"));

        let p = PerformanceMetrics::new();
        let s = format!("{}", p);
        assert!(s.contains("Performance Metrics"));

        let ms = MergeStatistics::new();
        let s = format!("{}", ms);
        assert!(s.contains("Merge Statistics"));
    }
}
