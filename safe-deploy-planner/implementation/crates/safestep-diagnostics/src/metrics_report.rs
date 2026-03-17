//! Performance metrics reporting for the SafeStep planner.
//!
//! Provides structured collection, formatting, and comparison of performance
//! metrics gathered during planning phases. Supports text (with Unicode bar
//! charts), JSON, and CSV output as well as baseline-vs-current benchmark
//! comparison.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that may arise during metrics formatting or comparison.
#[derive(Debug, thiserror::Error)]
pub enum MetricsError {
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("no phases recorded in report")]
    EmptyReport,
}

// ---------------------------------------------------------------------------
// PhaseMetrics
// ---------------------------------------------------------------------------

/// Timing and detail information for a single planning phase.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PhaseMetrics {
    pub name: String,
    pub duration_ms: u64,
    pub percentage: f64,
    pub details: HashMap<String, String>,
}

impl PhaseMetrics {
    /// Create a new `PhaseMetrics` with the given name and duration.
    /// The percentage field is initialised to `0.0` and will be recomputed by
    /// [`MetricsReport::recompute_percentages`].
    pub fn new(name: &str, duration_ms: u64) -> Self {
        Self {
            name: name.to_string(),
            duration_ms,
            percentage: 0.0,
            details: HashMap::new(),
        }
    }

    /// Builder-style helper to attach an arbitrary key/value detail.
    pub fn with_detail(mut self, key: &str, value: &str) -> Self {
        self.details.insert(key.to_string(), value.to_string());
        self
    }
}

// ---------------------------------------------------------------------------
// TotalMetrics
// ---------------------------------------------------------------------------

/// Aggregate resource-usage numbers for an entire planning run.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TotalMetrics {
    pub total_duration_ms: u64,
    pub peak_memory_bytes: u64,
    pub variables_used: u64,
    pub clauses_generated: u64,
}

impl Default for TotalMetrics {
    fn default() -> Self {
        Self {
            total_duration_ms: 0,
            peak_memory_bytes: 0,
            variables_used: 0,
            clauses_generated: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// SolverStats
// ---------------------------------------------------------------------------

/// Low-level statistics exported by the SAT / constraint solver.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SolverStats {
    pub decisions: u64,
    pub propagations: u64,
    pub conflicts: u64,
    pub restarts: u64,
    pub learned_clauses: u64,
}

impl Default for SolverStats {
    fn default() -> Self {
        Self {
            decisions: 0,
            propagations: 0,
            conflicts: 0,
            restarts: 0,
            learned_clauses: 0,
        }
    }
}

impl SolverStats {
    /// Ratio of conflicts to decisions.  Returns `0.0` when there are no
    /// decisions.
    pub fn conflicts_per_decision(&self) -> f64 {
        if self.decisions == 0 {
            return 0.0;
        }
        self.conflicts as f64 / self.decisions as f64
    }

    /// Ratio of propagations to decisions.  Returns `0.0` when there are no
    /// decisions.
    pub fn propagations_per_decision(&self) -> f64 {
        if self.decisions == 0 {
            return 0.0;
        }
        self.propagations as f64 / self.decisions as f64
    }
}

// ---------------------------------------------------------------------------
// MetricsReport
// ---------------------------------------------------------------------------

/// A complete performance report for a planning run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsReport {
    pub phases: Vec<PhaseMetrics>,
    pub totals: TotalMetrics,
    pub solver_stats: SolverStats,
}

impl MetricsReport {
    /// Create an empty report with default totals and solver stats.
    pub fn new() -> Self {
        Self {
            phases: Vec::new(),
            totals: TotalMetrics::default(),
            solver_stats: SolverStats::default(),
        }
    }

    /// Append a phase and recompute all percentage values.
    pub fn add_phase(&mut self, phase: PhaseMetrics) {
        self.totals.total_duration_ms += phase.duration_ms;
        self.phases.push(phase);
        self.recompute_percentages();
    }

    /// Recalculate every phase's `percentage` field based on
    /// `totals.total_duration_ms`.
    pub fn recompute_percentages(&mut self) {
        let total = self.totals.total_duration_ms;
        for phase in &mut self.phases {
            if total == 0 {
                phase.percentage = 0.0;
            } else {
                phase.percentage = phase.duration_ms as f64 / total as f64 * 100.0;
            }
        }
    }

    /// The phase with the largest duration, or `None` if there are no phases.
    pub fn slowest_phase(&self) -> Option<&PhaseMetrics> {
        self.phases.iter().max_by_key(|p| p.duration_ms)
    }

    /// The phase with the smallest duration, or `None` if there are no phases.
    pub fn fastest_phase(&self) -> Option<&PhaseMetrics> {
        self.phases.iter().min_by_key(|p| p.duration_ms)
    }

    /// The number of recorded phases.
    pub fn phase_count(&self) -> usize {
        self.phases.len()
    }
}

impl Default for MetricsReport {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// PerformanceBreakdown
// ---------------------------------------------------------------------------

/// A human-oriented summary derived from a [`MetricsReport`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBreakdown {
    pub phases: Vec<(String, f64)>,
    pub bottleneck: String,
    pub suggestions: Vec<String>,
}

impl PerformanceBreakdown {
    /// Build a breakdown from an existing report.
    ///
    /// * `phases` — list of `(name, percentage)` pairs.
    /// * `bottleneck` — the name of the slowest phase (or `"none"` if no
    ///   phases exist).
    /// * `suggestions` — automatically generated hints derived from the phase
    ///   profile and solver statistics.
    pub fn from_report(report: &MetricsReport) -> Self {
        let phases: Vec<(String, f64)> = report
            .phases
            .iter()
            .map(|p| (p.name.clone(), p.percentage))
            .collect();

        let bottleneck = report
            .slowest_phase()
            .map(|p| p.name.clone())
            .unwrap_or_else(|| "none".to_string());

        let mut suggestions: Vec<String> = Vec::new();

        // Suggestion: dominant phase
        if let Some(slowest) = report.slowest_phase() {
            if slowest.percentage > 60.0 {
                suggestions.push(format!(
                    "Phase '{}' dominates at {:.1}% — consider breaking it into sub-phases or optimising it.",
                    slowest.name, slowest.percentage
                ));
            }
            if slowest.percentage > 40.0 && slowest.percentage <= 60.0 {
                suggestions.push(format!(
                    "Phase '{}' is the primary bottleneck at {:.1}%.",
                    slowest.name, slowest.percentage
                ));
            }
        }

        // Suggestion: high conflict ratio
        let cpd = report.solver_stats.conflicts_per_decision();
        if cpd > 0.5 {
            suggestions.push(format!(
                "High conflict-to-decision ratio ({:.2}) — consider improving variable ordering or adding learned-clause management.",
                cpd
            ));
        }

        // Suggestion: many restarts
        if report.solver_stats.restarts > 100 {
            suggestions.push(format!(
                "Solver restarted {} times — review restart policy.",
                report.solver_stats.restarts
            ));
        }

        // Suggestion: large clause set
        if report.totals.clauses_generated > 100_000 {
            suggestions.push(
                "Over 100 000 clauses generated — consider clause minimisation or pre-processing.".to_string(),
            );
        }

        // Suggestion: high memory
        if report.totals.peak_memory_bytes > 1_073_741_824 {
            suggestions.push(
                "Peak memory exceeds 1 GiB — look for opportunities to reduce working-set size.".to_string(),
            );
        }

        // Fallback: if nothing specific, give a generic positive note.
        if suggestions.is_empty() {
            suggestions.push("No specific issues detected — performance looks healthy.".to_string());
        }

        Self {
            phases,
            bottleneck,
            suggestions,
        }
    }
}

// ---------------------------------------------------------------------------
// MetricsFormatter
// ---------------------------------------------------------------------------

/// Renders a [`MetricsReport`] to various output formats.
pub struct MetricsFormatter;

/// Unicode block characters ordered by visual width.
const BAR_CHARS: [char; 8] = ['▏', '▎', '▍', '▌', '▋', '▊', '▉', '█'];

impl MetricsFormatter {
    // ------------------------------------------------------------------
    // Text
    // ------------------------------------------------------------------

    /// Render a human-readable text report, including Unicode bar charts for
    /// phase durations.
    pub fn format_text(report: &MetricsReport) -> String {
        let mut out = String::new();
        out.push_str("╔══════════════════════════════════════════╗\n");
        out.push_str("║       SafeStep Performance Report       ║\n");
        out.push_str("╚══════════════════════════════════════════╝\n\n");

        // --- Totals ---
        out.push_str(&format!(
            "Total duration : {} ms\n",
            report.totals.total_duration_ms
        ));
        out.push_str(&format!(
            "Peak memory    : {} bytes\n",
            report.totals.peak_memory_bytes
        ));
        out.push_str(&format!(
            "Variables      : {}\n",
            report.totals.variables_used
        ));
        out.push_str(&format!(
            "Clauses        : {}\n\n",
            report.totals.clauses_generated
        ));

        // --- Phases with bar chart ---
        if !report.phases.is_empty() {
            out.push_str("Phase Breakdown\n");
            out.push_str("───────────────\n");

            let max_name_len = report
                .phases
                .iter()
                .map(|p| p.name.len())
                .max()
                .unwrap_or(0);
            let bar_max_width: usize = 30;

            let max_duration = report
                .phases
                .iter()
                .map(|p| p.duration_ms)
                .max()
                .unwrap_or(1)
                .max(1);

            for phase in &report.phases {
                let bar = Self::render_bar(phase.duration_ms, max_duration, bar_max_width);
                out.push_str(&format!(
                    "  {:<width$}  {:>6} ms ({:>5.1}%)  {}\n",
                    phase.name,
                    phase.duration_ms,
                    phase.percentage,
                    bar,
                    width = max_name_len
                ));
            }
            out.push('\n');
        }

        // --- Solver stats ---
        let ss = &report.solver_stats;
        out.push_str("Solver Statistics\n");
        out.push_str("─────────────────\n");
        out.push_str(&format!("  Decisions       : {}\n", ss.decisions));
        out.push_str(&format!("  Propagations    : {}\n", ss.propagations));
        out.push_str(&format!("  Conflicts       : {}\n", ss.conflicts));
        out.push_str(&format!("  Restarts        : {}\n", ss.restarts));
        out.push_str(&format!("  Learned clauses : {}\n", ss.learned_clauses));
        out.push_str(&format!(
            "  Conflicts/dec   : {:.4}\n",
            ss.conflicts_per_decision()
        ));
        out.push_str(&format!(
            "  Propags/dec     : {:.4}\n",
            ss.propagations_per_decision()
        ));

        out
    }

    /// Build a Unicode bar of approximate length proportional to `value / max`.
    fn render_bar(value: u64, max: u64, width: usize) -> String {
        if max == 0 || width == 0 {
            return String::new();
        }
        let fraction = value as f64 / max as f64;
        let total_eighths = (fraction * (width as f64) * 8.0).round() as usize;
        let full_blocks = total_eighths / 8;
        let remainder = total_eighths % 8;

        let mut bar = String::new();
        for _ in 0..full_blocks {
            bar.push('█');
        }
        if remainder > 0 {
            bar.push(BAR_CHARS[remainder - 1]);
        }
        bar
    }

    // ------------------------------------------------------------------
    // JSON
    // ------------------------------------------------------------------

    /// Pretty-printed JSON via `serde_json`.
    pub fn format_json(report: &MetricsReport) -> String {
        serde_json::to_string_pretty(report).unwrap_or_else(|e| format!("{{\"error\":\"{}\"}}", e))
    }

    // ------------------------------------------------------------------
    // CSV
    // ------------------------------------------------------------------

    /// CSV with a header row followed by one row per phase.
    pub fn format_csv(report: &MetricsReport) -> String {
        let mut out = String::from("phase,duration_ms,percentage\n");
        for phase in &report.phases {
            out.push_str(&format!(
                "{},{},{:.2}\n",
                Self::csv_escape(&phase.name),
                phase.duration_ms,
                phase.percentage
            ));
        }
        out
    }

    /// Minimal CSV escaping: wrap in quotes if the value contains a comma,
    /// quote, or newline.
    fn csv_escape(value: &str) -> String {
        if value.contains(',') || value.contains('"') || value.contains('\n') {
            let escaped = value.replace('"', "\"\"");
            format!("\"{}\"", escaped)
        } else {
            value.to_string()
        }
    }
}

// ---------------------------------------------------------------------------
// ComparisonResult
// ---------------------------------------------------------------------------

/// Outcome of comparing two [`MetricsReport`]s.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonResult {
    pub improvements: Vec<String>,
    pub regressions: Vec<String>,
    pub unchanged: Vec<String>,
    pub summary: String,
}

impl ComparisonResult {
    /// `true` when at least one regression was detected.
    pub fn has_regressions(&self) -> bool {
        !self.regressions.is_empty()
    }
}

// ---------------------------------------------------------------------------
// BenchmarkComparison
// ---------------------------------------------------------------------------

/// Compares a *current* report against a *baseline* report.
pub struct BenchmarkComparison;

/// Threshold (in percent) below which a change is considered "unchanged".
const CHANGE_THRESHOLD_PCT: f64 = 5.0;

impl BenchmarkComparison {
    /// Compare two reports phase-by-phase.
    ///
    /// Matching is done by phase name.  A phase present in only one report is
    /// reported as an improvement (if removed from current) or regression (if
    /// new in current), but only when compared against a non-empty baseline.
    pub fn compare(current: &MetricsReport, baseline: &MetricsReport) -> ComparisonResult {
        let mut improvements: Vec<String> = Vec::new();
        let mut regressions: Vec<String> = Vec::new();
        let mut unchanged: Vec<String> = Vec::new();

        let baseline_map: HashMap<&str, &PhaseMetrics> = baseline
            .phases
            .iter()
            .map(|p| (p.name.as_str(), p))
            .collect();

        let current_map: HashMap<&str, &PhaseMetrics> = current
            .phases
            .iter()
            .map(|p| (p.name.as_str(), p))
            .collect();

        // Walk current phases.
        for phase in &current.phases {
            if let Some(base_phase) = baseline_map.get(phase.name.as_str()) {
                if base_phase.duration_ms == 0 && phase.duration_ms == 0 {
                    unchanged.push(format!("{}: 0 ms → 0 ms (no change)", phase.name));
                    continue;
                }
                let base_dur = base_phase.duration_ms.max(1) as f64;
                let change_pct =
                    (phase.duration_ms as f64 - base_dur) / base_dur * 100.0;

                if change_pct < -CHANGE_THRESHOLD_PCT {
                    improvements.push(format!(
                        "{}: {} ms → {} ms ({:.1}%)",
                        phase.name, base_phase.duration_ms, phase.duration_ms, change_pct
                    ));
                } else if change_pct > CHANGE_THRESHOLD_PCT {
                    regressions.push(format!(
                        "{}: {} ms → {} ms (+{:.1}%)",
                        phase.name, base_phase.duration_ms, phase.duration_ms, change_pct
                    ));
                } else {
                    unchanged.push(format!(
                        "{}: {} ms → {} ms ({:+.1}%)",
                        phase.name, base_phase.duration_ms, phase.duration_ms, change_pct
                    ));
                }
            } else {
                // New phase in current that was not in baseline.
                regressions.push(format!(
                    "{}: new phase ({} ms)",
                    phase.name, phase.duration_ms
                ));
            }
        }

        // Phases in baseline but not in current (removed → improvement).
        for phase in &baseline.phases {
            if !current_map.contains_key(phase.name.as_str()) {
                improvements.push(format!(
                    "{}: removed (was {} ms)",
                    phase.name, phase.duration_ms
                ));
            }
        }

        // Total duration comparison.
        let total_base = baseline.totals.total_duration_ms.max(1) as f64;
        let total_change = (current.totals.total_duration_ms as f64 - total_base) / total_base * 100.0;

        let summary = if regressions.is_empty() && improvements.is_empty() {
            "No significant changes detected.".to_string()
        } else if regressions.is_empty() {
            format!(
                "All changes are improvements. Overall duration changed by {:.1}%.",
                total_change
            )
        } else if improvements.is_empty() {
            format!(
                "All changes are regressions. Overall duration changed by +{:.1}%.",
                total_change.abs()
            )
        } else {
            format!(
                "{} improvement(s), {} regression(s). Overall duration changed by {:+.1}%.",
                improvements.len(),
                regressions.len(),
                total_change
            )
        };

        ComparisonResult {
            improvements,
            regressions,
            unchanged,
            summary,
        }
    }
}

// ---------------------------------------------------------------------------
// Sparkline
// ---------------------------------------------------------------------------

/// Renders a sequence of values as an inline sparkline string.
pub struct Sparkline;

/// Characters used for the eight quantisation levels in a sparkline.
const SPARK_CHARS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

impl Sparkline {
    /// Map each value to one of [`SPARK_CHARS`] based on its position within
    /// the range `[min, max]` of the input slice.  An empty slice yields an
    /// empty string.
    pub fn render(values: &[f64]) -> String {
        if values.is_empty() {
            return String::new();
        }

        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max - min;

        values
            .iter()
            .map(|&v| {
                if range == 0.0 {
                    // All values identical — use the middle character.
                    SPARK_CHARS[3]
                } else {
                    let normalized = (v - min) / range; // 0.0 ..= 1.0
                    let idx = ((normalized * 7.0).round() as usize).min(7);
                    SPARK_CHARS[idx]
                }
            })
            .collect()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ────────────────────────────────────────────────────────

    fn sample_report() -> MetricsReport {
        let mut report = MetricsReport::new();
        report.totals.peak_memory_bytes = 512_000;
        report.totals.variables_used = 42;
        report.totals.clauses_generated = 200;

        report.add_phase(
            PhaseMetrics::new("parsing", 100)
                .with_detail("file_count", "3"),
        );
        report.add_phase(PhaseMetrics::new("encoding", 300));
        report.add_phase(PhaseMetrics::new("solving", 600));

        report.solver_stats = SolverStats {
            decisions: 1000,
            propagations: 5000,
            conflicts: 200,
            restarts: 10,
            learned_clauses: 150,
        };
        report
    }

    // ── PhaseMetrics ──────────────────────────────────────────────────

    #[test]
    fn phase_metrics_new() {
        let pm = PhaseMetrics::new("test", 42);
        assert_eq!(pm.name, "test");
        assert_eq!(pm.duration_ms, 42);
        assert_eq!(pm.percentage, 0.0);
        assert!(pm.details.is_empty());
    }

    #[test]
    fn phase_metrics_with_detail() {
        let pm = PhaseMetrics::new("a", 1)
            .with_detail("k1", "v1")
            .with_detail("k2", "v2");
        assert_eq!(pm.details.len(), 2);
        assert_eq!(pm.details["k1"], "v1");
        assert_eq!(pm.details["k2"], "v2");
    }

    // ── SolverStats ───────────────────────────────────────────────────

    #[test]
    fn solver_stats_zero_decisions() {
        let ss = SolverStats::default();
        assert_eq!(ss.conflicts_per_decision(), 0.0);
        assert_eq!(ss.propagations_per_decision(), 0.0);
    }

    #[test]
    fn solver_stats_ratios() {
        let ss = SolverStats {
            decisions: 200,
            propagations: 1000,
            conflicts: 50,
            restarts: 3,
            learned_clauses: 40,
        };
        assert!((ss.conflicts_per_decision() - 0.25).abs() < 1e-9);
        assert!((ss.propagations_per_decision() - 5.0).abs() < 1e-9);
    }

    // ── MetricsReport ─────────────────────────────────────────────────

    #[test]
    fn report_new_is_empty() {
        let r = MetricsReport::new();
        assert_eq!(r.phase_count(), 0);
        assert!(r.slowest_phase().is_none());
        assert!(r.fastest_phase().is_none());
    }

    #[test]
    fn add_phase_updates_total_and_percentages() {
        let mut r = MetricsReport::new();
        r.add_phase(PhaseMetrics::new("a", 300));
        r.add_phase(PhaseMetrics::new("b", 700));

        assert_eq!(r.totals.total_duration_ms, 1000);
        assert!((r.phases[0].percentage - 30.0).abs() < 1e-9);
        assert!((r.phases[1].percentage - 70.0).abs() < 1e-9);
    }

    #[test]
    fn recompute_percentages_zero_total() {
        let mut r = MetricsReport::new();
        r.phases.push(PhaseMetrics::new("x", 100));
        // total is still 0 because we pushed directly
        r.recompute_percentages();
        assert_eq!(r.phases[0].percentage, 0.0);
    }

    #[test]
    fn slowest_and_fastest() {
        let r = sample_report();
        assert_eq!(r.slowest_phase().unwrap().name, "solving");
        assert_eq!(r.fastest_phase().unwrap().name, "parsing");
    }

    #[test]
    fn phase_count() {
        let r = sample_report();
        assert_eq!(r.phase_count(), 3);
    }

    #[test]
    fn single_phase_percentages() {
        let mut r = MetricsReport::new();
        r.add_phase(PhaseMetrics::new("only", 500));
        assert!((r.phases[0].percentage - 100.0).abs() < 1e-9);
    }

    // ── PerformanceBreakdown ──────────────────────────────────────────

    #[test]
    fn breakdown_from_report() {
        let r = sample_report();
        let bd = PerformanceBreakdown::from_report(&r);

        assert_eq!(bd.bottleneck, "solving");
        assert_eq!(bd.phases.len(), 3);
        assert!(!bd.suggestions.is_empty());
    }

    #[test]
    fn breakdown_empty_report() {
        let r = MetricsReport::new();
        let bd = PerformanceBreakdown::from_report(&r);
        assert_eq!(bd.bottleneck, "none");
        assert!(!bd.suggestions.is_empty()); // healthy fallback
    }

    #[test]
    fn breakdown_dominant_phase_suggestion() {
        let mut r = MetricsReport::new();
        r.add_phase(PhaseMetrics::new("heavy", 900));
        r.add_phase(PhaseMetrics::new("light", 100));

        let bd = PerformanceBreakdown::from_report(&r);
        assert!(bd
            .suggestions
            .iter()
            .any(|s| s.contains("dominates") && s.contains("heavy")));
    }

    #[test]
    fn breakdown_high_conflict_ratio() {
        let mut r = MetricsReport::new();
        r.add_phase(PhaseMetrics::new("x", 100));
        r.solver_stats = SolverStats {
            decisions: 100,
            propagations: 500,
            conflicts: 80,
            restarts: 5,
            learned_clauses: 70,
        };

        let bd = PerformanceBreakdown::from_report(&r);
        assert!(bd
            .suggestions
            .iter()
            .any(|s| s.contains("conflict-to-decision")));
    }

    // ── MetricsFormatter: text ────────────────────────────────────────

    #[test]
    fn format_text_contains_header() {
        let text = MetricsFormatter::format_text(&sample_report());
        assert!(text.contains("SafeStep Performance Report"));
    }

    #[test]
    fn format_text_contains_phases() {
        let text = MetricsFormatter::format_text(&sample_report());
        assert!(text.contains("parsing"));
        assert!(text.contains("encoding"));
        assert!(text.contains("solving"));
    }

    #[test]
    fn format_text_bar_chart_characters() {
        let text = MetricsFormatter::format_text(&sample_report());
        // The slowest phase should get a full-width bar with at least one █
        assert!(text.contains('█'));
    }

    #[test]
    fn format_text_solver_stats() {
        let text = MetricsFormatter::format_text(&sample_report());
        assert!(text.contains("Decisions"));
        assert!(text.contains("Propagations"));
        assert!(text.contains("Conflicts"));
    }

    #[test]
    fn format_text_empty_report() {
        let r = MetricsReport::new();
        let text = MetricsFormatter::format_text(&r);
        assert!(text.contains("Total duration"));
        // Should not panic and should still contain solver stats section.
        assert!(text.contains("Solver Statistics"));
    }

    // ── MetricsFormatter: JSON ────────────────────────────────────────

    #[test]
    fn format_json_roundtrip() {
        let r = sample_report();
        let json_str = MetricsFormatter::format_json(&r);
        let parsed: MetricsReport = serde_json::from_str(&json_str).unwrap();
        assert_eq!(parsed.phases.len(), r.phases.len());
        assert_eq!(parsed.totals.total_duration_ms, r.totals.total_duration_ms);
    }

    #[test]
    fn format_json_is_pretty() {
        let json_str = MetricsFormatter::format_json(&sample_report());
        // Pretty JSON contains newlines and indentation.
        assert!(json_str.contains('\n'));
        assert!(json_str.contains("  "));
    }

    // ── MetricsFormatter: CSV ─────────────────────────────────────────

    #[test]
    fn format_csv_header() {
        let csv = MetricsFormatter::format_csv(&sample_report());
        let first_line = csv.lines().next().unwrap();
        assert_eq!(first_line, "phase,duration_ms,percentage");
    }

    #[test]
    fn format_csv_row_count() {
        let csv = MetricsFormatter::format_csv(&sample_report());
        // 1 header + 3 data rows
        assert_eq!(csv.lines().count(), 4);
    }

    #[test]
    fn format_csv_escaping() {
        let mut r = MetricsReport::new();
        r.add_phase(PhaseMetrics::new("phase,with,commas", 100));
        let csv = MetricsFormatter::format_csv(&r);
        assert!(csv.contains("\"phase,with,commas\""));
    }

    #[test]
    fn format_csv_empty_report() {
        let r = MetricsReport::new();
        let csv = MetricsFormatter::format_csv(&r);
        assert_eq!(csv.lines().count(), 1); // header only
    }

    // ── BenchmarkComparison ───────────────────────────────────────────

    #[test]
    fn comparison_no_changes() {
        let r = sample_report();
        let result = BenchmarkComparison::compare(&r, &r);
        assert!(!result.has_regressions());
        assert!(result.improvements.is_empty());
        // All phases should be "unchanged".
        assert_eq!(result.unchanged.len(), 3);
    }

    #[test]
    fn comparison_detects_regression() {
        let baseline = sample_report();
        let mut current = sample_report();
        // Double the solving time → regression
        current.phases[2].duration_ms = 1200;
        current.totals.total_duration_ms = 100 + 300 + 1200;
        current.recompute_percentages();

        let result = BenchmarkComparison::compare(&current, &baseline);
        assert!(result.has_regressions());
        assert!(result
            .regressions
            .iter()
            .any(|r| r.contains("solving")));
    }

    #[test]
    fn comparison_detects_improvement() {
        let baseline = sample_report();
        let mut current = sample_report();
        // Halve the solving time → improvement
        current.phases[2].duration_ms = 300;
        current.totals.total_duration_ms = 100 + 300 + 300;
        current.recompute_percentages();

        let result = BenchmarkComparison::compare(&current, &baseline);
        assert!(result
            .improvements
            .iter()
            .any(|s| s.contains("solving")));
    }

    #[test]
    fn comparison_new_phase_is_regression() {
        let baseline = sample_report();
        let mut current = sample_report();
        current.add_phase(PhaseMetrics::new("extra", 200));

        let result = BenchmarkComparison::compare(&current, &baseline);
        assert!(result
            .regressions
            .iter()
            .any(|r| r.contains("extra") && r.contains("new phase")));
    }

    #[test]
    fn comparison_removed_phase_is_improvement() {
        let baseline = sample_report();
        let mut current = MetricsReport::new();
        // Only keep encoding + solving, skip parsing.
        current.add_phase(PhaseMetrics::new("encoding", 300));
        current.add_phase(PhaseMetrics::new("solving", 600));

        let result = BenchmarkComparison::compare(&current, &baseline);
        assert!(result
            .improvements
            .iter()
            .any(|s| s.contains("parsing") && s.contains("removed")));
    }

    #[test]
    fn comparison_empty_reports() {
        let a = MetricsReport::new();
        let b = MetricsReport::new();
        let result = BenchmarkComparison::compare(&a, &b);
        assert!(!result.has_regressions());
        assert_eq!(result.summary, "No significant changes detected.");
    }

    #[test]
    fn comparison_summary_all_improvements() {
        let baseline = sample_report();
        let mut current = sample_report();
        for p in &mut current.phases {
            p.duration_ms /= 2;
        }
        current.totals.total_duration_ms = current.phases.iter().map(|p| p.duration_ms).sum();
        current.recompute_percentages();

        let result = BenchmarkComparison::compare(&current, &baseline);
        assert!(!result.has_regressions());
        assert!(result.summary.contains("improvements"));
    }

    // ── Sparkline ─────────────────────────────────────────────────────

    #[test]
    fn sparkline_empty() {
        assert_eq!(Sparkline::render(&[]), "");
    }

    #[test]
    fn sparkline_single_value() {
        let s = Sparkline::render(&[42.0]);
        assert_eq!(s.chars().count(), 1);
        // Single value → all identical → middle char.
        assert_eq!(s, "▄");
    }

    #[test]
    fn sparkline_monotonic_increase() {
        let vals: Vec<f64> = (0..8).map(|i| i as f64).collect();
        let s = Sparkline::render(&vals);
        assert_eq!(s.chars().count(), 8);
        // First char should be lowest, last should be highest.
        let chars: Vec<char> = s.chars().collect();
        assert_eq!(chars[0], '▁');
        assert_eq!(chars[7], '█');
    }

    #[test]
    fn sparkline_all_same() {
        let s = Sparkline::render(&[5.0, 5.0, 5.0]);
        // All identical → middle char repeated.
        assert_eq!(s, "▄▄▄");
    }

    #[test]
    fn sparkline_two_values() {
        let s = Sparkline::render(&[0.0, 100.0]);
        let chars: Vec<char> = s.chars().collect();
        assert_eq!(chars[0], '▁');
        assert_eq!(chars[1], '█');
    }

    // ── Render bar helper ─────────────────────────────────────────────

    #[test]
    fn render_bar_full() {
        let bar = MetricsFormatter::render_bar(100, 100, 10);
        // Should be all full blocks.
        assert!(bar.chars().all(|c| c == '█'));
        assert_eq!(bar.chars().count(), 10);
    }

    #[test]
    fn render_bar_zero() {
        let bar = MetricsFormatter::render_bar(0, 100, 10);
        assert!(bar.is_empty());
    }

    #[test]
    fn render_bar_partial() {
        let bar = MetricsFormatter::render_bar(50, 100, 10);
        // Should be roughly 5 blocks — the exact rendering depends on rounding.
        let count = bar.chars().count();
        assert!(count >= 4 && count <= 6, "bar length was {}", count);
    }

    #[test]
    fn render_bar_zero_max() {
        let bar = MetricsFormatter::render_bar(50, 0, 10);
        assert!(bar.is_empty());
    }

    #[test]
    fn render_bar_zero_width() {
        let bar = MetricsFormatter::render_bar(50, 100, 0);
        assert!(bar.is_empty());
    }
}
