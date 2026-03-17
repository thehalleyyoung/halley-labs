//! Benchmark report generation.
//!
//! Generates text, JSON, and CSV reports from benchmark results and metrics.
//! Includes comparison tables and performance-chart data exports.

use std::collections::HashMap;
use std::fmt::Write as FmtWrite;

use crate::harness::{BenchmarkResult, Comparison, Statistics};
use crate::metrics::{AnalysisMetrics, ComparisonMetrics, MetricsCollector};

// ---------------------------------------------------------------------------
// Report data model
// ---------------------------------------------------------------------------

/// Top-level benchmark report.
#[derive(Debug, Clone)]
pub struct BenchmarkReport {
    pub title: String,
    pub timestamp: u64,
    pub sections: Vec<ReportSection>,
    pub metadata: HashMap<String, String>,
}

impl BenchmarkReport {
    pub fn new(title: impl Into<String>) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Self {
            title: title.into(),
            timestamp: now,
            sections: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    pub fn add_section(&mut self, section: ReportSection) {
        self.sections.push(section);
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Build a report from a set of benchmark results.
    pub fn from_results(title: &str, results: &[BenchmarkResult]) -> Self {
        let mut report = Self::new(title);

        // Summary section
        let mut summary = ReportSection::new("Summary");
        summary.add_entry("experiment_count", &results.len().to_string());
        let total_iters: usize = results.iter().map(|r| r.iteration_count()).sum();
        summary.add_entry("total_iterations", &total_iters.to_string());
        let total_time: f64 = results.iter().map(|r| r.total_duration.as_secs_f64()).sum();
        summary.add_entry("total_time_seconds", &format!("{:.3}", total_time));
        report.add_section(summary);

        // Per-experiment sections
        for result in results {
            let mut sec = ReportSection::new(&result.config.name);
            sec.add_entry("iterations", &result.iteration_count().to_string());
            sec.add_entry("mean_seconds", &format!("{:.6}", result.stats.mean));
            sec.add_entry("median_seconds", &format!("{:.6}", result.stats.median));
            sec.add_entry("std_dev", &format!("{:.6}", result.stats.std_dev));
            sec.add_entry("min_seconds", &format!("{:.6}", result.stats.min));
            sec.add_entry("max_seconds", &format!("{:.6}", result.stats.max));
            sec.add_entry("p95_seconds", &format!("{:.6}", result.stats.p95));
            sec.add_entry("p99_seconds", &format!("{:.6}", result.stats.p99));
            sec.add_entry("cv", &format!("{:.4}", result.stats.coefficient_of_variation()));
            sec.add_entry("throughput_per_sec", &format!("{:.2}", result.throughput_per_second()));

            for (k, v) in &result.config.parameters {
                sec.add_entry(&format!("param_{}", k), v);
            }
            report.add_section(sec);
        }

        report
    }

    /// Build a report section from a comparison.
    pub fn comparison_section(cmp: &Comparison) -> ReportSection {
        let mut sec = ReportSection::new(&format!("{} vs {}", cmp.baseline_name, cmp.candidate_name));
        sec.add_entry("speedup", &format!("{:.3}x", cmp.speedup));
        sec.add_entry("percent_change", &format!("{:.2}%", cmp.percent_change()));
        sec.add_entry("mean_diff_seconds", &format!("{:.6}", cmp.mean_diff_seconds));
        sec.add_entry("baseline_mean", &format!("{:.6}", cmp.baseline_stats.mean));
        sec.add_entry("candidate_mean", &format!("{:.6}", cmp.candidate_stats.mean));
        sec.add_entry("faster", &cmp.is_faster().to_string());
        sec
    }

    /// Build a report section from analysis metrics.
    pub fn metrics_section(label: &str, m: &AnalysisMetrics) -> ReportSection {
        let mut sec = ReportSection::new(label);
        sec.add_entry("analysis_time_ms", &format!("{:.3}", m.analysis_time.as_secs_f64() * 1000.0));
        sec.add_entry("peak_memory_mb", &format!("{:.2}", m.peak_memory_bytes as f64 / (1024.0 * 1024.0)));
        sec.add_entry("smt_calls", &m.smt_calls.to_string());
        sec.add_entry("smt_time_ms", &format!("{:.3}", m.smt_time.as_secs_f64() * 1000.0));
        sec.add_entry("anomalies_found", &m.anomalies_found.to_string());
        sec.add_entry("constraint_count", &m.constraint_count.to_string());
        sec.add_entry("variable_count", &m.variable_count.to_string());
        sec.add_entry("dependency_count", &m.dependency_count.to_string());
        sec.add_entry("smt_time_fraction", &format!("{:.4}", m.smt_time_fraction()));
        for (k, v) in &m.extra {
            sec.add_entry(k, &format!("{:.4}", v));
        }
        sec
    }
}

#[derive(Debug, Clone)]
pub struct ReportSection {
    pub name: String,
    pub entries: Vec<(String, String)>,
}

impl ReportSection {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), entries: Vec::new() }
    }

    pub fn add_entry(&mut self, key: &str, value: &str) {
        self.entries.push((key.to_string(), value.to_string()));
    }

    pub fn get_entry(&self, key: &str) -> Option<&str> {
        self.entries.iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }
}

// ---------------------------------------------------------------------------
// Text report
// ---------------------------------------------------------------------------

/// Render a report as human-readable plain text.
pub fn render_text(report: &BenchmarkReport) -> String {
    let mut out = String::with_capacity(4096);

    let bar = "=".repeat(70);
    let _ = writeln!(out, "{}", bar);
    let _ = writeln!(out, "  {}", report.title);
    let _ = writeln!(out, "  Generated at: {}", report.timestamp);
    let _ = writeln!(out, "{}", bar);
    let _ = writeln!(out);

    for (k, v) in &report.metadata {
        let _ = writeln!(out, "  {}: {}", k, v);
    }
    if !report.metadata.is_empty() {
        let _ = writeln!(out);
    }

    for section in &report.sections {
        let _ = writeln!(out, "--- {} ---", section.name);
        let max_key_len = section.entries.iter().map(|(k, _)| k.len()).max().unwrap_or(0);
        for (k, v) in &section.entries {
            let _ = writeln!(out, "  {:width$}  {}", k, v, width = max_key_len);
        }
        let _ = writeln!(out);
    }

    out
}

// ---------------------------------------------------------------------------
// JSON report
// ---------------------------------------------------------------------------

/// Render a report as a JSON string (hand-serialised to avoid serde dependency).
pub fn render_json(report: &BenchmarkReport) -> String {
    let mut out = String::with_capacity(4096);
    out.push_str("{\n");
    let _ = write!(out, "  \"title\": {},\n", json_str(&report.title));
    let _ = write!(out, "  \"timestamp\": {},\n", report.timestamp);

    // metadata
    out.push_str("  \"metadata\": {");
    let meta_items: Vec<String> = report.metadata.iter()
        .map(|(k, v)| format!("{}: {}", json_str(k), json_str(v)))
        .collect();
    out.push_str(&meta_items.join(", "));
    out.push_str("},\n");

    // sections
    out.push_str("  \"sections\": [\n");
    for (si, section) in report.sections.iter().enumerate() {
        out.push_str("    {\n");
        let _ = write!(out, "      \"name\": {},\n", json_str(&section.name));
        out.push_str("      \"entries\": {\n");
        for (ei, (k, v)) in section.entries.iter().enumerate() {
            let comma = if ei + 1 < section.entries.len() { "," } else { "" };
            let _ = writeln!(out, "        {}: {}{}", json_str(k), json_str(v), comma);
        }
        out.push_str("      }\n");
        let comma = if si + 1 < report.sections.len() { "," } else { "" };
        let _ = writeln!(out, "    }}{}", comma);
    }
    out.push_str("  ]\n");
    out.push_str("}\n");
    out
}

fn json_str(s: &str) -> String {
    let escaped = s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t");
    format!("\"{}\"", escaped)
}

// ---------------------------------------------------------------------------
// CSV export
// ---------------------------------------------------------------------------

/// Export benchmark results as CSV rows.
///
/// Returns a string with header and one row per benchmark result.
pub fn render_csv(results: &[BenchmarkResult]) -> String {
    let mut out = String::with_capacity(2048);
    out.push_str("name,iterations,mean_s,median_s,std_dev_s,min_s,max_s,p95_s,p99_s,cv,throughput\n");

    for r in results {
        let _ = writeln!(
            out,
            "{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.4},{:.2}",
            csv_escape(&r.config.name),
            r.iteration_count(),
            r.stats.mean,
            r.stats.median,
            r.stats.std_dev,
            r.stats.min,
            r.stats.max,
            r.stats.p95,
            r.stats.p99,
            r.stats.coefficient_of_variation(),
            r.throughput_per_second(),
        );
    }
    out
}

/// Export metrics snapshots as CSV.
pub fn render_metrics_csv(collector: &MetricsCollector) -> String {
    let mut out = String::with_capacity(2048);
    out.push_str("label,analysis_time_ms,peak_memory_mb,smt_calls,smt_time_ms,anomalies,constraints,variables,deps\n");

    for (label, snap) in collector.labels().iter().zip(collector.snapshots().iter()) {
        let _ = writeln!(
            out,
            "{},{:.3},{:.2},{},{:.3},{},{},{},{}",
            csv_escape(label),
            snap.analysis_time.as_secs_f64() * 1000.0,
            snap.peak_memory_bytes as f64 / (1024.0 * 1024.0),
            snap.smt_calls,
            snap.smt_time.as_secs_f64() * 1000.0,
            snap.anomalies_found,
            snap.constraint_count,
            snap.variable_count,
            snap.dependency_count,
        );
    }
    out
}

fn csv_escape(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

// ---------------------------------------------------------------------------
// Comparison table
// ---------------------------------------------------------------------------

/// Generate a text comparison table from a list of comparisons.
pub fn render_comparison_table(comparisons: &[Comparison]) -> String {
    let mut out = String::with_capacity(2048);
    let _ = writeln!(out, "{:<20} {:<20} {:>10} {:>12} {:>8}",
        "Baseline", "Candidate", "Speedup", "%Change", "Faster?");
    let _ = writeln!(out, "{}", "-".repeat(72));

    for cmp in comparisons {
        let _ = writeln!(out, "{:<20} {:<20} {:>10.3}x {:>11.2}% {:>8}",
            truncate(&cmp.baseline_name, 20),
            truncate(&cmp.candidate_name, 20),
            cmp.speedup,
            cmp.percent_change(),
            if cmp.is_faster() { "yes" } else { "no" },
        );
    }
    out
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}...", &s[..max - 3])
    }
}

// ---------------------------------------------------------------------------
// Chart data export
// ---------------------------------------------------------------------------

/// Data point for a performance chart.
#[derive(Debug, Clone)]
pub struct ChartDataPoint {
    pub x: f64,
    pub y: f64,
    pub label: String,
    pub series: String,
}

/// Generate chart data for time-vs-workload-size.
pub fn chart_time_vs_size(results: &[BenchmarkResult]) -> Vec<ChartDataPoint> {
    results.iter().enumerate().map(|(i, r)| {
        let size = r.config.parameters.get("size")
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(i as f64);
        ChartDataPoint {
            x: size,
            y: r.stats.mean,
            label: r.config.name.clone(),
            series: "mean_time".to_string(),
        }
    }).collect()
}

/// Generate chart data for time percentiles.
pub fn chart_percentiles(results: &[BenchmarkResult]) -> Vec<ChartDataPoint> {
    let mut points = Vec::new();
    for r in results {
        let name = &r.config.name;
        for (pct, val) in &[
            ("p25", r.stats.p25), ("p50", r.stats.median),
            ("p75", r.stats.p75), ("p90", r.stats.p90),
            ("p95", r.stats.p95), ("p99", r.stats.p99),
        ] {
            points.push(ChartDataPoint {
                x: match *pct {
                    "p25" => 25.0, "p50" => 50.0, "p75" => 75.0,
                    "p90" => 90.0, "p95" => 95.0, "p99" => 99.0,
                    _ => 0.0,
                },
                y: *val,
                label: name.clone(),
                series: pct.to_string(),
            });
        }
    }
    points
}

/// Export chart data as CSV.
pub fn render_chart_csv(points: &[ChartDataPoint]) -> String {
    let mut out = String::with_capacity(1024);
    out.push_str("series,label,x,y\n");
    for p in points {
        let _ = writeln!(out, "{},{},{:.6},{:.6}",
            csv_escape(&p.series), csv_escape(&p.label), p.x, p.y);
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::{BenchmarkHarness, ExperimentConfig};
    use std::time::Duration;

    fn make_results() -> Vec<BenchmarkResult> {
        let mut harness = BenchmarkHarness::new();
        let cfg1 = ExperimentConfig::new("fast")
            .with_warmup(0)
            .with_iterations(5)
            .with_param("size", "10");
        let cfg2 = ExperimentConfig::new("slow")
            .with_warmup(0)
            .with_iterations(5)
            .with_param("size", "100");

        let r1 = harness.run(&cfg1, |_| None);
        let r2 = harness.run(&cfg2, |_| { std::thread::sleep(Duration::from_micros(50)); None });
        vec![r1, r2]
    }

    #[test]
    fn test_report_from_results() {
        let results = make_results();
        let report = BenchmarkReport::from_results("Test Suite", &results);
        assert_eq!(report.title, "Test Suite");
        // summary + 2 experiment sections
        assert_eq!(report.sections.len(), 3);
        let summary = &report.sections[0];
        assert_eq!(summary.get_entry("experiment_count"), Some("2"));
    }

    #[test]
    fn test_render_text() {
        let results = make_results();
        let report = BenchmarkReport::from_results("Text Test", &results);
        let text = render_text(&report);
        assert!(text.contains("Text Test"));
        assert!(text.contains("fast"));
        assert!(text.contains("slow"));
        assert!(text.contains("mean_seconds"));
    }

    #[test]
    fn test_render_json() {
        let results = make_results();
        let report = BenchmarkReport::from_results("JSON Test", &results);
        let json = render_json(&report);
        assert!(json.contains("\"title\""));
        assert!(json.contains("\"sections\""));
        assert!(json.contains("JSON Test"));
    }

    #[test]
    fn test_render_csv() {
        let results = make_results();
        let csv = render_csv(&results);
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines.len(), 3); // header + 2 rows
        assert!(lines[0].starts_with("name,"));
        assert!(lines[1].starts_with("fast,"));
    }

    #[test]
    fn test_render_metrics_csv() {
        let mut collector = MetricsCollector::new();
        collector.record("run1", AnalysisMetrics::new()
            .with_time(Duration::from_millis(100))
            .with_smt(5, Duration::from_millis(50))
            .with_anomalies(2));
        let csv = render_metrics_csv(&collector);
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[1].starts_with("run1,"));
    }

    #[test]
    fn test_comparison_section() {
        let stats = Statistics::from_samples(&[1.0, 1.0, 1.0]);
        let cmp = Comparison {
            baseline_name: "A".to_string(),
            candidate_name: "B".to_string(),
            speedup: 2.0,
            mean_diff_seconds: -0.5,
            median_diff_seconds: -0.5,
            baseline_stats: stats.clone(),
            candidate_stats: Statistics::from_samples(&[0.5, 0.5, 0.5]),
        };
        let sec = BenchmarkReport::comparison_section(&cmp);
        assert_eq!(sec.get_entry("faster"), Some("true"));
    }

    #[test]
    fn test_comparison_table() {
        let stats1 = Statistics::from_samples(&[1.0, 1.0]);
        let stats2 = Statistics::from_samples(&[0.5, 0.5]);
        let cmp = Comparison {
            baseline_name: "pg".to_string(),
            candidate_name: "mysql".to_string(),
            speedup: 2.0,
            mean_diff_seconds: -0.5,
            median_diff_seconds: -0.5,
            baseline_stats: stats1,
            candidate_stats: stats2,
        };
        let table = render_comparison_table(&[cmp]);
        assert!(table.contains("pg"));
        assert!(table.contains("mysql"));
        assert!(table.contains("2.000x"));
    }

    #[test]
    fn test_chart_time_vs_size() {
        let results = make_results();
        let points = chart_time_vs_size(&results);
        assert_eq!(points.len(), 2);
        assert!((points[0].x - 10.0).abs() < 1e-9);
        assert!((points[1].x - 100.0).abs() < 1e-9);
    }

    #[test]
    fn test_chart_percentiles() {
        let results = make_results();
        let points = chart_percentiles(&results);
        // 6 percentiles × 2 results = 12
        assert_eq!(points.len(), 12);
    }

    #[test]
    fn test_chart_csv() {
        let points = vec![
            ChartDataPoint { x: 1.0, y: 2.0, label: "a".to_string(), series: "s1".to_string() },
            ChartDataPoint { x: 3.0, y: 4.0, label: "b".to_string(), series: "s1".to_string() },
        ];
        let csv = render_chart_csv(&points);
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines.len(), 3);
    }

    #[test]
    fn test_json_escaping() {
        assert_eq!(json_str("hello"), "\"hello\"");
        assert_eq!(json_str("a\"b"), "\"a\\\"b\"");
        assert_eq!(json_str("a\nb"), "\"a\\nb\"");
    }

    #[test]
    fn test_csv_escaping() {
        assert_eq!(csv_escape("simple"), "simple");
        assert_eq!(csv_escape("has,comma"), "\"has,comma\"");
        assert_eq!(csv_escape("has\"quote"), "\"has\"\"quote\"");
    }

    #[test]
    fn test_metrics_section() {
        let m = AnalysisMetrics::new()
            .with_time(Duration::from_millis(250))
            .with_smt(10, Duration::from_millis(100))
            .with_constraints(500, 200)
            .with_anomalies(3)
            .with_memory(2 * 1024 * 1024)
            .with_dependencies(15);
        let sec = BenchmarkReport::metrics_section("run_pg", &m);
        assert_eq!(sec.name, "run_pg");
        assert_eq!(sec.get_entry("anomalies_found"), Some("3"));
        assert_eq!(sec.get_entry("constraint_count"), Some("500"));
    }

    #[test]
    fn test_truncate() {
        assert_eq!(truncate("short", 20), "short");
        assert_eq!(truncate("this is a very long string indeed", 15), "this is a ve...");
    }

    #[test]
    fn test_report_with_metadata() {
        let report = BenchmarkReport::new("test")
            .with_metadata("version", "1.0.0")
            .with_metadata("host", "localhost");
        assert_eq!(report.metadata.len(), 2);
        let text = render_text(&report);
        assert!(text.contains("version"));
    }
}
