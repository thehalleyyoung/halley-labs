//! Output formatting for CLI results.
//!
//! Converts analysis findings, repair plans, summaries, and benchmark
//! results into human-readable tables, JSON, YAML, Markdown, or plain text.

use crate::commands::OutputFormat;
use serde::{Deserialize, Serialize};
use std::fmt::Write;

// ── Data transfer objects ──────────────────────────────────────────────────

/// A single finding summarised for display.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FindingSummary {
    pub severity: String,
    pub title: String,
    pub service: String,
    pub description: String,
}

/// A single repair plan summarised for display.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RepairSummary {
    pub description: String,
    pub changes: usize,
    pub cost: f64,
}

/// Aggregate statistics for an analysis run.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AnalysisSummary {
    pub services: usize,
    pub edges: usize,
    pub risks: usize,
    pub duration_ms: u64,
}

/// A single row in a benchmark report.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BenchmarkResult {
    pub topology: String,
    pub size: usize,
    pub time_ms: u64,
    pub risks_found: usize,
}

// ── Severity indicator ─────────────────────────────────────────────────────

/// Map a severity string to a coloured emoji indicator.
pub fn severity_indicator(severity: &str) -> &'static str {
    match severity.to_uppercase().as_str() {
        "CRITICAL" => "🔴",
        "HIGH" => "🟠",
        "MEDIUM" => "🟡",
        "LOW" => "🔵",
        "INFO" => "⚪",
        _ => "⚪",
    }
}

// ── Table formatter ────────────────────────────────────────────────────────

/// Utility for rendering fixed-width ASCII tables.
pub struct TableFormatter;

impl TableFormatter {
    /// Compute the minimum column widths that fit all data.
    pub fn compute_col_widths(headers: &[&str], rows: &[Vec<String>]) -> Vec<usize> {
        let mut widths: Vec<usize> = headers.iter().map(|h| h.len()).collect();
        for row in rows {
            for (i, cell) in row.iter().enumerate() {
                if i < widths.len() {
                    widths[i] = widths[i].max(cell.len());
                }
            }
        }
        widths
    }

    /// Render a table with headers and rows using the given column widths.
    pub fn format(headers: &[&str], rows: &[Vec<String>], col_widths: &[usize]) -> String {
        let mut out = String::new();

        // Separator line.
        let sep: String = col_widths
            .iter()
            .map(|&w| "-".repeat(w + 2))
            .collect::<Vec<_>>()
            .join("+");
        let sep_line = format!("+{}+", sep);

        // Header row.
        out.push_str(&sep_line);
        out.push('\n');
        let header_cells: Vec<String> = headers
            .iter()
            .enumerate()
            .map(|(i, h)| {
                let w = col_widths.get(i).copied().unwrap_or(h.len());
                format!(" {:<w$} ", h, w = w)
            })
            .collect();
        out.push('|');
        out.push_str(&header_cells.join("|"));
        out.push('|');
        out.push('\n');
        out.push_str(&sep_line);
        out.push('\n');

        // Data rows.
        for row in rows {
            let cells: Vec<String> = row
                .iter()
                .enumerate()
                .map(|(i, cell)| {
                    let w = col_widths.get(i).copied().unwrap_or(cell.len());
                    format!(" {:<w$} ", cell, w = w)
                })
                .collect();
            out.push('|');
            out.push_str(&cells.join("|"));
            out.push('|');
            out.push('\n');
        }

        out.push_str(&sep_line);
        out.push('\n');
        out
    }
}

// ── Progress display ───────────────────────────────────────────────────────

/// Simple progress tracker for long-running operations.
pub struct ProgressDisplay {
    total: usize,
    current: usize,
    message: String,
}

impl ProgressDisplay {
    pub fn new(total: usize, message: &str) -> Self {
        Self {
            total,
            current: 0,
            message: message.to_string(),
        }
    }

    /// Advance by one step and return a progress string.
    pub fn tick(&mut self) -> String {
        self.current = (self.current + 1).min(self.total);
        let pct = if self.total > 0 {
            (self.current as f64 / self.total as f64 * 100.0) as u32
        } else {
            100
        };
        let bar_width = 30;
        let filled = (pct as usize * bar_width) / 100;
        let empty = bar_width - filled;
        format!(
            "\r{} [{}{}] {}/{} ({}%)",
            self.message,
            "█".repeat(filled),
            "░".repeat(empty),
            self.current,
            self.total,
            pct,
        )
    }

    /// Return a final "done" message.
    pub fn finish(&self) -> String {
        format!(
            "{} [{}] {}/{} (100%) ✓",
            self.message,
            "█".repeat(30),
            self.total,
            self.total,
        )
    }
}

// ── CliOutput ──────────────────────────────────────────────────────────────

/// Main output renderer.
#[derive(Debug, Clone)]
pub struct CliOutput {
    pub color: bool,
}

impl CliOutput {
    pub fn new() -> Self {
        Self { color: true }
    }

    /// Render findings in the requested format.
    pub fn print_findings(
        &self,
        findings: &[FindingSummary],
        format: &OutputFormat,
    ) -> String {
        match format {
            OutputFormat::Json => self.findings_json(findings),
            OutputFormat::Yaml => self.findings_yaml(findings),
            OutputFormat::Markdown => self.findings_markdown(findings),
            OutputFormat::Table | OutputFormat::Sarif | OutputFormat::JUnit => {
                self.findings_table(findings)
            }
        }
    }

    /// Render repair plans in the requested format.
    pub fn print_repairs(
        &self,
        repairs: &[RepairSummary],
        format: &OutputFormat,
    ) -> String {
        match format {
            OutputFormat::Json => self.repairs_json(repairs),
            OutputFormat::Yaml => self.repairs_yaml(repairs),
            OutputFormat::Markdown => self.repairs_markdown(repairs),
            OutputFormat::Table | OutputFormat::Sarif | OutputFormat::JUnit => {
                self.repairs_table(repairs)
            }
        }
    }

    /// Render an analysis summary (always plain text).
    pub fn print_summary(&self, stats: &AnalysisSummary) -> String {
        let mut out = String::new();
        writeln!(out, "╭──────────────────────────────╮").unwrap();
        writeln!(out, "│   CascadeVerify Summary      │").unwrap();
        writeln!(out, "├──────────────────────────────┤").unwrap();
        writeln!(out, "│ Services analysed: {:>8}  │", stats.services).unwrap();
        writeln!(out, "│ Dependency edges:  {:>8}  │", stats.edges).unwrap();
        writeln!(out, "│ Risks detected:    {:>8}  │", stats.risks).unwrap();
        writeln!(out, "│ Analysis time:     {:>5}ms  │", stats.duration_ms).unwrap();
        writeln!(out, "╰──────────────────────────────╯").unwrap();
        out
    }

    /// Render benchmark results as a table.
    pub fn print_benchmark(&self, results: &[BenchmarkResult]) -> String {
        let headers = &["Topology", "Size", "Time (ms)", "Risks"];
        let rows: Vec<Vec<String>> = results
            .iter()
            .map(|r| {
                vec![
                    r.topology.clone(),
                    r.size.to_string(),
                    r.time_ms.to_string(),
                    r.risks_found.to_string(),
                ]
            })
            .collect();
        let widths = TableFormatter::compute_col_widths(headers, &rows);
        TableFormatter::format(headers, &rows, &widths)
    }

    // ── Private: Findings ──────────────────────────────────────────────

    fn findings_table(&self, findings: &[FindingSummary]) -> String {
        if findings.is_empty() {
            return "✅ No cascade risks detected.\n".to_string();
        }

        let headers = &["", "Severity", "Title", "Service", "Description"];
        let rows: Vec<Vec<String>> = findings
            .iter()
            .map(|f| {
                vec![
                    severity_indicator(&f.severity).to_string(),
                    f.severity.clone(),
                    f.title.clone(),
                    f.service.clone(),
                    f.description.clone(),
                ]
            })
            .collect();
        let widths = TableFormatter::compute_col_widths(headers, &rows);
        let mut out = format!("Found {} cascade risk(s):\n\n", findings.len());
        out.push_str(&TableFormatter::format(headers, &rows, &widths));
        out
    }

    fn findings_json(&self, findings: &[FindingSummary]) -> String {
        serde_json::to_string_pretty(findings).unwrap_or_else(|_| "[]".into())
    }

    fn findings_yaml(&self, findings: &[FindingSummary]) -> String {
        serde_yaml::to_string(findings).unwrap_or_else(|_| "---\n[]".into())
    }

    fn findings_markdown(&self, findings: &[FindingSummary]) -> String {
        if findings.is_empty() {
            return "## CascadeVerify Results\n\n✅ No cascade risks detected.\n".to_string();
        }

        let mut out = String::new();
        writeln!(out, "## CascadeVerify Results").unwrap();
        writeln!(out).unwrap();
        writeln!(out, "Found **{}** cascade risk(s):", findings.len()).unwrap();
        writeln!(out).unwrap();
        writeln!(out, "| {} | Severity | Title | Service | Description |", "").unwrap();
        writeln!(out, "|---|----------|-------|---------|-------------|").unwrap();
        for f in findings {
            writeln!(
                out,
                "| {} | {} | {} | {} | {} |",
                severity_indicator(&f.severity),
                f.severity,
                f.title,
                f.service,
                f.description,
            )
            .unwrap();
        }
        out
    }

    // ── Private: Repairs ───────────────────────────────────────────────

    fn repairs_table(&self, repairs: &[RepairSummary]) -> String {
        if repairs.is_empty() {
            return "No repairs needed.\n".to_string();
        }

        let headers = &["#", "Description", "Changes", "Cost"];
        let rows: Vec<Vec<String>> = repairs
            .iter()
            .enumerate()
            .map(|(i, r)| {
                vec![
                    (i + 1).to_string(),
                    r.description.clone(),
                    r.changes.to_string(),
                    format!("{:.2}", r.cost),
                ]
            })
            .collect();
        let widths = TableFormatter::compute_col_widths(headers, &rows);
        let mut out = format!("Synthesised {} repair plan(s):\n\n", repairs.len());
        out.push_str(&TableFormatter::format(headers, &rows, &widths));
        out
    }

    fn repairs_json(&self, repairs: &[RepairSummary]) -> String {
        serde_json::to_string_pretty(repairs).unwrap_or_else(|_| "[]".into())
    }

    fn repairs_yaml(&self, repairs: &[RepairSummary]) -> String {
        serde_yaml::to_string(repairs).unwrap_or_else(|_| "---\n[]".into())
    }

    fn repairs_markdown(&self, repairs: &[RepairSummary]) -> String {
        if repairs.is_empty() {
            return "## Repair Plans\n\nNo repairs needed.\n".to_string();
        }
        let mut out = String::new();
        writeln!(out, "## Repair Plans").unwrap();
        writeln!(out).unwrap();
        writeln!(out, "| # | Description | Changes | Cost |").unwrap();
        writeln!(out, "|---|-------------|---------|------|").unwrap();
        for (i, r) in repairs.iter().enumerate() {
            writeln!(
                out,
                "| {} | {} | {} | {:.2} |",
                i + 1,
                r.description,
                r.changes,
                r.cost,
            )
            .unwrap();
        }
        out
    }
}

impl Default for CliOutput {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── severity_indicator ─────────────────────────────────────────────

    #[test]
    fn severity_indicator_critical() {
        assert_eq!(severity_indicator("CRITICAL"), "🔴");
        assert_eq!(severity_indicator("critical"), "🔴");
        assert_eq!(severity_indicator("Critical"), "🔴");
    }

    #[test]
    fn severity_indicator_high() {
        assert_eq!(severity_indicator("HIGH"), "🟠");
    }

    #[test]
    fn severity_indicator_medium() {
        assert_eq!(severity_indicator("MEDIUM"), "🟡");
    }

    #[test]
    fn severity_indicator_low() {
        assert_eq!(severity_indicator("LOW"), "🔵");
    }

    #[test]
    fn severity_indicator_info() {
        assert_eq!(severity_indicator("INFO"), "⚪");
    }

    #[test]
    fn severity_indicator_unknown_maps_to_info() {
        assert_eq!(severity_indicator("UNKNOWN"), "⚪");
        assert_eq!(severity_indicator(""), "⚪");
    }

    // ── TableFormatter ─────────────────────────────────────────────────

    #[test]
    fn compute_col_widths_uses_header_lengths_as_minimum() {
        let headers = &["Name", "Value"];
        let rows: Vec<Vec<String>> = vec![vec!["a".into(), "b".into()]];
        let widths = TableFormatter::compute_col_widths(headers, &rows);
        assert_eq!(widths, vec![4, 5]);
    }

    #[test]
    fn compute_col_widths_expands_for_long_data() {
        let headers = &["X"];
        let rows: Vec<Vec<String>> = vec![vec!["longvalue".into()]];
        let widths = TableFormatter::compute_col_widths(headers, &rows);
        assert_eq!(widths, vec![9]);
    }

    #[test]
    fn table_format_renders_borders() {
        let headers = &["A", "B"];
        let rows = vec![vec!["1".into(), "2".into()]];
        let widths = vec![3, 3];
        let table = TableFormatter::format(headers, &rows, &widths);
        assert!(table.contains("+"));
        assert!(table.contains("|"));
        assert!(table.contains("A"));
        assert!(table.contains("1"));
    }

    #[test]
    fn table_format_empty_rows() {
        let headers = &["Col"];
        let rows: Vec<Vec<String>> = vec![];
        let widths = vec![5];
        let table = TableFormatter::format(headers, &rows, &widths);
        // Should have header + separator lines, no data rows.
        let lines: Vec<&str> = table.lines().collect();
        assert!(lines.len() >= 3); // top border, header, bottom border
    }

    // ── ProgressDisplay ────────────────────────────────────────────────

    #[test]
    fn progress_display_tick_advances() {
        let mut p = ProgressDisplay::new(10, "Testing");
        let s1 = p.tick();
        assert!(s1.contains("1/10"));
        let s2 = p.tick();
        assert!(s2.contains("2/10"));
    }

    #[test]
    fn progress_display_finish() {
        let p = ProgressDisplay::new(5, "Done");
        let s = p.finish();
        assert!(s.contains("5/5"));
        assert!(s.contains("100%"));
        assert!(s.contains("✓"));
    }

    #[test]
    fn progress_display_zero_total_does_not_panic() {
        let mut p = ProgressDisplay::new(0, "Empty");
        let s = p.tick();
        assert!(s.contains("100%"));
    }

    // ── CliOutput: findings ────────────────────────────────────────────

    fn sample_findings() -> Vec<FindingSummary> {
        vec![
            FindingSummary {
                severity: "CRITICAL".into(),
                title: "Retry storm".into(),
                service: "svc-a".into(),
                description: "Amplification factor 64x".into(),
            },
            FindingSummary {
                severity: "HIGH".into(),
                title: "Timeout cascade".into(),
                service: "svc-b".into(),
                description: "Timeout exceeds deadline by 5s".into(),
            },
        ]
    }

    #[test]
    fn print_findings_table_includes_all_rows() {
        let out = CliOutput::new();
        let s = out.print_findings(&sample_findings(), &OutputFormat::Table);
        assert!(s.contains("Retry storm"));
        assert!(s.contains("Timeout cascade"));
        assert!(s.contains("svc-a"));
        assert!(s.contains("svc-b"));
        assert!(s.contains("2 cascade risk(s)"));
    }

    #[test]
    fn print_findings_table_empty() {
        let out = CliOutput::new();
        let s = out.print_findings(&[], &OutputFormat::Table);
        assert!(s.contains("No cascade risks detected"));
    }

    #[test]
    fn print_findings_json_is_valid_json() {
        let out = CliOutput::new();
        let s = out.print_findings(&sample_findings(), &OutputFormat::Json);
        let parsed: Vec<FindingSummary> = serde_json::from_str(&s).unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].severity, "CRITICAL");
    }

    #[test]
    fn print_findings_yaml_is_valid_yaml() {
        let out = CliOutput::new();
        let s = out.print_findings(&sample_findings(), &OutputFormat::Yaml);
        let parsed: Vec<FindingSummary> = serde_yaml::from_str(&s).unwrap();
        assert_eq!(parsed.len(), 2);
    }

    #[test]
    fn print_findings_markdown_has_table_headers() {
        let out = CliOutput::new();
        let s = out.print_findings(&sample_findings(), &OutputFormat::Markdown);
        assert!(s.contains("## CascadeVerify Results"));
        assert!(s.contains("| Severity |"));
        assert!(s.contains("Retry storm"));
    }

    #[test]
    fn print_findings_markdown_empty() {
        let out = CliOutput::new();
        let s = out.print_findings(&[], &OutputFormat::Markdown);
        assert!(s.contains("No cascade risks detected"));
    }

    // ── CliOutput: repairs ─────────────────────────────────────────────

    fn sample_repairs() -> Vec<RepairSummary> {
        vec![
            RepairSummary {
                description: "Reduce retries on A→B".into(),
                changes: 1,
                cost: 2.0,
            },
            RepairSummary {
                description: "Adjust timeout on B→C".into(),
                changes: 1,
                cost: 1.5,
            },
        ]
    }

    #[test]
    fn print_repairs_table() {
        let out = CliOutput::new();
        let s = out.print_repairs(&sample_repairs(), &OutputFormat::Table);
        assert!(s.contains("Reduce retries"));
        assert!(s.contains("2 repair plan(s)"));
    }

    #[test]
    fn print_repairs_empty() {
        let out = CliOutput::new();
        let s = out.print_repairs(&[], &OutputFormat::Table);
        assert!(s.contains("No repairs needed"));
    }

    #[test]
    fn print_repairs_json() {
        let out = CliOutput::new();
        let s = out.print_repairs(&sample_repairs(), &OutputFormat::Json);
        let parsed: Vec<RepairSummary> = serde_json::from_str(&s).unwrap();
        assert_eq!(parsed.len(), 2);
    }

    #[test]
    fn print_repairs_yaml() {
        let out = CliOutput::new();
        let s = out.print_repairs(&sample_repairs(), &OutputFormat::Yaml);
        assert!(s.contains("Reduce retries"));
    }

    #[test]
    fn print_repairs_markdown() {
        let out = CliOutput::new();
        let s = out.print_repairs(&sample_repairs(), &OutputFormat::Markdown);
        assert!(s.contains("## Repair Plans"));
        assert!(s.contains("| # |"));
    }

    #[test]
    fn print_repairs_markdown_empty() {
        let out = CliOutput::new();
        let s = out.print_repairs(&[], &OutputFormat::Markdown);
        assert!(s.contains("No repairs needed"));
    }

    // ── CliOutput: summary ─────────────────────────────────────────────

    #[test]
    fn print_summary_includes_all_stats() {
        let out = CliOutput::new();
        let stats = AnalysisSummary {
            services: 42,
            edges: 100,
            risks: 3,
            duration_ms: 150,
        };
        let s = out.print_summary(&stats);
        assert!(s.contains("42"));
        assert!(s.contains("100"));
        assert!(s.contains("3"));
        assert!(s.contains("150"));
        assert!(s.contains("Summary"));
    }

    // ── CliOutput: benchmark ───────────────────────────────────────────

    #[test]
    fn print_benchmark_renders_table() {
        let out = CliOutput::new();
        let results = vec![
            BenchmarkResult {
                topology: "chain".into(),
                size: 10,
                time_ms: 5,
                risks_found: 2,
            },
            BenchmarkResult {
                topology: "mesh".into(),
                size: 50,
                time_ms: 120,
                risks_found: 8,
            },
        ];
        let s = out.print_benchmark(&results);
        assert!(s.contains("chain"));
        assert!(s.contains("mesh"));
        assert!(s.contains("Topology"));
        assert!(s.contains("10"));
        assert!(s.contains("120"));
    }

    #[test]
    fn print_benchmark_empty() {
        let out = CliOutput::new();
        let s = out.print_benchmark(&[]);
        // Should still render headers.
        assert!(s.contains("Topology"));
    }

    // ── CliOutput default ──────────────────────────────────────────────

    #[test]
    fn cli_output_default_has_color_enabled() {
        let out = CliOutput::default();
        assert!(out.color);
    }
}
