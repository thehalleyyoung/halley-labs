//! Human-readable and structured reporting for cascade analysis results.
//!
//! Supports plain text (terminal), Markdown, JSON, YAML, SARIF, and JUnit
//! output.  The [`Reporter`] struct acts as the single entry-point that
//! delegates to format-specific renderers.

use std::collections::HashMap;
use std::fmt::Write as FmtWrite;

use serde::{Deserialize, Serialize};

use cascade_types::report::{Evidence, Finding, Location, Severity};
use cascade_types::repair::{RepairAction, RepairPlan};

use crate::pipeline::PipelineResult;
use crate::sarif::{ReportMetadata, SarifGenerator};
use crate::junit::{JUnitGenerator, JUnitMetadata};

// ---------------------------------------------------------------------------
// ReportFormat
// ---------------------------------------------------------------------------

/// Available output formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ReportFormat {
    Plain,
    Markdown,
    Json,
    Yaml,
    Sarif,
    JUnit,
}

impl ReportFormat {
    /// Lenient parser that accepts common aliases.
    pub fn from_str_loose(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "plain" | "text" | "txt" => Self::Plain,
            "markdown" | "md" => Self::Markdown,
            "json" => Self::Json,
            "yaml" | "yml" => Self::Yaml,
            "sarif" => Self::Sarif,
            "junit" | "xml" => Self::JUnit,
            _ => Self::Plain,
        }
    }

    /// Canonical file extension for this format.
    pub fn file_extension(self) -> &'static str {
        match self {
            Self::Plain => "txt",
            Self::Markdown => "md",
            Self::Json => "json",
            Self::Yaml => "yaml",
            Self::Sarif => "sarif.json",
            Self::JUnit => "xml",
        }
    }
}

// ---------------------------------------------------------------------------
// Reporter
// ---------------------------------------------------------------------------

/// Central reporter that dispatches to format-specific renderers.
pub struct Reporter;

impl Reporter {
    /// Generate a formatted report string from pipeline results.
    pub fn generate_report(results: &PipelineResult, format: ReportFormat) -> String {
        match format {
            ReportFormat::Plain => Self::format_plain(results),
            ReportFormat::Markdown => Self::format_markdown(results),
            ReportFormat::Json => Self::format_json(results),
            ReportFormat::Yaml => Self::format_yaml(results),
            ReportFormat::Sarif => Self::format_sarif(results),
            ReportFormat::JUnit => Self::format_junit(results),
        }
    }

    // ---- Plain text -------------------------------------------------------

    pub fn format_plain(results: &PipelineResult) -> String {
        let mut out = String::with_capacity(4096);

        let sev = SeverityCounts::from_findings(&results.findings);

        let _ = writeln!(
            out,
            "============================================================"
        );
        let _ = writeln!(
            out,
            "              CascadeVerify Analysis Report"
        );
        let _ = writeln!(
            out,
            "============================================================"
        );
        let _ = writeln!(out);

        let _ = writeln!(
            out,
            "Found {} cascade risk(s) across analysis.",
            results.findings.len()
        );
        let _ = writeln!(
            out,
            "  {} critical  {} error  {} warning  {} info",
            sev.critical, sev.high, sev.medium, sev.info
        );
        let _ = writeln!(out);

        if results.findings.is_empty() {
            let _ = writeln!(out, "  No issues found. All checks passed.");
            return out;
        }

        let _ = writeln!(
            out,
            "--- Findings ---------------------------------------------------"
        );
        let _ = writeln!(out);

        for (i, finding) in results.findings.iter().enumerate() {
            let badge = SeverityFormatter::plain_badge(finding.severity);
            let _ = writeln!(out, "  [{:>2}] {} {}", i + 1, badge, finding.description);
            let _ = writeln!(out, "       ID: {}", finding.id);
            if let Some(file) = &finding.location.file {
                let _ = writeln!(
                    out,
                    "       Location: {}:{}:{}",
                    file,
                    finding.location.line.unwrap_or(0),
                    finding.location.column.unwrap_or(0)
                );
            }
            for ev in &finding.evidence {
                let _ = write!(out, "         - {}", ev.description);
                if let Some(src) = &ev.source {
                    let _ = write!(out, " ({})", src);
                }
                let _ = writeln!(out);
            }
            let _ = writeln!(out);
        }

        if let Some(repairs) = &results.repairs {
            if !repairs.is_empty() {
                let _ = writeln!(
                    out,
                    "--- Repair Suggestions -----------------------------------------"
                );
                let _ = writeln!(out);
                for (i, repair) in repairs.iter().enumerate() {
                    let _ = writeln!(out, "  Repair #{} (cost {:.2}):", i + 1, repair.cost);
                    for action in &repair.actions {
                        let _ = writeln!(
                            out,
                            "    {}",
                            action.description()
                        );
                    }
                    let _ = writeln!(out);
                }
            }
        }

        let _ = writeln!(
            out,
            "--- Summary ----------------------------------------------------"
        );
        let _ = writeln!(out, "  Total findings: {}, pass: {}", results.report.summary.total_findings, results.report.summary.pass);
        let _ = writeln!(out, "  Exit code: {}", results.exit_code);
        if let Some(dur) = results.stats.per_stage_duration.get("tier1_analysis") {
            let _ = writeln!(out, "  Tier-1 analysis: {}ms", dur);
        }
        let _ = writeln!(
            out,
            "  Total duration: {}ms",
            results.stats.total_duration_ms
        );

        out
    }

    // ---- Markdown ---------------------------------------------------------

    pub fn format_markdown(results: &PipelineResult) -> String {
        let mut md = String::with_capacity(8192);

        let sev = SeverityCounts::from_findings(&results.findings);

        let _ = writeln!(md, "# CascadeVerify Analysis Report\n");
        let _ = writeln!(
            md,
            "**Summary:** {} issue(s) found \u{2014} {} critical, {} error, {} warning, {} info\n",
            results.findings.len(),
            sev.critical,
            sev.high,
            sev.medium,
            sev.info,
        );

        if results.findings.is_empty() {
            let _ = writeln!(md, "> No cascade risks detected.\n");
            return md;
        }

        // Findings table
        let _ = writeln!(md, "## Findings\n");
        let _ = writeln!(md, "| # | Severity | ID | Description |");
        let _ = writeln!(md, "|---|----------|-----|-------------|");
        for (i, f) in results.findings.iter().enumerate() {
            let badge = SeverityFormatter::markdown_badge(f.severity);
            let desc = f.description.replace('|', "\\|");
            let _ = writeln!(md, "| {} | {} | `{}` | {} |", i + 1, badge, f.id, desc);
        }
        let _ = writeln!(md);

        // Details (collapsible)
        let _ = writeln!(md, "## Details\n");
        for finding in &results.findings {
            let _ = writeln!(
                md,
                "<details>\n<summary>{} <code>{}</code> \u{2014} {}</summary>\n",
                SeverityFormatter::markdown_badge(finding.severity),
                finding.id,
                finding.description,
            );
            if let Some(file) = &finding.location.file {
                let _ = writeln!(
                    md,
                    "**Location:** `{}:{}:{}`\n",
                    file,
                    finding.location.line.unwrap_or(0),
                    finding.location.column.unwrap_or(0),
                );
            }
            if !finding.evidence.is_empty() {
                let _ = writeln!(md, "**Evidence:**\n");
                for ev in &finding.evidence {
                    let _ = write!(md, "- {}", ev.description);
                    if let Some(src) = &ev.source {
                        let _ = write!(md, " (`{}`)", src);
                    }
                    let _ = writeln!(md);
                }
            }
            let _ = writeln!(md, "\n</details>\n");
        }

        // Repairs
        if let Some(repairs) = &results.repairs {
            if !repairs.is_empty() {
                let _ = writeln!(md, "## Repair Suggestions\n");
                for (i, repair) in repairs.iter().enumerate() {
                    let _ = writeln!(md, "### Repair #{} (cost {:.2})\n", i + 1, repair.cost);
                    let _ = writeln!(md, "```diff");
                    for action in &repair.actions {
                        let _ = writeln!(
                            md,
                            "+ {}",
                            action.description()
                        );
                    }
                    let _ = writeln!(md, "```\n");
                }
            }
        }

        // Timing table
        let _ = writeln!(md, "## Timing\n");
        let _ = writeln!(md, "| Stage | Duration (ms) |");
        let _ = writeln!(md, "|-------|---------------|");
        let mut stages: Vec<_> = results.stats.per_stage_duration.iter().collect();
        stages.sort_by_key(|(k, _)| k.clone());
        for (stage, ms) in &stages {
            let _ = writeln!(md, "| {} | {} |", stage, ms);
        }
        let _ = writeln!(
            md,
            "| **Total** | **{}** |",
            results.stats.total_duration_ms
        );
        let _ = writeln!(md, "\nExit code: `{}`", results.exit_code);

        md
    }

    // ---- JSON -------------------------------------------------------------

    pub fn format_json(results: &PipelineResult) -> String {
        serde_json::to_string_pretty(results).unwrap_or_else(|_| "{}".to_string())
    }

    // ---- YAML -------------------------------------------------------------

    pub fn format_yaml(results: &PipelineResult) -> String {
        serde_yaml::to_string(results).unwrap_or_else(|_| "---\n".to_string())
    }

    // ---- SARIF (delegates to sarif module) --------------------------------

    pub fn format_sarif(results: &PipelineResult) -> String {
        let metadata = ReportMetadata {
            tool_version: "0.1.0".into(),
            exit_code: Some(results.exit_code),
            ..Default::default()
        };
        let report = if let Some(repairs) = &results.repairs {
            SarifGenerator::generate_with_repairs(&results.findings, repairs, &metadata)
        } else {
            SarifGenerator::generate(&results.findings, &metadata)
        };
        SarifGenerator::to_json(&report)
    }

    // ---- JUnit (delegates to junit module) --------------------------------

    pub fn format_junit(results: &PipelineResult) -> String {
        let metadata = JUnitMetadata {
            report_name: "CascadeVerify".into(),
            hostname: "localhost".into(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            elapsed_sec: results.stats.total_duration_ms as f64 / 1000.0,
        };
        let report = JUnitGenerator::generate(&results.findings, &metadata);
        JUnitGenerator::to_xml(&report)
    }
}

// ---------------------------------------------------------------------------
// Severity formatting helpers
// ---------------------------------------------------------------------------

/// Utility for severity-dependent badges and colours.
pub struct SeverityFormatter;

impl SeverityFormatter {
    /// ASCII badge for terminal output.
    pub fn plain_badge(severity: Severity) -> &'static str {
        match severity {
            Severity::Critical => "[CRIT]",
            Severity::High => "[HIGH]",
            Severity::Medium => "[MED ]",
            Severity::Low => "[LOW ]",
            Severity::Info => "[INFO]",
        }
    }

    /// Emoji + label for Markdown.
    pub fn markdown_badge(severity: Severity) -> &'static str {
        match severity {
            Severity::Critical => "\u{1f534} Critical",
            Severity::High => "\u{1f7e0} High",
            Severity::Medium => "\u{1f7e1} Medium",
            Severity::Low => "\u{1f535} Low",
            Severity::Info => "\u{26aa} Info",
        }
    }

    /// ANSI escape code for the given severity.
    pub fn ansi_color(severity: Severity) -> &'static str {
        match severity {
            Severity::Critical => "\x1b[1;31m",
            Severity::High => "\x1b[31m",
            Severity::Medium => "\x1b[33m",
            Severity::Low => "\x1b[34m",
            Severity::Info => "\x1b[36m",
        }
    }

    /// ANSI reset sequence.
    pub fn ansi_reset() -> &'static str {
        "\x1b[0m"
    }
}

// ---------------------------------------------------------------------------
// Table formatter
// ---------------------------------------------------------------------------

/// Renders tabular data with aligned columns for terminal output.
pub struct TableFormatter;

impl TableFormatter {
    /// Render `rows` under `headers` as an aligned ASCII table.
    pub fn format(headers: &[&str], rows: &[Vec<String>]) -> String {
        let col_count = headers.len();
        let mut widths: Vec<usize> = headers.iter().map(|h| h.len()).collect();
        for row in rows {
            for (i, cell) in row.iter().enumerate() {
                if i < col_count {
                    widths[i] = widths[i].max(cell.len());
                }
            }
        }

        let mut out = String::new();
        // Header row
        for (i, hdr) in headers.iter().enumerate() {
            if i > 0 {
                out.push_str(" | ");
            }
            let _ = write!(out, "{:<width$}", hdr, width = widths[i]);
        }
        out.push('\n');
        // Separator
        for (i, w) in widths.iter().enumerate() {
            if i > 0 {
                out.push_str("-+-");
            }
            for _ in 0..*w {
                out.push('-');
            }
        }
        out.push('\n');
        // Data rows
        for row in rows {
            for (i, cell) in row.iter().enumerate() {
                if i >= col_count {
                    break;
                }
                if i > 0 {
                    out.push_str(" | ");
                }
                let _ = write!(out, "{:<width$}", cell, width = widths[i]);
            }
            out.push('\n');
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Progress reporter
// ---------------------------------------------------------------------------

/// Emits real-time progress updates to stderr for CLI feedback.
pub struct ProgressReporter {
    stage_start: Option<std::time::Instant>,
    verbose: bool,
}

impl ProgressReporter {
    pub fn new(verbose: bool) -> Self {
        Self {
            stage_start: None,
            verbose,
        }
    }

    pub fn begin_stage(&mut self, stage: &str) {
        self.stage_start = Some(std::time::Instant::now());
        if self.verbose {
            eprintln!("[cascade-verify] > {}", stage);
        }
    }

    pub fn end_stage(&mut self, stage: &str) {
        let elapsed = self
            .stage_start
            .map(|s| s.elapsed().as_millis())
            .unwrap_or(0);
        if self.verbose {
            eprintln!("[cascade-verify] done {} ({}ms)", stage, elapsed);
        }
        self.stage_start = None;
    }

    pub fn report_finding(&self, finding: &Finding) {
        if self.verbose {
            eprintln!(
                "[cascade-verify]   {} {}",
                SeverityFormatter::plain_badge(finding.severity),
                finding.description
            );
        }
    }

    pub fn report_summary(&self, total: usize, critical: usize, errors: usize) {
        eprintln!(
            "[cascade-verify] Found {} issue(s): {} critical, {} error",
            total, critical, errors
        );
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

struct SeverityCounts {
    critical: usize,
    high: usize,
    medium: usize,
    low: usize,
    info: usize,
}

impl SeverityCounts {
    fn from_findings(findings: &[Finding]) -> Self {
        let mut s = Self {
            critical: 0,
            high: 0,
            medium: 0,
            low: 0,
            info: 0,
        };
        for f in findings {
            match f.severity {
                Severity::Critical => s.critical += 1,
                Severity::High => s.high += 1,
                Severity::Medium => s.medium += 1,
                Severity::Low => s.low += 1,
                Severity::Info => s.info += 1,
            }
        }
        s
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::{PipelineResult, PipelineStats};
    use cascade_types::report::{AnalysisReport, ReportSummary};
    use cascade_types::topology::EdgeId;

    fn sample_result() -> PipelineResult {
        let findings = vec![
            Finding {
                id: "AMP-001".into(),
                severity: Severity::Critical,
                title: "Retry amplification".into(),
                description: "retry amplification 64x on gateway -> api -> db".into(),
                evidence: vec![Evidence {
                    description: "gateway -> api factor 4x".into(),
                    value: None,
                    source: Some("envoy.yaml:10:3".into()),
                }],
                location: Location {
                    file: Some("envoy.yaml".into()),
                    service: None,
                    edge: None,
                    line: Some(10),
                    column: Some(3),
                },
                code_flow: None,
                remediation: None,
            },
            Finding {
                id: "TMO-002".into(),
                severity: Severity::Medium,
                title: "Timeout budget exceeded".into(),
                description: "timeout budget exceeded on api -> auth".into(),
                evidence: vec![],
                location: Location {
                    file: None,
                    service: None,
                    edge: None,
                    line: None,
                    column: None,
                },
                code_flow: None,
                remediation: None,
            },
        ];
        let report = AnalysisReport {
            metadata: cascade_types::report::ReportMetadata {
                tool_name: "cascade-verify".into(),
                tool_version: "0.1.0".into(),
                analysis_timestamp: chrono::Utc::now(),
                target: "test".into(),
                duration_ms: 0,
            },
            findings: findings.clone(),
            summary: ReportSummary {
                total_findings: 2,
                by_severity: std::collections::BTreeMap::new(),
                services_analyzed: 0,
                edges_analyzed: 0,
                pass: false,
            },
            raw_data: None,
        };
        let mut per_stage = HashMap::new();
        per_stage.insert("tier1_analysis".into(), 42u64);
        PipelineResult {
            findings,
            repairs: Some(vec![RepairPlan {
                id: "R1".into(),
                changes: vec![],
                actions: vec![RepairAction::ModifyRetryCount {
                    edge_id: EdgeId("gateway->api".into()),
                    new_count: 1,
                }],
                cost: 2.0,
                effectiveness: 0.0,
                description: "Reduce retry count".into(),
            }]),
            report,
            exit_code: 2,
            stats: PipelineStats {
                total_duration_ms: 100,
                per_stage_duration: per_stage,
            },
        }
    }

    fn empty_result() -> PipelineResult {
        PipelineResult {
            findings: vec![],
            repairs: None,
            report: AnalysisReport {
                metadata: cascade_types::report::ReportMetadata {
                    tool_name: "cascade-verify".into(),
                    tool_version: "0.1.0".into(),
                    analysis_timestamp: chrono::Utc::now(),
                    target: "test".into(),
                    duration_ms: 0,
                },
                findings: vec![],
                summary: ReportSummary {
                    total_findings: 0,
                    by_severity: std::collections::BTreeMap::new(),
                    services_analyzed: 0,
                    edges_analyzed: 0,
                    pass: true,
                },
                raw_data: None,
            },
            exit_code: 0,
            stats: PipelineStats::default(),
        }
    }

    #[test]
    fn test_plain_has_header() {
        let out = Reporter::format_plain(&sample_result());
        assert!(out.contains("CascadeVerify Analysis Report"));
    }

    #[test]
    fn test_plain_shows_findings() {
        let out = Reporter::format_plain(&sample_result());
        assert!(out.contains("retry amplification"));
        assert!(out.contains("[CRIT]"));
    }

    #[test]
    fn test_plain_shows_repairs() {
        let out = Reporter::format_plain(&sample_result());
        assert!(out.contains("Repair #1"));
        assert!(out.contains("retry_count"));
    }

    #[test]
    fn test_plain_shows_location() {
        let out = Reporter::format_plain(&sample_result());
        assert!(out.contains("envoy.yaml"));
    }

    #[test]
    fn test_plain_empty() {
        let out = Reporter::format_plain(&empty_result());
        assert!(out.contains("No issues found"));
    }

    #[test]
    fn test_markdown_has_heading() {
        let md = Reporter::format_markdown(&sample_result());
        assert!(md.starts_with("# CascadeVerify"));
    }

    #[test]
    fn test_markdown_has_table() {
        let md = Reporter::format_markdown(&sample_result());
        assert!(md.contains("| # | Severity"));
    }

    #[test]
    fn test_markdown_has_details() {
        let md = Reporter::format_markdown(&sample_result());
        assert!(md.contains("<details>"));
    }

    #[test]
    fn test_markdown_has_diff_block() {
        let md = Reporter::format_markdown(&sample_result());
        assert!(md.contains("```diff"));
    }

    #[test]
    fn test_markdown_empty() {
        let md = Reporter::format_markdown(&empty_result());
        assert!(md.contains("No cascade risks"));
    }

    #[test]
    fn test_json_valid() {
        let json = Reporter::format_json(&sample_result());
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(parsed.get("findings").is_some());
    }

    #[test]
    fn test_yaml_valid() {
        let yaml = Reporter::format_yaml(&sample_result());
        assert!(yaml.contains("findings:"));
    }

    #[test]
    fn test_sarif_format() {
        let sarif = Reporter::format_sarif(&sample_result());
        assert!(sarif.contains("\"version\": \"2.1.0\""));
    }

    #[test]
    fn test_junit_format() {
        let xml = Reporter::format_junit(&sample_result());
        assert!(xml.contains("<testsuites"));
    }

    #[test]
    fn test_report_format_from_str() {
        assert_eq!(ReportFormat::from_str_loose("json"), ReportFormat::Json);
        assert_eq!(ReportFormat::from_str_loose("MD"), ReportFormat::Markdown);
        assert_eq!(ReportFormat::from_str_loose("sarif"), ReportFormat::Sarif);
        assert_eq!(ReportFormat::from_str_loose("unknown"), ReportFormat::Plain);
    }

    #[test]
    fn test_file_extensions() {
        assert_eq!(ReportFormat::Plain.file_extension(), "txt");
        assert_eq!(ReportFormat::Sarif.file_extension(), "sarif.json");
        assert_eq!(ReportFormat::JUnit.file_extension(), "xml");
    }

    #[test]
    fn test_table_formatter() {
        let headers = vec!["Name", "Value"];
        let rows = vec![
            vec!["alpha".into(), "1".into()],
            vec!["beta".into(), "22".into()],
        ];
        let table = TableFormatter::format(&headers, &rows);
        assert!(table.contains("alpha"));
        assert!(table.contains("beta"));
        assert!(table.contains("---"));
    }

    #[test]
    fn test_table_formatter_alignment() {
        let headers = vec!["A", "B"];
        let rows = vec![
            vec!["short".into(), "x".into()],
            vec!["longervalue".into(), "yy".into()],
        ];
        let table = TableFormatter::format(&headers, &rows);
        let lines: Vec<&str> = table.lines().collect();
        // All data lines should have the same length
        assert_eq!(lines[0].trim_end().len(), lines[2].trim_end().len());
    }

    #[test]
    fn test_severity_badges() {
        assert_eq!(SeverityFormatter::plain_badge(Severity::Critical), "[CRIT]");
        assert!(SeverityFormatter::markdown_badge(Severity::High).contains("High"));
    }

    #[test]
    fn test_ansi_color() {
        let color = SeverityFormatter::ansi_color(Severity::Medium);
        assert!(color.contains("\x1b["));
        assert_eq!(SeverityFormatter::ansi_reset(), "\x1b[0m");
    }

    #[test]
    fn test_progress_reporter_no_panic() {
        let mut pr = ProgressReporter::new(false);
        pr.begin_stage("test");
        pr.end_stage("test");
        pr.report_summary(5, 1, 2);
    }

    #[test]
    fn test_generate_report_dispatches() {
        let r = sample_result();
        let plain = Reporter::generate_report(&r, ReportFormat::Plain);
        let json = Reporter::generate_report(&r, ReportFormat::Json);
        assert!(plain.contains("CascadeVerify"));
        assert!(json.starts_with('{'));
    }
}
