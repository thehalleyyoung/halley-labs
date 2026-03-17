//! Multi-format export: JSON, CSV, Markdown, and HTML.
//!
//! Each [`Exporter`] implementation converts a [`BehavioralAtlas`] and a
//! [`LocalizationReport`] (from the [`summary`](super::summary) module) into a
//! byte-oriented output suitable for writing to disk or streaming.

use std::collections::HashMap;
use std::fmt::Write as FmtWrite;

use serde::{Deserialize, Serialize};

use crate::atlas::BehavioralAtlas;
use crate::summary::LocalizationReport;

// ── Config ──────────────────────────────────────────────────────────────────

/// Configuration for an export run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportConfig {
    /// Include per-interaction breakdown in the output.
    pub include_interactions: bool,
    /// Include raw stage differentials.
    pub include_raw_differentials: bool,
    /// Maximum number of findings to include (0 = unlimited).
    pub max_findings: usize,
    /// Output directory or prefix (informational; callers handle I/O).
    pub output_prefix: String,
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            include_interactions: true,
            include_raw_differentials: false,
            max_findings: 0,
            output_prefix: "report".into(),
        }
    }
}

// ── Exporter trait ──────────────────────────────────────────────────────────

/// Trait for converting a report + atlas into a serialised format.
pub trait Exporter {
    /// Human-readable format name (e.g. "JSON", "CSV").
    fn format_name(&self) -> &str;

    /// File extension (without leading dot).
    fn extension(&self) -> &str;

    /// Export the atlas alone.
    fn export_atlas(&self, atlas: &BehavioralAtlas, config: &ExportConfig) -> String;

    /// Export a full localization report.
    fn export_report(&self, report: &LocalizationReport, config: &ExportConfig) -> String;
}

// ── JSON exporter ───────────────────────────────────────────────────────────

/// Exports data as JSON.
#[derive(Debug, Clone, Default)]
pub struct JsonExporter {
    pub pretty: bool,
}

impl JsonExporter {
    pub fn new(pretty: bool) -> Self {
        Self { pretty }
    }
}

impl Exporter for JsonExporter {
    fn format_name(&self) -> &str {
        "JSON"
    }

    fn extension(&self) -> &str {
        "json"
    }

    fn export_atlas(&self, atlas: &BehavioralAtlas, _config: &ExportConfig) -> String {
        if self.pretty {
            serde_json::to_string_pretty(atlas).unwrap_or_default()
        } else {
            serde_json::to_string(atlas).unwrap_or_default()
        }
    }

    fn export_report(&self, report: &LocalizationReport, _config: &ExportConfig) -> String {
        if self.pretty {
            serde_json::to_string_pretty(report).unwrap_or_default()
        } else {
            serde_json::to_string(report).unwrap_or_default()
        }
    }
}

// ── CSV exporter ────────────────────────────────────────────────────────────

/// Exports stage-level data as CSV (one row per stage).
#[derive(Debug, Clone, Default)]
pub struct CsvExporter;

impl Exporter for CsvExporter {
    fn format_name(&self) -> &str {
        "CSV"
    }

    fn extension(&self) -> &str {
        "csv"
    }

    fn export_atlas(&self, atlas: &BehavioralAtlas, _config: &ExportConfig) -> String {
        let mut wtr = csv::Writer::from_writer(Vec::new());
        // Header
        let _ = wtr.write_record([
            "stage",
            "bfi",
            "interpretation",
            "suspiciousness",
            "rank",
            "violations",
            "total_tests",
        ]);
        for entry in &atlas.stages {
            let _ = wtr.write_record(&[
                entry.stage_name.clone(),
                format!("{:.6}", entry.bfi_value),
                entry.bfi_interpretation.to_string(),
                format!("{:.6}", entry.suspiciousness_score),
                entry.rank.to_string(),
                entry.coverage.violations.to_string(),
                entry.coverage.total_tests.to_string(),
            ]);
        }
        let _ = wtr.flush();
        String::from_utf8(wtr.into_inner().unwrap_or_default()).unwrap_or_default()
    }

    fn export_report(&self, report: &LocalizationReport, config: &ExportConfig) -> String {
        let mut wtr = csv::Writer::from_writer(Vec::new());
        let _ = wtr.write_record([
            "finding_id",
            "stage",
            "severity",
            "title",
            "suspiciousness",
        ]);
        let limit = if config.max_findings > 0 {
            config.max_findings
        } else {
            usize::MAX
        };
        for finding in report.findings.iter().take(limit) {
            let _ = wtr.write_record(&[
                finding.id.clone(),
                finding.stage_name.clone(),
                format!("{:?}", finding.severity),
                finding.title.clone(),
                format!("{:.6}", finding.suspiciousness),
            ]);
        }
        let _ = wtr.flush();
        String::from_utf8(wtr.into_inner().unwrap_or_default()).unwrap_or_default()
    }
}

// ── Markdown exporter ───────────────────────────────────────────────────────

/// Exports data as Markdown.
#[derive(Debug, Clone, Default)]
pub struct MarkdownExporter;

impl Exporter for MarkdownExporter {
    fn format_name(&self) -> &str {
        "Markdown"
    }

    fn extension(&self) -> &str {
        "md"
    }

    fn export_atlas(&self, atlas: &BehavioralAtlas, config: &ExportConfig) -> String {
        let mut out = String::new();
        out.push_str("# Behavioral Atlas Export\n\n");

        out.push_str("## Stages\n\n");
        out.push_str("| Stage | BFI | Suspiciousness | Rank | Violations |\n");
        out.push_str("|-------|-----|----------------|------|------------|\n");
        for e in &atlas.stages {
            let _ = writeln!(
                out,
                "| {} | {:.3} | {:.4} | {} | {}/{} |",
                e.stage_name,
                e.bfi_value,
                e.suspiciousness_score,
                e.rank,
                e.coverage.violations,
                e.coverage.total_tests,
            );
        }

        if config.include_interactions && !atlas.interactions.is_empty() {
            out.push_str("\n## Interactions\n\n");
            out.push_str("| Stage | Transformation | Mean Δ | Violations | BFI |\n");
            out.push_str("|-------|----------------|--------|------------|-----|\n");
            for i in &atlas.interactions {
                let _ = writeln!(
                    out,
                    "| {} | {} | {:.4} | {} | {:.3} |",
                    i.stage_name,
                    i.transformation_name,
                    i.mean_differential,
                    i.violation_count,
                    i.bfi_value,
                );
            }
        }
        out
    }

    fn export_report(&self, report: &LocalizationReport, config: &ExportConfig) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "# {}\n", report.header.title);
        let _ = writeln!(out, "**Pipeline:** {}\n", report.header.pipeline_name);
        let _ = writeln!(out, "**Generated:** {}\n", report.header.generated_at);

        out.push_str("## Findings\n\n");
        let limit = if config.max_findings > 0 {
            config.max_findings
        } else {
            usize::MAX
        };
        for f in report.findings.iter().take(limit) {
            let _ = writeln!(out, "### {} ({})\n", f.title, f.stage_name);
            let _ = writeln!(out, "**Severity:** {:?}  ", f.severity);
            let _ = writeln!(out, "**Suspiciousness:** {:.4}\n", f.suspiciousness);
            let _ = writeln!(out, "{}\n", f.description);
            if !f.evidence.is_empty() {
                out.push_str("**Evidence:**\n");
                for ev in &f.evidence {
                    let _ = writeln!(out, "- {}: {}", ev.label, ev.value);
                }
                out.push('\n');
            }
        }

        if !report.recommendations.is_empty() {
            out.push_str("## Recommendations\n\n");
            for r in &report.recommendations {
                let _ = writeln!(out, "- **{}** (priority {:?}): {}", r.title, r.priority, r.description);
            }
        }
        out
    }
}

// ── HTML exporter ───────────────────────────────────────────────────────────

/// Exports data as a self-contained HTML page.
#[derive(Debug, Clone, Default)]
pub struct HtmlExporter;

impl Exporter for HtmlExporter {
    fn format_name(&self) -> &str {
        "HTML"
    }

    fn extension(&self) -> &str {
        "html"
    }

    fn export_atlas(&self, atlas: &BehavioralAtlas, _config: &ExportConfig) -> String {
        let mut out = String::from(
            "<!DOCTYPE html><html><head><meta charset=\"utf-8\">\
             <title>Behavioral Atlas</title>\
             <style>body{font-family:sans-serif;margin:2em} \
             table{border-collapse:collapse;width:100%} \
             th,td{border:1px solid #ccc;padding:6px 10px;text-align:left} \
             th{background:#f5f5f5}</style></head><body>\n",
        );
        out.push_str("<h1>Behavioral Atlas</h1>\n<table><tr>");
        for h in &["Stage", "BFI", "Interpretation", "Suspiciousness", "Rank", "Violations", "Tests"] {
            let _ = write!(out, "<th>{h}</th>");
        }
        out.push_str("</tr>\n");
        for e in &atlas.stages {
            let _ = writeln!(
                out,
                "<tr><td>{}</td><td>{:.3}</td><td>{}</td><td>{:.4}</td><td>{}</td><td>{}</td><td>{}</td></tr>",
                e.stage_name, e.bfi_value, e.bfi_interpretation,
                e.suspiciousness_score, e.rank,
                e.coverage.violations, e.coverage.total_tests,
            );
        }
        out.push_str("</table></body></html>");
        out
    }

    fn export_report(&self, report: &LocalizationReport, config: &ExportConfig) -> String {
        let mut out = String::from(
            "<!DOCTYPE html><html><head><meta charset=\"utf-8\">\
             <title>Localization Report</title>\
             <style>body{font-family:sans-serif;margin:2em} \
             .finding{border:1px solid #ddd;padding:1em;margin:1em 0;border-radius:6px} \
             .severity-High,.severity-Critical{border-left:4px solid #e74c3c} \
             .severity-Medium{border-left:4px solid #f39c12} \
             .severity-Low{border-left:4px solid #3498db}</style></head><body>\n",
        );
        let _ = writeln!(out, "<h1>{}</h1>", report.header.title);
        let _ = writeln!(out, "<p>Pipeline: <strong>{}</strong></p>", report.header.pipeline_name);

        let limit = if config.max_findings > 0 {
            config.max_findings
        } else {
            usize::MAX
        };
        for f in report.findings.iter().take(limit) {
            let sev = format!("{:?}", f.severity);
            let _ = writeln!(
                out,
                "<div class=\"finding severity-{sev}\"><h3>{title} ({stage})</h3>\
                 <p><b>Severity:</b> {sev} | <b>Suspiciousness:</b> {susp:.4}</p>\
                 <p>{desc}</p></div>",
                title = f.title,
                stage = f.stage_name,
                susp = f.suspiciousness,
                desc = f.description,
            );
        }
        out.push_str("</body></html>");
        out
    }
}

// ── TemplateEngine ──────────────────────────────────────────────────────────

/// Minimal template engine that replaces `{{key}}` placeholders with values
/// from a context map.
#[derive(Debug, Clone, Default)]
pub struct TemplateEngine {
    helpers: HashMap<String, fn(&str) -> String>,
}

impl TemplateEngine {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a named helper function.
    pub fn register_helper(&mut self, name: impl Into<String>, f: fn(&str) -> String) {
        self.helpers.insert(name.into(), f);
    }

    /// Render a template string using the given context.
    ///
    /// Supports `{{key}}` for simple substitution and `{{helper:key}}` for
    /// applying a registered helper to the value.
    pub fn render(&self, template: &str, context: &HashMap<String, String>) -> String {
        let mut output = template.to_string();

        // Helper substitutions: {{helper_name:key}}
        for (helper_name, func) in &self.helpers {
            let prefix = format!("{{{{{helper_name}:");
            while let Some(start) = output.find(&prefix) {
                if let Some(rel_end) = output[start + prefix.len()..].find("}}") {
                    let end = start + prefix.len() + rel_end;
                    let key = &output[start + prefix.len()..end];
                    let value = context.get(key).map(|v| v.as_str()).unwrap_or("");
                    let replacement = func(value);
                    output.replace_range(start..end + 2, &replacement);
                } else {
                    break;
                }
            }
        }

        // Simple substitutions: {{key}}
        for (key, value) in context {
            let placeholder = format!("{{{{{key}}}}}");
            output = output.replace(&placeholder, value);
        }

        output
    }
}

// ── ExportBundle ────────────────────────────────────────────────────────────

/// A named output artifact produced by an export run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportArtifact {
    pub filename: String,
    pub format: String,
    pub content: String,
}

/// Collection of exported artifacts from a single export run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportBundle {
    pub artifacts: Vec<ExportArtifact>,
    pub metadata: HashMap<String, String>,
}

impl ExportBundle {
    /// Run all provided exporters and collect their output.
    pub fn generate(
        exporters: &[&dyn Exporter],
        atlas: &BehavioralAtlas,
        report: &LocalizationReport,
        config: &ExportConfig,
    ) -> Self {
        let mut artifacts = Vec::new();
        for exporter in exporters {
            let atlas_content = exporter.export_atlas(atlas, config);
            artifacts.push(ExportArtifact {
                filename: format!("{}_atlas.{}", config.output_prefix, exporter.extension()),
                format: exporter.format_name().to_string(),
                content: atlas_content,
            });
            let report_content = exporter.export_report(report, config);
            artifacts.push(ExportArtifact {
                filename: format!("{}_report.{}", config.output_prefix, exporter.extension()),
                format: exporter.format_name().to_string(),
                content: report_content,
            });
        }
        ExportBundle {
            artifacts,
            metadata: HashMap::new(),
        }
    }

    /// Number of artifacts in the bundle.
    pub fn len(&self) -> usize {
        self.artifacts.len()
    }

    /// Whether the bundle has no artifacts.
    pub fn is_empty(&self) -> bool {
        self.artifacts.is_empty()
    }

    /// Retrieve an artifact by filename.
    pub fn get(&self, filename: &str) -> Option<&ExportArtifact> {
        self.artifacts.iter().find(|a| a.filename == filename)
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atlas::BehavioralAtlas;
    use crate::summary::{Finding, FindingSeverity, LocalizationReport, ReportHeader};
    use std::collections::HashMap;

    fn sample_atlas() -> BehavioralAtlas {
        let stages = vec!["tok".into(), "pos".into()];
        let diffs = vec![vec![0.1, 0.2], vec![0.5, 0.6]];
        let violations = vec![vec![false, false], vec![true, true]];
        let susp = vec![0.3, 0.9];
        BehavioralAtlas::build(&stages, &diffs, &HashMap::new(), &violations, &susp)
    }

    fn sample_report() -> LocalizationReport {
        LocalizationReport {
            header: ReportHeader {
                title: "Test Report".into(),
                pipeline_name: "test-pipeline".into(),
                generated_at: "2024-01-01T00:00:00Z".into(),
                version: "0.1.0".into(),
                metadata: HashMap::new(),
            },
            findings: vec![Finding {
                id: "F-1".into(),
                stage_name: "pos".into(),
                title: "POS amplification".into(),
                description: "Stage amplifies divergence".into(),
                severity: FindingSeverity::High,
                suspiciousness: 0.9,
                evidence: vec![],
                recommendations: vec![],
            }],
            methodology: None,
            statistical_section: None,
            recommendations: vec![],
            appendices: vec![],
        }
    }

    #[test]
    fn test_json_exporter_atlas() {
        let exp = JsonExporter::new(true);
        let config = ExportConfig::default();
        let out = exp.export_atlas(&sample_atlas(), &config);
        assert!(out.contains("\"stage_name\""));
    }

    #[test]
    fn test_json_exporter_report() {
        let exp = JsonExporter::new(false);
        let config = ExportConfig::default();
        let out = exp.export_report(&sample_report(), &config);
        assert!(out.contains("Test Report"));
    }

    #[test]
    fn test_csv_exporter_atlas() {
        let exp = CsvExporter;
        let config = ExportConfig::default();
        let out = exp.export_atlas(&sample_atlas(), &config);
        assert!(out.contains("stage,bfi,interpretation"));
        assert!(out.contains("tok"));
    }

    #[test]
    fn test_csv_exporter_report() {
        let exp = CsvExporter;
        let config = ExportConfig::default();
        let out = exp.export_report(&sample_report(), &config);
        assert!(out.contains("finding_id"));
        assert!(out.contains("F-1"));
    }

    #[test]
    fn test_markdown_exporter() {
        let exp = MarkdownExporter;
        let config = ExportConfig::default();
        let out = exp.export_report(&sample_report(), &config);
        assert!(out.contains("# Test Report"));
        assert!(out.contains("POS amplification"));
    }

    #[test]
    fn test_html_exporter() {
        let exp = HtmlExporter;
        let config = ExportConfig::default();
        let out = exp.export_report(&sample_report(), &config);
        assert!(out.contains("<!DOCTYPE html>"));
        assert!(out.contains("Localization Report"));
    }

    #[test]
    fn test_template_engine_simple() {
        let engine = TemplateEngine::new();
        let mut ctx = HashMap::new();
        ctx.insert("name".into(), "POS".into());
        ctx.insert("score".into(), "0.95".into());
        let result = engine.render("Stage {{name}} scored {{score}}", &ctx);
        assert_eq!(result, "Stage POS scored 0.95");
    }

    #[test]
    fn test_template_engine_helper() {
        let mut engine = TemplateEngine::new();
        engine.register_helper("upper", |s: &str| s.to_uppercase());
        let mut ctx = HashMap::new();
        ctx.insert("name".into(), "pos".into());
        let result = engine.render("Stage: {{upper:name}}", &ctx);
        assert_eq!(result, "Stage: POS");
    }

    #[test]
    fn test_export_bundle() {
        let atlas = sample_atlas();
        let report = sample_report();
        let config = ExportConfig::default();
        let json = JsonExporter::new(false);
        let md = MarkdownExporter;
        let exporters: Vec<&dyn Exporter> = vec![&json, &md];
        let bundle = ExportBundle::generate(&exporters, &atlas, &report, &config);
        // 2 exporters × 2 artifacts each = 4
        assert_eq!(bundle.len(), 4);
        assert!(bundle.get("report_atlas.json").is_some());
        assert!(bundle.get("report_report.md").is_some());
    }

    #[test]
    fn test_max_findings_limit() {
        let exp = CsvExporter;
        let mut config = ExportConfig::default();
        config.max_findings = 0; // unlimited
        let report = sample_report();
        let out = exp.export_report(&report, &config);
        assert!(out.contains("F-1"));
    }
}
