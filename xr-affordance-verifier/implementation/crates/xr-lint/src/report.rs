//! Lint report: structured output with scoring and multi-format rendering.
//!
//! [`LintReport`](FullLintReport) aggregates diagnostics into a scored,
//! section-based report with text, JSON, and HTML fragment renderers.

use serde::{Deserialize, Serialize};
use uuid::Uuid;
use xr_types::error::Severity;

use crate::diagnostics::{
    group_by_element, group_by_severity, severity_counts, LintDiagnostic,
    SeverityCounts,
};

// ── Report ──────────────────────────────────────────────────────────────────

/// A full lint report with summary statistics, per-element results, and an
/// overall accessibility score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullLintReport {
    /// Scene name.
    pub scene_name: String,
    /// Overall score from 0 (many failures) to 100 (clean).
    pub score: u32,
    /// Letter grade (A–F).
    pub grade: String,
    /// Summary counts by severity.
    pub counts: SeverityCounts,
    /// Number of elements checked.
    pub elements_checked: usize,
    /// Number of rules evaluated.
    pub rules_evaluated: usize,
    /// Time spent linting (ms).
    pub elapsed_ms: f64,
    /// All diagnostics.
    pub diagnostics: Vec<LintDiagnostic>,
    /// Structured report sections.
    pub sections: Vec<ReportSection>,
}

impl FullLintReport {
    /// Build a report from a set of diagnostics.
    pub fn from_diagnostics(
        scene_name: impl Into<String>,
        diagnostics: Vec<LintDiagnostic>,
        elements_checked: usize,
        rules_evaluated: usize,
        elapsed_ms: f64,
    ) -> Self {
        let counts = severity_counts(&diagnostics);
        let score = Self::compute_score(&counts);
        let grade = Self::grade_from_score(score);
        let sections = Self::build_sections(&diagnostics);

        Self {
            scene_name: scene_name.into(),
            score,
            grade,
            counts,
            elements_checked,
            rules_evaluated,
            elapsed_ms,
            diagnostics,
            sections,
        }
    }

    /// Build from a [`LintResult`](crate::linter::LintResult).
    pub fn from_lint_result(result: &crate::linter::LintResult) -> Self {
        Self::from_diagnostics(
            &result.scene_name,
            result.diagnostics.clone(),
            result.elements_checked,
            result.rules_evaluated,
            result.elapsed_ms,
        )
    }

    /// Score: 100 minus penalties for each severity level.
    fn compute_score(counts: &SeverityCounts) -> u32 {
        let raw = 100i32
            - (counts.critical as i32 * 15)
            - (counts.error as i32 * 10)
            - (counts.warning as i32 * 2);
        raw.max(0) as u32
    }

    fn grade_from_score(score: u32) -> String {
        match score {
            90..=100 => "A".into(),
            80..=89 => "B".into(),
            70..=79 => "C".into(),
            60..=69 => "D".into(),
            _ => "F".into(),
        }
    }

    /// True if there are critical or error findings.
    pub fn has_errors(&self) -> bool {
        self.counts.has_errors()
    }

    // ── Section builder ─────────────────────────────────────────────────

    fn build_sections(diagnostics: &[LintDiagnostic]) -> Vec<ReportSection> {
        let mut sections = Vec::new();

        // Section 1: Summary
        let counts = severity_counts(diagnostics);
        sections.push(ReportSection {
            title: "Summary".into(),
            content: format!(
                "{} critical, {} errors, {} warnings, {} info — {} total findings",
                counts.critical,
                counts.error,
                counts.warning,
                counts.info,
                counts.total()
            ),
            subsections: Vec::new(),
        });

        // Section 2: By severity
        let by_sev = group_by_severity(diagnostics);
        let mut sev_subs = Vec::new();
        for &sev in &[Severity::Critical, Severity::Error, Severity::Warning, Severity::Info] {
            if let Some(items) = by_sev.get(&sev) {
                let label = match sev {
                    Severity::Critical => "Critical",
                    Severity::Error => "Errors",
                    Severity::Warning => "Warnings",
                    Severity::Info => "Info",
                };
                let detail = items
                    .iter()
                    .map(|d| {
                        let elem = d.element_name.as_deref().unwrap_or("(scene)");
                        format!("[{}] {}: {}", d.code, elem, d.message)
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                sev_subs.push(ReportSection {
                    title: format!("{} ({})", label, items.len()),
                    content: detail,
                    subsections: Vec::new(),
                });
            }
        }
        sections.push(ReportSection {
            title: "Findings by Severity".into(),
            content: String::new(),
            subsections: sev_subs,
        });

        // Section 3: By element
        let by_elem = group_by_element(diagnostics);
        let mut elem_subs = Vec::new();
        for (elem_id, items) in &by_elem {
            let name = items
                .first()
                .and_then(|d| d.element_name.as_deref())
                .unwrap_or("(scene)");
            let detail = items
                .iter()
                .map(|d| format!("[{}] {}", d.code, d.message))
                .collect::<Vec<_>>()
                .join("\n");
            elem_subs.push(ReportSection {
                title: format!(
                    "{} ({})",
                    name,
                    elem_id.map(|id| id.to_string()).unwrap_or_else(|| "scene".into())
                ),
                content: detail,
                subsections: Vec::new(),
            });
        }
        sections.push(ReportSection {
            title: "Findings by Element".into(),
            content: String::new(),
            subsections: elem_subs,
        });

        sections
    }

    // ── Output formats ──────────────────────────────────────────────────

    /// Render as human-readable text.
    pub fn to_text(&self) -> String {
        let mut out = String::with_capacity(2048);
        let border = "═".repeat(60);
        out.push_str(&border);
        out.push('\n');
        out.push_str(&format!(
            " Lint Report: {}  |  Score: {} ({})  |  {:.1}ms\n",
            self.scene_name, self.score, self.grade, self.elapsed_ms
        ));
        out.push_str(&border);
        out.push('\n');
        out.push_str(&format!(
            " {} elements checked, {} rules evaluated\n",
            self.elements_checked, self.rules_evaluated
        ));
        out.push_str(&format!(
            " {} critical, {} errors, {} warnings, {} info\n\n",
            self.counts.critical,
            self.counts.error,
            self.counts.warning,
            self.counts.info
        ));

        for section in &self.sections {
            out.push_str(&format!("── {} ──\n", section.title));
            if !section.content.is_empty() {
                out.push_str(&section.content);
                out.push('\n');
            }
            for sub in &section.subsections {
                out.push_str(&format!("  ── {} ──\n", sub.title));
                if !sub.content.is_empty() {
                    for line in sub.content.lines() {
                        out.push_str(&format!("    {}\n", line));
                    }
                }
            }
            out.push('\n');
        }
        out
    }

    /// Render as JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|e| {
            format!("{{\"error\": \"{}\"}}", e)
        })
    }

    /// Render as an HTML fragment (no <html> wrapper).
    pub fn to_html_fragment(&self) -> String {
        let mut out = String::with_capacity(4096);
        out.push_str("<div class=\"lint-report\">\n");
        out.push_str(&format!(
            "  <h2>Lint Report: {}</h2>\n",
            html_escape(&self.scene_name)
        ));
        out.push_str(&format!(
            "  <p class=\"score\">Score: <strong>{}</strong> ({})</p>\n",
            self.score, self.grade
        ));
        out.push_str(&format!(
            "  <p>{} elements checked, {} rules evaluated, {:.1}ms</p>\n",
            self.elements_checked, self.rules_evaluated, self.elapsed_ms
        ));

        out.push_str("  <table class=\"counts\">\n");
        out.push_str("    <tr><th>Critical</th><th>Error</th><th>Warning</th><th>Info</th></tr>\n");
        out.push_str(&format!(
            "    <tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>\n",
            self.counts.critical, self.counts.error, self.counts.warning, self.counts.info
        ));
        out.push_str("  </table>\n");

        if !self.diagnostics.is_empty() {
            out.push_str("  <ul class=\"diagnostics\">\n");
            for d in &self.diagnostics {
                let cls = match d.severity {
                    Severity::Critical => "critical",
                    Severity::Error => "error",
                    Severity::Warning => "warning",
                    Severity::Info => "info",
                };
                let elem = d.element_name.as_deref().unwrap_or("(scene)");
                out.push_str(&format!(
                    "    <li class=\"{}\"><code>{}</code> <strong>{}</strong>: {}</li>\n",
                    cls,
                    html_escape(&d.code),
                    html_escape(elem),
                    html_escape(&d.message),
                ));
            }
            out.push_str("  </ul>\n");
        }

        out.push_str("</div>\n");
        out
    }
}

// ── Report section ──────────────────────────────────────────────────────────

/// A section within a structured report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportSection {
    /// Section heading.
    pub title: String,
    /// Section body text.
    pub content: String,
    /// Nested subsections.
    pub subsections: Vec<ReportSection>,
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn sample_report() -> FullLintReport {
        let id = Uuid::new_v4();
        let diags = vec![
            LintDiagnostic::error("E001", "Height too low")
                .with_element(id, "btn_a"),
            LintDiagnostic::warning("W001", "Visual-only feedback")
                .with_element(id, "btn_a"),
            LintDiagnostic::critical("E010", "Cycle detected"),
        ];
        FullLintReport::from_diagnostics("test_scene", diags, 5, 12, 42.5)
    }

    #[test]
    fn test_score_computation() {
        let r = sample_report();
        // 100 - 15*1 - 10*1 - 2*1 = 73
        assert_eq!(r.score, 73);
        assert_eq!(r.grade, "C");
    }

    #[test]
    fn test_perfect_score() {
        let r = FullLintReport::from_diagnostics("clean", vec![], 10, 12, 1.0);
        assert_eq!(r.score, 100);
        assert_eq!(r.grade, "A");
        assert!(!r.has_errors());
    }

    #[test]
    fn test_zero_score() {
        let diags: Vec<LintDiagnostic> = (0..10)
            .map(|i| LintDiagnostic::critical(format!("C{:03}", i), "bad"))
            .collect();
        let r = FullLintReport::from_diagnostics("bad_scene", diags, 10, 12, 5.0);
        assert_eq!(r.score, 0); // 100 - 15*10 = -50, clamped to 0
        assert_eq!(r.grade, "F");
    }

    #[test]
    fn test_to_text() {
        let r = sample_report();
        let text = r.to_text();
        assert!(text.contains("test_scene"));
        assert!(text.contains("Score: 73"));
        assert!(text.contains("Summary"));
        assert!(text.contains("Findings by Severity"));
    }

    #[test]
    fn test_to_json_roundtrip() {
        let r = sample_report();
        let json = r.to_json();
        let parsed: FullLintReport = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.score, r.score);
        assert_eq!(parsed.diagnostics.len(), r.diagnostics.len());
    }

    #[test]
    fn test_to_html_fragment() {
        let r = sample_report();
        let html = r.to_html_fragment();
        assert!(html.starts_with("<div"));
        assert!(html.contains("lint-report"));
        assert!(html.contains("Score:"));
        assert!(html.contains("btn_a"));
        assert!(html.contains("</div>"));
    }

    #[test]
    fn test_html_escaping() {
        let diags = vec![
            LintDiagnostic::error("E001", "height < min & > max"),
        ];
        let r = FullLintReport::from_diagnostics("scene<>\"", diags, 1, 1, 1.0);
        let html = r.to_html_fragment();
        assert!(html.contains("scene&lt;&gt;&quot;"));
        assert!(html.contains("&lt; min &amp; &gt; max"));
    }

    #[test]
    fn test_sections_present() {
        let r = sample_report();
        assert!(r.sections.len() >= 3); // Summary, By Severity, By Element
        assert_eq!(r.sections[0].title, "Summary");
    }

    #[test]
    fn test_grade_boundaries() {
        assert_eq!(FullLintReport::grade_from_score(100), "A");
        assert_eq!(FullLintReport::grade_from_score(90), "A");
        assert_eq!(FullLintReport::grade_from_score(89), "B");
        assert_eq!(FullLintReport::grade_from_score(80), "B");
        assert_eq!(FullLintReport::grade_from_score(79), "C");
        assert_eq!(FullLintReport::grade_from_score(70), "C");
        assert_eq!(FullLintReport::grade_from_score(69), "D");
        assert_eq!(FullLintReport::grade_from_score(60), "D");
        assert_eq!(FullLintReport::grade_from_score(59), "F");
        assert_eq!(FullLintReport::grade_from_score(0), "F");
    }

    #[test]
    fn test_from_lint_result() {
        let result = crate::linter::LintResult {
            scene_name: "from_result".into(),
            diagnostics: vec![LintDiagnostic::warning("W001", "test")],
            elements_checked: 3,
            rules_evaluated: 5,
            counts: crate::diagnostics::SeverityCounts {
                critical: 0,
                error: 0,
                warning: 1,
                info: 0,
            },
            elapsed_ms: 10.0,
            rule_timings: HashMap::new(),
        };
        let report = FullLintReport::from_lint_result(&result);
        assert_eq!(report.scene_name, "from_result");
        assert_eq!(report.diagnostics.len(), 1);
        assert_eq!(report.score, 98); // 100 - 2*1
    }
}
