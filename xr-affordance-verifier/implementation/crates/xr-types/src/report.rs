//! Report generation types for verification output.
//!
//! Provides [`VerificationReport`] with per-element results, summary
//! statistics, compliance status, and export helpers for JSON, text,
//! and HTML formats.

use serde::{Deserialize, Serialize};
use uuid::Uuid;
use std::collections::HashMap;

use crate::accessibility::{AccessibilityResult, AccessibilityStats};
use crate::certificate::{CertificateGrade, CoverageCertificate};
use crate::error::{VerifierError, VerifierResult};
use crate::scene::InteractionType;
use crate::ElementId;

// ---------------------------------------------------------------------------
// ComplianceStandard
// ---------------------------------------------------------------------------

/// Compliance standard reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComplianceStandard {
    /// US Section 508 of the Rehabilitation Act.
    Section508,
    /// Americans with Disabilities Act, Title I.
    ADATitle1,
    /// EU Accessibility Act (Directive 2019/882).
    EUAccessibilityAct,
    /// WCAG 2.1 Level AA.
    Wcag21AA,
    /// EN 301 549 (European standard).
    EN301549,
    /// Custom / proprietary standard.
    Custom,
}

impl ComplianceStandard {
    /// Human-readable name.
    pub fn name(&self) -> &str {
        match self {
            Self::Section508 => "Section 508",
            Self::ADATitle1 => "ADA Title I",
            Self::EUAccessibilityAct => "EU Accessibility Act",
            Self::Wcag21AA => "WCAG 2.1 AA",
            Self::EN301549 => "EN 301 549",
            Self::Custom => "Custom",
        }
    }

    /// Minimum coverage required for compliance.
    pub fn min_coverage(&self) -> f64 {
        match self {
            Self::Section508 => 0.95,
            Self::ADATitle1 => 0.95,
            Self::EUAccessibilityAct => 0.90,
            Self::Wcag21AA => 0.90,
            Self::EN301549 => 0.90,
            Self::Custom => 0.80,
        }
    }

    /// All standard compliance standards.
    pub fn all() -> Vec<Self> {
        vec![
            Self::Section508,
            Self::ADATitle1,
            Self::EUAccessibilityAct,
            Self::Wcag21AA,
            Self::EN301549,
        ]
    }
}

impl std::fmt::Display for ComplianceStandard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ---------------------------------------------------------------------------
// Element result
// ---------------------------------------------------------------------------

/// Verification result for a single element.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElementReport {
    /// Element ID.
    pub element_id: ElementId,
    /// Element name.
    pub name: String,
    /// Interaction type.
    pub interaction_type: InteractionType,
    /// Per-device results.
    pub device_results: Vec<DeviceElementResult>,
    /// Overall accessibility coverage.
    pub overall_coverage: f64,
    /// Pass/fail status.
    pub status: ElementStatus,
    /// Diagnostics for this element.
    pub diagnostics: Vec<String>,
}

/// Per-device result for an element.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceElementResult {
    /// Device name.
    pub device_name: String,
    /// Accessibility statistics.
    pub stats: AccessibilityStats,
    /// Coverage achieved.
    pub coverage: f64,
    /// Pass/fail status against the target.
    pub passed: bool,
}

/// Status of an element's verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ElementStatus {
    /// All checks passed.
    Pass,
    /// Some checks failed.
    Fail,
    /// Some checks were inconclusive.
    Partial,
    /// Not yet checked.
    Pending,
}

impl std::fmt::Display for ElementStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ElementStatus::Pass => write!(f, "PASS"),
            ElementStatus::Fail => write!(f, "FAIL"),
            ElementStatus::Partial => write!(f, "PARTIAL"),
            ElementStatus::Pending => write!(f, "PENDING"),
        }
    }
}

impl ElementReport {
    /// Create from accessibility results.
    pub fn from_results(
        element_id: ElementId,
        name: impl Into<String>,
        interaction_type: InteractionType,
        results: &[AccessibilityResult],
        target_coverage: f64,
    ) -> Self {
        let device_results: Vec<DeviceElementResult> = results
            .iter()
            .map(|r| DeviceElementResult {
                device_name: r.device_name.clone(),
                stats: r.stats.clone(),
                coverage: r.stats.coverage,
                passed: r.stats.coverage >= target_coverage,
            })
            .collect();

        let overall = if device_results.is_empty() {
            0.0
        } else {
            device_results
                .iter()
                .map(|r| r.coverage)
                .fold(f64::INFINITY, f64::min)
        };

        let status = if device_results.iter().all(|r| r.passed) {
            ElementStatus::Pass
        } else if device_results.iter().any(|r| r.passed) {
            ElementStatus::Partial
        } else if device_results.is_empty() {
            ElementStatus::Pending
        } else {
            ElementStatus::Fail
        };

        Self {
            element_id,
            name: name.into(),
            interaction_type,
            device_results,
            overall_coverage: overall,
            status,
            diagnostics: Vec::new(),
        }
    }

    /// Add a diagnostic message.
    pub fn add_diagnostic(&mut self, msg: impl Into<String>) {
        self.diagnostics.push(msg.into());
    }
}

impl std::fmt::Display for ElementReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "  {} ({:?}): {} (coverage: {:.1}%)",
            self.name,
            self.interaction_type,
            self.status,
            self.overall_coverage * 100.0,
        )?;
        for dr in &self.device_results {
            write!(
                f,
                "\n    {} → {:.1}% ({})",
                dr.device_name,
                dr.coverage * 100.0,
                if dr.passed { "pass" } else { "fail" },
            )?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Summary statistics
// ---------------------------------------------------------------------------

/// Summary statistics across all elements.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportSummary {
    /// Total number of elements checked.
    pub total_elements: usize,
    /// Number passing.
    pub elements_passed: usize,
    /// Number failing.
    pub elements_failed: usize,
    /// Number partial / inconclusive.
    pub elements_partial: usize,
    /// Total number of sample checks.
    pub total_samples: usize,
    /// Overall minimum coverage.
    pub min_coverage: f64,
    /// Overall mean coverage.
    pub mean_coverage: f64,
    /// Total wall-clock time (seconds).
    pub total_time_s: f64,
    /// Certificate grade (if available).
    pub grade: Option<CertificateGrade>,
}

impl ReportSummary {
    /// Compute summary from element reports.
    pub fn from_element_reports(reports: &[ElementReport], total_time_s: f64) -> Self {
        let total_elements = reports.len();
        let elements_passed = reports
            .iter()
            .filter(|r| r.status == ElementStatus::Pass)
            .count();
        let elements_failed = reports
            .iter()
            .filter(|r| r.status == ElementStatus::Fail)
            .count();
        let elements_partial = reports
            .iter()
            .filter(|r| r.status == ElementStatus::Partial)
            .count();

        let total_samples: usize = reports
            .iter()
            .flat_map(|r| r.device_results.iter())
            .map(|dr| dr.stats.total)
            .sum();

        let coverages: Vec<f64> = reports.iter().map(|r| r.overall_coverage).collect();
        let min_coverage = coverages
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min)
            .min(1.0);
        let mean_coverage = if coverages.is_empty() {
            0.0
        } else {
            coverages.iter().sum::<f64>() / coverages.len() as f64
        };

        Self {
            total_elements,
            elements_passed,
            elements_failed,
            elements_partial,
            total_samples,
            min_coverage: if min_coverage == f64::INFINITY {
                0.0
            } else {
                min_coverage
            },
            mean_coverage,
            total_time_s,
            grade: None,
        }
    }
}

impl std::fmt::Display for ReportSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Summary:")?;
        writeln!(
            f,
            "  Elements: {} total, {} pass, {} fail, {} partial",
            self.total_elements, self.elements_passed, self.elements_failed, self.elements_partial,
        )?;
        writeln!(f, "  Samples: {}", self.total_samples)?;
        writeln!(
            f,
            "  Coverage: min={:.1}%, mean={:.1}%",
            self.min_coverage * 100.0,
            self.mean_coverage * 100.0,
        )?;
        writeln!(f, "  Time: {:.2}s", self.total_time_s)?;
        if let Some(grade) = &self.grade {
            writeln!(f, "  Grade: {grade}")?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Compliance result
// ---------------------------------------------------------------------------

/// Compliance check result for a specific standard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceResult {
    /// Standard being checked.
    pub standard: ComplianceStandard,
    /// Whether the scene is compliant.
    pub compliant: bool,
    /// Minimum coverage required.
    pub required_coverage: f64,
    /// Actual achieved coverage.
    pub achieved_coverage: f64,
    /// Elements that fail this standard.
    pub failing_elements: Vec<String>,
    /// Additional notes.
    pub notes: Vec<String>,
}

impl ComplianceResult {
    /// Check compliance for a standard against element reports.
    pub fn check(standard: ComplianceStandard, reports: &[ElementReport]) -> Self {
        let required = standard.min_coverage();
        let failing: Vec<String> = reports
            .iter()
            .filter(|r| r.overall_coverage < required)
            .map(|r| r.name.clone())
            .collect();

        let achieved = reports
            .iter()
            .map(|r| r.overall_coverage)
            .fold(f64::INFINITY, f64::min)
            .min(1.0);
        let achieved = if achieved == f64::INFINITY {
            0.0
        } else {
            achieved
        };

        Self {
            standard,
            compliant: failing.is_empty() && !reports.is_empty(),
            required_coverage: required,
            achieved_coverage: achieved,
            failing_elements: failing,
            notes: Vec::new(),
        }
    }
}

impl std::fmt::Display for ComplianceResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}: {} (required {:.0}%, achieved {:.1}%)",
            self.standard,
            if self.compliant { "COMPLIANT" } else { "NON-COMPLIANT" },
            self.required_coverage * 100.0,
            self.achieved_coverage * 100.0,
        )?;
        if !self.failing_elements.is_empty() {
            write!(f, " [{} failing]", self.failing_elements.len())?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ReportSection
// ---------------------------------------------------------------------------

/// Summary of a verification report section.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportSection {
    pub title: String,
    pub content: String,
    pub subsections: Vec<ReportSection>,
}

impl ReportSection {
    pub fn new(title: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            content: content.into(),
            subsections: Vec::new(),
        }
    }

    pub fn with_subsection(mut self, sub: ReportSection) -> Self {
        self.subsections.push(sub);
        self
    }

    fn render_text(&self, depth: usize) -> String {
        let prefix = "#".repeat(depth + 1);
        let mut out = format!("{} {}\n\n{}\n\n", prefix, self.title, self.content);
        for sub in &self.subsections {
            out.push_str(&sub.render_text(depth + 1));
        }
        out
    }

    fn render_html(&self, depth: usize) -> String {
        let h_level = (depth + 1).min(6);
        let mut out = format!(
            "<h{h}>{}</h{h}>\n<p>{}</p>\n",
            self.title,
            html_escape(&self.content),
            h = h_level,
        );
        for sub in &self.subsections {
            out.push_str(&sub.render_html(depth + 1));
        }
        out
    }
}

// ---------------------------------------------------------------------------
// VerificationReport
// ---------------------------------------------------------------------------

/// Full verification report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationReport {
    /// Report ID.
    pub id: Uuid,
    /// Scene name.
    pub scene_name: String,
    /// Scene ID.
    pub scene_id: Uuid,
    /// Timestamp (ISO-8601).
    pub timestamp: String,
    /// Per-element results.
    pub elements: Vec<ElementReport>,
    /// Summary statistics.
    pub summary: ReportSummary,
    /// Compliance results.
    pub compliance: Vec<ComplianceResult>,
    /// Certificate reference (if available).
    pub certificate_id: Option<Uuid>,
    /// Certificate grade.
    pub grade: Option<CertificateGrade>,
    /// Additional report sections.
    pub sections: Vec<ReportSection>,
    /// Metadata.
    pub metadata: HashMap<String, String>,
}

impl VerificationReport {
    /// Create a new report.
    pub fn new(
        scene_name: impl Into<String>,
        scene_id: Uuid,
        elements: Vec<ElementReport>,
        total_time_s: f64,
    ) -> Self {
        let summary = ReportSummary::from_element_reports(&elements, total_time_s);
        let compliance: Vec<ComplianceResult> = ComplianceStandard::all()
            .into_iter()
            .map(|s| ComplianceResult::check(s, &elements))
            .collect();

        let now = simple_timestamp();

        Self {
            id: Uuid::new_v4(),
            scene_name: scene_name.into(),
            scene_id,
            timestamp: now,
            elements,
            summary,
            compliance,
            certificate_id: None,
            grade: None,
            sections: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Attach a certificate.
    pub fn with_certificate(mut self, cert: &CoverageCertificate) -> Self {
        self.certificate_id = Some(cert.id);
        self.grade = Some(cert.grade);
        self.summary.grade = Some(cert.grade);
        self
    }

    /// Add a metadata entry.
    pub fn with_meta(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Add a section.
    pub fn with_section(mut self, section: ReportSection) -> Self {
        self.sections.push(section);
        self
    }

    /// Whether all compliance checks pass.
    pub fn all_compliant(&self) -> bool {
        self.compliance.iter().all(|c| c.compliant)
    }

    /// Number of failing elements.
    pub fn num_failing(&self) -> usize {
        self.summary.elements_failed
    }

    // -- Export methods --

    /// Export to JSON.
    pub fn to_json(&self) -> VerifierResult<String> {
        serde_json::to_string_pretty(self).map_err(VerifierError::from)
    }

    /// Import from JSON.
    pub fn from_json(json: &str) -> VerifierResult<Self> {
        serde_json::from_str(json).map_err(VerifierError::from)
    }

    /// Export to plain text.
    pub fn to_text(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "╔══════════════════════════════════════════╗\n\
             ║     XR Accessibility Verification Report ║\n\
             ╠══════════════════════════════════════════╣\n\
             ║ Scene: {}\n\
             ║ Time:  {}\n\
             ║ ID:    {}\n\
             ╚══════════════════════════════════════════╝\n\n",
            self.scene_name, self.timestamp, self.id,
        ));

        out.push_str(&format!("{}\n", self.summary));

        out.push_str("Elements:\n");
        for elem in &self.elements {
            out.push_str(&format!("{elem}\n"));
        }
        out.push('\n');

        out.push_str("Compliance:\n");
        for c in &self.compliance {
            out.push_str(&format!("  {c}\n"));
        }
        out.push('\n');

        for section in &self.sections {
            out.push_str(&section.render_text(0));
        }

        out
    }

    /// Export to HTML.
    pub fn to_html(&self) -> String {
        let mut out = String::from("<!DOCTYPE html>\n<html>\n<head>\n");
        out.push_str("<meta charset=\"utf-8\">\n");
        out.push_str(&format!(
            "<title>Verification Report – {}</title>\n",
            html_escape(&self.scene_name)
        ));
        out.push_str("<style>\n");
        out.push_str(
            "body { font-family: system-ui, sans-serif; max-width: 900px; margin: 2em auto; }\n\
             table { border-collapse: collapse; width: 100%; margin: 1em 0; }\n\
             th, td { border: 1px solid #ccc; padding: 0.5em; text-align: left; }\n\
             th { background: #f5f5f5; }\n\
             .pass { color: #2e7d32; font-weight: bold; }\n\
             .fail { color: #c62828; font-weight: bold; }\n\
             .partial { color: #f57f17; font-weight: bold; }\n",
        );
        out.push_str("</style>\n</head>\n<body>\n");

        out.push_str(&format!(
            "<h1>XR Accessibility Verification Report</h1>\n\
             <p>Scene: <strong>{}</strong></p>\n\
             <p>Generated: {}</p>\n\
             <p>Report ID: <code>{}</code></p>\n",
            html_escape(&self.scene_name),
            html_escape(&self.timestamp),
            self.id,
        ));

        // Summary.
        out.push_str("<h2>Summary</h2>\n<table>\n");
        out.push_str(&format!(
            "<tr><td>Elements</td><td>{}</td></tr>\n",
            self.summary.total_elements
        ));
        out.push_str(&format!(
            "<tr><td>Passed</td><td class=\"pass\">{}</td></tr>\n",
            self.summary.elements_passed
        ));
        out.push_str(&format!(
            "<tr><td>Failed</td><td class=\"fail\">{}</td></tr>\n",
            self.summary.elements_failed
        ));
        out.push_str(&format!(
            "<tr><td>Min Coverage</td><td>{:.1}%</td></tr>\n",
            self.summary.min_coverage * 100.0
        ));
        out.push_str(&format!(
            "<tr><td>Mean Coverage</td><td>{:.1}%</td></tr>\n",
            self.summary.mean_coverage * 100.0
        ));
        out.push_str(&format!(
            "<tr><td>Time</td><td>{:.2}s</td></tr>\n",
            self.summary.total_time_s
        ));
        if let Some(grade) = &self.grade {
            out.push_str(&format!(
                "<tr><td>Grade</td><td><strong>{grade}</strong></td></tr>\n"
            ));
        }
        out.push_str("</table>\n");

        // Elements.
        out.push_str("<h2>Elements</h2>\n<table>\n");
        out.push_str(
            "<tr><th>Element</th><th>Type</th><th>Coverage</th><th>Status</th></tr>\n",
        );
        for elem in &self.elements {
            let class = match elem.status {
                ElementStatus::Pass => "pass",
                ElementStatus::Fail => "fail",
                _ => "partial",
            };
            out.push_str(&format!(
                "<tr><td>{}</td><td>{:?}</td><td>{:.1}%</td><td class=\"{class}\">{}</td></tr>\n",
                html_escape(&elem.name),
                elem.interaction_type,
                elem.overall_coverage * 100.0,
                elem.status,
            ));
        }
        out.push_str("</table>\n");

        // Compliance.
        out.push_str("<h2>Compliance</h2>\n<table>\n");
        out.push_str(
            "<tr><th>Standard</th><th>Required</th><th>Achieved</th><th>Status</th></tr>\n",
        );
        for c in &self.compliance {
            let class = if c.compliant { "pass" } else { "fail" };
            out.push_str(&format!(
                "<tr><td>{}</td><td>{:.0}%</td><td>{:.1}%</td><td class=\"{class}\">{}</td></tr>\n",
                c.standard,
                c.required_coverage * 100.0,
                c.achieved_coverage * 100.0,
                if c.compliant { "COMPLIANT" } else { "NON-COMPLIANT" },
            ));
        }
        out.push_str("</table>\n");

        // Additional sections.
        for section in &self.sections {
            out.push_str(&section.render_html(1));
        }

        out.push_str("</body>\n</html>\n");
        out
    }

    /// Generate a simple SVG coverage bar chart for elements.
    pub fn to_svg_coverage_chart(&self) -> String {
        let bar_height = 30.0;
        let bar_gap = 5.0;
        let label_width = 200.0;
        let chart_width = 400.0;
        let total_width = label_width + chart_width + 60.0;
        let total_height =
            (bar_height + bar_gap) * self.elements.len() as f64 + bar_gap + 40.0;

        let mut svg = format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{total_width}\" height=\"{total_height}\">\n"
        );
        svg.push_str("<style>\n  text { font-family: system-ui, sans-serif; font-size: 12px; }\n  .bar-bg { fill: #eee; }\n  .bar-pass { fill: #4caf50; }\n  .bar-fail { fill: #f44336; }\n  .bar-partial { fill: #ff9800; }\n</style>\n");

        // Title.
        svg.push_str(&format!(
            "<text x=\"{}\" y=\"20\" text-anchor=\"middle\" font-weight=\"bold\" font-size=\"14\">Coverage by Element</text>\n",
            total_width / 2.0
        ));

        for (i, elem) in self.elements.iter().enumerate() {
            let y = 30.0 + i as f64 * (bar_height + bar_gap);
            let cov = elem.overall_coverage.min(1.0).max(0.0);
            let bar_w = cov * chart_width;

            let class = match elem.status {
                ElementStatus::Pass => "bar-pass",
                ElementStatus::Fail => "bar-fail",
                _ => "bar-partial",
            };

            // Truncate label to fit.
            let label = if elem.name.len() > 25 {
                format!("{}…", &elem.name[..24])
            } else {
                elem.name.clone()
            };

            svg.push_str(&format!(
                "<text x=\"{lx}\" y=\"{ty}\" text-anchor=\"end\">{label}</text>\n",
                lx = label_width - 5.0,
                ty = y + bar_height * 0.65,
            ));

            // Background bar.
            svg.push_str(&format!(
                "<rect class=\"bar-bg\" x=\"{lw}\" y=\"{y}\" width=\"{cw}\" height=\"{bh}\" rx=\"3\"/>\n",
                lw = label_width,
                cw = chart_width,
                bh = bar_height,
            ));

            // Coverage bar.
            if bar_w > 0.0 {
                svg.push_str(&format!(
                    "<rect class=\"{class}\" x=\"{lw}\" y=\"{y}\" width=\"{bw}\" height=\"{bh}\" rx=\"3\"/>\n",
                    lw = label_width,
                    bw = bar_w,
                    bh = bar_height,
                ));
            }

            // Percentage label.
            svg.push_str(&format!(
                "<text x=\"{tx}\" y=\"{ty}\">{pct:.1}%</text>\n",
                tx = label_width + chart_width + 5.0,
                ty = y + bar_height * 0.65,
                pct = cov * 100.0,
            ));
        }

        svg.push_str("</svg>\n");
        svg
    }
}

impl std::fmt::Display for VerificationReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_text())
    }
}

/// Simple timestamp without chrono.
fn simple_timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let dur = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = dur.as_secs();
    let days = secs / 86400;
    let years = 1970 + days / 365;
    let remaining_days = days % 365;
    let months = remaining_days / 30 + 1;
    let day = remaining_days % 30 + 1;
    let time_secs = secs % 86400;
    let hours = time_secs / 3600;
    let mins = (time_secs % 3600) / 60;
    let s = time_secs % 60;
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        years, months, day, hours, mins, s
    )
}

/// Basic HTML escaping.
fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_eid() -> ElementId {
        Uuid::parse_str("00000000-0000-0000-0000-000000000001").unwrap()
    }

    fn make_element_report(name: &str, coverage: f64) -> ElementReport {
        let status = if coverage >= 0.95 {
            ElementStatus::Pass
        } else if coverage >= 0.5 {
            ElementStatus::Partial
        } else {
            ElementStatus::Fail
        };

        ElementReport {
            element_id: Uuid::new_v4(),
            name: name.into(),
            interaction_type: InteractionType::Click,
            device_results: vec![DeviceElementResult {
                device_name: "Quest 3".into(),
                stats: AccessibilityStats {
                    total: 100,
                    accessible: (coverage * 100.0) as usize,
                    inaccessible: ((1.0 - coverage) * 100.0) as usize,
                    unknown: 0,
                    coverage,
                    mean_accessible_distance: 0.01,
                    mean_inaccessible_distance: 0.15,
                    total_time_s: 1.0,
                },
                coverage,
                passed: coverage >= 0.95,
            }],
            overall_coverage: coverage,
            status,
            diagnostics: Vec::new(),
        }
    }

    #[test]
    fn test_compliance_standard_display() {
        assert_eq!(format!("{}", ComplianceStandard::Section508), "Section 508");
        assert_eq!(format!("{}", ComplianceStandard::ADATitle1), "ADA Title I");
    }

    #[test]
    fn test_compliance_standard_coverage() {
        assert!(ComplianceStandard::Section508.min_coverage() >= 0.90);
        assert!(ComplianceStandard::Custom.min_coverage() >= 0.0);
    }

    #[test]
    fn test_compliance_standard_all() {
        let all = ComplianceStandard::all();
        assert!(all.len() >= 4);
    }

    #[test]
    fn test_element_status_display() {
        assert_eq!(format!("{}", ElementStatus::Pass), "PASS");
        assert_eq!(format!("{}", ElementStatus::Fail), "FAIL");
    }

    #[test]
    fn test_element_report_pass() {
        let er = make_element_report("Button A", 1.0);
        assert_eq!(er.status, ElementStatus::Pass);
        assert!((er.overall_coverage - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_element_report_fail() {
        let er = make_element_report("Slider", 0.3);
        assert_eq!(er.status, ElementStatus::Fail);
    }

    #[test]
    fn test_element_report_display() {
        let er = make_element_report("Button A", 0.95);
        let s = format!("{er}");
        assert!(s.contains("Button A"));
        assert!(s.contains("95.0%"));
    }

    #[test]
    fn test_report_summary() {
        let reports = vec![
            make_element_report("A", 1.0),
            make_element_report("B", 0.8),
            make_element_report("C", 0.3),
        ];
        let summary = ReportSummary::from_element_reports(&reports, 5.0);
        assert_eq!(summary.total_elements, 3);
        assert_eq!(summary.elements_passed, 1);
        assert_eq!(summary.elements_failed, 1);
        assert!((summary.total_time_s - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_report_summary_display() {
        let reports = vec![make_element_report("A", 0.9)];
        let summary = ReportSummary::from_element_reports(&reports, 1.0);
        let s = format!("{summary}");
        assert!(s.contains("Summary"));
        assert!(s.contains("Elements"));
    }

    #[test]
    fn test_compliance_check_pass() {
        let reports = vec![
            make_element_report("A", 1.0),
            make_element_report("B", 0.98),
        ];
        let result = ComplianceResult::check(ComplianceStandard::Section508, &reports);
        assert!(result.compliant);
    }

    #[test]
    fn test_compliance_check_fail() {
        let reports = vec![
            make_element_report("A", 1.0),
            make_element_report("B", 0.50),
        ];
        let result = ComplianceResult::check(ComplianceStandard::Section508, &reports);
        assert!(!result.compliant);
        assert_eq!(result.failing_elements.len(), 1);
        assert_eq!(result.failing_elements[0], "B");
    }

    #[test]
    fn test_compliance_display() {
        let reports = vec![make_element_report("A", 1.0)];
        let result = ComplianceResult::check(ComplianceStandard::Section508, &reports);
        let s = format!("{result}");
        assert!(s.contains("Section 508"));
        assert!(s.contains("COMPLIANT"));
    }

    #[test]
    fn test_verification_report_creation() {
        let elements = vec![
            make_element_report("Button A", 1.0),
            make_element_report("Slider B", 0.85),
        ];
        let report = VerificationReport::new("Test Scene", Uuid::new_v4(), elements, 10.0);
        assert_eq!(report.elements.len(), 2);
        assert_eq!(report.summary.total_elements, 2);
        assert!(!report.compliance.is_empty());
    }

    #[test]
    fn test_report_json_roundtrip() {
        let elements = vec![make_element_report("Button A", 1.0)];
        let report = VerificationReport::new("Test", Uuid::new_v4(), elements, 5.0);
        let json = report.to_json().unwrap();
        let back = VerificationReport::from_json(&json).unwrap();
        assert_eq!(report.id, back.id);
        assert_eq!(report.elements.len(), back.elements.len());
    }

    #[test]
    fn test_report_to_text() {
        let elements = vec![
            make_element_report("Button A", 1.0),
            make_element_report("Slider B", 0.85),
        ];
        let report = VerificationReport::new("Test Scene", Uuid::new_v4(), elements, 10.0);
        let text = report.to_text();
        assert!(text.contains("Test Scene"));
        assert!(text.contains("Button A"));
        assert!(text.contains("Slider B"));
        assert!(text.contains("Compliance"));
    }

    #[test]
    fn test_report_to_html() {
        let elements = vec![
            make_element_report("Button A", 1.0),
            make_element_report("Slider B", 0.60),
        ];
        let report = VerificationReport::new("Test Scene", Uuid::new_v4(), elements, 5.0);
        let html = report.to_html();
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("Test Scene"));
        assert!(html.contains("Button A"));
        assert!(html.contains("</table>"));
    }

    #[test]
    fn test_report_svg_chart() {
        let elements = vec![
            make_element_report("Button A", 1.0),
            make_element_report("Slider B", 0.70),
        ];
        let report = VerificationReport::new("Test", Uuid::new_v4(), elements, 1.0);
        let svg = report.to_svg_coverage_chart();
        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));
        assert!(svg.contains("Button A"));
    }

    #[test]
    fn test_report_all_compliant() {
        let elements = vec![
            make_element_report("A", 1.0),
            make_element_report("B", 0.99),
        ];
        let report = VerificationReport::new("Test", Uuid::new_v4(), elements, 1.0);
        assert!(report.all_compliant());
    }

    #[test]
    fn test_report_not_compliant() {
        let elements = vec![
            make_element_report("A", 1.0),
            make_element_report("B", 0.50),
        ];
        let report = VerificationReport::new("Test", Uuid::new_v4(), elements, 1.0);
        assert!(!report.all_compliant());
    }

    #[test]
    fn test_report_with_meta() {
        let report = VerificationReport::new("Test", Uuid::new_v4(), vec![], 0.0)
            .with_meta("version", "1.0");
        assert_eq!(
            report.metadata.get("version").map(|s| s.as_str()),
            Some("1.0")
        );
    }

    #[test]
    fn test_report_with_section() {
        let report = VerificationReport::new("Test", Uuid::new_v4(), vec![], 0.0)
            .with_section(ReportSection::new("Notes", "Some notes here"));
        assert_eq!(report.sections.len(), 1);
    }

    #[test]
    fn test_report_display() {
        let report = VerificationReport::new("Test", Uuid::new_v4(), vec![], 0.0);
        let s = format!("{report}");
        assert!(s.contains("Test"));
    }

    #[test]
    fn test_html_escape() {
        assert_eq!(html_escape("<script>"), "&lt;script&gt;");
        assert_eq!(html_escape("a&b"), "a&amp;b");
    }

    #[test]
    fn test_section_render_text() {
        let section = ReportSection::new("Title", "Content")
            .with_subsection(ReportSection::new("Sub", "SubContent"));
        let text = section.render_text(0);
        assert!(text.contains("# Title"));
        assert!(text.contains("## Sub"));
    }

    #[test]
    fn test_num_failing() {
        let elements = vec![
            make_element_report("A", 1.0),
            make_element_report("B", 0.3),
        ];
        let report = VerificationReport::new("Test", Uuid::new_v4(), elements, 1.0);
        assert_eq!(report.num_failing(), 1);
    }

    #[test]
    fn test_element_report_add_diagnostic() {
        let mut er = make_element_report("A", 0.9);
        er.add_diagnostic("Warning: near boundary");
        assert_eq!(er.diagnostics.len(), 1);
    }
}
