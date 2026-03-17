//! Human-readable and machine-readable reporting for conflict detection results.
//!
//! Converts raw verification results into formatted output suitable for
//! clinicians, EHR integration, and research reporting.

use std::fmt;
use serde::{Deserialize, Serialize};

use crate::types::{
    ConfirmedConflict, ConflictSeverity, SafetyVerdict, VerificationResult,
};

// ---------------------------------------------------------------------------
// Report formats
// ---------------------------------------------------------------------------

/// Supported output formats for conflict reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReportFormat {
    /// Plain-text summary.
    PlainText,
    /// JSON for machine consumption.
    Json,
    /// Markdown for documentation.
    Markdown,
    /// HTML for web display.
    Html,
}

// ---------------------------------------------------------------------------
// ReportGenerator
// ---------------------------------------------------------------------------

/// Generates formatted reports from verification results.
#[derive(Debug, Clone)]
pub struct ReportGenerator {
    pub format: ReportFormat,
    pub include_counterexamples: bool,
    pub include_recommendations: bool,
}

impl Default for ReportGenerator {
    fn default() -> Self {
        Self {
            format: ReportFormat::PlainText,
            include_counterexamples: true,
            include_recommendations: true,
        }
    }
}

impl ReportGenerator {
    /// Create a new report generator for the specified format.
    pub fn new(format: ReportFormat) -> Self {
        Self {
            format,
            ..Default::default()
        }
    }

    /// Render a verification result into a formatted string.
    pub fn render(&self, result: &VerificationResult) -> String {
        match self.format {
            ReportFormat::PlainText => self.render_text(result),
            ReportFormat::Markdown => self.render_markdown(result),
            ReportFormat::Json => self.render_json(result),
            ReportFormat::Html => self.render_html(result),
        }
    }

    fn render_text(&self, result: &VerificationResult) -> String {
        let mut out = String::new();
        out.push_str("=== GuardPharma Verification Report ===\n\n");
        out.push_str(&format!("Drug pair: {:?}\n", result.drug_pair));
        out.push_str(&format!("Verdict: {}\n", SeverityFormatter::verdict_label(result.verdict)));
        out.push_str(&format!("Conflicts found: {}\n", result.conflicts.len()));
        for (i, c) in result.conflicts.iter().enumerate() {
            out.push_str(&format!(
                "\n  [{}] {} — {:?}\n      {}\n",
                i + 1,
                c.id,
                c.severity,
                c.mechanism_description
            ));
        }
        out
    }

    fn render_markdown(&self, result: &VerificationResult) -> String {
        let mut out = String::new();
        out.push_str("# GuardPharma Verification Report\n\n");
        out.push_str(&format!("**Drug pair:** `{:?}`  \n", result.drug_pair));
        out.push_str(&format!(
            "**Verdict:** {}  \n",
            SeverityFormatter::verdict_label(result.verdict)
        ));
        out.push_str(&format!("**Conflicts:** {}  \n\n", result.conflicts.len()));
        for (i, c) in result.conflicts.iter().enumerate() {
            out.push_str(&format!(
                "### Conflict {} — {:?}\n\n{}\n\n",
                i + 1,
                c.severity,
                c.mechanism_description
            ));
        }
        out
    }

    fn render_json(&self, result: &VerificationResult) -> String {
        serde_json::to_string_pretty(result).unwrap_or_else(|e| format!("{{\"error\": \"{e}\"}}"))
    }

    fn render_html(&self, result: &VerificationResult) -> String {
        let mut out = String::from("<html><body>\n");
        out.push_str("<h1>GuardPharma Verification Report</h1>\n");
        out.push_str(&format!("<p><b>Drug pair:</b> {:?}</p>\n", result.drug_pair));
        out.push_str(&format!(
            "<p><b>Verdict:</b> {}</p>\n",
            SeverityFormatter::verdict_label(result.verdict)
        ));
        for (i, c) in result.conflicts.iter().enumerate() {
            out.push_str(&format!(
                "<h2>Conflict {} — {:?}</h2>\n<p>{}</p>\n",
                i + 1,
                c.severity,
                c.mechanism_description
            ));
        }
        out.push_str("</body></html>");
        out
    }
}

// ---------------------------------------------------------------------------
// Formatters
// ---------------------------------------------------------------------------

/// Utility for rendering conflict severities.
#[derive(Debug, Clone, Copy)]
pub struct SeverityFormatter;

impl SeverityFormatter {
    /// Emoji label for a severity.
    pub fn emoji(sev: ConflictSeverity) -> &'static str {
        match sev {
            ConflictSeverity::Critical => "🔴",
            ConflictSeverity::Major => "🟠",
            ConflictSeverity::Moderate => "🟡",
            ConflictSeverity::Minor => "🟢",
        }
    }

    /// Human-readable verdict label.
    pub fn verdict_label(verdict: SafetyVerdict) -> &'static str {
        match verdict {
            SafetyVerdict::Safe => "✅ SAFE",
            SafetyVerdict::PossiblySafe => "✅ POSSIBLY SAFE",
            SafetyVerdict::PossiblyUnsafe => "⚠️  POSSIBLY UNSAFE",
            SafetyVerdict::Unsafe => "🚨 UNSAFE",
        }
    }
}

/// Formats a timeline of counterexample trace steps.
#[derive(Debug, Clone, Copy)]
pub struct TimelineFormatter;

impl TimelineFormatter {
    /// Render a textual timeline from trace steps.
    pub fn render(steps: &[crate::types::TraceStep]) -> String {
        let mut out = String::new();
        for step in steps {
            out.push_str(&format!(
                "  Hour {:>6.1}: {}\n",
                step.time_hours, step.state_description
            ));
        }
        out
    }
}

/// Formats a table of drugs with their dosages.
#[derive(Debug, Clone, Copy)]
pub struct DrugTableFormatter;

impl DrugTableFormatter {
    /// Render a drug table in plain text.
    pub fn render(drugs: &[(String, String, String)]) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "{:<20} {:<15} {:<20}\n",
            "Drug", "Dose", "Route"
        ));
        out.push_str(&"-".repeat(55));
        out.push('\n');
        for (name, dose, route) in drugs {
            out.push_str(&format!("{:<20} {:<15} {:<20}\n", name, dose, route));
        }
        out
    }
}
