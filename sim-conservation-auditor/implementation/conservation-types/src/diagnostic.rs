//! Diagnostic types for reporting conservation analysis results.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::collections::HashMap;

/// A single diagnostic message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Diagnostic {
    pub id: String,
    pub level: DiagnosticLevel,
    pub message: String,
    pub detail: Option<String>,
    pub location: Option<DiagnosticLocation>,
    pub related: Vec<RelatedInfo>,
    pub fix_suggestions: Vec<FixSuggestion>,
    pub conservation_law: Option<String>,
    pub violation_order: Option<usize>,
    pub violation_magnitude: Option<f64>,
    pub confidence: f64,
    pub tier: AnalysisTier,
}

impl Diagnostic {
    pub fn new(level: DiagnosticLevel, message: impl Into<String>) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            level,
            message: message.into(),
            detail: None,
            location: None,
            related: Vec::new(),
            fix_suggestions: Vec::new(),
            conservation_law: None,
            violation_order: None,
            violation_magnitude: None,
            confidence: 1.0,
            tier: AnalysisTier::Static,
        }
    }

    pub fn error(msg: impl Into<String>) -> Self { Self::new(DiagnosticLevel::Error, msg) }
    pub fn warning(msg: impl Into<String>) -> Self { Self::new(DiagnosticLevel::Warning, msg) }
    pub fn info(msg: impl Into<String>) -> Self { Self::new(DiagnosticLevel::Info, msg) }

    pub fn with_detail(mut self, detail: impl Into<String>) -> Self {
        self.detail = Some(detail.into()); self
    }

    pub fn with_location(mut self, loc: DiagnosticLocation) -> Self {
        self.location = Some(loc); self
    }

    pub fn with_conservation_law(mut self, law: impl Into<String>) -> Self {
        self.conservation_law = Some(law.into()); self
    }

    pub fn with_violation(mut self, order: usize, magnitude: f64) -> Self {
        self.violation_order = Some(order);
        self.violation_magnitude = Some(magnitude);
        self
    }

    pub fn with_confidence(mut self, c: f64) -> Self {
        self.confidence = c; self
    }

    pub fn with_tier(mut self, tier: AnalysisTier) -> Self {
        self.tier = tier; self
    }

    pub fn add_fix(mut self, fix: FixSuggestion) -> Self {
        self.fix_suggestions.push(fix); self
    }

    pub fn add_related(mut self, info: RelatedInfo) -> Self {
        self.related.push(info); self
    }
}

impl fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let icon = match self.level {
            DiagnosticLevel::Error => "✗",
            DiagnosticLevel::Warning => "⚠",
            DiagnosticLevel::Info => "ℹ",
            DiagnosticLevel::Hint => "💡",
        };
        write!(f, "{} [{}] {}", icon, self.level, self.message)?;
        if let Some(ref loc) = self.location {
            write!(f, "\n  at {}", loc)?;
        }
        if let Some(ref detail) = self.detail {
            write!(f, "\n  {}", detail)?;
        }
        if let Some(ref law) = self.conservation_law {
            write!(f, "\n  Conservation law: {}", law)?;
        }
        if let (Some(order), Some(mag)) = (self.violation_order, self.violation_magnitude) {
            write!(f, "\n  Violation: O(h^{}) magnitude {:.2e}", order, mag)?;
        }
        Ok(())
    }
}

/// Severity level of a diagnostic.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum DiagnosticLevel {
    Error,
    Warning,
    Info,
    Hint,
}

impl fmt::Display for DiagnosticLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DiagnosticLevel::Error => write!(f, "error"),
            DiagnosticLevel::Warning => write!(f, "warning"),
            DiagnosticLevel::Info => write!(f, "info"),
            DiagnosticLevel::Hint => write!(f, "hint"),
        }
    }
}

/// Location of a diagnostic in source code.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticLocation {
    pub file: String,
    pub line: usize,
    pub column: Option<usize>,
    pub end_line: Option<usize>,
    pub end_column: Option<usize>,
    pub snippet: Option<String>,
}

impl DiagnosticLocation {
    pub fn new(file: impl Into<String>, line: usize) -> Self {
        Self { file: file.into(), line, column: None, end_line: None, end_column: None, snippet: None }
    }

    pub fn with_column(mut self, col: usize) -> Self { self.column = Some(col); self }

    pub fn with_range(mut self, end_line: usize, end_col: usize) -> Self {
        self.end_line = Some(end_line);
        self.end_column = Some(end_col);
        self
    }

    pub fn with_snippet(mut self, snippet: impl Into<String>) -> Self {
        self.snippet = Some(snippet.into()); self
    }
}

impl fmt::Display for DiagnosticLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.file, self.line)?;
        if let Some(col) = self.column { write!(f, ":{}", col)?; }
        Ok(())
    }
}

/// Related information for a diagnostic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelatedInfo {
    pub message: String,
    pub location: Option<DiagnosticLocation>,
}

/// A suggested fix for a diagnostic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixSuggestion {
    pub description: String,
    pub replacement: Option<CodeReplacement>,
    pub is_architectural: bool,
}

/// A code replacement suggestion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeReplacement {
    pub location: DiagnosticLocation,
    pub old_text: String,
    pub new_text: String,
}

/// Which analysis tier produced this diagnostic.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum AnalysisTier {
    Static,
    Dynamic,
    Hybrid,
}

/// A complete diagnostic report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticReport {
    pub project_name: String,
    pub timestamp: String,
    pub diagnostics: Vec<Diagnostic>,
    pub summary: ReportSummary,
    pub metadata: HashMap<String, String>,
}

impl DiagnosticReport {
    pub fn new(project_name: impl Into<String>) -> Self {
        Self {
            project_name: project_name.into(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            diagnostics: Vec::new(),
            summary: ReportSummary::default(),
            metadata: HashMap::new(),
        }
    }

    pub fn add_diagnostic(&mut self, diag: Diagnostic) {
        match diag.level {
            DiagnosticLevel::Error => self.summary.error_count += 1,
            DiagnosticLevel::Warning => self.summary.warning_count += 1,
            DiagnosticLevel::Info => self.summary.info_count += 1,
            DiagnosticLevel::Hint => self.summary.hint_count += 1,
        }
        self.diagnostics.push(diag);
    }

    pub fn errors(&self) -> Vec<&Diagnostic> {
        self.diagnostics.iter().filter(|d| d.level == DiagnosticLevel::Error).collect()
    }

    pub fn warnings(&self) -> Vec<&Diagnostic> {
        self.diagnostics.iter().filter(|d| d.level == DiagnosticLevel::Warning).collect()
    }

    pub fn has_errors(&self) -> bool { self.summary.error_count > 0 }

    pub fn to_sarif(&self) -> String {
        let mut sarif = serde_json::json!({
            "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
            "version": "2.1.0",
            "runs": [{
                "tool": {
                    "driver": {
                        "name": "ConservationLint",
                        "version": env!("CARGO_PKG_VERSION"),
                        "informationUri": "https://github.com/conservation-lint"
                    }
                },
                "results": self.diagnostics.iter().map(|d| {
                    let mut result = serde_json::json!({
                        "ruleId": format!("conservation/{}", d.conservation_law.as_deref().unwrap_or("unknown")),
                        "level": match d.level {
                            DiagnosticLevel::Error => "error",
                            DiagnosticLevel::Warning => "warning",
                            _ => "note",
                        },
                        "message": { "text": d.message.clone() }
                    });
                    if let Some(ref loc) = d.location {
                        result["locations"] = serde_json::json!([{
                            "physicalLocation": {
                                "artifactLocation": { "uri": loc.file.clone() },
                                "region": { "startLine": loc.line }
                            }
                        }]);
                    }
                    result
                }).collect::<Vec<_>>()
            }]
        });
        serde_json::to_string_pretty(&sarif).unwrap_or_default()
    }
}

/// Summary of a diagnostic report.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReportSummary {
    pub error_count: usize,
    pub warning_count: usize,
    pub info_count: usize,
    pub hint_count: usize,
    pub laws_checked: usize,
    pub laws_preserved: usize,
    pub laws_violated: usize,
    pub analysis_time_ms: u64,
}

impl fmt::Display for ReportSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Conservation Analysis: {} errors, {} warnings, {} info\n", self.error_count, self.warning_count, self.info_count)?;
        write!(f, "Laws: {}/{} preserved, {} violated", self.laws_preserved, self.laws_checked, self.laws_violated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diagnostic_creation() {
        let d = Diagnostic::error("Energy not conserved")
            .with_detail("Leapfrog step breaks energy symmetry")
            .with_location(DiagnosticLocation::new("sim.py", 42))
            .with_conservation_law("energy")
            .with_violation(2, 1e-3);
        assert_eq!(d.level, DiagnosticLevel::Error);
        assert!(d.violation_magnitude.unwrap() > 0.0);
    }

    #[test]
    fn test_diagnostic_report() {
        let mut report = DiagnosticReport::new("test_project");
        report.add_diagnostic(Diagnostic::error("test error"));
        report.add_diagnostic(Diagnostic::warning("test warning"));
        assert!(report.has_errors());
        assert_eq!(report.errors().len(), 1);
        assert_eq!(report.warnings().len(), 1);
    }

    #[test]
    fn test_report_summary_display() {
        let summary = ReportSummary {
            error_count: 2,
            warning_count: 3,
            info_count: 1,
            hint_count: 0,
            laws_checked: 5,
            laws_preserved: 3,
            laws_violated: 2,
            analysis_time_ms: 150,
        };
        let s = format!("{}", summary);
        assert!(s.contains("2 errors"));
        assert!(s.contains("3 warnings"));
    }
}
