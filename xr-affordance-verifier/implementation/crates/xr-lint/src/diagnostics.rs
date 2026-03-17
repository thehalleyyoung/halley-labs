//! Lint diagnostics: rich diagnostic output with formatting and grouping.
//!
//! Provides [`LintDiagnostic`] for detailed lint findings, [`DiagnosticFormatter`]
//! for text/JSON/compact output, and [`QuickFix`] for suggested code-level repairs.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use xr_types::error::Severity;

// ── Core diagnostic ─────────────────────────────────────────────────────────

/// A rich lint diagnostic produced by a lint rule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LintDiagnostic {
    /// Severity of the finding.
    pub severity: Severity,
    /// Machine-readable rule code (e.g. "E001", "W002").
    pub code: String,
    /// Human-readable message describing the issue.
    pub message: String,
    /// The element that triggered the diagnostic, if any.
    pub element_id: Option<Uuid>,
    /// Name of the affected element.
    pub element_name: Option<String>,
    /// Estimated population percentile affected (0.0–1.0).
    /// E.g., 0.15 means 15 % of the target population is impacted.
    pub affected_percentile: Option<f64>,
    /// Devices for which the issue is relevant.
    pub affected_devices: Vec<String>,
    /// Suggested fix text.
    pub suggestion: Option<String>,
    /// Structured quick-fix, if available.
    pub quick_fix: Option<QuickFix>,
    /// Additional key-value context for tooling.
    pub context: HashMap<String, String>,
}

impl LintDiagnostic {
    /// Create a critical-severity diagnostic.
    pub fn critical(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self::new(Severity::Critical, code, message)
    }

    /// Create an error-severity diagnostic.
    pub fn error(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self::new(Severity::Error, code, message)
    }

    /// Create a warning-severity diagnostic.
    pub fn warning(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self::new(Severity::Warning, code, message)
    }

    /// Create an info-severity diagnostic.
    pub fn info(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self::new(Severity::Info, code, message)
    }

    fn new(severity: Severity, code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            severity,
            code: code.into(),
            message: message.into(),
            element_id: None,
            element_name: None,
            affected_percentile: None,
            affected_devices: Vec::new(),
            suggestion: None,
            quick_fix: None,
            context: HashMap::new(),
        }
    }

    /// Attach an element to this diagnostic.
    pub fn with_element(mut self, id: Uuid, name: impl Into<String>) -> Self {
        self.element_id = Some(id);
        self.element_name = Some(name.into());
        self
    }

    /// Set the affected population percentile.
    pub fn with_affected_percentile(mut self, pct: f64) -> Self {
        self.affected_percentile = Some(pct);
        self
    }

    /// Set the affected devices.
    pub fn with_affected_devices(mut self, devices: Vec<String>) -> Self {
        self.affected_devices = devices;
        self
    }

    /// Attach a suggestion string.
    pub fn with_suggestion(mut self, s: impl Into<String>) -> Self {
        self.suggestion = Some(s.into());
        self
    }

    /// Attach a structured quick fix.
    pub fn with_quick_fix(mut self, qf: QuickFix) -> Self {
        self.quick_fix = Some(qf);
        self
    }

    /// Add a key-value context entry.
    pub fn with_context(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.context.insert(key.into(), value.into());
        self
    }

    /// Convert to an xr-types [`Diagnostic`](xr_types::error::Diagnostic).
    pub fn to_xr_diagnostic(&self) -> xr_types::error::Diagnostic {
        let mut d = match self.severity {
            Severity::Critical => xr_types::error::Diagnostic::critical(&self.code, &self.message),
            Severity::Error => xr_types::error::Diagnostic::error(&self.code, &self.message),
            Severity::Warning => xr_types::error::Diagnostic::warning(&self.code, &self.message),
            Severity::Info => xr_types::error::Diagnostic::info(&self.code, &self.message),
        };
        if let Some(id) = self.element_id {
            d = d.with_element(id);
        }
        if let Some(ref s) = self.suggestion {
            d = d.with_suggestion(s);
        }
        d
    }

    /// True when severity is Error or Critical.
    pub fn is_error(&self) -> bool {
        matches!(self.severity, Severity::Error | Severity::Critical)
    }

    /// True when severity is Warning.
    pub fn is_warning(&self) -> bool {
        matches!(self.severity, Severity::Warning)
    }
}

// ── Quick fix ───────────────────────────────────────────────────────────────

/// A structured fix suggestion with before/after values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuickFix {
    /// Human-readable description of the fix.
    pub description: String,
    /// The property being changed (e.g. "position.y").
    pub property: String,
    /// Current value.
    pub before: String,
    /// Suggested value.
    pub after: String,
    /// Confidence in the fix (0.0–1.0).
    pub confidence: f64,
}

impl QuickFix {
    pub fn new(
        description: impl Into<String>,
        property: impl Into<String>,
        before: impl Into<String>,
        after: impl Into<String>,
        confidence: f64,
    ) -> Self {
        Self {
            description: description.into(),
            property: property.into(),
            before: before.into(),
            after: after.into(),
            confidence: confidence.clamp(0.0, 1.0),
        }
    }

    /// Create a position-move fix.
    pub fn move_position(axis: &str, from: f64, to: f64, confidence: f64) -> Self {
        Self::new(
            format!("Move element along {} axis", axis),
            format!("position.{}", axis),
            format!("{:.4}", from),
            format!("{:.4}", to),
            confidence,
        )
    }

    /// Create a volume-resize fix.
    pub fn resize_volume(from_vol: f64, to_vol: f64, confidence: f64) -> Self {
        Self::new(
            "Resize activation volume",
            "activation_volume",
            format!("{:.6} m³", from_vol),
            format!("{:.6} m³", to_vol),
            confidence,
        )
    }
}

// ── Grouping helpers ────────────────────────────────────────────────────────

/// Group diagnostics by severity.
pub fn group_by_severity(diagnostics: &[LintDiagnostic]) -> HashMap<Severity, Vec<&LintDiagnostic>> {
    let mut map: HashMap<Severity, Vec<&LintDiagnostic>> = HashMap::new();
    for d in diagnostics {
        map.entry(d.severity).or_default().push(d);
    }
    map
}

/// Group diagnostics by element id.
pub fn group_by_element(diagnostics: &[LintDiagnostic]) -> HashMap<Option<Uuid>, Vec<&LintDiagnostic>> {
    let mut map: HashMap<Option<Uuid>, Vec<&LintDiagnostic>> = HashMap::new();
    for d in diagnostics {
        map.entry(d.element_id).or_default().push(d);
    }
    map
}

/// Group diagnostics by rule code.
pub fn group_by_rule(diagnostics: &[LintDiagnostic]) -> HashMap<String, Vec<&LintDiagnostic>> {
    let mut map: HashMap<String, Vec<&LintDiagnostic>> = HashMap::new();
    for d in diagnostics {
        map.entry(d.code.clone()).or_default().push(d);
    }
    map
}

/// Count diagnostics by severity.
pub fn severity_counts(diagnostics: &[LintDiagnostic]) -> SeverityCounts {
    let mut counts = SeverityCounts::default();
    for d in diagnostics {
        match d.severity {
            Severity::Critical => counts.critical += 1,
            Severity::Error => counts.error += 1,
            Severity::Warning => counts.warning += 1,
            Severity::Info => counts.info += 1,
        }
    }
    counts
}

/// Aggregated severity counts.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct SeverityCounts {
    pub critical: usize,
    pub error: usize,
    pub warning: usize,
    pub info: usize,
}

impl SeverityCounts {
    pub fn total(&self) -> usize {
        self.critical + self.error + self.warning + self.info
    }

    pub fn has_errors(&self) -> bool {
        self.critical > 0 || self.error > 0
    }
}

// ── Formatter ───────────────────────────────────────────────────────────────

/// Formats diagnostics into text, JSON, or compact representations.
pub struct DiagnosticFormatter;

impl DiagnosticFormatter {
    /// Format diagnostics as human-readable text.
    pub fn to_text(diagnostics: &[LintDiagnostic]) -> String {
        if diagnostics.is_empty() {
            return "No lint issues found. ✓\n".to_string();
        }
        let mut out = String::with_capacity(diagnostics.len() * 120);
        let counts = severity_counts(diagnostics);

        out.push_str(&format!(
            "Lint results: {} critical, {} error, {} warning, {} info ({} total)\n",
            counts.critical,
            counts.error,
            counts.warning,
            counts.info,
            counts.total(),
        ));
        out.push_str(&"─".repeat(72));
        out.push('\n');

        let grouped = group_by_severity(diagnostics);
        for &sev in &[Severity::Critical, Severity::Error, Severity::Warning, Severity::Info] {
            if let Some(items) = grouped.get(&sev) {
                for d in items {
                    out.push_str(&Self::format_one_text(d));
                }
            }
        }
        out
    }

    fn format_one_text(d: &LintDiagnostic) -> String {
        let sev_str = match d.severity {
            Severity::Critical => "CRIT",
            Severity::Error => "ERR ",
            Severity::Warning => "WARN",
            Severity::Info => "INFO",
        };
        let elem = d
            .element_name
            .as_deref()
            .unwrap_or("(scene)");
        let mut line = format!("[{}] {} | {}: {}\n", sev_str, d.code, elem, d.message);
        if let Some(pct) = d.affected_percentile {
            line.push_str(&format!("       Affected population: {:.1}%\n", pct * 100.0));
        }
        if !d.affected_devices.is_empty() {
            line.push_str(&format!(
                "       Affected devices: {}\n",
                d.affected_devices.join(", ")
            ));
        }
        if let Some(ref s) = d.suggestion {
            line.push_str(&format!("       Suggestion: {}\n", s));
        }
        if let Some(ref qf) = d.quick_fix {
            line.push_str(&format!(
                "       Fix ({}): {} → {} [confidence {:.0}%]\n",
                qf.property,
                qf.before,
                qf.after,
                qf.confidence * 100.0,
            ));
        }
        line
    }

    /// Format diagnostics as a JSON array.
    pub fn to_json(diagnostics: &[LintDiagnostic]) -> String {
        serde_json::to_string_pretty(diagnostics).unwrap_or_else(|e| {
            format!("{{\"error\": \"serialization failed: {}\"}}", e)
        })
    }

    /// Format diagnostics as a compact single-line-per-issue list.
    pub fn to_compact(diagnostics: &[LintDiagnostic]) -> String {
        let mut out = String::with_capacity(diagnostics.len() * 80);
        for d in diagnostics {
            let sev = match d.severity {
                Severity::Critical => "C",
                Severity::Error => "E",
                Severity::Warning => "W",
                Severity::Info => "I",
            };
            let elem = d.element_name.as_deref().unwrap_or("-");
            out.push_str(&format!("{} {} {} {}\n", sev, d.code, elem, d.message));
        }
        out
    }

    /// Format diagnostics grouped by element.
    pub fn to_element_grouped_text(diagnostics: &[LintDiagnostic]) -> String {
        let grouped = group_by_element(diagnostics);
        let mut out = String::new();

        // Scene-level diagnostics first
        if let Some(scene_diags) = grouped.get(&None) {
            out.push_str("Scene-level issues:\n");
            for d in scene_diags {
                out.push_str(&Self::format_one_text(d));
            }
            out.push('\n');
        }

        // Per-element diagnostics
        let mut element_groups: Vec<_> = grouped
            .iter()
            .filter(|(k, _)| k.is_some())
            .collect();
        element_groups.sort_by_key(|(_, v)| std::cmp::Reverse(v.len()));

        for (elem_id, diags) in element_groups {
            let name = diags
                .first()
                .and_then(|d| d.element_name.as_deref())
                .unwrap_or("unknown");
            out.push_str(&format!(
                "Element '{}' ({}) — {} issue(s):\n",
                name,
                elem_id.map(|id| id.to_string()).unwrap_or_default(),
                diags.len(),
            ));
            for d in diags {
                out.push_str(&Self::format_one_text(d));
            }
            out.push('\n');
        }
        out
    }

    /// Format diagnostics grouped by rule.
    pub fn to_rule_grouped_text(diagnostics: &[LintDiagnostic]) -> String {
        let grouped = group_by_rule(diagnostics);
        let mut out = String::new();

        let mut rules: Vec<_> = grouped.iter().collect();
        rules.sort_by_key(|(code, _)| code.clone());

        for (code, diags) in rules {
            out.push_str(&format!("Rule {} — {} finding(s):\n", code, diags.len()));
            for d in diags {
                let elem = d.element_name.as_deref().unwrap_or("(scene)");
                out.push_str(&format!("  {}: {}\n", elem, d.message));
            }
            out.push('\n');
        }
        out
    }
}

// ── Severity filter ─────────────────────────────────────────────────────────

/// Filter diagnostics to at least the given severity.
pub fn filter_min_severity(diagnostics: &[LintDiagnostic], min: Severity) -> Vec<&LintDiagnostic> {
    diagnostics.iter().filter(|d| d.severity >= min).collect()
}

/// Return only diagnostics that affect a given element.
pub fn for_element(diagnostics: &[LintDiagnostic], element_id: Uuid) -> Vec<&LintDiagnostic> {
    diagnostics
        .iter()
        .filter(|d| d.element_id == Some(element_id))
        .collect()
}

/// Merge two diagnostic lists, de-duplicating by (code, element_id).
pub fn merge_unique(a: &[LintDiagnostic], b: &[LintDiagnostic]) -> Vec<LintDiagnostic> {
    let mut seen = std::collections::HashSet::new();
    let mut result = Vec::with_capacity(a.len() + b.len());
    for d in a.iter().chain(b.iter()) {
        let key = (d.code.clone(), d.element_id);
        if seen.insert(key) {
            result.push(d.clone());
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_diagnostics() -> Vec<LintDiagnostic> {
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        vec![
            LintDiagnostic::error("E001", "Height too low")
                .with_element(id1, "button_a")
                .with_suggestion("Move to 1.2m")
                .with_affected_percentile(0.35),
            LintDiagnostic::warning("W001", "No haptic feedback")
                .with_element(id1, "button_a"),
            LintDiagnostic::error("E003", "Elements too close")
                .with_element(id2, "slider_b")
                .with_affected_devices(vec!["Quest 3".into()]),
            LintDiagnostic::critical("E010", "Scene dependency cycle"),
            LintDiagnostic::info("I001", "Scene loaded"),
        ]
    }

    #[test]
    fn test_severity_counts() {
        let diags = sample_diagnostics();
        let counts = severity_counts(&diags);
        assert_eq!(counts.critical, 1);
        assert_eq!(counts.error, 2);
        assert_eq!(counts.warning, 1);
        assert_eq!(counts.info, 1);
        assert_eq!(counts.total(), 5);
        assert!(counts.has_errors());
    }

    #[test]
    fn test_group_by_severity() {
        let diags = sample_diagnostics();
        let groups = group_by_severity(&diags);
        assert_eq!(groups.get(&Severity::Error).map(|v| v.len()), Some(2));
        assert_eq!(groups.get(&Severity::Critical).map(|v| v.len()), Some(1));
    }

    #[test]
    fn test_group_by_element() {
        let diags = sample_diagnostics();
        let groups = group_by_element(&diags);
        // Two diagnostics have no element (scene-level)
        let scene_count = groups.get(&None).map(|v| v.len()).unwrap_or(0);
        assert_eq!(scene_count, 2);
    }

    #[test]
    fn test_group_by_rule() {
        let diags = sample_diagnostics();
        let groups = group_by_rule(&diags);
        assert!(groups.contains_key("E001"));
        assert!(groups.contains_key("W001"));
    }

    #[test]
    fn test_format_text() {
        let diags = sample_diagnostics();
        let text = DiagnosticFormatter::to_text(&diags);
        assert!(text.contains("E001"));
        assert!(text.contains("button_a"));
        assert!(text.contains("CRIT"));
        assert!(text.contains("35.0%"));
    }

    #[test]
    fn test_format_json() {
        let diags = sample_diagnostics();
        let json = DiagnosticFormatter::to_json(&diags);
        let parsed: Vec<LintDiagnostic> = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.len(), diags.len());
    }

    #[test]
    fn test_format_compact() {
        let diags = sample_diagnostics();
        let compact = DiagnosticFormatter::to_compact(&diags);
        let lines: Vec<_> = compact.lines().collect();
        assert_eq!(lines.len(), 5);
        assert!(lines[0].starts_with('E'));
    }

    #[test]
    fn test_format_empty() {
        let text = DiagnosticFormatter::to_text(&[]);
        assert!(text.contains("No lint issues"));
    }

    #[test]
    fn test_quick_fix() {
        let qf = QuickFix::move_position("y", 0.1, 1.2, 0.85);
        assert!(qf.description.contains("y axis"));
        assert_eq!(qf.confidence, 0.85);
    }

    #[test]
    fn test_quick_fix_confidence_clamp() {
        let qf = QuickFix::new("test", "p", "a", "b", 1.5);
        assert_eq!(qf.confidence, 1.0);
    }

    #[test]
    fn test_merge_unique() {
        let id = Uuid::new_v4();
        let a = vec![LintDiagnostic::error("E001", "msg").with_element(id, "x")];
        let b = vec![
            LintDiagnostic::error("E001", "msg").with_element(id, "x"),
            LintDiagnostic::warning("W001", "other").with_element(id, "x"),
        ];
        let merged = merge_unique(&a, &b);
        assert_eq!(merged.len(), 2);
    }

    #[test]
    fn test_filter_min_severity() {
        let diags = sample_diagnostics();
        let errors = filter_min_severity(&diags, Severity::Error);
        assert_eq!(errors.len(), 3); // 1 critical + 2 error
    }

    #[test]
    fn test_for_element() {
        let id = Uuid::new_v4();
        let diags = vec![
            LintDiagnostic::error("E001", "a").with_element(id, "btn"),
            LintDiagnostic::error("E002", "b"),
            LintDiagnostic::warning("W001", "c").with_element(id, "btn"),
        ];
        let filtered = for_element(&diags, id);
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_to_xr_diagnostic() {
        let d = LintDiagnostic::error("E001", "test message")
            .with_element(Uuid::new_v4(), "btn")
            .with_suggestion("fix it");
        let xr_d = d.to_xr_diagnostic();
        assert_eq!(xr_d.code, "E001");
        assert!(xr_d.suggestion.is_some());
    }

    #[test]
    fn test_element_grouped_text() {
        let diags = sample_diagnostics();
        let text = DiagnosticFormatter::to_element_grouped_text(&diags);
        assert!(text.contains("Scene-level issues"));
        assert!(text.contains("button_a"));
    }

    #[test]
    fn test_rule_grouped_text() {
        let diags = sample_diagnostics();
        let text = DiagnosticFormatter::to_rule_grouped_text(&diags);
        assert!(text.contains("Rule E001"));
        assert!(text.contains("finding(s)"));
    }
}
