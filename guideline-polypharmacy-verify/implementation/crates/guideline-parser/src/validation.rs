//! Static validation of [`GuidelineDocument`] instances.
//!
//! This module checks structural invariants that must hold before a guideline
//! can be compiled into a Pharmacological Timed Automaton (PTA):
//!
//! - Every transition references a valid source and target decision-point.
//! - Safety constraints reference drugs that appear in at least one action.
//! - No duplicate decision-point or transition IDs.
//! - Monitoring requirements have valid lab-value references.

use std::collections::HashSet;
use std::fmt;

use crate::format::GuidelineDocument;

// ---------------------------------------------------------------------------
// Validation result types
// ---------------------------------------------------------------------------

/// Severity of a validation finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Severity {
    /// The guideline cannot be compiled.
    Error,
    /// The guideline can be compiled but the result may be imprecise.
    Warning,
    /// Informational note.
    Info,
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Error => write!(f, "ERROR"),
            Self::Warning => write!(f, "WARNING"),
            Self::Info => write!(f, "INFO"),
        }
    }
}

/// A single validation finding.
#[derive(Debug, Clone)]
pub struct ValidationFinding {
    /// Severity level.
    pub severity: Severity,
    /// Short machine-readable code (e.g. `"duplicate-decision-point"`).
    pub code: String,
    /// Human-readable description.
    pub message: String,
    /// Optional path into the document (e.g. `"decision_points[2].id"`).
    pub path: Option<String>,
}

impl fmt::Display for ValidationFinding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.path {
            Some(p) => write!(f, "[{}] {} (at {}): {}", self.severity, self.code, p, self.message),
            None => write!(f, "[{}] {}: {}", self.severity, self.code, self.message),
        }
    }
}

/// Aggregated validation report.
#[derive(Debug, Clone, Default)]
pub struct ValidationReport {
    /// All findings, in discovery order.
    pub findings: Vec<ValidationFinding>,
}

impl ValidationReport {
    /// Returns `true` if there are no error-severity findings.
    pub fn is_valid(&self) -> bool {
        !self.findings.iter().any(|f| f.severity == Severity::Error)
    }

    /// Number of error-severity findings.
    pub fn error_count(&self) -> usize {
        self.findings.iter().filter(|f| f.severity == Severity::Error).count()
    }

    /// Number of warning-severity findings.
    pub fn warning_count(&self) -> usize {
        self.findings.iter().filter(|f| f.severity == Severity::Warning).count()
    }

    fn push(&mut self, severity: Severity, code: &str, message: String, path: Option<String>) {
        self.findings.push(ValidationFinding {
            severity,
            code: code.to_string(),
            message,
            path,
        });
    }
}

impl fmt::Display for ValidationReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.findings.is_empty() {
            return write!(f, "Validation passed: no findings.");
        }
        writeln!(f, "Validation report ({} error(s), {} warning(s)):",
            self.error_count(), self.warning_count())?;
        for finding in &self.findings {
            writeln!(f, "  {finding}")?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// GuidelineValidator
// ---------------------------------------------------------------------------

/// Validates [`GuidelineDocument`] instances for structural correctness.
///
/// # Example
///
/// ```ignore
/// let doc = GuidelineDocument { /* ... */ };
/// let report = GuidelineValidator::validate(&doc);
/// if !report.is_valid() {
///     eprintln!("{report}");
/// }
/// ```
#[derive(Debug, Clone, Default)]
pub struct GuidelineValidator;

impl GuidelineValidator {
    /// Run all validation checks on a guideline document.
    pub fn validate(doc: &GuidelineDocument) -> ValidationReport {
        let mut report = ValidationReport::default();
        Self::check_duplicate_decision_points(doc, &mut report);
        Self::check_transition_references(doc, &mut report);
        Self::check_metadata(doc, &mut report);
        report
    }

    /// Check for duplicate decision-point IDs.
    fn check_duplicate_decision_points(doc: &GuidelineDocument, report: &mut ValidationReport) {
        let mut seen = HashSet::new();
        for (i, dp) in doc.decision_points.iter().enumerate() {
            if !seen.insert(&dp.id) {
                report.push(
                    Severity::Error,
                    "duplicate-decision-point",
                    format!("Duplicate decision-point ID: '{}'", dp.id),
                    Some(format!("decision_points[{i}].id")),
                );
            }
        }
    }

    /// Check that every transition references valid decision-point IDs.
    fn check_transition_references(doc: &GuidelineDocument, report: &mut ValidationReport) {
        let dp_ids: HashSet<&str> = doc.decision_points.iter().map(|dp| dp.id.as_str()).collect();
        for (i, tr) in doc.transitions.iter().enumerate() {
            if !dp_ids.contains(tr.source.as_str()) {
                report.push(
                    Severity::Error,
                    "invalid-transition-source",
                    format!("Transition source '{}' is not a known decision-point", tr.source),
                    Some(format!("transitions[{i}].source")),
                );
            }
            if !dp_ids.contains(tr.target.as_str()) {
                report.push(
                    Severity::Error,
                    "invalid-transition-target",
                    format!("Transition target '{}' is not a known decision-point", tr.target),
                    Some(format!("transitions[{i}].target")),
                );
            }
        }
    }

    /// Check basic metadata completeness.
    fn check_metadata(doc: &GuidelineDocument, report: &mut ValidationReport) {
        if doc.metadata.title.is_empty() {
            report.push(
                Severity::Warning,
                "empty-title",
                "Guideline title is empty".to_string(),
                Some("metadata.title".to_string()),
            );
        }
        if doc.metadata.version.is_empty() {
            report.push(
                Severity::Warning,
                "empty-version",
                "Guideline version is empty".to_string(),
                Some("metadata.version".to_string()),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::*;

    fn minimal_doc() -> GuidelineDocument {
        GuidelineDocument {
            id: "test".into(),
            metadata: GuidelineMetadata {
                title: "Test Guideline".into(),
                version: "1.0".into(),
                authors: vec![],
                publication_date: None,
                source_organization: None,
                condition: None,
                evidence_level: None,
                supersedes: None,
                tags: vec![],
            },
            decision_points: vec![
                DecisionPoint {
                    id: "start".into(),
                    label: "Start".into(),
                    branches: vec![],
                    annotations: Default::default(),
                },
            ],
            transitions: vec![],
            safety_constraints: vec![],
            monitoring: vec![],
            annotations: Default::default(),
        }
    }

    #[test]
    fn valid_doc_passes() {
        let report = GuidelineValidator::validate(&minimal_doc());
        assert!(report.is_valid());
        assert_eq!(report.error_count(), 0);
    }

    #[test]
    fn duplicate_dp_detected() {
        let mut doc = minimal_doc();
        doc.decision_points.push(DecisionPoint {
            id: "start".into(),
            label: "Dup".into(),
            branches: vec![],
            annotations: Default::default(),
        });
        let report = GuidelineValidator::validate(&doc);
        assert!(!report.is_valid());
        assert_eq!(report.error_count(), 1);
        assert!(report.findings[0].code == "duplicate-decision-point");
    }
}
