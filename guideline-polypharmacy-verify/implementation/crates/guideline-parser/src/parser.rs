//! JSON / YAML parsing of clinical guideline documents with validation.

use crate::format::{
    Branch, ComparisonOp, ConstraintSeverity, DecisionPoint, DoseSpec, EvidenceLevel,
    GuidelineAction, GuidelineDocument, GuidelineGuard, GuidelineMetadata, MedicationSpec,
    MonitoringRequirement, SafetyConstraint, TransitionRule, Urgency,
};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use thiserror::Error;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum ParseError {
    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("YAML parse error: {0}")]
    Yaml(#[from] serde_yaml::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Validation error: {issues:?}")]
    Validation { issues: Vec<ValidationIssue> },

    #[error("Unsupported file format: {0}")]
    UnsupportedFormat(String),

    #[error("Document is empty or missing required fields")]
    EmptyDocument,
}

/// A single validation issue found during parsing or post-parse checks.
#[derive(Debug, Clone)]
pub struct ValidationIssue {
    pub severity: IssueSeverity,
    pub code: String,
    pub message: String,
    pub location: Option<String>,
}

impl std::fmt::Display for ValidationIssue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{:?}] {}: {}", self.severity, self.code, self.message)?;
        if let Some(loc) = &self.location {
            write!(f, " (at {})", loc)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IssueSeverity {
    Error,
    Warning,
    Info,
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

/// Main entry point for parsing guideline documents from JSON or YAML.
#[derive(Debug, Clone)]
pub struct GuidelineParser {
    /// If true, the parser will return an error when validation issues are found.
    pub strict: bool,
    /// Maximum allowed depth for guard expression trees.
    pub max_guard_depth: usize,
    /// Maximum number of decision points allowed.
    pub max_decision_points: usize,
}

impl Default for GuidelineParser {
    fn default() -> Self {
        Self {
            strict: true,
            max_guard_depth: 20,
            max_decision_points: 500,
        }
    }
}

impl GuidelineParser {
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a lenient parser that collects warnings but does not fail on them.
    pub fn lenient() -> Self {
        Self {
            strict: false,
            ..Default::default()
        }
    }

    // ----- public parsing entry points ------------------------------------

    /// Parse a JSON string into a `GuidelineDocument`.
    pub fn parse_json(&self, input: &str) -> Result<GuidelineDocument, ParseError> {
        let doc: GuidelineDocument = serde_json::from_str(input)?;
        self.post_parse_validate(doc)
    }

    /// Parse a YAML string into a `GuidelineDocument`.
    pub fn parse_yaml(&self, input: &str) -> Result<GuidelineDocument, ParseError> {
        let doc: GuidelineDocument = serde_yaml::from_str(input)?;
        self.post_parse_validate(doc)
    }

    /// Parse a file (detecting format from extension).
    pub fn parse_file(&self, path: &Path) -> Result<GuidelineDocument, ParseError> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();
        let content = std::fs::read_to_string(path)?;
        match ext.as_str() {
            "json" => self.parse_json(&content),
            "yaml" | "yml" => self.parse_yaml(&content),
            other => Err(ParseError::UnsupportedFormat(other.to_string())),
        }
    }

    /// Parse from a `serde_json::Value` directly.
    pub fn parse_value(&self, value: serde_json::Value) -> Result<GuidelineDocument, ParseError> {
        let doc: GuidelineDocument = serde_json::from_value(value)?;
        self.post_parse_validate(doc)
    }

    // ----- validation pipeline --------------------------------------------

    fn post_parse_validate(
        &self,
        doc: GuidelineDocument,
    ) -> Result<GuidelineDocument, ParseError> {
        let mut issues = Vec::new();

        self.check_metadata(&doc, &mut issues);
        self.check_decision_points(&doc, &mut issues);
        self.check_transitions(&doc, &mut issues);
        self.check_references(&doc, &mut issues);
        self.check_circular_branches(&doc, &mut issues);
        self.check_guard_depth(&doc, &mut issues);
        self.check_safety_constraints(&doc, &mut issues);
        self.check_monitoring(&doc, &mut issues);
        self.check_duplicate_ids(&doc, &mut issues);

        let has_errors = issues.iter().any(|i| i.severity == IssueSeverity::Error);
        if self.strict && has_errors {
            Err(ParseError::Validation { issues })
        } else {
            Ok(doc)
        }
    }

    // ----- individual checks ----------------------------------------------

    fn check_metadata(&self, doc: &GuidelineDocument, issues: &mut Vec<ValidationIssue>) {
        if doc.metadata.title.trim().is_empty() {
            issues.push(ValidationIssue {
                severity: IssueSeverity::Error,
                code: "E001".into(),
                message: "Guideline title must not be empty".into(),
                location: Some("metadata.title".into()),
            });
        }
        if doc.metadata.version.trim().is_empty() {
            issues.push(ValidationIssue {
                severity: IssueSeverity::Warning,
                code: "W001".into(),
                message: "Guideline version is empty".into(),
                location: Some("metadata.version".into()),
            });
        }
        if doc.id.trim().is_empty() {
            issues.push(ValidationIssue {
                severity: IssueSeverity::Error,
                code: "E002".into(),
                message: "Guideline id must not be empty".into(),
                location: Some("id".into()),
            });
        }
    }

    fn check_decision_points(&self, doc: &GuidelineDocument, issues: &mut Vec<ValidationIssue>) {
        if doc.decision_points.is_empty() {
            issues.push(ValidationIssue {
                severity: IssueSeverity::Warning,
                code: "W002".into(),
                message: "No decision points defined".into(),
                location: None,
            });
            return;
        }

        if doc.decision_points.len() > self.max_decision_points {
            issues.push(ValidationIssue {
                severity: IssueSeverity::Error,
                code: "E003".into(),
                message: format!(
                    "Too many decision points ({} > {})",
                    doc.decision_points.len(),
                    self.max_decision_points,
                ),
                location: None,
            });
        }

        let initial_count = doc.decision_points.iter().filter(|dp| dp.is_initial).count();
        if initial_count == 0 {
            issues.push(ValidationIssue {
                severity: IssueSeverity::Warning,
                code: "W003".into(),
                message: "No initial decision point defined".into(),
                location: None,
            });
        }

        for dp in &doc.decision_points {
            if dp.id.trim().is_empty() {
                issues.push(ValidationIssue {
                    severity: IssueSeverity::Error,
                    code: "E004".into(),
                    message: "Decision point has empty id".into(),
                    location: Some(format!("decision_point[{}]", dp.label)),
                });
            }
            if dp.label.trim().is_empty() {
                issues.push(ValidationIssue {
                    severity: IssueSeverity::Warning,
                    code: "W004".into(),
                    message: format!("Decision point '{}' has empty label", dp.id),
                    location: Some(format!("decision_point[{}]", dp.id)),
                });
            }
            if !dp.is_terminal && dp.branches.is_empty() {
                issues.push(ValidationIssue {
                    severity: IssueSeverity::Warning,
                    code: "W005".into(),
                    message: format!(
                        "Non-terminal decision point '{}' has no branches",
                        dp.id
                    ),
                    location: Some(format!("decision_point[{}]", dp.id)),
                });
            }
            for br in &dp.branches {
                if br.id.trim().is_empty() {
                    issues.push(ValidationIssue {
                        severity: IssueSeverity::Error,
                        code: "E005".into(),
                        message: format!(
                            "Branch in decision point '{}' has empty id",
                            dp.id
                        ),
                        location: Some(format!("decision_point[{}].branch", dp.id)),
                    });
                }
                if br.target.trim().is_empty() {
                    issues.push(ValidationIssue {
                        severity: IssueSeverity::Error,
                        code: "E006".into(),
                        message: format!(
                            "Branch '{}' in '{}' has empty target",
                            br.id, dp.id
                        ),
                        location: Some(format!("decision_point[{}].branch[{}]", dp.id, br.id)),
                    });
                }
            }
        }
    }

    fn check_transitions(&self, doc: &GuidelineDocument, issues: &mut Vec<ValidationIssue>) {
        let dp_ids: HashSet<&str> = doc.decision_points.iter().map(|dp| dp.id.as_str()).collect();
        for tr in &doc.transitions {
            if !dp_ids.contains(tr.source.as_str()) {
                issues.push(ValidationIssue {
                    severity: IssueSeverity::Error,
                    code: "E007".into(),
                    message: format!(
                        "Transition '{}' references unknown source '{}'",
                        tr.id, tr.source
                    ),
                    location: Some(format!("transition[{}]", tr.id)),
                });
            }
            if !dp_ids.contains(tr.target.as_str()) {
                issues.push(ValidationIssue {
                    severity: IssueSeverity::Error,
                    code: "E008".into(),
                    message: format!(
                        "Transition '{}' references unknown target '{}'",
                        tr.id, tr.target
                    ),
                    location: Some(format!("transition[{}]", tr.id)),
                });
            }
        }
    }

    fn check_references(&self, doc: &GuidelineDocument, issues: &mut Vec<ValidationIssue>) {
        let dp_ids: HashSet<&str> = doc.decision_points.iter().map(|dp| dp.id.as_str()).collect();
        for dp in &doc.decision_points {
            for br in &dp.branches {
                if !dp_ids.contains(br.target.as_str()) {
                    issues.push(ValidationIssue {
                        severity: IssueSeverity::Error,
                        code: "E009".into(),
                        message: format!(
                            "Branch '{}' in '{}' targets unknown decision point '{}'",
                            br.id, dp.id, br.target
                        ),
                        location: Some(format!(
                            "decision_point[{}].branch[{}].target",
                            dp.id, br.id
                        )),
                    });
                }
            }
        }
    }

    /// Detect trivial self-loops on initial nodes that would form immediate
    /// infinite cycles with no escape.
    fn check_circular_branches(&self, doc: &GuidelineDocument, issues: &mut Vec<ValidationIssue>) {
        for dp in &doc.decision_points {
            let all_self = !dp.branches.is_empty()
                && dp.branches.iter().all(|br| br.target == dp.id);
            if all_self && dp.is_initial {
                issues.push(ValidationIssue {
                    severity: IssueSeverity::Warning,
                    code: "W010".into(),
                    message: format!(
                        "Initial decision point '{}' has only self-referencing branches",
                        dp.id
                    ),
                    location: Some(format!("decision_point[{}]", dp.id)),
                });
            }
        }
    }

    fn check_guard_depth(&self, doc: &GuidelineDocument, issues: &mut Vec<ValidationIssue>) {
        for dp in &doc.decision_points {
            for br in &dp.branches {
                if br.guard.depth() > self.max_guard_depth {
                    issues.push(ValidationIssue {
                        severity: IssueSeverity::Error,
                        code: "E010".into(),
                        message: format!(
                            "Guard in branch '{}' of '{}' exceeds max depth {}",
                            br.id, dp.id, self.max_guard_depth
                        ),
                        location: Some(format!(
                            "decision_point[{}].branch[{}].guard",
                            dp.id, br.id
                        )),
                    });
                }
            }
        }
    }

    fn check_safety_constraints(
        &self,
        doc: &GuidelineDocument,
        issues: &mut Vec<ValidationIssue>,
    ) {
        let dp_ids: HashSet<&str> = doc.decision_points.iter().map(|dp| dp.id.as_str()).collect();
        for sc in &doc.safety_constraints {
            if sc.id.trim().is_empty() {
                issues.push(ValidationIssue {
                    severity: IssueSeverity::Error,
                    code: "E011".into(),
                    message: "Safety constraint has empty id".into(),
                    location: None,
                });
            }
            for state in &sc.applies_to {
                if !dp_ids.contains(state.as_str()) {
                    issues.push(ValidationIssue {
                        severity: IssueSeverity::Warning,
                        code: "W011".into(),
                        message: format!(
                            "Safety constraint '{}' applies_to unknown state '{}'",
                            sc.id, state
                        ),
                        location: Some(format!("safety_constraint[{}]", sc.id)),
                    });
                }
            }
        }
    }

    fn check_monitoring(&self, doc: &GuidelineDocument, issues: &mut Vec<ValidationIssue>) {
        let dp_ids: HashSet<&str> = doc.decision_points.iter().map(|dp| dp.id.as_str()).collect();
        for mr in &doc.monitoring {
            if mr.interval_days == 0 {
                issues.push(ValidationIssue {
                    severity: IssueSeverity::Warning,
                    code: "W012".into(),
                    message: format!(
                        "Monitoring requirement '{}' has zero interval",
                        mr.id
                    ),
                    location: Some(format!("monitoring[{}]", mr.id)),
                });
            }
            for state in &mr.applies_to_states {
                if !dp_ids.contains(state.as_str()) {
                    issues.push(ValidationIssue {
                        severity: IssueSeverity::Warning,
                        code: "W013".into(),
                        message: format!(
                            "Monitoring '{}' applies_to unknown state '{}'",
                            mr.id, state
                        ),
                        location: Some(format!("monitoring[{}]", mr.id)),
                    });
                }
            }
        }
    }

    fn check_duplicate_ids(&self, doc: &GuidelineDocument, issues: &mut Vec<ValidationIssue>) {
        let mut seen: HashSet<String> = HashSet::new();
        for dp in &doc.decision_points {
            if !seen.insert(dp.id.clone()) {
                issues.push(ValidationIssue {
                    severity: IssueSeverity::Error,
                    code: "E012".into(),
                    message: format!("Duplicate decision point id '{}'", dp.id),
                    location: None,
                });
            }
        }
        let mut tr_seen: HashSet<String> = HashSet::new();
        for tr in &doc.transitions {
            if !tr_seen.insert(tr.id.clone()) {
                issues.push(ValidationIssue {
                    severity: IssueSeverity::Error,
                    code: "E013".into(),
                    message: format!("Duplicate transition id '{}'", tr.id),
                    location: None,
                });
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Convenience: build a GuidelineDocument programmatically
// ---------------------------------------------------------------------------

/// Fluent builder for constructing a `GuidelineDocument` piece by piece.
pub struct GuidelineDocumentBuilder {
    doc: GuidelineDocument,
}

impl GuidelineDocumentBuilder {
    pub fn new(title: &str) -> Self {
        Self {
            doc: GuidelineDocument::new(title),
        }
    }

    pub fn id(mut self, id: &str) -> Self {
        self.doc.id = id.to_string();
        self
    }

    pub fn version(mut self, v: &str) -> Self {
        self.doc.metadata.version = v.to_string();
        self
    }

    pub fn condition(mut self, c: &str) -> Self {
        self.doc.metadata.condition = Some(c.to_string());
        self
    }

    pub fn author(mut self, a: &str) -> Self {
        self.doc.metadata.authors.push(a.to_string());
        self
    }

    pub fn source_org(mut self, org: &str) -> Self {
        self.doc.metadata.source_organization = Some(org.to_string());
        self
    }

    pub fn tag(mut self, t: &str) -> Self {
        self.doc.metadata.tags.push(t.to_string());
        self
    }

    pub fn decision_point(mut self, dp: DecisionPoint) -> Self {
        self.doc.decision_points.push(dp);
        self
    }

    pub fn transition(mut self, tr: TransitionRule) -> Self {
        self.doc.transitions.push(tr);
        self
    }

    pub fn safety_constraint(mut self, sc: SafetyConstraint) -> Self {
        self.doc.safety_constraints.push(sc);
        self
    }

    pub fn monitoring(mut self, mr: MonitoringRequirement) -> Self {
        self.doc.monitoring.push(mr);
        self
    }

    pub fn annotate(mut self, key: &str, value: &str) -> Self {
        self.doc.annotations.insert(key.to_string(), value.to_string());
        self
    }

    pub fn build(self) -> GuidelineDocument {
        self.doc
    }
}

// ---------------------------------------------------------------------------
// Normalisation helpers
// ---------------------------------------------------------------------------

/// Normalise a parsed guideline document: ensure IDs are trimmed, assign
/// default priorities where missing, etc.
pub fn normalise_document(mut doc: GuidelineDocument) -> GuidelineDocument {
    doc.id = doc.id.trim().to_string();
    doc.metadata.title = doc.metadata.title.trim().to_string();
    doc.metadata.version = doc.metadata.version.trim().to_string();

    for dp in &mut doc.decision_points {
        dp.id = dp.id.trim().to_string();
        dp.label = dp.label.trim().to_string();
        for br in &mut dp.branches {
            br.id = br.id.trim().to_string();
            br.target = br.target.trim().to_string();
        }
        // Sort branches by priority (lower = higher priority).
        dp.branches.sort_by_key(|b| b.priority);
    }

    for tr in &mut doc.transitions {
        tr.id = tr.id.trim().to_string();
        tr.source = tr.source.trim().to_string();
        tr.target = tr.target.trim().to_string();
    }

    doc
}

/// Merge two guideline documents by appending the decision points, transitions,
/// safety constraints and monitoring of `other` into `base`.  Duplicate IDs in
/// `other` are prefixed with the other document's id to avoid clashes.
pub fn merge_documents(
    mut base: GuidelineDocument,
    other: &GuidelineDocument,
) -> GuidelineDocument {
    let existing_dp_ids: HashSet<String> = base
        .decision_points
        .iter()
        .map(|dp| dp.id.clone())
        .collect();

    let prefix = format!("{}_", other.id.replace('-', "_"));

    for mut dp in other.decision_points.clone() {
        if existing_dp_ids.contains(&dp.id) {
            dp.id = format!("{}{}", prefix, dp.id);
            for br in &mut dp.branches {
                if existing_dp_ids.contains(&br.target) {
                    // keep cross-references to base as-is
                } else {
                    br.target = format!("{}{}", prefix, br.target);
                }
            }
        }
        base.decision_points.push(dp);
    }

    for mut tr in other.transitions.clone() {
        if existing_dp_ids.contains(&tr.source) {
            // keep
        } else {
            tr.source = format!("{}{}", prefix, tr.source);
        }
        if existing_dp_ids.contains(&tr.target) {
            // keep
        } else {
            tr.target = format!("{}{}", prefix, tr.target);
        }
        tr.id = format!("{}{}", prefix, tr.id);
        base.transitions.push(tr);
    }

    for sc in &other.safety_constraints {
        base.safety_constraints.push(sc.clone());
    }
    for mr in &other.monitoring {
        base.monitoring.push(mr.clone());
    }

    base
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{standard_diabetes_template, standard_hypertension_template};

    fn sample_json() -> String {
        let doc = standard_diabetes_template();
        serde_json::to_string_pretty(&doc).unwrap()
    }

    fn sample_yaml() -> String {
        let doc = standard_diabetes_template();
        serde_yaml::to_string(&doc).unwrap()
    }

    #[test]
    fn test_parse_json_roundtrip() {
        let parser = GuidelineParser::new();
        let json = sample_json();
        let doc = parser.parse_json(&json).unwrap();
        assert_eq!(doc.metadata.title, "Type 2 Diabetes Management");
        assert!(doc.num_decision_points() >= 6);
    }

    #[test]
    fn test_parse_yaml_roundtrip() {
        let parser = GuidelineParser::new();
        let yaml = sample_yaml();
        let doc = parser.parse_yaml(&yaml).unwrap();
        assert_eq!(doc.metadata.title, "Type 2 Diabetes Management");
    }

    #[test]
    fn test_parse_invalid_json() {
        let parser = GuidelineParser::new();
        let result = parser.parse_json("not valid json");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_empty_title() {
        let parser = GuidelineParser::new();
        let mut doc = standard_diabetes_template();
        doc.metadata.title = "".into();
        let json = serde_json::to_string(&doc).unwrap();
        let result = parser.parse_json(&json);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_duplicate_dp_ids() {
        let parser = GuidelineParser::new();
        let mut doc = standard_diabetes_template();
        if let Some(dp) = doc.decision_points.get(1) {
            let mut dup = dp.clone();
            dup.label = "Duplicate".into();
            doc.decision_points.push(dup);
        }
        let json = serde_json::to_string(&doc).unwrap();
        let result = parser.parse_json(&json);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_bad_branch_target() {
        let parser = GuidelineParser::new();
        let mut doc = standard_diabetes_template();
        doc.decision_points[0].branches[0].target = "nonexistent_state".into();
        let json = serde_json::to_string(&doc).unwrap();
        let result = parser.parse_json(&json);
        assert!(result.is_err());
    }

    #[test]
    fn test_lenient_parser() {
        let parser = GuidelineParser::lenient();
        let mut doc = standard_diabetes_template();
        doc.metadata.version = "".into(); // warning, not error
        let json = serde_json::to_string(&doc).unwrap();
        let result = parser.parse_json(&json);
        assert!(result.is_ok());
    }

    #[test]
    fn test_normalise_document() {
        let mut doc = standard_diabetes_template();
        doc.metadata.title = "  padded title  ".into();
        doc.decision_points[0].id = " initial_assessment ".into();
        let normed = normalise_document(doc);
        assert_eq!(normed.metadata.title, "padded title");
        assert_eq!(normed.decision_points[0].id, "initial_assessment");
    }

    #[test]
    fn test_builder() {
        let doc = GuidelineDocumentBuilder::new("Test Guideline")
            .id("test-1")
            .version("1.0.0")
            .condition("Test Condition")
            .author("Dr. Test")
            .tag("test")
            .decision_point(DecisionPoint {
                id: "start".into(),
                label: "Start".into(),
                description: None,
                branches: vec![],
                is_initial: true,
                is_terminal: true,
                invariants: vec![],
                urgency: None,
            })
            .build();
        assert_eq!(doc.id, "test-1");
        assert_eq!(doc.metadata.title, "Test Guideline");
        assert_eq!(doc.decision_points.len(), 1);
    }

    #[test]
    fn test_merge_documents() {
        let base = standard_diabetes_template();
        let other = standard_hypertension_template();
        let merged = merge_documents(base.clone(), &other);
        // Should have decision points from both
        assert!(merged.decision_points.len() >= base.decision_points.len() + other.decision_points.len());
        // Safety constraints should be combined
        assert!(
            merged.safety_constraints.len()
                >= base.safety_constraints.len() + other.safety_constraints.len()
        );
    }

    #[test]
    fn test_parse_value() {
        let doc = standard_diabetes_template();
        let val = serde_json::to_value(&doc).unwrap();
        let parser = GuidelineParser::new();
        let parsed = parser.parse_value(val).unwrap();
        assert_eq!(parsed.metadata.title, doc.metadata.title);
    }

    #[test]
    fn test_validation_issue_display() {
        let issue = ValidationIssue {
            severity: IssueSeverity::Error,
            code: "E001".into(),
            message: "bad".into(),
            location: Some("here".into()),
        };
        let s = format!("{}", issue);
        assert!(s.contains("E001"));
        assert!(s.contains("here"));
    }

    #[test]
    fn test_unsupported_format() {
        let parser = GuidelineParser::new();
        let result = parser.parse_file(Path::new("test.xml"));
        match result {
            Err(ParseError::UnsupportedFormat(ext)) => assert_eq!(ext, "xml"),
            _ => panic!("Expected UnsupportedFormat"),
        }
    }
}
