//! Top-level validity oracle.
//!
//! `ValidityChecker` composes the agreement checker, subcategorization checker,
//! unification engine, and transformation-specific constraint sets to answer:
//! "Is this sentence grammatically valid w.r.t. the constraints our NLP
//! transformations require?"

use crate::agreement::AgreementChecker;
use crate::clause_types::validate_clause_structure;
use crate::constraints::{self, ConstraintSet, validate_sentence as validate_constraints};
use crate::subcategorization::SubcatChecker;
use crate::unification::UnificationEngine;
use shared_types::{ParseTree, Sentence};
use serde::{Deserialize, Serialize};
use std::fmt;

// ── ValidityViolation ───────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ViolationCategory {
    Agreement,
    Subcategorization,
    Unification,
    ClauseStructure,
    TransformationSpecific,
}

impl fmt::Display for ViolationCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Severity {
    Error,
    Warning,
    Info,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidityViolation {
    pub category: ViolationCategory,
    pub description: String,
    pub severity: Severity,
    /// Token index or span description.
    pub location: String,
}

impl ValidityViolation {
    pub fn error(
        category: ViolationCategory,
        description: impl Into<String>,
        location: impl Into<String>,
    ) -> Self {
        Self {
            category,
            description: description.into(),
            severity: Severity::Error,
            location: location.into(),
        }
    }

    pub fn warning(
        category: ViolationCategory,
        description: impl Into<String>,
        location: impl Into<String>,
    ) -> Self {
        Self {
            category,
            description: description.into(),
            severity: Severity::Warning,
            location: location.into(),
        }
    }
}

impl fmt::Display for ValidityViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{:?}] {} at {} ({:?})",
            self.severity, self.description, self.location, self.category
        )
    }
}

// ── ValidityResult ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidityResult {
    pub is_valid: bool,
    pub violations: Vec<ValidityViolation>,
    pub checked_constraints: usize,
}

impl ValidityResult {
    pub fn valid(checked: usize) -> Self {
        Self {
            is_valid: true,
            violations: Vec::new(),
            checked_constraints: checked,
        }
    }

    pub fn invalid(violations: Vec<ValidityViolation>, checked: usize) -> Self {
        Self {
            is_valid: false,
            violations,
            checked_constraints: checked,
        }
    }

    pub fn error_count(&self) -> usize {
        self.violations
            .iter()
            .filter(|v| v.severity == Severity::Error)
            .count()
    }

    pub fn warning_count(&self) -> usize {
        self.violations
            .iter()
            .filter(|v| v.severity == Severity::Warning)
            .count()
    }
}

impl fmt::Display for ValidityResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_valid {
            write!(f, "VALID ({} constraints checked)", self.checked_constraints)
        } else {
            write!(
                f,
                "INVALID: {} errors, {} warnings ({} constraints checked)",
                self.error_count(),
                self.warning_count(),
                self.checked_constraints
            )
        }
    }
}

// ── ValidityReport ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidityReport {
    pub result: ValidityResult,
    pub agreement_violations: Vec<String>,
    pub subcat_violations: Vec<String>,
    pub clause_issues: Vec<String>,
    pub transformation_issues: Vec<String>,
}

impl ValidityReport {
    pub fn new(result: ValidityResult) -> Self {
        Self {
            result,
            agreement_violations: Vec::new(),
            subcat_violations: Vec::new(),
            clause_issues: Vec::new(),
            transformation_issues: Vec::new(),
        }
    }

    pub fn total_issues(&self) -> usize {
        self.agreement_violations.len()
            + self.subcat_violations.len()
            + self.clause_issues.len()
            + self.transformation_issues.len()
    }
}

impl fmt::Display for ValidityReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Validity Report ===")?;
        writeln!(f, "{}", self.result)?;
        if !self.agreement_violations.is_empty() {
            writeln!(f, "Agreement:")?;
            for v in &self.agreement_violations {
                writeln!(f, "  - {v}")?;
            }
        }
        if !self.subcat_violations.is_empty() {
            writeln!(f, "Subcategorization:")?;
            for v in &self.subcat_violations {
                writeln!(f, "  - {v}")?;
            }
        }
        if !self.clause_issues.is_empty() {
            writeln!(f, "Clause structure:")?;
            for v in &self.clause_issues {
                writeln!(f, "  - {v}")?;
            }
        }
        if !self.transformation_issues.is_empty() {
            writeln!(f, "Transformation-specific:")?;
            for v in &self.transformation_issues {
                writeln!(f, "  - {v}")?;
            }
        }
        Ok(())
    }
}

// ── ValidityOracle trait ────────────────────────────────────────────────────

/// Trait for checking sentence grammatical validity.
///
/// Defined here because the `shared_types::traits` module may not yet exist.
pub trait ValidityOracle {
    fn is_valid(&self, sentence: &Sentence) -> bool;
    fn explain_invalidity(&self, sentence: &Sentence) -> Vec<String>;
}

// ── ValidityChecker ─────────────────────────────────────────────────────────

/// The primary grammar-checking façade.
pub struct ValidityChecker {
    pub agreement_checker: AgreementChecker,
    pub subcat_checker: SubcatChecker,
    pub unification_engine: UnificationEngine,
    pub constraint_sets: Vec<ConstraintSet>,
}

impl ValidityChecker {
    pub fn new(
        agreement_checker: AgreementChecker,
        subcat_checker: SubcatChecker,
        unification_engine: UnificationEngine,
        constraint_sets: Vec<ConstraintSet>,
    ) -> Self {
        Self {
            agreement_checker,
            subcat_checker,
            unification_engine,
            constraint_sets,
        }
    }

    /// Run all checks.
    pub fn check_validity(&self, sentence: &Sentence) -> ValidityResult {
        let mut violations = Vec::new();
        let mut checked: usize = 0;

        // 1. Agreement
        let agr = self.agreement_checker.check_all(sentence);
        checked += agr.len().max(1);
        for v in &agr {
            violations.push(ValidityViolation::error(
                ViolationCategory::Agreement,
                v.to_string(),
                format!("feature={}", v.violated_feature),
            ));
        }

        // 2. Subcategorization – check every verb
        let verbs = crate::agreement::VerbFinder::find_all_verbs(sentence);
        for vi in &verbs {
            let sub_v = self.subcat_checker.check_subcategorization(sentence, *vi);
            checked += 1;
            for sv in &sub_v {
                violations.push(ValidityViolation::error(
                    ViolationCategory::Subcategorization,
                    sv.to_string(),
                    format!("verb_index={vi}"),
                ));
            }
        }

        // 3. Clause structure
        let clause_issues = validate_clause_structure(sentence);
        checked += clause_issues.len().max(1);
        for issue in &clause_issues {
            violations.push(ValidityViolation::warning(
                ViolationCategory::ClauseStructure,
                issue.clone(),
                "clause",
            ));
        }

        if violations.is_empty() {
            ValidityResult::valid(checked)
        } else {
            ValidityResult::invalid(violations, checked)
        }
    }

    /// Fast check — only agreement.
    pub fn quick_check(&self, sentence: &Sentence) -> ValidityResult {
        let agr = self.agreement_checker.check_all(sentence);
        let mut violations = Vec::new();
        for v in &agr {
            violations.push(ValidityViolation::error(
                ViolationCategory::Agreement,
                v.to_string(),
                format!("feature={}", v.violated_feature),
            ));
        }
        if violations.is_empty() {
            ValidityResult::valid(1)
        } else {
            ValidityResult::invalid(violations, 1)
        }
    }

    /// Thorough check — all sub-systems plus unification on a parse tree.
    pub fn full_check(
        &self,
        sentence: &Sentence,
        tree: Option<&ParseTree>,
    ) -> ValidityReport {
        let base = self.check_validity(sentence);
        let mut report = ValidityReport::new(base);

        // Populate per-category lists.
        for v in &report.result.violations {
            match v.category {
                ViolationCategory::Agreement => {
                    report.agreement_violations.push(v.description.clone());
                }
                ViolationCategory::Subcategorization => {
                    report.subcat_violations.push(v.description.clone());
                }
                ViolationCategory::ClauseStructure => {
                    report.clause_issues.push(v.description.clone());
                }
                _ => {}
            }
        }

        // Unification on parse tree, if provided.
        if let Some(t) = tree {
            let uni_violations = self.unification_engine.check_all_constraints(t);
            for cv in &uni_violations {
                report.result.violations.push(ValidityViolation::error(
                    ViolationCategory::Unification,
                    cv.to_string(),
                    format!("{} ↔ {}", cv.node1, cv.node2),
                ));
            }
            if !uni_violations.is_empty() {
                report.result.is_valid = false;
            }
        }

        report
    }

    /// Check validity specifically for a named transformation.
    pub fn check_for_transformation(
        &self,
        sentence: &Sentence,
        transformation: &str,
    ) -> ValidityReport {
        let base = self.check_validity(sentence);
        let mut report = ValidityReport::new(base);

        let trans_issues = validate_constraints(sentence, transformation, &self.subcat_checker);
        for issue in &trans_issues {
            report.result.violations.push(ValidityViolation::error(
                ViolationCategory::TransformationSpecific,
                issue.clone(),
                format!("transformation={transformation}"),
            ));
            report.transformation_issues.push(issue.clone());
        }

        if !report.result.violations.is_empty() {
            report.result.is_valid = false;
        }

        report
    }
}

impl ValidityOracle for ValidityChecker {
    fn is_valid(&self, sentence: &Sentence) -> bool {
        self.check_validity(sentence).is_valid
    }

    fn explain_invalidity(&self, sentence: &Sentence) -> Vec<String> {
        let result = self.check_validity(sentence);
        result
            .violations
            .iter()
            .map(|v| v.to_string())
            .collect()
    }
}

// ── Factory ─────────────────────────────────────────────────────────────────

/// Create a fully configured default validity checker.
pub fn default_validity_checker() -> ValidityChecker {
    let agreement_checker = AgreementChecker::new();
    let subcat_checker = SubcatChecker::new();
    let unification_engine = UnificationEngine::with_defaults();

    let constraint_sets = vec![
        constraints::PassivizationConstraints::build(),
        constraints::CleftingConstraints::build(),
        constraints::TopicalizationConstraints::build(),
        constraints::RelativeClauseConstraints::build(),
        constraints::TenseChangeConstraints::build(),
        constraints::ThereInsertionConstraints::build(),
        constraints::DativeAlternationConstraints::build(),
        constraints::NegationConstraints::build(),
        constraints::AgreementPerturbationConstraints::build(),
        constraints::EmbeddingConstraints::build(),
    ];

    ValidityChecker::new(
        agreement_checker,
        subcat_checker,
        unification_engine,
        constraint_sets,
    )
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use shared_types::{DependencyEdge, DependencyRelation, PosTag, Token};

    fn make_sentence(words: &[(&str, PosTag)], edges: Vec<DependencyEdge>) -> Sentence {
        let tokens: Vec<Token> = words
            .iter()
            .enumerate()
            .map(|(i, (w, pos))| Token::new(*w, i).with_pos(*pos).with_lemma(w.to_lowercase()))
            .collect();
        Sentence {
            raw_text: words.iter().map(|(w, _)| *w).collect::<Vec<_>>().join(" "),
            tokens,
            dependency_edges: edges,
            entities: vec![],
        }
    }

    #[test]
    fn test_default_checker_creation() {
        let checker = default_validity_checker();
        assert!(!checker.constraint_sets.is_empty());
    }

    #[test]
    fn test_valid_sentence() {
        let checker = default_validity_checker();
        let s = make_sentence(
            &[("cats", PosTag::Noun), ("eat", PosTag::Verb), ("fish", PosTag::Noun)],
            vec![
                DependencyEdge::new(1, 0, DependencyRelation::Nsubj),
                DependencyEdge::new(1, 1, DependencyRelation::Root),
                DependencyEdge::new(1, 2, DependencyRelation::Dobj),
            ],
        );
        let result = checker.check_validity(&s);
        assert!(result.is_valid, "Expected valid, got: {:?}", result.violations);
    }

    #[test]
    fn test_quick_check() {
        let checker = default_validity_checker();
        let s = make_sentence(
            &[("cats", PosTag::Noun), ("eat", PosTag::Verb)],
            vec![
                DependencyEdge::new(1, 0, DependencyRelation::Nsubj),
                DependencyEdge::new(1, 1, DependencyRelation::Root),
            ],
        );
        let result = checker.quick_check(&s);
        assert!(result.is_valid);
    }

    #[test]
    fn test_is_valid_trait() {
        let checker = default_validity_checker();
        let s = make_sentence(
            &[("cats", PosTag::Noun), ("sleep", PosTag::Verb)],
            vec![
                DependencyEdge::new(1, 0, DependencyRelation::Nsubj),
                DependencyEdge::new(1, 1, DependencyRelation::Root),
            ],
        );
        assert!(checker.is_valid(&s));
    }

    #[test]
    fn test_explain_invalidity_returns_empty_for_valid() {
        let checker = default_validity_checker();
        let s = make_sentence(
            &[("cats", PosTag::Noun), ("sleep", PosTag::Verb)],
            vec![
                DependencyEdge::new(1, 0, DependencyRelation::Nsubj),
                DependencyEdge::new(1, 1, DependencyRelation::Root),
            ],
        );
        let explanations = checker.explain_invalidity(&s);
        assert!(explanations.is_empty());
    }

    #[test]
    fn test_full_check_no_tree() {
        let checker = default_validity_checker();
        let s = make_sentence(
            &[("cats", PosTag::Noun), ("sleep", PosTag::Verb)],
            vec![
                DependencyEdge::new(1, 0, DependencyRelation::Nsubj),
                DependencyEdge::new(1, 1, DependencyRelation::Root),
            ],
        );
        let report = checker.full_check(&s, None);
        assert!(report.result.is_valid);
    }

    #[test]
    fn test_check_for_transformation_passivization() {
        let checker = default_validity_checker();
        let s = make_sentence(
            &[("cats", PosTag::Noun), ("eat", PosTag::Verb), ("fish", PosTag::Noun)],
            vec![
                DependencyEdge::new(1, 0, DependencyRelation::Nsubj),
                DependencyEdge::new(1, 1, DependencyRelation::Root),
                DependencyEdge::new(1, 2, DependencyRelation::Dobj),
            ],
        );
        let report = checker.check_for_transformation(&s, "passivization");
        // Should pass — transitive verb with object
        assert!(
            report.transformation_issues.is_empty(),
            "Unexpected issues: {:?}",
            report.transformation_issues
        );
    }

    #[test]
    fn test_check_for_transformation_passive_intransitive() {
        let checker = default_validity_checker();
        let s = make_sentence(
            &[("cats", PosTag::Noun), ("sleep", PosTag::Verb)],
            vec![
                DependencyEdge::new(1, 0, DependencyRelation::Nsubj),
                DependencyEdge::new(1, 1, DependencyRelation::Root),
            ],
        );
        let report = checker.check_for_transformation(&s, "passivization");
        assert!(!report.transformation_issues.is_empty());
    }

    #[test]
    fn test_validity_result_display() {
        let r = ValidityResult::valid(5);
        let s = r.to_string();
        assert!(s.contains("VALID"));
    }

    #[test]
    fn test_severity_counts() {
        let violations = vec![
            ValidityViolation::error(ViolationCategory::Agreement, "err", "loc"),
            ValidityViolation::warning(ViolationCategory::ClauseStructure, "warn", "loc"),
        ];
        let result = ValidityResult::invalid(violations, 2);
        assert_eq!(result.error_count(), 1);
        assert_eq!(result.warning_count(), 1);
    }

    #[test]
    fn test_validity_report_display() {
        let result = ValidityResult::valid(3);
        let report = ValidityReport::new(result);
        let s = report.to_string();
        assert!(s.contains("Validity Report"));
    }
}
