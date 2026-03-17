//! Grammar-checker: a lightweight feature-unification validity checker for
//! English grammar.
//!
//! This crate answers one question: *"Is this candidate sentence grammatically
//! valid with respect to the feature constraints that our 15 NLP
//! transformations require?"*  It covers ~80 feature constraints across ~15
//! clause types.
//!
//! # Modules
//!
//! * [`features`] – Feature enums, bundles, and unification primitives.
//! * [`agreement`] – Subject–verb agreement checking.
//! * [`subcategorization`] – Verb subcategorization frame checking.
//! * [`unification`] – Rule-based feature unification engine (~80 rules).
//! * [`constraints`] – Per-transformation grammar constraint sets.
//! * [`clause_types`] – Clause type classification and validation.
//! * [`validity`] – Top-level validity oracle that composes everything.

pub mod agreement;
pub mod clause_types;
pub mod constraints;
pub mod features;
pub mod subcategorization;
pub mod unification;
pub mod validity;

// ── Re-exports ──────────────────────────────────────────────────────────────

pub use agreement::{AgreementChecker, AgreementViolation};
pub use clause_types::{ClauseClassifier, ClauseType, ClauseProperties, ClauseRelation};
pub use constraints::{ConstraintSet, get_constraints_for_transformation, validate_sentence};
pub use features::{
    AnimacyValue, AspectValue, CaseValue, DefinitenessValue, Feature, FeatureBundle,
    FeatureConflict, FeatureStructure, FinitenessValue, GenderValue, MoodValue, NumberValue,
    PersonValue, TenseValue, TransitivityValue, VoiceValue,
};
pub use subcategorization::{SubcatChecker, SubcatFrame, SubcatViolation};
pub use unification::{
    ConstraintViolation, DefaultConstraints, GrammarConstraint, UnificationEngine,
    UnificationResult, UnificationRule,
};
pub use validity::{
    ValidityChecker, ValidityOracle, ValidityReport, ValidityResult, ValidityViolation,
    default_validity_checker,
};

// ── High-level convenience wrapper ──────────────────────────────────────────

use serde::{Deserialize, Serialize};

/// A simplified grammar-check result for CLI use.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrammarCheckResult {
    pub is_valid: bool,
    pub score: f64,
    pub errors: Vec<GrammarIssue>,
}

/// A single grammar issue.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrammarIssue {
    pub position: usize,
    pub message: String,
}

/// Convenience grammar checker that wraps [`ValidityChecker`].
pub struct GrammarChecker {
    inner: ValidityChecker,
}

impl GrammarChecker {
    pub fn new() -> Self {
        Self {
            inner: default_validity_checker(),
        }
    }

    /// Check a raw sentence string for grammatical validity.
    pub fn check(&self, sentence: &str) -> GrammarCheckResult {
        let tokens = simple_tokenize(sentence);
        let sent = shared_types::Sentence {
            tokens,
            dependency_edges: Vec::new(),
            entities: Vec::new(),
            raw_text: sentence.to_string(),
            features: None,
            parse_tree: None,
        };
        let result = self.inner.check_validity(&sent);
        let errors: Vec<GrammarIssue> = result
            .violations
            .iter()
            .enumerate()
            .map(|(i, v)| GrammarIssue {
                position: i,
                message: v.description.clone(),
            })
            .collect();
        let score = if result.checked_constraints > 0 {
            1.0 - (result.violations.len() as f64 / result.checked_constraints as f64)
        } else {
            1.0
        };
        GrammarCheckResult {
            is_valid: result.is_valid,
            score,
            errors,
        }
    }
}

impl Default for GrammarChecker {
    fn default() -> Self {
        Self::new()
    }
}

fn simple_tokenize(text: &str) -> Vec<shared_types::Token> {
    text.split_whitespace()
        .enumerate()
        .map(|(i, word)| shared_types::Token {
            text: word.to_string(),
            lemma: Some(word.to_lowercase()),
            pos_tag: None,
            index: i,
            features: std::collections::HashMap::new(),
        })
        .collect()
}
