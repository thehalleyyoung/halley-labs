//! POS tag consistency oracle.
//!
//! Checks whether POS tagging is consistent across metamorphic transformations
//! where tag preservation is expected (e.g., synonym substitution preserves
//! the POS of the substituted word).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A POS-tagged token.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TaggedToken {
    pub text: String,
    pub lemma: String,
    pub pos: String,
    pub fine_pos: String,
    pub index: usize,
}

/// POS consistency oracle that checks tag preservation.
pub struct POSConsistencyOracle {
    /// Allowed tag equivalences (e.g., VBD ≈ VBN for some transformations).
    equivalences: HashMap<String, Vec<String>>,
    /// Tags that are exempt from checking.
    exempt_tags: Vec<String>,
    /// Expected changes for specific transformations.
    expected_changes: HashMap<String, Vec<POSChangeRule>>,
    total_checks: usize,
    violations: usize,
}

/// A rule specifying expected POS changes for a transformation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct POSChangeRule {
    pub from_tag: String,
    pub to_tag: String,
    pub context: String,
}

/// Result of a POS consistency check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct POSCheckResult {
    pub is_violation: bool,
    pub total_tokens_checked: usize,
    pub consistent_tokens: usize,
    pub inconsistent_tokens: Vec<POSInconsistency>,
    pub consistency_ratio: f64,
    pub expected_changes_found: Vec<(String, String)>,
}

/// A POS tag inconsistency between original and transformed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct POSInconsistency {
    pub token_text: String,
    pub original_pos: String,
    pub transformed_pos: String,
    pub is_expected: bool,
    pub explanation: String,
}

impl POSConsistencyOracle {
    pub fn new() -> Self {
        let mut equivalences = HashMap::new();
        // Standard POS equivalence classes.
        equivalences.insert(
            "VB".to_string(),
            vec!["VB".into(), "VBP".into(), "VBZ".into()],
        );
        equivalences.insert(
            "NN".to_string(),
            vec!["NN".into(), "NNS".into()],
        );
        equivalences.insert(
            "JJ".to_string(),
            vec!["JJ".into(), "JJR".into(), "JJS".into()],
        );
        equivalences.insert(
            "RB".to_string(),
            vec!["RB".into(), "RBR".into(), "RBS".into()],
        );

        Self {
            equivalences,
            exempt_tags: vec!["PUNCT".into(), "SPACE".into(), "SYM".into()],
            expected_changes: HashMap::new(),
            total_checks: 0,
            violations: 0,
        }
    }

    /// Add a POS tag equivalence class.
    pub fn add_equivalence(&mut self, base: impl Into<String>, equivalents: Vec<String>) {
        self.equivalences.insert(base.into(), equivalents);
    }

    /// Add expected POS change rules for a transformation.
    pub fn add_expected_changes(
        &mut self,
        transformation: impl Into<String>,
        rules: Vec<POSChangeRule>,
    ) {
        self.expected_changes.insert(transformation.into(), rules);
    }

    /// Set up standard expected changes for the 15 NLP transformations.
    pub fn with_standard_expectations(mut self) -> Self {
        // Passivization: VBD/VBZ → VBN (past participle).
        self.add_expected_changes(
            "passivization",
            vec![
                POSChangeRule {
                    from_tag: "VBD".into(),
                    to_tag: "VBN".into(),
                    context: "main verb in active clause".into(),
                },
                POSChangeRule {
                    from_tag: "VBZ".into(),
                    to_tag: "VBN".into(),
                    context: "main verb in active clause".into(),
                },
                POSChangeRule {
                    from_tag: "VBP".into(),
                    to_tag: "VBN".into(),
                    context: "main verb in active clause".into(),
                },
            ],
        );

        // Tense change: various verb form changes.
        self.add_expected_changes(
            "tense_change",
            vec![
                POSChangeRule {
                    from_tag: "VBD".into(),
                    to_tag: "VBZ".into(),
                    context: "past to present".into(),
                },
                POSChangeRule {
                    from_tag: "VBZ".into(),
                    to_tag: "VBD".into(),
                    context: "present to past".into(),
                },
            ],
        );

        // Agreement perturbation: NNS ↔ NN, VBZ ↔ VBP.
        self.add_expected_changes(
            "agreement_perturbation",
            vec![
                POSChangeRule {
                    from_tag: "NN".into(),
                    to_tag: "NNS".into(),
                    context: "singular to plural".into(),
                },
                POSChangeRule {
                    from_tag: "NNS".into(),
                    to_tag: "NN".into(),
                    context: "plural to singular".into(),
                },
                POSChangeRule {
                    from_tag: "VBZ".into(),
                    to_tag: "VBP".into(),
                    context: "verb agreement change".into(),
                },
                POSChangeRule {
                    from_tag: "VBP".into(),
                    to_tag: "VBZ".into(),
                    context: "verb agreement change".into(),
                },
            ],
        );

        self
    }

    /// Check POS consistency between aligned token sequences.
    pub fn check(
        &mut self,
        original: &[TaggedToken],
        transformed: &[TaggedToken],
        transformation: &str,
    ) -> POSCheckResult {
        self.total_checks += 1;

        let expected_rules = self.expected_changes.get(transformation).cloned().unwrap_or_default();

        // Align tokens by lemma.
        let alignment = self.align_by_lemma(original, transformed);

        let mut consistent = 0usize;
        let mut inconsistencies = Vec::new();
        let mut expected_changes_found = Vec::new();

        for (orig_idx, trans_idx) in &alignment {
            let orig = &original[*orig_idx];
            let trans = &transformed[*trans_idx];

            // Skip exempt tags.
            if self.exempt_tags.contains(&orig.pos) || self.exempt_tags.contains(&trans.pos) {
                consistent += 1;
                continue;
            }

            if orig.pos == trans.pos {
                consistent += 1;
                continue;
            }

            // Check equivalence classes.
            if self.are_equivalent(&orig.pos, &trans.pos) {
                consistent += 1;
                continue;
            }

            // Check expected changes.
            let is_expected = expected_rules.iter().any(|rule| {
                rule.from_tag == orig.pos && rule.to_tag == trans.pos
            });

            if is_expected {
                expected_changes_found.push((orig.pos.clone(), trans.pos.clone()));
                consistent += 1;
                continue;
            }

            inconsistencies.push(POSInconsistency {
                token_text: orig.text.clone(),
                original_pos: orig.pos.clone(),
                transformed_pos: trans.pos.clone(),
                is_expected: false,
                explanation: format!(
                    "'{}': {} → {} (unexpected for transformation '{}')",
                    orig.text, orig.pos, trans.pos, transformation
                ),
            });
        }

        let total_checked = alignment.len();
        let consistency_ratio = if total_checked == 0 {
            1.0
        } else {
            consistent as f64 / total_checked as f64
        };

        let is_violation = !inconsistencies.is_empty();
        if is_violation {
            self.violations += 1;
        }

        POSCheckResult {
            is_violation,
            total_tokens_checked: total_checked,
            consistent_tokens: consistent,
            inconsistent_tokens: inconsistencies,
            consistency_ratio,
            expected_changes_found,
        }
    }

    /// Align tokens by lemma matching.
    fn align_by_lemma(
        &self,
        original: &[TaggedToken],
        transformed: &[TaggedToken],
    ) -> Vec<(usize, usize)> {
        let mut alignment = Vec::new();
        let mut used_trans = std::collections::HashSet::new();

        for (i, orig) in original.iter().enumerate() {
            for (j, trans) in transformed.iter().enumerate() {
                if used_trans.contains(&j) {
                    continue;
                }
                if orig.lemma == trans.lemma {
                    alignment.push((i, j));
                    used_trans.insert(j);
                    break;
                }
            }
        }

        alignment
    }

    /// Check if two tags are in the same equivalence class.
    fn are_equivalent(&self, tag_a: &str, tag_b: &str) -> bool {
        for (_, equivalents) in &self.equivalences {
            if equivalents.contains(&tag_a.to_string())
                && equivalents.contains(&tag_b.to_string())
            {
                return true;
            }
        }
        false
    }

    pub fn violation_rate(&self) -> f64 {
        if self.total_checks == 0 { 0.0 } else { self.violations as f64 / self.total_checks as f64 }
    }
}

impl Default for POSConsistencyOracle {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_token(text: &str, lemma: &str, pos: &str, index: usize) -> TaggedToken {
        TaggedToken {
            text: text.to_string(),
            lemma: lemma.to_string(),
            pos: pos.to_string(),
            fine_pos: pos.to_string(),
            index,
        }
    }

    #[test]
    fn test_consistent_tagging() {
        let mut oracle = POSConsistencyOracle::new();
        let original = vec![
            make_token("The", "the", "DT", 0),
            make_token("cat", "cat", "NN", 1),
            make_token("sat", "sit", "VBD", 2),
        ];
        let transformed = vec![
            make_token("The", "the", "DT", 0),
            make_token("cat", "cat", "NN", 1),
            make_token("sat", "sit", "VBD", 2),
        ];
        let result = oracle.check(&original, &transformed, "topicalization");
        assert!(!result.is_violation);
        assert!((result.consistency_ratio - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_inconsistent_tagging() {
        let mut oracle = POSConsistencyOracle::new();
        let original = vec![
            make_token("cat", "cat", "NN", 0),
            make_token("runs", "run", "VBZ", 1),
        ];
        let transformed = vec![
            make_token("cat", "cat", "VB", 0), // NN → VB is wrong
            make_token("runs", "run", "VBZ", 1),
        ];
        let result = oracle.check(&original, &transformed, "clefting");
        assert!(result.is_violation);
        assert_eq!(result.inconsistent_tokens.len(), 1);
        assert_eq!(result.inconsistent_tokens[0].original_pos, "NN");
    }

    #[test]
    fn test_expected_passivization_change() {
        let mut oracle = POSConsistencyOracle::new().with_standard_expectations();
        let original = vec![
            make_token("wrote", "write", "VBD", 0),
        ];
        let transformed = vec![
            make_token("written", "write", "VBN", 0),
        ];
        let result = oracle.check(&original, &transformed, "passivization");
        assert!(!result.is_violation); // VBD→VBN is expected for passivization
        assert_eq!(result.expected_changes_found.len(), 1);
    }

    #[test]
    fn test_equivalence_classes() {
        let mut oracle = POSConsistencyOracle::new();
        let original = vec![make_token("run", "run", "VB", 0)];
        let transformed = vec![make_token("run", "run", "VBP", 0)];
        let result = oracle.check(&original, &transformed, "topicalization");
        assert!(!result.is_violation); // VB ≈ VBP
    }

    #[test]
    fn test_exempt_tags() {
        let mut oracle = POSConsistencyOracle::new();
        let original = vec![make_token(".", ".", "PUNCT", 0)];
        let transformed = vec![make_token("!", "!", "PUNCT", 0)];
        let result = oracle.check(&original, &transformed, "negation");
        // PUNCT is exempt, so no violation even though text differs.
        assert!(!result.is_violation);
    }
}
