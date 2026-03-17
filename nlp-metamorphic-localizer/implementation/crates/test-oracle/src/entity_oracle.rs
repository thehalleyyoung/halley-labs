//! Entity preservation oracle for NER pipeline stages.
//!
//! Checks whether named entities are preserved across metamorphic transformations
//! where entity preservation is expected.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// An entity span with type and text.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Entity {
    pub text: String,
    pub label: String,
    pub start: usize,
    pub end: usize,
}

impl Entity {
    pub fn new(text: impl Into<String>, label: impl Into<String>, start: usize, end: usize) -> Self {
        Self {
            text: text.into(),
            label: label.into(),
            start,
            end,
        }
    }

    pub fn token_count(&self) -> usize {
        self.text.split_whitespace().count()
    }

    /// Check if this entity overlaps with another.
    pub fn overlaps(&self, other: &Entity) -> bool {
        self.start < other.end && other.start < self.end
    }

    /// Compute the span overlap ratio with another entity.
    pub fn overlap_ratio(&self, other: &Entity) -> f64 {
        if !self.overlaps(other) {
            return 0.0;
        }
        let overlap_start = self.start.max(other.start);
        let overlap_end = self.end.min(other.end);
        let overlap_len = overlap_end - overlap_start;
        let union_len = (self.end - self.start).max(other.end - other.start);
        if union_len == 0 {
            return 0.0;
        }
        overlap_len as f64 / union_len as f64
    }
}

/// Oracle for checking entity preservation across transformations.
pub struct EntityPreservationOracle {
    /// Minimum overlap ratio to consider two entities as matching.
    overlap_threshold: f64,
    /// Whether to require exact label matching.
    strict_label_matching: bool,
    /// Whether to allow fuzzy text matching.
    fuzzy_text_matching: bool,
    /// Maximum allowed edit distance for fuzzy matching.
    max_edit_distance: usize,
    /// Entity types that are expected to be preserved.
    preserved_types: Vec<String>,
    /// Track statistics.
    total_checks: usize,
    violations: usize,
}

/// Result of an entity preservation check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityCheckResult {
    pub is_violation: bool,
    pub original_entities: Vec<Entity>,
    pub transformed_entities: Vec<Entity>,
    pub matched_pairs: Vec<EntityMatch>,
    pub missing_entities: Vec<Entity>,
    pub spurious_entities: Vec<Entity>,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
}

/// A matched pair of entities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityMatch {
    pub original: Entity,
    pub transformed: Entity,
    pub text_match: bool,
    pub label_match: bool,
    pub overlap_ratio: f64,
}

impl EntityPreservationOracle {
    pub fn new() -> Self {
        Self {
            overlap_threshold: 0.5,
            strict_label_matching: true,
            fuzzy_text_matching: false,
            max_edit_distance: 2,
            preserved_types: Vec::new(),
            total_checks: 0,
            violations: 0,
        }
    }

    pub fn with_overlap_threshold(mut self, threshold: f64) -> Self {
        self.overlap_threshold = threshold;
        self
    }

    pub fn with_fuzzy_matching(mut self, max_edit_dist: usize) -> Self {
        self.fuzzy_text_matching = true;
        self.max_edit_distance = max_edit_dist;
        self
    }

    pub fn with_preserved_types(mut self, types: Vec<String>) -> Self {
        self.preserved_types = types;
        self
    }

    pub fn with_strict_labels(mut self, strict: bool) -> Self {
        self.strict_label_matching = strict;
        self
    }

    /// Check entity preservation between original and transformed outputs.
    pub fn check(
        &mut self,
        original: &[Entity],
        transformed: &[Entity],
    ) -> EntityCheckResult {
        self.total_checks += 1;

        let filtered_original: Vec<&Entity> = if self.preserved_types.is_empty() {
            original.iter().collect()
        } else {
            original
                .iter()
                .filter(|e| self.preserved_types.contains(&e.label))
                .collect()
        };

        let filtered_transformed: Vec<&Entity> = if self.preserved_types.is_empty() {
            transformed.iter().collect()
        } else {
            transformed
                .iter()
                .filter(|e| self.preserved_types.contains(&e.label))
                .collect()
        };

        let mut matched_pairs = Vec::new();
        let mut matched_orig_indices = std::collections::HashSet::new();
        let mut matched_trans_indices = std::collections::HashSet::new();

        // Greedy matching: best overlap first.
        let mut candidates: Vec<(usize, usize, f64, bool, bool)> = Vec::new();
        for (i, orig) in filtered_original.iter().enumerate() {
            for (j, trans) in filtered_transformed.iter().enumerate() {
                let text_match = self.texts_match(&orig.text, &trans.text);
                let label_match = orig.label == trans.label;

                if text_match && (!self.strict_label_matching || label_match) {
                    let overlap = 1.0; // text-based match, full overlap.
                    candidates.push((i, j, overlap, text_match, label_match));
                }
            }
        }

        // Sort by overlap descending.
        candidates.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        for (i, j, overlap, text_match, label_match) in candidates {
            if matched_orig_indices.contains(&i) || matched_trans_indices.contains(&j) {
                continue;
            }
            matched_orig_indices.insert(i);
            matched_trans_indices.insert(j);
            matched_pairs.push(EntityMatch {
                original: filtered_original[i].clone(),
                transformed: filtered_transformed[j].clone(),
                text_match,
                label_match,
                overlap_ratio: overlap,
            });
        }

        let missing: Vec<Entity> = filtered_original
            .iter()
            .enumerate()
            .filter(|(i, _)| !matched_orig_indices.contains(i))
            .map(|(_, e)| (*e).clone())
            .collect();

        let spurious: Vec<Entity> = filtered_transformed
            .iter()
            .enumerate()
            .filter(|(i, _)| !matched_trans_indices.contains(i))
            .map(|(_, e)| (*e).clone())
            .collect();

        let precision = if filtered_transformed.is_empty() {
            1.0
        } else {
            matched_pairs.len() as f64 / filtered_transformed.len() as f64
        };

        let recall = if filtered_original.is_empty() {
            1.0
        } else {
            matched_pairs.len() as f64 / filtered_original.len() as f64
        };

        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        let is_violation = !missing.is_empty() || !spurious.is_empty();
        if is_violation {
            self.violations += 1;
        }

        EntityCheckResult {
            is_violation,
            original_entities: original.to_vec(),
            transformed_entities: transformed.to_vec(),
            matched_pairs,
            missing_entities: missing,
            spurious_entities: spurious,
            precision,
            recall,
            f1_score: f1,
        }
    }

    /// Check if two entity texts match (exact or fuzzy).
    fn texts_match(&self, a: &str, b: &str) -> bool {
        if a == b {
            return true;
        }
        if !self.fuzzy_text_matching {
            return false;
        }
        // Simple edit distance check.
        let a_lower = a.to_lowercase();
        let b_lower = b.to_lowercase();
        if a_lower == b_lower {
            return true;
        }
        self.levenshtein(&a_lower, &b_lower) <= self.max_edit_distance
    }

    fn levenshtein(&self, a: &str, b: &str) -> usize {
        let a_chars: Vec<char> = a.chars().collect();
        let b_chars: Vec<char> = b.chars().collect();
        let m = a_chars.len();
        let n = b_chars.len();
        let mut dp = vec![vec![0usize; n + 1]; m + 1];
        for i in 0..=m { dp[i][0] = i; }
        for j in 0..=n { dp[0][j] = j; }
        for i in 1..=m {
            for j in 1..=n {
                let cost = if a_chars[i - 1] == b_chars[j - 1] { 0 } else { 1 };
                dp[i][j] = (dp[i - 1][j] + 1).min(dp[i][j - 1] + 1).min(dp[i - 1][j - 1] + cost);
            }
        }
        dp[m][n]
    }

    pub fn violation_rate(&self) -> f64 {
        if self.total_checks == 0 { 0.0 } else { self.violations as f64 / self.total_checks as f64 }
    }
}

impl Default for EntityPreservationOracle {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_overlap() {
        let e1 = Entity::new("John Smith", "PERSON", 0, 10);
        let e2 = Entity::new("Smith Corp", "ORG", 5, 15);
        assert!(e1.overlaps(&e2));
        assert!(e1.overlap_ratio(&e2) > 0.0);
    }

    #[test]
    fn test_entity_no_overlap() {
        let e1 = Entity::new("John", "PERSON", 0, 4);
        let e2 = Entity::new("Corp", "ORG", 10, 14);
        assert!(!e1.overlaps(&e2));
        assert!(e1.overlap_ratio(&e2) < f64::EPSILON);
    }

    #[test]
    fn test_perfect_preservation() {
        let mut oracle = EntityPreservationOracle::new();
        let entities = vec![
            Entity::new("John", "PERSON", 0, 4),
            Entity::new("Acme", "ORG", 10, 14),
        ];
        let result = oracle.check(&entities, &entities);
        assert!(!result.is_violation);
        assert_eq!(result.matched_pairs.len(), 2);
        assert!(result.missing_entities.is_empty());
        assert!((result.f1_score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_missing_entity() {
        let mut oracle = EntityPreservationOracle::new();
        let original = vec![
            Entity::new("John", "PERSON", 0, 4),
            Entity::new("Acme", "ORG", 10, 14),
        ];
        let transformed = vec![Entity::new("John", "PERSON", 0, 4)];
        let result = oracle.check(&original, &transformed);
        assert!(result.is_violation);
        assert_eq!(result.missing_entities.len(), 1);
        assert_eq!(result.missing_entities[0].text, "Acme");
        assert!((result.recall - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_spurious_entity() {
        let mut oracle = EntityPreservationOracle::new();
        let original = vec![Entity::new("John", "PERSON", 0, 4)];
        let transformed = vec![
            Entity::new("John", "PERSON", 0, 4),
            Entity::new("Smith", "PERSON", 5, 10),
        ];
        let result = oracle.check(&original, &transformed);
        assert!(result.is_violation);
        assert_eq!(result.spurious_entities.len(), 1);
    }

    #[test]
    fn test_fuzzy_matching() {
        let mut oracle = EntityPreservationOracle::new().with_fuzzy_matching(2);
        let original = vec![Entity::new("John", "PERSON", 0, 4)];
        let transformed = vec![Entity::new("john", "PERSON", 0, 4)];
        let result = oracle.check(&original, &transformed);
        assert!(!result.is_violation);
    }

    #[test]
    fn test_label_mismatch_strict() {
        let mut oracle = EntityPreservationOracle::new().with_strict_labels(true);
        let original = vec![Entity::new("Apple", "ORG", 0, 5)];
        let transformed = vec![Entity::new("Apple", "PRODUCT", 0, 5)];
        let result = oracle.check(&original, &transformed);
        assert!(result.is_violation); // label mismatch in strict mode
    }

    #[test]
    fn test_label_mismatch_lenient() {
        let mut oracle = EntityPreservationOracle::new().with_strict_labels(false);
        let original = vec![Entity::new("Apple", "ORG", 0, 5)];
        let transformed = vec![Entity::new("Apple", "PRODUCT", 0, 5)];
        let result = oracle.check(&original, &transformed);
        assert!(!result.is_violation); // label mismatch OK in lenient mode
    }

    #[test]
    fn test_filtered_types() {
        let mut oracle = EntityPreservationOracle::new()
            .with_preserved_types(vec!["PERSON".to_string()]);
        let original = vec![
            Entity::new("John", "PERSON", 0, 4),
            Entity::new("Acme", "ORG", 10, 14),
        ];
        // Only John present, Acme (ORG) not checked.
        let transformed = vec![Entity::new("John", "PERSON", 0, 4)];
        let result = oracle.check(&original, &transformed);
        assert!(!result.is_violation); // Only PERSON types checked
    }
}
