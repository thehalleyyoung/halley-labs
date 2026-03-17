//! Structural oracle for dependency parse tree comparisons.
//!
//! Checks whether the structural properties of dependency trees are preserved
//! across metamorphic transformations.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A dependency arc in a parse tree.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DepArc {
    pub head_index: usize,
    pub dependent_index: usize,
    pub relation: String,
}

/// A dependency tree represented as a list of arcs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepTree {
    pub tokens: Vec<String>,
    pub arcs: Vec<DepArc>,
    pub root: usize,
}

impl DepTree {
    pub fn new(tokens: Vec<String>, arcs: Vec<DepArc>, root: usize) -> Self {
        Self { tokens, arcs, root }
    }

    /// Get the depth of the tree.
    pub fn depth(&self) -> usize {
        self.subtree_depth(self.root)
    }

    fn subtree_depth(&self, node: usize) -> usize {
        let children: Vec<usize> = self
            .arcs
            .iter()
            .filter(|a| a.head_index == node)
            .map(|a| a.dependent_index)
            .collect();
        if children.is_empty() {
            return 0;
        }
        1 + children
            .iter()
            .map(|&c| self.subtree_depth(c))
            .max()
            .unwrap_or(0)
    }

    /// Get the number of dependents for a node.
    pub fn degree(&self, node: usize) -> usize {
        self.arcs.iter().filter(|a| a.head_index == node).count()
    }

    /// Get the relation labels present in the tree.
    pub fn relation_set(&self) -> Vec<String> {
        let mut rels: Vec<String> = self
            .arcs
            .iter()
            .map(|a| a.relation.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        rels.sort();
        rels
    }

    /// Check if the tree has a subject relation.
    pub fn has_subject(&self) -> bool {
        self.arcs
            .iter()
            .any(|a| a.relation == "nsubj" || a.relation == "nsubjpass" || a.relation == "csubj")
    }

    /// Get the dependents of a node.
    pub fn dependents(&self, node: usize) -> Vec<(usize, String)> {
        self.arcs
            .iter()
            .filter(|a| a.head_index == node)
            .map(|a| (a.dependent_index, a.relation.clone()))
            .collect()
    }

    /// Compute tree edit distance with another tree (simplified Zhang-Shasha).
    pub fn tree_edit_distance(&self, other: &DepTree) -> usize {
        // Simplified: count arcs that differ.
        let self_arcs: std::collections::HashSet<(usize, String)> = self
            .arcs
            .iter()
            .map(|a| (a.head_index, a.relation.clone()))
            .collect();
        let other_arcs: std::collections::HashSet<(usize, String)> = other
            .arcs
            .iter()
            .map(|a| (a.head_index, a.relation.clone()))
            .collect();

        let only_self = self_arcs.difference(&other_arcs).count();
        let only_other = other_arcs.difference(&self_arcs).count();
        only_self + only_other
    }
}

/// Oracle for checking structural properties of dependency trees.
pub struct StructuralOracle {
    /// Maximum allowed depth difference.
    max_depth_diff: usize,
    /// Whether to check relation preservation.
    check_relations: bool,
    /// Relations that must be preserved.
    required_relations: Vec<String>,
    /// Maximum allowed tree edit distance ratio.
    max_edit_distance_ratio: f64,
    total_checks: usize,
    violations: usize,
}

/// Result of a structural check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralCheckResult {
    pub is_violation: bool,
    pub depth_original: usize,
    pub depth_transformed: usize,
    pub depth_diff: usize,
    pub relations_original: Vec<String>,
    pub relations_transformed: Vec<String>,
    pub missing_relations: Vec<String>,
    pub added_relations: Vec<String>,
    pub tree_edit_distance: usize,
    pub edit_distance_ratio: f64,
    pub issues: Vec<String>,
}

impl StructuralOracle {
    pub fn new() -> Self {
        Self {
            max_depth_diff: 3,
            check_relations: true,
            required_relations: vec![
                "nsubj".into(),
                "dobj".into(),
                "ROOT".into(),
            ],
            max_edit_distance_ratio: 0.5,
            total_checks: 0,
            violations: 0,
        }
    }

    pub fn with_max_depth_diff(mut self, diff: usize) -> Self {
        self.max_depth_diff = diff;
        self
    }

    pub fn with_required_relations(mut self, rels: Vec<String>) -> Self {
        self.required_relations = rels;
        self
    }

    pub fn with_max_edit_ratio(mut self, ratio: f64) -> Self {
        self.max_edit_distance_ratio = ratio;
        self
    }

    /// Check structural preservation between two dependency trees.
    pub fn check(
        &mut self,
        original: &DepTree,
        transformed: &DepTree,
    ) -> StructuralCheckResult {
        self.total_checks += 1;
        let mut issues = Vec::new();

        let depth_orig = original.depth();
        let depth_trans = transformed.depth();
        let depth_diff = (depth_orig as i64 - depth_trans as i64).unsigned_abs() as usize;

        if depth_diff > self.max_depth_diff {
            issues.push(format!(
                "Depth difference {} exceeds maximum {}",
                depth_diff, self.max_depth_diff
            ));
        }

        let rels_orig = original.relation_set();
        let rels_trans = transformed.relation_set();

        let missing: Vec<String> = rels_orig
            .iter()
            .filter(|r| !rels_trans.contains(r))
            .cloned()
            .collect();
        let added: Vec<String> = rels_trans
            .iter()
            .filter(|r| !rels_orig.contains(r))
            .cloned()
            .collect();

        if self.check_relations {
            for req in &self.required_relations {
                if rels_orig.contains(req) && !rels_trans.contains(req) {
                    issues.push(format!("Required relation '{}' missing in transformed tree", req));
                }
            }
        }

        let ted = original.tree_edit_distance(transformed);
        let max_arcs = original.arcs.len().max(transformed.arcs.len()).max(1);
        let edit_ratio = ted as f64 / max_arcs as f64;

        if edit_ratio > self.max_edit_distance_ratio {
            issues.push(format!(
                "Tree edit distance ratio {:.2} exceeds maximum {:.2}",
                edit_ratio, self.max_edit_distance_ratio
            ));
        }

        let is_violation = !issues.is_empty();
        if is_violation {
            self.violations += 1;
        }

        StructuralCheckResult {
            is_violation,
            depth_original: depth_orig,
            depth_transformed: depth_trans,
            depth_diff,
            relations_original: rels_orig,
            relations_transformed: rels_trans,
            missing_relations: missing,
            added_relations: added,
            tree_edit_distance: ted,
            edit_distance_ratio: edit_ratio,
            issues,
        }
    }

    pub fn violation_rate(&self) -> f64 {
        if self.total_checks == 0 { 0.0 } else { self.violations as f64 / self.total_checks as f64 }
    }
}

impl Default for StructuralOracle {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tree(tokens: &[&str], arcs: Vec<(usize, usize, &str)>, root: usize) -> DepTree {
        DepTree::new(
            tokens.iter().map(|s| s.to_string()).collect(),
            arcs.into_iter()
                .map(|(h, d, r)| DepArc {
                    head_index: h,
                    dependent_index: d,
                    relation: r.to_string(),
                })
                .collect(),
            root,
        )
    }

    #[test]
    fn test_tree_depth() {
        // "The cat sat" with root=2 (sat), cat→2, the→1
        let tree = make_tree(
            &["The", "cat", "sat"],
            vec![(2, 1, "nsubj"), (1, 0, "det")],
            2,
        );
        assert_eq!(tree.depth(), 2); // sat → cat → The
    }

    #[test]
    fn test_tree_relation_set() {
        let tree = make_tree(
            &["The", "cat", "sat"],
            vec![(2, 1, "nsubj"), (1, 0, "det")],
            2,
        );
        let rels = tree.relation_set();
        assert!(rels.contains(&"nsubj".to_string()));
        assert!(rels.contains(&"det".to_string()));
    }

    #[test]
    fn test_structural_check_pass() {
        let mut oracle = StructuralOracle::new();
        let orig = make_tree(
            &["The", "cat", "sat"],
            vec![(2, 1, "nsubj"), (1, 0, "det")],
            2,
        );
        let trans = make_tree(
            &["The", "cat", "sat"],
            vec![(2, 1, "nsubj"), (1, 0, "det")],
            2,
        );
        let result = oracle.check(&orig, &trans);
        assert!(!result.is_violation);
        assert_eq!(result.tree_edit_distance, 0);
    }

    #[test]
    fn test_structural_check_depth_violation() {
        let mut oracle = StructuralOracle::new().with_max_depth_diff(0);
        let orig = make_tree(
            &["The", "cat", "sat"],
            vec![(2, 1, "nsubj"), (1, 0, "det")],
            2,
        );
        let trans = make_tree(
            &["cat", "sat"],
            vec![(1, 0, "nsubj")],
            1,
        );
        let result = oracle.check(&orig, &trans);
        assert!(result.is_violation);
        assert!(result.depth_diff > 0);
    }

    #[test]
    fn test_tree_edit_distance() {
        let t1 = make_tree(
            &["a", "b", "c"],
            vec![(2, 1, "nsubj"), (1, 0, "det")],
            2,
        );
        let t2 = make_tree(
            &["a", "b", "c"],
            vec![(2, 1, "dobj"), (1, 0, "det")],
            2,
        );
        let dist = t1.tree_edit_distance(&t2);
        assert!(dist > 0);
    }

    #[test]
    fn test_has_subject() {
        let tree = make_tree(
            &["cat", "sat"],
            vec![(1, 0, "nsubj")],
            1,
        );
        assert!(tree.has_subject());

        let no_subj = make_tree(
            &["sat"],
            vec![],
            0,
        );
        assert!(!no_subj.has_subject());
    }
}
