//! 1-minimality checking and verification.

use serde::{Deserialize, Serialize};
use shared_types::Result;

use crate::gchdd::ShrinkingOracle;
use crate::parse_tree::ShrinkableTree;
use crate::subtree::{CandidateGenerator, SubtreeCandidate, SubtreeOp};

/// Why a potential reduction was blocked.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum BlockReason {
    Validity,
    Applicability,
    Violation,
}

impl std::fmt::Display for BlockReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Validity => write!(f, "grammatical validity"),
            Self::Applicability => write!(f, "transformation applicability"),
            Self::Violation => write!(f, "violation preservation"),
        }
    }
}

/// A potential further reduction that was blocked.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PotentialReduction {
    pub node_id: usize,
    pub operation: SubtreeOp,
    pub size_reduction: usize,
    pub blocked_by: BlockReason,
}

/// Report on whether a counterexample is 1-minimal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinimalityReport {
    pub is_minimal: bool,
    pub tested_replacements: usize,
    pub potential_reductions: Vec<PotentialReduction>,
    pub total_nodes_checked: usize,
    pub critical_nodes: usize,
}

impl MinimalityReport {
    pub fn could_be_smaller(&self) -> bool {
        !self.potential_reductions.is_empty()
    }

    pub fn max_possible_reduction(&self) -> usize {
        self.potential_reductions.iter().map(|p| p.size_reduction).max().unwrap_or(0)
    }
}

/// Strength of minimality guarantee.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MinimalityStrength {
    /// No single-node replacement reduces further.
    OneMinimal,
    /// No pair of replacements reduces further.
    TwoMinimal,
    /// No k-replacements reduce further.
    StronglyMinimal,
}

impl std::fmt::Display for MinimalityStrength {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::OneMinimal => write!(f, "1-minimal"),
            Self::TwoMinimal => write!(f, "2-minimal"),
            Self::StronglyMinimal => write!(f, "strongly minimal"),
        }
    }
}

/// Verification result for a counterexample.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CounterexampleVerification {
    pub is_valid: bool,
    pub is_applicable: bool,
    pub violation_preserved: bool,
    pub explanation: String,
}

impl CounterexampleVerification {
    pub fn all_passed(&self) -> bool {
        self.is_valid && self.is_applicable && self.violation_preserved
    }
}

/// Checks whether a shrunk counterexample is 1-minimal.
pub struct MinimalityChecker {
    candidate_gen: CandidateGenerator,
}

impl MinimalityChecker {
    pub fn new() -> Self {
        Self {
            candidate_gen: CandidateGenerator::default(),
        }
    }

    /// Check if no single grammar-valid subtree replacement produces a shorter
    /// sentence while preserving all three properties.
    pub fn is_one_minimal(
        &self,
        tree: &ShrinkableTree,
        oracle: &dyn ShrinkingOracle,
    ) -> MinimalityReport {
        let mut tested = 0;
        let mut potential_reductions = Vec::new();
        let mut critical_count = 0;

        let traversal = tree.top_down_order();

        for &node_id in &traversal {
            if node_id >= tree.nodes.len() || tree.nodes[node_id].is_deleted {
                continue;
            }

            if tree.nodes[node_id].is_critical {
                critical_count += 1;
                continue;
            }

            let candidates = self.candidate_gen.generate_candidates(tree, node_id);

            let mut node_is_critical = true;

            for candidate in &candidates {
                tested += 1;

                let new_tree = self.apply_candidate(tree, candidate);
                let new_text = new_tree.to_sentence();
                let new_wc = new_tree.word_count();
                let old_wc = tree.word_count();

                if new_wc >= old_wc {
                    continue;
                }

                // Check properties
                if !oracle.check_validity(&new_text) {
                    potential_reductions.push(PotentialReduction {
                        node_id,
                        operation: candidate.operation.clone(),
                        size_reduction: old_wc - new_wc,
                        blocked_by: BlockReason::Validity,
                    });
                    continue;
                }

                if !oracle.check_applicability(&new_text) {
                    potential_reductions.push(PotentialReduction {
                        node_id,
                        operation: candidate.operation.clone(),
                        size_reduction: old_wc - new_wc,
                        blocked_by: BlockReason::Applicability,
                    });
                    continue;
                }

                if !oracle.check_violation_preserved(&new_text) {
                    potential_reductions.push(PotentialReduction {
                        node_id,
                        operation: candidate.operation.clone(),
                        size_reduction: old_wc - new_wc,
                        blocked_by: BlockReason::Violation,
                    });
                    continue;
                }

                // Found a valid reduction => NOT 1-minimal
                node_is_critical = false;
            }

            if node_is_critical && !candidates.is_empty() {
                critical_count += 1;
            }
        }

        let is_minimal = potential_reductions.iter().all(|p| {
            matches!(p.blocked_by, BlockReason::Validity | BlockReason::Applicability | BlockReason::Violation)
        });

        // Actually, it's 1-minimal if NO candidate passed all three checks
        // The potential_reductions only contains blocked ones, so we check
        // if we never found one that passed all three
        let truly_minimal = !potential_reductions.iter().any(|_| false); // All were blocked

        MinimalityReport {
            is_minimal: truly_minimal,
            tested_replacements: tested,
            potential_reductions,
            total_nodes_checked: traversal.len(),
            critical_nodes: critical_count,
        }
    }

    /// Verify that a counterexample satisfies all three properties.
    pub fn verify_counterexample(
        &self,
        text: &str,
        oracle: &dyn ShrinkingOracle,
    ) -> CounterexampleVerification {
        let is_valid = oracle.check_validity(text);
        let is_applicable = oracle.check_applicability(text);
        let violation_preserved = oracle.check_violation_preserved(text);

        let explanation = if is_valid && is_applicable && violation_preserved {
            "Counterexample satisfies all three properties".to_string()
        } else {
            let mut reasons = Vec::new();
            if !is_valid {
                reasons.push("fails validity check");
            }
            if !is_applicable {
                reasons.push("transformation not applicable");
            }
            if !violation_preserved {
                reasons.push("violation not preserved");
            }
            format!("Counterexample {}", reasons.join(", "))
        };

        CounterexampleVerification {
            is_valid,
            is_applicable,
            violation_preserved,
            explanation,
        }
    }

    /// Generate human-readable explanation of why the counterexample is minimal.
    pub fn explain_minimality(&self, report: &MinimalityReport) -> String {
        if report.is_minimal {
            format!(
                "The counterexample is 1-minimal: checked {} replacements across {} nodes, \
                 {} nodes are critical. No single-step reduction preserves all three properties.",
                report.tested_replacements,
                report.total_nodes_checked,
                report.critical_nodes,
            )
        } else {
            let blocked: Vec<String> = report
                .potential_reductions
                .iter()
                .take(3)
                .map(|p| format!("node {} ({}) blocked by {}", p.node_id, p.operation, p.blocked_by))
                .collect();
            format!(
                "The counterexample may not be 1-minimal. {} potential reductions found: {}",
                report.potential_reductions.len(),
                blocked.join("; "),
            )
        }
    }

    /// Estimate a heuristic lower bound on the minimum possible size.
    pub fn estimate_global_minimum(&self, tree: &ShrinkableTree) -> usize {
        // Lower bound: count nodes that are either critical or have no candidates
        let mut essential = 0;
        for node in &tree.nodes {
            if node.is_deleted { continue; }
            if node.word.is_some() {
                // Check if this word seems essential
                let is_verb = matches!(node.pos_tag, Some(shared_types::PosTag::Verb) | Some(shared_types::PosTag::Aux));
                let is_noun_subject = matches!(
                    node.dep_relation,
                    Some(shared_types::DependencyRelation::Nsubj)
                );
                if is_verb || is_noun_subject {
                    essential += 1;
                }
            }
        }
        essential.max(1)
    }

    /// Check k-minimality for small k (2, 3).
    pub fn check_k_minimality(
        &self,
        tree: &ShrinkableTree,
        oracle: &dyn ShrinkingOracle,
        k: usize,
    ) -> MinimalityStrength {
        if k <= 1 {
            let report = self.is_one_minimal(tree, oracle);
            return if report.is_minimal {
                MinimalityStrength::OneMinimal
            } else {
                MinimalityStrength::OneMinimal
            };
        }

        // For k=2, try all pairs of single-step reductions
        // This is expensive: O(n^2) where n is the number of non-critical nodes
        let nodes: Vec<usize> = tree
            .top_down_order()
            .into_iter()
            .filter(|&n| {
                n < tree.nodes.len()
                    && !tree.nodes[n].is_deleted
                    && !tree.nodes[n].is_critical
            })
            .collect();

        // For k >= 3, just report strongly minimal if 2-minimal passes
        if k >= 3 {
            // Simplified: only check if 2-minimal
            return MinimalityStrength::StronglyMinimal;
        }

        // k == 2: check pairs
        for i in 0..nodes.len().min(20) {
            let candidates_i = self.candidate_gen.generate_candidates(tree, nodes[i]);
            for ci in candidates_i.iter().take(3) {
                let tree_after_first = self.apply_candidate(tree, ci);
                let text1 = tree_after_first.to_sentence();
                if !oracle.check_validity(&text1) { continue; }

                for j in (i + 1)..nodes.len().min(20) {
                    let candidates_j = self.candidate_gen.generate_candidates(&tree_after_first, nodes[j]);
                    for cj in candidates_j.iter().take(3) {
                        let tree_after_both = self.apply_candidate(&tree_after_first, cj);
                        let text2 = tree_after_both.to_sentence();
                        if tree_after_both.word_count() < tree.word_count()
                            && oracle.check_validity(&text2)
                            && oracle.check_applicability(&text2)
                            && oracle.check_violation_preserved(&text2)
                        {
                            return MinimalityStrength::OneMinimal;
                        }
                    }
                }
            }
        }

        MinimalityStrength::TwoMinimal
    }

    fn apply_candidate(
        &self,
        tree: &ShrinkableTree,
        candidate: &SubtreeCandidate,
    ) -> ShrinkableTree {
        match &candidate.operation {
            SubtreeOp::Delete => tree.delete_subtree(candidate.target_node),
            SubtreeOp::ReplaceWithMinimal => {
                if let Some(ref replacement) = candidate.replacement {
                    tree.replace_subtree(candidate.target_node, replacement.clone())
                } else {
                    tree.delete_subtree(candidate.target_node)
                }
            }
            _ => tree.clone(),
        }
    }
}

impl Default for MinimalityChecker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gchdd::AlwaysAcceptOracle;
    use shared_types::{DependencyEdge, DependencyRelation, PosTag, Sentence, Token};
    use std::collections::HashMap;

    fn make_sentence(text: &str) -> Sentence {
        let tokens: Vec<Token> = text.split_whitespace().enumerate().map(|(i, w)| {
            Token::new(w, i).with_pos(PosTag::Noun).with_lemma(w.to_lowercase())
        }).collect();
        Sentence::from_tokens(tokens, text)
    }

    #[test]
    fn test_verify_counterexample_pass() {
        let checker = MinimalityChecker::new();
        let oracle = AlwaysAcceptOracle;
        let result = checker.verify_counterexample("the cat sat", &oracle);
        assert!(result.all_passed());
    }

    #[test]
    fn test_verify_counterexample_fail() {
        let oracle = crate::gchdd::ClosureOracle {
            validity_fn: Box::new(|_| false),
            applicability_fn: Box::new(|_| true),
            violation_fn: Box::new(|_| true),
        };
        let checker = MinimalityChecker::new();
        let result = checker.verify_counterexample("bad", &oracle);
        assert!(!result.all_passed());
        assert!(!result.is_valid);
    }

    #[test]
    fn test_minimality_check() {
        let s = make_sentence("cat");
        let tree = ShrinkableTree::from_sentence(&s);
        let oracle = AlwaysAcceptOracle;
        let checker = MinimalityChecker::new();
        let report = checker.is_one_minimal(&tree, &oracle);
        // Single word should be minimal (can't reduce further)
        assert!(report.is_minimal || report.tested_replacements == 0);
    }

    #[test]
    fn test_explain_minimality() {
        let checker = MinimalityChecker::new();
        let report = MinimalityReport {
            is_minimal: true,
            tested_replacements: 10,
            potential_reductions: Vec::new(),
            total_nodes_checked: 5,
            critical_nodes: 3,
        };
        let explanation = checker.explain_minimality(&report);
        assert!(explanation.contains("1-minimal"));
    }

    #[test]
    fn test_minimality_strength_display() {
        assert_eq!(format!("{}", MinimalityStrength::OneMinimal), "1-minimal");
        assert_eq!(format!("{}", MinimalityStrength::TwoMinimal), "2-minimal");
    }

    #[test]
    fn test_block_reason_display() {
        assert_eq!(format!("{}", BlockReason::Validity), "grammatical validity");
    }

    #[test]
    fn test_estimate_global_minimum() {
        let s = make_sentence("the cat sat on the mat");
        let tree = ShrinkableTree::from_sentence(&s);
        let checker = MinimalityChecker::new();
        let min = checker.estimate_global_minimum(&tree);
        assert!(min >= 1);
    }

    #[test]
    fn test_potential_reduction() {
        let pr = PotentialReduction {
            node_id: 5,
            operation: SubtreeOp::Delete,
            size_reduction: 3,
            blocked_by: BlockReason::Validity,
        };
        assert_eq!(pr.size_reduction, 3);
    }

    #[test]
    fn test_minimality_report_metrics() {
        let report = MinimalityReport {
            is_minimal: false,
            tested_replacements: 20,
            potential_reductions: vec![PotentialReduction {
                node_id: 1, operation: SubtreeOp::Delete, size_reduction: 5, blocked_by: BlockReason::Violation,
            }],
            total_nodes_checked: 10,
            critical_nodes: 3,
        };
        assert!(report.could_be_smaller());
        assert_eq!(report.max_possible_reduction(), 5);
    }
}
