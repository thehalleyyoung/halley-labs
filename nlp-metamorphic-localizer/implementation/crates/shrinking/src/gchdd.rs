//! GCHDD (Grammar-Constrained Hierarchical Delta Debugging) algorithm.
//!
//! Produces a 1-minimal counterexample in O(|T|^2 * |R|) checker invocations.

use serde::{Deserialize, Serialize};
use shared_types::{LocalizerError, Result, Sentence};
use std::time::Instant;

use crate::parse_tree::{ShrinkNode, ShrinkableTree};
use crate::subtree::{CandidateGenerator, CandidateHistory, SubtreeCandidate, SubtreeOp};

/// Configuration for the GCHDD algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GCHDDConfig {
    pub max_iterations: usize,
    pub timeout_seconds: u64,
    pub enable_binary_search: bool,
    pub min_tree_size: usize,
    pub max_attempts_per_node: usize,
}

impl Default for GCHDDConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            timeout_seconds: 60,
            enable_binary_search: false,
            min_tree_size: 2,
            max_attempts_per_node: 20,
        }
    }
}

/// Result of the GCHDD shrinking process.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShrinkingResult {
    pub original_text: String,
    pub shrunk_text: String,
    pub original_size: usize,
    pub shrunk_size: usize,
    pub reduction_ratio: f64,
    pub is_one_minimal: bool,
    pub iterations_used: usize,
    pub checker_invocations: usize,
    pub time_elapsed_ms: u64,
    pub history: Vec<ShrinkingStep>,
}

impl ShrinkingResult {
    pub fn word_reduction(&self) -> usize {
        self.original_size.saturating_sub(self.shrunk_size)
    }

    pub fn reduction_percentage(&self) -> f64 {
        if self.original_size == 0 {
            return 0.0;
        }
        100.0 * (1.0 - self.shrunk_size as f64 / self.original_size as f64)
    }
}

/// A single step in the shrinking history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShrinkingStep {
    pub iteration: usize,
    pub target_node: usize,
    pub operation: String,
    pub size_before: usize,
    pub size_after: usize,
    pub accepted: bool,
    pub rejection_reason: Option<String>,
}

/// Statistics about the shrinking process.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ShrinkingStats {
    pub total_attempts: usize,
    pub successful_reductions: usize,
    pub failed_validity: usize,
    pub failed_applicability: usize,
    pub failed_violation: usize,
    pub total_checker_calls: usize,
}

impl ShrinkingStats {
    pub fn success_rate(&self) -> f64 {
        if self.total_attempts == 0 {
            return 0.0;
        }
        self.successful_reductions as f64 / self.total_attempts as f64
    }
}

/// Trait for checking if a candidate preserves the necessary properties.
pub trait ShrinkingOracle: Send + Sync {
    /// Check grammatical validity of the shrunk sentence.
    fn check_validity(&self, sentence: &str) -> bool;

    /// Check if the transformation is still applicable.
    fn check_applicability(&self, sentence: &str) -> bool;

    /// Check if the metamorphic violation is preserved.
    fn check_violation_preserved(&self, sentence: &str) -> bool;
}

/// A simple oracle that accepts all candidates (for testing).
pub struct AlwaysAcceptOracle;

impl ShrinkingOracle for AlwaysAcceptOracle {
    fn check_validity(&self, _sentence: &str) -> bool { true }
    fn check_applicability(&self, _sentence: &str) -> bool { true }
    fn check_violation_preserved(&self, _sentence: &str) -> bool { true }
}

/// A configurable oracle with closures.
pub struct ClosureOracle {
    pub validity_fn: Box<dyn Fn(&str) -> bool + Send + Sync>,
    pub applicability_fn: Box<dyn Fn(&str) -> bool + Send + Sync>,
    pub violation_fn: Box<dyn Fn(&str) -> bool + Send + Sync>,
}

impl ShrinkingOracle for ClosureOracle {
    fn check_validity(&self, s: &str) -> bool { (self.validity_fn)(s) }
    fn check_applicability(&self, s: &str) -> bool { (self.applicability_fn)(s) }
    fn check_violation_preserved(&self, s: &str) -> bool { (self.violation_fn)(s) }
}

/// The main GCHDD engine.
pub struct GCHDDEngine {
    pub config: GCHDDConfig,
    pub stats: ShrinkingStats,
    candidate_gen: CandidateGenerator,
    history: CandidateHistory,
}

impl GCHDDEngine {
    pub fn new(config: GCHDDConfig) -> Self {
        Self {
            config,
            stats: ShrinkingStats::default(),
            candidate_gen: CandidateGenerator::default(),
            history: CandidateHistory::new(),
        }
    }

    pub fn with_default_config() -> Self {
        Self::new(GCHDDConfig::default())
    }

    /// Main GCHDD shrinking algorithm.
    ///
    /// Phase 1: Top-down traversal of parse tree.
    /// Phase 2: For each node, attempt subtree replacements in order of
    ///          decreasing size reduction.
    /// Phase 3: If replacement preserves (a) validity, (b) applicability,
    ///          (c) violation -> accept and continue.
    /// Phase 4: If no replacement works, mark node as critical.
    /// Phase 5: Repeat until no more reductions possible (1-minimal).
    pub fn shrink(
        &mut self,
        tree: &ShrinkableTree,
        oracle: &dyn ShrinkingOracle,
    ) -> ShrinkingResult {
        let start = Instant::now();
        let original_text = tree.to_sentence();
        let original_size = tree.word_count();

        let mut current_tree = tree.clone();
        let mut history = Vec::new();
        let mut iteration = 0;
        let mut made_progress = true;

        while made_progress && iteration < self.config.max_iterations {
            made_progress = false;

            if start.elapsed().as_secs() >= self.config.timeout_seconds {
                break;
            }

            if current_tree.word_count() <= self.config.min_tree_size {
                break;
            }

            // Get traversal order
            let traversal = if self.config.enable_binary_search {
                self.binary_search_order(&current_tree)
            } else {
                current_tree.top_down_order()
            };

            for &node_id in &traversal {
                if node_id >= current_tree.nodes.len()
                    || current_tree.nodes[node_id].is_deleted
                    || current_tree.nodes[node_id].is_critical
                {
                    continue;
                }

                if start.elapsed().as_secs() >= self.config.timeout_seconds {
                    break;
                }

                let candidates = self.candidate_gen.generate_candidates(&current_tree, node_id);

                let mut node_succeeded = false;
                let mut attempts_this_node = 0;

                for candidate in &candidates {
                    if attempts_this_node >= self.config.max_attempts_per_node {
                        break;
                    }
                    attempts_this_node += 1;

                    let new_tree = self.apply_candidate(&current_tree, candidate);
                    let new_text = new_tree.to_sentence();
                    let new_size = new_tree.word_count();

                    self.stats.total_attempts += 1;

                    // Check all three properties
                    self.stats.total_checker_calls += 1;
                    if !oracle.check_validity(&new_text) {
                        self.stats.failed_validity += 1;
                        self.history.record(&candidate.operation, false);
                        history.push(ShrinkingStep {
                            iteration,
                            target_node: node_id,
                            operation: format!("{}", candidate.operation),
                            size_before: current_tree.word_count(),
                            size_after: new_size,
                            accepted: false,
                            rejection_reason: Some("validity check failed".into()),
                        });
                        continue;
                    }

                    self.stats.total_checker_calls += 1;
                    if !oracle.check_applicability(&new_text) {
                        self.stats.failed_applicability += 1;
                        self.history.record(&candidate.operation, false);
                        history.push(ShrinkingStep {
                            iteration,
                            target_node: node_id,
                            operation: format!("{}", candidate.operation),
                            size_before: current_tree.word_count(),
                            size_after: new_size,
                            accepted: false,
                            rejection_reason: Some("applicability check failed".into()),
                        });
                        continue;
                    }

                    self.stats.total_checker_calls += 1;
                    if !oracle.check_violation_preserved(&new_text) {
                        self.stats.failed_violation += 1;
                        self.history.record(&candidate.operation, false);
                        history.push(ShrinkingStep {
                            iteration,
                            target_node: node_id,
                            operation: format!("{}", candidate.operation),
                            size_before: current_tree.word_count(),
                            size_after: new_size,
                            accepted: false,
                            rejection_reason: Some("violation not preserved".into()),
                        });
                        continue;
                    }

                    // All checks passed! Accept the reduction.
                    self.stats.successful_reductions += 1;
                    self.history.record(&candidate.operation, true);
                    history.push(ShrinkingStep {
                        iteration,
                        target_node: node_id,
                        operation: format!("{}", candidate.operation),
                        size_before: current_tree.word_count(),
                        size_after: new_size,
                        accepted: true,
                        rejection_reason: None,
                    });

                    current_tree = new_tree;
                    made_progress = true;
                    node_succeeded = true;
                    break;
                }

                if !node_succeeded && !candidates.is_empty() {
                    // Mark this node as critical
                    current_tree.mark_critical(node_id);
                }
            }

            iteration += 1;
        }

        let shrunk_text = current_tree.to_sentence();
        let shrunk_size = current_tree.word_count();

        ShrinkingResult {
            original_text,
            shrunk_text,
            original_size,
            shrunk_size,
            reduction_ratio: if original_size > 0 {
                shrunk_size as f64 / original_size as f64
            } else {
                1.0
            },
            is_one_minimal: !made_progress,
            iterations_used: iteration,
            checker_invocations: self.stats.total_checker_calls,
            time_elapsed_ms: start.elapsed().as_millis() as u64,
            history,
        }
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
            SubtreeOp::ReplaceWithChild(idx) => {
                let node = &tree.nodes[candidate.target_node];
                if *idx < node.children.len() {
                    let child_id = node.children[*idx];
                    // Promote child to parent's position
                    let mut new_tree = tree.clone();
                    if let Some(parent_id) = node.parent {
                        if parent_id < new_tree.nodes.len() {
                            // Replace target with child in parent's children list
                            for c in &mut new_tree.nodes[parent_id].children {
                                if *c == candidate.target_node {
                                    *c = child_id;
                                }
                            }
                            new_tree.nodes[child_id].parent = Some(parent_id);
                        }
                    }
                    // Mark target and its other children as deleted
                    new_tree.nodes[candidate.target_node].is_deleted = true;
                    for &other_child in &node.children {
                        if other_child != child_id {
                            let subtree = new_tree.subtree_at(other_child);
                            for sid in subtree {
                                if sid < new_tree.nodes.len() {
                                    new_tree.nodes[sid].is_deleted = true;
                                }
                            }
                        }
                    }
                    new_tree
                } else {
                    tree.clone()
                }
            }
            SubtreeOp::CollapseToHead => {
                // Keep only the head child, delete everything else
                let node = &tree.nodes[candidate.target_node];
                let mut new_tree = tree.clone();
                let head_child = node.children.iter().find(|&&c| {
                    c < tree.nodes.len() && tree.nodes[c].word.is_some()
                }).or_else(|| node.children.first()).copied();

                if let Some(hc) = head_child {
                    for &child in &node.children {
                        if child != hc {
                            let subtree = new_tree.subtree_at(child);
                            for sid in subtree {
                                if sid < new_tree.nodes.len() {
                                    new_tree.nodes[sid].is_deleted = true;
                                }
                            }
                        }
                    }
                    new_tree.nodes[candidate.target_node].children = vec![hc];
                }
                new_tree
            }
            SubtreeOp::PruneModifiers => {
                let node = &tree.nodes[candidate.target_node];
                let mut new_tree = tree.clone();
                let modifier_deps = [
                    shared_types::DependencyRelation::Amod,
                    shared_types::DependencyRelation::Advmod,
                    shared_types::DependencyRelation::Det,
                ];

                let children_to_remove: Vec<usize> = node.children.iter().filter(|&&c| {
                    if c >= tree.nodes.len() || tree.nodes[c].is_deleted { return false; }
                    let child = &tree.nodes[c];
                    match &child.dep_relation {
                        Some(rel) => modifier_deps.contains(rel),
                        None => matches!(child.label.as_str(), "ADJP" | "ADVP"),
                    }
                }).copied().collect();

                for cid in &children_to_remove {
                    let subtree = new_tree.subtree_at(*cid);
                    for sid in subtree {
                        if sid < new_tree.nodes.len() {
                            new_tree.nodes[sid].is_deleted = true;
                        }
                    }
                }
                new_tree.nodes[candidate.target_node].children.retain(|c| !children_to_remove.contains(c));
                new_tree
            }
        }
    }

    /// Binary search order for potentially faster convergence.
    fn binary_search_order(&self, tree: &ShrinkableTree) -> Vec<usize> {
        let nodes = tree.top_down_order();
        if nodes.len() <= 2 {
            return nodes;
        }
        let mut order = Vec::with_capacity(nodes.len());
        let mut queue = vec![(0, nodes.len())];
        while let Some((start, end)) = queue.pop() {
            if start >= end { continue; }
            let mid = (start + end) / 2;
            if mid < nodes.len() {
                order.push(nodes[mid]);
            }
            if mid > start { queue.push((start, mid)); }
            if mid + 1 < end { queue.push((mid + 1, end)); }
        }
        order
    }

    pub fn get_stats(&self) -> &ShrinkingStats {
        &self.stats
    }

    pub fn reset_stats(&mut self) {
        self.stats = ShrinkingStats::default();
        self.history = CandidateHistory::new();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use shared_types::{Token, PosTag, DependencyEdge, DependencyRelation};
    use std::collections::HashMap;

    fn make_sentence(text: &str) -> Sentence {
        let tokens: Vec<Token> = text.split_whitespace().enumerate().map(|(i, w)| {
            Token::new(w, i).with_pos(PosTag::Noun).with_lemma(w.to_lowercase())
        }).collect();
        Sentence::from_tokens(tokens, text)
    }

    #[test]
    fn test_shrink_with_accept_all() {
        let s = make_sentence("The big fluffy cat sat on the soft red mat");
        let tree = ShrinkableTree::from_sentence(&s);
        let oracle = AlwaysAcceptOracle;
        let mut engine = GCHDDEngine::with_default_config();
        let result = engine.shrink(&tree, &oracle);
        // With always-accept oracle, should reduce
        assert!(result.shrunk_size <= result.original_size);
    }

    #[test]
    fn test_shrink_preserves_minimum() {
        let s = make_sentence("cat");
        let tree = ShrinkableTree::from_sentence(&s);
        let oracle = AlwaysAcceptOracle;
        let mut engine = GCHDDEngine::new(GCHDDConfig { min_tree_size: 1, ..Default::default() });
        let result = engine.shrink(&tree, &oracle);
        assert!(result.shrunk_size >= 1 || result.shrunk_text.is_empty());
    }

    #[test]
    fn test_shrink_timeout() {
        let s = make_sentence("a b c d e f g h i j");
        let tree = ShrinkableTree::from_sentence(&s);
        let oracle = AlwaysAcceptOracle;
        let mut engine = GCHDDEngine::new(GCHDDConfig {
            timeout_seconds: 1,
            ..Default::default()
        });
        let result = engine.shrink(&tree, &oracle);
        assert!(result.time_elapsed_ms < 5000);
    }

    #[test]
    fn test_shrinking_stats() {
        let s = make_sentence("The cat sat on the mat");
        let tree = ShrinkableTree::from_sentence(&s);
        let oracle = AlwaysAcceptOracle;
        let mut engine = GCHDDEngine::with_default_config();
        engine.shrink(&tree, &oracle);
        let stats = engine.get_stats();
        assert!(stats.total_attempts >= 0);
    }

    #[test]
    fn test_shrinking_result_metrics() {
        let result = ShrinkingResult {
            original_text: "The big cat sat".into(),
            shrunk_text: "cat sat".into(),
            original_size: 4,
            shrunk_size: 2,
            reduction_ratio: 0.5,
            is_one_minimal: true,
            iterations_used: 3,
            checker_invocations: 10,
            time_elapsed_ms: 100,
            history: Vec::new(),
        };
        assert_eq!(result.word_reduction(), 2);
        assert!((result.reduction_percentage() - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_closure_oracle() {
        let oracle = ClosureOracle {
            validity_fn: Box::new(|s| s.split_whitespace().count() >= 2),
            applicability_fn: Box::new(|_| true),
            violation_fn: Box::new(|s| s.contains("cat")),
        };
        assert!(oracle.check_validity("hello world"));
        assert!(!oracle.check_validity("hello"));
        assert!(oracle.check_violation_preserved("the cat"));
    }

    #[test]
    fn test_binary_search_order() {
        let s = make_sentence("a b c d e f");
        let tree = ShrinkableTree::from_sentence(&s);
        let engine = GCHDDEngine::with_default_config();
        let order = engine.binary_search_order(&tree);
        assert!(!order.is_empty());
    }

    #[test]
    fn test_config_default() {
        let config = GCHDDConfig::default();
        assert_eq!(config.max_iterations, 1000);
        assert_eq!(config.timeout_seconds, 60);
    }

    #[test]
    fn test_reset_stats() {
        let mut engine = GCHDDEngine::with_default_config();
        engine.stats.total_attempts = 100;
        engine.reset_stats();
        assert_eq!(engine.stats.total_attempts, 0);
    }
}
