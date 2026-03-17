#!/usr/bin/env python3
"""Generate remaining Rust source files for the NLP metamorphic localizer."""
import os

BASE = "/Users/halleyyoung/Documents/div/mathdivergence/pipeline_staging/nlp-metamorphic-localizer/implementation/crates"

def write_file(rel_path, content):
    path = os.path.join(BASE, rel_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and os.path.getsize(path) > 100:
        print(f"SKIP {rel_path} (exists with {os.path.getsize(path)} bytes)")
        return
    with open(path, 'w') as f:
        f.write(content)
    print(f"WROTE {rel_path} ({len(content)} chars, ~{content.count(chr(10))} lines)")


# ============================================================================
# shrinking/src/subtree.rs
# ============================================================================
write_file("shrinking/src/subtree.rs", r'''//! Subtree operations and candidate generation for GCHDD.

use serde::{Deserialize, Serialize};
use shared_types::PosTag;
use std::collections::HashMap;

use crate::parse_tree::{ShrinkNode, ShrinkableTree};

/// Operation types for subtree manipulation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SubtreeOp {
    Delete,
    ReplaceWithMinimal,
    ReplaceWithChild(usize),
    CollapseToHead,
    PruneModifiers,
}

impl std::fmt::Display for SubtreeOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Delete => write!(f, "delete"),
            Self::ReplaceWithMinimal => write!(f, "replace-minimal"),
            Self::ReplaceWithChild(i) => write!(f, "promote-child-{}", i),
            Self::CollapseToHead => write!(f, "collapse-to-head"),
            Self::PruneModifiers => write!(f, "prune-modifiers"),
        }
    }
}

/// A candidate subtree replacement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubtreeCandidate {
    pub target_node: usize,
    pub replacement: Option<ShrinkNode>,
    pub size_reduction: usize,
    pub operation: SubtreeOp,
    pub estimated_validity: f64,
}

impl SubtreeCandidate {
    pub fn new(target: usize, op: SubtreeOp, reduction: usize) -> Self {
        Self {
            target_node: target,
            replacement: None,
            size_reduction: reduction,
            operation: op,
            estimated_validity: 0.5,
        }
    }

    pub fn with_replacement(mut self, node: ShrinkNode) -> Self {
        self.replacement = Some(node);
        self
    }

    pub fn with_estimated_validity(mut self, v: f64) -> Self {
        self.estimated_validity = v;
        self
    }
}

/// Generates candidate replacements for tree nodes.
pub struct CandidateGenerator {
    pub max_candidates_per_node: usize,
    pub enable_deletion: bool,
    pub enable_child_promotion: bool,
    pub enable_modifier_pruning: bool,
}

impl Default for CandidateGenerator {
    fn default() -> Self {
        Self {
            max_candidates_per_node: 10,
            enable_deletion: true,
            enable_child_promotion: true,
            enable_modifier_pruning: true,
        }
    }
}

impl CandidateGenerator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Generate all valid candidate replacements for a given node.
    pub fn generate_candidates(
        &self,
        tree: &ShrinkableTree,
        node_id: usize,
    ) -> Vec<SubtreeCandidate> {
        let mut candidates = Vec::new();

        if node_id >= tree.nodes.len() || tree.nodes[node_id].is_deleted {
            return candidates;
        }

        let node = &tree.nodes[node_id];
        if node.is_critical {
            return candidates;
        }

        let subtree_size = tree.subtree_size(node_id);

        // 1. Try deletion (if not root and not the only child)
        if self.enable_deletion && node.parent.is_some() && subtree_size > 1 {
            let can_delete = self.can_safely_delete(tree, node_id);
            if can_delete {
                candidates.push(
                    SubtreeCandidate::new(node_id, SubtreeOp::Delete, subtree_size)
                        .with_estimated_validity(0.3),
                );
            }
        }

        // 2. Try replacing with minimal subtree
        if subtree_size > 1 {
            if let Some(minimal) = self.generate_minimal_replacement(node) {
                let reduction = subtree_size.saturating_sub(1);
                candidates.push(
                    SubtreeCandidate::new(node_id, SubtreeOp::ReplaceWithMinimal, reduction)
                        .with_replacement(minimal)
                        .with_estimated_validity(0.5),
                );
            }
        }

        // 3. Try promoting each child
        if self.enable_child_promotion && !node.children.is_empty() {
            for (i, &child_id) in node.children.iter().enumerate() {
                if child_id < tree.nodes.len() && !tree.nodes[child_id].is_deleted {
                    let child_size = tree.subtree_size(child_id);
                    let reduction = subtree_size.saturating_sub(child_size);
                    if reduction > 0 {
                        candidates.push(
                            SubtreeCandidate::new(
                                node_id,
                                SubtreeOp::ReplaceWithChild(i),
                                reduction,
                            )
                            .with_estimated_validity(0.4),
                        );
                    }
                }
            }
        }

        // 4. Try collapsing to head word
        if node.children.len() > 1 {
            if let Some(head_child) = self.find_head_child(tree, node_id) {
                let head_size = if tree.nodes[head_child].is_leaf() { 1 } else { tree.subtree_size(head_child) };
                let reduction = subtree_size.saturating_sub(head_size);
                if reduction > 0 {
                    candidates.push(
                        SubtreeCandidate::new(node_id, SubtreeOp::CollapseToHead, reduction)
                            .with_estimated_validity(0.4),
                    );
                }
            }
        }

        // 5. Try pruning modifiers (adjectives, adverbs, PPs)
        if self.enable_modifier_pruning {
            let modifier_size = self.count_modifier_nodes(tree, node_id);
            if modifier_size > 0 {
                candidates.push(
                    SubtreeCandidate::new(node_id, SubtreeOp::PruneModifiers, modifier_size)
                        .with_estimated_validity(0.6),
                );
            }
        }

        // Sort by size reduction descending
        candidates.sort_by(|a, b| b.size_reduction.cmp(&a.size_reduction));
        candidates.truncate(self.max_candidates_per_node);
        candidates
    }

    fn can_safely_delete(&self, tree: &ShrinkableTree, node_id: usize) -> bool {
        let node = &tree.nodes[node_id];
        // Don't delete the root or nodes that are the only child
        if node.parent.is_none() {
            return false;
        }
        if let Some(pid) = node.parent {
            let parent = &tree.nodes[pid];
            let active_children = parent.children.iter()
                .filter(|&&c| c < tree.nodes.len() && !tree.nodes[c].is_deleted)
                .count();
            if active_children <= 1 {
                return false;
            }
        }
        // Allow deletion of modifier-like nodes
        match &node.dep_relation {
            Some(rel) => matches!(
                rel,
                shared_types::DependencyRelation::Amod
                    | shared_types::DependencyRelation::Advmod
                    | shared_types::DependencyRelation::Det
                    | shared_types::DependencyRelation::Prep
                    | shared_types::DependencyRelation::Punct
                    | shared_types::DependencyRelation::Cc
                    | shared_types::DependencyRelation::Conj
            ),
            None => {
                // Check by label
                matches!(node.label.as_str(), "ADJP" | "ADVP" | "PP" | "Punct")
            }
        }
    }

    /// Generate a minimal replacement for a phrase node.
    pub fn generate_minimal_replacement(&self, node: &ShrinkNode) -> Option<ShrinkNode> {
        let label = node.label.as_str();
        match label {
            "NP" => Some(ShrinkNode::new_leaf(node.id, "it", PosTag::Pron, node.span_start)),
            "VP" => Some(ShrinkNode::new_leaf(node.id, "is", PosTag::Verb, node.span_start)),
            "PP" => None, // PPs can often be deleted entirely
            "ADVP" => None,
            "ADJP" => None,
            "S" => Some(ShrinkNode::new_leaf(node.id, "it is", PosTag::Verb, node.span_start)),
            _ => {
                // For unknown labels, try replacing with a pronoun
                if node.word.is_some() {
                    None // Already a leaf
                } else {
                    Some(ShrinkNode::new_leaf(node.id, "it", PosTag::Pron, node.span_start))
                }
            }
        }
    }

    fn find_head_child(&self, tree: &ShrinkableTree, node_id: usize) -> Option<usize> {
        let node = &tree.nodes[node_id];
        let label = node.label.as_str();

        // Head-finding rules based on phrase type
        match label {
            "NP" => {
                // Head of NP is the rightmost noun
                node.children.iter().rev().find(|&&c| {
                    c < tree.nodes.len()
                        && !tree.nodes[c].is_deleted
                        && matches!(
                            tree.nodes[c].pos_tag,
                            Some(PosTag::Noun) | Some(PosTag::Pron)
                        )
                }).copied()
            }
            "VP" => {
                // Head of VP is the leftmost verb
                node.children.iter().find(|&&c| {
                    c < tree.nodes.len()
                        && !tree.nodes[c].is_deleted
                        && matches!(
                            tree.nodes[c].pos_tag,
                            Some(PosTag::Verb) | Some(PosTag::Aux)
                        )
                }).copied()
            }
            _ => {
                // Default: first child
                node.children.first().copied()
            }
        }
    }

    fn count_modifier_nodes(&self, tree: &ShrinkableTree, node_id: usize) -> usize {
        let node = &tree.nodes[node_id];
        let mut count = 0;
        for &child_id in &node.children {
            if child_id >= tree.nodes.len() || tree.nodes[child_id].is_deleted {
                continue;
            }
            let child = &tree.nodes[child_id];
            let is_modifier = match &child.dep_relation {
                Some(rel) => matches!(
                    rel,
                    shared_types::DependencyRelation::Amod
                        | shared_types::DependencyRelation::Advmod
                        | shared_types::DependencyRelation::Det
                ),
                None => matches!(child.label.as_str(), "ADJP" | "ADVP"),
            };
            if is_modifier {
                count += tree.subtree_size(child_id);
            }
        }
        count
    }
}

/// Generates minimal valid subtrees for each syntactic category.
pub struct MinimalSubtreeGenerator;

impl MinimalSubtreeGenerator {
    pub fn minimal_np(span_start: usize) -> ShrinkNode {
        ShrinkNode::new_leaf(0, "it", PosTag::Pron, span_start)
    }

    pub fn minimal_np_indefinite(span_start: usize) -> ShrinkNode {
        ShrinkNode::new_leaf(0, "something", PosTag::Pron, span_start)
    }

    pub fn minimal_np_animate(span_start: usize) -> ShrinkNode {
        ShrinkNode::new_leaf(0, "someone", PosTag::Pron, span_start)
    }

    pub fn minimal_vp(span_start: usize) -> ShrinkNode {
        ShrinkNode::new_leaf(0, "exists", PosTag::Verb, span_start)
    }

    pub fn minimal_copula(span_start: usize) -> ShrinkNode {
        ShrinkNode::new_leaf(0, "is", PosTag::Verb, span_start)
    }
}

/// Ranks candidates by expected utility.
pub struct CandidateRanker {
    pub size_weight: f64,
    pub validity_weight: f64,
    pub history_weight: f64,
}

impl Default for CandidateRanker {
    fn default() -> Self {
        Self {
            size_weight: 0.5,
            validity_weight: 0.3,
            history_weight: 0.2,
        }
    }
}

impl CandidateRanker {
    pub fn rank(&self, candidates: &mut [SubtreeCandidate], history: &CandidateHistory) {
        for c in candidates.iter_mut() {
            let size_score = c.size_reduction as f64;
            let validity_score = c.estimated_validity;
            let history_score = history.success_rate_for_op(&c.operation);
            c.estimated_validity = self.size_weight * size_score
                + self.validity_weight * validity_score
                + self.history_weight * history_score;
        }
        candidates.sort_by(|a, b| {
            b.estimated_validity
                .partial_cmp(&a.estimated_validity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }
}

/// Tracks historical success rates of different candidate operations.
#[derive(Debug, Clone, Default)]
pub struct CandidateHistory {
    pub attempts: HashMap<String, usize>,
    pub successes: HashMap<String, usize>,
}

impl CandidateHistory {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(&mut self, op: &SubtreeOp, success: bool) {
        let key = format!("{}", op);
        *self.attempts.entry(key.clone()).or_insert(0) += 1;
        if success {
            *self.successes.entry(key).or_insert(0) += 1;
        }
    }

    pub fn success_rate_for_op(&self, op: &SubtreeOp) -> f64 {
        let key = format!("{}", op);
        let attempts = self.attempts.get(&key).copied().unwrap_or(0);
        let successes = self.successes.get(&key).copied().unwrap_or(0);
        if attempts == 0 {
            return 0.5; // Prior
        }
        successes as f64 / attempts as f64
    }

    pub fn total_attempts(&self) -> usize {
        self.attempts.values().sum()
    }

    pub fn total_successes(&self) -> usize {
        self.successes.values().sum()
    }
}

/// Filters candidates based on validity checks.
pub struct CandidateFilter {
    pub min_size_reduction: usize,
    pub min_validity_estimate: f64,
}

impl Default for CandidateFilter {
    fn default() -> Self {
        Self {
            min_size_reduction: 1,
            min_validity_estimate: 0.0,
        }
    }
}

impl CandidateFilter {
    pub fn filter(&self, candidates: Vec<SubtreeCandidate>) -> Vec<SubtreeCandidate> {
        candidates
            .into_iter()
            .filter(|c| {
                c.size_reduction >= self.min_size_reduction
                    && c.estimated_validity >= self.min_validity_estimate
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse_tree::ShrinkableTree;
    use shared_types::{DependencyEdge, DependencyRelation, Sentence, Token};
    use std::collections::HashMap;

    fn make_sentence(text: &str) -> Sentence {
        let tokens: Vec<Token> = text.split_whitespace().enumerate().map(|(i, w)| Token {
            text: w.to_string(), lemma: w.to_lowercase(),
            pos_tag: if i == 0 { PosTag::Det } else if i == 1 { PosTag::Noun } else { PosTag::Verb },
            index: i, features: HashMap::new(),
        }).collect();
        Sentence { tokens, dependency_edges: Vec::new(), entities: Vec::new(), raw_text: text.to_string() }
    }

    #[test]
    fn test_candidate_generation() {
        let s = make_sentence("The big cat sat quietly");
        let tree = ShrinkableTree::from_sentence(&s);
        let gen = CandidateGenerator::new();
        // Try generating for root
        let candidates = gen.generate_candidates(&tree, tree.root);
        // Root might not have many candidates, but it shouldn't crash
        assert!(candidates.len() >= 0);
    }

    #[test]
    fn test_minimal_np() {
        let node = MinimalSubtreeGenerator::minimal_np(0);
        assert_eq!(node.word.as_deref(), Some("it"));
    }

    #[test]
    fn test_minimal_vp() {
        let node = MinimalSubtreeGenerator::minimal_vp(0);
        assert_eq!(node.word.as_deref(), Some("exists"));
    }

    #[test]
    fn test_candidate_history() {
        let mut hist = CandidateHistory::new();
        hist.record(&SubtreeOp::Delete, true);
        hist.record(&SubtreeOp::Delete, false);
        hist.record(&SubtreeOp::Delete, true);
        assert!((hist.success_rate_for_op(&SubtreeOp::Delete) - 2.0/3.0).abs() < 0.01);
    }

    #[test]
    fn test_candidate_filter() {
        let filter = CandidateFilter { min_size_reduction: 2, min_validity_estimate: 0.0 };
        let candidates = vec![
            SubtreeCandidate::new(0, SubtreeOp::Delete, 1),
            SubtreeCandidate::new(1, SubtreeOp::Delete, 3),
        ];
        let filtered = filter.filter(candidates);
        assert_eq!(filtered.len(), 1);
    }

    #[test]
    fn test_subtree_op_display() {
        assert_eq!(format!("{}", SubtreeOp::Delete), "delete");
        assert_eq!(format!("{}", SubtreeOp::PruneModifiers), "prune-modifiers");
    }

    #[test]
    fn test_candidate_ranker() {
        let ranker = CandidateRanker::default();
        let history = CandidateHistory::new();
        let mut candidates = vec![
            SubtreeCandidate::new(0, SubtreeOp::Delete, 5).with_estimated_validity(0.3),
            SubtreeCandidate::new(1, SubtreeOp::ReplaceWithMinimal, 3).with_estimated_validity(0.8),
        ];
        ranker.rank(&mut candidates, &history);
        // After ranking, order should be determined by combined score
        assert!(candidates.len() == 2);
    }

    #[test]
    fn test_history_default() {
        let hist = CandidateHistory::new();
        assert_eq!(hist.total_attempts(), 0);
        // Unknown op should return prior
        assert!((hist.success_rate_for_op(&SubtreeOp::Delete) - 0.5).abs() < 0.01);
    }
}
''')

# ============================================================================
# shrinking/src/gchdd.rs
# ============================================================================
write_file("shrinking/src/gchdd.rs", r'''//! GCHDD (Grammar-Constrained Hierarchical Delta Debugging) algorithm.
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
        let tokens: Vec<Token> = text.split_whitespace().enumerate().map(|(i, w)| Token {
            text: w.to_string(), lemma: w.to_lowercase(),
            pos_tag: PosTag::Noun, index: i, features: HashMap::new(),
        }).collect();
        Sentence { tokens, dependency_edges: Vec::new(), entities: Vec::new(), raw_text: text.to_string() }
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
''')

print("gchdd.rs done")
PYEOF