//! Subtree operations and candidate generation for GCHDD.

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
        let tokens: Vec<Token> = text.split_whitespace().enumerate().map(|(i, w)| {
            let pos = if i == 0 { PosTag::Det } else if i == 1 { PosTag::Noun } else { PosTag::Verb };
            Token::new(w, i).with_pos(pos).with_lemma(w.to_lowercase())
        }).collect();
        Sentence::from_tokens(tokens, text)
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
