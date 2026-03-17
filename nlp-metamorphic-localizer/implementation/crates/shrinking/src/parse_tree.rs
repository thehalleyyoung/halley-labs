//! Parse tree representation for grammar-constrained shrinking.

use serde::{Deserialize, Serialize};
use shared_types::{DependencyEdge, DependencyRelation, PosTag, Sentence, Token};
use std::collections::{HashMap, HashSet, VecDeque};

/// A node in the shrinkable parse tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShrinkNode {
    pub id: usize,
    pub label: String,
    pub word: Option<String>,
    pub children: Vec<usize>,
    pub parent: Option<usize>,
    pub features: HashMap<String, String>,
    pub span_start: usize,
    pub span_end: usize,
    pub is_critical: bool,
    pub is_deleted: bool,
    pub pos_tag: Option<PosTag>,
    pub dep_relation: Option<DependencyRelation>,
}

impl ShrinkNode {
    pub fn new_internal(id: usize, label: &str, span_start: usize, span_end: usize) -> Self {
        Self {
            id,
            label: label.to_string(),
            word: None,
            children: Vec::new(),
            parent: None,
            features: HashMap::new(),
            span_start,
            span_end,
            is_critical: false,
            is_deleted: false,
            pos_tag: None,
            dep_relation: None,
        }
    }

    pub fn new_leaf(id: usize, word: &str, pos: PosTag, index: usize) -> Self {
        Self {
            id,
            label: format!("{:?}", pos),
            word: Some(word.to_string()),
            children: Vec::new(),
            parent: None,
            features: HashMap::new(),
            span_start: index,
            span_end: index + 1,
            is_critical: false,
            is_deleted: false,
            pos_tag: Some(pos),
            dep_relation: None,
        }
    }

    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    pub fn span_size(&self) -> usize {
        self.span_end.saturating_sub(self.span_start)
    }
}

/// A tree structure that supports shrinking operations for GCHDD.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShrinkableTree {
    pub nodes: Vec<ShrinkNode>,
    pub root: usize,
    pub original_text: String,
}

impl ShrinkableTree {
    /// Build a shrinkable tree from a `Sentence`.
    pub fn from_sentence(sentence: &Sentence) -> Self {
        if sentence.tokens.is_empty() {
            return Self {
                nodes: vec![ShrinkNode::new_internal(0, "S", 0, 0)],
                root: 0,
                original_text: sentence.raw_text.clone(),
            };
        }

        let mut nodes = Vec::new();
        let n_tokens = sentence.tokens.len();

        // Create root S node
        let root_id = 0;
        nodes.push(ShrinkNode::new_internal(root_id, "S", 0, n_tokens));

        // Build dependency-based tree structure
        let mut head_map: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut has_head: HashSet<usize> = HashSet::new();
        let mut dep_relations: HashMap<usize, DependencyRelation> = HashMap::new();

        for edge in &sentence.dependency_edges {
            head_map
                .entry(edge.head_index)
                .or_default()
                .push(edge.dependent_index);
            has_head.insert(edge.dependent_index);
            dep_relations.insert(edge.dependent_index, edge.relation.clone());
        }

        // Find the root token (no head)
        let root_token = (0..n_tokens).find(|i| !has_head.contains(i)).unwrap_or(0);

        // Create leaf nodes for each token
        let leaf_offset = 1;
        for (i, token) in sentence.tokens.iter().enumerate() {
            let pos = token.pos_tag.clone().unwrap_or(PosTag::Other);
            let mut leaf = ShrinkNode::new_leaf(leaf_offset + i, &token.text, pos, i);
            leaf.dep_relation = dep_relations.get(&i).cloned();
            nodes.push(leaf);
        }

        // Build phrase structure from dependency structure
        // Simple heuristic: create intermediate phrase nodes
        let mut phrase_id = leaf_offset + n_tokens;

        // Group tokens by their head into phrase nodes
        fn build_phrase_tree(
            head_idx: usize,
            head_map: &HashMap<usize, Vec<usize>>,
            nodes: &mut Vec<ShrinkNode>,
            leaf_offset: usize,
            phrase_id: &mut usize,
            tokens: &[Token],
        ) -> usize {
            let leaf_node_id = leaf_offset + head_idx;
            let dependents = head_map.get(&head_idx).cloned().unwrap_or_default();

            if dependents.is_empty() {
                return leaf_node_id;
            }

            // Create a phrase node
            let pid = *phrase_id;
            *phrase_id += 1;

            let mut all_indices: Vec<usize> = vec![head_idx];
            all_indices.extend(dependents.iter());
            all_indices.sort();

            let span_start = all_indices.iter().copied().min().unwrap_or(0);
            let span_end = all_indices.iter().copied().max().unwrap_or(0) + 1;

            let phrase_label = match tokens[head_idx].pos_tag {
                Some(PosTag::Noun) | Some(PosTag::Pron) => "NP",
                Some(PosTag::Verb) | Some(PosTag::Aux) => "VP",
                Some(PosTag::Adj) => "ADJP",
                Some(PosTag::Adv) => "ADVP",
                Some(PosTag::Prep) => "PP",
                _ => "XP",
            };

            let mut phrase_node = ShrinkNode::new_internal(pid, phrase_label, span_start, span_end);

            // Recursively build children
            let mut child_ids = Vec::new();
            for &dep_idx in &dependents {
                if dep_idx < head_idx {
                    let child_id = build_phrase_tree(dep_idx, head_map, nodes, leaf_offset, phrase_id, tokens);
                    child_ids.push(child_id);
                }
            }
            child_ids.push(leaf_node_id);
            for &dep_idx in &dependents {
                if dep_idx > head_idx {
                    let child_id = build_phrase_tree(dep_idx, head_map, nodes, leaf_offset, phrase_id, tokens);
                    child_ids.push(child_id);
                }
            }

            phrase_node.children = child_ids.clone();
            nodes.push(phrase_node);

            // Set parent pointers
            for &cid in &child_ids {
                if cid < nodes.len() {
                    nodes[cid].parent = Some(pid);
                }
            }

            pid
        }

        let top_phrase = build_phrase_tree(
            root_token,
            &head_map,
            &mut nodes,
            leaf_offset,
            &mut phrase_id,
            &sentence.tokens,
        );

        // Connect root to top phrase
        nodes[root_id].children = vec![top_phrase];
        if top_phrase < nodes.len() {
            nodes[top_phrase].parent = Some(root_id);
        }

        Self {
            nodes,
            root: root_id,
            original_text: sentence.raw_text.clone(),
        }
    }

    /// Build from a generic ParseTree.
    pub fn from_parse_tree(tree: &shared_types::ParseTree) -> Self {
        let mut nodes = Vec::new();
        for (i, pn) in tree.nodes.iter().enumerate() {
            let mut sn = ShrinkNode::new_internal(i, &pn.label, pn.span_start, pn.span_end);
            sn.word = pn.word.clone();
            sn.children = pn.children.clone();
            sn.parent = pn.parent;
            sn.features = pn.features.clone();
            nodes.push(sn);
        }
        let text = nodes
            .iter()
            .filter(|n| n.word.is_some())
            .map(|n| n.word.as_deref().unwrap_or(""))
            .collect::<Vec<_>>()
            .join(" ");
        Self {
            nodes,
            root: tree.root_index,
            original_text: text,
        }
    }

    pub fn node_count(&self) -> usize {
        self.nodes.iter().filter(|n| !n.is_deleted).count()
    }

    pub fn leaf_count(&self) -> usize {
        self.nodes
            .iter()
            .filter(|n| !n.is_deleted && n.is_leaf())
            .count()
    }

    pub fn depth(&self) -> usize {
        self.depth_of(self.root)
    }

    fn depth_of(&self, node_id: usize) -> usize {
        if node_id >= self.nodes.len() || self.nodes[node_id].is_deleted {
            return 0;
        }
        let node = &self.nodes[node_id];
        if node.children.is_empty() {
            return 1;
        }
        1 + node
            .children
            .iter()
            .map(|&c| self.depth_of(c))
            .max()
            .unwrap_or(0)
    }

    /// Convert tree back to sentence text.
    pub fn to_sentence(&self) -> String {
        let mut words = Vec::new();
        self.collect_words(self.root, &mut words);
        words.join(" ")
    }

    fn collect_words(&self, node_id: usize, words: &mut Vec<String>) {
        if node_id >= self.nodes.len() {
            return;
        }
        let node = &self.nodes[node_id];
        if node.is_deleted {
            return;
        }
        if let Some(ref w) = node.word {
            words.push(w.clone());
        } else {
            for &child in &node.children {
                self.collect_words(child, words);
            }
        }
    }

    /// Get all node IDs in a subtree.
    pub fn subtree_at(&self, node_id: usize) -> Vec<usize> {
        let mut result = Vec::new();
        let mut stack = vec![node_id];
        while let Some(nid) = stack.pop() {
            if nid >= self.nodes.len() || self.nodes[nid].is_deleted {
                continue;
            }
            result.push(nid);
            for &child in self.nodes[nid].children.iter().rev() {
                stack.push(child);
            }
        }
        result
    }

    pub fn subtree_text(&self, node_id: usize) -> String {
        let mut words = Vec::new();
        self.collect_words(node_id, &mut words);
        words.join(" ")
    }

    pub fn subtree_size(&self, node_id: usize) -> usize {
        self.subtree_at(node_id).len()
    }

    /// Replace a subtree with a single replacement node.
    pub fn replace_subtree(&self, node_id: usize, replacement: ShrinkNode) -> ShrinkableTree {
        let mut new_tree = self.clone();
        let subtree_ids: HashSet<usize> = self.subtree_at(node_id).into_iter().collect();

        // Mark all subtree nodes as deleted except the target
        for &sid in &subtree_ids {
            if sid != node_id {
                new_tree.nodes[sid].is_deleted = true;
            }
        }

        // Replace the target node
        if node_id < new_tree.nodes.len() {
            let parent = new_tree.nodes[node_id].parent;
            new_tree.nodes[node_id] = replacement;
            new_tree.nodes[node_id].id = node_id;
            new_tree.nodes[node_id].parent = parent;
            new_tree.nodes[node_id].children.clear();
        }

        new_tree
    }

    /// Delete a subtree entirely.
    pub fn delete_subtree(&self, node_id: usize) -> ShrinkableTree {
        let mut new_tree = self.clone();
        let subtree_ids: HashSet<usize> = self.subtree_at(node_id).into_iter().collect();

        for &sid in &subtree_ids {
            new_tree.nodes[sid].is_deleted = true;
        }

        // Remove from parent's children
        if let Some(parent_id) = self.nodes.get(node_id).and_then(|n| n.parent) {
            if parent_id < new_tree.nodes.len() {
                new_tree.nodes[parent_id]
                    .children
                    .retain(|&c| c != node_id);
            }
        }

        new_tree
    }

    pub fn mark_critical(&mut self, node_id: usize) {
        if node_id < self.nodes.len() {
            self.nodes[node_id].is_critical = true;
        }
    }

    pub fn is_leaf(&self, node_id: usize) -> bool {
        node_id < self.nodes.len() && self.nodes[node_id].is_leaf()
    }

    /// Chain of ancestors from node to root.
    pub fn parent_chain(&self, node_id: usize) -> Vec<usize> {
        let mut chain = Vec::new();
        let mut current = node_id;
        let mut visited = HashSet::new();
        while let Some(parent) = self.nodes.get(current).and_then(|n| n.parent) {
            if !visited.insert(parent) {
                break;
            }
            chain.push(parent);
            current = parent;
        }
        chain
    }

    pub fn siblings(&self, node_id: usize) -> Vec<usize> {
        if let Some(parent_id) = self.nodes.get(node_id).and_then(|n| n.parent) {
            if parent_id < self.nodes.len() {
                return self.nodes[parent_id]
                    .children
                    .iter()
                    .copied()
                    .filter(|&c| c != node_id)
                    .collect();
            }
        }
        Vec::new()
    }

    /// Nodes in top-down (BFS) order.
    pub fn top_down_order(&self) -> Vec<usize> {
        let mut order = Vec::new();
        let mut queue = VecDeque::new();
        queue.push_back(self.root);
        while let Some(nid) = queue.pop_front() {
            if nid >= self.nodes.len() || self.nodes[nid].is_deleted {
                continue;
            }
            order.push(nid);
            for &child in &self.nodes[nid].children {
                queue.push_back(child);
            }
        }
        order
    }

    /// Nodes in bottom-up (post-order) traversal.
    pub fn bottom_up_order(&self) -> Vec<usize> {
        let mut order = Vec::new();
        self.post_order(self.root, &mut order);
        order
    }

    fn post_order(&self, node_id: usize, order: &mut Vec<usize>) {
        if node_id >= self.nodes.len() || self.nodes[node_id].is_deleted {
            return;
        }
        for &child in &self.nodes[node_id].children {
            self.post_order(child, order);
        }
        order.push(node_id);
    }

    pub fn clone_tree(&self) -> Self {
        self.clone()
    }

    /// Word count (leaf nodes with words).
    pub fn word_count(&self) -> usize {
        self.nodes
            .iter()
            .filter(|n| !n.is_deleted && n.word.is_some())
            .count()
    }

    /// Validate tree structure: all parent/child references are consistent.
    pub fn is_valid(&self) -> bool {
        for node in &self.nodes {
            if node.is_deleted {
                continue;
            }
            for &child in &node.children {
                if child >= self.nodes.len() {
                    return false;
                }
                if self.nodes[child].is_deleted {
                    continue;
                }
                if self.nodes[child].parent != Some(node.id) {
                    return false;
                }
            }
        }
        true
    }
}

/// Compute differences between two shrinkable trees.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeDiff {
    pub added_nodes: Vec<usize>,
    pub removed_nodes: Vec<usize>,
    pub modified_nodes: Vec<usize>,
    pub size_change: isize,
    pub word_count_change: isize,
}

impl TreeDiff {
    pub fn compute(before: &ShrinkableTree, after: &ShrinkableTree) -> Self {
        let before_words: HashSet<String> = before
            .nodes
            .iter()
            .filter(|n| !n.is_deleted && n.word.is_some())
            .map(|n| format!("{}:{}", n.span_start, n.word.as_deref().unwrap_or("")))
            .collect();

        let after_words: HashSet<String> = after
            .nodes
            .iter()
            .filter(|n| !n.is_deleted && n.word.is_some())
            .map(|n| format!("{}:{}", n.span_start, n.word.as_deref().unwrap_or("")))
            .collect();

        let added: Vec<usize> = after
            .nodes
            .iter()
            .filter(|n| {
                !n.is_deleted
                    && n.word.is_some()
                    && !before_words.contains(&format!(
                        "{}:{}",
                        n.span_start,
                        n.word.as_deref().unwrap_or("")
                    ))
            })
            .map(|n| n.id)
            .collect();

        let removed: Vec<usize> = before
            .nodes
            .iter()
            .filter(|n| {
                !n.is_deleted
                    && n.word.is_some()
                    && !after_words.contains(&format!(
                        "{}:{}",
                        n.span_start,
                        n.word.as_deref().unwrap_or("")
                    ))
            })
            .map(|n| n.id)
            .collect();

        let before_size = before.node_count() as isize;
        let after_size = after.node_count() as isize;
        let before_wc = before.word_count() as isize;
        let after_wc = after.word_count() as isize;

        Self {
            added_nodes: added,
            removed_nodes: removed,
            modified_nodes: Vec::new(),
            size_change: after_size - before_size,
            word_count_change: after_wc - before_wc,
        }
    }

    pub fn is_reduction(&self) -> bool {
        self.size_change < 0
    }

    pub fn reduction_amount(&self) -> usize {
        if self.size_change < 0 {
            (-self.size_change) as usize
        } else {
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sentence(text: &str) -> Sentence {
        let tokens: Vec<Token> = text
            .split_whitespace()
            .enumerate()
            .map(|(i, w)| Token {
                text: w.to_string(),
                lemma: w.to_lowercase(),
                pos_tag: if i == 0 { PosTag::Det } else if i == 1 { PosTag::Noun } else { PosTag::Verb },
                index: i,
                features: HashMap::new(),
            })
            .collect();
        let edges = if tokens.len() >= 3 {
            vec![
                DependencyEdge { head_index: 2, dependent_index: 0, relation: DependencyRelation::Det },
                DependencyEdge { head_index: 2, dependent_index: 1, relation: DependencyRelation::Nsubj },
            ]
        } else {
            Vec::new()
        };
        Sentence {
            tokens,
            dependency_edges: edges,
            entities: Vec::new(),
            raw_text: text.to_string(),
        }
    }

    #[test]
    fn test_from_sentence() {
        let s = make_sentence("The cat sat");
        let tree = ShrinkableTree::from_sentence(&s);
        assert!(tree.node_count() > 0);
        assert!(tree.depth() > 0);
    }

    #[test]
    fn test_to_sentence() {
        let s = make_sentence("The cat sat");
        let tree = ShrinkableTree::from_sentence(&s);
        let text = tree.to_sentence();
        assert!(text.contains("The"));
        assert!(text.contains("cat"));
    }

    #[test]
    fn test_subtree_at() {
        let s = make_sentence("The cat sat");
        let tree = ShrinkableTree::from_sentence(&s);
        let subtree = tree.subtree_at(tree.root);
        assert!(!subtree.is_empty());
    }

    #[test]
    fn test_delete_subtree() {
        let s = make_sentence("The cat sat on the mat");
        let tree = ShrinkableTree::from_sentence(&s);
        let orig_count = tree.node_count();
        if tree.nodes.len() > 2 {
            let new_tree = tree.delete_subtree(1);
            assert!(new_tree.node_count() <= orig_count);
        }
    }

    #[test]
    fn test_mark_critical() {
        let s = make_sentence("The cat sat");
        let mut tree = ShrinkableTree::from_sentence(&s);
        tree.mark_critical(0);
        assert!(tree.nodes[0].is_critical);
    }

    #[test]
    fn test_top_down_order() {
        let s = make_sentence("The cat sat");
        let tree = ShrinkableTree::from_sentence(&s);
        let order = tree.top_down_order();
        assert_eq!(order[0], tree.root);
    }

    #[test]
    fn test_bottom_up_order() {
        let s = make_sentence("The cat sat");
        let tree = ShrinkableTree::from_sentence(&s);
        let order = tree.bottom_up_order();
        assert!(!order.is_empty());
        assert_eq!(*order.last().unwrap(), tree.root);
    }

    #[test]
    fn test_tree_diff() {
        let s1 = make_sentence("The cat sat on mat");
        let t1 = ShrinkableTree::from_sentence(&s1);
        let s2 = make_sentence("cat sat");
        let t2 = ShrinkableTree::from_sentence(&s2);
        let diff = TreeDiff::compute(&t1, &t2);
        assert!(diff.word_count_change < 0 || diff.word_count_change == 0);
    }

    #[test]
    fn test_parent_chain() {
        let s = make_sentence("The cat sat");
        let tree = ShrinkableTree::from_sentence(&s);
        if tree.nodes.len() > 2 {
            let chain = tree.parent_chain(1);
            assert!(!chain.is_empty() || tree.nodes[1].parent.is_none());
        }
    }

    #[test]
    fn test_word_count() {
        let s = make_sentence("The cat sat");
        let tree = ShrinkableTree::from_sentence(&s);
        assert_eq!(tree.word_count(), 3);
    }

    #[test]
    fn test_clone_tree() {
        let s = make_sentence("hello world there");
        let tree = ShrinkableTree::from_sentence(&s);
        let cloned = tree.clone_tree();
        assert_eq!(tree.node_count(), cloned.node_count());
    }
}
