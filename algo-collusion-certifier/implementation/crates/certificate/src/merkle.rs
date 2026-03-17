//! Merkle tree for evidence integrity.
//!
//! SHA-256 based Merkle tree implementation for binding certificate evidence,
//! with support for incremental construction and forest (multi-tree) management.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt;

// ── Hash type ────────────────────────────────────────────────────────────────

/// A SHA-256 hash represented as a hex string.
pub type Hash = String;

/// Compute the SHA-256 hash of arbitrary data.
pub fn compute_data_hash(data: &[u8]) -> Hash {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hex::encode(hasher.finalize())
}

/// Compute the SHA-256 hash of a string.
pub fn compute_string_hash(s: &str) -> Hash {
    compute_data_hash(s.as_bytes())
}

/// Combine two hashes into a parent hash (for internal Merkle nodes).
fn combine_hashes(left: &str, right: &str) -> Hash {
    let mut hasher = Sha256::new();
    hasher.update(left.as_bytes());
    hasher.update(right.as_bytes());
    hex::encode(hasher.finalize())
}

// ── Merkle node ──────────────────────────────────────────────────────────────

/// A node in a Merkle tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CertMerkleNode {
    Leaf {
        hash: Hash,
        data_ref: String,
        data_type: String,
    },
    Internal {
        hash: Hash,
        left: Box<CertMerkleNode>,
        right: Box<CertMerkleNode>,
    },
}

impl CertMerkleNode {
    pub fn hash(&self) -> &str {
        match self {
            CertMerkleNode::Leaf { hash, .. } => hash,
            CertMerkleNode::Internal { hash, .. } => hash,
        }
    }

    pub fn is_leaf(&self) -> bool {
        matches!(self, CertMerkleNode::Leaf { .. })
    }

    pub fn leaf_count(&self) -> usize {
        match self {
            CertMerkleNode::Leaf { .. } => 1,
            CertMerkleNode::Internal { left, right, .. } => {
                left.leaf_count() + right.leaf_count()
            }
        }
    }

    pub fn depth(&self) -> usize {
        match self {
            CertMerkleNode::Leaf { .. } => 0,
            CertMerkleNode::Internal { left, right, .. } => {
                1 + left.depth().max(right.depth())
            }
        }
    }

    /// Collect all leaf hashes in left-to-right order.
    pub fn leaf_hashes(&self) -> Vec<Hash> {
        match self {
            CertMerkleNode::Leaf { hash, .. } => vec![hash.clone()],
            CertMerkleNode::Internal { left, right, .. } => {
                let mut hashes = left.leaf_hashes();
                hashes.extend(right.leaf_hashes());
                hashes
            }
        }
    }
}

impl fmt::Display for CertMerkleNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CertMerkleNode::Leaf { hash, data_ref, .. } => {
                write!(f, "Leaf({}, {})", &hash[..8], data_ref)
            }
            CertMerkleNode::Internal { hash, .. } => {
                write!(f, "Internal({})", &hash[..8])
            }
        }
    }
}

// ── Evidence item for Merkle tree ────────────────────────────────────────────

/// An item to be included in the Merkle tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MerkleEvidenceItem {
    pub data_ref: String,
    pub data_type: String,
    pub data_hash: Hash,
}

impl MerkleEvidenceItem {
    pub fn new(data_ref: &str, data_type: &str, data: &[u8]) -> Self {
        Self {
            data_ref: data_ref.to_string(),
            data_type: data_type.to_string(),
            data_hash: compute_data_hash(data),
        }
    }

    pub fn from_hash(data_ref: &str, data_type: &str, hash: &str) -> Self {
        Self {
            data_ref: data_ref.to_string(),
            data_type: data_type.to_string(),
            data_hash: hash.to_string(),
        }
    }
}

// ── Merkle tree ──────────────────────────────────────────────────────────────

/// A SHA-256 Merkle tree built from evidence items.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertMerkleTree {
    pub root: Option<CertMerkleNode>,
    pub leaf_count: usize,
}

impl CertMerkleTree {
    pub fn new() -> Self {
        Self {
            root: None,
            leaf_count: 0,
        }
    }

    /// Build a Merkle tree from a list of evidence items.
    pub fn build(items: &[MerkleEvidenceItem]) -> Self {
        if items.is_empty() {
            return Self::new();
        }

        // Create leaf nodes
        let mut nodes: Vec<CertMerkleNode> = items
            .iter()
            .map(|item| CertMerkleNode::Leaf {
                hash: item.data_hash.clone(),
                data_ref: item.data_ref.clone(),
                data_type: item.data_type.clone(),
            })
            .collect();

        // If odd number of leaves, duplicate the last
        if nodes.len() % 2 == 1 && nodes.len() > 1 {
            nodes.push(nodes.last().unwrap().clone());
        }

        let leaf_count = items.len();

        // Build tree bottom-up
        while nodes.len() > 1 {
            let mut next_level = Vec::new();
            for chunk in nodes.chunks(2) {
                if chunk.len() == 2 {
                    let combined = combine_hashes(chunk[0].hash(), chunk[1].hash());
                    next_level.push(CertMerkleNode::Internal {
                        hash: combined,
                        left: Box::new(chunk[0].clone()),
                        right: Box::new(chunk[1].clone()),
                    });
                } else {
                    next_level.push(chunk[0].clone());
                }
            }
            nodes = next_level;
        }

        Self {
            root: Some(nodes.into_iter().next().unwrap()),
            leaf_count,
        }
    }

    pub fn root_hash(&self) -> Option<&str> {
        self.root.as_ref().map(|n| n.hash())
    }

    pub fn is_empty(&self) -> bool {
        self.root.is_none()
    }

    pub fn depth(&self) -> usize {
        self.root.as_ref().map(|n| n.depth()).unwrap_or(0)
    }
}

impl Default for CertMerkleTree {
    fn default() -> Self {
        Self::new()
    }
}

// ── Merkle proof ─────────────────────────────────────────────────────────────

/// An inclusion proof for a single item in the Merkle tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertMerkleProof {
    pub leaf_hash: Hash,
    pub siblings: Vec<ProofSibling>,
    pub root_hash: Hash,
}

/// A sibling node in the proof path.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofSibling {
    pub hash: Hash,
    pub is_left: bool,
}

impl CertMerkleProof {
    /// Verify this proof against a given root hash.
    pub fn verify(&self, root: &str) -> bool {
        if self.root_hash != root {
            return false;
        }
        let mut current = self.leaf_hash.clone();
        for sibling in &self.siblings {
            current = if sibling.is_left {
                combine_hashes(&sibling.hash, &current)
            } else {
                combine_hashes(&current, &sibling.hash)
            };
        }
        current == self.root_hash
    }
}

/// Generate a Merkle proof for the item at the given index.
pub fn generate_merkle_proof(
    tree: &CertMerkleTree,
    leaf_index: usize,
) -> Option<CertMerkleProof> {
    let root = tree.root.as_ref()?;
    let root_hash = root.hash().to_string();

    let leaf_hashes = root.leaf_hashes();
    if leaf_index >= leaf_hashes.len() {
        return None;
    }

    let leaf_hash = leaf_hashes[leaf_index].clone();

    // Walk the tree to collect siblings
    let mut siblings = Vec::new();
    collect_proof_path(root, leaf_index, &mut siblings);

    Some(CertMerkleProof {
        leaf_hash,
        siblings,
        root_hash,
    })
}

fn collect_proof_path(
    node: &CertMerkleNode,
    target_index: usize,
    siblings: &mut Vec<ProofSibling>,
) {
    match node {
        CertMerkleNode::Leaf { .. } => {}
        CertMerkleNode::Internal { left, right, .. } => {
            let left_count = left.leaf_count();
            if target_index < left_count {
                // Target is in left subtree, sibling is right
                siblings.push(ProofSibling {
                    hash: right.hash().to_string(),
                    is_left: false,
                });
                collect_proof_path(left, target_index, siblings);
            } else {
                // Target is in right subtree, sibling is left
                siblings.push(ProofSibling {
                    hash: left.hash().to_string(),
                    is_left: true,
                });
                collect_proof_path(right, target_index - left_count, siblings);
            }
        }
    }
}

// ── Incremental Merkle tree ──────────────────────────────────────────────────

/// A Merkle tree that supports adding items without rebuilding from scratch.
#[derive(Debug, Clone)]
pub struct IncrementalMerkleTree {
    items: Vec<MerkleEvidenceItem>,
    tree: CertMerkleTree,
    dirty: bool,
}

impl IncrementalMerkleTree {
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
            tree: CertMerkleTree::new(),
            dirty: false,
        }
    }

    pub fn add_item(&mut self, item: MerkleEvidenceItem) {
        self.items.push(item);
        self.dirty = true;
    }

    pub fn add_data(&mut self, data_ref: &str, data_type: &str, data: &[u8]) {
        self.add_item(MerkleEvidenceItem::new(data_ref, data_type, data));
    }

    /// Rebuild the tree if it's been modified.
    pub fn finalize(&mut self) -> &CertMerkleTree {
        if self.dirty {
            self.tree = CertMerkleTree::build(&self.items);
            self.dirty = false;
        }
        &self.tree
    }

    pub fn root_hash(&mut self) -> Option<String> {
        self.finalize();
        self.tree.root_hash().map(|s| s.to_string())
    }

    pub fn item_count(&self) -> usize {
        self.items.len()
    }
}

impl Default for IncrementalMerkleTree {
    fn default() -> Self {
        Self::new()
    }
}

// ── Merkle forest ────────────────────────────────────────────────────────────

/// Collection of Merkle trees for different evidence types.
#[derive(Debug, Clone)]
pub struct MerkleForest {
    trees: std::collections::HashMap<String, CertMerkleTree>,
}

impl MerkleForest {
    pub fn new() -> Self {
        Self {
            trees: std::collections::HashMap::new(),
        }
    }

    pub fn add_tree(&mut self, category: &str, tree: CertMerkleTree) {
        self.trees.insert(category.to_string(), tree);
    }

    pub fn build_tree(&mut self, category: &str, items: &[MerkleEvidenceItem]) {
        let tree = CertMerkleTree::build(items);
        self.trees.insert(category.to_string(), tree);
    }

    pub fn get_tree(&self, category: &str) -> Option<&CertMerkleTree> {
        self.trees.get(category)
    }

    pub fn categories(&self) -> Vec<String> {
        self.trees.keys().cloned().collect()
    }

    /// Compute a super-root by hashing all tree roots together.
    pub fn forest_root(&self) -> Hash {
        let mut sorted_categories: Vec<_> = self.trees.keys().collect();
        sorted_categories.sort();

        let mut hasher = Sha256::new();
        for cat in sorted_categories {
            if let Some(tree) = self.trees.get(cat.as_str()) {
                if let Some(root) = tree.root_hash() {
                    hasher.update(cat.as_bytes());
                    hasher.update(root.as_bytes());
                }
            }
        }
        hex::encode(hasher.finalize())
    }

    pub fn tree_count(&self) -> usize {
        self.trees.len()
    }
}

impl Default for MerkleForest {
    fn default() -> Self {
        Self::new()
    }
}

// ── Evidence integrity ───────────────────────────────────────────────────────

/// Verify the integrity of an entire evidence bundle.
pub struct EvidenceIntegrity;

impl EvidenceIntegrity {
    /// Verify that all items in the tree match their expected hashes.
    pub fn verify_tree(tree: &CertMerkleTree) -> bool {
        match &tree.root {
            None => true,
            Some(root) => Self::verify_node(root),
        }
    }

    fn verify_node(node: &CertMerkleNode) -> bool {
        match node {
            CertMerkleNode::Leaf { .. } => true,
            CertMerkleNode::Internal { hash, left, right } => {
                let expected = combine_hashes(left.hash(), right.hash());
                if *hash != expected {
                    return false;
                }
                Self::verify_node(left) && Self::verify_node(right)
            }
        }
    }

    /// Verify a single Merkle proof.
    pub fn verify_proof(proof: &CertMerkleProof, expected_root: &str) -> bool {
        proof.verify(expected_root)
    }

    /// Verify that a forest's super-root matches expectations.
    pub fn verify_forest(forest: &MerkleForest, expected_root: &str) -> bool {
        forest.forest_root() == expected_root
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_items(n: usize) -> Vec<MerkleEvidenceItem> {
        (0..n)
            .map(|i| {
                MerkleEvidenceItem::new(
                    &format!("item_{}", i),
                    "test_data",
                    format!("data_{}", i).as_bytes(),
                )
            })
            .collect()
    }

    #[test]
    fn test_compute_data_hash() {
        let h1 = compute_data_hash(b"hello");
        let h2 = compute_data_hash(b"hello");
        let h3 = compute_data_hash(b"world");
        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
        assert_eq!(h1.len(), 64); // SHA-256 = 32 bytes = 64 hex chars
    }

    #[test]
    fn test_compute_string_hash() {
        let h = compute_string_hash("test data");
        assert_eq!(h.len(), 64);
    }

    #[test]
    fn test_merkle_tree_empty() {
        let tree = CertMerkleTree::build(&[]);
        assert!(tree.is_empty());
        assert_eq!(tree.leaf_count, 0);
    }

    #[test]
    fn test_merkle_tree_single_item() {
        let items = make_items(1);
        let tree = CertMerkleTree::build(&items);
        assert!(!tree.is_empty());
        assert_eq!(tree.leaf_count, 1);
        assert!(tree.root_hash().is_some());
    }

    #[test]
    fn test_merkle_tree_two_items() {
        let items = make_items(2);
        let tree = CertMerkleTree::build(&items);
        assert_eq!(tree.leaf_count, 2);
        assert_eq!(tree.depth(), 1);
    }

    #[test]
    fn test_merkle_tree_four_items() {
        let items = make_items(4);
        let tree = CertMerkleTree::build(&items);
        assert_eq!(tree.leaf_count, 4);
        assert_eq!(tree.depth(), 2);
    }

    #[test]
    fn test_merkle_tree_odd_items() {
        let items = make_items(3);
        let tree = CertMerkleTree::build(&items);
        assert_eq!(tree.leaf_count, 3);
    }

    #[test]
    fn test_merkle_tree_integrity() {
        let items = make_items(4);
        let tree = CertMerkleTree::build(&items);
        assert!(EvidenceIntegrity::verify_tree(&tree));
    }

    #[test]
    fn test_merkle_proof_generation_and_verification() {
        let items = make_items(4);
        let tree = CertMerkleTree::build(&items);
        let root = tree.root_hash().unwrap();

        for i in 0..4 {
            let proof = generate_merkle_proof(&tree, i).unwrap();
            assert!(proof.verify(root));
        }
    }

    #[test]
    fn test_merkle_proof_fails_wrong_root() {
        let items = make_items(4);
        let tree = CertMerkleTree::build(&items);
        let proof = generate_merkle_proof(&tree, 0).unwrap();
        assert!(!proof.verify("wrong_root_hash"));
    }

    #[test]
    fn test_merkle_proof_out_of_bounds() {
        let items = make_items(2);
        let tree = CertMerkleTree::build(&items);
        assert!(generate_merkle_proof(&tree, 5).is_none());
    }

    #[test]
    fn test_incremental_merkle_tree() {
        let mut inc = IncrementalMerkleTree::new();
        inc.add_data("item_0", "test", b"data_0");
        inc.add_data("item_1", "test", b"data_1");
        let root1 = inc.root_hash().unwrap();
        assert!(!root1.is_empty());

        inc.add_data("item_2", "test", b"data_2");
        let root2 = inc.root_hash().unwrap();
        assert_ne!(root1, root2);
    }

    #[test]
    fn test_merkle_forest() {
        let mut forest = MerkleForest::new();
        forest.build_tree("tests", &make_items(3));
        forest.build_tree("deviations", &make_items(2));

        assert_eq!(forest.tree_count(), 2);
        let root = forest.forest_root();
        assert!(!root.is_empty());
        assert!(EvidenceIntegrity::verify_forest(&forest, &root));
    }

    #[test]
    fn test_merkle_forest_deterministic() {
        let mut f1 = MerkleForest::new();
        f1.build_tree("a", &make_items(2));
        f1.build_tree("b", &make_items(3));

        let mut f2 = MerkleForest::new();
        f2.build_tree("b", &make_items(3));
        f2.build_tree("a", &make_items(2));

        assert_eq!(f1.forest_root(), f2.forest_root());
    }

    #[test]
    fn test_leaf_hashes() {
        let items = make_items(3);
        let tree = CertMerkleTree::build(&items);
        let hashes = tree.root.as_ref().unwrap().leaf_hashes();
        // At least 3 leaf hashes (may be 4 due to padding)
        assert!(hashes.len() >= 3);
    }

    #[test]
    fn test_merkle_evidence_item() {
        let item = MerkleEvidenceItem::new("ref_0", "trajectory", b"raw_data");
        assert_eq!(item.data_ref, "ref_0");
        assert_eq!(item.data_type, "trajectory");
        assert!(!item.data_hash.is_empty());
    }

    #[test]
    fn test_node_display() {
        let node = CertMerkleNode::Leaf {
            hash: "abcdef0123456789".to_string(),
            data_ref: "test_ref".to_string(),
            data_type: "test".to_string(),
        };
        let s = format!("{}", node);
        assert!(s.contains("Leaf"));
    }

    #[test]
    fn test_merkle_proof_single_item() {
        let items = make_items(1);
        let tree = CertMerkleTree::build(&items);
        let root = tree.root_hash().unwrap();
        let proof = generate_merkle_proof(&tree, 0).unwrap();
        assert!(proof.verify(root));
    }
}
