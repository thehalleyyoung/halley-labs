//! Evidence types and Merkle tree integrity for the CollusionProof system.
//!
//! Evidence items describe different categories of collusion evidence.
//! An [`EvidenceBundle`] collects items and protects their integrity with
//! a [`MerkleTree`] built from SHA-256 hashes.

use crate::identifiers::BundleId;
use crate::statistics::HypothesisTestResult;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Evidence items
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// A single piece of evidence for or against collusion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvidenceItem {
    /// Raw price data observation.
    PriceData {
        description: String,
        mean_price: f64,
        competitive_price: f64,
        monopoly_price: f64,
        collusion_index: f64,
        sample_size: usize,
    },
    /// Result of a unilateral deviation experiment.
    DeviationResult {
        description: String,
        deviating_player: usize,
        deviation_round: usize,
        pre_deviation_profit: f64,
        deviation_profit: f64,
        post_deviation_profit: f64,
        punishment_detected: bool,
        punishment_duration: Option<usize>,
    },
    /// Detection of punishment behaviour after deviation.
    PunishmentDetection {
        description: String,
        punishment_type: String,
        severity: f64,
        duration: usize,
        recovery_rounds: Option<usize>,
    },
    /// A statistical test result used as evidence.
    StatisticalTest {
        test_result: HypothesisTestResult,
    },
    /// Equilibrium computation result.
    EquilibriumComputation {
        description: String,
        nash_price: f64,
        collusive_price: f64,
        observed_price: f64,
        deviation_incentive: f64,
        sustainable: bool,
    },
}

impl EvidenceItem {
    pub fn new(description: impl Into<String>) -> Self {
        EvidenceItem::PriceData {
            description: description.into(),
            mean_price: 0.0,
            competitive_price: 0.0,
            monopoly_price: 0.0,
            collusion_index: 0.0,
            sample_size: 0,
        }
    }

    /// Compute a deterministic hash of this evidence item.
    pub fn content_hash(&self) -> String {
        let json = serde_json::to_string(self).unwrap_or_default();
        let mut hasher = Sha256::new();
        hasher.update(json.as_bytes());
        hex::encode(hasher.finalize())
    }

    pub fn description(&self) -> &str {
        match self {
            EvidenceItem::PriceData { description, .. } => description,
            EvidenceItem::DeviationResult { description, .. } => description,
            EvidenceItem::PunishmentDetection { description, .. } => description,
            EvidenceItem::StatisticalTest { test_result } => &test_result.test_name,
            EvidenceItem::EquilibriumComputation { description, .. } => description,
        }
    }

    pub fn strength(&self) -> EvidenceStrength {
        match self {
            EvidenceItem::PriceData { collusion_index, .. } => {
                if *collusion_index > 0.8 { EvidenceStrength::Strong }
                else if *collusion_index > 0.5 { EvidenceStrength::Moderate }
                else if *collusion_index > 0.2 { EvidenceStrength::Weak }
                else { EvidenceStrength::Negligible }
            }
            EvidenceItem::DeviationResult { punishment_detected, .. } => {
                if *punishment_detected { EvidenceStrength::Strong } else { EvidenceStrength::Weak }
            }
            EvidenceItem::PunishmentDetection { severity, .. } => {
                if *severity > 0.5 { EvidenceStrength::Definitive }
                else if *severity > 0.3 { EvidenceStrength::Strong }
                else { EvidenceStrength::Moderate }
            }
            EvidenceItem::StatisticalTest { test_result } => {
                let p = test_result.p_value.value();
                if p < 0.001 { EvidenceStrength::Definitive }
                else if p < 0.01 { EvidenceStrength::Strong }
                else if p < 0.05 { EvidenceStrength::Moderate }
                else { EvidenceStrength::Weak }
            }
            EvidenceItem::EquilibriumComputation { sustainable, deviation_incentive, .. } => {
                if *sustainable && *deviation_incentive < 0.0 {
                    EvidenceStrength::Definitive
                } else if *sustainable {
                    EvidenceStrength::Strong
                } else {
                    EvidenceStrength::Weak
                }
            }
        }
    }

    pub fn is_supportive(&self) -> bool {
        !matches!(self.strength(), EvidenceStrength::Negligible)
    }
}

impl fmt::Display for EvidenceItem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.strength(), self.description())
    }
}

/// Strength of a single piece of evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum EvidenceStrength {
    Negligible,
    Weak,
    Moderate,
    Strong,
    Definitive,
}

impl EvidenceStrength {
    pub const Decisive: EvidenceStrength = EvidenceStrength::Definitive;

    pub fn numeric(&self) -> f64 {
        match self {
            EvidenceStrength::Negligible => 0.0,
            EvidenceStrength::Weak => 0.25,
            EvidenceStrength::Moderate => 0.5,
            EvidenceStrength::Strong => 0.75,
            EvidenceStrength::Definitive => 1.0,
        }
    }
}

impl fmt::Display for EvidenceStrength {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EvidenceStrength::Negligible => write!(f, "NEGLIGIBLE"),
            EvidenceStrength::Weak => write!(f, "WEAK"),
            EvidenceStrength::Moderate => write!(f, "MODERATE"),
            EvidenceStrength::Strong => write!(f, "STRONG"),
            EvidenceStrength::Definitive => write!(f, "DEFINITIVE"),
        }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Merkle tree
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// A node in the Merkle tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MerkleNode {
    pub hash: String,
    pub left: Option<Box<MerkleNode>>,
    pub right: Option<Box<MerkleNode>>,
}

impl MerkleNode {
    /// Create a leaf node from data.
    pub fn leaf(data: &str) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(b"leaf:");
        hasher.update(data.as_bytes());
        MerkleNode {
            hash: hex::encode(hasher.finalize()),
            left: None,
            right: None,
        }
    }

    /// Create an internal node from two children.
    pub fn branch(left: MerkleNode, right: MerkleNode) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(b"node:");
        hasher.update(left.hash.as_bytes());
        hasher.update(right.hash.as_bytes());
        MerkleNode {
            hash: hex::encode(hasher.finalize()),
            left: Some(Box::new(left)),
            right: Some(Box::new(right)),
        }
    }

    pub fn is_leaf(&self) -> bool {
        self.left.is_none() && self.right.is_none()
    }

    /// Count total nodes in this subtree.
    pub fn node_count(&self) -> usize {
        1 + self.left.as_ref().map_or(0, |n| n.node_count())
          + self.right.as_ref().map_or(0, |n| n.node_count())
    }

    /// Depth of the tree.
    pub fn depth(&self) -> usize {
        if self.is_leaf() {
            0
        } else {
            1 + self.left.as_ref().map_or(0, |n| n.depth())
                .max(self.right.as_ref().map_or(0, |n| n.depth()))
        }
    }
}

impl fmt::Display for MerkleNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MerkleNode({}..., depth={})", &self.hash[..8], self.depth())
    }
}

/// A Merkle tree for verifying evidence integrity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MerkleTree {
    pub root: Option<MerkleNode>,
    pub leaf_count: usize,
}

impl MerkleTree {
    /// Build a Merkle tree from a list of data items.
    pub fn build(items: &[String]) -> Self {
        if items.is_empty() {
            return MerkleTree { root: None, leaf_count: 0 };
        }

        let mut nodes: Vec<MerkleNode> = items.iter()
            .map(|item| MerkleNode::leaf(item))
            .collect();

        // If odd number of nodes, duplicate the last one.
        while nodes.len() > 1 {
            if nodes.len() % 2 == 1 {
                nodes.push(nodes.last().unwrap().clone());
            }
            let mut next_level = Vec::new();
            for chunk in nodes.chunks(2) {
                next_level.push(MerkleNode::branch(chunk[0].clone(), chunk[1].clone()));
            }
            nodes = next_level;
        }

        MerkleTree {
            root: nodes.into_iter().next(),
            leaf_count: items.len(),
        }
    }

    /// Root hash of the tree.
    pub fn root_hash(&self) -> Option<&str> {
        self.root.as_ref().map(|n| n.hash.as_str())
    }

    /// Verify a leaf is in the tree by recomputing from a proof path.
    pub fn verify_proof(root_hash: &str, leaf_data: &str, proof: &[ProofStep]) -> bool {
        let leaf = MerkleNode::leaf(leaf_data);
        let mut current_hash = leaf.hash;

        for step in proof {
            let mut hasher = Sha256::new();
            hasher.update(b"node:");
            match step.direction {
                ProofDirection::Left => {
                    hasher.update(step.sibling_hash.as_bytes());
                    hasher.update(current_hash.as_bytes());
                }
                ProofDirection::Right => {
                    hasher.update(current_hash.as_bytes());
                    hasher.update(step.sibling_hash.as_bytes());
                }
            }
            current_hash = hex::encode(hasher.finalize());
        }

        current_hash == root_hash
    }

    pub fn is_empty(&self) -> bool { self.root.is_none() }

    pub fn depth(&self) -> usize {
        self.root.as_ref().map_or(0, |n| n.depth())
    }
}

impl fmt::Display for MerkleTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.root {
            Some(node) => write!(f, "MerkleTree(root={}..., leaves={})",
                &node.hash[..8], self.leaf_count),
            None => write!(f, "MerkleTree(empty)"),
        }
    }
}

/// Direction in a Merkle proof.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ProofDirection {
    Left,
    Right,
}

/// A step in a Merkle inclusion proof.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofStep {
    pub sibling_hash: String,
    pub direction: ProofDirection,
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Evidence bundle
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// A bundle of evidence items with Merkle tree integrity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceBundle {
    pub id: BundleId,
    pub items: Vec<EvidenceItem>,
    pub merkle_tree: MerkleTree,
    pub overall_strength: EvidenceStrength,
    pub description: String,
}

impl EvidenceBundle {
    /// Create a new bundle and build its Merkle tree.
    pub fn new(items: Vec<EvidenceItem>, description: impl Into<String>) -> Self {
        let hashes: Vec<String> = items.iter().map(|i| i.content_hash()).collect();
        let merkle_tree = MerkleTree::build(&hashes);
        let overall_strength = Self::compute_overall_strength(&items);

        EvidenceBundle {
            id: BundleId::new(),
            items,
            merkle_tree,
            overall_strength,
            description: description.into(),
        }
    }

    /// Add an item and rebuild the Merkle tree.
    pub fn add_item(&mut self, item: EvidenceItem) {
        self.items.push(item);
        let hashes: Vec<String> = self.items.iter().map(|i| i.content_hash()).collect();
        self.merkle_tree = MerkleTree::build(&hashes);
        self.overall_strength = Self::compute_overall_strength(&self.items);
    }

    /// Alias for add_item.
    pub fn add(&mut self, item: EvidenceItem) {
        self.add_item(item);
    }

    /// Verify that the Merkle tree matches the current items.
    pub fn verify_integrity(&self) -> bool {
        let hashes: Vec<String> = self.items.iter().map(|i| i.content_hash()).collect();
        let fresh_tree = MerkleTree::build(&hashes);
        match (self.merkle_tree.root_hash(), fresh_tree.root_hash()) {
            (Some(a), Some(b)) => a == b,
            (None, None) => true,
            _ => false,
        }
    }

    pub fn num_items(&self) -> usize { self.items.len() }

    pub fn root_hash(&self) -> Option<&str> {
        self.merkle_tree.root_hash()
    }

    /// Count items by strength.
    pub fn strength_distribution(&self) -> std::collections::HashMap<EvidenceStrength, usize> {
        let mut dist = std::collections::HashMap::new();
        for item in &self.items {
            *dist.entry(item.strength()).or_insert(0) += 1;
        }
        dist
    }

    /// Items filtered by minimum strength.
    pub fn items_above_strength(&self, min: EvidenceStrength) -> Vec<&EvidenceItem> {
        self.items.iter().filter(|i| i.strength() >= min).collect()
    }

    /// Compute overall strength as the median of all item strengths.
    fn compute_overall_strength(items: &[EvidenceItem]) -> EvidenceStrength {
        if items.is_empty() { return EvidenceStrength::Negligible; }
        let mut strengths: Vec<EvidenceStrength> = items.iter().map(|i| i.strength()).collect();
        strengths.sort();
        strengths[strengths.len() / 2]
    }

    /// Serialize the bundle to JSON.
    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self).map_err(|e| format!("{}", e))
    }

    /// Deserialize from JSON.
    pub fn from_json(s: &str) -> Result<Self, String> {
        serde_json::from_str(s).map_err(|e| format!("{}", e))
    }
}

impl fmt::Display for EvidenceBundle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "EvidenceBundle({}, {} items, strength={})",
            self.id.short(), self.items.len(), self.overall_strength)
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Tests
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[cfg(test)]
mod tests {
    use super::*;
    use crate::statistics::TestStatistic;

    #[test]
    fn test_evidence_item_price_data() {
        let item = EvidenceItem::PriceData {
            description: "test".into(),
            mean_price: 8.0,
            competitive_price: 5.0,
            monopoly_price: 10.0,
            collusion_index: 0.6,
            sample_size: 100,
        };
        assert_eq!(item.strength(), EvidenceStrength::Moderate);
        assert!(item.is_supportive());
    }

    #[test]
    fn test_evidence_item_hash_deterministic() {
        let item = EvidenceItem::PriceData {
            description: "test".into(),
            mean_price: 8.0, competitive_price: 5.0, monopoly_price: 10.0,
            collusion_index: 0.6, sample_size: 100,
        };
        let h1 = item.content_hash();
        let h2 = item.content_hash();
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_evidence_strength_ordering() {
        assert!(EvidenceStrength::Weak < EvidenceStrength::Moderate);
        assert!(EvidenceStrength::Moderate < EvidenceStrength::Strong);
        assert!(EvidenceStrength::Strong < EvidenceStrength::Definitive);
    }

    #[test]
    fn test_merkle_tree_build() {
        let items = vec!["a".into(), "b".into(), "c".into(), "d".into()];
        let tree = MerkleTree::build(&items);
        assert_eq!(tree.leaf_count, 4);
        assert!(tree.root_hash().is_some());
    }

    #[test]
    fn test_merkle_tree_single_leaf() {
        let tree = MerkleTree::build(&["hello".into()]);
        assert_eq!(tree.leaf_count, 1);
        assert!(tree.root_hash().is_some());
    }

    #[test]
    fn test_merkle_tree_empty() {
        let tree = MerkleTree::build(&[]);
        assert!(tree.is_empty());
        assert_eq!(tree.leaf_count, 0);
    }

    #[test]
    fn test_merkle_tree_deterministic() {
        let items = vec!["x".into(), "y".into()];
        let t1 = MerkleTree::build(&items);
        let t2 = MerkleTree::build(&items);
        assert_eq!(t1.root_hash(), t2.root_hash());
    }

    #[test]
    fn test_merkle_tree_different_data() {
        let t1 = MerkleTree::build(&["a".into(), "b".into()]);
        let t2 = MerkleTree::build(&["a".into(), "c".into()]);
        assert_ne!(t1.root_hash(), t2.root_hash());
    }

    #[test]
    fn test_evidence_bundle_integrity() {
        let items = vec![
            EvidenceItem::PriceData {
                description: "test1".into(), mean_price: 8.0,
                competitive_price: 5.0, monopoly_price: 10.0,
                collusion_index: 0.6, sample_size: 100,
            },
            EvidenceItem::PriceData {
                description: "test2".into(), mean_price: 9.0,
                competitive_price: 5.0, monopoly_price: 10.0,
                collusion_index: 0.8, sample_size: 50,
            },
        ];
        let bundle = EvidenceBundle::new(items, "test bundle");
        assert!(bundle.verify_integrity());
        assert_eq!(bundle.num_items(), 2);
    }

    #[test]
    fn test_evidence_bundle_json_roundtrip() {
        let items = vec![EvidenceItem::PriceData {
            description: "rd".into(), mean_price: 7.0,
            competitive_price: 5.0, monopoly_price: 10.0,
            collusion_index: 0.4, sample_size: 200,
        }];
        let bundle = EvidenceBundle::new(items, "json test");
        let json = bundle.to_json().unwrap();
        let bundle2 = EvidenceBundle::from_json(&json).unwrap();
        assert_eq!(bundle.num_items(), bundle2.num_items());
        assert_eq!(bundle.root_hash(), bundle2.root_hash());
    }

    #[test]
    fn test_evidence_bundle_add_item() {
        let mut bundle = EvidenceBundle::new(vec![], "empty");
        let old_hash = bundle.root_hash().map(String::from);
        bundle.add_item(EvidenceItem::PriceData {
            description: "new".into(), mean_price: 6.0,
            competitive_price: 5.0, monopoly_price: 10.0,
            collusion_index: 0.2, sample_size: 30,
        });
        assert_eq!(bundle.num_items(), 1);
        assert_ne!(bundle.root_hash().map(String::from), old_hash);
        assert!(bundle.verify_integrity());
    }

    #[test]
    fn test_evidence_strength_distribution() {
        let items = vec![
            EvidenceItem::PriceData {
                description: "a".into(), mean_price: 9.0,
                competitive_price: 5.0, monopoly_price: 10.0,
                collusion_index: 0.9, sample_size: 100,
            },
            EvidenceItem::PriceData {
                description: "b".into(), mean_price: 6.0,
                competitive_price: 5.0, monopoly_price: 10.0,
                collusion_index: 0.3, sample_size: 100,
            },
        ];
        let bundle = EvidenceBundle::new(items, "dist test");
        let dist = bundle.strength_distribution();
        assert_eq!(dist.len(), 2);
    }

    #[test]
    fn test_merkle_node_depth() {
        let items = vec!["a".into(), "b".into(), "c".into(), "d".into()];
        let tree = MerkleTree::build(&items);
        assert!(tree.depth() >= 1);
    }

    #[test]
    fn test_evidence_item_display() {
        let item = EvidenceItem::PriceData {
            description: "supra-competitive".into(), mean_price: 8.0,
            competitive_price: 5.0, monopoly_price: 10.0,
            collusion_index: 0.6, sample_size: 100,
        };
        let s = format!("{}", item);
        assert!(s.contains("MODERATE"));
        assert!(s.contains("supra-competitive"));
    }
}
