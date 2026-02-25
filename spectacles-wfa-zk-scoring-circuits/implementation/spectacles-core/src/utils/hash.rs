//! Hashing utilities: BLAKE3, Merkle trees, hash chains, commitments.

use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Primary hasher using BLAKE3
#[derive(Debug, Clone)]
pub struct SpectaclesHasher {
    domain: String,
}

impl SpectaclesHasher {
    pub fn new() -> Self {
        Self { domain: String::new() }
    }
    
    pub fn with_domain(domain: &str) -> Self {
        Self { domain: domain.to_string() }
    }
    
    /// Hash arbitrary bytes
    pub fn hash(&self, data: &[u8]) -> [u8; 32] {
        let mut input = Vec::new();
        if !self.domain.is_empty() {
            input.extend_from_slice(self.domain.as_bytes());
            input.push(0); // domain separator
        }
        input.extend_from_slice(data);
        
        let hash = blake3::hash(&input);
        *hash.as_bytes()
    }
    
    /// Hash a string
    pub fn hash_str(&self, s: &str) -> [u8; 32] {
        self.hash(s.as_bytes())
    }
    
    /// Hash two values together
    pub fn hash_pair(&self, a: &[u8; 32], b: &[u8; 32]) -> [u8; 32] {
        let mut combined = Vec::with_capacity(64);
        combined.extend_from_slice(a);
        combined.extend_from_slice(b);
        self.hash(&combined)
    }
    
    /// Hash to a hex string
    pub fn hash_hex(&self, data: &[u8]) -> String {
        let h = self.hash(data);
        hex::encode(h)
    }
}

/// Domain-separated hasher for different proof components
#[derive(Debug, Clone)]
pub struct DomainSeparatedHasher {
    hashers: HashMap<String, SpectaclesHasher>,
}

impl DomainSeparatedHasher {
    pub fn new(domains: &[&str]) -> Self {
        let mut hashers = HashMap::new();
        for &domain in domains {
            hashers.insert(domain.to_string(), SpectaclesHasher::with_domain(domain));
        }
        Self { hashers }
    }
    
    pub fn hash(&self, domain: &str, data: &[u8]) -> [u8; 32] {
        self.hashers.get(domain)
            .unwrap_or(&SpectaclesHasher::new())
            .hash(data)
    }
    
    pub fn proof_domains() -> Self {
        Self::new(&["commitment", "challenge", "response", "merkle", "evaluation"])
    }
}

/// Merkle tree with BLAKE3 hashing
#[derive(Debug, Clone)]
pub struct MerkleTree {
    nodes: Vec<[u8; 32]>,
    leaf_count: usize,
    hasher: SpectaclesHasher,
}

impl MerkleTree {
    /// Build a Merkle tree from leaf data
    pub fn build(leaves: &[Vec<u8>]) -> Self {
        let hasher = SpectaclesHasher::with_domain("merkle");
        
        if leaves.is_empty() {
            return Self {
                nodes: vec![[0u8; 32]],
                leaf_count: 0,
                hasher,
            };
        }
        
        // Pad to power of 2
        let n = leaves.len().next_power_of_two();
        let mut nodes = Vec::with_capacity(2 * n);
        
        // Hash leaves
        let mut leaf_hashes: Vec<[u8; 32]> = leaves.iter()
            .map(|leaf| hasher.hash(leaf))
            .collect();
        
        // Pad with zero hashes
        while leaf_hashes.len() < n {
            leaf_hashes.push([0u8; 32]);
        }
        
        // Build tree bottom-up
        // Nodes layout: [root, left, right, ll, lr, rl, rr, ...]
        // Actually, use array layout: tree[1] = root, tree[2i] = left child, tree[2i+1] = right child
        // Leaves at tree[n..2n]
        nodes.resize(2 * n, [0u8; 32]);
        
        // Place leaves
        for (i, h) in leaf_hashes.iter().enumerate() {
            nodes[n + i] = *h;
        }
        
        // Build internal nodes
        for i in (1..n).rev() {
            nodes[i] = hasher.hash_pair(&nodes[2 * i], &nodes[2 * i + 1]);
        }
        
        Self {
            nodes,
            leaf_count: leaves.len(),
            hasher,
        }
    }
    
    /// Get the root hash
    pub fn root(&self) -> [u8; 32] {
        if self.nodes.len() > 1 {
            self.nodes[1]
        } else {
            self.nodes[0]
        }
    }
    
    /// Generate a Merkle proof for a leaf at the given index
    pub fn proof(&self, index: usize) -> MerkleProof {
        let n = self.nodes.len() / 2;
        let mut path = Vec::new();
        let mut idx = n + index;
        
        while idx > 1 {
            let sibling = if idx % 2 == 0 { idx + 1 } else { idx - 1 };
            if sibling < self.nodes.len() {
                path.push(MerkleProofStep {
                    hash: self.nodes[sibling],
                    is_right: idx % 2 == 0,
                });
            }
            idx /= 2;
        }
        
        MerkleProof {
            leaf_index: index,
            path,
            root: self.root(),
        }
    }
    
    /// Verify a Merkle proof
    pub fn verify_proof(proof: &MerkleProof, leaf_data: &[u8]) -> bool {
        let hasher = SpectaclesHasher::with_domain("merkle");
        let mut current = hasher.hash(leaf_data);
        
        for step in &proof.path {
            current = if step.is_right {
                hasher.hash_pair(&current, &step.hash)
            } else {
                hasher.hash_pair(&step.hash, &current)
            };
        }
        
        current == proof.root
    }
    
    pub fn leaf_count(&self) -> usize {
        self.leaf_count
    }
}

/// A Merkle proof (authentication path)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MerkleProof {
    pub leaf_index: usize,
    pub path: Vec<MerkleProofStep>,
    pub root: [u8; 32],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MerkleProofStep {
    pub hash: [u8; 32],
    pub is_right: bool,
}

/// Hash chain: H(H(H(...H(seed)...)))
#[derive(Debug, Clone)]
pub struct HashChain {
    chain: Vec<[u8; 32]>,
    hasher: SpectaclesHasher,
}

impl HashChain {
    /// Build a hash chain of the given length starting from a seed
    pub fn build(seed: &[u8], length: usize) -> Self {
        let hasher = SpectaclesHasher::with_domain("chain");
        let mut chain = Vec::with_capacity(length);
        
        let mut current = hasher.hash(seed);
        chain.push(current);
        
        for _ in 1..length {
            current = hasher.hash(&current);
            chain.push(current);
        }
        
        Self { chain, hasher }
    }
    
    /// Get the hash at a specific position
    pub fn get(&self, index: usize) -> Option<&[u8; 32]> {
        self.chain.get(index)
    }
    
    /// Get the final hash
    pub fn tip(&self) -> &[u8; 32] {
        self.chain.last().unwrap()
    }
    
    /// Verify that hash at position i+1 is H(hash at position i)
    pub fn verify_step(&self, index: usize) -> bool {
        if index + 1 >= self.chain.len() {
            return false;
        }
        let expected = self.hasher.hash(&self.chain[index]);
        expected == self.chain[index + 1]
    }
    
    /// Verify the entire chain
    pub fn verify_all(&self) -> bool {
        for i in 0..self.chain.len().saturating_sub(1) {
            if !self.verify_step(i) {
                return false;
            }
        }
        true
    }
    
    pub fn length(&self) -> usize {
        self.chain.len()
    }
}

/// Hash-based commitment scheme: commit(value, randomness) = H(value || randomness)
#[derive(Debug, Clone)]
pub struct Commitment {
    hash: [u8; 32],
}

impl Commitment {
    /// Create a commitment to a value with given randomness
    pub fn commit(value: &[u8], randomness: &[u8]) -> (Self, CommitmentOpening) {
        let hasher = SpectaclesHasher::with_domain("commitment");
        let mut data = Vec::with_capacity(value.len() + randomness.len());
        data.extend_from_slice(value);
        data.extend_from_slice(randomness);
        
        let hash = hasher.hash(&data);
        
        (
            Self { hash },
            CommitmentOpening {
                value: value.to_vec(),
                randomness: randomness.to_vec(),
            },
        )
    }
    
    /// Verify that an opening matches this commitment
    pub fn verify(&self, opening: &CommitmentOpening) -> bool {
        let hasher = SpectaclesHasher::with_domain("commitment");
        let mut data = Vec::with_capacity(opening.value.len() + opening.randomness.len());
        data.extend_from_slice(&opening.value);
        data.extend_from_slice(&opening.randomness);
        
        hasher.hash(&data) == self.hash
    }
    
    pub fn hash(&self) -> &[u8; 32] {
        &self.hash
    }
    
    pub fn hash_hex(&self) -> String {
        hex::encode(self.hash)
    }
}

/// Opening for a commitment
#[derive(Debug, Clone)]
pub struct CommitmentOpening {
    pub value: Vec<u8>,
    pub randomness: Vec<u8>,
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_hasher_basic() {
        let hasher = SpectaclesHasher::new();
        let h1 = hasher.hash(b"hello");
        let h2 = hasher.hash(b"hello");
        assert_eq!(h1, h2);
        
        let h3 = hasher.hash(b"world");
        assert_ne!(h1, h3);
    }
    
    #[test]
    fn test_hasher_domain_separation() {
        let h1 = SpectaclesHasher::with_domain("domain1");
        let h2 = SpectaclesHasher::with_domain("domain2");
        
        let hash1 = h1.hash(b"same data");
        let hash2 = h2.hash(b"same data");
        assert_ne!(hash1, hash2);
    }
    
    #[test]
    fn test_hasher_hex() {
        let hasher = SpectaclesHasher::new();
        let hex = hasher.hash_hex(b"test");
        assert_eq!(hex.len(), 64);
    }
    
    #[test]
    fn test_merkle_tree_single_leaf() {
        let tree = MerkleTree::build(&[b"leaf1".to_vec()]);
        assert_eq!(tree.leaf_count(), 1);
        let root = tree.root();
        assert_ne!(root, [0u8; 32]);
    }
    
    #[test]
    fn test_merkle_tree_multiple_leaves() {
        let leaves: Vec<Vec<u8>> = (0..4).map(|i| format!("leaf{}", i).into_bytes()).collect();
        let tree = MerkleTree::build(&leaves);
        assert_eq!(tree.leaf_count(), 4);
    }
    
    #[test]
    fn test_merkle_proof_verify() {
        let leaves: Vec<Vec<u8>> = (0..4).map(|i| format!("leaf{}", i).into_bytes()).collect();
        let tree = MerkleTree::build(&leaves);
        
        for i in 0..4 {
            let proof = tree.proof(i);
            assert!(MerkleTree::verify_proof(&proof, &leaves[i]),
                "Proof verification failed for leaf {}", i);
        }
    }
    
    #[test]
    fn test_merkle_proof_invalid() {
        let leaves: Vec<Vec<u8>> = (0..4).map(|i| format!("leaf{}", i).into_bytes()).collect();
        let tree = MerkleTree::build(&leaves);
        
        let proof = tree.proof(0);
        assert!(!MerkleTree::verify_proof(&proof, b"wrong data"));
    }
    
    #[test]
    fn test_merkle_tree_empty() {
        let tree = MerkleTree::build(&[]);
        assert_eq!(tree.leaf_count(), 0);
    }
    
    #[test]
    fn test_hash_chain() {
        let chain = HashChain::build(b"seed", 10);
        assert_eq!(chain.length(), 10);
        assert!(chain.verify_all());
    }
    
    #[test]
    fn test_hash_chain_step_verification() {
        let chain = HashChain::build(b"test_seed", 5);
        for i in 0..4 {
            assert!(chain.verify_step(i));
        }
    }
    
    #[test]
    fn test_hash_chain_deterministic() {
        let chain1 = HashChain::build(b"seed", 5);
        let chain2 = HashChain::build(b"seed", 5);
        assert_eq!(chain1.tip(), chain2.tip());
    }
    
    #[test]
    fn test_commitment_scheme() {
        let value = b"secret value";
        let randomness = b"random bytes 123456";
        
        let (commitment, opening) = Commitment::commit(value, randomness);
        assert!(commitment.verify(&opening));
    }
    
    #[test]
    fn test_commitment_binding() {
        let (commitment, _opening) = Commitment::commit(b"value1", b"rand1");
        
        let wrong_opening = CommitmentOpening {
            value: b"value2".to_vec(),
            randomness: b"rand1".to_vec(),
        };
        assert!(!commitment.verify(&wrong_opening));
    }
    
    #[test]
    fn test_commitment_hiding() {
        let (c1, _) = Commitment::commit(b"same", b"rand1");
        let (c2, _) = Commitment::commit(b"same", b"rand2");
        assert_ne!(c1.hash(), c2.hash());
    }
    
    #[test]
    fn test_domain_separated_hasher() {
        let dsh = DomainSeparatedHasher::proof_domains();
        
        let h1 = dsh.hash("commitment", b"data");
        let h2 = dsh.hash("challenge", b"data");
        assert_ne!(h1, h2);
    }
    
    #[test]
    fn test_commitment_hex() {
        let (commitment, _) = Commitment::commit(b"test", b"rand");
        let hex = commitment.hash_hex();
        assert_eq!(hex.len(), 64);
    }
    
    #[test]
    fn test_merkle_non_power_of_two() {
        let leaves: Vec<Vec<u8>> = (0..5).map(|i| format!("leaf{}", i).into_bytes()).collect();
        let tree = MerkleTree::build(&leaves);
        assert_eq!(tree.leaf_count(), 5);
        
        let proof = tree.proof(0);
        assert!(MerkleTree::verify_proof(&proof, &leaves[0]));
    }
}
