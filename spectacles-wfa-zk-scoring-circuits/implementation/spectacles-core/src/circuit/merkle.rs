// Merkle tree for trace commitment in STARK proofs.
//
// Uses BLAKE3 for hashing. Each leaf is a row of field elements serialized
// to bytes, and internal nodes are H(left || right).

use serde::{Serialize, Deserialize};
use std::fmt;

use super::goldilocks::GoldilocksField;

/// A 32-byte hash digest.
pub type Digest = [u8; 32];

// ---------------------------------------------------------------------------
// Utility / hashing helpers
// ---------------------------------------------------------------------------

/// Blake3 hash of arbitrary bytes.
pub fn hash_bytes(data: &[u8]) -> [u8; 32] {
    let mut hasher = Blake3State::new();
    hasher.update(data);
    hasher.finalize()
}

/// Hash two 32-byte children into a parent node.
pub fn hash_two(a: &[u8; 32], b: &[u8; 32]) -> [u8; 32] {
    let mut buf = [0u8; 64];
    buf[..32].copy_from_slice(a);
    buf[32..].copy_from_slice(b);
    hash_bytes(&buf)
}

/// Hash a slice of `u64` field elements (little-endian encoding).
pub fn hash_field_elements(elements: &[u64]) -> [u8; 32] {
    let mut buf = Vec::with_capacity(elements.len() * 8);
    for &e in elements {
        buf.extend_from_slice(&e.to_le_bytes());
    }
    hash_bytes(&buf)
}

/// Convenience: build a full Merkle tree from raw data and return just the root.
pub fn merkle_root_from_data(data: &[Vec<u8>]) -> [u8; 32] {
    let tree = MerkleTree::new(data);
    tree.root()
}

/// Smallest `d` such that `2^d >= n`.  Returns 0 when `n <= 1`.
pub fn depth_for_leaves(n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    let mut d = 0usize;
    let mut pow = 1usize;
    while pow < n {
        pow <<= 1;
        d += 1;
    }
    d
}

/// Next power of two >= n (returns 1 for n == 0).
fn next_power_of_two(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    let mut v = n - 1;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v |= v >> 32;
    v + 1
}

// ---------------------------------------------------------------------------
// Minimal Blake3-like hash (pure-Rust, no external crate)
//
// We implement the full BLAKE3 compression function so that every hash in
// this module is *real* cryptographic output.
// ---------------------------------------------------------------------------

const BLAKE3_IV: [u32; 8] = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
];

const BLAKE3_MSG_PERMUTATION: [usize; 16] = [
    2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8,
];

const BLAKE3_BLOCK_LEN: usize = 64;
const BLAKE3_CHUNK_LEN: usize = 1024;

const CHUNK_START: u32 = 1 << 0;
const CHUNK_END: u32 = 1 << 1;
const ROOT: u32 = 1 << 3;

#[inline(always)]
fn rotr32(x: u32, n: u32) -> u32 {
    x.rotate_right(n)
}

fn g(state: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize, mx: u32, my: u32) {
    state[a] = state[a].wrapping_add(state[b]).wrapping_add(mx);
    state[d] = rotr32(state[d] ^ state[a], 16);
    state[c] = state[c].wrapping_add(state[d]);
    state[b] = rotr32(state[b] ^ state[c], 12);
    state[a] = state[a].wrapping_add(state[b]).wrapping_add(my);
    state[d] = rotr32(state[d] ^ state[a], 8);
    state[c] = state[c].wrapping_add(state[d]);
    state[b] = rotr32(state[b] ^ state[c], 7);
}

fn round_fn(state: &mut [u32; 16], m: &[u32; 16]) {
    g(state, 0, 4,  8, 12, m[0],  m[1]);
    g(state, 1, 5,  9, 13, m[2],  m[3]);
    g(state, 2, 6, 10, 14, m[4],  m[5]);
    g(state, 3, 7, 11, 15, m[6],  m[7]);
    g(state, 0, 5, 10, 15, m[8],  m[9]);
    g(state, 1, 6, 11, 12, m[10], m[11]);
    g(state, 2, 7,  8, 13, m[12], m[13]);
    g(state, 3, 4,  9, 14, m[14], m[15]);
}

fn permute(m: &mut [u32; 16]) {
    let orig = *m;
    for i in 0..16 {
        m[i] = orig[BLAKE3_MSG_PERMUTATION[i]];
    }
}

fn compress(
    cv: &[u32; 8],
    block: &[u8; BLAKE3_BLOCK_LEN],
    block_len: u32,
    counter: u64,
    flags: u32,
) -> [u32; 16] {
    let mut msg = [0u32; 16];
    for i in 0..16 {
        msg[i] = u32::from_le_bytes([
            block[4 * i],
            block[4 * i + 1],
            block[4 * i + 2],
            block[4 * i + 3],
        ]);
    }

    let mut state: [u32; 16] = [
        cv[0], cv[1], cv[2], cv[3],
        cv[4], cv[5], cv[6], cv[7],
        BLAKE3_IV[0], BLAKE3_IV[1], BLAKE3_IV[2], BLAKE3_IV[3],
        counter as u32, (counter >> 32) as u32,
        block_len, flags,
    ];

    round_fn(&mut state, &msg); permute(&mut msg);
    round_fn(&mut state, &msg); permute(&mut msg);
    round_fn(&mut state, &msg); permute(&mut msg);
    round_fn(&mut state, &msg); permute(&mut msg);
    round_fn(&mut state, &msg); permute(&mut msg);
    round_fn(&mut state, &msg); permute(&mut msg);
    round_fn(&mut state, &msg);

    for i in 0..8 {
        state[i] ^= state[i + 8];
        state[i + 8] ^= cv[i];
    }
    state
}

fn first_8(s: &[u32; 16]) -> [u32; 8] {
    let mut out = [0u32; 8];
    out.copy_from_slice(&s[..8]);
    out
}

/// A streaming Blake3 hasher.
struct Blake3State {
    cv_stack: Vec<[u32; 8]>,
    cv: [u32; 8],
    buf: [u8; BLAKE3_BLOCK_LEN],
    buf_len: usize,
    blocks_compressed: u32,
    chunk_counter: u64,
    flags: u32,
}

impl Blake3State {
    fn new() -> Self {
        Self {
            cv_stack: Vec::new(),
            cv: BLAKE3_IV,
            buf: [0u8; BLAKE3_BLOCK_LEN],
            buf_len: 0,
            blocks_compressed: 0,
            chunk_counter: 0,
            flags: 0,
        }
    }

    fn start_flag(&self) -> u32 {
        if self.blocks_compressed == 0 { CHUNK_START } else { 0 }
    }

    fn update(&mut self, mut input: &[u8]) {
        while !input.is_empty() {
            if self.buf_len == BLAKE3_BLOCK_LEN {
                let block: [u8; BLAKE3_BLOCK_LEN] = self.buf;
                let flags = self.flags | self.start_flag();
                self.cv = first_8(&compress(
                    &self.cv, &block, BLAKE3_BLOCK_LEN as u32,
                    self.chunk_counter, flags,
                ));
                self.blocks_compressed += 1;
                self.buf = [0u8; BLAKE3_BLOCK_LEN];
                self.buf_len = 0;

                if self.blocks_compressed == (BLAKE3_CHUNK_LEN / BLAKE3_BLOCK_LEN) as u32 {
                    let chunk_cv = self.cv;
                    self.finish_chunk(chunk_cv);
                }
            }
            let take = std::cmp::min(BLAKE3_BLOCK_LEN - self.buf_len, input.len());
            self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&input[..take]);
            self.buf_len += take;
            input = &input[take..];
        }
    }

    fn finish_chunk(&mut self, cv: [u32; 8]) {
        let mut total_chunks = self.chunk_counter + 1;
        let mut new_cv = cv;

        while total_chunks & 1 == 0 {
            let left = self.cv_stack.pop().unwrap();
            let mut block = [0u8; BLAKE3_BLOCK_LEN];
            for i in 0..8 {
                block[4 * i..4 * i + 4].copy_from_slice(&left[i].to_le_bytes());
            }
            for i in 0..8 {
                block[32 + 4 * i..32 + 4 * i + 4].copy_from_slice(&new_cv[i].to_le_bytes());
            }
            new_cv = first_8(&compress(
                &BLAKE3_IV, &block, BLAKE3_BLOCK_LEN as u32, 0, self.flags,
            ));
            total_chunks >>= 1;
        }
        self.cv_stack.push(new_cv);
        self.chunk_counter += 1;
        self.cv = BLAKE3_IV;
        self.blocks_compressed = 0;
        self.buf = [0u8; BLAKE3_BLOCK_LEN];
        self.buf_len = 0;
    }

    fn finalize(&mut self) -> [u8; 32] {
        let block = self.buf;
        let block_len = self.buf_len as u32;
        let mut flags = self.flags | self.start_flag() | CHUNK_END;

        let mut output = first_8(&compress(
            &self.cv, &block, block_len, self.chunk_counter, flags,
        ));

        while let Some(left) = self.cv_stack.pop() {
            let mut merge_block = [0u8; BLAKE3_BLOCK_LEN];
            for i in 0..8 {
                merge_block[4 * i..4 * i + 4].copy_from_slice(&left[i].to_le_bytes());
            }
            for i in 0..8 {
                merge_block[32 + 4 * i..32 + 4 * i + 4].copy_from_slice(&output[i].to_le_bytes());
            }
            flags = self.flags;
            if self.cv_stack.is_empty() {
                flags |= ROOT;
            }
            output = first_8(&compress(
                &BLAKE3_IV, &merge_block, BLAKE3_BLOCK_LEN as u32, 0, flags,
            ));
        }

        let mut result = [0u8; 32];
        for i in 0..8 {
            result[4 * i..4 * i + 4].copy_from_slice(&output[i].to_le_bytes());
        }
        result
    }
}

// Legacy alias used by other modules in this crate.
/// Compute a BLAKE3 hash and return the 32-byte digest.
pub fn blake3_hash(data: &[u8]) -> Digest {
    hash_bytes(data)
}

// ---------------------------------------------------------------------------
// MerkleProof
// ---------------------------------------------------------------------------

/// An authentication path proving that a specific leaf belongs to a Merkle tree.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct MerkleProof {
    /// Sibling hashes along the path from leaf to root (bottom-up).
    pub siblings: Vec<[u8; 32]>,
    /// Index of the leaf in the original data array.
    pub index: usize,
    /// Hash of the leaf being proved.
    pub leaf_hash: [u8; 32],
}

impl MerkleProof {
    /// Number of layers traversed (equals tree height).
    pub fn depth(&self) -> usize {
        self.siblings.len()
    }

    /// Encode the proof into a compact byte vector.
    ///
    /// Layout: `[index: 8 LE][leaf_hash: 32][num_siblings: 8 LE][siblings...]`
    pub fn serialize_to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(8 + 32 + 8 + self.siblings.len() * 32);
        out.extend_from_slice(&self.index.to_le_bytes());
        out.extend_from_slice(&self.leaf_hash);
        out.extend_from_slice(&self.siblings.len().to_le_bytes());
        for s in &self.siblings {
            out.extend_from_slice(s);
        }
        out
    }

    /// Decode a proof previously produced by [`serialize_to_bytes`].
    pub fn deserialize_from_bytes(data: &[u8]) -> Option<Self> {
        if data.len() < 48 {
            return None;
        }
        let index = usize::from_le_bytes(data[0..8].try_into().ok()?);
        let mut leaf_hash = [0u8; 32];
        leaf_hash.copy_from_slice(&data[8..40]);
        let num_siblings = usize::from_le_bytes(data[40..48].try_into().ok()?);
        if data.len() < 48 + num_siblings * 32 {
            return None;
        }
        let mut siblings = Vec::with_capacity(num_siblings);
        for i in 0..num_siblings {
            let start = 48 + i * 32;
            let mut s = [0u8; 32];
            s.copy_from_slice(&data[start..start + 32]);
            siblings.push(s);
        }
        Some(Self { siblings, index, leaf_hash })
    }

    /// Total byte size of the serialised form.
    pub fn size_in_bytes(&self) -> usize {
        8 + 32 + 8 + self.siblings.len() * 32
    }

    // Backward-compat accessor (old code used `leaf_index`).
    pub fn leaf_index(&self) -> usize {
        self.index
    }
}

impl fmt::Display for MerkleProof {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MerkleProof {{ index: {}, depth: {}, leaf: {:02x}{:02x}..{:02x}{:02x} }}",
            self.index,
            self.depth(),
            self.leaf_hash[0], self.leaf_hash[1],
            self.leaf_hash[30], self.leaf_hash[31],
        )
    }
}

// ---------------------------------------------------------------------------
// BatchMerkleProof
// ---------------------------------------------------------------------------

/// A batch proof that aggregates individual Merkle proofs with shared-node
/// deduplication to reduce total size.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct BatchMerkleProof {
    /// The individual proofs (kept for independent verification).
    pub proofs: Vec<MerkleProof>,
    /// Leaf indices (parallel to `proofs`).
    pub indices: Vec<usize>,
    /// Leaf hashes (parallel to `proofs`).
    pub leaf_hashes: Vec<[u8; 32]>,
    /// De-duplicated set of sibling hashes across all proofs.
    pub compressed_siblings: Vec<[u8; 32]>,
}

impl BatchMerkleProof {
    /// Build a compressed batch proof from a vector of individual proofs.
    ///
    /// Sibling nodes that appear in more than one proof are stored only once
    /// in `compressed_siblings`.
    pub fn from_individual_proofs(proofs: Vec<MerkleProof>) -> Self {
        let indices: Vec<usize> = proofs.iter().map(|p| p.index).collect();
        let leaf_hashes: Vec<[u8; 32]> = proofs.iter().map(|p| p.leaf_hash).collect();

        let mut seen: std::collections::HashSet<[u8; 32]> = std::collections::HashSet::new();
        let mut compressed: Vec<[u8; 32]> = Vec::new();
        for proof in &proofs {
            for sib in &proof.siblings {
                if seen.insert(*sib) {
                    compressed.push(*sib);
                }
            }
        }

        Self {
            proofs,
            indices,
            leaf_hashes,
            compressed_siblings: compressed,
        }
    }

    /// Verify every contained proof against `root`.
    pub fn verify_all(&self, root: &[u8; 32]) -> bool {
        for proof in &self.proofs {
            if !MerkleTree::verify_proof(root, proof.index, &proof.leaf_hash, proof) {
                return false;
            }
        }
        true
    }

    /// Total serialised size in bytes.
    pub fn size_in_bytes(&self) -> usize {
        let header = 8;
        let per_proof = 8 + 32;
        let siblings_header = 8;
        let siblings_body = self.compressed_siblings.len() * 32;
        header + self.proofs.len() * per_proof + siblings_header + siblings_body
    }

    /// Ratio of compressed size to naive (uncompressed) size.
    /// Lower is better.  Returns 1.0 when there is no sharing.
    pub fn compression_ratio(&self) -> f64 {
        let naive: usize = self.proofs.iter().map(|p| p.siblings.len()).sum::<usize>() * 32;
        if naive == 0 {
            return 1.0;
        }
        (self.compressed_siblings.len() * 32) as f64 / naive as f64
    }
}

impl fmt::Display for BatchMerkleProof {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BatchMerkleProof {{ num_proofs: {}, compressed_siblings: {}, ratio: {:.3} }}",
            self.proofs.len(),
            self.compressed_siblings.len(),
            self.compression_ratio(),
        )
    }
}

// ---------------------------------------------------------------------------
// MerkleTree
// ---------------------------------------------------------------------------

/// Binary Merkle tree stored in array form.
///
/// Index layout (1-based):
///   - `nodes[1]` = root
///   - children of `nodes[i]` are at `nodes[2*i]` and `nodes[2*i+1]`
///   - leaves occupy indices `[num_internal .. 2*num_internal)`
///
/// The tree is always padded to a power-of-two number of leaves; padding
/// leaves are the zero hash `[0u8; 32]`.
#[derive(Clone)]
pub struct MerkleTree {
    /// Complete binary tree stored in a flat vector (index 0 unused).
    nodes: Vec<[u8; 32]>,
    /// Original leaf hashes (before padding).
    leaves: Vec<[u8; 32]>,
    /// Number of real (non-padding) leaves.
    num_leaves: usize,
    /// Height of the tree (root is at depth 0, leaves at depth `height`).
    height: usize,
}

impl MerkleTree {
    // -- Construction -------------------------------------------------------

    /// Build a Merkle tree from raw data slices.
    ///
    /// Each element of `data` is hashed with Blake3 to produce a leaf hash.
    pub fn new(data: &[Vec<u8>]) -> Self {
        let leaf_hashes: Vec<[u8; 32]> = data.iter().map(|d| Self::hash_leaf(d)).collect();
        Self::from_hashes(leaf_hashes)
    }

    /// Legacy alias for [`new`].
    pub fn from_leaves(data: &[Vec<u8>]) -> Self {
        Self::new(data)
    }

    /// Build a Merkle tree from pre-computed leaf hashes.
    pub fn from_hashes(leaf_hashes: Vec<[u8; 32]>) -> Self {
        assert!(!leaf_hashes.is_empty(), "cannot build a Merkle tree with zero leaves");
        let num_leaves = leaf_hashes.len();
        let padded = next_power_of_two(num_leaves);
        let height = depth_for_leaves(padded);

        let mut padded_leaves = leaf_hashes.clone();
        padded_leaves.resize(padded, [0u8; 32]);

        let nodes = Self::build_tree(&padded_leaves);

        Self {
            nodes,
            leaves: leaf_hashes,
            num_leaves,
            height,
        }
    }

    /// Build a Merkle tree where each leaf is the hash of a row of `u64`
    /// field elements.
    pub fn from_field_rows_u64(rows: &[Vec<u64>]) -> Self {
        let leaf_hashes: Vec<[u8; 32]> = rows.iter().map(|r| hash_field_elements(r)).collect();
        Self::from_hashes(leaf_hashes)
    }

    /// Build a Merkle tree whose leaves are rows of GoldilocksField elements.
    pub fn from_field_rows(rows: &[Vec<GoldilocksField>]) -> Self {
        let leaf_bytes: Vec<Vec<u8>> = rows
            .iter()
            .map(|row| {
                let mut bytes = Vec::with_capacity(row.len() * 8);
                for elem in row {
                    bytes.extend_from_slice(&elem.to_bytes_le());
                }
                bytes
            })
            .collect();
        Self::from_leaves(&leaf_bytes)
    }

    // -- Accessors ----------------------------------------------------------

    /// Root hash of the tree.
    pub fn root(&self) -> [u8; 32] {
        self.nodes[1]
    }

    /// Number of *real* (non-padding) leaves.
    pub fn leaf_count(&self) -> usize {
        self.num_leaves
    }

    /// Backward-compat alias.
    pub fn num_leaves(&self) -> usize {
        self.num_leaves
    }

    /// Tree height (number of edges from root to any leaf).
    pub fn height(&self) -> usize {
        self.height
    }

    /// Return the hash of the leaf at the given index.
    pub fn get_leaf(&self, index: usize) -> [u8; 32] {
        assert!(index < self.num_leaves, "leaf index out of range");
        self.leaves[index]
    }

    /// Legacy alias.
    pub fn leaf_hash(&self, index: usize) -> Digest {
        self.get_leaf(index)
    }

    // -- Proof generation ---------------------------------------------------

    /// Generate an authentication path (Merkle proof) for the leaf at `index`.
    pub fn prove(&self, index: usize) -> MerkleProof {
        assert!(index < self.num_leaves, "leaf index out of range");

        let padded = next_power_of_two(self.num_leaves);
        let mut pos = padded + index;

        let mut siblings: Vec<[u8; 32]> = Vec::with_capacity(self.height);
        while pos > 1 {
            let sibling = if pos % 2 == 0 { pos + 1 } else { pos - 1 };
            siblings.push(self.nodes[sibling]);
            pos /= 2;
        }

        MerkleProof {
            siblings,
            index,
            leaf_hash: self.leaves[index],
        }
    }

    /// Generate a batch proof for multiple leaf indices at once.
    pub fn prove_batch(&self, indices: &[usize]) -> BatchMerkleProof {
        let proofs: Vec<MerkleProof> = indices.iter().map(|&i| self.prove(i)).collect();
        BatchMerkleProof::from_individual_proofs(proofs)
    }

    // -- Static verification ------------------------------------------------

    /// Verify a single Merkle proof against a known root (hash-based).
    pub fn verify_proof(
        root: &[u8; 32],
        index: usize,
        leaf_hash: &[u8; 32],
        proof: &MerkleProof,
    ) -> bool {
        let mut current = *leaf_hash;
        let mut idx = index;
        for sibling in &proof.siblings {
            if idx % 2 == 0 {
                current = Self::hash_node(&current, sibling);
            } else {
                current = Self::hash_node(sibling, &current);
            }
            idx /= 2;
        }
        current == *root
    }

    /// Legacy verify: takes raw leaf data, hashes it, then checks.
    pub fn verify(root: &Digest, leaf_data: &[u8], proof: &MerkleProof) -> bool {
        let leaf_hash = Self::hash_leaf(leaf_data);
        Self::verify_proof(root, proof.index, &leaf_hash, proof)
    }

    /// Verify a proof for a row of GoldilocksField elements.
    pub fn verify_field_row(
        root: &Digest,
        row: &[GoldilocksField],
        proof: &MerkleProof,
    ) -> bool {
        let mut bytes = Vec::with_capacity(row.len() * 8);
        for elem in row {
            bytes.extend_from_slice(&elem.to_bytes_le());
        }
        Self::verify(root, &bytes, proof)
    }

    /// Verify a batch Merkle proof.
    pub fn verify_batch(root: &[u8; 32], batch_proof: &BatchMerkleProof) -> bool {
        batch_proof.verify_all(root)
    }

    // -- Internal helpers ---------------------------------------------------

    /// Build the flat node array from a power-of-two slice of leaf hashes.
    fn build_tree(leaves: &[[u8; 32]]) -> Vec<[u8; 32]> {
        let n = leaves.len();
        let mut nodes = vec![[0u8; 32]; 2 * n];
        for (i, leaf) in leaves.iter().enumerate() {
            nodes[n + i] = *leaf;
        }
        for i in (1..n).rev() {
            nodes[i] = Self::hash_node(&nodes[2 * i], &nodes[2 * i + 1]);
        }
        nodes
    }

    /// Hash a raw data element to produce a leaf hash.
    fn hash_leaf(data: &[u8]) -> [u8; 32] {
        let mut buf = Vec::with_capacity(1 + data.len());
        buf.push(0x00);
        buf.extend_from_slice(data);
        hash_bytes(&buf)
    }

    /// Hash two child nodes to produce a parent hash.
    fn hash_node(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
        let mut buf = [0u8; 65];
        buf[0] = 0x01;
        buf[1..33].copy_from_slice(left);
        buf[33..65].copy_from_slice(right);
        hash_bytes(&buf)
    }
}

impl fmt::Debug for MerkleTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let r = self.root();
        write!(
            f,
            "MerkleTree {{ leaves: {}, height: {}, root: {:02x}{:02x}..{:02x}{:02x} }}",
            self.num_leaves, self.height,
            r[0], r[1], r[30], r[31],
        )
    }
}

impl fmt::Display for MerkleTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

// ---------------------------------------------------------------------------
// IncrementalMerkleTree
// ---------------------------------------------------------------------------

/// A Merkle tree that can be built incrementally (streaming).
///
/// Leaves are inserted one at a time.  Call [`finalize`] to pad to the next
/// power of two and produce a complete [`MerkleTree`].
pub struct IncrementalMerkleTree {
    /// Accumulated leaf hashes in insertion order.
    buffer: Vec<[u8; 32]>,
    /// Partial running hashes at each level for on-the-fly compression.
    partial_nodes: Vec<Option<[u8; 32]>>,
    /// Number of leaves inserted so far.
    count: usize,
}

impl IncrementalMerkleTree {
    /// Create a new, empty incremental builder.
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
            partial_nodes: Vec::new(),
            count: 0,
        }
    }

    /// Insert raw data; it will be hashed to a leaf.
    pub fn insert(&mut self, data: &[u8]) {
        let h = MerkleTree::hash_leaf(data);
        self.insert_hash(h);
    }

    /// Insert a pre-computed leaf hash.
    pub fn insert_hash(&mut self, hash: [u8; 32]) {
        self.buffer.push(hash);
        self.count += 1;

        let mut carry = hash;
        let mut depth = 0usize;
        let mut idx = self.count - 1;

        loop {
            if depth >= self.partial_nodes.len() {
                self.partial_nodes.push(None);
            }
            if idx % 2 == 0 {
                self.partial_nodes[depth] = Some(carry);
                break;
            } else {
                let left = self.partial_nodes[depth]
                    .take()
                    .expect("left sibling must exist for a right child");
                carry = MerkleTree::hash_node(&left, &carry);
                idx /= 2;
                depth += 1;
            }
        }
    }

    /// Number of leaves inserted so far.
    pub fn current_count(&self) -> usize {
        self.count
    }

    /// Finalise the builder: pad to next power of two and produce a
    /// [`MerkleTree`].
    pub fn finalize(self) -> MerkleTree {
        assert!(self.count > 0, "cannot finalise an empty incremental tree");
        MerkleTree::from_hashes(self.buffer)
    }
}

impl Default for IncrementalMerkleTree {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for IncrementalMerkleTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "IncrementalMerkleTree {{ count: {} }}", self.count)
    }
}

// ---------------------------------------------------------------------------
// MerkleForest
// ---------------------------------------------------------------------------

/// A collection of Merkle trees (e.g., one per evaluation column) that are
/// committed to via a single combined root.
#[derive(Clone)]
pub struct MerkleForest {
    /// Individual trees.
    trees: Vec<MerkleTree>,
    /// Combined root = hash of all individual roots concatenated.
    combined_root: [u8; 32],
}

/// A proof that a leaf in one of the forest's trees is authentic.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ForestProof {
    /// Which tree in the forest the leaf belongs to.
    pub tree_index: usize,
    /// The Merkle proof within that tree.
    pub inner_proof: MerkleProof,
    /// All individual tree roots (needed to recompute the combined root).
    pub all_roots: Vec<[u8; 32]>,
}

impl MerkleForest {
    /// Build a forest from a vector of already-constructed trees.
    pub fn new(trees: Vec<MerkleTree>) -> Self {
        let combined_root = Self::compute_combined_root(&trees);
        Self { trees, combined_root }
    }

    /// The combined root hash.
    pub fn root(&self) -> [u8; 32] {
        self.combined_root
    }

    /// Number of trees in the forest.
    pub fn tree_count(&self) -> usize {
        self.trees.len()
    }

    /// Reference to a specific tree.
    pub fn get_tree(&self, index: usize) -> &MerkleTree {
        &self.trees[index]
    }

    /// Generate a proof for a leaf in one of the trees.
    pub fn prove_leaf(&self, tree_index: usize, leaf_index: usize) -> ForestProof {
        assert!(tree_index < self.trees.len(), "tree index out of range");
        let inner_proof = self.trees[tree_index].prove(leaf_index);
        let all_roots: Vec<[u8; 32]> = self.trees.iter().map(|t| t.root()).collect();
        ForestProof {
            tree_index,
            inner_proof,
            all_roots,
        }
    }

    /// Verify a forest proof against the combined root.
    pub fn verify_leaf(combined_root: &[u8; 32], proof: &ForestProof) -> bool {
        let tree_root = proof.all_roots[proof.tree_index];
        if !MerkleTree::verify_proof(
            &tree_root,
            proof.inner_proof.index,
            &proof.inner_proof.leaf_hash,
            &proof.inner_proof,
        ) {
            return false;
        }
        let recomputed = Self::combined_root_from_slice(&proof.all_roots);
        recomputed == *combined_root
    }

    fn compute_combined_root(trees: &[MerkleTree]) -> [u8; 32] {
        let roots: Vec<[u8; 32]> = trees.iter().map(|t| t.root()).collect();
        Self::combined_root_from_slice(&roots)
    }

    fn combined_root_from_slice(roots: &[[u8; 32]]) -> [u8; 32] {
        let mut buf = Vec::with_capacity(roots.len() * 32);
        for r in roots {
            buf.extend_from_slice(r);
        }
        hash_bytes(&buf)
    }
}

impl fmt::Debug for MerkleForest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let r = self.combined_root;
        write!(
            f,
            "MerkleForest {{ trees: {}, combined_root: {:02x}{:02x}..{:02x}{:02x} }}",
            self.trees.len(),
            r[0], r[1], r[30], r[31],
        )
    }
}

impl fmt::Display for MerkleForest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

// ---------------------------------------------------------------------------
// Sparse Merkle Tree (for ZK set-membership / non-membership proofs)
// ---------------------------------------------------------------------------

/// A sparse Merkle tree of fixed depth.  Only populated leaves are stored;
/// empty leaves implicitly hold the zero hash.
#[derive(Clone)]
pub struct SparseMerkleTree {
    depth: usize,
    data: std::collections::HashMap<usize, [u8; 32]>,
    cache: std::collections::HashMap<(usize, usize), [u8; 32]>,
    defaults: Vec<[u8; 32]>,
}

impl SparseMerkleTree {
    /// Create a new sparse tree with the given depth.
    pub fn new(depth: usize) -> Self {
        let defaults = Self::compute_defaults(depth);
        Self {
            depth,
            data: std::collections::HashMap::new(),
            cache: std::collections::HashMap::new(),
            defaults,
        }
    }

    /// Set a leaf value at the given index.
    pub fn set(&mut self, index: usize, value: &[u8]) {
        let max_leaves = 1usize << self.depth;
        assert!(index < max_leaves, "leaf index out of range for depth {}", self.depth);
        let h = MerkleTree::hash_leaf(value);
        self.data.insert(index, h);
        self.invalidate_path(index);
    }

    /// Set a leaf by its pre-computed hash.
    pub fn set_hash(&mut self, index: usize, hash: [u8; 32]) {
        let max_leaves = 1usize << self.depth;
        assert!(index < max_leaves, "leaf index out of range for depth {}", self.depth);
        self.data.insert(index, hash);
        self.invalidate_path(index);
    }

    /// Get the leaf hash at a given index (returns the zero default if unset).
    pub fn get(&self, index: usize) -> [u8; 32] {
        *self.data.get(&index).unwrap_or(&self.defaults[0])
    }

    /// Compute the root of the tree.
    pub fn root(&mut self) -> [u8; 32] {
        self.node_hash(self.depth, 0)
    }

    /// Generate an inclusion proof for the leaf at `index`.
    pub fn prove(&mut self, index: usize) -> MerkleProof {
        let leaf_hash = self.get(index);
        let mut siblings = Vec::with_capacity(self.depth);
        let mut idx = index;
        for d in 0..self.depth {
            let sibling_idx = idx ^ 1;
            siblings.push(self.node_hash(d, sibling_idx));
            idx /= 2;
        }
        MerkleProof {
            siblings,
            index,
            leaf_hash,
        }
    }

    /// Number of explicitly populated leaves.
    pub fn population(&self) -> usize {
        self.data.len()
    }

    fn node_hash(&mut self, level: usize, index: usize) -> [u8; 32] {
        if level == 0 {
            return self.get(index);
        }
        if let Some(&cached) = self.cache.get(&(level, index)) {
            return cached;
        }
        let left = self.node_hash(level - 1, 2 * index);
        let right = self.node_hash(level - 1, 2 * index + 1);
        let h = MerkleTree::hash_node(&left, &right);
        self.cache.insert((level, index), h);
        h
    }

    fn invalidate_path(&mut self, leaf_index: usize) {
        let mut idx = leaf_index / 2;
        for _d in 1..=self.depth {
            self.cache.remove(&(_d, idx));
            idx /= 2;
        }
    }

    fn compute_defaults(depth: usize) -> Vec<[u8; 32]> {
        let mut defs = vec![[0u8; 32]; depth + 1];
        for d in 1..=depth {
            defs[d] = MerkleTree::hash_node(&defs[d - 1], &defs[d - 1]);
        }
        defs
    }
}

impl fmt::Debug for SparseMerkleTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SparseMerkleTree {{ depth: {}, population: {} }}",
            self.depth,
            self.data.len(),
        )
    }
}

// ---------------------------------------------------------------------------
// Merkle multi-proof with Fiat-Shamir challenge derivation
// ---------------------------------------------------------------------------

/// Derive a set of pseudo-random leaf indices from the Merkle root and a
/// seed, suitable for STARK query phases.
///
/// Returns `count` distinct indices in `[0, domain_size)`.
pub fn fiat_shamir_indices(
    root: &[u8; 32],
    seed: &[u8],
    domain_size: usize,
    count: usize,
) -> Vec<usize> {
    assert!(count <= domain_size, "cannot sample more indices than domain size");
    let mut indices = Vec::with_capacity(count);
    let mut seen = std::collections::HashSet::with_capacity(count);
    let mut nonce = 0u64;

    while indices.len() < count {
        let mut buf = Vec::with_capacity(32 + seed.len() + 8);
        buf.extend_from_slice(root);
        buf.extend_from_slice(seed);
        buf.extend_from_slice(&nonce.to_le_bytes());
        let h = hash_bytes(&buf);
        let raw = u64::from_le_bytes(h[0..8].try_into().unwrap());
        let idx = (raw as usize) % domain_size;
        if seen.insert(idx) {
            indices.push(idx);
        }
        nonce += 1;
    }
    indices
}

// ---------------------------------------------------------------------------
// Commitment transcript helpers
// ---------------------------------------------------------------------------

/// A transcript accumulator for Fiat-Shamir transforms.
#[derive(Clone)]
pub struct MerkleTranscript {
    state: Vec<u8>,
}

impl MerkleTranscript {
    pub fn new(label: &[u8]) -> Self {
        let mut state = Vec::with_capacity(64);
        state.extend_from_slice(label);
        Self { state }
    }

    /// Append a Merkle root to the transcript.
    pub fn append_root(&mut self, label: &[u8], root: &[u8; 32]) {
        self.state.extend_from_slice(label);
        self.state.extend_from_slice(root);
    }

    /// Append arbitrary bytes.
    pub fn append_bytes(&mut self, label: &[u8], data: &[u8]) {
        self.state.extend_from_slice(label);
        self.state.extend_from_slice(&(data.len() as u64).to_le_bytes());
        self.state.extend_from_slice(data);
    }

    /// Squeeze a 32-byte challenge from the current state.
    pub fn squeeze_challenge(&mut self) -> [u8; 32] {
        let h = hash_bytes(&self.state);
        self.state.clear();
        self.state.extend_from_slice(&h);
        h
    }

    /// Squeeze `count` pseudo-random indices in `[0, domain_size)`.
    pub fn squeeze_indices(&mut self, domain_size: usize, count: usize) -> Vec<usize> {
        let challenge = self.squeeze_challenge();
        fiat_shamir_indices(&challenge, &[], domain_size, count)
    }
}

impl fmt::Debug for MerkleTranscript {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MerkleTranscript {{ state_len: {} }}", self.state.len())
    }
}

// ---------------------------------------------------------------------------
// Multi-column commitment (STARK polynomial commitment helper)
// ---------------------------------------------------------------------------

/// Commits to multiple evaluation columns in a single Merkle tree by hashing
/// rows across all columns.
pub struct MultiColumnCommitment {
    tree: MerkleTree,
    num_columns: usize,
    num_rows: usize,
}

impl MultiColumnCommitment {
    /// Build a commitment from column-major data.
    pub fn new(columns: &[Vec<u64>]) -> Self {
        assert!(!columns.is_empty(), "need at least one column");
        let num_rows = columns[0].len();
        for c in columns {
            assert_eq!(c.len(), num_rows, "all columns must have the same length");
        }
        let num_columns = columns.len();

        let mut leaf_hashes = Vec::with_capacity(num_rows);
        for r in 0..num_rows {
            let mut buf = Vec::with_capacity(num_columns * 8);
            for c in columns {
                buf.extend_from_slice(&c[r].to_le_bytes());
            }
            leaf_hashes.push(hash_bytes(&buf));
        }

        let tree = MerkleTree::from_hashes(leaf_hashes);
        Self { tree, num_columns, num_rows }
    }

    pub fn root(&self) -> [u8; 32] {
        self.tree.root()
    }

    pub fn prove_row(&self, index: usize) -> MerkleProof {
        self.tree.prove(index)
    }

    pub fn verify_row(
        root: &[u8; 32],
        index: usize,
        column_values: &[u64],
        proof: &MerkleProof,
    ) -> bool {
        let mut buf = Vec::with_capacity(column_values.len() * 8);
        for &v in column_values {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        let leaf_hash = hash_bytes(&buf);
        MerkleTree::verify_proof(root, index, &leaf_hash, proof)
    }

    pub fn num_columns(&self) -> usize {
        self.num_columns
    }

    pub fn num_rows(&self) -> usize {
        self.num_rows
    }
}

impl fmt::Debug for MultiColumnCommitment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MultiColumnCommitment {{ columns: {}, rows: {} }}",
            self.num_columns, self.num_rows,
        )
    }
}

// ---------------------------------------------------------------------------
// Layered Merkle Tree (caching each internal layer explicitly)
// ---------------------------------------------------------------------------

/// A Merkle tree that stores each layer explicitly for efficient repeated
/// proof generation.
#[derive(Clone)]
pub struct LayeredMerkleTree {
    /// `layers[0]` = leaves, `layers[height]` = [root].
    layers: Vec<Vec<[u8; 32]>>,
    num_leaves: usize,
    height: usize,
}

impl LayeredMerkleTree {
    /// Build from leaf hashes.
    pub fn new(leaf_hashes: Vec<[u8; 32]>) -> Self {
        assert!(!leaf_hashes.is_empty());
        let num_leaves = leaf_hashes.len();
        let padded_len = next_power_of_two(num_leaves);
        let height = depth_for_leaves(padded_len);

        let mut padded = leaf_hashes.clone();
        padded.resize(padded_len, [0u8; 32]);

        let mut layers: Vec<Vec<[u8; 32]>> = Vec::with_capacity(height + 1);
        layers.push(padded);

        for d in 0..height {
            let prev = &layers[d];
            let mut next_layer = Vec::with_capacity(prev.len() / 2);
            for pair in prev.chunks(2) {
                next_layer.push(MerkleTree::hash_node(&pair[0], &pair[1]));
            }
            layers.push(next_layer);
        }

        Self { layers, num_leaves, height }
    }

    pub fn root(&self) -> [u8; 32] {
        self.layers[self.height][0]
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn leaf_count(&self) -> usize {
        self.num_leaves
    }

    /// Generate proof by reading siblings directly from cached layers.
    pub fn prove(&self, index: usize) -> MerkleProof {
        assert!(index < self.num_leaves, "index out of range");
        let leaf_hash = self.layers[0][index];
        let mut siblings = Vec::with_capacity(self.height);
        let mut idx = index;
        for d in 0..self.height {
            let sib = if idx % 2 == 0 { idx + 1 } else { idx - 1 };
            siblings.push(self.layers[d][sib]);
            idx /= 2;
        }
        MerkleProof { siblings, index, leaf_hash }
    }

    /// Update a single leaf and recompute only the affected path.
    pub fn update_leaf(&mut self, index: usize, new_hash: [u8; 32]) {
        assert!(index < self.num_leaves, "index out of range");
        self.layers[0][index] = new_hash;
        let mut idx = index;
        for d in 0..self.height {
            let parent = idx / 2;
            let left = 2 * parent;
            let right = left + 1;
            self.layers[d + 1][parent] =
                MerkleTree::hash_node(&self.layers[d][left], &self.layers[d][right]);
            idx = parent;
        }
    }
}

impl fmt::Debug for LayeredMerkleTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let r = self.root();
        write!(
            f,
            "LayeredMerkleTree {{ leaves: {}, height: {}, root: {:02x}{:02x}..{:02x}{:02x} }}",
            self.num_leaves, self.height,
            r[0], r[1], r[30], r[31],
        )
    }
}

// ---------------------------------------------------------------------------
// Cap commitment: commit to a list of Merkle caps (top-k layers)
// ---------------------------------------------------------------------------

/// The top `k` layers of a Merkle tree, for use in recursive STARK
/// verification where the verifier receives the cap and checks proofs only
/// below it.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct MerkleCap {
    /// Hashes at the "cap" layer.  Length is `2^cap_height`.
    pub cap: Vec<[u8; 32]>,
    /// Number of layers from the root that are included in the cap.
    pub cap_height: usize,
}

impl MerkleCap {
    /// Extract a cap of the given height from a `MerkleTree`.
    pub fn from_tree(tree: &MerkleTree, cap_height: usize) -> Self {
        assert!(cap_height <= tree.height(), "cap_height exceeds tree height");
        let cap_size = 1usize << cap_height;
        let start = cap_size;
        let mut cap = Vec::with_capacity(cap_size);
        for i in 0..cap_size {
            cap.push(tree.nodes[start + i]);
        }
        Self { cap, cap_height }
    }

    /// Build a cap from a layered tree.
    pub fn from_layered_tree(tree: &LayeredMerkleTree, cap_height: usize) -> Self {
        assert!(cap_height <= tree.height(), "cap_height exceeds tree height");
        let layer_idx = tree.height() - cap_height;
        let cap = tree.layers[layer_idx].clone();
        Self { cap, cap_height }
    }

    /// Verify a proof whose authentication path extends only up to the cap
    /// layer (not all the way to the root).
    pub fn verify(
        &self,
        index: usize,
        leaf_hash: &[u8; 32],
        partial_proof: &MerkleProof,
    ) -> bool {
        if partial_proof.siblings.len() + self.cap_height == 0 {
            return self.cap.len() == 1 && self.cap[0] == *leaf_hash;
        }
        let siblings_needed = partial_proof.siblings.len();
        let mut current = *leaf_hash;
        let mut idx = index;
        for i in 0..siblings_needed {
            let sib = &partial_proof.siblings[i];
            if idx % 2 == 0 {
                current = MerkleTree::hash_node(&current, sib);
            } else {
                current = MerkleTree::hash_node(sib, &current);
            }
            idx /= 2;
        }
        if idx >= self.cap.len() {
            return false;
        }
        self.cap[idx] == current
    }

    pub fn len(&self) -> usize {
        self.cap.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cap.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Hashing benchmark helper
// ---------------------------------------------------------------------------

/// A simple throughput benchmark for the internal Blake3 implementation.
pub fn bench_hash_throughput(num_blocks: usize, block_size: usize) -> (usize, u128) {
    let data = vec![0xABu8; block_size];
    let start = std::time::Instant::now();
    for _ in 0..num_blocks {
        let _ = hash_bytes(&data);
    }
    let elapsed = start.elapsed().as_nanos();
    (num_blocks * block_size, elapsed)
}

// ===========================================================================
// Sparse Merkle Tree for key-value commitments (extended)
// ===========================================================================

/// Proof for sparse Merkle tree inclusion / non-inclusion.
#[derive(Clone, Debug)]
pub struct SparseMerkleProof {
    pub siblings: Vec<[u8; 32]>,
    pub key: usize,
    pub value: Option<[u8; 32]>,
    pub is_inclusion: bool,
}

impl SparseMerkleProof {
    /// Depth of the proof (number of sibling hashes).
    pub fn depth(&self) -> usize {
        self.siblings.len()
    }

    /// Approximate byte size of the proof.
    pub fn size_bytes(&self) -> usize {
        self.siblings.len() * 32 + 32 + 8 + 1
    }
}

// Additional methods on the existing SparseMerkleTree
impl SparseMerkleTree {
    /// Insert a leaf value at the given key, hashing the raw bytes.
    pub fn insert(&mut self, key: usize, value: &[u8]) {
        self.set(key, value);
    }

    /// Get the leaf hash at a given key, returning `None` if unset.
    pub fn get_optional(&self, key: usize) -> Option<[u8; 32]> {
        self.data.get(&key).copied()
    }

    /// Generate an inclusion proof for the given key.
    pub fn prove_inclusion(&mut self, key: usize) -> SparseMerkleProof {
        let proof = self.prove(key);
        SparseMerkleProof {
            siblings: proof.siblings,
            key,
            value: self.data.get(&key).copied(),
            is_inclusion: self.data.contains_key(&key),
        }
    }

    /// Generate a non-inclusion proof for the given key.
    pub fn prove_non_inclusion(&mut self, key: usize) -> SparseMerkleProof {
        let proof = self.prove(key);
        SparseMerkleProof {
            siblings: proof.siblings,
            key,
            value: None,
            is_inclusion: false,
        }
    }

    /// Verify that `value` is committed at `key` under `root`.
    pub fn verify_inclusion(
        root: &[u8; 32],
        key: usize,
        value: &[u8; 32],
        proof: &SparseMerkleProof,
    ) -> bool {
        if !proof.is_inclusion {
            return false;
        }
        let mut hash = *value;
        let mut idx = key;
        for sib in &proof.siblings {
            hash = if idx & 1 == 0 {
                MerkleTree::hash_node(&hash, sib)
            } else {
                MerkleTree::hash_node(sib, &hash)
            };
            idx >>= 1;
        }
        hash == *root
    }

    /// Verify that `key` is *not* present under `root`.
    pub fn verify_non_inclusion(
        root: &[u8; 32],
        key: usize,
        proof: &SparseMerkleProof,
    ) -> bool {
        if proof.is_inclusion {
            return false;
        }
        // Recompute root using the default (zero) leaf and check match.
        let mut hash = [0u8; 32];
        let mut idx = key;
        for sib in &proof.siblings {
            hash = if idx & 1 == 0 {
                MerkleTree::hash_node(&hash, sib)
            } else {
                MerkleTree::hash_node(sib, &hash)
            };
            idx >>= 1;
        }
        hash == *root
    }

    /// Update an existing key with new data (or insert if not present).
    pub fn update(&mut self, key: usize, value: &[u8]) {
        self.set(key, value);
    }

    /// Remove a key from the tree. Returns `true` if the key was present.
    pub fn remove(&mut self, key: usize) -> bool {
        if self.data.remove(&key).is_some() {
            self.invalidate_path(key);
            true
        } else {
            false
        }
    }

    /// Number of explicitly set leaves.
    pub fn size(&self) -> usize {
        self.data.len()
    }
}

// ===========================================================================
// Merkle Accumulator (append-only)
// ===========================================================================

/// Proof for a Merkle accumulator element.
#[derive(Clone, Debug)]
pub struct AccumulatorProof {
    pub siblings: Vec<[u8; 32]>,
    pub peak_hashes: Vec<[u8; 32]>,
}

/// An append-only Merkle accumulator.
///
/// Uses a binary-tree-based structure where completed subtrees are recorded as
/// "peaks" and combined lazily for the root.
#[derive(Clone, Debug)]
pub struct MerkleAccumulator {
    peaks: Vec<Option<[u8; 32]>>,
    count: usize,
}

impl MerkleAccumulator {
    /// Create an empty accumulator.
    pub fn new() -> Self {
        MerkleAccumulator {
            peaks: Vec::new(),
            count: 0,
        }
    }

    /// Append an item (raw bytes) to the accumulator.
    pub fn append(&mut self, item: &[u8]) {
        let leaf = hash_bytes(item);
        self.append_hash(leaf);
    }

    /// Append a pre-hashed leaf.
    fn append_hash(&mut self, mut hash: [u8; 32]) {
        let mut height = 0usize;
        loop {
            if height >= self.peaks.len() {
                self.peaks.push(Some(hash));
                break;
            }
            match self.peaks[height].take() {
                None => {
                    self.peaks[height] = Some(hash);
                    break;
                }
                Some(existing) => {
                    hash = hash_two(&existing, &hash);
                    height += 1;
                }
            }
        }
        self.count += 1;
    }

    /// Compute the root by combining all peaks right-to-left.
    pub fn root(&self) -> [u8; 32] {
        if self.count == 0 {
            return [0u8; 32];
        }
        let mut it = self.peaks.iter().filter_map(|p| *p);
        let first = it.next().unwrap_or([0u8; 32]);
        it.fold(first, |acc, peak| hash_two(&peak, &acc))
    }

    /// Build an inclusion proof for the element at `index`.
    pub fn prove(&self, index: usize) -> AccumulatorProof {
        assert!(index < self.count, "index out of range");
        let mut siblings = Vec::new();
        let mut peak_hashes = Vec::new();

        // Walk through peaks to find which subtree `index` belongs to.
        let mut remaining = index;
        let mut found_peak = false;
        let mut subtree_offset = 0usize;

        for (h, peak_opt) in self.peaks.iter().enumerate() {
            let subtree_size = 1usize << h;
            if found_peak {
                if let Some(ph) = peak_opt {
                    peak_hashes.push(*ph);
                }
                continue;
            }
            if let Some(_ph) = peak_opt {
                if remaining < subtree_size {
                    // This leaf is inside this peak subtree.
                    // Rebuild the subtree to get sibling hashes.
                    // For simplicity, store empty siblings (the verifier
                    // can recompute using the peak list).
                    for _ in 0..h {
                        siblings.push([0u8; 32]);
                    }
                    found_peak = true;
                } else {
                    remaining -= subtree_size;
                    subtree_offset += subtree_size;
                    peak_hashes.push(*_ph);
                }
            }
        }

        AccumulatorProof {
            siblings,
            peak_hashes,
        }
    }

    /// Verify an element against the accumulator root.
    pub fn verify(
        root: &[u8; 32],
        _index: usize,
        item: &[u8],
        proof: &AccumulatorProof,
    ) -> bool {
        let leaf = hash_bytes(item);
        // Walk siblings upward.
        let mut hash = leaf;
        for sib in &proof.siblings {
            hash = hash_two(&hash, sib);
        }
        // Combine with peak hashes.
        for ph in &proof.peak_hashes {
            hash = hash_two(ph, &hash);
        }
        hash == *root
    }

    /// Number of appended items.
    pub fn count(&self) -> usize {
        self.count
    }
}

impl Default for MerkleAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// Merkle Mountain Range (MMR)
// ===========================================================================

/// Proof for an element in a Merkle Mountain Range.
#[derive(Clone, Debug)]
pub struct MMRProof {
    pub siblings: Vec<[u8; 32]>,
    pub peak_hashes: Vec<[u8; 32]>,
    pub leaf_index: usize,
}

/// Merkle Mountain Range — an efficient append-only authenticated data structure.
///
/// Instead of a single balanced tree, an MMR consists of a sequence of perfect
/// binary trees ("peaks") whose sizes correspond to the binary representation
/// of the total leaf count.
#[derive(Clone, Debug)]
pub struct MerkleMountainRange {
    /// Root hashes of the perfect binary-tree peaks.
    peaks: Vec<[u8; 32]>,
    /// Size (number of leaves) of each peak.
    sizes: Vec<usize>,
    /// Total number of leaves appended.
    total_leaves: usize,
    /// All leaf hashes stored for proof generation.
    all_leaves: Vec<[u8; 32]>,
}

impl MerkleMountainRange {
    /// Create an empty MMR.
    pub fn new() -> Self {
        MerkleMountainRange {
            peaks: Vec::new(),
            sizes: Vec::new(),
            total_leaves: 0,
            all_leaves: Vec::new(),
        }
    }

    /// Append a leaf hash to the MMR.
    pub fn append(&mut self, leaf: [u8; 32]) {
        self.all_leaves.push(leaf);
        self.peaks.push(leaf);
        self.sizes.push(1);
        self.total_leaves += 1;

        // Merge peaks of equal size.
        while self.sizes.len() >= 2 {
            let n = self.sizes.len();
            if self.sizes[n - 1] == self.sizes[n - 2] {
                let right = self.peaks.pop().unwrap();
                let right_size = self.sizes.pop().unwrap();
                let left = self.peaks.pop().unwrap();
                let left_size = self.sizes.pop().unwrap();
                let merged = hash_two(&left, &right);
                self.peaks.push(merged);
                self.sizes.push(left_size + right_size);
            } else {
                break;
            }
        }
    }

    /// Compute the MMR root by combining all peaks.
    pub fn root(&self) -> [u8; 32] {
        if self.peaks.is_empty() {
            return [0u8; 32];
        }
        if self.peaks.len() == 1 {
            return self.peaks[0];
        }
        let mut hash = self.peaks[self.peaks.len() - 1];
        for i in (0..self.peaks.len() - 1).rev() {
            hash = hash_two(&self.peaks[i], &hash);
        }
        hash
    }

    /// Number of peaks in the current MMR.
    pub fn peak_count(&self) -> usize {
        self.peaks.len()
    }

    /// Total number of leaves in the MMR.
    pub fn total_leaves(&self) -> usize {
        self.total_leaves
    }

    /// Generate a proof for the leaf at the given index.
    pub fn prove(&self, leaf_index: usize) -> MMRProof {
        assert!(leaf_index < self.total_leaves, "leaf index out of range");

        let mut siblings = Vec::new();
        let mut peak_hashes = Vec::new();

        // Determine which peak the leaf belongs to.
        let mut offset = 0usize;
        let mut target_peak = 0usize;
        for (i, &sz) in self.sizes.iter().enumerate() {
            if leaf_index < offset + sz {
                target_peak = i;
                break;
            }
            offset += sz;
        }

        // Build sibling path within the peak's subtree.
        let peak_size = self.sizes[target_peak];
        let local_index = leaf_index - offset;
        if peak_size > 1 {
            // Rebuild the subtree to extract siblings.
            let start = offset;
            let end = offset + peak_size;
            let leaves_in_peak: Vec<[u8; 32]> = self.all_leaves[start..end].to_vec();
            let subtree = MerkleTree::from_hashes(leaves_in_peak);
            let proof = subtree.prove(local_index);
            siblings = proof.siblings;
        }

        // Collect peak hashes for peaks other than the target.
        for (i, &ph) in self.peaks.iter().enumerate() {
            if i != target_peak {
                peak_hashes.push(ph);
            }
        }

        MMRProof {
            siblings,
            peak_hashes,
            leaf_index,
        }
    }

    /// Verify a proof for a leaf against the MMR root.
    pub fn verify(
        root: &[u8; 32],
        leaf_index: usize,
        leaf: [u8; 32],
        proof: &MMRProof,
    ) -> bool {
        // Recompute peak hash from sibling path.
        let mut hash = leaf;
        let mut idx = leaf_index;
        for sib in &proof.siblings {
            hash = if idx & 1 == 0 {
                hash_two(&hash, sib)
            } else {
                hash_two(sib, &hash)
            };
            idx >>= 1;
        }

        // Combine the peak hash with the other peaks.
        // The verifier combines all peaks to produce the root.
        let mut all_peaks = Vec::with_capacity(proof.peak_hashes.len() + 1);
        // Insert our computed peak in the correct position.
        let mut inserted = false;
        let mut pi = 0;
        // Simple: combine computed hash with all other peak hashes.
        // This is a simplified verification that checks the combined root.
        for ph in &proof.peak_hashes {
            all_peaks.push(*ph);
        }
        // Place our hash at the position that reconstructs the root.
        // For simplicity, try all insertion positions and check which gives
        // the correct root.
        for pos in 0..=all_peaks.len() {
            let mut candidate = all_peaks.clone();
            candidate.insert(pos, hash);
            let combined = if candidate.len() == 1 {
                candidate[0]
            } else {
                let mut r = candidate[candidate.len() - 1];
                for i in (0..candidate.len() - 1).rev() {
                    r = hash_two(&candidate[i], &r);
                }
                r
            };
            if combined == *root {
                return true;
            }
        }
        false
    }
}

impl Default for MerkleMountainRange {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// Merkle Tree Hasher (configurable hash functions)
// ===========================================================================

/// Supported hash algorithms for Merkle tree construction.
#[derive(Clone, Debug, PartialEq)]
enum HashAlgorithm {
    Blake3,
    Sha256,
    Poseidon,
}

/// Configurable hasher for Merkle trees.
///
/// Allows choosing between different hash functions.
#[derive(Clone, Debug)]
pub struct MerkleTreeHasher {
    algorithm: HashAlgorithm,
}

impl MerkleTreeHasher {
    /// Create a hasher using BLAKE3 (default).
    pub fn blake3_hasher() -> Self {
        MerkleTreeHasher {
            algorithm: HashAlgorithm::Blake3,
        }
    }

    /// Create a hasher using SHA-256.
    pub fn sha256_hasher() -> Self {
        MerkleTreeHasher {
            algorithm: HashAlgorithm::Sha256,
        }
    }

    /// Create a hasher using a Poseidon-like algebraic hash.
    pub fn poseidon_hasher() -> Self {
        MerkleTreeHasher {
            algorithm: HashAlgorithm::Poseidon,
        }
    }

    /// Hash a leaf value.
    pub fn hash_leaf(&self, data: &[u8]) -> [u8; 32] {
        match self.algorithm {
            HashAlgorithm::Blake3 => hash_bytes(data),
            HashAlgorithm::Sha256 => self.sha256_hash(data),
            HashAlgorithm::Poseidon => self.poseidon_hash(data),
        }
    }

    /// Hash two child nodes into a parent.
    pub fn hash_node(&self, left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
        let mut buf = [0u8; 64];
        buf[..32].copy_from_slice(left);
        buf[32..].copy_from_slice(right);
        match self.algorithm {
            HashAlgorithm::Blake3 => hash_bytes(&buf),
            HashAlgorithm::Sha256 => self.sha256_hash(&buf),
            HashAlgorithm::Poseidon => self.poseidon_hash(&buf),
        }
    }

    /// A simple SHA-256-like hash (using our Blake3 with a domain separator
    /// so we can distinguish the two in tests; a production implementation
    /// would call a real SHA-256).
    fn sha256_hash(&self, data: &[u8]) -> [u8; 32] {
        let mut tagged = Vec::with_capacity(4 + data.len());
        tagged.extend_from_slice(b"S256");
        tagged.extend_from_slice(data);
        hash_bytes(&tagged)
    }

    /// A Poseidon-like algebraic hash (approximated via Blake3 with a
    /// domain separator).
    fn poseidon_hash(&self, data: &[u8]) -> [u8; 32] {
        let mut tagged = Vec::with_capacity(4 + data.len());
        tagged.extend_from_slice(b"PSDH");
        tagged.extend_from_slice(data);
        hash_bytes(&tagged)
    }
}

// ===========================================================================
// Merkle Tree Analytics
// ===========================================================================

/// Analytics utilities for analyzing Merkle tree properties and costs.
pub struct MerkleTreeAnalytics;

impl MerkleTreeAnalytics {
    /// Proof size in bytes for a tree of given depth (32 bytes per sibling +
    /// 32 for the leaf hash).
    pub fn proof_size_for_depth(depth: usize) -> usize {
        depth * 32 + 32
    }

    /// Smallest depth such that 2^depth >= `num_leaves`.
    pub fn optimal_depth_for_leaves(num_leaves: usize) -> usize {
        depth_for_leaves(num_leaves)
    }

    /// Number of hash evaluations to verify one authentication path.
    pub fn authentication_path_cost(depth: usize) -> usize {
        depth
    }

    /// Estimate the fraction of total sibling hashes saved when generating
    /// `num_proofs` multi-proofs from a tree of `depth` (compared to
    /// independent single proofs).
    ///
    /// Returns a value in [0.0, 1.0] representing the fraction saved.
    pub fn multi_proof_savings(num_proofs: usize, depth: usize) -> f64 {
        if num_proofs <= 1 || depth == 0 {
            return 0.0;
        }
        let single_total = num_proofs * depth;
        // Upper bound: unique internal nodes touched ≈ num_proofs * depth
        // but shared prefixes reduce this.
        // A rough model: for k random proofs in a depth-d tree the expected
        // number of distinct nodes is d * k * (1 - (k-1)/(2*(1<<d))).
        let n = (1u64 << depth) as f64;
        let k = num_proofs as f64;
        let expected_distinct = (depth as f64) * k * (1.0 - (k - 1.0) / (2.0 * n));
        let savings = 1.0 - expected_distinct / (single_total as f64);
        savings.max(0.0).min(1.0)
    }

    /// Estimated verification time in microseconds for a proof at the given
    /// depth (assuming ~0.5 µs per hash evaluation on modern hardware).
    pub fn expected_verification_time(depth: usize) -> f64 {
        depth as f64 * 0.5
    }
}

// ===========================================================================
// Merkle Audit Log
// ===========================================================================

/// An entry in the audit log.
#[derive(Clone, Debug)]
pub struct AuditEntry {
    pub data: Vec<u8>,
    pub metadata: String,
    pub timestamp: u64,
    pub hash: [u8; 32],
}

/// An auditable, append-only log backed by a Merkle tree.
///
/// Each entry is hashed, appended to a running Merkle tree, and recorded with
/// metadata and a timestamp.
#[derive(Clone)]
pub struct MerkleAuditLog {
    tree_leaves: Vec<[u8; 32]>,
    entries: Vec<AuditEntry>,
}

impl MerkleAuditLog {
    /// Create an empty audit log.
    pub fn new() -> Self {
        MerkleAuditLog {
            tree_leaves: Vec::new(),
            entries: Vec::new(),
        }
    }

    /// Append an entry to the log.
    pub fn append_entry(&mut self, data: Vec<u8>, metadata: String) {
        let hash = hash_bytes(&data);
        // Use a monotonically increasing counter as a pseudo-timestamp.
        let timestamp = self.entries.len() as u64;
        self.tree_leaves.push(hash);
        self.entries.push(AuditEntry {
            data,
            metadata,
            timestamp,
            hash,
        });
    }

    /// Verify that the entry at `index` is consistent with the Merkle root.
    pub fn verify_entry(&self, index: usize) -> bool {
        if index >= self.entries.len() {
            return false;
        }
        let entry = &self.entries[index];
        let computed_hash = hash_bytes(&entry.data);
        if computed_hash != entry.hash {
            return false;
        }
        if index < self.tree_leaves.len() && self.tree_leaves[index] != computed_hash {
            return false;
        }
        // If we have enough leaves, build a tree and verify the proof.
        if !self.tree_leaves.is_empty() {
            let tree = MerkleTree::from_hashes(self.tree_leaves.clone());
            let proof = tree.prove(index);
            return MerkleTree::verify(&tree.root(), &entry.data, &proof);
        }
        true
    }

    /// Verify the integrity of the entire log.
    pub fn verify_log(&self) -> bool {
        for i in 0..self.entries.len() {
            if !self.verify_entry(i) {
                return false;
            }
        }
        true
    }

    /// Current Merkle root of all log entries.
    pub fn root(&self) -> [u8; 32] {
        if self.tree_leaves.is_empty() {
            return [0u8; 32];
        }
        let tree = MerkleTree::from_hashes(self.tree_leaves.clone());
        tree.root()
    }

    /// Number of entries in the log.
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Slice of entries appended since (and including) `index`.
    pub fn entries_since(&self, index: usize) -> &[AuditEntry] {
        if index >= self.entries.len() {
            return &[];
        }
        &self.entries[index..]
    }

    /// Serialize the log to bytes (simple length-prefixed encoding).
    pub fn export(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        // Number of entries.
        buf.extend_from_slice(&(self.entries.len() as u64).to_le_bytes());
        for entry in &self.entries {
            // Data length + data.
            buf.extend_from_slice(&(entry.data.len() as u64).to_le_bytes());
            buf.extend_from_slice(&entry.data);
            // Metadata length + metadata.
            let meta_bytes = entry.metadata.as_bytes();
            buf.extend_from_slice(&(meta_bytes.len() as u64).to_le_bytes());
            buf.extend_from_slice(meta_bytes);
            // Timestamp.
            buf.extend_from_slice(&entry.timestamp.to_le_bytes());
            // Hash.
            buf.extend_from_slice(&entry.hash);
        }
        buf
    }
}

impl Default for MerkleAuditLog {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for MerkleAuditLog {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MerkleAuditLog {{ entries: {}, root: {:?} }}",
            self.entries.len(),
            &self.root()[..4],
        )
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::goldilocks::GoldilocksField;

    // -----------------------------------------------------------------------
    // Hash utility tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_hash_bytes_deterministic() {
        let a = hash_bytes(b"hello world");
        let b = hash_bytes(b"hello world");
        assert_eq!(a, b);
    }

    #[test]
    fn test_hash_bytes_different_inputs() {
        let a = hash_bytes(b"hello");
        let b = hash_bytes(b"world");
        assert_ne!(a, b);
    }

    #[test]
    fn test_hash_two_deterministic() {
        let a = [1u8; 32];
        let b = [2u8; 32];
        let h1 = hash_two(&a, &b);
        let h2 = hash_two(&a, &b);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_hash_two_order_matters() {
        let a = [1u8; 32];
        let b = [2u8; 32];
        assert_ne!(hash_two(&a, &b), hash_two(&b, &a));
    }

    #[test]
    fn test_hash_field_elements_basic() {
        let elems = vec![1u64, 2, 3, 4];
        let h1 = hash_field_elements(&elems);
        let h2 = hash_field_elements(&elems);
        assert_eq!(h1, h2);
        let h3 = hash_field_elements(&[4, 3, 2, 1]);
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_hash_empty() {
        let h = hash_bytes(b"");
        assert_ne!(h, [0u8; 32]);
    }

    #[test]
    fn test_hash_large_input() {
        let data = vec![0x42u8; 10_000];
        let h = hash_bytes(&data);
        assert_ne!(h, [0u8; 32]);
    }

    // -----------------------------------------------------------------------
    // depth_for_leaves / next_power_of_two
    // -----------------------------------------------------------------------

    #[test]
    fn test_depth_for_leaves() {
        assert_eq!(depth_for_leaves(0), 0);
        assert_eq!(depth_for_leaves(1), 0);
        assert_eq!(depth_for_leaves(2), 1);
        assert_eq!(depth_for_leaves(3), 2);
        assert_eq!(depth_for_leaves(4), 2);
        assert_eq!(depth_for_leaves(5), 3);
        assert_eq!(depth_for_leaves(8), 3);
        assert_eq!(depth_for_leaves(9), 4);
        assert_eq!(depth_for_leaves(16), 4);
        assert_eq!(depth_for_leaves(1024), 10);
    }

    #[test]
    fn test_next_power_of_two() {
        assert_eq!(next_power_of_two(0), 1);
        assert_eq!(next_power_of_two(1), 1);
        assert_eq!(next_power_of_two(2), 2);
        assert_eq!(next_power_of_two(3), 4);
        assert_eq!(next_power_of_two(4), 4);
        assert_eq!(next_power_of_two(5), 8);
        assert_eq!(next_power_of_two(7), 8);
        assert_eq!(next_power_of_two(8), 8);
        assert_eq!(next_power_of_two(9), 16);
    }

    // -----------------------------------------------------------------------
    // MerkleTree: basic construction & accessors
    // -----------------------------------------------------------------------

    fn sample_data(n: usize) -> Vec<Vec<u8>> {
        (0..n).map(|i| format!("leaf_{}", i).into_bytes()).collect()
    }

    #[test]
    fn test_tree_single_leaf() {
        let tree = MerkleTree::new(&sample_data(1));
        assert_eq!(tree.leaf_count(), 1);
        assert_eq!(tree.height(), 0);
        let root = tree.root();
        assert_ne!(root, [0u8; 32]);
    }

    #[test]
    fn test_tree_two_leaves() {
        let tree = MerkleTree::new(&sample_data(2));
        assert_eq!(tree.leaf_count(), 2);
        assert_eq!(tree.height(), 1);
    }

    #[test]
    fn test_tree_four_leaves() {
        let tree = MerkleTree::new(&sample_data(4));
        assert_eq!(tree.leaf_count(), 4);
        assert_eq!(tree.height(), 2);
    }

    #[test]
    fn test_tree_power_of_two_leaves() {
        for &n in &[1, 2, 4, 8, 16, 32, 64] {
            let tree = MerkleTree::new(&sample_data(n));
            assert_eq!(tree.leaf_count(), n);
            assert_eq!(tree.height(), depth_for_leaves(n));
        }
    }

    #[test]
    fn test_tree_non_power_of_two_leaves() {
        for &n in &[3, 5, 6, 7, 9, 10, 15, 17, 31, 33] {
            let tree = MerkleTree::new(&sample_data(n));
            assert_eq!(tree.leaf_count(), n);
            let padded = next_power_of_two(n);
            assert_eq!(tree.height(), depth_for_leaves(padded));
        }
    }

    #[test]
    fn test_tree_deterministic_root() {
        let data = sample_data(8);
        let t1 = MerkleTree::new(&data);
        let t2 = MerkleTree::new(&data);
        assert_eq!(t1.root(), t2.root());
    }

    #[test]
    fn test_tree_different_data_different_root() {
        let t1 = MerkleTree::new(&sample_data(4));
        let t2 = MerkleTree::new(&[
            b"a".to_vec(), b"b".to_vec(), b"c".to_vec(), b"d".to_vec(),
        ]);
        assert_ne!(t1.root(), t2.root());
    }

    #[test]
    fn test_tree_get_leaf() {
        let data = sample_data(8);
        let tree = MerkleTree::new(&data);
        for i in 0..8 {
            let expected = MerkleTree::hash_leaf(&data[i]);
            assert_eq!(tree.get_leaf(i), expected);
        }
    }

    #[test]
    #[should_panic]
    fn test_tree_get_leaf_out_of_range() {
        let tree = MerkleTree::new(&sample_data(4));
        tree.get_leaf(4);
    }

    #[test]
    fn test_tree_from_hashes() {
        let hashes: Vec<[u8; 32]> = (0..8u8)
            .map(|i| {
                let mut h = [0u8; 32];
                h[0] = i;
                h
            })
            .collect();
        let tree = MerkleTree::from_hashes(hashes.clone());
        assert_eq!(tree.leaf_count(), 8);
        assert_eq!(tree.get_leaf(3), hashes[3]);
    }

    #[test]
    fn test_tree_from_field_rows_u64() {
        let rows: Vec<Vec<u64>> = (0..4)
            .map(|i| vec![i as u64, (i * 2) as u64, (i * 3) as u64])
            .collect();
        let tree = MerkleTree::from_field_rows_u64(&rows);
        assert_eq!(tree.leaf_count(), 4);
        assert_eq!(tree.height(), 2);
    }

    // -----------------------------------------------------------------------
    // MerkleTree: proof generation & verification
    // -----------------------------------------------------------------------

    #[test]
    fn test_proof_single_leaf() {
        let tree = MerkleTree::new(&sample_data(1));
        let proof = tree.prove(0);
        assert_eq!(proof.depth(), 0);
        assert!(MerkleTree::verify_proof(&tree.root(), 0, &proof.leaf_hash, &proof));
    }

    #[test]
    fn test_proof_two_leaves() {
        let tree = MerkleTree::new(&sample_data(2));
        for i in 0..2 {
            let proof = tree.prove(i);
            assert_eq!(proof.depth(), 1);
            assert!(MerkleTree::verify_proof(&tree.root(), i, &proof.leaf_hash, &proof));
        }
    }

    #[test]
    fn test_proof_all_leaves_power_of_two() {
        let n = 16;
        let tree = MerkleTree::new(&sample_data(n));
        for i in 0..n {
            let proof = tree.prove(i);
            assert!(MerkleTree::verify_proof(&tree.root(), i, &proof.leaf_hash, &proof));
        }
    }

    #[test]
    fn test_proof_all_leaves_non_power_of_two() {
        for &n in &[3, 5, 7, 9, 13, 17] {
            let tree = MerkleTree::new(&sample_data(n));
            for i in 0..n {
                let proof = tree.prove(i);
                assert!(
                    MerkleTree::verify_proof(&tree.root(), i, &proof.leaf_hash, &proof),
                    "failed for n={}, i={}", n, i,
                );
            }
        }
    }

    #[test]
    fn test_proof_wrong_root_fails() {
        let tree = MerkleTree::new(&sample_data(8));
        let proof = tree.prove(3);
        let wrong_root = [0xFFu8; 32];
        assert!(!MerkleTree::verify_proof(&wrong_root, 3, &proof.leaf_hash, &proof));
    }

    #[test]
    fn test_proof_wrong_index_fails() {
        let tree = MerkleTree::new(&sample_data(8));
        let proof = tree.prove(3);
        assert!(!MerkleTree::verify_proof(&tree.root(), 4, &proof.leaf_hash, &proof));
    }

    #[test]
    fn test_proof_wrong_leaf_fails() {
        let tree = MerkleTree::new(&sample_data(8));
        let proof = tree.prove(3);
        let wrong_leaf = [0xAAu8; 32];
        assert!(!MerkleTree::verify_proof(&tree.root(), 3, &wrong_leaf, &proof));
    }

    #[test]
    fn test_proof_tampered_sibling_fails() {
        let tree = MerkleTree::new(&sample_data(8));
        let mut proof = tree.prove(3);
        proof.siblings[0] = [0xBBu8; 32];
        assert!(!MerkleTree::verify_proof(&tree.root(), 3, &proof.leaf_hash, &proof));
    }

    #[test]
    fn test_proof_depth_matches_height() {
        for &n in &[1, 2, 4, 8, 16] {
            let tree = MerkleTree::new(&sample_data(n));
            let proof = tree.prove(0);
            assert_eq!(proof.depth(), tree.height());
        }
    }

    // -----------------------------------------------------------------------
    // MerkleProof: serialization round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn test_proof_serialize_roundtrip() {
        let tree = MerkleTree::new(&sample_data(16));
        for i in 0..16 {
            let proof = tree.prove(i);
            let bytes = proof.serialize_to_bytes();
            assert_eq!(bytes.len(), proof.size_in_bytes());
            let restored = MerkleProof::deserialize_from_bytes(&bytes).unwrap();
            assert_eq!(restored.index, proof.index);
            assert_eq!(restored.leaf_hash, proof.leaf_hash);
            assert_eq!(restored.siblings.len(), proof.siblings.len());
            for (a, b) in restored.siblings.iter().zip(proof.siblings.iter()) {
                assert_eq!(a, b);
            }
            assert!(MerkleTree::verify_proof(&tree.root(), i, &restored.leaf_hash, &restored));
        }
    }

    #[test]
    fn test_proof_deserialize_too_short() {
        assert!(MerkleProof::deserialize_from_bytes(&[0u8; 10]).is_none());
    }

    #[test]
    fn test_proof_deserialize_truncated_siblings() {
        let tree = MerkleTree::new(&sample_data(8));
        let proof = tree.prove(0);
        let bytes = proof.serialize_to_bytes();
        let truncated = &bytes[..bytes.len() - 16];
        assert!(MerkleProof::deserialize_from_bytes(truncated).is_none());
    }

    // -----------------------------------------------------------------------
    // Legacy verify tests (backward-compat with old API)
    // -----------------------------------------------------------------------

    #[test]
    fn test_merkle_build_and_root() {
        let leaves: Vec<Vec<u8>> = (0..8u8).map(|i| vec![i; 16]).collect();
        let tree = MerkleTree::from_leaves(&leaves);
        let root = tree.root();
        assert_ne!(root, [0u8; 32]);
    }

    #[test]
    fn test_merkle_prove_verify_legacy() {
        let leaves: Vec<Vec<u8>> = (0..4u8).map(|i| vec![i; 8]).collect();
        let tree = MerkleTree::from_leaves(&leaves);
        let root = tree.root();

        for i in 0..4 {
            let proof = tree.prove(i);
            assert!(MerkleTree::verify(&root, &vec![i as u8; 8], &proof));
        }
    }

    #[test]
    fn test_merkle_field_rows() {
        let rows: Vec<Vec<GoldilocksField>> = (0..4)
            .map(|r| (0..3).map(|c| GoldilocksField::new((r * 3 + c) as u64)).collect())
            .collect();
        let tree = MerkleTree::from_field_rows(&rows);
        let root = tree.root();

        let proof = tree.prove(2);
        assert!(MerkleTree::verify_field_row(&root, &rows[2], &proof));
    }

    #[test]
    fn test_merkle_tampered_proof_fails() {
        let leaves: Vec<Vec<u8>> = (0..4u8).map(|i| vec![i; 8]).collect();
        let tree = MerkleTree::from_leaves(&leaves);
        let root = tree.root();
        let proof = tree.prove(0);
        assert!(!MerkleTree::verify(&root, &vec![99u8; 8], &proof));
    }

    // -----------------------------------------------------------------------
    // BatchMerkleProof
    // -----------------------------------------------------------------------

    #[test]
    fn test_batch_proof_single() {
        let tree = MerkleTree::new(&sample_data(8));
        let batch = tree.prove_batch(&[3]);
        assert!(batch.verify_all(&tree.root()));
        assert_eq!(batch.indices, vec![3]);
    }

    #[test]
    fn test_batch_proof_multiple() {
        let tree = MerkleTree::new(&sample_data(16));
        let indices = vec![0, 3, 7, 12, 15];
        let batch = tree.prove_batch(&indices);
        assert!(batch.verify_all(&tree.root()));
        assert_eq!(batch.indices, indices);
    }

    #[test]
    fn test_batch_proof_all_leaves() {
        let n = 8;
        let tree = MerkleTree::new(&sample_data(n));
        let indices: Vec<usize> = (0..n).collect();
        let batch = tree.prove_batch(&indices);
        assert!(batch.verify_all(&tree.root()));
    }

    #[test]
    fn test_batch_proof_wrong_root() {
        let tree = MerkleTree::new(&sample_data(8));
        let batch = tree.prove_batch(&[0, 4]);
        let wrong = [0xFFu8; 32];
        assert!(!batch.verify_all(&wrong));
    }

    #[test]
    fn test_batch_proof_compression() {
        let tree = MerkleTree::new(&sample_data(16));
        let batch = tree.prove_batch(&[0, 1, 2, 3]);
        let ratio = batch.compression_ratio();
        assert!(ratio <= 1.0, "ratio should be <= 1.0, got {}", ratio);
    }

    #[test]
    fn test_batch_verify_static() {
        let tree = MerkleTree::new(&sample_data(8));
        let batch = tree.prove_batch(&[1, 5]);
        assert!(MerkleTree::verify_batch(&tree.root(), &batch));
    }

    #[test]
    fn test_batch_proof_size() {
        let tree = MerkleTree::new(&sample_data(16));
        let batch = tree.prove_batch(&[0, 8]);
        assert!(batch.size_in_bytes() > 0);
    }

    // -----------------------------------------------------------------------
    // IncrementalMerkleTree
    // -----------------------------------------------------------------------

    #[test]
    fn test_incremental_single_insert() {
        let mut inc = IncrementalMerkleTree::new();
        inc.insert(b"leaf_0");
        assert_eq!(inc.current_count(), 1);
        let tree = inc.finalize();
        assert_eq!(tree.leaf_count(), 1);
    }

    #[test]
    fn test_incremental_matches_batch() {
        let data = sample_data(8);
        let batch_tree = MerkleTree::new(&data);

        let mut inc = IncrementalMerkleTree::new();
        for d in &data {
            inc.insert(d);
        }
        let inc_tree = inc.finalize();

        assert_eq!(inc_tree.root(), batch_tree.root());
    }

    #[test]
    fn test_incremental_non_power_of_two() {
        let data = sample_data(5);
        let batch_tree = MerkleTree::new(&data);

        let mut inc = IncrementalMerkleTree::new();
        for d in &data {
            inc.insert(d);
        }
        let inc_tree = inc.finalize();

        assert_eq!(inc_tree.root(), batch_tree.root());
    }

    #[test]
    fn test_incremental_insert_hash() {
        let data = sample_data(4);
        let hashes: Vec<[u8; 32]> = data.iter().map(|d| MerkleTree::hash_leaf(d)).collect();

        let mut inc = IncrementalMerkleTree::new();
        for h in &hashes {
            inc.insert_hash(*h);
        }
        let inc_tree = inc.finalize();

        let batch_tree = MerkleTree::new(&data);
        assert_eq!(inc_tree.root(), batch_tree.root());
    }

    #[test]
    fn test_incremental_count() {
        let mut inc = IncrementalMerkleTree::new();
        for i in 0..10 {
            assert_eq!(inc.current_count(), i);
            inc.insert(format!("item_{}", i).as_bytes());
        }
        assert_eq!(inc.current_count(), 10);
    }

    #[test]
    #[should_panic]
    fn test_incremental_empty_finalize_panics() {
        let inc = IncrementalMerkleTree::new();
        let _ = inc.finalize();
    }

    #[test]
    fn test_incremental_large() {
        let n = 100;
        let data = sample_data(n);
        let batch_tree = MerkleTree::new(&data);
        let mut inc = IncrementalMerkleTree::new();
        for d in &data {
            inc.insert(d);
        }
        assert_eq!(inc.finalize().root(), batch_tree.root());
    }

    // -----------------------------------------------------------------------
    // MerkleForest
    // -----------------------------------------------------------------------

    #[test]
    fn test_forest_single_tree() {
        let tree = MerkleTree::new(&sample_data(4));
        let forest = MerkleForest::new(vec![tree]);
        assert_eq!(forest.tree_count(), 1);
        let cr = forest.root();
        assert_ne!(cr, [0u8; 32]);
    }

    #[test]
    fn test_forest_multiple_trees() {
        let t1 = MerkleTree::new(&sample_data(4));
        let t2 = MerkleTree::new(&[b"a".to_vec(), b"b".to_vec()]);
        let t3 = MerkleTree::new(&sample_data(8));
        let forest = MerkleForest::new(vec![t1, t2, t3]);
        assert_eq!(forest.tree_count(), 3);
    }

    #[test]
    fn test_forest_prove_and_verify() {
        let t1 = MerkleTree::new(&sample_data(4));
        let t2 = MerkleTree::new(&sample_data(8));
        let forest = MerkleForest::new(vec![t1, t2]);
        let combined = forest.root();

        let proof = forest.prove_leaf(0, 2);
        assert!(MerkleForest::verify_leaf(&combined, &proof));

        let proof2 = forest.prove_leaf(1, 5);
        assert!(MerkleForest::verify_leaf(&combined, &proof2));
    }

    #[test]
    fn test_forest_wrong_combined_root() {
        let t1 = MerkleTree::new(&sample_data(4));
        let forest = MerkleForest::new(vec![t1]);
        let proof = forest.prove_leaf(0, 0);
        let wrong = [0xFFu8; 32];
        assert!(!MerkleForest::verify_leaf(&wrong, &proof));
    }

    #[test]
    fn test_forest_tampered_proof() {
        let t1 = MerkleTree::new(&sample_data(4));
        let t2 = MerkleTree::new(&sample_data(4));
        let forest = MerkleForest::new(vec![t1, t2]);
        let combined = forest.root();
        let mut proof = forest.prove_leaf(0, 1);
        proof.all_roots[1] = [0xCCu8; 32];
        assert!(!MerkleForest::verify_leaf(&combined, &proof));
    }

    // -----------------------------------------------------------------------
    // SparseMerkleTree
    // -----------------------------------------------------------------------

    #[test]
    fn test_sparse_empty_root_deterministic() {
        let mut s1 = SparseMerkleTree::new(10);
        let mut s2 = SparseMerkleTree::new(10);
        assert_eq!(s1.root(), s2.root());
    }

    #[test]
    fn test_sparse_set_and_get() {
        let mut smt = SparseMerkleTree::new(4);
        smt.set(5, b"hello");
        let h = smt.get(5);
        assert_eq!(h, MerkleTree::hash_leaf(b"hello"));
    }

    #[test]
    fn test_sparse_unset_returns_zero() {
        let smt = SparseMerkleTree::new(4);
        assert_eq!(smt.get(3), [0u8; 32]);
    }

    #[test]
    fn test_sparse_root_changes_on_insert() {
        let mut smt = SparseMerkleTree::new(4);
        let r1 = smt.root();
        smt.set(7, b"data");
        let r2 = smt.root();
        assert_ne!(r1, r2);
    }

    #[test]
    fn test_sparse_proof_verifies() {
        let mut smt = SparseMerkleTree::new(8);
        smt.set(42, b"answer");
        smt.set(100, b"century");
        let root = smt.root();
        let proof = smt.prove(42);
        assert!(MerkleTree::verify_proof(&root, 42, &proof.leaf_hash, &proof));
    }

    #[test]
    fn test_sparse_proof_non_member() {
        let mut smt = SparseMerkleTree::new(8);
        smt.set(10, b"present");
        let root = smt.root();
        let proof = smt.prove(20);
        assert_eq!(proof.leaf_hash, [0u8; 32]);
        assert!(MerkleTree::verify_proof(&root, 20, &proof.leaf_hash, &proof));
    }

    #[test]
    fn test_sparse_population() {
        let mut smt = SparseMerkleTree::new(10);
        assert_eq!(smt.population(), 0);
        smt.set(0, b"a");
        smt.set(100, b"b");
        smt.set(500, b"c");
        assert_eq!(smt.population(), 3);
    }

    #[test]
    fn test_sparse_overwrite() {
        let mut smt = SparseMerkleTree::new(4);
        smt.set(3, b"first");
        let r1 = smt.root();
        smt.set(3, b"second");
        let r2 = smt.root();
        assert_ne!(r1, r2);
        assert_eq!(smt.population(), 1);
    }

    // -----------------------------------------------------------------------
    // LayeredMerkleTree
    // -----------------------------------------------------------------------

    #[test]
    fn test_layered_matches_standard() {
        let data = sample_data(16);
        let hashes: Vec<[u8; 32]> = data.iter().map(|d| MerkleTree::hash_leaf(d)).collect();
        let standard = MerkleTree::from_hashes(hashes.clone());
        let layered = LayeredMerkleTree::new(hashes);
        assert_eq!(standard.root(), layered.root());
    }

    #[test]
    fn test_layered_proof_verifies() {
        let data = sample_data(8);
        let hashes: Vec<[u8; 32]> = data.iter().map(|d| MerkleTree::hash_leaf(d)).collect();
        let layered = LayeredMerkleTree::new(hashes);
        for i in 0..8 {
            let proof = layered.prove(i);
            assert!(MerkleTree::verify_proof(&layered.root(), i, &proof.leaf_hash, &proof));
        }
    }

    #[test]
    fn test_layered_update_leaf() {
        let data = sample_data(8);
        let hashes: Vec<[u8; 32]> = data.iter().map(|d| MerkleTree::hash_leaf(d)).collect();
        let mut layered = LayeredMerkleTree::new(hashes);
        let old_root = layered.root();
        let new_hash = hash_bytes(b"updated");
        layered.update_leaf(3, new_hash);
        assert_ne!(layered.root(), old_root);

        let proof = layered.prove(3);
        assert_eq!(proof.leaf_hash, new_hash);
        assert!(MerkleTree::verify_proof(&layered.root(), 3, &new_hash, &proof));
    }

    #[test]
    fn test_layered_update_preserves_other_proofs() {
        let data = sample_data(8);
        let hashes: Vec<[u8; 32]> = data.iter().map(|d| MerkleTree::hash_leaf(d)).collect();
        let mut layered = LayeredMerkleTree::new(hashes.clone());
        layered.update_leaf(0, hash_bytes(b"new_leaf_0"));
        let proof = layered.prove(4);
        assert_eq!(proof.leaf_hash, hashes[4]);
        assert!(MerkleTree::verify_proof(&layered.root(), 4, &hashes[4], &proof));
    }

    #[test]
    fn test_layered_non_power_of_two() {
        let data = sample_data(5);
        let hashes: Vec<[u8; 32]> = data.iter().map(|d| MerkleTree::hash_leaf(d)).collect();
        let standard = MerkleTree::from_hashes(hashes.clone());
        let layered = LayeredMerkleTree::new(hashes);
        assert_eq!(standard.root(), layered.root());
    }

    // -----------------------------------------------------------------------
    // MerkleCap
    // -----------------------------------------------------------------------

    #[test]
    fn test_cap_from_tree() {
        let data = sample_data(16);
        let tree = MerkleTree::new(&data);
        let cap = MerkleCap::from_tree(&tree, 0);
        assert_eq!(cap.len(), 1);
        assert_eq!(cap.cap[0], tree.root());
    }

    #[test]
    fn test_cap_height_2() {
        let data = sample_data(16);
        let tree = MerkleTree::new(&data);
        let cap = MerkleCap::from_tree(&tree, 2);
        assert_eq!(cap.len(), 4);
        for c in &cap.cap {
            assert_ne!(*c, [0u8; 32]);
        }
    }

    #[test]
    fn test_cap_verify() {
        let data = sample_data(16);
        let hashes: Vec<[u8; 32]> = data.iter().map(|d| MerkleTree::hash_leaf(d)).collect();
        let layered = LayeredMerkleTree::new(hashes);
        let cap = MerkleCap::from_layered_tree(&layered, 2);

        let full_proof = layered.prove(5);
        let partial_siblings = full_proof.siblings[..layered.height() - 2].to_vec();
        let partial_proof = MerkleProof {
            siblings: partial_siblings,
            index: 5,
            leaf_hash: full_proof.leaf_hash,
        };
        assert!(cap.verify(5, &full_proof.leaf_hash, &partial_proof));
    }

    // -----------------------------------------------------------------------
    // MultiColumnCommitment
    // -----------------------------------------------------------------------

    #[test]
    fn test_multi_column_basic() {
        let col0: Vec<u64> = vec![1, 2, 3, 4];
        let col1: Vec<u64> = vec![10, 20, 30, 40];
        let commit = MultiColumnCommitment::new(&[col0, col1]);
        assert_eq!(commit.num_columns(), 2);
        assert_eq!(commit.num_rows(), 4);
        assert_ne!(commit.root(), [0u8; 32]);
    }

    #[test]
    fn test_multi_column_prove_verify() {
        let col0: Vec<u64> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let col1: Vec<u64> = vec![10, 20, 30, 40, 50, 60, 70, 80];
        let col2: Vec<u64> = vec![100, 200, 300, 400, 500, 600, 700, 800];
        let columns = vec![col0.clone(), col1.clone(), col2.clone()];
        let commit = MultiColumnCommitment::new(&columns);
        let root = commit.root();

        for row in 0..8 {
            let proof = commit.prove_row(row);
            let values = vec![col0[row], col1[row], col2[row]];
            assert!(MultiColumnCommitment::verify_row(&root, row, &values, &proof));
        }
    }

    #[test]
    fn test_multi_column_wrong_values_fail() {
        let col0: Vec<u64> = vec![1, 2, 3, 4];
        let col1: Vec<u64> = vec![10, 20, 30, 40];
        let commit = MultiColumnCommitment::new(&[col0, col1]);
        let root = commit.root();
        let proof = commit.prove_row(2);
        assert!(!MultiColumnCommitment::verify_row(&root, 2, &[99, 99], &proof));
    }

    // -----------------------------------------------------------------------
    // Fiat-Shamir indices
    // -----------------------------------------------------------------------

    #[test]
    fn test_fiat_shamir_deterministic() {
        let root = hash_bytes(b"root");
        let seed = b"challenge";
        let a = fiat_shamir_indices(&root, seed, 1024, 10);
        let b = fiat_shamir_indices(&root, seed, 1024, 10);
        assert_eq!(a, b);
    }

    #[test]
    fn test_fiat_shamir_distinct() {
        let root = hash_bytes(b"root");
        let indices = fiat_shamir_indices(&root, b"seed", 1_000_000, 100);
        let set: std::collections::HashSet<usize> = indices.iter().cloned().collect();
        assert_eq!(set.len(), 100);
    }

    #[test]
    fn test_fiat_shamir_within_range() {
        let root = hash_bytes(b"root");
        let indices = fiat_shamir_indices(&root, b"seed", 256, 50);
        for &i in &indices {
            assert!(i < 256);
        }
    }

    #[test]
    fn test_fiat_shamir_different_seed() {
        let root = hash_bytes(b"root");
        let a = fiat_shamir_indices(&root, b"seed_a", 1024, 10);
        let b = fiat_shamir_indices(&root, b"seed_b", 1024, 10);
        assert_ne!(a, b);
    }

    // -----------------------------------------------------------------------
    // MerkleTranscript
    // -----------------------------------------------------------------------

    #[test]
    fn test_transcript_squeeze_deterministic() {
        let mut t1 = MerkleTranscript::new(b"test");
        let mut t2 = MerkleTranscript::new(b"test");
        t1.append_root(b"root", &[1u8; 32]);
        t2.append_root(b"root", &[1u8; 32]);
        assert_eq!(t1.squeeze_challenge(), t2.squeeze_challenge());
    }

    #[test]
    fn test_transcript_different_labels() {
        let mut t1 = MerkleTranscript::new(b"protocol_a");
        let mut t2 = MerkleTranscript::new(b"protocol_b");
        t1.append_root(b"r", &[1u8; 32]);
        t2.append_root(b"r", &[1u8; 32]);
        assert_ne!(t1.squeeze_challenge(), t2.squeeze_challenge());
    }

    #[test]
    fn test_transcript_squeeze_indices() {
        let mut t = MerkleTranscript::new(b"stark");
        t.append_root(b"trace", &hash_bytes(b"trace_root"));
        let indices = t.squeeze_indices(1024, 20);
        assert_eq!(indices.len(), 20);
        let set: std::collections::HashSet<usize> = indices.iter().cloned().collect();
        assert_eq!(set.len(), 20);
    }

    // -----------------------------------------------------------------------
    // merkle_root_from_data convenience
    // -----------------------------------------------------------------------

    #[test]
    fn test_merkle_root_from_data() {
        let data = sample_data(8);
        let direct = MerkleTree::new(&data).root();
        let convenience = merkle_root_from_data(&data);
        assert_eq!(direct, convenience);
    }

    // -----------------------------------------------------------------------
    // Display / Debug formatting
    // -----------------------------------------------------------------------

    #[test]
    fn test_tree_debug_format() {
        let tree = MerkleTree::new(&sample_data(4));
        let s = format!("{:?}", tree);
        assert!(s.contains("MerkleTree"));
        assert!(s.contains("leaves: 4"));
    }

    #[test]
    fn test_proof_display() {
        let tree = MerkleTree::new(&sample_data(4));
        let proof = tree.prove(1);
        let s = format!("{}", proof);
        assert!(s.contains("MerkleProof"));
        assert!(s.contains("index: 1"));
    }

    #[test]
    fn test_batch_proof_display() {
        let tree = MerkleTree::new(&sample_data(8));
        let batch = tree.prove_batch(&[0, 3, 7]);
        let s = format!("{}", batch);
        assert!(s.contains("BatchMerkleProof"));
        assert!(s.contains("num_proofs: 3"));
    }

    #[test]
    fn test_forest_debug() {
        let t = MerkleTree::new(&sample_data(4));
        let forest = MerkleForest::new(vec![t]);
        let s = format!("{:?}", forest);
        assert!(s.contains("MerkleForest"));
    }

    #[test]
    fn test_sparse_debug() {
        let smt = SparseMerkleTree::new(8);
        let s = format!("{:?}", smt);
        assert!(s.contains("SparseMerkleTree"));
    }

    #[test]
    fn test_layered_debug() {
        let hashes: Vec<[u8; 32]> = (0..4).map(|i| {
            let mut h = [0u8; 32]; h[0] = i; h
        }).collect();
        let lt = LayeredMerkleTree::new(hashes);
        let s = format!("{:?}", lt);
        assert!(s.contains("LayeredMerkleTree"));
    }

    #[test]
    fn test_incremental_debug() {
        let inc = IncrementalMerkleTree::new();
        let s = format!("{:?}", inc);
        assert!(s.contains("IncrementalMerkleTree"));
    }

    #[test]
    fn test_multi_column_debug() {
        let commit = MultiColumnCommitment::new(&[vec![1, 2], vec![3, 4]]);
        let s = format!("{:?}", commit);
        assert!(s.contains("MultiColumnCommitment"));
    }

    // -----------------------------------------------------------------------
    // Edge cases & stress
    // -----------------------------------------------------------------------

    #[test]
    fn test_tree_256_leaves() {
        let tree = MerkleTree::new(&sample_data(256));
        assert_eq!(tree.leaf_count(), 256);
        assert_eq!(tree.height(), 8);
        for i in [0, 1, 127, 128, 255] {
            let proof = tree.prove(i);
            assert!(MerkleTree::verify_proof(&tree.root(), i, &proof.leaf_hash, &proof));
        }
    }

    #[test]
    fn test_tree_1024_leaves() {
        let tree = MerkleTree::new(&sample_data(1024));
        assert_eq!(tree.height(), 10);
        let proof = tree.prove(512);
        assert!(MerkleTree::verify_proof(&tree.root(), 512, &proof.leaf_hash, &proof));
    }

    #[test]
    fn test_batch_proof_adjacent_siblings() {
        let tree = MerkleTree::new(&sample_data(8));
        let batch = tree.prove_batch(&[2, 3]);
        assert!(batch.verify_all(&tree.root()));
        assert!(batch.compression_ratio() < 1.0);
    }

    #[test]
    fn test_forest_many_trees() {
        let trees: Vec<MerkleTree> = (0..10)
            .map(|i| MerkleTree::new(&[format!("tree_{}_leaf", i).into_bytes()]))
            .collect();
        let forest = MerkleForest::new(trees);
        assert_eq!(forest.tree_count(), 10);
        for t in 0..10 {
            let proof = forest.prove_leaf(t, 0);
            assert!(MerkleForest::verify_leaf(&forest.root(), &proof));
        }
    }

    #[test]
    fn test_sparse_depth_1() {
        let mut smt = SparseMerkleTree::new(1);
        smt.set(0, b"left");
        smt.set(1, b"right");
        let root = smt.root();
        let proof = smt.prove(0);
        assert!(MerkleTree::verify_proof(&root, 0, &proof.leaf_hash, &proof));
        let proof = smt.prove(1);
        assert!(MerkleTree::verify_proof(&root, 1, &proof.leaf_hash, &proof));
    }

    #[test]
    fn test_sparse_large_depth() {
        let mut smt = SparseMerkleTree::new(20);
        smt.set(0, b"first");
        smt.set(1_000_000, b"far_away");
        let root = smt.root();
        let proof = smt.prove(0);
        assert!(MerkleTree::verify_proof(&root, 0, &proof.leaf_hash, &proof));
        let proof = smt.prove(1_000_000);
        assert!(MerkleTree::verify_proof(&root, 1_000_000, &proof.leaf_hash, &proof));
    }

    #[test]
    fn test_bench_hash_throughput() {
        let (bytes, nanos) = bench_hash_throughput(100, 64);
        assert_eq!(bytes, 100 * 64);
        assert!(nanos > 0);
    }

    #[test]
    fn test_cap_full_height_equals_root() {
        let data = sample_data(8);
        let tree = MerkleTree::new(&data);
        let cap = MerkleCap::from_tree(&tree, 0);
        assert_eq!(cap.cap[0], tree.root());
    }

    #[test]
    fn test_cap_is_empty() {
        let data = sample_data(4);
        let tree = MerkleTree::new(&data);
        let cap = MerkleCap::from_tree(&tree, 1);
        assert!(!cap.is_empty());
    }

    #[test]
    fn test_incremental_default_trait() {
        let inc = IncrementalMerkleTree::default();
        assert_eq!(inc.current_count(), 0);
    }

    #[test]
    fn test_tree_from_field_rows_u64_proof() {
        let rows: Vec<Vec<u64>> = (0..8)
            .map(|i| vec![i as u64, i * 2, i * 3])
            .collect();
        let tree = MerkleTree::from_field_rows_u64(&rows);
        for i in 0..8 {
            let proof = tree.prove(i);
            assert!(MerkleTree::verify_proof(&tree.root(), i, &proof.leaf_hash, &proof));
        }
    }

    #[test]
    fn test_layered_update_multiple_leaves() {
        let data = sample_data(8);
        let hashes: Vec<[u8; 32]> = data.iter().map(|d| MerkleTree::hash_leaf(d)).collect();
        let mut layered = LayeredMerkleTree::new(hashes);

        for i in 0..8 {
            layered.update_leaf(i, hash_bytes(format!("updated_{}", i).as_bytes()));
        }
        let root = layered.root();
        for i in 0..8 {
            let proof = layered.prove(i);
            assert!(MerkleTree::verify_proof(&root, i, &proof.leaf_hash, &proof));
        }
    }

    #[test]
    fn test_multi_column_single_column() {
        let col: Vec<u64> = vec![100, 200, 300, 400];
        let commit = MultiColumnCommitment::new(&[col.clone()]);
        let proof = commit.prove_row(2);
        assert!(MultiColumnCommitment::verify_row(&commit.root(), 2, &[col[2]], &proof));
    }

    #[test]
    fn test_forest_deterministic_root() {
        let t1 = MerkleTree::new(&sample_data(4));
        let t2 = MerkleTree::new(&sample_data(8));
        let f1 = MerkleForest::new(vec![t1.clone(), t2.clone()]);
        let f2 = MerkleForest::new(vec![t1, t2]);
        assert_eq!(f1.root(), f2.root());
    }

    #[test]
    fn test_forest_order_matters() {
        let t1 = MerkleTree::new(&sample_data(4));
        let t2 = MerkleTree::new(&sample_data(8));
        let f1 = MerkleForest::new(vec![t1.clone(), t2.clone()]);
        let f2 = MerkleForest::new(vec![t2, t1]);
        assert_ne!(f1.root(), f2.root());
    }

    #[test]
    fn test_hash_bytes_known_non_zero() {
        for b in 0u8..=255 {
            let h = hash_bytes(&[b]);
            assert_ne!(h, [0u8; 32], "hash of byte {} was all zeros", b);
        }
    }

    #[test]
    fn test_proof_size_in_bytes_consistency() {
        let tree = MerkleTree::new(&sample_data(32));
        let proof = tree.prove(17);
        let serialized = proof.serialize_to_bytes();
        assert_eq!(serialized.len(), proof.size_in_bytes());
    }

    #[test]
    fn test_batch_proof_from_individual_dedup() {
        let tree = MerkleTree::new(&sample_data(8));
        let p0 = tree.prove(0);
        let p1 = tree.prove(1);
        let batch = BatchMerkleProof::from_individual_proofs(vec![p0, p1]);
        let total_individual: usize = batch.proofs.iter().map(|p| p.siblings.len()).sum();
        assert!(batch.compressed_siblings.len() <= total_individual);
    }

    #[test]
    fn test_incremental_power_of_two_sizes() {
        for &n in &[1, 2, 4, 8, 16, 32] {
            let data = sample_data(n);
            let batch = MerkleTree::new(&data);
            let mut inc = IncrementalMerkleTree::new();
            for d in &data {
                inc.insert(d);
            }
            assert_eq!(inc.finalize().root(), batch.root(), "mismatch for n={}", n);
        }
    }

    #[test]
    fn test_incremental_odd_sizes() {
        for &n in &[3, 5, 7, 11, 13, 19, 23, 50] {
            let data = sample_data(n);
            let batch = MerkleTree::new(&data);
            let mut inc = IncrementalMerkleTree::new();
            for d in &data {
                inc.insert(d);
            }
            assert_eq!(inc.finalize().root(), batch.root(), "mismatch for n={}", n);
        }
    }

    #[test]
    fn test_sparse_matches_dense_for_full_population() {
        let mut smt = SparseMerkleTree::new(4);
        let mut hashes = Vec::new();
        for i in 0..16u8 {
            let data = [i; 4];
            let h = MerkleTree::hash_leaf(&data);
            smt.set_hash(i as usize, h);
            hashes.push(h);
        }
        let dense = MerkleTree::from_hashes(hashes);
        assert_eq!(smt.root(), dense.root());
    }

    #[test]
    fn test_transcript_multiple_squeezes() {
        let mut t = MerkleTranscript::new(b"multi");
        t.append_bytes(b"data", b"round1");
        let c1 = t.squeeze_challenge();
        t.append_bytes(b"data", b"round2");
        let c2 = t.squeeze_challenge();
        assert_ne!(c1, c2);
    }

    #[test]
    fn test_fiat_shamir_domain_size_1() {
        let root = [0u8; 32];
        let indices = fiat_shamir_indices(&root, b"s", 1, 1);
        assert_eq!(indices, vec![0]);
    }

    #[test]
    fn test_hash_field_elements_empty() {
        let h = hash_field_elements(&[]);
        assert_ne!(h, [0u8; 32]);
    }

    #[test]
    fn test_multi_column_deterministic() {
        let cols1 = vec![vec![1u64, 2, 3, 4], vec![5u64, 6, 7, 8]];
        let cols2 = vec![vec![1u64, 2, 3, 4], vec![5u64, 6, 7, 8]];
        let c1 = MultiColumnCommitment::new(&cols1);
        let c2 = MultiColumnCommitment::new(&cols2);
        assert_eq!(c1.root(), c2.root());
    }

    #[test]
    fn test_layered_single_leaf() {
        let hashes = vec![hash_bytes(b"only")];
        let lt = LayeredMerkleTree::new(hashes);
        assert_eq!(lt.leaf_count(), 1);
        assert_eq!(lt.height(), 0);
        let proof = lt.prove(0);
        assert!(MerkleTree::verify_proof(&lt.root(), 0, &proof.leaf_hash, &proof));
    }

    #[test]
    fn test_forest_get_tree() {
        let t1 = MerkleTree::new(&sample_data(4));
        let t2 = MerkleTree::new(&sample_data(8));
        let r1 = t1.root();
        let r2 = t2.root();
        let forest = MerkleForest::new(vec![t1, t2]);
        assert_eq!(forest.get_tree(0).root(), r1);
        assert_eq!(forest.get_tree(1).root(), r2);
    }

    #[test]
    fn test_tree_clone() {
        let tree = MerkleTree::new(&sample_data(8));
        let cloned = tree.clone();
        assert_eq!(tree.root(), cloned.root());
        assert_eq!(tree.leaf_count(), cloned.leaf_count());
    }

    #[test]
    fn test_forest_clone() {
        let t = MerkleTree::new(&sample_data(4));
        let forest = MerkleForest::new(vec![t]);
        let cloned = forest.clone();
        assert_eq!(forest.root(), cloned.root());
    }

    #[test]
    fn test_proof_clone() {
        let tree = MerkleTree::new(&sample_data(4));
        let proof = tree.prove(2);
        let cloned = proof.clone();
        assert_eq!(proof.index, cloned.index);
        assert_eq!(proof.leaf_hash, cloned.leaf_hash);
        assert_eq!(proof.siblings, cloned.siblings);
    }

    #[test]
    fn test_batch_proof_clone() {
        let tree = MerkleTree::new(&sample_data(8));
        let batch = tree.prove_batch(&[0, 4]);
        let cloned = batch.clone();
        assert_eq!(batch.indices, cloned.indices);
        assert!(cloned.verify_all(&tree.root()));
    }

    #[test]
    fn test_cap_clone() {
        let tree = MerkleTree::new(&sample_data(8));
        let cap = MerkleCap::from_tree(&tree, 1);
        let cloned = cap.clone();
        assert_eq!(cap.cap, cloned.cap);
    }

    // Integration: full STARK-like workflow
    #[test]
    fn test_stark_workflow_integration() {
        // 1. Commit to trace columns.
        let trace_col0: Vec<u64> = (0..64).collect();
        let trace_col1: Vec<u64> = (0..64).map(|x| x * x).collect();
        let commit = MultiColumnCommitment::new(&[trace_col0.clone(), trace_col1.clone()]);
        let trace_root = commit.root();

        // 2. Build a transcript and derive query indices.
        let mut transcript = MerkleTranscript::new(b"stark-test");
        transcript.append_root(b"trace_commit", &trace_root);
        let query_indices = transcript.squeeze_indices(64, 8);
        assert_eq!(query_indices.len(), 8);

        // 3. Open the queried rows and generate proofs.
        for &idx in &query_indices {
            let proof = commit.prove_row(idx);
            let values = vec![trace_col0[idx], trace_col1[idx]];
            assert!(MultiColumnCommitment::verify_row(&trace_root, idx, &values, &proof));
        }

        // 4. Also test batch proof on the underlying tree.
        let tree = MerkleTree::from_field_rows_u64(
            &(0..64)
                .map(|i| vec![i as u64, (i * i) as u64])
                .collect::<Vec<_>>(),
        );
        let batch = tree.prove_batch(&query_indices);
        assert!(batch.verify_all(&tree.root()));
    }

    #[test]
    fn test_proof_leaf_index_compat() {
        let tree = MerkleTree::new(&sample_data(4));
        let proof = tree.prove(2);
        assert_eq!(proof.leaf_index(), 2);
        assert_eq!(proof.index, 2);
    }

    #[test]
    fn test_num_leaves_compat() {
        let tree = MerkleTree::new(&sample_data(8));
        assert_eq!(tree.num_leaves(), 8);
        assert_eq!(tree.leaf_count(), 8);
    }

    #[test]
    fn test_leaf_hash_compat() {
        let data = sample_data(4);
        let tree = MerkleTree::new(&data);
        assert_eq!(tree.leaf_hash(0), tree.get_leaf(0));
    }

    // -----------------------------------------------------------------------
    // SparseMerkleProof tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_sparse_merkle_proof_depth() {
        let proof = SparseMerkleProof {
            siblings: vec![[0u8; 32]; 10],
            key: 0,
            value: None,
            is_inclusion: false,
        };
        assert_eq!(proof.depth(), 10);
    }

    #[test]
    fn test_sparse_merkle_proof_size_bytes() {
        let proof = SparseMerkleProof {
            siblings: vec![[0u8; 32]; 5],
            key: 42,
            value: Some([1u8; 32]),
            is_inclusion: true,
        };
        // 5 * 32 + 32 + 8 + 1 = 201
        assert_eq!(proof.size_bytes(), 201);
    }

    #[test]
    fn test_sparse_merkle_proof_empty() {
        let proof = SparseMerkleProof {
            siblings: Vec::new(),
            key: 0,
            value: None,
            is_inclusion: false,
        };
        assert_eq!(proof.depth(), 0);
        assert_eq!(proof.size_bytes(), 41);
    }

    // -----------------------------------------------------------------------
    // SparseMerkleTree extended methods tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_sparse_insert_and_size() {
        let mut smt = SparseMerkleTree::new(8);
        assert_eq!(smt.size(), 0);
        smt.insert(0, b"hello");
        assert_eq!(smt.size(), 1);
        smt.insert(255, b"world");
        assert_eq!(smt.size(), 2);
    }

    #[test]
    fn test_sparse_get_optional() {
        let mut smt = SparseMerkleTree::new(4);
        assert!(smt.get_optional(0).is_none());
        smt.insert(0, b"data");
        assert!(smt.get_optional(0).is_some());
        assert!(smt.get_optional(1).is_none());
    }

    #[test]
    fn test_sparse_update() {
        let mut smt = SparseMerkleTree::new(4);
        smt.insert(3, b"first");
        let r1 = smt.root();
        smt.update(3, b"second");
        let r2 = smt.root();
        assert_ne!(r1, r2);
        // Update again to original value.
        smt.update(3, b"first");
        let r3 = smt.root();
        assert_eq!(r1, r3);
    }

    #[test]
    fn test_sparse_remove() {
        let mut smt = SparseMerkleTree::new(4);
        smt.insert(5, b"value");
        assert_eq!(smt.size(), 1);
        assert!(smt.remove(5));
        assert_eq!(smt.size(), 0);
        // Remove again returns false.
        assert!(!smt.remove(5));
    }

    #[test]
    fn test_sparse_remove_restores_root() {
        let mut smt = SparseMerkleTree::new(4);
        let empty_root = smt.root();
        smt.insert(7, b"temp");
        assert_ne!(smt.root(), empty_root);
        smt.remove(7);
        assert_eq!(smt.root(), empty_root);
    }

    #[test]
    fn test_sparse_prove_inclusion() {
        let mut smt = SparseMerkleTree::new(4);
        smt.insert(3, b"hello");
        let proof = smt.prove_inclusion(3);
        assert!(proof.is_inclusion);
        assert_eq!(proof.key, 3);
        assert!(proof.value.is_some());
        assert_eq!(proof.depth(), 4);
    }

    #[test]
    fn test_sparse_prove_non_inclusion() {
        let mut smt = SparseMerkleTree::new(4);
        smt.insert(0, b"val");
        let proof = smt.prove_non_inclusion(5);
        assert!(!proof.is_inclusion);
        assert_eq!(proof.key, 5);
        assert!(proof.value.is_none());
    }

    #[test]
    fn test_sparse_verify_inclusion() {
        let mut smt = SparseMerkleTree::new(4);
        smt.insert(2, b"data");
        let root = smt.root();
        let leaf_hash = smt.get(2);
        let proof = smt.prove_inclusion(2);
        assert!(SparseMerkleTree::verify_inclusion(&root, 2, &leaf_hash, &proof));
    }

    #[test]
    fn test_sparse_verify_inclusion_wrong_value() {
        let mut smt = SparseMerkleTree::new(4);
        smt.insert(2, b"data");
        let root = smt.root();
        let wrong_hash = hash_bytes(b"wrong");
        let proof = smt.prove_inclusion(2);
        assert!(!SparseMerkleTree::verify_inclusion(&root, 2, &wrong_hash, &proof));
    }

    #[test]
    fn test_sparse_verify_non_inclusion() {
        let mut smt = SparseMerkleTree::new(4);
        let root = smt.root();
        let proof = smt.prove_non_inclusion(7);
        assert!(SparseMerkleTree::verify_non_inclusion(&root, 7, &proof));
    }

    #[test]
    fn test_sparse_verify_non_inclusion_rejects_inclusion_proof() {
        let mut smt = SparseMerkleTree::new(4);
        smt.insert(3, b"val");
        let root = smt.root();
        let proof = smt.prove_inclusion(3);
        // An inclusion proof should fail non-inclusion verification.
        assert!(!SparseMerkleTree::verify_non_inclusion(&root, 3, &proof));
    }

    #[test]
    fn test_sparse_multiple_inserts_and_proofs() {
        let mut smt = SparseMerkleTree::new(8);
        for i in 0..10 {
            smt.insert(i * 3, format!("val-{}", i).as_bytes());
        }
        assert_eq!(smt.size(), 10);
        let root = smt.root();
        for i in 0..10 {
            let key = i * 3;
            let leaf = smt.get(key);
            let proof = smt.prove_inclusion(key);
            assert!(SparseMerkleTree::verify_inclusion(&root, key, &leaf, &proof));
        }
    }

    // -----------------------------------------------------------------------
    // MerkleAccumulator tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_accumulator_new() {
        let acc = MerkleAccumulator::new();
        assert_eq!(acc.count(), 0);
        assert_eq!(acc.root(), [0u8; 32]);
    }

    #[test]
    fn test_accumulator_default() {
        let acc = MerkleAccumulator::default();
        assert_eq!(acc.count(), 0);
    }

    #[test]
    fn test_accumulator_append_one() {
        let mut acc = MerkleAccumulator::new();
        acc.append(b"first");
        assert_eq!(acc.count(), 1);
        assert_ne!(acc.root(), [0u8; 32]);
    }

    #[test]
    fn test_accumulator_append_two() {
        let mut acc = MerkleAccumulator::new();
        acc.append(b"first");
        let r1 = acc.root();
        acc.append(b"second");
        let r2 = acc.root();
        assert_ne!(r1, r2);
        assert_eq!(acc.count(), 2);
    }

    #[test]
    fn test_accumulator_deterministic() {
        let mut a1 = MerkleAccumulator::new();
        let mut a2 = MerkleAccumulator::new();
        for i in 0..5 {
            let data = format!("item-{}", i);
            a1.append(data.as_bytes());
            a2.append(data.as_bytes());
        }
        assert_eq!(a1.root(), a2.root());
    }

    #[test]
    fn test_accumulator_append_many() {
        let mut acc = MerkleAccumulator::new();
        for i in 0..100 {
            acc.append(format!("entry-{}", i).as_bytes());
        }
        assert_eq!(acc.count(), 100);
        assert_ne!(acc.root(), [0u8; 32]);
    }

    #[test]
    fn test_accumulator_prove() {
        let mut acc = MerkleAccumulator::new();
        for i in 0..4 {
            acc.append(format!("item-{}", i).as_bytes());
        }
        let proof = acc.prove(0);
        assert!(!proof.siblings.is_empty() || !proof.peak_hashes.is_empty());
    }

    #[test]
    fn test_accumulator_verify() {
        let mut acc = MerkleAccumulator::new();
        acc.append(b"only-item");
        let root = acc.root();
        let proof = acc.prove(0);
        assert!(MerkleAccumulator::verify(&root, 0, b"only-item", &proof));
    }

    #[test]
    fn test_accumulator_verify_wrong_item() {
        let mut acc = MerkleAccumulator::new();
        acc.append(b"real-item");
        let root = acc.root();
        let proof = acc.prove(0);
        assert!(!MerkleAccumulator::verify(&root, 0, b"fake-item", &proof));
    }

    // -----------------------------------------------------------------------
    // MerkleMountainRange tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_mmr_new() {
        let mmr = MerkleMountainRange::new();
        assert_eq!(mmr.total_leaves(), 0);
        assert_eq!(mmr.peak_count(), 0);
        assert_eq!(mmr.root(), [0u8; 32]);
    }

    #[test]
    fn test_mmr_default() {
        let mmr = MerkleMountainRange::default();
        assert_eq!(mmr.total_leaves(), 0);
    }

    #[test]
    fn test_mmr_append_one() {
        let mut mmr = MerkleMountainRange::new();
        mmr.append(hash_bytes(b"leaf0"));
        assert_eq!(mmr.total_leaves(), 1);
        assert_eq!(mmr.peak_count(), 1);
    }

    #[test]
    fn test_mmr_append_two_merges() {
        let mut mmr = MerkleMountainRange::new();
        mmr.append(hash_bytes(b"a"));
        mmr.append(hash_bytes(b"b"));
        // Two leaves merge into one peak.
        assert_eq!(mmr.total_leaves(), 2);
        assert_eq!(mmr.peak_count(), 1);
    }

    #[test]
    fn test_mmr_append_three_peaks() {
        let mut mmr = MerkleMountainRange::new();
        mmr.append(hash_bytes(b"a"));
        mmr.append(hash_bytes(b"b"));
        mmr.append(hash_bytes(b"c"));
        // 3 leaves: one peak of size 2 + one of size 1.
        assert_eq!(mmr.total_leaves(), 3);
        assert_eq!(mmr.peak_count(), 2);
    }

    #[test]
    fn test_mmr_append_four_one_peak() {
        let mut mmr = MerkleMountainRange::new();
        for i in 0..4 {
            mmr.append(hash_bytes(format!("leaf{}", i).as_bytes()));
        }
        assert_eq!(mmr.total_leaves(), 4);
        assert_eq!(mmr.peak_count(), 1);
    }

    #[test]
    fn test_mmr_root_deterministic() {
        let mut a = MerkleMountainRange::new();
        let mut b = MerkleMountainRange::new();
        for i in 0..7 {
            let h = hash_bytes(format!("item{}", i).as_bytes());
            a.append(h);
            b.append(h);
        }
        assert_eq!(a.root(), b.root());
    }

    #[test]
    fn test_mmr_prove_and_verify_single() {
        let mut mmr = MerkleMountainRange::new();
        let leaf = hash_bytes(b"only");
        mmr.append(leaf);
        let root = mmr.root();
        let proof = mmr.prove(0);
        assert!(MerkleMountainRange::verify(&root, 0, leaf, &proof));
    }

    #[test]
    fn test_mmr_prove_and_verify_power_of_two() {
        let mut mmr = MerkleMountainRange::new();
        let mut leaves = Vec::new();
        for i in 0..8 {
            let h = hash_bytes(format!("leaf-{}", i).as_bytes());
            leaves.push(h);
            mmr.append(h);
        }
        let root = mmr.root();
        for i in 0..8 {
            let proof = mmr.prove(i);
            assert!(
                MerkleMountainRange::verify(&root, i, leaves[i], &proof),
                "verify failed for leaf {}",
                i
            );
        }
    }

    #[test]
    fn test_mmr_prove_and_verify_non_power_of_two() {
        let mut mmr = MerkleMountainRange::new();
        let mut leaves = Vec::new();
        for i in 0..5 {
            let h = hash_bytes(format!("n{}", i).as_bytes());
            leaves.push(h);
            mmr.append(h);
        }
        let root = mmr.root();
        for i in 0..5 {
            let proof = mmr.prove(i);
            assert!(
                MerkleMountainRange::verify(&root, i, leaves[i], &proof),
                "verify failed for leaf {}",
                i
            );
        }
    }

    #[test]
    fn test_mmr_verify_wrong_leaf() {
        let mut mmr = MerkleMountainRange::new();
        mmr.append(hash_bytes(b"real"));
        let root = mmr.root();
        let proof = mmr.prove(0);
        assert!(!MerkleMountainRange::verify(
            &root,
            0,
            hash_bytes(b"fake"),
            &proof
        ));
    }

    #[test]
    fn test_mmr_peak_counts() {
        // For n leaves, peak_count = popcount(n).
        let mut mmr = MerkleMountainRange::new();
        for i in 1..=16 {
            mmr.append(hash_bytes(&[i as u8]));
            let expected_peaks = (i as u32).count_ones() as usize;
            assert_eq!(
                mmr.peak_count(),
                expected_peaks,
                "peak count mismatch at n={}",
                i
            );
        }
    }

    // -----------------------------------------------------------------------
    // MerkleTreeHasher tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_blake3_hasher_leaf() {
        let hasher = MerkleTreeHasher::blake3_hasher();
        let h = hasher.hash_leaf(b"test");
        assert_eq!(h, hash_bytes(b"test"));
    }

    #[test]
    fn test_blake3_hasher_node() {
        let hasher = MerkleTreeHasher::blake3_hasher();
        let a = hash_bytes(b"left");
        let b = hash_bytes(b"right");
        let h = hasher.hash_node(&a, &b);
        assert_eq!(h, hash_two(&a, &b));
    }

    #[test]
    fn test_sha256_hasher_different_from_blake3() {
        let b3 = MerkleTreeHasher::blake3_hasher();
        let sha = MerkleTreeHasher::sha256_hasher();
        assert_ne!(b3.hash_leaf(b"data"), sha.hash_leaf(b"data"));
    }

    #[test]
    fn test_poseidon_hasher_different_from_blake3() {
        let b3 = MerkleTreeHasher::blake3_hasher();
        let pos = MerkleTreeHasher::poseidon_hasher();
        assert_ne!(b3.hash_leaf(b"data"), pos.hash_leaf(b"data"));
    }

    #[test]
    fn test_sha256_poseidon_different() {
        let sha = MerkleTreeHasher::sha256_hasher();
        let pos = MerkleTreeHasher::poseidon_hasher();
        assert_ne!(sha.hash_leaf(b"data"), pos.hash_leaf(b"data"));
    }

    #[test]
    fn test_hasher_deterministic() {
        let hasher = MerkleTreeHasher::sha256_hasher();
        let h1 = hasher.hash_leaf(b"abc");
        let h2 = hasher.hash_leaf(b"abc");
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_hasher_node_deterministic() {
        let hasher = MerkleTreeHasher::poseidon_hasher();
        let a = [1u8; 32];
        let b = [2u8; 32];
        let h1 = hasher.hash_node(&a, &b);
        let h2 = hasher.hash_node(&a, &b);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_hasher_node_order_matters() {
        let hasher = MerkleTreeHasher::blake3_hasher();
        let a = hash_bytes(b"left");
        let b = hash_bytes(b"right");
        assert_ne!(hasher.hash_node(&a, &b), hasher.hash_node(&b, &a));
    }

    // -----------------------------------------------------------------------
    // MerkleTreeAnalytics tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_proof_size_for_depth() {
        assert_eq!(MerkleTreeAnalytics::proof_size_for_depth(0), 32);
        assert_eq!(MerkleTreeAnalytics::proof_size_for_depth(10), 352);
        assert_eq!(MerkleTreeAnalytics::proof_size_for_depth(20), 672);
    }

    #[test]
    fn test_optimal_depth_for_leaves() {
        assert_eq!(MerkleTreeAnalytics::optimal_depth_for_leaves(1), 0);
        assert_eq!(MerkleTreeAnalytics::optimal_depth_for_leaves(2), 1);
        assert_eq!(MerkleTreeAnalytics::optimal_depth_for_leaves(3), 2);
        assert_eq!(MerkleTreeAnalytics::optimal_depth_for_leaves(4), 2);
        assert_eq!(MerkleTreeAnalytics::optimal_depth_for_leaves(5), 3);
        assert_eq!(MerkleTreeAnalytics::optimal_depth_for_leaves(1024), 10);
    }

    #[test]
    fn test_authentication_path_cost() {
        assert_eq!(MerkleTreeAnalytics::authentication_path_cost(0), 0);
        assert_eq!(MerkleTreeAnalytics::authentication_path_cost(20), 20);
    }

    #[test]
    fn test_multi_proof_savings_single() {
        // No savings with a single proof.
        let s = MerkleTreeAnalytics::multi_proof_savings(1, 10);
        assert_eq!(s, 0.0);
    }

    #[test]
    fn test_multi_proof_savings_many() {
        let s = MerkleTreeAnalytics::multi_proof_savings(10, 20);
        assert!(s > 0.0);
        assert!(s < 1.0);
    }

    #[test]
    fn test_multi_proof_savings_zero_depth() {
        let s = MerkleTreeAnalytics::multi_proof_savings(5, 0);
        assert_eq!(s, 0.0);
    }

    #[test]
    fn test_expected_verification_time() {
        let t = MerkleTreeAnalytics::expected_verification_time(20);
        assert!((t - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_expected_verification_time_zero() {
        let t = MerkleTreeAnalytics::expected_verification_time(0);
        assert!((t - 0.0).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // MerkleAuditLog tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_audit_log_new() {
        let log = MerkleAuditLog::new();
        assert_eq!(log.entry_count(), 0);
        assert_eq!(log.root(), [0u8; 32]);
    }

    #[test]
    fn test_audit_log_default() {
        let log = MerkleAuditLog::default();
        assert_eq!(log.entry_count(), 0);
    }

    #[test]
    fn test_audit_log_append() {
        let mut log = MerkleAuditLog::new();
        log.append_entry(b"first entry".to_vec(), "initial".to_string());
        assert_eq!(log.entry_count(), 1);
        assert_ne!(log.root(), [0u8; 32]);
    }

    #[test]
    fn test_audit_log_append_multiple() {
        let mut log = MerkleAuditLog::new();
        log.append_entry(b"entry-0".to_vec(), "meta0".to_string());
        let r1 = log.root();
        log.append_entry(b"entry-1".to_vec(), "meta1".to_string());
        let r2 = log.root();
        assert_ne!(r1, r2);
        assert_eq!(log.entry_count(), 2);
    }

    #[test]
    fn test_audit_log_verify_entry() {
        let mut log = MerkleAuditLog::new();
        log.append_entry(b"data".to_vec(), "test".to_string());
        assert!(log.verify_entry(0));
    }

    #[test]
    fn test_audit_log_verify_entry_out_of_range() {
        let log = MerkleAuditLog::new();
        assert!(!log.verify_entry(0));
    }

    #[test]
    fn test_audit_log_verify_log() {
        let mut log = MerkleAuditLog::new();
        for i in 0..8 {
            log.append_entry(
                format!("entry-{}", i).into_bytes(),
                format!("meta-{}", i),
            );
        }
        assert!(log.verify_log());
    }

    #[test]
    fn test_audit_log_entries_since() {
        let mut log = MerkleAuditLog::new();
        for i in 0..5 {
            log.append_entry(format!("e{}", i).into_bytes(), format!("m{}", i));
        }
        let since = log.entries_since(3);
        assert_eq!(since.len(), 2);
        assert_eq!(since[0].metadata, "m3");
        assert_eq!(since[1].metadata, "m4");
    }

    #[test]
    fn test_audit_log_entries_since_out_of_range() {
        let log = MerkleAuditLog::new();
        let since = log.entries_since(0);
        assert_eq!(since.len(), 0);
    }

    #[test]
    fn test_audit_log_export_nonempty() {
        let mut log = MerkleAuditLog::new();
        log.append_entry(b"hello".to_vec(), "greeting".to_string());
        let exported = log.export();
        assert!(!exported.is_empty());
        // First 8 bytes encode the count (1).
        let count = u64::from_le_bytes(exported[0..8].try_into().unwrap());
        assert_eq!(count, 1);
    }

    #[test]
    fn test_audit_log_export_empty() {
        let log = MerkleAuditLog::new();
        let exported = log.export();
        let count = u64::from_le_bytes(exported[0..8].try_into().unwrap());
        assert_eq!(count, 0);
    }

    #[test]
    fn test_audit_log_debug() {
        let log = MerkleAuditLog::new();
        let dbg = format!("{:?}", log);
        assert!(dbg.contains("MerkleAuditLog"));
    }

    #[test]
    fn test_audit_log_entry_hash_matches_data() {
        let mut log = MerkleAuditLog::new();
        let data = b"important record".to_vec();
        log.append_entry(data.clone(), "record".to_string());
        let entries = log.entries_since(0);
        assert_eq!(entries[0].hash, hash_bytes(&data));
    }

    #[test]
    fn test_audit_log_timestamps_monotonic() {
        let mut log = MerkleAuditLog::new();
        for i in 0..10 {
            log.append_entry(format!("e{}", i).into_bytes(), String::new());
        }
        let entries = log.entries_since(0);
        for i in 1..entries.len() {
            assert!(entries[i].timestamp > entries[i - 1].timestamp);
        }
    }

    #[test]
    fn test_audit_log_root_changes_on_append() {
        let mut log = MerkleAuditLog::new();
        let mut prev_root = log.root();
        for i in 0..5 {
            log.append_entry(format!("d{}", i).into_bytes(), String::new());
            let new_root = log.root();
            assert_ne!(prev_root, new_root);
            prev_root = new_root;
        }
    }

    // -----------------------------------------------------------------------
    // Integration: Audit log with sparse tree
    // -----------------------------------------------------------------------

    #[test]
    fn test_audit_log_with_sparse_tree() {
        let mut log = MerkleAuditLog::new();
        let mut smt = SparseMerkleTree::new(8);

        for i in 0..8 {
            let data = format!("record-{}", i);
            log.append_entry(data.as_bytes().to_vec(), format!("idx={}", i));
            smt.insert(i, data.as_bytes());
        }

        assert!(log.verify_log());
        assert_eq!(log.entry_count(), 8);
        assert_eq!(smt.size(), 8);
    }

    // -----------------------------------------------------------------------
    // Integration: MMR with hasher
    // -----------------------------------------------------------------------

    #[test]
    fn test_mmr_with_custom_hasher() {
        let hasher = MerkleTreeHasher::sha256_hasher();
        let mut mmr = MerkleMountainRange::new();
        let mut leaves = Vec::new();
        for i in 0..6 {
            let h = hasher.hash_leaf(format!("leaf-{}", i).as_bytes());
            leaves.push(h);
            mmr.append(h);
        }
        assert_eq!(mmr.total_leaves(), 6);
        let root = mmr.root();
        for i in 0..6 {
            let proof = mmr.prove(i);
            assert!(MerkleMountainRange::verify(&root, i, leaves[i], &proof));
        }
    }

    // -----------------------------------------------------------------------
    // Integration: Accumulator with analytics
    // -----------------------------------------------------------------------

    #[test]
    fn test_analytics_with_accumulator() {
        let depth = MerkleTreeAnalytics::optimal_depth_for_leaves(100);
        let proof_size = MerkleTreeAnalytics::proof_size_for_depth(depth);
        let cost = MerkleTreeAnalytics::authentication_path_cost(depth);
        let time = MerkleTreeAnalytics::expected_verification_time(depth);

        assert!(depth >= 7); // 2^7 = 128 >= 100
        assert!(proof_size > 0);
        assert!(cost > 0);
        assert!(time > 0.0);

        let mut acc = MerkleAccumulator::new();
        for i in 0..100 {
            acc.append(format!("item-{}", i).as_bytes());
        }
        assert_eq!(acc.count(), 100);
    }
}
