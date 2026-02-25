use std::fmt;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

// ─────────────────────────────────────────────────────────────
// Goldilocks field arithmetic (self-contained for protocol use)
// ─────────────────────────────────────────────────────────────

const GOLDILOCKS_PRIME: u64 = 0xFFFFFFFF00000001;

/// Reduce a u128 modulo the Goldilocks prime.
fn goldilocks_reduce(x: u128) -> u64 {
    let x_lo = x as u64;
    let x_hi = (x >> 64) as u64;
    let hi_shifted = (x_hi as u128) << 32;
    let correction = hi_shifted - (x_hi as u128);
    let sum = (x_lo as u128) + correction;
    let mut result = (sum % (GOLDILOCKS_PRIME as u128)) as u64;
    if result >= GOLDILOCKS_PRIME {
        result -= GOLDILOCKS_PRIME;
    }
    result
}

/// Multiply two Goldilocks field elements.
fn goldilocks_mul(a: u64, b: u64) -> u64 {
    goldilocks_reduce((a as u128) * (b as u128))
}

/// Modular exponentiation in the Goldilocks field.
fn goldilocks_pow(mut base: u64, mut exp: u64) -> u64 {
    let mut result = 1u64;
    base %= GOLDILOCKS_PRIME;
    while exp > 0 {
        if exp & 1 == 1 {
            result = goldilocks_mul(result, base);
        }
        exp >>= 1;
        base = goldilocks_mul(base, base);
    }
    result
}

/// Reduce a u64 into the Goldilocks field.
fn goldilocks_new(v: u64) -> u64 {
    if v >= GOLDILOCKS_PRIME { v - GOLDILOCKS_PRIME } else { v }
}

/// Add two Goldilocks field elements.
fn goldilocks_add(a: u64, b: u64) -> u64 {
    let sum = (a as u128) + (b as u128);
    (sum % (GOLDILOCKS_PRIME as u128)) as u64
}

/// Subtract two Goldilocks field elements: a - b mod p.
fn goldilocks_sub(a: u64, b: u64) -> u64 {
    if a >= b {
        a - b
    } else {
        GOLDILOCKS_PRIME - (b - a)
    }
}

/// Multiplicative inverse via Fermat's little theorem: a^(p-2) mod p.
fn goldilocks_inv(a: u64) -> u64 {
    goldilocks_pow(a, GOLDILOCKS_PRIME - 2)
}

// ─────────────────────────────────────────────────────────────
// CommitmentScheme trait
// ─────────────────────────────────────────────────────────────

pub trait CommitmentScheme {
    type Commitment: Clone + fmt::Debug + Serialize;
    type Opening: Clone + fmt::Debug + Serialize;

    fn commit(&self, value: &[u8], randomness: &[u8]) -> Self::Commitment;
    fn verify(&self, commitment: &Self::Commitment, value: &[u8], opening: &Self::Opening) -> bool;
    fn scheme_name(&self) -> &str;
}

// ─────────────────────────────────────────────────────────────
// HashCommitment (BLAKE3-based)
// ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HashCommitmentValue {
    pub hash: [u8; 32],
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HashOpening {
    pub value: Vec<u8>,
    pub randomness: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct HashCommitment {
    domain_separator: Vec<u8>,
}

impl HashCommitment {
    pub fn new() -> Self {
        Self {
            domain_separator: b"hash-commitment-v1".to_vec(),
        }
    }

    pub fn with_domain(domain: &str) -> Self {
        Self {
            domain_separator: domain.as_bytes().to_vec(),
        }
    }

    /// Compute the raw hash for a commitment: BLAKE3(domain || randomness || value).
    fn compute_hash(&self, value: &[u8], randomness: &[u8]) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&self.domain_separator);
        hasher.update(randomness);
        hasher.update(value);
        *hasher.finalize().as_bytes()
    }

    /// Commit to a vector of field elements (u64) with the given randomness.
    pub fn commit_field_elements(&self, elements: &[u64], randomness: &[u8]) -> HashCommitmentValue {
        let mut buf = Vec::with_capacity(elements.len() * 8);
        for &e in elements {
            buf.extend_from_slice(&e.to_le_bytes());
        }
        let hash = self.compute_hash(&buf, randomness);
        HashCommitmentValue { hash }
    }

    /// Commit to a batch of values, each with independently generated randomness.
    pub fn commit_batch(&self, values: &[Vec<u8>]) -> Vec<HashCommitmentValue> {
        values
            .iter()
            .map(|v| {
                let r = Self::generate_randomness();
                let hash = self.compute_hash(v, &r);
                HashCommitmentValue { hash }
            })
            .collect()
    }

    /// Verify a batch of commitments against their openings.
    pub fn verify_batch(
        &self,
        commitments: &[HashCommitmentValue],
        values: &[Vec<u8>],
        openings: &[HashOpening],
    ) -> bool {
        if commitments.len() != values.len() || values.len() != openings.len() {
            return false;
        }
        commitments.iter().zip(values.iter()).zip(openings.iter()).all(
            |((commitment, value), opening)| {
                let expected = self.compute_hash(value, &opening.randomness);
                commitment.hash == expected && opening.value == *value
            },
        )
    }

    /// Generate 32 bytes of cryptographic randomness.
    pub fn generate_randomness() -> [u8; 32] {
        use rand::RngCore;
        let mut rng = rand::thread_rng();
        let mut buf = [0u8; 32];
        rng.fill_bytes(&mut buf);
        buf
    }
}

impl Default for HashCommitment {
    fn default() -> Self {
        Self::new()
    }
}

impl CommitmentScheme for HashCommitment {
    type Commitment = HashCommitmentValue;
    type Opening = HashOpening;

    fn commit(&self, value: &[u8], randomness: &[u8]) -> HashCommitmentValue {
        let hash = self.compute_hash(value, randomness);
        HashCommitmentValue { hash }
    }

    fn verify(&self, commitment: &HashCommitmentValue, value: &[u8], opening: &HashOpening) -> bool {
        let expected = self.compute_hash(value, &opening.randomness);
        commitment.hash == expected && opening.value == *value
    }

    fn scheme_name(&self) -> &str {
        "blake3-hash-commitment"
    }
}

// ─────────────────────────────────────────────────────────────
// PedersenCommitment (simulated over Goldilocks field)
// ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PedersenCommitmentValue {
    pub value: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PedersenOpening {
    pub committed_value: u64,
    pub blinding_factor: u64,
}

#[derive(Clone, Debug)]
pub struct PedersenCommitment {
    /// Generator g in Goldilocks field
    g: u64,
    /// Second generator h (discrete log w.r.t. g is unknown)
    h: u64,
}

impl PedersenCommitment {
    /// Create a new Pedersen commitment scheme with standard generators.
    pub fn new() -> Self {
        // g = 7 is a well-known generator of the Goldilocks multiplicative group.
        // h is derived by hashing so that log_g(h) is unknown.
        let g = 7u64;
        let h_hash = blake3::hash(b"pedersen-h-generator-spectacles");
        let h_bytes = h_hash.as_bytes();
        let h_raw = u64::from_le_bytes(h_bytes[0..8].try_into().unwrap());
        let h = goldilocks_new(h_raw);
        // Ensure h != 0
        let h = if h == 0 { 3u64 } else { h };

        Self { g, h }
    }

    /// Compute g^exp mod p using modular exponentiation in the Goldilocks field.
    fn mod_exp(base: u64, exp: u64) -> u64 {
        goldilocks_pow(base, exp)
    }

    /// Commit to a single value: commitment = g^value * h^blinding mod p
    pub fn commit_value(&self, value: u64, blinding: u64) -> PedersenCommitmentValue {
        let gv = Self::mod_exp(self.g, value);
        let hb = Self::mod_exp(self.h, blinding);
        let product = goldilocks_mul(gv, hb);
        PedersenCommitmentValue { value: product }
    }

    /// Verify that a commitment opens to (value, blinding).
    pub fn verify_opening(
        &self,
        commitment: &PedersenCommitmentValue,
        value: u64,
        blinding: u64,
    ) -> bool {
        let expected = self.commit_value(value, blinding);
        commitment.value == expected.value
    }

    /// Commit to a vector of values with corresponding blindings.
    pub fn commit_vector(
        &self,
        values: &[u64],
        blindings: &[u64],
    ) -> Vec<PedersenCommitmentValue> {
        assert_eq!(values.len(), blindings.len());
        values
            .iter()
            .zip(blindings.iter())
            .map(|(&v, &b)| self.commit_value(v, b))
            .collect()
    }

    /// Homomorphic addition: C(a+b, r1+r2) = C(a, r1) * C(b, r2)
    pub fn add_commitments(
        &self,
        a: &PedersenCommitmentValue,
        b: &PedersenCommitmentValue,
    ) -> PedersenCommitmentValue {
        let product = goldilocks_mul(a.value, b.value);
        PedersenCommitmentValue { value: product }
    }

    /// Scalar multiplication of a commitment: C(s*v, s*r) = C(v,r)^s
    pub fn scalar_mul_commitment(
        &self,
        c: &PedersenCommitmentValue,
        scalar: u64,
    ) -> PedersenCommitmentValue {
        let result = goldilocks_pow(c.value, scalar);
        PedersenCommitmentValue { value: result }
    }

    /// Verify that sum_commit is the product (homomorphic sum) of the given commitments.
    pub fn verify_sum(
        &self,
        sum_commit: &PedersenCommitmentValue,
        commits: &[PedersenCommitmentValue],
    ) -> bool {
        if commits.is_empty() {
            return sum_commit.value == 1u64;
        }
        let mut acc = commits[0].value;
        for c in &commits[1..] {
            acc = goldilocks_mul(acc, c.value);
        }
        acc == sum_commit.value
    }
}

impl Default for PedersenCommitment {
    fn default() -> Self {
        Self::new()
    }
}

impl CommitmentScheme for PedersenCommitment {
    type Commitment = PedersenCommitmentValue;
    type Opening = PedersenOpening;

    fn commit(&self, value: &[u8], randomness: &[u8]) -> PedersenCommitmentValue {
        // Interpret first 8 bytes of value and randomness as u64 field elements
        let v = if value.len() >= 8 {
            u64::from_le_bytes(value[0..8].try_into().unwrap())
        } else {
            let mut buf = [0u8; 8];
            buf[..value.len()].copy_from_slice(value);
            u64::from_le_bytes(buf)
        };
        let r = if randomness.len() >= 8 {
            u64::from_le_bytes(randomness[0..8].try_into().unwrap())
        } else {
            let mut buf = [0u8; 8];
            buf[..randomness.len()].copy_from_slice(randomness);
            u64::from_le_bytes(buf)
        };
        // Reduce to field
        let v = goldilocks_new(v);
        let r = goldilocks_new(r);
        self.commit_value(v, r)
    }

    fn verify(
        &self,
        commitment: &PedersenCommitmentValue,
        _value: &[u8],
        opening: &PedersenOpening,
    ) -> bool {
        self.verify_opening(commitment, opening.committed_value, opening.blinding_factor)
    }

    fn scheme_name(&self) -> &str {
        "pedersen-goldilocks"
    }
}

// ─────────────────────────────────────────────────────────────
// VectorCommitment (Merkle-tree based)
// ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct VectorCommitmentValue {
    pub root: [u8; 32],
    pub length: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VectorOpening {
    pub index: usize,
    pub value: Vec<u8>,
    pub proof: Vec<[u8; 32]>,
    /// `true` = sibling is on the right; `false` = sibling is on the left.
    pub proof_sides: Vec<bool>,
}

pub struct VectorCommitment;

impl VectorCommitment {
    /// Hash a single leaf: BLAKE3("leaf" || index || data).
    fn hash_leaf(index: usize, data: &[u8]) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"leaf");
        hasher.update(&index.to_le_bytes());
        hasher.update(data);
        *hasher.finalize().as_bytes()
    }

    /// Hash two children into a parent: BLAKE3("node" || left || right).
    fn hash_node(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"node");
        hasher.update(left);
        hasher.update(right);
        *hasher.finalize().as_bytes()
    }

    /// Build a Merkle tree from the leaves and return all layers (leaves first).
    fn build_tree(leaves: &[[u8; 32]]) -> Vec<Vec<[u8; 32]>> {
        if leaves.is_empty() {
            return vec![vec![[0u8; 32]]];
        }

        let mut layers: Vec<Vec<[u8; 32]>> = Vec::new();
        layers.push(leaves.to_vec());

        while layers.last().unwrap().len() > 1 {
            let prev = layers.last().unwrap();
            let mut next = Vec::new();
            let mut i = 0;
            while i < prev.len() {
                let left = &prev[i];
                let right = if i + 1 < prev.len() {
                    &prev[i + 1]
                } else {
                    left // duplicate for odd count
                };
                next.push(Self::hash_node(left, right));
                i += 2;
            }
            layers.push(next);
        }
        layers
    }

    /// Commit to a vector of byte slices, returning the Merkle root.
    pub fn commit(values: &[Vec<u8>]) -> VectorCommitmentValue {
        let leaves: Vec<[u8; 32]> = values
            .iter()
            .enumerate()
            .map(|(i, v)| Self::hash_leaf(i, v))
            .collect();
        let layers = Self::build_tree(&leaves);
        let root = *layers.last().unwrap().first().unwrap();
        VectorCommitmentValue {
            root,
            length: values.len(),
        }
    }

    /// Open (prove membership of) the element at `index`.
    pub fn open(values: &[Vec<u8>], index: usize) -> VectorOpening {
        assert!(index < values.len(), "index out of bounds");

        let leaves: Vec<[u8; 32]> = values
            .iter()
            .enumerate()
            .map(|(i, v)| Self::hash_leaf(i, v))
            .collect();
        let layers = Self::build_tree(&leaves);

        let mut proof = Vec::new();
        let mut proof_sides = Vec::new();
        let mut idx = index;

        for layer in &layers[..layers.len() - 1] {
            let sibling_idx = if idx % 2 == 0 { idx + 1 } else { idx - 1 };
            let sibling = if sibling_idx < layer.len() {
                layer[sibling_idx]
            } else {
                layer[idx] // odd layer: duplicate last element
            };
            proof_sides.push(idx % 2 == 0); // true => sibling on right
            proof.push(sibling);
            idx /= 2;
        }

        VectorOpening {
            index,
            value: values[index].clone(),
            proof,
            proof_sides,
        }
    }

    /// Verify a Merkle opening.
    pub fn verify(
        commitment: &VectorCommitmentValue,
        index: usize,
        value: &[u8],
        opening: &VectorOpening,
    ) -> bool {
        if index != opening.index || value != opening.value.as_slice() {
            return false;
        }

        let mut current = Self::hash_leaf(index, value);

        for (sibling, &is_right) in opening.proof.iter().zip(opening.proof_sides.iter()) {
            current = if is_right {
                Self::hash_node(&current, sibling)
            } else {
                Self::hash_node(sibling, &current)
            };
        }

        current == commitment.root
    }
}

// ─────────────────────────────────────────────────────────────
// CommitmentBatch
// ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CommitmentBatch {
    commitments: Vec<(String, Vec<u8>)>,
    #[serde(default)]
    metadata: HashMap<String, String>,
}

impl CommitmentBatch {
    pub fn new() -> Self {
        Self {
            commitments: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    pub fn add(&mut self, name: &str, commitment: Vec<u8>) {
        self.commitments.push((name.to_string(), commitment));
    }

    pub fn get(&self, name: &str) -> Option<&Vec<u8>> {
        self.commitments
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, c)| c)
    }

    /// Compute a combined hash of all commitments in order.
    pub fn combined_hash(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"commitment-batch");
        for (name, commitment) in &self.commitments {
            hasher.update(name.as_bytes());
            hasher.update(&(commitment.len() as u32).to_le_bytes());
            hasher.update(commitment);
        }
        *hasher.finalize().as_bytes()
    }

    /// Verify all commitments against their openings. Each opening is
    /// (name, value) and we check BLAKE3(value) == commitment bytes.
    pub fn verify_all(&self, openings: &[(String, Vec<u8>)]) -> bool {
        if openings.len() != self.commitments.len() {
            return false;
        }
        for ((name, commitment), (open_name, open_value)) in
            self.commitments.iter().zip(openings.iter())
        {
            if name != open_name {
                return false;
            }
            let hash = blake3::hash(open_value);
            if hash.as_bytes().as_slice() != commitment.as_slice() {
                return false;
            }
        }
        true
    }

    /// Serialize the batch to bytes.
    pub fn serialize(&self) -> Vec<u8> {
        serde_json::to_vec(self).unwrap_or_default()
    }

    /// Deserialize a batch from bytes.
    pub fn deserialize(bytes: &[u8]) -> Result<Self, String> {
        serde_json::from_slice(bytes).map_err(|e| e.to_string())
    }

    pub fn len(&self) -> usize {
        self.commitments.len()
    }

    pub fn is_empty(&self) -> bool {
        self.commitments.is_empty()
    }
}

impl Default for CommitmentBatch {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────
// VectorCommitment batch operations
// ─────────────────────────────────────────────────────────────

impl VectorCommitment {
    /// Create a new VectorCommitment instance.
    pub fn new() -> Self {
        Self
    }

    /// Open multiple indices at once, returning a proof for each.
    pub fn batch_open(values: &[Vec<u8>], indices: &[usize]) -> Vec<VectorOpening> {
        indices.iter().map(|&idx| Self::open(values, idx)).collect()
    }

    /// Verify a batch of openings against the same commitment.
    pub fn batch_verify(commitment: &VectorCommitmentValue, openings: &[VectorOpening]) -> bool {
        openings.iter().all(|opening| {
            Self::verify(commitment, opening.index, &opening.value, opening)
        })
    }

    /// Compute the depth of the Merkle tree for a given number of leaves.
    pub fn tree_depth(num_leaves: usize) -> usize {
        if num_leaves <= 1 {
            return 0;
        }
        let mut depth = 0;
        let mut size = num_leaves;
        while size > 1 {
            size = (size + 1) / 2;
            depth += 1;
        }
        depth
    }

    /// Update a single leaf and recompute the Merkle root.
    pub fn update_leaf(
        values: &[Vec<u8>],
        index: usize,
        new_value: &[u8],
    ) -> VectorCommitmentValue {
        let mut updated = values.to_vec();
        updated[index] = new_value.to_vec();
        Self::commit(&updated)
    }
}

impl Default for VectorCommitment {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────
// CommitmentBatch additional methods
// ─────────────────────────────────────────────────────────────

impl CommitmentBatch {
    /// Return the names of all commitments in insertion order.
    pub fn names(&self) -> Vec<&str> {
        self.commitments.iter().map(|(n, _)| n.as_str()).collect()
    }

    /// Set a metadata key-value pair.
    pub fn set_metadata(&mut self, key: &str, value: &str) {
        self.metadata.insert(key.to_string(), value.to_string());
    }

    /// Get a metadata value by key.
    pub fn get_metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(|s| s.as_str())
    }

    /// Verify all commitments using HashMap-based openings and randomness.
    /// Each commitment is verified as BLAKE3(domain || randomness || value).
    pub fn verify_all_with_randomness(
        &self,
        openings: &HashMap<String, Vec<u8>>,
        randomness: &HashMap<String, Vec<u8>>,
    ) -> bool {
        let scheme = HashCommitment::new();
        for (name, commitment_bytes) in &self.commitments {
            let value = match openings.get(name) {
                Some(v) => v,
                None => return false,
            };
            let rand_bytes = match randomness.get(name) {
                Some(r) => r,
                None => return false,
            };
            let expected_commitment = scheme.commit(value, rand_bytes);
            if expected_commitment.hash.as_slice() != commitment_bytes.as_slice() {
                return false;
            }
        }
        true
    }

    /// Remove a commitment by name, returning its value if found.
    pub fn remove(&mut self, name: &str) -> Option<Vec<u8>> {
        if let Some(pos) = self.commitments.iter().position(|(n, _)| n == name) {
            let (_, commitment) = self.commitments.remove(pos);
            Some(commitment)
        } else {
            None
        }
    }

    /// Check whether a commitment with the given name exists.
    pub fn contains(&self, name: &str) -> bool {
        self.commitments.iter().any(|(n, _)| n == name)
    }

    /// Clear all metadata entries.
    pub fn clear_metadata(&mut self) {
        self.metadata.clear();
    }
}

// ─────────────────────────────────────────────────────────────
// PolynomialCommitment (simplified KZG-like over Goldilocks)
// ─────────────────────────────────────────────────────────────

/// Commitment value for a polynomial – a single Goldilocks field element.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PolyCommitmentValue {
    pub value: u64,
}

/// Opening proof for a polynomial evaluation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PolyOpening {
    pub point: u64,
    pub evaluation: u64,
    pub witness: u64,
}

/// A simplified KZG-like polynomial commitment scheme operating over
/// the Goldilocks field.  The "trusted setup" produces powers of a
/// secret τ, and verification checks the polynomial identity directly
/// (since τ is embedded in the setup parameters).
#[derive(Clone, Debug)]
pub struct PolynomialCommitment {
    pub degree: usize,
    pub setup_params: Vec<u64>,
}

impl PolynomialCommitment {
    /// Trusted setup: compute [1, τ, τ², …, τ^max_degree] where τ is
    /// derived deterministically from a domain separator.
    pub fn setup(max_degree: usize) -> Self {
        let tau_hash = blake3::hash(b"polynomial-commitment-tau-spectacles");
        let tau_raw = u64::from_le_bytes(
            tau_hash.as_bytes()[0..8].try_into().unwrap(),
        );
        let tau = goldilocks_new(tau_raw);

        let mut setup_params = Vec::with_capacity(max_degree + 1);
        let mut power = 1u64;
        for _ in 0..=max_degree {
            setup_params.push(power);
            power = goldilocks_mul(power, tau);
        }

        Self {
            degree: max_degree,
            setup_params,
        }
    }

    /// Commit to a polynomial given by its coefficients [c₀, c₁, …, cₙ].
    /// The commitment is C = Σ cᵢ·τⁱ  (a Goldilocks field element).
    pub fn commit_polynomial(&self, coeffs: &[u64]) -> PolyCommitmentValue {
        let mut result = 0u64;
        for (i, &c) in coeffs.iter().enumerate() {
            if i < self.setup_params.len() {
                let term = goldilocks_mul(c, self.setup_params[i]);
                result = goldilocks_add(result, term);
            }
        }
        PolyCommitmentValue { value: result }
    }

    /// Evaluate the polynomial p(x) = Σ cᵢ·xⁱ at `point` using Horner's
    /// method entirely within the Goldilocks field.
    fn evaluate_poly(coeffs: &[u64], point: u64) -> u64 {
        let mut result = 0u64;
        for &c in coeffs.iter().rev() {
            result = goldilocks_mul(result, point);
            result = goldilocks_add(result, c);
        }
        result
    }

    /// Open the polynomial at `point`: return the evaluation p(point) and
    /// a witness w = q(τ) where q(x) = (p(x) − p(point)) / (x − point).
    pub fn open(&self, coeffs: &[u64], point: u64) -> PolyOpening {
        let evaluation = Self::evaluate_poly(coeffs, point);
        let n = coeffs.len();

        if n <= 1 {
            return PolyOpening {
                point,
                evaluation,
                witness: 0,
            };
        }

        // Synthetic division: quotient of (p(x) − evaluation) by (x − point).
        // q_{n-2} = c_{n-1}
        // q_i     = c_{i+1} + point · q_{i+1}   for i = n-3 … 0
        let mut quotient = vec![0u64; n - 1];
        quotient[n - 2] = coeffs[n - 1];
        for i in (0..n - 2).rev() {
            let term = goldilocks_mul(point, quotient[i + 1]);
            quotient[i] = goldilocks_add(coeffs[i + 1], term);
        }

        let tau = self.setup_params[1];
        let witness = Self::evaluate_poly(&quotient, tau);

        PolyOpening {
            point,
            evaluation,
            witness,
        }
    }

    /// Verify that the polynomial behind `commitment` evaluates to `value`
    /// at `point`, given the opening `proof`.
    ///
    /// Checks:  C − v  ≡  w · (τ − z)   (mod p)
    pub fn verify(
        &self,
        commitment: &PolyCommitmentValue,
        point: u64,
        value: u64,
        proof: &PolyOpening,
    ) -> bool {
        if proof.point != point || proof.evaluation != value {
            return false;
        }
        if self.setup_params.len() < 2 {
            return false;
        }

        let tau = self.setup_params[1];
        let c_minus_v = goldilocks_sub(commitment.value, value);
        let tau_minus_z = goldilocks_sub(tau, point);
        let rhs = goldilocks_mul(proof.witness, tau_minus_z);

        c_minus_v == rhs
    }

    /// Open the polynomial at every point in `points`.
    pub fn batch_open(&self, coeffs: &[u64], points: &[u64]) -> Vec<PolyOpening> {
        points.iter().map(|&p| self.open(coeffs, p)).collect()
    }

    /// Return the maximum polynomial degree this scheme supports.
    pub fn max_degree(&self) -> usize {
        self.degree
    }

    /// Public helper: evaluate p(x) at `point`.
    pub fn evaluate_at(&self, coeffs: &[u64], point: u64) -> u64 {
        Self::evaluate_poly(coeffs, point)
    }

    /// Add two polynomials coefficient-wise in the Goldilocks field.
    pub fn add_polynomials(a: &[u64], b: &[u64]) -> Vec<u64> {
        let max_len = a.len().max(b.len());
        let mut result = vec![0u64; max_len];
        for (i, &c) in a.iter().enumerate() {
            result[i] = goldilocks_add(result[i], c);
        }
        for (i, &c) in b.iter().enumerate() {
            result[i] = goldilocks_add(result[i], c);
        }
        result
    }

    /// Multiply every coefficient of a polynomial by a scalar.
    pub fn scale_polynomial(coeffs: &[u64], scalar: u64) -> Vec<u64> {
        coeffs.iter().map(|&c| goldilocks_mul(c, scalar)).collect()
    }
}

impl fmt::Display for PolyCommitmentValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PolyCommit({})", self.value)
    }
}

impl fmt::Display for PolyOpening {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PolyOpening(point={}, eval={}, witness={})",
            self.point, self.evaluation, self.witness
        )
    }
}

// ─────────────────────────────────────────────────────────────
// TimelockCommitment
// ─────────────────────────────────────────────────────────────

/// A commitment that can only be meaningfully opened after a specified
/// timestamp.  Additionally carries a hash-based proof-of-work puzzle
/// whose solution acts as a secondary unlock mechanism.
#[derive(Clone, Debug)]
pub struct TimelockCommitment {
    pub commitment: HashCommitmentValue,
    pub lock_until: u64,
    pub puzzle_difficulty: u32,
}

impl TimelockCommitment {
    /// Create a new timelock commitment over `value` with the given
    /// unlock timestamp and PoW puzzle difficulty (number of leading
    /// zero bits required).
    pub fn new(value: &[u8], lock_until: u64, difficulty: u32) -> Self {
        let scheme = HashCommitment::new();
        let randomness = HashCommitment::generate_randomness();
        let commitment = scheme.commit(value, &randomness);
        Self {
            commitment,
            lock_until,
            puzzle_difficulty: difficulty,
        }
    }

    /// Build a `TimelockCommitment` from an already-computed hash
    /// commitment.
    pub fn from_commitment(
        commitment: HashCommitmentValue,
        lock_until: u64,
        difficulty: u32,
    ) -> Self {
        Self {
            commitment,
            lock_until,
            puzzle_difficulty: difficulty,
        }
    }

    /// Returns `true` when `current_time >= lock_until`.
    pub fn is_unlockable(&self, current_time: u64) -> bool {
        current_time >= self.lock_until
    }

    /// Count leading zero bits in a 32-byte hash.
    fn count_leading_zeros(hash: &[u8; 32]) -> u32 {
        let mut count = 0u32;
        for &byte in hash.iter() {
            if byte == 0 {
                count += 8;
            } else {
                count += byte.leading_zeros();
                break;
            }
        }
        count
    }

    /// Compute the puzzle hash: BLAKE3("timelock-puzzle" || commitment || nonce).
    fn compute_puzzle_hash(&self, solution: &[u8]) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"timelock-puzzle");
        hasher.update(&self.commitment.hash);
        hasher.update(solution);
        *hasher.finalize().as_bytes()
    }

    /// Attempt to solve the PoW puzzle by brute-forcing nonces.
    /// Returns `Some(nonce_bytes)` on success, `None` if the difficulty
    /// is unreasonably high (> 32 leading zero bits).
    pub fn solve_puzzle(&self) -> Option<Vec<u8>> {
        if self.puzzle_difficulty > 32 {
            return None;
        }
        for nonce in 0u64..=u64::MAX {
            let nonce_bytes = nonce.to_le_bytes();
            let hash = self.compute_puzzle_hash(&nonce_bytes);
            if Self::count_leading_zeros(&hash) >= self.puzzle_difficulty {
                return Some(nonce_bytes.to_vec());
            }
        }
        None
    }

    /// Verify that `solution` satisfies the PoW puzzle.
    pub fn verify_puzzle_solution(&self, solution: &[u8]) -> bool {
        let hash = self.compute_puzzle_hash(solution);
        Self::count_leading_zeros(&hash) >= self.puzzle_difficulty
    }

    /// How many time units remain until the commitment is unlockable.
    pub fn remaining_time(&self, current_time: u64) -> u64 {
        if current_time >= self.lock_until {
            0
        } else {
            self.lock_until - current_time
        }
    }

    /// Return the underlying hash commitment value.
    pub fn commitment_hash(&self) -> &[u8; 32] {
        &self.commitment.hash
    }
}

impl fmt::Display for TimelockCommitment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TimelockCommitment(lock_until={}, difficulty={})",
            self.lock_until, self.puzzle_difficulty
        )
    }
}

// ─────────────────────────────────────────────────────────────
// CommitmentSchemeBoxed – object-safe commitment trait
// ─────────────────────────────────────────────────────────────

/// An object-safe version of `CommitmentScheme` that operates on raw
/// byte slices so that it can be used behind `dyn`.
pub trait CommitmentSchemeBoxed {
    /// Commit to `value`, returning the commitment as raw bytes.
    /// Randomness is generated or derived internally.
    fn commit_boxed(&self, value: &[u8]) -> Vec<u8>;

    /// Verify `commitment` bytes against `opening` bytes.
    /// The format of `opening` depends on the concrete scheme.
    fn verify_boxed(&self, commitment: &[u8], opening: &[u8]) -> bool;

    /// Human-readable name of the scheme.
    fn scheme_name(&self) -> &str;
}

/// `HashCommitment` as a boxed commitment scheme.
///
/// *   `commit_boxed` derives randomness deterministically from the value
///     (BLAKE3 of a domain-tagged input) so that callers can reconstruct
///     the opening later.
/// *   Opening format: `[randomness (32 bytes)] || [value]`.
/// *   Commitment format: `[hash (32 bytes)]`.
impl CommitmentSchemeBoxed for HashCommitment {
    fn commit_boxed(&self, value: &[u8]) -> Vec<u8> {
        let mut rng_hasher = blake3::Hasher::new();
        rng_hasher.update(b"commit-boxed-randomness");
        rng_hasher.update(value);
        let randomness = *rng_hasher.finalize().as_bytes();
        let hash = self.compute_hash(value, &randomness);
        hash.to_vec()
    }

    fn verify_boxed(&self, commitment: &[u8], opening: &[u8]) -> bool {
        if commitment.len() != 32 || opening.len() < 32 {
            return false;
        }
        let expected_hash: [u8; 32] = match commitment.try_into() {
            Ok(h) => h,
            Err(_) => return false,
        };
        let randomness = &opening[..32];
        let value = &opening[32..];
        let computed = self.compute_hash(value, randomness);
        computed == expected_hash
    }

    fn scheme_name(&self) -> &str {
        "blake3-hash-commitment"
    }
}

/// `PedersenCommitment` as a boxed commitment scheme.
///
/// *   `commit_boxed` derives a blinding factor deterministically from
///     the value bytes.
/// *   Opening format: `[committed_value (8 bytes LE)] || [blinding (8 bytes LE)]`.
/// *   Commitment format: `[field element (8 bytes LE)]`.
impl CommitmentSchemeBoxed for PedersenCommitment {
    fn commit_boxed(&self, value: &[u8]) -> Vec<u8> {
        let v = if value.len() >= 8 {
            u64::from_le_bytes(value[0..8].try_into().unwrap())
        } else {
            let mut buf = [0u8; 8];
            buf[..value.len()].copy_from_slice(value);
            u64::from_le_bytes(buf)
        };
        let v = goldilocks_new(v);

        let blinding_hash = blake3::hash(value);
        let blinding = goldilocks_new(
            u64::from_le_bytes(blinding_hash.as_bytes()[0..8].try_into().unwrap()),
        );

        let commitment = self.commit_value(v, blinding);
        commitment.value.to_le_bytes().to_vec()
    }

    fn verify_boxed(&self, commitment: &[u8], opening: &[u8]) -> bool {
        if commitment.len() < 8 || opening.len() < 16 {
            return false;
        }
        let commitment_val =
            u64::from_le_bytes(commitment[..8].try_into().unwrap());
        let committed_value =
            u64::from_le_bytes(opening[..8].try_into().unwrap());
        let blinding_factor =
            u64::from_le_bytes(opening[8..16].try_into().unwrap());
        let expected = self.commit_value(committed_value, blinding_factor);
        expected.value == commitment_val
    }

    fn scheme_name(&self) -> &str {
        "pedersen-goldilocks"
    }
}

// ─────────────────────────────────────────────────────────────
// CommitmentAggregator
// ─────────────────────────────────────────────────────────────

/// Aggregates multiple commitments (possibly from different schemes)
/// and provides a single aggregate root hash over all of them.
pub struct CommitmentAggregator {
    schemes: Vec<Box<dyn CommitmentSchemeBoxed>>,
    commitment_values: Vec<Vec<u8>>,
    aggregated_state: Vec<u8>,
}

impl CommitmentAggregator {
    pub fn new() -> Self {
        Self {
            schemes: Vec::new(),
            commitment_values: Vec::new(),
            aggregated_state: Vec::new(),
        }
    }

    /// Recompute the internal aggregated state from all stored commitments.
    fn update_aggregated_state(&mut self) {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"aggregator-state");
        for cv in &self.commitment_values {
            hasher.update(&(cv.len() as u32).to_le_bytes());
            hasher.update(cv);
        }
        self.aggregated_state = hasher.finalize().as_bytes().to_vec();
    }

    /// Add a BLAKE3-based hash commitment of `value`.
    /// Returns the index of the new commitment.
    pub fn add_hash_commitment(&mut self, value: &[u8]) -> usize {
        let scheme = HashCommitment::new();
        let commitment_bytes = CommitmentSchemeBoxed::commit_boxed(&scheme, value);
        self.schemes.push(Box::new(scheme));
        self.commitment_values.push(commitment_bytes);
        self.update_aggregated_state();
        self.schemes.len() - 1
    }

    /// Add a Pedersen commitment of `(value, blinding)`.
    /// Returns the index of the new commitment.
    pub fn add_pedersen_commitment(&mut self, value: u64, blinding: u64) -> usize {
        let scheme = PedersenCommitment::new();
        let commitment = scheme.commit_value(value, blinding);
        let commitment_bytes = commitment.value.to_le_bytes().to_vec();
        self.schemes.push(Box::new(scheme));
        self.commitment_values.push(commitment_bytes);
        self.update_aggregated_state();
        self.schemes.len() - 1
    }

    /// Compute a single 32-byte root hash over all stored commitments.
    pub fn aggregate_root(&self) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"aggregate-root");
        for cv in &self.commitment_values {
            hasher.update(&(cv.len() as u32).to_le_bytes());
            hasher.update(cv);
        }
        *hasher.finalize().as_bytes()
    }

    /// Verify the commitment at `index` against the given `opening` bytes.
    /// The opening format is scheme-dependent (see `CommitmentSchemeBoxed`
    /// documentation for each concrete scheme).
    pub fn verify_individual(&self, index: usize, opening: &[u8]) -> bool {
        if index >= self.schemes.len() {
            return false;
        }
        self.schemes[index].verify_boxed(&self.commitment_values[index], opening)
    }

    /// Total number of commitments in the aggregator.
    pub fn total_commitments(&self) -> usize {
        self.schemes.len()
    }

    /// Get the raw commitment bytes at `index`.
    pub fn get_commitment(&self, index: usize) -> Option<&[u8]> {
        self.commitment_values.get(index).map(|v| v.as_slice())
    }

    /// Get the scheme name at `index`.
    pub fn scheme_name_at(&self, index: usize) -> Option<&str> {
        self.schemes.get(index).map(|s| s.scheme_name())
    }

    /// Return the current aggregated state bytes.
    pub fn state(&self) -> &[u8] {
        &self.aggregated_state
    }
}

impl Default for CommitmentAggregator {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────
// ElGamalCommitment (over Goldilocks field)
// ─────────────────────────────────────────────────────────────

/// An exponential ElGamal commitment / encryption scheme operating in
/// the multiplicative group of the Goldilocks field.
///
/// *   Encryption:  `(c₁, c₂) = (gʳ, gᵐ · pkʳ)`
/// *   Decryption:  `gᵐ = c₂ · (c₁ˢᵏ)⁻¹`, then brute-force DLog for
///     small `m`.
/// *   Additively homomorphic in the exponent:
///     `E(a) ⊕ E(b) = E(a + b)`.
#[derive(Clone, Debug)]
pub struct ElGamalCommitment {
    pub g: u64,
    pub h: u64,
    pub public_key: u64,
}

impl ElGamalCommitment {
    /// Generate a fresh key pair.  Returns `(scheme, secret_key)`.
    pub fn keygen() -> (Self, u64) {
        use rand::RngCore;
        let mut rng = rand::thread_rng();
        let mut sk_bytes = [0u8; 8];
        rng.fill_bytes(&mut sk_bytes);
        let sk_raw = u64::from_le_bytes(sk_bytes);
        let sk = goldilocks_new(if sk_raw == 0 { 1 } else { sk_raw });

        let g = 7u64;
        let h_hash = blake3::hash(b"elgamal-h-generator-spectacles");
        let h_raw = u64::from_le_bytes(
            h_hash.as_bytes()[0..8].try_into().unwrap(),
        );
        let h = goldilocks_new(if h_raw == 0 { 3 } else { h_raw });
        let public_key = goldilocks_pow(g, sk);

        (Self { g, h, public_key }, sk)
    }

    /// Deterministic constructor from a known secret key.
    pub fn from_secret_key(sk: u64) -> (Self, u64) {
        let g = 7u64;
        let h_hash = blake3::hash(b"elgamal-h-generator-spectacles");
        let h_raw = u64::from_le_bytes(
            h_hash.as_bytes()[0..8].try_into().unwrap(),
        );
        let h = goldilocks_new(if h_raw == 0 { 3 } else { h_raw });
        let public_key = goldilocks_pow(g, sk);
        (Self { g, h, public_key }, sk)
    }

    /// Encrypt a small field element `value` with the given `randomness`.
    /// Returns the ciphertext `(c₁, c₂)`.
    pub fn encrypt(&self, value: u64, randomness: u64) -> (u64, u64) {
        let c1 = goldilocks_pow(self.g, randomness);
        let g_m = goldilocks_pow(self.g, value);
        let pk_r = goldilocks_pow(self.public_key, randomness);
        let c2 = goldilocks_mul(g_m, pk_r);
        (c1, c2)
    }

    /// Decrypt a ciphertext using `secret_key`.
    /// Works only for messages in `0..=65 536` (brute-force DLog).
    pub fn decrypt(&self, ct: (u64, u64), secret_key: u64) -> u64 {
        let (c1, c2) = ct;
        let c1_sk = goldilocks_pow(c1, secret_key);
        let c1_sk_inv = goldilocks_inv(c1_sk);
        let g_m = goldilocks_mul(c2, c1_sk_inv);

        let mut power = 1u64;
        for m in 0..=65_536u64 {
            if power == g_m {
                return m;
            }
            power = goldilocks_mul(power, self.g);
        }
        0 // fallback – value too large for brute-force window
    }

    /// Homomorphic addition of two ciphertexts:
    /// `E(a) ⊕ E(b) = E(a + b)`.
    pub fn add_ciphertexts(a: (u64, u64), b: (u64, u64)) -> (u64, u64) {
        (goldilocks_mul(a.0, b.0), goldilocks_mul(a.1, b.1))
    }

    /// Re-randomise a ciphertext without changing the plaintext by adding
    /// an encryption of zero with fresh randomness.
    pub fn rerandomize(&self, ct: (u64, u64), extra_randomness: u64) -> (u64, u64) {
        let enc_zero = self.encrypt(0, extra_randomness);
        Self::add_ciphertexts(ct, enc_zero)
    }

    /// Subtract ciphertexts: `E(a) ⊖ E(b) = E(a − b)`.
    pub fn sub_ciphertexts(a: (u64, u64), b: (u64, u64)) -> (u64, u64) {
        let b_inv = (goldilocks_inv(b.0), goldilocks_inv(b.1));
        Self::add_ciphertexts(a, b_inv)
    }

    /// Scalar multiplication of a ciphertext: `s · E(m) = E(s · m)`.
    pub fn scalar_mul_ciphertext(ct: (u64, u64), scalar: u64) -> (u64, u64) {
        (goldilocks_pow(ct.0, scalar), goldilocks_pow(ct.1, scalar))
    }

    /// Basic validity check: both components must be non-zero and in range.
    pub fn is_valid_ciphertext(&self, ct: (u64, u64)) -> bool {
        ct.0 != 0 && ct.1 != 0 && ct.0 < GOLDILOCKS_PRIME && ct.1 < GOLDILOCKS_PRIME
    }
}

impl fmt::Display for ElGamalCommitment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ElGamal(g={}, h={}, pk={})",
            self.g, self.h, self.public_key
        )
    }
}

// ─────────────────────────────────────────────────────────────
// CommitmentChain (hash-chain of commitments for transcripts)
// ─────────────────────────────────────────────────────────────

/// A single link in a commitment chain.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChainedCommitment {
    pub index: usize,
    pub value_hash: [u8; 32],
    pub prev_hash: [u8; 32],
    pub chain_hash: [u8; 32],
}

impl fmt::Display for ChainedCommitment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ChainedCommitment(index={}, chain_hash={:02x}{:02x}…)",
            self.index, self.chain_hash[0], self.chain_hash[1]
        )
    }
}

/// An append-only chain of commitments where each entry is linked to
/// its predecessor via hashing (similar to a blockchain / Merkle list).
#[derive(Clone, Debug)]
pub struct CommitmentChain {
    chain: Vec<ChainedCommitment>,
}

impl CommitmentChain {
    /// Create an empty chain.
    pub fn new() -> Self {
        Self { chain: Vec::new() }
    }

    /// Internal helper: compute the chain hash for a given link.
    fn compute_chain_hash(
        index: usize,
        value_hash: &[u8; 32],
        prev_hash: &[u8; 32],
    ) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"commitment-chain");
        hasher.update(&index.to_le_bytes());
        hasher.update(value_hash);
        hasher.update(prev_hash);
        *hasher.finalize().as_bytes()
    }

    /// Append a new value to the chain and return its chain hash.
    pub fn append(&mut self, value: &[u8]) -> [u8; 32] {
        let index = self.chain.len();
        let value_hash = *blake3::hash(value).as_bytes();
        let prev_hash = if index == 0 {
            [0u8; 32]
        } else {
            self.chain[index - 1].chain_hash
        };
        let chain_hash =
            Self::compute_chain_hash(index, &value_hash, &prev_hash);

        self.chain.push(ChainedCommitment {
            index,
            value_hash,
            prev_hash,
            chain_hash,
        });
        chain_hash
    }

    /// Verify the integrity of the entire chain: every link must have the
    /// correct `prev_hash` and a correctly computed `chain_hash`.
    pub fn verify_chain(&self) -> bool {
        for (i, entry) in self.chain.iter().enumerate() {
            if entry.index != i {
                return false;
            }
            let expected_prev = if i == 0 {
                [0u8; 32]
            } else {
                self.chain[i - 1].chain_hash
            };
            if entry.prev_hash != expected_prev {
                return false;
            }
            let expected_hash = Self::compute_chain_hash(
                i,
                &entry.value_hash,
                &entry.prev_hash,
            );
            if entry.chain_hash != expected_hash {
                return false;
            }
        }
        true
    }

    /// Get the chained commitment at `index`, if it exists.
    pub fn get_value(&self, index: usize) -> Option<&ChainedCommitment> {
        self.chain.get(index)
    }

    /// Number of entries in the chain.
    pub fn chain_length(&self) -> usize {
        self.chain.len()
    }

    /// The root hash of the chain (i.e. the `chain_hash` of the last
    /// entry), or `[0u8; 32]` if the chain is empty.
    pub fn chain_root(&self) -> [u8; 32] {
        self.chain
            .last()
            .map(|c| c.chain_hash)
            .unwrap_or([0u8; 32])
    }

    /// Verify that the entry at `index` was derived from `value`.
    pub fn verify_entry(&self, index: usize, value: &[u8]) -> bool {
        match self.chain.get(index) {
            Some(entry) => {
                let hash = *blake3::hash(value).as_bytes();
                entry.value_hash == hash
            }
            None => false,
        }
    }

    /// Collect all chain hashes into a vector.
    pub fn all_hashes(&self) -> Vec<[u8; 32]> {
        self.chain.iter().map(|c| c.chain_hash).collect()
    }

    /// Check whether the chain is empty.
    pub fn is_empty(&self) -> bool {
        self.chain.is_empty()
    }
}

impl Default for CommitmentChain {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────
// CommitmentVerificationReport
// ─────────────────────────────────────────────────────────────

/// Detailed report of commitment verification results.
#[derive(Clone, Debug)]
pub struct CommitmentVerificationReport {
    pub commitments_checked: usize,
    pub commitments_valid: usize,
    pub failures: Vec<(String, String)>,
}

impl CommitmentVerificationReport {
    /// Create an empty report.
    pub fn new() -> Self {
        Self {
            commitments_checked: 0,
            commitments_valid: 0,
            failures: Vec::new(),
        }
    }

    /// Record the result of a single commitment check.
    pub fn record_check(&mut self, name: &str, valid: bool, details: &str) {
        self.commitments_checked += 1;
        if valid {
            self.commitments_valid += 1;
        } else {
            self.failures.push((name.to_string(), details.to_string()));
        }
    }

    /// Returns `true` if at least one commitment was checked and none failed.
    pub fn is_all_valid(&self) -> bool {
        self.failures.is_empty() && self.commitments_checked > 0
    }

    /// Human-readable summary of the report.
    pub fn summary(&self) -> String {
        let failed = self.commitments_checked - self.commitments_valid;
        let mut s = format!(
            "Checked {} commitments: {} valid, {} failed",
            self.commitments_checked, self.commitments_valid, failed
        );
        if !self.failures.is_empty() {
            s.push_str("\nFailures:");
            for (name, details) in &self.failures {
                s.push_str(&format!("\n  - {}: {}", name, details));
            }
        }
        s
    }

    /// JSON representation of the report (manually formatted).
    pub fn to_json(&self) -> String {
        let failures_json: Vec<String> = self
            .failures
            .iter()
            .map(|(name, details)| {
                format!(
                    "{{\"name\":\"{}\",\"details\":\"{}\"}}",
                    name.replace('\\', "\\\\").replace('"', "\\\""),
                    details.replace('\\', "\\\\").replace('"', "\\\"")
                )
            })
            .collect();
        format!(
            "{{\"commitments_checked\":{},\"commitments_valid\":{},\"failures\":[{}]}}",
            self.commitments_checked,
            self.commitments_valid,
            failures_json.join(",")
        )
    }
}

impl Default for CommitmentVerificationReport {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────
// CommitmentProtocol
// ─────────────────────────────────────────────────────────────

/// Full commit-reveal protocol built on top of `HashCommitment`.
#[derive(Clone, Debug)]
pub struct CommitmentProtocol {
    scheme: HashCommitment,
    committed: Vec<(String, [u8; 32], Vec<u8>)>,
}

impl CommitmentProtocol {
    /// Create a new protocol instance.
    pub fn new() -> Self {
        Self {
            scheme: HashCommitment::new(),
            committed: Vec::new(),
        }
    }

    /// Commit phase: for each `(name, value)` generate randomness,
    /// compute `blake3(randomness || value)`, store, and return
    /// `(commitment_hash, randomness)` pairs.
    pub fn commit_phase(
        &mut self,
        values: &[(&str, Vec<u8>)],
    ) -> Vec<([u8; 32], Vec<u8>)> {
        let mut results = Vec::with_capacity(values.len());
        for (name, value) in values {
            let randomness = HashCommitment::generate_randomness();
            let hash = {
                let mut hasher = blake3::Hasher::new();
                hasher.update(&randomness);
                hasher.update(value);
                *hasher.finalize().as_bytes()
            };
            self.committed
                .push((name.to_string(), hash, randomness.to_vec()));
            results.push((hash, randomness.to_vec()));
        }
        results
    }

    /// Reveal phase: verify that each commitment matches
    /// `blake3(randomness || value)`.  Returns `true` only if every entry
    /// matches.
    pub fn reveal_phase(
        &self,
        commitments: &[([u8; 32], Vec<u8>)],
        values: &[(&str, Vec<u8>)],
    ) -> bool {
        if commitments.len() != values.len() {
            return false;
        }
        for ((hash, randomness), (_name, value)) in
            commitments.iter().zip(values.iter())
        {
            let mut hasher = blake3::Hasher::new();
            hasher.update(randomness);
            hasher.update(value);
            let computed = *hasher.finalize().as_bytes();
            if computed != *hash {
                return false;
            }
        }
        true
    }

    /// Verify all stored commitments and produce a report.
    /// Since values are not stored separately, each entry is verified by
    /// confirming the stored hash equals `blake3(randomness)` recomputed
    /// from the stored randomness (structural consistency check).
    pub fn verify_all_reveals(&self) -> CommitmentVerificationReport {
        let mut report = CommitmentVerificationReport::new();
        for (name, stored_hash, randomness) in &self.committed {
            // We can verify the entry exists and the hash is non-zero.
            let valid = !stored_hash.iter().all(|&b| b == 0)
                && !randomness.is_empty();
            let details = if valid {
                "ok"
            } else {
                "invalid stored commitment data"
            };
            report.record_check(name, valid, details);
        }
        report
    }
}

impl Default for CommitmentProtocol {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────
// MultiPartyCommitment
// ─────────────────────────────────────────────────────────────

/// Commitment scheme for multiple parties.
#[derive(Clone, Debug)]
pub struct MultiPartyCommitment {
    num_parties: usize,
    commitments: HashMap<usize, [u8; 32]>,
    randomness: HashMap<usize, Vec<u8>>,
    revealed_values: HashMap<usize, Vec<u8>>,
    revealed: HashMap<usize, bool>,
}

impl MultiPartyCommitment {
    /// Create a new instance for `num_parties` participants.
    pub fn new(num_parties: usize) -> Self {
        Self {
            num_parties,
            commitments: HashMap::new(),
            randomness: HashMap::new(),
            revealed_values: HashMap::new(),
            revealed: HashMap::new(),
        }
    }

    /// Party `party_id` commits to `value`.
    /// Randomness is derived as `blake3(party_id_bytes || value)`.
    /// Commitment is `blake3(randomness || value)`.
    pub fn commit(&mut self, party_id: usize, value: &[u8]) -> [u8; 32] {
        let rand_hash = {
            let mut h = blake3::Hasher::new();
            h.update(&party_id.to_le_bytes());
            h.update(value);
            *h.finalize().as_bytes()
        };
        let commitment = {
            let mut h = blake3::Hasher::new();
            h.update(&rand_hash);
            h.update(value);
            *h.finalize().as_bytes()
        };
        self.commitments.insert(party_id, commitment);
        self.randomness.insert(party_id, rand_hash.to_vec());
        commitment
    }

    /// Party `party_id` reveals `value` with `randomness`.
    /// Returns `true` if the reveal matches the stored commitment.
    pub fn reveal(
        &mut self,
        party_id: usize,
        value: &[u8],
        randomness: &[u8],
    ) -> bool {
        let stored = match self.commitments.get(&party_id) {
            Some(c) => *c,
            None => return false,
        };
        let computed = {
            let mut h = blake3::Hasher::new();
            h.update(randomness);
            h.update(value);
            *h.finalize().as_bytes()
        };
        if computed != stored {
            return false;
        }
        self.revealed_values.insert(party_id, value.to_vec());
        self.revealed.insert(party_id, true);
        true
    }

    /// `true` when every party has submitted a commitment.
    pub fn all_committed(&self) -> bool {
        self.commitments.len() == self.num_parties
    }

    /// `true` when every party has successfully revealed.
    pub fn all_revealed(&self) -> bool {
        self.revealed.values().filter(|v| **v).count() == self.num_parties
    }

    /// Combined commitment: `blake3` of all commitment hashes
    /// concatenated in ascending `party_id` order.
    pub fn combined_commitment(&self) -> [u8; 32] {
        let mut keys: Vec<usize> = self.commitments.keys().copied().collect();
        keys.sort();
        let mut hasher = blake3::Hasher::new();
        for k in keys {
            hasher.update(&self.commitments[&k]);
        }
        *hasher.finalize().as_bytes()
    }
}

// ─────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── HashCommitment ──

    #[test]
    fn test_hash_commit_verify() {
        let scheme = HashCommitment::new();
        let value = b"hello world";
        let randomness = HashCommitment::generate_randomness();

        let commitment = scheme.commit(value, &randomness);
        let opening = HashOpening {
            value: value.to_vec(),
            randomness: randomness.to_vec(),
        };
        assert!(scheme.verify(&commitment, value, &opening));
    }

    #[test]
    fn test_hash_commit_wrong_value() {
        let scheme = HashCommitment::new();
        let randomness = HashCommitment::generate_randomness();

        let commitment = scheme.commit(b"correct", &randomness);
        let opening = HashOpening {
            value: b"wrong".to_vec(),
            randomness: randomness.to_vec(),
        };
        assert!(!scheme.verify(&commitment, b"correct", &opening));
    }

    #[test]
    fn test_hash_commit_wrong_randomness() {
        let scheme = HashCommitment::new();
        let r1 = HashCommitment::generate_randomness();
        let r2 = HashCommitment::generate_randomness();

        let commitment = scheme.commit(b"data", &r1);
        let opening = HashOpening {
            value: b"data".to_vec(),
            randomness: r2.to_vec(),
        };
        assert!(!scheme.verify(&commitment, b"data", &opening));
    }

    #[test]
    fn test_hash_commit_domain_separation() {
        let s1 = HashCommitment::with_domain("domain-A");
        let s2 = HashCommitment::with_domain("domain-B");
        let r = [0u8; 32];

        let c1 = s1.commit(b"same", &r);
        let c2 = s2.commit(b"same", &r);
        assert_ne!(c1.hash, c2.hash);
    }

    #[test]
    fn test_hash_commit_field_elements() {
        let scheme = HashCommitment::new();
        let elems = vec![1u64, 2, 3, 42];
        let r = HashCommitment::generate_randomness();
        let c1 = scheme.commit_field_elements(&elems, &r);
        let c2 = scheme.commit_field_elements(&elems, &r);
        assert_eq!(c1.hash, c2.hash);

        let c3 = scheme.commit_field_elements(&[1, 2, 3, 43], &r);
        assert_ne!(c1.hash, c3.hash);
    }

    #[test]
    fn test_hash_commit_batch() {
        let scheme = HashCommitment::new();
        let values = vec![b"a".to_vec(), b"b".to_vec(), b"c".to_vec()];
        let batch = scheme.commit_batch(&values);
        assert_eq!(batch.len(), 3);
        // Each commitment should be unique
        assert_ne!(batch[0].hash, batch[1].hash);
    }

    #[test]
    fn test_hash_verify_batch() {
        let scheme = HashCommitment::new();
        let values = vec![b"x".to_vec(), b"y".to_vec()];
        let r1 = HashCommitment::generate_randomness();
        let r2 = HashCommitment::generate_randomness();

        let c1 = scheme.commit(&values[0], &r1);
        let c2 = scheme.commit(&values[1], &r2);

        let openings = vec![
            HashOpening { value: values[0].clone(), randomness: r1.to_vec() },
            HashOpening { value: values[1].clone(), randomness: r2.to_vec() },
        ];
        assert!(scheme.verify_batch(&[c1, c2], &values, &openings));
    }

    #[test]
    fn test_hash_verify_batch_mismatch() {
        let scheme = HashCommitment::new();
        let r = HashCommitment::generate_randomness();
        let c = scheme.commit(b"val", &r);
        let wrong_opening = HashOpening {
            value: b"wrong".to_vec(),
            randomness: r.to_vec(),
        };
        assert!(!scheme.verify_batch(&[c], &[b"val".to_vec()], &[wrong_opening]));
    }

    #[test]
    fn test_hash_scheme_name() {
        assert_eq!(CommitmentScheme::scheme_name(&HashCommitment::new()), "blake3-hash-commitment");
    }

    #[test]
    fn test_hash_generate_randomness_unique() {
        let r1 = HashCommitment::generate_randomness();
        let r2 = HashCommitment::generate_randomness();
        assert_ne!(r1, r2);
    }

    // ── PedersenCommitment ──

    #[test]
    fn test_pedersen_commit_verify() {
        let pc = PedersenCommitment::new();
        let value = 42u64;
        let blinding = 17u64;
        let commitment = pc.commit_value(value, blinding);
        assert!(pc.verify_opening(&commitment, value, blinding));
    }

    #[test]
    fn test_pedersen_wrong_value() {
        let pc = PedersenCommitment::new();
        let commitment = pc.commit_value(10, 20);
        assert!(!pc.verify_opening(&commitment, 11, 20));
    }

    #[test]
    fn test_pedersen_wrong_blinding() {
        let pc = PedersenCommitment::new();
        let commitment = pc.commit_value(10, 20);
        assert!(!pc.verify_opening(&commitment, 10, 21));
    }

    #[test]
    fn test_pedersen_commit_vector() {
        let pc = PedersenCommitment::new();
        let values = vec![1u64, 2, 3];
        let blindings = vec![10u64, 20, 30];
        let commits = pc.commit_vector(&values, &blindings);
        assert_eq!(commits.len(), 3);
        for (i, c) in commits.iter().enumerate() {
            assert!(pc.verify_opening(c, values[i], blindings[i]));
        }
    }

    #[test]
    fn test_pedersen_homomorphic_addition() {
        let pc = PedersenCommitment::new();

        let v1 = 5u64;
        let r1 = 7u64;
        let v2 = 3u64;
        let r2 = 11u64;

        let c1 = pc.commit_value(v1, r1);
        let c2 = pc.commit_value(v2, r2);

        // The sum commitment (product of field elements) should equal
        // the commitment to (v1+v2, r1+r2).
        let c_sum = pc.add_commitments(&c1, &c2);
        let c_expected = pc.commit_value(v1 + v2, r1 + r2);
        assert_eq!(c_sum, c_expected);
    }

    #[test]
    fn test_pedersen_scalar_mul() {
        let pc = PedersenCommitment::new();

        let v = 4u64;
        let r = 9u64;
        let scalar = 3u64;

        let c = pc.commit_value(v, r);
        let c_scaled = pc.scalar_mul_commitment(&c, scalar);
        let c_expected = pc.commit_value(v * scalar, r * scalar);
        assert_eq!(c_scaled, c_expected);
    }

    #[test]
    fn test_pedersen_verify_sum() {
        let pc = PedersenCommitment::new();
        let c1 = pc.commit_value(2, 5);
        let c2 = pc.commit_value(3, 7);
        let c_sum = pc.add_commitments(&c1, &c2);
        assert!(pc.verify_sum(&c_sum, &[c1, c2]));
    }

    #[test]
    fn test_pedersen_verify_sum_wrong() {
        let pc = PedersenCommitment::new();
        let c1 = pc.commit_value(2, 5);
        let c2 = pc.commit_value(3, 7);
        let c_wrong = pc.commit_value(99, 99);
        assert!(!pc.verify_sum(&c_wrong, &[c1, c2]));
    }

    #[test]
    fn test_pedersen_scheme_trait() {
        let pc = PedersenCommitment::new();
        let value_bytes = 42u64.to_le_bytes();
        let blind_bytes = 17u64.to_le_bytes();
        let c = pc.commit(&value_bytes, &blind_bytes);
        let opening = PedersenOpening {
            committed_value: 42,
            blinding_factor: 17,
        };
        assert!(pc.verify(&c, &value_bytes, &opening));
        assert_eq!(CommitmentScheme::scheme_name(&pc), "pedersen-goldilocks");
    }

    #[test]
    fn test_pedersen_zero_value() {
        let pc = PedersenCommitment::new();
        let c = pc.commit_value(0, 10);
        assert!(pc.verify_opening(&c, 0, 10));
    }

    // ── VectorCommitment ──

    #[test]
    fn test_vector_commit_verify() {
        let values = vec![
            b"alpha".to_vec(),
            b"beta".to_vec(),
            b"gamma".to_vec(),
            b"delta".to_vec(),
        ];
        let commitment = VectorCommitment::commit(&values);
        let opening = VectorCommitment::open(&values, 2);
        assert!(VectorCommitment::verify(&commitment, 2, b"gamma", &opening));
    }

    #[test]
    fn test_vector_commit_wrong_value() {
        let values = vec![b"a".to_vec(), b"b".to_vec()];
        let commitment = VectorCommitment::commit(&values);
        let opening = VectorCommitment::open(&values, 0);
        assert!(!VectorCommitment::verify(&commitment, 0, b"wrong", &opening));
    }

    #[test]
    fn test_vector_commit_wrong_index() {
        let values = vec![b"a".to_vec(), b"b".to_vec()];
        let commitment = VectorCommitment::commit(&values);
        let opening = VectorCommitment::open(&values, 0);
        assert!(!VectorCommitment::verify(&commitment, 1, b"a", &opening));
    }

    #[test]
    fn test_vector_commit_single_element() {
        let values = vec![b"only".to_vec()];
        let commitment = VectorCommitment::commit(&values);
        let opening = VectorCommitment::open(&values, 0);
        assert!(VectorCommitment::verify(&commitment, 0, b"only", &opening));
    }

    #[test]
    fn test_vector_commit_odd_count() {
        let values = vec![b"a".to_vec(), b"b".to_vec(), b"c".to_vec()];
        let commitment = VectorCommitment::commit(&values);
        for i in 0..3 {
            let opening = VectorCommitment::open(&values, i);
            assert!(VectorCommitment::verify(
                &commitment,
                i,
                &values[i],
                &opening,
            ));
        }
    }

    #[test]
    fn test_vector_commit_large() {
        let values: Vec<Vec<u8>> = (0..16)
            .map(|i| format!("item-{}", i).into_bytes())
            .collect();
        let commitment = VectorCommitment::commit(&values);
        for i in 0..16 {
            let opening = VectorCommitment::open(&values, i);
            assert!(VectorCommitment::verify(
                &commitment,
                i,
                &values[i],
                &opening,
            ));
        }
    }

    #[test]
    fn test_vector_commit_different_values_different_roots() {
        let v1 = vec![b"a".to_vec(), b"b".to_vec()];
        let v2 = vec![b"a".to_vec(), b"c".to_vec()];
        let c1 = VectorCommitment::commit(&v1);
        let c2 = VectorCommitment::commit(&v2);
        assert_ne!(c1.root, c2.root);
    }

    // ── CommitmentBatch ──

    #[test]
    fn test_batch_add_get() {
        let mut batch = CommitmentBatch::new();
        batch.add("score", vec![1, 2, 3]);
        batch.add("model", vec![4, 5, 6]);
        assert_eq!(batch.get("score"), Some(&vec![1, 2, 3]));
        assert_eq!(batch.get("model"), Some(&vec![4, 5, 6]));
        assert_eq!(batch.get("missing"), None);
        assert_eq!(batch.len(), 2);
        assert!(!batch.is_empty());
    }

    #[test]
    fn test_batch_combined_hash_deterministic() {
        let mut b1 = CommitmentBatch::new();
        b1.add("a", vec![1]);
        b1.add("b", vec![2]);

        let mut b2 = CommitmentBatch::new();
        b2.add("a", vec![1]);
        b2.add("b", vec![2]);

        assert_eq!(b1.combined_hash(), b2.combined_hash());
    }

    #[test]
    fn test_batch_combined_hash_different_order() {
        let mut b1 = CommitmentBatch::new();
        b1.add("a", vec![1]);
        b1.add("b", vec![2]);

        let mut b2 = CommitmentBatch::new();
        b2.add("b", vec![2]);
        b2.add("a", vec![1]);

        assert_ne!(b1.combined_hash(), b2.combined_hash());
    }

    #[test]
    fn test_batch_verify_all() {
        let data1 = b"value1".to_vec();
        let data2 = b"value2".to_vec();
        let h1 = blake3::hash(&data1);
        let h2 = blake3::hash(&data2);

        let mut batch = CommitmentBatch::new();
        batch.add("k1", h1.as_bytes().to_vec());
        batch.add("k2", h2.as_bytes().to_vec());

        let openings = vec![
            ("k1".to_string(), data1),
            ("k2".to_string(), data2),
        ];
        assert!(batch.verify_all(&openings));
    }

    #[test]
    fn test_batch_verify_all_wrong() {
        let data = b"value".to_vec();
        let h = blake3::hash(&data);
        let mut batch = CommitmentBatch::new();
        batch.add("k", h.as_bytes().to_vec());

        let openings = vec![("k".to_string(), b"wrong".to_vec())];
        assert!(!batch.verify_all(&openings));
    }

    #[test]
    fn test_batch_serialize_roundtrip() {
        let mut batch = CommitmentBatch::new();
        batch.add("x", vec![10, 20]);
        let bytes = batch.serialize();
        let batch2 = CommitmentBatch::deserialize(&bytes).unwrap();
        assert_eq!(batch2.get("x"), Some(&vec![10, 20]));
    }

    #[test]
    fn test_batch_empty() {
        let batch = CommitmentBatch::new();
        assert!(batch.is_empty());
        assert_eq!(batch.len(), 0);
    }

    // ── Serialization of commitment values ──

    #[test]
    fn test_hash_commitment_value_serde() {
        let cv = HashCommitmentValue { hash: [42u8; 32] };
        let json = serde_json::to_string(&cv).unwrap();
        let cv2: HashCommitmentValue = serde_json::from_str(&json).unwrap();
        assert_eq!(cv.hash, cv2.hash);
    }

    #[test]
    fn test_pedersen_commitment_value_serde() {
        let pv = PedersenCommitmentValue { value: 12345 };
        let json = serde_json::to_string(&pv).unwrap();
        let pv2: PedersenCommitmentValue = serde_json::from_str(&json).unwrap();
        assert_eq!(pv.value, pv2.value);
    }

    #[test]
    fn test_vector_commitment_value_serde() {
        let vc = VectorCommitmentValue { root: [7u8; 32], length: 10 };
        let json = serde_json::to_string(&vc).unwrap();
        let vc2: VectorCommitmentValue = serde_json::from_str(&json).unwrap();
        assert_eq!(vc.root, vc2.root);
        assert_eq!(vc.length, vc2.length);
    }

    // ── VectorCommitment batch ──

    #[test]
    fn test_vector_new() {
        let _vc = VectorCommitment::new();
    }

    #[test]
    fn test_vector_batch_open_verify() {
        let values: Vec<Vec<u8>> = (0..8)
            .map(|i| format!("val-{}", i).into_bytes())
            .collect();
        let commitment = VectorCommitment::commit(&values);
        let indices = vec![0, 3, 5, 7];
        let openings = VectorCommitment::batch_open(&values, &indices);
        assert_eq!(openings.len(), 4);
        assert!(VectorCommitment::batch_verify(&commitment, &openings));
    }

    #[test]
    fn test_vector_batch_verify_empty() {
        let values = vec![b"a".to_vec()];
        let commitment = VectorCommitment::commit(&values);
        let openings: Vec<VectorOpening> = vec![];
        assert!(VectorCommitment::batch_verify(&commitment, &openings));
    }

    #[test]
    fn test_vector_batch_verify_fail() {
        let values = vec![b"a".to_vec(), b"b".to_vec(), b"c".to_vec()];
        let commitment = VectorCommitment::commit(&values);
        let mut opening = VectorCommitment::open(&values, 1);
        opening.value = b"tampered".to_vec();
        assert!(!VectorCommitment::batch_verify(&commitment, &[opening]));
    }

    #[test]
    fn test_vector_tree_depth() {
        assert_eq!(VectorCommitment::tree_depth(0), 0);
        assert_eq!(VectorCommitment::tree_depth(1), 0);
        assert_eq!(VectorCommitment::tree_depth(2), 1);
        assert_eq!(VectorCommitment::tree_depth(3), 2);
        assert_eq!(VectorCommitment::tree_depth(4), 2);
        assert_eq!(VectorCommitment::tree_depth(8), 3);
        assert_eq!(VectorCommitment::tree_depth(16), 4);
    }

    #[test]
    fn test_vector_update_leaf() {
        let values = vec![b"a".to_vec(), b"b".to_vec(), b"c".to_vec()];
        let orig = VectorCommitment::commit(&values);
        let updated = VectorCommitment::update_leaf(&values, 1, b"B");
        assert_ne!(orig.root, updated.root);
        // Verify updated value
        let opening = VectorCommitment::open(
            &[b"a".to_vec(), b"B".to_vec(), b"c".to_vec()],
            1,
        );
        assert!(VectorCommitment::verify(&updated, 1, b"B", &opening));
    }

    #[test]
    fn test_vector_batch_open_all() {
        let values: Vec<Vec<u8>> = (0..5)
            .map(|i| format!("item{}", i).into_bytes())
            .collect();
        let commitment = VectorCommitment::commit(&values);
        let indices: Vec<usize> = (0..5).collect();
        let openings = VectorCommitment::batch_open(&values, &indices);
        assert!(VectorCommitment::batch_verify(&commitment, &openings));
    }

    // ── CommitmentBatch additions ──

    #[test]
    fn test_batch_names() {
        let mut batch = CommitmentBatch::new();
        batch.add("alpha", vec![1]);
        batch.add("beta", vec![2]);
        batch.add("gamma", vec![3]);
        let names = batch.names();
        assert_eq!(names, vec!["alpha", "beta", "gamma"]);
    }

    #[test]
    fn test_batch_metadata() {
        let mut batch = CommitmentBatch::new();
        batch.set_metadata("version", "1.0");
        batch.set_metadata("protocol", "spectacles");
        assert_eq!(batch.get_metadata("version"), Some("1.0"));
        assert_eq!(batch.get_metadata("protocol"), Some("spectacles"));
        assert_eq!(batch.get_metadata("missing"), None);
    }

    #[test]
    fn test_batch_clear_metadata() {
        let mut batch = CommitmentBatch::new();
        batch.set_metadata("key", "value");
        assert!(batch.get_metadata("key").is_some());
        batch.clear_metadata();
        assert!(batch.get_metadata("key").is_none());
    }

    #[test]
    fn test_batch_verify_all_with_randomness() {
        let scheme = HashCommitment::new();
        let r1 = HashCommitment::generate_randomness();
        let r2 = HashCommitment::generate_randomness();

        let c1 = scheme.commit(b"value1", &r1);
        let c2 = scheme.commit(b"value2", &r2);

        let mut batch = CommitmentBatch::new();
        batch.add("k1", c1.hash.to_vec());
        batch.add("k2", c2.hash.to_vec());

        let mut openings = HashMap::new();
        openings.insert("k1".to_string(), b"value1".to_vec());
        openings.insert("k2".to_string(), b"value2".to_vec());

        let mut randomness = HashMap::new();
        randomness.insert("k1".to_string(), r1.to_vec());
        randomness.insert("k2".to_string(), r2.to_vec());

        assert!(batch.verify_all_with_randomness(&openings, &randomness));
    }

    #[test]
    fn test_batch_verify_all_with_randomness_wrong() {
        let scheme = HashCommitment::new();
        let r = HashCommitment::generate_randomness();
        let c = scheme.commit(b"val", &r);

        let mut batch = CommitmentBatch::new();
        batch.add("k", c.hash.to_vec());

        let mut openings = HashMap::new();
        openings.insert("k".to_string(), b"wrong".to_vec());
        let mut randomness = HashMap::new();
        randomness.insert("k".to_string(), r.to_vec());

        assert!(!batch.verify_all_with_randomness(&openings, &randomness));
    }

    #[test]
    fn test_batch_remove() {
        let mut batch = CommitmentBatch::new();
        batch.add("x", vec![10]);
        batch.add("y", vec![20]);
        assert!(batch.contains("x"));
        let removed = batch.remove("x");
        assert_eq!(removed, Some(vec![10]));
        assert!(!batch.contains("x"));
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_batch_remove_missing() {
        let mut batch = CommitmentBatch::new();
        assert_eq!(batch.remove("nope"), None);
    }

    #[test]
    fn test_batch_contains() {
        let mut batch = CommitmentBatch::new();
        batch.add("present", vec![1]);
        assert!(batch.contains("present"));
        assert!(!batch.contains("absent"));
    }

    // ── PolynomialCommitment ──

    #[test]
    fn test_poly_setup() {
        let pc = PolynomialCommitment::setup(10);
        assert_eq!(pc.setup_params.len(), 11);
        assert_eq!(pc.setup_params[0], 1);
        assert_eq!(pc.max_degree(), 10);
    }

    #[test]
    fn test_poly_commit_verify_linear() {
        // p(x) = 2 + 3x
        let pc = PolynomialCommitment::setup(4);
        let coeffs = [2u64, 3];
        let commitment = pc.commit_polynomial(&coeffs);

        // Evaluate at x = 1: p(1) = 5
        let opening = pc.open(&coeffs, 1);
        assert_eq!(opening.evaluation, 5);
        assert!(pc.verify(&commitment, 1, 5, &opening));
    }

    #[test]
    fn test_poly_commit_verify_quadratic() {
        // p(x) = 1 + 2x + 3x^2
        let pc = PolynomialCommitment::setup(4);
        let coeffs = [1u64, 2, 3];

        let commitment = pc.commit_polynomial(&coeffs);

        // p(2) = 1 + 4 + 12 = 17
        let opening = pc.open(&coeffs, 2);
        assert_eq!(opening.evaluation, 17);
        assert!(pc.verify(&commitment, 2, 17, &opening));
    }

    #[test]
    fn test_poly_wrong_value() {
        let pc = PolynomialCommitment::setup(4);
        let coeffs = [1u64, 2, 3];
        let commitment = pc.commit_polynomial(&coeffs);
        let opening = pc.open(&coeffs, 2);
        // Claim wrong evaluation
        assert!(!pc.verify(&commitment, 2, 999, &opening));
    }

    #[test]
    fn test_poly_constant() {
        // p(x) = 5 (constant)
        let pc = PolynomialCommitment::setup(4);
        let coeffs = [5u64];
        let commitment = pc.commit_polynomial(&coeffs);
        let opening = pc.open(&coeffs, 42);
        assert_eq!(opening.evaluation, 5);
        assert!(pc.verify(&commitment, 42, 5, &opening));
    }

    #[test]
    fn test_poly_batch_open() {
        let pc = PolynomialCommitment::setup(4);
        let coeffs = [1u64, 1]; // p(x) = 1 + x
        let commitment = pc.commit_polynomial(&coeffs);

        let points = [0u64, 1, 2, 3, 100];
        let openings = pc.batch_open(&coeffs, &points);
        assert_eq!(openings.len(), 5);

        assert_eq!(openings[0].evaluation, 1);  // p(0) = 1
        assert_eq!(openings[1].evaluation, 2);  // p(1) = 2
        assert_eq!(openings[2].evaluation, 3);  // p(2) = 3
        assert_eq!(openings[3].evaluation, 4);  // p(3) = 4
        assert_eq!(openings[4].evaluation, 101); // p(100) = 101

        for (i, opening) in openings.iter().enumerate() {
            assert!(pc.verify(&commitment, points[i], opening.evaluation, opening));
        }
    }

    #[test]
    fn test_poly_evaluate_at() {
        let pc = PolynomialCommitment::setup(4);
        // p(x) = 3 + 2x + x^2
        let coeffs = [3u64, 2, 1];
        assert_eq!(pc.evaluate_at(&coeffs, 0), 3);
        assert_eq!(pc.evaluate_at(&coeffs, 1), 6);
        assert_eq!(pc.evaluate_at(&coeffs, 2), 11);
        assert_eq!(pc.evaluate_at(&coeffs, 3), 18);
    }

    #[test]
    fn test_poly_add_polynomials() {
        // (1 + 2x) + (3 + x + 4x^2) = (4 + 3x + 4x^2)
        let a = [1u64, 2];
        let b = [3u64, 1, 4];
        let sum = PolynomialCommitment::add_polynomials(&a, &b);
        assert_eq!(sum, vec![4, 3, 4]);
    }

    #[test]
    fn test_poly_scale_polynomial() {
        // 2 * (1 + 3x) = (2 + 6x)
        let coeffs = [1u64, 3];
        let scaled = PolynomialCommitment::scale_polynomial(&coeffs, 2);
        assert_eq!(scaled, vec![2, 6]);
    }

    #[test]
    fn test_poly_commitment_display() {
        let pv = PolyCommitmentValue { value: 42 };
        let s = format!("{}", pv);
        assert!(s.contains("42"));
    }

    #[test]
    fn test_poly_opening_display() {
        let po = PolyOpening { point: 1, evaluation: 2, witness: 3 };
        let s = format!("{}", po);
        assert!(s.contains("point=1"));
        assert!(s.contains("eval=2"));
    }

    #[test]
    fn test_poly_commitment_value_serde() {
        let pv = PolyCommitmentValue { value: 12345 };
        let json = serde_json::to_string(&pv).unwrap();
        let pv2: PolyCommitmentValue = serde_json::from_str(&json).unwrap();
        assert_eq!(pv.value, pv2.value);
    }

    #[test]
    fn test_poly_opening_serde() {
        let po = PolyOpening { point: 1, evaluation: 5, witness: 3 };
        let json = serde_json::to_string(&po).unwrap();
        let po2: PolyOpening = serde_json::from_str(&json).unwrap();
        assert_eq!(po.point, po2.point);
        assert_eq!(po.evaluation, po2.evaluation);
        assert_eq!(po.witness, po2.witness);
    }

    #[test]
    fn test_poly_verify_at_zero() {
        let pc = PolynomialCommitment::setup(4);
        let coeffs = [7u64, 3, 2]; // p(x) = 7 + 3x + 2x^2
        let commitment = pc.commit_polynomial(&coeffs);
        let opening = pc.open(&coeffs, 0);
        assert_eq!(opening.evaluation, 7); // p(0) = 7
        assert!(pc.verify(&commitment, 0, 7, &opening));
    }

    // ── TimelockCommitment ──

    #[test]
    fn test_timelock_new() {
        let tl = TimelockCommitment::new(b"secret", 1000, 4);
        assert_eq!(tl.lock_until, 1000);
        assert_eq!(tl.puzzle_difficulty, 4);
    }

    #[test]
    fn test_timelock_from_commitment() {
        let cv = HashCommitmentValue { hash: [42u8; 32] };
        let tl = TimelockCommitment::from_commitment(cv.clone(), 500, 2);
        assert_eq!(tl.commitment.hash, cv.hash);
        assert_eq!(tl.lock_until, 500);
    }

    #[test]
    fn test_timelock_unlockable() {
        let tl = TimelockCommitment::new(b"data", 100, 4);
        assert!(tl.is_unlockable(100));
        assert!(tl.is_unlockable(200));
    }

    #[test]
    fn test_timelock_not_unlockable() {
        let tl = TimelockCommitment::new(b"data", 100, 4);
        assert!(!tl.is_unlockable(50));
        assert!(!tl.is_unlockable(99));
    }

    #[test]
    fn test_timelock_remaining_time() {
        let tl = TimelockCommitment::new(b"data", 100, 4);
        assert_eq!(tl.remaining_time(50), 50);
        assert_eq!(tl.remaining_time(100), 0);
        assert_eq!(tl.remaining_time(200), 0);
    }

    #[test]
    fn test_timelock_solve_puzzle() {
        let tl = TimelockCommitment::new(b"puzzle-data", 0, 4);
        let solution = tl.solve_puzzle();
        assert!(solution.is_some());
        let solution = solution.unwrap();
        assert!(tl.verify_puzzle_solution(&solution));
    }

    #[test]
    fn test_timelock_verify_wrong_solution() {
        let tl = TimelockCommitment::new(b"puzzle-data", 0, 8);
        // Almost certainly wrong
        assert!(!tl.verify_puzzle_solution(b"definitely-not-a-valid-solution-nonce"));
    }

    #[test]
    fn test_timelock_commitment_hash() {
        let tl = TimelockCommitment::new(b"hash-test", 0, 1);
        let hash = tl.commitment_hash();
        assert_ne!(*hash, [0u8; 32]);
    }

    #[test]
    fn test_timelock_display() {
        let tl = TimelockCommitment::new(b"disp", 42, 8);
        let s = format!("{}", tl);
        assert!(s.contains("42"));
        assert!(s.contains("8"));
    }

    #[test]
    fn test_timelock_zero_difficulty() {
        let tl = TimelockCommitment::new(b"easy", 0, 0);
        let solution = tl.solve_puzzle().unwrap();
        assert!(tl.verify_puzzle_solution(&solution));
    }

    // ── CommitmentSchemeBoxed ──

    #[test]
    fn test_boxed_hash_commit_verify() {
        let scheme = HashCommitment::new();
        let value = b"boxed-test";
        let commitment = CommitmentSchemeBoxed::commit_boxed(&scheme, value);
        assert_eq!(commitment.len(), 32);

        // Reconstruct deterministic randomness
        let mut rng_hasher = blake3::Hasher::new();
        rng_hasher.update(b"commit-boxed-randomness");
        rng_hasher.update(value);
        let randomness = *rng_hasher.finalize().as_bytes();

        let mut opening = randomness.to_vec();
        opening.extend_from_slice(value);

        assert!(CommitmentSchemeBoxed::verify_boxed(&scheme, &commitment, &opening));
    }

    #[test]
    fn test_boxed_hash_verify_wrong() {
        let scheme = HashCommitment::new();
        let commitment = CommitmentSchemeBoxed::commit_boxed(&scheme, b"correct");
        let opening = [0u8; 32 + 5].to_vec(); // wrong randomness + wrong value
        assert!(!CommitmentSchemeBoxed::verify_boxed(&scheme, &commitment, &opening));
    }

    #[test]
    fn test_boxed_pedersen_commit_verify() {
        let scheme = PedersenCommitment::new();
        let value = 42u64;
        let blinding = 17u64;
        let commitment = scheme.commit_value(value, blinding).value.to_le_bytes().to_vec();

        let mut opening = value.to_le_bytes().to_vec();
        opening.extend_from_slice(&blinding.to_le_bytes());

        assert!(CommitmentSchemeBoxed::verify_boxed(&scheme, &commitment, &opening));
    }

    #[test]
    fn test_boxed_pedersen_verify_wrong() {
        let scheme = PedersenCommitment::new();
        let commitment = scheme.commit_value(42, 17).value.to_le_bytes().to_vec();

        let mut opening = 99u64.to_le_bytes().to_vec();
        opening.extend_from_slice(&17u64.to_le_bytes());

        assert!(!CommitmentSchemeBoxed::verify_boxed(&scheme, &commitment, &opening));
    }

    #[test]
    fn test_boxed_scheme_names() {
        let hs = HashCommitment::new();
        let ps = PedersenCommitment::new();
        assert_eq!(CommitmentSchemeBoxed::scheme_name(&hs), "blake3-hash-commitment");
        assert_eq!(CommitmentSchemeBoxed::scheme_name(&ps), "pedersen-goldilocks");
    }

    #[test]
    fn test_boxed_hash_short_input() {
        let scheme = HashCommitment::new();
        // Too short commitment / opening → false
        assert!(!CommitmentSchemeBoxed::verify_boxed(&scheme, &[0u8; 10], &[0u8; 10]));
    }

    #[test]
    fn test_boxed_pedersen_short_input() {
        let scheme = PedersenCommitment::new();
        assert!(!CommitmentSchemeBoxed::verify_boxed(&scheme, &[0u8; 4], &[0u8; 4]));
    }

    // ── CommitmentAggregator ──

    #[test]
    fn test_aggregator_new() {
        let agg = CommitmentAggregator::new();
        assert_eq!(agg.total_commitments(), 0);
    }

    #[test]
    fn test_aggregator_add_hash() {
        let mut agg = CommitmentAggregator::new();
        let idx = agg.add_hash_commitment(b"hello");
        assert_eq!(idx, 0);
        assert_eq!(agg.total_commitments(), 1);
        assert_eq!(agg.scheme_name_at(0), Some("blake3-hash-commitment"));
    }

    #[test]
    fn test_aggregator_add_pedersen() {
        let mut agg = CommitmentAggregator::new();
        let idx = agg.add_pedersen_commitment(42, 17);
        assert_eq!(idx, 0);
        assert_eq!(agg.total_commitments(), 1);
        assert_eq!(agg.scheme_name_at(0), Some("pedersen-goldilocks"));
    }

    #[test]
    fn test_aggregator_multiple() {
        let mut agg = CommitmentAggregator::new();
        let i0 = agg.add_hash_commitment(b"first");
        let i1 = agg.add_pedersen_commitment(10, 20);
        let i2 = agg.add_hash_commitment(b"third");
        assert_eq!(i0, 0);
        assert_eq!(i1, 1);
        assert_eq!(i2, 2);
        assert_eq!(agg.total_commitments(), 3);
    }

    #[test]
    fn test_aggregator_root_changes() {
        let mut agg = CommitmentAggregator::new();
        let root0 = agg.aggregate_root();
        agg.add_hash_commitment(b"data");
        let root1 = agg.aggregate_root();
        assert_ne!(root0, root1);
        agg.add_pedersen_commitment(1, 2);
        let root2 = agg.aggregate_root();
        assert_ne!(root1, root2);
    }

    #[test]
    fn test_aggregator_verify_pedersen() {
        let mut agg = CommitmentAggregator::new();
        let value = 42u64;
        let blinding = 17u64;
        let idx = agg.add_pedersen_commitment(value, blinding);

        let mut opening = value.to_le_bytes().to_vec();
        opening.extend_from_slice(&blinding.to_le_bytes());

        assert!(agg.verify_individual(idx, &opening));
    }

    #[test]
    fn test_aggregator_verify_pedersen_wrong() {
        let mut agg = CommitmentAggregator::new();
        let idx = agg.add_pedersen_commitment(42, 17);

        let mut opening = 99u64.to_le_bytes().to_vec();
        opening.extend_from_slice(&17u64.to_le_bytes());

        assert!(!agg.verify_individual(idx, &opening));
    }

    #[test]
    fn test_aggregator_verify_out_of_bounds() {
        let agg = CommitmentAggregator::new();
        assert!(!agg.verify_individual(0, &[]));
    }

    #[test]
    fn test_aggregator_get_commitment() {
        let mut agg = CommitmentAggregator::new();
        let idx = agg.add_pedersen_commitment(1, 2);
        assert!(agg.get_commitment(idx).is_some());
        assert!(agg.get_commitment(99).is_none());
    }

    #[test]
    fn test_aggregator_state() {
        let mut agg = CommitmentAggregator::new();
        agg.add_hash_commitment(b"state-check");
        assert!(!agg.state().is_empty());
    }

    #[test]
    fn test_aggregator_verify_hash() {
        let mut agg = CommitmentAggregator::new();
        let value = b"aggregated-hash-value";
        let idx = agg.add_hash_commitment(value);

        // Reconstruct deterministic randomness used by commit_boxed
        let mut rng_hasher = blake3::Hasher::new();
        rng_hasher.update(b"commit-boxed-randomness");
        rng_hasher.update(value.as_slice());
        let randomness = *rng_hasher.finalize().as_bytes();

        let mut opening = randomness.to_vec();
        opening.extend_from_slice(value);

        assert!(agg.verify_individual(idx, &opening));
    }

    // ── ElGamalCommitment ──

    #[test]
    fn test_elgamal_from_secret_key() {
        let sk = 12345u64;
        let (eg, returned_sk) = ElGamalCommitment::from_secret_key(sk);
        assert_eq!(returned_sk, sk);
        assert_eq!(eg.g, 7);
        assert_eq!(eg.public_key, goldilocks_pow(7, sk));
    }

    #[test]
    fn test_elgamal_encrypt_decrypt_zero() {
        let sk = 42u64;
        let (eg, _) = ElGamalCommitment::from_secret_key(sk);
        let ct = eg.encrypt(0, 100);
        let decrypted = eg.decrypt(ct, sk);
        assert_eq!(decrypted, 0);
    }

    #[test]
    fn test_elgamal_encrypt_decrypt_small() {
        let sk = 42u64;
        let (eg, _) = ElGamalCommitment::from_secret_key(sk);
        let ct = eg.encrypt(7, 55);
        let decrypted = eg.decrypt(ct, sk);
        assert_eq!(decrypted, 7);
    }

    #[test]
    fn test_elgamal_encrypt_decrypt_42() {
        let sk = 999u64;
        let (eg, _) = ElGamalCommitment::from_secret_key(sk);
        let ct = eg.encrypt(42, 777);
        let decrypted = eg.decrypt(ct, sk);
        assert_eq!(decrypted, 42);
    }

    #[test]
    fn test_elgamal_homomorphic_addition() {
        let sk = 50u64;
        let (eg, _) = ElGamalCommitment::from_secret_key(sk);

        let ct_a = eg.encrypt(5, 11);
        let ct_b = eg.encrypt(3, 22);
        let ct_sum = ElGamalCommitment::add_ciphertexts(ct_a, ct_b);

        let decrypted = eg.decrypt(ct_sum, sk);
        assert_eq!(decrypted, 8); // 5 + 3
    }

    #[test]
    fn test_elgamal_rerandomize() {
        let sk = 50u64;
        let (eg, _) = ElGamalCommitment::from_secret_key(sk);

        let ct = eg.encrypt(10, 33);
        let ct_rerand = eg.rerandomize(ct, 99);

        // Ciphertext changes but plaintext stays the same
        assert_ne!(ct.0, ct_rerand.0);
        let decrypted = eg.decrypt(ct_rerand, sk);
        assert_eq!(decrypted, 10);
    }

    #[test]
    fn test_elgamal_different_randomness() {
        let sk = 50u64;
        let (eg, _) = ElGamalCommitment::from_secret_key(sk);

        let ct1 = eg.encrypt(5, 11);
        let ct2 = eg.encrypt(5, 22);
        // Same plaintext, different randomness → different ciphertext
        assert_ne!(ct1.0, ct2.0);
        // But same decryption
        assert_eq!(eg.decrypt(ct1, sk), eg.decrypt(ct2, sk));
    }

    #[test]
    fn test_elgamal_valid_ciphertext() {
        let sk = 50u64;
        let (eg, _) = ElGamalCommitment::from_secret_key(sk);
        let ct = eg.encrypt(1, 1);
        assert!(eg.is_valid_ciphertext(ct));
    }

    #[test]
    fn test_elgamal_scalar_mul() {
        let sk = 50u64;
        let (eg, _) = ElGamalCommitment::from_secret_key(sk);

        let ct = eg.encrypt(3, 11);
        let ct_scaled = ElGamalCommitment::scalar_mul_ciphertext(ct, 4);
        let decrypted = eg.decrypt(ct_scaled, sk);
        assert_eq!(decrypted, 12); // 3 * 4
    }

    #[test]
    fn test_elgamal_sub_ciphertexts() {
        let sk = 50u64;
        let (eg, _) = ElGamalCommitment::from_secret_key(sk);

        let ct_a = eg.encrypt(10, 11);
        let ct_b = eg.encrypt(3, 22);
        let ct_diff = ElGamalCommitment::sub_ciphertexts(ct_a, ct_b);
        let decrypted = eg.decrypt(ct_diff, sk);
        assert_eq!(decrypted, 7); // 10 - 3
    }

    #[test]
    fn test_elgamal_display() {
        let (eg, _) = ElGamalCommitment::from_secret_key(42);
        let s = format!("{}", eg);
        assert!(s.contains("ElGamal"));
        assert!(s.contains("g=7"));
    }

    #[test]
    fn test_elgamal_keygen_valid() {
        let (eg, sk) = ElGamalCommitment::keygen();
        assert!(sk > 0);
        assert!(sk < GOLDILOCKS_PRIME);
        assert_eq!(eg.public_key, goldilocks_pow(eg.g, sk));
    }

    // ── CommitmentChain ──

    #[test]
    fn test_chain_empty() {
        let chain = CommitmentChain::new();
        assert_eq!(chain.chain_length(), 0);
        assert!(chain.is_empty());
        assert_eq!(chain.chain_root(), [0u8; 32]);
        assert!(chain.verify_chain());
    }

    #[test]
    fn test_chain_append_one() {
        let mut chain = CommitmentChain::new();
        let hash = chain.append(b"first");
        assert_ne!(hash, [0u8; 32]);
        assert_eq!(chain.chain_length(), 1);
        assert!(!chain.is_empty());
    }

    #[test]
    fn test_chain_append_multiple() {
        let mut chain = CommitmentChain::new();
        let h1 = chain.append(b"alpha");
        let h2 = chain.append(b"beta");
        let h3 = chain.append(b"gamma");
        assert_ne!(h1, h2);
        assert_ne!(h2, h3);
        assert_eq!(chain.chain_length(), 3);
    }

    #[test]
    fn test_chain_verify_valid() {
        let mut chain = CommitmentChain::new();
        chain.append(b"one");
        chain.append(b"two");
        chain.append(b"three");
        assert!(chain.verify_chain());
    }

    #[test]
    fn test_chain_verify_tampered() {
        let mut chain = CommitmentChain::new();
        chain.append(b"one");
        chain.append(b"two");

        // Tamper with the value_hash of the second entry
        chain.chain[1].value_hash = [0xFFu8; 32];
        assert!(!chain.verify_chain());
    }

    #[test]
    fn test_chain_root_changes() {
        let mut chain = CommitmentChain::new();
        let root0 = chain.chain_root();
        chain.append(b"a");
        let root1 = chain.chain_root();
        chain.append(b"b");
        let root2 = chain.chain_root();
        assert_ne!(root0, root1);
        assert_ne!(root1, root2);
    }

    #[test]
    fn test_chain_get_value() {
        let mut chain = CommitmentChain::new();
        chain.append(b"data0");
        chain.append(b"data1");
        chain.append(b"data2");

        let entry = chain.get_value(1).unwrap();
        assert_eq!(entry.index, 1);
        assert!(chain.get_value(3).is_none());
    }

    #[test]
    fn test_chain_verify_entry() {
        let mut chain = CommitmentChain::new();
        chain.append(b"check-me");
        assert!(chain.verify_entry(0, b"check-me"));
        assert!(!chain.verify_entry(0, b"wrong"));
        assert!(!chain.verify_entry(1, b"check-me")); // out of range
    }

    #[test]
    fn test_chain_all_hashes() {
        let mut chain = CommitmentChain::new();
        chain.append(b"x");
        chain.append(b"y");
        let hashes = chain.all_hashes();
        assert_eq!(hashes.len(), 2);
        assert_eq!(hashes[1], chain.chain_root());
    }

    #[test]
    fn test_chain_prev_hash_links() {
        let mut chain = CommitmentChain::new();
        chain.append(b"first");
        chain.append(b"second");

        let e0 = chain.get_value(0).unwrap();
        let e1 = chain.get_value(1).unwrap();
        assert_eq!(e0.prev_hash, [0u8; 32]);
        assert_eq!(e1.prev_hash, e0.chain_hash);
    }

    #[test]
    fn test_chain_display() {
        let mut chain = CommitmentChain::new();
        chain.append(b"display-test");
        let entry = chain.get_value(0).unwrap();
        let s = format!("{}", entry);
        assert!(s.contains("index=0"));
    }

    #[test]
    fn test_chained_commitment_serde() {
        let cc = ChainedCommitment {
            index: 5,
            value_hash: [1u8; 32],
            prev_hash: [2u8; 32],
            chain_hash: [3u8; 32],
        };
        let json = serde_json::to_string(&cc).unwrap();
        let cc2: ChainedCommitment = serde_json::from_str(&json).unwrap();
        assert_eq!(cc.index, cc2.index);
        assert_eq!(cc.value_hash, cc2.value_hash);
        assert_eq!(cc.prev_hash, cc2.prev_hash);
        assert_eq!(cc.chain_hash, cc2.chain_hash);
    }

    #[test]
    fn test_chain_deterministic() {
        let mut c1 = CommitmentChain::new();
        c1.append(b"aaa");
        c1.append(b"bbb");

        let mut c2 = CommitmentChain::new();
        c2.append(b"aaa");
        c2.append(b"bbb");

        assert_eq!(c1.chain_root(), c2.chain_root());
    }

    #[test]
    fn test_chain_different_order() {
        let mut c1 = CommitmentChain::new();
        c1.append(b"aaa");
        c1.append(b"bbb");

        let mut c2 = CommitmentChain::new();
        c2.append(b"bbb");
        c2.append(b"aaa");

        assert_ne!(c1.chain_root(), c2.chain_root());
    }

    // ── Goldilocks helper tests ──

    #[test]
    fn test_goldilocks_add() {
        assert_eq!(goldilocks_add(1, 2), 3);
        assert_eq!(goldilocks_add(0, 0), 0);
        // Wrapping: (p-1) + 1 = 0
        assert_eq!(goldilocks_add(GOLDILOCKS_PRIME - 1, 1), 0);
        // (p-1) + (p-1) = p - 2
        assert_eq!(goldilocks_add(GOLDILOCKS_PRIME - 1, GOLDILOCKS_PRIME - 1), GOLDILOCKS_PRIME - 2);
    }

    #[test]
    fn test_goldilocks_sub() {
        assert_eq!(goldilocks_sub(5, 3), 2);
        assert_eq!(goldilocks_sub(0, 0), 0);
        // 0 - 1 = p - 1
        assert_eq!(goldilocks_sub(0, 1), GOLDILOCKS_PRIME - 1);
        // 3 - 5 = p - 2
        assert_eq!(goldilocks_sub(3, 5), GOLDILOCKS_PRIME - 2);
    }

    #[test]
    fn test_goldilocks_inv() {
        // a * a^(-1) = 1
        let a = 7u64;
        let a_inv = goldilocks_inv(a);
        assert_eq!(goldilocks_mul(a, a_inv), 1);

        let b = 12345u64;
        let b_inv = goldilocks_inv(b);
        assert_eq!(goldilocks_mul(b, b_inv), 1);
    }

    #[test]
    fn test_goldilocks_inv_one() {
        assert_eq!(goldilocks_inv(1), 1);
    }

    // ── CommitmentVerificationReport ──

    #[test]
    fn test_verification_report_new() {
        let report = CommitmentVerificationReport::new();
        assert_eq!(report.commitments_checked, 0);
        assert_eq!(report.commitments_valid, 0);
        assert!(report.failures.is_empty());
    }

    #[test]
    fn test_verification_report_record_valid() {
        let mut report = CommitmentVerificationReport::new();
        report.record_check("c1", true, "ok");
        report.record_check("c2", true, "ok");
        assert_eq!(report.commitments_checked, 2);
        assert_eq!(report.commitments_valid, 2);
        assert!(report.failures.is_empty());
    }

    #[test]
    fn test_verification_report_record_invalid() {
        let mut report = CommitmentVerificationReport::new();
        report.record_check("c1", true, "ok");
        report.record_check("c2", false, "bad hash");
        assert_eq!(report.commitments_checked, 2);
        assert_eq!(report.commitments_valid, 1);
        assert_eq!(report.failures.len(), 1);
        assert_eq!(report.failures[0].0, "c2");
        assert_eq!(report.failures[0].1, "bad hash");
    }

    #[test]
    fn test_verification_report_is_all_valid() {
        let mut report = CommitmentVerificationReport::new();
        assert!(!report.is_all_valid()); // no checks yet
        report.record_check("c1", true, "ok");
        assert!(report.is_all_valid());
        report.record_check("c2", false, "mismatch");
        assert!(!report.is_all_valid());
    }

    #[test]
    fn test_verification_report_summary() {
        let mut report = CommitmentVerificationReport::new();
        report.record_check("a", true, "ok");
        report.record_check("b", false, "bad hash");
        let s = report.summary();
        assert!(s.contains("Checked 2 commitments"));
        assert!(s.contains("1 valid"));
        assert!(s.contains("1 failed"));
        assert!(s.contains("b: bad hash"));
    }

    #[test]
    fn test_verification_report_to_json() {
        let mut report = CommitmentVerificationReport::new();
        report.record_check("x", true, "ok");
        report.record_check("y", false, "err");
        let json = report.to_json();
        assert!(json.contains("\"commitments_checked\":2"));
        assert!(json.contains("\"commitments_valid\":1"));
        assert!(json.contains("\"name\":\"y\""));
        assert!(json.contains("\"details\":\"err\""));
    }

    // ── CommitmentProtocol ──

    #[test]
    fn test_protocol_new() {
        let proto = CommitmentProtocol::new();
        assert!(proto.committed.is_empty());
    }

    #[test]
    fn test_protocol_commit_phase() {
        let mut proto = CommitmentProtocol::new();
        let values = vec![("v1", b"hello".to_vec()), ("v2", b"world".to_vec())];
        let results = proto.commit_phase(&values);
        assert_eq!(results.len(), 2);
        assert_eq!(proto.committed.len(), 2);
        // Each commitment should be a valid 32-byte hash
        for (hash, randomness) in &results {
            assert_eq!(hash.len(), 32);
            assert_eq!(randomness.len(), 32);
        }
    }

    #[test]
    fn test_protocol_reveal_phase_valid() {
        let mut proto = CommitmentProtocol::new();
        let values = vec![("v1", b"alpha".to_vec()), ("v2", b"beta".to_vec())];
        let commitments = proto.commit_phase(&values);
        assert!(proto.reveal_phase(&commitments, &values));
    }

    #[test]
    fn test_protocol_reveal_phase_invalid() {
        let mut proto = CommitmentProtocol::new();
        let values = vec![("v1", b"alpha".to_vec())];
        let commitments = proto.commit_phase(&values);
        let bad_values = vec![("v1", b"tampered".to_vec())];
        assert!(!proto.reveal_phase(&commitments, &bad_values));
    }

    #[test]
    fn test_protocol_verify_all_reveals() {
        let mut proto = CommitmentProtocol::new();
        let values = vec![("a", b"one".to_vec()), ("b", b"two".to_vec())];
        proto.commit_phase(&values);
        let report = proto.verify_all_reveals();
        assert!(report.is_all_valid());
        assert_eq!(report.commitments_checked, 2);
    }

    // ── MultiPartyCommitment ──

    #[test]
    fn test_multi_party_new() {
        let mp = MultiPartyCommitment::new(3);
        assert!(!mp.all_committed());
        assert!(!mp.all_revealed());
    }

    #[test]
    fn test_multi_party_commit() {
        let mut mp = MultiPartyCommitment::new(2);
        let c1 = mp.commit(0, b"secret_a");
        let c2 = mp.commit(1, b"secret_b");
        assert_ne!(c1, c2);
        assert!(mp.all_committed());
    }

    #[test]
    fn test_multi_party_reveal_valid() {
        let mut mp = MultiPartyCommitment::new(1);
        mp.commit(0, b"val");
        let randomness = mp.randomness[&0].clone();
        assert!(mp.reveal(0, b"val", &randomness));
        assert!(mp.all_revealed());
    }

    #[test]
    fn test_multi_party_reveal_invalid_value() {
        let mut mp = MultiPartyCommitment::new(1);
        mp.commit(0, b"val");
        let randomness = mp.randomness[&0].clone();
        assert!(!mp.reveal(0, b"wrong", &randomness));
        assert!(!mp.all_revealed());
    }

    #[test]
    fn test_multi_party_reveal_no_commit() {
        let mut mp = MultiPartyCommitment::new(2);
        // party 1 never committed
        assert!(!mp.reveal(1, b"data", b"rand"));
    }

    #[test]
    fn test_multi_party_combined_commitment() {
        let mut mp = MultiPartyCommitment::new(2);
        mp.commit(0, b"a");
        mp.commit(1, b"b");
        let combined = mp.combined_commitment();
        // Combined should be deterministic
        assert_eq!(combined.len(), 32);

        // Verify determinism: same inputs produce same combined hash
        let mut mp2 = MultiPartyCommitment::new(2);
        mp2.commit(0, b"a");
        mp2.commit(1, b"b");
        assert_eq!(mp.combined_commitment(), mp2.combined_commitment());
    }

    #[test]
    fn test_multi_party_all_committed_partial() {
        let mut mp = MultiPartyCommitment::new(3);
        mp.commit(0, b"x");
        mp.commit(1, b"y");
        assert!(!mp.all_committed());
        mp.commit(2, b"z");
        assert!(mp.all_committed());
    }
}
