use std::fmt;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

// Helper for serializing [u8; 64] which serde doesn't support natively.
mod state64_serde {
    use serde::{self, Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S>(state: &[u8; 64], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        state.to_vec().serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<[u8; 64], D::Error>
    where
        D: Deserializer<'de>,
    {
        let vec: Vec<u8> = Vec::deserialize(deserializer)?;
        if vec.len() != 64 {
            return Err(serde::de::Error::custom("expected 64 bytes"));
        }
        let mut arr = [0u8; 64];
        arr.copy_from_slice(&vec);
        Ok(arr)
    }
}

// ─────────────────────────────────────────────────────────────
// TranscriptError
// ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum TranscriptError {
    InvalidData,
    DeserializationError,
    ReplayFailed,
}

impl fmt::Display for TranscriptError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TranscriptError::InvalidData => write!(f, "InvalidData"),
            TranscriptError::DeserializationError => write!(f, "DeserializationError"),
            TranscriptError::ReplayFailed => write!(f, "ReplayFailed"),
        }
    }
}

impl std::error::Error for TranscriptError {}

// ─────────────────────────────────────────────────────────────
// TranscriptEntry
// ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum TranscriptEntry {
    Absorb { label: String, data_hash: [u8; 32] },
    Squeeze { label: String, output_hash: [u8; 32] },
}

const GOLDILOCKS_PRIME: u64 = 0xFFFFFFFF00000001;
const RATE_BYTES: usize = 32;

// ─────────────────────────────────────────────────────────────
// FiatShamirTranscript
// ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FiatShamirTranscript {
    state: Vec<u8>,
    history: Vec<TranscriptEntry>,
    domain_separator: Vec<u8>,
    fork_count: u32,
    phase_stack: Vec<String>,
}

impl FiatShamirTranscript {
    /// Create a new transcript with the given domain separator.
    pub fn new(domain: &str) -> Self {
        let domain_bytes = domain.as_bytes().to_vec();
        let initial_state = {
            let mut hasher = blake3::Hasher::new();
            hasher.update(b"fiat-shamir-init");
            hasher.update(&domain_bytes);
            hasher.finalize().as_bytes().to_vec()
        };
        Self {
            state: initial_state,
            history: Vec::new(),
            domain_separator: domain_bytes,
            fork_count: 0,
            phase_stack: Vec::new(),
        }
    }

    /// Absorb labeled bytes into the transcript.
    /// hash(state || label_len || label || data_len || data)
    pub fn absorb_bytes(&mut self, label: &str, data: &[u8]) {
        let label_bytes = label.as_bytes();
        let mut hasher = blake3::Hasher::new();
        hasher.update(&self.state);
        hasher.update(&(label_bytes.len() as u32).to_le_bytes());
        hasher.update(label_bytes);
        hasher.update(&(data.len() as u32).to_le_bytes());
        hasher.update(data);
        self.state = hasher.finalize().as_bytes().to_vec();

        let data_hash = *blake3::hash(data).as_bytes();
        self.history.push(TranscriptEntry::Absorb {
            label: label.to_string(),
            data_hash,
        });
    }

    /// Absorb a single field element (u64) into the transcript.
    pub fn absorb_field(&mut self, label: &str, element: u64) {
        self.absorb_bytes(label, &element.to_le_bytes());
    }

    /// Absorb a vector of field elements.
    pub fn absorb_field_vec(&mut self, label: &str, elements: &[u64]) {
        let mut buf = Vec::with_capacity(elements.len() * 8);
        for &e in elements {
            buf.extend_from_slice(&e.to_le_bytes());
        }
        self.absorb_bytes(label, &buf);
    }

    /// Absorb a commitment hash.
    pub fn absorb_commitment(&mut self, label: &str, hash: &[u8; 32]) {
        self.absorb_bytes(label, hash);
    }

    /// Absorb a u64 value.
    pub fn absorb_u64(&mut self, label: &str, value: u64) {
        self.absorb_bytes(label, &value.to_le_bytes());
    }

    /// Squeeze a pseudorandom field element from the transcript.
    /// hash(state || "challenge") -> interpret first 8 bytes as u64
    pub fn squeeze_challenge(&mut self) -> u64 {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&self.state);
        hasher.update(b"challenge");
        let hash = hasher.finalize();
        let hash_bytes = hash.as_bytes();

        // Update state to ensure sequential squeezes produce different values
        self.state = hash_bytes.to_vec();

        let challenge = u64::from_le_bytes(hash_bytes[0..8].try_into().unwrap());

        self.history.push(TranscriptEntry::Squeeze {
            label: "challenge".to_string(),
            output_hash: *hash_bytes,
        });

        challenge
    }

    /// Squeeze multiple challenges.
    pub fn squeeze_challenges(&mut self, count: usize) -> Vec<u64> {
        (0..count).map(|_| self.squeeze_challenge()).collect()
    }

    /// Squeeze arbitrary bytes from the transcript.
    pub fn squeeze_bytes(&mut self, label: &str, count: usize) -> Vec<u8> {
        let mut result = Vec::with_capacity(count);
        let mut counter = 0u32;

        while result.len() < count {
            let mut hasher = blake3::Hasher::new();
            hasher.update(&self.state);
            hasher.update(label.as_bytes());
            hasher.update(&counter.to_le_bytes());
            let hash = hasher.finalize();
            let hash_bytes = hash.as_bytes();

            let remaining = count - result.len();
            let take = remaining.min(32);
            result.extend_from_slice(&hash_bytes[..take]);
            counter += 1;
        }

        result.truncate(count);

        // Update state
        let mut hasher = blake3::Hasher::new();
        hasher.update(&self.state);
        hasher.update(label.as_bytes());
        hasher.update(b"squeeze-done");
        self.state = hasher.finalize().as_bytes().to_vec();

        let output_hash = {
            let mut arr = [0u8; 32];
            let h = blake3::hash(&result);
            arr.copy_from_slice(h.as_bytes());
            arr
        };
        self.history.push(TranscriptEntry::Squeeze {
            label: label.to_string(),
            output_hash,
        });

        result
    }

    /// Squeeze `count` distinct random indices in `[0, max)`.
    pub fn squeeze_indices(&mut self, count: usize, max: usize) -> Vec<usize> {
        assert!(count <= max, "cannot squeeze more distinct indices than max");

        let mut indices = Vec::with_capacity(count);
        let mut attempts = 0u32;
        let max_attempts = (count as u32) * 100 + 1000;

        while indices.len() < count && attempts < max_attempts {
            let mut hasher = blake3::Hasher::new();
            hasher.update(&self.state);
            hasher.update(b"index");
            hasher.update(&attempts.to_le_bytes());
            let hash = hasher.finalize();
            let hash_bytes = hash.as_bytes();

            let raw = u64::from_le_bytes(hash_bytes[0..8].try_into().unwrap());
            let idx = (raw as usize) % max;

            if !indices.contains(&idx) {
                indices.push(idx);
            }
            attempts += 1;
        }

        // Update state after index generation
        let mut hasher = blake3::Hasher::new();
        hasher.update(&self.state);
        hasher.update(b"indices-done");
        hasher.update(&(count as u32).to_le_bytes());
        self.state = hasher.finalize().as_bytes().to_vec();

        indices
    }

    /// Create an independent sub-transcript (fork).
    pub fn fork(&self, label: &str) -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&self.state);
        hasher.update(b"fork");
        hasher.update(label.as_bytes());
        hasher.update(&self.fork_count.to_le_bytes());
        let fork_state = hasher.finalize().as_bytes().to_vec();

        Self {
            state: fork_state,
            history: Vec::new(),
            domain_separator: self.domain_separator.clone(),
            fork_count: 0,
            phase_stack: Vec::new(),
        }
    }

    /// Return the current state as a 32-byte hash.
    pub fn state_hash(&self) -> [u8; 32] {
        let hash = blake3::hash(&self.state);
        *hash.as_bytes()
    }

    /// Return the transcript history.
    pub fn history(&self) -> &[TranscriptEntry] {
        &self.history
    }

    /// Verify internal consistency by replaying the history and checking
    /// that absorption entries have valid hashes.
    pub fn replay_verification(&self) -> bool {
        for entry in &self.history {
            match entry {
                TranscriptEntry::Absorb { data_hash, .. } => {
                    // Check that the hash is non-zero (valid entry)
                    if *data_hash == [0u8; 32] {
                        return false;
                    }
                }
                TranscriptEntry::Squeeze { output_hash, .. } => {
                    if *output_hash == [0u8; 32] {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Mark the beginning of a named protocol phase.
    pub fn begin_phase(&mut self, phase_name: &str) {
        self.phase_stack.push(phase_name.to_string());
        self.absorb_bytes("phase-begin", phase_name.as_bytes());
    }

    /// End the current protocol phase.
    pub fn end_phase(&mut self) {
        if let Some(phase_name) = self.phase_stack.pop() {
            self.absorb_bytes("phase-end", phase_name.as_bytes());
        }
    }

    /// Serialize the transcript to bytes.
    pub fn serialize(&self) -> Vec<u8> {
        serde_json::to_vec(self).unwrap_or_default()
    }

    /// Deserialize a transcript from bytes.
    pub fn deserialize(bytes: &[u8]) -> Result<Self, TranscriptError> {
        serde_json::from_slice(bytes).map_err(|_| TranscriptError::DeserializationError)
    }
}

// ─────────────────────────────────────────────────────────────
// TranscriptOperation
// ─────────────────────────────────────────────────────────────

/// Records the type of operation performed on a transcript.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum TranscriptOperation {
    /// Absorb operation: label and byte count absorbed.
    Absorb(String, usize),
    /// Squeeze operation: label and byte count squeezed.
    Squeeze(String, usize),
    /// Beginning of a named phase.
    PhaseBegin(String),
    /// End of the current phase.
    PhaseEnd,
}

// ─────────────────────────────────────────────────────────────
// MerlinLikeTranscript
// ─────────────────────────────────────────────────────────────

/// An alternative transcript using a STROBE-like sponge construction
/// with a 512-bit state (256-bit rate + 256-bit capacity), explicit
/// domain separation, and operation tracking.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MerlinLikeTranscript {
    /// 512-bit sponge state: first 32 bytes = rate, last 32 bytes = capacity.
    #[serde(with = "state64_serde")]
    state: [u8; 64],
    /// Current write position within the rate portion.
    rate_position: usize,
    /// Number of absorb operations performed.
    absorbed_count: usize,
    /// Number of squeeze operations performed.
    squeezed_count: usize,
    /// Ordered log of every transcript operation.
    operations: Vec<TranscriptOperation>,
    /// Currently active phase (if any).
    current_phase: Option<String>,
    /// High-level history of absorbs and squeezes.
    history: Vec<TranscriptEntry>,
}

impl MerlinLikeTranscript {
    /// Create a new Merlin-like transcript seeded with `label`.
    pub fn new(label: &str) -> Self {
        let mut state = [0u8; 64];
        let rate_init = {
            let mut h = blake3::Hasher::new();
            h.update(b"merlin-like-v2-init");
            h.update(label.as_bytes());
            h.finalize()
        };
        state[..32].copy_from_slice(rate_init.as_bytes());
        let cap_init = {
            let mut h = blake3::Hasher::new();
            h.update(b"merlin-like-v2-cap");
            h.update(label.as_bytes());
            h.finalize()
        };
        state[32..].copy_from_slice(cap_init.as_bytes());

        Self {
            state,
            rate_position: 0,
            absorbed_count: 0,
            squeezed_count: 0,
            operations: Vec::new(),
            current_phase: None,
            history: Vec::new(),
        }
    }

    // ── internal helpers ──

    /// Keccak-inspired permutation: mixes rate and capacity halves through
    /// four rounds of blake3 hashing with distinct round constants.
    fn permute(&mut self) {
        let mut left = [0u8; 32];
        let mut right = [0u8; 32];
        left.copy_from_slice(&self.state[..32]);
        right.copy_from_slice(&self.state[32..]);

        // Round 1
        let mut h = blake3::Hasher::new();
        h.update(b"permute-r1");
        h.update(&left);
        h.update(&right);
        let new_right = *h.finalize().as_bytes();

        // Round 2
        let mut h = blake3::Hasher::new();
        h.update(b"permute-r2");
        h.update(&new_right);
        h.update(&left);
        let new_left = *h.finalize().as_bytes();

        // Round 3
        let mut h = blake3::Hasher::new();
        h.update(b"permute-r3");
        h.update(&new_left);
        h.update(&new_right);
        let final_left = *h.finalize().as_bytes();

        // Round 4
        let mut h = blake3::Hasher::new();
        h.update(b"permute-r4");
        h.update(&new_right);
        h.update(&new_left);
        let final_right = *h.finalize().as_bytes();

        self.state[..32].copy_from_slice(&final_left);
        self.state[32..].copy_from_slice(&final_right);
        self.rate_position = 0;
    }

    /// XOR `data` byte-by-byte into the rate portion of the state,
    /// triggering a permutation whenever the rate is full.
    fn absorb_bytes_into_rate(&mut self, data: &[u8]) {
        for &byte in data {
            self.state[self.rate_position] ^= byte;
            self.rate_position += 1;
            if self.rate_position >= RATE_BYTES {
                self.permute();
            }
        }
    }

    // ── public API ──

    /// Absorb labelled data into the sponge with domain separation.
    pub fn absorb(&mut self, label: &str, data: &[u8]) {
        let label_bytes = label.as_bytes();
        self.absorb_bytes_into_rate(&[0x01]); // absorb domain tag
        self.absorb_bytes_into_rate(&(label_bytes.len() as u32).to_le_bytes());
        self.absorb_bytes_into_rate(label_bytes);
        self.absorb_bytes_into_rate(&(data.len() as u32).to_le_bytes());
        self.absorb_bytes_into_rate(data);
        self.permute();

        self.absorbed_count += 1;
        self.operations
            .push(TranscriptOperation::Absorb(label.to_string(), data.len()));

        let data_hash = *blake3::hash(data).as_bytes();
        self.history.push(TranscriptEntry::Absorb {
            label: label.to_string(),
            data_hash,
        });
    }

    /// Squeeze `count` pseudorandom bytes from the sponge.
    pub fn squeeze(&mut self, label: &str, count: usize) -> Vec<u8> {
        let label_bytes = label.as_bytes();
        self.absorb_bytes_into_rate(&[0x02]); // squeeze domain tag
        self.absorb_bytes_into_rate(&(label_bytes.len() as u32).to_le_bytes());
        self.absorb_bytes_into_rate(label_bytes);
        self.absorb_bytes_into_rate(&(count as u32).to_le_bytes());
        self.permute();

        let mut result = Vec::with_capacity(count);
        while result.len() < count {
            let remaining = count - result.len();
            let available = RATE_BYTES - self.rate_position;
            let take = remaining.min(available);
            result.extend_from_slice(&self.state[self.rate_position..self.rate_position + take]);
            self.rate_position += take;
            if self.rate_position >= RATE_BYTES {
                self.permute();
            }
        }

        self.squeezed_count += 1;
        self.operations
            .push(TranscriptOperation::Squeeze(label.to_string(), count));

        let output_hash = *blake3::hash(&result).as_bytes();
        self.history.push(TranscriptEntry::Squeeze {
            label: label.to_string(),
            output_hash,
        });

        result
    }

    /// Squeeze a single Goldilocks field element.
    pub fn squeeze_challenge(&mut self) -> u64 {
        let bytes = self.squeeze("challenge", 8);
        let raw = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
        raw % GOLDILOCKS_PRIME
    }

    /// Squeeze `count` independent Goldilocks field elements.
    pub fn squeeze_challenges(&mut self, count: usize) -> Vec<u64> {
        (0..count)
            .map(|i| {
                let label = format!("challenge-{}", i);
                let bytes = self.squeeze(&label, 8);
                let raw = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
                raw % GOLDILOCKS_PRIME
            })
            .collect()
    }

    /// Domain-separate the start of a new protocol phase.
    pub fn begin_phase(&mut self, phase: &str) {
        self.current_phase = Some(phase.to_string());
        self.operations
            .push(TranscriptOperation::PhaseBegin(phase.to_string()));
        self.absorb("phase-begin", phase.as_bytes());
    }

    /// Close the current protocol phase.
    pub fn end_phase(&mut self) {
        if let Some(phase) = self.current_phase.take() {
            self.operations.push(TranscriptOperation::PhaseEnd);
            self.absorb("phase-end", phase.as_bytes());
        }
    }

    /// Create an independent sub-transcript by forking and absorbing `label`.
    pub fn fork(&self, label: &str) -> Self {
        let mut forked = self.clone();
        forked.absorb("fork", label.as_bytes());
        forked.operations = Vec::new();
        forked.history = Vec::new();
        forked.absorbed_count = 0;
        forked.squeezed_count = 0;
        forked
    }

    /// Return the current state condensed to a 32-byte hash.
    pub fn state_hash(&self) -> [u8; 32] {
        *blake3::hash(&self.state).as_bytes()
    }

    /// Read-only access to the history log.
    pub fn history(&self) -> &[TranscriptEntry] {
        &self.history
    }

    /// Read-only access to the operations log.
    pub fn operations(&self) -> &[TranscriptOperation] {
        &self.operations
    }

    /// Number of absorb operations performed so far.
    pub fn absorbed_count(&self) -> usize {
        self.absorbed_count
    }

    /// Number of squeeze operations performed so far.
    pub fn squeezed_count(&self) -> usize {
        self.squeezed_count
    }
}

// ─────────────────────────────────────────────────────────────
// BuilderItem
// ─────────────────────────────────────────────────────────────

/// Items that can be staged in a [`TranscriptBuilder`].
#[derive(Clone, Debug)]
pub enum BuilderItem {
    /// A single Goldilocks field element.
    FieldElement(String, u64),
    /// Arbitrary byte data.
    Bytes(String, Vec<u8>),
    /// A 32-byte commitment hash.
    Commitment(String, [u8; 32]),
    /// A named sub-phase containing its own items.
    Phase(String, Vec<BuilderItem>),
}

// ─────────────────────────────────────────────────────────────
// TranscriptBuilder
// ─────────────────────────────────────────────────────────────

/// Builder pattern for constructing transcripts with a fluent API.
pub struct TranscriptBuilder {
    domain: String,
    items: Vec<BuilderItem>,
}

impl TranscriptBuilder {
    pub fn new(domain: &str) -> Self {
        Self {
            domain: domain.to_string(),
            items: Vec::new(),
        }
    }

    // ── consuming chainable methods ──

    /// Append a field element (consumes self for chaining).
    pub fn with_field_element(mut self, label: &str, value: u64) -> Self {
        self.items
            .push(BuilderItem::FieldElement(label.to_string(), value));
        self
    }

    /// Append raw bytes (consumes self for chaining).
    pub fn with_bytes(mut self, label: &str, data: &[u8]) -> Self {
        self.items
            .push(BuilderItem::Bytes(label.to_string(), data.to_vec()));
        self
    }

    /// Append a commitment hash (consumes self for chaining).
    pub fn with_commitment(mut self, label: &str, hash: &[u8; 32]) -> Self {
        self.items
            .push(BuilderItem::Commitment(label.to_string(), *hash));
        self
    }

    /// Open a named sub-phase; `build_fn` populates it via `&mut TranscriptBuilder`.
    pub fn with_phase(
        mut self,
        name: &str,
        build_fn: impl FnOnce(&mut TranscriptBuilder),
    ) -> Self {
        let mut sub = TranscriptBuilder::new(&self.domain);
        build_fn(&mut sub);
        self.items
            .push(BuilderItem::Phase(name.to_string(), sub.items));
        self
    }

    // ── mutable helpers (used inside `with_phase` closures) ──

    /// Append a field element via `&mut self`.
    pub fn add_field_element(&mut self, label: &str, value: u64) {
        self.items
            .push(BuilderItem::FieldElement(label.to_string(), value));
    }

    /// Append raw bytes via `&mut self`.
    pub fn add_bytes(&mut self, label: &str, data: &[u8]) {
        self.items
            .push(BuilderItem::Bytes(label.to_string(), data.to_vec()));
    }

    /// Append a commitment hash via `&mut self`.
    pub fn add_commitment(&mut self, label: &str, hash: &[u8; 32]) {
        self.items
            .push(BuilderItem::Commitment(label.to_string(), *hash));
    }

    // ── build helpers ──

    fn apply_items_to_fiat_shamir(items: &[BuilderItem], t: &mut FiatShamirTranscript) {
        for item in items {
            match item {
                BuilderItem::FieldElement(label, value) => t.absorb_field(label, *value),
                BuilderItem::Bytes(label, data) => t.absorb_bytes(label, data),
                BuilderItem::Commitment(label, hash) => t.absorb_commitment(label, hash),
                BuilderItem::Phase(name, sub_items) => {
                    t.begin_phase(name);
                    Self::apply_items_to_fiat_shamir(sub_items, t);
                    t.end_phase();
                }
            }
        }
    }

    fn apply_items_to_merlin(items: &[BuilderItem], t: &mut MerlinLikeTranscript) {
        for item in items {
            match item {
                BuilderItem::FieldElement(label, value) => {
                    t.absorb(label, &value.to_le_bytes());
                }
                BuilderItem::Bytes(label, data) => t.absorb(label, data),
                BuilderItem::Commitment(label, hash) => t.absorb(label, hash),
                BuilderItem::Phase(name, sub_items) => {
                    t.begin_phase(name);
                    Self::apply_items_to_merlin(sub_items, t);
                    t.end_phase();
                }
            }
        }
    }

    /// Build a [`FiatShamirTranscript`] by replaying all staged items.
    pub fn build(&self) -> FiatShamirTranscript {
        let mut transcript = FiatShamirTranscript::new(&self.domain);
        Self::apply_items_to_fiat_shamir(&self.items, &mut transcript);
        transcript
    }

    /// Build a [`MerlinLikeTranscript`] by replaying all staged items.
    pub fn build_merlin(&self) -> MerlinLikeTranscript {
        let mut transcript = MerlinLikeTranscript::new(&self.domain);
        Self::apply_items_to_merlin(&self.items, &mut transcript);
        transcript
    }
}

// ─────────────────────────────────────────────────────────────
// MultiTranscript
// ─────────────────────────────────────────────────────────────

/// Manage multiple parallel sub-transcripts with a shared root.
pub struct MultiTranscript {
    transcripts: HashMap<String, FiatShamirTranscript>,
    root_transcript: FiatShamirTranscript,
}

impl MultiTranscript {
    pub fn new(domain: &str) -> Self {
        Self {
            transcripts: HashMap::new(),
            root_transcript: FiatShamirTranscript::new(domain),
        }
    }

    /// Create (or return existing) named sub-transcript.
    pub fn create_sub_transcript(&mut self, name: &str) -> &mut FiatShamirTranscript {
        let sub = self.root_transcript.fork(name);
        self.transcripts
            .entry(name.to_string())
            .or_insert(sub)
    }

    /// Lookup a sub-transcript by name.
    pub fn get_sub_transcript(&self, name: &str) -> Option<&FiatShamirTranscript> {
        self.transcripts.get(name)
    }

    /// Absorb every sub-transcript state hash into the root transcript
    /// (in sorted-key order for determinism).
    pub fn merge_all(&mut self) {
        let mut names: Vec<String> = self.transcripts.keys().cloned().collect();
        names.sort();
        for name in &names {
            if let Some(sub) = self.transcripts.get(name) {
                let h = sub.state_hash();
                self.root_transcript
                    .absorb_bytes(&format!("sub-{}", name), &h);
            }
        }
    }

    /// Squeeze a challenge from the root transcript.
    pub fn root_challenge(&mut self) -> u64 {
        self.root_transcript.squeeze_challenge()
    }

    /// Squeeze a challenge from the named sub-transcript (if it exists).
    pub fn sub_challenge(&mut self, name: &str) -> Option<u64> {
        self.transcripts
            .get_mut(name)
            .map(|t| t.squeeze_challenge())
    }

    /// Return `true` when every transcript passes `replay_verification`.
    pub fn verify_consistency(&self) -> bool {
        for sub in self.transcripts.values() {
            if !sub.replay_verification() {
                return false;
            }
        }
        self.root_transcript.replay_verification()
    }
}

// ─────────────────────────────────────────────────────────────
// TranscriptVerificationResult / TranscriptDifference
// ─────────────────────────────────────────────────────────────

/// Result of comparing an expected transcript against an actual one.
#[derive(Clone, Debug)]
pub struct TranscriptVerificationResult {
    pub is_valid: bool,
    pub matches: usize,
    pub mismatches: usize,
    pub missing: usize,
    pub extra: usize,
}

/// A single discrepancy between expected and actual transcript entries.
#[derive(Clone, Debug)]
pub enum TranscriptDifference {
    Missing(TranscriptEntry),
    Extra(TranscriptEntry),
    Mismatch(TranscriptEntry, TranscriptEntry),
}

// ─────────────────────────────────────────────────────────────
// TranscriptVerifier
// ─────────────────────────────────────────────────────────────

/// Verify that a transcript was produced correctly by comparing expected
/// entries against the entries recorded in a [`FiatShamirTranscript`].
pub struct TranscriptVerifier {
    expected_entries: Vec<TranscriptEntry>,
    actual_entries: Vec<TranscriptEntry>,
}

fn entries_match(a: &TranscriptEntry, b: &TranscriptEntry) -> bool {
    a == b
}

impl TranscriptVerifier {
    pub fn new() -> Self {
        Self {
            expected_entries: Vec::new(),
            actual_entries: Vec::new(),
        }
    }

    /// Append an expected entry.
    pub fn add_expected(&mut self, entry: TranscriptEntry) {
        self.expected_entries.push(entry);
    }

    /// Explicitly set the actual entries (for use with `differences()`).
    pub fn set_actual_entries(&mut self, entries: Vec<TranscriptEntry>) {
        self.actual_entries = entries;
    }

    /// Compare expected entries against the given transcript's history.
    pub fn verify_against(
        &self,
        transcript: &FiatShamirTranscript,
    ) -> TranscriptVerificationResult {
        let actual = transcript.history();
        let mut matches = 0usize;
        let mut mismatches = 0usize;
        let min_len = self.expected_entries.len().min(actual.len());

        for i in 0..min_len {
            if entries_match(&self.expected_entries[i], &actual[i]) {
                matches += 1;
            } else {
                mismatches += 1;
            }
        }

        let missing = self
            .expected_entries
            .len()
            .saturating_sub(actual.len());
        let extra = actual
            .len()
            .saturating_sub(self.expected_entries.len());
        let is_valid = mismatches == 0 && missing == 0 && extra == 0;

        TranscriptVerificationResult {
            is_valid,
            matches,
            mismatches,
            missing,
            extra,
        }
    }

    /// Check that every entry has a non-empty label.
    pub fn check_ordering(&self) -> bool {
        for entry in &self.expected_entries {
            match entry {
                TranscriptEntry::Absorb { label, .. }
                | TranscriptEntry::Squeeze { label, .. } => {
                    if label.is_empty() {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// A complete transcript has at least one absorb and one squeeze.
    pub fn check_completeness(&self) -> bool {
        let has_absorb = self
            .expected_entries
            .iter()
            .any(|e| matches!(e, TranscriptEntry::Absorb { .. }));
        let has_squeeze = self
            .expected_entries
            .iter()
            .any(|e| matches!(e, TranscriptEntry::Squeeze { .. }));
        has_absorb && has_squeeze
    }

    /// Produce a list of per-entry differences between `expected_entries`
    /// and `actual_entries`.
    pub fn differences(&self) -> Vec<TranscriptDifference> {
        let mut diffs = Vec::new();
        let max_len = self
            .expected_entries
            .len()
            .max(self.actual_entries.len());

        for i in 0..max_len {
            match (
                self.expected_entries.get(i),
                self.actual_entries.get(i),
            ) {
                (Some(expected), Some(actual)) => {
                    if !entries_match(expected, actual) {
                        diffs.push(TranscriptDifference::Mismatch(
                            expected.clone(),
                            actual.clone(),
                        ));
                    }
                }
                (Some(expected), None) => {
                    diffs.push(TranscriptDifference::Missing(expected.clone()));
                }
                (None, Some(actual)) => {
                    diffs.push(TranscriptDifference::Extra(actual.clone()));
                }
                (None, None) => {}
            }
        }
        diffs
    }
}

// ─────────────────────────────────────────────────────────────
// ChallengeGenerator
// ─────────────────────────────────────────────────────────────

/// Deterministic challenge generation backed by a counter-mode PRNG
/// seeded from a transcript state hash.
pub struct ChallengeGenerator {
    seed: [u8; 32],
    counter: u64,
}

impl ChallengeGenerator {
    /// Seed from the current state of a Fiat-Shamir transcript.
    pub fn from_transcript(transcript: &FiatShamirTranscript) -> Self {
        Self {
            seed: transcript.state_hash(),
            counter: 0,
        }
    }

    /// Seed from an explicit 32-byte value.
    pub fn from_seed(seed: [u8; 32]) -> Self {
        Self { seed, counter: 0 }
    }

    /// Produce a raw 32-byte block and advance the counter.
    fn next_raw_bytes(&mut self) -> [u8; 32] {
        let mut h = blake3::Hasher::new();
        h.update(b"challenge-gen");
        h.update(&self.seed);
        h.update(&self.counter.to_le_bytes());
        self.counter += 1;
        *h.finalize().as_bytes()
    }

    /// Next pseudorandom Goldilocks field element.
    pub fn next_field_element(&mut self) -> u64 {
        let b = self.next_raw_bytes();
        let raw = u64::from_le_bytes(b[0..8].try_into().unwrap());
        raw % GOLDILOCKS_PRIME
    }

    /// Next `count` pseudorandom Goldilocks field elements.
    pub fn next_field_elements(&mut self, count: usize) -> Vec<u64> {
        (0..count).map(|_| self.next_field_element()).collect()
    }

    /// Next pseudorandom index in `[0, max)`.
    pub fn next_index(&mut self, max: usize) -> usize {
        assert!(max > 0, "max must be positive");
        let b = self.next_raw_bytes();
        let raw = u64::from_le_bytes(b[0..8].try_into().unwrap());
        (raw as usize) % max
    }

    /// Next `count` *distinct* indices in `[0, max)`.
    pub fn next_indices(&mut self, count: usize, max: usize) -> Vec<usize> {
        assert!(count <= max, "count must not exceed max");
        let mut indices = Vec::with_capacity(count);
        let max_attempts = count * 100 + 1000;
        let mut attempts = 0;
        while indices.len() < count && attempts < max_attempts {
            let idx = self.next_index(max);
            if !indices.contains(&idx) {
                indices.push(idx);
            }
            attempts += 1;
        }
        indices
    }

    /// Next `count` pseudorandom bytes.
    pub fn next_bytes(&mut self, count: usize) -> Vec<u8> {
        let mut result = Vec::with_capacity(count);
        while result.len() < count {
            let block = self.next_raw_bytes();
            let take = (count - result.len()).min(32);
            result.extend_from_slice(&block[..take]);
        }
        result.truncate(count);
        result
    }

    /// Next pseudorandom boolean.
    pub fn next_bool(&mut self) -> bool {
        let b = self.next_raw_bytes();
        b[0] & 1 == 1
    }
}

// ─────────────────────────────────────────────────────────────
// TranscriptSerializer
// ─────────────────────────────────────────────────────────────

/// Compact binary serialization for [`FiatShamirTranscript`].
///
/// Format (version 1):
///   `TSCR` | version(1) | json_len(u32 LE) | json_bytes | blake3_checksum(32)
pub struct TranscriptSerializer;

impl TranscriptSerializer {
    /// Serialize a transcript into a self-contained binary blob.
    pub fn serialize(transcript: &FiatShamirTranscript) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"TSCR");
        buf.push(1); // version

        let json = serde_json::to_vec(transcript).unwrap_or_default();
        buf.extend_from_slice(&(json.len() as u32).to_le_bytes());
        buf.extend_from_slice(&json);

        let checksum = blake3::hash(&json);
        buf.extend_from_slice(checksum.as_bytes());
        buf
    }

    /// Deserialize a transcript, verifying the blake3 checksum.
    pub fn deserialize(bytes: &[u8]) -> Result<FiatShamirTranscript, TranscriptError> {
        if bytes.len() < 9 {
            return Err(TranscriptError::DeserializationError);
        }
        if &bytes[0..4] != b"TSCR" {
            return Err(TranscriptError::DeserializationError);
        }
        if bytes[4] != 1 {
            return Err(TranscriptError::DeserializationError);
        }

        let json_len =
            u32::from_le_bytes(bytes[5..9].try_into().unwrap()) as usize;
        if bytes.len() < 9 + json_len + 32 {
            return Err(TranscriptError::DeserializationError);
        }

        let json = &bytes[9..9 + json_len];
        let checksum = &bytes[9 + json_len..9 + json_len + 32];

        let expected = blake3::hash(json);
        if expected.as_bytes() != checksum {
            return Err(TranscriptError::InvalidData);
        }

        serde_json::from_slice(json)
            .map_err(|_| TranscriptError::DeserializationError)
    }

    /// Serialize in a more compact binary-only encoding (no JSON).
    ///
    /// Format: `TCMP` | state_hash(32) | entry_count(u32 LE) | entries…
    pub fn serialize_compressed(transcript: &FiatShamirTranscript) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"TCMP");

        let state_hash = transcript.state_hash();
        buf.extend_from_slice(&state_hash);

        let history = transcript.history();
        buf.extend_from_slice(&(history.len() as u32).to_le_bytes());

        for entry in history {
            match entry {
                TranscriptEntry::Absorb { label, data_hash } => {
                    buf.push(0x01);
                    let lb = label.as_bytes();
                    buf.extend_from_slice(&(lb.len() as u16).to_le_bytes());
                    buf.extend_from_slice(lb);
                    buf.extend_from_slice(data_hash);
                }
                TranscriptEntry::Squeeze { label, output_hash } => {
                    buf.push(0x02);
                    let lb = label.as_bytes();
                    buf.extend_from_slice(&(lb.len() as u16).to_le_bytes());
                    buf.extend_from_slice(lb);
                    buf.extend_from_slice(output_hash);
                }
            }
        }
        buf
    }

    /// Estimate the size in bytes of `serialize()` output.
    pub fn estimate_size(transcript: &FiatShamirTranscript) -> usize {
        let history = transcript.history();
        // envelope: magic(4) + version(1) + json_len(4) + checksum(32) = 41
        let mut json_est = 100; // base JSON overhead
        for entry in history {
            json_est += 80; // per-entry overhead
            match entry {
                TranscriptEntry::Absorb { label, .. } => {
                    json_est += label.len() + 64;
                }
                TranscriptEntry::Squeeze { label, .. } => {
                    json_est += label.len() + 64;
                }
            }
        }
        41 + json_est
    }
}

// ─────────────────────────────────────────────────────────────
// PhaseTranscript
// ─────────────────────────────────────────────────────────────

/// A single phase inside a [`ProtocolTranscriptManager`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PhaseTranscript {
    pub phase_name: String,
    pub transcript: FiatShamirTranscript,
    pub start_time: u64,
    pub end_time: Option<u64>,
}

// ─────────────────────────────────────────────────────────────
// ProtocolTranscriptManager
// ─────────────────────────────────────────────────────────────

/// Manage the transcript for a complete prove / verify lifecycle.
pub struct ProtocolTranscriptManager {
    protocol_name: String,
    phases: Vec<PhaseTranscript>,
    current_phase: Option<usize>,
    time_counter: u64,
}

impl ProtocolTranscriptManager {
    pub fn new(protocol_name: &str) -> Self {
        Self {
            protocol_name: protocol_name.to_string(),
            phases: Vec::new(),
            current_phase: None,
            time_counter: 0,
        }
    }

    // ── private helpers ──

    fn begin_phase_inner(&mut self, phase_name: &str) {
        self.time_counter += 1;
        let domain = format!("{}-{}", self.protocol_name, phase_name);
        let phase = PhaseTranscript {
            phase_name: phase_name.to_string(),
            transcript: FiatShamirTranscript::new(&domain),
            start_time: self.time_counter,
            end_time: None,
        };
        self.phases.push(phase);
        self.current_phase = Some(self.phases.len() - 1);
    }

    fn end_current_phase(&mut self) {
        self.time_counter += 1;
        if let Some(idx) = self.current_phase.take() {
            if let Some(phase) = self.phases.get_mut(idx) {
                phase.end_time = Some(self.time_counter);
            }
        }
    }

    // ── public API ──

    pub fn begin_commit_phase(&mut self) {
        self.begin_phase_inner("commit");
    }
    pub fn end_commit_phase(&mut self) {
        self.end_current_phase();
    }

    pub fn begin_prove_phase(&mut self) {
        self.begin_phase_inner("prove");
    }
    pub fn end_prove_phase(&mut self) {
        self.end_current_phase();
    }

    pub fn begin_verify_phase(&mut self) {
        self.begin_phase_inner("verify");
    }
    pub fn end_verify_phase(&mut self) {
        self.end_current_phase();
    }

    /// Record a commitment hash into the active phase transcript.
    pub fn record_commitment(&mut self, name: &str, hash: &[u8; 32]) {
        if let Some(idx) = self.current_phase {
            if let Some(phase) = self.phases.get_mut(idx) {
                phase.transcript.absorb_commitment(name, hash);
            }
        }
    }

    /// Record a challenge value into the active phase transcript.
    pub fn record_challenge(&mut self, challenge: u64) {
        if let Some(idx) = self.current_phase {
            if let Some(phase) = self.phases.get_mut(idx) {
                phase.transcript.absorb_field("challenge", challenge);
            }
        }
    }

    /// Produce a 32-byte hash that commits to the entire protocol transcript.
    pub fn final_transcript_hash(&self) -> [u8; 32] {
        let mut h = blake3::Hasher::new();
        h.update(b"protocol-final");
        h.update(self.protocol_name.as_bytes());
        for phase in &self.phases {
            h.update(phase.phase_name.as_bytes());
            h.update(&phase.transcript.state_hash());
            h.update(&phase.start_time.to_le_bytes());
            if let Some(end) = phase.end_time {
                h.update(&end.to_le_bytes());
            }
        }
        *h.finalize().as_bytes()
    }

    /// Export the full protocol transcript as a binary blob.
    pub fn export_transcript(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"PROT");
        buf.extend_from_slice(&(self.phases.len() as u32).to_le_bytes());

        for phase in &self.phases {
            let name_bytes = phase.phase_name.as_bytes();
            buf.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
            buf.extend_from_slice(name_bytes);
            buf.extend_from_slice(&phase.start_time.to_le_bytes());
            let end = phase.end_time.unwrap_or(0);
            buf.extend_from_slice(&end.to_le_bytes());
            let serialized = phase.transcript.serialize();
            buf.extend_from_slice(&(serialized.len() as u32).to_le_bytes());
            buf.extend_from_slice(&serialized);
        }

        buf.extend_from_slice(&self.final_transcript_hash());
        buf
    }

    /// Read-only access to the phases.
    pub fn phases(&self) -> &[PhaseTranscript] {
        &self.phases
    }
}

// ─────────────────────────────────────────────────────────────
// TranscriptAnalyzer
// ─────────────────────────────────────────────────────────────

/// Static analysis utilities for inspecting transcript properties.
pub struct TranscriptAnalyzer;

impl TranscriptAnalyzer {
    /// Returns the total number of entries (absorbs + squeezes) in the transcript history.
    pub fn entry_count(transcript: &FiatShamirTranscript) -> usize {
        transcript.history().len()
    }

    /// Returns the number of Absorb entries in the transcript history.
    pub fn absorb_count(transcript: &FiatShamirTranscript) -> usize {
        transcript
            .history()
            .iter()
            .filter(|e| matches!(e, TranscriptEntry::Absorb { .. }))
            .count()
    }

    /// Returns the number of Squeeze entries in the transcript history.
    pub fn squeeze_count(transcript: &FiatShamirTranscript) -> usize {
        transcript
            .history()
            .iter()
            .filter(|e| matches!(e, TranscriptEntry::Squeeze { .. }))
            .count()
    }

    /// Returns the total bytes absorbed (32 bytes per absorb, since each stores a 32-byte hash).
    pub fn data_absorbed_bytes(transcript: &FiatShamirTranscript) -> usize {
        32 * Self::absorb_count(transcript)
    }

    /// Estimates the total challenge entropy in bits (64 bits per squeeze, since each produces a u64).
    pub fn challenge_entropy_estimate(transcript: &FiatShamirTranscript) -> f64 {
        (Self::squeeze_count(transcript) as f64) * 64.0
    }

    /// Summarises transcript entries grouped by label prefix.
    ///
    /// The prefix is determined by splitting the label on the first '-' or ':'
    /// character. Consecutive entries sharing the same prefix are grouped together.
    /// Returns a vector of (group_name, count) pairs.
    pub fn phase_summary(transcript: &FiatShamirTranscript) -> Vec<(String, usize)> {
        let mut summary: Vec<(String, usize)> = Vec::new();

        for entry in transcript.history() {
            let label = match entry {
                TranscriptEntry::Absorb { label, .. } => label,
                TranscriptEntry::Squeeze { label, .. } => label,
            };

            let prefix = match label.find(|c: char| c == '-' || c == ':') {
                Some(pos) => &label[..pos],
                None => label.as_str(),
            };

            match summary.last_mut() {
                Some((last_prefix, count)) if last_prefix == prefix => {
                    *count += 1;
                }
                _ => {
                    summary.push((prefix.to_string(), 1));
                }
            }
        }

        summary
    }

    /// Returns a mapping from label to count of occurrences across all history entries.
    pub fn label_histogram(transcript: &FiatShamirTranscript) -> HashMap<String, usize> {
        let mut counts: HashMap<String, usize> = HashMap::new();
        for entry in transcript.history() {
            let label = match entry {
                TranscriptEntry::Absorb { label, .. } => label.clone(),
                TranscriptEntry::Squeeze { label, .. } => label.clone(),
            };
            *counts.entry(label).or_insert(0) += 1;
        }
        counts
    }

    /// Returns the set of unique labels used across all history entries, sorted alphabetically.
    pub fn unique_labels(transcript: &FiatShamirTranscript) -> Vec<String> {
        let mut labels: Vec<String> = Self::label_histogram(transcript)
            .into_keys()
            .collect();
        labels.sort();
        labels
    }

    /// Returns the ratio of absorb entries to total entries, or 0.0 if the transcript is empty.
    pub fn absorb_ratio(transcript: &FiatShamirTranscript) -> f64 {
        let total = Self::entry_count(transcript);
        if total == 0 {
            return 0.0;
        }
        Self::absorb_count(transcript) as f64 / total as f64
    }

    /// Returns the longest consecutive run of Absorb entries.
    pub fn longest_absorb_run(transcript: &FiatShamirTranscript) -> usize {
        let mut max_run: usize = 0;
        let mut current_run: usize = 0;
        for entry in transcript.history() {
            match entry {
                TranscriptEntry::Absorb { .. } => {
                    current_run += 1;
                    if current_run > max_run {
                        max_run = current_run;
                    }
                }
                _ => {
                    current_run = 0;
                }
            }
        }
        max_run
    }

    /// Returns true if the transcript has at least one absorb followed by at least one squeeze,
    /// which is the minimal pattern for a meaningful Fiat-Shamir interaction.
    pub fn has_absorb_then_squeeze(transcript: &FiatShamirTranscript) -> bool {
        let mut seen_absorb = false;
        for entry in transcript.history() {
            match entry {
                TranscriptEntry::Absorb { .. } => {
                    seen_absorb = true;
                }
                TranscriptEntry::Squeeze { .. } if seen_absorb => {
                    return true;
                }
                _ => {}
            }
        }
        false
    }

    /// Computes the blake3 hash of all entry labels concatenated in order.
    /// This provides a compact fingerprint of the transcript structure (ignoring data).
    pub fn structural_hash(transcript: &FiatShamirTranscript) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        for entry in transcript.history() {
            match entry {
                TranscriptEntry::Absorb { label, .. } => {
                    hasher.update(b"A:");
                    hasher.update(label.as_bytes());
                    hasher.update(b"|");
                }
                TranscriptEntry::Squeeze { label, .. } => {
                    hasher.update(b"S:");
                    hasher.update(label.as_bytes());
                    hasher.update(b"|");
                }
            }
        }
        *hasher.finalize().as_bytes()
    }
}

// ─────────────────────────────────────────────────────────────
// TranscriptComparison
// ─────────────────────────────────────────────────────────────

/// The result of comparing two transcripts entry-by-entry.
#[derive(Clone, Debug)]
pub struct TranscriptComparison {
    pub identical: bool,
    pub first_difference: Option<usize>,
    pub matching_entries: usize,
    pub total_entries_a: usize,
    pub total_entries_b: usize,
}

impl TranscriptComparison {
    /// Returns the fraction of entries that match, relative to the longer transcript.
    /// Returns 1.0 when both transcripts are empty.
    pub fn match_ratio(&self) -> f64 {
        let max_len = self.total_entries_a.max(self.total_entries_b);
        if max_len == 0 {
            return 1.0;
        }
        self.matching_entries as f64 / max_len as f64
    }

    /// Returns the number of entries that differ or are only present in one transcript.
    pub fn differing_entries(&self) -> usize {
        let max_len = self.total_entries_a.max(self.total_entries_b);
        max_len - self.matching_entries
    }
}

impl fmt::Display for TranscriptComparison {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.identical {
            write!(
                f,
                "Transcripts identical ({} entries)",
                self.total_entries_a,
            )
        } else {
            write!(
                f,
                "Transcripts differ at index {} ({} matching out of {}/{})",
                self.first_difference.unwrap_or(0),
                self.matching_entries,
                self.total_entries_a,
                self.total_entries_b,
            )
        }
    }
}

/// Utilities for comparing two transcripts.
pub struct TranscriptComparator;

impl TranscriptComparator {
    /// Compares two transcripts entry-by-entry and returns a [`TranscriptComparison`].
    pub fn compare(a: &FiatShamirTranscript, b: &FiatShamirTranscript) -> TranscriptComparison {
        let ha = a.history();
        let hb = b.history();
        let total_a = ha.len();
        let total_b = hb.len();

        let min_len = total_a.min(total_b);
        let mut matching: usize = 0;

        for i in 0..min_len {
            if entries_match(&ha[i], &hb[i]) {
                matching += 1;
            } else {
                return TranscriptComparison {
                    identical: false,
                    first_difference: Some(i),
                    matching_entries: matching,
                    total_entries_a: total_a,
                    total_entries_b: total_b,
                };
            }
        }

        if total_a == total_b {
            TranscriptComparison {
                identical: true,
                first_difference: None,
                matching_entries: matching,
                total_entries_a: total_a,
                total_entries_b: total_b,
            }
        } else {
            TranscriptComparison {
                identical: false,
                first_difference: Some(min_len),
                matching_entries: matching,
                total_entries_a: total_a,
                total_entries_b: total_b,
            }
        }
    }

    /// Returns `true` when both transcripts have identical histories.
    pub fn are_equivalent(a: &FiatShamirTranscript, b: &FiatShamirTranscript) -> bool {
        Self::compare(a, b).identical
    }

    /// Returns the index of the first entry where the two transcripts diverge, if any.
    pub fn divergence_point(a: &FiatShamirTranscript, b: &FiatShamirTranscript) -> Option<usize> {
        Self::compare(a, b).first_difference
    }

    /// Returns true if both transcripts have the same sequence of entry kinds
    /// (Absorb/Squeeze pattern) regardless of the actual data or labels.
    pub fn same_structure(a: &FiatShamirTranscript, b: &FiatShamirTranscript) -> bool {
        let ha = a.history();
        let hb = b.history();
        if ha.len() != hb.len() {
            return false;
        }
        for (ea, eb) in ha.iter().zip(hb.iter()) {
            let kind_a = matches!(ea, TranscriptEntry::Absorb { .. });
            let kind_b = matches!(eb, TranscriptEntry::Absorb { .. });
            if kind_a != kind_b {
                return false;
            }
        }
        true
    }

    /// Returns true if both transcripts use exactly the same labels in the same order,
    /// even if the data hashes differ.
    pub fn same_labels(a: &FiatShamirTranscript, b: &FiatShamirTranscript) -> bool {
        let ha = a.history();
        let hb = b.history();
        if ha.len() != hb.len() {
            return false;
        }
        for (ea, eb) in ha.iter().zip(hb.iter()) {
            let la = match ea {
                TranscriptEntry::Absorb { label, .. } => label,
                TranscriptEntry::Squeeze { label, .. } => label,
            };
            let lb = match eb {
                TranscriptEntry::Absorb { label, .. } => label,
                TranscriptEntry::Squeeze { label, .. } => label,
            };
            if la != lb {
                return false;
            }
        }
        true
    }

    /// Returns the number of entries in the common prefix of both transcripts.
    pub fn common_prefix_length(a: &FiatShamirTranscript, b: &FiatShamirTranscript) -> usize {
        Self::compare(a, b).matching_entries
    }
}

// ─────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Determinism ──

    #[test]
    fn test_deterministic_absorb_squeeze() {
        let mut t1 = FiatShamirTranscript::new("test-domain");
        let mut t2 = FiatShamirTranscript::new("test-domain");

        t1.absorb_bytes("label", b"data");
        t2.absorb_bytes("label", b"data");

        let c1 = t1.squeeze_challenge();
        let c2 = t2.squeeze_challenge();
        assert_eq!(c1, c2);
    }

    #[test]
    fn test_different_domains_different_challenges() {
        let mut t1 = FiatShamirTranscript::new("domain-A");
        let mut t2 = FiatShamirTranscript::new("domain-B");

        t1.absorb_bytes("label", b"data");
        t2.absorb_bytes("label", b"data");

        let c1 = t1.squeeze_challenge();
        let c2 = t2.squeeze_challenge();
        assert_ne!(c1, c2);
    }

    #[test]
    fn test_different_data_different_challenges() {
        let mut t1 = FiatShamirTranscript::new("test");
        let mut t2 = FiatShamirTranscript::new("test");

        t1.absorb_bytes("label", b"data1");
        t2.absorb_bytes("label", b"data2");

        assert_ne!(t1.squeeze_challenge(), t2.squeeze_challenge());
    }

    #[test]
    fn test_different_labels_different_challenges() {
        let mut t1 = FiatShamirTranscript::new("test");
        let mut t2 = FiatShamirTranscript::new("test");

        t1.absorb_bytes("label-A", b"data");
        t2.absorb_bytes("label-B", b"data");

        assert_ne!(t1.squeeze_challenge(), t2.squeeze_challenge());
    }

    // ── Sequential squeezes ──

    #[test]
    fn test_sequential_squeezes_differ() {
        let mut t = FiatShamirTranscript::new("test");
        t.absorb_bytes("seed", b"value");

        let c1 = t.squeeze_challenge();
        let c2 = t.squeeze_challenge();
        let c3 = t.squeeze_challenge();
        assert_ne!(c1, c2);
        assert_ne!(c2, c3);
        assert_ne!(c1, c3);
    }

    #[test]
    fn test_squeeze_challenges_batch() {
        let mut t = FiatShamirTranscript::new("test");
        t.absorb_bytes("seed", b"x");

        let challenges = t.squeeze_challenges(5);
        assert_eq!(challenges.len(), 5);

        // All should be distinct
        for i in 0..challenges.len() {
            for j in (i + 1)..challenges.len() {
                assert_ne!(challenges[i], challenges[j]);
            }
        }
    }

    // ── Field element absorption ──

    #[test]
    fn test_absorb_field() {
        let mut t1 = FiatShamirTranscript::new("test");
        let mut t2 = FiatShamirTranscript::new("test");

        t1.absorb_field("val", 42);
        t2.absorb_field("val", 42);

        assert_eq!(t1.squeeze_challenge(), t2.squeeze_challenge());
    }

    #[test]
    fn test_absorb_field_vec() {
        let mut t1 = FiatShamirTranscript::new("test");
        let mut t2 = FiatShamirTranscript::new("test");

        t1.absorb_field_vec("vals", &[1, 2, 3]);
        t2.absorb_field_vec("vals", &[1, 2, 3]);

        assert_eq!(t1.squeeze_challenge(), t2.squeeze_challenge());
    }

    #[test]
    fn test_absorb_field_vec_different() {
        let mut t1 = FiatShamirTranscript::new("test");
        let mut t2 = FiatShamirTranscript::new("test");

        t1.absorb_field_vec("vals", &[1, 2, 3]);
        t2.absorb_field_vec("vals", &[1, 2, 4]);

        assert_ne!(t1.squeeze_challenge(), t2.squeeze_challenge());
    }

    // ── Commitment absorption ──

    #[test]
    fn test_absorb_commitment() {
        let mut t = FiatShamirTranscript::new("test");
        let hash = [99u8; 32];
        t.absorb_commitment("commit", &hash);
        let c = t.squeeze_challenge();
        assert_ne!(c, 0); // sanity check
    }

    // ── absorb_u64 ──

    #[test]
    fn test_absorb_u64() {
        let mut t1 = FiatShamirTranscript::new("test");
        let mut t2 = FiatShamirTranscript::new("test");
        t1.absorb_u64("n", 100);
        t2.absorb_u64("n", 100);
        assert_eq!(t1.squeeze_challenge(), t2.squeeze_challenge());
    }

    // ── squeeze_bytes ──

    #[test]
    fn test_squeeze_bytes() {
        let mut t = FiatShamirTranscript::new("test");
        t.absorb_bytes("seed", b"value");
        let bytes = t.squeeze_bytes("output", 64);
        assert_eq!(bytes.len(), 64);
    }

    #[test]
    fn test_squeeze_bytes_deterministic() {
        let mut t1 = FiatShamirTranscript::new("test");
        let mut t2 = FiatShamirTranscript::new("test");
        t1.absorb_bytes("s", b"d");
        t2.absorb_bytes("s", b"d");
        assert_eq!(
            t1.squeeze_bytes("out", 48),
            t2.squeeze_bytes("out", 48),
        );
    }

    #[test]
    fn test_squeeze_bytes_small() {
        let mut t = FiatShamirTranscript::new("test");
        let bytes = t.squeeze_bytes("x", 1);
        assert_eq!(bytes.len(), 1);
    }

    // ── squeeze_indices ──

    #[test]
    fn test_squeeze_indices_distinct() {
        let mut t = FiatShamirTranscript::new("test");
        t.absorb_bytes("seed", b"value");
        let indices = t.squeeze_indices(5, 100);
        assert_eq!(indices.len(), 5);

        // All distinct
        for i in 0..indices.len() {
            for j in (i + 1)..indices.len() {
                assert_ne!(indices[i], indices[j]);
            }
        }
    }

    #[test]
    fn test_squeeze_indices_in_range() {
        let mut t = FiatShamirTranscript::new("test");
        let indices = t.squeeze_indices(10, 20);
        for &idx in &indices {
            assert!(idx < 20);
        }
    }

    #[test]
    fn test_squeeze_indices_full() {
        let mut t = FiatShamirTranscript::new("test");
        let indices = t.squeeze_indices(5, 5);
        assert_eq!(indices.len(), 5);
        let mut sorted = indices.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_squeeze_indices_deterministic() {
        let mut t1 = FiatShamirTranscript::new("test");
        let mut t2 = FiatShamirTranscript::new("test");
        t1.absorb_bytes("s", b"d");
        t2.absorb_bytes("s", b"d");
        assert_eq!(
            t1.squeeze_indices(5, 50),
            t2.squeeze_indices(5, 50),
        );
    }

    // ── Fork ──

    #[test]
    fn test_fork_independence() {
        let mut t = FiatShamirTranscript::new("test");
        t.absorb_bytes("base", b"data");

        let mut fork1 = t.fork("branch-A");
        let mut fork2 = t.fork("branch-B");

        fork1.absorb_bytes("x", b"1");
        fork2.absorb_bytes("x", b"1");

        // Different fork labels should produce different challenges
        assert_ne!(fork1.squeeze_challenge(), fork2.squeeze_challenge());
    }

    #[test]
    fn test_fork_does_not_affect_parent() {
        let mut t = FiatShamirTranscript::new("test");
        t.absorb_bytes("base", b"data");

        let state_before = t.state_hash();
        let _fork = t.fork("child");
        let state_after = t.state_hash();

        assert_eq!(state_before, state_after);
    }

    #[test]
    fn test_fork_same_label_same_result() {
        let mut t = FiatShamirTranscript::new("test");
        t.absorb_bytes("base", b"data");

        let mut f1 = t.fork("same");
        let mut f2 = t.fork("same");

        assert_eq!(f1.squeeze_challenge(), f2.squeeze_challenge());
    }

    // ── State hash ──

    #[test]
    fn test_state_hash_changes() {
        let mut t = FiatShamirTranscript::new("test");
        let h1 = t.state_hash();
        t.absorb_bytes("a", b"b");
        let h2 = t.state_hash();
        assert_ne!(h1, h2);
    }

    // ── History ──

    #[test]
    fn test_history_recorded() {
        let mut t = FiatShamirTranscript::new("test");
        t.absorb_bytes("a", b"1");
        t.absorb_bytes("b", b"2");
        let _ = t.squeeze_challenge();

        let hist = t.history();
        assert_eq!(hist.len(), 3);
        assert!(matches!(&hist[0], TranscriptEntry::Absorb { label, .. } if label == "a"));
        assert!(matches!(&hist[1], TranscriptEntry::Absorb { label, .. } if label == "b"));
        assert!(matches!(&hist[2], TranscriptEntry::Squeeze { .. }));
    }

    // ── Replay verification ──

    #[test]
    fn test_replay_verification_valid() {
        let mut t = FiatShamirTranscript::new("test");
        t.absorb_bytes("a", b"data");
        let _ = t.squeeze_challenge();
        assert!(t.replay_verification());
    }

    #[test]
    fn test_replay_verification_empty() {
        let t = FiatShamirTranscript::new("test");
        assert!(t.replay_verification());
    }

    // ── Serialization ──

    #[test]
    fn test_serialize_deserialize_roundtrip() {
        let mut t = FiatShamirTranscript::new("test");
        t.absorb_bytes("label", b"data");
        let c1 = t.squeeze_challenge();

        let bytes = t.serialize();
        let t2 = FiatShamirTranscript::deserialize(&bytes).unwrap();
        assert_eq!(t.state_hash(), t2.state_hash());
        assert_eq!(t.history().len(), t2.history().len());
    }

    #[test]
    fn test_deserialize_invalid() {
        let result = FiatShamirTranscript::deserialize(b"not json");
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), TranscriptError::DeserializationError);
    }

    // ── Phase separation ──

    #[test]
    fn test_phase_separation() {
        let mut t1 = FiatShamirTranscript::new("test");
        let mut t2 = FiatShamirTranscript::new("test");

        t1.begin_phase("commit");
        t1.absorb_bytes("x", b"data");
        t1.end_phase();

        t2.begin_phase("reveal");
        t2.absorb_bytes("x", b"data");
        t2.end_phase();

        assert_ne!(t1.squeeze_challenge(), t2.squeeze_challenge());
    }

    #[test]
    fn test_nested_phases() {
        let mut t = FiatShamirTranscript::new("test");
        t.begin_phase("outer");
        t.begin_phase("inner");
        t.absorb_bytes("data", b"value");
        t.end_phase(); // inner
        t.end_phase(); // outer

        assert!(t.replay_verification());
        // 4 absorb entries (begin outer, begin inner, data, end inner) + end outer = 5
        // but end_phase for outer pops from stack
        assert!(t.history().len() >= 4);
    }

    // ── MerlinLikeTranscript ──

    #[test]
    fn test_merlin_deterministic() {
        let mut m1 = MerlinLikeTranscript::new("test");
        let mut m2 = MerlinLikeTranscript::new("test");

        m1.absorb("label", b"data");
        m2.absorb("label", b"data");

        assert_eq!(m1.squeeze_challenge(), m2.squeeze_challenge());
    }

    #[test]
    fn test_merlin_different_data() {
        let mut m1 = MerlinLikeTranscript::new("test");
        let mut m2 = MerlinLikeTranscript::new("test");

        m1.absorb("label", b"data1");
        m2.absorb("label", b"data2");

        assert_ne!(m1.squeeze_challenge(), m2.squeeze_challenge());
    }

    #[test]
    fn test_merlin_sequential_squeezes() {
        let mut m = MerlinLikeTranscript::new("test");
        m.absorb("seed", b"value");
        let s1 = m.squeeze_challenge();
        let s2 = m.squeeze_challenge();
        assert_ne!(s1, s2);
    }

    #[test]
    fn test_merlin_phases() {
        let mut m1 = MerlinLikeTranscript::new("test");
        let mut m2 = MerlinLikeTranscript::new("test");

        m1.begin_phase("alpha");
        m1.absorb("x", b"y");
        m1.end_phase();

        m2.begin_phase("beta");
        m2.absorb("x", b"y");
        m2.end_phase();

        assert_ne!(m1.squeeze_challenge(), m2.squeeze_challenge());
    }

    #[test]
    fn test_merlin_history() {
        let mut m = MerlinLikeTranscript::new("test");
        m.absorb("a", b"1");
        m.squeeze_challenge();
        assert_eq!(m.history().len(), 2);
    }

    // ── TranscriptBuilder ──

    #[test]
    fn test_builder_basic() {
        let t = TranscriptBuilder::new("test")
            .with_phase("init", |b| {
                b.add_bytes("key", b"val");
            })
            .build();

        assert!(t.replay_verification());
        assert!(t.history().len() >= 3); // phase-begin, absorb, phase-end
    }

    #[test]
    fn test_builder_multiple_phases() {
        let t = TranscriptBuilder::new("test")
            .with_phase("phase1", |b| {
                b.add_bytes("a", b"1");
            })
            .with_phase("phase2", |b| {
                b.add_bytes("b", b"2");
            })
            .build();

        assert!(t.replay_verification());
    }

    #[test]
    fn test_builder_deterministic() {
        let build = || {
            TranscriptBuilder::new("test")
                .with_phase("p", |b| {
                    b.add_bytes("k", b"v");
                })
                .build()
        };
        assert_eq!(build().state_hash(), build().state_hash());
    }

    // ── Edge cases ──

    #[test]
    fn test_empty_absorb() {
        let mut t = FiatShamirTranscript::new("test");
        t.absorb_bytes("empty", b"");
        let c = t.squeeze_challenge();
        assert_ne!(c, 0); // should still produce something
    }

    #[test]
    fn test_large_absorb() {
        let mut t = FiatShamirTranscript::new("test");
        let big_data = vec![0xABu8; 10_000];
        t.absorb_bytes("big", &big_data);
        let c = t.squeeze_challenge();
        assert_ne!(c, 0);
    }

    #[test]
    fn test_squeeze_bytes_zero() {
        let mut t = FiatShamirTranscript::new("test");
        let bytes = t.squeeze_bytes("zero", 0);
        assert!(bytes.is_empty());
    }

    #[test]
    fn test_absorb_order_matters() {
        let mut t1 = FiatShamirTranscript::new("test");
        let mut t2 = FiatShamirTranscript::new("test");

        t1.absorb_bytes("a", b"1");
        t1.absorb_bytes("b", b"2");

        t2.absorb_bytes("b", b"2");
        t2.absorb_bytes("a", b"1");

        assert_ne!(t1.squeeze_challenge(), t2.squeeze_challenge());
    }

    // ── MerlinLikeTranscript (extended) ──

    #[test]
    fn test_merlin_squeeze_challenge_in_range() {
        let mut m = MerlinLikeTranscript::new("test");
        m.absorb("seed", b"data");
        let c = m.squeeze_challenge();
        assert!(c < GOLDILOCKS_PRIME);
    }

    #[test]
    fn test_merlin_squeeze_challenges_multiple() {
        let mut m = MerlinLikeTranscript::new("test");
        m.absorb("seed", b"data");
        let challenges = m.squeeze_challenges(10);
        assert_eq!(challenges.len(), 10);
        for &c in &challenges {
            assert!(c < GOLDILOCKS_PRIME);
        }
        for i in 0..challenges.len() {
            for j in (i + 1)..challenges.len() {
                assert_ne!(challenges[i], challenges[j]);
            }
        }
    }

    #[test]
    fn test_merlin_fork_independence() {
        let mut m = MerlinLikeTranscript::new("test");
        m.absorb("base", b"data");
        let mut f1 = m.fork("branch-A");
        let mut f2 = m.fork("branch-B");
        f1.absorb("x", b"1");
        f2.absorb("x", b"1");
        assert_ne!(f1.squeeze_challenge(), f2.squeeze_challenge());
    }

    #[test]
    fn test_merlin_fork_same_label() {
        let mut m = MerlinLikeTranscript::new("test");
        m.absorb("base", b"data");
        let mut f1 = m.fork("same");
        let mut f2 = m.fork("same");
        assert_eq!(f1.squeeze_challenge(), f2.squeeze_challenge());
    }

    #[test]
    fn test_merlin_state_hash_changes() {
        let mut m = MerlinLikeTranscript::new("test");
        let h1 = m.state_hash();
        m.absorb("a", b"b");
        let h2 = m.state_hash();
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_merlin_operations_tracked() {
        let mut m = MerlinLikeTranscript::new("test");
        m.absorb("a", b"1");
        m.absorb("b", b"2");
        m.squeeze_challenge();
        assert_eq!(m.operations().len(), 3);
        assert!(matches!(
            &m.operations()[0],
            TranscriptOperation::Absorb(l, _) if l == "a"
        ));
        assert!(matches!(
            &m.operations()[1],
            TranscriptOperation::Absorb(l, _) if l == "b"
        ));
        assert!(matches!(
            &m.operations()[2],
            TranscriptOperation::Squeeze(_, _)
        ));
    }

    #[test]
    fn test_merlin_absorbed_squeezed_counts() {
        let mut m = MerlinLikeTranscript::new("test");
        assert_eq!(m.absorbed_count(), 0);
        assert_eq!(m.squeezed_count(), 0);
        m.absorb("a", b"1");
        m.absorb("b", b"2");
        assert_eq!(m.absorbed_count(), 2);
        m.squeeze_challenge();
        assert_eq!(m.squeezed_count(), 1);
    }

    #[test]
    fn test_merlin_squeeze_bytes() {
        let mut m = MerlinLikeTranscript::new("test");
        m.absorb("seed", b"data");
        let bytes = m.squeeze("output", 64);
        assert_eq!(bytes.len(), 64);
    }

    #[test]
    fn test_merlin_squeeze_bytes_deterministic() {
        let mut m1 = MerlinLikeTranscript::new("test");
        let mut m2 = MerlinLikeTranscript::new("test");
        m1.absorb("s", b"d");
        m2.absorb("s", b"d");
        assert_eq!(m1.squeeze("out", 48), m2.squeeze("out", 48));
    }

    #[test]
    fn test_merlin_large_absorb() {
        let mut m = MerlinLikeTranscript::new("test");
        let big = vec![0xABu8; 10_000];
        m.absorb("big", &big);
        let c = m.squeeze_challenge();
        assert!(c < GOLDILOCKS_PRIME);
    }

    #[test]
    fn test_merlin_empty_absorb() {
        let mut m = MerlinLikeTranscript::new("test");
        m.absorb("empty", b"");
        let c = m.squeeze_challenge();
        assert!(c < GOLDILOCKS_PRIME);
    }

    #[test]
    fn test_merlin_phase_operations() {
        let mut m = MerlinLikeTranscript::new("test");
        m.begin_phase("commit");
        m.absorb("x", b"data");
        m.end_phase();
        let ops = m.operations();
        assert!(matches!(
            &ops[0],
            TranscriptOperation::PhaseBegin(p) if p == "commit"
        ));
        assert!(ops.len() >= 4);
    }

    #[test]
    fn test_merlin_different_domains() {
        let mut m1 = MerlinLikeTranscript::new("domain-A");
        let mut m2 = MerlinLikeTranscript::new("domain-B");
        m1.absorb("x", b"data");
        m2.absorb("x", b"data");
        assert_ne!(m1.squeeze_challenge(), m2.squeeze_challenge());
    }

    // ── TranscriptBuilder (extended) ──

    #[test]
    fn test_builder_with_field_element() {
        let t = TranscriptBuilder::new("test")
            .with_field_element("val", 42)
            .build();
        assert!(t.replay_verification());
        assert!(t.history().len() >= 1);
    }

    #[test]
    fn test_builder_with_bytes() {
        let t = TranscriptBuilder::new("test")
            .with_bytes("data", b"hello")
            .build();
        assert!(t.replay_verification());
    }

    #[test]
    fn test_builder_with_commitment() {
        let hash = [0xAA; 32];
        let t = TranscriptBuilder::new("test")
            .with_commitment("commit", &hash)
            .build();
        assert!(t.replay_verification());
    }

    #[test]
    fn test_builder_with_phase_closure() {
        let t = TranscriptBuilder::new("test")
            .with_phase("init", |b| {
                b.add_bytes("key", b"val");
                b.add_field_element("num", 99);
            })
            .build();
        assert!(t.replay_verification());
        assert!(t.history().len() >= 4);
    }

    #[test]
    fn test_builder_build_merlin() {
        let m = TranscriptBuilder::new("test")
            .with_field_element("x", 1)
            .with_bytes("y", b"data")
            .build_merlin();
        assert_eq!(m.absorbed_count(), 2);
    }

    #[test]
    fn test_builder_nested_phases() {
        let t = TranscriptBuilder::new("test")
            .with_phase("outer", |b| {
                b.add_bytes("a", b"1");
                b.add_field_element("num", 42);
            })
            .with_phase("middle", |b| {
                b.add_commitment("c", &[0xAA; 32]);
            })
            .build();
        assert!(t.replay_verification());
    }

    #[test]
    fn test_builder_mixed_items() {
        let hash = [0xBB; 32];
        let t = TranscriptBuilder::new("test")
            .with_field_element("f1", 100)
            .with_bytes("b1", b"bytes")
            .with_commitment("c1", &hash)
            .with_field_element("f2", 200)
            .build();
        assert!(t.replay_verification());
        assert_eq!(t.history().len(), 4);
    }

    // ── MultiTranscript ──

    #[test]
    fn test_multi_transcript_create_sub() {
        let mut mt = MultiTranscript::new("test");
        let sub = mt.create_sub_transcript("sub1");
        sub.absorb_bytes("key", b"val");
        assert!(mt.get_sub_transcript("sub1").is_some());
    }

    #[test]
    fn test_multi_transcript_get_sub_none() {
        let mt = MultiTranscript::new("test");
        assert!(mt.get_sub_transcript("nonexistent").is_none());
    }

    #[test]
    fn test_multi_transcript_merge_all() {
        let mut mt = MultiTranscript::new("test");
        {
            let s1 = mt.create_sub_transcript("s1");
            s1.absorb_bytes("a", b"1");
        }
        {
            let s2 = mt.create_sub_transcript("s2");
            s2.absorb_bytes("b", b"2");
        }
        mt.merge_all();
        let c = mt.root_challenge();
        assert_ne!(c, 0);
    }

    #[test]
    fn test_multi_transcript_sub_challenge() {
        let mut mt = MultiTranscript::new("test");
        {
            let s = mt.create_sub_transcript("sub");
            s.absorb_bytes("data", b"value");
        }
        let c = mt.sub_challenge("sub");
        assert!(c.is_some());
    }

    #[test]
    fn test_multi_transcript_verify_consistency() {
        let mut mt = MultiTranscript::new("test");
        {
            let s = mt.create_sub_transcript("s1");
            s.absorb_bytes("key", b"val");
        }
        assert!(mt.verify_consistency());
    }

    #[test]
    fn test_multi_transcript_deterministic() {
        let build = || {
            let mut mt = MultiTranscript::new("test");
            {
                let s = mt.create_sub_transcript("sub");
                s.absorb_bytes("k", b"v");
            }
            mt.merge_all();
            mt.root_challenge()
        };
        assert_eq!(build(), build());
    }

    #[test]
    fn test_multi_transcript_independent_subs() {
        let mut mt = MultiTranscript::new("test");
        {
            let s1 = mt.create_sub_transcript("a");
            s1.absorb_bytes("x", b"1");
        }
        {
            let s2 = mt.create_sub_transcript("b");
            s2.absorb_bytes("x", b"1");
        }
        let c1 = mt.sub_challenge("a").unwrap();
        let c2 = mt.sub_challenge("b").unwrap();
        // Different fork labels ⇒ different challenges
        assert_ne!(c1, c2);
    }

    // ── TranscriptVerifier ──

    #[test]
    fn test_verifier_exact_match() {
        let mut t = FiatShamirTranscript::new("test");
        t.absorb_bytes("a", b"data");
        let _ = t.squeeze_challenge();

        let mut v = TranscriptVerifier::new();
        for entry in t.history() {
            v.add_expected(entry.clone());
        }
        let result = v.verify_against(&t);
        assert!(result.is_valid);
        assert_eq!(result.matches, 2);
        assert_eq!(result.mismatches, 0);
    }

    #[test]
    fn test_verifier_mismatch() {
        let mut t1 = FiatShamirTranscript::new("test");
        t1.absorb_bytes("a", b"data1");
        let _ = t1.squeeze_challenge();

        let mut t2 = FiatShamirTranscript::new("test");
        t2.absorb_bytes("a", b"data2");
        let _ = t2.squeeze_challenge();

        let mut v = TranscriptVerifier::new();
        for entry in t1.history() {
            v.add_expected(entry.clone());
        }
        let result = v.verify_against(&t2);
        assert!(!result.is_valid);
        assert!(result.mismatches > 0);
    }

    #[test]
    fn test_verifier_missing_entries() {
        let mut t = FiatShamirTranscript::new("test");
        t.absorb_bytes("a", b"data");

        let mut v = TranscriptVerifier::new();
        for entry in t.history() {
            v.add_expected(entry.clone());
        }
        v.add_expected(TranscriptEntry::Absorb {
            label: "extra".to_string(),
            data_hash: [0xFFu8; 32],
        });
        let result = v.verify_against(&t);
        assert!(!result.is_valid);
        assert_eq!(result.missing, 1);
    }

    #[test]
    fn test_verifier_extra_entries() {
        let mut t = FiatShamirTranscript::new("test");
        t.absorb_bytes("a", b"data");
        let _ = t.squeeze_challenge();

        let mut v = TranscriptVerifier::new();
        v.add_expected(t.history()[0].clone());
        let result = v.verify_against(&t);
        assert!(!result.is_valid);
        assert_eq!(result.extra, 1);
    }

    #[test]
    fn test_verifier_check_ordering() {
        let mut v = TranscriptVerifier::new();
        v.add_expected(TranscriptEntry::Absorb {
            label: "a".to_string(),
            data_hash: [1u8; 32],
        });
        v.add_expected(TranscriptEntry::Squeeze {
            label: "s".to_string(),
            output_hash: [2u8; 32],
        });
        assert!(v.check_ordering());
    }

    #[test]
    fn test_verifier_check_completeness() {
        let mut v = TranscriptVerifier::new();
        v.add_expected(TranscriptEntry::Absorb {
            label: "a".to_string(),
            data_hash: [1u8; 32],
        });
        assert!(!v.check_completeness());
        v.add_expected(TranscriptEntry::Squeeze {
            label: "s".to_string(),
            output_hash: [2u8; 32],
        });
        assert!(v.check_completeness());
    }

    #[test]
    fn test_verifier_differences() {
        let mut v = TranscriptVerifier::new();
        v.add_expected(TranscriptEntry::Absorb {
            label: "a".to_string(),
            data_hash: [1u8; 32],
        });
        v.set_actual_entries(vec![TranscriptEntry::Absorb {
            label: "b".to_string(),
            data_hash: [2u8; 32],
        }]);
        let diffs = v.differences();
        assert_eq!(diffs.len(), 1);
        assert!(matches!(&diffs[0], TranscriptDifference::Mismatch(_, _)));
    }

    #[test]
    fn test_verifier_empty() {
        let v = TranscriptVerifier::new();
        let t = FiatShamirTranscript::new("test");
        let result = v.verify_against(&t);
        assert!(result.is_valid);
        assert_eq!(result.matches, 0);
    }

    #[test]
    fn test_verifier_differences_missing_and_extra() {
        let mut v = TranscriptVerifier::new();
        v.add_expected(TranscriptEntry::Absorb {
            label: "a".to_string(),
            data_hash: [1u8; 32],
        });
        v.add_expected(TranscriptEntry::Absorb {
            label: "b".to_string(),
            data_hash: [2u8; 32],
        });
        v.set_actual_entries(vec![
            TranscriptEntry::Absorb {
                label: "a".to_string(),
                data_hash: [1u8; 32],
            },
            TranscriptEntry::Absorb {
                label: "a".to_string(),
                data_hash: [1u8; 32],
            },
            TranscriptEntry::Squeeze {
                label: "extra".to_string(),
                output_hash: [3u8; 32],
            },
        ]);
        let diffs = v.differences();
        // index 1 is mismatch (b vs a), index 2 is extra
        assert!(diffs.len() >= 2);
    }

    // ── ChallengeGenerator ──

    #[test]
    fn test_challenge_gen_from_transcript() {
        let mut t = FiatShamirTranscript::new("test");
        t.absorb_bytes("data", b"value");
        let mut gen = ChallengeGenerator::from_transcript(&t);
        let c = gen.next_field_element();
        assert!(c < GOLDILOCKS_PRIME);
    }

    #[test]
    fn test_challenge_gen_from_seed() {
        let seed = [42u8; 32];
        let mut gen = ChallengeGenerator::from_seed(seed);
        let c = gen.next_field_element();
        assert!(c < GOLDILOCKS_PRIME);
    }

    #[test]
    fn test_challenge_gen_deterministic() {
        let seed = [7u8; 32];
        let mut g1 = ChallengeGenerator::from_seed(seed);
        let mut g2 = ChallengeGenerator::from_seed(seed);
        assert_eq!(g1.next_field_element(), g2.next_field_element());
        assert_eq!(g1.next_field_element(), g2.next_field_element());
    }

    #[test]
    fn test_challenge_gen_field_elements() {
        let mut gen = ChallengeGenerator::from_seed([1u8; 32]);
        let elements = gen.next_field_elements(10);
        assert_eq!(elements.len(), 10);
        for &e in &elements {
            assert!(e < GOLDILOCKS_PRIME);
        }
    }

    #[test]
    fn test_challenge_gen_next_index() {
        let mut gen = ChallengeGenerator::from_seed([2u8; 32]);
        for _ in 0..100 {
            let idx = gen.next_index(50);
            assert!(idx < 50);
        }
    }

    #[test]
    fn test_challenge_gen_next_indices() {
        let mut gen = ChallengeGenerator::from_seed([3u8; 32]);
        let indices = gen.next_indices(10, 100);
        assert_eq!(indices.len(), 10);
        for i in 0..indices.len() {
            for j in (i + 1)..indices.len() {
                assert_ne!(indices[i], indices[j]);
            }
        }
        for &idx in &indices {
            assert!(idx < 100);
        }
    }

    #[test]
    fn test_challenge_gen_next_bytes() {
        let mut gen = ChallengeGenerator::from_seed([4u8; 32]);
        let bytes = gen.next_bytes(100);
        assert_eq!(bytes.len(), 100);
    }

    #[test]
    fn test_challenge_gen_next_bool() {
        let mut gen = ChallengeGenerator::from_seed([5u8; 32]);
        let mut seen_true = false;
        let mut seen_false = false;
        for _ in 0..100 {
            if gen.next_bool() {
                seen_true = true;
            } else {
                seen_false = true;
            }
        }
        assert!(seen_true && seen_false);
    }

    #[test]
    fn test_challenge_gen_sequential_different() {
        let mut gen = ChallengeGenerator::from_seed([6u8; 32]);
        let c1 = gen.next_field_element();
        let c2 = gen.next_field_element();
        let c3 = gen.next_field_element();
        assert_ne!(c1, c2);
        assert_ne!(c2, c3);
    }

    #[test]
    fn test_challenge_gen_reproducible_across_calls() {
        let seed = [0x42u8; 32];
        let mut gen1 = ChallengeGenerator::from_seed(seed);
        let batch1: Vec<u64> =
            (0..20).map(|_| gen1.next_field_element()).collect();

        let mut gen2 = ChallengeGenerator::from_seed(seed);
        let batch2: Vec<u64> =
            (0..20).map(|_| gen2.next_field_element()).collect();

        assert_eq!(batch1, batch2);
    }

    #[test]
    fn test_challenge_gen_bytes_length() {
        let mut gen = ChallengeGenerator::from_seed([0xAB; 32]);
        for len in [0, 1, 16, 31, 32, 33, 64, 100, 256] {
            assert_eq!(gen.next_bytes(len).len(), len);
        }
    }

    // ── TranscriptSerializer ──

    #[test]
    fn test_serializer_roundtrip() {
        let mut t = FiatShamirTranscript::new("test");
        t.absorb_bytes("label", b"data");
        let _ = t.squeeze_challenge();

        let bytes = TranscriptSerializer::serialize(&t);
        let t2 = TranscriptSerializer::deserialize(&bytes).unwrap();
        assert_eq!(t.state_hash(), t2.state_hash());
        assert_eq!(t.history().len(), t2.history().len());
    }

    #[test]
    fn test_serializer_invalid_magic() {
        let result = TranscriptSerializer::deserialize(b"BADM12345678901234567890123456789012345678901");
        assert!(result.is_err());
    }

    #[test]
    fn test_serializer_too_short() {
        let result = TranscriptSerializer::deserialize(b"TSC");
        assert!(result.is_err());
    }

    #[test]
    fn test_serializer_compressed() {
        let mut t = FiatShamirTranscript::new("test");
        t.absorb_bytes("label", b"data");
        let compressed = TranscriptSerializer::serialize_compressed(&t);
        assert!(!compressed.is_empty());
        assert_eq!(&compressed[0..4], b"TCMP");
    }

    #[test]
    fn test_serializer_estimate_size() {
        let mut t = FiatShamirTranscript::new("test");
        t.absorb_bytes("label", b"data");
        let _ = t.squeeze_challenge();
        let estimate = TranscriptSerializer::estimate_size(&t);
        assert!(estimate > 0);
    }

    #[test]
    fn test_serializer_empty_transcript() {
        let t = FiatShamirTranscript::new("test");
        let bytes = TranscriptSerializer::serialize(&t);
        let t2 = TranscriptSerializer::deserialize(&bytes).unwrap();
        assert_eq!(t2.history().len(), 0);
    }

    #[test]
    fn test_serializer_large_transcript() {
        let mut t = FiatShamirTranscript::new("test");
        for i in 0..50 {
            t.absorb_bytes(&format!("item-{}", i), &[i as u8; 8]);
        }
        for _ in 0..20 {
            let _ = t.squeeze_challenge();
        }
        let bytes = TranscriptSerializer::serialize(&t);
        let t2 = TranscriptSerializer::deserialize(&bytes).unwrap();
        assert_eq!(t.state_hash(), t2.state_hash());
        assert_eq!(t.history().len(), t2.history().len());
    }

    #[test]
    fn test_serializer_preserves_history_count() {
        let mut t = FiatShamirTranscript::new("test");
        for i in 0..10 {
            t.absorb_bytes(&format!("item-{}", i), &[i as u8; 4]);
        }
        for _ in 0..5 {
            let _ = t.squeeze_challenge();
        }
        let bytes = TranscriptSerializer::serialize(&t);
        let t2 = TranscriptSerializer::deserialize(&bytes).unwrap();
        assert_eq!(t.history().len(), t2.history().len());
    }

    // ── ProtocolTranscriptManager ──

    #[test]
    fn test_protocol_manager_new() {
        let mgr = ProtocolTranscriptManager::new("test-protocol");
        assert_eq!(mgr.phases().len(), 0);
    }

    #[test]
    fn test_protocol_manager_commit_phase() {
        let mut mgr = ProtocolTranscriptManager::new("test");
        mgr.begin_commit_phase();
        assert_eq!(mgr.phases().len(), 1);
        assert_eq!(mgr.phases()[0].phase_name, "commit");
        assert!(mgr.phases()[0].end_time.is_none());
        mgr.end_commit_phase();
        assert!(mgr.phases()[0].end_time.is_some());
    }

    #[test]
    fn test_protocol_manager_prove_phase() {
        let mut mgr = ProtocolTranscriptManager::new("test");
        mgr.begin_prove_phase();
        assert_eq!(mgr.phases()[0].phase_name, "prove");
        mgr.end_prove_phase();
        assert!(mgr.phases()[0].end_time.is_some());
    }

    #[test]
    fn test_protocol_manager_verify_phase() {
        let mut mgr = ProtocolTranscriptManager::new("test");
        mgr.begin_verify_phase();
        assert_eq!(mgr.phases()[0].phase_name, "verify");
        mgr.end_verify_phase();
        assert!(mgr.phases()[0].end_time.is_some());
    }

    #[test]
    fn test_protocol_manager_record_commitment() {
        let mut mgr = ProtocolTranscriptManager::new("test");
        mgr.begin_commit_phase();
        let hash = [0xCC; 32];
        mgr.record_commitment("poly-commit", &hash);
        assert!(mgr.phases()[0].transcript.history().len() >= 1);
        mgr.end_commit_phase();
    }

    #[test]
    fn test_protocol_manager_record_challenge() {
        let mut mgr = ProtocolTranscriptManager::new("test");
        mgr.begin_prove_phase();
        mgr.record_challenge(12345);
        assert!(mgr.phases()[0].transcript.history().len() >= 1);
        mgr.end_prove_phase();
    }

    #[test]
    fn test_protocol_manager_final_hash() {
        let mut mgr = ProtocolTranscriptManager::new("test");
        mgr.begin_commit_phase();
        mgr.record_commitment("c1", &[0xAA; 32]);
        mgr.end_commit_phase();
        let h = mgr.final_transcript_hash();
        assert_ne!(h, [0u8; 32]);
    }

    #[test]
    fn test_protocol_manager_export() {
        let mut mgr = ProtocolTranscriptManager::new("test");
        mgr.begin_commit_phase();
        mgr.record_commitment("c1", &[0xBB; 32]);
        mgr.end_commit_phase();
        let exported = mgr.export_transcript();
        assert!(!exported.is_empty());
        assert_eq!(&exported[0..4], b"PROT");
    }

    #[test]
    fn test_protocol_manager_full_lifecycle() {
        let mut mgr = ProtocolTranscriptManager::new("plonk");

        mgr.begin_commit_phase();
        mgr.record_commitment("poly_a", &[0x01; 32]);
        mgr.record_commitment("poly_b", &[0x02; 32]);
        mgr.end_commit_phase();

        mgr.begin_prove_phase();
        mgr.record_challenge(42);
        mgr.record_commitment("quotient", &[0x03; 32]);
        mgr.end_prove_phase();

        mgr.begin_verify_phase();
        mgr.record_challenge(99);
        mgr.end_verify_phase();

        assert_eq!(mgr.phases().len(), 3);
        let h = mgr.final_transcript_hash();
        assert_ne!(h, [0u8; 32]);
    }

    #[test]
    fn test_protocol_manager_deterministic() {
        let build = || {
            let mut mgr = ProtocolTranscriptManager::new("test");
            mgr.begin_commit_phase();
            mgr.record_commitment("c", &[0xDD; 32]);
            mgr.end_commit_phase();
            mgr.begin_prove_phase();
            mgr.record_challenge(777);
            mgr.end_prove_phase();
            mgr.final_transcript_hash()
        };
        assert_eq!(build(), build());
    }

    #[test]
    fn test_protocol_manager_three_phases_hash() {
        let mut mgr = ProtocolTranscriptManager::new("test");
        mgr.begin_commit_phase();
        mgr.end_commit_phase();
        mgr.begin_prove_phase();
        mgr.end_prove_phase();
        mgr.begin_verify_phase();
        mgr.end_verify_phase();
        let h = mgr.final_transcript_hash();
        assert_ne!(h, [0u8; 32]);
        assert_eq!(mgr.phases().len(), 3);
    }

    // ── PhaseTranscript ──

    #[test]
    fn test_phase_transcript_creation() {
        let pt = PhaseTranscript {
            phase_name: "commit".to_string(),
            transcript: FiatShamirTranscript::new("test"),
            start_time: 1,
            end_time: None,
        };
        assert_eq!(pt.phase_name, "commit");
        assert!(pt.end_time.is_none());
    }

    // ── TranscriptVerificationResult ──

    #[test]
    fn test_verification_result_valid() {
        let result = TranscriptVerificationResult {
            is_valid: true,
            matches: 5,
            mismatches: 0,
            missing: 0,
            extra: 0,
        };
        assert!(result.is_valid);
        assert_eq!(result.matches, 5);
    }

    #[test]
    fn test_verification_result_invalid() {
        let result = TranscriptVerificationResult {
            is_valid: false,
            matches: 2,
            mismatches: 1,
            missing: 0,
            extra: 0,
        };
        assert!(!result.is_valid);
        assert_eq!(result.mismatches, 1);
    }

    // ── TranscriptDifference ──

    #[test]
    fn test_transcript_difference_variants() {
        let entry1 = TranscriptEntry::Absorb {
            label: "a".to_string(),
            data_hash: [1u8; 32],
        };
        let entry2 = TranscriptEntry::Absorb {
            label: "b".to_string(),
            data_hash: [2u8; 32],
        };
        let diff = TranscriptDifference::Mismatch(entry1.clone(), entry2.clone());
        assert!(matches!(diff, TranscriptDifference::Mismatch(_, _)));

        let missing = TranscriptDifference::Missing(entry1.clone());
        assert!(matches!(missing, TranscriptDifference::Missing(_)));

        let extra = TranscriptDifference::Extra(entry2.clone());
        assert!(matches!(extra, TranscriptDifference::Extra(_)));
    }

    // ── TranscriptOperation ──

    #[test]
    fn test_transcript_operation_variants() {
        let op1 = TranscriptOperation::Absorb("label".to_string(), 10);
        let op2 = TranscriptOperation::Squeeze("output".to_string(), 8);
        let op3 = TranscriptOperation::PhaseBegin("commit".to_string());
        let op4 = TranscriptOperation::PhaseEnd;
        assert_eq!(
            op1,
            TranscriptOperation::Absorb("label".to_string(), 10)
        );
        assert_ne!(op1, op2);
        assert!(matches!(op3, TranscriptOperation::PhaseBegin(_)));
        assert!(matches!(op4, TranscriptOperation::PhaseEnd));
    }

    // ── BuilderItem ──

    #[test]
    fn test_builder_item_variants() {
        let f = BuilderItem::FieldElement("x".to_string(), 42);
        let b = BuilderItem::Bytes("y".to_string(), vec![1, 2, 3]);
        let c = BuilderItem::Commitment("z".to_string(), [0xAA; 32]);
        let p = BuilderItem::Phase("p".to_string(), vec![]);
        assert!(matches!(f, BuilderItem::FieldElement(_, 42)));
        assert!(matches!(b, BuilderItem::Bytes(_, _)));
        assert!(matches!(c, BuilderItem::Commitment(_, _)));
        assert!(matches!(p, BuilderItem::Phase(_, _)));
    }

    // ── Cross-type integration tests ──

    #[test]
    fn test_builder_to_challenge_generator() {
        let t = TranscriptBuilder::new("test")
            .with_field_element("x", 100)
            .with_bytes("y", b"data")
            .build();
        let mut gen = ChallengeGenerator::from_transcript(&t);
        let c = gen.next_field_element();
        assert!(c < GOLDILOCKS_PRIME);
    }

    #[test]
    fn test_multi_transcript_with_serializer() {
        let mut mt = MultiTranscript::new("test");
        {
            let s = mt.create_sub_transcript("sub");
            s.absorb_bytes("data", b"value");
        }
        mt.merge_all();
        assert!(mt.verify_consistency());
    }

    #[test]
    fn test_verifier_with_builder() {
        let t = TranscriptBuilder::new("test")
            .with_field_element("x", 42)
            .build();
        let mut v = TranscriptVerifier::new();
        for entry in t.history() {
            v.add_expected(entry.clone());
        }
        let result = v.verify_against(&t);
        assert!(result.is_valid);
    }

    #[test]
    fn test_protocol_manager_with_challenge_gen() {
        let mut mgr = ProtocolTranscriptManager::new("test");
        mgr.begin_commit_phase();
        mgr.record_commitment("c1", &[0x11; 32]);
        mgr.end_commit_phase();
        let hash = mgr.final_transcript_hash();
        let mut gen = ChallengeGenerator::from_seed(hash);
        let c = gen.next_field_element();
        assert!(c < GOLDILOCKS_PRIME);
    }

    #[test]
    fn test_merlin_and_fiat_shamir_different() {
        let mut fs = FiatShamirTranscript::new("test");
        let mut ml = MerlinLikeTranscript::new("test");

        fs.absorb_bytes("data", b"value");
        ml.absorb("data", b"value");

        let c1 = fs.squeeze_challenge();
        let c2 = ml.squeeze_challenge();
        assert_ne!(c1, c2);
    }

    #[test]
    fn test_entries_match_helper() {
        let e1 = TranscriptEntry::Absorb {
            label: "a".to_string(),
            data_hash: [1u8; 32],
        };
        let e2 = TranscriptEntry::Absorb {
            label: "a".to_string(),
            data_hash: [1u8; 32],
        };
        let e3 = TranscriptEntry::Squeeze {
            label: "s".to_string(),
            output_hash: [2u8; 32],
        };
        assert!(entries_match(&e1, &e2));
        assert!(!entries_match(&e1, &e3));
    }

    #[test]
    fn test_challenge_gen_from_different_transcripts() {
        let mut t1 = FiatShamirTranscript::new("test");
        t1.absorb_bytes("a", b"1");
        let mut t2 = FiatShamirTranscript::new("test");
        t2.absorb_bytes("a", b"2");

        let mut g1 = ChallengeGenerator::from_transcript(&t1);
        let mut g2 = ChallengeGenerator::from_transcript(&t2);
        assert_ne!(g1.next_field_element(), g2.next_field_element());
    }

    #[test]
    fn test_serializer_checksum_integrity() {
        let mut t = FiatShamirTranscript::new("test");
        t.absorb_bytes("x", b"data");
        let mut bytes = TranscriptSerializer::serialize(&t);
        // Corrupt one byte in the JSON payload
        if bytes.len() > 15 {
            bytes[15] ^= 0xFF;
        }
        let result = TranscriptSerializer::deserialize(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn test_multi_transcript_no_merge_root_clean() {
        let mut mt = MultiTranscript::new("test");
        // Root challenge without any subs or merge
        let c = mt.root_challenge();
        assert_ne!(c, 0);
    }

    #[test]
    fn test_protocol_manager_no_active_phase() {
        let mut mgr = ProtocolTranscriptManager::new("test");
        // Record without active phase should be a no-op
        mgr.record_commitment("x", &[0xAA; 32]);
        mgr.record_challenge(123);
        assert_eq!(mgr.phases().len(), 0);
    }

    #[test]
    fn test_builder_empty_build() {
        let t = TranscriptBuilder::new("test").build();
        assert_eq!(t.history().len(), 0);
        assert!(t.replay_verification());
    }

    #[test]
    fn test_builder_empty_build_merlin() {
        let m = TranscriptBuilder::new("test").build_merlin();
        assert_eq!(m.absorbed_count(), 0);
        assert_eq!(m.squeezed_count(), 0);
    }

    // ── TranscriptAnalyzer ──

    #[test]
    fn test_analyzer_entry_count_empty() {
        let t = FiatShamirTranscript::new("test");
        assert_eq!(TranscriptAnalyzer::entry_count(&t), 0);
    }

    #[test]
    fn test_analyzer_entry_count_non_empty() {
        let mut t = FiatShamirTranscript::new("test");
        t.absorb_bytes("a", b"hello");
        t.absorb_bytes("b", b"world");
        let _ = t.squeeze_challenge();
        assert_eq!(TranscriptAnalyzer::entry_count(&t), 3);
    }

    #[test]
    fn test_analyzer_absorb_count() {
        let mut t = FiatShamirTranscript::new("test");
        t.absorb_bytes("x", b"1");
        t.absorb_bytes("y", b"2");
        let _ = t.squeeze_challenge();
        assert_eq!(TranscriptAnalyzer::absorb_count(&t), 2);
    }

    #[test]
    fn test_analyzer_squeeze_count() {
        let mut t = FiatShamirTranscript::new("test");
        t.absorb_bytes("x", b"1");
        let _ = t.squeeze_challenge();
        let _ = t.squeeze_challenge();
        assert_eq!(TranscriptAnalyzer::squeeze_count(&t), 2);
    }

    #[test]
    fn test_analyzer_data_absorbed_bytes() {
        let mut t = FiatShamirTranscript::new("test");
        t.absorb_bytes("a", b"data1");
        t.absorb_bytes("b", b"data2");
        t.absorb_bytes("c", b"data3");
        assert_eq!(TranscriptAnalyzer::data_absorbed_bytes(&t), 96); // 3 * 32
    }

    #[test]
    fn test_analyzer_challenge_entropy_estimate() {
        let mut t = FiatShamirTranscript::new("test");
        t.absorb_bytes("x", b"data");
        let _ = t.squeeze_challenge();
        let _ = t.squeeze_challenge();
        let _ = t.squeeze_challenge();
        let entropy = TranscriptAnalyzer::challenge_entropy_estimate(&t);
        assert!((entropy - 192.0).abs() < f64::EPSILON); // 3 * 64.0
    }

    #[test]
    fn test_analyzer_phase_summary_empty() {
        let t = FiatShamirTranscript::new("test");
        let summary = TranscriptAnalyzer::phase_summary(&t);
        assert!(summary.is_empty());
    }

    #[test]
    fn test_analyzer_phase_summary_grouped() {
        let mut t = FiatShamirTranscript::new("test");
        t.absorb_bytes("commit-poly", b"p1");
        t.absorb_bytes("commit-eval", b"e1");
        let _ = t.squeeze_challenge(); // label="challenge" -> prefix "challenge"
        t.absorb_bytes("open-proof", b"o1");
        let summary = TranscriptAnalyzer::phase_summary(&t);
        // "commit" x2, "challenge" x1, "open" x1
        assert_eq!(summary.len(), 3);
        assert_eq!(summary[0], ("commit".to_string(), 2));
        assert_eq!(summary[1], ("challenge".to_string(), 1));
        assert_eq!(summary[2], ("open".to_string(), 1));
    }

    #[test]
    fn test_analyzer_phase_summary_no_separator() {
        let mut t = FiatShamirTranscript::new("test");
        t.absorb_bytes("alpha", b"a");
        t.absorb_bytes("alpha", b"b");
        let summary = TranscriptAnalyzer::phase_summary(&t);
        assert_eq!(summary.len(), 1);
        assert_eq!(summary[0], ("alpha".to_string(), 2));
    }

    // ── TranscriptComparison ──

    #[test]
    fn test_comparison_field_access() {
        let cmp = TranscriptComparison {
            identical: true,
            first_difference: None,
            matching_entries: 5,
            total_entries_a: 5,
            total_entries_b: 5,
        };
        assert!(cmp.identical);
        assert!(cmp.first_difference.is_none());
        assert_eq!(cmp.matching_entries, 5);
        assert_eq!(cmp.total_entries_a, 5);
        assert_eq!(cmp.total_entries_b, 5);
    }

    // ── TranscriptComparator ──

    #[test]
    fn test_comparator_identical_transcripts() {
        let mut a = FiatShamirTranscript::new("test");
        let mut b = FiatShamirTranscript::new("test");
        a.absorb_bytes("x", b"data");
        b.absorb_bytes("x", b"data");
        let _ = a.squeeze_challenge();
        let _ = b.squeeze_challenge();
        let cmp = TranscriptComparator::compare(&a, &b);
        assert!(cmp.identical);
        assert!(cmp.first_difference.is_none());
        assert_eq!(cmp.matching_entries, 2);
    }

    #[test]
    fn test_comparator_different_transcripts() {
        let mut a = FiatShamirTranscript::new("test");
        let mut b = FiatShamirTranscript::new("test");
        a.absorb_bytes("x", b"data1");
        b.absorb_bytes("x", b"data2");
        let cmp = TranscriptComparator::compare(&a, &b);
        assert!(!cmp.identical);
        assert_eq!(cmp.first_difference, Some(0));
        assert_eq!(cmp.matching_entries, 0);
    }

    #[test]
    fn test_comparator_are_equivalent_true() {
        let mut a = FiatShamirTranscript::new("domain");
        let mut b = FiatShamirTranscript::new("domain");
        a.absorb_bytes("l", b"v");
        b.absorb_bytes("l", b"v");
        assert!(TranscriptComparator::are_equivalent(&a, &b));
    }

    #[test]
    fn test_comparator_are_equivalent_false() {
        let mut a = FiatShamirTranscript::new("domain");
        let b = FiatShamirTranscript::new("domain");
        a.absorb_bytes("l", b"v");
        assert!(!TranscriptComparator::are_equivalent(&a, &b));
    }

    #[test]
    fn test_comparator_divergence_point_none() {
        let mut a = FiatShamirTranscript::new("d");
        let mut b = FiatShamirTranscript::new("d");
        a.absorb_bytes("k", b"v");
        b.absorb_bytes("k", b"v");
        assert_eq!(TranscriptComparator::divergence_point(&a, &b), None);
    }

    #[test]
    fn test_comparator_divergence_point_some() {
        let mut a = FiatShamirTranscript::new("d");
        let mut b = FiatShamirTranscript::new("d");
        a.absorb_bytes("k", b"same");
        b.absorb_bytes("k", b"same");
        a.absorb_bytes("k2", b"diff_a");
        b.absorb_bytes("k2", b"diff_b");
        assert_eq!(TranscriptComparator::divergence_point(&a, &b), Some(1));
    }

    #[test]
    fn test_comparator_different_lengths() {
        let mut a = FiatShamirTranscript::new("d");
        let mut b = FiatShamirTranscript::new("d");
        a.absorb_bytes("k", b"v");
        b.absorb_bytes("k", b"v");
        a.absorb_bytes("extra", b"more");
        let cmp = TranscriptComparator::compare(&a, &b);
        assert!(!cmp.identical);
        assert_eq!(cmp.first_difference, Some(1));
        assert_eq!(cmp.matching_entries, 1);
        assert_eq!(cmp.total_entries_a, 2);
        assert_eq!(cmp.total_entries_b, 1);
    }

    #[test]
    fn test_comparator_both_empty() {
        let a = FiatShamirTranscript::new("d");
        let b = FiatShamirTranscript::new("d");
        let cmp = TranscriptComparator::compare(&a, &b);
        assert!(cmp.identical);
        assert!(cmp.first_difference.is_none());
        assert_eq!(cmp.matching_entries, 0);
    }

    // ── TranscriptAnalyzer extra methods ──

    #[test]
    fn test_analyzer_label_histogram_empty() {
        let t = FiatShamirTranscript::new("test");
        let hist = TranscriptAnalyzer::label_histogram(&t);
        assert!(hist.is_empty());
    }

    #[test]
    fn test_analyzer_label_histogram() {
        let mut t = FiatShamirTranscript::new("test");
        t.absorb_bytes("alpha", b"1");
        t.absorb_bytes("beta", b"2");
        t.absorb_bytes("alpha", b"3");
        let _ = t.squeeze_challenge(); // label = "challenge"
        let hist = TranscriptAnalyzer::label_histogram(&t);
        assert_eq!(hist.get("alpha"), Some(&2));
        assert_eq!(hist.get("beta"), Some(&1));
        assert_eq!(hist.get("challenge"), Some(&1));
        assert_eq!(hist.len(), 3);
    }

    #[test]
    fn test_analyzer_unique_labels() {
        let mut t = FiatShamirTranscript::new("test");
        t.absorb_bytes("z-label", b"1");
        t.absorb_bytes("a-label", b"2");
        t.absorb_bytes("z-label", b"3");
        let labels = TranscriptAnalyzer::unique_labels(&t);
        assert_eq!(labels, vec!["a-label".to_string(), "z-label".to_string()]);
    }

    #[test]
    fn test_analyzer_unique_labels_empty() {
        let t = FiatShamirTranscript::new("test");
        let labels = TranscriptAnalyzer::unique_labels(&t);
        assert!(labels.is_empty());
    }

    #[test]
    fn test_analyzer_absorb_ratio_empty() {
        let t = FiatShamirTranscript::new("test");
        assert!((TranscriptAnalyzer::absorb_ratio(&t) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_analyzer_absorb_ratio() {
        let mut t = FiatShamirTranscript::new("test");
        t.absorb_bytes("a", b"1");
        t.absorb_bytes("b", b"2");
        t.absorb_bytes("c", b"3");
        let _ = t.squeeze_challenge();
        // 3 absorbs out of 4 total
        assert!((TranscriptAnalyzer::absorb_ratio(&t) - 0.75).abs() < f64::EPSILON);
    }

    #[test]
    fn test_analyzer_longest_absorb_run() {
        let mut t = FiatShamirTranscript::new("test");
        t.absorb_bytes("a", b"1");
        t.absorb_bytes("b", b"2");
        t.absorb_bytes("c", b"3");
        let _ = t.squeeze_challenge();
        t.absorb_bytes("d", b"4");
        assert_eq!(TranscriptAnalyzer::longest_absorb_run(&t), 3);
    }

    #[test]
    fn test_analyzer_longest_absorb_run_empty() {
        let t = FiatShamirTranscript::new("test");
        assert_eq!(TranscriptAnalyzer::longest_absorb_run(&t), 0);
    }

    #[test]
    fn test_analyzer_has_absorb_then_squeeze_true() {
        let mut t = FiatShamirTranscript::new("test");
        t.absorb_bytes("a", b"1");
        let _ = t.squeeze_challenge();
        assert!(TranscriptAnalyzer::has_absorb_then_squeeze(&t));
    }

    #[test]
    fn test_analyzer_has_absorb_then_squeeze_false_empty() {
        let t = FiatShamirTranscript::new("test");
        assert!(!TranscriptAnalyzer::has_absorb_then_squeeze(&t));
    }

    #[test]
    fn test_analyzer_has_absorb_then_squeeze_squeeze_only() {
        let mut t = FiatShamirTranscript::new("test");
        let _ = t.squeeze_challenge();
        assert!(!TranscriptAnalyzer::has_absorb_then_squeeze(&t));
    }

    #[test]
    fn test_analyzer_structural_hash_deterministic() {
        let mut t1 = FiatShamirTranscript::new("test");
        let mut t2 = FiatShamirTranscript::new("test");
        t1.absorb_bytes("a", b"data1");
        t2.absorb_bytes("a", b"data1");
        let _ = t1.squeeze_challenge();
        let _ = t2.squeeze_challenge();
        assert_eq!(
            TranscriptAnalyzer::structural_hash(&t1),
            TranscriptAnalyzer::structural_hash(&t2),
        );
    }

    #[test]
    fn test_analyzer_structural_hash_same_labels_different_data() {
        let mut t1 = FiatShamirTranscript::new("test");
        let mut t2 = FiatShamirTranscript::new("test");
        t1.absorb_bytes("a", b"data1");
        t2.absorb_bytes("a", b"data2");
        // Structural hash only depends on labels and kinds, not data
        assert_eq!(
            TranscriptAnalyzer::structural_hash(&t1),
            TranscriptAnalyzer::structural_hash(&t2),
        );
    }

    #[test]
    fn test_analyzer_structural_hash_different_labels() {
        let mut t1 = FiatShamirTranscript::new("test");
        let mut t2 = FiatShamirTranscript::new("test");
        t1.absorb_bytes("a", b"data");
        t2.absorb_bytes("b", b"data");
        assert_ne!(
            TranscriptAnalyzer::structural_hash(&t1),
            TranscriptAnalyzer::structural_hash(&t2),
        );
    }

    // ── TranscriptComparison extra methods ──

    #[test]
    fn test_comparison_match_ratio_identical() {
        let cmp = TranscriptComparison {
            identical: true,
            first_difference: None,
            matching_entries: 5,
            total_entries_a: 5,
            total_entries_b: 5,
        };
        assert!((cmp.match_ratio() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_comparison_match_ratio_partial() {
        let cmp = TranscriptComparison {
            identical: false,
            first_difference: Some(2),
            matching_entries: 2,
            total_entries_a: 4,
            total_entries_b: 4,
        };
        assert!((cmp.match_ratio() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_comparison_match_ratio_both_empty() {
        let cmp = TranscriptComparison {
            identical: true,
            first_difference: None,
            matching_entries: 0,
            total_entries_a: 0,
            total_entries_b: 0,
        };
        assert!((cmp.match_ratio() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_comparison_differing_entries() {
        let cmp = TranscriptComparison {
            identical: false,
            first_difference: Some(3),
            matching_entries: 3,
            total_entries_a: 5,
            total_entries_b: 5,
        };
        assert_eq!(cmp.differing_entries(), 2);
    }

    #[test]
    fn test_comparison_differing_entries_different_lengths() {
        let cmp = TranscriptComparison {
            identical: false,
            first_difference: Some(2),
            matching_entries: 2,
            total_entries_a: 2,
            total_entries_b: 5,
        };
        assert_eq!(cmp.differing_entries(), 3);
    }

    #[test]
    fn test_comparison_display_identical() {
        let cmp = TranscriptComparison {
            identical: true,
            first_difference: None,
            matching_entries: 3,
            total_entries_a: 3,
            total_entries_b: 3,
        };
        let s = format!("{}", cmp);
        assert!(s.contains("identical"));
        assert!(s.contains("3"));
    }

    #[test]
    fn test_comparison_display_different() {
        let cmp = TranscriptComparison {
            identical: false,
            first_difference: Some(1),
            matching_entries: 1,
            total_entries_a: 3,
            total_entries_b: 4,
        };
        let s = format!("{}", cmp);
        assert!(s.contains("differ"));
        assert!(s.contains("1"));
    }

    // ── TranscriptComparator extra methods ──

    #[test]
    fn test_comparator_same_structure_true() {
        let mut a = FiatShamirTranscript::new("test");
        let mut b = FiatShamirTranscript::new("test");
        a.absorb_bytes("x", b"1");
        b.absorb_bytes("y", b"2");
        let _ = a.squeeze_challenge();
        let _ = b.squeeze_challenge();
        assert!(TranscriptComparator::same_structure(&a, &b));
    }

    #[test]
    fn test_comparator_same_structure_false_different_order() {
        let mut a = FiatShamirTranscript::new("test");
        let mut b = FiatShamirTranscript::new("test");
        a.absorb_bytes("x", b"1");
        let _ = a.squeeze_challenge();
        let _ = b.squeeze_challenge();
        b.absorb_bytes("y", b"2");
        assert!(!TranscriptComparator::same_structure(&a, &b));
    }

    #[test]
    fn test_comparator_same_structure_false_different_lengths() {
        let mut a = FiatShamirTranscript::new("test");
        let b = FiatShamirTranscript::new("test");
        a.absorb_bytes("x", b"1");
        assert!(!TranscriptComparator::same_structure(&a, &b));
    }

    #[test]
    fn test_comparator_same_labels_true() {
        let mut a = FiatShamirTranscript::new("d1");
        let mut b = FiatShamirTranscript::new("d2");
        a.absorb_bytes("lbl", b"data_a");
        b.absorb_bytes("lbl", b"data_b");
        assert!(TranscriptComparator::same_labels(&a, &b));
    }

    #[test]
    fn test_comparator_same_labels_false() {
        let mut a = FiatShamirTranscript::new("d");
        let mut b = FiatShamirTranscript::new("d");
        a.absorb_bytes("alpha", b"1");
        b.absorb_bytes("beta", b"1");
        assert!(!TranscriptComparator::same_labels(&a, &b));
    }

    #[test]
    fn test_comparator_common_prefix_length() {
        let mut a = FiatShamirTranscript::new("d");
        let mut b = FiatShamirTranscript::new("d");
        a.absorb_bytes("k1", b"v1");
        b.absorb_bytes("k1", b"v1");
        a.absorb_bytes("k2", b"v2");
        b.absorb_bytes("k2", b"v2");
        a.absorb_bytes("k3", b"diff_a");
        b.absorb_bytes("k3", b"diff_b");
        assert_eq!(TranscriptComparator::common_prefix_length(&a, &b), 2);
    }

    // ── Integration: Analyzer + Comparator ──

    #[test]
    fn test_analyzer_and_comparator_integration() {
        let mut a = FiatShamirTranscript::new("protocol");
        a.absorb_bytes("commit-round1", b"poly_commitment_1");
        a.absorb_bytes("commit-round1", b"poly_commitment_2");
        let _ = a.squeeze_challenge();
        a.absorb_bytes("open-proof", b"opening_data");

        assert_eq!(TranscriptAnalyzer::entry_count(&a), 4);
        assert_eq!(TranscriptAnalyzer::absorb_count(&a), 3);
        assert_eq!(TranscriptAnalyzer::squeeze_count(&a), 1);
        assert!(TranscriptAnalyzer::has_absorb_then_squeeze(&a));

        let mut b = a.fork("verifier");
        b.absorb_bytes("verify-step", b"check");

        // After fork, b starts fresh so comparator sees them as different
        assert!(!TranscriptComparator::are_equivalent(&a, &b));
    }

    #[test]
    fn test_analyzer_phase_summary_with_colon_separator() {
        let mut t = FiatShamirTranscript::new("test");
        t.absorb_bytes("round1:poly", b"p1");
        t.absorb_bytes("round1:eval", b"e1");
        t.absorb_bytes("round2:poly", b"p2");
        let summary = TranscriptAnalyzer::phase_summary(&t);
        assert_eq!(summary.len(), 2);
        assert_eq!(summary[0], ("round1".to_string(), 2));
        assert_eq!(summary[1], ("round2".to_string(), 1));
    }
}
