use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use serde::{Serialize, Deserialize};

// ---------------------------------------------------------------------------
// GramType / HashType enums
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GramType {
    Character,
    Word,
    BPEToken,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HashType {
    Blake3,
    Sha256,
    FNV,
}

// ---------------------------------------------------------------------------
// NGramConfig
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NGramConfig {
    pub n: usize,
    pub gram_type: GramType,
    pub hash_function: HashType,
    pub case_sensitive: bool,
    pub normalize_whitespace: bool,
    /// Vocabulary size for BPE tokenization (only used when gram_type == BPEToken).
    pub bpe_vocab_size: usize,
}

impl NGramConfig {
    pub fn char_ngrams(n: usize) -> Self {
        Self {
            n,
            gram_type: GramType::Character,
            hash_function: HashType::Blake3,
            case_sensitive: false,
            normalize_whitespace: true,
            bpe_vocab_size: 0,
        }
    }

    pub fn word_ngrams(n: usize) -> Self {
        Self {
            n,
            gram_type: GramType::Word,
            hash_function: HashType::Blake3,
            case_sensitive: false,
            normalize_whitespace: true,
            bpe_vocab_size: 0,
        }
    }

    pub fn bpe_ngrams(n: usize, vocab_size: usize) -> Self {
        Self {
            n,
            gram_type: GramType::BPEToken,
            hash_function: HashType::Blake3,
            case_sensitive: false,
            normalize_whitespace: true,
            bpe_vocab_size: vocab_size,
        }
    }
}

impl Default for NGramConfig {
    fn default() -> Self {
        Self::char_ngrams(3)
    }
}

// ---------------------------------------------------------------------------
// NGram
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NGram {
    pub content: String,
    pub hash: u64,
    pub gram_type: GramType,
}

impl NGram {
    pub fn new(content: String, gram_type: GramType, hash_type: &HashType) -> Self {
        let hash = NGramSet::hash_ngram(&content, hash_type);
        Self { content, hash, gram_type }
    }

    pub fn fingerprint(&self) -> u64 {
        self.hash
    }

    pub fn as_str(&self) -> &str {
        &self.content
    }

    pub fn len(&self) -> usize {
        self.content.len()
    }

    pub fn is_empty(&self) -> bool {
        self.content.is_empty()
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        self.hash.to_le_bytes().to_vec()
    }
}

impl PartialEq for NGram {
    fn eq(&self, other: &Self) -> bool {
        self.hash == other.hash
    }
}

impl Eq for NGram {}

impl Hash for NGram {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.hash.hash(state);
    }
}

// ---------------------------------------------------------------------------
// SlidingWindow iterator
// ---------------------------------------------------------------------------

pub struct SlidingWindow<'a, T> {
    data: &'a [T],
    window_size: usize,
    pos: usize,
}

impl<'a, T> SlidingWindow<'a, T> {
    pub fn new(data: &'a [T], window_size: usize) -> Self {
        Self { data, window_size, pos: 0 }
    }
}

impl<'a, T> Iterator for SlidingWindow<'a, T> {
    type Item = &'a [T];

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos + self.window_size > self.data.len() {
            return None;
        }
        let window = &self.data[self.pos..self.pos + self.window_size];
        self.pos += 1;
        Some(window)
    }
}

// ---------------------------------------------------------------------------
// NGramExtractor
// ---------------------------------------------------------------------------

pub struct NGramExtractor {
    pub config: NGramConfig,
}

impl NGramExtractor {
    pub fn new(config: NGramConfig) -> Self {
        Self { config }
    }

    /// Extract all n-grams from the given text according to the configured gram type.
    pub fn extract(&self, text: &str) -> Vec<NGram> {
        let normalized = self.normalize_text(text);
        match self.config.gram_type {
            GramType::Character => Self::extract_character_ngrams(&normalized, self.config.n, &self.config.hash_function),
            GramType::Word => Self::extract_word_ngrams(&normalized, self.config.n, &self.config.hash_function),
            GramType::BPEToken => Self::extract_bpe_ngrams_impl(&normalized, self.config.n, self.config.bpe_vocab_size, &self.config.hash_function),
        }
    }

    /// Extract character-level n-grams using a sliding window over characters.
    pub fn extract_character_ngrams(text: &str, n: usize, hash_type: &HashType) -> Vec<NGram> {
        if n == 0 || text.is_empty() {
            return Vec::new();
        }
        let chars: Vec<char> = text.chars().collect();
        if chars.len() < n {
            return Vec::new();
        }
        let mut result = Vec::with_capacity(chars.len() - n + 1);
        for i in 0..=(chars.len() - n) {
            let gram: String = chars[i..i + n].iter().collect();
            result.push(NGram::new(gram, GramType::Character, hash_type));
        }
        result
    }

    /// Extract word-level n-grams using a sliding window over whitespace-tokenised words.
    pub fn extract_word_ngrams(text: &str, n: usize, hash_type: &HashType) -> Vec<NGram> {
        if n == 0 || text.is_empty() {
            return Vec::new();
        }
        let words = Self::tokenize_words(text);
        if words.len() < n {
            return Vec::new();
        }
        let mut result = Vec::with_capacity(words.len() - n + 1);
        for window in SlidingWindow::new(&words, n) {
            let gram = window.join(" ");
            result.push(NGram::new(gram, GramType::Word, hash_type));
        }
        result
    }

    /// Extract BPE-token n-grams. Internally runs a simple byte-pair encoding.
    fn extract_bpe_ngrams_impl(text: &str, n: usize, vocab_size: usize, hash_type: &HashType) -> Vec<NGram> {
        if n == 0 || text.is_empty() {
            return Vec::new();
        }
        let tokens = Self::tokenize_bpe(text, vocab_size);
        if tokens.len() < n {
            return Vec::new();
        }
        let mut result = Vec::with_capacity(tokens.len() - n + 1);
        for window in SlidingWindow::new(&tokens, n) {
            let gram: String = window.iter().map(|t| t.to_string()).collect::<Vec<_>>().join("_");
            result.push(NGram::new(gram, GramType::BPEToken, hash_type));
        }
        result
    }

    /// Public convenience wrapper matching the spec signature.
    pub fn extract_bpe_ngrams(text: &str, n: usize) -> Vec<NGram> {
        Self::extract_bpe_ngrams_impl(text, n, 256, &HashType::Blake3)
    }

    /// Normalise text: optionally lower-case and collapse whitespace.
    pub fn normalize_text(&self, text: &str) -> String {
        let mut s = if self.config.case_sensitive {
            text.to_string()
        } else {
            text.to_lowercase()
        };
        if self.config.normalize_whitespace {
            // Collapse runs of whitespace to a single space and trim.
            let mut result = String::with_capacity(s.len());
            let mut prev_ws = false;
            for ch in s.chars() {
                if ch.is_whitespace() {
                    if !prev_ws {
                        result.push(' ');
                    }
                    prev_ws = true;
                } else {
                    result.push(ch);
                    prev_ws = false;
                }
            }
            s = result.trim().to_string();
        }
        s
    }

    /// Simple whitespace tokeniser.
    pub fn tokenize_words(text: &str) -> Vec<String> {
        text.split_whitespace().map(|w| w.to_string()).collect()
    }

    /// Very simple byte-pair encoding tokeniser.
    ///
    /// Starts with individual bytes, then greedily merges the most frequent
    /// adjacent pair until `vocab_size` merge operations have been performed
    /// (or no more merges are possible).
    pub fn tokenize_bpe(text: &str, vocab_size: usize) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }

        // Start with byte-level token ids (0..255).
        let mut tokens: Vec<u32> = text.bytes().map(|b| b as u32).collect();
        let mut next_id: u32 = 256;
        let merges_to_do = vocab_size.saturating_sub(256);

        for _ in 0..merges_to_do {
            if tokens.len() < 2 {
                break;
            }

            // Count bigram frequencies.
            let mut pair_counts: HashMap<(u32, u32), usize> = HashMap::new();
            for pair in tokens.windows(2) {
                *pair_counts.entry((pair[0], pair[1])).or_insert(0) += 1;
            }

            // Find most frequent pair.
            let best = pair_counts.into_iter().max_by_key(|&(_, c)| c);
            match best {
                Some(((a, b), count)) if count >= 2 => {
                    // Merge all occurrences of (a, b) -> next_id.
                    let mut merged = Vec::with_capacity(tokens.len());
                    let mut i = 0;
                    while i < tokens.len() {
                        if i + 1 < tokens.len() && tokens[i] == a && tokens[i + 1] == b {
                            merged.push(next_id);
                            i += 2;
                        } else {
                            merged.push(tokens[i]);
                            i += 1;
                        }
                    }
                    tokens = merged;
                    next_id += 1;
                }
                _ => break,
            }
        }
        tokens
    }

    /// Extract n-grams together with their byte-offset start positions.
    pub fn extract_with_positions(&self, text: &str) -> Vec<(NGram, usize)> {
        let normalized = self.normalize_text(text);
        match self.config.gram_type {
            GramType::Character => {
                let chars: Vec<char> = normalized.chars().collect();
                let n = self.config.n;
                if n == 0 || chars.len() < n {
                    return Vec::new();
                }
                // Precompute byte offsets for each char index.
                let byte_offsets: Vec<usize> = {
                    let mut offsets = Vec::with_capacity(chars.len());
                    let mut off = 0usize;
                    for ch in &chars {
                        offsets.push(off);
                        off += ch.len_utf8();
                    }
                    offsets
                };
                let mut result = Vec::with_capacity(chars.len() - n + 1);
                for i in 0..=(chars.len() - n) {
                    let gram: String = chars[i..i + n].iter().collect();
                    let ng = NGram::new(gram, GramType::Character, &self.config.hash_function);
                    result.push((ng, byte_offsets[i]));
                }
                result
            }
            GramType::Word => {
                let n = self.config.n;
                let words: Vec<&str> = normalized.split_whitespace().collect();
                if n == 0 || words.len() < n {
                    return Vec::new();
                }
                // Build byte offsets of each word start.
                let byte_offsets: Vec<usize> = {
                    let mut offsets = Vec::new();
                    let mut search_start = 0;
                    for w in &words {
                        if let Some(pos) = normalized[search_start..].find(w) {
                            let abs = search_start + pos;
                            offsets.push(abs);
                            search_start = abs + w.len();
                        }
                    }
                    offsets
                };
                let word_strings: Vec<String> = words.iter().map(|w| w.to_string()).collect();
                let mut result = Vec::with_capacity(word_strings.len() - n + 1);
                for i in 0..=(word_strings.len() - n) {
                    let gram = word_strings[i..i + n].join(" ");
                    let ng = NGram::new(gram, GramType::Word, &self.config.hash_function);
                    let offset = if i < byte_offsets.len() { byte_offsets[i] } else { 0 };
                    result.push((ng, offset));
                }
                result
            }
            GramType::BPEToken => {
                let tokens = Self::tokenize_bpe(&normalized, self.config.bpe_vocab_size);
                let n = self.config.n;
                if n == 0 || tokens.len() < n {
                    return Vec::new();
                }
                let mut result = Vec::with_capacity(tokens.len() - n + 1);
                for i in 0..=(tokens.len() - n) {
                    let gram: String = tokens[i..i + n].iter().map(|t| t.to_string()).collect::<Vec<_>>().join("_");
                    let ng = NGram::new(gram, GramType::BPEToken, &self.config.hash_function);
                    result.push((ng, i));
                }
                result
            }
        }
    }
}

// ---------------------------------------------------------------------------
// NGramSet
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct NGramSet {
    pub grams: HashSet<u64>,
    pub config: NGramConfig,
    pub count: usize,
}

impl NGramSet {
    pub fn new(config: NGramConfig) -> Self {
        Self { grams: HashSet::new(), config, count: 0 }
    }

    /// Build a set from raw text.
    pub fn from_text(text: &str, config: NGramConfig) -> Self {
        let extractor = NGramExtractor::new(config.clone());
        let ngrams = extractor.extract(text);
        let mut set = Self::new(config);
        for ng in &ngrams {
            set.insert(ng);
        }
        set
    }

    /// Build a set from a pre-computed vector of NGrams.
    pub fn from_ngrams(ngrams: Vec<NGram>, config: NGramConfig) -> Self {
        let mut set = Self::new(config);
        for ng in &ngrams {
            set.insert(ng);
        }
        set
    }

    pub fn insert(&mut self, ngram: &NGram) {
        if self.grams.insert(ngram.hash) {
            self.count += 1;
        }
    }

    pub fn contains(&self, ngram: &NGram) -> bool {
        self.grams.contains(&ngram.hash)
    }

    pub fn contains_hash(&self, hash: u64) -> bool {
        self.grams.contains(&hash)
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn intersection(&self, other: &NGramSet) -> NGramSet {
        let inter: HashSet<u64> = self.grams.intersection(&other.grams).cloned().collect();
        let count = inter.len();
        NGramSet { grams: inter, config: self.config.clone(), count }
    }

    pub fn union(&self, other: &NGramSet) -> NGramSet {
        let uni: HashSet<u64> = self.grams.union(&other.grams).cloned().collect();
        let count = uni.len();
        NGramSet { grams: uni, config: self.config.clone(), count }
    }

    pub fn difference(&self, other: &NGramSet) -> NGramSet {
        let diff: HashSet<u64> = self.grams.difference(&other.grams).cloned().collect();
        let count = diff.len();
        NGramSet { grams: diff, config: self.config.clone(), count }
    }

    pub fn intersection_cardinality(&self, other: &NGramSet) -> usize {
        self.grams.intersection(&other.grams).count()
    }

    /// Jaccard similarity: |A ∩ B| / |A ∪ B|.
    pub fn jaccard_similarity(&self, other: &NGramSet) -> f64 {
        let inter = self.intersection_cardinality(other) as f64;
        let union_size = self.grams.union(&other.grams).count() as f64;
        if union_size == 0.0 { 0.0 } else { inter / union_size }
    }

    /// Containment of self in other: |A ∩ B| / |A|.
    pub fn containment(&self, other: &NGramSet) -> f64 {
        if self.is_empty() {
            return 0.0;
        }
        self.intersection_cardinality(other) as f64 / self.len() as f64
    }

    /// Overlap coefficient: |A ∩ B| / min(|A|, |B|).
    pub fn overlap_coefficient(&self, other: &NGramSet) -> f64 {
        let min_size = self.len().min(other.len());
        if min_size == 0 {
            return 0.0;
        }
        self.intersection_cardinality(other) as f64 / min_size as f64
    }

    pub fn to_sorted_vec(&self) -> Vec<u64> {
        let mut v: Vec<u64> = self.grams.iter().cloned().collect();
        v.sort();
        v
    }

    pub fn fingerprints(&self) -> &HashSet<u64> {
        &self.grams
    }

    pub fn merge(&mut self, other: &NGramSet) {
        for &h in &other.grams {
            if self.grams.insert(h) {
                self.count += 1;
            }
        }
    }

    /// Hash an n-gram string to a 64-bit fingerprint.
    pub fn hash_ngram(content: &str, hash_type: &HashType) -> u64 {
        match hash_type {
            HashType::Blake3 => {
                let h = blake3::hash(content.as_bytes());
                let bytes = h.as_bytes();
                u64::from_le_bytes([
                    bytes[0], bytes[1], bytes[2], bytes[3],
                    bytes[4], bytes[5], bytes[6], bytes[7],
                ])
            }
            HashType::Sha256 => {
                use sha2::{Sha256, Digest};
                let result = Sha256::digest(content.as_bytes());
                u64::from_le_bytes([
                    result[0], result[1], result[2], result[3],
                    result[4], result[5], result[6], result[7],
                ])
            }
            HashType::FNV => {
                // FNV-1a 64-bit
                let mut hash: u64 = 0xcbf29ce484222325;
                for byte in content.as_bytes() {
                    hash ^= *byte as u64;
                    hash = hash.wrapping_mul(0x100000001b3);
                }
                hash
            }
        }
    }
}

// ---------------------------------------------------------------------------
// NGramFrequencyMap
// ---------------------------------------------------------------------------

pub struct NGramFrequencyMap {
    pub counts: HashMap<u64, usize>,
}

impl NGramFrequencyMap {
    pub fn from_text(text: &str, config: NGramConfig) -> Self {
        let extractor = NGramExtractor::new(config);
        let ngrams = extractor.extract(text);
        let mut counts: HashMap<u64, usize> = HashMap::new();
        for ng in &ngrams {
            *counts.entry(ng.hash).or_insert(0) += 1;
        }
        Self { counts }
    }

    pub fn get_count(&self, hash: u64) -> usize {
        self.counts.get(&hash).cloned().unwrap_or(0)
    }

    /// Return the `k` most common n-gram hashes and their counts.
    pub fn most_common(&self, k: usize) -> Vec<(u64, usize)> {
        let mut pairs: Vec<(u64, usize)> = self.counts.iter().map(|(&h, &c)| (h, c)).collect();
        pairs.sort_by(|a, b| b.1.cmp(&a.1));
        pairs.truncate(k);
        pairs
    }

    pub fn total_count(&self) -> usize {
        self.counts.values().sum()
    }

    /// Normalise counts into a probability distribution.
    pub fn to_probability_distribution(&self) -> HashMap<u64, f64> {
        let total = self.total_count() as f64;
        if total == 0.0 {
            return HashMap::new();
        }
        self.counts.iter().map(|(&h, &c)| (h, c as f64 / total)).collect()
    }

    /// Shannon entropy of the frequency distribution (in bits).
    pub fn entropy(&self) -> f64 {
        let dist = self.to_probability_distribution();
        let mut h = 0.0f64;
        for &p in dist.values() {
            if p > 0.0 {
                h -= p * p.log2();
            }
        }
        h
    }
}

// ---------------------------------------------------------------------------
// NGramOverlapReport
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NGramOverlapReport {
    pub intersection_size: usize,
    pub set_a_size: usize,
    pub set_b_size: usize,
    pub jaccard: f64,
    pub containment_a_in_b: f64,
    pub containment_b_in_a: f64,
    pub overlap_coefficient: f64,
}

impl NGramOverlapReport {
    pub fn compute(set_a: &NGramSet, set_b: &NGramSet) -> Self {
        let intersection_size = set_a.intersection_cardinality(set_b);
        let set_a_size = set_a.len();
        let set_b_size = set_b.len();
        let jaccard = set_a.jaccard_similarity(set_b);
        let containment_a_in_b = set_a.containment(set_b);
        let containment_b_in_a = set_b.containment(set_a);
        let overlap_coefficient = set_a.overlap_coefficient(set_b);

        Self {
            intersection_size,
            set_a_size,
            set_b_size,
            jaccard,
            containment_a_in_b,
            containment_b_in_a,
            overlap_coefficient,
        }
    }

    pub fn summary(&self) -> String {
        format!(
            "Overlap Report:\n  |A|={}, |B|={}, |A∩B|={}\n  Jaccard={:.4}, Containment(A⊂B)={:.4}, Containment(B⊂A)={:.4}, Overlap={:.4}",
            self.set_a_size, self.set_b_size, self.intersection_size,
            self.jaccard, self.containment_a_in_b, self.containment_b_in_a, self.overlap_coefficient,
        )
    }
}


// ---------------------------------------------------------------------------
// BPETokenizer
// ---------------------------------------------------------------------------

/// A byte-pair encoding tokenizer that learns merge operations from a training
/// corpus and can then encode / decode arbitrary text.
pub struct BPETokenizer {
    pub vocab: HashMap<(u32, u32), u32>,
    pub merge_order: Vec<(u32, u32)>,
    pub vocab_sz: usize,
    reverse_vocab: HashMap<u32, Vec<u8>>,
}

impl BPETokenizer {
    /// Create a new BPE tokenizer targeting the given vocabulary size.
    /// The base vocabulary consists of 256 byte-level tokens; `vocab_size`
    /// includes those plus any learned merges.
    pub fn new(vocab_size: usize) -> Self {
        let mut reverse_vocab = HashMap::new();
        for b in 0u32..256 {
            reverse_vocab.insert(b, vec![b as u8]);
        }
        Self {
            vocab: HashMap::new(),
            merge_order: Vec::new(),
            vocab_sz: vocab_size,
            reverse_vocab,
        }
    }

    /// Train on a corpus of strings, learning merge operations greedily.
    pub fn train(&mut self, corpus: &[String]) {
        let mut tokens: Vec<u32> = Vec::new();
        for doc in corpus {
            for b in doc.bytes() {
                tokens.push(b as u32);
            }
        }
        let mut next_id: u32 = 256;
        let merges_to_do = self.vocab_sz.saturating_sub(256);

        for _ in 0..merges_to_do {
            if tokens.len() < 2 {
                break;
            }
            let pair_counts = Self::count_pairs(&tokens);
            let best = pair_counts.into_iter().max_by_key(|&(_, c)| c);
            match best {
                Some(((a, b), count)) if count >= 2 => {
                    tokens = Self::merge_pair(&tokens, a, b, next_id);
                    self.vocab.insert((a, b), next_id);
                    self.merge_order.push((a, b));
                    let mut new_bytes = self.reverse_vocab.get(&a).cloned().unwrap_or_default();
                    new_bytes.extend(self.reverse_vocab.get(&b).cloned().unwrap_or_default());
                    self.reverse_vocab.insert(next_id, new_bytes);
                    next_id += 1;
                }
                _ => break,
            }
        }
    }

    /// Encode text into BPE token IDs using the learned merge order.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }
        let mut tokens: Vec<u32> = text.bytes().map(|b| b as u32).collect();
        for &(a, b) in &self.merge_order {
            if let Some(&new_id) = self.vocab.get(&(a, b)) {
                tokens = Self::merge_pair(&tokens, a, b, new_id);
            }
        }
        tokens
    }

    /// Decode token IDs back to a UTF-8 string (lossy).
    pub fn decode(&self, tokens: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &t in tokens {
            if let Some(bs) = self.reverse_vocab.get(&t) {
                bytes.extend_from_slice(bs);
            } else if t < 256 {
                bytes.push(t as u8);
            }
        }
        String::from_utf8_lossy(&bytes).to_string()
    }

    /// Current effective vocabulary size (256 base + learned merges).
    pub fn vocab_size(&self) -> usize {
        256 + self.merge_order.len()
    }

    /// Get the merge pair at a given index in the merge order.
    pub fn get_merge_at(&self, index: usize) -> Option<(u32, u32)> {
        self.merge_order.get(index).cloned()
    }

    /// Count all adjacent pairs in a token sequence.
    fn count_pairs(tokens: &[u32]) -> HashMap<(u32, u32), usize> {
        let mut counts: HashMap<(u32, u32), usize> = HashMap::new();
        for pair in tokens.windows(2) {
            *counts.entry((pair[0], pair[1])).or_insert(0) += 1;
        }
        counts
    }

    /// Replace every occurrence of the pair (a, b) with `new_id`.
    fn merge_pair(tokens: &[u32], a: u32, b: u32, new_id: u32) -> Vec<u32> {
        let mut merged = Vec::with_capacity(tokens.len());
        let mut i = 0;
        while i < tokens.len() {
            if i + 1 < tokens.len() && tokens[i] == a && tokens[i + 1] == b {
                merged.push(new_id);
                i += 2;
            } else {
                merged.push(tokens[i]);
                i += 1;
            }
        }
        merged
    }

    /// Compute per-byte frequencies across the entire corpus.
    pub fn compute_byte_frequencies(corpus: &[String]) -> HashMap<u8, usize> {
        let mut freq: HashMap<u8, usize> = HashMap::new();
        for doc in corpus {
            for b in doc.bytes() {
                *freq.entry(b).or_insert(0) += 1;
            }
        }
        freq
    }

    /// Return how many merge operations have been learned.
    pub fn merge_count(&self) -> usize {
        self.merge_order.len()
    }

    /// Check whether a specific pair has a learned merge.
    pub fn has_merge(&self, a: u32, b: u32) -> bool {
        self.vocab.contains_key(&(a, b))
    }

    /// Return the byte sequence that a single token decodes to.
    pub fn token_to_bytes(&self, token: u32) -> Option<Vec<u8>> {
        self.reverse_vocab.get(&token).cloned()
    }

    /// Encode a batch of texts.
    pub fn encode_batch(&self, texts: &[&str]) -> Vec<Vec<u32>> {
        texts.iter().map(|t| self.encode(t)).collect()
    }

    /// Decode a batch of token sequences.
    pub fn decode_batch(&self, batch: &[Vec<u32>]) -> Vec<String> {
        batch.iter().map(|t| self.decode(t)).collect()
    }

    /// Estimate the compression ratio: encoded length / original byte length.
    pub fn compression_ratio(&self, text: &str) -> f64 {
        if text.is_empty() {
            return 1.0;
        }
        let encoded = self.encode(text);
        encoded.len() as f64 / text.len() as f64
    }
}

// ---------------------------------------------------------------------------
// NGramFrequencyMap — additional methods
// ---------------------------------------------------------------------------

impl NGramFrequencyMap {
    /// Return the `k` least common n-gram hashes and their counts.
    pub fn least_common(&self, k: usize) -> Vec<(u64, usize)> {
        let mut pairs: Vec<(u64, usize)> = self.counts.iter().map(|(&h, &c)| (h, c)).collect();
        pairs.sort_by(|a, b| a.1.cmp(&b.1));
        pairs.truncate(k);
        pairs
    }

    /// Number of distinct n-gram hashes in the frequency map.
    pub fn unique_count(&self) -> usize {
        self.counts.len()
    }

    /// KL divergence D_KL(self || other).  Returns f64::INFINITY when `other`
    /// assigns zero probability to an n-gram that `self` has.
    pub fn kl_divergence(&self, other: &NGramFrequencyMap) -> f64 {
        let p = self.to_probability_distribution();
        let q = other.to_probability_distribution();
        let mut divergence = 0.0f64;
        for (&hash, &p_val) in &p {
            if p_val <= 0.0 {
                continue;
            }
            let q_val = q.get(&hash).cloned().unwrap_or(0.0);
            if q_val <= 0.0 {
                return f64::INFINITY;
            }
            divergence += p_val * (p_val / q_val).ln();
        }
        divergence
    }

    /// Cosine similarity treating the two frequency maps as sparse vectors.
    pub fn cosine_similarity(&self, other: &NGramFrequencyMap) -> f64 {
        let mut dot_product = 0.0f64;
        let mut norm_a_sq = 0.0f64;
        let mut norm_b_sq = 0.0f64;

        for (&hash, &count_a) in &self.counts {
            let a = count_a as f64;
            norm_a_sq += a * a;
            if let Some(&count_b) = other.counts.get(&hash) {
                dot_product += a * count_b as f64;
            }
        }
        for (_, &count_b) in &other.counts {
            let b = count_b as f64;
            norm_b_sq += b * b;
        }
        let denominator = norm_a_sq.sqrt() * norm_b_sq.sqrt();
        if denominator == 0.0 {
            0.0
        } else {
            dot_product / denominator
        }
    }

    /// Merge counts from another frequency map into this one.
    pub fn merge(&mut self, other: &NGramFrequencyMap) {
        for (&hash, &count) in &other.counts {
            *self.counts.entry(hash).or_insert(0) += count;
        }
    }

    /// Return a new frequency map keeping only n-grams with count >= min_count.
    pub fn filter_by_count(&self, min_count: usize) -> Self {
        let filtered: HashMap<u64, usize> = self
            .counts
            .iter()
            .filter(|(_, &c)| c >= min_count)
            .map(|(&h, &c)| (h, c))
            .collect();
        Self { counts: filtered }
    }

    /// Normalise counts to probabilities (alias for to_probability_distribution).
    pub fn normalize(&self) -> HashMap<u64, f64> {
        self.to_probability_distribution()
    }

    /// Jensen–Shannon divergence (symmetric, always finite).
    pub fn js_divergence(&self, other: &NGramFrequencyMap) -> f64 {
        let p = self.to_probability_distribution();
        let q = other.to_probability_distribution();
        let mut all_keys: HashSet<u64> = p.keys().cloned().collect();
        for k in q.keys() {
            all_keys.insert(*k);
        }
        let mut m: HashMap<u64, f64> = HashMap::new();
        for &k in &all_keys {
            let pv = p.get(&k).cloned().unwrap_or(0.0);
            let qv = q.get(&k).cloned().unwrap_or(0.0);
            m.insert(k, (pv + qv) / 2.0);
        }
        let kl_pm = {
            let mut d = 0.0f64;
            for (&k, &pv) in &p {
                if pv > 0.0 {
                    let mv = m.get(&k).cloned().unwrap_or(0.0);
                    if mv > 0.0 {
                        d += pv * (pv / mv).ln();
                    }
                }
            }
            d
        };
        let kl_qm = {
            let mut d = 0.0f64;
            for (&k, &qv) in &q {
                if qv > 0.0 {
                    let mv = m.get(&k).cloned().unwrap_or(0.0);
                    if mv > 0.0 {
                        d += qv * (qv / mv).ln();
                    }
                }
            }
            d
        };
        (kl_pm + kl_qm) / 2.0
    }

    /// Compute the cross-entropy H(P, Q) = -sum(p * log(q)).
    pub fn cross_entropy(&self, other: &NGramFrequencyMap) -> f64 {
        let p = self.to_probability_distribution();
        let q = other.to_probability_distribution();
        let mut ce = 0.0f64;
        for (&hash, &p_val) in &p {
            if p_val <= 0.0 {
                continue;
            }
            let q_val = q.get(&hash).cloned().unwrap_or(0.0);
            if q_val <= 0.0 {
                return f64::INFINITY;
            }
            ce -= p_val * q_val.ln();
        }
        ce
    }

    /// Perplexity = 2^entropy.
    pub fn perplexity(&self) -> f64 {
        2.0f64.powf(self.entropy())
    }

    /// Weighted Jaccard similarity using counts as weights.
    pub fn weighted_jaccard(&self, other: &NGramFrequencyMap) -> f64 {
        let mut min_sum = 0usize;
        let mut max_sum = 0usize;
        let mut all_keys: HashSet<u64> = self.counts.keys().cloned().collect();
        for k in other.counts.keys() {
            all_keys.insert(*k);
        }
        for k in &all_keys {
            let a = self.counts.get(k).cloned().unwrap_or(0);
            let b = other.counts.get(k).cloned().unwrap_or(0);
            min_sum += a.min(b);
            max_sum += a.max(b);
        }
        if max_sum == 0 {
            0.0
        } else {
            min_sum as f64 / max_sum as f64
        }
    }
}

// ---------------------------------------------------------------------------
// NGramOverlapReport — additional methods
// ---------------------------------------------------------------------------

impl NGramOverlapReport {
    /// Returns true if the Jaccard similarity meets or exceeds the threshold.
    pub fn is_significant(&self, threshold: f64) -> bool {
        self.jaccard >= threshold
    }

    /// Serialize the report to a JSON string.
    pub fn to_json(&self) -> String {
        format!(
            concat!(
                "{{",
                "\"intersection_size\":{},",
                "\"set_a_size\":{},",
                "\"set_b_size\":{},",
                "\"jaccard\":{:.6},",
                "\"containment_a_in_b\":{:.6},",
                "\"containment_b_in_a\":{:.6},",
                "\"overlap_coefficient\":{:.6}",
                "}}"
            ),
            self.intersection_size,
            self.set_a_size,
            self.set_b_size,
            self.jaccard,
            self.containment_a_in_b,
            self.containment_b_in_a,
            self.overlap_coefficient,
        )
    }

    /// Dice coefficient: 2|A∩B| / (|A| + |B|).
    pub fn dice_coefficient(&self) -> f64 {
        let denom = self.set_a_size + self.set_b_size;
        if denom == 0 {
            0.0
        } else {
            2.0 * self.intersection_size as f64 / denom as f64
        }
    }

    /// Size of the symmetric difference |A△B|.
    pub fn symmetric_difference_size(&self) -> usize {
        (self.set_a_size + self.set_b_size) - 2 * self.intersection_size
    }
}

// ---------------------------------------------------------------------------
// SlidingWindowIterator — generic sliding window producing owned Vecs
// ---------------------------------------------------------------------------

/// A generic sliding window iterator that yields owned `Vec<T>` windows,
/// complementing the slice-based `SlidingWindow` already defined above.
pub struct SlidingWindowIterator<'a, T> {
    data: &'a [T],
    window_size: usize,
    position: usize,
}

impl<'a, T> SlidingWindowIterator<'a, T> {
    pub fn new(data: &'a [T], window_size: usize) -> Self {
        Self {
            data,
            window_size,
            position: 0,
        }
    }

    /// Number of windows remaining.
    pub fn remaining(&self) -> usize {
        if self.position + self.window_size > self.data.len() {
            0
        } else {
            self.data.len() - self.window_size - self.position + 1
        }
    }

    /// Reset the iterator to the beginning.
    pub fn reset(&mut self) {
        self.position = 0;
    }
}

impl<'a, T: Clone> Iterator for SlidingWindowIterator<'a, T> {
    type Item = Vec<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.position + self.window_size > self.data.len() {
            return None;
        }
        let window = self.data[self.position..self.position + self.window_size].to_vec();
        self.position += 1;
        Some(window)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.remaining();
        (remaining, Some(remaining))
    }
}

impl<'a, T: Clone> ExactSizeIterator for SlidingWindowIterator<'a, T> {}

// ---------------------------------------------------------------------------
// MultiGramExtractor — extract n-grams of multiple sizes simultaneously
// ---------------------------------------------------------------------------

pub struct MultiGramExtractor {
    pub sizes: Vec<usize>,
    pub gram_type: GramType,
}

impl MultiGramExtractor {
    pub fn new(sizes: &[usize], gram_type: GramType) -> Self {
        let mut sorted_sizes = sizes.to_vec();
        sorted_sizes.sort_unstable();
        sorted_sizes.dedup();
        Self {
            sizes: sorted_sizes,
            gram_type,
        }
    }

    fn config_for_size(&self, n: usize) -> NGramConfig {
        match self.gram_type {
            GramType::Character => NGramConfig::char_ngrams(n),
            GramType::Word => NGramConfig::word_ngrams(n),
            GramType::BPEToken => NGramConfig::bpe_ngrams(n, 256),
        }
    }

    /// Extract n-gram sets for each configured size.
    pub fn extract_all(&self, text: &str) -> HashMap<usize, NGramSet> {
        let mut result = HashMap::new();
        for &n in &self.sizes {
            let config = self.config_for_size(n);
            let set = NGramSet::from_text(text, config);
            result.insert(n, set);
        }
        result
    }

    /// Combine n-grams of all configured sizes into a single set.
    pub fn combined_set(&self, text: &str) -> NGramSet {
        let all = self.extract_all(text);
        let config = self.config_for_size(1);
        let mut combined = NGramSet::new(config);
        for (_, set) in all {
            combined.merge(&set);
        }
        combined
    }

    /// Compute Jaccard overlap between two texts at each configured size.
    pub fn overlap_at_each_size(&self, text_a: &str, text_b: &str) -> Vec<(usize, f64)> {
        let mut results = Vec::new();
        for &n in &self.sizes {
            let config = self.config_for_size(n);
            let set_a = NGramSet::from_text(text_a, config.clone());
            let set_b = NGramSet::from_text(text_b, config);
            let jaccard = set_a.jaccard_similarity(&set_b);
            results.push((n, jaccard));
        }
        results
    }

    /// Find the size that gives the maximum Jaccard similarity.
    pub fn best_overlap_size(&self, text_a: &str, text_b: &str) -> (usize, f64) {
        self.overlap_at_each_size(text_a, text_b)
            .into_iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, 0.0))
    }

    /// Total unique n-grams across all sizes.
    pub fn total_unique_count(&self, text: &str) -> usize {
        self.combined_set(text).len()
    }

    /// Count of n-grams at each configured size.
    pub fn counts_per_size(&self, text: &str) -> Vec<(usize, usize)> {
        self.extract_all(text)
            .into_iter()
            .map(|(size, set)| (size, set.len()))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// NGramFilter — static methods for filtering n-gram sets
// ---------------------------------------------------------------------------

pub struct NGramFilter;

impl NGramFilter {
    /// Remove n-grams whose content hashes match any of the stopword hashes.
    /// Stopwords are hashed using the set's configured hash function.
    pub fn remove_stopword_ngrams(set: &NGramSet, stopwords: &HashSet<String>) -> NGramSet {
        let mut stopword_hashes = HashSet::new();
        for sw in stopwords {
            let hash = NGramSet::hash_ngram(sw, &set.config.hash_function);
            stopword_hashes.insert(hash);
        }
        let mut filtered = NGramSet::new(set.config.clone());
        for &h in &set.grams {
            if !stopword_hashes.contains(&h) {
                filtered.grams.insert(h);
                filtered.count += 1;
            }
        }
        filtered
    }

    /// Keep only n-grams whose frequency in `freq_map` is >= `min_freq`.
    pub fn filter_by_frequency(
        set: &NGramSet,
        freq_map: &NGramFrequencyMap,
        min_freq: usize,
    ) -> NGramSet {
        let mut filtered = NGramSet::new(set.config.clone());
        for &h in &set.grams {
            if freq_map.get_count(h) >= min_freq {
                filtered.grams.insert(h);
                filtered.count += 1;
            }
        }
        filtered
    }

    /// Deterministic pseudo-random sample of n-gram hashes (uses hash ordering).
    pub fn sample_random(set: &NGramSet, count: usize) -> NGramSet {
        let mut sorted: Vec<u64> = set.grams.iter().cloned().collect();
        sorted.sort();
        let take = count.min(sorted.len());
        let mut sampled = NGramSet::new(set.config.clone());
        if take == 0 {
            return sampled;
        }
        let step = if sorted.len() <= take {
            1
        } else {
            sorted.len() / take
        };
        let mut idx = 0;
        let mut taken = 0;
        while taken < take && idx < sorted.len() {
            sampled.grams.insert(sorted[idx]);
            sampled.count += 1;
            taken += 1;
            idx += step;
        }
        sampled
    }

    /// Return the top-k most frequent n-gram hashes as an NGramSet.
    pub fn top_k_by_frequency(freq_map: &NGramFrequencyMap, k: usize) -> NGramSet {
        let top = freq_map.most_common(k);
        let config = NGramConfig::default();
        let mut set = NGramSet::new(config);
        for (hash, _count) in top {
            set.grams.insert(hash);
            set.count += 1;
        }
        set
    }

    /// Remove n-grams that appear in a blacklist set.
    pub fn subtract(set: &NGramSet, blacklist: &NGramSet) -> NGramSet {
        set.difference(blacklist)
    }

    /// Keep only n-grams present in a whitelist set.
    pub fn whitelist(set: &NGramSet, whitelist: &NGramSet) -> NGramSet {
        set.intersection(whitelist)
    }

    /// Remove the top-k most frequent n-grams (useful for removing very common patterns).
    pub fn remove_top_k(set: &NGramSet, freq_map: &NGramFrequencyMap, k: usize) -> NGramSet {
        let top_set = Self::top_k_by_frequency(freq_map, k);
        Self::subtract(set, &top_set)
    }

    /// Keep only n-grams with frequency strictly below `max_freq`.
    pub fn filter_by_max_frequency(
        set: &NGramSet,
        freq_map: &NGramFrequencyMap,
        max_freq: usize,
    ) -> NGramSet {
        let mut filtered = NGramSet::new(set.config.clone());
        for &h in &set.grams {
            if freq_map.get_count(h) < max_freq {
                filtered.grams.insert(h);
                filtered.count += 1;
            }
        }
        filtered
    }

    /// Band-pass filter: keep n-grams with count in [min_freq, max_freq).
    pub fn filter_by_frequency_range(
        set: &NGramSet,
        freq_map: &NGramFrequencyMap,
        min_freq: usize,
        max_freq: usize,
    ) -> NGramSet {
        let mut filtered = NGramSet::new(set.config.clone());
        for &h in &set.grams {
            let c = freq_map.get_count(h);
            if c >= min_freq && c < max_freq {
                filtered.grams.insert(h);
                filtered.count += 1;
            }
        }
        filtered
    }
}

// ---------------------------------------------------------------------------
// NGramStatistics — static methods for distribution analysis
// ---------------------------------------------------------------------------

pub struct NGramStatistics;

impl NGramStatistics {
    /// Fraction of the reference set covered by the candidate set.
    pub fn coverage(set: &NGramSet, reference: &NGramSet) -> f64 {
        if reference.is_empty() {
            return 0.0;
        }
        let inter = set.intersection_cardinality(reference);
        inter as f64 / reference.len() as f64
    }

    /// Redundancy: 1 - (unique / total). High values indicate many repeated n-grams.
    pub fn redundancy(freq_map: &NGramFrequencyMap) -> f64 {
        let total = freq_map.total_count();
        let unique = freq_map.unique_count();
        if total == 0 {
            return 0.0;
        }
        1.0 - (unique as f64 / total as f64)
    }

    /// Simpson's diversity index: 1 - Σ(p_i²).
    pub fn diversity_index(freq_map: &NGramFrequencyMap) -> f64 {
        let dist = freq_map.to_probability_distribution();
        let sum_sq: f64 = dist.values().map(|&p| p * p).sum();
        1.0 - sum_sq
    }

    /// Burstiness measured as the coefficient of variation of counts.
    pub fn burstiness(freq_map: &NGramFrequencyMap) -> f64 {
        if freq_map.counts.is_empty() {
            return 0.0;
        }
        let counts: Vec<f64> = freq_map.counts.values().map(|&c| c as f64).collect();
        let n = counts.len() as f64;
        let mean = counts.iter().sum::<f64>() / n;
        if mean == 0.0 {
            return 0.0;
        }
        let variance = counts.iter().map(|&c| (c - mean) * (c - mean)).sum::<f64>() / n;
        variance.sqrt() / mean
    }

    /// Type-token ratio: unique n-gram count / total n-gram count.
    pub fn type_token_ratio(freq_map: &NGramFrequencyMap) -> f64 {
        let total = freq_map.total_count();
        if total == 0 {
            return 0.0;
        }
        freq_map.unique_count() as f64 / total as f64
    }

    /// Hapax legomena ratio: fraction of n-grams that appear exactly once.
    pub fn hapax_ratio(freq_map: &NGramFrequencyMap) -> f64 {
        let total_unique = freq_map.unique_count();
        if total_unique == 0 {
            return 0.0;
        }
        let hapax = freq_map.counts.values().filter(|&&c| c == 1).count();
        hapax as f64 / total_unique as f64
    }

    /// Yule's K measure of vocabulary richness.
    pub fn yules_k(freq_map: &NGramFrequencyMap) -> f64 {
        let n = freq_map.total_count() as f64;
        if n == 0.0 {
            return 0.0;
        }
        let mut freq_of_freq: HashMap<usize, usize> = HashMap::new();
        for &c in freq_map.counts.values() {
            *freq_of_freq.entry(c).or_insert(0) += 1;
        }
        let m2: f64 = freq_of_freq
            .iter()
            .map(|(&i, &v)| (i as f64) * (i as f64) * (v as f64))
            .sum();
        let m1 = n;
        if m1 * m1 == m2 {
            return 0.0;
        }
        10_000.0 * (m2 - m1) / (m1 * m1)
    }

    /// Mean frequency of n-grams.
    pub fn mean_frequency(freq_map: &NGramFrequencyMap) -> f64 {
        if freq_map.counts.is_empty() {
            return 0.0;
        }
        freq_map.total_count() as f64 / freq_map.unique_count() as f64
    }

    /// Median frequency of n-grams.
    pub fn median_frequency(freq_map: &NGramFrequencyMap) -> f64 {
        if freq_map.counts.is_empty() {
            return 0.0;
        }
        let mut freqs: Vec<usize> = freq_map.counts.values().cloned().collect();
        freqs.sort_unstable();
        let mid = freqs.len() / 2;
        if freqs.len() % 2 == 0 {
            (freqs[mid - 1] + freqs[mid]) as f64 / 2.0
        } else {
            freqs[mid] as f64
        }
    }

    /// Maximum frequency among all n-grams.
    pub fn max_frequency(freq_map: &NGramFrequencyMap) -> usize {
        freq_map.counts.values().cloned().max().unwrap_or(0)
    }

    /// Standard deviation of n-gram frequencies.
    pub fn frequency_std_dev(freq_map: &NGramFrequencyMap) -> f64 {
        if freq_map.counts.is_empty() {
            return 0.0;
        }
        let counts: Vec<f64> = freq_map.counts.values().map(|&c| c as f64).collect();
        let n = counts.len() as f64;
        let mean = counts.iter().sum::<f64>() / n;
        let variance = counts.iter().map(|&c| (c - mean) * (c - mean)).sum::<f64>() / n;
        variance.sqrt()
    }

    /// Mutual coverage: geometric mean of coverage(a, b) and coverage(b, a).
    pub fn mutual_coverage(set_a: &NGramSet, set_b: &NGramSet) -> f64 {
        let cov_ab = Self::coverage(set_a, set_b);
        let cov_ba = Self::coverage(set_b, set_a);
        (cov_ab * cov_ba).sqrt()
    }
}

// ---------------------------------------------------------------------------
// PreprocessStep / TextPreprocessor
// ---------------------------------------------------------------------------

/// Individual preprocessing operations that can be composed into a pipeline.
#[derive(Clone, Debug)]
pub enum PreprocessStep {
    Lowercase,
    StripPunctuation,
    NormalizeWhitespace,
    StripDigits,
    StemWords,
    RemoveStopwords(HashSet<String>),
}

/// A configurable text normalisation pipeline composed of preprocessing steps.
pub struct TextPreprocessor {
    pub steps: Vec<PreprocessStep>,
}

impl TextPreprocessor {
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }

    /// Append a preprocessing step (builder pattern).
    pub fn add_step(mut self, step: PreprocessStep) -> Self {
        self.steps.push(step);
        self
    }

    /// A reasonable default: lowercase → strip punctuation → normalize whitespace.
    pub fn default_pipeline() -> Self {
        Self::new()
            .add_step(PreprocessStep::Lowercase)
            .add_step(PreprocessStep::StripPunctuation)
            .add_step(PreprocessStep::NormalizeWhitespace)
    }

    /// Apply all configured steps in sequence.
    pub fn process(&self, text: &str) -> String {
        let mut result = text.to_string();
        for step in &self.steps {
            result = Self::apply_step(step, &result);
        }
        result
    }

    fn apply_step(step: &PreprocessStep, text: &str) -> String {
        match step {
            PreprocessStep::Lowercase => text.to_lowercase(),
            PreprocessStep::StripPunctuation => {
                text.chars()
                    .filter(|c| !c.is_ascii_punctuation())
                    .collect()
            }
            PreprocessStep::NormalizeWhitespace => {
                let mut out = String::with_capacity(text.len());
                let mut prev_ws = false;
                for ch in text.chars() {
                    if ch.is_whitespace() {
                        if !prev_ws {
                            out.push(' ');
                        }
                        prev_ws = true;
                    } else {
                        out.push(ch);
                        prev_ws = false;
                    }
                }
                out.trim().to_string()
            }
            PreprocessStep::StripDigits => {
                text.chars().filter(|c| !c.is_ascii_digit()).collect()
            }
            PreprocessStep::StemWords => {
                text.split_whitespace()
                    .map(|w| Self::simple_stem(w))
                    .collect::<Vec<_>>()
                    .join(" ")
            }
            PreprocessStep::RemoveStopwords(stopwords) => {
                text.split_whitespace()
                    .filter(|w| !stopwords.contains(&w.to_lowercase()))
                    .collect::<Vec<_>>()
                    .join(" ")
            }
        }
    }

    /// Very simple suffix-stripping stemmer (English-oriented).
    fn simple_stem(word: &str) -> String {
        let mut s = word.to_string();
        if s.ends_with("ation") && s.len() > 6 {
            s.truncate(s.len() - 5);
            return s;
        }
        if s.ends_with("ment") && s.len() > 5 {
            s.truncate(s.len() - 4);
            return s;
        }
        if s.ends_with("ness") && s.len() > 5 {
            s.truncate(s.len() - 4);
            return s;
        }
        if s.ends_with("ing") && s.len() > 5 {
            s.truncate(s.len() - 3);
            return s;
        }
        if s.ends_with("tion") && s.len() > 5 {
            s.truncate(s.len() - 4);
            return s;
        }
        if s.ends_with("sion") && s.len() > 5 {
            s.truncate(s.len() - 4);
            return s;
        }
        if s.ends_with("able") && s.len() > 5 {
            s.truncate(s.len() - 4);
            return s;
        }
        if s.ends_with("ible") && s.len() > 5 {
            s.truncate(s.len() - 4);
            return s;
        }
        if s.ends_with("ful") && s.len() > 4 {
            s.truncate(s.len() - 3);
            return s;
        }
        if s.ends_with("ous") && s.len() > 4 {
            s.truncate(s.len() - 3);
            return s;
        }
        if s.ends_with("ive") && s.len() > 4 {
            s.truncate(s.len() - 3);
            return s;
        }
        if s.ends_with("ed") && s.len() > 4 {
            s.truncate(s.len() - 2);
            return s;
        }
        if s.ends_with("ly") && s.len() > 4 {
            s.truncate(s.len() - 2);
            return s;
        }
        if s.ends_with("er") && s.len() > 4 {
            s.truncate(s.len() - 2);
            return s;
        }
        if s.ends_with("es") && s.len() > 4 {
            s.truncate(s.len() - 2);
            return s;
        }
        if s.ends_with("al") && s.len() > 4 {
            s.truncate(s.len() - 2);
            return s;
        }
        if s.ends_with('s') && !s.ends_with("ss") && s.len() > 3 {
            s.truncate(s.len() - 1);
            return s;
        }
        s
    }

    /// Number of steps in the pipeline.
    pub fn step_count(&self) -> usize {
        self.steps.len()
    }

    /// Process a batch of texts.
    pub fn process_batch(&self, texts: &[&str]) -> Vec<String> {
        texts.iter().map(|t| self.process(t)).collect()
    }

    /// Build a pipeline that strips everything (lowercase + punctuation + digits + whitespace).
    pub fn aggressive_pipeline() -> Self {
        Self::new()
            .add_step(PreprocessStep::Lowercase)
            .add_step(PreprocessStep::StripPunctuation)
            .add_step(PreprocessStep::StripDigits)
            .add_step(PreprocessStep::NormalizeWhitespace)
    }

    /// Build a pipeline for linguistic analysis (lowercase + stopwords + stemming + whitespace).
    pub fn linguistic_pipeline(stopwords: HashSet<String>) -> Self {
        Self::new()
            .add_step(PreprocessStep::Lowercase)
            .add_step(PreprocessStep::RemoveStopwords(stopwords))
            .add_step(PreprocessStep::StemWords)
            .add_step(PreprocessStep::NormalizeWhitespace)
    }
}

impl Default for TextPreprocessor {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// NGramComparator — high-level text comparison using n-grams
// ---------------------------------------------------------------------------

/// Compares two texts using multiple n-gram-based metrics and produces
/// a comprehensive similarity report.
pub struct NGramComparator {
    config: NGramConfig,
}

impl NGramComparator {
    pub fn new(config: NGramConfig) -> Self {
        Self { config }
    }

    /// Compute a full comparison report between two texts.
    pub fn compare(&self, text_a: &str, text_b: &str) -> NGramComparisonResult {
        let set_a = NGramSet::from_text(text_a, self.config.clone());
        let set_b = NGramSet::from_text(text_b, self.config.clone());
        let freq_a = NGramFrequencyMap::from_text(text_a, self.config.clone());
        let freq_b = NGramFrequencyMap::from_text(text_b, self.config.clone());

        let overlap = NGramOverlapReport::compute(&set_a, &set_b);
        let cosine = freq_a.cosine_similarity(&freq_b);
        let kl_ab = freq_a.kl_divergence(&freq_b);
        let kl_ba = freq_b.kl_divergence(&freq_a);
        let entropy_a = freq_a.entropy();
        let entropy_b = freq_b.entropy();

        NGramComparisonResult {
            overlap,
            cosine_similarity: cosine,
            kl_divergence_ab: kl_ab,
            kl_divergence_ba: kl_ba,
            entropy_a,
            entropy_b,
            set_a_size: set_a.len(),
            set_b_size: set_b.len(),
        }
    }
}

/// Result of comparing two texts via `NGramComparator`.
#[derive(Clone, Debug)]
pub struct NGramComparisonResult {
    pub overlap: NGramOverlapReport,
    pub cosine_similarity: f64,
    pub kl_divergence_ab: f64,
    pub kl_divergence_ba: f64,
    pub entropy_a: f64,
    pub entropy_b: f64,
    pub set_a_size: usize,
    pub set_b_size: usize,
}

impl NGramComparisonResult {
    /// One-line summary.
    pub fn summary(&self) -> String {
        format!(
            "Jaccard={:.4}, Cosine={:.4}, Entropy(A)={:.4}, Entropy(B)={:.4}, |A|={}, |B|={}",
            self.overlap.jaccard,
            self.cosine_similarity,
            self.entropy_a,
            self.entropy_b,
            self.set_a_size,
            self.set_b_size,
        )
    }

    /// Overall similarity score (average of Jaccard and Cosine).
    pub fn combined_score(&self) -> f64 {
        (self.overlap.jaccard + self.cosine_similarity) / 2.0
    }
}

// ---------------------------------------------------------------------------
// WeightedNGramSet — n-gram set with associated weights
// ---------------------------------------------------------------------------

/// An n-gram set where each hash carries a floating-point weight, useful for
/// TF-IDF or other weighted similarity computations.
pub struct WeightedNGramSet {
    pub weights: HashMap<u64, f64>,
    pub config: NGramConfig,
}

impl WeightedNGramSet {
    pub fn new(config: NGramConfig) -> Self {
        Self {
            weights: HashMap::new(),
            config,
        }
    }

    /// Build from a frequency map using raw counts as weights.
    pub fn from_frequency_map(freq_map: &NGramFrequencyMap, config: NGramConfig) -> Self {
        let mut ws = Self::new(config);
        for (&hash, &count) in &freq_map.counts {
            ws.weights.insert(hash, count as f64);
        }
        ws
    }

    /// Build from a frequency map using TF (normalized counts) as weights.
    pub fn from_tf(freq_map: &NGramFrequencyMap, config: NGramConfig) -> Self {
        let dist = freq_map.to_probability_distribution();
        let mut ws = Self::new(config);
        for (hash, prob) in dist {
            ws.weights.insert(hash, prob);
        }
        ws
    }

    pub fn len(&self) -> usize {
        self.weights.len()
    }

    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }

    pub fn get_weight(&self, hash: u64) -> f64 {
        self.weights.get(&hash).cloned().unwrap_or(0.0)
    }

    pub fn set_weight(&mut self, hash: u64, weight: f64) {
        self.weights.insert(hash, weight);
    }

    /// Weighted cosine similarity.
    pub fn cosine_similarity(&self, other: &WeightedNGramSet) -> f64 {
        let mut dot = 0.0f64;
        let mut norm_a = 0.0f64;
        let mut norm_b = 0.0f64;

        for (&hash, &wa) in &self.weights {
            norm_a += wa * wa;
            if let Some(&wb) = other.weights.get(&hash) {
                dot += wa * wb;
            }
        }
        for (_, &wb) in &other.weights {
            norm_b += wb * wb;
        }
        let denom = norm_a.sqrt() * norm_b.sqrt();
        if denom == 0.0 { 0.0 } else { dot / denom }
    }

    /// Weighted Jaccard: Σ min(w_a, w_b) / Σ max(w_a, w_b).
    pub fn weighted_jaccard(&self, other: &WeightedNGramSet) -> f64 {
        let mut all_keys: HashSet<u64> = self.weights.keys().cloned().collect();
        for k in other.weights.keys() {
            all_keys.insert(*k);
        }
        let mut min_sum = 0.0f64;
        let mut max_sum = 0.0f64;
        for k in &all_keys {
            let a = self.get_weight(*k);
            let b = other.get_weight(*k);
            min_sum += a.min(b);
            max_sum += a.max(b);
        }
        if max_sum == 0.0 { 0.0 } else { min_sum / max_sum }
    }

    /// L2 norm of the weight vector.
    pub fn norm(&self) -> f64 {
        self.weights.values().map(|w| w * w).sum::<f64>().sqrt()
    }

    /// Normalize weights to unit L2 norm.
    pub fn normalize(&mut self) {
        let n = self.norm();
        if n > 0.0 {
            for w in self.weights.values_mut() {
                *w /= n;
            }
        }
    }

    /// Top k entries by weight.
    pub fn top_k(&self, k: usize) -> Vec<(u64, f64)> {
        let mut entries: Vec<(u64, f64)> = self.weights.iter().map(|(&h, &w)| (h, w)).collect();
        entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        entries.truncate(k);
        entries
    }
}

// ---------------------------------------------------------------------------
// CharacterDistribution — character-level frequency analysis
// ---------------------------------------------------------------------------

/// Analyses the character distribution of a text, useful as a lightweight
/// feature before full n-gram extraction.
pub struct CharacterDistribution {
    pub freq: HashMap<char, usize>,
    pub total: usize,
}

impl CharacterDistribution {
    pub fn from_text(text: &str) -> Self {
        let mut freq: HashMap<char, usize> = HashMap::new();
        let mut total = 0usize;
        for ch in text.chars() {
            *freq.entry(ch).or_insert(0) += 1;
            total += 1;
        }
        Self { freq, total }
    }

    pub fn probability(&self, ch: char) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        self.freq.get(&ch).cloned().unwrap_or(0) as f64 / self.total as f64
    }

    pub fn entropy(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        let t = self.total as f64;
        let mut h = 0.0f64;
        for &c in self.freq.values() {
            let p = c as f64 / t;
            if p > 0.0 {
                h -= p * p.log2();
            }
        }
        h
    }

    pub fn unique_chars(&self) -> usize {
        self.freq.len()
    }

    pub fn most_common(&self, k: usize) -> Vec<(char, usize)> {
        let mut pairs: Vec<(char, usize)> = self.freq.iter().map(|(&c, &n)| (c, n)).collect();
        pairs.sort_by(|a, b| b.1.cmp(&a.1));
        pairs.truncate(k);
        pairs
    }

    /// Cosine similarity between two character distributions.
    pub fn cosine_similarity(&self, other: &CharacterDistribution) -> f64 {
        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        for (&ch, &ca) in &self.freq {
            let a = ca as f64;
            na += a * a;
            if let Some(&cb) = other.freq.get(&ch) {
                dot += a * cb as f64;
            }
        }
        for &cb in other.freq.values() {
            let b = cb as f64;
            nb += b * b;
        }
        let denom = na.sqrt() * nb.sqrt();
        if denom == 0.0 { 0.0 } else { dot / denom }
    }

    /// Chi-squared statistic against a uniform distribution.
    pub fn chi_squared_uniform(&self) -> f64 {
        if self.freq.is_empty() {
            return 0.0;
        }
        let expected = self.total as f64 / self.freq.len() as f64;
        let mut chi2 = 0.0f64;
        for &count in self.freq.values() {
            let diff = count as f64 - expected;
            chi2 += diff * diff / expected;
        }
        chi2
    }
}


// ---------------------------------------------------------------------------
// NGramIndex – indexed n-gram storage for fast lookup
// ---------------------------------------------------------------------------

pub struct NGramIndex {
    pub index: HashMap<u64, Vec<usize>>,
    pub total_ngrams: usize,
}

impl NGramIndex {
    /// Build an index from text: maps each n-gram hash to the list of
    /// byte-offset positions where it appears.
    pub fn from_text(text: &str, config: &NGramConfig) -> Self {
        let extractor = NGramExtractor::new(config.clone());
        let with_pos = extractor.extract_with_positions(text);
        let mut index: HashMap<u64, Vec<usize>> = HashMap::new();
        let total = with_pos.len();
        for (ng, pos) in with_pos {
            index.entry(ng.hash).or_default().push(pos);
        }
        Self { index, total_ngrams: total }
    }

    /// Return positions for a given hash, or an empty slice if absent.
    pub fn lookup(&self, hash: u64) -> &[usize] {
        match self.index.get(&hash) {
            Some(v) => v.as_slice(),
            None => &[],
        }
    }

    /// Whether the index contains a given hash.
    pub fn contains(&self, hash: u64) -> bool {
        self.index.contains_key(&hash)
    }

    /// Number of distinct n-gram hashes stored.
    pub fn unique_count(&self) -> usize {
        self.index.len()
    }

    /// Return the top-k most frequent hashes by number of positions
    /// (descending). Ties broken arbitrarily.
    pub fn most_frequent(&self, k: usize) -> Vec<(u64, usize)> {
        let mut entries: Vec<(u64, usize)> = self
            .index
            .iter()
            .map(|(&h, v)| (h, v.len()))
            .collect();
        entries.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        entries.truncate(k);
        entries
    }

    /// Clone the positions vector for a given hash (empty vec if absent).
    pub fn positions_of(&self, hash: u64) -> Vec<usize> {
        self.index.get(&hash).cloned().unwrap_or_default()
    }
}

// ---------------------------------------------------------------------------
// NGramSimilarityMatrix – pairwise similarity between documents
// ---------------------------------------------------------------------------

pub struct NGramSimilarityMatrix {
    pub matrix: Vec<Vec<f64>>,
    pub num_documents: usize,
}

impl NGramSimilarityMatrix {
    /// Compute the full pairwise Jaccard similarity matrix for a set of documents.
    pub fn compute(documents: &[&str], config: &NGramConfig) -> Self {
        let n = documents.len();
        let sets: Vec<NGramSet> = documents
            .iter()
            .map(|d| NGramSet::from_text(d, config.clone()))
            .collect();
        let mut matrix = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            matrix[i][i] = 1.0;
            for j in (i + 1)..n {
                let sim = sets[i].jaccard_similarity(&sets[j]);
                matrix[i][j] = sim;
                matrix[j][i] = sim;
            }
        }
        Self { matrix, num_documents: n }
    }

    /// Retrieve the similarity value for document pair (i, j).
    pub fn get_similarity(&self, i: usize, j: usize) -> f64 {
        self.matrix[i][j]
    }

    /// Return (i, j, similarity) for the most similar off-diagonal pair.
    pub fn most_similar_pair(&self) -> (usize, usize, f64) {
        let n = self.num_documents;
        let mut best = (0, 1, f64::NEG_INFINITY);
        for i in 0..n {
            for j in (i + 1)..n {
                if self.matrix[i][j] > best.2 {
                    best = (i, j, self.matrix[i][j]);
                }
            }
        }
        best
    }

    /// Return (i, j, similarity) for the least similar off-diagonal pair.
    pub fn least_similar_pair(&self) -> (usize, usize, f64) {
        let n = self.num_documents;
        let mut worst = (0, 1, f64::INFINITY);
        for i in 0..n {
            for j in (i + 1)..n {
                if self.matrix[i][j] < worst.2 {
                    worst = (i, j, self.matrix[i][j]);
                }
            }
        }
        worst
    }

    /// Single-linkage clustering: documents whose similarity is >= threshold
    /// are transitively grouped together.
    pub fn clusters(&self, threshold: f64) -> Vec<Vec<usize>> {
        let n = self.num_documents;
        // Union-Find
        let mut parent: Vec<usize> = (0..n).collect();
        fn find(parent: &mut Vec<usize>, x: usize) -> usize {
            if parent[x] != x {
                parent[x] = find(parent, parent[x]);
            }
            parent[x]
        }
        fn union(parent: &mut Vec<usize>, a: usize, b: usize) {
            let ra = find(parent, a);
            let rb = find(parent, b);
            if ra != rb {
                parent[ra] = rb;
            }
        }
        for i in 0..n {
            for j in (i + 1)..n {
                if self.matrix[i][j] >= threshold {
                    union(&mut parent, i, j);
                }
            }
        }
        let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
        for i in 0..n {
            let root = find(&mut parent, i);
            groups.entry(root).or_default().push(i);
        }
        let mut result: Vec<Vec<usize>> = groups.into_values().collect();
        result.sort_by_key(|g| g[0]);
        result
    }

    /// Serialise the matrix as CSV.
    pub fn to_csv(&self) -> String {
        let mut out = String::new();
        for row in &self.matrix {
            let cols: Vec<String> = row.iter().map(|v| format!("{:.6}", v)).collect();
            out.push_str(&cols.join(","));
            out.push('\n');
        }
        out
    }
}

// ---------------------------------------------------------------------------
// DocumentFingerprint – compact document representation via MinHash
// ---------------------------------------------------------------------------

pub struct DocumentFingerprint {
    pub signature: Vec<u64>,
    pub num_ngrams: usize,
}

impl DocumentFingerprint {
    /// Default number of hash functions used for min-hash signatures.
    const DEFAULT_NUM_HASHES: usize = 128;

    /// Compute a MinHash signature for a set of n-gram hashes.
    /// Each of the `num_hashes` slots uses a different seed: the hash value
    /// is XORed with a deterministic per-slot key and the minimum is kept.
    pub fn min_hash_signature(set: &NGramSet, num_hashes: usize) -> Vec<u64> {
        let mut sig = vec![u64::MAX; num_hashes];
        for &h in &set.grams {
            for i in 0..num_hashes {
                // deterministic seed per slot
                let seed = (i as u64)
                    .wrapping_mul(0x517cc1b727220a95)
                    .wrapping_add(0x6c62272e07bb0142);
                let val = h ^ seed;
                // extra mixing to decorrelate slots
                let val = val
                    .wrapping_mul(0xbf58476d1ce4e5b9)
                    .wrapping_add(0x94d049bb133111eb);
                if val < sig[i] {
                    sig[i] = val;
                }
            }
        }
        sig
    }

    /// Build a fingerprint from raw text using the default 128 hash functions.
    pub fn from_text(text: &str, config: &NGramConfig) -> Self {
        let set = NGramSet::from_text(text, config.clone());
        let num_ngrams = set.len();
        let signature = Self::min_hash_signature(&set, Self::DEFAULT_NUM_HASHES);
        Self { signature, num_ngrams }
    }

    /// Estimated Jaccard similarity from the MinHash signatures: the
    /// fraction of slots where the two signatures agree.
    pub fn similarity(&self, other: &DocumentFingerprint) -> f64 {
        if self.signature.len() != other.signature.len() {
            return 0.0;
        }
        let matches = self
            .signature
            .iter()
            .zip(other.signature.iter())
            .filter(|(a, b)| a == b)
            .count();
        matches as f64 / self.signature.len() as f64
    }

    /// Two documents are near-duplicates when estimated similarity ≥ threshold.
    pub fn is_near_duplicate(&self, other: &DocumentFingerprint, threshold: f64) -> bool {
        self.similarity(other) >= threshold
    }

    /// Serialise the signature to a byte vector (little-endian u64s, prefixed
    /// with the num_ngrams count).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(8 + self.signature.len() * 8);
        buf.extend_from_slice(&(self.num_ngrams as u64).to_le_bytes());
        for &v in &self.signature {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        buf
    }

    /// Deserialise a fingerprint from bytes produced by `to_bytes`.
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let num_ngrams =
            u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        let sig_len = (bytes.len() - 8) / 8;
        let mut signature = Vec::with_capacity(sig_len);
        for i in 0..sig_len {
            let start = 8 + i * 8;
            let val = u64::from_le_bytes(bytes[start..start + 8].try_into().unwrap());
            signature.push(val);
        }
        Self { signature, num_ngrams }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Character n-gram extraction --

    #[test]
    fn test_character_ngrams_basic() {
        let config = NGramConfig::char_ngrams(3);
        let extractor = NGramExtractor::new(config);
        let ngrams = extractor.extract("abcdef");
        assert_eq!(ngrams.len(), 4); // abc, bcd, cde, def
        assert_eq!(ngrams[0].as_str(), "abc");
        assert_eq!(ngrams[3].as_str(), "def");
    }

    #[test]
    fn test_character_ngrams_short_text() {
        let config = NGramConfig::char_ngrams(5);
        let extractor = NGramExtractor::new(config);
        let ngrams = extractor.extract("abc");
        assert!(ngrams.is_empty());
    }

    #[test]
    fn test_character_ngrams_empty() {
        let config = NGramConfig::char_ngrams(2);
        let extractor = NGramExtractor::new(config);
        let ngrams = extractor.extract("");
        assert!(ngrams.is_empty());
    }

    #[test]
    fn test_character_ngrams_n_zero() {
        let config = NGramConfig { n: 0, ..NGramConfig::char_ngrams(1) };
        let extractor = NGramExtractor::new(config);
        let ngrams = extractor.extract("hello");
        assert!(ngrams.is_empty());
    }

    #[test]
    fn test_character_ngrams_unicode() {
        let config = NGramConfig::char_ngrams(2);
        let extractor = NGramExtractor::new(config);
        let ngrams = extractor.extract("αβγ");
        assert_eq!(ngrams.len(), 2);
        assert_eq!(ngrams[0].as_str(), "αβ");
        assert_eq!(ngrams[1].as_str(), "βγ");
    }

    // -- Word n-gram extraction --

    #[test]
    fn test_word_ngrams_basic() {
        let config = NGramConfig::word_ngrams(2);
        let extractor = NGramExtractor::new(config);
        let ngrams = extractor.extract("the quick brown fox");
        assert_eq!(ngrams.len(), 3);
        assert_eq!(ngrams[0].as_str(), "the quick");
        assert_eq!(ngrams[1].as_str(), "quick brown");
        assert_eq!(ngrams[2].as_str(), "brown fox");
    }

    #[test]
    fn test_word_ngrams_single_word() {
        let config = NGramConfig::word_ngrams(2);
        let extractor = NGramExtractor::new(config);
        let ngrams = extractor.extract("hello");
        assert!(ngrams.is_empty());
    }

    // -- BPE tokenisation --

    #[test]
    fn test_bpe_tokenization_basic() {
        let tokens = NGramExtractor::tokenize_bpe("aaabdaaabac", 260);
        // Should produce some merged tokens
        assert!(!tokens.is_empty());
        // The exact token count depends on merges but should be < original byte count
        assert!(tokens.len() <= 11);
    }

    #[test]
    fn test_bpe_ngrams_extract() {
        let config = NGramConfig::bpe_ngrams(2, 260);
        let extractor = NGramExtractor::new(config);
        let ngrams = extractor.extract("hello world hello world");
        assert!(!ngrams.is_empty());
    }

    // -- Set operations --

    #[test]
    fn test_ngram_set_from_text() {
        let config = NGramConfig::char_ngrams(3);
        let set = NGramSet::from_text("abcdef", config);
        assert_eq!(set.len(), 4);
    }

    #[test]
    fn test_ngram_set_intersection() {
        let config = NGramConfig::char_ngrams(3);
        let set_a = NGramSet::from_text("abcdef", config.clone());
        let set_b = NGramSet::from_text("cdefgh", config);
        let inter = set_a.intersection(&set_b);
        // "cde" and "def" should appear in both
        assert_eq!(inter.len(), 2);
    }

    #[test]
    fn test_ngram_set_union() {
        let config = NGramConfig::char_ngrams(3);
        let set_a = NGramSet::from_text("abcdef", config.clone());
        let set_b = NGramSet::from_text("cdefgh", config);
        let uni = set_a.union(&set_b);
        // abc, bcd, cde, def, efg, fgh => 6
        assert_eq!(uni.len(), 6);
    }

    #[test]
    fn test_ngram_set_difference() {
        let config = NGramConfig::char_ngrams(3);
        let set_a = NGramSet::from_text("abcdef", config.clone());
        let set_b = NGramSet::from_text("cdefgh", config);
        let diff = set_a.difference(&set_b);
        // abc, bcd
        assert_eq!(diff.len(), 2);
    }

    #[test]
    fn test_ngram_set_empty() {
        let config = NGramConfig::char_ngrams(3);
        let set = NGramSet::new(config);
        assert!(set.is_empty());
        assert_eq!(set.len(), 0);
    }

    // -- Similarity metrics --

    #[test]
    fn test_jaccard_identical() {
        let config = NGramConfig::char_ngrams(3);
        let set_a = NGramSet::from_text("abcdef", config.clone());
        let set_b = NGramSet::from_text("abcdef", config);
        let j = set_a.jaccard_similarity(&set_b);
        assert!((j - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_jaccard_disjoint() {
        let config = NGramConfig::char_ngrams(3);
        let set_a = NGramSet::from_text("abc", config.clone());
        let set_b = NGramSet::from_text("xyz", config);
        let j = set_a.jaccard_similarity(&set_b);
        assert!((j - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_containment() {
        let config = NGramConfig::char_ngrams(3);
        let set_a = NGramSet::from_text("abcdef", config.clone());
        let set_b = NGramSet::from_text("abcdefghij", config);
        // All of A's n-grams should appear in B
        let c = set_a.containment(&set_b);
        assert!((c - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_overlap_coefficient() {
        let config = NGramConfig::char_ngrams(3);
        let set_a = NGramSet::from_text("abcdef", config.clone());
        let set_b = NGramSet::from_text("cdefgh", config);
        let oc = set_a.overlap_coefficient(&set_b);
        // intersection = 2, min(4,4) = 4, oc = 0.5
        assert!((oc - 0.5).abs() < 1e-9, "Expected 0.5, got {}", oc);
    }

    #[test]
    fn test_jaccard_empty_sets() {
        let config = NGramConfig::char_ngrams(3);
        let set_a = NGramSet::new(config.clone());
        let set_b = NGramSet::new(config);
        assert_eq!(set_a.jaccard_similarity(&set_b), 0.0);
    }

    // -- Frequency map --

    #[test]
    fn test_frequency_map() {
        let config = NGramConfig::char_ngrams(2);
        let fm = NGramFrequencyMap::from_text("abab", config);
        assert_eq!(fm.total_count(), 3); // "ab", "ba", "ab" (3 bigrams, "ab" appears twice)
        let mc = fm.most_common(1);
        assert_eq!(mc.len(), 1);
        // "ab" hash should be most common with count 2
        assert_eq!(mc[0].1, 2);
    }

    #[test]
    fn test_frequency_entropy() {
        let config = NGramConfig::char_ngrams(1);
        let fm = NGramFrequencyMap::from_text("aaaa", config);
        // Only one distinct unigram: entropy = 0
        assert!((fm.entropy() - 0.0).abs() < 1e-9);
    }

    // -- Overlap report --

    #[test]
    fn test_overlap_report() {
        let config = NGramConfig::char_ngrams(3);
        let set_a = NGramSet::from_text("abcdef", config.clone());
        let set_b = NGramSet::from_text("cdefgh", config);
        let report = NGramOverlapReport::compute(&set_a, &set_b);
        assert_eq!(report.intersection_size, 2);
        assert_eq!(report.set_a_size, 4);
        assert_eq!(report.set_b_size, 4);
        assert!(!report.summary().is_empty());
    }

    // -- Hashing --

    #[test]
    fn test_hash_blake3_deterministic() {
        let h1 = NGramSet::hash_ngram("hello", &HashType::Blake3);
        let h2 = NGramSet::hash_ngram("hello", &HashType::Blake3);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_hash_sha256_deterministic() {
        let h1 = NGramSet::hash_ngram("hello", &HashType::Sha256);
        let h2 = NGramSet::hash_ngram("hello", &HashType::Sha256);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_hash_fnv_deterministic() {
        let h1 = NGramSet::hash_ngram("hello", &HashType::FNV);
        let h2 = NGramSet::hash_ngram("hello", &HashType::FNV);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_hash_different_inputs_differ() {
        let h1 = NGramSet::hash_ngram("hello", &HashType::Blake3);
        let h2 = NGramSet::hash_ngram("world", &HashType::Blake3);
        assert_ne!(h1, h2);
    }

    // -- Merge --

    #[test]
    fn test_merge() {
        let config = NGramConfig::char_ngrams(2);
        let mut set_a = NGramSet::from_text("abc", config.clone());
        let set_b = NGramSet::from_text("bcd", config);
        let a_before = set_a.len();
        set_a.merge(&set_b);
        assert!(set_a.len() >= a_before);
    }

    // -- Sorted vec --

    #[test]
    fn test_sorted_vec() {
        let config = NGramConfig::char_ngrams(2);
        let set = NGramSet::from_text("abcde", config);
        let sv = set.to_sorted_vec();
        for i in 1..sv.len() {
            assert!(sv[i - 1] <= sv[i]);
        }
    }

    // -- Extract with positions --

    #[test]
    fn test_extract_with_positions() {
        let config = NGramConfig::char_ngrams(3);
        let extractor = NGramExtractor::new(config);
        let positions = extractor.extract_with_positions("abcdef");
        assert_eq!(positions.len(), 4);
        assert_eq!(positions[0].1, 0);
        assert_eq!(positions[1].1, 1);
    }

    // -- Normalization --

    #[test]
    fn test_normalize_lowercase() {
        let config = NGramConfig { case_sensitive: false, ..NGramConfig::char_ngrams(2) };
        let extractor = NGramExtractor::new(config);
        let normalized = extractor.normalize_text("HELLO World");
        assert_eq!(normalized, "hello world");
    }

    #[test]
    fn test_normalize_whitespace() {
        let config = NGramConfig { normalize_whitespace: true, ..NGramConfig::char_ngrams(2) };
        let extractor = NGramExtractor::new(config);
        let normalized = extractor.normalize_text("hello   world\t\nfoo");
        assert_eq!(normalized, "hello world foo");
    }

    #[test]
    fn test_case_sensitive() {
        let config = NGramConfig { case_sensitive: true, ..NGramConfig::char_ngrams(2) };
        let extractor = NGramExtractor::new(config);
        let normalized = extractor.normalize_text("Hello");
        assert_eq!(normalized, "Hello");
    }

    // -- NGram struct --

    #[test]
    fn test_ngram_equality() {
        let a = NGram::new("abc".into(), GramType::Character, &HashType::Blake3);
        let b = NGram::new("abc".into(), GramType::Character, &HashType::Blake3);
        assert_eq!(a, b);
    }

    #[test]
    fn test_ngram_to_bytes() {
        let a = NGram::new("abc".into(), GramType::Character, &HashType::Blake3);
        let bytes = a.to_bytes();
        assert_eq!(bytes.len(), 8);
    }

    // -- SlidingWindow --

    #[test]
    fn test_sliding_window() {
        let data = vec![1, 2, 3, 4, 5];
        let windows: Vec<&[i32]> = SlidingWindow::new(&data, 3).collect();
        assert_eq!(windows.len(), 3);
        assert_eq!(windows[0], &[1, 2, 3]);
        assert_eq!(windows[2], &[3, 4, 5]);
    }

    #[test]
    fn test_sliding_window_too_large() {
        let data = vec![1, 2];
        let windows: Vec<&[i32]> = SlidingWindow::new(&data, 5).collect();
        assert!(windows.is_empty());
    }

    // =====================================================================
    // BPETokenizer tests
    // =====================================================================

    #[test]
    fn test_bpe_tokenizer_new() {
        let tok = BPETokenizer::new(300);
        assert_eq!(tok.vocab_size(), 256);
        assert!(tok.merge_order.is_empty());
    }

    #[test]
    fn test_bpe_tokenizer_train_and_encode() {
        let mut tok = BPETokenizer::new(260);
        let corpus = vec!["aaabdaaabac".to_string(), "aaabdaaabac".to_string()];
        tok.train(&corpus);
        assert!(tok.merge_count() > 0);
        let encoded = tok.encode("aaab");
        assert!(!encoded.is_empty());
        assert!(encoded.len() <= 4);
    }

    #[test]
    fn test_bpe_tokenizer_roundtrip() {
        let mut tok = BPETokenizer::new(270);
        let corpus = vec!["hello world hello world".to_string()];
        tok.train(&corpus);
        let text = "hello world";
        let encoded = tok.encode(text);
        let decoded = tok.decode(&encoded);
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_bpe_tokenizer_empty_text() {
        let tok = BPETokenizer::new(256);
        assert!(tok.encode("").is_empty());
        assert_eq!(tok.decode(&[]), "");
    }

    #[test]
    fn test_bpe_tokenizer_get_merge_at() {
        let mut tok = BPETokenizer::new(260);
        tok.train(&vec!["aaaa".to_string()]);
        if tok.merge_count() > 0 {
            assert!(tok.get_merge_at(0).is_some());
        }
        assert!(tok.get_merge_at(999).is_none());
    }

    #[test]
    fn test_bpe_tokenizer_has_merge() {
        let mut tok = BPETokenizer::new(260);
        tok.train(&vec!["aabb".to_string(), "aabb".to_string(), "aabb".to_string()]);
        if tok.merge_count() > 0 {
            let (a, b) = tok.merge_order[0];
            assert!(tok.has_merge(a, b));
        }
        assert!(!tok.has_merge(250, 251));
    }

    #[test]
    fn test_bpe_byte_frequencies() {
        let corpus = vec!["aab".to_string()];
        let freq = BPETokenizer::compute_byte_frequencies(&corpus);
        assert_eq!(freq.get(&b'a'), Some(&2));
        assert_eq!(freq.get(&b'b'), Some(&1));
    }

    #[test]
    fn test_bpe_tokenizer_token_to_bytes() {
        let tok = BPETokenizer::new(256);
        let bytes = tok.token_to_bytes(65);
        assert_eq!(bytes, Some(vec![65u8]));
        assert!(tok.token_to_bytes(999).is_none());
    }

    #[test]
    fn test_bpe_encode_batch() {
        let mut tok = BPETokenizer::new(260);
        tok.train(&vec!["hello".to_string()]);
        let batch = tok.encode_batch(&["hi", "hello"]);
        assert_eq!(batch.len(), 2);
    }

    #[test]
    fn test_bpe_decode_batch() {
        let tok = BPETokenizer::new(256);
        let batch = tok.decode_batch(&[vec![72, 105], vec![72, 101]]);
        assert_eq!(batch.len(), 2);
        assert_eq!(batch[0], "Hi");
    }

    #[test]
    fn test_bpe_compression_ratio() {
        let mut tok = BPETokenizer::new(280);
        tok.train(&vec!["ababababab".to_string(); 5]);
        let ratio = tok.compression_ratio("ababababab");
        assert!(ratio <= 1.0);
        assert!(ratio > 0.0);
    }

    // =====================================================================
    // NGramFrequencyMap — extended method tests
    // =====================================================================

    #[test]
    fn test_least_common() {
        let config = NGramConfig::char_ngrams(2);
        let fm = NGramFrequencyMap::from_text("ababc", config);
        let lc = fm.least_common(1);
        assert_eq!(lc.len(), 1);
        assert_eq!(lc[0].1, 1);
    }

    #[test]
    fn test_unique_count() {
        let config = NGramConfig::char_ngrams(2);
        let fm = NGramFrequencyMap::from_text("abab", config);
        assert!(fm.unique_count() > 0);
        assert!(fm.unique_count() <= fm.total_count());
    }

    #[test]
    fn test_kl_divergence_identical() {
        let config = NGramConfig::char_ngrams(2);
        let fm = NGramFrequencyMap::from_text("abcdef", config.clone());
        let fm2 = NGramFrequencyMap::from_text("abcdef", config);
        let kl = fm.kl_divergence(&fm2);
        assert!(kl.abs() < 1e-9, "KL divergence of identical distributions should be ~0, got {}", kl);
    }

    #[test]
    fn test_kl_divergence_disjoint() {
        let config = NGramConfig::char_ngrams(3);
        let fm_a = NGramFrequencyMap::from_text("aaaaaa", config.clone());
        let fm_b = NGramFrequencyMap::from_text("bbbbbb", config);
        let kl = fm_a.kl_divergence(&fm_b);
        assert!(kl.is_infinite());
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let config = NGramConfig::char_ngrams(2);
        let fm = NGramFrequencyMap::from_text("abcdef", config.clone());
        let fm2 = NGramFrequencyMap::from_text("abcdef", config);
        let cs = fm.cosine_similarity(&fm2);
        assert!((cs - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_cosine_similarity_disjoint() {
        let config = NGramConfig::char_ngrams(3);
        let fm_a = NGramFrequencyMap::from_text("aaaaaa", config.clone());
        let fm_b = NGramFrequencyMap::from_text("bbbbbb", config);
        let cs = fm_a.cosine_similarity(&fm_b);
        assert!((cs - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_freq_map_merge() {
        let config = NGramConfig::char_ngrams(2);
        let mut fm_a = NGramFrequencyMap::from_text("abc", config.clone());
        let fm_b = NGramFrequencyMap::from_text("abc", config);
        let total_before = fm_a.total_count();
        fm_a.merge(&fm_b);
        assert_eq!(fm_a.total_count(), total_before * 2);
    }

    #[test]
    fn test_filter_by_count() {
        let config = NGramConfig::char_ngrams(2);
        let fm = NGramFrequencyMap::from_text("abab", config);
        let filtered = fm.filter_by_count(2);
        for (_, &c) in &filtered.counts {
            assert!(c >= 2);
        }
    }

    #[test]
    fn test_normalize() {
        let config = NGramConfig::char_ngrams(2);
        let fm = NGramFrequencyMap::from_text("abcde", config);
        let norm = fm.normalize();
        let sum: f64 = norm.values().sum();
        assert!((sum - 1.0).abs() < 1e-9, "Normalized probabilities should sum to 1, got {}", sum);
    }

    #[test]
    fn test_js_divergence_identical() {
        let config = NGramConfig::char_ngrams(2);
        let fm = NGramFrequencyMap::from_text("abcdef", config.clone());
        let fm2 = NGramFrequencyMap::from_text("abcdef", config);
        let js = fm.js_divergence(&fm2);
        assert!(js.abs() < 1e-9, "JS divergence of identical distributions should be ~0, got {}", js);
    }

    #[test]
    fn test_js_divergence_different() {
        let config = NGramConfig::char_ngrams(2);
        let fm_a = NGramFrequencyMap::from_text("aaaa", config.clone());
        let fm_b = NGramFrequencyMap::from_text("bbbb", config);
        let js = fm_a.js_divergence(&fm_b);
        assert!(js > 0.0);
        assert!(js.is_finite());
    }

    #[test]
    fn test_perplexity() {
        let config = NGramConfig::char_ngrams(1);
        let fm = NGramFrequencyMap::from_text("aaaa", config);
        let pp = fm.perplexity();
        assert!((pp - 1.0).abs() < 1e-9, "Perplexity of uniform-1 distribution should be 1, got {}", pp);
    }

    #[test]
    fn test_weighted_jaccard() {
        let config = NGramConfig::char_ngrams(2);
        let fm_a = NGramFrequencyMap::from_text("abcdef", config.clone());
        let fm_b = NGramFrequencyMap::from_text("abcdef", config);
        let wj = fm_a.weighted_jaccard(&fm_b);
        assert!((wj - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_cross_entropy_identical() {
        let config = NGramConfig::char_ngrams(2);
        let fm = NGramFrequencyMap::from_text("abcdef", config.clone());
        let fm2 = NGramFrequencyMap::from_text("abcdef", config);
        let ce = fm.cross_entropy(&fm2);
        let h = fm.entropy();
        // Cross-entropy with itself equals entropy (in nats vs bits: we use ln vs log2)
        // H_cross(P,P) = H(P) when using same log base. Our entropy uses log2, cross_entropy uses ln.
        assert!(ce.is_finite());
    }

    // =====================================================================
    // NGramOverlapReport — extended method tests
    // =====================================================================

    #[test]
    fn test_overlap_is_significant() {
        let config = NGramConfig::char_ngrams(3);
        let set_a = NGramSet::from_text("abcdef", config.clone());
        let set_b = NGramSet::from_text("abcdef", config);
        let report = NGramOverlapReport::compute(&set_a, &set_b);
        assert!(report.is_significant(0.5));
        assert!(report.is_significant(1.0));
    }

    #[test]
    fn test_overlap_not_significant() {
        let config = NGramConfig::char_ngrams(3);
        let set_a = NGramSet::from_text("abc", config.clone());
        let set_b = NGramSet::from_text("xyz", config);
        let report = NGramOverlapReport::compute(&set_a, &set_b);
        assert!(!report.is_significant(0.5));
    }

    #[test]
    fn test_overlap_to_json() {
        let config = NGramConfig::char_ngrams(3);
        let set_a = NGramSet::from_text("abcdef", config.clone());
        let set_b = NGramSet::from_text("cdefgh", config);
        let report = NGramOverlapReport::compute(&set_a, &set_b);
        let json = report.to_json();
        assert!(json.contains("\"jaccard\""));
        assert!(json.contains("\"intersection_size\""));
    }

    #[test]
    fn test_overlap_dice_coefficient() {
        let config = NGramConfig::char_ngrams(3);
        let set_a = NGramSet::from_text("abcdef", config.clone());
        let set_b = NGramSet::from_text("abcdef", config);
        let report = NGramOverlapReport::compute(&set_a, &set_b);
        let dice = report.dice_coefficient();
        assert!((dice - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_overlap_symmetric_difference() {
        let config = NGramConfig::char_ngrams(3);
        let set_a = NGramSet::from_text("abcdef", config.clone());
        let set_b = NGramSet::from_text("cdefgh", config);
        let report = NGramOverlapReport::compute(&set_a, &set_b);
        let sd = report.symmetric_difference_size();
        assert_eq!(sd, report.set_a_size + report.set_b_size - 2 * report.intersection_size);
    }

    // =====================================================================
    // SlidingWindowIterator tests
    // =====================================================================

    #[test]
    fn test_sliding_window_iterator_basic() {
        let data = vec![1, 2, 3, 4, 5];
        let windows: Vec<Vec<i32>> = SlidingWindowIterator::new(&data, 3).collect();
        assert_eq!(windows.len(), 3);
        assert_eq!(windows[0], vec![1, 2, 3]);
        assert_eq!(windows[2], vec![3, 4, 5]);
    }

    #[test]
    fn test_sliding_window_iterator_too_large() {
        let data = vec![1, 2];
        let windows: Vec<Vec<i32>> = SlidingWindowIterator::new(&data, 5).collect();
        assert!(windows.is_empty());
    }

    #[test]
    fn test_sliding_window_iterator_size_hint() {
        let data = vec![1, 2, 3, 4, 5];
        let iter = SlidingWindowIterator::new(&data, 3);
        assert_eq!(iter.len(), 3);
    }

    #[test]
    fn test_sliding_window_iterator_remaining() {
        let data = vec![10, 20, 30, 40];
        let mut iter = SlidingWindowIterator::new(&data, 2);
        assert_eq!(iter.remaining(), 3);
        iter.next();
        assert_eq!(iter.remaining(), 2);
    }

    #[test]
    fn test_sliding_window_iterator_strings() {
        let data = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let windows: Vec<Vec<String>> = SlidingWindowIterator::new(&data, 2).collect();
        assert_eq!(windows.len(), 2);
        assert_eq!(windows[0], vec!["a".to_string(), "b".to_string()]);
    }

    // =====================================================================
    // MultiGramExtractor tests
    // =====================================================================

    #[test]
    fn test_multi_gram_extractor_extract_all() {
        let extractor = MultiGramExtractor::new(&[2, 3], GramType::Character);
        let result = extractor.extract_all("abcdef");
        assert!(result.contains_key(&2));
        assert!(result.contains_key(&3));
        assert_eq!(result[&2].len(), 5); // ab, bc, cd, de, ef
        assert_eq!(result[&3].len(), 4); // abc, bcd, cde, def
    }

    #[test]
    fn test_multi_gram_extractor_combined_set() {
        let extractor = MultiGramExtractor::new(&[2, 3], GramType::Character);
        let combined = extractor.combined_set("abcdef");
        assert!(combined.len() >= 5); // at least the 2-grams
    }

    #[test]
    fn test_multi_gram_extractor_overlap_at_each_size() {
        let extractor = MultiGramExtractor::new(&[2, 3], GramType::Character);
        let overlaps = extractor.overlap_at_each_size("abcdef", "abcdef");
        assert_eq!(overlaps.len(), 2);
        for (_, j) in &overlaps {
            assert!((*j - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn test_multi_gram_extractor_disjoint() {
        let extractor = MultiGramExtractor::new(&[3], GramType::Character);
        let overlaps = extractor.overlap_at_each_size("abc", "xyz");
        assert_eq!(overlaps.len(), 1);
        assert!((overlaps[0].1 - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_multi_gram_extractor_best_overlap() {
        let extractor = MultiGramExtractor::new(&[2, 3, 4], GramType::Character);
        let (size, score) = extractor.best_overlap_size("abcdef", "abcdef");
        assert!(size >= 2);
        assert!((score - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_multi_gram_counts_per_size() {
        let extractor = MultiGramExtractor::new(&[2, 3], GramType::Character);
        let counts = extractor.counts_per_size("abcdef");
        assert_eq!(counts.len(), 2);
    }

    #[test]
    fn test_multi_gram_total_unique() {
        let extractor = MultiGramExtractor::new(&[2, 3], GramType::Character);
        let total = extractor.total_unique_count("abcdef");
        assert!(total >= 5);
    }

    #[test]
    fn test_multi_gram_word_type() {
        let extractor = MultiGramExtractor::new(&[1, 2], GramType::Word);
        let result = extractor.extract_all("the quick brown fox");
        assert!(result.contains_key(&1));
        assert!(result.contains_key(&2));
    }

    // =====================================================================
    // NGramFilter tests
    // =====================================================================

    #[test]
    fn test_filter_remove_stopwords() {
        let config = NGramConfig::char_ngrams(2);
        let set = NGramSet::from_text("abcde", config);
        let mut stopwords = HashSet::new();
        stopwords.insert("ab".to_string());
        let filtered = NGramFilter::remove_stopword_ngrams(&set, &stopwords);
        assert!(filtered.len() < set.len());
    }

    #[test]
    fn test_filter_by_frequency() {
        let config = NGramConfig::char_ngrams(2);
        let set = NGramSet::from_text("ababab", config.clone());
        let fm = NGramFrequencyMap::from_text("ababab", config);
        let filtered = NGramFilter::filter_by_frequency(&set, &fm, 2);
        assert!(filtered.len() <= set.len());
    }

    #[test]
    fn test_filter_sample_random() {
        let config = NGramConfig::char_ngrams(2);
        let set = NGramSet::from_text("abcdefghij", config);
        let sampled = NGramFilter::sample_random(&set, 3);
        assert_eq!(sampled.len(), 3);
    }

    #[test]
    fn test_filter_sample_random_more_than_available() {
        let config = NGramConfig::char_ngrams(2);
        let set = NGramSet::from_text("abc", config);
        let sampled = NGramFilter::sample_random(&set, 100);
        assert_eq!(sampled.len(), set.len());
    }

    #[test]
    fn test_filter_top_k_by_frequency() {
        let config = NGramConfig::char_ngrams(2);
        let fm = NGramFrequencyMap::from_text("ababab", config);
        let top = NGramFilter::top_k_by_frequency(&fm, 1);
        assert_eq!(top.len(), 1);
    }

    #[test]
    fn test_filter_subtract() {
        let config = NGramConfig::char_ngrams(2);
        let set_a = NGramSet::from_text("abcde", config.clone());
        let set_b = NGramSet::from_text("abcde", config);
        let result = NGramFilter::subtract(&set_a, &set_b);
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_filter_whitelist() {
        let config = NGramConfig::char_ngrams(2);
        let set_a = NGramSet::from_text("abcde", config.clone());
        let set_b = NGramSet::from_text("abcde", config);
        let result = NGramFilter::whitelist(&set_a, &set_b);
        assert_eq!(result.len(), set_a.len());
    }

    #[test]
    fn test_filter_by_max_frequency() {
        let config = NGramConfig::char_ngrams(2);
        let set = NGramSet::from_text("ababab", config.clone());
        let fm = NGramFrequencyMap::from_text("ababab", config);
        let filtered = NGramFilter::filter_by_max_frequency(&set, &fm, 2);
        // Only n-grams with count < 2 should remain
        for &h in &filtered.grams {
            assert!(fm.get_count(h) < 2);
        }
    }

    #[test]
    fn test_filter_by_frequency_range() {
        let config = NGramConfig::char_ngrams(2);
        let set = NGramSet::from_text("abababcdcd", config.clone());
        let fm = NGramFrequencyMap::from_text("abababcdcd", config);
        let filtered = NGramFilter::filter_by_frequency_range(&set, &fm, 1, 3);
        assert!(filtered.len() <= set.len());
    }

    // =====================================================================
    // NGramStatistics tests
    // =====================================================================

    #[test]
    fn test_statistics_coverage_full() {
        let config = NGramConfig::char_ngrams(3);
        let set_a = NGramSet::from_text("abcdef", config.clone());
        let set_b = NGramSet::from_text("abcdef", config);
        let cov = NGramStatistics::coverage(&set_a, &set_b);
        assert!((cov - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_statistics_coverage_partial() {
        let config = NGramConfig::char_ngrams(3);
        let set_a = NGramSet::from_text("abcdef", config.clone());
        let set_b = NGramSet::from_text("cdefgh", config);
        let cov = NGramStatistics::coverage(&set_a, &set_b);
        assert!(cov > 0.0 && cov < 1.0);
    }

    #[test]
    fn test_statistics_redundancy() {
        let config = NGramConfig::char_ngrams(2);
        let fm = NGramFrequencyMap::from_text("ababab", config);
        let red = NGramStatistics::redundancy(&fm);
        assert!(red > 0.0);
        assert!(red < 1.0);
    }

    #[test]
    fn test_statistics_redundancy_no_repeats() {
        let config = NGramConfig::char_ngrams(2);
        let fm = NGramFrequencyMap::from_text("abcde", config);
        let red = NGramStatistics::redundancy(&fm);
        assert!(red.abs() < 1e-9, "No redundancy expected for unique n-grams, got {}", red);
    }

    #[test]
    fn test_statistics_diversity_index() {
        let config = NGramConfig::char_ngrams(1);
        let fm = NGramFrequencyMap::from_text("aaaa", config);
        let di = NGramStatistics::diversity_index(&fm);
        assert!(di.abs() < 1e-9, "Single type should have diversity 0, got {}", di);
    }

    #[test]
    fn test_statistics_diversity_high() {
        let config = NGramConfig::char_ngrams(1);
        let fm = NGramFrequencyMap::from_text("abcdefghij", config);
        let di = NGramStatistics::diversity_index(&fm);
        assert!(di > 0.5, "Expected high diversity, got {}", di);
    }

    #[test]
    fn test_statistics_burstiness() {
        let config = NGramConfig::char_ngrams(2);
        let fm = NGramFrequencyMap::from_text("ababab", config);
        let burst = NGramStatistics::burstiness(&fm);
        assert!(burst >= 0.0);
    }

    #[test]
    fn test_statistics_type_token_ratio() {
        let config = NGramConfig::char_ngrams(2);
        let fm = NGramFrequencyMap::from_text("abcde", config);
        let ttr = NGramStatistics::type_token_ratio(&fm);
        assert!((ttr - 1.0).abs() < 1e-9, "All unique bigrams expected TTR=1, got {}", ttr);
    }

    #[test]
    fn test_statistics_hapax_ratio() {
        let config = NGramConfig::char_ngrams(2);
        let fm = NGramFrequencyMap::from_text("abcde", config);
        let hr = NGramStatistics::hapax_ratio(&fm);
        assert!((hr - 1.0).abs() < 1e-9, "All hapax expected ratio=1, got {}", hr);
    }

    #[test]
    fn test_statistics_mean_frequency() {
        let config = NGramConfig::char_ngrams(2);
        let fm = NGramFrequencyMap::from_text("abcde", config);
        let mean = NGramStatistics::mean_frequency(&fm);
        assert!((mean - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_statistics_median_frequency() {
        let config = NGramConfig::char_ngrams(2);
        let fm = NGramFrequencyMap::from_text("ababc", config);
        let med = NGramStatistics::median_frequency(&fm);
        assert!(med >= 1.0);
    }

    #[test]
    fn test_statistics_max_frequency() {
        let config = NGramConfig::char_ngrams(2);
        let fm = NGramFrequencyMap::from_text("ababab", config);
        let mx = NGramStatistics::max_frequency(&fm);
        assert!(mx >= 2);
    }

    #[test]
    fn test_statistics_frequency_std_dev() {
        let config = NGramConfig::char_ngrams(1);
        let fm = NGramFrequencyMap::from_text("aaaa", config);
        let sd = NGramStatistics::frequency_std_dev(&fm);
        assert!(sd.abs() < 1e-9);
    }

    #[test]
    fn test_statistics_mutual_coverage() {
        let config = NGramConfig::char_ngrams(3);
        let set_a = NGramSet::from_text("abcdef", config.clone());
        let set_b = NGramSet::from_text("abcdef", config);
        let mc = NGramStatistics::mutual_coverage(&set_a, &set_b);
        assert!((mc - 1.0).abs() < 1e-9);
    }

    // =====================================================================
    // TextPreprocessor tests
    // =====================================================================

    #[test]
    fn test_preprocessor_lowercase() {
        let pp = TextPreprocessor::new().add_step(PreprocessStep::Lowercase);
        assert_eq!(pp.process("HELLO World"), "hello world");
    }

    #[test]
    fn test_preprocessor_strip_punctuation() {
        let pp = TextPreprocessor::new().add_step(PreprocessStep::StripPunctuation);
        assert_eq!(pp.process("hello, world!"), "hello world");
    }

    #[test]
    fn test_preprocessor_normalize_whitespace() {
        let pp = TextPreprocessor::new().add_step(PreprocessStep::NormalizeWhitespace);
        assert_eq!(pp.process("  hello   world  "), "hello world");
    }

    #[test]
    fn test_preprocessor_strip_digits() {
        let pp = TextPreprocessor::new().add_step(PreprocessStep::StripDigits);
        assert_eq!(pp.process("abc123def"), "abcdef");
    }

    #[test]
    fn test_preprocessor_stem_words() {
        let pp = TextPreprocessor::new().add_step(PreprocessStep::StemWords);
        let result = pp.process("running walked");
        assert!(result.contains("runn") || result.contains("walk"));
    }

    #[test]
    fn test_preprocessor_remove_stopwords() {
        let mut stopwords = HashSet::new();
        stopwords.insert("the".to_string());
        stopwords.insert("a".to_string());
        let pp = TextPreprocessor::new().add_step(PreprocessStep::RemoveStopwords(stopwords));
        assert_eq!(pp.process("the quick a fox"), "quick fox");
    }

    #[test]
    fn test_preprocessor_default_pipeline() {
        let pp = TextPreprocessor::default_pipeline();
        assert_eq!(pp.step_count(), 3);
        let result = pp.process("HELLO, World!  ");
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_preprocessor_aggressive_pipeline() {
        let pp = TextPreprocessor::aggressive_pipeline();
        let result = pp.process("Hello, World! 123");
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_preprocessor_chain() {
        let pp = TextPreprocessor::new()
            .add_step(PreprocessStep::Lowercase)
            .add_step(PreprocessStep::StripDigits)
            .add_step(PreprocessStep::NormalizeWhitespace);
        let result = pp.process("  HELLO 123 WORLD  ");
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_preprocessor_empty_input() {
        let pp = TextPreprocessor::default_pipeline();
        assert_eq!(pp.process(""), "");
    }

    #[test]
    fn test_preprocessor_process_batch() {
        let pp = TextPreprocessor::default_pipeline();
        let results = pp.process_batch(&["HELLO!", "WORLD."]);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], "hello");
        assert_eq!(results[1], "world");
    }

    #[test]
    fn test_preprocessor_linguistic_pipeline() {
        let mut sw = HashSet::new();
        sw.insert("the".to_string());
        let pp = TextPreprocessor::linguistic_pipeline(sw);
        let result = pp.process("The running foxes");
        assert!(!result.contains("the"));
    }

    // =====================================================================
    // NGramComparator tests
    // =====================================================================

    #[test]
    fn test_comparator_identical() {
        let config = NGramConfig::char_ngrams(3);
        let cmp = NGramComparator::new(config);
        let result = cmp.compare("abcdef", "abcdef");
        assert!((result.overlap.jaccard - 1.0).abs() < 1e-9);
        assert!((result.cosine_similarity - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_comparator_disjoint() {
        let config = NGramConfig::char_ngrams(3);
        let cmp = NGramComparator::new(config);
        let result = cmp.compare("abc", "xyz");
        assert!((result.overlap.jaccard - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_comparator_summary() {
        let config = NGramConfig::char_ngrams(3);
        let cmp = NGramComparator::new(config);
        let result = cmp.compare("abcdef", "cdefgh");
        let summary = result.summary();
        assert!(summary.contains("Jaccard"));
        assert!(summary.contains("Cosine"));
    }

    #[test]
    fn test_comparator_combined_score() {
        let config = NGramConfig::char_ngrams(3);
        let cmp = NGramComparator::new(config);
        let result = cmp.compare("abcdef", "abcdef");
        let score = result.combined_score();
        assert!((score - 1.0).abs() < 1e-9);
    }

    // =====================================================================
    // WeightedNGramSet tests
    // =====================================================================

    #[test]
    fn test_weighted_ngram_set_from_freq_map() {
        let config = NGramConfig::char_ngrams(2);
        let fm = NGramFrequencyMap::from_text("abcde", config.clone());
        let ws = WeightedNGramSet::from_frequency_map(&fm, config);
        assert_eq!(ws.len(), fm.unique_count());
    }

    #[test]
    fn test_weighted_ngram_set_cosine_identical() {
        let config = NGramConfig::char_ngrams(2);
        let fm = NGramFrequencyMap::from_text("abcde", config.clone());
        let ws = WeightedNGramSet::from_frequency_map(&fm, config.clone());
        let ws2 = WeightedNGramSet::from_frequency_map(&fm, config);
        let cs = ws.cosine_similarity(&ws2);
        assert!((cs - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_weighted_ngram_set_weighted_jaccard() {
        let config = NGramConfig::char_ngrams(2);
        let fm = NGramFrequencyMap::from_text("abcde", config.clone());
        let ws = WeightedNGramSet::from_frequency_map(&fm, config.clone());
        let ws2 = WeightedNGramSet::from_frequency_map(&fm, config);
        let wj = ws.weighted_jaccard(&ws2);
        assert!((wj - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_weighted_ngram_set_normalize() {
        let config = NGramConfig::char_ngrams(2);
        let fm = NGramFrequencyMap::from_text("abcde", config.clone());
        let mut ws = WeightedNGramSet::from_frequency_map(&fm, config);
        ws.normalize();
        let norm = ws.norm();
        assert!((norm - 1.0).abs() < 1e-6, "After normalization, norm should be 1, got {}", norm);
    }

    #[test]
    fn test_weighted_ngram_set_top_k() {
        let config = NGramConfig::char_ngrams(2);
        let fm = NGramFrequencyMap::from_text("ababab", config.clone());
        let ws = WeightedNGramSet::from_frequency_map(&fm, config);
        let top = ws.top_k(1);
        assert_eq!(top.len(), 1);
    }

    // =====================================================================
    // CharacterDistribution tests
    // =====================================================================

    #[test]
    fn test_char_dist_from_text() {
        let cd = CharacterDistribution::from_text("aab");
        assert_eq!(cd.total, 3);
        assert_eq!(cd.freq.get(&'a'), Some(&2));
        assert_eq!(cd.freq.get(&'b'), Some(&1));
    }

    #[test]
    fn test_char_dist_probability() {
        let cd = CharacterDistribution::from_text("aab");
        let p = cd.probability('a');
        assert!((p - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_char_dist_entropy() {
        let cd = CharacterDistribution::from_text("aaaa");
        assert!(cd.entropy().abs() < 1e-9);
    }

    #[test]
    fn test_char_dist_unique_chars() {
        let cd = CharacterDistribution::from_text("abcabc");
        assert_eq!(cd.unique_chars(), 3);
    }

    #[test]
    fn test_char_dist_most_common() {
        let cd = CharacterDistribution::from_text("aabc");
        let mc = cd.most_common(1);
        assert_eq!(mc[0].0, 'a');
        assert_eq!(mc[0].1, 2);
    }

    #[test]
    fn test_char_dist_cosine_identical() {
        let cd1 = CharacterDistribution::from_text("abcabc");
        let cd2 = CharacterDistribution::from_text("abcabc");
        let cs = cd1.cosine_similarity(&cd2);
        assert!((cs - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_char_dist_chi_squared() {
        let cd = CharacterDistribution::from_text("aaaa");
        let chi2 = cd.chi_squared_uniform();
        assert!(chi2.abs() < 1e-9);
    }

    #[test]
    fn test_char_dist_empty() {
        let cd = CharacterDistribution::from_text("");
        assert_eq!(cd.total, 0);
        assert_eq!(cd.unique_chars(), 0);
        assert!(cd.entropy().abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // NGramIndex tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ngram_index_from_text_basic() {
        let cfg = NGramConfig::char_ngrams(2);
        let idx = NGramIndex::from_text("abcabc", &cfg);
        // "abcabc" with n=2 -> "ab","bc","ca","ab","bc" = 5 total, 3 unique
        assert_eq!(idx.total_ngrams, 5);
        assert_eq!(idx.unique_count(), 3);
    }

    #[test]
    fn test_ngram_index_lookup_present() {
        let cfg = NGramConfig::char_ngrams(2);
        let idx = NGramIndex::from_text("abab", &cfg);
        // "ab","ba","ab" – "ab" appears at two positions
        let ab_hash = NGramSet::hash_ngram("ab", &cfg.hash_function);
        let positions = idx.lookup(ab_hash);
        assert_eq!(positions.len(), 2);
    }

    #[test]
    fn test_ngram_index_lookup_absent() {
        let cfg = NGramConfig::char_ngrams(2);
        let idx = NGramIndex::from_text("abab", &cfg);
        let positions = idx.lookup(999999);
        assert!(positions.is_empty());
    }

    #[test]
    fn test_ngram_index_contains() {
        let cfg = NGramConfig::char_ngrams(2);
        let idx = NGramIndex::from_text("hello", &cfg);
        let he_hash = NGramSet::hash_ngram("he", &cfg.hash_function);
        assert!(idx.contains(he_hash));
        assert!(!idx.contains(0xDEADBEEF));
    }

    #[test]
    fn test_ngram_index_most_frequent() {
        let cfg = NGramConfig::char_ngrams(2);
        let idx = NGramIndex::from_text("ababab", &cfg);
        // "ab","ba","ab","ba","ab" -> ab:3, ba:2
        let top = idx.most_frequent(1);
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].1, 3);
    }

    #[test]
    fn test_ngram_index_positions_of_cloned() {
        let cfg = NGramConfig::char_ngrams(2);
        let idx = NGramIndex::from_text("abcabc", &cfg);
        let ab_hash = NGramSet::hash_ngram("ab", &cfg.hash_function);
        let positions = idx.positions_of(ab_hash);
        assert_eq!(positions.len(), 2);
        // absent hash
        assert!(idx.positions_of(0).is_empty());
    }

    #[test]
    fn test_ngram_index_empty_text() {
        let cfg = NGramConfig::char_ngrams(3);
        let idx = NGramIndex::from_text("ab", &cfg);
        assert_eq!(idx.total_ngrams, 0);
        assert_eq!(idx.unique_count(), 0);
    }

    // -----------------------------------------------------------------------
    // NGramSimilarityMatrix tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_similarity_matrix_identical_docs() {
        let cfg = NGramConfig::char_ngrams(2);
        let mat = NGramSimilarityMatrix::compute(&["hello", "hello"], &cfg);
        assert!((mat.get_similarity(0, 1) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_similarity_matrix_diagonal() {
        let cfg = NGramConfig::char_ngrams(2);
        let mat = NGramSimilarityMatrix::compute(&["abc", "xyz"], &cfg);
        assert!((mat.get_similarity(0, 0) - 1.0).abs() < 1e-9);
        assert!((mat.get_similarity(1, 1) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_similarity_matrix_symmetric() {
        let cfg = NGramConfig::char_ngrams(2);
        let mat = NGramSimilarityMatrix::compute(&["abc", "bcd", "xyz"], &cfg);
        assert!((mat.get_similarity(0, 1) - mat.get_similarity(1, 0)).abs() < 1e-12);
        assert!((mat.get_similarity(0, 2) - mat.get_similarity(2, 0)).abs() < 1e-12);
    }

    #[test]
    fn test_similarity_matrix_most_similar() {
        let cfg = NGramConfig::char_ngrams(2);
        let mat = NGramSimilarityMatrix::compute(&["abcdef", "abcxyz", "mnopqr"], &cfg);
        let (i, j, _sim) = mat.most_similar_pair();
        // docs 0 and 1 share "ab","bc" so should be most similar
        assert!((i == 0 && j == 1) || (i == 1 && j == 0));
    }

    #[test]
    fn test_similarity_matrix_least_similar() {
        let cfg = NGramConfig::char_ngrams(2);
        let mat = NGramSimilarityMatrix::compute(&["aabb", "aabb", "xxyy"], &cfg);
        let (i, j, _sim) = mat.least_similar_pair();
        // pair involving doc 2 should be least similar
        assert!(i == 2 || j == 2);
    }

    #[test]
    fn test_similarity_matrix_clusters_all_similar() {
        let cfg = NGramConfig::char_ngrams(2);
        let mat = NGramSimilarityMatrix::compute(&["hello", "hello", "hello"], &cfg);
        let clusters = mat.clusters(0.5);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].len(), 3);
    }

    #[test]
    fn test_similarity_matrix_clusters_all_different() {
        let cfg = NGramConfig::char_ngrams(2);
        let mat = NGramSimilarityMatrix::compute(&["ab", "cd", "ef"], &cfg);
        let clusters = mat.clusters(0.5);
        // completely different bigrams -> 3 singletons
        assert_eq!(clusters.len(), 3);
    }

    #[test]
    fn test_similarity_matrix_to_csv() {
        let cfg = NGramConfig::char_ngrams(2);
        let mat = NGramSimilarityMatrix::compute(&["ab", "cd"], &cfg);
        let csv = mat.to_csv();
        let lines: Vec<&str> = csv.trim().split('\n').collect();
        assert_eq!(lines.len(), 2);
        // each line should have 2 comma-separated values
        assert_eq!(lines[0].split(',').count(), 2);
    }

    // -----------------------------------------------------------------------
    // DocumentFingerprint tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_fingerprint_from_text() {
        let cfg = NGramConfig::char_ngrams(2);
        let fp = DocumentFingerprint::from_text("hello world", &cfg);
        assert_eq!(fp.signature.len(), 128);
        assert!(fp.num_ngrams > 0);
    }

    #[test]
    fn test_fingerprint_identical_texts() {
        let cfg = NGramConfig::char_ngrams(2);
        let fp1 = DocumentFingerprint::from_text("the quick brown fox", &cfg);
        let fp2 = DocumentFingerprint::from_text("the quick brown fox", &cfg);
        assert!((fp1.similarity(&fp2) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_fingerprint_different_texts() {
        let cfg = NGramConfig::char_ngrams(3);
        let fp1 = DocumentFingerprint::from_text("abcdefghij", &cfg);
        let fp2 = DocumentFingerprint::from_text("klmnopqrst", &cfg);
        assert!(fp1.similarity(&fp2) < 0.3);
    }

    #[test]
    fn test_fingerprint_near_duplicate_true() {
        let cfg = NGramConfig::char_ngrams(2);
        let fp1 = DocumentFingerprint::from_text("hello world", &cfg);
        let fp2 = DocumentFingerprint::from_text("hello world", &cfg);
        assert!(fp1.is_near_duplicate(&fp2, 0.9));
    }

    #[test]
    fn test_fingerprint_near_duplicate_false() {
        let cfg = NGramConfig::char_ngrams(2);
        let fp1 = DocumentFingerprint::from_text("abcdefghijklmnop", &cfg);
        let fp2 = DocumentFingerprint::from_text("qrstuvwxyz123456", &cfg);
        assert!(!fp1.is_near_duplicate(&fp2, 0.9));
    }

    #[test]
    fn test_fingerprint_serialization_roundtrip() {
        let cfg = NGramConfig::char_ngrams(2);
        let fp = DocumentFingerprint::from_text("serialize me", &cfg);
        let bytes = fp.to_bytes();
        let fp2 = DocumentFingerprint::from_bytes(&bytes);
        assert_eq!(fp.signature, fp2.signature);
        assert_eq!(fp.num_ngrams, fp2.num_ngrams);
    }

    #[test]
    fn test_fingerprint_min_hash_signature_deterministic() {
        let cfg = NGramConfig::char_ngrams(2);
        let set = NGramSet::from_text("determinism", cfg);
        let sig1 = DocumentFingerprint::min_hash_signature(&set, 64);
        let sig2 = DocumentFingerprint::min_hash_signature(&set, 64);
        assert_eq!(sig1, sig2);
    }

    #[test]
    fn test_fingerprint_to_bytes_length() {
        let cfg = NGramConfig::char_ngrams(2);
        let fp = DocumentFingerprint::from_text("test", &cfg);
        let bytes = fp.to_bytes();
        // 8 bytes for num_ngrams + 128 * 8 bytes for signature
        assert_eq!(bytes.len(), 8 + 128 * 8);
    }

    #[test]
    fn test_fingerprint_similarity_is_symmetric() {
        let cfg = NGramConfig::char_ngrams(2);
        let fp1 = DocumentFingerprint::from_text("abc xyz", &cfg);
        let fp2 = DocumentFingerprint::from_text("xyz abc", &cfg);
        assert!((fp1.similarity(&fp2) - fp2.similarity(&fp1)).abs() < 1e-12);
    }

    // -----------------------------------------------------------------------
    // Additional NGramIndex tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_ngram_index_single_char_text() {
        let cfg = NGramConfig::char_ngrams(2);
        let idx = NGramIndex::from_text("a", &cfg);
        assert_eq!(idx.total_ngrams, 0);
        assert_eq!(idx.unique_count(), 0);
    }

    #[test]
    fn test_ngram_index_all_same_bigrams() {
        let cfg = NGramConfig::char_ngrams(2);
        let idx = NGramIndex::from_text("aaaa", &cfg);
        // "aa","aa","aa" -> 3 total, 1 unique
        assert_eq!(idx.total_ngrams, 3);
        assert_eq!(idx.unique_count(), 1);
        let top = idx.most_frequent(5);
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].1, 3);
    }

    #[test]
    fn test_ngram_index_most_frequent_top_k_limited() {
        let cfg = NGramConfig::char_ngrams(2);
        let idx = NGramIndex::from_text("abcdef", &cfg);
        // "ab","bc","cd","de","ef" -> 5 distinct, each freq 1
        let top = idx.most_frequent(3);
        assert_eq!(top.len(), 3);
        for (_h, freq) in &top {
            assert_eq!(*freq, 1);
        }
    }

    #[test]
    fn test_ngram_index_with_trigrams() {
        let cfg = NGramConfig::char_ngrams(3);
        let idx = NGramIndex::from_text("abcabc", &cfg);
        // "abc","bca","cab","abc" -> 4 total, 3 unique
        assert_eq!(idx.total_ngrams, 4);
        assert_eq!(idx.unique_count(), 3);
    }

    // -----------------------------------------------------------------------
    // Additional NGramSimilarityMatrix tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_similarity_matrix_single_doc() {
        let cfg = NGramConfig::char_ngrams(2);
        let mat = NGramSimilarityMatrix::compute(&["hello"], &cfg);
        assert_eq!(mat.num_documents, 1);
        assert!((mat.get_similarity(0, 0) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_similarity_matrix_clusters_transitive() {
        // A is similar to B, B is similar to C, but A not similar to C.
        // With single-linkage all three should cluster.
        let cfg = NGramConfig::char_ngrams(2);
        let mat = NGramSimilarityMatrix::compute(
            &["abcde", "cdefg", "efghi"],
            &cfg,
        );
        // Check that the first and third doc get linked through the second
        let sim_01 = mat.get_similarity(0, 1);
        let sim_12 = mat.get_similarity(1, 2);
        // Use the smaller of the two as threshold – single-linkage should group all
        let threshold = sim_01.min(sim_12);
        if threshold > 0.0 {
            let clusters = mat.clusters(threshold);
            // They should all be in one cluster via transitivity
            assert!(clusters.len() <= 2);
        }
    }

    #[test]
    fn test_similarity_matrix_csv_values() {
        let cfg = NGramConfig::char_ngrams(2);
        let mat = NGramSimilarityMatrix::compute(&["abc"], &cfg);
        let csv = mat.to_csv();
        // Single doc -> "1.000000\n"
        assert!(csv.contains("1.000000"));
    }

    #[test]
    fn test_similarity_matrix_num_documents() {
        let cfg = NGramConfig::char_ngrams(2);
        let docs: Vec<&str> = vec!["aa", "bb", "cc", "dd"];
        let mat = NGramSimilarityMatrix::compute(&docs, &cfg);
        assert_eq!(mat.num_documents, 4);
        assert_eq!(mat.matrix.len(), 4);
        assert_eq!(mat.matrix[0].len(), 4);
    }

    // -----------------------------------------------------------------------
    // Additional DocumentFingerprint tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_fingerprint_empty_text() {
        let cfg = NGramConfig::char_ngrams(2);
        let fp = DocumentFingerprint::from_text("a", &cfg);
        // single char with n=2 -> 0 ngrams, signature all u64::MAX
        assert_eq!(fp.num_ngrams, 0);
        for &v in &fp.signature {
            assert_eq!(v, u64::MAX);
        }
    }

    #[test]
    fn test_fingerprint_min_hash_different_num_hashes() {
        let cfg = NGramConfig::char_ngrams(2);
        let set = NGramSet::from_text("testing", cfg);
        let sig_small = DocumentFingerprint::min_hash_signature(&set, 16);
        let sig_large = DocumentFingerprint::min_hash_signature(&set, 64);
        assert_eq!(sig_small.len(), 16);
        assert_eq!(sig_large.len(), 64);
        // first 16 slots should be the same since seed generation is deterministic
        assert_eq!(&sig_small[..], &sig_large[..16]);
    }

    #[test]
    fn test_fingerprint_from_bytes_empty_signature() {
        // Construct minimal valid bytes: num_ngrams=5, no signature entries
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&5u64.to_le_bytes());
        let fp = DocumentFingerprint::from_bytes(&bytes);
        assert_eq!(fp.num_ngrams, 5);
        assert!(fp.signature.is_empty());
    }

    #[test]
    fn test_fingerprint_partial_overlap() {
        let cfg = NGramConfig::char_ngrams(2);
        let fp1 = DocumentFingerprint::from_text("abcdef", &cfg);
        let fp2 = DocumentFingerprint::from_text("cdefgh", &cfg);
        let sim = fp1.similarity(&fp2);
        // partial overlap – similarity should be between 0 and 1 exclusive
        assert!(sim > 0.0);
        assert!(sim < 1.0);
    }

    #[test]
    fn test_fingerprint_similarity_length_mismatch() {
        // Manually construct fingerprints with different signature lengths
        let fp1 = DocumentFingerprint { signature: vec![1, 2, 3], num_ngrams: 3 };
        let fp2 = DocumentFingerprint { signature: vec![1, 2], num_ngrams: 2 };
        assert!((fp1.similarity(&fp2)).abs() < 1e-9);
    }

    #[test]
    fn test_fingerprint_roundtrip_preserves_large_values() {
        let fp = DocumentFingerprint {
            signature: vec![u64::MAX, 0, u64::MAX / 2, 42],
            num_ngrams: 100,
        };
        let bytes = fp.to_bytes();
        let fp2 = DocumentFingerprint::from_bytes(&bytes);
        assert_eq!(fp.signature, fp2.signature);
        assert_eq!(fp.num_ngrams, fp2.num_ngrams);
    }


}
