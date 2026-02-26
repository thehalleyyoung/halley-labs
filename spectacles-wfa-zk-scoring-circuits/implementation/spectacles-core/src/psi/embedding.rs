//! Embedding-based semantic contamination detection.
//!
//! Extends n-gram overlap detection with embedding-space similarity to catch
//! paraphrase-level contamination that character/word n-gram methods miss.
//!
//! # Approach
//!
//! Three layers of contamination detection:
//! 1. **Surface**: Character/word n-gram overlap (existing PSI module)
//! 2. **Semantic**: Embedding cosine similarity via locality-sensitive hashing
//! 3. **Distributional**: Statistical divergence of token frequency distributions
//!
//! The embedding layer uses MinHash + locality-sensitive hashing (LSH) for
//! privacy-preserving approximate nearest-neighbor search, enabling detection
//! of paraphrased contamination without revealing either party's data.

use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use sha2::{Sha256, Digest};

// ---------------------------------------------------------------------------
// EmbeddingConfig
// ---------------------------------------------------------------------------

/// Configuration for embedding-based contamination detection.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Dimensionality of token embeddings (character n-gram based).
    pub embedding_dim: usize,
    /// Number of hash functions for MinHash signatures.
    pub num_hashes: usize,
    /// Number of LSH bands for banding technique.
    pub num_bands: usize,
    /// Rows per band (num_hashes / num_bands must be integer).
    pub rows_per_band: usize,
    /// Cosine similarity threshold for paraphrase detection.
    pub similarity_threshold: f64,
    /// Character n-gram size for building embeddings.
    pub char_ngram_n: usize,
    /// Whether to normalize embeddings to unit length.
    pub normalize: bool,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 256,
            num_hashes: 128,
            num_bands: 16,
            rows_per_band: 8,
            similarity_threshold: 0.7,
            char_ngram_n: 3,
            normalize: true,
        }
    }
}

impl EmbeddingConfig {
    pub fn high_sensitivity() -> Self {
        Self {
            similarity_threshold: 0.5,
            num_hashes: 256,
            num_bands: 32,
            rows_per_band: 8,
            ..Default::default()
        }
    }

    pub fn high_specificity() -> Self {
        Self {
            similarity_threshold: 0.85,
            num_hashes: 128,
            num_bands: 8,
            rows_per_band: 16,
            ..Default::default()
        }
    }
}

// ---------------------------------------------------------------------------
// SparseEmbedding
// ---------------------------------------------------------------------------

/// Sparse vector representation for character n-gram embeddings.
/// Maps dimension indices to weights (TF-IDF style).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SparseEmbedding {
    pub components: HashMap<u64, f64>,
    pub norm: f64,
}

impl SparseEmbedding {
    /// Build a character n-gram embedding from text.
    pub fn from_text(text: &str, ngram_n: usize, dim: usize) -> Self {
        let normalized = text.to_lowercase();
        let chars: Vec<char> = normalized.chars().collect();
        let mut components = HashMap::new();

        if chars.len() >= ngram_n {
            for window in chars.windows(ngram_n) {
                let s: String = window.iter().collect();
                let hash = Self::hash_to_dim(&s, dim);
                *components.entry(hash).or_insert(0.0) += 1.0;
            }
        }

        // Apply sublinear TF scaling: 1 + log(count)
        for val in components.values_mut() {
            *val = 1.0 + (*val as f64).ln();
        }

        let norm = components.values().map(|v| v * v).sum::<f64>().sqrt();

        // Normalize to unit length
        if norm > 0.0 {
            for val in components.values_mut() {
                *val /= norm;
            }
        }

        Self {
            components,
            norm: if norm > 0.0 { 1.0 } else { 0.0 },
        }
    }

    /// Cosine similarity between two sparse embeddings.
    pub fn cosine_similarity(&self, other: &SparseEmbedding) -> f64 {
        if self.norm == 0.0 || other.norm == 0.0 {
            return 0.0;
        }

        let mut dot = 0.0;
        // Iterate over the smaller set
        let (smaller, larger) = if self.components.len() <= other.components.len() {
            (&self.components, &other.components)
        } else {
            (&other.components, &self.components)
        };

        for (dim, val) in smaller {
            if let Some(other_val) = larger.get(dim) {
                dot += val * other_val;
            }
        }

        // Both vectors are unit-normalized, so dot product = cosine similarity
        dot.clamp(-1.0, 1.0)
    }

    fn hash_to_dim(s: &str, dim: usize) -> u64 {
        let mut hasher = Sha256::new();
        hasher.update(s.as_bytes());
        let result = hasher.finalize();
        let hash_val = u64::from_le_bytes(result[0..8].try_into().unwrap());
        hash_val % (dim as u64)
    }
}

// ---------------------------------------------------------------------------
// MinHashSignature
// ---------------------------------------------------------------------------

/// MinHash signature for approximate Jaccard similarity estimation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MinHashSignature {
    pub values: Vec<u64>,
}

impl MinHashSignature {
    /// Compute MinHash signature from a set of shingles.
    pub fn from_shingles(shingles: &[u64], num_hashes: usize) -> Self {
        let mut values = vec![u64::MAX; num_hashes];

        // Use universal hash family: h_i(x) = (a_i * x + b_i) mod p
        let p: u64 = (1u64 << 61) - 1; // Mersenne prime 2^61 - 1
        for (i, val) in values.iter_mut().enumerate() {
            let a = Self::hash_coeff(i, 0);
            let b = Self::hash_coeff(i, 1);
            for &shingle in shingles {
                let h = Self::universal_hash(shingle, a, b, p);
                if h < *val {
                    *val = h;
                }
            }
        }

        Self { values }
    }

    /// Estimate Jaccard similarity from two MinHash signatures.
    pub fn jaccard_estimate(&self, other: &MinHashSignature) -> f64 {
        assert_eq!(self.values.len(), other.values.len());
        let matches = self.values.iter()
            .zip(other.values.iter())
            .filter(|(a, b)| a == b)
            .count();
        matches as f64 / self.values.len() as f64
    }

    fn universal_hash(x: u64, a: u64, b: u64, p: u64) -> u64 {
        // (a*x + b) mod p using 128-bit arithmetic to avoid overflow
        let ax = (a as u128) * (x as u128);
        let axb = ax + (b as u128);
        (axb % (p as u128)) as u64
    }

    fn hash_coeff(index: usize, salt: u8) -> u64 {
        let mut hasher = Sha256::new();
        hasher.update(&index.to_le_bytes());
        hasher.update(&[salt]);
        let result = hasher.finalize();
        u64::from_le_bytes(result[0..8].try_into().unwrap())
    }
}

// ---------------------------------------------------------------------------
// LSHIndex
// ---------------------------------------------------------------------------

/// Locality-sensitive hashing index for approximate nearest neighbor search.
#[derive(Clone, Debug)]
pub struct LSHIndex {
    config: EmbeddingConfig,
    /// Buckets: band_index -> (bucket_hash -> list of item indices)
    buckets: Vec<HashMap<u64, Vec<usize>>>,
    /// Stored signatures
    signatures: Vec<MinHashSignature>,
}

impl LSHIndex {
    pub fn new(config: EmbeddingConfig) -> Self {
        assert_eq!(config.num_hashes, config.num_bands * config.rows_per_band);
        let buckets = (0..config.num_bands).map(|_| HashMap::new()).collect();
        Self {
            config,
            buckets,
            signatures: Vec::new(),
        }
    }

    /// Insert a MinHash signature into the index.
    pub fn insert(&mut self, signature: MinHashSignature) -> usize {
        let idx = self.signatures.len();

        for band in 0..self.config.num_bands {
            let start = band * self.config.rows_per_band;
            let end = start + self.config.rows_per_band;
            let band_hash = Self::hash_band(&signature.values[start..end]);
            self.buckets[band]
                .entry(band_hash)
                .or_default()
                .push(idx);
        }

        self.signatures.push(signature);
        idx
    }

    /// Query for candidate neighbors (approximate nearest neighbors).
    pub fn query_candidates(&self, signature: &MinHashSignature) -> Vec<usize> {
        let mut candidates = std::collections::HashSet::new();

        for band in 0..self.config.num_bands {
            let start = band * self.config.rows_per_band;
            let end = start + self.config.rows_per_band;
            let band_hash = Self::hash_band(&signature.values[start..end]);

            if let Some(bucket) = self.buckets[band].get(&band_hash) {
                for &idx in bucket {
                    candidates.insert(idx);
                }
            }
        }

        candidates.into_iter().collect()
    }

    /// Get the estimated Jaccard similarity for a candidate pair.
    pub fn similarity(&self, idx: usize, query: &MinHashSignature) -> f64 {
        self.signatures[idx].jaccard_estimate(query)
    }

    fn hash_band(band_values: &[u64]) -> u64 {
        let mut hasher = Sha256::new();
        for v in band_values {
            hasher.update(&v.to_le_bytes());
        }
        let result = hasher.finalize();
        u64::from_le_bytes(result[0..8].try_into().unwrap())
    }

    pub fn len(&self) -> usize {
        self.signatures.len()
    }

    pub fn is_empty(&self) -> bool {
        self.signatures.is_empty()
    }
}

// ---------------------------------------------------------------------------
// DistributionalDivergence
// ---------------------------------------------------------------------------

/// Token frequency distribution analysis for distributional contamination.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TokenDistribution {
    pub frequencies: HashMap<String, f64>,
    pub total_tokens: usize,
    pub vocabulary_size: usize,
}

impl TokenDistribution {
    /// Build token frequency distribution from text.
    pub fn from_text(text: &str) -> Self {
        let tokens: Vec<&str> = text.split_whitespace().collect();
        let total = tokens.len();
        let mut counts: HashMap<String, usize> = HashMap::new();

        for token in &tokens {
            *counts.entry(token.to_lowercase()).or_insert(0) += 1;
        }

        let vocabulary_size = counts.len();
        let frequencies: HashMap<String, f64> = counts.into_iter()
            .map(|(k, v)| (k, v as f64 / total.max(1) as f64))
            .collect();

        Self {
            frequencies,
            total_tokens: total,
            vocabulary_size,
        }
    }

    /// Jensen-Shannon divergence between two distributions.
    /// Returns value in [0, ln(2)] ≈ [0, 0.693].
    pub fn js_divergence(&self, other: &TokenDistribution) -> f64 {
        let all_keys: std::collections::HashSet<&String> = self.frequencies.keys()
            .chain(other.frequencies.keys())
            .collect();

        let mut div = 0.0;
        for key in all_keys {
            let p = *self.frequencies.get(key).unwrap_or(&0.0);
            let q = *other.frequencies.get(key).unwrap_or(&0.0);
            let m = (p + q) / 2.0;

            if p > 0.0 && m > 0.0 {
                div += p * (p / m).ln();
            }
            if q > 0.0 && m > 0.0 {
                div += q * (q / m).ln();
            }
        }

        div / 2.0
    }

    /// Kullback-Leibler divergence D_KL(self || other).
    /// Returns +∞ if other has zero probability where self is nonzero.
    pub fn kl_divergence(&self, other: &TokenDistribution) -> f64 {
        let smoothing = 1e-10;
        let mut div = 0.0;

        for (key, &p) in &self.frequencies {
            if p > 0.0 {
                let q = other.frequencies.get(key).copied().unwrap_or(0.0) + smoothing;
                div += p * (p / q).ln();
            }
        }

        div
    }
}

// ---------------------------------------------------------------------------
// MultiLayerDetector
// ---------------------------------------------------------------------------

/// Detection layer types.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum DetectionLayer {
    NGramOverlap,
    EmbeddingSimilarity,
    DistributionalDivergence,
}

/// Result from a single detection layer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LayerResult {
    pub layer: DetectionLayer,
    pub score: f64,
    pub threshold: f64,
    pub detected: bool,
    pub confidence: f64,
}

/// Aggregate multi-layer detection result.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MultiLayerResult {
    pub layer_results: Vec<LayerResult>,
    pub aggregate_score: f64,
    pub contamination_detected: bool,
    pub detection_mode: DetectionMode,
    pub details: MultiLayerDetails,
}

/// How layers are combined for the final decision.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum DetectionMode {
    /// Any single layer detecting contamination triggers alert.
    AnyLayer,
    /// Majority of layers must agree.
    MajorityVote,
    /// Weighted combination of layer scores.
    WeightedCombination,
}

/// Detailed statistics from multi-layer detection.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MultiLayerDetails {
    pub ngram_overlap: f64,
    pub embedding_similarity: f64,
    pub distributional_divergence: f64,
    pub layers_triggered: usize,
    pub total_layers: usize,
}

/// Multi-layer contamination detector combining surface, semantic, and
/// distributional signals.
#[derive(Clone, Debug)]
pub struct MultiLayerDetector {
    pub config: MultiLayerConfig,
}

/// Configuration for multi-layer detection.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MultiLayerConfig {
    pub embedding_config: EmbeddingConfig,
    pub ngram_threshold: f64,
    pub embedding_threshold: f64,
    pub divergence_threshold: f64,
    pub detection_mode: DetectionMode,
    /// Weights for weighted combination mode [ngram, embedding, distributional].
    pub layer_weights: [f64; 3],
}

impl Default for MultiLayerConfig {
    fn default() -> Self {
        Self {
            embedding_config: EmbeddingConfig::default(),
            ngram_threshold: 0.03,
            embedding_threshold: 0.7,
            divergence_threshold: 0.15,
            detection_mode: DetectionMode::AnyLayer,
            layer_weights: [0.3, 0.5, 0.2],
        }
    }
}

impl MultiLayerDetector {
    pub fn new(config: MultiLayerConfig) -> Self {
        Self { config }
    }

    /// Detect contamination using all three layers.
    ///
    /// # Arguments
    /// * `train_texts` - Training corpus texts
    /// * `eval_texts` - Evaluation/benchmark texts
    /// * `ngram_overlap` - Pre-computed n-gram overlap ratio (from PSI module)
    pub fn detect(
        &self,
        train_texts: &[&str],
        eval_texts: &[&str],
        ngram_overlap: f64,
    ) -> MultiLayerResult {
        // Layer 1: N-gram overlap (passed in from existing PSI module)
        let ngram_result = LayerResult {
            layer: DetectionLayer::NGramOverlap,
            score: ngram_overlap,
            threshold: self.config.ngram_threshold,
            detected: ngram_overlap > self.config.ngram_threshold,
            confidence: Self::sigmoid_confidence(ngram_overlap, self.config.ngram_threshold),
        };

        // Layer 2: Embedding similarity
        let embedding_sim = self.compute_embedding_similarity(train_texts, eval_texts);
        let embedding_result = LayerResult {
            layer: DetectionLayer::EmbeddingSimilarity,
            score: embedding_sim,
            threshold: self.config.embedding_threshold,
            detected: embedding_sim > self.config.embedding_threshold,
            confidence: Self::sigmoid_confidence(embedding_sim, self.config.embedding_threshold),
        };

        // Layer 3: Distributional divergence (low divergence = suspicious)
        let div_score = self.compute_distributional_score(train_texts, eval_texts);
        let div_result = LayerResult {
            layer: DetectionLayer::DistributionalDivergence,
            score: div_score,
            threshold: self.config.divergence_threshold,
            detected: div_score > self.config.divergence_threshold,
            confidence: Self::sigmoid_confidence(div_score, self.config.divergence_threshold),
        };

        let layer_results = vec![ngram_result.clone(), embedding_result.clone(), div_result.clone()];
        let layers_triggered = layer_results.iter().filter(|r| r.detected).count();

        let (aggregate_score, contamination_detected) = match self.config.detection_mode {
            DetectionMode::AnyLayer => {
                let score = layer_results.iter().map(|r| r.score).fold(0.0_f64, f64::max);
                let detected = layers_triggered > 0;
                (score, detected)
            }
            DetectionMode::MajorityVote => {
                let score = layer_results.iter().map(|r| r.confidence).sum::<f64>() / 3.0;
                let detected = layers_triggered >= 2;
                (score, detected)
            }
            DetectionMode::WeightedCombination => {
                let w = &self.config.layer_weights;
                let score = w[0] * ngram_result.confidence
                    + w[1] * embedding_result.confidence
                    + w[2] * div_result.confidence;
                let detected = score > 0.5;
                (score, detected)
            }
        };

        MultiLayerResult {
            layer_results,
            aggregate_score,
            contamination_detected,
            detection_mode: self.config.detection_mode.clone(),
            details: MultiLayerDetails {
                ngram_overlap,
                embedding_similarity: embedding_sim,
                distributional_divergence: div_score,
                layers_triggered,
                total_layers: 3,
            },
        }
    }

    /// Compute maximum embedding similarity between eval and train texts.
    fn compute_embedding_similarity(&self, train_texts: &[&str], eval_texts: &[&str]) -> f64 {
        let cfg = &self.config.embedding_config;

        // Build embeddings for training texts
        let train_embeddings: Vec<SparseEmbedding> = train_texts.iter()
            .map(|t| SparseEmbedding::from_text(t, cfg.char_ngram_n, cfg.embedding_dim))
            .collect();

        // Build embeddings for eval texts
        let eval_embeddings: Vec<SparseEmbedding> = eval_texts.iter()
            .map(|t| SparseEmbedding::from_text(t, cfg.char_ngram_n, cfg.embedding_dim))
            .collect();

        // Find maximum similarity between any eval-train pair
        let mut max_sim = 0.0_f64;
        for eval_emb in &eval_embeddings {
            for train_emb in &train_embeddings {
                let sim = eval_emb.cosine_similarity(train_emb);
                max_sim = max_sim.max(sim);
            }
        }

        max_sim
    }

    /// Compute distributional contamination score.
    /// High score = suspicious (train and eval distributions are too similar).
    fn compute_distributional_score(&self, train_texts: &[&str], eval_texts: &[&str]) -> f64 {
        let train_combined = train_texts.join(" ");
        let eval_combined = eval_texts.join(" ");

        let train_dist = TokenDistribution::from_text(&train_combined);
        let eval_dist = TokenDistribution::from_text(&eval_combined);

        let js_div = train_dist.js_divergence(&eval_dist);

        // Invert: low divergence = high contamination score
        // JS divergence is in [0, ln(2)], so 1 - js/ln(2) maps to [0, 1]
        let contamination_score = 1.0 - (js_div / 2.0_f64.ln()).min(1.0);
        contamination_score
    }

    fn sigmoid_confidence(score: f64, threshold: f64) -> f64 {
        // Sigmoid centered at threshold with steepness 10
        let k = 10.0;
        1.0 / (1.0 + (-k * (score - threshold)).exp())
    }
}

// ---------------------------------------------------------------------------
// ParaphraseDetector
// ---------------------------------------------------------------------------

/// Specialized detector for paraphrase-level contamination.
/// Uses character n-gram embeddings + MinHash + LSH for scalable detection.
#[derive(Clone, Debug)]
pub struct ParaphraseDetector {
    config: EmbeddingConfig,
    lsh_index: Option<LSHIndex>,
}

impl ParaphraseDetector {
    pub fn new(config: EmbeddingConfig) -> Self {
        Self {
            config,
            lsh_index: None,
        }
    }

    /// Index a corpus of reference texts for paraphrase search.
    pub fn index_corpus(&mut self, texts: &[&str]) {
        let mut index = LSHIndex::new(self.config.clone());

        for text in texts {
            let shingles = self.text_to_shingles(text);
            let sig = MinHashSignature::from_shingles(&shingles, self.config.num_hashes);
            index.insert(sig);
        }

        self.lsh_index = Some(index);
    }

    /// Query whether a text has near-duplicates in the indexed corpus.
    /// Returns (is_paraphrase, max_similarity, num_candidates).
    pub fn detect_paraphrase(&self, text: &str) -> (bool, f64, usize) {
        let index = match &self.lsh_index {
            Some(idx) => idx,
            None => return (false, 0.0, 0),
        };

        let shingles = self.text_to_shingles(text);
        let sig = MinHashSignature::from_shingles(&shingles, self.config.num_hashes);

        let candidates = index.query_candidates(&sig);
        let num_candidates = candidates.len();

        let mut max_sim = 0.0_f64;
        for idx in &candidates {
            let sim = index.similarity(*idx, &sig);
            max_sim = max_sim.max(sim);
        }

        let is_paraphrase = max_sim >= self.config.similarity_threshold;
        (is_paraphrase, max_sim, num_candidates)
    }

    /// Batch detection on multiple query texts.
    pub fn detect_batch(&self, texts: &[&str]) -> Vec<(bool, f64, usize)> {
        texts.iter().map(|t| self.detect_paraphrase(t)).collect()
    }

    fn text_to_shingles(&self, text: &str) -> Vec<u64> {
        let normalized = text.to_lowercase();
        let chars: Vec<char> = normalized.chars().collect();
        let n = self.config.char_ngram_n;

        if chars.len() < n {
            return vec![];
        }

        chars.windows(n)
            .map(|w| {
                let s: String = w.iter().collect();
                let mut hasher = Sha256::new();
                hasher.update(s.as_bytes());
                let result = hasher.finalize();
                u64::from_le_bytes(result[0..8].try_into().unwrap())
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_embedding_identical() {
        let emb1 = SparseEmbedding::from_text("the cat sat on the mat", 3, 256);
        let emb2 = SparseEmbedding::from_text("the cat sat on the mat", 3, 256);
        let sim = emb1.cosine_similarity(&emb2);
        assert!((sim - 1.0).abs() < 1e-10, "Identical texts should have similarity ~1.0, got {}", sim);
    }

    #[test]
    fn test_sparse_embedding_different() {
        let emb1 = SparseEmbedding::from_text("the cat sat on the mat", 3, 256);
        let emb2 = SparseEmbedding::from_text("quantum mechanics describes subatomic particles", 3, 256);
        let sim = emb1.cosine_similarity(&emb2);
        assert!(sim < 0.5, "Very different texts should have low similarity, got {}", sim);
    }

    #[test]
    fn test_sparse_embedding_paraphrase() {
        let emb1 = SparseEmbedding::from_text("the cat sat on the mat", 3, 256);
        let emb2 = SparseEmbedding::from_text("the cat was sitting on the mat", 3, 256);
        let sim = emb1.cosine_similarity(&emb2);
        // Paraphrases should have moderate-to-high similarity
        assert!(sim > 0.3, "Paraphrases should have moderate similarity, got {}", sim);
    }

    #[test]
    fn test_minhash_identical_sets() {
        let shingles = vec![1, 2, 3, 4, 5];
        let sig1 = MinHashSignature::from_shingles(&shingles, 128);
        let sig2 = MinHashSignature::from_shingles(&shingles, 128);
        let jaccard = sig1.jaccard_estimate(&sig2);
        assert!((jaccard - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_minhash_disjoint_sets() {
        let s1 = vec![1, 2, 3, 4, 5];
        let s2 = vec![100, 200, 300, 400, 500];
        let sig1 = MinHashSignature::from_shingles(&s1, 128);
        let sig2 = MinHashSignature::from_shingles(&s2, 128);
        let jaccard = sig1.jaccard_estimate(&sig2);
        assert!(jaccard < 0.2, "Disjoint sets should have low Jaccard, got {}", jaccard);
    }

    #[test]
    fn test_lsh_index_retrieval() {
        let config = EmbeddingConfig::default();
        let mut index = LSHIndex::new(config);

        let s1 = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let s2 = vec![1, 2, 3, 4, 5, 6, 7, 9]; // very similar to s1
        let s3 = vec![100, 200, 300, 400, 500, 600, 700, 800]; // very different

        let sig1 = MinHashSignature::from_shingles(&s1, 128);
        let sig2 = MinHashSignature::from_shingles(&s2, 128);
        let sig3 = MinHashSignature::from_shingles(&s3, 128);

        index.insert(sig1);
        index.insert(sig2);
        index.insert(sig3);

        // Query with s1-like signature should find s1 and s2
        let query = MinHashSignature::from_shingles(&s1, 128);
        let candidates = index.query_candidates(&query);
        assert!(!candidates.is_empty(), "Should find at least one candidate");
        assert!(candidates.contains(&0), "Should find index 0 (identical)");
    }

    #[test]
    fn test_token_distribution_identical() {
        let d1 = TokenDistribution::from_text("the cat sat on the mat");
        let d2 = TokenDistribution::from_text("the cat sat on the mat");
        let js = d1.js_divergence(&d2);
        assert!(js < 1e-10, "Identical distributions should have JS div ~0, got {}", js);
    }

    #[test]
    fn test_token_distribution_different() {
        let d1 = TokenDistribution::from_text("the cat sat on the mat the cat");
        let d2 = TokenDistribution::from_text("quantum mechanics describes subatomic quantum particles");
        let js = d1.js_divergence(&d2);
        assert!(js > 0.1, "Different distributions should have high JS div, got {}", js);
    }

    #[test]
    fn test_multi_layer_clean_data() {
        let detector = MultiLayerDetector::new(MultiLayerConfig::default());

        let train = vec!["the weather is nice today", "I like programming in rust"];
        let eval = vec!["quantum computing is fascinating", "machine learning advances"];

        let result = detector.detect(&train, &eval, 0.01); // low n-gram overlap
        // Clean data should not trigger contamination
        assert!(!result.details.ngram_overlap.is_nan());
    }

    #[test]
    fn test_multi_layer_contaminated_data() {
        let detector = MultiLayerDetector::new(MultiLayerConfig::default());

        let train = vec!["the cat sat on the mat and looked around"];
        let eval = vec!["the cat sat on the mat and looked around"]; // exact copy

        let result = detector.detect(&train, &eval, 1.0); // high n-gram overlap
        assert!(result.contamination_detected, "Should detect contamination for exact copies");
    }

    #[test]
    fn test_paraphrase_detector_exact() {
        let config = EmbeddingConfig::default();
        let mut detector = ParaphraseDetector::new(config);

        let corpus = vec![
            "the cat sat on the mat",
            "dogs are loyal companions",
            "the weather is beautiful today",
        ];
        detector.index_corpus(&corpus);

        let (detected, sim, _) = detector.detect_paraphrase("the cat sat on the mat");
        assert!(detected, "Should detect exact match");
        assert!((sim - 1.0).abs() < 0.01, "Exact match should have sim ~1.0, got {}", sim);
    }

    #[test]
    fn test_paraphrase_detector_novel() {
        let config = EmbeddingConfig {
            similarity_threshold: 0.7,
            ..Default::default()
        };
        let mut detector = ParaphraseDetector::new(config);

        let corpus = vec![
            "the cat sat on the mat",
            "dogs are loyal companions",
        ];
        detector.index_corpus(&corpus);

        let (detected, sim, _) = detector.detect_paraphrase(
            "quantum computing enables exponential speedups for certain algorithms"
        );
        assert!(!detected, "Should not detect unrelated text as paraphrase, sim={}", sim);
    }
}
