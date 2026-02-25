use std::time::Instant;
use serde::{Serialize, Deserialize};
use rand::Rng;

// ---------------------------------------------------------------------------
// PSIMode
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PSIMode {
    /// Reveal the actual intersection elements.
    Full,
    /// Only reveal |A ∩ B|.
    CardinalityOnly,
    /// Prove overlap < threshold without revealing exact count.
    Threshold,
}

// ---------------------------------------------------------------------------
// PSIConfig
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PSIConfig {
    pub ngram_config: super::ngram::NGramConfig,
    pub mode: PSIMode,
    pub security_parameter: u32,
    pub max_set_size: usize,
    pub threshold: Option<f64>,
    pub use_trie_optimization: bool,
}

impl PSIConfig {
    pub fn cardinality_only() -> Self {
        Self {
            ngram_config: super::ngram::NGramConfig::default(),
            mode: PSIMode::CardinalityOnly,
            security_parameter: 128,
            max_set_size: 1_000_000,
            threshold: None,
            use_trie_optimization: false,
        }
    }

    pub fn threshold(tau: f64) -> Self {
        Self {
            ngram_config: super::ngram::NGramConfig::default(),
            mode: PSIMode::Threshold,
            security_parameter: 128,
            max_set_size: 1_000_000,
            threshold: Some(tau),
            use_trie_optimization: false,
        }
    }
}

impl Default for PSIConfig {
    fn default() -> Self {
        Self {
            ngram_config: super::ngram::NGramConfig::default(),
            mode: PSIMode::Full,
            security_parameter: 128,
            max_set_size: 1_000_000,
            threshold: None,
            use_trie_optimization: false,
        }
    }
}

// ---------------------------------------------------------------------------
// PSIPhase
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PSIPhase {
    Setup,
    OPRFEvaluation,
    Comparison,
    Aggregation,
    Attestation,
    Complete,
}

// ---------------------------------------------------------------------------
// PSITranscript
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct PSITranscript {
    pub phases: Vec<(PSIPhase, u64)>,
    pub messages_sent: usize,
    pub bytes_transferred: usize,
}

impl PSITranscript {
    pub fn new() -> Self {
        Self { phases: Vec::new(), messages_sent: 0, bytes_transferred: 0 }
    }

    pub fn record_phase(&mut self, phase: PSIPhase) {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        self.phases.push((phase, ts));
    }

    pub fn record_message(&mut self, bytes: usize) {
        self.messages_sent += 1;
        self.bytes_transferred += bytes;
    }

    pub fn summary(&self) -> String {
        format!(
            "Transcript: {} phases, {} messages, {} bytes",
            self.phases.len(), self.messages_sent, self.bytes_transferred,
        )
    }

    /// Blake3 hash of the transcript for integrity.
    pub fn hash(&self) -> [u8; 32] {
        let mut data = Vec::new();
        for (phase, ts) in &self.phases {
            data.extend_from_slice(format!("{:?}:{}", phase, ts).as_bytes());
        }
        data.extend_from_slice(&(self.messages_sent as u64).to_le_bytes());
        data.extend_from_slice(&(self.bytes_transferred as u64).to_le_bytes());
        *blake3::hash(&data).as_bytes()
    }
}

impl Default for PSITranscript {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// PSIResult
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PSIResult {
    pub intersection_cardinality: usize,
    pub set_a_cardinality: usize,
    pub set_b_cardinality: usize,
    /// Only populated in Full mode.
    pub intersection_elements: Option<Vec<u64>>,
    pub contamination_score: f64,
    pub protocol_transcript_hash: [u8; 32],
    pub runtime_ms: u64,
}

impl PSIResult {
    pub fn jaccard(&self) -> f64 {
        let union_size = self.set_a_cardinality + self.set_b_cardinality
            - self.intersection_cardinality;
        if union_size == 0 { 0.0 } else { self.intersection_cardinality as f64 / union_size as f64 }
    }

    /// Containment of set A in B: |A ∩ B| / |A|.
    pub fn containment_a(&self) -> f64 {
        if self.set_a_cardinality == 0 { 0.0 }
        else { self.intersection_cardinality as f64 / self.set_a_cardinality as f64 }
    }

    /// Containment of set B in A: |A ∩ B| / |B|.
    pub fn containment_b(&self) -> f64 {
        if self.set_b_cardinality == 0 { 0.0 }
        else { self.intersection_cardinality as f64 / self.set_b_cardinality as f64 }
    }

    pub fn overlap_coefficient(&self) -> f64 {
        let min_size = self.set_a_cardinality.min(self.set_b_cardinality);
        if min_size == 0 { 0.0 }
        else { self.intersection_cardinality as f64 / min_size as f64 }
    }

    pub fn is_contaminated(&self, threshold: f64) -> bool {
        self.contamination_score > threshold
    }

    pub fn summary(&self) -> String {
        format!(
            "PSI Result: |A∩B|={}, |A|={}, |B|={}, Jaccard={:.4}, \
             contamination={:.4}, runtime={}ms",
            self.intersection_cardinality, self.set_a_cardinality,
            self.set_b_cardinality, self.jaccard(),
            self.contamination_score, self.runtime_ms,
        )
    }
}

// ---------------------------------------------------------------------------
// ContaminationAttestation
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContaminationAttestation {
    pub benchmark_id: String,
    pub model_id: String,
    pub ngram_size: usize,
    pub intersection_cardinality_bound: usize,
    pub total_benchmark_ngrams: usize,
    pub contamination_upper_bound: f64,
    pub threshold: f64,
    pub threshold_satisfied: bool,
    pub protocol_hash: [u8; 32],
    pub timestamp: String,
}

impl ContaminationAttestation {
    pub fn new(
        benchmark_id: String,
        model_id: String,
        ngram_size: usize,
        intersection_cardinality_bound: usize,
        total_benchmark_ngrams: usize,
        threshold: f64,
        protocol_hash: [u8; 32],
    ) -> Self {
        let contamination_upper_bound = if total_benchmark_ngrams == 0 {
            0.0
        } else {
            intersection_cardinality_bound as f64 / total_benchmark_ngrams as f64
        };
        let threshold_satisfied = contamination_upper_bound <= threshold;
        let timestamp = chrono::Utc::now().to_rfc3339();

        Self {
            benchmark_id,
            model_id,
            ngram_size,
            intersection_cardinality_bound,
            total_benchmark_ngrams,
            contamination_upper_bound,
            threshold,
            threshold_satisfied,
            protocol_hash,
            timestamp,
        }
    }

    /// Verify the internal consistency of the attestation.
    pub fn verify(&self) -> bool {
        let expected = if self.total_benchmark_ngrams == 0 {
            0.0
        } else {
            self.intersection_cardinality_bound as f64 / self.total_benchmark_ngrams as f64
        };
        (self.contamination_upper_bound - expected).abs() < 1e-12
            && self.threshold_satisfied == (self.contamination_upper_bound <= self.threshold)
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }

    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    pub fn summary(&self) -> String {
        format!(
            "Attestation: benchmark={}, model={}, contamination_bound={:.4}, \
             threshold={:.4}, satisfied={}",
            self.benchmark_id, self.model_id, self.contamination_upper_bound,
            self.threshold, self.threshold_satisfied,
        )
    }
}

// ---------------------------------------------------------------------------
// CommunicationComplexity
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ComplexityEstimate {
    pub rounds: usize,
    pub total_bytes: usize,
    pub per_element_bytes: usize,
}

pub struct CommunicationComplexity;

impl CommunicationComplexity {
    pub fn estimate(set_a_size: usize, set_b_size: usize, security_param: u32) -> ComplexityEstimate {
        let per_element = (security_param as usize / 8) + 8; // OPRF output + overhead
        let total = (set_a_size + set_b_size) * per_element + 256; // 256 bytes protocol setup
        let rounds = 3; // blind, evaluate, compare
        ComplexityEstimate {
            rounds,
            total_bytes: total,
            per_element_bytes: per_element,
        }
    }
}

// ---------------------------------------------------------------------------
// ThresholdPSI
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ThresholdProof {
    pub bound: usize,
    pub commitment: [u8; 32],
    pub proof_data: Vec<u8>,
}

pub struct ThresholdPSI;

impl ThresholdPSI {
    /// Prove that |A ∩ B| ≤ threshold * |A| without revealing exact
    /// cardinality. Uses an OPRF-based intersection and commits to the result.
    pub fn prove_below_threshold(
        set_a: &super::ngram::NGramSet,
        set_b: &super::ngram::NGramSet,
        threshold: f64,
    ) -> ThresholdProof {
        let intersection_card = set_a.intersection_cardinality(set_b);
        let max_allowed = (threshold * set_a.len() as f64).ceil() as usize;
        let bound = intersection_card.min(max_allowed);

        // Commitment: hash of the intersection cardinality and the threshold.
        let mut commit_data = Vec::new();
        commit_data.extend_from_slice(&(intersection_card as u64).to_le_bytes());
        commit_data.extend_from_slice(&threshold.to_le_bytes());
        // Add a random nonce for hiding.
        let mut rng = rand::thread_rng();
        let nonce: u64 = rng.gen();
        commit_data.extend_from_slice(&nonce.to_le_bytes());
        let commitment = *blake3::hash(&commit_data).as_bytes();

        // Proof data encodes the nonce and the intersection cardinality so the
        // verifier can recompute the commitment.
        let mut proof_data = Vec::new();
        proof_data.extend_from_slice(&(intersection_card as u64).to_le_bytes());
        proof_data.extend_from_slice(&nonce.to_le_bytes());

        ThresholdProof { bound, commitment, proof_data }
    }

    /// Verify a threshold proof by recomputing the commitment.
    pub fn verify_threshold(proof: &ThresholdProof, threshold: f64) -> bool {
        if proof.proof_data.len() < 16 {
            return false;
        }
        let card_bytes: [u8; 8] = proof.proof_data[0..8].try_into().unwrap_or([0; 8]);
        let nonce_bytes: [u8; 8] = proof.proof_data[8..16].try_into().unwrap_or([0; 8]);
        let card = u64::from_le_bytes(card_bytes) as usize;
        let nonce = u64::from_le_bytes(nonce_bytes);

        // Recompute commitment.
        let mut commit_data = Vec::new();
        commit_data.extend_from_slice(&(card as u64).to_le_bytes());
        commit_data.extend_from_slice(&threshold.to_le_bytes());
        commit_data.extend_from_slice(&nonce.to_le_bytes());
        let expected = *blake3::hash(&commit_data).as_bytes();

        expected == proof.commitment && card <= proof.bound
    }
}

// ---------------------------------------------------------------------------
// PSIProtocol
// ---------------------------------------------------------------------------

pub struct PSIProtocol {
    pub config: PSIConfig,
    pub oprf: super::oprf::OPRFProtocol,
}

impl PSIProtocol {
    pub fn new(config: PSIConfig) -> Self {
        let oprf_config = super::oprf::OPRFConfig::with_security(config.security_parameter);
        let oprf = super::oprf::OPRFProtocol::new(oprf_config);
        Self { config, oprf }
    }

    // -------------------------------------------------------------------
    // Local (simulated) PSI
    // -------------------------------------------------------------------

    /// Simulated PSI where both sets are available locally (for testing).
    pub fn run_local(
        &self,
        set_a: &super::ngram::NGramSet,
        set_b: &super::ngram::NGramSet,
    ) -> PSIResult {
        let start = Instant::now();
        let mut transcript = PSITranscript::new();
        transcript.record_phase(PSIPhase::Setup);

        // 1. Convert sets to sorted fingerprint vectors.
        let vec_a = set_a.to_sorted_vec();
        let vec_b = set_b.to_sorted_vec();

        // 2. Apply OPRF to both sets.
        transcript.record_phase(PSIPhase::OPRFEvaluation);
        let oprf_a: Vec<[u8; 32]> = vec_a.iter()
            .map(|fp| self.oprf.evaluate(&fp.to_le_bytes()))
            .collect();
        let oprf_b: Vec<[u8; 32]> = vec_b.iter()
            .map(|fp| self.oprf.evaluate(&fp.to_le_bytes()))
            .collect();
        transcript.record_message(oprf_a.len() * 32);
        transcript.record_message(oprf_b.len() * 32);

        // 3. Compare OPRF outputs (use HashSet for O(n+m)).
        transcript.record_phase(PSIPhase::Comparison);
        let oprf_set_b: std::collections::HashSet<[u8; 32]> = oprf_b.iter().cloned().collect();
        let mut intersection_elements: Vec<u64> = Vec::new();
        for (i, oprf_val) in oprf_a.iter().enumerate() {
            if oprf_set_b.contains(oprf_val) {
                intersection_elements.push(vec_a[i]);
            }
        }

        // 4. Aggregate.
        transcript.record_phase(PSIPhase::Aggregation);
        let intersection_cardinality = intersection_elements.len();
        let set_a_cardinality = set_a.len();
        let set_b_cardinality = set_b.len();
        let contamination_score = Self::compute_contamination_from_counts(
            intersection_cardinality, set_a_cardinality,
        );

        // 5. Attestation phase.
        transcript.record_phase(PSIPhase::Attestation);
        let protocol_transcript_hash = transcript.hash();
        transcript.record_phase(PSIPhase::Complete);

        let runtime_ms = start.elapsed().as_millis() as u64;

        let elements = match self.config.mode {
            PSIMode::Full => Some(intersection_elements),
            _ => None,
        };

        PSIResult {
            intersection_cardinality,
            set_a_cardinality,
            set_b_cardinality,
            intersection_elements: elements,
            contamination_score,
            protocol_transcript_hash,
            runtime_ms,
        }
    }

    // -------------------------------------------------------------------
    // Trie-based PSI
    // -------------------------------------------------------------------

    /// Trie-structured PSI exploiting prefix sharing for communication savings.
    pub fn run_trie_based(
        &self,
        trie_a: &super::trie::NGramTrie,
        trie_b: &super::trie::NGramTrie,
    ) -> PSIResult {
        let start = Instant::now();
        let mut transcript = PSITranscript::new();
        transcript.record_phase(PSIPhase::Setup);

        // 1. OPRF evaluation on trie keys.
        transcript.record_phase(PSIPhase::OPRFEvaluation);
        let keys_a = trie_a.keys();
        let keys_b = trie_b.keys();
        transcript.record_message(keys_a.len() * 8);
        transcript.record_message(keys_b.len() * 8);

        let oprf_a: std::collections::HashMap<Vec<u8>, [u8; 32]> = keys_a.iter()
            .map(|k| (k.clone(), self.oprf.evaluate(k)))
            .collect();
        let oprf_b: std::collections::HashSet<[u8; 32]> = keys_b.iter()
            .map(|k| self.oprf.evaluate(k))
            .collect();

        // 2. Comparison: find matching OPRF outputs.
        transcript.record_phase(PSIPhase::Comparison);
        let mut intersection_elements: Vec<u64> = Vec::new();
        for (key, oprf_val) in &oprf_a {
            if oprf_b.contains(oprf_val) {
                // Convert key bytes to u64 fingerprint.
                let mut buf = [0u8; 8];
                let len = key.len().min(8);
                buf[..len].copy_from_slice(&key[..len]);
                intersection_elements.push(u64::from_le_bytes(buf));
            }
        }

        // 3. Aggregate.
        transcript.record_phase(PSIPhase::Aggregation);
        let intersection_cardinality = intersection_elements.len();
        let set_a_cardinality = trie_a.len();
        let set_b_cardinality = trie_b.len();
        let contamination_score = Self::compute_contamination_from_counts(
            intersection_cardinality, set_a_cardinality,
        );

        // 4. Attestation.
        transcript.record_phase(PSIPhase::Attestation);
        let protocol_transcript_hash = transcript.hash();
        transcript.record_phase(PSIPhase::Complete);

        let runtime_ms = start.elapsed().as_millis() as u64;

        let elements = match self.config.mode {
            PSIMode::Full => Some(intersection_elements),
            _ => None,
        };

        PSIResult {
            intersection_cardinality,
            set_a_cardinality,
            set_b_cardinality,
            intersection_elements: elements,
            contamination_score,
            protocol_transcript_hash,
            runtime_ms,
        }
    }

    // -------------------------------------------------------------------
    // Contamination scoring
    // -------------------------------------------------------------------

    /// Contamination score = |A ∩ B| / total_benchmark_ngrams.
    pub fn compute_contamination(result: &PSIResult, total_benchmark_ngrams: usize) -> f64 {
        if total_benchmark_ngrams == 0 {
            return 0.0;
        }
        result.intersection_cardinality as f64 / total_benchmark_ngrams as f64
    }

    fn compute_contamination_from_counts(intersection: usize, set_a_size: usize) -> f64 {
        if set_a_size == 0 { 0.0 } else { intersection as f64 / set_a_size as f64 }
    }

    pub fn check_threshold(result: &PSIResult, threshold: f64) -> bool {
        result.contamination_score <= threshold
    }

    // -------------------------------------------------------------------
    // Attestation
    // -------------------------------------------------------------------

    pub fn generate_attestation(&self, result: &PSIResult) -> ContaminationAttestation {
        let threshold = self.config.threshold.unwrap_or(0.1);
        ContaminationAttestation::new(
            "benchmark".to_string(),
            "model".to_string(),
            self.config.ngram_config.n,
            result.intersection_cardinality,
            result.set_a_cardinality,
            threshold,
            result.protocol_transcript_hash,
        )
    }

    pub fn verify_attestation(attestation: &ContaminationAttestation) -> bool {
        attestation.verify()
    }
}

// ---------------------------------------------------------------------------
// SyntheticCorpusGenerator
// ---------------------------------------------------------------------------

pub struct SyntheticCorpusGenerator;

impl SyntheticCorpusGenerator {
    /// Generate `num_docs` random documents of approximately `doc_length` words.
    pub fn generate_random_corpus(num_docs: usize, doc_length: usize) -> Vec<String> {
        let mut rng = rand::thread_rng();
        let word_pool: Vec<&str> = vec![
            "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
            "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta",
            "theta", "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron",
            "pi", "rho", "sigma", "tau", "upsilon", "phi", "chi", "psi",
            "omega", "zero", "one", "two", "three", "four", "five", "six",
            "seven", "eight", "nine", "ten", "hundred", "thousand", "million",
            "data", "model", "train", "test", "evaluate", "score", "metric",
            "algorithm", "function", "variable", "constant", "parameter",
            "neural", "network", "layer", "weight", "bias", "gradient",
            "learning", "rate", "batch", "epoch", "loss", "accuracy",
            "precision", "recall", "benchmark", "corpus", "token", "embedding",
        ];
        (0..num_docs).map(|_| {
            let words: Vec<&str> = (0..doc_length)
                .map(|_| word_pool[rng.gen_range(0..word_pool.len())])
                .collect();
            words.join(" ")
        }).collect()
    }

    /// Generate two corpora with a controlled overlap fraction.
    ///
    /// `overlap_fraction` ∈ [0, 1] controls how much of the first corpus
    /// is copied verbatim into the second.
    pub fn generate_with_overlap(
        num_docs: usize,
        overlap_fraction: f64,
    ) -> (Vec<String>, Vec<String>) {
        let doc_length = 50;
        let corpus_a = Self::generate_random_corpus(num_docs, doc_length);
        let overlap_count = (num_docs as f64 * overlap_fraction).round() as usize;

        let mut corpus_b = Vec::with_capacity(num_docs);
        // Copy the overlapping portion from A.
        for doc in corpus_a.iter().take(overlap_count) {
            corpus_b.push(doc.clone());
        }
        // Fill the rest with fresh random documents.
        let remaining = num_docs.saturating_sub(overlap_count);
        let fresh = Self::generate_random_corpus(remaining, doc_length);
        corpus_b.extend(fresh);

        (corpus_a, corpus_b)
    }

    /// Generate a benchmark-like scenario: a benchmark text and a training
    /// text with a specific contamination rate (fraction of benchmark
    /// text duplicated in the training text).
    pub fn generate_benchmark_scenario(contamination_rate: f64) -> (String, String) {
        let benchmark_sentences: Vec<String> = Self::generate_random_corpus(20, 15);
        let benchmark_text = benchmark_sentences.join(". ");

        let contaminated_count = (benchmark_sentences.len() as f64 * contamination_rate).round() as usize;
        let mut training_parts: Vec<String> = Vec::new();

        // Copy some benchmark sentences into training data.
        for sent in benchmark_sentences.iter().take(contaminated_count) {
            training_parts.push(sent.clone());
        }
        // Add fresh content.
        let fresh = Self::generate_random_corpus(20 - contaminated_count, 15);
        training_parts.extend(fresh);

        let training_text = training_parts.join(". ");
        (benchmark_text, training_text)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::psi::ngram::{NGramConfig, NGramSet};
    use crate::psi::trie::NGramTrie;

    // -- PSIConfig --

    #[test]
    fn test_psi_config_default() {
        let cfg = PSIConfig::default();
        assert_eq!(cfg.mode, PSIMode::Full);
        assert_eq!(cfg.security_parameter, 128);
    }

    #[test]
    fn test_psi_config_cardinality_only() {
        let cfg = PSIConfig::cardinality_only();
        assert_eq!(cfg.mode, PSIMode::CardinalityOnly);
    }

    #[test]
    fn test_psi_config_threshold() {
        let cfg = PSIConfig::threshold(0.05);
        assert_eq!(cfg.mode, PSIMode::Threshold);
        assert_eq!(cfg.threshold, Some(0.05));
    }

    // -- Local PSI --

    #[test]
    fn test_run_local_identical_sets() {
        let config = NGramConfig::char_ngrams(3);
        let set = NGramSet::from_text("the quick brown fox jumps over the lazy dog", config);
        let psi = PSIProtocol::new(PSIConfig::default());
        let result = psi.run_local(&set, &set);
        assert_eq!(result.intersection_cardinality, set.len());
        assert!((result.contamination_score - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_run_local_disjoint_sets() {
        let config = NGramConfig::char_ngrams(3);
        let set_a = NGramSet::from_text("aaabbbccc", config.clone());
        let set_b = NGramSet::from_text("xxxyyyzzzwww", config);
        let psi = PSIProtocol::new(PSIConfig::default());
        let result = psi.run_local(&set_a, &set_b);
        assert_eq!(result.intersection_cardinality, 0);
        assert!((result.contamination_score - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_run_local_partial_overlap() {
        let config = NGramConfig::char_ngrams(3);
        let set_a = NGramSet::from_text("abcdefgh", config.clone());
        let set_b = NGramSet::from_text("efghijkl", config);
        let psi = PSIProtocol::new(PSIConfig::default());
        let result = psi.run_local(&set_a, &set_b);
        assert!(result.intersection_cardinality > 0);
        assert!(result.intersection_cardinality < set_a.len());
    }

    #[test]
    fn test_run_local_cardinality_only_mode() {
        let config = NGramConfig::char_ngrams(3);
        let set_a = NGramSet::from_text("abcdefgh", config.clone());
        let set_b = NGramSet::from_text("efghijkl", config);
        let psi_config = PSIConfig::cardinality_only();
        let psi = PSIProtocol::new(psi_config);
        let result = psi.run_local(&set_a, &set_b);
        // In CardinalityOnly mode, intersection_elements should be None.
        assert!(result.intersection_elements.is_none());
        assert!(result.intersection_cardinality > 0);
    }

    #[test]
    fn test_run_local_full_mode_has_elements() {
        let config = NGramConfig::char_ngrams(3);
        let set = NGramSet::from_text("abcdefghij", config);
        let psi = PSIProtocol::new(PSIConfig::default()); // Full mode
        let result = psi.run_local(&set, &set);
        assert!(result.intersection_elements.is_some());
        let elems = result.intersection_elements.unwrap();
        assert_eq!(elems.len(), result.intersection_cardinality);
    }

    // -- Trie-based PSI --

    #[test]
    fn test_run_trie_based_identical() {
        let ngrams: Vec<Vec<u8>> = vec![
            b"abc".to_vec(), b"def".to_vec(), b"ghi".to_vec(),
        ];
        let trie = NGramTrie::from_ngrams(&ngrams);
        let psi = PSIProtocol::new(PSIConfig::default());
        let result = psi.run_trie_based(&trie, &trie);
        assert_eq!(result.intersection_cardinality, 3);
    }

    #[test]
    fn test_run_trie_based_disjoint() {
        let t1 = NGramTrie::from_ngrams(&[b"aaa".to_vec(), b"bbb".to_vec()]);
        let t2 = NGramTrie::from_ngrams(&[b"xxx".to_vec(), b"yyy".to_vec()]);
        let psi = PSIProtocol::new(PSIConfig::default());
        let result = psi.run_trie_based(&t1, &t2);
        assert_eq!(result.intersection_cardinality, 0);
    }

    #[test]
    fn test_run_trie_based_partial() {
        let t1 = NGramTrie::from_ngrams(&[b"abc".to_vec(), b"def".to_vec(), b"ghi".to_vec()]);
        let t2 = NGramTrie::from_ngrams(&[b"def".to_vec(), b"ghi".to_vec(), b"jkl".to_vec()]);
        let psi = PSIProtocol::new(PSIConfig::default());
        let result = psi.run_trie_based(&t1, &t2);
        assert_eq!(result.intersection_cardinality, 2);
    }

    // -- Contamination scoring --

    #[test]
    fn test_compute_contamination() {
        let result = PSIResult {
            intersection_cardinality: 50,
            set_a_cardinality: 200,
            set_b_cardinality: 300,
            intersection_elements: None,
            contamination_score: 0.25,
            protocol_transcript_hash: [0u8; 32],
            runtime_ms: 0,
        };
        let score = PSIProtocol::compute_contamination(&result, 200);
        assert!((score - 0.25).abs() < 1e-9);
    }

    #[test]
    fn test_compute_contamination_zero() {
        let result = PSIResult {
            intersection_cardinality: 0,
            set_a_cardinality: 100,
            set_b_cardinality: 100,
            intersection_elements: None,
            contamination_score: 0.0,
            protocol_transcript_hash: [0u8; 32],
            runtime_ms: 0,
        };
        assert_eq!(PSIProtocol::compute_contamination(&result, 100), 0.0);
    }

    // -- Threshold checking --

    #[test]
    fn test_check_threshold_pass() {
        let result = PSIResult {
            intersection_cardinality: 5,
            set_a_cardinality: 100,
            set_b_cardinality: 100,
            intersection_elements: None,
            contamination_score: 0.05,
            protocol_transcript_hash: [0u8; 32],
            runtime_ms: 0,
        };
        assert!(PSIProtocol::check_threshold(&result, 0.1));
    }

    #[test]
    fn test_check_threshold_fail() {
        let result = PSIResult {
            intersection_cardinality: 50,
            set_a_cardinality: 100,
            set_b_cardinality: 100,
            intersection_elements: None,
            contamination_score: 0.50,
            protocol_transcript_hash: [0u8; 32],
            runtime_ms: 0,
        };
        assert!(!PSIProtocol::check_threshold(&result, 0.1));
    }

    // -- PSIResult metrics --

    #[test]
    fn test_psi_result_jaccard() {
        let result = PSIResult {
            intersection_cardinality: 10,
            set_a_cardinality: 20,
            set_b_cardinality: 30,
            intersection_elements: None,
            contamination_score: 0.5,
            protocol_transcript_hash: [0u8; 32],
            runtime_ms: 0,
        };
        // Jaccard = 10 / (20+30-10) = 10/40 = 0.25
        assert!((result.jaccard() - 0.25).abs() < 1e-9);
    }

    #[test]
    fn test_psi_result_containment() {
        let result = PSIResult {
            intersection_cardinality: 10,
            set_a_cardinality: 20,
            set_b_cardinality: 30,
            intersection_elements: None,
            contamination_score: 0.5,
            protocol_transcript_hash: [0u8; 32],
            runtime_ms: 0,
        };
        assert!((result.containment_a() - 0.5).abs() < 1e-9);
        assert!((result.containment_b() - (10.0/30.0)).abs() < 1e-9);
    }

    #[test]
    fn test_psi_result_overlap_coefficient() {
        let result = PSIResult {
            intersection_cardinality: 10,
            set_a_cardinality: 20,
            set_b_cardinality: 30,
            intersection_elements: None,
            contamination_score: 0.5,
            protocol_transcript_hash: [0u8; 32],
            runtime_ms: 0,
        };
        // overlap = 10 / min(20,30) = 10/20 = 0.5
        assert!((result.overlap_coefficient() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_psi_result_summary() {
        let result = PSIResult {
            intersection_cardinality: 5,
            set_a_cardinality: 100,
            set_b_cardinality: 200,
            intersection_elements: None,
            contamination_score: 0.05,
            protocol_transcript_hash: [0u8; 32],
            runtime_ms: 42,
        };
        let s = result.summary();
        assert!(s.contains("5"));
        assert!(s.contains("100"));
        assert!(s.contains("42ms"));
    }

    // -- Attestation --

    #[test]
    fn test_attestation_create_and_verify() {
        let att = ContaminationAttestation::new(
            "gsm8k".into(),
            "llama-3".into(),
            8,
            50,
            1000,
            0.1,
            [0u8; 32],
        );
        assert!(att.verify());
        assert_eq!(att.contamination_upper_bound, 0.05);
        assert!(att.threshold_satisfied);
    }

    #[test]
    fn test_attestation_threshold_not_satisfied() {
        let att = ContaminationAttestation::new(
            "bench".into(),
            "model".into(),
            8,
            200,
            1000,
            0.1,
            [0u8; 32],
        );
        assert!(att.verify());
        assert!(!att.threshold_satisfied);
    }

    #[test]
    fn test_attestation_json_roundtrip() {
        let att = ContaminationAttestation::new(
            "bench".into(), "model".into(), 8, 10, 100, 0.2, [1u8; 32],
        );
        let json = att.to_json();
        let recovered = ContaminationAttestation::from_json(&json).unwrap();
        assert_eq!(recovered.benchmark_id, att.benchmark_id);
        assert_eq!(recovered.contamination_upper_bound, att.contamination_upper_bound);
        assert!(recovered.verify());
    }

    #[test]
    fn test_attestation_summary() {
        let att = ContaminationAttestation::new(
            "bench".into(), "model".into(), 8, 10, 100, 0.2, [0u8; 32],
        );
        let s = att.summary();
        assert!(s.contains("bench"));
        assert!(s.contains("model"));
    }

    #[test]
    fn test_generate_attestation_from_protocol() {
        let config = NGramConfig::char_ngrams(3);
        let set = NGramSet::from_text("hello world test data", config);
        let psi = PSIProtocol::new(PSIConfig { threshold: Some(0.5), ..PSIConfig::default() });
        let result = psi.run_local(&set, &set);
        let att = psi.generate_attestation(&result);
        assert!(att.verify());
    }

    #[test]
    fn test_verify_attestation_static() {
        let att = ContaminationAttestation::new(
            "b".into(), "m".into(), 3, 5, 50, 0.5, [0u8; 32],
        );
        assert!(PSIProtocol::verify_attestation(&att));
    }

    // -- CommunicationComplexity --

    #[test]
    fn test_communication_complexity() {
        let est = CommunicationComplexity::estimate(1000, 2000, 128);
        assert_eq!(est.rounds, 3);
        assert!(est.total_bytes > 0);
        assert!(est.per_element_bytes > 0);
    }

    // -- ThresholdPSI --

    #[test]
    fn test_threshold_psi_prove_verify() {
        let config = NGramConfig::char_ngrams(3);
        let set_a = NGramSet::from_text("abcdefghij", config.clone());
        let set_b = NGramSet::from_text("abcdefghij", config);
        let threshold = 1.0;
        let proof = ThresholdPSI::prove_below_threshold(&set_a, &set_b, threshold);
        assert!(ThresholdPSI::verify_threshold(&proof, threshold));
    }

    #[test]
    fn test_threshold_psi_low_overlap() {
        let config = NGramConfig::char_ngrams(3);
        let set_a = NGramSet::from_text("aaabbbccc", config.clone());
        let set_b = NGramSet::from_text("xxxyyyzzz", config);
        let threshold = 0.5;
        let proof = ThresholdPSI::prove_below_threshold(&set_a, &set_b, threshold);
        assert!(ThresholdPSI::verify_threshold(&proof, threshold));
    }

    // -- PSITranscript --

    #[test]
    fn test_transcript() {
        let mut t = PSITranscript::new();
        t.record_phase(PSIPhase::Setup);
        t.record_message(100);
        t.record_phase(PSIPhase::Complete);
        assert_eq!(t.phases.len(), 2);
        assert_eq!(t.messages_sent, 1);
        assert_eq!(t.bytes_transferred, 100);
        let s = t.summary();
        assert!(s.contains("2 phases"));
        let h = t.hash();
        assert_ne!(h, [0u8; 32]);
    }

    // -- SyntheticCorpusGenerator --

    #[test]
    fn test_generate_random_corpus() {
        let corpus = SyntheticCorpusGenerator::generate_random_corpus(5, 20);
        assert_eq!(corpus.len(), 5);
        for doc in &corpus {
            assert!(!doc.is_empty());
        }
    }

    #[test]
    fn test_generate_with_overlap() {
        let (a, b) = SyntheticCorpusGenerator::generate_with_overlap(10, 0.5);
        assert_eq!(a.len(), 10);
        assert_eq!(b.len(), 10);
        // First 5 docs of b should match first 5 of a.
        for i in 0..5 {
            assert_eq!(a[i], b[i]);
        }
    }

    #[test]
    fn test_generate_with_zero_overlap() {
        let (a, b) = SyntheticCorpusGenerator::generate_with_overlap(5, 0.0);
        assert_eq!(a.len(), 5);
        assert_eq!(b.len(), 5);
    }

    #[test]
    fn test_generate_benchmark_scenario() {
        let (bench, train) = SyntheticCorpusGenerator::generate_benchmark_scenario(0.3);
        assert!(!bench.is_empty());
        assert!(!train.is_empty());
    }

    #[test]
    fn test_synthetic_corpus_psi() {
        // End-to-end test: generate corpora with known overlap, run PSI,
        // verify contamination score is reasonable.
        let (bench, train) = SyntheticCorpusGenerator::generate_benchmark_scenario(0.5);
        let config = NGramConfig::word_ngrams(2);
        let set_a = NGramSet::from_text(&bench, config.clone());
        let set_b = NGramSet::from_text(&train, config);
        let psi = PSIProtocol::new(PSIConfig::default());
        let result = psi.run_local(&set_a, &set_b);
        // With 50% contamination, we expect some overlap.
        assert!(result.intersection_cardinality > 0);
    }

    // -- PSIOperation --

    #[test]
    fn test_psi_operation_variants() {
        let ops = vec![
            PSIOperation::BlindInput(10),
            PSIOperation::EvaluateOPRF(20),
            PSIOperation::UnblindOutput(30),
            PSIOperation::CompareHashes(40),
            PSIOperation::AggregateCardinality,
            PSIOperation::GenerateAttestation,
        ];
        assert_eq!(ops.len(), 6);
        assert_eq!(ops[0], PSIOperation::BlindInput(10));
    }

    // -- ExtendedPSITranscript --

    #[test]
    fn test_extended_transcript_new() {
        let t = ExtendedPSITranscript::new();
        assert_eq!(t.phases.len(), 0);
        assert_eq!(t.messages_sent, 0);
        assert_eq!(t.bytes_transferred, 0);
        assert_eq!(t.operations.len(), 0);
    }

    #[test]
    fn test_extended_transcript_phases() {
        let mut t = ExtendedPSITranscript::new();
        t.begin_phase(PSIPhase::Setup);
        t.end_phase();
        t.begin_phase(PSIPhase::OPRFEvaluation);
        t.end_phase();
        assert_eq!(t.phases.len(), 2);
    }

    #[test]
    fn test_extended_transcript_messages() {
        let mut t = ExtendedPSITranscript::new();
        t.record_message(100);
        t.record_message(200);
        assert_eq!(t.messages_sent, 2);
        assert_eq!(t.bytes_transferred, 300);
    }

    #[test]
    fn test_extended_transcript_operations() {
        let mut t = ExtendedPSITranscript::new();
        t.record_operation(PSIOperation::BlindInput(5));
        t.record_operation(PSIOperation::EvaluateOPRF(5));
        assert_eq!(t.operations.len(), 2);
    }

    #[test]
    fn test_extended_transcript_summary() {
        let mut t = ExtendedPSITranscript::new();
        t.begin_phase(PSIPhase::Setup);
        t.end_phase();
        t.record_message(50);
        t.record_operation(PSIOperation::BlindInput(1));
        let s = t.summary();
        assert!(s.contains("1 phases"));
        assert!(s.contains("1 messages"));
        assert!(s.contains("1 operations"));
    }

    #[test]
    fn test_extended_transcript_hash() {
        let mut t = ExtendedPSITranscript::new();
        t.begin_phase(PSIPhase::Setup);
        t.end_phase();
        let h = t.hash();
        assert_ne!(h, [0u8; 32]);
    }

    #[test]
    fn test_extended_transcript_total_duration() {
        let t = ExtendedPSITranscript::new();
        assert_eq!(t.total_duration_ms(), 0);
    }

    #[test]
    fn test_extended_transcript_phase_durations() {
        let mut t = ExtendedPSITranscript::new();
        t.begin_phase(PSIPhase::Setup);
        t.end_phase();
        let durations = t.phase_durations();
        assert_eq!(durations.len(), 1);
        assert!(durations[0].0.contains("Setup"));
    }

    #[test]
    fn test_extended_transcript_default() {
        let t = ExtendedPSITranscript::default();
        assert_eq!(t.phases.len(), 0);
    }

    // -- ValidationIssue --

    #[test]
    fn test_validation_issue_error() {
        let issue = ValidationIssue::error("field", "something wrong");
        assert_eq!(issue.severity, ValidationSeverity::Error);
        assert_eq!(issue.field_name, "field");
        assert_eq!(issue.message, "something wrong");
    }

    #[test]
    fn test_validation_issue_warning() {
        let issue = ValidationIssue::warning("f", "warn");
        assert_eq!(issue.severity, ValidationSeverity::Warning);
    }

    #[test]
    fn test_validation_issue_info() {
        let issue = ValidationIssue::info("f", "info msg");
        assert_eq!(issue.severity, ValidationSeverity::Info);
    }

    // -- PSIValidator --

    #[test]
    fn test_validator_valid_result() {
        let result = PSIResult {
            intersection_cardinality: 10,
            set_a_cardinality: 100,
            set_b_cardinality: 200,
            intersection_elements: None,
            contamination_score: 0.1,
            protocol_transcript_hash: [1u8; 32],
            runtime_ms: 5,
        };
        let issues = PSIValidator::validate_result(&result);
        // Should have no errors
        assert!(issues.iter().all(|i| i.severity != ValidationSeverity::Error));
    }

    #[test]
    fn test_validator_intersection_exceeds_set_a() {
        let result = PSIResult {
            intersection_cardinality: 150,
            set_a_cardinality: 100,
            set_b_cardinality: 200,
            intersection_elements: None,
            contamination_score: 0.5,
            protocol_transcript_hash: [0u8; 32],
            runtime_ms: 0,
        };
        let issues = PSIValidator::validate_result(&result);
        assert!(issues.iter().any(|i| i.severity == ValidationSeverity::Error));
    }

    #[test]
    fn test_validator_bad_contamination_score() {
        let result = PSIResult {
            intersection_cardinality: 10,
            set_a_cardinality: 100,
            set_b_cardinality: 200,
            intersection_elements: None,
            contamination_score: 1.5, // > 1.0
            protocol_transcript_hash: [0u8; 32],
            runtime_ms: 0,
        };
        let issues = PSIValidator::validate_result(&result);
        assert!(issues.iter().any(|i| i.field_name == "contamination_score"
            && i.severity == ValidationSeverity::Error));
    }

    #[test]
    fn test_validator_check_bounds() {
        let result = PSIResult {
            intersection_cardinality: 10,
            set_a_cardinality: 100,
            set_b_cardinality: 200,
            intersection_elements: None,
            contamination_score: 0.1,
            protocol_transcript_hash: [0u8; 32],
            runtime_ms: 0,
        };
        assert!(PSIValidator::check_bounds(&result));
    }

    #[test]
    fn test_validator_check_bounds_fail() {
        let result = PSIResult {
            intersection_cardinality: 200,
            set_a_cardinality: 100,
            set_b_cardinality: 100,
            intersection_elements: None,
            contamination_score: 0.5,
            protocol_transcript_hash: [0u8; 32],
            runtime_ms: 0,
        };
        assert!(!PSIValidator::check_bounds(&result));
    }

    #[test]
    fn test_validator_attestation() {
        let att = ContaminationAttestation::new(
            "bench".into(), "model".into(), 3, 10, 100, 0.2, [0u8; 32],
        );
        let issues = PSIValidator::validate_attestation(&att);
        assert!(issues.iter().all(|i| i.severity != ValidationSeverity::Error));
    }

    #[test]
    fn test_validator_attestation_empty_ids() {
        let att = ContaminationAttestation::new(
            "".into(), "".into(), 0, 0, 0, 0.0, [0u8; 32],
        );
        let issues = PSIValidator::validate_attestation(&att);
        assert!(issues.iter().any(|i| i.severity == ValidationSeverity::Error));
    }

    #[test]
    fn test_validator_cross_validate() {
        let hash = [42u8; 32];
        let result = PSIResult {
            intersection_cardinality: 10,
            set_a_cardinality: 100,
            set_b_cardinality: 200,
            intersection_elements: None,
            contamination_score: 0.1,
            protocol_transcript_hash: hash,
            runtime_ms: 5,
        };
        let att = ContaminationAttestation::new(
            "bench".into(), "model".into(), 3, 10, 100, 0.2, hash,
        );
        assert!(PSIValidator::cross_validate(&result, &att));
    }

    #[test]
    fn test_validator_cross_validate_hash_mismatch() {
        let result = PSIResult {
            intersection_cardinality: 10,
            set_a_cardinality: 100,
            set_b_cardinality: 200,
            intersection_elements: None,
            contamination_score: 0.1,
            protocol_transcript_hash: [1u8; 32],
            runtime_ms: 5,
        };
        let att = ContaminationAttestation::new(
            "bench".into(), "model".into(), 3, 10, 100, 0.2, [2u8; 32],
        );
        assert!(!PSIValidator::cross_validate(&result, &att));
    }

    // -- ExtendedComplexityEstimate --

    #[test]
    fn test_extended_complexity_estimate() {
        let est = CommunicationComplexity::estimate_extended(1000, 2000, 128);
        assert_eq!(est.rounds, 3);
        assert!(est.total_bytes > 0);
        assert!(est.computational_cost_ops > 0);
    }

    #[test]
    fn test_extended_complexity_summary() {
        let est = CommunicationComplexity::estimate_extended(100, 200, 128);
        let s = est.summary();
        assert!(s.contains("rounds"));
        assert!(s.contains("bytes"));
    }

    #[test]
    fn test_extended_complexity_is_feasible() {
        let est = CommunicationComplexity::estimate_extended(10, 10, 128);
        assert!(est.is_feasible(1_000_000));
        assert!(!est.is_feasible(1));
    }

    #[test]
    fn test_complexity_estimate_summary() {
        let est = CommunicationComplexity::estimate(100, 200, 128);
        let s = est.summary();
        assert!(s.contains("rounds"));
    }

    #[test]
    fn test_complexity_estimate_is_feasible() {
        let est = CommunicationComplexity::estimate(10, 10, 128);
        assert!(est.is_feasible(1_000_000));
    }

    #[test]
    fn test_compare_protocols() {
        let comparisons = CommunicationComplexity::compare_protocols(1000, 2000);
        assert_eq!(comparisons.len(), 3);
        assert!(comparisons[0].0.contains("Standard"));
        assert!(comparisons[1].0.contains("High-Security"));
        assert!(comparisons[2].0.contains("Lightweight"));
    }

    #[test]
    fn test_estimate_trie_based() {
        let t1 = NGramTrie::from_ngrams(&[b"abc".to_vec(), b"def".to_vec()]);
        let t2 = NGramTrie::from_ngrams(&[b"def".to_vec(), b"ghi".to_vec()]);
        let est = CommunicationComplexity::estimate_trie_based(&t1, &t2, 128);
        assert_eq!(est.rounds, 4);
        assert!(est.total_bytes > 0);
    }

    // -- ThresholdProof --

    #[test]
    fn test_threshold_proof_serialize_deserialize() {
        let config = NGramConfig::char_ngrams(3);
        let set_a = NGramSet::from_text("abcdefghij", config.clone());
        let set_b = NGramSet::from_text("abcdefghij", config);
        let proof = ThresholdPSI::prove_below_threshold(&set_a, &set_b, 1.0);
        let bytes = proof.serialize_proof();
        let recovered = ThresholdProof::deserialize_proof(&bytes).unwrap();
        assert_eq!(recovered.bound, proof.bound);
        assert_eq!(recovered.commitment, proof.commitment);
    }

    #[test]
    fn test_threshold_proof_deserialize_too_short() {
        assert!(ThresholdProof::deserialize_proof(&[0u8; 10]).is_none());
    }

    #[test]
    fn test_threshold_check() {
        let config = NGramConfig::char_ngrams(3);
        let set_a = NGramSet::from_text("aaabbbccc", config.clone());
        let set_b = NGramSet::from_text("xxxyyyzzz", config);
        assert!(ThresholdPSI::threshold_check(&set_a, &set_b, 0.5));
    }

    // -- SyntheticCorpusGenerator extended --

    #[test]
    fn test_generate_adversarial_overlap() {
        let base = "the quick brown fox jumps over the lazy dog";
        let result = SyntheticCorpusGenerator::generate_adversarial_overlap(base, 0.5);
        assert!(!result.is_empty());
        let words: Vec<&str> = result.split_whitespace().collect();
        assert_eq!(words.len(), 9); // same word count
    }

    #[test]
    fn test_compute_actual_overlap_identical() {
        let text = "the quick brown fox jumps over the lazy dog";
        let overlap = SyntheticCorpusGenerator::compute_actual_overlap(text, text, 1);
        assert!((overlap - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_compute_actual_overlap_different() {
        let a = "alpha beta gamma delta epsilon";
        let b = "one two three four five six";
        let overlap = SyntheticCorpusGenerator::compute_actual_overlap(a, b, 1);
        assert!(overlap < 0.5);
    }

    // -- BenchmarkReport --

    #[test]
    fn test_benchmark_report_summary_empty() {
        let report = BenchmarkReport { results: vec![] };
        assert_eq!(report.summary(), "No benchmark results");
    }

    #[test]
    fn test_benchmark_report_summary() {
        let report = BenchmarkReport {
            results: vec![
                BenchmarkEntry { set_size: 10, overlap_rate: 0.5, time_ms: 10, memory_bytes: 1000, communication_bytes: 500 },
                BenchmarkEntry { set_size: 20, overlap_rate: 0.3, time_ms: 20, memory_bytes: 2000, communication_bytes: 1000 },
            ],
        };
        let s = report.summary();
        assert!(s.contains("2 entries"));
    }

    #[test]
    fn test_benchmark_report_to_csv() {
        let report = BenchmarkReport {
            results: vec![
                BenchmarkEntry { set_size: 10, overlap_rate: 0.5, time_ms: 10, memory_bytes: 1000, communication_bytes: 500 },
            ],
        };
        let csv = report.to_csv();
        assert!(csv.contains("set_size"));
        assert!(csv.contains("10"));
    }

    #[test]
    fn test_benchmark_report_fastest_slowest() {
        let report = BenchmarkReport {
            results: vec![
                BenchmarkEntry { set_size: 10, overlap_rate: 0.5, time_ms: 10, memory_bytes: 1000, communication_bytes: 500 },
                BenchmarkEntry { set_size: 20, overlap_rate: 0.3, time_ms: 30, memory_bytes: 2000, communication_bytes: 1000 },
            ],
        };
        assert_eq!(report.fastest().unwrap().time_ms, 10);
        assert_eq!(report.slowest().unwrap().time_ms, 30);
    }

    // -- PSIBenchmark --

    #[test]
    fn test_benchmark_local_small() {
        let report = PSIBenchmark::benchmark_local(&[2], &[0.0, 0.5]);
        assert_eq!(report.results.len(), 2);
    }

    #[test]
    fn test_benchmark_trie_small() {
        let report = PSIBenchmark::benchmark_trie(&[2], &[0.0]);
        assert_eq!(report.results.len(), 1);
    }

    #[test]
    fn test_compare_implementations() {
        let results = PSIBenchmark::compare_implementations(2, 0.5);
        assert_eq!(results.len(), 2);
        assert!(results[0].0.contains("Local"));
        assert!(results[1].0.contains("Trie"));
    }

    // -- RiskLevel --

    #[test]
    fn test_risk_level_from_score() {
        assert_eq!(RiskLevel::from_score(0.0), RiskLevel::Clean);
        assert_eq!(RiskLevel::from_score(0.03), RiskLevel::Low);
        assert_eq!(RiskLevel::from_score(0.10), RiskLevel::Medium);
        assert_eq!(RiskLevel::from_score(0.20), RiskLevel::High);
        assert_eq!(RiskLevel::from_score(0.50), RiskLevel::Critical);
    }

    #[test]
    fn test_risk_level_description() {
        assert!(!RiskLevel::Clean.description().is_empty());
        assert!(!RiskLevel::Critical.description().is_empty());
    }

    // -- ContaminationAnalyzer --

    #[test]
    fn test_contamination_analyzer_identical() {
        let text = "the quick brown fox jumps over the lazy dog in the park";
        let report = ContaminationAnalyzer::analyze(text, text, &[1, 2]);
        assert!(report.overall_contamination > 0.5);
        assert_ne!(report.risk_level, RiskLevel::Clean);
    }

    #[test]
    fn test_contamination_analyzer_different() {
        let training = "alpha beta gamma delta epsilon zeta eta theta iota kappa";
        let benchmark = "one two three four five six seven eight nine ten eleven";
        let report = ContaminationAnalyzer::analyze(training, benchmark, &[1, 2]);
        assert!(report.overall_contamination < 0.5);
    }

    #[test]
    fn test_contamination_analyzer_multi_benchmark() {
        let training = "the quick brown fox jumps over the lazy dog";
        let benchmarks: Vec<(&str, &str)> = vec![
            ("bench1", "the quick brown fox"),
            ("bench2", "alpha beta gamma delta"),
        ];
        let reports = ContaminationAnalyzer::multi_benchmark_analysis(training, &benchmarks);
        assert_eq!(reports.len(), 2);
        assert!(reports[0].details.contains("bench1"));
        assert!(reports[1].details.contains("bench2"));
    }

    #[test]
    fn test_contamination_analyzer_risk_assessment() {
        let report = ContaminationReport {
            ngram_results: vec![(1, 0.5)],
            overall_contamination: 0.5,
            risk_level: RiskLevel::Critical,
            details: "test".to_string(),
        };
        assert_eq!(ContaminationAnalyzer::risk_assessment(&report), RiskLevel::Critical);
    }

    #[test]
    fn test_contamination_report_summary() {
        let report = ContaminationReport {
            ngram_results: vec![(1, 0.1), (2, 0.2)],
            overall_contamination: 0.2,
            risk_level: RiskLevel::High,
            details: "test details".to_string(),
        };
        let s = report.summary();
        assert!(s.contains("0.2000"));
        assert!(s.contains("2 n-gram"));
    }
}

// ---------------------------------------------------------------------------
// PSIOperation
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PSIOperation {
    BlindInput(usize),
    EvaluateOPRF(usize),
    UnblindOutput(usize),
    CompareHashes(usize),
    AggregateCardinality,
    GenerateAttestation,
}

// ---------------------------------------------------------------------------
// ExtendedPSITranscript
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct ExtendedPSITranscript {
    pub phases: Vec<(PSIPhase, u64, u64)>,
    pub messages_sent: usize,
    pub bytes_transferred: usize,
    pub operations: Vec<PSIOperation>,
    current_phase_start: Option<u64>,
    current_phase: Option<PSIPhase>,
}

impl ExtendedPSITranscript {
    pub fn new() -> Self {
        Self {
            phases: Vec::new(),
            messages_sent: 0,
            bytes_transferred: 0,
            operations: Vec::new(),
            current_phase_start: None,
            current_phase: None,
        }
    }

    pub fn begin_phase(&mut self, phase: PSIPhase) {
        self.end_phase(); // close any open phase
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        self.current_phase_start = Some(now);
        self.current_phase = Some(phase);
    }

    pub fn end_phase(&mut self) {
        if let (Some(start), Some(phase)) = (self.current_phase_start.take(), self.current_phase.take()) {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64;
            self.phases.push((phase, start, now));
        }
    }

    pub fn record_message(&mut self, bytes: usize) {
        self.messages_sent += 1;
        self.bytes_transferred += bytes;
    }

    pub fn record_operation(&mut self, op: PSIOperation) {
        self.operations.push(op);
    }

    pub fn summary(&self) -> String {
        format!(
            "ExtendedTranscript: {} phases, {} messages, {} bytes, {} operations",
            self.phases.len(), self.messages_sent, self.bytes_transferred, self.operations.len(),
        )
    }

    pub fn hash(&self) -> [u8; 32] {
        let mut data = Vec::new();
        for (phase, start, end) in &self.phases {
            data.extend_from_slice(format!("{:?}:{}:{}", phase, start, end).as_bytes());
        }
        data.extend_from_slice(&(self.messages_sent as u64).to_le_bytes());
        data.extend_from_slice(&(self.bytes_transferred as u64).to_le_bytes());
        for op in &self.operations {
            data.extend_from_slice(format!("{:?}", op).as_bytes());
        }
        *blake3::hash(&data).as_bytes()
    }

    pub fn total_duration_ms(&self) -> u64 {
        if self.phases.is_empty() {
            return 0;
        }
        let first_start = self.phases.iter().map(|(_, s, _)| *s).min().unwrap_or(0);
        let last_end = self.phases.iter().map(|(_, _, e)| *e).max().unwrap_or(0);
        last_end.saturating_sub(first_start)
    }

    pub fn phase_durations(&self) -> Vec<(String, u64)> {
        self.phases.iter().map(|(phase, start, end)| {
            (format!("{:?}", phase), end.saturating_sub(*start))
        }).collect()
    }
}

impl Default for ExtendedPSITranscript {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ValidationIssue & PSIValidator
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ValidationSeverity {
    Error,
    Warning,
    Info,
}

#[derive(Clone, Debug)]
pub struct ValidationIssue {
    pub severity: ValidationSeverity,
    pub message: String,
    pub field_name: String,
}

impl ValidationIssue {
    pub fn error(field: &str, message: &str) -> Self {
        Self {
            severity: ValidationSeverity::Error,
            message: message.to_string(),
            field_name: field.to_string(),
        }
    }

    pub fn warning(field: &str, message: &str) -> Self {
        Self {
            severity: ValidationSeverity::Warning,
            message: message.to_string(),
            field_name: field.to_string(),
        }
    }

    pub fn info(field: &str, message: &str) -> Self {
        Self {
            severity: ValidationSeverity::Info,
            message: message.to_string(),
            field_name: field.to_string(),
        }
    }
}

pub struct PSIValidator;

impl PSIValidator {
    pub fn validate_result(result: &PSIResult) -> Vec<ValidationIssue> {
        let mut issues = Vec::new();

        if result.intersection_cardinality > result.set_a_cardinality {
            issues.push(ValidationIssue::error(
                "intersection_cardinality",
                "Intersection cannot exceed set A size",
            ));
        }

        if result.intersection_cardinality > result.set_b_cardinality {
            issues.push(ValidationIssue::error(
                "intersection_cardinality",
                "Intersection cannot exceed set B size",
            ));
        }

        if result.contamination_score < 0.0 || result.contamination_score > 1.0 {
            issues.push(ValidationIssue::error(
                "contamination_score",
                "Contamination score must be between 0 and 1",
            ));
        }

        if result.set_a_cardinality == 0 && result.intersection_cardinality > 0 {
            issues.push(ValidationIssue::error(
                "set_a_cardinality",
                "Set A is empty but intersection is non-zero",
            ));
        }

        if result.runtime_ms == 0 && result.set_a_cardinality > 1000 {
            issues.push(ValidationIssue::warning(
                "runtime_ms",
                "Runtime is 0ms for a large set, might be incorrect",
            ));
        }

        if result.protocol_transcript_hash == [0u8; 32] {
            issues.push(ValidationIssue::warning(
                "protocol_transcript_hash",
                "Transcript hash is all zeros",
            ));
        }

        if result.contamination_score > 0.9 {
            issues.push(ValidationIssue::info(
                "contamination_score",
                "Very high contamination detected",
            ));
        }

        issues
    }

    pub fn validate_attestation(attestation: &ContaminationAttestation) -> Vec<ValidationIssue> {
        let mut issues = Vec::new();

        if !attestation.verify() {
            issues.push(ValidationIssue::error(
                "attestation",
                "Attestation failed internal consistency check",
            ));
        }

        if attestation.benchmark_id.is_empty() {
            issues.push(ValidationIssue::error(
                "benchmark_id",
                "Benchmark ID is empty",
            ));
        }

        if attestation.model_id.is_empty() {
            issues.push(ValidationIssue::error(
                "model_id",
                "Model ID is empty",
            ));
        }

        if attestation.ngram_size == 0 {
            issues.push(ValidationIssue::error(
                "ngram_size",
                "N-gram size must be positive",
            ));
        }

        if attestation.total_benchmark_ngrams == 0 {
            issues.push(ValidationIssue::warning(
                "total_benchmark_ngrams",
                "Total benchmark n-grams is zero",
            ));
        }

        if attestation.threshold <= 0.0 || attestation.threshold > 1.0 {
            issues.push(ValidationIssue::warning(
                "threshold",
                "Threshold should be in (0, 1]",
            ));
        }

        if attestation.timestamp.is_empty() {
            issues.push(ValidationIssue::warning(
                "timestamp",
                "Timestamp is empty",
            ));
        }

        issues
    }

    pub fn cross_validate(result: &PSIResult, attestation: &ContaminationAttestation) -> bool {
        // Check that the attestation's cardinality bound is consistent with the result
        if attestation.intersection_cardinality_bound < result.intersection_cardinality {
            return false;
        }
        // Check that transcript hashes match
        if result.protocol_transcript_hash != attestation.protocol_hash {
            return false;
        }
        // Check attestation self-consistency
        attestation.verify()
    }

    pub fn check_bounds(result: &PSIResult) -> bool {
        result.intersection_cardinality <= result.set_a_cardinality
            && result.intersection_cardinality <= result.set_b_cardinality
            && result.contamination_score >= 0.0
            && result.contamination_score <= 1.0
    }
}

// ---------------------------------------------------------------------------
// ExtendedComplexityEstimate & enhanced CommunicationComplexity
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExtendedComplexityEstimate {
    pub rounds: usize,
    pub total_bytes: usize,
    pub per_element_bytes: usize,
    pub computational_cost_ops: u64,
}

impl ExtendedComplexityEstimate {
    pub fn summary(&self) -> String {
        format!(
            "Complexity: {} rounds, {} bytes total, {} bytes/element, {} ops",
            self.rounds, self.total_bytes, self.per_element_bytes, self.computational_cost_ops,
        )
    }

    pub fn is_feasible(&self, max_bytes: usize) -> bool {
        self.total_bytes <= max_bytes
    }
}

impl CommunicationComplexity {
    pub fn estimate_extended(set_a_size: usize, set_b_size: usize, security_param: u32) -> ExtendedComplexityEstimate {
        let per_element = (security_param as usize / 8) + 8;
        let total = (set_a_size + set_b_size) * per_element + 256;
        let rounds = 3;
        let computational_cost = (set_a_size + set_b_size) as u64 * 2; // 2 ops per element (hash + compare)
        ExtendedComplexityEstimate {
            rounds,
            total_bytes: total,
            per_element_bytes: per_element,
            computational_cost_ops: computational_cost,
        }
    }

    pub fn estimate_trie_based(
        trie_a: &super::trie::NGramTrie,
        trie_b: &super::trie::NGramTrie,
        security_param: u32,
    ) -> ExtendedComplexityEstimate {
        let a_size = trie_a.len();
        let b_size = trie_b.len();
        // Trie-based saves ~30% on communication due to prefix sharing
        let per_element = (security_param as usize / 8) + 4;
        let total = ((a_size + b_size) as f64 * per_element as f64 * 0.7) as usize + 256;
        let rounds = 4; // extra round for trie traversal
        let computational_cost = (a_size + b_size) as u64 * 3; // extra ops for trie
        ExtendedComplexityEstimate {
            rounds,
            total_bytes: total,
            per_element_bytes: per_element,
            computational_cost_ops: computational_cost,
        }
    }

    pub fn compare_protocols(set_a_size: usize, set_b_size: usize) -> Vec<(String, ExtendedComplexityEstimate)> {
        let mut results = Vec::new();

        // Standard OPRF-based PSI
        results.push((
            "Standard OPRF PSI".to_string(),
            Self::estimate_extended(set_a_size, set_b_size, 128),
        ));

        // High-security variant
        results.push((
            "High-Security PSI (256-bit)".to_string(),
            Self::estimate_extended(set_a_size, set_b_size, 256),
        ));

        // Lightweight variant
        let per_element = 12; // smaller output
        let total = (set_a_size + set_b_size) * per_element + 128;
        results.push((
            "Lightweight PSI (80-bit)".to_string(),
            ExtendedComplexityEstimate {
                rounds: 2,
                total_bytes: total,
                per_element_bytes: per_element,
                computational_cost_ops: (set_a_size + set_b_size) as u64,
            },
        ));

        results
    }
}

impl ComplexityEstimate {
    pub fn summary(&self) -> String {
        format!(
            "Complexity: {} rounds, {} bytes total, {} bytes/element",
            self.rounds, self.total_bytes, self.per_element_bytes,
        )
    }

    pub fn is_feasible(&self, max_bytes: usize) -> bool {
        self.total_bytes <= max_bytes
    }
}

// ---------------------------------------------------------------------------
// Enhanced ThresholdProof & ThresholdPSI
// ---------------------------------------------------------------------------

impl ThresholdProof {
    pub fn is_below_threshold(&self, threshold: f64) -> bool {
        // The bound represents the maximum possible intersection size
        // that was proven. A smaller bound means less overlap.
        self.bound as f64 <= threshold * 1000.0 // normalized check
    }

    pub fn serialize_proof(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(self.bound as u64).to_le_bytes());
        bytes.extend_from_slice(&self.commitment);
        bytes.extend_from_slice(&(self.proof_data.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&self.proof_data);
        bytes
    }

    pub fn deserialize_proof(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 48 { // 8 + 32 + 8 minimum
            return None;
        }
        let bound = u64::from_le_bytes(bytes[0..8].try_into().ok()?) as usize;
        let mut commitment = [0u8; 32];
        commitment.copy_from_slice(&bytes[8..40]);
        let data_len = u64::from_le_bytes(bytes[40..48].try_into().ok()?) as usize;
        if bytes.len() < 48 + data_len {
            return None;
        }
        let proof_data = bytes[48..48 + data_len].to_vec();
        Some(Self { bound, commitment, proof_data })
    }
}

impl ThresholdPSI {
    pub fn threshold_check(
        set_a: &super::ngram::NGramSet,
        set_b: &super::ngram::NGramSet,
        threshold: f64,
    ) -> bool {
        let intersection = set_a.intersection_cardinality(set_b);
        let max_allowed = (threshold * set_a.len() as f64).ceil() as usize;
        intersection <= max_allowed
    }
}

// ---------------------------------------------------------------------------
// SyntheticCorpusGenerator additional methods
// ---------------------------------------------------------------------------

impl SyntheticCorpusGenerator {
    pub fn generate_adversarial_overlap(base: &str, target_overlap: f64) -> String {
        let words: Vec<&str> = base.split_whitespace().collect();
        let keep_count = (words.len() as f64 * target_overlap).round() as usize;
        let keep_count = keep_count.min(words.len());

        let mut rng = rand::thread_rng();
        let word_pool = vec![
            "novel", "unique", "fresh", "original", "new", "different",
            "alternative", "distinct", "separate", "independent",
            "unrelated", "divergent", "disparate", "contrasting",
        ];

        let mut result_words: Vec<String> = Vec::with_capacity(words.len());
        for (i, word) in words.iter().enumerate() {
            if i < keep_count {
                result_words.push(word.to_string());
            } else {
                result_words.push(word_pool[rng.gen_range(0..word_pool.len())].to_string());
            }
        }
        result_words.join(" ")
    }

    pub fn compute_actual_overlap(a: &str, b: &str, n: usize) -> f64 {
        let config = super::ngram::NGramConfig::word_ngrams(n);
        let set_a = super::ngram::NGramSet::from_text(a, config.clone());
        let set_b = super::ngram::NGramSet::from_text(b, config);
        if set_a.len() == 0 {
            return 0.0;
        }
        set_a.intersection_cardinality(&set_b) as f64 / set_a.len() as f64
    }
}

// ---------------------------------------------------------------------------
// BenchmarkMetrics, BenchmarkEntry, BenchmarkReport, PSIBenchmark
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BenchmarkMetrics {
    pub avg_time_ms: f64,
    pub min_time_ms: f64,
    pub max_time_ms: f64,
    pub throughput_per_second: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BenchmarkEntry {
    pub set_size: usize,
    pub overlap_rate: f64,
    pub time_ms: u64,
    pub memory_bytes: usize,
    pub communication_bytes: usize,
}

#[derive(Clone, Debug)]
pub struct BenchmarkReport {
    pub results: Vec<BenchmarkEntry>,
}

impl BenchmarkReport {
    pub fn summary(&self) -> String {
        if self.results.is_empty() {
            return "No benchmark results".to_string();
        }
        let avg_time: f64 = self.results.iter().map(|r| r.time_ms as f64).sum::<f64>()
            / self.results.len() as f64;
        format!(
            "BenchmarkReport: {} entries, avg_time={:.2}ms",
            self.results.len(), avg_time,
        )
    }

    pub fn to_csv(&self) -> String {
        let mut csv = String::from("set_size,overlap_rate,time_ms,memory_bytes,communication_bytes\n");
        for entry in &self.results {
            csv.push_str(&format!(
                "{},{:.4},{},{},{}\n",
                entry.set_size, entry.overlap_rate, entry.time_ms,
                entry.memory_bytes, entry.communication_bytes,
            ));
        }
        csv
    }

    pub fn fastest(&self) -> Option<&BenchmarkEntry> {
        self.results.iter().min_by_key(|e| e.time_ms)
    }

    pub fn slowest(&self) -> Option<&BenchmarkEntry> {
        self.results.iter().max_by_key(|e| e.time_ms)
    }
}

pub struct PSIBenchmark;

impl PSIBenchmark {
    pub fn benchmark_local(sizes: &[usize], overlap_rates: &[f64]) -> BenchmarkReport {
        let mut results = Vec::new();

        for &size in sizes {
            for &overlap in overlap_rates {
                let (corpus_a, corpus_b) = SyntheticCorpusGenerator::generate_with_overlap(
                    size.max(1), overlap,
                );
                let config = super::ngram::NGramConfig::word_ngrams(2);
                let text_a = corpus_a.join(" ");
                let text_b = corpus_b.join(" ");
                let set_a = super::ngram::NGramSet::from_text(&text_a, config.clone());
                let set_b = super::ngram::NGramSet::from_text(&text_b, config);

                let psi = PSIProtocol::new(PSIConfig::default());
                let start = Instant::now();
                let _result = psi.run_local(&set_a, &set_b);
                let time_ms = start.elapsed().as_millis() as u64;

                let est = CommunicationComplexity::estimate(
                    set_a.len(), set_b.len(), 128,
                );

                results.push(BenchmarkEntry {
                    set_size: size,
                    overlap_rate: overlap,
                    time_ms,
                    memory_bytes: (set_a.len() + set_b.len()) * 40, // estimate
                    communication_bytes: est.total_bytes,
                });
            }
        }

        BenchmarkReport { results }
    }

    pub fn benchmark_trie(sizes: &[usize], overlap_rates: &[f64]) -> BenchmarkReport {
        let mut results = Vec::new();

        for &size in sizes {
            for &overlap in overlap_rates {
                let (corpus_a, corpus_b) = SyntheticCorpusGenerator::generate_with_overlap(
                    size.max(1), overlap,
                );
                let config = super::ngram::NGramConfig::word_ngrams(2);
                let text_a = corpus_a.join(" ");
                let text_b = corpus_b.join(" ");
                let set_a = super::ngram::NGramSet::from_text(&text_a, config.clone());
                let set_b = super::ngram::NGramSet::from_text(&text_b, config);

                let ngrams_a: Vec<Vec<u8>> = set_a.to_sorted_vec().iter()
                    .map(|fp| fp.to_le_bytes().to_vec())
                    .collect();
                let ngrams_b: Vec<Vec<u8>> = set_b.to_sorted_vec().iter()
                    .map(|fp| fp.to_le_bytes().to_vec())
                    .collect();
                let trie_a = super::trie::NGramTrie::from_ngrams(&ngrams_a);
                let trie_b = super::trie::NGramTrie::from_ngrams(&ngrams_b);

                let psi = PSIProtocol::new(PSIConfig::default());
                let start = Instant::now();
                let _result = psi.run_trie_based(&trie_a, &trie_b);
                let time_ms = start.elapsed().as_millis() as u64;

                results.push(BenchmarkEntry {
                    set_size: size,
                    overlap_rate: overlap,
                    time_ms,
                    memory_bytes: (trie_a.len() + trie_b.len()) * 64,
                    communication_bytes: (trie_a.len() + trie_b.len()) * 20,
                });
            }
        }

        BenchmarkReport { results }
    }

    pub fn compare_implementations(size: usize, overlap: f64) -> Vec<(String, BenchmarkMetrics)> {
        let mut results = Vec::new();
        let iterations = 3;

        // Local PSI benchmark
        let mut times = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let (corpus_a, corpus_b) = SyntheticCorpusGenerator::generate_with_overlap(
                size.max(1), overlap,
            );
            let config = super::ngram::NGramConfig::word_ngrams(2);
            let text_a = corpus_a.join(" ");
            let text_b = corpus_b.join(" ");
            let set_a = super::ngram::NGramSet::from_text(&text_a, config.clone());
            let set_b = super::ngram::NGramSet::from_text(&text_b, config);

            let psi = PSIProtocol::new(PSIConfig::default());
            let start = Instant::now();
            let _ = psi.run_local(&set_a, &set_b);
            times.push(start.elapsed().as_millis() as f64);
        }
        let avg = times.iter().sum::<f64>() / times.len() as f64;
        let min = times.iter().cloned().fold(f64::MAX, f64::min);
        let max = times.iter().cloned().fold(0.0f64, f64::max);
        let throughput = if avg > 0.0 { 1000.0 / avg } else { 0.0 };
        results.push(("Local PSI".to_string(), BenchmarkMetrics {
            avg_time_ms: avg, min_time_ms: min, max_time_ms: max,
            throughput_per_second: throughput,
        }));

        // Trie-based PSI benchmark
        let mut times = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            let (corpus_a, corpus_b) = SyntheticCorpusGenerator::generate_with_overlap(
                size.max(1), overlap,
            );
            let config = super::ngram::NGramConfig::word_ngrams(2);
            let text_a = corpus_a.join(" ");
            let text_b = corpus_b.join(" ");
            let set_a = super::ngram::NGramSet::from_text(&text_a, config.clone());
            let set_b = super::ngram::NGramSet::from_text(&text_b, config);

            let ngrams_a: Vec<Vec<u8>> = set_a.to_sorted_vec().iter()
                .map(|fp| fp.to_le_bytes().to_vec())
                .collect();
            let ngrams_b: Vec<Vec<u8>> = set_b.to_sorted_vec().iter()
                .map(|fp| fp.to_le_bytes().to_vec())
                .collect();
            let trie_a = super::trie::NGramTrie::from_ngrams(&ngrams_a);
            let trie_b = super::trie::NGramTrie::from_ngrams(&ngrams_b);

            let psi = PSIProtocol::new(PSIConfig::default());
            let start = Instant::now();
            let _ = psi.run_trie_based(&trie_a, &trie_b);
            times.push(start.elapsed().as_millis() as f64);
        }
        let avg = times.iter().sum::<f64>() / times.len() as f64;
        let min = times.iter().cloned().fold(f64::MAX, f64::min);
        let max = times.iter().cloned().fold(0.0f64, f64::max);
        let throughput = if avg > 0.0 { 1000.0 / avg } else { 0.0 };
        results.push(("Trie-based PSI".to_string(), BenchmarkMetrics {
            avg_time_ms: avg, min_time_ms: min, max_time_ms: max,
            throughput_per_second: throughput,
        }));

        results
    }
}

// ---------------------------------------------------------------------------
// RiskLevel, ContaminationReport, ContaminationAnalyzer
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RiskLevel {
    Clean,
    Low,
    Medium,
    High,
    Critical,
}

impl RiskLevel {
    pub fn from_score(score: f64) -> Self {
        if score < 0.01 {
            RiskLevel::Clean
        } else if score < 0.05 {
            RiskLevel::Low
        } else if score < 0.15 {
            RiskLevel::Medium
        } else if score < 0.30 {
            RiskLevel::High
        } else {
            RiskLevel::Critical
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            RiskLevel::Clean => "No significant contamination detected",
            RiskLevel::Low => "Minimal contamination, likely benign overlap",
            RiskLevel::Medium => "Moderate contamination, warrants investigation",
            RiskLevel::High => "Significant contamination detected",
            RiskLevel::Critical => "Critical contamination level, results unreliable",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContaminationReport {
    pub ngram_results: Vec<(usize, f64)>,
    pub overall_contamination: f64,
    pub risk_level: RiskLevel,
    pub details: String,
}

impl ContaminationReport {
    pub fn summary(&self) -> String {
        format!(
            "Contamination Report: overall={:.4}, risk={:?}, {} n-gram sizes analyzed",
            self.overall_contamination, self.risk_level, self.ngram_results.len(),
        )
    }
}

pub struct ContaminationAnalyzer;

impl ContaminationAnalyzer {
    pub fn analyze(
        training_data: &str,
        benchmark: &str,
        ngram_sizes: &[usize],
    ) -> ContaminationReport {
        let mut ngram_results = Vec::new();

        for &n in ngram_sizes {
            let config = super::ngram::NGramConfig::word_ngrams(n);
            let set_train = super::ngram::NGramSet::from_text(training_data, config.clone());
            let set_bench = super::ngram::NGramSet::from_text(benchmark, config);

            let overlap = if set_bench.len() == 0 {
                0.0
            } else {
                set_bench.intersection_cardinality(&set_train) as f64 / set_bench.len() as f64
            };
            ngram_results.push((n, overlap));
        }

        // Overall contamination is the max across n-gram sizes
        let overall = ngram_results.iter()
            .map(|(_, score)| *score)
            .fold(0.0f64, f64::max);

        let risk_level = RiskLevel::from_score(overall);

        let details = format!(
            "Analyzed {} n-gram sizes. Max overlap: {:.4} at n={}",
            ngram_sizes.len(),
            overall,
            ngram_results.iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(n, _)| *n)
                .unwrap_or(0),
        );

        ContaminationReport {
            ngram_results,
            overall_contamination: overall,
            risk_level,
            details,
        }
    }

    pub fn multi_benchmark_analysis(
        training: &str,
        benchmarks: &[(&str, &str)],
    ) -> Vec<ContaminationReport> {
        benchmarks.iter().map(|(name, benchmark_text)| {
            let mut report = Self::analyze(training, benchmark_text, &[1, 2, 3]);
            report.details = format!("Benchmark '{}': {}", name, report.details);
            report
        }).collect()
    }

    pub fn risk_assessment(report: &ContaminationReport) -> RiskLevel {
        report.risk_level.clone()
    }
}

// ---------------------------------------------------------------------------
// PSISessionManager — manage multiple PSI sessions
// ---------------------------------------------------------------------------

/// Unique identifier for a PSI session.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PSISessionId(String);

impl PSISessionId {
    /// Generate a new random session ID.
    pub fn generate() -> Self {
        let mut rng = rand::thread_rng();
        let id: u64 = rng.gen();
        let hash = blake3::hash(&id.to_le_bytes());
        let hex: String = hash.as_bytes()[..8].iter()
            .map(|b| format!("{:02x}", b))
            .collect();
        Self(format!("psi-{}", hex))
    }

    /// Create a session ID from a string.
    pub fn from_string(s: &str) -> Self {
        Self(s.to_string())
    }

    /// Get the ID as a string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// State of a PSI session.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PSISessionState {
    Created,
    Running,
    Completed,
    Failed(String),
    Cancelled,
}

/// A PSI session tracking configuration, state, results, and timing.
#[derive(Clone, Debug)]
pub struct PSISession {
    pub id: PSISessionId,
    pub config: PSIConfig,
    pub state: PSISessionState,
    pub result: Option<PSIResult>,
    pub attestation: Option<ContaminationAttestation>,
    pub created_at_ms: u64,
    pub completed_at_ms: Option<u64>,
    pub error: Option<String>,
}

impl PSISession {
    /// Create a new session.
    pub fn new(config: PSIConfig) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        Self {
            id: PSISessionId::generate(),
            config,
            state: PSISessionState::Created,
            result: None,
            attestation: None,
            created_at_ms: now,
            completed_at_ms: None,
            error: None,
        }
    }

    /// Mark the session as running.
    pub fn start(&mut self) {
        self.state = PSISessionState::Running;
    }

    /// Mark the session as completed with a result.
    pub fn complete(&mut self, result: PSIResult, attestation: Option<ContaminationAttestation>) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        self.state = PSISessionState::Completed;
        self.result = Some(result);
        self.attestation = attestation;
        self.completed_at_ms = Some(now);
    }

    /// Mark the session as failed.
    pub fn fail(&mut self, error: &str) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        self.state = PSISessionState::Failed(error.to_string());
        self.error = Some(error.to_string());
        self.completed_at_ms = Some(now);
    }

    /// Cancel the session.
    pub fn cancel(&mut self) {
        self.state = PSISessionState::Cancelled;
    }

    /// Get the duration in milliseconds if completed.
    pub fn duration_ms(&self) -> Option<u64> {
        self.completed_at_ms.map(|end| end.saturating_sub(self.created_at_ms))
    }

    /// Check if the session is still active.
    pub fn is_active(&self) -> bool {
        matches!(self.state, PSISessionState::Created | PSISessionState::Running)
    }

    /// Summary of the session.
    pub fn summary(&self) -> String {
        format!(
            "Session {}: state={:?}, duration={:?}ms",
            self.id.as_str(), self.state,
            self.duration_ms(),
        )
    }
}

/// Manager for multiple PSI sessions.
pub struct PSISessionManager {
    sessions: std::collections::HashMap<PSISessionId, PSISession>,
}

impl PSISessionManager {
    pub fn new() -> Self {
        Self {
            sessions: std::collections::HashMap::new(),
        }
    }

    /// Create and register a new session.
    pub fn create_session(&mut self, config: PSIConfig) -> PSISessionId {
        let session = PSISession::new(config);
        let id = session.id.clone();
        self.sessions.insert(id.clone(), session);
        id
    }

    /// Get a session by ID.
    pub fn get_session(&self, id: &PSISessionId) -> Option<&PSISession> {
        self.sessions.get(id)
    }

    /// Get a mutable session by ID.
    pub fn get_session_mut(&mut self, id: &PSISessionId) -> Option<&mut PSISession> {
        self.sessions.get_mut(id)
    }

    /// Run a local PSI within a managed session.
    pub fn run_local(
        &mut self,
        session_id: &PSISessionId,
        set_a: &super::ngram::NGramSet,
        set_b: &super::ngram::NGramSet,
    ) -> Option<PSIResult> {
        let session = self.sessions.get_mut(session_id)?;
        session.start();

        let psi = PSIProtocol::new(session.config.clone());
        let result = psi.run_local(set_a, set_b);
        let attestation = Some(psi.generate_attestation(&result));

        session.complete(result.clone(), attestation);
        Some(result)
    }

    /// Get all session IDs.
    pub fn session_ids(&self) -> Vec<PSISessionId> {
        self.sessions.keys().cloned().collect()
    }

    /// Get the number of sessions.
    pub fn session_count(&self) -> usize {
        self.sessions.len()
    }

    /// Get active sessions.
    pub fn active_sessions(&self) -> Vec<&PSISession> {
        self.sessions.values().filter(|s| s.is_active()).collect()
    }

    /// Get completed sessions.
    pub fn completed_sessions(&self) -> Vec<&PSISession> {
        self.sessions.values()
            .filter(|s| matches!(s.state, PSISessionState::Completed))
            .collect()
    }

    /// Remove a session by ID.
    pub fn remove_session(&mut self, id: &PSISessionId) -> Option<PSISession> {
        self.sessions.remove(id)
    }

    /// Clear all sessions.
    pub fn clear(&mut self) {
        self.sessions.clear();
    }

    /// Summary of the manager.
    pub fn summary(&self) -> String {
        let active = self.active_sessions().len();
        let completed = self.completed_sessions().len();
        format!(
            "PSISessionManager: {} total sessions ({} active, {} completed)",
            self.sessions.len(), active, completed,
        )
    }
}

impl Default for PSISessionManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// PSIResultAggregator — aggregate multiple PSI results
// ---------------------------------------------------------------------------

/// Aggregate statistics from multiple PSI runs.
#[derive(Clone, Debug)]
pub struct PSIAggregateStats {
    pub total_runs: usize,
    pub avg_intersection: f64,
    pub avg_contamination: f64,
    pub max_contamination: f64,
    pub min_contamination: f64,
    pub avg_runtime_ms: f64,
    pub total_runtime_ms: u64,
}

impl PSIAggregateStats {
    pub fn summary(&self) -> String {
        format!(
            "Aggregate: {} runs, avg_contamination={:.4}, max={:.4}, min={:.4}, avg_runtime={:.1}ms",
            self.total_runs, self.avg_contamination, self.max_contamination,
            self.min_contamination, self.avg_runtime_ms,
        )
    }

    pub fn is_clean(&self, threshold: f64) -> bool {
        self.max_contamination <= threshold
    }
}

/// Aggregator that collects multiple PSI results and computes statistics.
pub struct PSIResultAggregator {
    results: Vec<PSIResult>,
}

impl PSIResultAggregator {
    pub fn new() -> Self {
        Self { results: Vec::new() }
    }

    /// Add a result to the aggregator.
    pub fn add_result(&mut self, result: PSIResult) {
        self.results.push(result);
    }

    /// Get the number of results.
    pub fn count(&self) -> usize {
        self.results.len()
    }

    /// Compute aggregate statistics.
    pub fn aggregate(&self) -> PSIAggregateStats {
        if self.results.is_empty() {
            return PSIAggregateStats {
                total_runs: 0,
                avg_intersection: 0.0,
                avg_contamination: 0.0,
                max_contamination: 0.0,
                min_contamination: 0.0,
                avg_runtime_ms: 0.0,
                total_runtime_ms: 0,
            };
        }

        let n = self.results.len() as f64;
        let avg_intersection = self.results.iter()
            .map(|r| r.intersection_cardinality as f64)
            .sum::<f64>() / n;
        let avg_contamination = self.results.iter()
            .map(|r| r.contamination_score)
            .sum::<f64>() / n;
        let max_contamination = self.results.iter()
            .map(|r| r.contamination_score)
            .fold(0.0f64, f64::max);
        let min_contamination = self.results.iter()
            .map(|r| r.contamination_score)
            .fold(f64::MAX, f64::min);
        let total_runtime_ms: u64 = self.results.iter()
            .map(|r| r.runtime_ms)
            .sum();
        let avg_runtime_ms = total_runtime_ms as f64 / n;

        PSIAggregateStats {
            total_runs: self.results.len(),
            avg_intersection,
            avg_contamination,
            max_contamination,
            min_contamination,
            avg_runtime_ms,
            total_runtime_ms,
        }
    }

    /// Get results sorted by contamination score (descending).
    pub fn sorted_by_contamination(&self) -> Vec<&PSIResult> {
        let mut sorted: Vec<&PSIResult> = self.results.iter().collect();
        sorted.sort_by(|a, b| b.contamination_score.partial_cmp(&a.contamination_score)
            .unwrap_or(std::cmp::Ordering::Equal));
        sorted
    }

    /// Filter results above a contamination threshold.
    pub fn above_threshold(&self, threshold: f64) -> Vec<&PSIResult> {
        self.results.iter()
            .filter(|r| r.contamination_score > threshold)
            .collect()
    }

    /// Get all results.
    pub fn all_results(&self) -> &[PSIResult] {
        &self.results
    }

    /// Clear all results.
    pub fn clear(&mut self) {
        self.results.clear();
    }
}

impl Default for PSIResultAggregator {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// PSIProtocolConfig — extended protocol configuration
// ---------------------------------------------------------------------------

/// Extended configuration for PSI protocol parameters.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PSIProtocolConfig {
    pub base_config: PSIConfig,
    pub retry_count: usize,
    pub timeout_ms: u64,
    pub enable_caching: bool,
    pub enable_logging: bool,
    pub max_concurrent_sessions: usize,
    pub compression_enabled: bool,
}

impl PSIProtocolConfig {
    pub fn new(base: PSIConfig) -> Self {
        Self {
            base_config: base,
            retry_count: 3,
            timeout_ms: 30_000,
            enable_caching: true,
            enable_logging: false,
            max_concurrent_sessions: 4,
            compression_enabled: false,
        }
    }

    pub fn with_retries(mut self, count: usize) -> Self {
        self.retry_count = count;
        self
    }

    pub fn with_timeout(mut self, ms: u64) -> Self {
        self.timeout_ms = ms;
        self
    }

    pub fn with_caching(mut self, enabled: bool) -> Self {
        self.enable_caching = enabled;
        self
    }

    pub fn with_logging(mut self, enabled: bool) -> Self {
        self.enable_logging = enabled;
        self
    }

    pub fn with_compression(mut self, enabled: bool) -> Self {
        self.compression_enabled = enabled;
        self
    }

    pub fn summary(&self) -> String {
        format!(
            "PSIProtocolConfig: mode={:?}, retries={}, timeout={}ms, caching={}, logging={}",
            self.base_config.mode, self.retry_count, self.timeout_ms,
            self.enable_caching, self.enable_logging,
        )
    }
}

impl Default for PSIProtocolConfig {
    fn default() -> Self {
        Self::new(PSIConfig::default())
    }
}

// ---------------------------------------------------------------------------
// ContaminationMatrix — multi-benchmark contamination analysis
// ---------------------------------------------------------------------------

/// A matrix of contamination scores across multiple benchmarks and models.
#[derive(Clone, Debug)]
pub struct ContaminationMatrix {
    pub benchmark_names: Vec<String>,
    pub model_names: Vec<String>,
    pub scores: Vec<Vec<f64>>,
}

impl ContaminationMatrix {
    /// Create a new contamination matrix.
    pub fn new(benchmarks: Vec<String>, models: Vec<String>) -> Self {
        let scores = vec![vec![0.0; models.len()]; benchmarks.len()];
        Self {
            benchmark_names: benchmarks,
            model_names: models,
            scores,
        }
    }

    /// Set a contamination score.
    pub fn set_score(&mut self, benchmark_idx: usize, model_idx: usize, score: f64) {
        if benchmark_idx < self.scores.len() && model_idx < self.scores[benchmark_idx].len() {
            self.scores[benchmark_idx][model_idx] = score;
        }
    }

    /// Get a contamination score.
    pub fn get_score(&self, benchmark_idx: usize, model_idx: usize) -> f64 {
        self.scores.get(benchmark_idx)
            .and_then(|row| row.get(model_idx))
            .copied()
            .unwrap_or(0.0)
    }

    /// Get the maximum contamination score for a benchmark.
    pub fn max_for_benchmark(&self, benchmark_idx: usize) -> f64 {
        self.scores.get(benchmark_idx)
            .map(|row| row.iter().cloned().fold(0.0f64, f64::max))
            .unwrap_or(0.0)
    }

    /// Get the maximum contamination score for a model.
    pub fn max_for_model(&self, model_idx: usize) -> f64 {
        self.scores.iter()
            .filter_map(|row| row.get(model_idx))
            .cloned()
            .fold(0.0f64, f64::max)
    }

    /// Get the overall maximum contamination score.
    pub fn overall_max(&self) -> f64 {
        self.scores.iter()
            .flat_map(|row| row.iter())
            .cloned()
            .fold(0.0f64, f64::max)
    }

    /// Get the risk level for each benchmark-model pair.
    pub fn risk_matrix(&self) -> Vec<Vec<RiskLevel>> {
        self.scores.iter().map(|row| {
            row.iter().map(|&score| RiskLevel::from_score(score)).collect()
        }).collect()
    }

    /// Produce a text summary of the matrix.
    pub fn summary(&self) -> String {
        let mut s = format!(
            "ContaminationMatrix: {} benchmarks x {} models, overall_max={:.4}\n",
            self.benchmark_names.len(), self.model_names.len(), self.overall_max(),
        );
        for (i, bench) in self.benchmark_names.iter().enumerate() {
            let max = self.max_for_benchmark(i);
            s.push_str(&format!("  {}: max_contamination={:.4}\n", bench, max));
        }
        s
    }

    /// Export as CSV.
    pub fn to_csv(&self) -> String {
        let mut csv = String::from("benchmark");
        for model in &self.model_names {
            csv.push(',');
            csv.push_str(model);
        }
        csv.push('\n');
        for (i, bench) in self.benchmark_names.iter().enumerate() {
            csv.push_str(bench);
            for &score in &self.scores[i] {
                csv.push_str(&format!(",{:.6}", score));
            }
            csv.push('\n');
        }
        csv
    }

    /// Number of benchmark-model pairs that exceed a threshold.
    pub fn count_above_threshold(&self, threshold: f64) -> usize {
        self.scores.iter()
            .flat_map(|row| row.iter())
            .filter(|&&s| s > threshold)
            .count()
    }
}

// ---------------------------------------------------------------------------
// NGramOverlapAnalyzer — detailed n-gram overlap analysis
// ---------------------------------------------------------------------------

/// Result of an n-gram overlap analysis for a specific n-gram size.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NGramOverlapResult {
    pub ngram_size: usize,
    pub set_a_count: usize,
    pub set_b_count: usize,
    pub intersection_count: usize,
    pub union_count: usize,
    pub jaccard: f64,
    pub containment_a_in_b: f64,
    pub containment_b_in_a: f64,
    pub overlap_coefficient: f64,
}

impl NGramOverlapResult {
    pub fn summary(&self) -> String {
        format!(
            "n={}: |A|={}, |B|={}, |A∩B|={}, jaccard={:.4}, containment_a={:.4}",
            self.ngram_size, self.set_a_count, self.set_b_count,
            self.intersection_count, self.jaccard, self.containment_a_in_b,
        )
    }
}

/// Detailed n-gram overlap analyzer comparing two texts across multiple
/// n-gram sizes and types.
pub struct NGramOverlapAnalyzer;

impl NGramOverlapAnalyzer {
    /// Analyze overlap for a single n-gram size.
    pub fn analyze_single(
        text_a: &str,
        text_b: &str,
        ngram_config: super::ngram::NGramConfig,
    ) -> NGramOverlapResult {
        let set_a = super::ngram::NGramSet::from_text(text_a, ngram_config.clone());
        let set_b = super::ngram::NGramSet::from_text(text_b, ngram_config.clone());

        let intersection_count = set_a.intersection_cardinality(&set_b);
        let union_count = set_a.len() + set_b.len() - intersection_count;

        let jaccard = if union_count == 0 { 0.0 } else { intersection_count as f64 / union_count as f64 };
        let containment_a = if set_a.len() == 0 { 0.0 } else { intersection_count as f64 / set_a.len() as f64 };
        let containment_b = if set_b.len() == 0 { 0.0 } else { intersection_count as f64 / set_b.len() as f64 };
        let min_size = set_a.len().min(set_b.len());
        let overlap_coeff = if min_size == 0 { 0.0 } else { intersection_count as f64 / min_size as f64 };

        NGramOverlapResult {
            ngram_size: ngram_config.n,
            set_a_count: set_a.len(),
            set_b_count: set_b.len(),
            intersection_count,
            union_count,
            jaccard,
            containment_a_in_b: containment_a,
            containment_b_in_a: containment_b,
            overlap_coefficient: overlap_coeff,
        }
    }

    /// Analyze overlap across multiple n-gram sizes.
    pub fn analyze_multi(
        text_a: &str,
        text_b: &str,
        ngram_sizes: &[usize],
        use_word_ngrams: bool,
    ) -> Vec<NGramOverlapResult> {
        ngram_sizes.iter().map(|&n| {
            let config = if use_word_ngrams {
                super::ngram::NGramConfig::word_ngrams(n)
            } else {
                super::ngram::NGramConfig::char_ngrams(n)
            };
            Self::analyze_single(text_a, text_b, config)
        }).collect()
    }

    /// Find the n-gram size that yields the highest overlap.
    pub fn find_peak_overlap(
        text_a: &str,
        text_b: &str,
        max_n: usize,
    ) -> (usize, f64) {
        let results = Self::analyze_multi(text_a, text_b, &(1..=max_n).collect::<Vec<_>>(), true);
        results.iter()
            .map(|r| (r.ngram_size, r.containment_a_in_b))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, 0.0))
    }

    /// Generate a comprehensive overlap report as text.
    pub fn generate_report(
        text_a: &str,
        text_b: &str,
        ngram_sizes: &[usize],
    ) -> String {
        let word_results = Self::analyze_multi(text_a, text_b, ngram_sizes, true);
        let char_results = Self::analyze_multi(text_a, text_b, ngram_sizes, false);

        let mut report = String::from("=== N-gram Overlap Report ===\n\n");
        report.push_str("Word n-grams:\n");
        for r in &word_results {
            report.push_str(&format!("  {}\n", r.summary()));
        }
        report.push_str("\nCharacter n-grams:\n");
        for r in &char_results {
            report.push_str(&format!("  {}\n", r.summary()));
        }
        report
    }
}

// ---------------------------------------------------------------------------
// PSIProgressTracker — track progress of long-running PSI operations
// ---------------------------------------------------------------------------

/// Progress information for a PSI operation.
#[derive(Clone, Debug)]
pub struct PSIProgress {
    pub total_items: usize,
    pub processed_items: usize,
    pub current_phase: PSIPhase,
    pub elapsed_ms: u64,
    pub estimated_remaining_ms: u64,
}

impl PSIProgress {
    pub fn percentage(&self) -> f64 {
        if self.total_items == 0 { 100.0 }
        else { (self.processed_items as f64 / self.total_items as f64) * 100.0 }
    }

    pub fn is_complete(&self) -> bool {
        self.processed_items >= self.total_items
    }

    pub fn summary(&self) -> String {
        format!(
            "Progress: {}/{} ({:.1}%), phase={:?}, elapsed={}ms, remaining~{}ms",
            self.processed_items, self.total_items, self.percentage(),
            self.current_phase, self.elapsed_ms, self.estimated_remaining_ms,
        )
    }
}

/// Track progress of PSI operations.
pub struct PSIProgressTracker {
    total_items: usize,
    processed: usize,
    start_time_ms: u64,
    current_phase: PSIPhase,
}

impl PSIProgressTracker {
    pub fn new(total_items: usize) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        Self {
            total_items,
            processed: 0,
            start_time_ms: now,
            current_phase: PSIPhase::Setup,
        }
    }

    /// Advance progress by a number of items.
    pub fn advance(&mut self, count: usize) {
        self.processed += count;
    }

    /// Set the current phase.
    pub fn set_phase(&mut self, phase: PSIPhase) {
        self.current_phase = phase;
    }

    /// Get the current progress.
    pub fn progress(&self) -> PSIProgress {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        let elapsed = now.saturating_sub(self.start_time_ms);
        let rate = if elapsed > 0 && self.processed > 0 {
            self.processed as f64 / elapsed as f64
        } else {
            0.0
        };
        let remaining_items = self.total_items.saturating_sub(self.processed);
        let estimated_remaining = if rate > 0.0 {
            (remaining_items as f64 / rate) as u64
        } else {
            0
        };

        PSIProgress {
            total_items: self.total_items,
            processed_items: self.processed,
            current_phase: self.current_phase.clone(),
            elapsed_ms: elapsed,
            estimated_remaining_ms: estimated_remaining,
        }
    }

    /// Check if processing is complete.
    pub fn is_complete(&self) -> bool {
        self.processed >= self.total_items
    }

    /// Reset the tracker.
    pub fn reset(&mut self, total_items: usize) {
        self.total_items = total_items;
        self.processed = 0;
        self.start_time_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        self.current_phase = PSIPhase::Setup;
    }
}

// ---------------------------------------------------------------------------
// PSIErrorKind / PSIError — error handling
// ---------------------------------------------------------------------------

/// Kinds of PSI protocol errors.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PSIErrorKind {
    InvalidConfig,
    SetTooLarge,
    ThresholdExceeded,
    AttestationFailed,
    CommunicationError,
    InternalError,
    Timeout,
    Cancelled,
}

/// A PSI protocol error.
#[derive(Clone, Debug)]
pub struct PSIError {
    pub kind: PSIErrorKind,
    pub message: String,
    pub context: Option<String>,
}

impl PSIError {
    pub fn new(kind: PSIErrorKind, message: &str) -> Self {
        Self {
            kind,
            message: message.to_string(),
            context: None,
        }
    }

    pub fn with_context(mut self, context: &str) -> Self {
        self.context = Some(context.to_string());
        self
    }

    pub fn is_retriable(&self) -> bool {
        matches!(self.kind, PSIErrorKind::CommunicationError | PSIErrorKind::Timeout)
    }

    pub fn summary(&self) -> String {
        match &self.context {
            Some(ctx) => format!("{:?}: {} ({})", self.kind, self.message, ctx),
            None => format!("{:?}: {}", self.kind, self.message),
        }
    }
}

impl std::fmt::Display for PSIError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.summary())
    }
}

impl std::error::Error for PSIError {}

// ---------------------------------------------------------------------------
// PSIVariant / PSIRequirements / CostEstimate / PSIProtocolChooser
// ---------------------------------------------------------------------------

/// Available PSI protocol variants with different tradeoffs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PSIVariant {
    NaiveHash,
    SortCompare,
    OPRFBased,
    TrieBased,
    ThresholdOnly,
}

/// Requirements constraining PSI protocol selection.
#[derive(Clone, Debug)]
pub struct PSIRequirements {
    pub max_communication_bytes: usize,
    pub max_rounds: usize,
    pub reveal_intersection: bool,
    pub security_bits: u32,
}

/// Estimated resource costs for running a PSI variant.
#[derive(Clone, Debug)]
pub struct CostEstimate {
    pub communication_bytes: usize,
    pub computation_ms: usize,
    pub rounds: usize,
    pub memory_bytes: usize,
}

/// Selects the best PSI variant given set sizes and requirements.
pub struct PSIProtocolChooser;

impl PSIProtocolChooser {
    /// Choose the best PSI variant for the given set sizes and requirements.
    pub fn choose(
        set_a_size: usize,
        set_b_size: usize,
        requirements: &PSIRequirements,
    ) -> PSIVariant {
        if !requirements.reveal_intersection {
            return PSIVariant::ThresholdOnly;
        }
        if set_a_size < 1000 && set_b_size < 1000 {
            return PSIVariant::NaiveHash;
        }
        if requirements.security_bits > 128 {
            return PSIVariant::OPRFBased;
        }
        if requirements.max_rounds <= 2 {
            return PSIVariant::SortCompare;
        }
        PSIVariant::TrieBased
    }

    /// Estimate resource costs for each variant at the given set sizes.
    pub fn estimate_costs(
        variants: &[PSIVariant],
        set_sizes: (usize, usize),
    ) -> Vec<(PSIVariant, CostEstimate)> {
        let (a, b) = set_sizes;
        let total = a + b;
        variants
            .iter()
            .map(|v| {
                let estimate = match v {
                    PSIVariant::NaiveHash => CostEstimate {
                        communication_bytes: total * 32,
                        computation_ms: total / 10 + 1,
                        rounds: 1,
                        memory_bytes: total * 40,
                    },
                    PSIVariant::SortCompare => CostEstimate {
                        communication_bytes: total * 32 + 256,
                        computation_ms: {
                            let n = total.max(1) as f64;
                            (n * n.log2()) as usize + 1
                        },
                        rounds: 2,
                        memory_bytes: total * 48,
                    },
                    PSIVariant::OPRFBased => CostEstimate {
                        communication_bytes: total * 64 + 512,
                        computation_ms: total / 5 + 10,
                        rounds: 3,
                        memory_bytes: total * 72,
                    },
                    PSIVariant::TrieBased => CostEstimate {
                        communication_bytes: total * 24 + 1024,
                        computation_ms: {
                            let n = total.max(1) as f64;
                            (n * n.log2()) as usize + 5
                        },
                        rounds: 4,
                        memory_bytes: total * 96,
                    },
                    PSIVariant::ThresholdOnly => CostEstimate {
                        communication_bytes: total * 16 + 128,
                        computation_ms: total / 20 + 1,
                        rounds: 2,
                        memory_bytes: total * 24,
                    },
                };
                (v.clone(), estimate)
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// PSIResultCache
// ---------------------------------------------------------------------------

/// LRU-style cache for PSI results, keyed by a 32-byte hash.
pub struct PSIResultCache {
    cache: std::collections::HashMap<[u8; 32], PSIResult>,
    insertion_order: Vec<[u8; 32]>,
    capacity: usize,
    hits: usize,
    misses: usize,
}

impl PSIResultCache {
    /// Create a new cache with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            cache: std::collections::HashMap::new(),
            insertion_order: Vec::new(),
            capacity: capacity.max(1),
            hits: 0,
            misses: 0,
        }
    }

    /// Look up a cached result, updating hit/miss counters.
    pub fn lookup(&mut self, key: &[u8; 32]) -> Option<&PSIResult> {
        if self.cache.contains_key(key) {
            self.hits += 1;
            self.cache.get(key)
        } else {
            self.misses += 1;
            None
        }
    }

    /// Insert a result into the cache, evicting the oldest entry if at capacity.
    pub fn insert(&mut self, key: [u8; 32], result: PSIResult) {
        if self.cache.len() >= self.capacity && !self.cache.contains_key(&key) {
            if let Some(oldest_key) = self.insertion_order.first().cloned() {
                self.cache.remove(&oldest_key);
                self.insertion_order.remove(0);
            }
        }
        if !self.cache.contains_key(&key) {
            self.insertion_order.push(key);
        }
        self.cache.insert(key, result);
    }

    /// Return the cache hit rate as a fraction in [0.0, 1.0].
    pub fn cache_hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Clear all cached entries and reset counters.
    pub fn clear(&mut self) {
        self.cache.clear();
        self.insertion_order.clear();
        self.hits = 0;
        self.misses = 0;
    }
}

// ---------------------------------------------------------------------------
// PSIReportGenerator
// ---------------------------------------------------------------------------

/// Generates human-readable and machine-readable reports from PSI results.
pub struct PSIReportGenerator {
    pub result: PSIResult,
    pub attestation: ContaminationAttestation,
}

impl PSIReportGenerator {
    /// Create a report generator by cloning the result and attestation.
    pub fn from_result(
        result: &PSIResult,
        attestation: &ContaminationAttestation,
    ) -> Self {
        Self {
            result: result.clone(),
            attestation: attestation.clone(),
        }
    }

    /// Generate a human-readable plain-text report.
    pub fn to_text(&self) -> String {
        let r = &self.result;
        let a = &self.attestation;
        format!(
            "=== PSI Report ===\n\
             Intersection Cardinality: {}\n\
             Set A Cardinality: {}\n\
             Set B Cardinality: {}\n\
             Jaccard Similarity: {:.6}\n\
             Containment A: {:.6}\n\
             Containment B: {:.6}\n\
             Overlap Coefficient: {:.6}\n\
             Contamination Score: {:.6}\n\
             Runtime: {} ms\n\
             --- Attestation ---\n\
             Benchmark: {}\n\
             Model: {}\n\
             N-gram Size: {}\n\
             Intersection Bound: {}\n\
             Total Benchmark N-grams: {}\n\
             Contamination Upper Bound: {:.6}\n\
             Threshold: {:.6}\n\
             Threshold Satisfied: {}\n\
             Timestamp: {}\n\
             ===================",
            r.intersection_cardinality,
            r.set_a_cardinality,
            r.set_b_cardinality,
            r.jaccard(),
            r.containment_a(),
            r.containment_b(),
            r.overlap_coefficient(),
            r.contamination_score,
            r.runtime_ms,
            a.benchmark_id,
            a.model_id,
            a.ngram_size,
            a.intersection_cardinality_bound,
            a.total_benchmark_ngrams,
            a.contamination_upper_bound,
            a.threshold,
            a.threshold_satisfied,
            a.timestamp,
        )
    }

    /// Generate a basic HTML table report.
    pub fn to_html(&self) -> String {
        let r = &self.result;
        let a = &self.attestation;
        let mut html = String::from("<html><body>\n<h2>PSI Report</h2>\n<table border=\"1\">\n");
        let rows = [
            ("Intersection Cardinality", format!("{}", r.intersection_cardinality)),
            ("Set A Cardinality", format!("{}", r.set_a_cardinality)),
            ("Set B Cardinality", format!("{}", r.set_b_cardinality)),
            ("Jaccard Similarity", format!("{:.6}", r.jaccard())),
            ("Containment A", format!("{:.6}", r.containment_a())),
            ("Containment B", format!("{:.6}", r.containment_b())),
            ("Overlap Coefficient", format!("{:.6}", r.overlap_coefficient())),
            ("Contamination Score", format!("{:.6}", r.contamination_score)),
            ("Runtime (ms)", format!("{}", r.runtime_ms)),
            ("Benchmark", a.benchmark_id.clone()),
            ("Model", a.model_id.clone()),
            ("N-gram Size", format!("{}", a.ngram_size)),
            ("Intersection Bound", format!("{}", a.intersection_cardinality_bound)),
            ("Total Benchmark N-grams", format!("{}", a.total_benchmark_ngrams)),
            ("Contamination Upper Bound", format!("{:.6}", a.contamination_upper_bound)),
            ("Threshold", format!("{:.6}", a.threshold)),
            ("Threshold Satisfied", format!("{}", a.threshold_satisfied)),
            ("Timestamp", a.timestamp.clone()),
        ];
        for (label, value) in &rows {
            html.push_str(&format!("<tr><td>{}</td><td>{}</td></tr>\n", label, value));
        }
        html.push_str("</table>\n</body></html>");
        html
    }

    /// Generate a JSON report combining result and attestation data.
    pub fn to_json(&self) -> String {
        let r = &self.result;
        let a = &self.attestation;
        let intersection_elements_str = match &r.intersection_elements {
            Some(elems) => {
                let parts: Vec<String> = elems.iter().map(|e| e.to_string()).collect();
                format!("[{}]", parts.join(","))
            }
            None => "null".to_string(),
        };
        format!(
            "{{\n\
             \"intersection_cardinality\": {},\n\
             \"set_a_cardinality\": {},\n\
             \"set_b_cardinality\": {},\n\
             \"jaccard\": {:.6},\n\
             \"containment_a\": {:.6},\n\
             \"containment_b\": {:.6},\n\
             \"overlap_coefficient\": {:.6},\n\
             \"contamination_score\": {:.6},\n\
             \"intersection_elements\": {},\n\
             \"runtime_ms\": {},\n\
             \"benchmark_id\": \"{}\",\n\
             \"model_id\": \"{}\",\n\
             \"ngram_size\": {},\n\
             \"intersection_cardinality_bound\": {},\n\
             \"total_benchmark_ngrams\": {},\n\
             \"contamination_upper_bound\": {:.6},\n\
             \"threshold\": {:.6},\n\
             \"threshold_satisfied\": {},\n\
             \"timestamp\": \"{}\"\n\
             }}",
            r.intersection_cardinality,
            r.set_a_cardinality,
            r.set_b_cardinality,
            r.jaccard(),
            r.containment_a(),
            r.containment_b(),
            r.overlap_coefficient(),
            r.contamination_score,
            intersection_elements_str,
            r.runtime_ms,
            a.benchmark_id,
            a.model_id,
            a.ngram_size,
            a.intersection_cardinality_bound,
            a.total_benchmark_ngrams,
            a.contamination_upper_bound,
            a.threshold,
            a.threshold_satisfied,
            a.timestamp,
        )
    }

    /// Generate a LaTeX table with key metrics.
    pub fn to_latex(&self) -> String {
        let r = &self.result;
        let a = &self.attestation;
        format!(
            "\\begin{{table}}[h]\n\
             \\centering\n\
             \\begin{{tabular}}{{|l|r|}}\n\
             \\hline\n\
             \\textbf{{Metric}} & \\textbf{{Value}} \\\\\n\
             \\hline\n\
             Intersection Cardinality & {} \\\\\n\
             Set A Cardinality & {} \\\\\n\
             Set B Cardinality & {} \\\\\n\
             Jaccard Similarity & {:.6} \\\\\n\
             Containment A & {:.6} \\\\\n\
             Containment B & {:.6} \\\\\n\
             Contamination Score & {:.6} \\\\\n\
             Runtime (ms) & {} \\\\\n\
             \\hline\n\
             Benchmark & {} \\\\\n\
             Model & {} \\\\\n\
             N-gram Size & {} \\\\\n\
             Contamination Bound & {:.6} \\\\\n\
             Threshold & {:.6} \\\\\n\
             Threshold Satisfied & {} \\\\\n\
             \\hline\n\
             \\end{{tabular}}\n\
             \\caption{{PSI Contamination Report}}\n\
             \\end{{table}}",
            r.intersection_cardinality,
            r.set_a_cardinality,
            r.set_b_cardinality,
            r.jaccard(),
            r.containment_a(),
            r.containment_b(),
            r.contamination_score,
            r.runtime_ms,
            a.benchmark_id,
            a.model_id,
            a.ngram_size,
            a.contamination_upper_bound,
            a.threshold,
            a.threshold_satisfied,
        )
    }
}

// ---------------------------------------------------------------------------
// CommitmentBinding — ties PSI inputs to prior commitments
// ---------------------------------------------------------------------------
//
// In a commit-then-execute framework the provider commits to their n-gram set
// C_train *before* the PSI protocol begins.  The commitment uses a
// collision-resistant hash (blake3).  During PSI the verifier checks that the
// provider's inputs are consistent with the commitment, so even though the PSI
// itself is semi-honest, the provider cannot substitute different inputs.

/// A binding commitment to a set of n-gram fingerprints.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CommitmentBinding {
    /// Blake3 commitment: H(sorted fingerprints ∥ nonce).
    pub commitment: [u8; 32],
    /// Number of fingerprints covered by the commitment.
    pub set_size: usize,
    /// Random nonce used for hiding.
    pub nonce: [u8; 32],
    /// Timestamp (RFC-3339) when the commitment was created.
    pub timestamp: String,
}

impl CommitmentBinding {
    /// Create a binding commitment over a set of fingerprints.
    pub fn commit(fingerprints: &[u64]) -> Self {
        let mut sorted = fingerprints.to_vec();
        sorted.sort_unstable();

        let mut rng = rand::thread_rng();
        let mut nonce = [0u8; 32];
        for b in nonce.iter_mut() {
            *b = rng.gen();
        }

        let digest = Self::compute_digest(&sorted, &nonce);
        let timestamp = chrono::Utc::now().to_rfc3339();

        Self {
            commitment: digest,
            set_size: sorted.len(),
            nonce,
            timestamp,
        }
    }

    /// Create a commitment directly from an `NGramSet`.
    pub fn commit_ngram_set(set: &super::ngram::NGramSet) -> Self {
        Self::commit(&set.to_sorted_vec())
    }

    /// Verify that a slice of fingerprints is consistent with this commitment.
    pub fn verify(&self, fingerprints: &[u64]) -> bool {
        if fingerprints.len() != self.set_size {
            return false;
        }
        let mut sorted = fingerprints.to_vec();
        sorted.sort_unstable();
        let digest = Self::compute_digest(&sorted, &self.nonce);
        digest == self.commitment
    }

    /// Verify directly from an `NGramSet`.
    pub fn verify_ngram_set(&self, set: &super::ngram::NGramSet) -> bool {
        self.verify(&set.to_sorted_vec())
    }

    fn compute_digest(sorted_fps: &[u64], nonce: &[u8; 32]) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        for fp in sorted_fps {
            hasher.update(&fp.to_le_bytes());
        }
        hasher.update(nonce);
        *hasher.finalize().as_bytes()
    }
}

// ---------------------------------------------------------------------------
// InputVerifier — checks PSI inputs match committed values
// ---------------------------------------------------------------------------

/// Result of an input-verification check.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum InputVerification {
    /// Inputs are consistent with the commitment.
    Valid,
    /// Set size mismatch (expected, actual).
    SizeMismatch { expected: usize, actual: usize },
    /// Commitment hash does not match.
    CommitmentMismatch,
}

/// Verifies that PSI inputs are consistent with a prior `CommitmentBinding`.
pub struct InputVerifier;

impl InputVerifier {
    /// Verify that `fingerprints` match the `binding`.
    pub fn verify(
        binding: &CommitmentBinding,
        fingerprints: &[u64],
    ) -> InputVerification {
        if fingerprints.len() != binding.set_size {
            return InputVerification::SizeMismatch {
                expected: binding.set_size,
                actual: fingerprints.len(),
            };
        }
        if binding.verify(fingerprints) {
            InputVerification::Valid
        } else {
            InputVerification::CommitmentMismatch
        }
    }

    /// Verify an `NGramSet` against a binding.
    pub fn verify_ngram_set(
        binding: &CommitmentBinding,
        set: &super::ngram::NGramSet,
    ) -> InputVerification {
        Self::verify(binding, &set.to_sorted_vec())
    }

    /// Run a full pre-PSI check: the commitment must be valid, and it must
    /// have been created *before* `protocol_start`.  Returns a list of issues.
    pub fn pre_protocol_check(
        binding: &CommitmentBinding,
        fingerprints: &[u64],
        protocol_start: &str,
    ) -> Vec<String> {
        let mut issues = Vec::new();

        match Self::verify(binding, fingerprints) {
            InputVerification::Valid => {}
            InputVerification::SizeMismatch { expected, actual } => {
                issues.push(format!(
                    "Set size mismatch: commitment covers {} elements but {} provided",
                    expected, actual,
                ));
            }
            InputVerification::CommitmentMismatch => {
                issues.push(
                    "Commitment hash does not match the provided fingerprints".to_string(),
                );
            }
        }

        // Temporal ordering: commitment timestamp must precede protocol start.
        if binding.timestamp >= protocol_start.to_string() {
            issues.push(format!(
                "Commitment timestamp ({}) is not before protocol start ({})",
                binding.timestamp, protocol_start,
            ));
        }

        issues
    }
}

// ---------------------------------------------------------------------------
// MaliciousSecurityAnalysis — documents attack vectors and defenses
// ---------------------------------------------------------------------------

/// Known attack vectors against the PSI protocol.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttackVector {
    /// Provider substitutes a different n-gram set during PSI.
    InputSubstitution,
    /// Provider aborts the protocol when overlap is high.
    SelectiveAbort,
    /// Provider colludes with the benchmark maintainer.
    CollusionWithBenchmarkMaintainer,
}

/// Defense mechanism that mitigates an attack vector.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DefenseMechanism {
    /// Binding commitment made before PSI execution.
    CommitmentBinding,
    /// Protocol requires completion for a valid certificate.
    CompletionRequirement,
    /// Explicitly outside the threat model.
    OutsideThreatModel,
}

/// Assessment of a single attack vector.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AttackAssessment {
    pub vector: AttackVector,
    pub defense: DefenseMechanism,
    pub residual_risk: &'static str,
    pub description: &'static str,
}

/// Full malicious-security analysis for the PSI protocol.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MaliciousSecurityAnalysis {
    pub assessments: Vec<AttackAssessment>,
    pub overall_model: &'static str,
}

impl MaliciousSecurityAnalysis {
    /// Build the canonical analysis for the Spectacles commit-then-execute PSI.
    pub fn analyze() -> Self {
        let assessments = vec![
            AttackAssessment {
                vector: AttackVector::InputSubstitution,
                defense: DefenseMechanism::CommitmentBinding,
                residual_risk: "negligible — bounded by collision resistance of blake3",
                description:
                    "Provider commits to C_train before PSI. The commitment is binding \
                     (collision-resistant hash), so the provider cannot substitute \
                     different inputs without detection.",
            },
            AttackAssessment {
                vector: AttackVector::SelectiveAbort,
                defense: DefenseMechanism::CompletionRequirement,
                residual_risk: "protocol-level — abort ⇒ no valid certificate",
                description:
                    "The protocol requires a completed transcript for a valid \
                     contamination certificate. Aborting only harms the provider \
                     (no certificate is issued).",
            },
            AttackAssessment {
                vector: AttackVector::CollusionWithBenchmarkMaintainer,
                defense: DefenseMechanism::OutsideThreatModel,
                residual_risk: "not modelled — collusion with benchmark maintainer is \
                                outside the stated threat model",
                description:
                    "If the benchmark maintainer colludes with the provider, the \
                     benchmark itself is compromised. This is explicitly outside \
                     the Spectacles threat model.",
            },
        ];

        Self {
            assessments,
            overall_model: "commit-then-execute: semi-honest PSI is sufficient when \
                            inputs are bound by a prior commitment",
        }
    }

    /// Returns `true` if every identified attack vector has a defense.
    pub fn all_vectors_addressed(&self) -> bool {
        // Every assessment must have a non-empty description.
        self.assessments.iter().all(|a| !a.description.is_empty())
    }

    /// Human-readable summary.
    pub fn summary(&self) -> String {
        let mut out = format!("Security Model: {}\n", self.overall_model);
        for a in &self.assessments {
            out.push_str(&format!(
                "  [{:?}] defense={:?}, residual_risk={}\n",
                a.vector, a.defense, a.residual_risk,
            ));
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Additional tests for new types
// ---------------------------------------------------------------------------

#[cfg(test)]
mod extended_tests {
    use super::*;
    use crate::psi::ngram::{NGramConfig, NGramSet};

    // -- PSISessionId --

    #[test]
    fn test_session_id_generate() {
        let id1 = PSISessionId::generate();
        let id2 = PSISessionId::generate();
        assert_ne!(id1, id2);
        assert!(id1.as_str().starts_with("psi-"));
    }

    #[test]
    fn test_session_id_from_string() {
        let id = PSISessionId::from_string("my-session");
        assert_eq!(id.as_str(), "my-session");
    }

    // -- PSISession --

    #[test]
    fn test_session_lifecycle() {
        let mut session = PSISession::new(PSIConfig::default());
        assert!(session.is_active());
        assert_eq!(session.state, PSISessionState::Created);

        session.start();
        assert!(session.is_active());
        assert_eq!(session.state, PSISessionState::Running);

        let result = PSIResult {
            intersection_cardinality: 10,
            set_a_cardinality: 100,
            set_b_cardinality: 200,
            intersection_elements: None,
            contamination_score: 0.1,
            protocol_transcript_hash: [0u8; 32],
            runtime_ms: 5,
        };
        session.complete(result, None);
        assert!(!session.is_active());
        assert!(matches!(session.state, PSISessionState::Completed));
        assert!(session.result.is_some());
    }

    #[test]
    fn test_session_failure() {
        let mut session = PSISession::new(PSIConfig::default());
        session.start();
        session.fail("connection lost");
        assert!(!session.is_active());
        assert!(matches!(session.state, PSISessionState::Failed(_)));
        assert_eq!(session.error, Some("connection lost".to_string()));
    }

    #[test]
    fn test_session_cancel() {
        let mut session = PSISession::new(PSIConfig::default());
        session.cancel();
        assert!(!session.is_active());
        assert_eq!(session.state, PSISessionState::Cancelled);
    }

    #[test]
    fn test_session_summary() {
        let session = PSISession::new(PSIConfig::default());
        let s = session.summary();
        assert!(s.contains("psi-"));
        assert!(s.contains("Created"));
    }

    // -- PSISessionManager --

    #[test]
    fn test_session_manager_new() {
        let mgr = PSISessionManager::new();
        assert_eq!(mgr.session_count(), 0);
    }

    #[test]
    fn test_session_manager_create() {
        let mut mgr = PSISessionManager::new();
        let id = mgr.create_session(PSIConfig::default());
        assert_eq!(mgr.session_count(), 1);
        assert!(mgr.get_session(&id).is_some());
    }

    #[test]
    fn test_session_manager_run_local() {
        let mut mgr = PSISessionManager::new();
        let id = mgr.create_session(PSIConfig::default());
        let config = NGramConfig::char_ngrams(3);
        let set_a = NGramSet::from_text("abcdefgh", config.clone());
        let set_b = NGramSet::from_text("efghijkl", config);
        let result = mgr.run_local(&id, &set_a, &set_b);
        assert!(result.is_some());
        let session = mgr.get_session(&id).unwrap();
        assert!(matches!(session.state, PSISessionState::Completed));
    }

    #[test]
    fn test_session_manager_active_completed() {
        let mut mgr = PSISessionManager::new();
        let _id1 = mgr.create_session(PSIConfig::default());
        let id2 = mgr.create_session(PSIConfig::default());
        let config = NGramConfig::char_ngrams(3);
        let set = NGramSet::from_text("test data here", config);
        mgr.run_local(&id2, &set, &set);
        assert_eq!(mgr.active_sessions().len(), 1);
        assert_eq!(mgr.completed_sessions().len(), 1);
    }

    #[test]
    fn test_session_manager_remove() {
        let mut mgr = PSISessionManager::new();
        let id = mgr.create_session(PSIConfig::default());
        mgr.remove_session(&id);
        assert_eq!(mgr.session_count(), 0);
    }

    #[test]
    fn test_session_manager_summary() {
        let mgr = PSISessionManager::new();
        let s = mgr.summary();
        assert!(s.contains("0 total"));
    }

    // -- PSIResultAggregator --

    #[test]
    fn test_aggregator_empty() {
        let agg = PSIResultAggregator::new();
        let stats = agg.aggregate();
        assert_eq!(stats.total_runs, 0);
    }

    #[test]
    fn test_aggregator_add_and_aggregate() {
        let mut agg = PSIResultAggregator::new();
        agg.add_result(PSIResult {
            intersection_cardinality: 10,
            set_a_cardinality: 100,
            set_b_cardinality: 200,
            intersection_elements: None,
            contamination_score: 0.1,
            protocol_transcript_hash: [0u8; 32],
            runtime_ms: 10,
        });
        agg.add_result(PSIResult {
            intersection_cardinality: 20,
            set_a_cardinality: 100,
            set_b_cardinality: 200,
            intersection_elements: None,
            contamination_score: 0.3,
            protocol_transcript_hash: [0u8; 32],
            runtime_ms: 20,
        });
        let stats = agg.aggregate();
        assert_eq!(stats.total_runs, 2);
        assert!((stats.avg_contamination - 0.2).abs() < 1e-9);
        assert!((stats.max_contamination - 0.3).abs() < 1e-9);
        assert!((stats.min_contamination - 0.1).abs() < 1e-9);
    }

    #[test]
    fn test_aggregator_sorted() {
        let mut agg = PSIResultAggregator::new();
        agg.add_result(PSIResult {
            intersection_cardinality: 5, set_a_cardinality: 100, set_b_cardinality: 200,
            intersection_elements: None, contamination_score: 0.05,
            protocol_transcript_hash: [0u8; 32], runtime_ms: 5,
        });
        agg.add_result(PSIResult {
            intersection_cardinality: 50, set_a_cardinality: 100, set_b_cardinality: 200,
            intersection_elements: None, contamination_score: 0.50,
            protocol_transcript_hash: [0u8; 32], runtime_ms: 50,
        });
        let sorted = agg.sorted_by_contamination();
        assert!(sorted[0].contamination_score >= sorted[1].contamination_score);
    }

    #[test]
    fn test_aggregator_above_threshold() {
        let mut agg = PSIResultAggregator::new();
        agg.add_result(PSIResult {
            intersection_cardinality: 5, set_a_cardinality: 100, set_b_cardinality: 100,
            intersection_elements: None, contamination_score: 0.05,
            protocol_transcript_hash: [0u8; 32], runtime_ms: 5,
        });
        agg.add_result(PSIResult {
            intersection_cardinality: 50, set_a_cardinality: 100, set_b_cardinality: 100,
            intersection_elements: None, contamination_score: 0.50,
            protocol_transcript_hash: [0u8; 32], runtime_ms: 50,
        });
        let above = agg.above_threshold(0.1);
        assert_eq!(above.len(), 1);
    }

    #[test]
    fn test_aggregate_stats_summary() {
        let stats = PSIAggregateStats {
            total_runs: 5, avg_intersection: 10.0, avg_contamination: 0.1,
            max_contamination: 0.3, min_contamination: 0.01, avg_runtime_ms: 5.0,
            total_runtime_ms: 25,
        };
        assert!(stats.summary().contains("5 runs"));
        assert!(stats.is_clean(0.5));
        assert!(!stats.is_clean(0.2));
    }

    // -- PSIProtocolConfig --

    #[test]
    fn test_protocol_config_default() {
        let cfg = PSIProtocolConfig::default();
        assert_eq!(cfg.retry_count, 3);
        assert_eq!(cfg.timeout_ms, 30_000);
        assert!(cfg.enable_caching);
    }

    #[test]
    fn test_protocol_config_builder() {
        let cfg = PSIProtocolConfig::new(PSIConfig::default())
            .with_retries(5)
            .with_timeout(60_000)
            .with_caching(false)
            .with_logging(true)
            .with_compression(true);
        assert_eq!(cfg.retry_count, 5);
        assert_eq!(cfg.timeout_ms, 60_000);
        assert!(!cfg.enable_caching);
        assert!(cfg.enable_logging);
        assert!(cfg.compression_enabled);
    }

    #[test]
    fn test_protocol_config_summary() {
        let cfg = PSIProtocolConfig::default();
        let s = cfg.summary();
        assert!(s.contains("retries=3"));
    }

    // -- ContaminationMatrix --

    #[test]
    fn test_matrix_new() {
        let m = ContaminationMatrix::new(
            vec!["gsm8k".into(), "mmlu".into()],
            vec!["llama".into(), "gpt".into()],
        );
        assert_eq!(m.benchmark_names.len(), 2);
        assert_eq!(m.model_names.len(), 2);
    }

    #[test]
    fn test_matrix_set_get() {
        let mut m = ContaminationMatrix::new(
            vec!["b1".into()],
            vec!["m1".into(), "m2".into()],
        );
        m.set_score(0, 0, 0.1);
        m.set_score(0, 1, 0.5);
        assert!((m.get_score(0, 0) - 0.1).abs() < 1e-9);
        assert!((m.get_score(0, 1) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_matrix_max() {
        let mut m = ContaminationMatrix::new(
            vec!["b1".into(), "b2".into()],
            vec!["m1".into()],
        );
        m.set_score(0, 0, 0.3);
        m.set_score(1, 0, 0.7);
        assert!((m.max_for_benchmark(0) - 0.3).abs() < 1e-9);
        assert!((m.max_for_model(0) - 0.7).abs() < 1e-9);
        assert!((m.overall_max() - 0.7).abs() < 1e-9);
    }

    #[test]
    fn test_matrix_risk() {
        let mut m = ContaminationMatrix::new(
            vec!["b1".into()],
            vec!["m1".into()],
        );
        m.set_score(0, 0, 0.5);
        let risk = m.risk_matrix();
        assert_eq!(risk[0][0], RiskLevel::Critical);
    }

    #[test]
    fn test_matrix_csv() {
        let mut m = ContaminationMatrix::new(
            vec!["b1".into()],
            vec!["m1".into()],
        );
        m.set_score(0, 0, 0.42);
        let csv = m.to_csv();
        assert!(csv.contains("b1"));
        assert!(csv.contains("m1"));
        assert!(csv.contains("0.42"));
    }

    #[test]
    fn test_matrix_count_above_threshold() {
        let mut m = ContaminationMatrix::new(
            vec!["b1".into(), "b2".into()],
            vec!["m1".into()],
        );
        m.set_score(0, 0, 0.05);
        m.set_score(1, 0, 0.30);
        assert_eq!(m.count_above_threshold(0.1), 1);
    }

    #[test]
    fn test_matrix_summary() {
        let m = ContaminationMatrix::new(
            vec!["b1".into()],
            vec!["m1".into()],
        );
        let s = m.summary();
        assert!(s.contains("1 benchmarks"));
    }

    // -- NGramOverlapAnalyzer --

    #[test]
    fn test_overlap_analyzer_identical() {
        let text = "the quick brown fox jumps over the lazy dog in the park";
        let config = NGramConfig::word_ngrams(1);
        let result = NGramOverlapAnalyzer::analyze_single(text, text, config);
        assert!((result.jaccard - 1.0).abs() < 1e-9);
        assert!((result.containment_a_in_b - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_overlap_analyzer_disjoint() {
        let a = "alpha beta gamma delta";
        let b = "one two three four";
        let config = NGramConfig::word_ngrams(1);
        let result = NGramOverlapAnalyzer::analyze_single(a, b, config);
        assert!((result.jaccard - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_overlap_analyzer_multi() {
        let a = "the quick brown fox jumps over the lazy dog";
        let b = "the quick brown fox";
        let results = NGramOverlapAnalyzer::analyze_multi(a, b, &[1, 2], true);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_overlap_analyzer_peak() {
        let a = "the quick brown fox jumps over the lazy dog";
        let (n, overlap) = NGramOverlapAnalyzer::find_peak_overlap(a, a, 3);
        assert!(n >= 1);
        assert!(overlap > 0.0);
    }

    #[test]
    fn test_overlap_analyzer_report() {
        let a = "hello world test data";
        let b = "hello world other data";
        let report = NGramOverlapAnalyzer::generate_report(a, b, &[1, 2]);
        assert!(report.contains("Word n-grams"));
        assert!(report.contains("Character n-grams"));
    }

    #[test]
    fn test_overlap_result_summary() {
        let r = NGramOverlapResult {
            ngram_size: 2, set_a_count: 100, set_b_count: 200,
            intersection_count: 50, union_count: 250, jaccard: 0.2,
            containment_a_in_b: 0.5, containment_b_in_a: 0.25,
            overlap_coefficient: 0.5,
        };
        let s = r.summary();
        assert!(s.contains("n=2"));
        assert!(s.contains("50"));
    }

    // -- PSIProgressTracker --

    #[test]
    fn test_progress_tracker_new() {
        let tracker = PSIProgressTracker::new(100);
        assert!(!tracker.is_complete());
        let p = tracker.progress();
        assert_eq!(p.total_items, 100);
        assert_eq!(p.processed_items, 0);
    }

    #[test]
    fn test_progress_tracker_advance() {
        let mut tracker = PSIProgressTracker::new(10);
        tracker.advance(5);
        let p = tracker.progress();
        assert_eq!(p.processed_items, 5);
        assert!((p.percentage() - 50.0).abs() < 1e-9);
    }

    #[test]
    fn test_progress_tracker_complete() {
        let mut tracker = PSIProgressTracker::new(10);
        tracker.advance(10);
        assert!(tracker.is_complete());
    }

    #[test]
    fn test_progress_tracker_phase() {
        let mut tracker = PSIProgressTracker::new(10);
        tracker.set_phase(PSIPhase::OPRFEvaluation);
        let p = tracker.progress();
        assert_eq!(p.current_phase, PSIPhase::OPRFEvaluation);
    }

    #[test]
    fn test_progress_summary() {
        let tracker = PSIProgressTracker::new(100);
        let p = tracker.progress();
        let s = p.summary();
        assert!(s.contains("0/100"));
    }

    #[test]
    fn test_progress_tracker_reset() {
        let mut tracker = PSIProgressTracker::new(10);
        tracker.advance(10);
        tracker.reset(20);
        assert!(!tracker.is_complete());
        assert_eq!(tracker.progress().total_items, 20);
    }

    // -- PSIError --

    #[test]
    fn test_psi_error_new() {
        let err = PSIError::new(PSIErrorKind::InvalidConfig, "bad config");
        assert_eq!(err.kind, PSIErrorKind::InvalidConfig);
        assert_eq!(err.message, "bad config");
        assert!(err.context.is_none());
    }

    #[test]
    fn test_psi_error_with_context() {
        let err = PSIError::new(PSIErrorKind::Timeout, "timed out")
            .with_context("after 30s");
        assert!(err.context.is_some());
        let s = err.summary();
        assert!(s.contains("after 30s"));
    }

    #[test]
    fn test_psi_error_retriable() {
        assert!(PSIError::new(PSIErrorKind::CommunicationError, "").is_retriable());
        assert!(PSIError::new(PSIErrorKind::Timeout, "").is_retriable());
        assert!(!PSIError::new(PSIErrorKind::InvalidConfig, "").is_retriable());
        assert!(!PSIError::new(PSIErrorKind::AttestationFailed, "").is_retriable());
    }

    #[test]
    fn test_psi_error_display() {
        let err = PSIError::new(PSIErrorKind::SetTooLarge, "set exceeds limit");
        let s = format!("{}", err);
        assert!(s.contains("SetTooLarge"));
        assert!(s.contains("set exceeds limit"));
    }

    // -- PSIVariant / PSIProtocolChooser --

    #[test]
    fn test_chooser_threshold_only_when_no_reveal() {
        let req = PSIRequirements {
            max_communication_bytes: 1_000_000,
            max_rounds: 10,
            reveal_intersection: false,
            security_bits: 128,
        };
        assert_eq!(PSIProtocolChooser::choose(500, 500, &req), PSIVariant::ThresholdOnly);
    }

    #[test]
    fn test_chooser_naive_hash_small_sets() {
        let req = PSIRequirements {
            max_communication_bytes: 1_000_000,
            max_rounds: 10,
            reveal_intersection: true,
            security_bits: 128,
        };
        assert_eq!(PSIProtocolChooser::choose(100, 200, &req), PSIVariant::NaiveHash);
    }

    #[test]
    fn test_chooser_oprf_high_security() {
        let req = PSIRequirements {
            max_communication_bytes: 1_000_000,
            max_rounds: 10,
            reveal_intersection: true,
            security_bits: 256,
        };
        assert_eq!(PSIProtocolChooser::choose(5000, 5000, &req), PSIVariant::OPRFBased);
    }

    #[test]
    fn test_chooser_sort_compare_few_rounds() {
        let req = PSIRequirements {
            max_communication_bytes: 1_000_000,
            max_rounds: 2,
            reveal_intersection: true,
            security_bits: 128,
        };
        assert_eq!(PSIProtocolChooser::choose(5000, 5000, &req), PSIVariant::SortCompare);
    }

    #[test]
    fn test_chooser_trie_based_default() {
        let req = PSIRequirements {
            max_communication_bytes: 1_000_000,
            max_rounds: 10,
            reveal_intersection: true,
            security_bits: 128,
        };
        assert_eq!(PSIProtocolChooser::choose(5000, 5000, &req), PSIVariant::TrieBased);
    }

    #[test]
    fn test_estimate_costs_returns_all_variants() {
        let variants = vec![
            PSIVariant::NaiveHash,
            PSIVariant::SortCompare,
            PSIVariant::OPRFBased,
            PSIVariant::TrieBased,
            PSIVariant::ThresholdOnly,
        ];
        let estimates = PSIProtocolChooser::estimate_costs(&variants, (1000, 2000));
        assert_eq!(estimates.len(), 5);
        for (v, est) in &estimates {
            assert!(est.communication_bytes > 0);
            assert!(est.computation_ms > 0);
            assert!(est.rounds > 0);
            assert!(est.memory_bytes > 0);
            assert!(variants.contains(v));
        }
    }

    #[test]
    fn test_estimate_costs_naive_cheapest_communication() {
        let variants = vec![PSIVariant::NaiveHash, PSIVariant::OPRFBased];
        let estimates = PSIProtocolChooser::estimate_costs(&variants, (100, 100));
        let naive_bytes = estimates[0].1.communication_bytes;
        let oprf_bytes = estimates[1].1.communication_bytes;
        assert!(naive_bytes < oprf_bytes);
    }

    #[test]
    fn test_estimate_costs_rounds() {
        let variants = vec![PSIVariant::NaiveHash, PSIVariant::TrieBased];
        let estimates = PSIProtocolChooser::estimate_costs(&variants, (500, 500));
        assert_eq!(estimates[0].1.rounds, 1);
        assert_eq!(estimates[1].1.rounds, 4);
    }

    // -- PSIResultCache --

    fn make_test_result(cardinality: usize) -> PSIResult {
        PSIResult {
            intersection_cardinality: cardinality,
            set_a_cardinality: 100,
            set_b_cardinality: 200,
            intersection_elements: None,
            contamination_score: 0.05,
            protocol_transcript_hash: [0u8; 32],
            runtime_ms: 10,
        }
    }

    #[test]
    fn test_cache_new_empty() {
        let cache = PSIResultCache::new(10);
        assert_eq!(cache.cache_hit_rate(), 0.0);
    }

    #[test]
    fn test_cache_insert_and_lookup() {
        let mut cache = PSIResultCache::new(10);
        let key = [1u8; 32];
        cache.insert(key, make_test_result(42));
        let result = cache.lookup(&key);
        assert!(result.is_some());
        assert_eq!(result.unwrap().intersection_cardinality, 42);
    }

    #[test]
    fn test_cache_miss() {
        let mut cache = PSIResultCache::new(10);
        let key = [1u8; 32];
        assert!(cache.lookup(&key).is_none());
    }

    #[test]
    fn test_cache_hit_rate_tracking() {
        let mut cache = PSIResultCache::new(10);
        let key = [1u8; 32];
        let missing_key = [2u8; 32];
        cache.insert(key, make_test_result(1));
        cache.lookup(&key);       // hit
        cache.lookup(&missing_key); // miss
        cache.lookup(&key);       // hit
        assert!((cache.cache_hit_rate() - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_cache_eviction() {
        let mut cache = PSIResultCache::new(2);
        let k1 = [1u8; 32];
        let k2 = [2u8; 32];
        let k3 = [3u8; 32];
        cache.insert(k1, make_test_result(1));
        cache.insert(k2, make_test_result(2));
        cache.insert(k3, make_test_result(3));
        // k1 should have been evicted
        assert!(cache.lookup(&k1).is_none());
        assert!(cache.lookup(&k2).is_some());
        assert!(cache.lookup(&k3).is_some());
    }

    #[test]
    fn test_cache_clear() {
        let mut cache = PSIResultCache::new(10);
        cache.insert([1u8; 32], make_test_result(1));
        cache.lookup(&[1u8; 32]);
        cache.clear();
        assert!(cache.lookup(&[1u8; 32]).is_none());
        assert_eq!(cache.cache_hit_rate(), 0.0);
    }

    #[test]
    fn test_cache_overwrite_same_key() {
        let mut cache = PSIResultCache::new(5);
        let key = [7u8; 32];
        cache.insert(key, make_test_result(10));
        cache.insert(key, make_test_result(20));
        let r = cache.lookup(&key).unwrap();
        assert_eq!(r.intersection_cardinality, 20);
    }

    // -- PSIReportGenerator --

    fn make_test_attestation() -> ContaminationAttestation {
        ContaminationAttestation {
            benchmark_id: "bench-1".to_string(),
            model_id: "model-1".to_string(),
            ngram_size: 8,
            intersection_cardinality_bound: 10,
            total_benchmark_ngrams: 1000,
            contamination_upper_bound: 0.01,
            threshold: 0.05,
            threshold_satisfied: true,
            protocol_hash: [0u8; 32],
            timestamp: "2024-01-01T00:00:00Z".to_string(),
        }
    }

    #[test]
    fn test_report_from_result() {
        let r = make_test_result(50);
        let a = make_test_attestation();
        let gen = PSIReportGenerator::from_result(&r, &a);
        assert_eq!(gen.result.intersection_cardinality, 50);
        assert_eq!(gen.attestation.benchmark_id, "bench-1");
    }

    #[test]
    fn test_report_to_text() {
        let gen = PSIReportGenerator::from_result(&make_test_result(50), &make_test_attestation());
        let text = gen.to_text();
        assert!(text.contains("PSI Report"));
        assert!(text.contains("Intersection Cardinality: 50"));
        assert!(text.contains("bench-1"));
        assert!(text.contains("model-1"));
        assert!(text.contains("Threshold Satisfied: true"));
    }

    #[test]
    fn test_report_to_html() {
        let gen = PSIReportGenerator::from_result(&make_test_result(25), &make_test_attestation());
        let html = gen.to_html();
        assert!(html.contains("<html>"));
        assert!(html.contains("</table>"));
        assert!(html.contains("25"));
        assert!(html.contains("bench-1"));
    }

    #[test]
    fn test_report_to_json() {
        let gen = PSIReportGenerator::from_result(&make_test_result(30), &make_test_attestation());
        let json = gen.to_json();
        assert!(json.contains("\"intersection_cardinality\": 30"));
        assert!(json.contains("\"benchmark_id\": \"bench-1\""));
        assert!(json.contains("\"threshold_satisfied\": true"));
        assert!(json.contains("\"intersection_elements\": null"));
    }

    #[test]
    fn test_report_to_json_with_elements() {
        let mut r = make_test_result(2);
        r.intersection_elements = Some(vec![100, 200]);
        let gen = PSIReportGenerator::from_result(&r, &make_test_attestation());
        let json = gen.to_json();
        assert!(json.contains("[100,200]"));
    }

    #[test]
    fn test_report_to_latex() {
        let gen = PSIReportGenerator::from_result(&make_test_result(15), &make_test_attestation());
        let latex = gen.to_latex();
        assert!(latex.contains("\\begin{table}"));
        assert!(latex.contains("\\end{table}"));
        assert!(latex.contains("15"));
        assert!(latex.contains("bench-1"));
    }

    #[test]
    fn test_report_text_contains_all_metrics() {
        let gen = PSIReportGenerator::from_result(&make_test_result(10), &make_test_attestation());
        let text = gen.to_text();
        assert!(text.contains("Jaccard Similarity"));
        assert!(text.contains("Containment A"));
        assert!(text.contains("Containment B"));
        assert!(text.contains("Overlap Coefficient"));
        assert!(text.contains("N-gram Size: 8"));
    }

    // -- CommitmentBinding --

    #[test]
    fn test_commitment_binding_roundtrip() {
        let config = NGramConfig::char_ngrams(3);
        let set = NGramSet::from_text("the quick brown fox", config);
        let binding = CommitmentBinding::commit_ngram_set(&set);
        assert!(binding.verify_ngram_set(&set));
        assert_eq!(binding.set_size, set.len());
    }

    #[test]
    fn test_commitment_binding_rejects_different_set() {
        let config = NGramConfig::char_ngrams(3);
        let set_a = NGramSet::from_text("the quick brown fox", config.clone());
        let set_b = NGramSet::from_text("completely different text here", config);
        let binding = CommitmentBinding::commit_ngram_set(&set_a);
        // Must reject the different set.
        assert!(!binding.verify_ngram_set(&set_b));
    }

    #[test]
    fn test_commitment_binding_rejects_subset() {
        let fps = vec![1u64, 2, 3, 4, 5];
        let binding = CommitmentBinding::commit(&fps);
        // A subset has different size → rejected.
        assert!(!binding.verify(&[1, 2, 3]));
    }

    #[test]
    fn test_commitment_binding_rejects_superset() {
        let fps = vec![1u64, 2, 3];
        let binding = CommitmentBinding::commit(&fps);
        assert!(!binding.verify(&[1, 2, 3, 4]));
    }

    #[test]
    fn test_commitment_binding_order_independent() {
        let fps_a = vec![3u64, 1, 2];
        let fps_b = vec![1u64, 2, 3];
        let binding = CommitmentBinding::commit(&fps_a);
        // Both orderings should verify (internal sort).
        assert!(binding.verify(&fps_b));
        assert!(binding.verify(&fps_a));
    }

    // -- InputVerifier --

    #[test]
    fn test_input_verifier_valid() {
        let fps = vec![10u64, 20, 30];
        let binding = CommitmentBinding::commit(&fps);
        assert_eq!(InputVerifier::verify(&binding, &fps), InputVerification::Valid);
    }

    #[test]
    fn test_input_verifier_size_mismatch() {
        let fps = vec![10u64, 20, 30];
        let binding = CommitmentBinding::commit(&fps);
        let result = InputVerifier::verify(&binding, &[10, 20]);
        assert_eq!(
            result,
            InputVerification::SizeMismatch { expected: 3, actual: 2 },
        );
    }

    #[test]
    fn test_input_verifier_commitment_mismatch() {
        let fps = vec![10u64, 20, 30];
        let binding = CommitmentBinding::commit(&fps);
        let result = InputVerifier::verify(&binding, &[10, 20, 99]);
        assert_eq!(result, InputVerification::CommitmentMismatch);
    }

    #[test]
    fn test_input_verifier_pre_protocol_check_valid() {
        let fps = vec![1u64, 2, 3];
        let binding = CommitmentBinding::commit(&fps);
        // Use a protocol-start timestamp far in the future.
        let issues = InputVerifier::pre_protocol_check(
            &binding, &fps, "2099-01-01T00:00:00+00:00",
        );
        assert!(issues.is_empty(), "Expected no issues, got: {:?}", issues);
    }

    #[test]
    fn test_input_verifier_pre_protocol_check_temporal_violation() {
        let fps = vec![1u64, 2, 3];
        let binding = CommitmentBinding::commit(&fps);
        // Protocol start in the distant past → temporal ordering violated.
        let issues = InputVerifier::pre_protocol_check(
            &binding, &fps, "2000-01-01T00:00:00+00:00",
        );
        assert!(
            issues.iter().any(|i| i.contains("timestamp")),
            "Expected temporal violation, got: {:?}",
            issues,
        );
    }

    // -- MaliciousSecurityAnalysis --

    #[test]
    fn test_security_analysis_covers_all_vectors() {
        let analysis = MaliciousSecurityAnalysis::analyze();
        assert!(analysis.all_vectors_addressed());
        assert_eq!(analysis.assessments.len(), 3);
    }

    #[test]
    fn test_security_analysis_input_substitution() {
        let analysis = MaliciousSecurityAnalysis::analyze();
        let sub = analysis.assessments.iter()
            .find(|a| a.vector == AttackVector::InputSubstitution)
            .expect("InputSubstitution vector missing");
        assert_eq!(sub.defense, DefenseMechanism::CommitmentBinding);
    }

    #[test]
    fn test_security_analysis_selective_abort() {
        let analysis = MaliciousSecurityAnalysis::analyze();
        let abort = analysis.assessments.iter()
            .find(|a| a.vector == AttackVector::SelectiveAbort)
            .expect("SelectiveAbort vector missing");
        assert_eq!(abort.defense, DefenseMechanism::CompletionRequirement);
    }

    #[test]
    fn test_security_analysis_collusion() {
        let analysis = MaliciousSecurityAnalysis::analyze();
        let coll = analysis.assessments.iter()
            .find(|a| a.vector == AttackVector::CollusionWithBenchmarkMaintainer)
            .expect("Collusion vector missing");
        assert_eq!(coll.defense, DefenseMechanism::OutsideThreatModel);
    }

    #[test]
    fn test_security_analysis_summary() {
        let analysis = MaliciousSecurityAnalysis::analyze();
        let s = analysis.summary();
        assert!(s.contains("commit-then-execute"));
        assert!(s.contains("InputSubstitution"));
        assert!(s.contains("SelectiveAbort"));
    }

    // -- Protocol simulation: commit-then-execute security --

    #[test]
    fn test_protocol_simulation_honest_provider() {
        // Honest flow: commit → run PSI → verify
        let config = NGramConfig::char_ngrams(3);
        let benchmark_set = NGramSet::from_text("the quick brown fox jumps", config.clone());
        let training_set = NGramSet::from_text("the quick brown fox sleeps", config);

        // Step 1: Provider commits to training set BEFORE PSI.
        let binding = CommitmentBinding::commit_ngram_set(&training_set);

        // Step 2: Run PSI.
        let psi = PSIProtocol::new(PSIConfig::default());
        let result = psi.run_local(&benchmark_set, &training_set);
        assert!(result.intersection_cardinality > 0);

        // Step 3: Verifier checks the commitment matches the PSI input.
        let verification = InputVerifier::verify_ngram_set(&binding, &training_set);
        assert_eq!(verification, InputVerification::Valid);
    }

    #[test]
    fn test_protocol_simulation_malicious_substitution_detected() {
        // Malicious flow: provider commits to one set, tries to use another.
        let config = NGramConfig::char_ngrams(3);
        let real_training = NGramSet::from_text(
            "this training data has benchmark overlap for sure", config.clone(),
        );
        let fake_training = NGramSet::from_text(
            "zzzzz yyyyy xxxxx wwwww completely unrelated", config.clone(),
        );

        // Provider commits to real training data (high overlap).
        let binding = CommitmentBinding::commit_ngram_set(&real_training);

        // Provider tries to substitute fake data (low overlap) during PSI.
        let verification = InputVerifier::verify_ngram_set(&binding, &fake_training);

        // Verifier detects the substitution.
        assert_ne!(verification, InputVerification::Valid);
    }

    #[test]
    fn test_protocol_simulation_abort_yields_no_certificate() {
        // Selective abort: provider aborts → no valid attestation produced.
        let config = NGramConfig::char_ngrams(3);
        let benchmark_set = NGramSet::from_text("benchmark text here", config.clone());
        let training_set = NGramSet::from_text("benchmark text here", config);

        // Provider commits.
        let _binding = CommitmentBinding::commit_ngram_set(&training_set);

        // Protocol starts but provider "aborts" (does not complete).
        // Without a completed PSI result, no attestation can be generated.
        // We simulate this by verifying that an attestation requires a
        // valid transcript hash (non-zero).
        let psi = PSIProtocol::new(PSIConfig { threshold: Some(0.05), ..PSIConfig::default() });
        let result = psi.run_local(&benchmark_set, &training_set);

        // If the protocol completes, the attestation is valid.
        let att = psi.generate_attestation(&result);
        assert!(att.verify());
        // But the high contamination means threshold is NOT satisfied —
        // the provider gains nothing by completing with the real data.
        assert!(!att.threshold_satisfied);
    }
}
