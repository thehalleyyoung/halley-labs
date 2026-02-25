//! Example: PSI-based contamination detection.
//!
//! Demonstrates how to check if test data appears in training data
//! using hash-based private set intersection.

use spectacles_core::scoring::{
    ScoringPair, TripleMetric,
    exact_match::ExactMatchScorer,
    token_f1::TokenF1Scorer,
};
use spectacles_core::utils::{
    hash::{SpectaclesHasher, MerkleTree, HashChain, Commitment},
    serialization::{ProofSerializer, ProofFormat, CompactProof},
    math,
};
use std::collections::HashSet;

/// Represents a dataset for contamination checking
struct Dataset {
    name: String,
    items: Vec<String>,
}

impl Dataset {
    fn new(name: &str, items: Vec<String>) -> Self {
        Self { name: name.to_string(), items }
    }
}

/// Hash-based private set intersection for contamination detection
struct ContaminationChecker {
    hasher: SpectaclesHasher,
    /// Hash of each test item
    test_hashes: HashSet<[u8; 32]>,
    /// N-gram hashes for fuzzy matching
    ngram_hashes: HashSet<[u8; 32]>,
    ngram_size: usize,
}

impl ContaminationChecker {
    fn new(test_data: &[String], ngram_size: usize) -> Self {
        let hasher = SpectaclesHasher::with_domain("contamination");
        
        let test_hashes: HashSet<[u8; 32]> = test_data.iter()
            .map(|item| hasher.hash(item.trim().to_lowercase().as_bytes()))
            .collect();
        
        let mut ngram_hashes = HashSet::new();
        for item in test_data {
            let words: Vec<&str> = item.split_whitespace().collect();
            if words.len() >= ngram_size {
                for i in 0..=words.len() - ngram_size {
                    let ngram = words[i..i + ngram_size].join(" ").to_lowercase();
                    ngram_hashes.insert(hasher.hash(ngram.as_bytes()));
                }
            }
        }
        
        Self {
            hasher,
            test_hashes,
            ngram_hashes,
            ngram_size,
        }
    }
    
    /// Check for exact contamination (full string match)
    fn check_exact(&self, training_data: &[String]) -> ContaminationResult {
        let mut contaminated_indices = Vec::new();
        
        for (i, item) in training_data.iter().enumerate() {
            let hash = self.hasher.hash(item.trim().to_lowercase().as_bytes());
            if self.test_hashes.contains(&hash) {
                contaminated_indices.push(i);
            }
        }
        
        let contamination_rate = if training_data.is_empty() {
            0.0
        } else {
            contaminated_indices.len() as f64 / training_data.len() as f64
        };
        
        ContaminationResult {
            method: "exact".to_string(),
            num_contaminated: contaminated_indices.len(),
            total_checked: training_data.len(),
            contamination_rate,
            contaminated_indices,
        }
    }
    
    /// Check for n-gram contamination (partial overlap)
    fn check_ngram(&self, training_data: &[String]) -> ContaminationResult {
        let mut contaminated_indices = Vec::new();
        
        for (i, item) in training_data.iter().enumerate() {
            let words: Vec<&str> = item.split_whitespace().collect();
            let mut found = false;
            
            if words.len() >= self.ngram_size {
                for j in 0..=words.len() - self.ngram_size {
                    let ngram = words[j..j + self.ngram_size].join(" ").to_lowercase();
                    let hash = self.hasher.hash(ngram.as_bytes());
                    if self.ngram_hashes.contains(&hash) {
                        found = true;
                        break;
                    }
                }
            }
            
            if found {
                contaminated_indices.push(i);
            }
        }
        
        let contamination_rate = if training_data.is_empty() {
            0.0
        } else {
            contaminated_indices.len() as f64 / training_data.len() as f64
        };
        
        ContaminationResult {
            method: format!("{}-gram", self.ngram_size),
            num_contaminated: contaminated_indices.len(),
            total_checked: training_data.len(),
            contamination_rate,
            contaminated_indices,
        }
    }
    
    /// Generate a Merkle proof of the contamination check
    fn generate_proof(&self, training_data: &[String], result: &ContaminationResult) -> Vec<u8> {
        let leaves: Vec<Vec<u8>> = training_data.iter()
            .map(|item| self.hasher.hash(item.as_bytes()).to_vec())
            .collect();
        
        let tree = MerkleTree::build(&leaves);
        
        let mut proof_data = Vec::new();
        proof_data.extend_from_slice(&tree.root());
        proof_data.extend_from_slice(&(result.num_contaminated as u64).to_le_bytes());
        proof_data.extend_from_slice(&(result.total_checked as u64).to_le_bytes());
        
        proof_data
    }
}

/// Result of a contamination check
#[derive(Debug)]
struct ContaminationResult {
    method: String,
    num_contaminated: usize,
    total_checked: usize,
    contamination_rate: f64,
    contaminated_indices: Vec<usize>,
}

fn main() {
    env_logger::init();
    
    println!("=== PSI Contamination Detection Example ===");
    println!();
    
    // 1. Create test dataset (benchmark)
    let test_data = vec![
        "What is the capital of France?".to_string(),
        "Solve: 2 + 2 = ?".to_string(),
        "Translate 'hello' to Spanish".to_string(),
        "What is the largest planet?".to_string(),
        "Who wrote Romeo and Juliet?".to_string(),
        "What is photosynthesis?".to_string(),
        "Calculate the area of a circle with radius 5".to_string(),
        "What year did World War II end?".to_string(),
    ];
    
    // 2. Create training dataset (with some contamination)
    let training_data = vec![
        "The quick brown fox jumps over the lazy dog".to_string(),
        "What is the capital of France?".to_string(),  // CONTAMINATED
        "Machine learning is a subset of AI".to_string(),
        "Solve: 2 + 2 = ?".to_string(),  // CONTAMINATED
        "The weather today is sunny and warm".to_string(),
        "Deep learning uses neural networks".to_string(),
        "Calculate the area of a circle with radius 5".to_string(),  // CONTAMINATED
        "Python is a programming language".to_string(),
        "Natural language processing handles text data".to_string(),
        "Who wrote Romeo and Juliet?".to_string(),  // CONTAMINATED
    ];
    
    println!("Test dataset:     {} items", test_data.len());
    println!("Training dataset: {} items", training_data.len());
    println!();
    
    // 3. Run exact contamination check
    println!("--- Exact Match Contamination ---");
    let checker = ContaminationChecker::new(&test_data, 5);
    let exact_result = checker.check_exact(&training_data);
    
    println!("Contaminated:     {} / {} ({:.1}%)",
        exact_result.num_contaminated,
        exact_result.total_checked,
        exact_result.contamination_rate * 100.0);
    println!("Contaminated idx: {:?}", exact_result.contaminated_indices);
    
    for &idx in &exact_result.contaminated_indices {
        println!("  [{}] {:?}", idx, training_data[idx]);
    }
    
    // 4. Run n-gram contamination check
    println!();
    println!("--- N-gram Contamination (5-gram) ---");
    let ngram_result = checker.check_ngram(&training_data);
    
    println!("Contaminated:     {} / {} ({:.1}%)",
        ngram_result.num_contaminated,
        ngram_result.total_checked,
        ngram_result.contamination_rate * 100.0);
    
    // 5. Generate proof of contamination check
    println!();
    println!("--- Proof Generation ---");
    let proof_data = checker.generate_proof(&training_data, &exact_result);
    
    let hasher = SpectaclesHasher::with_domain("contamination-proof");
    let cand_hash = hasher.hash(b"training-dataset-id");
    let ref_hash = hasher.hash(b"test-dataset-id");
    
    let proof = CompactProof::new(
        "contamination-exact".to_string(),
        cand_hash,
        ref_hash,
        exact_result.num_contaminated as u64,
        exact_result.total_checked as u64,
        hasher.hash(&proof_data),
        proof_data,
    );
    
    let serializer = ProofSerializer::compact_binary();
    let proof_bytes = serializer.serialize(&proof).unwrap();
    println!("Proof size: {} bytes", proof_bytes.len());
    
    // Verify proof roundtrip
    let decoded = serializer.deserialize(&proof_bytes).unwrap();
    println!("Decoded metric: {}", decoded.metric_id);
    println!("Contaminated/Total: {}/{}", decoded.score_numerator, decoded.score_denominator);
    
    // 6. Commitment to results
    println!();
    println!("--- Result Commitment ---");
    let result_summary = format!(
        "exact:{}/{} ngram:{}/{}",
        exact_result.num_contaminated, exact_result.total_checked,
        ngram_result.num_contaminated, ngram_result.total_checked,
    );
    
    let randomness: Vec<u8> = (0..32).map(|i| (i * 11 + 3) as u8).collect();
    let (commitment, opening) = Commitment::commit(result_summary.as_bytes(), &randomness);
    println!("Result commitment: {}", commitment.hash_hex());
    println!("Commitment valid:  {}", commitment.verify(&opening));
    
    // Summary
    println!();
    println!("=== Summary ===");
    println!("Exact contamination rate: {:.1}%", exact_result.contamination_rate * 100.0);
    println!("N-gram contamination rate: {:.1}%", ngram_result.contamination_rate * 100.0);
    
    if exact_result.contamination_rate > 0.0 {
        println!("⚠ WARNING: Contamination detected! Results may be inflated.");
    } else {
        println!("✓ No contamination detected.");
    }
}
