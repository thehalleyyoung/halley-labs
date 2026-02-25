//! Example: Certify a BLEU score with proof generation.
//!
//! Demonstrates the full pipeline: compute BLEU, verify triple agreement,
//! generate a commitment, and serialize a proof.

use spectacles_core::scoring::{
    ScoringPair, TripleMetric, GoldilocksField,
    bleu::{BleuScorer, BleuConfig, SmoothingMethod},
    differential::DifferentialTester,
};
use spectacles_core::utils::{
    hash::{SpectaclesHasher, MerkleTree, Commitment},
    serialization::{ProofSerializer, ProofFormat, CompactProof, estimate_proof_size},
};

fn main() {
    env_logger::init();
    
    println!("=== BLEU Score Certification Example ===");
    println!();
    
    // 1. Define candidate and reference translations
    let pairs = vec![
        ScoringPair {
            candidate: "the cat sat on the mat".to_string(),
            reference: "the cat is sitting on the mat".to_string(),
        },
        ScoringPair {
            candidate: "there is a dog in the house".to_string(),
            reference: "a dog is in the house".to_string(),
        },
        ScoringPair {
            candidate: "the quick brown fox jumps over the lazy dog".to_string(),
            reference: "the quick brown fox jumps over the lazy dog".to_string(),
        },
    ];
    
    // 2. Compute BLEU with triple verification
    let scorer = BleuScorer::with_smoothing(SmoothingMethod::Add1);
    
    println!("--- Per-Sentence BLEU ---");
    let mut all_agree = true;
    for (i, pair) in pairs.iter().enumerate() {
        let result = scorer.score_and_verify(pair);
        println!("[{}] BLEU={:.4} BP={:.4} Agreement={}",
            i, result.reference.score, result.reference.brevity_penalty, result.agreement);
        for (n, p) in result.reference.precisions.iter().enumerate() {
            println!("     {}-gram precision: {:.4}", n + 1, p);
        }
        if !result.agreement {
            all_agree = false;
        }
    }
    
    // 3. Compute corpus BLEU
    println!();
    println!("--- Corpus BLEU ---");
    let corpus_result = scorer.reference_score_corpus(&pairs);
    println!("Corpus BLEU:     {:.4}", corpus_result.score);
    println!("Brevity Penalty: {:.4}", corpus_result.brevity_penalty);
    println!("Candidate len:   {}", corpus_result.candidate_length);
    println!("Reference len:   {}", corpus_result.reference_length);
    
    // 4. Generate commitment
    println!();
    println!("--- Proof Generation ---");
    let hasher = SpectaclesHasher::with_domain("bleu-certification");
    
    let score_bytes = format!("{:.6}", corpus_result.score).into_bytes();
    let randomness: Vec<u8> = (0..32).map(|i| (i * 7 + 13) as u8).collect();
    let (commitment, opening) = Commitment::commit(&score_bytes, &randomness);
    
    println!("Score commitment: {}", commitment.hash_hex());
    println!("Commitment valid: {}", commitment.verify(&opening));
    
    // 5. Build Merkle tree of individual scores
    let score_leaves: Vec<Vec<u8>> = pairs.iter().map(|pair| {
        let r = scorer.reference_score_sentence(&pair.candidate, &pair.reference);
        format!("{:.6}", r.score).into_bytes()
    }).collect();
    
    let tree = MerkleTree::build(&score_leaves);
    println!("Merkle root:      {}", hex::encode(tree.root()));
    
    // Verify Merkle proofs
    for i in 0..score_leaves.len() {
        let proof = tree.proof(i);
        let valid = MerkleTree::verify_proof(&proof, &score_leaves[i]);
        println!("Leaf {} proof:     {}", i, if valid { "VALID" } else { "INVALID" });
    }
    
    // 6. Serialize proof
    let compact_proof = CompactProof::new(
        "bleu-add1".to_string(),
        hasher.hash(pairs[0].candidate.as_bytes()),
        hasher.hash(pairs[0].reference.as_bytes()),
        (corpus_result.score * 10000.0).round() as u64,
        10000,
        *commitment.hash(),
        tree.root().to_vec(),
    );
    
    let serializer = ProofSerializer::compact_binary();
    let proof_bytes = serializer.serialize(&compact_proof).unwrap();
    println!();
    println!("Proof size:       {} bytes", proof_bytes.len());
    
    let estimated = estimate_proof_size(100, 50, 128);
    println!("Estimated STARK:  {} bytes", estimated);
    
    // 7. Verify serialized proof
    let decoded = serializer.deserialize(&proof_bytes).unwrap();
    println!("Decoded metric:   {}", decoded.metric_id);
    println!("Decoded score:    {}/{}", decoded.score_numerator, decoded.score_denominator);
    
    println!();
    if all_agree {
        println!("✓ All triple implementations agree.");
        println!("✓ BLEU certification complete.");
    } else {
        println!("✗ Triple implementation disagreement detected!");
    }
}
