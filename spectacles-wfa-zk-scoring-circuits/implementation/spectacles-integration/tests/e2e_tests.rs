//! End-to-end integration tests for the Spectacles framework.
//!
//! Tests each metric through the full pipeline:
//! EvalSpec → Scoring → Triple Verification → Certificate

use spectacles_core::scoring::{
    self, ScoringPair, TripleMetric, GoldilocksField,
    exact_match::{ExactMatchScorer, NormalizedExactMatchScorer, MultiAnswerExactMatchScorer, ExactMatchConfig},
    token_f1::{TokenF1Scorer, MacroF1Scorer, MicroF1Scorer, F1Score, TokenF1Config},
    bleu::{BleuScorer, BleuConfig, SmoothingMethod},
    rouge::{RougeNScorer, RougeLScorer, RougeConfig},
    regex_match::{RegexMatchScorer, RegexCompiler, Nfa, Dfa},
    pass_at_k::{PassAtKScorer, PassAtKConfig},
    differential::{DifferentialTester, standard_test_suite, random_test_pairs},
};
use spectacles_core::utils::{
    hash::{SpectaclesHasher, MerkleTree, HashChain, Commitment},
    serialization::{ProofSerializer, ProofFormat, CompactProof},
    math::{extended_gcd, mod_pow, mod_inv, polynomial_eval, lagrange_interpolate},
};
use spectacles_integration::run_scoring_pipeline;

// ============================================================
// Full Pipeline Tests
// ============================================================

#[test]
fn test_full_pipeline_exact_match() {
    let result = run_scoring_pipeline("hello world", "hello world", "exact_match");
    assert_eq!(result.score, "true");
    assert!(result.triple_agreement);
    assert!(!result.commitment.is_empty());
}

#[test]
fn test_full_pipeline_token_f1() {
    let result = run_scoring_pipeline("the cat sat on the mat", "the cat sat on the mat", "token_f1");
    assert_eq!(result.score, "1.0000");
    assert!(result.triple_agreement);
}

#[test]
fn test_full_pipeline_bleu() {
    let result = run_scoring_pipeline("the cat sat on the mat", "the cat sat on the mat", "bleu");
    assert!(result.triple_agreement);
    let score: f64 = result.score.parse().unwrap();
    assert!(score > 0.9);
}

#[test]
fn test_full_pipeline_rouge1() {
    let result = run_scoring_pipeline("the cat sat", "the cat sat on", "rouge1");
    assert!(result.triple_agreement);
}

#[test]
fn test_full_pipeline_rouge_l() {
    let result = run_scoring_pipeline("the cat sat", "the cat sat", "rouge_l");
    assert!(result.triple_agreement);
}

// ============================================================
// Differential Testing Integration
// ============================================================

#[test]
fn test_differential_all_metrics_standard_suite() {
    let tester = DifferentialTester::new();
    let pairs = standard_test_suite();
    let reports = tester.test_all_metrics(&pairs);
    
    for (metric, report) in &reports {
        assert!(report.is_perfect(),
            "Metric {} failed: {} disagreements out of {} tests",
            metric, report.disagreements, report.total_tests);
    }
}

#[test]
fn test_differential_all_metrics_random() {
    let tester = DifferentialTester::new();
    let pairs = random_test_pairs(100, 54321);
    let reports = tester.test_all_metrics(&pairs);
    
    for (metric, report) in &reports {
        assert!(report.agreement_rate >= 0.99,
            "Metric {} agreement rate {:.2}% is too low",
            metric, report.agreement_rate * 100.0);
    }
}

// ============================================================
// Metric-specific E2E Tests
// ============================================================

#[test]
fn test_exact_match_e2e() {
    let scorer = ExactMatchScorer::case_sensitive();
    let test_cases = vec![
        ("hello", "hello", true),
        ("Hello", "hello", false),
        ("", "", true),
        ("a", "b", false),
    ];
    
    for (cand, ref_str, expected) in test_cases {
        let pair = ScoringPair { candidate: cand.to_string(), reference: ref_str.to_string() };
        let result = scorer.score_and_verify(&pair);
        assert!(result.agreement, "Triple disagreement on ({}, {})", cand, ref_str);
        assert_eq!(result.reference, expected, "Wrong score for ({}, {})", cand, ref_str);
    }
}

#[test]
fn test_normalized_exact_match_e2e() {
    let scorer = NormalizedExactMatchScorer::new();
    assert!(scorer.score("Hello, World!", "hello world"));
    assert!(!scorer.score("hello", "world"));
}

#[test]
fn test_multi_answer_exact_match_e2e() {
    let scorer = MultiAnswerExactMatchScorer::case_insensitive();
    let result = scorer.score_and_verify("Paris", &["paris", "Paris", "PARIS"]);
    assert!(result.agreement);
    assert!(result.reference);
}

#[test]
fn test_token_f1_e2e() {
    let scorer = TokenF1Scorer::default_scorer();
    
    let pair = ScoringPair {
        candidate: "the cat sat on the mat".to_string(),
        reference: "the cat sat on the mat".to_string(),
    };
    let result = scorer.score_and_verify(&pair);
    assert!(result.agreement);
    assert!((result.reference.f1 - 1.0).abs() < 1e-10);
    
    let pair2 = ScoringPair {
        candidate: "the cat sat".to_string(),
        reference: "the dog sat".to_string(),
    };
    let result2 = scorer.score_and_verify(&pair2);
    assert!(result2.agreement);
    assert!((result2.reference.f1 - 2.0 / 3.0).abs() < 1e-10);
}

#[test]
fn test_macro_micro_f1_e2e() {
    let pairs = vec![
        ScoringPair { candidate: "a b".to_string(), reference: "a b".to_string() },
        ScoringPair { candidate: "x".to_string(), reference: "y".to_string() },
    ];
    
    let macro_scorer = MacroF1Scorer::default_scorer();
    let macro_f1 = macro_scorer.reference_score(&pairs);
    assert!((macro_f1.f1 - 0.5).abs() < 1e-10);
    
    let micro_scorer = MicroF1Scorer::default_scorer();
    let micro_f1 = micro_scorer.reference_score(&pairs);
    assert!(micro_f1.f1 > 0.0 && micro_f1.f1 < 1.0);
}

#[test]
fn test_bleu_e2e() {
    let scorer = BleuScorer::with_smoothing(SmoothingMethod::Add1);
    
    let pair = ScoringPair {
        candidate: "the cat sat on the mat".to_string(),
        reference: "the cat sat on the mat".to_string(),
    };
    let result = scorer.score_and_verify(&pair);
    assert!(result.agreement);
    assert!(result.reference.score > 0.9);
}

#[test]
fn test_bleu_corpus_e2e() {
    let scorer = BleuScorer::with_smoothing(SmoothingMethod::Add1);
    let pairs = vec![
        ScoringPair { candidate: "the cat sat".to_string(), reference: "the cat sat on".to_string() },
        ScoringPair { candidate: "hello world".to_string(), reference: "hello world".to_string() },
    ];
    let result = scorer.reference_score_corpus(&pairs);
    assert!(result.score > 0.0);
}

#[test]
fn test_rouge_e2e() {
    let scorer1 = RougeNScorer::rouge1();
    let pair = ScoringPair {
        candidate: "the cat sat".to_string(),
        reference: "the dog sat on".to_string(),
    };
    let result = scorer1.score_and_verify(&pair);
    assert!(result.agreement);
    
    let scorer_l = RougeLScorer::default_scorer();
    let result_l = scorer_l.score_and_verify(&pair);
    assert!(result_l.agreement);
}

#[test]
fn test_regex_match_e2e() {
    let scorer = RegexMatchScorer::new("(ab)+c").unwrap();
    let pair_match = ScoringPair { candidate: "abc".to_string(), reference: "".to_string() };
    let pair_no = ScoringPair { candidate: "ac".to_string(), reference: "".to_string() };
    
    let r1 = scorer.score_and_verify(&pair_match);
    assert!(r1.agreement);
    assert!(r1.reference);
    
    let r2 = scorer.score_and_verify(&pair_no);
    assert!(r2.agreement);
    assert!(!r2.reference);
}

#[test]
fn test_pass_at_k_e2e() {
    let scorer = PassAtKScorer::pass_at_1();
    let samples = vec![
        vec!["42".to_string()],
        vec!["43".to_string()],
        vec!["42".to_string()],
    ];
    let expected = vec!["42".to_string()];
    
    let ref_result = scorer.reference_score(&samples, &expected);
    let aut_result = scorer.automaton_score(&samples, &expected);
    let cir_result = scorer.circuit_score(&samples, &expected);
    
    assert_eq!(ref_result.c, aut_result.c);
    assert_eq!(aut_result.c, cir_result.c);
    assert!((ref_result.score - aut_result.score).abs() < 1e-10);
}

// ============================================================
// Certificate Generation Tests
// ============================================================

#[test]
fn test_certificate_generation() {
    let hasher = SpectaclesHasher::with_domain("scoring");
    
    let candidate = "the cat sat on the mat";
    let reference = "the cat sat on the mat";
    
    let cand_hash = hasher.hash(candidate.as_bytes());
    let ref_hash = hasher.hash(reference.as_bytes());
    
    let scorer = BleuScorer::with_smoothing(SmoothingMethod::Add1);
    let pair = ScoringPair {
        candidate: candidate.to_string(),
        reference: reference.to_string(),
    };
    let bleu_result = scorer.score_and_verify(&pair);
    assert!(bleu_result.agreement);
    
    let score_num = (bleu_result.reference.score * 10000.0).round() as u64;
    let score_den = 10000u64;
    
    let proof = CompactProof::new(
        "bleu".to_string(),
        cand_hash,
        ref_hash,
        score_num,
        score_den,
        [0u8; 32],
        vec![1, 2, 3, 4],
    );
    
    let serializer = ProofSerializer::compact_binary();
    let bytes = serializer.serialize(&proof).unwrap();
    let decoded = serializer.deserialize(&bytes).unwrap();
    
    assert_eq!(decoded.metric_id, "bleu");
    assert_eq!(decoded.score_numerator, score_num);
}

#[test]
fn test_merkle_tree_certificate() {
    let leaves: Vec<Vec<u8>> = (0..8).map(|i| {
        let hasher = SpectaclesHasher::new();
        hasher.hash(format!("data-{}", i).as_bytes()).to_vec()
    }).collect();
    
    let tree = MerkleTree::build(&leaves);
    
    for i in 0..8 {
        let proof = tree.proof(i);
        assert!(MerkleTree::verify_proof(&proof, &leaves[i]));
    }
}

// ============================================================
// Utils Integration Tests
// ============================================================

#[test]
fn test_hash_chain_integrity() {
    let chain = HashChain::build(b"spectacles-seed", 100);
    assert!(chain.verify_all());
}

#[test]
fn test_commitment_integrity() {
    let (commitment, opening) = Commitment::commit(b"secret score", b"randomness123456");
    assert!(commitment.verify(&opening));
}

#[test]
fn test_math_polynomial_consistency() {
    let coeffs = vec![1.0, 2.0, 3.0]; // 1 + 2x + 3x^2
    let x = 5.0;
    let result = polynomial_eval(&coeffs, x);
    assert!((result - 86.0).abs() < 1e-10); // 1 + 10 + 75
}

#[test]
fn test_math_lagrange_consistency() {
    let points = vec![(1.0, 1.0), (2.0, 4.0), (3.0, 9.0)]; // y = x^2
    let result = lagrange_interpolate(&points, 4.0);
    assert!((result - 16.0).abs() < 1e-10);
}

#[test]
fn test_goldilocks_field_operations() {
    let a = GoldilocksField::new(12345);
    let b = GoldilocksField::new(67890);
    
    let sum = a.add(b);
    let prod = a.mul(b);
    
    let inv_a = a.inv();
    assert_eq!(a.mul(inv_a), GoldilocksField::one());
    
    let div_result = prod.div(b);
    assert_eq!(div_result, a);
}

#[test]
fn test_proof_serialization_roundtrip() {
    let proof = CompactProof::new(
        "token_f1".to_string(),
        [42u8; 32],
        [43u8; 32],
        750,
        1000,
        [44u8; 32],
        vec![10, 20, 30],
    );
    
    // Test binary roundtrip
    let binary_serializer = ProofSerializer::compact_binary();
    let bytes = binary_serializer.serialize(&proof).unwrap();
    let decoded = binary_serializer.deserialize(&bytes).unwrap();
    assert_eq!(decoded.metric_id, proof.metric_id);
    assert_eq!(decoded.score_numerator, proof.score_numerator);
    
    // Test JSON roundtrip
    let json_serializer = ProofSerializer::json();
    let json_bytes = json_serializer.serialize(&proof).unwrap();
    let json_decoded = json_serializer.deserialize(&json_bytes).unwrap();
    assert_eq!(json_decoded.metric_id, proof.metric_id);
}

// ============================================================
// Cross-cutting Integration Tests
// ============================================================

#[test]
fn test_scoring_with_commitment() {
    let scorer = TokenF1Scorer::default_scorer();
    let pair = ScoringPair {
        candidate: "the quick brown fox".to_string(),
        reference: "the slow brown cat".to_string(),
    };
    
    let result = scorer.score_and_verify(&pair);
    assert!(result.agreement);
    
    let score_bytes = format!("{:.6}", result.reference.f1).into_bytes();
    let randomness = b"unique-random-value-12345678";
    
    let (commitment, opening) = Commitment::commit(&score_bytes, randomness);
    assert!(commitment.verify(&opening));
}

#[test]
fn test_all_metrics_on_same_input() {
    let pair = ScoringPair {
        candidate: "the quick brown fox jumps".to_string(),
        reference: "the slow brown cat jumps".to_string(),
    };
    
    let em = ExactMatchScorer::case_sensitive();
    let f1 = TokenF1Scorer::default_scorer();
    let bleu = BleuScorer::with_smoothing(SmoothingMethod::Add1);
    let r1 = RougeNScorer::rouge1();
    let rl = RougeLScorer::default_scorer();
    
    let em_result = em.score_and_verify(&pair);
    let f1_result = f1.score_and_verify(&pair);
    let bleu_result = bleu.score_and_verify(&pair);
    let r1_result = r1.score_and_verify(&pair);
    let rl_result = rl.score_and_verify(&pair);
    
    assert!(em_result.agreement);
    assert!(f1_result.agreement);
    assert!(bleu_result.agreement);
    assert!(r1_result.agreement);
    assert!(rl_result.agreement);
    
    assert!(!em_result.reference); // Not an exact match
    assert!(f1_result.reference.f1 > 0.0); // Some overlap
    assert!(r1_result.reference.f1 > 0.0);
    assert!(rl_result.reference.f1 > 0.0);
}

// ============================================================
// Real Benchmark Pipeline Integration Tests
// ============================================================

#[cfg(test)]
mod real_benchmark_tests {
    use spectacles_core::scoring::{
        ScoringPair, TripleMetric,
        exact_match::ExactMatchScorer,
        token_f1::TokenF1Scorer,
        bleu::{BleuScorer, SmoothingMethod},
    };
    use spectacles_core::psi::protocol::{CommitmentBinding, PSIConfig, PSIProtocol};
    use spectacles_core::psi::ngram::{NGramExtractor, NGramSet, NGramConfig};
    use spectacles_core::utils::hash::Commitment;
    use spectacles_core::utils::serialization::{ProofSerializer, CompactProof};

    /// End-to-end test with real MMLU-style benchmark data.
    /// Exercises the complete pipeline:
    /// 1. Define benchmark questions and model outputs
    /// 2. Compute exact_match score with triple verification
    /// 3. Generate n-gram fingerprints for contamination check
    /// 4. Create commitment binding for PSI inputs
    /// 5. Run PSI protocol (simulated, local)
    /// 6. Verify contamination bound
    /// 7. Generate final certificate
    #[test]
    fn test_mmlu_style_benchmark_pipeline() {
        // --- Step 1: MMLU-style multiple-choice Q&A pairs ---
        let benchmark_items = vec![
            ScoringPair { candidate: "A".to_string(), reference: "A".to_string() },
            ScoringPair { candidate: "B".to_string(), reference: "C".to_string() },
            ScoringPair { candidate: "D".to_string(), reference: "D".to_string() },
            ScoringPair { candidate: "A".to_string(), reference: "B".to_string() },
            ScoringPair { candidate: "C".to_string(), reference: "C".to_string() },
        ];

        // --- Step 2: Exact-match scoring with triple verification ---
        let scorer = ExactMatchScorer::case_sensitive();
        let mut correct = 0usize;
        for pair in &benchmark_items {
            let result = scorer.score_and_verify(pair);
            assert!(result.agreement, "Triple verification must agree");
            if result.reference {
                correct += 1;
            }
        }
        let accuracy = correct as f64 / benchmark_items.len() as f64;
        assert!((accuracy - 0.6).abs() < 1e-10, "Expected 3/5 = 0.6 accuracy");

        // --- Step 3: N-gram fingerprints for contamination detection ---
        let benchmark_corpus: String = benchmark_items
            .iter()
            .map(|p| p.reference.clone())
            .collect::<Vec<_>>()
            .join(" ");
        let model_corpus: String = benchmark_items
            .iter()
            .map(|p| p.candidate.clone())
            .collect::<Vec<_>>()
            .join(" ");

        let ngram_config = NGramConfig::char_ngrams(2);
        let benchmark_set = NGramSet::from_text(&benchmark_corpus, ngram_config.clone());
        let model_set = NGramSet::from_text(&model_corpus, ngram_config.clone());
        assert!(benchmark_set.count > 0);
        assert!(model_set.count > 0);

        // --- Step 4: CommitmentBinding before PSI ---
        let binding = CommitmentBinding::commit_ngram_set(&benchmark_set);
        assert_eq!(binding.set_size, benchmark_set.grams.len());
        assert!(binding.verify_ngram_set(&benchmark_set), "Commitment must verify against original set");
        assert!(!binding.verify_ngram_set(&model_set), "Commitment must NOT verify against a different set");

        // --- Step 5: PSI protocol (local simulation) ---
        let psi_config = PSIConfig::threshold(0.5);
        let psi = PSIProtocol::new(psi_config);
        let psi_result = psi.run_local(&benchmark_set, &model_set);
        assert!(psi_result.intersection_cardinality <= benchmark_set.count);
        assert!(psi_result.contamination_score >= 0.0);
        assert!(psi_result.contamination_score <= 1.0);

        // --- Step 6: Verify contamination attestation ---
        let attestation = psi.generate_attestation(&psi_result);
        assert!(PSIProtocol::verify_attestation(&attestation));

        // --- Step 7: Generate final certificate ---
        let score_num = (accuracy * 10000.0).round() as u64;
        let score_den = 10000u64;
        let proof = CompactProof::new(
            "exact_match".to_string(),
            [0u8; 32],
            [0u8; 32],
            score_num,
            score_den,
            attestation.protocol_hash,
            Vec::new(),
        );
        let serializer = ProofSerializer::compact_binary();
        let bytes = serializer.serialize(&proof).unwrap();
        let decoded = serializer.deserialize(&bytes).unwrap();
        assert_eq!(decoded.metric_id, "exact_match");
        assert_eq!(decoded.score_numerator, 6000);
        assert_eq!(decoded.score_denominator, 10000);
    }

    /// SQuAD-style extractive QA with Token F1 scoring.
    /// Tests longer text answers through the full pipeline including
    /// commitment binding and PSI contamination check.
    #[test]
    fn test_squad_style_token_f1_pipeline() {
        // --- SQuAD-style question-answer pairs ---
        let qa_pairs = vec![
            ScoringPair {
                candidate: "the Eiffel Tower is located in Paris France".to_string(),
                reference: "the Eiffel Tower is in Paris".to_string(),
            },
            ScoringPair {
                candidate: "photosynthesis converts sunlight into chemical energy".to_string(),
                reference: "photosynthesis converts light energy into chemical energy in plants".to_string(),
            },
            ScoringPair {
                candidate: "water freezes at zero degrees Celsius".to_string(),
                reference: "water freezes at zero degrees Celsius".to_string(),
            },
            ScoringPair {
                candidate: "the mitochondria is the powerhouse of the cell".to_string(),
                reference: "mitochondria are the powerhouse of the cell".to_string(),
            },
        ];

        // --- Token F1 scoring with triple verification ---
        let scorer = TokenF1Scorer::default_scorer();
        let mut f1_scores = Vec::new();
        for pair in &qa_pairs {
            let result = scorer.score_and_verify(pair);
            assert!(result.agreement, "Triple verification must agree for: {}", pair.candidate);
            f1_scores.push(result.reference.f1);
        }

        // Perfect match should yield F1 = 1.0
        assert!((f1_scores[2] - 1.0).abs() < 1e-10, "Perfect match should have F1=1.0");
        // All partial matches should have positive F1
        for (i, &f1) in f1_scores.iter().enumerate() {
            assert!(f1 > 0.0, "Pair {} should have positive F1", i);
        }

        // --- Macro average F1 ---
        let macro_f1: f64 = f1_scores.iter().sum::<f64>() / f1_scores.len() as f64;
        assert!(macro_f1 > 0.5, "Macro F1 should be > 0.5 for mostly-correct answers");

        // --- Contamination check via PSI + CommitmentBinding ---
        let all_references: String = qa_pairs
            .iter()
            .map(|p| p.reference.clone())
            .collect::<Vec<_>>()
            .join(" ");
        let all_candidates: String = qa_pairs
            .iter()
            .map(|p| p.candidate.clone())
            .collect::<Vec<_>>()
            .join(" ");

        let config = NGramConfig::char_ngrams(5);
        let ref_set = NGramSet::from_text(&all_references, config.clone());
        let cand_set = NGramSet::from_text(&all_candidates, config.clone());

        // Commit to reference set before PSI
        let binding = CommitmentBinding::commit_ngram_set(&ref_set);
        assert!(binding.verify_ngram_set(&ref_set));
        assert_eq!(binding.set_size, ref_set.grams.len());

        let psi = PSIProtocol::new(PSIConfig::cardinality_only());
        let psi_result = psi.run_local(&ref_set, &cand_set);
        assert!(psi_result.contamination_score <= 1.0);

        let attestation = psi.generate_attestation(&psi_result);
        assert!(PSIProtocol::verify_attestation(&attestation));

        // --- Certificate with F1 score ---
        let score_num = (macro_f1 * 10000.0).round() as u64;
        let proof = CompactProof::new(
            "token_f1".to_string(),
            [1u8; 32],
            [2u8; 32],
            score_num,
            10000,
            attestation.protocol_hash,
            Vec::new(),
        );
        let serializer = ProofSerializer::json();
        let json = serializer.serialize(&proof).unwrap();
        let decoded = serializer.deserialize(&json).unwrap();
        assert_eq!(decoded.metric_id, "token_f1");
        assert_eq!(decoded.score_numerator, score_num);
    }

    /// Translation-style benchmark with BLEU scoring.
    /// Tests the full pipeline on longer sentence pairs typical of
    /// machine translation evaluation (WMT-style).
    #[test]
    fn test_translation_bleu_pipeline() {
        // --- WMT-style translation pairs (candidate vs reference) ---
        let translation_pairs = vec![
            ScoringPair {
                candidate: "the cat sat on the mat in the room".to_string(),
                reference: "the cat sat on the mat in the room".to_string(),
            },
            ScoringPair {
                candidate: "the dog ran through the park quickly".to_string(),
                reference: "the dog ran quickly through the large park".to_string(),
            },
            ScoringPair {
                candidate: "she went to the store to buy some groceries for dinner".to_string(),
                reference: "she visited the store to purchase groceries for dinner tonight".to_string(),
            },
            ScoringPair {
                candidate: "the weather is nice today".to_string(),
                reference: "today the weather is very nice".to_string(),
            },
        ];

        // --- BLEU scoring with triple verification ---
        let scorer = BleuScorer::with_smoothing(SmoothingMethod::Add1);
        let mut bleu_scores = Vec::new();
        for pair in &translation_pairs {
            let result = scorer.score_and_verify(pair);
            assert!(result.agreement, "Triple verification must agree for BLEU");
            bleu_scores.push(result.reference.score);
        }

        // Perfect match should yield BLEU close to 1.0
        assert!(bleu_scores[0] > 0.95, "Perfect match should have BLEU > 0.95");
        // Partial overlaps should still be positive
        for (i, &score) in bleu_scores.iter().enumerate() {
            assert!(score > 0.0, "Translation pair {} should have positive BLEU", i);
        }

        // --- Corpus-level BLEU ---
        let corpus_result = scorer.reference_score_corpus(&translation_pairs);
        assert!(corpus_result.score > 0.0, "Corpus BLEU must be positive");

        // --- Contamination check with commitment binding ---
        let ref_corpus: String = translation_pairs
            .iter()
            .map(|p| p.reference.clone())
            .collect::<Vec<_>>()
            .join(" ");
        let cand_corpus: String = translation_pairs
            .iter()
            .map(|p| p.candidate.clone())
            .collect::<Vec<_>>()
            .join(" ");

        let config = NGramConfig::char_ngrams(4);
        let ref_set = NGramSet::from_text(&ref_corpus, config.clone());
        let cand_set = NGramSet::from_text(&cand_corpus, config.clone());

        // CommitmentBinding: commit, then run PSI
        let ref_binding = CommitmentBinding::commit_ngram_set(&ref_set);
        let cand_binding = CommitmentBinding::commit_ngram_set(&cand_set);

        // Each binding verifies only against its own set
        assert!(ref_binding.verify_ngram_set(&ref_set));
        assert!(cand_binding.verify_ngram_set(&cand_set));
        assert!(!ref_binding.verify_ngram_set(&cand_set));
        assert!(!cand_binding.verify_ngram_set(&ref_set));

        // Raw fingerprint verification also works
        let ref_fps = ref_set.to_sorted_vec();
        assert!(ref_binding.verify(&ref_fps));

        let psi = PSIProtocol::new(PSIConfig::default());
        let psi_result = psi.run_local(&ref_set, &cand_set);

        // Translation data has significant n-gram overlap
        assert!(psi_result.intersection_cardinality > 0);
        assert!(psi_result.contamination_score >= 0.0);

        let attestation = psi.generate_attestation(&psi_result);
        assert!(PSIProtocol::verify_attestation(&attestation));

        // --- Final certificate ---
        let score_num = (corpus_result.score * 10000.0).round() as u64;
        let proof = CompactProof::new(
            "bleu".to_string(),
            [3u8; 32],
            [4u8; 32],
            score_num,
            10000,
            attestation.protocol_hash,
            Vec::new(),
        );
        let serializer = ProofSerializer::compact_binary();
        let bytes = serializer.serialize(&proof).unwrap();
        let decoded = serializer.deserialize(&bytes).unwrap();
        assert_eq!(decoded.metric_id, "bleu");
        assert_eq!(decoded.score_numerator, score_num);
        assert_eq!(decoded.score_denominator, 10000);
    }
}
