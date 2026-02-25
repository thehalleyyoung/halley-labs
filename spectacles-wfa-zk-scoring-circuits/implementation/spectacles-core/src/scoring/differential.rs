//! Differential testing framework for triple implementations.
//!
//! Tests that reference, automaton, and circuit implementations agree
//! on random and structured inputs.

use std::collections::{HashMap, HashSet};
use super::{
    TripleMetric, ScoringPair, FixedPointScore,
    exact_match::ExactMatchScorer,
    token_f1::{TokenF1Scorer, F1Score},
    bleu::{BleuScorer, BleuResult, SmoothingMethod},
    rouge::{RougeNScorer, RougeLScorer, RougeScore},
};
use serde::{Serialize, Deserialize};

/// Result of running all three implementations on one input
#[derive(Debug, Clone)]
pub struct DifferentialResult<S> {
    pub reference: S,
    pub automaton: S,
    pub circuit: S,
    pub agreement: bool,
}

/// Summary of differential testing across many inputs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgreementReport {
    pub total_tests: usize,
    pub agreements: usize,
    pub disagreements: usize,
    pub agreement_rate: f64,
    pub disagreement_details: Vec<DisagreementDetail>,
    pub coverage: CoverageReport,
}

impl AgreementReport {
    pub fn is_perfect(&self) -> bool {
        self.disagreements == 0
    }
}

/// Details of a single disagreement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisagreementDetail {
    pub test_index: usize,
    pub candidate: String,
    pub reference: String,
    pub metric: String,
    pub description: String,
}

/// Coverage tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageReport {
    pub empty_inputs: bool,
    pub single_token: bool,
    pub multi_token: bool,
    pub exact_match_true: bool,
    pub exact_match_false: bool,
    pub partial_overlap: bool,
    pub length_mismatch: bool,
    pub unicode_inputs: bool,
    pub repeated_tokens: bool,
}

impl Default for CoverageReport {
    fn default() -> Self {
        Self {
            empty_inputs: false,
            single_token: false,
            multi_token: false,
            exact_match_true: false,
            exact_match_false: false,
            partial_overlap: false,
            length_mismatch: false,
            unicode_inputs: false,
            repeated_tokens: false,
        }
    }
}

/// Differential tester that runs all three implementations and checks agreement
#[derive(Debug)]
pub struct DifferentialTester {
    tolerance: f64,
    max_disagreements: usize,
}

impl DifferentialTester {
    pub fn new() -> Self {
        Self {
            tolerance: 1e-7, // 2^-24 ≈ 5.96e-8
            max_disagreements: 100,
        }
    }
    
    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }
    
    pub fn with_max_disagreements(mut self, max: usize) -> Self {
        self.max_disagreements = max;
        self
    }
    
    /// Update coverage report based on a pair
    fn update_coverage(coverage: &mut CoverageReport, pair: &ScoringPair) {
        let cand_words: Vec<&str> = pair.candidate.split_whitespace().collect();
        let ref_words: Vec<&str> = pair.reference.split_whitespace().collect();
        
        if pair.candidate.is_empty() || pair.reference.is_empty() {
            coverage.empty_inputs = true;
        }
        if cand_words.len() == 1 || ref_words.len() == 1 {
            coverage.single_token = true;
        }
        if cand_words.len() > 1 && ref_words.len() > 1 {
            coverage.multi_token = true;
        }
        if pair.candidate == pair.reference {
            coverage.exact_match_true = true;
        } else {
            coverage.exact_match_false = true;
        }
        if cand_words.len() != ref_words.len() {
            coverage.length_mismatch = true;
        }
        
        // Check for partial overlap
        let cand_set: std::collections::HashSet<&str> = cand_words.iter().copied().collect();
        let ref_set: std::collections::HashSet<&str> = ref_words.iter().copied().collect();
        let overlap = cand_set.intersection(&ref_set).count();
        if overlap > 0 && overlap < cand_set.len().max(ref_set.len()) {
            coverage.partial_overlap = true;
        }
        
        // Check for unicode
        if pair.candidate.chars().any(|c| !c.is_ascii()) || pair.reference.chars().any(|c| !c.is_ascii()) {
            coverage.unicode_inputs = true;
        }
        
        // Check for repeated tokens
        let mut seen = std::collections::HashSet::new();
        for w in &cand_words {
            if !seen.insert(w) {
                coverage.repeated_tokens = true;
                break;
            }
        }
    }
    
    /// Test exact match across all three implementations
    pub fn test_exact_match(&self, pairs: &[ScoringPair]) -> AgreementReport {
        let scorer = ExactMatchScorer::case_sensitive();
        self.run_tests(&scorer, pairs, "exact_match")
    }
    
    /// Test token F1 across all three implementations
    pub fn test_token_f1(&self, pairs: &[ScoringPair]) -> AgreementReport {
        let scorer = TokenF1Scorer::default_scorer();
        self.run_f1_tests(&scorer, pairs, "token_f1")
    }
    
    /// Test BLEU across all three implementations
    pub fn test_bleu(&self, pairs: &[ScoringPair]) -> AgreementReport {
        let scorer = BleuScorer::with_smoothing(SmoothingMethod::Add1);
        self.run_bleu_tests(&scorer, pairs, "bleu")
    }
    
    /// Test ROUGE-1 across all three implementations
    pub fn test_rouge1(&self, pairs: &[ScoringPair]) -> AgreementReport {
        let scorer = RougeNScorer::rouge1();
        self.run_rouge_tests(&scorer, pairs, "rouge1")
    }
    
    /// Test ROUGE-L across all three implementations
    pub fn test_rouge_l(&self, pairs: &[ScoringPair]) -> AgreementReport {
        let scorer = RougeLScorer::default_scorer();
        self.run_rouge_tests_l(&scorer, pairs, "rouge_l")
    }
    
    /// Run tests for a boolean-valued metric
    fn run_tests(
        &self,
        scorer: &ExactMatchScorer,
        pairs: &[ScoringPair],
        metric_name: &str,
    ) -> AgreementReport {
        let mut agreements = 0;
        let mut disagreements = Vec::new();
        let mut coverage = CoverageReport::default();
        
        for (i, pair) in pairs.iter().enumerate() {
            Self::update_coverage(&mut coverage, pair);
            
            let result = scorer.score_and_verify(pair);
            
            if result.agreement {
                agreements += 1;
            } else if disagreements.len() < self.max_disagreements {
                disagreements.push(DisagreementDetail {
                    test_index: i,
                    candidate: pair.candidate.clone(),
                    reference: pair.reference.clone(),
                    metric: metric_name.to_string(),
                    description: format!(
                        "ref={}, aut={}, cir={}",
                        result.reference, result.automaton, result.circuit
                    ),
                });
            }
        }
        
        let total = pairs.len();
        let num_disagreements = total - agreements;
        
        AgreementReport {
            total_tests: total,
            agreements,
            disagreements: num_disagreements,
            agreement_rate: if total > 0 { agreements as f64 / total as f64 } else { 1.0 },
            disagreement_details: disagreements,
            coverage,
        }
    }
    
    /// Run tests for F1-valued metric
    fn run_f1_tests(
        &self,
        scorer: &TokenF1Scorer,
        pairs: &[ScoringPair],
        metric_name: &str,
    ) -> AgreementReport {
        let mut agreements = 0;
        let mut disagreements = Vec::new();
        let mut coverage = CoverageReport::default();
        
        for (i, pair) in pairs.iter().enumerate() {
            Self::update_coverage(&mut coverage, pair);
            
            let result = scorer.score_and_verify(pair);
            
            if result.agreement {
                agreements += 1;
            } else if disagreements.len() < self.max_disagreements {
                disagreements.push(DisagreementDetail {
                    test_index: i,
                    candidate: pair.candidate.clone(),
                    reference: pair.reference.clone(),
                    metric: metric_name.to_string(),
                    description: format!(
                        "ref={:?}, aut={:?}, cir={:?}",
                        result.reference, result.automaton, result.circuit
                    ),
                });
            }
        }
        
        let total = pairs.len();
        let num_disagreements = total - agreements;
        
        AgreementReport {
            total_tests: total,
            agreements,
            disagreements: num_disagreements,
            agreement_rate: if total > 0 { agreements as f64 / total as f64 } else { 1.0 },
            disagreement_details: disagreements,
            coverage,
        }
    }
    
    /// Run tests for BLEU
    fn run_bleu_tests(
        &self,
        scorer: &BleuScorer,
        pairs: &[ScoringPair],
        metric_name: &str,
    ) -> AgreementReport {
        let mut agreements = 0;
        let mut disagreements = Vec::new();
        let mut coverage = CoverageReport::default();
        
        for (i, pair) in pairs.iter().enumerate() {
            Self::update_coverage(&mut coverage, pair);
            
            let result = scorer.score_and_verify(pair);
            
            let agree = (result.reference.score - result.automaton.score).abs() < self.tolerance
                && (result.automaton.score - result.circuit.score).abs() < self.tolerance;
            
            if agree {
                agreements += 1;
            } else if disagreements.len() < self.max_disagreements {
                disagreements.push(DisagreementDetail {
                    test_index: i,
                    candidate: pair.candidate.clone(),
                    reference: pair.reference.clone(),
                    metric: metric_name.to_string(),
                    description: format!(
                        "ref={:.6}, aut={:.6}, cir={:.6}",
                        result.reference.score, result.automaton.score, result.circuit.score
                    ),
                });
            }
        }
        
        let total = pairs.len();
        let num_disagreements = total - agreements;
        
        AgreementReport {
            total_tests: total,
            agreements,
            disagreements: num_disagreements,
            agreement_rate: if total > 0 { agreements as f64 / total as f64 } else { 1.0 },
            disagreement_details: disagreements,
            coverage,
        }
    }
    
    /// Run tests for ROUGE
    fn run_rouge_tests(
        &self,
        scorer: &RougeNScorer,
        pairs: &[ScoringPair],
        metric_name: &str,
    ) -> AgreementReport {
        let mut agreements = 0;
        let mut disagreements = Vec::new();
        let mut coverage = CoverageReport::default();
        
        for (i, pair) in pairs.iter().enumerate() {
            Self::update_coverage(&mut coverage, pair);
            
            let result = scorer.score_and_verify(pair);
            
            let agree = (result.reference.f1 - result.automaton.f1).abs() < self.tolerance
                && (result.automaton.f1 - result.circuit.f1).abs() < self.tolerance;
            
            if agree {
                agreements += 1;
            } else if disagreements.len() < self.max_disagreements {
                disagreements.push(DisagreementDetail {
                    test_index: i,
                    candidate: pair.candidate.clone(),
                    reference: pair.reference.clone(),
                    metric: metric_name.to_string(),
                    description: format!(
                        "ref={:.6}, aut={:.6}, cir={:.6}",
                        result.reference.f1, result.automaton.f1, result.circuit.f1
                    ),
                });
            }
        }
        
        let total = pairs.len();
        let num_disagreements = total - agreements;
        
        AgreementReport {
            total_tests: total,
            agreements,
            disagreements: num_disagreements,
            agreement_rate: if total > 0 { agreements as f64 / total as f64 } else { 1.0 },
            disagreement_details: disagreements,
            coverage,
        }
    }
    
    /// Run tests for ROUGE-L
    fn run_rouge_tests_l(
        &self,
        scorer: &RougeLScorer,
        pairs: &[ScoringPair],
        metric_name: &str,
    ) -> AgreementReport {
        let mut agreements = 0;
        let mut disagreements = Vec::new();
        let mut coverage = CoverageReport::default();
        
        for (i, pair) in pairs.iter().enumerate() {
            Self::update_coverage(&mut coverage, pair);
            
            let result = scorer.score_and_verify(pair);
            
            let agree = (result.reference.f1 - result.automaton.f1).abs() < self.tolerance
                && (result.automaton.f1 - result.circuit.f1).abs() < self.tolerance;
            
            if agree {
                agreements += 1;
            } else if disagreements.len() < self.max_disagreements {
                disagreements.push(DisagreementDetail {
                    test_index: i,
                    candidate: pair.candidate.clone(),
                    reference: pair.reference.clone(),
                    metric: metric_name.to_string(),
                    description: format!(
                        "ref={:.6}, aut={:.6}, cir={:.6}",
                        result.reference.f1, result.automaton.f1, result.circuit.f1
                    ),
                });
            }
        }
        
        let total = pairs.len();
        let num_disagreements = total - agreements;
        
        AgreementReport {
            total_tests: total,
            agreements,
            disagreements: num_disagreements,
            agreement_rate: if total > 0 { agreements as f64 / total as f64 } else { 1.0 },
            disagreement_details: disagreements,
            coverage,
        }
    }
    
    /// Run all metrics on the same set of pairs
    pub fn test_all_metrics(&self, pairs: &[ScoringPair]) -> HashMap<String, AgreementReport> {
        let mut results = HashMap::new();
        results.insert("exact_match".to_string(), self.test_exact_match(pairs));
        results.insert("token_f1".to_string(), self.test_token_f1(pairs));
        results.insert("bleu".to_string(), self.test_bleu(pairs));
        results.insert("rouge1".to_string(), self.test_rouge1(pairs));
        results.insert("rouge_l".to_string(), self.test_rouge_l(pairs));
        results
    }
}

/// Generate a standard test suite of scoring pairs
pub fn standard_test_suite() -> Vec<ScoringPair> {
    vec![
        // Empty strings
        ScoringPair { candidate: "".to_string(), reference: "".to_string() },
        ScoringPair { candidate: "hello".to_string(), reference: "".to_string() },
        ScoringPair { candidate: "".to_string(), reference: "hello".to_string() },
        
        // Single tokens
        ScoringPair { candidate: "hello".to_string(), reference: "hello".to_string() },
        ScoringPair { candidate: "hello".to_string(), reference: "world".to_string() },
        
        // Multiple tokens - exact match
        ScoringPair { candidate: "the cat sat".to_string(), reference: "the cat sat".to_string() },
        
        // Partial overlap
        ScoringPair { candidate: "the cat sat".to_string(), reference: "the dog sat".to_string() },
        ScoringPair { candidate: "a b c d".to_string(), reference: "c d e f".to_string() },
        
        // Length mismatch
        ScoringPair { candidate: "a b".to_string(), reference: "a b c d".to_string() },
        ScoringPair { candidate: "a b c d".to_string(), reference: "a b".to_string() },
        
        // Repeated tokens
        ScoringPair { candidate: "the the the".to_string(), reference: "the cat dog".to_string() },
        
        // Longer sequences
        ScoringPair {
            candidate: "the quick brown fox jumps over the lazy dog".to_string(),
            reference: "the quick brown fox jumps over the lazy dog".to_string(),
        },
        ScoringPair {
            candidate: "the quick brown fox".to_string(),
            reference: "the slow brown cat".to_string(),
        },
        
        // Numbers and mixed
        ScoringPair { candidate: "42".to_string(), reference: "42".to_string() },
        ScoringPair { candidate: "answer is 42".to_string(), reference: "the answer is 42".to_string() },
    ]
}

/// Generate random scoring pairs for stress testing
pub fn random_test_pairs(count: usize, seed: u64) -> Vec<ScoringPair> {
    let words = vec![
        "the", "cat", "sat", "on", "mat", "dog", "ran", "in",
        "a", "an", "is", "was", "are", "were", "has", "had",
        "big", "small", "red", "blue", "green", "fast", "slow",
        "hello", "world", "test", "data", "code", "rust", "math",
    ];
    
    let mut pairs = Vec::new();
    let mut state = seed;
    
    for _ in 0..count {
        // Simple LCG random number generator
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let cand_len = ((state >> 32) % 8 + 1) as usize;
        
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let ref_len = ((state >> 32) % 8 + 1) as usize;
        
        let mut cand_words = Vec::new();
        for _ in 0..cand_len {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let idx = ((state >> 32) as usize) % words.len();
            cand_words.push(words[idx]);
        }
        
        let mut ref_words = Vec::new();
        for _ in 0..ref_len {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let idx = ((state >> 32) as usize) % words.len();
            ref_words.push(words[idx]);
        }
        
        pairs.push(ScoringPair {
            candidate: cand_words.join(" "),
            reference: ref_words.join(" "),
        });
    }
    
    pairs
}

/// Statistics about a production corpus
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorpusStats {
    pub unique_tokens: usize,
    pub mean_sequence_length: f64,
    pub max_sequence_length: usize,
    pub min_sequence_length: usize,
    pub overlap_distribution: Vec<f64>,
}

/// Returns a BPE-like vocabulary of 600+ unique tokens.
pub fn bpe_vocabulary() -> Vec<&'static str> {
    let mut vocab: Vec<&str> = Vec::with_capacity(700);

    // --- Common English words (120) ---
    vocab.extend_from_slice(&[
        "the", "and", "is", "of", "to", "in", "for", "that", "with", "was",
        "on", "it", "as", "at", "by", "from", "or", "an", "be", "this",
        "are", "but", "not", "have", "had", "has", "his", "her", "they", "we",
        "which", "their", "been", "will", "would", "could", "should", "more", "about", "up",
        "into", "do", "did", "can", "may", "its", "than", "other", "out", "so",
        "what", "if", "no", "when", "who", "all", "also", "how", "each", "she",
        "he", "my", "our", "your", "some", "them", "then", "these", "those", "any",
        "just", "only", "very", "most", "much", "such", "both", "between", "through", "after",
        "before", "where", "over", "under", "again", "here", "there", "because", "during", "while",
        "same", "different", "new", "first", "last", "long", "great", "little", "own", "still",
        "well", "back", "even", "way", "many", "too", "made", "find", "know", "take",
        "people", "come", "make", "like", "time", "used", "look", "number", "part", "get",
    ]);

    // --- Domain-specific NLP/ML tokens (120) ---
    vocab.extend_from_slice(&[
        "transformer", "attention", "embedding", "gradient", "loss", "accuracy", "precision",
        "recall", "encoder", "decoder", "layer", "activation", "softmax", "sigmoid", "relu",
        "dropout", "batch", "epoch", "optimizer", "learning", "rate", "weight", "bias",
        "convolution", "pooling", "recurrent", "lstm", "hidden", "output", "input", "feature",
        "vector", "matrix", "tensor", "dimension", "token", "sequence", "context", "query",
        "key", "value", "head", "multihead", "feedforward", "normalization", "residual",
        "positional", "masking", "padding", "vocabulary", "logit", "probability", "distribution",
        "cross", "entropy", "backpropagation", "checkpoint", "finetune", "pretrain", "inference",
        "latency", "throughput", "parameter", "hyperparameter", "regularization", "overfitting",
        "underfitting", "generalization", "benchmark", "baseline", "evaluation", "metric",
        "perplexity", "coherence", "fluency", "bleu", "rouge", "meteor", "bertscore",
        "alignment", "tokenizer", "subword", "wordpiece", "sentencepiece", "unigram", "bigram",
        "trigram", "ngram", "frequency", "corpus", "dataset", "annotation", "label", "classifier",
        "regression", "clustering", "segmentation", "generation", "translation", "summarization",
        "extraction", "classification", "detection", "recognition", "representation", "latent",
        "manifold", "interpolation", "augmentation", "distillation", "pruning", "quantization",
        "sparsity", "architecture", "pipeline", "framework", "deployment", "scalability",
        "convergence", "divergence", "stability", "robustness",
    ]);

    // --- Subword pieces (110) ---
    vocab.extend_from_slice(&[
        "##ing", "##tion", "##ly", "##ed", "##er", "##ness", "##ment", "##able", "##ive",
        "##ous", "##al", "##ful", "##less", "##ity", "##ence", "##ance", "##ize", "##ist",
        "##ism", "##ory", "##ure", "##ant", "##ent", "##ary", "##ial", "##ual", "##ious",
        "##eous", "##ible", "##ular", "##ular", "##ific", "##ology", "##ograph", "##wards",
        "##wise", "##fold", "##like", "##ward", "##ship", "##dom", "##hood", "##work",
        "un##", "re##", "pre##", "dis##", "mis##", "over##", "under##", "out##", "sub##",
        "super##", "inter##", "trans##", "multi##", "semi##", "anti##", "non##", "co##",
        "de##", "counter##", "auto##", "self##", "cross##", "hyper##", "meta##", "para##",
        "proto##", "pseudo##", "micro##", "macro##", "mini##", "mega##", "ultra##", "infra##",
        "##ation", "##ition", "##ution", "##ction", "##sion", "##gion", "##nion", "##mion",
        "##ling", "##ting", "##ning", "##ring", "##sing", "##ping", "##king", "##ding",
        "##ness", "##ment", "##ence", "##ance", "##ture", "##sure", "##duce", "##rupt",
        "##form", "##port", "##vert", "##plex", "##cede", "##ceed", "##cept", "##tain",
        "##pend", "##fend", "##tend", "##vene", "##sume", "##dure", "##pose", "##voke",
    ]);

    // --- Numeric tokens (110) ---
    for i in 0..100 {
        vocab.push(match i {
            0 => "0", 1 => "1", 2 => "2", 3 => "3", 4 => "4",
            5 => "5", 6 => "6", 7 => "7", 8 => "8", 9 => "9",
            10 => "10", 11 => "11", 12 => "12", 13 => "13", 14 => "14",
            15 => "15", 16 => "16", 17 => "17", 18 => "18", 19 => "19",
            20 => "20", 21 => "21", 22 => "22", 23 => "23", 24 => "24",
            25 => "25", 26 => "26", 27 => "27", 28 => "28", 29 => "29",
            30 => "30", 31 => "31", 32 => "32", 33 => "33", 34 => "34",
            35 => "35", 36 => "36", 37 => "37", 38 => "38", 39 => "39",
            40 => "40", 41 => "41", 42 => "42", 43 => "43", 44 => "44",
            45 => "45", 46 => "46", 47 => "47", 48 => "48", 49 => "49",
            50 => "50", 51 => "51", 52 => "52", 53 => "53", 54 => "54",
            55 => "55", 56 => "56", 57 => "57", 58 => "58", 59 => "59",
            60 => "60", 61 => "61", 62 => "62", 63 => "63", 64 => "64",
            65 => "65", 66 => "66", 67 => "67", 68 => "68", 69 => "69",
            70 => "70", 71 => "71", 72 => "72", 73 => "73", 74 => "74",
            75 => "75", 76 => "76", 77 => "77", 78 => "78", 79 => "79",
            80 => "80", 81 => "81", 82 => "82", 83 => "83", 84 => "84",
            85 => "85", 86 => "86", 87 => "87", 88 => "88", 89 => "89",
            90 => "90", 91 => "91", 92 => "92", 93 => "93", 94 => "94",
            95 => "95", 96 => "96", 97 => "97", 98 => "98", 99 => "99",
            _ => unreachable!(),
        });
    }
    vocab.extend_from_slice(&[
        "1.0", "0.5", "0.1", "0.01", "0.001", "100", "1000", "10000", "256", "512",
    ]);

    // --- Rare / domain / multilingual tokens (ASCII approximations, 110) ---
    vocab.extend_from_slice(&[
        "zusammen", "analyse", "donnees", "ergebnis", "fonction", "methode", "processus",
        "algorithme", "struktur", "beispiel", "resultat", "systeme", "modele", "probleme",
        "recherche", "wissenschaft", "entwicklung", "berechnung", "darstellung", "verfahren",
        "gleichung", "naherung", "beziehung", "eigenschaft", "anwendung", "auswertung",
        "vorbereitung", "durchschnitt", "verteilung", "wahrscheinlichkeit",
        "elemento", "resultado", "problema", "sistema", "proceso", "metodo", "analisis",
        "funcion", "estructura", "ejemplo", "ecuacion", "variable", "constante", "integral",
        "derivada", "converge", "diverge", "continuo", "discreto", "finito",
        "infinito", "conjunto", "subconjunto", "espacio", "dimension", "isomorfo",
        "homomorfismo", "topologia", "geometria", "algebra",
        "matrice", "vettore", "spazio", "insieme", "funzione", "teorema", "dimostrazione",
        "definizione", "proprieta", "relazione", "trasformazione", "applicazione",
        "otobrazhenie", "preobrazovanie", "prostranstvo", "podmnozhestvo",
        "funktsiya", "uravnenie", "reshenie", "dokazatelstvo",
        "henkan", "kukan", "shori", "bunseki", "kekka", "mondai", "houhou",
        "kouzou", "keisan", "hyouka", "jikken", "riron",
        "suanfa", "shuju", "moxing", "fenxi", "jieguo", "wenti", "fangfa",
        "jiegou", "jisuan", "pingjia", "shiyan", "lilun",
        "kwahak", "yeongu", "bunseok", "gyeolgwa", "bangbeop",
        "gujo", "gyesan", "pyeongga", "silheom", "iron",
    ]);

    // --- Punctuation / special tokens (55) ---
    vocab.extend_from_slice(&[
        "[PAD]", "[UNK]", "[SEP]", "[CLS]", "[MASK]", "[BOS]", "[EOS]",
        "<pad>", "<unk>", "<s>", "</s>", "<mask>", "<sep>", "<cls>",
        ",", ".", "!", "?", ";", ":", "'", "\"", "(", ")", "[", "]",
        "{", "}", "-", "--", "...", "/", "\\", "@", "#", "$", "%",
        "&", "*", "+", "=", "<", ">", "|", "^", "~", "_",
        "\t", "0x", "NaN", "null", "true", "false", "None", "undefined",
    ]);

    vocab
}

/// Generate production-scale test pairs with realistic distributions.
pub fn production_test_pairs(count: usize, seed: u64) -> Vec<ScoringPair> {
    let vocab = bpe_vocabulary();
    let vocab_len = vocab.len();
    let mut pairs = Vec::with_capacity(count);
    let mut state = seed;

    // LCG helper closure is inlined below for determinism
    macro_rules! next_rand {
        ($st:expr) => {{
            $st = $st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ($st >> 32) as u64
        }};
    }

    // Pre-allocate edge-case budget
    let empty_count = (count / 50).max(1); // ~2%
    let short_count = (count / 20).max(1); // ~5% very short (1-3 tokens)
    let long_count = (count / 10).max(1);  // ~10% very long (80-100 tokens)
    let exact_count = (count / 10).max(1); // ~10% exact matches
    let overlap_count = (count * 3 / 10).max(1); // ~30% shared tokens
    let normal_start = empty_count + short_count + long_count + exact_count + overlap_count;

    for i in 0..count {
        if i < empty_count {
            // Edge case: empty strings
            let variant = next_rand!(state) % 3;
            let (c, r) = match variant {
                0 => (String::new(), String::new()),
                1 => {
                    let len = (next_rand!(state) % 10 + 1) as usize;
                    let s = (0..len)
                        .map(|_| vocab[(next_rand!(state) as usize) % vocab_len])
                        .collect::<Vec<_>>()
                        .join(" ");
                    (s, String::new())
                }
                _ => {
                    let len = (next_rand!(state) % 10 + 1) as usize;
                    let s = (0..len)
                        .map(|_| vocab[(next_rand!(state) as usize) % vocab_len])
                        .collect::<Vec<_>>()
                        .join(" ");
                    (String::new(), s)
                }
            };
            pairs.push(ScoringPair { candidate: c, reference: r });
        } else if i < empty_count + short_count {
            // Edge case: very short sequences (1-3 tokens)
            let cand_len = (next_rand!(state) % 3 + 1) as usize;
            let ref_len = (next_rand!(state) % 3 + 1) as usize;
            let c = (0..cand_len)
                .map(|_| vocab[(next_rand!(state) as usize) % vocab_len])
                .collect::<Vec<_>>()
                .join(" ");
            let r = (0..ref_len)
                .map(|_| vocab[(next_rand!(state) as usize) % vocab_len])
                .collect::<Vec<_>>()
                .join(" ");
            pairs.push(ScoringPair { candidate: c, reference: r });
        } else if i < empty_count + short_count + long_count {
            // Edge case: very long sequences (80-100 tokens)
            let cand_len = (next_rand!(state) % 21 + 80) as usize;
            let ref_len = (next_rand!(state) % 21 + 80) as usize;
            let c = (0..cand_len)
                .map(|_| vocab[(next_rand!(state) as usize) % vocab_len])
                .collect::<Vec<_>>()
                .join(" ");
            let r = (0..ref_len)
                .map(|_| vocab[(next_rand!(state) as usize) % vocab_len])
                .collect::<Vec<_>>()
                .join(" ");
            pairs.push(ScoringPair { candidate: c, reference: r });
        } else if i < empty_count + short_count + long_count + exact_count {
            // ~10% exact matches
            let len = (next_rand!(state) % 91 + 10) as usize;
            let s = (0..len)
                .map(|_| vocab[(next_rand!(state) as usize) % vocab_len])
                .collect::<Vec<_>>()
                .join(" ");
            pairs.push(ScoringPair { candidate: s.clone(), reference: s });
        } else if i < normal_start {
            // ~30% controlled overlap: build reference, then mutate a portion
            let len = (next_rand!(state) % 51 + 10) as usize;
            let base: Vec<&str> = (0..len)
                .map(|_| vocab[(next_rand!(state) as usize) % vocab_len])
                .collect();
            let mut cand_tokens = base.clone();
            // Replace ~40-60% of tokens to create partial overlap
            let replace_count = (next_rand!(state) % 21 + 40) as usize * len / 100;
            for _ in 0..replace_count {
                let pos = (next_rand!(state) as usize) % cand_tokens.len();
                cand_tokens[pos] = vocab[(next_rand!(state) as usize) % vocab_len];
            }
            // Optionally change length
            let len_change = next_rand!(state) % 4;
            if len_change == 0 && cand_tokens.len() > 3 {
                let trim = (next_rand!(state) as usize) % (cand_tokens.len() / 3).max(1);
                cand_tokens.truncate(cand_tokens.len() - trim);
            } else if len_change == 1 {
                let extra = (next_rand!(state) % 5 + 1) as usize;
                for _ in 0..extra {
                    cand_tokens.push(vocab[(next_rand!(state) as usize) % vocab_len]);
                }
            }
            pairs.push(ScoringPair {
                candidate: cand_tokens.join(" "),
                reference: base.join(" "),
            });
        } else {
            // Normal random pairs (variable length 10-100)
            let cand_len = (next_rand!(state) % 91 + 10) as usize;
            let ref_len = (next_rand!(state) % 91 + 10) as usize;
            let c = (0..cand_len)
                .map(|_| vocab[(next_rand!(state) as usize) % vocab_len])
                .collect::<Vec<_>>()
                .join(" ");
            let r = (0..ref_len)
                .map(|_| vocab[(next_rand!(state) as usize) % vocab_len])
                .collect::<Vec<_>>()
                .join(" ");
            pairs.push(ScoringPair { candidate: c, reference: r });
        }
    }

    pairs
}

/// Compute statistics about a corpus of scoring pairs.
pub fn production_corpus_stats(pairs: &[ScoringPair]) -> CorpusStats {
    let mut unique_tokens: HashSet<&str> = HashSet::new();
    let mut total_len: usize = 0;
    let mut max_len: usize = 0;
    let mut min_len: usize = usize::MAX;
    let mut overlaps: Vec<f64> = Vec::with_capacity(pairs.len());

    for pair in pairs {
        let cand_tokens: Vec<&str> = if pair.candidate.is_empty() {
            Vec::new()
        } else {
            pair.candidate.split_whitespace().collect()
        };
        let ref_tokens: Vec<&str> = if pair.reference.is_empty() {
            Vec::new()
        } else {
            pair.reference.split_whitespace().collect()
        };

        for &t in &cand_tokens { unique_tokens.insert(t); }
        for &t in &ref_tokens { unique_tokens.insert(t); }

        let cand_len = cand_tokens.len();
        let ref_len = ref_tokens.len();
        total_len += cand_len + ref_len;
        max_len = max_len.max(cand_len).max(ref_len);
        min_len = min_len.min(cand_len).min(ref_len);

        let cand_set: HashSet<&str> = cand_tokens.into_iter().collect();
        let ref_set: HashSet<&str> = ref_tokens.into_iter().collect();
        let union_size = cand_set.union(&ref_set).count();
        let inter_size = cand_set.intersection(&ref_set).count();
        let jaccard = if union_size > 0 { inter_size as f64 / union_size as f64 } else { 0.0 };
        overlaps.push(jaccard);
    }

    if pairs.is_empty() {
        min_len = 0;
    }

    let seq_count = if pairs.is_empty() { 1 } else { pairs.len() * 2 };
    let mean_sequence_length = total_len as f64 / seq_count as f64;

    CorpusStats {
        unique_tokens: unique_tokens.len(),
        mean_sequence_length,
        max_sequence_length: max_len,
        min_sequence_length: min_len,
        overlap_distribution: overlaps,
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_differential_exact_match() {
        let tester = DifferentialTester::new();
        let pairs = standard_test_suite();
        let report = tester.test_exact_match(&pairs);
        assert!(report.is_perfect(), "Exact match had {} disagreements: {:?}",
            report.disagreements, report.disagreement_details);
    }
    
    #[test]
    fn test_differential_token_f1() {
        let tester = DifferentialTester::new();
        let pairs = standard_test_suite();
        let report = tester.test_token_f1(&pairs);
        assert!(report.is_perfect(), "Token F1 had {} disagreements: {:?}",
            report.disagreements, report.disagreement_details);
    }
    
    #[test]
    fn test_differential_bleu() {
        let tester = DifferentialTester::new();
        let pairs = standard_test_suite();
        let report = tester.test_bleu(&pairs);
        assert!(report.is_perfect(), "BLEU had {} disagreements: {:?}",
            report.disagreements, report.disagreement_details);
    }
    
    #[test]
    fn test_differential_rouge1() {
        let tester = DifferentialTester::new();
        let pairs = standard_test_suite();
        let report = tester.test_rouge1(&pairs);
        assert!(report.is_perfect(), "ROUGE-1 had {} disagreements: {:?}",
            report.disagreements, report.disagreement_details);
    }
    
    #[test]
    fn test_differential_rouge_l() {
        let tester = DifferentialTester::new();
        let pairs = standard_test_suite();
        let report = tester.test_rouge_l(&pairs);
        assert!(report.is_perfect(), "ROUGE-L had {} disagreements: {:?}",
            report.disagreements, report.disagreement_details);
    }
    
    #[test]
    fn test_differential_all_metrics() {
        let tester = DifferentialTester::new();
        let pairs = standard_test_suite();
        let reports = tester.test_all_metrics(&pairs);
        
        for (metric, report) in &reports {
            assert!(report.is_perfect(), "{} had {} disagreements",
                metric, report.disagreements);
        }
    }
    
    #[test]
    fn test_random_differential() {
        let tester = DifferentialTester::new();
        let pairs = random_test_pairs(50, 12345);
        let reports = tester.test_all_metrics(&pairs);
        
        for (metric, report) in &reports {
            assert!(report.agreement_rate >= 0.99,
                "{} agreement rate too low: {} ({} disagreements)",
                metric, report.agreement_rate, report.disagreements);
        }
    }
    
    #[test]
    fn test_coverage_tracking() {
        let tester = DifferentialTester::new();
        let pairs = standard_test_suite();
        let report = tester.test_exact_match(&pairs);
        
        assert!(report.coverage.empty_inputs);
        assert!(report.coverage.single_token);
        assert!(report.coverage.multi_token);
        assert!(report.coverage.exact_match_true);
        assert!(report.coverage.exact_match_false);
        assert!(report.coverage.length_mismatch);
    }
    
    #[test]
    fn test_standard_test_suite() {
        let suite = standard_test_suite();
        assert!(suite.len() >= 10, "Suite should have at least 10 test cases");
    }
    
    #[test]
    fn test_random_test_pairs() {
        let pairs = random_test_pairs(100, 42);
        assert_eq!(pairs.len(), 100);
        
        // Check all pairs are non-empty (candidate and reference have words)
        for pair in &pairs {
            assert!(!pair.candidate.is_empty());
            assert!(!pair.reference.is_empty());
        }
    }
    
    #[test]
    fn test_agreement_report() {
        let report = AgreementReport {
            total_tests: 100,
            agreements: 100,
            disagreements: 0,
            agreement_rate: 1.0,
            disagreement_details: vec![],
            coverage: CoverageReport::default(),
        };
        assert!(report.is_perfect());
    }
}
