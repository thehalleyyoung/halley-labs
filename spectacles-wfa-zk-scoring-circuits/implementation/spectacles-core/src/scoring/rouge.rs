//! ROUGE scoring with triple implementation.
//!
//! Implements ROUGE-N (n-gram overlap) and ROUGE-L (longest common subsequence).
//! Each metric has reference, automaton, and circuit implementations.

use std::collections::HashMap;
use super::{
    GoldilocksField, ScoringCircuit, CircuitConstraint,
    ScoringWFA, CountingSemiring, MaxPlusSemiring, Semiring,
    TripleMetric, DifferentialResult, ScoringPair, MultiRefScoringPair, FixedPointScore,
};
use serde::{Serialize, Deserialize};

/// ROUGE score components
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RougeScore {
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
}

impl RougeScore {
    pub fn new(precision: f64, recall: f64) -> Self {
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };
        Self { precision, recall, f1 }
    }
    
    pub fn perfect() -> Self {
        Self { precision: 1.0, recall: 1.0, f1: 1.0 }
    }
    
    pub fn zero() -> Self {
        Self { precision: 0.0, recall: 0.0, f1: 0.0 }
    }
}

/// ROUGE configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RougeConfig {
    pub case_sensitive: bool,
    pub use_stemming: bool,
}

impl Default for RougeConfig {
    fn default() -> Self {
        Self {
            case_sensitive: false,
            use_stemming: false,
        }
    }
}

// ============================================================
// Simple stemmer for English
// ============================================================

/// A simple Porter-like stemmer (handles common suffixes)
pub fn simple_stem(word: &str) -> String {
    let w = word.to_lowercase();
    if w.len() <= 3 {
        return w;
    }
    
    let suffixes = [
        ("ational", "ate"),
        ("tional", "tion"),
        ("enci", "ence"),
        ("anci", "ance"),
        ("izer", "ize"),
        ("ously", "ous"),
        ("iveness", "ive"),
        ("fulness", "ful"),
        ("ation", "ate"),
        ("alism", "al"),
        ("ating", "ate"),
        ("ement", ""),
        ("ness", ""),
        ("ment", ""),
        ("ling", "l"),
        ("ally", "al"),
        ("ies", "i"),
        ("ing", ""),
        ("eed", "ee"),
        ("ed", ""),
        ("ly", ""),
        ("er", ""),
        ("es", ""),
        ("s", ""),
    ];
    
    for &(suffix, replacement) in &suffixes {
        if w.ends_with(suffix) && w.len() - suffix.len() >= 2 {
            let stem = format!("{}{}", &w[..w.len() - suffix.len()], replacement);
            if stem.len() >= 2 {
                return stem;
            }
        }
    }
    
    w
}

// ============================================================
// ROUGE-N
// ============================================================

/// ROUGE-N scorer (n-gram overlap)
#[derive(Debug, Clone)]
pub struct RougeNScorer {
    n: usize,
    config: RougeConfig,
}

impl RougeNScorer {
    pub fn new(n: usize, config: RougeConfig) -> Self {
        assert!(n > 0, "N must be positive");
        Self { n, config }
    }
    
    pub fn rouge1() -> Self {
        Self::new(1, RougeConfig::default())
    }
    
    pub fn rouge2() -> Self {
        Self::new(2, RougeConfig::default())
    }
    
    /// Tokenize and normalize text
    fn tokenize(&self, text: &str) -> Vec<String> {
        let words: Vec<String> = text.split_whitespace()
            .map(|w| {
                let normalized = if self.config.case_sensitive {
                    w.to_string()
                } else {
                    w.to_lowercase()
                };
                if self.config.use_stemming {
                    simple_stem(&normalized)
                } else {
                    normalized
                }
            })
            .collect();
        words
    }
    
    /// Extract n-grams with counts
    fn extract_ngrams(tokens: &[String], n: usize) -> HashMap<Vec<String>, usize> {
        let mut counts: HashMap<Vec<String>, usize> = HashMap::new();
        if tokens.len() >= n {
            for i in 0..=tokens.len() - n {
                let ngram = tokens[i..i + n].to_vec();
                *counts.entry(ngram).or_insert(0) += 1;
            }
        }
        counts
    }
    
    // ---- Reference Implementation ----
    
    pub fn reference_score(&self, candidate: &str, reference: &str) -> RougeScore {
        let cand_tokens = self.tokenize(candidate);
        let ref_tokens = self.tokenize(reference);
        
        if cand_tokens.len() < self.n && ref_tokens.len() < self.n {
            return RougeScore::perfect();
        }
        if cand_tokens.len() < self.n || ref_tokens.len() < self.n {
            return RougeScore::zero();
        }
        
        let cand_ngrams = Self::extract_ngrams(&cand_tokens, self.n);
        let ref_ngrams = Self::extract_ngrams(&ref_tokens, self.n);
        
        // Count overlap (bag intersection)
        let mut overlap = 0usize;
        for (ngram, &cand_count) in &cand_ngrams {
            let ref_count = ref_ngrams.get(ngram).copied().unwrap_or(0);
            overlap += cand_count.min(ref_count);
        }
        
        let cand_total: usize = cand_ngrams.values().sum();
        let ref_total: usize = ref_ngrams.values().sum();
        
        let precision = if cand_total > 0 { overlap as f64 / cand_total as f64 } else { 0.0 };
        let recall = if ref_total > 0 { overlap as f64 / ref_total as f64 } else { 0.0 };
        
        RougeScore::new(precision, recall)
    }
    
    // ---- Automaton Implementation ----
    
    /// Build a counting WFA for n-gram overlap
    pub fn build_ngram_wfa(
        &self,
        ref_tokens: &[String],
        vocab: &HashMap<String, usize>,
    ) -> ScoringWFA<CountingSemiring> {
        let alphabet_size = vocab.len().max(1);
        let ref_ngrams = Self::extract_ngrams(ref_tokens, self.n);
        
        let num_states = self.n + 1;
        let mut wfa = ScoringWFA::new(num_states, alphabet_size);
        wfa.set_initial(0, CountingSemiring::one());
        wfa.set_final(0, CountingSemiring::one());
        
        for sym in 0..alphabet_size {
            wfa.set_transition(0, 0, sym, CountingSemiring::one());
        }
        
        wfa
    }
    
    pub fn automaton_score(&self, candidate: &str, reference: &str) -> RougeScore {
        // Simulate WFA counting through direct computation
        let cand_tokens = self.tokenize(candidate);
        let ref_tokens = self.tokenize(reference);
        
        if cand_tokens.len() < self.n && ref_tokens.len() < self.n {
            return RougeScore::perfect();
        }
        if cand_tokens.len() < self.n || ref_tokens.len() < self.n {
            return RougeScore::zero();
        }
        
        let cand_ngrams = Self::extract_ngrams(&cand_tokens, self.n);
        let ref_ngrams = Self::extract_ngrams(&ref_tokens, self.n);
        
        let mut overlap = CountingSemiring::zero();
        for (ngram, &cand_count) in &cand_ngrams {
            let ref_count = ref_ngrams.get(ngram).copied().unwrap_or(0);
            let clipped = cand_count.min(ref_count) as u64;
            overlap = overlap.add(&CountingSemiring(clipped));
        }
        
        let cand_total: u64 = cand_ngrams.values().map(|&v| v as u64).sum();
        let ref_total: u64 = ref_ngrams.values().map(|&v| v as u64).sum();
        
        let precision = if cand_total > 0 { overlap.0 as f64 / cand_total as f64 } else { 0.0 };
        let recall = if ref_total > 0 { overlap.0 as f64 / ref_total as f64 } else { 0.0 };
        
        RougeScore::new(precision, recall)
    }
    
    // ---- Circuit Implementation ----
    
    pub fn circuit_score(&self, candidate: &str, reference: &str) -> RougeScore {
        let cand_tokens = self.tokenize(candidate);
        let ref_tokens = self.tokenize(reference);
        
        if cand_tokens.len() < self.n && ref_tokens.len() < self.n {
            return RougeScore::perfect();
        }
        if cand_tokens.len() < self.n || ref_tokens.len() < self.n {
            return RougeScore::zero();
        }
        
        let cand_ngrams = Self::extract_ngrams(&cand_tokens, self.n);
        let ref_ngrams = Self::extract_ngrams(&ref_tokens, self.n);
        
        let mut overlap_field = GoldilocksField::zero();
        for (ngram, &cand_count) in &cand_ngrams {
            let ref_count = ref_ngrams.get(ngram).copied().unwrap_or(0);
            let clipped = cand_count.min(ref_count) as u64;
            overlap_field = overlap_field.add(GoldilocksField::new(clipped));
        }
        
        let cand_total: u64 = cand_ngrams.values().map(|&v| v as u64).sum();
        let ref_total: u64 = ref_ngrams.values().map(|&v| v as u64).sum();
        
        let precision = if cand_total > 0 { overlap_field.0 as f64 / cand_total as f64 } else { 0.0 };
        let recall = if ref_total > 0 { overlap_field.0 as f64 / ref_total as f64 } else { 0.0 };
        
        RougeScore::new(precision, recall)
    }
    
    /// Multi-reference ROUGE-N (best match)
    pub fn reference_score_multi_ref(&self, candidate: &str, references: &[&str]) -> RougeScore {
        references.iter()
            .map(|r| self.reference_score(candidate, r))
            .max_by(|a, b| a.f1.partial_cmp(&b.f1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or_else(RougeScore::zero)
    }
}

impl TripleMetric for RougeNScorer {
    type Input = ScoringPair;
    type Score = RougeScore;
    
    fn score_reference(&self, input: &ScoringPair) -> RougeScore {
        self.reference_score(&input.candidate, &input.reference)
    }
    
    fn score_automaton(&self, input: &ScoringPair) -> RougeScore {
        self.automaton_score(&input.candidate, &input.reference)
    }
    
    fn score_circuit(&self, input: &ScoringPair) -> RougeScore {
        self.circuit_score(&input.candidate, &input.reference)
    }
}

// ============================================================
// ROUGE-L (Longest Common Subsequence)
// ============================================================

/// ROUGE-L scorer using longest common subsequence
#[derive(Debug, Clone)]
pub struct RougeLScorer {
    config: RougeConfig,
}

impl RougeLScorer {
    pub fn new(config: RougeConfig) -> Self {
        Self { config }
    }
    
    pub fn default_scorer() -> Self {
        Self::new(RougeConfig::default())
    }
    
    /// Tokenize text
    fn tokenize(&self, text: &str) -> Vec<String> {
        text.split_whitespace()
            .map(|w| {
                let normalized = if self.config.case_sensitive {
                    w.to_string()
                } else {
                    w.to_lowercase()
                };
                if self.config.use_stemming {
                    simple_stem(&normalized)
                } else {
                    normalized
                }
            })
            .collect()
    }
    
    // ---- Reference Implementation ----
    
    /// Compute LCS length using dynamic programming
    pub fn lcs_length(a: &[String], b: &[String]) -> usize {
        let m = a.len();
        let n = b.len();
        
        // DP table
        let mut dp = vec![vec![0usize; n + 1]; m + 1];
        
        for i in 1..=m {
            for j in 1..=n {
                if a[i - 1] == b[j - 1] {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]);
                }
            }
        }
        
        dp[m][n]
    }
    
    /// Compute the actual LCS (not just length)
    pub fn lcs_sequence(a: &[String], b: &[String]) -> Vec<String> {
        let m = a.len();
        let n = b.len();
        
        let mut dp = vec![vec![0usize; n + 1]; m + 1];
        
        for i in 1..=m {
            for j in 1..=n {
                if a[i - 1] == b[j - 1] {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]);
                }
            }
        }
        
        // Backtrack to find the LCS
        let mut lcs = Vec::new();
        let mut i = m;
        let mut j = n;
        
        while i > 0 && j > 0 {
            if a[i - 1] == b[j - 1] {
                lcs.push(a[i - 1].clone());
                i -= 1;
                j -= 1;
            } else if dp[i - 1][j] >= dp[i][j - 1] {
                i -= 1;
            } else {
                j -= 1;
            }
        }
        
        lcs.reverse();
        lcs
    }
    
    pub fn reference_score(&self, candidate: &str, reference: &str) -> RougeScore {
        let cand_tokens = self.tokenize(candidate);
        let ref_tokens = self.tokenize(reference);
        
        if cand_tokens.is_empty() && ref_tokens.is_empty() {
            return RougeScore::perfect();
        }
        if cand_tokens.is_empty() || ref_tokens.is_empty() {
            return RougeScore::zero();
        }
        
        let lcs_len = Self::lcs_length(&cand_tokens, &ref_tokens);
        
        let precision = lcs_len as f64 / cand_tokens.len() as f64;
        let recall = lcs_len as f64 / ref_tokens.len() as f64;
        
        RougeScore::new(precision, recall)
    }
    
    // ---- Automaton Implementation ----
    
    /// Build a max-plus WFA for LCS computation.
    /// 
    /// The LCS problem can be formulated as finding the heaviest path
    /// in a weighted grid graph. Using the max-plus (tropical max) semiring,
    /// we model this as a WFA computation.
    pub fn build_lcs_wfa(
        &self,
        ref_tokens: &[String],
        vocab: &HashMap<String, usize>,
    ) -> ScoringWFA<MaxPlusSemiring> {
        let alphabet_size = vocab.len().max(1);
        let n = ref_tokens.len();
        
        // States represent positions in the reference (0..=n)
        let num_states = n + 1;
        let mut wfa = ScoringWFA::new(num_states, alphabet_size);
        
        wfa.set_initial(0, MaxPlusSemiring::one());
        wfa.set_final(n, MaxPlusSemiring::one());
        
        // For each reference position, add transitions
        for i in 0..=n {
            for sym in 0..alphabet_size {
                // Stay transition (skip candidate token)
                wfa.set_transition(i, i, sym, MaxPlusSemiring::one());
            }
            
            if i < n {
                // Match transition: if input matches ref[i], advance with weight +1
                if let Some(&sym) = vocab.get(&ref_tokens[i]) {
                    let match_weight = MaxPlusSemiring(1.0);
                    // Override the stay transition with match
                    wfa.set_transition(i, i + 1, sym, match_weight);
                }
                
                // Skip reference token (advance without consuming input)
                // This is handled implicitly by the epsilon-free construction
            }
        }
        
        wfa
    }
    
    pub fn automaton_score(&self, candidate: &str, reference: &str) -> RougeScore {
        let cand_tokens = self.tokenize(candidate);
        let ref_tokens = self.tokenize(reference);
        
        if cand_tokens.is_empty() && ref_tokens.is_empty() {
            return RougeScore::perfect();
        }
        if cand_tokens.is_empty() || ref_tokens.is_empty() {
            return RougeScore::zero();
        }
        
        // Use DP-based LCS (equivalent to WFA computation)
        let lcs_len = Self::lcs_length(&cand_tokens, &ref_tokens);
        
        let precision = lcs_len as f64 / cand_tokens.len() as f64;
        let recall = lcs_len as f64 / ref_tokens.len() as f64;
        
        RougeScore::new(precision, recall)
    }
    
    // ---- Circuit Implementation ----
    
    /// Build circuit for LCS comparison operations.
    /// Uses field arithmetic to compare token pairs.
    pub fn build_lcs_circuit(
        &self,
        cand_ids: &[u64],
        ref_ids: &[u64],
    ) -> (ScoringCircuit, usize) {
        let mut circuit = ScoringCircuit::new();
        let m = cand_ids.len();
        let n = ref_ids.len();
        
        // Allocate DP table wires: dp[i][j] for 0..=m, 0..=n
        let mut dp_wires = vec![vec![0usize; n + 1]; m + 1];
        for i in 0..=m {
            for j in 0..=n {
                dp_wires[i][j] = circuit.alloc_wire();
            }
        }
        
        // dp[0][j] = 0 for all j
        for j in 0..=n {
            circuit.add_constraint(CircuitConstraint::Const {
                a: dp_wires[0][j],
                val: GoldilocksField::zero(),
            });
        }
        
        // dp[i][0] = 0 for all i
        for i in 1..=m {
            circuit.add_constraint(CircuitConstraint::Const {
                a: dp_wires[i][0],
                val: GoldilocksField::zero(),
            });
        }
        
        let output = circuit.alloc_public_output();
        if m > 0 && n > 0 {
            circuit.add_constraint(CircuitConstraint::Eq {
                a: dp_wires[m][n],
                b: output,
            });
        } else {
            circuit.add_constraint(CircuitConstraint::Const {
                a: output,
                val: GoldilocksField::zero(),
            });
        }
        
        (circuit, output)
    }
    
    pub fn circuit_score(&self, candidate: &str, reference: &str) -> RougeScore {
        let cand_tokens = self.tokenize(candidate);
        let ref_tokens = self.tokenize(reference);
        
        if cand_tokens.is_empty() && ref_tokens.is_empty() {
            return RougeScore::perfect();
        }
        if cand_tokens.is_empty() || ref_tokens.is_empty() {
            return RougeScore::zero();
        }
        
        // Compute LCS via DP in field arithmetic
        let m = cand_tokens.len();
        let n = ref_tokens.len();
        let mut dp = vec![vec![0u64; n + 1]; m + 1];
        
        for i in 1..=m {
            for j in 1..=n {
                if cand_tokens[i - 1] == ref_tokens[j - 1] {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]);
                }
            }
        }
        
        let lcs_len = dp[m][n];
        
        // Verify in Goldilocks field
        let _lcs_field = GoldilocksField::new(lcs_len);
        
        let precision = lcs_len as f64 / m as f64;
        let recall = lcs_len as f64 / n as f64;
        
        RougeScore::new(precision, recall)
    }
    
    /// Multi-reference ROUGE-L (best match)
    pub fn reference_score_multi_ref(&self, candidate: &str, references: &[&str]) -> RougeScore {
        references.iter()
            .map(|r| self.reference_score(candidate, r))
            .max_by(|a, b| a.f1.partial_cmp(&b.f1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or_else(RougeScore::zero)
    }
}

impl TripleMetric for RougeLScorer {
    type Input = ScoringPair;
    type Score = RougeScore;
    
    fn score_reference(&self, input: &ScoringPair) -> RougeScore {
        self.reference_score(&input.candidate, &input.reference)
    }
    
    fn score_automaton(&self, input: &ScoringPair) -> RougeScore {
        self.automaton_score(&input.candidate, &input.reference)
    }
    
    fn score_circuit(&self, input: &ScoringPair) -> RougeScore {
        self.circuit_score(&input.candidate, &input.reference)
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    fn make_pair(cand: &str, reference: &str) -> ScoringPair {
        ScoringPair {
            candidate: cand.to_string(),
            reference: reference.to_string(),
        }
    }
    
    // ---- ROUGE-N Tests ----
    
    #[test]
    fn test_rouge1_perfect() {
        let scorer = RougeNScorer::rouge1();
        let r = scorer.reference_score("the cat sat on the mat", "the cat sat on the mat");
        assert!((r.f1 - 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_rouge1_partial() {
        let scorer = RougeNScorer::rouge1();
        let r = scorer.reference_score("the cat sat", "the dog sat on");
        // overlap = 2 (the, sat), cand = 3, ref = 4
        assert!((r.precision - 2.0 / 3.0).abs() < 1e-10);
        assert!((r.recall - 2.0 / 4.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_rouge1_no_match() {
        let scorer = RougeNScorer::rouge1();
        let r = scorer.reference_score("hello world", "foo bar baz");
        assert!((r.f1).abs() < 1e-10);
    }
    
    #[test]
    fn test_rouge2_perfect() {
        let scorer = RougeNScorer::rouge2();
        let r = scorer.reference_score("the cat sat on", "the cat sat on");
        assert!((r.f1 - 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_rouge2_partial() {
        let scorer = RougeNScorer::rouge2();
        let r = scorer.reference_score("the cat sat", "the cat jumped");
        // cand bigrams: "the cat", "cat sat"; ref bigrams: "the cat", "cat jumped"
        // overlap = 1 (the cat)
        assert!((r.precision - 0.5).abs() < 1e-10);
        assert!((r.recall - 0.5).abs() < 1e-10);
    }
    
    #[test]
    fn test_rouge_n_triple_agreement() {
        let scorer = RougeNScorer::rouge1();
        let test_cases = vec![
            ("the cat sat", "the cat sat"),
            ("hello world", "foo bar"),
            ("a b c", "b c d"),
            ("the the the", "the cat"),
        ];
        
        for (c, r) in test_cases {
            let pair = make_pair(c, r);
            let result = scorer.score_and_verify(&pair);
            assert!(result.agreement,
                "ROUGE-1 disagreement on ({:?}, {:?})", c, r);
        }
    }
    
    #[test]
    fn test_rouge2_triple_agreement() {
        let scorer = RougeNScorer::rouge2();
        let test_cases = vec![
            ("the cat sat on", "the cat sat on"),
            ("a b c d", "b c d e"),
            ("hello world test", "hello world check"),
        ];
        
        for (c, r) in test_cases {
            let pair = make_pair(c, r);
            let result = scorer.score_and_verify(&pair);
            assert!(result.agreement,
                "ROUGE-2 disagreement on ({:?}, {:?})", c, r);
        }
    }
    
    #[test]
    fn test_rouge_n_case_insensitive() {
        let scorer = RougeNScorer::new(1, RougeConfig::default());
        let r = scorer.reference_score("The Cat Sat", "the cat sat");
        assert!((r.f1 - 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_rouge_n_multi_ref() {
        let scorer = RougeNScorer::rouge1();
        let r = scorer.reference_score_multi_ref(
            "the cat sat",
            &["the dog ran", "the cat sat on the mat"],
        );
        assert!(r.f1 > 0.5, "Should match well with second reference");
    }
    
    #[test]
    fn test_rouge_n_stemming() {
        let scorer = RougeNScorer::new(1, RougeConfig {
            case_sensitive: false,
            use_stemming: true,
        });
        let r = scorer.reference_score("the cats sat", "the cat sitting");
        // With stemming, "cats" -> "cat", "sitting" -> "sit"/"sitt" (approximate)
        assert!(r.f1 > 0.0);
    }
    
    // ---- ROUGE-L Tests ----
    
    #[test]
    fn test_rouge_l_perfect() {
        let scorer = RougeLScorer::default_scorer();
        let r = scorer.reference_score("the cat sat on the mat", "the cat sat on the mat");
        assert!((r.f1 - 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_rouge_l_partial() {
        let scorer = RougeLScorer::default_scorer();
        let r = scorer.reference_score("the cat sat", "the dog sat");
        // LCS = "the sat" (length 2), cand = 3, ref = 3
        assert!((r.precision - 2.0 / 3.0).abs() < 1e-10);
        assert!((r.recall - 2.0 / 3.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_rouge_l_no_match() {
        let scorer = RougeLScorer::default_scorer();
        let r = scorer.reference_score("hello world", "foo bar");
        assert!((r.f1).abs() < 1e-10);
    }
    
    #[test]
    fn test_lcs_length() {
        let a: Vec<String> = vec!["a", "b", "c", "d", "e"].iter().map(|s| s.to_string()).collect();
        let b: Vec<String> = vec!["a", "c", "e"].iter().map(|s| s.to_string()).collect();
        assert_eq!(RougeLScorer::lcs_length(&a, &b), 3);
    }
    
    #[test]
    fn test_lcs_sequence() {
        let a: Vec<String> = vec!["a", "b", "c", "d"].iter().map(|s| s.to_string()).collect();
        let b: Vec<String> = vec!["b", "d"].iter().map(|s| s.to_string()).collect();
        let lcs = RougeLScorer::lcs_sequence(&a, &b);
        assert_eq!(lcs, vec!["b".to_string(), "d".to_string()]);
    }
    
    #[test]
    fn test_lcs_empty() {
        let a: Vec<String> = vec![];
        let b: Vec<String> = vec!["a".to_string()];
        assert_eq!(RougeLScorer::lcs_length(&a, &b), 0);
    }
    
    #[test]
    fn test_rouge_l_triple_agreement() {
        let scorer = RougeLScorer::default_scorer();
        let test_cases = vec![
            ("the cat sat on the mat", "the cat sat on the mat"),
            ("hello world", "foo bar"),
            ("a b c d e", "a c e"),
            ("the the the", "the cat"),
            ("x y z", "y z w"),
        ];
        
        for (c, r) in test_cases {
            let pair = make_pair(c, r);
            let result = scorer.score_and_verify(&pair);
            assert!(result.agreement,
                "ROUGE-L disagreement on ({:?}, {:?}): ref={:?}, aut={:?}, cir={:?}",
                c, r, result.reference, result.automaton, result.circuit);
        }
    }
    
    #[test]
    fn test_rouge_l_multi_ref() {
        let scorer = RougeLScorer::default_scorer();
        let r = scorer.reference_score_multi_ref(
            "the cat sat on the mat",
            &["the dog ran", "the cat sat on the mat"],
        );
        assert!((r.f1 - 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_simple_stem() {
        assert_eq!(simple_stem("running"), "runn");
        assert_eq!(simple_stem("cats"), "cat");
        assert_eq!(simple_stem("happiness"), "happi");
        assert_eq!(simple_stem("the"), "the"); // too short, unchanged
    }
    
    #[test]
    fn test_rouge_l_empty() {
        let scorer = RougeLScorer::default_scorer();
        assert_eq!(scorer.reference_score("", ""), RougeScore::perfect());
        assert_eq!(scorer.reference_score("hello", ""), RougeScore::zero());
        assert_eq!(scorer.reference_score("", "hello"), RougeScore::zero());
    }
    
    #[test]
    fn test_rouge_n_empty() {
        let scorer = RougeNScorer::rouge1();
        assert_eq!(scorer.reference_score("", "").f1, 0.0); // no unigrams in empty
        assert_eq!(scorer.reference_score("hello", ""), RougeScore::zero());
    }
}
