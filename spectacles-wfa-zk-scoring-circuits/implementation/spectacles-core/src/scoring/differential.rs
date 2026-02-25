//! Differential testing framework for triple implementations.
//!
//! Tests that reference, automaton, and circuit implementations agree
//! on random and structured inputs.

use std::collections::HashMap;
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
