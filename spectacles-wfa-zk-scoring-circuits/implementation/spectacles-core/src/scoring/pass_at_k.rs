//! pass@k scoring (Chen et al. 2021) with triple implementation.
//!
//! Estimates the probability that at least one of k code samples passes
//! all test cases. Implements the unbiased estimator.

use std::collections::HashMap;
use super::{
    GoldilocksField, ScoringCircuit, CircuitConstraint,
    ScoringWFA, BooleanSemiring, CountingSemiring, Semiring,
    TripleMetric, DifferentialResult, ScoringPair, FixedPointScore,
    exact_match::ExactMatchScorer,
};
use serde::{Serialize, Deserialize};

/// Configuration for pass@k
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PassAtKConfig {
    pub k: usize,
    pub num_samples: usize,
}

impl Default for PassAtKConfig {
    fn default() -> Self {
        Self { k: 1, num_samples: 10 }
    }
}

/// A single test case for code evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestCase {
    pub input: String,
    pub expected_output: String,
}

/// Result of evaluating a code sample against test cases
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleResult {
    pub sample_id: usize,
    pub outputs: Vec<String>,
    pub passed: Vec<bool>,
    pub all_passed: bool,
}

impl SampleResult {
    pub fn new(sample_id: usize, outputs: Vec<String>, expected: &[String]) -> Self {
        let passed: Vec<bool> = outputs.iter()
            .zip(expected.iter())
            .map(|(out, exp)| out.trim() == exp.trim())
            .collect();
        let all_passed = passed.iter().all(|&p| p);
        Self { sample_id, outputs, passed, all_passed }
    }
}

/// pass@k scorer with triple implementation
#[derive(Debug, Clone)]
pub struct PassAtKScorer {
    config: PassAtKConfig,
}

/// pass@k result
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PassAtKResult {
    pub score: f64,
    pub k: usize,
    pub n: usize,
    pub c: usize,
    pub details: PassAtKDetails,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PassAtKDetails {
    pub num_correct: usize,
    pub num_total: usize,
    pub per_sample: Vec<bool>,
}

impl PassAtKScorer {
    pub fn new(config: PassAtKConfig) -> Self {
        Self { config }
    }
    
    pub fn pass_at_1() -> Self {
        Self::new(PassAtKConfig { k: 1, num_samples: 10 })
    }
    
    pub fn pass_at_k(k: usize) -> Self {
        Self::new(PassAtKConfig { k, num_samples: 10.max(k) })
    }
    
    // ---- Reference Implementation ----
    
    /// Unbiased pass@k estimator: 1 - C(n-c, k) / C(n, k)
    /// where n = total samples, c = correct samples, k = draws
    pub fn unbiased_estimator(n: usize, c: usize, k: usize) -> f64 {
        if n == 0 || k == 0 {
            return 0.0;
        }
        if c >= n {
            return 1.0;
        }
        if k > n {
            return if c > 0 { 1.0 } else { 0.0 };
        }
        if c == 0 {
            return 0.0;
        }
        
        // Compute 1 - C(n-c, k) / C(n, k) using log to avoid overflow
        // C(n-c, k) / C(n, k) = product_{i=0}^{k-1} (n - c - i) / (n - i)
        let mut log_ratio = 0.0f64;
        for i in 0..k {
            let numerator = (n - c) as f64 - i as f64;
            let denominator = n as f64 - i as f64;
            if numerator <= 0.0 {
                // C(n-c, k) = 0, so pass@k = 1
                return 1.0;
            }
            log_ratio += (numerator / denominator).ln();
        }
        
        1.0 - log_ratio.exp()
    }
    
    /// Reference implementation: check samples against test cases
    pub fn reference_score(
        &self,
        sample_outputs: &[Vec<String>],
        expected_outputs: &[String],
    ) -> PassAtKResult {
        let n = sample_outputs.len();
        let results: Vec<SampleResult> = sample_outputs.iter()
            .enumerate()
            .map(|(i, outputs)| SampleResult::new(i, outputs.clone(), expected_outputs))
            .collect();
        
        let c = results.iter().filter(|r| r.all_passed).count();
        let per_sample: Vec<bool> = results.iter().map(|r| r.all_passed).collect();
        
        let score = Self::unbiased_estimator(n, c, self.config.k);
        
        PassAtKResult {
            score,
            k: self.config.k,
            n,
            c,
            details: PassAtKDetails {
                num_correct: c,
                num_total: n,
                per_sample,
            },
        }
    }
    
    // ---- Automaton Implementation ----
    
    /// Build a WFA that checks if a sample output matches the expected output.
    /// Uses exact match WFA for each test case, then counts passing samples.
    pub fn automaton_score(
        &self,
        sample_outputs: &[Vec<String>],
        expected_outputs: &[String],
    ) -> PassAtKResult {
        let n = sample_outputs.len();
        let exact_matcher = ExactMatchScorer::case_sensitive();
        
        let per_sample: Vec<bool> = sample_outputs.iter().map(|outputs| {
            outputs.iter()
                .zip(expected_outputs.iter())
                .all(|(out, exp)| exact_matcher.automaton_score(out.trim(), exp.trim()))
        }).collect();
        
        let c = per_sample.iter().filter(|&&p| p).count();
        let score = Self::unbiased_estimator(n, c, self.config.k);
        
        PassAtKResult {
            score,
            k: self.config.k,
            n,
            c,
            details: PassAtKDetails {
                num_correct: c,
                num_total: n,
                per_sample,
            },
        }
    }
    
    // ---- Circuit Implementation ----
    
    /// Circuit-based pass@k: equality checks in Goldilocks field
    pub fn circuit_score(
        &self,
        sample_outputs: &[Vec<String>],
        expected_outputs: &[String],
    ) -> PassAtKResult {
        let n = sample_outputs.len();
        
        let per_sample: Vec<bool> = sample_outputs.iter().map(|outputs| {
            outputs.iter()
                .zip(expected_outputs.iter())
                .all(|(out, exp)| {
                    let out_trimmed = out.trim();
                    let exp_trimmed = exp.trim();
                    
                    if out_trimmed.len() != exp_trimmed.len() {
                        return false;
                    }
                    
                    // Check each character in Goldilocks field
                    out_trimmed.chars().zip(exp_trimmed.chars()).all(|(a, b)| {
                        GoldilocksField::new(a as u64) == GoldilocksField::new(b as u64)
                    })
                })
        }).collect();
        
        let c = per_sample.iter().filter(|&&p| p).count();
        let score = Self::unbiased_estimator(n, c, self.config.k);
        
        PassAtKResult {
            score,
            k: self.config.k,
            n,
            c,
            details: PassAtKDetails {
                num_correct: c,
                num_total: n,
                per_sample,
            },
        }
    }
    
    /// Build circuit for pass@k evaluation
    pub fn build_passk_circuit(
        &self,
        num_samples: usize,
        num_tests: usize,
        max_output_len: usize,
    ) -> ScoringCircuit {
        let mut circuit = ScoringCircuit::new();
        
        // Input wires for expected outputs
        for _ in 0..num_tests * max_output_len {
            circuit.alloc_public_input();
        }
        
        // Input wires for each sample's outputs
        for _ in 0..num_samples * num_tests * max_output_len {
            circuit.alloc_wire();
        }
        
        // Boolean wires for each sample's pass/fail
        let mut pass_wires = Vec::new();
        for _ in 0..num_samples {
            let w = circuit.alloc_wire();
            circuit.add_constraint(CircuitConstraint::Bool { a: w });
            pass_wires.push(w);
        }
        
        // Count wire (sum of pass_wires)
        let count_wire = circuit.alloc_public_output();
        if !pass_wires.is_empty() {
            // Build addition chain
            let mut acc = pass_wires[0];
            for &pw in &pass_wires[1..] {
                let new_acc = circuit.alloc_wire();
                circuit.add_constraint(CircuitConstraint::Add { a: acc, b: pw, c: new_acc });
                acc = new_acc;
            }
            circuit.add_constraint(CircuitConstraint::Eq { a: acc, b: count_wire });
        }
        
        circuit
    }
}

/// Compute binomial coefficient C(n, k)
pub fn binomial(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    if k == 0 || k == n {
        return 1.0;
    }
    
    let k = k.min(n - k);
    let mut result = 1.0f64;
    for i in 0..k {
        result *= (n - i) as f64;
        result /= (i + 1) as f64;
    }
    result
}

/// Compute pass@k for multiple problems (average)
pub fn corpus_pass_at_k(
    problems: &[(Vec<Vec<String>>, Vec<String>)], // (samples, expected) per problem
    k: usize,
) -> f64 {
    if problems.is_empty() {
        return 0.0;
    }
    
    let total: f64 = problems.iter().map(|(samples, expected)| {
        let scorer = PassAtKScorer::pass_at_k(k);
        scorer.reference_score(samples, expected).score
    }).sum();
    
    total / problems.len() as f64
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_pass_at_1_all_correct() {
        let scorer = PassAtKScorer::pass_at_1();
        let samples = vec![
            vec!["42".to_string()],
            vec!["42".to_string()],
            vec!["42".to_string()],
        ];
        let expected = vec!["42".to_string()];
        
        let result = scorer.reference_score(&samples, &expected);
        assert!((result.score - 1.0).abs() < 1e-10);
        assert_eq!(result.c, 3);
    }
    
    #[test]
    fn test_pass_at_1_none_correct() {
        let scorer = PassAtKScorer::pass_at_1();
        let samples = vec![
            vec!["41".to_string()],
            vec!["43".to_string()],
        ];
        let expected = vec!["42".to_string()];
        
        let result = scorer.reference_score(&samples, &expected);
        assert!((result.score).abs() < 1e-10);
    }
    
    #[test]
    fn test_pass_at_1_some_correct() {
        let scorer = PassAtKScorer::pass_at_1();
        let samples = vec![
            vec!["42".to_string()],
            vec!["43".to_string()],
            vec!["42".to_string()],
            vec!["44".to_string()],
        ];
        let expected = vec!["42".to_string()];
        
        let result = scorer.reference_score(&samples, &expected);
        // n=4, c=2, k=1: 1 - C(2,1)/C(4,1) = 1 - 2/4 = 0.5
        assert!((result.score - 0.5).abs() < 1e-10);
    }
    
    #[test]
    fn test_pass_at_k_larger() {
        let scorer = PassAtKScorer::pass_at_k(2);
        let samples = vec![
            vec!["42".to_string()],
            vec!["43".to_string()],
            vec!["44".to_string()],
            vec!["42".to_string()],
        ];
        let expected = vec!["42".to_string()];
        
        let result = scorer.reference_score(&samples, &expected);
        // n=4, c=2, k=2: 1 - C(2,2)/C(4,2) = 1 - 1/6
        assert!((result.score - (1.0 - 1.0 / 6.0)).abs() < 1e-10);
    }
    
    #[test]
    fn test_unbiased_estimator_edge_cases() {
        assert!((PassAtKScorer::unbiased_estimator(0, 0, 1)).abs() < 1e-10);
        assert!((PassAtKScorer::unbiased_estimator(10, 10, 1) - 1.0).abs() < 1e-10);
        assert!((PassAtKScorer::unbiased_estimator(10, 0, 1)).abs() < 1e-10);
        assert!((PassAtKScorer::unbiased_estimator(5, 3, 10) - 1.0).abs() < 1e-10); // k > n
    }
    
    #[test]
    fn test_multiple_test_cases() {
        let scorer = PassAtKScorer::pass_at_1();
        let samples = vec![
            vec!["3".to_string(), "5".to_string()],  // correct
            vec!["3".to_string(), "6".to_string()],  // wrong on test 2
            vec!["4".to_string(), "5".to_string()],  // wrong on test 1
        ];
        let expected = vec!["3".to_string(), "5".to_string()];
        
        let result = scorer.reference_score(&samples, &expected);
        assert_eq!(result.c, 1);
        assert_eq!(result.details.per_sample, vec![true, false, false]);
    }
    
    #[test]
    fn test_triple_agreement() {
        let scorer = PassAtKScorer::pass_at_1();
        let samples = vec![
            vec!["hello".to_string()],
            vec!["world".to_string()],
            vec!["hello".to_string()],
        ];
        let expected = vec!["hello".to_string()];
        
        let ref_result = scorer.reference_score(&samples, &expected);
        let aut_result = scorer.automaton_score(&samples, &expected);
        let cir_result = scorer.circuit_score(&samples, &expected);
        
        assert_eq!(ref_result.score, aut_result.score);
        assert_eq!(aut_result.score, cir_result.score);
        assert_eq!(ref_result.c, aut_result.c);
        assert_eq!(aut_result.c, cir_result.c);
    }
    
    #[test]
    fn test_binomial() {
        assert!((binomial(5, 2) - 10.0).abs() < 1e-10);
        assert!((binomial(10, 3) - 120.0).abs() < 1e-10);
        assert!((binomial(0, 0) - 1.0).abs() < 1e-10);
        assert!((binomial(5, 0) - 1.0).abs() < 1e-10);
        assert!((binomial(3, 5)).abs() < 1e-10); // k > n
    }
    
    #[test]
    fn test_corpus_pass_at_k() {
        let problems = vec![
            (
                vec![vec!["42".to_string()], vec!["42".to_string()]],
                vec!["42".to_string()],
            ),
            (
                vec![vec!["7".to_string()], vec!["8".to_string()]],
                vec!["7".to_string()],
            ),
        ];
        
        let score = corpus_pass_at_k(&problems, 1);
        // Problem 1: pass@1 = 1.0 (both correct)
        // Problem 2: pass@1 = 0.5 (1 of 2 correct)
        assert!((score - 0.75).abs() < 1e-10);
    }
    
    #[test]
    fn test_circuit_construction() {
        let scorer = PassAtKScorer::pass_at_1();
        let circuit = scorer.build_passk_circuit(5, 3, 10);
        assert!(circuit.num_wires > 0);
        assert!(!circuit.public_outputs.is_empty());
    }
    
    #[test]
    fn test_sample_result() {
        let result = SampleResult::new(
            0,
            vec!["42".to_string(), "hello".to_string()],
            &["42".to_string(), "hello".to_string()],
        );
        assert!(result.all_passed);
        assert_eq!(result.passed, vec![true, true]);
        
        let result2 = SampleResult::new(
            1,
            vec!["42".to_string(), "world".to_string()],
            &["42".to_string(), "hello".to_string()],
        );
        assert!(!result2.all_passed);
        assert_eq!(result2.passed, vec![true, false]);
    }
    
    #[test]
    fn test_whitespace_handling() {
        let scorer = PassAtKScorer::pass_at_1();
        let samples = vec![
            vec!["  42  ".to_string()],
        ];
        let expected = vec!["42".to_string()];
        
        let result = scorer.reference_score(&samples, &expected);
        assert_eq!(result.c, 1); // Should match after trimming
    }
}
