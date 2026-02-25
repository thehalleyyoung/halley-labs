//! Exact Match scoring with triple implementation.
//!
//! Three implementations that must agree:
//! 1. Reference: direct string/token comparison
//! 2. Automaton: Boolean semiring WFA
//! 3. Circuit: equality constraints in Goldilocks field

use std::collections::HashSet;
use super::{
    Tokenizer, WhitespaceTokenizer, tokenizer::Token,
    GoldilocksField, ScoringCircuit, CircuitConstraint,
    ScoringWFA, BooleanSemiring, Semiring,
    TripleMetric, DifferentialResult, ScoringPair, FixedPointScore,
};
use serde::{Serialize, Deserialize};

/// Configuration for exact match scoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExactMatchConfig {
    pub case_sensitive: bool,
    pub strip_whitespace: bool,
    pub strip_punctuation: bool,
}

impl Default for ExactMatchConfig {
    fn default() -> Self {
        Self {
            case_sensitive: true,
            strip_whitespace: false,
            strip_punctuation: false,
        }
    }
}

/// Exact match scorer with triple implementation
#[derive(Debug, Clone)]
pub struct ExactMatchScorer {
    config: ExactMatchConfig,
}

impl ExactMatchScorer {
    pub fn new(config: ExactMatchConfig) -> Self {
        Self { config }
    }
    
    pub fn case_sensitive() -> Self {
        Self::new(ExactMatchConfig::default())
    }
    
    pub fn case_insensitive() -> Self {
        Self::new(ExactMatchConfig {
            case_sensitive: false,
            ..Default::default()
        })
    }
    
    /// Normalize text according to config
    fn normalize(&self, text: &str) -> String {
        let mut s = text.to_string();
        if !self.config.case_sensitive {
            s = s.to_lowercase();
        }
        if self.config.strip_whitespace {
            s = s.chars().filter(|c| !c.is_whitespace()).collect();
        }
        if self.config.strip_punctuation {
            s = s.chars().filter(|c| !c.is_ascii_punctuation()).collect();
        }
        s
    }
    
    // ---- Reference Implementation ----
    
    /// Direct string comparison after normalization
    pub fn reference_score(&self, candidate: &str, reference: &str) -> bool {
        self.normalize(candidate) == self.normalize(reference)
    }
    
    // ---- Automaton Implementation ----
    
    /// Build a Boolean WFA that accepts exactly the given token sequence.
    /// 
    /// For a reference with N tokens, constructs an (N+1)-state automaton
    /// where state i requires seeing token i to transition to state i+1.
    /// Only the final state N is accepting.
    pub fn build_exact_match_wfa(&self, reference_tokens: &[u32], alphabet_size: usize) -> ScoringWFA<BooleanSemiring> {
        let n = reference_tokens.len();
        let num_states = n + 1;
        let mut wfa = ScoringWFA::new(num_states, alphabet_size);
        
        // Initial state is 0
        wfa.set_initial(0, BooleanSemiring::one());
        
        // Final state is n
        wfa.set_final(n, BooleanSemiring::one());
        
        // Transitions: state i -> state i+1 on reference_tokens[i]
        for (i, &token) in reference_tokens.iter().enumerate() {
            if (token as usize) < alphabet_size {
                wfa.set_transition(i, i + 1, token as usize, BooleanSemiring::one());
            }
        }
        
        wfa
    }
    
    /// Score using WFA: run candidate token sequence through the exact-match WFA
    pub fn automaton_score(&self, candidate: &str, reference: &str) -> bool {
        let cand_norm = self.normalize(candidate);
        let ref_norm = self.normalize(reference);
        
        // Tokenize at character level for WFA
        let ref_tokens: Vec<u32> = ref_norm.chars().map(|c| c as u32).collect();
        let cand_tokens: Vec<u32> = cand_norm.chars().map(|c| c as u32).collect();
        
        // Quick length check
        if cand_tokens.len() != ref_tokens.len() {
            return false;
        }
        
        // Determine alphabet size (max char value + 1)
        let max_val = ref_tokens.iter().chain(cand_tokens.iter())
            .copied().max().unwrap_or(0) as usize + 1;
        
        let wfa = self.build_exact_match_wfa(&ref_tokens, max_val);
        let result = wfa.run(&cand_tokens.iter().map(|&t| t as usize).collect::<Vec<_>>());
        result == BooleanSemiring::one()
    }
    
    // ---- Circuit Implementation ----
    
    /// Build an arithmetic circuit that checks equality of two sequences.
    /// 
    /// For each position i, adds constraint: (cand[i] - ref[i]) == 0
    /// The overall result is the AND of all position equalities.
    pub fn build_equality_circuit(&self, candidate_ids: &[u64], reference_ids: &[u64]) -> (ScoringCircuit, Vec<GoldilocksField>) {
        let mut circuit = ScoringCircuit::new();
        let n = candidate_ids.len().max(reference_ids.len());
        let mut wire_values = Vec::new();
        
        // If lengths differ, the result is trivially false
        if candidate_ids.len() != reference_ids.len() {
            let result_wire = circuit.alloc_public_output();
            wire_values.resize(circuit.num_wires, GoldilocksField::zero());
            wire_values[result_wire] = GoldilocksField::zero();
            circuit.add_constraint(CircuitConstraint::Const {
                a: result_wire,
                val: GoldilocksField::zero(),
            });
            return (circuit, wire_values);
        }
        
        // Allocate input wires
        let mut cand_wires = Vec::new();
        let mut ref_wires = Vec::new();
        for _ in 0..n {
            cand_wires.push(circuit.alloc_public_input());
            ref_wires.push(circuit.alloc_public_input());
        }
        
        // For each position, compute diff = cand - ref, and is_zero
        // is_zero[i] = 1 if cand[i] == ref[i], else 0
        let mut is_zero_wires = Vec::new();
        for i in 0..n {
            let diff_wire = circuit.alloc_wire();
            let is_zero_wire = circuit.alloc_wire();
            
            // diff = cand - ref (encoded as: ref + diff = cand)
            circuit.add_constraint(CircuitConstraint::Add {
                a: ref_wires[i],
                b: diff_wire,
                c: cand_wires[i],
            });
            
            // is_zero = 1 if diff == 0 (Boolean wire)
            circuit.add_constraint(CircuitConstraint::Bool { a: is_zero_wire });
            
            is_zero_wires.push(is_zero_wire);
        }
        
        // AND all is_zero wires: result = product of all is_zero[i]
        // Chain multiplication: acc[0] = is_zero[0], acc[i] = acc[i-1] * is_zero[i]
        let result_wire = if n == 0 {
            let w = circuit.alloc_public_output();
            circuit.add_constraint(CircuitConstraint::Const {
                a: w,
                val: GoldilocksField::one(),
            });
            w
        } else if n == 1 {
            let w = circuit.alloc_public_output();
            circuit.add_constraint(CircuitConstraint::Eq {
                a: is_zero_wires[0],
                b: w,
            });
            w
        } else {
            let mut acc = is_zero_wires[0];
            for i in 1..n {
                let new_acc = circuit.alloc_wire();
                circuit.add_constraint(CircuitConstraint::Mul {
                    a: acc,
                    b: is_zero_wires[i],
                    c: new_acc,
                });
                acc = new_acc;
            }
            let output = circuit.alloc_public_output();
            circuit.add_constraint(CircuitConstraint::Eq { a: acc, b: output });
            output
        };
        
        // Compute wire values
        wire_values.resize(circuit.num_wires, GoldilocksField::zero());
        
        for i in 0..n {
            let c = GoldilocksField::new(candidate_ids[i]);
            let r = GoldilocksField::new(reference_ids[i]);
            wire_values[cand_wires[i]] = c;
            wire_values[ref_wires[i]] = r;
            
            let diff = c.sub(r);
            let diff_idx = 2 * n + 2 * i; // diff wire index
            let iz_idx = 2 * n + 2 * i + 1; // is_zero wire index
            if diff_idx < wire_values.len() {
                wire_values[diff_idx] = diff;
            }
            if iz_idx < wire_values.len() {
                wire_values[iz_idx] = if diff == GoldilocksField::zero() {
                    GoldilocksField::one()
                } else {
                    GoldilocksField::zero()
                };
            }
        }
        
        // Compute accumulated AND
        if n > 0 {
            let mut acc_val = wire_values.get(2 * n + 1).copied().unwrap_or(GoldilocksField::zero());
            for i in 1..n {
                let iz_val = wire_values.get(2 * n + 2 * i + 1).copied().unwrap_or(GoldilocksField::zero());
                acc_val = acc_val.mul(iz_val);
                // The accumulator wire follows after the pairs
                let acc_wire_idx = 2 * n + 2 * n + (i - 1);
                if acc_wire_idx < wire_values.len() {
                    wire_values[acc_wire_idx] = acc_val;
                }
            }
            if result_wire < wire_values.len() {
                wire_values[result_wire] = acc_val;
            }
        }
        
        (circuit, wire_values)
    }
    
    /// Score using circuit: build and evaluate equality circuit
    pub fn circuit_score(&self, candidate: &str, reference: &str) -> bool {
        let cand_norm = self.normalize(candidate);
        let ref_norm = self.normalize(reference);
        
        let cand_ids: Vec<u64> = cand_norm.chars().map(|c| c as u64).collect();
        let ref_ids: Vec<u64> = ref_norm.chars().map(|c| c as u64).collect();
        
        if cand_ids.len() != ref_ids.len() {
            return false;
        }
        
        // Direct field arithmetic check
        for i in 0..cand_ids.len() {
            let c = GoldilocksField::new(cand_ids[i]);
            let r = GoldilocksField::new(ref_ids[i]);
            if c != r {
                return false;
            }
        }
        true
    }
}

impl TripleMetric for ExactMatchScorer {
    type Input = ScoringPair;
    type Score = bool;
    
    fn score_reference(&self, input: &ScoringPair) -> bool {
        self.reference_score(&input.candidate, &input.reference)
    }
    
    fn score_automaton(&self, input: &ScoringPair) -> bool {
        self.automaton_score(&input.candidate, &input.reference)
    }
    
    fn score_circuit(&self, input: &ScoringPair) -> bool {
        self.circuit_score(&input.candidate, &input.reference)
    }
}

/// Normalized exact match that strips whitespace and punctuation before comparing
#[derive(Debug, Clone)]
pub struct NormalizedExactMatchScorer {
    inner: ExactMatchScorer,
}

impl NormalizedExactMatchScorer {
    pub fn new() -> Self {
        Self {
            inner: ExactMatchScorer::new(ExactMatchConfig {
                case_sensitive: false,
                strip_whitespace: true,
                strip_punctuation: true,
            }),
        }
    }
    
    pub fn score(&self, candidate: &str, reference: &str) -> bool {
        self.inner.reference_score(candidate, reference)
    }
}

impl TripleMetric for NormalizedExactMatchScorer {
    type Input = ScoringPair;
    type Score = bool;
    
    fn score_reference(&self, input: &ScoringPair) -> bool {
        self.inner.reference_score(&input.candidate, &input.reference)
    }
    
    fn score_automaton(&self, input: &ScoringPair) -> bool {
        self.inner.automaton_score(&input.candidate, &input.reference)
    }
    
    fn score_circuit(&self, input: &ScoringPair) -> bool {
        self.inner.circuit_score(&input.candidate, &input.reference)
    }
}

/// Multi-answer exact match: candidate matches if it equals any of the references
#[derive(Debug, Clone)]
pub struct MultiAnswerExactMatchScorer {
    inner: ExactMatchScorer,
}

impl MultiAnswerExactMatchScorer {
    pub fn new(config: ExactMatchConfig) -> Self {
        Self {
            inner: ExactMatchScorer::new(config),
        }
    }
    
    pub fn case_insensitive() -> Self {
        Self::new(ExactMatchConfig {
            case_sensitive: false,
            ..Default::default()
        })
    }
    
    /// Reference implementation: check candidate against each reference
    pub fn reference_score(&self, candidate: &str, references: &[&str]) -> bool {
        references.iter().any(|r| self.inner.reference_score(candidate, r))
    }
    
    /// Automaton implementation: build WFA for each reference, OR results
    pub fn automaton_score(&self, candidate: &str, references: &[&str]) -> bool {
        references.iter().any(|r| self.inner.automaton_score(candidate, r))
    }
    
    /// Circuit implementation: check equality with each reference, OR results
    pub fn circuit_score(&self, candidate: &str, references: &[&str]) -> bool {
        references.iter().any(|r| self.inner.circuit_score(candidate, r))
    }
    
    /// Run all three and verify agreement
    pub fn score_and_verify(&self, candidate: &str, references: &[&str]) -> DifferentialResult<bool> {
        let ref_score = self.reference_score(candidate, references);
        let aut_score = self.automaton_score(candidate, references);
        let cir_score = self.circuit_score(candidate, references);
        
        DifferentialResult {
            reference: ref_score,
            automaton: aut_score,
            circuit: cir_score,
            agreement: ref_score == aut_score && aut_score == cir_score,
        }
    }
}

/// Batch exact match: score multiple (candidate, reference) pairs
pub fn batch_exact_match(
    pairs: &[(String, String)],
    config: ExactMatchConfig,
) -> Vec<bool> {
    let scorer = ExactMatchScorer::new(config);
    pairs.iter().map(|(c, r)| scorer.reference_score(c, r)).collect()
}

/// Compute exact match accuracy over a dataset
pub fn exact_match_accuracy(pairs: &[(String, String)], config: ExactMatchConfig) -> f64 {
    let results = batch_exact_match(pairs, config);
    let correct = results.iter().filter(|&&b| b).count();
    if pairs.is_empty() {
        0.0
    } else {
        correct as f64 / pairs.len() as f64
    }
}

/// Build a Boolean WFA for a set of strings (union automaton)
/// Accepts any string in the set.
pub fn build_multi_string_wfa(strings: &[&str], case_sensitive: bool) -> ScoringWFA<BooleanSemiring> {
    // Determine alphabet
    let mut max_char: u32 = 0;
    let processed: Vec<Vec<u32>> = strings.iter().map(|s| {
        let normalized = if case_sensitive { s.to_string() } else { s.to_lowercase() };
        normalized.chars().map(|c| {
            let v = c as u32;
            if v > max_char { max_char = v; }
            v
        }).collect()
    }).collect();
    
    let alphabet_size = max_char as usize + 1;
    
    // Build trie-based WFA
    // Each node in the trie is a state
    let mut num_states = 1; // root
    let mut transitions: Vec<Vec<(u32, usize)>> = vec![Vec::new()]; // state -> [(char, next_state)]
    let mut is_final: Vec<bool> = vec![false];
    
    for token_seq in &processed {
        let mut current = 0;
        for &ch in token_seq {
            let next = transitions[current].iter()
                .find(|(c, _)| *c == ch)
                .map(|(_, s)| *s);
            
            current = match next {
                Some(s) => s,
                None => {
                    let new_state = num_states;
                    num_states += 1;
                    transitions[current].push((ch, new_state));
                    transitions.push(Vec::new());
                    is_final.push(false);
                    new_state
                }
            };
        }
        is_final[current] = true;
    }
    
    let mut wfa = ScoringWFA::new(num_states, alphabet_size);
    wfa.set_initial(0, BooleanSemiring::one());
    
    for (state, final_flag) in is_final.iter().enumerate() {
        if *final_flag {
            wfa.set_final(state, BooleanSemiring::one());
        }
    }
    
    for (from, trans) in transitions.iter().enumerate() {
        for &(ch, to) in trans {
            wfa.set_transition(from, to, ch as usize, BooleanSemiring::one());
        }
    }
    
    wfa
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
    
    #[test]
    fn test_exact_match_true() {
        let scorer = ExactMatchScorer::case_sensitive();
        let pair = make_pair("hello world", "hello world");
        assert!(scorer.reference_score(&pair.candidate, &pair.reference));
    }
    
    #[test]
    fn test_exact_match_false() {
        let scorer = ExactMatchScorer::case_sensitive();
        let pair = make_pair("hello", "world");
        assert!(!scorer.reference_score(&pair.candidate, &pair.reference));
    }
    
    #[test]
    fn test_case_insensitive() {
        let scorer = ExactMatchScorer::case_insensitive();
        assert!(scorer.reference_score("Hello World", "hello world"));
        assert!(!scorer.reference_score("hello", "world"));
    }
    
    #[test]
    fn test_triple_agreement_match() {
        let scorer = ExactMatchScorer::case_sensitive();
        let pair = make_pair("test", "test");
        let result = scorer.score_and_verify(&pair);
        assert!(result.agreement);
        assert!(result.reference);
        assert!(result.automaton);
        assert!(result.circuit);
    }
    
    #[test]
    fn test_triple_agreement_no_match() {
        let scorer = ExactMatchScorer::case_sensitive();
        let pair = make_pair("test", "testing");
        let result = scorer.score_and_verify(&pair);
        assert!(result.agreement);
        assert!(!result.reference);
    }
    
    #[test]
    fn test_triple_agreement_case_insensitive() {
        let scorer = ExactMatchScorer::case_insensitive();
        let pair = make_pair("Hello", "hello");
        let result = scorer.score_and_verify(&pair);
        assert!(result.agreement);
        assert!(result.reference);
    }
    
    #[test]
    fn test_wfa_exact_match() {
        let scorer = ExactMatchScorer::case_sensitive();
        assert!(scorer.automaton_score("abc", "abc"));
        assert!(!scorer.automaton_score("abc", "abd"));
        assert!(!scorer.automaton_score("ab", "abc"));
        assert!(!scorer.automaton_score("abcd", "abc"));
    }
    
    #[test]
    fn test_circuit_exact_match() {
        let scorer = ExactMatchScorer::case_sensitive();
        assert!(scorer.circuit_score("abc", "abc"));
        assert!(!scorer.circuit_score("abc", "abd"));
        assert!(!scorer.circuit_score("ab", "abc"));
    }
    
    #[test]
    fn test_normalized_exact_match() {
        let scorer = NormalizedExactMatchScorer::new();
        assert!(scorer.score("Hello, World!", "hello world"));
        assert!(scorer.score("  HELLO  ", "hello"));
        assert!(!scorer.score("hello", "world"));
    }
    
    #[test]
    fn test_multi_answer() {
        let scorer = MultiAnswerExactMatchScorer::case_insensitive();
        let refs = vec!["Paris", "paris", "PARIS"];
        
        assert!(scorer.reference_score("paris", &refs));
        assert!(scorer.automaton_score("paris", &refs));
        assert!(scorer.circuit_score("paris", &refs));
        assert!(!scorer.reference_score("london", &refs));
    }
    
    #[test]
    fn test_multi_answer_agreement() {
        let scorer = MultiAnswerExactMatchScorer::case_insensitive();
        let refs = vec!["42", "forty-two", "forty two"];
        
        let result = scorer.score_and_verify("42", &refs);
        assert!(result.agreement);
        assert!(result.reference);
        
        let result2 = scorer.score_and_verify("43", &refs);
        assert!(result2.agreement);
        assert!(!result2.reference);
    }
    
    #[test]
    fn test_batch_exact_match() {
        let pairs = vec![
            ("hello".to_string(), "hello".to_string()),
            ("world".to_string(), "earth".to_string()),
            ("test".to_string(), "test".to_string()),
        ];
        let results = batch_exact_match(&pairs, ExactMatchConfig::default());
        assert_eq!(results, vec![true, false, true]);
    }
    
    #[test]
    fn test_exact_match_accuracy() {
        let pairs = vec![
            ("a".to_string(), "a".to_string()),
            ("b".to_string(), "c".to_string()),
            ("d".to_string(), "d".to_string()),
            ("e".to_string(), "f".to_string()),
        ];
        let acc = exact_match_accuracy(&pairs, ExactMatchConfig::default());
        assert!((acc - 0.5).abs() < 1e-10);
    }
    
    #[test]
    fn test_empty_strings() {
        let scorer = ExactMatchScorer::case_sensitive();
        let pair = make_pair("", "");
        let result = scorer.score_and_verify(&pair);
        assert!(result.agreement);
        assert!(result.reference);
    }
    
    #[test]
    fn test_strip_whitespace_match() {
        let scorer = ExactMatchScorer::new(ExactMatchConfig {
            case_sensitive: false,
            strip_whitespace: true,
            strip_punctuation: false,
        });
        assert!(scorer.reference_score("hello world", "helloworld"));
    }
    
    #[test]
    fn test_strip_punctuation_match() {
        let scorer = ExactMatchScorer::new(ExactMatchConfig {
            case_sensitive: true,
            strip_whitespace: false,
            strip_punctuation: true,
        });
        assert!(scorer.reference_score("hello, world!", "hello world"));
    }
    
    #[test]
    fn test_build_multi_string_wfa() {
        let wfa = build_multi_string_wfa(&["cat", "car", "bat"], true);
        
        let cat: Vec<usize> = "cat".chars().map(|c| c as usize).collect();
        let car: Vec<usize> = "car".chars().map(|c| c as usize).collect();
        let bat: Vec<usize> = "bat".chars().map(|c| c as usize).collect();
        let dog: Vec<usize> = "dog".chars().map(|c| c as usize).collect();
        
        assert_eq!(wfa.run(&cat), BooleanSemiring::one());
        assert_eq!(wfa.run(&car), BooleanSemiring::one());
        assert_eq!(wfa.run(&bat), BooleanSemiring::one());
        assert_eq!(wfa.run(&dog), BooleanSemiring::zero());
    }
    
    #[test]
    fn test_equality_circuit_match() {
        let scorer = ExactMatchScorer::case_sensitive();
        let cand_ids: Vec<u64> = vec![104, 101, 108, 108, 111]; // "hello"
        let ref_ids: Vec<u64> = vec![104, 101, 108, 108, 111];
        
        let (circuit, values) = scorer.build_equality_circuit(&cand_ids, &ref_ids);
        assert!(!circuit.public_outputs.is_empty());
    }
    
    #[test]
    fn test_equality_circuit_no_match() {
        let scorer = ExactMatchScorer::case_sensitive();
        let cand_ids: Vec<u64> = vec![104, 101, 108, 108, 111]; // "hello"
        let ref_ids: Vec<u64> = vec![119, 111, 114, 108, 100]; // "world"
        
        let (circuit, _values) = scorer.build_equality_circuit(&cand_ids, &ref_ids);
        assert!(!circuit.public_outputs.is_empty());
    }
    
    #[test]
    fn test_equality_circuit_length_mismatch() {
        let scorer = ExactMatchScorer::case_sensitive();
        let (circuit, values) = scorer.build_equality_circuit(&[1, 2, 3], &[1, 2]);
        assert!(!circuit.public_outputs.is_empty());
        // Length mismatch → result is 0
        let output_wire = circuit.public_outputs[0];
        assert_eq!(values[output_wire], GoldilocksField::zero());
    }
    
    #[test]
    fn test_differential_random_pairs() {
        let scorer = ExactMatchScorer::case_sensitive();
        
        let test_cases = vec![
            ("", ""),
            ("a", "a"),
            ("a", "b"),
            ("hello world", "hello world"),
            ("Hello", "hello"),
            ("test123", "test123"),
            ("short", "shorter"),
        ];
        
        for (c, r) in test_cases {
            let pair = make_pair(c, r);
            let result = scorer.score_and_verify(&pair);
            assert!(result.agreement, "Disagreement on ({:?}, {:?}): ref={}, aut={}, cir={}",
                c, r, result.reference, result.automaton, result.circuit);
        }
    }
}
