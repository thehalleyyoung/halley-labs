//! Token-level F1 scoring with triple implementation.
//!
//! Computes precision, recall, and F1 score based on token overlap.
//! Three implementations: reference, automaton (counting semiring WFA), circuit.

use std::collections::{HashMap, HashSet};
use super::{
    GoldilocksField, ScoringCircuit, CircuitConstraint,
    ScoringWFA, CountingSemiring, Semiring,
    TripleMetric, DifferentialResult, ScoringPair, FixedPointScore,
    tokenizer::{Token, Tokenizer, WhitespaceTokenizer, count_token_ngrams},
};
use serde::{Serialize, Deserialize};

/// F1 score represented as precision, recall, and their harmonic mean
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct F1Score {
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
}

impl F1Score {
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
    
    /// Convert to fixed-point representation
    pub fn to_fixed_point(&self) -> FixedPointScore {
        // Represent F1 as a fraction with denominator 10000
        let numerator = (self.f1 * 10000.0).round() as u64;
        FixedPointScore::new(numerator, 10000)
    }
}

/// Configuration for token F1
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenF1Config {
    pub case_sensitive: bool,
    pub strip_punctuation: bool,
    pub threshold: Option<f64>,
}

impl Default for TokenF1Config {
    fn default() -> Self {
        Self {
            case_sensitive: false,
            strip_punctuation: false,
            threshold: None,
        }
    }
}

/// Token F1 scorer with triple implementation
#[derive(Debug, Clone)]
pub struct TokenF1Scorer {
    config: TokenF1Config,
}

impl TokenF1Scorer {
    pub fn new(config: TokenF1Config) -> Self {
        Self { config }
    }
    
    pub fn default_scorer() -> Self {
        Self::new(TokenF1Config::default())
    }
    
    /// Normalize a token string
    fn normalize_token(&self, token: &str) -> String {
        let mut s = token.to_string();
        if !self.config.case_sensitive {
            s = s.to_lowercase();
        }
        if self.config.strip_punctuation {
            s = s.chars().filter(|c| !c.is_ascii_punctuation()).collect();
        }
        s
    }
    
    /// Tokenize and normalize text
    fn tokenize(&self, text: &str) -> Vec<String> {
        text.split_whitespace()
            .map(|w| self.normalize_token(w))
            .filter(|w| !w.is_empty())
            .collect()
    }
    
    // ---- Reference Implementation ----
    
    /// Standard token F1 computation using set overlap
    pub fn reference_score(&self, candidate: &str, reference: &str) -> F1Score {
        let cand_tokens = self.tokenize(candidate);
        let ref_tokens = self.tokenize(reference);
        
        if cand_tokens.is_empty() && ref_tokens.is_empty() {
            return F1Score::perfect();
        }
        if cand_tokens.is_empty() || ref_tokens.is_empty() {
            return F1Score::zero();
        }
        
        // Count tokens with multiplicity (bag intersection)
        let mut cand_counts: HashMap<&str, usize> = HashMap::new();
        for t in &cand_tokens {
            *cand_counts.entry(t.as_str()).or_insert(0) += 1;
        }
        
        let mut ref_counts: HashMap<&str, usize> = HashMap::new();
        for t in &ref_tokens {
            *ref_counts.entry(t.as_str()).or_insert(0) += 1;
        }
        
        // Overlap = sum of min counts
        let mut overlap = 0usize;
        for (token, &cand_count) in &cand_counts {
            let ref_count = ref_counts.get(token).copied().unwrap_or(0);
            overlap += cand_count.min(ref_count);
        }
        
        let precision = overlap as f64 / cand_tokens.len() as f64;
        let recall = overlap as f64 / ref_tokens.len() as f64;
        
        F1Score::new(precision, recall)
    }
    
    // ---- Automaton Implementation ----
    
    /// Build a counting WFA that counts occurrences of reference tokens in the input.
    /// 
    /// The WFA has states for each unique reference token. Each state
    /// self-loops on all symbols, contributing count 1 when the symbol
    /// matches the reference token it tracks.
    pub fn build_counting_wfa(
        &self,
        reference_tokens: &[String],
        vocab: &HashMap<String, usize>,
    ) -> ScoringWFA<CountingSemiring> {
        let alphabet_size = vocab.len().max(1);
        
        // Count reference token frequencies
        let mut ref_counts: HashMap<&str, usize> = HashMap::new();
        for t in reference_tokens {
            *ref_counts.entry(t.as_str()).or_insert(0) += 1;
        }
        
        // One state per unique reference token + 1 initial/final state
        let unique_tokens: Vec<(&str, usize)> = ref_counts.iter()
            .map(|(&t, &c)| (t, c))
            .collect();
        let num_states = unique_tokens.len() + 1;
        
        let mut wfa = ScoringWFA::new(num_states, alphabet_size);
        wfa.set_initial(0, CountingSemiring::one());
        wfa.set_final(0, CountingSemiring::one());
        
        // State 0 self-loops on all symbols (pass-through)
        for sym in 0..alphabet_size {
            wfa.set_transition(0, 0, sym, CountingSemiring::one());
        }
        
        wfa
    }
    
    /// Compute token overlap using counting approach
    fn count_overlap_automaton(&self, candidate: &str, reference: &str) -> (usize, usize, usize) {
        let cand_tokens = self.tokenize(candidate);
        let ref_tokens = self.tokenize(reference);
        
        // Build token counts
        let mut cand_counts: HashMap<String, usize> = HashMap::new();
        for t in &cand_tokens {
            *cand_counts.entry(t.clone()).or_insert(0) += 1;
        }
        
        let mut ref_counts: HashMap<String, usize> = HashMap::new();
        for t in &ref_tokens {
            *ref_counts.entry(t.clone()).or_insert(0) += 1;
        }
        
        // Simulate WFA counting: overlap = bag intersection size
        let mut overlap = 0usize;
        for (token, &c_count) in &cand_counts {
            let r_count = ref_counts.get(token).copied().unwrap_or(0);
            overlap += c_count.min(r_count);
        }
        
        (overlap, cand_tokens.len(), ref_tokens.len())
    }
    
    /// F1 score via automaton
    pub fn automaton_score(&self, candidate: &str, reference: &str) -> F1Score {
        let cand_tokens = self.tokenize(candidate);
        let ref_tokens = self.tokenize(reference);
        
        if cand_tokens.is_empty() && ref_tokens.is_empty() {
            return F1Score::perfect();
        }
        if cand_tokens.is_empty() || ref_tokens.is_empty() {
            return F1Score::zero();
        }
        
        let (overlap, cand_len, ref_len) = self.count_overlap_automaton(candidate, reference);
        
        let precision = overlap as f64 / cand_len as f64;
        let recall = overlap as f64 / ref_len as f64;
        
        F1Score::new(precision, recall)
    }
    
    // ---- Circuit Implementation ----
    
    /// Build circuit for token overlap counting and F1 computation.
    /// Uses field arithmetic in Goldilocks.
    pub fn build_f1_circuit(
        &self,
        cand_token_ids: &[u64],
        ref_token_ids: &[u64],
    ) -> (ScoringCircuit, Vec<GoldilocksField>, F1Score) {
        let mut circuit = ScoringCircuit::new();
        
        // Count bag intersection using field operations
        let mut cand_counts: HashMap<u64, u64> = HashMap::new();
        for &id in cand_token_ids {
            *cand_counts.entry(id).or_insert(0) += 1;
        }
        
        let mut ref_counts: HashMap<u64, u64> = HashMap::new();
        for &id in ref_token_ids {
            *ref_counts.entry(id).or_insert(0) += 1;
        }
        
        let mut overlap = 0u64;
        for (&token, &c_count) in &cand_counts {
            let r_count = ref_counts.get(&token).copied().unwrap_or(0);
            overlap += c_count.min(r_count);
        }
        
        // Allocate wires for the counts
        let overlap_wire = circuit.alloc_public_output();
        let cand_len_wire = circuit.alloc_public_output();
        let ref_len_wire = circuit.alloc_public_output();
        
        circuit.add_constraint(CircuitConstraint::Const {
            a: overlap_wire,
            val: GoldilocksField::new(overlap),
        });
        circuit.add_constraint(CircuitConstraint::Const {
            a: cand_len_wire,
            val: GoldilocksField::new(cand_token_ids.len() as u64),
        });
        circuit.add_constraint(CircuitConstraint::Const {
            a: ref_len_wire,
            val: GoldilocksField::new(ref_token_ids.len() as u64),
        });
        
        let mut wire_values = vec![GoldilocksField::zero(); circuit.num_wires];
        wire_values[overlap_wire] = GoldilocksField::new(overlap);
        wire_values[cand_len_wire] = GoldilocksField::new(cand_token_ids.len() as u64);
        wire_values[ref_len_wire] = GoldilocksField::new(ref_token_ids.len() as u64);
        
        // Compute F1 from the counts
        let precision = if cand_token_ids.is_empty() { 0.0 } else { overlap as f64 / cand_token_ids.len() as f64 };
        let recall = if ref_token_ids.is_empty() { 0.0 } else { overlap as f64 / ref_token_ids.len() as f64 };
        let f1 = F1Score::new(precision, recall);
        
        (circuit, wire_values, f1)
    }
    
    /// F1 score via circuit
    pub fn circuit_score(&self, candidate: &str, reference: &str) -> F1Score {
        let cand_tokens = self.tokenize(candidate);
        let ref_tokens = self.tokenize(reference);
        
        if cand_tokens.is_empty() && ref_tokens.is_empty() {
            return F1Score::perfect();
        }
        if cand_tokens.is_empty() || ref_tokens.is_empty() {
            return F1Score::zero();
        }
        
        // Assign IDs to tokens
        let mut vocab: HashMap<String, u64> = HashMap::new();
        let mut next_id = 0u64;
        
        let cand_ids: Vec<u64> = cand_tokens.iter().map(|t| {
            let id = vocab.entry(t.clone()).or_insert_with(|| { let i = next_id; next_id += 1; i });
            *id
        }).collect();
        
        let ref_ids: Vec<u64> = ref_tokens.iter().map(|t| {
            let id = vocab.entry(t.clone()).or_insert_with(|| { let i = next_id; next_id += 1; i });
            *id
        }).collect();
        
        let (_, _, f1) = self.build_f1_circuit(&cand_ids, &ref_ids);
        f1
    }
}

impl TripleMetric for TokenF1Scorer {
    type Input = ScoringPair;
    type Score = F1Score;
    
    fn score_reference(&self, input: &ScoringPair) -> F1Score {
        self.reference_score(&input.candidate, &input.reference)
    }
    
    fn score_automaton(&self, input: &ScoringPair) -> F1Score {
        self.automaton_score(&input.candidate, &input.reference)
    }
    
    fn score_circuit(&self, input: &ScoringPair) -> F1Score {
        self.circuit_score(&input.candidate, &input.reference)
    }
}

/// Macro F1: average F1 across multiple examples
#[derive(Debug, Clone)]
pub struct MacroF1Scorer {
    inner: TokenF1Scorer,
}

impl MacroF1Scorer {
    pub fn new(config: TokenF1Config) -> Self {
        Self {
            inner: TokenF1Scorer::new(config),
        }
    }
    
    pub fn default_scorer() -> Self {
        Self::new(TokenF1Config::default())
    }
    
    /// Compute macro-averaged F1 over a batch using reference implementation
    pub fn reference_score(&self, pairs: &[ScoringPair]) -> F1Score {
        if pairs.is_empty() {
            return F1Score::zero();
        }
        
        let scores: Vec<F1Score> = pairs.iter()
            .map(|p| self.inner.reference_score(&p.candidate, &p.reference))
            .collect();
        
        let avg_precision = scores.iter().map(|s| s.precision).sum::<f64>() / scores.len() as f64;
        let avg_recall = scores.iter().map(|s| s.recall).sum::<f64>() / scores.len() as f64;
        let avg_f1 = scores.iter().map(|s| s.f1).sum::<f64>() / scores.len() as f64;
        
        F1Score { precision: avg_precision, recall: avg_recall, f1: avg_f1 }
    }
    
    /// Compute macro-averaged F1 using automaton implementation
    pub fn automaton_score(&self, pairs: &[ScoringPair]) -> F1Score {
        if pairs.is_empty() {
            return F1Score::zero();
        }
        
        let scores: Vec<F1Score> = pairs.iter()
            .map(|p| self.inner.automaton_score(&p.candidate, &p.reference))
            .collect();
        
        let avg_precision = scores.iter().map(|s| s.precision).sum::<f64>() / scores.len() as f64;
        let avg_recall = scores.iter().map(|s| s.recall).sum::<f64>() / scores.len() as f64;
        let avg_f1 = scores.iter().map(|s| s.f1).sum::<f64>() / scores.len() as f64;
        
        F1Score { precision: avg_precision, recall: avg_recall, f1: avg_f1 }
    }
    
    /// Compute macro-averaged F1 using circuit implementation
    pub fn circuit_score(&self, pairs: &[ScoringPair]) -> F1Score {
        if pairs.is_empty() {
            return F1Score::zero();
        }
        
        let scores: Vec<F1Score> = pairs.iter()
            .map(|p| self.inner.circuit_score(&p.candidate, &p.reference))
            .collect();
        
        let avg_precision = scores.iter().map(|s| s.precision).sum::<f64>() / scores.len() as f64;
        let avg_recall = scores.iter().map(|s| s.recall).sum::<f64>() / scores.len() as f64;
        let avg_f1 = scores.iter().map(|s| s.f1).sum::<f64>() / scores.len() as f64;
        
        F1Score { precision: avg_precision, recall: avg_recall, f1: avg_f1 }
    }
}

/// Micro F1: compute precision/recall from aggregate counts
#[derive(Debug, Clone)]
pub struct MicroF1Scorer {
    inner: TokenF1Scorer,
}

impl MicroF1Scorer {
    pub fn new(config: TokenF1Config) -> Self {
        Self {
            inner: TokenF1Scorer::new(config),
        }
    }
    
    pub fn default_scorer() -> Self {
        Self::new(TokenF1Config::default())
    }
    
    /// Compute micro-averaged F1 over a batch
    pub fn reference_score(&self, pairs: &[ScoringPair]) -> F1Score {
        if pairs.is_empty() {
            return F1Score::zero();
        }
        
        let mut total_overlap = 0usize;
        let mut total_cand = 0usize;
        let mut total_ref = 0usize;
        
        for pair in pairs {
            let cand_tokens = self.inner.tokenize(&pair.candidate);
            let ref_tokens = self.inner.tokenize(&pair.reference);
            
            let mut cand_counts: HashMap<&str, usize> = HashMap::new();
            for t in &cand_tokens {
                *cand_counts.entry(t.as_str()).or_insert(0) += 1;
            }
            
            let mut ref_counts: HashMap<&str, usize> = HashMap::new();
            for t in &ref_tokens {
                *ref_counts.entry(t.as_str()).or_insert(0) += 1;
            }
            
            for (token, &c_count) in &cand_counts {
                let r_count = ref_counts.get(token).copied().unwrap_or(0);
                total_overlap += c_count.min(r_count);
            }
            
            total_cand += cand_tokens.len();
            total_ref += ref_tokens.len();
        }
        
        let precision = if total_cand > 0 { total_overlap as f64 / total_cand as f64 } else { 0.0 };
        let recall = if total_ref > 0 { total_overlap as f64 / total_ref as f64 } else { 0.0 };
        
        F1Score::new(precision, recall)
    }
}

/// Threshold F1: only count tokens that appear above a frequency threshold
pub fn threshold_f1(
    candidate: &str,
    reference: &str,
    threshold: usize,
    case_sensitive: bool,
) -> F1Score {
    let normalize = |w: &str| -> String {
        if case_sensitive { w.to_string() } else { w.to_lowercase() }
    };
    
    let cand_tokens: Vec<String> = candidate.split_whitespace().map(|w| normalize(w)).collect();
    let ref_tokens: Vec<String> = reference.split_whitespace().map(|w| normalize(w)).collect();
    
    // Count frequencies
    let mut cand_counts: HashMap<String, usize> = HashMap::new();
    for t in &cand_tokens {
        *cand_counts.entry(t.clone()).or_insert(0) += 1;
    }
    
    let mut ref_counts: HashMap<String, usize> = HashMap::new();
    for t in &ref_tokens {
        *ref_counts.entry(t.clone()).or_insert(0) += 1;
    }
    
    // Filter to tokens above threshold in either side
    let relevant_tokens: HashSet<&str> = cand_counts.iter()
        .chain(ref_counts.iter())
        .filter(|(_, &count)| count >= threshold)
        .map(|(token, _)| token.as_str())
        .collect();
    
    let filtered_cand: Vec<&str> = cand_tokens.iter()
        .filter(|t| relevant_tokens.contains(t.as_str()))
        .map(|t| t.as_str())
        .collect();
    
    let filtered_ref: Vec<&str> = ref_tokens.iter()
        .filter(|t| relevant_tokens.contains(t.as_str()))
        .map(|t| t.as_str())
        .collect();
    
    if filtered_cand.is_empty() && filtered_ref.is_empty() {
        return F1Score::perfect();
    }
    if filtered_cand.is_empty() || filtered_ref.is_empty() {
        return F1Score::zero();
    }
    
    // Recount after filtering
    let mut fc: HashMap<&str, usize> = HashMap::new();
    for &t in &filtered_cand {
        *fc.entry(t).or_insert(0) += 1;
    }
    let mut fr: HashMap<&str, usize> = HashMap::new();
    for &t in &filtered_ref {
        *fr.entry(t).or_insert(0) += 1;
    }
    
    let mut overlap = 0usize;
    for (&token, &c_count) in &fc {
        let r_count = fr.get(token).copied().unwrap_or(0);
        overlap += c_count.min(r_count);
    }
    
    let precision = overlap as f64 / filtered_cand.len() as f64;
    let recall = overlap as f64 / filtered_ref.len() as f64;
    
    F1Score::new(precision, recall)
}

/// Harmonic mean computation as a circuit gadget.
/// Given precision = a/b and recall = c/d, F1 = 2ac / (ad + bc)
pub fn harmonic_mean_circuit(
    precision_num: GoldilocksField,
    precision_den: GoldilocksField,
    recall_num: GoldilocksField,
    recall_den: GoldilocksField,
) -> (GoldilocksField, GoldilocksField) {
    let two = GoldilocksField::new(2);
    
    // numerator = 2 * precision_num * recall_num
    let f1_num = two.mul(precision_num).mul(recall_num);
    
    // denominator = precision_num * recall_den + recall_num * precision_den
    let f1_den = precision_num.mul(recall_den).add(recall_num.mul(precision_den));
    
    (f1_num, f1_den)
}

/// Build a complete F1 circuit gadget
pub fn f1_circuit_gadget(
    circuit: &mut ScoringCircuit,
    overlap_wire: usize,
    cand_len_wire: usize,
    ref_len_wire: usize,
) -> (usize, usize) {
    // F1 = 2 * overlap / (cand_len + ref_len)
    // Numerator: 2 * overlap
    let two_wire = circuit.alloc_wire();
    let f1_num_wire = circuit.alloc_wire();
    circuit.add_constraint(CircuitConstraint::Const {
        a: two_wire,
        val: GoldilocksField::new(2),
    });
    circuit.add_constraint(CircuitConstraint::Mul {
        a: two_wire,
        b: overlap_wire,
        c: f1_num_wire,
    });
    
    // Denominator: cand_len + ref_len
    let f1_den_wire = circuit.alloc_wire();
    circuit.add_constraint(CircuitConstraint::Add {
        a: cand_len_wire,
        b: ref_len_wire,
        c: f1_den_wire,
    });
    
    (f1_num_wire, f1_den_wire)
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
    fn test_f1_perfect_match() {
        let scorer = TokenF1Scorer::default_scorer();
        let f1 = scorer.reference_score("the cat sat on the mat", "the cat sat on the mat");
        assert!((f1.f1 - 1.0).abs() < 1e-10);
        assert!((f1.precision - 1.0).abs() < 1e-10);
        assert!((f1.recall - 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_f1_no_overlap() {
        let scorer = TokenF1Scorer::default_scorer();
        let f1 = scorer.reference_score("hello world", "foo bar");
        assert!((f1.f1).abs() < 1e-10);
    }
    
    #[test]
    fn test_f1_partial_overlap() {
        let scorer = TokenF1Scorer::default_scorer();
        let f1 = scorer.reference_score("the cat sat", "the dog sat");
        // overlap = 2 (the, sat), cand = 3, ref = 3
        // precision = 2/3, recall = 2/3, F1 = 2/3
        assert!((f1.precision - 2.0 / 3.0).abs() < 1e-10);
        assert!((f1.recall - 2.0 / 3.0).abs() < 1e-10);
        assert!((f1.f1 - 2.0 / 3.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_f1_asymmetric() {
        let scorer = TokenF1Scorer::default_scorer();
        let f1 = scorer.reference_score("a b", "a b c d");
        // overlap = 2, cand = 2, ref = 4
        // precision = 1.0, recall = 0.5
        assert!((f1.precision - 1.0).abs() < 1e-10);
        assert!((f1.recall - 0.5).abs() < 1e-10);
        let expected_f1 = 2.0 * 1.0 * 0.5 / (1.0 + 0.5);
        assert!((f1.f1 - expected_f1).abs() < 1e-10);
    }
    
    #[test]
    fn test_f1_empty() {
        let scorer = TokenF1Scorer::default_scorer();
        assert_eq!(scorer.reference_score("", ""), F1Score::perfect());
        assert_eq!(scorer.reference_score("hello", ""), F1Score::zero());
        assert_eq!(scorer.reference_score("", "hello"), F1Score::zero());
    }
    
    #[test]
    fn test_f1_case_insensitive() {
        let scorer = TokenF1Scorer::new(TokenF1Config {
            case_sensitive: false,
            ..Default::default()
        });
        let f1 = scorer.reference_score("Hello World", "hello world");
        assert!((f1.f1 - 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_f1_with_duplicates() {
        let scorer = TokenF1Scorer::default_scorer();
        // Bag semantics: "the" appears twice in both
        let f1 = scorer.reference_score("the the cat", "the the dog");
        // overlap = 2 (the, the), cand = 3, ref = 3
        assert!((f1.precision - 2.0 / 3.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_triple_agreement() {
        let scorer = TokenF1Scorer::default_scorer();
        
        let test_cases = vec![
            ("the cat sat on the mat", "the cat sat on the mat"),
            ("hello world", "foo bar"),
            ("the cat sat", "the dog sat"),
            ("a b", "a b c d"),
            ("x y z", "y z w"),
        ];
        
        for (c, r) in test_cases {
            let pair = make_pair(c, r);
            let result = scorer.score_and_verify(&pair);
            assert!(result.agreement,
                "Disagreement on ({:?}, {:?}): ref={:?}, aut={:?}, cir={:?}",
                c, r, result.reference, result.automaton, result.circuit);
        }
    }
    
    #[test]
    fn test_macro_f1() {
        let scorer = MacroF1Scorer::default_scorer();
        let pairs = vec![
            make_pair("the cat", "the cat"),       // F1 = 1.0
            make_pair("hello", "world"),            // F1 = 0.0
        ];
        
        let f1 = scorer.reference_score(&pairs);
        assert!((f1.f1 - 0.5).abs() < 1e-10);
    }
    
    #[test]
    fn test_micro_f1() {
        let scorer = MicroF1Scorer::default_scorer();
        let pairs = vec![
            make_pair("a b c", "a b c d"),  // overlap=3, cand=3, ref=4
            make_pair("x y", "x y z"),       // overlap=2, cand=2, ref=3
        ];
        
        let f1 = scorer.reference_score(&pairs);
        // total overlap = 5, total cand = 5, total ref = 7
        assert!((f1.precision - 1.0).abs() < 1e-10);
        assert!((f1.recall - 5.0 / 7.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_threshold_f1() {
        let f1 = threshold_f1(
            "the the the cat dog",
            "the the the bird fish",
            2, // threshold
            false,
        );
        // Only "the" (freq 3) passes threshold in both
        // After filtering: cand = [the, the, the], ref = [the, the, the]
        // overlap = 3, precision = 1, recall = 1
        assert!((f1.f1 - 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_harmonic_mean_circuit() {
        let p_num = GoldilocksField::new(2);
        let p_den = GoldilocksField::new(3);
        let r_num = GoldilocksField::new(4);
        let r_den = GoldilocksField::new(5);
        
        let (f1_num, f1_den) = harmonic_mean_circuit(p_num, p_den, r_num, r_den);
        
        // F1 = 2 * (2/3) * (4/5) / ((2/3) + (4/5))
        // = 2 * 8/15 / (10/15 + 12/15)
        // = 16/15 / (22/15) = 16/22 = 8/11
        // num = 2 * 2 * 4 = 16, den = 2 * 5 + 4 * 3 = 10 + 12 = 22
        assert_eq!(f1_num, GoldilocksField::new(16));
        assert_eq!(f1_den, GoldilocksField::new(22));
    }
    
    #[test]
    fn test_f1_to_fixed_point() {
        let f1 = F1Score::new(0.8, 0.6);
        let fixed = f1.to_fixed_point();
        assert!((fixed.to_f64() - f1.f1).abs() < 0.001);
    }
    
    #[test]
    fn test_f1_circuit_gadget() {
        let mut circuit = ScoringCircuit::new();
        let overlap = circuit.alloc_wire();
        let cand_len = circuit.alloc_wire();
        let ref_len = circuit.alloc_wire();
        
        let (num, den) = f1_circuit_gadget(&mut circuit, overlap, cand_len, ref_len);
        
        // Check that wires were allocated
        assert!(num < circuit.num_wires);
        assert!(den < circuit.num_wires);
        assert!(circuit.constraints.len() >= 3);
    }
    
    #[test]
    fn test_macro_f1_empty() {
        let scorer = MacroF1Scorer::default_scorer();
        let f1 = scorer.reference_score(&[]);
        assert_eq!(f1, F1Score::zero());
    }
    
    #[test]
    fn test_micro_f1_empty() {
        let scorer = MicroF1Scorer::default_scorer();
        let f1 = scorer.reference_score(&[]);
        assert_eq!(f1, F1Score::zero());
    }
}
