//! Scoring Function Library with Triple Implementations
//!
//! Each NLP metric has three implementations:
//! 1. Reference: standard algorithm implementation
//! 2. Automaton: WFA-based implementation using semiring operations
//! 3. Circuit: field arithmetic representation for ZK proof generation
//!
//! Differential testing verifies all three agree on every input.

pub mod tokenizer;
pub mod exact_match;
pub mod token_f1;
pub mod bleu;
pub mod rouge;
pub mod regex_match;
pub mod pass_at_k;
pub mod differential;

pub use tokenizer::{Tokenizer, WhitespaceTokenizer, WordPieceTokenizer, CharacterTokenizer, NGramTokenizer};
pub use exact_match::{ExactMatchScorer, NormalizedExactMatchScorer};
pub use token_f1::{TokenF1Scorer, MacroF1Scorer, MicroF1Scorer};
pub use bleu::{BleuScorer, SmoothingMethod, BleuConfig};
pub use rouge::{RougeNScorer, RougeLScorer, RougeConfig};
pub use regex_match::{RegexMatchScorer, RegexCompiler};
pub use pass_at_k::{PassAtKScorer, PassAtKConfig};
pub use differential::{DifferentialTester, DifferentialResult, AgreementReport};

use serde::{Serialize, Deserialize};

/// The Goldilocks prime field modulus: p = 2^64 - 2^32 + 1
pub const GOLDILOCKS_MODULUS: u64 = 0xFFFFFFFF00000001;

/// Trait for metrics that provide all three implementations.
pub trait TripleMetric {
    /// Input type for the metric
    type Input;
    /// Output score type
    type Score: PartialEq + std::fmt::Debug;
    
    /// Reference implementation (standard algorithm)
    fn score_reference(&self, input: &Self::Input) -> Self::Score;
    
    /// Automaton-based implementation (WFA with appropriate semiring)
    fn score_automaton(&self, input: &Self::Input) -> Self::Score;
    
    /// Circuit-based implementation (Goldilocks field arithmetic)
    fn score_circuit(&self, input: &Self::Input) -> Self::Score;
    
    /// Run all three and check agreement
    fn score_and_verify(&self, input: &Self::Input) -> DifferentialResult<Self::Score> {
        let ref_score = self.score_reference(input);
        let aut_score = self.score_automaton(input);
        let cir_score = self.score_circuit(input);
        
        let all_agree = ref_score == aut_score && aut_score == cir_score;
        
        DifferentialResult {
            reference: ref_score,
            automaton: aut_score,
            circuit: cir_score,
            agreement: all_agree,
        }
    }
}

/// A scored pair of candidate and reference text
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoringPair {
    pub candidate: String,
    pub reference: String,
}

/// A scored pair with multiple references
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiRefScoringPair {
    pub candidate: String,
    pub references: Vec<String>,
}

/// Fixed-point score representation in Goldilocks field
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FixedPointScore {
    /// Numerator in the field
    pub numerator: u64,
    /// Denominator in the field (never zero)
    pub denominator: u64,
}

impl FixedPointScore {
    pub fn new(numerator: u64, denominator: u64) -> Self {
        assert!(denominator > 0, "Denominator must be non-zero");
        Self { numerator, denominator }
    }
    
    pub fn zero() -> Self {
        Self { numerator: 0, denominator: 1 }
    }
    
    pub fn one() -> Self {
        Self { numerator: 1, denominator: 1 }
    }
    
    pub fn to_f64(&self) -> f64 {
        self.numerator as f64 / self.denominator as f64
    }
    
    /// Reduce to lowest terms
    pub fn reduce(&self) -> Self {
        let g = gcd(self.numerator, self.denominator);
        Self {
            numerator: self.numerator / g,
            denominator: self.denominator / g,
        }
    }
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// Goldilocks field element
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GoldilocksField(pub u64);

impl GoldilocksField {
    pub const MODULUS: u64 = GOLDILOCKS_MODULUS;
    
    pub fn new(val: u64) -> Self {
        Self(val % Self::MODULUS)
    }
    
    pub fn zero() -> Self { Self(0) }
    pub fn one() -> Self { Self(1) }
    
    pub fn add(self, other: Self) -> Self {
        let sum = (self.0 as u128 + other.0 as u128) % Self::MODULUS as u128;
        Self(sum as u64)
    }
    
    pub fn sub(self, other: Self) -> Self {
        if self.0 >= other.0 {
            Self(self.0 - other.0)
        } else {
            Self(Self::MODULUS - other.0 + self.0)
        }
    }
    
    pub fn mul(self, other: Self) -> Self {
        let prod = (self.0 as u128 * other.0 as u128) % Self::MODULUS as u128;
        Self(prod as u64)
    }
    
    /// Modular inverse via Fermat's little theorem: a^(p-2) mod p
    pub fn inv(self) -> Self {
        assert!(self.0 != 0, "Cannot invert zero");
        self.pow(Self::MODULUS - 2)
    }
    
    pub fn pow(self, mut exp: u64) -> Self {
        let mut base = self;
        let mut result = Self::one();
        while exp > 0 {
            if exp & 1 == 1 {
                result = result.mul(base);
            }
            base = base.mul(base);
            exp >>= 1;
        }
        result
    }
    
    pub fn div(self, other: Self) -> Self {
        self.mul(other.inv())
    }
}

/// Circuit constraint types used in scoring
#[derive(Debug, Clone)]
pub enum CircuitConstraint {
    /// a * b = c (multiplication gate)
    Mul { a: usize, b: usize, c: usize },
    /// a + b = c (addition gate)
    Add { a: usize, b: usize, c: usize },
    /// a == b (equality constraint)
    Eq { a: usize, b: usize },
    /// a == constant
    Const { a: usize, val: GoldilocksField },
    /// Boolean constraint: a * (1 - a) = 0
    Bool { a: usize },
}

/// A simple arithmetic circuit for scoring
#[derive(Debug, Clone)]
pub struct ScoringCircuit {
    pub num_wires: usize,
    pub constraints: Vec<CircuitConstraint>,
    pub public_inputs: Vec<usize>,
    pub public_outputs: Vec<usize>,
}

impl ScoringCircuit {
    pub fn new() -> Self {
        Self {
            num_wires: 0,
            constraints: Vec::new(),
            public_inputs: Vec::new(),
            public_outputs: Vec::new(),
        }
    }
    
    pub fn alloc_wire(&mut self) -> usize {
        let w = self.num_wires;
        self.num_wires += 1;
        w
    }
    
    pub fn alloc_public_input(&mut self) -> usize {
        let w = self.alloc_wire();
        self.public_inputs.push(w);
        w
    }
    
    pub fn alloc_public_output(&mut self) -> usize {
        let w = self.alloc_wire();
        self.public_outputs.push(w);
        w
    }
    
    pub fn add_constraint(&mut self, constraint: CircuitConstraint) {
        self.constraints.push(constraint);
    }
    
    /// Add equality check gadget, returns wire for result (0 or 1)
    pub fn add_equality_gadget(&mut self, a: usize, b: usize) -> usize {
        let diff = self.alloc_wire();
        let is_zero = self.alloc_wire();
        self.add_constraint(CircuitConstraint::Add { a: b, b: diff, c: a });
        self.add_constraint(CircuitConstraint::Bool { a: is_zero });
        is_zero
    }
    
    /// Evaluate circuit with given wire values, returns true if all constraints satisfied
    pub fn check_satisfaction(&self, values: &[GoldilocksField]) -> bool {
        if values.len() < self.num_wires {
            return false;
        }
        for c in &self.constraints {
            match c {
                CircuitConstraint::Mul { a, b, c } => {
                    if values[*a].mul(values[*b]) != values[*c] { return false; }
                }
                CircuitConstraint::Add { a, b, c } => {
                    if values[*a].add(values[*b]) != values[*c] { return false; }
                }
                CircuitConstraint::Eq { a, b } => {
                    if values[*a] != values[*b] { return false; }
                }
                CircuitConstraint::Const { a, val } => {
                    if values[*a] != *val { return false; }
                }
                CircuitConstraint::Bool { a } => {
                    let v = values[*a];
                    if v.mul(GoldilocksField::one().sub(v)) != GoldilocksField::zero() { return false; }
                }
            }
        }
        true
    }
}

/// Semiring trait for WFA-based scoring
pub trait Semiring: Clone + std::fmt::Debug + PartialEq {
    fn zero() -> Self;
    fn one() -> Self;
    fn add(&self, other: &Self) -> Self;
    fn mul(&self, other: &Self) -> Self;
}

/// Boolean semiring for exact match
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BooleanSemiring(pub bool);

impl Semiring for BooleanSemiring {
    fn zero() -> Self { Self(false) }
    fn one() -> Self { Self(true) }
    fn add(&self, other: &Self) -> Self { Self(self.0 || other.0) }
    fn mul(&self, other: &Self) -> Self { Self(self.0 && other.0) }
}

/// Counting (natural number) semiring for F1/BLEU
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CountingSemiring(pub u64);

impl Semiring for CountingSemiring {
    fn zero() -> Self { Self(0) }
    fn one() -> Self { Self(1) }
    fn add(&self, other: &Self) -> Self { Self(self.0.saturating_add(other.0)) }
    fn mul(&self, other: &Self) -> Self { Self(self.0.saturating_mul(other.0)) }
}

/// Bounded counting semiring (for clipped n-gram counts in BLEU)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoundedCountingSemiring {
    pub count: u64,
    pub bound: u64,
}

impl BoundedCountingSemiring {
    pub fn new(count: u64, bound: u64) -> Self {
        Self { count: count.min(bound), bound }
    }
}

impl Semiring for BoundedCountingSemiring {
    fn zero() -> Self { Self { count: 0, bound: u64::MAX } }
    fn one() -> Self { Self { count: 1, bound: u64::MAX } }
    fn add(&self, other: &Self) -> Self {
        let b = self.bound.min(other.bound);
        Self { count: self.count.saturating_add(other.count).min(b), bound: b }
    }
    fn mul(&self, other: &Self) -> Self {
        let b = self.bound.min(other.bound);
        Self { count: self.count.saturating_mul(other.count).min(b), bound: b }
    }
}

/// Tropical semiring (for ROUGE-L / LCS via shortest paths)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TropicalSemiring(pub f64);

impl Semiring for TropicalSemiring {
    fn zero() -> Self { Self(f64::INFINITY) }
    fn one() -> Self { Self(0.0) }
    fn add(&self, other: &Self) -> Self { Self(self.0.min(other.0)) }
    fn mul(&self, other: &Self) -> Self { Self(self.0 + other.0) }
}

/// Max-plus semiring (alternative tropical for longest paths)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MaxPlusSemiring(pub f64);

impl Semiring for MaxPlusSemiring {
    fn zero() -> Self { Self(f64::NEG_INFINITY) }
    fn one() -> Self { Self(0.0) }
    fn add(&self, other: &Self) -> Self { Self(self.0.max(other.0)) }
    fn mul(&self, other: &Self) -> Self { Self(self.0 + other.0) }
}

/// A generic weighted finite automaton for scoring
#[derive(Debug, Clone)]
pub struct ScoringWFA<S: Semiring> {
    pub num_states: usize,
    pub alphabet_size: usize,
    pub initial_weights: Vec<S>,
    pub final_weights: Vec<S>,
    pub transitions: Vec<Vec<Vec<S>>>,
}

impl<S: Semiring> ScoringWFA<S> {
    pub fn new(num_states: usize, alphabet_size: usize) -> Self {
        let zero = S::zero();
        Self {
            num_states,
            alphabet_size,
            initial_weights: vec![zero.clone(); num_states],
            final_weights: vec![zero.clone(); num_states],
            transitions: vec![vec![vec![zero; alphabet_size]; num_states]; num_states],
        }
    }
    
    pub fn set_initial(&mut self, state: usize, weight: S) {
        self.initial_weights[state] = weight;
    }
    
    pub fn set_final(&mut self, state: usize, weight: S) {
        self.final_weights[state] = weight;
    }
    
    pub fn set_transition(&mut self, from: usize, to: usize, symbol: usize, weight: S) {
        self.transitions[from][to][symbol] = weight;
    }
    
    /// Run the WFA on a sequence of symbols, returns the total weight
    pub fn run(&self, input: &[usize]) -> S {
        let mut current = self.initial_weights.clone();
        
        for &sym in input {
            let mut next = vec![S::zero(); self.num_states];
            for to in 0..self.num_states {
                let mut sum = S::zero();
                for from in 0..self.num_states {
                    let w = current[from].mul(&self.transitions[from][to][sym]);
                    sum = sum.add(&w);
                }
                next[to] = sum;
            }
            current = next;
        }
        
        let mut total = S::zero();
        for (i, w) in current.iter().enumerate() {
            total = total.add(&w.mul(&self.final_weights[i]));
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_goldilocks_field_arithmetic() {
        let a = GoldilocksField::new(42);
        let b = GoldilocksField::new(17);
        
        let sum = a.add(b);
        assert_eq!(sum, GoldilocksField::new(59));
        
        let prod = a.mul(b);
        assert_eq!(prod, GoldilocksField::new(714));
        
        let one = GoldilocksField::one();
        assert_eq!(a.mul(one), a);
        assert_eq!(a.add(GoldilocksField::zero()), a);
    }
    
    #[test]
    fn test_goldilocks_inverse() {
        let a = GoldilocksField::new(42);
        let inv = a.inv();
        let prod = a.mul(inv);
        assert_eq!(prod, GoldilocksField::one());
    }
    
    #[test]
    fn test_boolean_semiring() {
        let t = BooleanSemiring::one();
        let f = BooleanSemiring::zero();
        assert_eq!(t.mul(&f), f);
        assert_eq!(t.add(&f), t);
        assert_eq!(f.add(&f), f);
        assert_eq!(t.mul(&t), t);
    }
    
    #[test]
    fn test_counting_semiring() {
        let a = CountingSemiring(3);
        let b = CountingSemiring(5);
        assert_eq!(a.add(&b), CountingSemiring(8));
        assert_eq!(a.mul(&b), CountingSemiring(15));
    }
    
    #[test]
    fn test_tropical_semiring() {
        let a = TropicalSemiring(3.0);
        let b = TropicalSemiring(5.0);
        assert_eq!(a.add(&b).0, 3.0); // min
        assert_eq!(a.mul(&b).0, 8.0); // sum
    }
    
    #[test]
    fn test_simple_wfa_boolean() {
        // WFA that accepts only the string [0, 1]
        let mut wfa = ScoringWFA::<BooleanSemiring>::new(3, 2);
        wfa.set_initial(0, BooleanSemiring::one());
        wfa.set_final(2, BooleanSemiring::one());
        wfa.set_transition(0, 1, 0, BooleanSemiring::one());
        wfa.set_transition(1, 2, 1, BooleanSemiring::one());
        
        assert_eq!(wfa.run(&[0, 1]), BooleanSemiring::one());
        assert_eq!(wfa.run(&[1, 0]), BooleanSemiring::zero());
        assert_eq!(wfa.run(&[0]), BooleanSemiring::zero());
    }
    
    #[test]
    fn test_scoring_circuit_satisfaction() {
        let mut circuit = ScoringCircuit::new();
        let a = circuit.alloc_wire();
        let b = circuit.alloc_wire();
        let c = circuit.alloc_wire();
        circuit.add_constraint(CircuitConstraint::Mul { a, b, c });
        
        let values = vec![
            GoldilocksField::new(3),
            GoldilocksField::new(7),
            GoldilocksField::new(21),
        ];
        assert!(circuit.check_satisfaction(&values));
        
        let bad_values = vec![
            GoldilocksField::new(3),
            GoldilocksField::new(7),
            GoldilocksField::new(22),
        ];
        assert!(!circuit.check_satisfaction(&bad_values));
    }
    
    #[test]
    fn test_fixed_point_score() {
        let s = FixedPointScore::new(3, 4);
        assert!((s.to_f64() - 0.75).abs() < 1e-10);
        
        let s2 = FixedPointScore::new(6, 8);
        assert_eq!(s2.reduce(), s.reduce());
    }
}
