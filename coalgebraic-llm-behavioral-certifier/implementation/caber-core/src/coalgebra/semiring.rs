//! Semiring implementations for weighted automata and coalgebraic computations.
//!
//! A semiring (S, ⊕, ⊗, 0̄, 1̄) provides the algebraic backbone for
//! weighted transitions. Different semirings yield different interpretations:
//! - ProbabilitySemiring: probabilistic behavior
//! - TropicalSemiring: shortest-path / Viterbi decoding
//! - BooleanSemiring: reachability
//! - CountingSemiring: path counting
//! - LogSemiring: numerically stable probabilistic computation

use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::ops::{Add, Mul};

use nalgebra::{DMatrix, DVector};
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Semiring trait
// ---------------------------------------------------------------------------

/// A semiring (S, ⊕, ⊗, 0̄, 1̄) satisfying:
///   - (S, ⊕, 0̄) is a commutative monoid
///   - (S, ⊗, 1̄) is a monoid
///   - ⊗ distributes over ⊕
///   - 0̄ annihilates under ⊗
pub trait Semiring: Clone + PartialEq + fmt::Debug + Send + Sync + 'static {
    fn zero() -> Self;
    fn one() -> Self;
    fn add(&self, other: &Self) -> Self;
    fn mul(&self, other: &Self) -> Self;

    fn is_zero(&self) -> bool {
        *self == Self::zero()
    }

    fn is_one(&self) -> bool {
        *self == Self::one()
    }

    /// Compute self^n via repeated squaring.
    fn pow(&self, mut n: u32) -> Self {
        if n == 0 {
            return Self::one();
        }
        let mut result = self.clone();
        let mut base = self.clone();
        n -= 1;
        while n > 0 {
            if n % 2 == 1 {
                result = result.mul(&base);
            }
            base = base.mul(&base);
            n /= 2;
        }
        result
    }

    /// Sum a slice of elements.
    fn sum_iter<'a, I: Iterator<Item = &'a Self>>(iter: I) -> Self
    where
        Self: 'a,
    {
        iter.fold(Self::zero(), |acc, x| acc.add(x))
    }

    /// Product of a slice of elements.
    fn product_iter<'a, I: Iterator<Item = &'a Self>>(iter: I) -> Self
    where
        Self: 'a,
    {
        iter.fold(Self::one(), |acc, x| acc.mul(x))
    }
}

/// A star-semiring additionally supports the Kleene star: a* = 1 ⊕ a ⊕ a² ⊕ ...
pub trait StarSemiring: Semiring {
    fn star(&self) -> Self;

    /// Plus closure: a⁺ = a ⊗ a*
    fn plus(&self) -> Self {
        self.mul(&self.star())
    }
}

/// An ordered semiring where elements have a natural ordering compatible
/// with the semiring operations.
pub trait OrderedSemiring: Semiring + PartialOrd {}

/// A complete semiring where infinite sums are well-defined.
pub trait CompleteSemiring: Semiring {
    fn infinite_sum<I: IntoIterator<Item = Self>>(iter: I) -> Self;
}

// ---------------------------------------------------------------------------
// ProbabilitySemiring: [0, 1] with + = +, × = ×
// ---------------------------------------------------------------------------

/// The probability semiring ([0,1], +, ×, 0, 1) where addition is clamped
/// to [0,1]. Used for probabilistic coalgebras.
#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct ProbabilitySemiring {
    pub value: f64,
}

impl ProbabilitySemiring {
    pub fn new(value: f64) -> Self {
        debug_assert!(value >= -1e-10 && value <= 1.0 + 1e-10,
            "Probability value out of range: {}", value);
        Self {
            value: value.clamp(0.0, 1.0),
        }
    }

    pub fn from_log(log_value: f64) -> Self {
        Self::new(log_value.exp())
    }

    pub fn to_log(&self) -> f64 {
        if self.value <= 0.0 {
            f64::NEG_INFINITY
        } else {
            self.value.ln()
        }
    }

    pub fn complement(&self) -> Self {
        Self::new(1.0 - self.value)
    }

    /// Interpolate between self and other: self*(1-t) + other*t
    pub fn interpolate(&self, other: &Self, t: f64) -> Self {
        let t = t.clamp(0.0, 1.0);
        Self::new(self.value * (1.0 - t) + other.value * t)
    }
}

impl fmt::Debug for ProbabilitySemiring {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "P({:.6})", self.value)
    }
}

impl fmt::Display for ProbabilitySemiring {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.6}", self.value)
    }
}

impl PartialEq for ProbabilitySemiring {
    fn eq(&self, other: &Self) -> bool {
        (self.value - other.value).abs() < 1e-12
    }
}

impl Eq for ProbabilitySemiring {}

impl Hash for ProbabilitySemiring {
    fn hash<H: Hasher>(&self, state: &mut H) {
        OrderedFloat(self.value).hash(state);
    }
}

impl PartialOrd for ProbabilitySemiring {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.value.partial_cmp(&other.value)
    }
}

impl Semiring for ProbabilitySemiring {
    fn zero() -> Self {
        Self { value: 0.0 }
    }

    fn one() -> Self {
        Self { value: 1.0 }
    }

    fn add(&self, other: &Self) -> Self {
        Self::new((self.value + other.value).min(1.0))
    }

    fn mul(&self, other: &Self) -> Self {
        Self::new(self.value * other.value)
    }
}

impl StarSemiring for ProbabilitySemiring {
    fn star(&self) -> Self {
        if self.value >= 1.0 {
            Self::new(1.0)
        } else {
            Self::new(1.0 / (1.0 - self.value))
        }
    }
}

impl OrderedSemiring for ProbabilitySemiring {}

// ---------------------------------------------------------------------------
// TropicalSemiring: (ℝ ∪ {∞}, min, +, ∞, 0)
// ---------------------------------------------------------------------------

/// The tropical (min-plus) semiring used for shortest-path computations.
/// ⊕ = min, ⊗ = +, 0̄ = ∞, 1̄ = 0
#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct TropicalSemiring {
    pub value: f64,
}

impl TropicalSemiring {
    pub fn new(value: f64) -> Self {
        Self { value }
    }

    pub fn infinity() -> Self {
        Self {
            value: f64::INFINITY,
        }
    }

    pub fn is_infinite(&self) -> bool {
        self.value.is_infinite()
    }

    pub fn from_probability(p: f64) -> Self {
        if p <= 0.0 {
            Self::infinity()
        } else {
            Self::new(-p.ln())
        }
    }

    pub fn to_probability(&self) -> f64 {
        (-self.value).exp()
    }
}

impl fmt::Debug for TropicalSemiring {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_infinite() {
            write!(f, "T(∞)")
        } else {
            write!(f, "T({:.6})", self.value)
        }
    }
}

impl fmt::Display for TropicalSemiring {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_infinite() {
            write!(f, "∞")
        } else {
            write!(f, "{:.6}", self.value)
        }
    }
}

impl PartialEq for TropicalSemiring {
    fn eq(&self, other: &Self) -> bool {
        if self.value.is_infinite() && other.value.is_infinite() {
            return true;
        }
        (self.value - other.value).abs() < 1e-12
    }
}

impl Eq for TropicalSemiring {}

impl Hash for TropicalSemiring {
    fn hash<H: Hasher>(&self, state: &mut H) {
        OrderedFloat(self.value).hash(state);
    }
}

impl PartialOrd for TropicalSemiring {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // In tropical semiring, smaller value = "larger" in the semiring order
        // since ⊕ = min, x ≤ y iff x ⊕ y = x iff x ≤_ℝ y
        self.value.partial_cmp(&other.value)
    }
}

impl Semiring for TropicalSemiring {
    fn zero() -> Self {
        Self::infinity()
    }

    fn one() -> Self {
        Self::new(0.0)
    }

    fn add(&self, other: &Self) -> Self {
        Self::new(self.value.min(other.value))
    }

    fn mul(&self, other: &Self) -> Self {
        if self.is_infinite() || other.is_infinite() {
            Self::infinity()
        } else {
            Self::new(self.value + other.value)
        }
    }

    fn is_zero(&self) -> bool {
        self.is_infinite()
    }
}

impl StarSemiring for TropicalSemiring {
    fn star(&self) -> Self {
        // a* = 0̄ if a ≥ 0 (since min(0, a, 2a, ...) = 0 for a ≥ 0)
        if self.value >= 0.0 || self.is_infinite() {
            Self::one()
        } else {
            // For negative values, the star is -∞, but we handle it as infinity
            Self::new(f64::NEG_INFINITY)
        }
    }
}

impl OrderedSemiring for TropicalSemiring {}

// ---------------------------------------------------------------------------
// ViterbiSemiring: ([0, 1], max, ×, 0, 1)
// ---------------------------------------------------------------------------

/// The Viterbi semiring for computing most-probable paths.
/// ⊕ = max, ⊗ = ×, 0̄ = 0, 1̄ = 1
#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct ViterbiSemiring {
    pub value: f64,
}

impl ViterbiSemiring {
    pub fn new(value: f64) -> Self {
        Self {
            value: value.clamp(0.0, 1.0),
        }
    }

    pub fn from_log(log_value: f64) -> Self {
        Self::new(log_value.exp())
    }

    pub fn to_log(&self) -> f64 {
        if self.value <= 0.0 {
            f64::NEG_INFINITY
        } else {
            self.value.ln()
        }
    }
}

impl fmt::Debug for ViterbiSemiring {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "V({:.6})", self.value)
    }
}

impl fmt::Display for ViterbiSemiring {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.6}", self.value)
    }
}

impl PartialEq for ViterbiSemiring {
    fn eq(&self, other: &Self) -> bool {
        (self.value - other.value).abs() < 1e-12
    }
}

impl Eq for ViterbiSemiring {}

impl Hash for ViterbiSemiring {
    fn hash<H: Hasher>(&self, state: &mut H) {
        OrderedFloat(self.value).hash(state);
    }
}

impl PartialOrd for ViterbiSemiring {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.value.partial_cmp(&other.value)
    }
}

impl Semiring for ViterbiSemiring {
    fn zero() -> Self {
        Self { value: 0.0 }
    }

    fn one() -> Self {
        Self { value: 1.0 }
    }

    fn add(&self, other: &Self) -> Self {
        Self::new(self.value.max(other.value))
    }

    fn mul(&self, other: &Self) -> Self {
        Self::new(self.value * other.value)
    }
}

impl StarSemiring for ViterbiSemiring {
    fn star(&self) -> Self {
        // max(1, a, a², ...) = 1 for a ≤ 1
        Self::one()
    }
}

impl OrderedSemiring for ViterbiSemiring {}

// ---------------------------------------------------------------------------
// BooleanSemiring: ({false, true}, ∨, ∧, false, true)
// ---------------------------------------------------------------------------

/// The Boolean semiring for reachability analysis.
/// ⊕ = ∨, ⊗ = ∧, 0̄ = false, 1̄ = true
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BooleanSemiring {
    pub value: bool,
}

impl BooleanSemiring {
    pub fn new(value: bool) -> Self {
        Self { value }
    }

    pub fn true_val() -> Self {
        Self { value: true }
    }

    pub fn false_val() -> Self {
        Self { value: false }
    }
}

impl fmt::Debug for BooleanSemiring {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "B({})", self.value)
    }
}

impl fmt::Display for BooleanSemiring {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

impl PartialOrd for BooleanSemiring {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.value.partial_cmp(&other.value)
    }
}

impl Semiring for BooleanSemiring {
    fn zero() -> Self {
        Self { value: false }
    }

    fn one() -> Self {
        Self { value: true }
    }

    fn add(&self, other: &Self) -> Self {
        Self {
            value: self.value || other.value,
        }
    }

    fn mul(&self, other: &Self) -> Self {
        Self {
            value: self.value && other.value,
        }
    }
}

impl StarSemiring for BooleanSemiring {
    fn star(&self) -> Self {
        Self::one()
    }
}

impl OrderedSemiring for BooleanSemiring {}

// ---------------------------------------------------------------------------
// CountingSemiring: (ℕ, +, ×, 0, 1)
// ---------------------------------------------------------------------------

/// The counting (natural number) semiring for path counting.
/// ⊕ = +, ⊗ = ×, 0̄ = 0, 1̄ = 1
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CountingSemiring {
    pub value: u64,
}

impl CountingSemiring {
    pub fn new(value: u64) -> Self {
        Self { value }
    }
}

impl fmt::Debug for CountingSemiring {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "N({})", self.value)
    }
}

impl fmt::Display for CountingSemiring {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

impl PartialOrd for CountingSemiring {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.value.partial_cmp(&other.value)
    }
}

impl Ord for CountingSemiring {
    fn cmp(&self, other: &Self) -> Ordering {
        self.value.cmp(&other.value)
    }
}

impl Semiring for CountingSemiring {
    fn zero() -> Self {
        Self { value: 0 }
    }

    fn one() -> Self {
        Self { value: 1 }
    }

    fn add(&self, other: &Self) -> Self {
        Self {
            value: self.value.saturating_add(other.value),
        }
    }

    fn mul(&self, other: &Self) -> Self {
        Self {
            value: self.value.saturating_mul(other.value),
        }
    }
}

impl OrderedSemiring for CountingSemiring {}

// ---------------------------------------------------------------------------
// LogSemiring: (ℝ ∪ {-∞}, log-sum-exp, +, -∞, 0)
// ---------------------------------------------------------------------------

/// The log semiring for numerically stable probabilistic computation.
/// Values represent log-probabilities.
/// ⊕ = log-sum-exp, ⊗ = +, 0̄ = -∞, 1̄ = 0
#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct LogSemiring {
    pub value: f64,
}

impl LogSemiring {
    pub fn new(value: f64) -> Self {
        Self { value }
    }

    pub fn neg_infinity() -> Self {
        Self {
            value: f64::NEG_INFINITY,
        }
    }

    pub fn from_probability(p: f64) -> Self {
        if p <= 0.0 {
            Self::neg_infinity()
        } else {
            Self::new(p.ln())
        }
    }

    pub fn to_probability(&self) -> f64 {
        self.value.exp()
    }

    pub fn is_neg_infinity(&self) -> bool {
        self.value == f64::NEG_INFINITY
    }

    /// Numerically stable log-sum-exp: log(exp(a) + exp(b))
    fn log_sum_exp(a: f64, b: f64) -> f64 {
        if a == f64::NEG_INFINITY {
            return b;
        }
        if b == f64::NEG_INFINITY {
            return a;
        }
        let max_val = a.max(b);
        max_val + ((a - max_val).exp() + (b - max_val).exp()).ln()
    }

    /// Numerically stable log-sum-exp for a sequence.
    pub fn log_sum_exp_seq(values: &[f64]) -> f64 {
        if values.is_empty() {
            return f64::NEG_INFINITY;
        }
        let max_val = values
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        if max_val == f64::NEG_INFINITY {
            return f64::NEG_INFINITY;
        }
        max_val + values.iter().map(|v| (v - max_val).exp()).sum::<f64>().ln()
    }
}

impl fmt::Debug for LogSemiring {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_neg_infinity() {
            write!(f, "L(-∞)")
        } else {
            write!(f, "L({:.6})", self.value)
        }
    }
}

impl fmt::Display for LogSemiring {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_neg_infinity() {
            write!(f, "-∞")
        } else {
            write!(f, "{:.6}", self.value)
        }
    }
}

impl PartialEq for LogSemiring {
    fn eq(&self, other: &Self) -> bool {
        if self.is_neg_infinity() && other.is_neg_infinity() {
            return true;
        }
        (self.value - other.value).abs() < 1e-12
    }
}

impl Eq for LogSemiring {}

impl Hash for LogSemiring {
    fn hash<H: Hasher>(&self, state: &mut H) {
        OrderedFloat(self.value).hash(state);
    }
}

impl PartialOrd for LogSemiring {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.value.partial_cmp(&other.value)
    }
}

impl Semiring for LogSemiring {
    fn zero() -> Self {
        Self::neg_infinity()
    }

    fn one() -> Self {
        Self::new(0.0)
    }

    fn add(&self, other: &Self) -> Self {
        Self::new(Self::log_sum_exp(self.value, other.value))
    }

    fn mul(&self, other: &Self) -> Self {
        if self.is_neg_infinity() || other.is_neg_infinity() {
            Self::neg_infinity()
        } else {
            Self::new(self.value + other.value)
        }
    }

    fn is_zero(&self) -> bool {
        self.is_neg_infinity()
    }
}

impl StarSemiring for LogSemiring {
    fn star(&self) -> Self {
        // a* = 1/(1 - exp(a)) in the probability domain
        // In log domain: -log(1 - exp(a))
        if self.is_neg_infinity() {
            return Self::one(); // 0* = 1
        }
        let p = self.to_probability();
        if p >= 1.0 {
            Self::new(f64::INFINITY)
        } else {
            Self::from_probability(1.0 / (1.0 - p))
        }
    }
}

impl OrderedSemiring for LogSemiring {}

// ---------------------------------------------------------------------------
// Formal power series semiring: S[[x₁, ..., xₙ]]
// ---------------------------------------------------------------------------

/// A formal power series over a semiring S, represented as a map from
/// multi-indices to coefficients. Used for generating functions.
#[derive(Debug, Clone)]
pub struct FormalPowerSeries<S: Semiring> {
    coefficients: std::collections::BTreeMap<Vec<u32>, S>,
    num_variables: usize,
    max_degree: usize,
}

impl<S: Semiring> FormalPowerSeries<S> {
    pub fn new(num_variables: usize, max_degree: usize) -> Self {
        Self {
            coefficients: std::collections::BTreeMap::new(),
            num_variables,
            max_degree,
        }
    }

    pub fn zero(num_variables: usize, max_degree: usize) -> Self {
        Self::new(num_variables, max_degree)
    }

    pub fn one(num_variables: usize, max_degree: usize) -> Self {
        let mut fps = Self::new(num_variables, max_degree);
        fps.set_coefficient(vec![0; num_variables], S::one());
        fps
    }

    pub fn monomial(exponents: Vec<u32>, coefficient: S, num_variables: usize, max_degree: usize) -> Self {
        let mut fps = Self::new(num_variables, max_degree);
        fps.set_coefficient(exponents, coefficient);
        fps
    }

    pub fn set_coefficient(&mut self, exponents: Vec<u32>, value: S) {
        assert_eq!(exponents.len(), self.num_variables);
        let degree: u32 = exponents.iter().sum();
        if (degree as usize) <= self.max_degree {
            if value.is_zero() {
                self.coefficients.remove(&exponents);
            } else {
                self.coefficients.insert(exponents, value);
            }
        }
    }

    pub fn get_coefficient(&self, exponents: &[u32]) -> S {
        self.coefficients
            .get(exponents)
            .cloned()
            .unwrap_or_else(S::zero)
    }

    pub fn add_fps(&self, other: &Self) -> Self {
        assert_eq!(self.num_variables, other.num_variables);
        let max_deg = self.max_degree.max(other.max_degree);
        let mut result = Self::new(self.num_variables, max_deg);

        for (exp, coeff) in &self.coefficients {
            let other_coeff = other.get_coefficient(exp);
            result.set_coefficient(exp.clone(), coeff.add(&other_coeff));
        }
        for (exp, coeff) in &other.coefficients {
            if !self.coefficients.contains_key(exp) {
                result.set_coefficient(exp.clone(), coeff.clone());
            }
        }
        result
    }

    pub fn mul_fps(&self, other: &Self) -> Self {
        assert_eq!(self.num_variables, other.num_variables);
        let max_deg = self.max_degree;
        let mut result = Self::new(self.num_variables, max_deg);

        for (exp1, coeff1) in &self.coefficients {
            for (exp2, coeff2) in &other.coefficients {
                let mut new_exp = vec![0u32; self.num_variables];
                let mut valid = true;
                let mut total_degree = 0u32;
                for i in 0..self.num_variables {
                    new_exp[i] = exp1[i] + exp2[i];
                    total_degree += new_exp[i];
                    if (total_degree as usize) > max_deg {
                        valid = false;
                        break;
                    }
                }
                if valid {
                    let existing = result.get_coefficient(&new_exp);
                    let prod = coeff1.mul(coeff2);
                    result.set_coefficient(new_exp, existing.add(&prod));
                }
            }
        }
        result
    }

    pub fn scale(&self, factor: &S) -> Self {
        let mut result = Self::new(self.num_variables, self.max_degree);
        for (exp, coeff) in &self.coefficients {
            result.set_coefficient(exp.clone(), coeff.mul(factor));
        }
        result
    }

    pub fn num_terms(&self) -> usize {
        self.coefficients.len()
    }

    pub fn is_zero_fps(&self) -> bool {
        self.coefficients.is_empty()
    }

    /// Evaluate the series at a specific point.
    pub fn evaluate(&self, point: &[S]) -> S {
        assert_eq!(point.len(), self.num_variables);
        let mut result = S::zero();
        for (exp, coeff) in &self.coefficients {
            let mut term = coeff.clone();
            for (i, &e) in exp.iter().enumerate() {
                term = term.mul(&point[i].pow(e));
            }
            result = result.add(&term);
        }
        result
    }

    /// Truncate to a given degree.
    pub fn truncate(&self, degree: usize) -> Self {
        let mut result = Self::new(self.num_variables, degree);
        for (exp, coeff) in &self.coefficients {
            let total: u32 = exp.iter().sum();
            if (total as usize) <= degree {
                result.set_coefficient(exp.clone(), coeff.clone());
            }
        }
        result
    }
}

// ---------------------------------------------------------------------------
// Matrix operations over semirings
// ---------------------------------------------------------------------------

/// Matrix over an arbitrary semiring, for weighted automata operations.
#[derive(Debug, Clone)]
pub struct SemiringMatrix<S: Semiring> {
    data: Vec<Vec<S>>,
    rows: usize,
    cols: usize,
}

impl<S: Semiring> SemiringMatrix<S> {
    pub fn new(rows: usize, cols: usize) -> Self {
        let data = vec![vec![S::zero(); cols]; rows];
        Self { data, rows, cols }
    }

    pub fn identity(n: usize) -> Self {
        let mut m = Self::new(n, n);
        for i in 0..n {
            m.data[i][i] = S::one();
        }
        m
    }

    pub fn from_fn<F: Fn(usize, usize) -> S>(rows: usize, cols: usize, f: F) -> Self {
        let data = (0..rows)
            .map(|i| (0..cols).map(|j| f(i, j)).collect())
            .collect();
        Self { data, rows, cols }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn get(&self, i: usize, j: usize) -> &S {
        &self.data[i][j]
    }

    pub fn set(&mut self, i: usize, j: usize, value: S) {
        self.data[i][j] = value;
    }

    pub fn get_row(&self, i: usize) -> &[S] {
        &self.data[i]
    }

    pub fn add_matrix(&self, other: &Self) -> Self {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        Self::from_fn(self.rows, self.cols, |i, j| {
            self.data[i][j].add(&other.data[i][j])
        })
    }

    pub fn mul_matrix(&self, other: &Self) -> Self {
        assert_eq!(self.cols, other.rows);
        Self::from_fn(self.rows, other.cols, |i, j| {
            let mut sum = S::zero();
            for k in 0..self.cols {
                sum = sum.add(&self.data[i][k].mul(&other.data[k][j]));
            }
            sum
        })
    }

    pub fn scale_matrix(&self, factor: &S) -> Self {
        Self::from_fn(self.rows, self.cols, |i, j| {
            self.data[i][j].mul(factor)
        })
    }

    pub fn transpose(&self) -> Self {
        Self::from_fn(self.cols, self.rows, |i, j| self.data[j][i].clone())
    }

    /// Matrix power via repeated squaring.
    pub fn pow_matrix(&self, mut n: u32) -> Self {
        assert_eq!(self.rows, self.cols, "Matrix must be square for power");
        if n == 0 {
            return Self::identity(self.rows);
        }
        let mut result = self.clone();
        let mut base = self.clone();
        n -= 1;
        while n > 0 {
            if n % 2 == 1 {
                result = result.mul_matrix(&base);
            }
            base = base.mul_matrix(&base);
            n /= 2;
        }
        result
    }

    /// Compute the Kleene star (I ⊕ M ⊕ M² ⊕ ...) via iterative squaring.
    /// Converges for matrices over star-semirings.
    pub fn star_matrix(&self, max_iterations: usize) -> Self
    where
        S: StarSemiring,
    {
        assert_eq!(self.rows, self.cols);
        let n = self.rows;

        // Floyd-Warshall style computation for the star
        let mut result = self.clone();
        for k in 0..n {
            let star_kk = result.data[k][k].star();
            for i in 0..n {
                for j in 0..n {
                    if i != k && j != k {
                        let ik_kj = result.data[i][k].mul(&star_kk).mul(&result.data[k][j]);
                        result.data[i][j] = result.data[i][j].add(&ik_kj);
                    }
                }
            }
            for i in 0..n {
                if i != k {
                    result.data[i][k] = result.data[i][k].mul(&star_kk);
                    result.data[k][i] = star_kk.mul(&result.data[k][i]);
                }
            }
            result.data[k][k] = star_kk;
        }
        let _ = max_iterations;
        result
    }

    /// Matrix-vector multiplication.
    pub fn mul_vector(&self, vec: &SemiringVector<S>) -> SemiringVector<S> {
        assert_eq!(self.cols, vec.len());
        let data: Vec<S> = (0..self.rows)
            .map(|i| {
                let mut sum = S::zero();
                for j in 0..self.cols {
                    sum = sum.add(&self.data[i][j].mul(&vec.data[j]));
                }
                sum
            })
            .collect();
        SemiringVector { data }
    }

    /// Check if all entries are zero.
    pub fn is_zero_matrix(&self) -> bool {
        self.data.iter().all(|row| row.iter().all(|x| x.is_zero()))
    }

    /// Hadamard (element-wise) product.
    pub fn hadamard(&self, other: &Self) -> Self {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        Self::from_fn(self.rows, self.cols, |i, j| {
            self.data[i][j].mul(&other.data[i][j])
        })
    }

    /// Kronecker product.
    pub fn kronecker(&self, other: &Self) -> Self {
        let new_rows = self.rows * other.rows;
        let new_cols = self.cols * other.cols;
        Self::from_fn(new_rows, new_cols, |i, j| {
            let ai = i / other.rows;
            let bi = i % other.rows;
            let aj = j / other.cols;
            let bj = j % other.cols;
            self.data[ai][aj].mul(&other.data[bi][bj])
        })
    }

    /// Trace (sum of diagonal elements).
    pub fn trace_sr(&self) -> S {
        assert_eq!(self.rows, self.cols);
        let mut sum = S::zero();
        for i in 0..self.rows {
            sum = sum.add(&self.data[i][i]);
        }
        sum
    }

    /// Extract a submatrix.
    pub fn submatrix(&self, row_indices: &[usize], col_indices: &[usize]) -> Self {
        Self::from_fn(row_indices.len(), col_indices.len(), |i, j| {
            self.data[row_indices[i]][col_indices[j]].clone()
        })
    }

    /// Row permutation.
    pub fn permute_rows(&self, perm: &[usize]) -> Self {
        assert_eq!(perm.len(), self.rows);
        Self::from_fn(self.rows, self.cols, |i, j| {
            self.data[perm[i]][j].clone()
        })
    }

    /// Column permutation.
    pub fn permute_cols(&self, perm: &[usize]) -> Self {
        assert_eq!(perm.len(), self.cols);
        Self::from_fn(self.rows, self.cols, |i, j| {
            self.data[i][perm[j]].clone()
        })
    }

    /// Direct sum (block diagonal).
    pub fn direct_sum(&self, other: &Self) -> Self {
        let new_rows = self.rows + other.rows;
        let new_cols = self.cols + other.cols;
        Self::from_fn(new_rows, new_cols, |i, j| {
            if i < self.rows && j < self.cols {
                self.data[i][j].clone()
            } else if i >= self.rows && j >= self.cols {
                other.data[i - self.rows][j - self.cols].clone()
            } else {
                S::zero()
            }
        })
    }
}

/// Vector over a semiring.
#[derive(Debug, Clone)]
pub struct SemiringVector<S: Semiring> {
    data: Vec<S>,
}

impl<S: Semiring> SemiringVector<S> {
    pub fn new(size: usize) -> Self {
        Self {
            data: vec![S::zero(); size],
        }
    }

    pub fn from_vec(data: Vec<S>) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn get(&self, i: usize) -> &S {
        &self.data[i]
    }

    pub fn set(&mut self, i: usize, value: S) {
        self.data[i] = value;
    }

    pub fn add_vector(&self, other: &Self) -> Self {
        assert_eq!(self.len(), other.len());
        Self {
            data: self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| a.add(b))
                .collect(),
        }
    }

    pub fn scale_vector(&self, factor: &S) -> Self {
        Self {
            data: self.data.iter().map(|x| x.mul(factor)).collect(),
        }
    }

    pub fn dot(&self, other: &Self) -> S {
        assert_eq!(self.len(), other.len());
        self.data
            .iter()
            .zip(other.data.iter())
            .fold(S::zero(), |acc, (a, b)| acc.add(&a.mul(b)))
    }

    /// Outer product: v ⊗ w yields a matrix.
    pub fn outer_product(&self, other: &Self) -> SemiringMatrix<S> {
        SemiringMatrix::from_fn(self.len(), other.len(), |i, j| {
            self.data[i].mul(&other.data[j])
        })
    }
}

// ---------------------------------------------------------------------------
// Weighted finite automaton
// ---------------------------------------------------------------------------

/// A weighted finite automaton (WFA) over a semiring S.
/// Recognizes formal power series in S⟨⟨Σ*⟩⟩.
#[derive(Debug, Clone)]
pub struct WeightedFiniteAutomaton<S: Semiring> {
    pub num_states: usize,
    pub alphabet_size: usize,
    pub initial: SemiringVector<S>,
    pub final_weights: SemiringVector<S>,
    pub transitions: Vec<SemiringMatrix<S>>, // one matrix per alphabet symbol
}

impl<S: Semiring> WeightedFiniteAutomaton<S> {
    pub fn new(num_states: usize, alphabet_size: usize) -> Self {
        let transitions = (0..alphabet_size)
            .map(|_| SemiringMatrix::new(num_states, num_states))
            .collect();
        Self {
            num_states,
            alphabet_size,
            initial: SemiringVector::new(num_states),
            final_weights: SemiringVector::new(num_states),
            transitions,
        }
    }

    pub fn set_initial(&mut self, state: usize, weight: S) {
        self.initial.set(state, weight);
    }

    pub fn set_final(&mut self, state: usize, weight: S) {
        self.final_weights.set(state, weight);
    }

    pub fn set_transition(&mut self, from: usize, to: usize, symbol: usize, weight: S) {
        self.transitions[symbol].set(from, to, weight);
    }

    /// Compute the weight of a word (sequence of symbol indices).
    pub fn weight_of_word(&self, word: &[usize]) -> S {
        let mut state_vec = self.initial.clone();
        for &sym in word {
            assert!(sym < self.alphabet_size);
            state_vec = self.transitions[sym].mul_vector(&state_vec);
        }
        state_vec.dot(&self.final_weights)
    }

    /// Compute the weight of all words up to length n.
    pub fn weight_of_all_words_up_to(&self, max_len: usize) -> Vec<(Vec<usize>, S)> {
        let mut results = Vec::new();
        let mut frontier: Vec<(Vec<usize>, SemiringVector<S>)> = vec![(Vec::new(), self.initial.clone())];

        // Empty word
        let empty_weight = self.initial.dot(&self.final_weights);
        if !empty_weight.is_zero() {
            results.push((Vec::new(), empty_weight));
        }

        for _depth in 0..max_len {
            let mut next_frontier = Vec::new();
            for (prefix, state_vec) in &frontier {
                for sym in 0..self.alphabet_size {
                    let new_vec = self.transitions[sym].mul_vector(state_vec);
                    let weight = new_vec.dot(&self.final_weights);
                    let mut new_word = prefix.clone();
                    new_word.push(sym);
                    if !weight.is_zero() {
                        results.push((new_word.clone(), weight));
                    }
                    next_frontier.push((new_word, new_vec));
                }
            }
            frontier = next_frontier;
        }
        results
    }

    /// Minimize the WFA using the forward-backward algorithm.
    pub fn minimize(&self) -> Self
    where
        S: PartialEq,
    {
        // Compute forward and backward reachability matrices
        let n = self.num_states;
        if n == 0 {
            return self.clone();
        }

        // Hankel-like matrix approach: compute the forward vectors for all words
        // and backward vectors, then find linearly independent ones.
        // For now, use a basic state-merging approach.
        let mut merged = vec![false; n];
        let mut representatives: Vec<usize> = (0..n).collect();

        for i in 0..n {
            if merged[i] {
                continue;
            }
            for j in (i + 1)..n {
                if merged[j] {
                    continue;
                }
                // Check if states i and j are equivalent
                let equiv = self.check_state_equivalence(i, j, 10);
                if equiv {
                    merged[j] = true;
                    representatives[j] = i;
                }
            }
        }

        // Build the minimized automaton
        let active: Vec<usize> = (0..n).filter(|&i| !merged[i]).collect();
        let num_active = active.len();
        let mut state_map = vec![0usize; n];
        for (new_idx, &old_idx) in active.iter().enumerate() {
            state_map[old_idx] = new_idx;
        }
        for i in 0..n {
            if merged[i] {
                state_map[i] = state_map[representatives[i]];
            }
        }

        let mut result = Self::new(num_active, self.alphabet_size);
        for (new_idx, &old_idx) in active.iter().enumerate() {
            result.set_initial(new_idx, self.initial.get(old_idx).clone());
            result.set_final(new_idx, self.final_weights.get(old_idx).clone());
        }

        for sym in 0..self.alphabet_size {
            for &from in &active {
                for to in 0..n {
                    let w = self.transitions[sym].get(from, to);
                    if !w.is_zero() {
                        let new_from = state_map[from];
                        let new_to = state_map[to];
                        let existing = result.transitions[sym].get(new_from, new_to).clone();
                        result.transitions[sym].set(new_from, new_to, existing.add(w));
                    }
                }
            }
        }

        result
    }

    fn check_state_equivalence(&self, s1: usize, s2: usize, depth: usize) -> bool {
        // Check if final weights match
        if self.final_weights.get(s1) != self.final_weights.get(s2) {
            return false;
        }

        if depth == 0 {
            return true;
        }

        // Check if transition weights match for one step
        for sym in 0..self.alphabet_size {
            for target in 0..self.num_states {
                let w1 = self.transitions[sym].get(s1, target);
                let w2 = self.transitions[sym].get(s2, target);
                if w1 != w2 {
                    return false;
                }
            }
        }

        true
    }

    /// Product of two WFAs (intersection for Boolean, product for probability).
    pub fn product(&self, other: &Self) -> Self {
        assert_eq!(self.alphabet_size, other.alphabet_size);
        let n1 = self.num_states;
        let n2 = other.num_states;
        let n = n1 * n2;

        let mut result = Self::new(n, self.alphabet_size);

        for i in 0..n1 {
            for j in 0..n2 {
                let state = i * n2 + j;
                let init = self.initial.get(i).mul(other.initial.get(j));
                result.set_initial(state, init);
                let fin = self.final_weights.get(i).mul(other.final_weights.get(j));
                result.set_final(state, fin);
            }
        }

        for sym in 0..self.alphabet_size {
            for i1 in 0..n1 {
                for j1 in 0..n2 {
                    for i2 in 0..n1 {
                        for j2 in 0..n2 {
                            let from = i1 * n2 + j1;
                            let to = i2 * n2 + j2;
                            let w = self.transitions[sym]
                                .get(i1, i2)
                                .mul(other.transitions[sym].get(j1, j2));
                            result.transitions[sym].set(from, to, w);
                        }
                    }
                }
            }
        }

        result
    }

    /// Sum of two WFAs (union).
    pub fn sum(&self, other: &Self) -> Self {
        assert_eq!(self.alphabet_size, other.alphabet_size);
        let n1 = self.num_states;
        let n2 = other.num_states;
        let n = n1 + n2;

        let mut result = Self::new(n, self.alphabet_size);

        for i in 0..n1 {
            result.set_initial(i, self.initial.get(i).clone());
            result.set_final(i, self.final_weights.get(i).clone());
        }
        for i in 0..n2 {
            result.set_initial(n1 + i, other.initial.get(i).clone());
            result.set_final(n1 + i, other.final_weights.get(i).clone());
        }

        for sym in 0..self.alphabet_size {
            for i in 0..n1 {
                for j in 0..n1 {
                    result.transitions[sym].set(i, j, self.transitions[sym].get(i, j).clone());
                }
            }
            for i in 0..n2 {
                for j in 0..n2 {
                    result.transitions[sym].set(
                        n1 + i,
                        n1 + j,
                        other.transitions[sym].get(i, j).clone(),
                    );
                }
            }
        }

        result
    }

    /// Compute spectral radius of transition matrices (approximation).
    pub fn spectral_radius_approx(&self, iterations: usize) -> f64
    where
        S: Into<f64> + Copy,
    {
        // Power iteration on the sum of all transition matrices
        let n = self.num_states;
        if n == 0 {
            return 0.0;
        }

        let mut v = vec![1.0 / (n as f64).sqrt(); n];
        let mut eigenvalue = 0.0;

        for _ in 0..iterations {
            let mut new_v = vec![0.0; n];
            for sym in 0..self.alphabet_size {
                for i in 0..n {
                    for j in 0..n {
                        let w: f64 = (*self.transitions[sym].get(i, j)).into();
                        new_v[i] += w * v[j];
                    }
                }
            }

            let norm: f64 = new_v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 1e-15 {
                eigenvalue = norm;
                for x in new_v.iter_mut() {
                    *x /= norm;
                }
            } else {
                return 0.0;
            }
            v = new_v;
        }

        eigenvalue
    }
}

// ---------------------------------------------------------------------------
// Conversion utilities
// ---------------------------------------------------------------------------

impl From<ProbabilitySemiring> for f64 {
    fn from(p: ProbabilitySemiring) -> f64 {
        p.value
    }
}

impl From<ViterbiSemiring> for f64 {
    fn from(v: ViterbiSemiring) -> f64 {
        v.value
    }
}

impl From<LogSemiring> for f64 {
    fn from(l: LogSemiring) -> f64 {
        l.value
    }
}

impl From<TropicalSemiring> for f64 {
    fn from(t: TropicalSemiring) -> f64 {
        t.value
    }
}

impl From<CountingSemiring> for f64 {
    fn from(c: CountingSemiring) -> f64 {
        c.value as f64
    }
}

impl From<BooleanSemiring> for f64 {
    fn from(b: BooleanSemiring) -> f64 {
        if b.value { 1.0 } else { 0.0 }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- Probability semiring tests ---

    #[test]
    fn test_prob_semiring_axioms() {
        let zero = ProbabilitySemiring::zero();
        let one = ProbabilitySemiring::one();
        let a = ProbabilitySemiring::new(0.3);
        let b = ProbabilitySemiring::new(0.5);
        let c = ProbabilitySemiring::new(0.2);

        // Identity
        assert_eq!(a.add(&zero), a);
        assert_eq!(a.mul(&one), a);

        // Annihilation
        assert_eq!(a.mul(&zero), zero);

        // Commutativity of add
        assert_eq!(a.add(&b), b.add(&a));

        // Associativity of mul
        assert_eq!(a.mul(&b).mul(&c), a.mul(&b.mul(&c)));

        // Distributivity
        let ab_plus_c = a.mul(&b.add(&c));
        let ab_plus_ac = a.mul(&b).add(&a.mul(&c));
        assert!((ab_plus_c.value - ab_plus_ac.value).abs() < 1e-10);
    }

    #[test]
    fn test_prob_complement() {
        let p = ProbabilitySemiring::new(0.3);
        assert!((p.complement().value - 0.7).abs() < 1e-10);
    }

    #[test]
    fn test_prob_interpolate() {
        let a = ProbabilitySemiring::new(0.0);
        let b = ProbabilitySemiring::new(1.0);
        let mid = a.interpolate(&b, 0.5);
        assert!((mid.value - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_prob_log_roundtrip() {
        let p = ProbabilitySemiring::new(0.42);
        let log_p = p.to_log();
        let p2 = ProbabilitySemiring::from_log(log_p);
        assert!((p.value - p2.value).abs() < 1e-10);
    }

    #[test]
    fn test_prob_star() {
        let p = ProbabilitySemiring::new(0.5);
        let s = p.star();
        // 1/(1-0.5) = 2.0, but clamped to 1.0
        assert!((s.value - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_prob_pow() {
        let p = ProbabilitySemiring::new(0.5);
        let p3 = p.pow(3);
        assert!((p3.value - 0.125).abs() < 1e-10);
    }

    // --- Tropical semiring tests ---

    #[test]
    fn test_tropical_axioms() {
        let zero = TropicalSemiring::zero();
        let one = TropicalSemiring::one();
        let a = TropicalSemiring::new(3.0);
        let b = TropicalSemiring::new(5.0);
        let c = TropicalSemiring::new(2.0);

        // Identity
        assert_eq!(a.add(&zero), a);
        assert_eq!(a.mul(&one), a);

        // Annihilation
        assert_eq!(a.mul(&zero), zero);

        // Commutativity of ⊕ = min
        assert_eq!(a.add(&b), b.add(&a));

        // Associativity
        assert_eq!(a.add(&b).add(&c), a.add(&b.add(&c)));
        assert_eq!(a.mul(&b).mul(&c), a.mul(&b.mul(&c)));

        // Distributivity: a ⊗ (b ⊕ c) = (a ⊗ b) ⊕ (a ⊗ c)
        // a + min(b, c) = min(a+b, a+c)
        assert_eq!(a.mul(&b.add(&c)), a.mul(&b).add(&a.mul(&c)));
    }

    #[test]
    fn test_tropical_operations() {
        let a = TropicalSemiring::new(3.0);
        let b = TropicalSemiring::new(5.0);

        // min(3, 5) = 3
        assert_eq!(a.add(&b), a);

        // 3 + 5 = 8
        assert_eq!(a.mul(&b), TropicalSemiring::new(8.0));
    }

    #[test]
    fn test_tropical_probability_conversion() {
        let p = 0.3f64;
        let t = TropicalSemiring::from_probability(p);
        let p2 = t.to_probability();
        assert!((p - p2).abs() < 1e-10);
    }

    #[test]
    fn test_tropical_star() {
        let t = TropicalSemiring::new(5.0);
        let s = t.star();
        assert_eq!(s, TropicalSemiring::one()); // star of non-negative = 0
    }

    // --- Viterbi semiring tests ---

    #[test]
    fn test_viterbi_axioms() {
        let zero = ViterbiSemiring::zero();
        let one = ViterbiSemiring::one();
        let a = ViterbiSemiring::new(0.3);
        let b = ViterbiSemiring::new(0.7);

        assert_eq!(a.add(&zero), a);
        assert_eq!(a.mul(&one), a);
        assert_eq!(a.mul(&zero), zero);
        assert_eq!(a.add(&b), b.add(&a));
    }

    #[test]
    fn test_viterbi_operations() {
        let a = ViterbiSemiring::new(0.3);
        let b = ViterbiSemiring::new(0.7);

        // max(0.3, 0.7) = 0.7
        assert!((a.add(&b).value - 0.7).abs() < 1e-10);

        // 0.3 * 0.7 = 0.21
        assert!((a.mul(&b).value - 0.21).abs() < 1e-10);
    }

    // --- Boolean semiring tests ---

    #[test]
    fn test_boolean_axioms() {
        let f = BooleanSemiring::false_val();
        let t = BooleanSemiring::true_val();

        assert_eq!(t.add(&f), t);
        assert_eq!(f.add(&f), f);
        assert_eq!(t.mul(&t), t);
        assert_eq!(t.mul(&f), f);
        assert_eq!(f.mul(&f), f);
    }

    // --- Counting semiring tests ---

    #[test]
    fn test_counting_axioms() {
        let zero = CountingSemiring::zero();
        let one = CountingSemiring::one();
        let a = CountingSemiring::new(3);
        let b = CountingSemiring::new(5);

        assert_eq!(a.add(&zero), a);
        assert_eq!(a.mul(&one), a);
        assert_eq!(a.mul(&zero), zero);
        assert_eq!(a.add(&b), CountingSemiring::new(8));
        assert_eq!(a.mul(&b), CountingSemiring::new(15));
    }

    #[test]
    fn test_counting_pow() {
        let c = CountingSemiring::new(3);
        assert_eq!(c.pow(4), CountingSemiring::new(81));
    }

    // --- Log semiring tests ---

    #[test]
    fn test_log_axioms() {
        let zero = LogSemiring::zero();
        let one = LogSemiring::one();
        let a = LogSemiring::from_probability(0.3);
        let b = LogSemiring::from_probability(0.5);

        assert_eq!(a.add(&zero), a);
        assert_eq!(a.mul(&one), a);
        assert_eq!(a.mul(&zero), zero);
    }

    #[test]
    fn test_log_sum_exp() {
        let a = LogSemiring::from_probability(0.3);
        let b = LogSemiring::from_probability(0.5);
        let sum = a.add(&b);
        let expected = 0.3 + 0.5;
        assert!((sum.to_probability() - expected).abs() < 1e-10);
    }

    #[test]
    fn test_log_mul() {
        let a = LogSemiring::from_probability(0.3);
        let b = LogSemiring::from_probability(0.5);
        let prod = a.mul(&b);
        let expected = 0.3 * 0.5;
        assert!((prod.to_probability() - expected).abs() < 1e-10);
    }

    #[test]
    fn test_log_sum_exp_seq() {
        let values = vec![
            (0.1f64).ln(),
            (0.2f64).ln(),
            (0.3f64).ln(),
        ];
        let result = LogSemiring::log_sum_exp_seq(&values);
        assert!((result.exp() - 0.6).abs() < 1e-10);
    }

    // --- Formal power series tests ---

    #[test]
    fn test_fps_basic() {
        let f = FormalPowerSeries::<CountingSemiring>::one(2, 3);
        assert_eq!(f.get_coefficient(&[0, 0]), CountingSemiring::one());
        assert_eq!(f.get_coefficient(&[1, 0]), CountingSemiring::zero());
    }

    #[test]
    fn test_fps_add() {
        let mut f = FormalPowerSeries::<CountingSemiring>::new(1, 3);
        f.set_coefficient(vec![1], CountingSemiring::new(2));
        let mut g = FormalPowerSeries::<CountingSemiring>::new(1, 3);
        g.set_coefficient(vec![1], CountingSemiring::new(3));

        let h = f.add_fps(&g);
        assert_eq!(h.get_coefficient(&[1]), CountingSemiring::new(5));
    }

    #[test]
    fn test_fps_mul() {
        // (1 + 2x) * (1 + 3x) = 1 + 5x + 6x²
        let mut f = FormalPowerSeries::<CountingSemiring>::new(1, 3);
        f.set_coefficient(vec![0], CountingSemiring::one());
        f.set_coefficient(vec![1], CountingSemiring::new(2));

        let mut g = FormalPowerSeries::<CountingSemiring>::new(1, 3);
        g.set_coefficient(vec![0], CountingSemiring::one());
        g.set_coefficient(vec![1], CountingSemiring::new(3));

        let h = f.mul_fps(&g);
        assert_eq!(h.get_coefficient(&[0]), CountingSemiring::one());
        assert_eq!(h.get_coefficient(&[1]), CountingSemiring::new(5));
        assert_eq!(h.get_coefficient(&[2]), CountingSemiring::new(6));
    }

    #[test]
    fn test_fps_evaluate() {
        // f(x) = 1 + 2x + 3x²
        let mut f = FormalPowerSeries::<CountingSemiring>::new(1, 3);
        f.set_coefficient(vec![0], CountingSemiring::one());
        f.set_coefficient(vec![1], CountingSemiring::new(2));
        f.set_coefficient(vec![2], CountingSemiring::new(3));

        let result = f.evaluate(&[CountingSemiring::new(2)]);
        // 1 + 2*2 + 3*4 = 1 + 4 + 12 = 17
        assert_eq!(result, CountingSemiring::new(17));
    }

    // --- Semiring matrix tests ---

    #[test]
    fn test_matrix_identity() {
        let id = SemiringMatrix::<CountingSemiring>::identity(3);
        assert_eq!(*id.get(0, 0), CountingSemiring::one());
        assert_eq!(*id.get(0, 1), CountingSemiring::zero());
    }

    #[test]
    fn test_matrix_add() {
        let a = SemiringMatrix::from_fn(2, 2, |i, j| {
            CountingSemiring::new((i * 2 + j + 1) as u64)
        });
        let b = SemiringMatrix::from_fn(2, 2, |i, j| {
            CountingSemiring::new((i * 2 + j + 5) as u64)
        });
        let c = a.add_matrix(&b);
        assert_eq!(*c.get(0, 0), CountingSemiring::new(6)); // 1 + 5
        assert_eq!(*c.get(1, 1), CountingSemiring::new(12)); // 4 + 8
    }

    #[test]
    fn test_matrix_mul() {
        // [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]]
        let a = SemiringMatrix::from_fn(2, 2, |i, j| {
            CountingSemiring::new((i * 2 + j + 1) as u64)
        });
        let b = SemiringMatrix::from_fn(2, 2, |i, j| {
            CountingSemiring::new((i * 2 + j + 5) as u64)
        });
        let c = a.mul_matrix(&b);
        assert_eq!(*c.get(0, 0), CountingSemiring::new(19));
        assert_eq!(*c.get(0, 1), CountingSemiring::new(22));
        assert_eq!(*c.get(1, 0), CountingSemiring::new(43));
        assert_eq!(*c.get(1, 1), CountingSemiring::new(50));
    }

    #[test]
    fn test_matrix_transpose() {
        let a = SemiringMatrix::from_fn(2, 3, |i, j| {
            CountingSemiring::new((i * 3 + j + 1) as u64)
        });
        let t = a.transpose();
        assert_eq!(t.rows(), 3);
        assert_eq!(t.cols(), 2);
        assert_eq!(*t.get(0, 0), CountingSemiring::new(1));
        assert_eq!(*t.get(0, 1), CountingSemiring::new(4));
    }

    #[test]
    fn test_matrix_pow() {
        let a = SemiringMatrix::from_fn(2, 2, |i, j| {
            CountingSemiring::new((i * 2 + j + 1) as u64)
        });
        let a0 = a.pow_matrix(0);
        assert_eq!(*a0.get(0, 0), CountingSemiring::one());
        assert_eq!(*a0.get(0, 1), CountingSemiring::zero());

        let a1 = a.pow_matrix(1);
        assert_eq!(*a1.get(0, 0), *a.get(0, 0));
    }

    #[test]
    fn test_matrix_vector_mul() {
        let m = SemiringMatrix::from_fn(2, 2, |i, j| {
            CountingSemiring::new((i * 2 + j + 1) as u64)
        });
        let v = SemiringVector::from_vec(vec![
            CountingSemiring::new(1),
            CountingSemiring::new(2),
        ]);
        let result = m.mul_vector(&v);
        // [1*1+2*2, 3*1+4*2] = [5, 11]
        assert_eq!(*result.get(0), CountingSemiring::new(5));
        assert_eq!(*result.get(1), CountingSemiring::new(11));
    }

    #[test]
    fn test_matrix_hadamard() {
        let a = SemiringMatrix::from_fn(2, 2, |_, _| CountingSemiring::new(3));
        let b = SemiringMatrix::from_fn(2, 2, |_, _| CountingSemiring::new(4));
        let c = a.hadamard(&b);
        assert_eq!(*c.get(0, 0), CountingSemiring::new(12));
    }

    #[test]
    fn test_matrix_kronecker() {
        let a = SemiringMatrix::from_fn(2, 2, |i, j| {
            CountingSemiring::new((i * 2 + j + 1) as u64)
        });
        let b = SemiringMatrix::identity(2);
        let k = a.kronecker(&b);
        assert_eq!(k.rows(), 4);
        assert_eq!(k.cols(), 4);
    }

    #[test]
    fn test_matrix_trace() {
        let a = SemiringMatrix::from_fn(3, 3, |i, j| {
            if i == j {
                CountingSemiring::new(i as u64 + 1)
            } else {
                CountingSemiring::zero()
            }
        });
        assert_eq!(a.trace_sr(), CountingSemiring::new(6)); // 1 + 2 + 3
    }

    #[test]
    fn test_matrix_direct_sum() {
        let a = SemiringMatrix::identity(2);
        let b = SemiringMatrix::identity(3);
        let ds = a.direct_sum(&b);
        assert_eq!(ds.rows(), 5);
        assert_eq!(ds.cols(), 5);
        assert_eq!(*ds.get(0, 0), CountingSemiring::one());
        assert_eq!(*ds.get(0, 2), CountingSemiring::zero());
        assert_eq!(*ds.get(2, 2), CountingSemiring::one());
    }

    #[test]
    fn test_matrix_star_tropical() {
        // All-pairs shortest paths example
        let mut m = SemiringMatrix::<TropicalSemiring>::new(3, 3);
        m.set(0, 1, TropicalSemiring::new(1.0));
        m.set(1, 2, TropicalSemiring::new(2.0));
        m.set(0, 2, TropicalSemiring::new(5.0));

        let star = m.star_matrix(100);
        // Shortest path 0->2 should be 3.0 (via 1)
        assert!((star.get(0, 2).value - 3.0).abs() < 1e-10);
    }

    // --- Vector tests ---

    #[test]
    fn test_vector_dot() {
        let v1 = SemiringVector::from_vec(vec![
            CountingSemiring::new(1),
            CountingSemiring::new(2),
            CountingSemiring::new(3),
        ]);
        let v2 = SemiringVector::from_vec(vec![
            CountingSemiring::new(4),
            CountingSemiring::new(5),
            CountingSemiring::new(6),
        ]);
        // 1*4 + 2*5 + 3*6 = 32
        assert_eq!(v1.dot(&v2), CountingSemiring::new(32));
    }

    #[test]
    fn test_vector_outer_product() {
        let v1 = SemiringVector::from_vec(vec![
            CountingSemiring::new(1),
            CountingSemiring::new(2),
        ]);
        let v2 = SemiringVector::from_vec(vec![
            CountingSemiring::new(3),
            CountingSemiring::new(4),
        ]);
        let m = v1.outer_product(&v2);
        assert_eq!(*m.get(0, 0), CountingSemiring::new(3));
        assert_eq!(*m.get(1, 1), CountingSemiring::new(8));
    }

    // --- WFA tests ---

    #[test]
    fn test_wfa_basic() {
        // Simple 2-state automaton over {a, b} with counting semiring
        let mut wfa = WeightedFiniteAutomaton::<CountingSemiring>::new(2, 2);
        wfa.set_initial(0, CountingSemiring::one());
        wfa.set_final(1, CountingSemiring::one());
        wfa.set_transition(0, 0, 0, CountingSemiring::one()); // state 0, 'a', stay
        wfa.set_transition(0, 1, 1, CountingSemiring::one()); // state 0, 'b', go to 1
        wfa.set_transition(1, 1, 0, CountingSemiring::one()); // state 1, 'a', stay

        // word "b" = 1 path
        let w = wfa.weight_of_word(&[1]);
        assert_eq!(w, CountingSemiring::one());

        // word "ab" = 1 path (0 --a--> 0 --b--> 1)
        let w = wfa.weight_of_word(&[0, 1]);
        assert_eq!(w, CountingSemiring::one());

        // word "a" = 0 paths (can't reach final state 1 with just 'a')
        let w = wfa.weight_of_word(&[0]);
        assert_eq!(w, CountingSemiring::zero());
    }

    #[test]
    fn test_wfa_enumerate_words() {
        let mut wfa = WeightedFiniteAutomaton::<BooleanSemiring>::new(2, 2);
        wfa.set_initial(0, BooleanSemiring::one());
        wfa.set_final(1, BooleanSemiring::one());
        wfa.set_transition(0, 1, 0, BooleanSemiring::one()); // 'a' -> final
        wfa.set_transition(0, 0, 1, BooleanSemiring::one()); // 'b' -> loop

        let words = wfa.weight_of_all_words_up_to(3);
        assert!(words.iter().any(|(w, _)| *w == vec![0])); // "a" is accepted
        assert!(words.iter().any(|(w, _)| *w == vec![1, 0])); // "ba" is accepted
    }

    #[test]
    fn test_wfa_product() {
        let mut a = WeightedFiniteAutomaton::<BooleanSemiring>::new(2, 1);
        a.set_initial(0, BooleanSemiring::one());
        a.set_final(1, BooleanSemiring::one());
        a.set_transition(0, 1, 0, BooleanSemiring::one());

        let mut b = WeightedFiniteAutomaton::<BooleanSemiring>::new(2, 1);
        b.set_initial(0, BooleanSemiring::one());
        b.set_final(1, BooleanSemiring::one());
        b.set_transition(0, 1, 0, BooleanSemiring::one());

        let prod = a.product(&b);
        assert_eq!(prod.num_states, 4);
        let w = prod.weight_of_word(&[0]);
        assert_eq!(w, BooleanSemiring::one()); // both accept "a"
    }

    #[test]
    fn test_wfa_sum() {
        let mut a = WeightedFiniteAutomaton::<CountingSemiring>::new(1, 1);
        a.set_initial(0, CountingSemiring::one());
        a.set_final(0, CountingSemiring::one());

        let mut b = WeightedFiniteAutomaton::<CountingSemiring>::new(1, 1);
        b.set_initial(0, CountingSemiring::one());
        b.set_final(0, CountingSemiring::one());

        let sum = a.sum(&b);
        assert_eq!(sum.num_states, 2);
        // Empty word should have weight 2 (accepted by both)
        let w = sum.weight_of_word(&[]);
        assert_eq!(w, CountingSemiring::new(2));
    }

    #[test]
    fn test_wfa_minimize() {
        // Two equivalent states
        let mut wfa = WeightedFiniteAutomaton::<BooleanSemiring>::new(3, 1);
        wfa.set_initial(0, BooleanSemiring::one());
        wfa.set_final(1, BooleanSemiring::one());
        wfa.set_final(2, BooleanSemiring::one());
        wfa.set_transition(0, 1, 0, BooleanSemiring::one());
        wfa.set_transition(0, 2, 0, BooleanSemiring::one()); // 1 and 2 look the same

        let min = wfa.minimize();
        assert!(min.num_states <= wfa.num_states);
    }
}
