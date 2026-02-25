//! Semiring types for weighted finite automata.
//!
//! Provides a comprehensive library of semiring implementations used in WFA-based
//! scoring circuits. Includes tropical, boolean, counting, log, Viterbi, expectation,
//! and finite-field semirings, along with matrix and polynomial abstractions over
//! arbitrary semirings, and structure-preserving homomorphisms.

use std::fmt;
use std::hash::Hash;
use std::marker::PhantomData;
use std::ops;

use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};
use thiserror::Error;

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Errors that can arise during semiring operations.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum SemiringError {
    #[error("matrix dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: String, got: String },

    #[error("matrix must be square for this operation (got {rows}x{cols})")]
    NotSquare { rows: usize, cols: usize },

    #[error("index out of bounds: ({row}, {col}) in {rows}x{cols} matrix")]
    IndexOutOfBounds {
        row: usize,
        col: usize,
        rows: usize,
        cols: usize,
    },

    #[error("division by zero in field arithmetic")]
    DivisionByZero,

    #[error("element has no multiplicative inverse")]
    NoInverse,

    #[error("polynomial evaluation error: {0}")]
    PolynomialError(String),
}

// ---------------------------------------------------------------------------
// Core traits
// ---------------------------------------------------------------------------

/// A semiring (S, ⊕, ⊗, 0, 1) where ⊕ is commutative and associative, ⊗ is
/// associative, 0 is the identity for ⊕, 1 is the identity for ⊗, 0 annihilates
/// under ⊗, and ⊗ distributes over ⊕.
pub trait Semiring: Clone + fmt::Debug + PartialEq + Sized + Send + Sync {
    /// The additive identity.
    fn zero() -> Self;

    /// The multiplicative identity.
    fn one() -> Self;

    /// Semiring addition (⊕).
    fn add(&self, other: &Self) -> Self;

    /// Semiring multiplication (⊗).
    fn mul(&self, other: &Self) -> Self;

    /// In-place addition.
    fn add_assign(&mut self, other: &Self) {
        *self = self.add(other);
    }

    /// In-place multiplication.
    fn mul_assign(&mut self, other: &Self) {
        *self = self.mul(other);
    }

    /// Check if this element is the additive identity.
    fn is_zero(&self) -> bool {
        *self == Self::zero()
    }

    /// Check if this element is the multiplicative identity.
    fn is_one(&self) -> bool {
        *self == Self::one()
    }

    /// Display the element in a human-readable form.
    fn display(&self) -> String {
        format!("{:?}", self)
    }

    /// Compute self ⊗ self ⊗ … ⊗ self (n times) via repeated squaring.
    fn pow(&self, mut n: u64) -> Self {
        if n == 0 {
            return Self::one();
        }
        let mut base = self.clone();
        let mut result = Self::one();
        while n > 1 {
            if n % 2 == 1 {
                result = result.mul(&base);
            }
            base = base.mul(&base);
            n /= 2;
        }
        result.mul(&base)
    }
}

/// A star semiring extends a semiring with a Kleene closure operator a* such
/// that a* = 1 ⊕ a ⊗ a*.
pub trait StarSemiring: Semiring {
    /// Kleene closure: a* = 1 ⊕ a ⊗ a*
    fn star(&self) -> Self;

    /// Kleene plus: a⁺ = a ⊗ a*
    fn plus(&self) -> Self {
        self.mul(&self.star())
    }
}

// ---------------------------------------------------------------------------
// CountingSemiring  (ℕ, +, ·)
// ---------------------------------------------------------------------------

/// The counting semiring (ℕ, +, ·) counts the number of accepting paths.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct CountingSemiring {
    pub value: u64,
}

impl CountingSemiring {
    pub fn new(value: u64) -> Self {
        Self { value }
    }
}

impl fmt::Display for CountingSemiring {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value)
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

    fn is_zero(&self) -> bool {
        self.value == 0
    }

    fn is_one(&self) -> bool {
        self.value == 1
    }

    fn display(&self) -> String {
        format!("{}", self.value)
    }
}

// ---------------------------------------------------------------------------
// BooleanSemiring  ({0,1}, ∨, ∧)
// ---------------------------------------------------------------------------

/// The Boolean semiring ({false, true}, ∨, ∧). Addition is OR, multiplication
/// is AND. Used for reachability / language membership.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct BooleanSemiring {
    pub value: bool,
}

impl BooleanSemiring {
    pub fn new(value: bool) -> Self {
        Self { value }
    }
}

impl fmt::Display for BooleanSemiring {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", if self.value { "1" } else { "0" })
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

    fn is_zero(&self) -> bool {
        !self.value
    }

    fn is_one(&self) -> bool {
        self.value
    }

    fn display(&self) -> String {
        if self.value {
            "1".to_string()
        } else {
            "0".to_string()
        }
    }
}

impl StarSemiring for BooleanSemiring {
    fn star(&self) -> Self {
        // In the Boolean semiring a* = 1 for all a.
        Self::one()
    }
}

// ---------------------------------------------------------------------------
// TropicalSemiring  (ℝ ∪ {+∞}, min, +)
// ---------------------------------------------------------------------------

/// The tropical semiring (ℝ ∪ {+∞}, min, +). Used for shortest-path / edit
/// distance computations.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TropicalSemiring {
    pub value: OrderedFloat<f64>,
}

impl TropicalSemiring {
    pub fn new(value: f64) -> Self {
        Self {
            value: OrderedFloat(value),
        }
    }

    pub fn infinity() -> Self {
        Self {
            value: OrderedFloat(f64::INFINITY),
        }
    }

    pub fn raw(&self) -> f64 {
        self.value.into_inner()
    }

    pub fn from_value(value: f64) -> Self {
        Self::new(value)
    }
}

impl fmt::Display for TropicalSemiring {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.value.into_inner().is_infinite() && self.value.into_inner() > 0.0 {
            write!(f, "∞")
        } else {
            write!(f, "{}", self.value)
        }
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
        Self {
            value: self.value.min(other.value),
        }
    }

    fn mul(&self, other: &Self) -> Self {
        Self {
            value: OrderedFloat(self.value.into_inner() + other.value.into_inner()),
        }
    }

    fn is_zero(&self) -> bool {
        self.value.into_inner().is_infinite() && self.value.into_inner() > 0.0
    }

    fn is_one(&self) -> bool {
        self.value.into_inner() == 0.0
    }

    fn display(&self) -> String {
        format!("{}", self)
    }
}

impl StarSemiring for TropicalSemiring {
    fn star(&self) -> Self {
        // a* = 0 if a >= 0 (since min of 0 and anything >=0 is 0)
        // If a < 0 the star diverges; we clamp to -∞ to indicate.
        if self.value.into_inner() >= 0.0 {
            Self::one()
        } else {
            Self::new(f64::NEG_INFINITY)
        }
    }
}

// ---------------------------------------------------------------------------
// BoundedCountingSemiring
// ---------------------------------------------------------------------------

/// A counting semiring with an upper bound. Values are clamped to [0, bound].
/// Useful for BLEU-style n-gram clipping where matches are capped at a
/// reference count.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BoundedCountingSemiring {
    pub value: u64,
    pub bound: u64,
}

impl BoundedCountingSemiring {
    pub fn new(value: u64, bound: u64) -> Self {
        Self {
            value: value.min(bound),
            bound,
        }
    }

    fn clamp(&self, v: u64) -> u64 {
        v.min(self.bound)
    }
}

impl fmt::Display for BoundedCountingSemiring {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}(≤{})", self.value, self.bound)
    }
}

impl Semiring for BoundedCountingSemiring {
    fn zero() -> Self {
        Self {
            value: 0,
            bound: u64::MAX,
        }
    }

    fn one() -> Self {
        Self {
            value: 1,
            bound: u64::MAX,
        }
    }

    fn add(&self, other: &Self) -> Self {
        let bound = self.bound.min(other.bound);
        Self {
            value: self.value.saturating_add(other.value).min(bound),
            bound,
        }
    }

    fn mul(&self, other: &Self) -> Self {
        let bound = self.bound.min(other.bound);
        Self {
            value: self.value.saturating_mul(other.value).min(bound),
            bound,
        }
    }

    fn is_zero(&self) -> bool {
        self.value == 0
    }

    fn is_one(&self) -> bool {
        self.value == 1
    }

    fn display(&self) -> String {
        format!("{}", self)
    }
}

// ---------------------------------------------------------------------------
// GoldilocksField  (F_p where p = 2^64 - 2^32 + 1)
// ---------------------------------------------------------------------------

/// The Goldilocks prime: p = 2^64 − 2^32 + 1
pub const GOLDILOCKS_PRIME: u64 = 0xFFFF_FFFF_0000_0001;

/// A field element in the Goldilocks prime field F_p, p = 2^64 − 2^32 + 1.
/// This prime is popular in zero-knowledge proof systems (Plonky2, etc.)
/// because its structure allows efficient modular reduction.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, Copy)]
pub struct GoldilocksField {
    /// The canonical representative in [0, p).
    pub value: u64,
}

impl GoldilocksField {
    /// Create a new field element, reducing modulo p.
    pub fn new(value: u64) -> Self {
        Self {
            value: Self::reduce(value as u128),
        }
    }

    /// Create from a value already known to be in [0, p).
    pub fn from_canonical(value: u64) -> Self {
        debug_assert!(value < GOLDILOCKS_PRIME);
        Self { value }
    }

    /// Reduce a u128 modulo the Goldilocks prime.
    /// p = 2^64 - 2^32 + 1
    /// We use the identity: 2^64 ≡ 2^32 - 1 (mod p).
    fn reduce(x: u128) -> u64 {
        let p = GOLDILOCKS_PRIME as u128;
        (x % p) as u64
    }

    /// Modular addition without overflow.
    fn add_mod(a: u64, b: u64) -> u64 {
        let sum = (a as u128) + (b as u128);
        let p = GOLDILOCKS_PRIME as u128;
        (sum % p) as u64
    }

    /// Modular subtraction.
    fn sub_mod(a: u64, b: u64) -> u64 {
        if a >= b {
            let diff = a - b;
            if diff >= GOLDILOCKS_PRIME {
                diff - GOLDILOCKS_PRIME
            } else {
                diff
            }
        } else {
            // a - b + p
            let p = GOLDILOCKS_PRIME as u128;
            let result = (a as u128) + p - (b as u128);
            (result % p) as u64
        }
    }

    /// Modular multiplication using u128.
    fn mul_mod(a: u64, b: u64) -> u64 {
        let prod = (a as u128) * (b as u128);
        let p = GOLDILOCKS_PRIME as u128;
        (prod % p) as u64
    }

    /// Modular exponentiation via repeated squaring.
    pub fn pow_mod(mut base: u64, mut exp: u64) -> u64 {
        let mut result: u64 = 1;
        base = Self::reduce(base as u128);
        while exp > 0 {
            if exp & 1 == 1 {
                result = Self::mul_mod(result, base);
            }
            exp >>= 1;
            if exp > 0 {
                base = Self::mul_mod(base, base);
            }
        }
        result
    }

    /// Multiplicative inverse using Fermat's little theorem: a^{-1} = a^{p-2} mod p.
    pub fn inv(&self) -> Result<Self, SemiringError> {
        if self.value == 0 {
            return Err(SemiringError::DivisionByZero);
        }
        Ok(Self {
            value: Self::pow_mod(self.value, GOLDILOCKS_PRIME - 2),
        })
    }

    /// Subtraction in the field.
    pub fn sub(&self, other: &Self) -> Self {
        Self {
            value: Self::sub_mod(self.value, other.value),
        }
    }

    /// Division in the field.
    pub fn div(&self, other: &Self) -> Result<Self, SemiringError> {
        let inv = other.inv()?;
        Ok(self.mul(&inv))
    }

    /// Negate: -a = p - a.
    pub fn neg(&self) -> Self {
        if self.value == 0 {
            Self { value: 0 }
        } else {
            Self {
                value: GOLDILOCKS_PRIME - self.value,
            }
        }
    }

    /// Field exponentiation.
    pub fn field_pow(&self, exp: u64) -> Self {
        Self {
            value: Self::pow_mod(self.value, exp),
        }
    }

    /// Check if this element is a quadratic residue (has a square root).
    pub fn is_quadratic_residue(&self) -> bool {
        if self.value == 0 {
            return true;
        }
        // Euler's criterion: a^((p-1)/2) == 1 mod p
        let exp = (GOLDILOCKS_PRIME - 1) / 2;
        Self::pow_mod(self.value, exp) == 1
    }

    /// Compute square root if it exists (Tonelli-Shanks).
    pub fn sqrt(&self) -> Option<Self> {
        if self.value == 0 {
            return Some(Self { value: 0 });
        }
        if !self.is_quadratic_residue() {
            return None;
        }

        let p = GOLDILOCKS_PRIME;
        // p - 1 = 2^s * q where q is odd
        let mut s: u32 = 0;
        let mut q = p - 1;
        while q % 2 == 0 {
            s += 1;
            q /= 2;
        }

        if s == 1 {
            // p ≡ 3 (mod 4)
            let r = Self::pow_mod(self.value, (p + 1) / 4);
            return Some(Self { value: r });
        }

        // Find a non-residue
        let mut z: u64 = 2;
        while {
            let exp = (p - 1) / 2;
            Self::pow_mod(z, exp) != p - 1
        } {
            z += 1;
        }

        let mut m = s;
        let mut c = Self::pow_mod(z, q);
        let mut t = Self::pow_mod(self.value, q);
        let mut r = Self::pow_mod(self.value, (q + 1) / 2);

        loop {
            if t == 0 {
                return Some(Self { value: 0 });
            }
            if t == 1 {
                return Some(Self { value: r });
            }

            // Find the least i such that t^(2^i) = 1
            let mut i: u32 = 1;
            let mut tmp = Self::mul_mod(t, t);
            while tmp != 1 {
                tmp = Self::mul_mod(tmp, tmp);
                i += 1;
                if i >= m {
                    return None;
                }
            }

            let b = Self::pow_mod(c, 1u64 << (m - i - 1));
            m = i;
            c = Self::mul_mod(b, b);
            t = Self::mul_mod(t, c);
            r = Self::mul_mod(r, b);
        }
    }

    /// Batch inversion using Montgomery's trick.
    /// Computes inverses of all elements in a slice, returning an error if any is zero.
    pub fn batch_inverse(elements: &[Self]) -> Result<Vec<Self>, SemiringError> {
        let n = elements.len();
        if n == 0 {
            return Ok(vec![]);
        }

        // Compute prefix products
        let mut prefix = Vec::with_capacity(n);
        prefix.push(elements[0]);
        for i in 1..n {
            if elements[i].value == 0 {
                return Err(SemiringError::DivisionByZero);
            }
            prefix.push(prefix[i - 1].mul(&elements[i]));
        }

        // Invert the total product
        let mut inv_total = prefix[n - 1].inv()?;

        // Back-propagate
        let mut result = vec![Self { value: 0 }; n];
        for i in (1..n).rev() {
            result[i] = Self {
                value: Self::mul_mod(inv_total.value, prefix[i - 1].value),
            };
            inv_total = Self {
                value: Self::mul_mod(inv_total.value, elements[i].value),
            };
        }
        result[0] = inv_total;

        Ok(result)
    }
}

impl fmt::Display for GoldilocksField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

impl Semiring for GoldilocksField {
    fn zero() -> Self {
        Self { value: 0 }
    }

    fn one() -> Self {
        Self { value: 1 }
    }

    fn add(&self, other: &Self) -> Self {
        Self {
            value: Self::add_mod(self.value, other.value),
        }
    }

    fn mul(&self, other: &Self) -> Self {
        Self {
            value: Self::mul_mod(self.value, other.value),
        }
    }

    fn is_zero(&self) -> bool {
        self.value == 0
    }

    fn is_one(&self) -> bool {
        self.value == 1
    }

    fn display(&self) -> String {
        format!("{}", self.value)
    }
}

// ---------------------------------------------------------------------------
// FieldSemiring<P> — generic prime field wrapper
// ---------------------------------------------------------------------------

/// Trait to supply a prime modulus at the type level.
pub trait PrimeModulus: Clone + fmt::Debug + PartialEq + Eq + Send + Sync + 'static {
    fn modulus() -> u64;
}

/// A small prime field F_p parameterized by a `PrimeModulus` trait.
#[derive(Clone, Debug, Eq, Serialize, Deserialize)]
pub struct FieldSemiring<P: PrimeModulus> {
    pub value: u64,
    #[serde(skip)]
    _phantom: PhantomData<P>,
}

impl<P: PrimeModulus> PartialEq for FieldSemiring<P> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<P: PrimeModulus> Hash for FieldSemiring<P> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.value.hash(state);
    }
}

impl<P: PrimeModulus> FieldSemiring<P> {
    pub fn new(value: u64) -> Self {
        Self {
            value: value % P::modulus(),
            _phantom: PhantomData,
        }
    }

    pub fn inv(&self) -> Result<Self, SemiringError> {
        if self.value == 0 {
            return Err(SemiringError::DivisionByZero);
        }
        let p = P::modulus();
        // Fermat's little theorem: a^{p-2} mod p
        let mut result: u64 = 1;
        let mut base = self.value;
        let mut exp = p - 2;
        while exp > 0 {
            if exp & 1 == 1 {
                result = ((result as u128 * base as u128) % p as u128) as u64;
            }
            exp >>= 1;
            if exp > 0 {
                base = ((base as u128 * base as u128) % p as u128) as u64;
            }
        }
        Ok(Self {
            value: result,
            _phantom: PhantomData,
        })
    }

    pub fn sub(&self, other: &Self) -> Self {
        let p = P::modulus();
        let value = if self.value >= other.value {
            self.value - other.value
        } else {
            p - (other.value - self.value)
        };
        Self {
            value,
            _phantom: PhantomData,
        }
    }

    pub fn neg(&self) -> Self {
        if self.value == 0 {
            self.clone()
        } else {
            Self {
                value: P::modulus() - self.value,
                _phantom: PhantomData,
            }
        }
    }
}

impl<P: PrimeModulus> fmt::Display for FieldSemiring<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} (mod {})", self.value, P::modulus())
    }
}

impl<P: PrimeModulus> Semiring for FieldSemiring<P> {
    fn zero() -> Self {
        Self {
            value: 0,
            _phantom: PhantomData,
        }
    }

    fn one() -> Self {
        Self {
            value: 1,
            _phantom: PhantomData,
        }
    }

    fn add(&self, other: &Self) -> Self {
        let p = P::modulus() as u128;
        let sum = (self.value as u128 + other.value as u128) % p;
        Self {
            value: sum as u64,
            _phantom: PhantomData,
        }
    }

    fn mul(&self, other: &Self) -> Self {
        let p = P::modulus() as u128;
        let prod = (self.value as u128 * other.value as u128) % p;
        Self {
            value: prod as u64,
            _phantom: PhantomData,
        }
    }

    fn is_zero(&self) -> bool {
        self.value == 0
    }

    fn is_one(&self) -> bool {
        self.value == 1
    }

    fn display(&self) -> String {
        format!("{}", self)
    }
}

/// Example: the Mersenne prime 2^31 - 1.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MersennePrime31;
impl PrimeModulus for MersennePrime31 {
    fn modulus() -> u64 {
        (1u64 << 31) - 1
    }
}

/// Example: BN254 scalar field characteristic (a small stand-in for testing).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SmallTestPrime;
impl PrimeModulus for SmallTestPrime {
    fn modulus() -> u64 {
        1000000007
    }
}

// ---------------------------------------------------------------------------
// RealSemiring  (ℝ, +, ·)
// ---------------------------------------------------------------------------

/// The real semiring (ℝ, +, ·) with IEEE 754 doubles.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct RealSemiring {
    pub value: OrderedFloat<f64>,
}

impl RealSemiring {
    pub fn new(value: f64) -> Self {
        Self {
            value: OrderedFloat(value),
        }
    }

    pub fn raw(&self) -> f64 {
        self.value.into_inner()
    }
}

impl fmt::Display for RealSemiring {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

impl Semiring for RealSemiring {
    fn zero() -> Self {
        Self::new(0.0)
    }

    fn one() -> Self {
        Self::new(1.0)
    }

    fn add(&self, other: &Self) -> Self {
        Self::new(self.value.into_inner() + other.value.into_inner())
    }

    fn mul(&self, other: &Self) -> Self {
        Self::new(self.value.into_inner() * other.value.into_inner())
    }

    fn is_zero(&self) -> bool {
        self.value.into_inner() == 0.0
    }

    fn is_one(&self) -> bool {
        self.value.into_inner() == 1.0
    }

    fn display(&self) -> String {
        format!("{}", self.value)
    }
}

impl StarSemiring for RealSemiring {
    fn star(&self) -> Self {
        let v = self.value.into_inner();
        if v.abs() < 1.0 {
            Self::new(1.0 / (1.0 - v))
        } else {
            // Divergent; return infinity.
            Self::new(f64::INFINITY)
        }
    }
}

// ---------------------------------------------------------------------------
// LogSemiring  (ℝ ∪ {-∞}, ⊕_log, +)
// ---------------------------------------------------------------------------

/// The log semiring (ℝ ∪ {−∞}, log-add, +) used in probabilistic models.
///
/// ⊕: log(exp(a) + exp(b))  implemented via log-sum-exp for numerical stability
/// ⊗: a + b
/// 0: −∞
/// 1: 0
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LogSemiring {
    pub value: OrderedFloat<f64>,
}

impl LogSemiring {
    pub fn new(value: f64) -> Self {
        Self {
            value: OrderedFloat(value),
        }
    }

    pub fn neg_infinity() -> Self {
        Self::new(f64::NEG_INFINITY)
    }

    pub fn raw(&self) -> f64 {
        self.value.into_inner()
    }

    /// Numerically stable log-sum-exp: log(exp(a) + exp(b)).
    fn log_add(a: f64, b: f64) -> f64 {
        if a == f64::NEG_INFINITY {
            return b;
        }
        if b == f64::NEG_INFINITY {
            return a;
        }
        let max = a.max(b);
        let min = a.min(b);
        max + (1.0 + (min - max).exp()).ln()
    }
}

impl fmt::Display for LogSemiring {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let v = self.value.into_inner();
        if v == f64::NEG_INFINITY {
            write!(f, "-∞")
        } else {
            write!(f, "{:.6}", v)
        }
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
        Self::new(Self::log_add(
            self.value.into_inner(),
            other.value.into_inner(),
        ))
    }

    fn mul(&self, other: &Self) -> Self {
        Self::new(self.value.into_inner() + other.value.into_inner())
    }

    fn is_zero(&self) -> bool {
        self.value.into_inner() == f64::NEG_INFINITY
    }

    fn is_one(&self) -> bool {
        self.value.into_inner() == 0.0
    }

    fn display(&self) -> String {
        format!("{}", self)
    }
}

impl StarSemiring for LogSemiring {
    fn star(&self) -> Self {
        let v = self.value.into_inner();
        if v < 0.0 {
            // a* = -log(1 - exp(a)) when a < 0  (convergent geometric series in log domain)
            let exp_v = v.exp();
            if exp_v < 1.0 {
                Self::new(-(1.0 - exp_v).ln())
            } else {
                Self::new(f64::INFINITY)
            }
        } else {
            Self::new(f64::INFINITY)
        }
    }
}

// ---------------------------------------------------------------------------
// MaxPlusSemiring  (ℝ ∪ {-∞}, max, +)
// ---------------------------------------------------------------------------

/// The max-plus semiring (ℝ ∪ {−∞}, max, +), also called the "schedule algebra".
/// Used in dynamic programming, scheduling, and longest-path problems.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MaxPlusSemiring {
    pub value: OrderedFloat<f64>,
}

impl MaxPlusSemiring {
    pub fn new(value: f64) -> Self {
        Self {
            value: OrderedFloat(value),
        }
    }

    pub fn neg_infinity() -> Self {
        Self::new(f64::NEG_INFINITY)
    }

    pub fn raw(&self) -> f64 {
        self.value.into_inner()
    }
}

impl fmt::Display for MaxPlusSemiring {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let v = self.value.into_inner();
        if v == f64::NEG_INFINITY {
            write!(f, "-∞")
        } else {
            write!(f, "{}", v)
        }
    }
}

impl Semiring for MaxPlusSemiring {
    fn zero() -> Self {
        Self::neg_infinity()
    }

    fn one() -> Self {
        Self::new(0.0)
    }

    fn add(&self, other: &Self) -> Self {
        Self {
            value: self.value.max(other.value),
        }
    }

    fn mul(&self, other: &Self) -> Self {
        let a = self.value.into_inner();
        let b = other.value.into_inner();
        if a == f64::NEG_INFINITY || b == f64::NEG_INFINITY {
            Self::neg_infinity()
        } else {
            Self::new(a + b)
        }
    }

    fn is_zero(&self) -> bool {
        self.value.into_inner() == f64::NEG_INFINITY
    }

    fn is_one(&self) -> bool {
        self.value.into_inner() == 0.0
    }

    fn display(&self) -> String {
        format!("{}", self)
    }
}

impl StarSemiring for MaxPlusSemiring {
    fn star(&self) -> Self {
        // a* = 0 (the multiplicative identity) when a <= 0
        // Otherwise the sequence diverges; we return +∞.
        if self.value.into_inner() <= 0.0 {
            Self::one()
        } else {
            Self::new(f64::INFINITY)
        }
    }
}

// ---------------------------------------------------------------------------
// MinMaxSemiring  (ℝ, min, max)
// ---------------------------------------------------------------------------

/// The min-max semiring (ℝ, min, max) — a bounded distributive lattice.
/// ⊕ = min, ⊗ = max, 0 = +∞, 1 = −∞.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MinMaxSemiring {
    pub value: OrderedFloat<f64>,
}

impl MinMaxSemiring {
    pub fn new(value: f64) -> Self {
        Self {
            value: OrderedFloat(value),
        }
    }

    pub fn raw(&self) -> f64 {
        self.value.into_inner()
    }
}

impl fmt::Display for MinMaxSemiring {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let v = self.value.into_inner();
        if v == f64::INFINITY {
            write!(f, "+∞")
        } else if v == f64::NEG_INFINITY {
            write!(f, "-∞")
        } else {
            write!(f, "{}", v)
        }
    }
}

impl Semiring for MinMaxSemiring {
    fn zero() -> Self {
        Self::new(f64::INFINITY)
    }

    fn one() -> Self {
        Self::new(f64::NEG_INFINITY)
    }

    fn add(&self, other: &Self) -> Self {
        Self {
            value: self.value.min(other.value),
        }
    }

    fn mul(&self, other: &Self) -> Self {
        Self {
            value: self.value.max(other.value),
        }
    }

    fn is_zero(&self) -> bool {
        self.value.into_inner() == f64::INFINITY
    }

    fn is_one(&self) -> bool {
        self.value.into_inner() == f64::NEG_INFINITY
    }

    fn display(&self) -> String {
        format!("{}", self)
    }
}

impl StarSemiring for MinMaxSemiring {
    fn star(&self) -> Self {
        // In the min-max semiring, a* = 1 = −∞ for all a (since max with −∞
        // is a, and min collapses the series to −∞).
        Self::one()
    }
}

// ---------------------------------------------------------------------------
// ViterbiSemiring  ([0,1], max, ·)
// ---------------------------------------------------------------------------

/// The Viterbi semiring ([0, 1], max, ·) used for finding the most probable
/// path in a probabilistic automaton.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ViterbiSemiring {
    pub value: OrderedFloat<f64>,
}

impl ViterbiSemiring {
    pub fn new(value: f64) -> Self {
        debug_assert!(
            value >= 0.0,
            "ViterbiSemiring values must be non-negative"
        );
        Self {
            value: OrderedFloat(value),
        }
    }

    pub fn raw(&self) -> f64 {
        self.value.into_inner()
    }
}

impl fmt::Display for ViterbiSemiring {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.6}", self.value)
    }
}

impl Semiring for ViterbiSemiring {
    fn zero() -> Self {
        Self::new(0.0)
    }

    fn one() -> Self {
        Self::new(1.0)
    }

    fn add(&self, other: &Self) -> Self {
        Self {
            value: self.value.max(other.value),
        }
    }

    fn mul(&self, other: &Self) -> Self {
        Self::new(self.value.into_inner() * other.value.into_inner())
    }

    fn is_zero(&self) -> bool {
        self.value.into_inner() == 0.0
    }

    fn is_one(&self) -> bool {
        self.value.into_inner() == 1.0
    }

    fn display(&self) -> String {
        format!("{}", self)
    }
}

impl StarSemiring for ViterbiSemiring {
    fn star(&self) -> Self {
        // In the Viterbi semiring a* = 1 for all a in [0,1] since
        // max(1, a, a^2, ...) = 1.
        Self::one()
    }
}

// ---------------------------------------------------------------------------
// ExpectationSemiring<S>
// ---------------------------------------------------------------------------

/// The expectation semiring pairs a probability (value) with an expected cost
/// (expectation). Used for computing expected values through WFA composition.
///
/// (v₁, e₁) ⊕ (v₂, e₂) = (v₁ ⊕ v₂, e₁ ⊕ e₂)
/// (v₁, e₁) ⊗ (v₂, e₂) = (v₁ ⊗ v₂, v₁ ⊗ e₂ ⊕ e₁ ⊗ v₂)
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ExpectationSemiring<S: Semiring> {
    pub value: S,
    pub expectation: S,
}

impl<S: Semiring> ExpectationSemiring<S> {
    pub fn new(value: S, expectation: S) -> Self {
        Self { value, expectation }
    }

    pub fn from_value(value: S) -> Self {
        Self {
            expectation: S::zero(),
            value,
        }
    }

    pub fn from_expectation(expectation: S) -> Self {
        Self {
            value: S::zero(),
            expectation,
        }
    }
}

impl<S: Semiring + fmt::Display> fmt::Display for ExpectationSemiring<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.value, self.expectation)
    }
}

impl<S: Semiring + Eq + Hash> Semiring for ExpectationSemiring<S> {
    fn zero() -> Self {
        Self {
            value: S::zero(),
            expectation: S::zero(),
        }
    }

    fn one() -> Self {
        Self {
            value: S::one(),
            expectation: S::zero(),
        }
    }

    fn add(&self, other: &Self) -> Self {
        Self {
            value: self.value.add(&other.value),
            expectation: self.expectation.add(&other.expectation),
        }
    }

    fn mul(&self, other: &Self) -> Self {
        // (v1, e1) ⊗ (v2, e2) = (v1⊗v2, v1⊗e2 ⊕ e1⊗v2)
        Self {
            value: self.value.mul(&other.value),
            expectation: self
                .value
                .mul(&other.expectation)
                .add(&self.expectation.mul(&other.value)),
        }
    }

    fn is_zero(&self) -> bool {
        self.value.is_zero() && self.expectation.is_zero()
    }

    fn is_one(&self) -> bool {
        self.value.is_one() && self.expectation.is_zero()
    }

    fn display(&self) -> String {
        format!("({}, {})", self.value.display(), self.expectation.display())
    }
}

// ---------------------------------------------------------------------------
// ProductSemiring<A, B>
// ---------------------------------------------------------------------------

/// The direct product of two semirings, operating component-wise.
/// (a₁, b₁) ⊕ (a₂, b₂) = (a₁ ⊕ a₂, b₁ ⊕ b₂)
/// (a₁, b₁) ⊗ (a₂, b₂) = (a₁ ⊗ a₂, b₁ ⊗ b₂)
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProductSemiring<A: Semiring, B: Semiring> {
    pub first: A,
    pub second: B,
}

impl<A: Semiring, B: Semiring> ProductSemiring<A, B> {
    pub fn new(first: A, second: B) -> Self {
        Self { first, second }
    }
}

impl<A: Semiring + fmt::Display, B: Semiring + fmt::Display> fmt::Display
    for ProductSemiring<A, B>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.first, self.second)
    }
}

impl<A: Semiring + Eq + Hash, B: Semiring + Eq + Hash> Semiring for ProductSemiring<A, B> {
    fn zero() -> Self {
        Self {
            first: A::zero(),
            second: B::zero(),
        }
    }

    fn one() -> Self {
        Self {
            first: A::one(),
            second: B::one(),
        }
    }

    fn add(&self, other: &Self) -> Self {
        Self {
            first: self.first.add(&other.first),
            second: self.second.add(&other.second),
        }
    }

    fn mul(&self, other: &Self) -> Self {
        Self {
            first: self.first.mul(&other.first),
            second: self.second.mul(&other.second),
        }
    }

    fn is_zero(&self) -> bool {
        self.first.is_zero() && self.second.is_zero()
    }

    fn is_one(&self) -> bool {
        self.first.is_one() && self.second.is_one()
    }

    fn display(&self) -> String {
        format!("({}, {})", self.first.display(), self.second.display())
    }
}

impl<A: StarSemiring + Eq + Hash, B: StarSemiring + Eq + Hash> StarSemiring
    for ProductSemiring<A, B>
{
    fn star(&self) -> Self {
        Self {
            first: self.first.star(),
            second: self.second.star(),
        }
    }
}

// ---------------------------------------------------------------------------
// FreeMonoidSemiring  (strings over an alphabet, union, concatenation)
// ---------------------------------------------------------------------------

/// The free monoid semiring over strings. Addition is set union (collecting
/// into a sorted/deduped set of strings) and multiplication is pairwise
/// concatenation.
///
/// This is used for computing the set of all strings accepted by a WFA.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FreeMonoidSemiring {
    /// A sorted, deduplicated set of strings.
    pub strings: Vec<String>,
}

impl FreeMonoidSemiring {
    pub fn new(strings: Vec<String>) -> Self {
        let mut strings = strings;
        strings.sort();
        strings.dedup();
        Self { strings }
    }

    pub fn singleton(s: &str) -> Self {
        Self {
            strings: vec![s.to_string()],
        }
    }

    pub fn empty_string() -> Self {
        Self {
            strings: vec![String::new()],
        }
    }

    fn merge_sorted(a: &[String], b: &[String]) -> Vec<String> {
        let mut result = Vec::with_capacity(a.len() + b.len());
        let (mut i, mut j) = (0, 0);
        while i < a.len() && j < b.len() {
            match a[i].cmp(&b[j]) {
                std::cmp::Ordering::Less => {
                    result.push(a[i].clone());
                    i += 1;
                }
                std::cmp::Ordering::Greater => {
                    result.push(b[j].clone());
                    j += 1;
                }
                std::cmp::Ordering::Equal => {
                    result.push(a[i].clone());
                    i += 1;
                    j += 1;
                }
            }
        }
        while i < a.len() {
            result.push(a[i].clone());
            i += 1;
        }
        while j < b.len() {
            result.push(b[j].clone());
            j += 1;
        }
        result
    }
}

impl fmt::Display for FreeMonoidSemiring {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{")?;
        for (i, s) in self.strings.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            if s.is_empty() {
                write!(f, "ε")?;
            } else {
                write!(f, "\"{}\"", s)?;
            }
        }
        write!(f, "}}")
    }
}

impl Semiring for FreeMonoidSemiring {
    fn zero() -> Self {
        Self {
            strings: Vec::new(),
        }
    }

    fn one() -> Self {
        Self::empty_string()
    }

    fn add(&self, other: &Self) -> Self {
        Self {
            strings: Self::merge_sorted(&self.strings, &other.strings),
        }
    }

    fn mul(&self, other: &Self) -> Self {
        if self.strings.is_empty() || other.strings.is_empty() {
            return Self::zero();
        }
        let mut result = Vec::with_capacity(self.strings.len() * other.strings.len());
        for a in &self.strings {
            for b in &other.strings {
                let mut s = a.clone();
                s.push_str(b);
                result.push(s);
            }
        }
        result.sort();
        result.dedup();
        Self { strings: result }
    }

    fn is_zero(&self) -> bool {
        self.strings.is_empty()
    }

    fn is_one(&self) -> bool {
        self.strings.len() == 1 && self.strings[0].is_empty()
    }

    fn display(&self) -> String {
        format!("{}", self)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SemiringMatrix<S>
// ═══════════════════════════════════════════════════════════════════════════

/// A dense matrix over an arbitrary semiring, stored in row-major order.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SemiringMatrix<S: Semiring> {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<S>,
}

impl<S: Semiring> SemiringMatrix<S> {
    // -- Construction -------------------------------------------------------

    /// Create a matrix of zeros.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![S::zero(); rows * cols],
        }
    }

    /// Create an identity matrix (requires square dimensions).
    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m.data[i * n + i] = S::one();
        }
        m
    }

    /// Create a matrix from a flat row-major vector.
    pub fn from_vec(rows: usize, cols: usize, data: Vec<S>) -> Result<Self, SemiringError> {
        if data.len() != rows * cols {
            return Err(SemiringError::DimensionMismatch {
                expected: format!("{}x{} = {} elements", rows, cols, rows * cols),
                got: format!("{} elements", data.len()),
            });
        }
        Ok(Self { rows, cols, data })
    }

    /// Create a matrix from a 2D vector (vector of rows).
    pub fn from_rows(rows_data: Vec<Vec<S>>) -> Result<Self, SemiringError> {
        if rows_data.is_empty() {
            return Ok(Self::zeros(0, 0));
        }
        let rows = rows_data.len();
        let cols = rows_data[0].len();
        let mut data = Vec::with_capacity(rows * cols);
        for (i, row) in rows_data.iter().enumerate() {
            if row.len() != cols {
                return Err(SemiringError::DimensionMismatch {
                    expected: format!("{} columns (matching row 0)", cols),
                    got: format!("{} columns in row {}", row.len(), i),
                });
            }
            data.extend(row.iter().cloned());
        }
        Ok(Self { rows, cols, data })
    }

    /// Create a 1x1 matrix from a scalar.
    pub fn scalar(value: S) -> Self {
        Self {
            rows: 1,
            cols: 1,
            data: vec![value],
        }
    }

    /// Create a diagonal matrix from a vector.
    pub fn diagonal(diag: &[S]) -> Self {
        let n = diag.len();
        let mut m = Self::zeros(n, n);
        for (i, v) in diag.iter().enumerate() {
            m.data[i * n + i] = v.clone();
        }
        m
    }

    // -- Entry access -------------------------------------------------------

    /// Get a reference to the element at (row, col).
    pub fn get(&self, row: usize, col: usize) -> Result<&S, SemiringError> {
        if row >= self.rows || col >= self.cols {
            return Err(SemiringError::IndexOutOfBounds {
                row,
                col,
                rows: self.rows,
                cols: self.cols,
            });
        }
        Ok(&self.data[row * self.cols + col])
    }

    /// Get a mutable reference to the element at (row, col).
    pub fn get_mut(&mut self, row: usize, col: usize) -> Result<&mut S, SemiringError> {
        if row >= self.rows || col >= self.cols {
            return Err(SemiringError::IndexOutOfBounds {
                row,
                col,
                rows: self.rows,
                cols: self.cols,
            });
        }
        let idx = row * self.cols + col;
        Ok(&mut self.data[idx])
    }

    /// Set the element at (row, col).
    pub fn set(&mut self, row: usize, col: usize, value: S) -> Result<(), SemiringError> {
        if row >= self.rows || col >= self.cols {
            return Err(SemiringError::IndexOutOfBounds {
                row,
                col,
                rows: self.rows,
                cols: self.cols,
            });
        }
        self.data[row * self.cols + col] = value;
        Ok(())
    }

    // -- Arithmetic ---------------------------------------------------------

    /// Matrix addition.
    pub fn add(&self, other: &Self) -> Result<Self, SemiringError> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(SemiringError::DimensionMismatch {
                expected: format!("{}x{}", self.rows, self.cols),
                got: format!("{}x{}", other.rows, other.cols),
            });
        }
        let data: Vec<S> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a.add(b))
            .collect();
        Ok(Self {
            rows: self.rows,
            cols: self.cols,
            data,
        })
    }

    /// Matrix multiplication.
    pub fn mul(&self, other: &Self) -> Result<Self, SemiringError> {
        if self.cols != other.rows {
            return Err(SemiringError::DimensionMismatch {
                expected: format!("inner dimension {} (self.cols)", self.cols),
                got: format!("inner dimension {} (other.rows)", other.rows),
            });
        }
        let mut result = Self::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for k in 0..self.cols {
                let a_ik = &self.data[i * self.cols + k];
                if a_ik.is_zero() {
                    continue;
                }
                for j in 0..other.cols {
                    let prod = a_ik.mul(&other.data[k * other.cols + j]);
                    result.data[i * other.cols + j].add_assign(&prod);
                }
            }
        }
        Ok(result)
    }

    /// Scalar multiplication: multiply every entry by a scalar.
    pub fn scalar_mul(&self, scalar: &S) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().map(|x| x.mul(scalar)).collect(),
        }
    }

    /// Matrix transpose.
    pub fn transpose(&self) -> Self {
        let mut data = vec![S::zero(); self.rows * self.cols];
        for i in 0..self.rows {
            for j in 0..self.cols {
                data[j * self.rows + i] = self.data[i * self.cols + j].clone();
            }
        }
        Self {
            rows: self.cols,
            cols: self.rows,
            data,
        }
    }

    /// Kronecker (tensor) product of two matrices.
    pub fn kronecker(&self, other: &Self) -> Self {
        let new_rows = self.rows * other.rows;
        let new_cols = self.cols * other.cols;
        let mut data = vec![S::zero(); new_rows * new_cols];
        for i1 in 0..self.rows {
            for j1 in 0..self.cols {
                let a = &self.data[i1 * self.cols + j1];
                for i2 in 0..other.rows {
                    for j2 in 0..other.cols {
                        let b = &other.data[i2 * other.cols + j2];
                        let row = i1 * other.rows + i2;
                        let col = j1 * other.cols + j2;
                        data[row * new_cols + col] = a.mul(b);
                    }
                }
            }
        }
        Self {
            rows: new_rows,
            cols: new_cols,
            data,
        }
    }

    /// Matrix power via repeated squaring.
    pub fn mat_pow(&self, mut n: u64) -> Result<Self, SemiringError> {
        if self.rows != self.cols {
            return Err(SemiringError::NotSquare {
                rows: self.rows,
                cols: self.cols,
            });
        }
        if n == 0 {
            return Ok(Self::identity(self.rows));
        }
        let mut base = self.clone();
        let mut result = Self::identity(self.rows);
        while n > 1 {
            if n % 2 == 1 {
                result = result.mul(&base)?;
            }
            base = base.mul(&base)?;
            n /= 2;
        }
        result.mul(&base)
    }

    /// Extract a row as a vector.
    pub fn row(&self, i: usize) -> Result<Vec<S>, SemiringError> {
        if i >= self.rows {
            return Err(SemiringError::IndexOutOfBounds {
                row: i,
                col: 0,
                rows: self.rows,
                cols: self.cols,
            });
        }
        Ok(self.data[i * self.cols..(i + 1) * self.cols].to_vec())
    }

    /// Extract a column as a vector.
    pub fn col(&self, j: usize) -> Result<Vec<S>, SemiringError> {
        if j >= self.cols {
            return Err(SemiringError::IndexOutOfBounds {
                row: 0,
                col: j,
                rows: self.rows,
                cols: self.cols,
            });
        }
        Ok((0..self.rows)
            .map(|i| self.data[i * self.cols + j].clone())
            .collect())
    }

    /// Check if this is a square matrix.
    pub fn is_square(&self) -> bool {
        self.rows == self.cols
    }

    /// Trace: sum of diagonal elements.
    pub fn trace(&self) -> Result<S, SemiringError> {
        if !self.is_square() {
            return Err(SemiringError::NotSquare {
                rows: self.rows,
                cols: self.cols,
            });
        }
        let mut t = S::zero();
        for i in 0..self.rows {
            t.add_assign(&self.data[i * self.cols + i]);
        }
        Ok(t)
    }

    /// Extract a sub-matrix.
    pub fn submatrix(
        &self,
        row_start: usize,
        col_start: usize,
        num_rows: usize,
        num_cols: usize,
    ) -> Result<Self, SemiringError> {
        if row_start + num_rows > self.rows || col_start + num_cols > self.cols {
            return Err(SemiringError::IndexOutOfBounds {
                row: row_start + num_rows,
                col: col_start + num_cols,
                rows: self.rows,
                cols: self.cols,
            });
        }
        let mut data = Vec::with_capacity(num_rows * num_cols);
        for i in row_start..row_start + num_rows {
            for j in col_start..col_start + num_cols {
                data.push(self.data[i * self.cols + j].clone());
            }
        }
        Ok(Self {
            rows: num_rows,
            cols: num_cols,
            data,
        })
    }
}

impl<S: StarSemiring> SemiringMatrix<S> {
    /// Kleene closure (matrix star) via the Floyd-Warshall / Lehmann algorithm.
    ///
    /// For a square matrix A of dimension n, computes A* = I ⊕ A ⊕ A² ⊕ …
    /// using the recursive block decomposition:
    ///
    ///   A* uses the fact that for each intermediate node k, we update
    ///   A[i][j] = A[i][j] ⊕ A[i][k] ⊗ A[k][k]* ⊗ A[k][j]
    ///
    /// This is a direct generalization of Floyd-Warshall all-pairs shortest
    /// paths to arbitrary star semirings.
    pub fn matrix_star(&self) -> Result<Self, SemiringError> {
        if !self.is_square() {
            return Err(SemiringError::NotSquare {
                rows: self.rows,
                cols: self.cols,
            });
        }
        let n = self.rows;
        if n == 0 {
            return Ok(Self::zeros(0, 0));
        }

        // Start with I ⊕ A
        let mut result = Self::identity(n).add(self)?;

        // Floyd-Warshall style iteration
        for k in 0..n {
            let akk_star = result.data[k * n + k].star();
            for i in 0..n {
                for j in 0..n {
                    if i == k && j == k {
                        result.data[k * n + k] = akk_star.clone();
                    } else {
                        let a_ik = result.data[i * n + k].clone();
                        let a_kj = result.data[k * n + j].clone();
                        let update = a_ik.mul(&akk_star).mul(&a_kj);
                        result.data[i * n + j].add_assign(&update);
                    }
                }
            }
        }

        Ok(result)
    }

    /// Kleene plus: A⁺ = A ⊗ A*
    pub fn matrix_plus(&self) -> Result<Self, SemiringError> {
        let star = self.matrix_star()?;
        self.mul(&star)
    }
}

impl<S: Semiring + fmt::Display> fmt::Display for SemiringMatrix<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Matrix({}x{}):", self.rows, self.cols)?;
        for i in 0..self.rows {
            write!(f, "  [")?;
            for j in 0..self.cols {
                if j > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", self.data[i * self.cols + j])?;
            }
            writeln!(f, "]")?;
        }
        Ok(())
    }
}

// Implement ops::Add and ops::Mul for matrices for ergonomic use.

impl<S: Semiring> ops::Add for &SemiringMatrix<S> {
    type Output = Result<SemiringMatrix<S>, SemiringError>;
    fn add(self, rhs: Self) -> Self::Output {
        SemiringMatrix::add(self, rhs)
    }
}

impl<S: Semiring> ops::Mul for &SemiringMatrix<S> {
    type Output = Result<SemiringMatrix<S>, SemiringError>;
    fn mul(self, rhs: Self) -> Self::Output {
        SemiringMatrix::mul(self, rhs)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SemiringPolynomial<S>
// ═══════════════════════════════════════════════════════════════════════════

/// A univariate polynomial over a semiring S.
///
/// p(x) = c₀ ⊕ c₁ ⊗ x ⊕ c₂ ⊗ x² ⊕ … ⊕ cₙ ⊗ xⁿ
///
/// Coefficients are stored in order of ascending degree: `coeffs[i]` is the
/// coefficient of xⁱ.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SemiringPolynomial<S: Semiring> {
    pub coeffs: Vec<S>,
}

impl<S: Semiring> SemiringPolynomial<S> {
    /// Create a polynomial from coefficients (ascending degree order).
    pub fn new(coeffs: Vec<S>) -> Self {
        let mut p = Self { coeffs };
        p.normalize();
        p
    }

    /// The zero polynomial.
    pub fn zero() -> Self {
        Self {
            coeffs: Vec::new(),
        }
    }

    /// A constant polynomial.
    pub fn constant(c: S) -> Self {
        if c.is_zero() {
            Self::zero()
        } else {
            Self { coeffs: vec![c] }
        }
    }

    /// The monomial x^k with coefficient c.
    pub fn monomial(c: S, k: usize) -> Self {
        if c.is_zero() {
            return Self::zero();
        }
        let mut coeffs = vec![S::zero(); k + 1];
        coeffs[k] = c;
        Self { coeffs }
    }

    /// The polynomial x (identity monomial).
    pub fn x() -> Self {
        Self::monomial(S::one(), 1)
    }

    /// Remove trailing zero coefficients.
    fn normalize(&mut self) {
        while self.coeffs.last().map_or(false, |c| c.is_zero()) {
            self.coeffs.pop();
        }
    }

    /// Degree of the polynomial, or None for the zero polynomial.
    pub fn degree(&self) -> Option<usize> {
        if self.coeffs.is_empty() {
            None
        } else {
            Some(self.coeffs.len() - 1)
        }
    }

    /// Leading coefficient, or None for the zero polynomial.
    pub fn leading_coeff(&self) -> Option<&S> {
        self.coeffs.last()
    }

    /// Get the coefficient of x^k (zero if k is out of range).
    pub fn coeff(&self, k: usize) -> S {
        if k < self.coeffs.len() {
            self.coeffs[k].clone()
        } else {
            S::zero()
        }
    }

    /// Evaluate the polynomial at a point using Horner's method.
    pub fn evaluate(&self, x: &S) -> S {
        if self.coeffs.is_empty() {
            return S::zero();
        }
        let mut result = self.coeffs.last().unwrap().clone();
        for i in (0..self.coeffs.len() - 1).rev() {
            result = result.mul(x).add(&self.coeffs[i]);
        }
        result
    }

    /// Polynomial addition.
    pub fn add(&self, other: &Self) -> Self {
        let max_len = self.coeffs.len().max(other.coeffs.len());
        let mut coeffs = Vec::with_capacity(max_len);
        for i in 0..max_len {
            let a = self.coeff(i);
            let b = other.coeff(i);
            coeffs.push(a.add(&b));
        }
        let mut p = Self { coeffs };
        p.normalize();
        p
    }

    /// Polynomial multiplication (convolution).
    pub fn mul(&self, other: &Self) -> Self {
        if self.coeffs.is_empty() || other.coeffs.is_empty() {
            return Self::zero();
        }
        let result_len = self.coeffs.len() + other.coeffs.len() - 1;
        let mut coeffs = vec![S::zero(); result_len];
        for (i, a) in self.coeffs.iter().enumerate() {
            if a.is_zero() {
                continue;
            }
            for (j, b) in other.coeffs.iter().enumerate() {
                let prod = a.mul(b);
                coeffs[i + j].add_assign(&prod);
            }
        }
        let mut p = Self { coeffs };
        p.normalize();
        p
    }

    /// Scalar multiplication.
    pub fn scalar_mul(&self, scalar: &S) -> Self {
        let coeffs: Vec<S> = self.coeffs.iter().map(|c| c.mul(scalar)).collect();
        let mut p = Self { coeffs };
        p.normalize();
        p
    }

    /// Is this the zero polynomial?
    pub fn is_zero(&self) -> bool {
        self.coeffs.is_empty()
    }

    /// Polynomial power via repeated squaring.
    pub fn pow(&self, mut n: u64) -> Self {
        if n == 0 {
            return Self::constant(S::one());
        }
        let mut base = self.clone();
        let mut result = Self::constant(S::one());
        while n > 1 {
            if n % 2 == 1 {
                result = result.mul(&base);
            }
            base = base.mul(&base);
            n /= 2;
        }
        result.mul(&base)
    }

    /// Formal derivative: p'(x) = Σ i · cᵢ · x^{i-1}
    /// Note: in a general semiring, "i · c" means c ⊕ c ⊕ … ⊕ c (i times).
    pub fn formal_derivative(&self) -> Self {
        if self.coeffs.len() <= 1 {
            return Self::zero();
        }
        let mut coeffs = Vec::with_capacity(self.coeffs.len() - 1);
        for i in 1..self.coeffs.len() {
            // Multiply coefficient by i (add it to itself i times)
            let mut c = S::zero();
            for _ in 0..i {
                c.add_assign(&self.coeffs[i]);
            }
            coeffs.push(c);
        }
        let mut p = Self { coeffs };
        p.normalize();
        p
    }

    /// Compose: compute p(q(x)).
    pub fn compose(&self, other: &Self) -> Self {
        if self.coeffs.is_empty() {
            return Self::zero();
        }
        // Horner's method adapted for polynomial composition
        let mut result = Self::constant(self.coeffs.last().unwrap().clone());
        for i in (0..self.coeffs.len() - 1).rev() {
            result = result.mul(other).add(&Self::constant(self.coeffs[i].clone()));
        }
        result
    }
}

impl<S: Semiring + fmt::Display> fmt::Display for SemiringPolynomial<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.coeffs.is_empty() {
            return write!(f, "0");
        }
        let mut first = true;
        for (i, c) in self.coeffs.iter().enumerate() {
            if c.is_zero() {
                continue;
            }
            if !first {
                write!(f, " + ")?;
            }
            first = false;
            match i {
                0 => write!(f, "{}", c)?,
                1 => write!(f, "{}·x", c)?,
                _ => write!(f, "{}·x^{}", c, i)?,
            }
        }
        if first {
            write!(f, "0")?;
        }
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SemiringHomomorphism trait and implementations
// ═══════════════════════════════════════════════════════════════════════════

/// A semiring homomorphism h: S → T preserving:
///   h(0_S) = 0_T
///   h(1_S) = 1_T
///   h(a ⊕ b) = h(a) ⊕ h(b)
///   h(a ⊗ b) = h(a) ⊗ h(b)
pub trait SemiringHomomorphism<S: Semiring, T: Semiring>: fmt::Debug + Send + Sync {
    /// Apply the homomorphism.
    fn apply(&self, x: &S) -> T;

    /// Name of this homomorphism (for debugging).
    fn name(&self) -> &str;
}

// -- CountingToBoolean: ℕ → {0,1} via (n ↦ n > 0) -------------------------

/// Maps counting semiring to Boolean: n ↦ (n > 0).
#[derive(Clone, Debug)]
pub struct CountingToBoolean;

impl SemiringHomomorphism<CountingSemiring, BooleanSemiring> for CountingToBoolean {
    fn apply(&self, x: &CountingSemiring) -> BooleanSemiring {
        BooleanSemiring::new(x.value > 0)
    }

    fn name(&self) -> &str {
        "CountingToBoolean"
    }
}

// -- CountingToGoldilocks: ℕ → F_p via reduction mod p ---------------------

/// Maps counting semiring into the Goldilocks field by reduction mod p.
#[derive(Clone, Debug)]
pub struct CountingToGoldilocks;

impl SemiringHomomorphism<CountingSemiring, GoldilocksField> for CountingToGoldilocks {
    fn apply(&self, x: &CountingSemiring) -> GoldilocksField {
        GoldilocksField::new(x.value)
    }

    fn name(&self) -> &str {
        "CountingToGoldilocks"
    }
}

// -- BooleanToGoldilocks: {0,1} → F_p via (false↦0, true↦1) ----------------

/// Maps Boolean semiring into the Goldilocks field.
#[derive(Clone, Debug)]
pub struct BooleanToGoldilocks;

impl SemiringHomomorphism<BooleanSemiring, GoldilocksField> for BooleanToGoldilocks {
    fn apply(&self, x: &BooleanSemiring) -> GoldilocksField {
        if x.value {
            GoldilocksField::one()
        } else {
            GoldilocksField::zero()
        }
    }

    fn name(&self) -> &str {
        "BooleanToGoldilocks"
    }
}

// -- BooleanToCounting: {0,1} → ℕ via (false↦0, true↦1) --------------------

/// Maps Boolean semiring into the counting semiring.
#[derive(Clone, Debug)]
pub struct BooleanToCounting;

impl SemiringHomomorphism<BooleanSemiring, CountingSemiring> for BooleanToCounting {
    fn apply(&self, x: &BooleanSemiring) -> CountingSemiring {
        CountingSemiring::new(if x.value { 1 } else { 0 })
    }

    fn name(&self) -> &str {
        "BooleanToCounting"
    }
}

// -- RealToTropical: (ℝ,+,·) → (ℝ∪{+∞}, min, +) via -log -----------------

/// Maps the real semiring (probabilities) to the tropical semiring via -log.
/// This is useful for converting probabilistic automata to shortest-path form.
#[derive(Clone, Debug)]
pub struct RealToTropical;

impl SemiringHomomorphism<RealSemiring, TropicalSemiring> for RealToTropical {
    fn apply(&self, x: &RealSemiring) -> TropicalSemiring {
        let v = x.raw();
        if v <= 0.0 {
            TropicalSemiring::infinity()
        } else {
            TropicalSemiring::new(-v.ln())
        }
    }

    fn name(&self) -> &str {
        "RealToTropical"
    }
}

// -- RealToLog: (ℝ,+,·) → log-semiring via log -----------------------------

/// Maps the real semiring to the log semiring via the logarithm.
#[derive(Clone, Debug)]
pub struct RealToLog;

impl SemiringHomomorphism<RealSemiring, LogSemiring> for RealToLog {
    fn apply(&self, x: &RealSemiring) -> LogSemiring {
        let v = x.raw();
        if v <= 0.0 {
            LogSemiring::neg_infinity()
        } else {
            LogSemiring::new(v.ln())
        }
    }

    fn name(&self) -> &str {
        "RealToLog"
    }
}

// -- RealToViterbi: (ℝ,+,·) → ([0,1], max, ·) via clamping ----------------

/// Maps the real semiring to the Viterbi semiring by clamping to [0, 1].
#[derive(Clone, Debug)]
pub struct RealToViterbi;

impl SemiringHomomorphism<RealSemiring, ViterbiSemiring> for RealToViterbi {
    fn apply(&self, x: &RealSemiring) -> ViterbiSemiring {
        let v = x.raw().max(0.0).min(1.0);
        ViterbiSemiring::new(v)
    }

    fn name(&self) -> &str {
        "RealToViterbi"
    }
}

// -- TropicalToMaxPlus: flip sign -------------------------------------------

/// Maps tropical (min, +) to max-plus by negating.
#[derive(Clone, Debug)]
pub struct TropicalToMaxPlus;

impl SemiringHomomorphism<TropicalSemiring, MaxPlusSemiring> for TropicalToMaxPlus {
    fn apply(&self, x: &TropicalSemiring) -> MaxPlusSemiring {
        let v = x.raw();
        if v.is_infinite() && v > 0.0 {
            MaxPlusSemiring::neg_infinity()
        } else {
            MaxPlusSemiring::new(-v)
        }
    }

    fn name(&self) -> &str {
        "TropicalToMaxPlus"
    }
}

// -- GoldilocksToFieldSemiring: embedding -----------------------------------

/// Maps Goldilocks field elements into a generic FieldSemiring with a larger
/// or equal prime by embedding (value mod new_prime).
#[derive(Clone, Debug)]
pub struct GoldilocksToField<P: PrimeModulus> {
    _phantom: PhantomData<P>,
}

impl<P: PrimeModulus> GoldilocksToField<P> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<P: PrimeModulus> SemiringHomomorphism<GoldilocksField, FieldSemiring<P>>
    for GoldilocksToField<P>
{
    fn apply(&self, x: &GoldilocksField) -> FieldSemiring<P> {
        FieldSemiring::new(x.value)
    }

    fn name(&self) -> &str {
        "GoldilocksToField"
    }
}

// -- CountingToReal: ℕ → ℝ via casting --------------------------------------

/// Natural embedding of the counting semiring into the reals.
#[derive(Clone, Debug)]
pub struct CountingToReal;

impl SemiringHomomorphism<CountingSemiring, RealSemiring> for CountingToReal {
    fn apply(&self, x: &CountingSemiring) -> RealSemiring {
        RealSemiring::new(x.value as f64)
    }

    fn name(&self) -> &str {
        "CountingToReal"
    }
}

// -- IdentityHomomorphism ---------------------------------------------------

/// The identity homomorphism on any semiring.
#[derive(Clone, Debug)]
pub struct IdentityHomomorphism<S: Semiring> {
    _phantom: PhantomData<S>,
}

impl<S: Semiring> IdentityHomomorphism<S> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<S: Semiring> SemiringHomomorphism<S, S> for IdentityHomomorphism<S> {
    fn apply(&self, x: &S) -> S {
        x.clone()
    }

    fn name(&self) -> &str {
        "Identity"
    }
}

// -- ComposedHomomorphism ---------------------------------------------------

/// Composition of two homomorphisms h₂ ∘ h₁ : S → U via intermediate T.
#[derive(Debug)]
pub struct ComposedHomomorphism<S, T, U, H1, H2>
where
    S: Semiring,
    T: Semiring,
    U: Semiring,
    H1: SemiringHomomorphism<S, T>,
    H2: SemiringHomomorphism<T, U>,
{
    first: H1,
    second: H2,
    _phantom: PhantomData<(S, T, U)>,
}

impl<S, T, U, H1, H2> ComposedHomomorphism<S, T, U, H1, H2>
where
    S: Semiring,
    T: Semiring,
    U: Semiring,
    H1: SemiringHomomorphism<S, T>,
    H2: SemiringHomomorphism<T, U>,
{
    pub fn new(first: H1, second: H2) -> Self {
        Self {
            first,
            second,
            _phantom: PhantomData,
        }
    }
}

impl<S, T, U, H1, H2> SemiringHomomorphism<S, U> for ComposedHomomorphism<S, T, U, H1, H2>
where
    S: Semiring,
    T: Semiring,
    U: Semiring,
    H1: SemiringHomomorphism<S, T>,
    H2: SemiringHomomorphism<T, U>,
{
    fn apply(&self, x: &S) -> U {
        let intermediate = self.first.apply(x);
        self.second.apply(&intermediate)
    }

    fn name(&self) -> &str {
        "Composed"
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Utility functions
// ═══════════════════════════════════════════════════════════════════════════

/// Compute the semiring "sum" (⊕-aggregate) of a slice.
pub fn semiring_sum<S: Semiring>(values: &[S]) -> S {
    values.iter().fold(S::zero(), |acc, v| acc.add(v))
}

/// Compute the semiring "product" (⊗-aggregate) of a slice.
pub fn semiring_product<S: Semiring>(values: &[S]) -> S {
    values.iter().fold(S::one(), |acc, v| acc.mul(v))
}

/// Compute the dot product of two vectors over a semiring.
pub fn semiring_dot<S: Semiring>(a: &[S], b: &[S]) -> Result<S, SemiringError> {
    if a.len() != b.len() {
        return Err(SemiringError::DimensionMismatch {
            expected: format!("length {}", a.len()),
            got: format!("length {}", b.len()),
        });
    }
    let mut result = S::zero();
    for (x, y) in a.iter().zip(b.iter()) {
        result.add_assign(&x.mul(y));
    }
    Ok(result)
}

/// Apply a homomorphism element-wise to a matrix.
pub fn map_matrix<S: Semiring, T: Semiring>(
    matrix: &SemiringMatrix<S>,
    h: &dyn SemiringHomomorphism<S, T>,
) -> SemiringMatrix<T> {
    let data: Vec<T> = matrix.data.iter().map(|x| h.apply(x)).collect();
    SemiringMatrix {
        rows: matrix.rows,
        cols: matrix.cols,
        data,
    }
}

/// Apply a homomorphism to each coefficient of a polynomial.
pub fn map_polynomial<S: Semiring, T: Semiring>(
    poly: &SemiringPolynomial<S>,
    h: &dyn SemiringHomomorphism<S, T>,
) -> SemiringPolynomial<T> {
    let coeffs: Vec<T> = poly.coeffs.iter().map(|c| h.apply(c)).collect();
    SemiringPolynomial::new(coeffs)
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: check semiring axioms for a given set of elements.
    fn check_semiring_axioms<S: Semiring + fmt::Debug>(a: &S, b: &S, c: &S, name: &str) {
        // Commutativity of ⊕
        assert_eq!(
            a.add(b),
            b.add(a),
            "{}: add commutativity failed for {:?}, {:?}",
            name,
            a,
            b
        );

        // Associativity of ⊕
        assert_eq!(
            a.add(&b.add(c)),
            a.add(b).add(c),
            "{}: add associativity failed",
            name
        );

        // Associativity of ⊗
        assert_eq!(
            a.mul(&b.mul(c)),
            a.mul(b).mul(c),
            "{}: mul associativity failed",
            name
        );

        // Identity for ⊕
        assert_eq!(
            a.add(&S::zero()),
            *a,
            "{}: additive identity failed",
            name
        );
        assert_eq!(
            S::zero().add(a),
            *a,
            "{}: additive identity (left) failed",
            name
        );

        // Identity for ⊗
        assert_eq!(
            a.mul(&S::one()),
            *a,
            "{}: multiplicative identity failed",
            name
        );
        assert_eq!(
            S::one().mul(a),
            *a,
            "{}: multiplicative identity (left) failed",
            name
        );

        // Zero annihilates
        assert_eq!(
            a.mul(&S::zero()),
            S::zero(),
            "{}: zero annihilation (right) failed",
            name
        );
        assert_eq!(
            S::zero().mul(a),
            S::zero(),
            "{}: zero annihilation (left) failed",
            name
        );

        // Left distributivity: a ⊗ (b ⊕ c) = (a ⊗ b) ⊕ (a ⊗ c)
        assert_eq!(
            a.mul(&b.add(c)),
            a.mul(b).add(&a.mul(c)),
            "{}: left distributivity failed",
            name
        );

        // Right distributivity: (a ⊕ b) ⊗ c = (a ⊗ c) ⊕ (b ⊗ c)
        assert_eq!(
            a.add(b).mul(c),
            a.mul(c).add(&b.mul(c)),
            "{}: right distributivity failed",
            name
        );
    }

    // -----------------------------------------------------------------------
    // CountingSemiring tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_counting_semiring_axioms() {
        let a = CountingSemiring::new(3);
        let b = CountingSemiring::new(5);
        let c = CountingSemiring::new(7);
        check_semiring_axioms(&a, &b, &c, "CountingSemiring");
    }

    #[test]
    fn test_counting_semiring_basic() {
        let a = CountingSemiring::new(3);
        let b = CountingSemiring::new(5);
        assert_eq!(a.add(&b), CountingSemiring::new(8));
        assert_eq!(a.mul(&b), CountingSemiring::new(15));
        assert!(CountingSemiring::zero().is_zero());
        assert!(CountingSemiring::one().is_one());
        assert!(!a.is_zero());
        assert!(!a.is_one());
    }

    #[test]
    fn test_counting_semiring_overflow() {
        let a = CountingSemiring::new(u64::MAX);
        let b = CountingSemiring::new(1);
        // Saturating add
        assert_eq!(a.add(&b), CountingSemiring::new(u64::MAX));
        // Saturating mul
        let c = CountingSemiring::new(u64::MAX);
        assert_eq!(a.mul(&c), CountingSemiring::new(u64::MAX));
    }

    #[test]
    fn test_counting_semiring_pow() {
        let a = CountingSemiring::new(2);
        assert_eq!(a.pow(0), CountingSemiring::one());
        assert_eq!(a.pow(1), CountingSemiring::new(2));
        assert_eq!(a.pow(10), CountingSemiring::new(1024));
    }

    // -----------------------------------------------------------------------
    // BooleanSemiring tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_boolean_semiring_axioms() {
        let vals = [BooleanSemiring::new(false), BooleanSemiring::new(true)];
        for a in &vals {
            for b in &vals {
                for c in &vals {
                    check_semiring_axioms(a, b, c, "BooleanSemiring");
                }
            }
        }
    }

    #[test]
    fn test_boolean_semiring_basic() {
        let t = BooleanSemiring::new(true);
        let f = BooleanSemiring::new(false);
        assert_eq!(f.add(&f), f);
        assert_eq!(f.add(&t), t);
        assert_eq!(t.add(&f), t);
        assert_eq!(t.add(&t), t);
        assert_eq!(f.mul(&f), f);
        assert_eq!(f.mul(&t), f);
        assert_eq!(t.mul(&f), f);
        assert_eq!(t.mul(&t), t);
    }

    #[test]
    fn test_boolean_star() {
        let t = BooleanSemiring::new(true);
        let f = BooleanSemiring::new(false);
        assert_eq!(t.star(), BooleanSemiring::one());
        assert_eq!(f.star(), BooleanSemiring::one());
    }

    // -----------------------------------------------------------------------
    // TropicalSemiring tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_tropical_semiring_axioms() {
        let a = TropicalSemiring::new(1.0);
        let b = TropicalSemiring::new(3.0);
        let c = TropicalSemiring::new(5.0);
        check_semiring_axioms(&a, &b, &c, "TropicalSemiring");
    }

    #[test]
    fn test_tropical_semiring_basic() {
        let a = TropicalSemiring::new(3.0);
        let b = TropicalSemiring::new(5.0);
        // min(3, 5) = 3
        assert_eq!(a.add(&b), TropicalSemiring::new(3.0));
        // 3 + 5 = 8
        assert_eq!(a.mul(&b), TropicalSemiring::new(8.0));
        assert!(TropicalSemiring::infinity().is_zero());
        assert!(TropicalSemiring::new(0.0).is_one());
    }

    #[test]
    fn test_tropical_star() {
        let a = TropicalSemiring::new(5.0);
        assert_eq!(a.star(), TropicalSemiring::one());
        let b = TropicalSemiring::new(-1.0);
        assert_eq!(b.star(), TropicalSemiring::new(f64::NEG_INFINITY));
    }

    #[test]
    fn test_tropical_infinity_handling() {
        let inf = TropicalSemiring::infinity();
        let a = TropicalSemiring::new(3.0);
        // min(∞, 3) = 3
        assert_eq!(inf.add(&a), a);
        // ∞ + 3 = ∞
        assert!(inf.mul(&a).is_zero());
    }

    // -----------------------------------------------------------------------
    // BoundedCountingSemiring tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_bounded_counting_semiring() {
        let a = BoundedCountingSemiring::new(3, 5);
        let b = BoundedCountingSemiring::new(4, 5);
        // 3 + 4 = 7 but clamped to 5
        let sum = a.add(&b);
        assert_eq!(sum.value, 5);
        assert_eq!(sum.bound, 5);

        let c = BoundedCountingSemiring::new(2, 5);
        let d = BoundedCountingSemiring::new(3, 5);
        let prod = c.mul(&d);
        // 2 * 3 = 6 clamped to 5
        assert_eq!(prod.value, 5);
    }

    #[test]
    fn test_bounded_counting_no_clamp() {
        let a = BoundedCountingSemiring::new(2, 10);
        let b = BoundedCountingSemiring::new(3, 10);
        assert_eq!(a.add(&b).value, 5);
        assert_eq!(a.mul(&b).value, 6);
    }

    // -----------------------------------------------------------------------
    // GoldilocksField tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_goldilocks_axioms() {
        let a = GoldilocksField::new(42);
        let b = GoldilocksField::new(1337);
        let c = GoldilocksField::new(999999);
        check_semiring_axioms(&a, &b, &c, "GoldilocksField");
    }

    #[test]
    fn test_goldilocks_basic_arithmetic() {
        let a = GoldilocksField::new(10);
        let b = GoldilocksField::new(20);
        assert_eq!(a.add(&b), GoldilocksField::new(30));
        assert_eq!(a.mul(&b), GoldilocksField::new(200));
        assert_eq!(a.sub(&b), GoldilocksField::new(GOLDILOCKS_PRIME - 10));
    }

    #[test]
    fn test_goldilocks_modular_reduction() {
        let p = GOLDILOCKS_PRIME;
        let a = GoldilocksField::new(p - 1);
        let b = GoldilocksField::new(2);
        // (p-1) + 2 = p+1 ≡ 1 (mod p)... wait, (p-1)+2 = p+1 ≡ 1
        let sum = a.add(&b);
        assert_eq!(sum.value, 1);
    }

    #[test]
    fn test_goldilocks_inverse() {
        let a = GoldilocksField::new(42);
        let inv = a.inv().unwrap();
        let product = a.mul(&inv);
        assert_eq!(product, GoldilocksField::one());
    }

    #[test]
    fn test_goldilocks_inverse_zero() {
        let z = GoldilocksField::zero();
        assert!(z.inv().is_err());
    }

    #[test]
    fn test_goldilocks_fermat_little_theorem() {
        // a^p ≡ a (mod p) for a ≠ 0
        let a = GoldilocksField::new(12345);
        let a_pow_p = a.field_pow(GOLDILOCKS_PRIME);
        assert_eq!(a_pow_p, a);

        // a^{p-1} ≡ 1 (mod p) for a ≠ 0
        let a_pow_pm1 = a.field_pow(GOLDILOCKS_PRIME - 1);
        assert_eq!(a_pow_pm1, GoldilocksField::one());
    }

    #[test]
    fn test_goldilocks_negation() {
        let a = GoldilocksField::new(100);
        let neg_a = a.neg();
        let sum = a.add(&neg_a);
        assert_eq!(sum, GoldilocksField::zero());
    }

    #[test]
    fn test_goldilocks_division() {
        let a = GoldilocksField::new(100);
        let b = GoldilocksField::new(25);
        let result = a.div(&b).unwrap();
        assert_eq!(result, GoldilocksField::new(4));
    }

    #[test]
    fn test_goldilocks_batch_inverse() {
        let elements: Vec<GoldilocksField> = (1..=10).map(|i| GoldilocksField::new(i)).collect();
        let inverses = GoldilocksField::batch_inverse(&elements).unwrap();
        for (elem, inv) in elements.iter().zip(inverses.iter()) {
            assert_eq!(elem.mul(inv), GoldilocksField::one());
        }
    }

    #[test]
    fn test_goldilocks_sqrt() {
        // 4 should have a square root (2 or p-2)
        let four = GoldilocksField::new(4);
        let root = four.sqrt();
        assert!(root.is_some());
        let r = root.unwrap();
        assert_eq!(r.mul(&r), four);
    }

    #[test]
    fn test_goldilocks_pow() {
        let a = GoldilocksField::new(3);
        assert_eq!(a.field_pow(0), GoldilocksField::one());
        assert_eq!(a.field_pow(1), a);
        assert_eq!(a.field_pow(2), GoldilocksField::new(9));
        assert_eq!(a.field_pow(3), GoldilocksField::new(27));
    }

    #[test]
    fn test_goldilocks_large_values() {
        let p = GOLDILOCKS_PRIME;
        let a = GoldilocksField::new(p - 1);
        let b = GoldilocksField::new(p - 1);
        // (p-1)*(p-1) = p^2 - 2p + 1 ≡ 1 (mod p)
        let prod = a.mul(&b);
        assert_eq!(prod, GoldilocksField::one());
    }

    // -----------------------------------------------------------------------
    // FieldSemiring tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_field_semiring_axioms() {
        type F = FieldSemiring<SmallTestPrime>;
        let a = F::new(42);
        let b = F::new(1337);
        let c = F::new(999999);
        check_semiring_axioms(&a, &b, &c, "FieldSemiring<SmallTestPrime>");
    }

    #[test]
    fn test_field_semiring_inverse() {
        type F = FieldSemiring<SmallTestPrime>;
        let a = F::new(42);
        let inv = a.inv().unwrap();
        let product = a.mul(&inv);
        assert_eq!(product, F::one());
    }

    #[test]
    fn test_field_semiring_mersenne() {
        type F = FieldSemiring<MersennePrime31>;
        let a = F::new(100);
        let b = F::new(200);
        assert_eq!(a.add(&b), F::new(300));
        assert_eq!(a.mul(&b), F::new(20000));
        let inv = a.inv().unwrap();
        assert_eq!(a.mul(&inv), F::one());
    }

    // -----------------------------------------------------------------------
    // RealSemiring tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_real_semiring_axioms() {
        let a = RealSemiring::new(1.5);
        let b = RealSemiring::new(2.5);
        let c = RealSemiring::new(3.5);
        check_semiring_axioms(&a, &b, &c, "RealSemiring");
    }

    #[test]
    fn test_real_star() {
        let a = RealSemiring::new(0.5);
        let star = a.star();
        // 1/(1-0.5) = 2
        assert!((star.raw() - 2.0).abs() < 1e-10);

        let b = RealSemiring::new(2.0);
        let star_b = b.star();
        assert!(star_b.raw().is_infinite());
    }

    // -----------------------------------------------------------------------
    // LogSemiring tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_log_semiring_axioms() {
        let a = LogSemiring::new(-1.0);
        let b = LogSemiring::new(-2.0);
        let c = LogSemiring::new(-3.0);
        // LogSemiring uses log-sum-exp which has floating point precision limits
        // Verify axioms with approximate comparison
        let eps = 1e-12;
        // Commutativity
        assert!((a.add(&b).raw() - b.add(&a).raw()).abs() < eps, "add commutativity");
        // Associativity (log-sum-exp can lose precision)
        assert!((a.add(&b.add(&c)).raw() - a.add(&b).add(&c).raw()).abs() < eps, "add associativity");
        assert!((a.mul(&b.mul(&c)).raw() - a.mul(&b).mul(&c).raw()).abs() < eps, "mul associativity");
        // Identity
        assert!((a.add(&LogSemiring::zero()).raw() - a.raw()).abs() < eps, "additive identity");
        assert!((a.mul(&LogSemiring::one()).raw() - a.raw()).abs() < eps, "multiplicative identity");
        // Annihilation
        assert!(a.mul(&LogSemiring::zero()).is_zero(), "annihilation");
        // Distributivity
        assert!((a.mul(&b.add(&c)).raw() - a.mul(&b).add(&a.mul(&c)).raw()).abs() < eps, "left distributivity");
    }

    #[test]
    fn test_log_semiring_basic() {
        let a = LogSemiring::new(2.0);
        let b = LogSemiring::new(3.0);
        // mul is addition in log domain
        assert_eq!(a.mul(&b), LogSemiring::new(5.0));
        // add is log-sum-exp
        let sum = a.add(&b);
        let expected = (2.0f64.exp() + 3.0f64.exp()).ln();
        assert!((sum.raw() - expected).abs() < 1e-10);
    }

    #[test]
    fn test_log_semiring_neg_infinity() {
        let neg_inf = LogSemiring::neg_infinity();
        let a = LogSemiring::new(2.0);
        assert_eq!(neg_inf.add(&a), a);
        assert!(neg_inf.is_zero());
    }

    // -----------------------------------------------------------------------
    // MaxPlusSemiring tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_max_plus_semiring_axioms() {
        let a = MaxPlusSemiring::new(1.0);
        let b = MaxPlusSemiring::new(3.0);
        let c = MaxPlusSemiring::new(5.0);
        check_semiring_axioms(&a, &b, &c, "MaxPlusSemiring");
    }

    #[test]
    fn test_max_plus_basic() {
        let a = MaxPlusSemiring::new(3.0);
        let b = MaxPlusSemiring::new(5.0);
        assert_eq!(a.add(&b), MaxPlusSemiring::new(5.0)); // max
        assert_eq!(a.mul(&b), MaxPlusSemiring::new(8.0)); // +
    }

    #[test]
    fn test_max_plus_star() {
        let a = MaxPlusSemiring::new(-3.0);
        assert_eq!(a.star(), MaxPlusSemiring::one()); // 0
        let b = MaxPlusSemiring::new(2.0);
        assert_eq!(b.star().raw(), f64::INFINITY);
    }

    // -----------------------------------------------------------------------
    // MinMaxSemiring tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_min_max_semiring_axioms() {
        let a = MinMaxSemiring::new(1.0);
        let b = MinMaxSemiring::new(3.0);
        let c = MinMaxSemiring::new(5.0);
        check_semiring_axioms(&a, &b, &c, "MinMaxSemiring");
    }

    #[test]
    fn test_min_max_basic() {
        let a = MinMaxSemiring::new(3.0);
        let b = MinMaxSemiring::new(5.0);
        assert_eq!(a.add(&b), MinMaxSemiring::new(3.0)); // min
        assert_eq!(a.mul(&b), MinMaxSemiring::new(5.0)); // max
    }

    // -----------------------------------------------------------------------
    // ViterbiSemiring tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_viterbi_semiring_axioms() {
        let a = ViterbiSemiring::new(0.2);
        let b = ViterbiSemiring::new(0.5);
        let c = ViterbiSemiring::new(0.8);
        check_semiring_axioms(&a, &b, &c, "ViterbiSemiring");
    }

    #[test]
    fn test_viterbi_basic() {
        let a = ViterbiSemiring::new(0.3);
        let b = ViterbiSemiring::new(0.7);
        assert_eq!(a.add(&b), ViterbiSemiring::new(0.7)); // max
        let prod = a.mul(&b);
        assert!((prod.raw() - 0.21).abs() < 1e-10); // product
    }

    #[test]
    fn test_viterbi_star() {
        let a = ViterbiSemiring::new(0.5);
        assert_eq!(a.star(), ViterbiSemiring::one());
    }

    // -----------------------------------------------------------------------
    // ExpectationSemiring tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_expectation_semiring_axioms() {
        type E = ExpectationSemiring<RealSemiring>;
        let a = E::new(RealSemiring::new(1.0), RealSemiring::new(2.0));
        let b = E::new(RealSemiring::new(3.0), RealSemiring::new(4.0));
        let c = E::new(RealSemiring::new(5.0), RealSemiring::new(6.0));
        check_semiring_axioms(&a, &b, &c, "ExpectationSemiring<Real>");
    }

    #[test]
    fn test_expectation_mul() {
        type E = ExpectationSemiring<RealSemiring>;
        let a = E::new(RealSemiring::new(2.0), RealSemiring::new(3.0));
        let b = E::new(RealSemiring::new(4.0), RealSemiring::new(5.0));
        let prod = a.mul(&b);
        // value: 2*4 = 8
        assert!((prod.value.raw() - 8.0).abs() < 1e-10);
        // expectation: 2*5 + 3*4 = 10 + 12 = 22
        assert!((prod.expectation.raw() - 22.0).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // ProductSemiring tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_product_semiring_axioms() {
        type P = ProductSemiring<CountingSemiring, BooleanSemiring>;
        let a = P::new(CountingSemiring::new(1), BooleanSemiring::new(true));
        let b = P::new(CountingSemiring::new(2), BooleanSemiring::new(false));
        let c = P::new(CountingSemiring::new(3), BooleanSemiring::new(true));
        check_semiring_axioms(&a, &b, &c, "ProductSemiring<Counting,Boolean>");
    }

    #[test]
    fn test_product_semiring_basic() {
        type P = ProductSemiring<CountingSemiring, TropicalSemiring>;
        let a = P::new(CountingSemiring::new(2), TropicalSemiring::new(3.0));
        let b = P::new(CountingSemiring::new(3), TropicalSemiring::new(5.0));
        let sum = a.add(&b);
        assert_eq!(sum.first, CountingSemiring::new(5));
        assert_eq!(sum.second, TropicalSemiring::new(3.0));
        let prod = a.mul(&b);
        assert_eq!(prod.first, CountingSemiring::new(6));
        assert_eq!(prod.second, TropicalSemiring::new(8.0));
    }

    // -----------------------------------------------------------------------
    // FreeMonoidSemiring tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_free_monoid_axioms() {
        let a = FreeMonoidSemiring::singleton("a");
        let b = FreeMonoidSemiring::singleton("b");
        let c = FreeMonoidSemiring::singleton("c");
        check_semiring_axioms(&a, &b, &c, "FreeMonoidSemiring");
    }

    #[test]
    fn test_free_monoid_basic() {
        let a = FreeMonoidSemiring::singleton("x");
        let b = FreeMonoidSemiring::singleton("y");
        let sum = a.add(&b);
        assert_eq!(sum.strings, vec!["x".to_string(), "y".to_string()]);
        let prod = a.mul(&b);
        assert_eq!(prod.strings, vec!["xy".to_string()]);
    }

    #[test]
    fn test_free_monoid_identity() {
        let a = FreeMonoidSemiring::singleton("hello");
        let one = FreeMonoidSemiring::one();
        assert_eq!(a.mul(&one), a);
        assert_eq!(one.mul(&a), a);
    }

    #[test]
    fn test_free_monoid_zero() {
        let a = FreeMonoidSemiring::singleton("hello");
        let zero = FreeMonoidSemiring::zero();
        assert_eq!(a.mul(&zero), zero);
        assert_eq!(zero.mul(&a), zero);
    }

    #[test]
    fn test_free_monoid_set_union() {
        let a = FreeMonoidSemiring::new(vec!["a".into(), "b".into()]);
        let b = FreeMonoidSemiring::new(vec!["b".into(), "c".into()]);
        let sum = a.add(&b);
        assert_eq!(
            sum.strings,
            vec!["a".to_string(), "b".to_string(), "c".to_string()]
        );
    }

    #[test]
    fn test_free_monoid_concat_product() {
        let a = FreeMonoidSemiring::new(vec!["a".into(), "b".into()]);
        let b = FreeMonoidSemiring::new(vec!["x".into(), "y".into()]);
        let prod = a.mul(&b);
        assert_eq!(
            prod.strings,
            vec![
                "ax".to_string(),
                "ay".to_string(),
                "bx".to_string(),
                "by".to_string()
            ]
        );
    }

    // -----------------------------------------------------------------------
    // SemiringMatrix tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_matrix_construction() {
        let m = SemiringMatrix::<CountingSemiring>::zeros(3, 4);
        assert_eq!(m.rows, 3);
        assert_eq!(m.cols, 4);
        assert_eq!(m.data.len(), 12);
        assert!(m.data.iter().all(|x| x.is_zero()));

        let id = SemiringMatrix::<CountingSemiring>::identity(3);
        assert_eq!(id.get(0, 0).unwrap(), &CountingSemiring::one());
        assert_eq!(id.get(0, 1).unwrap(), &CountingSemiring::zero());
        assert_eq!(id.get(1, 1).unwrap(), &CountingSemiring::one());
    }

    #[test]
    fn test_matrix_from_vec() {
        let m = SemiringMatrix::from_vec(
            2,
            2,
            vec![
                CountingSemiring::new(1),
                CountingSemiring::new(2),
                CountingSemiring::new(3),
                CountingSemiring::new(4),
            ],
        )
        .unwrap();
        assert_eq!(m.get(0, 0).unwrap().value, 1);
        assert_eq!(m.get(0, 1).unwrap().value, 2);
        assert_eq!(m.get(1, 0).unwrap().value, 3);
        assert_eq!(m.get(1, 1).unwrap().value, 4);
    }

    #[test]
    fn test_matrix_from_vec_wrong_size() {
        let result = SemiringMatrix::from_vec(
            2,
            2,
            vec![
                CountingSemiring::new(1),
                CountingSemiring::new(2),
                CountingSemiring::new(3),
            ],
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_matrix_addition() {
        let a = SemiringMatrix::from_vec(
            2,
            2,
            vec![
                CountingSemiring::new(1),
                CountingSemiring::new(2),
                CountingSemiring::new(3),
                CountingSemiring::new(4),
            ],
        )
        .unwrap();
        let b = SemiringMatrix::from_vec(
            2,
            2,
            vec![
                CountingSemiring::new(5),
                CountingSemiring::new(6),
                CountingSemiring::new(7),
                CountingSemiring::new(8),
            ],
        )
        .unwrap();
        let c = a.add(&b).unwrap();
        assert_eq!(c.get(0, 0).unwrap().value, 6);
        assert_eq!(c.get(0, 1).unwrap().value, 8);
        assert_eq!(c.get(1, 0).unwrap().value, 10);
        assert_eq!(c.get(1, 1).unwrap().value, 12);
    }

    #[test]
    fn test_matrix_multiplication() {
        // [1 2] * [5 6] = [1*5+2*7, 1*6+2*8] = [19, 22]
        // [3 4]   [7 8]   [3*5+4*7, 3*6+4*8]   [43, 50]
        let a = SemiringMatrix::from_vec(
            2,
            2,
            vec![
                CountingSemiring::new(1),
                CountingSemiring::new(2),
                CountingSemiring::new(3),
                CountingSemiring::new(4),
            ],
        )
        .unwrap();
        let b = SemiringMatrix::from_vec(
            2,
            2,
            vec![
                CountingSemiring::new(5),
                CountingSemiring::new(6),
                CountingSemiring::new(7),
                CountingSemiring::new(8),
            ],
        )
        .unwrap();
        let c = a.mul(&b).unwrap();
        assert_eq!(c.get(0, 0).unwrap().value, 19);
        assert_eq!(c.get(0, 1).unwrap().value, 22);
        assert_eq!(c.get(1, 0).unwrap().value, 43);
        assert_eq!(c.get(1, 1).unwrap().value, 50);
    }

    #[test]
    fn test_matrix_dimension_mismatch() {
        let a = SemiringMatrix::<CountingSemiring>::zeros(2, 3);
        let b = SemiringMatrix::<CountingSemiring>::zeros(2, 3);
        assert!(a.mul(&b).is_err());
    }

    #[test]
    fn test_matrix_transpose() {
        let m = SemiringMatrix::from_vec(
            2,
            3,
            vec![
                CountingSemiring::new(1),
                CountingSemiring::new(2),
                CountingSemiring::new(3),
                CountingSemiring::new(4),
                CountingSemiring::new(5),
                CountingSemiring::new(6),
            ],
        )
        .unwrap();
        let t = m.transpose();
        assert_eq!(t.rows, 3);
        assert_eq!(t.cols, 2);
        assert_eq!(t.get(0, 0).unwrap().value, 1);
        assert_eq!(t.get(1, 0).unwrap().value, 2);
        assert_eq!(t.get(2, 0).unwrap().value, 3);
        assert_eq!(t.get(0, 1).unwrap().value, 4);
        assert_eq!(t.get(1, 1).unwrap().value, 5);
        assert_eq!(t.get(2, 1).unwrap().value, 6);
    }

    #[test]
    fn test_matrix_identity_mul() {
        let a = SemiringMatrix::from_vec(
            2,
            2,
            vec![
                CountingSemiring::new(1),
                CountingSemiring::new(2),
                CountingSemiring::new(3),
                CountingSemiring::new(4),
            ],
        )
        .unwrap();
        let id = SemiringMatrix::<CountingSemiring>::identity(2);
        assert_eq!(a.mul(&id).unwrap(), a);
        assert_eq!(id.mul(&a).unwrap(), a);
    }

    #[test]
    fn test_matrix_pow() {
        let a = SemiringMatrix::from_vec(
            2,
            2,
            vec![
                CountingSemiring::new(1),
                CountingSemiring::new(1),
                CountingSemiring::new(1),
                CountingSemiring::new(0),
            ],
        )
        .unwrap();
        let a2 = a.mat_pow(2).unwrap();
        let expected = a.mul(&a).unwrap();
        assert_eq!(a2, expected);

        let a0 = a.mat_pow(0).unwrap();
        assert_eq!(a0, SemiringMatrix::<CountingSemiring>::identity(2));
    }

    #[test]
    fn test_matrix_trace() {
        let m = SemiringMatrix::from_vec(
            3,
            3,
            vec![
                CountingSemiring::new(1),
                CountingSemiring::new(0),
                CountingSemiring::new(0),
                CountingSemiring::new(0),
                CountingSemiring::new(2),
                CountingSemiring::new(0),
                CountingSemiring::new(0),
                CountingSemiring::new(0),
                CountingSemiring::new(3),
            ],
        )
        .unwrap();
        assert_eq!(m.trace().unwrap(), CountingSemiring::new(6));
    }

    #[test]
    fn test_matrix_kronecker() {
        let a = SemiringMatrix::from_vec(
            2,
            2,
            vec![
                CountingSemiring::new(1),
                CountingSemiring::new(2),
                CountingSemiring::new(3),
                CountingSemiring::new(4),
            ],
        )
        .unwrap();
        let b = SemiringMatrix::<CountingSemiring>::identity(2);
        let k = a.kronecker(&b);
        assert_eq!(k.rows, 4);
        assert_eq!(k.cols, 4);
        // Top-left 2x2 block should be 1*I = I
        assert_eq!(k.get(0, 0).unwrap().value, 1);
        assert_eq!(k.get(0, 1).unwrap().value, 0);
        assert_eq!(k.get(1, 0).unwrap().value, 0);
        assert_eq!(k.get(1, 1).unwrap().value, 1);
        // Top-right 2x2 block should be 2*I
        assert_eq!(k.get(0, 2).unwrap().value, 2);
        assert_eq!(k.get(0, 3).unwrap().value, 0);
    }

    #[test]
    fn test_matrix_star_boolean() {
        // Adjacency matrix for a simple 3-node graph: 0→1, 1→2
        let mut m = SemiringMatrix::<BooleanSemiring>::zeros(3, 3);
        m.set(0, 1, BooleanSemiring::new(true)).unwrap();
        m.set(1, 2, BooleanSemiring::new(true)).unwrap();

        let star = m.matrix_star().unwrap();
        // star[0][0] should be true (identity path)
        assert!(star.get(0, 0).unwrap().value);
        // star[0][2] should be true (path 0→1→2)
        assert!(star.get(0, 2).unwrap().value);
        // star[2][0] should be false (no path 2→0)
        assert!(!star.get(2, 0).unwrap().value);
    }

    #[test]
    fn test_matrix_star_tropical() {
        // All-pairs shortest paths
        // Graph: 0→1 cost 2, 1→2 cost 3, 0→2 cost 10
        let inf = f64::INFINITY;
        let m = SemiringMatrix::from_vec(
            3,
            3,
            vec![
                TropicalSemiring::new(inf),
                TropicalSemiring::new(2.0),
                TropicalSemiring::new(10.0),
                TropicalSemiring::new(inf),
                TropicalSemiring::new(inf),
                TropicalSemiring::new(3.0),
                TropicalSemiring::new(inf),
                TropicalSemiring::new(inf),
                TropicalSemiring::new(inf),
            ],
        )
        .unwrap();

        let star = m.matrix_star().unwrap();
        // 0→0: 0 (identity)
        assert_eq!(star.get(0, 0).unwrap().raw(), 0.0);
        // 0→1: 2
        assert_eq!(star.get(0, 1).unwrap().raw(), 2.0);
        // 0→2: min(10, 2+3) = 5
        assert_eq!(star.get(0, 2).unwrap().raw(), 5.0);
    }

    #[test]
    fn test_matrix_row_col() {
        let m = SemiringMatrix::from_vec(
            2,
            3,
            vec![
                CountingSemiring::new(1),
                CountingSemiring::new(2),
                CountingSemiring::new(3),
                CountingSemiring::new(4),
                CountingSemiring::new(5),
                CountingSemiring::new(6),
            ],
        )
        .unwrap();
        let row0 = m.row(0).unwrap();
        assert_eq!(row0.len(), 3);
        assert_eq!(row0[0].value, 1);
        assert_eq!(row0[2].value, 3);

        let col1 = m.col(1).unwrap();
        assert_eq!(col1.len(), 2);
        assert_eq!(col1[0].value, 2);
        assert_eq!(col1[1].value, 5);
    }

    #[test]
    fn test_matrix_submatrix() {
        let m = SemiringMatrix::from_vec(
            3,
            3,
            vec![
                CountingSemiring::new(1),
                CountingSemiring::new(2),
                CountingSemiring::new(3),
                CountingSemiring::new(4),
                CountingSemiring::new(5),
                CountingSemiring::new(6),
                CountingSemiring::new(7),
                CountingSemiring::new(8),
                CountingSemiring::new(9),
            ],
        )
        .unwrap();
        let sub = m.submatrix(0, 0, 2, 2).unwrap();
        assert_eq!(sub.rows, 2);
        assert_eq!(sub.cols, 2);
        assert_eq!(sub.get(0, 0).unwrap().value, 1);
        assert_eq!(sub.get(1, 1).unwrap().value, 5);
    }

    #[test]
    fn test_matrix_scalar_mul() {
        let m = SemiringMatrix::from_vec(
            2,
            2,
            vec![
                CountingSemiring::new(1),
                CountingSemiring::new(2),
                CountingSemiring::new(3),
                CountingSemiring::new(4),
            ],
        )
        .unwrap();
        let scaled = m.scalar_mul(&CountingSemiring::new(3));
        assert_eq!(scaled.get(0, 0).unwrap().value, 3);
        assert_eq!(scaled.get(1, 1).unwrap().value, 12);
    }

    #[test]
    fn test_matrix_diagonal() {
        let diag = SemiringMatrix::diagonal(&[
            CountingSemiring::new(2),
            CountingSemiring::new(3),
            CountingSemiring::new(5),
        ]);
        assert_eq!(diag.get(0, 0).unwrap().value, 2);
        assert_eq!(diag.get(1, 1).unwrap().value, 3);
        assert_eq!(diag.get(2, 2).unwrap().value, 5);
        assert_eq!(diag.get(0, 1).unwrap().value, 0);
    }

    // -----------------------------------------------------------------------
    // SemiringPolynomial tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_polynomial_construction() {
        let p = SemiringPolynomial::new(vec![
            CountingSemiring::new(1),
            CountingSemiring::new(2),
            CountingSemiring::new(3),
        ]);
        assert_eq!(p.degree(), Some(2));
        assert_eq!(p.leading_coeff().unwrap().value, 3);
        assert_eq!(p.coeff(0).value, 1);
        assert_eq!(p.coeff(1).value, 2);
        assert_eq!(p.coeff(5).value, 0);
    }

    #[test]
    fn test_polynomial_zero() {
        let z = SemiringPolynomial::<CountingSemiring>::zero();
        assert!(z.is_zero());
        assert_eq!(z.degree(), None);
    }

    #[test]
    fn test_polynomial_normalize() {
        let p = SemiringPolynomial::new(vec![
            CountingSemiring::new(1),
            CountingSemiring::new(0),
            CountingSemiring::new(0),
        ]);
        assert_eq!(p.degree(), Some(0));
        assert_eq!(p.coeffs.len(), 1);
    }

    #[test]
    fn test_polynomial_addition() {
        // (1 + 2x) + (3 + 4x + 5x^2) = (4 + 6x + 5x^2)
        let a = SemiringPolynomial::new(vec![
            CountingSemiring::new(1),
            CountingSemiring::new(2),
        ]);
        let b = SemiringPolynomial::new(vec![
            CountingSemiring::new(3),
            CountingSemiring::new(4),
            CountingSemiring::new(5),
        ]);
        let c = a.add(&b);
        assert_eq!(c.coeff(0).value, 4);
        assert_eq!(c.coeff(1).value, 6);
        assert_eq!(c.coeff(2).value, 5);
    }

    #[test]
    fn test_polynomial_multiplication() {
        // (1 + 2x) * (3 + 4x) = 3 + (4+6)x + 8x^2 = 3 + 10x + 8x^2
        let a = SemiringPolynomial::new(vec![
            CountingSemiring::new(1),
            CountingSemiring::new(2),
        ]);
        let b = SemiringPolynomial::new(vec![
            CountingSemiring::new(3),
            CountingSemiring::new(4),
        ]);
        let c = a.mul(&b);
        assert_eq!(c.coeff(0).value, 3);
        assert_eq!(c.coeff(1).value, 10);
        assert_eq!(c.coeff(2).value, 8);
    }

    #[test]
    fn test_polynomial_evaluation() {
        // p(x) = 1 + 2x + 3x^2, evaluate at x=2: 1 + 4 + 12 = 17
        let p = SemiringPolynomial::new(vec![
            CountingSemiring::new(1),
            CountingSemiring::new(2),
            CountingSemiring::new(3),
        ]);
        let result = p.evaluate(&CountingSemiring::new(2));
        assert_eq!(result.value, 17);
    }

    #[test]
    fn test_polynomial_evaluation_zero() {
        let p = SemiringPolynomial::new(vec![
            CountingSemiring::new(5),
            CountingSemiring::new(3),
            CountingSemiring::new(7),
        ]);
        let result = p.evaluate(&CountingSemiring::zero());
        assert_eq!(result.value, 5);
    }

    #[test]
    fn test_polynomial_monomial() {
        let m = SemiringPolynomial::monomial(CountingSemiring::new(5), 3);
        assert_eq!(m.degree(), Some(3));
        assert_eq!(m.coeff(0).value, 0);
        assert_eq!(m.coeff(3).value, 5);
    }

    #[test]
    fn test_polynomial_pow() {
        // (1 + x)^2 = 1 + 2x + x^2
        let p = SemiringPolynomial::new(vec![
            CountingSemiring::new(1),
            CountingSemiring::new(1),
        ]);
        let p2 = p.pow(2);
        assert_eq!(p2.coeff(0).value, 1);
        assert_eq!(p2.coeff(1).value, 2);
        assert_eq!(p2.coeff(2).value, 1);
    }

    #[test]
    fn test_polynomial_formal_derivative() {
        // p(x) = 3 + 2x + 5x^2, p'(x) = 2 + 10x
        let p = SemiringPolynomial::new(vec![
            CountingSemiring::new(3),
            CountingSemiring::new(2),
            CountingSemiring::new(5),
        ]);
        let dp = p.formal_derivative();
        assert_eq!(dp.coeff(0).value, 2); // 1 * 2
        assert_eq!(dp.coeff(1).value, 10); // 2 * 5
        assert_eq!(dp.degree(), Some(1));
    }

    #[test]
    fn test_polynomial_compose() {
        // p(x) = x^2, q(x) = 2x + 1
        // p(q(x)) = (2x+1)^2 = 4x^2 + 4x + 1
        let p = SemiringPolynomial::new(vec![
            CountingSemiring::new(0),
            CountingSemiring::new(0),
            CountingSemiring::new(1),
        ]);
        let q = SemiringPolynomial::new(vec![
            CountingSemiring::new(1),
            CountingSemiring::new(2),
        ]);
        let comp = p.compose(&q);
        assert_eq!(comp.coeff(0).value, 1);
        assert_eq!(comp.coeff(1).value, 4);
        assert_eq!(comp.coeff(2).value, 4);
    }

    #[test]
    fn test_polynomial_tropical() {
        // In tropical semiring: min and + operate on polynomials
        let a = SemiringPolynomial::new(vec![
            TropicalSemiring::new(1.0),
            TropicalSemiring::new(3.0),
        ]);
        let b = SemiringPolynomial::new(vec![
            TropicalSemiring::new(2.0),
            TropicalSemiring::new(1.0),
        ]);
        let sum = a.add(&b);
        // coeff 0: min(1, 2) = 1
        assert_eq!(sum.coeff(0).raw(), 1.0);
        // coeff 1: min(3, 1) = 1
        assert_eq!(sum.coeff(1).raw(), 1.0);
    }

    // -----------------------------------------------------------------------
    // Homomorphism tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_counting_to_boolean() {
        let h = CountingToBoolean;
        assert_eq!(
            h.apply(&CountingSemiring::new(0)),
            BooleanSemiring::new(false)
        );
        assert_eq!(
            h.apply(&CountingSemiring::new(1)),
            BooleanSemiring::new(true)
        );
        assert_eq!(
            h.apply(&CountingSemiring::new(42)),
            BooleanSemiring::new(true)
        );
    }

    #[test]
    fn test_counting_to_boolean_preserves_structure() {
        let h = CountingToBoolean;
        let a = CountingSemiring::new(3);
        let b = CountingSemiring::new(5);

        // h(a ⊕ b) = h(a) ⊕ h(b)
        assert_eq!(h.apply(&a.add(&b)), h.apply(&a).add(&h.apply(&b)));
        // h(a ⊗ b) = h(a) ⊗ h(b)
        assert_eq!(h.apply(&a.mul(&b)), h.apply(&a).mul(&h.apply(&b)));
        // h(0) = 0
        assert_eq!(
            h.apply(&CountingSemiring::zero()),
            BooleanSemiring::zero()
        );
        // h(1) = 1
        assert_eq!(h.apply(&CountingSemiring::one()), BooleanSemiring::one());
    }

    #[test]
    fn test_counting_to_goldilocks() {
        let h = CountingToGoldilocks;
        assert_eq!(
            h.apply(&CountingSemiring::new(42)),
            GoldilocksField::new(42)
        );
        assert_eq!(
            h.apply(&CountingSemiring::zero()),
            GoldilocksField::zero()
        );
        assert_eq!(h.apply(&CountingSemiring::one()), GoldilocksField::one());
    }

    #[test]
    fn test_counting_to_goldilocks_preserves_structure() {
        let h = CountingToGoldilocks;
        let a = CountingSemiring::new(100);
        let b = CountingSemiring::new(200);
        assert_eq!(h.apply(&a.add(&b)), h.apply(&a).add(&h.apply(&b)));
        assert_eq!(h.apply(&a.mul(&b)), h.apply(&a).mul(&h.apply(&b)));
    }

    #[test]
    fn test_boolean_to_goldilocks() {
        let h = BooleanToGoldilocks;
        assert_eq!(
            h.apply(&BooleanSemiring::new(false)),
            GoldilocksField::zero()
        );
        assert_eq!(
            h.apply(&BooleanSemiring::new(true)),
            GoldilocksField::one()
        );
    }

    #[test]
    fn test_boolean_to_counting() {
        let h = BooleanToCounting;
        assert_eq!(
            h.apply(&BooleanSemiring::new(false)),
            CountingSemiring::new(0)
        );
        assert_eq!(
            h.apply(&BooleanSemiring::new(true)),
            CountingSemiring::new(1)
        );
    }

    #[test]
    fn test_real_to_tropical() {
        let h = RealToTropical;
        // Positive real maps to -ln(x)
        let result = h.apply(&RealSemiring::new(1.0));
        assert!((result.raw() - 0.0).abs() < 1e-10);
        // Zero maps to infinity
        let zero_result = h.apply(&RealSemiring::new(0.0));
        assert!(zero_result.is_zero());
    }

    #[test]
    fn test_real_to_log() {
        let h = RealToLog;
        let result = h.apply(&RealSemiring::new(1.0));
        assert!((result.raw() - 0.0).abs() < 1e-10);
        let result_e = h.apply(&RealSemiring::new(std::f64::consts::E));
        assert!((result_e.raw() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_real_to_viterbi() {
        let h = RealToViterbi;
        assert_eq!(h.apply(&RealSemiring::new(0.5)), ViterbiSemiring::new(0.5));
        // Clamped to [0, 1]
        assert_eq!(h.apply(&RealSemiring::new(2.0)), ViterbiSemiring::new(1.0));
        assert_eq!(
            h.apply(&RealSemiring::new(-1.0)),
            ViterbiSemiring::new(0.0)
        );
    }

    #[test]
    fn test_tropical_to_max_plus() {
        let h = TropicalToMaxPlus;
        let result = h.apply(&TropicalSemiring::new(5.0));
        assert_eq!(result.raw(), -5.0);
        let inf_result = h.apply(&TropicalSemiring::infinity());
        assert!(inf_result.is_zero()); // -∞
    }

    #[test]
    fn test_counting_to_real() {
        let h = CountingToReal;
        assert_eq!(h.apply(&CountingSemiring::new(42)), RealSemiring::new(42.0));
        assert_eq!(h.apply(&CountingSemiring::zero()), RealSemiring::zero());
    }

    #[test]
    fn test_identity_homomorphism() {
        let h = IdentityHomomorphism::<CountingSemiring>::new();
        let x = CountingSemiring::new(42);
        assert_eq!(h.apply(&x), x);
    }

    #[test]
    fn test_composed_homomorphism() {
        let h1 = CountingToBoolean;
        let h2 = BooleanToGoldilocks;
        let composed = ComposedHomomorphism::new(h1, h2);

        let zero = CountingSemiring::zero();
        let three = CountingSemiring::new(3);

        assert_eq!(composed.apply(&zero), GoldilocksField::zero());
        assert_eq!(composed.apply(&three), GoldilocksField::one());
    }

    #[test]
    fn test_goldilocks_to_field() {
        let h = GoldilocksToField::<SmallTestPrime>::new();
        let x = GoldilocksField::new(42);
        let result = h.apply(&x);
        assert_eq!(result.value, 42);
    }

    // -----------------------------------------------------------------------
    // Utility function tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_semiring_sum() {
        let values = vec![
            CountingSemiring::new(1),
            CountingSemiring::new(2),
            CountingSemiring::new(3),
        ];
        assert_eq!(semiring_sum(&values), CountingSemiring::new(6));
        assert_eq!(
            semiring_sum::<CountingSemiring>(&[]),
            CountingSemiring::zero()
        );
    }

    #[test]
    fn test_semiring_product() {
        let values = vec![
            CountingSemiring::new(2),
            CountingSemiring::new(3),
            CountingSemiring::new(4),
        ];
        assert_eq!(semiring_product(&values), CountingSemiring::new(24));
        assert_eq!(
            semiring_product::<CountingSemiring>(&[]),
            CountingSemiring::one()
        );
    }

    #[test]
    fn test_semiring_dot() {
        let a = vec![CountingSemiring::new(1), CountingSemiring::new(2)];
        let b = vec![CountingSemiring::new(3), CountingSemiring::new(4)];
        let dot = semiring_dot(&a, &b).unwrap();
        assert_eq!(dot.value, 11); // 1*3 + 2*4
    }

    #[test]
    fn test_semiring_dot_mismatched_lengths() {
        let a = vec![CountingSemiring::new(1)];
        let b = vec![CountingSemiring::new(2), CountingSemiring::new(3)];
        assert!(semiring_dot(&a, &b).is_err());
    }

    #[test]
    fn test_map_matrix() {
        let m = SemiringMatrix::from_vec(
            2,
            2,
            vec![
                CountingSemiring::new(0),
                CountingSemiring::new(3),
                CountingSemiring::new(0),
                CountingSemiring::new(5),
            ],
        )
        .unwrap();
        let h = CountingToBoolean;
        let mapped = map_matrix(&m, &h);
        assert_eq!(mapped.get(0, 0).unwrap(), &BooleanSemiring::new(false));
        assert_eq!(mapped.get(0, 1).unwrap(), &BooleanSemiring::new(true));
        assert_eq!(mapped.get(1, 0).unwrap(), &BooleanSemiring::new(false));
        assert_eq!(mapped.get(1, 1).unwrap(), &BooleanSemiring::new(true));
    }

    #[test]
    fn test_map_polynomial() {
        let p = SemiringPolynomial::new(vec![
            CountingSemiring::new(0),
            CountingSemiring::new(5),
            CountingSemiring::new(3),
        ]);
        let h = CountingToBoolean;
        let mapped = map_polynomial(&p, &h);
        assert_eq!(mapped.coeff(0), BooleanSemiring::new(false));
        assert_eq!(mapped.coeff(1), BooleanSemiring::new(true));
        assert_eq!(mapped.coeff(2), BooleanSemiring::new(true));
    }

    // -----------------------------------------------------------------------
    // Edge case / stress tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_matrix_empty() {
        let m = SemiringMatrix::<CountingSemiring>::zeros(0, 0);
        assert_eq!(m.rows, 0);
        assert_eq!(m.cols, 0);
    }

    #[test]
    fn test_matrix_1x1() {
        let s = SemiringMatrix::scalar(CountingSemiring::new(42));
        assert_eq!(s.get(0, 0).unwrap().value, 42);
        let sq = s.mul(&s).unwrap();
        assert_eq!(sq.get(0, 0).unwrap().value, 42 * 42);
    }

    #[test]
    fn test_polynomial_zero_mul() {
        let p = SemiringPolynomial::new(vec![
            CountingSemiring::new(1),
            CountingSemiring::new(2),
        ]);
        let z = SemiringPolynomial::<CountingSemiring>::zero();
        assert!(p.mul(&z).is_zero());
        assert!(z.mul(&p).is_zero());
    }

    #[test]
    fn test_polynomial_scalar_mul() {
        let p = SemiringPolynomial::new(vec![
            CountingSemiring::new(1),
            CountingSemiring::new(2),
            CountingSemiring::new(3),
        ]);
        let scaled = p.scalar_mul(&CountingSemiring::new(10));
        assert_eq!(scaled.coeff(0).value, 10);
        assert_eq!(scaled.coeff(1).value, 20);
        assert_eq!(scaled.coeff(2).value, 30);
    }

    #[test]
    fn test_goldilocks_quadratic_residue() {
        let one = GoldilocksField::new(1);
        assert!(one.is_quadratic_residue());
        let four = GoldilocksField::new(4);
        assert!(four.is_quadratic_residue());
        let zero = GoldilocksField::new(0);
        assert!(zero.is_quadratic_residue());
    }

    #[test]
    fn test_tropical_semiring_with_matrix() {
        // Floyd-Warshall shortest paths via matrix star
        let inf = f64::INFINITY;
        // 3 nodes, edges: 0→1:1, 1→2:2, 0→2:5
        let m = SemiringMatrix::from_vec(
            3,
            3,
            vec![
                TropicalSemiring::new(inf),
                TropicalSemiring::new(1.0),
                TropicalSemiring::new(5.0),
                TropicalSemiring::new(inf),
                TropicalSemiring::new(inf),
                TropicalSemiring::new(2.0),
                TropicalSemiring::new(inf),
                TropicalSemiring::new(inf),
                TropicalSemiring::new(inf),
            ],
        )
        .unwrap();

        let star = m.matrix_star().unwrap();
        // Shortest 0→2: min(5, 1+2) = 3
        assert_eq!(star.get(0, 2).unwrap().raw(), 3.0);
        // Diagonal should be 0
        assert_eq!(star.get(0, 0).unwrap().raw(), 0.0);
        assert_eq!(star.get(1, 1).unwrap().raw(), 0.0);
        assert_eq!(star.get(2, 2).unwrap().raw(), 0.0);
    }

    #[test]
    fn test_goldilocks_consistency_with_field_semiring() {
        // Goldilocks and FieldSemiring with the same prime should agree
        // (We can't easily parameterize by the Goldilocks prime since it's near u64::MAX,
        //  so we just test that Goldilocks arithmetic is self-consistent.)
        let a = GoldilocksField::new(123456789);
        let b = GoldilocksField::new(987654321);
        let sum = a.add(&b);
        let prod = a.mul(&b);
        // Verify commutativity
        assert_eq!(sum, b.add(&a));
        assert_eq!(prod, b.mul(&a));
        // Verify distributivity
        let c = GoldilocksField::new(555555555);
        assert_eq!(a.mul(&b.add(&c)), a.mul(&b).add(&a.mul(&c)));
    }

    #[test]
    fn test_add_assign_and_mul_assign() {
        let mut a = CountingSemiring::new(3);
        a.add_assign(&CountingSemiring::new(5));
        assert_eq!(a.value, 8);
        a.mul_assign(&CountingSemiring::new(2));
        assert_eq!(a.value, 16);
    }

    #[test]
    fn test_display_methods() {
        assert_eq!(CountingSemiring::new(42).display(), "42");
        assert_eq!(BooleanSemiring::new(true).display(), "1");
        assert_eq!(BooleanSemiring::new(false).display(), "0");
        assert_eq!(GoldilocksField::new(100).display(), "100");
        assert_eq!(TropicalSemiring::new(3.5).display(), "3.5");
    }

    #[test]
    fn test_expectation_semiring_identity() {
        type E = ExpectationSemiring<RealSemiring>;
        let one = E::one();
        let a = E::new(RealSemiring::new(2.0), RealSemiring::new(3.0));
        assert_eq!(a.mul(&one), a);
        assert_eq!(one.mul(&a), a);
    }

    #[test]
    fn test_product_semiring_star() {
        type P = ProductSemiring<BooleanSemiring, BooleanSemiring>;
        let a = P::new(BooleanSemiring::new(true), BooleanSemiring::new(false));
        let star = a.star();
        assert_eq!(star, P::one());
    }

    #[test]
    fn test_matrix_non_square_pow_error() {
        let m = SemiringMatrix::<CountingSemiring>::zeros(2, 3);
        assert!(m.mat_pow(2).is_err());
    }

    #[test]
    fn test_matrix_trace_non_square_error() {
        let m = SemiringMatrix::<CountingSemiring>::zeros(2, 3);
        assert!(m.trace().is_err());
    }

    #[test]
    fn test_polynomial_x() {
        let x = SemiringPolynomial::<CountingSemiring>::x();
        assert_eq!(x.degree(), Some(1));
        assert_eq!(x.coeff(0), CountingSemiring::zero());
        assert_eq!(x.coeff(1), CountingSemiring::one());
        // Evaluate x at 5 => 5
        assert_eq!(x.evaluate(&CountingSemiring::new(5)), CountingSemiring::new(5));
    }

    #[test]
    fn test_polynomial_constant() {
        let c = SemiringPolynomial::constant(CountingSemiring::new(7));
        assert_eq!(c.degree(), Some(0));
        assert_eq!(c.evaluate(&CountingSemiring::new(999)), CountingSemiring::new(7));
    }

    #[test]
    fn test_min_max_star() {
        let a = MinMaxSemiring::new(42.0);
        assert_eq!(a.star(), MinMaxSemiring::one());
    }

    #[test]
    fn test_log_semiring_star_convergent() {
        let a = LogSemiring::new(-1.0);
        let star = a.star();
        // Should converge since exp(-1) < 1
        let expected = -(1.0 - (-1.0f64).exp()).ln();
        assert!((star.raw() - expected).abs() < 1e-10);
    }

    #[test]
    fn test_matrix_ops_tropical() {
        let a = SemiringMatrix::from_vec(
            2,
            2,
            vec![
                TropicalSemiring::new(1.0),
                TropicalSemiring::new(2.0),
                TropicalSemiring::new(3.0),
                TropicalSemiring::new(4.0),
            ],
        )
        .unwrap();
        let b = SemiringMatrix::from_vec(
            2,
            2,
            vec![
                TropicalSemiring::new(5.0),
                TropicalSemiring::new(6.0),
                TropicalSemiring::new(7.0),
                TropicalSemiring::new(8.0),
            ],
        )
        .unwrap();
        let sum = a.add(&b).unwrap();
        // min element-wise
        assert_eq!(sum.get(0, 0).unwrap().raw(), 1.0);
        assert_eq!(sum.get(0, 1).unwrap().raw(), 2.0);

        let prod = a.mul(&b).unwrap();
        // prod[0][0] = min(1+5, 2+7) = min(6, 9) = 6
        assert_eq!(prod.get(0, 0).unwrap().raw(), 6.0);
        // prod[0][1] = min(1+6, 2+8) = min(7, 10) = 7
        assert_eq!(prod.get(0, 1).unwrap().raw(), 7.0);
    }

    #[test]
    fn test_goldilocks_edge_p_minus_1() {
        let pm1 = GoldilocksField::from_canonical(GOLDILOCKS_PRIME - 1);
        let one = GoldilocksField::one();
        let sum = pm1.add(&one);
        assert_eq!(sum, GoldilocksField::zero());
    }

    #[test]
    fn test_from_rows() {
        let m = SemiringMatrix::from_rows(vec![
            vec![CountingSemiring::new(1), CountingSemiring::new(2)],
            vec![CountingSemiring::new(3), CountingSemiring::new(4)],
        ])
        .unwrap();
        assert_eq!(m.rows, 2);
        assert_eq!(m.cols, 2);
        assert_eq!(m.get(1, 0).unwrap().value, 3);
    }

    #[test]
    fn test_from_rows_ragged() {
        let result = SemiringMatrix::from_rows(vec![
            vec![CountingSemiring::new(1), CountingSemiring::new(2)],
            vec![CountingSemiring::new(3)],
        ]);
        assert!(result.is_err());
    }

    #[test]
    fn test_matrix_set() {
        let mut m = SemiringMatrix::<CountingSemiring>::zeros(2, 2);
        m.set(0, 1, CountingSemiring::new(42)).unwrap();
        assert_eq!(m.get(0, 1).unwrap().value, 42);
    }

    #[test]
    fn test_bounded_counting_zero_bound() {
        let a = BoundedCountingSemiring::new(5, 0);
        assert_eq!(a.value, 0);
        let b = BoundedCountingSemiring::new(3, 0);
        assert_eq!(a.add(&b).value, 0);
    }

    #[test]
    fn test_free_monoid_display() {
        let s = FreeMonoidSemiring::new(vec!["hello".into(), "world".into()]);
        let d = format!("{}", s);
        assert!(d.contains("hello"));
        assert!(d.contains("world"));
    }

    #[test]
    fn test_free_monoid_empty_string_display() {
        let s = FreeMonoidSemiring::one();
        let d = format!("{}", s);
        assert!(d.contains("ε"));
    }
}
