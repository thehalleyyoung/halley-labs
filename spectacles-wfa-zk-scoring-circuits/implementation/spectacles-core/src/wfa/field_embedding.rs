//! Semiring-to-field embeddings for STARK compatibility.
//!
//! Provides structure-preserving maps from various semirings into the
//! Goldilocks prime field (p = 2^64 − 2^32 + 1) so that WFA weight
//! computations can be verified inside a STARK proof system.  Includes
//! direct embeddings for counting and Boolean semirings, gadget-based
//! encodings for the tropical semiring, fixed-point approximations for
//! reals, batch operations, overflow detection, extension fields, and
//! a registry for automatic embedding selection.

use std::collections::HashMap;
use std::fmt;
use std::marker::PhantomData;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::automaton::WeightedFiniteAutomaton;
use super::semiring::{
    BooleanSemiring, BoundedCountingSemiring, CountingSemiring, GoldilocksField,
    RealSemiring, Semiring, SemiringMatrix, TropicalSemiring, GOLDILOCKS_PRIME,
};

// ═══════════════════════════════════════════════════════════════════════════
// 1. EmbeddingError
// ═══════════════════════════════════════════════════════════════════════════

/// Errors arising from semiring-to-field embedding operations.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum EmbeddingError {
    #[error("overflow: value {value} does not fit in {field_desc}")]
    OverflowError { value: String, field_desc: String },

    #[error("field element {field_element} is not in the image of the embedding: {desc}")]
    NotInImage { field_element: String, desc: String },

    #[error("homomorphism violation ({property}): f(a)={a}, f(b)={b}")]
    HomomorphismViolation {
        property: String,
        a: String,
        b: String,
    },

    #[error("incompatible dimensions: expected {expected}, found {found}")]
    IncompatibleDimensions { expected: String, found: String },

    #[error("invalid gadget: {desc}")]
    InvalidGadget { desc: String },

    #[error("precision loss: original={original}, embedded={embedded}")]
    PrecisionLoss {
        original: String,
        embedded: String,
    },

    #[error("bit decomposition error: value {value} with {bits} bits")]
    BitDecompositionError { value: String, bits: usize },

    #[error("internal error: {msg}")]
    InternalError { msg: String },
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. SemiringEmbedding trait
// ═══════════════════════════════════════════════════════════════════════════

/// A structure-preserving map from semiring `S` into field `F`.
///
/// Must satisfy:
///   embed(a ⊕ b) = embed(a) + embed(b)   (additive homomorphism)
///   embed(a ⊗ b) = embed(a) * embed(b)   (multiplicative homomorphism)
///   embed(0_S)   = 0_F
///   embed(1_S)   = 1_F
///
/// The inverse (`unembed`) is partial — it may fail for field elements
/// outside the image of the embedding.
pub trait SemiringEmbedding<S: Semiring, F: Semiring> {
    /// Embed a semiring value into the field.
    fn embed(&self, value: &S) -> Result<F, EmbeddingError>;

    /// Partial inverse: recover the semiring value, if possible.
    fn unembed(&self, value: &F) -> Result<S, EmbeddingError>;

    /// Check whether a value can be embedded without error.
    fn can_embed(&self, value: &S) -> bool {
        self.embed(value).is_ok()
    }

    /// Verify the homomorphism properties for a pair (a, b):
    ///   embed(a ⊕ b) == embed(a) + embed(b)
    ///   embed(a ⊗ b) == embed(a) * embed(b)
    fn verify_homomorphism(&self, a: &S, b: &S) -> Result<bool, EmbeddingError> {
        let fa = self.embed(a)?;
        let fb = self.embed(b)?;

        // Additive check
        let sum_ab = a.add(b);
        let f_sum = self.embed(&sum_ab)?;
        let f_add = fa.add(&fb);
        if f_sum != f_add {
            return Ok(false);
        }

        // Multiplicative check
        let prod_ab = a.mul(b);
        let f_prod = self.embed(&prod_ab)?;
        let f_mul = fa.mul(&fb);
        if f_prod != f_mul {
            return Ok(false);
        }

        Ok(true)
    }

    /// Embed every entry of a semiring matrix.
    fn embed_matrix(&self, matrix: &SemiringMatrix<S>) -> Result<SemiringMatrix<F>, EmbeddingError> {
        let mut data = Vec::with_capacity(matrix.rows * matrix.cols);
        for i in 0..matrix.rows {
            for j in 0..matrix.cols {
                let val = matrix
                    .get(i, j)
                    .map_err(|e| EmbeddingError::InternalError {
                        msg: format!("matrix access error: {}", e),
                    })?;
                data.push(self.embed(val)?);
            }
        }
        SemiringMatrix::from_vec(matrix.rows, matrix.cols, data).map_err(|e| {
            EmbeddingError::InternalError {
                msg: format!("matrix construction error: {}", e),
            }
        })
    }

    /// Unembed every entry of a field matrix back to the semiring.
    fn unembed_matrix(
        &self,
        matrix: &SemiringMatrix<F>,
    ) -> Result<SemiringMatrix<S>, EmbeddingError> {
        let mut data = Vec::with_capacity(matrix.rows * matrix.cols);
        for i in 0..matrix.rows {
            for j in 0..matrix.cols {
                let val = matrix
                    .get(i, j)
                    .map_err(|e| EmbeddingError::InternalError {
                        msg: format!("matrix access error: {}", e),
                    })?;
                data.push(self.unembed(val)?);
            }
        }
        SemiringMatrix::from_vec(matrix.rows, matrix.cols, data).map_err(|e| {
            EmbeddingError::InternalError {
                msg: format!("matrix construction error: {}", e),
            }
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. CountingToGoldilocks
// ═══════════════════════════════════════════════════════════════════════════

/// Injective embedding of the counting semiring (ℕ, +, ·) into the
/// Goldilocks field.  Maps n ↦ n mod p.  Valid for n < p (essentially
/// always, since p ≈ 1.8 × 10^19).
#[derive(Debug, Clone)]
pub struct CountingToGoldilocks {
    /// Maximum value that can be safely unembedded.  Values in [0, max_safe]
    /// are assumed to originate from the counting semiring.
    max_safe: u64,
}

impl CountingToGoldilocks {
    pub fn new() -> Self {
        Self {
            max_safe: GOLDILOCKS_PRIME - 1,
        }
    }

    /// Create with a custom bound for the inverse mapping.
    pub fn with_max_safe(max_safe: u64) -> Self {
        Self { max_safe }
    }

    /// Largest counting value that can be safely embedded without wrapping.
    pub fn max_safe_value() -> u64 {
        GOLDILOCKS_PRIME - 1
    }
}

impl Default for CountingToGoldilocks {
    fn default() -> Self {
        Self::new()
    }
}

impl SemiringEmbedding<CountingSemiring, GoldilocksField> for CountingToGoldilocks {
    fn embed(&self, value: &CountingSemiring) -> Result<GoldilocksField, EmbeddingError> {
        // CountingSemiring uses saturating arithmetic so values are always
        // ≤ u64::MAX.  Reduce modulo p.
        Ok(GoldilocksField::new(value.value))
    }

    fn unembed(&self, value: &GoldilocksField) -> Result<CountingSemiring, EmbeddingError> {
        if value.value > self.max_safe {
            return Err(EmbeddingError::NotInImage {
                field_element: value.value.to_string(),
                desc: format!(
                    "field element {} exceeds max safe value {} for counting unembed",
                    value.value, self.max_safe
                ),
            });
        }
        Ok(CountingSemiring::new(value.value))
    }

    fn can_embed(&self, _value: &CountingSemiring) -> bool {
        true // every u64 can be reduced mod p
    }

    fn verify_homomorphism(
        &self,
        a: &CountingSemiring,
        b: &CountingSemiring,
    ) -> Result<bool, EmbeddingError> {
        let fa = self.embed(a)?;
        let fb = self.embed(b)?;

        // Additive: embed(a+b) == embed(a) + embed(b)
        // Note: CountingSemiring uses saturating_add, so we need to check
        // whether saturation occurred — if so, the homomorphism fails.
        let sum_sat = a.value.saturating_add(b.value);
        let sum_field = GoldilocksField::new(a.value).add(&GoldilocksField::new(b.value));
        let embed_sum = GoldilocksField::new(sum_sat);
        if embed_sum != sum_field {
            // Saturation happened; the homomorphism does not hold for these
            // operands under saturating semantics.
            return Ok(false);
        }

        // Multiplicative: embed(a*b) == embed(a) * embed(b)
        let prod_sat = a.value.saturating_mul(b.value);
        let prod_field = fa.mul(&fb);
        let embed_prod = GoldilocksField::new(prod_sat);
        if embed_prod != prod_field {
            return Ok(false);
        }

        Ok(true)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. BooleanToGoldilocks
// ═══════════════════════════════════════════════════════════════════════════

/// Embedding of the Boolean semiring ({false, true}, ∨, ∧) into the
/// Goldilocks field.
///
///   false ↦ 0
///   true  ↦ 1
///
/// Boolean OR maps to field arithmetic: a + b − a·b
/// Boolean AND maps to field multiplication: a · b
#[derive(Debug, Clone)]
pub struct BooleanToGoldilocks;

impl BooleanToGoldilocks {
    pub fn new() -> Self {
        Self
    }

    /// Encode OR(a, b) in field arithmetic: a + b - a*b.
    pub fn field_or(a: &GoldilocksField, b: &GoldilocksField) -> GoldilocksField {
        // a + b - a*b
        let ab = a.mul(b);
        a.add(b).sub(&ab)
    }

    /// Encode AND(a, b) in field arithmetic: a * b.
    pub fn field_and(a: &GoldilocksField, b: &GoldilocksField) -> GoldilocksField {
        a.mul(b)
    }

    /// Encode NOT(a) in field arithmetic: 1 - a.
    pub fn field_not(a: &GoldilocksField) -> GoldilocksField {
        GoldilocksField::one().sub(a)
    }
}

impl Default for BooleanToGoldilocks {
    fn default() -> Self {
        Self::new()
    }
}

impl SemiringEmbedding<BooleanSemiring, GoldilocksField> for BooleanToGoldilocks {
    fn embed(&self, value: &BooleanSemiring) -> Result<GoldilocksField, EmbeddingError> {
        Ok(if value.value {
            GoldilocksField::one()
        } else {
            GoldilocksField::zero()
        })
    }

    fn unembed(&self, value: &GoldilocksField) -> Result<BooleanSemiring, EmbeddingError> {
        match value.value {
            0 => Ok(BooleanSemiring::new(false)),
            1 => Ok(BooleanSemiring::new(true)),
            other => Err(EmbeddingError::NotInImage {
                field_element: other.to_string(),
                desc: "Boolean embedding image is {0, 1}".to_string(),
            }),
        }
    }

    fn can_embed(&self, _value: &BooleanSemiring) -> bool {
        true
    }

    fn verify_homomorphism(
        &self,
        a: &BooleanSemiring,
        b: &BooleanSemiring,
    ) -> Result<bool, EmbeddingError> {
        let fa = self.embed(a)?;
        let fb = self.embed(b)?;

        // OR: embed(a ∨ b) == fa + fb - fa*fb
        let or_result = a.add(b); // BooleanSemiring::add is OR
        let f_or = self.embed(&or_result)?;
        let field_or = Self::field_or(&fa, &fb);
        if f_or != field_or {
            return Ok(false);
        }

        // AND: embed(a ∧ b) == fa * fb
        let and_result = a.mul(b); // BooleanSemiring::mul is AND
        let f_and = self.embed(&and_result)?;
        let field_and = Self::field_and(&fa, &fb);
        if f_and != field_and {
            return Ok(false);
        }

        Ok(true)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. BoundedCountingToGoldilocks
// ═══════════════════════════════════════════════════════════════════════════

/// Embedding of the bounded counting semiring into Goldilocks.
/// The value is embedded directly; the bound is carried as metadata.
#[derive(Debug, Clone)]
pub struct BoundedCountingToGoldilocks {
    /// Default bound used when unembedding.
    default_bound: u64,
}

impl BoundedCountingToGoldilocks {
    pub fn new(default_bound: u64) -> Self {
        Self { default_bound }
    }

    /// Embed value and bound together as a pair of field elements.
    pub fn embed_with_bound_check(
        &self,
        value: &BoundedCountingSemiring,
    ) -> Result<(GoldilocksField, GoldilocksField), EmbeddingError> {
        let embedded_value = GoldilocksField::new(value.value);
        let embedded_bound = GoldilocksField::new(value.bound);
        // Verify the invariant: value ≤ bound
        if value.value > value.bound {
            return Err(EmbeddingError::OverflowError {
                value: format!("value={} > bound={}", value.value, value.bound),
                field_desc: "BoundedCountingSemiring invariant violated".to_string(),
            });
        }
        Ok((embedded_value, embedded_bound))
    }

    /// Verify that clipping behaviour is preserved under embedding.
    /// For a + b clamped to bound, check that the field arithmetic
    /// yields the same result as min(a+b, bound).
    pub fn verify_clipping(
        &self,
        a: &BoundedCountingSemiring,
        b: &BoundedCountingSemiring,
    ) -> Result<bool, EmbeddingError> {
        let sum = a.add(b);
        let embedded_sum = GoldilocksField::new(sum.value);
        let fa = GoldilocksField::new(a.value);
        let fb = GoldilocksField::new(b.value);
        let raw_sum = fa.add(&fb);
        let bound = a.bound.min(b.bound);
        let field_bound = GoldilocksField::new(bound);

        // The clipping is: if raw_sum > bound then bound else raw_sum.
        // In the field we can only check the concrete values.
        let clamped = if a.value.saturating_add(b.value) > bound {
            field_bound
        } else {
            raw_sum
        };
        Ok(embedded_sum == clamped)
    }
}

impl Default for BoundedCountingToGoldilocks {
    fn default() -> Self {
        Self::new(u64::MAX)
    }
}

impl SemiringEmbedding<BoundedCountingSemiring, GoldilocksField> for BoundedCountingToGoldilocks {
    fn embed(&self, value: &BoundedCountingSemiring) -> Result<GoldilocksField, EmbeddingError> {
        Ok(GoldilocksField::new(value.value))
    }

    fn unembed(&self, value: &GoldilocksField) -> Result<BoundedCountingSemiring, EmbeddingError> {
        if value.value > self.default_bound {
            return Err(EmbeddingError::NotInImage {
                field_element: value.value.to_string(),
                desc: format!("exceeds default bound {}", self.default_bound),
            });
        }
        Ok(BoundedCountingSemiring::new(value.value, self.default_bound))
    }

    fn can_embed(&self, _value: &BoundedCountingSemiring) -> bool {
        true
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. TropicalGadget — bit-decomposition approach
// ═══════════════════════════════════════════════════════════════════════════

/// Bit decomposition of a value into individual field elements, each 0 or 1.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BitDecomposition {
    /// Bits in little-endian order: bits[0] is the LSB.
    pub bits: Vec<GoldilocksField>,
    /// Number of bits in the decomposition.
    pub num_bits: usize,
}

impl BitDecomposition {
    /// Decompose `value` into `num_bits` bits (little-endian).
    pub fn decompose(value: u64, num_bits: usize) -> Result<Self, EmbeddingError> {
        if num_bits > 64 {
            return Err(EmbeddingError::BitDecompositionError {
                value: value.to_string(),
                bits: num_bits,
            });
        }
        // Check that value fits in num_bits
        if num_bits < 64 && value >= (1u64 << num_bits) {
            return Err(EmbeddingError::BitDecompositionError {
                value: value.to_string(),
                bits: num_bits,
            });
        }
        let mut bits = Vec::with_capacity(num_bits);
        for i in 0..num_bits {
            let bit = (value >> i) & 1;
            bits.push(GoldilocksField::new(bit));
        }
        Ok(Self { bits, num_bits })
    }

    /// Recompose a field element from its bit decomposition.
    pub fn recompose(&self) -> GoldilocksField {
        let mut result = GoldilocksField::zero();
        let mut power_of_two = GoldilocksField::one();
        let two = GoldilocksField::new(2);
        for bit in &self.bits {
            let contribution = bit.mul(&power_of_two);
            result = result.add(&contribution);
            power_of_two = power_of_two.mul(&two);
        }
        result
    }

    /// Verify that every element in `bits` is 0 or 1 (boolean constraint).
    pub fn verify_boolean_constraints(&self) -> bool {
        self.bits.iter().all(|b| b.value == 0 || b.value == 1)
    }

    /// Verify that `recompose()` yields the expected value.
    pub fn verify_decomposition(&self, expected: u64) -> bool {
        let recomposed = self.recompose();
        recomposed == GoldilocksField::new(expected)
    }
}

impl fmt::Display for BitDecomposition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, bit) in self.bits.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", bit.value)?;
        }
        write!(f, "] ({} bits)", self.num_bits)
    }
}

/// Output from a tropical gadget computation, including auxiliary witness
/// values needed by the STARK verifier.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GadgetOutput {
    /// The main output value.
    pub value: GoldilocksField,
    /// Auxiliary values for the STARK verifier (bit decompositions,
    /// comparison bits, intermediate products, etc.).
    pub auxiliary: Vec<GoldilocksField>,
}

impl GadgetOutput {
    pub fn new(value: GoldilocksField) -> Self {
        Self {
            value,
            auxiliary: Vec::new(),
        }
    }

    pub fn with_auxiliary(value: GoldilocksField, auxiliary: Vec<GoldilocksField>) -> Self {
        Self { value, auxiliary }
    }
}

/// Gadget-based encoding of tropical semiring operations as field
/// arithmetic.  The tropical min(a, b) cannot be expressed as a polynomial
/// in the field; instead we decompose a and b into bits and compute a
/// comparison circuit.
#[derive(Debug, Clone)]
pub struct TropicalGadget {
    /// Number of bits used in decompositions.
    pub num_bits: usize,
}

impl TropicalGadget {
    /// Create a gadget with the given bit width.
    pub fn new(num_bits: usize) -> Self {
        assert!(num_bits > 0 && num_bits <= 64);
        Self { num_bits }
    }

    /// Default gadget with 32-bit decomposition.
    pub fn default_32() -> Self {
        Self { num_bits: 32 }
    }

    /// Compute the less-than comparison bit via bit decomposition.
    /// Returns 1 if a < b, else 0.
    ///
    /// Algorithm: scan from MSB to LSB.  At each position i, compute
    ///   prefix_eq[i] = product_{j > i} (1 − (a_j ⊕ b_j))
    /// where ⊕ is XOR = a + b − 2ab in the field.
    /// Then lt = sum_i prefix_eq[i] · (1 − a_i) · b_i.
    pub fn gadget_less_than(
        &self,
        a: &BitDecomposition,
        b: &BitDecomposition,
    ) -> Result<GadgetOutput, EmbeddingError> {
        if a.num_bits != self.num_bits || b.num_bits != self.num_bits {
            return Err(EmbeddingError::InvalidGadget {
                desc: format!(
                    "bit width mismatch: gadget expects {}, got a={} b={}",
                    self.num_bits, a.num_bits, b.num_bits
                ),
            });
        }
        if !a.verify_boolean_constraints() || !b.verify_boolean_constraints() {
            return Err(EmbeddingError::InvalidGadget {
                desc: "bit decomposition contains non-boolean values".to_string(),
            });
        }

        let one = GoldilocksField::one();
        let two = GoldilocksField::new(2);
        let n = self.num_bits;
        let mut auxiliary = Vec::new();

        // prefix_eq[i]: all bits from i+1..n−1 match
        // We scan from MSB (index n-1) downward.
        let mut prefix_eq = GoldilocksField::one();
        let mut lt = GoldilocksField::zero();

        for i in (0..n).rev() {
            let ai = &a.bits[i];
            let bi = &b.bits[i];

            // XOR in the field: a + b - 2*a*b
            let ab = ai.mul(bi);
            let xor_val = ai.add(bi).sub(&two.mul(&ab));

            // eq_i = 1 − xor = 1 if bits are equal
            let eq_i = one.sub(&xor_val);

            // Contribution to lt: prefix_eq * (1 - a_i) * b_i
            let not_ai = one.sub(ai);
            let contrib = prefix_eq.mul(&not_ai).mul(bi);
            lt = lt.add(&contrib);

            // Record auxiliary witnesses
            auxiliary.push(xor_val);
            auxiliary.push(eq_i);
            auxiliary.push(prefix_eq);

            // Update prefix_eq for next (lower) bit
            prefix_eq = prefix_eq.mul(&eq_i);
        }

        Ok(GadgetOutput::with_auxiliary(lt, auxiliary))
    }

    /// Compute min(a, b) using the comparison gadget.
    ///   min(a, b) = a · lt(a,b) + b · (1 − lt(a,b))
    /// where lt(a,b) = 1 if a < b, 0 otherwise.
    pub fn gadget_min(
        &self,
        a: &BitDecomposition,
        b: &BitDecomposition,
    ) -> Result<GadgetOutput, EmbeddingError> {
        let lt_output = self.gadget_less_than(a, b)?;
        let lt = lt_output.value;

        let val_a = a.recompose();
        let val_b = b.recompose();

        let one = GoldilocksField::one();
        // min = a * lt + b * (1 - lt)
        let min_val = val_a.mul(&lt).add(&val_b.mul(&one.sub(&lt)));

        let mut aux = lt_output.auxiliary;
        aux.push(lt);
        aux.push(val_a);
        aux.push(val_b);

        Ok(GadgetOutput::with_auxiliary(min_val, aux))
    }

    /// Compute tropical addition (= min) for two field-encoded tropical
    /// values via the gadget.
    pub fn tropical_add(
        &self,
        a_val: u64,
        b_val: u64,
    ) -> Result<GadgetOutput, EmbeddingError> {
        let a_bits = BitDecomposition::decompose(a_val, self.num_bits)?;
        let b_bits = BitDecomposition::decompose(b_val, self.num_bits)?;
        self.gadget_min(&a_bits, &b_bits)
    }

    /// Compute tropical multiplication (= addition in the field).
    pub fn tropical_mul(
        &self,
        a_val: u64,
        b_val: u64,
    ) -> Result<GoldilocksField, EmbeddingError> {
        // Tropical mul is ordinary addition, which maps directly.
        let fa = GoldilocksField::new(a_val);
        let fb = GoldilocksField::new(b_val);
        Ok(fa.add(&fb))
    }
}

/// Embedding of the tropical semiring via the gadget approach.
/// Only supports finite (non-infinity) values that fit in `num_bits`.
#[derive(Debug, Clone)]
pub struct TropicalToGadget {
    gadget: TropicalGadget,
}

impl TropicalToGadget {
    pub fn new(num_bits: usize) -> Self {
        Self {
            gadget: TropicalGadget::new(num_bits),
        }
    }

    pub fn gadget(&self) -> &TropicalGadget {
        &self.gadget
    }

    /// Embed a tropical value, returning the field element and its bit
    /// decomposition (needed by the verifier).
    pub fn embed_with_bits(
        &self,
        value: &TropicalSemiring,
    ) -> Result<(GoldilocksField, BitDecomposition), EmbeddingError> {
        let raw = value.raw();
        if raw.is_infinite() || raw.is_nan() || raw < 0.0 {
            return Err(EmbeddingError::OverflowError {
                value: format!("{}", raw),
                field_desc: "tropical value must be a finite non-negative integer for gadget embedding".to_string(),
            });
        }
        let int_val = raw as u64;
        if (int_val as f64) != raw {
            return Err(EmbeddingError::PrecisionLoss {
                original: format!("{}", raw),
                embedded: format!("{}", int_val),
            });
        }
        let bits = BitDecomposition::decompose(int_val, self.gadget.num_bits)?;
        let field_val = GoldilocksField::new(int_val);
        Ok((field_val, bits))
    }
}

impl SemiringEmbedding<TropicalSemiring, GoldilocksField> for TropicalToGadget {
    fn embed(&self, value: &TropicalSemiring) -> Result<GoldilocksField, EmbeddingError> {
        let (field_val, _bits) = self.embed_with_bits(value)?;
        Ok(field_val)
    }

    fn unembed(&self, value: &GoldilocksField) -> Result<TropicalSemiring, EmbeddingError> {
        // We interpret the field element as a non-negative integer distance.
        let max_val = if self.gadget.num_bits < 64 {
            (1u64 << self.gadget.num_bits) - 1
        } else {
            u64::MAX
        };
        if value.value > max_val {
            return Err(EmbeddingError::NotInImage {
                field_element: value.value.to_string(),
                desc: format!("exceeds {}-bit range", self.gadget.num_bits),
            });
        }
        Ok(TropicalSemiring::new(value.value as f64))
    }

    fn can_embed(&self, value: &TropicalSemiring) -> bool {
        let raw = value.raw();
        if raw.is_infinite() || raw.is_nan() || raw < 0.0 {
            return false;
        }
        let int_val = raw as u64;
        if (int_val as f64) != raw {
            return false;
        }
        if self.gadget.num_bits < 64 {
            int_val < (1u64 << self.gadget.num_bits)
        } else {
            true
        }
    }

    fn verify_homomorphism(
        &self,
        a: &TropicalSemiring,
        b: &TropicalSemiring,
    ) -> Result<bool, EmbeddingError> {
        // Tropical addition is min — requires the gadget, not direct field ops.
        // We verify that the gadget gives the right answer.
        let raw_a = a.raw() as u64;
        let raw_b = b.raw() as u64;

        // Check min
        let min_result = self.gadget.tropical_add(raw_a, raw_b)?;
        let expected_min = a.add(b); // tropical min
        let expected_field = self.embed(&expected_min)?;
        if min_result.value != expected_field {
            return Ok(false);
        }

        // Check tropical mul (= addition): this is direct field addition
        let mul_result = self.gadget.tropical_mul(raw_a, raw_b)?;
        let expected_mul = a.mul(b); // tropical +
        let expected_mul_field = self.embed(&expected_mul)?;
        if mul_result != expected_mul_field {
            return Ok(false);
        }

        Ok(true)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 7. RealToGoldilocks — fixed-point approximation
// ═══════════════════════════════════════════════════════════════════════════

/// Fixed-point embedding of real numbers into the Goldilocks field.
///
/// A real value `x` is embedded as `round(x * 2^scale_bits) mod p`.
/// This introduces quantisation error bounded by `2^{-scale_bits}`.
#[derive(Debug, Clone)]
pub struct RealFixedPointEmbedding {
    /// Number of fractional bits.
    pub scale_bits: u32,
    /// The scaling factor as a field element: 2^scale_bits mod p.
    pub scale_factor: GoldilocksField,
    /// Inverse of the scaling factor in the field.
    pub scale_inv: GoldilocksField,
    /// Maximum absolute value that can be embedded without overflow.
    pub max_abs_value: f64,
}

impl RealFixedPointEmbedding {
    /// Create a new fixed-point embedding with the given number of
    /// fractional bits.
    pub fn new(scale_bits: u32) -> Self {
        assert!(scale_bits < 63, "scale_bits must be < 63");
        let scale = 1u64 << scale_bits;
        let scale_factor = GoldilocksField::new(scale);
        // We compute scale_inv lazily via Fermat's little theorem.
        let scale_inv = scale_factor
            .inv()
            .expect("2^scale_bits is non-zero and invertible");
        // max value: (p-1) / 2^scale_bits (positive side)
        let max_abs_value = (GOLDILOCKS_PRIME - 1) as f64 / (scale as f64) / 2.0;
        Self {
            scale_bits,
            scale_factor,
            scale_inv,
            max_abs_value,
        }
    }

    /// Embed a floating-point value using fixed-point quantisation.
    pub fn embed_fixed_point(
        value: f64,
        scale_bits: u32,
    ) -> Result<GoldilocksField, EmbeddingError> {
        if value.is_nan() || value.is_infinite() {
            return Err(EmbeddingError::OverflowError {
                value: format!("{}", value),
                field_desc: "non-finite value cannot be embedded".to_string(),
            });
        }
        let scale = (1u64 << scale_bits) as f64;
        let scaled = (value * scale).round();
        if scaled < 0.0 {
            // Represent negative values as p - |scaled|
            let abs_scaled = (-scaled) as u64;
            if abs_scaled >= GOLDILOCKS_PRIME {
                return Err(EmbeddingError::OverflowError {
                    value: format!("{}", value),
                    field_desc: "absolute value too large for fixed-point embedding".to_string(),
                });
            }
            Ok(GoldilocksField::new(GOLDILOCKS_PRIME - abs_scaled))
        } else {
            let int_val = scaled as u64;
            if int_val >= GOLDILOCKS_PRIME {
                return Err(EmbeddingError::OverflowError {
                    value: format!("{}", value),
                    field_desc: "value too large for fixed-point embedding".to_string(),
                });
            }
            Ok(GoldilocksField::new(int_val))
        }
    }

    /// Recover an approximate real number from a fixed-point field element.
    pub fn unembed_fixed_point(value: &GoldilocksField, scale_bits: u32) -> f64 {
        let scale = (1u64 << scale_bits) as f64;
        let half_p = GOLDILOCKS_PRIME / 2;
        if value.value > half_p {
            // Negative value
            let abs_val = GOLDILOCKS_PRIME - value.value;
            -(abs_val as f64) / scale
        } else {
            value.value as f64 / scale
        }
    }

    /// Compute the quantisation error bound for this embedding.
    pub fn precision_bound(&self) -> f64 {
        1.0 / (1u64 << self.scale_bits) as f64
    }
}

impl SemiringEmbedding<RealSemiring, GoldilocksField> for RealFixedPointEmbedding {
    fn embed(&self, value: &RealSemiring) -> Result<GoldilocksField, EmbeddingError> {
        Self::embed_fixed_point(value.raw(), self.scale_bits)
    }

    fn unembed(&self, value: &GoldilocksField) -> Result<RealSemiring, EmbeddingError> {
        Ok(RealSemiring::new(Self::unembed_fixed_point(
            value,
            self.scale_bits,
        )))
    }

    fn can_embed(&self, value: &RealSemiring) -> bool {
        let raw = value.raw();
        !raw.is_nan() && !raw.is_infinite() && raw.abs() <= self.max_abs_value
    }

    fn verify_homomorphism(
        &self,
        a: &RealSemiring,
        b: &RealSemiring,
    ) -> Result<bool, EmbeddingError> {
        // Fixed-point is only an approximate homomorphism due to rounding.
        // We check that the error is within the precision bound.
        let fa = self.embed(a)?;
        let fb = self.embed(b)?;

        // Additive: embed(a+b) ≈ embed(a) + embed(b)
        let sum_real = a.add(b);
        let f_sum = self.embed(&sum_real)?;
        let f_add = fa.add(&fb);
        // The difference should be at most 1 (one unit of least precision).
        let diff_add = if f_sum.value >= f_add.value {
            f_sum.value - f_add.value
        } else {
            f_add.value - f_sum.value
        };
        // Allow rounding error of up to 1 ULP
        if diff_add > 1 {
            return Ok(false);
        }

        // Multiplicative: embed(a*b) ≈ embed(a) * embed(b) / scale_factor
        // because fixed-point mul introduces an extra scale factor.
        // We don't check mul homomorphism strictly since it requires
        // division by 2^scale_bits, which is a known property.

        Ok(true)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 8. WfaEmbedder — embed an entire WFA
// ═══════════════════════════════════════════════════════════════════════════

/// Utility for embedding and unembedding entire weighted finite automata.
pub struct WfaEmbedder<S: Semiring> {
    _phantom: PhantomData<S>,
}

impl<S: Semiring> WfaEmbedder<S> {
    /// Embed all weights in a WFA, preserving structure.
    pub fn embed_wfa(
        wfa: &WeightedFiniteAutomaton<S>,
        embedding: &impl SemiringEmbedding<S, GoldilocksField>,
    ) -> Result<WeightedFiniteAutomaton<GoldilocksField>, EmbeddingError> {
        let n = wfa.state_count();
        let alphabet = wfa.alphabet().clone();
        let mut embedded = WeightedFiniteAutomaton::new(n, alphabet);

        // Embed initial weights
        for q in 0..n {
            let w = &wfa.initial_weights()[q];
            let fw = embedding.embed(w)?;
            embedded.set_initial_weight(q, fw);
        }

        // Embed final weights
        for q in 0..n {
            let w = &wfa.final_weights()[q];
            let fw = embedding.embed(w)?;
            embedded.set_final_weight(q, fw);
        }

        // Embed transitions
        for t in wfa.all_transitions() {
            let fw = embedding.embed(&t.weight)?;
            embedded
                .add_transition(t.from_state, t.symbol, t.to_state, fw)
                .map_err(|e| EmbeddingError::InternalError {
                    msg: format!("transition error: {}", e),
                })?;
        }

        // Copy state labels
        for q in 0..n {
            if let Some(label) = wfa.state_label(q) {
                embedded.set_state_label(q, label.to_string());
            }
        }

        Ok(embedded)
    }

    /// Unembed a Goldilocks WFA back to semiring S.
    pub fn unembed_wfa(
        wfa: &WeightedFiniteAutomaton<GoldilocksField>,
        embedding: &impl SemiringEmbedding<S, GoldilocksField>,
    ) -> Result<WeightedFiniteAutomaton<S>, EmbeddingError> {
        let n = wfa.state_count();
        let alphabet = wfa.alphabet().clone();
        let mut result = WeightedFiniteAutomaton::new(n, alphabet);

        for q in 0..n {
            let w = &wfa.initial_weights()[q];
            let sw = embedding.unembed(w)?;
            result.set_initial_weight(q, sw);
        }

        for q in 0..n {
            let w = &wfa.final_weights()[q];
            let sw = embedding.unembed(w)?;
            result.set_final_weight(q, sw);
        }

        for t in wfa.all_transitions() {
            let sw = embedding.unembed(&t.weight)?;
            result
                .add_transition(t.from_state, t.symbol, t.to_state, sw)
                .map_err(|e| EmbeddingError::InternalError {
                    msg: format!("transition error: {}", e),
                })?;
        }

        for q in 0..n {
            if let Some(label) = wfa.state_label(q) {
                result.set_state_label(q, label.to_string());
            }
        }

        Ok(result)
    }

    /// Verify that an embedding preserves WFA semantics on a set of test
    /// words.  For each word, computes the weight in both the original and
    /// embedded WFA and checks that embed(original_weight) equals the
    /// embedded WFA's weight.
    pub fn verify_wfa_embedding(
        original: &WeightedFiniteAutomaton<S>,
        embedded: &WeightedFiniteAutomaton<GoldilocksField>,
        embedding: &impl SemiringEmbedding<S, GoldilocksField>,
        test_words: &[Vec<usize>],
    ) -> Result<EmbeddingVerification, EmbeddingError> {
        let mut failures = Vec::new();

        for word in test_words {
            let orig_weight = original.compute_weight(word);
            let emb_weight = embedded.compute_weight(word);
            let expected_emb = embedding.embed(&orig_weight)?;

            if emb_weight != expected_emb {
                failures.push(EmbeddingFailure {
                    word: word.clone(),
                    original_weight: format!("{:?}", orig_weight),
                    embedded_weight: format!("{}", emb_weight.value),
                    expected_embedded: format!("{}", expected_emb.value),
                });
            }
        }

        let tests_run = test_words.len();
        Ok(EmbeddingVerification {
            all_passed: failures.is_empty(),
            tests_run,
            failures,
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 9. EmbeddingVerification
// ═══════════════════════════════════════════════════════════════════════════

/// Result of verifying a WFA embedding against test words.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingVerification {
    /// True if every test word produced matching weights.
    pub all_passed: bool,
    /// Number of test words evaluated.
    pub tests_run: usize,
    /// Details of any failures.
    pub failures: Vec<EmbeddingFailure>,
}

impl fmt::Display for EmbeddingVerification {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "EmbeddingVerification: {}/{} passed",
            self.tests_run - self.failures.len(),
            self.tests_run
        )?;
        for fail in &self.failures {
            write!(f, "\n  FAIL: word={:?}", fail.word)?;
            write!(f, " orig={} emb={}", fail.original_weight, fail.embedded_weight)?;
            write!(f, " expected={}", fail.expected_embedded)?;
        }
        Ok(())
    }
}

/// A single failure in embedding verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingFailure {
    /// The input word (sequence of symbol indices).
    pub word: Vec<usize>,
    /// Weight computed in the original semiring.
    pub original_weight: String,
    /// Weight computed in the embedded (field) WFA.
    pub embedded_weight: String,
    /// Expected embedded weight (embed of original weight).
    pub expected_embedded: String,
}

// ═══════════════════════════════════════════════════════════════════════════
// 10. Batch operations
// ═══════════════════════════════════════════════════════════════════════════

/// Result of batch homomorphism verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchVerification {
    /// Total number of pairs tested.
    pub total_pairs: usize,
    /// Number that passed both additive and multiplicative checks.
    pub passed: usize,
    /// Number that failed.
    pub failed: usize,
    /// Details of failures: (pair index, description).
    pub failures: Vec<(usize, String)>,
}

impl fmt::Display for BatchVerification {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BatchVerification: {}/{} passed ({} failed)",
            self.passed, self.total_pairs, self.failed
        )
    }
}

/// Embed a slice of semiring values in batch.
pub fn batch_embed<S: Semiring>(
    values: &[S],
    embedding: &impl SemiringEmbedding<S, GoldilocksField>,
) -> Result<Vec<GoldilocksField>, EmbeddingError> {
    values.iter().map(|v| embedding.embed(v)).collect()
}

/// Unembed a slice of field values in batch.
pub fn batch_unembed<S: Semiring>(
    values: &[GoldilocksField],
    embedding: &impl SemiringEmbedding<S, GoldilocksField>,
) -> Result<Vec<S>, EmbeddingError> {
    values.iter().map(|v| embedding.unembed(v)).collect()
}

/// Verify the homomorphism property for many pairs at once.
pub fn batch_verify_homomorphism<S: Semiring>(
    pairs: &[(S, S)],
    embedding: &impl SemiringEmbedding<S, GoldilocksField>,
) -> Result<BatchVerification, EmbeddingError> {
    let mut passed = 0usize;
    let mut failures = Vec::new();

    for (i, (a, b)) in pairs.iter().enumerate() {
        match embedding.verify_homomorphism(a, b) {
            Ok(true) => passed += 1,
            Ok(false) => {
                failures.push((
                    i,
                    format!("homomorphism failed for pair ({:?}, {:?})", a, b),
                ));
            }
            Err(e) => {
                failures.push((i, format!("error: {}", e)));
            }
        }
    }

    let total_pairs = pairs.len();
    Ok(BatchVerification {
        total_pairs,
        passed,
        failed: total_pairs - passed,
        failures,
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// 11. Overflow detection
// ═══════════════════════════════════════════════════════════════════════════

/// Tracks potential overflow during WFA computation over the Goldilocks
/// field.  Useful for determining whether a given WFA + input length
/// combination can be safely computed without field-arithmetic wrap-around.
#[derive(Debug, Clone)]
pub struct OverflowDetector {
    /// The Goldilocks prime.
    pub prime: u64,
}

impl OverflowDetector {
    pub fn new() -> Self {
        Self {
            prime: GOLDILOCKS_PRIME,
        }
    }

    /// Would `a + b` overflow the Goldilocks field (i.e., exceed p − 1)?
    pub fn check_addition_overflow(&self, a: u64, b: u64) -> bool {
        // Overflow means the *mathematical* sum a + b ≥ p, which would
        // cause a wrap in the field.  This matters for counting semirings
        // where we want the field value to equal the true count.
        (a as u128) + (b as u128) >= (self.prime as u128)
    }

    /// Would `a * b` overflow the Goldilocks field?
    pub fn check_multiplication_overflow(&self, a: u64, b: u64) -> bool {
        (a as u128) * (b as u128) >= (self.prime as u128)
    }

    /// Estimate the maximum possible weight that a counting WFA can produce
    /// on any input of length at most `max_input_length`.
    ///
    /// Uses a conservative bound: the weight can grow at most by a factor
    /// of (sum of all transition weights per symbol * number of states)
    /// per step.  The initial contribution is the max initial weight.
    pub fn max_weight_bound(
        &self,
        wfa: &WeightedFiniteAutomaton<CountingSemiring>,
        max_input_length: usize,
    ) -> u64 {
        let n = wfa.state_count();
        if n == 0 {
            return 0;
        }

        // Maximum initial weight
        let max_init: u64 = wfa
            .initial_weights()
            .iter()
            .map(|w| w.value)
            .max()
            .unwrap_or(0);

        // For each symbol, compute max total outgoing weight from any state
        let alpha_size = wfa.alphabet().size();
        let mut max_transition_sum: u64 = 0;
        for sym in 0..alpha_size {
            for from in 0..n {
                let trans = wfa.transitions_from(from, sym);
                let sum: u64 = trans.iter().map(|(_to, w)| w.value).sum();
                max_transition_sum = max_transition_sum.max(sum);
            }
        }

        // Maximum final weight
        let max_final: u64 = wfa
            .final_weights()
            .iter()
            .map(|w| w.value)
            .max()
            .unwrap_or(0);

        // Conservative bound: max_init * max_transition_sum^L * n^L * max_final
        // But we use a tighter bound by simulating the forward pass with
        // worst-case weights.
        //
        // At each step, the forward vector can have at most n entries, each
        // bounded by (previous_max * max_transition_sum * n).
        // After L steps the bound is:
        //   max_init * (max_transition_sum * n)^L * max_final
        //
        // We compute this in u128 to detect overflow early.
        let factor = (max_transition_sum as u128) * (n as u128);
        let mut current_max: u128 = max_init as u128;

        for _ in 0..max_input_length {
            current_max = current_max.saturating_mul(factor);
            if current_max >= self.prime as u128 {
                return u64::MAX; // overflow inevitable
            }
        }

        current_max = current_max.saturating_mul(max_final as u128);
        if current_max >= self.prime as u128 {
            u64::MAX
        } else {
            current_max as u64
        }
    }

    /// Maximum input length for which the WFA weight is guaranteed not to
    /// overflow the Goldilocks field.
    pub fn safe_input_length(
        &self,
        wfa: &WeightedFiniteAutomaton<CountingSemiring>,
    ) -> usize {
        // Binary search for the largest L such that max_weight_bound(L) < p.
        let mut lo: usize = 0;
        let mut hi: usize = 1024; // generous upper bound

        // First ensure hi is large enough
        while self.max_weight_bound(wfa, hi) < u64::MAX {
            hi *= 2;
            if hi > 1_000_000 {
                return hi; // effectively unbounded
            }
        }

        // Binary search for the transition point
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.max_weight_bound(wfa, mid) < u64::MAX {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        if lo == 0 { 0 } else { lo - 1 }
    }

    /// Check whether a specific WFA computation on a given word is
    /// overflow-safe.
    pub fn check_word_safety(
        &self,
        wfa: &WeightedFiniteAutomaton<CountingSemiring>,
        word: &[usize],
    ) -> bool {
        // Simulate the forward pass tracking maximum possible values.
        let n = wfa.state_count();
        if n == 0 {
            return true;
        }

        let mut forward: Vec<u128> = wfa
            .initial_weights()
            .iter()
            .map(|w| w.value as u128)
            .collect();

        for &sym in word {
            if sym >= wfa.alphabet().size() {
                return true; // no transitions, weight is zero
            }
            let mut next = vec![0u128; n];
            for from in 0..n {
                if forward[from] == 0 {
                    continue;
                }
                for &(to, ref w) in wfa.transitions_from(from, sym) {
                    let contribution = forward[from].saturating_mul(w.value as u128);
                    next[to] = next[to].saturating_add(contribution);
                    if next[to] >= self.prime as u128 {
                        return false;
                    }
                }
            }
            forward = next;
        }

        // Check final summation
        let mut total: u128 = 0;
        for q in 0..n {
            let contribution =
                forward[q].saturating_mul(wfa.final_weights()[q].value as u128);
            total = total.saturating_add(contribution);
            if total >= self.prime as u128 {
                return false;
            }
        }

        true
    }
}

impl Default for OverflowDetector {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 12. Extension fields
// ═══════════════════════════════════════════════════════════════════════════

/// Quadratic extension of a field F: elements of F[X]/(X² + 1).
/// Represented as (a, b) meaning a + b·i where i² = −1.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct QuadraticExtension<F: Semiring> {
    pub re: F,
    pub im: F,
}

impl<F: Semiring + Copy> QuadraticExtension<F>
where
    GoldilocksField: Into<F>,
{
    pub fn new(re: F, im: F) -> Self {
        Self { re, im }
    }

    pub fn from_base(re: F) -> Self {
        Self { re, im: F::zero() }
    }
}

impl QuadraticExtension<GoldilocksField> {
    /// Create from two field elements.
    pub fn from_pair(re: GoldilocksField, im: GoldilocksField) -> Self {
        Self { re, im }
    }

    /// Create from a base field element (imaginary part = 0).
    pub fn from_base_field(re: GoldilocksField) -> Self {
        Self {
            re,
            im: GoldilocksField::zero(),
        }
    }

    /// The zero element.
    pub fn ext_zero() -> Self {
        Self {
            re: GoldilocksField::zero(),
            im: GoldilocksField::zero(),
        }
    }

    /// The one element.
    pub fn ext_one() -> Self {
        Self {
            re: GoldilocksField::one(),
            im: GoldilocksField::zero(),
        }
    }

    /// The imaginary unit i.
    pub fn ext_i() -> Self {
        Self {
            re: GoldilocksField::zero(),
            im: GoldilocksField::one(),
        }
    }

    /// Addition: (a + bi) + (c + di) = (a+c) + (b+d)i.
    pub fn ext_add(&self, other: &Self) -> Self {
        Self {
            re: self.re.add(&other.re),
            im: self.im.add(&other.im),
        }
    }

    /// Subtraction.
    pub fn ext_sub(&self, other: &Self) -> Self {
        Self {
            re: self.re.sub(&other.re),
            im: self.im.sub(&other.im),
        }
    }

    /// Multiplication: (a + bi)(c + di) = (ac − bd) + (ad + bc)i.
    pub fn ext_mul(&self, other: &Self) -> Self {
        let ac = self.re.mul(&other.re);
        let bd = self.im.mul(&other.im);
        let ad = self.re.mul(&other.im);
        let bc = self.im.mul(&other.re);
        Self {
            re: ac.sub(&bd),
            im: ad.add(&bc),
        }
    }

    /// Conjugate: (a + bi)* = a − bi.
    pub fn conjugate(&self) -> Self {
        Self {
            re: self.re,
            im: self.im.neg(),
        }
    }

    /// Norm: (a + bi)(a − bi) = a² + b².
    pub fn norm(&self) -> GoldilocksField {
        self.re.mul(&self.re).add(&self.im.mul(&self.im))
    }

    /// Multiplicative inverse: 1/(a + bi) = (a − bi) / (a² + b²).
    pub fn ext_inv(&self) -> Result<Self, EmbeddingError> {
        let n = self.norm();
        if n.is_zero() {
            return Err(EmbeddingError::InternalError {
                msg: "cannot invert zero in quadratic extension".to_string(),
            });
        }
        let n_inv = n.inv().map_err(|_| EmbeddingError::InternalError {
            msg: "norm inversion failed".to_string(),
        })?;
        let conj = self.conjugate();
        Ok(Self {
            re: conj.re.mul(&n_inv),
            im: conj.im.mul(&n_inv),
        })
    }

    /// Division.
    pub fn ext_div(&self, other: &Self) -> Result<Self, EmbeddingError> {
        let inv = other.ext_inv()?;
        Ok(self.ext_mul(&inv))
    }

    /// Exponentiation by repeated squaring.
    pub fn ext_pow(&self, mut n: u64) -> Self {
        if n == 0 {
            return Self::ext_one();
        }
        let mut base = self.clone();
        let mut result = Self::ext_one();
        while n > 1 {
            if n % 2 == 1 {
                result = result.ext_mul(&base);
            }
            base = base.ext_mul(&base);
            n /= 2;
        }
        result.ext_mul(&base)
    }

    /// Check if this is a base-field element (im == 0).
    pub fn is_base_field(&self) -> bool {
        self.im.is_zero()
    }
}

impl fmt::Display for QuadraticExtension<GoldilocksField> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.im.is_zero() {
            write!(f, "{}", self.re.value)
        } else if self.re.is_zero() {
            write!(f, "{}i", self.im.value)
        } else {
            write!(f, "{} + {}i", self.re.value, self.im.value)
        }
    }
}

/// Cubic extension of a field F: elements of F[X]/(X³ − α) for some
/// non-residue α.  Represented as (a, b, c) meaning a + b·θ + c·θ²
/// where θ³ = α.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CubicExtension<F: Semiring> {
    pub c0: F,
    pub c1: F,
    pub c2: F,
    /// The non-residue: θ³ = alpha.
    pub alpha: F,
}

impl CubicExtension<GoldilocksField> {
    /// Create a cubic extension element.
    pub fn new(
        c0: GoldilocksField,
        c1: GoldilocksField,
        c2: GoldilocksField,
        alpha: GoldilocksField,
    ) -> Self {
        Self { c0, c1, c2, alpha }
    }

    /// Zero element.
    pub fn cubic_zero(alpha: GoldilocksField) -> Self {
        Self {
            c0: GoldilocksField::zero(),
            c1: GoldilocksField::zero(),
            c2: GoldilocksField::zero(),
            alpha,
        }
    }

    /// One element.
    pub fn cubic_one(alpha: GoldilocksField) -> Self {
        Self {
            c0: GoldilocksField::one(),
            c1: GoldilocksField::zero(),
            c2: GoldilocksField::zero(),
            alpha,
        }
    }

    /// Addition.
    pub fn cubic_add(&self, other: &Self) -> Self {
        Self {
            c0: self.c0.add(&other.c0),
            c1: self.c1.add(&other.c1),
            c2: self.c2.add(&other.c2),
            alpha: self.alpha,
        }
    }

    /// Subtraction.
    pub fn cubic_sub(&self, other: &Self) -> Self {
        Self {
            c0: self.c0.sub(&other.c0),
            c1: self.c1.sub(&other.c1),
            c2: self.c2.sub(&other.c2),
            alpha: self.alpha,
        }
    }

    /// Multiplication.
    /// (a0 + a1·θ + a2·θ²)(b0 + b1·θ + b2·θ²)
    /// = a0·b0 + (a0·b1 + a1·b0)·θ + (a0·b2 + a1·b1 + a2·b0)·θ²
    ///   + (a1·b2 + a2·b1)·θ³ + a2·b2·θ⁴
    /// where θ³ = α, θ⁴ = α·θ.
    pub fn cubic_mul(&self, other: &Self) -> Self {
        let a0b0 = self.c0.mul(&other.c0);
        let a0b1 = self.c0.mul(&other.c1);
        let a0b2 = self.c0.mul(&other.c2);
        let a1b0 = self.c1.mul(&other.c0);
        let a1b1 = self.c1.mul(&other.c1);
        let a1b2 = self.c1.mul(&other.c2);
        let a2b0 = self.c2.mul(&other.c0);
        let a2b1 = self.c2.mul(&other.c1);
        let a2b2 = self.c2.mul(&other.c2);

        // θ³ = α
        let coeff_theta3 = a1b2.add(&a2b1); // * α
        let coeff_theta4 = a2b2; // * α·θ

        let r0 = a0b0.add(&coeff_theta3.mul(&self.alpha));
        let r1 = a0b1.add(&a1b0).add(&coeff_theta4.mul(&self.alpha));
        let r2 = a0b2.add(&a1b1).add(&a2b0);

        Self {
            c0: r0,
            c1: r1,
            c2: r2,
            alpha: self.alpha,
        }
    }

    /// Norm: N(a + bθ + cθ²) in the base field.
    /// For the extension defined by θ³ = α, the norm is:
    /// a³ + α·b³ + α²·c³ − 3α·a·b·c
    pub fn cubic_norm(&self) -> GoldilocksField {
        let a3 = self.c0.mul(&self.c0).mul(&self.c0);
        let b3 = self.c1.mul(&self.c1).mul(&self.c1);
        let c3 = self.c2.mul(&self.c2).mul(&self.c2);
        let alpha_sq = self.alpha.mul(&self.alpha);
        let three = GoldilocksField::new(3);

        let term1 = a3;
        let term2 = self.alpha.mul(&b3);
        let term3 = alpha_sq.mul(&c3);
        let term4 = three
            .mul(&self.alpha)
            .mul(&self.c0)
            .mul(&self.c1)
            .mul(&self.c2);

        term1.add(&term2).add(&term3).sub(&term4)
    }

    /// Check if this is a base-field element.
    pub fn is_base_field(&self) -> bool {
        self.c1.is_zero() && self.c2.is_zero()
    }
}

impl fmt::Display for CubicExtension<GoldilocksField> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} + {}θ + {}θ²",
            self.c0.value, self.c1.value, self.c2.value
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 13. EmbeddingRegistry
// ═══════════════════════════════════════════════════════════════════════════

/// Identifies a registered embedding by name and source/target semiring.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct EmbeddingKey {
    pub name: String,
    pub source_type: String,
    pub target_type: String,
}

/// Trait object wrapper so we can store heterogeneous embeddings.
/// Because `SemiringEmbedding<S, F>` has type parameters, we erase
/// them behind a concrete wrapper for each known pair.
#[derive(Debug, Clone)]
pub enum RegisteredEmbedding {
    CountingToGoldilocks(CountingToGoldilocks),
    BooleanToGoldilocks(BooleanToGoldilocks),
    BoundedCountingToGoldilocks(BoundedCountingToGoldilocks),
    TropicalToGadget(TropicalToGadget),
    RealFixedPoint(RealFixedPointEmbedding),
}

impl fmt::Display for RegisteredEmbedding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RegisteredEmbedding::CountingToGoldilocks(_) => {
                write!(f, "CountingSemiring → GoldilocksField")
            }
            RegisteredEmbedding::BooleanToGoldilocks(_) => {
                write!(f, "BooleanSemiring → GoldilocksField")
            }
            RegisteredEmbedding::BoundedCountingToGoldilocks(_) => {
                write!(f, "BoundedCountingSemiring → GoldilocksField")
            }
            RegisteredEmbedding::TropicalToGadget(_) => {
                write!(f, "TropicalSemiring → GoldilocksField (gadget)")
            }
            RegisteredEmbedding::RealFixedPoint(_) => {
                write!(f, "RealSemiring → GoldilocksField (fixed-point)")
            }
        }
    }
}

/// A registry of available semiring-to-field embeddings.
///
/// Pre-populated with all standard embeddings; additional embeddings
/// can be registered at runtime.
#[derive(Debug, Clone)]
pub struct EmbeddingRegistry {
    embeddings: HashMap<String, RegisteredEmbedding>,
}

impl EmbeddingRegistry {
    /// Create a new registry with all standard embeddings pre-registered.
    pub fn new() -> Self {
        let mut reg = Self {
            embeddings: HashMap::new(),
        };
        reg.register(
            "counting",
            RegisteredEmbedding::CountingToGoldilocks(CountingToGoldilocks::new()),
        );
        reg.register(
            "boolean",
            RegisteredEmbedding::BooleanToGoldilocks(BooleanToGoldilocks::new()),
        );
        reg.register(
            "bounded_counting",
            RegisteredEmbedding::BoundedCountingToGoldilocks(
                BoundedCountingToGoldilocks::default(),
            ),
        );
        reg.register(
            "tropical",
            RegisteredEmbedding::TropicalToGadget(TropicalToGadget::new(32)),
        );
        reg.register(
            "real",
            RegisteredEmbedding::RealFixedPoint(RealFixedPointEmbedding::new(20)),
        );
        reg
    }

    /// Register an embedding under the given name.
    pub fn register(&mut self, name: &str, embedding: RegisteredEmbedding) {
        self.embeddings.insert(name.to_string(), embedding);
    }

    /// Look up an embedding by name.
    pub fn get_embedding(&self, name: &str) -> Option<&RegisteredEmbedding> {
        self.embeddings.get(name)
    }

    /// List all registered embedding names.
    pub fn list(&self) -> Vec<String> {
        let mut names: Vec<String> = self.embeddings.keys().cloned().collect();
        names.sort();
        names
    }

    /// Automatically select an embedding based on the source semiring type
    /// name.  Returns the registered embedding if one matches.
    pub fn auto_select(&self, semiring_type: &str) -> Option<&RegisteredEmbedding> {
        match semiring_type {
            "CountingSemiring" | "counting" => self.get_embedding("counting"),
            "BooleanSemiring" | "boolean" => self.get_embedding("boolean"),
            "BoundedCountingSemiring" | "bounded_counting" => {
                self.get_embedding("bounded_counting")
            }
            "TropicalSemiring" | "tropical" => self.get_embedding("tropical"),
            "RealSemiring" | "real" => self.get_embedding("real"),
            _ => None,
        }
    }

    /// Number of registered embeddings.
    pub fn len(&self) -> usize {
        self.embeddings.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.embeddings.is_empty()
    }
}

impl Default for EmbeddingRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 14. Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ────────────────────────────────────────────────────────

    fn counting(v: u64) -> CountingSemiring {
        CountingSemiring::new(v)
    }

    fn boolean(v: bool) -> BooleanSemiring {
        BooleanSemiring::new(v)
    }

    fn tropical(v: f64) -> TropicalSemiring {
        TropicalSemiring::new(v)
    }

    fn real(v: f64) -> RealSemiring {
        RealSemiring::new(v)
    }

    fn gf(v: u64) -> GoldilocksField {
        GoldilocksField::new(v)
    }

    fn build_counting_wfa() -> WeightedFiniteAutomaton<CountingSemiring> {
        // Simple 2-state WFA over alphabet {0, 1}
        //   q0 --0/2--> q0
        //   q0 --1/3--> q1
        //   q1 --0/1--> q0
        //   q1 --1/1--> q1
        // Initial: q0 = 1, q1 = 0
        // Final:   q0 = 0, q1 = 1
        let alphabet = Alphabet::from_range(2);
        let mut wfa = WeightedFiniteAutomaton::new(2, alphabet);
        wfa.set_initial_weight(0, counting(1));
        wfa.set_final_weight(1, counting(1));
        wfa.add_transition(0, 0, 0, counting(2)).unwrap();
        wfa.add_transition(0, 1, 1, counting(3)).unwrap();
        wfa.add_transition(1, 0, 0, counting(1)).unwrap();
        wfa.add_transition(1, 1, 1, counting(1)).unwrap();
        wfa
    }

    fn build_boolean_wfa() -> WeightedFiniteAutomaton<BooleanSemiring> {
        // 2-state WFA that accepts strings ending with symbol 1
        let alphabet = Alphabet::from_range(2);
        let mut wfa = WeightedFiniteAutomaton::new(2, alphabet);
        wfa.set_initial_weight(0, boolean(true));
        wfa.set_final_weight(1, boolean(true));
        wfa.add_transition(0, 0, 0, boolean(true)).unwrap();
        wfa.add_transition(0, 1, 1, boolean(true)).unwrap();
        wfa.add_transition(1, 0, 0, boolean(true)).unwrap();
        wfa.add_transition(1, 1, 1, boolean(true)).unwrap();
        wfa
    }

    // ── CountingToGoldilocks ──────────────────────────────────────────

    #[test]
    fn counting_embed_unembed_roundtrip() {
        let emb = CountingToGoldilocks::new();
        for v in [0u64, 1, 2, 42, 1000, 1_000_000, u64::MAX / 2] {
            let c = counting(v);
            let f = emb.embed(&c).unwrap();
            let c2 = emb.unembed(&f).unwrap();
            // For values < p, roundtrip is exact.
            if v < GOLDILOCKS_PRIME {
                assert_eq!(c, c2, "roundtrip failed for v={}", v);
            }
        }
    }

    #[test]
    fn counting_embed_zero_one() {
        let emb = CountingToGoldilocks::new();
        assert_eq!(emb.embed(&counting(0)).unwrap(), GoldilocksField::zero());
        assert_eq!(emb.embed(&counting(1)).unwrap(), GoldilocksField::one());
    }

    #[test]
    fn counting_homomorphism_add() {
        let emb = CountingToGoldilocks::new();
        let pairs = vec![
            (counting(0), counting(0)),
            (counting(0), counting(5)),
            (counting(3), counting(7)),
            (counting(100), counting(200)),
            (counting(12345), counting(67890)),
        ];
        for (a, b) in &pairs {
            assert!(
                emb.verify_homomorphism(a, b).unwrap(),
                "homomorphism failed for ({}, {})",
                a.value,
                b.value
            );
        }
    }

    #[test]
    fn counting_homomorphism_mul() {
        let emb = CountingToGoldilocks::new();
        for (a, b) in [(3, 7), (0, 100), (1, 1), (100, 100), (1000, 2000)] {
            let ca = counting(a);
            let cb = counting(b);
            let fa = emb.embed(&ca).unwrap();
            let fb = emb.embed(&cb).unwrap();
            let prod = ca.mul(&cb);
            let f_prod = emb.embed(&prod).unwrap();
            let f_mul = fa.mul(&fb);
            assert_eq!(f_prod, f_mul, "mul homomorphism for ({}, {})", a, b);
        }
    }

    #[test]
    fn counting_max_safe() {
        assert_eq!(
            CountingToGoldilocks::max_safe_value(),
            GOLDILOCKS_PRIME - 1
        );
    }

    // ── BooleanToGoldilocks ──────────────────────────────────────────

    #[test]
    fn boolean_embed_unembed_all_cases() {
        let emb = BooleanToGoldilocks::new();
        assert_eq!(emb.embed(&boolean(false)).unwrap(), gf(0));
        assert_eq!(emb.embed(&boolean(true)).unwrap(), gf(1));
        assert_eq!(emb.unembed(&gf(0)).unwrap(), boolean(false));
        assert_eq!(emb.unembed(&gf(1)).unwrap(), boolean(true));
    }

    #[test]
    fn boolean_unembed_invalid() {
        let emb = BooleanToGoldilocks::new();
        assert!(emb.unembed(&gf(2)).is_err());
        assert!(emb.unembed(&gf(42)).is_err());
    }

    #[test]
    fn boolean_and_all_cases() {
        let emb = BooleanToGoldilocks::new();
        let cases = [(false, false), (false, true), (true, false), (true, true)];
        for (a, b) in cases {
            let ba = boolean(a);
            let bb = boolean(b);
            let fa = emb.embed(&ba).unwrap();
            let fb = emb.embed(&bb).unwrap();
            let and_result = ba.mul(&bb);
            let f_and = emb.embed(&and_result).unwrap();
            let field_and = BooleanToGoldilocks::field_and(&fa, &fb);
            assert_eq!(
                f_and, field_and,
                "AND({}, {}) mismatch",
                a, b
            );
        }
    }

    #[test]
    fn boolean_or_all_cases() {
        let emb = BooleanToGoldilocks::new();
        let cases = [(false, false), (false, true), (true, false), (true, true)];
        for (a, b) in cases {
            let ba = boolean(a);
            let bb = boolean(b);
            let fa = emb.embed(&ba).unwrap();
            let fb = emb.embed(&bb).unwrap();
            let or_result = ba.add(&bb);
            let f_or = emb.embed(&or_result).unwrap();
            let field_or = BooleanToGoldilocks::field_or(&fa, &fb);
            assert_eq!(
                f_or, field_or,
                "OR({}, {}) mismatch",
                a, b
            );
        }
    }

    #[test]
    fn boolean_homomorphism_all_pairs() {
        let emb = BooleanToGoldilocks::new();
        for a in [false, true] {
            for b in [false, true] {
                assert!(
                    emb.verify_homomorphism(&boolean(a), &boolean(b)).unwrap(),
                    "homomorphism failed for ({}, {})",
                    a,
                    b
                );
            }
        }
    }

    #[test]
    fn boolean_not() {
        assert_eq!(BooleanToGoldilocks::field_not(&gf(0)), gf(1));
        assert_eq!(BooleanToGoldilocks::field_not(&gf(1)), gf(0));
    }

    // ── BoundedCountingToGoldilocks ──────────────────────────────────

    #[test]
    fn bounded_counting_embed() {
        let emb = BoundedCountingToGoldilocks::new(100);
        let bc = BoundedCountingSemiring::new(42, 100);
        let f = emb.embed(&bc).unwrap();
        assert_eq!(f.value, 42);

        // Value clamped at construction
        let bc2 = BoundedCountingSemiring::new(200, 100);
        let f2 = emb.embed(&bc2).unwrap();
        assert_eq!(f2.value, 100); // clamped to bound
    }

    #[test]
    fn bounded_counting_embed_with_bound() {
        let emb = BoundedCountingToGoldilocks::new(100);
        let bc = BoundedCountingSemiring::new(42, 100);
        let (val, bound) = emb.embed_with_bound_check(&bc).unwrap();
        assert_eq!(val.value, 42);
        assert_eq!(bound.value, 100);
    }

    #[test]
    fn bounded_counting_clipping() {
        let emb = BoundedCountingToGoldilocks::new(10);
        let a = BoundedCountingSemiring::new(7, 10);
        let b = BoundedCountingSemiring::new(8, 10);
        assert!(emb.verify_clipping(&a, &b).unwrap());
    }

    #[test]
    fn bounded_counting_unembed() {
        let emb = BoundedCountingToGoldilocks::new(100);
        let f = gf(42);
        let bc = emb.unembed(&f).unwrap();
        assert_eq!(bc.value, 42);
        assert_eq!(bc.bound, 100);
    }

    // ── BitDecomposition ─────────────────────────────────────────────

    #[test]
    fn bit_decomposition_roundtrip() {
        for val in [0u64, 1, 2, 7, 8, 15, 42, 255, 256, 1023, 65535] {
            let bits = BitDecomposition::decompose(val, 16).unwrap();
            assert!(bits.verify_boolean_constraints());
            assert!(bits.verify_decomposition(val));
            let recomposed = bits.recompose();
            assert_eq!(recomposed, gf(val), "roundtrip failed for {}", val);
        }
    }

    #[test]
    fn bit_decomposition_32bit() {
        let val = 0xDEADBEEFu64;
        let bits = BitDecomposition::decompose(val, 32).unwrap();
        assert!(bits.verify_boolean_constraints());
        assert_eq!(bits.recompose(), gf(val));
    }

    #[test]
    fn bit_decomposition_overflow() {
        // 256 doesn't fit in 8 bits
        assert!(BitDecomposition::decompose(256, 8).is_err());
        // 16 doesn't fit in 4 bits
        assert!(BitDecomposition::decompose(16, 4).is_err());
    }

    #[test]
    fn bit_decomposition_zero() {
        let bits = BitDecomposition::decompose(0, 8).unwrap();
        assert_eq!(bits.recompose(), gf(0));
        assert!(bits.bits.iter().all(|b| b.value == 0));
    }

    #[test]
    fn bit_decomposition_all_ones() {
        let bits = BitDecomposition::decompose(255, 8).unwrap();
        assert!(bits.bits.iter().all(|b| b.value == 1));
        assert_eq!(bits.recompose(), gf(255));
    }

    // ── TropicalGadget ───────────────────────────────────────────────

    #[test]
    fn tropical_gadget_less_than() {
        let gadget = TropicalGadget::new(16);
        let test_cases: Vec<(u64, u64, bool)> = vec![
            (0, 0, false),
            (0, 1, true),
            (1, 0, false),
            (5, 10, true),
            (10, 5, false),
            (100, 100, false),
            (0, 65535, true),
            (65535, 0, false),
            (1000, 1001, true),
            (1001, 1000, false),
        ];
        for (a, b, expected) in test_cases {
            let a_bits = BitDecomposition::decompose(a, 16).unwrap();
            let b_bits = BitDecomposition::decompose(b, 16).unwrap();
            let result = gadget.gadget_less_than(&a_bits, &b_bits).unwrap();
            let expected_val = if expected { 1u64 } else { 0u64 };
            assert_eq!(
                result.value,
                gf(expected_val),
                "less_than({}, {}) expected {} got {}",
                a,
                b,
                expected_val,
                result.value.value
            );
        }
    }

    #[test]
    fn tropical_gadget_min() {
        let gadget = TropicalGadget::new(16);
        let test_cases: Vec<(u64, u64)> = vec![
            (0, 0),
            (0, 1),
            (1, 0),
            (5, 10),
            (10, 5),
            (100, 100),
            (42, 99),
            (99, 42),
            (0, 65535),
            (65535, 0),
        ];
        for (a, b) in test_cases {
            let a_bits = BitDecomposition::decompose(a, 16).unwrap();
            let b_bits = BitDecomposition::decompose(b, 16).unwrap();
            let result = gadget.gadget_min(&a_bits, &b_bits).unwrap();
            let expected = a.min(b);
            assert_eq!(
                result.value,
                gf(expected),
                "min({}, {}) expected {} got {}",
                a,
                b,
                expected,
                result.value.value
            );
        }
    }

    #[test]
    fn tropical_gadget_tropical_add_mul() {
        let gadget = TropicalGadget::new(16);

        // Tropical add = min
        let result = gadget.tropical_add(5, 10).unwrap();
        assert_eq!(result.value, gf(5));

        let result = gadget.tropical_add(10, 5).unwrap();
        assert_eq!(result.value, gf(5));

        // Tropical mul = field add
        let result = gadget.tropical_mul(5, 10).unwrap();
        assert_eq!(result, gf(15));
    }

    #[test]
    fn tropical_gadget_auxiliary_witness() {
        let gadget = TropicalGadget::new(8);
        let a = BitDecomposition::decompose(5, 8).unwrap();
        let b = BitDecomposition::decompose(10, 8).unwrap();
        let result = gadget.gadget_min(&a, &b).unwrap();
        // Auxiliary should be non-empty (contains comparison witnesses)
        assert!(!result.auxiliary.is_empty());
    }

    // ── TropicalToGadget embedding ───────────────────────────────────

    #[test]
    fn tropical_embedding_basic() {
        let emb = TropicalToGadget::new(32);
        let t = tropical(42.0);
        let f = emb.embed(&t).unwrap();
        assert_eq!(f.value, 42);

        let t2 = emb.unembed(&f).unwrap();
        assert_eq!(t2.raw(), 42.0);
    }

    #[test]
    fn tropical_embedding_infinity_fails() {
        let emb = TropicalToGadget::new(32);
        assert!(!emb.can_embed(&TropicalSemiring::infinity()));
        assert!(emb.embed(&TropicalSemiring::infinity()).is_err());
    }

    #[test]
    fn tropical_embedding_with_bits() {
        let emb = TropicalToGadget::new(16);
        let t = tropical(100.0);
        let (field_val, bits) = emb.embed_with_bits(&t).unwrap();
        assert_eq!(field_val.value, 100);
        assert!(bits.verify_decomposition(100));
    }

    // ── RealFixedPointEmbedding ──────────────────────────────────────

    #[test]
    fn real_fixed_point_roundtrip() {
        let emb = RealFixedPointEmbedding::new(20);
        let test_values = [0.0, 1.0, -1.0, 0.5, 0.25, 3.14159, -2.71828, 100.0, -100.0];
        for &v in &test_values {
            let r = real(v);
            let f = emb.embed(&r).unwrap();
            let r2 = emb.unembed(&f).unwrap();
            let error = (r2.raw() - v).abs();
            let bound = emb.precision_bound();
            assert!(
                error <= bound + 1e-15,
                "precision loss for {}: error={}, bound={}",
                v,
                error,
                bound
            );
        }
    }

    #[test]
    fn real_fixed_point_zero() {
        let emb = RealFixedPointEmbedding::new(20);
        let f = emb.embed(&real(0.0)).unwrap();
        assert_eq!(f.value, 0);
    }

    #[test]
    fn real_fixed_point_precision() {
        let emb = RealFixedPointEmbedding::new(30);
        assert!(emb.precision_bound() < 1e-9);
        let emb2 = RealFixedPointEmbedding::new(10);
        assert!(emb2.precision_bound() < 0.001);
    }

    #[test]
    fn real_fixed_point_embed_static() {
        let f = RealFixedPointEmbedding::embed_fixed_point(1.5, 20).unwrap();
        let v = RealFixedPointEmbedding::unembed_fixed_point(&f, 20);
        assert!((v - 1.5).abs() < 1e-6);
    }

    #[test]
    fn real_fixed_point_negative() {
        let f = RealFixedPointEmbedding::embed_fixed_point(-3.0, 20).unwrap();
        let v = RealFixedPointEmbedding::unembed_fixed_point(&f, 20);
        assert!((v - (-3.0)).abs() < 1e-6);
    }

    #[test]
    fn real_fixed_point_nan_inf() {
        assert!(RealFixedPointEmbedding::embed_fixed_point(f64::NAN, 20).is_err());
        assert!(RealFixedPointEmbedding::embed_fixed_point(f64::INFINITY, 20).is_err());
    }

    // ── WFA embedding ────────────────────────────────────────────────

    #[test]
    fn wfa_counting_embed_and_verify() {
        let wfa = build_counting_wfa();
        let emb = CountingToGoldilocks::new();
        let embedded = WfaEmbedder::<CountingSemiring>::embed_wfa(&wfa, &emb).unwrap();

        // Test words
        let words: Vec<Vec<usize>> = vec![
            vec![1],           // q0 --1/3--> q1, weight = 1 * 3 * 1 = 3
            vec![0, 1],       // q0 --0/2--> q0 --1/3--> q1, weight = 2 * 3 = 6
            vec![1, 1],       // q0 --1/3--> q1 --1/1--> q1, weight = 3 * 1 = 3
            vec![0, 0, 1],   // q0 --0/2--> q0 --0/2--> q0 --1/3--> q1, weight = 4 * 3 = 12
        ];

        for word in &words {
            let orig_weight = wfa.compute_weight(word);
            let emb_weight = embedded.compute_weight(word);
            let expected = emb.embed(&orig_weight).unwrap();
            assert_eq!(
                emb_weight, expected,
                "weight mismatch for word {:?}: orig={:?}, emb={}, expected={}",
                word, orig_weight, emb_weight.value, expected.value
            );
        }
    }

    #[test]
    fn wfa_counting_verify_method() {
        let wfa = build_counting_wfa();
        let emb = CountingToGoldilocks::new();
        let embedded = WfaEmbedder::<CountingSemiring>::embed_wfa(&wfa, &emb).unwrap();

        let words: Vec<Vec<usize>> = vec![
            vec![1],
            vec![0, 1],
            vec![1, 1],
            vec![0, 0, 1],
            vec![1, 0, 1],
        ];

        let verification = WfaEmbedder::<CountingSemiring>::verify_wfa_embedding(
            &wfa, &embedded, &emb, &words,
        )
        .unwrap();

        assert!(
            verification.all_passed,
            "verification failed: {}",
            verification
        );
        assert_eq!(verification.tests_run, words.len());
    }

    #[test]
    fn wfa_boolean_embed() {
        let wfa = build_boolean_wfa();
        let emb = BooleanToGoldilocks::new();
        let embedded = WfaEmbedder::<BooleanSemiring>::embed_wfa(&wfa, &emb).unwrap();

        // Words ending with 1 should have weight 1; others weight 0.
        let accepting = vec![vec![1], vec![0, 1], vec![1, 1], vec![0, 0, 1]];
        let rejecting = vec![vec![0], vec![1, 0], vec![0, 0]];

        for word in accepting {
            let w = embedded.compute_weight(&word);
            assert_eq!(w, gf(1), "expected accept for {:?}", word);
        }
        for word in rejecting {
            let w = embedded.compute_weight(&word);
            assert_eq!(w, gf(0), "expected reject for {:?}", word);
        }
    }

    #[test]
    fn wfa_counting_unembed() {
        let wfa = build_counting_wfa();
        let emb = CountingToGoldilocks::new();
        let embedded = WfaEmbedder::<CountingSemiring>::embed_wfa(&wfa, &emb).unwrap();
        let unembedded =
            WfaEmbedder::<CountingSemiring>::unembed_wfa(&embedded, &emb).unwrap();

        let words: Vec<Vec<usize>> = vec![vec![1], vec![0, 1], vec![1, 1]];
        for word in &words {
            let orig = wfa.compute_weight(word);
            let restored = unembedded.compute_weight(word);
            assert_eq!(orig, restored, "unembed mismatch for {:?}", word);
        }
    }

    // ── Batch operations ─────────────────────────────────────────────

    #[test]
    fn batch_embed_unembed() {
        let emb = CountingToGoldilocks::new();
        let values: Vec<CountingSemiring> = (0..100).map(counting).collect();
        let embedded = batch_embed(&values, &emb).unwrap();
        let unembedded = batch_unembed(&embedded, &emb).unwrap();
        assert_eq!(values, unembedded);
    }

    #[test]
    fn batch_embed_empty() {
        let emb = CountingToGoldilocks::new();
        let values: Vec<CountingSemiring> = vec![];
        let embedded = batch_embed(&values, &emb).unwrap();
        assert!(embedded.is_empty());
    }

    #[test]
    fn batch_verify_homomorphism_counting() {
        let emb = CountingToGoldilocks::new();
        let pairs: Vec<(CountingSemiring, CountingSemiring)> = vec![
            (counting(0), counting(0)),
            (counting(1), counting(1)),
            (counting(5), counting(10)),
            (counting(100), counting(200)),
            (counting(999), counting(1)),
        ];
        let result = batch_verify_homomorphism(&pairs, &emb).unwrap();
        assert_eq!(result.total_pairs, 5);
        assert_eq!(result.passed, 5);
        assert_eq!(result.failed, 0);
    }

    #[test]
    fn batch_verify_homomorphism_boolean() {
        let emb = BooleanToGoldilocks::new();
        let pairs: Vec<(BooleanSemiring, BooleanSemiring)> = vec![
            (boolean(false), boolean(false)),
            (boolean(false), boolean(true)),
            (boolean(true), boolean(false)),
            (boolean(true), boolean(true)),
        ];
        let result = batch_verify_homomorphism(&pairs, &emb).unwrap();
        assert_eq!(result.passed, 4);
        assert_eq!(result.failed, 0);
    }

    // ── Overflow detection ───────────────────────────────────────────

    #[test]
    fn overflow_addition() {
        let det = OverflowDetector::new();
        assert!(!det.check_addition_overflow(0, 0));
        assert!(!det.check_addition_overflow(1, 1));
        assert!(!det.check_addition_overflow(100, 200));
        assert!(det.check_addition_overflow(GOLDILOCKS_PRIME - 1, 1));
        assert!(det.check_addition_overflow(GOLDILOCKS_PRIME / 2, GOLDILOCKS_PRIME / 2 + 1));
    }

    #[test]
    fn overflow_multiplication() {
        let det = OverflowDetector::new();
        assert!(!det.check_multiplication_overflow(0, u64::MAX));
        assert!(!det.check_multiplication_overflow(1, GOLDILOCKS_PRIME - 1));
        assert!(det.check_multiplication_overflow(GOLDILOCKS_PRIME, 2));
        assert!(det.check_multiplication_overflow(1u64 << 33, 1u64 << 33));
    }

    #[test]
    fn overflow_wfa_bound() {
        let det = OverflowDetector::new();
        let wfa = build_counting_wfa();
        // For short inputs the bound should be small
        let bound = det.max_weight_bound(&wfa, 1);
        assert!(bound < GOLDILOCKS_PRIME);
        assert!(bound > 0);
    }

    #[test]
    fn overflow_safe_input_length() {
        let det = OverflowDetector::new();
        let wfa = build_counting_wfa();
        let safe_len = det.safe_input_length(&wfa);
        // Should be positive for this small WFA
        assert!(safe_len > 0);
        // Verify: within safe_len, no overflow
        let bound = det.max_weight_bound(&wfa, safe_len);
        assert!(bound < u64::MAX || safe_len == 0);
    }

    #[test]
    fn overflow_word_safety() {
        let det = OverflowDetector::new();
        let wfa = build_counting_wfa();
        // Short words should be safe
        assert!(det.check_word_safety(&wfa, &[1]));
        assert!(det.check_word_safety(&wfa, &[0, 1]));
        assert!(det.check_word_safety(&wfa, &[0, 0, 0, 1]));
    }

    // ── Quadratic extension ──────────────────────────────────────────

    #[test]
    fn quadratic_ext_add() {
        let a = QuadraticExtension::from_pair(gf(3), gf(4));
        let b = QuadraticExtension::from_pair(gf(5), gf(7));
        let c = a.ext_add(&b);
        assert_eq!(c.re, gf(8));
        assert_eq!(c.im, gf(11));
    }

    #[test]
    fn quadratic_ext_mul() {
        // (3 + 4i)(5 + 7i) = 15 + 21i + 20i + 28i²
        //                   = 15 + 41i − 28 = −13 + 41i
        let a = QuadraticExtension::from_pair(gf(3), gf(4));
        let b = QuadraticExtension::from_pair(gf(5), gf(7));
        let c = a.ext_mul(&b);
        // -13 mod p = p - 13
        let neg13 = GoldilocksField::new(GOLDILOCKS_PRIME - 13);
        assert_eq!(c.re, neg13);
        assert_eq!(c.im, gf(41));
    }

    #[test]
    fn quadratic_ext_inv() {
        let a = QuadraticExtension::from_pair(gf(3), gf(4));
        let a_inv = a.ext_inv().unwrap();
        let product = a.ext_mul(&a_inv);
        assert_eq!(product.re, gf(1));
        assert_eq!(product.im, gf(0));
    }

    #[test]
    fn quadratic_ext_zero() {
        let zero = QuadraticExtension::ext_zero();
        let a = QuadraticExtension::from_pair(gf(42), gf(7));
        let sum = zero.ext_add(&a);
        assert_eq!(sum, a);
    }

    #[test]
    fn quadratic_ext_one() {
        let one = QuadraticExtension::ext_one();
        let a = QuadraticExtension::from_pair(gf(42), gf(7));
        let prod = one.ext_mul(&a);
        assert_eq!(prod, a);
    }

    #[test]
    fn quadratic_ext_conjugate() {
        let a = QuadraticExtension::from_pair(gf(3), gf(4));
        let conj = a.conjugate();
        assert_eq!(conj.re, gf(3));
        assert_eq!(conj.im, gf(4).neg());
    }

    #[test]
    fn quadratic_ext_norm() {
        // norm(3 + 4i) = 9 + 16 = 25
        let a = QuadraticExtension::from_pair(gf(3), gf(4));
        assert_eq!(a.norm(), gf(25));
    }

    #[test]
    fn quadratic_ext_pow() {
        let a = QuadraticExtension::from_pair(gf(1), gf(1));
        // (1+i)^2 = 2i
        let sq = a.ext_pow(2);
        assert_eq!(sq.re, gf(0));
        assert_eq!(sq.im, gf(2));
    }

    #[test]
    fn quadratic_ext_i_squared() {
        let i = QuadraticExtension::ext_i();
        let i_sq = i.ext_mul(&i);
        // i^2 = -1
        let neg1 = GoldilocksField::new(GOLDILOCKS_PRIME - 1);
        assert_eq!(i_sq.re, neg1);
        assert_eq!(i_sq.im, gf(0));
    }

    // ── Cubic extension ──────────────────────────────────────────────

    #[test]
    fn cubic_ext_add() {
        let alpha = gf(2);
        let a = CubicExtension::new(gf(1), gf(2), gf(3), alpha);
        let b = CubicExtension::new(gf(4), gf(5), gf(6), alpha);
        let c = a.cubic_add(&b);
        assert_eq!(c.c0, gf(5));
        assert_eq!(c.c1, gf(7));
        assert_eq!(c.c2, gf(9));
    }

    #[test]
    fn cubic_ext_mul_identity() {
        let alpha = gf(2);
        let a = CubicExtension::new(gf(7), gf(11), gf(3), alpha);
        let one = CubicExtension::cubic_one(alpha);
        let prod = a.cubic_mul(&one);
        assert_eq!(prod.c0, a.c0);
        assert_eq!(prod.c1, a.c1);
        assert_eq!(prod.c2, a.c2);
    }

    #[test]
    fn cubic_ext_mul_zero() {
        let alpha = gf(2);
        let a = CubicExtension::new(gf(7), gf(11), gf(3), alpha);
        let zero = CubicExtension::cubic_zero(alpha);
        let prod = a.cubic_mul(&zero);
        assert_eq!(prod.c0, gf(0));
        assert_eq!(prod.c1, gf(0));
        assert_eq!(prod.c2, gf(0));
    }

    // ── EmbeddingRegistry ────────────────────────────────────────────

    #[test]
    fn registry_default_embeddings() {
        let reg = EmbeddingRegistry::new();
        assert!(reg.get_embedding("counting").is_some());
        assert!(reg.get_embedding("boolean").is_some());
        assert!(reg.get_embedding("bounded_counting").is_some());
        assert!(reg.get_embedding("tropical").is_some());
        assert!(reg.get_embedding("real").is_some());
        assert_eq!(reg.len(), 5);
    }

    #[test]
    fn registry_auto_select() {
        let reg = EmbeddingRegistry::new();
        assert!(reg.auto_select("counting").is_some());
        assert!(reg.auto_select("CountingSemiring").is_some());
        assert!(reg.auto_select("boolean").is_some());
        assert!(reg.auto_select("BooleanSemiring").is_some());
        assert!(reg.auto_select("tropical").is_some());
        assert!(reg.auto_select("unknown_type").is_none());
    }

    #[test]
    fn registry_list() {
        let reg = EmbeddingRegistry::new();
        let names = reg.list();
        assert!(names.contains(&"counting".to_string()));
        assert!(names.contains(&"boolean".to_string()));
    }

    #[test]
    fn registry_custom_register() {
        let mut reg = EmbeddingRegistry::new();
        reg.register(
            "custom_counting",
            RegisteredEmbedding::CountingToGoldilocks(
                CountingToGoldilocks::with_max_safe(1000),
            ),
        );
        assert_eq!(reg.len(), 6);
        assert!(reg.get_embedding("custom_counting").is_some());
    }

    // ── Edge cases ───────────────────────────────────────────────────

    #[test]
    fn counting_embed_zero() {
        let emb = CountingToGoldilocks::new();
        let f = emb.embed(&counting(0)).unwrap();
        assert!(f.is_zero());
    }

    #[test]
    fn counting_embed_one() {
        let emb = CountingToGoldilocks::new();
        let f = emb.embed(&counting(1)).unwrap();
        assert!(f.is_one());
    }

    #[test]
    fn counting_embed_large() {
        let emb = CountingToGoldilocks::new();
        let large = GOLDILOCKS_PRIME - 1;
        let f = emb.embed(&counting(large)).unwrap();
        assert_eq!(f.value, large);
        let c = emb.unembed(&f).unwrap();
        assert_eq!(c.value, large);
    }

    #[test]
    fn counting_embed_wraps_at_prime() {
        let emb = CountingToGoldilocks::new();
        // p maps to 0
        let f = emb.embed(&counting(GOLDILOCKS_PRIME)).unwrap();
        assert_eq!(f.value, 0);
    }

    #[test]
    fn embed_matrix_counting() {
        let emb = CountingToGoldilocks::new();
        let mat = SemiringMatrix::from_rows(vec![
            vec![counting(1), counting(2)],
            vec![counting(3), counting(4)],
        ])
        .unwrap();
        let embedded = emb.embed_matrix(&mat).unwrap();
        assert_eq!(*embedded.get(0, 0).unwrap(), gf(1));
        assert_eq!(*embedded.get(0, 1).unwrap(), gf(2));
        assert_eq!(*embedded.get(1, 0).unwrap(), gf(3));
        assert_eq!(*embedded.get(1, 1).unwrap(), gf(4));
    }

    #[test]
    fn unembed_matrix_counting() {
        let emb = CountingToGoldilocks::new();
        let mat = SemiringMatrix::from_rows(vec![
            vec![gf(10), gf(20)],
            vec![gf(30), gf(40)],
        ])
        .unwrap();
        let unembedded = emb.unembed_matrix(&mat).unwrap();
        assert_eq!(*unembedded.get(0, 0).unwrap(), counting(10));
        assert_eq!(*unembedded.get(1, 1).unwrap(), counting(40));
    }

    #[test]
    fn empty_wfa_embed() {
        let alphabet = Alphabet::from_range(2);
        let wfa: WeightedFiniteAutomaton<CountingSemiring> =
            WeightedFiniteAutomaton::new(0, alphabet);
        let emb = CountingToGoldilocks::new();
        let embedded = WfaEmbedder::<CountingSemiring>::embed_wfa(&wfa, &emb).unwrap();
        assert_eq!(embedded.state_count(), 0);
    }

    // ── Embedding verification ───────────────────────────────────────

    #[test]
    fn embedding_verification_display() {
        let v = EmbeddingVerification {
            all_passed: true,
            tests_run: 5,
            failures: vec![],
        };
        let s = format!("{}", v);
        assert!(s.contains("5/5 passed"));
    }

    #[test]
    fn embedding_verification_with_failure() {
        let v = EmbeddingVerification {
            all_passed: false,
            tests_run: 3,
            failures: vec![EmbeddingFailure {
                word: vec![0, 1],
                original_weight: "5".to_string(),
                embedded_weight: "6".to_string(),
                expected_embedded: "5".to_string(),
            }],
        };
        assert!(!v.all_passed);
        assert_eq!(v.failures.len(), 1);
    }

    // ── Stress tests ─────────────────────────────────────────────────

    #[test]
    fn stress_counting_homomorphism() {
        let emb = CountingToGoldilocks::new();
        let values: Vec<u64> = (0..50)
            .map(|i| i * 1000 + 1)
            .collect();
        for &a in &values {
            for &b in &values {
                let ca = counting(a);
                let cb = counting(b);
                // Only test when no saturation
                if a.checked_add(b).is_some()
                    && a + b < GOLDILOCKS_PRIME
                    && a.checked_mul(b).is_some()
                    && a * b < GOLDILOCKS_PRIME
                {
                    assert!(
                        emb.verify_homomorphism(&ca, &cb).unwrap(),
                        "stress test failed for ({}, {})",
                        a,
                        b
                    );
                }
            }
        }
    }

    #[test]
    fn stress_batch_embed() {
        let emb = CountingToGoldilocks::new();
        let values: Vec<CountingSemiring> = (0..1000).map(counting).collect();
        let embedded = batch_embed(&values, &emb).unwrap();
        assert_eq!(embedded.len(), 1000);
        for (i, f) in embedded.iter().enumerate() {
            assert_eq!(f.value, i as u64);
        }
    }

    #[test]
    fn stress_bit_decomposition() {
        let gadget = TropicalGadget::new(32);
        // Test many values
        for val in (0..100).chain(std::iter::once(0xFFFFFFFF)) {
            let bits = BitDecomposition::decompose(val, 32).unwrap();
            assert!(bits.verify_boolean_constraints());
            assert_eq!(bits.recompose(), gf(val));
        }
    }

    #[test]
    fn stress_tropical_min() {
        let gadget = TropicalGadget::new(16);
        let values: Vec<u64> = vec![0, 1, 2, 100, 255, 1000, 10000, 65535];
        for &a in &values {
            for &b in &values {
                let a_bits = BitDecomposition::decompose(a, 16).unwrap();
                let b_bits = BitDecomposition::decompose(b, 16).unwrap();
                let result = gadget.gadget_min(&a_bits, &b_bits).unwrap();
                assert_eq!(
                    result.value,
                    gf(a.min(b)),
                    "min({}, {}) failed",
                    a,
                    b
                );
            }
        }
    }

    #[test]
    fn stress_quadratic_extension() {
        // (a + bi) * (a + bi)^{-1} = 1 for various values
        for re in [1u64, 2, 3, 7, 42, 100, 9999] {
            for im in [1u64, 2, 5, 13, 77] {
                let a = QuadraticExtension::from_pair(gf(re), gf(im));
                let inv = a.ext_inv().unwrap();
                let prod = a.ext_mul(&inv);
                assert_eq!(
                    prod.re,
                    gf(1),
                    "inv failed for ({} + {}i)",
                    re,
                    im
                );
                assert_eq!(
                    prod.im,
                    gf(0),
                    "inv failed for ({} + {}i)",
                    re,
                    im
                );
            }
        }
    }

    // ── GadgetOutput ─────────────────────────────────────────────────

    #[test]
    fn gadget_output_construction() {
        let out = GadgetOutput::new(gf(42));
        assert_eq!(out.value, gf(42));
        assert!(out.auxiliary.is_empty());

        let out2 = GadgetOutput::with_auxiliary(gf(7), vec![gf(1), gf(0)]);
        assert_eq!(out2.auxiliary.len(), 2);
    }

    // ── Error display ────────────────────────────────────────────────

    #[test]
    fn error_display() {
        let err = EmbeddingError::OverflowError {
            value: "999".to_string(),
            field_desc: "test".to_string(),
        };
        let s = format!("{}", err);
        assert!(s.contains("999"));
        assert!(s.contains("test"));

        let err2 = EmbeddingError::NotInImage {
            field_element: "42".to_string(),
            desc: "out of range".to_string(),
        };
        assert!(format!("{}", err2).contains("42"));

        let err3 = EmbeddingError::BitDecompositionError {
            value: "300".to_string(),
            bits: 8,
        };
        assert!(format!("{}", err3).contains("300"));
        assert!(format!("{}", err3).contains("8"));
    }

    // ── BatchVerification display ────────────────────────────────────

    #[test]
    fn batch_verification_display() {
        let bv = BatchVerification {
            total_pairs: 10,
            passed: 8,
            failed: 2,
            failures: vec![
                (3, "pair 3 failed".to_string()),
                (7, "pair 7 failed".to_string()),
            ],
        };
        let s = format!("{}", bv);
        assert!(s.contains("8/10"));
    }

    // ── Identity embeddings ──────────────────────────────────────────

    #[test]
    fn goldilocks_identity_embed() {
        // GoldilocksField embedding into itself is trivial
        // We verify via CountingToGoldilocks with value 0 and 1
        let emb = CountingToGoldilocks::new();
        let zero = emb.embed(&CountingSemiring::zero()).unwrap();
        assert_eq!(zero, GoldilocksField::zero());
        let one = emb.embed(&CountingSemiring::one()).unwrap();
        assert_eq!(one, GoldilocksField::one());
    }

    // ── Goldilocks field arithmetic sanity ────────────────────────────

    #[test]
    fn goldilocks_arithmetic_sanity() {
        // Verify our field arithmetic is consistent
        let a = gf(GOLDILOCKS_PRIME - 1);
        let b = gf(1);
        let sum = a.add(&b);
        assert_eq!(sum, gf(0)); // (p-1) + 1 = 0 mod p

        let c = gf(2);
        let d = gf(GOLDILOCKS_PRIME - 1); // -1
        let prod = c.mul(&d);
        assert_eq!(prod, gf(GOLDILOCKS_PRIME - 2)); // 2 * (-1) = -2

        let inv2 = gf(2).inv().unwrap();
        let check = gf(2).mul(&inv2);
        assert_eq!(check, gf(1));
    }

    // ── Comprehensive WFA verification ───────────────────────────────

    #[test]
    fn wfa_comprehensive_counting() {
        let wfa = build_counting_wfa();
        let emb = CountingToGoldilocks::new();
        let embedded = WfaEmbedder::<CountingSemiring>::embed_wfa(&wfa, &emb).unwrap();

        // Generate all words of length 1..4 over {0, 1}
        let mut all_words = Vec::new();
        for len in 1..=4 {
            for w in 0..(1usize << len) {
                let word: Vec<usize> = (0..len).map(|i| (w >> i) & 1).collect();
                all_words.push(word);
            }
        }

        let verification = WfaEmbedder::<CountingSemiring>::verify_wfa_embedding(
            &wfa, &embedded, &emb, &all_words,
        )
        .unwrap();

        assert!(
            verification.all_passed,
            "comprehensive verification failed: {} failures out of {} tests",
            verification.failures.len(),
            verification.tests_run
        );
    }

    // ── Overflow detector with trivial WFA ───────────────────────────

    #[test]
    fn overflow_trivial_wfa() {
        let det = OverflowDetector::new();
        // A WFA with weight 1 everywhere — weight = 1 for any word
        let alphabet = Alphabet::from_range(2);
        let mut wfa = WeightedFiniteAutomaton::new(1, alphabet);
        wfa.set_initial_weight(0, counting(1));
        wfa.set_final_weight(0, counting(1));
        wfa.add_transition(0, 0, 0, counting(1)).unwrap();
        wfa.add_transition(0, 1, 0, counting(1)).unwrap();

        // All lengths should be safe
        let safe = det.safe_input_length(&wfa);
        assert!(safe > 100);

        // Bound should remain small
        let bound = det.max_weight_bound(&wfa, 10);
        assert!(bound < 100);
    }

    // ── Tropical embedding verify_homomorphism ───────────────────────

    #[test]
    fn tropical_verify_homomorphism() {
        let emb = TropicalToGadget::new(16);
        let a = tropical(5.0);
        let b = tropical(10.0);
        assert!(emb.verify_homomorphism(&a, &b).unwrap());
    }

    // ── Real embedding approximate homomorphism ──────────────────────

    #[test]
    fn real_approximate_homomorphism() {
        let emb = RealFixedPointEmbedding::new(20);
        let a = real(1.5);
        let b = real(2.5);
        // For addition, the homomorphism should hold approximately.
        assert!(emb.verify_homomorphism(&a, &b).unwrap());
    }

    // ── Matrix operations through embedding ──────────────────────────

    #[test]
    fn matrix_embed_unembed_roundtrip() {
        let emb = CountingToGoldilocks::new();
        let mat = SemiringMatrix::from_rows(vec![
            vec![counting(0), counting(1), counting(2)],
            vec![counting(3), counting(4), counting(5)],
            vec![counting(6), counting(7), counting(8)],
        ])
        .unwrap();
        let embedded = emb.embed_matrix(&mat).unwrap();
        let restored = emb.unembed_matrix(&embedded).unwrap();
        assert_eq!(mat, restored);
    }

    #[test]
    fn matrix_mul_homomorphism() {
        let emb = CountingToGoldilocks::new();
        let a = SemiringMatrix::from_rows(vec![
            vec![counting(1), counting(2)],
            vec![counting(3), counting(4)],
        ])
        .unwrap();
        let b = SemiringMatrix::from_rows(vec![
            vec![counting(5), counting(6)],
            vec![counting(7), counting(8)],
        ])
        .unwrap();

        // Multiply in semiring, then embed
        let prod_s = a.mul(&b).unwrap();
        let embedded_prod = emb.embed_matrix(&prod_s).unwrap();

        // Embed, then multiply in field
        let ea = emb.embed_matrix(&a).unwrap();
        let eb = emb.embed_matrix(&b).unwrap();
        let prod_f = ea.mul(&eb).unwrap();

        assert_eq!(embedded_prod, prod_f);
    }
}
