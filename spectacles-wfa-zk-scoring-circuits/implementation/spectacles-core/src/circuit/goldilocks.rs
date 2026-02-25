// Goldilocks field: F_p where p = 2^64 - 2^32 + 1
//
// This prime has excellent properties for NTT and CPU-efficient arithmetic:
// - Fits in a u64 (just barely)
// - Has a multiplicative group of order 2^64 - 2^32, divisible by 2^32
// - Supports NTTs of length up to 2^32
// - Reduction is efficient via the special form of p

use std::fmt;
use std::ops::{Add, Sub, Mul, Div, Neg, AddAssign, SubAssign, MulAssign};
use std::hash::{Hash, Hasher};
use serde::{Serialize, Deserialize};

/// The Goldilocks prime: p = 2^64 - 2^32 + 1
pub const GOLDILOCKS_PRIME: u64 = 0xFFFFFFFF00000001;

/// Number of bytes in a field element
pub const FIELD_BYTES: usize = 8;

/// A primitive root of unity of order 2^32 in the Goldilocks field.
/// g = 7 is a generator of the multiplicative group; we compute
/// g^((p-1)/2^32) to get the 2^32-th root of unity.
pub const TWO_ADIC_ROOT_OF_UNITY: u64 = 0x185629DCDA58878C;

/// The maximum power-of-two subgroup order: 2^32
pub const MAX_TWO_ADICITY: u32 = 32;

/// Montgomery form constant R = 2^64 mod p
const MONTGOMERY_R: u64 = 0xFFFFFFFF;

/// Montgomery form constant R^2 mod p
const MONTGOMERY_R2: u64 = 0xFFFFFFFE00000001;

/// Montgomery form constant: -p^{-1} mod 2^64
const MONTGOMERY_INV: u64 = 0xFFFFFFFEFFFFFFFF;

// ─────────────────────────────────────────────────────────────
// GoldilocksField
// ─────────────────────────────────────────────────────────────

/// A field element in the Goldilocks prime field F_p, p = 2^64 - 2^32 + 1.
///
/// Internally stored in canonical form [0, p).
#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct GoldilocksField {
    /// The value in [0, p)
    val: u64,
}

impl GoldilocksField {
    /// The additive identity.
    pub const ZERO: Self = Self { val: 0 };
    /// The multiplicative identity.
    pub const ONE: Self = Self { val: 1 };
    /// The value 2.
    pub const TWO: Self = Self { val: 2 };
    /// The prime modulus.
    pub const MODULUS: u64 = GOLDILOCKS_PRIME;

    /// Create a new field element from a u64, reducing mod p.
    #[inline]
    pub fn new(v: u64) -> Self {
        Self { val: Self::reduce(v) }
    }

    /// Create a field element from a value known to be in [0, p).
    #[inline]
    pub const fn from_canonical(v: u64) -> Self {
        Self { val: v }
    }

    /// Return the canonical u64 representation in [0, p).
    #[inline]
    pub const fn to_canonical(&self) -> u64 {
        self.val
    }

    /// Reduce a u64 modulo p.
    #[inline]
    fn reduce(v: u64) -> u64 {
        if v >= GOLDILOCKS_PRIME {
            v - GOLDILOCKS_PRIME
        } else {
            v
        }
    }

    /// Reduce a u128 modulo p using the special structure of Goldilocks prime.
    /// p = 2^64 - 2^32 + 1, so 2^64 ≡ 2^32 - 1 (mod p).
    /// For x = x_hi * 2^64 + x_lo, we get x ≡ x_lo + x_hi * (2^32 - 1) (mod p).
    #[inline]
    fn reduce_u128(x: u128) -> u64 {
        let x_lo = x as u64;
        let x_hi = (x >> 64) as u64;

        // x_hi * (2^32 - 1) = x_hi * 2^32 - x_hi
        let hi_shifted = (x_hi as u128) << 32;
        let correction = hi_shifted - (x_hi as u128);

        let sum = (x_lo as u128) + correction;
        let result_lo = sum as u64;
        let carry = (sum >> 64) as u64;

        // Handle the carry: again, 2^64 ≡ 2^32 - 1
        let with_carry = result_lo as u128 + (carry as u128) * ((1u128 << 32) - 1);

        let mut result = with_carry as u64;
        let carry2 = (with_carry >> 64) as u64;

        if carry2 > 0 {
            // One more level of reduction
            let adj = (carry2 as u128) * ((1u128 << 32) - 1);
            let total = result as u128 + adj;
            result = total as u64;
            // At this point, result should be < 2p, so one conditional subtraction suffices.
        }

        if result >= GOLDILOCKS_PRIME {
            result -= GOLDILOCKS_PRIME;
        }
        result
    }

    /// Additive inverse: -a mod p.
    #[inline]
    pub fn neg_elem(self) -> Self {
        if self.val == 0 {
            Self::ZERO
        } else {
            Self { val: GOLDILOCKS_PRIME - self.val }
        }
    }

    /// Field addition.
    #[inline]
    pub fn add_elem(self, rhs: Self) -> Self {
        let (sum, carry) = self.val.overflowing_add(rhs.val);
        let reduced = if carry {
            // sum + (2^32 - 1) since 2^64 ≡ 2^32 - 1 mod p
            sum.wrapping_add(0xFFFFFFFF)
        } else if sum >= GOLDILOCKS_PRIME {
            sum - GOLDILOCKS_PRIME
        } else {
            sum
        };
        Self { val: reduced }
    }

    /// Field subtraction.
    #[inline]
    pub fn sub_elem(self, rhs: Self) -> Self {
        if self.val >= rhs.val {
            Self { val: self.val - rhs.val }
        } else {
            Self { val: GOLDILOCKS_PRIME - (rhs.val - self.val) }
        }
    }

    /// Field multiplication using u128.
    #[inline]
    pub fn mul_elem(self, rhs: Self) -> Self {
        let product = (self.val as u128) * (rhs.val as u128);
        Self { val: Self::reduce_u128(product) }
    }

    /// Square the field element.
    #[inline]
    pub fn square(self) -> Self {
        self.mul_elem(self)
    }

    /// Double the field element.
    #[inline]
    pub fn double(self) -> Self {
        self.add_elem(self)
    }

    /// Compute self^exp using binary exponentiation.
    pub fn pow(self, mut exp: u64) -> Self {
        let mut base = self;
        let mut result = Self::ONE;

        while exp > 0 {
            if exp & 1 == 1 {
                result = result.mul_elem(base);
            }
            base = base.square();
            exp >>= 1;
        }

        result
    }

    /// Compute the multiplicative inverse using Fermat's little theorem:
    /// a^{-1} = a^{p-2} mod p.
    pub fn inv(self) -> Option<Self> {
        if self.val == 0 {
            return None;
        }
        // p - 2 = 0xFFFFFFFEFFFFFFFF
        Some(self.pow(GOLDILOCKS_PRIME - 2))
    }

    /// Compute the multiplicative inverse, panicking on zero.
    pub fn inv_or_panic(self) -> Self {
        self.inv().expect("attempted to invert zero")
    }

    /// Field division: a / b = a * b^{-1}.
    pub fn div_elem(self, rhs: Self) -> Option<Self> {
        rhs.inv().map(|inv_rhs| self.mul_elem(inv_rhs))
    }

    /// Test if this is zero.
    #[inline]
    pub fn is_zero(self) -> bool {
        self.val == 0
    }

    /// Test if this is one.
    #[inline]
    pub fn is_one(self) -> bool {
        self.val == 1
    }

    /// Convert from bytes (little-endian).
    pub fn from_bytes_le(bytes: &[u8]) -> Self {
        let mut arr = [0u8; 8];
        let len = bytes.len().min(8);
        arr[..len].copy_from_slice(&bytes[..len]);
        let v = u64::from_le_bytes(arr);
        Self::new(v)
    }

    /// Convert to bytes (little-endian).
    pub fn to_bytes_le(self) -> [u8; 8] {
        self.val.to_le_bytes()
    }

    /// Convert from bytes (big-endian).
    pub fn from_bytes_be(bytes: &[u8]) -> Self {
        let mut arr = [0u8; 8];
        let len = bytes.len().min(8);
        let start = 8 - len;
        arr[start..start + len].copy_from_slice(&bytes[..len]);
        let v = u64::from_be_bytes(arr);
        Self::new(v)
    }

    /// Convert to bytes (big-endian).
    pub fn to_bytes_be(self) -> [u8; 8] {
        self.val.to_be_bytes()
    }

    /// Create a random field element using the provided RNG.
    pub fn random<R: rand::Rng>(rng: &mut R) -> Self {
        loop {
            let v: u64 = rng.gen();
            if v < GOLDILOCKS_PRIME {
                return Self { val: v };
            }
        }
    }

    /// Compute the Legendre symbol (self / p).
    /// Returns 1 if self is a quadratic residue, -1 if not, 0 if self is 0.
    pub fn legendre(self) -> i32 {
        if self.is_zero() {
            return 0;
        }
        let exp = (GOLDILOCKS_PRIME - 1) / 2;
        let result = self.pow(exp);
        if result.is_one() {
            1
        } else {
            -1
        }
    }

    /// Compute the square root using Tonelli-Shanks algorithm.
    /// Returns None if self is not a quadratic residue.
    pub fn sqrt(self) -> Option<Self> {
        if self.is_zero() {
            return Some(Self::ZERO);
        }

        if self.legendre() != 1 {
            return None;
        }

        // For Goldilocks: p - 1 = 2^32 * q where q = 2^32 - 1
        // Tonelli-Shanks with s = 32
        let s: u32 = 32;
        let q: u64 = (GOLDILOCKS_PRIME - 1) >> s; // q = 2^32 - 1

        // Find a non-residue
        let mut z = Self::new(2);
        while z.legendre() != -1 {
            z = z.add_elem(Self::ONE);
        }

        let mut m = s;
        let mut c = z.pow(q);
        let mut t = self.pow(q);
        let mut r = self.pow((q + 1) / 2);

        loop {
            if t.is_zero() {
                return Some(Self::ZERO);
            }
            if t.is_one() {
                return Some(r);
            }

            // Find the least i such that t^{2^i} = 1
            let mut i = 0u32;
            let mut tmp = t;
            while !tmp.is_one() {
                tmp = tmp.square();
                i += 1;
                if i == m {
                    return None; // Should not happen if legendre check passed
                }
            }

            let b = c.pow(1u64 << (m - i - 1));
            m = i;
            c = b.square();
            t = t.mul_elem(c);
            r = r.mul_elem(b);
        }
    }

    /// Montgomery multiplication: compute a*b*R^{-1} mod p in Montgomery form.
    /// This is an alternative multiplication used when doing many multiplications
    /// on the same values.
    pub fn montgomery_mul(a: u64, b: u64) -> u64 {
        let product = (a as u128) * (b as u128);
        Self::montgomery_reduce(product)
    }

    /// Montgomery reduction: given a u128 in Montgomery form, reduce to u64.
    fn montgomery_reduce(x: u128) -> u64 {
        let x_lo = x as u64;
        let m = x_lo.wrapping_mul(MONTGOMERY_INV);
        let mp = (m as u128) * (GOLDILOCKS_PRIME as u128);
        let t = (x.wrapping_add(mp)) >> 64;
        let mut result = t as u64;
        if result >= GOLDILOCKS_PRIME {
            result -= GOLDILOCKS_PRIME;
        }
        result
    }

    /// Convert to Montgomery form: self * R mod p.
    pub fn to_montgomery(self) -> u64 {
        Self::montgomery_mul(self.val, MONTGOMERY_R2)
    }

    /// Convert from Montgomery form: self * R^{-1} mod p.
    pub fn from_montgomery(mont: u64) -> Self {
        let result = Self::montgomery_reduce(mont as u128);
        Self { val: result }
    }

    /// Batch inversion using Montgomery's trick.
    /// Given [a_0, ..., a_{n-1}], computes [a_0^{-1}, ..., a_{n-1}^{-1}]
    /// using only a single field inversion (and 3(n-1) multiplications).
    pub fn batch_inversion(elements: &[Self]) -> Vec<Self> {
        let n = elements.len();
        if n == 0 {
            return vec![];
        }

        // Compute prefix products
        let mut prefix = Vec::with_capacity(n);
        prefix.push(elements[0]);
        for i in 1..n {
            if elements[i].is_zero() {
                prefix.push(prefix[i - 1]);
            } else {
                prefix.push(prefix[i - 1].mul_elem(elements[i]));
            }
        }

        // Invert the total product
        let mut inv_total = prefix[n - 1].inv_or_panic();

        // Walk backwards, peeling off inverses
        let mut inverses = vec![Self::ZERO; n];
        for i in (1..n).rev() {
            if elements[i].is_zero() {
                inverses[i] = Self::ZERO;
            } else {
                inverses[i] = inv_total.mul_elem(prefix[i - 1]);
                inv_total = inv_total.mul_elem(elements[i]);
            }
        }
        if elements[0].is_zero() {
            inverses[0] = Self::ZERO;
        } else {
            inverses[0] = inv_total;
        }

        inverses
    }

    /// Compute a primitive n-th root of unity, where n must be a power of 2 and n ≤ 2^32.
    pub fn root_of_unity(n: usize) -> Self {
        assert!(n.is_power_of_two(), "n must be a power of 2");
        let log_n = n.trailing_zeros();
        assert!(log_n <= MAX_TWO_ADICITY, "n too large for Goldilocks field");

        // The 2^32-th root of unity is TWO_ADIC_ROOT_OF_UNITY.
        // The n-th root of unity is TWO_ADIC_ROOT_OF_UNITY^{2^{32 - log_n}}.
        let exp = 1u64 << (MAX_TWO_ADICITY - log_n);
        Self::from_canonical(TWO_ADIC_ROOT_OF_UNITY).pow(exp)
    }

    /// Compute all n-th roots of unity (powers of the primitive root).
    pub fn roots_of_unity(n: usize) -> Vec<Self> {
        let omega = Self::root_of_unity(n);
        let mut roots = Vec::with_capacity(n);
        let mut current = Self::ONE;
        for _ in 0..n {
            roots.push(current);
            current = current.mul_elem(omega);
        }
        roots
    }

    /// Sum a slice of field elements.
    pub fn sum_slice(elements: &[Self]) -> Self {
        let mut acc = Self::ZERO;
        for &e in elements {
            acc = acc.add_elem(e);
        }
        acc
    }

    /// Inner product of two slices.
    pub fn inner_product(a: &[Self], b: &[Self]) -> Self {
        assert_eq!(a.len(), b.len(), "inner product requires equal lengths");
        let mut acc = Self::ZERO;
        for i in 0..a.len() {
            acc = acc.add_elem(a[i].mul_elem(b[i]));
        }
        acc
    }

    /// Evaluate polynomial at a point using Horner's method.
    /// coeffs[i] is the coefficient of x^i.
    pub fn eval_poly(coeffs: &[Self], x: Self) -> Self {
        if coeffs.is_empty() {
            return Self::ZERO;
        }
        let mut result = coeffs[coeffs.len() - 1];
        for i in (0..coeffs.len() - 1).rev() {
            result = result.mul_elem(x).add_elem(coeffs[i]);
        }
        result
    }

    /// Interpolate a polynomial through the given points using Lagrange interpolation.
    /// Returns coefficients [c_0, c_1, ..., c_{n-1}] where p(x) = sum c_i * x^i.
    pub fn lagrange_interpolation(xs: &[Self], ys: &[Self]) -> Vec<Self> {
        assert_eq!(xs.len(), ys.len(), "xs and ys must have equal length");
        let n = xs.len();
        if n == 0 {
            return vec![];
        }

        // Compute the polynomial via the Lagrange basis form and accumulate coefficients
        let mut result = vec![Self::ZERO; n];

        for i in 0..n {
            // Compute the i-th Lagrange basis polynomial coefficients
            let mut basis = vec![Self::ZERO; n];
            basis[0] = Self::ONE;
            let mut degree = 0usize;

            // Denominator: product of (x_i - x_j) for j != i
            let mut denom = Self::ONE;

            for j in 0..n {
                if j == i {
                    continue;
                }

                denom = denom.mul_elem(xs[i].sub_elem(xs[j]));

                // Multiply current basis by (x - x_j)
                degree += 1;
                for k in (1..=degree).rev() {
                    basis[k] = basis[k - 1].sub_elem(basis[k].mul_elem(xs[j]));
                }
                basis[0] = Self::ZERO.sub_elem(basis[0].mul_elem(xs[j]));
            }

            // Scale by y_i / denom
            let scale = ys[i].mul_elem(denom.inv_or_panic());
            for k in 0..n {
                result[k] = result[k].add_elem(basis[k].mul_elem(scale));
            }
        }

        result
    }

    /// Multiply two polynomials (convolution).
    pub fn poly_mul(a: &[Self], b: &[Self]) -> Vec<Self> {
        if a.is_empty() || b.is_empty() {
            return vec![];
        }
        let result_len = a.len() + b.len() - 1;
        let mut result = vec![Self::ZERO; result_len];
        for i in 0..a.len() {
            for j in 0..b.len() {
                result[i + j] = result[i + j].add_elem(a[i].mul_elem(b[j]));
            }
        }
        result
    }

    /// Add two polynomials.
    pub fn poly_add(a: &[Self], b: &[Self]) -> Vec<Self> {
        let max_len = a.len().max(b.len());
        let mut result = vec![Self::ZERO; max_len];
        for i in 0..a.len() {
            result[i] = result[i].add_elem(a[i]);
        }
        for i in 0..b.len() {
            result[i] = result[i].add_elem(b[i]);
        }
        result
    }

    /// Subtract two polynomials: a - b.
    pub fn poly_sub(a: &[Self], b: &[Self]) -> Vec<Self> {
        let max_len = a.len().max(b.len());
        let mut result = vec![Self::ZERO; max_len];
        for i in 0..a.len() {
            result[i] = result[i].add_elem(a[i]);
        }
        for i in 0..b.len() {
            result[i] = result[i].sub_elem(b[i]);
        }
        result
    }

    /// Polynomial division with remainder: returns (quotient, remainder).
    pub fn poly_div(a: &[Self], b: &[Self]) -> (Vec<Self>, Vec<Self>) {
        assert!(!b.is_empty(), "division by zero polynomial");
        // Find actual degree of b
        let mut b_deg = b.len() - 1;
        while b_deg > 0 && b[b_deg].is_zero() {
            b_deg -= 1;
        }
        assert!(!b[b_deg].is_zero(), "division by zero polynomial");

        let mut remainder: Vec<Self> = a.to_vec();
        let a_deg = if a.is_empty() { 0 } else { a.len() - 1 };

        if a_deg < b_deg {
            return (vec![Self::ZERO], remainder);
        }

        let quot_len = a_deg - b_deg + 1;
        let mut quotient = vec![Self::ZERO; quot_len];

        let b_lead_inv = b[b_deg].inv_or_panic();

        for i in (0..quot_len).rev() {
            let idx = i + b_deg;
            if idx >= remainder.len() {
                continue;
            }
            let coeff = remainder[idx].mul_elem(b_lead_inv);
            quotient[i] = coeff;
            for j in 0..=b_deg {
                remainder[i + j] = remainder[i + j].sub_elem(coeff.mul_elem(b[j]));
            }
        }

        // Trim trailing zeros from remainder
        while remainder.len() > 1 && remainder.last().map_or(false, |x| x.is_zero()) {
            remainder.pop();
        }

        (quotient, remainder)
    }
}

// ─────────────────────────────────────────────────────────────
// Operator traits
// ─────────────────────────────────────────────────────────────

impl Add for GoldilocksField {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self { self.add_elem(rhs) }
}

impl Sub for GoldilocksField {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self { self.sub_elem(rhs) }
}

impl Mul for GoldilocksField {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self { self.mul_elem(rhs) }
}

impl Div for GoldilocksField {
    type Output = Self;
    #[inline]
    fn div(self, rhs: Self) -> Self {
        self.div_elem(rhs).expect("division by zero")
    }
}

impl Neg for GoldilocksField {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self { self.neg_elem() }
}

impl AddAssign for GoldilocksField {
    #[inline]
    fn add_assign(&mut self, rhs: Self) { *self = self.add_elem(rhs); }
}

impl SubAssign for GoldilocksField {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) { *self = self.sub_elem(rhs); }
}

impl MulAssign for GoldilocksField {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) { *self = self.mul_elem(rhs); }
}

impl PartialEq for GoldilocksField {
    fn eq(&self, other: &Self) -> bool { self.val == other.val }
}

impl Eq for GoldilocksField {}

impl Hash for GoldilocksField {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.val.hash(state);
    }
}

impl fmt::Debug for GoldilocksField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GF({})", self.val)
    }
}

impl fmt::Display for GoldilocksField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.val)
    }
}

impl Default for GoldilocksField {
    fn default() -> Self { Self::ZERO }
}

impl From<u64> for GoldilocksField {
    fn from(v: u64) -> Self { Self::new(v) }
}

impl From<u32> for GoldilocksField {
    fn from(v: u32) -> Self { Self::from_canonical(v as u64) }
}

impl From<usize> for GoldilocksField {
    fn from(v: usize) -> Self { Self::new(v as u64) }
}

impl From<GoldilocksField> for u64 {
    fn from(f: GoldilocksField) -> u64 { f.val }
}

// ─────────────────────────────────────────────────────────────
// NTT (Number Theoretic Transform)
// ─────────────────────────────────────────────────────────────

/// Compute the NTT (forward) of `coeffs` in-place.
/// `coeffs.len()` must be a power of 2.
pub fn ntt(coeffs: &mut [GoldilocksField]) {
    let n = coeffs.len();
    assert!(n.is_power_of_two(), "NTT length must be power of 2");

    if n <= 1 {
        return;
    }

    // Bit-reversal permutation
    bit_reverse_permutation(coeffs);

    // Cooley-Tukey butterfly
    let mut half_size = 1;
    while half_size < n {
        let size = half_size * 2;
        let omega = GoldilocksField::root_of_unity(size);

        let mut k = 0;
        while k < n {
            let mut w = GoldilocksField::ONE;
            for j in 0..half_size {
                let u = coeffs[k + j];
                let v = coeffs[k + j + half_size].mul_elem(w);
                coeffs[k + j] = u.add_elem(v);
                coeffs[k + j + half_size] = u.sub_elem(v);
                w = w.mul_elem(omega);
            }
            k += size;
        }
        half_size = size;
    }
}

/// Compute the inverse NTT (INTT) of `values` in-place.
/// `values.len()` must be a power of 2.
pub fn intt(values: &mut [GoldilocksField]) {
    let n = values.len();
    assert!(n.is_power_of_two(), "INTT length must be power of 2");

    if n <= 1 {
        return;
    }

    // Bit-reversal permutation
    bit_reverse_permutation(values);

    // Gentleman-Sande butterfly (inverse direction)
    let mut half_size = 1;
    while half_size < n {
        let size = half_size * 2;
        // Use the inverse root of unity (conjugate in the cyclic group)
        let omega_inv = GoldilocksField::root_of_unity(size)
            .pow(GOLDILOCKS_PRIME - 2); // omega^{-1} = omega^{p-2}

        let mut k = 0;
        while k < n {
            let mut w = GoldilocksField::ONE;
            for j in 0..half_size {
                let u = values[k + j];
                let v = values[k + j + half_size].mul_elem(w);
                values[k + j] = u.add_elem(v);
                values[k + j + half_size] = u.sub_elem(v);
                w = w.mul_elem(omega_inv);
            }
            k += size;
        }
        half_size = size;
    }

    // Normalize by 1/n
    let n_inv = GoldilocksField::new(n as u64).inv_or_panic();
    for val in values.iter_mut() {
        *val = val.mul_elem(n_inv);
    }
}

/// Multiply two polynomials using NTT. Input and output in coefficient form.
pub fn ntt_poly_mul(a: &[GoldilocksField], b: &[GoldilocksField]) -> Vec<GoldilocksField> {
    let result_len = a.len() + b.len() - 1;
    let n = result_len.next_power_of_two();

    let mut a_padded = vec![GoldilocksField::ZERO; n];
    let mut b_padded = vec![GoldilocksField::ZERO; n];
    a_padded[..a.len()].copy_from_slice(a);
    b_padded[..b.len()].copy_from_slice(b);

    ntt(&mut a_padded);
    ntt(&mut b_padded);

    for i in 0..n {
        a_padded[i] = a_padded[i].mul_elem(b_padded[i]);
    }

    intt(&mut a_padded);

    a_padded.truncate(result_len);
    a_padded
}

/// Bit-reversal permutation in-place.
fn bit_reverse_permutation(data: &mut [GoldilocksField]) {
    let n = data.len();
    let log_n = n.trailing_zeros();

    for i in 0..n {
        let j = bit_reverse(i as u32, log_n) as usize;
        if i < j {
            data.swap(i, j);
        }
    }
}

/// Reverse the lower `bits` bits of `x`.
fn bit_reverse(x: u32, bits: u32) -> u32 {
    let mut result = 0u32;
    let mut val = x;
    for _ in 0..bits {
        result = (result << 1) | (val & 1);
        val >>= 1;
    }
    result
}

/// Evaluate a polynomial on a coset: the set {c * omega^i : i = 0, ..., n-1}
/// where omega is the n-th root of unity.
pub fn evaluate_on_coset(
    coeffs: &[GoldilocksField],
    coset_shift: GoldilocksField,
    eval_size: usize,
) -> Vec<GoldilocksField> {
    assert!(eval_size.is_power_of_two(), "eval_size must be power of 2");
    let n = eval_size;

    // Shift coefficients: c_i -> c_i * coset_shift^i
    let mut shifted = vec![GoldilocksField::ZERO; n];
    let mut shift_power = GoldilocksField::ONE;
    for i in 0..coeffs.len().min(n) {
        shifted[i] = coeffs[i].mul_elem(shift_power);
        shift_power = shift_power.mul_elem(coset_shift);
    }

    ntt(&mut shifted);
    shifted
}

/// Interpolate from evaluations on a coset back to coefficient form.
pub fn interpolate_from_coset(
    evals: &[GoldilocksField],
    coset_shift: GoldilocksField,
) -> Vec<GoldilocksField> {
    let n = evals.len();
    assert!(n.is_power_of_two(), "length must be power of 2");

    let mut coeffs = evals.to_vec();
    intt(&mut coeffs);

    // Unshift: c_i -> c_i / coset_shift^i
    let shift_inv = coset_shift.inv_or_panic();
    let mut shift_power = GoldilocksField::ONE;
    for c in coeffs.iter_mut() {
        *c = c.mul_elem(shift_power);
        shift_power = shift_power.mul_elem(shift_inv);
    }

    coeffs
}

// ─────────────────────────────────────────────────────────────
// GoldilocksExt: Quadratic extension F_{p^2}
// ─────────────────────────────────────────────────────────────

/// Quadratic extension of the Goldilocks field.
/// Elements are a + b*W where W^2 = 7 (a non-residue in F_p).
/// This gives a field of order p^2.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GoldilocksExt {
    /// Real part.
    pub a: GoldilocksField,
    /// Imaginary part (coefficient of W).
    pub b: GoldilocksField,
}

/// The non-residue used for extension: W^2 = NON_RESIDUE
const NON_RESIDUE: u64 = 7;

impl GoldilocksExt {
    pub const ZERO: Self = Self { a: GoldilocksField::ZERO, b: GoldilocksField::ZERO };
    pub const ONE: Self = Self { a: GoldilocksField::ONE, b: GoldilocksField::ZERO };

    /// Create a new extension element.
    pub fn new(a: GoldilocksField, b: GoldilocksField) -> Self {
        Self { a, b }
    }

    /// Create from the base field (b = 0).
    pub fn from_base(a: GoldilocksField) -> Self {
        Self { a, b: GoldilocksField::ZERO }
    }

    /// Addition in the extension.
    pub fn add_ext(self, rhs: Self) -> Self {
        Self {
            a: self.a + rhs.a,
            b: self.b + rhs.b,
        }
    }

    /// Subtraction in the extension.
    pub fn sub_ext(self, rhs: Self) -> Self {
        Self {
            a: self.a - rhs.a,
            b: self.b - rhs.b,
        }
    }

    /// Multiplication in the extension.
    /// (a0 + b0*W)(a1 + b1*W) = (a0*a1 + b0*b1*NON_RESIDUE) + (a0*b1 + a1*b0)*W
    pub fn mul_ext(self, rhs: Self) -> Self {
        let nr = GoldilocksField::new(NON_RESIDUE);
        Self {
            a: self.a * rhs.a + self.b * rhs.b * nr,
            b: self.a * rhs.b + self.b * rhs.a,
        }
    }

    /// Square in the extension (slightly more efficient than mul).
    pub fn square_ext(self) -> Self {
        let nr = GoldilocksField::new(NON_RESIDUE);
        // (a + bW)^2 = a^2 + b^2*NR + 2ab*W
        Self {
            a: self.a.square() + self.b.square() * nr,
            b: self.a * self.b + self.a * self.b,
        }
    }

    /// Negation in the extension.
    pub fn neg_ext(self) -> Self {
        Self {
            a: -self.a,
            b: -self.b,
        }
    }

    /// Conjugate: a + bW -> a - bW.
    pub fn conjugate(self) -> Self {
        Self {
            a: self.a,
            b: -self.b,
        }
    }

    /// Norm: (a + bW)(a - bW) = a^2 - b^2 * NON_RESIDUE.
    pub fn norm(self) -> GoldilocksField {
        let nr = GoldilocksField::new(NON_RESIDUE);
        self.a.square() - self.b.square() * nr
    }

    /// Inverse in the extension field.
    /// (a + bW)^{-1} = conjugate / norm.
    pub fn inv_ext(self) -> Option<Self> {
        let n = self.norm();
        n.inv().map(|n_inv| {
            let conj = self.conjugate();
            Self {
                a: conj.a * n_inv,
                b: conj.b * n_inv,
            }
        })
    }

    /// Exponentiation by squaring.
    pub fn pow_ext(self, mut exp: u64) -> Self {
        let mut base = self;
        let mut result = Self::ONE;
        while exp > 0 {
            if exp & 1 == 1 {
                result = result.mul_ext(base);
            }
            base = base.square_ext();
            exp >>= 1;
        }
        result
    }

    /// Test if zero.
    pub fn is_zero_ext(self) -> bool {
        self.a.is_zero() && self.b.is_zero()
    }

    /// Test if one.
    pub fn is_one_ext(self) -> bool {
        self.a.is_one() && self.b.is_zero()
    }

    /// Create a random extension element.
    pub fn random<R: rand::Rng>(rng: &mut R) -> Self {
        Self {
            a: GoldilocksField::random(rng),
            b: GoldilocksField::random(rng),
        }
    }

    /// Frobenius endomorphism: (a + bW)^p = a + b*W^p.
    /// Since W^2 = NR, W^p = W^{(p-1)/2} * W = legendre(NR) * W.
    /// For NR = 7 which is a non-residue, W^p = -W, so frobenius(a + bW) = a - bW = conjugate.
    pub fn frobenius(self) -> Self {
        self.conjugate()
    }

    /// Serialize to bytes.
    pub fn to_bytes(self) -> [u8; 16] {
        let mut out = [0u8; 16];
        out[..8].copy_from_slice(&self.a.to_bytes_le());
        out[8..].copy_from_slice(&self.b.to_bytes_le());
        out
    }

    /// Deserialize from bytes.
    pub fn from_bytes(bytes: &[u8; 16]) -> Self {
        Self {
            a: GoldilocksField::from_bytes_le(&bytes[..8]),
            b: GoldilocksField::from_bytes_le(&bytes[8..]),
        }
    }
}

impl Add for GoldilocksExt {
    type Output = Self;
    fn add(self, rhs: Self) -> Self { self.add_ext(rhs) }
}

impl Sub for GoldilocksExt {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self { self.sub_ext(rhs) }
}

impl Mul for GoldilocksExt {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self { self.mul_ext(rhs) }
}

impl Neg for GoldilocksExt {
    type Output = Self;
    fn neg(self) -> Self { self.neg_ext() }
}

impl fmt::Debug for GoldilocksExt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GFExt({} + {}*W)", self.a, self.b)
    }
}

impl fmt::Display for GoldilocksExt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.b.is_zero() {
            write!(f, "{}", self.a)
        } else if self.a.is_zero() {
            write!(f, "{}*W", self.b)
        } else {
            write!(f, "{} + {}*W", self.a, self.b)
        }
    }
}

impl Default for GoldilocksExt {
    fn default() -> Self { Self::ZERO }
}

// ─────────────────────────────────────────────────────────────
// Polynomial operations over extension field
// ─────────────────────────────────────────────────────────────

/// Evaluate an extension-field polynomial at a base-field point.
pub fn eval_ext_poly_at_base(
    coeffs: &[GoldilocksExt],
    x: GoldilocksField,
) -> GoldilocksExt {
    if coeffs.is_empty() {
        return GoldilocksExt::ZERO;
    }
    let x_ext = GoldilocksExt::from_base(x);
    let mut result = coeffs[coeffs.len() - 1];
    for i in (0..coeffs.len() - 1).rev() {
        result = result.mul_ext(x_ext).add_ext(coeffs[i]);
    }
    result
}

/// Evaluate a base-field polynomial at an extension-field point.
pub fn eval_base_poly_at_ext(
    coeffs: &[GoldilocksField],
    x: GoldilocksExt,
) -> GoldilocksExt {
    if coeffs.is_empty() {
        return GoldilocksExt::ZERO;
    }
    let mut result = GoldilocksExt::from_base(coeffs[coeffs.len() - 1]);
    for i in (0..coeffs.len() - 1).rev() {
        result = result.mul_ext(x).add_ext(GoldilocksExt::from_base(coeffs[i]));
    }
    result
}

// ─────────────────────────────────────────────────────────────
// Vanishing polynomial
// ─────────────────────────────────────────────────────────────

/// Compute the vanishing polynomial Z_H(x) = x^n - 1 for a multiplicative subgroup H of order n.
/// Returns the coefficients: [-1, 0, ..., 0, 1] (length n+1).
pub fn vanishing_poly(n: usize) -> Vec<GoldilocksField> {
    let mut coeffs = vec![GoldilocksField::ZERO; n + 1];
    coeffs[0] = -GoldilocksField::ONE;
    coeffs[n] = GoldilocksField::ONE;
    coeffs
}

/// Evaluate the vanishing polynomial Z_H(x) = x^n - 1 at a point.
pub fn eval_vanishing_poly(x: GoldilocksField, n: usize) -> GoldilocksField {
    x.pow(n as u64) - GoldilocksField::ONE
}

// ─────────────────────────────────────────────────────────────
// Multi-precision polynomial operations
// ─────────────────────────────────────────────────────────────

/// Return true if polynomial is the zero polynomial.
pub fn poly_is_zero(p: &[GoldilocksField]) -> bool {
    p.is_empty() || p.iter().all(|c| c.is_zero())
}

/// Return the degree of a polynomial (index of highest non-zero coefficient).
/// Returns 0 for the zero polynomial.
pub fn poly_degree(p: &[GoldilocksField]) -> usize {
    if poly_is_zero(p) {
        return 0;
    }
    let mut d = p.len() - 1;
    while d > 0 && p[d].is_zero() {
        d -= 1;
    }
    d
}

/// Remove trailing zero coefficients from a polynomial.
pub fn poly_trim(p: &mut Vec<GoldilocksField>) {
    while p.len() > 1 && p.last().map_or(false, |c| c.is_zero()) {
        p.pop();
    }
}

/// Polynomial remainder: a mod m.
pub fn poly_mod(a: &[GoldilocksField], m: &[GoldilocksField]) -> Vec<GoldilocksField> {
    let (_q, r) = GoldilocksField::poly_div(a, m);
    r
}

/// Euclidean GCD of two polynomials.
pub fn poly_gcd(a: &[GoldilocksField], b: &[GoldilocksField]) -> Vec<GoldilocksField> {
    let mut r0: Vec<GoldilocksField> = a.to_vec();
    let mut r1: Vec<GoldilocksField> = b.to_vec();
    poly_trim(&mut r0);
    poly_trim(&mut r1);

    while !poly_is_zero(&r1) {
        let rem = poly_mod(&r0, &r1);
        r0 = r1;
        r1 = rem;
        poly_trim(&mut r1);
    }

    // Make monic
    poly_trim(&mut r0);
    if !r0.is_empty() {
        let lead = r0[poly_degree(&r0)];
        if !lead.is_zero() {
            let lead_inv = lead.inv_or_panic();
            for c in r0.iter_mut() {
                *c = c.mul_elem(lead_inv);
            }
        }
    }
    r0
}

/// Formal derivative of a polynomial: p'(x).
pub fn poly_derivative(p: &[GoldilocksField]) -> Vec<GoldilocksField> {
    if p.len() <= 1 {
        return vec![GoldilocksField::ZERO];
    }
    let mut result = Vec::with_capacity(p.len() - 1);
    for i in 1..p.len() {
        result.push(p[i].mul_elem(GoldilocksField::new(i as u64)));
    }
    if result.is_empty() {
        result.push(GoldilocksField::ZERO);
    }
    result
}

/// Scale a polynomial by a constant: c * p(x).
pub fn poly_scale(p: &[GoldilocksField], c: GoldilocksField) -> Vec<GoldilocksField> {
    p.iter().map(|&coeff| coeff.mul_elem(c)).collect()
}

/// Shift a polynomial: compute p(x + c).
pub fn poly_shift(p: &[GoldilocksField], c: GoldilocksField) -> Vec<GoldilocksField> {
    let n = p.len();
    if n == 0 {
        return vec![];
    }
    // Use the fact that p(x+c) can be computed by evaluating at enough points and re-interpolating,
    // or directly via Taylor expansion. We'll use direct computation.
    // p(x + c) = sum_{i} p_i * (x+c)^i
    // Expand each (x+c)^i using binomial theorem and accumulate.
    let mut result = vec![GoldilocksField::ZERO; n];

    // Precompute powers of c
    let mut c_powers = vec![GoldilocksField::ONE; n];
    for i in 1..n {
        c_powers[i] = c_powers[i - 1].mul_elem(c);
    }

    // Precompute binomial coefficients mod p using Pascal's triangle
    let mut binom = vec![vec![GoldilocksField::ZERO; n]; n];
    for i in 0..n {
        binom[i][0] = GoldilocksField::ONE;
        for j in 1..=i {
            binom[i][j] = binom[i - 1][j - 1].add_elem(
                if j < i { binom[i - 1][j] } else { GoldilocksField::ZERO }
            );
        }
    }

    // p(x+c) = sum_i p_i * sum_j C(i,j) * c^(i-j) * x^j
    for i in 0..n {
        for j in 0..=i {
            let contrib = p[i].mul_elem(binom[i][j]).mul_elem(c_powers[i - j]);
            result[j] = result[j].add_elem(contrib);
        }
    }

    result
}

/// Compose two polynomials: compute f(g(x)).
pub fn poly_compose(f: &[GoldilocksField], g: &[GoldilocksField]) -> Vec<GoldilocksField> {
    if f.is_empty() {
        return vec![];
    }
    // Horner's method: f(g(x)) = f_n * g(x)^n + ... = (...((f_n * g(x) + f_{n-1}) * g(x)) + ...)
    let mut result = vec![f[f.len() - 1]];
    for i in (0..f.len() - 1).rev() {
        result = GoldilocksField::poly_mul(&result, g);
        // Add f[i] to constant term
        if result.is_empty() {
            result.push(f[i]);
        } else {
            result[0] = result[0].add_elem(f[i]);
        }
    }
    result
}

/// Build polynomial from roots: (x - r_0)(x - r_1)...(x - r_{n-1}).
pub fn poly_from_roots(roots: &[GoldilocksField]) -> Vec<GoldilocksField> {
    if roots.is_empty() {
        return vec![GoldilocksField::ONE];
    }
    let mut result = vec![roots[0].neg_elem(), GoldilocksField::ONE]; // (x - r_0)
    for i in 1..roots.len() {
        let factor = vec![roots[i].neg_elem(), GoldilocksField::ONE];
        result = GoldilocksField::poly_mul(&result, &factor);
    }
    result
}

// ─────────────────────────────────────────────────────────────
// Matrix operations over GoldilocksField
// ─────────────────────────────────────────────────────────────

/// A matrix over the Goldilocks field stored in row-major order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FieldMatrix {
    pub data: Vec<Vec<GoldilocksField>>,
    pub rows: usize,
    pub cols: usize,
}

impl FieldMatrix {
    /// Create a new zero matrix of given dimensions.
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![vec![GoldilocksField::ZERO; cols]; rows],
            rows,
            cols,
        }
    }

    /// Create a zero matrix (alias).
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self::new(rows, cols)
    }

    /// Create an identity matrix.
    pub fn identity(n: usize) -> Self {
        let mut m = Self::new(n, n);
        for i in 0..n {
            m.data[i][i] = GoldilocksField::ONE;
        }
        m
    }

    /// Create from row vectors.
    pub fn from_rows(rows: Vec<Vec<GoldilocksField>>) -> Self {
        let r = rows.len();
        let c = if r > 0 { rows[0].len() } else { 0 };
        Self { data: rows, rows: r, cols: c }
    }

    /// Get element at (i, j).
    pub fn get(&self, i: usize, j: usize) -> GoldilocksField {
        self.data[i][j]
    }

    /// Set element at (i, j).
    pub fn set(&mut self, i: usize, j: usize, val: GoldilocksField) {
        self.data[i][j] = val;
    }

    /// Matrix multiplication.
    pub fn mul_matrix(&self, other: &FieldMatrix) -> FieldMatrix {
        assert_eq!(self.cols, other.rows, "matrix dimension mismatch for multiplication");
        let mut result = FieldMatrix::new(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = GoldilocksField::ZERO;
                for k in 0..self.cols {
                    sum = sum.add_elem(self.data[i][k].mul_elem(other.data[k][j]));
                }
                result.data[i][j] = sum;
            }
        }
        result
    }

    /// Matrix-vector multiplication.
    pub fn mul_vector(&self, v: &[GoldilocksField]) -> Vec<GoldilocksField> {
        assert_eq!(self.cols, v.len(), "matrix-vector dimension mismatch");
        let mut result = vec![GoldilocksField::ZERO; self.rows];
        for i in 0..self.rows {
            let mut sum = GoldilocksField::ZERO;
            for j in 0..self.cols {
                sum = sum.add_elem(self.data[i][j].mul_elem(v[j]));
            }
            result[i] = sum;
        }
        result
    }

    /// Transpose.
    pub fn transpose(&self) -> FieldMatrix {
        let mut result = FieldMatrix::new(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[j][i] = self.data[i][j];
            }
        }
        result
    }

    /// Matrix trace (sum of diagonal elements).
    pub fn trace(&self) -> GoldilocksField {
        assert_eq!(self.rows, self.cols, "trace requires square matrix");
        let mut sum = GoldilocksField::ZERO;
        for i in 0..self.rows {
            sum = sum.add_elem(self.data[i][i]);
        }
        sum
    }

    /// Determinant via row reduction (Gaussian elimination).
    pub fn determinant(&self) -> GoldilocksField {
        assert_eq!(self.rows, self.cols, "determinant requires square matrix");
        let n = self.rows;
        let mut m = self.data.clone();
        let mut det = GoldilocksField::ONE;

        for col in 0..n {
            // Find pivot
            let mut pivot_row = None;
            for row in col..n {
                if !m[row][col].is_zero() {
                    pivot_row = Some(row);
                    break;
                }
            }
            let pivot_row = match pivot_row {
                Some(r) => r,
                None => return GoldilocksField::ZERO,
            };

            if pivot_row != col {
                m.swap(col, pivot_row);
                det = det.neg_elem();
            }

            det = det.mul_elem(m[col][col]);
            let pivot_inv = m[col][col].inv_or_panic();

            for row in (col + 1)..n {
                let factor = m[row][col].mul_elem(pivot_inv);
                for j in col..n {
                    let sub = factor.mul_elem(m[col][j]);
                    m[row][j] = m[row][j].sub_elem(sub);
                }
            }
        }

        det
    }

    /// Inverse via Gauss-Jordan elimination. Returns None if singular.
    pub fn inverse(&self) -> Option<FieldMatrix> {
        assert_eq!(self.rows, self.cols, "inverse requires square matrix");
        let n = self.rows;
        let mut aug = vec![vec![GoldilocksField::ZERO; 2 * n]; n];

        for i in 0..n {
            for j in 0..n {
                aug[i][j] = self.data[i][j];
            }
            aug[i][n + i] = GoldilocksField::ONE;
        }

        for col in 0..n {
            // Find pivot
            let mut pivot_row = None;
            for row in col..n {
                if !aug[row][col].is_zero() {
                    pivot_row = Some(row);
                    break;
                }
            }
            let pivot_row = pivot_row?;

            if pivot_row != col {
                aug.swap(col, pivot_row);
            }

            let pivot_inv = aug[col][col].inv_or_panic();
            for j in 0..2 * n {
                aug[col][j] = aug[col][j].mul_elem(pivot_inv);
            }

            for row in 0..n {
                if row == col {
                    continue;
                }
                let factor = aug[row][col];
                for j in 0..2 * n {
                    let sub = factor.mul_elem(aug[col][j]);
                    aug[row][j] = aug[row][j].sub_elem(sub);
                }
            }
        }

        let mut result = FieldMatrix::new(n, n);
        for i in 0..n {
            for j in 0..n {
                result.data[i][j] = aug[i][n + j];
            }
        }
        Some(result)
    }

    /// Row echelon form (reduced) and rank. Returns (RREF matrix, rank).
    pub fn row_echelon(&self) -> (FieldMatrix, usize) {
        let mut m = self.clone();
        let mut pivot_col = 0;
        let mut rank = 0;

        for row in 0..m.rows {
            if pivot_col >= m.cols {
                break;
            }

            // Find pivot in this column
            let mut found = None;
            for r in row..m.rows {
                if !m.data[r][pivot_col].is_zero() {
                    found = Some(r);
                    break;
                }
            }

            let pivot_row = match found {
                Some(r) => r,
                None => {
                    pivot_col += 1;
                    continue;
                }
            };

            if pivot_row != row {
                m.data.swap(row, pivot_row);
            }

            // Scale pivot row
            let pivot_inv = m.data[row][pivot_col].inv_or_panic();
            for j in 0..m.cols {
                m.data[row][j] = m.data[row][j].mul_elem(pivot_inv);
            }

            // Eliminate all other rows
            for r in 0..m.rows {
                if r == row {
                    continue;
                }
                let factor = m.data[r][pivot_col];
                if !factor.is_zero() {
                    for j in 0..m.cols {
                        let sub = factor.mul_elem(m.data[row][j]);
                        m.data[r][j] = m.data[r][j].sub_elem(sub);
                    }
                }
            }

            rank += 1;
            pivot_col += 1;
        }

        (m, rank)
    }

    /// Rank of the matrix.
    pub fn rank(&self) -> usize {
        let (_, r) = self.row_echelon();
        r
    }

    /// Null space (kernel) basis vectors.
    pub fn kernel(&self) -> Vec<Vec<GoldilocksField>> {
        let (rref, rank) = self.row_echelon();
        let n = self.cols;
        if rank == n {
            return vec![];
        }

        // Identify pivot columns
        let mut pivot_cols = Vec::new();
        let mut col = 0;
        for row in 0..rref.rows {
            while col < rref.cols && rref.data[row][col].is_zero() {
                col += 1;
            }
            if col < rref.cols {
                pivot_cols.push(col);
                col += 1;
            }
        }

        let free_cols: Vec<usize> = (0..n).filter(|c| !pivot_cols.contains(c)).collect();
        let mut basis = Vec::new();

        for &fc in &free_cols {
            let mut v = vec![GoldilocksField::ZERO; n];
            v[fc] = GoldilocksField::ONE;

            for (row_idx, &pc) in pivot_cols.iter().enumerate() {
                if row_idx < rref.rows {
                    v[pc] = rref.data[row_idx][fc].neg_elem();
                }
            }
            basis.push(v);
        }

        basis
    }

    /// Matrix exponentiation by squaring. Matrix must be square.
    pub fn pow_matrix(&self, mut exp: u64) -> FieldMatrix {
        assert_eq!(self.rows, self.cols, "pow requires square matrix");
        let n = self.rows;
        let mut base = self.clone();
        let mut result = FieldMatrix::identity(n);

        while exp > 0 {
            if exp & 1 == 1 {
                result = result.mul_matrix(&base);
            }
            base = base.mul_matrix(&base);
            exp >>= 1;
        }

        result
    }

    /// Characteristic polynomial of a square matrix using Faddeev-LeVerrier algorithm.
    /// Returns coefficients [c_0, c_1, ..., c_n] of det(xI - A) = c_0 + c_1*x + ... + c_n*x^n.
    pub fn characteristic_polynomial(&self) -> Vec<GoldilocksField> {
        assert_eq!(self.rows, self.cols, "characteristic polynomial requires square matrix");
        let n = self.rows;
        let mut coeffs = vec![GoldilocksField::ZERO; n + 1];
        coeffs[n] = GoldilocksField::ONE; // Leading coefficient is 1

        let mut m = FieldMatrix::identity(n);
        for k in 1..=n {
            m = self.mul_matrix(&m);
            // c_{n-k} = -1/k * trace(A * M_{k-1})
            let tr = m.trace();
            let k_field = GoldilocksField::new(k as u64);
            let c = tr.neg_elem().mul_elem(k_field.inv_or_panic());
            coeffs[n - k] = c;

            // M_k = A * M_{k-1} + c_{n-k} * I
            for i in 0..n {
                m.data[i][i] = m.data[i][i].add_elem(c);
            }
        }

        coeffs
    }
}

/// Solve the linear system Ax = b. Returns None if no unique solution exists.
pub fn solve_linear(a: &FieldMatrix, b: &[GoldilocksField]) -> Option<Vec<GoldilocksField>> {
    assert_eq!(a.rows, b.len(), "dimension mismatch in solve_linear");
    assert_eq!(a.rows, a.cols, "solve_linear requires square matrix");
    let n = a.rows;

    // Augmented matrix [A | b]
    let mut aug = vec![vec![GoldilocksField::ZERO; n + 1]; n];
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = a.data[i][j];
        }
        aug[i][n] = b[i];
    }

    // Forward elimination with partial pivoting
    for col in 0..n {
        let mut pivot_row = None;
        for row in col..n {
            if !aug[row][col].is_zero() {
                pivot_row = Some(row);
                break;
            }
        }
        let pivot_row = pivot_row?;

        if pivot_row != col {
            aug.swap(col, pivot_row);
        }

        let pivot_inv = aug[col][col].inv_or_panic();
        for j in col..=n {
            aug[col][j] = aug[col][j].mul_elem(pivot_inv);
        }

        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[row][col];
            for j in col..=n {
                let sub = factor.mul_elem(aug[col][j]);
                aug[row][j] = aug[row][j].sub_elem(sub);
            }
        }
    }

    let x: Vec<GoldilocksField> = (0..n).map(|i| aug[i][n]).collect();
    Some(x)
}

// ─────────────────────────────────────────────────────────────
// Discrete Fourier Transform utilities
// ─────────────────────────────────────────────────────────────

/// Precompute twiddle factors (powers of the n-th root of unity).
pub fn precompute_twiddle_factors(n: usize) -> Vec<GoldilocksField> {
    assert!(n.is_power_of_two(), "n must be a power of 2");
    let omega = GoldilocksField::root_of_unity(n);
    let mut factors = Vec::with_capacity(n);
    let mut w = GoldilocksField::ONE;
    for _ in 0..n {
        factors.push(w);
        w = w.mul_elem(omega);
    }
    factors
}

/// NTT using precomputed roots of unity.
pub fn ntt_with_precomputed_roots(data: &mut [GoldilocksField], roots: &[GoldilocksField]) {
    let n = data.len();
    assert!(n.is_power_of_two(), "NTT length must be power of 2");
    assert!(roots.len() >= n, "not enough precomputed roots");

    if n <= 1 {
        return;
    }

    bit_reverse_permutation(data);

    let mut half_size = 1;
    while half_size < n {
        let size = half_size * 2;
        let step = n / size;

        let mut k = 0;
        while k < n {
            for j in 0..half_size {
                let w = roots[j * step];
                let u = data[k + j];
                let v = data[k + j + half_size].mul_elem(w);
                data[k + j] = u.add_elem(v);
                data[k + j + half_size] = u.sub_elem(v);
            }
            k += size;
        }
        half_size = size;
    }
}

/// NTT on a coset (domain shifted by `shift`).
pub fn coset_ntt(data: &mut [GoldilocksField], shift: GoldilocksField) {
    let n = data.len();
    // Multiply coefficients by shift^i
    let mut s = GoldilocksField::ONE;
    for d in data.iter_mut() {
        *d = d.mul_elem(s);
        s = s.mul_elem(shift);
    }
    ntt(data);
}

/// Inverse NTT on a coset (undo coset_ntt).
pub fn coset_intt(data: &mut [GoldilocksField], shift: GoldilocksField) {
    let n = data.len();
    intt(data);
    let shift_inv = shift.inv_or_panic();
    let mut s = GoldilocksField::ONE;
    for d in data.iter_mut() {
        *d = d.mul_elem(s);
        s = s.mul_elem(shift_inv);
    }
}

/// Batch NTT on multiple polynomials (all must have the same power-of-two length).
pub fn multi_ntt(polys: &mut [Vec<GoldilocksField>]) {
    for poly in polys.iter_mut() {
        ntt(poly);
    }
}

/// Batch INTT on multiple polynomials.
pub fn multi_intt(polys: &mut [Vec<GoldilocksField>]) {
    for poly in polys.iter_mut() {
        intt(poly);
    }
}

// ─────────────────────────────────────────────────────────────
// Reed-Solomon encoding/decoding
// ─────────────────────────────────────────────────────────────

/// Reed-Solomon encode: evaluate polynomial (given by `data` coefficients) on a larger
/// domain of size `data.len() * expansion_factor`.
pub fn reed_solomon_encode(
    data: &[GoldilocksField],
    expansion_factor: usize,
) -> Vec<GoldilocksField> {
    assert!(expansion_factor >= 1, "expansion factor must be >= 1");
    let original_len = data.len();
    let expanded_len = (original_len * expansion_factor).next_power_of_two();

    let mut padded = vec![GoldilocksField::ZERO; expanded_len];
    padded[..original_len].copy_from_slice(data);

    ntt(&mut padded);
    padded
}

/// Reed-Solomon decode: given evaluations at specific points, interpolate back
/// to recover the original polynomial of `original_length` coefficients.
pub fn reed_solomon_decode(
    evaluations: &[GoldilocksField],
    evaluation_points: &[GoldilocksField],
    original_length: usize,
) -> Vec<GoldilocksField> {
    assert_eq!(evaluations.len(), evaluation_points.len());
    assert!(evaluations.len() >= original_length);

    // Use first `original_length` points to interpolate
    let xs = &evaluation_points[..original_length];
    let ys = &evaluations[..original_length];

    GoldilocksField::lagrange_interpolation(xs, ys)
}

/// Erasure decoding: recover a polynomial of given degree from partial evaluations.
/// `evaluations[i]` is `Some(v)` if known, `None` if erased.
/// `domain` contains all evaluation points. Need at least `degree + 1` known values.
pub fn erasure_decode(
    evaluations: &[Option<GoldilocksField>],
    domain: &[GoldilocksField],
    degree: usize,
) -> Option<Vec<GoldilocksField>> {
    assert_eq!(evaluations.len(), domain.len());

    let mut known_xs = Vec::new();
    let mut known_ys = Vec::new();

    for (i, eval) in evaluations.iter().enumerate() {
        if let Some(v) = eval {
            known_xs.push(domain[i]);
            known_ys.push(*v);
        }
    }

    if known_xs.len() < degree + 1 {
        return None;
    }

    // Use exactly degree + 1 known points
    let xs = &known_xs[..degree + 1];
    let ys = &known_ys[..degree + 1];

    Some(GoldilocksField::lagrange_interpolation(xs, ys))
}

// ─────────────────────────────────────────────────────────────
// Multilinear extension
// ─────────────────────────────────────────────────────────────

/// A multilinear polynomial defined by its evaluations over the boolean hypercube {0,1}^n.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MultilinearExtension {
    pub evaluations: Vec<GoldilocksField>,
    num_vars: usize,
}

impl MultilinearExtension {
    /// Create a new multilinear extension from evaluations on the boolean hypercube.
    /// Length must be a power of two.
    pub fn new(evaluations: Vec<GoldilocksField>) -> Self {
        let len = evaluations.len();
        assert!(len.is_power_of_two(), "evaluations length must be power of 2");
        let num_vars = len.trailing_zeros() as usize;
        Self { evaluations, num_vars }
    }

    /// Number of variables.
    pub fn num_variables(&self) -> usize {
        self.num_vars
    }

    /// Evaluate the multilinear extension at an arbitrary point using the eq polynomial.
    /// point must have length == num_variables.
    pub fn evaluate(&self, point: &[GoldilocksField]) -> GoldilocksField {
        assert_eq!(point.len(), self.num_vars, "point dimension mismatch");
        let eq_evals = Self::eq_polynomial(point);
        GoldilocksField::inner_product(&self.evaluations, &eq_evals)
    }

    /// Compute the eq polynomial evaluations: eq(point, x) for all x in {0,1}^n.
    /// eq(r, x) = prod_i (r_i * x_i + (1 - r_i)(1 - x_i))
    pub fn eq_polynomial(point: &[GoldilocksField]) -> Vec<GoldilocksField> {
        let n = point.len();
        let size = 1usize << n;
        let mut evals = vec![GoldilocksField::ONE; size];

        for i in 0..n {
            let r_i = point[i];
            let one_minus_ri = GoldilocksField::ONE.sub_elem(r_i);
            let half = 1usize << i;
            // Process in reverse to avoid overwriting values we still need
            for j in (0..half).rev() {
                evals[2 * j + 1] = evals[j].mul_elem(r_i);
                evals[2 * j] = evals[j].mul_elem(one_minus_ri);
            }
        }

        evals
    }

    /// Fix one variable to a specific value, returning a multilinear extension
    /// with one fewer variable.
    pub fn partial_evaluate(&self, var: usize, val: GoldilocksField) -> Self {
        assert!(var < self.num_vars, "variable index out of range");
        let new_num_vars = self.num_vars - 1;
        let new_size = 1usize << new_num_vars;
        let mut new_evals = vec![GoldilocksField::ZERO; new_size];

        let stride = 1usize << var;
        let block_size = stride * 2;
        let one_minus_val = GoldilocksField::ONE.sub_elem(val);

        for i in 0..new_size {
            // Determine which block and position within block
            let block = i / stride;
            let pos = i % stride;
            let idx0 = block * block_size + pos;
            let idx1 = idx0 + stride;

            new_evals[i] = self.evaluations[idx0].mul_elem(one_minus_val)
                .add_elem(self.evaluations[idx1].mul_elem(val));
        }

        Self {
            evaluations: new_evals,
            num_vars: new_num_vars,
        }
    }

    /// Sum over the boolean hypercube: sum_{x in {0,1}^n} f(x).
    pub fn sum_over_boolean_hypercube(&self) -> GoldilocksField {
        GoldilocksField::sum_slice(&self.evaluations)
    }
}

/// Tensor product of two vectors: a ⊗ b.
pub fn tensor_product(
    a: &[GoldilocksField],
    b: &[GoldilocksField],
) -> Vec<GoldilocksField> {
    let mut result = Vec::with_capacity(a.len() * b.len());
    for &ai in a {
        for &bj in b {
            result.push(ai.mul_elem(bj));
        }
    }
    result
}

// ─────────────────────────────────────────────────────────────
// Hash-to-field functions
// ─────────────────────────────────────────────────────────────

/// Hash arbitrary bytes to a Goldilocks field element using BLAKE3.
pub fn hash_to_field(data: &[u8]) -> GoldilocksField {
    let hash = blake3::hash(data);
    let bytes = hash.as_bytes();
    let val = u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3],
        bytes[4], bytes[5], bytes[6], bytes[7],
    ]);
    GoldilocksField::new(val)
}

/// Hash bytes to a vector of `count` field elements using BLAKE3 in XOF mode.
pub fn hash_to_field_vec(data: &[u8], count: usize) -> Vec<GoldilocksField> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(data);
    let mut reader = hasher.finalize_xof();
    let mut result = Vec::with_capacity(count);
    for _ in 0..count {
        let mut buf = [0u8; 8];
        reader.fill(&mut buf);
        let val = u64::from_le_bytes(buf);
        result.push(GoldilocksField::new(val));
    }
    result
}

/// Hash a pair of byte slices to a field element.
pub fn hash_pair_to_field(a: &[u8], b: &[u8]) -> GoldilocksField {
    let mut hasher = blake3::Hasher::new();
    hasher.update(a);
    hasher.update(b);
    let hash = hasher.finalize();
    let bytes = hash.as_bytes();
    let val = u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3],
        bytes[4], bytes[5], bytes[6], bytes[7],
    ]);
    GoldilocksField::new(val)
}

/// Hash a slice of u64 values to a field element.
pub fn hash_u64s_to_field(values: &[u64]) -> GoldilocksField {
    let mut hasher = blake3::Hasher::new();
    for v in values {
        hasher.update(&v.to_le_bytes());
    }
    let hash = hasher.finalize();
    let bytes = hash.as_bytes();
    let val = u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3],
        bytes[4], bytes[5], bytes[6], bytes[7],
    ]);
    GoldilocksField::new(val)
}

// ─────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_axioms() {
        let a = GoldilocksField::new(42);
        let b = GoldilocksField::new(1337);
        let c = GoldilocksField::new(999999);

        // Commutativity
        assert_eq!(a + b, b + a);
        assert_eq!(a * b, b * a);

        // Associativity
        assert_eq!((a + b) + c, a + (b + c));
        assert_eq!((a * b) * c, a * (b * c));

        // Identity
        assert_eq!(a + GoldilocksField::ZERO, a);
        assert_eq!(a * GoldilocksField::ONE, a);

        // Inverse
        assert_eq!(a + (-a), GoldilocksField::ZERO);
        assert_eq!(a * a.inv_or_panic(), GoldilocksField::ONE);

        // Distributivity
        assert_eq!(a * (b + c), a * b + a * c);
    }

    #[test]
    fn test_subtraction() {
        let a = GoldilocksField::new(5);
        let b = GoldilocksField::new(10);
        let result = a - b;
        // Should wrap around modulo p
        assert_eq!(result + b, a);
    }

    #[test]
    fn test_division() {
        let a = GoldilocksField::new(100);
        let b = GoldilocksField::new(7);
        let q = a / b;
        assert_eq!(q * b, a);
    }

    #[test]
    fn test_pow() {
        let a = GoldilocksField::new(3);
        assert_eq!(a.pow(0), GoldilocksField::ONE);
        assert_eq!(a.pow(1), a);
        assert_eq!(a.pow(2), a * a);
        assert_eq!(a.pow(3), a * a * a);
    }

    #[test]
    fn test_fermat_little() {
        let a = GoldilocksField::new(42);
        // a^p = a (mod p)
        assert_eq!(a.pow(GOLDILOCKS_PRIME), a);
        // a^{p-1} = 1 (mod p) for a != 0
        assert_eq!(a.pow(GOLDILOCKS_PRIME - 1), GoldilocksField::ONE);
    }

    #[test]
    fn test_batch_inversion() {
        let elements: Vec<GoldilocksField> = (1..=10)
            .map(|i| GoldilocksField::new(i))
            .collect();
        let inverses = GoldilocksField::batch_inversion(&elements);
        for (a, a_inv) in elements.iter().zip(inverses.iter()) {
            assert_eq!(*a * *a_inv, GoldilocksField::ONE);
        }
    }

    #[test]
    fn test_root_of_unity() {
        for log_n in 1..=16 {
            let n = 1usize << log_n;
            let omega = GoldilocksField::root_of_unity(n);
            // omega^n = 1
            assert_eq!(omega.pow(n as u64), GoldilocksField::ONE);
            // omega^{n/2} != 1 (primitive)
            if n > 1 {
                assert_ne!(omega.pow((n / 2) as u64), GoldilocksField::ONE);
            }
        }
    }

    #[test]
    fn test_ntt_intt_roundtrip() {
        let n = 8;
        let original: Vec<GoldilocksField> = (0..n)
            .map(|i| GoldilocksField::new(i as u64 + 1))
            .collect();

        let mut data = original.clone();
        ntt(&mut data);
        intt(&mut data);

        assert_eq!(data, original);
    }

    #[test]
    fn test_ntt_poly_mul() {
        let a = vec![
            GoldilocksField::new(1),
            GoldilocksField::new(2),
            GoldilocksField::new(3),
        ]; // 1 + 2x + 3x^2
        let b = vec![
            GoldilocksField::new(4),
            GoldilocksField::new(5),
        ]; // 4 + 5x

        let naive = GoldilocksField::poly_mul(&a, &b);
        let ntt_result = ntt_poly_mul(&a, &b);

        assert_eq!(naive.len(), ntt_result.len());
        for i in 0..naive.len() {
            assert_eq!(naive[i], ntt_result[i]);
        }
    }

    #[test]
    fn test_poly_div() {
        // (x^2 + 2x + 1) / (x + 1) = (x + 1), remainder 0
        let a = vec![
            GoldilocksField::new(1),
            GoldilocksField::new(2),
            GoldilocksField::new(1),
        ];
        let b = vec![
            GoldilocksField::new(1),
            GoldilocksField::new(1),
        ];
        let (q, r) = GoldilocksField::poly_div(&a, &b);
        assert_eq!(q, vec![GoldilocksField::new(1), GoldilocksField::new(1)]);
        assert!(r.iter().all(|x| x.is_zero()));
    }

    #[test]
    fn test_lagrange_interpolation() {
        // Interpolate through (1,3), (2,7), (3,13)
        let xs = vec![
            GoldilocksField::new(1),
            GoldilocksField::new(2),
            GoldilocksField::new(3),
        ];
        let ys = vec![
            GoldilocksField::new(3),
            GoldilocksField::new(7),
            GoldilocksField::new(13),
        ];
        let coeffs = GoldilocksField::lagrange_interpolation(&xs, &ys);

        for (x, y) in xs.iter().zip(ys.iter()) {
            assert_eq!(GoldilocksField::eval_poly(&coeffs, *x), *y);
        }
    }

    #[test]
    fn test_sqrt() {
        // 4 is a perfect square
        let four = GoldilocksField::new(4);
        let root = four.sqrt().unwrap();
        assert_eq!(root.square(), four);

        // 9 is a perfect square
        let nine = GoldilocksField::new(9);
        let root = nine.sqrt().unwrap();
        assert_eq!(root.square(), nine);
    }

    #[test]
    fn test_serialization() {
        let a = GoldilocksField::new(0x123456789ABCDEF0 % GOLDILOCKS_PRIME);
        let bytes_le = a.to_bytes_le();
        assert_eq!(GoldilocksField::from_bytes_le(&bytes_le), a);

        let bytes_be = a.to_bytes_be();
        assert_eq!(GoldilocksField::from_bytes_be(&bytes_be), a);
    }

    #[test]
    fn test_extension_field() {
        let a = GoldilocksExt::new(GoldilocksField::new(3), GoldilocksField::new(5));
        let b = GoldilocksExt::new(GoldilocksField::new(7), GoldilocksField::new(11));

        // (a + b) - b = a
        assert_eq!((a + b) - b, a);

        // a * 1 = a
        assert_eq!(a * GoldilocksExt::ONE, a);

        // a * a^{-1} = 1
        let a_inv = a.inv_ext().unwrap();
        let product = a * a_inv;
        assert!(product.a == GoldilocksField::ONE && product.b == GoldilocksField::ZERO);

        // Frobenius: sigma(a) * sigma(b) = sigma(a*b)
        assert_eq!(a.frobenius() * b.frobenius(), (a * b).frobenius());
    }

    #[test]
    fn test_coset_evaluation() {
        let coeffs = vec![
            GoldilocksField::new(1),
            GoldilocksField::new(2),
            GoldilocksField::new(3),
            GoldilocksField::new(4),
        ];
        let shift = GoldilocksField::new(5);

        let evals = evaluate_on_coset(&coeffs, shift, 4);
        let recovered = interpolate_from_coset(&evals, shift);

        for i in 0..4 {
            assert_eq!(coeffs[i], recovered[i]);
        }
    }

    #[test]
    fn test_vanishing_poly() {
        let n = 4;
        let roots = GoldilocksField::roots_of_unity(n);
        let v_coeffs = vanishing_poly(n);

        for &root in &roots {
            let val = GoldilocksField::eval_poly(&v_coeffs, root);
            assert!(val.is_zero(), "vanishing polynomial should be 0 at root of unity");
        }
    }

    #[test]
    fn test_inner_product() {
        let a = vec![GoldilocksField::new(1), GoldilocksField::new(2), GoldilocksField::new(3)];
        let b = vec![GoldilocksField::new(4), GoldilocksField::new(5), GoldilocksField::new(6)];
        let ip = GoldilocksField::inner_product(&a, &b);
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert_eq!(ip, GoldilocksField::new(32));
    }

    #[test]
    fn test_extension_serialization() {
        let a = GoldilocksExt::new(GoldilocksField::new(42), GoldilocksField::new(1337));
        let bytes = a.to_bytes();
        let recovered = GoldilocksExt::from_bytes(&bytes);
        assert_eq!(a, recovered);
    }

    #[test]
    fn test_montgomery() {
        let a = GoldilocksField::new(42);
        let b = GoldilocksField::new(1337);

        let a_mont = a.to_montgomery();
        let b_mont = b.to_montgomery();

        let product_mont = GoldilocksField::montgomery_mul(a_mont, b_mont);
        let product = GoldilocksField::from_montgomery(product_mont);

        assert_eq!(product, a * b);
    }

    // ─────────────────────────────────────────────────────────
    // Tests for multi-precision polynomial operations
    // ─────────────────────────────────────────────────────────

    #[test]
    fn test_poly_is_zero() {
        assert!(poly_is_zero(&[]));
        assert!(poly_is_zero(&[GoldilocksField::ZERO]));
        assert!(poly_is_zero(&[GoldilocksField::ZERO, GoldilocksField::ZERO]));
        assert!(!poly_is_zero(&[GoldilocksField::ONE]));
        assert!(!poly_is_zero(&[GoldilocksField::ZERO, GoldilocksField::ONE]));
    }

    #[test]
    fn test_poly_degree() {
        assert_eq!(poly_degree(&[]), 0);
        assert_eq!(poly_degree(&[GoldilocksField::ZERO]), 0);
        assert_eq!(poly_degree(&[GoldilocksField::new(5)]), 0);
        assert_eq!(poly_degree(&[GoldilocksField::new(1), GoldilocksField::new(2)]), 1);
        // Trailing zeros should not count
        assert_eq!(poly_degree(&[GoldilocksField::new(1), GoldilocksField::new(2), GoldilocksField::ZERO]), 1);
    }

    #[test]
    fn test_poly_trim() {
        let mut p = vec![GoldilocksField::new(1), GoldilocksField::new(2), GoldilocksField::ZERO, GoldilocksField::ZERO];
        poly_trim(&mut p);
        assert_eq!(p.len(), 2);
        assert_eq!(p[0], GoldilocksField::new(1));
        assert_eq!(p[1], GoldilocksField::new(2));

        // Should keep at least one element
        let mut zero_poly = vec![GoldilocksField::ZERO, GoldilocksField::ZERO];
        poly_trim(&mut zero_poly);
        assert_eq!(zero_poly.len(), 1);
    }

    #[test]
    fn test_poly_mod() {
        // (x^2 + 2x + 1) mod (x + 1) = 0
        let a = vec![GoldilocksField::new(1), GoldilocksField::new(2), GoldilocksField::new(1)];
        let m = vec![GoldilocksField::new(1), GoldilocksField::new(1)];
        let r = poly_mod(&a, &m);
        assert!(r.iter().all(|c| c.is_zero()));

        // (x^2 + 1) mod (x + 1) should give remainder 2
        let a2 = vec![GoldilocksField::new(1), GoldilocksField::ZERO, GoldilocksField::new(1)];
        let r2 = poly_mod(&a2, &m);
        assert_eq!(r2[0], GoldilocksField::new(2));
    }

    #[test]
    fn test_poly_gcd() {
        // gcd of (x^2 - 1) and (x - 1) should be (x - 1) (monic)
        let p = GOLDILOCKS_PRIME;
        let a = vec![GoldilocksField::new(p - 1), GoldilocksField::ZERO, GoldilocksField::ONE]; // x^2 - 1
        let b = vec![GoldilocksField::new(p - 1), GoldilocksField::ONE]; // x - 1
        let g = poly_gcd(&a, &b);
        // Result should be monic (x - 1) = [-1, 1]
        assert_eq!(g.len(), 2);
        assert_eq!(g[1], GoldilocksField::ONE);
        assert_eq!(g[0], GoldilocksField::new(p - 1));
    }

    #[test]
    fn test_poly_derivative() {
        // d/dx (3x^3 + 2x^2 + x + 5) = 9x^2 + 4x + 1
        let p = vec![
            GoldilocksField::new(5),
            GoldilocksField::new(1),
            GoldilocksField::new(2),
            GoldilocksField::new(3),
        ];
        let dp = poly_derivative(&p);
        assert_eq!(dp.len(), 3);
        assert_eq!(dp[0], GoldilocksField::new(1));
        assert_eq!(dp[1], GoldilocksField::new(4));
        assert_eq!(dp[2], GoldilocksField::new(9));

        // Derivative of constant is zero
        let c = vec![GoldilocksField::new(42)];
        let dc = poly_derivative(&c);
        assert_eq!(dc.len(), 1);
        assert!(dc[0].is_zero());
    }

    #[test]
    fn test_poly_scale() {
        let p = vec![GoldilocksField::new(1), GoldilocksField::new(2), GoldilocksField::new(3)];
        let c = GoldilocksField::new(5);
        let scaled = poly_scale(&p, c);
        assert_eq!(scaled[0], GoldilocksField::new(5));
        assert_eq!(scaled[1], GoldilocksField::new(10));
        assert_eq!(scaled[2], GoldilocksField::new(15));
    }

    #[test]
    fn test_poly_shift() {
        // p(x) = x^2 + 1, shift by c=2 => p(x+2) = (x+2)^2 + 1 = x^2 + 4x + 5
        let p = vec![GoldilocksField::new(1), GoldilocksField::ZERO, GoldilocksField::ONE];
        let shifted = poly_shift(&p, GoldilocksField::new(2));
        assert_eq!(shifted[0], GoldilocksField::new(5)); // constant
        assert_eq!(shifted[1], GoldilocksField::new(4)); // x coeff
        assert_eq!(shifted[2], GoldilocksField::ONE);    // x^2 coeff

        // Verify by evaluation: p(x+2) at x=3 should equal p(5)
        let val_shifted = GoldilocksField::eval_poly(&shifted, GoldilocksField::new(3));
        let val_original = GoldilocksField::eval_poly(&p, GoldilocksField::new(5));
        assert_eq!(val_shifted, val_original);
    }

    #[test]
    fn test_poly_compose() {
        // f(x) = x^2 + 1, g(x) = 2x + 3
        // f(g(x)) = (2x+3)^2 + 1 = 4x^2 + 12x + 10
        let f = vec![GoldilocksField::new(1), GoldilocksField::ZERO, GoldilocksField::ONE];
        let g = vec![GoldilocksField::new(3), GoldilocksField::new(2)];
        let composed = poly_compose(&f, &g);
        assert_eq!(composed[0], GoldilocksField::new(10));
        assert_eq!(composed[1], GoldilocksField::new(12));
        assert_eq!(composed[2], GoldilocksField::new(4));

        // Verify: f(g(1)) = f(5) = 26
        let val = GoldilocksField::eval_poly(&composed, GoldilocksField::new(1));
        assert_eq!(val, GoldilocksField::new(26));
    }

    #[test]
    fn test_poly_from_roots() {
        // Roots: 1, 2, 3 => (x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x - 6
        let roots = vec![
            GoldilocksField::new(1),
            GoldilocksField::new(2),
            GoldilocksField::new(3),
        ];
        let p = poly_from_roots(&roots);
        assert_eq!(p.len(), 4);

        // Verify all roots evaluate to zero
        for &r in &roots {
            assert!(GoldilocksField::eval_poly(&p, r).is_zero());
        }

        // Verify a non-root doesn't evaluate to zero
        assert!(!GoldilocksField::eval_poly(&p, GoldilocksField::new(4)).is_zero());
    }

    #[test]
    fn test_poly_from_roots_empty() {
        let p = poly_from_roots(&[]);
        assert_eq!(p.len(), 1);
        assert_eq!(p[0], GoldilocksField::ONE);
    }

    // ─────────────────────────────────────────────────────────
    // Tests for FieldMatrix
    // ─────────────────────────────────────────────────────────

    #[test]
    fn test_matrix_identity() {
        let id = FieldMatrix::identity(3);
        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    assert_eq!(id.get(i, j), GoldilocksField::ONE);
                } else {
                    assert_eq!(id.get(i, j), GoldilocksField::ZERO);
                }
            }
        }
    }

    #[test]
    fn test_matrix_mul() {
        // [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]]
        let a = FieldMatrix::from_rows(vec![
            vec![GoldilocksField::new(1), GoldilocksField::new(2)],
            vec![GoldilocksField::new(3), GoldilocksField::new(4)],
        ]);
        let b = FieldMatrix::from_rows(vec![
            vec![GoldilocksField::new(5), GoldilocksField::new(6)],
            vec![GoldilocksField::new(7), GoldilocksField::new(8)],
        ]);
        let c = a.mul_matrix(&b);
        assert_eq!(c.get(0, 0), GoldilocksField::new(19));
        assert_eq!(c.get(0, 1), GoldilocksField::new(22));
        assert_eq!(c.get(1, 0), GoldilocksField::new(43));
        assert_eq!(c.get(1, 1), GoldilocksField::new(50));
    }

    #[test]
    fn test_matrix_mul_identity() {
        let a = FieldMatrix::from_rows(vec![
            vec![GoldilocksField::new(1), GoldilocksField::new(2)],
            vec![GoldilocksField::new(3), GoldilocksField::new(4)],
        ]);
        let id = FieldMatrix::identity(2);
        assert_eq!(a.mul_matrix(&id), a);
        assert_eq!(id.mul_matrix(&a), a);
    }

    #[test]
    fn test_matrix_mul_vector() {
        let a = FieldMatrix::from_rows(vec![
            vec![GoldilocksField::new(1), GoldilocksField::new(2)],
            vec![GoldilocksField::new(3), GoldilocksField::new(4)],
        ]);
        let v = vec![GoldilocksField::new(5), GoldilocksField::new(6)];
        let result = a.mul_vector(&v);
        assert_eq!(result[0], GoldilocksField::new(17)); // 1*5 + 2*6
        assert_eq!(result[1], GoldilocksField::new(39)); // 3*5 + 4*6
    }

    #[test]
    fn test_matrix_transpose() {
        let a = FieldMatrix::from_rows(vec![
            vec![GoldilocksField::new(1), GoldilocksField::new(2), GoldilocksField::new(3)],
            vec![GoldilocksField::new(4), GoldilocksField::new(5), GoldilocksField::new(6)],
        ]);
        let t = a.transpose();
        assert_eq!(t.rows, 3);
        assert_eq!(t.cols, 2);
        assert_eq!(t.get(0, 0), GoldilocksField::new(1));
        assert_eq!(t.get(0, 1), GoldilocksField::new(4));
        assert_eq!(t.get(2, 1), GoldilocksField::new(6));
    }

    #[test]
    fn test_matrix_trace() {
        let a = FieldMatrix::from_rows(vec![
            vec![GoldilocksField::new(1), GoldilocksField::new(2)],
            vec![GoldilocksField::new(3), GoldilocksField::new(4)],
        ]);
        assert_eq!(a.trace(), GoldilocksField::new(5)); // 1 + 4
    }

    #[test]
    fn test_matrix_determinant() {
        // det([[1,2],[3,4]]) = 1*4 - 2*3 = -2
        let a = FieldMatrix::from_rows(vec![
            vec![GoldilocksField::new(1), GoldilocksField::new(2)],
            vec![GoldilocksField::new(3), GoldilocksField::new(4)],
        ]);
        let det = a.determinant();
        // -2 mod p = p - 2
        assert_eq!(det, GoldilocksField::new(GOLDILOCKS_PRIME - 2));
    }

    #[test]
    fn test_matrix_determinant_singular() {
        // Singular matrix: rows are linearly dependent
        let a = FieldMatrix::from_rows(vec![
            vec![GoldilocksField::new(1), GoldilocksField::new(2)],
            vec![GoldilocksField::new(2), GoldilocksField::new(4)],
        ]);
        assert_eq!(a.determinant(), GoldilocksField::ZERO);
    }

    #[test]
    fn test_matrix_inverse() {
        let a = FieldMatrix::from_rows(vec![
            vec![GoldilocksField::new(1), GoldilocksField::new(2)],
            vec![GoldilocksField::new(3), GoldilocksField::new(4)],
        ]);
        let a_inv = a.inverse().unwrap();
        let product = a.mul_matrix(&a_inv);
        let id = FieldMatrix::identity(2);
        assert_eq!(product, id);
    }

    #[test]
    fn test_matrix_inverse_singular() {
        let a = FieldMatrix::from_rows(vec![
            vec![GoldilocksField::new(1), GoldilocksField::new(2)],
            vec![GoldilocksField::new(2), GoldilocksField::new(4)],
        ]);
        assert!(a.inverse().is_none());
    }

    #[test]
    fn test_matrix_row_echelon_and_rank() {
        let a = FieldMatrix::from_rows(vec![
            vec![GoldilocksField::new(1), GoldilocksField::new(2), GoldilocksField::new(3)],
            vec![GoldilocksField::new(4), GoldilocksField::new(5), GoldilocksField::new(6)],
            vec![GoldilocksField::new(7), GoldilocksField::new(8), GoldilocksField::new(9)],
        ]);
        let rank = a.rank();
        // This matrix has rank 2 (third row = 2*second - first)
        assert_eq!(rank, 2);
    }

    #[test]
    fn test_matrix_kernel() {
        // Rank-2 matrix with 3 columns => kernel dimension 1
        let a = FieldMatrix::from_rows(vec![
            vec![GoldilocksField::new(1), GoldilocksField::new(2), GoldilocksField::new(3)],
            vec![GoldilocksField::new(4), GoldilocksField::new(5), GoldilocksField::new(6)],
        ]);
        let ker = a.kernel();
        assert_eq!(ker.len(), 1);
        // Verify: A * v = 0
        let v = &ker[0];
        let result = a.mul_vector(v);
        for r in &result {
            assert!(r.is_zero());
        }
    }

    #[test]
    fn test_matrix_pow() {
        let a = FieldMatrix::from_rows(vec![
            vec![GoldilocksField::new(1), GoldilocksField::new(1)],
            vec![GoldilocksField::new(1), GoldilocksField::new(0)],
        ]);
        // A^0 = I
        assert_eq!(a.pow_matrix(0), FieldMatrix::identity(2));
        // A^1 = A
        assert_eq!(a.pow_matrix(1), a);
        // A^2 = A*A
        assert_eq!(a.pow_matrix(2), a.mul_matrix(&a));
        // A^3 = A*A*A
        assert_eq!(a.pow_matrix(3), a.mul_matrix(&a).mul_matrix(&a));
    }

    #[test]
    fn test_matrix_characteristic_polynomial() {
        // For a 2x2 matrix [[a,b],[c,d]], char poly is x^2 - (a+d)x + (ad-bc)
        let m = FieldMatrix::from_rows(vec![
            vec![GoldilocksField::new(1), GoldilocksField::new(2)],
            vec![GoldilocksField::new(3), GoldilocksField::new(4)],
        ]);
        let cp = m.characteristic_polynomial();
        assert_eq!(cp.len(), 3);
        // c_2 = 1
        assert_eq!(cp[2], GoldilocksField::ONE);
        // c_1 = -(1+4) = -5
        assert_eq!(cp[1], GoldilocksField::new(GOLDILOCKS_PRIME - 5));
        // c_0 = 1*4 - 2*3 = -2
        assert_eq!(cp[0], GoldilocksField::new(GOLDILOCKS_PRIME - 2));
    }

    #[test]
    fn test_solve_linear() {
        // Solve: [[1,2],[3,4]] * x = [5, 11]
        // x = [-1, 3] since 1*(-1)+2*3=5, 3*(-1)+4*3=9... let's pick a solvable system
        // [[1,0],[0,1]] * x = [3,7] => x = [3,7]
        let a = FieldMatrix::identity(2);
        let b = vec![GoldilocksField::new(3), GoldilocksField::new(7)];
        let x = solve_linear(&a, &b).unwrap();
        assert_eq!(x[0], GoldilocksField::new(3));
        assert_eq!(x[1], GoldilocksField::new(7));
    }

    #[test]
    fn test_solve_linear_nontrivial() {
        // [[2,1],[1,3]] * x = [5, 10]
        // Solution: 2x+y=5, x+3y=10 => x=1, y=3
        let a = FieldMatrix::from_rows(vec![
            vec![GoldilocksField::new(2), GoldilocksField::new(1)],
            vec![GoldilocksField::new(1), GoldilocksField::new(3)],
        ]);
        let b = vec![GoldilocksField::new(5), GoldilocksField::new(10)];
        let x = solve_linear(&a, &b).unwrap();
        assert_eq!(x[0], GoldilocksField::new(1));
        assert_eq!(x[1], GoldilocksField::new(3));
    }

    #[test]
    fn test_matrix_from_rows_and_get_set() {
        let mut m = FieldMatrix::from_rows(vec![
            vec![GoldilocksField::new(1), GoldilocksField::new(2)],
            vec![GoldilocksField::new(3), GoldilocksField::new(4)],
        ]);
        assert_eq!(m.get(0, 1), GoldilocksField::new(2));
        m.set(0, 1, GoldilocksField::new(99));
        assert_eq!(m.get(0, 1), GoldilocksField::new(99));
    }

    #[test]
    fn test_matrix_zeros() {
        let m = FieldMatrix::zeros(2, 3);
        assert_eq!(m.rows, 2);
        assert_eq!(m.cols, 3);
        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(m.get(i, j), GoldilocksField::ZERO);
            }
        }
    }

    // ─────────────────────────────────────────────────────────
    // Tests for DFT utilities
    // ─────────────────────────────────────────────────────────

    #[test]
    fn test_precompute_twiddle_factors() {
        let n = 8;
        let factors = precompute_twiddle_factors(n);
        assert_eq!(factors.len(), n);
        assert_eq!(factors[0], GoldilocksField::ONE);
        // factors[n] should wrap around to 1
        let omega = GoldilocksField::root_of_unity(n);
        for i in 0..n {
            assert_eq!(factors[i], omega.pow(i as u64));
        }
    }

    #[test]
    fn test_ntt_with_precomputed_roots() {
        let n = 8;
        let roots = precompute_twiddle_factors(n);
        let original: Vec<GoldilocksField> = (0..n)
            .map(|i| GoldilocksField::new(i as u64 + 1))
            .collect();

        let mut data1 = original.clone();
        ntt(&mut data1);

        let mut data2 = original.clone();
        ntt_with_precomputed_roots(&mut data2, &roots);

        assert_eq!(data1, data2);
    }

    #[test]
    fn test_coset_ntt_intt_roundtrip() {
        let n = 8;
        let shift = GoldilocksField::new(5);
        let original: Vec<GoldilocksField> = (0..n)
            .map(|i| GoldilocksField::new(i as u64 + 1))
            .collect();

        let mut data = original.clone();
        coset_ntt(&mut data, shift);
        coset_intt(&mut data, shift);

        assert_eq!(data, original);
    }

    #[test]
    fn test_multi_ntt_intt() {
        let n = 4;
        let original1: Vec<GoldilocksField> = (0..n).map(|i| GoldilocksField::new(i as u64 + 1)).collect();
        let original2: Vec<GoldilocksField> = (0..n).map(|i| GoldilocksField::new(i as u64 + 10)).collect();

        let mut polys = vec![original1.clone(), original2.clone()];
        multi_ntt(&mut polys);

        // Verify NTT was applied
        let mut check1 = original1.clone();
        ntt(&mut check1);
        assert_eq!(polys[0], check1);

        multi_intt(&mut polys);
        assert_eq!(polys[0], original1);
        assert_eq!(polys[1], original2);
    }

    // ─────────────────────────────────────────────────────────
    // Tests for Reed-Solomon encoding/decoding
    // ─────────────────────────────────────────────────────────

    #[test]
    fn test_reed_solomon_encode_decode() {
        // Create a polynomial: 1 + 2x + 3x^2 + 4x^3
        let data = vec![
            GoldilocksField::new(1),
            GoldilocksField::new(2),
            GoldilocksField::new(3),
            GoldilocksField::new(4),
        ];

        let encoded = reed_solomon_encode(&data, 2);
        assert!(encoded.len() >= data.len() * 2);

        // Verify that the encoded values match polynomial evaluations on the domain
        let n = encoded.len();
        let omega = GoldilocksField::root_of_unity(n);
        for i in 0..n {
            let point = omega.pow(i as u64);
            let expected = GoldilocksField::eval_poly(&data, point);
            assert_eq!(encoded[i], expected);
        }
    }

    #[test]
    fn test_reed_solomon_decode_interpolation() {
        let data = vec![
            GoldilocksField::new(1),
            GoldilocksField::new(2),
            GoldilocksField::new(3),
            GoldilocksField::new(4),
        ];

        // Evaluate at specific points
        let points: Vec<GoldilocksField> = (0..8)
            .map(|i| GoldilocksField::new(i as u64 + 1))
            .collect();
        let evals: Vec<GoldilocksField> = points.iter()
            .map(|&x| GoldilocksField::eval_poly(&data, x))
            .collect();

        let recovered = reed_solomon_decode(&evals, &points, 4);
        assert_eq!(recovered.len(), 4);
        for i in 0..4 {
            assert_eq!(recovered[i], data[i]);
        }
    }

    #[test]
    fn test_erasure_decode() {
        // Polynomial: 1 + 2x (degree 1)
        let poly = vec![GoldilocksField::new(1), GoldilocksField::new(2)];
        let domain: Vec<GoldilocksField> = (0..4)
            .map(|i| GoldilocksField::new(i as u64 + 1))
            .collect();
        let evals: Vec<Option<GoldilocksField>> = domain.iter()
            .map(|&x| Some(GoldilocksField::eval_poly(&poly, x)))
            .collect();

        // Erase some evaluations
        let mut partial = evals.clone();
        partial[1] = None;
        partial[3] = None;

        let recovered = erasure_decode(&partial, &domain, 1).unwrap();
        assert_eq!(recovered[0], GoldilocksField::new(1));
        assert_eq!(recovered[1], GoldilocksField::new(2));
    }

    #[test]
    fn test_erasure_decode_insufficient() {
        let domain = vec![GoldilocksField::new(1), GoldilocksField::new(2)];
        let evals = vec![None, Some(GoldilocksField::new(5))];
        // Need degree+1=2 known values for degree 1, only have 1
        assert!(erasure_decode(&evals, &domain, 1).is_none());
    }

    // ─────────────────────────────────────────────────────────
    // Tests for MultilinearExtension
    // ─────────────────────────────────────────────────────────

    #[test]
    fn test_multilinear_extension_basic() {
        // f(x) = 3*(1-x) + 7*x = 3 + 4x, so evaluations = [3, 7]
        let mle = MultilinearExtension::new(vec![
            GoldilocksField::new(3),
            GoldilocksField::new(7),
        ]);
        assert_eq!(mle.num_variables(), 1);

        // f(0) = 3
        assert_eq!(mle.evaluate(&[GoldilocksField::ZERO]), GoldilocksField::new(3));
        // f(1) = 7
        assert_eq!(mle.evaluate(&[GoldilocksField::ONE]), GoldilocksField::new(7));
        // f(2) = 3 + 4*2 = 11
        assert_eq!(mle.evaluate(&[GoldilocksField::new(2)]), GoldilocksField::new(11));
    }

    #[test]
    fn test_multilinear_extension_2var() {
        // f(x1, x2) with evaluations on {0,1}^2:
        // f(0,0)=1, f(1,0)=2, f(0,1)=3, f(1,1)=4
        let mle = MultilinearExtension::new(vec![
            GoldilocksField::new(1), // (0,0)
            GoldilocksField::new(2), // (1,0)
            GoldilocksField::new(3), // (0,1)
            GoldilocksField::new(4), // (1,1)
        ]);
        assert_eq!(mle.num_variables(), 2);

        // Check boolean hypercube evaluations
        assert_eq!(mle.evaluate(&[GoldilocksField::ZERO, GoldilocksField::ZERO]), GoldilocksField::new(1));
        assert_eq!(mle.evaluate(&[GoldilocksField::ONE, GoldilocksField::ZERO]), GoldilocksField::new(2));
        assert_eq!(mle.evaluate(&[GoldilocksField::ZERO, GoldilocksField::ONE]), GoldilocksField::new(3));
        assert_eq!(mle.evaluate(&[GoldilocksField::ONE, GoldilocksField::ONE]), GoldilocksField::new(4));
    }

    #[test]
    fn test_eq_polynomial() {
        // eq((r), x) for x in {0, 1} should give [(1-r), r]
        let r = GoldilocksField::new(3);
        let eq = MultilinearExtension::eq_polynomial(&[r]);
        assert_eq!(eq.len(), 2);
        assert_eq!(eq[0], GoldilocksField::ONE - r); // 1-r
        assert_eq!(eq[1], r);
    }

    #[test]
    fn test_partial_evaluate() {
        // f(x1, x2) with evaluations [1, 2, 3, 4]
        let mle = MultilinearExtension::new(vec![
            GoldilocksField::new(1),
            GoldilocksField::new(2),
            GoldilocksField::new(3),
            GoldilocksField::new(4),
        ]);

        // Fix x1 = 0: should get [f(0,0), f(0,1)] = [1, 3]
        let partial = mle.partial_evaluate(0, GoldilocksField::ZERO);
        assert_eq!(partial.num_variables(), 1);
        assert_eq!(partial.evaluations[0], GoldilocksField::new(1));
        assert_eq!(partial.evaluations[1], GoldilocksField::new(3));

        // Fix x1 = 1: should get [f(1,0), f(1,1)] = [2, 4]
        let partial1 = mle.partial_evaluate(0, GoldilocksField::ONE);
        assert_eq!(partial1.evaluations[0], GoldilocksField::new(2));
        assert_eq!(partial1.evaluations[1], GoldilocksField::new(4));
    }

    #[test]
    fn test_sum_over_boolean_hypercube() {
        let mle = MultilinearExtension::new(vec![
            GoldilocksField::new(1),
            GoldilocksField::new(2),
            GoldilocksField::new(3),
            GoldilocksField::new(4),
        ]);
        assert_eq!(mle.sum_over_boolean_hypercube(), GoldilocksField::new(10));
    }

    #[test]
    fn test_tensor_product() {
        let a = vec![GoldilocksField::new(1), GoldilocksField::new(2)];
        let b = vec![GoldilocksField::new(3), GoldilocksField::new(4), GoldilocksField::new(5)];
        let tp = tensor_product(&a, &b);
        assert_eq!(tp.len(), 6);
        assert_eq!(tp[0], GoldilocksField::new(3));  // 1*3
        assert_eq!(tp[1], GoldilocksField::new(4));  // 1*4
        assert_eq!(tp[2], GoldilocksField::new(5));  // 1*5
        assert_eq!(tp[3], GoldilocksField::new(6));  // 2*3
        assert_eq!(tp[4], GoldilocksField::new(8));  // 2*4
        assert_eq!(tp[5], GoldilocksField::new(10)); // 2*5
    }

    #[test]
    fn test_partial_evaluate_consistency() {
        // Partially evaluating all variables should match full evaluate
        let mle = MultilinearExtension::new(vec![
            GoldilocksField::new(1),
            GoldilocksField::new(2),
            GoldilocksField::new(3),
            GoldilocksField::new(4),
        ]);
        let r0 = GoldilocksField::new(5);
        let r1 = GoldilocksField::new(7);

        let full_eval = mle.evaluate(&[r0, r1]);

        let partial = mle.partial_evaluate(0, r0);
        let final_eval = partial.evaluate(&[r1]);

        assert_eq!(full_eval, final_eval);
    }

    // ─────────────────────────────────────────────────────────
    // Tests for hash-to-field functions
    // ─────────────────────────────────────────────────────────

    #[test]
    fn test_hash_to_field_deterministic() {
        let data = b"hello world";
        let h1 = hash_to_field(data);
        let h2 = hash_to_field(data);
        assert_eq!(h1, h2);

        // Different input should (almost certainly) give different output
        let h3 = hash_to_field(b"hello world!");
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_hash_to_field_in_range() {
        let h = hash_to_field(b"test");
        assert!(h.to_canonical() < GOLDILOCKS_PRIME);
    }

    #[test]
    fn test_hash_to_field_vec() {
        let data = b"generate multiple";
        let result = hash_to_field_vec(data, 5);
        assert_eq!(result.len(), 5);

        // All values should be valid field elements
        for r in &result {
            assert!(r.to_canonical() < GOLDILOCKS_PRIME);
        }

        // Elements should be distinct (with overwhelming probability)
        for i in 0..result.len() {
            for j in (i + 1)..result.len() {
                assert_ne!(result[i], result[j]);
            }
        }
    }

    #[test]
    fn test_hash_pair_to_field() {
        let a = b"first";
        let b_data = b"second";
        let h1 = hash_pair_to_field(a, b_data);
        let h2 = hash_pair_to_field(a, b_data);
        assert_eq!(h1, h2);

        // Order matters
        let h3 = hash_pair_to_field(b_data, a);
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_hash_u64s_to_field() {
        let values = vec![1u64, 2, 3, 4, 5];
        let h1 = hash_u64s_to_field(&values);
        let h2 = hash_u64s_to_field(&values);
        assert_eq!(h1, h2);

        let h3 = hash_u64s_to_field(&[1, 2, 3, 4, 6]);
        assert_ne!(h1, h3);

        assert!(h1.to_canonical() < GOLDILOCKS_PRIME);
    }

    #[test]
    fn test_hash_to_field_empty() {
        // Should not panic on empty input
        let h = hash_to_field(b"");
        assert!(h.to_canonical() < GOLDILOCKS_PRIME);
        let hv = hash_to_field_vec(b"", 3);
        assert_eq!(hv.len(), 3);
    }

    #[test]
    fn test_matrix_3x3_determinant() {
        // [[2,1,1],[1,3,2],[1,0,0]] det = 2*(0)-1*(0-2)+1*(0-3) = 0+2-3 = -1
        let m = FieldMatrix::from_rows(vec![
            vec![GoldilocksField::new(2), GoldilocksField::new(1), GoldilocksField::new(1)],
            vec![GoldilocksField::new(1), GoldilocksField::new(3), GoldilocksField::new(2)],
            vec![GoldilocksField::new(1), GoldilocksField::ZERO, GoldilocksField::ZERO],
        ]);
        let det = m.determinant();
        assert_eq!(det, GoldilocksField::new(GOLDILOCKS_PRIME - 1));
    }

    #[test]
    fn test_matrix_3x3_inverse() {
        let m = FieldMatrix::from_rows(vec![
            vec![GoldilocksField::new(2), GoldilocksField::new(1), GoldilocksField::new(1)],
            vec![GoldilocksField::new(1), GoldilocksField::new(3), GoldilocksField::new(2)],
            vec![GoldilocksField::new(1), GoldilocksField::ZERO, GoldilocksField::ZERO],
        ]);
        let m_inv = m.inverse().unwrap();
        let product = m.mul_matrix(&m_inv);
        assert_eq!(product, FieldMatrix::identity(3));
    }

    #[test]
    fn test_solve_linear_3x3() {
        let a = FieldMatrix::from_rows(vec![
            vec![GoldilocksField::new(2), GoldilocksField::new(1), GoldilocksField::new(1)],
            vec![GoldilocksField::new(1), GoldilocksField::new(3), GoldilocksField::new(2)],
            vec![GoldilocksField::new(1), GoldilocksField::ZERO, GoldilocksField::ZERO],
        ]);
        let x_expected = vec![
            GoldilocksField::new(1),
            GoldilocksField::new(2),
            GoldilocksField::new(3),
        ];
        let b = a.mul_vector(&x_expected);
        let x = solve_linear(&a, &b).unwrap();
        assert_eq!(x, x_expected);
    }

    #[test]
    fn test_coset_ntt_correctness() {
        // Coset NTT should give evaluations on the coset {shift * omega^i}
        let coeffs = vec![
            GoldilocksField::new(1),
            GoldilocksField::new(2),
            GoldilocksField::new(3),
            GoldilocksField::new(4),
        ];
        let shift = GoldilocksField::new(7);
        let n = coeffs.len();

        let mut data = coeffs.clone();
        coset_ntt(&mut data, shift);

        let omega = GoldilocksField::root_of_unity(n);
        for i in 0..n {
            let point = shift.mul_elem(omega.pow(i as u64));
            let expected = GoldilocksField::eval_poly(&coeffs, point);
            assert_eq!(data[i], expected);
        }
    }

    #[test]
    fn test_poly_gcd_coprime() {
        // gcd(x + 1, x + 2) should be 1 (constant, monic = [1])
        let a = vec![GoldilocksField::new(1), GoldilocksField::ONE];
        let b = vec![GoldilocksField::new(2), GoldilocksField::ONE];
        let g = poly_gcd(&a, &b);
        assert_eq!(g.len(), 1);
        assert_eq!(g[0], GoldilocksField::ONE);
    }

    #[test]
    fn test_multilinear_extension_3var() {
        // 3 variables => 8 evaluations
        let evals: Vec<GoldilocksField> = (0..8).map(|i| GoldilocksField::new(i as u64 + 1)).collect();
        let mle = MultilinearExtension::new(evals.clone());
        assert_eq!(mle.num_variables(), 3);

        // Check corners of the hypercube
        assert_eq!(mle.evaluate(&[GoldilocksField::ZERO, GoldilocksField::ZERO, GoldilocksField::ZERO]),
                   GoldilocksField::new(1));
        assert_eq!(mle.evaluate(&[GoldilocksField::ONE, GoldilocksField::ONE, GoldilocksField::ONE]),
                   GoldilocksField::new(8));
    }

    #[test]
    fn test_poly_derivative_constant() {
        let p = vec![];
        let dp = poly_derivative(&p);
        assert_eq!(dp, vec![GoldilocksField::ZERO]);
    }

    #[test]
    fn test_poly_compose_identity() {
        // f(g(x)) where g(x) = x should equal f(x)
        let f = vec![GoldilocksField::new(1), GoldilocksField::new(2), GoldilocksField::new(3)];
        let g = vec![GoldilocksField::ZERO, GoldilocksField::ONE]; // g(x) = x
        let composed = poly_compose(&f, &g);
        // composed may have trailing zeros, but should evaluate the same
        for i in 0..10 {
            let x = GoldilocksField::new(i);
            assert_eq!(
                GoldilocksField::eval_poly(&composed, x),
                GoldilocksField::eval_poly(&f, x)
            );
        }
    }

    #[test]
    fn test_matrix_pow_fibonacci() {
        // Fibonacci via matrix exponentiation:
        // [[1,1],[1,0]]^n gives F(n+1) in top-left
        let fib_mat = FieldMatrix::from_rows(vec![
            vec![GoldilocksField::new(1), GoldilocksField::new(1)],
            vec![GoldilocksField::new(1), GoldilocksField::ZERO],
        ]);

        let m10 = fib_mat.pow_matrix(10);
        // F(11) = 89, F(10) = 55
        assert_eq!(m10.get(0, 0), GoldilocksField::new(89));
        assert_eq!(m10.get(0, 1), GoldilocksField::new(55));
    }
}
