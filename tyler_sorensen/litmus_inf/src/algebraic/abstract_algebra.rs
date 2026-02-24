//! Abstract algebra library for LITMUS∞.
//!
//! Provides algebraic traits (Magma, Semigroup, Monoid, Group, Ring, Field),
//! finite field arithmetic (GF(p)), polynomial rings over finite fields,
//! field extensions, matrix algebra over finite fields, and vector spaces.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::ops::{Add, Sub, Mul, Div, Neg, AddAssign, MulAssign};
use std::hash::{Hash, Hasher};

// ═══════════════════════════════════════════════════════════════════════
// Core algebraic traits
// ═══════════════════════════════════════════════════════════════════════

/// A set with a binary operation.
pub trait Magma: Clone + PartialEq + fmt::Debug {
    /// The binary operation.
    fn op(&self, other: &Self) -> Self;
}

/// A magma with an associative operation.
pub trait Semigroup: Magma {}

/// A semigroup with an identity element.
pub trait Monoid: Semigroup {
    /// The identity element.
    fn identity() -> Self;

    /// Check if this is the identity element.
    fn is_identity(&self) -> bool {
        *self == Self::identity()
    }
}

/// A monoid where every element has an inverse.
pub trait Group: Monoid {
    /// The inverse of this element.
    fn inverse(&self) -> Self;

    /// Compute self^n for integer n (repeated application).
    fn pow(&self, n: i64) -> Self {
        if n == 0 {
            return Self::identity();
        }
        if n < 0 {
            return self.inverse().pow(-n);
        }
        let mut result = Self::identity();
        let mut base = self.clone();
        let mut exp = n as u64;
        while exp > 0 {
            if exp & 1 == 1 {
                result = result.op(&base);
            }
            base = base.op(&base);
            exp >>= 1;
        }
        result
    }

    /// Compute the order of this element (smallest n > 0 such that self^n = identity).
    fn order(&self) -> usize {
        let mut current = self.clone();
        let mut n = 1;
        loop {
            if current == Self::identity() { return n; }
            current = current.op(self);
            n += 1;
            if n > 10000 { break; }
        }
        n
    }
}

/// An abelian (commutative) group.
pub trait AbelianGroup: Group {}

/// A ring: abelian group under addition, monoid under multiplication, with distributivity.
pub trait Ring: Clone + PartialEq + fmt::Debug + Sized {
    /// Additive identity (zero).
    fn zero() -> Self;
    /// Multiplicative identity (one).
    fn one() -> Self;
    /// Addition.
    fn ring_add(&self, other: &Self) -> Self;
    /// Additive inverse (negation).
    fn ring_neg(&self) -> Self;
    /// Subtraction.
    fn ring_sub(&self, other: &Self) -> Self {
        self.ring_add(&other.ring_neg())
    }
    /// Multiplication.
    fn ring_mul(&self, other: &Self) -> Self;
    /// Check if zero.
    fn is_zero(&self) -> bool {
        *self == Self::zero()
    }
    /// Check if one.
    fn is_one(&self) -> bool {
        *self == Self::one()
    }
}

/// A commutative ring.
pub trait CommutativeRing: Ring {}

/// A field: a commutative ring where every non-zero element has a multiplicative inverse.
pub trait Field: CommutativeRing {
    /// Multiplicative inverse (undefined for zero).
    fn field_inv(&self) -> Self;

    /// Division.
    fn field_div(&self, other: &Self) -> Self {
        self.ring_mul(&other.field_inv())
    }

    /// Characteristic of the field.
    fn characteristic() -> usize;
}

/// A Euclidean domain: a commutative ring with a division algorithm.
pub trait EuclideanDomain: CommutativeRing {
    /// The Euclidean function (degree/norm).
    fn euclidean_degree(&self) -> usize;

    /// Division with remainder: self = q * other + r where degree(r) < degree(other).
    fn div_rem(&self, other: &Self) -> (Self, Self);

    /// GCD via Euclidean algorithm.
    fn gcd(&self, other: &Self) -> Self {
        if other.is_zero() {
            return self.clone();
        }
        let (_, r) = self.div_rem(other);
        other.gcd(&r)
    }

    /// Extended GCD: returns (g, s, t) such that g = s*self + t*other.
    fn extended_gcd(&self, other: &Self) -> (Self, Self, Self) {
        if other.is_zero() {
            return (self.clone(), Self::one(), Self::zero());
        }
        let (q, r) = self.div_rem(other);
        let (g, s, t) = other.extended_gcd(&r);
        (g, t.clone(), s.ring_sub(&q.ring_mul(&t)))
    }
}

// ═══════════════════════════════════════════════════════════════════════
// FiniteField — GF(p)
// ═══════════════════════════════════════════════════════════════════════

/// An element of the finite field GF(p) for prime p.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct FiniteFieldElement {
    /// The value (0 <= value < p).
    pub value: u64,
    /// The prime modulus.
    pub modulus: u64,
}

impl FiniteFieldElement {
    /// Create a new element of GF(p).
    pub fn new(value: u64, modulus: u64) -> Self {
        FiniteFieldElement {
            value: value % modulus,
            modulus,
        }
    }

    /// The zero element of GF(p).
    pub fn gf_zero(p: u64) -> Self {
        FiniteFieldElement { value: 0, modulus: p }
    }

    /// The one element of GF(p).
    pub fn gf_one(p: u64) -> Self {
        FiniteFieldElement { value: 1, modulus: p }
    }

    /// Addition in GF(p).
    pub fn gf_add(self, other: Self) -> Self {
        FiniteFieldElement::new(self.value + other.value, self.modulus)
    }

    /// Subtraction in GF(p).
    pub fn gf_sub(self, other: Self) -> Self {
        FiniteFieldElement::new(self.value + self.modulus - other.value, self.modulus)
    }

    /// Multiplication in GF(p).
    pub fn gf_mul(self, other: Self) -> Self {
        FiniteFieldElement::new(self.value * other.value, self.modulus)
    }

    /// Negation in GF(p).
    pub fn gf_neg(self) -> Self {
        if self.value == 0 {
            self
        } else {
            FiniteFieldElement::new(self.modulus - self.value, self.modulus)
        }
    }

    /// Modular exponentiation.
    pub fn gf_pow(self, mut exp: u64) -> Self {
        let mut result = Self::gf_one(self.modulus);
        let mut base = self;
        while exp > 0 {
            if exp & 1 == 1 {
                result = result.gf_mul(base);
            }
            base = base.gf_mul(base);
            exp >>= 1;
        }
        result
    }

    /// Multiplicative inverse via Fermat's little theorem: a^(-1) = a^(p-2) mod p.
    pub fn gf_inv(self) -> Self {
        self.gf_pow(self.modulus - 2)
    }

    /// Division in GF(p).
    pub fn gf_div(self, other: Self) -> Self {
        self.gf_mul(other.gf_inv())
    }

    /// Order of this element (smallest n > 0 such that self^n = 1).
    pub fn element_order(self) -> u64 {
        if self.value == 0 { return 0; }
        let mut current = self;
        let mut n = 1u64;
        while current.value != 1 {
            current = current.gf_mul(self);
            n += 1;
        }
        n
    }

    /// Check if this is a generator (primitive element) of GF(p)*.
    pub fn is_generator(self) -> bool {
        if self.value == 0 { return false; }
        self.element_order() == self.modulus - 1
    }

    /// Find a generator of GF(p)*.
    pub fn find_generator(p: u64) -> Self {
        for g in 2..p {
            let elem = FiniteFieldElement::new(g, p);
            if elem.is_generator() {
                return elem;
            }
        }
        FiniteFieldElement::new(1, p) // fallback (only for p=2)
    }

    /// Iterate over all elements of GF(p).
    pub fn all_elements(p: u64) -> Vec<Self> {
        (0..p).map(|v| FiniteFieldElement::new(v, p)).collect()
    }

    /// The modulus (characteristic).
    pub fn characteristic(self) -> u64 {
        self.modulus
    }

    /// The order of the field.
    pub fn field_order(self) -> u64 {
        self.modulus
    }
}

impl fmt::Debug for FiniteFieldElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

impl fmt::Display for FiniteFieldElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

// Implement Ring trait for FiniteFieldElement
impl Ring for FiniteFieldElement {
    fn zero() -> Self { FiniteFieldElement { value: 0, modulus: 2 } }
    fn one() -> Self { FiniteFieldElement { value: 1, modulus: 2 } }
    fn ring_add(&self, other: &Self) -> Self { self.gf_add(*other) }
    fn ring_neg(&self) -> Self { self.gf_neg() }
    fn ring_mul(&self, other: &Self) -> Self { self.gf_mul(*other) }
}

impl CommutativeRing for FiniteFieldElement {}

impl Field for FiniteFieldElement {
    fn field_inv(&self) -> Self { self.gf_inv() }
    fn characteristic() -> usize { 0 } // depends on instance
}

/// Type alias for GF(2).
pub type GF2 = FiniteFieldElement;

/// Create a GF(2) element.
pub fn gf2(val: u64) -> GF2 {
    FiniteFieldElement::new(val, 2)
}

/// Create a GF(p) element.
pub fn gfp(val: u64, p: u64) -> FiniteFieldElement {
    FiniteFieldElement::new(val, p)
}

// ═══════════════════════════════════════════════════════════════════════
// Polynomial<F> — polynomial over a field
// ═══════════════════════════════════════════════════════════════════════

/// A polynomial with coefficients in a finite field.
/// coeffs[i] is the coefficient of x^i.
#[derive(Clone, PartialEq, Eq)]
pub struct Polynomial {
    /// Coefficients, coeffs[i] = coefficient of x^i.
    pub coeffs: Vec<FiniteFieldElement>,
    /// The field modulus.
    pub modulus: u64,
}

impl Polynomial {
    /// Create a new polynomial from coefficients (lowest degree first).
    pub fn new(coeffs: Vec<FiniteFieldElement>, modulus: u64) -> Self {
        let mut p = Polynomial { coeffs, modulus };
        p.normalize();
        p
    }

    /// Create the zero polynomial.
    pub fn zero(modulus: u64) -> Self {
        Polynomial { coeffs: Vec::new(), modulus }
    }

    /// Create a constant polynomial.
    pub fn constant(val: u64, modulus: u64) -> Self {
        let c = FiniteFieldElement::new(val, modulus);
        if c.value == 0 {
            Self::zero(modulus)
        } else {
            Polynomial::new(vec![c], modulus)
        }
    }

    /// Create the polynomial x.
    pub fn x(modulus: u64) -> Self {
        Polynomial {
            coeffs: vec![
                FiniteFieldElement::gf_zero(modulus),
                FiniteFieldElement::gf_one(modulus),
            ],
            modulus,
        }
    }

    /// Create a monomial c * x^n.
    pub fn monomial(coeff: u64, degree: usize, modulus: u64) -> Self {
        let mut coeffs = vec![FiniteFieldElement::gf_zero(modulus); degree + 1];
        coeffs[degree] = FiniteFieldElement::new(coeff, modulus);
        let mut p = Polynomial { coeffs, modulus };
        p.normalize();
        p
    }

    /// Remove trailing zero coefficients.
    fn normalize(&mut self) {
        while let Some(last) = self.coeffs.last() {
            if last.value == 0 {
                self.coeffs.pop();
            } else {
                break;
            }
        }
    }

    /// Degree of the polynomial (-1 for zero polynomial, represented as None).
    pub fn degree(&self) -> Option<usize> {
        if self.coeffs.is_empty() {
            None
        } else {
            Some(self.coeffs.len() - 1)
        }
    }

    /// Is this the zero polynomial?
    pub fn is_zero(&self) -> bool {
        self.coeffs.is_empty()
    }

    /// Leading coefficient.
    pub fn leading_coeff(&self) -> FiniteFieldElement {
        self.coeffs.last().copied().unwrap_or(FiniteFieldElement::gf_zero(self.modulus))
    }

    /// Get coefficient of x^i.
    pub fn coeff(&self, i: usize) -> FiniteFieldElement {
        if i < self.coeffs.len() {
            self.coeffs[i]
        } else {
            FiniteFieldElement::gf_zero(self.modulus)
        }
    }

    /// Evaluate the polynomial at a point.
    pub fn evaluate(&self, x: FiniteFieldElement) -> FiniteFieldElement {
        if self.is_zero() {
            return FiniteFieldElement::gf_zero(self.modulus);
        }
        // Horner's method
        let mut result = FiniteFieldElement::gf_zero(self.modulus);
        for i in (0..self.coeffs.len()).rev() {
            result = result.gf_mul(x).gf_add(self.coeffs[i]);
        }
        result
    }

    /// Polynomial addition.
    pub fn poly_add(&self, other: &Self) -> Self {
        let len = std::cmp::max(self.coeffs.len(), other.coeffs.len());
        let mut coeffs = Vec::with_capacity(len);
        for i in 0..len {
            let a = self.coeff(i);
            let b = other.coeff(i);
            coeffs.push(a.gf_add(b));
        }
        Polynomial::new(coeffs, self.modulus)
    }

    /// Polynomial subtraction.
    pub fn poly_sub(&self, other: &Self) -> Self {
        let len = std::cmp::max(self.coeffs.len(), other.coeffs.len());
        let mut coeffs = Vec::with_capacity(len);
        for i in 0..len {
            let a = self.coeff(i);
            let b = other.coeff(i);
            coeffs.push(a.gf_sub(b));
        }
        Polynomial::new(coeffs, self.modulus)
    }

    /// Polynomial multiplication.
    pub fn poly_mul(&self, other: &Self) -> Self {
        if self.is_zero() || other.is_zero() {
            return Self::zero(self.modulus);
        }
        let len = self.coeffs.len() + other.coeffs.len() - 1;
        let mut coeffs = vec![FiniteFieldElement::gf_zero(self.modulus); len];
        for i in 0..self.coeffs.len() {
            for j in 0..other.coeffs.len() {
                coeffs[i + j] = coeffs[i + j].gf_add(self.coeffs[i].gf_mul(other.coeffs[j]));
            }
        }
        Polynomial::new(coeffs, self.modulus)
    }

    /// Scalar multiplication.
    pub fn scalar_mul(&self, s: FiniteFieldElement) -> Self {
        let coeffs: Vec<_> = self.coeffs.iter().map(|c| c.gf_mul(s)).collect();
        Polynomial::new(coeffs, self.modulus)
    }

    /// Polynomial division with remainder: self = q * other + r.
    pub fn poly_div_rem(&self, other: &Self) -> (Self, Self) {

        if self.is_zero() || self.degree() < other.degree() {
            return (Self::zero(self.modulus), self.clone());
        }

        let mut remainder = self.clone();
        let d = other.degree().unwrap();
        let lc_inv = other.leading_coeff().gf_inv();
        let mut quotient_coeffs = vec![FiniteFieldElement::gf_zero(self.modulus);
            self.degree().unwrap() - d + 1];

        while !remainder.is_zero() && remainder.degree() >= other.degree() {
            let rd = remainder.degree().unwrap();
            let coeff = remainder.leading_coeff().gf_mul(lc_inv);
            let deg_diff = rd - d;
            quotient_coeffs[deg_diff] = coeff;

            let sub = other.scalar_mul(coeff);
            let shifted = Self::monomial(1, deg_diff, self.modulus).poly_mul(&sub);
            remainder = remainder.poly_sub(&shifted);
        }

        (Polynomial::new(quotient_coeffs, self.modulus), remainder)
    }

    /// Polynomial GCD.
    pub fn poly_gcd(&self, other: &Self) -> Self {
        if other.is_zero() {
            // Make monic
            if self.is_zero() { return self.clone(); }
            let lc_inv = self.leading_coeff().gf_inv();
            return self.scalar_mul(lc_inv);
        }
        let (_, r) = self.poly_div_rem(other);
        other.poly_gcd(&r)
    }

    /// Check if the polynomial is irreducible over GF(p).
    /// Uses brute force for small degrees.
    pub fn is_irreducible(&self) -> bool {
        let deg = match self.degree() {
            None => return false,
            Some(0) => return false,
            Some(1) => return true,
            Some(d) => d,
        };

        // Check if it has any roots
        for v in 0..self.modulus {
            let x = FiniteFieldElement::new(v, self.modulus);
            if self.evaluate(x).value == 0 {
                return false;
            }
        }

        // For degree <= 3, no roots implies irreducible
        if deg <= 3 {
            return true;
        }

        // For higher degrees, try all possible factor degrees
        // Check if gcd(f, x^(p^k) - x) is non-trivial for k = 1, ..., deg/2
        let x_poly = Self::x(self.modulus);
        for k in 1..=deg/2 {
            // Compute x^(p^k) mod f
            let pk = self.modulus.pow(k as u32);
            let xpk = poly_pow_mod(&x_poly, pk, self);
            let diff = xpk.poly_sub(&x_poly);
            let g = self.poly_gcd(&diff);
            if let Some(gd) = g.degree() {
                if gd > 0 {
                    return false;
                }
            }
        }

        true
    }

    /// Find all roots of the polynomial in GF(p).
    pub fn roots(&self) -> Vec<FiniteFieldElement> {
        let mut roots = Vec::new();
        for v in 0..self.modulus {
            let x = FiniteFieldElement::new(v, self.modulus);
            if self.evaluate(x).value == 0 {
                roots.push(x);
            }
        }
        roots
    }

    /// Factor the polynomial over GF(p) (brute force for small polynomials).
    pub fn factor(&self) -> Vec<(Polynomial, usize)> {
        if self.is_zero() {
            return Vec::new();
        }

        let mut f = self.clone();
        let mut factors = Vec::new();

        // Extract linear factors
        for v in 0..self.modulus {
            let x = FiniteFieldElement::new(v, self.modulus);
            let mut mult = 0;
            while f.evaluate(x) == FiniteFieldElement::gf_zero(self.modulus) {
                // Divide by (x - v)
                let linear = Polynomial::new(
                    vec![
                        FiniteFieldElement::new(self.modulus - v, self.modulus),
                        FiniteFieldElement::gf_one(self.modulus),
                    ],
                    self.modulus,
                );
                let (q, _) = f.poly_div_rem(&linear);
                f = q;
                mult += 1;
            }
            if mult > 0 {
                let linear = Polynomial::new(
                    vec![
                        FiniteFieldElement::new(self.modulus - v, self.modulus),
                        FiniteFieldElement::gf_one(self.modulus),
                    ],
                    self.modulus,
                );
                factors.push((linear, mult));
            }
        }

        // Remaining factor (if any)
        if !f.is_zero() && f.degree() > Some(0) {
            let lc_inv = f.leading_coeff().gf_inv();
            let monic = f.scalar_mul(lc_inv);
            factors.push((monic, 1));
        }

        factors
    }

    /// Make the polynomial monic.
    pub fn make_monic(&self) -> Self {
        if self.is_zero() { return self.clone(); }
        let lc_inv = self.leading_coeff().gf_inv();
        self.scalar_mul(lc_inv)
    }
}

/// Compute base^exp mod modpoly using repeated squaring.
fn poly_pow_mod(base: &Polynomial, exp: u64, modpoly: &Polynomial) -> Polynomial {
    if exp == 0 {
        return Polynomial::constant(1, base.modulus);
    }
    let mut result = Polynomial::constant(1, base.modulus);
    let mut b = base.clone();
    let mut e = exp;
    while e > 0 {
        if e & 1 == 1 {
            result = result.poly_mul(&b);
            let (_, r) = result.poly_div_rem(modpoly);
            result = r;
        }
        b = b.poly_mul(&b);
        let (_, r) = b.poly_div_rem(modpoly);
        b = r;
        e >>= 1;
    }
    result
}

impl fmt::Debug for Polynomial {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            return write!(f, "0");
        }
        let mut first = true;
        for (i, coeff) in self.coeffs.iter().enumerate().rev() {
            if coeff.value == 0 { continue; }
            if !first { write!(f, " + ")?; }
            first = false;
            match i {
                0 => write!(f, "{}", coeff.value)?,
                1 => {
                    if coeff.value == 1 {
                        write!(f, "x")?;
                    } else {
                        write!(f, "{}*x", coeff.value)?;
                    }
                }
                _ => {
                    if coeff.value == 1 {
                        write!(f, "x^{}", i)?;
                    } else {
                        write!(f, "{}*x^{}", coeff.value, i)?;
                    }
                }
            }
        }
        if first {
            write!(f, "0")?;
        }
        Ok(())
    }
}

impl fmt::Display for Polynomial {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

// ═══════════════════════════════════════════════════════════════════════
// FieldExtension — GF(p^n)
// ═══════════════════════════════════════════════════════════════════════

/// An element of the field extension GF(p^n) = GF(p)[x] / (f(x)).
#[derive(Clone, PartialEq, Eq)]
pub struct FieldExtensionElement {
    /// The polynomial representative.
    pub poly: Polynomial,
    /// The irreducible polynomial defining the extension.
    pub modpoly: Polynomial,
    /// The prime.
    pub p: u64,
    /// The extension degree.
    pub n: usize,
}

impl FieldExtensionElement {
    /// Create a new element of GF(p^n).
    pub fn new(poly: Polynomial, modpoly: Polynomial) -> Self {
        let p = poly.modulus;
        let n = modpoly.degree().unwrap();
        let (_, reduced) = poly.poly_div_rem(&modpoly);
        FieldExtensionElement { poly: reduced, modpoly, p, n }
    }

    /// Zero element of GF(p^n).
    pub fn ext_zero(modpoly: &Polynomial) -> Self {
        let p = modpoly.modulus;
        Self::new(Polynomial::zero(p), modpoly.clone())
    }

    /// One element of GF(p^n).
    pub fn ext_one(modpoly: &Polynomial) -> Self {
        let p = modpoly.modulus;
        Self::new(Polynomial::constant(1, p), modpoly.clone())
    }

    /// Addition.
    pub fn ext_add(&self, other: &Self) -> Self {
        Self::new(self.poly.poly_add(&other.poly), self.modpoly.clone())
    }

    /// Subtraction.
    pub fn ext_sub(&self, other: &Self) -> Self {
        Self::new(self.poly.poly_sub(&other.poly), self.modpoly.clone())
    }

    /// Multiplication.
    pub fn ext_mul(&self, other: &Self) -> Self {
        Self::new(self.poly.poly_mul(&other.poly), self.modpoly.clone())
    }

    /// Inverse (via extended GCD).
    pub fn ext_inv(&self) -> Self {
        // Find s such that s * self.poly ≡ 1 mod modpoly
        // Using extended GCD on polynomials
        let (g, s, _) = poly_extended_gcd(&self.poly, &self.modpoly);
        // g should be constant (since modpoly is irreducible)
        let lc_inv = g.leading_coeff().gf_inv();
        Self::new(s.scalar_mul(lc_inv), self.modpoly.clone())
    }

    /// Frobenius endomorphism: x ↦ x^p.
    pub fn frobenius(&self) -> Self {
        let p = self.p;
        let mut result = Self::ext_one(&self.modpoly);
        let mut base = self.clone();
        let mut exp = p;
        while exp > 0 {
            if exp & 1 == 1 {
                result = result.ext_mul(&base);
            }
            base = base.ext_mul(&base);
            exp >>= 1;
        }
        result
    }

    /// The order of the field GF(p^n).
    pub fn field_order(&self) -> u64 {
        self.p.pow(self.n as u32)
    }
}

/// Extended GCD for polynomials.
fn poly_extended_gcd(a: &Polynomial, b: &Polynomial) -> (Polynomial, Polynomial, Polynomial) {
    if b.is_zero() {
        return (a.clone(), Polynomial::constant(1, a.modulus), Polynomial::zero(a.modulus));
    }
    let (q, r) = a.poly_div_rem(b);
    let (g, s, t) = poly_extended_gcd(b, &r);
    (g, t.clone(), s.poly_sub(&q.poly_mul(&t)))
}

impl fmt::Debug for FieldExtensionElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.poly)
    }
}

// ═══════════════════════════════════════════════════════════════════════
// FieldMatrix — matrix over a finite field
// ═══════════════════════════════════════════════════════════════════════

/// A matrix over a finite field GF(p).
#[derive(Clone, PartialEq, Eq)]
pub struct FieldMatrix {
    /// Number of rows.
    pub rows: usize,
    /// Number of columns.
    pub cols: usize,
    /// Entries in row-major order.
    pub data: Vec<FiniteFieldElement>,
    /// The field modulus.
    pub modulus: u64,
}

impl FieldMatrix {
    /// Create a zero matrix.
    pub fn zeros(rows: usize, cols: usize, modulus: u64) -> Self {
        FieldMatrix {
            rows,
            cols,
            data: vec![FiniteFieldElement::gf_zero(modulus); rows * cols],
            modulus,
        }
    }

    /// Create an identity matrix.
    pub fn identity(n: usize, modulus: u64) -> Self {
        let mut m = Self::zeros(n, n, modulus);
        for i in 0..n {
            m.set(i, i, FiniteFieldElement::gf_one(modulus));
        }
        m
    }

    /// Create from a 2D array of values.
    pub fn from_values(rows: &[Vec<u64>], modulus: u64) -> Self {
        let r = rows.len();
        let c = rows[0].len();
        let mut data = Vec::with_capacity(r * c);
        for row in rows {
            for &v in row {
                data.push(FiniteFieldElement::new(v, modulus));
            }
        }
        FieldMatrix { rows: r, cols: c, data, modulus }
    }

    /// Get element.
    pub fn get(&self, i: usize, j: usize) -> FiniteFieldElement {
        self.data[i * self.cols + j]
    }

    /// Set element.
    pub fn set(&mut self, i: usize, j: usize, val: FiniteFieldElement) {
        self.data[i * self.cols + j] = val;
    }

    /// Matrix addition.
    pub fn mat_add(&self, other: &Self) -> Self {
        let data: Vec<_> = self.data.iter().zip(&other.data)
            .map(|(a, b)| a.gf_add(*b))
            .collect();
        FieldMatrix { rows: self.rows, cols: self.cols, data, modulus: self.modulus }
    }

    /// Matrix subtraction.
    pub fn mat_sub(&self, other: &Self) -> Self {
        let data: Vec<_> = self.data.iter().zip(&other.data)
            .map(|(a, b)| a.gf_sub(*b))
            .collect();
        FieldMatrix { rows: self.rows, cols: self.cols, data, modulus: self.modulus }
    }

    /// Matrix multiplication.
    pub fn mat_mul(&self, other: &Self) -> Self {
        let mut result = Self::zeros(self.rows, other.cols, self.modulus);
        for i in 0..self.rows {
            for k in 0..self.cols {
                let a = self.get(i, k);
                for j in 0..other.cols {
                    let val = result.get(i, j).gf_add(a.gf_mul(other.get(k, j)));
                    result.set(i, j, val);
                }
            }
        }
        result
    }

    /// Scalar multiplication.
    pub fn scalar_mul(&self, s: FiniteFieldElement) -> Self {
        let data: Vec<_> = self.data.iter().map(|x| x.gf_mul(s)).collect();
        FieldMatrix { rows: self.rows, cols: self.cols, data, modulus: self.modulus }
    }

    /// Transpose.
    pub fn transpose(&self) -> Self {
        let mut result = Self::zeros(self.cols, self.rows, self.modulus);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.set(j, i, self.get(i, j));
            }
        }
        result
    }

    /// Trace.
    pub fn trace(&self) -> FiniteFieldElement {
        let mut t = FiniteFieldElement::gf_zero(self.modulus);
        for i in 0..self.rows {
            t = t.gf_add(self.get(i, i));
        }
        t
    }

    /// Row echelon form via Gaussian elimination. Returns (echelon form, rank, det_sign).
    pub fn row_echelon(&self) -> (Self, usize, FiniteFieldElement) {
        let mut m = self.clone();
        let mut rank = 0;
        let mut det = FiniteFieldElement::gf_one(self.modulus);
        let mut pivot_col = 0;

        for row in 0..m.rows {
            if pivot_col >= m.cols { break; }

            // Find pivot
            let mut pivot_row = None;
            for r in row..m.rows {
                if m.get(r, pivot_col).value != 0 {
                    pivot_row = Some(r);
                    break;
                }
            }

            let pivot_row = match pivot_row {
                Some(r) => r,
                None => {
                    pivot_col += 1;
                    continue;
                }
            };

            // Swap rows
            if pivot_row != row {
                for j in 0..m.cols {
                    let tmp = m.get(row, j);
                    m.set(row, j, m.get(pivot_row, j));
                    m.set(pivot_row, j, tmp);
                }
                det = det.gf_neg();
            }

            // Scale pivot row
            let pivot = m.get(row, pivot_col);
            det = det.gf_mul(pivot);
            let pivot_inv = pivot.gf_inv();
            for j in 0..m.cols {
                m.set(row, j, m.get(row, j).gf_mul(pivot_inv));
            }

            // Eliminate below
            for r in (row + 1)..m.rows {
                let factor = m.get(r, pivot_col);
                if factor.value != 0 {
                    for j in 0..m.cols {
                        let val = m.get(r, j).gf_sub(factor.gf_mul(m.get(row, j)));
                        m.set(r, j, val);
                    }
                }
            }

            rank += 1;
            pivot_col += 1;
        }

        (m, rank, det)
    }

    /// Reduced row echelon form.
    pub fn rref(&self) -> (Self, usize) {
        let (mut m, rank, _) = self.row_echelon();

        // Back-substitute
        for row in (0..rank).rev() {
            // Find pivot column
            let mut pivot_col = 0;
            while pivot_col < m.cols && m.get(row, pivot_col).value == 0 {
                pivot_col += 1;
            }
            if pivot_col >= m.cols { continue; }

            // Eliminate above
            for r in 0..row {
                let factor = m.get(r, pivot_col);
                if factor.value != 0 {
                    for j in 0..m.cols {
                        let val = m.get(r, j).gf_sub(factor.gf_mul(m.get(row, j)));
                        m.set(r, j, val);
                    }
                }
            }
        }

        (m, rank)
    }

    /// Determinant (square matrices only).
    pub fn determinant(&self) -> FiniteFieldElement {
        let (_, rank, det) = self.row_echelon();
        if rank < self.rows {
            FiniteFieldElement::gf_zero(self.modulus)
        } else {
            det
        }
    }

    /// Rank.
    pub fn rank(&self) -> usize {
        let (_, rank, _) = self.row_echelon();
        rank
    }

    /// Inverse (square matrices only).
    pub fn inverse(&self) -> Option<Self> {
        let n = self.rows;

        // Augment with identity
        let mut aug = Self::zeros(n, 2 * n, self.modulus);
        for i in 0..n {
            for j in 0..n {
                aug.set(i, j, self.get(i, j));
            }
            aug.set(i, n + i, FiniteFieldElement::gf_one(self.modulus));
        }

        let (aug_rref, rank) = aug.rref();
        if rank < n { return None; }

        let mut result = Self::zeros(n, n, self.modulus);
        for i in 0..n {
            for j in 0..n {
                result.set(i, j, aug_rref.get(i, n + j));
            }
        }
        Some(result)
    }

    /// Null space (kernel): vectors v such that M*v = 0.
    pub fn null_space(&self) -> Vec<Vec<FiniteFieldElement>> {
        let (rref_mat, rank) = self.rref();
        let n = self.cols;
        let mut result = Vec::new();

        // Find pivot and free columns
        let mut pivot_cols = Vec::new();
        let mut free_cols = Vec::new();
        let mut pivot_row = 0;
        for j in 0..n {
            if pivot_row < self.rows && rref_mat.get(pivot_row, j).value == 1 {
                pivot_cols.push(j);
                pivot_row += 1;
            } else {
                free_cols.push(j);
            }
        }

        // For each free variable, create a null space vector
        for &free_col in &free_cols {
            let mut v = vec![FiniteFieldElement::gf_zero(self.modulus); self.cols];
            v[free_col] = FiniteFieldElement::gf_one(self.modulus);
            for (i, &pcol) in pivot_cols.iter().enumerate() {
                v[pcol] = rref_mat.get(i, free_col).gf_neg();
            }
            result.push(v);
        }

        result
    }

    /// Column space: a basis for the column space.
    pub fn column_space(&self) -> Vec<Vec<FiniteFieldElement>> {
        let (_, rank) = self.rref();
        let mut basis = Vec::new();

        // Find pivot columns in RREF, return original columns
        let (rref_mat, _) = self.rref();
        let mut pivot_row = 0;
        for j in 0..self.cols {
            if pivot_row < self.rows && rref_mat.get(pivot_row, j).value != 0 {
                let col: Vec<_> = (0..self.rows).map(|i| self.get(i, j)).collect();
                basis.push(col);
                pivot_row += 1;
            }
        }

        basis
    }
}

impl fmt::Debug for FieldMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for i in 0..self.rows {
            for j in 0..self.cols {
            }
        }
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════
// VectorSpace — finite-dimensional vector space over a finite field
// ═══════════════════════════════════════════════════════════════════════

/// A finite-dimensional vector space over GF(p).
#[derive(Debug, Clone)]
pub struct VectorSpace {
    /// Basis vectors (stored as rows).
    pub basis: Vec<Vec<FiniteFieldElement>>,
    /// Dimension of the ambient space.
    pub ambient_dim: usize,
    /// The field modulus.
    pub modulus: u64,
}

impl VectorSpace {
    /// Create a vector space from a spanning set.
    pub fn from_spanning_set(vectors: Vec<Vec<FiniteFieldElement>>, modulus: u64) -> Self {
        if vectors.is_empty() {
        }
        let ambient_dim = vectors[0].len();
        // Compute a basis via row reduction
        let mat = FieldMatrix {
            rows: vectors.len(),
            cols: ambient_dim,
            data: vectors.iter().flatten().cloned().collect(),
            modulus,
        };
        let (rref_mat, rank) = mat.rref();

        let mut basis = Vec::new();
        for i in 0..rank {
            let row: Vec<_> = (0..ambient_dim).map(|j| rref_mat.get(i, j)).collect();
            basis.push(row);
        }

        VectorSpace { basis, ambient_dim, modulus }
    }

    /// Create the full space GF(p)^n.
    pub fn full_space(n: usize, modulus: u64) -> Self {
        let mut basis = Vec::new();
        for i in 0..n {
            let mut v = vec![FiniteFieldElement::gf_zero(modulus); n];
            v[i] = FiniteFieldElement::gf_one(modulus);
            basis.push(v);
        }
        VectorSpace { basis, ambient_dim: n, modulus }
    }

    /// Create the zero space.
    pub fn zero_space(ambient_dim: usize, modulus: u64) -> Self {
        VectorSpace { basis: Vec::new(), ambient_dim, modulus }
    }

    /// Dimension of the vector space.
    pub fn dimension(&self) -> usize {
        self.basis.len()
    }

    /// Check if a vector is in the space.
    pub fn contains(&self, v: &[FiniteFieldElement]) -> bool {
        let mut extended = self.basis.clone();
        extended.push(v.to_vec());
        let new_space = Self::from_spanning_set(extended, self.modulus);
        new_space.dimension() == self.dimension()
    }

    /// Intersection of two vector spaces.
    pub fn intersection(&self, other: &Self) -> Self {
        // Use the fact that V ∩ W = kernel of [V; W] restricted appropriately
        let total_vecs: Vec<_> = self.basis.iter().chain(other.basis.iter()).cloned().collect();
        if total_vecs.is_empty() {
            return Self::zero_space(self.ambient_dim, self.modulus);
        }

        let mat = FieldMatrix {
            rows: total_vecs.len(),
            cols: self.ambient_dim,
            data: total_vecs.iter().flatten().cloned().collect(),
            modulus: self.modulus,
        };

        let null = mat.transpose().null_space();
        // The intersection is spanned by combinations of V basis using null space
        let v_dim = self.basis.len();
        let mut int_vecs = Vec::new();
        for null_vec in &null {
            let mut v = vec![FiniteFieldElement::gf_zero(self.modulus); self.ambient_dim];
            for (i, &coeff) in null_vec.iter().enumerate().take(v_dim) {
                for j in 0..self.ambient_dim {
                    v[j] = v[j].gf_add(coeff.gf_mul(self.basis[i][j]));
                }
            }
            // Check it's not zero
            if v.iter().any(|x| x.value != 0) {
                int_vecs.push(v);
            }
        }

        Self::from_spanning_set(int_vecs, self.modulus)
    }

    /// Sum of two vector spaces.
    pub fn sum(&self, other: &Self) -> Self {
        let mut vecs = self.basis.clone();
        vecs.extend(other.basis.iter().cloned());
        Self::from_spanning_set(vecs, self.modulus)
    }

    /// Check linear independence of a set of vectors.
    pub fn are_linearly_independent(vectors: &[Vec<FiniteFieldElement>], modulus: u64) -> bool {
        if vectors.is_empty() { return true; }
        let n = vectors[0].len();
        let mat = FieldMatrix {
            rows: vectors.len(),
            cols: n,
            data: vectors.iter().flatten().cloned().collect(),
            modulus,
        };
        mat.rank() == vectors.len()
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── GF(2) tests ──────────────────────────────────────────────────

    #[test]
    fn test_gf2_arithmetic() {
        let zero = gf2(0);
        let one = gf2(1);

    }

    #[test]
    fn test_gf2_inverse() {
        let one = gf2(1);
    }

    // ── GF(7) tests ──────────────────────────────────────────────────

    #[test]
    fn test_gf7_arithmetic() {
        let a = gfp(3, 7);
        let b = gfp(5, 7);

    }

    #[test]
    fn test_gf7_inverse() {
        for v in 1..7u64 {
            let a = gfp(v, 7);
            let inv = a.gf_inv();
        }
    }

    #[test]
    fn test_gf7_pow() {
        let a = gfp(3, 7);
        // 3^6 ≡ 1 mod 7 (Fermat's little theorem)
    }

    #[test]
    fn test_gf7_generator() {
        let g = FiniteFieldElement::find_generator(7);
    }

    #[test]
    fn test_gf7_all_elements() {
        let elems = FiniteFieldElement::all_elements(7);
    }

    // ── Polynomial tests ─────────────────────────────────────────────

    #[test]
    fn test_poly_add() {
        // Over GF(7): (x + 1) + (2x + 3) = 3x + 4
        let a = Polynomial::new(vec![gfp(1,7), gfp(1,7)], 7);
        let b = Polynomial::new(vec![gfp(3,7), gfp(2,7)], 7);
        let c = a.poly_add(&b);
    }

    #[test]
    fn test_poly_mul() {
        // Over GF(7): (x + 1)(x + 2) = x^2 + 3x + 2
        let a = Polynomial::new(vec![gfp(1,7), gfp(1,7)], 7);
        let b = Polynomial::new(vec![gfp(2,7), gfp(1,7)], 7);
        let c = a.poly_mul(&b);
    }

    #[test]
    fn test_poly_evaluate() {
        // x^2 + 1 evaluated at x=2 in GF(7): 4+1 = 5
    }

    #[test]
    fn test_poly_div_rem() {
        // Over GF(7): (x^2 + 3x + 2) / (x + 1) = (x + 2) remainder 0
        let f = Polynomial::new(vec![gfp(2,7), gfp(3,7), gfp(1,7)], 7);
        let g = Polynomial::new(vec![gfp(1,7), gfp(1,7)], 7);
        let (q, r) = f.poly_div_rem(&g);
    }

    #[test]
    fn test_poly_gcd() {
        // GCD of (x^2 - 1) and (x - 1) over GF(7)
        // x^2 - 1 = (x-1)(x+1), so gcd should be x-1 (monic)
        let f = Polynomial::new(vec![gfp(6,7), gfp(0,7), gfp(1,7)], 7); // x^2 - 1
        let g = Polynomial::new(vec![gfp(6,7), gfp(1,7)], 7); // x - 1
        let gcd = f.poly_gcd(&g);
    }

    #[test]
    fn test_poly_irreducible() {
        // x^2 + x + 1 is irreducible over GF(2)

        // x^2 + 1 = (x+1)^2 over GF(2), so not irreducible
    }

    #[test]
    fn test_poly_roots() {
        // (x-1)(x-2) = x^2 - 3x + 2 over GF(7)
        let p = Polynomial::new(vec![gfp(2,7), gfp(4,7), gfp(1,7)], 7); // x^2 - 3x + 2
        let roots = p.roots();
        let root_vals: HashSet<u64> = roots.iter().map(|r| r.value).collect();
    }

    #[test]
    fn test_poly_factor() {
        // x^2 + 3x + 2 = (x+1)(x+2) over GF(7)
        let p = Polynomial::new(vec![gfp(2,7), gfp(3,7), gfp(1,7)], 7);
        let factors = p.factor();
    }

    // ── Field extension tests ────────────────────────────────────────

    #[test]
    fn test_field_extension_gf4() {
        // GF(4) = GF(2)[x] / (x^2 + x + 1)
        let modpoly = Polynomial::new(vec![gfp(1,2), gfp(1,2), gfp(1,2)], 2); // x^2 + x + 1
        let zero = FieldExtensionElement::ext_zero(&modpoly);
        let one = FieldExtensionElement::ext_one(&modpoly);

        let sum = zero.ext_add(&one);

        // x * x = x + 1 (since x^2 ≡ x + 1 mod x^2+x+1)
        let x = FieldExtensionElement::new(Polynomial::x(2), modpoly.clone());
        let x2 = x.ext_mul(&x);
    }

    #[test]
    fn test_field_extension_inverse() {
        let modpoly = Polynomial::new(vec![gfp(1,2), gfp(1,2), gfp(1,2)], 2);
        let x = FieldExtensionElement::new(Polynomial::x(2), modpoly.clone());
        let x_inv = x.ext_inv();
        let product = x.ext_mul(&x_inv);
    }

    // ── Matrix over finite field tests ───────────────────────────────

    #[test]
    fn test_field_matrix_identity() {
        let id = FieldMatrix::identity(3, 7);
    }

    #[test]
    fn test_field_matrix_multiply() {
        let a = FieldMatrix::from_values(&[vec![1,2], vec![3,4]], 7);
        let b = FieldMatrix::from_values(&[vec![5,6], vec![0,1]], 7);
        let c = a.mat_mul(&b);
    }

    #[test]
    fn test_field_matrix_determinant() {
        // det = 1*4 - 2*3 = -2 ≡ 5 mod 7
    }

    #[test]
    fn test_field_matrix_inverse() {
        let m = FieldMatrix::from_values(&[vec![1,2], vec![3,4]], 7);
        let inv = m.inverse().unwrap();
        let product = m.mat_mul(&inv);
        let id = FieldMatrix::identity(2, 7);
    }

    #[test]
    fn test_field_matrix_rank() {
    }

    #[test]
    fn test_field_matrix_null_space() {
        let m = FieldMatrix::from_values(&[vec![1,2,3], vec![2,4,6]], 7); // rank-1 matrix
        let null = m.null_space();
    }

    // ── Vector space tests ───────────────────────────────────────────

    #[test]
    fn test_vector_space_dimension() {
    }

    #[test]
    fn test_vector_space_contains() {
    }

    #[test]
    fn test_vector_space_sum() {
        let s1 = VectorSpace::from_spanning_set(
            vec![vec![gfp(1,7), gfp(0,7)]],
            7,
        );
        let s2 = VectorSpace::from_spanning_set(
            vec![vec![gfp(0,7), gfp(1,7)]],
            7,
        );
        let sum = s1.sum(&s2);
    }

    #[test]
    fn test_linear_independence() {

    }
}
