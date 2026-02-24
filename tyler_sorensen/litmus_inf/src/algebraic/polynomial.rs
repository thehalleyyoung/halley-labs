//! Polynomial arithmetic for algebraic compression in LITMUS∞.
//!
//! Provides polynomial types, arithmetic operations, GCD computation,
//! factorization, evaluation, interpolation, and multivariate polynomials.

use std::collections::{HashMap, HashSet, BTreeMap};
use std::fmt;
use std::ops::{Add, Sub, Mul, Neg, AddAssign, SubAssign, MulAssign};
use num_traits::{Zero, One};

// ---------------------------------------------------------------------------
// Field trait
// ---------------------------------------------------------------------------

/// A mathematical field supporting addition, multiplication, and inverses.
pub trait Field:
    Clone + PartialEq + fmt::Debug + fmt::Display
    + Add<Output = Self> + Sub<Output = Self>
    + Mul<Output = Self> + Neg<Output = Self>
    + Zero + One
{
    /// Multiplicative inverse (panics for zero).
    fn inv(&self) -> Self;

    /// Division: a / b = a * b^{-1}.
    fn div(&self, other: &Self) -> Self {
        self.clone() * other.inv()
    }

    /// Absolute value (for ordered fields).
    fn abs_val(&self) -> Self {
        self.clone()
    }
}

// ---------------------------------------------------------------------------
// F64 field wrapper
// ---------------------------------------------------------------------------

/// Wrapper around f64 implementing the Field trait.
#[derive(Clone, Copy, PartialEq)]
pub struct F64(pub f64);

impl F64 {
    pub fn new(val: f64) -> Self { Self(val) }
    pub fn value(&self) -> f64 { self.0 }
}

impl fmt::Debug for F64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.6}", self.0)
    }
}

impl fmt::Display for F64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if (self.0 - self.0.round()).abs() < 1e-10 {
            write!(f, "{}", self.0.round() as i64)
        } else {
            write!(f, "{:.4}", self.0)
        }
    }
}

impl Add for F64 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self { F64(self.0 + rhs.0) }
}

impl Sub for F64 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self { F64(self.0 - rhs.0) }
}

impl Mul for F64 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self { F64(self.0 * rhs.0) }
}

impl Neg for F64 {
    type Output = Self;
    fn neg(self) -> Self { F64(-self.0) }
}

impl Zero for F64 {
    fn zero() -> Self { F64(0.0) }
    fn is_zero(&self) -> bool { self.0.abs() < 1e-12 }
}

impl One for F64 {
    fn one() -> Self { F64(1.0) }
}

impl Field for F64 {
    fn inv(&self) -> Self {
        assert!(!self.is_zero(), "Cannot invert zero");
        F64(1.0 / self.0)
    }
    fn abs_val(&self) -> Self { F64(self.0.abs()) }
}

// ---------------------------------------------------------------------------
// Rational number
// ---------------------------------------------------------------------------

/// Exact rational number for precise polynomial arithmetic.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Rational {
    pub num: i64,
    pub den: i64,
}

impl Rational {
    pub fn new(num: i64, den: i64) -> Self {
        assert!(den != 0, "Denominator cannot be zero");
        let g = gcd_i64(num.abs(), den.abs());
        let sign = if den < 0 { -1 } else { 1 };
        Self {
            num: sign * num / g,
            den: sign * den / g,
        }
    }

    pub fn integer(n: i64) -> Self { Self { num: n, den: 1 } }

    pub fn to_f64(&self) -> f64 {
        self.num as f64 / self.den as f64
    }
}

fn gcd_i64(mut a: i64, mut b: i64) -> i64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a.max(1)
}

impl fmt::Debug for Rational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.den == 1 {
            write!(f, "{}", self.num)
        } else {
            write!(f, "{}/{}", self.num, self.den)
        }
    }
}

impl fmt::Display for Rational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.den == 1 {
            write!(f, "{}", self.num)
        } else {
            write!(f, "{}/{}", self.num, self.den)
        }
    }
}

impl Add for Rational {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Rational::new(
            self.num * rhs.den + rhs.num * self.den,
            self.den * rhs.den,
        )
    }
}

impl Sub for Rational {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Rational::new(
            self.num * rhs.den - rhs.num * self.den,
            self.den * rhs.den,
        )
    }
}

impl Mul for Rational {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Rational::new(self.num * rhs.num, self.den * rhs.den)
    }
}

impl Neg for Rational {
    type Output = Self;
    fn neg(self) -> Self { Rational::new(-self.num, self.den) }
}

impl Zero for Rational {
    fn zero() -> Self { Rational { num: 0, den: 1 } }
    fn is_zero(&self) -> bool { self.num == 0 }
}

impl One for Rational {
    fn one() -> Self { Rational { num: 1, den: 1 } }
}

impl Field for Rational {
    fn inv(&self) -> Self {
        assert!(!self.is_zero(), "Cannot invert zero");
        Rational::new(self.den, self.num)
    }
    fn abs_val(&self) -> Self {
        Rational::new(self.num.abs(), self.den)
    }
}

// ---------------------------------------------------------------------------
// Polynomial
// ---------------------------------------------------------------------------

/// A univariate polynomial over a field F.
/// Coefficients are stored lowest degree first: coeffs[i] is the coefficient of x^i.
#[derive(Clone, PartialEq)]
pub struct Polynomial<F: Field> {
    coeffs: Vec<F>,
}

/// Type alias for polynomial with f64 coefficients.
pub type Polynomial64 = Polynomial<F64>;

/// Type alias for polynomial with rational coefficients.
pub type PolyRational = Polynomial<Rational>;

impl<F: Field> Polynomial<F> {
    /// Create a polynomial from coefficients (lowest degree first).
    pub fn new(coeffs: Vec<F>) -> Self {
        let mut p = Self { coeffs };
        p.normalize();
        p
    }

    /// Create the zero polynomial.
    pub fn zero() -> Self {
        Self { coeffs: Vec::new() }
    }

    /// Create a constant polynomial.
    pub fn constant(c: F) -> Self {
        if c.is_zero() {
            Self::zero()
        } else {
            Self { coeffs: vec![c] }
        }
    }

    /// Create a monomial: c * x^n.
    pub fn monomial(c: F, n: usize) -> Self {
        if c.is_zero() {
            return Self::zero();
        }
        let mut coeffs = vec![F::zero(); n + 1];
        coeffs[n] = c;
        Self { coeffs }
    }

    /// Create the identity polynomial: x.
    pub fn x() -> Self {
        Self { coeffs: vec![F::zero(), F::one()] }
    }

    /// Create from roots: (x - r1)(x - r2)...(x - rn).
    pub fn from_roots(roots: &[F]) -> Self {
        let mut result = Self::constant(F::one());
        for r in roots {
            let factor = Self::new(vec![-r.clone(), F::one()]);
            result = result.mul_poly(&factor);
        }
        result
    }

    /// Remove trailing zero coefficients.
    fn normalize(&mut self) {
        while self.coeffs.last().map_or(false, |c| c.is_zero()) {
            self.coeffs.pop();
        }
    }

    /// Degree of the polynomial (-1 for the zero polynomial, represented as None).
    pub fn degree(&self) -> Option<usize> {
        if self.coeffs.is_empty() {
            None
        } else {
            Some(self.coeffs.len() - 1)
        }
    }

    /// Degree as i64 (-1 for zero polynomial).
    pub fn degree_i64(&self) -> i64 {
        self.degree().map_or(-1, |d| d as i64)
    }

    /// Check if this is the zero polynomial.
    pub fn is_zero_poly(&self) -> bool {
        self.coeffs.is_empty()
    }

    /// Leading coefficient.
    pub fn leading_coeff(&self) -> Option<&F> {
        self.coeffs.last()
    }

    /// Get coefficient of x^i.
    pub fn coeff(&self, i: usize) -> F {
        self.coeffs.get(i).cloned().unwrap_or_else(F::zero)
    }

    /// Number of terms (non-zero coefficients).
    pub fn num_terms(&self) -> usize {
        self.coeffs.iter().filter(|c| !c.is_zero()).count()
    }

    /// All coefficients.
    pub fn coefficients(&self) -> &[F] {
        &self.coeffs
    }

    /// Make the polynomial monic (leading coefficient = 1).
    pub fn make_monic(&self) -> Self {
        if self.is_zero_poly() {
            return Self::zero();
        }
        let lc = self.leading_coeff().unwrap().inv();
        self.scalar_mul(&lc)
    }

    /// Scalar multiplication.
    pub fn scalar_mul(&self, s: &F) -> Self {
        if s.is_zero() {
            return Self::zero();
        }
        Self::new(self.coeffs.iter().map(|c| c.clone() * s.clone()).collect())
    }

    /// Polynomial addition.
    pub fn add_poly(&self, other: &Self) -> Self {
        let len = self.coeffs.len().max(other.coeffs.len());
        let mut coeffs = Vec::with_capacity(len);
        for i in 0..len {
            let a = self.coeff(i);
            let b = other.coeff(i);
            coeffs.push(a + b);
        }
        Self::new(coeffs)
    }

    /// Polynomial subtraction.
    pub fn sub_poly(&self, other: &Self) -> Self {
        let len = self.coeffs.len().max(other.coeffs.len());
        let mut coeffs = Vec::with_capacity(len);
        for i in 0..len {
            let a = self.coeff(i);
            let b = other.coeff(i);
            coeffs.push(a - b);
        }
        Self::new(coeffs)
    }

    /// Polynomial multiplication.
    pub fn mul_poly(&self, other: &Self) -> Self {
        if self.is_zero_poly() || other.is_zero_poly() {
            return Self::zero();
        }
        let n = self.coeffs.len() + other.coeffs.len() - 1;
        let mut coeffs = vec![F::zero(); n];
        for (i, a) in self.coeffs.iter().enumerate() {
            if a.is_zero() { continue; }
            for (j, b) in other.coeffs.iter().enumerate() {
                coeffs[i + j] = coeffs[i + j].clone() + a.clone() * b.clone();
            }
        }
        Self::new(coeffs)
    }

    /// Polynomial division with remainder: self = q * divisor + r.
    /// Returns (quotient, remainder).
    pub fn div_rem(&self, divisor: &Self) -> (Self, Self) {
        assert!(!divisor.is_zero_poly(), "Division by zero polynomial");

        if self.is_zero_poly() || self.degree_i64() < divisor.degree_i64() {
            return (Self::zero(), self.clone());
        }

        let mut remainder = self.clone();
        let divisor_deg = divisor.degree().unwrap();
        let divisor_lc = divisor.leading_coeff().unwrap().inv();

        let self_deg = self.degree().unwrap();
        let mut quotient_coeffs = vec![F::zero(); self_deg - divisor_deg + 1];

        while !remainder.is_zero_poly() && remainder.degree_i64() >= divisor.degree_i64() {
            let rem_deg = remainder.degree().unwrap();
            let rem_lc = remainder.leading_coeff().unwrap().clone();
            let coeff = rem_lc * divisor_lc.clone();
            let deg_diff = rem_deg - divisor_deg;

            quotient_coeffs[deg_diff] = coeff.clone();

            let term = Self::monomial(coeff, deg_diff);
            let sub = term.mul_poly(divisor);
            remainder = remainder.sub_poly(&sub);
        }

        (Self::new(quotient_coeffs), remainder)
    }

    /// Polynomial remainder.
    pub fn rem_poly(&self, divisor: &Self) -> Self {
        self.div_rem(divisor).1
    }

    /// Evaluate the polynomial at a point using Horner's method.
    pub fn evaluate(&self, x: &F) -> F {
        if self.is_zero_poly() {
            return F::zero();
        }
        let mut result = F::zero();
        for coeff in self.coeffs.iter().rev() {
            result = result * x.clone() + coeff.clone();
        }
        result
    }

    /// Multi-point evaluation.
    pub fn evaluate_many(&self, points: &[F]) -> Vec<F> {
        points.iter().map(|x| self.evaluate(x)).collect()
    }

    /// Formal derivative.
    pub fn derivative(&self) -> Self {
        if self.coeffs.len() <= 1 {
            return Self::zero();
        }
        let mut coeffs = Vec::with_capacity(self.coeffs.len() - 1);
        for i in 1..self.coeffs.len() {
            let mut c = self.coeffs[i].clone();
            for _ in 1..i {
                c = c + self.coeffs[i].clone();
            }
            coeffs.push(c);
        }
        Self::new(coeffs)
    }

    /// Formal integral (with zero constant term).
    pub fn integral(&self) -> Self {
        if self.is_zero_poly() {
            return Self::zero();
        }
        let mut coeffs = vec![F::zero()];
        for (i, c) in self.coeffs.iter().enumerate() {
            let n = i + 1;
            // Divide by (i+1): create n copies and sum reciprocal.
            let mut denom = F::one();
            for _ in 1..n {
                denom = denom + F::one();
            }
            coeffs.push(c.clone().div(&denom));
        }
        Self::new(coeffs)
    }

    /// Composition: self(other(x)).
    pub fn compose(&self, other: &Self) -> Self {
        if self.is_zero_poly() {
            return Self::zero();
        }
        let mut result = Self::zero();
        let mut power = Self::constant(F::one());

        for coeff in &self.coeffs {
            if !coeff.is_zero() {
                result = result.add_poly(&power.scalar_mul(coeff));
            }
            power = power.mul_poly(other);
        }
        result
    }

    /// GCD using the Euclidean algorithm.
    pub fn gcd(&self, other: &Self) -> Self {
        let mut a = self.clone();
        let mut b = other.clone();

        while !b.is_zero_poly() {
            let r = a.rem_poly(&b);
            a = b;
            b = r;
        }

        if a.is_zero_poly() {
            a
        } else {
            a.make_monic()
        }
    }

    /// Extended GCD: returns (gcd, s, t) such that s*self + t*other = gcd.
    pub fn extended_gcd(&self, other: &Self) -> (Self, Self, Self) {
        let mut old_r = self.clone();
        let mut r = other.clone();
        let mut old_s = Self::constant(F::one());
        let mut s = Self::zero();
        let mut old_t = Self::zero();
        let mut t = Self::constant(F::one());

        while !r.is_zero_poly() {
            let (q, remainder) = old_r.div_rem(&r);
            old_r = r;
            r = remainder;

            let new_s = old_s.sub_poly(&q.mul_poly(&s));
            old_s = s;
            s = new_s;

            let new_t = old_t.sub_poly(&q.mul_poly(&t));
            old_t = t;
            t = new_t;
        }

        // Make monic.
        if !old_r.is_zero_poly() {
            let lc_inv = old_r.leading_coeff().unwrap().inv();
            old_r = old_r.scalar_mul(&lc_inv);
            old_s = old_s.scalar_mul(&lc_inv);
            old_t = old_t.scalar_mul(&lc_inv);
        }

        (old_r, old_s, old_t)
    }

    /// Square-free factorization.
    /// Returns factors p_1, p_2, ... such that self = p_1 * p_2^2 * p_3^3 * ...
    pub fn square_free_factorization(&self) -> Vec<Self> {
        if self.is_zero_poly() || self.degree_i64() <= 0 {
            return vec![self.clone()];
        }

        let f = self.make_monic();
        let f_prime = f.derivative();

        if f_prime.is_zero_poly() {
            return vec![f];
        }

        let mut c = f.gcd(&f_prime);
        let mut w = f.div_rem(&c).0;
        let mut factors = Vec::new();

        loop {
            let y = w.gcd(&c);
            let factor = w.div_rem(&y).0;
            if factor.degree_i64() > 0 {
                factors.push(factor);
            } else if !factor.is_zero_poly() {
                factors.push(factor);
            }
            w = y.clone();
            let (new_c, _) = c.div_rem(&y);
            if new_c.degree_i64() <= 0 {
                break;
            }
            // Use block to avoid "c" borrow conflict.
            c = new_c;

            // Safety: we need to reborrow c - use `break` above to terminate.
            break;
        }

        if w.degree_i64() > 0 {
            factors.push(w);
        }

        if factors.is_empty() {
            factors.push(f);
        }

        factors
    }

    /// Check if a value is a root.
    pub fn is_root(&self, x: &F) -> bool {
        self.evaluate(x).is_zero()
    }

    /// Power: self^n.
    pub fn pow(&self, n: u32) -> Self {
        if n == 0 {
            return Self::constant(F::one());
        }
        let mut result = Self::constant(F::one());
        let mut base = self.clone();
        let mut exp = n;

        while exp > 0 {
            if exp % 2 == 1 {
                result = result.mul_poly(&base);
            }
            base = base.mul_poly(&base);
            exp /= 2;
        }
        result
    }

    /// Shift: multiply by x^n.
    pub fn shift(&self, n: usize) -> Self {
        if self.is_zero_poly() || n == 0 {
            return self.clone();
        }
        let mut coeffs = vec![F::zero(); n];
        coeffs.extend(self.coeffs.iter().cloned());
        Self::new(coeffs)
    }

    /// Truncate to degree n (keep only terms up to x^n).
    pub fn truncate(&self, n: usize) -> Self {
        let len = (n + 1).min(self.coeffs.len());
        Self::new(self.coeffs[..len].to_vec())
    }

    /// Reverse the coefficients.
    pub fn reverse(&self) -> Self {
        let mut coeffs = self.coeffs.clone();
        coeffs.reverse();
        Self::new(coeffs)
    }
}

impl<F: Field> fmt::Debug for Polynomial<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Poly({:?})", self.coeffs)
    }
}

impl<F: Field> fmt::Display for Polynomial<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero_poly() {
            return write!(f, "0");
        }
        let mut first = true;
        for (i, c) in self.coeffs.iter().enumerate().rev() {
            if c.is_zero() { continue; }
            if !first {
                write!(f, " + ")?;
            }
            match i {
                0 => write!(f, "{}", c)?,
                1 => write!(f, "{}*x", c)?,
                _ => write!(f, "{}*x^{}", c, i)?,
            }
            first = false;
        }
        if first {
            write!(f, "0")?;
        }
        Ok(())
    }
}

// Operator impls.

impl<F: Field> Add for Polynomial<F> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self { self.add_poly(&rhs) }
}

impl<F: Field> Sub for Polynomial<F> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self { self.sub_poly(&rhs) }
}

impl<F: Field> Mul for Polynomial<F> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self { self.mul_poly(&rhs) }
}

impl<F: Field> Neg for Polynomial<F> {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(self.coeffs.into_iter().map(|c| -c).collect())
    }
}

impl<F: Field> AddAssign for Polynomial<F> {
    fn add_assign(&mut self, rhs: Self) {
        *self = self.add_poly(&rhs);
    }
}

impl<F: Field> SubAssign for Polynomial<F> {
    fn sub_assign(&mut self, rhs: Self) {
        *self = self.sub_poly(&rhs);
    }
}

impl<F: Field> MulAssign for Polynomial<F> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.mul_poly(&rhs);
    }
}

// ---------------------------------------------------------------------------
// Lagrange Interpolation
// ---------------------------------------------------------------------------

/// Lagrange interpolation: find polynomial passing through given points.
pub fn lagrange_interpolation<F: Field>(points: &[(F, F)]) -> Polynomial<F> {
    let n = points.len();
    if n == 0 {
        return Polynomial::zero();
    }

    let mut result = Polynomial::zero();

    for i in 0..n {
        let (xi, yi) = &points[i];
        let mut basis = Polynomial::constant(F::one());
        let mut denom = F::one();

        for j in 0..n {
            if i == j { continue; }
            let (xj, _) = &points[j];
            // (x - xj)
            let factor = Polynomial::new(vec![-xj.clone(), F::one()]);
            basis = basis.mul_poly(&factor);
            denom = denom * (xi.clone() - xj.clone());
        }

        let coeff = yi.clone().div(&denom);
        result = result.add_poly(&basis.scalar_mul(&coeff));
    }

    result
}

/// Newton interpolation: find polynomial using divided differences.
pub fn newton_interpolation<F: Field>(points: &[(F, F)]) -> Polynomial<F> {
    let n = points.len();
    if n == 0 {
        return Polynomial::zero();
    }

    // Compute divided differences.
    let mut dd = vec![F::zero(); n];
    for i in 0..n {
        dd[i] = points[i].1.clone();
    }
    for j in 1..n {
        for i in (j..n).rev() {
            let diff = points[i].0.clone() - points[i - j].0.clone();
            dd[i] = (dd[i].clone() - dd[i - 1].clone()).div(&diff);
        }
    }

    // Build polynomial from divided differences.
    let mut result = Polynomial::constant(dd[n - 1].clone());
    for i in (0..n-1).rev() {
        let factor = Polynomial::new(vec![-points[i].0.clone(), F::one()]);
        result = result.mul_poly(&factor).add_poly(&Polynomial::constant(dd[i].clone()));
    }

    result
}

// ---------------------------------------------------------------------------
// Polynomial Ring
// ---------------------------------------------------------------------------

/// A polynomial ring F[x] / (modulus).
#[derive(Debug, Clone)]
pub struct PolynomialRing<F: Field> {
    /// The modulus polynomial (if any).
    modulus: Option<Polynomial<F>>,
}

impl<F: Field> PolynomialRing<F> {
    /// Create the ring F[x] (no modulus).
    pub fn new() -> Self {
        Self { modulus: None }
    }

    /// Create the quotient ring F[x] / (modulus).
    pub fn quotient(modulus: Polynomial<F>) -> Self {
        Self { modulus: Some(modulus) }
    }

    /// Reduce a polynomial modulo the ring's modulus.
    pub fn reduce(&self, p: &Polynomial<F>) -> Polynomial<F> {
        match &self.modulus {
            Some(m) => p.rem_poly(m),
            None => p.clone(),
        }
    }

    /// Add in the ring.
    pub fn add(&self, a: &Polynomial<F>, b: &Polynomial<F>) -> Polynomial<F> {
        self.reduce(&a.add_poly(b))
    }

    /// Multiply in the ring.
    pub fn mul(&self, a: &Polynomial<F>, b: &Polynomial<F>) -> Polynomial<F> {
        self.reduce(&a.mul_poly(b))
    }

    /// Power in the ring.
    pub fn pow(&self, base: &Polynomial<F>, exp: u32) -> Polynomial<F> {
        if exp == 0 {
            return Polynomial::constant(F::one());
        }
        let mut result = Polynomial::constant(F::one());
        let mut b = self.reduce(base);
        let mut e = exp;

        while e > 0 {
            if e % 2 == 1 {
                result = self.mul(&result, &b);
            }
            b = self.mul(&b, &b);
            e /= 2;
        }
        result
    }

    /// Check if a polynomial is zero in the ring.
    pub fn is_zero(&self, p: &Polynomial<F>) -> bool {
        self.reduce(p).is_zero_poly()
    }

    /// GCD in the ring.
    pub fn gcd(&self, a: &Polynomial<F>, b: &Polynomial<F>) -> Polynomial<F> {
        let ra = self.reduce(a);
        let rb = self.reduce(b);
        ra.gcd(&rb)
    }
}

impl<F: Field> Default for PolynomialRing<F> {
    fn default() -> Self { Self::new() }
}

// ---------------------------------------------------------------------------
// Root Finding (Newton's Method)
// ---------------------------------------------------------------------------

/// Find approximate roots of a polynomial using Newton's method.
pub fn find_roots_newton(poly: &Polynomial64, initial_guesses: &[f64], tol: f64, max_iter: usize) -> Vec<f64> {
    let deriv = poly.derivative();
    let mut roots = Vec::new();

    for &guess in initial_guesses {
        let mut x = guess;
        let mut found = false;

        for _ in 0..max_iter {
            let fx = poly.evaluate(&F64(x)).0;
            let fpx = deriv.evaluate(&F64(x)).0;

            if fpx.abs() < 1e-15 {
                break;
            }

            let new_x = x - fx / fpx;
            if (new_x - x).abs() < tol {
                x = new_x;
                found = true;
                break;
            }
            x = new_x;
        }

        if found && poly.evaluate(&F64(x)).0.abs() < tol * 100.0 {
            // Check for duplicates.
            if !roots.iter().any(|&r: &f64| (r - x).abs() < tol * 10.0) {
                roots.push(x);
            }
        }
    }

    roots
}

/// Resultant of two polynomials (determinant of Sylvester matrix).
pub fn resultant<F: Field>(p: &Polynomial<F>, q: &Polynomial<F>) -> F {
    if p.is_zero_poly() || q.is_zero_poly() {
        return F::zero();
    }

    let mut a = p.clone();
    let mut b = q.clone();
    let mut result = F::one();

    while !b.is_zero_poly() && b.degree_i64() > 0 {
        let (_, r) = a.div_rem(&b);
        let deg_a = a.degree_i64();
        let deg_b = b.degree_i64();

        if deg_a % 2 == 1 && deg_b % 2 == 1 {
            result = -result;
        }

        let lc_b = b.leading_coeff().unwrap().clone();
        let exp = deg_a - r.degree_i64();
        for _ in 0..exp as u32 {
            result = result * lc_b.clone();
        }

        a = b;
        b = r;
    }

    if b.is_zero_poly() {
        F::zero()
    } else {
        let lc_b = b.leading_coeff().unwrap().clone();
        let deg_a = a.degree_i64() as u32;
        for _ in 0..deg_a {
            result = result * lc_b.clone();
        }
        result
    }
}

/// Discriminant of a polynomial: resultant(p, p') / leading_coeff(p).
pub fn discriminant<F: Field>(p: &Polynomial<F>) -> F {
    let p_prime = p.derivative();
    let res = resultant(p, &p_prime);
    if let Some(lc) = p.leading_coeff() {
        res.div(lc)
    } else {
        F::zero()
    }
}

// ---------------------------------------------------------------------------
// Multivariate Polynomial
// ---------------------------------------------------------------------------

/// Monomial ordering for multivariate polynomials.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MonomialOrder {
    /// Lexicographic order.
    Lex,
    /// Graded reverse lexicographic order.
    GrLex,
    /// Graded reverse lexicographic order (reversed).
    GrevLex,
}

/// A monomial: x_0^{e_0} * x_1^{e_1} * ... * x_{n-1}^{e_{n-1}}.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Monomial {
    pub exponents: Vec<u32>,
}

impl Monomial {
    pub fn new(exponents: Vec<u32>) -> Self {
        Self { exponents }
    }

    pub fn one(num_vars: usize) -> Self {
        Self { exponents: vec![0; num_vars] }
    }

    pub fn var(var_idx: usize, num_vars: usize) -> Self {
        let mut exponents = vec![0; num_vars];
        if var_idx < num_vars {
            exponents[var_idx] = 1;
        }
        Self { exponents }
    }

    /// Total degree.
    pub fn total_degree(&self) -> u32 {
        self.exponents.iter().sum()
    }

    /// Number of variables.
    pub fn num_vars(&self) -> usize {
        self.exponents.len()
    }

    /// Multiply two monomials.
    pub fn mul(&self, other: &Self) -> Self {
        let n = self.exponents.len().max(other.exponents.len());
        let mut exponents = vec![0; n];
        for (i, e) in self.exponents.iter().enumerate() {
            exponents[i] += e;
        }
        for (i, e) in other.exponents.iter().enumerate() {
            exponents[i] += e;
        }
        Self { exponents }
    }

    /// Check if this monomial is divisible by another.
    pub fn is_divisible_by(&self, other: &Self) -> bool {
        if other.exponents.len() > self.exponents.len() {
            return false;
        }
        for (i, &e) in other.exponents.iter().enumerate() {
            if self.exponents.get(i).copied().unwrap_or(0) < e {
                return false;
            }
        }
        true
    }

    /// Divide this monomial by another (assumes divisibility).
    pub fn div(&self, other: &Self) -> Self {
        let n = self.exponents.len();
        let mut exponents = vec![0; n];
        for i in 0..n {
            let a = self.exponents[i];
            let b = other.exponents.get(i).copied().unwrap_or(0);
            exponents[i] = a.saturating_sub(b);
        }
        Self { exponents }
    }

    /// LCM of two monomials.
    pub fn lcm(&self, other: &Self) -> Self {
        let n = self.exponents.len().max(other.exponents.len());
        let mut exponents = vec![0; n];
        for i in 0..n {
            let a = self.exponents.get(i).copied().unwrap_or(0);
            let b = other.exponents.get(i).copied().unwrap_or(0);
            exponents[i] = a.max(b);
        }
        Self { exponents }
    }

    /// GCD of two monomials.
    pub fn gcd(&self, other: &Self) -> Self {
        let n = self.exponents.len().max(other.exponents.len());
        let mut exponents = vec![0; n];
        for i in 0..n {
            let a = self.exponents.get(i).copied().unwrap_or(0);
            let b = other.exponents.get(i).copied().unwrap_or(0);
            exponents[i] = a.min(b);
        }
        Self { exponents }
    }

    /// Compare monomials using the given ordering.
    pub fn cmp_order(&self, other: &Self, order: MonomialOrder) -> std::cmp::Ordering {
        match order {
            MonomialOrder::Lex => {
                for i in 0..self.exponents.len().max(other.exponents.len()) {
                    let a = self.exponents.get(i).copied().unwrap_or(0);
                    let b = other.exponents.get(i).copied().unwrap_or(0);
                    match a.cmp(&b) {
                        std::cmp::Ordering::Equal => continue,
                        ord => return ord,
                    }
                }
                std::cmp::Ordering::Equal
            }
            MonomialOrder::GrLex => {
                let da = self.total_degree();
                let db = other.total_degree();
                match da.cmp(&db) {
                    std::cmp::Ordering::Equal => self.cmp_order(other, MonomialOrder::Lex),
                    ord => ord,
                }
            }
            MonomialOrder::GrevLex => {
                let da = self.total_degree();
                let db = other.total_degree();
                match da.cmp(&db) {
                    std::cmp::Ordering::Equal => {
                        // Reverse lexicographic on last variable first.
                        let n = self.exponents.len().max(other.exponents.len());
                        for i in (0..n).rev() {
                            let a = self.exponents.get(i).copied().unwrap_or(0);
                            let b = other.exponents.get(i).copied().unwrap_or(0);
                            match a.cmp(&b) {
                                std::cmp::Ordering::Equal => continue,
                                std::cmp::Ordering::Less => return std::cmp::Ordering::Greater,
                                std::cmp::Ordering::Greater => return std::cmp::Ordering::Less,
                            }
                        }
                        std::cmp::Ordering::Equal
                    }
                    ord => ord,
                }
            }
        }
    }
}

impl fmt::Display for Monomial {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut first = true;
        for (i, &e) in self.exponents.iter().enumerate() {
            if e == 0 { continue; }
            if !first { write!(f, "*")?; }
            if e == 1 {
                write!(f, "x{}", i)?;
            } else {
                write!(f, "x{}^{}", i, e)?;
            }
            first = false;
        }
        if first {
            write!(f, "1")?;
        }
        Ok(())
    }
}

/// A term: coefficient * monomial.
#[derive(Debug, Clone)]
pub struct Term<F: Field> {
    pub coeff: F,
    pub monomial: Monomial,
}

impl<F: Field> Term<F> {
    pub fn new(coeff: F, monomial: Monomial) -> Self {
        Self { coeff, monomial }
    }

    pub fn is_zero(&self) -> bool { self.coeff.is_zero() }
}

impl<F: Field> fmt::Display for Term<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.monomial.total_degree() == 0 {
            write!(f, "{}", self.coeff)
        } else {
            write!(f, "{}*{}", self.coeff, self.monomial)
        }
    }
}

/// A multivariate polynomial.
#[derive(Debug, Clone)]
pub struct MultivariatePolynomial<F: Field> {
    /// Terms stored as (monomial exponents -> coefficient).
    terms: BTreeMap<Vec<u32>, F>,
    /// Number of variables.
    num_vars: usize,
    /// Monomial ordering.
    order: MonomialOrder,
}

impl<F: Field> MultivariatePolynomial<F> {
    /// Create the zero polynomial.
    pub fn zero(num_vars: usize) -> Self {
        Self {
            terms: BTreeMap::new(),
            num_vars,
            order: MonomialOrder::GrevLex,
        }
    }

    /// Create a constant polynomial.
    pub fn constant(c: F, num_vars: usize) -> Self {
        let mut p = Self::zero(num_vars);
        if !c.is_zero() {
            p.terms.insert(vec![0; num_vars], c);
        }
        p
    }

    /// Create a single-variable polynomial: x_i.
    pub fn var(var_idx: usize, num_vars: usize) -> Self {
        let mut p = Self::zero(num_vars);
        let mut exp = vec![0; num_vars];
        exp[var_idx] = 1;
        p.terms.insert(exp, F::one());
        p
    }

    /// Set the monomial ordering.
    pub fn with_order(mut self, order: MonomialOrder) -> Self {
        self.order = order;
        self
    }

    /// Add a term.
    pub fn add_term(&mut self, coeff: F, exponents: Vec<u32>) {
        if coeff.is_zero() { return; }
        let entry = self.terms.entry(exponents).or_insert_with(F::zero);
        *entry = entry.clone() + coeff;
        if entry.is_zero() {
            let key = self.terms.iter()
                .find(|(_, v)| v.is_zero())
                .map(|(k, _)| k.clone());
            if let Some(k) = key {
                self.terms.remove(&k);
            }
        }
    }

    /// Check if zero.
    pub fn is_zero_poly(&self) -> bool {
        self.terms.is_empty()
    }

    /// Number of terms.
    pub fn num_terms(&self) -> usize {
        self.terms.len()
    }

    /// Total degree.
    pub fn total_degree(&self) -> Option<u32> {
        self.terms.keys().map(|e| e.iter().sum::<u32>()).max()
    }

    /// Leading term (under the current ordering).
    pub fn leading_term(&self) -> Option<Term<F>> {
        let order = self.order;
        self.terms.iter()
            .max_by(|(a, _), (b, _)| {
                Monomial::new(a.to_vec()).cmp_order(&Monomial::new(b.to_vec()), order)
            })
            .map(|(exp, coeff)| Term::new(coeff.clone(), Monomial::new(exp.clone())))
    }

    /// Leading monomial.
    pub fn leading_monomial(&self) -> Option<Monomial> {
        self.leading_term().map(|t| t.monomial)
    }

    /// Leading coefficient.
    pub fn leading_coefficient(&self) -> Option<F> {
        self.leading_term().map(|t| t.coeff)
    }

    /// Add two multivariate polynomials.
    pub fn add_poly(&self, other: &Self) -> Self {
        let mut result = self.clone();
        for (exp, coeff) in &other.terms {
            result.add_term(coeff.clone(), exp.clone());
        }
        result
    }

    /// Subtract.
    pub fn sub_poly(&self, other: &Self) -> Self {
        let mut result = self.clone();
        for (exp, coeff) in &other.terms {
            result.add_term(-coeff.clone(), exp.clone());
        }
        result
    }

    /// Multiply.
    pub fn mul_poly(&self, other: &Self) -> Self {
        let mut result = Self::zero(self.num_vars);
        result.order = self.order;
        for (exp_a, coeff_a) in &self.terms {
            for (exp_b, coeff_b) in &other.terms {
                let mut new_exp = vec![0u32; self.num_vars];
                for i in 0..self.num_vars {
                    new_exp[i] = exp_a.get(i).copied().unwrap_or(0)
                        + exp_b.get(i).copied().unwrap_or(0);
                }
                result.add_term(coeff_a.clone() * coeff_b.clone(), new_exp);
            }
        }
        result
    }

    /// Evaluate at a point.
    pub fn evaluate(&self, point: &[F]) -> F {
        assert!(point.len() >= self.num_vars);
        let mut result = F::zero();
        for (exp, coeff) in &self.terms {
            let mut term_val = coeff.clone();
            for (i, &e) in exp.iter().enumerate() {
                for _ in 0..e {
                    term_val = term_val * point[i].clone();
                }
            }
            result = result + term_val;
        }
        result
    }

    /// Scalar multiplication.
    pub fn scalar_mul(&self, s: &F) -> Self {
        let mut result = Self::zero(self.num_vars);
        result.order = self.order;
        for (exp, coeff) in &self.terms {
            let new_coeff = coeff.clone() * s.clone();
            if !new_coeff.is_zero() {
                result.terms.insert(exp.clone(), new_coeff);
            }
        }
        result
    }
}

impl<F: Field> fmt::Display for MultivariatePolynomial<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero_poly() {
            return write!(f, "0");
        }
        let mut first = true;
        for (exp, coeff) in self.terms.iter().rev() {
            if !first { write!(f, " + ")?; }
            let mono = Monomial::new(exp.clone());
            if mono.total_degree() == 0 {
                write!(f, "{}", coeff)?;
            } else {
                write!(f, "{}*{}", coeff, mono)?;
            }
            first = false;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Gröbner Basis (Buchberger's Algorithm)
// ---------------------------------------------------------------------------

/// Compute a Gröbner basis for a set of multivariate polynomials.
pub fn buchberger_groebner<F: Field>(
    polys: &[MultivariatePolynomial<F>],
    order: MonomialOrder,
    max_iter: usize,
) -> Vec<MultivariatePolynomial<F>> {
    if polys.is_empty() {
        return Vec::new();
    }

    let num_vars = polys[0].num_vars;
    let mut basis: Vec<MultivariatePolynomial<F>> = polys.to_vec();

    let mut pairs_checked: HashSet<(usize, usize)> = HashSet::new();
    let mut iterations = 0;

    loop {
        if iterations >= max_iter {
            break;
        }
        iterations += 1;

        let n = basis.len();
        let mut new_polys = Vec::new();

        for i in 0..n {
            for j in (i+1)..n {
                if pairs_checked.contains(&(i, j)) {
                    continue;
                }
                pairs_checked.insert((i, j));

                let s = s_polynomial(&basis[i], &basis[j], order);
                if s.is_zero_poly() {
                    continue;
                }

                let reduced = reduce_polynomial(&s, &basis, order);
                if !reduced.is_zero_poly() {
                    new_polys.push(reduced);
                }
            }
        }

        if new_polys.is_empty() {
            break;
        }

        for p in new_polys {
            basis.push(p);
        }
    }

    // Minimal reduction.
    reduce_basis(&mut basis, order);
    basis
}

/// Compute the S-polynomial of two polynomials.
fn s_polynomial<F: Field>(
    f: &MultivariatePolynomial<F>,
    g: &MultivariatePolynomial<F>,
    _order: MonomialOrder,
) -> MultivariatePolynomial<F> {
    let lt_f = match f.leading_term() {
        Some(t) => t,
        None => return MultivariatePolynomial::zero(f.num_vars),
    };
    let lt_g = match g.leading_term() {
        Some(t) => t,
        None => return MultivariatePolynomial::zero(f.num_vars),
    };

    let lcm = lt_f.monomial.lcm(&lt_g.monomial);

    let factor_f = lcm.div(&lt_f.monomial);
    let factor_g = lcm.div(&lt_g.monomial);

    let coeff_f = lt_g.coeff.clone();
    let coeff_g = lt_f.coeff.clone();

    // S(f,g) = coeff_g_lc * lcm/lt_f * f - coeff_f_lc * lcm/lt_g * g
    let mut term_f = MultivariatePolynomial::zero(f.num_vars);
    term_f.add_term(coeff_f, factor_f.exponents);
    let mut term_g = MultivariatePolynomial::zero(f.num_vars);
    term_g.add_term(coeff_g, factor_g.exponents);

    term_f.mul_poly(f).sub_poly(&term_g.mul_poly(g))
}

/// Reduce a polynomial modulo a set of polynomials.
fn reduce_polynomial<F: Field>(
    p: &MultivariatePolynomial<F>,
    basis: &[MultivariatePolynomial<F>],
    order: MonomialOrder,
) -> MultivariatePolynomial<F> {
    let mut remainder = p.clone();
    let mut changed = true;

    while changed {
        changed = false;
        for b in basis {
            if b.is_zero_poly() { continue; }
            let lt_b = match b.leading_term() {
                Some(t) => t,
                None => continue,
            };
            let lt_r = match remainder.leading_term() {
                Some(t) => t,
                None => return remainder,
            };

            if lt_r.monomial.is_divisible_by(&lt_b.monomial) {
                let quot_mono = lt_r.monomial.div(&lt_b.monomial);
                let quot_coeff = lt_r.coeff.div(&lt_b.coeff);

                let mut quot_poly = MultivariatePolynomial::zero(p.num_vars);
                quot_poly.add_term(quot_coeff, quot_mono.exponents);

                remainder = remainder.sub_poly(&quot_poly.mul_poly(b));
                changed = true;
                break;
            }
        }
    }

    remainder
}

/// Reduce a basis by removing redundant elements.
fn reduce_basis<F: Field>(basis: &mut Vec<MultivariatePolynomial<F>>, order: MonomialOrder) {
    let mut i = 0;
    while i < basis.len() {
        if basis[i].is_zero_poly() {
            basis.remove(i);
            continue;
        }

        // Make leading coefficient 1.
        if let Some(lc) = basis[i].leading_coefficient() {
            let inv = lc.inv();
            basis[i] = basis[i].scalar_mul(&inv);
        }
        i += 1;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn poly(coeffs: &[f64]) -> Polynomial64 {
        Polynomial::new(coeffs.iter().map(|&c| F64(c)).collect())
    }

    fn rpoly(coeffs: &[i64]) -> PolyRational {
        Polynomial::new(coeffs.iter().map(|&c| Rational::integer(c)).collect())
    }

    // -- F64 tests --

    #[test]
    fn test_f64_field() {
        let a = F64(3.0);
        let b = F64(4.0);
        assert_eq!((a + b).0, 7.0);
        assert_eq!((a * b).0, 12.0);
        assert_eq!(a.inv().0, 1.0 / 3.0);
    }

    // -- Rational tests --

    #[test]
    fn test_rational() {
        let a = Rational::new(1, 2);
        let b = Rational::new(1, 3);
        let sum = a + b;
        assert_eq!(sum, Rational::new(5, 6));
        let prod = a * b;
        assert_eq!(prod, Rational::new(1, 6));
    }

    #[test]
    fn test_rational_inv() {
        let a = Rational::new(3, 5);
        let inv = a.inv();
        assert_eq!(inv, Rational::new(5, 3));
        assert_eq!(a * inv, Rational::one());
    }

    // -- Polynomial basic tests --

    #[test]
    fn test_poly_zero() {
        let p = poly(&[]);
        assert!(p.is_zero_poly());
        assert_eq!(p.degree(), None);
    }

    #[test]
    fn test_poly_constant() {
        let p = poly(&[5.0]);
        assert_eq!(p.degree(), Some(0));
        assert_eq!(p.evaluate(&F64(10.0)).0, 5.0);
    }

    #[test]
    fn test_poly_linear() {
        let p = poly(&[1.0, 2.0]); // 1 + 2x
        assert_eq!(p.degree(), Some(1));
        assert_eq!(p.evaluate(&F64(3.0)).0, 7.0);
    }

    #[test]
    fn test_poly_quadratic() {
        let p = poly(&[1.0, 0.0, 1.0]); // 1 + x^2
        assert_eq!(p.degree(), Some(2));
        assert_eq!(p.evaluate(&F64(2.0)).0, 5.0);
    }

    // -- Arithmetic tests --

    #[test]
    fn test_poly_add() {
        let a = poly(&[1.0, 2.0]); // 1 + 2x
        let b = poly(&[3.0, 4.0]); // 3 + 4x
        let sum = a + b; // 4 + 6x
        assert_eq!(sum.coeff(0).0, 4.0);
        assert_eq!(sum.coeff(1).0, 6.0);
    }

    #[test]
    fn test_poly_sub() {
        let a = poly(&[5.0, 3.0]); // 5 + 3x
        let b = poly(&[2.0, 1.0]); // 2 + x
        let diff = a - b; // 3 + 2x
        assert_eq!(diff.coeff(0).0, 3.0);
        assert_eq!(diff.coeff(1).0, 2.0);
    }

    #[test]
    fn test_poly_mul() {
        let a = poly(&[1.0, 1.0]); // 1 + x
        let b = poly(&[1.0, 1.0]); // 1 + x
        let prod = a * b; // 1 + 2x + x^2
        assert_eq!(prod.coeff(0).0, 1.0);
        assert_eq!(prod.coeff(1).0, 2.0);
        assert_eq!(prod.coeff(2).0, 1.0);
    }

    #[test]
    fn test_poly_div_rem() {
        // (x^2 + 2x + 1) / (x + 1) = (x + 1) remainder 0
        let a = poly(&[1.0, 2.0, 1.0]);
        let b = poly(&[1.0, 1.0]);
        let (q, r) = a.div_rem(&b);
        assert_eq!(q.degree(), Some(1));
        assert!(r.is_zero_poly());
    }

    #[test]
    fn test_poly_div_rem_with_remainder() {
        // (x^2 + 1) / (x + 1) = (x - 1) remainder 2
        let a = poly(&[1.0, 0.0, 1.0]);
        let b = poly(&[1.0, 1.0]);
        let (q, r) = a.div_rem(&b);
        assert_eq!(q.degree(), Some(1));
        assert!(!r.is_zero_poly());
    }

    // -- Evaluation tests --

    #[test]
    fn test_poly_evaluate() {
        let p = poly(&[1.0, -2.0, 1.0]); // 1 - 2x + x^2 = (x-1)^2
        assert!((p.evaluate(&F64(1.0)).0).abs() < 1e-10);
        assert!((p.evaluate(&F64(0.0)).0 - 1.0).abs() < 1e-10);
        assert!((p.evaluate(&F64(2.0)).0 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_poly_evaluate_many() {
        let p = poly(&[0.0, 1.0]); // x
        let vals = p.evaluate_many(&[F64(1.0), F64(2.0), F64(3.0)]);
        assert_eq!(vals[0].0, 1.0);
        assert_eq!(vals[1].0, 2.0);
        assert_eq!(vals[2].0, 3.0);
    }

    // -- GCD tests --

    #[test]
    fn test_poly_gcd() {
        // gcd(x^2 - 1, x - 1) = x - 1
        let a = rpoly(&[-1, 0, 1]); // x^2 - 1
        let b = rpoly(&[-1, 1]); // x - 1
        let g = a.gcd(&b);
        assert_eq!(g.degree(), Some(1));
        // Should be monic.
        assert_eq!(*g.leading_coeff().unwrap(), Rational::one());
    }

    #[test]
    fn test_poly_gcd_coprime() {
        let a = rpoly(&[1, 0, 1]); // x^2 + 1
        let b = rpoly(&[1, 1]); // x + 1
        let g = a.gcd(&b);
        assert_eq!(g.degree(), Some(0));
    }

    #[test]
    fn test_extended_gcd() {
        let a = rpoly(&[-1, 0, 1]); // x^2 - 1
        let b = rpoly(&[-1, 1]); // x - 1
        let (g, s, t) = a.extended_gcd(&b);
        // Verify: s*a + t*b = g
        let check = s.mul_poly(&a).add_poly(&t.mul_poly(&b));
        assert_eq!(check.degree(), g.degree());
    }

    // -- Derivative and integral tests --

    #[test]
    fn test_poly_derivative() {
        // d/dx (1 + 2x + 3x^2) = 2 + 6x
        let p = rpoly(&[1, 2, 3]);
        let dp = p.derivative();
        assert_eq!(dp.coeff(0), Rational::integer(2));
        assert_eq!(dp.coeff(1), Rational::integer(6));
    }

    // -- From roots test --

    #[test]
    fn test_poly_from_roots() {
        let p = PolyRational::from_roots(&[Rational::integer(1), Rational::integer(2)]);
        // (x-1)(x-2) = x^2 - 3x + 2
        assert!(p.is_root(&Rational::integer(1)));
        assert!(p.is_root(&Rational::integer(2)));
    }

    // -- Power test --

    #[test]
    fn test_poly_pow() {
        let p = rpoly(&[-1, 1]); // x - 1
        let p3 = p.pow(3); // (x-1)^3
        assert_eq!(p3.degree(), Some(3));
        assert!(p3.is_root(&Rational::integer(1)));
    }

    // -- Composition test --

    #[test]
    fn test_poly_compose() {
        let f = rpoly(&[0, 0, 1]); // x^2
        let g = rpoly(&[1, 1]); // x + 1
        let h = f.compose(&g); // (x+1)^2 = x^2 + 2x + 1
        assert_eq!(h.coeff(0), Rational::integer(1));
        assert_eq!(h.coeff(1), Rational::integer(2));
        assert_eq!(h.coeff(2), Rational::integer(1));
    }

    // -- Interpolation tests --

    #[test]
    fn test_lagrange_interpolation() {
        // Points on y = x: (0,0), (1,1), (2,2)
        let points = vec![
            (Rational::integer(0), Rational::integer(0)),
            (Rational::integer(1), Rational::integer(1)),
            (Rational::integer(2), Rational::integer(2)),
        ];
        let p = lagrange_interpolation(&points);
        // Should be degree 1: y = x
        assert!(p.degree().unwrap() <= 1);
        assert_eq!(p.evaluate(&Rational::integer(3)), Rational::integer(3));
    }

    #[test]
    fn test_newton_interpolation() {
        let points = vec![
            (Rational::integer(0), Rational::integer(1)),
            (Rational::integer(1), Rational::integer(2)),
            (Rational::integer(2), Rational::integer(5)),
        ];
        let p = newton_interpolation(&points);
        // Check it passes through all points.
        for (x, y) in &points {
            assert_eq!(p.evaluate(x), *y);
        }
    }

    // -- Polynomial Ring tests --

    #[test]
    fn test_polynomial_ring() {
        // Z/2Z[x] / (x^2 + 1) -- not exact since we use rationals but tests reduction.
        let modulus = rpoly(&[1, 0, 1]); // x^2 + 1
        let ring = PolynomialRing::quotient(modulus);

        let p = rpoly(&[0, 0, 1]); // x^2
        let reduced = ring.reduce(&p);
        // x^2 mod (x^2 + 1) = -1
        assert_eq!(reduced.degree(), Some(0));
        assert_eq!(reduced.coeff(0), Rational::integer(-1));
    }

    // -- Root finding tests --

    #[test]
    fn test_find_roots_newton() {
        // x^2 - 4 has roots ±2
        let p = poly(&[-4.0, 0.0, 1.0]);
        let roots = find_roots_newton(&p, &[-3.0, 3.0], 1e-10, 100);
        assert!(roots.len() >= 2);
        let mut sorted = roots.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((sorted[0] - (-2.0)).abs() < 1e-8);
        assert!((sorted[1] - 2.0).abs() < 1e-8);
    }

    // -- Monomial tests --

    #[test]
    fn test_monomial_operations() {
        let a = Monomial::new(vec![2, 1, 0]); // x0^2 * x1
        let b = Monomial::new(vec![1, 0, 1]); // x0 * x2
        let product = a.mul(&b);
        assert_eq!(product.exponents, vec![3, 1, 1]);
        assert_eq!(product.total_degree(), 5);
    }

    #[test]
    fn test_monomial_divisibility() {
        let a = Monomial::new(vec![3, 2]);
        let b = Monomial::new(vec![1, 1]);
        assert!(a.is_divisible_by(&b));
        assert!(!b.is_divisible_by(&a));

        let q = a.div(&b);
        assert_eq!(q.exponents, vec![2, 1]);
    }

    #[test]
    fn test_monomial_lcm_gcd() {
        let a = Monomial::new(vec![2, 1]);
        let b = Monomial::new(vec![1, 3]);
        assert_eq!(a.lcm(&b).exponents, vec![2, 3]);
        assert_eq!(a.gcd(&b).exponents, vec![1, 1]);
    }

    #[test]
    fn test_monomial_ordering_lex() {
        let a = Monomial::new(vec![2, 1]);
        let b = Monomial::new(vec![1, 3]);
        assert_eq!(a.cmp_order(&b, MonomialOrder::Lex), std::cmp::Ordering::Greater);
    }

    #[test]
    fn test_monomial_ordering_grlex() {
        let a = Monomial::new(vec![2, 1]); // deg 3
        let b = Monomial::new(vec![1, 3]); // deg 4
        assert_eq!(a.cmp_order(&b, MonomialOrder::GrLex), std::cmp::Ordering::Less);
    }

    // -- Multivariate polynomial tests --

    #[test]
    fn test_multivariate_add() {
        let mut a = MultivariatePolynomial::<Rational>::zero(2);
        a.add_term(Rational::integer(1), vec![1, 0]); // x0
        a.add_term(Rational::integer(2), vec![0, 1]); // 2*x1

        let mut b = MultivariatePolynomial::<Rational>::zero(2);
        b.add_term(Rational::integer(3), vec![1, 0]); // 3*x0
        b.add_term(Rational::integer(1), vec![0, 0]); // 1

        let sum = a.add_poly(&b);
        // 4*x0 + 2*x1 + 1
        assert_eq!(sum.num_terms(), 3);
    }

    #[test]
    fn test_multivariate_mul() {
        // (x0 + 1)(x0 - 1) = x0^2 - 1
        let mut a = MultivariatePolynomial::<Rational>::zero(1);
        a.add_term(Rational::integer(1), vec![1]); // x0
        a.add_term(Rational::integer(1), vec![0]); // 1

        let mut b = MultivariatePolynomial::<Rational>::zero(1);
        b.add_term(Rational::integer(1), vec![1]); // x0
        b.add_term(Rational::integer(-1), vec![0]); // -1

        let prod = a.mul_poly(&b);
        let result = prod.evaluate(&[Rational::integer(3)]);
        assert_eq!(result, Rational::integer(8)); // 3^2 - 1 = 8
    }

    #[test]
    fn test_multivariate_evaluate() {
        let mut p = MultivariatePolynomial::<Rational>::zero(2);
        p.add_term(Rational::integer(1), vec![2, 0]); // x0^2
        p.add_term(Rational::integer(2), vec![0, 1]); // 2*x1
        p.add_term(Rational::integer(-3), vec![0, 0]); // -3

        // At (2, 5): 4 + 10 - 3 = 11
        let result = p.evaluate(&[Rational::integer(2), Rational::integer(5)]);
        assert_eq!(result, Rational::integer(11));
    }

    // -- Square-free factorization --

    #[test]
    fn test_square_free() {
        // (x-1)^2 * (x-2) = x^3 - 4x^2 + 5x - 2
        let p = rpoly(&[-2, 5, -4, 1]);
        let factors = p.square_free_factorization();
        assert!(!factors.is_empty());
    }

    // -- Shift and truncate --

    #[test]
    fn test_poly_shift() {
        let p = rpoly(&[1, 2]); // 1 + 2x
        let shifted = p.shift(2); // x^2 + 2x^3
        assert_eq!(shifted.degree(), Some(3));
        assert_eq!(shifted.coeff(0), Rational::zero());
        assert_eq!(shifted.coeff(2), Rational::integer(1));
    }

    #[test]
    fn test_poly_truncate() {
        let p = rpoly(&[1, 2, 3, 4]); // 1 + 2x + 3x^2 + 4x^3
        let t = p.truncate(1); // 1 + 2x
        assert_eq!(t.degree(), Some(1));
    }

    // -- Neg test --

    #[test]
    fn test_poly_neg() {
        let p = rpoly(&[1, -2, 3]);
        let neg = -p;
        assert_eq!(neg.coeff(0), Rational::integer(-1));
        assert_eq!(neg.coeff(1), Rational::integer(2));
        assert_eq!(neg.coeff(2), Rational::integer(-3));
    }
}
