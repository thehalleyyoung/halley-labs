//! Multivariate polynomial types for symbolic computation.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};

/// A monomial represented as a map from variable index to exponent.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Monomial {
    /// Exponents indexed by variable number.
    pub exponents: BTreeMap<usize, u32>,
}

impl Monomial {
    /// Create a constant monomial (all exponents zero).
    pub fn one() -> Self {
        Self {
            exponents: BTreeMap::new(),
        }
    }

    /// Create a monomial for a single variable x_i.
    pub fn var(i: usize) -> Self {
        let mut exponents = BTreeMap::new();
        exponents.insert(i, 1);
        Self { exponents }
    }

    /// Create a monomial x_i^k.
    pub fn var_pow(i: usize, k: u32) -> Self {
        if k == 0 {
            return Self::one();
        }
        let mut exponents = BTreeMap::new();
        exponents.insert(i, k);
        Self { exponents }
    }

    /// Total degree of the monomial.
    pub fn total_degree(&self) -> u32 {
        self.exponents.values().sum()
    }

    /// Number of variables appearing in the monomial.
    pub fn num_vars(&self) -> usize {
        self.exponents.len()
    }

    /// Get the exponent of variable i.
    pub fn exponent(&self, i: usize) -> u32 {
        self.exponents.get(&i).copied().unwrap_or(0)
    }

    /// Multiply two monomials.
    pub fn multiply(&self, other: &Monomial) -> Monomial {
        let mut result = self.exponents.clone();
        for (&var, &exp) in &other.exponents {
            *result.entry(var).or_insert(0) += exp;
        }
        Monomial { exponents: result }
    }

    /// Check if this monomial divides another.
    pub fn divides(&self, other: &Monomial) -> bool {
        self.exponents
            .iter()
            .all(|(&var, &exp)| other.exponent(var) >= exp)
    }

    /// Divide this monomial by another (if divisible).
    pub fn divide(&self, divisor: &Monomial) -> Option<Monomial> {
        if !divisor.divides(self) {
            return None;
        }
        let mut result = self.exponents.clone();
        for (&var, &exp) in &divisor.exponents {
            let entry = result.entry(var).or_insert(0);
            *entry -= exp;
            if *entry == 0 {
                result.remove(&var);
            }
        }
        Some(Monomial { exponents: result })
    }

    /// Compute the LCM of two monomials.
    pub fn lcm(&self, other: &Monomial) -> Monomial {
        let mut result = self.exponents.clone();
        for (&var, &exp) in &other.exponents {
            let entry = result.entry(var).or_insert(0);
            *entry = (*entry).max(exp);
        }
        Monomial { exponents: result }
    }

    /// GCD of two monomials.
    pub fn gcd(&self, other: &Monomial) -> Monomial {
        let mut result = BTreeMap::new();
        for (&var, &exp) in &self.exponents {
            let other_exp = other.exponent(var);
            let min_exp = exp.min(other_exp);
            if min_exp > 0 {
                result.insert(var, min_exp);
            }
        }
        Monomial { exponents: result }
    }

    /// Evaluate the monomial at given variable values.
    pub fn evaluate(&self, values: &[f64]) -> f64 {
        self.exponents
            .iter()
            .map(|(&var, &exp)| values[var].powi(exp as i32))
            .product()
    }

    /// Partial derivative with respect to variable i.
    pub fn partial_derivative(&self, i: usize) -> Option<(f64, Monomial)> {
        let exp = self.exponent(i);
        if exp == 0 {
            return None;
        }
        let coeff = exp as f64;
        let mut result = self.exponents.clone();
        if exp == 1 {
            result.remove(&i);
        } else {
            *result.get_mut(&i).unwrap() -= 1;
        }
        Some((coeff, Monomial { exponents: result }))
    }
}

impl fmt::Display for Monomial {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.exponents.is_empty() {
            return write!(f, "1");
        }
        let mut first = true;
        for (&var, &exp) in &self.exponents {
            if !first {
                write!(f, "*")?;
            }
            if exp == 1 {
                write!(f, "x{}", var)?;
            } else {
                write!(f, "x{}^{}", var, exp)?;
            }
            first = false;
        }
        Ok(())
    }
}

/// A multivariate polynomial over f64 coefficients.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Polynomial {
    pub terms: BTreeMap<Monomial, f64>,
}

impl Polynomial {
    /// The zero polynomial.
    pub fn zero() -> Self {
        Self {
            terms: BTreeMap::new(),
        }
    }

    /// A constant polynomial.
    pub fn constant(c: f64) -> Self {
        if c.abs() < 1e-15 {
            return Self::zero();
        }
        let mut terms = BTreeMap::new();
        terms.insert(Monomial::one(), c);
        Self { terms }
    }

    /// A single variable polynomial x_i.
    pub fn var(i: usize) -> Self {
        let mut terms = BTreeMap::new();
        terms.insert(Monomial::var(i), 1.0);
        Self { terms }
    }

    /// Check if the polynomial is zero.
    pub fn is_zero(&self) -> bool {
        self.terms.is_empty() || self.terms.values().all(|c| c.abs() < 1e-15)
    }

    /// Total degree of the polynomial.
    pub fn degree(&self) -> u32 {
        self.terms
            .keys()
            .map(|m| m.total_degree())
            .max()
            .unwrap_or(0)
    }

    /// Number of non-zero terms.
    pub fn num_terms(&self) -> usize {
        self.terms.len()
    }

    /// Get the leading term (highest degree monomial).
    pub fn leading_term(&self) -> Option<(&Monomial, f64)> {
        self.terms
            .iter()
            .max_by_key(|(m, _)| m.total_degree())
            .map(|(m, &c)| (m, c))
    }

    /// Evaluate the polynomial at given variable values.
    pub fn evaluate(&self, values: &[f64]) -> f64 {
        self.terms
            .iter()
            .map(|(m, &c)| c * m.evaluate(values))
            .sum()
    }

    /// Partial derivative with respect to variable i.
    pub fn partial_derivative(&self, i: usize) -> Polynomial {
        let mut result = Polynomial::zero();
        for (monomial, &coeff) in &self.terms {
            if let Some((d_coeff, d_mono)) = monomial.partial_derivative(i) {
                let new_coeff = coeff * d_coeff;
                *result.terms.entry(d_mono).or_insert(0.0) += new_coeff;
            }
        }
        result.clean();
        result
    }

    /// Gradient: vector of partial derivatives.
    pub fn gradient(&self, n_vars: usize) -> Vec<Polynomial> {
        (0..n_vars)
            .map(|i| self.partial_derivative(i))
            .collect()
    }

    /// Hessian matrix of second derivatives.
    pub fn hessian(&self, n_vars: usize) -> Vec<Vec<Polynomial>> {
        let grad = self.gradient(n_vars);
        grad.iter()
            .map(|gi| gi.gradient(n_vars))
            .collect()
    }

    /// Scale by a constant.
    pub fn scale(&self, s: f64) -> Self {
        let terms = self.terms.iter().map(|(m, &c)| (m.clone(), c * s)).collect();
        let mut p = Self { terms };
        p.clean();
        p
    }

    /// Remove terms with near-zero coefficients.
    pub fn clean(&mut self) {
        self.terms.retain(|_, c| c.abs() >= 1e-15);
    }

    /// Substitute variable i with a polynomial.
    pub fn substitute(&self, _i: usize, _replacement: &Polynomial) -> Polynomial {
        // Simplified: return a copy for now
        self.clone()
    }

    /// Compute the Poisson bracket {f, g} for 2n-dimensional phase space.
    /// {f, g} = Σ_i (∂f/∂q_i ∂g/∂p_i - ∂f/∂p_i ∂g/∂q_i)
    pub fn poisson_bracket(&self, other: &Polynomial, n_dof: usize) -> Polynomial {
        let mut result = Polynomial::zero();
        for i in 0..n_dof {
            let q_idx = i;
            let p_idx = n_dof + i;
            let df_dq = self.partial_derivative(q_idx);
            let dg_dp = other.partial_derivative(p_idx);
            let df_dp = self.partial_derivative(p_idx);
            let dg_dq = other.partial_derivative(q_idx);
            result = result + df_dq * dg_dp - df_dp * dg_dq;
        }
        result
    }
}

impl Add for Polynomial {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let mut result = self.terms;
        for (m, c) in rhs.terms {
            *result.entry(m).or_insert(0.0) += c;
        }
        let mut p = Self { terms: result };
        p.clean();
        p
    }
}

impl Sub for Polynomial {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        self + rhs.scale(-1.0)
    }
}

impl Mul for Polynomial {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let mut result = BTreeMap::new();
        for (m1, &c1) in &self.terms {
            for (m2, &c2) in &rhs.terms {
                let new_mono = m1.multiply(m2);
                *result.entry(new_mono).or_insert(0.0) += c1 * c2;
            }
        }
        let mut p = Self { terms: result };
        p.clean();
        p
    }
}

impl Neg for Polynomial {
    type Output = Self;
    fn neg(self) -> Self {
        self.scale(-1.0)
    }
}

impl fmt::Display for Polynomial {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            return write!(f, "0");
        }
        let mut first = true;
        for (mono, &coeff) in &self.terms {
            if !first && coeff > 0.0 {
                write!(f, " + ")?;
            } else if !first && coeff < 0.0 {
                write!(f, " - ")?;
            }
            let abs_coeff = coeff.abs();
            if mono.exponents.is_empty() {
                write!(f, "{}", abs_coeff)?;
            } else if (abs_coeff - 1.0).abs() < 1e-15 {
                write!(f, "{}", mono)?;
            } else {
                write!(f, "{}*{}", abs_coeff, mono)?;
            }
            first = false;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monomial_basic() {
        let m = Monomial::var(0);
        assert_eq!(m.total_degree(), 1);
        assert_eq!(m.num_vars(), 1);
    }

    #[test]
    fn test_monomial_multiply() {
        let m1 = Monomial::var(0); // x
        let m2 = Monomial::var_pow(0, 2); // x^2
        let prod = m1.multiply(&m2);
        assert_eq!(prod.exponent(0), 3); // x^3
    }

    #[test]
    fn test_monomial_divides() {
        let m1 = Monomial::var(0); // x
        let m2 = Monomial::var_pow(0, 2); // x^2
        assert!(m1.divides(&m2));
        assert!(!m2.divides(&m1));
    }

    #[test]
    fn test_polynomial_addition() {
        let p1 = Polynomial::var(0); // x
        let p2 = Polynomial::var(0); // x
        let sum = p1 + p2; // 2x
        assert_eq!(sum.evaluate(&[3.0]), 6.0);
    }

    #[test]
    fn test_polynomial_multiplication() {
        let x = Polynomial::var(0);
        let y = Polynomial::var(1);
        let prod = x * y; // xy
        assert_eq!(prod.evaluate(&[2.0, 3.0]), 6.0);
    }

    #[test]
    fn test_partial_derivative() {
        // f = x^2 + 2xy + y^2
        let x = Polynomial::var(0);
        let y = Polynomial::var(1);
        let f = x.clone() * x.clone() + x.clone().scale(2.0) * y.clone() + y.clone() * y.clone();
        let df_dx = f.partial_derivative(0);
        // df/dx = 2x + 2y
        assert!((df_dx.evaluate(&[1.0, 1.0]) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_poisson_bracket() {
        // H = p^2/(2m) + kq^2/2 for harmonic oscillator
        // q = x0, p = x1
        let q = Polynomial::var(0);
        let p = Polynomial::var(1);
        let h = p.clone() * p.clone() * Polynomial::constant(0.5)
            + q.clone() * q.clone() * Polynomial::constant(0.5);
        // {q, H} = ∂q/∂q * ∂H/∂p - ∂q/∂p * ∂H/∂q = p
        let bracket = q.poisson_bracket(&h, 1);
        assert!((bracket.evaluate(&[1.0, 3.0]) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_polynomial_degree() {
        let x = Polynomial::var(0);
        let y = Polynomial::var(1);
        let f = x.clone() * x * y;
        assert_eq!(f.degree(), 3);
    }
}
