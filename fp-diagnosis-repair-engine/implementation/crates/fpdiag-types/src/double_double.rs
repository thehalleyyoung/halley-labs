//! Double-double arithmetic.
//!
//! A double-double (dd) number is represented as the unevaluated sum of
//! two f64 values `hi + lo` where `|lo| ≤ ½ ulp(hi)`.  This gives
//! approximately 106 bits of significand (≈31 decimal digits) using
//! only native f64 operations—useful as an intermediate step between
//! f64 and full MPFR when the overhead of arbitrary precision is too high.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};

/// A double-double floating-point number: `value ≈ hi + lo`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DoubleDouble {
    /// High-order component.
    pub hi: f64,
    /// Low-order component (correction term).
    pub lo: f64,
}

impl DoubleDouble {
    /// Create a double-double from a single f64.
    pub fn from_f64(x: f64) -> Self {
        Self { hi: x, lo: 0.0 }
    }

    /// Create a double-double from hi and lo components.
    pub fn new(hi: f64, lo: f64) -> Self {
        Self { hi, lo }
    }

    /// Collapse to a single f64 (loses the extra precision).
    pub fn to_f64(self) -> f64 {
        self.hi + self.lo
    }

    /// Knuth's two-sum: compute `a + b = s + e` exactly.
    fn two_sum(a: f64, b: f64) -> (f64, f64) {
        let s = a + b;
        let v = s - a;
        let e = (a - (s - v)) + (b - v);
        (s, e)
    }

    /// Dekker's two-product: compute `a * b = p + e` exactly (using FMA).
    fn two_product(a: f64, b: f64) -> (f64, f64) {
        let p = a * b;
        let e = a.mul_add(b, -p);
        (p, e)
    }

    /// Quick two-sum (requires |a| ≥ |b|).
    fn quick_two_sum(a: f64, b: f64) -> (f64, f64) {
        let s = a + b;
        let e = b - (s - a);
        (s, e)
    }

    /// Absolute value.
    pub fn abs(self) -> Self {
        if self.hi < 0.0 || (self.hi == 0.0 && self.lo < 0.0) {
            -self
        } else {
            self
        }
    }

    /// Whether this value is zero.
    pub fn is_zero(self) -> bool {
        self.hi == 0.0 && self.lo == 0.0
    }

    /// Square root using Newton-Raphson refinement.
    pub fn sqrt(self) -> Self {
        if self.hi < 0.0 {
            return Self::new(f64::NAN, f64::NAN);
        }
        if self.is_zero() {
            return self;
        }
        let x = 1.0 / self.hi.sqrt();
        let ax = Self::from_f64(self.hi * x);
        // One Newton step: ax + (self - ax*ax) * (x/2)
        let diff = self - ax * ax;
        let half_x = Self::from_f64(x * 0.5);
        ax + diff * half_x
    }
}

impl PartialEq for DoubleDouble {
    fn eq(&self, other: &Self) -> bool {
        self.hi == other.hi && self.lo == other.lo
    }
}

impl fmt::Display for DoubleDouble {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.16e} + {:.16e}", self.hi, self.lo)
    }
}

impl From<f64> for DoubleDouble {
    fn from(x: f64) -> Self {
        Self::from_f64(x)
    }
}

impl Add for DoubleDouble {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let (s1, e1) = Self::two_sum(self.hi, rhs.hi);
        let (s2, e2) = Self::two_sum(self.lo, rhs.lo);
        let (s1, e1) = Self::quick_two_sum(s1, e1 + s2);
        let (s1, e1) = Self::quick_two_sum(s1, e1 + e2);
        Self { hi: s1, lo: e1 }
    }
}

impl Sub for DoubleDouble {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        self + (-rhs)
    }
}

impl Neg for DoubleDouble {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            hi: -self.hi,
            lo: -self.lo,
        }
    }
}

impl Mul for DoubleDouble {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let (p, e) = Self::two_product(self.hi, rhs.hi);
        let e = e + self.hi * rhs.lo + self.lo * rhs.hi;
        let (hi, lo) = Self::quick_two_sum(p, e);
        Self { hi, lo }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dd_add_exact() {
        let a = DoubleDouble::from_f64(1.0);
        let b = DoubleDouble::from_f64(1e-20);
        let c = a + b;
        // The sum should preserve the small component
        assert!((c.to_f64() - 1.0).abs() < 1e-15);
        assert!(c.lo != 0.0 || c.hi != 1.0); // lo captures the residual
    }

    #[test]
    fn dd_mul() {
        let a = DoubleDouble::from_f64(3.0);
        let b = DoubleDouble::from_f64(7.0);
        let c = a * b;
        assert!((c.to_f64() - 21.0).abs() < 1e-30);
    }
}
