//! Interval arithmetic for bounding conservation quantities.
//!
//! Provides rigorous interval arithmetic to track uncertainty
//! bounds through conservation law computations.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// A closed interval [lo, hi] representing a range of possible values.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Interval {
    pub lo: f64,
    pub hi: f64,
}

impl Interval {
    /// Create a new interval [lo, hi].
    pub fn new(lo: f64, hi: f64) -> Self {
        debug_assert!(lo <= hi, "Interval: lo ({}) > hi ({})", lo, hi);
        Self { lo, hi }
    }

    /// Create a point interval [x, x].
    pub fn point(x: f64) -> Self {
        Self { lo: x, hi: x }
    }

    /// Create the entire real line interval.
    pub fn entire() -> Self {
        Self {
            lo: f64::NEG_INFINITY,
            hi: f64::INFINITY,
        }
    }

    /// Create the empty interval.
    pub fn empty() -> Self {
        Self {
            lo: f64::INFINITY,
            hi: f64::NEG_INFINITY,
        }
    }

    /// Check if this interval is empty.
    pub fn is_empty(&self) -> bool {
        self.lo > self.hi
    }

    /// Width of the interval.
    pub fn width(&self) -> f64 {
        if self.is_empty() {
            0.0
        } else {
            self.hi - self.lo
        }
    }

    /// Midpoint of the interval.
    pub fn midpoint(&self) -> f64 {
        (self.lo + self.hi) / 2.0
    }

    /// Radius (half-width) of the interval.
    pub fn radius(&self) -> f64 {
        self.width() / 2.0
    }

    /// Check if a value is contained in this interval.
    pub fn contains(&self, x: f64) -> bool {
        self.lo <= x && x <= self.hi
    }

    /// Check if this interval contains another interval.
    pub fn contains_interval(&self, other: &Interval) -> bool {
        self.lo <= other.lo && other.hi <= self.hi
    }

    /// Check if two intervals overlap.
    pub fn overlaps(&self, other: &Interval) -> bool {
        self.lo <= other.hi && other.lo <= self.hi
    }

    /// Intersection of two intervals.
    pub fn intersection(&self, other: &Interval) -> Self {
        let lo = self.lo.max(other.lo);
        let hi = self.hi.min(other.hi);
        if lo > hi {
            Self::empty()
        } else {
            Self { lo, hi }
        }
    }

    /// Hull (smallest enclosing interval) of two intervals.
    pub fn hull(&self, other: &Interval) -> Self {
        Self {
            lo: self.lo.min(other.lo),
            hi: self.hi.max(other.hi),
        }
    }

    /// Apply absolute value.
    pub fn abs(&self) -> Self {
        if self.lo >= 0.0 {
            *self
        } else if self.hi <= 0.0 {
            Self {
                lo: -self.hi,
                hi: -self.lo,
            }
        } else {
            Self {
                lo: 0.0,
                hi: self.lo.abs().max(self.hi.abs()),
            }
        }
    }

    /// Square the interval.
    pub fn sqr(&self) -> Self {
        if self.lo >= 0.0 {
            Self {
                lo: self.lo * self.lo,
                hi: self.hi * self.hi,
            }
        } else if self.hi <= 0.0 {
            Self {
                lo: self.hi * self.hi,
                hi: self.lo * self.lo,
            }
        } else {
            Self {
                lo: 0.0,
                hi: self.lo.abs().max(self.hi.abs()).powi(2),
            }
        }
    }

    /// Square root of the interval.
    pub fn sqrt(&self) -> Self {
        let lo = if self.lo < 0.0 { 0.0 } else { self.lo.sqrt() };
        Self {
            lo,
            hi: self.hi.max(0.0).sqrt(),
        }
    }

    /// Exponential of the interval.
    pub fn exp(&self) -> Self {
        Self {
            lo: self.lo.exp(),
            hi: self.hi.exp(),
        }
    }

    /// Natural logarithm of the interval.
    pub fn ln(&self) -> Self {
        let lo = if self.lo <= 0.0 {
            f64::NEG_INFINITY
        } else {
            self.lo.ln()
        };
        Self {
            lo,
            hi: self.hi.ln(),
        }
    }

    /// Sine of the interval (conservative bound).
    pub fn sin(&self) -> Self {
        if self.width() >= 2.0 * std::f64::consts::PI {
            return Self::new(-1.0, 1.0);
        }
        let lo_sin = self.lo.sin();
        let hi_sin = self.hi.sin();
        let mut result_lo = lo_sin.min(hi_sin);
        let mut result_hi = lo_sin.max(hi_sin);
        let pi_half = std::f64::consts::FRAC_PI_2;
        let two_pi = 2.0 * std::f64::consts::PI;
        let k_lo = ((self.lo - pi_half) / two_pi).ceil() as i64;
        let k_hi = ((self.hi - pi_half) / two_pi).floor() as i64;
        if k_lo <= k_hi {
            result_hi = 1.0;
        }
        let k_lo2 = ((self.lo + pi_half) / two_pi).ceil() as i64;
        let k_hi2 = ((self.hi + pi_half) / two_pi).floor() as i64;
        if k_lo2 <= k_hi2 {
            result_lo = -1.0;
        }
        Self::new(result_lo, result_hi)
    }

    /// Cosine of the interval (conservative bound).
    pub fn cos(&self) -> Self {
        let shifted = Self::new(
            self.lo + std::f64::consts::FRAC_PI_2,
            self.hi + std::f64::consts::FRAC_PI_2,
        );
        shifted.sin()
    }

    /// Power function.
    pub fn pow(&self, n: i32) -> Self {
        if n == 0 {
            return Self::point(1.0);
        }
        if n == 1 {
            return *self;
        }
        if n == 2 {
            return self.sqr();
        }
        if n < 0 {
            let pos = self.pow(-n);
            return Self::point(1.0) / pos;
        }
        if n % 2 == 0 {
            let half = self.pow(n / 2);
            half.sqr()
        } else {
            let rest = self.pow(n - 1);
            *self * rest
        }
    }

    /// Relative error of this interval compared to an exact value.
    pub fn relative_error(&self, exact: f64) -> f64 {
        if exact.abs() < f64::EPSILON {
            self.width()
        } else {
            let max_err = (self.lo - exact).abs().max((self.hi - exact).abs());
            max_err / exact.abs()
        }
    }
}

impl fmt::Display for Interval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{:.6e}, {:.6e}]", self.lo, self.hi)
    }
}

impl Add for Interval {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            lo: self.lo + rhs.lo,
            hi: self.hi + rhs.hi,
        }
    }
}

impl Sub for Interval {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            lo: self.lo - rhs.hi,
            hi: self.hi - rhs.lo,
        }
    }
}

impl Mul for Interval {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let products = [
            self.lo * rhs.lo,
            self.lo * rhs.hi,
            self.hi * rhs.lo,
            self.hi * rhs.hi,
        ];
        Self {
            lo: products.iter().cloned().fold(f64::INFINITY, f64::min),
            hi: products.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        }
    }
}

impl Div for Interval {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        if rhs.contains(0.0) {
            Self::entire()
        } else {
            let inv = Self {
                lo: 1.0 / rhs.hi,
                hi: 1.0 / rhs.lo,
            };
            self * inv
        }
    }
}

impl Neg for Interval {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            lo: -self.hi,
            hi: -self.lo,
        }
    }
}

/// Multi-dimensional interval (box) for phase space regions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntervalBox {
    pub intervals: Vec<Interval>,
}

impl IntervalBox {
    pub fn new(intervals: Vec<Interval>) -> Self {
        Self { intervals }
    }

    pub fn dimension(&self) -> usize {
        self.intervals.len()
    }

    pub fn volume(&self) -> f64 {
        self.intervals.iter().map(|i| i.width()).product()
    }

    pub fn midpoint(&self) -> Vec<f64> {
        self.intervals.iter().map(|i| i.midpoint()).collect()
    }

    pub fn contains_point(&self, point: &[f64]) -> bool {
        self.intervals
            .iter()
            .zip(point.iter())
            .all(|(iv, &x)| iv.contains(x))
    }

    pub fn hull(&self, other: &IntervalBox) -> Self {
        Self {
            intervals: self
                .intervals
                .iter()
                .zip(other.intervals.iter())
                .map(|(a, b)| a.hull(b))
                .collect(),
        }
    }

    pub fn intersection(&self, other: &IntervalBox) -> Self {
        Self {
            intervals: self
                .intervals
                .iter()
                .zip(other.intervals.iter())
                .map(|(a, b)| a.intersection(b))
                .collect(),
        }
    }

    /// Split the box along the widest dimension.
    pub fn bisect(&self) -> (Self, Self) {
        let widest = self
            .intervals
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.width().partial_cmp(&b.1.width()).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        let mid = self.intervals[widest].midpoint();
        let mut lo_box = self.clone();
        let mut hi_box = self.clone();
        lo_box.intervals[widest].hi = mid;
        hi_box.intervals[widest].lo = mid;
        (lo_box, hi_box)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interval_basic() {
        let a = Interval::new(1.0, 3.0);
        assert_eq!(a.width(), 2.0);
        assert_eq!(a.midpoint(), 2.0);
        assert!(a.contains(2.0));
        assert!(!a.contains(4.0));
    }

    #[test]
    fn test_interval_arithmetic() {
        let a = Interval::new(1.0, 2.0);
        let b = Interval::new(3.0, 4.0);
        let sum = a + b;
        assert_eq!(sum.lo, 4.0);
        assert_eq!(sum.hi, 6.0);
        let diff = a - b;
        assert_eq!(diff.lo, -3.0);
        assert_eq!(diff.hi, -1.0);
        let prod = a * b;
        assert_eq!(prod.lo, 3.0);
        assert_eq!(prod.hi, 8.0);
    }

    #[test]
    fn test_interval_sqr() {
        let a = Interval::new(-2.0, 3.0);
        let sq = a.sqr();
        assert_eq!(sq.lo, 0.0);
        assert_eq!(sq.hi, 9.0);
    }

    #[test]
    fn test_interval_sin() {
        let a = Interval::new(0.0, std::f64::consts::PI);
        let s = a.sin();
        assert!(s.lo >= -0.01);
        assert!(s.hi <= 1.01);
    }

    #[test]
    fn test_interval_box() {
        let b = IntervalBox::new(vec![
            Interval::new(0.0, 1.0),
            Interval::new(0.0, 1.0),
        ]);
        assert_eq!(b.dimension(), 2);
        assert!((b.volume() - 1.0).abs() < 1e-10);
        assert!(b.contains_point(&[0.5, 0.5]));
    }

    #[test]
    fn test_interval_bisect() {
        let b = IntervalBox::new(vec![
            Interval::new(0.0, 4.0),
            Interval::new(0.0, 1.0),
        ]);
        let (lo, hi) = b.bisect();
        assert_eq!(lo.intervals[0].hi, 2.0);
        assert_eq!(hi.intervals[0].lo, 2.0);
    }
}
