//! Interval arithmetic for sound numerical computation.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// A closed interval [lo, hi] supporting interval arithmetic.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Interval {
    pub lo: f64,
    pub hi: f64,
}

impl Interval {
    /// Create a new interval. Panics if lo > hi.
    pub fn new(lo: f64, hi: f64) -> Self {
        assert!(lo <= hi, "Interval: lo ({lo}) must be <= hi ({hi})");
        Self { lo, hi }
    }

    /// A point interval [x, x].
    pub fn point(x: f64) -> Self {
        Self { lo: x, hi: x }
    }

    /// The entire real line (approximately).
    pub fn entire() -> Self {
        Self {
            lo: f64::NEG_INFINITY,
            hi: f64::INFINITY,
        }
    }

    /// Width of the interval.
    pub fn width(&self) -> f64 {
        self.hi - self.lo
    }

    /// Midpoint of the interval.
    pub fn midpoint(&self) -> f64 {
        (self.lo + self.hi) * 0.5
    }

    /// Radius (half-width).
    pub fn radius(&self) -> f64 {
        (self.hi - self.lo) * 0.5
    }

    /// Check if value is contained in the interval.
    pub fn contains(&self, x: f64) -> bool {
        x >= self.lo && x <= self.hi
    }

    /// Check if this interval overlaps another.
    pub fn overlaps(&self, other: &Interval) -> bool {
        self.lo <= other.hi && other.lo <= self.hi
    }

    /// Intersection of two intervals, or None if disjoint.
    pub fn intersection(&self, other: &Interval) -> Option<Interval> {
        let lo = self.lo.max(other.lo);
        let hi = self.hi.min(other.hi);
        if lo <= hi {
            Some(Interval { lo, hi })
        } else {
            None
        }
    }

    /// Hull (union bounding interval).
    pub fn hull(&self, other: &Interval) -> Interval {
        Interval {
            lo: self.lo.min(other.lo),
            hi: self.hi.max(other.hi),
        }
    }

    /// Apply cosine over the interval (conservative).
    pub fn cos(&self) -> Interval {
        let a = self.lo;
        let b = self.hi;
        if b - a >= 2.0 * std::f64::consts::PI {
            return Interval::new(-1.0, 1.0);
        }
        let ca = a.cos();
        let cb = b.cos();
        let lo = ca.min(cb);
        let hi = ca.max(cb);
        let n_lo = (a / std::f64::consts::PI).ceil() as i64;
        let n_hi = (b / std::f64::consts::PI).floor() as i64;
        let mut result_lo = lo;
        let mut result_hi = hi;
        for n in n_lo..=n_hi {
            if n % 2 == 0 {
                result_hi = 1.0;
            } else {
                result_lo = -1.0;
            }
        }
        Interval::new(result_lo, result_hi)
    }

    /// Apply sine over the interval (conservative).
    pub fn sin(&self) -> Interval {
        let shifted = Interval::new(
            self.lo - std::f64::consts::FRAC_PI_2,
            self.hi - std::f64::consts::FRAC_PI_2,
        );
        shifted.cos()
    }

    /// Square the interval.
    pub fn sqr(&self) -> Interval {
        if self.lo >= 0.0 {
            Interval::new(self.lo * self.lo, self.hi * self.hi)
        } else if self.hi <= 0.0 {
            Interval::new(self.hi * self.hi, self.lo * self.lo)
        } else {
            Interval::new(0.0, self.lo.abs().max(self.hi.abs()).powi(2))
        }
    }

    /// Square root (returns None if interval contains negatives).
    pub fn sqrt(&self) -> Option<Interval> {
        if self.hi < 0.0 {
            return None;
        }
        let lo = if self.lo < 0.0 { 0.0 } else { self.lo.sqrt() };
        Some(Interval::new(lo, self.hi.sqrt()))
    }

    /// Absolute value.
    pub fn abs(&self) -> Interval {
        if self.lo >= 0.0 {
            *self
        } else if self.hi <= 0.0 {
            Interval::new(-self.hi, -self.lo)
        } else {
            Interval::new(0.0, self.lo.abs().max(self.hi.abs()))
        }
    }

    /// Check if the interval is a subset of another.
    pub fn is_subset_of(&self, other: &Interval) -> bool {
        self.lo >= other.lo && self.hi <= other.hi
    }

    /// Expand the interval by a symmetric amount.
    pub fn expand(&self, amount: f64) -> Interval {
        Interval::new(self.lo - amount, self.hi + amount)
    }

    /// Split at midpoint.
    pub fn bisect(&self) -> (Interval, Interval) {
        let mid = self.midpoint();
        (Interval::new(self.lo, mid), Interval::new(mid, self.hi))
    }
}

impl fmt::Display for Interval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{:.4}, {:.4}]", self.lo, self.hi)
    }
}

impl Add for Interval {
    type Output = Interval;
    fn add(self, rhs: Interval) -> Interval {
        Interval::new(self.lo + rhs.lo, self.hi + rhs.hi)
    }
}

impl Sub for Interval {
    type Output = Interval;
    fn sub(self, rhs: Interval) -> Interval {
        Interval::new(self.lo - rhs.hi, self.hi - rhs.lo)
    }
}

impl Mul for Interval {
    type Output = Interval;
    fn mul(self, rhs: Interval) -> Interval {
        let products = [
            self.lo * rhs.lo,
            self.lo * rhs.hi,
            self.hi * rhs.lo,
            self.hi * rhs.hi,
        ];
        let lo = products.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = products.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        Interval::new(lo, hi)
    }
}

impl Mul<f64> for Interval {
    type Output = Interval;
    fn mul(self, rhs: f64) -> Interval {
        if rhs >= 0.0 {
            Interval::new(self.lo * rhs, self.hi * rhs)
        } else {
            Interval::new(self.hi * rhs, self.lo * rhs)
        }
    }
}

impl Neg for Interval {
    type Output = Interval;
    fn neg(self) -> Interval {
        Interval::new(-self.hi, -self.lo)
    }
}

impl Div for Interval {
    type Output = Interval;
    fn div(self, rhs: Interval) -> Interval {
        if rhs.contains(0.0) {
            Interval::entire()
        } else {
            let inv = Interval::new(1.0 / rhs.hi, 1.0 / rhs.lo);
            self * inv
        }
    }
}

impl PartialEq for Interval {
    fn eq(&self, other: &Self) -> bool {
        (self.lo - other.lo).abs() < 1e-12 && (self.hi - other.hi).abs() < 1e-12
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_ops() {
        let a = Interval::new(1.0, 3.0);
        let b = Interval::new(2.0, 4.0);
        let sum = a + b;
        assert!((sum.lo - 3.0).abs() < 1e-10);
        assert!((sum.hi - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_contains() {
        let i = Interval::new(-1.0, 1.0);
        assert!(i.contains(0.0));
        assert!(!i.contains(2.0));
    }

    #[test]
    fn test_cos() {
        let i = Interval::new(0.0, std::f64::consts::PI);
        let c = i.cos();
        assert!(c.lo <= -1.0 + 1e-10);
        assert!(c.hi >= 1.0 - 1e-10);
    }

    #[test]
    fn test_sqr() {
        let i = Interval::new(-2.0, 3.0);
        let s = i.sqr();
        assert!((s.lo - 0.0).abs() < 1e-10);
        assert!((s.hi - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_intersection() {
        let a = Interval::new(0.0, 5.0);
        let b = Interval::new(3.0, 8.0);
        let c = a.intersection(&b).unwrap();
        assert!((c.lo - 3.0).abs() < 1e-10);
        assert!((c.hi - 5.0).abs() < 1e-10);
    }
}

/// A vector of intervals (box in n-dimensional space).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntervalVector {
    pub components: Vec<Interval>,
}

impl IntervalVector {
    pub fn new(components: Vec<Interval>) -> Self { Self { components } }
    pub fn dim(&self) -> usize { self.components.len() }
    pub fn from_ranges(ranges: &[(f64, f64)]) -> Self {
        Self { components: ranges.iter().map(|&(lo, hi)| Interval::new(lo, hi)).collect() }
    }
    pub fn point(values: &[f64]) -> Self {
        Self { components: values.iter().map(|&v| Interval::point(v)).collect() }
    }
    pub fn midpoint(&self) -> Vec<f64> { self.components.iter().map(|c| c.midpoint()).collect() }
    pub fn max_width(&self) -> f64 { self.components.iter().map(|c| c.width()).fold(0.0, f64::max) }
    pub fn widest_dimension(&self) -> usize {
        self.components.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.width().partial_cmp(&b.width()).unwrap())
            .map(|(i, _)| i).unwrap_or(0)
    }
    pub fn volume(&self) -> f64 { self.components.iter().map(|c| c.width()).product() }
    pub fn contains_point(&self, point: &[f64]) -> bool {
        self.components.iter().zip(point.iter()).all(|(iv, &p)| iv.contains(p))
    }
    pub fn overlaps(&self, other: &IntervalVector) -> bool {
        self.components.iter().zip(other.components.iter()).all(|(a, b)| a.overlaps(b))
    }
    pub fn is_subset_of(&self, other: &IntervalVector) -> bool {
        self.components.iter().zip(other.components.iter()).all(|(a, b)| a.is_subset_of(b))
    }
    pub fn hull(&self, other: &IntervalVector) -> IntervalVector {
        IntervalVector {
            components: self.components.iter().zip(other.components.iter())
                .map(|(a, b)| a.hull(b)).collect()
        }
    }
    pub fn bisect_widest(&self) -> (IntervalVector, IntervalVector) {
        let dim = self.widest_dimension();
        let (lo, hi) = self.components[dim].bisect();
        let mut left = self.components.clone();
        let mut right = self.components.clone();
        left[dim] = lo;
        right[dim] = hi;
        (IntervalVector::new(left), IntervalVector::new(right))
    }
}

impl fmt::Display for IntervalVector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(")?;
        for (i, c) in self.components.iter().enumerate() {
            if i > 0 { write!(f, ", ")?; }
            write!(f, "{}", c)?;
        }
        write!(f, ")")
    }
}

/// A matrix of intervals.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntervalMatrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Interval>,
}

impl IntervalMatrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self { rows, cols, data: vec![Interval::point(0.0); rows * cols] }
    }
    pub fn identity(n: usize) -> Self {
        let mut m = Self::new(n, n);
        for i in 0..n { m.set(i, i, Interval::point(1.0)); }
        m
    }
    pub fn get(&self, row: usize, col: usize) -> Interval { self.data[row * self.cols + col] }
    pub fn set(&mut self, row: usize, col: usize, val: Interval) { self.data[row * self.cols + col] = val; }
    pub fn mul_vector(&self, v: &IntervalVector) -> IntervalVector {
        let mut result = Vec::with_capacity(self.rows);
        for i in 0..self.rows {
            let mut sum = Interval::point(0.0);
            for j in 0..self.cols { sum = sum + self.get(i, j) * v.components[j]; }
            result.push(sum);
        }
        IntervalVector::new(result)
    }
}
