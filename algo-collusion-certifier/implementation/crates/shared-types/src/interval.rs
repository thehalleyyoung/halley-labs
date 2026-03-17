//! Interval arithmetic for the numerical-to-formal verification bridge.
//!
//! Provides rigorous interval propagation so that every floating-point
//! computation can be bounded and later verified in exact arithmetic.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Core interval type
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// A closed interval [lo, hi] representing a range of possible values.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Interval<T> {
    pub lo: T,
    pub hi: T,
}

/// Convenience alias for `Interval<f64>`.
pub type IntervalF64 = Interval<f64>;

/// Result of comparing two intervals.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IntervalComparison {
    /// Every value in A is strictly less than every value in B.
    DefinitelyLess,
    /// Every value in A is strictly greater than every value in B.
    DefinitelyGreater,
    /// Every value in A equals every value in B (both are point intervals at the same value).
    DefinitelyEqual,
    /// The intervals overlap — the ordering is uncertain.
    Overlapping,
}

impl fmt::Display for IntervalComparison {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IntervalComparison::DefinitelyLess => write!(f, "<"),
            IntervalComparison::DefinitelyGreater => write!(f, ">"),
            IntervalComparison::DefinitelyEqual => write!(f, "="),
            IntervalComparison::Overlapping => write!(f, "~"),
        }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Interval<f64> implementation
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

impl Interval<f64> {
    /// Create a new interval. Panics if `lo > hi`.
    pub fn new(lo: f64, hi: f64) -> Self {
        debug_assert!(lo <= hi, "Interval: lo ({}) > hi ({})", lo, hi);
        Interval { lo, hi }
    }

    /// Create a point (degenerate) interval.
    pub fn point(value: f64) -> Self {
        Interval { lo: value, hi: value }
    }

    /// Create an interval from a value with symmetric error bound.
    pub fn with_error(value: f64, error: f64) -> Self {
        let abs_err = error.abs();
        Interval { lo: value - abs_err, hi: value + abs_err }
    }

    /// Create an interval enclosing an f64 and its rounding neighbors.
    /// This accounts for the fact that the true value could differ from `v`
    /// by up to one ULP (unit in the last place).
    pub fn from_f64_with_ulp(v: f64) -> Self {
        if v == 0.0 {
            return Interval { lo: -f64::MIN_POSITIVE, hi: f64::MIN_POSITIVE };
        }
        let ulp = (v.abs() * f64::EPSILON).max(f64::MIN_POSITIVE);
        Interval { lo: v - ulp, hi: v + ulp }
    }

    /// Width of the interval.
    pub fn width(&self) -> f64 {
        self.hi - self.lo
    }

    /// Midpoint of the interval.
    pub fn midpoint(&self) -> f64 {
        (self.lo + self.hi) / 2.0
    }

    /// Radius (half-width) of the interval.
    pub fn radius(&self) -> f64 {
        self.width() / 2.0
    }

    /// Relative width: width / |midpoint|. Returns `f64::INFINITY` for zero midpoint.
    pub fn relative_width(&self) -> f64 {
        let mid = self.midpoint().abs();
        if mid < f64::MIN_POSITIVE {
            if self.width() < f64::MIN_POSITIVE { 0.0 } else { f64::INFINITY }
        } else {
            self.width() / mid
        }
    }

    /// Whether the interval contains a specific value.
    pub fn contains(&self, value: f64) -> bool {
        self.lo <= value && value <= self.hi
    }

    /// Whether this interval contains zero.
    pub fn contains_zero(&self) -> bool {
        self.lo <= 0.0 && 0.0 <= self.hi
    }

    /// Whether two intervals overlap.
    pub fn overlaps(&self, other: &Self) -> bool {
        self.lo <= other.hi && other.lo <= self.hi
    }

    /// Whether this interval is a subset of another.
    pub fn is_subset_of(&self, other: &Self) -> bool {
        other.lo <= self.lo && self.hi <= other.hi
    }

    /// Whether the interval is a single point.
    pub fn is_point(&self) -> bool {
        (self.hi - self.lo).abs() < f64::EPSILON
    }

    /// Convex hull of two intervals.
    pub fn hull(&self, other: &Self) -> Self {
        Interval { lo: self.lo.min(other.lo), hi: self.hi.max(other.hi) }
    }

    /// Intersection of two intervals. Returns `None` if disjoint.
    pub fn intersection(&self, other: &Self) -> Option<Self> {
        let lo = self.lo.max(other.lo);
        let hi = self.hi.min(other.hi);
        if lo <= hi { Some(Interval { lo, hi }) } else { None }
    }

    /// Square the interval.
    pub fn square(&self) -> Self {
        if self.lo >= 0.0 {
            Interval { lo: self.lo * self.lo, hi: self.hi * self.hi }
        } else if self.hi <= 0.0 {
            Interval { lo: self.hi * self.hi, hi: self.lo * self.lo }
        } else {
            Interval { lo: 0.0, hi: self.lo.abs().max(self.hi.abs()).powi(2) }
        }
    }

    /// Square root (only valid for non-negative intervals).
    pub fn sqrt(&self) -> Self {
        let lo = self.lo.max(0.0).sqrt();
        let hi = self.hi.max(0.0).sqrt();
        Interval { lo, hi }
    }

    /// Absolute value interval.
    pub fn abs(&self) -> Self {
        if self.lo >= 0.0 {
            *self
        } else if self.hi <= 0.0 {
            Interval { lo: -self.hi, hi: -self.lo }
        } else {
            Interval { lo: 0.0, hi: self.lo.abs().max(self.hi.abs()) }
        }
    }

    /// Reciprocal. Returns `None` if the interval contains zero.
    pub fn recip(&self) -> Option<Self> {
        if self.contains_zero() {
            None
        } else {
            Some(Interval { lo: 1.0 / self.hi, hi: 1.0 / self.lo })
        }
    }

    /// Natural exponential of interval.
    pub fn exp(&self) -> Self {
        Interval { lo: self.lo.exp(), hi: self.hi.exp() }
    }

    /// Natural logarithm of interval (only valid for positive intervals).
    pub fn ln(&self) -> Option<Self> {
        if self.lo <= 0.0 { return None; }
        Some(Interval { lo: self.lo.ln(), hi: self.hi.ln() })
    }

    /// Power function: self^n for integer n.
    pub fn powi(&self, n: i32) -> Self {
        if n == 0 {
            return Interval::point(1.0);
        }
        if n == 1 {
            return *self;
        }
        if n < 0 {
            return match self.powi(-n).recip() {
                Some(r) => r,
                None => Interval { lo: f64::NEG_INFINITY, hi: f64::INFINITY },
            };
        }
        if n % 2 == 0 {
            self.square().powi(n / 2)
        } else {
            *self * self.powi(n - 1)
        }
    }

    /// Compare two intervals.
    pub fn compare(&self, other: &Self) -> IntervalComparison {
        if self.hi < other.lo {
            IntervalComparison::DefinitelyLess
        } else if self.lo > other.hi {
            IntervalComparison::DefinitelyGreater
        } else if self.is_point() && other.is_point() && (self.lo - other.lo).abs() < f64::EPSILON {
            IntervalComparison::DefinitelyEqual
        } else {
            IntervalComparison::Overlapping
        }
    }

    /// Clamp the interval to [lo_bound, hi_bound].
    pub fn clamp(&self, lo_bound: f64, hi_bound: f64) -> Self {
        Interval {
            lo: self.lo.max(lo_bound).min(hi_bound),
            hi: self.hi.max(lo_bound).min(hi_bound),
        }
    }

    /// Widen the interval symmetrically by `amount` on each side.
    pub fn widen(&self, amount: f64) -> Self {
        Interval { lo: self.lo - amount, hi: self.hi + amount }
    }
}

impl PartialEq for Interval<f64> {
    fn eq(&self, other: &Self) -> bool {
        (self.lo - other.lo).abs() < f64::EPSILON && (self.hi - other.hi).abs() < f64::EPSILON
    }
}

impl fmt::Display for Interval<f64> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{:.8}, {:.8}]", self.lo, self.hi)
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Arithmetic operations
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

impl ops::Add for Interval<f64> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Interval { lo: self.lo + rhs.lo, hi: self.hi + rhs.hi }
    }
}

impl ops::Sub for Interval<f64> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Interval { lo: self.lo - rhs.hi, hi: self.hi - rhs.lo }
    }
}

impl ops::Mul for Interval<f64> {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let products = [
            self.lo * rhs.lo,
            self.lo * rhs.hi,
            self.hi * rhs.lo,
            self.hi * rhs.hi,
        ];
        let lo = products.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = products.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        Interval { lo, hi }
    }
}

impl ops::Div for Interval<f64> {
    type Output = Self;
    /// Division. If rhs contains zero, result spans [-∞, +∞].
    fn div(self, rhs: Self) -> Self {
        if rhs.contains_zero() {
            return Interval { lo: f64::NEG_INFINITY, hi: f64::INFINITY };
        }
        self * Interval { lo: 1.0 / rhs.hi, hi: 1.0 / rhs.lo }
    }
}

impl ops::Neg for Interval<f64> {
    type Output = Self;
    fn neg(self) -> Self {
        Interval { lo: -self.hi, hi: -self.lo }
    }
}

// Scalar operations
impl ops::Mul<f64> for Interval<f64> {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        if rhs >= 0.0 {
            Interval { lo: self.lo * rhs, hi: self.hi * rhs }
        } else {
            Interval { lo: self.hi * rhs, hi: self.lo * rhs }
        }
    }
}

impl ops::Div<f64> for Interval<f64> {
    type Output = Self;
    fn div(self, rhs: f64) -> Self {
        if rhs == 0.0 {
            return Interval { lo: f64::NEG_INFINITY, hi: f64::INFINITY };
        }
        if rhs > 0.0 {
            Interval { lo: self.lo / rhs, hi: self.hi / rhs }
        } else {
            Interval { lo: self.hi / rhs, hi: self.lo / rhs }
        }
    }
}

impl ops::Add<f64> for Interval<f64> {
    type Output = Self;
    fn add(self, rhs: f64) -> Self {
        Interval { lo: self.lo + rhs, hi: self.hi + rhs }
    }
}

impl ops::Sub<f64> for Interval<f64> {
    type Output = Self;
    fn sub(self, rhs: f64) -> Self {
        Interval { lo: self.lo - rhs, hi: self.hi - rhs }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Compare free function
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Compare two intervals and return the comparison result.
pub fn compare_intervals(a: &IntervalF64, b: &IntervalF64) -> IntervalComparison {
    a.compare(b)
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Statistical interval operations
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Compute the interval for the mean of a collection of interval values.
///
/// mean([a₁, b₁], [a₂, b₂], ...) = [Σaᵢ/n, Σbᵢ/n]
pub fn mean_interval(intervals: &[IntervalF64]) -> IntervalF64 {
    if intervals.is_empty() {
        return Interval::point(0.0);
    }
    let n = intervals.len() as f64;
    let lo = intervals.iter().map(|i| i.lo).sum::<f64>() / n;
    let hi = intervals.iter().map(|i| i.hi).sum::<f64>() / n;
    Interval { lo, hi }
}

/// Compute an outer bound on the variance interval.
///
/// Uses the identity Var(X) = E[X²] − E[X]² with interval arithmetic.
pub fn variance_interval(intervals: &[IntervalF64]) -> IntervalF64 {
    if intervals.len() < 2 {
        return Interval::point(0.0);
    }
    let n = intervals.len() as f64;
    let mean_iv = mean_interval(intervals);
    let mean_sq = mean_iv.square();

    let sq_sum_lo: f64 = intervals.iter().map(|i| i.square().lo).sum::<f64>();
    let sq_sum_hi: f64 = intervals.iter().map(|i| i.square().hi).sum::<f64>();
    let mean_of_squares = Interval { lo: sq_sum_lo / n, hi: sq_sum_hi / n };

    // Var = E[X²] − (E[X])², clamped to [0, ∞)
    let var_raw = mean_of_squares - mean_sq;
    Interval { lo: var_raw.lo.max(0.0), hi: var_raw.hi.max(0.0) }
}

/// Compute an outer bound on the standard deviation interval.
pub fn stddev_interval(intervals: &[IntervalF64]) -> IntervalF64 {
    variance_interval(intervals).sqrt()
}

/// Compute interval for the sum.
pub fn sum_interval(intervals: &[IntervalF64]) -> IntervalF64 {
    if intervals.is_empty() {
        return Interval::point(0.0);
    }
    let lo = intervals.iter().map(|i| i.lo).sum::<f64>();
    let hi = intervals.iter().map(|i| i.hi).sum::<f64>();
    Interval { lo, hi }
}

/// Compute the interval hull enclosing all given intervals.
pub fn hull_all(intervals: &[IntervalF64]) -> IntervalF64 {
    if intervals.is_empty() {
        return Interval::point(0.0);
    }
    let lo = intervals.iter().map(|i| i.lo).fold(f64::INFINITY, f64::min);
    let hi = intervals.iter().map(|i| i.hi).fold(f64::NEG_INFINITY, f64::max);
    Interval { lo, hi }
}

/// Interval difference of means: mean(A) − mean(B).
pub fn difference_of_means(a: &[IntervalF64], b: &[IntervalF64]) -> IntervalF64 {
    mean_interval(a) - mean_interval(b)
}

/// Check whether a collection of interval comparisons are all consistent
/// with a given ordering (e.g., all DefinitelyLess).
pub fn verify_ordering_consistent(
    comparisons: &[IntervalComparison],
    expected: IntervalComparison,
) -> bool {
    comparisons.iter().all(|c| *c == expected)
}

/// Convert a slice of f64 values to point intervals.
pub fn to_point_intervals(values: &[f64]) -> Vec<IntervalF64> {
    values.iter().map(|v| Interval::point(*v)).collect()
}

/// Convert a slice of f64 values to intervals with uniform error bounds.
pub fn to_error_intervals(values: &[f64], error: f64) -> Vec<IntervalF64> {
    values.iter().map(|v| Interval::with_error(*v, error)).collect()
}

/// Convert a slice of f64 values to intervals with ULP-based bounds.
pub fn to_ulp_intervals(values: &[f64]) -> Vec<IntervalF64> {
    values.iter().map(|v| Interval::from_f64_with_ulp(*v)).collect()
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Rational interval conversion helpers
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Widen an interval by `n` ULPs on each side.
pub fn widen_by_ulps(iv: IntervalF64, n: u64) -> IntervalF64 {
    let lo_ulp = if iv.lo == 0.0 {
        f64::MIN_POSITIVE
    } else {
        iv.lo.abs() * f64::EPSILON
    };
    let hi_ulp = if iv.hi == 0.0 {
        f64::MIN_POSITIVE
    } else {
        iv.hi.abs() * f64::EPSILON
    };
    Interval {
        lo: iv.lo - lo_ulp * n as f64,
        hi: iv.hi + hi_ulp * n as f64,
    }
}

/// Create an interval guaranteed to contain the true sum `a + b`
/// accounting for floating-point rounding.
pub fn safe_add(a: f64, b: f64) -> IntervalF64 {
    let result = a + b;
    Interval::from_f64_with_ulp(result)
}

/// Create an interval guaranteed to contain the true product `a * b`.
pub fn safe_mul(a: f64, b: f64) -> IntervalF64 {
    let result = a * b;
    Interval::from_f64_with_ulp(result)
}

/// Create an interval guaranteed to contain the true quotient `a / b`.
pub fn safe_div(a: f64, b: f64) -> IntervalF64 {
    let result = a / b;
    Interval::from_f64_with_ulp(result)
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Tests
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interval_creation() {
        let iv = Interval::new(1.0, 3.0);
        assert_eq!(iv.lo, 1.0);
        assert_eq!(iv.hi, 3.0);
    }

    #[test]
    fn test_point_interval() {
        let iv = Interval::point(5.0);
        assert!(iv.is_point());
        assert_eq!(iv.width(), 0.0);
        assert_eq!(iv.midpoint(), 5.0);
    }

    #[test]
    fn test_interval_width() {
        let iv = Interval::new(1.0, 4.0);
        assert_eq!(iv.width(), 3.0);
        assert_eq!(iv.midpoint(), 2.5);
        assert_eq!(iv.radius(), 1.5);
    }

    #[test]
    fn test_contains() {
        let iv = Interval::new(1.0, 5.0);
        assert!(iv.contains(3.0));
        assert!(iv.contains(1.0));
        assert!(iv.contains(5.0));
        assert!(!iv.contains(0.0));
        assert!(!iv.contains(6.0));
    }

    #[test]
    fn test_overlaps() {
        let a = Interval::new(1.0, 3.0);
        let b = Interval::new(2.0, 4.0);
        let c = Interval::new(4.0, 6.0);
        assert!(a.overlaps(&b));
        assert!(!a.overlaps(&c));
    }

    #[test]
    fn test_hull() {
        let a = Interval::new(1.0, 3.0);
        let b = Interval::new(5.0, 7.0);
        let h = a.hull(&b);
        assert_eq!(h.lo, 1.0);
        assert_eq!(h.hi, 7.0);
    }

    #[test]
    fn test_intersection() {
        let a = Interval::new(1.0, 4.0);
        let b = Interval::new(3.0, 6.0);
        let i = a.intersection(&b).unwrap();
        assert_eq!(i.lo, 3.0);
        assert_eq!(i.hi, 4.0);
        assert!(Interval::new(1.0, 2.0).intersection(&Interval::new(3.0, 4.0)).is_none());
    }

    #[test]
    fn test_add() {
        let a = Interval::new(1.0, 2.0);
        let b = Interval::new(3.0, 4.0);
        let c = a + b;
        assert_eq!(c.lo, 4.0);
        assert_eq!(c.hi, 6.0);
    }

    #[test]
    fn test_sub() {
        let a = Interval::new(3.0, 5.0);
        let b = Interval::new(1.0, 2.0);
        let c = a - b;
        assert_eq!(c.lo, 1.0);
        assert_eq!(c.hi, 4.0);
    }

    #[test]
    fn test_mul() {
        let a = Interval::new(2.0, 3.0);
        let b = Interval::new(4.0, 5.0);
        let c = a * b;
        assert_eq!(c.lo, 8.0);
        assert_eq!(c.hi, 15.0);
    }

    #[test]
    fn test_mul_negative() {
        let a = Interval::new(-2.0, 3.0);
        let b = Interval::new(-1.0, 4.0);
        let c = a * b;
        assert_eq!(c.lo, -8.0);
        assert_eq!(c.hi, 12.0);
    }

    #[test]
    fn test_div() {
        let a = Interval::new(6.0, 8.0);
        let b = Interval::new(2.0, 4.0);
        let c = a / b;
        assert_eq!(c.lo, 1.5);
        assert_eq!(c.hi, 4.0);
    }

    #[test]
    fn test_div_by_zero() {
        let a = Interval::new(1.0, 2.0);
        let b = Interval::new(-1.0, 1.0);
        let c = a / b;
        assert!(c.lo.is_infinite());
    }

    #[test]
    fn test_compare_definitely_less() {
        let a = Interval::new(1.0, 2.0);
        let b = Interval::new(3.0, 4.0);
        assert_eq!(compare_intervals(&a, &b), IntervalComparison::DefinitelyLess);
    }

    #[test]
    fn test_compare_overlapping() {
        let a = Interval::new(1.0, 3.0);
        let b = Interval::new(2.0, 4.0);
        assert_eq!(compare_intervals(&a, &b), IntervalComparison::Overlapping);
    }

    #[test]
    fn test_square_positive() {
        let a = Interval::new(2.0, 3.0);
        let s = a.square();
        assert_eq!(s.lo, 4.0);
        assert_eq!(s.hi, 9.0);
    }

    #[test]
    fn test_square_crossing_zero() {
        let a = Interval::new(-2.0, 3.0);
        let s = a.square();
        assert_eq!(s.lo, 0.0);
        assert_eq!(s.hi, 9.0);
    }

    #[test]
    fn test_mean_interval() {
        let ivs = vec![
            Interval::new(1.0, 2.0),
            Interval::new(3.0, 4.0),
        ];
        let m = mean_interval(&ivs);
        assert_eq!(m.lo, 2.0);
        assert_eq!(m.hi, 3.0);
    }

    #[test]
    fn test_sum_interval() {
        let ivs = vec![
            Interval::new(1.0, 2.0),
            Interval::new(3.0, 4.0),
            Interval::new(5.0, 6.0),
        ];
        let s = sum_interval(&ivs);
        assert_eq!(s.lo, 9.0);
        assert_eq!(s.hi, 12.0);
    }

    #[test]
    fn test_scalar_mul() {
        let a = Interval::new(2.0, 4.0);
        let b = a * 3.0;
        assert_eq!(b.lo, 6.0);
        assert_eq!(b.hi, 12.0);
        let c = a * -2.0;
        assert_eq!(c.lo, -8.0);
        assert_eq!(c.hi, -4.0);
    }

    #[test]
    fn test_neg() {
        let a = Interval::new(1.0, 3.0);
        let b = -a;
        assert_eq!(b.lo, -3.0);
        assert_eq!(b.hi, -1.0);
    }

    #[test]
    fn test_exp_ln_roundtrip() {
        let a = Interval::new(1.0, 2.0);
        let e = a.exp();
        let l = e.ln().unwrap();
        assert!((l.lo - 1.0).abs() < 1e-10);
        assert!((l.hi - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_sqrt() {
        let a = Interval::new(4.0, 9.0);
        let s = a.sqrt();
        assert!((s.lo - 2.0).abs() < 1e-10);
        assert!((s.hi - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_abs_interval() {
        let a = Interval::new(-3.0, -1.0);
        let b = a.abs();
        assert_eq!(b.lo, 1.0);
        assert_eq!(b.hi, 3.0);

        let c = Interval::new(-2.0, 5.0).abs();
        assert_eq!(c.lo, 0.0);
        assert_eq!(c.hi, 5.0);
    }

    #[test]
    fn test_with_error() {
        let iv = Interval::with_error(10.0, 0.5);
        assert_eq!(iv.lo, 9.5);
        assert_eq!(iv.hi, 10.5);
    }

    #[test]
    fn test_clamp() {
        let iv = Interval::new(-1.0, 5.0);
        let c = iv.clamp(0.0, 3.0);
        assert_eq!(c.lo, 0.0);
        assert_eq!(c.hi, 3.0);
    }

    #[test]
    fn test_to_point_intervals() {
        let pts = to_point_intervals(&[1.0, 2.0, 3.0]);
        assert_eq!(pts.len(), 3);
        assert!(pts[0].is_point());
    }

    #[test]
    fn test_variance_interval() {
        let ivs = vec![
            Interval::point(2.0),
            Interval::point(4.0),
            Interval::point(4.0),
            Interval::point(4.0),
            Interval::point(5.0),
            Interval::point(5.0),
            Interval::point(7.0),
            Interval::point(9.0),
        ];
        let var = variance_interval(&ivs);
        assert!(var.lo >= 0.0);
        assert!(var.hi > 0.0);
    }

    #[test]
    fn test_safe_arithmetic() {
        let s = safe_add(1.0, 2.0);
        assert!(s.contains(3.0));
        let m = safe_mul(3.0, 4.0);
        assert!(m.contains(12.0));
        let d = safe_div(10.0, 3.0);
        assert!(d.contains(10.0 / 3.0));
    }

    #[test]
    fn test_hull_all() {
        let ivs = vec![
            Interval::new(5.0, 7.0),
            Interval::new(1.0, 3.0),
            Interval::new(4.0, 6.0),
        ];
        let h = hull_all(&ivs);
        assert_eq!(h.lo, 1.0);
        assert_eq!(h.hi, 7.0);
    }

    #[test]
    fn test_is_subset_of() {
        let a = Interval::new(2.0, 3.0);
        let b = Interval::new(1.0, 4.0);
        assert!(a.is_subset_of(&b));
        assert!(!b.is_subset_of(&a));
    }
}
