//! Numeric types and precision handling.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Scalar value type used throughout the system.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Scalar {
    pub value: f64,
    pub uncertainty: Option<f64>,
}

impl Scalar {
    pub fn new(value: f64) -> Self {
        Self { value, uncertainty: None }
    }

    pub fn with_uncertainty(value: f64, uncertainty: f64) -> Self {
        Self { value, uncertainty: Some(uncertainty) }
    }

    pub fn is_zero(&self, tol: f64) -> bool {
        self.value.abs() < tol
    }

    pub fn is_positive(&self) -> bool {
        self.value > 0.0
    }

    pub fn is_negative(&self) -> bool {
        self.value < 0.0
    }

    pub fn abs(&self) -> Self {
        Self { value: self.value.abs(), uncertainty: self.uncertainty }
    }

    pub fn relative_error(&self, reference: f64) -> f64 {
        if reference.abs() < 1e-15 { self.value.abs() } else { ((self.value - reference) / reference).abs() }
    }

    pub fn add(self, other: Self) -> Self {
        let v = self.value + other.value;
        let u = match (self.uncertainty, other.uncertainty) {
            (Some(a), Some(b)) => Some((a * a + b * b).sqrt()),
            (Some(a), None) | (None, Some(a)) => Some(a),
            (None, None) => None,
        };
        Self { value: v, uncertainty: u }
    }

    pub fn mul(self, other: Self) -> Self {
        let v = self.value * other.value;
        let u = match (self.uncertainty, other.uncertainty) {
            (Some(a), Some(b)) => {
                let ra = if self.value.abs() > 1e-15 { a / self.value.abs() } else { a };
                let rb = if other.value.abs() > 1e-15 { b / other.value.abs() } else { b };
                Some(v.abs() * (ra * ra + rb * rb).sqrt())
            }
            (Some(a), None) => Some(a * other.value.abs()),
            (None, Some(b)) => Some(b * self.value.abs()),
            (None, None) => None,
        };
        Self { value: v, uncertainty: u }
    }
}

impl fmt::Display for Scalar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.uncertainty {
            Some(u) => write!(f, "{:.6e} ± {:.2e}", self.value, u),
            None => write!(f, "{:.6e}", self.value),
        }
    }
}

impl From<f64> for Scalar {
    fn from(v: f64) -> Self { Self::new(v) }
}

/// Tolerance specification for numerical computations.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Tolerance {
    pub absolute: f64,
    pub relative: f64,
}

impl Tolerance {
    pub fn new(absolute: f64, relative: f64) -> Self {
        Self { absolute, relative }
    }

    pub fn strict() -> Self { Self { absolute: 1e-14, relative: 1e-12 } }
    pub fn standard() -> Self { Self { absolute: 1e-10, relative: 1e-8 } }
    pub fn relaxed() -> Self { Self { absolute: 1e-6, relative: 1e-4 } }

    pub fn is_satisfied(&self, value: f64, reference: f64) -> bool {
        let abs_err = (value - reference).abs();
        if abs_err < self.absolute { return true; }
        let rel_err = if reference.abs() > 1e-15 { abs_err / reference.abs() } else { abs_err };
        rel_err < self.relative
    }

    pub fn combined_error(&self, value: f64, reference: f64) -> f64 {
        let abs_err = (value - reference).abs();
        let rel_err = if reference.abs() > 1e-15 { abs_err / reference.abs() } else { abs_err };
        abs_err.min(rel_err)
    }
}

impl Default for Tolerance {
    fn default() -> Self { Self::standard() }
}

/// Precision level for computations.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum PrecisionLevel {
    Low,
    Medium,
    High,
    Exact,
}

impl PrecisionLevel {
    pub fn tolerance(&self) -> Tolerance {
        match self {
            PrecisionLevel::Low => Tolerance::relaxed(),
            PrecisionLevel::Medium => Tolerance::standard(),
            PrecisionLevel::High => Tolerance::strict(),
            PrecisionLevel::Exact => Tolerance::new(0.0, 0.0),
        }
    }

    pub fn max_iterations(&self) -> usize {
        match self {
            PrecisionLevel::Low => 100,
            PrecisionLevel::Medium => 1000,
            PrecisionLevel::High => 10000,
            PrecisionLevel::Exact => 100000,
        }
    }
}

/// Statistics for a set of numerical values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Statistics {
    pub count: usize,
    pub mean: f64,
    pub variance: f64,
    pub min: f64,
    pub max: f64,
    pub median: f64,
}

impl Statistics {
    pub fn from_values(values: &[f64]) -> Self {
        if values.is_empty() {
            return Self { count: 0, mean: 0.0, variance: 0.0, min: 0.0, max: 0.0, median: 0.0 };
        }
        let count = values.len();
        let mean = values.iter().sum::<f64>() / count as f64;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / count as f64;
        let min = values.iter().copied().fold(f64::INFINITY, f64::min);
        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = if count % 2 == 0 { (sorted[count / 2 - 1] + sorted[count / 2]) / 2.0 } else { sorted[count / 2] };
        Self { count, mean, variance, min, max, median }
    }

    pub fn std_dev(&self) -> f64 { self.variance.sqrt() }

    pub fn coefficient_of_variation(&self) -> f64 {
        if self.mean.abs() < 1e-15 { 0.0 } else { self.std_dev() / self.mean.abs() }
    }
}

/// A numerical interval [lo, hi].
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Interval {
    pub lo: f64,
    pub hi: f64,
}

impl Interval {
    pub fn new(lo: f64, hi: f64) -> Self {
        let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };
        Self { lo, hi }
    }

    pub fn point(v: f64) -> Self { Self { lo: v, hi: v } }
    pub fn width(&self) -> f64 { self.hi - self.lo }
    pub fn midpoint(&self) -> f64 { (self.lo + self.hi) / 2.0 }
    pub fn contains(&self, v: f64) -> bool { v >= self.lo && v <= self.hi }

    pub fn intersect(&self, other: &Self) -> Option<Self> {
        let lo = self.lo.max(other.lo);
        let hi = self.hi.min(other.hi);
        if lo <= hi { Some(Self { lo, hi }) } else { None }
    }

    pub fn union(&self, other: &Self) -> Self {
        Self { lo: self.lo.min(other.lo), hi: self.hi.max(other.hi) }
    }

    pub fn add(&self, other: &Self) -> Self {
        Self { lo: self.lo + other.lo, hi: self.hi + other.hi }
    }

    pub fn mul(&self, other: &Self) -> Self {
        let products = [self.lo * other.lo, self.lo * other.hi, self.hi * other.lo, self.hi * other.hi];
        let lo = products.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = products.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        Self { lo, hi }
    }
}

impl fmt::Display for Interval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{:.6e}, {:.6e}]", self.lo, self.hi)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_operations() {
        let a = Scalar::with_uncertainty(1.0, 0.1);
        let b = Scalar::with_uncertainty(2.0, 0.2);
        let sum = a.add(b);
        assert!((sum.value - 3.0).abs() < 1e-12);
        assert!(sum.uncertainty.is_some());
    }

    #[test]
    fn test_tolerance() {
        let tol = Tolerance::standard();
        assert!(tol.is_satisfied(1.0 + 1e-11, 1.0));
        assert!(!tol.is_satisfied(1.0 + 1.0, 1.0));
    }

    #[test]
    fn test_statistics() {
        let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = Statistics::from_values(&vals);
        assert!((stats.mean - 3.0).abs() < 1e-12);
        assert!((stats.median - 3.0).abs() < 1e-12);
        assert_eq!(stats.count, 5);
    }

    #[test]
    fn test_interval() {
        let a = Interval::new(1.0, 3.0);
        let b = Interval::new(2.0, 4.0);
        let c = a.intersect(&b).unwrap();
        assert!((c.lo - 2.0).abs() < 1e-12);
        assert!((c.hi - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_interval_arithmetic() {
        let a = Interval::new(1.0, 2.0);
        let b = Interval::new(3.0, 4.0);
        let sum = a.add(&b);
        assert!((sum.lo - 4.0).abs() < 1e-12);
        assert!((sum.hi - 6.0).abs() < 1e-12);
    }
}
