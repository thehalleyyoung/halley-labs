//! Error metrics, intervals, and summary statistics.
//!
//! Provides types for quantifying floating-point error in multiple
//! complementary ways: absolute, relative, ULP-based, and interval
//! representations.

use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};
use std::fmt;

// ─── ErrorMetric ────────────────────────────────────────────────────────────

/// A single error measurement expressed in one of several standard metrics.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ErrorMetric {
    /// Absolute error: |x̂ − x|.
    Absolute(f64),
    /// Relative error: |x̂ − x| / |x|.
    Relative(f64),
    /// ULP distance between the computed and exact values.
    Ulps(f64),
    /// Number of significant bits lost.
    BitsLost(f64),
}

impl ErrorMetric {
    /// Return the raw numerical value regardless of metric kind.
    pub fn value(&self) -> f64 {
        match self {
            Self::Absolute(v) | Self::Relative(v) | Self::Ulps(v) | Self::BitsLost(v) => *v,
        }
    }

    /// Human-readable unit string.
    pub fn unit(&self) -> &'static str {
        match self {
            Self::Absolute(_) => "abs",
            Self::Relative(_) => "rel",
            Self::Ulps(_) => "ulps",
            Self::BitsLost(_) => "bits",
        }
    }

    /// Convert an absolute error to relative, given the true value magnitude.
    pub fn to_relative(&self, true_magnitude: f64) -> Option<Self> {
        match self {
            Self::Absolute(v) if true_magnitude.abs() > 0.0 => {
                Some(Self::Relative(v / true_magnitude.abs()))
            }
            _ => None,
        }
    }
}

impl fmt::Display for ErrorMetric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.4e} {}", self.value(), self.unit())
    }
}

// ─── ErrorBound ─────────────────────────────────────────────────────────────

/// A certified upper bound on error, with the method that produced it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorBound {
    /// The bound value (always non-negative).
    pub bound: f64,
    /// The metric in which the bound is expressed.
    pub metric: ErrorMetric,
    /// How the bound was obtained.
    pub method: BoundMethod,
    /// Confidence level (1.0 = formally certified, <1.0 = empirical).
    pub confidence: f64,
}

/// How an error bound was derived.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BoundMethod {
    /// Formal interval arithmetic.
    IntervalArithmetic,
    /// First-order forward error analysis.
    FirstOrderAnalysis,
    /// Shadow-value comparison at higher precision.
    ShadowValue,
    /// Statistical (Monte-Carlo / stochastic arithmetic).
    Statistical,
    /// User-provided assertion.
    UserAsserted,
}

impl ErrorBound {
    /// Create a formally certified bound via interval arithmetic.
    pub fn certified(bound: f64, metric: ErrorMetric) -> Self {
        Self {
            bound,
            metric,
            method: BoundMethod::IntervalArithmetic,
            confidence: 1.0,
        }
    }

    /// Create an empirical bound from shadow-value comparison.
    pub fn empirical(bound: f64, metric: ErrorMetric, confidence: f64) -> Self {
        Self {
            bound,
            metric,
            method: BoundMethod::ShadowValue,
            confidence: confidence.clamp(0.0, 1.0),
        }
    }
}

// ─── ErrorInterval ──────────────────────────────────────────────────────────

/// A closed interval [lo, hi] bounding a floating-point value or error.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ErrorInterval {
    pub lo: OrderedFloat<f64>,
    pub hi: OrderedFloat<f64>,
}

impl ErrorInterval {
    /// Create an interval, ensuring lo ≤ hi.
    pub fn new(lo: f64, hi: f64) -> Self {
        let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };
        Self {
            lo: OrderedFloat(lo),
            hi: OrderedFloat(hi),
        }
    }

    /// A point interval [v, v].
    pub fn point(v: f64) -> Self {
        Self::new(v, v)
    }

    /// Width of the interval.
    pub fn width(&self) -> f64 {
        self.hi.0 - self.lo.0
    }

    /// Midpoint.
    pub fn midpoint(&self) -> f64 {
        (self.lo.0 + self.hi.0) / 2.0
    }

    /// Whether this interval is contained within `other`.
    pub fn is_subset_of(&self, other: &Self) -> bool {
        self.lo >= other.lo && self.hi <= other.hi
    }

    /// Intersection of two intervals, or `None` if disjoint.
    pub fn intersect(&self, other: &Self) -> Option<Self> {
        let lo = self.lo.max(other.lo);
        let hi = self.hi.min(other.hi);
        if lo <= hi {
            Some(Self { lo, hi })
        } else {
            None
        }
    }

    /// Union (hull) of two intervals.
    pub fn hull(&self, other: &Self) -> Self {
        Self {
            lo: self.lo.min(other.lo),
            hi: self.hi.max(other.hi),
        }
    }
}

impl fmt::Display for ErrorInterval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{:.6e}, {:.6e}]", self.lo.0, self.hi.0)
    }
}

// ─── ErrorSummary ───────────────────────────────────────────────────────────

/// Aggregate error statistics over an array or trace region.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorSummary {
    /// Number of data points.
    pub count: usize,
    /// Maximum error observed.
    pub max_error: ErrorMetric,
    /// Mean error.
    pub mean_error: ErrorMetric,
    /// Median error.
    pub median_error: ErrorMetric,
    /// 95th-percentile error.
    pub p95_error: ErrorMetric,
    /// 99th-percentile error.
    pub p99_error: ErrorMetric,
    /// Fraction of values with relative error > 1e-8.
    pub fraction_high_error: f64,
}

impl ErrorSummary {
    /// Create a summary from a sorted list of absolute errors.
    pub fn from_sorted_absolute_errors(errors: &[f64]) -> Self {
        let n = errors.len();
        if n == 0 {
            return Self {
                count: 0,
                max_error: ErrorMetric::Absolute(0.0),
                mean_error: ErrorMetric::Absolute(0.0),
                median_error: ErrorMetric::Absolute(0.0),
                p95_error: ErrorMetric::Absolute(0.0),
                p99_error: ErrorMetric::Absolute(0.0),
                fraction_high_error: 0.0,
            };
        }
        let sum: f64 = errors.iter().sum();
        let high_count = errors.iter().filter(|&&e| e > 1e-8).count();
        Self {
            count: n,
            max_error: ErrorMetric::Absolute(errors[n - 1]),
            mean_error: ErrorMetric::Absolute(sum / n as f64),
            median_error: ErrorMetric::Absolute(errors[n / 2]),
            p95_error: ErrorMetric::Absolute(errors[(n as f64 * 0.95) as usize]),
            p99_error: ErrorMetric::Absolute(
                errors[(n as f64 * 0.99).min((n - 1) as f64) as usize],
            ),
            fraction_high_error: high_count as f64 / n as f64,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interval_hull_and_intersect() {
        let a = ErrorInterval::new(1.0, 3.0);
        let b = ErrorInterval::new(2.0, 5.0);
        let hull = a.hull(&b);
        assert_eq!(hull.lo.0, 1.0);
        assert_eq!(hull.hi.0, 5.0);
        let inter = a.intersect(&b).unwrap();
        assert_eq!(inter.lo.0, 2.0);
        assert_eq!(inter.hi.0, 3.0);
    }

    #[test]
    fn error_metric_display() {
        let e = ErrorMetric::Ulps(42.0);
        let s = format!("{}", e);
        assert!(s.contains("ulps"));
    }
}
