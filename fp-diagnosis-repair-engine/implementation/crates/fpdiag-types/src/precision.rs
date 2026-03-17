//! Precision descriptors and cost modelling.
//!
//! This module defines [`Precision`] — a compact representation of the
//! floating-point precision used in a computation — together with helper
//! types for reasoning about precision requirements and the costs of
//! precision promotion.

use serde::{Deserialize, Serialize};
use std::fmt;

// ─── Precision ──────────────────────────────────────────────────────────────

/// Describes the precision of a floating-point computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Precision {
    /// IEEE 754 binary16 (half precision, 11-bit significand).
    Half,
    /// IEEE 754 binary32 (single precision, 24-bit significand).
    Single,
    /// IEEE 754 binary64 (double precision, 53-bit significand).
    Double,
    /// x87 extended 80-bit (64-bit significand).
    Extended,
    /// IEEE 754 binary128 (quad precision, 113-bit significand).
    Quad,
    /// Double-double: a pair of f64 values giving ~106 bits of significand.
    DoubleDouble,
    /// Arbitrary precision with the given number of significand bits.
    Arbitrary(u32),
}

impl Precision {
    /// Total number of significand bits (including the implicit leading 1
    /// for standard formats).
    pub fn significand_bits(&self) -> u32 {
        match self {
            Self::Half => 11,
            Self::Single => 24,
            Self::Double => 53,
            Self::Extended => 64,
            Self::Quad => 113,
            Self::DoubleDouble => 106,
            Self::Arbitrary(b) => *b,
        }
    }

    /// Approximate number of significant decimal digits.
    pub fn decimal_digits(&self) -> u32 {
        // floor(p * log10(2))
        ((self.significand_bits() as f64) * std::f64::consts::LOG10_2).floor() as u32
    }

    /// Machine epsilon: 2^{-(p-1)}.
    pub fn machine_epsilon(&self) -> f64 {
        2.0_f64.powi(-(self.significand_bits() as i32 - 1))
    }

    /// Whether an integer `n` can be represented exactly in this precision.
    ///
    /// An integer is exactly representable when |n| ≤ 2^p.
    pub fn can_represent_exactly(&self, n: i64) -> bool {
        let p = self.significand_bits();
        if p >= 64 {
            return true;
        }
        let limit = 1_u64 << p;
        (n.unsigned_abs()) <= limit
    }

    /// Construct a `Precision` from a raw bit count, mapping to the
    /// closest named variant when possible.
    pub fn from_bits(bits: u32) -> Self {
        match bits {
            11 => Self::Half,
            24 => Self::Single,
            53 => Self::Double,
            64 => Self::Extended,
            106 => Self::DoubleDouble,
            113 => Self::Quad,
            b => Self::Arbitrary(b),
        }
    }
}

impl PartialOrd for Precision {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Precision {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.significand_bits().cmp(&other.significand_bits())
    }
}

impl fmt::Display for Precision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Half => write!(f, "half ({} bits)", self.significand_bits()),
            Self::Single => write!(f, "single ({} bits)", self.significand_bits()),
            Self::Double => write!(f, "double ({} bits)", self.significand_bits()),
            Self::Extended => write!(f, "extended ({} bits)", self.significand_bits()),
            Self::Quad => write!(f, "quad ({} bits)", self.significand_bits()),
            Self::DoubleDouble => write!(f, "double-double ({} bits)", self.significand_bits()),
            Self::Arbitrary(b) => write!(f, "arbitrary ({b} bits)"),
        }
    }
}

// ─── PrecisionRequirement ───────────────────────────────────────────────────

/// Describes how much precision an operation or subexpression requires.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecisionRequirement {
    /// Minimum number of significand bits required.
    pub min_bits: u32,
    /// Recommended [`Precision`] variant that satisfies `min_bits`.
    pub recommended_precision: Precision,
    /// Human-readable reason for this requirement.
    pub reason: String,
}

impl PrecisionRequirement {
    /// Create a new requirement.
    pub fn new(min_bits: u32, reason: impl Into<String>) -> Self {
        let recommended_precision = Self::select_precision(min_bits);
        Self {
            min_bits,
            recommended_precision,
            reason: reason.into(),
        }
    }

    /// Returns `true` if the given precision satisfies the requirement.
    pub fn is_satisfied_by(&self, p: Precision) -> bool {
        p.significand_bits() >= self.min_bits
    }

    fn select_precision(bits: u32) -> Precision {
        if bits <= 11 {
            Precision::Half
        } else if bits <= 24 {
            Precision::Single
        } else if bits <= 53 {
            Precision::Double
        } else if bits <= 64 {
            Precision::Extended
        } else if bits <= 106 {
            Precision::DoubleDouble
        } else if bits <= 113 {
            Precision::Quad
        } else {
            Precision::Arbitrary(bits)
        }
    }
}

impl fmt::Display for PrecisionRequirement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "need ≥{} bits ({}): {}",
            self.min_bits, self.recommended_precision, self.reason
        )
    }
}

// ─── PrecisionCost ──────────────────────────────────────────────────────────

/// Models the computational cost of operating at a given precision.
///
/// All costs are expressed relative to `Double = 1.0`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PrecisionCost {
    /// The precision being costed.
    pub precision: Precision,
    /// Relative time cost (1.0 = double).
    pub time_factor: f64,
    /// Relative memory cost (1.0 = double).
    pub memory_factor: f64,
    /// Estimated conversion overhead when promoting from a cheaper precision.
    pub conversion_overhead: f64,
}

impl PrecisionCost {
    /// Create a cost entry.
    pub fn new(
        precision: Precision,
        time_factor: f64,
        memory_factor: f64,
        conversion_overhead: f64,
    ) -> Self {
        Self {
            precision,
            time_factor,
            memory_factor,
            conversion_overhead,
        }
    }

    /// Default cost table entry for the given precision.
    pub fn default_cost(precision: Precision) -> Self {
        let (t, m, c) = match precision {
            Precision::Half => (0.5, 0.25, 0.1),
            Precision::Single => (0.6, 0.5, 0.15),
            Precision::Double => (1.0, 1.0, 0.0),
            Precision::Extended => (1.5, 1.25, 0.2),
            Precision::DoubleDouble => (4.0, 2.0, 0.5),
            Precision::Quad => (8.0, 2.0, 0.6),
            Precision::Arbitrary(b) => {
                let ratio = b as f64 / 53.0;
                (ratio * ratio, ratio, ratio * 0.3)
            }
        };
        Self::new(precision, t, m, c)
    }

    /// Composite score (lower is better).  Weights can be adjusted.
    pub fn composite_score(&self, time_weight: f64, memory_weight: f64) -> f64 {
        time_weight * self.time_factor
            + memory_weight * self.memory_factor
            + self.conversion_overhead
    }
}

impl fmt::Display for PrecisionCost {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: time={:.2}× mem={:.2}× conv={:.2}",
            self.precision, self.time_factor, self.memory_factor, self.conversion_overhead
        )
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn significand_bits_known() {
        assert_eq!(Precision::Half.significand_bits(), 11);
        assert_eq!(Precision::Single.significand_bits(), 24);
        assert_eq!(Precision::Double.significand_bits(), 53);
        assert_eq!(Precision::Extended.significand_bits(), 64);
        assert_eq!(Precision::Quad.significand_bits(), 113);
        assert_eq!(Precision::DoubleDouble.significand_bits(), 106);
        assert_eq!(Precision::Arbitrary(200).significand_bits(), 200);
    }

    #[test]
    fn decimal_digits() {
        // double → 15 digits
        assert_eq!(Precision::Double.decimal_digits(), 15);
        // single → 7 digits
        assert_eq!(Precision::Single.decimal_digits(), 7);
    }

    #[test]
    fn machine_epsilon_double() {
        let eps = Precision::Double.machine_epsilon();
        assert!((eps - f64::EPSILON).abs() < 1e-30);
    }

    #[test]
    fn can_represent_exactly_simple() {
        assert!(Precision::Double.can_represent_exactly(1 << 53));
        assert!(!Precision::Double.can_represent_exactly((1_i64 << 53) + 1));
    }

    #[test]
    fn from_bits_round_trip() {
        assert_eq!(Precision::from_bits(53), Precision::Double);
        assert_eq!(Precision::from_bits(24), Precision::Single);
        assert_eq!(Precision::from_bits(200), Precision::Arbitrary(200));
    }

    #[test]
    fn ordering() {
        assert!(Precision::Half < Precision::Single);
        assert!(Precision::Single < Precision::Double);
        assert!(Precision::Double < Precision::Extended);
        assert!(Precision::Extended < Precision::DoubleDouble);
        assert!(Precision::DoubleDouble < Precision::Quad);
    }

    #[test]
    fn display_precision() {
        let s = Precision::Double.to_string();
        assert!(s.contains("53"));
        assert!(s.contains("double"));
    }

    #[test]
    fn requirement_satisfied() {
        let req = PrecisionRequirement::new(53, "need double");
        assert!(req.is_satisfied_by(Precision::Double));
        assert!(req.is_satisfied_by(Precision::Quad));
        assert!(!req.is_satisfied_by(Precision::Single));
    }

    #[test]
    fn requirement_display() {
        let req = PrecisionRequirement::new(64, "cancellation");
        let s = req.to_string();
        assert!(s.contains("64"));
    }

    #[test]
    fn cost_default_double() {
        let c = PrecisionCost::default_cost(Precision::Double);
        assert!((c.time_factor - 1.0).abs() < 1e-10);
        assert!((c.memory_factor - 1.0).abs() < 1e-10);
    }

    #[test]
    fn cost_composite_score() {
        let c = PrecisionCost::default_cost(Precision::Quad);
        let score = c.composite_score(1.0, 1.0);
        assert!(score > 0.0);
    }

    #[test]
    fn cost_display() {
        let c = PrecisionCost::default_cost(Precision::Single);
        let s = c.to_string();
        assert!(s.contains("single"));
    }

    #[test]
    fn serde_precision() {
        let p = Precision::Quad;
        let json = serde_json::to_string(&p).unwrap();
        let p2: Precision = serde_json::from_str(&json).unwrap();
        assert_eq!(p, p2);
    }

    #[test]
    fn serde_precision_requirement() {
        let req = PrecisionRequirement::new(80, "need extended");
        let json = serde_json::to_string(&req).unwrap();
        let req2: PrecisionRequirement = serde_json::from_str(&json).unwrap();
        assert_eq!(req.min_bits, req2.min_bits);
    }

    #[test]
    fn serde_precision_cost() {
        let c = PrecisionCost::default_cost(Precision::DoubleDouble);
        let json = serde_json::to_string(&c).unwrap();
        let c2: PrecisionCost = serde_json::from_str(&json).unwrap();
        assert!((c.time_factor - c2.time_factor).abs() < 1e-10);
    }
}
