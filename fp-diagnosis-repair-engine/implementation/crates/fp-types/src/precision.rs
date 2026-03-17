//! Precision modes for floating-point computation.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Precision level for floating-point computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Precision {
    /// 16-bit half precision (~3.3 decimal digits)
    Half,
    /// 32-bit single precision (~7.2 decimal digits)
    Single,
    /// 64-bit double precision (~15.9 decimal digits)
    Double,
    /// 80-bit extended precision (~18.5 decimal digits)
    Extended,
    /// 128-bit quadruple precision (~34.0 decimal digits)
    Quad,
    /// Double-double representation (~31.0 decimal digits)
    DoubleDouble,
    /// Arbitrary precision with given number of bits
    Arbitrary(u32),
}

impl Precision {
    /// Number of significand bits (including implicit leading bit).
    pub fn significand_bits(self) -> u32 {
        match self {
            Self::Half => 11,
            Self::Single => 24,
            Self::Double => 53,
            Self::Extended => 64,
            Self::Quad => 113,
            Self::DoubleDouble => 106,
            Self::Arbitrary(bits) => bits,
        }
    }

    /// Number of accurate decimal digits.
    pub fn decimal_digits(self) -> u32 {
        let bits = self.significand_bits() as f64;
        (bits * std::f64::consts::LOG10_2).floor() as u32
    }

    /// Machine epsilon for this precision.
    pub fn machine_epsilon(self) -> f64 {
        2.0_f64.powi(1 - self.significand_bits() as i32)
    }

    /// Unit roundoff (epsilon / 2).
    pub fn unit_roundoff(self) -> f64 {
        self.machine_epsilon() / 2.0
    }

    /// Whether this precision is strictly higher than another.
    pub fn is_higher_than(self, other: Precision) -> bool {
        self.significand_bits() > other.significand_bits()
    }

    /// Return the next higher standard precision, if available.
    pub fn promote(self) -> Option<Precision> {
        match self {
            Self::Half => Some(Self::Single),
            Self::Single => Some(Self::Double),
            Self::Double => Some(Self::Extended),
            Self::Extended => Some(Self::Quad),
            Self::Quad => None,
            Self::DoubleDouble => Some(Self::Quad),
            Self::Arbitrary(bits) => {
                if bits < 24 {
                    Some(Self::Single)
                } else if bits < 53 {
                    Some(Self::Double)
                } else if bits < 64 {
                    Some(Self::Extended)
                } else if bits < 113 {
                    Some(Self::Quad)
                } else {
                    Some(Self::Arbitrary(bits * 2))
                }
            }
        }
    }

    /// Return the next lower standard precision, if available.
    pub fn demote(self) -> Option<Precision> {
        match self {
            Self::Half => None,
            Self::Single => Some(Self::Half),
            Self::Double => Some(Self::Single),
            Self::Extended => Some(Self::Double),
            Self::Quad => Some(Self::Extended),
            Self::DoubleDouble => Some(Self::Double),
            Self::Arbitrary(bits) => {
                if bits > 113 {
                    Some(Self::Quad)
                } else if bits > 64 {
                    Some(Self::Extended)
                } else if bits > 53 {
                    Some(Self::Double)
                } else if bits > 24 {
                    Some(Self::Single)
                } else if bits > 11 {
                    Some(Self::Half)
                } else {
                    None
                }
            }
        }
    }

    /// Size in bytes for storage.
    pub fn byte_size(self) -> usize {
        match self {
            Self::Half => 2,
            Self::Single => 4,
            Self::Double => 8,
            Self::Extended => 10,
            Self::Quad => 16,
            Self::DoubleDouble => 16,
            Self::Arbitrary(bits) => ((bits + 7) / 8) as usize,
        }
    }

    /// Cost factor for computation at this precision (relative to Double = 1.0).
    pub fn computation_cost(self) -> f64 {
        match self {
            Self::Half => 0.25,
            Self::Single => 0.5,
            Self::Double => 1.0,
            Self::Extended => 1.5,
            Self::Quad => 10.0,
            Self::DoubleDouble => 8.0,
            Self::Arbitrary(bits) => {
                let ratio = bits as f64 / 53.0;
                ratio * ratio
            }
        }
    }
}

impl fmt::Display for Precision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Half => write!(f, "half"),
            Self::Single => write!(f, "single"),
            Self::Double => write!(f, "double"),
            Self::Extended => write!(f, "extended"),
            Self::Quad => write!(f, "quad"),
            Self::DoubleDouble => write!(f, "double-double"),
            Self::Arbitrary(bits) => write!(f, "arbitrary({})", bits),
        }
    }
}

impl Default for Precision {
    fn default() -> Self {
        Self::Double
    }
}

/// A mixed-precision assignment maps variables/operations to precision levels.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PrecisionAssignment {
    /// Map from node/variable ID to precision.
    pub assignments: indexmap::IndexMap<String, Precision>,
    /// Default precision for unassigned operations.
    pub default_precision: Precision,
}

impl PrecisionAssignment {
    /// Create a new assignment with a default precision.
    pub fn new(default: Precision) -> Self {
        Self {
            assignments: indexmap::IndexMap::new(),
            default_precision: default,
        }
    }

    /// Set the precision for a specific node.
    pub fn set(&mut self, node_id: impl Into<String>, precision: Precision) {
        self.assignments.insert(node_id.into(), precision);
    }

    /// Get the precision for a node.
    pub fn get(&self, node_id: &str) -> Precision {
        self.assignments
            .get(node_id)
            .copied()
            .unwrap_or(self.default_precision)
    }

    /// Number of nodes with non-default precision.
    pub fn promoted_count(&self) -> usize {
        self.assignments
            .values()
            .filter(|&&p| p != self.default_precision)
            .count()
    }

    /// Total computation cost relative to all-default.
    pub fn total_cost(&self) -> f64 {
        let default_cost = self.default_precision.computation_cost();
        let mut total = 0.0;
        for prec in self.assignments.values() {
            total += prec.computation_cost() - default_cost;
        }
        total
    }

    /// Memory overhead relative to all-default (in bytes).
    pub fn memory_overhead(&self) -> i64 {
        let default_size = self.default_precision.byte_size() as i64;
        let mut overhead = 0i64;
        for prec in self.assignments.values() {
            overhead += prec.byte_size() as i64 - default_size;
        }
        overhead
    }

    /// Merge another assignment into this one (other takes precedence).
    pub fn merge(&mut self, other: &PrecisionAssignment) {
        for (k, v) in &other.assignments {
            self.assignments.insert(k.clone(), *v);
        }
    }

    /// Check if any node is assigned higher than a given precision.
    pub fn any_above(&self, threshold: Precision) -> bool {
        self.assignments
            .values()
            .any(|p| p.is_higher_than(threshold))
    }
}

/// Precision requirement computed from error analysis.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PrecisionRequirement {
    /// Minimum number of significand bits needed.
    pub min_bits: u32,
    /// The recommended standard precision level.
    pub recommended: Precision,
    /// Confidence in this requirement (0.0 to 1.0).
    pub confidence: f64,
    /// The error bound that drove this requirement.
    pub error_bound: f64,
}

impl PrecisionRequirement {
    /// Compute from a target error bound.
    pub fn from_error_bound(target_error: f64) -> Self {
        if target_error <= 0.0 || target_error.is_nan() {
            return Self {
                min_bits: 113,
                recommended: Precision::Quad,
                confidence: 0.0,
                error_bound: target_error,
            };
        }

        let min_bits = (-target_error.log2()).ceil() as u32 + 1;
        let recommended = if min_bits <= 11 {
            Precision::Half
        } else if min_bits <= 24 {
            Precision::Single
        } else if min_bits <= 53 {
            Precision::Double
        } else if min_bits <= 64 {
            Precision::Extended
        } else if min_bits <= 113 {
            Precision::Quad
        } else {
            Precision::Arbitrary(min_bits)
        };

        Self {
            min_bits,
            recommended,
            confidence: 0.95,
            error_bound: target_error,
        }
    }

    /// Check if a precision satisfies this requirement.
    pub fn is_satisfied_by(&self, precision: Precision) -> bool {
        precision.significand_bits() >= self.min_bits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precision_ordering() {
        assert!(Precision::Double.is_higher_than(Precision::Single));
        assert!(Precision::Quad.is_higher_than(Precision::Double));
        assert!(!Precision::Single.is_higher_than(Precision::Double));
    }

    #[test]
    fn test_precision_promotion() {
        assert_eq!(Precision::Single.promote(), Some(Precision::Double));
        assert_eq!(Precision::Double.promote(), Some(Precision::Extended));
        assert_eq!(Precision::Quad.promote(), None);
    }

    #[test]
    fn test_precision_demotion() {
        assert_eq!(Precision::Double.demote(), Some(Precision::Single));
        assert_eq!(Precision::Half.demote(), None);
    }

    #[test]
    fn test_machine_epsilon() {
        let eps = Precision::Double.machine_epsilon();
        assert!((eps - f64::EPSILON).abs() < 1e-30);
    }

    #[test]
    fn test_precision_assignment() {
        let mut pa = PrecisionAssignment::new(Precision::Double);
        pa.set("node_1", Precision::Quad);
        pa.set("node_2", Precision::Single);

        assert_eq!(pa.get("node_1"), Precision::Quad);
        assert_eq!(pa.get("node_2"), Precision::Single);
        assert_eq!(pa.get("node_3"), Precision::Double);
        assert_eq!(pa.promoted_count(), 2);
    }

    #[test]
    fn test_precision_requirement() {
        let req = PrecisionRequirement::from_error_bound(1e-10);
        assert!(req.is_satisfied_by(Precision::Double));
        assert!(!req.is_satisfied_by(Precision::Single));

        let req2 = PrecisionRequirement::from_error_bound(1e-4);
        assert!(req2.is_satisfied_by(Precision::Single));
    }

    #[test]
    fn test_decimal_digits() {
        assert_eq!(Precision::Single.decimal_digits(), 7);
        assert_eq!(Precision::Double.decimal_digits(), 15);
    }

    #[test]
    fn test_computation_cost() {
        assert!(Precision::Single.computation_cost() < Precision::Double.computation_cost());
        assert!(Precision::Double.computation_cost() < Precision::Quad.computation_cost());
    }
}
