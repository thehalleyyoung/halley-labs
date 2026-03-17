//! Extended floating-point classification helpers.
//!
//! Goes beyond [`std::num::FpCategory`] with additional categories useful
//! for floating-point error analysis: near-zero, near-overflow, near
//! underflow, and cancelation-prone value pairs.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Extended floating-point classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FpClassification {
    /// Positive or negative zero.
    Zero,
    /// Subnormal (denormalized) number.
    Subnormal,
    /// Normal number in the "safe" range.
    Normal,
    /// Number very close to the overflow threshold.
    NearOverflow,
    /// Number very close to the underflow threshold.
    NearUnderflow,
    /// Positive or negative infinity.
    Infinite,
    /// Not a number.
    NaN,
}

impl FpClassification {
    /// Classify an f64 value.
    pub fn classify(x: f64) -> Self {
        if x.is_nan() {
            Self::NaN
        } else if x.is_infinite() {
            Self::Infinite
        } else if x == 0.0 {
            Self::Zero
        } else if x.abs() < f64::MIN_POSITIVE {
            Self::Subnormal
        } else if x.abs() > f64::MAX * 0.5 {
            Self::NearOverflow
        } else if x.abs() < f64::MIN_POSITIVE * 1e10 {
            Self::NearUnderflow
        } else {
            Self::Normal
        }
    }

    /// Whether this class is "dangerous" for floating-point accuracy.
    pub fn is_dangerous(&self) -> bool {
        matches!(
            self,
            Self::Subnormal | Self::NearOverflow | Self::NearUnderflow | Self::Infinite | Self::NaN
        )
    }

    /// Human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Zero => "zero",
            Self::Subnormal => "subnormal",
            Self::Normal => "normal",
            Self::NearOverflow => "near-overflow",
            Self::NearUnderflow => "near-underflow",
            Self::Infinite => "infinite",
            Self::NaN => "NaN",
        }
    }
}

impl fmt::Display for FpClassification {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// Detect whether two values are cancellation-prone (nearly equal, same sign,
/// about to be subtracted).
pub fn is_cancellation_prone(a: f64, b: f64) -> bool {
    if a == 0.0 && b == 0.0 {
        return false;
    }
    let max_abs = a.abs().max(b.abs());
    if max_abs == 0.0 {
        return false;
    }
    let diff = (a - b).abs();
    // If the difference is much smaller than the operands, cancellation is likely
    diff / max_abs < 1e-8
}

/// Detect whether addition will cause absorption (small addend lost).
pub fn is_absorption_prone(accumulator: f64, addend: f64) -> bool {
    if addend == 0.0 {
        return false;
    }
    let ratio = accumulator.abs() / addend.abs();
    // If the accumulator is more than 2^p times the addend, the addend is absorbed
    ratio > (1u64 << 52) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_normal() {
        assert_eq!(FpClassification::classify(1.0), FpClassification::Normal);
    }

    #[test]
    fn classify_subnormal() {
        assert_eq!(
            FpClassification::classify(5e-324),
            FpClassification::Subnormal
        );
    }

    #[test]
    fn cancellation_detection() {
        assert!(is_cancellation_prone(1.0000000001, 1.0));
        assert!(!is_cancellation_prone(1.0, 2.0));
    }
}
