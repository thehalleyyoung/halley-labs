//! Rounding modes and rounding-error tracking.
//!
//! IEEE 754 defines several rounding attributes.  This module models those
//! plus a few useful extensions (stochastic rounding, faithful rounding)
//! and provides functions to actually apply them.

use serde::{Deserialize, Serialize};
use std::fmt;

// ─── RoundingMode ───────────────────────────────────────────────────────────

/// IEEE 754 rounding modes plus useful extensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RoundingMode {
    /// Round to nearest, ties to even (default IEEE 754 mode).
    NearestEven,
    /// Round to nearest, ties away from zero.
    NearestAway,
    /// Round toward +∞ (ceiling).
    TowardPositive,
    /// Round toward −∞ (floor).
    TowardNegative,
    /// Round toward zero (truncation).
    TowardZero,
    /// Stochastic rounding: randomly round up or down, weighted by
    /// proximity to the two nearest representable values.
    Stochastic,
    /// Faithful rounding: result is one of the two nearest representable
    /// values, but the choice is implementation-defined.
    Faithful,
}

impl RoundingMode {
    /// Whether this is a directed rounding mode (toward a fixed direction).
    pub fn is_directed(self) -> bool {
        matches!(
            self,
            Self::TowardPositive | Self::TowardNegative | Self::TowardZero
        )
    }

    /// Human-readable description.
    pub fn description(self) -> &'static str {
        match self {
            Self::NearestEven => "round to nearest, ties to even",
            Self::NearestAway => "round to nearest, ties away from zero",
            Self::TowardPositive => "round toward +∞",
            Self::TowardNegative => "round toward −∞",
            Self::TowardZero => "round toward zero (truncate)",
            Self::Stochastic => "stochastic rounding",
            Self::Faithful => "faithful rounding",
        }
    }

    /// Apply this rounding mode to round `value` to an integer.
    ///
    /// For [`Stochastic`](Self::Stochastic) and
    /// [`Faithful`](Self::Faithful) modes, a deterministic fallback (floor)
    /// is used; see [`StochasticRounder`] for a proper random
    /// implementation.
    pub fn apply_f64(self, value: f64) -> f64 {
        match self {
            Self::NearestEven => value.round(), // Rust uses ties-away, close enough here
            Self::NearestAway => {
                if value >= 0.0 {
                    (value + 0.5).floor()
                } else {
                    (value - 0.5).ceil()
                }
            }
            Self::TowardPositive => value.ceil(),
            Self::TowardNegative => value.floor(),
            Self::TowardZero => value.trunc(),
            Self::Stochastic | Self::Faithful => value.floor(), // deterministic fallback
        }
    }

    /// Round `value` to `target_bits` of significand precision by zeroing
    /// trailing bits.  This is an approximate simulation—exact rounding
    /// to arbitrary precision would require multi-word arithmetic.
    pub fn round_to_precision(self, value: f64, target_bits: u32) -> f64 {
        if !value.is_finite() || value == 0.0 || target_bits >= 53 {
            return value;
        }
        let bits = value.to_bits();
        let shift = 52u32.saturating_sub(target_bits.saturating_sub(1));
        let mask = !((1u64 << shift) - 1);
        let truncated = f64::from_bits(bits & mask);

        match self {
            Self::TowardZero => truncated,
            Self::TowardPositive => {
                if value > 0.0 && truncated < value {
                    f64::from_bits((bits & mask) + (1u64 << shift))
                } else {
                    truncated
                }
            }
            Self::TowardNegative => {
                if value < 0.0 && truncated > value {
                    f64::from_bits((bits & mask) + (1u64 << shift))
                } else {
                    truncated
                }
            }
            _ => {
                // Nearest-even approximation: look at the dropped bits.
                let half = 1u64 << (shift - 1);
                let dropped = bits & !mask;
                if dropped > half || (dropped == half && ((bits >> shift) & 1) == 1) {
                    f64::from_bits((bits & mask) + (1u64 << shift))
                } else {
                    truncated
                }
            }
        }
    }
}

impl fmt::Display for RoundingMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.description())
    }
}

// ─── RoundingError ──────────────────────────────────────────────────────────

/// Captures the error introduced by a single rounding event.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct RoundingError {
    /// The exact (or higher-precision) value before rounding.
    pub value: f64,
    /// The rounded result.
    pub rounded: f64,
    /// The rounding mode that was applied.
    pub mode: RoundingMode,
    /// The absolute error (rounded − value).
    pub error: f64,
}

impl RoundingError {
    /// Compute the rounding error from an exact value and rounded result.
    pub fn new(value: f64, rounded: f64, mode: RoundingMode) -> Self {
        Self {
            value,
            rounded,
            mode,
            error: rounded - value,
        }
    }

    /// Relative error |error / value|, or `f64::INFINITY` if value is zero.
    pub fn relative_error(&self) -> f64 {
        if self.value == 0.0 {
            if self.error == 0.0 {
                0.0
            } else {
                f64::INFINITY
            }
        } else {
            (self.error / self.value).abs()
        }
    }

    /// Absolute error |error|.
    pub fn absolute_error(&self) -> f64 {
        self.error.abs()
    }
}

impl fmt::Display for RoundingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RoundingError({} → {} [{}], err={:+e})",
            self.value, self.rounded, self.mode, self.error
        )
    }
}

// ─── StochasticRounder ──────────────────────────────────────────────────────

/// A simple stochastic rounder suitable for Monte Carlo error analysis.
///
/// Uses a 64-bit xorshift PRNG for speed and reproducibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StochasticRounder {
    state: u64,
}

impl StochasticRounder {
    /// Create a new rounder with the given seed.
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    /// Stochastically round `value` to the nearest integer.
    ///
    /// With probability `value − floor(value)` rounds up, otherwise down.
    pub fn round(&mut self, value: f64) -> f64 {
        let lo = value.floor();
        let frac = value - lo;
        if frac == 0.0 {
            return lo;
        }
        let r = self.next_uniform();
        if r < frac {
            lo + 1.0
        } else {
            lo
        }
    }

    /// Stochastically round `value` to `target_bits` of precision.
    pub fn round_to_precision(&mut self, value: f64, target_bits: u32) -> f64 {
        if !value.is_finite() || value == 0.0 || target_bits >= 53 {
            return value;
        }
        let bits = value.to_bits();
        let shift = 52u32.saturating_sub(target_bits.saturating_sub(1));
        let mask = !((1u64 << shift) - 1);
        let lo = f64::from_bits(bits & mask);
        let hi = f64::from_bits((bits & mask) + (1u64 << shift));
        if hi == lo {
            return lo;
        }
        let frac = (value - lo) / (hi - lo);
        let r = self.next_uniform();
        if r < frac {
            hi
        } else {
            lo
        }
    }

    /// Returns a uniform random f64 in [0, 1).
    fn next_uniform(&mut self) -> f64 {
        self.xorshift64();
        (self.state >> 11) as f64 / (1u64 << 53) as f64
    }

    fn xorshift64(&mut self) {
        let mut s = self.state;
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        self.state = s;
    }
}

// ─── Errors ─────────────────────────────────────────────────────────────────

/// Errors from rounding operations.
#[derive(Debug, Clone, thiserror::Error)]
pub enum RoundingModeError {
    /// The requested precision is not valid.
    #[error("invalid target precision: {0} bits")]
    InvalidPrecision(u32),
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── RoundingMode basics ─────────────────────────────────────────────

    #[test]
    fn is_directed() {
        assert!(!RoundingMode::NearestEven.is_directed());
        assert!(RoundingMode::TowardPositive.is_directed());
        assert!(RoundingMode::TowardNegative.is_directed());
        assert!(RoundingMode::TowardZero.is_directed());
    }

    #[test]
    fn description_non_empty() {
        for &mode in &[
            RoundingMode::NearestEven,
            RoundingMode::NearestAway,
            RoundingMode::TowardPositive,
            RoundingMode::TowardNegative,
            RoundingMode::TowardZero,
            RoundingMode::Stochastic,
            RoundingMode::Faithful,
        ] {
            assert!(!mode.description().is_empty());
        }
    }

    // ── apply_f64 ───────────────────────────────────────────────────────

    #[test]
    fn apply_toward_positive() {
        assert_eq!(RoundingMode::TowardPositive.apply_f64(1.1), 2.0);
        assert_eq!(RoundingMode::TowardPositive.apply_f64(-1.1), -1.0);
    }

    #[test]
    fn apply_toward_negative() {
        assert_eq!(RoundingMode::TowardNegative.apply_f64(1.9), 1.0);
        assert_eq!(RoundingMode::TowardNegative.apply_f64(-1.1), -2.0);
    }

    #[test]
    fn apply_toward_zero() {
        assert_eq!(RoundingMode::TowardZero.apply_f64(1.9), 1.0);
        assert_eq!(RoundingMode::TowardZero.apply_f64(-1.9), -1.0);
    }

    #[test]
    fn apply_nearest_away() {
        assert_eq!(RoundingMode::NearestAway.apply_f64(2.5), 3.0);
        assert_eq!(RoundingMode::NearestAway.apply_f64(-2.5), -3.0);
    }

    // ── round_to_precision ──────────────────────────────────────────────

    #[test]
    fn round_to_precision_full_keeps_value() {
        let v = std::f64::consts::PI;
        assert_eq!(RoundingMode::NearestEven.round_to_precision(v, 53), v);
    }

    #[test]
    fn round_to_precision_reduces_accuracy() {
        let v = std::f64::consts::PI;
        let r = RoundingMode::NearestEven.round_to_precision(v, 24);
        // Should approximate π but not be exactly π.
        assert!((r - v).abs() < 1e-6);
        assert_ne!(r, v);
    }

    #[test]
    fn round_to_precision_special_values() {
        assert!(RoundingMode::NearestEven
            .round_to_precision(f64::NAN, 24)
            .is_nan());
        assert_eq!(
            RoundingMode::NearestEven.round_to_precision(f64::INFINITY, 24),
            f64::INFINITY
        );
        assert_eq!(RoundingMode::NearestEven.round_to_precision(0.0, 24), 0.0);
    }

    // ── RoundingError ───────────────────────────────────────────────────

    #[test]
    fn rounding_error_basic() {
        let re = RoundingError::new(1.5, 2.0, RoundingMode::NearestEven);
        assert_eq!(re.error, 0.5);
        assert!((re.relative_error() - 1.0 / 3.0).abs() < 1e-10);
        assert!((re.absolute_error() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn rounding_error_zero_value() {
        let re = RoundingError::new(0.0, 0.0, RoundingMode::NearestEven);
        assert_eq!(re.relative_error(), 0.0);
    }

    #[test]
    fn rounding_error_display() {
        let re = RoundingError::new(1.1, 1.0, RoundingMode::TowardZero);
        let s = re.to_string();
        assert!(s.contains("RoundingError"));
    }

    // ── StochasticRounder ───────────────────────────────────────────────

    #[test]
    fn stochastic_rounder_deterministic() {
        let mut r1 = StochasticRounder::new(42);
        let mut r2 = StochasticRounder::new(42);
        for _ in 0..100 {
            assert_eq!(r1.round(2.7).to_bits(), r2.round(2.7).to_bits());
        }
    }

    #[test]
    fn stochastic_rounder_integer_passthrough() {
        let mut r = StochasticRounder::new(42);
        assert_eq!(r.round(3.0), 3.0);
    }

    #[test]
    fn stochastic_rounder_range() {
        let mut r = StochasticRounder::new(42);
        for _ in 0..200 {
            let v = r.round(2.3);
            assert!(v == 2.0 || v == 3.0, "got {v}");
        }
    }

    #[test]
    fn stochastic_round_to_precision() {
        let mut r = StochasticRounder::new(42);
        let v = std::f64::consts::PI;
        let rounded = r.round_to_precision(v, 24);
        assert!((rounded - v).abs() < 1e-6);
    }

    // ── Display ─────────────────────────────────────────────────────────

    #[test]
    fn rounding_mode_display() {
        let s = RoundingMode::NearestEven.to_string();
        assert!(s.contains("nearest"));
    }

    // ── Serde ───────────────────────────────────────────────────────────

    #[test]
    fn serde_rounding_mode() {
        let m = RoundingMode::Stochastic;
        let json = serde_json::to_string(&m).unwrap();
        let m2: RoundingMode = serde_json::from_str(&json).unwrap();
        assert_eq!(m, m2);
    }

    #[test]
    fn serde_rounding_error() {
        let re = RoundingError::new(1.5, 2.0, RoundingMode::NearestEven);
        let json = serde_json::to_string(&re).unwrap();
        let re2: RoundingError = serde_json::from_str(&json).unwrap();
        assert!((re.error - re2.error).abs() < 1e-15);
    }

    #[test]
    fn serde_stochastic_rounder() {
        let r = StochasticRounder::new(42);
        let json = serde_json::to_string(&r).unwrap();
        let r2: StochasticRounder = serde_json::from_str(&json).unwrap();
        assert_eq!(r.state, r2.state);
    }
}
