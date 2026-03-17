//! IEEE 754 bit-level types and classification.
//!
//! This module provides types and functions for working with IEEE 754
//! floating-point numbers at the bit level: format descriptors, bit
//! decomposition / recomposition, classification, and next-representable-value
//! utilities.

use serde::{Deserialize, Serialize};
use std::fmt;

// ─── Ieee754Format ──────────────────────────────────────────────────────────

/// Describes a standard IEEE 754 binary interchange format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Ieee754Format {
    /// IEEE 754 binary16 (half precision).
    Binary16,
    /// IEEE 754 binary32 (single precision).
    Binary32,
    /// IEEE 754 binary64 (double precision).
    Binary64,
    /// IEEE 754 binary128 (quad precision).
    Binary128,
    /// x87 80-bit extended precision.
    Extended80,
}

impl Ieee754Format {
    /// Total number of bits in the encoding.
    pub fn total_bits(self) -> u32 {
        match self {
            Self::Binary16 => 16,
            Self::Binary32 => 32,
            Self::Binary64 => 64,
            Self::Binary128 => 128,
            Self::Extended80 => 80,
        }
    }

    /// Number of exponent bits.
    pub fn exponent_bits(self) -> u32 {
        match self {
            Self::Binary16 => 5,
            Self::Binary32 => 8,
            Self::Binary64 => 11,
            Self::Binary128 => 15,
            Self::Extended80 => 15,
        }
    }

    /// Number of stored significand (mantissa) bits (excluding the implicit
    /// leading bit for the non-extended formats).
    pub fn significand_bits(self) -> u32 {
        match self {
            Self::Binary16 => 10,
            Self::Binary32 => 23,
            Self::Binary64 => 52,
            Self::Binary128 => 112,
            Self::Extended80 => 64, // explicit integer bit
        }
    }

    /// Exponent bias (2^(e-1) - 1).
    pub fn bias(self) -> i32 {
        (1 << (self.exponent_bits() - 1)) - 1
    }

    /// Minimum (most negative) exponent for normal numbers.
    pub fn min_exponent(self) -> i32 {
        1 - self.bias()
    }

    /// Maximum exponent for normal numbers.
    pub fn max_exponent(self) -> i32 {
        self.bias()
    }

    /// Machine epsilon: 2^{-(p-1)} where p is the total significand
    /// precision (stored bits + 1 implicit bit, except Extended80 which
    /// stores the integer bit explicitly).
    pub fn machine_epsilon(self) -> f64 {
        let p = match self {
            Self::Extended80 => self.significand_bits(), // integer bit is explicit
            _ => self.significand_bits() + 1,
        };
        2.0_f64.powi(-(p as i32 - 1))
    }

    /// Smallest positive normal number.
    pub fn min_positive(self) -> f64 {
        2.0_f64.powi(self.min_exponent())
    }

    /// Largest finite representable value (approximation for formats wider
    /// than f64).
    pub fn max_value(self) -> f64 {
        let p = match self {
            Self::Extended80 => self.significand_bits(),
            _ => self.significand_bits() + 1,
        };
        let emax = self.max_exponent();
        (2.0 - 2.0_f64.powi(-(p as i32 - 1))) * 2.0_f64.powi(emax)
    }

    /// Smallest positive subnormal number.
    pub fn min_subnormal(self) -> f64 {
        let p = match self {
            Self::Extended80 => self.significand_bits(),
            _ => self.significand_bits() + 1,
        };
        2.0_f64.powi(self.min_exponent() - (p as i32 - 1))
    }
}

impl fmt::Display for Ieee754Format {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Binary16 => write!(f, "binary16"),
            Self::Binary32 => write!(f, "binary32"),
            Self::Binary64 => write!(f, "binary64"),
            Self::Binary128 => write!(f, "binary128"),
            Self::Extended80 => write!(f, "extended80"),
        }
    }
}

// ─── FpClass ────────────────────────────────────────────────────────────────

/// Detailed floating-point classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FpClass {
    /// Positive or negative zero.
    Zero,
    /// Subnormal (denormalised) number.
    Subnormal,
    /// Normal number.
    Normal,
    /// Positive or negative infinity.
    Infinite,
    /// Quiet NaN.
    NaN,
    /// Signalling NaN.
    SignalingNaN,
}

impl fmt::Display for FpClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Zero => write!(f, "zero"),
            Self::Subnormal => write!(f, "subnormal"),
            Self::Normal => write!(f, "normal"),
            Self::Infinite => write!(f, "infinite"),
            Self::NaN => write!(f, "NaN"),
            Self::SignalingNaN => write!(f, "sNaN"),
        }
    }
}

// ─── Ieee754Value ───────────────────────────────────────────────────────────

/// Bit-level decomposition of an IEEE 754 value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Ieee754Value {
    /// The originating format.
    pub format: Ieee754Format,
    /// Sign bit (0 = positive, 1 = negative).
    pub sign: u8,
    /// Biased exponent field (raw bits).
    pub exponent_raw: u32,
    /// Significand / mantissa field (raw bits, stored portion only).
    pub significand_raw: u64,
    /// Unbiased (true) exponent.
    pub exponent: i32,
    /// Floating-point classification.
    pub class: FpClass,
}

impl Ieee754Value {
    /// Whether the value is negative (sign bit = 1).
    pub fn is_negative(&self) -> bool {
        self.sign == 1
    }

    /// Whether this value is a NaN of either kind.
    pub fn is_nan(&self) -> bool {
        matches!(self.class, FpClass::NaN | FpClass::SignalingNaN)
    }

    /// Whether this value is ±0.
    pub fn is_zero(&self) -> bool {
        self.class == FpClass::Zero
    }

    /// Whether this value is ±∞.
    pub fn is_infinite(&self) -> bool {
        self.class == FpClass::Infinite
    }

    /// Whether this value is subnormal.
    pub fn is_subnormal(&self) -> bool {
        self.class == FpClass::Subnormal
    }

    /// Reconstruct the `f64` value (only meaningful for Binary64).
    pub fn to_f64(&self) -> Option<f64> {
        if self.format != Ieee754Format::Binary64 {
            return None;
        }
        Some(compose_f64(
            self.sign,
            self.exponent_raw,
            self.significand_raw,
        ))
    }

    /// Reconstruct the `f32` value (only meaningful for Binary32).
    pub fn to_f32(&self) -> Option<f32> {
        if self.format != Ieee754Format::Binary32 {
            return None;
        }
        Some(compose_f32(
            self.sign,
            self.exponent_raw,
            self.significand_raw as u32,
        ))
    }
}

impl fmt::Display for Ieee754Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let sign_char = if self.sign == 1 { '-' } else { '+' };
        write!(
            f,
            "[{} sign={} exp_raw={:#x} sig_raw={:#x} exp={} class={}]",
            self.format,
            sign_char,
            self.exponent_raw,
            self.significand_raw,
            self.exponent,
            self.class
        )
    }
}

// ─── Decomposition / composition helpers ────────────────────────────────────

/// Decompose an `f64` into its IEEE 754 binary64 fields.
pub fn decompose_f64(value: f64) -> Ieee754Value {
    let bits = value.to_bits();
    let sign = ((bits >> 63) & 1) as u8;
    let exponent_raw = ((bits >> 52) & 0x7FF) as u32;
    let significand_raw = bits & 0x000F_FFFF_FFFF_FFFF;

    let (class, exponent) = classify_fields_64(exponent_raw, significand_raw);

    Ieee754Value {
        format: Ieee754Format::Binary64,
        sign,
        exponent_raw,
        significand_raw,
        exponent,
        class,
    }
}

/// Decompose an `f32` into its IEEE 754 binary32 fields.
pub fn decompose_f32(value: f32) -> Ieee754Value {
    let bits = value.to_bits();
    let sign = ((bits >> 31) & 1) as u8;
    let exponent_raw = ((bits >> 23) & 0xFF) as u32;
    let significand_raw = (bits & 0x007F_FFFF) as u64;

    let (class, exponent) = classify_fields_32(exponent_raw, significand_raw as u32);

    Ieee754Value {
        format: Ieee754Format::Binary32,
        sign,
        exponent_raw,
        significand_raw,
        exponent,
        class,
    }
}

/// Recompose an `f64` from its IEEE 754 binary64 fields.
pub fn compose_f64(sign: u8, exponent_raw: u32, significand_raw: u64) -> f64 {
    let bits = ((sign as u64) << 63)
        | ((exponent_raw as u64 & 0x7FF) << 52)
        | (significand_raw & 0x000F_FFFF_FFFF_FFFF);
    f64::from_bits(bits)
}

/// Recompose an `f32` from its IEEE 754 binary32 fields.
pub fn compose_f32(sign: u8, exponent_raw: u32, significand_raw: u32) -> f32 {
    let bits =
        ((sign as u32) << 31) | ((exponent_raw & 0xFF) << 23) | (significand_raw & 0x007F_FFFF);
    f32::from_bits(bits)
}

// ─── Classification helpers (private) ───────────────────────────────────────

fn classify_fields_64(exponent_raw: u32, significand_raw: u64) -> (FpClass, i32) {
    let bias = 1023_i32;
    if exponent_raw == 0 {
        if significand_raw == 0 {
            (FpClass::Zero, 0)
        } else {
            (FpClass::Subnormal, 1 - bias)
        }
    } else if exponent_raw == 0x7FF {
        if significand_raw == 0 {
            (FpClass::Infinite, 0)
        } else {
            // Bit 51 of significand distinguishes quiet vs signalling.
            let is_quiet = (significand_raw >> 51) & 1 == 1;
            if is_quiet {
                (FpClass::NaN, 0)
            } else {
                (FpClass::SignalingNaN, 0)
            }
        }
    } else {
        (FpClass::Normal, exponent_raw as i32 - bias)
    }
}

fn classify_fields_32(exponent_raw: u32, significand_raw: u32) -> (FpClass, i32) {
    let bias = 127_i32;
    if exponent_raw == 0 {
        if significand_raw == 0 {
            (FpClass::Zero, 0)
        } else {
            (FpClass::Subnormal, 1 - bias)
        }
    } else if exponent_raw == 0xFF {
        if significand_raw == 0 {
            (FpClass::Infinite, 0)
        } else {
            let is_quiet = (significand_raw >> 22) & 1 == 1;
            if is_quiet {
                (FpClass::NaN, 0)
            } else {
                (FpClass::SignalingNaN, 0)
            }
        }
    } else {
        (FpClass::Normal, exponent_raw as i32 - bias)
    }
}

// ─── Public classification functions ────────────────────────────────────────

/// Classify an `f64` into a [`FpClass`].
pub fn classify_f64(value: f64) -> FpClass {
    decompose_f64(value).class
}

/// Classify an `f32` into a [`FpClass`].
pub fn classify_f32(value: f32) -> FpClass {
    decompose_f32(value).class
}

// ─── Next representable value ───────────────────────────────────────────────

/// Return the next IEEE 754 binary64 value toward +∞.
///
/// Follows the `nextUp` operation from IEEE 754-2008 §5.3.1.
pub fn next_up_f64(value: f64) -> f64 {
    if value.is_nan() {
        return value;
    }
    if value == f64::NEG_INFINITY {
        return -f64::MAX;
    }
    if value == 0.0 {
        // +0 or −0 → smallest positive subnormal
        return f64::from_bits(1);
    }
    let bits = value.to_bits();
    if value > 0.0 {
        f64::from_bits(bits + 1)
    } else {
        f64::from_bits(bits - 1)
    }
}

/// Return the next IEEE 754 binary64 value toward −∞.
///
/// Follows the `nextDown` operation from IEEE 754-2008 §5.3.1.
pub fn next_down_f64(value: f64) -> f64 {
    -next_up_f64(-value)
}

/// Return the next IEEE 754 binary32 value toward +∞.
pub fn next_up_f32(value: f32) -> f32 {
    if value.is_nan() {
        return value;
    }
    if value == f32::NEG_INFINITY {
        return -f32::MAX;
    }
    if value == 0.0 {
        return f32::from_bits(1);
    }
    let bits = value.to_bits();
    if value > 0.0 {
        f32::from_bits(bits + 1)
    } else {
        f32::from_bits(bits - 1)
    }
}

/// Return the next IEEE 754 binary32 value toward −∞.
pub fn next_down_f32(value: f32) -> f32 {
    -next_up_f32(-value)
}

// ─── Bit-pattern display helpers ────────────────────────────────────────────

/// Return a string showing the bit layout of an `f64`.
///
/// Format: `S EEEEEEEEEEE MMMM…MMMM` (1 + 11 + 52 bits).
pub fn f64_bit_string(value: f64) -> String {
    let bits = value.to_bits();
    let sign = (bits >> 63) & 1;
    let exp = (bits >> 52) & 0x7FF;
    let sig = bits & 0x000F_FFFF_FFFF_FFFF;
    format!("{:01b} {:011b} {:052b}", sign, exp, sig)
}

/// Return a string showing the bit layout of an `f32`.
///
/// Format: `S EEEEEEEE MMMMMMMMMMMMMMMMMMMMMMM` (1 + 8 + 23 bits).
pub fn f32_bit_string(value: f32) -> String {
    let bits = value.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = (bits >> 23) & 0xFF;
    let sig = bits & 0x007F_FFFF;
    format!("{:01b} {:08b} {:023b}", sign, exp, sig)
}

// ─── Errors ─────────────────────────────────────────────────────────────────

/// Errors that can arise from IEEE 754 operations in this module.
#[derive(Debug, Clone, thiserror::Error)]
pub enum Ieee754Error {
    /// The requested format is not supported for this operation.
    #[error("unsupported IEEE 754 format: {0}")]
    UnsupportedFormat(Ieee754Format),
    /// Attempted to recompose a value from invalid field widths.
    #[error("invalid field values for {format}: exponent_raw={exponent_raw}, significand_raw={significand_raw}")]
    InvalidFields {
        format: Ieee754Format,
        exponent_raw: u32,
        significand_raw: u64,
    },
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Format properties ───────────────────────────────────────────────

    #[test]
    fn format_total_bits() {
        assert_eq!(Ieee754Format::Binary16.total_bits(), 16);
        assert_eq!(Ieee754Format::Binary32.total_bits(), 32);
        assert_eq!(Ieee754Format::Binary64.total_bits(), 64);
        assert_eq!(Ieee754Format::Binary128.total_bits(), 128);
        assert_eq!(Ieee754Format::Extended80.total_bits(), 80);
    }

    #[test]
    fn format_exponent_bits() {
        assert_eq!(Ieee754Format::Binary16.exponent_bits(), 5);
        assert_eq!(Ieee754Format::Binary32.exponent_bits(), 8);
        assert_eq!(Ieee754Format::Binary64.exponent_bits(), 11);
    }

    #[test]
    fn format_significand_bits() {
        assert_eq!(Ieee754Format::Binary16.significand_bits(), 10);
        assert_eq!(Ieee754Format::Binary32.significand_bits(), 23);
        assert_eq!(Ieee754Format::Binary64.significand_bits(), 52);
    }

    #[test]
    fn format_bias() {
        assert_eq!(Ieee754Format::Binary16.bias(), 15);
        assert_eq!(Ieee754Format::Binary32.bias(), 127);
        assert_eq!(Ieee754Format::Binary64.bias(), 1023);
    }

    #[test]
    fn format_machine_epsilon_f64() {
        let eps = Ieee754Format::Binary64.machine_epsilon();
        assert!((eps - f64::EPSILON).abs() < 1e-30);
    }

    #[test]
    fn format_min_positive_f64() {
        let mp = Ieee754Format::Binary64.min_positive();
        // f64::MIN_POSITIVE == 2^{-1022}
        assert!((mp - f64::MIN_POSITIVE).abs() < 1e-320);
    }

    #[test]
    fn format_max_value_f64() {
        let mv = Ieee754Format::Binary64.max_value();
        assert!((mv - f64::MAX).abs() / f64::MAX < 1e-15);
    }

    #[test]
    fn format_min_subnormal_f64() {
        let ms = Ieee754Format::Binary64.min_subnormal();
        let expected = 5e-324_f64; // 2^{-1074}
        assert!((ms - expected).abs() <= expected);
    }

    // ── Decomposition / recomposition round-trips ───────────────────────

    #[test]
    fn decompose_compose_f64_round_trip() {
        for &v in &[
            0.0,
            -0.0,
            1.0,
            -1.0,
            f64::MIN_POSITIVE,
            f64::MAX,
            f64::INFINITY,
            f64::NEG_INFINITY,
            3.14159265,
        ] {
            let d = decompose_f64(v);
            let r = compose_f64(d.sign, d.exponent_raw, d.significand_raw);
            assert_eq!(v.to_bits(), r.to_bits(), "round-trip failed for {v}");
        }
    }

    #[test]
    fn decompose_compose_f32_round_trip() {
        for &v in &[
            0.0_f32,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::MAX,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ] {
            let d = decompose_f32(v);
            let r = compose_f32(d.sign, d.exponent_raw, d.significand_raw as u32);
            assert_eq!(v.to_bits(), r.to_bits(), "round-trip failed for {v}");
        }
    }

    // ── Classification ──────────────────────────────────────────────────

    #[test]
    fn classify_zero() {
        assert_eq!(classify_f64(0.0), FpClass::Zero);
        assert_eq!(classify_f64(-0.0), FpClass::Zero);
    }

    #[test]
    fn classify_normal() {
        assert_eq!(classify_f64(1.0), FpClass::Normal);
        assert_eq!(classify_f64(-42.5), FpClass::Normal);
    }

    #[test]
    fn classify_subnormal() {
        assert_eq!(classify_f64(5e-324), FpClass::Subnormal);
    }

    #[test]
    fn classify_infinite() {
        assert_eq!(classify_f64(f64::INFINITY), FpClass::Infinite);
        assert_eq!(classify_f64(f64::NEG_INFINITY), FpClass::Infinite);
    }

    #[test]
    fn classify_nan() {
        assert!(matches!(
            classify_f64(f64::NAN),
            FpClass::NaN | FpClass::SignalingNaN
        ));
    }

    // ── next_up / next_down ─────────────────────────────────────────────

    #[test]
    fn next_up_from_zero() {
        let nu = next_up_f64(0.0);
        assert!(nu > 0.0);
        assert_eq!(nu.to_bits(), 1);
    }

    #[test]
    fn next_down_from_zero() {
        let nd = next_down_f64(0.0);
        assert!(nd < 0.0);
    }

    #[test]
    fn next_up_positive() {
        let v = 1.0_f64;
        let nu = next_up_f64(v);
        assert!(nu > v);
        assert_eq!(nu.to_bits() - v.to_bits(), 1);
    }

    #[test]
    fn next_down_positive() {
        let v = 1.0_f64;
        let nd = next_down_f64(v);
        assert!(nd < v);
    }

    #[test]
    fn next_up_nan_is_nan() {
        assert!(next_up_f64(f64::NAN).is_nan());
    }

    #[test]
    fn next_up_neg_inf() {
        assert_eq!(next_up_f64(f64::NEG_INFINITY), -f64::MAX);
    }

    // ── Bit-string display ──────────────────────────────────────────────

    #[test]
    fn bit_string_zero() {
        let s = f64_bit_string(0.0);
        assert!(s.starts_with('0'));
        assert_eq!(s.replace(' ', "").len(), 64);
    }

    #[test]
    fn bit_string_one_f32() {
        let s = f32_bit_string(1.0);
        // 1.0 = 0 01111111 00000000000000000000000
        assert!(s.starts_with("0 01111111"));
    }

    // ── Display impls ───────────────────────────────────────────────────

    #[test]
    fn format_display() {
        assert_eq!(Ieee754Format::Binary64.to_string(), "binary64");
    }

    #[test]
    fn fpclass_display() {
        assert_eq!(FpClass::Subnormal.to_string(), "subnormal");
    }

    #[test]
    fn ieee754value_display() {
        let d = decompose_f64(1.0);
        let s = d.to_string();
        assert!(s.contains("binary64"));
    }

    // ── Ieee754Value helpers ────────────────────────────────────────────

    #[test]
    fn value_predicates() {
        assert!(decompose_f64(0.0).is_zero());
        assert!(!decompose_f64(0.0).is_negative());
        assert!(decompose_f64(-0.0).is_negative());
        assert!(decompose_f64(f64::INFINITY).is_infinite());
        assert!(decompose_f64(f64::NAN).is_nan());
        assert!(decompose_f64(5e-324).is_subnormal());
    }

    #[test]
    fn value_to_f64() {
        let v = 2.718281828;
        let d = decompose_f64(v);
        assert_eq!(d.to_f64(), Some(v));
    }

    #[test]
    fn value_to_f32() {
        let v = 2.5_f32;
        let d = decompose_f32(v);
        assert_eq!(d.to_f32(), Some(v));
    }

    // ── f32 classification ──────────────────────────────────────────────

    #[test]
    fn classify_f32_values() {
        assert_eq!(classify_f32(0.0_f32), FpClass::Zero);
        assert_eq!(classify_f32(1.0_f32), FpClass::Normal);
        assert_eq!(classify_f32(f32::INFINITY), FpClass::Infinite);
        // smallest f32 subnormal
        assert_eq!(classify_f32(f32::from_bits(1)), FpClass::Subnormal);
    }

    // ── f32 next_up / next_down ─────────────────────────────────────────

    #[test]
    fn next_up_f32_basic() {
        let nu = next_up_f32(1.0_f32);
        assert!(nu > 1.0);
    }

    #[test]
    fn next_down_f32_basic() {
        let nd = next_down_f32(1.0_f32);
        assert!(nd < 1.0);
    }

    // ── Serde round-trip ────────────────────────────────────────────────

    #[test]
    fn serde_ieee754_format() {
        let f = Ieee754Format::Binary64;
        let json = serde_json::to_string(&f).unwrap();
        let f2: Ieee754Format = serde_json::from_str(&json).unwrap();
        assert_eq!(f, f2);
    }

    #[test]
    fn serde_fp_class() {
        let c = FpClass::Subnormal;
        let json = serde_json::to_string(&c).unwrap();
        let c2: FpClass = serde_json::from_str(&json).unwrap();
        assert_eq!(c, c2);
    }

    #[test]
    fn serde_ieee754_value() {
        let v = decompose_f64(std::f64::consts::PI);
        let json = serde_json::to_string(&v).unwrap();
        let v2: Ieee754Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v, v2);
    }
}
