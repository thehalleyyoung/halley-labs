//! IEEE 754 bit-level manipulation and representation.

use serde::{Deserialize, Serialize};
use std::fmt;

/// IEEE 754 binary format specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Ieee754Format {
    /// 16-bit half precision (1-5-10)
    Binary16,
    /// 32-bit single precision (1-8-23)
    Binary32,
    /// 64-bit double precision (1-11-52)
    Binary64,
    /// 128-bit quadruple precision (1-15-112)
    Binary128,
    /// 80-bit extended precision (x86) (1-15-64)
    Extended80,
}

impl Ieee754Format {
    /// Total number of bits in the format.
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

    /// Number of significand bits (including implicit leading bit for normal numbers).
    pub fn significand_bits(self) -> u32 {
        match self {
            Self::Binary16 => 11,
            Self::Binary32 => 24,
            Self::Binary64 => 53,
            Self::Binary128 => 113,
            Self::Extended80 => 64,
        }
    }

    /// Number of stored (trailing) significand bits.
    pub fn trailing_significand_bits(self) -> u32 {
        match self {
            Self::Binary16 => 10,
            Self::Binary32 => 23,
            Self::Binary64 => 52,
            Self::Binary128 => 112,
            Self::Extended80 => 63,
        }
    }

    /// Exponent bias.
    pub fn exponent_bias(self) -> i32 {
        (1 << (self.exponent_bits() - 1)) - 1
    }

    /// Maximum finite exponent (unbiased).
    pub fn max_exponent(self) -> i32 {
        self.exponent_bias()
    }

    /// Minimum normal exponent (unbiased).
    pub fn min_exponent(self) -> i32 {
        1 - self.exponent_bias()
    }

    /// Machine epsilon: 2^(1 - significand_bits).
    pub fn machine_epsilon(self) -> f64 {
        let p = self.significand_bits() as f64;
        2.0_f64.powf(1.0 - p)
    }

    /// Unit roundoff: epsilon / 2.
    pub fn unit_roundoff(self) -> f64 {
        self.machine_epsilon() / 2.0
    }

    /// Smallest positive normal number.
    pub fn smallest_normal(self) -> f64 {
        2.0_f64.powi(self.min_exponent())
    }

    /// Largest finite number.
    pub fn largest_finite(self) -> f64 {
        let p = self.trailing_significand_bits() as f64;
        (2.0 - 2.0_f64.powf(-p)) * 2.0_f64.powi(self.max_exponent())
    }

    /// Smallest positive subnormal number.
    pub fn smallest_subnormal(self) -> f64 {
        2.0_f64.powi(self.min_exponent() - self.trailing_significand_bits() as i32)
    }
}

/// Bit-level representation of an IEEE 754 floating-point number.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Ieee754Bits {
    /// The raw bits stored as a u128 (supports up to Binary128).
    pub bits: u128,
    /// The format of this number.
    pub format: Ieee754Format,
}

impl Ieee754Bits {
    /// Create from raw bits.
    pub fn from_bits(bits: u128, format: Ieee754Format) -> Self {
        Self { bits, format }
    }

    /// Create from f32.
    pub fn from_f32(value: f32) -> Self {
        Self {
            bits: value.to_bits() as u128,
            format: Ieee754Format::Binary32,
        }
    }

    /// Create from f64.
    pub fn from_f64(value: f64) -> Self {
        Self {
            bits: value.to_bits() as u128,
            format: Ieee754Format::Binary64,
        }
    }

    /// Extract the sign bit (0 = positive, 1 = negative).
    pub fn sign_bit(self) -> u32 {
        let shift = self.format.total_bits() - 1;
        ((self.bits >> shift) & 1) as u32
    }

    /// True if the sign bit is set.
    pub fn is_negative(self) -> bool {
        self.sign_bit() == 1
    }

    /// Extract the biased exponent field.
    pub fn biased_exponent(self) -> u32 {
        let exp_bits = self.format.exponent_bits();
        let trail_bits = self.format.trailing_significand_bits();
        let mask = (1u128 << exp_bits) - 1;
        ((self.bits >> trail_bits) & mask) as u32
    }

    /// Extract the unbiased exponent.
    pub fn unbiased_exponent(self) -> i32 {
        let biased = self.biased_exponent() as i32;
        if biased == 0 {
            self.format.min_exponent()
        } else {
            biased - self.format.exponent_bias()
        }
    }

    /// Extract the trailing significand (mantissa) bits.
    pub fn trailing_significand(self) -> u128 {
        let trail_bits = self.format.trailing_significand_bits();
        let mask = (1u128 << trail_bits) - 1;
        self.bits & mask
    }

    /// Full significand including the implicit leading bit (for normal numbers).
    pub fn full_significand(self) -> u128 {
        let trailing = self.trailing_significand();
        let trail_bits = self.format.trailing_significand_bits();
        if self.biased_exponent() == 0 {
            trailing
        } else {
            trailing | (1u128 << trail_bits)
        }
    }

    /// Classify this floating-point number.
    pub fn classify(self) -> Ieee754Class {
        let exp_max = (1u32 << self.format.exponent_bits()) - 1;
        let biased = self.biased_exponent();
        let trailing = self.trailing_significand();

        if biased == exp_max {
            if trailing == 0 {
                if self.is_negative() {
                    Ieee754Class::NegativeInfinity
                } else {
                    Ieee754Class::PositiveInfinity
                }
            } else if trailing & (1u128 << (self.format.trailing_significand_bits() - 1)) != 0 {
                Ieee754Class::QuietNaN
            } else {
                Ieee754Class::SignalingNaN
            }
        } else if biased == 0 {
            if trailing == 0 {
                if self.is_negative() {
                    Ieee754Class::NegativeZero
                } else {
                    Ieee754Class::PositiveZero
                }
            } else if self.is_negative() {
                Ieee754Class::NegativeSubnormal
            } else {
                Ieee754Class::PositiveSubnormal
            }
        } else if self.is_negative() {
            Ieee754Class::NegativeNormal
        } else {
            Ieee754Class::PositiveNormal
        }
    }

    /// Convert to f64 (lossy for formats wider than Binary64).
    pub fn to_f64(self) -> f64 {
        match self.format {
            Ieee754Format::Binary32 => f32::from_bits(self.bits as u32) as f64,
            Ieee754Format::Binary64 => f64::from_bits(self.bits as u64),
            _ => {
                let sign = if self.is_negative() { -1.0 } else { 1.0 };
                let exp = self.unbiased_exponent();
                let sig = self.full_significand() as f64;
                let trail = self.format.trailing_significand_bits() as f64;
                sign * sig * 2.0_f64.powf(exp as f64 - trail)
            }
        }
    }

    /// Check if this is a NaN.
    pub fn is_nan(self) -> bool {
        matches!(self.classify(), Ieee754Class::QuietNaN | Ieee754Class::SignalingNaN)
    }

    /// Check if this is infinite.
    pub fn is_infinite(self) -> bool {
        matches!(
            self.classify(),
            Ieee754Class::PositiveInfinity | Ieee754Class::NegativeInfinity
        )
    }

    /// Check if this is zero.
    pub fn is_zero(self) -> bool {
        matches!(
            self.classify(),
            Ieee754Class::PositiveZero | Ieee754Class::NegativeZero
        )
    }

    /// Check if this is subnormal (denormalized).
    pub fn is_subnormal(self) -> bool {
        matches!(
            self.classify(),
            Ieee754Class::PositiveSubnormal | Ieee754Class::NegativeSubnormal
        )
    }

    /// Check if this is a normal number.
    pub fn is_normal(self) -> bool {
        matches!(
            self.classify(),
            Ieee754Class::PositiveNormal | Ieee754Class::NegativeNormal
        )
    }

    /// Return the next representable number away from zero.
    pub fn next_up(self) -> Self {
        if self.is_nan() || matches!(self.classify(), Ieee754Class::PositiveInfinity) {
            return self;
        }
        if self.is_zero() {
            return Self {
                bits: 1,
                format: self.format,
            };
        }
        if self.is_negative() {
            Self {
                bits: self.bits - 1,
                format: self.format,
            }
        } else {
            Self {
                bits: self.bits + 1,
                format: self.format,
            }
        }
    }

    /// Return the next representable number toward zero.
    pub fn next_down(self) -> Self {
        if self.is_nan() || matches!(self.classify(), Ieee754Class::NegativeInfinity) {
            return self;
        }
        if self.is_zero() {
            let total = self.format.total_bits();
            let neg_zero_bits = 1u128 << (total - 1);
            return Self {
                bits: neg_zero_bits | 1,
                format: self.format,
            };
        }
        if self.is_negative() {
            Self {
                bits: self.bits + 1,
                format: self.format,
            }
        } else {
            Self {
                bits: self.bits - 1,
                format: self.format,
            }
        }
    }

    /// Number of representable values between this and other (ULP distance).
    pub fn ulp_distance(self, other: Self) -> Option<u128> {
        if self.format != other.format || self.is_nan() || other.is_nan() {
            return None;
        }
        let a = self.to_ordered_int();
        let b = other.to_ordered_int();
        Some(if a > b { a - b } else { b - a })
    }

    fn to_ordered_int(self) -> u128 {
        let total = self.format.total_bits();
        let sign_mask = 1u128 << (total - 1);
        if self.bits & sign_mask != 0 {
            // Negative: flip all bits
            !self.bits & ((1u128 << total) - 1)
        } else {
            // Positive: flip sign bit
            self.bits | sign_mask
        }
    }
}

impl fmt::Debug for Ieee754Bits {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Ieee754Bits({:?}, sign={}, exp={}, sig=0x{:x}, class={:?})",
            self.format,
            self.sign_bit(),
            self.unbiased_exponent(),
            self.trailing_significand(),
            self.classify()
        )
    }
}

impl fmt::Display for Ieee754Bits {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_f64())
    }
}

/// Classification of IEEE 754 floating-point numbers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Ieee754Class {
    PositiveZero,
    NegativeZero,
    PositiveSubnormal,
    NegativeSubnormal,
    PositiveNormal,
    NegativeNormal,
    PositiveInfinity,
    NegativeInfinity,
    QuietNaN,
    SignalingNaN,
}

impl Ieee754Class {
    /// Whether this class represents a finite number.
    pub fn is_finite(self) -> bool {
        !matches!(
            self,
            Self::PositiveInfinity | Self::NegativeInfinity | Self::QuietNaN | Self::SignalingNaN
        )
    }

    /// Whether this class represents a numeric value (not NaN).
    pub fn is_numeric(self) -> bool {
        !matches!(self, Self::QuietNaN | Self::SignalingNaN)
    }

    /// Whether this class represents zero.
    pub fn is_zero(self) -> bool {
        matches!(self, Self::PositiveZero | Self::NegativeZero)
    }

    /// Whether this is a normal number.
    pub fn is_normal(self) -> bool {
        matches!(self, Self::PositiveNormal | Self::NegativeNormal)
    }

    /// Whether this is a subnormal number.
    pub fn is_subnormal(self) -> bool {
        matches!(self, Self::PositiveSubnormal | Self::NegativeSubnormal)
    }
}

/// Decomposed IEEE 754 number components.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Ieee754Decomposed {
    /// Sign: false = positive, true = negative.
    pub negative: bool,
    /// Unbiased exponent.
    pub exponent: i32,
    /// Significand as a normalized fraction in [1, 2) for normals.
    pub significand: f64,
    /// Format of the original number.
    pub format: Ieee754Format,
    /// Classification.
    pub class: Ieee754Class,
}

impl Ieee754Decomposed {
    /// Decompose an f64 into its IEEE 754 components.
    pub fn from_f64(value: f64) -> Self {
        let bits = Ieee754Bits::from_f64(value);
        let class = bits.classify();
        let negative = value.is_sign_negative();

        if value == 0.0 {
            return Self {
                negative,
                exponent: 0,
                significand: 0.0,
                format: Ieee754Format::Binary64,
                class,
            };
        }

        let abs_val = value.abs();
        if abs_val.is_infinite() || abs_val.is_nan() {
            return Self {
                negative,
                exponent: 0,
                significand: abs_val,
                format: Ieee754Format::Binary64,
                class,
            };
        }

        let exp = abs_val.log2().floor() as i32;
        let sig = abs_val / 2.0_f64.powi(exp);

        Self {
            negative,
            exponent: exp,
            significand: sig,
            format: Ieee754Format::Binary64,
            class,
        }
    }

    /// Reconstruct the f64 value.
    pub fn to_f64(self) -> f64 {
        let sign = if self.negative { -1.0 } else { 1.0 };
        sign * self.significand * 2.0_f64.powi(self.exponent)
    }

    /// Number of significant bits actually used.
    pub fn effective_precision_bits(self) -> u32 {
        if self.class.is_subnormal() {
            let bits = Ieee754Bits::from_f64(self.to_f64());
            let trailing = bits.trailing_significand();
            if trailing == 0 {
                return 0;
            }
            128 - trailing.leading_zeros() - trailing.trailing_zeros()
        } else {
            self.format.significand_bits()
        }
    }
}

/// Helper function: decompose an f64 into (sign, exponent, significand).
pub fn decompose_f64(value: f64) -> (bool, i32, u64) {
    let bits = value.to_bits();
    let sign = bits >> 63 != 0;
    let exp_bits = ((bits >> 52) & 0x7FF) as i32;
    let mantissa = bits & 0x000F_FFFF_FFFF_FFFF;

    if exp_bits == 0 {
        // Subnormal or zero
        (sign, -1022, mantissa)
    } else {
        // Normal
        (sign, exp_bits - 1023, mantissa | (1u64 << 52))
    }
}

/// Reconstruct an f64 from (sign, exponent, significand).
pub fn compose_f64(sign: bool, exponent: i32, significand: u64) -> f64 {
    let sign_bit = if sign { 1u64 << 63 } else { 0 };
    if significand & (1u64 << 52) == 0 {
        // Subnormal
        let bits = sign_bit | significand;
        f64::from_bits(bits)
    } else {
        let exp_bits = ((exponent + 1023) as u64) << 52;
        let mantissa = significand & 0x000F_FFFF_FFFF_FFFF;
        f64::from_bits(sign_bit | exp_bits | mantissa)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_properties() {
        let f64_fmt = Ieee754Format::Binary64;
        assert_eq!(f64_fmt.total_bits(), 64);
        assert_eq!(f64_fmt.exponent_bits(), 11);
        assert_eq!(f64_fmt.significand_bits(), 53);
        assert_eq!(f64_fmt.trailing_significand_bits(), 52);
        assert_eq!(f64_fmt.exponent_bias(), 1023);

        let f32_fmt = Ieee754Format::Binary32;
        assert_eq!(f32_fmt.total_bits(), 32);
        assert_eq!(f32_fmt.exponent_bits(), 8);
        assert_eq!(f32_fmt.significand_bits(), 24);
        assert_eq!(f32_fmt.exponent_bias(), 127);
    }

    #[test]
    fn test_machine_epsilon() {
        let eps = Ieee754Format::Binary64.machine_epsilon();
        assert!((eps - f64::EPSILON).abs() < 1e-30);

        let eps32 = Ieee754Format::Binary32.machine_epsilon();
        assert!((eps32 - f32::EPSILON as f64).abs() < 1e-15);
    }

    #[test]
    fn test_bits_from_f64() {
        let bits = Ieee754Bits::from_f64(1.0);
        assert_eq!(bits.sign_bit(), 0);
        assert_eq!(bits.unbiased_exponent(), 0);
        assert_eq!(bits.trailing_significand(), 0);
        assert!(bits.is_normal());
    }

    #[test]
    fn test_bits_classify() {
        assert!(Ieee754Bits::from_f64(0.0).is_zero());
        assert!(Ieee754Bits::from_f64(-0.0).is_zero());
        assert!(Ieee754Bits::from_f64(f64::INFINITY).is_infinite());
        assert!(Ieee754Bits::from_f64(f64::NEG_INFINITY).is_infinite());
        assert!(Ieee754Bits::from_f64(f64::NAN).is_nan());
        assert!(Ieee754Bits::from_f64(1.0).is_normal());
        assert!(Ieee754Bits::from_f64(5e-324).is_subnormal());
    }

    #[test]
    fn test_next_up_down() {
        let one = Ieee754Bits::from_f64(1.0);
        let next = one.next_up();
        let expected = 1.0 + f64::EPSILON;
        assert_eq!(next.to_f64(), expected);

        let prev = one.next_down();
        assert!(prev.to_f64() < 1.0);
        assert!(prev.to_f64() > 1.0 - f64::EPSILON);
    }

    #[test]
    fn test_ulp_distance() {
        let a = Ieee754Bits::from_f64(1.0);
        let b = Ieee754Bits::from_f64(1.0 + f64::EPSILON);
        assert_eq!(a.ulp_distance(b), Some(1));

        let c = Ieee754Bits::from_f64(2.0);
        let d = a.ulp_distance(c).unwrap();
        assert_eq!(d, (1u128 << 52)); // 2^52 ULPs between 1.0 and 2.0
    }

    #[test]
    fn test_decompose_compose() {
        let values = [0.0, 1.0, -1.0, 3.14, f64::MIN_POSITIVE, 1e308, -1e-308];
        for &v in &values {
            let (sign, exp, sig) = decompose_f64(v);
            let reconstructed = compose_f64(sign, exp, sig);
            assert_eq!(v.to_bits(), reconstructed.to_bits(), "Failed for {}", v);
        }
    }

    #[test]
    fn test_ieee754_decomposed() {
        let d = Ieee754Decomposed::from_f64(1.0);
        assert!(!d.negative);
        assert_eq!(d.exponent, 0);
        assert!((d.significand - 1.0).abs() < 1e-15);

        let d2 = Ieee754Decomposed::from_f64(-3.14);
        assert!(d2.negative);
        assert!((d2.to_f64() - (-3.14)).abs() < 1e-14);
    }

    #[test]
    fn test_f32_bits() {
        let bits = Ieee754Bits::from_f32(1.0f32);
        assert_eq!(bits.sign_bit(), 0);
        assert!(bits.is_normal());
        assert_eq!(bits.biased_exponent(), 127);
    }
}
