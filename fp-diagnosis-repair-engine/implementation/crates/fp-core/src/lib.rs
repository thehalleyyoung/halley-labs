//! Core floating-point analysis primitives for Penumbra.

use serde::{Deserialize, Serialize};

/// Double-double representation for extended precision arithmetic.
/// Represents a value as the unevaluated sum `hi + lo` where |lo| <= 0.5 * ulp(hi).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DoubleDouble {
    pub hi: f64,
    pub lo: f64,
}

impl DoubleDouble {
    pub fn new(hi: f64, lo: f64) -> Self {
        Self { hi, lo }
    }

    pub fn from_f64(x: f64) -> Self {
        Self { hi: x, lo: 0.0 }
    }

    pub fn to_f64(self) -> f64 {
        self.hi + self.lo
    }

    /// Two-sum: compute s = a + b exactly as s_hi + s_lo.
    pub fn two_sum(a: f64, b: f64) -> Self {
        let s = a + b;
        let v = s - a;
        let lo = (a - (s - v)) + (b - v);
        Self { hi: s, lo }
    }

    /// Two-product: compute p = a * b exactly as p_hi + p_lo (uses FMA).
    pub fn two_product(a: f64, b: f64) -> Self {
        let p = a * b;
        let lo = a.mul_add(b, -p);
        Self { hi: p, lo }
    }

    pub fn add(self, other: Self) -> Self {
        let s = Self::two_sum(self.hi, other.hi);
        let lo = s.lo + self.lo + other.lo;
        let r = Self::two_sum(s.hi, lo);
        r
    }

    pub fn sub(self, other: Self) -> Self {
        let neg = Self::new(-other.hi, -other.lo);
        self.add(neg)
    }

    pub fn mul(self, other: Self) -> Self {
        let p = Self::two_product(self.hi, other.hi);
        let lo = p.lo + (self.hi * other.lo + self.lo * other.hi);
        Self::two_sum(p.hi, lo)
    }

    pub fn div(self, other: Self) -> Self {
        let q1 = self.hi / other.hi;
        let r = self.sub(Self::from_f64(q1).mul(other));
        let q2 = r.hi / other.hi;
        Self::two_sum(q1, q2)
    }

    pub fn abs(self) -> Self {
        if self.hi < 0.0 {
            Self::new(-self.hi, -self.lo)
        } else {
            self
        }
    }

    pub fn sqrt(self) -> Self {
        if self.hi < 0.0 {
            return Self::new(f64::NAN, f64::NAN);
        }
        if self.hi == 0.0 && self.lo == 0.0 {
            return Self::new(0.0, 0.0);
        }
        let x = 1.0 / self.hi.sqrt();
        let ax = Self::from_f64(self.hi * x);
        let diff = self.sub(ax.mul(ax));
        let correction = diff.hi * x * 0.5;
        Self::two_sum(ax.hi, correction + ax.lo)
    }
}

impl Default for DoubleDouble {
    fn default() -> Self {
        Self::from_f64(0.0)
    }
}

/// Condition number of a floating-point computation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ConditionNumber {
    pub value: f64,
    pub is_estimated: bool,
}

impl ConditionNumber {
    pub fn new(value: f64) -> Self {
        Self {
            value,
            is_estimated: false,
        }
    }

    pub fn estimated(value: f64) -> Self {
        Self {
            value,
            is_estimated: true,
        }
    }
}

/// Unit in the last place for a given floating-point value.
pub fn ulp(x: f64) -> f64 {
    if x.is_nan() || x.is_infinite() {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::MIN_POSITIVE;
    }
    let bits = x.to_bits();
    let next = if x > 0.0 { bits + 1 } else { bits - 1 };
    (f64::from_bits(next) - x).abs()
}

/// Returns the next representable f64 above x.
pub fn next_up(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if x == f64::NEG_INFINITY {
        return f64::MIN;
    }
    if x == 0.0 || x == -0.0 {
        return f64::from_bits(1);
    }
    let bits = x.to_bits();
    if x > 0.0 {
        f64::from_bits(bits + 1)
    } else {
        f64::from_bits(bits - 1)
    }
}

/// Returns the next representable f64 below x.
pub fn next_down(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if x == f64::INFINITY {
        return f64::MAX;
    }
    if x == 0.0 || x == -0.0 {
        return f64::from_bits(1u64 | (1u64 << 63));
    }
    let bits = x.to_bits();
    if x > 0.0 {
        f64::from_bits(bits - 1)
    } else {
        f64::from_bits(bits + 1)
    }
}
