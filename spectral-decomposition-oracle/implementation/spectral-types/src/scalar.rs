//! Scalar type abstractions for numerical computation.
//!
//! Defines the [`Scalar`] trait extending `num_traits::Float` with additional
//! requirements for serialization, thread safety, and epsilon-aware comparisons.

use num_complex::Complex;
use num_traits::Float;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::iter::Sum;
use std::ops::{AddAssign, DivAssign, MulAssign, SubAssign};

/// Core scalar trait for all numerical computations.
pub trait Scalar:
    Float
    + Send
    + Sync
    + Copy
    + fmt::Debug
    + fmt::Display
    + Default
    + Serialize
    + for<'de> Deserialize<'de>
    + Sum
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + 'static
{
    /// Machine epsilon for this type.
    fn machine_epsilon() -> Self;

    /// A safe tolerance for comparisons (typically sqrt(machine_epsilon)).
    fn default_tolerance() -> Self;

    /// The "zero threshold" below which values are treated as zero.
    fn zero_threshold() -> Self;

    /// Convert from f64 lossily.
    fn from_f64_lossy(v: f64) -> Self;

    /// Convert to f64 lossily.
    fn to_f64_lossy(self) -> f64;

    /// Check if two values are approximately equal within tolerance.
    fn approx_eq(self, other: Self, tol: Self) -> bool {
        (self - other).abs() <= tol
    }

    /// Check if this value is approximately zero.
    fn is_approx_zero(self, tol: Self) -> bool {
        self.abs() <= tol
    }

    /// Safe division that returns None on divide-by-zero or non-finite result.
    fn safe_div(self, denom: Self) -> Option<Self> {
        if denom.is_approx_zero(Self::zero_threshold()) {
            None
        } else {
            let result = self / denom;
            if result.is_finite() {
                Some(result)
            } else {
                None
            }
        }
    }

    /// Clamp to a finite range, replacing NaN with zero.
    fn sanitize(self) -> Self {
        if self.is_nan() {
            Self::zero()
        } else if self == Self::infinity() {
            Self::max_value()
        } else if self == Self::neg_infinity() {
            Self::min_value()
        } else {
            self
        }
    }

    /// Returns the maximum of two values, handling NaN (NaN loses).
    fn ordered_max(self, other: Self) -> Self {
        if self.is_nan() {
            other
        } else if other.is_nan() {
            self
        } else if self >= other {
            self
        } else {
            other
        }
    }

    /// Returns the minimum of two values, handling NaN (NaN loses).
    fn ordered_min(self, other: Self) -> Self {
        if self.is_nan() {
            other
        } else if other.is_nan() {
            self
        } else if self <= other {
            self
        } else {
            other
        }
    }

    /// Square of the value.
    fn sq(self) -> Self {
        self * self
    }

    /// Cube of the value.
    fn cube(self) -> Self {
        self * self * self
    }

    /// Sign function: -1, 0, or 1.
    fn signum_int(self) -> i8 {
        if self.is_approx_zero(Self::zero_threshold()) {
            0
        } else if self > Self::zero() {
            1
        } else {
            -1
        }
    }

    /// Relative error between two values.
    fn relative_error(self, reference: Self) -> Self {
        if reference.is_approx_zero(Self::zero_threshold()) {
            self.abs()
        } else {
            (self - reference).abs() / reference.abs()
        }
    }
}

impl Scalar for f64 {
    fn machine_epsilon() -> Self {
        f64::EPSILON
    }
    fn default_tolerance() -> Self {
        1e-8
    }
    fn zero_threshold() -> Self {
        1e-15
    }
    fn from_f64_lossy(v: f64) -> Self {
        v
    }
    fn to_f64_lossy(self) -> f64 {
        self
    }
}

impl Scalar for f32 {
    fn machine_epsilon() -> Self {
        f32::EPSILON
    }
    fn default_tolerance() -> Self {
        1e-5
    }
    fn zero_threshold() -> Self {
        1e-7
    }
    fn from_f64_lossy(v: f64) -> Self {
        v as f32
    }
    fn to_f64_lossy(self) -> f64 {
        self as f64
    }
}

/// Wrapper around `num_complex::Complex<T>` with Serialize/Deserialize.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct ComplexScalar<T: Scalar> {
    pub re: T,
    pub im: T,
}

impl<T: Scalar> ComplexScalar<T> {
    pub fn new(re: T, im: T) -> Self {
        Self { re, im }
    }

    pub fn real(re: T) -> Self {
        Self {
            re,
            im: T::zero(),
        }
    }

    pub fn imaginary(im: T) -> Self {
        Self {
            re: T::zero(),
            im,
        }
    }

    pub fn zero() -> Self {
        Self {
            re: T::zero(),
            im: T::zero(),
        }
    }

    pub fn one() -> Self {
        Self {
            re: T::one(),
            im: T::zero(),
        }
    }

    pub fn i() -> Self {
        Self {
            re: T::zero(),
            im: T::one(),
        }
    }

    pub fn conjugate(self) -> Self {
        Self {
            re: self.re,
            im: T::zero() - self.im,
        }
    }

    pub fn norm_sqr(self) -> T {
        self.re * self.re + self.im * self.im
    }

    pub fn norm(self) -> T {
        self.norm_sqr().sqrt()
    }

    pub fn arg(self) -> T {
        self.im.atan2(self.re)
    }

    pub fn inverse(self) -> Option<Self> {
        let ns = self.norm_sqr();
        if ns.is_approx_zero(T::zero_threshold()) {
            None
        } else {
            Some(Self {
                re: self.re / ns,
                im: (T::zero() - self.im) / ns,
            })
        }
    }

    pub fn to_num_complex(self) -> Complex<T> {
        Complex::new(self.re, self.im)
    }

    pub fn from_num_complex(c: Complex<T>) -> Self {
        Self { re: c.re, im: c.im }
    }

    pub fn is_real(self, tol: T) -> bool {
        self.im.abs() <= tol
    }

    pub fn add(self, other: Self) -> Self {
        Self {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }

    pub fn sub(self, other: Self) -> Self {
        Self {
            re: self.re - other.re,
            im: self.im - other.im,
        }
    }

    pub fn mul(self, other: Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }

    pub fn scale(self, s: T) -> Self {
        Self {
            re: self.re * s,
            im: self.im * s,
        }
    }
}

impl<T: Scalar> Default for ComplexScalar<T> {
    fn default() -> Self {
        Self::zero()
    }
}

impl<T: Scalar> fmt::Display for ComplexScalar<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.im >= T::zero() {
            write!(f, "{}+{}i", self.re, self.im)
        } else {
            write!(f, "{}{}i", self.re, self.im)
        }
    }
}

/// NaN sentinel value used for missing features.
pub const NAN_SENTINEL: f64 = -999999.0;

/// Check if a value is the NaN sentinel.
pub fn is_nan_sentinel(v: f64) -> bool {
    (v - NAN_SENTINEL).abs() < 1.0
}

/// Replace NaN with sentinel.
pub fn nan_to_sentinel(v: f64) -> f64 {
    if v.is_nan() {
        NAN_SENTINEL
    } else {
        v
    }
}

/// Replace sentinel with NaN.
pub fn sentinel_to_nan(v: f64) -> f64 {
    if is_nan_sentinel(v) {
        f64::NAN
    } else {
        v
    }
}

/// Compute ordered comparison of f64 slices (lexicographic).
pub fn lexicographic_compare(a: &[f64], b: &[f64]) -> std::cmp::Ordering {
    for (x, y) in a.iter().zip(b.iter()) {
        match ordered_float::OrderedFloat(*x).cmp(&ordered_float::OrderedFloat(*y)) {
            std::cmp::Ordering::Equal => continue,
            other => return other,
        }
    }
    a.len().cmp(&b.len())
}

/// Kahan summation for improved numerical accuracy.
pub fn kahan_sum<T: Scalar>(values: &[T]) -> T {
    let mut sum = T::zero();
    let mut c = T::zero();
    for &v in values {
        let y = v - c;
        let t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    sum
}

/// Compute the L2 norm of a slice.
pub fn l2_norm<T: Scalar>(values: &[T]) -> T {
    let sum_sq: T = values.iter().copied().map(|v| v * v).fold(T::zero(), |a, b| a + b);
    sum_sq.sqrt()
}

/// Normalize a slice in-place to unit L2 norm. Returns the original norm.
pub fn normalize_l2<T: Scalar>(values: &mut [T]) -> T {
    let n = l2_norm(values);
    if !n.is_approx_zero(T::zero_threshold()) {
        for v in values.iter_mut() {
            *v = *v / n;
        }
    }
    n
}

/// Dot product of two slices.
pub fn dot_product<T: Scalar>(a: &[T], b: &[T]) -> T {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| x * y)
        .fold(T::zero(), |acc, v| acc + v)
}

/// Linear interpolation.
pub fn lerp<T: Scalar>(a: T, b: T, t: T) -> T {
    a + (b - a) * t
}

/// Clamp value to [lo, hi].
pub fn clamp<T: Scalar>(val: T, lo: T, hi: T) -> T {
    if val < lo {
        lo
    } else if val > hi {
        hi
    } else {
        val
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f64_scalar_epsilon() {
        assert!(f64::machine_epsilon() > 0.0);
        assert!(f64::default_tolerance() > f64::machine_epsilon());
    }

    #[test]
    fn test_f32_scalar_epsilon() {
        assert!(f32::machine_epsilon() > 0.0);
        assert!(f32::default_tolerance() > f32::machine_epsilon());
    }

    #[test]
    fn test_approx_eq() {
        assert!(1.0_f64.approx_eq(1.0 + 1e-10, 1e-8));
        assert!(!1.0_f64.approx_eq(2.0, 1e-8));
    }

    #[test]
    fn test_safe_div() {
        assert_eq!(6.0_f64.safe_div(3.0), Some(2.0));
        assert_eq!(1.0_f64.safe_div(0.0), None);
    }

    #[test]
    fn test_sanitize() {
        assert_eq!(f64::NAN.sanitize(), 0.0);
        assert_eq!(f64::INFINITY.sanitize(), f64::MAX);
        assert_eq!(42.0_f64.sanitize(), 42.0);
    }

    #[test]
    fn test_ordered_max_min() {
        assert_eq!(3.0_f64.ordered_max(5.0), 5.0);
        assert_eq!(3.0_f64.ordered_min(5.0), 3.0);
        assert_eq!(f64::NAN.ordered_max(5.0), 5.0);
        assert_eq!(f64::NAN.ordered_min(5.0), 5.0);
    }

    #[test]
    fn test_complex_basic() {
        let c = ComplexScalar::new(3.0_f64, 4.0);
        assert!((c.norm() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_complex_conjugate() {
        let c = ComplexScalar::new(1.0_f64, 2.0);
        let conj = c.conjugate();
        assert_eq!(conj.re, 1.0);
        assert_eq!(conj.im, -2.0);
    }

    #[test]
    fn test_complex_mul() {
        let a = ComplexScalar::new(1.0_f64, 2.0);
        let b = ComplexScalar::new(3.0, 4.0);
        let c = a.mul(b);
        assert!((c.re - (-5.0)).abs() < 1e-10);
        assert!((c.im - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_complex_inverse() {
        let c = ComplexScalar::new(1.0_f64, 1.0);
        let inv = c.inverse().unwrap();
        let prod = c.mul(inv);
        assert!(prod.re.approx_eq(1.0, 1e-10));
        assert!(prod.im.approx_eq(0.0, 1e-10));
    }

    #[test]
    fn test_nan_sentinel() {
        assert!(is_nan_sentinel(NAN_SENTINEL));
        assert!(!is_nan_sentinel(42.0));
        assert_eq!(nan_to_sentinel(f64::NAN), NAN_SENTINEL);
        assert!(sentinel_to_nan(NAN_SENTINEL).is_nan());
    }

    #[test]
    fn test_kahan_sum() {
        let vals: Vec<f64> = (0..1000).map(|_| 0.1).collect();
        let s = kahan_sum(&vals);
        assert!((s - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_l2_norm() {
        assert!((l2_norm(&[3.0_f64, 4.0]) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalize_l2() {
        let mut v = vec![3.0_f64, 4.0];
        let n = normalize_l2(&mut v);
        assert!((n - 5.0).abs() < 1e-10);
        assert!((l2_norm(&v) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_dot_product() {
        assert!((dot_product(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]) - 32.0).abs() < 1e-10);
    }

    #[test]
    fn test_lerp() {
        assert!((lerp(0.0_f64, 10.0, 0.5) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_lexicographic_compare() {
        assert_eq!(
            lexicographic_compare(&[1.0, 2.0], &[1.0, 3.0]),
            std::cmp::Ordering::Less
        );
    }

    #[test]
    fn test_relative_error() {
        assert!((2.0_f64.relative_error(1.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sq_cube() {
        assert_eq!(3.0_f64.sq(), 9.0);
        assert_eq!(2.0_f64.cube(), 8.0);
    }

    #[test]
    fn test_clamp_fn() {
        assert_eq!(clamp(5.0_f64, 0.0, 10.0), 5.0);
        assert_eq!(clamp(-1.0_f64, 0.0, 10.0), 0.0);
        assert_eq!(clamp(15.0_f64, 0.0, 10.0), 10.0);
    }
}
