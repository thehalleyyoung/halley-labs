//! Exact rational arithmetic wrapper and dual-path verification.
//!
//! Provides [`RationalNum`] — a wrapper around `num_rational::BigRational` —
//! and [`DualPath`] / [`DualPathVerifier`] for running computations in both
//! f64 and exact rational arithmetic, then comparing results.

use crate::interval::{Interval, IntervalF64};
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{FromPrimitive, One, ToPrimitive, Zero};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::fmt;
use std::ops;

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// RationalNum
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Wrapper around `BigRational` for exact arithmetic.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RationalNum(pub BigRational);

impl RationalNum {
    /// Create a rational from two integers.
    pub fn new(numer: i64, denom: i64) -> Self {
        RationalNum(BigRational::new(BigInt::from(numer), BigInt::from(denom)))
    }

    /// Create rational from a `BigRational`.
    pub fn from_big(r: BigRational) -> Self {
        RationalNum(r)
    }

    /// Zero.
    pub fn zero() -> Self {
        RationalNum(BigRational::zero())
    }

    /// One.
    pub fn one() -> Self {
        RationalNum(BigRational::one())
    }

    /// Convert from f64 by decomposing into integer ratio.
    /// Uses a continued-fraction approach to find a rational approximation
    /// with denominator at most `max_denom`.
    pub fn from_f64_approx(value: f64, max_denom: i64) -> Self {
        if value == 0.0 {
            return RationalNum::zero();
        }
        if !value.is_finite() {
            return if value > 0.0 {
                RationalNum::new(i64::MAX, 1)
            } else {
                RationalNum::new(i64::MIN, 1)
            };
        }

        let negative = value < 0.0;
        let v = value.abs();

        // Stern-Brocot / continued fraction mediant approach
        let mut p0: i64 = 0;
        let mut q0: i64 = 1;
        let mut p1: i64 = 1;
        let mut q1: i64 = 0;
        let mut x = v;

        for _ in 0..64 {
            let a = x as i64;
            let p2 = a.saturating_mul(p1).saturating_add(p0);
            let q2 = a.saturating_mul(q1).saturating_add(q0);
            if q2 > max_denom || q2 < 0 {
                break;
            }
            p0 = p1;
            q0 = q1;
            p1 = p2;
            q1 = q2;
            let frac = x - a as f64;
            if frac.abs() < 1e-15 {
                break;
            }
            x = 1.0 / frac;
        }

        let numer = if negative { -p1 } else { p1 };
        RationalNum::new(numer, q1.max(1))
    }

    /// Convert from f64 using the exact bit representation.
    /// Returns the rational and an interval bounding the conversion error.
    pub fn from_f64_certified(value: f64) -> (Self, IntervalF64) {
        // Use BigRational::from_f64 for exact conversion
        let exact = BigRational::from_f64(value)
            .unwrap_or_else(|| BigRational::zero());
        let error_bound = Interval::from_f64_with_ulp(value);
        (RationalNum(exact), error_bound)
    }

    /// Convert to f64 (lossy).
    pub fn to_f64(&self) -> f64 {
        self.0.to_f64().unwrap_or(f64::NAN)
    }

    /// Convert to an interval bounding the true rational value.
    pub fn to_interval(&self) -> IntervalF64 {
        let v = self.to_f64();
        Interval::from_f64_with_ulp(v)
    }

    /// Absolute value.
    pub fn abs(&self) -> Self {
        if self.0 < BigRational::zero() {
            RationalNum(-self.0.clone())
        } else {
            self.clone()
        }
    }

    /// Check if zero.
    pub fn is_zero(&self) -> bool {
        self.0.is_zero()
    }

    /// Check if positive.
    pub fn is_positive(&self) -> bool {
        self.0 > BigRational::zero()
    }

    /// Check if negative.
    pub fn is_negative(&self) -> bool {
        self.0 < BigRational::zero()
    }

    /// Reciprocal. Returns `None` for zero.
    pub fn recip(&self) -> Option<Self> {
        if self.is_zero() { None } else { Some(RationalNum(self.0.recip())) }
    }

    /// Power for non-negative integer exponent.
    pub fn pow(&self, exp: u32) -> Self {
        let mut result = BigRational::one();
        for _ in 0..exp {
            result = result * &self.0;
        }
        RationalNum(result)
    }

    /// Maximum of two rationals.
    pub fn max(&self, other: &Self) -> Self {
        if self.0 >= other.0 { self.clone() } else { other.clone() }
    }

    /// Minimum of two rationals.
    pub fn min(&self, other: &Self) -> Self {
        if self.0 <= other.0 { self.clone() } else { other.clone() }
    }

    /// Numerator as i64 (may overflow for large values).
    pub fn numer_i64(&self) -> Option<i64> {
        self.0.numer().to_i64()
    }

    /// Denominator as i64 (may overflow for large values).
    pub fn denom_i64(&self) -> Option<i64> {
        self.0.denom().to_i64()
    }
}

impl Default for RationalNum {
    fn default() -> Self { RationalNum::zero() }
}

impl fmt::Display for RationalNum {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.0.numer(), self.0.denom())
    }
}

impl From<i64> for RationalNum {
    fn from(v: i64) -> Self { RationalNum::new(v, 1) }
}

impl From<f64> for RationalNum {
    fn from(v: f64) -> Self { RationalNum::from_f64_approx(v, 1_000_000) }
}

// Arithmetic operations
impl ops::Add for RationalNum {
    type Output = Self;
    fn add(self, rhs: Self) -> Self { RationalNum(&self.0 + &rhs.0) }
}

impl ops::Sub for RationalNum {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self { RationalNum(&self.0 - &rhs.0) }
}

impl ops::Mul for RationalNum {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self { RationalNum(&self.0 * &rhs.0) }
}

impl ops::Div for RationalNum {
    type Output = Self;
    fn div(self, rhs: Self) -> Self { RationalNum(&self.0 / &rhs.0) }
}

impl ops::Neg for RationalNum {
    type Output = Self;
    fn neg(self) -> Self { RationalNum(-self.0) }
}

impl ops::AddAssign for RationalNum {
    fn add_assign(&mut self, rhs: Self) { self.0 = &self.0 + &rhs.0; }
}

impl ops::SubAssign for RationalNum {
    fn sub_assign(&mut self, rhs: Self) { self.0 = &self.0 - &rhs.0; }
}

impl ops::MulAssign for RationalNum {
    fn mul_assign(&mut self, rhs: Self) { self.0 = &self.0 * &rhs.0; }
}

// Serde: serialize as string "numer/denom"
impl Serialize for RationalNum {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let s = format!("{}/{}", self.0.numer(), self.0.denom());
        serializer.serialize_str(&s)
    }
}

impl<'de> Deserialize<'de> for RationalNum {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        let parts: Vec<&str> = s.split('/').collect();
        if parts.len() != 2 {
            return Err(serde::de::Error::custom("expected numer/denom"));
        }
        let numer: BigInt = parts[0].parse().map_err(serde::de::Error::custom)?;
        let denom: BigInt = parts[1].parse().map_err(serde::de::Error::custom)?;
        Ok(RationalNum(BigRational::new(numer, denom)))
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// DualPath: run f64 and rational in parallel
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// A dual-path value carrying both an f64 approximation and the exact rational.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DualPath {
    pub float_value: f64,
    pub rational_value: RationalNum,
    pub error_bound: IntervalF64,
}

impl DualPath {
    /// Create a DualPath from an f64. The rational is derived via certified conversion.
    pub fn from_f64(value: f64) -> Self {
        let (rational, interval) = RationalNum::from_f64_certified(value);
        DualPath {
            float_value: value,
            rational_value: rational,
            error_bound: interval,
        }
    }

    /// Create a DualPath from an exact rational and its f64 approximation.
    pub fn from_rational(rational: RationalNum) -> Self {
        let float_value = rational.to_f64();
        let error_bound = Interval::from_f64_with_ulp(float_value);
        DualPath { float_value, rational_value: rational, error_bound }
    }

    /// Create from explicit values.
    pub fn new(float_value: f64, rational_value: RationalNum) -> Self {
        let error = (float_value - rational_value.to_f64()).abs();
        let error_bound = Interval::with_error(float_value, error.max(f64::EPSILON));
        DualPath { float_value, rational_value, error_bound }
    }

    /// Check that the f64 value lies within the error bound of the rational.
    pub fn is_consistent(&self) -> bool {
        self.error_bound.contains(self.float_value)
    }

    /// The discrepancy between f64 and rational (in absolute terms).
    pub fn discrepancy(&self) -> f64 {
        (self.float_value - self.rational_value.to_f64()).abs()
    }
}

impl fmt::Display for DualPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Dual(f64={:.8}, rat={}, err={})",
            self.float_value, self.rational_value, self.error_bound)
    }
}

impl ops::Add for DualPath {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        DualPath {
            float_value: self.float_value + rhs.float_value,
            rational_value: self.rational_value + rhs.rational_value,
            error_bound: self.error_bound + rhs.error_bound,
        }
    }
}

impl ops::Sub for DualPath {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        DualPath {
            float_value: self.float_value - rhs.float_value,
            rational_value: self.rational_value - rhs.rational_value,
            error_bound: self.error_bound - rhs.error_bound,
        }
    }
}

impl ops::Mul for DualPath {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        DualPath {
            float_value: self.float_value * rhs.float_value,
            rational_value: self.rational_value * rhs.rational_value,
            error_bound: self.error_bound * rhs.error_bound,
        }
    }
}

impl ops::Div for DualPath {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        DualPath {
            float_value: self.float_value / rhs.float_value,
            rational_value: self.rational_value / rhs.rational_value,
            error_bound: self.error_bound / rhs.error_bound,
        }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// DualPathVerifier
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Verification result from a dual-path comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    pub ordering_preserved: bool,
    pub max_discrepancy: f64,
    pub num_comparisons: usize,
    pub num_failures: usize,
    pub details: Vec<String>,
}

impl VerificationResult {
    pub fn is_valid(&self) -> bool {
        self.ordering_preserved && self.num_failures == 0
    }
}

impl fmt::Display for VerificationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Verification(preserved={}, max_disc={:.2e}, {}/{} ok)",
            self.ordering_preserved, self.max_discrepancy,
            self.num_comparisons - self.num_failures, self.num_comparisons)
    }
}

/// Verifier that checks whether f64 orderings are preserved in rational arithmetic.
#[derive(Debug, Clone)]
pub struct DualPathVerifier {
    pub tolerance: f64,
    comparisons: Vec<(DualPath, DualPath, Ordering)>,
}

impl DualPathVerifier {
    /// Create a new verifier with the given tolerance for discrepancy.
    pub fn new(tolerance: f64) -> Self {
        DualPathVerifier { tolerance, comparisons: Vec::new() }
    }

    /// Record a comparison: a < b, a == b, or a > b in f64,
    /// to be verified later in rational arithmetic.
    pub fn record_comparison(&mut self, a: DualPath, b: DualPath) {
        let ordering = a.float_value.partial_cmp(&b.float_value)
            .unwrap_or(Ordering::Equal);
        self.comparisons.push((a, b, ordering));
    }

    /// Verify all recorded comparisons.
    pub fn verify(&self) -> VerificationResult {
        let mut max_disc: f64 = 0.0;
        let mut failures = 0;
        let mut details = Vec::new();

        for (i, (a, b, f64_ordering)) in self.comparisons.iter().enumerate() {
            let disc_a = a.discrepancy();
            let disc_b = b.discrepancy();
            max_disc = max_disc.max(disc_a).max(disc_b);

            let rat_ordering = a.rational_value.0.cmp(&b.rational_value.0);
            if *f64_ordering != rat_ordering {
                // Check if the discrepancy is within tolerance
                let diff = (a.float_value - b.float_value).abs();
                if diff > self.tolerance {
                    failures += 1;
                    details.push(format!(
                        "comparison {}: f64 says {:?} but rational says {:?} (diff={:.2e})",
                        i, f64_ordering, rat_ordering, diff
                    ));
                }
            }
        }

        VerificationResult {
            ordering_preserved: failures == 0,
            max_discrepancy: max_disc,
            num_comparisons: self.comparisons.len(),
            num_failures: failures,
            details,
        }
    }

    /// Reset the verifier, clearing all recorded comparisons.
    pub fn reset(&mut self) {
        self.comparisons.clear();
    }

    pub fn num_recorded(&self) -> usize {
        self.comparisons.len()
    }
}

/// Verify that a sequence of f64 values maintains the same ordering
/// when converted to exact rational arithmetic.
pub fn verify_ordering(values: &[f64]) -> VerificationResult {
    let mut verifier = DualPathVerifier::new(1e-10);
    let duals: Vec<DualPath> = values.iter().map(|v| DualPath::from_f64(*v)).collect();
    for i in 0..duals.len() {
        for j in (i + 1)..duals.len() {
            verifier.record_comparison(duals[i].clone(), duals[j].clone());
        }
    }
    verifier.verify()
}

/// Compare two f64 values using exact rational arithmetic.
/// Returns the rational ordering.
pub fn exact_compare(a: f64, b: f64) -> Ordering {
    let (ra, _) = RationalNum::from_f64_certified(a);
    let (rb, _) = RationalNum::from_f64_certified(b);
    ra.0.cmp(&rb.0)
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Tests
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rational_new() {
        let r = RationalNum::new(1, 3);
        assert_eq!(r.0, BigRational::new(BigInt::from(1), BigInt::from(3)));
    }

    #[test]
    fn test_rational_arithmetic() {
        let a = RationalNum::new(1, 3);
        let b = RationalNum::new(1, 6);
        let c = a + b;
        assert_eq!(c, RationalNum::new(1, 2));
    }

    #[test]
    fn test_rational_sub() {
        let a = RationalNum::new(1, 2);
        let b = RationalNum::new(1, 3);
        let c = a - b;
        assert_eq!(c, RationalNum::new(1, 6));
    }

    #[test]
    fn test_rational_mul() {
        let a = RationalNum::new(2, 3);
        let b = RationalNum::new(3, 4);
        let c = a * b;
        assert_eq!(c, RationalNum::new(1, 2));
    }

    #[test]
    fn test_rational_div() {
        let a = RationalNum::new(1, 2);
        let b = RationalNum::new(1, 4);
        let c = a / b;
        assert_eq!(c, RationalNum::new(2, 1));
    }

    #[test]
    fn test_rational_from_f64_approx() {
        let r = RationalNum::from_f64_approx(0.5, 100);
        assert_eq!(r.to_f64(), 0.5);
    }

    #[test]
    fn test_rational_from_f64_certified() {
        let (r, iv) = RationalNum::from_f64_certified(0.1);
        assert!(iv.contains(0.1));
        assert!((r.to_f64() - 0.1).abs() < 1e-15);
    }

    #[test]
    fn test_rational_display() {
        let r = RationalNum::new(3, 7);
        let s = format!("{}", r);
        assert!(s.contains("3") && s.contains("7"));
    }

    #[test]
    fn test_rational_ordering() {
        let a = RationalNum::new(1, 3);
        let b = RationalNum::new(1, 2);
        assert!(a < b);
    }

    #[test]
    fn test_rational_serde_roundtrip() {
        let r = RationalNum::new(7, 13);
        let json = serde_json::to_string(&r).unwrap();
        let r2: RationalNum = serde_json::from_str(&json).unwrap();
        assert_eq!(r, r2);
    }

    #[test]
    fn test_dual_path_from_f64() {
        let dp = DualPath::from_f64(0.5);
        assert!(dp.is_consistent());
        assert!(dp.discrepancy() < 1e-10);
    }

    #[test]
    fn test_dual_path_arithmetic() {
        let a = DualPath::from_f64(1.0);
        let b = DualPath::from_f64(2.0);
        let c = a + b;
        assert!((c.float_value - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_dual_path_verifier() {
        let mut v = DualPathVerifier::new(1e-10);
        v.record_comparison(DualPath::from_f64(1.0), DualPath::from_f64(2.0));
        v.record_comparison(DualPath::from_f64(3.0), DualPath::from_f64(1.0));
        let result = v.verify();
        assert!(result.is_valid());
        assert_eq!(result.num_comparisons, 2);
    }

    #[test]
    fn test_verify_ordering_consistent() {
        let result = verify_ordering(&[1.0, 2.0, 3.0, 4.0]);
        assert!(result.is_valid());
    }

    #[test]
    fn test_exact_compare() {
        assert_eq!(exact_compare(0.1, 0.2), Ordering::Less);
        assert_eq!(exact_compare(0.5, 0.5), Ordering::Equal);
        assert_eq!(exact_compare(1.0, 0.5), Ordering::Greater);
    }

    #[test]
    fn test_rational_abs() {
        let r = RationalNum::new(-3, 4);
        assert_eq!(r.abs(), RationalNum::new(3, 4));
    }

    #[test]
    fn test_rational_pow() {
        let r = RationalNum::new(2, 3);
        let p = r.pow(2);
        assert_eq!(p, RationalNum::new(4, 9));
    }

    #[test]
    fn test_rational_recip() {
        let r = RationalNum::new(3, 7);
        let inv = r.recip().unwrap();
        assert_eq!(inv, RationalNum::new(7, 3));
        assert!(RationalNum::zero().recip().is_none());
    }
}
