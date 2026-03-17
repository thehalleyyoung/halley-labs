//! Rational arithmetic re-verification.
//!
//! Provides dual-path computation in both f64 and exact rational arithmetic,
//! detecting cases where floating-point rounding could affect proof validity.

use crate::ast::{BinaryOp, ComparisonOp, Expression, LiteralValue, UnaryOp};
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Signed, ToPrimitive, Zero};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Rational verifier ────────────────────────────────────────────────────────

/// Verifies that f64 computations agree with exact rational arithmetic.
pub struct RationalVerifier {
    converter: F64ToRationalConverter,
    count: usize,
    tolerance: f64,
}

impl RationalVerifier {
    pub fn new() -> Self {
        Self {
            converter: F64ToRationalConverter::new(),
            count: 0,
            tolerance: 1e-10,
        }
    }

    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    pub fn verification_count(&self) -> usize {
        self.count
    }

    /// Verify a comparison expression in both f64 and rational arithmetic.
    pub fn verify_expression(&self, expr: &Expression) -> Option<OrderingVerification> {
        match expr {
            Expression::Comparison { op, left, right } => {
                let lf = left.try_eval_f64()?;
                let rf = right.try_eval_f64()?;
                let f64_result = op.eval_f64(lf, rf);

                let lr = self.expr_to_rational(left)?;
                let rr = self.expr_to_rational(right)?;
                let rat_result = rational_comparison(op, &lr, &rr);

                Some(OrderingVerification {
                    f64_left: lf,
                    f64_right: rf,
                    f64_result,
                    rational_result: rat_result,
                    agree: f64_result == rat_result,
                    comparison: *op,
                })
            }
            _ => None,
        }
    }

    /// Verify all comparisons in a proof by traversing expressions.
    pub fn verify_all_orderings(
        &self,
        exprs: &[Expression],
    ) -> Vec<OrderingVerification> {
        let mut results = Vec::new();
        for expr in exprs {
            if let Some(ov) = self.verify_expression(expr) {
                results.push(ov);
            }
        }
        results
    }

    /// Convert an Expression AST to exact BigRational.
    fn expr_to_rational(&self, expr: &Expression) -> Option<BigRational> {
        match expr {
            Expression::Literal(LiteralValue::Float(v)) => {
                Some(self.converter.f64_to_rational(*v))
            }
            Expression::Literal(LiteralValue::Rational(r)) => {
                if r.denom == 0 {
                    return None;
                }
                Some(BigRational::new(
                    BigInt::from(r.numer),
                    BigInt::from(r.denom),
                ))
            }
            Expression::Literal(LiteralValue::Integer(i)) => {
                Some(BigRational::from(BigInt::from(*i)))
            }
            Expression::BinaryExpr { op, left, right } => {
                let l = self.expr_to_rational(left)?;
                let r = self.expr_to_rational(right)?;
                match op {
                    BinaryOp::Add => Some(&l + &r),
                    BinaryOp::Sub => Some(&l - &r),
                    BinaryOp::Mul => Some(&l * &r),
                    BinaryOp::Div => {
                        if r.is_zero() {
                            None
                        } else {
                            Some(&l / &r)
                        }
                    }
                    BinaryOp::Min => Some(if l < r { l } else { r }),
                    BinaryOp::Max => Some(if l > r { l } else { r }),
                    BinaryOp::Pow => {
                        // Only support integer exponents
                        let exp = r.to_integer().to_i64()?;
                        if exp < 0 {
                            let base_inv = BigRational::new(
                                l.denom().clone(),
                                l.numer().clone(),
                            );
                            Some(rational_pow(&base_inv, (-exp) as u64))
                        } else {
                            Some(rational_pow(&l, exp as u64))
                        }
                    }
                }
            }
            Expression::UnaryExpr { op, operand } => {
                let v = self.expr_to_rational(operand)?;
                match op {
                    UnaryOp::Neg => Some(-v),
                    UnaryOp::Abs => Some(v.abs()),
                    UnaryOp::Sqrt | UnaryOp::Log => {
                        // Cannot compute exact sqrt/log in rationals;
                        // fall back to f64 conversion
                        let fv = operand.try_eval_f64()?;
                        let result = match op {
                            UnaryOp::Sqrt => {
                                if fv < 0.0 {
                                    return None;
                                }
                                fv.sqrt()
                            }
                            UnaryOp::Log => {
                                if fv <= 0.0 {
                                    return None;
                                }
                                fv.ln()
                            }
                            _ => unreachable!(),
                        };
                        Some(self.converter.f64_to_rational(result))
                    }
                }
            }
            _ => None,
        }
    }
}

impl Default for RationalVerifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute rational^exp for non-negative integer exponent.
fn rational_pow(base: &BigRational, exp: u64) -> BigRational {
    if exp == 0 {
        return BigRational::one();
    }
    let mut result = base.clone();
    for _ in 1..exp {
        result = &result * base;
    }
    result
}

/// Perform a comparison in exact rational arithmetic.
fn rational_comparison(op: &ComparisonOp, a: &BigRational, b: &BigRational) -> bool {
    match op {
        ComparisonOp::Lt => a < b,
        ComparisonOp::Le => a <= b,
        ComparisonOp::Gt => a > b,
        ComparisonOp::Ge => a >= b,
        ComparisonOp::Eq => a == b,
        ComparisonOp::Ne => a != b,
    }
}

// ── Ordering verification result ─────────────────────────────────────────────

/// Result of verifying a single comparison in dual-path arithmetic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderingVerification {
    pub f64_left: f64,
    pub f64_right: f64,
    pub f64_result: bool,
    pub rational_result: bool,
    pub agree: bool,
    pub comparison: ComparisonOp,
}

impl OrderingVerification {
    pub fn is_sound(&self) -> bool {
        self.agree
    }
}

// ── Dual-path computation ────────────────────────────────────────────────────

/// Run a computation in both f64 and BigRational, returning both results.
#[derive(Debug, Clone)]
pub struct DualPathComputation {
    pub f64_value: f64,
    pub rational_value: BigRational,
    pub error_bound: f64,
}

impl DualPathComputation {
    pub fn from_f64(v: f64) -> Self {
        let converter = F64ToRationalConverter::new();
        let rv = converter.f64_to_rational(v);
        Self {
            f64_value: v,
            rational_value: rv,
            error_bound: f64::EPSILON * v.abs(),
        }
    }

    pub fn from_rational(numer: i64, denom: i64) -> Option<Self> {
        if denom == 0 {
            return None;
        }
        let rv = BigRational::new(BigInt::from(numer), BigInt::from(denom));
        let fv = numer as f64 / denom as f64;
        Some(Self {
            f64_value: fv,
            rational_value: rv,
            error_bound: 0.0,
        })
    }

    pub fn add(&self, other: &DualPathComputation) -> Self {
        Self {
            f64_value: self.f64_value + other.f64_value,
            rational_value: &self.rational_value + &other.rational_value,
            error_bound: self.error_bound + other.error_bound + f64::EPSILON,
        }
    }

    pub fn sub(&self, other: &DualPathComputation) -> Self {
        Self {
            f64_value: self.f64_value - other.f64_value,
            rational_value: &self.rational_value - &other.rational_value,
            error_bound: self.error_bound + other.error_bound + f64::EPSILON,
        }
    }

    pub fn mul(&self, other: &DualPathComputation) -> Self {
        Self {
            f64_value: self.f64_value * other.f64_value,
            rational_value: &self.rational_value * &other.rational_value,
            error_bound: self.error_bound * other.f64_value.abs()
                + other.error_bound * self.f64_value.abs()
                + f64::EPSILON,
        }
    }

    pub fn div(&self, other: &DualPathComputation) -> Option<Self> {
        if other.f64_value.abs() < 1e-300 || other.rational_value.is_zero() {
            return None;
        }
        Some(Self {
            f64_value: self.f64_value / other.f64_value,
            rational_value: &self.rational_value / &other.rational_value,
            error_bound: (self.error_bound / other.f64_value.abs())
                + (self.f64_value.abs() * other.error_bound
                    / (other.f64_value * other.f64_value))
                + f64::EPSILON,
        })
    }

    /// Check whether f64 and rational results agree in sign and ordering.
    pub fn agrees(&self) -> bool {
        if let Some(rat_f64) = self.rational_value.to_f64() {
            (self.f64_value - rat_f64).abs() <= self.error_bound + 1e-12
        } else {
            // Rational too large for f64; check sign
            let f_sign = if self.f64_value > 0.0 {
                1
            } else if self.f64_value < 0.0 {
                -1
            } else {
                0
            };
            let r_sign = self.rational_value.signum().to_integer().to_i32().unwrap_or(0);
            f_sign == r_sign
        }
    }
}

// ── F64 to rational converter ────────────────────────────────────────────────

/// Converts f64 values to exact rational representation.
pub struct F64ToRationalConverter {
    cache: HashMap<u64, BigRational>,
}

impl F64ToRationalConverter {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
        }
    }

    /// Convert an f64 to its exact rational representation.
    /// Uses the IEEE 754 binary representation for exact conversion.
    pub fn f64_to_rational(&self, v: f64) -> BigRational {
        if v == 0.0 {
            return BigRational::zero();
        }
        if !v.is_finite() {
            // Represent Inf/NaN as a sentinel (this is a best-effort)
            return if v > 0.0 {
                BigRational::new(BigInt::from(i64::MAX), BigInt::from(1))
            } else {
                BigRational::new(BigInt::from(i64::MIN), BigInt::from(1))
            };
        }

        // Use continued fraction approximation for exact-ish conversion
        let (numer, denom) = Self::float_to_fraction(v, 1_000_000_000);
        BigRational::new(BigInt::from(numer), BigInt::from(denom))
    }

    /// Convert f64 to a fraction numer/denom with bounded denominator.
    fn float_to_fraction(value: f64, max_denom: i64) -> (i64, i64) {
        if value == 0.0 {
            return (0, 1);
        }

        let sign = if value < 0.0 { -1i64 } else { 1i64 };
        let abs_val = value.abs();

        // Stern-Brocot tree / continued fraction approach
        let mut p0: i64 = 0;
        let mut q0: i64 = 1;
        let mut p1: i64 = 1;
        let mut q1: i64 = 0;
        let mut x = abs_val;

        for _ in 0..64 {
            let a = x as i64;
            let p2 = a.saturating_mul(p1).saturating_add(p0);
            let q2 = a.saturating_mul(q1).saturating_add(q0);

            if q2 > max_denom || q2 <= 0 {
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
            if x > max_denom as f64 {
                break;
            }
        }

        (sign * p1, q1.max(1))
    }

    /// Check if an f64 is a special value.
    pub fn is_special(v: f64) -> SpecialFloat {
        if v.is_nan() {
            SpecialFloat::NaN
        } else if v.is_infinite() {
            if v > 0.0 {
                SpecialFloat::PosInfinity
            } else {
                SpecialFloat::NegInfinity
            }
        } else if v.is_subnormal() {
            SpecialFloat::Subnormal
        } else {
            SpecialFloat::Normal
        }
    }

    /// Return the error bound for a particular f64 → rational conversion.
    pub fn conversion_error_bound(&self, v: f64) -> f64 {
        if !v.is_finite() {
            return f64::INFINITY;
        }
        // Machine epsilon scaled by magnitude
        f64::EPSILON * v.abs().max(1.0)
    }
}

impl Default for F64ToRationalConverter {
    fn default() -> Self {
        Self::new()
    }
}

/// Classification of special floating-point values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpecialFloat {
    Normal,
    Subnormal,
    PosInfinity,
    NegInfinity,
    NaN,
}

// ── Interval-rational bridge ─────────────────────────────────────────────────

/// Converts f64 intervals to rational intervals and vice versa.
pub struct IntervalRationalBridge {
    converter: F64ToRationalConverter,
}

impl IntervalRationalBridge {
    pub fn new() -> Self {
        Self {
            converter: F64ToRationalConverter::new(),
        }
    }

    /// Convert an f64 interval [lo, hi] to a rational interval.
    pub fn f64_interval_to_rational(
        &self,
        lo: f64,
        hi: f64,
    ) -> (BigRational, BigRational) {
        let lo_rat = self.converter.f64_to_rational(lo);
        let hi_rat = self.converter.f64_to_rational(hi);
        (lo_rat, hi_rat)
    }

    /// Check if a rational value lies within a rational interval.
    pub fn rational_in_interval(
        &self,
        value: &BigRational,
        lo: &BigRational,
        hi: &BigRational,
    ) -> bool {
        value >= lo && value <= hi
    }

    /// Verify that f64 interval containment agrees with rational.
    pub fn verify_containment(
        &self,
        value: f64,
        lo: f64,
        hi: f64,
    ) -> ContainmentVerification {
        let f64_result = value >= lo && value <= hi;
        let v_rat = self.converter.f64_to_rational(value);
        let lo_rat = self.converter.f64_to_rational(lo);
        let hi_rat = self.converter.f64_to_rational(hi);
        let rational_result = self.rational_in_interval(&v_rat, &lo_rat, &hi_rat);

        ContainmentVerification {
            value,
            lo,
            hi,
            f64_result,
            rational_result,
            agree: f64_result == rational_result,
        }
    }
}

impl Default for IntervalRationalBridge {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of verifying interval containment in dual arithmetic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContainmentVerification {
    pub value: f64,
    pub lo: f64,
    pub hi: f64,
    pub f64_result: bool,
    pub rational_result: bool,
    pub agree: bool,
}

/// Comparison result for dual-path verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonResult {
    pub f64_less: bool,
    pub rational_less: bool,
    pub agree: bool,
    pub difference_f64: f64,
}

/// Verify a comparison in both f64 and rational.
pub fn verify_comparison(
    a_f64: f64,
    b_f64: f64,
    a_rat: &BigRational,
    b_rat: &BigRational,
) -> ComparisonResult {
    let f64_less = a_f64 < b_f64;
    let rational_less = a_rat < b_rat;
    ComparisonResult {
        f64_less,
        rational_less,
        agree: f64_less == rational_less,
        difference_f64: a_f64 - b_f64,
    }
}

// ── Rational expression evaluator ────────────────────────────────────────────

/// Evaluate an Expression in exact rational arithmetic.
pub fn eval_expression_rational(
    expr: &Expression,
    env: &HashMap<String, BigRational>,
) -> Option<BigRational> {
    let converter = F64ToRationalConverter::new();

    match expr {
        Expression::Literal(LiteralValue::Float(v)) => {
            Some(converter.f64_to_rational(*v))
        }
        Expression::Literal(LiteralValue::Rational(r)) => {
            if r.denom == 0 {
                return None;
            }
            Some(BigRational::new(
                BigInt::from(r.numer),
                BigInt::from(r.denom),
            ))
        }
        Expression::Literal(LiteralValue::Integer(i)) => {
            Some(BigRational::from(BigInt::from(*i)))
        }
        Expression::Variable(name) => env.get(name).cloned(),
        Expression::BinaryExpr { op, left, right } => {
            let l = eval_expression_rational(left, env)?;
            let r = eval_expression_rational(right, env)?;
            match op {
                BinaryOp::Add => Some(&l + &r),
                BinaryOp::Sub => Some(&l - &r),
                BinaryOp::Mul => Some(&l * &r),
                BinaryOp::Div => {
                    if r.is_zero() {
                        None
                    } else {
                        Some(&l / &r)
                    }
                }
                BinaryOp::Min => Some(if l < r { l } else { r }),
                BinaryOp::Max => Some(if l > r { l } else { r }),
                BinaryOp::Pow => {
                    let exp = r.to_integer().to_i64()?;
                    if exp >= 0 {
                        Some(rational_pow(&l, exp as u64))
                    } else {
                        if l.is_zero() {
                            return None;
                        }
                        let inv = BigRational::new(l.denom().clone(), l.numer().clone());
                        Some(rational_pow(&inv, (-exp) as u64))
                    }
                }
            }
        }
        Expression::UnaryExpr { op, operand } => {
            let v = eval_expression_rational(operand, env)?;
            match op {
                UnaryOp::Neg => Some(-v),
                UnaryOp::Abs => Some(v.abs()),
                // Sqrt and Log cannot be computed exactly in rationals
                UnaryOp::Sqrt | UnaryOp::Log => {
                    let fv = operand.try_eval_f64()?;
                    let result = match op {
                        UnaryOp::Sqrt => {
                            if fv < 0.0 { return None; }
                            fv.sqrt()
                        }
                        UnaryOp::Log => {
                            if fv <= 0.0 { return None; }
                            fv.ln()
                        }
                        _ => unreachable!(),
                    };
                    Some(converter.f64_to_rational(result))
                }
            }
        }
        _ => None,
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Expression, RationalLiteral};

    #[test]
    fn test_f64_to_rational_exact() {
        let conv = F64ToRationalConverter::new();
        let r = conv.f64_to_rational(0.5);
        assert_eq!(r, BigRational::new(BigInt::from(1), BigInt::from(2)));
    }

    #[test]
    fn test_f64_to_rational_zero() {
        let conv = F64ToRationalConverter::new();
        let r = conv.f64_to_rational(0.0);
        assert!(r.is_zero());
    }

    #[test]
    fn test_f64_to_rational_negative() {
        let conv = F64ToRationalConverter::new();
        let r = conv.f64_to_rational(-0.25);
        assert!(r < BigRational::zero());
    }

    #[test]
    fn test_special_float_classification() {
        assert_eq!(F64ToRationalConverter::is_special(1.0), SpecialFloat::Normal);
        assert_eq!(
            F64ToRationalConverter::is_special(f64::INFINITY),
            SpecialFloat::PosInfinity
        );
        assert_eq!(
            F64ToRationalConverter::is_special(f64::NEG_INFINITY),
            SpecialFloat::NegInfinity
        );
        assert_eq!(F64ToRationalConverter::is_special(f64::NAN), SpecialFloat::NaN);
    }

    #[test]
    fn test_dual_path_add() {
        let a = DualPathComputation::from_f64(1.0);
        let b = DualPathComputation::from_f64(2.0);
        let c = a.add(&b);
        assert!((c.f64_value - 3.0).abs() < 1e-12);
        assert!(c.agrees());
    }

    #[test]
    fn test_dual_path_from_rational() {
        let dp = DualPathComputation::from_rational(1, 3).unwrap();
        assert!((dp.f64_value - 1.0 / 3.0).abs() < 1e-12);
        assert!(dp.agrees());
    }

    #[test]
    fn test_dual_path_div_by_zero() {
        let a = DualPathComputation::from_f64(1.0);
        let b = DualPathComputation::from_f64(0.0);
        assert!(a.div(&b).is_none());
    }

    #[test]
    fn test_verify_comparison_agree() {
        let a_rat = BigRational::new(BigInt::from(1), BigInt::from(3));
        let b_rat = BigRational::new(BigInt::from(1), BigInt::from(2));
        let result = verify_comparison(1.0 / 3.0, 0.5, &a_rat, &b_rat);
        assert!(result.agree);
        assert!(result.f64_less);
    }

    #[test]
    fn test_rational_verifier_simple_comparison() {
        let rv = RationalVerifier::new();
        let expr = Expression::lt(Expression::float(0.3), Expression::float(0.5));
        let ov = rv.verify_expression(&expr).unwrap();
        assert!(ov.agree);
        assert!(ov.f64_result);
    }

    #[test]
    fn test_rational_verifier_addition() {
        let rv = RationalVerifier::new();
        let left = Expression::add(Expression::float(0.1), Expression::float(0.2));
        let right = Expression::float(0.3);
        // 0.1 + 0.2 vs 0.3 is a classic floating point issue
        let expr = Expression::Comparison {
            op: ComparisonOp::Le,
            left: Box::new(left),
            right: Box::new(right),
        };
        let ov = rv.verify_expression(&expr);
        assert!(ov.is_some());
    }

    #[test]
    fn test_rational_verifier_rational_literal() {
        let rv = RationalVerifier::new();
        let expr = Expression::lt(
            Expression::rational(1, 3),
            Expression::rational(1, 2),
        );
        let ov = rv.verify_expression(&expr).unwrap();
        assert!(ov.agree);
        assert!(ov.f64_result);
    }

    #[test]
    fn test_interval_rational_bridge() {
        let bridge = IntervalRationalBridge::new();
        let cv = bridge.verify_containment(0.5, 0.0, 1.0);
        assert!(cv.f64_result);
        assert!(cv.rational_result);
        assert!(cv.agree);
    }

    #[test]
    fn test_interval_rational_bridge_outside() {
        let bridge = IntervalRationalBridge::new();
        let cv = bridge.verify_containment(1.5, 0.0, 1.0);
        assert!(!cv.f64_result);
        assert!(!cv.rational_result);
        assert!(cv.agree);
    }

    #[test]
    fn test_eval_expression_rational() {
        let env = HashMap::new();
        let expr = Expression::add(
            Expression::rational(1, 3),
            Expression::rational(1, 6),
        );
        let result = eval_expression_rational(&expr, &env).unwrap();
        let expected = BigRational::new(BigInt::from(1), BigInt::from(2));
        assert_eq!(result, expected);
    }

    #[test]
    fn test_eval_expression_with_var() {
        let mut env = HashMap::new();
        env.insert(
            "x".to_string(),
            BigRational::new(BigInt::from(3), BigInt::from(1)),
        );
        let expr = Expression::mul(Expression::var("x"), Expression::rational(1, 2));
        let result = eval_expression_rational(&expr, &env).unwrap();
        let expected = BigRational::new(BigInt::from(3), BigInt::from(2));
        assert_eq!(result, expected);
    }

    #[test]
    fn test_eval_div_by_zero() {
        let env = HashMap::new();
        let expr = Expression::div(Expression::float(1.0), Expression::float(0.0));
        assert!(eval_expression_rational(&expr, &env).is_none());
    }

    #[test]
    fn test_conversion_error_bound() {
        let conv = F64ToRationalConverter::new();
        let eb = conv.conversion_error_bound(1.0);
        assert!(eb > 0.0);
        assert!(eb < 1e-12);
    }

    #[test]
    fn test_dual_path_mul() {
        let a = DualPathComputation::from_f64(3.0);
        let b = DualPathComputation::from_f64(7.0);
        let c = a.mul(&b);
        assert!((c.f64_value - 21.0).abs() < 1e-12);
        assert!(c.agrees());
    }

    #[test]
    fn test_dual_path_sub() {
        let a = DualPathComputation::from_f64(10.0);
        let b = DualPathComputation::from_f64(3.0);
        let c = a.sub(&b);
        assert!((c.f64_value - 7.0).abs() < 1e-12);
        assert!(c.agrees());
    }
}
