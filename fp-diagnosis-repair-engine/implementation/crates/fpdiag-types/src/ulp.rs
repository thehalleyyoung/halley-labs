//! ULP (Unit in the Last Place) computation utilities.
//!
//! Functions for computing ULP values and ULP distances between
//! floating-point numbers—fundamental metrics for quantifying
//! floating-point error.

/// Compute the ULP (unit in the last place) of an f64 value.
///
/// For a normal number, ULP = 2^(exponent - 52).
/// For subnormals, ULP = 2^(-1074).
/// For zero, returns the smallest subnormal.
pub fn ulp_f64(x: f64) -> f64 {
    let x = x.abs();
    if x == 0.0 {
        return f64::from_bits(1); // smallest subnormal
    }
    if x.is_nan() || x.is_infinite() {
        return f64::NAN;
    }
    let bits = x.to_bits();
    let exponent = ((bits >> 52) & 0x7FF) as i64;
    if exponent == 0 {
        // Subnormal: ULP is the smallest subnormal
        f64::from_bits(1)
    } else {
        // Normal: ULP = 2^(exponent - bias - 52) = 2^(exponent - 1023 - 52)
        // = 2^(exponent - 1075)
        let ulp_exp = exponent - 52;
        if ulp_exp > 0 {
            f64::from_bits((ulp_exp as u64) << 52)
        } else {
            // Subnormal ULP
            f64::from_bits(1u64 << (ulp_exp + 52 - 1).max(0) as u64)
        }
    }
}

/// Compute the ULP of an f32 value.
pub fn ulp_f32(x: f32) -> f32 {
    if x == 0.0 {
        return f32::MIN_POSITIVE * f32::EPSILON;
    }
    let bits = x.to_bits();
    let exponent = ((bits >> 23) & 0xFF) as i32;
    if exponent == 0 {
        f32::from_bits(1)
    } else if exponent == 0xFF {
        f32::NAN
    } else {
        f32::from_bits((exponent as u32) << 23) - f32::from_bits(((exponent - 1) as u32) << 23)
    }
}

/// Compute the ULP distance between two f64 values.
///
/// This counts the number of representable floating-point numbers between
/// `a` and `b` (inclusive at both ends minus 1).  Returns 0 if `a == b`.
/// Returns `u64::MAX` if either value is NaN or they have different signs
/// and neither is zero.
pub fn ulp_distance_f64(a: f64, b: f64) -> u64 {
    if a.is_nan() || b.is_nan() {
        return u64::MAX;
    }
    if a == b {
        return 0;
    }
    let a_bits = a.to_bits() as i64;
    let b_bits = b.to_bits() as i64;

    // Handle sign differences
    let a_signed = if a_bits < 0 {
        i64::MIN - a_bits
    } else {
        a_bits
    };
    let b_signed = if b_bits < 0 {
        i64::MIN - b_bits
    } else {
        b_bits
    };

    (a_signed - b_signed).unsigned_abs()
}

/// Compute the ULP distance between two f32 values.
pub fn ulp_distance_f32(a: f32, b: f32) -> u32 {
    if a.is_nan() || b.is_nan() {
        return u32::MAX;
    }
    if a == b {
        return 0;
    }
    let a_bits = a.to_bits() as i32;
    let b_bits = b.to_bits() as i32;

    let a_signed = if a_bits < 0 {
        i32::MIN - a_bits
    } else {
        a_bits
    };
    let b_signed = if b_bits < 0 {
        i32::MIN - b_bits
    } else {
        b_bits
    };

    (a_signed - b_signed).unsigned_abs()
}

/// Number of significant bits lost between the computed and exact values.
pub fn bits_lost(computed: f64, exact: f64) -> f64 {
    if exact == 0.0 {
        if computed == 0.0 {
            return 0.0;
        }
        return 53.0; // all bits lost
    }
    let rel_err = ((computed - exact) / exact).abs();
    if rel_err == 0.0 {
        return 0.0;
    }
    (-rel_err.log2()).max(0.0).min(53.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ulp_of_one() {
        let u = ulp_f64(1.0);
        assert!((u - f64::EPSILON).abs() < 1e-30);
    }

    #[test]
    fn ulp_distance_adjacent() {
        let a = 1.0_f64;
        let b = f64::from_bits(a.to_bits() + 1);
        assert_eq!(ulp_distance_f64(a, b), 1);
    }

    #[test]
    fn ulp_distance_same() {
        assert_eq!(ulp_distance_f64(42.0, 42.0), 0);
    }

    #[test]
    fn bits_lost_none() {
        assert_eq!(bits_lost(1.0, 1.0), 0.0);
    }
}
