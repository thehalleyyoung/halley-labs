//! Mathematical utilities: modular arithmetic, polynomials, FFT.

use serde::{Serialize, Deserialize};

/// Extended GCD: returns (gcd, x, y) such that a*x + b*y = gcd(a, b)
pub fn extended_gcd(a: i64, b: i64) -> (i64, i64, i64) {
    if a == 0 {
        return (b, 0, 1);
    }
    let (g, x1, y1) = extended_gcd(b % a, a);
    let x = y1 - (b / a) * x1;
    let y = x1;
    (g, x, y)
}

/// Modular exponentiation: base^exp mod modulus
pub fn mod_pow(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    if modulus == 1 {
        return 0;
    }
    let mut result = 1u64;
    base %= modulus;
    while exp > 0 {
        if exp & 1 == 1 {
            result = ((result as u128 * base as u128) % modulus as u128) as u64;
        }
        exp >>= 1;
        base = ((base as u128 * base as u128) % modulus as u128) as u64;
    }
    result
}

/// Modular inverse: a^(-1) mod m (requires gcd(a, m) = 1)
pub fn mod_inv(a: u64, m: u64) -> Option<u64> {
    let (g, x, _) = extended_gcd(a as i64, m as i64);
    if g != 1 {
        None
    } else {
        Some(((x % m as i64 + m as i64) % m as i64) as u64)
    }
}

/// Modular addition: (a + b) mod m
pub fn mod_add(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128 + b as u128) % m as u128) as u64
}

/// Modular subtraction: (a - b) mod m
pub fn mod_sub(a: u64, b: u64, m: u64) -> u64 {
    if a >= b {
        (a - b) % m
    } else {
        m - ((b - a) % m)
    }
}

/// Modular multiplication: (a * b) mod m
pub fn mod_mul(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128 * b as u128) % m as u128) as u64
}

/// Evaluate a polynomial at a point using Horner's method.
/// coeffs[0] = constant term, coeffs[n-1] = leading coefficient.
/// Returns coeffs[0] + coeffs[1]*x + coeffs[2]*x^2 + ...
pub fn polynomial_eval(coeffs: &[f64], x: f64) -> f64 {
    let mut result = 0.0;
    for coeff in coeffs.iter().rev() {
        result = result * x + coeff;
    }
    result
}

/// Evaluate a polynomial over a finite field
pub fn polynomial_eval_mod(coeffs: &[u64], x: u64, modulus: u64) -> u64 {
    let mut result = 0u64;
    for &coeff in coeffs.iter().rev() {
        result = mod_add(mod_mul(result, x, modulus), coeff, modulus);
    }
    result
}

/// Lagrange interpolation over the reals.
/// Given points (x_i, y_i), returns the polynomial evaluated at target.
pub fn lagrange_interpolate(points: &[(f64, f64)], target: f64) -> f64 {
    let n = points.len();
    let mut result = 0.0;
    
    for i in 0..n {
        let (xi, yi) = points[i];
        let mut basis = 1.0;
        
        for j in 0..n {
            if i != j {
                let (xj, _) = points[j];
                basis *= (target - xj) / (xi - xj);
            }
        }
        
        result += yi * basis;
    }
    
    result
}

/// Lagrange interpolation over a finite field.
pub fn lagrange_interpolate_mod(
    points: &[(u64, u64)],
    target: u64,
    modulus: u64,
) -> u64 {
    let n = points.len();
    let mut result = 0u64;
    
    for i in 0..n {
        let (xi, yi) = points[i];
        let mut num = 1u64;
        let mut den = 1u64;
        
        for j in 0..n {
            if i != j {
                let (xj, _) = points[j];
                num = mod_mul(num, mod_sub(target, xj, modulus), modulus);
                den = mod_mul(den, mod_sub(xi, xj, modulus), modulus);
            }
        }
        
        let basis = mod_mul(num, mod_inv(den, modulus).unwrap_or(0), modulus);
        result = mod_add(result, mod_mul(yi, basis, modulus), modulus);
    }
    
    result
}

/// Polynomial addition: c(x) = a(x) + b(x)
pub fn polynomial_add(a: &[f64], b: &[f64]) -> Vec<f64> {
    let len = a.len().max(b.len());
    let mut result = vec![0.0; len];
    for (i, &v) in a.iter().enumerate() {
        result[i] += v;
    }
    for (i, &v) in b.iter().enumerate() {
        result[i] += v;
    }
    result
}

/// Polynomial multiplication: c(x) = a(x) * b(x)
pub fn polynomial_mul(a: &[f64], b: &[f64]) -> Vec<f64> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut result = vec![0.0; a.len() + b.len() - 1];
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            result[i + j] += ai * bj;
        }
    }
    result
}

/// Polynomial multiplication over a finite field
pub fn polynomial_mul_mod(a: &[u64], b: &[u64], modulus: u64) -> Vec<u64> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let mut result = vec![0u64; a.len() + b.len() - 1];
    for (i, &ai) in a.iter().enumerate() {
        for (j, &bj) in b.iter().enumerate() {
            result[i + j] = mod_add(result[i + j], mod_mul(ai, bj, modulus), modulus);
        }
    }
    result
}

/// In-place radix-2 Cooley-Tukey FFT over complex numbers.
/// Input length must be a power of 2.
pub fn fft(data: &mut [(f64, f64)], inverse: bool) {
    let n = data.len();
    assert!(n.is_power_of_two(), "FFT length must be a power of 2");
    
    if n <= 1 {
        return;
    }
    
    // Bit-reversal permutation
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            data.swap(i, j);
        }
    }
    
    // Butterfly operations
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let angle = if inverse {
            2.0 * std::f64::consts::PI / len as f64
        } else {
            -2.0 * std::f64::consts::PI / len as f64
        };
        
        let w_step = (angle.cos(), angle.sin());
        
        for start in (0..n).step_by(len) {
            let mut w = (1.0, 0.0);
            for k in 0..half {
                let u = data[start + k];
                let t = complex_mul(w, data[start + k + half]);
                data[start + k] = complex_add(u, t);
                data[start + k + half] = complex_sub(u, t);
                w = complex_mul(w, w_step);
            }
        }
        
        len <<= 1;
    }
    
    if inverse {
        let n_f = n as f64;
        for x in data.iter_mut() {
            x.0 /= n_f;
            x.1 /= n_f;
        }
    }
}

fn complex_mul(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

fn complex_add(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0 + b.0, a.1 + b.1)
}

fn complex_sub(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0 - b.0, a.1 - b.1)
}

/// Polynomial multiplication using FFT
pub fn polynomial_mul_fft(a: &[f64], b: &[f64]) -> Vec<f64> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    
    let result_len = a.len() + b.len() - 1;
    let n = result_len.next_power_of_two();
    
    let mut fa: Vec<(f64, f64)> = a.iter().map(|&x| (x, 0.0)).collect();
    fa.resize(n, (0.0, 0.0));
    
    let mut fb: Vec<(f64, f64)> = b.iter().map(|&x| (x, 0.0)).collect();
    fb.resize(n, (0.0, 0.0));
    
    fft(&mut fa, false);
    fft(&mut fb, false);
    
    let mut fc: Vec<(f64, f64)> = fa.iter().zip(&fb)
        .map(|(&a, &b)| complex_mul(a, b))
        .collect();
    
    fft(&mut fc, true);
    
    fc.iter().take(result_len).map(|&(re, _)| re).collect()
}

/// Number Theoretic Transform (NTT) - FFT over finite fields.
/// Uses modulus p where p-1 is divisible by n (a power of 2).
pub fn ntt(data: &mut [u64], modulus: u64, root: u64, inverse: bool) {
    let n = data.len();
    assert!(n.is_power_of_two(), "NTT length must be a power of 2");
    
    if n <= 1 {
        return;
    }
    
    // Bit-reversal permutation
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            data.swap(i, j);
        }
    }
    
    let actual_root = if inverse {
        mod_inv(root, modulus).unwrap()
    } else {
        root
    };
    
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let w_base = mod_pow(actual_root, (modulus - 1) / len as u64, modulus);
        
        for start in (0..n).step_by(len) {
            let mut w = 1u64;
            for k in 0..half {
                let u = data[start + k];
                let t = mod_mul(w, data[start + k + half], modulus);
                data[start + k] = mod_add(u, t, modulus);
                data[start + k + half] = mod_sub(u, t, modulus);
                w = mod_mul(w, w_base, modulus);
            }
        }
        
        len <<= 1;
    }
    
    if inverse {
        let n_inv = mod_inv(n as u64, modulus).unwrap();
        for x in data.iter_mut() {
            *x = mod_mul(*x, n_inv, modulus);
        }
    }
}

/// Find a primitive root of unity of order n modulo p
pub fn find_primitive_root(n: u64, p: u64) -> Option<u64> {
    if (p - 1) % n != 0 {
        return None;
    }
    
    // Try small values as generators
    for g in 2..p.min(1000) {
        let root = mod_pow(g, (p - 1) / n, p);
        if root != 1 && mod_pow(root, n, p) == 1 {
            // Verify it's a primitive nth root
            let mut is_primitive = true;
            let mut k = 1;
            while k < n {
                if mod_pow(root, k, p) == 1 {
                    is_primitive = false;
                    break;
                }
                k += 1;
            }
            if is_primitive {
                return Some(root);
            }
        }
    }
    
    None
}

/// Check if a number is prime (trial division, small numbers only)
pub fn is_prime(n: u64) -> bool {
    if n < 2 { return false; }
    if n < 4 { return true; }
    if n % 2 == 0 || n % 3 == 0 { return false; }
    
    let mut i = 5;
    while i * i <= n {
        if n % i == 0 || n % (i + 2) == 0 {
            return false;
        }
        i += 6;
    }
    true
}

/// Miller-Rabin primality test (probabilistic)
pub fn is_probable_prime(n: u64, witnesses: &[u64]) -> bool {
    if n < 2 { return false; }
    if n == 2 || n == 3 { return true; }
    if n % 2 == 0 { return false; }
    
    // Write n-1 as 2^r * d
    let mut d = n - 1;
    let mut r = 0;
    while d % 2 == 0 {
        d /= 2;
        r += 1;
    }
    
    'witness: for &a in witnesses {
        if a >= n { continue; }
        
        let mut x = mod_pow(a, d, n);
        
        if x == 1 || x == n - 1 {
            continue;
        }
        
        for _ in 0..r - 1 {
            x = mod_mul(x, x, n);
            if x == n - 1 {
                continue 'witness;
            }
        }
        
        return false;
    }
    
    true
}

/// Chinese Remainder Theorem for two moduli
/// Solves: x ≡ a1 (mod m1), x ≡ a2 (mod m2)
pub fn crt(a1: u64, m1: u64, a2: u64, m2: u64) -> Option<(u64, u64)> {
    let (g, p, _) = extended_gcd(m1 as i64, m2 as i64);
    if g != 1 {
        // Check if solution exists
        if (a2 as i64 - a1 as i64) % g != 0 {
            return None;
        }
    }
    
    let m = m1 as u128 * m2 as u128;
    let diff = ((a2 as i128 - a1 as i128) % m as i128 + m as i128) % m as i128;
    let p_mod = ((p as i128 % m2 as i128) + m2 as i128) % m2 as i128;
    let x = (a1 as u128 + m1 as u128 * ((diff as u128 * p_mod as u128) % m2 as u128)) % m;
    
    Some((x as u64, (m1 as u128 * m2 as u128) as u64))
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_extended_gcd() {
        let (g, x, y) = extended_gcd(35, 15);
        assert_eq!(g, 5);
        assert_eq!(35 * x + 15 * y, 5);
    }
    
    #[test]
    fn test_extended_gcd_coprime() {
        let (g, x, y) = extended_gcd(7, 11);
        assert_eq!(g, 1);
        assert_eq!(7 * x + 11 * y, 1);
    }
    
    #[test]
    fn test_mod_pow() {
        assert_eq!(mod_pow(2, 10, 1000), 24);
        assert_eq!(mod_pow(3, 0, 7), 1);
        assert_eq!(mod_pow(2, 10, 1024), 0);
    }
    
    #[test]
    fn test_mod_inv() {
        let inv = mod_inv(3, 7).unwrap();
        assert_eq!((3 * inv) % 7, 1);
        
        assert!(mod_inv(2, 4).is_none()); // gcd(2,4) = 2 ≠ 1
    }
    
    #[test]
    fn test_mod_arithmetic() {
        assert_eq!(mod_add(7, 5, 10), 2);
        assert_eq!(mod_sub(3, 7, 10), 6);
        assert_eq!(mod_mul(7, 8, 10), 6);
    }
    
    #[test]
    fn test_polynomial_eval() {
        // p(x) = 2 + 3x + x^2
        let coeffs = vec![2.0, 3.0, 1.0];
        assert!((polynomial_eval(&coeffs, 0.0) - 2.0).abs() < 1e-10);
        assert!((polynomial_eval(&coeffs, 1.0) - 6.0).abs() < 1e-10);
        assert!((polynomial_eval(&coeffs, 2.0) - 12.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_polynomial_eval_mod() {
        // p(x) = 2 + 3x + x^2 mod 7
        let coeffs = vec![2, 3, 1];
        assert_eq!(polynomial_eval_mod(&coeffs, 0, 7), 2);
        assert_eq!(polynomial_eval_mod(&coeffs, 1, 7), 6);
        assert_eq!(polynomial_eval_mod(&coeffs, 2, 7), 12 % 7); // 5
    }
    
    #[test]
    fn test_lagrange_interpolation() {
        // Interpolate y = x^2 from points (0,0), (1,1), (2,4)
        let points = vec![(0.0, 0.0), (1.0, 1.0), (2.0, 4.0)];
        assert!((lagrange_interpolate(&points, 3.0) - 9.0).abs() < 1e-10);
        assert!((lagrange_interpolate(&points, 0.5) - 0.25).abs() < 1e-10);
    }
    
    #[test]
    fn test_lagrange_interpolation_mod() {
        // Interpolate y = x^2 mod 97 from points
        let points = vec![(0, 0), (1, 1), (2, 4)];
        assert_eq!(lagrange_interpolate_mod(&points, 3, 97), 9);
    }
    
    #[test]
    fn test_polynomial_add() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0];
        let c = polynomial_add(&a, &b);
        assert_eq!(c, vec![5.0, 7.0, 3.0]);
    }
    
    #[test]
    fn test_polynomial_mul() {
        // (1 + x) * (1 + x) = 1 + 2x + x^2
        let a = vec![1.0, 1.0];
        let b = vec![1.0, 1.0];
        let c = polynomial_mul(&a, &b);
        assert!((c[0] - 1.0).abs() < 1e-10);
        assert!((c[1] - 2.0).abs() < 1e-10);
        assert!((c[2] - 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_polynomial_mul_mod() {
        let a = vec![1u64, 1];
        let b = vec![1u64, 1];
        let c = polynomial_mul_mod(&a, &b, 97);
        assert_eq!(c, vec![1, 2, 1]);
    }
    
    #[test]
    fn test_fft_roundtrip() {
        let mut data: Vec<(f64, f64)> = vec![
            (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0),
        ];
        let original = data.clone();
        
        fft(&mut data, false);
        fft(&mut data, true);
        
        for (a, b) in data.iter().zip(&original) {
            assert!((a.0 - b.0).abs() < 1e-10);
            assert!((a.1 - b.1).abs() < 1e-10);
        }
    }
    
    #[test]
    fn test_fft_known_values() {
        let mut data = vec![(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)];
        fft(&mut data, false);
        // FFT of [1, 0, 0, 0] should be [1, 1, 1, 1]
        for (re, im) in &data {
            assert!((*re - 1.0).abs() < 1e-10);
            assert!(im.abs() < 1e-10);
        }
    }
    
    #[test]
    fn test_polynomial_mul_fft() {
        let a = vec![1.0, 1.0]; // 1 + x
        let b = vec![1.0, 1.0]; // 1 + x
        let c = polynomial_mul_fft(&a, &b);
        assert!((c[0] - 1.0).abs() < 1e-6);
        assert!((c[1] - 2.0).abs() < 1e-6);
        assert!((c[2] - 1.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_is_prime() {
        assert!(!is_prime(0));
        assert!(!is_prime(1));
        assert!(is_prime(2));
        assert!(is_prime(3));
        assert!(!is_prime(4));
        assert!(is_prime(97));
        assert!(!is_prime(100));
    }
    
    #[test]
    fn test_miller_rabin() {
        assert!(is_probable_prime(97, &[2, 3, 5, 7]));
        assert!(!is_probable_prime(100, &[2, 3, 5, 7]));
        assert!(is_probable_prime(104729, &[2, 3, 5, 7, 11, 13]));
    }
    
    #[test]
    fn test_ntt_roundtrip() {
        let p = 97u64; // p - 1 = 96 = 2^5 * 3, so n=4 divides p-1
        let root = find_primitive_root(4, p);
        if let Some(w) = root {
            let mut data = vec![1, 2, 3, 4];
            let original = data.clone();
            
            ntt(&mut data, p, w, false);
            ntt(&mut data, p, w, true);
            
            assert_eq!(data, original);
        }
    }
    
    #[test]
    fn test_crt() {
        // x ≡ 2 (mod 3), x ≡ 3 (mod 5)
        let (x, m) = crt(2, 3, 3, 5).unwrap();
        assert_eq!(x % 3, 2);
        assert_eq!(x % 5, 3);
        assert_eq!(m, 15);
    }
    
    #[test]
    fn test_crt_larger() {
        // x ≡ 1 (mod 7), x ≡ 4 (mod 11)
        let (x, m) = crt(1, 7, 4, 11).unwrap();
        assert_eq!(x % 7, 1);
        assert_eq!(x % 11, 4);
        assert_eq!(m, 77);
    }
    
    #[test]
    fn test_find_primitive_root() {
        // For p = 97, find 4th root of unity
        let root = find_primitive_root(4, 97);
        if let Some(w) = root {
            assert_eq!(mod_pow(w, 4, 97), 1);
            assert_ne!(mod_pow(w, 2, 97), 1);
        }
    }
}
