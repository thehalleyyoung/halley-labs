//! Mathematical utilities for audio and psychoacoustic computation.
//!
//! Includes complex arithmetic, FFT, interpolation, statistics, entropy,
//! Weber fractions, signal-to-noise ratio, and small matrix operations.

use std::f64::consts::PI;
use std::fmt;

// ===========================================================================
// Complex number
// ===========================================================================

/// A complex number with f64 components.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    pub const ZERO: Complex = Complex { re: 0.0, im: 0.0 };

    pub fn new(re: f64, im: f64) -> Self { Self { re, im } }

    pub fn from_polar(r: f64, theta: f64) -> Self {
        Self { re: r * theta.cos(), im: r * theta.sin() }
    }

    pub fn magnitude(self) -> f64 { self.re.hypot(self.im) }
    pub fn phase(self) -> f64 { self.im.atan2(self.re) }
    pub fn conjugate(self) -> Self { Self { re: self.re, im: -self.im } }
    pub fn norm_sq(self) -> f64 { self.re * self.re + self.im * self.im }

    pub fn inverse(self) -> Self {
        let d = self.norm_sq();
        Self { re: self.re / d, im: -self.im / d }
    }

    pub fn exp(self) -> Self {
        let r = self.re.exp();
        Self { re: r * self.im.cos(), im: r * self.im.sin() }
    }
}

impl fmt::Display for Complex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.im >= 0.0 {
            write!(f, "{:.6}+{:.6}i", self.re, self.im)
        } else {
            write!(f, "{:.6}{:.6}i", self.re, self.im)
        }
    }
}

impl std::ops::Add for Complex {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self { re: self.re + rhs.re, im: self.im + rhs.im }
    }
}

impl std::ops::Sub for Complex {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self { re: self.re - rhs.re, im: self.im - rhs.im }
    }
}

impl std::ops::Mul for Complex {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

impl std::ops::Div for Complex {
    type Output = Self;
    fn div(self, rhs: Self) -> Self { self * rhs.inverse() }
}

impl std::ops::Mul<f64> for Complex {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Self { re: self.re * rhs, im: self.im * rhs }
    }
}

// ===========================================================================
// FFT (Cooley-Tukey radix-2 DIT)
// ===========================================================================

/// In-place Cooley-Tukey radix-2 decimation-in-time FFT.
/// Buffer length must be a power of two.
pub fn fft(buf: &mut [Complex]) {
    let n = buf.len();
    assert!(n.is_power_of_two(), "FFT length must be a power of two");
    if n <= 1 { return; }

    // Bit-reversal permutation
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 { j ^= bit; bit >>= 1; }
        j ^= bit;
        if i < j { buf.swap(i, j); }
    }

    // Butterfly stages
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let angle = -2.0 * PI / len as f64;
        let wn = Complex::from_polar(1.0, angle);
        let mut start = 0;
        while start < n {
            let mut w = Complex::new(1.0, 0.0);
            for k in 0..half {
                let u = buf[start + k];
                let t = w * buf[start + k + half];
                buf[start + k] = u + t;
                buf[start + k + half] = u - t;
                w = w * wn;
            }
            start += len;
        }
        len <<= 1;
    }
}

/// Inverse FFT.
pub fn ifft(buf: &mut [Complex]) {
    let n = buf.len();
    for c in buf.iter_mut() { *c = c.conjugate(); }
    fft(buf);
    let scale = 1.0 / n as f64;
    for c in buf.iter_mut() { *c = c.conjugate() * scale; }
}

/// Compute the magnitude spectrum of a real signal. Returns N/2+1 bins.
pub fn magnitude_spectrum(signal: &[f64]) -> Vec<f64> {
    let n = signal.len().next_power_of_two();
    let mut buf: Vec<Complex> = signal.iter()
        .map(|&x| Complex::new(x, 0.0))
        .chain(std::iter::repeat(Complex::ZERO))
        .take(n)
        .collect();
    fft(&mut buf);
    buf[..n / 2 + 1].iter().map(|c| c.magnitude()).collect()
}

// ===========================================================================
// Interpolation
// ===========================================================================

/// Linear interpolation between a and b at parameter t in [0, 1].
pub fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

/// Cubic Hermite interpolation between four equally-spaced samples.
/// t in [0, 1] interpolates between y1 and y2.
pub fn cubic_interpolate(y0: f64, y1: f64, y2: f64, y3: f64, t: f64) -> f64 {
    let a = -0.5 * y0 + 1.5 * y1 - 1.5 * y2 + 0.5 * y3;
    let b = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3;
    let c = -0.5 * y0 + 0.5 * y2;
    let d = y1;
    ((a * t + b) * t + c) * t + d
}

/// Table lookup with linear interpolation. Assumes table sorted by x.
pub fn table_lookup(table: &[(f64, f64)], x: f64) -> f64 {
    if table.is_empty() { return 0.0; }
    if x <= table[0].0 { return table[0].1; }
    if x >= table[table.len() - 1].0 { return table[table.len() - 1].1; }
    for i in 0..table.len() - 1 {
        if x >= table[i].0 && x <= table[i + 1].0 {
            let t = (x - table[i].0) / (table[i + 1].0 - table[i].0);
            return lerp(table[i].1, table[i + 1].1, t);
        }
    }
    table[table.len() - 1].1
}

// ===========================================================================
// Common functions
// ===========================================================================

/// Gaussian function: exp(-(x-mu)^2 / (2*sigma^2)).
pub fn gaussian(x: f64, mu: f64, sigma: f64) -> f64 {
    (-((x - mu).powi(2)) / (2.0 * sigma * sigma)).exp()
}

/// Standard sigmoid: 1 / (1 + e^(-x)).
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Softmax of a slice, returning a Vec<f64> that sums to 1.
pub fn softmax(xs: &[f64]) -> Vec<f64> {
    if xs.is_empty() { return vec![]; }
    let max = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = xs.iter().map(|x| (x - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum == 0.0 { return vec![1.0 / xs.len() as f64; xs.len()]; }
    exps.into_iter().map(|e| e / sum).collect()
}

/// Polynomial evaluation using Horner's method.
/// Coefficients ordered [a0, a1, ..., an] for a0 + a1*x + ... + an*x^n.
pub fn horner(coeffs: &[f64], x: f64) -> f64 {
    if coeffs.is_empty() { return 0.0; }
    let mut result = coeffs[coeffs.len() - 1];
    for i in (0..coeffs.len() - 1).rev() {
        result = result * x + coeffs[i];
    }
    result
}

// ===========================================================================
// Statistics
// ===========================================================================

/// Arithmetic mean.
pub fn mean(data: &[f64]) -> f64 {
    if data.is_empty() { return 0.0; }
    data.iter().sum::<f64>() / data.len() as f64
}

/// Population variance.
pub fn variance(data: &[f64]) -> f64 {
    if data.len() < 2 { return 0.0; }
    let m = mean(data);
    data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / data.len() as f64
}

/// Population standard deviation.
pub fn std_dev(data: &[f64]) -> f64 {
    variance(data).sqrt()
}

/// Median (sorts a copy of the data).
pub fn median(data: &[f64]) -> f64 {
    if data.is_empty() { return 0.0; }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n % 2 == 0 { (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0 } else { sorted[n / 2] }
}

/// Compute a percentile (0..100) using linear interpolation.
pub fn percentile(data: &[f64], p: f64) -> f64 {
    if data.is_empty() { return 0.0; }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p = p.clamp(0.0, 100.0);
    let idx = (p / 100.0) * (sorted.len() - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    if lo == hi { sorted[lo] } else { lerp(sorted[lo], sorted[hi], idx - lo as f64) }
}

/// Histogram with num_bins bins. Returns (edges, counts).
pub fn histogram(data: &[f64], num_bins: usize) -> (Vec<f64>, Vec<usize>) {
    if data.is_empty() || num_bins == 0 { return (vec![], vec![]); }
    let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = if (max - min).abs() < 1e-15 { 1.0 } else { max - min };
    let bin_width = range / num_bins as f64;
    let edges: Vec<f64> = (0..=num_bins).map(|i| min + i as f64 * bin_width).collect();
    let mut counts = vec![0usize; num_bins];
    for &v in data {
        let mut idx = ((v - min) / bin_width).floor() as usize;
        if idx >= num_bins { idx = num_bins - 1; }
        counts[idx] += 1;
    }
    (edges, counts)
}

// ===========================================================================
// Information-theoretic measures
// ===========================================================================

/// Shannon entropy of a discrete probability distribution (in nats).
pub fn entropy(probs: &[f64]) -> f64 {
    probs.iter().filter(|&&p| p > 0.0).map(|&p| -p * p.ln()).sum()
}

/// Shannon entropy in bits.
pub fn entropy_bits(probs: &[f64]) -> f64 {
    entropy(probs) / 2.0_f64.ln()
}

/// Mutual information from a joint probability matrix (row-major). Returns nats.
pub fn mutual_information(joint: &[f64], rows: usize, cols: usize) -> f64 {
    assert_eq!(joint.len(), rows * cols, "Joint matrix size mismatch");
    let mut p_x = vec![0.0; rows];
    let mut p_y = vec![0.0; cols];
    for r in 0..rows {
        for c in 0..cols {
            let p = joint[r * cols + c];
            p_x[r] += p;
            p_y[c] += p;
        }
    }
    let mut mi = 0.0;
    for r in 0..rows {
        for c in 0..cols {
            let p_xy = joint[r * cols + c];
            if p_xy > 0.0 && p_x[r] > 0.0 && p_y[c] > 0.0 {
                mi += p_xy * (p_xy / (p_x[r] * p_y[c])).ln();
            }
        }
    }
    mi
}

// ===========================================================================
// Psychoacoustic / signal helpers
// ===========================================================================

/// Weber fraction: delta_I / I.
pub fn weber_fraction(intensity: f64, delta_intensity: f64) -> f64 {
    if intensity.abs() < 1e-30 { return f64::INFINITY; }
    delta_intensity / intensity
}

/// Signal-to-noise ratio in dB.
pub fn snr_db(signal_power: f64, noise_power: f64) -> f64 {
    if noise_power.abs() < 1e-30 { return f64::INFINITY; }
    10.0 * (signal_power / noise_power).log10()
}

/// Root-mean-square of a signal.
pub fn rms(signal: &[f64]) -> f64 {
    if signal.is_empty() { return 0.0; }
    (signal.iter().map(|x| x * x).sum::<f64>() / signal.len() as f64).sqrt()
}

/// Peak absolute value of a signal.
pub fn peak(signal: &[f64]) -> f64 {
    signal.iter().map(|x| x.abs()).fold(0.0_f64, f64::max)
}

// ===========================================================================
// Small matrix operations
// ===========================================================================

/// 2x2 matrix stored row-major.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mat2(pub [f64; 4]);

impl Mat2 {
    pub const IDENTITY: Self = Self([1.0, 0.0, 0.0, 1.0]);
    pub fn new(data: [f64; 4]) -> Self { Self(data) }
    pub fn get(&self, row: usize, col: usize) -> f64 { self.0[row * 2 + col] }

    pub fn determinant(&self) -> f64 {
        self.0[0] * self.0[3] - self.0[1] * self.0[2]
    }

    pub fn inverse(&self) -> Option<Self> {
        let d = self.determinant();
        if d.abs() < 1e-15 { return None; }
        Some(Self([self.0[3] / d, -self.0[1] / d, -self.0[2] / d, self.0[0] / d]))
    }

    pub fn mul_vec(&self, v: [f64; 2]) -> [f64; 2] {
        [self.0[0] * v[0] + self.0[1] * v[1], self.0[2] * v[0] + self.0[3] * v[1]]
    }

    pub fn mul_mat(&self, rhs: &Mat2) -> Mat2 {
        let (a, b) = (&self.0, &rhs.0);
        Mat2([
            a[0]*b[0]+a[1]*b[2], a[0]*b[1]+a[1]*b[3],
            a[2]*b[0]+a[3]*b[2], a[2]*b[1]+a[3]*b[3],
        ])
    }

    pub fn transpose(&self) -> Self {
        Self([self.0[0], self.0[2], self.0[1], self.0[3]])
    }
}

/// 3x3 matrix stored row-major.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mat3(pub [f64; 9]);

impl Mat3 {
    pub const IDENTITY: Self = Self([1.0,0.0,0.0, 0.0,1.0,0.0, 0.0,0.0,1.0]);
    pub fn new(data: [f64; 9]) -> Self { Self(data) }
    pub fn get(&self, row: usize, col: usize) -> f64 { self.0[row * 3 + col] }

    pub fn determinant(&self) -> f64 {
        let m = &self.0;
        m[0]*(m[4]*m[8]-m[5]*m[7]) - m[1]*(m[3]*m[8]-m[5]*m[6]) + m[2]*(m[3]*m[7]-m[4]*m[6])
    }

    pub fn transpose(&self) -> Self {
        let m = &self.0;
        Self([m[0],m[3],m[6], m[1],m[4],m[7], m[2],m[5],m[8]])
    }

    pub fn mul_vec(&self, v: [f64; 3]) -> [f64; 3] {
        let m = &self.0;
        [
            m[0]*v[0]+m[1]*v[1]+m[2]*v[2],
            m[3]*v[0]+m[4]*v[1]+m[5]*v[2],
            m[6]*v[0]+m[7]*v[1]+m[8]*v[2],
        ]
    }

    pub fn mul_mat(&self, rhs: &Mat3) -> Mat3 {
        let (a, b) = (&self.0, &rhs.0);
        let mut r = [0.0; 9];
        for i in 0..3 {
            for j in 0..3 {
                r[i*3+j] = a[i*3]*b[j] + a[i*3+1]*b[3+j] + a[i*3+2]*b[6+j];
            }
        }
        Mat3(r)
    }

    pub fn inverse(&self) -> Option<Self> {
        let d = self.determinant();
        if d.abs() < 1e-15 { return None; }
        let m = &self.0;
        let inv_d = 1.0 / d;
        Some(Self([
            (m[4]*m[8]-m[5]*m[7])*inv_d, (m[2]*m[7]-m[1]*m[8])*inv_d, (m[1]*m[5]-m[2]*m[4])*inv_d,
            (m[5]*m[6]-m[3]*m[8])*inv_d, (m[0]*m[8]-m[2]*m[6])*inv_d, (m[2]*m[3]-m[0]*m[5])*inv_d,
            (m[3]*m[7]-m[4]*m[6])*inv_d, (m[1]*m[6]-m[0]*m[7])*inv_d, (m[0]*m[4]-m[1]*m[3])*inv_d,
        ]))
    }
}

/// 4x4 matrix stored row-major.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mat4(pub [f64; 16]);

impl Mat4 {
    pub const IDENTITY: Self = Self([
        1.0,0.0,0.0,0.0, 0.0,1.0,0.0,0.0, 0.0,0.0,1.0,0.0, 0.0,0.0,0.0,1.0,
    ]);
    pub fn new(data: [f64; 16]) -> Self { Self(data) }
    pub fn get(&self, row: usize, col: usize) -> f64 { self.0[row * 4 + col] }

    pub fn transpose(&self) -> Self {
        let m = &self.0;
        let mut r = [0.0; 16];
        for i in 0..4 { for j in 0..4 { r[i*4+j] = m[j*4+i]; } }
        Self(r)
    }

    pub fn mul_vec(&self, v: [f64; 4]) -> [f64; 4] {
        let m = &self.0;
        let mut r = [0.0; 4];
        for i in 0..4 {
            r[i] = m[i*4]*v[0] + m[i*4+1]*v[1] + m[i*4+2]*v[2] + m[i*4+3]*v[3];
        }
        r
    }

    pub fn mul_mat(&self, rhs: &Mat4) -> Mat4 {
        let (a, b) = (&self.0, &rhs.0);
        let mut r = [0.0; 16];
        for i in 0..4 {
            for j in 0..4 {
                r[i*4+j] = a[i*4]*b[j] + a[i*4+1]*b[4+j] + a[i*4+2]*b[8+j] + a[i*4+3]*b[12+j];
            }
        }
        Mat4(r)
    }

    pub fn determinant(&self) -> f64 {
        let m = &self.0;
        let minor = |r0: usize, r1: usize, r2: usize, c0: usize, c1: usize, c2: usize| -> f64 {
            m[r0*4+c0]*(m[r1*4+c1]*m[r2*4+c2]-m[r1*4+c2]*m[r2*4+c1])
            - m[r0*4+c1]*(m[r1*4+c0]*m[r2*4+c2]-m[r1*4+c2]*m[r2*4+c0])
            + m[r0*4+c2]*(m[r1*4+c0]*m[r2*4+c1]-m[r1*4+c1]*m[r2*4+c0])
        };
        m[0]*minor(1,2,3,1,2,3) - m[1]*minor(1,2,3,0,2,3)
        + m[2]*minor(1,2,3,0,1,3) - m[3]*minor(1,2,3,0,1,2)
    }
}

/// A single FFT frequency bin with magnitude and phase.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrequencyBin {
    pub index: usize,
    pub frequency_hz: f64,
    pub magnitude: f64,
    pub phase: f64,
}

impl FrequencyBin {
    pub fn from_complex(index: usize, c: Complex, sample_rate: f64, fft_size: usize) -> Self {
        Self {
            index,
            frequency_hz: index as f64 * sample_rate / fft_size as f64,
            magnitude: c.magnitude(),
            phase: c.phase(),
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    const EPS: f64 = 1e-9;

    #[test]
    fn complex_arithmetic() {
        let a = Complex::new(1.0, 2.0);
        let b = Complex::new(3.0, 4.0);
        let sum = a + b;
        assert!((sum.re - 4.0).abs() < EPS && (sum.im - 6.0).abs() < EPS);
        let prod = a * b;
        assert!((prod.re - (-5.0)).abs() < EPS && (prod.im - 10.0).abs() < EPS);
    }

    #[test]
    fn complex_magnitude_phase() {
        let c = Complex::new(3.0, 4.0);
        assert!((c.magnitude() - 5.0).abs() < EPS);
        let c2 = Complex::from_polar(5.0, c.phase());
        assert!((c2.re - 3.0).abs() < 1e-6 && (c2.im - 4.0).abs() < 1e-6);
    }

    #[test]
    fn complex_inverse() {
        let c = Complex::new(1.0, 1.0);
        let prod = c * c.inverse();
        assert!((prod.re - 1.0).abs() < EPS && prod.im.abs() < EPS);
    }

    #[test]
    fn fft_dc_signal() {
        let mut buf = vec![Complex::new(1.0, 0.0); 8];
        fft(&mut buf);
        assert!((buf[0].re - 8.0).abs() < EPS);
        for i in 1..8 { assert!(buf[i].magnitude() < EPS); }
    }

    #[test]
    fn fft_ifft_roundtrip() {
        let original: Vec<Complex> = (0..16)
            .map(|i| Complex::new((i as f64 * 0.3).sin(), 0.0))
            .collect();
        let mut buf = original.clone();
        fft(&mut buf);
        ifft(&mut buf);
        for (a, b) in original.iter().zip(buf.iter()) {
            assert!((a.re - b.re).abs() < 1e-10 && (a.im - b.im).abs() < 1e-10);
        }
    }

    #[test]
    fn lerp_test() {
        assert!((lerp(0.0, 10.0, 0.5) - 5.0).abs() < EPS);
    }

    #[test]
    fn cubic_endpoints() {
        assert!((cubic_interpolate(0.0, 5.0, 10.0, 15.0, 0.0) - 5.0).abs() < EPS);
        assert!((cubic_interpolate(0.0, 5.0, 10.0, 15.0, 1.0) - 10.0).abs() < EPS);
    }

    #[test]
    fn gaussian_peak() {
        assert!((gaussian(5.0, 5.0, 1.0) - 1.0).abs() < EPS);
    }

    #[test]
    fn sigmoid_symmetry() {
        assert!((sigmoid(0.0) - 0.5).abs() < EPS);
        assert!((sigmoid(5.0) + sigmoid(-5.0) - 1.0).abs() < EPS);
    }

    #[test]
    fn softmax_sums_to_one() {
        let sm = softmax(&[1.0, 2.0, 3.0, 4.0]);
        assert!((sm.iter().sum::<f64>() - 1.0).abs() < EPS);
        for i in 1..sm.len() { assert!(sm[i] > sm[i - 1]); }
    }

    #[test]
    fn horner_polynomial() {
        assert!((horner(&[2.0, 3.0, 1.0], 2.0) - 12.0).abs() < EPS);
    }

    #[test]
    fn statistics_basic() {
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        assert!((mean(&data) - 5.0).abs() < EPS);
        assert!((median(&data) - 4.5).abs() < EPS);
    }

    #[test]
    fn histogram_basic() {
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let (edges, counts) = histogram(&data, 5);
        assert_eq!(edges.len(), 6);
        assert_eq!(counts.iter().sum::<usize>(), 10);
    }

    #[test]
    fn entropy_uniform() {
        let h = entropy_bits(&[0.25, 0.25, 0.25, 0.25]);
        assert!((h - 2.0).abs() < EPS);
    }

    #[test]
    fn mutual_information_independent() {
        let mi = mutual_information(&[0.25, 0.25, 0.25, 0.25], 2, 2);
        assert!(mi.abs() < EPS);
    }

    #[test]
    fn weber_and_snr() {
        assert!((weber_fraction(100.0, 1.0) - 0.01).abs() < EPS);
        assert!((snr_db(100.0, 1.0) - 20.0).abs() < EPS);
    }

    #[test]
    fn mat2_inverse() {
        let m = Mat2::new([1.0, 2.0, 3.0, 4.0]);
        let inv = m.inverse().unwrap();
        let prod = m.mul_mat(&inv);
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((prod.get(i, j) - expected).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn mat3_determinant() {
        let m = Mat3::new([1.0,2.0,3.0, 0.0,1.0,4.0, 5.0,6.0,0.0]);
        assert!((m.determinant() - 1.0).abs() < EPS);
    }

    #[test]
    fn mat4_identity() {
        assert_eq!(Mat4::IDENTITY.mul_vec([1.0,2.0,3.0,4.0]), [1.0,2.0,3.0,4.0]);
    }

    #[test]
    fn magnitude_spectrum_pure_tone() {
        let n = 64;
        let sr = 64.0;
        let freq = 8.0;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * freq * i as f64 / sr).sin())
            .collect();
        let spec = magnitude_spectrum(&signal);
        let peak_bin = spec.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        assert_eq!(peak_bin, 8);
    }

    #[test]
    fn rms_and_peak_test() {
        let s = vec![1.0, -1.0, 1.0, -1.0];
        assert!((rms(&s) - 1.0).abs() < EPS);
        assert!((peak(&s) - 1.0).abs() < EPS);
    }

    #[test]
    fn table_lookup_test() {
        let table = vec![(0.0, 0.0), (1.0, 10.0), (2.0, 20.0)];
        assert!((table_lookup(&table, 0.5) - 5.0).abs() < EPS);
        assert!((table_lookup(&table, -1.0)).abs() < EPS);
    }
}
