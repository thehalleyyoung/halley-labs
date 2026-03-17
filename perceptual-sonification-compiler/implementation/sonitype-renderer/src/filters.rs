//! Filter implementations: biquad, one-pole, state-variable, comb, cascade,
//! and DC-blocker.
//!
//! All filters operate on mono [`AudioBuf`](crate::AudioBuf) buffers and
//! support sample-accurate coefficient updates.

use std::f64::consts::PI;
use crate::AudioBuf;

const TWO_PI: f64 = 2.0 * PI;

// ---------------------------------------------------------------------------
// FilterType
// ---------------------------------------------------------------------------

/// Biquad filter mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterMode {
    Lowpass,
    Highpass,
    Bandpass,
    Notch,
    Allpass,
    LowShelf,
    HighShelf,
    PeakingEq,
}

// ---------------------------------------------------------------------------
// BiquadCoefficients
// ---------------------------------------------------------------------------

/// Second-order IIR coefficients (normalised: a0 = 1).
#[derive(Debug, Clone, Copy)]
pub struct BiquadCoefficients {
    pub b0: f64,
    pub b1: f64,
    pub b2: f64,
    pub a1: f64,
    pub a2: f64,
}

impl Default for BiquadCoefficients {
    fn default() -> Self {
        // Pass-through
        Self { b0: 1.0, b1: 0.0, b2: 0.0, a1: 0.0, a2: 0.0 }
    }
}

impl BiquadCoefficients {
    /// Compute coefficients for the given mode.
    ///
    /// * `freq` – centre / cutoff frequency in Hz
    /// * `q` – quality factor (resonance)
    /// * `gain_db` – gain in dB (used only for shelving / peaking)
    /// * `sample_rate` – sample rate in Hz
    pub fn design(mode: FilterMode, freq: f64, q: f64, gain_db: f64, sample_rate: f64) -> Self {
        let w0 = TWO_PI * freq / sample_rate;
        let cos_w0 = w0.cos();
        let sin_w0 = w0.sin();
        let alpha = sin_w0 / (2.0 * q);

        let (b0, b1, b2, a0, a1, a2);

        match mode {
            FilterMode::Lowpass => {
                b1 = 1.0 - cos_w0;
                b0 = b1 / 2.0;
                b2 = b0;
                a0 = 1.0 + alpha;
                a1 = -2.0 * cos_w0;
                a2 = 1.0 - alpha;
            }
            FilterMode::Highpass => {
                b1 = -(1.0 + cos_w0);
                b0 = (1.0 + cos_w0) / 2.0;
                b2 = b0;
                a0 = 1.0 + alpha;
                a1 = -2.0 * cos_w0;
                a2 = 1.0 - alpha;
            }
            FilterMode::Bandpass => {
                b0 = alpha;
                b1 = 0.0;
                b2 = -alpha;
                a0 = 1.0 + alpha;
                a1 = -2.0 * cos_w0;
                a2 = 1.0 - alpha;
            }
            FilterMode::Notch => {
                b0 = 1.0;
                b1 = -2.0 * cos_w0;
                b2 = 1.0;
                a0 = 1.0 + alpha;
                a1 = -2.0 * cos_w0;
                a2 = 1.0 - alpha;
            }
            FilterMode::Allpass => {
                b0 = 1.0 - alpha;
                b1 = -2.0 * cos_w0;
                b2 = 1.0 + alpha;
                a0 = 1.0 + alpha;
                a1 = -2.0 * cos_w0;
                a2 = 1.0 - alpha;
            }
            FilterMode::LowShelf => {
                let a_lin = 10.0_f64.powf(gain_db / 40.0);
                let two_sqrt_a_alpha = 2.0 * a_lin.sqrt() * alpha;
                b0 = a_lin * ((a_lin + 1.0) - (a_lin - 1.0) * cos_w0 + two_sqrt_a_alpha);
                b1 = 2.0 * a_lin * ((a_lin - 1.0) - (a_lin + 1.0) * cos_w0);
                b2 = a_lin * ((a_lin + 1.0) - (a_lin - 1.0) * cos_w0 - two_sqrt_a_alpha);
                a0 = (a_lin + 1.0) + (a_lin - 1.0) * cos_w0 + two_sqrt_a_alpha;
                a1 = -2.0 * ((a_lin - 1.0) + (a_lin + 1.0) * cos_w0);
                a2 = (a_lin + 1.0) + (a_lin - 1.0) * cos_w0 - two_sqrt_a_alpha;
            }
            FilterMode::HighShelf => {
                let a_lin = 10.0_f64.powf(gain_db / 40.0);
                let two_sqrt_a_alpha = 2.0 * a_lin.sqrt() * alpha;
                b0 = a_lin * ((a_lin + 1.0) + (a_lin - 1.0) * cos_w0 + two_sqrt_a_alpha);
                b1 = -2.0 * a_lin * ((a_lin - 1.0) + (a_lin + 1.0) * cos_w0);
                b2 = a_lin * ((a_lin + 1.0) + (a_lin - 1.0) * cos_w0 - two_sqrt_a_alpha);
                a0 = (a_lin + 1.0) - (a_lin - 1.0) * cos_w0 + two_sqrt_a_alpha;
                a1 = 2.0 * ((a_lin - 1.0) - (a_lin + 1.0) * cos_w0);
                a2 = (a_lin + 1.0) - (a_lin - 1.0) * cos_w0 - two_sqrt_a_alpha;
            }
            FilterMode::PeakingEq => {
                let a_lin = 10.0_f64.powf(gain_db / 40.0);
                b0 = 1.0 + alpha * a_lin;
                b1 = -2.0 * cos_w0;
                b2 = 1.0 - alpha * a_lin;
                a0 = 1.0 + alpha / a_lin;
                a1 = -2.0 * cos_w0;
                a2 = 1.0 - alpha / a_lin;
            }
        }

        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
        }
    }

    /// Linear interpolation between two coefficient sets.
    pub fn lerp(a: &Self, b: &Self, t: f64) -> Self {
        Self {
            b0: a.b0 + (b.b0 - a.b0) * t,
            b1: a.b1 + (b.b1 - a.b1) * t,
            b2: a.b2 + (b.b2 - a.b2) * t,
            a1: a.a1 + (b.a1 - a.a1) * t,
            a2: a.a2 + (b.a2 - a.a2) * t,
        }
    }
}

// ---------------------------------------------------------------------------
// BiquadFilter
// ---------------------------------------------------------------------------

/// Second-order IIR filter (Direct Form II Transposed).
#[derive(Debug, Clone)]
pub struct BiquadFilter {
    pub coeffs: BiquadCoefficients,
    target_coeffs: BiquadCoefficients,
    /// State variables for DFII-T.
    z1: f64,
    z2: f64,
    pub mode: FilterMode,
    pub sample_rate: f64,
    interpolation_samples: usize,
    interpolation_counter: usize,
    start_coeffs: BiquadCoefficients,
}

impl BiquadFilter {
    pub fn new(mode: FilterMode, freq: f64, q: f64, sample_rate: f64) -> Self {
        let coeffs = BiquadCoefficients::design(mode, freq, q, 0.0, sample_rate);
        Self {
            coeffs,
            target_coeffs: coeffs,
            z1: 0.0,
            z2: 0.0,
            mode,
            sample_rate,
            interpolation_samples: 0,
            interpolation_counter: 0,
            start_coeffs: coeffs,
        }
    }

    /// Recalculate coefficients; optionally interpolate over `smooth_samples`.
    pub fn set_parameters(&mut self, freq: f64, q: f64, gain_db: f64, smooth_samples: usize) {
        let new = BiquadCoefficients::design(self.mode, freq, q, gain_db, self.sample_rate);
        if smooth_samples == 0 {
            self.coeffs = new;
            self.target_coeffs = new;
            self.interpolation_samples = 0;
        } else {
            self.start_coeffs = self.coeffs;
            self.target_coeffs = new;
            self.interpolation_samples = smooth_samples;
            self.interpolation_counter = 0;
        }
    }

    pub fn set_mode(&mut self, mode: FilterMode) {
        self.mode = mode;
    }

    pub fn reset(&mut self) {
        self.z1 = 0.0;
        self.z2 = 0.0;
    }

    /// Process a single sample.
    #[inline]
    pub fn tick(&mut self, input: f64) -> f64 {
        // Coefficient interpolation
        if self.interpolation_samples > 0 && self.interpolation_counter < self.interpolation_samples
        {
            let t = self.interpolation_counter as f64 / self.interpolation_samples as f64;
            self.coeffs = BiquadCoefficients::lerp(&self.start_coeffs, &self.target_coeffs, t);
            self.interpolation_counter += 1;
            if self.interpolation_counter >= self.interpolation_samples {
                self.coeffs = self.target_coeffs;
            }
        }

        // Direct Form II Transposed
        let output = self.coeffs.b0 * input + self.z1;
        self.z1 = self.coeffs.b1 * input - self.coeffs.a1 * output + self.z2;
        self.z2 = self.coeffs.b2 * input - self.coeffs.a2 * output;
        output
    }

    /// Process a buffer in-place.
    pub fn process(&mut self, buf: &mut AudioBuf) {
        let frames = buf.frames();
        let ch = buf.channels;
        for f in 0..frames {
            for c in 0..ch {
                let inp = buf.get(f, c) as f64;
                let out = self.tick(inp);
                buf.set(f, c, out as f32);
            }
        }
    }

    /// Process input buffer → output buffer (mono).
    pub fn process_buf(&mut self, input: &AudioBuf, output: &mut AudioBuf) {
        let frames = input.frames().min(output.frames());
        for f in 0..frames {
            let inp = input.get(f, 0) as f64;
            let out = self.tick(inp);
            output.set(f, 0, out as f32);
        }
    }
}

// ---------------------------------------------------------------------------
// OnePoleFilter
// ---------------------------------------------------------------------------

/// Simple first-order (one-pole) filter for parameter smoothing and gentle
/// tone shaping.
#[derive(Debug, Clone)]
pub struct OnePoleFilter {
    coeff: f64,
    prev: f64,
    pub highpass: bool,
}

impl OnePoleFilter {
    /// Create a lowpass one-pole with the given cutoff frequency.
    pub fn lowpass(cutoff: f64, sample_rate: f64) -> Self {
        let coeff = (-TWO_PI * cutoff / sample_rate).exp();
        Self { coeff, prev: 0.0, highpass: false }
    }

    /// Create a highpass one-pole with the given cutoff frequency.
    pub fn highpass(cutoff: f64, sample_rate: f64) -> Self {
        let coeff = (-TWO_PI * cutoff / sample_rate).exp();
        Self { coeff, prev: 0.0, highpass: true }
    }

    pub fn set_cutoff(&mut self, cutoff: f64, sample_rate: f64) {
        self.coeff = (-TWO_PI * cutoff / sample_rate).exp();
    }

    pub fn reset(&mut self) {
        self.prev = 0.0;
    }

    #[inline]
    pub fn tick(&mut self, input: f64) -> f64 {
        let lp = input * (1.0 - self.coeff) + self.prev * self.coeff;
        self.prev = lp;
        if self.highpass { input - lp } else { lp }
    }

    pub fn process(&mut self, buf: &mut AudioBuf) {
        let frames = buf.frames();
        for f in 0..frames {
            let inp = buf.get(f, 0) as f64;
            buf.set(f, 0, self.tick(inp) as f32);
        }
    }
}

// ---------------------------------------------------------------------------
// StateVariableFilter
// ---------------------------------------------------------------------------

/// Multi-mode state-variable filter providing simultaneous lowpass, highpass,
/// bandpass, and notch outputs.
#[derive(Debug, Clone)]
pub struct StateVariableFilter {
    pub cutoff: f64,
    pub resonance: f64,
    pub sample_rate: f64,
    ic1: f64,
    ic2: f64,
}

/// Outputs from the state-variable filter.
#[derive(Debug, Clone, Copy, Default)]
pub struct SvfOutputs {
    pub lowpass: f64,
    pub highpass: f64,
    pub bandpass: f64,
    pub notch: f64,
}

impl StateVariableFilter {
    pub fn new(cutoff: f64, resonance: f64, sample_rate: f64) -> Self {
        Self { cutoff, resonance, sample_rate, ic1: 0.0, ic2: 0.0 }
    }

    pub fn set_cutoff(&mut self, cutoff: f64) {
        self.cutoff = cutoff;
    }

    pub fn set_resonance(&mut self, resonance: f64) {
        self.resonance = resonance;
    }

    pub fn reset(&mut self) {
        self.ic1 = 0.0;
        self.ic2 = 0.0;
    }

    /// Process one sample and return all four outputs.
    #[inline]
    pub fn tick(&mut self, input: f64) -> SvfOutputs {
        let g = (PI * self.cutoff / self.sample_rate).tan();
        let k = 2.0 - 2.0 * self.resonance; // damping factor
        let a1 = 1.0 / (1.0 + g * (g + k));
        let a2 = g * a1;
        let a3 = g * a2;

        let v3 = input - self.ic2;
        let v1 = a1 * self.ic1 + a2 * v3;
        let v2 = self.ic2 + a2 * self.ic1 + a3 * v3;
        self.ic1 = 2.0 * v1 - self.ic1;
        self.ic2 = 2.0 * v2 - self.ic2;

        SvfOutputs {
            lowpass: v2,
            bandpass: v1,
            highpass: input - k * v1 - v2,
            notch: input - k * v1,
        }
    }

    /// Process a buffer, writing the lowpass output in-place.
    pub fn process_lowpass(&mut self, buf: &mut AudioBuf) {
        let frames = buf.frames();
        for f in 0..frames {
            let inp = buf.get(f, 0) as f64;
            let out = self.tick(inp);
            buf.set(f, 0, out.lowpass as f32);
        }
    }

    /// Process a buffer, writing all four outputs into consecutive channels
    /// (requires ≥ 4 channels).
    pub fn process_multi(&mut self, input: &AudioBuf, output: &mut AudioBuf) {
        let frames = input.frames().min(output.frames());
        for f in 0..frames {
            let inp = input.get(f, 0) as f64;
            let out = self.tick(inp);
            output.set(f, 0, out.lowpass as f32);
            if output.channels > 1 { output.set(f, 1, out.highpass as f32); }
            if output.channels > 2 { output.set(f, 2, out.bandpass as f32); }
            if output.channels > 3 { output.set(f, 3, out.notch as f32); }
        }
    }
}

// ---------------------------------------------------------------------------
// CombFilter
// ---------------------------------------------------------------------------

/// Feed-forward and feedback comb filter.
#[derive(Debug, Clone)]
pub struct CombFilter {
    buffer: Vec<f64>,
    write_pos: usize,
    pub delay_samples: usize,
    /// Feed-forward gain (0 = disabled).
    pub ff_gain: f64,
    /// Feedback gain (0 = disabled).
    pub fb_gain: f64,
    /// Damping filter coefficient for the feedback path.
    pub damping: f64,
    damp_state: f64,
}

impl CombFilter {
    pub fn new(max_delay: usize, delay_samples: usize, ff_gain: f64, fb_gain: f64) -> Self {
        Self {
            buffer: vec![0.0; max_delay],
            write_pos: 0,
            delay_samples: delay_samples.min(max_delay),
            ff_gain,
            fb_gain,
            damping: 0.0,
            damp_state: 0.0,
        }
    }

    pub fn set_delay(&mut self, samples: usize) {
        self.delay_samples = samples.min(self.buffer.len());
    }

    pub fn set_damping(&mut self, d: f64) {
        self.damping = d.clamp(0.0, 1.0);
    }

    pub fn reset(&mut self) {
        for s in self.buffer.iter_mut() { *s = 0.0; }
        self.damp_state = 0.0;
        self.write_pos = 0;
    }

    #[inline]
    fn read_delayed(&self) -> f64 {
        let len = self.buffer.len();
        let idx = (self.write_pos + len - self.delay_samples) % len;
        self.buffer[idx]
    }

    #[inline]
    pub fn tick(&mut self, input: f64) -> f64 {
        let delayed = self.read_delayed();
        // Damping low-pass on feedback path
        let damped = delayed * (1.0 - self.damping) + self.damp_state * self.damping;
        self.damp_state = damped;

        let output = input + self.ff_gain * delayed;
        self.buffer[self.write_pos] = input + self.fb_gain * damped;
        self.write_pos = (self.write_pos + 1) % self.buffer.len();
        output
    }

    pub fn process(&mut self, buf: &mut AudioBuf) {
        let frames = buf.frames();
        for f in 0..frames {
            let inp = buf.get(f, 0) as f64;
            buf.set(f, 0, self.tick(inp) as f32);
        }
    }
}

// ---------------------------------------------------------------------------
// AllpassSection (used by Reverb & BiquadCascade helper)
// ---------------------------------------------------------------------------

/// First-order allpass section used inside reverb and comb cascades.
#[derive(Debug, Clone)]
pub struct AllpassSection {
    buffer: Vec<f64>,
    write_pos: usize,
    delay_samples: usize,
    pub gain: f64,
}

impl AllpassSection {
    pub fn new(delay_samples: usize, gain: f64) -> Self {
        Self {
            buffer: vec![0.0; delay_samples.max(1)],
            write_pos: 0,
            delay_samples,
            gain,
        }
    }

    pub fn reset(&mut self) {
        for s in self.buffer.iter_mut() { *s = 0.0; }
        self.write_pos = 0;
    }

    #[inline]
    pub fn tick(&mut self, input: f64) -> f64 {
        let delayed = self.buffer[self.write_pos];
        let temp = input + delayed * self.gain;
        self.buffer[self.write_pos] = temp;
        self.write_pos = (self.write_pos + 1) % self.buffer.len();
        delayed - temp * self.gain
    }
}

// ---------------------------------------------------------------------------
// BiquadCascade
// ---------------------------------------------------------------------------

/// Chain of biquad sections for higher-order filtering.
#[derive(Debug, Clone)]
pub struct BiquadCascade {
    sections: Vec<BiquadFilter>,
}

impl BiquadCascade {
    pub fn new() -> Self {
        Self { sections: Vec::new() }
    }

    pub fn add_section(&mut self, filter: BiquadFilter) {
        self.sections.push(filter);
    }

    /// Build a Butterworth lowpass of the given order (must be even).
    pub fn butterworth_lowpass(order: usize, freq: f64, sample_rate: f64) -> Self {
        assert!(order % 2 == 0 && order >= 2, "order must be even and >= 2");
        let num_sections = order / 2;
        let mut cascade = Self::new();
        for k in 0..num_sections {
            let q = 1.0 / (2.0 * (PI * (2 * k + 1) as f64 / (2.0 * order as f64)).cos());
            cascade.add_section(BiquadFilter::new(FilterMode::Lowpass, freq, q, sample_rate));
        }
        cascade
    }

    /// Build a Butterworth highpass of the given order (must be even).
    pub fn butterworth_highpass(order: usize, freq: f64, sample_rate: f64) -> Self {
        assert!(order % 2 == 0 && order >= 2);
        let num_sections = order / 2;
        let mut cascade = Self::new();
        for k in 0..num_sections {
            let q = 1.0 / (2.0 * (PI * (2 * k + 1) as f64 / (2.0 * order as f64)).cos());
            cascade.add_section(BiquadFilter::new(FilterMode::Highpass, freq, q, sample_rate));
        }
        cascade
    }

    /// Build a Chebyshev Type-I lowpass (approximation via Q factors).
    pub fn chebyshev_lowpass(order: usize, freq: f64, ripple_db: f64, sample_rate: f64) -> Self {
        assert!(order % 2 == 0 && order >= 2);
        let num_sections = order / 2;
        let epsilon = (10.0_f64.powf(ripple_db / 10.0) - 1.0).sqrt();
        let mut cascade = Self::new();
        for k in 0..num_sections {
            let theta = PI * (2 * k + 1) as f64 / (2.0 * order as f64);
            let sigma = -(1.0 / order as f64) * (1.0 / epsilon).asinh() * theta.sin();
            let omega = (1.0 / order as f64) * (1.0 / epsilon).asinh() * theta.cos();
            let q = 1.0 / (2.0 * sigma.abs()).max(0.5);
            let _ = omega; // used in precise designs; simplified here
            cascade.add_section(BiquadFilter::new(FilterMode::Lowpass, freq, q, sample_rate));
        }
        cascade
    }

    /// Build a Bessel lowpass (maximally flat group delay).
    pub fn bessel_lowpass(order: usize, freq: f64, sample_rate: f64) -> Self {
        assert!(order % 2 == 0 && order >= 2);
        let num_sections = order / 2;
        // Pre-computed Bessel Q factors for common orders
        let qs: Vec<f64> = match order {
            2 => vec![0.5774],
            4 => vec![0.8055, 0.5219],
            6 => vec![1.0234, 0.6112, 0.5103],
            8 => vec![1.2257, 0.7104, 0.5591, 0.5060],
            _ => (0..num_sections).map(|_| 0.7071).collect(),
        };
        let mut cascade = Self::new();
        for k in 0..num_sections {
            let q = qs.get(k).copied().unwrap_or(0.7071);
            cascade.add_section(BiquadFilter::new(FilterMode::Lowpass, freq, q, sample_rate));
        }
        cascade
    }

    pub fn reset(&mut self) {
        for s in &mut self.sections { s.reset(); }
    }

    pub fn order(&self) -> usize {
        self.sections.len() * 2
    }

    #[inline]
    pub fn tick(&mut self, mut input: f64) -> f64 {
        for section in &mut self.sections {
            input = section.tick(input);
        }
        input
    }

    pub fn process(&mut self, buf: &mut AudioBuf) {
        let frames = buf.frames();
        for f in 0..frames {
            let inp = buf.get(f, 0) as f64;
            buf.set(f, 0, self.tick(inp) as f32);
        }
    }
}

// ---------------------------------------------------------------------------
// DCBlocker
// ---------------------------------------------------------------------------

/// Simple DC-blocking filter (highpass at ~5 Hz equivalent).
///
/// y[n] = x[n] − x[n−1] + R · y[n−1], R ≈ 0.995.
#[derive(Debug, Clone)]
pub struct DCBlocker {
    r: f64,
    prev_in: f64,
    prev_out: f64,
}

impl DCBlocker {
    pub fn new() -> Self {
        Self { r: 0.995, prev_in: 0.0, prev_out: 0.0 }
    }

    pub fn with_r(r: f64) -> Self {
        Self { r, prev_in: 0.0, prev_out: 0.0 }
    }

    pub fn reset(&mut self) {
        self.prev_in = 0.0;
        self.prev_out = 0.0;
    }

    #[inline]
    pub fn tick(&mut self, input: f64) -> f64 {
        let output = input - self.prev_in + self.r * self.prev_out;
        self.prev_in = input;
        self.prev_out = output;
        output
    }

    pub fn process(&mut self, buf: &mut AudioBuf) {
        let frames = buf.frames();
        for f in 0..frames {
            let inp = buf.get(f, 0) as f64;
            buf.set(f, 0, self.tick(inp) as f32);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_buf(frames: usize) -> AudioBuf {
        AudioBuf::new(frames, 1, 44100)
    }

    fn impulse(frames: usize) -> AudioBuf {
        let mut b = make_buf(frames);
        b.set(0, 0, 1.0);
        b
    }

    // -- BiquadCoefficients ---------------------------------------------------

    #[test]
    fn lowpass_unity_at_dc() {
        let c = BiquadCoefficients::design(FilterMode::Lowpass, 1000.0, 0.7071, 0.0, 44100.0);
        // DC gain = (b0+b1+b2) / (1+a1+a2)
        let dc = (c.b0 + c.b1 + c.b2) / (1.0 + c.a1 + c.a2);
        assert!((dc - 1.0).abs() < 1e-6, "dc gain {dc}");
    }

    #[test]
    fn highpass_zero_at_dc() {
        let c = BiquadCoefficients::design(FilterMode::Highpass, 1000.0, 0.7071, 0.0, 44100.0);
        let dc = (c.b0 + c.b1 + c.b2) / (1.0 + c.a1 + c.a2);
        assert!(dc.abs() < 1e-6, "hp dc gain should be 0, got {dc}");
    }

    #[test]
    fn bandpass_zero_at_dc() {
        let c = BiquadCoefficients::design(FilterMode::Bandpass, 1000.0, 1.0, 0.0, 44100.0);
        let dc = (c.b0 + c.b1 + c.b2) / (1.0 + c.a1 + c.a2);
        assert!(dc.abs() < 1e-6);
    }

    // -- BiquadFilter ---------------------------------------------------------

    #[test]
    fn biquad_lowpass_attenuates_high_freq() {
        let mut filt = BiquadFilter::new(FilterMode::Lowpass, 200.0, 0.7071, 44100.0);
        // Feed 10 kHz sine
        let sr = 44100.0;
        let mut buf = make_buf(4096);
        for f in 0..buf.frames() {
            let s = (2.0 * PI * 10000.0 * f as f64 / sr).sin();
            buf.set(f, 0, s as f32);
        }
        filt.process(&mut buf);
        // RMS of output should be much lower than input
        let rms: f64 = (buf.data.iter().map(|&s| (s as f64).powi(2)).sum::<f64>()
            / buf.data.len() as f64)
            .sqrt();
        assert!(rms < 0.05, "high freq not attenuated: rms={rms}");
    }

    #[test]
    fn biquad_reset_clears_state() {
        let mut filt = BiquadFilter::new(FilterMode::Lowpass, 1000.0, 0.7071, 44100.0);
        filt.tick(1.0);
        filt.tick(0.5);
        filt.reset();
        assert!((filt.tick(0.0)).abs() < 1e-12);
    }

    #[test]
    fn biquad_coefficient_interpolation() {
        let mut filt = BiquadFilter::new(FilterMode::Lowpass, 1000.0, 0.7071, 44100.0);
        filt.set_parameters(5000.0, 0.7071, 0.0, 100);
        for _ in 0..100 {
            filt.tick(0.5);
        }
        // After interpolation completes, coeffs should equal target
        assert!((filt.coeffs.b0 - filt.target_coeffs.b0).abs() < 1e-10);
    }

    // -- OnePoleFilter --------------------------------------------------------

    #[test]
    fn one_pole_lowpass_smooths() {
        let mut f = OnePoleFilter::lowpass(100.0, 44100.0);
        let mut prev = 0.0f64;
        // Step input
        for _ in 0..1000 {
            prev = f.tick(1.0);
        }
        assert!((prev - 1.0).abs() < 0.01, "should converge to 1.0");
    }

    #[test]
    fn one_pole_highpass_blocks_dc() {
        let mut f = OnePoleFilter::highpass(100.0, 44100.0);
        let mut out = 0.0;
        for _ in 0..10000 {
            out = f.tick(1.0);
        }
        assert!(out.abs() < 0.01, "HP should block DC: {out}");
    }

    // -- StateVariableFilter --------------------------------------------------

    #[test]
    fn svf_lowpass_passes_dc() {
        let mut svf = StateVariableFilter::new(5000.0, 0.5, 44100.0);
        let mut out = SvfOutputs::default();
        for _ in 0..10000 {
            out = svf.tick(1.0);
        }
        assert!((out.lowpass - 1.0).abs() < 0.01);
    }

    #[test]
    fn svf_simultaneous_outputs() {
        let mut svf = StateVariableFilter::new(1000.0, 0.5, 44100.0);
        let out = svf.tick(1.0);
        // All outputs should have been computed
        let _sum = out.lowpass + out.highpass + out.bandpass + out.notch;
    }

    // -- CombFilter -----------------------------------------------------------

    #[test]
    fn comb_feedforward_echo() {
        let mut comb = CombFilter::new(1024, 10, 1.0, 0.0);
        let mut out = vec![];
        for i in 0..20 {
            let inp = if i == 0 { 1.0 } else { 0.0 };
            out.push(comb.tick(inp));
        }
        // Original impulse at t=0, echo at t=10
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[10] - 1.0).abs() < 1e-6);
    }

    // -- BiquadCascade --------------------------------------------------------

    #[test]
    fn butterworth_4th_order() {
        let mut casc = BiquadCascade::butterworth_lowpass(4, 1000.0, 44100.0);
        assert_eq!(casc.order(), 4);
        let mut buf = impulse(1024);
        casc.process(&mut buf);
        // Should produce non-zero output
        let energy: f64 = buf.data.iter().map(|&s| (s as f64).powi(2)).sum();
        assert!(energy > 0.0);
    }

    #[test]
    fn bessel_lowpass_produces_output() {
        let mut casc = BiquadCascade::bessel_lowpass(4, 2000.0, 44100.0);
        let mut buf = impulse(512);
        casc.process(&mut buf);
        let energy: f64 = buf.data.iter().map(|&s| (s as f64).powi(2)).sum();
        assert!(energy > 0.0);
    }

    // -- DCBlocker ------------------------------------------------------------

    #[test]
    fn dc_blocker_removes_dc() {
        let mut dc = DCBlocker::new();
        let mut out = 0.0;
        for _ in 0..100_000 {
            out = dc.tick(1.0);
        }
        assert!(out.abs() < 0.01, "DC not blocked: {out}");
    }

    #[test]
    fn dc_blocker_passes_ac() {
        let mut dc = DCBlocker::new();
        let sr = 44100.0;
        let mut energy = 0.0f64;
        for i in 0..4096 {
            let inp = (2.0 * PI * 1000.0 * i as f64 / sr).sin();
            let out = dc.tick(inp);
            energy += out * out;
        }
        assert!(energy > 100.0, "AC signal should pass through");
    }

    #[test]
    fn coefficient_lerp() {
        let a = BiquadCoefficients { b0: 0.0, b1: 0.0, b2: 0.0, a1: 0.0, a2: 0.0 };
        let b = BiquadCoefficients { b0: 1.0, b1: 2.0, b2: 3.0, a1: 4.0, a2: 5.0 };
        let mid = BiquadCoefficients::lerp(&a, &b, 0.5);
        assert!((mid.b0 - 0.5).abs() < 1e-12);
        assert!((mid.a2 - 2.5).abs() < 1e-12);
    }
}
