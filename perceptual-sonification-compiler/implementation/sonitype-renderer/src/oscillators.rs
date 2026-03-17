//! Oscillator implementations: wavetable, band-limited, FM, additive, and noise.
//!
//! All oscillators share a common phase-accumulator model and produce mono
//! output into an [`AudioBuf`](crate::AudioBuf).

use std::f64::consts::PI;

use crate::AudioBuf;

const TWO_PI: f64 = 2.0 * PI;
const WAVETABLE_SIZE: usize = 2048;
/// Number of octave-specific tables (C0 ≈ 16 Hz .. C10 ≈ 16 kHz → 11 octaves).
const NUM_OCTAVE_TABLES: usize = 11;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// PolyBLEP residual used to reduce aliasing in naive waveforms.
#[inline]
fn poly_blep(t: f64, dt: f64) -> f64 {
    if t < dt {
        let t = t / dt;
        2.0 * t - t * t - 1.0
    } else if t > 1.0 - dt {
        let t = (t - 1.0) / dt;
        t * t + 2.0 * t + 1.0
    } else {
        0.0
    }
}

/// Wrap phase into [0, 1).
#[inline]
fn wrap_phase(p: f64) -> f64 {
    let mut p = p % 1.0;
    if p < 0.0 {
        p += 1.0;
    }
    p
}

// ---------------------------------------------------------------------------
// WavetableOscillator
// ---------------------------------------------------------------------------

/// Band-limited wavetable oscillator with one table per octave and linear
/// interpolation.
#[derive(Debug, Clone)]
pub struct WavetableOscillator {
    /// tables\[octave\]\[sample\]
    tables: Vec<Vec<f32>>,
    phase: f64,
    frequency: f64,
    sample_rate: f64,
}

impl WavetableOscillator {
    /// Build a wavetable set from a harmonic generator function.
    ///
    /// `harmonic_fn(harmonic_number) -> amplitude` is called for each
    /// harmonic up to the Nyquist limit for the octave.
    pub fn new<F>(harmonic_fn: F, sample_rate: f64) -> Self
    where
        F: Fn(usize) -> f64,
    {
        let mut tables = Vec::with_capacity(NUM_OCTAVE_TABLES);
        for octave in 0..NUM_OCTAVE_TABLES {
            let base_freq = 16.0 * 2.0_f64.powi(octave as i32); // C of this octave
            let max_harmonic = ((sample_rate * 0.5) / base_freq).floor() as usize;
            let mut table = vec![0.0f32; WAVETABLE_SIZE];
            for h in 1..=max_harmonic {
                let amp = harmonic_fn(h);
                if amp.abs() < 1e-12 {
                    continue;
                }
                for i in 0..WAVETABLE_SIZE {
                    let phase = TWO_PI * h as f64 * i as f64 / WAVETABLE_SIZE as f64;
                    table[i] += (amp * phase.sin()) as f32;
                }
            }
            // Normalize peak to 1.0
            let peak = table.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
            if peak > 1e-12 {
                for s in table.iter_mut() {
                    *s /= peak;
                }
            }
            tables.push(table);
        }
        Self { tables, phase: 0.0, frequency: 440.0, sample_rate }
    }

    /// Build a sawtooth wavetable set.
    pub fn saw(sample_rate: f64) -> Self {
        Self::new(|h| 1.0 / h as f64, sample_rate)
    }

    /// Build a square wavetable set (odd harmonics only).
    pub fn square(sample_rate: f64) -> Self {
        Self::new(
            |h| if h % 2 == 1 { 1.0 / h as f64 } else { 0.0 },
            sample_rate,
        )
    }

    /// Build a triangle wavetable set.
    pub fn triangle(sample_rate: f64) -> Self {
        Self::new(
            |h| {
                if h % 2 == 1 {
                    let sign = if (h / 2) % 2 == 0 { 1.0 } else { -1.0 };
                    sign / (h as f64 * h as f64)
                } else {
                    0.0
                }
            },
            sample_rate,
        )
    }

    pub fn set_frequency(&mut self, freq: f64) {
        self.frequency = freq;
    }

    pub fn set_phase(&mut self, phase: f64) {
        self.phase = wrap_phase(phase);
    }

    pub fn reset(&mut self) {
        self.phase = 0.0;
    }

    /// Choose the appropriate octave table for a given frequency.
    fn octave_index(&self) -> usize {
        if self.frequency <= 0.0 {
            return 0;
        }
        let octave = (self.frequency / 16.0).log2().floor() as usize;
        octave.min(NUM_OCTAVE_TABLES - 1)
    }

    /// Read one sample via linear interpolation.
    #[inline]
    fn read_sample(&self) -> f32 {
        let idx = self.octave_index();
        let table = &self.tables[idx];
        let pos = self.phase * WAVETABLE_SIZE as f64;
        let i0 = pos as usize % WAVETABLE_SIZE;
        let i1 = (i0 + 1) % WAVETABLE_SIZE;
        let frac = pos.fract() as f32;
        table[i0] * (1.0 - frac) + table[i1] * frac
    }

    /// Render `frames` samples into `output` (mono, channel 0).
    pub fn process(&mut self, output: &mut AudioBuf) {
        let dt = self.frequency / self.sample_rate;
        let frames = output.frames();
        for i in 0..frames {
            let sample = self.read_sample();
            output.set(i, 0, sample);
            self.phase = wrap_phase(self.phase + dt);
        }
    }

    /// Render with per-sample frequency modulation.
    pub fn process_fm(&mut self, fm_buffer: &[f64], output: &mut AudioBuf) {
        let frames = output.frames();
        for i in 0..frames {
            let freq = self.frequency + fm_buffer.get(i).copied().unwrap_or(0.0);
            let dt = freq / self.sample_rate;
            let sample = self.read_sample();
            output.set(i, 0, sample);
            self.phase = wrap_phase(self.phase + dt);
        }
    }
}

// ---------------------------------------------------------------------------
// SineOscillator
// ---------------------------------------------------------------------------

/// Direct-computation sine oscillator: sin(2πft + φ).
#[derive(Debug, Clone)]
pub struct SineOscillator {
    pub phase: f64,
    pub frequency: f64,
    pub sample_rate: f64,
    pub phase_offset: f64,
}

impl SineOscillator {
    pub fn new(frequency: f64, sample_rate: f64) -> Self {
        Self { phase: 0.0, frequency, sample_rate, phase_offset: 0.0 }
    }

    pub fn set_frequency(&mut self, freq: f64) {
        self.frequency = freq;
    }

    pub fn set_phase_offset(&mut self, offset: f64) {
        self.phase_offset = offset;
    }

    pub fn reset(&mut self) {
        self.phase = 0.0;
    }

    #[inline]
    pub fn next_sample(&mut self) -> f32 {
        let out = (TWO_PI * self.phase + self.phase_offset).sin() as f32;
        self.phase += self.frequency / self.sample_rate;
        if self.phase >= 1.0 {
            self.phase -= 1.0;
        }
        out
    }

    pub fn process(&mut self, output: &mut AudioBuf) {
        let frames = output.frames();
        for i in 0..frames {
            output.set(i, 0, self.next_sample());
        }
    }
}

// ---------------------------------------------------------------------------
// SawOscillator (PolyBLEP)
// ---------------------------------------------------------------------------

/// Band-limited sawtooth oscillator using PolyBLEP anti-aliasing.
#[derive(Debug, Clone)]
pub struct SawOscillator {
    phase: f64,
    pub frequency: f64,
    pub sample_rate: f64,
}

impl SawOscillator {
    pub fn new(frequency: f64, sample_rate: f64) -> Self {
        Self { phase: 0.0, frequency, sample_rate }
    }

    pub fn set_frequency(&mut self, freq: f64) {
        self.frequency = freq;
    }

    pub fn reset(&mut self) {
        self.phase = 0.0;
    }

    #[inline]
    pub fn next_sample(&mut self) -> f32 {
        let dt = self.frequency / self.sample_rate;
        // Naive sawtooth: rises from -1 to 1
        let naive = 2.0 * self.phase - 1.0;
        let blep = poly_blep(self.phase, dt);
        self.phase = wrap_phase(self.phase + dt);
        (naive - blep) as f32
    }

    pub fn process(&mut self, output: &mut AudioBuf) {
        let frames = output.frames();
        for i in 0..frames {
            output.set(i, 0, self.next_sample());
        }
    }
}

// ---------------------------------------------------------------------------
// SquareOscillator (PolyBLEP)
// ---------------------------------------------------------------------------

/// Band-limited square wave oscillator using PolyBLEP.
#[derive(Debug, Clone)]
pub struct SquareOscillator {
    phase: f64,
    pub frequency: f64,
    pub sample_rate: f64,
    pub duty_cycle: f64,
}

impl SquareOscillator {
    pub fn new(frequency: f64, sample_rate: f64) -> Self {
        Self { phase: 0.0, frequency, sample_rate, duty_cycle: 0.5 }
    }

    pub fn set_frequency(&mut self, freq: f64) {
        self.frequency = freq;
    }

    pub fn set_duty_cycle(&mut self, duty: f64) {
        self.duty_cycle = duty.clamp(0.01, 0.99);
    }

    pub fn reset(&mut self) {
        self.phase = 0.0;
    }

    #[inline]
    pub fn next_sample(&mut self) -> f32 {
        let dt = self.frequency / self.sample_rate;
        let naive = if self.phase < self.duty_cycle { 1.0 } else { -1.0 };
        let mut blep = poly_blep(self.phase, dt);
        blep -= poly_blep(wrap_phase(self.phase - self.duty_cycle), dt);
        self.phase = wrap_phase(self.phase + dt);
        (naive + blep) as f32
    }

    pub fn process(&mut self, output: &mut AudioBuf) {
        let frames = output.frames();
        for i in 0..frames {
            output.set(i, 0, self.next_sample());
        }
    }
}

// ---------------------------------------------------------------------------
// TriangleOscillator (integrated PolyBLEP square → triangle)
// ---------------------------------------------------------------------------

/// Band-limited triangle wave derived by integrating a PolyBLEP square.
#[derive(Debug, Clone)]
pub struct TriangleOscillator {
    square: SquareOscillator,
    integrator: f64,
}

impl TriangleOscillator {
    pub fn new(frequency: f64, sample_rate: f64) -> Self {
        Self {
            square: SquareOscillator::new(frequency, sample_rate),
            integrator: 0.0,
        }
    }

    pub fn set_frequency(&mut self, freq: f64) {
        self.square.set_frequency(freq);
    }

    pub fn reset(&mut self) {
        self.square.reset();
        self.integrator = 0.0;
    }

    #[inline]
    pub fn next_sample(&mut self) -> f32 {
        let sq = self.square.next_sample() as f64;
        let dt = self.square.frequency / self.square.sample_rate;
        // Leaky integrator produces triangle shape; scale factor normalises amplitude.
        self.integrator += 4.0 * dt * sq;
        // DC-blocking leak
        self.integrator *= 0.9999;
        self.integrator as f32
    }

    pub fn process(&mut self, output: &mut AudioBuf) {
        let frames = output.frames();
        for i in 0..frames {
            output.set(i, 0, self.next_sample());
        }
    }
}

// ---------------------------------------------------------------------------
// PulseOscillator
// ---------------------------------------------------------------------------

/// Variable duty-cycle pulse oscillator (extends [`SquareOscillator`]).
#[derive(Debug, Clone)]
pub struct PulseOscillator {
    inner: SquareOscillator,
}

impl PulseOscillator {
    pub fn new(frequency: f64, sample_rate: f64, duty_cycle: f64) -> Self {
        let mut inner = SquareOscillator::new(frequency, sample_rate);
        inner.set_duty_cycle(duty_cycle);
        Self { inner }
    }

    pub fn set_frequency(&mut self, freq: f64) {
        self.inner.set_frequency(freq);
    }

    pub fn set_duty_cycle(&mut self, duty: f64) {
        self.inner.set_duty_cycle(duty);
    }

    pub fn reset(&mut self) {
        self.inner.reset();
    }

    #[inline]
    pub fn next_sample(&mut self) -> f32 {
        self.inner.next_sample()
    }

    pub fn process(&mut self, output: &mut AudioBuf) {
        self.inner.process(output);
    }
}

// ---------------------------------------------------------------------------
// NoiseOscillator
// ---------------------------------------------------------------------------

/// Noise colour.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoiseColor {
    White,
    Pink,
    Brown,
}

/// Noise generator: white, pink (Voss-McCartney), brown (integrated white).
#[derive(Debug, Clone)]
pub struct NoiseOscillator {
    color: NoiseColor,
    /// Simple LCG state (deterministic, no `rand` dependency in hot path).
    rng_state: u64,
    /// Voss–McCartney octave accumulators for pink noise.
    pink_rows: [f64; 16],
    pink_running_sum: f64,
    pink_index: u32,
    /// Brown noise: previous sample.
    brown_prev: f64,
}

impl NoiseOscillator {
    pub fn new(color: NoiseColor) -> Self {
        Self {
            color,
            rng_state: 0x12345678_9ABCDEF0,
            pink_rows: [0.0; 16],
            pink_running_sum: 0.0,
            pink_index: 0,
            brown_prev: 0.0,
        }
    }

    pub fn set_color(&mut self, color: NoiseColor) {
        self.color = color;
    }

    pub fn reset(&mut self) {
        self.pink_rows = [0.0; 16];
        self.pink_running_sum = 0.0;
        self.pink_index = 0;
        self.brown_prev = 0.0;
    }

    /// Cheap pseudo-random in (−1, 1).
    #[inline]
    fn rand(&mut self) -> f64 {
        // xorshift64
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 7;
        self.rng_state ^= self.rng_state << 17;
        // map to −1..1
        (self.rng_state as i64) as f64 / i64::MAX as f64
    }

    fn white_sample(&mut self) -> f64 {
        self.rand()
    }

    fn pink_sample(&mut self) -> f64 {
        // Voss-McCartney algorithm
        let num_rows = 12;
        let index = self.pink_index;
        self.pink_index = self.pink_index.wrapping_add(1);
        // Determine which rows to update
        let mut changed_bits = index ^ self.pink_index;
        for row in 0..num_rows {
            if changed_bits & 1 != 0 {
                self.pink_running_sum -= self.pink_rows[row];
                let new_val = self.rand();
                self.pink_running_sum += new_val;
                self.pink_rows[row] = new_val;
            }
            changed_bits >>= 1;
        }
        let white = self.rand();
        (self.pink_running_sum + white) / (num_rows as f64 + 1.0)
    }

    fn brown_sample(&mut self) -> f64 {
        let white = self.rand();
        self.brown_prev += white * 0.02;
        self.brown_prev = self.brown_prev.clamp(-1.0, 1.0);
        self.brown_prev
    }

    #[inline]
    pub fn next_sample(&mut self) -> f32 {
        let s = match self.color {
            NoiseColor::White => self.white_sample(),
            NoiseColor::Pink => self.pink_sample(),
            NoiseColor::Brown => self.brown_sample(),
        };
        s as f32
    }

    pub fn process(&mut self, output: &mut AudioBuf) {
        let frames = output.frames();
        for i in 0..frames {
            output.set(i, 0, self.next_sample());
        }
    }
}

// ---------------------------------------------------------------------------
// FMOscillator
// ---------------------------------------------------------------------------

/// Single FM operator (carrier or modulator).
#[derive(Debug, Clone)]
pub struct FMOperator {
    pub phase: f64,
    pub frequency: f64,
    pub amplitude: f64,
    pub feedback: f64,
    prev_output: f64,
}

impl FMOperator {
    pub fn new(frequency: f64, amplitude: f64) -> Self {
        Self { phase: 0.0, frequency, amplitude, feedback: 0.0, prev_output: 0.0 }
    }

    #[inline]
    pub fn tick(&mut self, modulation: f64, sample_rate: f64) -> f64 {
        let fb = self.feedback * self.prev_output;
        let out = (TWO_PI * self.phase + modulation + fb).sin() * self.amplitude;
        self.prev_output = out;
        self.phase += self.frequency / sample_rate;
        if self.phase >= 1.0 {
            self.phase -= self.phase.floor();
        }
        out
    }

    pub fn reset(&mut self) {
        self.phase = 0.0;
        self.prev_output = 0.0;
    }
}

/// Frequency modulation synthesiser supporting 2-op, 4-op, 6-op configurations.
#[derive(Debug, Clone)]
pub struct FMOscillator {
    pub operators: Vec<FMOperator>,
    /// Modulation index per modulator→carrier pair: (source_op, dest_op, index).
    pub connections: Vec<(usize, usize, f64)>,
    pub sample_rate: f64,
}

impl FMOscillator {
    /// Create a simple 2-op FM pair (modulator → carrier).
    pub fn two_op(carrier_freq: f64, mod_freq: f64, mod_index: f64, sample_rate: f64) -> Self {
        let carrier = FMOperator::new(carrier_freq, 1.0);
        let modulator = FMOperator::new(mod_freq, mod_index * mod_freq);
        Self {
            operators: vec![modulator, carrier],
            connections: vec![(0, 1, 1.0)],
            sample_rate,
        }
    }

    /// Create a 4-op FM setup.
    pub fn four_op(freqs: [f64; 4], amplitudes: [f64; 4], sample_rate: f64) -> Self {
        let ops: Vec<FMOperator> = freqs
            .iter()
            .zip(amplitudes.iter())
            .map(|(&f, &a)| FMOperator::new(f, a))
            .collect();
        // Default algorithm: 0→1→2→3 (serial chain)
        let connections = vec![(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)];
        Self { operators: ops, connections, sample_rate }
    }

    /// Create a 6-op FM setup.
    pub fn six_op(freqs: [f64; 6], amplitudes: [f64; 6], sample_rate: f64) -> Self {
        let ops: Vec<FMOperator> = freqs
            .iter()
            .zip(amplitudes.iter())
            .map(|(&f, &a)| FMOperator::new(f, a))
            .collect();
        let connections = vec![
            (0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0), (4, 5, 1.0),
        ];
        Self { operators: ops, connections, sample_rate }
    }

    pub fn reset(&mut self) {
        for op in &mut self.operators {
            op.reset();
        }
    }

    #[inline]
    pub fn next_sample(&mut self) -> f32 {
        let n = self.operators.len();
        let mut modulations = vec![0.0f64; n];

        // Accumulate modulation contributions
        for &(src, dst, idx) in &self.connections {
            if src < n && dst < n {
                modulations[dst] += self.operators[src].prev_output * idx;
            }
        }

        let mut output = 0.0f64;
        for (i, op) in self.operators.iter_mut().enumerate() {
            let out = op.tick(modulations[i], self.sample_rate);
            // Only the last operator is the carrier output by convention
            if i == n - 1 {
                output = out;
            }
        }
        output as f32
    }

    pub fn process(&mut self, output: &mut AudioBuf) {
        let frames = output.frames();
        for i in 0..frames {
            output.set(i, 0, self.next_sample());
        }
    }
}

// ---------------------------------------------------------------------------
// AdditiveOscillator
// ---------------------------------------------------------------------------

/// Specification of one partial in an additive oscillator.
#[derive(Debug, Clone, Copy)]
pub struct Partial {
    /// Harmonic number (1 = fundamental).
    pub harmonic: f64,
    /// Amplitude (linear, 0..1).
    pub amplitude: f64,
    /// Phase offset in radians.
    pub phase: f64,
}

impl Partial {
    pub fn new(harmonic: f64, amplitude: f64, phase: f64) -> Self {
        Self { harmonic, amplitude, phase }
    }
}

/// Additive oscillator: sum of individually-controllable partials.
#[derive(Debug, Clone)]
pub struct AdditiveOscillator {
    pub fundamental: f64,
    pub partials: Vec<Partial>,
    /// Per-partial phase accumulator.
    phases: Vec<f64>,
    pub sample_rate: f64,
}

impl AdditiveOscillator {
    pub fn new(fundamental: f64, partials: Vec<Partial>, sample_rate: f64) -> Self {
        let phases = vec![0.0; partials.len()];
        Self { fundamental, partials, phases, sample_rate }
    }

    /// Create a harmonic series with `n` partials, each with amplitude 1/h.
    pub fn harmonic_series(fundamental: f64, n: usize, sample_rate: f64) -> Self {
        let partials: Vec<Partial> = (1..=n)
            .map(|h| Partial::new(h as f64, 1.0 / h as f64, 0.0))
            .collect();
        Self::new(fundamental, partials, sample_rate)
    }

    pub fn set_fundamental(&mut self, freq: f64) {
        self.fundamental = freq;
    }

    pub fn set_partial_amplitude(&mut self, index: usize, amplitude: f64) {
        if let Some(p) = self.partials.get_mut(index) {
            p.amplitude = amplitude;
        }
    }

    pub fn add_partial(&mut self, partial: Partial) {
        self.partials.push(partial);
        self.phases.push(0.0);
    }

    pub fn remove_partial(&mut self, index: usize) {
        if index < self.partials.len() {
            self.partials.remove(index);
            self.phases.remove(index);
        }
    }

    pub fn reset(&mut self) {
        for p in self.phases.iter_mut() {
            *p = 0.0;
        }
    }

    #[inline]
    pub fn next_sample(&mut self) -> f32 {
        let mut sum = 0.0f64;
        for (i, partial) in self.partials.iter().enumerate() {
            let freq = self.fundamental * partial.harmonic;
            if freq > self.sample_rate * 0.5 {
                continue; // skip above Nyquist
            }
            sum += (TWO_PI * self.phases[i] + partial.phase).sin() * partial.amplitude;
            self.phases[i] += freq / self.sample_rate;
            if self.phases[i] >= 1.0 {
                self.phases[i] -= self.phases[i].floor();
            }
        }
        sum as f32
    }

    pub fn process(&mut self, output: &mut AudioBuf) {
        let frames = output.frames();
        for i in 0..frames {
            output.set(i, 0, self.next_sample());
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

    // -- SineOscillator -------------------------------------------------------

    #[test]
    fn sine_oscillator_zero_at_start() {
        let mut osc = SineOscillator::new(440.0, 44100.0);
        let sample = osc.next_sample();
        assert!(sample.abs() < 0.01, "sine should start near zero");
    }

    #[test]
    fn sine_oscillator_range() {
        let mut osc = SineOscillator::new(440.0, 44100.0);
        let mut buf = make_buf(44100);
        osc.process(&mut buf);
        for &s in &buf.data {
            assert!(s >= -1.001 && s <= 1.001);
        }
    }

    #[test]
    fn sine_oscillator_frequency_period() {
        let sr = 44100.0;
        let freq = 100.0;
        let mut osc = SineOscillator::new(freq, sr);
        let period_samples = (sr / freq) as usize;
        let mut buf = make_buf(period_samples * 4);
        osc.process(&mut buf);
        // After exactly one period the waveform should repeat
        let s0 = buf.get(0, 0);
        let sp = buf.get(period_samples, 0);
        assert!((s0 - sp).abs() < 0.02, "period mismatch");
    }

    // -- SawOscillator --------------------------------------------------------

    #[test]
    fn saw_oscillator_range() {
        let mut osc = SawOscillator::new(440.0, 44100.0);
        let mut buf = make_buf(4096);
        osc.process(&mut buf);
        for &s in &buf.data {
            assert!(s >= -1.5 && s <= 1.5, "saw out of range: {s}");
        }
    }

    // -- SquareOscillator -----------------------------------------------------

    #[test]
    fn square_oscillator_output() {
        let mut osc = SquareOscillator::new(100.0, 44100.0);
        let mut buf = make_buf(4096);
        osc.process(&mut buf);
        // Most samples should be close to ±1
        let near_one = buf.data.iter().filter(|&&s| (s.abs() - 1.0).abs() < 0.3).count();
        assert!(near_one > buf.data.len() / 2);
    }

    // -- TriangleOscillator ---------------------------------------------------

    #[test]
    fn triangle_oscillator_range() {
        let mut osc = TriangleOscillator::new(440.0, 44100.0);
        let mut buf = make_buf(8192);
        osc.process(&mut buf);
        for &s in &buf.data {
            assert!(s >= -2.0 && s <= 2.0, "triangle out of range: {s}");
        }
    }

    // -- PulseOscillator ------------------------------------------------------

    #[test]
    fn pulse_duty_cycle() {
        let mut osc = PulseOscillator::new(100.0, 44100.0, 0.25);
        let mut buf = make_buf(4410); // 10 periods at 100 Hz
        osc.process(&mut buf);
        let positive = buf.data.iter().filter(|&&s| s > 0.0).count();
        let ratio = positive as f64 / buf.data.len() as f64;
        assert!((ratio - 0.25).abs() < 0.1, "duty cycle off: {ratio}");
    }

    // -- NoiseOscillator ------------------------------------------------------

    #[test]
    fn white_noise_mean_near_zero() {
        let mut osc = NoiseOscillator::new(NoiseColor::White);
        let mut buf = make_buf(100_000);
        osc.process(&mut buf);
        let mean: f64 = buf.data.iter().map(|&s| s as f64).sum::<f64>() / buf.data.len() as f64;
        assert!(mean.abs() < 0.05, "white noise mean too far from 0: {mean}");
    }

    #[test]
    fn pink_noise_produces_output() {
        let mut osc = NoiseOscillator::new(NoiseColor::Pink);
        let mut buf = make_buf(1024);
        osc.process(&mut buf);
        let energy: f64 = buf.data.iter().map(|&s| (s as f64) * (s as f64)).sum();
        assert!(energy > 0.0);
    }

    #[test]
    fn brown_noise_range() {
        let mut osc = NoiseOscillator::new(NoiseColor::Brown);
        let mut buf = make_buf(10_000);
        osc.process(&mut buf);
        for &s in &buf.data {
            assert!(s >= -1.001 && s <= 1.001, "brown out of range: {s}");
        }
    }

    // -- WavetableOscillator --------------------------------------------------

    #[test]
    fn wavetable_saw_range() {
        let mut osc = WavetableOscillator::saw(44100.0);
        osc.set_frequency(440.0);
        let mut buf = make_buf(4096);
        osc.process(&mut buf);
        for &s in &buf.data {
            assert!(s >= -1.5 && s <= 1.5, "wavetable saw out of range: {s}");
        }
    }

    #[test]
    fn wavetable_square_range() {
        let mut osc = WavetableOscillator::square(44100.0);
        osc.set_frequency(440.0);
        let mut buf = make_buf(4096);
        osc.process(&mut buf);
        for &s in &buf.data {
            assert!(s >= -1.5 && s <= 1.5);
        }
    }

    // -- FMOscillator ---------------------------------------------------------

    #[test]
    fn fm_two_op_produces_output() {
        let mut fm = FMOscillator::two_op(440.0, 440.0, 2.0, 44100.0);
        let mut buf = make_buf(4096);
        fm.process(&mut buf);
        let energy: f64 = buf.data.iter().map(|&s| (s as f64).powi(2)).sum();
        assert!(energy > 0.0);
    }

    #[test]
    fn fm_four_op_produces_output() {
        let mut fm = FMOscillator::four_op(
            [440.0, 880.0, 1320.0, 1760.0],
            [1.0, 0.5, 0.3, 0.2],
            44100.0,
        );
        let mut buf = make_buf(2048);
        fm.process(&mut buf);
        let energy: f64 = buf.data.iter().map(|&s| (s as f64).powi(2)).sum();
        assert!(energy > 0.0);
    }

    // -- AdditiveOscillator ---------------------------------------------------

    #[test]
    fn additive_harmonic_series() {
        let mut osc = AdditiveOscillator::harmonic_series(440.0, 8, 44100.0);
        let mut buf = make_buf(4096);
        osc.process(&mut buf);
        let peak = buf.data.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        assert!(peak > 0.1, "additive should produce output");
    }

    #[test]
    fn additive_modify_partial() {
        let mut osc = AdditiveOscillator::harmonic_series(440.0, 4, 44100.0);
        osc.set_partial_amplitude(0, 0.0);
        let mut buf = make_buf(2048);
        osc.process(&mut buf);
        // Still produces output from remaining partials
        let energy: f64 = buf.data.iter().map(|&s| (s as f64).powi(2)).sum();
        assert!(energy > 0.0);
    }

    #[test]
    fn additive_nyquist_clamping() {
        // Fundamental so high that most partials exceed Nyquist
        let mut osc = AdditiveOscillator::harmonic_series(20000.0, 8, 44100.0);
        let mut buf = make_buf(1024);
        osc.process(&mut buf);
        // Should still run without panic
        let peak = buf.data.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        assert!(peak < 2.0);
    }
}
