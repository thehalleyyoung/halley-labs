//! Audio effects: delay, chorus, reverb, compressor, limiter, distortion.

use std::f64::consts::PI;
use crate::AudioBuf;

const TWO_PI: f64 = 2.0 * PI;

// ---------------------------------------------------------------------------
// Delay
// ---------------------------------------------------------------------------

/// Simple delay line with circular buffer, fractional delay via linear
/// interpolation, and feedback with damping.
#[derive(Debug, Clone)]
pub struct Delay {
    buffer: Vec<f64>,
    write_pos: usize,
    /// Delay in fractional samples.
    pub delay_samples: f64,
    /// Feedback gain (0..1).
    pub feedback: f64,
    /// Damping one-pole coefficient (0 = no damping, 1 = full damping).
    pub damping: f64,
    damp_state: f64,
    pub mix: f64,
}

impl Delay {
    /// Create a delay with `max_samples` capacity.
    pub fn new(max_samples: usize, delay_samples: f64) -> Self {
        Self {
            buffer: vec![0.0; max_samples.max(1)],
            write_pos: 0,
            delay_samples,
            feedback: 0.0,
            damping: 0.0,
            damp_state: 0.0,
            mix: 0.5,
        }
    }

    /// Create a delay specified in seconds.
    pub fn from_time(delay_sec: f64, sample_rate: f64) -> Self {
        let smp = (delay_sec * sample_rate) as usize;
        Self::new(smp + 1024, delay_sec * sample_rate)
    }

    pub fn set_delay(&mut self, samples: f64) {
        self.delay_samples = samples.clamp(0.0, (self.buffer.len() - 1) as f64);
    }

    pub fn set_feedback(&mut self, fb: f64) {
        self.feedback = fb.clamp(0.0, 0.999);
    }

    pub fn set_damping(&mut self, d: f64) {
        self.damping = d.clamp(0.0, 1.0);
    }

    pub fn reset(&mut self) {
        for s in self.buffer.iter_mut() { *s = 0.0; }
        self.damp_state = 0.0;
        self.write_pos = 0;
    }

    /// Read from the delay line with linear interpolation.
    #[inline]
    fn read(&self) -> f64 {
        let len = self.buffer.len();
        let rd = self.write_pos as f64 - self.delay_samples;
        let rd = if rd < 0.0 { rd + len as f64 } else { rd };
        let i0 = rd as usize % len;
        let i1 = (i0 + 1) % len;
        let frac = rd.fract();
        self.buffer[i0] * (1.0 - frac) + self.buffer[i1] * frac
    }

    #[inline]
    pub fn tick(&mut self, input: f64) -> f64 {
        let delayed = self.read();
        // Damping filter on feedback path
        let damped = delayed * (1.0 - self.damping) + self.damp_state * self.damping;
        self.damp_state = damped;

        self.buffer[self.write_pos] = input + damped * self.feedback;
        self.write_pos = (self.write_pos + 1) % self.buffer.len();

        input * (1.0 - self.mix) + delayed * self.mix
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
// Chorus
// ---------------------------------------------------------------------------

/// Multi-voice chorus effect using modulated delay taps.
#[derive(Debug, Clone)]
pub struct Chorus {
    buffer: Vec<f64>,
    write_pos: usize,
    /// LFO phases per voice (in samples).
    lfo_phases: Vec<f64>,
    pub num_voices: usize,
    /// Base delay in samples.
    pub base_delay: f64,
    /// LFO depth in samples.
    pub depth: f64,
    /// LFO rate in Hz.
    pub rate: f64,
    pub mix: f64,
    pub sample_rate: f64,
}

impl Chorus {
    pub fn new(num_voices: usize, base_delay_ms: f64, sample_rate: f64) -> Self {
        let max_samples = ((base_delay_ms + 20.0) * sample_rate / 1000.0) as usize + 1;
        let mut lfo_phases = Vec::with_capacity(num_voices);
        for v in 0..num_voices {
            lfo_phases.push(v as f64 / num_voices as f64);
        }
        Self {
            buffer: vec![0.0; max_samples],
            write_pos: 0,
            lfo_phases,
            num_voices,
            base_delay: base_delay_ms * sample_rate / 1000.0,
            depth: 3.0 * sample_rate / 1000.0, // 3 ms default depth
            rate: 0.5,
            mix: 0.5,
            sample_rate,
        }
    }

    pub fn set_rate(&mut self, hz: f64) {
        self.rate = hz;
    }

    pub fn set_depth_ms(&mut self, ms: f64) {
        self.depth = ms * self.sample_rate / 1000.0;
    }

    pub fn reset(&mut self) {
        for s in self.buffer.iter_mut() { *s = 0.0; }
        self.write_pos = 0;
        for (v, p) in self.lfo_phases.iter_mut().enumerate() {
            *p = v as f64 / self.num_voices as f64;
        }
    }

    fn read_interp(&self, delay: f64) -> f64 {
        let len = self.buffer.len() as f64;
        let rd = self.write_pos as f64 - delay;
        let rd = if rd < 0.0 { rd + len } else { rd };
        let i0 = rd as usize % self.buffer.len();
        let i1 = (i0 + 1) % self.buffer.len();
        let frac = rd.fract();
        self.buffer[i0] * (1.0 - frac) + self.buffer[i1] * frac
    }

    #[inline]
    pub fn tick(&mut self, input: f64) -> f64 {
        self.buffer[self.write_pos] = input;
        self.write_pos = (self.write_pos + 1) % self.buffer.len();

        let dt = self.rate / self.sample_rate;
        let buf_len = self.buffer.len();
        let write_pos = self.write_pos;
        let mut sum = 0.0;
        for phase in &mut self.lfo_phases {
            let mod_offset = (TWO_PI * *phase).sin() * self.depth;
            let delay = self.base_delay + mod_offset;
            let delay = delay.clamp(1.0, (buf_len - 1) as f64);
            // Inline read_interp to avoid borrow conflict.
            let rd = write_pos as f64 - delay;
            let rd = if rd < 0.0 { rd + buf_len as f64 } else { rd };
            let i0 = rd as usize % buf_len;
            let i1 = (i0 + 1) % buf_len;
            let frac = rd.fract();
            sum += self.buffer[i0] * (1.0 - frac) + self.buffer[i1] * frac;
            *phase += dt;
            if *phase >= 1.0 { *phase -= 1.0; }
        }
        let wet = sum / self.num_voices as f64;
        input * (1.0 - self.mix) + wet * self.mix
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
// Reverb (Schroeder)
// ---------------------------------------------------------------------------

/// Internal comb filter for the Schroeder reverb.
#[derive(Debug, Clone)]
struct ReverbComb {
    buffer: Vec<f64>,
    write_pos: usize,
    feedback: f64,
    damp: f64,
    damp_state: f64,
}

impl ReverbComb {
    fn new(size: usize, feedback: f64, damp: f64) -> Self {
        Self { buffer: vec![0.0; size], write_pos: 0, feedback, damp, damp_state: 0.0 }
    }

    fn reset(&mut self) {
        for s in self.buffer.iter_mut() { *s = 0.0; }
        self.damp_state = 0.0;
        self.write_pos = 0;
    }

    #[inline]
    fn tick(&mut self, input: f64) -> f64 {
        let output = self.buffer[self.write_pos];
        let damped = output * (1.0 - self.damp) + self.damp_state * self.damp;
        self.damp_state = damped;
        self.buffer[self.write_pos] = input + damped * self.feedback;
        self.write_pos = (self.write_pos + 1) % self.buffer.len();
        output
    }
}

/// Internal allpass for the Schroeder reverb.
#[derive(Debug, Clone)]
struct ReverbAllpass {
    buffer: Vec<f64>,
    write_pos: usize,
    gain: f64,
}

impl ReverbAllpass {
    fn new(size: usize, gain: f64) -> Self {
        Self { buffer: vec![0.0; size], write_pos: 0, gain }
    }

    fn reset(&mut self) {
        for s in self.buffer.iter_mut() { *s = 0.0; }
        self.write_pos = 0;
    }

    #[inline]
    fn tick(&mut self, input: f64) -> f64 {
        let delayed = self.buffer[self.write_pos];
        let temp = input + delayed * self.gain;
        self.buffer[self.write_pos] = temp;
        self.write_pos = (self.write_pos + 1) % self.buffer.len();
        delayed - temp * self.gain
    }
}

/// Algorithmic reverb based on the Schroeder model (4 comb + 2 allpass).
#[derive(Debug, Clone)]
pub struct Reverb {
    combs: Vec<ReverbComb>,
    allpasses: Vec<ReverbAllpass>,
    /// Pre-delay buffer.
    pre_delay_buf: Vec<f64>,
    pre_delay_pos: usize,
    pub pre_delay_samples: usize,
    pub decay_time: f64,
    pub damping: f64,
    pub mix: f64,
    pub sample_rate: f64,
    /// Stereo spread (additional offset for R channel combs).
    pub stereo_spread: usize,
    combs_r: Vec<ReverbComb>,
    allpasses_r: Vec<ReverbAllpass>,
}

impl Reverb {
    /// Standard comb delays (tuned for 44100 Hz; scaled for other rates).
    const COMB_TUNINGS: [usize; 4] = [1116, 1188, 1277, 1356];
    const AP_TUNINGS: [usize; 2] = [556, 441];
    const STEREO_SPREAD: usize = 23;

    pub fn new(decay_time: f64, damping: f64, sample_rate: f64) -> Self {
        let scale = sample_rate / 44100.0;
        let feedback = 0.84_f64.powf(1.0 / (decay_time.max(0.1) * 10.0));

        let combs: Vec<_> = Self::COMB_TUNINGS.iter()
            .map(|&t| ReverbComb::new((t as f64 * scale) as usize, feedback, damping))
            .collect();
        let allpasses: Vec<_> = Self::AP_TUNINGS.iter()
            .map(|&t| ReverbAllpass::new((t as f64 * scale) as usize, 0.5))
            .collect();
        let combs_r: Vec<_> = Self::COMB_TUNINGS.iter()
            .map(|&t| {
                let size = (t as f64 * scale) as usize + Self::STEREO_SPREAD;
                ReverbComb::new(size, feedback, damping)
            })
            .collect();
        let allpasses_r: Vec<_> = Self::AP_TUNINGS.iter()
            .map(|&t| {
                let size = (t as f64 * scale) as usize + Self::STEREO_SPREAD;
                ReverbAllpass::new(size, 0.5)
            })
            .collect();
        let pre_delay_max = (0.1 * sample_rate) as usize; // 100 ms max
        Self {
            combs,
            allpasses,
            pre_delay_buf: vec![0.0; pre_delay_max.max(1)],
            pre_delay_pos: 0,
            pre_delay_samples: 0,
            decay_time,
            damping,
            mix: 0.3,
            sample_rate,
            stereo_spread: Self::STEREO_SPREAD,
            combs_r,
            allpasses_r,
        }
    }

    pub fn set_pre_delay_ms(&mut self, ms: f64) {
        self.pre_delay_samples =
            ((ms / 1000.0 * self.sample_rate) as usize).min(self.pre_delay_buf.len() - 1);
    }

    pub fn set_decay(&mut self, time: f64) {
        self.decay_time = time;
        let fb = 0.84_f64.powf(1.0 / (time.max(0.1) * 10.0));
        for c in &mut self.combs { c.feedback = fb; }
        for c in &mut self.combs_r { c.feedback = fb; }
    }

    pub fn set_damping(&mut self, d: f64) {
        self.damping = d;
        for c in &mut self.combs { c.damp = d; }
        for c in &mut self.combs_r { c.damp = d; }
    }

    pub fn reset(&mut self) {
        for c in &mut self.combs { c.reset(); }
        for a in &mut self.allpasses { a.reset(); }
        for c in &mut self.combs_r { c.reset(); }
        for a in &mut self.allpasses_r { a.reset(); }
        for s in self.pre_delay_buf.iter_mut() { *s = 0.0; }
        self.pre_delay_pos = 0;
    }

    /// Process one sample, returning (left, right).
    #[inline]
    pub fn tick(&mut self, input: f64) -> (f64, f64) {
        // Pre-delay
        let pd_read = (self.pre_delay_pos + self.pre_delay_buf.len() - self.pre_delay_samples)
            % self.pre_delay_buf.len();
        let pre_delayed = self.pre_delay_buf[pd_read];
        self.pre_delay_buf[self.pre_delay_pos] = input;
        self.pre_delay_pos = (self.pre_delay_pos + 1) % self.pre_delay_buf.len();

        // Parallel combs (L)
        let mut out_l = 0.0;
        for c in &mut self.combs {
            out_l += c.tick(pre_delayed);
        }
        // Series allpasses (L)
        for a in &mut self.allpasses {
            out_l = a.tick(out_l);
        }

        // Parallel combs (R) – offset sizes for stereo decorrelation
        let mut out_r = 0.0;
        for c in &mut self.combs_r {
            out_r += c.tick(pre_delayed);
        }
        for a in &mut self.allpasses_r {
            out_r = a.tick(out_r);
        }

        let dry = 1.0 - self.mix;
        let wet = self.mix;
        (input * dry + out_l * wet, input * dry + out_r * wet)
    }

    /// Process a stereo buffer in-place.
    pub fn process(&mut self, buf: &mut AudioBuf) {
        let frames = buf.frames();
        for f in 0..frames {
            let mono = if buf.channels >= 2 {
                (buf.get(f, 0) as f64 + buf.get(f, 1) as f64) * 0.5
            } else {
                buf.get(f, 0) as f64
            };
            let (l, r) = self.tick(mono);
            buf.set(f, 0, l as f32);
            if buf.channels >= 2 {
                buf.set(f, 1, r as f32);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Compressor
// ---------------------------------------------------------------------------

/// Detection mode for the compressor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressorDetection {
    Peak,
    Rms,
}

/// Dynamics compressor with soft-knee, makeup gain, and envelope follower.
#[derive(Debug, Clone)]
pub struct Compressor {
    /// Threshold in dB.
    pub threshold_db: f64,
    /// Ratio (e.g. 4.0 means 4:1).
    pub ratio: f64,
    /// Attack time in seconds.
    pub attack_time: f64,
    /// Release time in seconds.
    pub release_time: f64,
    /// Knee width in dB (0 = hard knee).
    pub knee_db: f64,
    /// Makeup gain in dB.
    pub makeup_db: f64,
    pub detection: CompressorDetection,
    pub sample_rate: f64,
    // internal
    attack_coeff: f64,
    release_coeff: f64,
    env: f64,
    /// RMS running average state.
    rms_sq: f64,
}

impl Compressor {
    pub fn new(threshold_db: f64, ratio: f64, sample_rate: f64) -> Self {
        let attack_time = 0.01;
        let release_time = 0.1;
        Self {
            threshold_db,
            ratio,
            attack_time,
            release_time,
            knee_db: 0.0,
            makeup_db: 0.0,
            detection: CompressorDetection::Peak,
            sample_rate,
            attack_coeff: Self::time_to_coeff(attack_time, sample_rate),
            release_coeff: Self::time_to_coeff(release_time, sample_rate),
            env: 0.0,
            rms_sq: 0.0,
        }
    }

    fn time_to_coeff(t: f64, sr: f64) -> f64 {
        if t <= 0.0 || sr <= 0.0 { return 1.0; }
        1.0 - (-1.0 / (t * sr)).exp()
    }

    pub fn set_attack(&mut self, time: f64) {
        self.attack_time = time;
        self.attack_coeff = Self::time_to_coeff(time, self.sample_rate);
    }

    pub fn set_release(&mut self, time: f64) {
        self.release_time = time;
        self.release_coeff = Self::time_to_coeff(time, self.sample_rate);
    }

    pub fn set_knee(&mut self, db: f64) {
        self.knee_db = db.max(0.0);
    }

    pub fn set_makeup(&mut self, db: f64) {
        self.makeup_db = db;
    }

    pub fn reset(&mut self) {
        self.env = 0.0;
        self.rms_sq = 0.0;
    }

    /// Compute gain reduction in dB for a given input level in dB.
    fn gain_computer(&self, input_db: f64) -> f64 {
        let t = self.threshold_db;
        let r = self.ratio;
        let k = self.knee_db;

        if k <= 0.0 {
            // Hard knee
            if input_db <= t {
                input_db
            } else {
                t + (input_db - t) / r
            }
        } else {
            // Soft knee
            let half_k = k * 0.5;
            if input_db < t - half_k {
                input_db
            } else if input_db > t + half_k {
                t + (input_db - t) / r
            } else {
                let x = input_db - t + half_k;
                input_db + (1.0 / r - 1.0) * x * x / (2.0 * k)
            }
        }
    }

    #[inline]
    pub fn tick(&mut self, input: f64) -> f64 {
        let level = match self.detection {
            CompressorDetection::Peak => input.abs(),
            CompressorDetection::Rms => {
                self.rms_sq = self.rms_sq * 0.999 + input * input * 0.001;
                self.rms_sq.sqrt()
            }
        };

        // Envelope follower
        let coeff = if level > self.env { self.attack_coeff } else { self.release_coeff };
        self.env += coeff * (level - self.env);

        // Gain computation
        let env_db = if self.env > 1e-12 { 20.0 * self.env.log10() } else { -120.0 };
        let out_db = self.gain_computer(env_db);
        let gain_db = out_db - env_db + self.makeup_db;
        let gain = 10.0_f64.powf(gain_db / 20.0);

        input * gain
    }

    pub fn process(&mut self, buf: &mut AudioBuf) {
        let frames = buf.frames();
        let ch = buf.channels;
        for f in 0..frames {
            for c in 0..ch {
                let inp = buf.get(f, c) as f64;
                buf.set(f, c, self.tick(inp) as f32);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Limiter
// ---------------------------------------------------------------------------

/// Brick-wall limiter with lookahead and adaptive release.
#[derive(Debug, Clone)]
pub struct Limiter {
    /// Ceiling in linear amplitude.
    pub ceiling: f64,
    /// Lookahead in samples.
    pub lookahead: usize,
    /// Release time in seconds.
    pub release_time: f64,
    pub sample_rate: f64,
    // internal
    delay_buffer: Vec<f64>,
    delay_pos: usize,
    gain: f64,
    release_coeff: f64,
}

impl Limiter {
    pub fn new(ceiling_db: f64, lookahead_ms: f64, sample_rate: f64) -> Self {
        let ceiling = 10.0_f64.powf(ceiling_db / 20.0);
        let lookahead = (lookahead_ms / 1000.0 * sample_rate) as usize;
        let release_time = 0.05;
        let release_coeff = if release_time > 0.0 && sample_rate > 0.0 {
            1.0 - (-1.0 / (release_time * sample_rate)).exp()
        } else {
            1.0
        };
        Self {
            ceiling,
            lookahead,
            release_time,
            sample_rate,
            delay_buffer: vec![0.0; lookahead.max(1)],
            delay_pos: 0,
            gain: 1.0,
            release_coeff,
        }
    }

    pub fn set_ceiling_db(&mut self, db: f64) {
        self.ceiling = 10.0_f64.powf(db / 20.0);
    }

    pub fn reset(&mut self) {
        for s in self.delay_buffer.iter_mut() { *s = 0.0; }
        self.delay_pos = 0;
        self.gain = 1.0;
    }

    #[inline]
    pub fn tick(&mut self, input: f64) -> f64 {
        // Write into lookahead buffer and read the delayed sample
        let delayed = if self.lookahead > 0 {
            let out = self.delay_buffer[self.delay_pos];
            self.delay_buffer[self.delay_pos] = input;
            self.delay_pos = (self.delay_pos + 1) % self.delay_buffer.len();
            out
        } else {
            input
        };

        let abs = input.abs();
        let target_gain = if abs > self.ceiling {
            self.ceiling / abs
        } else {
            1.0
        };

        // Instant attack, smooth release
        if target_gain < self.gain {
            self.gain = target_gain;
        } else {
            self.gain += self.release_coeff * (target_gain - self.gain);
        }

        delayed * self.gain
    }

    pub fn process(&mut self, buf: &mut AudioBuf) {
        let frames = buf.frames();
        let ch = buf.channels;
        for f in 0..frames {
            for c in 0..ch {
                let inp = buf.get(f, c) as f64;
                buf.set(f, c, self.tick(inp) as f32);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Distortion
// ---------------------------------------------------------------------------

/// Distortion algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistortionType {
    SoftClip,
    HardClip,
    Tanh,
    SineFold,
}

/// Waveshaping distortion effect.
#[derive(Debug, Clone)]
pub struct Distortion {
    pub drive: f64,
    pub mode: DistortionType,
    pub mix: f64,
}

impl Distortion {
    pub fn new(drive: f64, mode: DistortionType) -> Self {
        Self { drive: drive.max(0.1), mode, mix: 1.0 }
    }

    pub fn set_drive(&mut self, drive: f64) {
        self.drive = drive.max(0.1);
    }

    pub fn set_mode(&mut self, mode: DistortionType) {
        self.mode = mode;
    }

    #[inline]
    pub fn tick(&self, input: f64) -> f64 {
        let driven = input * self.drive;
        let shaped = match self.mode {
            DistortionType::SoftClip => {
                // Cubic soft clip
                if driven.abs() < 2.0 / 3.0 {
                    driven
                } else if driven > 0.0 {
                    1.0 - (2.0 - 3.0 * driven).powi(2) / 3.0
                } else {
                    -1.0 + (2.0 + 3.0 * driven).powi(2) / 3.0
                }
            }
            DistortionType::HardClip => driven.clamp(-1.0, 1.0),
            DistortionType::Tanh => driven.tanh(),
            DistortionType::SineFold => (driven * PI).sin(),
        };
        input * (1.0 - self.mix) + shaped * self.mix
    }

    pub fn process(&mut self, buf: &mut AudioBuf) {
        let frames = buf.frames();
        let ch = buf.channels;
        for f in 0..frames {
            for c in 0..ch {
                let inp = buf.get(f, c) as f64;
                buf.set(f, c, self.tick(inp) as f32);
            }
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

    // -- Delay ----------------------------------------------------------------

    #[test]
    fn delay_passes_through_dry() {
        let mut dl = Delay::new(1024, 100.0);
        dl.mix = 0.0; // dry only
        let out = dl.tick(1.0);
        assert!((out - 1.0).abs() < 1e-6);
    }

    #[test]
    fn delay_echoes_at_correct_time() {
        let mut dl = Delay::new(1024, 10.0);
        dl.mix = 1.0; // wet only
        dl.feedback = 0.0;
        dl.tick(1.0); // write impulse at t=0
        for _ in 1..10 {
            let out = dl.tick(0.0);
            assert!(out.abs() < 1e-6);
        }
        let out = dl.tick(0.0); // t=10
        assert!((out - 1.0).abs() < 1e-3, "expected echo, got {out}");
    }

    #[test]
    fn delay_feedback_produces_repeats() {
        let mut dl = Delay::new(1024, 5.0);
        dl.mix = 1.0;
        dl.feedback = 0.5;
        dl.tick(1.0);
        for _ in 1..5 { dl.tick(0.0); }
        let first_echo = dl.tick(0.0);
        assert!(first_echo.abs() > 0.5);
        for _ in 0..4 { dl.tick(0.0); }
        let second_echo = dl.tick(0.0);
        // Second echo should be weaker
        assert!(second_echo.abs() < first_echo.abs());
    }

    // -- Chorus ---------------------------------------------------------------

    #[test]
    fn chorus_produces_output() {
        let mut ch = Chorus::new(3, 10.0, 44100.0);
        let mut buf = make_buf(2048);
        for f in 0..buf.frames() {
            buf.set(f, 0, (2.0 * PI * 440.0 * f as f64 / 44100.0).sin() as f32);
        }
        ch.process(&mut buf);
        let energy: f64 = buf.data.iter().map(|&s| (s as f64).powi(2)).sum();
        assert!(energy > 0.0);
    }

    #[test]
    fn chorus_reset_clears() {
        let mut ch = Chorus::new(2, 10.0, 44100.0);
        ch.tick(1.0);
        ch.reset();
        let out = ch.tick(0.0);
        assert!(out.abs() < 1e-3);
    }

    // -- Reverb ---------------------------------------------------------------

    #[test]
    fn reverb_produces_tail() {
        let mut rev = Reverb::new(1.0, 0.5, 44100.0);
        rev.mix = 1.0;
        // Feed impulse
        let (l, _r) = rev.tick(1.0);
        let _ = l;
        // After many samples, tail should still have energy
        let mut tail_energy = 0.0;
        for _ in 0..4410 {
            let (l, _) = rev.tick(0.0);
            tail_energy += l * l;
        }
        assert!(tail_energy > 0.001, "reverb tail too quiet");
    }

    #[test]
    fn reverb_stereo_decorrelation() {
        let mut rev = Reverb::new(1.0, 0.3, 44100.0);
        rev.mix = 1.0;
        rev.tick(1.0);
        let mut diff_count = 0;
        for _ in 0..1000 {
            let (l, r) = rev.tick(0.0);
            if (l - r).abs() > 1e-6 { diff_count += 1; }
        }
        assert!(diff_count > 100, "L and R should differ for stereo spread");
    }

    // -- Compressor -----------------------------------------------------------

    #[test]
    fn compressor_reduces_loud_signal() {
        let mut comp = Compressor::new(-20.0, 4.0, 44100.0);
        comp.set_attack(0.001);
        comp.set_release(0.01);
        // Feed a loud signal
        let mut peaks = vec![];
        for _ in 0..4410 {
            let out = comp.tick(1.0);
            peaks.push(out.abs());
        }
        let max_out = peaks.iter().cloned().fold(0.0f64, f64::max);
        assert!(max_out < 1.0, "compressor should reduce level, got {max_out}");
    }

    #[test]
    fn compressor_soft_knee() {
        let mut comp = Compressor::new(-20.0, 4.0, 44100.0);
        comp.set_knee(6.0);
        // Around threshold, gain should transition smoothly
        let a = comp.gain_computer(-23.0);
        let b = comp.gain_computer(-20.0);
        let c = comp.gain_computer(-17.0);
        assert!(a < b && b < c, "soft knee should be monotonic");
    }

    // -- Limiter --------------------------------------------------------------

    #[test]
    fn limiter_clamps_output() {
        let mut lim = Limiter::new(0.0, 0.0, 44100.0); // 0 dBFS ceiling, no lookahead
        for _ in 0..1000 {
            let out = lim.tick(2.0);
            assert!(out.abs() <= 1.01, "limiter output too loud: {out}");
        }
    }

    #[test]
    fn limiter_passes_quiet_signal() {
        let mut lim = Limiter::new(0.0, 0.0, 44100.0);
        let out = lim.tick(0.5);
        assert!((out - 0.5).abs() < 0.01);
    }

    // -- Distortion -----------------------------------------------------------

    #[test]
    fn distortion_hard_clip() {
        let dist = Distortion::new(10.0, DistortionType::HardClip);
        let out = dist.tick(0.5); // 0.5 * 10 = 5.0 → clip to 1.0
        assert!((out - 1.0).abs() < 1e-6);
    }

    #[test]
    fn distortion_tanh_saturates() {
        let dist = Distortion::new(5.0, DistortionType::Tanh);
        let out = dist.tick(1.0);
        assert!(out.abs() <= 1.0);
    }

    #[test]
    fn distortion_soft_clip_range() {
        let dist = Distortion::new(3.0, DistortionType::SoftClip);
        for i in 0..100 {
            let inp = (i as f64 - 50.0) / 50.0;
            let out = dist.tick(inp);
            assert!(out.abs() <= 1.5, "soft clip out of range: {out}");
        }
    }

    #[test]
    fn distortion_sine_fold() {
        let dist = Distortion::new(1.0, DistortionType::SineFold);
        let out = dist.tick(0.5);
        // sin(0.5 * π) ≈ 1.0
        assert!((out - 1.0).abs() < 0.01);
    }
}
