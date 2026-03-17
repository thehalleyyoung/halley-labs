//! Envelope generators: ADSR, multi-segment, LFO, and amplitude follower.
//!
//! Envelopes produce per-sample control values in the range \[0, 1\] (or
//! arbitrary for followers) and are typically multiplied into audio signals
//! for amplitude shaping.

use std::f64::consts::PI;
use crate::AudioBuf;

const TWO_PI: f64 = 2.0 * PI;

// ---------------------------------------------------------------------------
// Curve type
// ---------------------------------------------------------------------------

/// Shape of an envelope segment's transition.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CurveType {
    Linear,
    Exponential,
    Logarithmic,
}

// ---------------------------------------------------------------------------
// AdsrEnvelope
// ---------------------------------------------------------------------------

/// ADSR envelope state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdsrState {
    Idle,
    Attack,
    Decay,
    Sustain,
    Release,
}

/// Classic attack-decay-sustain-release envelope generator.
#[derive(Debug, Clone)]
pub struct AdsrEnvelope {
    pub attack_time: f64,
    pub decay_time: f64,
    pub sustain_level: f64,
    pub release_time: f64,
    pub attack_curve: CurveType,
    pub decay_curve: CurveType,
    pub release_curve: CurveType,
    pub sample_rate: f64,
    state: AdsrState,
    level: f64,
    /// Fractional position within the current stage (0..1).
    stage_pos: f64,
    stage_increment: f64,
    /// Level at the moment release was triggered (for correct release curve).
    release_start_level: f64,
}

impl AdsrEnvelope {
    pub fn new(
        attack: f64, decay: f64, sustain: f64, release: f64, sample_rate: f64,
    ) -> Self {
        Self {
            attack_time: attack,
            decay_time: decay,
            sustain_level: sustain.clamp(0.0, 1.0),
            release_time: release,
            attack_curve: CurveType::Linear,
            decay_curve: CurveType::Exponential,
            release_curve: CurveType::Exponential,
            sample_rate,
            state: AdsrState::Idle,
            level: 0.0,
            stage_pos: 0.0,
            stage_increment: 0.0,
            release_start_level: 0.0,
        }
    }

    pub fn with_curves(
        mut self, attack: CurveType, decay: CurveType, release: CurveType,
    ) -> Self {
        self.attack_curve = attack;
        self.decay_curve = decay;
        self.release_curve = release;
        self
    }

    fn time_to_increment(&self, time: f64) -> f64 {
        if time <= 0.0 || self.sample_rate <= 0.0 {
            1.0 // instant
        } else {
            1.0 / (time * self.sample_rate)
        }
    }

    /// Trigger the envelope (note-on). Supports retrigger from any state.
    pub fn trigger(&mut self) {
        self.state = AdsrState::Attack;
        self.stage_pos = 0.0;
        self.stage_increment = self.time_to_increment(self.attack_time);
    }

    /// Release the envelope (note-off).
    pub fn release(&mut self) {
        if self.state == AdsrState::Idle {
            return;
        }
        self.state = AdsrState::Release;
        self.release_start_level = self.level;
        self.stage_pos = 0.0;
        self.stage_increment = self.time_to_increment(self.release_time);
    }

    pub fn state(&self) -> AdsrState {
        self.state
    }

    pub fn level(&self) -> f64 {
        self.level
    }

    pub fn is_active(&self) -> bool {
        self.state != AdsrState::Idle
    }

    pub fn reset(&mut self) {
        self.state = AdsrState::Idle;
        self.level = 0.0;
        self.stage_pos = 0.0;
    }

    /// Apply curve to a normalised 0..1 position.
    fn apply_curve(pos: f64, curve: CurveType) -> f64 {
        let p = pos.clamp(0.0, 1.0);
        match curve {
            CurveType::Linear => p,
            CurveType::Exponential => {
                // Fast rise / fall
                p * p
            }
            CurveType::Logarithmic => {
                // Slow start, fast end
                1.0 - (1.0 - p) * (1.0 - p)
            }
        }
    }

    /// Compute the next sample of the envelope.
    #[inline]
    pub fn next_sample(&mut self) -> f64 {
        match self.state {
            AdsrState::Idle => {
                self.level = 0.0;
            }
            AdsrState::Attack => {
                self.stage_pos += self.stage_increment;
                if self.stage_pos >= 1.0 {
                    self.level = 1.0;
                    self.state = AdsrState::Decay;
                    self.stage_pos = 0.0;
                    self.stage_increment = self.time_to_increment(self.decay_time);
                } else {
                    self.level = Self::apply_curve(self.stage_pos, self.attack_curve);
                }
            }
            AdsrState::Decay => {
                self.stage_pos += self.stage_increment;
                if self.stage_pos >= 1.0 {
                    self.level = self.sustain_level;
                    self.state = AdsrState::Sustain;
                } else {
                    let curved = Self::apply_curve(self.stage_pos, self.decay_curve);
                    self.level = 1.0 - curved * (1.0 - self.sustain_level);
                }
            }
            AdsrState::Sustain => {
                self.level = self.sustain_level;
            }
            AdsrState::Release => {
                self.stage_pos += self.stage_increment;
                if self.stage_pos >= 1.0 {
                    self.level = 0.0;
                    self.state = AdsrState::Idle;
                } else {
                    let curved = Self::apply_curve(self.stage_pos, self.release_curve);
                    self.level = self.release_start_level * (1.0 - curved);
                }
            }
        }
        self.level
    }

    /// Render envelope into a mono buffer (multiplied with existing content).
    pub fn apply_to(&mut self, buf: &mut AudioBuf) {
        let frames = buf.frames();
        let ch = buf.channels;
        for f in 0..frames {
            let env = self.next_sample() as f32;
            for c in 0..ch {
                let s = buf.get(f, c);
                buf.set(f, c, s * env);
            }
        }
    }

    /// Render envelope values into a mono buffer (replacing content).
    pub fn render(&mut self, buf: &mut AudioBuf) {
        let frames = buf.frames();
        for f in 0..frames {
            buf.set(f, 0, self.next_sample() as f32);
        }
    }
}

// ---------------------------------------------------------------------------
// MultiSegmentEnvelope
// ---------------------------------------------------------------------------

/// One segment of a multi-segment envelope.
#[derive(Debug, Clone, Copy)]
pub struct EnvelopeSegment {
    /// Duration in seconds.
    pub duration: f64,
    /// Target level at end of segment.
    pub target_level: f64,
    pub curve: CurveType,
}

impl EnvelopeSegment {
    pub fn new(duration: f64, target_level: f64, curve: CurveType) -> Self {
        Self { duration, target_level, curve }
    }
}

/// Arbitrary multi-segment envelope with optional looping.
#[derive(Debug, Clone)]
pub struct MultiSegmentEnvelope {
    pub segments: Vec<EnvelopeSegment>,
    pub sample_rate: f64,
    pub loop_start: Option<usize>,
    pub loop_end: Option<usize>,
    current_segment: usize,
    stage_pos: f64,
    stage_increment: f64,
    level: f64,
    start_level: f64,
    active: bool,
}

impl MultiSegmentEnvelope {
    pub fn new(segments: Vec<EnvelopeSegment>, sample_rate: f64) -> Self {
        Self {
            segments,
            sample_rate,
            loop_start: None,
            loop_end: None,
            current_segment: 0,
            stage_pos: 0.0,
            stage_increment: 0.0,
            level: 0.0,
            start_level: 0.0,
            active: false,
        }
    }

    pub fn with_loop(mut self, start: usize, end: usize) -> Self {
        self.loop_start = Some(start);
        self.loop_end = Some(end);
        self
    }

    pub fn trigger(&mut self) {
        self.active = true;
        self.current_segment = 0;
        self.stage_pos = 0.0;
        self.level = 0.0;
        self.start_level = 0.0;
        self.begin_segment(0);
    }

    pub fn reset(&mut self) {
        self.active = false;
        self.level = 0.0;
        self.current_segment = 0;
        self.stage_pos = 0.0;
    }

    pub fn is_active(&self) -> bool {
        self.active
    }

    pub fn level(&self) -> f64 {
        self.level
    }

    fn begin_segment(&mut self, idx: usize) {
        if idx >= self.segments.len() {
            // Check looping
            if let (Some(ls), Some(le)) = (self.loop_start, self.loop_end) {
                if ls < self.segments.len() && le <= self.segments.len() && ls < le {
                    self.current_segment = ls;
                    self.stage_pos = 0.0;
                    self.start_level = self.level;
                    let seg = &self.segments[ls];
                    self.stage_increment = if seg.duration <= 0.0 || self.sample_rate <= 0.0 {
                        1.0
                    } else {
                        1.0 / (seg.duration * self.sample_rate)
                    };
                    return;
                }
            }
            self.active = false;
            return;
        }
        self.current_segment = idx;
        self.stage_pos = 0.0;
        self.start_level = self.level;
        let seg = &self.segments[idx];
        self.stage_increment = if seg.duration <= 0.0 || self.sample_rate <= 0.0 {
            1.0
        } else {
            1.0 / (seg.duration * self.sample_rate)
        };
    }

    #[inline]
    pub fn next_sample(&mut self) -> f64 {
        if !self.active || self.current_segment >= self.segments.len() {
            return self.level;
        }
        let seg = self.segments[self.current_segment];
        self.stage_pos += self.stage_increment;
        if self.stage_pos >= 1.0 {
            self.level = seg.target_level;
            let next = self.current_segment + 1;
            // Loop check
            if let (Some(_ls), Some(le)) = (self.loop_start, self.loop_end) {
                if next >= le {
                    self.begin_segment(self.loop_start.unwrap());
                    return self.level;
                }
            }
            self.begin_segment(next);
        } else {
            let curved = AdsrEnvelope::apply_curve(self.stage_pos, seg.curve);
            self.level = self.start_level + (seg.target_level - self.start_level) * curved;
        }
        self.level
    }

    pub fn render(&mut self, buf: &mut AudioBuf) {
        let frames = buf.frames();
        for f in 0..frames {
            buf.set(f, 0, self.next_sample() as f32);
        }
    }
}

// ---------------------------------------------------------------------------
// LfoEnvelope
// ---------------------------------------------------------------------------

/// LFO shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LfoShape {
    Sine,
    Triangle,
    Square,
    Random,
}

/// Low-frequency oscillator used as an envelope / modulation source.
#[derive(Debug, Clone)]
pub struct LfoEnvelope {
    pub shape: LfoShape,
    /// Rate in Hz.
    pub rate: f64,
    /// Depth (0..1).
    pub depth: f64,
    /// Phase offset in radians.
    pub phase_offset: f64,
    pub sample_rate: f64,
    phase: f64,
    rng_state: u64,
    random_target: f64,
    random_current: f64,
}

impl LfoEnvelope {
    pub fn new(shape: LfoShape, rate: f64, depth: f64, sample_rate: f64) -> Self {
        Self {
            shape,
            rate,
            depth,
            phase_offset: 0.0,
            sample_rate,
            phase: 0.0,
            rng_state: 0xDEADBEEF_CAFEBABE,
            random_target: 0.0,
            random_current: 0.0,
        }
    }

    pub fn set_rate(&mut self, rate: f64) {
        self.rate = rate;
    }

    pub fn set_depth(&mut self, depth: f64) {
        self.depth = depth.clamp(0.0, 1.0);
    }

    pub fn set_phase_offset(&mut self, offset: f64) {
        self.phase_offset = offset;
    }

    pub fn reset(&mut self) {
        self.phase = 0.0;
        self.random_current = 0.0;
        self.random_target = 0.0;
    }

    fn rand(&mut self) -> f64 {
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 7;
        self.rng_state ^= self.rng_state << 17;
        (self.rng_state as i64) as f64 / i64::MAX as f64
    }

    /// Returns a value in \[−depth, +depth\].
    #[inline]
    pub fn next_sample(&mut self) -> f64 {
        let dt = self.rate / self.sample_rate;
        let p = self.phase + self.phase_offset / TWO_PI;

        let raw = match self.shape {
            LfoShape::Sine => (TWO_PI * p).sin(),
            LfoShape::Triangle => {
                let t = (p % 1.0 + 1.0) % 1.0;
                if t < 0.5 { 4.0 * t - 1.0 } else { 3.0 - 4.0 * t }
            }
            LfoShape::Square => {
                let t = (p % 1.0 + 1.0) % 1.0;
                if t < 0.5 { 1.0 } else { -1.0 }
            }
            LfoShape::Random => {
                // Step random: pick new target at each cycle boundary
                let old_phase = self.phase;
                let new_phase = old_phase + dt;
                if new_phase.floor() > old_phase.floor() {
                    self.random_target = self.rand();
                }
                // Interpolate toward target
                self.random_current += 0.01 * (self.random_target - self.random_current);
                self.random_current
            }
        };

        self.phase += dt;
        if self.phase >= 1.0 {
            self.phase -= self.phase.floor();
        }

        raw * self.depth
    }

    pub fn render(&mut self, buf: &mut AudioBuf) {
        let frames = buf.frames();
        for f in 0..frames {
            buf.set(f, 0, self.next_sample() as f32);
        }
    }
}

// ---------------------------------------------------------------------------
// FollowerEnvelope
// ---------------------------------------------------------------------------

/// Detection mode for the amplitude follower.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectionMode {
    Peak,
    Rms,
}

/// Amplitude follower envelope that tracks the level of an input signal.
#[derive(Debug, Clone)]
pub struct FollowerEnvelope {
    pub mode: DetectionMode,
    pub attack_time: f64,
    pub release_time: f64,
    pub sample_rate: f64,
    attack_coeff: f64,
    release_coeff: f64,
    level: f64,
    /// Running sum for RMS window.
    rms_sum: f64,
    rms_buffer: Vec<f64>,
    rms_write_pos: usize,
}

impl FollowerEnvelope {
    pub fn new(mode: DetectionMode, attack: f64, release: f64, sample_rate: f64) -> Self {
        let attack_coeff = Self::time_to_coeff(attack, sample_rate);
        let release_coeff = Self::time_to_coeff(release, sample_rate);
        let rms_window = (0.05 * sample_rate) as usize; // 50 ms window
        Self {
            mode,
            attack_time: attack,
            release_time: release,
            sample_rate,
            attack_coeff,
            release_coeff,
            level: 0.0,
            rms_sum: 0.0,
            rms_buffer: vec![0.0; rms_window.max(1)],
            rms_write_pos: 0,
        }
    }

    fn time_to_coeff(time: f64, sample_rate: f64) -> f64 {
        if time <= 0.0 || sample_rate <= 0.0 { return 1.0; }
        1.0 - (-1.0 / (time * sample_rate)).exp()
    }

    pub fn set_attack(&mut self, time: f64) {
        self.attack_time = time;
        self.attack_coeff = Self::time_to_coeff(time, self.sample_rate);
    }

    pub fn set_release(&mut self, time: f64) {
        self.release_time = time;
        self.release_coeff = Self::time_to_coeff(time, self.sample_rate);
    }

    pub fn reset(&mut self) {
        self.level = 0.0;
        self.rms_sum = 0.0;
        for s in self.rms_buffer.iter_mut() { *s = 0.0; }
        self.rms_write_pos = 0;
    }

    pub fn level(&self) -> f64 {
        self.level
    }

    #[inline]
    pub fn next_sample(&mut self, input: f64) -> f64 {
        match self.mode {
            DetectionMode::Peak => {
                let abs = input.abs();
                let coeff = if abs > self.level {
                    self.attack_coeff
                } else {
                    self.release_coeff
                };
                self.level += coeff * (abs - self.level);
            }
            DetectionMode::Rms => {
                let sq = input * input;
                // Remove oldest, add newest
                let len = self.rms_buffer.len();
                self.rms_sum -= self.rms_buffer[self.rms_write_pos];
                self.rms_buffer[self.rms_write_pos] = sq;
                self.rms_sum += sq;
                self.rms_write_pos = (self.rms_write_pos + 1) % len;
                let rms = (self.rms_sum / len as f64).sqrt();
                let coeff = if rms > self.level {
                    self.attack_coeff
                } else {
                    self.release_coeff
                };
                self.level += coeff * (rms - self.level);
            }
        }
        self.level
    }

    /// Process an input buffer and write the follower level into `output`.
    pub fn process(&mut self, input: &AudioBuf, output: &mut AudioBuf) {
        let frames = input.frames().min(output.frames());
        for f in 0..frames {
            let inp = input.get(f, 0) as f64;
            let lev = self.next_sample(inp);
            output.set(f, 0, lev as f32);
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

    // -- AdsrEnvelope ---------------------------------------------------------

    #[test]
    fn adsr_idle_by_default() {
        let env = AdsrEnvelope::new(0.01, 0.1, 0.5, 0.2, 44100.0);
        assert_eq!(env.state(), AdsrState::Idle);
        assert!(!env.is_active());
    }

    #[test]
    fn adsr_trigger_reaches_peak() {
        let mut env = AdsrEnvelope::new(0.01, 0.1, 0.5, 0.2, 44100.0);
        env.trigger();
        let attack_samples = (0.01 * 44100.0) as usize + 10;
        let mut peak = 0.0f64;
        for _ in 0..attack_samples {
            let v = env.next_sample();
            if v > peak { peak = v; }
        }
        assert!(peak > 0.95, "peak should reach ~1.0, got {peak}");
    }

    #[test]
    fn adsr_sustain_level() {
        let mut env = AdsrEnvelope::new(0.001, 0.01, 0.6, 0.1, 44100.0);
        env.trigger();
        // Run through attack + decay
        for _ in 0..((0.001 + 0.01) * 44100.0) as usize + 100 {
            env.next_sample();
        }
        assert_eq!(env.state(), AdsrState::Sustain);
        assert!((env.level() - 0.6).abs() < 0.05);
    }

    #[test]
    fn adsr_release_to_idle() {
        let mut env = AdsrEnvelope::new(0.001, 0.001, 0.5, 0.01, 44100.0);
        env.trigger();
        // Attack + decay
        for _ in 0..500 { env.next_sample(); }
        env.release();
        // Run through release
        for _ in 0..((0.01 * 44100.0) as usize + 100) {
            env.next_sample();
        }
        assert_eq!(env.state(), AdsrState::Idle);
        assert!(env.level() < 0.01);
    }

    #[test]
    fn adsr_retrigger() {
        let mut env = AdsrEnvelope::new(0.01, 0.1, 0.5, 0.2, 44100.0);
        env.trigger();
        for _ in 0..100 { env.next_sample(); }
        env.trigger(); // retrigger during attack
        assert_eq!(env.state(), AdsrState::Attack);
    }

    #[test]
    fn adsr_apply_to_buffer() {
        let mut env = AdsrEnvelope::new(0.001, 0.001, 1.0, 0.1, 44100.0);
        env.trigger();
        let mut buf = make_buf(1024);
        for s in buf.data.iter_mut() { *s = 1.0; }
        env.apply_to(&mut buf);
        // First sample should be near 0 (start of attack), later samples near 1
        assert!(buf.data[0] < 0.5);
        assert!(buf.data[1023] > 0.5);
    }

    // -- MultiSegmentEnvelope -------------------------------------------------

    #[test]
    fn multi_segment_basic() {
        let segs = vec![
            EnvelopeSegment::new(0.01, 1.0, CurveType::Linear),
            EnvelopeSegment::new(0.01, 0.0, CurveType::Linear),
        ];
        let mut env = MultiSegmentEnvelope::new(segs, 44100.0);
        env.trigger();
        let mut peak = 0.0f64;
        for _ in 0..2000 {
            let v = env.next_sample();
            if v > peak { peak = v; }
        }
        assert!(peak > 0.9);
        assert!(!env.is_active());
    }

    #[test]
    fn multi_segment_looping() {
        let segs = vec![
            EnvelopeSegment::new(0.005, 1.0, CurveType::Linear),
            EnvelopeSegment::new(0.005, 0.0, CurveType::Linear),
        ];
        let mut env = MultiSegmentEnvelope::new(segs, 44100.0).with_loop(0, 2);
        env.trigger();
        // Should still be active after many samples (looping)
        for _ in 0..10000 {
            env.next_sample();
        }
        assert!(env.is_active());
    }

    // -- LfoEnvelope ----------------------------------------------------------

    #[test]
    fn lfo_sine_range() {
        let mut lfo = LfoEnvelope::new(LfoShape::Sine, 5.0, 1.0, 44100.0);
        let mut min = f64::MAX;
        let mut max = f64::MIN;
        for _ in 0..44100 {
            let v = lfo.next_sample();
            if v < min { min = v; }
            if v > max { max = v; }
        }
        assert!(max <= 1.01 && min >= -1.01);
    }

    #[test]
    fn lfo_square_shape() {
        let mut lfo = LfoEnvelope::new(LfoShape::Square, 10.0, 1.0, 44100.0);
        let mut buf = make_buf(4410);
        lfo.render(&mut buf);
        // Most samples should be near ±1
        let near = buf.data.iter().filter(|&&s| (s.abs() - 1.0).abs() < 0.01).count();
        assert!(near > buf.data.len() / 2);
    }

    #[test]
    fn lfo_depth_scaling() {
        let mut lfo = LfoEnvelope::new(LfoShape::Sine, 5.0, 0.5, 44100.0);
        let mut max = 0.0f64;
        for _ in 0..44100 {
            let v = lfo.next_sample().abs();
            if v > max { max = v; }
        }
        assert!(max <= 0.55, "depth should limit output to 0.5, got {max}");
    }

    // -- FollowerEnvelope -----------------------------------------------------

    #[test]
    fn follower_peak_tracks_level() {
        let mut fol = FollowerEnvelope::new(DetectionMode::Peak, 0.001, 0.01, 44100.0);
        for _ in 0..4410 {
            fol.next_sample(0.8);
        }
        assert!((fol.level() - 0.8).abs() < 0.05, "level={}", fol.level());
    }

    #[test]
    fn follower_rms_tracks_level() {
        let mut fol = FollowerEnvelope::new(DetectionMode::Rms, 0.001, 0.01, 44100.0);
        for _ in 0..44100 {
            fol.next_sample(0.5);
        }
        assert!((fol.level() - 0.5).abs() < 0.1, "rms level={}", fol.level());
    }

    #[test]
    fn follower_release_decays() {
        let mut fol = FollowerEnvelope::new(DetectionMode::Peak, 0.001, 0.05, 44100.0);
        for _ in 0..4410 { fol.next_sample(1.0); }
        for _ in 0..44100 { fol.next_sample(0.0); }
        assert!(fol.level() < 0.05, "should decay, level={}", fol.level());
    }

    #[test]
    fn follower_reset() {
        let mut fol = FollowerEnvelope::new(DetectionMode::Peak, 0.001, 0.01, 44100.0);
        fol.next_sample(1.0);
        fol.reset();
        assert!(fol.level() < 1e-12);
    }
}
