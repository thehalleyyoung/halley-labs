//! Detailed parameter types for each [`NodeType`](crate::NodeType).
//!
//! Every DSP node carries a [`NodeParameters`] enum that stores its
//! type-specific configuration.  Each variant wraps a dedicated params struct
//! (e.g. [`OscillatorParams`], [`FilterParams`]) that exposes validation,
//! interpolation helpers, and WCET-cost estimation.

use std::fmt;

// ---------------------------------------------------------------------------
// Enums shared by multiple parameter types
// ---------------------------------------------------------------------------

/// Oscillator waveform shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Waveform {
    Sine,
    Saw,
    Square,
    Triangle,
    Pulse,
    Noise,
}

impl fmt::Display for Waveform {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Sine => "Sine",
            Self::Saw => "Saw",
            Self::Square => "Square",
            Self::Triangle => "Triangle",
            Self::Pulse => "Pulse",
            Self::Noise => "Noise",
        };
        write!(f, "{}", s)
    }
}

/// Filter topology.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FilterType {
    LowPass,
    HighPass,
    BandPass,
    Notch,
    Allpass,
    LowShelf,
    HighShelf,
    Peaking,
}

impl fmt::Display for FilterType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// Envelope curve shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CurveType {
    Linear,
    Exponential,
    Logarithmic,
}

/// Modulation algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModulationType {
    AM,
    FM,
    Ring,
    PWM,
}

/// Noise spectrum colour.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NoiseType {
    White,
    Pink,
    Brown,
    Blue,
    Violet,
}

impl fmt::Display for NoiseType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

// ---------------------------------------------------------------------------
// Per-node parameter structs
// ---------------------------------------------------------------------------

/// Oscillator parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct OscillatorParams {
    pub waveform: Waveform,
    pub frequency: f64,
    pub phase: f64,
    pub amplitude: f64,
    pub detune: f64,
    pub pulse_width: f64,
}

impl Default for OscillatorParams {
    fn default() -> Self {
        Self {
            waveform: Waveform::Sine,
            frequency: 440.0,
            phase: 0.0,
            amplitude: 1.0,
            detune: 0.0,
            pulse_width: 0.5,
        }
    }
}

impl OscillatorParams {
    pub fn validate(&self) -> Vec<String> {
        let mut errs = Vec::new();
        if self.frequency < 0.0 || self.frequency > 22050.0 {
            errs.push(format!("frequency {} out of range [0, 22050]", self.frequency));
        }
        if self.phase < 0.0 || self.phase > std::f64::consts::TAU {
            errs.push(format!("phase {} out of range [0, 2π]", self.phase));
        }
        if self.amplitude < 0.0 || self.amplitude > 10.0 {
            errs.push(format!("amplitude {} out of range [0, 10]", self.amplitude));
        }
        if self.detune < -1200.0 || self.detune > 1200.0 {
            errs.push(format!("detune {} out of range [-1200, 1200] cents", self.detune));
        }
        if self.pulse_width < 0.0 || self.pulse_width > 1.0 {
            errs.push(format!("pulse_width {} out of range [0, 1]", self.pulse_width));
        }
        errs
    }

    /// Effective frequency including detune (cents).
    pub fn effective_frequency(&self) -> f64 {
        self.frequency * 2.0_f64.powf(self.detune / 1200.0)
    }

    /// Linearly interpolate towards `other` by factor `t ∈ [0,1]`.
    pub fn interpolate(&self, other: &Self, t: f64) -> Self {
        Self {
            waveform: if t < 0.5 { self.waveform } else { other.waveform },
            frequency: lerp(self.frequency, other.frequency, t),
            phase: lerp(self.phase, other.phase, t),
            amplitude: lerp(self.amplitude, other.amplitude, t),
            detune: lerp(self.detune, other.detune, t),
            pulse_width: lerp(self.pulse_width, other.pulse_width, t),
        }
    }

    pub fn wcet_us(&self) -> f64 {
        match self.waveform {
            Waveform::Sine => 10.0,
            Waveform::Saw | Waveform::Triangle => 8.0,
            Waveform::Square | Waveform::Pulse => 7.0,
            Waveform::Noise => 6.0,
        }
    }

    /// Approximate spectral bandwidth (Hz).
    pub fn spectral_bandwidth(&self) -> f64 {
        match self.waveform {
            Waveform::Sine => 0.0,
            Waveform::Saw | Waveform::Square | Waveform::Pulse => {
                // Harmonics up to Nyquist.
                22050.0 - self.effective_frequency()
            }
            Waveform::Triangle => (22050.0 - self.effective_frequency()).max(0.0) * 0.5,
            Waveform::Noise => 22050.0,
        }
    }
}

/// Filter parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct FilterParams {
    pub filter_type: FilterType,
    pub cutoff: f64,
    pub resonance: f64,
    pub gain: f64,
}

impl Default for FilterParams {
    fn default() -> Self {
        Self {
            filter_type: FilterType::LowPass,
            cutoff: 1000.0,
            resonance: 0.707,
            gain: 0.0,
        }
    }
}

impl FilterParams {
    pub fn validate(&self) -> Vec<String> {
        let mut errs = Vec::new();
        if self.cutoff < 20.0 || self.cutoff > 22050.0 {
            errs.push(format!("cutoff {} out of range [20, 22050]", self.cutoff));
        }
        if self.resonance < 0.1 || self.resonance > 30.0 {
            errs.push(format!("resonance {} out of range [0.1, 30]", self.resonance));
        }
        if self.gain < -60.0 || self.gain > 60.0 {
            errs.push(format!("gain {} out of range [-60, 60] dB", self.gain));
        }
        errs
    }

    pub fn interpolate(&self, other: &Self, t: f64) -> Self {
        Self {
            filter_type: if t < 0.5 { self.filter_type } else { other.filter_type },
            cutoff: lerp(self.cutoff, other.cutoff, t),
            resonance: lerp(self.resonance, other.resonance, t),
            gain: lerp(self.gain, other.gain, t),
        }
    }

    pub fn wcet_us(&self) -> f64 {
        match self.filter_type {
            FilterType::LowPass | FilterType::HighPass => 12.0,
            FilterType::BandPass | FilterType::Notch | FilterType::Allpass => 14.0,
            FilterType::LowShelf | FilterType::HighShelf | FilterType::Peaking => 16.0,
        }
    }

    /// Approximate -3 dB bandwidth (Hz) for bandpass-like filters.
    pub fn bandwidth_hz(&self) -> f64 {
        self.cutoff / self.resonance
    }
}

/// ADSR envelope parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct EnvelopeParams {
    pub attack: f64,
    pub decay: f64,
    pub sustain_level: f64,
    pub release: f64,
    pub curve_type: CurveType,
}

impl Default for EnvelopeParams {
    fn default() -> Self {
        Self {
            attack: 0.01,
            decay: 0.1,
            sustain_level: 0.7,
            release: 0.3,
            curve_type: CurveType::Linear,
        }
    }
}

impl EnvelopeParams {
    pub fn validate(&self) -> Vec<String> {
        let mut errs = Vec::new();
        if self.attack < 0.0 || self.attack > 30.0 {
            errs.push(format!("attack {} out of range [0, 30]s", self.attack));
        }
        if self.decay < 0.0 || self.decay > 30.0 {
            errs.push(format!("decay {} out of range [0, 30]s", self.decay));
        }
        if self.sustain_level < 0.0 || self.sustain_level > 1.0 {
            errs.push(format!("sustain_level {} out of range [0, 1]", self.sustain_level));
        }
        if self.release < 0.0 || self.release > 30.0 {
            errs.push(format!("release {} out of range [0, 30]s", self.release));
        }
        errs
    }

    pub fn interpolate(&self, other: &Self, t: f64) -> Self {
        Self {
            attack: lerp(self.attack, other.attack, t),
            decay: lerp(self.decay, other.decay, t),
            sustain_level: lerp(self.sustain_level, other.sustain_level, t),
            release: lerp(self.release, other.release, t),
            curve_type: if t < 0.5 { self.curve_type } else { other.curve_type },
        }
    }

    /// Total duration of the attack + decay + release phases.
    pub fn total_duration(&self) -> f64 {
        self.attack + self.decay + self.release
    }

    pub fn wcet_us(&self) -> f64 {
        match self.curve_type {
            CurveType::Linear => 4.0,
            CurveType::Exponential => 6.0,
            CurveType::Logarithmic => 6.0,
        }
    }

    /// Value of the envelope at time `t` assuming gate-on at 0, gate-off at `gate_off`.
    pub fn value_at(&self, t: f64, gate_off: f64) -> f64 {
        if t < 0.0 { return 0.0; }
        if t < self.attack {
            // Attack phase
            let frac = t / self.attack.max(1e-9);
            return self.apply_curve(frac);
        }
        let t_after_attack = t - self.attack;
        if t_after_attack < self.decay {
            let frac = t_after_attack / self.decay.max(1e-9);
            let curve_val = self.apply_curve(frac);
            return 1.0 - (1.0 - self.sustain_level) * curve_val;
        }
        if t < gate_off {
            return self.sustain_level;
        }
        let t_after_gate_off = t - gate_off;
        if t_after_gate_off < self.release {
            let frac = t_after_gate_off / self.release.max(1e-9);
            return self.sustain_level * (1.0 - self.apply_curve(frac));
        }
        0.0
    }

    fn apply_curve(&self, frac: f64) -> f64 {
        let f = frac.clamp(0.0, 1.0);
        match self.curve_type {
            CurveType::Linear => f,
            CurveType::Exponential => f * f,
            CurveType::Logarithmic => f.sqrt(),
        }
    }
}

/// Modulator parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct ModulatorParams {
    pub mod_type: ModulationType,
    pub rate: f64,
    pub depth: f64,
    pub waveform: Waveform,
}

impl Default for ModulatorParams {
    fn default() -> Self {
        Self {
            mod_type: ModulationType::AM,
            rate: 5.0,
            depth: 1.0,
            waveform: Waveform::Sine,
        }
    }
}

impl ModulatorParams {
    pub fn validate(&self) -> Vec<String> {
        let mut errs = Vec::new();
        if self.rate < 0.01 || self.rate > 20000.0 {
            errs.push(format!("rate {} out of range [0.01, 20000]", self.rate));
        }
        if self.depth < 0.0 || self.depth > 10.0 {
            errs.push(format!("depth {} out of range [0, 10]", self.depth));
        }
        errs
    }

    pub fn interpolate(&self, other: &Self, t: f64) -> Self {
        Self {
            mod_type: if t < 0.5 { self.mod_type } else { other.mod_type },
            rate: lerp(self.rate, other.rate, t),
            depth: lerp(self.depth, other.depth, t),
            waveform: if t < 0.5 { self.waveform } else { other.waveform },
        }
    }

    pub fn wcet_us(&self) -> f64 {
        match self.mod_type {
            ModulationType::AM => 10.0,
            ModulationType::FM => 14.0,
            ModulationType::Ring => 8.0,
            ModulationType::PWM => 12.0,
        }
    }
}

/// Compressor parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct CompressorParams {
    pub threshold: f64,
    pub ratio: f64,
    pub attack: f64,
    pub release: f64,
    pub knee: f64,
    pub makeup_gain: f64,
}

impl Default for CompressorParams {
    fn default() -> Self {
        Self {
            threshold: -20.0,
            ratio: 4.0,
            attack: 0.005,
            release: 0.1,
            knee: 6.0,
            makeup_gain: 0.0,
        }
    }
}

impl CompressorParams {
    pub fn validate(&self) -> Vec<String> {
        let mut errs = Vec::new();
        if self.threshold < -80.0 || self.threshold > 0.0 {
            errs.push(format!("threshold {} out of range [-80, 0] dB", self.threshold));
        }
        if self.ratio < 1.0 || self.ratio > 100.0 {
            errs.push(format!("ratio {} out of range [1, 100]", self.ratio));
        }
        if self.attack < 0.0001 || self.attack > 1.0 {
            errs.push(format!("attack {} out of range [0.0001, 1]s", self.attack));
        }
        if self.release < 0.001 || self.release > 5.0 {
            errs.push(format!("release {} out of range [0.001, 5]s", self.release));
        }
        if self.knee < 0.0 || self.knee > 40.0 {
            errs.push(format!("knee {} out of range [0, 40] dB", self.knee));
        }
        if self.makeup_gain < -20.0 || self.makeup_gain > 60.0 {
            errs.push(format!("makeup_gain {} out of range [-20, 60] dB", self.makeup_gain));
        }
        errs
    }

    pub fn interpolate(&self, other: &Self, t: f64) -> Self {
        Self {
            threshold: lerp(self.threshold, other.threshold, t),
            ratio: lerp(self.ratio, other.ratio, t),
            attack: lerp(self.attack, other.attack, t),
            release: lerp(self.release, other.release, t),
            knee: lerp(self.knee, other.knee, t),
            makeup_gain: lerp(self.makeup_gain, other.makeup_gain, t),
        }
    }

    /// Compute gain reduction in dB for a given input level in dBFS.
    pub fn gain_reduction_db(&self, input_db: f64) -> f64 {
        if input_db <= self.threshold - self.knee / 2.0 {
            0.0
        } else if input_db >= self.threshold + self.knee / 2.0 {
            (input_db - self.threshold) * (1.0 - 1.0 / self.ratio)
        } else {
            // Soft knee region.
            let x = input_db - self.threshold + self.knee / 2.0;
            (1.0 - 1.0 / self.ratio) * x * x / (2.0 * self.knee)
        }
    }

    pub fn wcet_us(&self) -> f64 { 20.0 }
}

/// Limiter parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct LimiterParams {
    pub threshold: f64,
    pub release: f64,
    pub lookahead: f64,
}

impl Default for LimiterParams {
    fn default() -> Self {
        Self {
            threshold: -1.0,
            release: 0.05,
            lookahead: 0.005,
        }
    }
}

impl LimiterParams {
    pub fn validate(&self) -> Vec<String> {
        let mut errs = Vec::new();
        if self.threshold < -40.0 || self.threshold > 0.0 {
            errs.push(format!("threshold {} out of range [-40, 0] dB", self.threshold));
        }
        if self.release < 0.001 || self.release > 2.0 {
            errs.push(format!("release {} out of range [0.001, 2]s", self.release));
        }
        if self.lookahead < 0.0 || self.lookahead > 0.05 {
            errs.push(format!("lookahead {} out of range [0, 0.05]s", self.lookahead));
        }
        errs
    }

    pub fn interpolate(&self, other: &Self, t: f64) -> Self {
        Self {
            threshold: lerp(self.threshold, other.threshold, t),
            release: lerp(self.release, other.release, t),
            lookahead: lerp(self.lookahead, other.lookahead, t),
        }
    }

    /// Lookahead in samples at a given sample rate.
    pub fn lookahead_samples(&self, sample_rate: f64) -> usize {
        (self.lookahead * sample_rate).round() as usize
    }

    pub fn wcet_us(&self) -> f64 { 15.0 }
}

/// Delay parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct DelayParams {
    pub delay_time: f64,
    pub feedback: f64,
    pub mix: f64,
}

impl Default for DelayParams {
    fn default() -> Self {
        Self {
            delay_time: 0.25,
            feedback: 0.3,
            mix: 0.5,
        }
    }
}

impl DelayParams {
    pub fn validate(&self) -> Vec<String> {
        let mut errs = Vec::new();
        if self.delay_time < 0.0 || self.delay_time > 10.0 {
            errs.push(format!("delay_time {} out of range [0, 10]s", self.delay_time));
        }
        if self.feedback < 0.0 || self.feedback > 1.0 {
            errs.push(format!("feedback {} out of range [0, 1]", self.feedback));
        }
        if self.mix < 0.0 || self.mix > 1.0 {
            errs.push(format!("mix {} out of range [0, 1]", self.mix));
        }
        errs
    }

    pub fn interpolate(&self, other: &Self, t: f64) -> Self {
        Self {
            delay_time: lerp(self.delay_time, other.delay_time, t),
            feedback: lerp(self.feedback, other.feedback, t),
            mix: lerp(self.mix, other.mix, t),
        }
    }

    /// Buffer size needed in samples.
    pub fn buffer_samples(&self, sample_rate: f64) -> usize {
        (self.delay_time * sample_rate).ceil() as usize
    }

    pub fn wcet_us(&self) -> f64 { 8.0 }
}

/// Pitch shifter parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct PitchShiftParams {
    pub semitones: f64,
    pub cents: f64,
    pub window_size: usize,
}

impl Default for PitchShiftParams {
    fn default() -> Self {
        Self {
            semitones: 0.0,
            cents: 0.0,
            window_size: 2048,
        }
    }
}

impl PitchShiftParams {
    pub fn validate(&self) -> Vec<String> {
        let mut errs = Vec::new();
        if self.semitones < -24.0 || self.semitones > 24.0 {
            errs.push(format!("semitones {} out of range [-24, 24]", self.semitones));
        }
        if self.cents < -100.0 || self.cents > 100.0 {
            errs.push(format!("cents {} out of range [-100, 100]", self.cents));
        }
        if self.window_size < 64 || self.window_size > 16384 {
            errs.push(format!("window_size {} out of range [64, 16384]", self.window_size));
        }
        errs
    }

    /// Total shift in cents.
    pub fn total_cents(&self) -> f64 {
        self.semitones * 100.0 + self.cents
    }

    /// Frequency ratio.
    pub fn ratio(&self) -> f64 {
        2.0_f64.powf(self.total_cents() / 1200.0)
    }

    pub fn interpolate(&self, other: &Self, t: f64) -> Self {
        Self {
            semitones: lerp(self.semitones, other.semitones, t),
            cents: lerp(self.cents, other.cents, t),
            window_size: if t < 0.5 { self.window_size } else { other.window_size },
        }
    }

    pub fn wcet_us(&self) -> f64 {
        // FFT-based: roughly O(N log N).
        let n = self.window_size as f64;
        10.0 + n * n.log2() * 0.002
    }
}

/// Time stretch parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct TimeStretchParams {
    pub ratio: f64,
    pub window_size: usize,
}

impl Default for TimeStretchParams {
    fn default() -> Self {
        Self {
            ratio: 1.0,
            window_size: 4096,
        }
    }
}

impl TimeStretchParams {
    pub fn validate(&self) -> Vec<String> {
        let mut errs = Vec::new();
        if self.ratio < 0.1 || self.ratio > 10.0 {
            errs.push(format!("ratio {} out of range [0.1, 10]", self.ratio));
        }
        if self.window_size < 128 || self.window_size > 16384 {
            errs.push(format!("window_size {} out of range [128, 16384]", self.window_size));
        }
        errs
    }

    pub fn interpolate(&self, other: &Self, t: f64) -> Self {
        Self {
            ratio: lerp(self.ratio, other.ratio, t),
            window_size: if t < 0.5 { self.window_size } else { other.window_size },
        }
    }

    pub fn wcet_us(&self) -> f64 {
        let n = self.window_size as f64;
        20.0 + n * n.log2() * 0.003
    }
}

/// Noise generator parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct NoiseParams {
    pub noise_type: NoiseType,
    pub amplitude: f64,
}

impl Default for NoiseParams {
    fn default() -> Self {
        Self { noise_type: NoiseType::White, amplitude: 1.0 }
    }
}

impl NoiseParams {
    pub fn validate(&self) -> Vec<String> {
        let mut errs = Vec::new();
        if self.amplitude < 0.0 || self.amplitude > 10.0 {
            errs.push(format!("amplitude {} out of range [0, 10]", self.amplitude));
        }
        errs
    }

    pub fn interpolate(&self, other: &Self, t: f64) -> Self {
        Self {
            noise_type: if t < 0.5 { self.noise_type } else { other.noise_type },
            amplitude: lerp(self.amplitude, other.amplitude, t),
        }
    }

    pub fn wcet_us(&self) -> f64 {
        match self.noise_type {
            NoiseType::White => 4.0,
            NoiseType::Pink => 8.0,
            NoiseType::Brown => 6.0,
            NoiseType::Blue => 8.0,
            NoiseType::Violet => 8.0,
        }
    }

    /// Approximate spectral centre frequency (Hz).
    pub fn spectral_centre(&self) -> f64 {
        match self.noise_type {
            NoiseType::White => 11025.0,
            NoiseType::Pink => 1000.0,
            NoiseType::Brown => 200.0,
            NoiseType::Blue => 15000.0,
            NoiseType::Violet => 18000.0,
        }
    }
}

/// Gain parameters (trivially simple but included for consistency).
#[derive(Debug, Clone, PartialEq)]
pub struct GainParams {
    pub level: f64,
}

impl Default for GainParams {
    fn default() -> Self { Self { level: 1.0 } }
}

impl GainParams {
    pub fn validate(&self) -> Vec<String> {
        let mut errs = Vec::new();
        if self.level < -100.0 || self.level > 100.0 {
            errs.push(format!("level {} out of range [-100, 100]", self.level));
        }
        errs
    }
    pub fn interpolate(&self, other: &Self, t: f64) -> Self {
        Self { level: lerp(self.level, other.level, t) }
    }
    pub fn wcet_us(&self) -> f64 { 2.0 }
}

/// Pan parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct PanParams {
    pub position: f64,
}

impl Default for PanParams {
    fn default() -> Self { Self { position: 0.0 } }
}

impl PanParams {
    pub fn validate(&self) -> Vec<String> {
        let mut errs = Vec::new();
        if self.position < -1.0 || self.position > 1.0 {
            errs.push(format!("position {} out of range [-1, 1]", self.position));
        }
        errs
    }
    pub fn interpolate(&self, other: &Self, t: f64) -> Self {
        Self { position: lerp(self.position, other.position, t) }
    }
    pub fn wcet_us(&self) -> f64 { 3.0 }
    /// Left channel gain (equal-power panning).
    pub fn left_gain(&self) -> f64 {
        ((1.0 - self.position) * std::f64::consts::FRAC_PI_4).cos()
    }
    /// Right channel gain.
    pub fn right_gain(&self) -> f64 {
        ((1.0 + self.position) * std::f64::consts::FRAC_PI_4).sin()
    }
}

/// Mixer parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct MixerParams {
    pub channel_count: usize,
    pub gains: Vec<f64>,
}

impl Default for MixerParams {
    fn default() -> Self {
        Self { channel_count: 2, gains: vec![1.0; 2] }
    }
}

impl MixerParams {
    pub fn new(channel_count: usize) -> Self {
        Self { channel_count, gains: vec![1.0; channel_count] }
    }
    pub fn validate(&self) -> Vec<String> {
        let mut errs = Vec::new();
        if self.channel_count == 0 || self.channel_count > 256 {
            errs.push(format!("channel_count {} out of range [1, 256]", self.channel_count));
        }
        if self.gains.len() != self.channel_count {
            errs.push(format!("gains length {} != channel_count {}", self.gains.len(), self.channel_count));
        }
        errs
    }
    pub fn wcet_us(&self) -> f64 { 3.0 + self.channel_count as f64 }
}

// ---------------------------------------------------------------------------
// NodeParameters enum
// ---------------------------------------------------------------------------

/// Union of all per-node parameter types.
#[derive(Debug, Clone, PartialEq)]
pub enum NodeParameters {
    None,
    Oscillator(OscillatorParams),
    Filter(FilterParams),
    Envelope(EnvelopeParams),
    Modulator(ModulatorParams),
    Compressor(CompressorParams),
    Limiter(LimiterParams),
    Delay(DelayParams),
    PitchShift(PitchShiftParams),
    TimeStretch(TimeStretchParams),
    Noise(NoiseParams),
    Gain(GainParams),
    Pan(PanParams),
    Mixer(MixerParams),
}

impl NodeParameters {
    /// Validate the parameters, returning a list of error messages.
    pub fn validate(&self) -> Vec<String> {
        match self {
            Self::None => vec![],
            Self::Oscillator(p) => p.validate(),
            Self::Filter(p) => p.validate(),
            Self::Envelope(p) => p.validate(),
            Self::Modulator(p) => p.validate(),
            Self::Compressor(p) => p.validate(),
            Self::Limiter(p) => p.validate(),
            Self::Delay(p) => p.validate(),
            Self::PitchShift(p) => p.validate(),
            Self::TimeStretch(p) => p.validate(),
            Self::Noise(p) => p.validate(),
            Self::Gain(p) => p.validate(),
            Self::Pan(p) => p.validate(),
            Self::Mixer(p) => p.validate(),
        }
    }

    /// WCET estimate for processing one block.
    pub fn wcet_us(&self) -> f64 {
        match self {
            Self::None => 0.5,
            Self::Oscillator(p) => p.wcet_us(),
            Self::Filter(p) => p.wcet_us(),
            Self::Envelope(p) => p.wcet_us(),
            Self::Modulator(p) => p.wcet_us(),
            Self::Compressor(p) => p.wcet_us(),
            Self::Limiter(p) => p.wcet_us(),
            Self::Delay(p) => p.wcet_us(),
            Self::PitchShift(p) => p.wcet_us(),
            Self::TimeStretch(p) => p.wcet_us(),
            Self::Noise(p) => p.wcet_us(),
            Self::Gain(p) => p.wcet_us(),
            Self::Pan(p) => p.wcet_us(),
            Self::Mixer(p) => p.wcet_us(),
        }
    }

    /// Interpolate sample-accurately towards `other`.
    pub fn interpolate(&self, other: &Self, t: f64) -> Self {
        match (self, other) {
            (Self::Oscillator(a), Self::Oscillator(b)) => Self::Oscillator(a.interpolate(b, t)),
            (Self::Filter(a), Self::Filter(b)) => Self::Filter(a.interpolate(b, t)),
            (Self::Envelope(a), Self::Envelope(b)) => Self::Envelope(a.interpolate(b, t)),
            (Self::Modulator(a), Self::Modulator(b)) => Self::Modulator(a.interpolate(b, t)),
            (Self::Compressor(a), Self::Compressor(b)) => Self::Compressor(a.interpolate(b, t)),
            (Self::Limiter(a), Self::Limiter(b)) => Self::Limiter(a.interpolate(b, t)),
            (Self::Delay(a), Self::Delay(b)) => Self::Delay(a.interpolate(b, t)),
            (Self::PitchShift(a), Self::PitchShift(b)) => Self::PitchShift(a.interpolate(b, t)),
            (Self::TimeStretch(a), Self::TimeStretch(b)) => Self::TimeStretch(a.interpolate(b, t)),
            (Self::Noise(a), Self::Noise(b)) => Self::Noise(a.interpolate(b, t)),
            (Self::Gain(a), Self::Gain(b)) => Self::Gain(a.interpolate(b, t)),
            (Self::Pan(a), Self::Pan(b)) => Self::Pan(a.interpolate(b, t)),
            _ => {
                // Incompatible types: snap at midpoint.
                if t < 0.5 { self.clone() } else { other.clone() }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Linear interpolation.
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

/// Convert dB to linear amplitude.
pub fn db_to_linear(db: f64) -> f64 {
    10.0_f64.powf(db / 20.0)
}

/// Convert linear amplitude to dB.
pub fn linear_to_db(lin: f64) -> f64 {
    20.0 * lin.max(1e-12).log10()
}

/// Convert frequency in Hz to Bark scale.
pub fn hz_to_bark(hz: f64) -> f64 {
    13.0 * (0.00076 * hz).atan() + 3.5 * ((hz / 7500.0).powi(2)).atan()
}

/// Convert Bark to Hz (approximate inverse).
pub fn bark_to_hz(bark: f64) -> f64 {
    // Newton iteration on the forward function; 3 iterations suffice.
    let mut hz = bark * 100.0; // initial guess
    for _ in 0..6 {
        let b = hz_to_bark(hz);
        let db = 13.0 * 0.00076 / (1.0 + (0.00076 * hz).powi(2))
            + 3.5 * 2.0 * hz / (7500.0 * 7500.0) / (1.0 + (hz / 7500.0).powi(2).powi(2) + 1e-12);
        hz -= (b - bark) / db.max(1e-12);
        hz = hz.max(0.0);
    }
    hz
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_oscillator_defaults() {
        let p = OscillatorParams::default();
        assert!(p.validate().is_empty());
        assert!((p.effective_frequency() - 440.0).abs() < 0.01);
    }

    #[test]
    fn test_oscillator_validation_frequency() {
        let mut p = OscillatorParams::default();
        p.frequency = -10.0;
        assert!(!p.validate().is_empty());
    }

    #[test]
    fn test_oscillator_interpolation() {
        let a = OscillatorParams { frequency: 200.0, ..Default::default() };
        let b = OscillatorParams { frequency: 600.0, ..Default::default() };
        let mid = a.interpolate(&b, 0.5);
        assert!((mid.frequency - 400.0).abs() < 0.01);
    }

    #[test]
    fn test_filter_defaults() {
        let p = FilterParams::default();
        assert!(p.validate().is_empty());
    }

    #[test]
    fn test_filter_bandwidth() {
        let p = FilterParams { cutoff: 1000.0, resonance: 2.0, ..Default::default() };
        assert!((p.bandwidth_hz() - 500.0).abs() < 0.01);
    }

    #[test]
    fn test_envelope_value_at() {
        let e = EnvelopeParams {
            attack: 1.0, decay: 1.0, sustain_level: 0.5, release: 1.0,
            curve_type: CurveType::Linear,
        };
        assert!((e.value_at(0.5, 10.0) - 0.5).abs() < 0.01); // mid-attack
        assert!((e.value_at(1.5, 10.0) - 0.75).abs() < 0.01); // mid-decay (1→0.5)
        assert!((e.value_at(5.0, 10.0) - 0.5).abs() < 0.01);  // sustain
    }

    #[test]
    fn test_compressor_gain_reduction() {
        let c = CompressorParams::default(); // threshold=-20, ratio=4
        // Well below threshold: no reduction.
        assert!((c.gain_reduction_db(-40.0)).abs() < 0.01);
        // Well above threshold: (input - threshold) * (1 - 1/ratio)
        let expected = (-10.0 - (-20.0)) * (1.0 - 0.25); // = 10 * 0.75 = 7.5
        assert!((c.gain_reduction_db(-10.0) - expected).abs() < 0.1);
    }

    #[test]
    fn test_limiter_lookahead_samples() {
        let l = LimiterParams { lookahead: 0.005, ..Default::default() };
        assert_eq!(l.lookahead_samples(48000.0), 240);
    }

    #[test]
    fn test_delay_buffer_samples() {
        let d = DelayParams { delay_time: 0.5, ..Default::default() };
        assert_eq!(d.buffer_samples(44100.0), 22050);
    }

    #[test]
    fn test_pitch_shift_ratio() {
        let p = PitchShiftParams { semitones: 12.0, cents: 0.0, ..Default::default() };
        assert!((p.ratio() - 2.0).abs() < 0.001); // octave up
    }

    #[test]
    fn test_noise_validation() {
        let n = NoiseParams { noise_type: NoiseType::Pink, amplitude: 11.0 };
        assert!(!n.validate().is_empty());
    }

    #[test]
    fn test_pan_equal_power() {
        let p = PanParams { position: 0.0 };
        let l = p.left_gain();
        let r = p.right_gain();
        assert!((l - r).abs() < 0.1); // centre → roughly equal
    }

    #[test]
    fn test_node_parameters_interpolation_mismatch() {
        let a = NodeParameters::Gain(GainParams { level: 0.5 });
        let b = NodeParameters::Filter(FilterParams::default());
        let mid = a.interpolate(&b, 0.3);
        // Should snap to a at t<0.5.
        assert!(matches!(mid, NodeParameters::Gain(_)));
    }

    #[test]
    fn test_db_to_linear() {
        assert!((db_to_linear(0.0) - 1.0).abs() < 0.001);
        assert!((db_to_linear(-6.0) - 0.501).abs() < 0.01);
    }

    #[test]
    fn test_hz_to_bark_roundtrip() {
        for &hz in &[100.0, 440.0, 1000.0, 4000.0, 10000.0] {
            let b = hz_to_bark(hz);
            let back = bark_to_hz(b);
            assert!((back - hz).abs() / hz < 0.05, "roundtrip failed for {}Hz: got {}Hz", hz, back);
        }
    }

    #[test]
    fn test_mixer_params_validation() {
        let m = MixerParams { channel_count: 3, gains: vec![1.0, 1.0] };
        assert!(!m.validate().is_empty());
    }

    #[test]
    fn test_time_stretch_validation() {
        let ts = TimeStretchParams { ratio: 0.05, window_size: 4096 };
        assert!(!ts.validate().is_empty());
    }
}
