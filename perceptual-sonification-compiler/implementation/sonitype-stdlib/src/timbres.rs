//! Timbral palettes for SoniType.
//!
//! Provides additive, FM, noise-band, and subtractive synthesis timbres plus
//! curated palettes for categorical sonification and timbre interpolation.

use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

fn clamp(v: f64, lo: f64, hi: f64) -> f64 {
    v.max(lo).min(hi)
}

fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

// ---------------------------------------------------------------------------
// Partial (shared harmonic descriptor)
// ---------------------------------------------------------------------------

/// A single partial / harmonic in an additive timbre.
#[derive(Debug, Clone)]
pub struct Partial {
    /// Harmonic number (1 = fundamental).
    pub harmonic: u32,
    /// Relative amplitude `[0, 1]`.
    pub amplitude: f64,
    /// Phase offset in radians.
    pub phase: f64,
}

impl Partial {
    pub fn new(harmonic: u32, amplitude: f64, phase: f64) -> Self {
        Self {
            harmonic,
            amplitude: clamp(amplitude, 0.0, 1.0),
            phase,
        }
    }
}

// ---------------------------------------------------------------------------
// AdditiveTimbre
// ---------------------------------------------------------------------------

/// Timbre defined by additive synthesis partials.
#[derive(Debug, Clone)]
pub struct AdditiveTimbre {
    pub name: String,
    pub partials: Vec<Partial>,
    /// Spectral roll-off exponent (higher = darker). Amplitude ∝ 1/n^rolloff.
    pub rolloff: f64,
}

impl AdditiveTimbre {
    pub fn new(name: impl Into<String>, partials: Vec<Partial>) -> Self {
        Self { name: name.into(), partials, rolloff: 1.0 }
    }

    pub fn with_rolloff(mut self, rolloff: f64) -> Self {
        self.rolloff = rolloff;
        self
    }

    /// Generate N harmonics with the given roll-off.
    pub fn from_rolloff(name: impl Into<String>, n_harmonics: u32, rolloff: f64) -> Self {
        let partials: Vec<Partial> = (1..=n_harmonics)
            .map(|h| Partial::new(h, 1.0 / (h as f64).powf(rolloff), 0.0))
            .collect();
        Self { name: name.into(), partials, rolloff }
    }

    /// Only odd harmonics (e.g. clarinet-like).
    pub fn odd_harmonics(name: impl Into<String>, n: u32, rolloff: f64) -> Self {
        let partials: Vec<Partial> = (0..n)
            .map(|i| {
                let h = 2 * i + 1;
                Partial::new(h, 1.0 / (h as f64).powf(rolloff), 0.0)
            })
            .collect();
        Self { name: name.into(), partials, rolloff }
    }

    /// Only even harmonics.
    pub fn even_harmonics(name: impl Into<String>, n: u32, rolloff: f64) -> Self {
        let partials: Vec<Partial> = (1..=n)
            .map(|i| {
                let h = 2 * i;
                Partial::new(h, 1.0 / (h as f64).powf(rolloff), 0.0)
            })
            .collect();
        Self { name: name.into(), partials, rolloff }
    }

    // --- instrument presets ---

    pub fn flute() -> Self {
        Self::new("flute", vec![
            Partial::new(1, 1.0, 0.0),
            Partial::new(2, 0.25, 0.0),
            Partial::new(3, 0.06, 0.0),
            Partial::new(4, 0.015, 0.0),
        ]).with_rolloff(2.5)
    }

    pub fn clarinet() -> Self {
        Self::new("clarinet", vec![
            Partial::new(1, 1.0, 0.0),
            Partial::new(3, 0.75, 0.0),
            Partial::new(5, 0.50, 0.0),
            Partial::new(7, 0.25, 0.0),
            Partial::new(9, 0.14, 0.0),
            Partial::new(11, 0.09, 0.0),
        ]).with_rolloff(1.5)
    }

    pub fn oboe() -> Self {
        Self::new("oboe", vec![
            Partial::new(1, 1.0, 0.0),
            Partial::new(2, 0.8, 0.0),
            Partial::new(3, 0.6, 0.0),
            Partial::new(4, 0.5, 0.0),
            Partial::new(5, 0.4, 0.0),
            Partial::new(6, 0.3, 0.0),
            Partial::new(7, 0.2, 0.0),
            Partial::new(8, 0.1, 0.0),
        ]).with_rolloff(1.0)
    }

    pub fn trumpet() -> Self {
        Self::new("trumpet", vec![
            Partial::new(1, 0.8, 0.0),
            Partial::new(2, 1.0, 0.0),
            Partial::new(3, 0.9, 0.0),
            Partial::new(4, 0.7, 0.0),
            Partial::new(5, 0.5, 0.0),
            Partial::new(6, 0.35, 0.0),
            Partial::new(7, 0.2, 0.0),
            Partial::new(8, 0.12, 0.0),
            Partial::new(9, 0.07, 0.0),
            Partial::new(10, 0.04, 0.0),
        ]).with_rolloff(0.8)
    }

    /// Compute a brightness metric (spectral centroid proxy, normalised).
    pub fn brightness(&self) -> f64 {
        if self.partials.is_empty() {
            return 0.0;
        }
        let total_amp: f64 = self.partials.iter().map(|p| p.amplitude).sum();
        if total_amp < 1e-12 {
            return 0.0;
        }
        let weighted: f64 = self.partials.iter()
            .map(|p| p.harmonic as f64 * p.amplitude)
            .sum();
        weighted / total_amp
    }

    /// Set even/odd harmonic balance. `balance` ∈ `[-1, 1]`:
    /// -1 = only odd, 0 = natural, 1 = only even.
    pub fn set_harmonic_balance(&mut self, balance: f64) {
        let b = clamp(balance, -1.0, 1.0);
        for p in &mut self.partials {
            let is_even = p.harmonic % 2 == 0;
            let factor = if is_even {
                (1.0 + b) / 2.0
            } else {
                (1.0 - b) / 2.0
            };
            p.amplitude *= factor;
        }
    }

    /// Synthesize one period of the waveform at the given number of samples.
    pub fn render_period(&self, num_samples: usize) -> Vec<f64> {
        let mut buf = vec![0.0; num_samples];
        for (i, sample) in buf.iter_mut().enumerate() {
            let phase = 2.0 * PI * (i as f64 / num_samples as f64);
            for p in &self.partials {
                *sample += p.amplitude * (p.harmonic as f64 * phase + p.phase).sin();
            }
        }
        // Normalise peak to 1.0
        let peak = buf.iter().copied().fold(0.0_f64, |a, b| a.max(b.abs()));
        if peak > 1e-12 {
            for s in &mut buf {
                *s /= peak;
            }
        }
        buf
    }

    /// Return the spectral envelope as (harmonic_number, amplitude) pairs.
    pub fn spectral_envelope(&self) -> Vec<(u32, f64)> {
        self.partials.iter().map(|p| (p.harmonic, p.amplitude)).collect()
    }
}

// ---------------------------------------------------------------------------
// FMTimbre
// ---------------------------------------------------------------------------

/// FM synthesis algorithm topology.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FMAlgorithm {
    /// Simple 2-operator: carrier ← modulator.
    TwoOp,
    /// 4-operator stack: C ← M1 ← M2 ← M3.
    FourOpStack,
    /// 4-operator parallel: C1 ← M1, C2 ← M2.
    FourOpParallel,
    /// 4-operator with feedback on last modulator.
    FourOpFeedback,
}

/// FM synthesis timbre.
#[derive(Debug, Clone)]
pub struct FMTimbre {
    pub name: String,
    pub algorithm: FMAlgorithm,
    /// Carrier:modulator frequency ratios.
    pub cm_ratios: Vec<(f64, f64)>,
    /// Modulation indices per operator pair.
    pub mod_indices: Vec<f64>,
    /// Feedback amount for feedback algorithms `[0, 1]`.
    pub feedback: f64,
}

impl FMTimbre {
    pub fn new(
        name: impl Into<String>,
        algorithm: FMAlgorithm,
        cm_ratios: Vec<(f64, f64)>,
        mod_indices: Vec<f64>,
    ) -> Self {
        Self {
            name: name.into(),
            algorithm,
            cm_ratios,
            mod_indices,
            feedback: 0.0,
        }
    }

    pub fn with_feedback(mut self, fb: f64) -> Self {
        self.feedback = clamp(fb, 0.0, 1.0);
        self
    }

    // --- Presets ---

    pub fn electric_piano() -> Self {
        Self::new("electric_piano", FMAlgorithm::TwoOp,
            vec![(1.0, 1.0)],
            vec![1.5],
        )
    }

    pub fn bell() -> Self {
        Self::new("bell", FMAlgorithm::TwoOp,
            vec![(1.0, 3.5)],
            vec![5.0],
        )
    }

    pub fn brass() -> Self {
        Self::new("brass", FMAlgorithm::TwoOp,
            vec![(1.0, 1.0)],
            vec![3.0],
        )
    }

    pub fn pad() -> Self {
        Self::new("pad", FMAlgorithm::FourOpParallel,
            vec![(1.0, 1.001), (1.0, 2.0)],
            vec![0.5, 0.3],
        )
    }

    pub fn metallic() -> Self {
        Self::new("metallic", FMAlgorithm::TwoOp,
            vec![(1.0, 1.414)],
            vec![8.0],
        )
    }

    pub fn organ() -> Self {
        Self::new("organ", FMAlgorithm::FourOpStack,
            vec![(1.0, 1.0), (1.0, 2.0), (1.0, 3.0)],
            vec![0.8, 0.6, 0.4],
        )
    }

    /// Map a modulation-index value to a brightness proxy.
    pub fn brightness(&self) -> f64 {
        if self.mod_indices.is_empty() {
            return 0.0;
        }
        let avg_idx: f64 = self.mod_indices.iter().sum::<f64>() / self.mod_indices.len() as f64;
        // Bessel-function approximation: sidebands ~ mod_index + 1
        clamp(avg_idx / 10.0, 0.0, 1.0)
    }

    /// Return an estimated number of significant sidebands.
    pub fn sideband_count(&self) -> usize {
        if self.mod_indices.is_empty() {
            return 1;
        }
        let max_idx = self.mod_indices.iter().copied().fold(0.0_f64, f64::max);
        (max_idx + 1.0).ceil() as usize
    }

    /// Render a single cycle of the 2-op FM waveform.
    pub fn render_period_2op(&self, num_samples: usize) -> Vec<f64> {
        if self.cm_ratios.is_empty() || self.mod_indices.is_empty() {
            return vec![0.0; num_samples];
        }
        let (c_ratio, m_ratio) = self.cm_ratios[0];
        let mod_index = self.mod_indices[0];
        let mut buf = vec![0.0; num_samples];
        for (i, sample) in buf.iter_mut().enumerate() {
            let phase = 2.0 * PI * (i as f64 / num_samples as f64);
            let modulator = (m_ratio * phase).sin();
            *sample = (c_ratio * phase + mod_index * modulator).sin();
        }
        buf
    }
}

// ---------------------------------------------------------------------------
// NoiseBandTimbre
// ---------------------------------------------------------------------------

/// Filtered noise band timbre specification.
#[derive(Debug, Clone)]
pub struct NoiseBandTimbre {
    pub name: String,
    /// Centre frequency in Hz.
    pub center_hz: f64,
    /// Bandwidth in Hz.
    pub bandwidth_hz: f64,
    /// Resonance / Q factor.
    pub resonance: f64,
    /// Amplitude `[0, 1]`.
    pub amplitude: f64,
}

impl NoiseBandTimbre {
    pub fn new(name: impl Into<String>, center_hz: f64, bandwidth_hz: f64) -> Self {
        Self {
            name: name.into(),
            center_hz,
            bandwidth_hz,
            resonance: 1.0,
            amplitude: 1.0,
        }
    }

    pub fn with_resonance(mut self, q: f64) -> Self {
        self.resonance = q.max(0.1);
        self
    }

    pub fn with_amplitude(mut self, amp: f64) -> Self {
        self.amplitude = clamp(amp, 0.0, 1.0);
        self
    }

    /// Q factor from centre and bandwidth.
    pub fn q_factor(&self) -> f64 {
        if self.bandwidth_hz > 0.0 {
            self.center_hz / self.bandwidth_hz
        } else {
            f64::INFINITY
        }
    }

    // --- Presets ---

    pub fn rain() -> Self {
        Self::new("rain", 6000.0, 8000.0).with_resonance(0.5).with_amplitude(0.6)
    }

    pub fn wind() -> Self {
        Self::new("wind", 800.0, 1200.0).with_resonance(0.8).with_amplitude(0.5)
    }

    pub fn ocean() -> Self {
        Self::new("ocean", 300.0, 500.0).with_resonance(0.6).with_amplitude(0.7)
    }

    pub fn static_noise() -> Self {
        Self::new("static", 4000.0, 8000.0).with_resonance(0.3).with_amplitude(0.4)
    }

    pub fn hiss() -> Self {
        Self::new("hiss", 8000.0, 4000.0).with_resonance(1.0).with_amplitude(0.3)
    }

    /// Brightness approximation based on centre frequency.
    pub fn brightness(&self) -> f64 {
        clamp(self.center_hz / 10000.0, 0.0, 1.0)
    }
}

// ---------------------------------------------------------------------------
// SubtractiveTimbre
// ---------------------------------------------------------------------------

/// Source waveform for subtractive synthesis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceWaveform {
    Sawtooth,
    Square,
    Triangle,
    Pulse,
    WhiteNoise,
}

/// Filter type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterType {
    LowPass,
    HighPass,
    BandPass,
    Notch,
}

/// Subtractive synthesis timbre.
#[derive(Debug, Clone)]
pub struct SubtractiveTimbre {
    pub name: String,
    pub source: SourceWaveform,
    pub filter_type: FilterType,
    /// Filter cutoff frequency in Hz.
    pub cutoff_hz: f64,
    /// Filter resonance `[0, 1]` (0 = none, 1 = self-oscillation).
    pub resonance: f64,
    /// Pulse width for `Pulse` waveform `[0.01, 0.99]`.
    pub pulse_width: f64,
}

impl SubtractiveTimbre {
    pub fn new(
        name: impl Into<String>,
        source: SourceWaveform,
        filter_type: FilterType,
        cutoff_hz: f64,
        resonance: f64,
    ) -> Self {
        Self {
            name: name.into(),
            source,
            filter_type,
            cutoff_hz,
            resonance: clamp(resonance, 0.0, 1.0),
            pulse_width: 0.5,
        }
    }

    pub fn with_pulse_width(mut self, pw: f64) -> Self {
        self.pulse_width = clamp(pw, 0.01, 0.99);
        self
    }

    // --- Presets ---

    pub fn moog_bass() -> Self {
        Self::new("moog_bass", SourceWaveform::Sawtooth, FilterType::LowPass, 800.0, 0.7)
    }

    pub fn acid_squelch() -> Self {
        Self::new("acid_squelch", SourceWaveform::Sawtooth, FilterType::LowPass, 400.0, 0.9)
    }

    pub fn warm_pad() -> Self {
        Self::new("warm_pad", SourceWaveform::Square, FilterType::LowPass, 2000.0, 0.2)
    }

    pub fn pluck() -> Self {
        Self::new("pluck", SourceWaveform::Sawtooth, FilterType::LowPass, 5000.0, 0.3)
    }

    pub fn pwm_lead() -> Self {
        Self::new("pwm_lead", SourceWaveform::Pulse, FilterType::LowPass, 3000.0, 0.5)
            .with_pulse_width(0.3)
    }

    /// Brightness estimate from filter cutoff.
    pub fn brightness(&self) -> f64 {
        clamp(self.cutoff_hz / 10000.0, 0.0, 1.0)
    }

    /// Render one period of the source waveform.
    pub fn render_source_period(&self, num_samples: usize) -> Vec<f64> {
        let mut buf = Vec::with_capacity(num_samples);
        for i in 0..num_samples {
            let t = i as f64 / num_samples as f64;
            let sample = match self.source {
                SourceWaveform::Sawtooth => 2.0 * t - 1.0,
                SourceWaveform::Square => if t < 0.5 { 1.0 } else { -1.0 },
                SourceWaveform::Triangle => {
                    if t < 0.25 { 4.0 * t }
                    else if t < 0.75 { 2.0 - 4.0 * t }
                    else { 4.0 * t - 4.0 }
                }
                SourceWaveform::Pulse => if t < self.pulse_width { 1.0 } else { -1.0 },
                SourceWaveform::WhiteNoise => {
                    // deterministic pseudo-noise for reproducibility in tests
                    let x = ((i as f64 * 12.9898).sin() * 43758.5453).fract();
                    2.0 * x - 1.0
                }
            };
            buf.push(sample);
        }
        buf
    }
}

// ---------------------------------------------------------------------------
// TimbrePalette
// ---------------------------------------------------------------------------

/// Descriptor for a generic timbre within a palette.
#[derive(Debug, Clone)]
pub struct TimbreDescriptor {
    pub name: String,
    /// Spectral brightness `[0, 1]`.
    pub brightness: f64,
    /// Noisiness `[0, 1]` (0 = pure tone, 1 = pure noise).
    pub noisiness: f64,
    /// Roughness / inharmonicity `[0, 1]`.
    pub roughness: f64,
    /// Spectral centroid estimate in Hz.
    pub centroid_hz: f64,
}

impl TimbreDescriptor {
    pub fn new(name: impl Into<String>, brightness: f64, noisiness: f64, roughness: f64) -> Self {
        Self {
            name: name.into(),
            brightness: clamp(brightness, 0.0, 1.0),
            noisiness: clamp(noisiness, 0.0, 1.0),
            roughness: clamp(roughness, 0.0, 1.0),
            centroid_hz: brightness * 8000.0 + 200.0,
        }
    }

    /// Perceptual distance between two timbre descriptors.
    pub fn distance(&self, other: &Self) -> f64 {
        let db = self.brightness - other.brightness;
        let dn = self.noisiness - other.noisiness;
        let dr = self.roughness - other.roughness;
        (db * db + dn * dn + dr * dr).sqrt()
    }
}

/// A curated collection of perceptually distinguishable timbres.
#[derive(Debug, Clone)]
pub struct TimbrePalette {
    pub name: String,
    pub descriptors: Vec<TimbreDescriptor>,
}

impl TimbrePalette {
    pub fn new(name: impl Into<String>, descriptors: Vec<TimbreDescriptor>) -> Self {
        Self { name: name.into(), descriptors }
    }

    /// 8-timbre palette optimised for maximal perceptual distance.
    pub fn palette_8() -> Self {
        Self::new("palette_8", vec![
            TimbreDescriptor::new("sine", 0.05, 0.0, 0.0),
            TimbreDescriptor::new("flute", 0.15, 0.05, 0.0),
            TimbreDescriptor::new("clarinet", 0.30, 0.02, 0.05),
            TimbreDescriptor::new("trumpet", 0.55, 0.05, 0.10),
            TimbreDescriptor::new("oboe", 0.40, 0.03, 0.15),
            TimbreDescriptor::new("bell", 0.70, 0.0, 0.50),
            TimbreDescriptor::new("noise_band", 0.50, 0.80, 0.30),
            TimbreDescriptor::new("metallic", 0.85, 0.10, 0.70),
        ])
    }

    /// 16-timbre extended palette.
    pub fn palette_16() -> Self {
        let mut descs = Self::palette_8().descriptors;
        descs.extend(vec![
            TimbreDescriptor::new("piano", 0.35, 0.01, 0.02),
            TimbreDescriptor::new("electric_piano", 0.40, 0.02, 0.08),
            TimbreDescriptor::new("pad", 0.20, 0.10, 0.05),
            TimbreDescriptor::new("pluck", 0.60, 0.03, 0.12),
            TimbreDescriptor::new("bass", 0.10, 0.02, 0.03),
            TimbreDescriptor::new("wind", 0.25, 0.70, 0.15),
            TimbreDescriptor::new("acid", 0.75, 0.05, 0.40),
            TimbreDescriptor::new("chime", 0.65, 0.01, 0.35),
        ]);
        Self::new("palette_16", descs)
    }

    /// Select a sub-palette of `n` maximally-spaced timbres (greedy farthest-first).
    pub fn select_n(&self, n: usize) -> Vec<&TimbreDescriptor> {
        if n == 0 || self.descriptors.is_empty() {
            return Vec::new();
        }
        let n = n.min(self.descriptors.len());
        let mut selected = vec![0usize]; // start with first
        let mut min_dists: Vec<f64> = self.descriptors.iter()
            .map(|d| d.distance(&self.descriptors[0]))
            .collect();

        for _ in 1..n {
            // pick the descriptor with the maximum min-distance
            let next = (0..self.descriptors.len())
                .filter(|i| !selected.contains(i))
                .max_by(|&a, &b| min_dists[a].partial_cmp(&min_dists[b]).unwrap())
                .unwrap();
            selected.push(next);
            // update min distances
            for i in 0..self.descriptors.len() {
                let d = self.descriptors[i].distance(&self.descriptors[next]);
                if d < min_dists[i] {
                    min_dists[i] = d;
                }
            }
        }

        selected.into_iter().map(|i| &self.descriptors[i]).collect()
    }

    /// Minimum pairwise distance in the palette.
    pub fn min_distance(&self) -> f64 {
        let mut min_d = f64::INFINITY;
        for i in 0..self.descriptors.len() {
            for j in (i + 1)..self.descriptors.len() {
                let d = self.descriptors[i].distance(&self.descriptors[j]);
                if d < min_d {
                    min_d = d;
                }
            }
        }
        if min_d.is_infinite() { 0.0 } else { min_d }
    }

    /// Return descriptors sorted by brightness.
    pub fn sorted_by_brightness(&self) -> Vec<&TimbreDescriptor> {
        let mut refs: Vec<&TimbreDescriptor> = self.descriptors.iter().collect();
        refs.sort_by(|a, b| a.brightness.partial_cmp(&b.brightness).unwrap());
        refs
    }

    /// Number of timbres in the palette.
    pub fn len(&self) -> usize {
        self.descriptors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.descriptors.is_empty()
    }

    /// Look up timbre by name.
    pub fn get(&self, name: &str) -> Option<&TimbreDescriptor> {
        self.descriptors.iter().find(|d| d.name == name)
    }
}

// ---------------------------------------------------------------------------
// TimbreInterpolator
// ---------------------------------------------------------------------------

/// Morph between two timbre descriptors.
#[derive(Debug, Clone)]
pub struct TimbreInterpolator {
    pub source: TimbreDescriptor,
    pub target: TimbreDescriptor,
}

impl TimbreInterpolator {
    pub fn new(source: TimbreDescriptor, target: TimbreDescriptor) -> Self {
        Self { source, target }
    }

    /// Interpolate at `t ∈ [0, 1]` (0 = source, 1 = target).
    pub fn interpolate(&self, t: f64) -> TimbreDescriptor {
        let t = clamp(t, 0.0, 1.0);
        TimbreDescriptor {
            name: format!("interp_{:.2}", t),
            brightness: lerp(self.source.brightness, self.target.brightness, t),
            noisiness: lerp(self.source.noisiness, self.target.noisiness, t),
            roughness: lerp(self.source.roughness, self.target.roughness, t),
            centroid_hz: lerp(self.source.centroid_hz, self.target.centroid_hz, t),
        }
    }

    /// Generate `n` equally-spaced interpolated descriptors (including endpoints).
    pub fn interpolate_n(&self, n: usize) -> Vec<TimbreDescriptor> {
        if n <= 1 {
            return vec![self.interpolate(0.0)];
        }
        (0..n)
            .map(|i| self.interpolate(i as f64 / (n - 1) as f64))
            .collect()
    }

    /// Interpolate spectral envelopes between two additive timbres.
    pub fn interpolate_additive(a: &AdditiveTimbre, b: &AdditiveTimbre, t: f64) -> AdditiveTimbre {
        let t = clamp(t, 0.0, 1.0);
        let max_h = a.partials.iter().map(|p| p.harmonic)
            .chain(b.partials.iter().map(|p| p.harmonic))
            .max()
            .unwrap_or(1);

        let mut partials = Vec::new();
        for h in 1..=max_h {
            let amp_a = a.partials.iter().find(|p| p.harmonic == h).map(|p| p.amplitude).unwrap_or(0.0);
            let amp_b = b.partials.iter().find(|p| p.harmonic == h).map(|p| p.amplitude).unwrap_or(0.0);
            let phase_a = a.partials.iter().find(|p| p.harmonic == h).map(|p| p.phase).unwrap_or(0.0);
            let phase_b = b.partials.iter().find(|p| p.harmonic == h).map(|p| p.phase).unwrap_or(0.0);
            let amp = lerp(amp_a, amp_b, t);
            if amp > 1e-6 {
                partials.push(Partial::new(h, amp, lerp(phase_a, phase_b, t)));
            }
        }
        let rolloff = lerp(a.rolloff, b.rolloff, t);
        AdditiveTimbre {
            name: format!("{}_to_{}_at_{:.2}", a.name, b.name, t),
            partials,
            rolloff,
        }
    }

    /// Crossfade between two palettes at parameter `t`.
    pub fn crossfade_palettes(a: &TimbrePalette, b: &TimbrePalette, t: f64) -> TimbrePalette {
        let t = clamp(t, 0.0, 1.0);
        let n = a.descriptors.len().max(b.descriptors.len());
        let mut descs = Vec::new();
        for i in 0..n {
            let da = a.descriptors.get(i).cloned()
                .unwrap_or_else(|| TimbreDescriptor::new("silence", 0.0, 0.0, 0.0));
            let db = b.descriptors.get(i).cloned()
                .unwrap_or_else(|| TimbreDescriptor::new("silence", 0.0, 0.0, 0.0));
            descs.push(TimbreDescriptor {
                name: format!("xfade_{}", i),
                brightness: lerp(da.brightness, db.brightness, t),
                noisiness: lerp(da.noisiness, db.noisiness, t),
                roughness: lerp(da.roughness, db.roughness, t),
                centroid_hz: lerp(da.centroid_hz, db.centroid_hz, t),
            });
        }
        TimbrePalette::new(format!("crossfade_{:.2}", t), descs)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_additive_flute_brightness() {
        let f = AdditiveTimbre::flute();
        assert!(f.brightness() > 0.0 && f.brightness() < 2.0);
    }

    #[test]
    fn test_additive_clarinet_odd_harmonics() {
        let c = AdditiveTimbre::clarinet();
        // All partials should be odd harmonics
        for p in &c.partials {
            assert!(p.harmonic % 2 == 1);
        }
    }

    #[test]
    fn test_additive_render_period_length() {
        let t = AdditiveTimbre::flute();
        let buf = t.render_period(256);
        assert_eq!(buf.len(), 256);
    }

    #[test]
    fn test_additive_render_period_normalised() {
        let t = AdditiveTimbre::trumpet();
        let buf = t.render_period(1024);
        let peak = buf.iter().copied().fold(0.0_f64, |a, b| a.max(b.abs()));
        assert!((peak - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_additive_harmonic_balance() {
        let mut t = AdditiveTimbre::from_rolloff("test", 8, 1.0);
        t.set_harmonic_balance(-1.0); // only odd
        for p in &t.partials {
            if p.harmonic % 2 == 0 {
                assert!(p.amplitude < 1e-6);
            }
        }
    }

    #[test]
    fn test_fm_electric_piano_brightness() {
        let ep = FMTimbre::electric_piano();
        assert!(ep.brightness() > 0.0);
    }

    #[test]
    fn test_fm_bell_sideband_count() {
        let b = FMTimbre::bell();
        assert!(b.sideband_count() >= 5);
    }

    #[test]
    fn test_fm_render_2op_length() {
        let ep = FMTimbre::electric_piano();
        let buf = ep.render_period_2op(512);
        assert_eq!(buf.len(), 512);
    }

    #[test]
    fn test_noise_band_q_factor() {
        let nb = NoiseBandTimbre::new("test", 1000.0, 200.0);
        let q = nb.q_factor();
        assert!((q - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_noise_band_rain_brightness() {
        let r = NoiseBandTimbre::rain();
        assert!(r.brightness() > 0.4);
    }

    #[test]
    fn test_subtractive_moog_bass() {
        let m = SubtractiveTimbre::moog_bass();
        assert_eq!(m.source, SourceWaveform::Sawtooth);
        assert!(m.brightness() < 0.5);
    }

    #[test]
    fn test_subtractive_render_sawtooth() {
        let s = SubtractiveTimbre::moog_bass();
        let buf = s.render_source_period(256);
        assert_eq!(buf.len(), 256);
        // Sawtooth: first sample ~ -1, last sample ~ +1
        assert!(buf[0] < -0.9);
        assert!(buf[buf.len() - 1] > 0.9);
    }

    #[test]
    fn test_palette_8_size() {
        let p = TimbrePalette::palette_8();
        assert_eq!(p.len(), 8);
    }

    #[test]
    fn test_palette_16_size() {
        let p = TimbrePalette::palette_16();
        assert_eq!(p.len(), 16);
    }

    #[test]
    fn test_palette_select_n() {
        let p = TimbrePalette::palette_8();
        let selected = p.select_n(4);
        assert_eq!(selected.len(), 4);
    }

    #[test]
    fn test_palette_min_distance_positive() {
        let p = TimbrePalette::palette_8();
        assert!(p.min_distance() > 0.0);
    }

    #[test]
    fn test_palette_sorted_by_brightness() {
        let p = TimbrePalette::palette_8();
        let sorted = p.sorted_by_brightness();
        for w in sorted.windows(2) {
            assert!(w[0].brightness <= w[1].brightness);
        }
    }

    #[test]
    fn test_timbre_interpolator_endpoints() {
        let a = TimbreDescriptor::new("a", 0.0, 0.0, 0.0);
        let b = TimbreDescriptor::new("b", 1.0, 1.0, 1.0);
        let interp = TimbreInterpolator::new(a.clone(), b.clone());
        let at0 = interp.interpolate(0.0);
        let at1 = interp.interpolate(1.0);
        assert!((at0.brightness - 0.0).abs() < 1e-6);
        assert!((at1.brightness - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_timbre_interpolator_midpoint() {
        let a = TimbreDescriptor::new("a", 0.0, 0.0, 0.0);
        let b = TimbreDescriptor::new("b", 1.0, 1.0, 1.0);
        let interp = TimbreInterpolator::new(a, b);
        let mid = interp.interpolate(0.5);
        assert!((mid.brightness - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_interpolate_additive() {
        let a = AdditiveTimbre::flute();
        let b = AdditiveTimbre::trumpet();
        let mid = TimbreInterpolator::interpolate_additive(&a, &b, 0.5);
        assert!(!mid.partials.is_empty());
    }

    #[test]
    fn test_crossfade_palettes() {
        let a = TimbrePalette::palette_8();
        let b = TimbrePalette::palette_16();
        let xf = TimbreInterpolator::crossfade_palettes(&a, &b, 0.5);
        assert_eq!(xf.len(), 16);
    }

    #[test]
    fn test_timbre_descriptor_distance() {
        let a = TimbreDescriptor::new("a", 0.0, 0.0, 0.0);
        let b = TimbreDescriptor::new("b", 1.0, 1.0, 1.0);
        let d = a.distance(&b);
        assert!((d - 3.0_f64.sqrt()).abs() < 1e-6);
    }

    #[test]
    fn test_palette_get_by_name() {
        let p = TimbrePalette::palette_8();
        assert!(p.get("flute").is_some());
        assert!(p.get("nonexistent").is_none());
    }

    #[test]
    fn test_interpolate_n() {
        let a = TimbreDescriptor::new("a", 0.0, 0.0, 0.0);
        let b = TimbreDescriptor::new("b", 1.0, 1.0, 1.0);
        let interp = TimbreInterpolator::new(a, b);
        let steps = interp.interpolate_n(5);
        assert_eq!(steps.len(), 5);
        assert!((steps[0].brightness - 0.0).abs() < 1e-6);
        assert!((steps[4].brightness - 1.0).abs() < 1e-6);
    }
}
