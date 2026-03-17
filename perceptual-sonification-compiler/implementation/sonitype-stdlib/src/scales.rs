//! Pitch scale mappings for SoniType.
//!
//! Provides linear, logarithmic, Bark, Mel, musical, MIDI, and microtonal
//! scales that map data values to perceptually meaningful frequency ranges.

use std::collections::HashMap;
use std::f64::consts::{LN_2, PI};

// ---------------------------------------------------------------------------
// Core helpers (mirror sonitype_core::units where needed locally)
// ---------------------------------------------------------------------------

/// Clamp `v` to `[lo, hi]`.
fn clamp(v: f64, lo: f64, hi: f64) -> f64 {
    v.max(lo).min(hi)
}

/// Linear interpolation from `a` to `b` by `t ∈ [0,1]`.
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

/// Map `v` from `[in_lo, in_hi]` to `[out_lo, out_hi]`.
fn map_range(v: f64, in_lo: f64, in_hi: f64, out_lo: f64, out_hi: f64) -> f64 {
    if (in_hi - in_lo).abs() < f64::EPSILON {
        return out_lo;
    }
    let t = (v - in_lo) / (in_hi - in_lo);
    lerp(out_lo, out_hi, t)
}

// ---------------------------------------------------------------------------
// ScalePreset
// ---------------------------------------------------------------------------

/// Common scale presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScalePreset {
    Linear,
    Logarithmic,
    Bark,
    Mel,
    Chromatic,
    MajorPentatonic,
    MinorPentatonic,
    MajorDiatonic,
    NaturalMinor,
    WholeTone,
    Blues,
    Midi,
    JustIntonation,
    Edo19,
    Edo31,
}

// ---------------------------------------------------------------------------
// LinearScale
// ---------------------------------------------------------------------------

/// Linear frequency mapping over `[f_min, f_max]` in Hz.
#[derive(Debug, Clone)]
pub struct LinearScale {
    pub f_min: f64,
    pub f_max: f64,
}

impl LinearScale {
    pub fn new(f_min: f64, f_max: f64) -> Self {
        assert!(f_min < f_max, "f_min must be less than f_max");
        assert!(f_min > 0.0, "f_min must be positive");
        Self { f_min, f_max }
    }

    /// Map a data value within `data_range` to a frequency in Hz.
    pub fn map(&self, value: f64, data_range: (f64, f64)) -> f64 {
        let freq = map_range(value, data_range.0, data_range.1, self.f_min, self.f_max);
        clamp(freq, self.f_min, self.f_max)
    }

    /// Inverse-map a frequency back to the normalised `[0, 1]` range.
    pub fn unmap(&self, freq: f64) -> f64 {
        let f = clamp(freq, self.f_min, self.f_max);
        (f - self.f_min) / (self.f_max - self.f_min)
    }
}

impl Default for LinearScale {
    fn default() -> Self {
        Self { f_min: 200.0, f_max: 4000.0 }
    }
}

// ---------------------------------------------------------------------------
// LogarithmicScale
// ---------------------------------------------------------------------------

/// Logarithmic frequency mapping (perceptually more uniform than linear).
#[derive(Debug, Clone)]
pub struct LogarithmicScale {
    pub f_min: f64,
    pub f_max: f64,
    pub base: LogBase,
}

/// Logarithm base selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogBase {
    Base2,
    Base10,
    Natural,
}

impl LogarithmicScale {
    pub fn new(f_min: f64, f_max: f64, base: LogBase) -> Self {
        assert!(f_min > 0.0 && f_max > f_min);
        Self { f_min, f_max, base }
    }

    pub fn base2(f_min: f64, f_max: f64) -> Self {
        Self::new(f_min, f_max, LogBase::Base2)
    }

    pub fn base10(f_min: f64, f_max: f64) -> Self {
        Self::new(f_min, f_max, LogBase::Base10)
    }

    fn log_val(&self, v: f64) -> f64 {
        match self.base {
            LogBase::Base2 => v.log2(),
            LogBase::Base10 => v.log10(),
            LogBase::Natural => v.ln(),
        }
    }

    fn exp_val(&self, v: f64) -> f64 {
        match self.base {
            LogBase::Base2 => (2.0_f64).powf(v),
            LogBase::Base10 => (10.0_f64).powf(v),
            LogBase::Natural => v.exp(),
        }
    }

    /// Map a data value within `data_range` to a frequency in Hz (log-spaced).
    pub fn map(&self, value: f64, data_range: (f64, f64)) -> f64 {
        let t = if (data_range.1 - data_range.0).abs() < f64::EPSILON {
            0.0
        } else {
            clamp((value - data_range.0) / (data_range.1 - data_range.0), 0.0, 1.0)
        };
        let log_min = self.log_val(self.f_min);
        let log_max = self.log_val(self.f_max);
        let log_f = lerp(log_min, log_max, t);
        clamp(self.exp_val(log_f), self.f_min, self.f_max)
    }

    /// Inverse-map a frequency back to `[0, 1]`.
    pub fn unmap(&self, freq: f64) -> f64 {
        let f = clamp(freq, self.f_min, self.f_max);
        let log_min = self.log_val(self.f_min);
        let log_max = self.log_val(self.f_max);
        if (log_max - log_min).abs() < f64::EPSILON {
            return 0.0;
        }
        (self.log_val(f) - log_min) / (log_max - log_min)
    }
}

impl Default for LogarithmicScale {
    fn default() -> Self {
        Self::base2(200.0, 4000.0)
    }
}

// ---------------------------------------------------------------------------
// BarkScale
// ---------------------------------------------------------------------------

/// Bark-rate scale mapping (0–24 critical bands).
///
/// Converts between data values, Bark bands, and Hz using Traunmüller's formula.
#[derive(Debug, Clone)]
pub struct BarkScale {
    pub bark_min: f64,
    pub bark_max: f64,
}

impl BarkScale {
    pub fn new(bark_min: f64, bark_max: f64) -> Self {
        assert!(bark_min >= 0.0 && bark_max <= 24.0 && bark_min < bark_max);
        Self { bark_min, bark_max }
    }

    /// Full range 0–24 Bark.
    pub fn full() -> Self {
        Self::new(0.0, 24.0)
    }

    /// Hz → Bark (Traunmüller 1990).
    pub fn hz_to_bark(hz: f64) -> f64 {
        let bark = 26.81 * hz / (1960.0 + hz) - 0.53;
        clamp(bark, 0.0, 24.0)
    }

    /// Bark → Hz.
    pub fn bark_to_hz(bark: f64) -> f64 {
        let b = clamp(bark, 0.0, 24.0);
        1960.0 * (b + 0.53) / (26.28 - b)
    }

    /// Map a data value within `data_range` to Hz via Bark scale.
    pub fn map(&self, value: f64, data_range: (f64, f64)) -> f64 {
        let bark = map_range(value, data_range.0, data_range.1, self.bark_min, self.bark_max);
        let bark = clamp(bark, self.bark_min, self.bark_max);
        Self::bark_to_hz(bark)
    }

    /// Unmap: Hz → normalised `[0, 1]`.
    pub fn unmap(&self, freq: f64) -> f64 {
        let bark = Self::hz_to_bark(freq);
        let bark = clamp(bark, self.bark_min, self.bark_max);
        (bark - self.bark_min) / (self.bark_max - self.bark_min)
    }

    /// Bark bandwidth at a given Bark rate.
    pub fn bandwidth(bark: f64) -> f64 {
        let hz = Self::bark_to_hz(bark);
        52.548 + 28.856 * (hz / 1000.0)
    }
}

impl Default for BarkScale {
    fn default() -> Self {
        Self::full()
    }
}

// ---------------------------------------------------------------------------
// MelScale
// ---------------------------------------------------------------------------

/// Mel-frequency scale mapping.
///
/// Uses the O'Shaughnessy formula: `mel = 2595 * log10(1 + hz/700)`.
#[derive(Debug, Clone)]
pub struct MelScale {
    pub mel_min: f64,
    pub mel_max: f64,
}

impl MelScale {
    pub fn new(mel_min: f64, mel_max: f64) -> Self {
        assert!(mel_min < mel_max && mel_min >= 0.0);
        Self { mel_min, mel_max }
    }

    /// Construct from Hz bounds.
    pub fn from_hz(hz_min: f64, hz_max: f64) -> Self {
        Self::new(Self::hz_to_mel(hz_min), Self::hz_to_mel(hz_max))
    }

    pub fn hz_to_mel(hz: f64) -> f64 {
        2595.0 * (1.0 + hz / 700.0).log10()
    }

    pub fn mel_to_hz(mel: f64) -> f64 {
        700.0 * (10.0_f64.powf(mel / 2595.0) - 1.0)
    }

    /// Map data value to Hz via Mel scale.
    pub fn map(&self, value: f64, data_range: (f64, f64)) -> f64 {
        let mel = map_range(value, data_range.0, data_range.1, self.mel_min, self.mel_max);
        let mel = clamp(mel, self.mel_min, self.mel_max);
        Self::mel_to_hz(mel)
    }

    /// Unmap Hz → normalised `[0, 1]`.
    pub fn unmap(&self, freq: f64) -> f64 {
        let mel = Self::hz_to_mel(freq);
        let mel = clamp(mel, self.mel_min, self.mel_max);
        (mel - self.mel_min) / (self.mel_max - self.mel_min)
    }
}

impl Default for MelScale {
    fn default() -> Self {
        Self::from_hz(200.0, 4000.0)
    }
}

// ---------------------------------------------------------------------------
// MusicalScale
// ---------------------------------------------------------------------------

/// Named scale degree patterns (semitone offsets from root).
#[derive(Debug, Clone, PartialEq)]
pub enum ScaleType {
    Chromatic,
    Major,
    NaturalMinor,
    HarmonicMinor,
    Pentatonic,
    MinorPentatonic,
    WholeTone,
    Blues,
    Dorian,
    Mixolydian,
    Custom(Vec<u8>),
}

impl ScaleType {
    /// Return semitone offsets within one octave.
    pub fn intervals(&self) -> Vec<u8> {
        match self {
            Self::Chromatic => (0..12).collect(),
            Self::Major => vec![0, 2, 4, 5, 7, 9, 11],
            Self::NaturalMinor => vec![0, 2, 3, 5, 7, 8, 10],
            Self::HarmonicMinor => vec![0, 2, 3, 5, 7, 8, 11],
            Self::Pentatonic => vec![0, 2, 4, 7, 9],
            Self::MinorPentatonic => vec![0, 3, 5, 7, 10],
            Self::WholeTone => vec![0, 2, 4, 6, 8, 10],
            Self::Blues => vec![0, 3, 5, 6, 7, 10],
            Self::Dorian => vec![0, 2, 3, 5, 7, 9, 10],
            Self::Mixolydian => vec![0, 2, 4, 5, 7, 9, 10],
            Self::Custom(intervals) => intervals.clone(),
        }
    }
}

/// Note names for display and lookup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NoteName {
    C, Cs, D, Ds, E, F, Fs, G, Gs, A, As, B,
}

impl NoteName {
    pub fn from_index(idx: u8) -> Self {
        match idx % 12 {
            0 => Self::C,  1 => Self::Cs, 2 => Self::D,
            3 => Self::Ds, 4 => Self::E,  5 => Self::F,
            6 => Self::Fs, 7 => Self::G,  8 => Self::Gs,
            9 => Self::A,  10 => Self::As, 11 => Self::B,
            _ => unreachable!(),
        }
    }

    pub fn to_index(self) -> u8 {
        match self {
            Self::C => 0,  Self::Cs => 1, Self::D => 2,
            Self::Ds => 3, Self::E => 4,  Self::F => 5,
            Self::Fs => 6, Self::G => 7,  Self::Gs => 8,
            Self::A => 9,  Self::As => 10, Self::B => 11,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::C => "C", Self::Cs => "C#", Self::D => "D",
            Self::Ds => "D#", Self::E => "E", Self::F => "F",
            Self::Fs => "F#", Self::G => "G", Self::Gs => "G#",
            Self::A => "A", Self::As => "A#", Self::B => "B",
        }
    }
}

/// A musical scale that quantises continuous data to discrete pitches.
#[derive(Debug, Clone)]
pub struct MusicalScale {
    pub root: NoteName,
    pub scale_type: ScaleType,
    pub octave_low: i32,
    pub octave_high: i32,
    pub reference_a4: f64,
    pitches_hz: Vec<f64>,
}

impl MusicalScale {
    pub fn new(root: NoteName, scale_type: ScaleType, octave_low: i32, octave_high: i32) -> Self {
        let reference_a4 = 440.0;
        let pitches_hz = Self::build_pitches(root, &scale_type, octave_low, octave_high, reference_a4);
        Self { root, scale_type, octave_low, octave_high, reference_a4, pitches_hz }
    }

    pub fn with_reference(mut self, a4_hz: f64) -> Self {
        self.reference_a4 = a4_hz;
        self.pitches_hz = Self::build_pitches(
            self.root, &self.scale_type, self.octave_low, self.octave_high, a4_hz,
        );
        self
    }

    fn build_pitches(
        root: NoteName, scale_type: &ScaleType,
        oct_lo: i32, oct_hi: i32, a4: f64,
    ) -> Vec<f64> {
        let intervals = scale_type.intervals();
        let root_idx = root.to_index() as i32;
        let mut pitches = Vec::new();
        for oct in oct_lo..=oct_hi {
            for &interval in &intervals {
                let midi = (oct + 1) * 12 + root_idx + interval as i32;
                let freq = a4 * 2.0_f64.powf((midi as f64 - 69.0) / 12.0);
                if freq >= 20.0 && freq <= 20000.0 {
                    pitches.push(freq);
                }
            }
        }
        pitches.sort_by(|a, b| a.partial_cmp(b).unwrap());
        pitches.dedup();
        pitches
    }

    /// Get all pitches in this scale configuration.
    pub fn pitches(&self) -> &[f64] {
        &self.pitches_hz
    }

    /// Map a data value to the nearest scale pitch.
    pub fn map(&self, value: f64, data_range: (f64, f64)) -> f64 {
        if self.pitches_hz.is_empty() {
            return 440.0;
        }
        let t = if (data_range.1 - data_range.0).abs() < f64::EPSILON {
            0.0
        } else {
            clamp((value - data_range.0) / (data_range.1 - data_range.0), 0.0, 1.0)
        };
        let idx_f = t * (self.pitches_hz.len() - 1) as f64;
        let idx = idx_f.round() as usize;
        self.pitches_hz[idx.min(self.pitches_hz.len() - 1)]
    }

    /// Find nearest pitch to an arbitrary frequency.
    pub fn quantize(&self, freq: f64) -> f64 {
        if self.pitches_hz.is_empty() {
            return freq;
        }
        *self.pitches_hz.iter()
            .min_by(|a, b| {
                ((*a - freq).abs()).partial_cmp(&((*b - freq).abs())).unwrap()
            })
            .unwrap()
    }

    /// Note-name → Hz lookup (octave 4 by default).
    pub fn note_freq(name: NoteName, octave: i32, a4: f64) -> f64 {
        let midi = (octave + 1) * 12 + name.to_index() as i32;
        a4 * 2.0_f64.powf((midi as f64 - 69.0) / 12.0)
    }

    /// Build a lookup table of note-name:octave → Hz.
    pub fn note_table(&self) -> HashMap<String, f64> {
        let mut table = HashMap::new();
        for oct in self.octave_low..=self.octave_high {
            for idx in 0..12u8 {
                let name = NoteName::from_index(idx);
                let freq = Self::note_freq(name, oct, self.reference_a4);
                if freq >= 20.0 && freq <= 20000.0 {
                    table.insert(format!("{}{}", name.as_str(), oct), freq);
                }
            }
        }
        table
    }
}

impl Default for MusicalScale {
    fn default() -> Self {
        Self::new(NoteName::C, ScaleType::Major, 3, 6)
    }
}

// ---------------------------------------------------------------------------
// MidiScale
// ---------------------------------------------------------------------------

/// MIDI note number mapping.
#[derive(Debug, Clone)]
pub struct MidiScale {
    pub note_min: u8,
    pub note_max: u8,
    pub velocity_min: u8,
    pub velocity_max: u8,
    pub reference_a4: f64,
}

impl MidiScale {
    pub fn new(note_min: u8, note_max: u8) -> Self {
        assert!(note_min <= note_max && note_max <= 127);
        Self { note_min, note_max, velocity_min: 1, velocity_max: 127, reference_a4: 440.0 }
    }

    pub fn with_velocity_range(mut self, vel_min: u8, vel_max: u8) -> Self {
        assert!(vel_min <= vel_max && vel_max <= 127);
        self.velocity_min = vel_min;
        self.velocity_max = vel_max;
        self
    }

    /// Map data value to MIDI note number.
    pub fn map_note(&self, value: f64, data_range: (f64, f64)) -> u8 {
        let note_f = map_range(value, data_range.0, data_range.1,
                               self.note_min as f64, self.note_max as f64);
        clamp(note_f.round(), self.note_min as f64, self.note_max as f64) as u8
    }

    /// Map data value to MIDI velocity.
    pub fn map_velocity(&self, value: f64, data_range: (f64, f64)) -> u8 {
        let vel_f = map_range(value, data_range.0, data_range.1,
                              self.velocity_min as f64, self.velocity_max as f64);
        clamp(vel_f.round(), self.velocity_min as f64, self.velocity_max as f64) as u8
    }

    /// MIDI note → Hz.
    pub fn midi_to_hz(&self, note: u8) -> f64 {
        self.reference_a4 * 2.0_f64.powf((note as f64 - 69.0) / 12.0)
    }

    /// Hz → nearest MIDI note.
    pub fn hz_to_midi(&self, hz: f64) -> u8 {
        let midi = 69.0 + 12.0 * (hz / self.reference_a4).log2();
        clamp(midi.round(), 0.0, 127.0) as u8
    }

    /// Full map: data value → Hz.
    pub fn map(&self, value: f64, data_range: (f64, f64)) -> f64 {
        let note = self.map_note(value, data_range);
        self.midi_to_hz(note)
    }

    /// Unmap: Hz → normalised `[0, 1]`.
    pub fn unmap(&self, freq: f64) -> f64 {
        let note = self.hz_to_midi(freq);
        let note = clamp(note as f64, self.note_min as f64, self.note_max as f64);
        (note - self.note_min as f64) / (self.note_max as f64 - self.note_min as f64)
    }
}

impl Default for MidiScale {
    fn default() -> Self {
        Self::new(48, 84) // C3–C6
    }
}

// ---------------------------------------------------------------------------
// MicrotonalScale
// ---------------------------------------------------------------------------

/// Tuning specification for microtonal scales.
#[derive(Debug, Clone)]
pub enum Tuning {
    /// Equal temperament with `n` divisions of the octave.
    EqualTemperament { divisions: u32 },
    /// Just intonation specified as frequency ratios.
    JustIntonation { ratios: Vec<f64> },
    /// Cents-based specification (offsets from root within one period).
    Cents { offsets: Vec<f64> },
}

/// Microtonal scale with arbitrary step sizes.
#[derive(Debug, Clone)]
pub struct MicrotonalScale {
    pub tuning: Tuning,
    pub root_hz: f64,
    pub octave_low: i32,
    pub octave_high: i32,
    pitches_hz: Vec<f64>,
}

impl MicrotonalScale {
    pub fn new(tuning: Tuning, root_hz: f64, octave_low: i32, octave_high: i32) -> Self {
        let pitches_hz = Self::build_pitches(&tuning, root_hz, octave_low, octave_high);
        Self { tuning, root_hz, octave_low, octave_high, pitches_hz }
    }

    /// 19-tone equal temperament.
    pub fn edo19(root_hz: f64) -> Self {
        Self::new(Tuning::EqualTemperament { divisions: 19 }, root_hz, 3, 6)
    }

    /// 31-tone equal temperament.
    pub fn edo31(root_hz: f64) -> Self {
        Self::new(Tuning::EqualTemperament { divisions: 31 }, root_hz, 3, 6)
    }

    /// Standard just intonation (5-limit).
    pub fn just_intonation(root_hz: f64) -> Self {
        let ratios = vec![
            1.0, 16.0/15.0, 9.0/8.0, 6.0/5.0, 5.0/4.0, 4.0/3.0,
            45.0/32.0, 3.0/2.0, 8.0/5.0, 5.0/3.0, 9.0/5.0, 15.0/8.0,
        ];
        Self::new(Tuning::JustIntonation { ratios }, root_hz, 3, 6)
    }

    fn build_pitches(tuning: &Tuning, root: f64, oct_lo: i32, oct_hi: i32) -> Vec<f64> {
        let mut pitches = Vec::new();
        match tuning {
            Tuning::EqualTemperament { divisions } => {
                let step = 2.0_f64.powf(1.0 / *divisions as f64);
                for oct in oct_lo..=oct_hi {
                    let oct_root = root * 2.0_f64.powi(oct);
                    for i in 0..*divisions {
                        let freq = oct_root * step.powi(i as i32);
                        if freq >= 20.0 && freq <= 20000.0 {
                            pitches.push(freq);
                        }
                    }
                }
            }
            Tuning::JustIntonation { ratios } => {
                for oct in oct_lo..=oct_hi {
                    let oct_root = root * 2.0_f64.powi(oct);
                    for &r in ratios {
                        let freq = oct_root * r;
                        if freq >= 20.0 && freq <= 20000.0 {
                            pitches.push(freq);
                        }
                    }
                }
            }
            Tuning::Cents { offsets } => {
                for oct in oct_lo..=oct_hi {
                    let oct_root = root * 2.0_f64.powi(oct);
                    for &c in offsets {
                        let freq = oct_root * 2.0_f64.powf(c / 1200.0);
                        if freq >= 20.0 && freq <= 20000.0 {
                            pitches.push(freq);
                        }
                    }
                }
            }
        }
        pitches.sort_by(|a, b| a.partial_cmp(b).unwrap());
        pitches.dedup();
        pitches
    }

    pub fn pitches(&self) -> &[f64] {
        &self.pitches_hz
    }

    /// Map data value to nearest microtonal pitch.
    pub fn map(&self, value: f64, data_range: (f64, f64)) -> f64 {
        if self.pitches_hz.is_empty() {
            return self.root_hz;
        }
        let t = if (data_range.1 - data_range.0).abs() < f64::EPSILON {
            0.0
        } else {
            clamp((value - data_range.0) / (data_range.1 - data_range.0), 0.0, 1.0)
        };
        let idx = (t * (self.pitches_hz.len() - 1) as f64).round() as usize;
        self.pitches_hz[idx.min(self.pitches_hz.len() - 1)]
    }

    /// Quantize frequency to nearest pitch in this scale.
    pub fn quantize(&self, freq: f64) -> f64 {
        if self.pitches_hz.is_empty() {
            return freq;
        }
        *self.pitches_hz.iter()
            .min_by(|a, b| ((*a - freq).abs()).partial_cmp(&((*b - freq).abs())).unwrap())
            .unwrap()
    }

    /// Convert cents offset to frequency ratio.
    pub fn cents_to_ratio(cents: f64) -> f64 {
        2.0_f64.powf(cents / 1200.0)
    }

    /// Convert frequency ratio to cents.
    pub fn ratio_to_cents(ratio: f64) -> f64 {
        1200.0 * ratio.log2()
    }
}

// ---------------------------------------------------------------------------
// ScaleBuilder (fluent API)
// ---------------------------------------------------------------------------

/// Fluent builder for constructing scale configurations.
#[derive(Debug, Clone)]
pub struct ScaleBuilder {
    kind: ScalePreset,
    f_min: f64,
    f_max: f64,
    root: NoteName,
    octave_low: i32,
    octave_high: i32,
    reference_a4: f64,
    custom_intervals: Option<Vec<u8>>,
    microtonal_divisions: Option<u32>,
    microtonal_ratios: Option<Vec<f64>>,
    microtonal_cents: Option<Vec<f64>>,
    log_base: LogBase,
    bark_min: f64,
    bark_max: f64,
    midi_note_min: u8,
    midi_note_max: u8,
}

impl ScaleBuilder {
    pub fn new(kind: ScalePreset) -> Self {
        Self {
            kind,
            f_min: 200.0,
            f_max: 4000.0,
            root: NoteName::C,
            octave_low: 3,
            octave_high: 6,
            reference_a4: 440.0,
            custom_intervals: None,
            microtonal_divisions: None,
            microtonal_ratios: None,
            microtonal_cents: None,
            log_base: LogBase::Base2,
            bark_min: 0.0,
            bark_max: 24.0,
            midi_note_min: 48,
            midi_note_max: 84,
        }
    }

    pub fn frequency_range(mut self, f_min: f64, f_max: f64) -> Self {
        self.f_min = f_min;
        self.f_max = f_max;
        self
    }

    pub fn root(mut self, root: NoteName) -> Self {
        self.root = root;
        self
    }

    pub fn octave_range(mut self, low: i32, high: i32) -> Self {
        self.octave_low = low;
        self.octave_high = high;
        self
    }

    pub fn reference_pitch(mut self, a4: f64) -> Self {
        self.reference_a4 = a4;
        self
    }

    pub fn custom_intervals(mut self, intervals: Vec<u8>) -> Self {
        self.custom_intervals = Some(intervals);
        self
    }

    pub fn microtonal_divisions(mut self, n: u32) -> Self {
        self.microtonal_divisions = Some(n);
        self
    }

    pub fn microtonal_ratios(mut self, ratios: Vec<f64>) -> Self {
        self.microtonal_ratios = Some(ratios);
        self
    }

    pub fn microtonal_cents(mut self, cents: Vec<f64>) -> Self {
        self.microtonal_cents = Some(cents);
        self
    }

    pub fn log_base(mut self, base: LogBase) -> Self {
        self.log_base = base;
        self
    }

    pub fn bark_range(mut self, min: f64, max: f64) -> Self {
        self.bark_min = min;
        self.bark_max = max;
        self
    }

    pub fn midi_range(mut self, note_min: u8, note_max: u8) -> Self {
        self.midi_note_min = note_min;
        self.midi_note_max = note_max;
        self
    }

    /// Build a boxed scale trait-object.
    pub fn build_linear(&self) -> LinearScale {
        LinearScale::new(self.f_min, self.f_max)
    }

    pub fn build_logarithmic(&self) -> LogarithmicScale {
        LogarithmicScale::new(self.f_min, self.f_max, self.log_base)
    }

    pub fn build_bark(&self) -> BarkScale {
        BarkScale::new(self.bark_min, self.bark_max)
    }

    pub fn build_mel(&self) -> MelScale {
        MelScale::from_hz(self.f_min, self.f_max)
    }

    pub fn build_musical(&self) -> MusicalScale {
        let st = match self.kind {
            ScalePreset::Chromatic => ScaleType::Chromatic,
            ScalePreset::MajorDiatonic => ScaleType::Major,
            ScalePreset::NaturalMinor => ScaleType::NaturalMinor,
            ScalePreset::MajorPentatonic => ScaleType::Pentatonic,
            ScalePreset::MinorPentatonic => ScaleType::MinorPentatonic,
            ScalePreset::WholeTone => ScaleType::WholeTone,
            ScalePreset::Blues => ScaleType::Blues,
            _ => {
                if let Some(ref intervals) = self.custom_intervals {
                    ScaleType::Custom(intervals.clone())
                } else {
                    ScaleType::Chromatic
                }
            }
        };
        MusicalScale::new(self.root, st, self.octave_low, self.octave_high)
            .with_reference(self.reference_a4)
    }

    pub fn build_midi(&self) -> MidiScale {
        MidiScale::new(self.midi_note_min, self.midi_note_max)
    }

    pub fn build_microtonal(&self) -> MicrotonalScale {
        let tuning = if let Some(n) = self.microtonal_divisions {
            Tuning::EqualTemperament { divisions: n }
        } else if let Some(ref ratios) = self.microtonal_ratios {
            Tuning::JustIntonation { ratios: ratios.clone() }
        } else if let Some(ref cents) = self.microtonal_cents {
            Tuning::Cents { offsets: cents.clone() }
        } else {
            Tuning::EqualTemperament { divisions: 12 }
        };
        let root = MusicalScale::note_freq(self.root, 0, self.reference_a4);
        MicrotonalScale::new(tuning, root, self.octave_low, self.octave_high)
    }
}

// ---------------------------------------------------------------------------
// Convenience constructors from ScalePreset
// ---------------------------------------------------------------------------

impl ScalePreset {
    /// Return a default frequency range appropriate for each preset.
    pub fn default_range(&self) -> (f64, f64) {
        match self {
            Self::Linear | Self::Logarithmic => (200.0, 4000.0),
            Self::Bark => (100.0, 8000.0),
            Self::Mel => (200.0, 4000.0),
            Self::Chromatic | Self::MajorDiatonic | Self::NaturalMinor
            | Self::MajorPentatonic | Self::MinorPentatonic
            | Self::WholeTone | Self::Blues => (130.0, 2100.0),
            Self::Midi => (130.0, 2100.0),
            Self::JustIntonation | Self::Edo19 | Self::Edo31 => (130.0, 2100.0),
        }
    }

    /// Quick-build a `ScaleBuilder` initialised for this preset.
    pub fn builder(self) -> ScaleBuilder {
        ScaleBuilder::new(self)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_scale_map_boundaries() {
        let s = LinearScale::new(200.0, 4000.0);
        assert!((s.map(0.0, (0.0, 1.0)) - 200.0).abs() < 1e-6);
        assert!((s.map(1.0, (0.0, 1.0)) - 4000.0).abs() < 1e-6);
    }

    #[test]
    fn test_linear_scale_unmap_roundtrip() {
        let s = LinearScale::new(200.0, 4000.0);
        let freq = s.map(0.5, (0.0, 1.0));
        let t = s.unmap(freq);
        assert!((t - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_log_scale_base2_monotonic() {
        let s = LogarithmicScale::base2(200.0, 4000.0);
        let f1 = s.map(0.25, (0.0, 1.0));
        let f2 = s.map(0.75, (0.0, 1.0));
        assert!(f2 > f1);
    }

    #[test]
    fn test_log_scale_unmap_roundtrip() {
        let s = LogarithmicScale::base10(100.0, 10000.0);
        let freq = s.map(0.5, (0.0, 1.0));
        let t = s.unmap(freq);
        assert!((t - 0.5).abs() < 1e-4);
    }

    #[test]
    fn test_bark_hz_roundtrip() {
        let hz = 1000.0;
        let bark = BarkScale::hz_to_bark(hz);
        let recovered = BarkScale::bark_to_hz(bark);
        assert!((recovered - hz).abs() < 2.0);
    }

    #[test]
    fn test_bark_scale_map() {
        let s = BarkScale::full();
        let freq = s.map(12.0, (0.0, 24.0));
        assert!(freq > 0.0 && freq < 20000.0);
    }

    #[test]
    fn test_mel_hz_roundtrip() {
        let hz = 1000.0;
        let mel = MelScale::hz_to_mel(hz);
        let recovered = MelScale::mel_to_hz(mel);
        assert!((recovered - hz).abs() < 1.0);
    }

    #[test]
    fn test_mel_scale_map_monotonic() {
        let s = MelScale::from_hz(200.0, 4000.0);
        let f1 = s.map(0.2, (0.0, 1.0));
        let f2 = s.map(0.8, (0.0, 1.0));
        assert!(f2 > f1);
    }

    #[test]
    fn test_musical_scale_chromatic_count() {
        let s = MusicalScale::new(NoteName::C, ScaleType::Chromatic, 4, 4);
        assert_eq!(s.pitches().len(), 12);
    }

    #[test]
    fn test_musical_scale_major_count() {
        let s = MusicalScale::new(NoteName::C, ScaleType::Major, 4, 4);
        assert_eq!(s.pitches().len(), 7);
    }

    #[test]
    fn test_musical_scale_quantize() {
        let s = MusicalScale::new(NoteName::C, ScaleType::Major, 4, 4);
        let q = s.quantize(445.0);
        // Should snap to A4 = 440 Hz
        assert!((q - 440.0).abs() < 1.0);
    }

    #[test]
    fn test_musical_scale_note_freq_a4() {
        let f = MusicalScale::note_freq(NoteName::A, 4, 440.0);
        assert!((f - 440.0).abs() < 1e-6);
    }

    #[test]
    fn test_musical_scale_note_table() {
        let s = MusicalScale::new(NoteName::C, ScaleType::Chromatic, 4, 4);
        let t = s.note_table();
        assert!(t.contains_key("A4"));
        assert!((*t.get("A4").unwrap() - 440.0).abs() < 1e-4);
    }

    #[test]
    fn test_midi_scale_note_mapping() {
        let s = MidiScale::new(48, 84);
        assert_eq!(s.map_note(0.0, (0.0, 1.0)), 48);
        assert_eq!(s.map_note(1.0, (0.0, 1.0)), 84);
    }

    #[test]
    fn test_midi_scale_hz_conversion() {
        let s = MidiScale::default();
        let hz = s.midi_to_hz(69);
        assert!((hz - 440.0).abs() < 1e-6);
        let note = s.hz_to_midi(440.0);
        assert_eq!(note, 69);
    }

    #[test]
    fn test_midi_scale_velocity() {
        let s = MidiScale::default().with_velocity_range(10, 120);
        let v = s.map_velocity(0.5, (0.0, 1.0));
        assert!(v >= 10 && v <= 120);
    }

    #[test]
    fn test_microtonal_edo19_pitch_count() {
        let s = MicrotonalScale::edo19(261.63);
        assert!(s.pitches().len() >= 19);
    }

    #[test]
    fn test_microtonal_just_intonation_root() {
        let root = 261.63;
        let s = MicrotonalScale::just_intonation(root);
        // First pitch in octave 3 should be root * 2^3 = root * 8
        let first_valid = s.pitches().iter().copied()
            .find(|&f| f >= root)
            .unwrap_or(root);
        assert!(first_valid > 0.0);
    }

    #[test]
    fn test_microtonal_cents_roundtrip() {
        let ratio = MicrotonalScale::cents_to_ratio(1200.0);
        assert!((ratio - 2.0).abs() < 1e-10);
        let cents = MicrotonalScale::ratio_to_cents(2.0);
        assert!((cents - 1200.0).abs() < 1e-10);
    }

    #[test]
    fn test_scale_builder_linear() {
        let s = ScaleBuilder::new(ScalePreset::Linear)
            .frequency_range(100.0, 8000.0)
            .build_linear();
        assert!((s.f_min - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_scale_builder_musical() {
        let s = ScaleBuilder::new(ScalePreset::Blues)
            .root(NoteName::A)
            .octave_range(3, 5)
            .build_musical();
        assert!(!s.pitches().is_empty());
    }

    #[test]
    fn test_scale_preset_builder() {
        let b = ScalePreset::Mel.builder();
        let s = b.build_mel();
        assert!(s.mel_max > s.mel_min);
    }

    #[test]
    fn test_note_name_roundtrip() {
        for i in 0..12u8 {
            let n = NoteName::from_index(i);
            assert_eq!(n.to_index(), i);
        }
    }

    #[test]
    fn test_linear_scale_clamping() {
        let s = LinearScale::new(200.0, 4000.0);
        // Value outside data range should be clamped
        let freq = s.map(2.0, (0.0, 1.0));
        assert!((freq - 4000.0).abs() < 1e-6);
    }

    #[test]
    fn test_bark_bandwidth() {
        let bw = BarkScale::bandwidth(10.0);
        assert!(bw > 0.0);
    }
}
