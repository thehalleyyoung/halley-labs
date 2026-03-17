//! Core domain types for the SoniType perceptual sonification compiler.
//!
//! Defines newtypes for physical and psychoacoustic quantities, data value
//! representations, distribution statistics, schema descriptors, stream
//! descriptors, mapping parameters, perceptual qualifiers, and cognitive
//! load budgets.

use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ===========================================================================
// Physical / acoustic newtypes
// ===========================================================================

/// Frequency in Hertz (Hz). Must be non-negative.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Frequency(pub f64);

impl Frequency {
    pub fn new(hz: f64) -> Self {
        assert!(hz >= 0.0, "Frequency must be non-negative");
        Self(hz)
    }

    pub fn hz(self) -> f64 {
        self.0
    }

    /// Human audible range lower bound.
    pub const AUDIBLE_MIN: Frequency = Frequency(20.0);
    /// Human audible range upper bound.
    pub const AUDIBLE_MAX: Frequency = Frequency(20_000.0);
    /// Concert pitch A4.
    pub const A4: Frequency = Frequency(440.0);

    /// Check whether this frequency is within the nominal audible range.
    pub fn is_audible(self) -> bool {
        self.0 >= Self::AUDIBLE_MIN.0 && self.0 <= Self::AUDIBLE_MAX.0
    }
}

impl fmt::Display for Frequency {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.2} Hz", self.0)
    }
}

impl Eq for Frequency {}
impl std::hash::Hash for Frequency {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        OrderedFloat(self.0).hash(state);
    }
}

impl Ord for Frequency {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        OrderedFloat(self.0).cmp(&OrderedFloat(other.0))
    }
}

// ---------------------------------------------------------------------------

/// Linear amplitude in the range [0.0, 1.0].
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Amplitude(pub f64);

impl Amplitude {
    pub fn new(val: f64) -> Self {
        Self(val.clamp(0.0, 1.0))
    }

    pub fn value(self) -> f64 {
        self.0
    }

    pub const SILENT: Amplitude = Amplitude(0.0);
    pub const FULL: Amplitude = Amplitude(1.0);
}

impl fmt::Display for Amplitude {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.4}", self.0)
    }
}

impl Eq for Amplitude {}
impl Ord for Amplitude {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        OrderedFloat(self.0).cmp(&OrderedFloat(other.0))
    }
}

// ---------------------------------------------------------------------------

/// Duration in seconds.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Duration(pub f64);

impl Duration {
    pub fn new(seconds: f64) -> Self {
        assert!(seconds >= 0.0, "Duration must be non-negative");
        Self(seconds)
    }

    pub fn from_millis(ms: f64) -> Self {
        Self(ms / 1000.0)
    }

    pub fn seconds(self) -> f64 {
        self.0
    }

    pub fn millis(self) -> f64 {
        self.0 * 1000.0
    }

    pub fn samples(self, sample_rate: u32) -> usize {
        (self.0 * sample_rate as f64).round() as usize
    }
}

impl fmt::Display for Duration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0 < 1.0 {
            write!(f, "{:.2} ms", self.0 * 1000.0)
        } else {
            write!(f, "{:.3} s", self.0)
        }
    }
}

impl Eq for Duration {}
impl Ord for Duration {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        OrderedFloat(self.0).cmp(&OrderedFloat(other.0))
    }
}

// ---------------------------------------------------------------------------

/// Phase angle in radians, normalized to [0, 2*pi).
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Phase(pub f64);

impl Phase {
    pub fn new(radians: f64) -> Self {
        let two_pi = 2.0 * std::f64::consts::PI;
        let mut r = radians % two_pi;
        if r < 0.0 {
            r += two_pi;
        }
        Self(r)
    }

    pub fn radians(self) -> f64 {
        self.0
    }

    pub fn degrees(self) -> f64 {
        self.0.to_degrees()
    }

    pub fn from_degrees(deg: f64) -> Self {
        Self::new(deg.to_radians())
    }
}

impl fmt::Display for Phase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.4} rad", self.0)
    }
}

impl Eq for Phase {}
impl Ord for Phase {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        OrderedFloat(self.0).cmp(&OrderedFloat(other.0))
    }
}

// ---------------------------------------------------------------------------

/// Audio sample rate in samples per second.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SampleRate(pub u32);

impl SampleRate {
    pub fn new(rate: u32) -> Self {
        assert!(rate > 0, "Sample rate must be positive");
        Self(rate)
    }

    pub fn value(self) -> u32 {
        self.0
    }

    /// Nyquist frequency for this sample rate.
    pub fn nyquist(self) -> Frequency {
        Frequency(self.0 as f64 / 2.0)
    }

    pub const CD_QUALITY: SampleRate = SampleRate(44100);
    pub const DVD_QUALITY: SampleRate = SampleRate(48000);
    pub const HIGH_RES: SampleRate = SampleRate(96000);
}

impl fmt::Display for SampleRate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} Hz", self.0)
    }
}

// ---------------------------------------------------------------------------

/// Audio buffer size in samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct BufferSize(pub usize);

impl BufferSize {
    pub fn new(size: usize) -> Self {
        assert!(size > 0, "Buffer size must be positive");
        Self(size)
    }

    pub fn value(self) -> usize {
        self.0
    }

    /// Duration of one buffer at the given sample rate.
    pub fn duration(self, sample_rate: SampleRate) -> Duration {
        Duration(self.0 as f64 / sample_rate.0 as f64)
    }

    /// Check if the size is a power of two (common requirement for FFT).
    pub fn is_power_of_two(self) -> bool {
        self.0.is_power_of_two()
    }
}

impl fmt::Display for BufferSize {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} samples", self.0)
    }
}

// ===========================================================================
// Psychoacoustic scale types
// ===========================================================================

/// Critical band index on the Bark scale (0-23 for 24 bands).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct BarkBand(pub u8);

impl BarkBand {
    pub fn new(band: u8) -> Self {
        assert!(band < 24, "Bark band must be in 0..23");
        Self(band)
    }

    pub fn value(self) -> u8 {
        self.0
    }

    /// Approximate centre frequency of this critical band.
    pub fn center_frequency(self) -> Frequency {
        let centers: [f64; 24] = [
            50.0, 150.0, 250.0, 350.0, 450.0, 570.0, 700.0, 840.0, 1000.0, 1170.0, 1370.0,
            1600.0, 1850.0, 2150.0, 2500.0, 2900.0, 3400.0, 4000.0, 4800.0, 5800.0, 7000.0,
            8500.0, 10500.0, 13500.0,
        ];
        Frequency(centers[self.0 as usize])
    }
}

impl fmt::Display for BarkBand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Bark {}", self.0)
    }
}

// ---------------------------------------------------------------------------

/// Mel-scale band index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct MelBand(pub u16);

impl MelBand {
    pub fn new(band: u16) -> Self {
        Self(band)
    }

    pub fn value(self) -> u16 {
        self.0
    }
}

impl fmt::Display for MelBand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Mel band {}", self.0)
    }
}

// ---------------------------------------------------------------------------

/// Sound pressure level in decibels (dB SPL).
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct DecibelSpl(pub f64);

impl DecibelSpl {
    pub fn new(db: f64) -> Self {
        Self(db)
    }

    pub fn value(self) -> f64 {
        self.0
    }

    pub const THRESHOLD_OF_HEARING: DecibelSpl = DecibelSpl(0.0);
    pub const CONVERSATIONAL_SPEECH: DecibelSpl = DecibelSpl(60.0);
    pub const PAIN_THRESHOLD: DecibelSpl = DecibelSpl(130.0);
}

impl fmt::Display for DecibelSpl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.1} dB SPL", self.0)
    }
}

impl Eq for DecibelSpl {}
impl Ord for DecibelSpl {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        OrderedFloat(self.0).cmp(&OrderedFloat(other.0))
    }
}

// ===========================================================================
// Musical types
// ===========================================================================

/// Chromatic pitch class (C through B).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum PitchClass {
    C, CSharp, D, DSharp, E, F, FSharp, G, GSharp, A, ASharp, B,
}

impl PitchClass {
    pub const ALL: [PitchClass; 12] = [
        PitchClass::C, PitchClass::CSharp, PitchClass::D, PitchClass::DSharp,
        PitchClass::E, PitchClass::F, PitchClass::FSharp, PitchClass::G,
        PitchClass::GSharp, PitchClass::A, PitchClass::ASharp, PitchClass::B,
    ];

    pub fn semitone(self) -> u8 {
        match self {
            PitchClass::C => 0, PitchClass::CSharp => 1, PitchClass::D => 2,
            PitchClass::DSharp => 3, PitchClass::E => 4, PitchClass::F => 5,
            PitchClass::FSharp => 6, PitchClass::G => 7, PitchClass::GSharp => 8,
            PitchClass::A => 9, PitchClass::ASharp => 10, PitchClass::B => 11,
        }
    }

    pub fn from_semitone(s: u8) -> Self {
        Self::ALL[(s % 12) as usize]
    }

    pub fn midi_note(self, octave: Octave) -> u8 {
        (octave.0 as i16 + 1) as u8 * 12 + self.semitone()
    }

    pub fn frequency(self, octave: Octave) -> Frequency {
        let midi = self.midi_note(octave) as f64;
        Frequency(440.0 * 2.0_f64.powf((midi - 69.0) / 12.0))
    }
}

impl fmt::Display for PitchClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            PitchClass::C => "C", PitchClass::CSharp => "C#", PitchClass::D => "D",
            PitchClass::DSharp => "D#", PitchClass::E => "E", PitchClass::F => "F",
            PitchClass::FSharp => "F#", PitchClass::G => "G", PitchClass::GSharp => "G#",
            PitchClass::A => "A", PitchClass::ASharp => "A#", PitchClass::B => "B",
        };
        write!(f, "{s}")
    }
}

// ---------------------------------------------------------------------------

/// Musical octave number (typically -1 to 10 in MIDI convention).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Octave(pub i8);

impl Octave {
    pub fn new(o: i8) -> Self { Self(o) }
    pub fn value(self) -> i8 { self.0 }
}

impl fmt::Display for Octave {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

// ---------------------------------------------------------------------------

/// A named musical note (pitch class + octave).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct MusicalNote {
    pub pitch_class: PitchClass,
    pub octave: Octave,
}

impl MusicalNote {
    pub fn new(pitch_class: PitchClass, octave: Octave) -> Self {
        Self { pitch_class, octave }
    }

    pub fn frequency(self) -> Frequency {
        self.pitch_class.frequency(self.octave)
    }

    pub fn midi_note(self) -> u8 {
        self.pitch_class.midi_note(self.octave)
    }

    pub fn from_midi(note: u8) -> Self {
        let pitch_class = PitchClass::from_semitone(note % 12);
        let octave = Octave((note / 12) as i8 - 1);
        Self { pitch_class, octave }
    }
}

impl fmt::Display for MusicalNote {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.pitch_class, self.octave)
    }
}

// ===========================================================================
// Spatialization
// ===========================================================================

/// Stereo pan position from -1.0 (full left) to 1.0 (full right).
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Pan(pub f64);

impl Pan {
    pub fn new(value: f64) -> Self {
        Self(value.clamp(-1.0, 1.0))
    }

    pub fn value(self) -> f64 { self.0 }

    pub const LEFT: Pan = Pan(-1.0);
    pub const CENTER: Pan = Pan(0.0);
    pub const RIGHT: Pan = Pan(1.0);

    /// Equal-power pan gains for left and right channels.
    pub fn equal_power_gains(self) -> (f64, f64) {
        let angle = (self.0 + 1.0) * std::f64::consts::FRAC_PI_4;
        (angle.cos(), angle.sin())
    }

    /// Linear pan gains.
    pub fn linear_gains(self) -> (f64, f64) {
        let right = (self.0 + 1.0) / 2.0;
        (1.0 - right, right)
    }
}

impl fmt::Display for Pan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0 < -0.01 {
            write!(f, "{:.0}% L", -self.0 * 100.0)
        } else if self.0 > 0.01 {
            write!(f, "{:.0}% R", self.0 * 100.0)
        } else {
            write!(f, "C")
        }
    }
}

impl Eq for Pan {}
impl Ord for Pan {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        OrderedFloat(self.0).cmp(&OrderedFloat(other.0))
    }
}

// ---------------------------------------------------------------------------

/// 3-D spatial position for HRTF / ambisonics rendering.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SpatialPosition {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl SpatialPosition {
    pub fn new(x: f64, y: f64, z: f64) -> Self { Self { x, y, z } }
    pub fn origin() -> Self { Self { x: 0.0, y: 0.0, z: 0.0 } }

    pub fn distance(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn distance_to(&self, other: &Self) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    pub fn azimuth(&self) -> f64 { self.x.atan2(self.z) }

    pub fn elevation(&self) -> f64 {
        let d = (self.x * self.x + self.z * self.z).sqrt();
        self.y.atan2(d)
    }
}

impl fmt::Display for SpatialPosition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:.2}, {:.2}, {:.2})", self.x, self.y, self.z)
    }
}

// ===========================================================================
// Data value representation
// ===========================================================================

/// A single data value to be sonified.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum DataValue {
    Continuous(f64),
    Discrete(i64),
    Categorical(String),
    Boolean(bool),
    TimeSeries(Vec<f64>),
    MultiDimensional(Vec<f64>),
}

impl DataValue {
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            DataValue::Continuous(v) => Some(*v),
            DataValue::Discrete(v) => Some(*v as f64),
            DataValue::Boolean(b) => Some(if *b { 1.0 } else { 0.0 }),
            _ => None,
        }
    }

    pub fn type_name(&self) -> &'static str {
        match self {
            DataValue::Continuous(_) => "Continuous",
            DataValue::Discrete(_) => "Discrete",
            DataValue::Categorical(_) => "Categorical",
            DataValue::Boolean(_) => "Boolean",
            DataValue::TimeSeries(_) => "TimeSeries",
            DataValue::MultiDimensional(_) => "MultiDimensional",
        }
    }

    pub fn dimensionality(&self) -> usize {
        match self {
            DataValue::TimeSeries(v) | DataValue::MultiDimensional(v) => v.len(),
            _ => 1,
        }
    }
}

impl fmt::Display for DataValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DataValue::Continuous(v) => write!(f, "{v:.6}"),
            DataValue::Discrete(v) => write!(f, "{v}"),
            DataValue::Categorical(s) => write!(f, "\"{s}\""),
            DataValue::Boolean(b) => write!(f, "{b}"),
            DataValue::TimeSeries(v) => write!(f, "TimeSeries[{}]", v.len()),
            DataValue::MultiDimensional(v) => write!(f, "MultiDim[{}]", v.len()),
        }
    }
}

// ===========================================================================
// Data distribution statistics
// ===========================================================================

/// Summary statistics of a data column distribution.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DataDistribution {
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub std_dev: f64,
    pub median: f64,
    pub percentiles: HashMap<OrderedFloat<f64>, f64>,
    pub histogram_counts: Vec<usize>,
    pub histogram_edges: Vec<f64>,
    pub count: usize,
}

impl DataDistribution {
    /// Compute statistics from raw f64 data.
    pub fn from_data(data: &[f64], num_bins: usize) -> Self {
        if data.is_empty() {
            return Self {
                min: 0.0, max: 0.0, mean: 0.0, std_dev: 0.0, median: 0.0,
                percentiles: HashMap::new(),
                histogram_counts: vec![], histogram_edges: vec![], count: 0,
            };
        }
        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = sorted.len();
        let min = sorted[0];
        let max = sorted[n - 1];
        let sum: f64 = sorted.iter().sum();
        let mean = sum / n as f64;
        let var = sorted.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        let std_dev = var.sqrt();
        let median = if n % 2 == 0 {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        } else {
            sorted[n / 2]
        };
        let mut percentiles = HashMap::new();
        for &p in &[5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0] {
            let idx = (p / 100.0) * (n - 1) as f64;
            let lo = idx.floor() as usize;
            let hi = idx.ceil() as usize;
            let val = if lo == hi { sorted[lo] }
                else { sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo as f64) };
            percentiles.insert(OrderedFloat(p), val);
        }
        let bins = num_bins.max(1);
        let range = if (max - min).abs() < 1e-15 { 1.0 } else { max - min };
        let bin_width = range / bins as f64;
        let edges: Vec<f64> = (0..=bins).map(|i| min + i as f64 * bin_width).collect();
        let mut counts = vec![0usize; bins];
        for &v in data {
            let mut idx = ((v - min) / bin_width).floor() as usize;
            if idx >= bins { idx = bins - 1; }
            counts[idx] += 1;
        }
        Self { min, max, mean, std_dev, median, percentiles,
               histogram_counts: counts, histogram_edges: edges, count: n }
    }

    pub fn range(&self) -> f64 { self.max - self.min }

    pub fn iqr(&self) -> f64 {
        let q25 = self.percentiles.get(&OrderedFloat(25.0)).copied().unwrap_or(self.min);
        let q75 = self.percentiles.get(&OrderedFloat(75.0)).copied().unwrap_or(self.max);
        q75 - q25
    }
}

impl fmt::Display for DataDistribution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Distribution(n={}, mean={:.4}, std={:.4}, [{:.4}, {:.4}])",
            self.count, self.mean, self.std_dev, self.min, self.max)
    }
}

// ===========================================================================
// Data schema
// ===========================================================================

/// The type of a data column.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DataColumnType {
    Continuous, Discrete, Categorical, Boolean, TimeSeries,
    MultiDimensional { dimensions: usize },
}

impl fmt::Display for DataColumnType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DataColumnType::Continuous => write!(f, "continuous"),
            DataColumnType::Discrete => write!(f, "discrete"),
            DataColumnType::Categorical => write!(f, "categorical"),
            DataColumnType::Boolean => write!(f, "boolean"),
            DataColumnType::TimeSeries => write!(f, "timeseries"),
            DataColumnType::MultiDimensional { dimensions } => write!(f, "multidim({dimensions})"),
        }
    }
}

/// Description of one column in the input data.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DataColumn {
    pub name: String,
    pub column_type: DataColumnType,
    pub distribution: Option<DataDistribution>,
    pub categories: Option<Vec<String>>,
    pub nullable: bool,
}

impl DataColumn {
    pub fn new(name: impl Into<String>, column_type: DataColumnType) -> Self {
        Self { name: name.into(), column_type, distribution: None, categories: None, nullable: false }
    }
}

/// Schema describing the full input data set.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DataSchema {
    pub columns: Vec<DataColumn>,
    pub row_count: Option<usize>,
}

impl DataSchema {
    pub fn new(columns: Vec<DataColumn>) -> Self {
        Self { columns, row_count: None }
    }

    pub fn column_count(&self) -> usize { self.columns.len() }

    pub fn column_by_name(&self, name: &str) -> Option<&DataColumn> {
        self.columns.iter().find(|c| c.name == name)
    }

    pub fn column_names(&self) -> Vec<&str> {
        self.columns.iter().map(|c| c.name.as_str()).collect()
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.columns.is_empty() {
            return Err("Schema must have at least one column".into());
        }
        let mut seen = std::collections::HashSet::new();
        for col in &self.columns {
            if !seen.insert(&col.name) {
                return Err(format!("Duplicate column name: {}", col.name));
            }
        }
        Ok(())
    }
}

impl fmt::Display for DataSchema {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Schema({} columns", self.columns.len())?;
        if let Some(n) = self.row_count { write!(f, ", {n} rows")?; }
        write!(f, ")")
    }
}

// ===========================================================================
// Stream descriptor
// ===========================================================================

/// Describes the auditory properties of a single sonification stream.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StreamDescriptor {
    pub frequency: Frequency,
    pub amplitude: Amplitude,
    pub pan: Pan,
    pub phase: Phase,
    pub timbre_index: usize,
    /// ADSR envelope: (attack_s, decay_s, sustain_level, release_s).
    pub envelope_adsr: (f64, f64, f64, f64),
    pub spatial: Option<SpatialPosition>,
}

impl StreamDescriptor {
    pub fn new(frequency: Frequency, amplitude: Amplitude) -> Self {
        Self {
            frequency, amplitude, pan: Pan::CENTER, phase: Phase(0.0),
            timbre_index: 0, envelope_adsr: (0.01, 0.1, 0.7, 0.2), spatial: None,
        }
    }
    pub fn attack(&self) -> f64 { self.envelope_adsr.0 }
    pub fn decay(&self) -> f64 { self.envelope_adsr.1 }
    pub fn sustain_level(&self) -> f64 { self.envelope_adsr.2 }
    pub fn release(&self) -> f64 { self.envelope_adsr.3 }
}

impl fmt::Display for StreamDescriptor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Stream(freq={}, amp={}, pan={})", self.frequency, self.amplitude, self.pan)
    }
}

// ===========================================================================
// Mapping parameter
// ===========================================================================

/// An audio parameter that a data dimension can be mapped to.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum MappingParameter {
    PitchRange(f64, f64),
    TimbreIndex(usize),
    PanPosition(f64),
    TemporalRate(f64),
    AmplitudeRange(f64, f64),
    FilterCutoff(f64),
}

impl MappingParameter {
    pub fn label(&self) -> &'static str {
        match self {
            MappingParameter::PitchRange(..) => "PitchRange",
            MappingParameter::TimbreIndex(_) => "TimbreIndex",
            MappingParameter::PanPosition(_) => "PanPosition",
            MappingParameter::TemporalRate(_) => "TemporalRate",
            MappingParameter::AmplitudeRange(..) => "AmplitudeRange",
            MappingParameter::FilterCutoff(_) => "FilterCutoff",
        }
    }
    pub fn is_pitch(&self) -> bool {
        matches!(self, MappingParameter::PitchRange(..) | MappingParameter::FilterCutoff(_))
    }
    pub fn is_spatial(&self) -> bool {
        matches!(self, MappingParameter::PanPosition(_))
    }
}

impl fmt::Display for MappingParameter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MappingParameter::PitchRange(lo, hi) => write!(f, "Pitch[{lo:.1}-{hi:.1} Hz]"),
            MappingParameter::TimbreIndex(i) => write!(f, "Timbre[{i}]"),
            MappingParameter::PanPosition(p) => write!(f, "Pan[{p:.2}]"),
            MappingParameter::TemporalRate(r) => write!(f, "Rate[{r:.2}/s]"),
            MappingParameter::AmplitudeRange(lo, hi) => write!(f, "Amp[{lo:.3}-{hi:.3}]"),
            MappingParameter::FilterCutoff(fc) => write!(f, "Filter[{fc:.1} Hz]"),
        }
    }
}

// ===========================================================================
// Perceptual qualifier
// ===========================================================================

/// Represents the psychoacoustic constraint state for a mapping.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PerceptualQualifier {
    pub jnd_pitch_hz: f64,
    pub jnd_loudness_db: f64,
    pub min_onset_asynchrony_ms: f64,
    pub max_masking_depth_db: f64,
    pub satisfied: bool,
    pub violations: Vec<String>,
}

impl PerceptualQualifier {
    pub fn default_thresholds() -> Self {
        Self {
            jnd_pitch_hz: 3.0, jnd_loudness_db: 1.0,
            min_onset_asynchrony_ms: 30.0, max_masking_depth_db: 6.0,
            satisfied: true, violations: Vec::new(),
        }
    }

    pub fn add_violation(&mut self, msg: impl Into<String>) {
        self.violations.push(msg.into());
        self.satisfied = false;
    }

    pub fn is_satisfied(&self) -> bool { self.satisfied && self.violations.is_empty() }
    pub fn violation_count(&self) -> usize { self.violations.len() }
}

impl fmt::Display for PerceptualQualifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.satisfied {
            write!(f, "PerceptualQualifier(OK)")
        } else {
            write!(f, "PerceptualQualifier(FAILED: {} violations)", self.violations.len())
        }
    }
}

// ===========================================================================
// Cognitive load budget
// ===========================================================================

/// Tracks the cognitive load budget for concurrent sonification streams.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CognitiveLoadBudget {
    pub capacity: f64,
    pub current_load: f64,
    pub stream_loads: Vec<(String, f64)>,
}

impl CognitiveLoadBudget {
    pub fn new(capacity: f64) -> Self {
        Self { capacity, current_load: 0.0, stream_loads: Vec::new() }
    }

    pub fn remaining(&self) -> f64 { (self.capacity - self.current_load).max(0.0) }

    pub fn can_add(&self, cost: f64) -> bool { self.current_load + cost <= self.capacity }

    pub fn add_stream(&mut self, label: impl Into<String>, cost: f64) -> bool {
        if !self.can_add(cost) { return false; }
        self.current_load += cost;
        self.stream_loads.push((label.into(), cost));
        true
    }

    pub fn remove_stream(&mut self, label: &str) -> Option<f64> {
        if let Some(pos) = self.stream_loads.iter().position(|(l, _)| l == label) {
            let (_, cost) = self.stream_loads.remove(pos);
            self.current_load -= cost;
            Some(cost)
        } else { None }
    }

    pub fn utilization(&self) -> f64 {
        if self.capacity <= 0.0 { return 1.0; }
        (self.current_load / self.capacity).clamp(0.0, 1.0)
    }

    pub fn stream_count(&self) -> usize { self.stream_loads.len() }
}

impl Default for CognitiveLoadBudget {
    fn default() -> Self { Self::new(4.0) }
}

impl fmt::Display for CognitiveLoadBudget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CogLoad({:.1}/{:.1}, {} streams)",
            self.current_load, self.capacity, self.stream_loads.len())
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frequency_basics() {
        let f = Frequency::new(440.0);
        assert_eq!(f.hz(), 440.0);
        assert!(f.is_audible());
        assert_eq!(format!("{f}"), "440.00 Hz");
    }

    #[test]
    fn frequency_ordering() {
        assert!(Frequency::new(200.0) < Frequency::new(400.0));
    }

    #[test]
    fn amplitude_clamping() {
        assert_eq!(Amplitude::new(1.5).value(), 1.0);
        assert_eq!(Amplitude::new(-0.5).value(), 0.0);
    }

    #[test]
    fn duration_conversions() {
        let d = Duration::from_millis(500.0);
        assert!((d.seconds() - 0.5).abs() < 1e-10);
        assert!((d.millis() - 500.0).abs() < 1e-10);
        assert_eq!(d.samples(44100), 22050);
    }

    #[test]
    fn phase_wrapping() {
        let p = Phase::new(3.0 * std::f64::consts::PI);
        assert!(p.radians() < 2.0 * std::f64::consts::PI);
        assert!(p.radians() >= 0.0);
    }

    #[test]
    fn sample_rate_nyquist() {
        assert_eq!(SampleRate::CD_QUALITY.nyquist().hz(), 22050.0);
    }

    #[test]
    fn buffer_size_duration_test() {
        let d = BufferSize::new(1024).duration(SampleRate::new(48000));
        assert!((d.seconds() - 1024.0 / 48000.0).abs() < 1e-10);
    }

    #[test]
    fn bark_band_center() {
        assert!((BarkBand::new(0).center_frequency().hz() - 50.0).abs() < 1e-10);
    }

    #[test]
    fn pitch_class_midi() {
        assert_eq!(PitchClass::A.midi_note(Octave(4)), 69);
        assert!((PitchClass::A.frequency(Octave(4)).hz() - 440.0).abs() < 0.01);
    }

    #[test]
    fn musical_note_roundtrip() {
        let n = MusicalNote::new(PitchClass::C, Octave(4));
        assert_eq!(n, MusicalNote::from_midi(n.midi_note()));
    }

    #[test]
    fn pan_gains() {
        let (l, r) = Pan::CENTER.equal_power_gains();
        assert!((l - r).abs() < 0.01);
    }

    #[test]
    fn spatial_distance() {
        assert!((SpatialPosition::new(3.0, 0.0, 4.0).distance() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn data_value_types() {
        assert_eq!(DataValue::Continuous(1.0).as_f64(), Some(1.0));
        assert_eq!(DataValue::Boolean(true).as_f64(), Some(1.0));
        assert_eq!(DataValue::Categorical("a".into()).as_f64(), None);
    }

    #[test]
    fn data_distribution_from_data() {
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let dist = DataDistribution::from_data(&data, 10);
        assert_eq!(dist.count, 100);
        assert!((dist.mean - 49.5).abs() < 0.01);
    }

    #[test]
    fn data_schema_validation() {
        let ok = DataSchema::new(vec![
            DataColumn::new("x", DataColumnType::Continuous),
            DataColumn::new("y", DataColumnType::Discrete),
        ]);
        assert!(ok.validate().is_ok());
        let bad = DataSchema::new(vec![
            DataColumn::new("x", DataColumnType::Continuous),
            DataColumn::new("x", DataColumnType::Continuous),
        ]);
        assert!(bad.validate().is_err());
    }

    #[test]
    fn stream_descriptor_basics() {
        let sd = StreamDescriptor::new(Frequency::new(440.0), Amplitude::new(0.5));
        assert!(sd.attack() > 0.0);
    }

    #[test]
    fn mapping_parameter_labels() {
        let mp = MappingParameter::PitchRange(200.0, 2000.0);
        assert_eq!(mp.label(), "PitchRange");
        assert!(mp.is_pitch());
    }

    #[test]
    fn perceptual_qualifier_violations() {
        let mut pq = PerceptualQualifier::default_thresholds();
        assert!(pq.is_satisfied());
        pq.add_violation("masking");
        assert!(!pq.is_satisfied());
    }

    #[test]
    fn cognitive_load_budget() {
        let mut clb = CognitiveLoadBudget::new(4.0);
        assert!(clb.add_stream("pitch", 1.5));
        assert!(clb.add_stream("timbre", 1.5));
        assert!(!clb.can_add(2.0));
        assert!((clb.utilization() - 0.75).abs() < 1e-10);
        assert_eq!(clb.remove_stream("pitch"), Some(1.5));
    }

    #[test]
    fn serde_roundtrip_data_value() {
        let v = DataValue::Continuous(3.14);
        let json = serde_json::to_string(&v).unwrap();
        let back: DataValue = serde_json::from_str(&json).unwrap();
        assert_eq!(v, back);
    }

    #[test]
    fn serde_roundtrip_mapping_param() {
        let mp = MappingParameter::PitchRange(100.0, 8000.0);
        let json = serde_json::to_string(&mp).unwrap();
        let back: MappingParameter = serde_json::from_str(&json).unwrap();
        assert_eq!(mp, back);
    }
}
