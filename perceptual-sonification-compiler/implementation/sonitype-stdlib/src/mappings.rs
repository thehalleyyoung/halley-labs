//! Data-to-sound mapping presets for SoniType.
//!
//! Map data dimensions (value, category, time, etc.) to audio parameters
//! (pitch, loudness, timbre, spatial position, filter).

use std::collections::HashMap;

use crate::scales::{LinearScale, LogarithmicScale, MusicalScale, ScaleType, NoteName, LogBase};
use crate::timbres::{TimbrePalette, TimbreDescriptor};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn clamp(v: f64, lo: f64, hi: f64) -> f64 {
    v.max(lo).min(hi)
}

fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

fn map_range(v: f64, in_lo: f64, in_hi: f64, out_lo: f64, out_hi: f64) -> f64 {
    if (in_hi - in_lo).abs() < f64::EPSILON {
        return out_lo;
    }
    let t = (v - in_lo) / (in_hi - in_lo);
    lerp(out_lo, out_hi, t)
}

// ---------------------------------------------------------------------------
// Polarity
// ---------------------------------------------------------------------------

/// Whether higher data values map to higher or lower audio values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Polarity {
    Ascending,
    Descending,
}

// ---------------------------------------------------------------------------
// PitchMapping
// ---------------------------------------------------------------------------

/// Pitch scale selection for `PitchMapping`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PitchScaleKind {
    Continuous,
    ContinuousLog,
    Musical,
}

/// Map data values to pitch (frequency).
#[derive(Debug, Clone)]
pub struct PitchMapping {
    pub scale_kind: PitchScaleKind,
    pub f_min: f64,
    pub f_max: f64,
    pub polarity: Polarity,
    pub data_range: (f64, f64),
    /// Optional musical scale for discrete mapping.
    musical_scale: Option<MusicalScale>,
}

impl PitchMapping {
    pub fn continuous(f_min: f64, f_max: f64, data_range: (f64, f64)) -> Self {
        Self {
            scale_kind: PitchScaleKind::Continuous,
            f_min, f_max,
            polarity: Polarity::Ascending,
            data_range,
            musical_scale: None,
        }
    }

    pub fn continuous_log(f_min: f64, f_max: f64, data_range: (f64, f64)) -> Self {
        Self {
            scale_kind: PitchScaleKind::ContinuousLog,
            f_min, f_max,
            polarity: Polarity::Ascending,
            data_range,
            musical_scale: None,
        }
    }

    pub fn musical(
        root: NoteName,
        scale_type: ScaleType,
        octave_low: i32,
        octave_high: i32,
        data_range: (f64, f64),
    ) -> Self {
        let ms = MusicalScale::new(root, scale_type, octave_low, octave_high);
        let pitches = ms.pitches();
        let f_min = pitches.first().copied().unwrap_or(200.0);
        let f_max = pitches.last().copied().unwrap_or(4000.0);
        Self {
            scale_kind: PitchScaleKind::Musical,
            f_min, f_max,
            polarity: Polarity::Ascending,
            data_range,
            musical_scale: Some(ms),
        }
    }

    pub fn with_polarity(mut self, pol: Polarity) -> Self {
        self.polarity = pol;
        self
    }

    /// Map data value to Hz.
    pub fn map(&self, value: f64) -> f64 {
        let v = clamp(value, self.data_range.0, self.data_range.1);
        let (out_lo, out_hi) = match self.polarity {
            Polarity::Ascending => (self.f_min, self.f_max),
            Polarity::Descending => (self.f_max, self.f_min),
        };

        match self.scale_kind {
            PitchScaleKind::Continuous => {
                let ls = LinearScale::new(out_lo.min(out_hi), out_lo.max(out_hi));
                match self.polarity {
                    Polarity::Ascending => ls.map(v, self.data_range),
                    Polarity::Descending => {
                        let freq = ls.map(v, self.data_range);
                        self.f_min + self.f_max - freq
                    }
                }
            }
            PitchScaleKind::ContinuousLog => {
                let ls = LogarithmicScale::new(out_lo.min(out_hi), out_lo.max(out_hi), LogBase::Base2);
                match self.polarity {
                    Polarity::Ascending => ls.map(v, self.data_range),
                    Polarity::Descending => {
                        let freq = ls.map(v, self.data_range);
                        self.f_min + self.f_max - freq
                    }
                }
            }
            PitchScaleKind::Musical => {
                if let Some(ref ms) = self.musical_scale {
                    match self.polarity {
                        Polarity::Ascending => ms.map(v, self.data_range),
                        Polarity::Descending => {
                            let inv_v = self.data_range.1 - (v - self.data_range.0);
                            ms.map(inv_v, self.data_range)
                        }
                    }
                } else {
                    440.0
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// LoudnessMapping
// ---------------------------------------------------------------------------

/// Loudness curve type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoudnessCurve {
    Linear,
    Logarithmic,
    EqualLoudnessCorrected,
}

/// Loudness unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoudnessUnit {
    Phon,
    Sone,
    DecibelsFullScale,
}

/// Map data values to loudness.
#[derive(Debug, Clone)]
pub struct LoudnessMapping {
    pub curve: LoudnessCurve,
    pub unit: LoudnessUnit,
    pub range_min: f64,
    pub range_max: f64,
    pub data_range: (f64, f64),
}

impl LoudnessMapping {
    pub fn new(curve: LoudnessCurve, unit: LoudnessUnit, range: (f64, f64), data_range: (f64, f64)) -> Self {
        Self {
            curve,
            unit,
            range_min: range.0,
            range_max: range.1,
            data_range,
        }
    }

    pub fn linear_db(db_min: f64, db_max: f64, data_range: (f64, f64)) -> Self {
        Self::new(LoudnessCurve::Linear, LoudnessUnit::DecibelsFullScale, (db_min, db_max), data_range)
    }

    pub fn log_phon(phon_min: f64, phon_max: f64, data_range: (f64, f64)) -> Self {
        Self::new(LoudnessCurve::Logarithmic, LoudnessUnit::Phon, (phon_min, phon_max), data_range)
    }

    pub fn equal_loudness_sone(sone_min: f64, sone_max: f64, data_range: (f64, f64)) -> Self {
        Self::new(LoudnessCurve::EqualLoudnessCorrected, LoudnessUnit::Sone, (sone_min, sone_max), data_range)
    }

    /// Map data value to loudness in the configured unit.
    pub fn map(&self, value: f64) -> f64 {
        let v = clamp(value, self.data_range.0, self.data_range.1);
        match self.curve {
            LoudnessCurve::Linear => {
                map_range(v, self.data_range.0, self.data_range.1, self.range_min, self.range_max)
            }
            LoudnessCurve::Logarithmic => {
                let t = if (self.data_range.1 - self.data_range.0).abs() < f64::EPSILON {
                    0.0
                } else {
                    (v - self.data_range.0) / (self.data_range.1 - self.data_range.0)
                };
                // Use a logarithmic curve: compress lower values
                let log_t = (1.0 + t * 9.0).log10(); // 0→0, 1→1
                lerp(self.range_min, self.range_max, log_t)
            }
            LoudnessCurve::EqualLoudnessCorrected => {
                // Apply Fletcher-Munson correction approximation
                let linear_val = map_range(v, self.data_range.0, self.data_range.1,
                                           self.range_min, self.range_max);
                // Boost quiet sounds, compress loud sounds (simplified equal-loudness)
                let mid = (self.range_min + self.range_max) / 2.0;
                let correction = 0.15 * (mid - linear_val).signum() * ((mid - linear_val).abs() / mid).min(1.0);
                clamp(linear_val + correction * (self.range_max - self.range_min),
                      self.range_min, self.range_max)
            }
        }
    }

    /// Convert phon to sone (approximate).
    pub fn phon_to_sone(phon: f64) -> f64 {
        if phon < 40.0 {
            (phon / 40.0).powf(2.642)
        } else {
            2.0_f64.powf((phon - 40.0) / 10.0)
        }
    }

    /// Convert sone to phon (approximate inverse).
    pub fn sone_to_phon(sone: f64) -> f64 {
        if sone < 1.0 {
            40.0 * sone.powf(1.0 / 2.642)
        } else {
            40.0 + 10.0 * sone.log2()
        }
    }

    /// Convert dBFS to linear amplitude.
    pub fn dbfs_to_amplitude(db: f64) -> f64 {
        10.0_f64.powf(db / 20.0)
    }

    /// Convert linear amplitude to dBFS.
    pub fn amplitude_to_dbfs(amp: f64) -> f64 {
        if amp <= 0.0 {
            -120.0
        } else {
            20.0 * amp.log10()
        }
    }
}

// ---------------------------------------------------------------------------
// TemporalMapping
// ---------------------------------------------------------------------------

/// Temporal mapping mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemporalMode {
    /// Higher value → faster rate.
    Rate,
    /// Higher value → longer duration.
    Duration,
    /// Select rhythm patterns by index.
    RhythmPattern,
}

/// Map data values to timing parameters.
#[derive(Debug, Clone)]
pub struct TemporalMapping {
    pub mode: TemporalMode,
    pub bpm_min: f64,
    pub bpm_max: f64,
    pub duration_min_s: f64,
    pub duration_max_s: f64,
    pub data_range: (f64, f64),
    /// Named rhythm patterns (as fractional beat sequences).
    pub patterns: Vec<(String, Vec<f64>)>,
}

impl TemporalMapping {
    pub fn rate(bpm_min: f64, bpm_max: f64, data_range: (f64, f64)) -> Self {
        Self {
            mode: TemporalMode::Rate,
            bpm_min, bpm_max,
            duration_min_s: 0.05,
            duration_max_s: 2.0,
            data_range,
            patterns: Vec::new(),
        }
    }

    pub fn duration(dur_min: f64, dur_max: f64, data_range: (f64, f64)) -> Self {
        Self {
            mode: TemporalMode::Duration,
            bpm_min: 60.0,
            bpm_max: 200.0,
            duration_min_s: dur_min,
            duration_max_s: dur_max,
            data_range,
            patterns: Vec::new(),
        }
    }

    pub fn rhythm(patterns: Vec<(String, Vec<f64>)>, data_range: (f64, f64)) -> Self {
        Self {
            mode: TemporalMode::RhythmPattern,
            bpm_min: 60.0,
            bpm_max: 200.0,
            duration_min_s: 0.05,
            duration_max_s: 2.0,
            data_range,
            patterns,
        }
    }

    pub fn with_default_patterns(mut self) -> Self {
        self.patterns = vec![
            ("quarter".into(), vec![1.0]),
            ("eighth".into(), vec![0.5, 0.5]),
            ("triplet".into(), vec![1.0/3.0, 1.0/3.0, 1.0/3.0]),
            ("sixteenth".into(), vec![0.25, 0.25, 0.25, 0.25]),
            ("dotted".into(), vec![0.75, 0.25]),
            ("swing".into(), vec![0.67, 0.33]),
        ];
        self
    }

    /// Map data value according to mode.
    pub fn map(&self, value: f64) -> TemporalResult {
        let v = clamp(value, self.data_range.0, self.data_range.1);
        match self.mode {
            TemporalMode::Rate => {
                let bpm = map_range(v, self.data_range.0, self.data_range.1, self.bpm_min, self.bpm_max);
                TemporalResult::Rate { bpm }
            }
            TemporalMode::Duration => {
                let dur = map_range(v, self.data_range.0, self.data_range.1,
                                    self.duration_min_s, self.duration_max_s);
                TemporalResult::Duration { seconds: dur }
            }
            TemporalMode::RhythmPattern => {
                if self.patterns.is_empty() {
                    return TemporalResult::Pattern { name: "quarter".into(), beats: vec![1.0] };
                }
                let t = (v - self.data_range.0) / (self.data_range.1 - self.data_range.0);
                let idx = (t * (self.patterns.len() - 1) as f64).round() as usize;
                let idx = idx.min(self.patterns.len() - 1);
                let (name, beats) = &self.patterns[idx];
                TemporalResult::Pattern { name: name.clone(), beats: beats.clone() }
            }
        }
    }
}

/// Result of a temporal mapping.
#[derive(Debug, Clone)]
pub enum TemporalResult {
    Rate { bpm: f64 },
    Duration { seconds: f64 },
    Pattern { name: String, beats: Vec<f64> },
}

// ---------------------------------------------------------------------------
// TimbreMapping
// ---------------------------------------------------------------------------

/// Map categorical data to timbres from a palette.
#[derive(Debug, Clone)]
pub struct TimbreMapping {
    pub palette: TimbrePalette,
    pub category_map: HashMap<String, usize>,
}

impl TimbreMapping {
    /// Create mapping from a list of category names; auto-assigns timbres from palette.
    pub fn auto_assign(categories: &[&str], palette: TimbrePalette) -> Self {
        let selected = palette.select_n(categories.len());
        let mut category_map = HashMap::new();
        for (i, &cat) in categories.iter().enumerate() {
            category_map.insert(cat.to_string(), i.min(selected.len() - 1));
        }
        Self { palette, category_map }
    }

    /// Create mapping with explicit category → timbre-index assignments.
    pub fn explicit(assignments: HashMap<String, usize>, palette: TimbrePalette) -> Self {
        Self { palette, category_map: assignments }
    }

    /// Look up the timbre descriptor for a category.
    pub fn map(&self, category: &str) -> Option<&TimbreDescriptor> {
        self.category_map.get(category)
            .and_then(|&idx| self.palette.descriptors.get(idx))
    }

    /// Number of mapped categories.
    pub fn num_categories(&self) -> usize {
        self.category_map.len()
    }
}

// ---------------------------------------------------------------------------
// SpatialMapping
// ---------------------------------------------------------------------------

/// Spatial parameter being mapped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpatialDimension {
    /// Left (-1) to right (+1).
    Pan,
    /// Azimuth angle (0–360°).
    Azimuth,
    /// Simulated distance (affects amplitude and filtering).
    Distance,
}

/// Map data values to spatial audio position.
#[derive(Debug, Clone)]
pub struct SpatialMapping {
    pub dimension: SpatialDimension,
    pub range: (f64, f64),
    pub data_range: (f64, f64),
}

impl SpatialMapping {
    pub fn pan(data_range: (f64, f64)) -> Self {
        Self { dimension: SpatialDimension::Pan, range: (-1.0, 1.0), data_range }
    }

    pub fn azimuth(data_range: (f64, f64)) -> Self {
        Self { dimension: SpatialDimension::Azimuth, range: (0.0, 360.0), data_range }
    }

    pub fn distance(max_distance: f64, data_range: (f64, f64)) -> Self {
        Self { dimension: SpatialDimension::Distance, range: (0.0, max_distance), data_range }
    }

    pub fn map(&self, value: f64) -> SpatialResult {
        let v = clamp(value, self.data_range.0, self.data_range.1);
        let mapped = map_range(v, self.data_range.0, self.data_range.1, self.range.0, self.range.1);
        match self.dimension {
            SpatialDimension::Pan => SpatialResult::Pan(clamp(mapped, -1.0, 1.0)),
            SpatialDimension::Azimuth => SpatialResult::Azimuth(mapped % 360.0),
            SpatialDimension::Distance => {
                let dist = mapped.max(0.0);
                // Inverse-square law amplitude attenuation
                let amplitude_factor = if dist < 0.01 { 1.0 } else { 1.0 / (1.0 + dist) };
                // Low-pass filter cutoff decreases with distance
                let lp_cutoff = 20000.0 / (1.0 + dist * 2.0);
                SpatialResult::Distance { distance: dist, amplitude_factor, lp_cutoff_hz: lp_cutoff }
            }
        }
    }
}

/// Spatial mapping result.
#[derive(Debug, Clone)]
pub enum SpatialResult {
    Pan(f64),
    Azimuth(f64),
    Distance {
        distance: f64,
        amplitude_factor: f64,
        lp_cutoff_hz: f64,
    },
}

// ---------------------------------------------------------------------------
// FilterMapping
// ---------------------------------------------------------------------------

/// Filter parameter target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterParameter {
    Cutoff,
    Resonance,
}

/// Map data values to filter parameters.
#[derive(Debug, Clone)]
pub struct FilterMapping {
    pub parameter: FilterParameter,
    pub range: (f64, f64),
    pub data_range: (f64, f64),
    pub log_scale: bool,
}

impl FilterMapping {
    pub fn cutoff(hz_min: f64, hz_max: f64, data_range: (f64, f64)) -> Self {
        Self {
            parameter: FilterParameter::Cutoff,
            range: (hz_min, hz_max),
            data_range,
            log_scale: true,
        }
    }

    pub fn resonance(q_min: f64, q_max: f64, data_range: (f64, f64)) -> Self {
        Self {
            parameter: FilterParameter::Resonance,
            range: (q_min, q_max),
            data_range,
            log_scale: false,
        }
    }

    pub fn with_log_scale(mut self, use_log: bool) -> Self {
        self.log_scale = use_log;
        self
    }

    /// Map data value to filter parameter value.
    pub fn map(&self, value: f64) -> f64 {
        let v = clamp(value, self.data_range.0, self.data_range.1);
        if self.log_scale && self.range.0 > 0.0 {
            let t = (v - self.data_range.0) / (self.data_range.1 - self.data_range.0);
            let t = clamp(t, 0.0, 1.0);
            let log_min = self.range.0.ln();
            let log_max = self.range.1.ln();
            (lerp(log_min, log_max, t)).exp()
        } else {
            map_range(v, self.data_range.0, self.data_range.1, self.range.0, self.range.1)
        }
    }
}

// ---------------------------------------------------------------------------
// CompositeMapping
// ---------------------------------------------------------------------------

/// An audio parameter that can be the target of a mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AudioParameter {
    Pitch,
    Loudness,
    Tempo,
    Timbre,
    Pan,
    FilterCutoff,
    FilterResonance,
}

/// A single layer in a composite mapping.
#[derive(Debug, Clone)]
pub struct MappingLayer {
    pub data_field: String,
    pub target: AudioParameter,
    pub priority: u8,
}

/// Combine multiple mappings for multi-dimensional sonification.
#[derive(Debug, Clone)]
pub struct CompositeMapping {
    pub layers: Vec<MappingLayer>,
    pub pitch: Option<PitchMapping>,
    pub loudness: Option<LoudnessMapping>,
    pub temporal: Option<TemporalMapping>,
    pub timbre: Option<TimbreMapping>,
    pub spatial: Option<SpatialMapping>,
    pub filter: Option<FilterMapping>,
}

impl CompositeMapping {
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            pitch: None,
            loudness: None,
            temporal: None,
            timbre: None,
            spatial: None,
            filter: None,
        }
    }

    pub fn add_pitch(mut self, field: &str, mapping: PitchMapping, priority: u8) -> Self {
        self.layers.push(MappingLayer { data_field: field.into(), target: AudioParameter::Pitch, priority });
        self.pitch = Some(mapping);
        self
    }

    pub fn add_loudness(mut self, field: &str, mapping: LoudnessMapping, priority: u8) -> Self {
        self.layers.push(MappingLayer { data_field: field.into(), target: AudioParameter::Loudness, priority });
        self.loudness = Some(mapping);
        self
    }

    pub fn add_temporal(mut self, field: &str, mapping: TemporalMapping, priority: u8) -> Self {
        self.layers.push(MappingLayer { data_field: field.into(), target: AudioParameter::Tempo, priority });
        self.temporal = Some(mapping);
        self
    }

    pub fn add_timbre(mut self, field: &str, mapping: TimbreMapping, priority: u8) -> Self {
        self.layers.push(MappingLayer { data_field: field.into(), target: AudioParameter::Timbre, priority });
        self.timbre = Some(mapping);
        self
    }

    pub fn add_spatial(mut self, field: &str, mapping: SpatialMapping, priority: u8) -> Self {
        self.layers.push(MappingLayer { data_field: field.into(), target: AudioParameter::Pan, priority });
        self.spatial = Some(mapping);
        self
    }

    pub fn add_filter(mut self, field: &str, mapping: FilterMapping, priority: u8) -> Self {
        self.layers.push(MappingLayer { data_field: field.into(), target: AudioParameter::FilterCutoff, priority });
        self.filter = Some(mapping);
        self
    }

    /// Get layers sorted by priority (highest first).
    pub fn sorted_layers(&self) -> Vec<&MappingLayer> {
        let mut sorted: Vec<&MappingLayer> = self.layers.iter().collect();
        sorted.sort_by(|a, b| b.priority.cmp(&a.priority));
        sorted
    }

    /// Map a set of named data values to audio parameters.
    pub fn map_values(&self, values: &HashMap<String, f64>) -> MappedAudioParams {
        let mut result = MappedAudioParams::default();
        if let Some(ref pm) = self.pitch {
            if let Some(field_layer) = self.layers.iter().find(|l| l.target == AudioParameter::Pitch) {
                if let Some(&v) = values.get(&field_layer.data_field) {
                    result.pitch_hz = Some(pm.map(v));
                }
            }
        }
        if let Some(ref lm) = self.loudness {
            if let Some(field_layer) = self.layers.iter().find(|l| l.target == AudioParameter::Loudness) {
                if let Some(&v) = values.get(&field_layer.data_field) {
                    result.loudness = Some(lm.map(v));
                }
            }
        }
        if let Some(ref sm) = self.spatial {
            if let Some(field_layer) = self.layers.iter().find(|l| l.target == AudioParameter::Pan) {
                if let Some(&v) = values.get(&field_layer.data_field) {
                    if let SpatialResult::Pan(p) = sm.map(v) {
                        result.pan = Some(p);
                    }
                }
            }
        }
        if let Some(ref fm) = self.filter {
            if let Some(field_layer) = self.layers.iter().find(|l| l.target == AudioParameter::FilterCutoff) {
                if let Some(&v) = values.get(&field_layer.data_field) {
                    result.filter_cutoff_hz = Some(fm.map(v));
                }
            }
        }
        result
    }
}

impl Default for CompositeMapping {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of a composite mapping.
#[derive(Debug, Clone, Default)]
pub struct MappedAudioParams {
    pub pitch_hz: Option<f64>,
    pub loudness: Option<f64>,
    pub tempo_bpm: Option<f64>,
    pub timbre_name: Option<String>,
    pub pan: Option<f64>,
    pub filter_cutoff_hz: Option<f64>,
    pub filter_resonance: Option<f64>,
}

// ---------------------------------------------------------------------------
// MappingBuilder (fluent API)
// ---------------------------------------------------------------------------

/// Fluent API for building composite mappings.
#[derive(Debug)]
pub struct MappingBuilder {
    composite: CompositeMapping,
}

impl MappingBuilder {
    pub fn new() -> Self {
        Self { composite: CompositeMapping::new() }
    }

    pub fn pitch(mut self, field: &str, mapping: PitchMapping) -> Self {
        self.composite = self.composite.add_pitch(field, mapping, 10);
        self
    }

    pub fn loudness(mut self, field: &str, mapping: LoudnessMapping) -> Self {
        self.composite = self.composite.add_loudness(field, mapping, 8);
        self
    }

    pub fn tempo(mut self, field: &str, mapping: TemporalMapping) -> Self {
        self.composite = self.composite.add_temporal(field, mapping, 6);
        self
    }

    pub fn timbre(mut self, field: &str, mapping: TimbreMapping) -> Self {
        self.composite = self.composite.add_timbre(field, mapping, 7);
        self
    }

    pub fn pan(mut self, field: &str, mapping: SpatialMapping) -> Self {
        self.composite = self.composite.add_spatial(field, mapping, 5);
        self
    }

    pub fn filter(mut self, field: &str, mapping: FilterMapping) -> Self {
        self.composite = self.composite.add_filter(field, mapping, 4);
        self
    }

    pub fn build(self) -> CompositeMapping {
        self.composite
    }
}

impl Default for MappingBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pitch_continuous_boundaries() {
        let pm = PitchMapping::continuous(200.0, 4000.0, (0.0, 100.0));
        let f0 = pm.map(0.0);
        let f1 = pm.map(100.0);
        assert!((f0 - 200.0).abs() < 1.0);
        assert!((f1 - 4000.0).abs() < 1.0);
    }

    #[test]
    fn test_pitch_descending() {
        let pm = PitchMapping::continuous(200.0, 4000.0, (0.0, 1.0))
            .with_polarity(Polarity::Descending);
        let f_lo = pm.map(0.0);
        let f_hi = pm.map(1.0);
        assert!(f_lo > f_hi);
    }

    #[test]
    fn test_pitch_musical_quantised() {
        let pm = PitchMapping::musical(NoteName::C, ScaleType::Major, 4, 4, (0.0, 1.0));
        let f = pm.map(0.5);
        // Should be one of the C-major pitches
        assert!(f > 200.0 && f < 600.0);
    }

    #[test]
    fn test_loudness_linear_db() {
        let lm = LoudnessMapping::linear_db(-60.0, 0.0, (0.0, 1.0));
        let lo = lm.map(0.0);
        let hi = lm.map(1.0);
        assert!((lo - (-60.0)).abs() < 1e-6);
        assert!((hi - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_loudness_log_monotonic() {
        let lm = LoudnessMapping::log_phon(20.0, 80.0, (0.0, 1.0));
        let v1 = lm.map(0.25);
        let v2 = lm.map(0.75);
        assert!(v2 > v1);
    }

    #[test]
    fn test_loudness_phon_sone_roundtrip() {
        let phon = 60.0;
        let sone = LoudnessMapping::phon_to_sone(phon);
        let recovered = LoudnessMapping::sone_to_phon(sone);
        assert!((recovered - phon).abs() < 0.5);
    }

    #[test]
    fn test_loudness_dbfs_amplitude_roundtrip() {
        let db = -6.0;
        let amp = LoudnessMapping::dbfs_to_amplitude(db);
        let recovered = LoudnessMapping::amplitude_to_dbfs(amp);
        assert!((recovered - db).abs() < 1e-6);
    }

    #[test]
    fn test_temporal_rate() {
        let tm = TemporalMapping::rate(60.0, 200.0, (0.0, 1.0));
        if let TemporalResult::Rate { bpm } = tm.map(0.5) {
            assert!(bpm > 60.0 && bpm < 200.0);
        } else {
            panic!("Expected Rate result");
        }
    }

    #[test]
    fn test_temporal_duration() {
        let tm = TemporalMapping::duration(0.05, 2.0, (0.0, 1.0));
        if let TemporalResult::Duration { seconds } = tm.map(0.0) {
            assert!((seconds - 0.05).abs() < 1e-6);
        } else {
            panic!("Expected Duration result");
        }
    }

    #[test]
    fn test_temporal_rhythm_pattern() {
        let tm = TemporalMapping::rhythm(vec![], (0.0, 1.0)).with_default_patterns();
        if let TemporalResult::Pattern { name, beats } = tm.map(0.0) {
            assert_eq!(name, "quarter");
            assert_eq!(beats, vec![1.0]);
        } else {
            panic!("Expected Pattern result");
        }
    }

    #[test]
    fn test_timbre_mapping_auto_assign() {
        let palette = TimbrePalette::palette_8();
        let tm = TimbreMapping::auto_assign(&["cat_a", "cat_b", "cat_c"], palette);
        assert!(tm.map("cat_a").is_some());
        assert!(tm.map("cat_b").is_some());
        assert_eq!(tm.num_categories(), 3);
    }

    #[test]
    fn test_spatial_pan() {
        let sm = SpatialMapping::pan((0.0, 1.0));
        if let SpatialResult::Pan(p) = sm.map(0.0) {
            assert!((p - (-1.0)).abs() < 1e-6);
        }
        if let SpatialResult::Pan(p) = sm.map(1.0) {
            assert!((p - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_spatial_distance() {
        let sm = SpatialMapping::distance(10.0, (0.0, 1.0));
        if let SpatialResult::Distance { distance, amplitude_factor, lp_cutoff_hz } = sm.map(1.0) {
            assert!((distance - 10.0).abs() < 1e-6);
            assert!(amplitude_factor < 1.0);
            assert!(lp_cutoff_hz < 20000.0);
        }
    }

    #[test]
    fn test_filter_cutoff_log() {
        let fm = FilterMapping::cutoff(200.0, 10000.0, (0.0, 1.0));
        let lo = fm.map(0.0);
        let hi = fm.map(1.0);
        assert!((lo - 200.0).abs() < 1.0);
        assert!((hi - 10000.0).abs() < 1.0);
        // midpoint should be geometric mean for log scale
        let mid = fm.map(0.5);
        let geo_mean = (200.0_f64 * 10000.0).sqrt();
        assert!((mid - geo_mean).abs() / geo_mean < 0.01);
    }

    #[test]
    fn test_filter_resonance_linear() {
        let fm = FilterMapping::resonance(0.5, 20.0, (0.0, 1.0));
        let mid = fm.map(0.5);
        assert!((mid - 10.25).abs() < 0.1);
    }

    #[test]
    fn test_composite_mapping() {
        let cm = CompositeMapping::new()
            .add_pitch("temperature", PitchMapping::continuous(200.0, 2000.0, (0.0, 100.0)), 10)
            .add_loudness("pressure", LoudnessMapping::linear_db(-40.0, 0.0, (0.0, 1.0)), 8);
        assert_eq!(cm.layers.len(), 2);
        let sorted = cm.sorted_layers();
        assert_eq!(sorted[0].priority, 10);
    }

    #[test]
    fn test_composite_map_values() {
        let cm = CompositeMapping::new()
            .add_pitch("x", PitchMapping::continuous(200.0, 2000.0, (0.0, 1.0)), 10)
            .add_spatial("y", SpatialMapping::pan((0.0, 1.0)), 5);
        let mut vals = HashMap::new();
        vals.insert("x".into(), 0.5);
        vals.insert("y".into(), 0.0);
        let result = cm.map_values(&vals);
        assert!(result.pitch_hz.is_some());
        assert!(result.pan.is_some());
    }

    #[test]
    fn test_mapping_builder() {
        let cm = MappingBuilder::new()
            .pitch("val", PitchMapping::continuous(200.0, 4000.0, (0.0, 1.0)))
            .loudness("vol", LoudnessMapping::linear_db(-60.0, 0.0, (0.0, 1.0)))
            .pan("pos", SpatialMapping::pan((-1.0, 1.0)))
            .build();
        assert_eq!(cm.layers.len(), 3);
    }

    #[test]
    fn test_pitch_continuous_log() {
        let pm = PitchMapping::continuous_log(200.0, 4000.0, (0.0, 1.0));
        let f = pm.map(0.5);
        // Log midpoint ≈ geometric mean
        let geo = (200.0_f64 * 4000.0).sqrt();
        assert!((f - geo).abs() / geo < 0.02);
    }
}
