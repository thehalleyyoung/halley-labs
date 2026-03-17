//! Output adaptation: frequency remapping, dynamic-range adaptation,
//! temporal adaptation, and spatial adaptation for hearing profiles.

use crate::hearing_profile::HearingProfile;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// FrequencyRemapper
// ---------------------------------------------------------------------------

/// Transposes sonification frequencies into the audible range of a hearing
/// profile, compresses the frequency range, avoids tinnitus frequencies,
/// and maintains relative pitch relationships.
#[derive(Debug, Clone)]
pub struct FrequencyRemapper {
    source_min_hz: f64,
    source_max_hz: f64,
    target_min_hz: f64,
    target_max_hz: f64,
    tinnitus_frequencies: Vec<f64>,
    tinnitus_avoidance_hz: f64,
    use_log_mapping: bool,
}

impl FrequencyRemapper {
    pub fn new(source_min: f64, source_max: f64) -> Self {
        Self {
            source_min_hz: source_min.max(20.0),
            source_max_hz: source_max.max(source_min + 1.0),
            target_min_hz: source_min.max(20.0),
            target_max_hz: source_max.min(16000.0),
            tinnitus_frequencies: Vec::new(),
            tinnitus_avoidance_hz: 500.0,
            use_log_mapping: true,
        }
    }

    /// Configure from a hearing profile: shrink target range to audible
    /// frequencies and register tinnitus avoidance bands.
    pub fn configure_for_profile(&mut self, profile: &HearingProfile) {
        let (lo, hi) = profile.effective_frequency_range();
        if hi > lo {
            self.target_min_hz = lo.max(100.0);
            self.target_max_hz = hi.min(12000.0);
        }
        self.tinnitus_frequencies = profile.tinnitus_frequencies.clone();
    }

    pub fn set_tinnitus_avoidance_band(&mut self, half_bandwidth_hz: f64) {
        self.tinnitus_avoidance_hz = half_bandwidth_hz.abs();
    }

    pub fn set_log_mapping(&mut self, enabled: bool) {
        self.use_log_mapping = enabled;
    }

    pub fn target_range(&self) -> (f64, f64) {
        (self.target_min_hz, self.target_max_hz)
    }

    /// Map a source frequency to the target range, avoiding tinnitus.
    pub fn remap(&self, freq_hz: f64) -> f64 {
        let mapped = if self.use_log_mapping {
            self.log_remap(freq_hz)
        } else {
            self.linear_remap(freq_hz)
        };
        self.avoid_tinnitus(mapped)
    }

    fn linear_remap(&self, freq: f64) -> f64 {
        let src_range = self.source_max_hz - self.source_min_hz;
        if src_range.abs() < 1e-9 {
            return self.target_min_hz;
        }
        let t = ((freq - self.source_min_hz) / src_range).clamp(0.0, 1.0);
        self.target_min_hz + t * (self.target_max_hz - self.target_min_hz)
    }

    fn log_remap(&self, freq: f64) -> f64 {
        let f = freq.max(1.0);
        let src_lo = self.source_min_hz.max(1.0).ln();
        let src_hi = self.source_max_hz.max(1.0).ln();
        let tgt_lo = self.target_min_hz.max(1.0).ln();
        let tgt_hi = self.target_max_hz.max(1.0).ln();
        if (src_hi - src_lo).abs() < 1e-9 {
            return self.target_min_hz;
        }
        let t = ((f.ln() - src_lo) / (src_hi - src_lo)).clamp(0.0, 1.0);
        (tgt_lo + t * (tgt_hi - tgt_lo)).exp()
    }

    fn avoid_tinnitus(&self, freq: f64) -> f64 {
        let mut result = freq;
        for &tf in &self.tinnitus_frequencies {
            let lo = tf - self.tinnitus_avoidance_hz;
            let hi = tf + self.tinnitus_avoidance_hz;
            if result >= lo && result <= hi {
                let dist_lo = (result - lo).abs();
                let dist_hi = (hi - result).abs();
                result = if dist_lo <= dist_hi {
                    (lo - 1.0).max(self.target_min_hz)
                } else {
                    (hi + 1.0).min(self.target_max_hz)
                };
            }
        }
        result
    }

    /// Map an array of frequencies, preserving relative ordering.
    pub fn remap_batch(&self, frequencies: &[f64]) -> Vec<f64> {
        frequencies.iter().map(|&f| self.remap(f)).collect()
    }

    /// Compression ratio (source octaves / target octaves).
    pub fn compression_ratio(&self) -> f64 {
        let src_oct = (self.source_max_hz / self.source_min_hz).log2();
        let tgt_oct = (self.target_max_hz / self.target_min_hz).log2();
        if tgt_oct.abs() < 1e-9 {
            f64::INFINITY
        } else {
            src_oct / tgt_oct
        }
    }
}

// ---------------------------------------------------------------------------
// DynamicRangeAdapter
// ---------------------------------------------------------------------------

/// Adapts the dynamic range to accommodate hearing loss: expansion of quiet
/// sounds, compression, and multi-band processing.
#[derive(Debug, Clone)]
pub struct DynamicRangeAdapter {
    input_min_db: f64,
    input_max_db: f64,
    output_min_db: f64,
    output_max_db: f64,
    compression_ratio: f64,
    threshold_db: f64,
    knee_db: f64,
    bands: Vec<CompressionBand>,
}

/// Per-band compression settings for multi-band dynamics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionBand {
    pub freq_lo_hz: f64,
    pub freq_hi_hz: f64,
    pub ratio: f64,
    pub threshold_db: f64,
    pub gain_db: f64,
}

impl DynamicRangeAdapter {
    pub fn new(input_range: (f64, f64), output_range: (f64, f64)) -> Self {
        Self {
            input_min_db: input_range.0,
            input_max_db: input_range.1,
            output_min_db: output_range.0,
            output_max_db: output_range.1,
            compression_ratio: 1.0,
            threshold_db: -20.0,
            knee_db: 6.0,
            bands: Vec::new(),
        }
    }

    /// Configure from a hearing profile.
    pub fn configure_for_profile(&mut self, profile: &HearingProfile) {
        self.output_min_db = profile.comfortable_min_db;
        self.output_max_db = profile.comfortable_max_db;
        self.compression_ratio = profile.compression_ratio;

        if profile.needs_compression {
            self.bands.clear();
            // Create bands around standard audiometric frequencies.
            let freqs = [250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0];
            for (i, &f) in freqs.iter().enumerate() {
                let lo = if i == 0 { 20.0 } else { (freqs[i - 1] + f) / 2.0 };
                let hi = if i == freqs.len() - 1 {
                    20000.0
                } else {
                    (f + freqs[i + 1]) / 2.0
                };
                let threshold = profile.threshold_at(f);
                let ratio = 1.0 + (threshold / 40.0).clamp(0.0, 3.0);
                let gain = (threshold * 0.5).clamp(0.0, 30.0);
                self.bands.push(CompressionBand {
                    freq_lo_hz: lo,
                    freq_hi_hz: hi,
                    ratio,
                    threshold_db: -20.0,
                    gain_db: gain,
                });
            }
        }
    }

    pub fn set_compression(&mut self, ratio: f64, threshold_db: f64) {
        self.compression_ratio = ratio.max(1.0);
        self.threshold_db = threshold_db;
    }

    pub fn set_knee(&mut self, knee_db: f64) {
        self.knee_db = knee_db.max(0.0);
    }

    /// Apply broadband compression to a dB value.
    pub fn compress(&self, input_db: f64) -> f64 {
        let above = input_db - self.threshold_db;
        let compressed_db = if above <= -self.knee_db / 2.0 {
            input_db
        } else if above >= self.knee_db / 2.0 {
            self.threshold_db + above / self.compression_ratio
        } else {
            // Soft knee
            let x = above + self.knee_db / 2.0;
            input_db + (1.0 / self.compression_ratio - 1.0) * x * x
                / (2.0 * self.knee_db)
        };
        self.map_to_output(compressed_db)
    }

    /// Simple linear mapping of a compressed dB value into the output range.
    fn map_to_output(&self, db: f64) -> f64 {
        let in_range = self.input_max_db - self.input_min_db;
        if in_range.abs() < 1e-9 {
            return self.output_min_db;
        }
        let t = ((db - self.input_min_db) / in_range).clamp(0.0, 1.0);
        self.output_min_db + t * (self.output_max_db - self.output_min_db)
    }

    /// Get the per-band gain to apply for a given frequency.
    pub fn band_gain_db(&self, freq_hz: f64) -> f64 {
        for band in &self.bands {
            if freq_hz >= band.freq_lo_hz && freq_hz < band.freq_hi_hz {
                return band.gain_db;
            }
        }
        0.0
    }

    pub fn bands(&self) -> &[CompressionBand] {
        &self.bands
    }

    pub fn output_range(&self) -> (f64, f64) {
        (self.output_min_db, self.output_max_db)
    }
}

// ---------------------------------------------------------------------------
// TemporalAdapter
// ---------------------------------------------------------------------------

/// Adapts the temporal characteristics of sonification for listeners who need
/// slower or more deliberate presentations.
#[derive(Debug, Clone)]
pub struct TemporalAdapter {
    speed_factor: f64,
    envelope_scale: f64,
    min_gap_seconds: f64,
    repeat_count: u32,
    repeat_delay_seconds: f64,
}

impl TemporalAdapter {
    pub fn new() -> Self {
        Self {
            speed_factor: 1.0,
            envelope_scale: 1.0,
            min_gap_seconds: 0.0,
            repeat_count: 1,
            repeat_delay_seconds: 0.5,
        }
    }

    /// Slow down playback.
    pub fn set_speed_factor(&mut self, factor: f64) {
        self.speed_factor = factor.clamp(0.1, 5.0);
    }

    /// Scale attack/decay/release envelopes.
    pub fn set_envelope_scale(&mut self, scale: f64) {
        self.envelope_scale = scale.clamp(0.5, 10.0);
    }

    /// Minimum gap between sonification events.
    pub fn set_min_gap(&mut self, seconds: f64) {
        self.min_gap_seconds = seconds.max(0.0);
    }

    /// Number of times an important event is repeated.
    pub fn set_repeat(&mut self, count: u32, delay_seconds: f64) {
        self.repeat_count = count.max(1);
        self.repeat_delay_seconds = delay_seconds.max(0.0);
    }

    pub fn speed_factor(&self) -> f64 {
        self.speed_factor
    }

    /// Adapt a duration in seconds.
    pub fn adapt_duration(&self, original_seconds: f64) -> f64 {
        (original_seconds / self.speed_factor).max(0.01)
    }

    /// Adapt an envelope time (attack, decay, release) in seconds.
    pub fn adapt_envelope(&self, original_seconds: f64) -> f64 {
        (original_seconds * self.envelope_scale).max(0.001)
    }

    /// Compute the gap to insert after an event.
    pub fn gap_after_event(&self, original_gap: f64) -> f64 {
        (original_gap / self.speed_factor).max(self.min_gap_seconds)
    }

    /// Generate a schedule of repetitions for an important event: returns
    /// offsets in seconds at which the event should be (re)played.
    pub fn repetition_offsets(&self) -> Vec<f64> {
        (0..self.repeat_count)
            .map(|i| i as f64 * self.repeat_delay_seconds)
            .collect()
    }

    pub fn repeat_count(&self) -> u32 {
        self.repeat_count
    }
}

impl Default for TemporalAdapter {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// SpatialAdapter
// ---------------------------------------------------------------------------

/// Remaps spatial/stereo information for listeners with single-sided deafness
/// or specific spatial preferences.
#[derive(Debug, Clone)]
pub struct SpatialAdapter {
    mode: SpatialMode,
    stereo_width: f64,
    enhance_factor: f64,
}

/// Spatial output mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpatialMode {
    /// Normal stereo output.
    Stereo,
    /// Sum to mono (both channels identical).
    Mono,
    /// Enhanced spatial cues (exaggerated panning).
    Enhanced,
    /// Swap left/right channels.
    Swapped,
}

impl SpatialAdapter {
    pub fn new(mode: SpatialMode) -> Self {
        Self {
            mode,
            stereo_width: 1.0,
            enhance_factor: 1.5,
        }
    }

    pub fn mode(&self) -> SpatialMode {
        self.mode
    }

    pub fn set_mode(&mut self, mode: SpatialMode) {
        self.mode = mode;
    }

    pub fn set_stereo_width(&mut self, width: f64) {
        self.stereo_width = width.clamp(0.0, 2.0);
    }

    pub fn set_enhance_factor(&mut self, factor: f64) {
        self.enhance_factor = factor.clamp(1.0, 3.0);
    }

    /// Adapt a pan value (−1.0 = left, 0.0 = center, 1.0 = right).
    pub fn adapt_pan(&self, pan: f64) -> f64 {
        match self.mode {
            SpatialMode::Mono => 0.0,
            SpatialMode::Stereo => (pan * self.stereo_width).clamp(-1.0, 1.0),
            SpatialMode::Enhanced => (pan * self.enhance_factor).clamp(-1.0, 1.0),
            SpatialMode::Swapped => (-pan).clamp(-1.0, 1.0),
        }
    }

    /// Adapt a stereo sample pair (left, right).
    pub fn adapt_stereo(&self, left: f32, right: f32) -> (f32, f32) {
        match self.mode {
            SpatialMode::Mono => {
                let mono = (left + right) * 0.5;
                (mono, mono)
            }
            SpatialMode::Stereo => {
                let mid = (left + right) * 0.5;
                let side = (left - right) * 0.5;
                let w = self.stereo_width as f32;
                (mid + side * w, mid - side * w)
            }
            SpatialMode::Enhanced => {
                let mid = (left + right) * 0.5;
                let side = (left - right) * 0.5;
                let e = self.enhance_factor as f32;
                ((mid + side * e).clamp(-1.0, 1.0), (mid - side * e).clamp(-1.0, 1.0))
            }
            SpatialMode::Swapped => (right, left),
        }
    }

    /// Configure from hearing profile — for single-sided deafness, use mono.
    pub fn configure_for_profile(&mut self, profile: &HearingProfile) {
        if profile.name.contains("Single-Sided") {
            self.mode = SpatialMode::Mono;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hearing_profile::{HearingProfile, HearingProfilePreset};

    #[test]
    fn freq_remapper_identity() {
        let fr = FrequencyRemapper::new(200.0, 2000.0);
        let m = fr.remap(1000.0);
        assert!(m >= 200.0 && m <= 2000.0);
    }

    #[test]
    fn freq_remapper_log_endpoints() {
        let fr = FrequencyRemapper::new(200.0, 2000.0);
        let lo = fr.remap(200.0);
        let hi = fr.remap(2000.0);
        assert!((lo - 200.0).abs() < 5.0);
        assert!((hi - 2000.0).abs() < 5.0);
    }

    #[test]
    fn freq_remapper_profile() {
        let mut fr = FrequencyRemapper::new(100.0, 10000.0);
        let profile = HearingProfilePreset::MildHighFrequencyLoss.to_profile();
        fr.configure_for_profile(&profile);
        let (lo, hi) = fr.target_range();
        assert!(lo >= 100.0);
        assert!(hi <= 12000.0);
    }

    #[test]
    fn freq_remapper_tinnitus_avoidance() {
        let mut fr = FrequencyRemapper::new(200.0, 8000.0);
        fr.tinnitus_frequencies = vec![4000.0];
        fr.tinnitus_avoidance_hz = 500.0;
        let mapped = fr.remap(4000.0);
        assert!((mapped - 4000.0).abs() > 100.0);
    }

    #[test]
    fn freq_remapper_batch() {
        let fr = FrequencyRemapper::new(200.0, 2000.0);
        let out = fr.remap_batch(&[200.0, 1000.0, 2000.0]);
        assert_eq!(out.len(), 3);
        assert!(out[0] <= out[1] && out[1] <= out[2]);
    }

    #[test]
    fn dynamic_range_compress_below_threshold() {
        let dra = DynamicRangeAdapter::new((-60.0, 0.0), (30.0, 90.0));
        let out = dra.compress(-40.0);
        assert!(out >= 30.0 && out <= 90.0);
    }

    #[test]
    fn dynamic_range_compress_above_threshold() {
        let mut dra = DynamicRangeAdapter::new((-60.0, 0.0), (30.0, 90.0));
        dra.set_compression(4.0, -20.0);
        let loud = dra.compress(-5.0);
        let quiet = dra.compress(-50.0);
        assert!(loud > quiet);
    }

    #[test]
    fn dynamic_range_profile_bands() {
        let mut dra = DynamicRangeAdapter::new((-60.0, 0.0), (30.0, 90.0));
        let profile = HearingProfilePreset::ModerateLoss.to_profile();
        dra.configure_for_profile(&profile);
        assert!(!dra.bands().is_empty());
    }

    #[test]
    fn dynamic_range_band_gain() {
        let mut dra = DynamicRangeAdapter::new((-60.0, 0.0), (30.0, 90.0));
        let profile = HearingProfilePreset::SevereLoss.to_profile();
        dra.configure_for_profile(&profile);
        let gain = dra.band_gain_db(4000.0);
        assert!(gain > 0.0);
    }

    #[test]
    fn temporal_adapter_slow_down() {
        let mut ta = TemporalAdapter::new();
        ta.set_speed_factor(0.5);
        assert!((ta.adapt_duration(1.0) - 2.0).abs() < 0.01);
    }

    #[test]
    fn temporal_adapter_envelope() {
        let mut ta = TemporalAdapter::new();
        ta.set_envelope_scale(2.0);
        assert!((ta.adapt_envelope(0.1) - 0.2).abs() < 0.001);
    }

    #[test]
    fn temporal_adapter_repetitions() {
        let mut ta = TemporalAdapter::new();
        ta.set_repeat(3, 0.25);
        let offsets = ta.repetition_offsets();
        assert_eq!(offsets.len(), 3);
        assert!((offsets[2] - 0.5).abs() < 0.001);
    }

    #[test]
    fn spatial_adapter_mono() {
        let sa = SpatialAdapter::new(SpatialMode::Mono);
        assert_eq!(sa.adapt_pan(0.7), 0.0);
        let (l, r) = sa.adapt_stereo(0.8, 0.2);
        assert!((l - r).abs() < 1e-6);
    }

    #[test]
    fn spatial_adapter_swapped() {
        let sa = SpatialAdapter::new(SpatialMode::Swapped);
        let (l, r) = sa.adapt_stereo(1.0, 0.0);
        assert!((l - 0.0).abs() < 1e-6);
        assert!((r - 1.0).abs() < 1e-6);
    }

    #[test]
    fn spatial_adapter_enhanced_pan() {
        let sa = SpatialAdapter::new(SpatialMode::Enhanced);
        let adapted = sa.adapt_pan(0.5);
        assert!(adapted.abs() >= 0.5);
    }

    #[test]
    fn spatial_adapter_profile() {
        let mut sa = SpatialAdapter::new(SpatialMode::Stereo);
        let profile = HearingProfilePreset::SingleSidedDeafness.to_profile();
        sa.configure_for_profile(&profile);
        assert_eq!(sa.mode(), SpatialMode::Mono);
    }
}
