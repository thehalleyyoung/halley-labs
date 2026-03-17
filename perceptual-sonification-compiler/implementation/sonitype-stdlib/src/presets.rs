//! Preset sonification recipes for SoniType.
//!
//! Ready-to-use configurations for common sonification tasks: time series,
//! categorical data, scatter plots, histograms, correlations, alerts, and
//! navigation feedback.

use std::collections::HashMap;

use crate::mappings::{
    PitchMapping, LoudnessMapping, TemporalMapping, TimbreMapping,
    SpatialMapping, CompositeMapping, MappingBuilder,
    Polarity, LoudnessCurve, LoudnessUnit, TemporalMode,
};
use crate::scales::{NoteName, ScaleType};
use crate::timbres::{TimbrePalette, TimbreDescriptor};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn clamp(v: f64, lo: f64, hi: f64) -> f64 {
    v.max(lo).min(hi)
}

// ---------------------------------------------------------------------------
// TimeSeriesPreset
// ---------------------------------------------------------------------------

/// How time-series streams are rendered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeSeriesMode {
    /// Single variable: pitch tracks value over time.
    SingleStream,
    /// Multiple variables: parallel streams with different timbres.
    MultiStream,
    /// Anomaly detection: change timbre/loudness on anomalies.
    AnomalyHighlight,
}

/// Configuration for time-series sonification.
#[derive(Debug, Clone)]
pub struct TimeSeriesPreset {
    pub mode: TimeSeriesMode,
    pub pitch_range: (f64, f64),
    pub bpm: f64,
    pub data_range: (f64, f64),
    pub num_streams: usize,
    /// Threshold (in standard deviations) for anomaly highlighting.
    pub anomaly_threshold_std: f64,
    palette: TimbrePalette,
}

impl TimeSeriesPreset {
    pub fn single_stream(data_range: (f64, f64)) -> Self {
        Self {
            mode: TimeSeriesMode::SingleStream,
            pitch_range: (200.0, 2000.0),
            bpm: 120.0,
            data_range,
            num_streams: 1,
            anomaly_threshold_std: 2.0,
            palette: TimbrePalette::palette_8(),
        }
    }

    pub fn multi_stream(num_streams: usize, data_range: (f64, f64)) -> Self {
        Self {
            mode: TimeSeriesMode::MultiStream,
            pitch_range: (200.0, 2000.0),
            bpm: 120.0,
            data_range,
            num_streams,
            anomaly_threshold_std: 2.0,
            palette: TimbrePalette::palette_16(),
        }
    }

    pub fn anomaly_highlight(data_range: (f64, f64), threshold_std: f64) -> Self {
        Self {
            mode: TimeSeriesMode::AnomalyHighlight,
            pitch_range: (200.0, 2000.0),
            bpm: 120.0,
            data_range,
            num_streams: 1,
            anomaly_threshold_std: threshold_std,
            palette: TimbrePalette::palette_8(),
        }
    }

    pub fn with_pitch_range(mut self, min: f64, max: f64) -> Self {
        self.pitch_range = (min, max);
        self
    }

    pub fn with_bpm(mut self, bpm: f64) -> Self {
        self.bpm = bpm;
        self
    }

    /// Build the composite mapping for this preset.
    pub fn build_mapping(&self) -> CompositeMapping {
        let pitch = PitchMapping::continuous(
            self.pitch_range.0, self.pitch_range.1, self.data_range,
        );
        let loudness = LoudnessMapping::linear_db(-40.0, -6.0, self.data_range);
        CompositeMapping::new()
            .add_pitch("value", pitch, 10)
            .add_loudness("intensity", loudness, 8)
    }

    /// Check whether a data point is anomalous given a distribution.
    pub fn is_anomaly(&self, value: f64, mean: f64, std_dev: f64) -> bool {
        if std_dev < 1e-12 {
            return false;
        }
        ((value - mean) / std_dev).abs() > self.anomaly_threshold_std
    }

    /// Get the timbre palette used by this preset.
    pub fn palette(&self) -> &TimbrePalette {
        &self.palette
    }
}

// ---------------------------------------------------------------------------
// CategoricalPreset
// ---------------------------------------------------------------------------

/// Configuration for categorical data sonification.
#[derive(Debug, Clone)]
pub struct CategoricalPreset {
    pub categories: Vec<String>,
    pub palette: TimbrePalette,
    pub base_pitch_hz: f64,
    /// Duration per category onset in seconds.
    pub onset_duration_s: f64,
    /// Gap between onsets in seconds.
    pub gap_s: f64,
}

impl CategoricalPreset {
    pub fn new(categories: &[&str]) -> Self {
        let n = categories.len();
        let palette = if n <= 8 {
            TimbrePalette::palette_8()
        } else {
            TimbrePalette::palette_16()
        };
        Self {
            categories: categories.iter().map(|s| s.to_string()).collect(),
            palette,
            base_pitch_hz: 440.0,
            onset_duration_s: 0.3,
            gap_s: 0.1,
        }
    }

    pub fn with_pitch(mut self, hz: f64) -> Self {
        self.base_pitch_hz = hz;
        self
    }

    pub fn with_timing(mut self, onset: f64, gap: f64) -> Self {
        self.onset_duration_s = onset;
        self.gap_s = gap;
        self
    }

    /// Build a timbre mapping for the categories.
    pub fn build_timbre_mapping(&self) -> TimbreMapping {
        let cat_strs: Vec<&str> = self.categories.iter().map(|s| s.as_str()).collect();
        TimbreMapping::auto_assign(&cat_strs, self.palette.clone())
    }

    /// Get the onset time for the i-th category event.
    pub fn event_time(&self, index: usize) -> f64 {
        index as f64 * (self.onset_duration_s + self.gap_s)
    }

    /// Total duration for a given number of category events.
    pub fn total_duration(&self, n_events: usize) -> f64 {
        if n_events == 0 {
            return 0.0;
        }
        n_events as f64 * self.onset_duration_s + (n_events - 1) as f64 * self.gap_s
    }
}

// ---------------------------------------------------------------------------
// ScatterplotPreset
// ---------------------------------------------------------------------------

/// Configuration for 2D scatter-plot sonification.
#[derive(Debug, Clone)]
pub struct ScatterplotPreset {
    pub x_range: (f64, f64),
    pub y_range: (f64, f64),
    /// Duration to scan from x_min to x_max.
    pub scan_duration_s: f64,
    pub pitch_range: (f64, f64),
    /// Whether to use timbre for a third dimension (e.g. colour).
    pub use_timbre_for_color: bool,
    palette: TimbrePalette,
}

impl ScatterplotPreset {
    pub fn new(x_range: (f64, f64), y_range: (f64, f64)) -> Self {
        Self {
            x_range,
            y_range,
            scan_duration_s: 5.0,
            pitch_range: (200.0, 2000.0),
            use_timbre_for_color: false,
            palette: TimbrePalette::palette_8(),
        }
    }

    pub fn with_color_timbre(mut self) -> Self {
        self.use_timbre_for_color = true;
        self
    }

    pub fn with_scan_duration(mut self, s: f64) -> Self {
        self.scan_duration_s = s;
        self
    }

    /// Build a composite mapping: X → time, Y → pitch.
    pub fn build_mapping(&self) -> CompositeMapping {
        let pitch = PitchMapping::continuous(
            self.pitch_range.0, self.pitch_range.1, self.y_range,
        );
        let pan = SpatialMapping::pan(self.x_range);
        CompositeMapping::new()
            .add_pitch("y", pitch, 10)
            .add_spatial("x", pan, 6)
    }

    /// Map an X value to a time position in the scan.
    pub fn x_to_time(&self, x: f64) -> f64 {
        let t = (x - self.x_range.0) / (self.x_range.1 - self.x_range.0);
        clamp(t, 0.0, 1.0) * self.scan_duration_s
    }

    /// Map a Y value to pitch.
    pub fn y_to_pitch(&self, y: f64) -> f64 {
        let pm = PitchMapping::continuous(
            self.pitch_range.0, self.pitch_range.1, self.y_range,
        );
        pm.map(y)
    }
}

// ---------------------------------------------------------------------------
// HistogramPreset
// ---------------------------------------------------------------------------

/// Configuration for histogram/distribution sonification.
#[derive(Debug, Clone)]
pub struct HistogramPreset {
    pub num_bins: usize,
    /// Duration to sweep through all bins.
    pub sweep_duration_s: f64,
    pub pitch_range: (f64, f64),
    /// Whether bins play in parallel or sequentially.
    pub parallel: bool,
}

impl HistogramPreset {
    pub fn new(num_bins: usize) -> Self {
        Self {
            num_bins,
            sweep_duration_s: 3.0,
            pitch_range: (200.0, 2000.0),
            parallel: false,
        }
    }

    pub fn parallel(num_bins: usize) -> Self {
        Self {
            num_bins,
            sweep_duration_s: 3.0,
            pitch_range: (200.0, 2000.0),
            parallel: true,
        }
    }

    /// Assign each bin a pitch.
    pub fn bin_pitches(&self) -> Vec<f64> {
        if self.num_bins == 0 {
            return Vec::new();
        }
        (0..self.num_bins)
            .map(|i| {
                let t = i as f64 / (self.num_bins - 1).max(1) as f64;
                self.pitch_range.0 + t * (self.pitch_range.1 - self.pitch_range.0)
            })
            .collect()
    }

    /// Map bin value (count) to loudness in dBFS.
    pub fn bin_loudness(&self, value: f64, max_value: f64) -> f64 {
        if max_value < 1e-12 {
            return -60.0;
        }
        let t = clamp(value / max_value, 0.0, 1.0);
        -60.0 + t * 54.0 // -60 to -6 dBFS
    }

    /// Time offset for a sequential bin.
    pub fn bin_time(&self, bin_index: usize) -> f64 {
        if self.parallel || self.num_bins <= 1 {
            return 0.0;
        }
        let t = bin_index as f64 / (self.num_bins - 1) as f64;
        t * self.sweep_duration_s
    }
}

// ---------------------------------------------------------------------------
// CorrelationPreset
// ---------------------------------------------------------------------------

/// Configuration for correlation sonification.
#[derive(Debug, Clone)]
pub struct CorrelationPreset {
    /// Base pitch in Hz.
    pub base_pitch_hz: f64,
    /// Duration of the correlation sonification in seconds.
    pub duration_s: f64,
}

impl CorrelationPreset {
    pub fn new() -> Self {
        Self {
            base_pitch_hz: 440.0,
            duration_s: 2.0,
        }
    }

    pub fn with_base_pitch(mut self, hz: f64) -> Self {
        self.base_pitch_hz = hz;
        self
    }

    /// Map a correlation coefficient `r ∈ [-1, 1]` to an interval.
    ///
    /// Returns `(freq_a, freq_b)` where:
    /// - r ≈ +1 → unison or octave (consonant)
    /// - r ≈ 0  → tritone (ambiguous)
    /// - r ≈ -1 → minor second (dissonant)
    pub fn map_correlation(&self, r: f64) -> (f64, f64) {
        let r = clamp(r, -1.0, 1.0);
        let base = self.base_pitch_hz;
        // Map correlation to interval ratio:
        // +1 → 2.0 (octave), 0 → sqrt(2) (tritone), -1 → 16/15 (minor second)
        let ratio = if r >= 0.0 {
            // Blend from tritone (r=0) to octave (r=1)
            let tritone = 2.0_f64.powf(6.0 / 12.0);
            let octave = 2.0;
            tritone + r * (octave - tritone)
        } else {
            // Blend from tritone (r=0) to minor second (r=-1)
            let tritone = 2.0_f64.powf(6.0 / 12.0);
            let minor_second = 2.0_f64.powf(1.0 / 12.0);
            tritone + r.abs() * (minor_second - tritone)
        };
        (base, base * ratio)
    }

    /// Descriptive label for a correlation value.
    pub fn correlation_label(r: f64) -> &'static str {
        let r = clamp(r, -1.0, 1.0);
        if r > 0.7 { "strong positive" }
        else if r > 0.3 { "moderate positive" }
        else if r > -0.3 { "weak / none" }
        else if r > -0.7 { "moderate negative" }
        else { "strong negative" }
    }

    /// Compute Pearson correlation between two equal-length slices.
    pub fn pearson(x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.is_empty() {
            return 0.0;
        }
        let n = x.len() as f64;
        let mean_x = x.iter().sum::<f64>() / n;
        let mean_y = y.iter().sum::<f64>() / n;
        let mut cov = 0.0;
        let mut var_x = 0.0;
        let mut var_y = 0.0;
        for i in 0..x.len() {
            let dx = x[i] - mean_x;
            let dy = y[i] - mean_y;
            cov += dx * dy;
            var_x += dx * dx;
            var_y += dy * dy;
        }
        let denom = (var_x * var_y).sqrt();
        if denom < 1e-12 { 0.0 } else { cov / denom }
    }
}

impl Default for CorrelationPreset {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// AlertPreset
// ---------------------------------------------------------------------------

/// Alert urgency level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum UrgencyLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Configuration for attention-grabbing alert sounds.
#[derive(Debug, Clone)]
pub struct AlertPreset {
    pub urgency: UrgencyLevel,
    /// Number of repetitions.
    pub repetitions: u32,
    /// Inter-onset interval in seconds.
    pub ioi_s: f64,
    /// Pitch in Hz.
    pub pitch_hz: f64,
    /// Duration of each onset in seconds.
    pub onset_duration_s: f64,
    /// Whether to escalate (increase pitch/rate) on each repetition.
    pub escalate: bool,
}

impl AlertPreset {
    pub fn new(urgency: UrgencyLevel) -> Self {
        let (reps, ioi, pitch, dur, esc) = match urgency {
            UrgencyLevel::Low => (1, 0.8, 440.0, 0.3, false),
            UrgencyLevel::Medium => (2, 0.5, 660.0, 0.25, false),
            UrgencyLevel::High => (3, 0.3, 880.0, 0.2, true),
            UrgencyLevel::Critical => (5, 0.15, 1200.0, 0.15, true),
        };
        Self {
            urgency,
            repetitions: reps,
            ioi_s: ioi,
            pitch_hz: pitch,
            onset_duration_s: dur,
            escalate: esc,
        }
    }

    pub fn with_repetitions(mut self, n: u32) -> Self {
        self.repetitions = n;
        self
    }

    pub fn with_escalation(mut self, esc: bool) -> Self {
        self.escalate = esc;
        self
    }

    /// Generate event schedule: `(time_s, pitch_hz, duration_s)`.
    pub fn event_schedule(&self) -> Vec<(f64, f64, f64)> {
        let mut events = Vec::new();
        for i in 0..self.repetitions {
            let t = i as f64 * self.ioi_s;
            let pitch = if self.escalate {
                self.pitch_hz * (1.0 + 0.1 * i as f64)
            } else {
                self.pitch_hz
            };
            let dur = if self.escalate {
                (self.onset_duration_s * (1.0 - 0.05 * i as f64)).max(0.05)
            } else {
                self.onset_duration_s
            };
            events.push((t, pitch, dur));
        }
        events
    }

    /// Total duration of the alert sequence.
    pub fn total_duration(&self) -> f64 {
        if self.repetitions == 0 {
            return 0.0;
        }
        (self.repetitions - 1) as f64 * self.ioi_s + self.onset_duration_s
    }
}

// ---------------------------------------------------------------------------
// NavigationPreset
// ---------------------------------------------------------------------------

/// Navigation sound type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NavigationSound {
    ScrollPosition,
    SelectionConfirm,
    BoundaryNotification,
    FocusChange,
    ErrorFeedback,
}

/// Configuration for data navigation sounds.
#[derive(Debug, Clone)]
pub struct NavigationPreset {
    pub sound: NavigationSound,
    pub pitch_hz: f64,
    pub duration_s: f64,
    pub volume_db: f64,
}

impl NavigationPreset {
    pub fn new(sound: NavigationSound) -> Self {
        let (pitch, dur, vol) = match sound {
            NavigationSound::ScrollPosition => (600.0, 0.05, -20.0),
            NavigationSound::SelectionConfirm => (880.0, 0.15, -12.0),
            NavigationSound::BoundaryNotification => (1200.0, 0.2, -10.0),
            NavigationSound::FocusChange => (660.0, 0.1, -15.0),
            NavigationSound::ErrorFeedback => (200.0, 0.3, -8.0),
        };
        Self { sound, pitch_hz: pitch, duration_s: dur, volume_db: vol }
    }

    /// Map a scroll position `[0, 1]` to pitch for scroll feedback.
    pub fn scroll_pitch(&self, position: f64) -> f64 {
        let pos = clamp(position, 0.0, 1.0);
        200.0 + pos * 1800.0
    }

    /// Whether this sound type should be short (earcon-style).
    pub fn is_earcon(&self) -> bool {
        matches!(self.sound,
            NavigationSound::SelectionConfirm
            | NavigationSound::BoundaryNotification
            | NavigationSound::FocusChange
            | NavigationSound::ErrorFeedback
        )
    }
}

// ---------------------------------------------------------------------------
// PresetRegistry
// ---------------------------------------------------------------------------

/// Registry for looking up and instantiating presets by name.
#[derive(Debug, Clone)]
pub struct PresetRegistry {
    entries: HashMap<String, PresetEntry>,
}

/// Metadata about a registered preset.
#[derive(Debug, Clone)]
pub struct PresetEntry {
    pub name: String,
    pub description: String,
    pub category: PresetCategory,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PresetCategory {
    TimeSeries,
    Categorical,
    Scatterplot,
    Histogram,
    Correlation,
    Alert,
    Navigation,
}

impl PresetRegistry {
    pub fn new() -> Self {
        Self { entries: HashMap::new() }
    }

    /// Create a registry with all built-in presets registered.
    pub fn with_builtins() -> Self {
        let mut reg = Self::new();
        reg.register("time_series_single", "Single-stream time series", PresetCategory::TimeSeries);
        reg.register("time_series_multi", "Multi-stream time series", PresetCategory::TimeSeries);
        reg.register("time_series_anomaly", "Time series with anomaly detection", PresetCategory::TimeSeries);
        reg.register("categorical", "Categorical data sonification", PresetCategory::Categorical);
        reg.register("scatterplot", "2D scatter plot sonification", PresetCategory::Scatterplot);
        reg.register("scatterplot_color", "Scatter plot with colour-to-timbre", PresetCategory::Scatterplot);
        reg.register("histogram_seq", "Sequential histogram sweep", PresetCategory::Histogram);
        reg.register("histogram_par", "Parallel histogram streams", PresetCategory::Histogram);
        reg.register("correlation", "Correlation sonification", PresetCategory::Correlation);
        reg.register("alert_low", "Low-urgency alert", PresetCategory::Alert);
        reg.register("alert_medium", "Medium-urgency alert", PresetCategory::Alert);
        reg.register("alert_high", "High-urgency alert", PresetCategory::Alert);
        reg.register("alert_critical", "Critical alert", PresetCategory::Alert);
        reg.register("nav_scroll", "Scroll position feedback", PresetCategory::Navigation);
        reg.register("nav_select", "Selection confirmation", PresetCategory::Navigation);
        reg.register("nav_boundary", "Boundary notification", PresetCategory::Navigation);
        reg
    }

    pub fn register(&mut self, name: &str, description: &str, category: PresetCategory) {
        self.entries.insert(name.to_string(), PresetEntry {
            name: name.to_string(),
            description: description.to_string(),
            category,
        });
    }

    pub fn get(&self, name: &str) -> Option<&PresetEntry> {
        self.entries.get(name)
    }

    pub fn list(&self) -> Vec<&PresetEntry> {
        self.entries.values().collect()
    }

    pub fn list_by_category(&self, category: PresetCategory) -> Vec<&PresetEntry> {
        self.entries.values().filter(|e| e.category == category).collect()
    }

    pub fn contains(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Default for PresetRegistry {
    fn default() -> Self {
        Self::with_builtins()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_series_single_stream() {
        let ts = TimeSeriesPreset::single_stream((0.0, 100.0));
        assert_eq!(ts.num_streams, 1);
        let mapping = ts.build_mapping();
        assert!(!mapping.layers.is_empty());
    }

    #[test]
    fn test_time_series_multi_stream() {
        let ts = TimeSeriesPreset::multi_stream(4, (0.0, 1.0));
        assert_eq!(ts.num_streams, 4);
    }

    #[test]
    fn test_time_series_anomaly_detection() {
        let ts = TimeSeriesPreset::anomaly_highlight((0.0, 100.0), 2.0);
        assert!(ts.is_anomaly(150.0, 50.0, 20.0)); // 5 std devs
        assert!(!ts.is_anomaly(55.0, 50.0, 20.0));  // 0.25 std devs
    }

    #[test]
    fn test_categorical_preset() {
        let cp = CategoricalPreset::new(&["apple", "banana", "cherry"]);
        assert_eq!(cp.categories.len(), 3);
        let tm = cp.build_timbre_mapping();
        assert!(tm.map("apple").is_some());
    }

    #[test]
    fn test_categorical_timing() {
        let cp = CategoricalPreset::new(&["a", "b"]).with_timing(0.5, 0.2);
        assert!((cp.event_time(0) - 0.0).abs() < 1e-6);
        assert!((cp.event_time(1) - 0.7).abs() < 1e-6);
        assert!((cp.total_duration(2) - 1.2).abs() < 1e-6);
    }

    #[test]
    fn test_scatterplot_preset() {
        let sp = ScatterplotPreset::new((0.0, 10.0), (0.0, 100.0));
        let t = sp.x_to_time(5.0);
        assert!((t - 2.5).abs() < 1e-6);
        let p = sp.y_to_pitch(50.0);
        assert!(p > 200.0 && p < 2000.0);
    }

    #[test]
    fn test_scatterplot_mapping() {
        let sp = ScatterplotPreset::new((0.0, 10.0), (0.0, 100.0));
        let mapping = sp.build_mapping();
        assert_eq!(mapping.layers.len(), 2);
    }

    #[test]
    fn test_histogram_bin_pitches() {
        let hp = HistogramPreset::new(5);
        let pitches = hp.bin_pitches();
        assert_eq!(pitches.len(), 5);
        assert!((pitches[0] - 200.0).abs() < 1e-6);
        assert!((pitches[4] - 2000.0).abs() < 1e-6);
    }

    #[test]
    fn test_histogram_loudness() {
        let hp = HistogramPreset::new(10);
        let l = hp.bin_loudness(50.0, 100.0);
        assert!(l > -60.0 && l < 0.0);
    }

    #[test]
    fn test_histogram_parallel_vs_sequential() {
        let seq = HistogramPreset::new(4);
        let par = HistogramPreset::parallel(4);
        assert!(!seq.parallel);
        assert!(par.parallel);
        assert!(seq.bin_time(2) > 0.0);
        assert!((par.bin_time(2) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_correlation_positive() {
        let cp = CorrelationPreset::new();
        let (a, b) = cp.map_correlation(1.0);
        // Octave interval for perfect positive correlation
        assert!((b / a - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_correlation_negative() {
        let cp = CorrelationPreset::new();
        let (a, b) = cp.map_correlation(-1.0);
        // Minor second for perfect negative correlation
        let ratio = b / a;
        let minor_second = 2.0_f64.powf(1.0 / 12.0);
        assert!((ratio - minor_second).abs() < 1e-4);
    }

    #[test]
    fn test_correlation_label() {
        assert_eq!(CorrelationPreset::correlation_label(0.9), "strong positive");
        assert_eq!(CorrelationPreset::correlation_label(-0.8), "strong negative");
        assert_eq!(CorrelationPreset::correlation_label(0.0), "weak / none");
    }

    #[test]
    fn test_pearson_perfect() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let r = CorrelationPreset::pearson(&x, &y);
        assert!((r - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_alert_low() {
        let a = AlertPreset::new(UrgencyLevel::Low);
        assert_eq!(a.repetitions, 1);
        let schedule = a.event_schedule();
        assert_eq!(schedule.len(), 1);
    }

    #[test]
    fn test_alert_critical_escalation() {
        let a = AlertPreset::new(UrgencyLevel::Critical);
        assert!(a.escalate);
        let schedule = a.event_schedule();
        assert!(schedule.len() >= 5);
        // Pitch should increase with each repetition
        assert!(schedule[1].1 > schedule[0].1);
    }

    #[test]
    fn test_alert_total_duration() {
        let a = AlertPreset::new(UrgencyLevel::Medium);
        let dur = a.total_duration();
        assert!(dur > 0.0);
    }

    #[test]
    fn test_navigation_scroll() {
        let nav = NavigationPreset::new(NavigationSound::ScrollPosition);
        let p = nav.scroll_pitch(0.5);
        assert!(p > 200.0 && p < 2000.0);
    }

    #[test]
    fn test_navigation_earcon() {
        let confirm = NavigationPreset::new(NavigationSound::SelectionConfirm);
        assert!(confirm.is_earcon());
        let scroll = NavigationPreset::new(NavigationSound::ScrollPosition);
        assert!(!scroll.is_earcon());
    }

    #[test]
    fn test_preset_registry_builtins() {
        let reg = PresetRegistry::with_builtins();
        assert!(reg.contains("time_series_single"));
        assert!(reg.contains("alert_critical"));
        assert!(!reg.contains("nonexistent"));
        assert!(reg.len() > 10);
    }

    #[test]
    fn test_preset_registry_category_filter() {
        let reg = PresetRegistry::with_builtins();
        let alerts = reg.list_by_category(PresetCategory::Alert);
        assert!(alerts.len() >= 4);
    }

    #[test]
    fn test_preset_registry_custom() {
        let mut reg = PresetRegistry::new();
        reg.register("custom_1", "My custom preset", PresetCategory::TimeSeries);
        assert!(reg.contains("custom_1"));
        assert_eq!(reg.len(), 1);
    }
}
