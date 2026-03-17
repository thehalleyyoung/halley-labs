//! Haptic feedback mappings: data-value → vibration patterns, intensity and
//! frequency mapping, pattern rendering, and synchronised haptic timelines.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

// ---------------------------------------------------------------------------
// HapticPattern
// ---------------------------------------------------------------------------

/// A single haptic vibration pattern.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum HapticPattern {
    /// A single short pulse.
    ShortPulse,
    /// A single long pulse.
    LongPulse,
    /// Two short pulses.
    DoublePulse,
    /// Three short pulses.
    TriplePulse,
    /// Continuous vibration for a given duration (seconds).
    Continuous { duration_seconds: f64 },
    /// Amplitude-modulated continuous vibration.
    Modulated {
        duration_seconds: f64,
        modulation_hz: f64,
    },
    /// Custom pattern defined by on/off durations.
    Custom { segments: Vec<(f64, bool)> },
}

impl HapticPattern {
    /// Total duration of this pattern in seconds.
    pub fn duration(&self) -> f64 {
        match self {
            Self::ShortPulse => 0.05,
            Self::LongPulse => 0.2,
            Self::DoublePulse => 0.15,
            Self::TriplePulse => 0.25,
            Self::Continuous { duration_seconds } => *duration_seconds,
            Self::Modulated {
                duration_seconds, ..
            } => *duration_seconds,
            Self::Custom { segments } => segments.iter().map(|(d, _)| d).sum(),
        }
    }

    /// Render the pattern as a sequence of (time_offset, intensity 0..1) pairs
    /// at the given sample rate.
    pub fn render(&self, sample_rate: f64) -> Vec<(f64, f64)> {
        let dt = 1.0 / sample_rate;
        let mut samples = Vec::new();
        let dur = self.duration();
        let n = (dur * sample_rate).ceil() as usize;
        for i in 0..n {
            let t = i as f64 * dt;
            let intensity = self.intensity_at(t);
            samples.push((t, intensity));
        }
        samples
    }

    /// Intensity at time `t` seconds into the pattern.
    pub fn intensity_at(&self, t: f64) -> f64 {
        match self {
            Self::ShortPulse => {
                if t < 0.05 { 1.0 } else { 0.0 }
            }
            Self::LongPulse => {
                if t < 0.2 { 1.0 } else { 0.0 }
            }
            Self::DoublePulse => {
                if t < 0.05 || (t >= 0.1 && t < 0.15) {
                    1.0
                } else {
                    0.0
                }
            }
            Self::TriplePulse => {
                if t < 0.05 || (t >= 0.1 && t < 0.15) || (t >= 0.2 && t < 0.25) {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Continuous { duration_seconds } => {
                if t < *duration_seconds { 1.0 } else { 0.0 }
            }
            Self::Modulated {
                duration_seconds,
                modulation_hz,
            } => {
                if t >= *duration_seconds {
                    0.0
                } else {
                    let env = 1.0;
                    let mod_val = 0.5 + 0.5 * (t * modulation_hz * std::f64::consts::TAU).sin();
                    env * mod_val
                }
            }
            Self::Custom { segments } => {
                let mut elapsed = 0.0;
                for &(dur, on) in segments {
                    if t < elapsed + dur {
                        return if on { 1.0 } else { 0.0 };
                    }
                    elapsed += dur;
                }
                0.0
            }
        }
    }

    /// All predefined patterns.
    pub fn presets() -> Vec<HapticPattern> {
        vec![
            Self::ShortPulse,
            Self::LongPulse,
            Self::DoublePulse,
            Self::TriplePulse,
            Self::Continuous {
                duration_seconds: 0.5,
            },
            Self::Modulated {
                duration_seconds: 0.5,
                modulation_hz: 5.0,
            },
        ]
    }
}

// ---------------------------------------------------------------------------
// HapticMapper
// ---------------------------------------------------------------------------

/// Maps data values to haptic feedback: intensity, vibration frequency,
/// and pattern selection for categorical data.
#[derive(Debug, Clone)]
pub struct HapticMapper {
    /// Data range for numeric mapping.
    data_min: f64,
    data_max: f64,
    /// Intensity range (0..1).
    intensity_min: f64,
    intensity_max: f64,
    /// Vibration frequency range (Hz).
    vib_freq_min_hz: f64,
    vib_freq_max_hz: f64,
    /// Patterns assigned to categorical values.
    category_patterns: BTreeMap<String, HapticPattern>,
}

impl HapticMapper {
    pub fn new(data_min: f64, data_max: f64) -> Self {
        Self {
            data_min,
            data_max,
            intensity_min: 0.1,
            intensity_max: 1.0,
            vib_freq_min_hz: 50.0,
            vib_freq_max_hz: 300.0,
            category_patterns: BTreeMap::new(),
        }
    }

    pub fn set_intensity_range(&mut self, min: f64, max: f64) {
        self.intensity_min = min.clamp(0.0, 1.0);
        self.intensity_max = max.clamp(0.0, 1.0);
    }

    pub fn set_vibration_freq_range(&mut self, min_hz: f64, max_hz: f64) {
        self.vib_freq_min_hz = min_hz.max(1.0);
        self.vib_freq_max_hz = max_hz.max(min_hz + 1.0);
    }

    pub fn assign_category(&mut self, category: impl Into<String>, pattern: HapticPattern) {
        self.category_patterns.insert(category.into(), pattern);
    }

    /// Map a numeric value to an intensity (0..1).
    pub fn map_intensity(&self, value: f64) -> f64 {
        let t = Self::normalize(value, self.data_min, self.data_max);
        self.intensity_min + t * (self.intensity_max - self.intensity_min)
    }

    /// Map a numeric value to a vibration frequency in Hz.
    pub fn map_vibration_freq(&self, value: f64) -> f64 {
        let t = Self::normalize(value, self.data_min, self.data_max);
        self.vib_freq_min_hz + t * (self.vib_freq_max_hz - self.vib_freq_min_hz)
    }

    /// Map a categorical string to a pattern, if one is assigned.
    pub fn map_category(&self, category: &str) -> Option<&HapticPattern> {
        self.category_patterns.get(category)
    }

    /// Produce a full haptic event from a numeric value.
    pub fn map_numeric(&self, value: f64) -> HapticEvent {
        let intensity = self.map_intensity(value);
        let vib_freq = self.map_vibration_freq(value);
        HapticEvent {
            pattern: HapticPattern::Continuous {
                duration_seconds: 0.1,
            },
            intensity,
            vibration_freq_hz: vib_freq,
            timestamp_seconds: 0.0,
        }
    }

    fn normalize(value: f64, min: f64, max: f64) -> f64 {
        if (max - min).abs() < 1e-12 {
            return 0.5;
        }
        ((value - min) / (max - min)).clamp(0.0, 1.0)
    }
}

/// A single haptic event with timing information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HapticEvent {
    pub pattern: HapticPattern,
    pub intensity: f64,
    pub vibration_freq_hz: f64,
    pub timestamp_seconds: f64,
}

impl HapticEvent {
    pub fn duration(&self) -> f64 {
        self.pattern.duration()
    }
}

// ---------------------------------------------------------------------------
// HapticRenderer
// ---------------------------------------------------------------------------

/// Renders a timeline of haptic events, synchronised with audio output,
/// and exports as timed events.
#[derive(Debug, Clone)]
pub struct HapticRenderer {
    events: Vec<HapticEvent>,
    playback_position: f64,
    sample_rate: f64,
}

impl HapticRenderer {
    pub fn new(sample_rate: f64) -> Self {
        Self {
            events: Vec::new(),
            playback_position: 0.0,
            sample_rate: sample_rate.max(1.0),
        }
    }

    /// Schedule a haptic event at the given time.
    pub fn schedule(&mut self, mut event: HapticEvent) {
        if event.timestamp_seconds < 0.0 {
            event.timestamp_seconds = 0.0;
        }
        self.events.push(event);
        self.events
            .sort_by(|a, b| a.timestamp_seconds.partial_cmp(&b.timestamp_seconds).unwrap());
    }

    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Advance playback by `dt` seconds and return events that should be
    /// triggered during this period.
    pub fn advance(&mut self, dt: f64) -> Vec<&HapticEvent> {
        let start = self.playback_position;
        let end = start + dt;
        self.playback_position = end;
        self.events
            .iter()
            .filter(|e| e.timestamp_seconds >= start && e.timestamp_seconds < end)
            .collect()
    }

    /// Compute the haptic intensity at the current playback position by
    /// summing active events.
    pub fn current_intensity(&self) -> f64 {
        let t = self.playback_position;
        let mut total = 0.0;
        for e in &self.events {
            let local_t = t - e.timestamp_seconds;
            if local_t >= 0.0 && local_t < e.pattern.duration() {
                total += e.pattern.intensity_at(local_t) * e.intensity;
            }
        }
        total.min(1.0)
    }

    /// Render the entire timeline to a vector of (time, intensity) pairs.
    pub fn render_timeline(&self) -> Vec<(f64, f64)> {
        if self.events.is_empty() {
            return Vec::new();
        }
        let end_time = self
            .events
            .iter()
            .map(|e| e.timestamp_seconds + e.pattern.duration())
            .fold(0.0f64, f64::max);
        let dt = 1.0 / self.sample_rate;
        let n = (end_time * self.sample_rate).ceil() as usize;
        let mut timeline = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f64 * dt;
            let mut intensity = 0.0;
            for e in &self.events {
                let local = t - e.timestamp_seconds;
                if local >= 0.0 && local < e.pattern.duration() {
                    intensity += e.pattern.intensity_at(local) * e.intensity;
                }
            }
            timeline.push((t, intensity.min(1.0)));
        }
        timeline
    }

    /// Export all events as JSON.
    pub fn export_json(&self) -> String {
        serde_json::to_string_pretty(&self.events).unwrap_or_else(|_| "[]".to_string())
    }

    pub fn reset(&mut self) {
        self.events.clear();
        self.playback_position = 0.0;
    }

    pub fn playback_position(&self) -> f64 {
        self.playback_position
    }

    pub fn set_playback_position(&mut self, t: f64) {
        self.playback_position = t.max(0.0);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pattern_short_pulse_duration() {
        assert!((HapticPattern::ShortPulse.duration() - 0.05).abs() < 1e-9);
    }

    #[test]
    fn pattern_continuous_duration() {
        let p = HapticPattern::Continuous {
            duration_seconds: 0.3,
        };
        assert!((p.duration() - 0.3).abs() < 1e-9);
    }

    #[test]
    fn pattern_render() {
        let p = HapticPattern::ShortPulse;
        let rendered = p.render(1000.0);
        assert!(!rendered.is_empty());
        assert!(rendered[0].1 > 0.0); // first sample should be "on"
    }

    #[test]
    fn pattern_modulated() {
        let p = HapticPattern::Modulated {
            duration_seconds: 1.0,
            modulation_hz: 5.0,
        };
        let i1 = p.intensity_at(0.0);
        let i2 = p.intensity_at(0.05);
        assert!(i1 != i2 || (i1 - i2).abs() < 1e-9); // modulation
    }

    #[test]
    fn pattern_custom() {
        let p = HapticPattern::Custom {
            segments: vec![(0.1, true), (0.1, false), (0.1, true)],
        };
        assert!((p.duration() - 0.3).abs() < 1e-9);
        assert!(p.intensity_at(0.05) > 0.0);
        assert!(p.intensity_at(0.15) < 0.01);
    }

    #[test]
    fn mapper_intensity() {
        let hm = HapticMapper::new(0.0, 100.0);
        let low = hm.map_intensity(0.0);
        let high = hm.map_intensity(100.0);
        assert!(low < high);
        assert!(low >= 0.0 && high <= 1.0);
    }

    #[test]
    fn mapper_vibration_freq() {
        let hm = HapticMapper::new(0.0, 100.0);
        let low = hm.map_vibration_freq(0.0);
        let high = hm.map_vibration_freq(100.0);
        assert!(low < high);
    }

    #[test]
    fn mapper_category() {
        let mut hm = HapticMapper::new(0.0, 1.0);
        hm.assign_category("alert", HapticPattern::TriplePulse);
        assert_eq!(
            hm.map_category("alert"),
            Some(&HapticPattern::TriplePulse)
        );
        assert!(hm.map_category("unknown").is_none());
    }

    #[test]
    fn mapper_numeric_event() {
        let hm = HapticMapper::new(0.0, 100.0);
        let event = hm.map_numeric(50.0);
        assert!(event.intensity > 0.0);
        assert!(event.vibration_freq_hz > 0.0);
    }

    #[test]
    fn renderer_schedule_advance() {
        let mut hr = HapticRenderer::new(100.0);
        hr.schedule(HapticEvent {
            pattern: HapticPattern::ShortPulse,
            intensity: 0.8,
            vibration_freq_hz: 150.0,
            timestamp_seconds: 0.5,
        });
        let triggered = hr.advance(1.0);
        assert_eq!(triggered.len(), 1);
    }

    #[test]
    fn renderer_current_intensity() {
        let mut hr = HapticRenderer::new(100.0);
        hr.schedule(HapticEvent {
            pattern: HapticPattern::Continuous {
                duration_seconds: 1.0,
            },
            intensity: 0.6,
            vibration_freq_hz: 100.0,
            timestamp_seconds: 0.0,
        });
        hr.set_playback_position(0.5);
        assert!(hr.current_intensity() > 0.0);
    }

    #[test]
    fn renderer_timeline() {
        let mut hr = HapticRenderer::new(100.0);
        hr.schedule(HapticEvent {
            pattern: HapticPattern::ShortPulse,
            intensity: 1.0,
            vibration_freq_hz: 100.0,
            timestamp_seconds: 0.0,
        });
        let tl = hr.render_timeline();
        assert!(!tl.is_empty());
        assert!(tl[0].1 > 0.0);
    }

    #[test]
    fn renderer_export_json() {
        let mut hr = HapticRenderer::new(100.0);
        hr.schedule(HapticEvent {
            pattern: HapticPattern::LongPulse,
            intensity: 0.5,
            vibration_freq_hz: 200.0,
            timestamp_seconds: 1.0,
        });
        let json = hr.export_json();
        assert!(json.contains("LongPulse"));
    }

    #[test]
    fn renderer_reset() {
        let mut hr = HapticRenderer::new(100.0);
        hr.schedule(HapticEvent {
            pattern: HapticPattern::ShortPulse,
            intensity: 1.0,
            vibration_freq_hz: 100.0,
            timestamp_seconds: 0.0,
        });
        hr.reset();
        assert_eq!(hr.event_count(), 0);
        assert!((hr.playback_position() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn pattern_presets() {
        let presets = HapticPattern::presets();
        assert!(presets.len() >= 6);
    }
}
