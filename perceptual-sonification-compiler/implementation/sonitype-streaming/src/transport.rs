//! Transport control for playback, recording, and timeline management.
//!
//! Provides play/pause/stop/record controls, position tracking in multiple
//! time formats, tempo and time-signature support, loop points, and a
//! timeline abstraction for mapping between absolute time and data position.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

// ---------------------------------------------------------------------------
// TransportState
// ---------------------------------------------------------------------------

/// High-level transport state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TransportState {
    Stopped,
    Playing,
    Paused,
    Recording,
}

impl Default for TransportState {
    fn default() -> Self {
        Self::Stopped
    }
}

// ---------------------------------------------------------------------------
// TransportEvent
// ---------------------------------------------------------------------------

/// Events that can be emitted or received by the transport controller.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TransportEvent {
    Play,
    Pause,
    Stop,
    Record,
    Seek { position_seconds: f64 },
    TempoChange { bpm: f64 },
    TimeSignatureChange { numerator: u8, denominator: u8 },
    LoopToggle { enabled: bool },
    LoopSet { start_seconds: f64, end_seconds: f64 },
    PlaybackSpeedChange { speed: f64 },
}

// ---------------------------------------------------------------------------
// TimePosition — multi-format position
// ---------------------------------------------------------------------------

/// A position expressed in multiple units.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TimePosition {
    pub samples: u64,
    pub seconds: f64,
    pub bar: u32,
    pub beat: u32,
    pub tick: u32,
}

impl Default for TimePosition {
    fn default() -> Self {
        Self {
            samples: 0,
            seconds: 0.0,
            bar: 1,
            beat: 1,
            tick: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// TimeSignature
// ---------------------------------------------------------------------------

/// Musical time signature.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TimeSignature {
    pub numerator: u8,
    pub denominator: u8,
}

impl Default for TimeSignature {
    fn default() -> Self {
        Self {
            numerator: 4,
            denominator: 4,
        }
    }
}

impl TimeSignature {
    pub fn new(numerator: u8, denominator: u8) -> Self {
        Self {
            numerator: numerator.max(1),
            denominator: denominator.max(1),
        }
    }

    /// Beats per bar.
    pub fn beats_per_bar(&self) -> u8 {
        self.numerator
    }

    /// Duration of one bar in seconds at a given tempo.
    pub fn bar_duration_seconds(&self, bpm: f64) -> f64 {
        let beat_duration = 60.0 / bpm;
        beat_duration * self.numerator as f64
    }
}

// ---------------------------------------------------------------------------
// LoopRegion
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LoopRegion {
    pub start_seconds: f64,
    pub end_seconds: f64,
}

impl LoopRegion {
    pub fn new(start: f64, end: f64) -> Self {
        let (s, e) = if start <= end {
            (start, end)
        } else {
            (end, start)
        };
        Self {
            start_seconds: s.max(0.0),
            end_seconds: e.max(0.0),
        }
    }

    pub fn duration(&self) -> f64 {
        self.end_seconds - self.start_seconds
    }

    pub fn contains(&self, seconds: f64) -> bool {
        seconds >= self.start_seconds && seconds <= self.end_seconds
    }
}

// ---------------------------------------------------------------------------
// TransportController
// ---------------------------------------------------------------------------

/// Central transport controller with play/pause/stop/record, position tracking,
/// tempo, time-signature, loop points, and scrubbing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransportController {
    state: TransportState,
    position_samples: u64,
    sample_rate: u32,
    bpm: f64,
    time_signature: TimeSignature,
    loop_enabled: bool,
    loop_region: Option<LoopRegion>,
    playback_speed: f64,
    scrub_position: Option<f64>,
    ticks_per_beat: u32,
    event_log: Vec<TransportEvent>,
}

impl TransportController {
    pub fn new(sample_rate: u32) -> Self {
        Self {
            state: TransportState::Stopped,
            position_samples: 0,
            sample_rate: sample_rate.max(1),
            bpm: 120.0,
            time_signature: TimeSignature::default(),
            loop_enabled: false,
            loop_region: None,
            playback_speed: 1.0,
            scrub_position: None,
            ticks_per_beat: 960,
            event_log: Vec::new(),
        }
    }

    // -- State transitions --------------------------------------------------

    pub fn state(&self) -> TransportState {
        self.state
    }

    pub fn play(&mut self) {
        self.state = TransportState::Playing;
        self.scrub_position = None;
        self.event_log.push(TransportEvent::Play);
    }

    pub fn pause(&mut self) {
        if self.state == TransportState::Playing || self.state == TransportState::Recording {
            self.state = TransportState::Paused;
            self.event_log.push(TransportEvent::Pause);
        }
    }

    pub fn stop(&mut self) {
        self.state = TransportState::Stopped;
        self.position_samples = 0;
        self.scrub_position = None;
        self.event_log.push(TransportEvent::Stop);
    }

    pub fn record(&mut self) {
        self.state = TransportState::Recording;
        self.event_log.push(TransportEvent::Record);
    }

    pub fn is_playing(&self) -> bool {
        self.state == TransportState::Playing
    }

    pub fn is_recording(&self) -> bool {
        self.state == TransportState::Recording
    }

    // -- Position -----------------------------------------------------------

    /// Current position in multiple formats.
    pub fn position(&self) -> TimePosition {
        let seconds = self.position_seconds();
        let beat_duration = 60.0 / self.bpm;
        let total_beats = seconds / beat_duration;
        let beats_per_bar = self.time_signature.numerator as f64;
        let bar = (total_beats / beats_per_bar).floor() as u32 + 1;
        let beat_in_bar = (total_beats % beats_per_bar).floor() as u32 + 1;
        let frac = total_beats.fract();
        let tick = (frac * self.ticks_per_beat as f64).floor() as u32;
        TimePosition {
            samples: self.position_samples,
            seconds,
            bar,
            beat: beat_in_bar,
            tick,
        }
    }

    pub fn position_seconds(&self) -> f64 {
        if let Some(scrub) = self.scrub_position {
            return scrub;
        }
        self.position_samples as f64 / self.sample_rate as f64
    }

    pub fn position_samples(&self) -> u64 {
        self.position_samples
    }

    /// Advance the position by the given number of samples, accounting for
    /// playback speed and loop points.
    pub fn advance(&mut self, samples: u64) {
        if self.state != TransportState::Playing && self.state != TransportState::Recording {
            return;
        }
        let effective = (samples as f64 * self.playback_speed).round() as u64;
        self.position_samples = self.position_samples.wrapping_add(effective);

        if self.loop_enabled {
            if let Some(region) = &self.loop_region {
                let end_sample =
                    (region.end_seconds * self.sample_rate as f64).round() as u64;
                let start_sample =
                    (region.start_seconds * self.sample_rate as f64).round() as u64;
                if self.position_samples >= end_sample && end_sample > start_sample {
                    let overshoot = self.position_samples - end_sample;
                    let loop_len = end_sample - start_sample;
                    self.position_samples = start_sample + (overshoot % loop_len);
                }
            }
        }
    }

    // -- Seek / scrub -------------------------------------------------------

    pub fn seek(&mut self, seconds: f64) {
        let s = seconds.max(0.0);
        self.position_samples = (s * self.sample_rate as f64).round() as u64;
        self.scrub_position = None;
        self.event_log.push(TransportEvent::Seek {
            position_seconds: s,
        });
    }

    pub fn scrub_start(&mut self, seconds: f64) {
        self.scrub_position = Some(seconds.max(0.0));
    }

    pub fn scrub_move(&mut self, seconds: f64) {
        self.scrub_position = Some(seconds.max(0.0));
    }

    pub fn scrub_end(&mut self) {
        if let Some(s) = self.scrub_position.take() {
            self.position_samples = (s * self.sample_rate as f64).round() as u64;
        }
    }

    // -- Tempo & time signature ---------------------------------------------

    pub fn bpm(&self) -> f64 {
        self.bpm
    }

    pub fn set_bpm(&mut self, bpm: f64) {
        self.bpm = bpm.clamp(20.0, 999.0);
        self.event_log.push(TransportEvent::TempoChange { bpm: self.bpm });
    }

    pub fn time_signature(&self) -> TimeSignature {
        self.time_signature
    }

    pub fn set_time_signature(&mut self, numerator: u8, denominator: u8) {
        self.time_signature = TimeSignature::new(numerator, denominator);
        self.event_log.push(TransportEvent::TimeSignatureChange {
            numerator,
            denominator,
        });
    }

    // -- Loop ---------------------------------------------------------------

    pub fn loop_enabled(&self) -> bool {
        self.loop_enabled
    }

    pub fn set_loop_enabled(&mut self, enabled: bool) {
        self.loop_enabled = enabled;
        self.event_log
            .push(TransportEvent::LoopToggle { enabled });
    }

    pub fn set_loop_region(&mut self, start: f64, end: f64) {
        let region = LoopRegion::new(start, end);
        self.event_log.push(TransportEvent::LoopSet {
            start_seconds: region.start_seconds,
            end_seconds: region.end_seconds,
        });
        self.loop_region = Some(region);
    }

    pub fn loop_region(&self) -> Option<&LoopRegion> {
        self.loop_region.as_ref()
    }

    // -- Playback speed -----------------------------------------------------

    pub fn playback_speed(&self) -> f64 {
        self.playback_speed
    }

    pub fn set_playback_speed(&mut self, speed: f64) {
        self.playback_speed = speed.clamp(0.1, 10.0);
        self.event_log.push(TransportEvent::PlaybackSpeedChange {
            speed: self.playback_speed,
        });
    }

    // -- Event log ----------------------------------------------------------

    pub fn drain_events(&mut self) -> Vec<TransportEvent> {
        std::mem::take(&mut self.event_log)
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}

// ---------------------------------------------------------------------------
// Timeline
// ---------------------------------------------------------------------------

/// Mapping between absolute wall-clock time and data position, with support
/// for variable playback speed and bookmarks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Timeline {
    /// Total duration of the underlying data in seconds.
    data_duration_seconds: f64,
    /// Playback speed multiplier.
    playback_speed: f64,
    /// Named bookmarks at specific data-time positions.
    bookmarks: BTreeMap<String, f64>,
    /// Mapping of absolute-time → data-time anchor points for non-linear
    /// playback (e.g. speed ramps). Linearly interpolated between anchors.
    anchors: Vec<(f64, f64)>,
}

impl Timeline {
    pub fn new(data_duration_seconds: f64) -> Self {
        Self {
            data_duration_seconds: data_duration_seconds.max(0.0),
            playback_speed: 1.0,
            bookmarks: BTreeMap::new(),
            anchors: Vec::new(),
        }
    }

    pub fn data_duration(&self) -> f64 {
        self.data_duration_seconds
    }

    pub fn set_playback_speed(&mut self, speed: f64) {
        self.playback_speed = speed.clamp(0.1, 10.0);
    }

    pub fn playback_speed(&self) -> f64 {
        self.playback_speed
    }

    /// Map an absolute wall-clock time to a data-time position.
    pub fn absolute_to_data(&self, absolute_seconds: f64) -> f64 {
        if !self.anchors.is_empty() {
            return self.interpolate_anchors(absolute_seconds);
        }
        let data_time = absolute_seconds * self.playback_speed;
        data_time.clamp(0.0, self.data_duration_seconds)
    }

    /// Map a data-time position back to absolute time.
    pub fn data_to_absolute(&self, data_seconds: f64) -> f64 {
        if !self.anchors.is_empty() {
            return self.reverse_anchors(data_seconds);
        }
        if self.playback_speed.abs() < 1e-12 {
            return 0.0;
        }
        data_seconds / self.playback_speed
    }

    fn interpolate_anchors(&self, abs_t: f64) -> f64 {
        if self.anchors.is_empty() {
            return abs_t * self.playback_speed;
        }
        if abs_t <= self.anchors[0].0 {
            return self.anchors[0].1;
        }
        for w in self.anchors.windows(2) {
            let (a0, d0) = w[0];
            let (a1, d1) = w[1];
            if abs_t >= a0 && abs_t <= a1 {
                let frac = if (a1 - a0).abs() < 1e-12 {
                    0.0
                } else {
                    (abs_t - a0) / (a1 - a0)
                };
                return d0 + frac * (d1 - d0);
            }
        }
        self.anchors.last().unwrap().1
    }

    fn reverse_anchors(&self, data_t: f64) -> f64 {
        if self.anchors.is_empty() {
            return if self.playback_speed.abs() < 1e-12 {
                0.0
            } else {
                data_t / self.playback_speed
            };
        }
        if data_t <= self.anchors[0].1 {
            return self.anchors[0].0;
        }
        for w in self.anchors.windows(2) {
            let (a0, d0) = w[0];
            let (a1, d1) = w[1];
            if data_t >= d0 && data_t <= d1 {
                let frac = if (d1 - d0).abs() < 1e-12 {
                    0.0
                } else {
                    (data_t - d0) / (d1 - d0)
                };
                return a0 + frac * (a1 - a0);
            }
        }
        self.anchors.last().unwrap().0
    }

    /// Add a time-mapping anchor.
    pub fn add_anchor(&mut self, absolute_seconds: f64, data_seconds: f64) {
        self.anchors.push((absolute_seconds, data_seconds));
        self.anchors.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    }

    pub fn clear_anchors(&mut self) {
        self.anchors.clear();
    }

    // -- Bookmarks ----------------------------------------------------------

    pub fn add_bookmark(&mut self, name: impl Into<String>, data_seconds: f64) {
        self.bookmarks
            .insert(name.into(), data_seconds.clamp(0.0, self.data_duration_seconds));
    }

    pub fn remove_bookmark(&mut self, name: &str) -> bool {
        self.bookmarks.remove(name).is_some()
    }

    pub fn bookmark(&self, name: &str) -> Option<f64> {
        self.bookmarks.get(name).copied()
    }

    pub fn bookmarks(&self) -> &BTreeMap<String, f64> {
        &self.bookmarks
    }

    /// Return the next bookmark after `data_seconds`, if any.
    pub fn next_bookmark_after(&self, data_seconds: f64) -> Option<(&str, f64)> {
        self.bookmarks
            .iter()
            .find(|(_, &t)| t > data_seconds)
            .map(|(n, &t)| (n.as_str(), t))
    }

    /// Return the previous bookmark before `data_seconds`, if any.
    pub fn prev_bookmark_before(&self, data_seconds: f64) -> Option<(&str, f64)> {
        self.bookmarks
            .iter()
            .rev()
            .find(|(_, &t)| t < data_seconds)
            .map(|(n, &t)| (n.as_str(), t))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transport_initial_state() {
        let tc = TransportController::new(44100);
        assert_eq!(tc.state(), TransportState::Stopped);
        assert_eq!(tc.position_samples(), 0);
    }

    #[test]
    fn transport_play_pause_stop() {
        let mut tc = TransportController::new(44100);
        tc.play();
        assert!(tc.is_playing());
        tc.pause();
        assert_eq!(tc.state(), TransportState::Paused);
        tc.play();
        tc.stop();
        assert_eq!(tc.state(), TransportState::Stopped);
        assert_eq!(tc.position_samples(), 0);
    }

    #[test]
    fn transport_record() {
        let mut tc = TransportController::new(44100);
        tc.record();
        assert!(tc.is_recording());
    }

    #[test]
    fn transport_advance() {
        let mut tc = TransportController::new(44100);
        tc.play();
        tc.advance(44100);
        assert!((tc.position_seconds() - 1.0).abs() < 0.01);
    }

    #[test]
    fn transport_seek() {
        let mut tc = TransportController::new(44100);
        tc.play();
        tc.seek(2.5);
        assert!((tc.position_seconds() - 2.5).abs() < 0.01);
    }

    #[test]
    fn transport_loop() {
        let mut tc = TransportController::new(100);
        tc.set_loop_region(1.0, 2.0);
        tc.set_loop_enabled(true);
        tc.play();
        tc.seek(1.5);
        tc.advance(100); // +1 second → pos 2.5, should wrap
        let pos = tc.position_seconds();
        assert!(pos >= 1.0 && pos <= 2.0, "pos = {}", pos);
    }

    #[test]
    fn transport_playback_speed() {
        let mut tc = TransportController::new(100);
        tc.set_playback_speed(2.0);
        tc.play();
        tc.advance(100); // 100 samples at 2x speed = 200 effective samples = 2.0s
        assert!((tc.position_seconds() - 2.0).abs() < 0.02);
    }

    #[test]
    fn transport_scrub() {
        let mut tc = TransportController::new(44100);
        tc.scrub_start(3.0);
        assert!((tc.position_seconds() - 3.0).abs() < 0.01);
        tc.scrub_move(5.0);
        assert!((tc.position_seconds() - 5.0).abs() < 0.01);
        tc.scrub_end();
        assert!((tc.position_seconds() - 5.0).abs() < 0.01);
    }

    #[test]
    fn transport_tempo_position() {
        let mut tc = TransportController::new(44100);
        tc.set_bpm(120.0);
        tc.set_time_signature(4, 4);
        tc.play();
        tc.seek(2.0); // at 120bpm, 2s = 4 beats = 1 bar
        let pos = tc.position();
        assert_eq!(pos.bar, 2);
    }

    #[test]
    fn transport_events_drained() {
        let mut tc = TransportController::new(44100);
        tc.play();
        tc.pause();
        let events = tc.drain_events();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0], TransportEvent::Play);
    }

    #[test]
    fn timeline_basic() {
        let tl = Timeline::new(10.0);
        assert!((tl.absolute_to_data(5.0) - 5.0).abs() < 0.01);
    }

    #[test]
    fn timeline_speed() {
        let mut tl = Timeline::new(10.0);
        tl.set_playback_speed(2.0);
        assert!((tl.absolute_to_data(3.0) - 6.0).abs() < 0.01);
    }

    #[test]
    fn timeline_bookmarks() {
        let mut tl = Timeline::new(60.0);
        tl.add_bookmark("intro", 0.0);
        tl.add_bookmark("chorus", 30.0);
        tl.add_bookmark("outro", 55.0);
        assert_eq!(tl.bookmark("chorus"), Some(30.0));
        let next = tl.next_bookmark_after(10.0);
        assert_eq!(next.map(|(n, _)| n), Some("chorus"));
    }

    #[test]
    fn timeline_anchors() {
        let mut tl = Timeline::new(10.0);
        tl.add_anchor(0.0, 0.0);
        tl.add_anchor(5.0, 10.0); // 2x speed for first half
        let data = tl.absolute_to_data(2.5);
        assert!((data - 5.0).abs() < 0.01);
    }

    #[test]
    fn time_signature_bar_duration() {
        let ts = TimeSignature::new(3, 4);
        let dur = ts.bar_duration_seconds(120.0);
        assert!((dur - 1.5).abs() < 0.01); // 3 beats × 0.5s
    }

    #[test]
    fn loop_region_contains() {
        let lr = LoopRegion::new(1.0, 3.0);
        assert!(lr.contains(2.0));
        assert!(!lr.contains(0.5));
        assert!((lr.duration() - 2.0).abs() < 0.01);
    }
}
