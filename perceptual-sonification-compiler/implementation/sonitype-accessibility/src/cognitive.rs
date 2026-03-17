//! Cognitive accessibility: simplification modes, guided listening,
//! progressive complexity, attention guides, and memory aids.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// SimplificationMode
// ---------------------------------------------------------------------------

/// How much the sonification is simplified for cognitive accessibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SimplificationMode {
    /// All streams active, no simplification.
    Full,
    /// Reduce to N most important streams.
    Reduced { max_streams: usize },
    /// Only one stream at a time (guided listening).
    Single,
}

// ---------------------------------------------------------------------------
// CognitiveSupportEngine
// ---------------------------------------------------------------------------

/// Manages cognitive accessibility features: simplification, guided listening,
/// progressive complexity, and training mode.
#[derive(Debug, Clone)]
pub struct CognitiveSupportEngine {
    mode: SimplificationMode,
    stream_names: Vec<String>,
    stream_priorities: Vec<u8>,
    active_guide_index: usize,
    progressive_level: usize,
    progressive_max: usize,
    training_mode: bool,
    training_step: usize,
}

impl CognitiveSupportEngine {
    pub fn new() -> Self {
        Self {
            mode: SimplificationMode::Full,
            stream_names: Vec::new(),
            stream_priorities: Vec::new(),
            active_guide_index: 0,
            progressive_level: 0,
            progressive_max: 0,
            training_mode: false,
            training_step: 0,
        }
    }

    /// Register a sonification stream with a priority (higher = more important).
    pub fn register_stream(&mut self, name: impl Into<String>, priority: u8) {
        self.stream_names.push(name.into());
        self.stream_priorities.push(priority);
        self.progressive_max = self.stream_names.len();
    }

    pub fn stream_count(&self) -> usize {
        self.stream_names.len()
    }

    pub fn set_mode(&mut self, mode: SimplificationMode) {
        self.mode = mode;
    }

    pub fn mode(&self) -> SimplificationMode {
        self.mode
    }

    /// Return which streams should be active given the current mode.
    pub fn active_streams(&self) -> Vec<&str> {
        if self.stream_names.is_empty() {
            return Vec::new();
        }
        match self.mode {
            SimplificationMode::Full => {
                if self.progressive_level == 0 || self.progressive_level >= self.stream_names.len()
                {
                    self.stream_names.iter().map(|s| s.as_str()).collect()
                } else {
                    self.top_n_streams(self.progressive_level)
                }
            }
            SimplificationMode::Reduced { max_streams } => self.top_n_streams(max_streams),
            SimplificationMode::Single => {
                let idx = self.active_guide_index % self.stream_names.len();
                vec![self.stream_names[idx].as_str()]
            }
        }
    }

    fn top_n_streams(&self, n: usize) -> Vec<&str> {
        let mut indexed: Vec<(usize, u8)> = self
            .stream_priorities
            .iter()
            .copied()
            .enumerate()
            .collect();
        indexed.sort_by(|a, b| b.1.cmp(&a.1));
        indexed
            .iter()
            .take(n)
            .map(|(i, _)| self.stream_names[*i].as_str())
            .collect()
    }

    // -- Guided listening ---------------------------------------------------

    /// Advance to the next stream in guided-listening mode.
    pub fn guide_next(&mut self) {
        if !self.stream_names.is_empty() {
            self.active_guide_index =
                (self.active_guide_index + 1) % self.stream_names.len();
        }
    }

    /// Go to the previous stream.
    pub fn guide_prev(&mut self) {
        if !self.stream_names.is_empty() {
            self.active_guide_index = if self.active_guide_index == 0 {
                self.stream_names.len() - 1
            } else {
                self.active_guide_index - 1
            };
        }
    }

    /// Set the guided-listening index directly.
    pub fn guide_set(&mut self, index: usize) {
        self.active_guide_index = index;
    }

    pub fn guide_index(&self) -> usize {
        self.active_guide_index
    }

    /// Name of the stream currently highlighted in guided listening.
    pub fn guide_current_name(&self) -> Option<&str> {
        if self.stream_names.is_empty() {
            None
        } else {
            Some(&self.stream_names[self.active_guide_index % self.stream_names.len()])
        }
    }

    // -- Progressive complexity ---------------------------------------------

    /// Set the progressive complexity level (1 = simplest, N = all streams).
    pub fn set_progressive_level(&mut self, level: usize) {
        self.progressive_level = level.min(self.progressive_max);
    }

    /// Increase complexity by one stream.
    pub fn increase_complexity(&mut self) {
        if self.progressive_level < self.progressive_max {
            self.progressive_level += 1;
        }
    }

    /// Decrease complexity by one stream.
    pub fn decrease_complexity(&mut self) {
        if self.progressive_level > 1 {
            self.progressive_level -= 1;
        }
    }

    pub fn progressive_level(&self) -> usize {
        self.progressive_level
    }

    // -- Training mode ------------------------------------------------------

    pub fn enter_training_mode(&mut self) {
        self.training_mode = true;
        self.training_step = 0;
    }

    pub fn exit_training_mode(&mut self) {
        self.training_mode = false;
    }

    pub fn is_training(&self) -> bool {
        self.training_mode
    }

    /// Get the current training step description.
    pub fn training_description(&self) -> String {
        if !self.training_mode || self.stream_names.is_empty() {
            return "Training not active.".to_string();
        }
        let idx = self.training_step % self.stream_names.len();
        format!(
            "Training step {}: listen to the '{}' stream. This stream will play in isolation so you can learn its sound.",
            self.training_step + 1,
            self.stream_names[idx]
        )
    }

    /// Advance to next training step.
    pub fn training_advance(&mut self) -> bool {
        if self.training_step + 1 < self.stream_names.len() {
            self.training_step += 1;
            true
        } else {
            false
        }
    }

    pub fn training_step(&self) -> usize {
        self.training_step
    }

    pub fn training_total_steps(&self) -> usize {
        self.stream_names.len()
    }
}

impl Default for CognitiveSupportEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// AttentionGuide
// ---------------------------------------------------------------------------

/// Provides audio cues (beacons, transition alerts) to guide user attention.
#[derive(Debug, Clone)]
pub struct AttentionGuide {
    /// Frequency of the alert/beacon tone (Hz).
    beacon_freq_hz: f64,
    beacon_duration_seconds: f64,
    pre_event_lead_seconds: f64,
    transition_sound_enabled: bool,
    orientation_interval_seconds: f64,
    events: Vec<AttentionEvent>,
}

/// A scheduled attention event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionEvent {
    pub kind: AttentionEventKind,
    pub timestamp_seconds: f64,
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttentionEventKind {
    /// Audio cue before an important data event.
    PreAlert,
    /// Beacon sound for orientation.
    Beacon,
    /// Transition sound between data regions.
    Transition,
    /// Highlight cue to draw attention to a specific stream.
    Highlight,
}

impl AttentionGuide {
    pub fn new() -> Self {
        Self {
            beacon_freq_hz: 880.0,
            beacon_duration_seconds: 0.15,
            pre_event_lead_seconds: 0.5,
            transition_sound_enabled: true,
            orientation_interval_seconds: 30.0,
            events: Vec::new(),
        }
    }

    pub fn set_beacon_frequency(&mut self, hz: f64) {
        self.beacon_freq_hz = hz.clamp(200.0, 4000.0);
    }

    pub fn set_beacon_duration(&mut self, seconds: f64) {
        self.beacon_duration_seconds = seconds.clamp(0.05, 1.0);
    }

    pub fn set_pre_event_lead(&mut self, seconds: f64) {
        self.pre_event_lead_seconds = seconds.clamp(0.1, 2.0);
    }

    pub fn set_transition_sound(&mut self, enabled: bool) {
        self.transition_sound_enabled = enabled;
    }

    pub fn set_orientation_interval(&mut self, seconds: f64) {
        self.orientation_interval_seconds = seconds.max(5.0);
    }

    /// Schedule a pre-alert before an important event.
    pub fn schedule_pre_alert(&mut self, event_time: f64, message: impl Into<String>) {
        let alert_time = (event_time - self.pre_event_lead_seconds).max(0.0);
        self.events.push(AttentionEvent {
            kind: AttentionEventKind::PreAlert,
            timestamp_seconds: alert_time,
            message: message.into(),
        });
        self.sort_events();
    }

    /// Schedule a beacon at a specific time.
    pub fn schedule_beacon(&mut self, time: f64, message: impl Into<String>) {
        self.events.push(AttentionEvent {
            kind: AttentionEventKind::Beacon,
            timestamp_seconds: time,
            message: message.into(),
        });
        self.sort_events();
    }

    /// Schedule a transition sound.
    pub fn schedule_transition(&mut self, time: f64, message: impl Into<String>) {
        if self.transition_sound_enabled {
            self.events.push(AttentionEvent {
                kind: AttentionEventKind::Transition,
                timestamp_seconds: time,
                message: message.into(),
            });
            self.sort_events();
        }
    }

    /// Generate periodic beacon events for orientation.
    pub fn generate_periodic_beacons(&mut self, total_duration: f64) {
        let mut t = self.orientation_interval_seconds;
        while t < total_duration {
            self.events.push(AttentionEvent {
                kind: AttentionEventKind::Beacon,
                timestamp_seconds: t,
                message: format!("Orientation beacon at {:.1}s", t),
            });
            t += self.orientation_interval_seconds;
        }
        self.sort_events();
    }

    /// Get events in the time window [start, end).
    pub fn events_in_range(&self, start: f64, end: f64) -> Vec<&AttentionEvent> {
        self.events
            .iter()
            .filter(|e| e.timestamp_seconds >= start && e.timestamp_seconds < end)
            .collect()
    }

    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    pub fn beacon_freq_hz(&self) -> f64 {
        self.beacon_freq_hz
    }

    pub fn beacon_duration_seconds(&self) -> f64 {
        self.beacon_duration_seconds
    }

    pub fn clear(&mut self) {
        self.events.clear();
    }

    fn sort_events(&mut self) {
        self.events
            .sort_by(|a, b| a.timestamp_seconds.partial_cmp(&b.timestamp_seconds).unwrap());
    }
}

impl Default for AttentionGuide {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// MemoryAid
// ---------------------------------------------------------------------------

/// Provides periodic reminders, reference tones, and replay functionality
/// to help listeners remember sonification mappings.
#[derive(Debug, Clone)]
pub struct MemoryAid {
    reminder_interval_seconds: f64,
    legend_text: String,
    reference_tones: Vec<ReferenceTone>,
    replay_buffer_seconds: f64,
    replay_history: VecDeque<ReplaySegment>,
    max_replay_segments: usize,
}

/// A reference tone the listener can request for scale orientation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceTone {
    pub label: String,
    pub frequency_hz: f64,
    pub amplitude: f64,
    pub duration_seconds: f64,
}

/// A segment of audio available for replay.
#[derive(Debug, Clone)]
pub struct ReplaySegment {
    pub start_seconds: f64,
    pub duration_seconds: f64,
    pub description: String,
}

impl MemoryAid {
    pub fn new(reminder_interval_seconds: f64) -> Self {
        Self {
            reminder_interval_seconds: reminder_interval_seconds.max(5.0),
            legend_text: String::new(),
            reference_tones: Vec::new(),
            replay_buffer_seconds: 30.0,
            replay_history: VecDeque::new(),
            max_replay_segments: 10,
        }
    }

    pub fn set_legend_text(&mut self, text: impl Into<String>) {
        self.legend_text = text.into();
    }

    pub fn legend_text(&self) -> &str {
        &self.legend_text
    }

    pub fn set_replay_buffer(&mut self, seconds: f64) {
        self.replay_buffer_seconds = seconds.max(1.0);
    }

    /// Add a reference tone for scale orientation.
    pub fn add_reference_tone(&mut self, tone: ReferenceTone) {
        self.reference_tones.push(tone);
    }

    pub fn reference_tones(&self) -> &[ReferenceTone] {
        &self.reference_tones
    }

    /// Should a reminder be displayed at time `t`?
    pub fn should_remind(&self, t: f64) -> bool {
        if self.reminder_interval_seconds <= 0.0 {
            return false;
        }
        let n = (t / self.reminder_interval_seconds).floor() as u64;
        let reminder_time = n as f64 * self.reminder_interval_seconds;
        (t - reminder_time).abs() < 0.1
    }

    /// Generate the reminder text.
    pub fn reminder_text(&self) -> String {
        if self.legend_text.is_empty() {
            "Reminder: refer to the sonification legend for mapping details.".to_string()
        } else {
            format!("Reminder: {}", self.legend_text)
        }
    }

    /// Record a replay segment.
    pub fn record_segment(&mut self, segment: ReplaySegment) {
        if self.replay_history.len() >= self.max_replay_segments {
            self.replay_history.pop_front();
        }
        self.replay_history.push_back(segment);
    }

    /// Get the last N seconds of replay segments.
    pub fn recent_segments(&self, last_seconds: f64) -> Vec<&ReplaySegment> {
        self.replay_history
            .iter()
            .filter(|s| s.start_seconds + s.duration_seconds >= self.total_time() - last_seconds)
            .collect()
    }

    fn total_time(&self) -> f64 {
        self.replay_history
            .back()
            .map(|s| s.start_seconds + s.duration_seconds)
            .unwrap_or(0.0)
    }

    pub fn replay_segment_count(&self) -> usize {
        self.replay_history.len()
    }

    pub fn clear_replay(&mut self) {
        self.replay_history.clear();
    }

    /// Set the reminder interval.
    pub fn set_reminder_interval(&mut self, seconds: f64) {
        self.reminder_interval_seconds = seconds.max(1.0);
    }

    pub fn reminder_interval(&self) -> f64 {
        self.reminder_interval_seconds
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cognitive_engine_full_mode() {
        let mut ce = CognitiveSupportEngine::new();
        ce.register_stream("temp", 10);
        ce.register_stream("pressure", 5);
        ce.set_mode(SimplificationMode::Full);
        let active = ce.active_streams();
        assert_eq!(active.len(), 2);
    }

    #[test]
    fn cognitive_engine_reduced() {
        let mut ce = CognitiveSupportEngine::new();
        ce.register_stream("a", 10);
        ce.register_stream("b", 5);
        ce.register_stream("c", 1);
        ce.set_mode(SimplificationMode::Reduced { max_streams: 2 });
        let active = ce.active_streams();
        assert_eq!(active.len(), 2);
        assert!(active.contains(&"a"));
    }

    #[test]
    fn cognitive_engine_single() {
        let mut ce = CognitiveSupportEngine::new();
        ce.register_stream("x", 10);
        ce.register_stream("y", 5);
        ce.set_mode(SimplificationMode::Single);
        let active = ce.active_streams();
        assert_eq!(active.len(), 1);
    }

    #[test]
    fn cognitive_engine_guided_nav() {
        let mut ce = CognitiveSupportEngine::new();
        ce.register_stream("a", 10);
        ce.register_stream("b", 5);
        ce.register_stream("c", 1);
        ce.set_mode(SimplificationMode::Single);
        assert_eq!(ce.guide_current_name(), Some("a"));
        ce.guide_next();
        assert_eq!(ce.guide_current_name(), Some("b"));
        ce.guide_next();
        assert_eq!(ce.guide_current_name(), Some("c"));
        ce.guide_next();
        assert_eq!(ce.guide_current_name(), Some("a")); // wraps
    }

    #[test]
    fn cognitive_engine_progressive() {
        let mut ce = CognitiveSupportEngine::new();
        ce.register_stream("a", 10);
        ce.register_stream("b", 5);
        ce.register_stream("c", 1);
        ce.set_progressive_level(1);
        let active = ce.active_streams();
        assert_eq!(active.len(), 1);
        ce.increase_complexity();
        assert_eq!(ce.progressive_level(), 2);
    }

    #[test]
    fn cognitive_engine_training() {
        let mut ce = CognitiveSupportEngine::new();
        ce.register_stream("temp", 10);
        ce.register_stream("wind", 5);
        ce.enter_training_mode();
        assert!(ce.is_training());
        let desc = ce.training_description();
        assert!(desc.contains("temp"));
        ce.training_advance();
        let desc2 = ce.training_description();
        assert!(desc2.contains("wind"));
    }

    #[test]
    fn attention_guide_pre_alert() {
        let mut ag = AttentionGuide::new();
        ag.set_pre_event_lead(0.5);
        ag.schedule_pre_alert(2.0, "spike coming");
        let events = ag.events_in_range(1.0, 2.0);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].kind, AttentionEventKind::PreAlert);
    }

    #[test]
    fn attention_guide_periodic_beacons() {
        let mut ag = AttentionGuide::new();
        ag.set_orientation_interval(10.0);
        ag.generate_periodic_beacons(35.0);
        assert!(ag.event_count() >= 3);
    }

    #[test]
    fn attention_guide_transition() {
        let mut ag = AttentionGuide::new();
        ag.schedule_transition(5.0, "new region");
        assert_eq!(ag.event_count(), 1);
    }

    #[test]
    fn memory_aid_reminder() {
        let ma = MemoryAid::new(30.0);
        assert!(ma.should_remind(30.0));
        assert!(!ma.should_remind(15.0));
    }

    #[test]
    fn memory_aid_reference_tones() {
        let mut ma = MemoryAid::new(30.0);
        ma.add_reference_tone(ReferenceTone {
            label: "min".into(),
            frequency_hz: 200.0,
            amplitude: 0.5,
            duration_seconds: 0.5,
        });
        assert_eq!(ma.reference_tones().len(), 1);
    }

    #[test]
    fn memory_aid_replay() {
        let mut ma = MemoryAid::new(30.0);
        ma.record_segment(ReplaySegment {
            start_seconds: 0.0,
            duration_seconds: 5.0,
            description: "first".into(),
        });
        ma.record_segment(ReplaySegment {
            start_seconds: 5.0,
            duration_seconds: 5.0,
            description: "second".into(),
        });
        assert_eq!(ma.replay_segment_count(), 2);
    }

    #[test]
    fn memory_aid_legend() {
        let mut ma = MemoryAid::new(30.0);
        ma.set_legend_text("Pitch = temperature");
        let txt = ma.reminder_text();
        assert!(txt.contains("Pitch = temperature"));
    }

    #[test]
    fn attention_guide_clear() {
        let mut ag = AttentionGuide::new();
        ag.schedule_beacon(1.0, "b");
        ag.clear();
        assert_eq!(ag.event_count(), 0);
    }

    #[test]
    fn cognitive_guide_prev() {
        let mut ce = CognitiveSupportEngine::new();
        ce.register_stream("a", 1);
        ce.register_stream("b", 2);
        ce.guide_prev();
        assert_eq!(ce.guide_current_name(), Some("b")); // wraps backwards
    }
}
