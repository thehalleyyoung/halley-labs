//! Execution trace recording with a circular buffer.
//!
//! Records events, state changes, and guard evaluations during runtime
//! execution. Supports export to JSON and CSV, filtering by type and time
//! range, and memory-bounded operation.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fmt;

// ---------------------------------------------------------------------------
// TraceEntry
// ---------------------------------------------------------------------------

/// A single entry in the execution trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TraceEntry {
    Event {
        timestamp: f64,
        kind: String,
        source: Option<String>,
    },
    StateChange {
        timestamp: f64,
        from_states: Vec<u32>,
        to_states: Vec<u32>,
        transition_id: Option<u32>,
    },
    GuardEval {
        timestamp: f64,
        transition_id: u32,
        guard_description: String,
        result: bool,
    },
}

impl TraceEntry {
    pub fn timestamp(&self) -> f64 {
        match self {
            Self::Event { timestamp, .. } => *timestamp,
            Self::StateChange { timestamp, .. } => *timestamp,
            Self::GuardEval { timestamp, .. } => *timestamp,
        }
    }

    pub fn entry_type(&self) -> TraceEntryType {
        match self {
            Self::Event { .. } => TraceEntryType::Event,
            Self::StateChange { .. } => TraceEntryType::StateChange,
            Self::GuardEval { .. } => TraceEntryType::GuardEval,
        }
    }
}

/// Type discriminant for filtering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TraceEntryType {
    Event,
    StateChange,
    GuardEval,
}

impl fmt::Display for TraceEntryType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Event => write!(f, "event"),
            Self::StateChange => write!(f, "state_change"),
            Self::GuardEval => write!(f, "guard_eval"),
        }
    }
}

// ---------------------------------------------------------------------------
// TraceRecorder
// ---------------------------------------------------------------------------

/// Records execution trace entries in a circular buffer.
#[derive(Debug)]
pub struct TraceRecorder {
    buffer: VecDeque<TraceEntry>,
    capacity: usize,
    total_recorded: u64,
    enabled: bool,
}

impl TraceRecorder {
    /// Create a new recorder with the given maximum capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity.min(65536)),
            capacity,
            total_recorded: 0,
            enabled: true,
        }
    }

    /// Create an unbounded recorder (only limited by memory).
    pub fn unbounded() -> Self {
        Self {
            buffer: VecDeque::new(),
            capacity: usize::MAX,
            total_recorded: 0,
            enabled: true,
        }
    }

    /// Enable or disable recording.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    // -----------------------------------------------------------------------
    // Recording
    // -----------------------------------------------------------------------

    /// Record an event occurrence.
    pub fn record_event(
        &mut self,
        timestamp: f64,
        kind: impl Into<String>,
        source: Option<String>,
    ) {
        if !self.enabled {
            return;
        }
        self.push(TraceEntry::Event {
            timestamp,
            kind: kind.into(),
            source,
        });
    }

    /// Record a state change.
    pub fn record_state_change(
        &mut self,
        timestamp: f64,
        from_states: Vec<u32>,
        to_states: Vec<u32>,
        transition_id: Option<u32>,
    ) {
        if !self.enabled {
            return;
        }
        self.push(TraceEntry::StateChange {
            timestamp,
            from_states,
            to_states,
            transition_id,
        });
    }

    /// Record a guard evaluation result.
    pub fn record_guard_eval(
        &mut self,
        timestamp: f64,
        transition_id: u32,
        guard_description: impl Into<String>,
        result: bool,
    ) {
        if !self.enabled {
            return;
        }
        self.push(TraceEntry::GuardEval {
            timestamp,
            transition_id,
            guard_description: guard_description.into(),
            result,
        });
    }

    fn push(&mut self, entry: TraceEntry) {
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(entry);
        self.total_recorded += 1;
    }

    // -----------------------------------------------------------------------
    // Retrieval
    // -----------------------------------------------------------------------

    /// Get all recorded entries (oldest first).
    pub fn get_trace(&self) -> Vec<TraceEntry> {
        self.buffer.iter().cloned().collect()
    }

    /// Number of entries currently in the buffer.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Total entries ever recorded (including evicted).
    pub fn total_recorded(&self) -> u64 {
        self.total_recorded
    }

    /// How many entries were evicted due to capacity limits.
    pub fn evicted_count(&self) -> u64 {
        self.total_recorded.saturating_sub(self.buffer.len() as u64)
    }

    /// Clear the buffer.
    pub fn clear(&mut self) {
        self.buffer.clear();
    }

    // -----------------------------------------------------------------------
    // Filtering
    // -----------------------------------------------------------------------

    /// Filter entries by type.
    pub fn filter_by_type(&self, entry_type: TraceEntryType) -> Vec<TraceEntry> {
        self.buffer
            .iter()
            .filter(|e| e.entry_type() == entry_type)
            .cloned()
            .collect()
    }

    /// Filter entries within a time range [start, end].
    pub fn filter_by_time(&self, start: f64, end: f64) -> Vec<TraceEntry> {
        self.buffer
            .iter()
            .filter(|e| {
                let t = e.timestamp();
                t >= start && t <= end
            })
            .cloned()
            .collect()
    }

    /// Filter by both type and time range.
    pub fn filter(
        &self,
        entry_type: Option<TraceEntryType>,
        start: Option<f64>,
        end: Option<f64>,
    ) -> Vec<TraceEntry> {
        self.buffer
            .iter()
            .filter(|e| {
                if let Some(et) = entry_type {
                    if e.entry_type() != et {
                        return false;
                    }
                }
                if let Some(s) = start {
                    if e.timestamp() < s {
                        return false;
                    }
                }
                if let Some(en) = end {
                    if e.timestamp() > en {
                        return false;
                    }
                }
                true
            })
            .cloned()
            .collect()
    }

    // -----------------------------------------------------------------------
    // Export
    // -----------------------------------------------------------------------

    /// Export the trace to JSON.
    pub fn export_json(&self) -> String {
        let entries: Vec<&TraceEntry> = self.buffer.iter().collect();
        serde_json::to_string_pretty(&entries)
            .unwrap_or_else(|e| format!("{{\"error\": \"{}\"}}", e))
    }

    /// Export the trace to CSV format.
    pub fn export_csv(&self) -> String {
        let mut out = String::new();
        out.push_str("timestamp,type,details\n");

        for entry in &self.buffer {
            match entry {
                TraceEntry::Event {
                    timestamp,
                    kind,
                    source,
                } => {
                    let src = source.as_deref().unwrap_or("");
                    out.push_str(&format!(
                        "{:.6},event,\"kind={} source={}\"\n",
                        timestamp, kind, src
                    ));
                }
                TraceEntry::StateChange {
                    timestamp,
                    from_states,
                    to_states,
                    transition_id,
                } => {
                    let tid = transition_id
                        .map(|t| t.to_string())
                        .unwrap_or_default();
                    out.push_str(&format!(
                        "{:.6},state_change,\"from={:?} to={:?} transition={}\"\n",
                        timestamp, from_states, to_states, tid
                    ));
                }
                TraceEntry::GuardEval {
                    timestamp,
                    transition_id,
                    guard_description,
                    result,
                } => {
                    out.push_str(&format!(
                        "{:.6},guard_eval,\"transition={} guard={} result={}\"\n",
                        timestamp, transition_id, guard_description, result
                    ));
                }
            }
        }
        out
    }
}

impl Default for TraceRecorder {
    fn default() -> Self {
        Self::new(10_000)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_and_retrieve() {
        let mut rec = TraceRecorder::new(100);
        rec.record_event(1.0, "grab_start", None);
        rec.record_state_change(1.0, vec![0], vec![1], Some(0));
        rec.record_guard_eval(1.0, 0, "inside(e1,r1)", true);

        let trace = rec.get_trace();
        assert_eq!(trace.len(), 3);
        assert_eq!(rec.total_recorded(), 3);
    }

    #[test]
    fn circular_buffer_eviction() {
        let mut rec = TraceRecorder::new(3);
        for i in 0..5 {
            rec.record_event(i as f64, &format!("evt{}", i), None);
        }
        assert_eq!(rec.len(), 3);
        assert_eq!(rec.total_recorded(), 5);
        assert_eq!(rec.evicted_count(), 2);

        let trace = rec.get_trace();
        // Should contain the last 3 entries
        assert!(matches!(&trace[0], TraceEntry::Event { kind, .. } if kind == "evt2"));
    }

    #[test]
    fn filter_by_type() {
        let mut rec = TraceRecorder::new(100);
        rec.record_event(1.0, "evt", None);
        rec.record_state_change(1.0, vec![0], vec![1], None);
        rec.record_event(2.0, "evt2", None);

        let events = rec.filter_by_type(TraceEntryType::Event);
        assert_eq!(events.len(), 2);

        let changes = rec.filter_by_type(TraceEntryType::StateChange);
        assert_eq!(changes.len(), 1);
    }

    #[test]
    fn filter_by_time() {
        let mut rec = TraceRecorder::new(100);
        rec.record_event(1.0, "a", None);
        rec.record_event(2.0, "b", None);
        rec.record_event(3.0, "c", None);

        let filtered = rec.filter_by_time(1.5, 2.5);
        assert_eq!(filtered.len(), 1);
    }

    #[test]
    fn combined_filter() {
        let mut rec = TraceRecorder::new(100);
        rec.record_event(1.0, "a", None);
        rec.record_state_change(1.5, vec![0], vec![1], None);
        rec.record_event(2.0, "b", None);
        rec.record_guard_eval(2.5, 0, "g", true);

        let filtered = rec.filter(
            Some(TraceEntryType::Event),
            Some(0.0),
            Some(1.5),
        );
        assert_eq!(filtered.len(), 1);
    }

    #[test]
    fn export_json_valid() {
        let mut rec = TraceRecorder::new(100);
        rec.record_event(1.0, "grab", Some("hand_l".into()));
        let json = rec.export_json();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(parsed.is_array());
    }

    #[test]
    fn export_csv_format() {
        let mut rec = TraceRecorder::new(100);
        rec.record_event(1.0, "grab", None);
        rec.record_state_change(1.0, vec![0], vec![1], Some(0));
        let csv = rec.export_csv();
        assert!(csv.starts_with("timestamp,type,details\n"));
        assert!(csv.contains("event"));
        assert!(csv.contains("state_change"));
    }

    #[test]
    fn disabled_does_not_record() {
        let mut rec = TraceRecorder::new(100);
        rec.set_enabled(false);
        rec.record_event(1.0, "ignored", None);
        assert!(rec.is_empty());
    }

    #[test]
    fn clear_empties_buffer() {
        let mut rec = TraceRecorder::new(100);
        rec.record_event(1.0, "a", None);
        rec.clear();
        assert!(rec.is_empty());
        assert_eq!(rec.total_recorded(), 1); // total still counted
    }

    #[test]
    fn unbounded_grows() {
        let mut rec = TraceRecorder::unbounded();
        for i in 0..1000 {
            rec.record_event(i as f64, "evt", None);
        }
        assert_eq!(rec.len(), 1000);
        assert_eq!(rec.evicted_count(), 0);
    }

    #[test]
    fn entry_timestamp() {
        let entry = TraceEntry::Event {
            timestamp: 42.0,
            kind: "test".into(),
            source: None,
        };
        assert_eq!(entry.timestamp(), 42.0);
    }

    #[test]
    fn entry_type_discriminant() {
        let e1 = TraceEntry::Event {
            timestamp: 0.0,
            kind: "x".into(),
            source: None,
        };
        let e2 = TraceEntry::StateChange {
            timestamp: 0.0,
            from_states: vec![],
            to_states: vec![],
            transition_id: None,
        };
        let e3 = TraceEntry::GuardEval {
            timestamp: 0.0,
            transition_id: 0,
            guard_description: "g".into(),
            result: false,
        };
        assert_eq!(e1.entry_type(), TraceEntryType::Event);
        assert_eq!(e2.entry_type(), TraceEntryType::StateChange);
        assert_eq!(e3.entry_type(), TraceEntryType::GuardEval);
    }
}
