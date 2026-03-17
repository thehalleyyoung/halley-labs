//! Temporal clinical modeling and concurrency analysis.

use serde::{Deserialize, Serialize};

/// Type of clinical event.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EventType {
    DoseAdministration,
    LabDraw,
    VitalSign,
    Intervention,
    Observation,
    StateChange,
}

/// A clinical event in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClinicalEvent {
    pub event_type: EventType,
    pub description: String,
    pub time_hours: f64,
    pub duration_hours: Option<f64>,
}

/// A window of time.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TimeWindow {
    pub start_hours: f64,
    pub end_hours: f64,
}

impl TimeWindow {
    pub fn new(start: f64, end: f64) -> Self {
        TimeWindow { start_hours: start, end_hours: end }
    }

    pub fn duration(&self) -> f64 { self.end_hours - self.start_hours }

    pub fn contains(&self, time: f64) -> bool {
        time >= self.start_hours && time <= self.end_hours
    }

    pub fn overlaps(&self, other: &TimeWindow) -> bool {
        self.start_hours <= other.end_hours && other.start_hours <= self.end_hours
    }
}

/// A temporal constraint between events.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalConstraint {
    pub event_a: String,
    pub event_b: String,
    pub min_separation_hours: f64,
    pub max_separation_hours: Option<f64>,
    pub reason: String,
}

/// Timeline of clinical events.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ClinicalTimeline {
    pub events: Vec<ClinicalEvent>,
    pub constraints: Vec<TemporalConstraint>,
}

impl ClinicalTimeline {
    pub fn new() -> Self { ClinicalTimeline { events: Vec::new(), constraints: Vec::new() } }

    pub fn add_event(&mut self, event: ClinicalEvent) { self.events.push(event); }

    pub fn events_in_window(&self, window: &TimeWindow) -> Vec<&ClinicalEvent> {
        self.events.iter().filter(|e| window.contains(e.time_hours)).collect()
    }
}

/// Concurrency analysis for overlapping drug exposures.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConcurrencyAnalysis {
    pub concurrent_pairs: Vec<(String, String)>,
    pub overlap_windows: Vec<TimeWindow>,
    pub max_concurrent_drugs: usize,
}

impl Default for ConcurrencyAnalysis {
    fn default() -> Self {
        ConcurrencyAnalysis {
            concurrent_pairs: Vec::new(),
            overlap_windows: Vec::new(),
            max_concurrent_drugs: 0,
        }
    }
}
