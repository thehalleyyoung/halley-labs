//! Trace event logging.
use serde::{Serialize, Deserialize};

/// Kinds of trace events.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventKind { ViolationDetected, CheckpointSaved, StepSizeChanged, CollisionDetected, Custom(String) }

/// A recorded event in the trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceEvent { pub time: f64, pub step: usize, pub kind: EventKind, pub message: String }

/// Event log collecting all events.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EventLog { pub events: Vec<TraceEvent> }
impl EventLog {
    pub fn new() -> Self { Self::default() }
    pub fn push(&mut self, event: TraceEvent) { self.events.push(event); }
    pub fn len(&self) -> usize { self.events.len() }
    pub fn is_empty(&self) -> bool { self.events.is_empty() }
}

/// Query events by criteria.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventQuery { pub kind_filter: Option<String>, pub time_range: Option<(f64, f64)> }
