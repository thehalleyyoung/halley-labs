//! Trace recording.
use serde::{Serialize, Deserialize};

/// Records simulation states to a trace.
#[derive(Debug, Clone, Default)]
pub struct TraceRecorder { pub metadata: TraceMetadata, states: Vec<Vec<f64>>, times: Vec<f64> }
impl TraceRecorder {
    pub fn new() -> Self { Self::default() }
    /// Record a state snapshot.
    pub fn record(&mut self, time: f64, state: &[f64]) { self.times.push(time); self.states.push(state.to_vec()); }
    /// Number of recorded snapshots.
    pub fn len(&self) -> usize { self.states.len() }
    pub fn is_empty(&self) -> bool { self.states.is_empty() }
}

/// Recording policy determines when to record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecordingPolicy { EveryStep, EveryN(usize), Adaptive(AdaptiveRecording) }

/// Adaptive recording criteria.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveRecording { pub threshold: f64 }

/// Metadata for a trace.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TraceMetadata { pub simulation_name: String, pub integrator_name: String, pub num_particles: usize, pub dt: f64 }
