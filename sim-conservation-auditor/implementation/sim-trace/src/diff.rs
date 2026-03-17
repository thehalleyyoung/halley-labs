//! Trace comparison utilities.
use serde::{Serialize, Deserialize};

/// Difference between two traces.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TraceDiff { pub state_diffs: Vec<StateDiff>, pub statistics: DiffStatistics }

/// Difference at a single time step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateDiff { pub time: f64, pub max_position_diff: f64, pub max_velocity_diff: f64 }

/// Per-particle difference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParticleDiff { pub index: usize, pub position_error: f64, pub velocity_error: f64 }

/// Summary statistics for trace differences.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DiffStatistics { pub max_error: f64, pub mean_error: f64, pub rms_error: f64 }
