//! Simulation checkpointing.
use serde::{Serialize, Deserialize};

/// A simulation checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint { pub time: f64, pub step: usize, pub state: Vec<f64>, pub metadata: std::collections::HashMap<String, String> }

/// Manages simulation checkpoints.
#[derive(Debug, Clone, Default)]
pub struct CheckpointManager { checkpoints: Vec<Checkpoint> }
impl CheckpointManager {
    pub fn new() -> Self { Self::default() }
    /// Save a checkpoint.
    pub fn save(&mut self, checkpoint: Checkpoint) { self.checkpoints.push(checkpoint); }
    /// Get the latest checkpoint.
    pub fn latest(&self) -> Option<&Checkpoint> { self.checkpoints.last() }
    /// Number of checkpoints.
    pub fn len(&self) -> usize { self.checkpoints.len() }
    pub fn is_empty(&self) -> bool { self.checkpoints.is_empty() }
}
