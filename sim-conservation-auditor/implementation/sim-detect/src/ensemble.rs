//! Ensemble detection methods.
use serde::{Serialize, Deserialize};

/// Voting strategy for ensemble detectors.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum VotingStrategy { Majority, Unanimous, Weighted }

/// Ensemble detector combining multiple detection methods.
#[derive(Debug, Clone)]
pub struct EnsembleDetector { pub strategy: VotingStrategy }
impl Default for EnsembleDetector { fn default() -> Self { Self { strategy: VotingStrategy::Majority } } }

/// Ensemble detection result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleResult { pub detected: bool, pub votes_for: usize, pub votes_against: usize, pub confidence: f64 }
