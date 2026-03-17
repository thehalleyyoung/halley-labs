//! Phase portrait analysis.
use serde::{Serialize, Deserialize};

/// Phase portrait data.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PhasePortrait { pub points: Vec<PhasePoint2D> }

/// A point in 2D phase space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhasePoint2D { pub q: f64, pub p: f64 }

/// A fixed point with stability classification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixedPoint { pub q: f64, pub p: f64, pub stability: StabilityKind }

/// Stability classification for fixed points.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum StabilityKind { Stable, Unstable, Center, Saddle, SpiralStable, SpiralUnstable }

/// Phase space area computation result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseAreaResult { pub area: f64, pub relative_change: f64 }

/// KAM torus detection result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KamTorusResult { pub detected: bool, pub winding_number: f64, pub residual: f64 }
