//! Poincaré section analysis.
use serde::{Serialize, Deserialize};

/// A Poincaré section definition and computed crossings.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PoincareSection { pub crossings: Vec<SectionCrossing> }

/// A crossing of the Poincaré section.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SectionCrossing { pub time: f64, pub q: f64, pub p: f64 }

/// Return map from Poincaré section data.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReturnMap { pub points: Vec<(f64, f64)> }

/// Island in a return map.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReturnMapIsland { pub center: (f64, f64), pub radius: f64 }

/// Winding number computation result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindingNumberResult { pub winding_number: f64, pub uncertainty: f64 }

/// Poincaré analysis engine.
#[derive(Debug, Clone, Default)]
pub struct PoincareAnalyzer;
