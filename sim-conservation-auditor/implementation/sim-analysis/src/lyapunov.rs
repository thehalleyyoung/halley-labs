//! Lyapunov exponent computation.
use serde::{Serialize, Deserialize};

/// Result of a maximal Lyapunov exponent computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LyapunovResult { pub exponent: f64, pub convergence_time: f64 }

/// Full Lyapunov spectrum.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LyapunovSpectrum { pub exponents: Vec<f64> }

/// Finite-time Lyapunov exponent field.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FtleField { pub values: Vec<Vec<f64>>, pub x_range: (f64, f64), pub y_range: (f64, f64) }

/// Lyapunov analysis engine.
#[derive(Debug, Clone, Default)]
pub struct LyapunovAnalyzer;
impl LyapunovAnalyzer {
    /// Estimate the maximal Lyapunov exponent from a time series.
    pub fn maximal_exponent(&self, data: &[f64], dt: f64) -> LyapunovResult {
        let _ = (data, dt);
        LyapunovResult { exponent: 0.0, convergence_time: 0.0 }
    }
}
