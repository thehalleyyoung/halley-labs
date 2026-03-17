//! Convergence analysis.
use serde::{Serialize, Deserialize};

/// Richardson extrapolation result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RichardsonResult { pub extrapolated_value: f64, pub estimated_order: f64, pub error_estimate: f64 }

/// Convergence order measurement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceOrderResult { pub order: f64, pub coefficient: f64 }

/// Stability region (set of z = h*lambda values).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilityRegion { pub boundary_points: Vec<(f64, f64)> }

/// Convergence analysis engine.
#[derive(Debug, Clone, Default)]
pub struct ConvergenceAnalyzer;
impl ConvergenceAnalyzer {
    /// Estimate convergence order from errors at different step sizes.
    pub fn estimate_order(&self, step_sizes: &[f64], errors: &[f64]) -> ConvergenceOrderResult {
        if step_sizes.len() < 2 || errors.len() < 2 { return ConvergenceOrderResult { order: 0.0, coefficient: 0.0 }; }
        let log_h1 = step_sizes[0].ln();
        let log_h2 = step_sizes[1].ln();
        let log_e1 = errors[0].max(1e-300).ln();
        let log_e2 = errors[1].max(1e-300).ln();
        let denom = log_h1 - log_h2;
        let order = if denom.abs() > 1e-30 { (log_e1 - log_e2) / denom } else { 0.0 };
        ConvergenceOrderResult { order, coefficient: errors[0] / step_sizes[0].powf(order) }
    }
}
