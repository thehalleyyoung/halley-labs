//! Trajectory analysis engine.
use serde::{Serialize, Deserialize};

/// Analyzes simulation trajectories for conservation law properties.
#[derive(Debug, Clone)]
pub struct TrajectoryAnalyzer { pub tolerance: f64 }
impl Default for TrajectoryAnalyzer { fn default() -> Self { Self { tolerance: 1e-8 } } }
impl TrajectoryAnalyzer {
    /// Analyze a time series of conserved quantity values.
    pub fn analyze(&self, times: &[f64], values: &[f64]) -> AnalysisReport {
        let n = values.len();
        if n == 0 { return AnalysisReport::default(); }
        let initial = values[0];
        let max_drift = values.iter().map(|v| (v - initial).abs()).fold(0.0_f64, f64::max);
        AnalysisReport { max_drift, violation_count: if max_drift > self.tolerance { 1 } else { 0 }, total_steps: n, intervals: Vec::new(), growth_rate: ErrorGrowthRate::default(), times_span: (*times.first().unwrap_or(&0.0), *times.last().unwrap_or(&0.0)) }
    }
}

/// Report from trajectory analysis.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AnalysisReport {
    pub max_drift: f64,
    pub violation_count: usize,
    pub total_steps: usize,
    pub intervals: Vec<ViolationInterval>,
    pub growth_rate: ErrorGrowthRate,
    pub times_span: (f64, f64),
}

/// A time interval during which a conservation violation persists.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViolationInterval {
    pub start_time: f64,
    pub end_time: f64,
    pub max_deviation: f64,
}

/// Characterization of error growth rate.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ErrorGrowthRate {
    pub linear_rate: f64,
    pub quadratic_rate: f64,
    pub exponential_rate: f64,
}
