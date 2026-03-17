//! Adaptive step size controllers.
use crate::AdaptiveStepResult;

/// Embedded Runge-Kutta pair for error estimation.
#[derive(Debug, Clone)]
pub struct EmbeddedPair {
    /// Safety factor for step size control.
    pub safety: f64,
    /// Minimum step size.
    pub min_dt: f64,
    /// Maximum step size.
    pub max_dt: f64,
}

impl Default for EmbeddedPair {
    fn default() -> Self { Self { safety: 0.9, min_dt: 1e-12, max_dt: 1.0 } }
}

impl EmbeddedPair {
    /// Compute the next step size given the current error estimate.
    pub fn control(&self, error: f64, order: u32, dt: f64) -> AdaptiveStepResult {
        if error < 1.0 {
            let factor = self.safety * error.powf(-1.0 / (order as f64 + 1.0));
            let next = (dt * factor).clamp(self.min_dt, self.max_dt);
            AdaptiveStepResult { accepted: true, error_estimate: error, next_dt: next }
        } else {
            let factor = self.safety * error.powf(-1.0 / order as f64);
            let next = (dt * factor).max(self.min_dt);
            AdaptiveStepResult { accepted: false, error_estimate: error, next_dt: next }
        }
    }
}

/// PI (proportional-integral) step size controller.
#[derive(Debug, Clone)]
pub struct PIController { pub k_i: f64, pub k_p: f64 }
impl Default for PIController { fn default() -> Self { Self { k_i: 0.3, k_p: 0.4 } } }

/// PID step size controller.
#[derive(Debug, Clone)]
pub struct PIDController { pub k_i: f64, pub k_p: f64, pub k_d: f64 }
impl Default for PIDController { fn default() -> Self { Self { k_i: 0.3, k_p: 0.4, k_d: 0.1 } } }

/// Generic step size controller trait.
pub trait StepSizeController {
    /// Compute the next step size.
    fn next_step(&self, error: f64, dt: f64, order: u32) -> AdaptiveStepResult;
}

impl StepSizeController for PIController {
    fn next_step(&self, error: f64, dt: f64, order: u32) -> AdaptiveStepResult {
        EmbeddedPair::default().control(error, order, dt)
    }
}

impl StepSizeController for PIDController {
    fn next_step(&self, error: f64, dt: f64, order: u32) -> AdaptiveStepResult {
        EmbeddedPair::default().control(error, order, dt)
    }
}
