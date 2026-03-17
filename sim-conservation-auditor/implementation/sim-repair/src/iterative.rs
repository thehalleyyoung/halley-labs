//! Iterative repair methods.

/// Iteratively repairs conservation violations.
#[derive(Debug, Clone)]
pub struct IterativeRepair { pub max_iterations: usize, pub tolerance: f64 }
impl Default for IterativeRepair { fn default() -> Self { Self { max_iterations: 100, tolerance: 1e-12 } } }
impl IterativeRepair {
    /// Iteratively apply corrections until the constraint is satisfied.
    pub fn repair(&self, state: &mut [f64], constraint: &dyn Fn(&[f64]) -> f64, correction: &dyn Fn(&mut [f64])) -> bool {
        for _ in 0..self.max_iterations {
            if constraint(state).abs() < self.tolerance { return true; }
            correction(state);
        }
        constraint(state).abs() < self.tolerance
    }
}
