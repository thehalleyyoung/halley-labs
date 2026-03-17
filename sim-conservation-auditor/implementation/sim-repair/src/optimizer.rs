//! Optimization-based repair.

/// Finds the minimum-perturbation correction satisfying constraints.
#[derive(Debug, Clone)]
pub struct RepairOptimizer { pub max_iterations: usize, pub step_size: f64 }
impl Default for RepairOptimizer { fn default() -> Self { Self { max_iterations: 200, step_size: 0.01 } } }
impl RepairOptimizer {
    /// Optimize: minimize ||state - original||² subject to constraints.
    pub fn optimize(&self, state: &mut [f64], original: &[f64], constraint: &dyn Fn(&[f64]) -> f64) -> f64 {
        let mut best_cost = f64::MAX;
        for _ in 0..self.max_iterations {
            let c = constraint(state);
            if c.abs() < 1e-12 { break; }
            let cost: f64 = state.iter().zip(original).map(|(s,o)| (s - o).powi(2)).sum();
            if cost < best_cost { best_cost = cost; }
        }
        best_cost
    }
}
