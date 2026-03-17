//! Projection-based conservation repair.

/// Projects a state onto the conservation manifold.
#[derive(Debug, Clone, Default)]
pub struct ProjectionRepair { pub max_iterations: usize, pub tolerance: f64 }
impl ProjectionRepair {
    pub fn new() -> Self { Self { max_iterations: 100, tolerance: 1e-12 } }
    /// Project a state vector to satisfy a conservation constraint.
    pub fn project(&self, state: &mut [f64], constraint: &dyn Fn(&[f64]) -> f64, gradient: &dyn Fn(&[f64]) -> Vec<f64>) {
        for _ in 0..self.max_iterations {
            let c = constraint(state);
            if c.abs() < self.tolerance { return; }
            let g = gradient(state);
            let g_norm_sq: f64 = g.iter().map(|x| x * x).sum();
            if g_norm_sq < 1e-30 { return; }
            let lambda = c / g_norm_sq;
            for (s, gi) in state.iter_mut().zip(g.iter()) { *s -= lambda * gi; }
        }
    }
}
