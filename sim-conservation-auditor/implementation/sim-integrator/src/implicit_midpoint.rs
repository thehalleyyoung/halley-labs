//! Implicit midpoint rule (2nd order, symplectic).
use crate::{Integrator, ImplicitConfig};

/// Implicit midpoint integrator.
#[derive(Debug, Clone)]
pub struct ImplicitMidpoint {
    /// Solver configuration.
    pub config: ImplicitConfig,
}

impl Default for ImplicitMidpoint {
    fn default() -> Self { Self { config: ImplicitConfig::default() } }
}

impl Integrator for ImplicitMidpoint {
    fn step(&self, y: &mut [f64], t: f64, dt: f64, f: &dyn Fn(f64, &[f64], &mut [f64])) {
        let n = y.len();
        let mut mid = y.to_vec();
        let mut k = vec![0.0; n];
        let t_mid = t + 0.5 * dt;
        for _ in 0..self.config.max_iters {
            f(t_mid, &mid, &mut k);
            let mut max_change = 0.0_f64;
            for i in 0..n {
                let new_mid = y[i] + 0.5 * dt * k[i];
                max_change = max_change.max((new_mid - mid[i]).abs());
                mid[i] = self.config.relaxation * new_mid + (1.0 - self.config.relaxation) * mid[i];
            }
            if max_change < self.config.tolerance { break; }
        }
        f(t_mid, &mid, &mut k);
        for i in 0..n { y[i] += dt * k[i]; }
    }
    fn name(&self) -> &str { "ImplicitMidpoint" }
    fn order(&self) -> u32 { 2 }
    fn is_symplectic(&self) -> bool { true }
}
