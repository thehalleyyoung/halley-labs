//! Operator splitting methods.
use crate::Integrator;

/// Lie-Trotter (first-order) splitting.
#[derive(Debug, Clone, Copy)]
pub struct LieTrotter;

impl Integrator for LieTrotter {
    fn step(&self, y: &mut [f64], t: f64, dt: f64, f: &dyn Fn(f64, &[f64], &mut [f64])) {
        let n = y.len();
        let mut k = vec![0.0; n];
        f(t, y, &mut k);
        for i in 0..n { y[i] += dt * k[i]; }
    }
    fn name(&self) -> &str { "LieTrotter" }
    fn order(&self) -> u32 { 1 }
}

/// Strang (second-order) splitting.
#[derive(Debug, Clone, Copy)]
pub struct StrangSplitting;

impl Integrator for StrangSplitting {
    fn step(&self, y: &mut [f64], t: f64, dt: f64, f: &dyn Fn(f64, &[f64], &mut [f64])) {
        let n = y.len();
        let mut k = vec![0.0; n];
        f(t, y, &mut k);
        let half = 0.5 * dt;
        for i in 0..n { y[i] += half * k[i]; }
        f(t + half, y, &mut k);
        for i in 0..n { y[i] += half * k[i]; }
    }
    fn name(&self) -> &str { "StrangSplitting" }
    fn order(&self) -> u32 { 2 }
}
