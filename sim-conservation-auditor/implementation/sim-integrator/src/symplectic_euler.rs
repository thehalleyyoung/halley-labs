//! Symplectic Euler integrators.
use crate::{Integrator, SeparableIntegrator};

/// Symplectic Euler A (kick-drift): update p then q.
#[derive(Debug, Clone, Copy)]
pub struct SymplecticEulerA;

impl Integrator for SymplecticEulerA {
    fn step(&self, state: &mut [f64], t: f64, dt: f64, deriv: &dyn Fn(f64, &[f64], &mut [f64])) {
        let n = state.len();
        let mut dy = vec![0.0; n];
        deriv(t, state, &mut dy);
        for i in 0..n { state[i] += dt * dy[i]; }
    }
    fn name(&self) -> &str { "SymplecticEulerA" }
    fn order(&self) -> u32 { 1 }
    fn is_symplectic(&self) -> bool { true }
}

impl SeparableIntegrator for SymplecticEulerA {
    fn step_separable(&self, q: &mut [f64], p: &mut [f64], t: f64, dt: f64,
                       force: &dyn Fn(f64, &[f64], &mut [f64]), mass: &[f64]) {
        let n = q.len();
        let mut f = vec![0.0; n];
        force(t, q, &mut f);
        for i in 0..n { p[i] += dt * f[i]; }
        for i in 0..n { q[i] += dt * p[i] / mass[i]; }
    }
    fn name(&self) -> &str { "SymplecticEulerA" }
    fn order(&self) -> u32 { 1 }
}

/// Symplectic Euler B (drift-kick): update q then p.
#[derive(Debug, Clone, Copy)]
pub struct SymplecticEulerB;

impl Integrator for SymplecticEulerB {
    fn step(&self, state: &mut [f64], t: f64, dt: f64, deriv: &dyn Fn(f64, &[f64], &mut [f64])) {
        let n = state.len();
        let mut dy = vec![0.0; n];
        deriv(t, state, &mut dy);
        for i in 0..n { state[i] += dt * dy[i]; }
    }
    fn name(&self) -> &str { "SymplecticEulerB" }
    fn order(&self) -> u32 { 1 }
    fn is_symplectic(&self) -> bool { true }
}

impl SeparableIntegrator for SymplecticEulerB {
    fn step_separable(&self, q: &mut [f64], p: &mut [f64], t: f64, dt: f64,
                       force: &dyn Fn(f64, &[f64], &mut [f64]), mass: &[f64]) {
        let n = q.len();
        for i in 0..n { q[i] += dt * p[i] / mass[i]; }
        let mut f = vec![0.0; n];
        force(t, q, &mut f);
        for i in 0..n { p[i] += dt * f[i]; }
    }
    fn name(&self) -> &str { "SymplecticEulerB" }
    fn order(&self) -> u32 { 1 }
}
