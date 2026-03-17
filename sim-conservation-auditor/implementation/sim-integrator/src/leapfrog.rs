//! Leapfrog integrators.
use crate::{Integrator, SeparableIntegrator};

/// Standard leapfrog (kick-drift-kick) integrator.
#[derive(Debug, Clone, Copy)]
pub struct Leapfrog;

impl Integrator for Leapfrog {
    fn step(&self, state: &mut [f64], t: f64, dt: f64, deriv: &dyn Fn(f64, &[f64], &mut [f64])) {
        let n = state.len();
        let mut k = vec![0.0; n];
        deriv(t, state, &mut k);
        let half = 0.5 * dt;
        for i in 0..n { state[i] += half * k[i]; }
        deriv(t + half, state, &mut k);
        for i in 0..n { state[i] += half * k[i]; }
    }
    fn name(&self) -> &str { "Leapfrog" }
    fn order(&self) -> u32 { 2 }
    fn is_symplectic(&self) -> bool { true }
}

impl SeparableIntegrator for Leapfrog {
    fn step_separable(&self, q: &mut [f64], p: &mut [f64], t: f64, dt: f64,
                       force: &dyn Fn(f64, &[f64], &mut [f64]), mass: &[f64]) {
        let n = q.len();
        let half = 0.5 * dt;
        let mut f = vec![0.0; n];
        force(t, q, &mut f);
        for i in 0..n { p[i] += half * f[i]; }
        for i in 0..n { q[i] += dt * p[i] / mass[i]; }
        force(t + dt, q, &mut f);
        for i in 0..n { p[i] += half * f[i]; }
    }
    fn name(&self) -> &str { "Leapfrog" }
    fn order(&self) -> u32 { 2 }
}

/// Leapfrog DKD (drift-kick-drift) variant.
#[derive(Debug, Clone, Copy)]
pub struct LeapfrogDKD;

impl Integrator for LeapfrogDKD {
    fn step(&self, state: &mut [f64], t: f64, dt: f64, deriv: &dyn Fn(f64, &[f64], &mut [f64])) {
        Leapfrog.step(state, t, dt, deriv);
    }
    fn name(&self) -> &str { "LeapfrogDKD" }
    fn order(&self) -> u32 { 2 }
    fn is_symplectic(&self) -> bool { true }
}

impl SeparableIntegrator for LeapfrogDKD {
    fn step_separable(&self, q: &mut [f64], p: &mut [f64], t: f64, dt: f64,
                       force: &dyn Fn(f64, &[f64], &mut [f64]), mass: &[f64]) {
        let n = q.len();
        let half = 0.5 * dt;
        for i in 0..n { q[i] += half * p[i] / mass[i]; }
        let mut f = vec![0.0; n];
        force(t + half, q, &mut f);
        for i in 0..n { p[i] += dt * f[i]; }
        for i in 0..n { q[i] += half * p[i] / mass[i]; }
    }
    fn name(&self) -> &str { "LeapfrogDKD" }
    fn order(&self) -> u32 { 2 }
}
