//! Störmer-Verlet and velocity Verlet integrators.
use crate::{Integrator, SeparableIntegrator};

/// Störmer-Verlet integrator (position form, 2nd order symplectic).
#[derive(Debug, Clone, Copy)]
pub struct StormerVerlet;

impl Integrator for StormerVerlet {
    fn step(&self, state: &mut [f64], t: f64, dt: f64, deriv: &dyn Fn(f64, &[f64], &mut [f64])) {
        let n = state.len();
        let mut k1 = vec![0.0; n];
        deriv(t, state, &mut k1);
        let half_dt = 0.5 * dt;
        for i in 0..n { state[i] += half_dt * k1[i]; }
        let mut k2 = vec![0.0; n];
        deriv(t + half_dt, state, &mut k2);
        for i in 0..n { state[i] += half_dt * k2[i]; }
    }
    fn name(&self) -> &str { "StormerVerlet" }
    fn order(&self) -> u32 { 2 }
    fn is_symplectic(&self) -> bool { true }
}

impl SeparableIntegrator for StormerVerlet {
    fn step_separable(&self, q: &mut [f64], p: &mut [f64], t: f64, dt: f64,
                       force: &dyn Fn(f64, &[f64], &mut [f64]), mass: &[f64]) {
        VelocityVerlet.step_separable(q, p, t, dt, force, mass);
    }
    fn name(&self) -> &str { "StormerVerlet" }
    fn order(&self) -> u32 { 2 }
}

/// Velocity Verlet integrator (2nd order symplectic).
#[derive(Debug, Clone, Copy)]
pub struct VelocityVerlet;

impl Integrator for VelocityVerlet {
    fn step(&self, state: &mut [f64], t: f64, dt: f64, deriv: &dyn Fn(f64, &[f64], &mut [f64])) {
        StormerVerlet.step(state, t, dt, deriv);
    }
    fn name(&self) -> &str { "VelocityVerlet" }
    fn order(&self) -> u32 { 2 }
    fn is_symplectic(&self) -> bool { true }
}

impl SeparableIntegrator for VelocityVerlet {
    fn step_separable(&self, q: &mut [f64], p: &mut [f64], t: f64, dt: f64,
                       force: &dyn Fn(f64, &[f64], &mut [f64]), mass: &[f64]) {
        let n = q.len();
        let mut f_old = vec![0.0; n];
        force(t, q, &mut f_old);
        let half_dt = 0.5 * dt;
        for i in 0..n { p[i] += half_dt * f_old[i]; }
        for i in 0..n { q[i] += dt * p[i] / mass[i]; }
        let mut f_new = vec![0.0; n];
        force(t + dt, q, &mut f_new);
        for i in 0..n { p[i] += half_dt * f_new[i]; }
    }
    fn name(&self) -> &str { "VelocityVerlet" }
    fn order(&self) -> u32 { 2 }
}

/// Position Verlet integrator variant.
#[derive(Debug, Clone, Copy)]
pub struct PositionVerlet;

impl Integrator for PositionVerlet {
    fn step(&self, state: &mut [f64], t: f64, dt: f64, deriv: &dyn Fn(f64, &[f64], &mut [f64])) {
        StormerVerlet.step(state, t, dt, deriv);
    }
    fn name(&self) -> &str { "PositionVerlet" }
    fn order(&self) -> u32 { 2 }
    fn is_symplectic(&self) -> bool { true }
}

impl SeparableIntegrator for PositionVerlet {
    fn step_separable(&self, q: &mut [f64], p: &mut [f64], t: f64, dt: f64,
                       force: &dyn Fn(f64, &[f64], &mut [f64]), mass: &[f64]) {
        VelocityVerlet.step_separable(q, p, t, dt, force, mass);
    }
    fn name(&self) -> &str { "PositionVerlet" }
    fn order(&self) -> u32 { 2 }
}
