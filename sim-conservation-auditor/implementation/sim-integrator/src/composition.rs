//! Composition methods for constructing high-order integrators.
use crate::{Integrator, SeparableIntegrator};

/// ABA composition pattern.
#[derive(Debug, Clone, Copy)]
pub struct ABAComposition;

impl Integrator for ABAComposition {
    fn step(&self, y: &mut [f64], t: f64, dt: f64, f: &dyn Fn(f64, &[f64], &mut [f64])) {
        crate::Leapfrog.step(y, t, dt, f);
    }
    fn name(&self) -> &str { "ABAComposition" }
    fn order(&self) -> u32 { 2 }
    fn is_symplectic(&self) -> bool { true }
}

impl SeparableIntegrator for ABAComposition {
    fn step_separable(&self, q: &mut [f64], p: &mut [f64], t: f64, dt: f64,
                       force: &dyn Fn(f64, &[f64], &mut [f64]), mass: &[f64]) {
        crate::Leapfrog.step_separable(q, p, t, dt, force, mass);
    }
    fn name(&self) -> &str { "ABAComposition" }
    fn order(&self) -> u32 { 2 }
}

/// Suzuki fractal composition for higher-order methods.
#[derive(Debug, Clone, Copy)]
pub struct SuzukiComposition;

impl Integrator for SuzukiComposition {
    fn step(&self, y: &mut [f64], t: f64, dt: f64, f: &dyn Fn(f64, &[f64], &mut [f64])) {
        crate::Yoshida4.step(y, t, dt, f);
    }
    fn name(&self) -> &str { "SuzukiComposition" }
    fn order(&self) -> u32 { 4 }
    fn is_symplectic(&self) -> bool { true }
}

impl SeparableIntegrator for SuzukiComposition {
    fn step_separable(&self, q: &mut [f64], p: &mut [f64], t: f64, dt: f64,
                       force: &dyn Fn(f64, &[f64], &mut [f64]), mass: &[f64]) {
        crate::Yoshida4.step_separable(q, p, t, dt, force, mass);
    }
    fn name(&self) -> &str { "SuzukiComposition" }
    fn order(&self) -> u32 { 4 }
}

/// Triple-jump composition method.
#[derive(Debug, Clone, Copy)]
pub struct TripleJump;

impl Integrator for TripleJump {
    fn step(&self, y: &mut [f64], t: f64, dt: f64, f: &dyn Fn(f64, &[f64], &mut [f64])) {
        crate::Yoshida4.step(y, t, dt, f);
    }
    fn name(&self) -> &str { "TripleJump" }
    fn order(&self) -> u32 { 4 }
    fn is_symplectic(&self) -> bool { true }
}

impl SeparableIntegrator for TripleJump {
    fn step_separable(&self, q: &mut [f64], p: &mut [f64], t: f64, dt: f64,
                       force: &dyn Fn(f64, &[f64], &mut [f64]), mass: &[f64]) {
        crate::Yoshida4.step_separable(q, p, t, dt, force, mass);
    }
    fn name(&self) -> &str { "TripleJump" }
    fn order(&self) -> u32 { 4 }
}
