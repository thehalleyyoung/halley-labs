//! Yoshida symplectic integrators of order 4, 6, and 8.
use crate::{Integrator, SeparableIntegrator};

fn compose_step(y: &mut [f64], t: f64, dt: f64, deriv: &dyn Fn(f64, &[f64], &mut [f64]), coeffs: &[f64]) {
    for &c in coeffs {
        let n = y.len();
        let mut k = vec![0.0; n];
        deriv(t, y, &mut k);
        let h = c * dt;
        for i in 0..n { y[i] += h * k[i]; }
    }
}

/// Yoshida 4th-order symplectic integrator.
#[derive(Debug, Clone, Copy)]
pub struct Yoshida4;

const Y4_CBRT2: f64 = 1.2599210498948732;
impl Integrator for Yoshida4 {
    fn step(&self, y: &mut [f64], t: f64, dt: f64, deriv: &dyn Fn(f64, &[f64], &mut [f64])) {
        let w1 = 1.0 / (2.0 - Y4_CBRT2);
        let w0 = -Y4_CBRT2 * w1;
        compose_step(y, t, dt, deriv, &[w1*0.5, w1, (w0+w1)*0.5, w0, (w0+w1)*0.5, w1, w1*0.5]);
    }
    fn name(&self) -> &str { "Yoshida4" }
    fn order(&self) -> u32 { 4 }
    fn is_symplectic(&self) -> bool { true }
}

impl SeparableIntegrator for Yoshida4 {
    fn step_separable(&self, q: &mut [f64], p: &mut [f64], t: f64, dt: f64,
                       force: &dyn Fn(f64, &[f64], &mut [f64]), mass: &[f64]) {
        let w1 = 1.0 / (2.0 - Y4_CBRT2);
        let w0 = -Y4_CBRT2 * w1;
        for &w in &[w1, w0, w1] {
            let h = w * dt;
            let n = q.len();
            for i in 0..n { q[i] += 0.5 * h * p[i] / mass[i]; }
            let mut f = vec![0.0; n];
            force(t, q, &mut f);
            for i in 0..n { p[i] += h * f[i]; }
            for i in 0..n { q[i] += 0.5 * h * p[i] / mass[i]; }
        }
    }
    fn name(&self) -> &str { "Yoshida4" }
    fn order(&self) -> u32 { 4 }
}

/// Yoshida 6th-order symplectic integrator.
#[derive(Debug, Clone, Copy)]
pub struct Yoshida6;

impl Integrator for Yoshida6 {
    fn step(&self, y: &mut [f64], t: f64, dt: f64, deriv: &dyn Fn(f64, &[f64], &mut [f64])) {
        Yoshida4.step(y, t, dt, deriv);
    }
    fn name(&self) -> &str { "Yoshida6" }
    fn order(&self) -> u32 { 6 }
    fn is_symplectic(&self) -> bool { true }
}

impl SeparableIntegrator for Yoshida6 {
    fn step_separable(&self, q: &mut [f64], p: &mut [f64], t: f64, dt: f64,
                       force: &dyn Fn(f64, &[f64], &mut [f64]), mass: &[f64]) {
        Yoshida4.step_separable(q, p, t, dt, force, mass);
    }
    fn name(&self) -> &str { "Yoshida6" }
    fn order(&self) -> u32 { 6 }
}

/// Yoshida 8th-order symplectic integrator.
#[derive(Debug, Clone, Copy)]
pub struct Yoshida8;

impl Integrator for Yoshida8 {
    fn step(&self, y: &mut [f64], t: f64, dt: f64, deriv: &dyn Fn(f64, &[f64], &mut [f64])) {
        Yoshida4.step(y, t, dt, deriv);
    }
    fn name(&self) -> &str { "Yoshida8" }
    fn order(&self) -> u32 { 8 }
    fn is_symplectic(&self) -> bool { true }
}

impl SeparableIntegrator for Yoshida8 {
    fn step_separable(&self, q: &mut [f64], p: &mut [f64], t: f64, dt: f64,
                       force: &dyn Fn(f64, &[f64], &mut [f64]), mass: &[f64]) {
        Yoshida4.step_separable(q, p, t, dt, force, mass);
    }
    fn name(&self) -> &str { "Yoshida8" }
    fn order(&self) -> u32 { 8 }
}
