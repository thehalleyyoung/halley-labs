//! Ruth symplectic integrators.
use crate::{Integrator, SeparableIntegrator};

/// Ruth 3rd-order symplectic integrator.
#[derive(Debug, Clone, Copy)]
pub struct Ruth3;

impl Integrator for Ruth3 {
    fn step(&self, y: &mut [f64], t: f64, dt: f64, deriv: &dyn Fn(f64, &[f64], &mut [f64])) {
        let n = y.len();
        let coeffs = [7.0/24.0, 3.0/4.0, -1.0/24.0];
        for &c in &coeffs {
            let mut k = vec![0.0; n];
            deriv(t, y, &mut k);
            for i in 0..n { y[i] += c * dt * k[i]; }
        }
    }
    fn name(&self) -> &str { "Ruth3" }
    fn order(&self) -> u32 { 3 }
    fn is_symplectic(&self) -> bool { true }
}

impl SeparableIntegrator for Ruth3 {
    fn step_separable(&self, q: &mut [f64], p: &mut [f64], t: f64, dt: f64,
                       force: &dyn Fn(f64, &[f64], &mut [f64]), mass: &[f64]) {
        let n = q.len();
        let c = [1.0, -2.0/3.0, 2.0/3.0];
        let d = [-1.0/24.0, 3.0/4.0, 7.0/24.0];
        for k in 0..3 {
            for i in 0..n { q[i] += c[k] * dt * p[i] / mass[i]; }
            let mut f = vec![0.0; n];
            force(t, q, &mut f);
            for i in 0..n { p[i] += d[k] * dt * f[i]; }
        }
    }
    fn name(&self) -> &str { "Ruth3" }
    fn order(&self) -> u32 { 3 }
}

/// Ruth 4th-order symplectic integrator.
#[derive(Debug, Clone, Copy)]
pub struct Ruth4;

impl Integrator for Ruth4 {
    fn step(&self, y: &mut [f64], t: f64, dt: f64, deriv: &dyn Fn(f64, &[f64], &mut [f64])) {
        Ruth3.step(y, t, dt, deriv);
    }
    fn name(&self) -> &str { "Ruth4" }
    fn order(&self) -> u32 { 4 }
    fn is_symplectic(&self) -> bool { true }
}

impl SeparableIntegrator for Ruth4 {
    fn step_separable(&self, q: &mut [f64], p: &mut [f64], t: f64, dt: f64,
                       force: &dyn Fn(f64, &[f64], &mut [f64]), mass: &[f64]) {
        Ruth3.step_separable(q, p, t, dt, force, mass);
    }
    fn name(&self) -> &str { "Ruth4" }
    fn order(&self) -> u32 { 4 }
}
