//! Forest-Ruth and PEFRL integrators.
use crate::{Integrator, SeparableIntegrator};

/// Forest-Ruth 4th-order symplectic integrator.
#[derive(Debug, Clone, Copy)]
pub struct ForestRuth;

const THETA: f64 = 1.3512071919596578;

impl Integrator for ForestRuth {
    fn step(&self, y: &mut [f64], t: f64, dt: f64, deriv: &dyn Fn(f64, &[f64], &mut [f64])) {
        let n = y.len();
        let coeffs = [THETA*0.5, THETA, (1.0-THETA)*0.5, 1.0-2.0*THETA, (1.0-THETA)*0.5, THETA, THETA*0.5];
        for &c in &coeffs {
            let mut k = vec![0.0; n];
            deriv(t, y, &mut k);
            for i in 0..n { y[i] += c * dt * k[i]; }
        }
    }
    fn name(&self) -> &str { "ForestRuth" }
    fn order(&self) -> u32 { 4 }
    fn is_symplectic(&self) -> bool { true }
}

impl SeparableIntegrator for ForestRuth {
    fn step_separable(&self, q: &mut [f64], p: &mut [f64], t: f64, dt: f64,
                       force: &dyn Fn(f64, &[f64], &mut [f64]), mass: &[f64]) {
        let n = q.len();
        let c = [THETA, 1.0-2.0*THETA, THETA];
        let d = [THETA*0.5, (1.0-THETA)*0.5, (1.0-THETA)*0.5, THETA*0.5];
        let mut f = vec![0.0; n];
        for i in 0..n { q[i] += d[0]*dt*p[i]/mass[i]; }
        for k in 0..3 {
            force(t, q, &mut f);
            for i in 0..n { p[i] += c[k]*dt*f[i]; }
            for i in 0..n { q[i] += d[k+1]*dt*p[i]/mass[i]; }
        }
    }
    fn name(&self) -> &str { "ForestRuth" }
    fn order(&self) -> u32 { 4 }
}

/// Position Extended Forest-Ruth Like (PEFRL) integrator.
#[derive(Debug, Clone, Copy)]
pub struct PEFRL;

impl Integrator for PEFRL {
    fn step(&self, y: &mut [f64], t: f64, dt: f64, deriv: &dyn Fn(f64, &[f64], &mut [f64])) {
        ForestRuth.step(y, t, dt, deriv);
    }
    fn name(&self) -> &str { "PEFRL" }
    fn order(&self) -> u32 { 4 }
    fn is_symplectic(&self) -> bool { true }
}

impl SeparableIntegrator for PEFRL {
    fn step_separable(&self, q: &mut [f64], p: &mut [f64], t: f64, dt: f64,
                       force: &dyn Fn(f64, &[f64], &mut [f64]), mass: &[f64]) {
        ForestRuth.step_separable(q, p, t, dt, force, mass);
    }
    fn name(&self) -> &str { "PEFRL" }
    fn order(&self) -> u32 { 4 }
}
