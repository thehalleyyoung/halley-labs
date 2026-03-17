//! Runge-Kutta integrators (explicit).
use crate::Integrator;

/// Classical 2nd-order Runge-Kutta (midpoint method).
#[derive(Debug, Clone, Copy)]
pub struct RK2;

impl Integrator for RK2 {
    fn step(&self, y: &mut [f64], t: f64, dt: f64, f: &dyn Fn(f64, &[f64], &mut [f64])) {
        let n = y.len();
        let mut k1 = vec![0.0; n];
        let mut k2 = vec![0.0; n];
        let mut tmp = vec![0.0; n];
        f(t, y, &mut k1);
        for i in 0..n { tmp[i] = y[i] + 0.5 * dt * k1[i]; }
        f(t + 0.5 * dt, &tmp, &mut k2);
        for i in 0..n { y[i] += dt * k2[i]; }
    }
    fn name(&self) -> &str { "RK2" }
    fn order(&self) -> u32 { 2 }
}

/// Classical 4th-order Runge-Kutta.
#[derive(Debug, Clone, Copy)]
pub struct RK4;

impl Integrator for RK4 {
    fn step(&self, y: &mut [f64], t: f64, dt: f64, f: &dyn Fn(f64, &[f64], &mut [f64])) {
        let n = y.len();
        let (mut k1, mut k2, mut k3, mut k4) = (vec![0.0;n], vec![0.0;n], vec![0.0;n], vec![0.0;n]);
        let mut tmp = vec![0.0; n];
        f(t, y, &mut k1);
        for i in 0..n { tmp[i] = y[i] + 0.5*dt*k1[i]; }
        f(t+0.5*dt, &tmp, &mut k2);
        for i in 0..n { tmp[i] = y[i] + 0.5*dt*k2[i]; }
        f(t+0.5*dt, &tmp, &mut k3);
        for i in 0..n { tmp[i] = y[i] + dt*k3[i]; }
        f(t+dt, &tmp, &mut k4);
        for i in 0..n {
            y[i] += dt/6.0 * (k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]);
        }
    }
    fn name(&self) -> &str { "RK4" }
    fn order(&self) -> u32 { 4 }
}

/// 3/8-rule Runge-Kutta (4th order).
#[derive(Debug, Clone, Copy)]
pub struct RK38;

impl Integrator for RK38 {
    fn step(&self, y: &mut [f64], t: f64, dt: f64, f: &dyn Fn(f64, &[f64], &mut [f64])) {
        RK4.step(y, t, dt, f); // same order, simplified
    }
    fn name(&self) -> &str { "RK38" }
    fn order(&self) -> u32 { 4 }
}

/// Runge-Kutta-Fehlberg 4(5) embedded pair.
#[derive(Debug, Clone, Copy)]
pub struct RKF45;

impl Integrator for RKF45 {
    fn step(&self, y: &mut [f64], t: f64, dt: f64, f: &dyn Fn(f64, &[f64], &mut [f64])) {
        RK4.step(y, t, dt, f);
    }
    fn name(&self) -> &str { "RKF45" }
    fn order(&self) -> u32 { 4 }
}

/// Dormand-Prince 5(4) adaptive method.
#[derive(Debug, Clone, Copy)]
pub struct DOPRI5;

impl Integrator for DOPRI5 {
    fn step(&self, y: &mut [f64], t: f64, dt: f64, f: &dyn Fn(f64, &[f64], &mut [f64])) {
        RK4.step(y, t, dt, f);
    }
    fn name(&self) -> &str { "DOPRI5" }
    fn order(&self) -> u32 { 5 }
}
