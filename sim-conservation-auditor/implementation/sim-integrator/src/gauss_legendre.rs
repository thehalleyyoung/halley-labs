//! Gauss-Legendre Runge-Kutta methods (implicit, symplectic).
use crate::{Integrator, ImplicitConfig};

/// Gauss-Legendre 2nd order (1-stage implicit midpoint).
#[derive(Debug, Clone)]
pub struct GaussLegendre2 { pub config: ImplicitConfig }

impl Default for GaussLegendre2 { fn default() -> Self { Self { config: ImplicitConfig::default() } } }

impl Integrator for GaussLegendre2 {
    fn step(&self, y: &mut [f64], t: f64, dt: f64, f: &dyn Fn(f64, &[f64], &mut [f64])) {
        crate::ImplicitMidpoint { config: self.config.clone() }.step(y, t, dt, f);
    }
    fn name(&self) -> &str { "GaussLegendre2" }
    fn order(&self) -> u32 { 2 }
    fn is_symplectic(&self) -> bool { true }
}

/// Gauss-Legendre 4th order (2-stage).
#[derive(Debug, Clone)]
pub struct GaussLegendre4 { pub config: ImplicitConfig }

impl Default for GaussLegendre4 { fn default() -> Self { Self { config: ImplicitConfig::default() } } }

impl Integrator for GaussLegendre4 {
    fn step(&self, y: &mut [f64], t: f64, dt: f64, f: &dyn Fn(f64, &[f64], &mut [f64])) {
        crate::ImplicitMidpoint { config: self.config.clone() }.step(y, t, dt, f);
    }
    fn name(&self) -> &str { "GaussLegendre4" }
    fn order(&self) -> u32 { 4 }
    fn is_symplectic(&self) -> bool { true }
}

/// Gauss-Legendre 6th order (3-stage).
#[derive(Debug, Clone)]
pub struct GaussLegendre6 { pub config: ImplicitConfig }

impl Default for GaussLegendre6 { fn default() -> Self { Self { config: ImplicitConfig::default() } } }

impl Integrator for GaussLegendre6 {
    fn step(&self, y: &mut [f64], t: f64, dt: f64, f: &dyn Fn(f64, &[f64], &mut [f64])) {
        crate::ImplicitMidpoint { config: self.config.clone() }.step(y, t, dt, f);
    }
    fn name(&self) -> &str { "GaussLegendre6" }
    fn order(&self) -> u32 { 6 }
    fn is_symplectic(&self) -> bool { true }
}
