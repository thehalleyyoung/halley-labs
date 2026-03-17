//! Vorticity and circulation conservation laws.

use sim_types::{SimulationState, ConservationKind, ConservedQuantity};

/// Computes vorticity magnitude from particle velocities.
#[derive(Debug, Clone)]
pub struct Vorticity;

impl crate::ConservationChecker for Vorticity {
    fn name(&self) -> &str { "Vorticity" }
    fn kind(&self) -> ConservationKind { ConservationKind::Vorticity }
    fn compute(&self, state: &SimulationState) -> ConservedQuantity {
        ConservedQuantity::scalar(self.compute_scalar(state))
    }
    fn compute_scalar(&self, state: &SimulationState) -> f64 {
        let n = state.particles.len();
        if n < 2 { return 0.0; }
        let mut vort = 0.0;
        for i in 0..n {
            for j in (i+1)..n {
                let dr = state.particles[j].position - state.particles[i].position;
                let dv = state.particles[j].velocity - state.particles[i].velocity;
                vort += dr.cross(dv).magnitude();
            }
        }
        vort / n as f64
    }
}

/// Computes circulation around a particle configuration.
#[derive(Debug, Clone)]
pub struct Circulation;

impl crate::ConservationChecker for Circulation {
    fn name(&self) -> &str { "Circulation" }
    fn kind(&self) -> ConservationKind { ConservationKind::Vorticity }
    fn compute(&self, state: &SimulationState) -> ConservedQuantity {
        ConservedQuantity::scalar(self.compute_scalar(state))
    }
    fn compute_scalar(&self, state: &SimulationState) -> f64 {
        let n = state.particles.len();
        if n < 2 { return 0.0; }
        let mut circ = 0.0;
        for i in 0..n {
            let j = (i + 1) % n;
            let dr = state.particles[j].position - state.particles[i].position;
            let avg_v = (state.particles[i].velocity + state.particles[j].velocity) * 0.5;
            circ += dr.dot(avg_v);
        }
        circ
    }
}

/// Checks Kelvin's circulation theorem.
#[derive(Debug, Clone)]
pub struct KelvinCirculationTheorem;

impl crate::ConservationChecker for KelvinCirculationTheorem {
    fn name(&self) -> &str { "KelvinCirculation" }
    fn kind(&self) -> ConservationKind { ConservationKind::Vorticity }
    fn compute(&self, state: &SimulationState) -> ConservedQuantity {
        ConservedQuantity::scalar(Circulation.compute_scalar(state))
    }
    fn compute_scalar(&self, state: &SimulationState) -> f64 {
        Circulation.compute_scalar(state)
    }
}

/// Computes enstrophy (integral of vorticity squared).
#[derive(Debug, Clone)]
pub struct Enstrophy;

impl crate::ConservationChecker for Enstrophy {
    fn name(&self) -> &str { "Enstrophy" }
    fn kind(&self) -> ConservationKind { ConservationKind::Vorticity }
    fn compute(&self, state: &SimulationState) -> ConservedQuantity {
        ConservedQuantity::scalar(self.compute_scalar(state))
    }
    fn compute_scalar(&self, state: &SimulationState) -> f64 {
        let v = Vorticity.compute_scalar(state);
        v * v
    }
}
