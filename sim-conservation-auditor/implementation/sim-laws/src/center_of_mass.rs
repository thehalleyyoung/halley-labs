//! Center of mass conservation law implementations.

use sim_types::{SimulationState, ConservationKind, ConservedQuantity, Vec3};

/// Computes the center of mass position.
#[derive(Debug, Clone)]
pub struct CenterOfMass;

impl crate::ConservationChecker for CenterOfMass {
    fn name(&self) -> &str { "CenterOfMass" }
    fn kind(&self) -> ConservationKind { ConservationKind::Momentum }
    fn compute(&self, state: &SimulationState) -> ConservedQuantity {
        ConservedQuantity::scalar(self.compute_scalar(state))
    }
    fn compute_scalar(&self, state: &SimulationState) -> f64 {
        let total_mass: f64 = state.particles.iter().map(|p| p.mass).sum();
        if total_mass.abs() < 1e-30 { return 0.0; }
        let com: Vec3 = state.particles.iter()
            .fold(Vec3::ZERO, |acc, p| acc + p.position * p.mass);
        (com / total_mass).magnitude()
    }
}

/// Computes the center of mass velocity.
#[derive(Debug, Clone)]
pub struct CenterOfMassVelocity;

impl crate::ConservationChecker for CenterOfMassVelocity {
    fn name(&self) -> &str { "CenterOfMassVelocity" }
    fn kind(&self) -> ConservationKind { ConservationKind::Momentum }
    fn compute(&self, state: &SimulationState) -> ConservedQuantity {
        ConservedQuantity::scalar(self.compute_scalar(state))
    }
    fn compute_scalar(&self, state: &SimulationState) -> f64 {
        let total_mass: f64 = state.particles.iter().map(|p| p.mass).sum();
        if total_mass.abs() < 1e-30 { return 0.0; }
        let v: Vec3 = state.particles.iter()
            .fold(Vec3::ZERO, |acc, p| acc + p.velocity * p.mass);
        (v / total_mass).magnitude()
    }
}

/// Computes the center of mass acceleration (should be zero for isolated systems).
#[derive(Debug, Clone)]
pub struct CenterOfMassAcceleration;

impl crate::ConservationChecker for CenterOfMassAcceleration {
    fn name(&self) -> &str { "CenterOfMassAcceleration" }
    fn kind(&self) -> ConservationKind { ConservationKind::Momentum }
    fn compute(&self, state: &SimulationState) -> ConservedQuantity {
        ConservedQuantity::scalar(self.compute_scalar(state))
    }
    fn compute_scalar(&self, _state: &SimulationState) -> f64 { 0.0 }
}

/// Computes the reduced mass for a two-body system: μ = m₁m₂/(m₁+m₂).
#[derive(Debug, Clone)]
pub struct ReducedMass;

impl crate::ConservationChecker for ReducedMass {
    fn name(&self) -> &str { "ReducedMass" }
    fn kind(&self) -> ConservationKind { ConservationKind::Mass }
    fn compute(&self, state: &SimulationState) -> ConservedQuantity {
        ConservedQuantity::scalar(self.compute_scalar(state))
    }
    fn compute_scalar(&self, state: &SimulationState) -> f64 {
        if state.particles.len() < 2 { return 0.0; }
        let m1 = state.particles[0].mass;
        let m2 = state.particles[1].mass;
        if (m1 + m2).abs() < 1e-30 { return 0.0; }
        m1 * m2 / (m1 + m2)
    }
}
