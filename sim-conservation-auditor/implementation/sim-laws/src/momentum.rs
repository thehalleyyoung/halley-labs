//! Linear momentum conservation law implementations.

use sim_types::{SimulationState, ConservationKind, ConservedQuantity, Vec3};

/// Computes total linear momentum: p = Σ mᵢvᵢ.
#[derive(Debug, Clone)]
pub struct TotalLinearMomentum;

impl crate::ConservationChecker for TotalLinearMomentum {
    fn name(&self) -> &str { "TotalLinearMomentum" }
    fn kind(&self) -> ConservationKind { ConservationKind::Momentum }
    fn compute(&self, state: &SimulationState) -> ConservedQuantity {
        ConservedQuantity::scalar(self.compute_scalar(state))
    }
    fn compute_scalar(&self, state: &SimulationState) -> f64 {
        let p: Vec3 = state.particles.iter()
            .fold(Vec3::ZERO, |acc, part| acc + part.velocity * part.mass);
        p.magnitude()
    }
}

/// Computes individual momentum components (px, py, pz).
#[derive(Debug, Clone)]
pub struct MomentumComponents;

impl MomentumComponents {
    /// Compute momentum vector for a state.
    pub fn compute_vector(&self, state: &SimulationState) -> Vec3 {
        state.particles.iter()
            .fold(Vec3::ZERO, |acc, p| acc + p.velocity * p.mass)
    }
}

impl crate::ConservationChecker for MomentumComponents {
    fn name(&self) -> &str { "MomentumComponents" }
    fn kind(&self) -> ConservationKind { ConservationKind::Momentum }
    fn compute(&self, state: &SimulationState) -> ConservedQuantity {
        ConservedQuantity::scalar(self.compute_scalar(state))
    }
    fn compute_scalar(&self, state: &SimulationState) -> f64 {
        self.compute_vector(state).magnitude()
    }
}

/// Tracks the velocity of the center of mass as a conserved quantity.
#[derive(Debug, Clone)]
pub struct CenterOfMassVelocityLaw;

impl crate::ConservationChecker for CenterOfMassVelocityLaw {
    fn name(&self) -> &str { "CenterOfMassVelocity" }
    fn kind(&self) -> ConservationKind { ConservationKind::Momentum }
    fn compute(&self, state: &SimulationState) -> ConservedQuantity {
        ConservedQuantity::scalar(self.compute_scalar(state))
    }
    fn compute_scalar(&self, state: &SimulationState) -> f64 {
        let total_mass: f64 = state.particles.iter().map(|p| p.mass).sum();
        if total_mass.abs() < 1e-30 { return 0.0; }
        let total_momentum: Vec3 = state.particles.iter()
            .fold(Vec3::ZERO, |acc, p| acc + p.velocity * p.mass);
        (total_momentum / total_mass).magnitude()
    }
}

/// Computes impulse (change in momentum) for a state.
#[derive(Debug, Clone)]
pub struct ImpulseCalculation;

impl crate::ConservationChecker for ImpulseCalculation {
    fn name(&self) -> &str { "Impulse" }
    fn kind(&self) -> ConservationKind { ConservationKind::Momentum }
    fn compute(&self, state: &SimulationState) -> ConservedQuantity {
        ConservedQuantity::scalar(self.compute_scalar(state))
    }
    fn compute_scalar(&self, state: &SimulationState) -> f64 {
        TotalLinearMomentum.compute_scalar(state)
    }
}
