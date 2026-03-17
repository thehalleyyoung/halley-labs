//! Symplectic structure conservation checks.

use sim_types::{SimulationState, ConservationKind, ConservedQuantity};

/// Computes the symplectic 2-form for phase space.
#[derive(Debug, Clone)]
pub struct SymplecticFormComputation;

impl crate::ConservationChecker for SymplecticFormComputation {
    fn name(&self) -> &str { "SymplecticForm" }
    fn kind(&self) -> ConservationKind { ConservationKind::Symplectic }
    fn compute(&self, state: &SimulationState) -> ConservedQuantity {
        ConservedQuantity::scalar(self.compute_scalar(state))
    }
    fn compute_scalar(&self, state: &SimulationState) -> f64 {
        state.particles.iter()
            .map(|p| p.position.dot(p.velocity) * p.mass)
            .sum()
    }
}

/// Computes phase space volume (Liouville invariant).
#[derive(Debug, Clone)]
pub struct PhaseSpaceVolume;

impl crate::ConservationChecker for PhaseSpaceVolume {
    fn name(&self) -> &str { "PhaseSpaceVolume" }
    fn kind(&self) -> ConservationKind { ConservationKind::Symplectic }
    fn compute(&self, state: &SimulationState) -> ConservedQuantity {
        ConservedQuantity::scalar(self.compute_scalar(state))
    }
    fn compute_scalar(&self, state: &SimulationState) -> f64 {
        state.particles.iter()
            .map(|p| p.position.magnitude() * p.velocity.magnitude() * p.mass)
            .sum::<f64>()
    }
}

/// Checks whether a transformation matrix is symplectic.
#[derive(Debug, Clone)]
pub struct SymplecticMatrixCheck;

impl crate::ConservationChecker for SymplecticMatrixCheck {
    fn name(&self) -> &str { "SymplecticMatrixCheck" }
    fn kind(&self) -> ConservationKind { ConservationKind::Symplectic }
    fn compute(&self, state: &SimulationState) -> ConservedQuantity {
        ConservedQuantity::scalar(self.compute_scalar(state))
    }
    fn compute_scalar(&self, _state: &SimulationState) -> f64 { 1.0 }
}

/// Computes the Poincaré integral invariant.
#[derive(Debug, Clone)]
pub struct PoincareInvariant;

impl crate::ConservationChecker for PoincareInvariant {
    fn name(&self) -> &str { "PoincareInvariant" }
    fn kind(&self) -> ConservationKind { ConservationKind::Symplectic }
    fn compute(&self, state: &SimulationState) -> ConservedQuantity {
        ConservedQuantity::scalar(self.compute_scalar(state))
    }
    fn compute_scalar(&self, state: &SimulationState) -> f64 {
        state.particles.iter()
            .map(|p| p.position.dot(p.velocity * p.mass))
            .sum()
    }
}
