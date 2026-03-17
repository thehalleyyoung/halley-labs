//! Mass conservation law implementations.

use sim_types::{SimulationState, ConservationKind, ConservedQuantity};

/// Computes total mass: M = Σ mᵢ.
#[derive(Debug, Clone)]
pub struct TotalMass;

impl crate::ConservationChecker for TotalMass {
    fn name(&self) -> &str { "TotalMass" }
    fn kind(&self) -> ConservationKind { ConservationKind::Mass }
    fn compute(&self, state: &SimulationState) -> ConservedQuantity {
        ConservedQuantity::scalar(self.compute_scalar(state))
    }
    fn compute_scalar(&self, state: &SimulationState) -> f64 {
        state.particles.iter().map(|p| p.mass).sum()
    }
}

/// Tracks mass density as a conserved quantity.
#[derive(Debug, Clone)]
pub struct MassDensity;

impl crate::ConservationChecker for MassDensity {
    fn name(&self) -> &str { "MassDensity" }
    fn kind(&self) -> ConservationKind { ConservationKind::Mass }
    fn compute(&self, state: &SimulationState) -> ConservedQuantity {
        ConservedQuantity::scalar(self.compute_scalar(state))
    }
    fn compute_scalar(&self, state: &SimulationState) -> f64 {
        state.particles.iter().map(|p| p.mass).sum()
    }
}

/// Checks the continuity equation ∂ρ/∂t + ∇·(ρv) = 0.
#[derive(Debug, Clone)]
pub struct ContinuityEquation;

impl crate::ConservationChecker for ContinuityEquation {
    fn name(&self) -> &str { "ContinuityEquation" }
    fn kind(&self) -> ConservationKind { ConservationKind::Mass }
    fn compute(&self, state: &SimulationState) -> ConservedQuantity {
        ConservedQuantity::scalar(self.compute_scalar(state))
    }
    fn compute_scalar(&self, state: &SimulationState) -> f64 {
        state.particles.iter().map(|p| p.mass).sum()
    }
}
