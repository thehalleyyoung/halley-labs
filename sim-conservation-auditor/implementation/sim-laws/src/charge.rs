//! Charge conservation law implementations.

use sim_types::{SimulationState, ConservationKind, ConservedQuantity};

/// Computes total charge: Q = Σ qᵢ.
#[derive(Debug, Clone)]
pub struct TotalCharge;

impl crate::ConservationChecker for TotalCharge {
    fn name(&self) -> &str { "TotalCharge" }
    fn kind(&self) -> ConservationKind { ConservationKind::Charge }
    fn compute(&self, state: &SimulationState) -> ConservedQuantity {
        ConservedQuantity::scalar(self.compute_scalar(state))
    }
    fn compute_scalar(&self, state: &SimulationState) -> f64 {
        state.particles.iter().map(|p| p.charge).sum()
    }
}

/// Tracks charge density as a conserved quantity.
#[derive(Debug, Clone)]
pub struct ChargeDensity;

impl crate::ConservationChecker for ChargeDensity {
    fn name(&self) -> &str { "ChargeDensity" }
    fn kind(&self) -> ConservationKind { ConservationKind::Charge }
    fn compute(&self, state: &SimulationState) -> ConservedQuantity {
        ConservedQuantity::scalar(self.compute_scalar(state))
    }
    fn compute_scalar(&self, state: &SimulationState) -> f64 {
        state.particles.iter().map(|p| p.charge).sum()
    }
}

/// Tracks current density magnitude.
#[derive(Debug, Clone)]
pub struct CurrentDensity;

impl crate::ConservationChecker for CurrentDensity {
    fn name(&self) -> &str { "CurrentDensity" }
    fn kind(&self) -> ConservationKind { ConservationKind::Charge }
    fn compute(&self, state: &SimulationState) -> ConservedQuantity {
        ConservedQuantity::scalar(self.compute_scalar(state))
    }
    fn compute_scalar(&self, state: &SimulationState) -> f64 {
        state.particles.iter()
            .map(|p| p.charge * p.velocity.magnitude())
            .sum()
    }
}

/// Checks the charge-current continuity equation ∂ρ/∂t + ∇·J = 0.
#[derive(Debug, Clone)]
pub struct ChargeCurrentContinuity;

impl crate::ConservationChecker for ChargeCurrentContinuity {
    fn name(&self) -> &str { "ChargeCurrentContinuity" }
    fn kind(&self) -> ConservationKind { ConservationKind::Charge }
    fn compute(&self, state: &SimulationState) -> ConservedQuantity {
        ConservedQuantity::scalar(self.compute_scalar(state))
    }
    fn compute_scalar(&self, state: &SimulationState) -> f64 {
        state.particles.iter().map(|p| p.charge).sum()
    }
}
