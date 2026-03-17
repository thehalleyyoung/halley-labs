//! Angular momentum conservation law implementations.

use sim_types::{SimulationState, ConservationKind, ConservedQuantity, Vec3};

/// Computes total angular momentum: L = Σ rᵢ × (mᵢvᵢ).
#[derive(Debug, Clone)]
pub struct TotalAngularMomentum;

impl crate::ConservationChecker for TotalAngularMomentum {
    fn name(&self) -> &str { "TotalAngularMomentum" }
    fn kind(&self) -> ConservationKind { ConservationKind::AngularMomentum }
    fn compute(&self, state: &SimulationState) -> ConservedQuantity {
        ConservedQuantity::scalar(self.compute_scalar(state))
    }
    fn compute_scalar(&self, state: &SimulationState) -> f64 {
        let l: Vec3 = state.particles.iter()
            .fold(Vec3::ZERO, |acc, p| acc + p.position.cross(p.velocity * p.mass));
        l.magnitude()
    }
}

/// Computes angular momentum about a specified point.
#[derive(Debug, Clone)]
pub struct AngularMomentumAboutPoint {
    /// Reference point.
    pub origin: Vec3,
}

impl crate::ConservationChecker for AngularMomentumAboutPoint {
    fn name(&self) -> &str { "AngularMomentumAboutPoint" }
    fn kind(&self) -> ConservationKind { ConservationKind::AngularMomentum }
    fn compute(&self, state: &SimulationState) -> ConservedQuantity {
        ConservedQuantity::scalar(self.compute_scalar(state))
    }
    fn compute_scalar(&self, state: &SimulationState) -> f64 {
        let l: Vec3 = state.particles.iter()
            .fold(Vec3::ZERO, |acc, p| {
                let r = p.position - self.origin;
                acc + r.cross(p.velocity * p.mass)
            });
        l.magnitude()
    }
}

/// Computes the spin angular momentum contribution.
#[derive(Debug, Clone)]
pub struct SpinAngularMomentum;

impl crate::ConservationChecker for SpinAngularMomentum {
    fn name(&self) -> &str { "SpinAngularMomentum" }
    fn kind(&self) -> ConservationKind { ConservationKind::AngularMomentum }
    fn compute(&self, state: &SimulationState) -> ConservedQuantity {
        ConservedQuantity::scalar(self.compute_scalar(state))
    }
    fn compute_scalar(&self, state: &SimulationState) -> f64 {
        state.particles.iter().map(|p| p.spin.magnitude()).sum()
    }
}

/// Computes the orbital angular momentum L_orbital = Σ rᵢ × pᵢ.
#[derive(Debug, Clone)]
pub struct OrbitalAngularMomentum;

impl crate::ConservationChecker for OrbitalAngularMomentum {
    fn name(&self) -> &str { "OrbitalAngularMomentum" }
    fn kind(&self) -> ConservationKind { ConservationKind::AngularMomentum }
    fn compute(&self, state: &SimulationState) -> ConservedQuantity {
        ConservedQuantity::scalar(self.compute_scalar(state))
    }
    fn compute_scalar(&self, state: &SimulationState) -> f64 {
        TotalAngularMomentum.compute_scalar(state)
    }
}

/// Computes the moment of inertia tensor trace.
#[derive(Debug, Clone)]
pub struct MomentOfInertia;

impl crate::ConservationChecker for MomentOfInertia {
    fn name(&self) -> &str { "MomentOfInertia" }
    fn kind(&self) -> ConservationKind { ConservationKind::AngularMomentum }
    fn compute(&self, state: &SimulationState) -> ConservedQuantity {
        ConservedQuantity::scalar(self.compute_scalar(state))
    }
    fn compute_scalar(&self, state: &SimulationState) -> f64 {
        state.particles.iter()
            .map(|p| p.mass * p.position.magnitude_squared())
            .sum()
    }
}
