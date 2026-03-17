//! Energy conservation law implementations.

use sim_types::{SimulationState, ConservationKind, ConservedQuantity};

/// Computes total kinetic energy: T = Σ ½mv².
#[derive(Debug, Clone)]
pub struct KineticEnergy;

impl crate::ConservationChecker for KineticEnergy {
    fn name(&self) -> &str { "KineticEnergy" }
    fn kind(&self) -> ConservationKind { ConservationKind::Energy }
    fn compute(&self, state: &SimulationState) -> ConservedQuantity {
        ConservedQuantity::scalar(self.compute_scalar(state))
    }
    fn compute_scalar(&self, state: &SimulationState) -> f64 {
        state.particles.iter().map(|p| 0.5 * p.mass * p.velocity.magnitude_squared()).sum()
    }
}

/// Computes gravitational potential energy: V = -Σ G·mᵢ·mⱼ / rᵢⱼ.
#[derive(Debug, Clone)]
pub struct GravitationalPotentialEnergy;

impl crate::ConservationChecker for GravitationalPotentialEnergy {
    fn name(&self) -> &str { "GravitationalPotentialEnergy" }
    fn kind(&self) -> ConservationKind { ConservationKind::Energy }
    fn compute(&self, state: &SimulationState) -> ConservedQuantity {
        ConservedQuantity::scalar(self.compute_scalar(state))
    }
    fn compute_scalar(&self, state: &SimulationState) -> f64 {
        let ps = &state.particles;
        let mut total = 0.0;
        for i in 0..ps.len() {
            for j in (i+1)..ps.len() {
                let r = crate::safe_distance(&ps[i], &ps[j], 1e-10);
                total -= crate::G_SI * ps[i].mass * ps[j].mass / r;
            }
        }
        total
    }
}

/// Computes spring potential energy: V = ½kx².
#[derive(Debug, Clone)]
pub struct SpringPotentialEnergy {
    /// Spring constant.
    pub k: f64,
    /// Equilibrium length.
    pub equilibrium_length: f64,
}

impl crate::ConservationChecker for SpringPotentialEnergy {
    fn name(&self) -> &str { "SpringPotentialEnergy" }
    fn kind(&self) -> ConservationKind { ConservationKind::Energy }
    fn compute(&self, state: &SimulationState) -> ConservedQuantity {
        ConservedQuantity::scalar(self.compute_scalar(state))
    }
    fn compute_scalar(&self, state: &SimulationState) -> f64 {
        let ps = &state.particles;
        let mut total = 0.0;
        for i in 0..ps.len() {
            for j in (i+1)..ps.len() {
                let r = ps[i].position.distance(ps[j].position);
                let dx = r - self.equilibrium_length;
                total += 0.5 * self.k * dx * dx;
            }
        }
        total
    }
}

/// Computes electrostatic energy: V = Σ kₑ·qᵢ·qⱼ / rᵢⱼ.
#[derive(Debug, Clone)]
pub struct ElectrostaticEnergy;

impl crate::ConservationChecker for ElectrostaticEnergy {
    fn name(&self) -> &str { "ElectrostaticEnergy" }
    fn kind(&self) -> ConservationKind { ConservationKind::Energy }
    fn compute(&self, state: &SimulationState) -> ConservedQuantity {
        ConservedQuantity::scalar(self.compute_scalar(state))
    }
    fn compute_scalar(&self, state: &SimulationState) -> f64 {
        let ps = &state.particles;
        let mut total = 0.0;
        for i in 0..ps.len() {
            for j in (i+1)..ps.len() {
                let r = crate::safe_distance(&ps[i], &ps[j], 1e-10);
                total += crate::K_COULOMB_SI * ps[i].charge * ps[j].charge / r;
            }
        }
        total
    }
}

/// Computes total mechanical energy: E = T + V (kinetic + gravitational potential).
#[derive(Debug, Clone)]
pub struct TotalMechanicalEnergy;

impl crate::ConservationChecker for TotalMechanicalEnergy {
    fn name(&self) -> &str { "TotalMechanicalEnergy" }
    fn kind(&self) -> ConservationKind { ConservationKind::Energy }
    fn compute(&self, state: &SimulationState) -> ConservedQuantity {
        ConservedQuantity::scalar(self.compute_scalar(state))
    }
    fn compute_scalar(&self, state: &SimulationState) -> f64 {
        let ke = KineticEnergy.compute_scalar(state);
        let pe = GravitationalPotentialEnergy.compute_scalar(state);
        ke + pe
    }
}
