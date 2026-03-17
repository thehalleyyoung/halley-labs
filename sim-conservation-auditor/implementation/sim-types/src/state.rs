use serde::{Deserialize, Serialize};
use crate::particle::Particle;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationState {
    pub particles: Vec<Particle>,
    pub time: f64,
}

impl SimulationState {
    pub fn new(particles: Vec<Particle>, time: f64) -> Self {
        Self { particles, time }
    }

    pub fn num_particles(&self) -> usize {
        self.particles.len()
    }

    pub fn total_mass(&self) -> f64 {
        self.particles.iter().map(|p| p.mass).sum()
    }

    pub fn total_kinetic_energy(&self) -> f64 {
        self.particles
            .iter()
            .map(|p| 0.5 * p.mass * p.velocity.magnitude_squared())
            .sum()
    }

    pub fn total_momentum(&self) -> crate::Vec3 {
        self.particles
            .iter()
            .fold(crate::Vec3::ZERO, |acc, p| acc + p.velocity * p.mass)
    }

    pub fn total_angular_momentum(&self) -> crate::Vec3 {
        self.particles
            .iter()
            .fold(crate::Vec3::ZERO, |acc, p| {
                acc + p.position.cross(p.velocity) * p.mass
            })
    }
}
