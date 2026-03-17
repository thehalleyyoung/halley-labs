//! N-body simulation helpers.
use serde::{Serialize, Deserialize};

/// State of an N-body system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NBodyState {
    /// Positions (flattened: [x0,y0,z0, x1,y1,z1, ...]).
    pub positions: Vec<f64>,
    /// Velocities (same layout).
    pub velocities: Vec<f64>,
    /// Masses.
    pub masses: Vec<f64>,
}

impl NBodyState {
    /// Create a new N-body state.
    pub fn new(n: usize) -> Self {
        Self {
            positions: vec![0.0; 3 * n],
            velocities: vec![0.0; 3 * n],
            masses: vec![1.0; n],
        }
    }
    /// Number of bodies.
    pub fn num_bodies(&self) -> usize { self.masses.len() }
    /// Total kinetic energy.
    pub fn kinetic_energy(&self) -> f64 {
        let mut ke = 0.0;
        for i in 0..self.num_bodies() {
            let vx = self.velocities[3*i];
            let vy = self.velocities[3*i+1];
            let vz = self.velocities[3*i+2];
            ke += 0.5 * self.masses[i] * (vx*vx + vy*vy + vz*vz);
        }
        ke
    }
}

/// N-body gravitational simulation.
#[derive(Debug, Clone)]
pub struct NBodySimulation {
    /// Gravitational constant.
    pub g: f64,
    /// Softening parameter.
    pub softening: f64,
}

impl Default for NBodySimulation {
    fn default() -> Self { Self { g: 1.0, softening: 1e-4 } }
}

impl NBodySimulation {
    /// Compute gravitational forces on all bodies.
    pub fn compute_forces(&self, state: &NBodyState) -> Vec<f64> {
        let n = state.num_bodies();
        let mut forces = vec![0.0; 3 * n];
        for i in 0..n {
            for j in (i+1)..n {
                let dx = state.positions[3*j] - state.positions[3*i];
                let dy = state.positions[3*j+1] - state.positions[3*i+1];
                let dz = state.positions[3*j+2] - state.positions[3*i+2];
                let r2 = dx*dx + dy*dy + dz*dz + self.softening * self.softening;
                let r = r2.sqrt();
                let f = self.g * state.masses[i] * state.masses[j] / (r * r2);
                forces[3*i] += f * dx; forces[3*i+1] += f * dy; forces[3*i+2] += f * dz;
                forces[3*j] -= f * dx; forces[3*j+1] -= f * dy; forces[3*j+2] -= f * dz;
            }
        }
        forces
    }
}
