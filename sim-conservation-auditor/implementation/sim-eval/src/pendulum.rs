//! Pendulum benchmarks.
use sim_types::{Particle, Vec3, SimulationState};

/// Simple pendulum.
#[derive(Debug, Clone)]
pub struct SimplePendulum { pub length: f64, pub mass: f64, pub g: f64 }
impl Default for SimplePendulum { fn default() -> Self { Self { length: 1.0, mass: 1.0, g: 9.81 } } }
impl SimplePendulum {
    pub fn setup(&self, theta0: f64) -> SimulationState {
        let x = self.length * theta0.sin();
        let y = -self.length * theta0.cos();
        SimulationState::new(vec![Particle::new(self.mass, Vec3::new(x, y, 0.0), Vec3::ZERO)], 0.0)
    }
}

/// Double pendulum (chaotic).
#[derive(Debug, Clone)]
pub struct DoublePendulum { pub l1: f64, pub l2: f64, pub m1: f64, pub m2: f64 }
impl Default for DoublePendulum { fn default() -> Self { Self { l1: 1.0, l2: 1.0, m1: 1.0, m2: 1.0 } } }

/// Spherical pendulum.
#[derive(Debug, Clone)]
pub struct SphericalPendulum { pub length: f64, pub mass: f64 }
impl Default for SphericalPendulum { fn default() -> Self { Self { length: 1.0, mass: 1.0 } } }
