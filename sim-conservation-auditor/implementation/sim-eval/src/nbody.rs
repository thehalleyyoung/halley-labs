//! N-body benchmark problems.
use sim_types::{Particle, Vec3, SimulationState};

/// Figure-eight three-body orbit (Chenciner-Montgomery).
#[derive(Debug, Clone, Default)]
pub struct FigureEightOrbit;
impl FigureEightOrbit {
    pub fn setup(&self) -> SimulationState {
        SimulationState::new(vec![
            Particle::new(1.0, Vec3::new(-0.97000436, 0.24308753, 0.0), Vec3::new(0.4662036850, 0.4323657300, 0.0)),
            Particle::new(1.0, Vec3::new(0.97000436, -0.24308753, 0.0), Vec3::new(0.4662036850, 0.4323657300, 0.0)),
            Particle::new(1.0, Vec3::new(0.0, 0.0, 0.0), Vec3::new(-0.9324073700, -0.8647314600, 0.0)),
        ], 0.0)
    }
}

/// Pythagorean three-body problem.
#[derive(Debug, Clone, Default)]
pub struct PythagoreanThreeBody;
impl PythagoreanThreeBody {
    pub fn setup(&self) -> SimulationState {
        SimulationState::new(vec![
            Particle::new(3.0, Vec3::new(1.0, 3.0, 0.0), Vec3::ZERO),
            Particle::new(4.0, Vec3::new(-2.0, -1.0, 0.0), Vec3::ZERO),
            Particle::new(5.0, Vec3::new(1.0, -1.0, 0.0), Vec3::ZERO),
        ], 0.0)
    }
}

/// Inner solar system (Sun + 4 planets).
#[derive(Debug, Clone, Default)]
pub struct SolarSystemInner;
impl SolarSystemInner {
    pub fn setup(&self) -> SimulationState {
        SimulationState::new(vec![
            Particle::new(1.0, Vec3::ZERO, Vec3::ZERO),
            Particle::new(1.66e-7, Vec3::new(0.387, 0.0, 0.0), Vec3::new(0.0, 8.17, 0.0)),
            Particle::new(2.45e-6, Vec3::new(0.723, 0.0, 0.0), Vec3::new(0.0, 5.84, 0.0)),
            Particle::new(3.00e-6, Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 5.26, 0.0)),
            Particle::new(3.23e-7, Vec3::new(1.524, 0.0, 0.0), Vec3::new(0.0, 4.02, 0.0)),
        ], 0.0)
    }
}

/// Plummer model (N-body galaxy model).
#[derive(Debug, Clone)]
pub struct PlummerModel { pub n: usize, pub total_mass: f64 }
impl Default for PlummerModel { fn default() -> Self { Self { n: 100, total_mass: 1.0 } } }
