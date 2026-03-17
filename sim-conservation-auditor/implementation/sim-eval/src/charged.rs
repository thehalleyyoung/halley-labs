//! Charged particle benchmarks.
use sim_types::{Particle, Vec3, SimulationState};

/// Cyclotron motion in uniform magnetic field.
#[derive(Debug, Clone)]
pub struct CyclotronMotion { pub b_field: f64, pub charge: f64, pub mass: f64 }
impl Default for CyclotronMotion { fn default() -> Self { Self { b_field: 1.0, charge: 1.0, mass: 1.0 } } }
impl CyclotronMotion {
    pub fn setup(&self, v_perp: f64) -> SimulationState {
        let mut p = Particle::new(self.mass, Vec3::ZERO, Vec3::new(v_perp, 0.0, 0.0));
        p.charge = self.charge;
        SimulationState::new(vec![p], 0.0)
    }
}

/// Coulomb scattering (Rutherford).
#[derive(Debug, Clone, Default)]
pub struct CoulombScattering;

/// E×B drift motion.
#[derive(Debug, Clone)]
pub struct ExBDrift { pub e_field: Vec3, pub b_field: Vec3 }
impl Default for ExBDrift { fn default() -> Self { Self { e_field: Vec3::new(1.0, 0.0, 0.0), b_field: Vec3::new(0.0, 0.0, 1.0) } } }

/// Magnetic bottle/mirror.
#[derive(Debug, Clone, Default)]
pub struct MagneticBottle;

/// Uniform field particle motion.
#[derive(Debug, Clone, Default)]
pub struct UniformFieldParticle;
