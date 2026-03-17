//! Spring/oscillator benchmarks.
use sim_types::{Particle, Vec3, SimulationState};

/// Simple harmonic oscillator: x(t) = A cos(ωt + φ).
#[derive(Debug, Clone)]
pub struct SimpleHarmonicOscillator { pub k: f64, pub mass: f64, pub amplitude: f64 }
impl Default for SimpleHarmonicOscillator { fn default() -> Self { Self { k: 1.0, mass: 1.0, amplitude: 1.0 } } }
impl SimpleHarmonicOscillator {
    pub fn setup(&self) -> SimulationState {
        SimulationState::new(vec![Particle::new(self.mass, Vec3::new(self.amplitude, 0.0, 0.0), Vec3::ZERO)], 0.0)
    }
    pub fn exact_energy(&self) -> f64 { 0.5 * self.k * self.amplitude * self.amplitude }
}

/// Anharmonic oscillator: V(x) = ½kx² + ¼λx⁴.
#[derive(Debug, Clone)]
pub struct AnharmonicOscillator { pub k: f64, pub lambda: f64, pub mass: f64 }
impl Default for AnharmonicOscillator { fn default() -> Self { Self { k: 1.0, lambda: 0.1, mass: 1.0 } } }

/// Coupled oscillators.
#[derive(Debug, Clone)]
pub struct CoupledOscillators { pub n: usize, pub k: f64, pub coupling: f64 }
impl Default for CoupledOscillators { fn default() -> Self { Self { n: 10, k: 1.0, coupling: 0.1 } } }

/// Damped oscillator (non-conservative, for testing violation detection).
#[derive(Debug, Clone)]
pub struct DampedOscillator { pub k: f64, pub damping: f64 }
impl Default for DampedOscillator { fn default() -> Self { Self { k: 1.0, damping: 0.1 } } }

/// Driven oscillator (forced, non-conservative).
#[derive(Debug, Clone)]
pub struct DrivenOscillator { pub k: f64, pub drive_amplitude: f64, pub drive_frequency: f64 }
impl Default for DrivenOscillator { fn default() -> Self { Self { k: 1.0, drive_amplitude: 0.5, drive_frequency: 1.0 } } }
