//! Exact analytical solutions for benchmarks.

/// Trait for exact solutions.
pub trait ExactSolution { fn position(&self, t: f64) -> (f64, f64, f64); fn velocity(&self, t: f64) -> (f64, f64, f64); fn energy(&self) -> f64; }

/// Free particle: x(t) = x₀ + v₀t.
#[derive(Debug, Clone)]
pub struct FreeParticleSolution { pub x0: f64, pub v0: f64, pub mass: f64 }
impl ExactSolution for FreeParticleSolution {
    fn position(&self, t: f64) -> (f64, f64, f64) { (self.x0 + self.v0 * t, 0.0, 0.0) }
    fn velocity(&self, _t: f64) -> (f64, f64, f64) { (self.v0, 0.0, 0.0) }
    fn energy(&self) -> f64 { 0.5 * self.mass * self.v0 * self.v0 }
}

/// Harmonic oscillator: x(t) = A cos(ωt).
#[derive(Debug, Clone)]
pub struct HarmonicOscillatorSolution { pub amplitude: f64, pub omega: f64, pub mass: f64 }
impl ExactSolution for HarmonicOscillatorSolution {
    fn position(&self, t: f64) -> (f64, f64, f64) { (self.amplitude * (self.omega * t).cos(), 0.0, 0.0) }
    fn velocity(&self, t: f64) -> (f64, f64, f64) { (-self.amplitude * self.omega * (self.omega * t).sin(), 0.0, 0.0) }
    fn energy(&self) -> f64 { 0.5 * self.mass * self.omega * self.omega * self.amplitude * self.amplitude }
}

/// Kepler orbit solution.
#[derive(Debug, Clone)]
pub struct KeplerOrbitSolution { pub semi_major: f64, pub eccentricity: f64, pub mu: f64 }
impl ExactSolution for KeplerOrbitSolution {
    fn position(&self, t: f64) -> (f64, f64, f64) {
        let r = self.semi_major * (1.0 - self.eccentricity);
        let omega = (self.mu / (self.semi_major.powi(3))).sqrt();
        (r * (omega * t).cos(), r * (omega * t).sin(), 0.0)
    }
    fn velocity(&self, t: f64) -> (f64, f64, f64) {
        let omega = (self.mu / (self.semi_major.powi(3))).sqrt();
        let r = self.semi_major * (1.0 - self.eccentricity);
        (-r * omega * (omega * t).sin(), r * omega * (omega * t).cos(), 0.0)
    }
    fn energy(&self) -> f64 { -self.mu / (2.0 * self.semi_major) }
}

/// Uniform gravity: x(t) = x₀ + v₀t + ½gt².
#[derive(Debug, Clone)]
pub struct UniformGravitySolution { pub x0: f64, pub v0: f64, pub g: f64, pub mass: f64 }
impl ExactSolution for UniformGravitySolution {
    fn position(&self, t: f64) -> (f64, f64, f64) { (0.0, self.x0 + self.v0 * t - 0.5 * self.g * t * t, 0.0) }
    fn velocity(&self, t: f64) -> (f64, f64, f64) { (0.0, self.v0 - self.g * t, 0.0) }
    fn energy(&self) -> f64 { 0.5 * self.mass * self.v0 * self.v0 + self.mass * self.g * self.x0 }
}

/// Cyclotron orbit solution.
#[derive(Debug, Clone)]
pub struct CyclotronOrbitSolution { pub radius: f64, pub omega_c: f64 }
impl ExactSolution for CyclotronOrbitSolution {
    fn position(&self, t: f64) -> (f64, f64, f64) { (self.radius * (self.omega_c * t).cos(), self.radius * (self.omega_c * t).sin(), 0.0) }
    fn velocity(&self, t: f64) -> (f64, f64, f64) { (-self.radius * self.omega_c * (self.omega_c * t).sin(), self.radius * self.omega_c * (self.omega_c * t).cos(), 0.0) }
    fn energy(&self) -> f64 { 0.5 * self.radius * self.radius * self.omega_c * self.omega_c }
}
