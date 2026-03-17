//! Potential energy functions for various physical interactions.

/// Gravitational potential V(r) = -G·m₁·m₂ / r.
#[derive(Debug, Clone)]
pub struct GravitationalPotential {
    /// Gravitational constant.
    pub g: f64,
}

impl GravitationalPotential {
    /// Compute the gravitational potential energy between two masses at distance r.
    pub fn energy(&self, m1: f64, m2: f64, r: f64) -> f64 {
        if r.abs() < 1e-30 { return 0.0; }
        -self.g * m1 * m2 / r
    }
    /// Compute the gravitational force magnitude.
    pub fn force(&self, m1: f64, m2: f64, r: f64) -> f64 {
        if r.abs() < 1e-30 { return 0.0; }
        self.g * m1 * m2 / (r * r)
    }
}

/// Coulomb potential V(r) = kₑ·q₁·q₂ / r.
#[derive(Debug, Clone)]
pub struct CoulombPotential {
    /// Coulomb constant.
    pub k: f64,
}

impl CoulombPotential {
    /// Compute the Coulomb energy.
    pub fn energy(&self, q1: f64, q2: f64, r: f64) -> f64 {
        if r.abs() < 1e-30 { return 0.0; }
        self.k * q1 * q2 / r
    }
}

/// Harmonic potential V(r) = ½k(r - r₀)².
#[derive(Debug, Clone)]
pub struct HarmonicPotential {
    /// Spring constant.
    pub k: f64,
    /// Equilibrium distance.
    pub r0: f64,
}

impl HarmonicPotential {
    /// Compute the harmonic potential energy.
    pub fn energy(&self, r: f64) -> f64 {
        let dr = r - self.r0;
        0.5 * self.k * dr * dr
    }
    /// Compute the restoring force magnitude.
    pub fn force(&self, r: f64) -> f64 {
        -self.k * (r - self.r0)
    }
}

/// Lennard-Jones potential V(r) = 4ε[(σ/r)¹² - (σ/r)⁶].
#[derive(Debug, Clone)]
pub struct LennardJonesPotential {
    /// Well depth ε.
    pub epsilon: f64,
    /// Size parameter σ.
    pub sigma: f64,
}

impl LennardJonesPotential {
    /// Compute the LJ potential energy.
    pub fn energy(&self, r: f64) -> f64 {
        if r.abs() < 1e-30 { return 0.0; }
        let sr = self.sigma / r;
        let sr6 = sr.powi(6);
        4.0 * self.epsilon * (sr6 * sr6 - sr6)
    }
}

/// Morse potential V(r) = Dₑ[1 - e^{-a(r-rₑ)}]².
#[derive(Debug, Clone)]
pub struct MorsePotential {
    /// Well depth.
    pub d_e: f64,
    /// Width parameter.
    pub a: f64,
    /// Equilibrium distance.
    pub r_e: f64,
}

impl MorsePotential {
    /// Compute the Morse potential energy.
    pub fn energy(&self, r: f64) -> f64 {
        let x = 1.0 - (-self.a * (r - self.r_e)).exp();
        self.d_e * x * x
    }
}

/// Yukawa potential V(r) = -g²·e^{-μr} / r.
#[derive(Debug, Clone)]
pub struct YukawaPotential {
    /// Coupling constant g².
    pub g_squared: f64,
    /// Mass parameter μ (inverse range).
    pub mu: f64,
}

impl YukawaPotential {
    /// Compute the Yukawa potential energy.
    pub fn energy(&self, r: f64) -> f64 {
        if r.abs() < 1e-30 { return 0.0; }
        -self.g_squared * (-self.mu * r).exp() / r
    }
}
