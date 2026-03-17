//! Evaluation benchmarks with known-buggy simulations for testing conservation auditors.
//!
//! This crate provides a comprehensive suite of physics benchmarks, each with:
//! - Known analytical solutions for validation
//! - Intentionally buggy integrators that violate specific conservation laws
//! - Convergence studies to measure integrator order
//!
//! Use these benchmarks to test that your conservation-law auditing tools
//! correctly detect violations in numerical simulations.

pub mod benchmark;
pub mod kepler;
pub mod nbody;
pub mod spring;
pub mod pendulum;
pub mod rigid_body;
pub mod charged;
pub mod fluid;
pub mod wave;
pub mod scenarios;
pub mod exact_solutions;

pub use benchmark::{Benchmark, BenchmarkResult, BenchmarkSuite, ConvergenceResult};
pub use kepler::{CircularOrbit, EllipticalOrbit, KeplerSolver};
pub use nbody::{FigureEightOrbit, PythagoreanThreeBody, SolarSystemInner, PlummerModel};
pub use spring::{
    AnharmonicOscillator, CoupledOscillators, DampedOscillator, DrivenOscillator,
    SimpleHarmonicOscillator,
};
pub use pendulum::{DoublePendulum, SimplePendulum, SphericalPendulum};
pub use rigid_body::{AsymmetricTop, FreeRigidBody, SymmetricTop};
pub use charged::{
    CoulombScattering, CyclotronMotion, ExBDrift, MagneticBottle, UniformFieldParticle,
};
pub use fluid::{AdvectionEquation, BurgersEquation, ShallowWater1D, SodShockTube};
pub use wave::{StandingWave, TravelingWave, WaveEquation1D};
pub use scenarios::{BuggyScenario, ScenarioKind};
pub use exact_solutions::{
    CyclotronOrbitSolution, ExactSolution, FreeParticleSolution, HarmonicOscillatorSolution,
    KeplerOrbitSolution, UniformGravitySolution,
};

/// Gravitational constant in SI units (m^3 kg^-1 s^-2).
pub const G_SI: f64 = 6.674e-11;

/// Gravitational constant in natural units (G = 1).
pub const G_NATURAL: f64 = 1.0;

/// Coulomb constant in SI units (N m^2 C^-2).
pub const K_COULOMB: f64 = 8.9875e9;

/// Speed of light in SI (m/s).
pub const C_LIGHT: f64 = 2.998e8;

/// Elementary charge (C).
pub const E_CHARGE: f64 = 1.602e-19;

/// Electron mass (kg).
pub const M_ELECTRON: f64 = 9.109e-31;

/// Proton mass (kg).
pub const M_PROTON: f64 = 1.673e-27;

/// Standard gravitational acceleration (m/s^2).
pub const G_ACCEL: f64 = 9.80665;

/// Two-pi constant.
pub const TWO_PI: f64 = 2.0 * std::f64::consts::PI;

/// Helper: compute position error between two states.
pub fn position_error(exact: &sim_types::SimulationState, numerical: &sim_types::SimulationState) -> f64 {
    let mut max_err = 0.0_f64;
    let n = exact.particles.len().min(numerical.particles.len());
    for i in 0..n {
        let err = exact.particles[i].position.distance(numerical.particles[i].position);
        max_err = max_err.max(err);
    }
    max_err
}

/// Helper: compute velocity error between two states.
pub fn velocity_error(exact: &sim_types::SimulationState, numerical: &sim_types::SimulationState) -> f64 {
    let mut max_err = 0.0_f64;
    let n = exact.particles.len().min(numerical.particles.len());
    for i in 0..n {
        let err = exact.particles[i].velocity.distance(numerical.particles[i].velocity);
        max_err = max_err.max(err);
    }
    max_err
}
