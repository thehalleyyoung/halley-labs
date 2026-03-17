//! Conservation laws library for physics simulation auditing.
//!
//! This crate provides implementations of fundamental conservation laws
//! (energy, momentum, angular momentum, mass, charge, etc.) along with
//! utilities for computing conserved quantities, checking violations,
//! and managing collections of laws via a registry.

pub mod energy;
pub mod momentum;
pub mod angular_momentum;
pub mod mass;
pub mod charge;
pub mod center_of_mass;
pub mod symplectic;
pub mod vorticity;
pub mod custom;
pub mod registry;
pub mod potentials;

pub use energy::{
    KineticEnergy, GravitationalPotentialEnergy, SpringPotentialEnergy,
    ElectrostaticEnergy, TotalMechanicalEnergy,
};
pub use momentum::{
    TotalLinearMomentum, MomentumComponents, CenterOfMassVelocityLaw, ImpulseCalculation,
};
pub use angular_momentum::{
    TotalAngularMomentum, AngularMomentumAboutPoint, SpinAngularMomentum,
    OrbitalAngularMomentum, MomentOfInertia,
};
pub use mass::{TotalMass, MassDensity, ContinuityEquation};
pub use charge::{TotalCharge, ChargeDensity, CurrentDensity, ChargeCurrentContinuity};
pub use center_of_mass::{
    CenterOfMass, CenterOfMassVelocity, CenterOfMassAcceleration, ReducedMass,
};
pub use symplectic::{
    SymplecticFormComputation, PhaseSpaceVolume, SymplecticMatrixCheck, PoincareInvariant,
};
pub use vorticity::{Vorticity, Circulation, KelvinCirculationTheorem, Enstrophy};
pub use custom::{CustomLaw, CompositeConservation, ConditionalConservation};
pub use registry::LawRegistry;
pub use potentials::{
    GravitationalPotential, CoulombPotential, HarmonicPotential,
    LennardJonesPotential, MorsePotential, YukawaPotential,
};

use sim_types::{
    ConservationKind, ConservedQuantity, SimulationState, Tolerance, Violation, ViolationSeverity,
};

// ─── Physical Constants ─────────────────────────────────────────────────────

/// Gravitational constant in SI units (m³ kg⁻¹ s⁻²)
pub const G_SI: f64 = 6.674_30e-11;

/// Coulomb constant in SI units (N m² C⁻²)
pub const K_COULOMB_SI: f64 = 8.987_551_792_3e9;

/// Speed of light in vacuum (m/s)
pub const C_LIGHT: f64 = 2.997_924_58e8;

/// Boltzmann constant (J/K)
pub const K_BOLTZMANN: f64 = 1.380_649e-23;

/// Planck constant (J·s)
pub const H_PLANCK: f64 = 6.626_070_15e-34;

/// Reduced Planck constant (J·s)
pub const HBAR: f64 = 1.054_571_817e-34;

/// Elementary charge (C)
pub const E_CHARGE: f64 = 1.602_176_634e-19;

/// Vacuum permittivity (F/m)
pub const EPSILON_0: f64 = 8.854_187_817e-12;

/// Pi
pub const PI: f64 = std::f64::consts::PI;

// ─── Core Trait ─────────────────────────────────────────────────────────────

/// Trait for objects that can compute a conserved quantity from a simulation state,
/// and check whether that quantity is conserved between two states.
pub trait ConservationChecker: Send + Sync {
    /// Human-readable name of this conservation law.
    fn name(&self) -> &str;

    /// The kind of conservation law (energy, momentum, etc.).
    fn kind(&self) -> ConservationKind;

    /// Compute the conserved quantity for a given state.
    fn compute(&self, state: &SimulationState) -> ConservedQuantity;

    /// Compute a scalar value representing the conserved quantity.
    fn compute_scalar(&self, state: &SimulationState) -> f64 {
        self.compute(state).value
    }

    /// Check conservation between an initial and current state.
    /// Returns Some(Violation) if conservation is violated beyond tolerance.
    fn check(
        &self,
        initial: &SimulationState,
        current: &SimulationState,
        tolerance: &Tolerance,
    ) -> Option<Violation> {
        let expected = self.compute_scalar(initial);
        let actual = self.compute_scalar(current);
        if tolerance.check(expected, actual) {
            None
        } else {
            let severity = classify_violation_severity(expected, actual);
            Some(Violation::new(
                self.kind(),
                severity,
                current.time,
                expected,
                actual,
            ))
        }
    }

    /// Compute a time series of the conserved quantity across multiple states.
    fn time_series(&self, states: &[SimulationState]) -> sim_types::TimeSeries {
        let times: Vec<f64> = states.iter().map(|s| s.time).collect();
        let values: Vec<f64> = states.iter().map(|s| self.compute_scalar(s)).collect();
        sim_types::TimeSeries::new(times, values)
    }

    /// Check conservation across a sequence of states, returning all violations.
    fn check_sequence(
        &self,
        states: &[SimulationState],
        tolerance: &Tolerance,
    ) -> Vec<Violation> {
        if states.is_empty() {
            return Vec::new();
        }
        let initial = &states[0];
        states[1..]
            .iter()
            .filter_map(|current| self.check(initial, current, tolerance))
            .collect()
    }

    /// Compute the maximum drift of the conserved quantity across a sequence.
    fn max_drift(&self, states: &[SimulationState]) -> f64 {
        if states.is_empty() {
            return 0.0;
        }
        let initial_val = self.compute_scalar(&states[0]);
        states[1..]
            .iter()
            .map(|s| (self.compute_scalar(s) - initial_val).abs())
            .fold(0.0_f64, f64::max)
    }

    /// Compute the drift rate (change per unit time) between two states.
    fn drift_rate(
        &self,
        initial: &SimulationState,
        current: &SimulationState,
    ) -> f64 {
        let dt = current.time - initial.time;
        if dt.abs() < 1e-30 {
            return 0.0;
        }
        let dq = self.compute_scalar(current) - self.compute_scalar(initial);
        dq / dt
    }
}

/// Classify a conservation violation by its severity based on relative error.
pub fn classify_violation_severity(expected: f64, actual: f64) -> ViolationSeverity {
    let abs_err = (actual - expected).abs();
    let rel_err = if expected.abs() > 1e-30 {
        abs_err / expected.abs()
    } else {
        abs_err
    };

    if rel_err < 1e-10 {
        ViolationSeverity::Info
    } else if rel_err < 1e-6 {
        ViolationSeverity::Warning
    } else if rel_err < 1e-2 {
        ViolationSeverity::Error
    } else {
        ViolationSeverity::Critical
    }
}

// ─── Helper Utilities ───────────────────────────────────────────────────────

/// Compute the distance between two particles, clamped to a minimum to avoid singularities.
#[inline]
pub fn safe_distance(p1: &sim_types::Particle, p2: &sim_types::Particle, min_r: f64) -> f64 {
    let r = p1.position.distance(p2.position);
    r.max(min_r)
}

/// Compute pairwise sum over all unique particle pairs.
pub fn pairwise_sum<F>(particles: &[sim_types::Particle], f: F) -> f64
where
    F: Fn(&sim_types::Particle, &sim_types::Particle) -> f64,
{
    let n = particles.len();
    let mut total = 0.0;
    for i in 0..n {
        for j in (i + 1)..n {
            total += f(&particles[i], &particles[j]);
        }
    }
    total
}

/// Compute pairwise vector sum over all unique particle pairs.
pub fn pairwise_vec_sum<F>(
    particles: &[sim_types::Particle],
    f: F,
) -> sim_types::Vec3
where
    F: Fn(&sim_types::Particle, &sim_types::Particle) -> sim_types::Vec3,
{
    let n = particles.len();
    let mut total = sim_types::Vec3::ZERO;
    for i in 0..n {
        for j in (i + 1)..n {
            total = total + f(&particles[i], &particles[j]);
        }
    }
    total
}

/// Numerically estimate the derivative of a scalar quantity over a time series.
pub fn numerical_derivative(ts: &sim_types::TimeSeries) -> sim_types::TimeSeries {
    if ts.len() < 2 {
        return sim_types::TimeSeries::new(ts.times.clone(), vec![0.0; ts.len()]);
    }
    let n = ts.len();
    let mut deriv_times = Vec::with_capacity(n);
    let mut deriv_values = Vec::with_capacity(n);

    // Forward difference for first point
    let dt0 = ts.times[1] - ts.times[0];
    if dt0.abs() > 1e-30 {
        deriv_times.push(ts.times[0]);
        deriv_values.push((ts.values[1] - ts.values[0]) / dt0);
    }

    // Central differences for interior points
    for i in 1..(n - 1) {
        let dt = ts.times[i + 1] - ts.times[i - 1];
        if dt.abs() > 1e-30 {
            deriv_times.push(ts.times[i]);
            deriv_values.push((ts.values[i + 1] - ts.values[i - 1]) / dt);
        }
    }

    // Backward difference for last point
    let dt_last = ts.times[n - 1] - ts.times[n - 2];
    if dt_last.abs() > 1e-30 {
        deriv_times.push(ts.times[n - 1]);
        deriv_values.push((ts.values[n - 1] - ts.values[n - 2]) / dt_last);
    }

    sim_types::TimeSeries::new(deriv_times, deriv_values)
}

/// Statistics for a conservation quantity time series.
#[derive(Debug, Clone)]
pub struct ConservationStats {
    pub initial_value: f64,
    pub final_value: f64,
    pub mean_value: f64,
    pub std_deviation: f64,
    pub max_absolute_drift: f64,
    pub max_relative_drift: f64,
    pub mean_drift_rate: f64,
}

impl ConservationStats {
    /// Compute statistics from a time series of conserved quantity values.
    pub fn from_time_series(ts: &sim_types::TimeSeries) -> Self {
        if ts.is_empty() {
            return Self {
                initial_value: 0.0,
                final_value: 0.0,
                mean_value: 0.0,
                std_deviation: 0.0,
                max_absolute_drift: 0.0,
                max_relative_drift: 0.0,
                mean_drift_rate: 0.0,
            };
        }

        let initial = ts.values[0];
        let final_val = *ts.values.last().unwrap();
        let mean = ts.mean();
        let std_dev = ts.std_dev();

        let max_abs_drift = ts
            .values
            .iter()
            .map(|v| (v - initial).abs())
            .fold(0.0_f64, f64::max);

        let max_rel_drift = if initial.abs() > 1e-30 {
            max_abs_drift / initial.abs()
        } else {
            max_abs_drift
        };

        let total_time = ts.times.last().unwrap_or(&0.0) - ts.times.first().unwrap_or(&0.0);
        let mean_drift_rate = if total_time.abs() > 1e-30 {
            (final_val - initial) / total_time
        } else {
            0.0
        };

        Self {
            initial_value: initial,
            final_value: final_val,
            mean_value: mean,
            std_deviation: std_dev,
            max_absolute_drift: max_abs_drift,
            max_relative_drift: max_rel_drift,
            mean_drift_rate,
        }
    }

    /// Check if the conservation law is well-preserved (drift within tolerance).
    pub fn is_conserved(&self, tolerance: &Tolerance) -> bool {
        tolerance.check(self.initial_value, self.initial_value + self.max_absolute_drift)
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use sim_types::{Particle, Vec3, TimeSeries};

    const EPS: f64 = 1e-12;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPS
    }

    #[test]
    fn test_classify_violation_severity() {
        assert_eq!(
            classify_violation_severity(100.0, 100.0 + 1e-12),
            ViolationSeverity::Info
        );
        assert_eq!(
            classify_violation_severity(100.0, 100.0 + 1e-5),
            ViolationSeverity::Warning
        );
        assert_eq!(
            classify_violation_severity(100.0, 100.0 + 0.5),
            ViolationSeverity::Error
        );
        assert_eq!(
            classify_violation_severity(100.0, 200.0),
            ViolationSeverity::Critical
        );
    }

    #[test]
    fn test_safe_distance() {
        let p1 = Particle::new(1.0, Vec3::ZERO, Vec3::ZERO);
        let p2 = Particle::new(1.0, Vec3::new(3.0, 4.0, 0.0), Vec3::ZERO);
        assert!(approx_eq(safe_distance(&p1, &p2, 1e-10), 5.0));

        // Test clamping
        let p3 = Particle::new(1.0, Vec3::ZERO, Vec3::ZERO);
        assert!(approx_eq(safe_distance(&p1, &p3, 0.01), 0.01));
    }

    #[test]
    fn test_pairwise_sum() {
        let particles = vec![
            Particle::new(1.0, Vec3::new(0.0, 0.0, 0.0), Vec3::ZERO),
            Particle::new(2.0, Vec3::new(1.0, 0.0, 0.0), Vec3::ZERO),
            Particle::new(3.0, Vec3::new(0.0, 1.0, 0.0), Vec3::ZERO),
        ];
        // Sum of product of masses for each pair
        let result = pairwise_sum(&particles, |a, b| a.mass * b.mass);
        // Pairs: (1*2)+(1*3)+(2*3) = 2+3+6 = 11
        assert!(approx_eq(result, 11.0));
    }

    #[test]
    fn test_numerical_derivative() {
        // Derivative of x^2 should be 2x
        let ts = TimeSeries::from_fn(0.0, 5.0, 100, |t| t * t);
        let deriv = numerical_derivative(&ts);
        // Check at t=2.5 (approximately)
        let mid_idx = deriv.len() / 2;
        let t_mid = deriv.times[mid_idx];
        let expected_deriv = 2.0 * t_mid;
        assert!((deriv.values[mid_idx] - expected_deriv).abs() < 0.1);
    }

    #[test]
    fn test_conservation_stats() {
        let ts = TimeSeries::new(
            vec![0.0, 1.0, 2.0, 3.0, 4.0],
            vec![100.0, 100.01, 99.99, 100.02, 99.98],
        );
        let stats = ConservationStats::from_time_series(&ts);
        assert!(approx_eq(stats.initial_value, 100.0));
        assert!(approx_eq(stats.final_value, 99.98));
        assert!(stats.max_absolute_drift < 0.05);
        assert!(stats.is_conserved(&Tolerance::absolute(0.1)));
    }

    #[test]
    fn test_conservation_stats_empty() {
        let ts = TimeSeries::new(vec![], vec![]);
        let stats = ConservationStats::from_time_series(&ts);
        assert!(approx_eq(stats.initial_value, 0.0));
        assert!(approx_eq(stats.max_absolute_drift, 0.0));
    }

    #[test]
    fn test_pairwise_vec_sum() {
        let particles = vec![
            Particle::new(1.0, Vec3::new(1.0, 0.0, 0.0), Vec3::ZERO),
            Particle::new(1.0, Vec3::new(0.0, 1.0, 0.0), Vec3::ZERO),
        ];
        let result = pairwise_vec_sum(&particles, |a, b| a.position + b.position);
        assert!(approx_eq(result.x, 1.0));
        assert!(approx_eq(result.y, 1.0));
    }
}
