//! Kepler two-body problem with exact analytical solutions.
//!
//! Implements circular and elliptical orbits with:
//! - Kepler equation solver via Newton iteration
//! - Exact position and velocity via eccentric/true anomaly
//! - Energy, angular momentum, and Laplace-Runge-Lenz vector conservation
//! - Known bug injection: asymmetric force evaluation

use crate::benchmark::Benchmark;
use sim_types::{ConservationKind, Particle, SimulationState, Vec3};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Kepler equation solver
// ---------------------------------------------------------------------------

/// Solver for the Kepler equation M = E - e*sin(E).
#[derive(Debug, Clone, Copy)]
pub struct KeplerSolver {
    pub max_iterations: usize,
    pub tolerance: f64,
}

impl KeplerSolver {
    pub fn new() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-15,
        }
    }

    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    /// Solve M = E - e*sin(E) for E given M and e, using Newton-Raphson.
    pub fn solve(&self, mean_anomaly: f64, eccentricity: f64) -> f64 {
        let m = mean_anomaly % (2.0 * std::f64::consts::PI);
        let e = eccentricity;

        // Initial guess: E = M for small e, or better initial guess
        let mut big_e = if e < 0.8 {
            m
        } else {
            std::f64::consts::PI
        };

        for _ in 0..self.max_iterations {
            let f = big_e - e * big_e.sin() - m;
            let f_prime = 1.0 - e * big_e.cos();
            if f_prime.abs() < 1e-30 {
                break;
            }
            let delta = f / f_prime;
            big_e -= delta;
            if delta.abs() < self.tolerance {
                break;
            }
        }

        big_e
    }

    /// Solve using Halley's method for faster convergence.
    pub fn solve_halley(&self, mean_anomaly: f64, eccentricity: f64) -> f64 {
        let m = mean_anomaly % (2.0 * std::f64::consts::PI);
        let e = eccentricity;

        let mut big_e = if e < 0.8 { m } else { std::f64::consts::PI };

        for _ in 0..self.max_iterations {
            let sin_e = big_e.sin();
            let cos_e = big_e.cos();
            let f = big_e - e * sin_e - m;
            let f_prime = 1.0 - e * cos_e;
            let f_double_prime = e * sin_e;

            if f_prime.abs() < 1e-30 {
                break;
            }

            // Halley's method: delta = f * f' / (f'^2 - 0.5 * f * f'')
            let denom = f_prime * f_prime - 0.5 * f * f_double_prime;
            if denom.abs() < 1e-30 {
                big_e -= f / f_prime;
            } else {
                big_e -= f * f_prime / denom;
            }

            if f.abs() < self.tolerance {
                break;
            }
        }

        big_e
    }

    /// Compute the true anomaly from the eccentric anomaly.
    pub fn true_anomaly_from_eccentric(eccentric_anomaly: f64, eccentricity: f64) -> f64 {
        let e = eccentricity;
        let big_e = eccentric_anomaly;
        let half = big_e / 2.0;
        2.0 * ((1.0 + e).sqrt() * half.sin()).atan2((1.0 - e).sqrt() * half.cos())
    }

    /// Compute eccentric anomaly from true anomaly.
    pub fn eccentric_from_true(true_anomaly: f64, eccentricity: f64) -> f64 {
        let e = eccentricity;
        let nu = true_anomaly;
        2.0 * ((1.0 - e).sqrt() * (nu / 2.0).sin()).atan2((1.0 + e).sqrt() * (nu / 2.0).cos())
    }
}

impl Default for KeplerSolver {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Orbital elements
// ---------------------------------------------------------------------------

/// Orbital elements for a Kepler orbit.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct OrbitalElements {
    /// Semi-major axis.
    pub semi_major_axis: f64,
    /// Eccentricity (0 = circular, 0 < e < 1 = elliptical).
    pub eccentricity: f64,
    /// Inclination (rad).
    pub inclination: f64,
    /// Longitude of ascending node (rad).
    pub longitude_ascending: f64,
    /// Argument of periapsis (rad).
    pub argument_periapsis: f64,
    /// Mean anomaly at epoch (rad).
    pub mean_anomaly_epoch: f64,
    /// Gravitational parameter mu = G*(M+m).
    pub mu: f64,
}

impl OrbitalElements {
    /// Orbital period T = 2*pi*sqrt(a^3/mu).
    pub fn period(&self) -> f64 {
        2.0 * std::f64::consts::PI * (self.semi_major_axis.powi(3) / self.mu).sqrt()
    }

    /// Specific orbital energy E = -mu/(2*a).
    pub fn specific_energy(&self) -> f64 {
        -self.mu / (2.0 * self.semi_major_axis)
    }

    /// Specific angular momentum h = sqrt(mu * a * (1 - e^2)).
    pub fn specific_angular_momentum(&self) -> f64 {
        (self.mu * self.semi_major_axis * (1.0 - self.eccentricity.powi(2))).sqrt()
    }

    /// Semi-latus rectum p = a*(1-e^2).
    pub fn semi_latus_rectum(&self) -> f64 {
        self.semi_major_axis * (1.0 - self.eccentricity.powi(2))
    }

    /// Periapsis distance r_p = a*(1-e).
    pub fn periapsis(&self) -> f64 {
        self.semi_major_axis * (1.0 - self.eccentricity)
    }

    /// Apoapsis distance r_a = a*(1+e).
    pub fn apoapsis(&self) -> f64 {
        self.semi_major_axis * (1.0 + self.eccentricity)
    }

    /// Mean motion n = 2*pi/T = sqrt(mu/a^3).
    pub fn mean_motion(&self) -> f64 {
        (self.mu / self.semi_major_axis.powi(3)).sqrt()
    }

    /// Position and velocity at time t (from epoch).
    pub fn state_at_time(&self, t: f64) -> (Vec3, Vec3) {
        let solver = KeplerSolver::new();
        let n = self.mean_motion();
        let mean_anomaly = self.mean_anomaly_epoch + n * t;
        let eccentric_anomaly = solver.solve(mean_anomaly, self.eccentricity);
        let true_anomaly =
            KeplerSolver::true_anomaly_from_eccentric(eccentric_anomaly, self.eccentricity);

        let e = self.eccentricity;
        let a = self.semi_major_axis;
        let p = self.semi_latus_rectum();
        let h = self.specific_angular_momentum();
        let mu = self.mu;

        // Radius at true anomaly
        let r = p / (1.0 + e * true_anomaly.cos());

        // Position in orbital plane (perifocal frame)
        let x_pf = r * true_anomaly.cos();
        let y_pf = r * true_anomaly.sin();

        // Velocity in orbital plane
        let vx_pf = -(mu / h) * true_anomaly.sin();
        let vy_pf = (mu / h) * (e + true_anomaly.cos());

        // Rotate to inertial frame using Euler angles
        let omega = self.argument_periapsis;
        let big_omega = self.longitude_ascending;
        let inc = self.inclination;

        let cos_o = omega.cos();
        let sin_o = omega.sin();
        let cos_O = big_omega.cos();
        let sin_O = big_omega.sin();
        let cos_i = inc.cos();
        let sin_i = inc.sin();

        // Rotation matrix columns for perifocal -> inertial
        let px = cos_O * cos_o - sin_O * sin_o * cos_i;
        let py = sin_O * cos_o + cos_O * sin_o * cos_i;
        let pz = sin_o * sin_i;

        let qx = -cos_O * sin_o - sin_O * cos_o * cos_i;
        let qy = -sin_O * sin_o + cos_O * cos_o * cos_i;
        let qz = cos_o * sin_i;

        let pos = Vec3::new(
            x_pf * px + y_pf * qx,
            x_pf * py + y_pf * qy,
            x_pf * pz + y_pf * qz,
        );
        let vel = Vec3::new(
            vx_pf * px + vy_pf * qx,
            vx_pf * py + vy_pf * qy,
            vx_pf * pz + vy_pf * qz,
        );

        (pos, vel)
    }
}

// ---------------------------------------------------------------------------
// Circular orbit benchmark
// ---------------------------------------------------------------------------

/// Circular orbit benchmark: a particle in a circular orbit around a central mass.
///
/// r = a, v = sqrt(mu/a), period T = 2*pi*sqrt(a^3/mu).
/// All orbital elements except semi-major axis default to simple values.
#[derive(Debug, Clone)]
pub struct CircularOrbit {
    pub central_mass: f64,
    pub orbiting_mass: f64,
    pub radius: f64,
    pub g_const: f64,
}

impl CircularOrbit {
    pub fn new(central_mass: f64, orbiting_mass: f64, radius: f64, g_const: f64) -> Self {
        Self {
            central_mass,
            orbiting_mass,
            radius,
            g_const,
        }
    }

    /// Standard test case: unit masses, G=1, radius=1.
    pub fn unit() -> Self {
        Self::new(1.0, 1.0, 1.0, 1.0)
    }

    /// Earth-like orbit around solar mass (SI units).
    pub fn earth_like() -> Self {
        Self::new(1.989e30, 5.972e24, 1.496e11, crate::G_SI)
    }

    fn mu(&self) -> f64 {
        self.g_const * (self.central_mass + self.orbiting_mass)
    }

    /// Circular orbital velocity.
    pub fn orbital_velocity(&self) -> f64 {
        (self.mu() / self.radius).sqrt()
    }

    /// Orbital period.
    pub fn period(&self) -> f64 {
        2.0 * std::f64::consts::PI * (self.radius.powi(3) / self.mu()).sqrt()
    }

    /// Gravitational force function for this orbit.
    pub fn gravity_force(&self) -> impl Fn(&SimulationState) -> Vec<Vec3> + '_ {
        move |state: &SimulationState| {
            let n = state.particles.len();
            let mut forces = vec![Vec3::ZERO; n];
            if n >= 2 {
                let r_vec = state.particles[1].position - state.particles[0].position;
                let r = r_vec.magnitude();
                if r > 1e-15 {
                    let f_mag = self.g_const * state.particles[0].mass * state.particles[1].mass
                        / (r * r);
                    let f_dir = r_vec.normalized();
                    forces[0] = f_dir * f_mag;
                    forces[1] = f_dir * (-f_mag);
                }
            }
            forces
        }
    }

    /// BUGGY force: asymmetric evaluation that breaks angular momentum.
    /// The force on particle 1 uses a slightly different position than
    /// the force on particle 0, violating Newton's third law.
    pub fn buggy_asymmetric_force(&self) -> impl Fn(&SimulationState) -> Vec<Vec3> + '_ {
        move |state: &SimulationState| {
            let n = state.particles.len();
            let mut forces = vec![Vec3::ZERO; n];
            if n >= 2 {
                let r_vec = state.particles[1].position - state.particles[0].position;
                let r = r_vec.magnitude();
                if r > 1e-15 {
                    let f_mag = self.g_const * state.particles[0].mass * state.particles[1].mass
                        / (r * r);
                    let f_dir = r_vec.normalized();
                    forces[0] = f_dir * f_mag;

                    // BUG: use slightly shifted position for the reaction force
                    let shift = Vec3::new(1e-6, 0.0, 0.0);
                    let r_vec_shifted =
                        state.particles[1].position - (state.particles[0].position + shift);
                    let r_s = r_vec_shifted.magnitude();
                    let f_mag_s = self.g_const * state.particles[0].mass * state.particles[1].mass
                        / (r_s * r_s);
                    let f_dir_s = r_vec_shifted.normalized();
                    forces[1] = f_dir_s * (-f_mag_s);
                }
            }
            forces
        }
    }

    /// Compute the Laplace-Runge-Lenz vector for the orbiting particle.
    /// A = (v × L)/mu - r_hat, where L = m * r × v.
    pub fn laplace_runge_lenz(&self, state: &SimulationState) -> Vec3 {
        if state.particles.len() < 2 {
            return Vec3::ZERO;
        }
        let p = &state.particles[1];
        let r_vec = p.position - state.particles[0].position;
        let r = r_vec.magnitude();
        if r < 1e-15 {
            return Vec3::ZERO;
        }
        let v = p.velocity - state.particles[0].velocity;
        let mu = self.mu();
        let l = r_vec.cross(v);
        // A = (v × L) / mu - r_hat
        v.cross(l) / mu - r_vec.normalized()
    }
}

impl Benchmark for CircularOrbit {
    fn name(&self) -> &str {
        "Circular Kepler Orbit"
    }

    fn setup(&self) -> SimulationState {
        let v = self.orbital_velocity();
        let central = Particle::new(self.central_mass, Vec3::ZERO, Vec3::ZERO).with_id("central");
        let orbiter = Particle::new(
            self.orbiting_mass,
            Vec3::new(self.radius, 0.0, 0.0),
            Vec3::new(0.0, v, 0.0),
        )
        .with_id("orbiter");
        SimulationState::new(vec![central, orbiter], 0.0)
    }

    fn exact_solution(&self, t: f64) -> SimulationState {
        let v = self.orbital_velocity();
        let omega = v / self.radius;
        let angle = omega * t;

        let central = Particle::new(self.central_mass, Vec3::ZERO, Vec3::ZERO).with_id("central");
        let orbiter = Particle::new(
            self.orbiting_mass,
            Vec3::new(self.radius * angle.cos(), self.radius * angle.sin(), 0.0),
            Vec3::new(-v * angle.sin(), v * angle.cos(), 0.0),
        )
        .with_id("orbiter");

        SimulationState::new(vec![central, orbiter], t)
    }

    fn conservation_laws(&self) -> Vec<ConservationKind> {
        vec![
            ConservationKind::Energy,
            ConservationKind::Momentum,
            ConservationKind::AngularMomentum,
        ]
    }

    fn conserved_quantity(&self, kind: ConservationKind, state: &SimulationState) -> f64 {
        match kind {
            ConservationKind::Energy => {
                let ke: f64 = state.particles.iter().map(|p| p.kinetic_energy()).sum();
                let pe = if state.particles.len() >= 2 {
                    state.particles[0].gravitational_potential(&state.particles[1], self.g_const)
                } else {
                    0.0
                };
                ke + pe
            }
            ConservationKind::Momentum => {
                let p: Vec3 = state
                    .particles
                    .iter()
                    .fold(Vec3::ZERO, |acc, p| acc + p.momentum());
                p.magnitude()
            }
            ConservationKind::AngularMomentum => {
                let l: Vec3 = state
                    .particles
                    .iter()
                    .fold(Vec3::ZERO, |acc, p| acc + p.angular_momentum());
                l.magnitude()
            }
            _ => 0.0,
        }
    }

    fn characteristic_time(&self) -> f64 {
        self.period()
    }
}

// ---------------------------------------------------------------------------
// Elliptical orbit benchmark
// ---------------------------------------------------------------------------

/// Elliptical orbit benchmark with known analytical solution via Kepler equation.
#[derive(Debug, Clone)]
pub struct EllipticalOrbit {
    pub elements: OrbitalElements,
    pub central_mass: f64,
    pub orbiting_mass: f64,
    pub g_const: f64,
}

impl EllipticalOrbit {
    pub fn new(
        central_mass: f64,
        orbiting_mass: f64,
        semi_major_axis: f64,
        eccentricity: f64,
        g_const: f64,
    ) -> Self {
        let mu = g_const * (central_mass + orbiting_mass);
        Self {
            elements: OrbitalElements {
                semi_major_axis,
                eccentricity,
                inclination: 0.0,
                longitude_ascending: 0.0,
                argument_periapsis: 0.0,
                mean_anomaly_epoch: 0.0,
                mu,
            },
            central_mass,
            orbiting_mass,
            g_const,
        }
    }

    /// Standard test: e=0.5, a=1, G=1, unit masses.
    pub fn standard() -> Self {
        Self::new(1.0, 0.001, 1.0, 0.5, 1.0)
    }

    /// Highly eccentric orbit for testing.
    pub fn high_eccentricity() -> Self {
        Self::new(1.0, 0.001, 1.0, 0.9, 1.0)
    }

    /// Mercury-like orbit with eccentricity ~0.2.
    pub fn mercury_like() -> Self {
        Self::new(1.0, 0.001, 0.387, 0.206, 1.0)
    }

    /// Halley's comet-like orbit with very high eccentricity.
    pub fn halley_like() -> Self {
        Self::new(1.0, 1e-10, 17.8, 0.967, 1.0)
    }

    /// Gravity force function for two-body.
    pub fn gravity_force(&self) -> impl Fn(&SimulationState) -> Vec<Vec3> + '_ {
        move |state: &SimulationState| {
            let n = state.particles.len();
            let mut forces = vec![Vec3::ZERO; n];
            if n >= 2 {
                let r_vec = state.particles[1].position - state.particles[0].position;
                let r = r_vec.magnitude();
                if r > 1e-15 {
                    let f_mag = self.g_const * state.particles[0].mass * state.particles[1].mass
                        / (r * r);
                    let f_dir = r_vec.normalized();
                    forces[0] = f_dir * f_mag;
                    forces[1] = f_dir * (-f_mag);
                }
            }
            forces
        }
    }

    /// Compute the Laplace-Runge-Lenz vector for the orbiting particle.
    pub fn laplace_runge_lenz(&self, state: &SimulationState) -> Vec3 {
        if state.particles.len() < 2 {
            return Vec3::ZERO;
        }
        let r_vec = state.particles[1].position - state.particles[0].position;
        let r = r_vec.magnitude();
        if r < 1e-15 {
            return Vec3::ZERO;
        }
        let v = state.particles[1].velocity - state.particles[0].velocity;
        let mu = self.elements.mu;
        let l = r_vec.cross(v);
        v.cross(l) / mu - r_vec.normalized()
    }

    /// Vis-viva equation: v^2 = mu*(2/r - 1/a).
    pub fn vis_viva_velocity(&self, r: f64) -> f64 {
        let mu = self.elements.mu;
        let a = self.elements.semi_major_axis;
        (mu * (2.0 / r - 1.0 / a)).sqrt()
    }
}

impl Benchmark for EllipticalOrbit {
    fn name(&self) -> &str {
        "Elliptical Kepler Orbit"
    }

    fn setup(&self) -> SimulationState {
        let (pos, vel) = self.elements.state_at_time(0.0);
        let central = Particle::new(self.central_mass, Vec3::ZERO, Vec3::ZERO).with_id("central");
        let orbiter =
            Particle::new(self.orbiting_mass, pos, vel).with_id("orbiter");
        SimulationState::new(vec![central, orbiter], 0.0)
    }

    fn exact_solution(&self, t: f64) -> SimulationState {
        let (pos, vel) = self.elements.state_at_time(t);
        let central = Particle::new(self.central_mass, Vec3::ZERO, Vec3::ZERO).with_id("central");
        let orbiter =
            Particle::new(self.orbiting_mass, pos, vel).with_id("orbiter");
        SimulationState::new(vec![central, orbiter], t)
    }

    fn conservation_laws(&self) -> Vec<ConservationKind> {
        vec![
            ConservationKind::Energy,
            ConservationKind::Momentum,
            ConservationKind::AngularMomentum,
        ]
    }

    fn conserved_quantity(&self, kind: ConservationKind, state: &SimulationState) -> f64 {
        match kind {
            ConservationKind::Energy => {
                let ke: f64 = state.particles.iter().map(|p| p.kinetic_energy()).sum();
                let pe = if state.particles.len() >= 2 {
                    state.particles[0].gravitational_potential(&state.particles[1], self.g_const)
                } else {
                    0.0
                };
                ke + pe
            }
            ConservationKind::Momentum => {
                let p: Vec3 = state
                    .particles
                    .iter()
                    .fold(Vec3::ZERO, |acc, part| acc + part.momentum());
                p.magnitude()
            }
            ConservationKind::AngularMomentum => {
                let l: Vec3 = state
                    .particles
                    .iter()
                    .fold(Vec3::ZERO, |acc, part| acc + part.angular_momentum());
                l.magnitude()
            }
            _ => 0.0,
        }
    }

    fn characteristic_time(&self) -> f64 {
        self.elements.period()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_kepler_solver_circular() {
        let solver = KeplerSolver::new();
        // For circular orbit (e=0), E = M
        for m in [0.0, 0.5, 1.0, 2.0, 3.0, 5.0] {
            let e_anom = solver.solve(m, 0.0);
            assert!(approx_eq(e_anom, m % (2.0 * std::f64::consts::PI), 1e-12));
        }
    }

    #[test]
    fn test_kepler_solver_moderate_eccentricity() {
        let solver = KeplerSolver::new();
        let e = 0.5;
        // Verify: E - e*sin(E) = M
        for m_deg in [0, 30, 60, 90, 120, 180, 270] {
            let m = (m_deg as f64).to_radians();
            let big_e = solver.solve(m, e);
            let residual = big_e - e * big_e.sin() - m;
            assert!(residual.abs() < 1e-12, "Residual {residual} for M={m_deg}°");
        }
    }

    #[test]
    fn test_kepler_solver_high_eccentricity() {
        let solver = KeplerSolver::new();
        let e = 0.95;
        for m_deg in [1, 10, 45, 90, 135, 179] {
            let m = (m_deg as f64).to_radians();
            let big_e = solver.solve(m, e);
            let residual = big_e - e * big_e.sin() - m;
            assert!(
                residual.abs() < 1e-10,
                "Residual {residual} for M={m_deg}° e={e}"
            );
        }
    }

    #[test]
    fn test_kepler_solver_halley_method() {
        let solver = KeplerSolver::new();
        let e = 0.9;
        let m = 1.0;
        let e_newton = solver.solve(m, e);
        let e_halley = solver.solve_halley(m, e);
        assert!(approx_eq(e_newton, e_halley, 1e-10));
    }

    #[test]
    fn test_true_anomaly_roundtrip() {
        let e = 0.3;
        for nu_deg in [0, 30, 60, 90, 120, 150, 180] {
            let nu = (nu_deg as f64).to_radians();
            let big_e = KeplerSolver::eccentric_from_true(nu, e);
            let nu_back = KeplerSolver::true_anomaly_from_eccentric(big_e, e);
            assert!(
                approx_eq(nu, nu_back, 1e-12),
                "Roundtrip failed for nu={nu_deg}°"
            );
        }
    }

    #[test]
    fn test_circular_orbit_setup() {
        let orbit = CircularOrbit::unit();
        let state = orbit.setup();
        assert_eq!(state.particles.len(), 2);
        let r = state.particles[1].position.magnitude();
        assert!(approx_eq(r, 1.0, EPS));
        let v = state.particles[1].velocity.magnitude();
        let v_expected = orbit.orbital_velocity();
        assert!(approx_eq(v, v_expected, EPS));
    }

    #[test]
    fn test_circular_orbit_exact_solution_returns_to_start() {
        let orbit = CircularOrbit::unit();
        let t = orbit.period();
        let state = orbit.exact_solution(t);
        let initial = orbit.setup();
        let pos_err = state.particles[1].position.distance(initial.particles[1].position);
        assert!(pos_err < 1e-10, "Position error after one period: {pos_err}");
    }

    #[test]
    fn test_circular_orbit_energy_conservation() {
        let orbit = CircularOrbit::unit();
        let e0 = orbit.conserved_quantity(ConservationKind::Energy, &orbit.setup());
        for frac in [0.25, 0.5, 0.75, 1.0] {
            let t = frac * orbit.period();
            let state = orbit.exact_solution(t);
            let e = orbit.conserved_quantity(ConservationKind::Energy, &state);
            assert!(approx_eq(e, e0, 1e-8), "Energy at t={t}: {e} vs {e0}");
        }
    }

    #[test]
    fn test_circular_orbit_angular_momentum_conservation() {
        let orbit = CircularOrbit::unit();
        let l0 = orbit.conserved_quantity(ConservationKind::AngularMomentum, &orbit.setup());
        for frac in [0.1, 0.3, 0.5, 0.8, 1.0] {
            let t = frac * orbit.period();
            let state = orbit.exact_solution(t);
            let l = orbit.conserved_quantity(ConservationKind::AngularMomentum, &state);
            assert!(approx_eq(l, l0, 1e-8), "L at t={t}: {l} vs {l0}");
        }
    }

    #[test]
    fn test_elliptical_orbit_period() {
        let orbit = EllipticalOrbit::standard();
        let t = orbit.elements.period();
        let initial = orbit.setup();
        let state = orbit.exact_solution(t);
        let pos_err = state.particles[1].position.distance(initial.particles[1].position);
        assert!(pos_err < 1e-6, "Position error after period: {pos_err}");
    }

    #[test]
    fn test_elliptical_orbit_energy() {
        let orbit = EllipticalOrbit::standard();
        let e0 = orbit.conserved_quantity(ConservationKind::Energy, &orbit.setup());
        let expected_e = orbit.elements.specific_energy() * orbit.orbiting_mass;
        // Energy should be close to -mu/(2a) * m
        assert!(
            (e0 - expected_e).abs() / e0.abs() < 0.1,
            "e0={e0} vs expected {expected_e}"
        );
    }

    #[test]
    fn test_elliptical_vis_viva() {
        let orbit = EllipticalOrbit::standard();
        let r_p = orbit.elements.periapsis();
        let v_p = orbit.vis_viva_velocity(r_p);
        let r_a = orbit.elements.apoapsis();
        let v_a = orbit.vis_viva_velocity(r_a);
        // v_p > v_a for an elliptical orbit
        assert!(v_p > v_a, "Periapsis velocity should exceed apoapsis velocity");
    }

    #[test]
    fn test_orbital_elements_semi_latus_rectum() {
        let orbit = EllipticalOrbit::standard();
        let p = orbit.elements.semi_latus_rectum();
        let a = orbit.elements.semi_major_axis;
        let e = orbit.elements.eccentricity;
        assert!(approx_eq(p, a * (1.0 - e * e), EPS));
    }

    #[test]
    fn test_laplace_runge_lenz_conserved() {
        let orbit = CircularOrbit::unit();
        let lrl0 = orbit.laplace_runge_lenz(&orbit.setup());
        for frac in [0.25, 0.5, 0.75] {
            let t = frac * orbit.period();
            let state = orbit.exact_solution(t);
            let lrl = orbit.laplace_runge_lenz(&state);
            let diff = (lrl - lrl0).magnitude();
            assert!(diff < 1e-6, "LRL vector changed by {diff} at t={t}");
        }
    }

    #[test]
    fn test_elliptical_angular_momentum_conserved() {
        let orbit = EllipticalOrbit::standard();
        let l0 = orbit.conserved_quantity(ConservationKind::AngularMomentum, &orbit.setup());
        let period = orbit.elements.period();
        for frac in [0.1, 0.25, 0.5, 0.75, 0.9] {
            let t = frac * period;
            let state = orbit.exact_solution(t);
            let l = orbit.conserved_quantity(ConservationKind::AngularMomentum, &state);
            assert!(
                approx_eq(l, l0, 1e-6),
                "Angular momentum at t/T={frac}: {l} vs {l0}"
            );
        }
    }

    #[test]
    fn test_high_eccentricity_orbit() {
        let orbit = EllipticalOrbit::high_eccentricity();
        let period = orbit.elements.period();
        let state = orbit.exact_solution(period);
        let initial = orbit.setup();
        let pos_err = state.particles[1].position.distance(initial.particles[1].position);
        assert!(pos_err < 1e-4, "High-e orbit period error: {pos_err}");
    }
}
