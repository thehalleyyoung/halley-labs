//! Benchmark framework for evaluating numerical simulation accuracy.
//!
//! Provides traits and types for running physics benchmarks,
//! computing errors against analytical solutions, and performing convergence studies.

use sim_types::{
    ConservationKind, Particle, SimulationState, TimeSeries, Tolerance, Vec3,
    Violation, ViolationSeverity,
};
use serde::{Deserialize, Serialize};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Benchmark trait
// ---------------------------------------------------------------------------

/// A physics benchmark with a known analytical solution.
///
/// Implementors provide initial conditions, exact solutions at arbitrary times,
/// and the list of conserved quantities that a correct integrator must preserve.
pub trait Benchmark: Send + Sync {
    /// Human-readable name of this benchmark.
    fn name(&self) -> &str;

    /// Set up the initial simulation state at t = 0.
    fn setup(&self) -> SimulationState;

    /// Exact analytical solution at time `t`.
    fn exact_solution(&self, t: f64) -> SimulationState;

    /// Conservation laws that should hold for this system.
    fn conservation_laws(&self) -> Vec<ConservationKind>;

    /// Evaluate the conserved quantity `kind` from `state`.
    fn conserved_quantity(&self, kind: ConservationKind, state: &SimulationState) -> f64;

    /// Description of any intentional bugs in the benchmark.
    fn bug_description(&self) -> &str {
        ""
    }

    /// Whether this benchmark has an intentional bug.
    fn is_buggy(&self) -> bool {
        !self.bug_description().is_empty()
    }

    /// Characteristic timescale (e.g., orbital period).
    fn characteristic_time(&self) -> f64 {
        1.0
    }
}

// ---------------------------------------------------------------------------
// Conservation violation record
// ---------------------------------------------------------------------------

/// A record of a conservation violation detected during a benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConservationViolation {
    pub kind: ConservationKind,
    pub initial_value: f64,
    pub final_value: f64,
    pub max_deviation: f64,
    pub relative_error: f64,
    pub time_series: Vec<f64>,
}

impl ConservationViolation {
    /// Absolute change from initial to final value.
    pub fn absolute_drift(&self) -> f64 {
        (self.final_value - self.initial_value).abs()
    }

    /// Whether this violation exceeds the given tolerance.
    pub fn exceeds_tolerance(&self, tol: &Tolerance) -> bool {
        !tol.check(self.initial_value, self.final_value)
    }
}

// ---------------------------------------------------------------------------
// Benchmark result
// ---------------------------------------------------------------------------

/// Comprehensive result from running a benchmark.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub name: String,
    pub max_position_error: f64,
    pub max_velocity_error: f64,
    pub max_energy_error: f64,
    pub conservation_violations: Vec<ConservationViolation>,
    pub elapsed_seconds: f64,
    pub num_steps: usize,
    pub dt: f64,
    pub final_time: f64,
    pub position_error_history: Vec<f64>,
    pub energy_error_history: Vec<f64>,
}

impl BenchmarkResult {
    /// Check if any conservation violation exceeds the tolerance.
    pub fn has_violations(&self, tolerance: &Tolerance) -> bool {
        self.conservation_violations
            .iter()
            .any(|v| v.exceeds_tolerance(tolerance))
    }

    /// Worst conservation relative error across all quantities.
    pub fn worst_conservation_error(&self) -> f64 {
        self.conservation_violations
            .iter()
            .map(|v| v.relative_error)
            .fold(0.0_f64, f64::max)
    }

    /// Check if all position errors are within tolerance.
    pub fn position_within_tolerance(&self, tol: f64) -> bool {
        self.max_position_error < tol
    }

    /// Compute the convergence order given two results at different dt.
    pub fn convergence_order(coarse: &BenchmarkResult, fine: &BenchmarkResult) -> f64 {
        if coarse.max_position_error < 1e-30 || fine.max_position_error < 1e-30 {
            return 0.0;
        }
        let ratio = coarse.dt / fine.dt;
        let error_ratio = coarse.max_position_error / fine.max_position_error;
        error_ratio.ln() / ratio.ln()
    }
}

// ---------------------------------------------------------------------------
// Integrator type for benchmarking
// ---------------------------------------------------------------------------

/// A simple integrator function type: takes (state, dt, force_fn) -> new state.
pub type IntegratorFn = Box<dyn Fn(&SimulationState, f64, &dyn Fn(&SimulationState) -> Vec<Vec3>) -> SimulationState>;

/// Apply forward Euler integration.
pub fn euler_step(
    state: &SimulationState,
    dt: f64,
    force_fn: &dyn Fn(&SimulationState) -> Vec<Vec3>,
) -> SimulationState {
    let forces = force_fn(state);
    let mut new_particles = state.particles.clone();
    for (i, p) in new_particles.iter_mut().enumerate() {
        let acc = forces[i] / p.mass;
        p.velocity = p.velocity + acc * dt;
        p.position = p.position + p.velocity * dt;
    }
    SimulationState::new(new_particles, state.time + dt)
}

/// Apply symplectic Euler (semi-implicit) integration.
pub fn symplectic_euler_step(
    state: &SimulationState,
    dt: f64,
    force_fn: &dyn Fn(&SimulationState) -> Vec<Vec3>,
) -> SimulationState {
    let forces = force_fn(state);
    let mut new_particles = state.particles.clone();
    for (i, p) in new_particles.iter_mut().enumerate() {
        let acc = forces[i] / p.mass;
        p.velocity = p.velocity + acc * dt;
        p.position = p.position + p.velocity * dt;
    }
    SimulationState::new(new_particles, state.time + dt)
}

/// Apply velocity Verlet (Störmer-Verlet) integration.
pub fn verlet_step(
    state: &SimulationState,
    dt: f64,
    force_fn: &dyn Fn(&SimulationState) -> Vec<Vec3>,
) -> SimulationState {
    let forces_old = force_fn(state);
    let n = state.particles.len();
    let mut mid_particles = state.particles.clone();

    // Half-kick + drift
    for i in 0..n {
        let acc = forces_old[i] / mid_particles[i].mass;
        mid_particles[i].velocity = mid_particles[i].velocity + acc * (dt * 0.5);
        mid_particles[i].position = mid_particles[i].position + mid_particles[i].velocity * dt;
    }

    let mid_state = SimulationState::new(mid_particles.clone(), state.time + dt);
    let forces_new = force_fn(&mid_state);

    // Second half-kick
    for i in 0..n {
        let acc = forces_new[i] / mid_particles[i].mass;
        mid_particles[i].velocity = mid_particles[i].velocity + acc * (dt * 0.5);
    }

    SimulationState::new(mid_particles, state.time + dt)
}

/// Apply classical RK4 integration (non-symplectic).
pub fn rk4_step(
    state: &SimulationState,
    dt: f64,
    force_fn: &dyn Fn(&SimulationState) -> Vec<Vec3>,
) -> SimulationState {
    let n = state.particles.len();

    // k1
    let f1 = force_fn(state);

    // k2: evaluate at midpoint using k1
    let mut s2_particles = state.particles.clone();
    for i in 0..n {
        let a = f1[i] / s2_particles[i].mass;
        s2_particles[i].position = state.particles[i].position + state.particles[i].velocity * (dt * 0.5);
        s2_particles[i].velocity = state.particles[i].velocity + a * (dt * 0.5);
    }
    let s2 = SimulationState::new(s2_particles, state.time + dt * 0.5);
    let f2 = force_fn(&s2);

    // k3: evaluate at midpoint using k2
    let mut s3_particles = state.particles.clone();
    for i in 0..n {
        let a = f2[i] / s3_particles[i].mass;
        s3_particles[i].position = state.particles[i].position + s2.particles[i].velocity * (dt * 0.5);
        s3_particles[i].velocity = state.particles[i].velocity + a * (dt * 0.5);
    }
    let s3 = SimulationState::new(s3_particles, state.time + dt * 0.5);
    let f3 = force_fn(&s3);

    // k4: evaluate at endpoint using k3
    let mut s4_particles = state.particles.clone();
    for i in 0..n {
        let a = f3[i] / s4_particles[i].mass;
        s4_particles[i].position = state.particles[i].position + s3.particles[i].velocity * dt;
        s4_particles[i].velocity = state.particles[i].velocity + a * dt;
    }
    let s4 = SimulationState::new(s4_particles, state.time + dt);
    let f4 = force_fn(&s4);

    // Combine
    let mut final_particles = state.particles.clone();
    for i in 0..n {
        let a1 = f1[i] / final_particles[i].mass;
        let a2 = f2[i] / final_particles[i].mass;
        let a3 = f3[i] / final_particles[i].mass;
        let a4 = f4[i] / final_particles[i].mass;

        let v1 = state.particles[i].velocity;
        let v2 = s2.particles[i].velocity;
        let v3 = s3.particles[i].velocity;
        let v4 = s4.particles[i].velocity;

        final_particles[i].velocity = state.particles[i].velocity
            + (a1 + a2 * 2.0 + a3 * 2.0 + a4) * (dt / 6.0);
        final_particles[i].position = state.particles[i].position
            + (v1 + v2 * 2.0 + v3 * 2.0 + v4) * (dt / 6.0);
    }

    SimulationState::new(final_particles, state.time + dt)
}

// ---------------------------------------------------------------------------
// BenchmarkSuite
// ---------------------------------------------------------------------------

/// A suite of benchmarks that can be run together.
pub struct BenchmarkSuite {
    benchmarks: Vec<Box<dyn Benchmark>>,
}

impl BenchmarkSuite {
    pub fn new() -> Self {
        Self {
            benchmarks: Vec::new(),
        }
    }

    pub fn add(&mut self, benchmark: Box<dyn Benchmark>) {
        self.benchmarks.push(benchmark);
    }

    /// Run all benchmarks with a given integrator and timestep.
    pub fn run_all(
        &self,
        integrator: &dyn Fn(&SimulationState, f64, &dyn Fn(&SimulationState) -> Vec<Vec3>) -> SimulationState,
        force_fns: &[Box<dyn Fn(&SimulationState) -> Vec<Vec3>>],
        dt: f64,
        t_final: f64,
    ) -> Vec<BenchmarkResult> {
        self.benchmarks
            .iter()
            .zip(force_fns.iter())
            .map(|(bench, force_fn)| {
                run_benchmark(bench.as_ref(), integrator, force_fn.as_ref(), dt, t_final)
            })
            .collect()
    }

    /// Number of benchmarks in the suite.
    pub fn len(&self) -> usize {
        self.benchmarks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.benchmarks.is_empty()
    }
}

impl Default for BenchmarkSuite {
    fn default() -> Self {
        Self::new()
    }
}

/// Run a single benchmark and collect results.
pub fn run_benchmark(
    bench: &dyn Benchmark,
    integrator: &dyn Fn(&SimulationState, f64, &dyn Fn(&SimulationState) -> Vec<Vec3>) -> SimulationState,
    force_fn: &dyn Fn(&SimulationState) -> Vec<Vec3>,
    dt: f64,
    t_final: f64,
) -> BenchmarkResult {
    let start_time = Instant::now();
    let initial = bench.setup();
    let laws = bench.conservation_laws();

    // Record initial conserved quantities
    let initial_quantities: Vec<(ConservationKind, f64)> = laws
        .iter()
        .map(|&k| (k, bench.conserved_quantity(k, &initial)))
        .collect();

    let mut state = initial;
    let num_steps = ((t_final / dt).ceil()) as usize;
    let mut max_pos_err = 0.0_f64;
    let mut max_vel_err = 0.0_f64;
    let mut max_energy_err = 0.0_f64;
    let mut pos_err_history = Vec::with_capacity(num_steps);
    let mut energy_err_history = Vec::with_capacity(num_steps);
    let mut quantity_history: Vec<Vec<f64>> = vec![Vec::with_capacity(num_steps); laws.len()];

    for step in 0..num_steps {
        state = integrator(&state, dt, force_fn);
        let t = (step + 1) as f64 * dt;
        let exact = bench.exact_solution(t);

        // Position error
        let pos_err = crate::position_error(&exact, &state);
        max_pos_err = max_pos_err.max(pos_err);
        pos_err_history.push(pos_err);

        // Velocity error
        let vel_err = crate::velocity_error(&exact, &state);
        max_vel_err = max_vel_err.max(vel_err);

        // Conservation quantities
        for (j, &(kind, _)) in initial_quantities.iter().enumerate() {
            let val = bench.conserved_quantity(kind, &state);
            quantity_history[j].push(val);
            if kind == ConservationKind::Energy {
                let exact_e = bench.conserved_quantity(kind, &exact);
                let e_err = (val - exact_e).abs();
                max_energy_err = max_energy_err.max(e_err);
                energy_err_history.push(e_err);
            }
        }
    }

    let elapsed = start_time.elapsed().as_secs_f64();

    // Build conservation violations
    let mut violations = Vec::new();
    for (j, &(kind, initial_val)) in initial_quantities.iter().enumerate() {
        let history = &quantity_history[j];
        if history.is_empty() {
            continue;
        }
        let final_val = *history.last().unwrap();
        let max_dev = history
            .iter()
            .map(|&v| (v - initial_val).abs())
            .fold(0.0_f64, f64::max);
        let rel_err = if initial_val.abs() > 1e-30 {
            max_dev / initial_val.abs()
        } else {
            max_dev
        };
        violations.push(ConservationViolation {
            kind,
            initial_value: initial_val,
            final_value: final_val,
            max_deviation: max_dev,
            relative_error: rel_err,
            time_series: history.clone(),
        });
    }

    BenchmarkResult {
        name: bench.name().to_string(),
        max_position_error: max_pos_err,
        max_velocity_error: max_vel_err,
        max_energy_error: max_energy_err,
        conservation_violations: violations,
        elapsed_seconds: elapsed,
        num_steps,
        dt,
        final_time: t_final,
        position_error_history: pos_err_history,
        energy_error_history: energy_err_history,
    }
}

// ---------------------------------------------------------------------------
// Convergence study
// ---------------------------------------------------------------------------

/// Result of a convergence study at multiple timestep values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceResult {
    pub dt_values: Vec<f64>,
    pub position_errors: Vec<f64>,
    pub energy_errors: Vec<f64>,
    pub estimated_order: f64,
    pub estimated_energy_order: f64,
}

impl ConvergenceResult {
    /// Estimate convergence order from position errors using least-squares log-log fit.
    pub fn estimate_order(dt_values: &[f64], errors: &[f64]) -> f64 {
        if dt_values.len() < 2 {
            return 0.0;
        }
        let n = dt_values.len() as f64;
        let mut sum_log_dt = 0.0;
        let mut sum_log_err = 0.0;
        let mut sum_log_dt_sq = 0.0;
        let mut sum_log_dt_err = 0.0;
        let mut count = 0.0;

        for (dt, err) in dt_values.iter().zip(errors.iter()) {
            if *err < 1e-30 || *dt < 1e-30 {
                continue;
            }
            let log_dt = dt.ln();
            let log_err = err.ln();
            sum_log_dt += log_dt;
            sum_log_err += log_err;
            sum_log_dt_sq += log_dt * log_dt;
            sum_log_dt_err += log_dt * log_err;
            count += 1.0;
        }

        if count < 2.0 {
            return 0.0;
        }

        (count * sum_log_dt_err - sum_log_dt * sum_log_err)
            / (count * sum_log_dt_sq - sum_log_dt * sum_log_dt)
    }
}

/// Run a convergence study with multiple timestep values.
pub fn convergence_study(
    bench: &dyn Benchmark,
    integrator: &dyn Fn(&SimulationState, f64, &dyn Fn(&SimulationState) -> Vec<Vec3>) -> SimulationState,
    force_fn: &dyn Fn(&SimulationState) -> Vec<Vec3>,
    dt_values: &[f64],
    t_final: f64,
) -> ConvergenceResult {
    let mut position_errors = Vec::new();
    let mut energy_errors = Vec::new();

    for &dt in dt_values {
        let result = run_benchmark(bench, integrator, force_fn, dt, t_final);
        position_errors.push(result.max_position_error);
        energy_errors.push(result.max_energy_error);
    }

    let order = ConvergenceResult::estimate_order(dt_values, &position_errors);
    let energy_order = ConvergenceResult::estimate_order(dt_values, &energy_errors);

    ConvergenceResult {
        dt_values: dt_values.to_vec(),
        position_errors,
        energy_errors,
        estimated_order: order,
        estimated_energy_order: energy_order,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-10;

    struct TrivialBenchmark;

    impl Benchmark for TrivialBenchmark {
        fn name(&self) -> &str {
            "Trivial Free Particle"
        }

        fn setup(&self) -> SimulationState {
            let p = Particle::new(1.0, Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0)).with_id("p0");
            SimulationState::new(vec![p], 0.0)
        }

        fn exact_solution(&self, t: f64) -> SimulationState {
            let p = Particle::new(1.0, Vec3::new(t, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0))
                .with_id("p0");
            SimulationState::new(vec![p], t)
        }

        fn conservation_laws(&self) -> Vec<ConservationKind> {
            vec![ConservationKind::Energy, ConservationKind::Momentum]
        }

        fn conserved_quantity(&self, kind: ConservationKind, state: &SimulationState) -> f64 {
            match kind {
                ConservationKind::Energy => {
                    state.particles.iter().map(|p| p.kinetic_energy()).sum()
                }
                ConservationKind::Momentum => state
                    .particles
                    .iter()
                    .fold(Vec3::ZERO, |acc, p| acc + p.momentum())
                    .magnitude(),
                _ => 0.0,
            }
        }
    }

    fn zero_force(_state: &SimulationState) -> Vec<Vec3> {
        vec![Vec3::ZERO]
    }

    #[test]
    fn test_euler_free_particle() {
        let bench = TrivialBenchmark;
        let result = run_benchmark(&bench, &euler_step, &zero_force, 0.01, 1.0);
        assert!(result.max_position_error < 0.02);
        assert_eq!(result.name, "Trivial Free Particle");
    }

    #[test]
    fn test_verlet_free_particle() {
        let bench = TrivialBenchmark;
        let result = run_benchmark(&bench, &verlet_step, &zero_force, 0.01, 1.0);
        assert!(result.max_position_error < 1e-12);
    }

    #[test]
    fn test_rk4_free_particle() {
        let bench = TrivialBenchmark;
        let result = run_benchmark(&bench, &rk4_step, &zero_force, 0.01, 1.0);
        assert!(result.max_position_error < 1e-10);
    }

    #[test]
    fn test_benchmark_result_violations() {
        let result = BenchmarkResult {
            name: "test".into(),
            max_position_error: 0.01,
            max_velocity_error: 0.001,
            max_energy_error: 0.0001,
            conservation_violations: vec![ConservationViolation {
                kind: ConservationKind::Energy,
                initial_value: 1.0,
                final_value: 1.01,
                max_deviation: 0.01,
                relative_error: 0.01,
                time_series: vec![1.005, 1.01],
            }],
            elapsed_seconds: 0.1,
            num_steps: 100,
            dt: 0.01,
            final_time: 1.0,
            position_error_history: vec![],
            energy_error_history: vec![],
        };

        assert!(result.has_violations(&Tolerance::absolute(0.001)));
        assert!(!result.has_violations(&Tolerance::absolute(0.1)));
        assert!((result.worst_conservation_error() - 0.01).abs() < EPS);
    }

    #[test]
    fn test_convergence_order_estimate() {
        // Linear convergence: error proportional to dt
        let dts = vec![0.1, 0.05, 0.025, 0.0125];
        let errors: Vec<f64> = dts.iter().map(|&dt| dt).collect();
        let order = ConvergenceResult::estimate_order(&dts, &errors);
        assert!((order - 1.0).abs() < 0.1);

        // Quadratic convergence: error proportional to dt^2
        let errors2: Vec<f64> = dts.iter().map(|&dt| dt * dt).collect();
        let order2 = ConvergenceResult::estimate_order(&dts, &errors2);
        assert!((order2 - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_suite_creation() {
        let mut suite = BenchmarkSuite::new();
        suite.add(Box::new(TrivialBenchmark));
        assert_eq!(suite.len(), 1);
        assert!(!suite.is_empty());
    }

    #[test]
    fn test_conservation_violation_absolute_drift() {
        let v = ConservationViolation {
            kind: ConservationKind::Energy,
            initial_value: 10.0,
            final_value: 10.05,
            max_deviation: 0.05,
            relative_error: 0.005,
            time_series: vec![],
        };
        assert!((v.absolute_drift() - 0.05).abs() < EPS);
    }
}
