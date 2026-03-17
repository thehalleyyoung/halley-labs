//! # sim-integrator
//!
//! A comprehensive library of numerical integrators for physics simulation.
//!
//! This crate provides a wide range of integration methods including:
//! - Euler methods (forward, backward, improved)
//! - Symplectic Euler methods
//! - Störmer-Verlet and Velocity Verlet
//! - Leapfrog integrators
//! - Runge-Kutta methods (RK2, RK4, RK38, RKF45, DOPRI5)
//! - Yoshida symplectic integrators (4th, 6th, 8th order)
//! - Ruth symplectic methods
//! - Forest-Ruth and PEFRL integrators
//! - Implicit midpoint rule
//! - Gauss-Legendre Runge-Kutta methods
//! - Composition methods (Suzuki, triple-jump)
//! - Adaptive step size control (PI, PID controllers)
//! - Operator splitting (Lie-Trotter, Strang)
//! - N-body simulation helpers

pub mod euler;
pub mod symplectic_euler;
pub mod verlet;
pub mod leapfrog;
pub mod runge_kutta;
pub mod yoshida;
pub mod ruth;
pub mod forest_ruth;
pub mod implicit_midpoint;
pub mod gauss_legendre;
pub mod composition;
pub mod adaptive;
pub mod splitting;
pub mod nbody;

pub use euler::{ForwardEuler, BackwardEuler, ImprovedEuler};
pub use symplectic_euler::{SymplecticEulerA, SymplecticEulerB};
pub use verlet::{StormerVerlet, VelocityVerlet, PositionVerlet};
pub use leapfrog::{Leapfrog, LeapfrogDKD};
pub use runge_kutta::{RK2, RK4, RK38, RKF45, DOPRI5};
pub use yoshida::{Yoshida4, Yoshida6, Yoshida8};
pub use ruth::{Ruth3, Ruth4};
pub use forest_ruth::{ForestRuth, PEFRL};
pub use implicit_midpoint::ImplicitMidpoint;
pub use gauss_legendre::{GaussLegendre2, GaussLegendre4, GaussLegendre6};
pub use composition::{ABAComposition, SuzukiComposition, TripleJump};
pub use adaptive::{EmbeddedPair, PIController, PIDController, StepSizeController};
pub use splitting::{LieTrotter, StrangSplitting};
pub use nbody::{NBodyState, NBodySimulation};

use thiserror::Error;

/// Errors that can occur during integration.
#[derive(Error, Debug, Clone)]
pub enum IntegratorError {
    #[error("Newton iteration failed to converge after {max_iters} iterations (residual: {residual:.2e})")]
    ConvergenceFailure { max_iters: usize, residual: f64 },

    #[error("Step size {dt:.2e} is below minimum allowed {min_dt:.2e}")]
    StepSizeTooSmall { dt: f64, min_dt: f64 },

    #[error("NaN or infinity detected in state at index {index}")]
    NonFiniteValue { index: usize },

    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },

    #[error("Invalid parameter: {message}")]
    InvalidParameter { message: String },
}

/// The core integrator trait for stepping ODE systems forward in time.
///
/// The state vector `y` and its derivative `dy/dt` are represented as `&[f64]` slices.
/// The derivative function `deriv` computes `dy/dt = f(t, y)` and writes the result
/// into the provided output slice.
pub trait Integrator {
    /// Advance the state by one time step `dt`.
    ///
    /// # Arguments
    /// * `state` - The current state vector (modified in place)
    /// * `t` - The current time
    /// * `dt` - The time step
    /// * `deriv` - Function computing dy/dt = f(t, y, out)
    fn step(&self, state: &mut [f64], t: f64, dt: f64, deriv: &dyn Fn(f64, &[f64], &mut [f64]));

    /// Return the name of this integrator.
    fn name(&self) -> &str;

    /// Return the order of accuracy of this integrator.
    fn order(&self) -> u32;

    /// Return whether this integrator is symplectic (preserves phase space volume).
    fn is_symplectic(&self) -> bool {
        false
    }
}

/// Trait for separable Hamiltonian integrators (H = T(p) + V(q)).
///
/// These integrators work with systems where the Hamiltonian splits into
/// kinetic energy T depending only on momenta, and potential energy V
/// depending only on positions.
pub trait SeparableIntegrator {
    /// Advance the separable Hamiltonian system by one step.
    ///
    /// # Arguments
    /// * `q` - Position coordinates (modified in place)
    /// * `p` - Momentum coordinates (modified in place)
    /// * `t` - Current time
    /// * `dt` - Time step
    /// * `force` - Function computing force = -dV/dq, writing into output slice
    /// * `mass` - Mass values for kinetic energy T = p^2/(2m)
    fn step_separable(
        &self,
        q: &mut [f64],
        p: &mut [f64],
        t: f64,
        dt: f64,
        force: &dyn Fn(f64, &[f64], &mut [f64]),
        mass: &[f64],
    );

    fn name(&self) -> &str;
    fn order(&self) -> u32;
    fn is_symplectic(&self) -> bool {
        true
    }
}

/// Configuration for implicit solvers.
#[derive(Debug, Clone)]
pub struct ImplicitConfig {
    /// Maximum number of iterations for Newton/fixed-point solver.
    pub max_iters: usize,
    /// Convergence tolerance.
    pub tolerance: f64,
    /// Relaxation parameter for fixed-point iteration (0 < omega <= 1).
    pub relaxation: f64,
}

impl Default for ImplicitConfig {
    fn default() -> Self {
        Self {
            max_iters: 50,
            tolerance: 1e-12,
            relaxation: 1.0,
        }
    }
}

/// Result of an adaptive integration step.
#[derive(Debug, Clone)]
pub struct AdaptiveStepResult {
    /// Whether the step was accepted.
    pub accepted: bool,
    /// The error estimate for this step.
    pub error_estimate: f64,
    /// The suggested next step size.
    pub next_dt: f64,
}

/// Integrate a system from t0 to t_end using the given integrator.
///
/// Returns the final state and the times at which steps were taken.
pub fn integrate(
    integrator: &dyn Integrator,
    state: &mut [f64],
    t0: f64,
    t_end: f64,
    dt: f64,
    deriv: &dyn Fn(f64, &[f64], &mut [f64]),
) -> Vec<f64> {
    let mut t = t0;
    let mut times = vec![t0];
    while t < t_end - 1e-14 {
        let step = if t + dt > t_end { t_end - t } else { dt };
        integrator.step(state, t, step, deriv);
        t += step;
        times.push(t);
    }
    times
}

/// Integrate and record the full trajectory.
///
/// Returns (times, states) where states[i] is the state at times[i].
pub fn integrate_trajectory(
    integrator: &dyn Integrator,
    initial_state: &[f64],
    t0: f64,
    t_end: f64,
    dt: f64,
    deriv: &dyn Fn(f64, &[f64], &mut [f64]),
) -> (Vec<f64>, Vec<Vec<f64>>) {
    let mut state = initial_state.to_vec();
    let mut times = vec![t0];
    let mut states = vec![initial_state.to_vec()];
    let mut t = t0;
    while t < t_end - 1e-14 {
        let step = if t + dt > t_end { t_end - t } else { dt };
        integrator.step(&mut state, t, step, deriv);
        t += step;
        times.push(t);
        states.push(state.clone());
    }
    (times, states)
}

/// Compute the L2 norm of a vector.
pub fn l2_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Compute the infinity norm (max absolute value) of a vector.
pub fn linf_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x.abs()).fold(0.0_f64, f64::max)
}

/// Check that all values in a state vector are finite.
pub fn check_finite(state: &[f64]) -> Result<(), IntegratorError> {
    for (i, &val) in state.iter().enumerate() {
        if !val.is_finite() {
            return Err(IntegratorError::NonFiniteValue { index: i });
        }
    }
    Ok(())
}

/// Compute the weighted RMS norm used for error estimation.
///
/// norm = sqrt(1/n * sum((err_i / (atol + rtol * |y_i|))^2))
pub fn weighted_rms_norm(err: &[f64], y: &[f64], atol: f64, rtol: f64) -> f64 {
    let n = err.len();
    if n == 0 {
        return 0.0;
    }
    let sum: f64 = err
        .iter()
        .zip(y.iter())
        .map(|(&e, &yi)| {
            let scale = atol + rtol * yi.abs();
            (e / scale) * (e / scale)
        })
        .sum();
    (sum / n as f64).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l2_norm() {
        assert!((l2_norm(&[3.0, 4.0]) - 5.0).abs() < 1e-15);
        assert!((l2_norm(&[1.0, 1.0, 1.0]) - 3.0_f64.sqrt()).abs() < 1e-15);
        assert!(l2_norm(&[]) == 0.0);
    }

    #[test]
    fn test_linf_norm() {
        assert!((linf_norm(&[1.0, -3.0, 2.0]) - 3.0).abs() < 1e-15);
        assert!((linf_norm(&[-5.0, 3.0, 1.0]) - 5.0).abs() < 1e-15);
    }

    #[test]
    fn test_check_finite() {
        assert!(check_finite(&[1.0, 2.0, 3.0]).is_ok());
        assert!(check_finite(&[1.0, f64::NAN, 3.0]).is_err());
        assert!(check_finite(&[f64::INFINITY]).is_err());
    }

    #[test]
    fn test_weighted_rms_norm() {
        let err = vec![1e-6, 1e-6];
        let y = vec![1.0, 1.0];
        let norm = weighted_rms_norm(&err, &y, 1e-8, 1e-6);
        // scale = 1e-8 + 1e-6 * 1.0 = 1.001e-6
        // each term = (1e-6 / 1.001e-6)^2 ≈ 0.998
        assert!(norm > 0.9 && norm < 1.1);
    }

    #[test]
    fn test_integrate_simple() {
        // dy/dt = 1.0, y(0) = 0, solution y(t) = t
        let euler = ForwardEuler;
        let mut state = vec![0.0];
        let deriv = |_t: f64, _y: &[f64], out: &mut [f64]| {
            out[0] = 1.0;
        };
        integrate(&euler, &mut state, 0.0, 1.0, 0.01, &deriv);
        assert!((state[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_integrate_trajectory_records() {
        let euler = ForwardEuler;
        let deriv = |_t: f64, _y: &[f64], out: &mut [f64]| {
            out[0] = 1.0;
        };
        let (times, states) = integrate_trajectory(&euler, &[0.0], 0.0, 0.5, 0.1, &deriv);
        assert_eq!(times.len(), states.len());
        assert!(times.len() >= 5);
        for (i, (t, s)) in times.iter().zip(states.iter()).enumerate() {
            if i > 0 {
                assert!((s[0] - *t).abs() < 1e-10, "state should equal time for dy/dt=1");
            }
        }
    }

    #[test]
    fn test_implicit_config_default() {
        let cfg = ImplicitConfig::default();
        assert_eq!(cfg.max_iters, 50);
        assert!((cfg.tolerance - 1e-12).abs() < 1e-20);
        assert!((cfg.relaxation - 1.0).abs() < 1e-20);
    }

    #[test]
    fn test_integrator_error_display() {
        let e = IntegratorError::ConvergenceFailure {
            max_iters: 100,
            residual: 1e-3,
        };
        let msg = format!("{}", e);
        assert!(msg.contains("100"));
        assert!(msg.contains("converge"));
    }
}
