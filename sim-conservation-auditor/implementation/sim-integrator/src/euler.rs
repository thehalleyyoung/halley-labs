//! Euler family of integrators.
//!
//! This module implements the classical Euler methods:
//! - **Forward Euler** (explicit, first order)
//! - **Backward Euler** (implicit, first order, A-stable)
//! - **Improved Euler / Heun's method** (explicit, second order)

use crate::{Integrator, ImplicitConfig};

// ─────────────────────────────────────────────────────────────
// Forward Euler
// ─────────────────────────────────────────────────────────────

/// Forward (explicit) Euler method.
///
/// The simplest possible integrator:
///   y_{n+1} = y_n + dt * f(t_n, y_n)
///
/// First-order accurate, conditionally stable.
#[derive(Debug, Clone, Copy)]
pub struct ForwardEuler;

impl Integrator for ForwardEuler {
    fn step(&self, state: &mut [f64], t: f64, dt: f64, deriv: &dyn Fn(f64, &[f64], &mut [f64])) {
        let n = state.len();
        let mut k = vec![0.0; n];
        deriv(t, state, &mut k);
        for i in 0..n {
            state[i] += dt * k[i];
        }
    }

    fn name(&self) -> &str {
        "Forward Euler"
    }

    fn order(&self) -> u32 {
        1
    }
}

impl ForwardEuler {
    /// Create a new ForwardEuler integrator.
    pub fn new() -> Self {
        Self
    }

    /// Step with recording of the derivative for diagnostics.
    pub fn step_with_derivative(
        &self,
        state: &mut [f64],
        t: f64,
        dt: f64,
        deriv: &dyn Fn(f64, &[f64], &mut [f64]),
    ) -> Vec<f64> {
        let n = state.len();
        let mut k = vec![0.0; n];
        deriv(t, state, &mut k);
        for i in 0..n {
            state[i] += dt * k[i];
        }
        k
    }

    /// Integrate multiple steps, returning the trajectory.
    pub fn integrate_steps(
        &self,
        state: &mut [f64],
        t0: f64,
        dt: f64,
        n_steps: usize,
        deriv: &dyn Fn(f64, &[f64], &mut [f64]),
    ) -> Vec<Vec<f64>> {
        let mut trajectory = Vec::with_capacity(n_steps + 1);
        trajectory.push(state.to_vec());
        let mut t = t0;
        for _ in 0..n_steps {
            self.step(state, t, dt, deriv);
            t += dt;
            trajectory.push(state.to_vec());
        }
        trajectory
    }

    /// Estimate the local truncation error by comparing with a half-step method.
    ///
    /// Takes one full step and two half-steps, returns the difference.
    pub fn estimate_error(
        &self,
        state: &[f64],
        t: f64,
        dt: f64,
        deriv: &dyn Fn(f64, &[f64], &mut [f64]),
    ) -> Vec<f64> {
        let n = state.len();

        // Full step
        let mut full = state.to_vec();
        self.step(&mut full, t, dt, deriv);

        // Two half steps
        let mut half = state.to_vec();
        self.step(&mut half, t, dt / 2.0, deriv);
        self.step(&mut half, t + dt / 2.0, dt / 2.0, deriv);

        // Error estimate (Richardson extrapolation factor = 2^p - 1 = 1 for first order)
        let mut err = vec![0.0; n];
        for i in 0..n {
            err[i] = (half[i] - full[i]).abs();
        }
        err
    }
}

// ─────────────────────────────────────────────────────────────
// Backward Euler
// ─────────────────────────────────────────────────────────────

/// Backward (implicit) Euler method.
///
/// Solves: y_{n+1} = y_n + dt * f(t_{n+1}, y_{n+1})
///
/// Uses Newton iteration (with numerical Jacobian) or fixed-point iteration
/// to solve the implicit equation. First-order accurate, A-stable and L-stable.
#[derive(Debug, Clone)]
pub struct BackwardEuler {
    config: ImplicitConfig,
    /// Use Newton iteration if true, fixed-point iteration if false.
    use_newton: bool,
}

impl BackwardEuler {
    /// Create a backward Euler integrator with default settings.
    pub fn new() -> Self {
        Self {
            config: ImplicitConfig::default(),
            use_newton: true,
        }
    }

    /// Create with custom configuration.
    pub fn with_config(config: ImplicitConfig) -> Self {
        Self {
            config,
            use_newton: true,
        }
    }

    /// Create using fixed-point iteration instead of Newton.
    pub fn with_fixed_point(config: ImplicitConfig) -> Self {
        Self {
            config,
            use_newton: false,
        }
    }

    /// Solve the implicit equation using Newton iteration with numerical Jacobian.
    ///
    /// We solve: G(y) = y - y_n - dt * f(t+dt, y) = 0
    /// Using: y^{k+1} = y^k - J^{-1} * G(y^k)
    /// where J ≈ I - dt * df/dy (computed numerically).
    fn newton_solve(
        &self,
        y_n: &[f64],
        t_next: f64,
        dt: f64,
        deriv: &dyn Fn(f64, &[f64], &mut [f64]),
    ) -> Vec<f64> {
        let n = y_n.len();
        let mut y = y_n.to_vec();
        let mut f_val = vec![0.0; n];
        let mut f_pert = vec![0.0; n];
        let mut g = vec![0.0; n];
        let eps = 1e-8;

        // Initial guess: forward Euler step
        deriv(t_next - dt, y_n, &mut f_val);
        for i in 0..n {
            y[i] = y_n[i] + dt * f_val[i];
        }

        for _iter in 0..self.config.max_iters {
            // Evaluate G(y) = y - y_n - dt * f(t+dt, y)
            deriv(t_next, &y, &mut f_val);
            let mut max_g = 0.0_f64;
            for i in 0..n {
                g[i] = y[i] - y_n[i] - dt * f_val[i];
                max_g = max_g.max(g[i].abs());
            }

            if max_g < self.config.tolerance {
                return y;
            }

            // For 1D or small systems, use direct Newton with numerical Jacobian
            if n <= 4 {
                // Build Jacobian J = I - dt * df/dy (numerically)
                let mut jacobian = vec![vec![0.0; n]; n];
                for j in 0..n {
                    let mut y_pert = y.clone();
                    let h = eps * (1.0 + y[j].abs());
                    y_pert[j] += h;
                    deriv(t_next, &y_pert, &mut f_pert);
                    for i in 0..n {
                        let df_dy = (f_pert[i] - f_val[i]) / h;
                        jacobian[i][j] = if i == j { 1.0 } else { 0.0 } - dt * df_dy;
                    }
                }

                // Solve J * delta = -G using Gaussian elimination
                let delta = solve_linear_system(&jacobian, &g.iter().map(|x| -x).collect::<Vec<_>>());
                for i in 0..n {
                    y[i] += delta[i];
                }
            } else {
                // For larger systems, use diagonal approximation (simplified Newton)
                for j in 0..n {
                    let mut y_pert = y.clone();
                    let h = eps * (1.0 + y[j].abs());
                    y_pert[j] += h;
                    deriv(t_next, &y_pert, &mut f_pert);
                    let df_dy_jj = (f_pert[j] - f_val[j]) / h;
                    let jac_jj = 1.0 - dt * df_dy_jj;
                    if jac_jj.abs() > 1e-15 {
                        y[j] -= g[j] / jac_jj;
                    }
                }
            }
        }
        y
    }

    /// Solve the implicit equation using fixed-point iteration.
    ///
    /// Iterate: y^{k+1} = y_n + dt * f(t+dt, y^k)
    fn fixed_point_solve(
        &self,
        y_n: &[f64],
        t_next: f64,
        dt: f64,
        deriv: &dyn Fn(f64, &[f64], &mut [f64]),
    ) -> Vec<f64> {
        let n = y_n.len();
        let mut y = y_n.to_vec();
        let mut f_val = vec![0.0; n];

        for _iter in 0..self.config.max_iters {
            deriv(t_next, &y, &mut f_val);
            let mut max_diff = 0.0_f64;
            for i in 0..n {
                let y_new = y_n[i] + dt * f_val[i];
                let diff = (y_new - y[i]).abs();
                max_diff = max_diff.max(diff);
                y[i] = self.config.relaxation * y_new + (1.0 - self.config.relaxation) * y[i];
            }
            if max_diff < self.config.tolerance {
                return y;
            }
        }
        y
    }
}

impl Integrator for BackwardEuler {
    fn step(&self, state: &mut [f64], t: f64, dt: f64, deriv: &dyn Fn(f64, &[f64], &mut [f64])) {
        let y_n = state.to_vec();
        let t_next = t + dt;
        let result = if self.use_newton {
            self.newton_solve(&y_n, t_next, dt, deriv)
        } else {
            self.fixed_point_solve(&y_n, t_next, dt, deriv)
        };
        state.copy_from_slice(&result);
    }

    fn name(&self) -> &str {
        "Backward Euler"
    }

    fn order(&self) -> u32 {
        1
    }
}

// ─────────────────────────────────────────────────────────────
// Improved Euler (Heun's Method)
// ─────────────────────────────────────────────────────────────

/// Improved Euler method (Heun's method / Explicit Trapezoidal Rule).
///
///   k1 = f(t_n, y_n)
///   k2 = f(t_n + dt, y_n + dt * k1)
///   y_{n+1} = y_n + dt/2 * (k1 + k2)
///
/// Second-order accurate, explicit.
#[derive(Debug, Clone, Copy)]
pub struct ImprovedEuler;

impl ImprovedEuler {
    /// Create a new ImprovedEuler integrator.
    pub fn new() -> Self {
        Self
    }

    /// Integrate multiple steps and return the trajectory.
    pub fn integrate_steps(
        &self,
        state: &mut [f64],
        t0: f64,
        dt: f64,
        n_steps: usize,
        deriv: &dyn Fn(f64, &[f64], &mut [f64]),
    ) -> Vec<Vec<f64>> {
        let mut trajectory = Vec::with_capacity(n_steps + 1);
        trajectory.push(state.to_vec());
        let mut t = t0;
        for _ in 0..n_steps {
            self.step(state, t, dt, deriv);
            t += dt;
            trajectory.push(state.to_vec());
        }
        trajectory
    }

    /// Estimate the local truncation error by step-doubling (Richardson).
    pub fn estimate_error(
        &self,
        state: &[f64],
        t: f64,
        dt: f64,
        deriv: &dyn Fn(f64, &[f64], &mut [f64]),
    ) -> Vec<f64> {
        let n = state.len();
        let mut full = state.to_vec();
        self.step(&mut full, t, dt, deriv);

        let mut half = state.to_vec();
        self.step(&mut half, t, dt / 2.0, deriv);
        self.step(&mut half, t + dt / 2.0, dt / 2.0, deriv);

        // For second-order method, Richardson factor = 2^2 - 1 = 3
        let mut err = vec![0.0; n];
        for i in 0..n {
            err[i] = (half[i] - full[i]).abs() / 3.0;
        }
        err
    }
}

impl Integrator for ImprovedEuler {
    fn step(&self, state: &mut [f64], t: f64, dt: f64, deriv: &dyn Fn(f64, &[f64], &mut [f64])) {
        let n = state.len();
        let mut k1 = vec![0.0; n];
        let mut k2 = vec![0.0; n];
        let mut y_temp = vec![0.0; n];

        // k1 = f(t, y)
        deriv(t, state, &mut k1);

        // y_temp = y + dt * k1
        for i in 0..n {
            y_temp[i] = state[i] + dt * k1[i];
        }

        // k2 = f(t + dt, y + dt * k1)
        deriv(t + dt, &y_temp, &mut k2);

        // y_{n+1} = y_n + dt/2 * (k1 + k2)
        for i in 0..n {
            state[i] += dt / 2.0 * (k1[i] + k2[i]);
        }
    }

    fn name(&self) -> &str {
        "Improved Euler (Heun)"
    }

    fn order(&self) -> u32 {
        2
    }
}

// ─────────────────────────────────────────────────────────────
// Helper: small linear system solver
// ─────────────────────────────────────────────────────────────

/// Solve a small dense linear system Ax = b using Gaussian elimination with partial pivoting.
fn solve_linear_system(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    if n == 0 {
        return vec![];
    }

    // Build augmented matrix
    let mut aug = vec![vec![0.0; n + 1]; n];
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = a[i][j];
        }
        aug[i][n] = b[i];
    }

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_val = aug[col][col].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }

        if max_val < 1e-15 {
            continue; // singular or nearly singular
        }

        // Swap rows
        if max_row != col {
            aug.swap(col, max_row);
        }

        // Eliminate below
        let pivot = aug[col][col];
        for row in (col + 1)..n {
            let factor = aug[row][col] / pivot;
            for j in col..=n {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = aug[i][n];
        for j in (i + 1)..n {
            sum -= aug[i][j] * x[j];
        }
        if aug[i][i].abs() > 1e-15 {
            x[i] = sum / aug[i][i];
        }
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Integrator;

    /// Exponential decay: y' = -y, y(0) = 1, exact: y(t) = e^{-t}
    fn exp_decay_deriv(_t: f64, y: &[f64], out: &mut [f64]) {
        out[0] = -y[0];
    }

    /// Linear ODE: y' = 2*t, y(0) = 0, exact: y(t) = t^2
    fn linear_deriv(t: f64, _y: &[f64], out: &mut [f64]) {
        out[0] = 2.0 * t;
    }

    /// System: y' = -lambda*y for stiff testing
    fn stiff_deriv(lambda: f64) -> impl Fn(f64, &[f64], &mut [f64]) {
        move |_t: f64, y: &[f64], out: &mut [f64]| {
            out[0] = -lambda * y[0];
        }
    }

    // ── Forward Euler tests ──

    #[test]
    fn test_forward_euler_exponential_decay() {
        let euler = ForwardEuler;
        let dt = 0.001;
        let mut state = vec![1.0];
        let mut t = 0.0;
        let t_end = 1.0_f64;
        while t < t_end - 1e-14 {
            euler.step(&mut state, t, dt, &exp_decay_deriv);
            t += dt;
        }
        let exact = (-1.0_f64).exp();
        let error = (state[0] - exact).abs();
        // First-order: error ~ O(dt) ~ O(0.001)
        assert!(error < 0.01, "Forward Euler error too large: {}", error);
    }

    #[test]
    fn test_forward_euler_convergence_order() {
        // Run with dt and dt/2, check that error ratio ≈ 2 (first order)
        let euler = ForwardEuler;
        let t_end = 1.0_f64;
        let exact = (-t_end).exp();

        let dt1 = 0.01;
        let mut state1 = vec![1.0];
        let mut t = 0.0;
        while t < t_end - 1e-14 {
            euler.step(&mut state1, t, dt1, &exp_decay_deriv);
            t += dt1;
        }
        let err1 = (state1[0] - exact).abs();

        let dt2 = 0.005;
        let mut state2 = vec![1.0];
        t = 0.0;
        while t < t_end - 1e-14 {
            euler.step(&mut state2, t, dt2, &exp_decay_deriv);
            t += dt2;
        }
        let err2 = (state2[0] - exact).abs();

        let ratio = err1 / err2;
        // For first-order method, error ratio should be ~2.0
        assert!(
            ratio > 1.8 && ratio < 2.3,
            "Expected ratio ~2.0 for first order, got {}",
            ratio
        );
    }

    #[test]
    fn test_forward_euler_linear() {
        let euler = ForwardEuler;
        let dt = 0.01;
        let mut state = vec![0.0];
        let mut t = 0.0;
        while t < 1.0 - 1e-14 {
            euler.step(&mut state, t, dt, &linear_deriv);
            t += dt;
        }
        // Exact: y(1) = 1.0. Euler has first-order error
        assert!(
            (state[0] - 1.0).abs() < 0.02,
            "Forward Euler linear error: {}",
            (state[0] - 1.0).abs()
        );
    }

    #[test]
    fn test_forward_euler_integrate_steps() {
        let euler = ForwardEuler;
        let mut state = vec![1.0];
        let traj = euler.integrate_steps(&mut state, 0.0, 0.1, 10, &exp_decay_deriv);
        assert_eq!(traj.len(), 11);
        // Values should be decreasing
        for i in 1..traj.len() {
            assert!(traj[i][0] < traj[i - 1][0], "Should decay monotonically");
        }
    }

    #[test]
    fn test_forward_euler_error_estimate() {
        let euler = ForwardEuler;
        let state = vec![1.0];
        let err = euler.estimate_error(&state, 0.0, 0.01, &exp_decay_deriv);
        assert!(err[0] > 0.0, "Error estimate should be positive");
        assert!(err[0] < 0.001, "Error estimate should be small for small dt");
    }

    #[test]
    fn test_forward_euler_name_order() {
        let euler = ForwardEuler;
        assert_eq!(euler.name(), "Forward Euler");
        assert_eq!(euler.order(), 1);
        assert!(!euler.is_symplectic());
    }

    // ── Backward Euler tests ──

    #[test]
    fn test_backward_euler_exponential_decay() {
        let euler = BackwardEuler::new();
        let dt = 0.001;
        let mut state = vec![1.0];
        let mut t = 0.0;
        while t < 1.0 - 1e-14 {
            euler.step(&mut state, t, dt, &exp_decay_deriv);
            t += dt;
        }
        let exact = (-1.0_f64).exp();
        let error = (state[0] - exact).abs();
        assert!(error < 0.01, "Backward Euler error: {}", error);
    }

    #[test]
    fn test_backward_euler_convergence_order() {
        let euler = BackwardEuler::new();
        let t_end = 1.0_f64;
        let exact = (-t_end).exp();

        let dt1 = 0.01;
        let mut state1 = vec![1.0];
        let mut t = 0.0;
        while t < t_end - 1e-14 {
            euler.step(&mut state1, t, dt1, &exp_decay_deriv);
            t += dt1;
        }
        let err1 = (state1[0] - exact).abs();

        let dt2 = 0.005;
        let mut state2 = vec![1.0];
        t = 0.0;
        while t < t_end - 1e-14 {
            euler.step(&mut state2, t, dt2, &exp_decay_deriv);
            t += dt2;
        }
        let err2 = (state2[0] - exact).abs();

        let ratio = err1 / err2;
        assert!(
            ratio > 1.5 && ratio < 2.5,
            "Backward Euler ratio: {} (expected ~2.0)",
            ratio
        );
    }

    #[test]
    fn test_backward_euler_stability_stiff() {
        // For a stiff problem with large lambda, backward Euler should remain stable
        let lambda = 100.0;
        let euler = BackwardEuler::new();
        let dt = 0.1; // dt * lambda = 10 >> 1, unstable for forward Euler
        let mut state = vec![1.0];
        let mut t = 0.0;
        let deriv = stiff_deriv(lambda);
        while t < 1.0 - 1e-14 {
            euler.step(&mut state, t, dt, &deriv);
            t += dt;
        }
        // Solution should decay toward 0, not blow up
        assert!(
            state[0].abs() < 1.0,
            "Backward Euler should be stable: y = {}",
            state[0]
        );
        assert!(state[0] >= 0.0, "Should remain non-negative");
    }

    #[test]
    fn test_backward_euler_fixed_point() {
        let config = ImplicitConfig {
            max_iters: 100,
            tolerance: 1e-12,
            relaxation: 0.8,
        };
        let euler = BackwardEuler::with_fixed_point(config);
        let dt = 0.001;
        let mut state = vec![1.0];
        let mut t = 0.0;
        while t < 1.0 - 1e-14 {
            euler.step(&mut state, t, dt, &exp_decay_deriv);
            t += dt;
        }
        let exact = (-1.0_f64).exp();
        assert!((state[0] - exact).abs() < 0.01);
    }

    #[test]
    fn test_backward_euler_name_order() {
        let euler = BackwardEuler::new();
        assert_eq!(euler.name(), "Backward Euler");
        assert_eq!(euler.order(), 1);
    }

    // ── Improved Euler tests ──

    #[test]
    fn test_improved_euler_exponential_decay() {
        let heun = ImprovedEuler;
        let dt = 0.01;
        let mut state = vec![1.0];
        let mut t = 0.0;
        while t < 1.0 - 1e-14 {
            heun.step(&mut state, t, dt, &exp_decay_deriv);
            t += dt;
        }
        let exact = (-1.0_f64).exp();
        let error = (state[0] - exact).abs();
        assert!(error < 0.001, "Improved Euler error: {}", error);
    }

    #[test]
    fn test_improved_euler_convergence_order() {
        let heun = ImprovedEuler;
        let t_end = 1.0_f64;
        let exact = (-t_end).exp();

        let dt1 = 0.02;
        let mut state1 = vec![1.0];
        let mut t = 0.0;
        while t < t_end - 1e-14 {
            heun.step(&mut state1, t, dt1, &exp_decay_deriv);
            t += dt1;
        }
        let err1 = (state1[0] - exact).abs();

        let dt2 = 0.01;
        let mut state2 = vec![1.0];
        t = 0.0;
        while t < t_end - 1e-14 {
            heun.step(&mut state2, t, dt2, &exp_decay_deriv);
            t += dt2;
        }
        let err2 = (state2[0] - exact).abs();

        let ratio = err1 / err2;
        // Second-order: error ratio ≈ 4.0
        assert!(
            ratio > 3.5 && ratio < 4.5,
            "Improved Euler ratio: {} (expected ~4.0)",
            ratio
        );
    }

    #[test]
    fn test_improved_euler_exact_for_linear() {
        // Heun's method is exact for polynomials of degree <= 2
        let heun = ImprovedEuler;
        let dt = 0.1;
        let mut state = vec![0.0];
        let mut t = 0.0;
        while t < 1.0 - 1e-14 {
            heun.step(&mut state, t, dt, &linear_deriv);
            t += dt;
        }
        assert!(
            (state[0] - 1.0).abs() < 1e-10,
            "Heun should be very accurate for quadratic: err = {}",
            (state[0] - 1.0).abs()
        );
    }

    #[test]
    fn test_improved_euler_name_order() {
        let heun = ImprovedEuler;
        assert_eq!(heun.name(), "Improved Euler (Heun)");
        assert_eq!(heun.order(), 2);
        assert!(!heun.is_symplectic());
    }

    #[test]
    fn test_improved_euler_error_estimate() {
        let heun = ImprovedEuler;
        let state = vec![1.0];
        let err = heun.estimate_error(&state, 0.0, 0.01, &exp_decay_deriv);
        assert!(err[0] > 0.0);
        assert!(err[0] < 1e-5, "Error should be O(dt^3) ≈ 1e-6");
    }

    // ── 2D system test ──

    #[test]
    fn test_forward_euler_2d_system() {
        // y1' = y2, y2' = -y1 (harmonic oscillator)
        // y1(0) = 1, y2(0) = 0
        // exact: y1(t) = cos(t), y2(t) = -sin(t)
        let deriv = |_t: f64, y: &[f64], out: &mut [f64]| {
            out[0] = y[1];
            out[1] = -y[0];
        };

        let euler = ForwardEuler;
        let dt = 0.001;
        let mut state = vec![1.0, 0.0];
        let mut t = 0.0;
        while t < 1.0 - 1e-14 {
            euler.step(&mut state, t, dt, &deriv);
            t += dt;
        }
        let exact_y1 = 1.0_f64.cos();
        let exact_y2 = -(1.0_f64.sin());
        assert!(
            (state[0] - exact_y1).abs() < 0.01,
            "y1 error: {}",
            (state[0] - exact_y1).abs()
        );
        assert!(
            (state[1] - exact_y2).abs() < 0.01,
            "y2 error: {}",
            (state[1] - exact_y2).abs()
        );
    }

    #[test]
    fn test_improved_euler_2d_harmonic() {
        let deriv = |_t: f64, y: &[f64], out: &mut [f64]| {
            out[0] = y[1];
            out[1] = -y[0];
        };

        let heun = ImprovedEuler;
        let dt = 0.01;
        let mut state = vec![1.0, 0.0];
        let mut t = 0.0;
        while t < 2.0 * std::f64::consts::PI - 1e-14 {
            heun.step(&mut state, t, dt, &deriv);
            t += dt;
        }
        // After one full period, should return close to initial conditions
        assert!(
            (state[0] - 1.0).abs() < 0.01,
            "y1 after period: {}",
            state[0]
        );
        assert!(
            state[1].abs() < 0.01,
            "y2 after period: {}",
            state[1]
        );
    }

    #[test]
    fn test_linear_system_solver() {
        // Solve: 2x + y = 5, x + 3y = 7 => x = 8/5, y = 9/5
        let a = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let b = vec![5.0, 7.0];
        let x = solve_linear_system(&a, &b);
        assert!((x[0] - 1.6).abs() < 1e-10);
        assert!((x[1] - 1.8).abs() < 1e-10);
    }

    #[test]
    fn test_linear_system_solver_3x3() {
        // Identity system: Ix = b
        let a = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let b = vec![3.0, 7.0, 11.0];
        let x = solve_linear_system(&a, &b);
        assert!((x[0] - 3.0).abs() < 1e-10);
        assert!((x[1] - 7.0).abs() < 1e-10);
        assert!((x[2] - 11.0).abs() < 1e-10);
    }

    #[test]
    fn test_backward_euler_2d_harmonic() {
        let deriv = |_t: f64, y: &[f64], out: &mut [f64]| {
            out[0] = y[1];
            out[1] = -y[0];
        };
        let euler = BackwardEuler::new();
        let dt = 0.01;
        let mut state = vec![1.0, 0.0];
        let mut t = 0.0;
        while t < 1.0 - 1e-14 {
            euler.step(&mut state, t, dt, &deriv);
            t += dt;
        }
        let exact_y1 = 1.0_f64.cos();
        // Backward Euler is dissipative, but should be in the right ballpark
        assert!(
            (state[0] - exact_y1).abs() < 0.1,
            "Backward Euler 2D: y1 = {}, expected {}",
            state[0],
            exact_y1
        );
    }
}
