//! ODE integration methods for pharmacokinetic systems.
//!
//! Provides Euler, RK4, adaptive RK45 (Dormand-Prince), and interval-validated
//! integration methods for solving PK ODEs.

use serde::{Deserialize, Serialize};
use guardpharma_types::error::PkModelError;

// ---------------------------------------------------------------------------
// OdeSolver trait
// ---------------------------------------------------------------------------

/// Trait for ODE solvers.
pub trait OdeSolver: Send + Sync {
    fn solve(
        &self,
        f: &dyn Fn(f64, &[f64]) -> Vec<f64>,
        y0: &[f64],
        t_span: (f64, f64),
        dt: f64,
    ) -> Vec<(f64, Vec<f64>)>;

    fn name(&self) -> &str;
}

// ---------------------------------------------------------------------------
// EulerSolver
// ---------------------------------------------------------------------------

/// Forward Euler method: y_{n+1} = y_n + dt * f(t_n, y_n).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EulerSolver;

impl OdeSolver for EulerSolver {
    fn solve(
        &self,
        f: &dyn Fn(f64, &[f64]) -> Vec<f64>,
        y0: &[f64],
        t_span: (f64, f64),
        dt: f64,
    ) -> Vec<(f64, Vec<f64>)> {
        let (t0, tf) = t_span;
        let n = ((tf - t0) / dt).ceil() as usize;
        let mut results = Vec::with_capacity(n + 1);
        let mut t = t0;
        let mut y = y0.to_vec();
        results.push((t, y.clone()));

        for _ in 0..n {
            let h = dt.min(tf - t);
            if h <= 0.0 {
                break;
            }
            let dy = f(t, &y);
            for j in 0..y.len() {
                y[j] += h * dy[j];
            }
            t += h;
            results.push((t, y.clone()));
        }
        results
    }

    fn name(&self) -> &str {
        "Euler"
    }
}

// ---------------------------------------------------------------------------
// RungeKutta4Solver
// ---------------------------------------------------------------------------

/// Classic fourth-order Runge-Kutta method.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RungeKutta4Solver;

impl OdeSolver for RungeKutta4Solver {
    fn solve(
        &self,
        f: &dyn Fn(f64, &[f64]) -> Vec<f64>,
        y0: &[f64],
        t_span: (f64, f64),
        dt: f64,
    ) -> Vec<(f64, Vec<f64>)> {
        let (t0, tf) = t_span;
        let n = ((tf - t0) / dt).ceil() as usize;
        let dim = y0.len();
        let mut results = Vec::with_capacity(n + 1);
        let mut t = t0;
        let mut y = y0.to_vec();
        results.push((t, y.clone()));

        for _ in 0..n {
            let h = dt.min(tf - t);
            if h <= 0.0 {
                break;
            }

            let k1 = f(t, &y);

            let mut y2 = vec![0.0; dim];
            for j in 0..dim {
                y2[j] = y[j] + 0.5 * h * k1[j];
            }
            let k2 = f(t + 0.5 * h, &y2);

            let mut y3 = vec![0.0; dim];
            for j in 0..dim {
                y3[j] = y[j] + 0.5 * h * k2[j];
            }
            let k3 = f(t + 0.5 * h, &y3);

            let mut y4 = vec![0.0; dim];
            for j in 0..dim {
                y4[j] = y[j] + h * k3[j];
            }
            let k4 = f(t + h, &y4);

            for j in 0..dim {
                y[j] += h / 6.0 * (k1[j] + 2.0 * k2[j] + 2.0 * k3[j] + k4[j]);
            }
            t += h;
            results.push((t, y.clone()));
        }
        results
    }

    fn name(&self) -> &str {
        "RK4"
    }
}

// ---------------------------------------------------------------------------
// AdaptiveRK45Solver (Dormand-Prince)
// ---------------------------------------------------------------------------

/// Adaptive Runge-Kutta-Fehlberg 4(5) method (Dormand-Prince).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveRK45Solver {
    pub atol: f64,
    pub rtol: f64,
    pub max_step: f64,
    pub min_step: f64,
    pub safety_factor: f64,
}

impl Default for AdaptiveRK45Solver {
    fn default() -> Self {
        Self {
            atol: 1e-8,
            rtol: 1e-6,
            max_step: 1.0,
            min_step: 1e-12,
            safety_factor: 0.9,
        }
    }
}

impl AdaptiveRK45Solver {
    pub fn new(atol: f64, rtol: f64) -> Self {
        Self {
            atol,
            rtol,
            ..Default::default()
        }
    }

    pub fn with_step_bounds(mut self, min: f64, max: f64) -> Self {
        self.min_step = min;
        self.max_step = max;
        self
    }
}

impl OdeSolver for AdaptiveRK45Solver {
    fn solve(
        &self,
        f: &dyn Fn(f64, &[f64]) -> Vec<f64>,
        y0: &[f64],
        t_span: (f64, f64),
        dt: f64,
    ) -> Vec<(f64, Vec<f64>)> {
        let (t0, tf) = t_span;
        let dim = y0.len();
        let mut results = Vec::new();
        let mut t = t0;
        let mut y = y0.to_vec();
        let mut h = dt.min(self.max_step);
        results.push((t, y.clone()));

        // Dormand-Prince coefficients (Butcher tableau)
        let a21 = 1.0 / 5.0;
        let a31 = 3.0 / 40.0;
        let a32 = 9.0 / 40.0;
        let a41 = 44.0 / 45.0;
        let a42 = -56.0 / 15.0;
        let a43 = 32.0 / 9.0;
        let a51 = 19372.0 / 6561.0;
        let a52 = -25360.0 / 2187.0;
        let a53 = 64448.0 / 6561.0;
        let a54 = -212.0 / 729.0;
        let a61 = 9017.0 / 3168.0;
        let a62 = -355.0 / 33.0;
        let a63 = 46732.0 / 5247.0;
        let a64 = 49.0 / 176.0;
        let a65 = -5103.0 / 18656.0;

        // 5th order coefficients
        let b1 = 35.0 / 384.0;
        let b3 = 500.0 / 1113.0;
        let b4 = 125.0 / 192.0;
        let b5 = -2187.0 / 6784.0;
        let b6 = 11.0 / 84.0;

        // Error coefficients (difference between 4th and 5th order)
        let e1 = 71.0 / 57600.0;
        let e3 = -71.0 / 16695.0;
        let e4 = 71.0 / 1920.0;
        let e5 = -17253.0 / 339200.0;
        let e6 = 22.0 / 525.0;
        let e7 = -1.0 / 40.0;

        let max_iter = 1_000_000;
        let mut iter = 0;

        while t < tf - 1e-15 && iter < max_iter {
            iter += 1;
            h = h.min(tf - t).max(self.min_step);

            let k1 = f(t, &y);

            let mut yt = vec![0.0; dim];
            for j in 0..dim {
                yt[j] = y[j] + h * a21 * k1[j];
            }
            let k2 = f(t + h / 5.0, &yt);

            for j in 0..dim {
                yt[j] = y[j] + h * (a31 * k1[j] + a32 * k2[j]);
            }
            let k3 = f(t + 3.0 * h / 10.0, &yt);

            for j in 0..dim {
                yt[j] = y[j] + h * (a41 * k1[j] + a42 * k2[j] + a43 * k3[j]);
            }
            let k4 = f(t + 4.0 * h / 5.0, &yt);

            for j in 0..dim {
                yt[j] = y[j] + h * (a51 * k1[j] + a52 * k2[j] + a53 * k3[j] + a54 * k4[j]);
            }
            let k5 = f(t + 8.0 * h / 9.0, &yt);

            for j in 0..dim {
                yt[j] = y[j]
                    + h * (a61 * k1[j] + a62 * k2[j] + a63 * k3[j] + a64 * k4[j]
                        + a65 * k5[j]);
            }
            let k6 = f(t + h, &yt);

            // 5th order solution
            let mut y_new = vec![0.0; dim];
            for j in 0..dim {
                y_new[j] =
                    y[j] + h * (b1 * k1[j] + b3 * k3[j] + b4 * k4[j] + b5 * k5[j] + b6 * k6[j]);
            }

            // Error estimate (need k7 for DP)
            let k7 = f(t + h, &y_new);
            let mut err = 0.0;
            for j in 0..dim {
                let e_j = h
                    * (e1 * k1[j] + e3 * k3[j] + e4 * k4[j] + e5 * k5[j] + e6 * k6[j]
                        + e7 * k7[j]);
                let sc = self.atol + self.rtol * y[j].abs().max(y_new[j].abs());
                err += (e_j / sc).powi(2);
            }
            err = (err / dim as f64).sqrt();

            if err <= 1.0 || h <= self.min_step {
                // Accept step
                t += h;
                y = y_new;
                results.push((t, y.clone()));
            }

            // Adjust step size
            if err > 1e-15 {
                let factor = self.safety_factor * (1.0 / err).powf(0.2);
                h *= factor.max(0.2).min(5.0);
            } else {
                h *= 2.0;
            }
            h = h.max(self.min_step).min(self.max_step);
        }

        results
    }

    fn name(&self) -> &str {
        "AdaptiveRK45"
    }
}

// ---------------------------------------------------------------------------
// Interval arithmetic types
// ---------------------------------------------------------------------------

/// A closed interval [lo, hi].
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Interval {
    pub lo: f64,
    pub hi: f64,
}

impl Interval {
    pub fn new(lo: f64, hi: f64) -> Self {
        debug_assert!(lo <= hi + 1e-15, "Interval: lo={} > hi={}", lo, hi);
        Self { lo, hi }
    }

    pub fn point(v: f64) -> Self {
        Self { lo: v, hi: v }
    }

    pub fn width(&self) -> f64 {
        self.hi - self.lo
    }

    pub fn midpoint(&self) -> f64 {
        (self.lo + self.hi) / 2.0
    }

    pub fn contains(&self, v: f64) -> bool {
        v >= self.lo - 1e-15 && v <= self.hi + 1e-15
    }

    pub fn add(self, other: Self) -> Self {
        Self::new(self.lo + other.lo, self.hi + other.hi)
    }

    pub fn sub(self, other: Self) -> Self {
        Self::new(self.lo - other.hi, self.hi - other.lo)
    }

    pub fn mul(self, other: Self) -> Self {
        let products = [
            self.lo * other.lo,
            self.lo * other.hi,
            self.hi * other.lo,
            self.hi * other.hi,
        ];
        let lo = products.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = products.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        Self::new(lo, hi)
    }

    pub fn div(self, other: Self) -> Option<Self> {
        if other.lo <= 0.0 && other.hi >= 0.0 {
            return None; // Division by interval containing zero
        }
        let inv = Self::new(1.0 / other.hi, 1.0 / other.lo);
        Some(self.mul(inv))
    }

    pub fn scale(self, s: f64) -> Self {
        if s >= 0.0 {
            Self::new(self.lo * s, self.hi * s)
        } else {
            Self::new(self.hi * s, self.lo * s)
        }
    }

    pub fn union(self, other: Self) -> Self {
        Self::new(self.lo.min(other.lo), self.hi.max(other.hi))
    }

    pub fn intersection(self, other: Self) -> Option<Self> {
        let lo = self.lo.max(other.lo);
        let hi = self.hi.min(other.hi);
        if lo <= hi { Some(Self::new(lo, hi)) } else { None }
    }
}

/// Vector of intervals.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntervalVector(pub Vec<Interval>);

impl IntervalVector {
    pub fn new(intervals: Vec<Interval>) -> Self {
        Self(intervals)
    }

    pub fn dim(&self) -> usize {
        self.0.len()
    }

    pub fn from_point(v: &[f64]) -> Self {
        Self(v.iter().map(|&x| Interval::point(x)).collect())
    }

    pub fn midpoint(&self) -> Vec<f64> {
        self.0.iter().map(|i| i.midpoint()).collect()
    }

    pub fn width(&self) -> f64 {
        self.0.iter().map(|i| i.width()).fold(0.0_f64, f64::max)
    }

    pub fn add(&self, other: &Self) -> Self {
        Self(
            self.0
                .iter()
                .zip(other.0.iter())
                .map(|(a, b)| a.add(*b))
                .collect(),
        )
    }

    pub fn scale(&self, s: f64) -> Self {
        Self(self.0.iter().map(|i| i.scale(s)).collect())
    }

    pub fn contains(&self, v: &[f64]) -> bool {
        self.0.iter().zip(v.iter()).all(|(i, &x)| i.contains(x))
    }
}

/// Matrix of intervals.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntervalMatrix {
    pub data: Vec<Vec<Interval>>,
    pub rows: usize,
    pub cols: usize,
}

impl IntervalMatrix {
    pub fn new(data: Vec<Vec<Interval>>) -> Self {
        let rows = data.len();
        let cols = if rows > 0 { data[0].len() } else { 0 };
        Self { data, rows, cols }
    }

    pub fn mul_vector(&self, v: &IntervalVector) -> IntervalVector {
        let mut result = Vec::with_capacity(self.rows);
        for i in 0..self.rows {
            let mut sum = Interval::point(0.0);
            for j in 0..self.cols {
                sum = sum.add(self.data[i][j].mul(v.0[j]));
            }
            result.push(sum);
        }
        IntervalVector(result)
    }
}

// ---------------------------------------------------------------------------
// IntervalOdeSolver
// ---------------------------------------------------------------------------

/// Interval method selection.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum IntervalMethod {
    EulerInterval,
    TaylorInterval,
}

/// Interval-based ODE solver for validated numerics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntervalOdeSolver {
    pub dt: f64,
    pub method: IntervalMethod,
}

impl IntervalOdeSolver {
    pub fn new(dt: f64, method: IntervalMethod) -> Self {
        Self { dt, method }
    }

    /// Solve ODE with interval initial conditions using Euler enclosure.
    pub fn solve_interval(
        &self,
        f: &dyn Fn(f64, &[f64]) -> Vec<f64>,
        y0: &IntervalVector,
        t_span: (f64, f64),
    ) -> Vec<(f64, IntervalVector)> {
        let (t0, tf) = t_span;
        let n = ((tf - t0) / self.dt).ceil() as usize;
        let dim = y0.dim();
        let mut results = vec![(t0, y0.clone())];
        let mut current = y0.clone();
        let mut t = t0;

        for _ in 0..n {
            let h = self.dt.min(tf - t);
            if h <= 0.0 {
                break;
            }

            // Evaluate f at midpoint and bounds
            let mid = current.midpoint();
            let f_mid = f(t, &mid);

            // Evaluate at lo and hi corners for enclosure
            let lo: Vec<f64> = current.0.iter().map(|i| i.lo).collect();
            let hi: Vec<f64> = current.0.iter().map(|i| i.hi).collect();
            let f_lo = f(t, &lo);
            let f_hi = f(t, &hi);

            // Enclose the derivative
            let mut new_intervals = Vec::with_capacity(dim);
            for j in 0..dim {
                let df_lo = f_lo[j].min(f_hi[j]).min(f_mid[j]);
                let df_hi = f_lo[j].max(f_hi[j]).max(f_mid[j]);
                let new_lo = current.0[j].lo + h * df_lo;
                let new_hi = current.0[j].hi + h * df_hi;
                new_intervals.push(Interval::new(new_lo, new_hi));
            }

            current = IntervalVector(new_intervals);
            t += h;
            results.push((t, current.clone()));
        }

        results
    }
}

// ---------------------------------------------------------------------------
// Solver type enum
// ---------------------------------------------------------------------------

/// Solver type selection.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SolverType {
    Euler,
    RK4,
    AdaptiveRK45,
}

// ---------------------------------------------------------------------------
// PkOdeSystem
// ---------------------------------------------------------------------------

/// PK ODE system: dc/dt = M*c + b(t) with pulsed dosing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PkOdeSystem {
    pub rate_matrix: Vec<Vec<f64>>,
    pub input_rates: Vec<f64>,
    pub doses: Vec<f64>,
    pub dosing_intervals: Vec<f64>,
    pub bioavailabilities: Vec<f64>,
    pub absorption_rates: Vec<f64>,
    pub n_compartments: usize,
}

impl PkOdeSystem {
    pub fn new(n: usize) -> Self {
        Self {
            rate_matrix: vec![vec![0.0; n]; n],
            input_rates: vec![0.0; n],
            doses: vec![0.0; n],
            dosing_intervals: vec![24.0; n],
            bioavailabilities: vec![1.0; n],
            absorption_rates: vec![0.0; n],
            n_compartments: n,
        }
    }

    /// Check if we're at a dosing time for a compartment.
    pub fn is_dosing_time(&self, t: f64, compartment: usize) -> bool {
        let tau = self.dosing_intervals[compartment];
        if tau <= 0.0 {
            return false;
        }
        let remainder = t % tau;
        remainder < 0.1 || (tau - remainder) < 0.1
    }

    /// Compute dosing input vector at time t.
    pub fn dosing_input(&self, t: f64) -> Vec<f64> {
        let mut input = self.input_rates.clone();
        for i in 0..self.n_compartments {
            if self.doses[i] > 0.0 && self.dosing_intervals[i] > 0.0 {
                // Continuous approximation: spread dose over small interval
                let tau = self.dosing_intervals[i];
                let fi = self.bioavailabilities[i];
                input[i] += fi * self.doses[i] / tau;
            }
        }
        input
    }

    /// Right-hand side: dc/dt = M*c + b(t).
    pub fn rhs(&self, t: f64, state: &[f64]) -> Vec<f64> {
        let n = self.n_compartments;
        let input = self.dosing_input(t);
        let mut deriv = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                deriv[i] += self.rate_matrix[i][j] * state[j];
            }
            deriv[i] += input[i];
        }
        deriv
    }
}

// ---------------------------------------------------------------------------
// Trajectory
// ---------------------------------------------------------------------------

/// Solution trajectory storing time-state pairs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trajectory {
    pub times: Vec<f64>,
    pub states: Vec<Vec<f64>>,
    pub n_compartments: usize,
}

impl Trajectory {
    pub fn new(n_compartments: usize) -> Self {
        Self {
            times: Vec::new(),
            states: Vec::new(),
            n_compartments,
        }
    }

    pub fn push(&mut self, t: f64, state: Vec<f64>) {
        self.times.push(t);
        self.states.push(state);
    }

    pub fn len(&self) -> usize {
        self.times.len()
    }

    pub fn is_empty(&self) -> bool {
        self.times.is_empty()
    }

    pub fn get_compartment(&self, idx: usize) -> Vec<f64> {
        self.states.iter().map(|s| s[idx]).collect()
    }

    pub fn final_state(&self) -> &[f64] {
        self.states.last().map(|s| s.as_slice()).unwrap_or(&[])
    }

    pub fn peak(&self, compartment: usize) -> (f64, f64) {
        let mut best = (0.0, 0.0);
        for (i, s) in self.states.iter().enumerate() {
            if s[compartment] > best.1 {
                best = (self.times[i], s[compartment]);
            }
        }
        best
    }

    pub fn trough(&self, compartment: usize) -> (f64, f64) {
        // Skip initial transient (first 10% of trajectory)
        let skip = self.states.len() / 10;
        let mut best = (0.0, f64::INFINITY);
        for i in skip..self.states.len() {
            if self.states[i][compartment] < best.1 {
                best = (self.times[i], self.states[i][compartment]);
            }
        }
        if best.1 == f64::INFINITY {
            best.1 = 0.0;
        }
        best
    }

    /// AUC using trapezoidal rule for one compartment.
    pub fn auc(&self, compartment: usize) -> f64 {
        let concs = self.get_compartment(compartment);
        let mut auc = 0.0;
        for i in 1..self.times.len() {
            let dt = self.times[i] - self.times[i - 1];
            auc += 0.5 * dt * (concs[i - 1] + concs[i]);
        }
        auc
    }

    /// Build from solver output.
    pub fn from_solver_output(
        data: Vec<(f64, Vec<f64>)>,
        n_compartments: usize,
    ) -> Self {
        let mut traj = Self::new(n_compartments);
        for (t, s) in data {
            traj.push(t, s);
        }
        traj
    }
}

// ---------------------------------------------------------------------------
// Convenience functions
// ---------------------------------------------------------------------------

/// Solve a PK system using the specified solver.
pub fn solve_pk_system(
    system: &PkOdeSystem,
    t_end: f64,
    dt: f64,
    solver_type: SolverType,
) -> Trajectory {
    let y0 = vec![0.0; system.n_compartments];
    let rhs = |t: f64, y: &[f64]| system.rhs(t, y);

    let data = match solver_type {
        SolverType::Euler => EulerSolver.solve(&rhs, &y0, (0.0, t_end), dt),
        SolverType::RK4 => RungeKutta4Solver.solve(&rhs, &y0, (0.0, t_end), dt),
        SolverType::AdaptiveRK45 => {
            AdaptiveRK45Solver::new(1e-8, 1e-6).solve(&rhs, &y0, (0.0, t_end), dt)
        }
    };

    Trajectory::from_solver_output(data, system.n_compartments)
}

/// Solve until steady state is reached.
pub fn solve_to_steady_state(
    system: &PkOdeSystem,
    tolerance: f64,
    max_intervals: usize,
    solver_type: SolverType,
) -> Result<Trajectory, PkModelError> {
    let tau = system
        .dosing_intervals
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    if tau <= 0.0 || tau == f64::INFINITY {
        return Err(PkModelError::InvalidCompartmentModel(
            "Invalid dosing interval".into(),
        ));
    }

    let dt = tau / 100.0;
    let n = system.n_compartments;
    let mut y = vec![0.0; n];
    let mut traj = Trajectory::new(n);
    let mut t = 0.0;
    traj.push(t, y.clone());

    for interval in 0..max_intervals {
        // Solve one dosing interval
        let rhs = |t_: f64, y_: &[f64]| system.rhs(t_, y_);
        let data = match solver_type {
            SolverType::Euler => EulerSolver.solve(&rhs, &y, (t, t + tau), dt),
            SolverType::RK4 => RungeKutta4Solver.solve(&rhs, &y, (t, t + tau), dt),
            SolverType::AdaptiveRK45 => {
                AdaptiveRK45Solver::new(1e-8, 1e-6).solve(&rhs, &y, (t, t + tau), dt)
            }
        };

        for (ti, si) in &data[1..] {
            traj.push(*ti, si.clone());
        }

        let y_new = data.last().map(|(_, s)| s.clone()).unwrap_or_else(|| y.clone());

        // Check convergence
        if interval > 0 {
            let max_diff: f64 = y
                .iter()
                .zip(y_new.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            let max_val: f64 = y_new.iter().cloned().fold(1e-15_f64, f64::max);
            if max_diff / max_val < tolerance {
                return Ok(traj);
            }
        }

        y = y_new;
        t += tau;
    }

    Err(PkModelError::SteadyStateNotReached {
        iterations: max_intervals,
        residual: 0.0,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euler_exponential_decay() {
        let sol = EulerSolver.solve(
            &|_t, y| vec![-y[0]],
            &[1.0],
            (0.0, 3.0),
            0.001,
        );
        let last = sol.last().unwrap();
        let expected = (-3.0_f64).exp();
        assert!((last.1[0] - expected).abs() < 0.01);
    }

    #[test]
    fn test_rk4_exponential_decay() {
        let sol = RungeKutta4Solver.solve(
            &|_t, y| vec![-y[0]],
            &[1.0],
            (0.0, 3.0),
            0.01,
        );
        let last = sol.last().unwrap();
        let expected = (-3.0_f64).exp();
        assert!((last.1[0] - expected).abs() < 1e-6);
    }

    #[test]
    fn test_rk4_harmonic_oscillator() {
        // dx/dt = v, dv/dt = -x
        let sol = RungeKutta4Solver.solve(
            &|_t, y| vec![y[1], -y[0]],
            &[1.0, 0.0],
            (0.0, 6.28),
            0.01,
        );
        let last = sol.last().unwrap();
        // Should return near (1, 0) after one period
        assert!((last.1[0] - 1.0).abs() < 0.01);
        assert!(last.1[1].abs() < 0.01);
    }

    #[test]
    fn test_adaptive_rk45_accuracy() {
        let sol = AdaptiveRK45Solver::new(1e-10, 1e-8).solve(
            &|_t, y| vec![-y[0]],
            &[1.0],
            (0.0, 5.0),
            0.1,
        );
        let last = sol.last().unwrap();
        let expected = (-5.0_f64).exp();
        assert!((last.1[0] - expected).abs() < 1e-6);
    }

    #[test]
    fn test_interval_arithmetic() {
        let a = Interval::new(1.0, 3.0);
        let b = Interval::new(2.0, 4.0);
        let sum = a.add(b);
        assert!((sum.lo - 3.0).abs() < 1e-10);
        assert!((sum.hi - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_interval_mul() {
        let a = Interval::new(-2.0, 3.0);
        let b = Interval::new(-1.0, 4.0);
        let prod = a.mul(b);
        assert!(prod.lo <= -8.0 + 1e-10);
        assert!(prod.hi >= 12.0 - 1e-10);
    }

    #[test]
    fn test_interval_ode_enclosure() {
        let solver = IntervalOdeSolver::new(0.01, IntervalMethod::EulerInterval);
        let y0 = IntervalVector::new(vec![Interval::new(0.9, 1.1)]);
        let results = solver.solve_interval(
            &|_t, y| vec![-y[0]],
            &y0,
            (0.0, 1.0),
        );
        let last = &results.last().unwrap().1;
        let exact = (-1.0_f64).exp();
        assert!(last.0[0].contains(exact));
    }

    #[test]
    fn test_pk_ode_system() {
        let mut sys = PkOdeSystem::new(1);
        sys.rate_matrix[0][0] = -0.2; // ke = 0.2 h^-1
        sys.doses[0] = 100.0;
        sys.dosing_intervals[0] = 12.0;
        sys.bioavailabilities[0] = 1.0;

        let traj = solve_pk_system(&sys, 120.0, 0.1, SolverType::RK4);
        assert!(traj.len() > 100);
        let final_conc = traj.final_state()[0];
        assert!(final_conc > 0.0);
    }

    #[test]
    fn test_trajectory_auc() {
        let mut traj = Trajectory::new(1);
        traj.push(0.0, vec![0.0]);
        traj.push(1.0, vec![2.0]);
        traj.push(2.0, vec![0.0]);
        let auc = traj.auc(0);
        assert!((auc - 2.0).abs() < 0.01); // triangle area = 2
    }

    #[test]
    fn test_trajectory_peak_trough() {
        let mut traj = Trajectory::new(1);
        for i in 0..100 {
            let t = i as f64 * 0.1;
            let c = (t * 0.5).sin().abs();
            traj.push(t, vec![c]);
        }
        let (_, peak_val) = traj.peak(0);
        assert!(peak_val > 0.0);
    }

    #[test]
    fn test_interval_vector_contains() {
        let iv = IntervalVector::new(vec![
            Interval::new(0.0, 2.0),
            Interval::new(1.0, 3.0),
        ]);
        assert!(iv.contains(&[1.0, 2.0]));
        assert!(!iv.contains(&[3.0, 2.0]));
    }
}
