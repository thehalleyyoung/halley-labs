//! sim-repair: Projection-based repair to restore conservation properties.
//!
//! This crate provides a comprehensive suite of repair strategies that can
//! restore conservation quantities (energy, momentum, angular momentum, etc.)
//! to their target values after numerical drift or integration errors.
//!
//! # Architecture
//!
//! The central abstraction is the [`RepairStrategy`] trait. Each module provides
//! one or more implementations:
//!
//! - [`projection`]: Orthogonal projection onto constraint manifolds
//! - [`manifold`]: SHAKE/RATTLE constraint satisfaction algorithms
//! - [`energy_repair`]: Velocity-scaling and thermostat methods for energy
//! - [`momentum_repair`]: COM correction and momentum redistribution
//! - [`symmetric`]: Time-reversible repair strategies
//! - [`iterative`]: Newton-Raphson, Broyden, BFGS solvers
//! - [`constraint`]: Constraint definitions and evaluation
//! - [`optimizer`]: Minimum-perturbation optimisation-based repair
//! - [`selective`]: Particle-selective and mass-weighted repair
//! - [`pipeline`]: Chaining multiple repair strategies

pub mod projection;
pub mod manifold;
pub mod energy_repair;
pub mod momentum_repair;
pub mod symmetric;
pub mod iterative;
pub mod constraint;
pub mod optimizer;
pub mod selective;
pub mod pipeline;

use serde::{Deserialize, Serialize};
use sim_types::{ConservationKind, SimulationState, Vec3};

// ─── Core types ───────────────────────────────────────────────────────────────

/// Result of a single repair operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepairResult {
    /// Whether the repair converged to the requested tolerance.
    pub success: bool,
    /// Number of iterations consumed (0 for direct methods).
    pub iterations: usize,
    /// Final residual (constraint violation norm).
    pub residual: f64,
    /// Norm of the total state change applied.
    pub state_change_norm: f64,
}

impl RepairResult {
    pub fn success(iterations: usize, residual: f64, state_change_norm: f64) -> Self {
        Self { success: true, iterations, residual, state_change_norm }
    }

    pub fn failure(iterations: usize, residual: f64, state_change_norm: f64) -> Self {
        Self { success: false, iterations, residual, state_change_norm }
    }
}

/// A target value for a conserved quantity.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ConservationTarget {
    pub kind: ConservationKind,
    pub value: f64,
}

impl ConservationTarget {
    pub fn new(kind: ConservationKind, value: f64) -> Self {
        Self { kind, value }
    }
}

/// Central trait for all repair strategies.
pub trait RepairStrategy {
    /// Attempt to repair `state` so that each conservation quantity listed in
    /// `targets` reaches its specified value.
    fn repair(
        &self,
        state: &mut SimulationState,
        targets: &[ConservationTarget],
    ) -> RepairResult;

    /// Human-readable name for logging / diagnostics.
    fn name(&self) -> &str;
}

// ─── Error types ──────────────────────────────────────────────────────────────

/// Errors that can occur during repair.
#[derive(Debug, Clone, thiserror::Error)]
pub enum RepairError {
    #[error("singular Jacobian encountered during projection")]
    SingularJacobian,
    #[error("repair did not converge within {max_iter} iterations (residual={residual:.3e})")]
    DidNotConverge { max_iter: usize, residual: f64 },
    #[error("no particles in state")]
    EmptyState,
    #[error("constraint index {0} out of range")]
    ConstraintIndexOutOfRange(usize),
    #[error("incompatible constraint dimensions: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    #[error("repair budget exceeded: perturbation {perturbation:.6e} > budget {budget:.6e}")]
    BudgetExceeded { perturbation: f64, budget: f64 },
    #[error("line search failed after {steps} steps")]
    LineSearchFailed { steps: usize },
    #[error("{0}")]
    Other(String),
}

// ─── Configuration ────────────────────────────────────────────────────────────

/// Shared configuration for iterative repair methods.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct RepairConfig {
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Absolute tolerance on the constraint residual.
    pub tolerance: f64,
    /// Finite-difference step for numerical Jacobians.
    pub fd_step: f64,
    /// Damping factor for Newton-type methods (1.0 = full step).
    pub damping: f64,
}

impl Default for RepairConfig {
    fn default() -> Self {
        Self {
            max_iter: 200,
            tolerance: 1e-12,
            fd_step: 1e-8,
            damping: 1.0,
        }
    }
}

// ─── Helper functions (used across modules) ───────────────────────────────────

/// Compute total kinetic energy of a state.
pub fn kinetic_energy(state: &SimulationState) -> f64 {
    state.particles.iter().map(|p| p.kinetic_energy()).sum()
}

/// Compute total linear momentum of a state.
pub fn total_momentum(state: &SimulationState) -> Vec3 {
    state.particles.iter().fold(Vec3::ZERO, |acc, p| acc + p.momentum())
}

/// Compute total angular momentum about the origin.
pub fn total_angular_momentum(state: &SimulationState) -> Vec3 {
    state.particles.iter().fold(Vec3::ZERO, |acc, p| acc + p.angular_momentum())
}

/// Compute the total mass of all particles.
pub fn total_mass(state: &SimulationState) -> f64 {
    state.particles.iter().map(|p| p.mass).sum()
}

/// Compute the total charge of all particles.
pub fn total_charge(state: &SimulationState) -> f64 {
    state.particles.iter().map(|p| p.charge).sum()
}

/// Evaluate a specific conservation quantity for the given state.
pub fn evaluate_conservation(state: &SimulationState, kind: ConservationKind) -> f64 {
    match kind {
        ConservationKind::Energy => kinetic_energy(state),
        ConservationKind::Momentum => {
            let p = total_momentum(state);
            p.magnitude()
        }
        ConservationKind::AngularMomentum => {
            let l = total_angular_momentum(state);
            l.magnitude()
        }
        ConservationKind::Mass => total_mass(state),
        ConservationKind::Charge => total_charge(state),
        ConservationKind::Custom => 0.0,
        ConservationKind::Symplectic => 0.0,
        ConservationKind::Vorticity => 0.0,
    }
}

/// Evaluate the vector-valued conservation quantity (for momentum / angular momentum).
pub fn evaluate_conservation_vec(state: &SimulationState, kind: ConservationKind) -> Vec3 {
    match kind {
        ConservationKind::Momentum => total_momentum(state),
        ConservationKind::AngularMomentum => total_angular_momentum(state),
        _ => Vec3::ZERO,
    }
}

/// Flatten a simulation state's velocities into a contiguous Vec<f64>.
pub fn flatten_velocities(state: &SimulationState) -> Vec<f64> {
    let mut v = Vec::with_capacity(state.particles.len() * 3);
    for p in &state.particles {
        v.push(p.velocity.x);
        v.push(p.velocity.y);
        v.push(p.velocity.z);
    }
    v
}

/// Write flattened velocities back into a simulation state.
pub fn unflatten_velocities(state: &mut SimulationState, v: &[f64]) {
    assert_eq!(v.len(), state.particles.len() * 3);
    for (i, p) in state.particles.iter_mut().enumerate() {
        p.velocity.x = v[i * 3];
        p.velocity.y = v[i * 3 + 1];
        p.velocity.z = v[i * 3 + 2];
    }
}

/// Flatten a simulation state's positions into a contiguous Vec<f64>.
pub fn flatten_positions(state: &SimulationState) -> Vec<f64> {
    let mut v = Vec::with_capacity(state.particles.len() * 3);
    for p in &state.particles {
        v.push(p.position.x);
        v.push(p.position.y);
        v.push(p.position.z);
    }
    v
}

/// Write flattened positions back into a simulation state.
pub fn unflatten_positions(state: &mut SimulationState, v: &[f64]) {
    assert_eq!(v.len(), state.particles.len() * 3);
    for (i, p) in state.particles.iter_mut().enumerate() {
        p.position.x = v[i * 3];
        p.position.y = v[i * 3 + 1];
        p.position.z = v[i * 3 + 2];
    }
}

/// Compute the L2 norm of a slice.
pub fn vec_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Compute the dot product of two slices.
pub fn vec_dot(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Add two vectors: result = a + b.
pub fn vec_add(a: &[f64], b: &[f64]) -> Vec<f64> {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

/// Subtract two vectors: result = a - b.
pub fn vec_sub(a: &[f64], b: &[f64]) -> Vec<f64> {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

/// Scale a vector: result = alpha * a.
pub fn vec_scale(a: &[f64], alpha: f64) -> Vec<f64> {
    a.iter().map(|x| x * alpha).collect()
}

/// Add scaled vector in place: a += alpha * b.
pub fn vec_axpy(a: &mut [f64], alpha: f64, b: &[f64]) {
    assert_eq!(a.len(), b.len());
    for (ai, bi) in a.iter_mut().zip(b.iter()) {
        *ai += alpha * *bi;
    }
}

/// Dense matrix-vector product y = A * x  (A stored row-major, m×n).
pub fn mat_vec_mul(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    a.iter().map(|row| vec_dot(row, x)).collect()
}

/// Dense matrix-transpose-vector product y = A^T * x.
pub fn mat_t_vec_mul(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    if a.is_empty() {
        return vec![];
    }
    let n = a[0].len();
    let m = a.len();
    assert_eq!(x.len(), m);
    let mut y = vec![0.0; n];
    for (i, row) in a.iter().enumerate() {
        for (j, val) in row.iter().enumerate() {
            y[j] += val * x[i];
        }
    }
    y
}

/// Solve a small dense linear system A x = b via LU decomposition with
/// partial pivoting. Returns None if the matrix is singular.
pub fn solve_dense(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = a.len();
    if n == 0 {
        return Some(vec![]);
    }
    assert_eq!(b.len(), n);
    for row in a {
        assert_eq!(row.len(), n);
    }

    // Build augmented matrix
    let mut aug: Vec<Vec<f64>> = a
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let mut r = row.clone();
            r.push(b[i]);
            r
        })
        .collect();

    let mut pivot_rows: Vec<usize> = (0..n).collect();

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_val = aug[col][col].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            let v = aug[row][col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_val < 1e-30 {
            return None; // singular
        }
        if max_row != col {
            aug.swap(col, max_row);
            pivot_rows.swap(col, max_row);
        }
        let diag = aug[col][col];
        for row in (col + 1)..n {
            let factor = aug[row][col] / diag;
            for j in col..=n {
                let val = aug[col][j];
                aug[row][j] -= factor * val;
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
        if aug[i][i].abs() < 1e-30 {
            return None;
        }
        x[i] = sum / aug[i][i];
    }
    Some(x)
}

/// Solve A x = b where A is symmetric positive-definite via Cholesky
/// factorization. Returns None if A is not SPD.
pub fn solve_cholesky(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = a.len();
    if n == 0 {
        return Some(vec![]);
    }

    // Cholesky: A = L L^T
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i][j];
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }
            if i == j {
                if sum <= 0.0 {
                    return None; // not positive definite
                }
                l[i][j] = sum.sqrt();
            } else {
                if l[j][j].abs() < 1e-30 {
                    return None;
                }
                l[i][j] = sum / l[j][j];
            }
        }
    }

    // Forward solve L y = b
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut sum = b[i];
        for j in 0..i {
            sum -= l[i][j] * y[j];
        }
        y[i] = sum / l[i][i];
    }

    // Back solve L^T x = y
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = y[i];
        for j in (i + 1)..n {
            sum -= l[j][i] * x[j];
        }
        x[i] = sum / l[i][i];
    }
    Some(x)
}

/// Outer product of two vectors: result[i][j] = a[i] * b[j].
pub fn vec_outer(a: &[f64], b: &[f64]) -> Vec<Vec<f64>> {
    a.iter().map(|&ai| b.iter().map(|&bj| ai * bj).collect()).collect()
}

/// Compute m×m product A * A^T given row-major A (m rows, n cols each).
pub fn mat_aat(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = a.len();
    let mut result = vec![vec![0.0; m]; m];
    for i in 0..m {
        for j in 0..=i {
            let dot: f64 = a[i].iter().zip(a[j].iter()).map(|(x, y)| x * y).sum();
            result[i][j] = dot;
            result[j][i] = dot;
        }
    }
    result
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use sim_types::Particle;

    const EPS: f64 = 1e-10;

    fn make_two_body() -> SimulationState {
        let p1 = Particle::new(1.0, Vec3::new(-1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0));
        let p2 = Particle::new(1.0, Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, -1.0, 0.0));
        SimulationState::new(vec![p1, p2], 0.0)
    }

    #[test]
    fn test_kinetic_energy() {
        let s = make_two_body();
        assert!((kinetic_energy(&s) - 1.0).abs() < EPS);
    }

    #[test]
    fn test_total_momentum_zero() {
        let s = make_two_body();
        let p = total_momentum(&s);
        assert!(p.magnitude() < EPS);
    }

    #[test]
    fn test_angular_momentum() {
        let s = make_two_body();
        let l = total_angular_momentum(&s);
        // L = (-1,0,0)x(0,1,0) + (1,0,0)x(0,-1,0) = (0,0,-1) + (0,0,-1) = (0,0,-2)
        assert!((l.z - (-2.0)).abs() < EPS);
    }

    #[test]
    fn test_flatten_unflatten_velocities() {
        let mut s = make_two_body();
        let v = flatten_velocities(&s);
        assert_eq!(v.len(), 6);
        assert!((v[1] - 1.0).abs() < EPS);
        assert!((v[4] - (-1.0)).abs() < EPS);
        let v2 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        unflatten_velocities(&mut s, &v2);
        assert!((s.particles[0].velocity.x - 1.0).abs() < EPS);
        assert!((s.particles[1].velocity.z - 6.0).abs() < EPS);
    }

    #[test]
    fn test_flatten_unflatten_positions() {
        let mut s = make_two_body();
        let p = flatten_positions(&s);
        assert_eq!(p.len(), 6);
        assert!((p[0] - (-1.0)).abs() < EPS);
        assert!((p[3] - 1.0).abs() < EPS);
        let p2 = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        unflatten_positions(&mut s, &p2);
        assert!((s.particles[0].position.x - 10.0).abs() < EPS);
        assert!((s.particles[1].position.z - 60.0).abs() < EPS);
    }

    #[test]
    fn test_vec_norm() {
        assert!((vec_norm(&[3.0, 4.0]) - 5.0).abs() < EPS);
    }

    #[test]
    fn test_vec_dot_product() {
        assert!((vec_dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]) - 32.0).abs() < EPS);
    }

    #[test]
    fn test_vec_add_sub() {
        let a = vec![1.0, 2.0];
        let b = vec![3.0, 4.0];
        let sum = vec_add(&a, &b);
        assert!((sum[0] - 4.0).abs() < EPS);
        assert!((sum[1] - 6.0).abs() < EPS);
        let diff = vec_sub(&a, &b);
        assert!((diff[0] - (-2.0)).abs() < EPS);
    }

    #[test]
    fn test_vec_scale_and_axpy() {
        let a = vec![1.0, 2.0, 3.0];
        let s = vec_scale(&a, 2.0);
        assert!((s[2] - 6.0).abs() < EPS);

        let mut c = vec![1.0, 1.0, 1.0];
        vec_axpy(&mut c, 3.0, &[1.0, 0.0, -1.0]);
        assert!((c[0] - 4.0).abs() < EPS);
        assert!((c[1] - 1.0).abs() < EPS);
        assert!((c[2] - (-2.0)).abs() < EPS);
    }

    #[test]
    fn test_mat_vec_mul() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let x = vec![1.0, 1.0];
        let y = mat_vec_mul(&a, &x);
        assert!((y[0] - 3.0).abs() < EPS);
        assert!((y[1] - 7.0).abs() < EPS);
    }

    #[test]
    fn test_mat_t_vec_mul() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let x = vec![1.0, 1.0];
        let y = mat_t_vec_mul(&a, &x);
        assert!((y[0] - 4.0).abs() < EPS);
        assert!((y[1] - 6.0).abs() < EPS);
    }

    #[test]
    fn test_solve_dense_identity() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let b = vec![3.0, 7.0];
        let x = solve_dense(&a, &b).unwrap();
        assert!((x[0] - 3.0).abs() < EPS);
        assert!((x[1] - 7.0).abs() < EPS);
    }

    #[test]
    fn test_solve_dense_2x2() {
        let a = vec![vec![2.0, 1.0], vec![5.0, 3.0]];
        let b = vec![4.0, 7.0];
        let x = solve_dense(&a, &b).unwrap();
        // 2x+y=4, 5x+3y=7 => x=5, y=-6
        assert!((x[0] - 5.0).abs() < EPS);
        assert!((x[1] - (-6.0)).abs() < EPS);
    }

    #[test]
    fn test_solve_dense_singular() {
        let a = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
        let b = vec![3.0, 6.0];
        assert!(solve_dense(&a, &b).is_none());
    }

    #[test]
    fn test_solve_cholesky() {
        // SPD matrix: [[4, 2], [2, 3]]
        let a = vec![vec![4.0, 2.0], vec![2.0, 3.0]];
        let b = vec![8.0, 7.0];
        let x = solve_cholesky(&a, &b).unwrap();
        // Verify A x = b
        let r0 = 4.0 * x[0] + 2.0 * x[1];
        let r1 = 2.0 * x[0] + 3.0 * x[1];
        assert!((r0 - 8.0).abs() < EPS);
        assert!((r1 - 7.0).abs() < EPS);
    }

    #[test]
    fn test_solve_cholesky_not_spd() {
        let a = vec![vec![-1.0, 0.0], vec![0.0, 1.0]];
        let b = vec![1.0, 1.0];
        assert!(solve_cholesky(&a, &b).is_none());
    }

    #[test]
    fn test_vec_outer() {
        let a = vec![1.0, 2.0];
        let b = vec![3.0, 4.0, 5.0];
        let m = vec_outer(&a, &b);
        assert_eq!(m.len(), 2);
        assert_eq!(m[0].len(), 3);
        assert!((m[1][2] - 10.0).abs() < EPS);
    }

    #[test]
    fn test_mat_aat() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let aat = mat_aat(&a);
        assert!((aat[0][0] - 5.0).abs() < EPS);  // 1*1+2*2
        assert!((aat[0][1] - 11.0).abs() < EPS); // 1*3+2*4
        assert!((aat[1][1] - 25.0).abs() < EPS); // 9+16
    }

    #[test]
    fn test_evaluate_conservation_energy() {
        let s = make_two_body();
        let e = evaluate_conservation(&s, ConservationKind::Energy);
        assert!((e - 1.0).abs() < EPS);
    }

    #[test]
    fn test_evaluate_conservation_momentum() {
        let s = make_two_body();
        let p = evaluate_conservation(&s, ConservationKind::Momentum);
        assert!(p < EPS);
    }

    #[test]
    fn test_repair_result_constructors() {
        let r = RepairResult::success(10, 1e-13, 0.001);
        assert!(r.success);
        assert_eq!(r.iterations, 10);
        let r2 = RepairResult::failure(100, 0.5, 0.0);
        assert!(!r2.success);
    }

    #[test]
    fn test_solve_dense_3x3() {
        let a = vec![
            vec![2.0, 1.0, -1.0],
            vec![-3.0, -1.0, 2.0],
            vec![-2.0, 1.0, 2.0],
        ];
        let b = vec![8.0, -11.0, -3.0];
        let x = solve_dense(&a, &b).unwrap();
        assert!((x[0] - 2.0).abs() < 1e-8);
        assert!((x[1] - 3.0).abs() < 1e-8);
        assert!((x[2] - (-1.0)).abs() < 1e-8);
    }
}
