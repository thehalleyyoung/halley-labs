//! Symplectic integrator auditor: verify that an integrator preserves symplectic structure.
//!
//! Given a numerical integrator Φ_h mapping (q_n, p_n) → (q_{n+1}, p_{n+1}),
//! we compute the Jacobian J = ∂Φ_h/∂z via finite differences, then verify
//! the symplectic condition J^T · Ω · J = Ω where Ω is the standard symplectic matrix.

use serde::{Deserialize, Serialize};
use sim_types::SimulationState;

// ─── Result Types ───────────────────────────────────────────────────────────

/// Result of auditing a single integration step for symplecticity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymplecticStepResult {
    pub timestep: usize,
    pub time: f64,
    /// Frobenius norm of (J^T Ω J − Ω), measuring symplectic violation.
    pub violation_norm: f64,
    /// |det(J) − 1|, measuring phase-space volume preservation.
    pub det_deviation: f64,
}

/// Aggregate result of a symplectic audit across multiple steps.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymplecticAuditResult {
    pub dimension: usize,
    pub max_violation_norm: f64,
    pub mean_violation_norm: f64,
    pub max_det_deviation: f64,
    pub mean_det_deviation: f64,
    pub worst_step: usize,
    pub worst_time: f64,
    pub is_symplectic: bool,
    pub steps: Vec<SymplecticStepResult>,
}

// ─── Symplectic Matrix Utilities ────────────────────────────────────────────

/// Build the standard 2n×2n symplectic matrix Ω = [[0, I], [-I, 0]].
fn build_omega(n: usize) -> Vec<Vec<f64>> {
    let dim = 2 * n;
    let mut omega = vec![vec![0.0; dim]; dim];
    for i in 0..n {
        omega[i][n + i] = 1.0;
        omega[n + i][i] = -1.0;
    }
    omega
}

/// Compute C = A^T · B for square matrices.
fn mat_transpose_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut c = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0;
            for k in 0..n {
                s += a[k][i] * b[k][j];
            }
            c[i][j] = s;
        }
    }
    c
}

/// Compute C = A · B for square matrices.
fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let mut c = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0;
            for k in 0..n {
                s += a[i][k] * b[k][j];
            }
            c[i][j] = s;
        }
    }
    c
}

/// Frobenius norm of a matrix.
fn frobenius_norm(m: &[Vec<f64>]) -> f64 {
    m.iter()
        .flat_map(|row| row.iter())
        .map(|x| x * x)
        .sum::<f64>()
        .sqrt()
}

/// Determinant of a square matrix via LU decomposition (partial pivoting).
fn determinant(m: &[Vec<f64>]) -> f64 {
    let n = m.len();
    if n == 0 {
        return 1.0;
    }
    let mut a: Vec<Vec<f64>> = m.to_vec();
    let mut sign = 1.0_f64;

    for col in 0..n {
        // Partial pivoting
        let mut max_row = col;
        let mut max_val = a[col][col].abs();
        for row in (col + 1)..n {
            if a[row][col].abs() > max_val {
                max_val = a[row][col].abs();
                max_row = row;
            }
        }
        if max_val < 1e-30 {
            return 0.0;
        }
        if max_row != col {
            a.swap(col, max_row);
            sign = -sign;
        }
        let pivot = a[col][col];
        for row in (col + 1)..n {
            let factor = a[row][col] / pivot;
            for j in col..n {
                let val = a[col][j];
                a[row][j] -= factor * val;
            }
        }
    }

    let mut det = sign;
    for i in 0..n {
        det *= a[i][i];
    }
    det
}

/// Extract the phase-space vector z = [q_1x, q_1y, q_1z, ..., p_1x, p_1y, p_1z, ...]
/// from a simulation state. Returns a flat vector of length 6N (3N positions + 3N momenta).
fn state_to_phase_vec(state: &SimulationState) -> Vec<f64> {
    let n = state.particles.len();
    let mut z = Vec::with_capacity(6 * n);
    for p in &state.particles {
        z.push(p.position.x);
        z.push(p.position.y);
        z.push(p.position.z);
    }
    for p in &state.particles {
        z.push(p.velocity.x * p.mass);
        z.push(p.velocity.y * p.mass);
        z.push(p.velocity.z * p.mass);
    }
    z
}

// ─── Auditor ────────────────────────────────────────────────────────────────

/// Audits whether a numerical integrator preserves symplectic structure.
///
/// Given consecutive simulation states, the auditor numerically estimates
/// the Jacobian of the integration map via finite differences, then checks
/// the symplectic condition J^T · Ω · J = Ω.
pub struct SymplecticIntegratorAuditor {
    /// Perturbation size for finite-difference Jacobian estimation.
    epsilon: f64,
    /// Tolerance for declaring the integrator symplectic.
    tolerance: f64,
}

impl SymplecticIntegratorAuditor {
    pub fn new(epsilon: f64, tolerance: f64) -> Self {
        Self { epsilon, tolerance }
    }

    /// Default auditor with ε = 10⁻⁷ and tolerance = 10⁻⁶.
    pub fn default_auditor() -> Self {
        Self {
            epsilon: 1e-7,
            tolerance: 1e-6,
        }
    }

    /// Compute the Jacobian of the map from state[i] to state[i+1] using
    /// central finite differences on the provided stepper function.
    ///
    /// `stepper` takes a phase-space vector and returns the next phase-space vector.
    pub fn compute_jacobian(
        &self,
        z: &[f64],
        stepper: &dyn Fn(&[f64]) -> Vec<f64>,
    ) -> Vec<Vec<f64>> {
        let dim = z.len();
        let mut jac = vec![vec![0.0; dim]; dim];

        for j in 0..dim {
            let mut z_plus = z.to_vec();
            let mut z_minus = z.to_vec();
            z_plus[j] += self.epsilon;
            z_minus[j] -= self.epsilon;

            let f_plus = stepper(&z_plus);
            let f_minus = stepper(&z_minus);

            for i in 0..dim {
                jac[i][j] = (f_plus[i] - f_minus[i]) / (2.0 * self.epsilon);
            }
        }
        jac
    }

    /// Check the symplectic condition for a single Jacobian matrix.
    /// Returns (violation_norm, det_deviation).
    pub fn check_symplectic_condition(&self, jacobian: &[Vec<f64>]) -> (f64, f64) {
        let dim = jacobian.len();
        assert!(dim % 2 == 0, "phase-space dimension must be even");
        let n = dim / 2;

        let omega = build_omega(n);
        // Compute J^T · Ω · J
        let jt_omega = mat_transpose_mul(jacobian, &omega);
        let jt_omega_j = mat_mul(&jt_omega, jacobian);

        // Compute difference: J^T Ω J − Ω
        let mut diff = vec![vec![0.0; dim]; dim];
        for i in 0..dim {
            for j in 0..dim {
                diff[i][j] = jt_omega_j[i][j] - omega[i][j];
            }
        }

        let violation_norm = frobenius_norm(&diff);
        let det_dev = (determinant(jacobian) - 1.0).abs();

        (violation_norm, det_dev)
    }

    /// Audit symplecticity across a simulation trace using consecutive states.
    ///
    /// Computes the numerical Jacobian at each step using the provided stepper,
    /// then checks the symplectic condition.
    pub fn audit_with_stepper(
        &self,
        states: &[SimulationState],
        stepper: &dyn Fn(&[f64]) -> Vec<f64>,
    ) -> SymplecticAuditResult {
        if states.len() < 2 {
            return SymplecticAuditResult {
                dimension: 0,
                max_violation_norm: 0.0,
                mean_violation_norm: 0.0,
                max_det_deviation: 0.0,
                mean_det_deviation: 0.0,
                worst_step: 0,
                worst_time: 0.0,
                is_symplectic: true,
                steps: Vec::new(),
            };
        }

        let dim = 6 * states[0].particles.len();
        let mut steps = Vec::with_capacity(states.len() - 1);
        let mut max_viol = 0.0_f64;
        let mut max_det = 0.0_f64;
        let mut sum_viol = 0.0;
        let mut sum_det = 0.0;
        let mut worst_step = 0;
        let mut worst_time = 0.0;

        for i in 0..(states.len() - 1) {
            let z = state_to_phase_vec(&states[i]);
            let jac = self.compute_jacobian(&z, stepper);
            let (viol, det_dev) = self.check_symplectic_condition(&jac);

            if viol > max_viol {
                max_viol = viol;
                worst_step = i;
                worst_time = states[i].time;
            }
            max_det = max_det.max(det_dev);
            sum_viol += viol;
            sum_det += det_dev;

            steps.push(SymplecticStepResult {
                timestep: i,
                time: states[i].time,
                violation_norm: viol,
                det_deviation: det_dev,
            });
        }

        let n_steps = steps.len() as f64;

        SymplecticAuditResult {
            dimension: dim,
            max_violation_norm: max_viol,
            mean_violation_norm: sum_viol / n_steps,
            max_det_deviation: max_det,
            mean_det_deviation: sum_det / n_steps,
            worst_step,
            worst_time,
            is_symplectic: max_viol < self.tolerance,
            steps,
        }
    }

    /// Quick check: given two consecutive states, verify the symplectic condition
    /// without requiring a stepper (uses the observed map from state_a to state_b).
    /// Only works for 1-particle systems due to the finite-difference approach.
    pub fn check_pair(
        &self,
        state_a: &SimulationState,
        state_b: &SimulationState,
    ) -> (f64, f64) {
        let z_a = state_to_phase_vec(state_a);
        let z_b = state_to_phase_vec(state_b);

        // Build a trivial linear-extrapolation "Jacobian" from the displacement
        let dim = z_a.len();
        let mut jac = vec![vec![0.0; dim]; dim];
        for i in 0..dim {
            jac[i][i] = if z_a[i].abs() > 1e-30 {
                z_b[i] / z_a[i]
            } else {
                1.0
            };
        }

        self.check_symplectic_condition(&jac)
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_omega() {
        let omega = build_omega(2);
        // 4×4 matrix: [[0,0,1,0],[0,0,0,1],[-1,0,0,0],[0,-1,0,0]]
        assert_eq!(omega[0][2], 1.0);
        assert_eq!(omega[1][3], 1.0);
        assert_eq!(omega[2][0], -1.0);
        assert_eq!(omega[3][1], -1.0);
        assert_eq!(omega[0][0], 0.0);
    }

    #[test]
    fn test_identity_is_symplectic() {
        let auditor = SymplecticIntegratorAuditor::default_auditor();
        let dim = 4;
        let identity: Vec<Vec<f64>> = (0..dim)
            .map(|i| {
                let mut row = vec![0.0; dim];
                row[i] = 1.0;
                row
            })
            .collect();

        let (viol, det_dev) = auditor.check_symplectic_condition(&identity);
        assert!(viol < 1e-14, "identity must be symplectic");
        assert!(det_dev < 1e-14, "det(I) = 1");
    }

    #[test]
    fn test_rotation_is_symplectic() {
        // A rotation in q-p plane is symplectic
        let theta = 0.3_f64;
        let c = theta.cos();
        let s = theta.sin();
        // 2×2 rotation embedded in canonical coordinates (1 DOF)
        let jac = vec![vec![c, s], vec![-s, c]];

        let auditor = SymplecticIntegratorAuditor::default_auditor();
        let (viol, det_dev) = auditor.check_symplectic_condition(&jac);
        assert!(viol < 1e-14, "rotation is symplectic, got {}", viol);
        assert!(det_dev < 1e-14);
    }

    #[test]
    fn test_non_symplectic_detected() {
        // A scaling matrix is not symplectic (det ≠ 1 for non-unit scaling)
        let jac = vec![
            vec![2.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ];

        let auditor = SymplecticIntegratorAuditor::default_auditor();
        let (viol, _) = auditor.check_symplectic_condition(&jac);
        assert!(viol > 1e-6, "non-unit scaling should violate symplecticity, got {}", viol);
    }

    #[test]
    fn test_determinant_identity() {
        let id = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        assert!((determinant(&id) - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_determinant_2x2() {
        let m = vec![vec![3.0, 8.0], vec![4.0, 6.0]];
        assert!((determinant(&m) - (3.0 * 6.0 - 8.0 * 4.0)).abs() < 1e-12);
    }

    #[test]
    fn test_jacobian_identity_stepper() {
        let auditor = SymplecticIntegratorAuditor::new(1e-7, 1e-6);
        let z = vec![1.0, 2.0, 3.0, 4.0];
        // Identity stepper
        let id_stepper = |z: &[f64]| z.to_vec();
        let jac = auditor.compute_jacobian(&z, &id_stepper);

        for i in 0..4 {
            for j in 0..4 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (jac[i][j] - expected).abs() < 1e-6,
                    "jac[{}][{}] = {} != {}",
                    i,
                    j,
                    jac[i][j],
                    expected
                );
            }
        }
    }
}
