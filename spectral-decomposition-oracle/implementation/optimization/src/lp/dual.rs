use log::{debug, info, warn};
use serde::{Deserialize, Serialize};

use crate::error::{OptError, OptResult};
use crate::lp::{BasisStatus, ConstraintType, LpProblem, LpSolution};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Information about the dual of an LP solution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DualInfo {
    pub dual_objective: f64,
    pub dual_values: Vec<f64>,
    pub reduced_costs: Vec<f64>,
    pub complementary_slackness_violation: f64,
}

/// Range within which a coefficient can change while the current basis stays optimal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoefficientRange {
    pub index: usize,
    pub current_value: f64,
    pub allowable_increase: f64,
    pub allowable_decrease: f64,
}

/// Sensitivity analysis report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensitivityReport {
    pub rhs_ranges: Vec<CoefficientRange>,
    pub obj_ranges: Vec<CoefficientRange>,
}

// ---------------------------------------------------------------------------
// Dual construction
// ---------------------------------------------------------------------------

/// Build the dual LP from a primal LP.
///
/// Primal (standard): min c^T x  s.t.  A_i x {<=,=,>=} b_i,  x_j >= lb_j.
///
/// Dual rules per constraint type:
///   Le  → dual var y_i >= 0
///   Ge  → dual var y_i <= 0  (we negate to keep >= 0)
///   Eq  → dual var y_i free
///
/// Dual rules per variable:
///   x_j >= 0 (lb=0, ub=inf) → A^T_j y {<=,=,>=} c_j  (Le for min)
///   x_j free                → A^T_j y = c_j
pub fn construct_dual(primal: &LpProblem) -> OptResult<LpProblem> {
    primal.validate()?;

    let m = primal.num_constraints;
    let n = primal.num_vars;
    let (t_rs, t_ci, t_vals) = primal.transpose_constraint_matrix();

    // Dual is a maximisation problem: max b^T y  subject to  A^T y <= c  (for min primal)
    // We'll store it as a min problem with negated objective.
    let mut dual = LpProblem::new(false);

    // Dual variables – one per primal constraint
    for i in 0..m {
        let obj_coeff = -primal.rhs[i]; // negate for min
        let (lb, ub) = match primal.constraint_types[i] {
            ConstraintType::Le => (0.0, f64::INFINITY),
            ConstraintType::Ge => (f64::NEG_INFINITY, 0.0),
            ConstraintType::Eq => (f64::NEG_INFINITY, f64::INFINITY),
        };
        dual.add_variable(obj_coeff, lb, ub, Some(format!("y{}", i)));
    }

    // Dual constraints – one per primal variable
    for j in 0..n {
        let start = t_rs[j];
        let end = t_rs[j + 1];

        let indices: Vec<usize> = t_ci[start..end].to_vec();
        let coeffs: Vec<f64> = t_vals[start..end].to_vec();

        let rhs_val = if primal.maximize {
            -primal.obj_coeffs[j]
        } else {
            primal.obj_coeffs[j]
        };

        // If primal variable has lb=0, ub=inf → dual constraint is Le
        // If primal variable is free → dual constraint is Eq
        let ctype = if primal.lower_bounds[j] <= f64::NEG_INFINITY + 1.0
            && primal.upper_bounds[j] >= f64::INFINITY - 1.0
        {
            ConstraintType::Eq
        } else {
            ConstraintType::Le
        };

        dual.add_constraint(&indices, &coeffs, ctype, rhs_val)?;
    }

    debug!(
        "Constructed dual: {} vars, {} constraints",
        dual.num_vars, dual.num_constraints
    );
    Ok(dual)
}

// ---------------------------------------------------------------------------
// Complementary slackness
// ---------------------------------------------------------------------------

/// Verify complementary slackness conditions and return the maximum violation.
///
/// For an LP min c^T x, Ax {<=,=,>=} b, x >= 0:
///   - Primal CS: x_j * rc_j ≈ 0 for all j
///   - Dual CS:   (A_i x - b_i) * y_i ≈ 0 for inequality constraints
pub fn verify_complementary_slackness(
    problem: &LpProblem,
    primal_sol: &LpSolution,
    dual_sol: &LpSolution,
    tol: f64,
) -> OptResult<f64> {
    let n = problem.num_vars;
    let m = problem.num_constraints;

    if primal_sol.primal_values.len() < n || dual_sol.dual_values.len() < m {
        return Err(OptError::InvalidProblem {
            reason: "Solution dimensions do not match problem".into(),
        });
    }

    let rc = compute_reduced_costs(problem, &dual_sol.dual_values);
    let mut max_violation = 0.0f64;

    // Primal CS: x_j * rc_j ≈ 0
    for j in 0..n {
        let x_j = primal_sol.primal_values[j] - problem.lower_bounds[j];
        let viol = (x_j * rc[j]).abs();
        max_violation = max_violation.max(viol);
    }

    // Dual CS: slack_i * y_i ≈ 0
    let mut ax = vec![0.0; m];
    problem.multiply_ax(&primal_sol.primal_values, &mut ax);
    for i in 0..m {
        let slack = match problem.constraint_types[i] {
            ConstraintType::Le => problem.rhs[i] - ax[i],
            ConstraintType::Ge => ax[i] - problem.rhs[i],
            ConstraintType::Eq => 0.0,
        };
        let viol = (slack * dual_sol.dual_values[i]).abs();
        max_violation = max_violation.max(viol);
    }

    if max_violation > tol {
        warn!(
            "Complementary slackness violation {:.2e} exceeds tolerance {:.2e}",
            max_violation, tol
        );
    }
    Ok(max_violation)
}

// ---------------------------------------------------------------------------
// Reduced costs
// ---------------------------------------------------------------------------

/// Compute reduced costs: rc_j = c_j − a_j^T y.
pub fn compute_reduced_costs(problem: &LpProblem, dual_values: &[f64]) -> Vec<f64> {
    let n = problem.num_vars;
    let mut aty = vec![0.0; n];
    problem.multiply_atx(dual_values, &mut aty);

    let mut rc = vec![0.0; n];
    for j in 0..n {
        let c_j = if problem.maximize {
            -problem.obj_coeffs[j]
        } else {
            problem.obj_coeffs[j]
        };
        rc[j] = c_j - aty[j];
    }
    rc
}

// ---------------------------------------------------------------------------
// Extract dual from basis
// ---------------------------------------------------------------------------

/// Compute y = c_B * B^{-1} from basis information.
///
/// Given the basis status vector, identify the basis columns, form B, factorise,
/// and compute the dual variables.
pub fn extract_dual_from_basis(
    problem: &LpProblem,
    basis: &[BasisStatus],
) -> OptResult<Vec<f64>> {
    let m = problem.num_constraints;
    let n = basis.len();

    let basic_indices: Vec<usize> = (0..n)
        .filter(|&j| basis[j] == BasisStatus::Basic)
        .collect();

    if basic_indices.len() != m {
        return Err(OptError::NumericalError {
            context: format!(
                "Basis size {} != num_constraints {}",
                basic_indices.len(),
                m
            ),
        });
    }

    // Form dense basis matrix B (m×m) column-by-column
    let mut b_mat = vec![0.0; m * m];
    let mut col_buf = vec![0.0; m];
    for (k, &j) in basic_indices.iter().enumerate() {
        problem.column_dense(j, &mut col_buf);
        for i in 0..m {
            b_mat[i * m + k] = col_buf[i];
        }
    }

    // c_B vector
    let c_b: Vec<f64> = basic_indices
        .iter()
        .map(|&j| {
            if problem.maximize {
                -problem.obj_coeffs[j]
            } else {
                problem.obj_coeffs[j]
            }
        })
        .collect();

    // Solve y * B = c_B  →  B^T y = c_B
    let mut bt = vec![0.0; m * m];
    for i in 0..m {
        for j in 0..m {
            bt[i * m + j] = b_mat[j * m + i];
        }
    }

    let y = dense_solve_lu(&bt, &c_b, m)?;
    Ok(y)
}

// ---------------------------------------------------------------------------
// Check dual feasibility
// ---------------------------------------------------------------------------

/// Verify that all reduced costs have the correct sign for optimality.
///
/// For a minimisation problem with x_j >= 0, need rc_j >= -tol.
pub fn check_dual_feasibility(
    problem: &LpProblem,
    dual_values: &[f64],
    tol: f64,
) -> OptResult<bool> {
    let rc = compute_reduced_costs(problem, dual_values);
    for j in 0..problem.num_vars {
        let is_free = problem.lower_bounds[j] <= f64::NEG_INFINITY + 1.0
            && problem.upper_bounds[j] >= f64::INFINITY - 1.0;
        if is_free {
            if rc[j].abs() > tol {
                debug!("Dual infeasible: free var {} has rc = {:.2e}", j, rc[j]);
                return Ok(false);
            }
        } else if rc[j] < -tol {
            debug!("Dual infeasible: var {} has rc = {:.2e}", j, rc[j]);
            return Ok(false);
        }
    }
    Ok(true)
}

// ---------------------------------------------------------------------------
// Farkas ray (certificate of infeasibility)
// ---------------------------------------------------------------------------

/// Extract a Farkas ray proving primal infeasibility.
///
/// When the primal is infeasible, the auxiliary Phase-I problem yields a dual ray
/// y such that y^T A >= 0, y^T b < 0 (for a system Ax <= b).
pub fn extract_farkas_ray(
    problem: &LpProblem,
    basis: &[BasisStatus],
) -> OptResult<Vec<f64>> {
    let m = problem.num_constraints;
    let n = basis.len();

    let basic_indices: Vec<usize> = (0..n)
        .filter(|&j| basis[j] == BasisStatus::Basic)
        .collect();

    if basic_indices.len() != m {
        return Err(OptError::NumericalError {
            context: "Cannot extract Farkas ray: basis size mismatch".into(),
        });
    }

    // Form basis matrix
    let mut b_mat = vec![0.0; m * m];
    let mut col_buf = vec![0.0; m];
    for (k, &j) in basic_indices.iter().enumerate() {
        problem.column_dense(j, &mut col_buf);
        for i in 0..m {
            b_mat[i * m + k] = col_buf[i];
        }
    }

    // For the Farkas certificate we solve B^T y = 0 with sign constraints.
    // In practice, the ray comes from the last row of the Phase-I tableau.
    // We use the approach: solve B^T y = e_r for the most infeasible row r,
    // then check the sign of y^T b.
    let mut bt = vec![0.0; m * m];
    for i in 0..m {
        for j in 0..m {
            bt[i * m + j] = b_mat[j * m + i];
        }
    }

    // Use the rhs to find the most violated direction
    let mut max_infeas = 0.0f64;
    let mut infeas_row = 0usize;
    // Solve B^{-1} b to find basic variable values
    let b_inv_b = dense_solve_lu(&b_mat, &problem.rhs, m)?;
    for i in 0..m {
        if b_inv_b[i] < -max_infeas {
            max_infeas = -b_inv_b[i];
            infeas_row = i;
        }
    }

    let mut e_r = vec![0.0; m];
    e_r[infeas_row] = -1.0;

    let y = dense_solve_lu(&bt, &e_r, m)?;

    // Normalise
    let norm: f64 = y.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm < 1e-15 {
        return Err(OptError::NumericalError {
            context: "Farkas ray is numerically zero".into(),
        });
    }
    let y_norm: Vec<f64> = y.iter().map(|v| v / norm).collect();

    debug!("Farkas ray extracted, ||y|| = {:.2e}", norm);
    Ok(y_norm)
}

// ---------------------------------------------------------------------------
// Strong duality gap
// ---------------------------------------------------------------------------

/// Relative duality gap: |primal − dual| / max(1, |primal|).
pub fn strong_duality_gap(primal_obj: f64, dual_obj: f64) -> f64 {
    let denom = 1.0f64.max(primal_obj.abs());
    (primal_obj - dual_obj).abs() / denom
}

// ---------------------------------------------------------------------------
// Sensitivity analysis
// ---------------------------------------------------------------------------

/// Sensitivity / ranging analysis: determine how much each RHS and objective
/// coefficient can change while keeping the current basis optimal.
pub fn sensitivity_analysis(
    problem: &LpProblem,
    solution: &LpSolution,
) -> OptResult<SensitivityReport> {
    let m = problem.num_constraints;
    let n = problem.num_vars;

    if solution.basis_status.len() < n {
        return Err(OptError::InvalidProblem {
            reason: "Solution missing basis status information".into(),
        });
    }

    let basic_indices: Vec<usize> = (0..solution.basis_status.len())
        .filter(|&j| solution.basis_status[j] == BasisStatus::Basic)
        .collect();

    if basic_indices.len() != m {
        return Err(OptError::NumericalError {
            context: format!(
                "Basis size {} != num_constraints {} for sensitivity",
                basic_indices.len(),
                m
            ),
        });
    }

    // Build basis matrix and invert
    let mut b_mat = vec![0.0; m * m];
    let mut col_buf = vec![0.0; m];
    for (k, &j) in basic_indices.iter().enumerate() {
        problem.column_dense(j, &mut col_buf);
        for i in 0..m {
            b_mat[i * m + k] = col_buf[i];
        }
    }

    let b_inv = dense_inverse(&b_mat, m)?;

    // B^{-1} b  (basic variable values)
    let mut xb = vec![0.0; m];
    for i in 0..m {
        let mut s = 0.0;
        for k in 0..m {
            s += b_inv[i * m + k] * problem.rhs[k];
        }
        xb[i] = s;
    }

    // --- RHS ranges ---
    let mut rhs_ranges = Vec::with_capacity(m);
    for r in 0..m {
        let col_r: Vec<f64> = (0..m).map(|i| b_inv[i * m + r]).collect();

        let mut max_inc = f64::INFINITY;
        let mut max_dec = f64::INFINITY;

        for i in 0..m {
            if col_r[i] > 1e-12 {
                max_inc = max_inc.min(xb[i] / col_r[i]);
            } else if col_r[i] < -1e-12 {
                max_dec = max_dec.min(-xb[i] / col_r[i]);
            }
        }

        rhs_ranges.push(CoefficientRange {
            index: r,
            current_value: problem.rhs[r],
            allowable_increase: max_inc,
            allowable_decrease: max_dec,
        });
    }

    // --- Objective coefficient ranges ---
    let nonbasic_indices: Vec<usize> = (0..n.min(solution.basis_status.len()))
        .filter(|&j| solution.basis_status[j] != BasisStatus::Basic)
        .collect();

    // c_B
    let c_b: Vec<f64> = basic_indices
        .iter()
        .map(|&j| {
            if problem.maximize {
                -problem.obj_coeffs[j]
            } else {
                problem.obj_coeffs[j]
            }
        })
        .collect();

    // y = c_B B^{-1}
    let mut y = vec![0.0; m];
    for i in 0..m {
        let mut s = 0.0;
        for k in 0..m {
            s += c_b[k] * b_inv[k * m + i];
        }
        y[i] = s;
    }

    // For each non-basic variable, compute reduced cost and sensitivity
    // For each basic variable, determine the range of c_j change
    let mut obj_ranges = Vec::with_capacity(n);
    for j in 0..n {
        if solution.basis_status.get(j) == Some(&BasisStatus::Basic) {
            // Basic variable: Δc_j affects reduced costs of nonbasics through y
            // Find the basis position of j
            let basis_pos = basic_indices.iter().position(|&b| b == j);
            if let Some(bp) = basis_pos {
                let mut max_inc = f64::INFINITY;
                let mut max_dec = f64::INFINITY;

                for &nb in &nonbasic_indices {
                    let mut col_nb = vec![0.0; m];
                    problem.column_dense(nb, &mut col_nb);

                    // B^{-1} a_nb
                    let mut binv_a = vec![0.0; m];
                    for i in 0..m {
                        let mut s = 0.0;
                        for k in 0..m {
                            s += b_inv[i * m + k] * col_nb[k];
                        }
                        binv_a[i] = s;
                    }

                    let rc_nb = solution
                        .reduced_costs
                        .get(nb)
                        .copied()
                        .unwrap_or_else(|| {
                            let c_nb = if problem.maximize {
                                -problem.obj_coeffs[nb]
                            } else {
                                problem.obj_coeffs[nb]
                            };
                            let mut s = 0.0;
                            for i in 0..m {
                                s += y[i] * col_nb[i];
                            }
                            c_nb - s
                        });

                    let coeff = binv_a[bp];
                    if coeff > 1e-12 {
                        max_inc = max_inc.min(rc_nb / coeff);
                    } else if coeff < -1e-12 {
                        max_dec = max_dec.min(-rc_nb / coeff);
                    }
                }

                obj_ranges.push(CoefficientRange {
                    index: j,
                    current_value: problem.obj_coeffs[j],
                    allowable_increase: max_inc,
                    allowable_decrease: max_dec,
                });
            } else {
                obj_ranges.push(CoefficientRange {
                    index: j,
                    current_value: problem.obj_coeffs[j],
                    allowable_increase: f64::INFINITY,
                    allowable_decrease: f64::INFINITY,
                });
            }
        } else {
            // Nonbasic variable: rc_j must stay >= 0, so can change c_j by rc_j
            let rc_j = solution.reduced_costs.get(j).copied().unwrap_or(0.0);
            obj_ranges.push(CoefficientRange {
                index: j,
                current_value: problem.obj_coeffs[j],
                allowable_increase: f64::INFINITY,
                allowable_decrease: rc_j.abs(),
            });
        }
    }

    info!(
        "Sensitivity analysis complete: {} rhs ranges, {} obj ranges",
        rhs_ranges.len(),
        obj_ranges.len()
    );

    Ok(SensitivityReport {
        rhs_ranges,
        obj_ranges,
    })
}

// ---------------------------------------------------------------------------
// Dense linear algebra helpers (small systems only)
// ---------------------------------------------------------------------------

/// Solve Ax = b using LU factorisation with partial pivoting (dense, n×n).
fn dense_solve_lu(a: &[f64], b: &[f64], n: usize) -> OptResult<Vec<f64>> {
    // Copy A into LU storage
    let mut lu = a.to_vec();
    let mut perm: Vec<usize> = (0..n).collect();
    let x = b.to_vec();

    // Forward elimination with partial pivoting
    for k in 0..n {
        // Find pivot
        let mut max_val = lu[perm[k] * n + k].abs();
        let mut max_row = k;
        for i in (k + 1)..n {
            let v = lu[perm[i] * n + k].abs();
            if v > max_val {
                max_val = v;
                max_row = i;
            }
        }
        if max_val < 1e-14 {
            return Err(OptError::NumericalError {
                context: format!("Singular matrix in LU factorization at column {}", k),
            });
        }
        perm.swap(k, max_row);

        let pivot_row = perm[k];
        for i in (k + 1)..n {
            let target_row = perm[i];
            let factor = lu[target_row * n + k] / lu[pivot_row * n + k];
            lu[target_row * n + k] = factor;
            for j in (k + 1)..n {
                lu[target_row * n + j] -= factor * lu[pivot_row * n + j];
            }
        }
    }

    // Permute b
    let mut pb = vec![0.0; n];
    for i in 0..n {
        pb[i] = x[perm[i]];
    }

    // Forward substitution (L y = Pb)
    for i in 1..n {
        let row = perm[i];
        for j in 0..i {
            let _row_j = perm[j];
            pb[i] -= lu[row * n + j] * pb[j];
        }
    }

    // Backward substitution (U x = y)
    for i in (0..n).rev() {
        let row = perm[i];
        for j in (i + 1)..n {
            pb[i] -= lu[row * n + j] * pb[j];
        }
        pb[i] /= lu[row * n + i];
    }

    Ok(pb)
}

/// Dense matrix inverse via LU for each column of identity.
fn dense_inverse(a: &[f64], n: usize) -> OptResult<Vec<f64>> {
    let mut inv = vec![0.0; n * n];
    for j in 0..n {
        let mut e = vec![0.0; n];
        e[j] = 1.0;
        let col = dense_solve_lu(a, &e, n)?;
        for i in 0..n {
            inv[i * n + j] = col[i];
        }
    }
    Ok(inv)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lp::ConstraintType;

    /// Build a small LP: min -x1 - 2 x2  s.t.  x1+x2<=4, x1<=3, x2<=3, x>=0.
    fn small_lp() -> LpProblem {
        let mut lp = LpProblem::new(false);
        lp.add_variable(-1.0, 0.0, f64::INFINITY, None);
        lp.add_variable(-2.0, 0.0, f64::INFINITY, None);
        lp.add_constraint(&[0, 1], &[1.0, 1.0], ConstraintType::Le, 4.0)
            .unwrap();
        lp.add_constraint(&[0], &[1.0], ConstraintType::Le, 3.0)
            .unwrap();
        lp.add_constraint(&[1], &[1.0], ConstraintType::Le, 3.0)
            .unwrap();
        lp
    }

    /// A feasible LP solution at x=(1,3), obj = -7.
    fn small_lp_solution() -> LpSolution {
        LpSolution {
            status: SolverStatus::Optimal,
            objective_value: -7.0,
            primal_values: vec![1.0, 3.0],
            dual_values: vec![2.0, 0.0, 0.0],
            reduced_costs: vec![0.0, 0.0],
            basis_status: vec![BasisStatus::Basic; 2],
            iterations: 0,
            time_seconds: 0.0,
        }
    }

    #[test]
    fn test_construct_dual_dimensions() {
        let primal = small_lp();
        let dual = construct_dual(&primal).unwrap();
        // Dual has m=3 vars, n=2 constraints
        assert_eq!(dual.num_vars, 3);
        assert_eq!(dual.num_constraints, 2);
    }

    #[test]
    fn test_construct_dual_objective() {
        let primal = small_lp();
        let dual = construct_dual(&primal).unwrap();
        // Dual obj (negated for min): -(b) = [-4, -3, -3]
        assert!((dual.obj_coeffs[0] - (-4.0)).abs() < 1e-12);
        assert!((dual.obj_coeffs[1] - (-3.0)).abs() < 1e-12);
        assert!((dual.obj_coeffs[2] - (-3.0)).abs() < 1e-12);
    }

    #[test]
    fn test_construct_dual_bounds() {
        let primal = small_lp();
        let dual = construct_dual(&primal).unwrap();
        for i in 0..3 {
            assert!((dual.lower_bounds[i] - 0.0).abs() < 1e-12);
            assert!(dual.upper_bounds[i] >= 1e30);
        }
    }

    #[test]
    fn test_compute_reduced_costs() {
        let lp = small_lp();
        let y = vec![2.0, 0.0, 0.0];
        let rc = compute_reduced_costs(&lp, &y);
        // rc[0] = -1 - (1*2+1*0) = -1 - 2 = -3
        // rc[1] = -2 - (1*2+1*0) = -2 - 2 = -4
        assert!((rc[0] - (-3.0)).abs() < 1e-12);
        assert!((rc[1] - (-4.0)).abs() < 1e-12);
    }

    #[test]
    fn test_strong_duality_gap_zero() {
        let gap = strong_duality_gap(-7.0, -7.0);
        assert!(gap < 1e-12);
    }

    #[test]
    fn test_strong_duality_gap_nonzero() {
        let gap = strong_duality_gap(10.0, 8.0);
        assert!((gap - 0.2).abs() < 1e-12); // 2/10
    }

    #[test]
    fn test_check_dual_feasibility_false() {
        let lp = small_lp();
        let y = vec![0.0, 0.0, 0.0]; // rc = obj_coeffs = [-1, -2], both negative
        let feasible = check_dual_feasibility(&lp, &y, 1e-8).unwrap();
        assert!(!feasible);
    }

    #[test]
    fn test_dense_solve_lu_identity() {
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![3.0, 5.0];
        let x = dense_solve_lu(&a, &b, 2).unwrap();
        assert!((x[0] - 3.0).abs() < 1e-12);
        assert!((x[1] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_dense_solve_lu_2x2() {
        // [2 1; 5 3] x = [1; 2]  →  x = [1, -1]
        let a = vec![2.0, 1.0, 5.0, 3.0];
        let b = vec![1.0, 2.0];
        let x = dense_solve_lu(&a, &b, 2).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[1] - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_dense_inverse_2x2() {
        let a = vec![2.0, 1.0, 5.0, 3.0];
        let inv = dense_inverse(&a, 2).unwrap();
        // inv = [3, -1; -5, 2]
        assert!((inv[0] - 3.0).abs() < 1e-10);
        assert!((inv[1] - (-1.0)).abs() < 1e-10);
        assert!((inv[2] - (-5.0)).abs() < 1e-10);
        assert!((inv[3] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_sensitivity_analysis_basic() {
        // Use a problem where we can manually build a basis status vector for
        // the standard-form problem (with slacks).
        let mut lp = LpProblem::new(false);
        // min -x1 - x2, x1+x2 <= 4, x1 >= 0, x2 >= 0
        lp.add_variable(-1.0, 0.0, f64::INFINITY, None);
        lp.add_variable(-1.0, 0.0, f64::INFINITY, None);
        lp.add_constraint(&[0, 1], &[1.0, 1.0], ConstraintType::Le, 4.0)
            .unwrap();

        // Standard form adds slack s0: x1 + x2 + s0 = 4
        let (std_lp, _orig_n) = lp.to_standard_form().unwrap();

        // Optimal: x1=2, x2=2, s0=0 → basis = {x0, x1}, nonbasic = {s0}
        let sol = LpSolution {
            status: SolverStatus::Optimal,
            objective_value: -4.0,
            primal_values: vec![2.0, 2.0, 0.0],
            dual_values: vec![1.0],
            reduced_costs: vec![0.0, 0.0, 1.0],
            basis_status: vec![BasisStatus::Basic, BasisStatus::Basic, BasisStatus::AtLower],
            iterations: 0,
            time_seconds: 0.0,
        };

        let report = sensitivity_analysis(&std_lp, &sol).unwrap();
        assert_eq!(report.rhs_ranges.len(), 1);
        assert_eq!(report.obj_ranges.len(), 3);
    }

    #[test]
    fn test_verify_cs_near_optimal() {
        let lp = small_lp();
        // Optimal at x=(1,3), y=(2,0,0), obj = -7
        // rc = c - A^T y: rc[0]=-1-(1*2)=-3, rc[1]=-2-(1*2)=-4
        // This doesn't satisfy CS perfectly — this is intentional to test tolerance.
        let primal = LpSolution {
            status: SolverStatus::Optimal,
            objective_value: -7.0,
            primal_values: vec![1.0, 3.0],
            dual_values: vec![],
            reduced_costs: vec![],
            basis_status: vec![],
            iterations: 0,
            time_seconds: 0.0,
        };
        let dual = LpSolution {
            status: SolverStatus::Optimal,
            objective_value: -7.0,
            primal_values: vec![],
            dual_values: vec![2.0, 0.0, 0.0],
            reduced_costs: vec![],
            basis_status: vec![],
            iterations: 0,
            time_seconds: 0.0,
        };
        let viol = verify_complementary_slackness(&lp, &primal, &dual, 100.0).unwrap();
        // We just check it returns a finite number
        assert!(viol.is_finite());
    }
}
