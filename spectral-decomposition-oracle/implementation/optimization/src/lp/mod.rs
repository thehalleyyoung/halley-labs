pub mod simplex;
pub mod interior_point;
pub mod presolve;
pub mod dual;

use serde::{Deserialize, Serialize};
use std::fmt;

use crate::error::{OptError, OptResult};

// ---------------------------------------------------------------------------
// Constraint type
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConstraintType {
    Le,
    Eq,
    Ge,
}

impl fmt::Display for ConstraintType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConstraintType::Le => write!(f, "<="),
            ConstraintType::Eq => write!(f, "="),
            ConstraintType::Ge => write!(f, ">="),
        }
    }
}

// ---------------------------------------------------------------------------
// Solver / basis status
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SolverStatus {
    Optimal,
    Infeasible,
    Unbounded,
    IterationLimit,
    TimeLimit,
    NumericalError,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BasisStatus {
    Basic,
    AtLower,
    AtUpper,
    Free,
    Fixed,
}

// ---------------------------------------------------------------------------
// LP problem (sparse row-major storage)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LpProblem {
    pub num_vars: usize,
    pub num_constraints: usize,
    pub obj_coeffs: Vec<f64>,
    // CSR sparse matrix
    pub row_starts: Vec<usize>,
    pub col_indices: Vec<usize>,
    pub values: Vec<f64>,
    pub constraint_types: Vec<ConstraintType>,
    pub rhs: Vec<f64>,
    pub lower_bounds: Vec<f64>,
    pub upper_bounds: Vec<f64>,
    pub var_names: Vec<String>,
    pub maximize: bool,
}

impl LpProblem {
    /// Create an empty LP problem.
    pub fn new(maximize: bool) -> Self {
        Self {
            num_vars: 0,
            num_constraints: 0,
            obj_coeffs: Vec::new(),
            row_starts: vec![0],
            col_indices: Vec::new(),
            values: Vec::new(),
            constraint_types: Vec::new(),
            rhs: Vec::new(),
            lower_bounds: Vec::new(),
            upper_bounds: Vec::new(),
            var_names: Vec::new(),
            maximize,
        }
    }

    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    pub fn num_constraints(&self) -> usize {
        self.num_constraints
    }

    /// Add a variable with objective coefficient and bounds. Returns variable index.
    pub fn add_variable(&mut self, obj: f64, lb: f64, ub: f64, name: Option<String>) -> usize {
        let idx = self.num_vars;
        self.num_vars += 1;
        self.obj_coeffs.push(obj);
        self.lower_bounds.push(lb);
        self.upper_bounds.push(ub);
        self.var_names
            .push(name.unwrap_or_else(|| format!("x{}", idx)));
        idx
    }

    /// Add a constraint given as a sparse row (indices, coefficients), type, and rhs.
    pub fn add_constraint(
        &mut self,
        indices: &[usize],
        coeffs: &[f64],
        ctype: ConstraintType,
        rhs_val: f64,
    ) -> OptResult<usize> {
        if indices.len() != coeffs.len() {
            return Err(OptError::InvalidProblem {
                reason: "indices and coeffs length mismatch".into(),
            });
        }
        for &idx in indices {
            if idx >= self.num_vars {
                return Err(OptError::InvalidProblem {
                    reason: format!("variable index {} >= num_vars {}", idx, self.num_vars),
                });
            }
        }
        let con_idx = self.num_constraints;
        self.num_constraints += 1;
        for (&i, &v) in indices.iter().zip(coeffs.iter()) {
            self.col_indices.push(i);
            self.values.push(v);
        }
        self.row_starts.push(self.col_indices.len());
        self.constraint_types.push(ctype);
        self.rhs.push(rhs_val);
        Ok(con_idx)
    }

    /// Convert the problem to standard form: min c^T x, Ax = b, x >= 0.
    /// Adds slack / surplus / artificial bookkeeping and flips objective if maximise.
    /// Returns (standard_problem, original_var_count).
    pub fn to_standard_form(&self) -> OptResult<(LpProblem, usize)> {
        self.validate()?;

        let orig_n = self.num_vars;

        // Count extra variables needed
        let mut extra = 0usize;
        for ct in &self.constraint_types {
            match ct {
                ConstraintType::Le | ConstraintType::Ge => extra += 1,
                ConstraintType::Eq => {}
            }
        }
        let total_vars = orig_n + extra;

        let mut std = LpProblem::new(false);

        // Add original variables (shifted to be >= 0)
        for j in 0..orig_n {
            let c = if self.maximize {
                -self.obj_coeffs[j]
            } else {
                self.obj_coeffs[j]
            };
            std.add_variable(c, 0.0, f64::INFINITY, Some(self.var_names[j].clone()));
        }
        // Add slack/surplus variables
        for k in 0..extra {
            std.add_variable(0.0, 0.0, f64::INFINITY, Some(format!("_s{}", k)));
        }

        let mut slack_idx = orig_n;
        for i in 0..self.num_constraints {
            let rs = self.row_starts[i];
            let re = self.row_starts[i + 1];
            let mut indices: Vec<usize> = self.col_indices[rs..re].to_vec();
            let mut coeffs: Vec<f64> = self.values[rs..re].to_vec();

            // Shift variables by their lower bounds: x_orig = x_new + lb
            let mut rhs_adj = self.rhs[i];
            for pos in 0..indices.len() {
                let j = indices[pos];
                rhs_adj -= coeffs[pos] * self.lower_bounds[j];
            }

            match self.constraint_types[i] {
                ConstraintType::Le => {
                    indices.push(slack_idx);
                    coeffs.push(1.0);
                    slack_idx += 1;
                }
                ConstraintType::Ge => {
                    indices.push(slack_idx);
                    coeffs.push(-1.0);
                    slack_idx += 1;
                }
                ConstraintType::Eq => {}
            }

            std.add_constraint(&indices, &coeffs, ConstraintType::Eq, rhs_adj)?;
        }

        debug_assert_eq!(std.num_vars, total_vars);
        Ok((std, orig_n))
    }

    /// Basic validation.
    pub fn validate(&self) -> OptResult<()> {
        if self.obj_coeffs.len() != self.num_vars {
            return Err(OptError::InvalidProblem {
                reason: format!(
                    "obj_coeffs length {} != num_vars {}",
                    self.obj_coeffs.len(),
                    self.num_vars
                ),
            });
        }
        if self.lower_bounds.len() != self.num_vars || self.upper_bounds.len() != self.num_vars {
            return Err(OptError::InvalidProblem {
                reason: "bounds length mismatch".into(),
            });
        }
        if self.constraint_types.len() != self.num_constraints
            || self.rhs.len() != self.num_constraints
        {
            return Err(OptError::InvalidProblem {
                reason: "constraint metadata length mismatch".into(),
            });
        }
        if self.row_starts.len() != self.num_constraints + 1 {
            return Err(OptError::InvalidProblem {
                reason: format!(
                    "row_starts length {} != num_constraints+1 {}",
                    self.row_starts.len(),
                    self.num_constraints + 1
                ),
            });
        }
        for j in 0..self.num_vars {
            if self.lower_bounds[j] > self.upper_bounds[j] + 1e-12 {
                return Err(OptError::InvalidProblem {
                    reason: format!(
                        "variable {} has lb {} > ub {}",
                        j, self.lower_bounds[j], self.upper_bounds[j]
                    ),
                });
            }
        }
        Ok(())
    }

    /// Return the transpose of the constraint matrix in CSC form (as another CSR
    /// struct where "rows" are original columns).
    pub fn transpose_constraint_matrix(&self) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
        let m = self.num_constraints;
        let n = self.num_vars;
        let nnz = self.col_indices.len();

        // Count entries per column
        let mut col_count = vec![0usize; n];
        for &c in &self.col_indices {
            if c < n {
                col_count[c] += 1;
            }
        }

        // Build row_starts for transpose
        let mut t_row_starts = vec![0usize; n + 1];
        for j in 0..n {
            t_row_starts[j + 1] = t_row_starts[j] + col_count[j];
        }

        let mut t_col_indices = vec![0usize; nnz];
        let mut t_values = vec![0.0f64; nnz];
        let mut pos = vec![0usize; n];

        for i in 0..m {
            let rs = self.row_starts[i];
            let re = self.row_starts[i + 1];
            for k in rs..re {
                let j = self.col_indices[k];
                if j < n {
                    let dest = t_row_starts[j] + pos[j];
                    t_col_indices[dest] = i;
                    t_values[dest] = self.values[k];
                    pos[j] += 1;
                }
            }
        }

        (t_row_starts, t_col_indices, t_values)
    }

    /// Get the dense column `j` of the constraint matrix.
    pub fn column_dense(&self, j: usize, out: &mut [f64]) {
        for v in out.iter_mut() {
            *v = 0.0;
        }
        for i in 0..self.num_constraints {
            let rs = self.row_starts[i];
            let re = self.row_starts[i + 1];
            for k in rs..re {
                if self.col_indices[k] == j {
                    out[i] = self.values[k];
                }
            }
        }
    }

    /// Multiply constraint matrix by vector: out = A * x.
    pub fn multiply_ax(&self, x: &[f64], out: &mut [f64]) {
        for i in 0..self.num_constraints {
            let rs = self.row_starts[i];
            let re = self.row_starts[i + 1];
            let mut sum = 0.0;
            for k in rs..re {
                sum += self.values[k] * x[self.col_indices[k]];
            }
            out[i] = sum;
        }
    }

    /// Multiply constraint matrix transpose by vector: out = A^T * y.
    pub fn multiply_atx(&self, y: &[f64], out: &mut [f64]) {
        for v in out.iter_mut() {
            *v = 0.0;
        }
        for i in 0..self.num_constraints {
            let rs = self.row_starts[i];
            let re = self.row_starts[i + 1];
            for k in rs..re {
                out[self.col_indices[k]] += self.values[k] * y[i];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// LP solution
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LpSolution {
    pub status: SolverStatus,
    pub objective_value: f64,
    pub primal_values: Vec<f64>,
    pub dual_values: Vec<f64>,
    pub reduced_costs: Vec<f64>,
    pub basis_status: Vec<BasisStatus>,
    pub iterations: usize,
    pub time_seconds: f64,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn small_lp() -> LpProblem {
        // min -x1 - 2*x2  s.t. x1+x2 <= 4, x1 <= 3, x2 <= 3, x1,x2 >= 0
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

    #[test]
    fn test_lp_new() {
        let lp = LpProblem::new(false);
        assert_eq!(lp.num_vars(), 0);
        assert_eq!(lp.num_constraints(), 0);
        assert!(!lp.maximize);
    }

    #[test]
    fn test_add_variable() {
        let mut lp = LpProblem::new(false);
        let idx = lp.add_variable(1.0, 0.0, 10.0, Some("y".into()));
        assert_eq!(idx, 0);
        assert_eq!(lp.num_vars(), 1);
        assert_eq!(lp.var_names[0], "y");
    }

    #[test]
    fn test_add_constraint() {
        let mut lp = LpProblem::new(false);
        lp.add_variable(1.0, 0.0, f64::INFINITY, None);
        lp.add_variable(2.0, 0.0, f64::INFINITY, None);
        lp.add_constraint(&[0, 1], &[3.0, 4.0], ConstraintType::Le, 10.0)
            .unwrap();
        assert_eq!(lp.num_constraints(), 1);
        assert_eq!(lp.values, vec![3.0, 4.0]);
    }

    #[test]
    fn test_add_constraint_bad_index() {
        let mut lp = LpProblem::new(false);
        lp.add_variable(1.0, 0.0, 10.0, None);
        let res = lp.add_constraint(&[5], &[1.0], ConstraintType::Le, 1.0);
        assert!(res.is_err());
    }

    #[test]
    fn test_validate() {
        let lp = small_lp();
        assert!(lp.validate().is_ok());
    }

    #[test]
    fn test_to_standard_form() {
        let lp = small_lp();
        let (std, orig_n) = lp.to_standard_form().unwrap();
        assert_eq!(orig_n, 2);
        // 2 original + 3 slacks
        assert_eq!(std.num_vars(), 5);
        assert_eq!(std.num_constraints(), 3);
        for ct in &std.constraint_types {
            assert_eq!(*ct, ConstraintType::Eq);
        }
    }

    #[test]
    fn test_transpose_constraint_matrix() {
        let lp = small_lp();
        let (t_rs, t_ci, t_vals) = lp.transpose_constraint_matrix();
        // Variable 0 appears in rows 0 and 1
        let r0_start = t_rs[0];
        let r0_end = t_rs[1];
        assert_eq!(r0_end - r0_start, 2);
        // Variable 1 appears in rows 0 and 2
        let r1_start = t_rs[1];
        let r1_end = t_rs[2];
        assert_eq!(r1_end - r1_start, 2);
        assert_eq!(t_ci[r1_start], 0);
        assert_eq!(t_ci[r1_start + 1], 2);
        assert!((t_vals[r1_start] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_multiply_ax() {
        let lp = small_lp();
        let x = vec![1.0, 2.0];
        let mut out = vec![0.0; 3];
        lp.multiply_ax(&x, &mut out);
        assert!((out[0] - 3.0).abs() < 1e-12);
        assert!((out[1] - 1.0).abs() < 1e-12);
        assert!((out[2] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_multiply_atx() {
        let lp = small_lp();
        let y = vec![1.0, 1.0, 1.0];
        let mut out = vec![0.0; 2];
        lp.multiply_atx(&y, &mut out);
        // col 0: 1+1 = 2, col 1: 1+1 = 2
        assert!((out[0] - 2.0).abs() < 1e-12);
        assert!((out[1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_column_dense() {
        let lp = small_lp();
        let mut col = vec![0.0; 3];
        lp.column_dense(1, &mut col);
        assert!((col[0] - 1.0).abs() < 1e-12);
        assert!((col[1] - 0.0).abs() < 1e-12);
        assert!((col[2] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_maximize_standard_form_flips() {
        let mut lp = LpProblem::new(true);
        lp.add_variable(3.0, 0.0, f64::INFINITY, None);
        lp.add_variable(5.0, 0.0, f64::INFINITY, None);
        lp.add_constraint(&[0, 1], &[1.0, 1.0], ConstraintType::Le, 4.0)
            .unwrap();
        let (std, _) = lp.to_standard_form().unwrap();
        // maximise flipped to minimise: coeffs negated
        assert!((std.obj_coeffs[0] - (-3.0)).abs() < 1e-12);
        assert!((std.obj_coeffs[1] - (-5.0)).abs() < 1e-12);
    }
}
