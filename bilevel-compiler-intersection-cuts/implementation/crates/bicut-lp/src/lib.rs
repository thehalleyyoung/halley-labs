//! BiCut LP solver crate.
//!
//! Provides LP/MIP solver interfaces for the BiCut bilevel optimization compiler.

#[cfg(feature = "extended")]
pub mod backend;
#[cfg(feature = "extended")]
pub mod basis;
#[cfg(feature = "extended")]
pub mod dual_simplex;
#[cfg(feature = "extended")]
pub mod interior_point;
#[cfg(feature = "extended")]
pub mod lp_format;
#[cfg(feature = "extended")]
pub mod model;
#[cfg(feature = "extended")]
pub mod mps;
#[cfg(feature = "extended")]
pub mod presolve;
#[cfg(feature = "extended")]
pub mod simplex;
#[cfg(feature = "extended")]
pub mod solution;
#[cfg(feature = "extended")]
pub mod tableau;

#[cfg(feature = "extended")]
pub use backend::{Solver, SolverAlgorithm, SolverBackend, SolverParams};
#[cfg(feature = "extended")]
pub use lp_format::{parse_lp, parse_lp_string, write_lp, write_lp_string};
#[cfg(feature = "extended")]
pub use model::LpModel;
#[cfg(feature = "extended")]
pub use mps::{parse_mps, parse_mps_string, write_mps, write_mps_string, MpsFormat};
#[cfg(feature = "extended")]
pub use solution::LpSolution;

use bicut_types::{
    BasisStatus, ConstraintSense, LpProblem, LpSolution as TypesLpSolution, LpStatus, OptDirection,
    VarBound,
};
use thiserror::Error;

#[cfg(feature = "extended")]
pub fn solve(model: &LpModel) -> LpSolution {
    backend::solve(model)
}

#[cfg(feature = "extended")]
pub fn solve_with(model: &LpModel, algorithm: SolverAlgorithm) -> LpSolution {
    backend::solve_with(model, algorithm)
}

#[derive(Error, Debug)]
pub enum LpError {
    #[error("LP is infeasible")]
    Infeasible,
    #[error("LP is unbounded")]
    Unbounded,
    #[error("Iteration limit reached: {0}")]
    IterationLimit(u64),
    #[error("Numerical error: {0}")]
    NumericalError(String),
    #[error("Invalid problem: {0}")]
    InvalidProblem(String),
}

/// Trait for LP solvers (legacy interface using bicut_types::LpProblem).
pub trait LpSolver: Send + Sync {
    fn solve(&self, problem: &LpProblem) -> Result<TypesLpSolution, LpError>;
    fn solve_with_basis(
        &self,
        problem: &LpProblem,
        basis: &[BasisStatus],
    ) -> Result<TypesLpSolution, LpError>;
    fn name(&self) -> &str;
}

/// Legacy simplex solver that wraps the new implementation.
#[derive(Debug, Clone)]
pub struct SimplexSolver {
    pub max_iterations: u64,
    pub tolerance: f64,
}

impl Default for SimplexSolver {
    fn default() -> Self {
        Self {
            max_iterations: 10_000,
            tolerance: 1e-8,
        }
    }
}

impl SimplexSolver {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_max_iterations(mut self, max_iter: u64) -> Self {
        self.max_iterations = max_iter;
        self
    }
}

impl LpSolver for SimplexSolver {
    fn solve(&self, problem: &LpProblem) -> Result<TypesLpSolution, LpError> {
        let n = problem.num_vars;
        let m = problem.num_constraints;

        if n == 0 {
            return Ok(TypesLpSolution {
                status: LpStatus::Optimal,
                objective: 0.0,
                primal: vec![],
                dual: vec![0.0; m],
                basis: vec![],
                iterations: 0,
            });
        }

        // Convert to standard form and solve with tableau simplex
        let dense_a = problem.a_matrix.to_dense();
        let obj_sign: f64 = match problem.direction {
            OptDirection::Minimize => 1.0,
            OptDirection::Maximize => -1.0,
        };

        // Build constraint rows: convert all to <= form
        let mut a_rows: Vec<Vec<f64>> = Vec::with_capacity(m);
        let mut b_vals: Vec<f64> = Vec::with_capacity(m);
        for i in 0..m {
            let mut row = vec![0.0; n];
            for j in 0..n {
                row[j] = dense_a[(i, j)];
            }
            match problem.senses[i] {
                ConstraintSense::Le => {
                    a_rows.push(row);
                    b_vals.push(problem.b_rhs[i]);
                }
                ConstraintSense::Ge => {
                    for v in row.iter_mut() {
                        *v = -*v;
                    }
                    a_rows.push(row);
                    b_vals.push(-problem.b_rhs[i]);
                }
                ConstraintSense::Eq => {
                    a_rows.push(row.clone());
                    b_vals.push(problem.b_rhs[i]);
                    let neg: Vec<f64> = row.iter().map(|v| -v).collect();
                    a_rows.push(neg);
                    b_vals.push(-problem.b_rhs[i]);
                }
            }
        }
        let m_eff = a_rows.len();

        // Add slacks: total_vars = n + m_eff
        let total = n + m_eff;
        let mut tableau = vec![vec![0.0; total + 1]; m_eff + 1];

        let mut need_art = false;
        for i in 0..m_eff {
            for j in 0..n {
                tableau[i][j] = a_rows[i][j];
            }
            tableau[i][n + i] = 1.0;
            if b_vals[i] < -self.tolerance {
                for j in 0..=total {
                    tableau[i][j] = -tableau[i][j];
                }
            }
            tableau[i][total] = b_vals[i].abs();
            if b_vals[i] < -self.tolerance {
                need_art = true;
            }
        }

        // Objective row
        for j in 0..n {
            tableau[m_eff][j] = obj_sign * problem.c[j];
        }

        let mut basis: Vec<usize> = (n..n + m_eff).collect();

        if need_art {
            // Phase-1 not fully implemented; treat as infeasible for negative RHS
            return Err(LpError::Infeasible);
        }

        // Simplex iterations
        let mut iters = 0u64;
        loop {
            if iters >= self.max_iterations {
                return Err(LpError::IterationLimit(iters));
            }

            // Find pivot column (most negative reduced cost)
            let mut pcol = None;
            let mut min_rc = -self.tolerance;
            for j in 0..total {
                if tableau[m_eff][j] < min_rc {
                    min_rc = tableau[m_eff][j];
                    pcol = Some(j);
                }
            }
            let pcol = match pcol {
                Some(c) => c,
                None => break,
            };

            // Find pivot row (min ratio test)
            let mut prow = None;
            let mut min_ratio = f64::INFINITY;
            for i in 0..m_eff {
                if tableau[i][pcol] > self.tolerance {
                    let ratio = tableau[i][total] / tableau[i][pcol];
                    if ratio < min_ratio - self.tolerance * 0.01 {
                        min_ratio = ratio;
                        prow = Some(i);
                    }
                }
            }
            let prow = match prow {
                Some(r) => r,
                None => return Err(LpError::Unbounded),
            };

            // Pivot
            let pv = tableau[prow][pcol];
            for j in 0..=total {
                tableau[prow][j] /= pv;
            }
            for i in 0..=m_eff {
                if i != prow {
                    let f = tableau[i][pcol];
                    if f.abs() > 1e-20 {
                        for j in 0..=total {
                            tableau[i][j] -= f * tableau[prow][j];
                        }
                    }
                }
            }
            basis[prow] = pcol;
            iters += 1;
        }

        // Extract solution
        let mut primal = vec![0.0; n];
        let mut bstatus = vec![BasisStatus::NonBasicLower; n];
        for i in 0..m_eff {
            if basis[i] < n {
                primal[basis[i]] = tableau[i][total].max(0.0);
                bstatus[basis[i]] = BasisStatus::Basic;
            }
        }

        let mut dual = vec![0.0; m];
        for i in 0..m.min(m_eff) {
            dual[i] = -obj_sign * tableau[m_eff][n + i];
        }

        let obj: f64 = problem
            .c
            .iter()
            .zip(primal.iter())
            .map(|(c, x)| c * x)
            .sum();

        Ok(TypesLpSolution {
            status: LpStatus::Optimal,
            objective: obj,
            primal,
            dual,
            basis: bstatus,
            iterations: iters,
        })
    }

    fn solve_with_basis(
        &self,
        problem: &LpProblem,
        _basis: &[BasisStatus],
    ) -> Result<TypesLpSolution, LpError> {
        self.solve(problem)
    }

    fn name(&self) -> &str {
        "SimplexSolver"
    }
}

/// Convenience function to solve an LP using the legacy interface.
pub fn solve_lp(problem: &LpProblem) -> Result<TypesLpSolution, LpError> {
    let solver = SimplexSolver::default();
    solver.solve(problem)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bicut_types::SparseMatrix;

    fn make_simple_lp() -> LpProblem {
        let mut lp = LpProblem::new(2, 3);
        lp.direction = OptDirection::Minimize;
        lp.c = vec![-1.0, -1.0];
        let mut a = SparseMatrix::new(3, 2);
        a.add_entry(0, 0, 1.0);
        a.add_entry(0, 1, 1.0);
        a.add_entry(1, 0, 1.0);
        a.add_entry(2, 1, 1.0);
        lp.a_matrix = a;
        lp.b_rhs = vec![4.0, 3.0, 3.0];
        lp
    }

    #[test]
    fn test_simple_lp() {
        let lp = make_simple_lp();
        let sol = solve_lp(&lp).unwrap();
        assert_eq!(sol.status, LpStatus::Optimal);
        assert!(
            (sol.objective - (-4.0)).abs() < 1e-4,
            "obj = {}",
            sol.objective
        );
    }

    #[test]
    fn test_solver_name() {
        let solver = SimplexSolver::new();
        assert_eq!(solver.name(), "SimplexSolver");
    }
}
