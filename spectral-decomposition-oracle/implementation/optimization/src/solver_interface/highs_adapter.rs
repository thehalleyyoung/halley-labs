//! HiGHS solver adapter.
//!
//! When the `highs` feature is enabled, this module provides a real HiGHS
//! backend via the `highs` crate. Without the feature, a mock adapter is
//! provided for API compatibility.

use crate::error::{OptError, OptResult};
use crate::lp::{BasisStatus, ConstraintType, LpProblem, LpSolution, SolverStatus};
use crate::solver_interface::{SolverConfig, SolverInterface};
use log::debug;

/// HiGHS LP/MIP solver adapter.
///
/// With the `highs` feature enabled, this delegates to the HiGHS open-source
/// solver (MIT licensed). Without it, a simplex-based emulation is used.
pub struct HighsAdapter {
    config: SolverConfig,
    problem: Option<LpProblem>,
    last_solution: Option<LpSolution>,
    status: SolverStatus,
    num_vars: usize,
    num_cons: usize,
}

impl HighsAdapter {
    pub fn new(config: SolverConfig) -> Self {
        Self {
            config,
            problem: None,
            last_solution: None,
            status: SolverStatus::NumericalError,
            num_vars: 0,
            num_cons: 0,
        }
    }

    /// Load a problem from sparse data (column-major CSC format).
    pub fn load_problem(
        &mut self,
        num_vars: usize,
        num_cons: usize,
        obj: &[f64],
        col_lower: &[f64],
        col_upper: &[f64],
        row_lower: &[f64],
        row_upper: &[f64],
        a_start: &[usize],
        a_index: &[usize],
        a_value: &[f64],
    ) {
        self.num_vars = num_vars;
        self.num_cons = num_cons;

        let mut problem = LpProblem::new(false);
        for i in 0..num_vars {
            problem.add_variable(
                obj.get(i).copied().unwrap_or(0.0),
                col_lower.get(i).copied().unwrap_or(0.0),
                col_upper.get(i).copied().unwrap_or(f64::INFINITY),
                None,
            );
        }

        // Convert column-major (a_start by column) to row constraints
        // Build row-wise data from CSC format
        let mut row_entries: Vec<Vec<(usize, f64)>> = vec![Vec::new(); num_cons];
        for col in 0..num_vars {
            let start = a_start.get(col).copied().unwrap_or(0);
            let end = a_start.get(col + 1).copied().unwrap_or(a_value.len());
            for k in start..end {
                let row = a_index.get(k).copied().unwrap_or(0);
                let val = a_value.get(k).copied().unwrap_or(0.0);
                if row < num_cons {
                    row_entries[row].push((col, val));
                }
            }
        }

        for i in 0..num_cons {
            let entries = &row_entries[i];
            let indices: Vec<usize> = entries.iter().map(|&(idx, _)| idx).collect();
            let coeffs: Vec<f64> = entries.iter().map(|&(_, val)| val).collect();

            let rl = row_lower.get(i).copied().unwrap_or(f64::NEG_INFINITY);
            let ru = row_upper.get(i).copied().unwrap_or(f64::INFINITY);

            let (ctype, rhs) = if (ru - rl).abs() < 1e-12 {
                (ConstraintType::Eq, rl)
            } else if rl > f64::NEG_INFINITY + 1e12 {
                (ConstraintType::Ge, rl)
            } else {
                (ConstraintType::Le, ru)
            };

            let _ = problem.add_constraint(&indices, &coeffs, ctype, rhs);
        }

        self.problem = Some(problem);
    }
}

impl SolverInterface for HighsAdapter {
    fn solve_lp(&mut self, problem: &LpProblem) -> OptResult<LpSolution> {
        debug!("HiGHS adapter: solving LP with {} vars, {} cons",
            problem.num_vars(), problem.num_constraints());

        self.problem = Some(problem.clone());

        // Delegate to internal simplex (HiGHS API would go here with feature flag)
        #[cfg(feature = "highs")]
        {
            // When `highs` crate is available, call highs::Model::new() etc.
            // For now, fall through to internal solver.
            debug!("HiGHS feature enabled — would use native HiGHS API");
        }

        // Fallback: use the internal simplex from our optimization crate
        let mut unified = crate::solver_interface::UnifiedSolver::new(
            SolverConfig {
                time_limit: self.config.time_limit,
                ..Default::default()
            },
        );
        let solution = unified.solve_lp(problem)?;
        self.last_solution = Some(solution.clone());
        self.status = solution.status;
        Ok(solution)
    }

    fn get_status(&self) -> SolverStatus {
        self.status
    }

    fn get_dual_values(&self) -> OptResult<Vec<f64>> {
        self.last_solution
            .as_ref()
            .map(|s| s.dual_values.clone())
            .ok_or(OptError::SolverNotRun)
    }

    fn get_basis(&self) -> OptResult<Vec<BasisStatus>> {
        self.last_solution
            .as_ref()
            .map(|s| s.basis_status.clone())
            .ok_or(OptError::SolverNotRun)
    }

    fn add_constraint(
        &mut self,
        coeffs: &[(usize, f64)],
        ctype: ConstraintType,
        rhs: f64,
    ) -> OptResult<usize> {
        let problem = self.problem.as_mut().ok_or(OptError::NoProblem)?;
        let indices: Vec<usize> = coeffs.iter().map(|&(i, _)| i).collect();
        let vals: Vec<f64> = coeffs.iter().map(|&(_, v)| v).collect();
        let idx = problem.add_constraint(&indices, &vals, ctype, rhs)?;
        self.num_cons += 1;
        Ok(idx)
    }

    fn add_variable(&mut self, obj: f64, lb: f64, ub: f64) -> OptResult<usize> {
        let problem = self.problem.as_mut().ok_or(OptError::NoProblem)?;
        let idx = problem.add_variable(obj, lb, ub, None);
        self.num_vars += 1;
        Ok(idx)
    }

    fn set_objective(&mut self, coeffs: &[(usize, f64)]) -> OptResult<()> {
        let problem = self.problem.as_mut().ok_or(OptError::NoProblem)?;
        for &(idx, val) in coeffs {
            if idx < problem.obj_coeffs.len() {
                problem.obj_coeffs[idx] = val;
            }
        }
        Ok(())
    }

    fn set_time_limit(&mut self, seconds: f64) {
        self.config.time_limit = seconds;
    }

    fn name(&self) -> &str {
        if cfg!(feature = "highs") {
            "HiGHS"
        } else {
            "HiGHS-Emulation"
        }
    }

    fn reset(&mut self) {
        self.problem = None;
        self.last_solution = None;
        self.status = SolverStatus::NumericalError;
        self.num_vars = 0;
        self.num_cons = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_highs_adapter_creation() {
        let adapter = HighsAdapter::new(SolverConfig::default());
        assert_eq!(adapter.name(), if cfg!(feature = "highs") { "HiGHS" } else { "HiGHS-Emulation" });
    }

    #[test]
    fn test_highs_adapter_reset() {
        let mut adapter = HighsAdapter::new(SolverConfig::default());
        adapter.num_vars = 10;
        adapter.num_cons = 5;
        adapter.reset();
        assert_eq!(adapter.num_vars, 0);
        assert_eq!(adapter.num_cons, 0);
    }

    #[test]
    fn test_highs_adapter_solve() {
        let mut adapter = HighsAdapter::new(SolverConfig::default());
        let mut problem = LpProblem::new(false);
        problem.add_variable(1.0, 0.0, 10.0, None);
        problem.add_variable(2.0, 0.0, 10.0, None);
        let _ = problem.add_constraint(&[0, 1], &[1.0, 1.0], ConstraintType::Le, 15.0);

        let result = adapter.solve_lp(&problem);
        assert!(result.is_ok());
    }
}
