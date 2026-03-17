//! Solver backend trait and implementations.
//!
//! SolverBackend trait, InternalSolver (uses simplex/IPM), configuration,
//! solve status, parameter management.

use crate::dual_simplex::{DualSimplex, DualSimplexConfig, DualSimplexResult};
use crate::interior_point::{self, InteriorPointConfig, InteriorPointResult};
use crate::model::LpModel;
use crate::presolve::{self, PresolveConfig, PresolveResult};
use crate::simplex::{RevisedSimplex, SimplexConfig, SimplexResult};
use crate::solution::{self, LpSolution};
use crate::tableau::PricingStrategy;
use bicut_types::{LpStatus, OptDirection};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::time::Instant;

/// Solver algorithm to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SolverAlgorithm {
    /// Primal simplex method.
    PrimalSimplex,
    /// Dual simplex method.
    DualSimplex,
    /// Interior point method.
    InteriorPoint,
    /// Automatic selection.
    Auto,
}

impl Default for SolverAlgorithm {
    fn default() -> Self {
        SolverAlgorithm::Auto
    }
}

/// Solver parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverParams {
    pub algorithm: SolverAlgorithm,
    pub max_iterations: usize,
    pub time_limit_seconds: f64,
    pub feasibility_tol: f64,
    pub optimality_tol: f64,
    pub pivot_tol: f64,
    pub presolve: bool,
    pub scaling: bool,
    pub verbose: bool,
    pub crossover: bool,
    pub extra: HashMap<String, String>,
}

impl Default for SolverParams {
    fn default() -> Self {
        Self {
            algorithm: SolverAlgorithm::Auto,
            max_iterations: 1_000_000,
            time_limit_seconds: 1e20,
            feasibility_tol: 1e-8,
            optimality_tol: 1e-8,
            pivot_tol: 1e-10,
            presolve: true,
            scaling: true,
            verbose: false,
            crossover: true,
            extra: HashMap::new(),
        }
    }
}

impl SolverParams {
    pub fn set(&mut self, key: &str, value: &str) {
        match key {
            "algorithm" => {
                self.algorithm = match value {
                    "primal" | "primal_simplex" => SolverAlgorithm::PrimalSimplex,
                    "dual" | "dual_simplex" => SolverAlgorithm::DualSimplex,
                    "ipm" | "interior_point" => SolverAlgorithm::InteriorPoint,
                    _ => SolverAlgorithm::Auto,
                };
            }
            "max_iterations" => {
                if let Ok(v) = value.parse() {
                    self.max_iterations = v;
                }
            }
            "time_limit" => {
                if let Ok(v) = value.parse() {
                    self.time_limit_seconds = v;
                }
            }
            "feasibility_tol" | "feas_tol" => {
                if let Ok(v) = value.parse() {
                    self.feasibility_tol = v;
                }
            }
            "optimality_tol" | "opt_tol" => {
                if let Ok(v) = value.parse() {
                    self.optimality_tol = v;
                }
            }
            "presolve" => self.presolve = value == "true" || value == "1",
            "scaling" => self.scaling = value == "true" || value == "1",
            "verbose" => self.verbose = value == "true" || value == "1",
            "crossover" => self.crossover = value == "true" || value == "1",
            _ => {
                self.extra.insert(key.to_string(), value.to_string());
            }
        }
    }

    pub fn get(&self, key: &str) -> Option<String> {
        match key {
            "algorithm" => Some(format!("{:?}", self.algorithm)),
            "max_iterations" => Some(self.max_iterations.to_string()),
            "time_limit" => Some(self.time_limit_seconds.to_string()),
            "feasibility_tol" => Some(self.feasibility_tol.to_string()),
            "optimality_tol" => Some(self.optimality_tol.to_string()),
            "presolve" => Some(self.presolve.to_string()),
            "verbose" => Some(self.verbose.to_string()),
            _ => self.extra.get(key).cloned(),
        }
    }
}

/// Trait for LP solver backends.
pub trait SolverBackend: fmt::Debug {
    /// Get the name of this solver backend.
    fn name(&self) -> &str;

    /// Solve an LP model, returning a complete solution.
    fn solve(&self, model: &LpModel, params: &SolverParams) -> LpSolution;

    /// Check if this backend supports MIP solving.
    fn supports_mip(&self) -> bool;

    /// Get default parameters for this backend.
    fn default_params(&self) -> SolverParams;

    /// Clone the backend as a boxed trait object.
    fn clone_box(&self) -> Box<dyn SolverBackend>;
}

impl Clone for Box<dyn SolverBackend> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// Internal solver using the built-in simplex and IPM implementations.
#[derive(Debug, Clone)]
pub struct InternalSolver {
    name: String,
}

impl InternalSolver {
    pub fn new() -> Self {
        Self {
            name: "BiCut-LP Internal".to_string(),
        }
    }

    /// Select algorithm based on problem characteristics.
    fn auto_select_algorithm(&self, model: &LpModel) -> SolverAlgorithm {
        let stats = model.stats();
        let n = stats.num_vars;
        let m = stats.num_constraints;

        // Heuristic: use IPM for large dense problems,
        // dual simplex for problems with many constraints,
        // primal simplex for others
        if n > 5000 && m > 5000 && stats.density > 0.1 {
            SolverAlgorithm::InteriorPoint
        } else if m > n * 2 {
            SolverAlgorithm::DualSimplex
        } else {
            SolverAlgorithm::PrimalSimplex
        }
    }

    fn solve_primal_simplex(&self, model: &LpModel, params: &SolverParams) -> LpSolution {
        let config = SimplexConfig {
            max_iterations: params.max_iterations,
            feasibility_tol: params.feasibility_tol,
            optimality_tol: params.optimality_tol,
            pivot_tol: params.pivot_tol,
            verbose: params.verbose,
            pricing: PricingStrategy::Dantzig,
            ..SimplexConfig::default()
        };

        let solver = RevisedSimplex::new(config);
        let result = solver.solve(model);

        solution::extract_solution(
            model,
            result.status,
            result.primal,
            result.dual,
            result.reduced_costs,
            &result.basis_indices,
            result.iterations,
            0.0,
        )
    }

    fn solve_dual_simplex(&self, model: &LpModel, params: &SolverParams) -> LpSolution {
        let config = DualSimplexConfig {
            max_iterations: params.max_iterations,
            feasibility_tol: params.feasibility_tol,
            optimality_tol: params.optimality_tol,
            pivot_tol: params.pivot_tol,
            verbose: params.verbose,
            ..DualSimplexConfig::default()
        };

        let solver = DualSimplex::new(config);
        let result = solver.solve(model);

        solution::extract_solution(
            model,
            result.status,
            result.primal,
            result.dual,
            result.reduced_costs,
            &result.basis_indices,
            result.iterations,
            0.0,
        )
    }

    fn solve_interior_point(&self, model: &LpModel, params: &SolverParams) -> LpSolution {
        let config = InteriorPointConfig {
            max_iterations: params.max_iterations.min(200),
            tolerance: params.optimality_tol,
            crossover: params.crossover,
            verbose: params.verbose,
            ..InteriorPointConfig::default()
        };

        let result = interior_point::solve_interior_point(model, &config);

        solution::extract_solution(
            model,
            result.status,
            result.primal,
            result.dual,
            result.reduced_costs,
            &[], // IPM doesn't produce a basis directly
            result.iterations,
            0.0,
        )
    }
}

impl Default for InternalSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl SolverBackend for InternalSolver {
    fn name(&self) -> &str {
        &self.name
    }

    fn solve(&self, model: &LpModel, params: &SolverParams) -> LpSolution {
        let start = Instant::now();

        // Presolve
        let (work_model, presolve_result) = if params.presolve {
            let ps_config = PresolveConfig::default();
            let ps_result = presolve::presolve(model, &ps_config);
            if ps_result.is_infeasible {
                return LpSolution::infeasible();
            }
            (ps_result.presolved_model.clone(), Some(ps_result))
        } else {
            (model.clone(), None)
        };

        // Scaling
        let mut scaled_model = work_model.clone();
        let scale_info = if params.scaling {
            let info = scaled_model.scale();
            Some(info)
        } else {
            None
        };

        // Select and run algorithm
        let algorithm = match params.algorithm {
            SolverAlgorithm::Auto => self.auto_select_algorithm(&scaled_model),
            other => other,
        };

        let mut sol = match algorithm {
            SolverAlgorithm::PrimalSimplex => self.solve_primal_simplex(&scaled_model, params),
            SolverAlgorithm::DualSimplex => self.solve_dual_simplex(&scaled_model, params),
            SolverAlgorithm::InteriorPoint => self.solve_interior_point(&scaled_model, params),
            SolverAlgorithm::Auto => self.solve_primal_simplex(&scaled_model, params),
        };

        // Unscale solution
        if let Some((row_scale, col_scale)) = &scale_info {
            sol.primal = work_model.unscale_primal(&sol.primal, col_scale);
            sol.dual = work_model.unscale_dual(&sol.dual, row_scale);
        }

        // Postsolve
        if let Some(ps_result) = &presolve_result {
            let (primal, dual) = presolve::postsolve(
                &sol.primal,
                &sol.dual,
                &ps_result.ops,
                &ps_result.var_map,
                model.num_vars(),
                model.num_constraints(),
            );
            sol.primal = primal;
            sol.dual = dual;
            // Recompute derived values
            sol.objective = solution::compute_objective(model, &sol.primal);
            sol.slacks = solution::compute_slacks(model, &sol.primal);
        }

        sol.solve_time = start.elapsed().as_secs_f64();
        sol
    }

    fn supports_mip(&self) -> bool {
        false
    }

    fn default_params(&self) -> SolverParams {
        SolverParams::default()
    }

    fn clone_box(&self) -> Box<dyn SolverBackend> {
        Box::new(self.clone())
    }
}

/// A solver facade that wraps a backend and manages parameters.
#[derive(Clone)]
pub struct Solver {
    backend: Box<dyn SolverBackend>,
    params: SolverParams,
}

impl fmt::Debug for Solver {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Solver")
            .field("backend", &self.backend.name())
            .field("params", &self.params)
            .finish()
    }
}

impl Solver {
    /// Create a solver with the internal backend.
    pub fn new() -> Self {
        let backend = InternalSolver::new();
        let params = backend.default_params();
        Self {
            backend: Box::new(backend),
            params,
        }
    }

    /// Create a solver with a specific backend.
    pub fn with_backend(backend: Box<dyn SolverBackend>) -> Self {
        let params = backend.default_params();
        Self { backend, params }
    }

    /// Set a parameter by name.
    pub fn set_param(&mut self, key: &str, value: &str) {
        self.params.set(key, value);
    }

    /// Get a parameter by name.
    pub fn get_param(&self, key: &str) -> Option<String> {
        self.params.get(key)
    }

    /// Set the algorithm.
    pub fn set_algorithm(&mut self, alg: SolverAlgorithm) {
        self.params.algorithm = alg;
    }

    /// Enable/disable presolve.
    pub fn set_presolve(&mut self, enabled: bool) {
        self.params.presolve = enabled;
    }

    /// Enable/disable verbose output.
    pub fn set_verbose(&mut self, enabled: bool) {
        self.params.verbose = enabled;
    }

    /// Set maximum iterations.
    pub fn set_max_iterations(&mut self, max_iter: usize) {
        self.params.max_iterations = max_iter;
    }

    /// Set feasibility tolerance.
    pub fn set_feasibility_tol(&mut self, tol: f64) {
        self.params.feasibility_tol = tol;
    }

    /// Set optimality tolerance.
    pub fn set_optimality_tol(&mut self, tol: f64) {
        self.params.optimality_tol = tol;
    }

    /// Get solver parameters.
    pub fn params(&self) -> &SolverParams {
        &self.params
    }

    /// Get mutable solver parameters.
    pub fn params_mut(&mut self) -> &mut SolverParams {
        &mut self.params
    }

    /// Solve an LP model.
    pub fn solve(&self, model: &LpModel) -> LpSolution {
        self.backend.solve(model, &self.params)
    }

    /// Solve and validate the solution.
    pub fn solve_and_validate(&self, model: &LpModel) -> (LpSolution, bool) {
        let sol = self.solve(model);
        if sol.status != LpStatus::Optimal {
            return (sol, false);
        }
        let validation =
            solution::validate_primal_feasibility(model, &sol.primal, self.params.feasibility_tol);
        (sol, validation.is_feasible)
    }

    /// Get the backend name.
    pub fn backend_name(&self) -> &str {
        self.backend.name()
    }
}

impl Default for Solver {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function: solve an LP model with default settings.
pub fn solve(model: &LpModel) -> LpSolution {
    let solver = Solver::new();
    solver.solve(model)
}

/// Convenience function: solve with specific algorithm.
pub fn solve_with(model: &LpModel, algorithm: SolverAlgorithm) -> LpSolution {
    let mut solver = Solver::new();
    solver.set_algorithm(algorithm);
    solver.solve(model)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Constraint, Variable};
    use bicut_types::ConstraintSense;

    fn make_test_model() -> LpModel {
        let mut m = LpModel::new("backend_test");
        m.sense = OptDirection::Minimize;
        let x = m.add_variable(Variable::continuous("x", 0.0, f64::INFINITY));
        let y = m.add_variable(Variable::continuous("y", 0.0, f64::INFINITY));
        m.set_obj_coeff(x, -1.0);
        m.set_obj_coeff(y, -1.0);

        let mut c0 = Constraint::new("c0", ConstraintSense::Le, 4.0);
        c0.add_term(x, 1.0);
        c0.add_term(y, 1.0);
        m.add_constraint(c0);

        let mut c1 = Constraint::new("c1", ConstraintSense::Le, 6.0);
        c1.add_term(x, 2.0);
        c1.add_term(y, 1.0);
        m.add_constraint(c1);

        m
    }

    #[test]
    fn test_solver_default() {
        let model = make_test_model();
        let sol = solve(&model);
        assert!(
            sol.status == LpStatus::Optimal || sol.status == LpStatus::IterationLimit,
            "status = {:?}",
            sol.status
        );
    }

    #[test]
    fn test_solver_primal_simplex() {
        let model = make_test_model();
        let sol = solve_with(&model, SolverAlgorithm::PrimalSimplex);
        assert!(sol.status == LpStatus::Optimal || sol.status == LpStatus::IterationLimit);
    }

    #[test]
    fn test_solver_dual_simplex() {
        let model = make_test_model();
        let sol = solve_with(&model, SolverAlgorithm::DualSimplex);
        assert!(sol.status == LpStatus::Optimal || sol.status == LpStatus::IterationLimit);
    }

    #[test]
    fn test_solver_ipm() {
        let model = make_test_model();
        let sol = solve_with(&model, SolverAlgorithm::InteriorPoint);
        assert!(sol.status == LpStatus::Optimal || sol.status == LpStatus::IterationLimit);
    }

    #[test]
    fn test_solver_params() {
        let mut solver = Solver::new();
        solver.set_param("max_iterations", "500");
        solver.set_param("verbose", "false");
        solver.set_param("presolve", "false");
        assert_eq!(solver.params.max_iterations, 500);
        assert!(!solver.params.verbose);
        assert!(!solver.params.presolve);
    }

    #[test]
    fn test_solver_get_param() {
        let solver = Solver::new();
        assert!(solver.get_param("max_iterations").is_some());
        assert!(solver.get_param("algorithm").is_some());
    }

    #[test]
    fn test_solve_and_validate() {
        let model = make_test_model();
        let mut solver = Solver::new();
        solver.set_presolve(false);
        let (sol, valid) = solver.solve_and_validate(&model);
        if sol.status == LpStatus::Optimal {
            assert!(valid);
        }
    }

    #[test]
    fn test_internal_solver_name() {
        let solver = InternalSolver::new();
        assert_eq!(solver.name(), "BiCut-LP Internal");
    }

    #[test]
    fn test_solver_clone() {
        let solver = Solver::new();
        let cloned = solver.clone();
        assert_eq!(cloned.backend_name(), solver.backend_name());
    }

    #[test]
    fn test_auto_select() {
        let solver = InternalSolver::new();
        let model = make_test_model();
        let alg = solver.auto_select_algorithm(&model);
        assert_eq!(alg, SolverAlgorithm::PrimalSimplex);
    }

    #[test]
    fn test_solver_no_presolve() {
        let model = make_test_model();
        let mut solver = Solver::new();
        solver.set_presolve(false);
        let sol = solver.solve(&model);
        assert!(sol.status == LpStatus::Optimal || sol.status == LpStatus::IterationLimit);
    }
}
