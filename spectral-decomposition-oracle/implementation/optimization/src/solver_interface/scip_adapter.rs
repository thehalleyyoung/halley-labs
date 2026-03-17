//! SCIP solver adapter (mock/emulation for standalone use).
//!
//! Emulates SCIP-like behavior for testing and standalone usage without
//! requiring actual SCIP C bindings.

use crate::error::{OptError, OptResult};
use crate::lp::{BasisStatus, ConstraintType, LpProblem, LpSolution, SolverStatus};
use crate::solver_interface::{SolverConfig, SolverInterface};
use indexmap::IndexMap;
use log::{debug, info};
use std::time::Instant;

/// Statistics from a SCIP emulation solve.
#[derive(Debug, Clone, Default)]
pub struct ScipStats {
    pub nodes_explored: usize,
    pub lp_iterations: usize,
    pub cuts_applied: usize,
    pub presolve_time: f64,
    pub solve_time: f64,
}

/// SCIP solver mock adapter.
pub struct ScipAdapter {
    config: SolverConfig,
    problem: Option<LpProblem>,
    last_solution: Option<LpSolution>,
    status: SolverStatus,
    parameters: IndexMap<String, f64>,
    stats: ScipStats,
    partition: Option<Vec<usize>>,
}

impl ScipAdapter {
    pub fn new(config: SolverConfig) -> Self {
        Self {
            config,
            problem: None,
            last_solution: None,
            status: SolverStatus::NumericalError,
            parameters: Self::default_parameters(),
            stats: ScipStats::default(),
            partition: None,
        }
    }

    /// Set a SCIP parameter.
    pub fn set_parameter(&mut self, name: &str, value: f64) {
        self.parameters.insert(name.to_string(), value);
        match name {
            "limits/time" => self.config.time_limit = value,
            "limits/gap" => self.config.gap_tolerance = value,
            _ => {}
        }
    }

    /// Get a SCIP parameter value.
    pub fn get_parameter(&self, name: &str) -> Option<f64> {
        self.parameters.get(name).copied()
    }

    /// Set variable partition for Benders decomposition.
    pub fn set_partition(&mut self, partition: &[usize]) {
        self.partition = Some(partition.to_vec());
    }

    /// Get solve statistics.
    pub fn get_stats(&self) -> &ScipStats {
        &self.stats
    }

    /// Default SCIP-like parameters.
    pub fn default_parameters() -> IndexMap<String, f64> {
        let mut params = IndexMap::new();
        params.insert("limits/time".to_string(), 3600.0);
        params.insert("limits/gap".to_string(), 1e-4);
        params.insert("display/verblevel".to_string(), 4.0);
        params.insert("presolving/maxrounds".to_string(), -1.0);
        params.insert("separating/maxroundsroot".to_string(), -1.0);
        params.insert("lp/threads".to_string(), 1.0);
        params.insert("numerics/epsilon".to_string(), 1e-9);
        params.insert("numerics/feastol".to_string(), 1e-6);
        params.insert("numerics/dualfeastol".to_string(), 1e-7);
        params.insert("branching/scoreparam".to_string(), 0.167);
        params.insert("heuristics/feaspump/freq".to_string(), 20.0);
        params.insert("conflict/enable".to_string(), 1.0);
        params
    }

    /// Simple presolve: remove fixed variables and tighten bounds.
    fn emulate_presolve(&self, problem: &LpProblem) -> LpProblem {
        let start = Instant::now();
        let mut reduced = problem.clone();
        let n = reduced.num_vars;

        // Fix variables with equal bounds
        for i in 0..n {
            if (reduced.lower_bounds[i] - reduced.upper_bounds[i]).abs() < 1e-10 {
                debug!("SCIP presolve: fixing var {} to {}", i, reduced.lower_bounds[i]);
            }
        }

        // Simple bound tightening from constraints
        for ci in 0..reduced.num_constraints {
            let row_start = reduced.row_starts[ci];
            let row_end = if ci + 1 < reduced.row_starts.len() {
                reduced.row_starts[ci + 1]
            } else {
                reduced.col_indices.len()
            };

            // Single-variable constraints
            let vars_in_row: Vec<(usize, f64)> = (row_start..row_end)
                .filter_map(|idx| {
                    if idx < reduced.col_indices.len() {
                        Some((reduced.col_indices[idx], reduced.values[idx]))
                    } else {
                        None
                    }
                })
                .collect();

            if vars_in_row.len() == 1 {
                let (var, coeff) = vars_in_row[0];
                let rhs = reduced.rhs[ci];
                if coeff.abs() > 1e-10 {
                    let bound = rhs / coeff;
                    match reduced.constraint_types.get(ci) {
                        Some(ConstraintType::Le) => {
                            if coeff > 0.0 {
                                reduced.upper_bounds[var] =
                                    reduced.upper_bounds[var].min(bound);
                            } else {
                                reduced.lower_bounds[var] =
                                    reduced.lower_bounds[var].max(bound);
                            }
                        }
                        Some(ConstraintType::Ge) => {
                            if coeff > 0.0 {
                                reduced.lower_bounds[var] =
                                    reduced.lower_bounds[var].max(bound);
                            } else {
                                reduced.upper_bounds[var] =
                                    reduced.upper_bounds[var].min(bound);
                            }
                        }
                        Some(ConstraintType::Eq) => {
                            reduced.lower_bounds[var] = bound;
                            reduced.upper_bounds[var] = bound;
                        }
                        None => {}
                    }
                }
            }
        }

        debug!("SCIP presolve: {:.4}s", start.elapsed().as_secs_f64());
        reduced
    }

    /// Emulate SCIP Benders decomposition.
    pub fn emulate_benders(&mut self, problem: &LpProblem) -> OptResult<LpSolution> {
        info!("SCIP: emulating Benders decomposition");

        if let Some(ref partition) = self.partition {
            // Use internal Benders
            let config = crate::benders::BendersConfig::default();
            let mut benders =
                crate::benders::decomposition::BendersDecomposition::new(problem, partition, config)?;
            let result = benders.solve()?;

            Ok(LpSolution {
                status: match result.status {
                    crate::benders::BendersStatus::Optimal
                    | crate::benders::BendersStatus::GapClosed => SolverStatus::Optimal,
                    crate::benders::BendersStatus::Infeasible => SolverStatus::Infeasible,
                    crate::benders::BendersStatus::TimeLimit => SolverStatus::TimeLimit,
                    crate::benders::BendersStatus::IterationLimit => SolverStatus::IterationLimit,
                    crate::benders::BendersStatus::NumericalError => SolverStatus::NumericalError,
                },
                objective_value: result.upper_bound,
                primal_values: result.master_solution,
                dual_values: Vec::new(),
                reduced_costs: Vec::new(),
                basis_status: Vec::new(),
                iterations: result.iterations,
                time_seconds: result.time_seconds,
            })
        } else {
            // No partition, just solve directly
            self.solve_internal(problem)
        }
    }

    /// Internal LP solve using unified solver.
    fn solve_internal(&self, problem: &LpProblem) -> OptResult<LpSolution> {
        let mut solver =
            crate::solver_interface::unified::UnifiedSolver::new(self.config.clone());
        solver.solve_lp(problem)
    }
}

impl SolverInterface for ScipAdapter {
    fn solve_lp(&mut self, problem: &LpProblem) -> OptResult<LpSolution> {
        let start = Instant::now();
        self.problem = Some(problem.clone());

        // Presolve
        let presolve_start = Instant::now();
        let reduced = self.emulate_presolve(problem);
        self.stats.presolve_time = presolve_start.elapsed().as_secs_f64();

        // Solve
        let result = self.solve_internal(&reduced)?;

        self.stats.solve_time = start.elapsed().as_secs_f64();
        self.stats.lp_iterations = result.iterations;
        self.stats.nodes_explored = 1;

        self.status = result.status;
        self.last_solution = Some(result.clone());

        info!(
            "SCIP: solved in {:.4}s, {} iterations, status={:?}",
            self.stats.solve_time, self.stats.lp_iterations, self.status
        );

        Ok(result)
    }

    fn get_status(&self) -> SolverStatus {
        self.status
    }

    fn get_dual_values(&self) -> OptResult<Vec<f64>> {
        self.last_solution
            .as_ref()
            .map(|s| s.dual_values.clone())
            .ok_or_else(|| OptError::solver("No solution available"))
    }

    fn get_basis(&self) -> OptResult<Vec<BasisStatus>> {
        self.last_solution
            .as_ref()
            .map(|s| s.basis_status.clone())
            .ok_or_else(|| OptError::solver("No solution available"))
    }

    fn add_constraint(
        &mut self,
        coeffs: &[(usize, f64)],
        ctype: ConstraintType,
        rhs: f64,
    ) -> OptResult<usize> {
        let problem = self
            .problem
            .as_mut()
            .ok_or_else(|| OptError::solver("No problem loaded"))?;
        let idx = problem.num_constraints;
        let (indices, vals): (Vec<usize>, Vec<f64>) = coeffs.iter().copied().unzip();
        problem.add_constraint(&indices, &vals, ctype, rhs)?;
        Ok(idx)
    }

    fn add_variable(&mut self, obj: f64, lb: f64, ub: f64) -> OptResult<usize> {
        let problem = self
            .problem
            .as_mut()
            .ok_or_else(|| OptError::solver("No problem loaded"))?;
        let idx = problem.num_vars;
        problem.add_variable(obj, lb, ub, None);
        Ok(idx)
    }

    fn set_objective(&mut self, coeffs: &[(usize, f64)]) -> OptResult<()> {
        let problem = self
            .problem
            .as_mut()
            .ok_or_else(|| OptError::solver("No problem loaded"))?;
        for &(i, val) in coeffs {
            if i < problem.obj_coeffs.len() {
                problem.obj_coeffs[i] = val;
            }
        }
        Ok(())
    }

    fn set_time_limit(&mut self, seconds: f64) {
        self.config.time_limit = seconds;
        self.parameters
            .insert("limits/time".to_string(), seconds);
    }

    fn name(&self) -> &str {
        "SCIP-Emulation"
    }

    fn reset(&mut self) {
        self.problem = None;
        self.last_solution = None;
        self.status = SolverStatus::NumericalError;
        self.stats = ScipStats::default();
        self.partition = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_lp() -> LpProblem {
        let mut lp = LpProblem::new(false);
        lp.obj_coeffs = vec![1.0, 2.0];
        lp.lower_bounds = vec![0.0, 0.0];
        lp.upper_bounds = vec![5.0, 5.0];
        lp.row_starts = vec![0, 2];
        lp.col_indices = vec![0, 1];
        lp.values = vec![1.0, 1.0];
        lp.constraint_types = vec![ConstraintType::Le];
        lp.rhs = vec![4.0];
        lp.num_constraints = 1;
        lp
    }

    #[test]
    fn test_scip_creation() {
        let config = SolverConfig::default().with_type(SolverType::ScipEmulation);
        let adapter = ScipAdapter::new(config);
        assert_eq!(adapter.name(), "SCIP-Emulation");
    }

    #[test]
    fn test_scip_default_parameters() {
        let params = ScipAdapter::default_parameters();
        assert!(params.contains_key("limits/time"));
        assert!(params.contains_key("limits/gap"));
    }

    #[test]
    fn test_scip_set_parameter() {
        let config = SolverConfig::default();
        let mut adapter = ScipAdapter::new(config);
        adapter.set_parameter("limits/time", 120.0);
        assert_eq!(adapter.get_parameter("limits/time"), Some(120.0));
        assert!((adapter.config.time_limit - 120.0).abs() < 1e-10);
    }

    #[test]
    fn test_scip_solve() {
        let config = SolverConfig::default().with_type(SolverType::ScipEmulation);
        let mut adapter = ScipAdapter::new(config);
        let lp = make_test_lp();
        let result = adapter.solve_lp(&lp);
        assert!(result.is_ok());
    }

    #[test]
    fn test_scip_stats() {
        let config = SolverConfig::default();
        let mut adapter = ScipAdapter::new(config);
        let lp = make_test_lp();
        adapter.solve_lp(&lp).unwrap();
        let stats = adapter.get_stats();
        assert!(stats.solve_time >= 0.0);
        assert_eq!(stats.nodes_explored, 1);
    }

    #[test]
    fn test_scip_presolve() {
        let config = SolverConfig::default();
        let adapter = ScipAdapter::new(config);
        let lp = make_test_lp();
        let reduced = adapter.emulate_presolve(&lp);
        assert_eq!(reduced.num_vars, lp.num_vars);
    }

    #[test]
    fn test_scip_set_partition() {
        let config = SolverConfig::default();
        let mut adapter = ScipAdapter::new(config);
        adapter.set_partition(&[0, 1, 0, 1]);
        assert!(adapter.partition.is_some());
    }

    #[test]
    fn test_scip_reset() {
        let config = SolverConfig::default();
        let mut adapter = ScipAdapter::new(config);
        let lp = make_test_lp();
        adapter.solve_lp(&lp).unwrap();
        adapter.reset();
        assert!(adapter.get_dual_values().is_err());
        assert!(adapter.partition.is_none());
    }

    #[test]
    fn test_scip_get_status() {
        let config = SolverConfig::default();
        let mut adapter = ScipAdapter::new(config);
        let lp = make_test_lp();
        adapter.solve_lp(&lp).unwrap();
        let status = adapter.get_status();
        assert!(matches!(
            status,
            SolverStatus::Optimal | SolverStatus::Infeasible
        ));
    }

    #[test]
    fn test_scip_time_limit() {
        let config = SolverConfig::default();
        let mut adapter = ScipAdapter::new(config);
        adapter.set_time_limit(30.0);
        assert!((adapter.config.time_limit - 30.0).abs() < 1e-10);
        assert_eq!(adapter.get_parameter("limits/time"), Some(30.0));
    }

    #[test]
    fn test_scip_unknown_parameter() {
        let config = SolverConfig::default();
        let adapter = ScipAdapter::new(config);
        assert_eq!(adapter.get_parameter("nonexistent/param"), None);
    }
}
