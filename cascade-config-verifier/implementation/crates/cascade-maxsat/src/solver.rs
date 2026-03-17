//! MaxSAT solvers: Fu-Malik, linear search, WBO, plus a basic SAT oracle.
//!
//! All solvers implement [`MaxSatSolver`] and can be used interchangeably.

use serde::{Deserialize, Serialize};

use crate::formula::{Literal, MaxSatFormula};

// ---------------------------------------------------------------------------
// Result / statistics types
// ---------------------------------------------------------------------------

/// A truth assignment for every variable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    /// Indexed by variable (0-based): `assignments[v]` is `true` when variable
    /// `v+1` is assigned positively.
    pub assignments: Vec<bool>,
}

impl Model {
    pub fn value(&self, var: u32) -> bool {
        self.assignments
            .get(var.saturating_sub(1) as usize)
            .copied()
            .unwrap_or(false)
    }

    pub fn eval_literal(&self, lit: Literal) -> bool {
        if lit > 0 {
            self.value(lit as u32)
        } else {
            !self.value((-lit) as u32)
        }
    }

    pub fn eval_clause(&self, clause: &[Literal]) -> bool {
        clause.iter().any(|&l| self.eval_literal(l))
    }
}

/// Outcome status of a MaxSAT solver run.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum MaxSatStatus {
    Optimal,
    Satisfiable,
    Unsatisfiable,
    Timeout,
    Unknown,
}

/// Complete result returned by a solver.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaxSatResult {
    pub status: MaxSatStatus,
    pub model: Option<Model>,
    pub cost: u64,
    pub statistics: SolverStatistics,
}

/// Runtime statistics.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SolverStatistics {
    pub solver_calls: u32,
    pub time_ms: u64,
    pub variables: u32,
    pub clauses: u32,
    pub iterations: u32,
}

// ---------------------------------------------------------------------------
// Configuration / strategy
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SolverStrategy {
    FuMalik,
    LinearSearch,
    Wbo,
    Adaptive,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverConfig {
    pub strategy: SolverStrategy,
    pub timeout_ms: u64,
    pub max_iterations: u32,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            strategy: SolverStrategy::FuMalik,
            timeout_ms: 60_000,
            max_iterations: 10_000,
        }
    }
}

// ---------------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------------

/// Common interface for MaxSAT solvers.
pub trait MaxSatSolver {
    fn solve(&self, formula: &MaxSatFormula) -> MaxSatResult;
    fn name(&self) -> &str;
}

// ---------------------------------------------------------------------------
// SAT oracle (DPLL-style)
// ---------------------------------------------------------------------------

/// A lightweight SAT oracle backed by a recursive DPLL search.
#[derive(Debug, Clone)]
pub struct SatOracle {
    pub max_decisions: u32,
}

impl SatOracle {
    pub fn new(max_decisions: u32) -> Self {
        Self { max_decisions }
    }

    pub fn check_sat(
        &self,
        clauses: &[Vec<Literal>],
        num_vars: u32,
    ) -> Option<Model> {
        let mut assignment = vec![None; num_vars as usize + 1];
        let mut decisions = 0u32;
        if self.dpll(clauses, &mut assignment, &mut decisions) {
            let assignments = (1..=num_vars as usize)
                .map(|i| assignment[i].unwrap_or(false))
                .collect();
            Some(Model { assignments })
        } else {
            None
        }
    }

    fn dpll(
        &self,
        clauses: &[Vec<Literal>],
        assignment: &mut [Option<bool>],
        decisions: &mut u32,
    ) -> bool {
        if *decisions > self.max_decisions {
            return false;
        }

        // Unit propagation
        loop {
            let mut unit_found = false;
            for clause in clauses {
                let mut unset = None;
                let mut satisfied = false;
                let mut unset_count = 0;

                for &lit in clause {
                    let var = lit.unsigned_abs() as usize;
                    match assignment[var] {
                        Some(val) => {
                            if (lit > 0 && val) || (lit < 0 && !val) {
                                satisfied = true;
                                break;
                            }
                        }
                        None => {
                            unset_count += 1;
                            unset = Some(lit);
                        }
                    }
                }

                if satisfied {
                    continue;
                }
                if unset_count == 0 {
                    return false;
                }
                if unset_count == 1 {
                    let lit = unset.unwrap();
                    let var = lit.unsigned_abs() as usize;
                    assignment[var] = Some(lit > 0);
                    unit_found = true;
                }
            }
            if !unit_found {
                break;
            }
        }

        // Check if all clauses are satisfied
        let all_sat = clauses.iter().all(|clause| {
            clause.iter().any(|&lit| {
                let var = lit.unsigned_abs() as usize;
                match assignment[var] {
                    Some(val) => (lit > 0 && val) || (lit < 0 && !val),
                    None => false,
                }
            })
        });
        if all_sat {
            return true;
        }

        // Pick an unassigned variable
        let pick = assignment
            .iter()
            .enumerate()
            .skip(1)
            .find(|(_, v)| v.is_none())
            .map(|(i, _)| i);

        let var = match pick {
            Some(v) => v,
            None => return false,
        };

        *decisions += 1;
        let saved: Vec<Option<bool>> = assignment.to_vec();

        assignment[var] = Some(true);
        if self.dpll(clauses, assignment, decisions) {
            return true;
        }

        assignment.copy_from_slice(&saved);
        assignment[var] = Some(false);
        self.dpll(clauses, assignment, decisions)
    }
}

/// A basic CDCL-inspired solver (simplified).
#[derive(Debug, Clone)]
pub struct CdclSolver {
    pub max_conflicts: u32,
}

impl CdclSolver {
    pub fn new(max_conflicts: u32) -> Self {
        Self { max_conflicts }
    }

    pub fn solve_sat(
        &self,
        clauses: &[Vec<Literal>],
        num_vars: u32,
    ) -> Option<Model> {
        let oracle = SatOracle::new(self.max_conflicts);
        oracle.check_sat(clauses, num_vars)
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn find_initial_model(formula: &MaxSatFormula, oracle: &SatOracle) -> Option<Model> {
    let hard_only: Vec<Vec<Literal>> = formula
        .hard_clauses
        .iter()
        .map(|c| c.literals.clone())
        .collect();
    oracle.check_sat(&hard_only, formula.num_variables)
}

fn compute_cost(formula: &MaxSatFormula, model: &Model) -> u64 {
    formula
        .soft_clauses
        .iter()
        .filter(|sc| !model.eval_clause(&sc.literals))
        .map(|sc| sc.weight)
        .sum()
}

// ---------------------------------------------------------------------------
// Fu-Malik solver
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct FuMalikSolver {
    pub config: SolverConfig,
}

impl FuMalikSolver {
    pub fn new(config: SolverConfig) -> Self {
        Self { config }
    }
}

impl MaxSatSolver for FuMalikSolver {
    fn solve(&self, formula: &MaxSatFormula) -> MaxSatResult {
        let start = std::time::Instant::now();
        let oracle = SatOracle::new(self.config.max_iterations * 100);
        let mut stats = SolverStatistics {
            variables: formula.num_variables,
            clauses: (formula.num_hard() + formula.num_soft()) as u32,
            ..Default::default()
        };

        let hard_clauses: Vec<Vec<Literal>> = formula
            .hard_clauses
            .iter()
            .map(|c| c.literals.clone())
            .collect();
        stats.solver_calls += 1;
        if oracle.check_sat(&hard_clauses, formula.num_variables).is_none() {
            stats.time_ms = start.elapsed().as_millis() as u64;
            return MaxSatResult {
                status: MaxSatStatus::Unsatisfiable,
                model: None,
                cost: u64::MAX,
                statistics: stats,
            };
        }

        let mut relaxation_vars: Vec<Literal> = Vec::new();
        let mut working_clauses = hard_clauses;
        let mut next_relax_var = formula.num_variables + 1;

        for sc in &formula.soft_clauses {
            let relax_var = next_relax_var as Literal;
            next_relax_var += 1;
            let mut cl = sc.literals.clone();
            cl.push(relax_var);
            working_clauses.push(cl);
            relaxation_vars.push(relax_var);
        }

        let mut best_model: Option<Model> = None;
        let mut best_cost = u64::MAX;

        for iter in 0..self.config.max_iterations {
            if start.elapsed().as_millis() as u64 > self.config.timeout_ms {
                stats.time_ms = start.elapsed().as_millis() as u64;
                stats.iterations = iter;
                return MaxSatResult {
                    status: if best_model.is_some() {
                        MaxSatStatus::Satisfiable
                    } else {
                        MaxSatStatus::Timeout
                    },
                    model: best_model,
                    cost: best_cost,
                    statistics: stats,
                };
            }

            stats.solver_calls += 1;
            if let Some(model) = oracle.check_sat(&working_clauses, next_relax_var - 1) {
                let cost = compute_cost(formula, &model);
                if cost < best_cost {
                    best_cost = cost;
                    best_model = Some(model);
                }
                stats.iterations = iter + 1;
                break;
            } else if let Some(&rv) = relaxation_vars.get(iter as usize) {
                working_clauses.push(vec![rv]);
            } else {
                break;
            }
        }

        stats.time_ms = start.elapsed().as_millis() as u64;
        if let Some(ref model) = best_model {
            let cost = compute_cost(formula, model);
            MaxSatResult {
                status: MaxSatStatus::Optimal,
                model: best_model,
                cost,
                statistics: stats,
            }
        } else {
            MaxSatResult {
                status: MaxSatStatus::Unknown,
                model: None,
                cost: u64::MAX,
                statistics: stats,
            }
        }
    }

    fn name(&self) -> &str {
        "FuMalik"
    }
}

// ---------------------------------------------------------------------------
// Linear search solver
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct LinearSearchSolver {
    pub config: SolverConfig,
}

impl LinearSearchSolver {
    pub fn new(config: SolverConfig) -> Self {
        Self { config }
    }
}

impl MaxSatSolver for LinearSearchSolver {
    fn solve(&self, formula: &MaxSatFormula) -> MaxSatResult {
        let start = std::time::Instant::now();
        let oracle = SatOracle::new(self.config.max_iterations * 100);
        let mut stats = SolverStatistics {
            variables: formula.num_variables,
            clauses: (formula.num_hard() + formula.num_soft()) as u32,
            ..Default::default()
        };

        let model = find_initial_model(formula, &oracle);
        stats.solver_calls += 1;
        let mut best_model = match model {
            Some(m) => m,
            None => {
                stats.time_ms = start.elapsed().as_millis() as u64;
                return MaxSatResult {
                    status: MaxSatStatus::Unsatisfiable,
                    model: None,
                    cost: u64::MAX,
                    statistics: stats,
                };
            }
        };
        let mut best_cost = compute_cost(formula, &best_model);

        for iter in 0..self.config.max_iterations {
            if start.elapsed().as_millis() as u64 > self.config.timeout_ms || best_cost == 0 {
                break;
            }

            let violated: Vec<Vec<Literal>> = formula
                .soft_clauses
                .iter()
                .filter(|sc| !best_model.eval_clause(&sc.literals))
                .map(|sc| sc.literals.clone())
                .collect();

            if violated.is_empty() {
                break;
            }

            let mut constrained: Vec<Vec<Literal>> = formula
                .hard_clauses
                .iter()
                .map(|c| c.literals.clone())
                .collect();
            for sc in &formula.soft_clauses {
                constrained.push(sc.literals.clone());
            }
            constrained.push(violated[0].clone());

            stats.solver_calls += 1;
            if let Some(m) = oracle.check_sat(&constrained, formula.num_variables) {
                let cost = compute_cost(formula, &m);
                if cost < best_cost {
                    best_cost = cost;
                    best_model = m;
                }
            } else {
                break;
            }

            stats.iterations = iter + 1;
        }

        stats.time_ms = start.elapsed().as_millis() as u64;
        MaxSatResult {
            status: MaxSatStatus::Optimal,
            model: Some(best_model),
            cost: best_cost,
            statistics: stats,
        }
    }

    fn name(&self) -> &str {
        "LinearSearch"
    }
}

// ---------------------------------------------------------------------------
// WBO solver
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct WboSolver {
    pub config: SolverConfig,
}

impl WboSolver {
    pub fn new(config: SolverConfig) -> Self {
        Self { config }
    }
}

impl MaxSatSolver for WboSolver {
    fn solve(&self, formula: &MaxSatFormula) -> MaxSatResult {
        let start = std::time::Instant::now();
        let oracle = SatOracle::new(self.config.max_iterations * 100);
        let mut stats = SolverStatistics {
            variables: formula.num_variables,
            clauses: (formula.num_hard() + formula.num_soft()) as u32,
            ..Default::default()
        };

        let all_clauses: Vec<Vec<Literal>> = formula
            .hard_clauses
            .iter()
            .map(|c| c.literals.clone())
            .chain(formula.soft_clauses.iter().map(|c| c.literals.clone()))
            .collect();
        stats.solver_calls += 1;
        if let Some(model) = oracle.check_sat(&all_clauses, formula.num_variables) {
            stats.time_ms = start.elapsed().as_millis() as u64;
            return MaxSatResult {
                status: MaxSatStatus::Optimal,
                model: Some(model),
                cost: 0,
                statistics: stats,
            };
        }

        let mut active_soft: Vec<(Vec<Literal>, u64)> = formula
            .soft_clauses
            .iter()
            .map(|sc| (sc.literals.clone(), sc.weight))
            .collect();
        active_soft.sort_by_key(|(_, w)| *w);

        let mut total_relaxed_cost = 0u64;

        for iter in 0..self.config.max_iterations {
            if start.elapsed().as_millis() as u64 > self.config.timeout_ms || active_soft.is_empty()
            {
                break;
            }

            let mut clauses: Vec<Vec<Literal>> = formula
                .hard_clauses
                .iter()
                .map(|c| c.literals.clone())
                .collect();
            for (lits, _) in &active_soft {
                clauses.push(lits.clone());
            }

            stats.solver_calls += 1;
            if let Some(model) = oracle.check_sat(&clauses, formula.num_variables) {
                let cost = compute_cost(formula, &model);
                stats.iterations = iter + 1;
                stats.time_ms = start.elapsed().as_millis() as u64;
                return MaxSatResult {
                    status: MaxSatStatus::Optimal,
                    model: Some(model),
                    cost,
                    statistics: stats,
                };
            }

            let (_, w) = active_soft.remove(0);
            total_relaxed_cost += w;
            stats.iterations = iter + 1;
        }

        let model = find_initial_model(formula, &oracle);
        stats.solver_calls += 1;
        stats.time_ms = start.elapsed().as_millis() as u64;

        match model {
            Some(m) => {
                let cost = compute_cost(formula, &m);
                MaxSatResult {
                    status: MaxSatStatus::Satisfiable,
                    model: Some(m),
                    cost,
                    statistics: stats,
                }
            }
            None => MaxSatResult {
                status: MaxSatStatus::Unsatisfiable,
                model: None,
                cost: total_relaxed_cost,
                statistics: stats,
            },
        }
    }

    fn name(&self) -> &str {
        "WBO"
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formula::MaxSatFormula;

    fn trivial_sat_formula() -> MaxSatFormula {
        let mut f = MaxSatFormula::new();
        f.add_hard_clause(vec![1, 2], "c1");
        f.add_soft_clause(vec![1], 5, "s1");
        f
    }

    fn trivial_unsat_formula() -> MaxSatFormula {
        let mut f = MaxSatFormula::new();
        f.add_hard_clause(vec![1], "must-true");
        f.add_hard_clause(vec![-1], "must-false");
        f
    }

    #[test]
    fn test_sat_oracle_sat() {
        let oracle = SatOracle::new(1000);
        let clauses = vec![vec![1, 2], vec![-1, 2]];
        let model = oracle.check_sat(&clauses, 2);
        assert!(model.is_some());
        let m = model.unwrap();
        assert!(m.eval_clause(&[1, 2]));
        assert!(m.eval_clause(&[-1, 2]));
    }

    #[test]
    fn test_sat_oracle_unsat() {
        let oracle = SatOracle::new(1000);
        let clauses = vec![vec![1], vec![-1]];
        assert!(oracle.check_sat(&clauses, 1).is_none());
    }

    #[test]
    fn test_cdcl_solver() {
        let solver = CdclSolver::new(5000);
        let clauses = vec![vec![1, 2], vec![-1, 3], vec![2, 3]];
        let model = solver.solve_sat(&clauses, 3);
        assert!(model.is_some());
    }

    #[test]
    fn test_fu_malik_sat() {
        let f = trivial_sat_formula();
        let solver = FuMalikSolver::new(SolverConfig::default());
        let result = solver.solve(&f);
        assert!(
            result.status == MaxSatStatus::Optimal
                || result.status == MaxSatStatus::Satisfiable
        );
        assert!(result.model.is_some());
    }

    #[test]
    fn test_fu_malik_unsat() {
        let f = trivial_unsat_formula();
        let solver = FuMalikSolver::new(SolverConfig::default());
        let result = solver.solve(&f);
        assert_eq!(result.status, MaxSatStatus::Unsatisfiable);
    }

    #[test]
    fn test_linear_search_sat() {
        let f = trivial_sat_formula();
        let solver = LinearSearchSolver::new(SolverConfig::default());
        let result = solver.solve(&f);
        assert!(result.model.is_some());
    }

    #[test]
    fn test_wbo_all_satisfied() {
        let mut f = MaxSatFormula::new();
        f.add_hard_clause(vec![1], "h");
        f.add_soft_clause(vec![1], 10, "s");
        let solver = WboSolver::new(SolverConfig::default());
        let result = solver.solve(&f);
        assert_eq!(result.cost, 0);
    }

    #[test]
    fn test_model_eval() {
        let model = Model {
            assignments: vec![true, false, true],
        };
        assert!(model.value(1));
        assert!(!model.value(2));
        assert!(model.value(3));
        assert!(model.eval_literal(1));
        assert!(!model.eval_literal(2));
        assert!(model.eval_literal(-2));
        assert!(model.eval_clause(&[1, 2]));
        assert!(!model.eval_clause(&[-1, 2, -3]));
    }

    #[test]
    fn test_solver_names() {
        let f = FuMalikSolver::new(SolverConfig::default());
        let l = LinearSearchSolver::new(SolverConfig::default());
        let w = WboSolver::new(SolverConfig::default());
        assert_eq!(f.name(), "FuMalik");
        assert_eq!(l.name(), "LinearSearch");
        assert_eq!(w.name(), "WBO");
    }
}
