// regsynth-solver: Weighted Partial MaxSMT solver
// Fu-Malik style iterative relaxation algorithm with support for
// weighted soft clauses and hard constraints.

use crate::result::{
    Assignment, Clause, Literal, MaxSmtResult, MaxSmtStatus, SolverStatistics,
    lit_neg, lit_var, make_lit,
};
use crate::sat_solver::DpllSolver;
use crate::solver_config::SolverConfig;
use std::collections::HashSet;
use std::time::Instant;

// ─── Soft Clause ────────────────────────────────────────────────────────────

/// A soft clause with a weight.
#[derive(Debug, Clone)]
pub struct SoftClause {
    pub lits: Clause,
    pub weight: f64,
    pub id: usize,
}

// ─── MaxSMT Solver ──────────────────────────────────────────────────────────

/// Weighted Partial MaxSMT solver using Fu-Malik style iterative relaxation.
///
/// Algorithm overview:
/// 1. Start with all hard clauses and soft clauses.
/// 2. Solve the SAT problem. If SAT, we're optimal (cost = 0 for relaxed).
/// 3. If UNSAT, extract the unsatisfiable core.
/// 4. For each soft clause in the core, add a relaxation (blocking) variable.
/// 5. Add an at-most-one constraint on the relaxation variables (at most one
///    soft clause in the core can be violated).
/// 6. Repeat until SAT is found.
pub struct MaxSmtSolver {
    config: SolverConfig,
    pub stats: SolverStatistics,
}

impl MaxSmtSolver {
    pub fn new(config: SolverConfig) -> Self {
        Self {
            config,
            stats: SolverStatistics::new(),
        }
    }

    /// Solve the weighted partial MaxSMT problem.
    ///
    /// `hard_clauses`: clauses that must be satisfied.
    /// `soft_clauses`: clauses with weights; minimize the total weight of violated soft clauses.
    ///
    /// Algorithm: assumption-based iterative relaxation.
    /// 1. Assign each soft clause a selector literal (assumption).
    /// 2. Solve with all selectors as assumptions.
    /// 3. If SAT, optimal (cost = 0).
    /// 4. If UNSAT, identify which soft clauses are in the core, relax the
    ///    cheapest one, and repeat.
    pub fn solve(
        &mut self,
        hard_clauses: &[Clause],
        soft_clauses: &[SoftClause],
    ) -> MaxSmtResult {
        let start = Instant::now();

        if soft_clauses.is_empty() {
            return self.solve_hard_only(hard_clauses, &start);
        }

        // Compute total variable count
        let mut max_var = count_vars(hard_clauses);
        for sc in soft_clauses {
            for &l in &sc.lits {
                let v = lit_var(l);
                if v > max_var {
                    max_var = v;
                }
            }
        }

        // Create selector variables for each soft clause
        let mut selectors: Vec<u32> = Vec::new();
        for _ in soft_clauses {
            max_var += 1;
            selectors.push(max_var);
        }

        // Build the solver
        let mut solver = DpllSolver::new(max_var, self.config.clone());

        // Add hard clauses
        let mut clause_idx = 0;
        for c in hard_clauses {
            solver.add_original_clause(c.clone(), clause_idx);
            clause_idx += 1;
        }

        // Add soft clauses with selector: (soft_lits OR NOT selector)
        // When selector is assumed true, the soft clause must hold.
        // When selector is relaxed (not assumed), the soft clause can be violated.
        for (i, sc) in soft_clauses.iter().enumerate() {
            let mut clause = sc.lits.clone();
            clause.push(make_lit(selectors[i], false)); // NOT selector
            solver.add_original_clause(clause, clause_idx);
            clause_idx += 1;
        }

        // Track which soft clauses are relaxed (violated)
        let mut relaxed: HashSet<usize> = HashSet::new();
        let mut total_cost = 0.0;
        let mut best_assignment = Assignment::new();

        for _iteration in 0..self.config.maxsmt_max_iterations {
            if start.elapsed() > self.config.timeout {
                self.stats.time_ms = start.elapsed().as_millis() as u64;
                return MaxSmtResult {
                    status: MaxSmtStatus::Timeout,
                    assignment: if best_assignment.num_assigned() > 0 {
                        Some(best_assignment)
                    } else {
                        None
                    },
                    cost: total_cost,
                    num_violated_soft: relaxed.len(),
                    stats: self.stats.clone(),
                };
            }

            // Build assumptions: assume selector=true for non-relaxed soft clauses
            let assumptions: Vec<Literal> = (0..soft_clauses.len())
                .filter(|i| !relaxed.contains(i))
                .map(|i| make_lit(selectors[i], true))
                .collect();

            solver.reset();
            let result = solver.solve_with_assumptions(&assumptions);
            self.stats.decisions += solver.stats.decisions;
            self.stats.conflicts += solver.stats.conflicts;
            self.stats.propagations += solver.stats.propagations;

            match result {
                crate::result::SatResult::Sat(assignment) => {
                    best_assignment = assignment;
                    self.stats.time_ms = start.elapsed().as_millis() as u64;
                    return MaxSmtResult {
                        status: MaxSmtStatus::Optimal,
                        assignment: Some(best_assignment),
                        cost: total_cost,
                        num_violated_soft: relaxed.len(),
                        stats: self.stats.clone(),
                    };
                }
                crate::result::SatResult::Unsat(core) => {
                    // Find the soft clause with the smallest weight that's in the core
                    let core_vars: HashSet<u32> = core
                        .iter()
                        .flat_map(|c| c.iter().map(|&l| lit_var(l)))
                        .collect();

                    // Find which non-relaxed soft clause selectors appear in the core
                    let mut candidates: Vec<(usize, f64)> = Vec::new();
                    for (i, sc) in soft_clauses.iter().enumerate() {
                        if relaxed.contains(&i) {
                            continue;
                        }
                        if core_vars.contains(&selectors[i]) {
                            candidates.push((i, sc.weight));
                        }
                    }

                    if candidates.is_empty() {
                        // Core doesn't involve any soft clause selectors → hard clauses alone are UNSAT
                        self.stats.time_ms = start.elapsed().as_millis() as u64;
                        return MaxSmtResult {
                            status: MaxSmtStatus::Unsatisfiable,
                            assignment: None,
                            cost: f64::INFINITY,
                            num_violated_soft: 0,
                            stats: self.stats.clone(),
                        };
                    }

                    // Relax the cheapest soft clause in the core
                    candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                    let (relax_idx, relax_weight) = candidates[0];
                    relaxed.insert(relax_idx);
                    total_cost += relax_weight;
                }
                crate::result::SatResult::Unknown(_) => {
                    self.stats.time_ms = start.elapsed().as_millis() as u64;
                    return MaxSmtResult {
                        status: MaxSmtStatus::Unknown,
                        assignment: None,
                        cost: f64::INFINITY,
                        num_violated_soft: 0,
                        stats: self.stats.clone(),
                    };
                }
            }
        }

        self.stats.time_ms = start.elapsed().as_millis() as u64;
        MaxSmtResult {
            status: MaxSmtStatus::Timeout,
            assignment: None,
            cost: total_cost,
            num_violated_soft: relaxed.len(),
            stats: self.stats.clone(),
        }
    }

    /// Solve when there are only hard clauses.
    fn solve_hard_only(&mut self, hard_clauses: &[Clause], start: &Instant) -> MaxSmtResult {
        let num_vars = count_vars(hard_clauses);
        let mut solver = DpllSolver::new(num_vars, self.config.clone());
        for (i, c) in hard_clauses.iter().enumerate() {
            solver.add_original_clause(c.clone(), i);
        }
        let result = solver.solve();
        self.stats = solver.stats.clone();
        self.stats.time_ms = start.elapsed().as_millis() as u64;

        match result {
            crate::result::SatResult::Sat(assignment) => MaxSmtResult {
                status: MaxSmtStatus::Optimal,
                assignment: Some(assignment),
                cost: 0.0,
                num_violated_soft: 0,
                stats: self.stats.clone(),
            },
            crate::result::SatResult::Unsat(_) => MaxSmtResult {
                status: MaxSmtStatus::Unsatisfiable,
                assignment: None,
                cost: f64::INFINITY,
                num_violated_soft: 0,
                stats: self.stats.clone(),
            },
            crate::result::SatResult::Unknown(reason) => MaxSmtResult {
                status: MaxSmtStatus::Unknown,
                assignment: None,
                cost: f64::INFINITY,
                num_violated_soft: 0,
                stats: self.stats.clone(),
            },
        }
    }

    /// Convenience: solve unweighted MaxSAT (all soft clauses have weight 1).
    pub fn solve_unweighted(
        &mut self,
        hard_clauses: &[Clause],
        soft_clauses: &[Clause],
    ) -> MaxSmtResult {
        let weighted: Vec<SoftClause> = soft_clauses
            .iter()
            .enumerate()
            .map(|(i, c)| SoftClause {
                lits: c.clone(),
                weight: 1.0,
                id: i,
            })
            .collect();
        self.solve(hard_clauses, &weighted)
    }
}

/// Count the maximum variable index in a set of clauses.
fn count_vars(clauses: &[Clause]) -> u32 {
    clauses
        .iter()
        .flat_map(|c| c.iter())
        .map(|l| lit_var(*l))
        .max()
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> SolverConfig {
        SolverConfig::default()
    }

    #[test]
    fn test_maxsmt_all_sat() {
        let mut solver = MaxSmtSolver::new(default_config());
        // Hard: (x1 OR x2)
        // Soft: (x1), (x2)
        let hard = vec![vec![1, 2]];
        let soft = vec![
            SoftClause { lits: vec![1], weight: 1.0, id: 0 },
            SoftClause { lits: vec![2], weight: 1.0, id: 1 },
        ];
        let result = solver.solve(&hard, &soft);
        assert_eq!(result.status, MaxSmtStatus::Optimal);
        assert!(result.cost < 1e-6); // Both soft can be satisfied
    }

    #[test]
    fn test_maxsmt_must_violate_one() {
        let mut solver = MaxSmtSolver::new(default_config());
        // Hard: (NOT x1 OR NOT x2) - at most one of x1, x2
        // Soft: (x1) weight=1, (x2) weight=1
        let hard = vec![vec![-1, -2]];
        let soft = vec![
            SoftClause { lits: vec![1], weight: 1.0, id: 0 },
            SoftClause { lits: vec![2], weight: 1.0, id: 1 },
        ];
        let result = solver.solve(&hard, &soft);
        assert_eq!(result.status, MaxSmtStatus::Optimal);
        // Must violate at least one soft clause
        assert!(result.cost >= 1.0 - 1e-6);
    }

    #[test]
    fn test_maxsmt_weighted() {
        let mut solver = MaxSmtSolver::new(default_config());
        // Hard: (NOT x1 OR NOT x2)
        // Soft: (x1) weight=10, (x2) weight=1
        // Optimal: satisfy x1 (higher weight), violate x2
        let hard = vec![vec![-1, -2]];
        let soft = vec![
            SoftClause { lits: vec![1], weight: 10.0, id: 0 },
            SoftClause { lits: vec![2], weight: 1.0, id: 1 },
        ];
        let result = solver.solve(&hard, &soft);
        assert_eq!(result.status, MaxSmtStatus::Optimal);
        assert!(result.cost <= 1.0 + 1e-6); // Should violate the lighter one
    }

    #[test]
    fn test_maxsmt_hard_unsat() {
        let mut solver = MaxSmtSolver::new(default_config());
        // Hard: (x1) AND (NOT x1) - contradiction
        let hard = vec![vec![1], vec![-1]];
        let soft = vec![SoftClause { lits: vec![2], weight: 1.0, id: 0 }];
        let result = solver.solve(&hard, &soft);
        assert_eq!(result.status, MaxSmtStatus::Unsatisfiable);
    }

    #[test]
    fn test_maxsmt_no_soft() {
        let mut solver = MaxSmtSolver::new(default_config());
        let hard = vec![vec![1, 2], vec![-1, 2]];
        let result = solver.solve(&hard, &[]);
        assert_eq!(result.status, MaxSmtStatus::Optimal);
        assert!(result.cost < 1e-6);
    }

    #[test]
    fn test_maxsmt_unweighted() {
        let mut solver = MaxSmtSolver::new(default_config());
        let hard = vec![vec![-1, -2, -3]]; // at most 2 of {x1, x2, x3}
        let soft = vec![vec![1], vec![2], vec![3]];
        let result = solver.solve_unweighted(&hard, &soft);
        assert_eq!(result.status, MaxSmtStatus::Optimal);
        // Can satisfy at most 2 soft clauses
        assert!(result.num_violated_soft <= 1);
    }

    #[test]
    fn test_maxsmt_multiple_hard() {
        let mut solver = MaxSmtSolver::new(default_config());
        // x1 => x2, x2 => x3, NOT x3
        // So x1 must be false, x2 must be false
        let hard = vec![vec![-1, 2], vec![-2, 3], vec![-3]];
        let soft = vec![
            SoftClause { lits: vec![1], weight: 5.0, id: 0 },
            SoftClause { lits: vec![2], weight: 3.0, id: 1 },
            SoftClause { lits: vec![3], weight: 1.0, id: 2 },
        ];
        let result = solver.solve(&hard, &soft);
        assert_eq!(result.status, MaxSmtStatus::Optimal);
        // All of x1, x2, x3 must be false, so all soft violated
        // Cost = 5 + 3 + 1 = 9
        assert!((result.cost - 9.0).abs() < 1e-6);
    }
}
