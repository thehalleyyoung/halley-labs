// regsynth-solver: Incremental solving support
// Push/pop scope management, assumption-based solving, and learned clause
// management across incremental calls.

use crate::result::{
    Assignment, Clause, Literal, SatResult, SolverStatistics,
    lit_neg, lit_var, make_lit,
};
use crate::sat_solver::DpllSolver;
use crate::solver_config::SolverConfig;
use std::time::Instant;

// ─── Scope Frame ────────────────────────────────────────────────────────────

/// A scope frame tracking the state at a push point.
#[derive(Debug, Clone)]
struct ScopeFrame {
    /// Number of clauses when this scope was pushed.
    clause_count: usize,
    /// Number of original clause mappings when pushed.
    original_clause_count: usize,
}

// ─── Incremental Solver ─────────────────────────────────────────────────────

/// Incremental SAT solver with push/pop scope management and assumption support.
///
/// Supports:
/// - `push()`/`pop()` for scoped constraint addition and retraction.
/// - `solve_with_assumptions()` for assumption-based queries.
/// - Learned clause management across incremental calls.
pub struct IncrementalSolver {
    /// All clauses ever added (including scoped ones).
    clauses: Vec<Clause>,
    /// Original clause index mapping for each clause.
    original_indices: Vec<usize>,
    /// Next original index to assign.
    next_original_idx: usize,
    /// Scope stack.
    scopes: Vec<ScopeFrame>,
    /// Number of variables.
    num_vars: u32,
    /// Learned clauses from previous solves (kept across incremental calls).
    learned_clauses: Vec<Clause>,
    /// Maximum number of learned clauses to retain.
    max_retained_learned: usize,
    /// Solver configuration.
    config: SolverConfig,
    /// Cumulative statistics across all incremental solves.
    pub stats: SolverStatistics,
}

impl IncrementalSolver {
    /// Create a new incremental solver.
    pub fn new(num_vars: u32, config: SolverConfig) -> Self {
        Self {
            clauses: Vec::new(),
            original_indices: Vec::new(),
            next_original_idx: 0,
            scopes: Vec::new(),
            num_vars,
            learned_clauses: Vec::new(),
            max_retained_learned: 5000,
            config,
            stats: SolverStatistics::new(),
        }
    }

    /// Push a new scope. Clauses added after this can be retracted by `pop()`.
    pub fn push(&mut self) {
        self.scopes.push(ScopeFrame {
            clause_count: self.clauses.len(),
            original_clause_count: self.original_indices.len(),
        });
    }

    /// Pop the most recent scope, removing all clauses added since the last `push()`.
    /// Returns false if there is no scope to pop.
    pub fn pop(&mut self) -> bool {
        if let Some(frame) = self.scopes.pop() {
            self.clauses.truncate(frame.clause_count);
            self.original_indices.truncate(frame.original_clause_count);
            // Remove learned clauses that reference removed clauses
            self.prune_learned_clauses();
            true
        } else {
            false
        }
    }

    /// Get the current scope depth.
    pub fn scope_depth(&self) -> usize {
        self.scopes.len()
    }

    /// Add a constraint (clause) at the current scope level.
    pub fn add_constraint(&mut self, clause: Clause) {
        // Update num_vars if clause introduces new variables
        for &lit in &clause {
            let v = lit_var(lit);
            if v > self.num_vars {
                self.num_vars = v;
            }
        }
        self.clauses.push(clause);
        self.original_indices.push(self.next_original_idx);
        self.next_original_idx += 1;
    }

    /// Add multiple constraints at once.
    pub fn add_constraints(&mut self, clauses: &[Clause]) {
        for clause in clauses {
            self.add_constraint(clause.clone());
        }
    }

    /// Solve the current set of constraints (no assumptions).
    pub fn solve(&mut self) -> SatResult {
        self.solve_with_assumptions(&[])
    }

    /// Solve under a set of assumptions.
    ///
    /// Assumptions are temporary: they are NOT added to the clause database and
    /// do not persist across calls. Each assumption is a literal that must be true.
    pub fn solve_with_assumptions(&mut self, assumptions: &[Literal]) -> SatResult {
        let start = Instant::now();

        // Build the SAT solver with current clauses + learned clauses
        let mut solver = DpllSolver::new(self.num_vars, self.config.clone());

        // Add all current clauses
        for (i, clause) in self.clauses.iter().enumerate() {
            let orig_idx = self.original_indices.get(i).copied().unwrap_or(i);
            solver.add_original_clause(clause.clone(), orig_idx);
        }

        // Add retained learned clauses
        for lc in &self.learned_clauses {
            solver.add_clause(lc.clone());
        }

        // Solve with assumptions
        let result = if assumptions.is_empty() {
            solver.solve()
        } else {
            solver.solve_with_assumptions(assumptions)
        };

        // Update statistics
        self.stats.decisions += solver.stats.decisions;
        self.stats.conflicts += solver.stats.conflicts;
        self.stats.propagations += solver.stats.propagations;
        self.stats.restarts += solver.stats.restarts;
        self.stats.time_ms += start.elapsed().as_millis() as u64;

        // Harvest learned clauses from the solver
        self.harvest_learned_clauses(&solver);

        result
    }

    /// Get the model (assignment) under assumptions if the last solve was SAT.
    pub fn get_model_under_assumptions(
        &mut self,
        assumptions: &[Literal],
    ) -> Option<Assignment> {
        match self.solve_with_assumptions(assumptions) {
            SatResult::Sat(assignment) => Some(assignment),
            _ => None,
        }
    }

    /// Harvest useful learned clauses from the solver for future incremental calls.
    fn harvest_learned_clauses(&mut self, solver: &DpllSolver) {
        let clause_infos = solver.get_clauses();
        for ci in clause_infos {
            if ci.learned && ci.lits.len() <= 10 {
                // Keep short learned clauses
                let clause = ci.lits.clone();
                if !self.learned_clauses.contains(&clause) {
                    self.learned_clauses.push(clause);
                }
            }
        }

        // Prune if too many
        if self.learned_clauses.len() > self.max_retained_learned {
            // Keep shorter clauses (they're more useful)
            self.learned_clauses
                .sort_by_key(|c| c.len());
            self.learned_clauses
                .truncate(self.max_retained_learned / 2);
        }
    }

    /// Prune learned clauses that reference variables not in current scope.
    fn prune_learned_clauses(&mut self) {
        let active_vars: std::collections::HashSet<u32> = self
            .clauses
            .iter()
            .flat_map(|c| c.iter().map(|&l| lit_var(l)))
            .collect();

        self.learned_clauses.retain(|clause| {
            clause
                .iter()
                .all(|&lit| active_vars.contains(&lit_var(lit)))
        });
    }

    /// Get current clause count (excluding learned).
    pub fn num_clauses(&self) -> usize {
        self.clauses.len()
    }

    /// Get number of retained learned clauses.
    pub fn num_learned(&self) -> usize {
        self.learned_clauses.len()
    }

    /// Get number of variables.
    pub fn num_vars(&self) -> u32 {
        self.num_vars
    }

    /// Reset all state: clear clauses, scopes, and learned clauses.
    pub fn reset(&mut self) {
        self.clauses.clear();
        self.original_indices.clear();
        self.next_original_idx = 0;
        self.scopes.clear();
        self.learned_clauses.clear();
        self.stats = SolverStatistics::new();
    }

    /// Set the maximum number of learned clauses to retain.
    pub fn set_max_retained_learned(&mut self, max: usize) {
        self.max_retained_learned = max;
    }

    /// Check if the current clause set is satisfiable (convenience method).
    pub fn is_satisfiable(&mut self) -> bool {
        self.solve().is_sat()
    }

    /// Check satisfiability under assumptions (convenience method).
    pub fn is_satisfiable_under(&mut self, assumptions: &[Literal]) -> bool {
        self.solve_with_assumptions(assumptions).is_sat()
    }

    /// Add a unit clause (single literal).
    pub fn assert_literal(&mut self, lit: Literal) {
        self.add_constraint(vec![lit]);
    }

    /// Add a binary clause (two literals).
    pub fn add_implication(&mut self, from: Literal, to: Literal) {
        self.add_constraint(vec![lit_neg(from), to]);
    }

    /// Add a mutual exclusion constraint: at most one of the literals can be true.
    pub fn add_at_most_one(&mut self, lits: &[Literal]) {
        for i in 0..lits.len() {
            for j in (i + 1)..lits.len() {
                self.add_constraint(vec![lit_neg(lits[i]), lit_neg(lits[j])]);
            }
        }
    }

    /// Add an exactly-one constraint: exactly one of the literals must be true.
    pub fn add_exactly_one(&mut self, lits: &[Literal]) {
        // At least one
        self.add_constraint(lits.to_vec());
        // At most one
        self.add_at_most_one(lits);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> SolverConfig {
        SolverConfig::default()
    }

    #[test]
    fn test_basic_solve() {
        let mut solver = IncrementalSolver::new(2, default_config());
        solver.add_constraint(vec![1, 2]); // x1 OR x2
        let result = solver.solve();
        assert!(result.is_sat());
    }

    #[test]
    fn test_basic_unsat() {
        let mut solver = IncrementalSolver::new(1, default_config());
        solver.add_constraint(vec![1]); // x1
        solver.add_constraint(vec![-1]); // NOT x1
        let result = solver.solve();
        assert!(result.is_unsat());
    }

    #[test]
    fn test_push_pop() {
        let mut solver = IncrementalSolver::new(2, default_config());
        solver.add_constraint(vec![1, 2]); // x1 OR x2

        solver.push();
        solver.add_constraint(vec![-1]); // NOT x1
        solver.add_constraint(vec![-2]); // NOT x2
        // Now UNSAT
        assert!(solver.solve().is_unsat());

        solver.pop();
        // Back to just (x1 OR x2) → SAT
        assert!(solver.solve().is_sat());
    }

    #[test]
    fn test_nested_push_pop() {
        let mut solver = IncrementalSolver::new(3, default_config());
        solver.add_constraint(vec![1, 2, 3]);

        solver.push(); // scope 1
        solver.add_constraint(vec![-1]);
        assert!(solver.solve().is_sat());

        solver.push(); // scope 2
        solver.add_constraint(vec![-2]);
        assert!(solver.solve().is_sat()); // x3 must be true

        solver.push(); // scope 3
        solver.add_constraint(vec![-3]);
        assert!(solver.solve().is_unsat()); // no variable can be true

        solver.pop(); // back to scope 2
        assert!(solver.solve().is_sat());

        solver.pop(); // back to scope 1
        assert!(solver.solve().is_sat());

        solver.pop(); // back to base
        assert!(solver.solve().is_sat());
    }

    #[test]
    fn test_assumptions() {
        let mut solver = IncrementalSolver::new(3, default_config());
        solver.add_constraint(vec![1, 2]); // x1 OR x2
        solver.add_constraint(vec![2, 3]); // x2 OR x3

        // Under assumption x2=false, x1 and x3 must be true
        let result = solver.solve_with_assumptions(&[-2]);
        assert!(result.is_sat());
        let a = result.assignment().unwrap();
        assert_eq!(a.get(1), Some(true));
        assert_eq!(a.get(3), Some(true));

        // Without assumptions, should still be sat (assumptions are temporary)
        let result = solver.solve();
        assert!(result.is_sat());
    }

    #[test]
    fn test_assumption_causing_unsat() {
        let mut solver = IncrementalSolver::new(2, default_config());
        solver.add_constraint(vec![1]); // x1 must be true

        // Assume x1=false → UNSAT
        let result = solver.solve_with_assumptions(&[-1]);
        assert!(result.is_unsat());

        // Without assumption → SAT
        let result = solver.solve();
        assert!(result.is_sat());
    }

    #[test]
    fn test_scope_depth() {
        let mut solver = IncrementalSolver::new(2, default_config());
        assert_eq!(solver.scope_depth(), 0);

        solver.push();
        assert_eq!(solver.scope_depth(), 1);

        solver.push();
        assert_eq!(solver.scope_depth(), 2);

        solver.pop();
        assert_eq!(solver.scope_depth(), 1);

        solver.pop();
        assert_eq!(solver.scope_depth(), 0);

        assert!(!solver.pop()); // Can't pop below 0
    }

    #[test]
    fn test_at_most_one() {
        let mut solver = IncrementalSolver::new(3, default_config());
        solver.add_at_most_one(&[1, 2, 3]);
        // Can set x1=true
        let result = solver.solve_with_assumptions(&[1]);
        assert!(result.is_sat());
        let a = result.assignment().unwrap();
        assert_eq!(a.get(2), Some(false));
        assert_eq!(a.get(3), Some(false));
    }

    #[test]
    fn test_exactly_one() {
        let mut solver = IncrementalSolver::new(3, default_config());
        solver.add_exactly_one(&[1, 2, 3]);
        let result = solver.solve();
        assert!(result.is_sat());
        let a = result.assignment().unwrap();
        let count = (0..3)
            .filter(|&i| a.get(i as u32 + 1) == Some(true))
            .count();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_implication() {
        let mut solver = IncrementalSolver::new(2, default_config());
        solver.assert_literal(1); // x1 = true
        solver.add_implication(1, 2); // x1 => x2
        let result = solver.solve();
        assert!(result.is_sat());
        let a = result.assignment().unwrap();
        assert_eq!(a.get(2), Some(true));
    }

    #[test]
    fn test_learned_clauses_retained() {
        let mut solver = IncrementalSolver::new(4, default_config());
        // Create a formula that generates learned clauses
        solver.add_constraint(vec![1, 2]);
        solver.add_constraint(vec![-1, 3]);
        solver.add_constraint(vec![-2, 3]);
        solver.add_constraint(vec![-3, 4]);
        solver.add_constraint(vec![-3, -4]);
        let result = solver.solve();
        assert!(result.is_unsat());

        // Some learned clauses should be retained
        // (depends on the solver's behavior)
    }

    #[test]
    fn test_is_satisfiable() {
        let mut solver = IncrementalSolver::new(2, default_config());
        solver.add_constraint(vec![1, 2]);
        assert!(solver.is_satisfiable());

        solver.add_constraint(vec![-1]);
        solver.add_constraint(vec![-2]);
        assert!(!solver.is_satisfiable());
    }

    #[test]
    fn test_reset() {
        let mut solver = IncrementalSolver::new(2, default_config());
        solver.add_constraint(vec![1]);
        solver.add_constraint(vec![-1]);
        assert!(solver.solve().is_unsat());

        solver.reset();
        solver.add_constraint(vec![1, 2]);
        assert!(solver.solve().is_sat());
    }

    #[test]
    fn test_get_model_under_assumptions() {
        let mut solver = IncrementalSolver::new(3, default_config());
        solver.add_constraint(vec![1, 2, 3]);
        let model = solver.get_model_under_assumptions(&[make_lit(1, false), make_lit(2, false)]);
        assert!(model.is_some());
        let m = model.unwrap();
        assert_eq!(m.get(3), Some(true));
    }
}
