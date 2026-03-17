// Incremental SAT solving: assumption-based interface, clause activation/deactivation,
// push/pop context management.

use crate::cdcl::{CdclSolver, SatResult};
use crate::config::SolverConfig;
use crate::variable::{Literal, LiteralVec, Variable};
use smallvec::smallvec;
use std::collections::HashMap;
use std::fmt;

// ── ActivationLiteral ─────────────────────────────────────────────────────────

/// A literal that controls activation/deactivation of a clause group.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ActivationLiteral {
    pub literal: Literal,
}

impl ActivationLiteral {
    pub fn new(literal: Literal) -> Self {
        ActivationLiteral { literal }
    }

    /// The positive literal (activates clauses).
    pub fn active(&self) -> Literal {
        self.literal
    }

    /// The negative literal (deactivates clauses).
    pub fn inactive(&self) -> Literal {
        self.literal.negated()
    }
}

// ── AssumptionSet ─────────────────────────────────────────────────────────────

/// A set of assumption literals for incremental solving.
#[derive(Debug, Clone, Default)]
pub struct AssumptionSet {
    literals: Vec<Literal>,
}

impl AssumptionSet {
    pub fn new() -> Self {
        AssumptionSet {
            literals: Vec::new(),
        }
    }

    /// Add an assumption.
    pub fn add(&mut self, lit: Literal) {
        if !self.literals.contains(&lit) {
            self.literals.push(lit);
        }
    }

    /// Remove an assumption.
    pub fn remove(&mut self, lit: Literal) {
        self.literals.retain(|&l| l != lit);
    }

    /// Activate a clause group.
    pub fn activate(&mut self, act: &ActivationLiteral) {
        self.add(act.active());
        self.remove(act.inactive());
    }

    /// Deactivate a clause group.
    pub fn deactivate(&mut self, act: &ActivationLiteral) {
        self.add(act.inactive());
        self.remove(act.active());
    }

    /// Get all assumption literals.
    pub fn literals(&self) -> &[Literal] {
        &self.literals
    }

    /// Clear all assumptions.
    pub fn clear(&mut self) {
        self.literals.clear();
    }

    pub fn is_empty(&self) -> bool {
        self.literals.is_empty()
    }

    pub fn len(&self) -> usize {
        self.literals.len()
    }
}

// ── ClauseGroup ───────────────────────────────────────────────────────────────

/// A group of related clauses with a shared activation literal.
#[derive(Debug, Clone)]
pub struct ClauseGroup {
    /// Name/label of this clause group.
    pub name: String,
    /// Activation literal for this group.
    pub activation: ActivationLiteral,
    /// Clauses in this group (with the activation literal prepended).
    pub clause_count: usize,
    /// Whether the group is currently active.
    pub active: bool,
}

impl ClauseGroup {
    pub fn new(name: impl Into<String>, activation: ActivationLiteral) -> Self {
        ClauseGroup {
            name: name.into(),
            activation,
            clause_count: 0,
            active: true,
        }
    }
}

// ── IncrementalState ──────────────────────────────────────────────────────────

/// Tracks what has been added at each push level.
#[derive(Debug, Clone)]
struct IncrementalLevel {
    /// Clause groups added at this level.
    group_names: Vec<String>,
    /// Number of permanent clauses added at this level.
    permanent_clause_count: usize,
}

impl IncrementalLevel {
    fn new() -> Self {
        IncrementalLevel {
            group_names: Vec::new(),
            permanent_clause_count: 0,
        }
    }
}

// ── IncrementalSolver ─────────────────────────────────────────────────────────

/// Wrapper for incremental SAT solving with assumption-based clause activation.
pub struct IncrementalSolver {
    /// The underlying SAT solver.
    solver: CdclSolver,
    /// Assumptions for the next solve call.
    assumptions: AssumptionSet,
    /// Clause groups indexed by name.
    groups: HashMap<String, ClauseGroup>,
    /// Stack of incremental levels for push/pop.
    levels: Vec<IncrementalLevel>,
    /// Next variable index for activation literals.
    next_act_var: u32,
    max_user_var: u32,
    /// Total permanent clauses added.
    total_permanent: usize,
    /// Configuration.
    config: SolverConfig,
}

impl IncrementalSolver {
    /// Create a new incremental solver.
    pub fn new(config: SolverConfig) -> Self {
        IncrementalSolver {
            solver: CdclSolver::new(config.clone()),
            assumptions: AssumptionSet::new(),
            groups: HashMap::new(),
            levels: vec![IncrementalLevel::new()],
            next_act_var: 1000,
            max_user_var: 0,
            total_permanent: 0,
            config,
        }
    }

    /// Create with default configuration.
    pub fn default_solver() -> Self {
        Self::new(SolverConfig::incremental())
    }

    /// Add a permanent clause (always active, survives push/pop).
    pub fn add_permanent_clause(&mut self, clause: &[i32]) {
        for &d in clause { let v = d.unsigned_abs(); if v > self.max_user_var { self.max_user_var = v; self.next_act_var = self.next_act_var.max(v + 100); } }
        self.solver.add_clause_dimacs(clause);
        self.total_permanent += 1;
        if let Some(level) = self.levels.last_mut() {
            level.permanent_clause_count += 1;
        }
    }

    /// Add a permanent clause from literals.
    pub fn add_permanent_clause_lits(&mut self, lits: LiteralVec) {
        self.solver.add_clause_lits(lits);
        self.total_permanent += 1;
        if let Some(level) = self.levels.last_mut() {
            level.permanent_clause_count += 1;
        }
    }

    /// Create a new clause group with the given name.
    /// Returns the activation literal for the group.
    pub fn new_group(&mut self, name: impl Into<String>) -> ActivationLiteral {
        let name = name.into();
        let act_var = Variable::new(self.next_act_var);
        self.next_act_var += 1;
        let activation = ActivationLiteral::new(act_var.positive());
        let group = ClauseGroup::new(name.clone(), activation);
        self.groups.insert(name.clone(), group);

        if let Some(level) = self.levels.last_mut() {
            level.group_names.push(name);
        }

        activation
    }

    /// Add a clause to a named group. The activation literal is automatically prepended.
    pub fn add_to_group(&mut self, group_name: &str, clause: &[i32]) -> bool {
        let group = match self.groups.get_mut(group_name) {
            Some(g) => g,
            None => return false,
        };
        let act_neg = group.activation.inactive();
        group.clause_count += 1;

        // Build clause: act_neg ∨ original_clause.
        // When act_lit is true (active), the clause reduces to the original.
        // When act_lit is false (inactive), the clause is trivially satisfied.
        let mut lits: LiteralVec = smallvec![act_neg];
        for &d in clause {
            lits.push(Literal::from_dimacs(d));
        }
        self.solver.add_clause_lits(lits);
        true
    }

    /// Add a clause to a group from literals.
    pub fn add_to_group_lits(&mut self, group_name: &str, clause: LiteralVec) -> bool {
        let group = match self.groups.get_mut(group_name) {
            Some(g) => g,
            None => return false,
        };
        let act_neg = group.activation.inactive();
        group.clause_count += 1;

        let mut lits: LiteralVec = smallvec![act_neg];
        lits.extend(clause);
        self.solver.add_clause_lits(lits);
        true
    }

    /// Activate a clause group.
    pub fn activate_group(&mut self, group_name: &str) -> bool {
        let group = match self.groups.get_mut(group_name) {
            Some(g) => g,
            None => return false,
        };
        group.active = true;
        self.assumptions.activate(&group.activation);
        true
    }

    /// Deactivate a clause group.
    pub fn deactivate_group(&mut self, group_name: &str) -> bool {
        let group = match self.groups.get_mut(group_name) {
            Some(g) => g,
            None => return false,
        };
        group.active = false;
        self.assumptions.deactivate(&group.activation);
        true
    }

    /// Add a direct assumption.
    pub fn assume(&mut self, lit: Literal) {
        self.assumptions.add(lit);
    }

    /// Clear all assumptions (but keep activation-based ones).
    pub fn clear_user_assumptions(&mut self) {
        // Rebuild assumptions from active groups.
        self.assumptions.clear();
        for group in self.groups.values() {
            if group.active {
                self.assumptions.activate(&group.activation);
            } else {
                self.assumptions.deactivate(&group.activation);
            }
        }
    }

    /// Push a new context level.
    pub fn push(&mut self) {
        self.levels.push(IncrementalLevel::new());
    }

    /// Pop the last context level.
    /// Note: We can't actually remove clauses from the solver, but we deactivate
    /// the groups that were created at this level.
    pub fn pop(&mut self) -> bool {
        if self.levels.len() <= 1 {
            return false; // Can't pop the base level.
        }
        let level = self.levels.pop().unwrap();

        // Deactivate groups created at this level.
        for name in &level.group_names {
            if let Some(group) = self.groups.get_mut(name) {
                group.active = false;
                self.assumptions.deactivate(&group.activation);
            }
        }

        true
    }

    /// Solve under the current set of assumptions.
    pub fn solve(&mut self) -> SatResult {
        // Set up assumptions on the solver.
        self.solver.clear_assumptions();

        // Add activation assumptions for active groups.
        for group in self.groups.values() {
            if group.active {
                self.solver.assume(group.activation.active());
            }
            // For inactive groups, we don't need to assume anything;
            // the activation literal being unset means the clause is satisfied.
        }

        // Add user assumptions.
        for &lit in self.assumptions.literals() {
            self.solver.assume(lit);
        }

        self.solver.solve()
    }

    /// Solve under specific assumptions (overrides stored assumptions).
    pub fn solve_under(&mut self, assumptions: &[Literal]) -> SatResult {
        self.solver.clear_assumptions();

        // Still add activation assumptions.
        for group in self.groups.values() {
            if group.active {
                self.solver.assume(group.activation.active());
            }
        }

        for &lit in assumptions {
            self.solver.assume(lit);
        }

        self.solver.solve()
    }

    /// Get statistics from the underlying solver.
    pub fn statistics(&self) -> &crate::cdcl::SolverStats {
        self.solver.statistics()
    }

    /// Get current number of clause groups.
    pub fn num_groups(&self) -> usize {
        self.groups.len()
    }

    /// Get current push level depth.
    pub fn level_depth(&self) -> usize {
        self.levels.len()
    }

    /// Check if a group exists.
    pub fn has_group(&self, name: &str) -> bool {
        self.groups.contains_key(name)
    }

    /// Get info about a group.
    pub fn group_info(&self, name: &str) -> Option<&ClauseGroup> {
        self.groups.get(name)
    }
}

impl fmt::Debug for IncrementalSolver {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IncrementalSolver")
            .field("groups", &self.groups.len())
            .field("levels", &self.levels.len())
            .field("permanent_clauses", &self.total_permanent)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::variable::Variable;

    fn lit(v: i32) -> Literal {
        Literal::from_dimacs(v)
    }

    #[test]
    fn test_activation_literal() {
        let act = ActivationLiteral::new(Variable::new(1).positive());
        assert!(act.active().is_positive());
        assert!(act.inactive().is_negative());
    }

    #[test]
    fn test_assumption_set() {
        let mut aset = AssumptionSet::new();
        aset.add(lit(1));
        aset.add(lit(-2));
        assert_eq!(aset.len(), 2);
        aset.remove(lit(1));
        assert_eq!(aset.len(), 1);
        assert_eq!(aset.literals()[0], lit(-2));
    }

    #[test]
    fn test_assumption_set_no_duplicates() {
        let mut aset = AssumptionSet::new();
        aset.add(lit(1));
        aset.add(lit(1));
        assert_eq!(aset.len(), 1);
    }

    #[test]
    fn test_assumption_set_activate_deactivate() {
        let mut aset = AssumptionSet::new();
        let act = ActivationLiteral::new(Variable::new(1).positive());
        aset.activate(&act);
        assert!(aset.literals().contains(&act.active()));
        aset.deactivate(&act);
        assert!(aset.literals().contains(&act.inactive()));
        assert!(!aset.literals().contains(&act.active()));
    }

    #[test]
    fn test_clause_group() {
        let act = ActivationLiteral::new(Variable::new(1).positive());
        let group = ClauseGroup::new("test_group", act);
        assert_eq!(group.name, "test_group");
        assert_eq!(group.clause_count, 0);
        assert!(group.active);
    }

    #[test]
    fn test_incremental_solver_permanent_clauses() {
        let mut solver = IncrementalSolver::default_solver();
        solver.add_permanent_clause(&[1, 2]);
        solver.add_permanent_clause(&[-1, 2]);
        let result = solver.solve();
        assert!(result.is_sat());
        if let SatResult::Satisfiable(asgn) = result {
            assert_eq!(asgn.get(Variable::new(2)), Some(true));
        }
    }

    #[test]
    fn test_incremental_solver_groups() {
        let mut solver = IncrementalSolver::default_solver();
        // Permanent: x1 ∨ x2
        solver.add_permanent_clause(&[1, 2]);

        // Group "constraint_a": forces x1=false.
        let _act_a = solver.new_group("constraint_a");
        solver.add_to_group("constraint_a", &[-1]);
        solver.activate_group("constraint_a");

        let result = solver.solve();
        assert!(result.is_sat());
        if let SatResult::Satisfiable(asgn) = result {
            assert_eq!(asgn.get(Variable::new(2)), Some(true));
        }
    }

    #[test]
    fn test_incremental_solver_deactivate_group() {
        let mut solver = IncrementalSolver::default_solver();
        solver.add_permanent_clause(&[1, 2]);

        let _act = solver.new_group("force_neg");
        solver.add_to_group("force_neg", &[-1]);
        solver.add_to_group("force_neg", &[-2]);
        solver.activate_group("force_neg");

        // With group active: (1∨2) ∧ ¬1 ∧ ¬2 → UNSAT.
        let r1 = solver.solve();
        assert!(r1.is_unsat());

        // Deactivate the group.
        solver.deactivate_group("force_neg");
        let r2 = solver.solve();
        assert!(r2.is_sat());
    }

    #[test]
    fn test_incremental_solver_push_pop() {
        let mut solver = IncrementalSolver::default_solver();
        solver.add_permanent_clause(&[1, 2]);

        solver.push();

        let _act = solver.new_group("temp");
        solver.add_to_group("temp", &[-1]);
        solver.add_to_group("temp", &[-2]);
        solver.activate_group("temp");

        let r1 = solver.solve();
        assert!(r1.is_unsat());

        solver.pop();
        let r2 = solver.solve();
        assert!(r2.is_sat());
    }

    #[test]
    fn test_incremental_solver_nested_push_pop() {
        let mut solver = IncrementalSolver::default_solver();
        solver.add_permanent_clause(&[1, 2, 3]);
        assert_eq!(solver.level_depth(), 1);

        solver.push();
        assert_eq!(solver.level_depth(), 2);
        let _g1 = solver.new_group("level1");
        solver.add_to_group("level1", &[-1]);
        solver.activate_group("level1");

        solver.push();
        assert_eq!(solver.level_depth(), 3);
        let _g2 = solver.new_group("level2");
        solver.add_to_group("level2", &[-2]);
        solver.add_to_group("level2", &[-3]);
        solver.activate_group("level2");

        let r1 = solver.solve();
        assert!(r1.is_unsat());

        solver.pop();
        assert_eq!(solver.level_depth(), 2);
        let r2 = solver.solve();
        assert!(r2.is_sat()); // Only group "level1" active: ¬1 but (1∨2∨3) → 2 or 3 can satisfy.

        solver.pop();
        assert_eq!(solver.level_depth(), 1);
    }

    #[test]
    fn test_incremental_solver_solve_under() {
        let mut solver = IncrementalSolver::default_solver();
        solver.add_permanent_clause(&[1, 2]);
        solver.add_permanent_clause(&[-1, 2]);

        // Solve under assumption x2=false.
        let result = solver.solve_under(&[lit(-2)]);
        assert!(result.is_unsat());

        // Without the assumption, it's SAT.
        let result2 = solver.solve();
        assert!(result2.is_sat());
    }

    #[test]
    fn test_incremental_solver_has_group() {
        let mut solver = IncrementalSolver::default_solver();
        assert!(!solver.has_group("foo"));
        solver.new_group("foo");
        assert!(solver.has_group("foo"));
    }

    #[test]
    fn test_incremental_solver_group_info() {
        let mut solver = IncrementalSolver::default_solver();
        solver.new_group("bar");
        solver.add_to_group("bar", &[1, 2]);
        solver.add_to_group("bar", &[-3]);
        let info = solver.group_info("bar").unwrap();
        assert_eq!(info.clause_count, 2);
        assert!(info.active);
    }

    #[test]
    fn test_incremental_solver_pop_base_level() {
        let mut solver = IncrementalSolver::default_solver();
        assert!(!solver.pop()); // Can't pop the base level.
    }

    #[test]
    fn test_incremental_solver_multiple_groups() {
        let mut solver = IncrementalSolver::default_solver();
        solver.add_permanent_clause(&[1, 2, 3]);

        solver.new_group("g1");
        solver.add_to_group("g1", &[-1]);
        solver.activate_group("g1");

        solver.new_group("g2");
        solver.add_to_group("g2", &[-2]);
        solver.activate_group("g2");

        // (1∨2∨3) ∧ ¬1 ∧ ¬2 → must have 3=true.
        let result = solver.solve();
        assert!(result.is_sat());
        if let SatResult::Satisfiable(asgn) = result {
            assert_eq!(asgn.get(Variable::new(3)), Some(true));
        }

        assert_eq!(solver.num_groups(), 2);
    }

    #[test]
    fn test_incremental_solver_clear_user_assumptions() {
        let mut solver = IncrementalSolver::default_solver();
        solver.add_permanent_clause(&[1, 2]);
        solver.assume(lit(-1));
        solver.assume(lit(-2));

        let r1 = solver.solve();
        assert!(r1.is_unsat());

        solver.clear_user_assumptions();
        let r2 = solver.solve();
        assert!(r2.is_sat());
    }

    #[test]
    fn test_add_to_nonexistent_group() {
        let mut solver = IncrementalSolver::default_solver();
        assert!(!solver.add_to_group("nonexistent", &[1, 2]));
    }
}
