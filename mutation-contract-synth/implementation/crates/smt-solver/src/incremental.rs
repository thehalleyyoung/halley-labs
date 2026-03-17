//! Incremental SMT solving via push/pop scoping and assumption literals.
//!
//! The incremental solver wraps a base solver context and provides efficient
//! mechanisms for performing multiple related queries without re-sending the
//! entire assertion stack.

use std::collections::HashMap;

use crate::ast::{SmtExpr, SmtSort};
use crate::context::SmtContext;
use crate::model::SmtModel;
use crate::solver::{SmtSolver, SolverResult};

// ---------------------------------------------------------------------------
// Activation literal pool
// ---------------------------------------------------------------------------

/// An activation literal — a fresh boolean used to control assertion
/// groups via assumptions.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ActivationLiteral {
    /// The SMT symbol name (e.g. `_act_3`).
    pub name: String,
    /// Monotonically increasing id.
    pub id: u64,
}

impl ActivationLiteral {
    fn new(id: u64) -> Self {
        ActivationLiteral {
            name: format!("_act_{}", id),
            id,
        }
    }

    /// Return the positive literal `_act_N`.
    pub fn positive(&self) -> SmtExpr {
        SmtExpr::sym(&self.name)
    }

    /// Return the negative literal `(not _act_N)`.
    pub fn negative(&self) -> SmtExpr {
        SmtExpr::not(SmtExpr::sym(&self.name))
    }
}

// ---------------------------------------------------------------------------
// Incremental solver
// ---------------------------------------------------------------------------

/// Manages incremental queries against an SMT solver, using either push/pop
/// or assumption-based activation literals.
#[derive(Debug)]
pub struct IncrementalSolver {
    /// The underlying context.
    context: SmtContext,
    /// Next activation literal id.
    next_act_id: u64,
    /// Active activation literals (group name → literal).
    active_groups: HashMap<String, ActivationLiteral>,
    /// Deactivated groups (still declared but negated in assumptions).
    deactivated_groups: HashMap<String, ActivationLiteral>,
    /// Query history (for debugging / provenance).
    query_log: Vec<IncrementalQuery>,
    /// Configuration.
    config: IncrementalConfig,
}

/// Configuration for the incremental solver.
#[derive(Debug, Clone)]
pub struct IncrementalConfig {
    /// Use activation literals instead of push/pop.
    pub use_activation_literals: bool,
    /// Maximum number of queries before a full reset.
    pub max_queries_before_reset: usize,
    /// Log all queries for debugging.
    pub log_queries: bool,
}

impl Default for IncrementalConfig {
    fn default() -> Self {
        IncrementalConfig {
            use_activation_literals: true,
            max_queries_before_reset: 1000,
            log_queries: false,
        }
    }
}

/// Record of a single incremental query.
#[derive(Debug, Clone)]
pub struct IncrementalQuery {
    /// Human-readable query description.
    pub description: String,
    /// Result of the query.
    pub result: QueryOutcome,
    /// Wall-clock time in microseconds.
    pub time_us: u64,
}

/// Outcome of a query.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueryOutcome {
    Sat,
    Unsat,
    Unknown,
    Error(String),
}

impl IncrementalSolver {
    /// Create a new incremental solver with a QF_LIA context.
    pub fn new() -> Self {
        Self::with_config(IncrementalConfig::default())
    }

    /// Create with custom configuration.
    pub fn with_config(config: IncrementalConfig) -> Self {
        let mut ctx = SmtContext::qf_lia_with_cores();
        if config.use_activation_literals {
            // No special setup needed, literals declared on demand.
        }
        let _ = &mut ctx;
        IncrementalSolver {
            context: ctx,
            next_act_id: 0,
            active_groups: HashMap::new(),
            deactivated_groups: HashMap::new(),
            query_log: Vec::new(),
            config,
        }
    }

    /// Get a reference to the underlying context.
    pub fn context(&self) -> &SmtContext {
        &self.context
    }

    /// Get a mutable reference to the underlying context.
    pub fn context_mut(&mut self) -> &mut SmtContext {
        &mut self.context
    }

    /// Declare a constant in the underlying context.
    pub fn declare_int(&mut self, name: &str) {
        self.context.declare_int(name);
    }

    /// Declare a boolean in the underlying context.
    pub fn declare_bool(&mut self, name: &str) {
        self.context.declare_bool(name);
    }

    /// Assert a formula that persists across all queries.
    pub fn assert_persistent(&mut self, expr: SmtExpr) {
        self.context.assert(expr);
    }

    /// Create a new assertion group. Returns a handle that can be
    /// activated/deactivated without push/pop.
    pub fn create_group(&mut self, name: &str) -> ActivationLiteral {
        let lit = ActivationLiteral::new(self.next_act_id);
        self.next_act_id += 1;
        self.context.declare_const(&lit.name, SmtSort::Bool);
        self.active_groups.insert(name.to_string(), lit.clone());
        lit
    }

    /// Assert within a named group (guarded by activation literal).
    pub fn assert_in_group(&mut self, group: &str, expr: SmtExpr) {
        if let Some(lit) = self.active_groups.get(group) {
            let guarded = SmtExpr::implies(lit.positive(), expr);
            self.context.assert(guarded);
        }
    }

    /// Deactivate a group (its assertions become inactive).
    pub fn deactivate_group(&mut self, name: &str) {
        if let Some(lit) = self.active_groups.remove(name) {
            self.deactivated_groups.insert(name.to_string(), lit);
        }
    }

    /// Reactivate a previously deactivated group.
    pub fn reactivate_group(&mut self, name: &str) {
        if let Some(lit) = self.deactivated_groups.remove(name) {
            self.active_groups.insert(name.to_string(), lit);
        }
    }

    /// Build the set of assumptions for the current query:
    /// positive literals for active groups, negative for deactivated.
    pub fn current_assumptions(&self) -> Vec<SmtExpr> {
        let mut assumptions = Vec::new();
        for lit in self.active_groups.values() {
            assumptions.push(lit.positive());
        }
        for lit in self.deactivated_groups.values() {
            assumptions.push(lit.negative());
        }
        assumptions
    }

    /// Push a solver scope (when not using activation literals).
    pub fn push_scope(&mut self) {
        self.context.push();
    }

    /// Pop a solver scope.
    pub fn pop_scope(&mut self) {
        self.context.pop();
    }

    /// Record a query outcome.
    fn record_query(&mut self, desc: &str, outcome: QueryOutcome, time_us: u64) {
        if self.config.log_queries {
            self.query_log.push(IncrementalQuery {
                description: desc.to_string(),
                result: outcome,
                time_us,
            });
        }
    }

    /// Get query history.
    pub fn query_log(&self) -> &[IncrementalQuery] {
        &self.query_log
    }

    /// Number of queries performed.
    pub fn query_count(&self) -> usize {
        self.query_log.len()
    }

    /// Check satisfiability using a concrete solver implementation.
    pub fn check_sat(&mut self, solver: &mut dyn SmtSolver) -> SolverResult {
        let start = std::time::Instant::now();
        self.context.check_sat();
        let script_text = self.context.render();
        let result = solver.check_sat_with_text(&script_text);
        let elapsed = start.elapsed().as_micros() as u64;

        let outcome = match &result {
            SolverResult::Sat(_) => QueryOutcome::Sat,
            SolverResult::Unsat => QueryOutcome::Unsat,
            SolverResult::Unknown(msg) => QueryOutcome::Unknown,
            SolverResult::Error(msg) => QueryOutcome::Error(msg.clone()),
        };
        let _ = msg_placeholder_for_unknown_and_error(&outcome);
        self.record_query("check-sat", outcome, elapsed);
        result
    }

    /// Check satisfiability with assumptions.
    pub fn check_sat_assuming(
        &mut self,
        solver: &mut dyn SmtSolver,
        extra_assumptions: &[SmtExpr],
    ) -> SolverResult {
        let start = std::time::Instant::now();
        let mut all_assumptions = self.current_assumptions();
        all_assumptions.extend(extra_assumptions.iter().cloned());

        self.context.push();
        for a in &all_assumptions {
            self.context.assert(a.clone());
        }
        self.context.check_sat();
        let script_text = self.context.render();
        let result = solver.check_sat_with_text(&script_text);
        self.context.pop();

        let elapsed = start.elapsed().as_micros() as u64;
        let outcome = match &result {
            SolverResult::Sat(_) => QueryOutcome::Sat,
            SolverResult::Unsat => QueryOutcome::Unsat,
            _ => QueryOutcome::Unknown,
        };
        self.record_query("check-sat-assuming", outcome, elapsed);
        result
    }

    /// Reset the solver state completely.
    pub fn reset(&mut self) {
        self.context = SmtContext::qf_lia_with_cores();
        self.active_groups.clear();
        self.deactivated_groups.clear();
    }
}

fn msg_placeholder_for_unknown_and_error(_outcome: &QueryOutcome) -> &'static str {
    ""
}

impl Default for IncrementalSolver {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Entailment checker (utility built on IncrementalSolver)
// ---------------------------------------------------------------------------

/// Batch entailment checker. Efficiently checks multiple φ ⊢ ψ queries
/// sharing the same background theory.
#[derive(Debug)]
pub struct EntailmentChecker {
    solver: IncrementalSolver,
    /// Number of entailment checks performed.
    pub checks_performed: u64,
    /// Number of entailments confirmed.
    pub entailments_confirmed: u64,
}

impl EntailmentChecker {
    pub fn new() -> Self {
        EntailmentChecker {
            solver: IncrementalSolver::new(),
            checks_performed: 0,
            entailments_confirmed: 0,
        }
    }

    /// Add a background axiom that holds for all queries.
    pub fn add_axiom(&mut self, axiom: SmtExpr) {
        self.solver.assert_persistent(axiom);
    }

    /// Declare a variable used across queries.
    pub fn declare_int(&mut self, name: &str) {
        self.solver.declare_int(name);
    }

    /// Check if `premise` entails `conclusion`.
    /// Returns `true` if the entailment holds (UNSAT check).
    pub fn check_entailment(
        &mut self,
        premise: &SmtExpr,
        conclusion: &SmtExpr,
        smt_solver: &mut dyn SmtSolver,
    ) -> bool {
        self.checks_performed += 1;
        self.solver.push_scope();
        let negated = SmtExpr::and(premise.clone(), SmtExpr::not(conclusion.clone()));
        self.solver.assert_persistent(negated);
        let result = self.solver.check_sat(smt_solver);
        self.solver.pop_scope();
        let is_entailed = matches!(result, SolverResult::Unsat);
        if is_entailed {
            self.entailments_confirmed += 1;
        }
        is_entailed
    }

    /// Check mutual entailment (logical equivalence).
    pub fn check_equivalence(
        &mut self,
        lhs: &SmtExpr,
        rhs: &SmtExpr,
        smt_solver: &mut dyn SmtSolver,
    ) -> bool {
        self.check_entailment(lhs, rhs, smt_solver) && self.check_entailment(rhs, lhs, smt_solver)
    }

    /// Statistics.
    pub fn stats(&self) -> (u64, u64) {
        (self.checks_performed, self.entailments_confirmed)
    }
}

impl Default for EntailmentChecker {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_activation_literal() {
        let lit = ActivationLiteral::new(42);
        assert_eq!(lit.name, "_act_42");
        assert_eq!(lit.positive(), SmtExpr::sym("_act_42"));
    }

    #[test]
    fn test_incremental_solver_creation() {
        let solver = IncrementalSolver::new();
        assert_eq!(solver.query_count(), 0);
        assert_eq!(solver.context().logic(), "QF_LIA");
    }

    #[test]
    fn test_group_management() {
        let mut solver = IncrementalSolver::new();
        let _lit = solver.create_group("group1");
        solver.assert_in_group("group1", SmtExpr::le(SmtExpr::sym("x"), SmtExpr::int(10)));
        assert_eq!(solver.current_assumptions().len(), 1);

        solver.deactivate_group("group1");
        let assumptions = solver.current_assumptions();
        assert_eq!(assumptions.len(), 1);

        solver.reactivate_group("group1");
        assert_eq!(solver.current_assumptions().len(), 1);
    }

    #[test]
    fn test_push_pop_scope() {
        let mut solver = IncrementalSolver::new();
        solver.declare_int("x");
        solver.push_scope();
        solver.declare_int("y");
        assert!(solver.context().is_declared("y"));
        solver.pop_scope();
        assert!(!solver.context().is_declared("y"));
    }

    #[test]
    fn test_reset() {
        let mut solver = IncrementalSolver::new();
        solver.declare_int("x");
        solver.create_group("g1");
        solver.reset();
        assert!(!solver.context().is_declared("x"));
        assert!(solver.current_assumptions().is_empty());
    }

    #[test]
    fn test_entailment_checker_creation() {
        let checker = EntailmentChecker::new();
        assert_eq!(checker.stats(), (0, 0));
    }

    #[test]
    fn test_config_defaults() {
        let config = IncrementalConfig::default();
        assert!(config.use_activation_literals);
        assert_eq!(config.max_queries_before_reset, 1000);
        assert!(!config.log_queries);
    }

    #[test]
    fn test_query_outcome() {
        let o = QueryOutcome::Sat;
        assert_eq!(o, QueryOutcome::Sat);
        assert_ne!(o, QueryOutcome::Unsat);
    }
}
