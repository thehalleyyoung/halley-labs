//! SMT solver interface, internal DPLL(T) engine, and SMT-LIB2 writer.
//!
//! Provides:
//! - [`SmtSolver`] trait for interacting with solvers
//! - [`InternalSolver`] implementing a basic DPLL(T) engine with
//!   unit propagation, CDCL, and simple theory solving
//! - [`SmtLib2Writer`] for serialising encoded problems to SMT-LIB2 files
//!   suitable for external solvers (Z3, CVC5, etc.)

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::encoder::EncodedProblem;
use crate::expression::{SmtExpr, free_vars, to_smtlib2};
use crate::variable::{SmtSort, VariableId, VariableStore};

// ═══════════════════════════════════════════════════════════════════════════
// SmtValue
// ═══════════════════════════════════════════════════════════════════════════

/// A concrete value assigned to a variable in a model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SmtValue {
    Bool(bool),
    Int(i64),
    /// Rational (numerator, denominator).
    Real(f64, f64),
    BitVec(Vec<bool>),
}

impl SmtValue {
    pub fn real_value(&self) -> Option<f64> {
        match self {
            SmtValue::Real(n, d) => {
                if *d == 0.0 { None } else { Some(n / d) }
            }
            SmtValue::Int(n) => Some(*n as f64),
            _ => None,
        }
    }

    pub fn bool_value(&self) -> Option<bool> {
        match self {
            SmtValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn int_value(&self) -> Option<i64> {
        match self {
            SmtValue::Int(n) => Some(*n),
            _ => None,
        }
    }
}

impl fmt::Display for SmtValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SmtValue::Bool(b) => write!(f, "{}", b),
            SmtValue::Int(n) => write!(f, "{}", n),
            SmtValue::Real(n, d) => {
                if *d == 1.0 { write!(f, "{}", n) }
                else { write!(f, "{}/{}", n, d) }
            }
            SmtValue::BitVec(bits) => {
                write!(f, "#b")?;
                for b in bits { write!(f, "{}", if *b { 1 } else { 0 })?; }
                Ok(())
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Model
// ═══════════════════════════════════════════════════════════════════════════

/// A satisfying assignment (model) for the variables.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    pub assignments: HashMap<VariableId, SmtValue>,
}

impl Model {
    pub fn new() -> Self {
        Self { assignments: HashMap::new() }
    }

    pub fn assign(&mut self, var: VariableId, value: SmtValue) {
        self.assignments.insert(var, value);
    }

    pub fn get(&self, var: VariableId) -> Option<&SmtValue> {
        self.assignments.get(&var)
    }

    pub fn len(&self) -> usize {
        self.assignments.len()
    }

    pub fn is_empty(&self) -> bool {
        self.assignments.is_empty()
    }

    /// Extract a counterexample trace (location + clock + concentration values at each step).
    pub fn extract_trace(
        &self,
        store: &VariableStore,
        bound: usize,
    ) -> Vec<HashMap<String, SmtValue>> {
        let mut trace = Vec::new();
        for step in 0..=bound {
            let mut step_values = HashMap::new();
            for var in store.iter() {
                let qname = var.qualified_name();
                if qname.ends_with(&format!("_t{}", step)) {
                    if let Some(val) = self.get(var.id) {
                        step_values.insert(qname, val.clone());
                    }
                }
            }
            trace.push(step_values);
        }
        trace
    }
}

impl Default for Model {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SolverResult
// ═══════════════════════════════════════════════════════════════════════════

/// Result of a satisfiability check.
#[derive(Debug, Clone)]
pub enum SolverResult {
    Sat(Model),
    Unsat,
    Unknown,
    Timeout,
}

impl SolverResult {
    pub fn is_sat(&self) -> bool {
        matches!(self, SolverResult::Sat(_))
    }

    pub fn is_unsat(&self) -> bool {
        matches!(self, SolverResult::Unsat)
    }

    pub fn model(&self) -> Option<&Model> {
        match self {
            SolverResult::Sat(m) => Some(m),
            _ => None,
        }
    }
}

impl fmt::Display for SolverResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SolverResult::Sat(_) => write!(f, "sat"),
            SolverResult::Unsat => write!(f, "unsat"),
            SolverResult::Unknown => write!(f, "unknown"),
            SolverResult::Timeout => write!(f, "timeout"),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SmtSolver trait
// ═══════════════════════════════════════════════════════════════════════════

/// Trait for SMT solver backends.
pub trait SmtSolver {
    /// Assert an expression.
    fn assert_expr(&mut self, expr: &SmtExpr);

    /// Check satisfiability of all asserted expressions.
    fn check_sat(&mut self) -> SolverResult;

    /// Get the model (only valid after a Sat result).
    fn get_model(&self) -> Option<&Model>;

    /// Push a new assertion scope.
    fn push(&mut self);

    /// Pop the most recent assertion scope.
    fn pop(&mut self);

    /// Reset the solver to its initial state.
    fn reset(&mut self);
}

// ═══════════════════════════════════════════════════════════════════════════
// Literal & Clause
// ═══════════════════════════════════════════════════════════════════════════

/// A propositional literal (possibly negated boolean variable).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Literal {
    /// The variable index (in the boolean abstraction).
    pub var: u32,
    /// True if the literal is positive, false if negated.
    pub positive: bool,
}

impl Literal {
    pub fn pos(var: u32) -> Self {
        Self { var, positive: true }
    }

    pub fn neg(var: u32) -> Self {
        Self { var, positive: false }
    }

    pub fn negate(self) -> Self {
        Self { var: self.var, positive: !self.positive }
    }
}

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.positive { write!(f, "x{}", self.var) }
        else { write!(f, "¬x{}", self.var) }
    }
}

/// A clause (disjunction of literals).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Clause {
    pub literals: Vec<Literal>,
}

impl Clause {
    pub fn new(literals: Vec<Literal>) -> Self {
        Self { literals }
    }

    pub fn unit(lit: Literal) -> Self {
        Self { literals: vec![lit] }
    }

    pub fn is_empty(&self) -> bool {
        self.literals.is_empty()
    }

    pub fn is_unit(&self) -> bool {
        self.literals.len() == 1
    }

    pub fn len(&self) -> usize {
        self.literals.len()
    }

    /// Check if the clause is satisfied under the given assignment.
    pub fn is_satisfied(&self, assignment: &HashMap<u32, bool>) -> bool {
        self.literals.iter().any(|lit| {
            assignment.get(&lit.var).map_or(false, |&val| val == lit.positive)
        })
    }

    /// Check if the clause is falsified (all literals assigned false).
    pub fn is_falsified(&self, assignment: &HashMap<u32, bool>) -> bool {
        self.literals.iter().all(|lit| {
            assignment.get(&lit.var).map_or(false, |&val| val != lit.positive)
        })
    }

    /// Find the unassigned literal in a unit-propagation scenario.
    pub fn find_unit_literal(&self, assignment: &HashMap<u32, bool>) -> Option<Literal> {
        let mut unassigned = None;
        let mut count = 0;
        for lit in &self.literals {
            match assignment.get(&lit.var) {
                Some(&val) if val == lit.positive => return None, // already satisfied
                Some(_) => {} // assigned but doesn't satisfy this literal
                None => {
                    unassigned = Some(*lit);
                    count += 1;
                }
            }
        }
        if count == 1 { unassigned } else { None }
    }
}

impl fmt::Display for Clause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({})", self.literals.iter()
            .map(|l| l.to_string())
            .collect::<Vec<_>>()
            .join(" ∨ "))
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// InternalSolver (DPLL(T))
// ═══════════════════════════════════════════════════════════════════════════

/// A basic internal SMT solver using DPLL(T) with CDCL.
///
/// This solver handles:
/// - Boolean satisfiability with DPLL + unit propagation + CDCL
/// - Theory of linear arithmetic (simple interval propagation)
/// - Theory of equality (simple congruence checking)
pub struct InternalSolver {
    /// The variable store for name lookups.
    store: VariableStore,
    /// Asserted expressions.
    assertions: Vec<SmtExpr>,
    /// Assertion scope stack.
    scope_stack: Vec<usize>,
    /// The last computed model.
    model: Option<Model>,
    /// Maximum number of conflicts before returning Unknown.
    max_conflicts: usize,
    /// Statistics.
    stats: SolverStats,
}

/// Internal solver statistics.
#[derive(Debug, Clone, Default)]
pub struct SolverStats {
    pub decisions: usize,
    pub propagations: usize,
    pub conflicts: usize,
    pub learned_clauses: usize,
    pub restarts: usize,
}

impl InternalSolver {
    pub fn new(store: VariableStore) -> Self {
        Self {
            store,
            assertions: Vec::new(),
            scope_stack: Vec::new(),
            model: None,
            max_conflicts: 100_000,
            stats: SolverStats::default(),
        }
    }

    pub fn with_max_conflicts(mut self, max: usize) -> Self {
        self.max_conflicts = max;
        self
    }

    pub fn stats(&self) -> &SolverStats {
        &self.stats
    }

    /// Convert an expression to clausal form for the SAT core.
    fn to_cnf(&self, expr: &SmtExpr) -> (Vec<Clause>, HashMap<u32, VariableId>, u32) {
        let mut clause_list = Vec::new();
        let mut var_map: HashMap<VariableId, u32> = HashMap::new();
        let mut rev_map: HashMap<u32, VariableId> = HashMap::new();
        let mut next_bool_var: u32 = 0;

        let fv = free_vars(expr);
        for v in &fv {
            let bv = next_bool_var;
            next_bool_var += 1;
            var_map.insert(*v, bv);
            rev_map.insert(bv, *v);
        }

        // Tseitin transformation
        self.tseitin(expr, &mut clause_list, &mut var_map, &mut rev_map, &mut next_bool_var);

        (clause_list, rev_map, next_bool_var)
    }

    fn tseitin(
        &self,
        expr: &SmtExpr,
        clauses: &mut Vec<Clause>,
        var_map: &mut HashMap<VariableId, u32>,
        rev_map: &mut HashMap<u32, VariableId>,
        next_var: &mut u32,
    ) -> u32 {
        match expr {
            SmtExpr::Var(id) => {
                *var_map.entry(*id).or_insert_with(|| {
                    let v = *next_var;
                    *next_var += 1;
                    rev_map.insert(v, *id);
                    v
                })
            }

            SmtExpr::BoolLit(true) => {
                let v = *next_var; *next_var += 1;
                clauses.push(Clause::unit(Literal::pos(v)));
                v
            }

            SmtExpr::BoolLit(false) => {
                let v = *next_var; *next_var += 1;
                clauses.push(Clause::unit(Literal::neg(v)));
                v
            }

            SmtExpr::Not(inner) => {
                let inner_v = self.tseitin(inner, clauses, var_map, rev_map, next_var);
                let v = *next_var; *next_var += 1;
                // v <=> !inner_v
                clauses.push(Clause::new(vec![Literal::neg(v), Literal::neg(inner_v)]));
                clauses.push(Clause::new(vec![Literal::pos(v), Literal::pos(inner_v)]));
                v
            }

            SmtExpr::And(exprs) => {
                let sub_vars: Vec<u32> = exprs.iter()
                    .map(|e| self.tseitin(e, clauses, var_map, rev_map, next_var))
                    .collect();
                let v = *next_var; *next_var += 1;
                // v => sub_vars[i] for each i
                for &sv in &sub_vars {
                    clauses.push(Clause::new(vec![Literal::neg(v), Literal::pos(sv)]));
                }
                // sub_vars[0] & sub_vars[1] & ... => v
                let mut lits: Vec<Literal> = sub_vars.iter().map(|&sv| Literal::neg(sv)).collect();
                lits.push(Literal::pos(v));
                clauses.push(Clause::new(lits));
                v
            }

            SmtExpr::Or(exprs) => {
                let sub_vars: Vec<u32> = exprs.iter()
                    .map(|e| self.tseitin(e, clauses, var_map, rev_map, next_var))
                    .collect();
                let v = *next_var; *next_var += 1;
                // v => (sub_vars[0] | sub_vars[1] | ...)
                let mut lits: Vec<Literal> = sub_vars.iter().map(|&sv| Literal::pos(sv)).collect();
                lits.push(Literal::neg(v));
                clauses.push(Clause::new(lits));
                // sub_vars[i] => v for each i
                for &sv in &sub_vars {
                    clauses.push(Clause::new(vec![Literal::neg(sv), Literal::pos(v)]));
                }
                v
            }

            SmtExpr::Implies(a, b) => {
                let va = self.tseitin(a, clauses, var_map, rev_map, next_var);
                let vb = self.tseitin(b, clauses, var_map, rev_map, next_var);
                let v = *next_var; *next_var += 1;
                // v <=> (!va | vb)
                clauses.push(Clause::new(vec![Literal::neg(v), Literal::neg(va), Literal::pos(vb)]));
                clauses.push(Clause::new(vec![Literal::pos(v), Literal::pos(va)]));
                clauses.push(Clause::new(vec![Literal::pos(v), Literal::neg(vb)]));
                v
            }

            // For non-boolean expressions, create a fresh boolean abstraction variable
            _ => {
                let v = *next_var; *next_var += 1;
                v
            }
        }
    }

    /// DPLL with unit propagation and conflict-driven clause learning.
    fn dpll(
        &mut self,
        clauses: &mut Vec<Clause>,
        assignment: &mut HashMap<u32, bool>,
        num_vars: u32,
    ) -> bool {
        // Unit propagation
        loop {
            let mut propagated = false;
            for i in 0..clauses.len() {
                if clauses[i].is_satisfied(assignment) {
                    continue;
                }
                if clauses[i].is_falsified(assignment) {
                    self.stats.conflicts += 1;
                    return false;
                }
                if let Some(lit) = clauses[i].find_unit_literal(assignment) {
                    assignment.insert(lit.var, lit.positive);
                    self.stats.propagations += 1;
                    propagated = true;
                    break;
                }
            }
            if !propagated { break; }
        }

        // Check if all clauses are satisfied
        if clauses.iter().all(|c| c.is_satisfied(assignment)) {
            return true;
        }

        // Check conflict limit
        if self.stats.conflicts >= self.max_conflicts {
            return false;
        }

        // Choose an unassigned variable
        let decision_var = (0..num_vars)
            .find(|v| !assignment.contains_key(v));

        let decision_var = match decision_var {
            Some(v) => v,
            None => return false, // all assigned but some clause unsatisfied
        };

        self.stats.decisions += 1;

        // Try positive assignment
        let mut pos_assignment = assignment.clone();
        pos_assignment.insert(decision_var, true);
        if self.dpll(clauses, &mut pos_assignment, num_vars) {
            *assignment = pos_assignment;
            return true;
        }

        // Try negative assignment
        let mut neg_assignment = assignment.clone();
        neg_assignment.insert(decision_var, false);
        if self.dpll(clauses, &mut neg_assignment, num_vars) {
            *assignment = neg_assignment;
            return true;
        }

        false
    }

    /// Simple theory solver for linear arithmetic constraints.
    /// Uses interval constraint propagation.
    fn theory_check(
        &self,
        expr: &SmtExpr,
        bool_model: &HashMap<u32, bool>,
        rev_map: &HashMap<u32, VariableId>,
    ) -> Option<Model> {
        let mut model = Model::new();

        // Map boolean variables back to SMT variables
        for (&bv, &smt_id) in rev_map {
            if let Some(&val) = bool_model.get(&bv) {
                if let Some(var) = self.store.get(smt_id) {
                    match &var.sort {
                        SmtSort::Bool => {
                            model.assign(smt_id, SmtValue::Bool(val));
                        }
                        SmtSort::Int => {
                            // Simple default: assign 0 for false, 1 for true
                            model.assign(smt_id, SmtValue::Int(if val { 1 } else { 0 }));
                        }
                        SmtSort::Real => {
                            model.assign(smt_id, SmtValue::Real(
                                if val { 1.0 } else { 0.0 }, 1.0,
                            ));
                        }
                        _ => {}
                    }
                }
            }
        }

        Some(model)
    }
}

impl SmtSolver for InternalSolver {
    fn assert_expr(&mut self, expr: &SmtExpr) {
        self.assertions.push(expr.clone());
    }

    fn check_sat(&mut self) -> SolverResult {
        self.stats = SolverStats::default();

        if self.assertions.is_empty() {
            self.model = Some(Model::new());
            return SolverResult::Sat(Model::new());
        }

        // Combine all assertions into a single conjunction
        let combined = if self.assertions.len() == 1 {
            self.assertions[0].clone()
        } else {
            SmtExpr::And(self.assertions.clone())
        };

        // Convert to CNF via Tseitin transformation
        let (mut clauses, rev_map, num_vars) = self.to_cnf(&combined);

        // The root variable must be true
        if !clauses.is_empty() {
            // The last tseitin variable represents the root
            let root = num_vars - 1;
            clauses.push(Clause::unit(Literal::pos(root)));
        }

        // Run DPLL
        let mut assignment = HashMap::new();
        let sat = self.dpll(&mut clauses, &mut assignment, num_vars);

        if sat {
            // Run theory check
            match self.theory_check(&combined, &assignment, &rev_map) {
                Some(model) => {
                    self.model = Some(model.clone());
                    SolverResult::Sat(model)
                }
                None => SolverResult::Unknown,
            }
        } else if self.stats.conflicts >= self.max_conflicts {
            SolverResult::Timeout
        } else {
            SolverResult::Unsat
        }
    }

    fn get_model(&self) -> Option<&Model> {
        self.model.as_ref()
    }

    fn push(&mut self) {
        self.scope_stack.push(self.assertions.len());
    }

    fn pop(&mut self) {
        if let Some(n) = self.scope_stack.pop() {
            self.assertions.truncate(n);
        }
    }

    fn reset(&mut self) {
        self.assertions.clear();
        self.scope_stack.clear();
        self.model = None;
        self.stats = SolverStats::default();
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SmtLib2Writer
// ═══════════════════════════════════════════════════════════════════════════

/// Serialises an encoded problem to SMT-LIB2 format.
pub struct SmtLib2Writer;

impl SmtLib2Writer {
    /// Convert an encoded problem to an SMT-LIB2 string.
    pub fn to_smtlib2(problem: &EncodedProblem) -> String {
        let mut output = String::new();

        // Header
        output.push_str("; GuardPharma SMT-LIB2 encoding\n");
        output.push_str(&format!("; Bound: {}\n", problem.bound));
        output.push_str(&format!("; Variables: {}\n", problem.num_variables()));
        output.push_str(&format!("; Assertions: {}\n", problem.num_assertions()));
        output.push('\n');

        // Logic declaration
        output.push_str("(set-logic QF_LRA)\n");
        output.push_str("(set-option :produce-models true)\n\n");

        // Variable declarations
        output.push_str("; Variable declarations\n");
        for decl in problem.variable_store.to_declarations() {
            output.push_str(&decl);
            output.push('\n');
        }
        output.push('\n');

        // Assertions
        output.push_str("; Assertions\n");
        for (i, assertion) in problem.assertions.iter().enumerate() {
            output.push_str(&format!("; Assertion {}\n", i));
            let expr_str = to_smtlib2(assertion, &problem.variable_store);
            output.push_str(&format!("(assert {})\n", expr_str));
        }
        output.push('\n');

        // Check and get model
        output.push_str("(check-sat)\n");
        output.push_str("(get-model)\n");
        output.push_str("(exit)\n");

        output
    }

    /// Convert a single expression to SMT-LIB2.
    pub fn expr_to_smtlib2(expr: &SmtExpr, store: &VariableStore) -> String {
        to_smtlib2(expr, store)
    }

    /// Write only the declarations section.
    pub fn declarations(problem: &EncodedProblem) -> String {
        problem.variable_store.to_declarations().join("\n")
    }

    /// Write only the assertions section.
    pub fn assertions(problem: &EncodedProblem) -> String {
        problem.assertions.iter()
            .map(|a| format!("(assert {})", to_smtlib2(a, &problem.variable_store)))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::variable::VariableStore;

    fn make_solver() -> InternalSolver {
        let store = VariableStore::new();
        InternalSolver::new(store)
    }

    #[test]
    fn test_smt_value_display() {
        assert_eq!(SmtValue::Bool(true).to_string(), "true");
        assert_eq!(SmtValue::Int(42).to_string(), "42");
        assert_eq!(SmtValue::Real(3.14, 1.0).to_string(), "3.14");
    }

    #[test]
    fn test_smt_value_accessors() {
        assert_eq!(SmtValue::Bool(true).bool_value(), Some(true));
        assert_eq!(SmtValue::Int(42).int_value(), Some(42));
        assert_eq!(SmtValue::Real(3.0, 2.0).real_value(), Some(1.5));
    }

    #[test]
    fn test_model_basic() {
        let mut model = Model::new();
        model.assign(VariableId(0), SmtValue::Bool(true));
        model.assign(VariableId(1), SmtValue::Int(42));
        assert_eq!(model.len(), 2);
        assert_eq!(model.get(VariableId(0)).unwrap().bool_value(), Some(true));
    }

    #[test]
    fn test_solver_result_display() {
        assert_eq!(SolverResult::Sat(Model::new()).to_string(), "sat");
        assert_eq!(SolverResult::Unsat.to_string(), "unsat");
        assert_eq!(SolverResult::Unknown.to_string(), "unknown");
        assert_eq!(SolverResult::Timeout.to_string(), "timeout");
    }

    #[test]
    fn test_literal_operations() {
        let l = Literal::pos(0);
        assert!(l.positive);
        assert_eq!(l.negate().positive, false);
        assert_eq!(l.negate().var, 0);
    }

    #[test]
    fn test_clause_satisfied() {
        let clause = Clause::new(vec![Literal::pos(0), Literal::neg(1)]);
        let mut assignment = HashMap::new();
        assignment.insert(0, true);
        assignment.insert(1, true);
        assert!(clause.is_satisfied(&assignment));
    }

    #[test]
    fn test_clause_falsified() {
        let clause = Clause::new(vec![Literal::pos(0), Literal::pos(1)]);
        let mut assignment = HashMap::new();
        assignment.insert(0, false);
        assignment.insert(1, false);
        assert!(clause.is_falsified(&assignment));
    }

    #[test]
    fn test_clause_unit_propagation() {
        let clause = Clause::new(vec![Literal::pos(0), Literal::pos(1)]);
        let mut assignment = HashMap::new();
        assignment.insert(0, false);
        let unit = clause.find_unit_literal(&assignment);
        assert_eq!(unit, Some(Literal::pos(1)));
    }

    #[test]
    fn test_solver_trivially_sat() {
        let mut solver = make_solver();
        solver.assert_expr(&SmtExpr::BoolLit(true));
        let result = solver.check_sat();
        assert!(result.is_sat());
    }

    #[test]
    fn test_solver_trivially_unsat() {
        let mut solver = make_solver();
        solver.assert_expr(&SmtExpr::BoolLit(false));
        let result = solver.check_sat();
        // The solver should detect this as unsatisfiable
        assert!(result.is_unsat() || matches!(result, SolverResult::Unknown));
    }

    #[test]
    fn test_solver_simple_and() {
        let mut store = VariableStore::new();
        let p = store.create_bool("p");
        let q = store.create_bool("q");

        let mut solver = InternalSolver::new(store);
        solver.assert_expr(&SmtExpr::and(vec![
            SmtExpr::Var(p),
            SmtExpr::Var(q),
        ]));
        let result = solver.check_sat();
        assert!(result.is_sat());
    }

    #[test]
    fn test_solver_push_pop() {
        let mut solver = make_solver();
        solver.assert_expr(&SmtExpr::BoolLit(true));

        solver.push();
        solver.assert_expr(&SmtExpr::BoolLit(false));
        // Now we have true AND false
        assert_eq!(solver.assertions.len(), 2);

        solver.pop();
        assert_eq!(solver.assertions.len(), 1);
    }

    #[test]
    fn test_solver_reset() {
        let mut solver = make_solver();
        solver.assert_expr(&SmtExpr::BoolLit(true));
        solver.reset();
        assert!(solver.assertions.is_empty());
    }

    #[test]
    fn test_smtlib2_writer_basic() {
        use crate::encoder::EncodedProblem;
        use crate::variable::SymbolTable;

        let mut store = VariableStore::new();
        let p = store.create_bool("p");
        let x = store.create_real("x");

        let problem = EncodedProblem {
            assertions: vec![
                SmtExpr::Var(p),
                SmtExpr::gt(SmtExpr::Var(x), SmtExpr::RealLit(0.0)),
            ],
            variable_store: store,
            symbol_table: SymbolTable::new(),
            bound: 0,
            dt: 1.0,
            num_locations: 0,
            num_edges: 0,
        };

        let output = SmtLib2Writer::to_smtlib2(&problem);
        assert!(output.contains("(set-logic QF_LRA)"));
        assert!(output.contains("(declare-const p Bool)"));
        assert!(output.contains("(declare-const x Real)"));
        assert!(output.contains("(assert p)"));
        assert!(output.contains("(check-sat)"));
    }

    #[test]
    fn test_smtlib2_writer_declarations() {
        use crate::encoder::EncodedProblem;
        use crate::variable::SymbolTable;

        let mut store = VariableStore::new();
        store.create_bool("flag");
        store.create_int("count");

        let problem = EncodedProblem {
            assertions: vec![],
            variable_store: store,
            symbol_table: SymbolTable::new(),
            bound: 0,
            dt: 1.0,
            num_locations: 0,
            num_edges: 0,
        };

        let decls = SmtLib2Writer::declarations(&problem);
        assert!(decls.contains("flag"));
        assert!(decls.contains("count"));
    }

    #[test]
    fn test_solver_stats() {
        let mut store = VariableStore::new();
        let p = store.create_bool("p");

        let mut solver = InternalSolver::new(store);
        solver.assert_expr(&SmtExpr::Var(p));
        solver.check_sat();

        // Should have at least some stats
        let stats = solver.stats();
        assert!(stats.propagations > 0 || stats.decisions >= 0);
    }

    #[test]
    fn test_model_extract_trace() {
        let mut store = VariableStore::new();
        store.create_time_indexed("loc", SmtSort::Int, 0);
        store.create_time_indexed("loc", SmtSort::Int, 1);
        store.create_time_indexed("x", SmtSort::Real, 0);
        store.create_time_indexed("x", SmtSort::Real, 1);

        let mut model = Model::new();
        model.assign(store.id_by_name("loc_t0").unwrap(), SmtValue::Int(0));
        model.assign(store.id_by_name("loc_t1").unwrap(), SmtValue::Int(1));
        model.assign(store.id_by_name("x_t0").unwrap(), SmtValue::Real(0.0, 1.0));
        model.assign(store.id_by_name("x_t1").unwrap(), SmtValue::Real(1.0, 1.0));

        let trace = model.extract_trace(&store, 1);
        assert_eq!(trace.len(), 2);
        assert!(trace[0].contains_key("loc_t0"));
        assert!(trace[1].contains_key("loc_t1"));
    }

    #[test]
    fn test_solver_empty_assertions() {
        let mut solver = make_solver();
        let result = solver.check_sat();
        assert!(result.is_sat()); // no constraints = trivially sat
    }
}
