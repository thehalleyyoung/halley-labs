// SMT solver: DPLL(T) combining CDCL SAT solving with theory propagators.
// Includes: SmtFormula, Tseitin transformation, FormulaBuilder, Model.

use crate::cdcl::{CdclSolver, SatResult};
use crate::config::SolverConfig;
use crate::theory::{TheoryLemma, TheoryPropagator, TheoryResult};
use crate::variable::{Assignment, Literal, LiteralVec, Variable, VariableManager};
use smallvec::smallvec;
use std::collections::HashMap;
use std::fmt;

// ── SmtFormula ────────────────────────────────────────────────────────────────

/// SMT formula combining Boolean structure with theory atoms.
#[derive(Debug, Clone)]
pub enum SmtFormula {
    // Boolean connectives
    BoolVar(u32),
    BoolConst(bool),
    And(Vec<SmtFormula>),
    Or(Vec<SmtFormula>),
    Not(Box<SmtFormula>),
    Implies(Box<SmtFormula>, Box<SmtFormula>),
    Iff(Box<SmtFormula>, Box<SmtFormula>),

    // Integer arithmetic
    IntVar(u32),
    IntConst(i64),
    Add(Box<SmtFormula>, Box<SmtFormula>),
    Mul(i64, Box<SmtFormula>), // scalar multiplication
    Neg(Box<SmtFormula>),

    // Comparisons (theory atoms)
    Le(Box<SmtFormula>, Box<SmtFormula>),
    Lt(Box<SmtFormula>, Box<SmtFormula>),
    Eq(Box<SmtFormula>, Box<SmtFormula>),
    Ge(Box<SmtFormula>, Box<SmtFormula>),
    Gt(Box<SmtFormula>, Box<SmtFormula>),
}

impl SmtFormula {
    /// Whether this formula is an atom (leaf-level Boolean or theory atom).
    pub fn is_atom(&self) -> bool {
        matches!(
            self,
            SmtFormula::BoolVar(_)
                | SmtFormula::BoolConst(_)
                | SmtFormula::Le(..)
                | SmtFormula::Lt(..)
                | SmtFormula::Eq(..)
                | SmtFormula::Ge(..)
                | SmtFormula::Gt(..)
        )
    }

    /// Whether this is a Boolean connective.
    pub fn is_connective(&self) -> bool {
        matches!(
            self,
            SmtFormula::And(_)
                | SmtFormula::Or(_)
                | SmtFormula::Not(_)
                | SmtFormula::Implies(..)
                | SmtFormula::Iff(..)
        )
    }
}

// ── FormulaBuilder ────────────────────────────────────────────────────────────

/// Ergonomic API for building SMT formulas.
pub struct FormulaBuilder {
    next_bool_var: u32,
    next_int_var: u32,
}

impl FormulaBuilder {
    pub fn new() -> Self {
        FormulaBuilder {
            next_bool_var: 1,
            next_int_var: 1,
        }
    }

    /// Create a new Boolean variable.
    pub fn bool_var(&mut self) -> SmtFormula {
        let id = self.next_bool_var;
        self.next_bool_var += 1;
        SmtFormula::BoolVar(id)
    }

    /// Create a new integer variable.
    pub fn int_var(&mut self) -> SmtFormula {
        let id = self.next_int_var;
        self.next_int_var += 1;
        SmtFormula::IntVar(id)
    }

    /// Integer constant.
    pub fn int_const(&self, val: i64) -> SmtFormula {
        SmtFormula::IntConst(val)
    }

    /// Boolean constant.
    pub fn bool_const(&self, val: bool) -> SmtFormula {
        SmtFormula::BoolConst(val)
    }

    /// Conjunction.
    pub fn and(&self, formulas: Vec<SmtFormula>) -> SmtFormula {
        SmtFormula::And(formulas)
    }

    /// Disjunction.
    pub fn or(&self, formulas: Vec<SmtFormula>) -> SmtFormula {
        SmtFormula::Or(formulas)
    }

    /// Negation.
    pub fn not(&self, f: SmtFormula) -> SmtFormula {
        SmtFormula::Not(Box::new(f))
    }

    /// Implication.
    pub fn implies(&self, a: SmtFormula, b: SmtFormula) -> SmtFormula {
        SmtFormula::Implies(Box::new(a), Box::new(b))
    }

    /// Biconditional.
    pub fn iff(&self, a: SmtFormula, b: SmtFormula) -> SmtFormula {
        SmtFormula::Iff(Box::new(a), Box::new(b))
    }

    /// Less than or equal.
    pub fn le(&self, a: SmtFormula, b: SmtFormula) -> SmtFormula {
        SmtFormula::Le(Box::new(a), Box::new(b))
    }

    /// Less than.
    pub fn lt(&self, a: SmtFormula, b: SmtFormula) -> SmtFormula {
        SmtFormula::Lt(Box::new(a), Box::new(b))
    }

    /// Equal.
    pub fn eq(&self, a: SmtFormula, b: SmtFormula) -> SmtFormula {
        SmtFormula::Eq(Box::new(a), Box::new(b))
    }

    /// Greater than or equal.
    pub fn ge(&self, a: SmtFormula, b: SmtFormula) -> SmtFormula {
        SmtFormula::Ge(Box::new(a), Box::new(b))
    }

    /// Greater than.
    pub fn gt(&self, a: SmtFormula, b: SmtFormula) -> SmtFormula {
        SmtFormula::Gt(Box::new(a), Box::new(b))
    }

    /// Addition.
    pub fn add(&self, a: SmtFormula, b: SmtFormula) -> SmtFormula {
        SmtFormula::Add(Box::new(a), Box::new(b))
    }

    /// Scalar multiplication.
    pub fn mul(&self, coeff: i64, a: SmtFormula) -> SmtFormula {
        SmtFormula::Mul(coeff, Box::new(a))
    }
}

impl Default for FormulaBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tseitin ───────────────────────────────────────────────────────────────────

/// Tseitin transformation: converts arbitrary formulas to equisatisfiable CNF.
pub struct Tseitin {
    var_manager: VariableManager,
    clauses: Vec<LiteralVec>,
    /// Maps sub-formula identities to their Tseitin variables.
    formula_vars: HashMap<u64, Variable>,
    next_formula_id: u64,
    /// Maps Boolean variable ids in the formula to SAT variables.
    bool_var_map: HashMap<u32, Variable>,
    /// Maps theory atoms to their guard variables.
    theory_atom_vars: Vec<(Variable, SmtFormula)>,
}

impl Tseitin {
    pub fn new() -> Self {
        Tseitin {
            var_manager: VariableManager::new(),
            clauses: Vec::new(),
            formula_vars: HashMap::new(),
            next_formula_id: 0,
            bool_var_map: HashMap::new(),
            theory_atom_vars: Vec::new(),
        }
    }

    /// Transform a formula to CNF and return the clauses + variable mapping.
    pub fn transform(&mut self, formula: &SmtFormula) -> TseitinResult {
        let root_lit = self.encode(formula);
        // Assert the root literal.
        self.clauses.push(smallvec![root_lit]);

        TseitinResult {
            clauses: self.clauses.clone(),
            num_vars: self.var_manager.variable_count() as usize,
            bool_var_map: self.bool_var_map.clone(),
            theory_atoms: self.theory_atom_vars.clone(),
        }
    }

    fn fresh_var(&mut self) -> Variable {
        self.var_manager.new_variable()
    }

    fn alloc_formula_id(&mut self) -> u64 {
        let id = self.next_formula_id;
        self.next_formula_id += 1;
        id
    }

    /// Recursively encode a formula, returning the literal representing its truth.
    fn encode(&mut self, formula: &SmtFormula) -> Literal {
        match formula {
            SmtFormula::BoolVar(id) => {
                let var = self
                    .bool_var_map
                    .entry(*id)
                    .or_insert_with(|| self.var_manager.new_variable());
                var.positive()
            }
            SmtFormula::BoolConst(true) => {
                let v = self.fresh_var();
                self.clauses.push(smallvec![v.positive()]);
                v.positive()
            }
            SmtFormula::BoolConst(false) => {
                let v = self.fresh_var();
                self.clauses.push(smallvec![v.negative()]);
                v.negative()
            }
            SmtFormula::Not(inner) => {
                let inner_lit = self.encode(inner);
                inner_lit.negated()
            }
            SmtFormula::And(children) => {
                if children.is_empty() {
                    return self.encode(&SmtFormula::BoolConst(true));
                }
                let child_lits: Vec<Literal> = children.iter().map(|c| self.encode(c)).collect();
                let gate = self.fresh_var();
                let gate_lit = gate.positive();

                // gate → child_i  (for each i):  ¬gate ∨ child_i
                for &cl in &child_lits {
                    self.clauses.push(smallvec![gate_lit.negated(), cl]);
                }
                // child_1 ∧ ... ∧ child_n → gate:  ¬child_1 ∨ ... ∨ ¬child_n ∨ gate
                let mut clause: LiteralVec = child_lits.iter().map(|l| l.negated()).collect();
                clause.push(gate_lit);
                self.clauses.push(clause);

                gate_lit
            }
            SmtFormula::Or(children) => {
                if children.is_empty() {
                    return self.encode(&SmtFormula::BoolConst(false));
                }
                let child_lits: Vec<Literal> = children.iter().map(|c| self.encode(c)).collect();
                let gate = self.fresh_var();
                let gate_lit = gate.positive();

                // gate → child_1 ∨ ... ∨ child_n:  ¬gate ∨ child_1 ∨ ... ∨ child_n
                let mut clause: LiteralVec = smallvec![gate_lit.negated()];
                clause.extend(child_lits.iter().cloned());
                self.clauses.push(clause);

                // child_i → gate (for each i):  ¬child_i ∨ gate
                for &cl in &child_lits {
                    self.clauses.push(smallvec![cl.negated(), gate_lit]);
                }

                gate_lit
            }
            SmtFormula::Implies(a, b) => {
                // a → b ≡ ¬a ∨ b
                let or_formula = SmtFormula::Or(vec![
                    SmtFormula::Not(a.clone()),
                    *b.clone(),
                ]);
                self.encode(&or_formula)
            }
            SmtFormula::Iff(a, b) => {
                // a ↔ b ≡ (a → b) ∧ (b → a)
                let a_lit = self.encode(a);
                let b_lit = self.encode(b);
                let gate = self.fresh_var();
                let gate_lit = gate.positive();

                // gate → (a_lit ↔ b_lit):
                // gate → (¬a ∨ b):  ¬gate ∨ ¬a ∨ b
                self.clauses
                    .push(smallvec![gate_lit.negated(), a_lit.negated(), b_lit]);
                // gate → (¬b ∨ a):  ¬gate ∨ ¬b ∨ a
                self.clauses
                    .push(smallvec![gate_lit.negated(), b_lit.negated(), a_lit]);
                // (a ↔ b) → gate:
                // (a ∧ b) → gate:  ¬a ∨ ¬b ∨ gate
                self.clauses
                    .push(smallvec![a_lit.negated(), b_lit.negated(), gate_lit]);
                // (¬a ∧ ¬b) → gate:  a ∨ b ∨ gate
                self.clauses.push(smallvec![a_lit, b_lit, gate_lit]);

                gate_lit
            }
            // Theory atoms: create a Boolean variable to represent them.
            SmtFormula::Le(..)
            | SmtFormula::Lt(..)
            | SmtFormula::Eq(..)
            | SmtFormula::Ge(..)
            | SmtFormula::Gt(..) => {
                let var = self.fresh_var();
                self.theory_atom_vars.push((var, formula.clone()));
                var.positive()
            }
            // Arithmetic terms shouldn't appear at the top level of a Boolean formula.
            SmtFormula::IntVar(_)
            | SmtFormula::IntConst(_)
            | SmtFormula::Add(..)
            | SmtFormula::Mul(..)
            | SmtFormula::Neg(_) => {
                // Shouldn't be used as a Boolean formula; treat as error by returning a fresh var.
                let v = self.fresh_var();
                v.positive()
            }
        }
    }
}

impl Default for Tseitin {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of Tseitin transformation.
#[derive(Debug, Clone)]
pub struct TseitinResult {
    pub clauses: Vec<LiteralVec>,
    pub num_vars: usize,
    pub bool_var_map: HashMap<u32, Variable>,
    pub theory_atoms: Vec<(Variable, SmtFormula)>,
}

// ── Model ─────────────────────────────────────────────────────────────────────

/// An SMT model: maps Boolean and integer variables to values.
#[derive(Debug, Clone, Default)]
pub struct Model {
    pub bool_values: HashMap<u32, bool>,
    pub int_values: HashMap<u32, i64>,
}

impl Model {
    pub fn new() -> Self {
        Model {
            bool_values: HashMap::new(),
            int_values: HashMap::new(),
        }
    }

    pub fn set_bool(&mut self, var_id: u32, value: bool) {
        self.bool_values.insert(var_id, value);
    }

    pub fn set_int(&mut self, var_id: u32, value: i64) {
        self.int_values.insert(var_id, value);
    }

    pub fn get_bool(&self, var_id: u32) -> Option<bool> {
        self.bool_values.get(&var_id).copied()
    }

    pub fn get_int(&self, var_id: u32) -> Option<i64> {
        self.int_values.get(&var_id).copied()
    }
}

impl fmt::Display for Model {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Model {{ bools: {:?}, ints: {:?} }}", self.bool_values, self.int_values)
    }
}

// ── SmtResult ─────────────────────────────────────────────────────────────────

/// Result of an SMT solving attempt.
#[derive(Debug, Clone)]
pub enum SmtResult {
    Sat(Model),
    Unsat,
    Unknown(String),
}

impl SmtResult {
    pub fn is_sat(&self) -> bool {
        matches!(self, SmtResult::Sat(_))
    }
    pub fn is_unsat(&self) -> bool {
        matches!(self, SmtResult::Unsat)
    }
}

// ── SmtSolver ─────────────────────────────────────────────────────────────────

/// DPLL(T) SMT solver combining CDCL with theory propagators.
pub struct SmtSolver {
    /// The underlying SAT solver.
    sat_solver: CdclSolver,
    /// Theory propagators.
    theories: Vec<Box<dyn TheoryPropagator>>,
    /// Assertions (formulas added by the user).
    assertions: Vec<SmtFormula>,
    /// Solver configuration.
    config: SolverConfig,
    /// Push/pop assertion stack.
    assertion_stack: Vec<usize>,
    /// Mapping from formula bool vars to SAT vars.
    bool_var_map: HashMap<u32, Variable>,
    /// Theory atom guards.
    theory_atoms: Vec<(Variable, SmtFormula)>,
    /// Whether the solver needs re-encoding.
    dirty: bool,
}

impl SmtSolver {
    /// Create a new SMT solver.
    pub fn new(config: SolverConfig) -> Self {
        SmtSolver {
            sat_solver: CdclSolver::new(config.clone()),
            theories: Vec::new(),
            assertions: Vec::new(),
            config,
            assertion_stack: Vec::new(),
            bool_var_map: HashMap::new(),
            theory_atoms: Vec::new(),
            dirty: true,
        }
    }

    /// Create with default configuration.
    pub fn default_solver() -> Self {
        Self::new(SolverConfig::default())
    }

    /// Add a theory propagator.
    pub fn add_theory(&mut self, theory: Box<dyn TheoryPropagator>) {
        self.theories.push(theory);
    }

    /// Add an assertion (formula that must be satisfied).
    pub fn add_assertion(&mut self, formula: SmtFormula) {
        self.assertions.push(formula);
        self.dirty = true;
    }

    /// Push a new assertion context.
    pub fn push(&mut self) {
        self.assertion_stack.push(self.assertions.len());
    }

    /// Pop the last assertion context.
    pub fn pop(&mut self) {
        if let Some(level) = self.assertion_stack.pop() {
            self.assertions.truncate(level);
            self.dirty = true;
        }
    }

    /// Encode all assertions into SAT clauses and check satisfiability.
    pub fn check(&mut self) -> SmtResult {
        if self.dirty {
            self.encode();
        }

        // DPLL(T) loop.
        let max_theory_iterations = 100;
        for _ in 0..max_theory_iterations {
            let sat_result = self.sat_solver.solve();

            match sat_result {
                SatResult::Satisfiable(assignment) => {
                    // Check theories.
                    let theory_ok = self.check_theories(&assignment);
                    match theory_ok {
                        TheoryCheckResult::Consistent => {
                            let model = self.extract_model(&assignment);
                            return SmtResult::Sat(model);
                        }
                        TheoryCheckResult::Inconsistent(lemma) => {
                            // Add theory lemma as a clause and re-solve.
                            self.sat_solver.add_clause_lits(lemma.literals.clone());
                            self.sat_solver.reset();
                            // Re-add all clauses and unit propagations.
                            continue;
                        }
                        TheoryCheckResult::Propagation(lemmas) => {
                            for lemma in lemmas {
                                self.sat_solver.add_clause_lits(lemma.literals.clone());
                            }
                            self.sat_solver.reset();
                            continue;
                        }
                    }
                }
                SatResult::Unsatisfiable(_) => {
                    return SmtResult::Unsat;
                }
                SatResult::Unknown(reason) => {
                    return SmtResult::Unknown(reason);
                }
            }
        }

        SmtResult::Unknown("theory iteration limit".into())
    }

    /// Encode all assertions into CNF via Tseitin transformation.
    fn encode(&mut self) {
        self.sat_solver = CdclSolver::new(self.config.clone());

        // Combine all assertions into a single And.
        let combined = if self.assertions.len() == 1 {
            self.assertions[0].clone()
        } else {
            SmtFormula::And(self.assertions.clone())
        };

        let mut tseitin = Tseitin::new();
        let result = tseitin.transform(&combined);

        self.bool_var_map = result.bool_var_map;
        self.theory_atoms = result.theory_atoms;

        for clause in &result.clauses {
            self.sat_solver.add_clause_lits(clause.clone());
        }

        self.dirty = false;
    }

    /// Check theory consistency.
    fn check_theories(&mut self, assignment: &Assignment) -> TheoryCheckResult {
        for theory in &mut self.theories {
            match theory.check_consistency(assignment) {
                TheoryResult::Consistent => continue,
                TheoryResult::Inconsistent(explanation) => {
                    // Build a conflict clause: negation of the explanation.
                    let clause: LiteralVec = explanation.iter().map(|l| l.negated()).collect();
                    return TheoryCheckResult::Inconsistent(TheoryLemma::new(
                        clause,
                        theory.name(),
                    ));
                }
                TheoryResult::Propagation(lit, explanation) => {
                    let mut clause: LiteralVec = explanation.iter().map(|l| l.negated()).collect();
                    clause.push(lit);
                    return TheoryCheckResult::Propagation(vec![TheoryLemma::new(
                        clause,
                        theory.name(),
                    )]);
                }
            }
        }

        // Also check theory propagations.
        let mut all_lemmas = Vec::new();
        for theory in &mut self.theories {
            let lemmas = theory.propagate(assignment);
            all_lemmas.extend(lemmas);
        }

        if all_lemmas.is_empty() {
            TheoryCheckResult::Consistent
        } else {
            TheoryCheckResult::Propagation(all_lemmas)
        }
    }

    /// Extract a model from a satisfying assignment.
    fn extract_model(&self, assignment: &Assignment) -> Model {
        let mut model = Model::new();

        for (&formula_id, &sat_var) in &self.bool_var_map {
            if let Some(val) = assignment.get(sat_var) {
                model.set_bool(formula_id, val);
            }
        }

        model
    }
}

/// Internal result of theory checking.
enum TheoryCheckResult {
    Consistent,
    Inconsistent(TheoryLemma),
    Propagation(Vec<TheoryLemma>),
}

impl fmt::Debug for SmtSolver {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SmtSolver")
            .field("assertions", &self.assertions.len())
            .field("theories", &self.theories.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_formula_builder_basic() {
        let mut fb = FormulaBuilder::new();
        let a = fb.bool_var();
        let b = fb.bool_var();
        let formula = fb.and(vec![a, b]);
        assert!(matches!(formula, SmtFormula::And(_)));
    }

    #[test]
    fn test_formula_builder_arithmetic() {
        let mut fb = FormulaBuilder::new();
        let x = fb.int_var();
        let c = fb.int_const(5);
        let leq = fb.le(x, c);
        assert!(leq.is_atom());
    }

    #[test]
    fn test_tseitin_single_var() {
        let mut tseitin = Tseitin::new();
        let formula = SmtFormula::BoolVar(1);
        let result = tseitin.transform(&formula);
        assert!(!result.clauses.is_empty());
        assert!(result.num_vars >= 1);
    }

    #[test]
    fn test_tseitin_and() {
        let mut tseitin = Tseitin::new();
        let formula = SmtFormula::And(vec![SmtFormula::BoolVar(1), SmtFormula::BoolVar(2)]);
        let result = tseitin.transform(&formula);
        // Should produce clauses encoding AND gate.
        assert!(result.clauses.len() >= 3);
    }

    #[test]
    fn test_tseitin_or() {
        let mut tseitin = Tseitin::new();
        let formula = SmtFormula::Or(vec![SmtFormula::BoolVar(1), SmtFormula::BoolVar(2)]);
        let result = tseitin.transform(&formula);
        assert!(!result.clauses.is_empty());
    }

    #[test]
    fn test_tseitin_not() {
        let mut tseitin = Tseitin::new();
        let formula = SmtFormula::Not(Box::new(SmtFormula::BoolVar(1)));
        let result = tseitin.transform(&formula);
        assert!(!result.clauses.is_empty());
    }

    #[test]
    fn test_tseitin_implies() {
        let mut tseitin = Tseitin::new();
        let formula = SmtFormula::Implies(
            Box::new(SmtFormula::BoolVar(1)),
            Box::new(SmtFormula::BoolVar(2)),
        );
        let result = tseitin.transform(&formula);
        assert!(!result.clauses.is_empty());
    }

    #[test]
    fn test_tseitin_iff() {
        let mut tseitin = Tseitin::new();
        let formula = SmtFormula::Iff(
            Box::new(SmtFormula::BoolVar(1)),
            Box::new(SmtFormula::BoolVar(2)),
        );
        let result = tseitin.transform(&formula);
        assert!(result.clauses.len() >= 4);
    }

    #[test]
    fn test_tseitin_theory_atom() {
        let mut tseitin = Tseitin::new();
        let formula = SmtFormula::Le(
            Box::new(SmtFormula::IntVar(1)),
            Box::new(SmtFormula::IntConst(5)),
        );
        let result = tseitin.transform(&formula);
        assert_eq!(result.theory_atoms.len(), 1);
    }

    #[test]
    fn test_model_basic() {
        let mut model = Model::new();
        model.set_bool(1, true);
        model.set_int(1, 42);
        assert_eq!(model.get_bool(1), Some(true));
        assert_eq!(model.get_int(1), Some(42));
        assert_eq!(model.get_bool(99), None);
    }

    #[test]
    fn test_smt_solver_pure_boolean_sat() {
        let mut solver = SmtSolver::default_solver();
        // a ∧ b
        solver.add_assertion(SmtFormula::And(vec![
            SmtFormula::BoolVar(1),
            SmtFormula::BoolVar(2),
        ]));
        let result = solver.check();
        assert!(result.is_sat());
        if let SmtResult::Sat(model) = result {
            assert_eq!(model.get_bool(1), Some(true));
            assert_eq!(model.get_bool(2), Some(true));
        }
    }

    #[test]
    fn test_smt_solver_pure_boolean_unsat() {
        let mut solver = SmtSolver::default_solver();
        // a ∧ ¬a
        solver.add_assertion(SmtFormula::And(vec![
            SmtFormula::BoolVar(1),
            SmtFormula::Not(Box::new(SmtFormula::BoolVar(1))),
        ]));
        let result = solver.check();
        assert!(result.is_unsat());
    }

    #[test]
    fn test_smt_solver_or() {
        let mut solver = SmtSolver::default_solver();
        // a ∨ b
        solver.add_assertion(SmtFormula::Or(vec![
            SmtFormula::BoolVar(1),
            SmtFormula::BoolVar(2),
        ]));
        let result = solver.check();
        assert!(result.is_sat());
    }

    #[test]
    fn test_smt_solver_implies() {
        let mut solver = SmtSolver::default_solver();
        // a ∧ (a → b). This should force a=true and b=true.
        solver.add_assertion(SmtFormula::And(vec![
            SmtFormula::BoolVar(1),
            SmtFormula::Implies(
                Box::new(SmtFormula::BoolVar(1)),
                Box::new(SmtFormula::BoolVar(2)),
            ),
        ]));
        let result = solver.check();
        assert!(result.is_sat());
        if let SmtResult::Sat(model) = result {
            assert_eq!(model.get_bool(1), Some(true));
            assert_eq!(model.get_bool(2), Some(true));
        }
    }

    #[test]
    fn test_smt_solver_push_pop() {
        let mut solver = SmtSolver::default_solver();
        solver.add_assertion(SmtFormula::BoolVar(1));
        solver.push();
        solver.add_assertion(SmtFormula::Not(Box::new(SmtFormula::BoolVar(1))));
        let r1 = solver.check();
        assert!(r1.is_unsat());

        solver.pop();
        let r2 = solver.check();
        assert!(r2.is_sat());
    }

    #[test]
    fn test_formula_is_atom() {
        assert!(SmtFormula::BoolVar(1).is_atom());
        assert!(SmtFormula::BoolConst(true).is_atom());
        // Empty And is still the And variant, so is_connective returns true.
        assert!(SmtFormula::And(vec![]).is_connective());
        assert!(SmtFormula::And(vec![SmtFormula::BoolVar(1)]).is_connective());
        assert!(!SmtFormula::IntVar(1).is_atom());
    }

    #[test]
    fn test_smt_result_display() {
        let model = Model::new();
        let s = format!("{}", model);
        assert!(s.contains("Model"));
    }

    #[test]
    fn test_smt_solver_multiple_assertions() {
        let mut solver = SmtSolver::default_solver();
        solver.add_assertion(SmtFormula::BoolVar(1));
        solver.add_assertion(SmtFormula::BoolVar(2));
        solver.add_assertion(SmtFormula::Or(vec![
            SmtFormula::Not(Box::new(SmtFormula::BoolVar(1))),
            SmtFormula::BoolVar(3),
        ]));
        let result = solver.check();
        assert!(result.is_sat());
        if let SmtResult::Sat(model) = result {
            assert_eq!(model.get_bool(1), Some(true));
            assert_eq!(model.get_bool(2), Some(true));
            assert_eq!(model.get_bool(3), Some(true));
        }
    }
}
