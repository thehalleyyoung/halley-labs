//! SAT encoding of memory model checking for LITMUS∞.
//!
//! Provides CNF formula construction, Tseitin transformation,
//! DPLL solver, CDCL solver with clause learning, memory model
//! SAT encoding, and incremental SAT solving.

use std::collections::{HashMap, HashSet, VecDeque, BTreeSet};
use std::fmt;

// ═══════════════════════════════════════════════════════════════════════
// Literal — SAT literal
// ═══════════════════════════════════════════════════════════════════════

/// A SAT literal: a variable with polarity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Literal {
    /// Encoded as var * 2 + (if positive then 0 else 1).
    code: u32,
}

impl Literal {
    /// Create a positive literal for the given variable.
    pub fn positive(var: usize) -> Self {
        Literal { code: (var as u32) * 2 }
    }

    /// Create a negative literal for the given variable.
    pub fn negative(var: usize) -> Self {
        Literal { code: (var as u32) * 2 + 1 }
    }

    /// Create a literal with given polarity.
    pub fn new(var: usize, positive: bool) -> Self {
        if positive { Self::positive(var) } else { Self::negative(var) }
    }

    /// Get the variable index.
    pub fn var(&self) -> usize {
        (self.code / 2) as usize
    }

    /// Is this a positive literal?
    pub fn is_positive(&self) -> bool {
        self.code % 2 == 0
    }

    /// Is this a negative literal?
    pub fn is_negative(&self) -> bool {
        !self.is_positive()
    }

    /// Negate this literal.
    pub fn negate(&self) -> Self {
        Literal { code: self.code ^ 1 }
    }
}

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_positive() {
            write!(f, "x{}", self.var())
        } else {
            write!(f, "¬x{}", self.var())
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Clause — disjunction of literals
// ═══════════════════════════════════════════════════════════════════════

/// A clause: a disjunction (OR) of literals.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Clause {
    /// The literals in this clause.
    pub literals: Vec<Literal>,
}

impl Clause {
    /// Create an empty clause (always false).
    pub fn new() -> Self {
        Clause { literals: Vec::new() }
    }

    /// Create a clause from literals.
    pub fn from_literals(literals: Vec<Literal>) -> Self {
        Clause { literals }
    }

    /// Create a unit clause.
    pub fn unit(lit: Literal) -> Self {
        Clause { literals: vec![lit] }
    }

    /// Add a literal to the clause.
    pub fn add_literal(&mut self, lit: Literal) {
        self.literals.push(lit);
    }

    /// Number of literals.
    pub fn len(&self) -> usize {
        self.literals.len()
    }

    /// Is the clause empty (always false)?
    pub fn is_empty(&self) -> bool {
        self.literals.is_empty()
    }

    /// Is this a unit clause?
    pub fn is_unit(&self) -> bool {
        self.literals.len() == 1
    }

    /// Is this a tautology (contains both x and ¬x)?
    pub fn is_tautology(&self) -> bool {
        let pos: HashSet<usize> = self.literals.iter()
            .filter(|l| l.is_positive()).map(|l| l.var()).collect();
        self.literals.iter()
            .any(|l| l.is_negative() && pos.contains(&l.var()))
    }

    /// Get the literals.
    pub fn literals(&self) -> &[Literal] {
        &self.literals
    }

    /// Remove duplicate literals and sort.
    pub fn normalize(&mut self) {
        self.literals.sort();
        self.literals.dedup();
    }
}

impl fmt::Display for Clause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            write!(f, "⊥")
        } else {
            let parts: Vec<String> = self.literals.iter().map(|l| l.to_string()).collect();
            write!(f, "({})", parts.join(" ∨ "))
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// CnfFormula — conjunction of clauses
// ═══════════════════════════════════════════════════════════════════════

/// A CNF formula: a conjunction (AND) of clauses.
#[derive(Debug, Clone)]
pub struct CnfFormula {
    /// The clauses.
    pub clauses: Vec<Clause>,
    /// Number of variables.
    num_vars: usize,
}

impl CnfFormula {
    /// Create an empty formula (always true).
    pub fn new() -> Self {
        CnfFormula { clauses: Vec::new(), num_vars: 0 }
    }

    /// Create with a known number of variables.
    pub fn with_vars(num_vars: usize) -> Self {
        CnfFormula { clauses: Vec::new(), num_vars }
    }

    /// Allocate a new variable and return its index.
    pub fn new_var(&mut self) -> usize {
        let v = self.num_vars;
        self.num_vars += 1;
        v
    }

    /// Add a clause to the formula.
    pub fn add_clause(&mut self, clause: Clause) {
        // Update num_vars
        for lit in &clause.literals {
            if lit.var() >= self.num_vars {
                self.num_vars = lit.var() + 1;
            }
        }
        self.clauses.push(clause);
    }

    /// Number of variables.
    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    /// Number of clauses.
    pub fn num_clauses(&self) -> usize {
        self.clauses.len()
    }

    /// Get all clauses.
    pub fn clauses(&self) -> &[Clause] {
        &self.clauses
    }

    /// Check if the formula is empty (trivially satisfiable).
    pub fn is_empty(&self) -> bool {
        self.clauses.is_empty()
    }

    /// Check if the formula contains the empty clause (trivially unsatisfiable).
    pub fn has_empty_clause(&self) -> bool {
        self.clauses.iter().any(|c| c.is_empty())
    }
}

impl fmt::Display for CnfFormula {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.clauses.is_empty() {
            write!(f, "⊤")
        } else {
            let parts: Vec<String> = self.clauses.iter().map(|c| c.to_string()).collect();
            write!(f, "{}", parts.join(" ∧ "))
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Assignment — variable assignment
// ═══════════════════════════════════════════════════════════════════════

/// A (partial) assignment of variables to truth values.
#[derive(Debug, Clone)]
pub struct Assignment {
    /// Variable assignments (None = unassigned).
    values: Vec<Option<bool>>,
}

impl Assignment {
    /// Create an empty assignment for n variables.
    pub fn new(num_vars: usize) -> Self {
        Assignment { values: vec![None; num_vars] }
    }

    /// Assign a value to a variable.
    pub fn assign(&mut self, var: usize, val: bool) {
        if var >= self.values.len() {
            self.values.resize(var + 1, None);
        }
        self.values[var] = Some(val);
    }

    /// Unassign a variable.
    pub fn unassign(&mut self, var: usize) {
        if var < self.values.len() {
            self.values[var] = None;
        }
    }

    /// Get the value of a variable.
    pub fn get(&self, var: usize) -> Option<bool> {
        self.values.get(var).copied().flatten()
    }

    /// Evaluate a literal under this assignment.
    pub fn evaluate_literal(&self, lit: &Literal) -> Option<bool> {
        self.get(lit.var()).map(|v| if lit.is_positive() { v } else { !v })
    }

    /// Evaluate a clause under this assignment.
    pub fn evaluate_clause(&self, clause: &Clause) -> Option<bool> {
        let mut has_unassigned = false;
        for lit in &clause.literals {
            match self.evaluate_literal(lit) {
                Some(true) => return Some(true),
                None => has_unassigned = true,
                Some(false) => {}
            }
        }
        if has_unassigned { None } else { Some(false) }
    }

    /// Evaluate a formula under this assignment.
    pub fn evaluate_formula(&self, formula: &CnfFormula) -> Option<bool> {
        let mut all_true = true;
        for clause in &formula.clauses {
            match self.evaluate_clause(clause) {
                Some(false) => return Some(false),
                None => all_true = false,
                Some(true) => {}
            }
        }
        if all_true { Some(true) } else { None }
    }

    /// Get all assigned variables.
    pub fn assigned_vars(&self) -> Vec<(usize, bool)> {
        self.values.iter().enumerate()
            .filter_map(|(i, v)| v.map(|val| (i, val)))
            .collect()
    }

    /// Number of assigned variables.
    pub fn num_assigned(&self) -> usize {
        self.values.iter().filter(|v| v.is_some()).count()
    }

    /// Find the first unassigned variable.
    pub fn first_unassigned(&self) -> Option<usize> {
        self.values.iter().position(|v| v.is_none())
    }
}

// ═══════════════════════════════════════════════════════════════════════
// TseitinEncoder — Tseitin transformation
// ═══════════════════════════════════════════════════════════════════════

/// Tseitin transformation for encoding boolean circuits as CNF.
#[derive(Debug)]
pub struct TseitinEncoder {
    /// The formula being built.
    pub formula: CnfFormula,
}

impl TseitinEncoder {
    /// Create a new encoder.
    pub fn new() -> Self {
        TseitinEncoder { formula: CnfFormula::new() }
    }

    /// Allocate a fresh variable.
    pub fn new_var(&mut self) -> usize {
        self.formula.new_var()
    }

    /// Create a literal for a fresh variable.
    pub fn fresh_literal(&mut self) -> Literal {
        Literal::positive(self.new_var())
    }

    /// Encode: result ↔ (a ∧ b). Returns the result literal.
    pub fn encode_and(&mut self, a: Literal, b: Literal) -> Literal {
        let r = self.fresh_literal();
        // r → a: ¬r ∨ a
        self.formula.add_clause(Clause::from_literals(vec![r.negate(), a]));
        // r → b: ¬r ∨ b
        self.formula.add_clause(Clause::from_literals(vec![r.negate(), b]));
        // a ∧ b → r: ¬a ∨ ¬b ∨ r
        self.formula.add_clause(Clause::from_literals(vec![a.negate(), b.negate(), r]));
        r
    }

    /// Encode: result ↔ (a ∨ b). Returns the result literal.
    pub fn encode_or(&mut self, a: Literal, b: Literal) -> Literal {
        let r = self.fresh_literal();
        // r → a ∨ b: ¬r ∨ a ∨ b
        self.formula.add_clause(Clause::from_literals(vec![r.negate(), a, b]));
        // a → r: ¬a ∨ r
        self.formula.add_clause(Clause::from_literals(vec![a.negate(), r]));
        // b → r: ¬b ∨ r
        self.formula.add_clause(Clause::from_literals(vec![b.negate(), r]));
        r
    }

    /// Encode: result ↔ ¬a. Returns the result literal.
    pub fn encode_not(&mut self, a: Literal) -> Literal {
        let r = self.fresh_literal();
        // r → ¬a: ¬r ∨ ¬a
        self.formula.add_clause(Clause::from_literals(vec![r.negate(), a.negate()]));
        // ¬a → r: a ∨ r
        self.formula.add_clause(Clause::from_literals(vec![a, r]));
        r
    }

    /// Encode: result ↔ (a → b). Returns the result literal.
    pub fn encode_implies(&mut self, a: Literal, b: Literal) -> Literal {
        // a → b ≡ ¬a ∨ b
        self.encode_or(a.negate(), b)
    }

    /// Encode: result ↔ (a ↔ b). Returns the result literal.
    pub fn encode_iff(&mut self, a: Literal, b: Literal) -> Literal {
        let r = self.fresh_literal();
        // r → (a → b): ¬r ∨ ¬a ∨ b
        self.formula.add_clause(Clause::from_literals(vec![r.negate(), a.negate(), b]));
        // r → (b → a): ¬r ∨ a ∨ ¬b
        self.formula.add_clause(Clause::from_literals(vec![r.negate(), a, b.negate()]));
        // (a ↔ b) → r: a ∨ b ∨ r
        self.formula.add_clause(Clause::from_literals(vec![a, b, r]));
        // (a ↔ b) → r: ¬a ∨ ¬b ∨ r
        self.formula.add_clause(Clause::from_literals(vec![a.negate(), b.negate(), r]));
        r
    }

    /// Encode: result ↔ (a ⊕ b) (XOR). Returns the result literal.
    pub fn encode_xor(&mut self, a: Literal, b: Literal) -> Literal {
        let iff = self.encode_iff(a, b);
        self.encode_not(iff)
    }

    /// Encode: at most one of the given literals is true (pairwise encoding).
    pub fn encode_at_most_one(&mut self, lits: &[Literal]) {
        for i in 0..lits.len() {
            for j in (i+1)..lits.len() {
                // ¬l_i ∨ ¬l_j
                self.formula.add_clause(Clause::from_literals(
                    vec![lits[i].negate(), lits[j].negate()]
                ));
            }
        }
    }

    /// Encode: at least one of the given literals is true.
    pub fn encode_at_least_one(&mut self, lits: &[Literal]) {
        self.formula.add_clause(Clause::from_literals(lits.to_vec()));
    }

    /// Encode: exactly one of the given literals is true.
    pub fn encode_exactly_one(&mut self, lits: &[Literal]) {
        self.encode_at_least_one(lits);
        self.encode_at_most_one(lits);
    }

    /// Get the constructed formula.
    pub fn into_formula(self) -> CnfFormula {
        self.formula
    }
}

// ═══════════════════════════════════════════════════════════════════════
// DpllSolver — basic DPLL SAT solver
// ═══════════════════════════════════════════════════════════════════════

/// A basic DPLL SAT solver.
#[derive(Debug)]
pub struct DpllSolver {
    /// Statistics.
    pub decisions: usize,
    pub propagations: usize,
}

impl DpllSolver {
    /// Create a new DPLL solver.
    pub fn new() -> Self {
        DpllSolver { decisions: 0, propagations: 0 }
    }

    /// Solve the formula. Returns an assignment if satisfiable.
    pub fn solve(&mut self, formula: &CnfFormula) -> Option<Assignment> {
        self.decisions = 0;
        self.propagations = 0;
        let mut assignment = Assignment::new(formula.num_vars());
        let simplified = self.simplify_formula(formula, &assignment);
        if self.dpll(&simplified, &mut assignment) {
            Some(assignment)
        } else {
            None
        }
    }

    fn dpll(&mut self, formula: &CnfFormula, assignment: &mut Assignment) -> bool {
        // Unit propagation
        let formula = self.unit_propagate(formula, assignment);

        // Check for empty clause (conflict)
        if formula.has_empty_clause() {
            return false;
        }

        // Check if all clauses satisfied
        if formula.clauses.is_empty() {
            return true;
        }

        // Pure literal elimination
        let formula = self.pure_literal_eliminate(&formula, assignment);
        if formula.clauses.is_empty() {
            return true;
        }

        // Choose a variable to branch on
        let var = match self.choose_variable(&formula, assignment) {
            Some(v) => v,
            None => return formula.clauses.is_empty(),
        };

        self.decisions += 1;

        // Try true
        assignment.assign(var, true);
        let f_true = self.simplify_formula(&formula, assignment);
        if self.dpll(&f_true, assignment) {
            return true;
        }

        // Try false
        assignment.assign(var, false);
        let f_false = self.simplify_formula(&formula, assignment);
        if self.dpll(&f_false, assignment) {
            return true;
        }

        // Backtrack
        assignment.unassign(var);
        false
    }

    fn unit_propagate(&mut self, formula: &CnfFormula, assignment: &mut Assignment) -> CnfFormula {
        let mut formula = formula.clone();
        loop {
            let mut found_unit = false;
            for clause in &formula.clauses {
                if clause.len() == 1 {
                    let lit = clause.literals[0];
                    let val = lit.is_positive();
                    if assignment.get(lit.var()).is_none() {
                        assignment.assign(lit.var(), val);
                        self.propagations += 1;
                        found_unit = true;
                    }
                }
            }
            if !found_unit { break; }
            formula = self.simplify_formula(&formula, assignment);
        }
        formula
    }

    fn pure_literal_eliminate(&self, formula: &CnfFormula, assignment: &mut Assignment) -> CnfFormula {
        let mut pos = HashSet::new();
        let mut neg = HashSet::new();
        for clause in &formula.clauses {
            for lit in &clause.literals {
                if assignment.get(lit.var()).is_none() {
                    if lit.is_positive() {
                        pos.insert(lit.var());
                    } else {
                        neg.insert(lit.var());
                    }
                }
            }
        }

        // Pure positive: appears only positive
        for &v in &pos {
            if !neg.contains(&v) {
                assignment.assign(v, true);
            }
        }
        // Pure negative: appears only negative
        for &v in &neg {
            if !pos.contains(&v) {
                assignment.assign(v, false);
            }
        }

        self.simplify_formula(formula, assignment)
    }

    fn simplify_formula(&self, formula: &CnfFormula, assignment: &Assignment) -> CnfFormula {
        let mut new_clauses = Vec::new();
        for clause in &formula.clauses {
            match self.simplify_clause(clause, assignment) {
                ClauseStatus::Satisfied => {} // Remove satisfied clauses
                ClauseStatus::Unsatisfied => {
                    new_clauses.push(Clause::new()); // Empty clause
                }
                ClauseStatus::Simplified(c) => {
                    new_clauses.push(c);
                }
            }
        }
        let mut f = CnfFormula::with_vars(formula.num_vars());
        for c in new_clauses {
            f.add_clause(c);
        }
        f
    }

    fn simplify_clause(&self, clause: &Clause, assignment: &Assignment) -> ClauseStatus {
        let mut remaining = Vec::new();
        for &lit in &clause.literals {
            match assignment.evaluate_literal(&lit) {
                Some(true) => return ClauseStatus::Satisfied,
                Some(false) => {} // Remove false literals
                None => remaining.push(lit),
            }
        }
        if remaining.is_empty() {
            ClauseStatus::Unsatisfied
        } else {
            ClauseStatus::Simplified(Clause::from_literals(remaining))
        }
    }

    fn choose_variable(&self, formula: &CnfFormula, assignment: &Assignment) -> Option<usize> {
        // Choose the variable that appears most frequently
        let mut counts: HashMap<usize, usize> = HashMap::new();
        for clause in &formula.clauses {
            for lit in &clause.literals {
                if assignment.get(lit.var()).is_none() {
                    *counts.entry(lit.var()).or_insert(0) += 1;
                }
            }
        }
        counts.into_iter().max_by_key(|&(_, c)| c).map(|(v, _)| v)
    }
}

enum ClauseStatus {
    Satisfied,
    Unsatisfied,
    Simplified(Clause),
}

// ═══════════════════════════════════════════════════════════════════════
// CdclSolver — CDCL solver with clause learning
// ═══════════════════════════════════════════════════════════════════════

/// Decision level and reason for an assignment.
#[derive(Debug, Clone)]
struct AssignmentInfo {
    value: bool,
    decision_level: usize,
    reason: Option<usize>, // clause index that forced this (None = decision)
}

/// CDCL SAT solver with clause learning and non-chronological backtracking.
#[derive(Debug)]
pub struct CdclSolver {
    /// Number of variables.
    num_vars: usize,
    /// Clauses (original + learned).
    clauses: Vec<Clause>,
    /// Assignment trail.
    trail: Vec<Literal>,
    /// Assignment info per variable.
    info: Vec<Option<AssignmentInfo>>,
    /// Current decision level.
    decision_level: usize,
    /// Trail indices at each decision level.
    trail_lim: Vec<usize>,
    /// VSIDS activity scores.
    activity: Vec<f64>,
    /// Activity decay factor.
    var_decay: f64,
    /// Activity increment.
    var_inc: f64,
    /// Number of conflicts.
    pub num_conflicts: usize,
    /// Number of decisions.
    pub num_decisions: usize,
    /// Number of learned clauses.
    pub num_learned: usize,
    /// Restart interval.
    restart_interval: usize,
    /// Conflicts until next restart.
    conflicts_until_restart: usize,
}

impl CdclSolver {
    /// Create a new CDCL solver.
    pub fn new() -> Self {
        CdclSolver {
            num_vars: 0,
            clauses: Vec::new(),
            trail: Vec::new(),
            info: Vec::new(),
            decision_level: 0,
            trail_lim: Vec::new(),
            activity: Vec::new(),
            var_decay: 0.95,
            var_inc: 1.0,
            num_conflicts: 0,
            num_decisions: 0,
            num_learned: 0,
            restart_interval: 100,
            conflicts_until_restart: 100,
        }
    }

    /// Solve a CNF formula.
    pub fn solve(&mut self, formula: &CnfFormula) -> Option<Assignment> {
        self.num_vars = formula.num_vars();
        self.clauses = formula.clauses.clone();
        self.trail = Vec::new();
        self.info = vec![None; self.num_vars];
        self.decision_level = 0;
        self.trail_lim = Vec::new();
        self.activity = vec![0.0; self.num_vars];
        self.num_conflicts = 0;
        self.num_decisions = 0;
        self.num_learned = 0;
        self.conflicts_until_restart = self.restart_interval;

        // Initial unit propagation
        if let Some(_conflict) = self.propagate() {
            return None; // UNSAT at level 0
        }

        loop {
            // Check if all variables assigned
            if self.trail.len() == self.num_vars {
                // Build assignment
                let mut assignment = Assignment::new(self.num_vars);
                for &lit in &self.trail {
                    assignment.assign(lit.var(), lit.is_positive());
                }
                return Some(assignment);
            }

            // Restart check
            if self.conflicts_until_restart == 0 {
                self.backtrack_to(0);
                self.conflicts_until_restart = self.restart_interval;
                self.restart_interval = (self.restart_interval as f64 * 1.5) as usize;
            }

            // Make a decision
            let var = match self.pick_branch_variable() {
                Some(v) => v,
                None => {
                    // All assigned, check satisfaction
                    let mut assignment = Assignment::new(self.num_vars);
                    for &lit in &self.trail {
                        assignment.assign(lit.var(), lit.is_positive());
                    }
                    return Some(assignment);
                }
            };

            self.num_decisions += 1;
            self.decision_level += 1;
            self.trail_lim.push(self.trail.len());
            self.assign(var, true, None);

            // Propagate
            loop {
                match self.propagate() {
                    None => break, // No conflict
                    Some(conflict_clause) => {
                        self.num_conflicts += 1;
                        self.conflicts_until_restart = self.conflicts_until_restart.saturating_sub(1);

                        if self.decision_level == 0 {
                            return None; // UNSAT
                        }

                        // Analyze conflict
                        let (learned, backtrack_level) = self.analyze_conflict(conflict_clause);
                        self.num_learned += 1;

                        // Backtrack
                        self.backtrack_to(backtrack_level);

                        // Add learned clause
                        let clause_idx = self.clauses.len();
                        self.clauses.push(learned.clone());

                        // If unit, propagate the asserting literal
                        if !learned.literals.is_empty() {
                            let asserting = learned.literals[0];
                            self.assign(asserting.var(), asserting.is_positive(), Some(clause_idx));
                        }
                    }
                }
            }
        }
    }

    fn assign(&mut self, var: usize, value: bool, reason: Option<usize>) {
        self.info[var] = Some(AssignmentInfo {
            value,
            decision_level: self.decision_level,
            reason,
        });
        self.trail.push(Literal::new(var, value));
    }

    fn value(&self, var: usize) -> Option<bool> {
        self.info[var].as_ref().map(|i| i.value)
    }

    fn lit_value(&self, lit: &Literal) -> Option<bool> {
        self.value(lit.var()).map(|v| if lit.is_positive() { v } else { !v })
    }

    fn propagate(&mut self) -> Option<usize> {
        let mut i = 0;
        while i < self.trail.len() {
            let _propagated = self.trail[i];
            i += 1;

            // Check all clauses for unit propagation
            for ci in 0..self.clauses.len() {
                let clause = &self.clauses[ci];
                let mut unassigned_lit = None;
                let mut num_unassigned = 0;
                let mut satisfied = false;

                for &lit in &clause.literals {
                    match self.lit_value(&lit) {
                        Some(true) => { satisfied = true; break; }
                        Some(false) => {}
                        None => {
                            num_unassigned += 1;
                            unassigned_lit = Some(lit);
                        }
                    }
                }

                if satisfied { continue; }

                if num_unassigned == 0 {
                    return Some(ci); // Conflict
                }

                if num_unassigned == 1 {
                    if let Some(lit) = unassigned_lit {
                        if self.value(lit.var()).is_none() {
                            self.assign(lit.var(), lit.is_positive(), Some(ci));
                        }
                    }
                }
            }
        }
        None
    }

    fn analyze_conflict(&mut self, conflict_clause: usize) -> (Clause, usize) {
        let mut learned = Vec::new();
        let mut seen = vec![false; self.num_vars];
        let mut counter = 0;
        let mut backtrack_level = 0;

        // Start with the conflict clause
        let conflict_lits: Vec<_> = self.clauses[conflict_clause].literals.clone();
        for &lit in &conflict_lits {
            let var = lit.var();
            if !seen[var] {
                seen[var] = true;
                if let Some(info) = &self.info[var] {
                    if info.decision_level == self.decision_level {
                        counter += 1;
                    } else if info.decision_level > 0 {
                        learned.push(lit.negate());
                        backtrack_level = backtrack_level.max(info.decision_level);
                    }
                }
            }
            // Bump activity
            self.bump_activity(var);
        }

        // Resolve along the trail
        for i in (0..self.trail.len()).rev() {
            if counter <= 1 { break; }

            let lit = self.trail[i];
            let var = lit.var();
            if !seen[var] { continue; }

            if let Some(info) = &self.info[var].clone() {
                if info.decision_level == self.decision_level {
                    if let Some(reason_idx) = info.reason {
                        counter -= 1;
                        let reason = self.clauses[reason_idx].clone();
                        for &rlit in &reason.literals {
                            let rvar = rlit.var();
                            if !seen[rvar] {
                                seen[rvar] = true;
                                if let Some(rinfo) = &self.info[rvar] {
                                    if rinfo.decision_level == self.decision_level {
                                        counter += 1;
                                    } else if rinfo.decision_level > 0 {
                                        learned.push(rlit.negate());
                                        backtrack_level = backtrack_level.max(rinfo.decision_level);
                                    }
                                }
                            }
                            self.bump_activity(rvar);
                        }
                    } else {
                        // Decision variable - this is the UIP
                        learned.push(lit.negate());
                        counter -= 1;
                    }
                }
            }
        }

        // Find the asserting literal (should be first)
        // The asserting literal is the one at the current decision level
        let asserting_idx = learned.iter().position(|lit| {
            self.info[lit.var()].as_ref()
                .map_or(false, |i| i.decision_level == self.decision_level)
        });

        if let Some(idx) = asserting_idx {
            learned.swap(0, idx);
        }

        if backtrack_level == 0 && learned.len() > 1 {
            backtrack_level = self.decision_level.saturating_sub(1);
        }

        self.decay_activities();

        (Clause::from_literals(learned), backtrack_level)
    }

    fn backtrack_to(&mut self, level: usize) {
        if self.decision_level <= level { return; }

        while self.trail_lim.len() > level {
            let lim = self.trail_lim.pop().unwrap();
            while self.trail.len() > lim {
                let lit = self.trail.pop().unwrap();
                self.info[lit.var()] = None;
            }
        }
        self.decision_level = level;
    }

    fn pick_branch_variable(&self) -> Option<usize> {
        // VSIDS: pick unassigned variable with highest activity
        let mut best = None;
        let mut best_activity = -1.0f64;
        for v in 0..self.num_vars {
            if self.value(v).is_none() {
                if self.activity[v] > best_activity {
                    best_activity = self.activity[v];
                    best = Some(v);
                }
            }
        }
        best
    }

    fn bump_activity(&mut self, var: usize) {
        if var < self.activity.len() {
            self.activity[var] += self.var_inc;
        }
    }

    fn decay_activities(&mut self) {
        self.var_inc /= self.var_decay;
        // Rescale if too large
        if self.var_inc > 1e100 {
            for a in &mut self.activity {
                *a *= 1e-100;
            }
            self.var_inc *= 1e-100;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// MemoryModelEncoder — encode memory model constraints as SAT
// ═══════════════════════════════════════════════════════════════════════

/// Encode memory model checking problems as SAT instances.
#[derive(Debug)]
pub struct MemoryModelEncoder {
    /// Number of events.
    num_events: usize,
    /// The formula being constructed.
    pub formula: CnfFormula,
    /// Variable map: (relation, i, j) -> variable index.
    var_map: HashMap<(String, usize, usize), usize>,
}

impl MemoryModelEncoder {
    /// Create a new encoder for an execution with n events.
    pub fn new(num_events: usize) -> Self {
        MemoryModelEncoder {
            num_events,
            formula: CnfFormula::new(),
            var_map: HashMap::new(),
        }
    }

    /// Get or create a variable for a relation edge.
    pub fn edge_var(&mut self, relation: &str, i: usize, j: usize) -> usize {
        let key = (relation.to_string(), i, j);
        if let Some(&var) = self.var_map.get(&key) {
            var
        } else {
            let var = self.formula.new_var();
            self.var_map.insert(key, var);
            var
        }
    }

    /// Encode that a relation is acyclic (no path from any node back to itself).
    /// Uses bounded path encoding: for each pair (i,j), path_k(i,j) means
    /// there's a path of length ≤ k from i to j.
    pub fn encode_acyclicity(&mut self, relation: &str) {
        let n = self.num_events;

        // For acyclicity, encode ¬(i →+ i) for all i.
        // path_1(i,j) = edge(i,j)
        // path_k(i,j) = path_{k-1}(i,j) ∨ ∃m: path_{k-1}(i,m) ∧ edge(m,j)
        // ¬path_n(i,i)

        // Simple encoding: for each pair, the edge variable
        // For small n, direct transitivity encoding
        for i in 0..n {
            // No self-loops
            let var = self.edge_var(relation, i, i);
            self.formula.add_clause(Clause::unit(Literal::negative(var)));
        }

        // Floyd-Warshall style: if i->k and k->j, then path(i,j)
        // And ¬path(i,i)
        // We use auxiliary variables for transitive paths
        for k in 0..n {
            for i in 0..n {
                if i == k { continue; }
                for j in 0..n {
                    if j == k || j == i { continue; }
                    let ik = self.edge_var(relation, i, k);
                    let kj = self.edge_var(relation, k, j);
                    let ij = self.edge_var(relation, i, j);
                    // If i->k and k->j then i->j (transitivity)
                    // ¬ik ∨ ¬kj ∨ ij
                    self.formula.add_clause(Clause::from_literals(vec![
                        Literal::negative(ik),
                        Literal::negative(kj),
                        Literal::positive(ij),
                    ]));
                }
            }
        }
    }

    /// Encode that a relation is total on a set of events.
    /// For every pair (i,j) with i≠j, either edge(i,j) or edge(j,i).
    pub fn encode_totality(&mut self, relation: &str, events: &[usize]) {
        for (idx_a, &i) in events.iter().enumerate() {
            for &j in &events[idx_a+1..] {
                let ij = self.edge_var(relation, i, j);
                let ji = self.edge_var(relation, j, i);
                // At least one: ij ∨ ji
                self.formula.add_clause(Clause::from_literals(vec![
                    Literal::positive(ij),
                    Literal::positive(ji),
                ]));
                // At most one: ¬ij ∨ ¬ji
                self.formula.add_clause(Clause::from_literals(vec![
                    Literal::negative(ij),
                    Literal::negative(ji),
                ]));
            }
        }
    }

    /// Encode that reads-from is functional: each read has exactly one write.
    pub fn encode_functional_rf(&mut self, reads: &[usize], writes: &[usize]) {
        for &r in reads {
            let mut write_lits = Vec::new();
            for &w in writes {
                let var = self.edge_var("rf", w, r);
                write_lits.push(Literal::positive(var));
            }
            // Exactly one write feeds each read
            // At least one
            self.formula.add_clause(Clause::from_literals(write_lits.clone()));
            // At most one (pairwise)
            for i in 0..write_lits.len() {
                for j in (i+1)..write_lits.len() {
                    self.formula.add_clause(Clause::from_literals(vec![
                        write_lits[i].negate(),
                        write_lits[j].negate(),
                    ]));
                }
            }
        }
    }

    /// Get the constructed formula.
    pub fn into_formula(self) -> CnfFormula {
        self.formula
    }

    /// Get the number of variables used.
    pub fn num_vars(&self) -> usize {
        self.formula.num_vars()
    }

    /// Get the number of clauses.
    pub fn num_clauses(&self) -> usize {
        self.formula.num_clauses()
    }
}

// ═══════════════════════════════════════════════════════════════════════
// IncrementalSolver — incremental SAT via assumptions
// ═══════════════════════════════════════════════════════════════════════

/// Incremental SAT solver that supports adding clauses and solving under assumptions.
#[derive(Debug)]
pub struct IncrementalSolver {
    /// The base formula.
    formula: CnfFormula,
    /// The underlying solver.
    solver: CdclSolver,
}

impl IncrementalSolver {
    /// Create a new incremental solver.
    pub fn new() -> Self {
        IncrementalSolver {
            formula: CnfFormula::new(),
            solver: CdclSolver::new(),
        }
    }

    /// Add a clause to the solver.
    pub fn add_clause(&mut self, clause: Clause) {
        self.formula.add_clause(clause);
    }

    /// Allocate a new variable.
    pub fn new_var(&mut self) -> usize {
        self.formula.new_var()
    }

    /// Solve the formula.
    pub fn solve(&mut self) -> Option<Assignment> {
        self.solver = CdclSolver::new();
        self.solver.solve(&self.formula)
    }

    /// Solve under assumptions (add temporary unit clauses).
    pub fn solve_under_assumptions(&mut self, assumptions: &[Literal]) -> Option<Assignment> {
        let mut formula = self.formula.clone();
        for &lit in assumptions {
            formula.add_clause(Clause::unit(lit));
        }
        let mut solver = CdclSolver::new();
        solver.solve(&formula)
    }

    /// Get statistics.
    pub fn statistics(&self) -> (usize, usize, usize) {
        (self.solver.num_conflicts, self.solver.num_decisions, self.solver.num_learned)
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── Literal tests ────────────────────────────────────────────────

    #[test]
    fn test_literal_creation() {
        let p = Literal::positive(0);
        assert!(p.is_positive());
        assert_eq!(p.var(), 0);

        let n = Literal::negative(0);
        assert!(n.is_negative());
        assert_eq!(n.var(), 0);
    }

    #[test]
    fn test_literal_negate() {
        let p = Literal::positive(3);
        let n = p.negate();
        assert!(n.is_negative());
        assert_eq!(n.var(), 3);
        assert_eq!(n.negate(), p);
    }

    // ── Clause tests ─────────────────────────────────────────────────

    #[test]
    fn test_clause_empty() {
        let c = Clause::new();
        assert!(c.is_empty());
        assert!(!c.is_unit());
    }

    #[test]
    fn test_clause_unit() {
        let c = Clause::unit(Literal::positive(0));
        assert!(c.is_unit());
        assert!(!c.is_empty());
    }

    #[test]
    fn test_clause_tautology() {
        let c = Clause::from_literals(vec![
            Literal::positive(0),
            Literal::negative(0),
        ]);
        assert!(c.is_tautology());
    }

    // ── Formula tests ────────────────────────────────────────────────

    #[test]
    fn test_formula_creation() {
        let mut f = CnfFormula::new();
        f.add_clause(Clause::from_literals(vec![
            Literal::positive(0),
            Literal::positive(1),
        ]));
        assert_eq!(f.num_vars(), 2);
        assert_eq!(f.num_clauses(), 1);
    }

    // ── Tseitin tests ────────────────────────────────────────────────

    #[test]
    fn test_tseitin_and() {
        let mut enc = TseitinEncoder::new();
        let a = Literal::positive(enc.new_var());
        let b = Literal::positive(enc.new_var());
        let r = enc.encode_and(a, b);

        // Verify: if a=T, b=T, then r should be forced T
        let formula = enc.into_formula();
        let mut solver = DpllSolver::new();

        let mut f = formula.clone();
        f.add_clause(Clause::unit(a));
        f.add_clause(Clause::unit(b));
        let result = solver.solve(&f);
        assert!(result.is_some());
        let assignment = result.unwrap();
        assert_eq!(assignment.get(r.var()), Some(true));
    }

    #[test]
    fn test_tseitin_or() {
        let mut enc = TseitinEncoder::new();
        let a = Literal::positive(enc.new_var());
        let b = Literal::positive(enc.new_var());
        let r = enc.encode_or(a, b);

        let formula = enc.into_formula();
        let mut solver = DpllSolver::new();

        // If a=F, b=F, then r should be F
        let mut f = formula.clone();
        f.add_clause(Clause::unit(a.negate()));
        f.add_clause(Clause::unit(b.negate()));
        let result = solver.solve(&f);
        assert!(result.is_some());
        let assignment = result.unwrap();
        assert_eq!(assignment.get(r.var()), Some(false));
    }

    #[test]
    fn test_tseitin_exactly_one() {
        let mut enc = TseitinEncoder::new();
        let lits: Vec<_> = (0..3).map(|_| Literal::positive(enc.new_var())).collect();
        enc.encode_exactly_one(&lits);

        let formula = enc.into_formula();
        let mut solver = DpllSolver::new();
        let result = solver.solve(&formula);
        assert!(result.is_some());
        let assignment = result.unwrap();

        // Exactly one should be true
        let true_count = lits.iter()
            .filter(|l| assignment.get(l.var()) == Some(true))
            .count();
        assert_eq!(true_count, 1);
    }

    // ── DPLL solver tests ────────────────────────────────────────────

    #[test]
    fn test_dpll_simple_sat() {
        // (x0 ∨ x1) ∧ (¬x0 ∨ x1)
        let mut f = CnfFormula::new();
        f.add_clause(Clause::from_literals(vec![
            Literal::positive(0),
            Literal::positive(1),
        ]));
        f.add_clause(Clause::from_literals(vec![
            Literal::negative(0),
            Literal::positive(1),
        ]));

        let mut solver = DpllSolver::new();
        let result = solver.solve(&f);
        assert!(result.is_some());
        let assignment = result.unwrap();
        assert_eq!(assignment.get(1), Some(true));
    }

    #[test]
    fn test_dpll_simple_unsat() {
        // (x0) ∧ (¬x0)
        let mut f = CnfFormula::new();
        f.add_clause(Clause::unit(Literal::positive(0)));
        f.add_clause(Clause::unit(Literal::negative(0)));

        let mut solver = DpllSolver::new();
        let result = solver.solve(&f);
        assert!(result.is_none());
    }

    #[test]
    fn test_dpll_three_coloring() {
        // 3-coloring of a triangle: sat
        let mut f = CnfFormula::new();

        // Variables: x_{v,c} for vertex v∈{0,1,2}, color c∈{0,1,2}
        // x_{v,c} is variable v*3 + c
        for v in 0..3 {
            // At least one color
            f.add_clause(Clause::from_literals(vec![
                Literal::positive(v * 3),
                Literal::positive(v * 3 + 1),
                Literal::positive(v * 3 + 2),
            ]));
            // At most one color
            for c1 in 0..3 {
                for c2 in (c1+1)..3 {
                    f.add_clause(Clause::from_literals(vec![
                        Literal::negative(v * 3 + c1),
                        Literal::negative(v * 3 + c2),
                    ]));
                }
            }
        }

        // Adjacent vertices different colors
        let edges = [(0, 1), (1, 2), (0, 2)];
        for &(u, v) in &edges {
            for c in 0..3 {
                f.add_clause(Clause::from_literals(vec![
                    Literal::negative(u * 3 + c),
                    Literal::negative(v * 3 + c),
                ]));
            }
        }

        let mut solver = DpllSolver::new();
        let result = solver.solve(&f);
        assert!(result.is_some());
    }

    // ── CDCL solver tests ────────────────────────────────────────────

    #[test]
    fn test_cdcl_simple_sat() {
        let mut f = CnfFormula::new();
        f.add_clause(Clause::from_literals(vec![
            Literal::positive(0),
            Literal::positive(1),
        ]));
        f.add_clause(Clause::from_literals(vec![
            Literal::negative(0),
            Literal::positive(1),
        ]));

        let mut solver = CdclSolver::new();
        let result = solver.solve(&f);
        assert!(result.is_some());
    }

    #[test]
    fn test_cdcl_simple_unsat() {
        let mut f = CnfFormula::new();
        f.add_clause(Clause::unit(Literal::positive(0)));
        f.add_clause(Clause::unit(Literal::negative(0)));

        let mut solver = CdclSolver::new();
        let result = solver.solve(&f);
        assert!(result.is_none());
    }

    #[test]
    fn test_cdcl_pigeonhole_2_1() {
        // 2 pigeons, 1 hole: UNSAT
        // Variables: x_{p,h} = pigeon p in hole h
        // p0 → hole 0, p1 → hole 0
        let mut f = CnfFormula::new();
        // Each pigeon in at least one hole
        f.add_clause(Clause::unit(Literal::positive(0))); // pigeon 0 in hole 0
        f.add_clause(Clause::unit(Literal::positive(1))); // pigeon 1 in hole 0
        // No two pigeons in same hole
        f.add_clause(Clause::from_literals(vec![
            Literal::negative(0),
            Literal::negative(1),
        ]));

        let mut solver = CdclSolver::new();
        let result = solver.solve(&f);
        assert!(result.is_none());
    }

    // ── Memory model encoder tests ───────────────────────────────────

    #[test]
    fn test_encode_totality() {
        let mut enc = MemoryModelEncoder::new(3);
        enc.encode_totality("co", &[0, 1, 2]);

        // Should be satisfiable (total order exists)
        let formula = enc.into_formula();
        let mut solver = DpllSolver::new();
        let result = solver.solve(&formula);
        assert!(result.is_some());
    }

    #[test]
    fn test_encode_functional_rf() {
        let mut enc = MemoryModelEncoder::new(4);
        enc.encode_functional_rf(&[2, 3], &[0, 1]);

        let formula = enc.into_formula();
        let mut solver = DpllSolver::new();
        let result = solver.solve(&formula);
        assert!(result.is_some());
    }

    // ── Incremental solver tests ─────────────────────────────────────

    #[test]
    fn test_incremental_solve() {
        let mut solver = IncrementalSolver::new();
        let v0 = solver.new_var();
        let v1 = solver.new_var();

        solver.add_clause(Clause::from_literals(vec![
            Literal::positive(v0),
            Literal::positive(v1),
        ]));

        let result = solver.solve();
        assert!(result.is_some());

        // Add contradictory clauses
        solver.add_clause(Clause::unit(Literal::negative(v0)));
        solver.add_clause(Clause::unit(Literal::negative(v1)));

        let result = solver.solve();
        assert!(result.is_none());
    }

    #[test]
    fn test_incremental_assumptions() {
        let mut solver = IncrementalSolver::new();
        let v0 = solver.new_var();
        let v1 = solver.new_var();

        solver.add_clause(Clause::from_literals(vec![
            Literal::positive(v0),
            Literal::positive(v1),
        ]));

        // Under assumption ¬v0, v1 must be true
        let result = solver.solve_under_assumptions(&[Literal::negative(v0)]);
        assert!(result.is_some());
        let assignment = result.unwrap();
        assert_eq!(assignment.get(v1), Some(true));
    }

    // ── Assignment tests ─────────────────────────────────────────────

    #[test]
    fn test_assignment_evaluate() {
        let mut a = Assignment::new(3);
        a.assign(0, true);
        a.assign(1, false);

        let clause = Clause::from_literals(vec![
            Literal::positive(0),
            Literal::negative(1),
        ]);
        assert_eq!(a.evaluate_clause(&clause), Some(true));

        let clause2 = Clause::from_literals(vec![
            Literal::negative(0),
            Literal::positive(1),
        ]);
        assert_eq!(a.evaluate_clause(&clause2), Some(false));
    }
}
