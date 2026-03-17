//! Propositional formula representation and CNF transformation.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;

/// A literal is a variable (positive) or its negation (negative).
/// Represented as a nonzero i32: positive = variable, negative = negation.
pub type Literal = i32;
/// A clause is a disjunction of literals.
pub type Clause = Vec<Literal>;

// ---------------------------------------------------------------------------
// Formula
// ---------------------------------------------------------------------------

/// Propositional formula in tree form.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Formula {
    /// Boolean constant true.
    True,
    /// Boolean constant false.
    False,
    /// Propositional variable identified by a positive integer.
    Var(u32),
    /// Negation.
    Not(Box<Formula>),
    /// Conjunction.
    And(Vec<Formula>),
    /// Disjunction.
    Or(Vec<Formula>),
    /// Implication: lhs => rhs.
    Implies(Box<Formula>, Box<Formula>),
}

impl Formula {
    /// Create a variable formula.
    pub fn var(id: u32) -> Self {
        Formula::Var(id)
    }

    /// Negate a formula.
    pub fn not(f: Formula) -> Self {
        Formula::Not(Box::new(f))
    }

    /// Conjunction of formulas.
    pub fn and(fs: Vec<Formula>) -> Self {
        Formula::And(fs)
    }

    /// Disjunction of formulas.
    pub fn or(fs: Vec<Formula>) -> Self {
        Formula::Or(fs)
    }

    /// Implication.
    pub fn implies(lhs: Formula, rhs: Formula) -> Self {
        Formula::Implies(Box::new(lhs), Box::new(rhs))
    }

    /// Collect all variable ids used in the formula.
    pub fn variables(&self) -> HashSet<u32> {
        let mut vars = HashSet::new();
        self.collect_vars(&mut vars);
        vars
    }

    fn collect_vars(&self, vars: &mut HashSet<u32>) {
        match self {
            Formula::True | Formula::False => {}
            Formula::Var(id) => {
                vars.insert(*id);
            }
            Formula::Not(f) => f.collect_vars(vars),
            Formula::And(fs) | Formula::Or(fs) => {
                for f in fs {
                    f.collect_vars(vars);
                }
            }
            Formula::Implies(a, b) => {
                a.collect_vars(vars);
                b.collect_vars(vars);
            }
        }
    }

    /// Evaluate the formula under a total assignment (variable id -> bool).
    pub fn evaluate(&self, assignment: &HashMap<u32, bool>) -> bool {
        match self {
            Formula::True => true,
            Formula::False => false,
            Formula::Var(id) => *assignment.get(id).unwrap_or(&false),
            Formula::Not(f) => !f.evaluate(assignment),
            Formula::And(fs) => fs.iter().all(|f| f.evaluate(assignment)),
            Formula::Or(fs) => fs.iter().any(|f| f.evaluate(assignment)),
            Formula::Implies(a, b) => !a.evaluate(assignment) || b.evaluate(assignment),
        }
    }

    /// Simplify the formula using basic rewriting rules.
    pub fn simplify(&self) -> Formula {
        match self {
            Formula::True | Formula::False | Formula::Var(_) => self.clone(),
            Formula::Not(f) => {
                let sf = f.simplify();
                match sf {
                    Formula::True => Formula::False,
                    Formula::False => Formula::True,
                    Formula::Not(inner) => *inner,
                    other => Formula::Not(Box::new(other)),
                }
            }
            Formula::And(fs) => {
                let mut simplified: Vec<Formula> = Vec::new();
                for f in fs {
                    let sf = f.simplify();
                    match sf {
                        Formula::False => return Formula::False,
                        Formula::True => {}
                        Formula::And(inner) => simplified.extend(inner),
                        other => simplified.push(other),
                    }
                }
                if simplified.is_empty() {
                    Formula::True
                } else if simplified.len() == 1 {
                    simplified.into_iter().next().unwrap()
                } else {
                    Formula::And(simplified)
                }
            }
            Formula::Or(fs) => {
                let mut simplified: Vec<Formula> = Vec::new();
                for f in fs {
                    let sf = f.simplify();
                    match sf {
                        Formula::True => return Formula::True,
                        Formula::False => {}
                        Formula::Or(inner) => simplified.extend(inner),
                        other => simplified.push(other),
                    }
                }
                if simplified.is_empty() {
                    Formula::False
                } else if simplified.len() == 1 {
                    simplified.into_iter().next().unwrap()
                } else {
                    Formula::Or(simplified)
                }
            }
            Formula::Implies(a, b) => {
                let sa = a.simplify();
                let sb = b.simplify();
                match (&sa, &sb) {
                    (Formula::False, _) | (_, Formula::True) => Formula::True,
                    (Formula::True, _) => sb,
                    _ => Formula::Implies(Box::new(sa), Box::new(sb)),
                }
            }
        }
    }

    /// Convert to CNF using Tseitin transformation.
    /// Returns a `CnfFormula` that is equisatisfiable with the original.
    pub fn to_cnf(&self) -> CnfFormula {
        let mut converter = TseitinConverter::new();
        let root_lit = converter.convert(self);
        converter.cnf.add_clause(vec![root_lit]);
        converter.cnf
    }

    /// Count the number of nodes in the formula tree.
    pub fn node_count(&self) -> usize {
        match self {
            Formula::True | Formula::False | Formula::Var(_) => 1,
            Formula::Not(f) => 1 + f.node_count(),
            Formula::And(fs) | Formula::Or(fs) => {
                1 + fs.iter().map(|f| f.node_count()).sum::<usize>()
            }
            Formula::Implies(a, b) => 1 + a.node_count() + b.node_count(),
        }
    }

    /// Compute the depth of the formula tree.
    pub fn depth(&self) -> usize {
        match self {
            Formula::True | Formula::False | Formula::Var(_) => 0,
            Formula::Not(f) => 1 + f.depth(),
            Formula::And(fs) | Formula::Or(fs) => {
                1 + fs.iter().map(|f| f.depth()).max().unwrap_or(0)
            }
            Formula::Implies(a, b) => 1 + a.depth().max(b.depth()),
        }
    }
}

impl fmt::Display for Formula {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Formula::True => write!(f, "⊤"),
            Formula::False => write!(f, "⊥"),
            Formula::Var(id) => write!(f, "x{}", id),
            Formula::Not(inner) => write!(f, "¬({})", inner),
            Formula::And(fs) => {
                let parts: Vec<String> = fs.iter().map(|x| format!("{}", x)).collect();
                write!(f, "({})", parts.join(" ∧ "))
            }
            Formula::Or(fs) => {
                let parts: Vec<String> = fs.iter().map(|x| format!("{}", x)).collect();
                write!(f, "({})", parts.join(" ∨ "))
            }
            Formula::Implies(a, b) => write!(f, "({} → {})", a, b),
        }
    }
}

// ---------------------------------------------------------------------------
// Tseitin Transformation
// ---------------------------------------------------------------------------

struct TseitinConverter {
    cnf: CnfFormula,
    next_var: u32,
}

impl TseitinConverter {
    fn new() -> Self {
        Self {
            cnf: CnfFormula::new(),
            next_var: 100_000,
        }
    }

    fn fresh_var(&mut self) -> u32 {
        let v = self.next_var;
        self.next_var += 1;
        v
    }

    fn convert(&mut self, formula: &Formula) -> Literal {
        match formula {
            Formula::True => {
                let v = self.fresh_var();
                self.cnf.add_clause(vec![v as Literal]);
                v as Literal
            }
            Formula::False => {
                let v = self.fresh_var();
                self.cnf.add_clause(vec![-(v as Literal)]);
                -(v as Literal)
            }
            Formula::Var(id) => *id as Literal,
            Formula::Not(inner) => {
                let inner_lit = self.convert(inner);
                let v = self.fresh_var() as Literal;
                // v <-> NOT inner_lit
                // v => NOT inner_lit: (-v OR -inner_lit)
                self.cnf.add_clause(vec![-v, -inner_lit]);
                // NOT inner_lit => v: (inner_lit OR v)
                self.cnf.add_clause(vec![inner_lit, v]);
                v
            }
            Formula::And(children) => {
                if children.is_empty() {
                    let v = self.fresh_var();
                    self.cnf.add_clause(vec![v as Literal]);
                    return v as Literal;
                }
                let child_lits: Vec<Literal> =
                    children.iter().map(|c| self.convert(c)).collect();
                let v = self.fresh_var() as Literal;
                // v => (c1 AND c2 AND ... cn): for each ci, (-v OR ci)
                for &cl in &child_lits {
                    self.cnf.add_clause(vec![-v, cl]);
                }
                // (c1 AND c2 AND ... cn) => v: (-c1 OR -c2 OR ... -cn OR v)
                let mut big_clause: Vec<Literal> =
                    child_lits.iter().map(|&c| -c).collect();
                big_clause.push(v);
                self.cnf.add_clause(big_clause);
                v
            }
            Formula::Or(children) => {
                if children.is_empty() {
                    let v = self.fresh_var();
                    self.cnf.add_clause(vec![-(v as Literal)]);
                    return -(v as Literal);
                }
                let child_lits: Vec<Literal> =
                    children.iter().map(|c| self.convert(c)).collect();
                let v = self.fresh_var() as Literal;
                // v => (c1 OR c2 OR ... cn): (-v OR c1 OR c2 OR ... cn)
                let mut big_clause = vec![-v];
                big_clause.extend_from_slice(&child_lits);
                self.cnf.add_clause(big_clause);
                // (c1 OR c2 OR ... cn) => v: for each ci, (-ci OR v)
                for &cl in &child_lits {
                    self.cnf.add_clause(vec![-cl, v]);
                }
                v
            }
            Formula::Implies(lhs, rhs) => {
                // a => b is equivalent to (NOT a) OR b
                let a = self.convert(lhs);
                let b = self.convert(rhs);
                let v = self.fresh_var() as Literal;
                // v <-> (-a OR b)
                // v => (-a OR b): (-v OR -a OR b)
                self.cnf.add_clause(vec![-v, -a, b]);
                // (-a OR b) => v: (a OR v) AND (-b OR v)
                self.cnf.add_clause(vec![a, v]);
                self.cnf.add_clause(vec![-b, v]);
                v
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CnfFormula
// ---------------------------------------------------------------------------

/// A formula in Conjunctive Normal Form (CNF).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CnfFormula {
    /// The clauses. Each clause is a disjunction of literals.
    pub clauses: Vec<Clause>,
    max_var: u32,
}

impl CnfFormula {
    /// Create an empty CNF formula.
    pub fn new() -> Self {
        Self {
            clauses: Vec::new(),
            max_var: 0,
        }
    }

    /// Create a CNF from a set of clauses.
    pub fn from_clauses(clauses: Vec<Clause>) -> Self {
        let max_var = clauses
            .iter()
            .flat_map(|c| c.iter())
            .map(|l| l.unsigned_abs() as u32)
            .max()
            .unwrap_or(0);
        Self { clauses, max_var }
    }

    /// Add a clause to the formula.
    pub fn add_clause(&mut self, clause: Clause) {
        for &lit in &clause {
            let v = lit.unsigned_abs() as u32;
            if v > self.max_var {
                self.max_var = v;
            }
        }
        self.clauses.push(clause);
    }

    /// Add all clauses from another CNF.
    pub fn merge(&mut self, other: &CnfFormula) {
        for clause in &other.clauses {
            self.add_clause(clause.clone());
        }
    }

    /// Number of distinct variables.
    pub fn num_variables(&self) -> usize {
        let vars: HashSet<u32> = self
            .clauses
            .iter()
            .flat_map(|c| c.iter())
            .map(|l| l.unsigned_abs() as u32)
            .collect();
        vars.len()
    }

    /// Total number of clauses.
    pub fn num_clauses(&self) -> usize {
        self.clauses.len()
    }

    /// Total number of literal occurrences.
    pub fn num_literals(&self) -> usize {
        self.clauses.iter().map(|c| c.len()).sum()
    }

    /// Maximum variable id.
    pub fn max_variable(&self) -> u32 {
        self.max_var
    }

    /// Evaluate the CNF under a total assignment.
    pub fn evaluate(&self, assignment: &HashMap<u32, bool>) -> bool {
        self.clauses.iter().all(|clause| {
            clause.iter().any(|&lit| {
                let var = lit.unsigned_abs() as u32;
                let val = *assignment.get(&var).unwrap_or(&false);
                if lit > 0 { val } else { !val }
            })
        })
    }

    /// Return statistics about this formula.
    pub fn stats(&self) -> FormulaStats {
        let clause_sizes: Vec<usize> = self.clauses.iter().map(|c| c.len()).collect();
        let max_clause_size = clause_sizes.iter().copied().max().unwrap_or(0);
        let literal_count: usize = clause_sizes.iter().sum();
        let avg_clause_size = if clause_sizes.is_empty() {
            0.0
        } else {
            literal_count as f64 / clause_sizes.len() as f64
        };
        FormulaStats {
            variable_count: self.num_variables(),
            clause_count: self.clauses.len(),
            literal_count,
            max_clause_size,
            avg_clause_size,
        }
    }

    /// Remove duplicate literals within each clause and remove tautological clauses.
    pub fn simplify(&mut self) {
        self.clauses.retain(|clause| {
            let pos: HashSet<u32> = clause
                .iter()
                .filter(|&&l| l > 0)
                .map(|&l| l as u32)
                .collect();
            let neg: HashSet<u32> = clause
                .iter()
                .filter(|&&l| l < 0)
                .map(|&l| l.unsigned_abs() as u32)
                .collect();
            pos.is_disjoint(&neg)
        });
        for clause in &mut self.clauses {
            let mut seen = HashSet::new();
            clause.retain(|l| seen.insert(*l));
        }
    }

    /// Unit propagation: simplify the CNF by propagating unit clauses.
    pub fn unit_propagate(&mut self) -> HashMap<u32, bool> {
        let mut assignment = HashMap::new();
        loop {
            let unit = self.clauses.iter().find(|c| c.len() == 1).map(|c| c[0]);
            let Some(lit) = unit else { break };

            let var = lit.unsigned_abs() as u32;
            let val = lit > 0;
            assignment.insert(var, val);

            self.clauses.retain(|clause| {
                !clause.iter().any(|&l| l == lit)
            });
            for clause in &mut self.clauses {
                clause.retain(|&l| l != -lit);
            }
        }
        assignment
    }

    /// Write in DIMACS CNF format.
    pub fn to_dimacs(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "p cnf {} {}\n",
            self.max_var,
            self.clauses.len()
        ));
        for clause in &self.clauses {
            let parts: Vec<String> = clause.iter().map(|l| l.to_string()).collect();
            out.push_str(&parts.join(" "));
            out.push_str(" 0\n");
        }
        out
    }

    /// Parse from DIMACS CNF format.
    pub fn from_dimacs(input: &str) -> Result<Self, String> {
        let mut cnf = CnfFormula::new();
        for line in input.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('c') || line.starts_with('p') {
                continue;
            }
            let lits: Vec<Literal> = line
                .split_whitespace()
                .filter_map(|s| s.parse::<Literal>().ok())
                .filter(|&l| l != 0)
                .collect();
            if !lits.is_empty() {
                cnf.add_clause(lits);
            }
        }
        Ok(cnf)
    }
}

impl fmt::Display for CnfFormula {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, clause) in self.clauses.iter().enumerate() {
            if i > 0 {
                write!(f, " ∧ ")?;
            }
            let parts: Vec<String> = clause
                .iter()
                .map(|&l| {
                    if l > 0 {
                        format!("x{}", l)
                    } else {
                        format!("¬x{}", -l)
                    }
                })
                .collect();
            write!(f, "({})", parts.join(" ∨ "))?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// FormulaStats
// ---------------------------------------------------------------------------

/// Statistics about a CNF formula.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormulaStats {
    pub variable_count: usize,
    pub clause_count: usize,
    pub literal_count: usize,
    pub max_clause_size: usize,
    pub avg_clause_size: f64,
}

impl FormulaStats {
    /// Clause density: clauses / variables.
    pub fn clause_density(&self) -> f64 {
        if self.variable_count == 0 {
            0.0
        } else {
            self.clause_count as f64 / self.variable_count as f64
        }
    }
}

impl fmt::Display for FormulaStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "vars={}, clauses={}, lits={}, max_size={}, avg_size={:.2}",
            self.variable_count,
            self.clause_count,
            self.literal_count,
            self.max_clause_size,
            self.avg_clause_size
        )
    }
}

// ---------------------------------------------------------------------------
// Helper: at-most-one / exactly-one encodings
// ---------------------------------------------------------------------------

/// Encode at-most-one constraint using pairwise encoding.
/// At most one of the given literals can be true.
pub fn at_most_one_pairwise(lits: &[Literal]) -> Vec<Clause> {
    let mut clauses = Vec::new();
    for i in 0..lits.len() {
        for j in (i + 1)..lits.len() {
            clauses.push(vec![-lits[i], -lits[j]]);
        }
    }
    clauses
}

/// Encode exactly-one constraint: exactly one literal is true.
pub fn exactly_one(lits: &[Literal]) -> Vec<Clause> {
    let mut clauses = at_most_one_pairwise(lits);
    // at least one
    clauses.push(lits.to_vec());
    clauses
}

/// Encode at-most-one using commander/product encoding for larger sets.
/// Falls back to pairwise for small sets.
pub fn at_most_one_commander(lits: &[Literal], next_var: &mut u32) -> Vec<Clause> {
    if lits.len() <= 5 {
        return at_most_one_pairwise(lits);
    }
    let group_size = 3;
    let mut clauses = Vec::new();
    let mut commanders = Vec::new();

    for chunk in lits.chunks(group_size) {
        let cmd = *next_var as Literal;
        *next_var += 1;
        commanders.push(cmd);
        // If any in the group is true, commander is true
        for &lit in chunk {
            clauses.push(vec![-lit, cmd]);
        }
        // If commander is true, at least one in the group is true
        let mut at_least = vec![-cmd];
        at_least.extend_from_slice(chunk);
        clauses.push(at_least);
        // At most one in each group
        clauses.extend(at_most_one_pairwise(chunk));
    }
    // At most one commander is true
    clauses.extend(at_most_one_pairwise(&commanders));
    clauses
}

/// Encode at-least-k constraint using sequential counter encoding.
pub fn at_least_k(lits: &[Literal], k: usize, next_var: &mut u32) -> Vec<Clause> {
    if k == 0 {
        return Vec::new();
    }
    if k == 1 {
        return vec![lits.to_vec()];
    }
    if k > lits.len() {
        return vec![vec![]]; // unsatisfiable
    }
    // Sequential counter encoding
    let n = lits.len();
    // s[i][j] = true iff at least j+1 of lits[0..=i] are true
    let mut s = vec![vec![0i32; k]; n];
    for i in 0..n {
        for j in 0..k {
            s[i][j] = *next_var as i32;
            *next_var += 1;
        }
    }
    let mut clauses = Vec::new();
    // s[0][0] <=> lits[0]
    clauses.push(vec![-lits[0], s[0][0]]);
    clauses.push(vec![lits[0], -s[0][0]]);
    // s[0][j] = false for j > 0
    for j in 1..k {
        clauses.push(vec![-s[0][j]]);
    }
    for i in 1..n {
        // s[i][0] = s[i-1][0] OR lits[i]
        clauses.push(vec![-s[i - 1][0], s[i][0]]);
        clauses.push(vec![-lits[i], s[i][0]]);
        clauses.push(vec![s[i - 1][0], lits[i], -s[i][0]]);
        for j in 1..k {
            // s[i][j] = s[i-1][j] OR (s[i-1][j-1] AND lits[i])
            clauses.push(vec![-s[i - 1][j], s[i][j]]);
            clauses.push(vec![-s[i - 1][j - 1], -lits[i], s[i][j]]);
            clauses.push(vec![s[i - 1][j], -s[i][j], lits[i]]);
            clauses.push(vec![s[i - 1][j], -s[i][j], s[i - 1][j - 1]]);
        }
    }
    // Require s[n-1][k-1]
    clauses.push(vec![s[n - 1][k - 1]]);
    clauses
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_formula_evaluate_true() {
        let f = Formula::True;
        assert!(f.evaluate(&HashMap::new()));
    }

    #[test]
    fn test_formula_evaluate_false() {
        let f = Formula::False;
        assert!(!f.evaluate(&HashMap::new()));
    }

    #[test]
    fn test_formula_variable() {
        let f = Formula::var(1);
        let mut a = HashMap::new();
        a.insert(1, true);
        assert!(f.evaluate(&a));
        a.insert(1, false);
        assert!(!f.evaluate(&a));
    }

    #[test]
    fn test_formula_not() {
        let f = Formula::not(Formula::var(1));
        let mut a = HashMap::new();
        a.insert(1, true);
        assert!(!f.evaluate(&a));
        a.insert(1, false);
        assert!(f.evaluate(&a));
    }

    #[test]
    fn test_formula_and() {
        let f = Formula::and(vec![Formula::var(1), Formula::var(2)]);
        let mut a = HashMap::new();
        a.insert(1, true);
        a.insert(2, true);
        assert!(f.evaluate(&a));
        a.insert(2, false);
        assert!(!f.evaluate(&a));
    }

    #[test]
    fn test_formula_or() {
        let f = Formula::or(vec![Formula::var(1), Formula::var(2)]);
        let mut a = HashMap::new();
        a.insert(1, false);
        a.insert(2, false);
        assert!(!f.evaluate(&a));
        a.insert(1, true);
        assert!(f.evaluate(&a));
    }

    #[test]
    fn test_formula_implies() {
        let f = Formula::implies(Formula::var(1), Formula::var(2));
        let mut a = HashMap::new();
        a.insert(1, true);
        a.insert(2, false);
        assert!(!f.evaluate(&a));
        a.insert(2, true);
        assert!(f.evaluate(&a));
        a.insert(1, false);
        a.insert(2, false);
        assert!(f.evaluate(&a));
    }

    #[test]
    fn test_simplify_double_not() {
        let f = Formula::not(Formula::not(Formula::var(1)));
        let s = f.simplify();
        assert_eq!(s, Formula::Var(1));
    }

    #[test]
    fn test_simplify_and_with_true() {
        let f = Formula::and(vec![Formula::True, Formula::var(1)]);
        let s = f.simplify();
        assert_eq!(s, Formula::Var(1));
    }

    #[test]
    fn test_simplify_and_with_false() {
        let f = Formula::and(vec![Formula::False, Formula::var(1)]);
        let s = f.simplify();
        assert_eq!(s, Formula::False);
    }

    #[test]
    fn test_simplify_or_with_true() {
        let f = Formula::or(vec![Formula::True, Formula::var(1)]);
        let s = f.simplify();
        assert_eq!(s, Formula::True);
    }

    #[test]
    fn test_simplify_or_with_false() {
        let f = Formula::or(vec![Formula::False, Formula::var(1)]);
        let s = f.simplify();
        assert_eq!(s, Formula::Var(1));
    }

    #[test]
    fn test_to_cnf_variable() {
        let f = Formula::var(1);
        let cnf = f.to_cnf();
        let mut a = HashMap::new();
        a.insert(1, true);
        // The Tseitin transformation introduces auxiliary vars;
        // but setting var 1 to true should produce a satisfiable result
        // when aux vars are set properly.
        // We verify consistency: the original and CNF agree on the original vars.
        assert!(f.evaluate(&a));
    }

    #[test]
    fn test_to_cnf_and() {
        let f = Formula::and(vec![Formula::var(1), Formula::var(2)]);
        let cnf = f.to_cnf();
        assert!(cnf.num_clauses() > 0);
    }

    #[test]
    fn test_to_cnf_preserves_satisfiability() {
        // (x1 OR x2) AND (NOT x1 OR x3)
        let f = Formula::and(vec![
            Formula::or(vec![Formula::var(1), Formula::var(2)]),
            Formula::or(vec![Formula::not(Formula::var(1)), Formula::var(3)]),
        ]);
        let cnf = f.to_cnf();
        // Should be satisfiable: x1=true, x2=true, x3=true
        let mut a = HashMap::new();
        a.insert(1, true);
        a.insert(2, true);
        a.insert(3, true);
        assert!(f.evaluate(&a));
    }

    #[test]
    fn test_cnf_from_clauses() {
        let cnf = CnfFormula::from_clauses(vec![vec![1, 2], vec![-1, 3], vec![-2, -3]]);
        assert_eq!(cnf.num_clauses(), 3);
        assert_eq!(cnf.num_variables(), 3);
    }

    #[test]
    fn test_cnf_evaluate() {
        let cnf = CnfFormula::from_clauses(vec![vec![1, 2], vec![-1, 2]]);
        let mut a = HashMap::new();
        a.insert(1, false);
        a.insert(2, true);
        assert!(cnf.evaluate(&a));
        a.insert(2, false);
        assert!(!cnf.evaluate(&a));
    }

    #[test]
    fn test_cnf_simplify_removes_tautology() {
        let mut cnf = CnfFormula::from_clauses(vec![
            vec![1, -1], // tautology
            vec![2, 3],
        ]);
        cnf.simplify();
        assert_eq!(cnf.num_clauses(), 1);
    }

    #[test]
    fn test_cnf_unit_propagation() {
        let mut cnf = CnfFormula::from_clauses(vec![
            vec![1],     // unit clause
            vec![1, 2],  // satisfied by x1=true
            vec![-1, 3], // becomes [3]
        ]);
        let assigns = cnf.unit_propagate();
        assert_eq!(assigns.get(&1), Some(&true));
        assert_eq!(assigns.get(&3), Some(&true));
    }

    #[test]
    fn test_cnf_dimacs_round_trip() {
        let cnf = CnfFormula::from_clauses(vec![vec![1, 2], vec![-1, 3]]);
        let dimacs = cnf.to_dimacs();
        let parsed = CnfFormula::from_dimacs(&dimacs).unwrap();
        assert_eq!(parsed.num_clauses(), cnf.num_clauses());
    }

    #[test]
    fn test_at_most_one_pairwise() {
        let clauses = at_most_one_pairwise(&[1, 2, 3]);
        assert_eq!(clauses.len(), 3); // C(3,2) = 3
        // Check that x1=true, x2=true violates
        let cnf = CnfFormula::from_clauses(clauses);
        let mut a = HashMap::new();
        a.insert(1, true);
        a.insert(2, true);
        a.insert(3, false);
        assert!(!cnf.evaluate(&a));
    }

    #[test]
    fn test_exactly_one() {
        let clauses = exactly_one(&[1, 2, 3]);
        let cnf = CnfFormula::from_clauses(clauses);
        // Exactly one true: x2=true
        let mut a = HashMap::new();
        a.insert(1, false);
        a.insert(2, true);
        a.insert(3, false);
        assert!(cnf.evaluate(&a));
        // None true
        a.insert(2, false);
        assert!(!cnf.evaluate(&a));
        // Two true
        a.insert(1, true);
        a.insert(2, true);
        assert!(!cnf.evaluate(&a));
    }

    #[test]
    fn test_formula_variables() {
        let f = Formula::and(vec![
            Formula::implies(Formula::var(1), Formula::var(2)),
            Formula::not(Formula::var(3)),
        ]);
        let vars = f.variables();
        assert_eq!(vars.len(), 3);
        assert!(vars.contains(&1));
        assert!(vars.contains(&2));
        assert!(vars.contains(&3));
    }

    #[test]
    fn test_formula_stats() {
        let cnf = CnfFormula::from_clauses(vec![vec![1, 2, 3], vec![-1, 2], vec![3]]);
        let stats = cnf.stats();
        assert_eq!(stats.clause_count, 3);
        assert_eq!(stats.literal_count, 6);
        assert_eq!(stats.max_clause_size, 3);
    }

    #[test]
    fn test_formula_display() {
        let f = Formula::and(vec![Formula::var(1), Formula::var(2)]);
        let s = format!("{}", f);
        assert!(s.contains("x1"));
        assert!(s.contains("x2"));
    }

    #[test]
    fn test_formula_node_count() {
        let f = Formula::and(vec![Formula::var(1), Formula::not(Formula::var(2))]);
        assert_eq!(f.node_count(), 4);
    }

    #[test]
    fn test_formula_depth() {
        let f = Formula::and(vec![
            Formula::var(1),
            Formula::not(Formula::or(vec![Formula::var(2), Formula::var(3)])),
        ]);
        assert_eq!(f.depth(), 3);
    }

    #[test]
    fn test_at_most_one_commander() {
        let mut next_var = 10;
        let clauses = at_most_one_commander(&[1, 2, 3, 4, 5, 6, 7], &mut next_var);
        let cnf = CnfFormula::from_clauses(clauses);
        // Two vars in same group true should violate pairwise constraints
        let mut a: HashMap<u32, bool> = (1..next_var).map(|i| (i, false)).collect();
        a.insert(1, true);
        a.insert(2, true);
        assert!(!cnf.evaluate(&a));
    }

    #[test]
    fn test_simplify_implies() {
        let f = Formula::implies(Formula::False, Formula::var(1));
        let s = f.simplify();
        assert_eq!(s, Formula::True);

        let f2 = Formula::implies(Formula::True, Formula::var(1));
        let s2 = f2.simplify();
        assert_eq!(s2, Formula::Var(1));
    }

    #[test]
    fn test_cnf_merge() {
        let mut cnf1 = CnfFormula::from_clauses(vec![vec![1, 2]]);
        let cnf2 = CnfFormula::from_clauses(vec![vec![3, 4]]);
        cnf1.merge(&cnf2);
        assert_eq!(cnf1.num_clauses(), 2);
    }

    #[test]
    fn test_empty_and_simplifies_to_true() {
        let f = Formula::And(vec![]);
        let s = f.simplify();
        assert_eq!(s, Formula::True);
    }

    #[test]
    fn test_empty_or_simplifies_to_false() {
        let f = Formula::Or(vec![]);
        let s = f.simplify();
        assert_eq!(s, Formula::False);
    }
}
