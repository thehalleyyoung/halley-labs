//! MaxSAT formula representation with hard and soft clauses.
//!
//! Provides [`MaxSatFormula`] as the central data structure together with
//! helpers for statistics, simplification, DIMACS WCNF I/O, subsumption
//! checking, clause databases, and variable domain tracking.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// Core type aliases
// ---------------------------------------------------------------------------

/// A literal: positive means the variable is asserted, negative means negated.
pub type Literal = i32;

/// Clause identifier (dense, zero-based).
pub type ClauseId = usize;

/// Alias used when the formula is a weighted partial MaxSAT instance.
pub type WeightedPartialMaxSat = MaxSatFormula;

// ---------------------------------------------------------------------------
// Literal helpers
// ---------------------------------------------------------------------------

/// Negate a literal.
pub fn negate_literal(lit: Literal) -> Literal { -lit }

/// Create a positive literal from a variable number.
pub fn positive_literal(var: u32) -> Literal { var as Literal }

/// Create a negative literal from a variable number.
pub fn negative_literal(var: u32) -> Literal { -(var as Literal) }

/// Extract the variable index from a literal.
pub fn lit_var(lit: Literal) -> u32 { lit.unsigned_abs() }

/// Check whether a literal is positive.
pub fn lit_is_positive(lit: Literal) -> bool { lit > 0 }

/// Convert a literal to the index used for two-watched-literal arrays.
pub fn lit_index(lit: Literal) -> usize {
    let v = lit.unsigned_abs() as usize;
    if lit > 0 { 2 * v } else { 2 * v + 1 }
}

/// Evaluate a literal under a (possibly partial) assignment.
pub fn eval_literal(lit: Literal, assignment: &[Option<bool>]) -> Option<bool> {
    let v = lit.unsigned_abs() as usize;
    if v >= assignment.len() { return None; }
    assignment[v].map(|val| if lit > 0 { val } else { !val })
}

// ---------------------------------------------------------------------------
// Clause types
// ---------------------------------------------------------------------------

/// A hard clause that *must* be satisfied.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardClause {
    pub literals: Vec<Literal>,
    pub label: String,
    pub id: ClauseId,
}

impl HardClause {
    pub fn new(literals: Vec<Literal>, label: &str) -> Self {
        Self { literals, label: label.to_string(), id: 0 }
    }

    pub fn is_unit(&self) -> bool { self.literals.len() == 1 }
    pub fn is_empty(&self) -> bool { self.literals.is_empty() }

    pub fn contains(&self, lit: Literal) -> bool { self.literals.contains(&lit) }

    pub fn variables(&self) -> HashSet<u32> {
        self.literals.iter().map(|l| l.unsigned_abs()).collect()
    }

    pub fn is_tautology(&self) -> bool { is_tautology(&self.literals) }

    /// Check whether `self` subsumes `other`.
    pub fn subsumes(&self, other: &HardClause) -> bool {
        self.literals.iter().all(|l| other.literals.contains(l))
    }
}

/// A soft clause whose violation incurs a weighted penalty.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoftClause {
    pub literals: Vec<Literal>,
    pub weight: u64,
    pub label: String,
    pub id: ClauseId,
}

impl SoftClause {
    pub fn new(literals: Vec<Literal>, weight: u64, label: &str) -> Self {
        Self { literals, weight, label: label.to_string(), id: 0 }
    }

    pub fn is_unit(&self) -> bool { self.literals.len() == 1 }
    pub fn is_empty(&self) -> bool { self.literals.is_empty() }
    pub fn contains(&self, lit: Literal) -> bool { self.literals.contains(&lit) }
}

/// A unified clause representation (used for interchange with the SAT oracle).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Clause {
    pub literals: Vec<Literal>,
    pub is_hard: bool,
    pub weight: u64,
}

impl Clause {
    pub fn new(literals: Vec<Literal>) -> Self {
        Self { literals, is_hard: true, weight: u64::MAX }
    }

    pub fn hard(literals: Vec<Literal>) -> Self {
        Self { literals, is_hard: true, weight: u64::MAX }
    }

    pub fn soft(literals: Vec<Literal>, weight: u64) -> Self {
        Self { literals, is_hard: false, weight }
    }

    pub fn is_empty(&self) -> bool { self.literals.is_empty() }
    pub fn len(&self) -> usize { self.literals.len() }
    pub fn is_unit(&self) -> bool { self.literals.len() == 1 }
    pub fn contains(&self, lit: Literal) -> bool { self.literals.contains(&lit) }

    pub fn is_tautology(&self) -> bool { is_tautology(&self.literals) }

    pub fn variables(&self) -> HashSet<u32> {
        self.literals.iter().map(|l| l.unsigned_abs()).collect()
    }

    pub fn subsumes(&self, other: &Clause) -> bool {
        self.literals.iter().all(|l| other.literals.contains(l))
    }

    pub fn evaluate(&self, assignment: &HashMap<u32, bool>) -> Option<bool> {
        let mut all_known = true;
        for &lit in &self.literals {
            let v = lit.unsigned_abs();
            match assignment.get(&v) {
                Some(&val) => {
                    let lit_val = if lit > 0 { val } else { !val };
                    if lit_val { return Some(true); }
                }
                None => { all_known = false; }
            }
        }
        if all_known { Some(false) } else { None }
    }
}

/// Check whether a set of literals contains both a literal and its negation.
pub fn is_tautology(literals: &[Literal]) -> bool {
    for (i, &a) in literals.iter().enumerate() {
        for &b in &literals[i + 1..] {
            if a == -b { return true; }
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Formula
// ---------------------------------------------------------------------------

/// A weighted partial MaxSAT formula.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaxSatFormula {
    pub hard_clauses: Vec<HardClause>,
    pub soft_clauses: Vec<SoftClause>,
    pub num_variables: u32,
    pub next_clause_id: ClauseId,
}

impl MaxSatFormula {
    pub fn new() -> Self {
        Self { hard_clauses: Vec::new(), soft_clauses: Vec::new(), num_variables: 0, next_clause_id: 0 }
    }

    pub fn add_hard_clause(&mut self, literals: Vec<Literal>, label: &str) -> ClauseId {
        let id = self.next_clause_id;
        self.next_clause_id += 1;
        self.track_variables(&literals);
        self.hard_clauses.push(HardClause { literals, label: label.to_string(), id });
        id
    }

    pub fn add_soft_clause(&mut self, literals: Vec<Literal>, weight: u64, label: &str) -> ClauseId {
        let id = self.next_clause_id;
        self.next_clause_id += 1;
        self.track_variables(&literals);
        self.soft_clauses.push(SoftClause { literals, weight, label: label.to_string(), id });
        id
    }

    pub fn num_hard(&self) -> usize { self.hard_clauses.len() }
    pub fn num_soft(&self) -> usize { self.soft_clauses.len() }
    pub fn num_clauses(&self) -> usize { self.hard_clauses.len() + self.soft_clauses.len() }

    pub fn total_soft_weight(&self) -> u64 { self.soft_clauses.iter().map(|c| c.weight).sum() }
    pub fn max_soft_weight(&self) -> u64 { self.soft_clauses.iter().map(|c| c.weight).max().unwrap_or(0) }

    pub fn fresh_variable(&mut self) -> u32 {
        self.num_variables += 1;
        self.num_variables
    }

    pub fn fresh_variables(&mut self, n: u32) -> u32 {
        let first = self.num_variables + 1;
        self.num_variables += n;
        first
    }

    pub fn stats(&self) -> FormulaStats {
        FormulaStats {
            num_hard: self.hard_clauses.len(),
            num_soft: self.soft_clauses.len(),
            num_vars: self.num_variables as usize,
            total_soft_weight: self.total_soft_weight(),
            max_weight: self.max_soft_weight(),
        }
    }

    pub fn referenced_variables(&self) -> HashSet<u32> {
        let mut vars = HashSet::new();
        for hc in &self.hard_clauses { for &l in &hc.literals { vars.insert(l.unsigned_abs()); } }
        for sc in &self.soft_clauses { for &l in &sc.literals { vars.insert(l.unsigned_abs()); } }
        vars
    }

    pub fn to_wcnf_string(&self) -> String { self.to_dimacs_wcnf() }

    pub fn to_dimacs_wcnf(&self) -> String {
        let top = self.total_soft_weight() + 1;
        let total = self.num_clauses();
        let mut out = String::with_capacity(total * 20);
        out.push_str(&format!("p wcnf {} {} {}\n", self.num_variables, total, top));
        for hc in &self.hard_clauses {
            out.push_str(&format!("{top} "));
            for &lit in &hc.literals { out.push_str(&format!("{lit} ")); }
            out.push_str("0\n");
        }
        for sc in &self.soft_clauses {
            out.push_str(&format!("{} ", sc.weight));
            for &lit in &sc.literals { out.push_str(&format!("{lit} ")); }
            out.push_str("0\n");
        }
        out
    }

    pub fn from_wcnf_string(s: &str) -> Result<Self, String> {
        let mut formula = MaxSatFormula::new();
        let mut top: Option<u64> = None;
        let mut found_header = false;
        for line in s.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('c') { continue; }
            if line.starts_with("p ") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() < 4 || parts[1] != "wcnf" {
                    return Err("expected 'p wcnf <vars> <clauses> [top]'".into());
                }
                formula.num_variables = parts[2].parse().map_err(|_| "bad var count")?;
                if parts.len() >= 5 { top = Some(parts[4].parse().map_err(|_| "bad top")?); }
                found_header = true;
                continue;
            }
            if !found_header { return Err("clause before header".into()); }
            let tokens: Vec<&str> = line.split_whitespace().collect();
            if tokens.is_empty() { continue; }
            let weight: u64 = tokens[0].parse().map_err(|_| "bad weight")?;
            let mut literals = Vec::new();
            for tok in &tokens[1..] {
                let val: i32 = tok.parse().map_err(|_| "bad literal")?;
                if val == 0 { break; }
                literals.push(val);
            }
            if top.map_or(false, |t| weight >= t) {
                formula.add_hard_clause(literals, "");
            } else {
                formula.add_soft_clause(literals, weight, "");
            }
        }
        if !found_header { return Err("no header".into()); }
        Ok(formula)
    }

    pub fn remove_tautologies(&mut self) {
        self.hard_clauses.retain(|hc| !is_tautology(&hc.literals));
        self.soft_clauses.retain(|sc| !is_tautology(&sc.literals));
    }

    pub fn deduplicate_literals(&mut self) {
        for hc in &mut self.hard_clauses {
            let mut seen = HashSet::new();
            hc.literals.retain(|l| seen.insert(*l));
        }
        for sc in &mut self.soft_clauses {
            let mut seen = HashSet::new();
            sc.literals.retain(|l| seen.insert(*l));
        }
    }

    fn track_variables(&mut self, literals: &[Literal]) {
        for &lit in literals {
            let var = lit.unsigned_abs();
            if var > self.num_variables { self.num_variables = var; }
        }
    }
}

impl Default for MaxSatFormula {
    fn default() -> Self { Self::new() }
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormulaStats {
    pub num_hard: usize,
    pub num_soft: usize,
    pub num_vars: usize,
    pub total_soft_weight: u64,
    pub max_weight: u64,
}

impl std::fmt::Display for FormulaStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "vars={}, hard={}, soft={}, total_w={}, max_w={}",
            self.num_vars, self.num_hard, self.num_soft, self.total_soft_weight, self.max_weight)
    }
}

// ---------------------------------------------------------------------------
// Simplifier
// ---------------------------------------------------------------------------

/// Simplifier: tautology removal, unit propagation, pure literal elimination.
#[derive(Debug, Clone)]
pub struct FormulaSimplifier {
    pub assignments: HashMap<u32, bool>,
    pub removed_count: usize,
}

impl FormulaSimplifier {
    pub fn new() -> Self { Self { assignments: HashMap::new(), removed_count: 0 } }

    pub fn simplify(&mut self, formula: &mut MaxSatFormula) -> usize {
        let before = formula.num_clauses();
        self.unit_propagation(formula);
        self.pure_literal_elimination(formula);
        formula.remove_tautologies();
        formula.deduplicate_literals();
        self.removed_count = before.saturating_sub(formula.num_clauses());
        self.removed_count
    }

    pub fn unit_propagation(&mut self, formula: &mut MaxSatFormula) {
        loop {
            let unit = formula.hard_clauses.iter()
                .find(|hc| hc.literals.len() == 1)
                .map(|hc| hc.literals[0]);
            match unit {
                Some(lit) => {
                    self.assignments.insert(lit.unsigned_abs(), lit > 0);
                    propagate_literal_formula(formula, lit);
                }
                None => break,
            }
        }
    }

    pub fn pure_literal_elimination(&mut self, formula: &mut MaxSatFormula) {
        let mut pos = HashSet::new();
        let mut neg = HashSet::new();
        for hc in &formula.hard_clauses {
            for &lit in &hc.literals {
                if lit > 0 { pos.insert(lit.unsigned_abs()); }
                else { neg.insert(lit.unsigned_abs()); }
            }
        }
        for v in pos.difference(&neg).copied().collect::<Vec<_>>() {
            self.assignments.insert(v, true);
            propagate_literal_formula(formula, v as i32);
        }
        for v in neg.difference(&pos).copied().collect::<Vec<_>>() {
            self.assignments.insert(v, false);
            propagate_literal_formula(formula, -(v as i32));
        }
    }
}

impl Default for FormulaSimplifier {
    fn default() -> Self { Self::new() }
}

fn propagate_literal_formula(formula: &mut MaxSatFormula, lit: Literal) {
    let neg = -lit;
    formula.hard_clauses.retain(|hc| !hc.literals.contains(&lit));
    for hc in &mut formula.hard_clauses { hc.literals.retain(|&l| l != neg); }
    formula.soft_clauses.retain(|sc| !sc.literals.contains(&lit));
    for sc in &mut formula.soft_clauses { sc.literals.retain(|&l| l != neg); }
}

// ---------------------------------------------------------------------------
// ClauseDatabase
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClauseDatabase {
    pub formula: MaxSatFormula,
    #[serde(skip)]
    occurrence: HashMap<u32, Vec<usize>>,
}

impl ClauseDatabase {
    pub fn new() -> Self { Self { formula: MaxSatFormula::new(), occurrence: HashMap::new() } }

    pub fn from_formula(formula: MaxSatFormula) -> Self {
        let mut db = Self { formula, occurrence: HashMap::new() };
        db.rebuild_occurrence();
        db
    }

    fn rebuild_occurrence(&mut self) {
        self.occurrence.clear();
        for (i, hc) in self.formula.hard_clauses.iter().enumerate() {
            for &lit in &hc.literals {
                self.occurrence.entry(lit.unsigned_abs()).or_default().push(i);
            }
        }
    }

    pub fn add_hard(&mut self, literals: Vec<Literal>, label: &str) -> ClauseId {
        let idx = self.formula.hard_clauses.len();
        for &lit in &literals { self.occurrence.entry(lit.unsigned_abs()).or_default().push(idx); }
        self.formula.add_hard_clause(literals, label)
    }

    pub fn add_soft(&mut self, literals: Vec<Literal>, weight: u64, label: &str) -> ClauseId {
        self.formula.add_soft_clause(literals, weight, label)
    }

    pub fn clause_count(&self) -> usize { self.formula.num_clauses() }

    pub fn clauses_with_variable(&self, var: u32) -> &[usize] {
        self.occurrence.get(&var).map_or(&[], |v| v.as_slice())
    }

    pub fn find_subsumed(&self, clause: &[Literal]) -> Vec<usize> {
        self.formula.hard_clauses.iter().enumerate()
            .filter(|(_, hc)| clause.iter().all(|l| hc.literals.contains(l)) && hc.literals.len() > clause.len())
            .map(|(i, _)| i).collect()
    }
}

impl Default for ClauseDatabase { fn default() -> Self { Self::new() } }

// ---------------------------------------------------------------------------
// Variable domain tracking
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DomainType {
    Boolean,
    IntRange { min: i64, max: i64 },
    Enumeration(Vec<String>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariableDomain {
    pub variable: u32,
    pub name: String,
    pub domain_type: DomainType,
}

#[derive(Debug, Clone, Default)]
pub struct VariableDomainTracker {
    domains: HashMap<u32, VariableDomain>,
}

impl VariableDomainTracker {
    pub fn new() -> Self { Self { domains: HashMap::new() } }

    pub fn add_boolean(&mut self, var: u32, name: &str) {
        self.domains.insert(var, VariableDomain { variable: var, name: name.to_string(), domain_type: DomainType::Boolean });
    }

    pub fn add_int_range(&mut self, var: u32, name: &str, min: i64, max: i64) {
        self.domains.insert(var, VariableDomain { variable: var, name: name.to_string(), domain_type: DomainType::IntRange { min, max } });
    }

    pub fn get(&self, var: u32) -> Option<&VariableDomain> { self.domains.get(&var) }
    pub fn len(&self) -> usize { self.domains.len() }
    pub fn is_empty(&self) -> bool { self.domains.is_empty() }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_formula_empty() {
        let f = MaxSatFormula::new();
        assert_eq!(f.num_hard(), 0);
        assert_eq!(f.num_soft(), 0);
        assert_eq!(f.num_variables, 0);
    }

    #[test]
    fn test_add_hard_clause() {
        let mut f = MaxSatFormula::new();
        let id = f.add_hard_clause(vec![1, -2, 3], "test");
        assert_eq!(id, 0);
        assert_eq!(f.num_hard(), 1);
        assert_eq!(f.num_variables, 3);
    }

    #[test]
    fn test_add_soft_clause() {
        let mut f = MaxSatFormula::new();
        f.add_soft_clause(vec![1, 2], 10, "s1");
        f.add_soft_clause(vec![-3], 5, "s2");
        assert_eq!(f.num_soft(), 2);
        assert_eq!(f.total_soft_weight(), 15);
        assert_eq!(f.num_variables, 3);
    }

    #[test]
    fn test_fresh_variable() {
        let mut f = MaxSatFormula::new();
        f.add_hard_clause(vec![1, 2], "h");
        let v = f.fresh_variable();
        assert_eq!(v, 3);
    }

    #[test]
    fn test_fresh_variables_batch() {
        let mut f = MaxSatFormula::new();
        f.num_variables = 5;
        let first = f.fresh_variables(3);
        assert_eq!(first, 6);
        assert_eq!(f.num_variables, 8);
    }

    #[test]
    fn test_wcnf_round_trip() {
        let mut f = MaxSatFormula::new();
        f.add_hard_clause(vec![1, -2], "h1");
        f.add_hard_clause(vec![2, 3], "h2");
        f.add_soft_clause(vec![1], 5, "s1");
        f.add_soft_clause(vec![-3], 10, "s2");
        let wcnf = f.to_wcnf_string();
        let f2 = MaxSatFormula::from_wcnf_string(&wcnf).unwrap();
        assert_eq!(f2.num_hard(), 2);
        assert_eq!(f2.num_soft(), 2);
        assert_eq!(f2.total_soft_weight(), 15);
    }

    #[test]
    fn test_wcnf_parse_comments() {
        let wcnf = "c comment\np wcnf 3 3 16\n16 1 -2 0\n5 1 0\n10 -3 0\n";
        let f = MaxSatFormula::from_wcnf_string(wcnf).unwrap();
        assert_eq!(f.num_hard(), 1);
        assert_eq!(f.num_soft(), 2);
    }

    #[test]
    fn test_wcnf_parse_error() {
        assert!(MaxSatFormula::from_wcnf_string("1 1 -2 0").is_err());
    }

    #[test]
    fn test_formula_stats() {
        let mut f = MaxSatFormula::new();
        f.add_hard_clause(vec![1], "h");
        f.add_soft_clause(vec![2], 7, "s");
        f.add_soft_clause(vec![-2], 13, "s2");
        let st = f.stats();
        assert_eq!(st.num_hard, 1);
        assert_eq!(st.num_soft, 2);
        assert_eq!(st.total_soft_weight, 20);
        assert_eq!(st.max_weight, 13);
    }

    #[test]
    fn test_simplifier_tautology() {
        let mut f = MaxSatFormula::new();
        f.add_hard_clause(vec![1, -1], "taut");
        f.add_hard_clause(vec![2, 3], "ok");
        let removed = FormulaSimplifier::new().simplify(&mut f);
        assert_eq!(removed, 1);
        assert_eq!(f.num_hard(), 1);
    }

    #[test]
    fn test_simplifier_unit_prop() {
        let mut f = MaxSatFormula::new();
        f.add_hard_clause(vec![1], "unit");
        f.add_hard_clause(vec![-1, 2], "c2");
        let mut simp = FormulaSimplifier::new();
        simp.unit_propagation(&mut f);
        assert_eq!(*simp.assignments.get(&1).unwrap(), true);
    }

    #[test]
    fn test_simplifier_pure_literal() {
        let mut f = MaxSatFormula::new();
        f.add_hard_clause(vec![1, 2], "c1");
        f.add_hard_clause(vec![1, -2], "c2");
        let mut simp = FormulaSimplifier::new();
        simp.pure_literal_elimination(&mut f);
        assert_eq!(*simp.assignments.get(&1).unwrap(), true);
        assert_eq!(f.num_hard(), 0);
    }

    #[test]
    fn test_clause_subsumes() {
        let small = Clause::new(vec![1]);
        let big = Clause::new(vec![1, -2]);
        assert!(small.subsumes(&big));
        assert!(!big.subsumes(&small));
    }

    #[test]
    fn test_clause_evaluate() {
        let c = Clause::new(vec![1, -2]);
        let mut a = HashMap::new();
        a.insert(1u32, true);
        assert_eq!(c.evaluate(&a), Some(true));
        a.clear();
        a.insert(1, false); a.insert(2, true);
        assert_eq!(c.evaluate(&a), Some(false));
    }

    #[test]
    fn test_clause_database() {
        let mut db = ClauseDatabase::new();
        db.add_hard(vec![1, 2], "h1");
        db.add_soft(vec![-1], 5, "s1");
        assert_eq!(db.clause_count(), 2);
        assert!(!db.clauses_with_variable(1).is_empty());
    }

    #[test]
    fn test_domain_tracker() {
        let mut t = VariableDomainTracker::new();
        t.add_boolean(1, "x1");
        t.add_int_range(2, "timeout", 100, 30000);
        assert_eq!(t.len(), 2);
        assert!(matches!(t.get(1).unwrap().domain_type, DomainType::Boolean));
    }

    #[test]
    fn test_remove_tautologies() {
        let mut f = MaxSatFormula::new();
        f.add_hard_clause(vec![1, -1], "t");
        f.add_hard_clause(vec![2], "ok");
        f.remove_tautologies();
        assert_eq!(f.num_hard(), 1);
    }

    #[test]
    fn test_deduplicate_literals() {
        let mut f = MaxSatFormula::new();
        f.add_hard_clause(vec![1, 1, -2], "dup");
        f.deduplicate_literals();
        assert_eq!(f.hard_clauses[0].literals.len(), 2);
    }

    #[test]
    fn test_literal_helpers() {
        assert_eq!(lit_var(5), 5);
        assert_eq!(lit_var(-3), 3);
        assert!(lit_is_positive(5));
        assert!(!lit_is_positive(-5));
        assert_eq!(negate_literal(3), -3);
        assert_eq!(positive_literal(4), 4);
        assert_eq!(negative_literal(4), -4);
    }

    #[test]
    fn test_formula_serialization() {
        let mut f = MaxSatFormula::new();
        f.add_hard_clause(vec![1, -2], "h");
        f.add_soft_clause(vec![3], 10, "s");
        let json = serde_json::to_string(&f).unwrap();
        let deser: MaxSatFormula = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.num_hard(), 1);
        assert_eq!(deser.num_soft(), 1);
    }

    #[test]
    fn test_referenced_variables() {
        let mut f = MaxSatFormula::new();
        f.add_hard_clause(vec![1, -2], "h");
        f.add_soft_clause(vec![3], 5, "s");
        let vars = f.referenced_variables();
        assert!(vars.contains(&1) && vars.contains(&2) && vars.contains(&3));
    }

    #[test]
    fn test_hard_clause_methods() {
        let hc = HardClause::new(vec![1, -2, 3], "test");
        assert!(!hc.is_unit());
        assert!(!hc.is_empty());
        assert!(hc.contains(1));
        assert!(hc.contains(-2));
        assert!(!hc.contains(4));
        assert!(!hc.is_tautology());
    }

    #[test]
    fn test_hard_clause_subsumption() {
        let s = HardClause::new(vec![1], "s");
        let b = HardClause::new(vec![1, -2], "b");
        assert!(s.subsumes(&b));
        assert!(!b.subsumes(&s));
    }
}
