// regsynth-solver: result types
// Unified result types for SAT, SMT, MaxSMT, ILP, and Pareto solvers.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::time::Duration;

// ─── Fundamental Types ──────────────────────────────────────────────────────

/// A Boolean variable, identified by a non-negative integer.
pub type Variable = u32;

/// A literal: a signed integer where `abs(lit)` is the variable and
/// the sign encodes polarity (positive = true, negative = false).
/// Variable 0 is reserved; valid variables start at 1.
pub type Literal = i32;

/// Return the variable of a literal.
#[inline]
pub fn lit_var(lit: Literal) -> Variable {
    lit.unsigned_abs()
}

/// Return the polarity of a literal (true = positive).
#[inline]
pub fn lit_sign(lit: Literal) -> bool {
    lit > 0
}

/// Negate a literal.
#[inline]
pub fn lit_neg(lit: Literal) -> Literal {
    -lit
}

/// Create a literal from a variable and polarity.
#[inline]
pub fn make_lit(var: Variable, positive: bool) -> Literal {
    if positive {
        var as Literal
    } else {
        -(var as Literal)
    }
}

/// A clause is a disjunction of literals.
pub type Clause = Vec<Literal>;

// ─── Assignment ─────────────────────────────────────────────────────────────

/// A (partial or total) Boolean assignment.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Assignment {
    pub values: HashMap<Variable, bool>,
}

impl Assignment {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set(&mut self, var: Variable, val: bool) {
        self.values.insert(var, val);
    }

    pub fn get(&self, var: Variable) -> Option<bool> {
        self.values.get(&var).copied()
    }

    pub fn eval_lit(&self, lit: Literal) -> Option<bool> {
        self.get(lit_var(lit)).map(|v| if lit_sign(lit) { v } else { !v })
    }

    pub fn num_assigned(&self) -> usize {
        self.values.len()
    }
}

impl fmt::Display for Assignment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut pairs: Vec<_> = self.values.iter().collect();
        pairs.sort_by_key(|(v, _)| *v);
        write!(f, "{{")?;
        for (i, (var, val)) in pairs.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "x{}={}", var, val)?;
        }
        write!(f, "}}")
    }
}

// ─── SAT Result ─────────────────────────────────────────────────────────────

/// Result of a SAT solver invocation.
#[derive(Debug, Clone)]
pub enum SatResult {
    Sat(Assignment),
    Unsat(Vec<Clause>),
    Unknown(String),
}

impl SatResult {
    pub fn is_sat(&self) -> bool {
        matches!(self, SatResult::Sat(_))
    }

    pub fn is_unsat(&self) -> bool {
        matches!(self, SatResult::Unsat(_))
    }

    pub fn assignment(&self) -> Option<&Assignment> {
        match self {
            SatResult::Sat(a) => Some(a),
            _ => None,
        }
    }

    pub fn unsat_core(&self) -> Option<&[Clause]> {
        match self {
            SatResult::Unsat(core) => Some(core),
            _ => None,
        }
    }
}

impl fmt::Display for SatResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SatResult::Sat(a) => write!(f, "SAT: {}", a),
            SatResult::Unsat(core) => write!(f, "UNSAT (core size: {})", core.len()),
            SatResult::Unknown(reason) => write!(f, "UNKNOWN: {}", reason),
        }
    }
}

// ─── SMT Model ──────────────────────────────────────────────────────────────

/// A model from an SMT solver, mapping variable names to values.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Model {
    pub bool_values: HashMap<String, bool>,
    pub int_values: HashMap<String, i64>,
    pub real_values: HashMap<String, f64>,
}

impl Model {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_bool(&mut self, name: impl Into<String>, val: bool) {
        self.bool_values.insert(name.into(), val);
    }

    pub fn set_int(&mut self, name: impl Into<String>, val: i64) {
        self.int_values.insert(name.into(), val);
    }

    pub fn set_real(&mut self, name: impl Into<String>, val: f64) {
        self.real_values.insert(name.into(), val);
    }

    pub fn get_bool(&self, name: &str) -> Option<bool> {
        self.bool_values.get(name).copied()
    }

    pub fn get_int(&self, name: &str) -> Option<i64> {
        self.int_values.get(name).copied()
    }

    pub fn get_real(&self, name: &str) -> Option<f64> {
        self.real_values.get(name).copied()
    }
}

impl fmt::Display for Model {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Model{{")?;
        let mut first = true;
        for (k, v) in &self.bool_values {
            if !first {
                write!(f, ", ")?;
            }
            write!(f, "{}={}", k, v)?;
            first = false;
        }
        for (k, v) in &self.int_values {
            if !first {
                write!(f, ", ")?;
            }
            write!(f, "{}={}", k, v)?;
            first = false;
        }
        for (k, v) in &self.real_values {
            if !first {
                write!(f, ", ")?;
            }
            write!(f, "{}={:.4}", k, v)?;
            first = false;
        }
        write!(f, "}}")
    }
}

// ─── SMT Result ─────────────────────────────────────────────────────────────

/// Result of an SMT solver invocation.
#[derive(Debug, Clone)]
pub enum SmtResult {
    Sat(Model),
    Unsat(Vec<usize>),
    Unknown(String),
}

impl SmtResult {
    pub fn is_sat(&self) -> bool {
        matches!(self, SmtResult::Sat(_))
    }

    pub fn is_unsat(&self) -> bool {
        matches!(self, SmtResult::Unsat(_))
    }

    pub fn model(&self) -> Option<&Model> {
        match self {
            SmtResult::Sat(m) => Some(m),
            _ => None,
        }
    }

    pub fn unsat_core_indices(&self) -> Option<&[usize]> {
        match self {
            SmtResult::Unsat(core) => Some(core),
            _ => None,
        }
    }
}

impl fmt::Display for SmtResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SmtResult::Sat(m) => write!(f, "SAT: {}", m),
            SmtResult::Unsat(core) => write!(f, "UNSAT (core indices: {:?})", core),
            SmtResult::Unknown(r) => write!(f, "UNKNOWN: {}", r),
        }
    }
}

// ─── MaxSMT Result ──────────────────────────────────────────────────────────

/// Status of a MaxSMT computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MaxSmtStatus {
    Optimal,
    Satisfiable,
    Unsatisfiable,
    Timeout,
    Unknown,
}

impl fmt::Display for MaxSmtStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Optimal => write!(f, "OPTIMAL"),
            Self::Satisfiable => write!(f, "SATISFIABLE"),
            Self::Unsatisfiable => write!(f, "UNSATISFIABLE"),
            Self::Timeout => write!(f, "TIMEOUT"),
            Self::Unknown => write!(f, "UNKNOWN"),
        }
    }
}

/// Result of a weighted partial MaxSMT computation.
#[derive(Debug, Clone)]
pub struct MaxSmtResult {
    pub status: MaxSmtStatus,
    pub assignment: Option<Assignment>,
    pub cost: f64,
    pub num_violated_soft: usize,
    pub stats: SolverStatistics,
}

impl fmt::Display for MaxSmtResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MaxSMT[{}]: cost={:.4}, violated={}",
            self.status, self.cost, self.num_violated_soft
        )
    }
}

// ─── ILP Result ─────────────────────────────────────────────────────────────

/// A solution to an ILP problem.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IlpSolution {
    pub values: HashMap<String, f64>,
    pub objective_value: f64,
}

impl IlpSolution {
    pub fn get(&self, var: &str) -> Option<f64> {
        self.values.get(var).copied()
    }
}

impl fmt::Display for IlpSolution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "obj={:.4}, vars={{", self.objective_value)?;
        let mut pairs: Vec<_> = self.values.iter().collect();
        pairs.sort_by_key(|(k, _)| k.clone());
        for (i, (k, v)) in pairs.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}={:.4}", k, v)?;
        }
        write!(f, "}}")
    }
}

/// Result of an ILP solver invocation.
#[derive(Debug, Clone)]
pub enum IlpResult {
    Optimal(IlpSolution),
    Feasible(IlpSolution),
    Infeasible,
    Unbounded,
    Timeout,
}

impl IlpResult {
    pub fn is_optimal(&self) -> bool {
        matches!(self, IlpResult::Optimal(_))
    }

    pub fn solution(&self) -> Option<&IlpSolution> {
        match self {
            IlpResult::Optimal(s) | IlpResult::Feasible(s) => Some(s),
            _ => None,
        }
    }
}

impl fmt::Display for IlpResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IlpResult::Optimal(s) => write!(f, "OPTIMAL: {}", s),
            IlpResult::Feasible(s) => write!(f, "FEASIBLE: {}", s),
            IlpResult::Infeasible => write!(f, "INFEASIBLE"),
            IlpResult::Unbounded => write!(f, "UNBOUNDED"),
            IlpResult::Timeout => write!(f, "TIMEOUT"),
        }
    }
}

// ─── Pareto Result ──────────────────────────────────────────────────────────

/// A single point on the Pareto frontier.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParetoPoint {
    pub costs: Vec<f64>,
    pub solution: HashMap<String, f64>,
}

impl ParetoPoint {
    /// Returns true if `self` dominates `other` (weakly in all, strictly in ≥1).
    pub fn dominates(&self, other: &ParetoPoint) -> bool {
        if self.costs.len() != other.costs.len() {
            return false;
        }
        let all_le = self.costs.iter().zip(&other.costs).all(|(a, b)| a <= b);
        let some_lt = self.costs.iter().zip(&other.costs).any(|(a, b)| a < b);
        all_le && some_lt
    }
}

impl fmt::Display for ParetoPoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(")?;
        for (i, c) in self.costs.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{:.4}", c)?;
        }
        write!(f, ")")
    }
}

/// The Pareto frontier: a set of non-dominated points.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParetoFrontier {
    pub points: Vec<ParetoPoint>,
    pub epsilon: f64,
}

impl ParetoFrontier {
    pub fn new(epsilon: f64) -> Self {
        Self {
            points: Vec::new(),
            epsilon,
        }
    }

    /// Add a point if it is not dominated by any existing point.
    /// Also remove any existing points dominated by the new one.
    pub fn add_point(&mut self, point: ParetoPoint) -> bool {
        if self.points.iter().any(|p| p.dominates(&point)) {
            return false;
        }
        self.points.retain(|p| !point.dominates(p));
        self.points.push(point);
        true
    }

    pub fn size(&self) -> usize {
        self.points.len()
    }
}

impl fmt::Display for ParetoFrontier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ParetoFrontier[{} points, ε={:.4}]", self.points.len(), self.epsilon)
    }
}

/// Result of Pareto frontier enumeration.
#[derive(Debug, Clone)]
pub enum ParetoResult {
    Complete(ParetoFrontier),
    Partial(ParetoFrontier),
    Infeasible,
    Timeout,
}

impl ParetoResult {
    pub fn frontier(&self) -> Option<&ParetoFrontier> {
        match self {
            ParetoResult::Complete(f) | ParetoResult::Partial(f) => Some(f),
            _ => None,
        }
    }
}

impl fmt::Display for ParetoResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParetoResult::Complete(pf) => write!(f, "COMPLETE: {}", pf),
            ParetoResult::Partial(pf) => write!(f, "PARTIAL: {}", pf),
            ParetoResult::Infeasible => write!(f, "INFEASIBLE"),
            ParetoResult::Timeout => write!(f, "TIMEOUT"),
        }
    }
}

// ─── Solver Statistics ──────────────────────────────────────────────────────

/// Statistics collected during a solver run.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SolverStatistics {
    pub decisions: u64,
    pub conflicts: u64,
    pub propagations: u64,
    pub restarts: u64,
    pub learned_clauses: u64,
    pub time_ms: u64,
}

impl SolverStatistics {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn elapsed(&self) -> Duration {
        Duration::from_millis(self.time_ms)
    }
}

impl fmt::Display for SolverStatistics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Stats{{decisions={}, conflicts={}, props={}, restarts={}, learned={}, time={}ms}}",
            self.decisions, self.conflicts, self.propagations,
            self.restarts, self.learned_clauses, self.time_ms
        )
    }
}

// ─── Unified Result ─────────────────────────────────────────────────────────

/// Unified solver result across all solver backends.
#[derive(Debug, Clone)]
pub enum SolverResult {
    Sat(SatResult),
    Smt(SmtResult),
    MaxSmt(MaxSmtResult),
    Ilp(IlpResult),
    Pareto(ParetoResult),
}

impl fmt::Display for SolverResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SolverResult::Sat(r) => write!(f, "SAT: {}", r),
            SolverResult::Smt(r) => write!(f, "SMT: {}", r),
            SolverResult::MaxSmt(r) => write!(f, "MaxSMT: {}", r),
            SolverResult::Ilp(r) => write!(f, "ILP: {}", r),
            SolverResult::Pareto(r) => write!(f, "Pareto: {}", r),
        }
    }
}

// ─── MUS Result ─────────────────────────────────────────────────────────────

/// A Minimal Unsatisfiable Subset mapped back to regulatory obligations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinimalUnsatisfiableSubset {
    pub constraint_indices: Vec<usize>,
    pub constraint_ids: Vec<String>,
    pub obligations: Vec<String>,
}

impl fmt::Display for MinimalUnsatisfiableSubset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MUS[{} constraints]: {:?}", self.constraint_indices.len(), self.constraint_ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_literal_operations() {
        let lit = make_lit(5, true);
        assert_eq!(lit, 5);
        assert_eq!(lit_var(lit), 5);
        assert!(lit_sign(lit));
        assert_eq!(lit_neg(lit), -5);

        let neg = make_lit(5, false);
        assert_eq!(neg, -5);
        assert_eq!(lit_var(neg), 5);
        assert!(!lit_sign(neg));
    }

    #[test]
    fn test_assignment() {
        let mut a = Assignment::new();
        a.set(1, true);
        a.set(2, false);
        assert_eq!(a.get(1), Some(true));
        assert_eq!(a.eval_lit(1), Some(true));
        assert_eq!(a.eval_lit(-1), Some(false));
        assert_eq!(a.eval_lit(-2), Some(true));
        assert_eq!(a.eval_lit(3), None);
    }

    #[test]
    fn test_pareto_dominance() {
        let p1 = ParetoPoint {
            costs: vec![1.0, 1.0],
            solution: HashMap::new(),
        };
        let p2 = ParetoPoint {
            costs: vec![3.0, 2.0],
            solution: HashMap::new(),
        };
        let p3 = ParetoPoint {
            costs: vec![2.0, 3.0],
            solution: HashMap::new(),
        };
        assert!(p1.dominates(&p2));
        assert!(!p2.dominates(&p1));
        assert!(p1.dominates(&p3));
        assert!(!p3.dominates(&p1));
        // p2 and p3 don't dominate each other (tradeoff)
        assert!(!p2.dominates(&p3));
        assert!(!p3.dominates(&p2));
    }

    #[test]
    fn test_pareto_frontier_add() {
        let mut frontier = ParetoFrontier::new(0.01);
        let p1 = ParetoPoint {
            costs: vec![1.0, 3.0],
            solution: HashMap::new(),
        };
        let p2 = ParetoPoint {
            costs: vec![2.0, 1.0],
            solution: HashMap::new(),
        };
        let p3 = ParetoPoint {
            costs: vec![3.0, 4.0],
            solution: HashMap::new(),
        };
        assert!(frontier.add_point(p1));
        assert!(frontier.add_point(p2));
        // p3 is dominated by p2 (in one dim) but not all - actually 3>2 and 4>1, so p2 dominates p3
        assert!(!frontier.add_point(p3));
        assert_eq!(frontier.size(), 2);
    }

    #[test]
    fn test_model() {
        let mut m = Model::new();
        m.set_bool("x", true);
        m.set_int("y", 42);
        m.set_real("z", 3.14);
        assert_eq!(m.get_bool("x"), Some(true));
        assert_eq!(m.get_int("y"), Some(42));
        assert_eq!(m.get_real("z"), Some(3.14));
        assert_eq!(m.get_bool("w"), None);
    }

    #[test]
    fn test_solver_statistics_display() {
        let stats = SolverStatistics {
            decisions: 100,
            conflicts: 20,
            propagations: 500,
            restarts: 3,
            learned_clauses: 15,
            time_ms: 42,
        };
        let s = format!("{}", stats);
        assert!(s.contains("decisions=100"));
        assert!(s.contains("conflicts=20"));
    }

    #[test]
    fn test_ilp_solution() {
        let sol = IlpSolution {
            values: vec![("x".into(), 1.0), ("y".into(), 2.0)].into_iter().collect(),
            objective_value: 5.0,
        };
        assert_eq!(sol.get("x"), Some(1.0));
        assert_eq!(sol.get("z"), None);
    }
}
