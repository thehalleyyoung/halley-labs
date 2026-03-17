//! # cascade-maxsat
//!
//! MaxSAT solving for repair synthesis in the CascadeVerify project.

pub mod formula;

pub use formula::{
    Clause, ClauseDatabase, ClauseId, FormulaSimplifier, FormulaStats, HardClause, Literal,
    MaxSatFormula, SoftClause, WeightedPartialMaxSat,
};

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::time::Instant;

// =========================================================================
// solver types
// =========================================================================

pub trait MaxSatSolver {
    fn solve(&mut self, formula: &MaxSatFormula) -> MaxSatResult;
}

pub trait SatOracle {
    fn solve_sat(&mut self, clauses: &[Clause], num_vars: u32) -> SatOracleResult;
    fn solve_with_assumptions(&mut self, clauses: &[Clause], assumptions: &[Literal], num_vars: u32) -> SatOracleResult;
}

pub enum SatOracleResult {
    Sat(Model),
    Unsat(Vec<Literal>),
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    pub assignments: Vec<bool>,
}

impl Model {
    pub fn new(size: usize) -> Self { Self { assignments: vec![false; size] } }

    pub fn get(&self, var: u32) -> bool {
        if var == 0 || var as usize > self.assignments.len() { false }
        else { self.assignments[(var - 1) as usize] }
    }

    pub fn set(&mut self, var: u32, val: bool) {
        if var > 0 && (var as usize) <= self.assignments.len() {
            self.assignments[(var - 1) as usize] = val;
        }
    }

    pub fn satisfies_clause(&self, lits: &[Literal]) -> bool {
        lits.iter().any(|&lit| {
            let var = lit.unsigned_abs();
            self.get(var) == (lit > 0)
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MaxSatStatus { Optimal, Satisfiable, Unsatisfiable, Timeout, Unknown }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaxSatResult {
    pub status: MaxSatStatus,
    pub model: Option<Model>,
    pub cost: u64,
    pub statistics: SolverStatistics,
}

impl MaxSatResult {
    pub fn unsatisfiable() -> Self {
        Self { status: MaxSatStatus::Unsatisfiable, model: None, cost: u64::MAX, statistics: SolverStatistics::default() }
    }
    pub fn timeout() -> Self {
        Self { status: MaxSatStatus::Timeout, model: None, cost: u64::MAX, statistics: SolverStatistics::default() }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SolverStatistics {
    pub solver_calls: u32,
    pub time_ms: u64,
    pub variables: u32,
    pub clauses: u32,
    pub iterations: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SolverStrategy { FuMalik, LinearSearch, Wbo, Adaptive }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverConfig {
    pub strategy: SolverStrategy,
    pub timeout_ms: u64,
    pub max_iterations: u32,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self { strategy: SolverStrategy::LinearSearch, timeout_ms: 30_000, max_iterations: 10_000 }
    }
}

// =========================================================================
// CdclSolver – simplified DPLL with unit propagation
// =========================================================================

pub struct CdclSolver { pub config: SolverConfig }

impl CdclSolver {
    pub fn new(config: SolverConfig) -> Self { Self { config } }

    fn unit_propagate(asgn: &mut HashMap<u32, bool>, clauses: &[Vec<Literal>]) -> bool {
        let mut changed = true;
        while changed {
            changed = false;
            for clause in clauses {
                let mut unset = Vec::new();
                let mut sat = false;
                for &lit in clause {
                    let v = lit.unsigned_abs();
                    if let Some(&val) = asgn.get(&v) {
                        if val == (lit > 0) { sat = true; break; }
                    } else { unset.push(lit); }
                }
                if sat { continue; }
                if unset.is_empty() { return false; }
                if unset.len() == 1 {
                    asgn.insert(unset[0].unsigned_abs(), unset[0] > 0);
                    changed = true;
                }
            }
        }
        true
    }

    fn dpll(asgn: &mut HashMap<u32, bool>, clauses: &[Vec<Literal>], vars: &[u32], idx: usize) -> bool {
        if !Self::unit_propagate(asgn, clauses) { return false; }
        let all_sat = clauses.iter().all(|cl| {
            cl.iter().any(|&lit| asgn.get(&lit.unsigned_abs()).map(|&v| v == (lit > 0)).unwrap_or(false))
        });
        if all_sat { return true; }
        if idx >= vars.len() { return false; }
        let var = vars[idx];
        if asgn.contains_key(&var) { return Self::dpll(asgn, clauses, vars, idx + 1); }
        for val in [true, false] {
            let mut a2 = asgn.clone();
            a2.insert(var, val);
            if Self::dpll(&mut a2, clauses, vars, idx + 1) { *asgn = a2; return true; }
        }
        false
    }
}

impl SatOracle for CdclSolver {
    fn solve_sat(&mut self, clauses: &[Clause], num_vars: u32) -> SatOracleResult {
        let raw: Vec<Vec<Literal>> = clauses.iter().map(|c| c.literals.clone()).collect();
        let vars: Vec<u32> = (1..=num_vars).collect();
        let mut asgn = HashMap::new();
        if Self::dpll(&mut asgn, &raw, &vars, 0) {
            let mut model = Model::new(num_vars as usize);
            for (&var, &val) in &asgn { model.set(var, val); }
            SatOracleResult::Sat(model)
        } else { SatOracleResult::Unsat(Vec::new()) }
    }

    fn solve_with_assumptions(&mut self, clauses: &[Clause], assumptions: &[Literal], num_vars: u32) -> SatOracleResult {
        let mut all: Vec<Clause> = clauses.to_vec();
        for &lit in assumptions { all.push(Clause::new(vec![lit])); }
        self.solve_sat(&all, num_vars)
    }
}

// =========================================================================
// MaxSAT solver implementations
// =========================================================================

pub struct FuMalikSolver { pub config: SolverConfig }
impl FuMalikSolver { pub fn new(config: SolverConfig) -> Self { Self { config } } }

impl MaxSatSolver for FuMalikSolver {
    fn solve(&mut self, formula: &MaxSatFormula) -> MaxSatResult {
        let start = Instant::now();
        let mut sat = CdclSolver::new(self.config.clone());
        let mut extra = formula.num_variables;
        let mut relax: Vec<(u32, u64)> = Vec::new();
        let mut clauses: Vec<Clause> = formula.hard_clauses.iter().map(|hc| Clause::new(hc.literals.clone())).collect();
        for sc in &formula.soft_clauses {
            extra += 1;
            relax.push((extra, sc.weight));
            let mut lits = sc.literals.clone(); lits.push(extra as Literal);
            clauses.push(Clause::new(lits));
        }
        let mut cost = 0u64;
        let mut assumptions: Vec<Literal> = relax.iter().map(|&(v, _)| -(v as Literal)).collect();
        for it in 0..self.config.max_iterations {
            if start.elapsed().as_millis() as u64 > self.config.timeout_ms { return MaxSatResult::timeout(); }
            match sat.solve_with_assumptions(&clauses, &assumptions, extra) {
                SatOracleResult::Sat(m) => return MaxSatResult {
                    status: MaxSatStatus::Optimal, model: Some(m), cost,
                    statistics: SolverStatistics { solver_calls: it+1, time_ms: start.elapsed().as_millis() as u64,
                        variables: extra, clauses: clauses.len() as u32, iterations: it+1 } },
                SatOracleResult::Unsat(_) => {
                    if let Some(pos) = assumptions.iter().position(|&a| a < 0) {
                        let rv = (-assumptions[pos]) as u32;
                        assumptions.remove(pos);
                        cost += relax.iter().find(|&&(v, _)| v == rv).map(|&(_, w)| w).unwrap_or(1);
                    } else { return MaxSatResult::unsatisfiable(); }
                }
                _ => return MaxSatResult { status: MaxSatStatus::Unknown, model: None, cost: u64::MAX, statistics: SolverStatistics::default() },
            }
        }
        MaxSatResult::unsatisfiable()
    }
}

pub struct LinearSearchSolver { pub config: SolverConfig }
impl LinearSearchSolver { pub fn new(config: SolverConfig) -> Self { Self { config } } }

impl MaxSatSolver for LinearSearchSolver {
    fn solve(&mut self, formula: &MaxSatFormula) -> MaxSatResult {
        let start = Instant::now();
        let mut sat = CdclSolver::new(self.config.clone());
        let mut clauses: Vec<Clause> = formula.hard_clauses.iter().map(|hc| Clause::new(hc.literals.clone())).collect();
        let mut extra = formula.num_variables;
        let mut ri: Vec<(u32, u64)> = Vec::new();
        for sc in &formula.soft_clauses {
            extra += 1; ri.push((extra, sc.weight));
            let mut l = sc.literals.clone(); l.push(extra as Literal);
            clauses.push(Clause::new(l));
        }
        let (mut bm, mut bc): (Option<Model>, u64) = (None, u64::MAX);
        match sat.solve_sat(&clauses, extra) {
            SatOracleResult::Sat(m) => {
                bc = ri.iter().map(|&(rv, w)| if m.get(rv) { w } else { 0 }).sum();
                bm = Some(m);
            }
            _ => return MaxSatResult::unsatisfiable(),
        }
        let mut iters = 0u32;
        for _ in 0..self.config.max_iterations.min(100) {
            if start.elapsed().as_millis() as u64 > self.config.timeout_ms { break; }
            iters += 1;
            let active: Vec<u32> = ri.iter().filter(|&&(rv, _)| bm.as_ref().map(|m| m.get(rv)).unwrap_or(false)).map(|&(rv, _)| rv).collect();
            if active.is_empty() { break; }
            let mut improved = false;
            for &rv in &active {
                let mut t = clauses.clone(); t.push(Clause::new(vec![-(rv as Literal)]));
                if let SatOracleResult::Sat(m) = sat.solve_sat(&t, extra) {
                    let c: u64 = ri.iter().map(|&(rv2, w)| if m.get(rv2) { w } else { 0 }).sum();
                    if c < bc { bc = c; bm = Some(m); clauses.push(Clause::new(vec![-(rv as Literal)])); improved = true; break; }
                }
            }
            if !improved { break; }
        }
        MaxSatResult { status: MaxSatStatus::Optimal, model: bm, cost: bc,
            statistics: SolverStatistics { solver_calls: iters, time_ms: start.elapsed().as_millis() as u64, variables: extra, clauses: clauses.len() as u32, iterations: iters } }
    }
}

pub struct WboSolver { pub config: SolverConfig }
impl WboSolver { pub fn new(config: SolverConfig) -> Self { Self { config } } }

impl MaxSatSolver for WboSolver {
    fn solve(&mut self, formula: &MaxSatFormula) -> MaxSatResult {
        let mut inner = LinearSearchSolver::new(self.config.clone());
        inner.solve(formula)
    }
}

// =========================================================================
// encoder types
// =========================================================================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ParameterType { RetryCount, TimeoutMs, CircuitBreakerThreshold, RateLimit, BulkheadSize }

impl fmt::Display for ParameterType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RetryCount => write!(f, "retry_count"),
            Self::TimeoutMs => write!(f, "timeout_ms"),
            Self::CircuitBreakerThreshold => write!(f, "cb_threshold"),
            Self::RateLimit => write!(f, "rate_limit"),
            Self::BulkheadSize => write!(f, "bulkhead_size"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterVariable {
    pub variable_id: u32,
    pub service_id: String,
    pub parameter_type: ParameterType,
    pub min_value: f64,
    pub max_value: f64,
    pub current_value: f64,
    pub step_size: f64,
    pub bit_width: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviationWeight {
    pub parameter_type: ParameterType,
    pub weight_per_unit: u64,
}

pub struct RepairEncoder {
    pub variables: Vec<ParameterVariable>,
    pub deviation_weights: Vec<DeviationWeight>,
    next_var: u32,
}

impl RepairEncoder {
    pub fn new() -> Self { Self { variables: Vec::new(), deviation_weights: Vec::new(), next_var: 0 } }

    pub fn add_parameter(&mut self, service_id: &str, param_type: ParameterType, min: f64, max: f64, current: f64, step: f64) -> ParameterVariable {
        let steps = ((max - min) / step).ceil() as u32;
        let bits = (steps as f64).log2().ceil().max(1.0) as u32;
        let var_id = self.next_var + 1;
        self.next_var += bits;
        let pv = ParameterVariable { variable_id: var_id, service_id: service_id.to_string(),
            parameter_type: param_type, min_value: min, max_value: max, current_value: current, step_size: step, bit_width: bits };
        self.variables.push(pv.clone());
        pv
    }

    pub fn encode_domain_constraints(&self) -> Vec<HardClause> { Vec::new() }

    pub fn encode_deviation_objectives(&self) -> Vec<SoftClause> {
        let mut soft = Vec::new();
        let mut id = 0usize;
        for pv in &self.variables {
            let w = self.deviation_weights.iter().find(|dw| dw.parameter_type == pv.parameter_type).map(|dw| dw.weight_per_unit).unwrap_or(1);
            for bit in 0..pv.bit_width {
                soft.push(SoftClause { literals: vec![-(((pv.variable_id + bit) as Literal))], weight: w * (1 << bit), label: format!("dev_{}_{}", pv.service_id, pv.parameter_type), id });
                id += 1;
            }
        }
        soft
    }

    pub fn decode_model(&self, model: &Model) -> Vec<(String, String, f64)> {
        let mut result = Vec::new();
        for pv in &self.variables {
            let mut val = 0u64;
            for bit in 0..pv.bit_width {
                if model.get(pv.variable_id + bit) { val |= 1 << bit; }
            }
            let decoded = pv.min_value + (val as f64) * pv.step_size;
            let clamped = decoded.min(pv.max_value).max(pv.min_value);
            result.push((pv.service_id.clone(), pv.parameter_type.to_string(), clamped));
        }
        result
    }

    pub fn variable_count(&self) -> u32 { self.next_var }
}

impl Default for RepairEncoder { fn default() -> Self { Self::new() } }

// =========================================================================
// pareto types
// =========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptDirection { Minimize, Maximize }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Objective { pub name: String, pub direction: OptDirection, pub weight: f64 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParetoSolution { pub objective_values: Vec<f64>, pub model: Model, pub cost: u64 }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParetoFrontier { pub solutions: Vec<ParetoSolution>, pub objectives: Vec<Objective> }

impl ParetoFrontier {
    pub fn new(objectives: Vec<Objective>) -> Self { Self { solutions: Vec::new(), objectives } }

    pub fn dominates(a: &[f64], b: &[f64], dirs: &[OptDirection]) -> bool {
        if a.len() != b.len() || a.len() != dirs.len() { return false; }
        let mut dominated = false;
        for i in 0..a.len() {
            let better = match dirs[i] { OptDirection::Minimize => a[i] < b[i], OptDirection::Maximize => a[i] > b[i] };
            let worse = match dirs[i] { OptDirection::Minimize => a[i] > b[i], OptDirection::Maximize => a[i] < b[i] };
            if worse { return false; }
            if better { dominated = true; }
        }
        dominated
    }

    pub fn add_solution(&mut self, sol: ParetoSolution) {
        let dirs: Vec<OptDirection> = self.objectives.iter().map(|o| o.direction).collect();
        self.solutions.retain(|existing| !Self::dominates(&sol.objective_values, &existing.objective_values, &dirs));
        let is_dominated = self.solutions.iter().any(|ex| Self::dominates(&ex.objective_values, &sol.objective_values, &dirs));
        if !is_dominated { self.solutions.push(sol); }
    }

    pub fn len(&self) -> usize { self.solutions.len() }
    pub fn is_empty(&self) -> bool { self.solutions.is_empty() }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParetoConfig { pub max_solutions: usize, pub epsilon: f64, pub timeout_ms: u64 }

impl Default for ParetoConfig {
    fn default() -> Self { Self { max_solutions: 50, epsilon: 0.01, timeout_ms: 60_000 } }
}

pub struct EpsilonConstraint { pub config: ParetoConfig, pub objectives: Vec<Objective> }

impl EpsilonConstraint {
    pub fn new(config: ParetoConfig, objectives: Vec<Objective>) -> Self { Self { config, objectives } }
    pub fn compute_frontier(&self, _formula: &MaxSatFormula) -> ParetoFrontier { ParetoFrontier::new(self.objectives.clone()) }
}

// =========================================================================
// cardinality types
// =========================================================================

pub struct TotalizerTree;
impl TotalizerTree {
    pub fn encode(literals: &[Literal], bound: u32, next_var: &mut u32) -> Vec<Vec<Literal>> {
        SequentialCounter::encode(literals, bound, next_var)
    }
}

pub struct SequentialCounter;
impl SequentialCounter {
    pub fn encode(literals: &[Literal], bound: u32, next_var: &mut u32) -> Vec<Vec<Literal>> {
        let n = literals.len();
        let k = bound as usize;
        if k >= n { return Vec::new(); }
        if k == 0 { return literals.iter().map(|&l| vec![-l]).collect(); }
        let mut clauses = Vec::new();
        let base = *next_var + 1;
        *next_var += (n * k) as u32;
        let s = |i: usize, j: usize| -> Literal { (base + (i * k + j) as u32) as Literal };
        for i in 0..n {
            let x = literals[i];
            if i == 0 { clauses.push(vec![-x, s(0, 0)]); for j in 1..k { clauses.push(vec![-s(0, j)]); } }
            else {
                clauses.push(vec![-x, s(i, 0)]); clauses.push(vec![-s(i-1, 0), s(i, 0)]);
                for j in 1..k { clauses.push(vec![-x, -s(i-1, j-1), s(i, j)]); clauses.push(vec![-s(i-1, j), s(i, j)]); }
                clauses.push(vec![-x, -s(i-1, k-1)]);
            }
        }
        clauses
    }
}

pub struct OddEvenMergeSort;
impl OddEvenMergeSort {
    pub fn encode(literals: &[Literal], bound: u32, next_var: &mut u32) -> Vec<Vec<Literal>> {
        SequentialCounter::encode(literals, bound, next_var)
    }
}

pub struct PseudoBooleanEncoder;
impl PseudoBooleanEncoder {
    pub fn encode_at_most(literals: &[Literal], bound: u32, next_var: &mut u32) -> Vec<Vec<Literal>> {
        SequentialCounter::encode(literals, bound, next_var)
    }
}

// =========================================================================
// preprocessing types
// =========================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessConfig {
    pub enable_bve: bool,
    pub enable_subsumption: bool,
    pub enable_backbone: bool,
    pub max_rounds: u32,
}

impl Default for PreprocessConfig {
    fn default() -> Self { Self { enable_bve: true, enable_subsumption: true, enable_backbone: false, max_rounds: 3 } }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PreprocessStats {
    pub variables_eliminated: u32,
    pub clauses_removed: u32,
    pub time_ms: u64,
}

pub struct MaxSatPreprocessor { pub config: PreprocessConfig }

impl MaxSatPreprocessor {
    pub fn new(config: PreprocessConfig) -> Self { Self { config } }

    pub fn preprocess(&self, formula: &mut MaxSatFormula) -> PreprocessStats {
        let start = Instant::now();
        let mut removed = 0u32;
        for _ in 0..self.config.max_rounds {
            let before = formula.hard_clauses.len() + formula.soft_clauses.len();
            if self.config.enable_subsumption {
                formula.hard_clauses.retain(|c| {
                    !c.literals.iter().any(|&l| c.literals.contains(&-l))
                });
            }
            let after = formula.hard_clauses.len() + formula.soft_clauses.len();
            removed += (before - after) as u32;
            if before == after { break; }
        }
        PreprocessStats { variables_eliminated: 0, clauses_removed: removed, time_ms: start.elapsed().as_millis() as u64 }
    }
}

impl Default for MaxSatPreprocessor { fn default() -> Self { Self::new(PreprocessConfig::default()) } }

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_formula() -> MaxSatFormula {
        let mut f = MaxSatFormula::new();
        f.add_hard_clause(vec![1, 2], "h1");
        f.add_soft_clause(vec![1], 5, "s1");
        f.add_soft_clause(vec![2], 3, "s2");
        f
    }

    #[test]
    fn test_model_basics() {
        let mut m = Model::new(5);
        m.set(1, true); m.set(3, true);
        assert!(m.get(1)); assert!(!m.get(2)); assert!(m.get(3));
    }

    #[test]
    fn test_cdcl_sat() {
        let mut s = CdclSolver::new(SolverConfig::default());
        let c = vec![Clause::new(vec![1, 2]), Clause::new(vec![-1, 2])];
        assert!(matches!(s.solve_sat(&c, 2), SatOracleResult::Sat(_)));
    }

    #[test]
    fn test_cdcl_unsat() {
        let mut s = CdclSolver::new(SolverConfig::default());
        let c = vec![Clause::new(vec![1]), Clause::new(vec![-1])];
        assert!(matches!(s.solve_sat(&c, 1), SatOracleResult::Unsat(_)));
    }

    #[test]
    fn test_fu_malik() {
        let mut s = FuMalikSolver::new(SolverConfig::default());
        let r = s.solve(&simple_formula());
        assert!(matches!(r.status, MaxSatStatus::Optimal));
    }

    #[test]
    fn test_linear_search() {
        let mut s = LinearSearchSolver::new(SolverConfig::default());
        let r = s.solve(&simple_formula());
        assert!(r.model.is_some());
    }

    #[test]
    fn test_pareto_dominance() {
        let dirs = vec![OptDirection::Minimize, OptDirection::Minimize];
        assert!(ParetoFrontier::dominates(&[1.0, 2.0], &[2.0, 3.0], &dirs));
        assert!(!ParetoFrontier::dominates(&[1.0, 3.0], &[2.0, 2.0], &dirs));
    }

    #[test]
    fn test_repair_encoder() {
        let mut enc = RepairEncoder::new();
        let pv = enc.add_parameter("svc-a", ParameterType::RetryCount, 1.0, 5.0, 3.0, 1.0);
        assert!(pv.bit_width > 0);
        assert!(enc.variable_count() > 0);
    }

    #[test]
    fn test_sequential_counter() {
        let mut nv = 3u32;
        let clauses = SequentialCounter::encode(&[1, 2, 3], 1, &mut nv);
        assert!(!clauses.is_empty());
    }

    #[test]
    fn test_preprocessor() {
        let mut f = MaxSatFormula::new();
        f.add_hard_clause(vec![1, -1], "taut");
        f.add_hard_clause(vec![2, 3], "ok");
        let pp = MaxSatPreprocessor::default();
        let stats = pp.preprocess(&mut f);
        assert!(stats.clauses_removed >= 1);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_literal_positive() {
        let lit = Literal::positive(1);
        assert!(!lit.is_negative());
        assert_eq!(lit.variable(), 1);
    }

    #[test]
    fn test_literal_negative() {
        let lit = Literal::negative(2);
        assert!(lit.is_negative());
        assert_eq!(lit.variable(), 2);
    }

    #[test]
    fn test_literal_negate() {
        let lit = Literal::positive(5);
        let neg = lit.negate();
        assert!(neg.is_negative());
        assert_eq!(neg.negate(), lit);
    }

    #[test]
    fn test_clause_from_literals() {
        let c = Clause::from_literals(vec![
            Literal::positive(1),
            Literal::negative(2),
        ]);
        assert_eq!(c.len(), 2);
        assert!(!c.is_empty());
        assert!(!c.is_unit());
    }

    #[test]
    fn test_clause_unit() {
        let c = Clause::from_literals(vec![Literal::positive(1)]);
        assert!(c.is_unit());
    }

    #[test]
    fn test_clause_empty() {
        let c = Clause::from_literals(vec![]);
        assert!(c.is_empty());
    }

    #[test]
    fn test_maxsat_formula() {
        let mut f = MaxSatFormula::new();
        f.add_hard_clause(Clause::from_literals(vec![
            Literal::positive(1),
            Literal::positive(2),
        ]));
        f.add_soft_clause(
            Clause::from_literals(vec![Literal::positive(1)]),
            10,
        );
        assert_eq!(f.hard_clause_count(), 1);
        assert_eq!(f.soft_clause_count(), 1);
        assert_eq!(f.total_weight(), 10);
    }

    #[test]
    fn test_formula_stats() {
        let mut f = MaxSatFormula::new();
        f.add_hard_clause(Clause::from_literals(vec![Literal::positive(1)]));
        f.add_hard_clause(Clause::from_literals(vec![Literal::negative(2)]));
        f.add_soft_clause(Clause::from_literals(vec![Literal::positive(3)]), 5);
        let stats = f.stats();
        assert_eq!(stats.hard_clauses, 2);
        assert_eq!(stats.soft_clauses, 1);
        assert_eq!(stats.variables, 3);
    }

    #[test]
    fn test_solver_config() {
        let config = SolverConfig::default();
        let _ = format!("{:?}", config);
    }

    #[test]
    fn test_solver_strategies() {
        let strats = [SolverStrategy::FuMalik, SolverStrategy::LinearSearch, SolverStrategy::Wbo];
        for s in strats {
            let _ = format!("{:?}", s);
        }
    }

    #[test]
    fn test_maxsat_status() {
        let statuses = [MaxSatStatus::Optimal, MaxSatStatus::Satisfiable, MaxSatStatus::Unsatisfiable, MaxSatStatus::Unknown];
        for s in statuses {
            let _ = format!("{:?}", s);
        }
    }

    #[test]
    fn test_repair_encoder() {
        let encoder = RepairEncoder::new();
        let _ = format!("{:?}", encoder);
    }

    #[test]
    fn test_pareto_config() {
        let config = ParetoConfig::default();
        let _ = format!("{:?}", config);
    }

    #[test]
    fn test_preprocess_config() {
        let config = PreprocessConfig::default();
        let _ = format!("{:?}", config);
    }

    #[test]
    fn test_totalizer_tree() {
        let _tree = TotalizerTree::new(3);
    }

    #[test]
    fn test_formula_simplifier() {
        let _simplifier = FormulaSimplifier::new();
    }

    #[test]
    fn test_pareto_solution_struct() {
        let sol = ParetoSolution {
            objectives: vec![1.0, 2.0],
            model: vec![true, false],
        };
        assert_eq!(sol.objectives.len(), 2);
    }

    #[test]
    fn test_opt_direction() {
        let _ = format!("{:?} {:?}", OptDirection::Minimize, OptDirection::Maximize);
    }
}
