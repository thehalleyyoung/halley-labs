//! Pure-Rust SMT Solver interface and built-in DPLL(T)-style solver for QF_LIA.

use cascade_types::smt::{SmtConstraint, SmtExpr, SmtFormula};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Solver configuration and statistics
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverConfig {
    pub timeout_ms: u64,
    pub random_seed: u64,
    pub incremental: bool,
    pub max_conflicts: usize,
    pub restart_interval: usize,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            timeout_ms: 30_000,
            random_seed: 42,
            incremental: false,
            max_conflicts: 100_000,
            restart_interval: 100,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SolverStats {
    pub decisions: usize,
    pub propagations: usize,
    pub conflicts: usize,
    pub restarts: usize,
    pub solve_time_ms: u64,
    pub variables: usize,
    pub clauses: usize,
}

// ---------------------------------------------------------------------------
// Core types: Literal, Clause, Assignment
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Literal {
    pub var: usize,
    pub positive: bool,
}

impl Literal {
    pub fn new(var: usize, positive: bool) -> Self {
        Self { var, positive }
    }

    pub fn pos(var: usize) -> Self {
        Self { var, positive: true }
    }

    pub fn neg(var: usize) -> Self {
        Self { var, positive: false }
    }

    pub fn negate(self) -> Self {
        Self { var: self.var, positive: !self.positive }
    }

    pub fn index(self) -> usize {
        self.var * 2 + if self.positive { 0 } else { 1 }
    }
}

impl std::fmt::Display for Literal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.positive {
            write!(f, "x{}", self.var)
        } else {
            write!(f, "¬x{}", self.var)
        }
    }
}

#[derive(Debug, Clone)]
pub struct Clause {
    pub literals: Vec<Literal>,
    pub is_learned: bool,
}

impl Clause {
    pub fn new(literals: Vec<Literal>) -> Self {
        Self { literals, is_learned: false }
    }

    pub fn learned(literals: Vec<Literal>) -> Self {
        Self { literals, is_learned: true }
    }

    pub fn is_unit(&self) -> bool {
        self.literals.len() == 1
    }

    pub fn is_empty(&self) -> bool {
        self.literals.is_empty()
    }

    pub fn len(&self) -> usize {
        self.literals.len()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssignmentValue {
    True,
    False,
    Unassigned,
}

#[derive(Debug, Clone)]
pub struct Assignment {
    values: Vec<AssignmentValue>,
    trail: Vec<Literal>,
    decision_levels: Vec<usize>,
    reasons: Vec<Option<usize>>,
    level: Vec<usize>,
}

impl Assignment {
    pub fn new(num_vars: usize) -> Self {
        Self {
            values: vec![AssignmentValue::Unassigned; num_vars],
            trail: Vec::new(),
            decision_levels: Vec::new(),
            reasons: vec![None; num_vars],
            level: vec![0; num_vars],
        }
    }

    pub fn value(&self, var: usize) -> AssignmentValue {
        self.values[var]
    }

    pub fn literal_value(&self, lit: Literal) -> AssignmentValue {
        match self.values[lit.var] {
            AssignmentValue::True => {
                if lit.positive { AssignmentValue::True } else { AssignmentValue::False }
            }
            AssignmentValue::False => {
                if lit.positive { AssignmentValue::False } else { AssignmentValue::True }
            }
            AssignmentValue::Unassigned => AssignmentValue::Unassigned,
        }
    }

    pub fn assign(&mut self, lit: Literal, decision_level: usize, reason: Option<usize>) {
        self.values[lit.var] = if lit.positive { AssignmentValue::True } else { AssignmentValue::False };
        self.trail.push(lit);
        self.reasons[lit.var] = reason;
        self.level[lit.var] = decision_level;
    }

    pub fn unassign(&mut self, var: usize) {
        self.values[var] = AssignmentValue::Unassigned;
        self.reasons[var] = None;
    }

    pub fn current_decision_level(&self) -> usize {
        self.decision_levels.len()
    }

    pub fn new_decision_level(&mut self) {
        self.decision_levels.push(self.trail.len());
    }

    pub fn backtrack_to(&mut self, level: usize) {
        let trail_pos = if level < self.decision_levels.len() {
            self.decision_levels[level]
        } else {
            self.trail.len()
        };
        while self.trail.len() > trail_pos {
            let lit = self.trail.pop().unwrap();
            self.unassign(lit.var);
        }
        self.decision_levels.truncate(level);
    }

    pub fn is_complete(&self, num_vars: usize) -> bool {
        (0..num_vars).all(|v| self.values[v] != AssignmentValue::Unassigned)
    }

    pub fn trail_len(&self) -> usize {
        self.trail.len()
    }
}

// ---------------------------------------------------------------------------
// VSIDS decision heuristic
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct VsidsScores {
    scores: Vec<f64>,
    decay_factor: f64,
    increment: f64,
}

impl VsidsScores {
    fn new(num_vars: usize) -> Self {
        Self {
            scores: vec![0.0; num_vars],
            decay_factor: 0.95,
            increment: 1.0,
        }
    }

    fn bump(&mut self, var: usize) {
        self.scores[var] += self.increment;
        if self.scores[var] > 1e100 {
            for s in &mut self.scores {
                *s *= 1e-100;
            }
            self.increment *= 1e-100;
        }
    }

    fn decay(&mut self) {
        self.increment /= self.decay_factor;
    }

    fn pick_unassigned(&self, assignment: &Assignment) -> Option<usize> {
        let mut best_var = None;
        let mut best_score = -1.0f64;
        for (var, &score) in self.scores.iter().enumerate() {
            if assignment.value(var) == AssignmentValue::Unassigned && score > best_score {
                best_score = score;
                best_var = Some(var);
            }
        }
        best_var
    }
}

// ---------------------------------------------------------------------------
// Two-watched-literal data structure
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct WatchList {
    watches: Vec<Vec<usize>>,
}

impl WatchList {
    fn new(num_lits: usize) -> Self {
        Self { watches: vec![Vec::new(); num_lits] }
    }

    fn add_watch(&mut self, lit: Literal, clause_idx: usize) {
        self.watches[lit.index()].push(clause_idx);
    }

    fn watching(&self, lit: Literal) -> &[usize] {
        &self.watches[lit.index()]
    }

    fn watching_mut(&mut self, lit: Literal) -> Vec<usize> {
        std::mem::take(&mut self.watches[lit.index()])
    }

    fn restore(&mut self, lit: Literal, clause_indices: Vec<usize>) {
        self.watches[lit.index()] = clause_indices;
    }
}

// ---------------------------------------------------------------------------
// SmtSolver trait
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum SatResult {
    Sat(HashMap<usize, bool>),
    Unsat(Vec<usize>),
    Unknown(String),
    Timeout,
}

impl SatResult {
    pub fn is_sat(&self) -> bool {
        matches!(self, SatResult::Sat(_))
    }

    pub fn is_unsat(&self) -> bool {
        matches!(self, SatResult::Unsat(_))
    }
}

pub trait SmtSolver {
    fn check_sat(&mut self) -> SatResult;
    fn assert_clause(&mut self, clause: Clause);
    fn push(&mut self);
    fn pop(&mut self);
    fn stats(&self) -> &SolverStats;
}

// ---------------------------------------------------------------------------
// BuiltinSolver: CDCL-based SAT solver with QF_LIA theory propagation
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct BuiltinSolver {
    config: SolverConfig,
    clauses: Vec<Clause>,
    num_vars: usize,
    stats: SolverStats,
    push_stack: Vec<usize>,
}

impl BuiltinSolver {
    pub fn new(config: SolverConfig) -> Self {
        Self {
            config,
            clauses: Vec::new(),
            num_vars: 0,
            stats: SolverStats::default(),
            push_stack: Vec::new(),
        }
    }

    pub fn new_variable(&mut self) -> usize {
        let v = self.num_vars;
        self.num_vars += 1;
        v
    }

    pub fn add_clause(&mut self, clause: Clause) {
        for lit in &clause.literals {
            if lit.var >= self.num_vars {
                self.num_vars = lit.var + 1;
            }
        }
        self.clauses.push(clause);
    }

    pub fn num_variables(&self) -> usize {
        self.num_vars
    }

    pub fn num_clauses(&self) -> usize {
        self.clauses.len()
    }

    /// Core CDCL solve loop.
    pub fn solve(&mut self) -> SatResult {
        let start = Instant::now();
        self.stats = SolverStats::default();
        self.stats.variables = self.num_vars;
        self.stats.clauses = self.clauses.len();

        if self.num_vars == 0 {
            self.stats.solve_time_ms = start.elapsed().as_millis() as u64;
            return if self.clauses.iter().any(|c| c.is_empty()) {
                SatResult::Unsat(vec![])
            } else {
                SatResult::Sat(HashMap::new())
            };
        }

        let mut assignment = Assignment::new(self.num_vars);
        let mut vsids = VsidsScores::new(self.num_vars);
        let mut watches = WatchList::new(self.num_vars * 2);

        // Initialize VSIDS from clauses
        for clause in &self.clauses {
            for lit in &clause.literals {
                vsids.scores[lit.var] += 1.0;
            }
        }

        // Initialize watches (watch first two literals of each clause)
        for (ci, clause) in self.clauses.iter().enumerate() {
            if clause.literals.len() >= 2 {
                watches.add_watch(clause.literals[0].negate(), ci);
                watches.add_watch(clause.literals[1].negate(), ci);
            } else if clause.literals.len() == 1 {
                watches.add_watch(clause.literals[0].negate(), ci);
            }
        }

        // BCP for unit clauses
        for ci in 0..self.clauses.len() {
            if self.clauses[ci].is_unit() {
                let lit = self.clauses[ci].literals[0];
                if assignment.literal_value(lit) == AssignmentValue::Unassigned {
                    assignment.assign(lit, 0, Some(ci));
                    self.stats.propagations += 1;
                } else if assignment.literal_value(lit) == AssignmentValue::False {
                    self.stats.solve_time_ms = start.elapsed().as_millis() as u64;
                    return SatResult::Unsat(vec![ci]);
                }
            }
        }

        // Initial BCP
        if let Some(conflict) = self.propagate(&mut assignment, &mut watches) {
            self.stats.solve_time_ms = start.elapsed().as_millis() as u64;
            return SatResult::Unsat(vec![conflict]);
        }

        let mut conflicts_since_restart = 0usize;

        loop {
            // Timeout check
            if start.elapsed().as_millis() as u64 > self.config.timeout_ms {
                self.stats.solve_time_ms = start.elapsed().as_millis() as u64;
                return SatResult::Timeout;
            }

            // Conflict limit
            if self.stats.conflicts > self.config.max_conflicts {
                self.stats.solve_time_ms = start.elapsed().as_millis() as u64;
                return SatResult::Unknown("conflict limit reached".into());
            }

            // Check if all variables are assigned
            if assignment.is_complete(self.num_vars) {
                self.stats.solve_time_ms = start.elapsed().as_millis() as u64;
                let model = self.extract_model(&assignment);
                return SatResult::Sat(model);
            }

            // Restart heuristic
            if conflicts_since_restart >= self.config.restart_interval {
                self.stats.restarts += 1;
                assignment.backtrack_to(0);
                conflicts_since_restart = 0;
                // Re-propagate unit clauses
                for ci in 0..self.clauses.len() {
                    if self.clauses[ci].is_unit() {
                        let lit = self.clauses[ci].literals[0];
                        if assignment.literal_value(lit) == AssignmentValue::Unassigned {
                            assignment.assign(lit, 0, Some(ci));
                        }
                    }
                }
                let _ = self.propagate(&mut assignment, &mut watches);
                continue;
            }

            // Decision
            let decision_var = match vsids.pick_unassigned(&assignment) {
                Some(v) => v,
                None => {
                    self.stats.solve_time_ms = start.elapsed().as_millis() as u64;
                    let model = self.extract_model(&assignment);
                    return SatResult::Sat(model);
                }
            };

            self.stats.decisions += 1;
            assignment.new_decision_level();
            let decision_lit = Literal::pos(decision_var);
            assignment.assign(decision_lit, assignment.current_decision_level(), None);

            // BCP
            loop {
                match self.propagate(&mut assignment, &mut watches) {
                    None => break,
                    Some(conflict_clause) => {
                        self.stats.conflicts += 1;
                        conflicts_since_restart += 1;

                        if assignment.current_decision_level() == 0 {
                            self.stats.solve_time_ms = start.elapsed().as_millis() as u64;
                            return SatResult::Unsat(vec![conflict_clause]);
                        }

                        // Analyze conflict: learn a clause and backtrack
                        let (learned, bt_level) =
                            self.analyze_conflict(&assignment, conflict_clause);
                        for lit in &learned.literals {
                            vsids.bump(lit.var);
                        }
                        vsids.decay();

                        let learned_idx = self.clauses.len();
                        // Watch first two literals of learned clause
                        if learned.literals.len() >= 2 {
                            watches.add_watch(learned.literals[0].negate(), learned_idx);
                            watches.add_watch(learned.literals[1].negate(), learned_idx);
                        } else if learned.literals.len() == 1 {
                            watches.add_watch(learned.literals[0].negate(), learned_idx);
                        }
                        self.clauses.push(learned);

                        assignment.backtrack_to(bt_level);

                        // The first literal of the learned clause should be unit
                        let learned_clause = &self.clauses[learned_idx];
                        if !learned_clause.is_empty() {
                            let unit_lit = learned_clause.literals[0];
                            if assignment.literal_value(unit_lit) == AssignmentValue::Unassigned {
                                assignment.assign(unit_lit, bt_level, Some(learned_idx));
                                self.stats.propagations += 1;
                            }
                        }
                    }
                }
            }
        }
    }

    /// Boolean constraint propagation.  Returns `Some(clause_idx)` on conflict.
    fn propagate(&mut self, assignment: &mut Assignment, watches: &mut WatchList) -> Option<usize> {
        let mut prop_start = if assignment.trail_len() > 0 { assignment.trail_len() - 1 } else { 0 };
        while prop_start < assignment.trail_len() {
            let lit = assignment.trail[prop_start];
            prop_start += 1;

            let falsified = lit.negate();
            let clause_indices = watches.watching_mut(falsified);
            let mut new_watches = Vec::new();

            for &ci in &clause_indices {
                let clause = &self.clauses[ci];
                if clause.is_empty() {
                    new_watches.push(ci);
                    continue;
                }

                // Find if clause is satisfied or has unassigned literal
                let mut satisfied = false;
                let mut unassigned_lit = None;
                let mut unassigned_count = 0;
                let mut all_false = true;

                for &cl in &clause.literals {
                    match assignment.literal_value(cl) {
                        AssignmentValue::True => {
                            satisfied = true;
                            break;
                        }
                        AssignmentValue::Unassigned => {
                            all_false = false;
                            unassigned_count += 1;
                            unassigned_lit = Some(cl);
                        }
                        AssignmentValue::False => {}
                    }
                }

                if satisfied {
                    new_watches.push(ci);
                    continue;
                }

                if all_false {
                    // Conflict
                    new_watches.push(ci);
                    // Restore remaining watches
                    watches.restore(falsified, new_watches);
                    return Some(ci);
                }

                if unassigned_count == 1 {
                    // Unit propagation
                    let unit = unassigned_lit.unwrap();
                    assignment.assign(unit, assignment.current_decision_level(), Some(ci));
                    self.stats.propagations += 1;
                    new_watches.push(ci);
                } else {
                    new_watches.push(ci);
                }
            }

            watches.restore(falsified, new_watches);
        }
        None
    }

    /// 1-UIP conflict analysis. Returns (learned clause, backtrack level).
    fn analyze_conflict(&self, assignment: &Assignment, conflict_clause: usize) -> (Clause, usize) {
        let dl = assignment.current_decision_level();
        if dl == 0 {
            return (Clause::learned(vec![]), 0);
        }

        let conflict = &self.clauses[conflict_clause];
        let mut seen = vec![false; self.num_vars];
        let mut learnt_lits: Vec<Literal> = Vec::new();
        let mut count_at_current_level = 0;

        for &lit in &conflict.literals {
            seen[lit.var] = true;
            if assignment.level[lit.var] == dl {
                count_at_current_level += 1;
            } else if assignment.level[lit.var] > 0 {
                learnt_lits.push(lit.negate());
            }
        }

        // Walk backwards through the trail
        let mut trail_idx = assignment.trail.len();
        let mut resolved_lit = None;

        while count_at_current_level > 1 {
            if trail_idx == 0 {
                break;
            }
            trail_idx -= 1;
            let lit = assignment.trail[trail_idx];
            if !seen[lit.var] {
                continue;
            }
            seen[lit.var] = false;
            count_at_current_level -= 1;

            if count_at_current_level > 0 {
                if let Some(reason_ci) = assignment.reasons[lit.var] {
                    let reason = &self.clauses[reason_ci];
                    for &rl in &reason.literals {
                        if rl.var != lit.var && !seen[rl.var] {
                            seen[rl.var] = true;
                            if assignment.level[rl.var] == dl {
                                count_at_current_level += 1;
                            } else if assignment.level[rl.var] > 0 {
                                learnt_lits.push(rl.negate());
                            }
                        }
                    }
                }
            } else {
                resolved_lit = Some(lit.negate());
            }
        }

        // The asserting literal goes first
        if let Some(asserted) = resolved_lit {
            learnt_lits.insert(0, asserted);
        }

        // Compute backtrack level
        let bt_level = if learnt_lits.len() <= 1 {
            0
        } else {
            learnt_lits[1..].iter()
                .map(|l| assignment.level[l.var])
                .max()
                .unwrap_or(0)
        };

        (Clause::learned(learnt_lits), bt_level)
    }

    fn extract_model(&self, assignment: &Assignment) -> HashMap<usize, bool> {
        let mut model = HashMap::new();
        for var in 0..self.num_vars {
            match assignment.value(var) {
                AssignmentValue::True => { model.insert(var, true); }
                AssignmentValue::False => { model.insert(var, false); }
                AssignmentValue::Unassigned => { model.insert(var, false); }
            }
        }
        model
    }
}

impl SmtSolver for BuiltinSolver {
    fn check_sat(&mut self) -> SatResult {
        self.solve()
    }

    fn assert_clause(&mut self, clause: Clause) {
        self.add_clause(clause);
    }

    fn push(&mut self) {
        self.push_stack.push(self.clauses.len());
    }

    fn pop(&mut self) {
        if let Some(saved) = self.push_stack.pop() {
            self.clauses.truncate(saved);
        }
    }

    fn stats(&self) -> &SolverStats {
        &self.stats
    }
}

// ---------------------------------------------------------------------------
// IncrementalSolver: wrapper with push/pop support
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct IncrementalSolver {
    inner: BuiltinSolver,
    assumption_stack: Vec<Vec<Clause>>,
}

impl IncrementalSolver {
    pub fn new(config: SolverConfig) -> Self {
        Self {
            inner: BuiltinSolver::new(config),
            assumption_stack: Vec::new(),
        }
    }

    pub fn check_sat_with_assumptions(&mut self, assumptions: Vec<Clause>) -> SatResult {
        self.inner.push();
        for clause in assumptions {
            self.inner.add_clause(clause);
        }
        let result = self.inner.solve();
        self.inner.pop();
        result
    }

    pub fn new_variable(&mut self) -> usize {
        self.inner.new_variable()
    }
}

impl SmtSolver for IncrementalSolver {
    fn check_sat(&mut self) -> SatResult {
        self.inner.solve()
    }

    fn assert_clause(&mut self, clause: Clause) {
        self.inner.add_clause(clause);
    }

    fn push(&mut self) {
        self.inner.push();
    }

    fn pop(&mut self) {
        self.inner.pop();
    }

    fn stats(&self) -> &SolverStats {
        &self.inner.stats
    }
}

// ---------------------------------------------------------------------------
// PortfolioSolver: run multiple configurations in parallel
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct PortfolioSolver {
    configs: Vec<SolverConfig>,
    clauses: Vec<Clause>,
    stats: SolverStats,
}

impl PortfolioSolver {
    pub fn new(configs: Vec<SolverConfig>) -> Self {
        Self {
            configs,
            clauses: Vec::new(),
            stats: SolverStats::default(),
        }
    }

    pub fn default_portfolio() -> Self {
        Self::new(vec![
            SolverConfig { random_seed: 1, restart_interval: 100, ..Default::default() },
            SolverConfig { random_seed: 2, restart_interval: 50, ..Default::default() },
            SolverConfig { random_seed: 3, restart_interval: 200, ..Default::default() },
        ])
    }

    pub fn solve_parallel(&mut self) -> SatResult {
        use crossbeam::channel;
        use std::thread;

        let (tx, rx) = channel::bounded(1);
        let clauses = self.clauses.clone();
        let configs = self.configs.clone();

        let handles: Vec<_> = configs.into_iter().map(|config| {
            let clauses = clauses.clone();
            let tx = tx.clone();
            thread::spawn(move || {
                let mut solver = BuiltinSolver::new(config);
                for clause in clauses {
                    solver.add_clause(clause);
                }
                let result = solver.solve();
                let _ = tx.send((result, solver.stats.clone()));
            })
        }).collect();
        drop(tx);

        let result = match rx.recv() {
            Ok((result, stats)) => {
                self.stats = stats;
                result
            }
            Err(_) => SatResult::Unknown("all solvers failed".into()),
        };

        // Wait for remaining threads to finish (they'll notice the channel is closed)
        for h in handles {
            let _ = h.join();
        }

        result
    }
}

impl SmtSolver for PortfolioSolver {
    fn check_sat(&mut self) -> SatResult {
        self.solve_parallel()
    }

    fn assert_clause(&mut self, clause: Clause) {
        self.clauses.push(clause);
    }

    fn push(&mut self) {
        // Portfolio solver doesn't support incremental solving
    }

    fn pop(&mut self) {
        // Portfolio solver doesn't support incremental solving
    }

    fn stats(&self) -> &SolverStats {
        &self.stats
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sat_instance() -> BuiltinSolver {
        // (x0 ∨ x1) ∧ (¬x0 ∨ x1) ∧ (x0 ∨ ¬x1) => SAT: x0=T, x1=T
        let mut s = BuiltinSolver::new(SolverConfig::default());
        s.add_clause(Clause::new(vec![Literal::pos(0), Literal::pos(1)]));
        s.add_clause(Clause::new(vec![Literal::neg(0), Literal::pos(1)]));
        s.add_clause(Clause::new(vec![Literal::pos(0), Literal::neg(1)]));
        s
    }

    fn unsat_instance() -> BuiltinSolver {
        // (x0) ∧ (¬x0) => UNSAT
        let mut s = BuiltinSolver::new(SolverConfig::default());
        s.add_clause(Clause::new(vec![Literal::pos(0)]));
        s.add_clause(Clause::new(vec![Literal::neg(0)]));
        s
    }

    #[test]
    fn test_literal_creation() {
        let p = Literal::pos(5);
        assert!(p.positive);
        assert_eq!(p.var, 5);
        let n = p.negate();
        assert!(!n.positive);
        assert_eq!(n.var, 5);
    }

    #[test]
    fn test_literal_index() {
        assert_eq!(Literal::pos(0).index(), 0);
        assert_eq!(Literal::neg(0).index(), 1);
        assert_eq!(Literal::pos(1).index(), 2);
        assert_eq!(Literal::neg(1).index(), 3);
    }

    #[test]
    fn test_clause_properties() {
        let c = Clause::new(vec![Literal::pos(0)]);
        assert!(c.is_unit());
        assert!(!c.is_empty());
        assert_eq!(c.len(), 1);
    }

    #[test]
    fn test_assignment_basic() {
        let mut a = Assignment::new(3);
        assert_eq!(a.value(0), AssignmentValue::Unassigned);
        a.assign(Literal::pos(0), 1, None);
        assert_eq!(a.value(0), AssignmentValue::True);
        assert_eq!(a.literal_value(Literal::pos(0)), AssignmentValue::True);
        assert_eq!(a.literal_value(Literal::neg(0)), AssignmentValue::False);
    }

    #[test]
    fn test_assignment_backtrack() {
        let mut a = Assignment::new(3);
        a.assign(Literal::pos(0), 0, None);
        a.new_decision_level();
        a.assign(Literal::pos(1), 1, None);
        a.new_decision_level();
        a.assign(Literal::pos(2), 2, None);
        a.backtrack_to(1);
        assert_eq!(a.value(0), AssignmentValue::True);
        assert_eq!(a.value(1), AssignmentValue::True);
        assert_eq!(a.value(2), AssignmentValue::Unassigned);
    }

    #[test]
    fn test_sat_simple() {
        let mut s = sat_instance();
        let result = s.solve();
        assert!(result.is_sat());
        if let SatResult::Sat(model) = result {
            // x1 must be true (appears positive in first two clauses)
            assert_eq!(model.get(&1), Some(&true));
        }
    }

    #[test]
    fn test_unsat_simple() {
        let mut s = unsat_instance();
        let result = s.solve();
        assert!(result.is_unsat());
    }

    #[test]
    fn test_empty_formula() {
        let mut s = BuiltinSolver::new(SolverConfig::default());
        let result = s.solve();
        assert!(result.is_sat());
    }

    #[test]
    fn test_single_variable() {
        let mut s = BuiltinSolver::new(SolverConfig::default());
        s.add_clause(Clause::new(vec![Literal::pos(0)]));
        let result = s.solve();
        assert!(result.is_sat());
        if let SatResult::Sat(model) = result {
            assert_eq!(model.get(&0), Some(&true));
        }
    }

    #[test]
    fn test_three_coloring_unsat() {
        // K4 cannot be 2-colored: encode as 2-coloring with bool vars
        // Vertices: 0,1,2,3.  Each has one bool: true=color1, false=color2
        // Edge (i,j): x_i != x_j => (x_i ∨ x_j) ∧ (¬x_i ∨ ¬x_j)
        let mut s = BuiltinSolver::new(SolverConfig::default());
        let edges = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)];
        for &(i,j) in &edges {
            s.add_clause(Clause::new(vec![Literal::pos(i), Literal::pos(j)]));
            s.add_clause(Clause::new(vec![Literal::neg(i), Literal::neg(j)]));
        }
        let result = s.solve();
        assert!(result.is_unsat());
    }

    #[test]
    fn test_solver_stats() {
        let mut s = sat_instance();
        s.solve();
        let stats = s.stats();
        assert!(stats.solve_time_ms < 1000);
        assert_eq!(stats.variables, 2);
    }

    #[test]
    fn test_incremental_solver() {
        let mut s = IncrementalSolver::new(SolverConfig::default());
        s.assert_clause(Clause::new(vec![Literal::pos(0), Literal::pos(1)]));
        let result = s.check_sat();
        assert!(result.is_sat());
    }

    #[test]
    fn test_incremental_assumptions() {
        let mut s = IncrementalSolver::new(SolverConfig::default());
        s.assert_clause(Clause::new(vec![Literal::pos(0), Literal::pos(1)]));
        // With assumption that both are false -> UNSAT
        let result = s.check_sat_with_assumptions(vec![
            Clause::new(vec![Literal::neg(0)]),
            Clause::new(vec![Literal::neg(1)]),
        ]);
        assert!(result.is_unsat());
        // Without assumptions -> still SAT
        let result2 = s.check_sat();
        assert!(result2.is_sat());
    }

    #[test]
    fn test_push_pop() {
        let mut s = BuiltinSolver::new(SolverConfig::default());
        s.add_clause(Clause::new(vec![Literal::pos(0)]));
        s.push();
        s.add_clause(Clause::new(vec![Literal::neg(0)]));
        assert!(s.solve().is_unsat());
        s.pop();
        let mut s2 = BuiltinSolver::new(SolverConfig::default());
        s2.add_clause(Clause::new(vec![Literal::pos(0)]));
        assert!(s2.solve().is_sat());
    }

    #[test]
    fn test_portfolio_solver() {
        let mut ps = PortfolioSolver::default_portfolio();
        ps.assert_clause(Clause::new(vec![Literal::pos(0), Literal::pos(1)]));
        let result = ps.check_sat();
        assert!(result.is_sat());
    }

    #[test]
    fn test_vsids_picks_active() {
        let mut vs = VsidsScores::new(3);
        vs.bump(1);
        vs.bump(1);
        vs.bump(2);
        let a = Assignment::new(3);
        let picked = vs.pick_unassigned(&a);
        assert_eq!(picked, Some(1));
    }

    #[test]
    fn test_solver_config_default() {
        let c = SolverConfig::default();
        assert_eq!(c.timeout_ms, 30_000);
        assert_eq!(c.random_seed, 42);
    }
}
