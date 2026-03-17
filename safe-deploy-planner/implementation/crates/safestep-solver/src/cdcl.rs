// Full CDCL SAT solver: conflict-driven clause learning with VSIDS, First-UIP,
// non-chronological backtracking, restarts, phase saving, and incremental solving.

use crate::clause::{Clause, ClauseDatabase, ClauseId, WatchList};
use crate::config::{PhasePolicy, RestartStrategy, SolverConfig, VarSelectionHeuristic};
use crate::propagation::{init_watches, PropagationEngine, PropagationResult};
use crate::variable::{
    Assignment, DecisionLevel, Literal, LiteralVec, PhaseSaver, Reason, Variable, VsidsActivity,
};
use smallvec::smallvec;
use std::collections::HashSet;
use std::fmt;

// ── SatResult ─────────────────────────────────────────────────────────────────

/// Result of a SAT solving attempt.
#[derive(Debug, Clone)]
pub enum SatResult {
    /// Formula is satisfiable with the given assignment.
    Satisfiable(Assignment),
    /// Formula is unsatisfiable (with optional UNSAT core of assumptions).
    Unsatisfiable(UnsatCore),
    /// Solver was unable to determine satisfiability within limits.
    Unknown(String),
}

impl SatResult {
    pub fn is_sat(&self) -> bool {
        matches!(self, SatResult::Satisfiable(_))
    }
    pub fn is_unsat(&self) -> bool {
        matches!(self, SatResult::Unsatisfiable(_))
    }
}

// ── UnsatCore ─────────────────────────────────────────────────────────────────

/// A (possibly minimal) subset of assumptions that cause unsatisfiability.
#[derive(Debug, Clone, Default)]
pub struct UnsatCore {
    pub literals: Vec<Literal>,
}

impl UnsatCore {
    pub fn new() -> Self {
        UnsatCore {
            literals: Vec::new(),
        }
    }

    pub fn from_literals(lits: Vec<Literal>) -> Self {
        UnsatCore { literals: lits }
    }

    pub fn is_empty(&self) -> bool {
        self.literals.is_empty()
    }

    pub fn len(&self) -> usize {
        self.literals.len()
    }
}

// ── SolverStats ───────────────────────────────────────────────────────────────

/// Solver statistics.
#[derive(Debug, Clone, Default)]
pub struct SolverStats {
    pub decisions: u64,
    pub propagations: u64,
    pub conflicts: u64,
    pub restarts: u64,
    pub learned_clauses: u64,
    pub deleted_clauses: u64,
    pub max_decision_level: u32,
}

impl fmt::Display for SolverStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "decisions={} propagations={} conflicts={} restarts={} learned={}",
            self.decisions,
            self.propagations,
            self.conflicts,
            self.restarts,
            self.learned_clauses
        )
    }
}

// ── RestartPolicy ─────────────────────────────────────────────────────────────

/// Manages restart scheduling.
#[derive(Debug, Clone)]
pub struct RestartPolicy {
    strategy: RestartStrategy,
    /// Conflicts until next restart.
    conflicts_until_restart: u64,
    /// Number of restarts performed.
    num_restarts: u64,
    /// Current Luby index (for Luby restarts).
    luby_index: u64,
    /// Current interval (for geometric restarts).
    current_interval: f64,
}

impl RestartPolicy {
    pub fn new(strategy: RestartStrategy) -> Self {
        let initial_limit = match strategy {
            RestartStrategy::Luby { base_interval } => base_interval,
            RestartStrategy::Geometric { initial, .. } => initial,
            RestartStrategy::Fixed { interval } => interval,
            RestartStrategy::Never => u64::MAX,
        };
        RestartPolicy {
            strategy,
            conflicts_until_restart: initial_limit,
            num_restarts: 0,
            luby_index: 0,
            current_interval: match strategy {
                RestartStrategy::Geometric { initial, .. } => initial as f64,
                _ => 0.0,
            },
        }
    }

    /// Returns true if it's time to restart.
    pub fn should_restart(&self) -> bool {
        self.conflicts_until_restart == 0
    }

    /// Record a conflict and decrement the counter.
    pub fn on_conflict(&mut self) {
        if self.conflicts_until_restart > 0 {
            self.conflicts_until_restart -= 1;
        }
    }

    /// Called after a restart is performed.
    pub fn on_restart(&mut self) {
        self.num_restarts += 1;
        match self.strategy {
            RestartStrategy::Luby { base_interval } => {
                self.luby_index += 1;
                self.conflicts_until_restart =
                    base_interval * SolverConfig::luby_value(self.luby_index);
            }
            RestartStrategy::Geometric { factor, .. } => {
                self.current_interval *= factor;
                self.conflicts_until_restart = self.current_interval as u64;
            }
            RestartStrategy::Fixed { interval } => {
                self.conflicts_until_restart = interval;
            }
            RestartStrategy::Never => {
                self.conflicts_until_restart = u64::MAX;
            }
        }
    }
}

// ── ConflictAnalysis ──────────────────────────────────────────────────────────

/// Result of conflict analysis.
#[derive(Debug, Clone)]
pub struct ConflictAnalysisResult {
    /// The learned clause.
    pub learned_clause: LiteralVec,
    /// The backtrack level.
    pub backtrack_level: DecisionLevel,
    /// LBD of the learned clause.
    pub lbd: u32,
}

// ── CdclSolver ────────────────────────────────────────────────────────────────

/// Full CDCL SAT solver.
pub struct CdclSolver {
    /// Solver configuration.
    config: SolverConfig,
    /// Clause database.
    pub clause_db: ClauseDatabase,
    /// Watch lists.
    watch_list: WatchList,
    /// Propagation engine (trail + BCP).
    prop_engine: PropagationEngine,
    /// Variable assignment.
    pub assignment: Assignment,
    /// Number of variables.
    num_vars: usize,
    /// VSIDS activity scores.
    vsids: VsidsActivity,
    /// Phase saver.
    phase_saver: PhaseSaver,
    /// Restart policy.
    restart_policy: RestartPolicy,
    /// Current assumptions.
    assumptions: Vec<Literal>,
    /// Solver statistics.
    pub stats: SolverStats,
    /// Temporary buffer for conflict analysis (reused across calls).
    seen: Vec<bool>,
    /// Counter for learned clause limit growth.
    learned_limit: usize,
    /// UNSAT core (assumptions that participate in conflict).
    unsat_core: UnsatCore,
    /// Flag set when a top-level conflict is detected during clause addition.
    top_level_conflict: bool,
}

impl CdclSolver {
    /// Create a new CDCL solver with the given configuration.
    pub fn new(config: SolverConfig) -> Self {
        CdclSolver {
            restart_policy: RestartPolicy::new(config.restart_strategy),
            learned_limit: config.max_learned_clauses,
            config,
            clause_db: ClauseDatabase::new(),
            watch_list: WatchList::new(0),
            prop_engine: PropagationEngine::new(0),
            assignment: Assignment::new(0),
            num_vars: 0,
            vsids: VsidsActivity::new(0, 0.95, 1.0),
            phase_saver: PhaseSaver::new(0),
            assumptions: Vec::new(),
            stats: SolverStats::default(),
            seen: Vec::new(),
            unsat_core: UnsatCore::new(),
            top_level_conflict: false,
        }
    }

    /// Create a solver with default configuration.
    pub fn default_solver() -> Self {
        Self::new(SolverConfig::default())
    }

    /// Ensure internal data structures can accommodate `n` variables.
    fn ensure_vars(&mut self, n: usize) {
        if n > self.num_vars {
            let old_count = self.num_vars;
            self.num_vars = n;
            self.assignment.resize(n);
            let lit_codes = (n + 1) * 2;
            self.watch_list.resize(lit_codes);
            self.prop_engine.resize(n);
            self.vsids.resize(n);
            self.phase_saver.resize(n);
            self.seen.resize(n, false);
            // Insert new variables into the VSIDS heap.
            for i in (old_count + 1)..=n {
                self.vsids.insert(Variable::new(i as u32));
            }
        }
    }

    /// Add a clause to the solver. Literals are given in DIMACS format.
    pub fn add_clause_dimacs(&mut self, dimacs: &[i32]) -> Option<ClauseId> {
        let lits: LiteralVec = dimacs.iter().map(|&d| Literal::from_dimacs(d)).collect();
        self.add_clause_lits(lits)
    }

    /// Add a clause from a LiteralVec.
    pub fn add_clause_lits(&mut self, mut lits: LiteralVec) -> Option<ClauseId> {
        // Find max variable and ensure capacity.
        let max_var = lits.iter().map(|l| l.var().0).max().unwrap_or(0) as usize;
        self.ensure_vars(max_var);

        // Remove duplicates and check for tautology.
        lits.sort();
        lits.dedup();
        for i in 0..lits.len() {
            for j in (i + 1)..lits.len() {
                if lits[i] == lits[j].negated() {
                    return None; // Tautological clause.
                }
            }
        }

        if lits.is_empty() {
            // Empty clause: problem is trivially UNSAT.
            self.top_level_conflict = true;
            let clause = Clause::new(lits);
            let cid = self.clause_db.add_clause(clause);
            return Some(cid);
        }

        if lits.len() == 1 {
            // Unit clause: enqueue immediately.
            let lit = lits[0];
            let clause = Clause::new(lits);
            let cid = self.clause_db.add_clause(clause);
            match self.assignment.get(lit.var()) {
                None => {
                    self.prop_engine
                        .enqueue_unit(lit, Reason::Propagation(cid.0), &mut self.assignment);
                }
                Some(v) if v != lit.polarity() => {
                    // Conflicting unit clause: the variable is already assigned
                    // with the opposite polarity.
                    self.top_level_conflict = true;
                }
                _ => {
                    // Already assigned with the same polarity; no action needed.
                }
            }
            return Some(cid);
        }

        let clause = Clause::new(lits);
        let w0 = clause.literals[0];
        let w1 = clause.literals[1];
        let cid = self.clause_db.add_clause(clause);
        self.watch_list.add_watch(w0, cid, w1);
        self.watch_list.add_watch(w1, cid, w0);
        Some(cid)
    }

    /// Add an assumption for the next solve call.
    pub fn assume(&mut self, lit: Literal) {
        let max_var = lit.var().0 as usize;
        self.ensure_vars(max_var);
        self.assumptions.push(lit);
    }

    /// Clear all assumptions.
    pub fn clear_assumptions(&mut self) {
        self.assumptions.clear();
    }

    /// Main solve loop. Returns SAT/UNSAT/Unknown.
    pub fn solve(&mut self) -> SatResult {
        // Check for conflicts detected during clause addition.
        if self.top_level_conflict {
            return SatResult::Unsatisfiable(UnsatCore::new());
        }

        // Initial unit propagation.
        match self.prop_engine.propagate(
            &mut self.assignment,
            &mut self.clause_db,
            &mut self.watch_list,
        ) {
            PropagationResult::Conflict(_) => {
                return SatResult::Unsatisfiable(UnsatCore::new());
            }
            PropagationResult::Ok => {}
        }

        // Try to assign assumption literals.
        let assumptions = self.assumptions.clone();
        let mut assumption_idx = 0;

        let mut iteration_limit: u64 = match self.config.timeout {
            Some(d) => d.as_millis() as u64 * 1000, // rough proxy
            None => u64::MAX,
        };

        loop {
            if iteration_limit == 0 {
                return SatResult::Unknown("iteration limit reached".into());
            }
            iteration_limit = iteration_limit.saturating_sub(1);

            // Try to propagate.
            match self.prop_engine.propagate(
                &mut self.assignment,
                &mut self.clause_db,
                &mut self.watch_list,
            ) {
                PropagationResult::Conflict(conflict_clause) => {
                    self.stats.conflicts += 1;
                    self.restart_policy.on_conflict();

                    if self.prop_engine.current_level() == 0 {
                        // Top-level conflict: UNSAT.
                        self.extract_unsat_core(&assumptions);
                        return SatResult::Unsatisfiable(self.unsat_core.clone());
                    }

                    // Analyze conflict.
                    let analysis = self.analyze_conflict(conflict_clause);
                    self.stats.learned_clauses += 1;

                    // Backtrack.
                    let unassigned =
                        self.prop_engine
                            .backtrack_to(analysis.backtrack_level, &mut self.assignment);

                    // Save phases of unassigned variables.
                    for &lit in &unassigned {
                        self.phase_saver.save(lit.var(), lit.polarity());
                    }

                    // Re-insert unassigned variables into VSIDS heap.
                    for &lit in &unassigned {
                        self.vsids.insert(lit.var());
                    }

                    // If assumption_idx is beyond what's still assigned, rewind it.
                    assumption_idx = assumption_idx.min(
                        assumptions
                            .iter()
                            .position(|a| self.assignment.get(a.var()).is_none())
                            .unwrap_or(assumptions.len()),
                    );

                    // Learn the clause.
                    self.learn_clause(analysis.learned_clause, analysis.lbd);

                    // Clause deletion.
                    if self.clause_db.learned_count() > self.learned_limit {
                        self.reduce_learned_clauses();
                    }
                }
                PropagationResult::Ok => {
                    // Restart check.
                    if self.restart_policy.should_restart()
                        && self.prop_engine.current_level() > 0
                    {
                        self.restart(&assumptions);
                        assumption_idx = 0;
                        self.stats.restarts += 1;
                        self.restart_policy.on_restart();
                        continue;
                    }

                    // Try to assign the next assumption.
                    if assumption_idx < assumptions.len() {
                        let assume_lit = assumptions[assumption_idx];
                        assumption_idx += 1;

                        match self.assignment.get(assume_lit.var()) {
                            Some(v) if v == assume_lit.polarity() => {
                                // Already assigned correctly; continue.
                                continue;
                            }
                            Some(_) => {
                                // Conflict with an assumption.
                                self.extract_unsat_core(&assumptions);
                                return SatResult::Unsatisfiable(self.unsat_core.clone());
                            }
                            None => {
                                self.prop_engine.decide(assume_lit, &mut self.assignment);
                                self.stats.decisions += 1;
                                continue;
                            }
                        }
                    }

                    // Pick a decision variable.
                    match self.decide() {
                        Some(_) => {
                            self.stats.decisions += 1;
                        }
                        None => {
                            // All variables are assigned: SAT!
                            return SatResult::Satisfiable(self.assignment.clone());
                        }
                    }
                }
            }
        }
    }

    /// Pick a decision variable and assign it using the configured heuristic.
    fn decide(&mut self) -> Option<Literal> {
        match self.config.var_selection {
            VarSelectionHeuristic::Vsids => self.decide_vsids(),
            VarSelectionHeuristic::Sequential => self.decide_sequential(),
            VarSelectionHeuristic::Random => self.decide_sequential(), // fallback
        }
    }

    fn decide_vsids(&mut self) -> Option<Literal> {
        loop {
            match self.vsids.pop_max() {
                Some(var) => {
                    if self.assignment.get(var).is_none() {
                        let polarity = self.pick_phase(var);
                        let lit = if polarity {
                            var.positive()
                        } else {
                            var.negative()
                        };
                        self.prop_engine.decide(lit, &mut self.assignment);
                        return Some(lit);
                    }
                    // Already assigned, skip.
                }
                None => return None,
            }
        }
    }

    fn decide_sequential(&mut self) -> Option<Literal> {
        for i in 1..=self.num_vars {
            let var = Variable::new(i as u32);
            if self.assignment.get(var).is_none() {
                let polarity = self.pick_phase(var);
                let lit = if polarity {
                    var.positive()
                } else {
                    var.negative()
                };
                self.prop_engine.decide(lit, &mut self.assignment);
                return Some(lit);
            }
        }
        None
    }

    fn pick_phase(&self, var: Variable) -> bool {
        match self.config.phase_policy {
            PhasePolicy::Positive => true,
            PhasePolicy::Negative => false,
            PhasePolicy::PhaseSaving => self.phase_saver.get(var),
            PhasePolicy::Random => {
                // Simple deterministic "random" based on variable index.
                var.0 % 2 == 0
            }
        }
    }

    /// First-UIP conflict analysis.
    fn analyze_conflict(&mut self, conflict_clause: ClauseId) -> ConflictAnalysisResult {
        let current_level = self.prop_engine.current_level();
        let mut learned: LiteralVec = smallvec![];
        let mut counter = 0; // literals at current level yet to be resolved
        let seen = &mut self.seen;

        // Clear seen array.
        for s in seen.iter_mut() {
            *s = false;
        }

        // Start with the conflict clause.
        let conflict_lits: Vec<Literal> = self.clause_db.get(conflict_clause).literals.to_vec();
        self.clause_db.bump_activity(conflict_clause);

        for &lit in &conflict_lits {
            let var = lit.var();
            if var.array_index() < seen.len() {
                seen[var.array_index()] = true;
                let var_level = self.assignment.level(var);
                if var_level == current_level {
                    counter += 1;
                } else if var_level > 0 {
                    learned.push(lit.negated());
                }
                self.vsids.bump(var);
            }
        }

        // Walk the trail backwards until we find the UIP.
        let trail = self.prop_engine.trail.all_entries();
        let mut trail_idx = trail.len();

        loop {
            if counter <= 1 {
                break;
            }
            trail_idx -= 1;
            let entry = &trail[trail_idx];
            let var = entry.literal.var();
            if var.array_index() >= seen.len() || !seen[var.array_index()] {
                continue;
            }

            // If this literal was propagated by a clause, resolve.
            if let Reason::Propagation(cid) = entry.reason {
                let reason_cid = ClauseId::new(cid);
                if self.clause_db.is_active(reason_cid) {
                    let reason_lits: Vec<Literal> =
                        self.clause_db.get(reason_cid).literals.to_vec();
                    self.clause_db.bump_activity(reason_cid);

                    for &lit in &reason_lits {
                        let v = lit.var();
                        if v.array_index() < seen.len() && !seen[v.array_index()] {
                            seen[v.array_index()] = true;
                            let v_level = self.assignment.level(v);
                            if v_level == current_level {
                                counter += 1;
                            } else if v_level > 0 {
                                learned.push(lit.negated());
                            }
                            self.vsids.bump(v);
                        }
                    }
                }
            }

            counter -= 1;
        }

        // Find the UIP literal: the last seen literal at the current level in the trail.
        // Walk forward from trail_idx to find it.
        let mut uip_lit = None;
        for i in (0..trail.len()).rev() {
            let entry = &trail[i];
            let var = entry.literal.var();
            if var.array_index() < seen.len()
                && seen[var.array_index()]
                && entry.level == current_level
            {
                uip_lit = Some(entry.literal.negated());
                break;
            }
        }

        // The UIP literal goes first in the learned clause.
        let uip = uip_lit.unwrap_or_else(|| {
            // Fallback: shouldn't happen in a correct implementation, but safety first.
            Literal::from_dimacs(1)
        });

        // Build final learned clause: UIP + other literals.
        let mut final_clause: LiteralVec = smallvec![uip];
        final_clause.extend(learned.iter().cloned());

        // Minimization: try to remove literals that are implied by others in the clause.
        if self.config.minimize_learned && final_clause.len() > 2 {
            self.minimize_clause(&mut final_clause);
        }

        // Determine backtrack level: second-highest level in the learned clause.
        let backtrack_level = if final_clause.len() <= 1 {
            0
        } else {
            let mut max_level = 0;
            let mut max_idx = 1;
            for i in 1..final_clause.len() {
                let level = self.assignment.level(final_clause[i].var());
                if level > max_level {
                    max_level = level;
                    max_idx = i;
                }
            }
            // Swap the literal with highest level to position 1 (for watch setup).
            if max_idx != 1 {
                final_clause.swap(1, max_idx);
            }
            max_level
        };

        // Compute LBD.
        let mut level_set = HashSet::new();
        for &lit in &final_clause {
            level_set.insert(self.assignment.level(lit.var()));
        }
        let lbd = level_set.len() as u32;

        // Decay VSIDS.
        self.vsids.decay();
        self.clause_db.decay_activities();

        ConflictAnalysisResult {
            learned_clause: final_clause,
            backtrack_level,
            lbd,
        }
    }

    /// Clause minimization: remove redundant literals.
    fn minimize_clause(&self, clause: &mut LiteralVec) {
        if clause.len() <= 2 {
            return;
        }

        let mut levels_in_clause = HashSet::new();
        for &lit in clause.iter() {
            levels_in_clause.insert(self.assignment.level(lit.var()));
        }

        let mut i = 1; // Keep the UIP at position 0.
        while i < clause.len() {
            let var = clause[i].var();
            if self.lit_is_redundant(var, &levels_in_clause) {
                clause.swap_remove(i);
            } else {
                i += 1;
            }
        }
    }

    /// Check if a literal is redundant in the learned clause (all its antecedents
    /// are at levels represented in the clause).
    fn lit_is_redundant(&self, var: Variable, levels: &HashSet<DecisionLevel>) -> bool {
        match self.assignment.reason(var) {
            Reason::Propagation(cid) => {
                let clause_id = ClauseId::new(cid);
                if !self.clause_db.is_active(clause_id) {
                    return false;
                }
                let reason = self.clause_db.get(clause_id);
                for &lit in &reason.literals {
                    if lit.var() == var {
                        continue;
                    }
                    let lit_level = self.assignment.level(lit.var());
                    if lit_level == 0 {
                        continue;
                    }
                    if !levels.contains(&lit_level) {
                        return false;
                    }
                }
                true
            }
            _ => false,
        }
    }

    /// Add a learned clause and enqueue the asserting literal.
    fn learn_clause(&mut self, lits: LiteralVec, lbd: u32) {
        if lits.is_empty() {
            return;
        }

        if lits.len() == 1 {
            // Unit learned clause: enqueue at level 0.
            let lit = lits[0];
            self.prop_engine
                .enqueue_unit(lit, Reason::Assumption, &mut self.assignment);
            let clause = Clause::new_learned(lits, lbd);
            self.clause_db.add_clause(clause);
            return;
        }

        let asserting_lit = lits[0];
        let clause = Clause::new_learned(lits.clone(), lbd);
        let cid = self.clause_db.add_clause(clause);

        // Set up watches for the learned clause.
        self.watch_list.add_watch(lits[0], cid, lits[1]);
        self.watch_list.add_watch(lits[1], cid, lits[0]);

        // The asserting literal is unit at the backtrack level.
        self.prop_engine
            .enqueue_propagated(asserting_lit, cid, &mut self.assignment);
    }

    /// Reduce learned clauses via garbage collection.
    fn reduce_learned_clauses(&mut self) {
        let (activity_threshold, max_lbd) = match self.config.clause_deletion {
            crate::config::ClauseDeletionStrategy::Activity { threshold } => (threshold, u32::MAX),
            crate::config::ClauseDeletionStrategy::Lbd { max_lbd } => (0.0, max_lbd),
            crate::config::ClauseDeletionStrategy::Combined {
                activity_threshold,
                max_lbd,
            } => (activity_threshold, max_lbd),
        };

        let deleted = self.clause_db.gc_learned_clauses(
            self.learned_limit,
            activity_threshold,
            max_lbd,
        );

        // Remove watches for deleted clauses.
        for &cid in &deleted {
            // We don't eagerly clean watch lists; they'll be cleaned lazily during propagation.
        }

        self.stats.deleted_clauses += deleted.len() as u64;
        self.learned_limit =
            (self.learned_limit as f64 * self.config.learned_clause_growth) as usize;
    }

    /// Restart: backtrack to level 0.
    fn restart(&mut self, _assumptions: &[Literal]) {
        if self.prop_engine.current_level() > 0 {
            let unassigned = self.prop_engine.backtrack_to(0, &mut self.assignment);
            for &lit in &unassigned {
                self.phase_saver.save(lit.var(), lit.polarity());
                self.vsids.insert(lit.var());
            }
        }
    }

    /// Extract UNSAT core from assumptions.
    fn extract_unsat_core(&mut self, assumptions: &[Literal]) {
        let assumption_set: HashSet<Variable> =
            assumptions.iter().map(|l| l.var()).collect();
        let mut core = Vec::new();

        // Walk the trail to find which assumptions were involved in the conflict.
        for entry in self.prop_engine.trail.all_entries() {
            if assumption_set.contains(&entry.literal.var()) {
                core.push(entry.literal);
            }
        }

        if core.is_empty() {
            core = assumptions.to_vec();
        }

        self.unsat_core = UnsatCore::from_literals(core);
    }

    /// Get solver statistics.
    pub fn statistics(&self) -> &SolverStats {
        &self.stats
    }

    /// Get the number of variables.
    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    /// Get the current number of clauses.
    pub fn num_clauses(&self) -> usize {
        self.clause_db.clause_count()
    }

    /// Reset the solver state but keep clauses.
    pub fn reset(&mut self) {
        self.prop_engine.reset();
        self.assignment.clear();
        self.assumptions.clear();
        self.stats = SolverStats::default();
        self.unsat_core = UnsatCore::new();
        self.top_level_conflict = false;
        // Re-initialize watches.
        self.watch_list.clear();
        init_watches(&self.clause_db, &mut self.watch_list);
        // Re-insert all variables into VSIDS heap.
        for i in 1..=self.num_vars {
            self.vsids.insert(Variable::new(i as u32));
        }
        // Re-enqueue unit clauses.
        for cid in self.clause_db.active_clause_ids().collect::<Vec<_>>() {
            let clause = self.clause_db.get(cid);
            if clause.len() == 1 {
                let lit = clause.literals[0];
                match self.assignment.get(lit.var()) {
                    None => {
                        self.prop_engine.enqueue_unit(
                            lit,
                            Reason::Propagation(cid.0),
                            &mut self.assignment,
                        );
                    }
                    Some(v) if v != lit.polarity() => {
                        self.top_level_conflict = true;
                    }
                    _ => {}
                }
            }
        }
    }
}

impl fmt::Debug for CdclSolver {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CdclSolver")
            .field("num_vars", &self.num_vars)
            .field("num_clauses", &self.clause_db.clause_count())
            .field("stats", &self.stats)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn new_solver() -> CdclSolver {
        CdclSolver::new(SolverConfig::default())
    }

    #[test]
    fn test_empty_solver_is_sat() {
        let mut solver = new_solver();
        // No clauses: trivially SAT.
        let result = solver.solve();
        assert!(result.is_sat());
    }

    #[test]
    fn test_single_unit_clause() {
        let mut solver = new_solver();
        solver.add_clause_dimacs(&[1]);
        let result = solver.solve();
        assert!(result.is_sat());
        if let SatResult::Satisfiable(asgn) = result {
            assert_eq!(asgn.get(Variable::new(1)), Some(true));
        }
    }

    #[test]
    fn test_two_unit_clauses_consistent() {
        let mut solver = new_solver();
        solver.add_clause_dimacs(&[1]);
        solver.add_clause_dimacs(&[-2]);
        let result = solver.solve();
        assert!(result.is_sat());
        if let SatResult::Satisfiable(asgn) = result {
            assert_eq!(asgn.get(Variable::new(1)), Some(true));
            assert_eq!(asgn.get(Variable::new(2)), Some(false));
        }
    }

    #[test]
    fn test_contradictory_unit_clauses() {
        let mut solver = new_solver();
        solver.add_clause_dimacs(&[1]);
        solver.add_clause_dimacs(&[-1]);
        let result = solver.solve();
        assert!(result.is_unsat());
    }

    #[test]
    fn test_simple_sat() {
        // (x1 ∨ x2) ∧ (¬x1 ∨ x2)
        let mut solver = new_solver();
        solver.add_clause_dimacs(&[1, 2]);
        solver.add_clause_dimacs(&[-1, 2]);
        let result = solver.solve();
        assert!(result.is_sat());
        if let SatResult::Satisfiable(asgn) = result {
            assert_eq!(asgn.get(Variable::new(2)), Some(true));
        }
    }

    #[test]
    fn test_simple_unsat() {
        // (x1) ∧ (x2) ∧ (¬x1 ∨ ¬x2)
        let mut solver = new_solver();
        solver.add_clause_dimacs(&[1]);
        solver.add_clause_dimacs(&[2]);
        solver.add_clause_dimacs(&[-1, -2]);
        let result = solver.solve();
        assert!(result.is_unsat());
    }

    #[test]
    fn test_pigeonhole_2_1() {
        // 2 pigeons, 1 hole: UNSAT.
        // p1 -> h1, p2 -> h1, at most one pigeon per hole.
        // (p1h1) ∧ (p2h1) ∧ (¬p1h1 ∨ ¬p2h1)
        let mut solver = new_solver();
        solver.add_clause_dimacs(&[1]); // pigeon 1 in hole 1
        solver.add_clause_dimacs(&[2]); // pigeon 2 in hole 1
        solver.add_clause_dimacs(&[-1, -2]); // at most one
        let result = solver.solve();
        assert!(result.is_unsat());
    }

    #[test]
    fn test_implication_chain() {
        // x1 → x2 → x3 → x4 (chain of implications), plus x1 is forced true.
        // (x1), (¬x1 ∨ x2), (¬x2 ∨ x3), (¬x3 ∨ x4)
        let mut solver = new_solver();
        solver.add_clause_dimacs(&[1]);
        solver.add_clause_dimacs(&[-1, 2]);
        solver.add_clause_dimacs(&[-2, 3]);
        solver.add_clause_dimacs(&[-3, 4]);
        let result = solver.solve();
        assert!(result.is_sat());
        if let SatResult::Satisfiable(asgn) = result {
            assert_eq!(asgn.get(Variable::new(1)), Some(true));
            assert_eq!(asgn.get(Variable::new(2)), Some(true));
            assert_eq!(asgn.get(Variable::new(3)), Some(true));
            assert_eq!(asgn.get(Variable::new(4)), Some(true));
        }
    }

    #[test]
    fn test_3sat_satisfiable() {
        // A random-ish 3-SAT instance that is satisfiable.
        let mut solver = new_solver();
        solver.add_clause_dimacs(&[1, 2, 3]);
        solver.add_clause_dimacs(&[-1, -2, 3]);
        solver.add_clause_dimacs(&[1, -2, -3]);
        solver.add_clause_dimacs(&[-1, 2, -3]);
        let result = solver.solve();
        assert!(result.is_sat());
    }

    #[test]
    fn test_incremental_assumptions() {
        // (x1 ∨ x2), solve under assumption x1=false.
        let mut solver = new_solver();
        solver.add_clause_dimacs(&[1, 2]);
        solver.assume(Literal::from_dimacs(-1));
        let result = solver.solve();
        assert!(result.is_sat());
        if let SatResult::Satisfiable(asgn) = result {
            assert_eq!(asgn.get(Variable::new(2)), Some(true));
        }
    }

    #[test]
    fn test_assumptions_causing_unsat() {
        // (x1 ∨ x2), (¬x2). Under assumption ¬x1 → needs x2=true, but (¬x2) forces x2=false.
        let mut solver = new_solver();
        solver.add_clause_dimacs(&[1, 2]);
        solver.add_clause_dimacs(&[-2]);
        solver.assume(Literal::from_dimacs(-1));
        let result = solver.solve();
        assert!(result.is_unsat());
    }

    #[test]
    fn test_solver_stats() {
        let mut solver = new_solver();
        solver.add_clause_dimacs(&[1, 2]);
        solver.add_clause_dimacs(&[-1, 2]);
        solver.add_clause_dimacs(&[1, -2]);
        solver.add_clause_dimacs(&[-1, -2]);
        let _result = solver.solve();
        // Should have made at least one decision.
        assert!(solver.stats.conflicts > 0 || solver.stats.decisions > 0);
    }

    #[test]
    fn test_tautological_clause_ignored() {
        let mut solver = new_solver();
        let result = solver.add_clause_dimacs(&[1, -1]);
        assert!(result.is_none()); // Tautological clause should be None.
    }

    #[test]
    fn test_larger_sat_instance() {
        // A slightly larger satisfiable instance.
        let mut solver = new_solver();
        // Encode: at least one of {1,2,3} and at least one of {4,5,6} and various constraints.
        solver.add_clause_dimacs(&[1, 2, 3]);
        solver.add_clause_dimacs(&[4, 5, 6]);
        solver.add_clause_dimacs(&[-1, -4]);
        solver.add_clause_dimacs(&[-2, -5]);
        solver.add_clause_dimacs(&[-3, -6]);
        let result = solver.solve();
        assert!(result.is_sat());
    }

    #[test]
    fn test_restart_policy_luby() {
        let mut rp = RestartPolicy::new(RestartStrategy::Luby { base_interval: 10 });
        // Initial limit is 10.
        for _ in 0..10 {
            rp.on_conflict();
        }
        assert!(rp.should_restart());
        rp.on_restart();
        // Next limit should be 10 * luby(1) = 10.
        for _ in 0..10 {
            rp.on_conflict();
        }
        assert!(rp.should_restart());
    }

    #[test]
    fn test_restart_policy_geometric() {
        let mut rp = RestartPolicy::new(RestartStrategy::Geometric {
            initial: 100,
            factor: 2.0,
        });
        for _ in 0..100 {
            rp.on_conflict();
        }
        assert!(rp.should_restart());
        rp.on_restart();
        // Next limit is 200.
        assert!(!rp.should_restart());
    }

    #[test]
    fn test_unsat_core() {
        let core = UnsatCore::from_literals(vec![
            Literal::from_dimacs(1),
            Literal::from_dimacs(-2),
        ]);
        assert_eq!(core.len(), 2);
        assert!(!core.is_empty());
    }

    #[test]
    fn test_solver_reset() {
        let mut solver = new_solver();
        solver.add_clause_dimacs(&[1]);
        solver.add_clause_dimacs(&[-1]);
        let r1 = solver.solve();
        assert!(r1.is_unsat());

        solver.reset();
        // After reset, re-adding clauses and solving should work.
        // Note: clauses are still in the database.
        let r2 = solver.solve();
        assert!(r2.is_unsat()); // Same clauses, still UNSAT.
    }

    #[test]
    fn test_solver_num_vars_and_clauses() {
        let mut solver = new_solver();
        solver.add_clause_dimacs(&[1, 2, 3]);
        solver.add_clause_dimacs(&[-1, 4]);
        assert_eq!(solver.num_vars(), 4);
        assert_eq!(solver.num_clauses(), 2);
    }

    #[test]
    fn test_graph_coloring_sat() {
        // 3-coloring of a triangle (K3): SAT.
        // Variables: x_{node}_{color}, 1-based.
        // Node 0: vars 1,2,3 (colors R,G,B)
        // Node 1: vars 4,5,6
        // Node 2: vars 7,8,9
        let mut solver = new_solver();

        // At least one color per node.
        solver.add_clause_dimacs(&[1, 2, 3]);
        solver.add_clause_dimacs(&[4, 5, 6]);
        solver.add_clause_dimacs(&[7, 8, 9]);

        // At most one color per node (pairwise).
        for &(a, b) in &[
            (-1, -2), (-1, -3), (-2, -3),
            (-4, -5), (-4, -6), (-5, -6),
            (-7, -8), (-7, -9), (-8, -9),
        ] {
            solver.add_clause_dimacs(&[a, b]);
        }

        // Adjacent nodes have different colors.
        // Edge (0,1): ¬(same color).
        for c in 0..3 {
            solver.add_clause_dimacs(&[-(1 + c), -(4 + c)]);
        }
        // Edge (0,2):
        for c in 0..3 {
            solver.add_clause_dimacs(&[-(1 + c), -(7 + c)]);
        }
        // Edge (1,2):
        for c in 0..3 {
            solver.add_clause_dimacs(&[-(4 + c), -(7 + c)]);
        }

        let result = solver.solve();
        assert!(result.is_sat());
    }

    #[test]
    fn test_graph_coloring_unsat() {
        // 2-coloring of a triangle (K3): UNSAT.
        // Node 0: vars 1,2
        // Node 1: vars 3,4
        // Node 2: vars 5,6
        let mut solver = new_solver();

        // At least one color per node.
        solver.add_clause_dimacs(&[1, 2]);
        solver.add_clause_dimacs(&[3, 4]);
        solver.add_clause_dimacs(&[5, 6]);

        // At most one color per node.
        solver.add_clause_dimacs(&[-1, -2]);
        solver.add_clause_dimacs(&[-3, -4]);
        solver.add_clause_dimacs(&[-5, -6]);

        // Edges: different colors.
        // Edge (0,1):
        solver.add_clause_dimacs(&[-1, -3]);
        solver.add_clause_dimacs(&[-2, -4]);
        // Edge (0,2):
        solver.add_clause_dimacs(&[-1, -5]);
        solver.add_clause_dimacs(&[-2, -6]);
        // Edge (1,2):
        solver.add_clause_dimacs(&[-3, -5]);
        solver.add_clause_dimacs(&[-4, -6]);

        let result = solver.solve();
        assert!(result.is_unsat());
    }
}
